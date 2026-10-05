import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { OPS, FILTERS } from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS } from '../src/renderer/ui/palette-panel.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import { assertLeakFree, assertWrapperBalance, assertNoHeapGrowth } from './_leak-helpers.mjs';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cvtColor === 'function'; }
async function waitForCv() {
  if (isReady(cv)) return;
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}
await waitForCv();

function flattenPaletteOps() {
  const ops = [];
  for (const section of TOOL_SECTIONS) ops.push(...section.ops);
  return ops;
}

// ===================================================================
// ภาพทดสอบสังเคราะห์ที่มีลวดลาย/มุมชัดเจนพอให้ ORB/AKAZE/FAST เจอ keypoint จริง (ไม่ใช่ noise สุ่มล้วน
// เพื่อให้ผล deterministic ข้าม run — สี่เหลี่ยม/วงกลมหลายขนาดกระจายในภาพ ให้มีขอบ/มุมจำนวนมาก)
// ===================================================================
function texturedFeatureImage(w, h) {
  const src = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  let seed = 0;
  for (let y = 20; y < h - 20; y += 25) {
    for (let x = 20; x < w - 20; x += 25) {
      seed++;
      const size = 8 + (seed % 5) * 2;
      if (seed % 2 === 0) {
        cv.rectangle(src, new cv.Point(x, y), new cv.Point(x + size, y + size), new cv.Scalar(255), -1);
      } else {
        cv.circle(src, new cv.Point(x + size / 2, y + size / 2), size / 2, new cv.Scalar(255), -1);
      }
    }
  }
  return src;
}

// ctx.info มี 2 รูปแบบ: "พบจุดเด่น N จุด (X)" (ไม่ตัด) หรือ "พบจุดเด่น TOTAL จุด — แสดง N จุดที่คะแนน
// ดีที่สุด (X)" (ตัดแล้ว) — ต้องอ่านตัวเลข "ที่วาดจริง" (หลัง "แสดง" ถ้ามี ไม่ใช่ TOTAL ก่อนตัด)
function reportedDrawnCount(info) {
  const truncated = info.match(/แสดง (\d+) จุดที่คะแนนดีที่สุด/);
  if (truncated) return Number(truncated[1]);
  const plain = info.match(/พบจุดเด่น (\d+) จุด/);
  return Number(plain[1]);
}

function countNonBackground(mat, bgValue) {
  let n = 0;
  for (let i = 0; i < mat.data.length; i++) if (mat.data[i] !== bgValue) n++;
  return n;
}

// ===================================================================
// Python execution harness (แบบเดียวกับ tests/groupA-ops.test.mjs — raw byte dump, ไม่ใช้ PNG)
// ===================================================================
// เก็บกวาดโฟลเดอร์ 'keypoints-parity-*' เก่าที่ค้างจาก run ก่อน ๆ (เฉพาะ prefix นี้เท่านั้น — ไม่แตะ
// 'groupA-parity-*' ของไฟล์เทสต์อื่น) แล้วลบโฟลเดอร์ของ run นี้เองตอนจบด้วย (mkdtempSync ไม่เคยถูกลบมาก่อน
// เลยสะสมค้างใน os.tmpdir() นับสิบโฟลเดอร์)
for (const name of fs.readdirSync(os.tmpdir())) {
  if (name.startsWith('keypoints-parity-')) {
    try { fs.rmSync(path.join(os.tmpdir(), name), { recursive: true, force: true }); } catch { /* best-effort */ }
  }
}
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'keypoints-parity-'));
process.on('exit', () => {
  try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ }
});

function matToRaw(mat) {
  return { data: Buffer.from(mat.data), width: mat.cols, height: mat.rows, channels: mat.channels() };
}

function grayMatToColorRaw(grayMat) {
  const w = grayMat.cols, h = grayMat.rows;
  const color = new cv.Mat(h, w, cv.CV_8UC3);
  for (let i = 0; i < w * h; i++) {
    const v = grayMat.data[i];
    color.data[i * 3] = v; color.data[i * 3 + 1] = v; color.data[i * 3 + 2] = v;
  }
  const raw = matToRaw(color);
  color.delete();
  return raw;
}

function runGeneratedPython(pipeline, inputRaw) {
  const code = buildPythonCode(pipeline);
  const shape = inputRaw.channels === 1
    ? `(${inputRaw.height}, ${inputRaw.width})`
    : `(${inputRaw.height}, ${inputRaw.width}, ${inputRaw.channels})`;
  const loaderLine = 'img = cv2.imread(path)  # อ่านภาพ (BGR)';
  assert.ok(code.includes(loaderLine), 'generated code must contain the expected cv2.imread line to patch');
  let patched = code.replace(loaderLine, `img = np.fromfile(path, dtype=np.uint8).reshape(${shape})`);
  const mainIdx = patched.indexOf('if __name__ == "__main__":');
  assert.ok(mainIdx > 0, 'generated code must contain a __main__ guard to strip (would hang on cv2.imshow/waitKey)');
  patched = patched.slice(0, mainIdx);
  patched += `
import sys
_result = process(sys.argv[1])
_result = np.ascontiguousarray(_result)
_result.astype(np.uint8).tofile(sys.argv[2])
_ch = _result.shape[2] if _result.ndim == 3 else 1
with open(sys.argv[3], 'w') as _f:
    _f.write(f"{_result.shape[0]},{_result.shape[1]},{_ch}")
`;
  const id = Math.random().toString(36).slice(2);
  const scriptPath = path.join(TMP_DIR, `script_${id}.py`);
  const inPath = path.join(TMP_DIR, `in_${id}.bin`);
  const outPath = path.join(TMP_DIR, `out_${id}.bin`);
  const shapePath = path.join(TMP_DIR, `shape_${id}.txt`);
  fs.writeFileSync(scriptPath, patched, 'utf8');
  fs.writeFileSync(inPath, inputRaw.data);
  execFileSync('python', [scriptPath, inPath, outPath, shapePath], { encoding: 'utf8' });
  const [h, w, c] = fs.readFileSync(shapePath, 'utf8').trim().split(',').map(Number);
  const outData = fs.readFileSync(outPath);
  return { data: outData, width: w, height: h, channels: c, code };
}

// เรียก detector ตรง ๆ (ไม่ผ่าน OPS.apply) เพื่อดึง keypoint จริงมาคำนวณ "ความคาดหวัง" อิสระในเทสต์
// (เช่น top-N by response) — ไม่พึ่งพา OPS.apply ที่กำลังถูกทดสอบอยู่แม้แต่บรรทัดเดียว
function detectRaw(cvNs, mat, DetectorCtor, ctorArgs) {
  const detector = ctorArgs === undefined ? new DetectorCtor() : new DetectorCtor(ctorArgs);
  const kpv = new cvNs.KeyPointVector();
  const mask = new cvNs.Mat();
  detector.detect(mat, kpv, mask);
  const out = [];
  for (let i = 0; i < kpv.size(); i++) {
    const kp = kpv.get(i);
    out.push({ x: kp.pt.x, y: kp.pt.y, size: kp.size, angle: kp.angle, response: kp.response });
  }
  detector.delete();
  kpv.delete();
  mask.delete();
  return out;
}

// ===================================================================
// Total-order comparator (เขียนแยกอิสระจาก rankKeypoints() ใน registry.js โดยตั้งใจ — เทสต์ต้องไม่พึ่ง
// การ implement ของโค้ด production แม้แต่บรรทัดเดียว มิฉะนั้น mutant ที่แก้ comparator ผิดจะรอดเทสต์ไปด้วย
// เพราะเทสต์ก็อป logic เดียวกันมาเปรียบเทียบกับตัวเอง) — ต้องตรงกับ comparator ใน registry.js/codegen.js
// เป๊ะทุก key/ทุกทิศทาง: response DESC, y ASC, x ASC, size DESC, angle ASC, octave ASC
// ===================================================================
function totalOrderCompare(a, b) {
  if (b.response !== a.response) return b.response - a.response;
  if (a.y !== b.y) return a.y - b.y;
  if (a.x !== b.x) return a.x - b.x;
  if (b.size !== a.size) return b.size - a.size;
  if (a.angle !== b.angle) return a.angle - b.angle;
  return a.octave - b.octave;
}

// คำนวณ "ชุดที่ควรถูกวาด" อย่างอิสระ (ไม่เรียก OPS.apply เลย) — รัน detector เองบน analysis image
// เดียวกับที่ op ใช้จริงภายใน (ย่อไม่เกิน 480px ด้วย INTER_AREA แบบเดียวกัน) แล้วจัดอันดับด้วย
// totalOrderCompare ข้างบน ตัดเอา cap ตัวแรก แล้วคูณ invScale กลับ — ctorArg ต้องตรงกับค่าที่ op ส่งเข้า
// detector constructor จริง (สำคัญมากสำหรับ ORB: nfeatures ไม่ใช่แค่ post-hoc cap แต่กระทบผลลัพธ์ภายใน
// อัลกอริทึมเองด้วย ยืนยันเชิงประจักษ์แล้วว่าถ้า ctorArg ไม่ตรงกับ maxFeatures ที่ทดสอบ ผลจะไม่ตรงกับที่ op
// วาดจริงเลย แม้ total order จะถูกต้องก็ตาม)
function expectedKeptForOp(cvNs, src, DetectorCtor, ctorArg, cap) {
  const maxAnalysisSide = 480;
  const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
  const analysis = new cvNs.Mat();
  if (scale < 1) {
    const w = Math.max(1, Math.round(src.cols * scale));
    const h = Math.max(1, Math.round(src.rows * scale));
    cvNs.resize(src, analysis, new cvNs.Size(w, h), 0, 0, cvNs.INTER_AREA);
  } else {
    src.copyTo(analysis);
  }
  const detector = ctorArg === undefined ? new DetectorCtor() : new DetectorCtor(ctorArg);
  const kpv = new cvNs.KeyPointVector();
  const mask = new cvNs.Mat();
  detector.detect(analysis, kpv, mask);
  const all = [];
  for (let i = 0; i < kpv.size(); i++) {
    const kp = kpv.get(i);
    all.push({ x: kp.pt.x, y: kp.pt.y, size: kp.size, angle: kp.angle, response: kp.response, octave: kp.octave });
  }
  detector.delete();
  kpv.delete();
  mask.delete();
  analysis.delete();
  all.sort(totalOrderCompare);
  const invScale = 1 / scale;
  const kept = all.slice(0, cap).map((k) => ({ x: k.x * invScale, y: k.y * invScale, angle: k.angle }));
  return { total: all.length, kept };
}

// ===================================================================
// Proxy spy ครอบ cv.circle/cv.line — จับ argument จริงที่ op เรียกวาด (ไม่ใช่ค่าที่ detectRaw คำนวณเอง
// แบบที่ keypoints-9/10 เดิมทำ ซึ่ง qa ชี้ว่า "ไม่ได้ตรวจสิ่งที่ op วาดจริง") ยืนยันแล้วว่า `new proxy.Mat()`/
// `new proxy.ORB(...)` ฯลฯ ยังทำงานได้ปกติผ่าน Proxy นี้ (get trap คืน target[prop] ตรง ๆ ไม่ผูก `this`)
// ===================================================================
function makeSpyCv(baseCv) {
  const calls = { circle: [], line: [] };
  const proxy = new Proxy(baseCv, {
    get(target, prop) {
      if (prop === 'circle') {
        return (img, center, radius, color, thickness) => {
          calls.circle.push({ x: center.x, y: center.y, radius });
          return target.circle(img, center, radius, color, thickness);
        };
      }
      if (prop === 'line') {
        return (img, p1, p2, color, thickness) => {
          calls.line.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y });
          return target.line(img, p1, p2, color, thickness);
        };
      }
      return target[prop];
    },
  });
  return { cv: proxy, calls };
}

// cv.circle ที่ throw เสมอ — ใช้ทดสอบว่า op เก็บกวาด Mat/detector/KeyPointVector ครบแม้ throw กลางลูปวาด
// (mid-loop JS-side failure ไม่ใช่แค่ C++ throw) ยืนยันแล้วว่า resize/cvtColor/detect/line ฯลฯ ที่เหลือยัง
// ทำงานปกติผ่าน Proxy นี้ (แค่ circle ตัวเดียวที่ถูก override)
function makeThrowingCircleCv(baseCv) {
  return new Proxy(baseCv, {
    get(target, prop) {
      if (prop === 'circle') return () => { throw new Error('injected test failure (circle)'); };
      return target[prop];
    },
  });
}

// ===================================================================
// malloc-fingerprint leak probe (เทคนิคเดียวกับ tests/groupA-ops.test.mjs's groupA-0 — วางไว้เป็นเทสต์แรก
// สุดในไฟล์โดยตั้งใจ เพื่อรันบน heap ที่ยังสะอาดที่สุด ดู comment เต็มของเหตุผลใน groupA-ops.test.mjs)
// ===================================================================
const FP_SIZES = [8, 16, 24, 40, 56, 72, 104, 136, 200, 264, 520, 1032, 2056, 4104, 8200, 16392, 65544, 262152, 1048584, 4 << 20, 48 << 20];
function fingerprint() {
  const ptrs = [];
  for (const s of FP_SIZES) for (let r = 0; r < 4; r++) ptrs.push(cv._malloc(s));
  ptrs.sort((a, b) => a - b);
  const sig = ptrs.join(',');
  for (let i = ptrs.length - 1; i >= 0; i--) cv._free(ptrs[i]);
  return sig + '|' + cv.HEAP8.buffer.byteLength;
}

// ตาราง path ของ keypoints-0 — ใช้ร่วมกันระหว่าง keypoints-0 (fingerprint, process สะอาด) และ keypoints-0w (wrapper-balance)
// ให้ทั้งสองด่านครอบ path ชุดเดียวกันเป๊ะ (P-14)
function keypoints0Fixtures() {
  const src = texturedFeatureImage(160, 140);
  const flat = new cv.Mat(160, 140, cv.CV_8UC1, new cv.Scalar(128));
  const tiny = new cv.Mat(1, 1, cv.CV_8UC1, new cv.Scalar(50));
  return { src, flat, tiny, dispose() { src.delete(); flat.delete(); tiny.delete(); } };
}
function keypoints0Paths({ src, flat, tiny }) {
  return {
    'orb (untruncated)': (c) => { const o = OPS.orb_keypoints.apply(c, src, { maxFeatures: 500 }, {}); o.delete(); },
    'orb (truncated)': (c) => { const o = OPS.orb_keypoints.apply(c, src, { maxFeatures: 5 }, {}); o.delete(); },
    'orb (zero-kp)': (c) => { const o = OPS.orb_keypoints.apply(c, flat, {}, {}); o.delete(); },
    'orb (tiny image <3px)': (c) => { const o = OPS.orb_keypoints.apply(c, tiny, {}, {}); o.delete(); },
    'orb (JS-exception mid-draw)': (c) => { try { const o = OPS.orb_keypoints.apply(makeThrowingCircleCv(c), src, {}, {}); o.delete(); } catch { /* expected */ } },
    'akaze (untruncated)': (c) => { const o = OPS.akaze_keypoints.apply(c, src, { maxPoints: 500 }, {}); o.delete(); },
    'akaze (truncated)': (c) => { const o = OPS.akaze_keypoints.apply(c, src, { maxPoints: 5 }, {}); o.delete(); },
    'akaze (zero-kp)': (c) => { const o = OPS.akaze_keypoints.apply(c, flat, {}, {}); o.delete(); },
    'akaze (tiny image <3px)': (c) => { const o = OPS.akaze_keypoints.apply(c, tiny, {}, {}); o.delete(); },
    'akaze (JS-exception mid-draw)': (c) => { try { const o = OPS.akaze_keypoints.apply(makeThrowingCircleCv(c), src, {}, {}); o.delete(); } catch { /* expected */ } },
    'fast (untruncated)': (c) => { const o = OPS.fast_keypoints.apply(c, src, { threshold: 5, maxPoints: 2000 }, {}); o.delete(); },
    'fast (truncated)': (c) => { const o = OPS.fast_keypoints.apply(c, src, { threshold: 5, maxPoints: 5 }, {}); o.delete(); },
    'fast (zero-kp)': (c) => { const o = OPS.fast_keypoints.apply(c, flat, {}, {}); o.delete(); },
    'fast (tiny image <3px)': (c) => { const o = OPS.fast_keypoints.apply(c, tiny, {}, {}); o.delete(); },
    'fast (JS-exception mid-draw)': (c) => { try { const o = OPS.fast_keypoints.apply(makeThrowingCircleCv(c), src, { threshold: 5 }, {}); o.delete(); } catch { /* expected */ } },
  };
}

test('keypoints-0 (malloc-fingerprint). orb/akaze/fast_keypoints leave the allocator fingerprint unchanged across untruncated, truncated, zero-keypoint, tiny-image (<3px), and JS-exception (mid-draw throw) paths', () => {
  const fx = keypoints0Fixtures();
  const runOnce = Object.fromEntries(Object.entries(keypoints0Paths(fx)).map(([k, f]) => [k, () => f(cv)]));

  for (const [label, run] of Object.entries(runOnce)) {
    const warmup = 40;
    for (let i = 0; i < warmup; i++) run();
    const fpA = fingerprint();
    const iters = 150;
    for (let i = 0; i < iters; i++) run();
    const fpB = fingerprint();
    assert.equal(fpB, fpA, `${label}: malloc fingerprint changed after ${iters} more calls -> leak`);
  }
  fx.dispose();
});

// keypoints-0w — ด่านหลัก (wrapper-balance) ของ path ชุดเดียวกับ keypoints-0 (รวม JS-exception กลางลูปวาด ที่ throwingCv
// ถูกสร้างจาก tracker.cv จึงนับ detector/KeyPointVector/Mat ที่ op สร้างก่อน throw ด้วย) — P-14
test('keypoints-0w (wrapper-balance companion of keypoints-0). every keypoints-0 path deletes every embind wrapper it creates (Mats, KeyPointVector, ORB/AKAZE/FastFeatureDetector), including the tiny-image early return and the mid-draw JS exception', () => {
  const fx = keypoints0Fixtures();
  try {
    for (const [label, run] of Object.entries(keypoints0Paths(fx))) assertWrapperBalance(cv, label, run, { minCreated: 1 });
  } finally { fx.dispose(); }
});

// ===================================================================
// 1. ORB basic
// ===================================================================

test('keypoints-1. ORB basic: finds keypoints on a textured image, reports count, draws a green circle near a real keypoint', () => {
  const src = texturedFeatureImage(240, 200);
  const ctx = {};
  const out = OPS.orb_keypoints.apply(cv, src, {}, ctx);
  assert.equal(out.cols, 240);
  assert.equal(out.rows, 200);
  assert.match(ctx.info, /พบจุดเด่น \d+ จุด \(ORB\)/);
  // ต้องมีพิกเซลสีเขียว (0,255,0) อยู่จริงในผลลัพธ์ (วงกลม/เส้นทิศทางที่วาด)
  let foundGreen = false;
  for (let i = 0; i < out.data.length; i += 3) {
    if (out.data[i] === 0 && out.data[i + 1] === 255 && out.data[i + 2] === 0) { foundGreen = true; break; }
  }
  assert.ok(foundGreen, 'expected at least one drawn green (0,255,0) pixel in the ORB output');
  out.delete();
  src.delete();
});

// ===================================================================
// 2. ORB maxFeatures cap
// ===================================================================

// พบเชิงประจักษ์ (ยืนยันซ้ำในทั้ง opencv.js และ cv2 5.0.0 จริง — ดูรายงานถึง coordinator) ว่า ORB's
// native `nfeatures` constructor arg ไม่ใช่ hard cap จริงเสมอไป (ต่างจากที่ spec ระบุไว้ ซึ่งทดสอบไว้แค่
// บนภาพ noise สุ่มล้วน) — บนภาพลวดลาย/ขอบชัดเจนแบบนี้ maxFeatures=20 อาจได้ raw detector output ถึง 74
// จุด จึงต้องมี response-based truncation ซ้ำอีกชั้นเหมือน AKAZE/FAST การันตีว่า "จำนวนจุดสูงสุด" ที่โชว์
// ใน UI เป็นจริงเสมอ ไม่ใช่แค่ raw detector count
test('keypoints-2. ORB maxFeatures cap: the number actually drawn/reported never exceeds maxFeatures, even when the native detector itself returns far more', () => {
  const src = texturedFeatureImage(300, 260);
  const ctx = {};
  const out = OPS.orb_keypoints.apply(cv, src, { maxFeatures: 20 }, ctx);
  const count = reportedDrawnCount(ctx.info);
  assert.ok(count <= 20, `expected drawn/reported count <=20 with maxFeatures=20, got ${count} (info: "${ctx.info}")`);
  assert.ok(count > 0, 'expected at least some keypoints on a densely-textured image');
  // ต้องรายงานด้วยว่ามี "ของจริง" มากกว่านี้ (raw detector output > 20) — ไม่ใช่แค่บังเอิญเจอ <=20 พอดี
  assert.match(ctx.info, /แสดง 20 จุดที่คะแนนดีที่สุด \(ORB\)/, `expected truncation message, got: "${ctx.info}"`);
  out.delete();
  src.delete();
});

// ===================================================================
// 3. AKAZE basic
// ===================================================================

test('keypoints-3. AKAZE basic: finds keypoints, draws circles, reports count', () => {
  const src = texturedFeatureImage(240, 200);
  const ctx = {};
  const out = OPS.akaze_keypoints.apply(cv, src, {}, ctx);
  assert.equal(out.cols, 240);
  assert.equal(out.rows, 200);
  assert.match(ctx.info, /พบจุดเด่น \d+ จุด.*\(AKAZE\)/);
  let foundColor = false;
  for (let i = 0; i < out.data.length; i += 3) {
    if (out.data[i] === 0 && out.data[i + 1] === 200 && out.data[i + 2] === 255) { foundColor = true; break; }
  }
  assert.ok(foundColor, 'expected at least one drawn AKAZE-colored (0,200,255) pixel');
  out.delete();
  src.delete();
});

// ===================================================================
// 4. AKAZE maxPoints truncation + response ranking
// ===================================================================

test('keypoints-4. AKAZE maxPoints truncation: keeps exactly the top-5 by response, drawn circle centers match', () => {
  const src = texturedFeatureImage(320, 260);
  // ตรวจสอบก่อนว่ามี candidate มากกว่า 5 จุดจริง ๆ (ไม่งั้นเทสต์นี้ไม่มีความหมาย)
  // ใช้ analysis image ขนาดเดียวกับที่ op ใช้จริงภายใน (<=480px จึงไม่ย่อ, scale=1) เพื่อให้พิกัดตรงกันเป๊ะ
  const allKp = detectRaw(cv, src, cv.AKAZE);
  assert.ok(allKp.length > 5, `test setup: expected >5 AKAZE candidates on the textured image, got ${allKp.length}`);
  const expectedTop5 = [...allKp].sort((a, b) => b.response - a.response).slice(0, 5);

  const ctx = {};
  const out = OPS.akaze_keypoints.apply(cv, src, { maxPoints: 5 }, ctx);
  assert.match(ctx.info, /แสดง 5 จุดที่คะแนนดีที่สุด \(AKAZE\)/);

  // นับ "จุดศูนย์กลางวงกลม" ที่วาดจริงบน output โดยเช็คว่าตำแหน่งที่คาดไว้ (จาก expectedTop5) มีสี AKAZE
  // อยู่ใกล้ ๆ จริง (รัศมีเล็ก ๆ กันความคลาดเคลื่อนจาก rounding ของ Math.round(r))
  for (const kp of expectedTop5) {
    const x = Math.round(kp.x), y = Math.round(kp.y);
    let nearby = false;
    for (let dy = -2; dy <= 2 && !nearby; dy++) {
      for (let dx = -2; dx <= 2 && !nearby; dx++) {
        const px = x + dx, py = y + dy;
        if (px < 0 || py < 0 || px >= out.cols || py >= out.rows) continue;
        const idx = (py * out.cols + px) * 3;
        if (out.data[idx] === 0 && out.data[idx + 1] === 200 && out.data[idx + 2] === 255) nearby = true;
      }
    }
    assert.ok(nearby, `expected a drawn AKAZE circle near top-response keypoint at (${x},${y})`);
  }
  out.delete();
  src.delete();
});

// ===================================================================
// 5. FAST basic + threshold effect
// ===================================================================

test('keypoints-5. FAST basic + threshold effect: higher threshold yields fewer (or equal) keypoints', () => {
  const src = texturedFeatureImage(240, 200);
  const ctxLow = {}, ctxHigh = {};
  const outLow = OPS.fast_keypoints.apply(cv, src, { threshold: 5 }, ctxLow);
  const outHigh = OPS.fast_keypoints.apply(cv, src, { threshold: 50 }, ctxHigh);
  const nLow = reportedDrawnCount(ctxLow.info);
  const nHigh = reportedDrawnCount(ctxHigh.info);
  assert.ok(nLow > 0, 'expected some keypoints at low threshold');
  assert.ok(nHigh <= nLow, `expected threshold=50 (${nHigh}) <= threshold=5 (${nLow})`);
  outLow.delete();
  outHigh.delete();
  src.delete();
});

// ===================================================================
// 6. FAST never draws a direction line
// ===================================================================

test('keypoints-6. FAST keypoints always report angle=-1 (locks in the "never draw a direction line" assumption)', () => {
  const src = texturedFeatureImage(240, 200);
  const kps = detectRaw(cv, src, cv.FastFeatureDetector, 10);
  assert.ok(kps.length > 0, 'test setup: expected some FAST keypoints');
  for (const kp of kps) assert.equal(kp.angle, -1, 'every FAST keypoint must report angle=-1');
  src.delete();
});

// ===================================================================
// 7. Zero-keypoint guard
// ===================================================================

test('keypoints-7. Zero-keypoint guard: a flat textureless image is handled gracefully by all 3 detectors', () => {
  const flat = new cv.Mat(120, 100, cv.CV_8UC1, new cv.Scalar(128));
  for (const [opId, label] of [['orb_keypoints', 'ORB'], ['akaze_keypoints', 'AKAZE'], ['fast_keypoints', 'FAST']]) {
    const ctx = {};
    let out;
    assert.doesNotThrow(() => { out = OPS[opId].apply(cv, flat, {}, ctx); }, `${label}: must not throw on a flat image`);
    assert.equal(out.cols, 100, `${label}: output width must be unchanged`);
    assert.equal(out.rows, 120, `${label}: output height must be unchanged`);
    assert.match(ctx.info, /ไม่พบจุดเด่น/, `${label}: ctx.info must report "no keypoints found"`);
    out.delete();
  }
  flat.delete();
});

// ===================================================================
// 8. AKAZE construction guard
// ===================================================================

test('keypoints-8. AKAZE construction guard: passing any constructor args throws (locks in the confirmed build limitation)', () => {
  assert.throws(() => new cv.AKAZE(5, 0, 3, 0.01), /unbound types|AKAZE/i);
  // no-arg construction must still work (regression guard the other direction)
  assert.doesNotThrow(() => { const a = new cv.AKAZE(); a.delete(); });
});

// ===================================================================
// 9-10. Codegen + Python execution — ORB / FAST parity
// ===================================================================

// ภาพทดสอบ parity ต้อง >480px ด้านยาว เพื่อ exercise เส้นทาง downscale-then-rescale ของฝั่ง JS
// (Python รันเต็มความละเอียดตามที่ตั้งใจไว้ใน §3.8 ของสเปก — ไม่ใช่บั๊ก)
function parityTestImage(w, h) {
  const src = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  let seed = 0;
  for (let y = 30; y < h - 30; y += 40) {
    for (let x = 30; x < w - 30; x += 40) {
      seed++;
      const size = 14 + (seed % 6) * 3;
      if (seed % 2 === 0) cv.rectangle(src, new cv.Point(x, y), new cv.Point(x + size, y + size), new cv.Scalar(255), -1);
      else cv.circle(src, new cv.Point(x + size / 2, y + size / 2), size / 2, new cv.Scalar(255), -1);
    }
  }
  return src;
}

function parseKeypointsFromPythonInfo(stdoutJson) {
  return JSON.parse(stdoutJson);
}

// รัน Python โดยแทรก json.dump ของ keypoint list (ตำแหน่ง+response) ก่อนบรรทัด "return img" สุดท้ายใน
// process() — codegen จริงเสมอเก็บผลลัพธ์ keypoint detection ไว้ในตัวแปรชื่อ kps เสมอก่อนวาดวงกลม (ดู
// GEN.orb_keypoints/akaze_keypoints/fast_keypoints ใน codegen.js) จึงดึงออกมาเทียบตำแหน่งกับฝั่ง JS ได้
// โดยไม่ต้องแก้โค้ด production เลย
//
// สำคัญ (แก้ตอน qa-Q1 recheck เพิ่มการย่อภาพ mirror ให้ตรงแอป): ตัวแปร `kps` ที่ codegen เก็บไว้ตอนนี้เป็น
// พิกัดใน "พื้นที่ analysis" (ก่อนย่อภาพกลับ) ไม่ใช่พิกัดในภาพเต็มความละเอียดที่ใช้วาดจริง (การคูณ inv_scale
// เกิดขึ้นเป็น local variable x,y ภายใน for-loop ท่อนวาด ไม่ได้เขียนกลับเข้า k.pt) — ต้องคูณ `k.pt[0]*inv_scale`
// ตรงจุด dump เอง (ตัวแปร `inv_scale` มีอยู่แล้วในสโคปเดียวกัน ประกาศไว้ก่อนหน้า return เสมอในทุก op ที่ใช้
// helper นี้) ไม่เช่นนั้นจะเทียบตำแหน่งผิดพื้นที่พิกัดกับฝั่ง JS ทุกครั้งที่ภาพเกิน 480px (ยืนยันแล้วว่าไม่แก้
// จุดนี้จะเห็น top-10 overlap เป็น 0/10 ทั้งที่ฝั่งวาดจริงตรงกันเป๊ะ — เป็นบั๊กของ test harness เอง ไม่ใช่ของโค้ด)
function runGeneratedPythonKeypoints(pipeline, inputRaw) {
  const code = buildPythonCode(pipeline);
  const shape = `(${inputRaw.height}, ${inputRaw.width}, ${inputRaw.channels})`;
  const loaderLine = 'img = cv2.imread(path)  # อ่านภาพ (BGR)';
  assert.ok(code.includes(loaderLine), 'generated code must contain the expected cv2.imread line to patch');
  let patched = code.replace(loaderLine, `img = np.fromfile(path, dtype=np.uint8).reshape(${shape})`);
  const returnIdx = patched.lastIndexOf('    return img');
  assert.ok(returnIdx > 0, 'generated code must end with "    return img" inside process() to patch for keypoint dump');
  patched = patched.slice(0, returnIdx)
    + `    with open(sys.argv[2], 'w') as _f:\n        json.dump([[float(k.pt[0]*inv_scale), float(k.pt[1]*inv_scale), float(k.response)] for k in kps], _f)\n    return img\n`
    + patched.slice(returnIdx + '    return img'.length);
  const mainIdx = patched.indexOf('if __name__ == "__main__":');
  assert.ok(mainIdx > 0, 'generated code must contain a __main__ guard to strip');
  patched = patched.slice(0, mainIdx);
  patched += `
import sys, json
process(sys.argv[1])
`;
  const id = Math.random().toString(36).slice(2);
  const scriptPath = path.join(TMP_DIR, `kpscript_${id}.py`);
  const inPath = path.join(TMP_DIR, `kpin_${id}.bin`);
  const kpPath = path.join(TMP_DIR, `kpout_${id}.json`);
  fs.writeFileSync(scriptPath, patched, 'utf8');
  fs.writeFileSync(inPath, inputRaw.data);
  execFileSync('python', [scriptPath, inPath, kpPath], { encoding: 'utf8' });
  return JSON.parse(fs.readFileSync(kpPath, 'utf8')).map(([x, y, response]) => ({ x, y, response }));
}

function topNOverlap(jsKps, pyKps, n, pixelTol) {
  const jsTop = [...jsKps].sort((a, b) => b.response - a.response).slice(0, n);
  const pyTop = [...pyKps].sort((a, b) => b.response - a.response).slice(0, n);
  let matched = 0;
  for (const j of jsTop) {
    if (pyTop.some((py) => Math.abs(py.x - j.x) <= pixelTol && Math.abs(py.y - j.y) <= pixelTol)) matched++;
  }
  return { matched, total: jsTop.length };
}

// ---------------------------------------------------------------------------------------------
// อัปเดต (qa-Q1, S4 recheck): เดิม comment ตรงนี้อธิบายว่า >480px "ไม่ควรตรงกัน" เพราะ codegen.js
// ตั้งใจรัน Python เต็มความละเอียดโดยไม่ย่อภาพ (asymmetry ที่ตั้งใจ) — **การตัดสินใจนั้นถูกกลับแล้ว**:
// พบว่าขัดกับ fft_magnitude/fft_filter ที่ mirror การย่อภาพไว้อยู่แล้ว และภาพกล้องทั่วไป (640x480 ก็เกิน
// 480px ด้านยาว) ทำให้แอปสดกับสคริปต์ export เห็นจุดเด่นคนละชุดกันแทบทุกครั้ง (วัดได้แค่ 3/10 ตรงกัน) —
// ไม่ใช่กรณีขอบ ๆ แต่เป็นกรณีปกติ จึงแก้ codegen.js ให้ mirror การย่อภาพ ≤480px แบบเดียวกับ registry.js
// เป๊ะ (harris_corners/shi_tomasi_corners/orb_keypoints/akaze_keypoints/fast_keypoints ทั้งหมด) ทำให้
// ตอนนี้ parity ที่ >480px **ควรตรงกันเป๊ะเหมือน <=480px** อีกครั้ง (ยืนยันแล้วด้วยสคริปต์ทดสอบแยก: ภาพ
// 600x500 ORB/FAST ได้ top-10 overlap 10/10 ทั้งคู่ หลังแก้) — เทสต์ keypoints-9b/10b ด้านล่างจึงเปลี่ยนจาก
// "แค่ order-of-magnitude" เป็น parity เข้มแบบเดียวกับ keypoints-9/10 แล้ว เพื่อให้เป็นเทสต์ที่ fail จริง
// ถ้าใครย้อนกลับไปเอาการย่อภาพออกจาก codegen.js — **ข้อควรระวังสำคัญของ test harness เอง**: ตัวแปร `kps`
// ที่ codegen.js เก็บไว้เป็นพิกัดในพื้นที่ analysis (ก่อนย่อกลับ) การคูณ inv_scale เกิดเป็น local variable
// x,y ภายใน for-loop วาดภาพเท่านั้น ไม่ได้เขียนกลับเข้า k.pt — runGeneratedPythonKeypoints() ด้านบนจึงต้อง
// คูณ inv_scale ตอน dump JSON เองด้วย (แก้ไว้แล้ว) ไม่เช่นนั้นจะเทียบพิกัดผิดพื้นที่กับฝั่ง JS ทุกครั้งที่ภาพ
// เกิน 480px (พิสูจน์แล้วว่าถ้าไม่แก้ตรงนี้ จะเห็น top-10 overlap เป็น 0/10 ทั้งที่ภาพที่วาดจริงตรงกันเป๊ะ)
// ---------------------------------------------------------------------------------------------

test('keypoints-9. Same-resolution (<=480px, no JS downscale) JS-vs-Python ORB parity: exact count and exact top-10 position match', () => {
  const src = parityTestImage(400, 350);
  const raw = matToRaw((() => { const c = new cv.Mat(); cv.cvtColor(src, c, cv.COLOR_GRAY2RGB); return c; })());
  const ctx = {};
  const jsOut = OPS.orb_keypoints.apply(cv, src, {}, ctx);
  const jsCount = reportedDrawnCount(ctx.info);

  const pipeline = [{ id: 'orb_keypoints', params: {} }];
  const pyKps = runGeneratedPythonKeypoints(pipeline, raw);
  const pyCount = pyKps.length;

  assert.ok(pyCount > 0, 'expected Python ORB to find some keypoints');
  // <=480px แปลว่า scale=1 ทั้งสองฝั่ง (JS ไม่ downscale เลย) — ต้องตรงกันเป๊ะ ไม่ใช่แค่ tolerance กว้าง ๆ
  assert.equal(jsCount, pyCount, `expected exact ORB count match at <=480px (no JS downscale triggered): JS=${jsCount} Python=${pyCount}`);

  const jsKpsRaw = detectRaw(cv, src, cv.ORB, 300);
  const jsKps = jsKpsRaw.map((k) => ({ x: k.x, y: k.y, response: k.response }));
  const { matched, total } = topNOverlap(jsKps, pyKps, 10, 2);
  assert.equal(matched, total, `expected all top-10-by-response ORB keypoints to match within 2px at identical resolution, got ${matched}/${total}`);

  jsOut.delete();
  src.delete();
});

test('keypoints-10. Same-resolution (<=480px, no JS downscale) JS-vs-Python FAST parity: exact count and exact top-10 position match', () => {
  const src = parityTestImage(400, 350);
  const c = new cv.Mat(); cv.cvtColor(src, c, cv.COLOR_GRAY2RGB);
  const raw = matToRaw(c);
  c.delete();
  const ctx = {};
  const jsOut = OPS.fast_keypoints.apply(cv, src, { threshold: 10 }, ctx);
  const jsCount = reportedDrawnCount(ctx.info);

  const pipeline = [{ id: 'fast_keypoints', params: { threshold: 10 } }];
  const pyKps = runGeneratedPythonKeypoints(pipeline, raw);
  const pyCount = pyKps.length;

  assert.ok(pyCount > 0, 'expected Python FAST to find some keypoints');
  assert.equal(jsCount, pyCount, `expected exact FAST count match at <=480px (no JS downscale triggered): JS=${jsCount} Python=${pyCount}`);

  const jsKpsRaw = detectRaw(cv, src, cv.FastFeatureDetector, 10);
  const jsKps = jsKpsRaw.map((k) => ({ x: k.x, y: k.y, response: k.response }));
  const { matched, total } = topNOverlap(jsKps, pyKps, 10, 2);
  assert.equal(matched, total, `expected all top-10-by-response FAST keypoints to match within 2px at identical resolution, got ${matched}/${total}`);

  jsOut.delete();
  src.delete();
});

test('keypoints-9b/10b. qa-Q1 fix: >480px downscale path (ORB + FAST) now matches the app exactly — Python mirrors the same <=480px analysis downscale, so count AND top-10 position parity hold just like the <=480px case (this is the regression test that fails if the downscale-mirroring is ever reverted)', () => {
  const w = 600, h = 500; // deliberately >480px on the long side to exercise the downscale path on BOTH sides
  const src = parityTestImage(w, h);
  const c = new cv.Mat(); cv.cvtColor(src, c, cv.COLOR_GRAY2RGB);
  const raw = matToRaw(c);
  c.delete();

  const scale = Math.min(1, 480 / Math.max(w, h));
  assert.ok(scale < 1, 'test setup sanity: this image must actually trigger the downscale path');
  const invScale = 1 / scale;
  const analysis = new cv.Mat();
  cv.resize(src, analysis, new cv.Size(Math.round(w * scale), Math.round(h * scale)), 0, 0, cv.INTER_AREA);

  for (const [opId, params, DetectorCtor, ctorArg] of [
    ['orb_keypoints', {}, cv.ORB, 300],
    ['fast_keypoints', { threshold: 10 }, cv.FastFeatureDetector, 10],
  ]) {
    const ctx = {};
    const jsOut = OPS[opId].apply(cv, src, params, ctx);
    const jsCount = reportedDrawnCount(ctx.info);
    const pyKps = runGeneratedPythonKeypoints([{ id: opId, params }], raw); // already rescaled to full-image space by the harness above
    const pyCount = pyKps.length;

    assert.ok(pyCount > 0, `${opId}: expected Python to find some keypoints on a >480px image`);
    assert.ok(jsCount > 0, `${opId}: expected JS to find some keypoints on a >480px image`);
    // now that both sides analyze the identical downscaled image, the count must match exactly again
    assert.equal(jsCount, pyCount, `${opId}: expected exact count match at >480px now that Python mirrors the app's downscale: JS=${jsCount} Python=${pyCount}`);

    const jsKpsRaw = detectRaw(cv, analysis, DetectorCtor, ctorArg)
      .map((k) => ({ x: k.x * invScale, y: k.y * invScale, response: k.response }));
    const { matched, total } = topNOverlap(jsKpsRaw, pyKps, 10, 2);
    assert.equal(matched, total, `${opId}: expected all top-10-by-response keypoints to match within 2px at >480px (mirrored downscale), got ${matched}/${total}`);

    jsOut.delete();
  }
  analysis.delete();
  src.delete();
});

// ===================================================================
// 11. Codegen + Python execution — AKAZE (expect skip on this machine: no opencv-contrib-python)
// ===================================================================

test('keypoints-11. Codegen + Python execution — AKAZE: skip gracefully if opencv-contrib-python is missing (AttributeError on AKAZE_create specifically), fail on any other error', (t) => {
  const src = parityTestImage(300, 260);
  const c = new cv.Mat(); cv.cvtColor(src, c, cv.COLOR_GRAY2RGB);
  const raw = matToRaw(c);
  c.delete();
  const pipeline = [{ id: 'akaze_keypoints', params: {} }];
  try {
    runGeneratedPython(pipeline, raw);
    // ถ้าไม่ throw แปลว่าเครื่องนี้มี opencv-contrib-python ติดตั้งจริง — ถือว่าผ่าน (ไม่ skip)
  } catch (err) {
    const msg = String(err.stderr || err.message || err);
    // เจาะจงเป๊ะที่ attribute 'AKAZE_create' (มี quote ล้อมรอบ) ไม่ใช่แค่ regex คำว่า AKAZE_create เฉย ๆ —
    // ป้องกันไม่ให้ match ผิดกับ error ที่หน้าตาคล้ายกันแต่เป็นคนละปัญหาจริง (เช่น attribute ชื่อ
    // 'AKAZE_createX' ที่ควร fail ไม่ใช่ skip) ต้องเจอทั้ง 'AttributeError' และข้อความ attribute ที่ตรงเป๊ะ
    if (/AttributeError/.test(msg) && /has no attribute 'AKAZE_create'/.test(msg)) {
      t.skip('opencv-contrib-python not installed on this machine (cv2.AKAZE_create missing) — expected, confirmed by CV eng smoke test');
    } else {
      throw err;
    }
  }
  src.delete();
});

// ===================================================================
// 12. Codegen comment presence — AKAZE opencv-contrib-python warning
// ===================================================================

test('keypoints-12. AKAZE generated Python code contains the opencv-contrib-python warning comment', () => {
  const code = buildPythonCode([{ id: 'akaze_keypoints', params: {} }]);
  assert.ok(code.includes('opencv-contrib-python'), 'generated AKAZE code must contain a visible opencv-contrib-python warning comment');
});

// ===================================================================
// 13. Mat-leak sanity (malloc-fingerprint + pointer-drift probes, matching groupA-ops.test.mjs convention)
// ===================================================================

// P-14: เดิมเป็น sentinel 1500x1500x3 byteOffset + HEAP8.buffer.byteLength — ตอนนี้ assertLeakFree (wrapper-balance +
// heap census) ชื่อเทสต์คงเดิม
test('keypoints-13. leak probe: orb_keypoints/akaze_keypoints/fast_keypoints do not leak Mats/KeyPointVector/detector (150-200 iterations each)', () => {
  const src = texturedFeatureImage(160, 140);
  const flat = new cv.Mat(160, 140, cv.CV_8UC1, new cv.Scalar(128));

  const runOnce = {
    'object_detection/orb_keypoints': (c) => { const o = OPS.orb_keypoints.apply(c, src, { maxFeatures: 100 }, {}); o.delete(); },
    'object_detection/akaze_keypoints': (c) => { const o = OPS.akaze_keypoints.apply(c, src, { maxPoints: 100 }, {}); o.delete(); },
    'object_detection/fast_keypoints': (c) => { const o = OPS.fast_keypoints.apply(c, src, { threshold: 10, maxPoints: 100 }, {}); o.delete(); },
    // เส้นทาง zero-keypoint (ต้องไม่ leak เช่นกัน — early return ก่อนถึงส่วนวาด)
    'object_detection/orb_keypoints (zero-kp path)': (c) => { const o = OPS.orb_keypoints.apply(c, flat, {}, {}); o.delete(); },
  };

  for (const [label, run] of Object.entries(runOnce)) {
    assertLeakFree(cv, label, run, { minCreated: 3, iters: 150, warmup: 10 });
  }
  src.delete();
  flat.delete();
});

// keypoints-13b — leak "ฝั่ง native" ของภาพเล็ก (<3px): ORB/AKAZE .detect() throw C++ exception แล้วรั่วข้างใน WASM
// (~60KB/ครั้งสำหรับ ORB, ~3.5KB AKAZE — วัดใน tests/leak-gates.test.mjs gates-9) และ FAST บนภาพ 1xN รั่ว ~60-100B/ครั้ง
// โดยไม่มี wrapper JS ค้างเลย — tracker จึงมองไม่เห็น ต้องใช้ heap census เท่านั้น (นโยบาย P-14 ข้อ 3) guard
// tooSmallForKeypoints ใน registry.js คือสิ่งที่เทสต์นี้คุ้มกัน: ถ้า guard หาย census ต้อง trip
// run กลืน throw เองโดยตั้งใจ (ให้เทสต์นี้วัด "การรั่ว" ไม่ใช่ "การ throw" — การ throw/ข้อความไทยมีเทสต์อื่นคุมอยู่แล้ว)
// จำนวน throw ต่ำ (ไม่เกิน ~70 ครั้งรวม) เพราะ throw สะสมหลายร้อยครั้งทำ heap ของ opencv.js พังได้ (WORKLOG P-14 measure)
test('keypoints-13b. native-leak census: orb/akaze/fast_keypoints on tiny (<3px: 1x50, 50x1) images do not grow the WASM heap (guards the tooSmallForKeypoints skip — a detector throw there leaks inside WASM with no JS wrapper left)', () => {
  const thin = new cv.Mat(1, 50, cv.CV_8UC1, new cv.Scalar(90));
  const tall = new cv.Mat(50, 1, cv.CV_8UC1, new cv.Scalar(90));
  const swallow = (fn) => () => { try { const o = fn(); o.delete(); } catch { /* วัด leak เท่านั้น */ } };
  for (const [name, img] of [['1x50', thin], ['50x1', tall]]) {
    assertNoHeapGrowth(cv, `orb tiny ${name}`, swallow(() => OPS.orb_keypoints.apply(cv, img, {}, {})), { iters: 10, warmup: 2 });
    assertNoHeapGrowth(cv, `akaze tiny ${name}`, swallow(() => OPS.akaze_keypoints.apply(cv, img, {}, {})), { iters: 10, warmup: 2 });
    assertNoHeapGrowth(cv, `fast tiny ${name}`, swallow(() => OPS.fast_keypoints.apply(cv, img, {}, {})), { iters: 60, warmup: 5, tolerance: 1024 });
  }
  thin.delete();
  tall.delete();
});

// ===================================================================
// 14. Wiring sanity
// ===================================================================

test('keypoints-14. wiring: orb_keypoints/akaze_keypoints/fast_keypoints present in FILTERS.object_detection mode options and in TOOL_SECTIONS', () => {
  const modeParam = FILTERS.object_detection.params.find((p) => p.key === 'mode');
  assert.ok(modeParam, 'object_detection must have a mode select');
  for (const opId of ['orb_keypoints', 'akaze_keypoints', 'fast_keypoints']) {
    assert.ok(modeParam.options.some((o) => o.value === opId), `object_detection's mode select must include ${opId}`);
  }
  const paletteOps = flattenPaletteOps();
  for (const opId of ['orb_keypoints', 'akaze_keypoints', 'fast_keypoints']) {
    assert.ok(paletteOps.includes(opId), `expected ${opId} to be listed in some TOOL_SECTIONS entry`);
  }
});

// ===================================================================
// 15. Downscale-coordinate-rescaling sanity
// ===================================================================

test('keypoints-15. downscale-then-rescale sanity: an isolated high-contrast feature on a >480px image is detected within a few pixels of its known location', () => {
  const w = 700, h = 600;
  const src = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  // checkerboard-corner-style pattern: 4 alternating quadrants around a known center, gives a very
  // strong, unambiguous corner right at the center point
  const cx = 500, cy = 400, half = 40;
  cv.rectangle(src, new cv.Point(cx - half, cy - half), new cv.Point(cx, cy), new cv.Scalar(255), -1);
  cv.rectangle(src, new cv.Point(cx, cy), new cv.Point(cx + half, cy + half), new cv.Scalar(255), -1);
  // some background texture elsewhere so detectors don't only look at our marker
  for (let y = 20; y < h - 20; y += 60) {
    for (let x = 20; x < 400; x += 60) cv.circle(src, new cv.Point(x, y), 10, new cv.Scalar(255), -1);
  }

  const ctx = {};
  const out = OPS.orb_keypoints.apply(cv, src, { maxFeatures: 500 }, ctx);
  let nearby = false;
  const tol = 8;
  for (let dy = -tol; dy <= tol && !nearby; dy++) {
    for (let dx = -tol; dx <= tol && !nearby; dx++) {
      const px = cx + dx, py = cy + dy;
      if (px < 0 || py < 0 || px >= out.cols || py >= out.rows) continue;
      const idx = (py * out.cols + px) * 3;
      if (out.data[idx] === 0 && out.data[idx + 1] === 255 && out.data[idx + 2] === 0) nearby = true;
    }
  }
  assert.ok(nearby, `expected a drawn ORB keypoint circle within ${tol}px of the known checkerboard-corner location (${cx},${cy}) after the downscale-then-rescale round trip`);
  out.delete();
  src.delete();
});

// ===================================================================
// 16. Drawn-output spy: circle-call count/positions and line-call count must match what the op ACTUALLY
// draws (not ctx.info, not a separately-run detectRaw() on unrelated ctorArgs) — closes qa's MAJOR-2(a)/(b):
// mutants like "remove ORB truncation", "sort ascending", "invScale=1", "FAST draws a line" survived the
// old tests because keypoints-2 read ctx.info (computed independently of drawing) and keypoints-9/10 took
// JS positions from detectRaw() with a fixed ctorArg instead of the op's own actual constructor argument.
// ===================================================================

test('keypoints-16. drawn-output spy: circle count == min(total,cap), drawn centers match an independently-computed top-N (same total order incl. tie-break), for ORB/AKAZE/FAST at <=480px and >480px; FAST never draws a line; ORB/AKAZE line count == kept-with-angle!=-1', () => {
  for (const [w, h] of [[300, 260], [600, 500]]) {
    const src = texturedFeatureImage(w, h);
    // ctorArg ต้องตรงกับค่าที่ op ส่งเข้า constructor จริง (ORB: nfeatures=maxFeatures เป๊ะ เพราะมันกระทบ
    // ผลลัพธ์ภายในอัลกอริทึมเอง ไม่ใช่แค่ post-hoc cap — ยืนยันเชิงประจักษ์แล้วว่าถ้าใช้ ctorArg คงที่ต่างจาก
    // maxFeatures ที่ทดสอบ ผลจะไม่ตรงกับที่ op วาดจริงเลย แม้ total order จะถูกต้อง; AKAZE ไม่มี ctor arg เลย;
    // FAST: ctorArg=threshold ซึ่งไม่ผูกกับ maxPoints/cap)
    const cases = [
      { opId: 'orb_keypoints', params: { maxFeatures: 15 }, DetectorCtor: cv.ORB, ctorArg: 15, cap: 15, isFast: false },
      { opId: 'akaze_keypoints', params: { maxPoints: 15 }, DetectorCtor: cv.AKAZE, ctorArg: undefined, cap: 15, isFast: false },
      { opId: 'fast_keypoints', params: { threshold: 5, maxPoints: 15 }, DetectorCtor: cv.FastFeatureDetector, ctorArg: 5, cap: 15, isFast: true },
    ];
    for (const { opId, params, DetectorCtor, ctorArg, cap, isFast } of cases) {
      const { cv: spyCv, calls } = makeSpyCv(cv);
      const ctx = {};
      const out = OPS[opId].apply(spyCv, src, params, ctx);
      const expected = expectedKeptForOp(cv, src, DetectorCtor, ctorArg, cap);
      assert.ok(expected.total > cap, `${opId} @ ${w}x${h}: test setup needs raw total>cap to exercise truncation, got total=${expected.total} cap=${cap}`);

      assert.equal(calls.circle.length, Math.min(expected.total, cap), `${opId} @ ${w}x${h}: circle-call count must equal min(total,cap) -- catches removed/weakened truncation`);

      const gotSet = calls.circle.map((c) => `${c.x.toFixed(3)},${c.y.toFixed(3)}`).sort();
      const wantSet = expected.kept.map((k) => `${k.x.toFixed(3)},${k.y.toFixed(3)}`).sort();
      assert.deepEqual(gotSet, wantSet, `${opId} @ ${w}x${h}: drawn circle centers must equal the independently-computed top-N set -- catches ascending sort / invScale=1 / wrong tie-break`);

      if (isFast) {
        assert.equal(calls.line.length, 0, `${opId} @ ${w}x${h}: FAST must never draw a direction line`);
      } else {
        const expectedLines = expected.kept.filter((k) => k.angle !== -1).length;
        assert.equal(calls.line.length, expectedLines, `${opId} @ ${w}x${h}: line-call count must equal kept keypoints with angle != -1`);
      }
      out.delete();
    }
    src.delete();
  }
});

// ===================================================================
// 17. MAJOR-1 tie-break parity regression: at <=480px (scale=1, no JS downscale) on a tie-heavy image, the
// SET of centers the JS op actually draws (via spy) must equal exactly the kept set the generated Python
// script computes (reusing runGeneratedPythonKeypoints's kps dump, post-sort-and-slice) — proves the shared
// total-order tie-break (rankKeypoints in registry.js / KEYPOINT_SORT_KEY in codegen.js) picks the identical
// subset on both sides even though many keypoints tie on response (this exact image size is qa's own repro:
// "textured 400x350: 373==373, 0 set differences, 122 ordered mismatches without the tie-break fix")
// ===================================================================

function centerSet(points, digits = 3) {
  return points.map((p) => `${p.x.toFixed(digits)},${p.y.toFixed(digits)}`).sort();
}

test('keypoints-17. MAJOR-1 regression: JS-drawn set == Python-kept set exactly at <=480px on a tie-heavy image, for ORB (default 300 and maxFeatures 50) and FAST (threshold 10, maxPoints 300)', () => {
  const src = parityTestImage(400, 350); // <=480px both sides -> scale=1 exactly, JS never downscales here
  const c = new cv.Mat(); cv.cvtColor(src, c, cv.COLOR_GRAY2RGB);
  const raw = matToRaw(c);
  c.delete();

  const cases = [
    { opId: 'orb_keypoints', params: {} },
    { opId: 'orb_keypoints', params: { maxFeatures: 50 } },
    { opId: 'fast_keypoints', params: { threshold: 10, maxPoints: 300 } },
  ];
  for (const { opId, params } of cases) {
    const { cv: spyCv, calls } = makeSpyCv(cv);
    const jsOut = OPS[opId].apply(spyCv, src, params, {});
    const jsSet = centerSet(calls.circle);

    const pyKps = runGeneratedPythonKeypoints([{ id: opId, params }], raw);
    const pySet = centerSet(pyKps);

    assert.deepEqual(jsSet, pySet, `${opId} params=${JSON.stringify(params)}: JS-drawn centers must equal Python-kept centers exactly (tie-break parity) -- catches missing/inconsistent tie-break`);
    jsOut.delete();
  }
  src.delete();
});

// ===================================================================
// 18. Explicit-0 params + shared codegen tie-break sort-key string parity
// ===================================================================

test('keypoints-18. codegen: explicit 0 params clamp identically to JS (ORB maxFeatures:0 -> nfeatures=1, FAST threshold:0 -> threshold=1), and the shared tie-break sort key string is present verbatim in all 3 generated snippets', () => {
  const orbCode = buildPythonCode([{ id: 'orb_keypoints', params: { maxFeatures: 0 } }]);
  assert.match(orbCode, /nfeatures=1\)/, 'ORB maxFeatures:0 must clamp to nfeatures=1 in generated Python (matches JS Math.max(1, ...))');

  const fastCode = buildPythonCode([{ id: 'fast_keypoints', params: { threshold: 0 } }]);
  assert.match(fastCode, /threshold=1\)/, 'FAST threshold:0 must clamp to threshold=1 in generated Python (matches JS Math.max(1, ...))');

  const akazeCode = buildPythonCode([{ id: 'akaze_keypoints', params: {} }]);

  const sortKeySnippet = 'key=lambda k: (-k.response, k.pt[1], k.pt[0], -k.size, k.angle, k.octave)';
  for (const [label, code] of [['ORB', orbCode], ['AKAZE', akazeCode], ['FAST', fastCode]]) {
    assert.ok(code.includes(sortKeySnippet), `${label}: generated code must contain the shared tie-break sort key string verbatim`);
  }
});

// ===================================================================
// m3 (minor): ORB maxFeatures upper-bound clamp [1,1000] — a hand-edited saved project could contain an
// absurd maxFeatures (e.g. >=1e9), which makes native cv.ORB(nfeatures)/cv2.ORB_create(nfeatures=...) throw
// an allocation error. Both JS and the generated Python must clamp the upper bound identically.
// ===================================================================

test('keypoints-m3. ORB maxFeatures upper-bound clamp: an absurd maxFeatures (>=1e9, only reachable via a hand-edited saved project) does not crash the JS op and clamps to nfeatures=1000 in generated Python', () => {
  const src = new cv.Mat.zeros(50, 50, cv.CV_8UC1);
  let out;
  assert.doesNotThrow(() => { out = OPS.orb_keypoints.apply(cv, src, { maxFeatures: 2_000_000_000 }, {}); }, 'ORB with maxFeatures=2e9 must not throw an allocation error (must clamp to 1000 before constructing cv.ORB)');
  out.delete();
  src.delete();

  const code = buildPythonCode([{ id: 'orb_keypoints', params: { maxFeatures: 2_000_000_000 } }]);
  assert.match(code, /nfeatures=1000\)/, 'ORB maxFeatures=2e9 must clamp to nfeatures=1000 in generated Python (matches JS Math.min(1000, ...))');
});

// ===================================================================
// 19. MAJOR-4: tiny images (either side <3px in the analysis image) must not throw, dims unchanged, info
// carries the "too small" message — for all 3 ops (also exercised inside the keypoints-0 fingerprint leak
// test above, this test focuses on the user-facing dims/message contract)
// ===================================================================

test('keypoints-19. tiny-image guard: 1x1, 2x1, 1x2, 1x2000, 2000x1, 5000x2 never throw for ORB/AKAZE/FAST, output dims unchanged, ctx.info reports the "too small" message', () => {
  const sizes = [[1, 1], [2, 1], [1, 2], [1, 2000], [2000, 1], [5000, 2]];
  for (const [w, h] of sizes) {
    const src = new cv.Mat.zeros(h, w, cv.CV_8UC1);
    for (const opId of ['orb_keypoints', 'akaze_keypoints', 'fast_keypoints']) {
      const ctx = {};
      let out;
      assert.doesNotThrow(() => { out = OPS[opId].apply(cv, src, {}, ctx); }, `${opId} @ ${w}x${h}: must not throw`);
      assert.equal(out.cols, w, `${opId} @ ${w}x${h}: output width must be unchanged`);
      assert.equal(out.rows, h, `${opId} @ ${w}x${h}: output height must be unchanged`);
      assert.match(ctx.info, /ภาพเล็กเกินไปสำหรับหาจุดเด่น/, `${opId} @ ${w}x${h}: ctx.info must report the too-small message`);
      out.delete();
    }
    src.delete();
  }
});

// ===================================================================
// qa-Q1 (S4 recheck): mirror the app's <=480px analysis downscale into the generated Python for
// harris_corners and shi_tomasi_corners too, consistent with orb/akaze/fast_keypoints and with the
// existing fft_magnitude/fft_filter precedent from Tier 1.5. Applies the same >480px pipeline used above.
// ===================================================================

function harrisCornerPositions(pipeline, inputRaw) {
  const code = buildPythonCode(pipeline);
  const shape = `(${inputRaw.height}, ${inputRaw.width}, ${inputRaw.channels})`;
  const loaderLine = 'img = cv2.imread(path)  # อ่านภาพ (BGR)';
  let patched = code.replace(loaderLine, `img = np.fromfile(path, dtype=np.uint8).reshape(${shape})`);
  const returnIdx = patched.lastIndexOf('    return img');
  assert.ok(returnIdx > 0, 'harris_corners codegen must end with "    return img" inside process()');
  patched = patched.slice(0, returnIdx)
    + `    with open(sys.argv[2], 'w') as _f:\n        json.dump([[float(x*inv_scale), float(y*inv_scale)] for x, y in zip(xs, ys)], _f)\n    return img\n`
    + patched.slice(returnIdx + '    return img'.length);
  const mainIdx = patched.indexOf('if __name__ == "__main__":');
  patched = patched.slice(0, mainIdx) + '\nimport sys, json\nprocess(sys.argv[1])\n';
  const id = Math.random().toString(36).slice(2);
  const scriptPath = path.join(TMP_DIR, `hscript_${id}.py`);
  const inPath = path.join(TMP_DIR, `hin_${id}.bin`);
  const outPath = path.join(TMP_DIR, `hout_${id}.json`);
  fs.writeFileSync(scriptPath, patched, 'utf8');
  fs.writeFileSync(inPath, inputRaw.data);
  execFileSync('python', [scriptPath, inPath, outPath], { encoding: 'utf8' });
  return JSON.parse(fs.readFileSync(outPath, 'utf8'));
}

function shiTomasiCornerPositions(pipeline, inputRaw) {
  const code = buildPythonCode(pipeline);
  const shape = `(${inputRaw.height}, ${inputRaw.width}, ${inputRaw.channels})`;
  const loaderLine = 'img = cv2.imread(path)  # อ่านภาพ (BGR)';
  let patched = code.replace(loaderLine, `img = np.fromfile(path, dtype=np.uint8).reshape(${shape})`);
  const returnIdx = patched.lastIndexOf('    return img');
  assert.ok(returnIdx > 0, 'shi_tomasi_corners codegen must end with "    return img" inside process()');
  patched = patched.slice(0, returnIdx)
    + `    _pts = [] if corners is None else [[float(x*inv_scale), float(y*inv_scale)] for x, y in corners.reshape(-1, 2)]\n    with open(sys.argv[2], 'w') as _f:\n        json.dump(_pts, _f)\n    return img\n`
    + patched.slice(returnIdx + '    return img'.length);
  const mainIdx = patched.indexOf('if __name__ == "__main__":');
  patched = patched.slice(0, mainIdx) + '\nimport sys, json\nprocess(sys.argv[1])\n';
  const id = Math.random().toString(36).slice(2);
  const scriptPath = path.join(TMP_DIR, `stscript_${id}.py`);
  const inPath = path.join(TMP_DIR, `stin_${id}.bin`);
  const outPath = path.join(TMP_DIR, `stout_${id}.json`);
  fs.writeFileSync(scriptPath, patched, 'utf8');
  fs.writeFileSync(inPath, inputRaw.data);
  execFileSync('python', [scriptPath, inPath, outPath], { encoding: 'utf8' });
  return JSON.parse(fs.readFileSync(outPath, 'utf8'));
}

test('keypoints-20. qa-Q1: harris_corners and shi_tomasi_corners generated Python now contains the same <=480px downscale-mirroring block as orb/akaze/fast_keypoints (structural regression guard — fails if the downscale mirroring is ever removed from either)', () => {
  for (const [id, params] of [
    ['harris_corners', { blockSize: 2, ksize: 3, k: 0.04, thresh: 1 }],
    ['shi_tomasi_corners', { maxCorners: 100, quality: 1, minDist: 10, blockSize: 3, useHarris: false }],
  ]) {
    const code = buildPythonCode([{ id, params }]);
    assert.match(code, /scale = min\(1\.0, 480 \/ max\(rows, cols\)\)/, `${id}: generated code must mirror the app's 480px analysis downscale`);
    assert.match(code, /int\(cols \* scale \+ 0\.5\)/, `${id}: must use int(x + 0.5), not round(), for the runtime-dependent resize dimensions (Tier 1.5 lesson)`);
  }
});

// ---------------------------------------------------------------------------------------------
// qa delta #17 MAJOR-A: the previous version of this test only compared the corner COUNT with
// maxCorners=100 -- and on this test image BOTH sides hit that cap (uncapped there are ~714 corners
// at 600x500), so the count comparison was vacuous. Confirmed by qa's mutation testing: a mutant that
// sets `scale = 1.0` in shi_tomasi_corners's codegen only (leaving `minDist * scale` untouched)
// survived (only the structural regex test keypoints-20 caught it), and a mutant that drops `* scale`
// from minDist survived too (the whole suite passed).
//
// Fixed by (1) raising maxCorners to 5000 so neither side is ever capped, and (2) comparing actual
// drawn POSITIONS instead of just a count -- JS positions come from a cv Proxy spy on cv.circle (same
// technique as keypoints-16/17, so this tests what the op actually draws, not a separately-computed
// detectRaw() set); Python positions come from shiTomasiCornerPositions()'s dump, which is already
// multiplied by inv_scale.
//
// A byte-exact position-set match is too strict across environments: even with byte-identical analysis
// images (INTER_AREA output is byte-identical between opencv.js and cv2 on this input -- confirmed
// separately by qa), goodFeaturesToTrack in opencv.js 4.10 vs cv2 5.0.0 can disagree on a small number
// of corners at the library level. Measured on this exact fixed code (scratch probe, logged in the
// WORKLOG): with parityTestImage at maxCorners=5000, JS/Python position overlap within 1px is 642/642
// (100%) at 600x500 and 1622/1622 (100%) at 1001x777 -- so a 97% floor leaves comfortable headroom for
// library-version drift on other machines while still failing hard on real bugs. Re-verified both
// required mutations now fail this rewritten test at both sizes (both fail on the count assertion
// alone, before the position check even runs):
//   scale=1.0 only         -> 600x500:  count 642 vs 616 (mismatch), within-1px 356/642 (55%)
//                              1001x777: count 1622 vs 1728 (mismatch), within-1px 118/1622 (7%)
//   minDist without *scale -> 600x500:  count 642 vs 564 (mismatch), within-1px 564/642 (88%)
//                              1001x777: count 1622 vs 650 (mismatch), within-1px 618/1622 (38%)
// ---------------------------------------------------------------------------------------------
const SHI_TOMASI_POSITION_MATCH_FLOOR_PCT = 97;

test('keypoints-21. qa delta #17 MAJOR-A fix: Shi-Tomasi JS-vs-Python parity at >480px compared by actual drawn POSITIONS (via a cv.circle spy, not just a count that maxCorners capping can hide) -- exact corner count plus >=97% of JS-drawn centers within 1px of a Python-drawn center, with maxCorners=5000 so neither side is ever capped', () => {
  for (const [w, h] of [[600, 500], [1001, 777]]) {
    const src = parityTestImage(w, h);
    const c = new cv.Mat(); cv.cvtColor(src, c, cv.COLOR_GRAY2RGB);
    const raw = matToRaw(c);
    c.delete();

    // maxCorners deliberately huge (both sides find far fewer than 5000 real corners on this image) so
    // the comparison is never hidden behind a shared cap -- this is what makes the test non-vacuous
    const params = { maxCorners: 5000, quality: 1, minDist: 10, blockSize: 3, useHarris: false };
    const { cv: spyCv, calls } = makeSpyCv(cv);
    const jsOut = OPS.shi_tomasi_corners.apply(spyCv, src, params, {});
    const jsPts = calls.circle;

    const pyPts = shiTomasiCornerPositions([{ id: 'shi_tomasi_corners', params }], raw);

    assert.ok(jsPts.length > 0, `expected JS to find corners at ${w}x${h}`);
    assert.equal(jsPts.length, pyPts.length, `expected exact corner count match at ${w}x${h} (uncapped, maxCorners=5000): JS=${jsPts.length} Python=${pyPts.length}`);

    let within1 = 0;
    for (const j of jsPts) {
      if (pyPts.some((p) => Math.abs(p[0] - j.x) <= 1 && Math.abs(p[1] - j.y) <= 1)) within1++;
    }
    const pct = (100 * within1) / jsPts.length;
    assert.ok(pct >= SHI_TOMASI_POSITION_MATCH_FLOOR_PCT,
      `expected >=${SHI_TOMASI_POSITION_MATCH_FLOOR_PCT}% of JS-drawn corners to have a Python-drawn corner within 1px at ${w}x${h}, got ${within1}/${jsPts.length} (${pct.toFixed(2)}%)`);

    jsOut.delete();
    src.delete();
  }
});

// qa delta #17 MINOR: the previous version of this test asserted `jsCount > pyCount`, i.e. it asserted
// that the known Harris JS/Python divergence is PRESENT -- so the day someone fixes it (ports the
// dilate/merge logic into the Python codegen), this test starts failing and nudges them to "keep the
// bug" instead of celebrating the fix. Converted to node:test's `todo` form: it asserts the DESIRED
// parity (jsCount === pyCount) so it (a) shows up as `todo` (not a hard failure) in `node --test`
// output right now, keeping the suite green while the divergence is still visible/documented, and
// (b) turns into a real, silently-passing test the moment someone actually fixes harris_corners's
// codegen to match the app's dilate-then-count behavior.
test('keypoints-22. qa-Q1 scope note (backlog): harris_corners count should match between JS and Python at ANY resolution (including <=480px, i.e. independent of the downscale fix) -- currently does NOT, because JS merges adjacent corner pixels via cv.dilate before counting while the Python codegen counts raw thresholded pixels directly. This is a pre-existing, separate divergence that the qa-Q1 downscale-mirroring round intentionally did not fix (out of scope). Marked todo so the suite stays green and the gap stays visible until someone ports the dilate/merge logic into the Python codegen.', { todo: 'known Harris JS/Python corner-count divergence (JS dilates/merges before counting, Python counts raw pixels) — backlog, not part of qa-Q1' }, () => {
  const w = 400, h = 350; // <=480px -- isolates the divergence from the downscale question entirely
  const src = parityTestImage(w, h);
  const c = new cv.Mat(); cv.cvtColor(src, c, cv.COLOR_GRAY2RGB);
  const raw = matToRaw(c);
  c.delete();

  const params = { blockSize: 2, ksize: 3, k: 0.04, thresh: 60 };
  const ctx = {};
  const jsOut = OPS.harris_corners.apply(cv, src, params, ctx);
  const jsCountMatch = /พบมุม (\d+) จุด/.exec(ctx.info);
  assert.ok(jsCountMatch, 'expected a parseable corner count in ctx.info');
  const jsCount = Number(jsCountMatch[1]);

  const pyPts = harrisCornerPositions([{ id: 'harris_corners', params }], raw);
  // Desired end state (currently fails -- that's the point of `todo`): JS and Python should report the
  // same corner count. Today JS counts individual pixels of a dilated/merged mask (systematically much
  // larger); Python counts raw thresholded pixels one-for-one.
  assert.equal(jsCount, pyPts.length, `expected Harris corner count parity (currently a known gap): JS=${jsCount} Python=${pyPts.length}`);

  jsOut.delete();
  src.delete();
});

// ===================================================================
// 23-26. qa delta #17 MAJOR-B: the image the generated Python actually DRAWS was never verified by any
// test before this round -- every parity test above only checked keypoint POSITIONS dumped via JSON
// (runGeneratedPythonKeypoints) or CORNER positions (harris/shi-tomasi dumps), never the pixels of the
// image the script writes out. qa's mutation testing found real survivors this way: drawing without
// inv_scale (ORB, Shi-Tomasi), radius computed by truncating int() instead of mirroring the app's
// unrounded-float-then-round approach (every FAST circle r=3 in the export vs the app's r=4), and
// AKAZE/FAST drawing colors never converted RGB->BGR (unlike the watershed/hough precedent) so the
// exported image showed the wrong color entirely (FAST blue instead of orange, AKAZE yellow instead of
// blue).
//
// Closes the gap with a real pixel-level parity test: run the generated Python, dump the raw output
// image bytes (runGeneratedPython, defined near the top of this file -- already used by keypoints-11),
// and compare against the JS op's actual output Mat pixel-for-pixel. JS output is RGB, Python's is BGR
// -- swap channels when comparing (pixelDiffCount below does jsR/G/B vs pyB/G/R).
// ===================================================================

function pixelDiffCount(jsMat, pyRaw) {
  assert.equal(jsMat.cols, pyRaw.width, 'JS/Python output width must match');
  assert.equal(jsMat.rows, pyRaw.height, 'JS/Python output height must match');
  assert.equal(pyRaw.channels, 3, 'Python output must be 3-channel BGR');
  let diff = 0;
  const n = jsMat.cols * jsMat.rows;
  for (let i = 0; i < n; i++) {
    const jr = jsMat.data[i * 3], jg = jsMat.data[i * 3 + 1], jb = jsMat.data[i * 3 + 2];
    const pb = pyRaw.data[i * 3], pg = pyRaw.data[i * 3 + 1], pr = pyRaw.data[i * 3 + 2];
    if (jr !== pr || jg !== pg || jb !== pb) diff++;
  }
  return diff;
}

// ORB/AKAZE draw a floating-point orientation line (FAST does not -- see keypoints-24). Measured on
// this exact fixed code (scratch probe, logged in the WORKLOG): ORB diffPixels = 30 @ 480x480 (scale=1
// exactly, no downscale at all), 48 @ 640x480 (downscale triggers but happens to be an exact scale, so
// int(x+0.5) vs int(x) makes no difference there either), 0 @ 1001x777 -- all with IDENTICAL drawn
// centers/radii (independently verified by keypoints-16/17's spy-based tests) -- the residual is a
// handful of pixels along the orientation-line edge, consistent with cos/sin float-rounding differences
// between opencv.js's and cv2's line-rasterization implementations at certain angles, not a codegen
// correctness bug. 200 is >100x smaller than every mutation-induced diff measured at these 3 sizes for
// the mutations this test is meant to catch (smallest true-positive was 4470px, from the
// int(x+0.5)->int(x) mutation specifically at the non-exact-scale 1001x777 size -- confirming why the
// spec calls out 1001x777 as the size that actually exercises that rounding).
const LINE_DRAWING_PIXEL_TOLERANCE = 200;

test('keypoints-23. MAJOR-B pixel-level parity: ORB drawn output image matches the JS op\'s actual output pixel-for-pixel (within a tiny documented line-rasterization tolerance) at 480x480 (scale=1, no downscale), 640x480 (downscale triggers but is an exact scale), and 1001x777 (non-exact scale -- exercises int(x+0.5) rounding on the resize dims)', () => {
  for (const [w, h] of [[480, 480], [640, 480], [1001, 777]]) {
    const src = parityTestImage(w, h);
    const c = new cv.Mat(); cv.cvtColor(src, c, cv.COLOR_GRAY2RGB);
    const raw = matToRaw(c);
    c.delete();
    const ctx = {};
    const jsOut = OPS.orb_keypoints.apply(cv, src, {}, ctx);
    const pyOut = runGeneratedPython([{ id: 'orb_keypoints', params: {} }], raw);
    const diff = pixelDiffCount(jsOut, pyOut);
    assert.ok(diff <= LINE_DRAWING_PIXEL_TOLERANCE,
      `ORB @ ${w}x${h}: expected <=${LINE_DRAWING_PIXEL_TOLERANCE} differing pixels between JS and Python drawn output, got ${diff}`);
    jsOut.delete();
    src.delete();
  }
});

test('keypoints-24. MAJOR-B pixel-level parity: FAST drawn output image matches the JS op\'s actual output EXACTLY (0 differing pixels -- FAST never draws a direction line, so there is no line-rasterization residual to tolerate) at 480x480, 640x480, and 1001x777', () => {
  for (const [w, h] of [[480, 480], [640, 480], [1001, 777]]) {
    const src = parityTestImage(w, h);
    const c = new cv.Mat(); cv.cvtColor(src, c, cv.COLOR_GRAY2RGB);
    const raw = matToRaw(c);
    c.delete();
    const params = { threshold: 10 };
    const ctx = {};
    const jsOut = OPS.fast_keypoints.apply(cv, src, params, ctx);
    const pyOut = runGeneratedPython([{ id: 'fast_keypoints', params }], raw);
    const diff = pixelDiffCount(jsOut, pyOut);
    assert.equal(diff, 0, `FAST @ ${w}x${h}: expected exactly 0 differing pixels between JS and Python drawn output, got ${diff}`);
    jsOut.delete();
    src.delete();
  }
});

test('keypoints-25. MAJOR-B pixel-level parity: AKAZE drawn output image matches the JS op\'s actual output (within the same line-rasterization tolerance as ORB, since AKAZE also draws an orientation line) at 480x480, 640x480, and 1001x777 -- skipped gracefully if opencv-contrib-python is missing, same detection logic as keypoints-11', (t) => {
  for (const [w, h] of [[480, 480], [640, 480], [1001, 777]]) {
    const src = parityTestImage(w, h);
    const c = new cv.Mat(); cv.cvtColor(src, c, cv.COLOR_GRAY2RGB);
    const raw = matToRaw(c);
    c.delete();
    const ctx = {};
    const jsOut = OPS.akaze_keypoints.apply(cv, src, {}, ctx);
    let pyOut;
    try {
      pyOut = runGeneratedPython([{ id: 'akaze_keypoints', params: {} }], raw);
    } catch (err) {
      const msg = String(err.stderr || err.message || err);
      jsOut.delete();
      src.delete();
      if (/AttributeError/.test(msg) && /has no attribute 'AKAZE_create'/.test(msg)) {
        t.skip('opencv-contrib-python not installed on this machine (cv2.AKAZE_create missing) — expected, confirmed by CV eng smoke test');
        return;
      }
      throw err;
    }
    const diff = pixelDiffCount(jsOut, pyOut);
    assert.ok(diff <= LINE_DRAWING_PIXEL_TOLERANCE,
      `AKAZE @ ${w}x${h}: expected <=${LINE_DRAWING_PIXEL_TOLERANCE} differing pixels between JS and Python drawn output, got ${diff}`);
    jsOut.delete();
    src.delete();
  }
});

// Small, documented tolerance mirroring MAJOR-A's reasoning above (see keypoints-21's comment): even
// though this exact fixed code measures 0 differing pixels at every size probed for
// shi_tomasi_corners's default maxCorners=100 (scratch probe, logged in the WORKLOG), a small allowance
// is kept for the same library-version corner-selection differences documented there, so this test does
// not become environment-brittle on a machine where opencv.js/cv2 disagree on one more corner near the
// quality threshold. 500 is far smaller than every mutation-induced diff measured for the mutations
// this test is meant to catch (draw-without-inv_scale: 13056-13493px; scale=1.0: 8116-12992px).
const SHI_TOMASI_PIXEL_TOLERANCE = 500;

test('keypoints-26. MAJOR-B pixel-level parity: Shi-Tomasi drawn output image matches the JS op\'s actual output (within the tolerant bound from MAJOR-A) at 600x500 and 1001x777, using the default maxCorners=100 (the realistic exported-script case, not the uncapped 5000 used in keypoints-21 to avoid the vacuous-cap problem)', () => {
  for (const [w, h] of [[600, 500], [1001, 777]]) {
    const src = parityTestImage(w, h);
    const c = new cv.Mat(); cv.cvtColor(src, c, cv.COLOR_GRAY2RGB);
    const raw = matToRaw(c);
    c.delete();
    const params = { maxCorners: 100, quality: 1, minDist: 10, blockSize: 3, useHarris: false };
    const ctx = {};
    const jsOut = OPS.shi_tomasi_corners.apply(cv, src, params, ctx);
    const pyOut = runGeneratedPython([{ id: 'shi_tomasi_corners', params }], raw);
    const diff = pixelDiffCount(jsOut, pyOut);
    assert.ok(diff <= SHI_TOMASI_PIXEL_TOLERANCE,
      `Shi-Tomasi @ ${w}x${h}: expected <=${SHI_TOMASI_PIXEL_TOLERANCE} differing pixels between JS and Python drawn output, got ${diff}`);
    jsOut.delete();
    src.delete();
  }
});

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
import { assertLeakFree, assertWrapperBalance, assertNoExtraHeapGrowth } from './_leak-helpers.mjs';

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
// Python execution harness
// ===================================================================
// หมายเหตุ (ข้อยกเว้นจาก spec §9 item 9 ที่ระบุให้ใช้ PNG): ไม่มี PNG encoder library ใน node_modules
// ของโปรเจกต์นี้ (ตรวจแล้ว — ไม่มี pngjs/sharp/canvas) จึงใช้การ dump raw uint8 bytes ตรง ๆ แทน ซึ่ง
// "lossless" ยิ่งกว่า PNG อีก (ไม่มีการ encode/decode ใด ๆ เลย ไม่ใช่แค่ไม่ lossy แบบ JPEG) — เจตนาของ
// กฎ "ห้ามใช้ JPEG" ในสเปกคือกันความคลาดเคลื่อนจาก compression noise ปนเข้ามาในผลเทียบ ซึ่ง raw dump
// ตัดปัญหานี้ทั้งหมดออกไปโดยไม่ต้องพึ่ง dependency ใหม่
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'groupA-parity-'));

function matToRaw(mat) {
  return { data: Buffer.from(mat.data), width: mat.cols, height: mat.rows, channels: mat.channels() };
}

// buildPythonCode() ต่อ cv2.cvtColor(BGR2GRAY) เข้าไปเสมอสำหรับ op ที่ needs:"gray" (สมมติว่า
// cv2.imread() มาก่อนเสมอ = 3-channel) — ถ้าส่ง raw 1-channel ตรง ๆ ให้ Python จะ throw "invalid number
// of channels" ทันที จึงต้องส่งภาพ "สี" 3-channel ที่ทุกช่องมีค่าเท่ากับภาพเกรย์เดิมเป๊ะ ๆ แทน (ผลลัพธ์
// หลัง cv2.COLOR_BGR2GRAY ของภาพแบบนี้ = ค่าเดิมเป๊ะ ไม่ว่าน้ำหนักช่องสีจะตีความเป็น BGR หรือ RGB
// เพราะทุกช่องเท่ากันอยู่แล้ว 0.299+0.587+0.114=1 พอดี) — สอดคล้องกับที่แอปจริงส่งภาพกล้อง/ไฟล์สีเข้า
// worker แล้วแปลงเป็นเกรย์เองก่อนเสมอสำหรับ op กลุ่มนี้ ไม่ใช่ภาพ 1-channel ดิบ ๆ
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

// รัน pipeline เดียวผ่าน buildPythonCode() จริง (โค้ด production เดียวกับที่ผู้ใช้กด export) แล้ว
// แทนที่ท่ออ่านไฟล์ cv2.imread(...) ด้วย raw-loader + ตัด main-guard (cv2.imshow/waitKey จะค้าง
// ในสภาพแวดล้อม headless) ออก แล้วต่อ driver ของเราเองที่ dump ผลลัพธ์กลับเป็น raw bytes
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

function meanAbsDiff(a, b) {
  assert.equal(a.length, b.length, `buffer length mismatch: ${a.length} vs ${b.length}`);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / a.length;
}

// เหมือน meanAbsDiff แต่คืนสถิติละเอียดกว่า (mean/max/จำนวนพิกเซลที่ต่างกัน/จำนวนพิกเซลทั้งหมด) —
// qa's S5 re-verification: mean-abs-diff bound เดียวหลวมเกินไป ปล่อยให้ mutant บางตัวรอดได้ (banker's
// rounding ใน entropy downscale, BORDER_REPLICATE ของ entropy window, truncate ของ std normalize)
// เพราะ mean เจือจางความต่างที่กระจุกอยู่แค่บางพิกเซล/บางค่าลงจนต่ำกว่า threshold เดิม ต้องดู max diff
// และสัดส่วนพิกเซลที่ต่างกันด้วยจึงจับ mutant เหล่านี้ได้จริง
function diffStats(a, b) {
  assert.equal(a.length, b.length, `buffer length mismatch: ${a.length} vs ${b.length}`);
  let sum = 0, max = 0, diffCount = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    sum += d;
    if (d > max) max = d;
    if (d !== 0) diffCount++;
  }
  return { mean: sum / a.length, max, diffCount, total: a.length };
}

// ===================================================================
// malloc-fingerprint leak probe (qa's technique — a coarse probe aimed at fixed-size Mat leaks that the
// 1500x1500 byteOffset sentinel probe elsewhere in this file cannot see) — NOTE (qa S5 re-verification):
// this probe can still MISS very small (<~100 byte) per-call leaks when run in-file alongside ~20 other
// cv-heavy tests in the same process (e.g. a forgotten M.delete() on perspective_warp's 72-byte 3x3
// CV_64F did not make this test fail, with or without the address-sort above) — the allocator's free-list
// for small buckets is noisy enough from prior tests' malloc/free history that a single small leak per
// call doesn't reliably drift the fingerprint. It does still catch the larger leaks it was written for
// (labels/distCoeffs/kernel-sized Mats). Hygiene for very small Mats (3x3 homography/distCoeffs, etc.) is
// therefore verified by code review / try-finally discipline, not by this test — do not rely on it alone.
// ===================================================================
// วางเทสต์นี้ไว้เป็นตัวแรกสุดในไฟล์ (ก่อน groupA-1 เป็นต้นไป) โดยตั้งใจ — ทดลองจริงพบว่าถ้าวางไว้ท้ายไฟล์
// (หลังเทสต์อื่นอีก ~20 ตัวที่ใช้ cv หนัก ๆ ไปแล้วในโปรเซสเดียวกัน) allocator ของ opencv.js (dlmalloc-style
// free-list) จะทิ้ง bucket ขนาดเล็กบางขนาด (ตรงกับ FP_SIZES พอดี เช่น 72 byte = ขนาด M ของ perspective_warp)
// ไว้ในสถานะที่ "สลับที่กันไปมา" ระหว่างสอง snapshot ได้ แม้ cv.HEAP8.buffer.byteLength จะไม่โตเลยสักครั้ง
// (พิสูจน์แล้วว่าไม่ใช่ leak จริง) — วนอุ่นเครื่องเพิ่มอีกกี่ร้อยรอบก็ยังไม่นิ่ง เพราะสภาพเริ่มต้นของ arena
// ถูกกำหนดจากประวัติการ malloc/free ของเทสต์ก่อนหน้าไปแล้ว ในทางกลับกัน สคริปต์แยกที่รันเฉพาะ op เดียวจาก
// process ใหม่ (เริ่มจาก state สะอาด) จะ "นิ่ง" ได้เองภายในไม่กี่ร้อยรอบเสมอ (ยืนยันด้วย scratch script)
// จึงย้ายเทสต์นี้มาไว้ต้นไฟล์ ให้รันบน heap ที่ยังสะอาดที่สุดเท่าที่จะทำได้ในไฟล์เดียว
const FP_SIZES = [8, 16, 24, 40, 56, 72, 104, 136, 200, 264, 520, 1032, 2056, 4104, 8200, 16392, 65544, 262152, 1048584, 4 << 20, 48 << 20];
function fingerprint() {
  const ptrs = [];
  for (const s of FP_SIZES) for (let r = 0; r < 4; r++) ptrs.push(cv._malloc(s));
  // เรียง address ก่อนเทียบ (ไม่ใช่ insertion order): allocator อาจคืน "ชุด" address เดิมเป๊ะ แต่คนละลำดับ
  // ระหว่างสอง snapshot ได้โดยไม่มี leak จริง (heap byteLength ไม่เปลี่ยน) — เรียงก่อนตัด false-positive
  // จากการสลับลำดับนี้ทิ้ง แต่ยังจับ leak จริงได้เหมือนเดิม เพราะถ้ามี leak จริง "ชุด" ของ address ที่ได้
  // กลับมาจะเป็นชุดใหม่ที่ต่างจากเดิมจริง ๆ ไม่ใช่แค่สลับลำดับ
  ptrs.sort((a, b) => a - b);
  const sig = ptrs.join(',');
  for (let i = ptrs.length - 1; i >= 0; i--) cv._free(ptrs[i]);
  return sig + '|' + cv.HEAP8.buffer.byteLength;
}

// ตาราง path ของ groupA-0 — ใช้ร่วมกันระหว่าง groupA-0 (fingerprint, process สะอาด) และ groupA-0w (wrapper-balance)
// ให้ทั้งสองด่านครอบ path ชุดเดียวกันเป๊ะ (P-14)
function groupA0Fixtures() {
  const gray = new cv.Mat.zeros(240, 320, cv.CV_8UC1);
  const square = gray.roi(new cv.Rect(20, 20, 100, 100));
  square.setTo(new cv.Scalar(255));
  square.delete();
  const color = new cv.Mat(240, 320, cv.CV_8UC3, new cv.Scalar(100, 110, 120));
  // skeleton needs >240px on the longest side, entropy needs >320px, to exercise their downscale paths
  const bigGray = new cv.Mat.zeros(400, 400, cv.CV_8UC1);
  const bigBlob = bigGray.roi(new cv.Rect(50, 50, 200, 200));
  bigBlob.setTo(new cv.Scalar(255));
  bigBlob.delete();
  return { gray, color, bigGray, dispose() { gray.delete(); color.delete(); bigGray.delete(); } };
}
function groupA0Paths({ gray, color, bigGray }) {
  return {
    'perspective_warp': (c) => { const o = OPS.perspective_warp.apply(c, color, {}, {}); o.delete(); },
    'undistort': (c) => { const o = OPS.undistort.apply(c, color, { k1: -0.3, k2: 0 }, {}); o.delete(); },
    'local_std_filter': (c) => { const o = OPS.local_std_filter.apply(c, gray, { windowSize: 9 }, {}); o.delete(); },
    'local_entropy_filter (>320px downscale path)': (c) => { const o = OPS.local_entropy_filter.apply(c, bigGray, { windowSize: '5', bins: '16' }, {}); o.delete(); },
    'morphology/skeleton (>240px downscale path)': (c) => { const o = OPS.morphology.apply(c, bigGray, { op: 'skeleton', shape: 'rect', ksize: 5, iterations: 1 }, {}); o.delete(); },
    'morphology/reconstruct_open': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'reconstruct_open', shape: 'rect', ksize: 5, iterations: 1 }, {}); o.delete(); },
    'morphology/erode-diamond': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'erode', shape: 'diamond', ksize: 7, iterations: 1 }, {}); o.delete(); },
    'morphology/dilate-octagon': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'dilate', shape: 'octagon', ksize: 7, iterations: 1 }, {}); o.delete(); },
    'morphology/open-line': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'open', shape: 'line', ksize: 7, iterations: 1, lineAngle: 30 }, {}); o.delete(); },
    'morphology/gradient-diamond': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'gradient', shape: 'diamond', ksize: 7, iterations: 1 }, {}); o.delete(); },
  };
}

test('groupA-0 (malloc-fingerprint). every new op (incl. custom strel shapes with erode/dilate/open/gradient, skeleton >240px downscale path, entropy >320px downscale path, reconstruct_open) leaves the allocator fingerprint unchanged — inputs kept <=400x400 per qa\'s note that cv.dilate at 1920x1080 oscillates the fingerprint without a real leak', () => {
  const fx = groupA0Fixtures();
  const paths = groupA0Paths(fx);
  const runOnce = Object.fromEntries(Object.entries(paths).map(([k, f]) => [k, () => f(cv)]));

  for (const [label, run] of Object.entries(runOnce)) {
    const heavy = label.includes('skeleton') || label.includes('reconstruct') || label.includes('entropy');
    const warmup = heavy ? 60 : 300;
    for (let i = 0; i < warmup; i++) run();
    const fpA = fingerprint();
    const iters = heavy ? 30 : 100;
    for (let i = 0; i < iters; i++) run();
    const fpB = fingerprint();
    assert.equal(fpB, fpA, `${label}: malloc fingerprint changed after ${iters} more calls -> Mat leak (allocator state drifted)`);
  }
  fx.dispose();
});

// groupA-0w — ด่านหลัก (wrapper-balance) ของ path ชุดเดียวกับ groupA-0: fingerprint ด้านบนเชื่อได้เฉพาะในฐานะเทสต์แรก
// ของไฟล์ และไม่เห็น wrapper เล็ก ๆ (qa S5: M 3x3 ของ perspective_warp 72B หลุดรอด fingerprint) — tracker นับทุก wrapper
// เทียบ .delete() จึงจับได้ทุกขนาด (P-14)
test('groupA-0w (wrapper-balance companion of groupA-0). every groupA-0 path deletes every embind wrapper it creates (incl. tiny 3x3 homography / distCoeffs Mats the fingerprint cannot see)', () => {
  const fx = groupA0Fixtures();
  try {
    for (const [label, run] of Object.entries(groupA0Paths(fx))) assertWrapperBalance(cv, label, run, { minCreated: 2 });
  } finally { fx.dispose(); }
});

// ===================================================================
// 1. Perspective Warp
// ===================================================================

test('groupA-1. Perspective Warp: default params moves the top-left marker rightward, keeps output dims', () => {
  const src = new cv.Mat.zeros(100, 100, cv.CV_8UC1);
  // marker: a small bright square at the original top-left corner
  const marker = src.roi(new cv.Rect(0, 0, 5, 5));
  marker.setTo(new cv.Scalar(255));
  marker.delete();
  const out = OPS.perspective_warp.apply(cv, src, {}, {});
  assert.equal(out.cols, 100);
  assert.equal(out.rows, 100);
  // find the centroid x of nonzero pixels near the top rows — should have moved right of x=0..4
  let sumX = 0, count = 0;
  for (let y = 0; y < 15; y++) {
    for (let x = 0; x < 100; x++) {
      if (out.data[y * 100 + x] > 100) { sumX += x; count++; }
    }
  }
  assert.ok(count > 0, 'expected some bright pixels near the top after warp');
  const meanX = sumX / count;
  assert.ok(meanX > 5, `expected marker to move right of its original x<5 position, got mean x=${meanX}`);
  out.delete();
  src.delete();
});

// ===================================================================
// 2. Local Std Filter
// ===================================================================

test('groupA-2. Local Std Filter: flat half is much lower than noisy half', () => {
  const w = 60, h = 40;
  const src = new cv.Mat(h, w, cv.CV_8UC1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      src.data[y * w + x] = x < w / 2 ? 128 : ((x * 53 + y * 97) % 256);
    }
  }
  const out = OPS.local_std_filter.apply(cv, src, { windowSize: 9 }, {});
  let flatSum = 0, flatN = 0, noisySum = 0, noisyN = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x < w / 2 - 10) { flatSum += out.data[y * w + x]; flatN++; }
      else if (x > w / 2 + 10) { noisySum += out.data[y * w + x]; noisyN++; }
    }
  }
  const flatMean = flatSum / flatN, noisyMean = noisySum / noisyN;
  assert.ok(flatMean < noisyMean, `expected flat region (${flatMean}) much lower than noisy region (${noisyMean})`);
  out.delete();
  src.delete();
});

// ===================================================================
// 3. Local Entropy Filter
// ===================================================================

test('groupA-3. Local Entropy Filter: flat half ~=0, noisy half meaningfully higher, completes <150ms', () => {
  const w = 60, h = 40;
  const src = new cv.Mat(h, w, cv.CV_8UC1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      src.data[y * w + x] = x < w / 2 ? 128 : ((x * 53 + y * 97) % 256);
    }
  }
  const started = performance.now();
  const out = OPS.local_entropy_filter.apply(cv, src, { windowSize: '5', bins: '16' }, {});
  const elapsed = performance.now() - started;
  assert.ok(elapsed < 150, `expected under 150ms, took ${elapsed}ms`);
  let flatSum = 0, flatN = 0, noisySum = 0, noisyN = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x < w / 2 - 10) { flatSum += out.data[y * w + x]; flatN++; }
      else if (x > w / 2 + 10) { noisySum += out.data[y * w + x]; noisyN++; }
    }
  }
  const flatMean = flatSum / flatN, noisyMean = noisySum / noisyN;
  assert.ok(flatMean < 10, `expected near-zero entropy on a flat region, got ${flatMean}`);
  assert.ok(noisyMean > flatMean + 30, `expected noisy region entropy meaningfully higher, flat=${flatMean} noisy=${noisyMean}`);
  out.delete();
  src.delete();
});

// ===================================================================
// 4. Skeleton
// ===================================================================

test('groupA-4. Skeleton: thick rectangle thins a lot but keeps spanning its original extent', () => {
  const w = 60, h = 60;
  const src = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  const rect = src.roi(new cv.Rect(10, 25, 40, 10)); // 40x10 solid bar
  rect.setTo(new cv.Scalar(255));
  rect.delete();
  const inputCount = cv.countNonZero(src);
  const ctx = {};
  const out = OPS.morphology.apply(cv, src, { op: 'skeleton', shape: 'rect', ksize: 5, iterations: 1 }, ctx);
  const outCount = cv.countNonZero(out);
  assert.ok(outCount < inputCount / 3, `expected skeleton to thin heavily: input=${inputCount} output=${outCount}`);
  assert.ok(outCount > 0, 'expected skeleton to leave at least some pixels');
  // must still span roughly the original horizontal extent (topology/extent preserved)
  let minX = w, maxX = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (out.data[y * w + x] > 0) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); }
    }
  }
  assert.ok(maxX - minX > 25, `expected skeleton to still span most of the original 40px width, got span=${maxX - minX}`);
  assert.match(ctx.info, /Skeleton/);
  out.delete();
  src.delete();
});

test('groupA-4b. Morphology regression: plain erode/dilate still work unchanged after adding skeleton/reconstruct_open branches', () => {
  const src = new cv.Mat.zeros(40, 40, cv.CV_8UC1);
  const blob = src.roi(new cv.Rect(10, 10, 20, 20));
  blob.setTo(new cv.Scalar(255));
  blob.delete();
  const eroded = OPS.morphology.apply(cv, src, { op: 'erode', shape: 'rect', ksize: 3, iterations: 1 }, {});
  const dilated = OPS.morphology.apply(cv, src, { op: 'dilate', shape: 'rect', ksize: 3, iterations: 1 }, {});
  assert.ok(cv.countNonZero(eroded) < cv.countNonZero(src), 'erode must shrink the blob');
  assert.ok(cv.countNonZero(dilated) > cv.countNonZero(src), 'dilate must grow the blob');
  eroded.delete();
  dilated.delete();
  src.delete();
});

// ===================================================================
// 5. Opening by Reconstruction
// ===================================================================

test('groupA-5. Opening by Reconstruction: small speck disappears, sharp corner of the main blob survives (unlike plain Opening)', () => {
  const w = 80, h = 80;
  const src = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  // main blob with a deliberately sharp/angular corner (an L-shape) at (20,20)-(50,50)
  const blob = src.roi(new cv.Rect(20, 20, 30, 30));
  blob.setTo(new cv.Scalar(255));
  blob.delete();
  // small 3x3 speck far away, isolated — too small to survive a 5x5 erosion marker
  const speck = src.roi(new cv.Rect(65, 5, 3, 3));
  speck.setTo(new cv.Scalar(255));
  speck.delete();

  const ctx = {};
  const out = OPS.morphology.apply(cv, src, { op: 'reconstruct_open', shape: 'rect', ksize: 5, iterations: 1 }, ctx);
  // speck must be gone
  assert.equal(out.data[6 * w + 66], 0, 'small isolated speck must be removed');
  // sharp corner (20,20) must survive exactly (reconstruction preserves shape, unlike rounded Opening)
  assert.equal(out.data[21 * w + 21], 255, 'sharp corner pixel of the main blob must survive unchanged');
  assert.equal(out.data[20 * w + 20], 255, 'exact corner pixel must survive unchanged');
  assert.match(ctx.info, /Opening by Reconstruction/);
  out.delete();
  src.delete();
});

// ===================================================================
// 6. strel diamond/octagon/line
// ===================================================================

test('groupA-6. strel diamond/octagon/line: erode does not throw and produces geometry distinct from ellipse/rect', () => {
  const w = 80, h = 80;
  const src = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  // สำคัญ: ต้องเป็นวงกลม (มีขอบเฉียง) ไม่ใช่สี่เหลี่ยม — erosion ของสี่เหลี่ยมด้วย kernel นูนสมมาตรใด ๆ
  // (diamond/ellipse/rect ที่มี bounding box รัศมีเท่ากัน) ให้ผลเหมือนกันเป๊ะเสมอ (พิสูจน์แล้วด้วยการ
  // ทดลองจริง — erosion ของสี่เหลี่ยมขึ้นกับ "ระยะเอื้อม" ของ kernel ตามแนวแกน x/y เท่านั้น ไม่ขึ้นกับ
  // รูปทรงตามแนวทแยงเลย) ต้องใช้รูปที่มีขอบเฉียง/โค้งจริง ๆ ถึงจะเห็นความต่างของรูปทรง kernel
  cv.circle(src, new cv.Point(40, 40), 30, new cv.Scalar(255), -1);

  const ellipseOut = OPS.morphology.apply(cv, src, { op: 'erode', shape: 'ellipse', ksize: 15, iterations: 1 }, {});
  const diamondOut = OPS.morphology.apply(cv, src, { op: 'erode', shape: 'diamond', ksize: 15, iterations: 1 }, {});
  const octagonOut = OPS.morphology.apply(cv, src, { op: 'erode', shape: 'octagon', ksize: 15, iterations: 1 }, {});
  // เทียบทีละพิกเซล (ไม่ใช่แค่ countNonZero) — รูปทรงต่างกันอาจบังเอิญเหลือพื้นที่รวมเท่ากันได้
  // ทั้งที่ตำแหน่งพิกเซลต่างกันจริง (diamond มุมแหลมกว่า ellipse ที่มุมกลม)
  assert.notDeepEqual(Array.from(diamondOut.data), Array.from(ellipseOut.data), 'diamond erosion must differ pixel-for-pixel from ellipse erosion');
  assert.ok(cv.countNonZero(octagonOut) > 0, 'octagon erode must not throw and must leave some pixels');
  ellipseOut.delete(); diamondOut.delete(); octagonOut.delete();

  // line: anisotropic input (a horizontal bar) — angle 0 vs 90 must produce measurably different erosion
  const bar = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  const hbar = bar.roi(new cv.Rect(5, 18, 30, 4)); // horizontal bar, 4px tall
  hbar.setTo(new cv.Scalar(255));
  hbar.delete();
  const line0 = OPS.morphology.apply(cv, bar, { op: 'erode', shape: 'line', ksize: 9, iterations: 1, lineAngle: 0 }, {});
  const line90 = OPS.morphology.apply(cv, bar, { op: 'erode', shape: 'line', ksize: 9, iterations: 1, lineAngle: 90 }, {});
  assert.notEqual(cv.countNonZero(line0), cv.countNonZero(line90), 'lineAngle=0 vs 90 must produce different erosion on an anisotropic (horizontal bar) input');
  line0.delete(); line90.delete();
  src.delete();
  bar.delete();
});

// ===================================================================
// 7. Undistort
// ===================================================================

test('groupA-7. Undistort: grid line curvature changes measurably between k1=0 and k1=-0.5', () => {
  const w = 100, h = 100;
  const grid = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  for (let y = 0; y < h; y += 10) {
    const row = grid.roi(new cv.Rect(0, y, w, 1));
    row.setTo(new cv.Scalar(255));
    row.delete();
  }
  for (let x = 0; x < w; x += 10) {
    const col = grid.roi(new cv.Rect(x, 0, 1, h));
    col.setTo(new cv.Scalar(255));
    col.delete();
  }
  const identityish = OPS.undistort.apply(cv, grid, { k1: 0, k2: 0 }, {});
  const barrel = OPS.undistort.apply(cv, grid, { k1: -0.5, k2: 0 }, {});
  const diff = meanAbsDiff(identityish.data, barrel.data);
  assert.ok(diff > 3, `expected k1=-0.5 to visibly differ from k1=0 (mean abs diff=${diff})`);
  identityish.delete();
  barrel.delete();

  // flat/constant image stays visually similar under any k1/k2 (no texture to distort)
  const flat = new cv.Mat(h, w, cv.CV_8UC1, new cv.Scalar(120));
  const flatOut = OPS.undistort.apply(cv, flat, { k1: -0.5, k2: 0.2 }, {});
  const flatDiff = meanAbsDiff(flat.data, flatOut.data);
  assert.ok(flatDiff < 5, `expected a flat image to look near-unchanged under undistort, diff=${flatDiff}`);
  flat.delete();
  flatOut.delete();
  grid.delete();
});

// ===================================================================
// 8. Python codegen + real cv2 execution for every new op
// ===================================================================

test('groupA-8. Python codegen executes without error on real cv2 for every new op, producing a sane (non-constant, correct-shape) output', () => {
  const w = 48, h = 40;
  const gray = new cv.Mat(h, w, cv.CV_8UC1);
  for (let i = 0; i < gray.data.length; i++) gray.data[i] = (i * 37) % 256;
  const grayBin = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  const blob = grayBin.roi(new cv.Rect(10, 10, 20, 15));
  blob.setTo(new cv.Scalar(255));
  blob.delete();
  const color = new cv.Mat(h, w, cv.CV_8UC3);
  for (let i = 0; i < color.data.length; i++) color.data[i] = (i * 53) % 256;

  const cases = [
    { id: 'perspective_warp', mat: color, isGray: false, params: { tlX: 10, tlY: 0, trX: -10, trY: 0, brX: 0, brY: 0, blX: 0, blY: 0 } },
    { id: 'undistort', mat: color, isGray: false, params: { k1: -0.3, k2: 0 } },
    { id: 'local_std_filter', mat: gray, isGray: true, params: { windowSize: 7 } },
    { id: 'local_entropy_filter', mat: gray, isGray: true, params: { windowSize: '5', bins: '16' } },
    { id: 'morphology', mat: grayBin, isGray: true, params: { op: 'skeleton', shape: 'rect', ksize: 5, iterations: 1 } },
    { id: 'morphology', mat: grayBin, isGray: true, params: { op: 'reconstruct_open', shape: 'rect', ksize: 5, iterations: 1 } },
    { id: 'morphology', mat: grayBin, isGray: true, params: { op: 'erode', shape: 'diamond', ksize: 7, iterations: 1 } },
    { id: 'morphology', mat: grayBin, isGray: true, params: { op: 'erode', shape: 'octagon', ksize: 7, iterations: 1 } },
    { id: 'morphology', mat: grayBin, isGray: true, params: { op: 'erode', shape: 'line', ksize: 7, iterations: 1, lineAngle: 45 } },
  ];

  for (const c of cases) {
    const raw = c.isGray ? grayMatToColorRaw(c.mat) : matToRaw(c.mat);
    const result = runGeneratedPython([{ id: c.id, params: c.params }], raw);
    assert.equal(result.height, c.mat.rows, `${c.id}: unexpected output height`);
    assert.equal(result.width, c.mat.cols, `${c.id}: unexpected output width`);
    let min = 255, max = 0;
    for (let i = 0; i < result.data.length; i++) { min = Math.min(min, result.data[i]); max = Math.max(max, result.data[i]); }
    assert.ok(max > min, `${c.id}: expected non-constant Python output`);
  }
  gray.delete();
  grayBin.delete();
  color.delete();
});

// ===================================================================
// 9. JS-vs-Python numeric parity (perspective_warp, undistort, morphology-shape)
// ===================================================================

// textured (non-binary) gray pattern สำหรับเทสต์ parity ของ local_std_filter/local_entropy_filter —
// checkerboard 0/255 เดิม (ค่าเทาแค่ 2 ระดับ) ไม่พอที่จะเห็นความต่างจาก banker's-rounding ของ downscale,
// BORDER_REPLICATE ของ window เดิม, หรือ truncate ของ normalize เดิม เพราะพิกเซลข้างเคียงส่วนใหญ่มีค่า
// เท่ากันอยู่แล้ว (window เนื้อเดียว entropy/variance ต่ำหรือ 0 ทั้งคู่) ต้องใช้ภาพที่มีเนื้อ (texture)
// ต่อเนื่องแบบนี้จึงจะ exercise ความต่างเชิงตัวเลขจริง ๆ (ยืนยันแล้วว่า mutant ทั้งสามตัวถูกจับได้ด้วย
// input นี้ — ดู WORKLOG)
function texturedGraySrc(w, h) {
  const src = new cv.Mat(h, w, cv.CV_8UC1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      src.data[y * w + x] = (x * 53 + y * 97) % 256;
    }
  }
  return src;
}

// opts.full === true: คืน { mean, max, diffCount, total } (ดู diffStats) แทน mean เฉย ๆ — ใช้เมื่อ
// mean-abs-diff เดียวหลวมเกินไปสำหรับเทสต์ที่ต้องจับ mutant ละเอียด (groupA-19/20)
function parityCheck(id, params, w, h, needsGray, customSrc, opts) {
  let src = customSrc;
  let ownsSrc = false;
  if (!src) {
    ownsSrc = true;
    src = needsGray ? new cv.Mat.zeros(h, w, cv.CV_8UC1) : new cv.Mat(h, w, cv.CV_8UC3);
    if (needsGray) {
      // binary blob pattern (checkerboard of squares) so morphology shapes have real structure to act on
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          src.data[y * w + x] = (Math.floor(x / 10) + Math.floor(y / 10)) % 2 === 0 ? 255 : 0;
        }
      }
    } else {
      for (let i = 0; i < src.data.length; i++) src.data[i] = (i * 41 + 17) % 256;
    }
  }
  const jsOut = id === 'morphology' ? OPS.morphology.apply(cv, src, params, {}) : OPS[id].apply(cv, src, params, {});
  const raw = needsGray ? grayMatToColorRaw(src) : matToRaw(src);
  const pyOut = runGeneratedPython([{ id, params }], raw);
  assert.equal(pyOut.width, jsOut.cols, `${id} @ ${src.cols}x${src.rows}: width mismatch`);
  assert.equal(pyOut.height, jsOut.rows, `${id} @ ${src.cols}x${src.rows}: height mismatch`);
  const stats = diffStats(Buffer.from(jsOut.data), pyOut.data);
  jsOut.delete();
  if (ownsSrc) src.delete();
  return (opts && opts.full) ? stats : stats.mean;
}

test('groupA-9. JS-vs-Python parity: perspective_warp, undistort, morphology-diamond at <=320px and >640px', () => {
  const pwParams = { tlX: 10, tlY: 0, trX: -10, trY: 0, brX: 0, brY: 0, blX: 0, blY: 0 };
  const cases = [
    ['perspective_warp', pwParams, 200, 150, false],
    ['perspective_warp', pwParams, 700, 500, false],
    ['undistort', { k1: -0.3, k2: 0.05 }, 200, 150, false],
    ['undistort', { k1: -0.3, k2: 0.05 }, 700, 500, false],
    ['morphology', { op: 'erode', shape: 'diamond', ksize: 7, iterations: 1 }, 200, 150, true],
    ['morphology', { op: 'erode', shape: 'diamond', ksize: 7, iterations: 1 }, 700, 500, true],
  ];
  for (const [id, params, w, h, needsGray] of cases) {
    const diff = parityCheck(id, params, w, h, needsGray);
    assert.ok(diff <= 3, `${id} @ ${w}x${h}: expected mean abs diff <=3, got ${diff}`);
  }
});

// ===================================================================
// 10. Zhang-Suen JS-vs-Python edge-wraparound check
// ===================================================================

test('groupA-10. Skeleton JS-vs-Python parity on a blob that touches the image border (np.roll wraparound regression guard)', () => {
  const w = 40, h = 50;
  const src = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  // blob touching the top and left border simultaneously
  const blob1 = src.roi(new cv.Rect(0, 0, 20, 25));
  blob1.setTo(new cv.Scalar(255));
  blob1.delete();
  // separate blob touching the bottom-right border
  const blob2 = src.roi(new cv.Rect(25, 35, 15, 15));
  blob2.setTo(new cv.Scalar(255));
  blob2.delete();

  const diff = parityCheck('morphology', { op: 'skeleton', shape: 'rect', ksize: 5, iterations: 1 }, w, h, true, src);
  assert.equal(diff, 0, `expected pixel-perfect JS/Python skeleton parity on a border-touching blob (np.roll wraparound must not leak into the result), got mean abs diff=${diff}`);
  src.delete();
});

// ===================================================================
// §11 (spec numbering) — Mat-leak sanity for every new op
// ===================================================================

// P-14: เดิมเป็น sentinel 1500x1500x3 byteOffset + HEAP8.buffer.byteLength (ขึ้นกับที่อยู่ของบล็อก / โตทีละ 128MB)
// ตอนนี้ใช้ assertLeakFree = wrapper-balance (ด่านหลัก) + heap census (ไม่ขึ้นกับที่อยู่) ชื่อเทสต์คงเดิม
test('leak probe: cheap single-pass groupA ops (perspective_warp, undistort, local_std_filter, strel-shape erode/dilate) do not leak Mats (200 iterations each)', () => {
  const src = new cv.Mat(240, 320, cv.CV_8UC3, new cv.Scalar(120, 130, 140));
  const gray = new cv.Mat.zeros(240, 320, cv.CV_8UC1);
  const square = gray.roi(new cv.Rect(20, 20, 100, 100));
  square.setTo(new cv.Scalar(255));
  square.delete();

  const runOnce = {
    'geometry/perspective_warp': (c) => { const o = OPS.perspective_warp.apply(c, src, {}, {}); o.delete(); },
    'geometry/undistort': (c) => { const o = OPS.undistort.apply(c, src, { k1: -0.3, k2: 0 }, {}); o.delete(); },
    'shape_analysis/local_std_filter': (c) => { const o = OPS.local_std_filter.apply(c, gray, { windowSize: 9 }, {}); o.delete(); },
    'morphology/erode-diamond': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'erode', shape: 'diamond', ksize: 7, iterations: 1 }, {}); o.delete(); },
    'morphology/dilate-octagon': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'dilate', shape: 'octagon', ksize: 7, iterations: 1 }, {}); o.delete(); },
    'morphology/erode-line': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'erode', shape: 'line', ksize: 7, iterations: 1, lineAngle: 30 }, {}); o.delete(); },
  };

  for (const [label, run] of Object.entries(runOnce)) {
    assertLeakFree(cv, label, run, { minCreated: 2, iters: 200, warmup: 20 });
  }
  gray.delete();
  src.delete();
});

test('leak probe: iterative groupA ops (skeleton, reconstruct_open, local_entropy_filter) do not leak Mats (30-50 iterations, higher per-call cost)', () => {
  const gray = new cv.Mat.zeros(240, 320, cv.CV_8UC1);
  const square = gray.roi(new cv.Rect(20, 20, 100, 100));
  square.setTo(new cv.Scalar(255));
  square.delete();

  const runOnce = {
    'morphology/skeleton': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'skeleton', shape: 'rect', ksize: 5, iterations: 1 }, {}); o.delete(); },
    'morphology/reconstruct_open': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'reconstruct_open', shape: 'rect', ksize: 5, iterations: 1 }, {}); o.delete(); },
    'shape_analysis/local_entropy_filter': (c) => { const o = OPS.local_entropy_filter.apply(c, gray, { windowSize: '5', bins: '16' }, {}); o.delete(); },
  };

  for (const [label, run] of Object.entries(runOnce)) {
    const iters = label === 'morphology/skeleton' ? 30 : 50;
    assertLeakFree(cv, label, run, { minCreated: 2, iters, warmup: 5 });
  }
  gray.delete();
});

// ===================================================================
// §12 (spec numbering) — Wiring sanity
// ===================================================================

test('wiring: perspective_warp/undistort/local_std_filter/local_entropy_filter appear in their FILTERS mode select options', () => {
  const groupedNewOps = [
    ['geometry', 'perspective_warp'],
    ['geometry', 'undistort'],
    ['shape_analysis', 'local_std_filter'],
    ['shape_analysis', 'local_entropy_filter'],
  ];
  for (const [groupId, opId] of groupedNewOps) {
    const modeParam = FILTERS[groupId].params.find((p) => p.key === 'mode');
    assert.ok(modeParam, `${groupId} must have a mode select`);
    assert.ok(modeParam.options.some((o) => o.value === opId), `${groupId}'s mode select must include ${opId}`);
  }
});

test('wiring: skeleton/reconstruct_open appear in morphology\'s own flat op select; diamond/octagon/line appear in the shape select', () => {
  const opParam = FILTERS.morphology.params.find((p) => p.key === 'op');
  assert.ok(opParam.options.some((o) => o.value === 'skeleton'));
  assert.ok(opParam.options.some((o) => o.value === 'reconstruct_open'));
  const shapeParam = FILTERS.morphology.params.find((p) => p.key === 'shape');
  assert.ok(shapeParam.options.some((o) => o.value === 'diamond'));
  assert.ok(shapeParam.options.some((o) => o.value === 'octagon'));
  assert.ok(shapeParam.options.some((o) => o.value === 'line'));
});

test('wiring: all palette-visible new ops appear in TOOL_SECTIONS and resolve to a registered FILTERS group', () => {
  const paletteOps = flattenPaletteOps();
  const newOps = ['perspective_warp', 'undistort', 'local_std_filter', 'local_entropy_filter'];
  for (const opId of newOps) {
    assert.ok(paletteOps.includes(opId), `expected ${opId} to be listed in some TOOL_SECTIONS entry`);
  }
});

// ===================================================================
// §13 (spec numbering) — Backward-compat regression
// ===================================================================

test('backward-compat: saved morphology params with shape rect/ellipse/cross and no lineAngle key still work and match the direct getStructuringElement path', () => {
  const src = new cv.Mat.zeros(40, 40, cv.CV_8UC1);
  const blob = src.roi(new cv.Rect(10, 10, 20, 20));
  blob.setTo(new cv.Scalar(255));
  blob.delete();

  for (const shape of ['rect', 'ellipse', 'cross']) {
    // simulate an old saved project's params object: no "lineAngle" key at all
    const savedParams = { op: 'erode', shape, ksize: 5, iterations: 1 };
    assert.ok(!('lineAngle' in savedParams));
    const out = OPS.morphology.apply(cv, src, savedParams, {});

    const shapeConst = { rect: cv.MORPH_RECT, ellipse: cv.MORPH_ELLIPSE, cross: cv.MORPH_CROSS }[shape];
    const kernel = cv.getStructuringElement(shapeConst, new cv.Size(5, 5));
    const expected = new cv.Mat();
    cv.erode(src, expected, kernel, new cv.Point(-1, -1), 1);
    assert.deepEqual(Array.from(out.data), Array.from(expected.data), `${shape}: output must match the direct getStructuringElement path exactly`);
    out.delete();
    expected.delete();
    kernel.delete();
  }
  src.delete();
});

test('backward-compat: morphology fill_holes/clear_border/remove_small op branches untouched by the new skeleton/reconstruct_open/shape branches', () => {
  const src = new cv.Mat.zeros(60, 60, cv.CV_8UC1);
  const ring = src.roi(new cv.Rect(10, 10, 40, 40));
  ring.setTo(new cv.Scalar(255));
  ring.delete();
  const hole = src.roi(new cv.Rect(20, 20, 20, 20));
  hole.setTo(new cv.Scalar(0));
  hole.delete();
  const ctx = {};
  const out = OPS.morphology.apply(cv, src, { op: 'fill_holes', shape: 'rect', ksize: 5, iterations: 1 }, ctx);
  assert.match(ctx.info, /เติมรูที่ปิดสนิท/);
  assert.equal(out.data[30 * 60 + 30], 255, 'the enclosed hole must be filled');
  out.delete();
  src.delete();
});

// ===================================================================
// S5 qa-recheck regression tests (bugs found in the S5 bugs-only review)
// ===================================================================
// เทสต์ทั้งหมดในหมวดนี้ถูกเพิ่มเพื่ออุด "test gap" ที่ qa รายงานว่า mutant รอดจากเทสต์เดิม (S5 §"TEST GAPS")
// ทุกจุดยืนยันแล้วว่า FAIL จริงกับโค้ดก่อนแก้ (ดูรายละเอียดใน WORKLOG) แล้ว PASS หลังแก้

test('groupA-14. JS-vs-Python parity: perspective_warp/undistort with an explicit 0 value must NOT fall back to the default (regression for the `num(v) || d` falsy-zero codegen bug)', () => {
  const allZeroPw = { tlX: 0, tlY: 0, trX: 0, trY: 0, brX: 0, brY: 0, blX: 0, blY: 0 };
  const cases = [
    ['perspective_warp', allZeroPw, 200, 150, false],
    ['perspective_warp', allZeroPw, 700, 500, false],
    ['perspective_warp', { tlX: 0, trX: 0 }, 200, 150, false],
    ['undistort', { k1: 0, k2: 0.3 }, 200, 150, false],
    ['undistort', { k1: 0, k2: 0.3 }, 700, 500, false],
  ];
  for (const [id, params, w, h, needsGray] of cases) {
    const diff = parityCheck(id, params, w, h, needsGray);
    assert.ok(diff <= 3, `${id} with explicit 0 param(s) @ ${w}x${h}: expected mean abs diff <=3 (0 must not fall back to the default), got ${diff}`);
  }
});

test('groupA-15. JS-vs-Python parity: perspective_warp/undistort with a missing params object ({}) still agree on the default-driven output', () => {
  const cases = [
    ['perspective_warp', {}, 200, 150, false],
    ['undistort', {}, 200, 150, false],
  ];
  for (const [id, params, w, h, needsGray] of cases) {
    const diff = parityCheck(id, params, w, h, needsGray);
    assert.ok(diff <= 3, `${id} with empty params @ ${w}x${h}: expected mean abs diff <=3, got ${diff}`);
  }
});

// expected diamond/octagon footprint math — deliberately duplicated from buildCustomStrel() in
// registry.js (not imported, since it's a private module-level helper) so this test independently
// verifies the *actual geometry*, not just "differs from ellipse" (which a full-square mutant could
// still satisfy by accident on some inputs)
function expectedCustomStrelMask(shape, k) {
  const r = Math.floor(k / 2);
  const size = 2 * r + 1;
  const mask = new Array(size * size).fill(0);
  if (shape === 'diamond') {
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      if (Math.abs(x - r) + Math.abs(y - r) <= r) mask[y * size + x] = 1;
    }
  } else if (shape === 'octagon') {
    const cut = Math.round(r / 2.5);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const dx = Math.abs(x - r), dy = Math.abs(y - r);
      if (dx <= r && dy <= r && dx + dy <= 2 * r - cut) mask[y * size + x] = 1;
    }
  }
  return { mask, size, r };
}

test('groupA-16. strel diamond/octagon: dilating a single centred pixel reproduces the exact expected kernel footprint at ksize 1, 3, and an even ksize 4 (kills a "kernel replaced by a full square" mutant)', () => {
  for (const shape of ['diamond', 'octagon']) {
    for (const ksize of [1, 3, 4]) {
      const { mask, size, r } = expectedCustomStrelMask(shape, ksize);
      const canvas = size + 20;
      const src = new cv.Mat.zeros(canvas, canvas, cv.CV_8UC1);
      const cx = Math.floor(canvas / 2), cy = Math.floor(canvas / 2);
      src.data[cy * canvas + cx] = 255;
      const out = OPS.morphology.apply(cv, src, { op: 'dilate', shape, ksize, iterations: 1 }, {});
      const mismatches = [];
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const expected = mask[y * size + x];
          const actual = out.data[(cy - r + y) * canvas + (cx - r + x)] > 0 ? 1 : 0;
          if (expected !== actual) mismatches.push(`(${x},${y}) expected=${expected} actual=${actual}`);
        }
      }
      assert.equal(mismatches.length, 0, `${shape} ksize=${ksize}: dilated footprint must match the expected mask exactly, mismatches: ${mismatches.join('; ')}`);
      const expectedArea = mask.reduce((a, b) => a + b, 0);
      assert.equal(cv.countNonZero(out), expectedArea, `${shape} ksize=${ksize}: total nonzero pixel count must equal the expected mask area (${expectedArea}) — a full-square fallback would overcount`);
      out.delete();
      src.delete();
    }
  }
});

test('groupA-17. Opening by Reconstruction vs plain Opening: a >300px-long thin crack attached to a thick blob is fully recovered by reconstruction (unlike the old 300-iteration cap) but removed by plain Opening; an isolated thin line disappears under both', () => {
  const w = 900, h = 100;
  const src = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  cv.rectangle(src, new cv.Point(5, 5), new cv.Point(60, 60), new cv.Scalar(255), -1); // thick blob
  let lastT = 0;
  for (let t = 0; t < 800; t++) {
    const xx = 60 + t;
    const yy = 30 + Math.round(Math.sin(t / 15) * 8);
    if (xx + 2 >= w || yy + 2 >= h || yy - 1 < 0) break;
    cv.rectangle(src, new cv.Point(xx, yy - 1), new cv.Point(xx + 2, yy + 1), new cv.Scalar(255), -1); // 3px-wide zigzag crack
    lastT = t;
  }
  assert.ok(lastT > 300, `test setup: crack must extend past the old 300-iteration cap, got length ${lastT}`);
  cv.rectangle(src, new cv.Point(5, 90), new cv.Point(40, 92), new cv.Scalar(255), -1); // isolated thin line, not touching the blob

  const kernel5 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
  const plainOpen = new cv.Mat();
  cv.morphologyEx(src, plainOpen, cv.MORPH_OPEN, kernel5, new cv.Point(-1, -1), 1);
  kernel5.delete();

  const reconstructed = OPS.morphology.apply(cv, src, { op: 'reconstruct_open', shape: 'rect', ksize: 5, iterations: 1 }, {});

  const tipT = 700;
  const tipX = 60 + tipT, tipY = 30 + Math.round(Math.sin(tipT / 15) * 8);
  assert.equal(reconstructed.data[tipY * w + tipX], 255, 'reconstruction must fully recover the far tip of the >300px-long thin crack');
  assert.equal(plainOpen.data[tipY * w + tipX], 0, 'sanity: plain 5x5 Opening really does remove the thin crack here (proves the two ops genuinely differ on this input)');

  assert.equal(reconstructed.data[91 * w + 20], 0, 'an isolated thin line (too thin for a 5x5 marker) must be removed by reconstruction too');
  assert.equal(plainOpen.data[91 * w + 20], 0, 'an isolated thin line must be removed by plain Opening too');

  assert.equal(reconstructed.data[30 * w + 30], 255, 'sanity: the main blob interior survives reconstruction');
  assert.equal(plainOpen.data[30 * w + 30], 255, 'sanity: the main blob interior survives plain Opening too');

  reconstructed.delete();
  plainOpen.delete();
  src.delete();
});

test('groupA-18. Skeleton runs at full resolution (2026-10-21): a 40px-thick bar in a 640x480 image thins to a 1px line, and a 500x300 blob stays fast', () => {
  const bar = new cv.Mat.zeros(480, 640, cv.CV_8UC1);
  cv.rectangle(bar, new cv.Point(40, 220), new cv.Point(600, 259), new cv.Scalar(255), -1);
  const barOut = OPS.morph_skeleton.apply(cv, bar, {}, {});
  let thick = 0;
  for (let y = 0; y < 480; y++) if (barOut.ucharAt(y, 320) > 0) thick++;
  assert.equal(thick, 1, `skeleton line must be 1px thick at full resolution, got ${thick}px`);
  barOut.delete();
  bar.delete();

  const w = 500, h = 300;
  const src = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  cv.rectangle(src, new cv.Point(20, 20), new cv.Point(450, 280), new cv.Scalar(255), -1);
  const ctx = {};
  const started = performance.now();
  const out = OPS.morphology.apply(cv, src, { op: 'skeleton', shape: 'rect', ksize: 5, iterations: 1 }, ctx);
  const elapsed = performance.now() - started;
  assert.match(ctx.info || '', /ที่ 500x300px/);
  assert.equal(out.cols, w);
  assert.equal(out.rows, h);
  assert.ok(elapsed < 500, `expected skeleton on a 500x300 input to stay fast, took ${elapsed}ms`);
  out.delete();
  src.delete();
});

test("groupA-19. JS-vs-Python parity at banker's-rounding tie dimensions (kills a Python round() vs int(x+0.5) mutant): entropy at 640x361 (scale->180.5), skeleton at 480x241 (scale->120.5)", () => {
  // textured (non-binary) input สำหรับ entropy — checkerboard เดิมมีแค่ 2 ระดับสี ทำให้ banker's-rounding
  // ของ downscale ไม่ถูก exercise จริง (ดู texturedGraySrc ด้านบน) — qa วัดแล้วว่า parity จริง = 0 เป๊ะ
  // (max diff 0) ที่ขนาดนี้ ไม่ใช่แค่ <=5 แบบเดิมที่หลวมเกินไปจนมองไม่เห็น mutant
  const entropySrc = texturedGraySrc(640, 361);
  const entropyStats = parityCheck('local_entropy_filter', { windowSize: '5', bins: '16' }, 640, 361, true, entropySrc, { full: true });
  entropySrc.delete();
  assert.equal(entropyStats.max, 0, `local_entropy_filter @ 640x361 (banker's-rounding tie dimension): expected pixel-perfect parity (max diff 0), got max=${entropyStats.max} (${entropyStats.diffCount}/${entropyStats.total} px differ, mean=${entropyStats.mean})`);

  // skeleton ทำงานบนภาพ binary อยู่แล้ว (threshold ก่อน thinning) — checkerboard เดิมพอสำหรับเทสต์นี้
  const skeletonDiff = parityCheck('morphology', { op: 'skeleton', shape: 'rect', ksize: 5, iterations: 1 }, 480, 241, true);
  assert.equal(skeletonDiff, 0, `skeleton @ 480x241 (banker's-rounding tie dimension): expected pixel-perfect parity (mean abs diff 0), got ${skeletonDiff}`);
});

test('groupA-20. JS-vs-Python parity: local_std_filter and local_entropy_filter at <=320px and >640px', () => {
  // textured (non-binary) input — ดูเหตุผลเดียวกับ groupA-19: checkerboard 0/255 ไม่พอที่จะ exercise
  // BORDER_REPLICATE ของ entropy window เดิม หรือ truncate ของ std normalize เดิม
  for (const [w, h] of [[200, 150], [700, 500]]) {
    const src = texturedGraySrc(w, h);

    const stdStats = parityCheck('local_std_filter', { windowSize: 9 }, w, h, true, src, { full: true });
    assert.ok(stdStats.max <= 1, `local_std_filter @ ${w}x${h}: expected max abs diff <=1, got ${stdStats.max}`);
    const stdFrac = stdStats.diffCount / stdStats.total;
    assert.ok(stdFrac <= 0.0001, `local_std_filter @ ${w}x${h}: expected <=0.01% differing pixels (float32 boundary noise), got ${stdStats.diffCount}/${stdStats.total} (${(stdFrac * 100).toFixed(4)}%)`);

    const entropyStats = parityCheck('local_entropy_filter', { windowSize: '5', bins: '16' }, w, h, true, src, { full: true });
    assert.equal(entropyStats.max, 0, `local_entropy_filter @ ${w}x${h}: expected pixel-perfect parity (max diff 0), got max=${entropyStats.max} (${entropyStats.diffCount}/${entropyStats.total} px differ, mean=${entropyStats.mean})`);

    src.delete();
  }
});

test('groupA-21. buildCustomStrel guard: a non-finite ksize (corrupted saved project) falls back safely instead of throwing RangeError', () => {
  const src = new cv.Mat.zeros(40, 40, cv.CV_8UC1);
  const blob = src.roi(new cv.Rect(10, 10, 20, 20));
  blob.setTo(new cv.Scalar(255));
  blob.delete();
  assert.doesNotThrow(() => {
    const out = OPS.morphology.apply(cv, src, { op: 'erode', shape: 'diamond', ksize: NaN, iterations: 1 }, {});
    out.delete();
  }, 'a non-finite ksize on a custom-shape strel must not throw RangeError: Invalid array length');
  src.delete();
});

test('groupA-22. throw-path leak hardening: an empty (0x0) Mat input to reconstruct_open and to the generic morphology fallback throws cleanly without leaking the kernel/dst Mats (allocator pointer sanity)', () => {
  // P-14: เดิมเช็ค sentinel pointer ก่อน/หลัง throw หนึ่งครั้ง — ตอนนี้ (1) wrapper-balance: kernel/dst ทุกตัวที่สร้างก่อน throw
  // ต้องถูก .delete() (2) census เทียบ baseline: ทั้งสอง path throw cv::Exception จาก cv.erode ซึ่ง opencv.js รั่ว exception
  // object ~760B/ครั้งเองเสมอ (แก้ในแอปไม่ได้ — ดู _leak-helpers.mjs) จึงยืนยันว่า op "ไม่รั่วเพิ่ม" จากการเรียก cv.erode
  // บน Mat ว่างตรง ๆ ด้วยจำนวนรอบเท่ากัน
  const empty = new cv.Mat();
  const k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
  const rawErodeThrow = () => { const d = new cv.Mat(); try { cv.erode(empty, d, k); assert.fail('raw erode on empty must throw'); } catch (e) { if (e instanceof assert.AssertionError) throw e; } finally { d.delete(); } };
  for (const params of [
    { op: 'reconstruct_open', shape: 'rect', ksize: 5, iterations: 1 },
    { op: 'erode', shape: 'diamond', ksize: 5, iterations: 1 },
  ]) {
    const label = `op=${params.op} on an empty Mat (throw path)`;
    assertWrapperBalance(cv, label, (c) => { OPS.morphology.apply(c, empty, params, {}); }, { expectThrow: true, minCreated: 1 });
    assertNoExtraHeapGrowth(cv, label, () => {
      assert.throws(() => OPS.morphology.apply(cv, empty, params, {}), `expected op=${params.op} on an empty Mat to throw`);
    }, rawErodeThrow, { iters: 50, warmup: 5 });
  }
  k.delete();
  empty.delete();
});


// ===================================================================
// CV-engineer S6 recheck fixes (qa's open questions 2, 3, 4)
// ===================================================================

test('qa-Q2. Skeleton hint tells the learner to try Invert if the result is all-white (skeleton treats white pixels as the object)', () => {
  const opParam = FILTERS.morphology.params.find((p) => p.key === 'op');
  const skeletonOption = opParam.options.find((o) => o.value === 'skeleton');
  assert.match(skeletonOption.hint, /Invert/, 'skeleton hint should mention trying Invert first if the result looks all-white');

  // Empirical proof of the behavior this hint explains: outside the image counts as background (like bwmorph),
  // so a white background is thinned like an object — an all-white input collapses to a thin line, not a blank result.
  const allWhite = new cv.Mat(40, 40, cv.CV_8UC1, new cv.Scalar(255));
  const ctx = {};
  const out = OPS.morphology.apply(cv, allWhite, { op: 'skeleton' }, ctx);
  const n = cv.countNonZero(out);
  assert.ok(n > 0 && n < 40 * 3, `an all-white image should thin to a line (white = object), got ${n} white pixels`);
  allWhite.delete();
  out.delete();
});

test('qa-Q3. Line strel hint explains that diagonal angles can yield fewer pixels than ksize due to grid rounding (matches MATLAB strel(\'line\') behavior, not a bug)', () => {
  const shapeParam = FILTERS.morphology.params.find((p) => p.key === 'shape');
  const lineOption = shapeParam.options.find((o) => o.value === 'line');
  assert.match(lineOption.hint, /45|มุมเฉียง|ปัดพิกัด/, 'line hint should explain the diagonal-angle pixel-count caveat');

  // Empirical proof: at 45 degrees with ksize=5, the discretized line has fewer than 5 distinct pixels
  // (rounding collapses some points onto the same grid cell) -- this is the exact behavior being explained.
  const src = new cv.Mat.zeros(21, 21, cv.CV_8UC1);
  src.ucharPtr(10, 10)[0] = 255;
  const out = OPS.morphology.apply(cv, src, { op: 'dilate', shape: 'line', ksize: 5, lineAngle: 45, iterations: 1 }, {});
  const footprint = cv.countNonZero(out);
  assert.ok(footprint < 5, `expected the 45-degree ksize=5 line footprint to collapse below 5 distinct points due to grid rounding, got ${footprint}`);
  src.delete();
  out.delete();
});

test('qa-Q4. Morphology "shape" param label and the skeleton/reconstruct_open hints clearly state which controls have no effect on them (nested showIf isn\'t supported by the params renderer, so these stay visible-but-inert)', () => {
  const shapeParam = FILTERS.morphology.params.find((p) => p.key === 'shape');
  assert.match(shapeParam.label, /ไม่มีผลกับ Skeleton/, 'the "shape" param label itself should note it has no effect on Skeleton');
  assert.match(shapeParam.label, /Opening by Reconstruction/, 'the "shape" param label itself should note it has no effect on Opening by Reconstruction');

  const opParam = FILTERS.morphology.params.find((p) => p.key === 'op');
  const reconstructOption = opParam.options.find((o) => o.value === 'reconstruct_open');
  assert.match(reconstructOption.hint, /Rect เสมอ/, 'reconstruct_open hint should state it always erodes with a Rect kernel regardless of the chosen shape');

  // Empirical proof: reconstruct_open with shape="diamond" must produce the exact same result as
  // shape="rect" (proving shape truly has zero effect, matching what the hint now claims).
  //
  // The discriminating object here is a small filled diamond (|dx|+|dy| <= R=5, drawn pixel by pixel,
  // well away from the image border): a radius-5 diamond survives erosion by a radius-4 Diamond kernel
  // (erosion of an L1 ball of radius R by an L1 ball of radius r<R leaves a non-empty L1 ball of radius
  // R-r), but a 9x9 Rect kernel (erosion radius 4 in every direction, i.e. an L-infinity ball) needs an
  // 8px margin in the worst-case corner direction to survive, so it erases the R=5 diamond completely.
  // If reconstruct_open actually honored "shape" for its marker erosion, shape="diamond" would leave a
  // non-empty marker inside this diamond object and morphological reconstruction would restore the whole
  // object -- differing from shape="rect", which erases it. A big circle far away is included as a sanity
  // check that survives (and is identical) under both, so the diff below is not trivially satisfied by an
  // empty output.
  const R = 5, cx = 30, cy = 35;
  const src = new cv.Mat.zeros(70, 140, cv.CV_8UC1);
  for (let y = -R; y <= R; y++) {
    for (let x = -R; x <= R; x++) {
      if (Math.abs(x) + Math.abs(y) <= R) src.ucharPtr(cy + y, cx + x)[0] = 255;
    }
  }
  cv.circle(src, new cv.Point(100, 35), 25, new cv.Scalar(255), -1);
  const outRect = OPS.morphology.apply(cv, src, { op: 'reconstruct_open', shape: 'rect', ksize: 9, iterations: 1 }, {});
  const outDiamond = OPS.morphology.apply(cv, src, { op: 'reconstruct_open', shape: 'diamond', ksize: 9, iterations: 1 }, {});
  let diffCount = 0;
  for (let i = 0; i < outRect.data.length; i++) if (outRect.data[i] !== outDiamond.data[i]) diffCount++;
  assert.equal(diffCount, 0, 'reconstruct_open must ignore the "shape" param entirely (always Rect internally), matching the hint');

  // Direct check so the test cannot pass trivially: the small diamond object must be fully gone from the
  // output (its 61 pixels all zero) -- proving reconstruct_open's marker erosion really did use a Rect
  // kernel (which cannot fit inside this object) rather than a Diamond kernel (which could).
  let diamondPixelsRemaining = 0;
  for (let y = -R; y <= R; y++) {
    for (let x = -R; x <= R; x++) {
      if (Math.abs(x) + Math.abs(y) <= R && outRect.ucharPtr(cy + y, cx + x)[0] !== 0) diamondPixelsRemaining++;
    }
  }
  assert.equal(diamondPixelsRemaining, 0, 'the small filled diamond object should be fully erased by reconstruct_open\'s always-Rect marker erosion, regardless of the "shape" param');

  src.delete();
  outRect.delete();
  outDiamond.delete();
});

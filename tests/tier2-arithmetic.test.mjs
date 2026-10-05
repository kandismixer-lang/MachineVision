// Tier 2, 18b — acceptance tests สำหรับ imfuse/mask/template_matching/psnr/ssim (สเปก §C.2)
// ครอบคลุม: correctness (ค่าจริง ไม่ใช่แค่ "ไม่ throw"), no-src2 graceful, size-mismatch matchSize,
// template>search guard (edge case อันตรายที่ยืนยันแล้วว่า cv.matchTemplate ไม่ throw), ctx.values shape,
// malloc-fingerprint leak probe (150-200 รอบ) ทุก path, และ Python codegen parity ผ่าน cv2 5.0.0 จริง
// (imfuse/template_matching/psnr ตามที่สเปกสั่ง — เทคนิค runGeneratedPython2Image/matToRaw เดียวกับ
// tests/tier2-infra.test.mjs's tier2-11a แต่ reimplement ในไฟล์นี้เอง เพราะเป็นไฟล์แยก)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { OPS, FILTERS } from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS } from '../src/renderer/ui/palette-panel.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import { fingerprintLeakProbe, assertWrapperBalance, assertLeakFree, makeThrowingCv } from './_leak-helpers.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cvtColor === 'function'; }
async function waitForCv() {
  if (isReady(cv)) return;
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}
await waitForCv();

let __uid = 1;
function uniqueNodeId() { return 910000 + __uid++; } // ช่วงเลขแยกจาก tier2-infra.test.mjs กันชนกัน

// ===================================================================
// ภาพทดสอบสังเคราะห์
// ===================================================================
function solidRGB(w, h, r, g, b) { return new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(r, g, b)); }
function solidGray(w, h, v) { return new cv.Mat(h, w, cv.CV_8UC1, new cv.Scalar(v)); }
function withRectGray(w, h, bg, rect, fg) {
  const m = solidGray(w, h, bg);
  cv.rectangle(m, new cv.Point(rect.x, rect.y), new cv.Point(rect.x + rect.w, rect.y + rect.h), new cv.Scalar(fg), -1);
  return m;
}
// ภาพ "มีลวดลาย" (checkerboard เล็ก) — ต้องใช้แทนภาพสีเดียวล้วนสำหรับทดสอบ template_matching เพราะ
// TM_CCOEFF_NORMED หารด้วย variance ของ template — variance=0 (สีเดียวล้วน) ทำให้ normalize เพี้ยน
// (ยืนยันเชิงประจักษ์ด้วยสมอกเทสต์ก่อนเขียนไฟล์นี้: flat template ได้คะแนน 100% ทุกตำแหน่ง ไม่ใช่บั๊ก
// เป็นคุณสมบัติทางคณิตศาสตร์ของวิธีนี้เอง ตรงกับที่ CV eng เตือนไว้ในสเปก)
function texturedPatch(w, h, cell) {
  const m = new cv.Mat(h, w, cv.CV_8UC1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    m.ucharPtr(y, x)[0] = ((Math.floor(x / cell) + Math.floor(y / cell)) % 2 === 0) ? 220 : 30;
  }
  return m;
}
function pasteInto(dst, src, x0, y0) {
  for (let y = 0; y < src.rows; y++) for (let x = 0; x < src.cols; x++) {
    dst.ucharPtr(y0 + y, x0 + x)[0] = src.ucharPtr(y, x)[0];
  }
}

// ===================================================================
// leak-test (P-14 — นโยบายเต็มใน tests/_leak-helpers.mjs): fingerprint แบบเทียบที่อยู่ใช้ได้เฉพาะ tier2b-0 (เทสต์แรกของไฟล์)
// ด่านหลักของทุก path คือ wrapper-balance (tier2b-0w, tier2b-0b) + heap census (tier2b-0b)
// makeThrowingCv(cv, name) / makeThrowingCv(cv, name, { onCall: n }) มาจาก helper กลาง (ข้อความ error เหมือนตัวเดิมในไฟล์นี้เป๊ะ:
// `injected test failure (<name>)` / `injected test failure (<name> call #n)`) — onCall ใช้ทดสอบ error-path cleanup ของ
// template_matching's ROI re-score (Option B) ที่เรียก cv.matchTemplate 2 ครั้ง: throw เฉพาะครั้งที่ 2 เพื่อพิสูจน์ว่า
// roi/resultFull (Mat ที่ alloc ไปแล้วก่อนถึงบรรทัดที่ throw) ถูก delete() ใน finally จริง
// ===================================================================
const leakProbe = (label, run, iters = 180) => fingerprintLeakProbe(cv, label, run, { iters, warmup: 30 });

// ===================================================================
// tier2b-0 (malloc-fingerprint) — วางไว้ต้นไฟล์โดยตั้งใจ ก่อนเทสต์อื่นใดที่ทำงานกับ cv หนัก ๆ
// (correctness/Python-subprocess tests ด้านล่าง) เหตุผลเดียวกับที่ groupA-ops.test.mjs's groupA-0 และ
// keypoints-ops.test.mjs's keypoints-0 ทำไว้ก่อนแล้ว (ดูคอมเมนต์เต็มใน groupA-ops.test.mjs): allocator
// ของ opencv.js (dlmalloc-style free-list) จะทิ้ง bucket ขนาดเล็กบางขนาดไว้ในสถานะ "สลับที่กันไปมา" ระหว่าง
// สอง snapshot ได้ถ้ามีเทสต์อื่นอีกหลายสิบตัวที่ใช้ cv หนัก ๆ รันไปก่อนแล้วในโปรเซสเดียวกัน แม้
// cv.HEAP8.buffer.byteLength จะไม่โตเลยสักครั้ง (ไม่ใช่ leak จริง) — ยืนยันเจอเองตอนพัฒนาไฟล์นี้: ย้าย
// leak probe ของ psnr/ssim มาไว้ตรงนี้ (จากเดิมที่แทรกอยู่หลัง Python-subprocess parity tests) ทำให้
// deterministic-fail กลายเป็น deterministic-pass ทันที โดยไม่ต้องแก้โค้ด op เลย (พิสูจน์ด้วยการรันเดี่ยว ๆ
// ด้วย --test-name-pattern ก็ผ่านเสมอ แต่รันรวมทั้งไฟล์ (ก่อนย้าย) fail เสมอ — ตรงกับ pattern ที่
// groupA-ops.test.mjs อธิบายไว้เป๊ะ ไม่ใช่ปรากฏการณ์ใหม่)
// ตาราง path ของ tier2b-0 — ใช้ร่วมกันระหว่าง tier2b-0 (fingerprint, process สะอาด) และ tier2b-0w (wrapper-balance) (P-14)
// [label, run(c), iters ของ fingerprint, throws] — throws=true: path ที่ฉีด error (fingerprint กลืน error, tracker ต้องเห็น throw จริง)
function tier2b0Fixtures() {
  const a = solidRGB(24, 20, 120, 130, 140);
  const b = solidRGB(24, 20, 10, 20, 30);
  const bSmall = solidRGB(10, 8, 5, 5, 5);
  const bMask = withRectGray(24, 20, 40, { x: 4, y: 4, w: 10, h: 8 }, 200);
  const bMaskSmall = solidGray(10, 8, 200);
  const tmSearch = solidGray(60, 50, 60);
  const tmPatch = texturedPatch(12, 12, 3);
  pasteInto(tmSearch, tmPatch, 20, 15);
  const tmGood = new cv.Mat(); tmPatch.copyTo(tmGood);
  const tmBig = solidGray(200, 200, 100);
  const tmRandom = new cv.Mat(12, 12, cv.CV_8UC1);
  for (let y = 0; y < 12; y++) for (let x = 0; x < 12; x++) tmRandom.ucharPtr(y, x)[0] = (x * 53 + y * 17) % 256;
  const gA = solidGray(24, 20, 120);
  const gBSame = new cv.Mat(); gA.copyTo(gBSame);
  const gBDiff = solidGray(24, 20, 90);
  const gBSmall = solidGray(10, 8, 90);
  const sA = solidGray(240, 200, 100);
  const sBSame = new cv.Mat(); sA.copyTo(sBSame);
  const sBDiff = solidGray(240, 200, 180);
  const sBSmall = solidGray(50, 40, 180);
  const all = [a, b, bSmall, bMask, bMaskSmall, tmSearch, tmPatch, tmGood, tmBig, tmRandom, gA, gBSame, gBDiff, gBSmall, sA, sBSame, sBDiff, sBSmall];
  return { a, b, bSmall, bMask, bMaskSmall, tmSearch, tmGood, tmBig, tmRandom, gA, gBSame, gBDiff, gBSmall, sA, sBSame, sBDiff, sBSmall, dispose() { for (const m of all) m.delete(); } };
}
function tier2b0Paths({ a, b, bSmall, bMask, bMaskSmall, tmSearch, tmGood, tmBig, tmRandom, gA, gBSame, gBDiff, gBSmall, sA, sBSame, sBDiff, sBSmall }) {
  return [
    // ---- imfuse ----
    ['imfuse no-src2', (c) => { const out = OPS.imfuse.apply(c, a, { mode: 'blend', alpha: 40 }, {}); out.delete(); }, 180],
    ['imfuse blend normal', (c) => { const out = OPS.imfuse.apply(c, a, { mode: 'blend', alpha: 40 }, { src2: b }); out.delete(); }, 180],
    ['imfuse checkerboard normal', (c) => { const out = OPS.imfuse.apply(c, a, { mode: 'checkerboard', tileSize: 8 }, { src2: b }); out.delete(); }, 180],
    ['imfuse mismatched-size', (c) => { const out = OPS.imfuse.apply(c, a, { mode: 'blend', alpha: 40 }, { src2: bSmall }); out.delete(); }, 180],
    ['imfuse thrown-error (blend)', (c) => { const out = OPS.imfuse.apply(makeThrowingCv(c, 'addWeighted'), a, { mode: 'blend', alpha: 40 }, { src2: b }); out.delete(); }, 150, true],
    ['imfuse thrown-error (checkerboard)', (c) => { const out = OPS.imfuse.apply(makeThrowingCv(c, 'add'), a, { mode: 'checkerboard', tileSize: 8 }, { src2: b }); out.delete(); }, 150, true],
    // ---- mask ----
    ['mask no-src2', (c) => { const out = OPS.mask.apply(c, a, { invert: false }, {}); out.delete(); }, 180],
    ['mask normal', (c) => { const out = OPS.mask.apply(c, a, { invert: false }, { src2: bMask }); out.delete(); }, 180],
    ['mask invert', (c) => { const out = OPS.mask.apply(c, a, { invert: true }, { src2: bMask }); out.delete(); }, 180],
    ['mask mismatched-size', (c) => { const out = OPS.mask.apply(c, a, { invert: false }, { src2: bMaskSmall }); out.delete(); }, 180],
    ['mask thrown-error', (c) => { const out = OPS.mask.apply(makeThrowingCv(c, 'bitwise_and'), a, { invert: false }, { src2: bMask }); out.delete(); }, 150, true],
    // ---- template_matching ----
    ['template_matching no-src2', (c) => { const out = OPS.template_matching.apply(c, tmSearch, { method: 'ccoeff', threshold: 70 }, {}); out.delete(); }, 180],
    ['template_matching found', (c) => { const out = OPS.template_matching.apply(c, tmSearch, { method: 'ccoeff', threshold: 50 }, { src2: tmGood }); out.delete(); }, 180],
    ['template_matching not-found', (c) => { const out = OPS.template_matching.apply(c, tmSearch, { method: 'ccoeff', threshold: 99.9 }, { src2: tmRandom }); out.delete(); }, 180],
    ['template_matching template-too-big', (c) => { const out = OPS.template_matching.apply(c, tmSearch, { method: 'ccoeff', threshold: 70 }, { src2: tmBig }); out.delete(); }, 180],
    ['template_matching thrown-error', (c) => { const out = OPS.template_matching.apply(makeThrowingCv(c, 'matchTemplate'), tmSearch, { method: 'ccoeff', threshold: 50 }, { src2: tmGood }); out.delete(); }, 150, true],
    // ---- psnr ----
    ['psnr no-src2', (c) => { const out = OPS.psnr.apply(c, gA, {}, {}); out.delete(); }, 180],
    ['psnr normal', (c) => { const out = OPS.psnr.apply(c, gA, {}, { src2: gBDiff }); out.delete(); }, 180],
    ['psnr identical (Infinity path)', (c) => { const out = OPS.psnr.apply(c, gA, {}, { src2: gBSame }); out.delete(); }, 180],
    ['psnr mismatched-size', (c) => { const out = OPS.psnr.apply(c, gA, {}, { src2: gBSmall }); out.delete(); }, 180],
    ['psnr thrown-error', (c) => { const out = OPS.psnr.apply(makeThrowingCv(c, 'absdiff'), gA, {}, { src2: gBDiff }); out.delete(); }, 150, true],
    // ---- ssim ----
    ['ssim no-src2', (c) => { const out = OPS.ssim.apply(c, sA, {}, {}); out.delete(); }, 150],
    ['ssim normal', (c) => { const out = OPS.ssim.apply(c, sA, {}, { src2: sBDiff }); out.delete(); }, 150],
    ['ssim identical', (c) => { const out = OPS.ssim.apply(c, sA, {}, { src2: sBSame }); out.delete(); }, 150],
    ['ssim mismatched-size', (c) => { const out = OPS.ssim.apply(c, sA, {}, { src2: sBSmall }); out.delete(); }, 150],
    ['ssim thrown-error', (c) => { const out = OPS.ssim.apply(makeThrowingCv(c, 'GaussianBlur'), sA, {}, { src2: sBDiff }); out.delete(); }, 150, true],
  ];
}

test('tier2b-0 (malloc-fingerprint). imfuse/mask/template_matching/psnr/ssim leave the allocator fingerprint unchanged across normal, no-src2, mismatched-size (or template-too-big), and thrown-error paths', () => {
  const fx = tier2b0Fixtures();
  for (const [label, run, iters, throws] of tier2b0Paths(fx)) {
    leakProbe(label, throws ? () => { try { run(cv); } catch { /* expected */ } } : () => run(cv), iters);
  }
  fx.dispose();
});

// tier2b-0w — ด่านหลัก (wrapper-balance) ของ path ชุดเดียวกับ tier2b-0 (P-14): นับทุก wrapper ที่ op สร้าง (รวม Mat ที่ matchSize
// resize, roi, ผลกลาง ssim) เทียบ .delete() — path ที่ฉีด error ต้อง throw จริง และเก็บกวาดครบ
test('tier2b-0w (wrapper-balance companion of tier2b-0). every imfuse/mask/template_matching/psnr/ssim path of tier2b-0 (normal, no-src2, mismatched-size / template-too-big, thrown-error) deletes every embind wrapper it creates', () => {
  const fx = tier2b0Fixtures();
  try {
    for (const [label, run, , throws] of tier2b0Paths(fx)) {
      assertWrapperBalance(cv, label, run, { expectThrow: throws ? /injected test failure/ : false, minCreated: 1 });
    }
  } finally { fx.dispose(); }
});

// ===================================================================
// tier2b-0b (malloc-fingerprint, branch ใหม่รอบ scope-change ที่ 2) — ssim's speed param (fast ย่อจริง,
// full/ไม่มี key ไม่ย่อ) + template_matching's Option B (downscaled coarse search + full-res ROI re-score
// รวม edge-clamped ROI และ error-path กลาง ROI re-score)
// ใช้ภาพ "ผอมยาว" 660x60 โดยตั้งใจ: branch ใหม่ต้องการแค่ด้านยาว >640 (analysisScale ได้ scale<1) ไม่ต้อง
// ใช้พื้นที่ภาพเยอะ — ยืนยันเชิงประจักษ์ระหว่างพัฒนาว่าภาพ 700x650 ทำให้ fingerprint() เอง "ไม่ idempotent"
// (เรียก fingerprint() 2 ครั้งติดกันโดยไม่มี op ใด ๆ คั่นกลางยังได้ผลต่างกัน — allocator coalesce chunk ใหญ่
// ไม่ลงรอยเดิม) จนเทียบไม่ได้ไม่ว่าจะ warmup กี่รอบ (ทดลองถึง 1000) — ภาพผอมยาวใช้ buffer เล็ก fingerprint นิ่ง
// ทุก probe (ทดลองซ้ำ 2 รอบติด) จึงใช้ leakProbe เข้มงวดตัวเดิมได้ ไม่ต้องลดเหลือแค่เช็ค heap byteLength
// (ซึ่งจับ leak เล็ก ๆ เช่น roi header ที่ลืม delete ไม่ได้ เพราะ byteLength โตเฉพาะตอน WASM memory ขยาย)
test('tier2b-0b (wrapper-balance + heap census). ssim speed=fast/full/no-key and template_matching Option B (downscaled+ROI re-score: found, not-found, edge-clamped ROI, thrown-error mid-rescore) delete every embind wrapper and do not grow the WASM heap (thin 660x60 images so the >640px branches fire)', () => {
  const W = 660, H = 60; // ด้านยาว >640 -> scale<1 ทั้ง ssim(fast) และ template_matching
  const ssimA = new cv.Mat(H, W, cv.CV_8UC1);
  for (let i = 0; i < W * H; i++) ssimA.data[i] = (i * 37) % 256;
  const ssimB = new cv.Mat(H, W, cv.CV_8UC1);
  for (let i = 0; i < W * H; i++) ssimB.data[i] = (i * 41 + 30) % 256;

  const tmS = solidGray(W, H, 60);
  const tmP = texturedPatch(24, 24, 4);
  pasteInto(tmS, tmP, 300, 18); // กลางภาพ ไม่ชิดขอบ
  const tmT = new cv.Mat(); tmP.copyTo(tmT);
  const tmRnd = new cv.Mat(24, 24, cv.CV_8UC1);
  for (let i = 0; i < 24 * 24; i++) tmRnd.data[i] = (i * 53 + 17) % 256;
  // template ชิดมุม (0,0) -> ROI's roiX0/roiY0 ถูก clamp ที่ 0 จริง
  const edgeS = solidGray(W, H, 60);
  const edgeP = texturedPatch(24, 24, 4);
  pasteInto(edgeS, edgeP, 0, 0);
  const edgeT = new cv.Mat(); edgeP.copyTo(edgeT);

  // sanity: branch ใหม่ทำงานจริงบน fixture เหล่านี้ (ไม่ใช่ probe ที่วิ่งผ่าน path เก่าโดยไม่รู้ตัว)
  const c1 = { src2: tmT };
  OPS.template_matching.apply(cv, tmS, { method: 'ccoeff', threshold: 50 }, c1).delete();
  assert.ok(c1.values.fields[0].value > 99.9, `ROI re-score should give exact-match ~100%, got ${c1.values.fields[0].value}`);
  const c2 = { src2: ssimB };
  OPS.ssim.apply(cv, ssimA, { speed: 'fast' }, c2).delete();
  assert.match(c2.info, /ย่อขนาดลงก่อนเพื่อความเร็ว/, 'fast mode on a >640px image must actually downscale');

  // P-14: ไม่ใช่เทสต์แรกของไฟล์แล้วจึงห้ามใช้ fingerprint (หลัง tier2b-0 free-list ของ allocator หมุนแล้ว) — ใช้ assertLeakFree:
  // wrapper-balance (ด่านหลัก, นับ roi/resultFull/ผลย่อของ ssim ด้วย) + heap census 100 รอบ
  const probe = (label, run, opts = {}) => assertLeakFree(cv, label, run, { minCreated: 2, iters: 100, warmup: 10, ...opts });
  probe('template_matching found (downscaled + ROI re-score)', (c) => {
    const out = OPS.template_matching.apply(c, tmS, { method: 'ccoeff', threshold: 50 }, { src2: tmT }); out.delete();
  });
  probe('template_matching not-found (downscaled + ROI re-score)', (c) => {
    const out = OPS.template_matching.apply(c, tmS, { method: 'ccoeff', threshold: 99.9 }, { src2: tmRnd }); out.delete();
  });
  probe('template_matching found, edge-clamped ROI (template at (0,0))', (c) => {
    const out = OPS.template_matching.apply(c, edgeS, { method: 'ccoeff', threshold: 50 }, { src2: edgeT }); out.delete();
  });
  // throw เฉพาะ cv.matchTemplate ครั้งที่ 2 (อยู่ใน ROI re-score หลัง roi/resultFull ถูก alloc แล้ว) —
  // พิสูจน์ finally cleanup ของ roi/resultFull เอง ไม่ใช่แค่ cleanup ของรอบ coarse search (proxy ใหม่ทุกครั้ง: ตัวนับเริ่มใหม่)
  probe('template_matching thrown-error mid ROI re-score (2nd matchTemplate call)', (c) => {
    const out = OPS.template_matching.apply(makeThrowingCv(c, 'matchTemplate', { onCall: 2 }), tmS, { method: 'ccoeff', threshold: 50 }, { src2: tmT }); out.delete();
  }, { expectThrow: /injected test failure \(matchTemplate call #2\)/ });

  probe('ssim speed=fast (downscale branch fires)', (c) => {
    const out = OPS.ssim.apply(c, ssimA, { speed: 'fast' }, { src2: ssimB }); out.delete();
  });
  probe('ssim speed=full (no downscale even though >640px)', (c) => {
    const out = OPS.ssim.apply(c, ssimA, { speed: 'full' }, { src2: ssimB }); out.delete();
  });
  probe('ssim no speed key (backward-compat -> full)', (c) => {
    const out = OPS.ssim.apply(c, ssimA, {}, { src2: ssimB }); out.delete();
  });
  probe('ssim speed=fast thrown-error (after downscale resize)', (c) => {
    const out = OPS.ssim.apply(makeThrowingCv(c, 'GaussianBlur'), ssimA, { speed: 'fast' }, { src2: ssimB }); out.delete();
  }, { expectThrow: /injected test failure \(GaussianBlur\)/ });

  ssimA.delete(); ssimB.delete();
  tmS.delete(); tmP.delete(); tmT.delete(); tmRnd.delete();
  edgeS.delete(); edgeP.delete(); edgeT.delete();
});

// ===================================================================
// Python codegen parity harness (reimplemented ในไฟล์นี้ตามที่สเปกอนุญาต — เทคนิคเดียวกับ
// tests/tier2-infra.test.mjs's runGeneratedPython2Image/matToRaw เป๊ะ)
// ===================================================================
function matToRaw(mat) {
  return { data: Buffer.from(mat.data), width: mat.cols, height: mat.rows, channels: mat.channels() };
}
let __pyAvailable = null;
function pythonAvailable() {
  if (__pyAvailable !== null) return __pyAvailable;
  try {
    execFileSync('python', ['-c', 'import cv2, numpy'], { encoding: 'utf8' });
    __pyAvailable = true;
  } catch {
    __pyAvailable = false;
  }
  return __pyAvailable;
}
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'tier2b-parity-'));
process.on('exit', () => {
  try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ }
});
// PYTHONIOENCODING=utf-8: บังคับให้ stdout ของ python เป็น UTF-8 เสมอ ไม่พึ่ง codepage เริ่มต้นของ
// console บน Windows (cp874/cp1252 ฯลฯ) — ถ้าไม่ตั้งค่านี้ ข้อความไทยที่ print ออกมา (เช่น "พบตำแหน่งที่")
// จะกลายเป็นตัวอักษรเพี้ยน (mojibake) ตอน Node decode ด้วย 'utf8' ทำให้ regex ที่ match ข้อความไทยไม่เจอ —
// พบปัญหานี้จริงระหว่างพัฒนาเทสต์นี้ (tier2b-tm-7) แก้แล้วด้วยบรรทัดนี้
function runGeneratedPython2Image(code, rawA, rawB) {
  const shapeA = `(${rawA.height}, ${rawA.width}, ${rawA.channels})`;
  const shapeB = `(${rawB.height}, ${rawB.width}, ${rawB.channels})`;
  assert.ok(code.includes('img = cv2.imread(path_a)'), 'expected the two-image process(path_a, path_b) loader line for A');
  assert.ok(code.includes('img_b = cv2.imread(path_b)'), 'expected the two-image process(path_a, path_b) loader line for B');
  let patched = code
    .replace('img = cv2.imread(path_a)', `img = np.fromfile(path_a, dtype=np.uint8).reshape(${shapeA})`)
    .replace('img_b = cv2.imread(path_b)', `img_b = np.fromfile(path_b, dtype=np.uint8).reshape(${shapeB})`);
  const mainIdx = patched.indexOf('if __name__ == "__main__":');
  assert.ok(mainIdx > 0, 'generated code must contain a __main__ guard to strip');
  patched = patched.slice(0, mainIdx);
  patched += `
import sys
result = process(sys.argv[1], sys.argv[2])
result.tofile(sys.argv[3])
`;
  const id = Math.random().toString(36).slice(2);
  const scriptPath = path.join(TMP_DIR, `script_${id}.py`);
  const aPath = path.join(TMP_DIR, `a_${id}.bin`);
  const bPath = path.join(TMP_DIR, `b_${id}.bin`);
  const outPath = path.join(TMP_DIR, `out_${id}.bin`);
  fs.writeFileSync(scriptPath, patched, 'utf8');
  fs.writeFileSync(aPath, rawA.data);
  fs.writeFileSync(bPath, rawB.data);
  const stdout = execFileSync('python', [scriptPath, aPath, bPath, outPath], {
    encoding: 'utf8',
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });
  return { out: new Uint8Array(fs.readFileSync(outPath)), stdout };
}

// ===================================================================
// imfuse
// ===================================================================
test('tier2b-imfuse-1. no src2 connected -> graceful info message, no throw', () => {
  const src = solidGray(20, 16, 77);
  const ctx = {};
  let out;
  assert.doesNotThrow(() => { out = OPS.imfuse.apply(cv, src, { mode: 'blend', alpha: 50 }, ctx); });
  assert.match(ctx.info, /ยังไม่ได้ต่อสาย 'เข้า B'/);
  assert.equal(out.cols, 20);
  assert.equal(out.rows, 16);
  out.delete();
  src.delete();
});

test('tier2b-imfuse-2. size mismatch -> matchSize resize path fires, ctx.info mentions it, output size = A', () => {
  const a = solidRGB(50, 40, 100, 100, 100);
  const b = solidRGB(20, 15, 50, 50, 50);
  const ctx = { src2: b };
  const out = OPS.imfuse.apply(cv, a, { mode: 'blend', alpha: 50 }, ctx);
  assert.equal(out.cols, 50);
  assert.equal(out.rows, 40);
  assert.match(ctx.info, /ปรับขนาด B ให้เท่า A/);
  out.delete(); a.delete(); b.delete();
});

test('tier2b-imfuse-3. blend mode correctness: known alpha -> exact expected pixel value (hand-computed)', () => {
  const a = solidGray(20, 20, 200);
  const b = solidGray(20, 20, 0);
  for (const alpha of [0, 25, 50, 75, 100]) {
    const ctx = { src2: b };
    const out = OPS.imfuse.apply(cv, a, { mode: 'blend', alpha }, ctx);
    const expected = Math.round(200 * (alpha / 100)); // cv.addWeighted ปัดเศษแบบ round-to-nearest
    const actual = out.ucharPtr(10, 10)[0];
    assert.ok(Math.abs(actual - expected) <= 1, `alpha=${alpha}: expected ~${expected}, got ${actual}`);
    out.delete();
  }
  a.delete(); b.delete();
});

test('tier2b-imfuse-4. checkerboard mode correctness: known tile pattern -> A/B selection matches exactly, per-pixel', () => {
  const w = 32, h = 32, tile = 8;
  const a = solidGray(w, h, 255);
  const b = solidGray(w, h, 0);
  const ctx = { src2: b };
  const out = OPS.imfuse.apply(cv, a, { mode: 'checkerboard', tileSize: tile }, ctx);
  for (let y = 0; y < h; y += 3) {
    for (let x = 0; x < w; x += 3) {
      const tileX = Math.floor(x / tile), tileY = Math.floor(y / tile);
      const expectA = (tileX + tileY) % 2 === 0;
      assert.equal(out.ucharPtr(y, x)[0], expectA ? 255 : 0, `pixel (${x},${y}) tile(${tileX},${tileY})`);
    }
  }
  out.delete(); a.delete(); b.delete();
});

test('tier2b-imfuse-5. Python codegen (blend): buildPythonCode emits process(path_a, path_b), executed against real cv2 matches JS addWeighted output', { skip: !pythonAvailable() && 'python/cv2 not available in this environment' }, () => {
  const w = 28, h = 22;
  const matA = solidRGB(w, h, 180, 90, 30);
  const matB = solidRGB(w, h, 20, 200, 100);

  const pipeline = [{ id: 'image_arithmetic', params: { mode: 'imfuse', imfuse__mode: 'blend', imfuse__alpha: 35 }, enabled: true, nodeId: uniqueNodeId(), src2: { pipeline: [] } }];
  const code = buildPythonCode(pipeline);
  assert.match(code, /def process\(path_a, path_b\):/);

  const jsOut = OPS.imfuse.apply(cv, matA, { mode: 'blend', alpha: 35 }, { src2: matB });
  const { out: pyOut } = runGeneratedPython2Image(code, matToRaw(matA), matToRaw(matB));
  assert.equal(pyOut.length, jsOut.data.length);
  for (let i = 0; i < jsOut.data.length; i++) {
    assert.ok(Math.abs(pyOut[i] - jsOut.data[i]) <= 1, `byte ${i}: JS=${jsOut.data[i]} Python=${pyOut[i]}`);
  }
  jsOut.delete(); matA.delete(); matB.delete();
});

test('tier2b-imfuse-6. Python codegen (checkerboard): np.where selection matches JS bitwise_and-based selection exactly, byte-for-byte', { skip: !pythonAvailable() && 'python/cv2 not available in this environment' }, () => {
  const w = 24, h = 24;
  const matA = solidRGB(w, h, 255, 255, 255);
  const matB = solidRGB(w, h, 0, 0, 0);

  const pipeline = [{ id: 'image_arithmetic', params: { mode: 'imfuse', imfuse__mode: 'checkerboard', imfuse__tileSize: 8 }, enabled: true, nodeId: uniqueNodeId(), src2: { pipeline: [] } }];
  const code = buildPythonCode(pipeline);

  const jsOut = OPS.imfuse.apply(cv, matA, { mode: 'checkerboard', tileSize: 8 }, { src2: matB });
  const { out: pyOut } = runGeneratedPython2Image(code, matToRaw(matA), matToRaw(matB));
  assert.equal(pyOut.length, jsOut.data.length);
  for (let i = 0; i < jsOut.data.length; i++) {
    assert.equal(pyOut[i], jsOut.data[i], `byte ${i} mismatch (checkerboard is exact selection, no rounding at all)`);
  }
  jsOut.delete(); matA.delete(); matB.delete();
});

// ===================================================================
// mask
// ===================================================================
test('tier2b-mask-1. no src2 connected -> graceful info message, no throw', () => {
  const src = solidRGB(16, 12, 50, 60, 70);
  const ctx = {};
  let out;
  assert.doesNotThrow(() => { out = OPS.mask.apply(cv, src, { invert: false }, ctx); });
  assert.match(ctx.info, /ยังไม่ได้ต่อสาย 'เข้า B'/);
  out.delete(); src.delete();
});

test('tier2b-mask-2. size mismatch -> matchSize resize path fires, ctx.info mentions it', () => {
  const a = solidRGB(40, 30, 100, 100, 100);
  const bMask = solidGray(10, 8, 255);
  const ctx = { src2: bMask };
  const out = OPS.mask.apply(cv, a, { invert: false }, ctx);
  assert.equal(out.cols, 40);
  assert.equal(out.rows, 30);
  assert.match(ctx.info, /ปรับขนาด B ให้เท่า A/);
  out.delete(); a.delete(); bMask.delete();
});

test('tier2b-mask-3. correctness: B>127 keeps A, B<127 zeroes it out, invert flips the selection', () => {
  const w = 20, h = 20;
  const a = solidRGB(w, h, 200, 150, 100);
  const rect = { x: 5, y: 5, w: 8, h: 8 };
  const bMask = withRectGray(w, h, 50, rect, 200); // นอก rect=50(<127->off), ใน rect=200(>127->on)
  const ctx = { src2: bMask };
  const out = OPS.mask.apply(cv, a, { invert: false }, ctx);
  // นอก rect ต้องเป็น 0 ทุกช่อง
  assert.equal(out.data[(2 * w + 2) * 3], 0);
  // ใน rect ต้องคง A ไว้เป๊ะ
  const idx = ((rect.y + 2) * w + (rect.x + 2)) * 3;
  assert.equal(out.data[idx], 200);
  assert.equal(out.data[idx + 1], 150);
  assert.equal(out.data[idx + 2], 100);
  out.delete();

  // invert: สลับกัน
  const out2 = OPS.mask.apply(cv, a, { invert: true }, { src2: bMask });
  assert.equal(out2.data[(2 * w + 2) * 3], 200); // นอก rect ตอนนี้ "เปิด"
  const idx2 = ((rect.y + 2) * w + (rect.x + 2)) * 3;
  assert.equal(out2.data[idx2], 0); // ใน rect ตอนนี้ "ปิด"
  out2.delete();

  a.delete(); bMask.delete();
});

// ===================================================================
// template_matching
// ===================================================================
test('tier2b-tm-1. no src2 connected -> graceful info message, no throw', () => {
  const src = solidGray(40, 30, 60);
  const ctx = {};
  let out;
  assert.doesNotThrow(() => { out = OPS.template_matching.apply(cv, src, { method: 'ccoeff', threshold: 70 }, ctx); });
  assert.match(ctx.info, /ยังไม่ได้ต่อสาย 'เข้า B'/);
  out.delete(); src.delete();
});

test('tier2b-tm-2. planted-template correctness: finds the exact planted location (textured template, not flat)', () => {
  const search = solidGray(200, 150, 60);
  const patch = texturedPatch(30, 30, 5);
  pasteInto(search, patch, 90, 55);
  const tmpl = new cv.Mat(); patch.copyTo(tmpl);
  const ctx = { src2: tmpl };
  const out = OPS.template_matching.apply(cv, search, { method: 'ccoeff', threshold: 50 }, ctx);
  assert.equal(ctx.values.kind, 'scalars');
  const byKey = Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));
  assert.ok(byKey.match_score >= 95, `expected near-100% match score, got ${byKey.match_score}`);
  assert.ok(Math.abs(byKey.x - 90) <= 3, `expected x near 90, got ${byKey.x}`);
  assert.ok(Math.abs(byKey.y - 55) <= 3, `expected y near 55, got ${byKey.y}`);
  assert.match(ctx.info, /พบตำแหน่งที่/);
  out.delete(); search.delete(); patch.delete(); tmpl.delete();
});

test('tier2b-tm-3. template-larger-than-search guard: does not throw, does not draw a garbage box, graceful info', () => {
  const search = solidGray(20, 20, 100);
  const tmpl = solidGray(50, 50, 100);
  const ctx = { src2: tmpl };
  let out;
  assert.doesNotThrow(() => { out = OPS.template_matching.apply(cv, search, { method: 'ccoeff', threshold: 70 }, ctx); });
  assert.match(ctx.info, /เล็กกว่าภาพหลัก/);
  assert.equal(ctx.values, undefined, 'no ctx.values should be set when the guard rejects the input');
  // output ต้องเป็นภาพเดิม (RGB ของ search แปลงแล้ว) ไม่ใช่ผลลัพธ์ 31x31 เพี้ยน ๆ จาก matchTemplate ที่ไม่ throw
  assert.equal(out.cols, 20);
  assert.equal(out.rows, 20);
  out.delete(); search.delete(); tmpl.delete();
});

test('tier2b-tm-4. below-threshold -> "not found", no box drawn, ctx.values.x/y are null', () => {
  const search = solidGray(120, 100, 60);
  const patch = texturedPatch(20, 20, 4);
  pasteInto(search, patch, 50, 40);
  // template แบบสุ่มไม่เกี่ยวกับ search เลย -> คะแนนต้องต่ำ
  const tmpl = new cv.Mat(20, 20, cv.CV_8UC1);
  for (let y = 0; y < 20; y++) for (let x = 0; x < 20; x++) tmpl.ucharPtr(y, x)[0] = (x * 37 + y * 91) % 256;
  const ctx = { src2: tmpl };
  const out = OPS.template_matching.apply(cv, search, { method: 'ccoeff', threshold: 99.9 }, ctx);
  assert.match(ctx.info, /ไม่พบ/);
  assert.equal(ctx.values.kind, 'scalars');
  const byKey = Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));
  assert.equal(byKey.x, null);
  assert.equal(byKey.y, null);
  out.delete(); search.delete(); patch.delete(); tmpl.delete();
});

test('tier2b-tm-5. ctx.values shape validation: kind="scalars", fields=[match_score,x,y] with key/label/value', () => {
  const search = solidGray(80, 60, 60);
  const patch = texturedPatch(16, 16, 4);
  pasteInto(search, patch, 30, 20);
  const tmpl = new cv.Mat(); patch.copyTo(tmpl);
  const ctx = { src2: tmpl };
  const out = OPS.template_matching.apply(cv, search, { method: 'ccoeff', threshold: 50 }, ctx);
  assert.equal(ctx.values.kind, 'scalars');
  assert.equal(ctx.values.fields.length, 3);
  const keys = ctx.values.fields.map((f) => f.key);
  assert.deepEqual(keys, ['match_score', 'x', 'y']);
  for (const f of ctx.values.fields) {
    assert.equal(typeof f.key, 'string');
    assert.equal(typeof f.label, 'string');
    assert.ok(typeof f.value === 'number', `${f.key}.value should be a number in the found case`);
  }
  out.delete(); search.delete(); patch.delete(); tmpl.delete();
});

test('tier2b-tm-6. Python codegen: buildPythonCode matchTemplate score/x/y match JS within downscale-rounding tolerance (real cv2)', { skip: !pythonAvailable() && 'python/cv2 not available in this environment' }, () => {
  const w = 300, h = 240;
  const searchGray = solidGray(w, h, 60);
  const patch = texturedPatch(40, 40, 5);
  pasteInto(searchGray, patch, 120, 90);
  const searchColor = new cv.Mat(); cv.cvtColor(searchGray, searchColor, cv.COLOR_GRAY2RGB);
  const tmplGray = new cv.Mat(); patch.copyTo(tmplGray);
  const tmplColor = new cv.Mat(); cv.cvtColor(tmplGray, tmplColor, cv.COLOR_GRAY2RGB);

  const pipeline = [{ id: 'template_matching', params: { method: 'ccoeff', threshold: 50 }, enabled: true, nodeId: uniqueNodeId(), src2: { pipeline: [] } }];
  const code = buildPythonCode(pipeline);
  assert.match(code, /def process\(path_a, path_b\):/);

  const ctx = { src2: tmplColor };
  const jsOut = OPS.template_matching.apply(cv, searchGray, { method: 'ccoeff', threshold: 50 }, ctx);
  const byKey = Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));

  const { stdout } = runGeneratedPython2Image(code, matToRaw(searchColor), matToRaw(tmplColor));
  const m = stdout.match(/พบตำแหน่งที่ \((\d+), (\d+)\) — คะแนนความคล้าย ([\d.]+)%/);
  assert.ok(m, `expected a "found" line in Python stdout, got: ${stdout}`);
  const [, pyX, pyY, pyScore] = m;
  assert.ok(Math.abs(Number(pyX) - byKey.x) <= 3, `x: JS=${byKey.x} Python=${pyX}`);
  assert.ok(Math.abs(Number(pyY) - byKey.y) <= 3, `y: JS=${byKey.y} Python=${pyY}`);
  assert.ok(Math.abs(Number(pyScore) - byKey.match_score) <= 2, `score: JS=${byKey.match_score} Python=${pyScore}`);

  jsOut.delete(); searchGray.delete(); searchColor.delete(); patch.delete(); tmplGray.delete(); tmplColor.delete();
});

// ===================================================================
// psnr
// ===================================================================
test('tier2b-psnr-1. no src2 connected -> graceful info message, no throw', () => {
  const src = solidGray(20, 16, 90);
  const ctx = {};
  let out;
  assert.doesNotThrow(() => { out = OPS.psnr.apply(cv, src, {}, ctx); });
  assert.match(ctx.info, /ยังไม่ได้ต่อสาย 'เข้า B'/);
  out.delete(); src.delete();
});

test('tier2b-psnr-2. size mismatch -> matchSize resize path fires, ctx.info mentions it', () => {
  const a = solidGray(50, 40, 100);
  const b = solidGray(20, 15, 100);
  const ctx = { src2: b };
  const out = OPS.psnr.apply(cv, a, {}, ctx);
  assert.equal(out.cols, 50);
  assert.equal(out.rows, 40);
  assert.match(ctx.info, /ปรับขนาด B ให้เท่า A/);
  out.delete(); a.delete(); b.delete();
});

test('tier2b-psnr-3. identical images -> psnr=Infinity, mse=0, ctx.values shape correct, diff-map output is all-zero', () => {
  const a = solidGray(30, 30, 120);
  const b = new cv.Mat(); a.copyTo(b);
  const ctx = { src2: b };
  const out = OPS.psnr.apply(cv, a, {}, ctx);
  assert.equal(ctx.values.kind, 'scalars');
  const byKey = Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));
  assert.equal(byKey.psnr_db, Infinity);
  assert.equal(byKey.mse, 0);
  assert.match(ctx.info, /∞/);
  assert.equal(out.ucharPtr(15, 15)[0], 0);
  out.delete(); a.delete(); b.delete();
});

test('tier2b-psnr-4. known difference -> hand-computed MSE/PSNR match exactly', () => {
  const w = 20, h = 20;
  const a = solidGray(w, h, 120);
  const b = solidGray(w, h, 100); // diff=20 ทุกพิกเซล
  const ctx = { src2: b };
  const out = OPS.psnr.apply(cv, a, {}, ctx);
  const byKey = Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));
  const expectedMse = 20 * 20; // ทุกพิกเซลต่างกัน 20 พอดี -> MSE = 400
  const expectedPsnr = 10 * Math.log10((255 * 255) / expectedMse);
  assert.ok(Math.abs(byKey.mse - expectedMse) < 1e-6, `mse: expected ${expectedMse}, got ${byKey.mse}`);
  assert.ok(Math.abs(byKey.psnr_db - expectedPsnr) < 1e-3, `psnr: expected ${expectedPsnr}, got ${byKey.psnr_db}`);
  assert.equal(out.ucharPtr(10, 10)[0], 20);
  out.delete(); a.delete(); b.delete();
});

test('tier2b-psnr-5. Python codegen: MSE/PSNR match JS numerically (real cv2)', { skip: !pythonAvailable() && 'python/cv2 not available in this environment' }, () => {
  const w = 26, h = 18;
  const grayA = new cv.Mat(h, w, cv.CV_8UC1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) grayA.ucharPtr(y, x)[0] = (x * 7 + y * 13) % 256;
  const grayB = new cv.Mat(h, w, cv.CV_8UC1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) grayB.ucharPtr(y, x)[0] = (x * 3 + y * 29 + 17) % 256;
  const colorA = new cv.Mat(); cv.cvtColor(grayA, colorA, cv.COLOR_GRAY2RGB);
  const colorB = new cv.Mat(); cv.cvtColor(grayB, colorB, cv.COLOR_GRAY2RGB);

  const pipeline = [{ id: 'psnr', params: {}, enabled: true, nodeId: uniqueNodeId(), src2: { pipeline: [] } }];
  const code = buildPythonCode(pipeline);

  const ctx = { src2: colorB };
  const jsOut = OPS.psnr.apply(cv, grayA, {}, ctx);
  const byKey = Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));

  const { stdout } = runGeneratedPython2Image(code, matToRaw(colorA), matToRaw(colorB));
  const m = stdout.match(/PSNR = ([\d.]+) dB \(MSE = ([\d.]+)\)/);
  assert.ok(m, `expected a PSNR line in Python stdout, got: ${stdout}`);
  const [, pyPsnr, pyMse] = m;
  assert.ok(Math.abs(Number(pyMse) - byKey.mse) < 0.5, `mse: JS=${byKey.mse} Python=${pyMse}`);
  assert.ok(Math.abs(Number(pyPsnr) - byKey.psnr_db) < 0.1, `psnr: JS=${byKey.psnr_db} Python=${pyPsnr}`);

  jsOut.delete(); grayA.delete(); grayB.delete(); colorA.delete(); colorB.delete();
});

// ===================================================================
// ssim
// ===================================================================
test('tier2b-ssim-1. no src2 connected -> graceful info message, no throw', () => {
  const src = solidGray(20, 16, 90);
  const ctx = {};
  let out;
  assert.doesNotThrow(() => { out = OPS.ssim.apply(cv, src, {}, ctx); });
  assert.match(ctx.info, /ยังไม่ได้ต่อสาย 'เข้า B'/);
  out.delete(); src.delete();
});

test('tier2b-ssim-2. size mismatch -> matchSize resize path fires, ctx.info mentions it', () => {
  const a = solidGray(300, 240, 100);
  const b = solidGray(100, 80, 100);
  const ctx = { src2: b };
  const out = OPS.ssim.apply(cv, a, {}, ctx);
  assert.equal(out.cols, 300);
  assert.equal(out.rows, 240);
  assert.match(ctx.info, /ปรับขนาด B ให้เท่า A/);
  out.delete(); a.delete(); b.delete();
});

test('tier2b-ssim-3. identical images -> ssim essentially exactly 1.0 (documented tolerance: <1e-5, measured 0 in practice)', () => {
  const a = new cv.Mat(300, 300, cv.CV_8UC1);
  for (let y = 0; y < 300; y++) for (let x = 0; x < 300; x++) a.ucharPtr(y, x)[0] = (x * 41 + y * 71) % 256;
  const b = new cv.Mat(); a.copyTo(b);
  const ctx = { src2: b };
  const out = OPS.ssim.apply(cv, a, {}, ctx);
  assert.equal(ctx.values.kind, 'scalars');
  const ssim = ctx.values.fields.find((f) => f.key === 'ssim').value;
  assert.ok(Math.abs(ssim - 1) < 1e-5, `expected ssim ~1.0 for identical images, got ${ssim}`);
  assert.match(ctx.info, /SSIM = 1\.0000/);
  out.delete(); a.delete(); b.delete();
});

test('tier2b-ssim-4. very different images -> ssim noticeably below 1, output is a valid normalized 0-255 map at full resolution', () => {
  const w = 300, h = 240;
  const a = new cv.Mat(h, w, cv.CV_8UC1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) a.ucharPtr(y, x)[0] = (x * 41 + y * 71) % 256;
  const b = new cv.Mat(h, w, cv.CV_8UC1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) b.ucharPtr(y, x)[0] = ((x * 41 + y * 71) % 256 > 127) ? 0 : 255; // เกือบตรงข้ามกันทั่วภาพ
  const ctx = { src2: b };
  const out = OPS.ssim.apply(cv, a, {}, ctx);
  const ssim = ctx.values.fields.find((f) => f.key === 'ssim').value;
  assert.ok(ssim < 0.5, `expected clearly-different images to score well below 1, got ${ssim}`);
  assert.equal(out.cols, w);
  assert.equal(out.rows, h);
  assert.equal(out.channels(), 1);
  out.delete(); a.delete(); b.delete();
});

test('tier2b-ssim-5. ctx.values shape validation: kind="scalars", fields=[{key:"ssim",...}]', () => {
  const a = solidGray(100, 100, 100);
  const b = solidGray(100, 100, 150);
  const ctx = { src2: b };
  const out = OPS.ssim.apply(cv, a, {}, ctx);
  assert.equal(ctx.values.kind, 'scalars');
  assert.equal(ctx.values.fields.length, 1);
  const f = ctx.values.fields[0];
  assert.equal(f.key, 'ssim');
  assert.equal(typeof f.label, 'string');
  assert.equal(typeof f.value, 'number');
  out.delete(); a.delete(); b.delete();
});

test('tier2b-ssim-6. cv2 5.0.0 has no built-in SSIM helper (spec asks to check, not assume) — quality module and skimage both absent by default', { skip: !pythonAvailable() && 'python/cv2 not available in this environment' }, () => {
  let stdout;
  assert.doesNotThrow(() => {
    stdout = execFileSync('python', ['-c', `
import cv2
print("cv2_version", cv2.__version__)
print("has_quality_module", hasattr(cv2, "quality"))
try:
    import skimage
    print("has_skimage", True)
except ImportError:
    print("has_skimage", False)
`], { encoding: 'utf8' });
  });
  assert.match(stdout, /has_skimage False/, 'expected skimage NOT to be installed (confirms manual GaussianBlur SSIM port is necessary, not skippable)');
});

// ===================================================================
// รอบแก้ 18b (scope-change ที่ 2): ssim speed param + template_matching Option B (ROI re-score)
// ===================================================================
// ภาพ noise แบบ deterministic (LCG) — ต้องเป็นรายละเอียดความถี่สูงจริง เพราะเป็นกรณีที่ downscale ทำให้ค่า
// SSIM เพี้ยนมากที่สุด (qa วัด: full-res 0.34 vs downscaled 0.95) ใช้ seed ต่างกันเพื่อได้คู่ภาพที่ต่างกัน
function noiseGray(w, h, seed) {
  const m = new cv.Mat(h, w, cv.CV_8UC1);
  let s = seed >>> 0;
  for (let i = 0; i < w * h; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; m.data[i] = s >>> 24; }
  return m;
}
// B = A + noise เล็กน้อย (คู่ภาพ "คล้ายแต่ไม่เหมือน" แบบที่ผู้เรียนใช้วัดจริง เช่นภาพก่อน/หลังบีบอัด)
function noisyCopy(a, amp, seed) {
  const m = new cv.Mat(a.rows, a.cols, cv.CV_8UC1);
  let s = seed >>> 0;
  for (let i = 0; i < a.rows * a.cols; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const v = a.data[i] + ((s >>> 24) % (2 * amp + 1)) - amp;
    m.data[i] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  return m;
}
// ภาพพื้นเรียบ (sin/cos) + noise ละเอียดอิสระต่อภาพ (seed ต่างกัน) — แบบเดียวกับเคสที่ qa ใช้พิสูจน์ปัญหา:
// ที่ความละเอียดเต็ม noise ต่างกันทำให้ SSIM ต่ำ แต่การย่อ (INTER_AREA = low-pass) เฉลี่ย noise ทิ้งจนเหลือแต่
// พื้นเรียบที่เหมือนกัน -> SSIM สูงเกินจริง (วัดระหว่างพัฒนา amp=20: 1920x1080 full 0.185 vs fast 0.749,
// 1280x720 full 0.185 vs fast 0.519) — ภาพ noise ล้วนสองใบใช้ไม่ได้ เพราะความถี่สูงอยู่ทุกสเกล ค่าไม่ต่างกัน
function smoothPlusNoise(w, h, amp, seed) {
  const m = new cv.Mat(h, w, cv.CV_8UC1);
  let s = seed >>> 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const v = Math.round(128 + 60 * Math.sin(x / 50) * Math.cos(y / 40) + ((s >>> 24) % (2 * amp + 1)) - amp);
    m.data[y * w + x] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  return m;
}
const SSIM_FAST_NOTE = /ย่อขนาดลงก่อนเพื่อความเร็ว/;
const SSIM_BIG_HINT = /ภาพใหญ่ อาจคำนวณช้ากว่าปกติ/;
function ssimOf(ctx) { return ctx.values.fields.find((f) => f.key === 'ssim').value; }

test('tier2b-ssim-7. backward-compat: params without "speed" resolve to full mode — identical value to speed:"full" on a 1920x1080 pair where fast mode overstates similarity (qa\'s failure case)', () => {
  const a = smoothPlusNoise(1920, 1080, 20, 1);
  const b = smoothPlusNoise(1920, 1080, 20, 2);
  const cNo = { src2: b }, cFull = { src2: b }, cFast = { src2: b };
  OPS.ssim.apply(cv, a, {}, cNo).delete();
  OPS.ssim.apply(cv, a, { speed: 'full' }, cFull).delete();
  OPS.ssim.apply(cv, a, { speed: 'fast' }, cFast).delete();
  assert.equal(ssimOf(cNo), ssimOf(cFull), 'missing speed key must compute exactly like speed:"full"');
  assert.doesNotMatch(cNo.info, SSIM_FAST_NOTE, 'old projects (no speed key) must not silently run the downscaled path');
  // ยืนยันว่าเคสนี้เป็นเคสที่ downscale ทำให้ค่าเพี้ยนจริงระดับที่ qa รายงาน (ไม่งั้นการเทียบด้านบนไม่พิสูจน์อะไร)
  assert.ok(ssimOf(cFull) < 0.3, `full-res SSIM should be low for independently-noised images, got ${ssimOf(cFull)}`);
  assert.ok(ssimOf(cFast) - ssimOf(cFull) > 0.3, `expected downscale to inflate SSIM a lot, full=${ssimOf(cFull)} fast=${ssimOf(cFast)}`);
  assert.ok(ssimOf(cFast) > ssimOf(cFull), 'downscale (low-pass) should overstate similarity on noise — matches qa measurement direction');
  // FILTERS entry (ที่ UI ใช้จริง) ต้องประกาศ default เป็น full ด้วย
  const speedParam = FILTERS.ssim.params.find((p) => p.key === 'speed');
  assert.ok(speedParam, 'ssim must expose a "speed" param');
  assert.equal(speedParam.default, 'full');
  assert.deepEqual(speedParam.options.map((o) => o.value), ['full', 'fast']);
  a.delete(); b.delete();
});

test('tier2b-ssim-8. disclosure: fast-mode note appears ONLY when the downscale actually fires (fast + >640px); full mode on a big image shows the "slow, can switch to fast" hint instead; small images show neither', () => {
  const bigA = noiseGray(700, 500, 3), bigB = noisyCopy(bigA, 30, 4);
  const smallA = noiseGray(300, 200, 5), smallB = noisyCopy(smallA, 30, 6);
  const run = (a, b, p) => { const ctx = { src2: b }; OPS.ssim.apply(cv, a, p, ctx).delete(); return ctx.info; };

  const fastBig = run(bigA, bigB, { speed: 'fast' });
  assert.match(fastBig, SSIM_FAST_NOTE, 'fast + >640px must disclose that the number came from a downscaled image');
  assert.doesNotMatch(fastBig, SSIM_BIG_HINT);

  const fastSmall = run(smallA, smallB, { speed: 'fast' });
  assert.doesNotMatch(fastSmall, SSIM_FAST_NOTE, 'fast on a <=640px image does not downscale, so it must not claim it did');

  const fullBig = run(bigA, bigB, { speed: 'full' });
  assert.doesNotMatch(fullBig, SSIM_FAST_NOTE);
  assert.match(fullBig, SSIM_BIG_HINT, 'full mode on a big image should hint that fast mode exists');

  const fullSmall = run(smallA, smallB, {});
  assert.doesNotMatch(fullSmall, SSIM_FAST_NOTE);
  assert.doesNotMatch(fullSmall, SSIM_BIG_HINT);

  // disclosure ต่อท้ายได้พร้อมกับ note ของ matchSize (สองภาพขนาดไม่ตรงกัน) โดยไม่ทับกัน
  const bigBSmaller = noiseGray(400, 300, 7);
  const both = run(bigA, bigBSmaller, { speed: 'fast' });
  assert.match(both, /ปรับขนาด B ให้เท่า A/);
  assert.match(both, SSIM_FAST_NOTE);

  bigA.delete(); bigB.delete(); smallA.delete(); smallB.delete(); bigBSmaller.delete();
});

test('tier2b-ssim-9. Python codegen parity (real cv2): SSIM value matches JS within 1e-3 in BOTH speed modes on a noisy >640px pair, and the printed disclosure matches the mode', { skip: !pythonAvailable() && 'python/cv2 not available in this environment' }, () => {
  const grayA = smoothPlusNoise(1280, 720, 20, 11);
  const grayB = smoothPlusNoise(1280, 720, 20, 12);
  const colorA = new cv.Mat(); cv.cvtColor(grayA, colorA, cv.COLOR_GRAY2RGB);
  const colorB = new cv.Mat(); cv.cvtColor(grayB, colorB, cv.COLOR_GRAY2RGB);
  const results = {};
  for (const speed of ['full', 'fast']) {
    const pipeline = [{ id: 'ssim', params: { speed }, enabled: true, nodeId: uniqueNodeId(), src2: { pipeline: [] } }];
    const code = buildPythonCode(pipeline);
    const ctx = { src2: colorB };
    OPS.ssim.apply(cv, grayA, { speed }, ctx).delete();
    const js = ssimOf(ctx);
    const { stdout } = runGeneratedPython2Image(code, matToRaw(colorA), matToRaw(colorB));
    const m = stdout.match(/SSIM = (-?[\d.]+)/);
    assert.ok(m, `speed=${speed}: expected an SSIM line in Python stdout, got: ${stdout}`);
    const py = Number(m[1]);
    // ค่าที่ print มี 4 ตำแหน่งทศนิยม (ปัดได้สูงสุด 5e-5) + float noise ที่ qa วัดได้ ~2e-5..4e-5 — 1e-3 หลวมพอไม่ flaky
    // แต่ยังจับ formula regression ได้ (ต่างโหมดกันเกิน 0.05 ในเคสนี้ ดู tier2b-ssim-7)
    assert.ok(Math.abs(py - js) < 1e-3, `speed=${speed}: JS=${js} Python=${py}`);
    if (speed === 'fast') assert.match(stdout, SSIM_FAST_NOTE, 'Python fast mode must print the downscale disclosure');
    else assert.doesNotMatch(stdout, SSIM_FAST_NOTE, 'Python full mode must not print the downscale disclosure');
    results[speed] = js;
  }
  assert.ok(Math.abs(results.full - results.fast) > 0.01, 'sanity: the two modes should give different values on this noisy pair');
  grayA.delete(); grayB.delete(); colorA.delete(); colorB.delete();
});

test('tier2b-tm-8. exact match at 1920x1080 scores ~100% (full-res ROI re-score; downscaled-only score was 87.7% in qa\'s measurement) at the exact planted location', () => {
  const search = noiseGray(1920, 1080, 21);
  const x0 = 913, y0 = 457; // ไม่ลงตัวกับ scale 1/3 โดยตั้งใจ (ทดสอบว่า refine ใน ROI ได้ตำแหน่งเต็มความละเอียดเป๊ะ)
  const tmpl = search.roi(new cv.Rect(x0, y0, 100, 100));
  const tmplCopy = new cv.Mat(); tmpl.copyTo(tmplCopy); tmpl.delete();
  const ctx = { src2: tmplCopy };
  const out = OPS.template_matching.apply(cv, search, { method: 'ccoeff', threshold: 70 }, ctx);
  const byKey = Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));
  assert.ok(byKey.match_score > 99.9, `expected ~100% for an exact copy, got ${byKey.match_score}`);
  assert.equal(byKey.x, x0);
  assert.equal(byKey.y, y0);
  assert.match(ctx.info, /พบตำแหน่งที่/);
  out.delete(); search.delete(); tmplCopy.delete();
});

test('tier2b-tm-9. edge-clamped ROI correctness: templates cut from all four corners of a >640px image are found exactly with ~100%', () => {
  const W = 1280, H = 720, T = 90;
  const search = noiseGray(W, H, 31);
  for (const [x0, y0] of [[0, 0], [W - T, 0], [0, H - T], [W - T, H - T]]) {
    const r = search.roi(new cv.Rect(x0, y0, T, T));
    const tmpl = new cv.Mat(); r.copyTo(tmpl); r.delete();
    const ctx = { src2: tmpl };
    OPS.template_matching.apply(cv, search, { method: 'ccoeff', threshold: 70 }, ctx).delete();
    const byKey = Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));
    assert.ok(byKey.match_score > 99.9, `corner (${x0},${y0}): expected ~100%, got ${byKey.match_score}`);
    assert.equal(byKey.x, x0, `corner (${x0},${y0}) x`);
    assert.equal(byKey.y, y0, `corner (${x0},${y0}) y`);
    tmpl.delete();
  }
  search.delete();
});

test('tier2b-tm-10. Python codegen parity (real cv2) above 640px: ROI re-score gives the same score (within 0.1%) and the same full-res location as JS', { skip: !pythonAvailable() && 'python/cv2 not available in this environment' }, () => {
  const searchGray = noiseGray(1280, 720, 41);
  const r = searchGray.roi(new cv.Rect(611, 287, 80, 80));
  const tmplGray = new cv.Mat(); r.copyTo(tmplGray); r.delete();
  // ใส่ noise ลงใน template เล็กน้อย ให้คะแนนไม่ใช่ 100% เป๊ะ (ทดสอบว่าค่าคะแนนตรงกันจริง ไม่ใช่แค่ทั้งคู่ติดเพดาน)
  const tmplNoisy = noisyCopy(tmplGray, 25, 42);
  const searchColor = new cv.Mat(); cv.cvtColor(searchGray, searchColor, cv.COLOR_GRAY2RGB);
  const tmplColor = new cv.Mat(); cv.cvtColor(tmplNoisy, tmplColor, cv.COLOR_GRAY2RGB);

  const pipeline = [{ id: 'template_matching', params: { method: 'ccoeff', threshold: 50 }, enabled: true, nodeId: uniqueNodeId(), src2: { pipeline: [] } }];
  const code = buildPythonCode(pipeline);
  const ctx = { src2: tmplColor };
  OPS.template_matching.apply(cv, searchGray, { method: 'ccoeff', threshold: 50 }, ctx).delete();
  const byKey = Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));
  assert.ok(byKey.match_score < 99.5 && byKey.match_score > 50, `fixture sanity: want a non-saturated score, got ${byKey.match_score}`);

  const { stdout } = runGeneratedPython2Image(code, matToRaw(searchColor), matToRaw(tmplColor));
  const m = stdout.match(/พบตำแหน่งที่ \((\d+), (\d+)\) — คะแนนความคล้าย ([\d.]+)%/);
  assert.ok(m, `expected a "found" line in Python stdout, got: ${stdout}`);
  assert.equal(Number(m[1]), byKey.x, `x: JS=${byKey.x} Python=${m[1]}`);
  assert.equal(Number(m[2]), byKey.y, `y: JS=${byKey.y} Python=${m[2]}`);
  assert.ok(Math.abs(Number(m[3]) - byKey.match_score) <= 0.1, `score: JS=${byKey.match_score} Python=${m[3]}`);
  assert.equal(byKey.x, 611);
  assert.equal(byKey.y, 287);
  searchGray.delete(); tmplGray.delete(); tmplNoisy.delete(); searchColor.delete(); tmplColor.delete();
});

// ===================================================================
// wiring sanity (palette/registry) — เสริมจาก tier2-infra.test.mjs (ไม่ซ้ำ แค่เช็คของใหม่รอบนี้)
// ===================================================================
test('tier2b-wiring-1. FILTERS wiring: imfuse/mask added to image_arithmetic opIds, template_matching/psnr/ssim registered as inputs:2 single-op groups', () => {
  assert.ok(FILTERS.image_arithmetic);
  const modeParam = FILTERS.image_arithmetic.params.find((p) => p.key === 'mode');
  assert.ok(modeParam.options.some((o) => o.value === 'imfuse'));
  assert.ok(modeParam.options.some((o) => o.value === 'mask'));

  for (const id of ['template_matching', 'psnr', 'ssim']) {
    assert.ok(FILTERS[id], `expected FILTERS.${id} to exist`);
    assert.equal(FILTERS[id].inputs, 2, `FILTERS.${id}.inputs must be 2`);
    assert.equal(typeof FILTERS[id].apply, 'function');
  }
  assert.notEqual(FILTERS.template_matching.group, FILTERS.psnr.group);
  assert.equal(FILTERS.psnr.group, FILTERS.ssim.group, 'psnr/ssim share the "Image Quality" group display name');

  const flat = TOOL_SECTIONS.flatMap((s) => s.ops);
  for (const opId of ['imfuse', 'mask', 'template_matching', 'psnr', 'ssim']) {
    assert.ok(flat.includes(opId), `expected ${opId} listed in some TOOL_SECTIONS entry`);
  }
});

test('tier2b-wiring-2. GROUP_COLORS/GROUP_ORDER include the 2 new groups, with colors distinct from every pre-existing swatch', async () => {
  const { GROUP_COLORS, GROUP_ORDER } = await import('../src/renderer/filters/registry.js');
  assert.ok(GROUP_COLORS['Template Matching']);
  assert.ok(GROUP_COLORS['Image Quality']);
  assert.ok(GROUP_ORDER.includes('Template Matching'));
  assert.ok(GROUP_ORDER.includes('Image Quality'));
  // เช็คเฉพาะว่า 2 สีใหม่ไม่ชนกับสีเดิมตัวใดเลย — ไม่เช็ค global uniqueness ทั้งตาราง เพราะ GROUP_COLORS
  // เดิม (ก่อนรอบนี้) มี legacy group บางคู่ที่ตั้งใจใช้สีซ้ำกับ group หลักของมันอยู่แล้ว (เช่น "Enhancement"
  // ใช้สีเขียวเดียวกับ "Image Adjust" ที่มันถูกแทนที่ — เป็น design ที่มีอยู่ก่อนรอบนี้ ไม่ใช่สิ่งที่ต้องแก้)
  const newHex = new Set([GROUP_COLORS['Template Matching'].title.toLowerCase(), GROUP_COLORS['Image Quality'].title.toLowerCase()]);
  assert.equal(newHex.size, 2, 'the 2 new colors must not equal each other');
  for (const [group, c] of Object.entries(GROUP_COLORS)) {
    if (group === 'Template Matching' || group === 'Image Quality') continue;
    assert.ok(!newHex.has(c.title.toLowerCase()), `pre-existing group "${group}"'s color collides with one of the 2 new groups`);
  }
});

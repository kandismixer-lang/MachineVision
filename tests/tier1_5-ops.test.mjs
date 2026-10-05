import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import { OPS, FILTERS } from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS } from '../src/renderer/ui/palette-panel.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import { OP_PRIMARY } from '../src/renderer/filters/ui-meta.js';
import { assertLeakFree } from './_leak-helpers.mjs';

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
// A1 — hough_circles codegen defensive reshape
// ===================================================================

test('A1. codegen: hough_circles uses .reshape(-1, 3) (defensive hardening, not a confirmed bug on cv2 5.0.0)', () => {
  const code = buildPythonCode([{ id: 'object_detection', params: {
    mode: 'hough_circles',
    hough_circles__dp: 1, hough_circles__minDist: 30, hough_circles__param2: 40,
    hough_circles__minR: 0, hough_circles__maxR: 0,
  } }]);
  assert.ok(code.includes('.reshape(-1, 3)'), 'expected defensive reshape in generated hough_circles code');
});

// ===================================================================
// A2 — try/finally hardening for crop/resize/rotate: behavior must be unchanged
// ===================================================================

test('A2. crop/resize/rotate: happy-path behavior unchanged after try/finally hardening', () => {
  const src = new cv.Mat.zeros(100, 200, cv.CV_8UC3); // rows=100 (h), cols=200 (w)
  const cropped = OPS.crop.apply(cv, src, { x: 10, y: 10, w: 50, h: 50 }, {});
  assert.equal(cropped.cols, 100);
  assert.equal(cropped.rows, 50);
  cropped.delete();

  const resized = OPS.resize.apply(cv, src, { scale: 50 }, {});
  assert.equal(resized.cols, 100);
  assert.equal(resized.rows, 50);
  resized.delete();

  const rotated = OPS.rotate.apply(cv, src, { angle: 90, expand: false }, {});
  assert.equal(rotated.cols, 200);
  assert.equal(rotated.rows, 100);
  rotated.delete();
  src.delete();
});

// ===================================================================
// B1 — Custom Kernel Convolution
// ===================================================================

test('B1. Custom Kernel: identity preset (amount=1) is pixel-identical to input', () => {
  const src = new cv.Mat(30, 30, cv.CV_8UC3);
  for (let i = 0; i < src.data.length; i++) src.data[i] = (i * 37) % 256;
  const out = OPS.custom_kernel.apply(cv, src, { preset: 'identity', amount: 1 }, {});
  assert.deepEqual(Array.from(out.data), Array.from(src.data));
  out.delete();
  src.delete();
});

test('codegen: custom_kernel amount=0 must generate the addWeighted(1,0) path, not "img = filtered" (amount=0 must not become amount=1)', () => {
  const code = buildPythonCode([{ id: 'custom_kernel', params: { preset: 'sharpen', amount: 0 } }]);
  assert.ok(code.includes('cv2.addWeighted(img, 1, filtered, 0, 0)'), `expected addWeighted with weights 1 and 0 for amount=0, got:\n${code}`);
  assert.ok(!code.includes('img = filtered'), `amount=0 must not take the amount===1 shortcut ("img = filtered"), got:\n${code}`);
});

test('B1b. Custom Kernel: sharpen preset increases contrast at a step edge', () => {
  const src = new cv.Mat(20, 20, cv.CV_8UC1, new cv.Scalar(50));
  const roi = src.roi(new cv.Rect(10, 0, 10, 20));
  roi.setTo(new cv.Scalar(200));
  roi.delete();
  const mmSrc = cv.minMaxLoc(src);
  const out = OPS.custom_kernel.apply(cv, src, { preset: 'sharpen', amount: 1 }, {});
  const mmOut = cv.minMaxLoc(out);
  const rangeSrc = mmSrc.maxVal - mmSrc.minVal;
  const rangeOut = mmOut.maxVal - mmOut.minVal;
  assert.ok(rangeOut > rangeSrc, `expected sharpened range (${rangeOut}) > original range (${rangeSrc})`);
  out.delete();
  src.delete();
});

// ===================================================================
// B2 — Laplacian Edge + Scharr Edge
// ===================================================================

test('B2. Laplacian Edge: bright band at a vertical edge column, near-zero elsewhere', () => {
  const src = new cv.Mat(40, 40, cv.CV_8UC1, new cv.Scalar(20));
  const roi = src.roi(new cv.Rect(20, 0, 20, 40));
  roi.setTo(new cv.Scalar(220));
  roi.delete();
  const out = OPS.laplacian_edge.apply(cv, src, { ksize: 3 }, {});
  const edgeVal = out.data[20 * 40 + 20]; // near the edge column
  const flatVal = out.data[20 * 40 + 5]; // far from edge, flat region
  assert.ok(edgeVal > 50, `expected bright response near edge, got ${edgeVal}`);
  assert.ok(flatVal < 10, `expected near-zero response in flat region, got ${flatVal}`);
  out.delete();
  src.delete();
});

test('B2c. Laplacian Edge: NaN/undefined ksize must not throw (falls back to 3, stays odd)', () => {
  const src = new cv.Mat(20, 20, cv.CV_8UC1, new cv.Scalar(100));
  assert.doesNotThrow(() => {
    const out1 = OPS.laplacian_edge.apply(cv, src, { ksize: NaN }, {});
    out1.delete();
    const out2 = OPS.laplacian_edge.apply(cv, src, {}, {}); // ksize undefined
    out2.delete();
  });
  src.delete();
});

test('B2b. Scharr Edge: dir=x responds strongly to a vertical edge, dir=y does not', () => {
  const src = new cv.Mat(40, 40, cv.CV_8UC1, new cv.Scalar(20));
  const roi = src.roi(new cv.Rect(20, 0, 20, 40));
  roi.setTo(new cv.Scalar(220));
  roi.delete();
  const outX = OPS.scharr_edge.apply(cv, src, { dir: 'x' }, {});
  const outY = OPS.scharr_edge.apply(cv, src, { dir: 'y' }, {});
  const mmX = cv.minMaxLoc(outX);
  const mmY = cv.minMaxLoc(outY);
  assert.ok(mmX.maxVal > 100, `expected strong X response, got max=${mmX.maxVal}`);
  assert.ok(mmY.maxVal < 10, `expected near-zero Y response on a purely vertical edge, got max=${mmY.maxVal}`);
  outX.delete();
  outY.delete();
  src.delete();
});

test('codegen: morphology clear_border/remove_small binarize before connectedComponentsWithStats (matches JS apply(), and avoids treating a non-binary gray image as one giant border-touching component)', () => {
  const clearBorderCode = buildPythonCode([{ id: 'morphology', params: { op: 'clear_border' } }]);
  assert.ok(clearBorderCode.includes('cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)'), 'expected clear_border codegen to binarize before connectedComponentsWithStats');
  assert.ok(clearBorderCode.includes('connectedComponentsWithStats(bin_img'), 'expected clear_border to run connectedComponentsWithStats on the binarized image, not raw img');

  const removeSmallCode = buildPythonCode([{ id: 'morphology', params: { op: 'remove_small', minArea: 150 } }]);
  assert.ok(removeSmallCode.includes('cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)'), 'expected remove_small codegen to binarize before connectedComponentsWithStats');
  assert.ok(removeSmallCode.includes('connectedComponentsWithStats(bin_img'), 'expected remove_small to run connectedComponentsWithStats on the binarized image, not raw img');
});

// ===================================================================
// B3 — Clear Border (morphology op)
// ===================================================================

test('B3. Morphology Clear Border: removes corner-touching blob, keeps interior blob, reports 1 removed', () => {
  const src = new cv.Mat.zeros(60, 60, cv.CV_8UC1);
  const cornerBlob = src.roi(new cv.Rect(0, 0, 10, 10)); // touches top-left corner
  cornerBlob.setTo(new cv.Scalar(255));
  cornerBlob.delete();
  const interiorBlob = src.roi(new cv.Rect(30, 30, 10, 10)); // fully interior
  interiorBlob.setTo(new cv.Scalar(255));
  interiorBlob.delete();

  const ctx = {};
  const out = OPS.morphology.apply(cv, src, { op: 'clear_border', shape: 'rect', ksize: 5, iterations: 1, minArea: 150 }, ctx);
  assert.equal(out.data[5 * 60 + 5], 0, 'corner-touching blob must be removed');
  assert.equal(out.data[35 * 60 + 35], 255, 'interior blob must survive');
  assert.match(ctx.info, /ลบวัตถุที่แตะขอบภาพ 1 ชิ้น/);
  out.delete();
  src.delete();
});

// ===================================================================
// B4 — Remove Small Objects (morphology op)
// ===================================================================

test('B4. Morphology Remove Small Objects: 100px blob removed, 900px (30x30) blob kept, reports 1 removed', () => {
  const src = new cv.Mat.zeros(80, 80, cv.CV_8UC1);
  const small = src.roi(new cv.Rect(5, 5, 10, 10)); // 100px area
  small.setTo(new cv.Scalar(255));
  small.delete();
  const big = src.roi(new cv.Rect(30, 30, 30, 30)); // 900px area
  big.setTo(new cv.Scalar(255));
  big.delete();

  const ctx = {};
  const out = OPS.morphology.apply(cv, src, { op: 'remove_small', shape: 'rect', ksize: 5, iterations: 1, minArea: 200 }, ctx);
  assert.equal(out.data[10 * 80 + 10], 0, 'small 100px blob must be removed');
  assert.equal(out.data[45 * 80 + 45], 255, 'big 900px blob must survive');
  assert.match(ctx.info, /ลบวัตถุเล็ก 1 ชิ้น/);
  out.delete();
  src.delete();
});

// ===================================================================
// B5 — Distance Transform
// ===================================================================

test('B5. Distance Transform: center of a solid square is brighter than a pixel near its edge', () => {
  const src = new cv.Mat.zeros(80, 80, cv.CV_8UC1);
  const square = src.roi(new cv.Rect(10, 10, 60, 60));
  square.setTo(new cv.Scalar(255));
  square.delete();

  const out = OPS.distance_transform.apply(cv, src, { maskSize: '5' }, {});
  const center = out.data[40 * 80 + 40];
  const nearEdge = out.data[40 * 80 + 12]; // near left edge of the square
  assert.ok(center > nearEdge, `expected center (${center}) brighter than near-edge (${nearEdge})`);
  out.delete();
  src.delete();
});

test('B5b. Distance Transform: no-background (all-white) image reports the Thai "no background" note, not a huge px sentinel', () => {
  const src = new cv.Mat(20, 20, cv.CV_8UC1, new cv.Scalar(255)); // ทุกพิกเซลขาว -> ไม่มีพื้นหลัง (0) เลย
  const ctx = {};
  const out = OPS.distance_transform.apply(cv, src, { maskSize: '5' }, ctx);
  assert.match(ctx.info, /ไม่มีพื้นหลัง.*จึงวัดระยะไม่ได้/, `expected the "no background" note, got: ${ctx.info}`);
  out.delete();
  src.delete();
});

// ===================================================================
// B6 — Multi-level Otsu
// ===================================================================

test('B6. Multi-level Otsu: 3-band striped image maps to exactly {0,128,255}, fast (<200ms)', () => {
  const src = new cv.Mat(60, 60, cv.CV_8UC1);
  for (let y = 0; y < 60; y++) {
    for (let x = 0; x < 60; x++) {
      const v = y < 20 ? 20 : y < 40 ? 128 : 230;
      src.data[y * 60 + x] = v;
    }
  }
  const started = performance.now();
  const out = OPS.multi_otsu_threshold.apply(cv, src, {}, {});
  const elapsed = performance.now() - started;
  assert.ok(elapsed < 200, `expected multi-otsu to complete under 200ms, took ${elapsed}ms`);

  const values = new Set(Array.from(out.data));
  for (const v of values) assert.ok(v === 0 || v === 128 || v === 255, `unexpected output value ${v}`);
  assert.equal(out.data[5 * 60 + 5], 0, 'dark band (20) must map to 0');
  assert.equal(out.data[25 * 60 + 5], 128, 'mid band (128) must map to 128');
  assert.equal(out.data[45 * 60 + 5], 255, 'bright band (230) must map to 255');
  out.delete();
  src.delete();
});

test('B6b. Multi-level Otsu degenerate: flat 128 image stays 128 (not pushed to 255 by a tied, invalid t1/t2)', () => {
  const src = new cv.Mat(20, 20, cv.CV_8UC1, new cv.Scalar(128));
  const out = OPS.multi_otsu_threshold.apply(cv, src, {}, {});
  const values = new Set(Array.from(out.data));
  assert.deepEqual([...values], [128], `expected a flat 128 image to stay flat at 128, got distinct values ${[...values]}`);
  out.delete();
  src.delete();
});

test('B6c. Multi-level Otsu degenerate: two-level {30,200} image maps to {0,255} (not {128,255})', () => {
  const src = new cv.Mat(20, 20, cv.CV_8UC1, new cv.Scalar(30));
  const roi = src.roi(new cv.Rect(0, 0, 10, 20));
  roi.setTo(new cv.Scalar(200));
  roi.delete();
  const out = OPS.multi_otsu_threshold.apply(cv, src, {}, {});
  const values = new Set(Array.from(out.data));
  assert.deepEqual([...values].sort((a, b) => a - b), [0, 255], `expected a two-level {30,200} image to map to {0,255}, got ${[...values]}`);
  out.delete();
  src.delete();
});

test('codegen: multi_otsu_threshold Python loop also skips zero-weight (w0/w1/w2===0) candidates (parity with JS fix)', () => {
  const code = buildPythonCode([{ id: 'multi_otsu_threshold', params: {} }]);
  assert.ok(code.includes('if w0 == 0 or w1 == 0 or w2 == 0:'), `expected generated Python to skip degenerate (zero-weight) candidates, got:\n${code}`);
});

// ===================================================================
// B7 — Gray-World White Balance
// ===================================================================

test('B7. White Balance: color cast shrinks after gray-world correction; grayscale input is a safe no-op', () => {
  const src = new cv.Mat(40, 40, cv.CV_8UC3);
  // multi-color test pattern with a warm (R+80) cast baked in
  for (let y = 0; y < 40; y++) {
    for (let x = 0; x < 40; x++) {
      const idx = (y * 40 + x) * 3;
      const base = (x * 6) % 200;
      src.data[idx] = Math.min(255, base + 80); // R (cast)
      src.data[idx + 1] = base;
      src.data[idx + 2] = 255 - base;
    }
  }
  const meanBefore = cv.mean(src);
  const spreadBefore = Math.max(meanBefore[0], meanBefore[1], meanBefore[2]) - Math.min(meanBefore[0], meanBefore[1], meanBefore[2]);

  const out = OPS.white_balance.apply(cv, src, {}, {});
  const meanAfter = cv.mean(out);
  const spreadAfter = Math.max(meanAfter[0], meanAfter[1], meanAfter[2]) - Math.min(meanAfter[0], meanAfter[1], meanAfter[2]);
  assert.ok(spreadAfter < spreadBefore, `expected channel-mean spread to shrink: before=${spreadBefore} after=${spreadAfter}`);
  out.delete();
  src.delete();

  // Regression: pure grayscale input must not throw and must report the "no color" note.
  const gray = new cv.Mat.zeros(20, 20, cv.CV_8UC1);
  const ctx = {};
  assert.doesNotThrow(() => {
    const out2 = OPS.white_balance.apply(cv, gray, {}, ctx);
    out2.delete();
  });
  assert.match(ctx.info, /ไม่มีสีให้ปรับ/);
  gray.delete();
});

// ===================================================================
// B8 — FFT Magnitude Spectrum + Low/High-pass Filter
// ===================================================================

test('B8. FFT Magnitude: non-constant output, same dimensions as input', () => {
  const src = new cv.Mat(64, 64, cv.CV_8UC1);
  for (let i = 0; i < src.data.length; i++) src.data[i] = (i * 53) % 256;
  const out = OPS.fft_magnitude.apply(cv, src, {}, {});
  assert.equal(out.cols, 64);
  assert.equal(out.rows, 64);
  const mm = cv.minMaxLoc(out);
  assert.ok(mm.maxVal > mm.minVal, 'expected non-constant magnitude spectrum');
  out.delete();
  src.delete();
});

test('B8b. FFT Filter: low-pass barely changes a constant image; high-pass collapses it toward flat', () => {
  const src = new cv.Mat(64, 64, cv.CV_8UC1, new cv.Scalar(150));
  const low = OPS.fft_filter.apply(cv, src, { type: 'low', shape: 'gaussian', cutoff: 200 }, {});
  const mmLow = cv.minMaxLoc(low);
  assert.ok(mmLow.maxVal - mmLow.minVal < 20, `expected low-pass on a flat image to stay near-constant, range=${mmLow.maxVal - mmLow.minVal}`);
  low.delete();

  const high = OPS.fft_filter.apply(cv, src, { type: 'high', shape: 'gaussian', cutoff: 10 }, {});
  const mmHigh = cv.minMaxLoc(high);
  // high-pass removes the DC/near-DC energy that makes up ~all of a flat image's content
  assert.ok(mmHigh.maxVal - mmHigh.minVal < 40, `expected high-pass on a flat image to collapse toward a near-constant value, range=${mmHigh.maxVal - mmHigh.minVal}`);
  high.delete();
  src.delete();
});

test('B8c. FFT Filter: cutoff radius is resolution-independent (must NOT be rescaled by the >640px analysis downscale)', () => {
  // ภาพ 1280x720 (เกิน maxAnalysisSide=640 -> ถูกย่อวิเคราะห์ที่ scale=0.5) มีคลื่น sinusoid แนวนอน
  // 20 รอบเต็มภาพ (ความถี่ต่ำมาก เทียบกับรัศมีตัด 30 นับบนภาพจริงเสมอ) — ต้อง "รอด" การกรอง low-pass
  // ก่อนแก้บั๊ก cutoffScaled = cutoff*0.5 = 15 จะตัดความถี่นี้หายไปเกือบหมด (สนามคงที่ ~แบน)
  const w = 1280, h = 720, cycles = 20;
  const src = new cv.Mat(h, w, cv.CV_8UC1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = 128 + 100 * Math.sin((2 * Math.PI * cycles * x) / w);
      src.data[y * w + x] = Math.max(0, Math.min(255, Math.round(v)));
    }
  }
  const out = OPS.fft_filter.apply(cv, src, { type: 'low', shape: 'ideal', cutoff: 30 }, {});
  const mm = cv.minMaxLoc(out);
  const range = mm.maxVal - mm.minVal;
  assert.ok(range > 150, `expected the 20-cycle sinusoid to survive low-pass (cutoff=30, unscaled), range=${range} (old cutoff*0.5=15 code wipes this to near-flat)`);

  // เช็คว่ายังตามรูปคลื่นต้นฉบับ (correlate) ไม่ใช่แค่มี range กว้างเฉย ๆ โดยบังเอิญ
  let sumSrc = 0, sumOut = 0, n = w;
  const rowSrc = [], rowOut = [];
  for (let x = 0; x < w; x++) { rowSrc.push(src.data[x]); rowOut.push(out.data[x]); sumSrc += src.data[x]; sumOut += out.data[x]; }
  const meanSrc = sumSrc / n, meanOut = sumOut / n;
  let cov = 0, varSrc = 0, varOut = 0;
  for (let x = 0; x < w; x++) {
    const ds = rowSrc[x] - meanSrc, doo = rowOut[x] - meanOut;
    cov += ds * doo; varSrc += ds * ds; varOut += doo * doo;
  }
  const corr = cov / Math.sqrt(varSrc * varOut);
  assert.ok(corr > 0.8, `expected low-pass output to correlate with the input sinusoid, got corr=${corr}`);

  out.delete();
  src.delete();
});

test('B8d. FFT ops: extreme aspect ratio (2000x1) must not throw (downscaled side must clamp to >=1, not round to 0)', () => {
  const wide = new cv.Mat(1, 2000, cv.CV_8UC1, new cv.Scalar(100));
  assert.doesNotThrow(() => {
    const m = OPS.fft_magnitude.apply(cv, wide, {}, {});
    m.delete();
    const f = OPS.fft_filter.apply(cv, wide, { type: 'low', shape: 'gaussian', cutoff: 30 }, {});
    f.delete();
  });
  wide.delete();

  const tall = new cv.Mat(2000, 1, cv.CV_8UC1, new cv.Scalar(100));
  assert.doesNotThrow(() => {
    const m = OPS.fft_magnitude.apply(cv, tall, {}, {});
    m.delete();
    const f = OPS.fft_filter.apply(cv, tall, { type: 'low', shape: 'gaussian', cutoff: 30 }, {});
    f.delete();
  });
  tall.delete();
});

test('registry: fft_filter param label no longer claims a fixed "<=640px analysis image" scale (cutoff is now resolution-independent)', () => {
  const cutoffParam = OPS.fft_filter.params.find((p) => p.key === 'cutoff');
  assert.ok(cutoffParam, 'expected a cutoff param on fft_filter');
  assert.ok(!cutoffParam.label.includes('640'), `label should no longer reference the analysis-image scale, got: "${cutoffParam.label}"`);
});

// ===================================================================
// §11 — Mat-leak sanity for every new op (B1-B8) + A2-touched geometry ops
// ===================================================================

// P-14: เดิมใช้ sentinel 1500x1500x3 (ต้องใหญ่กว่าก้อนที่เคยรั่ว เช่น fft_magnitude ~307KB/frame) + HEAP8.buffer.byteLength
// — ทั้งคู่ขึ้นกับที่อยู่/ขนาด heap ทั้งก้อน (byteLength โตทีละ 128MB จึงแทบไม่ขยับ) ตอนนี้ใช้ด่านมาตรฐาน assertLeakFree:
// wrapper-balance (ทุก wrapper ที่ op สร้างต้องถูก .delete()) + heap census (byte ที่จองได้ก่อน/หลัง ไม่ขึ้นกับที่อยู่ —
// leak 307KB/รอบ หรือ 64B/รอบ ก็เห็น) ชื่อเทสต์คงเดิม ("large sentinel + heap-size check" = ชื่อประวัติ)
test('leak probe: A2 (crop/resize/rotate) + B1-B7 new ops do not leak Mats (200 iterations each, 320x240, large sentinel + heap-size check)', () => {
  const src = new cv.Mat(240, 320, cv.CV_8UC3, new cv.Scalar(120, 130, 140));
  const gray = new cv.Mat.zeros(240, 320, cv.CV_8UC1);
  const square = gray.roi(new cv.Rect(20, 20, 100, 100));
  square.setTo(new cv.Scalar(255));
  square.delete();
  const tiny = gray.roi(new cv.Rect(200, 200, 5, 5));
  tiny.setTo(new cv.Scalar(255));
  tiny.delete();

  const runOnce = {
    'geometry/crop(A2)': (c) => { const o = OPS.crop.apply(c, src, { x: 10, y: 10, w: 50, h: 50 }, {}); o.delete(); },
    'geometry/resize(A2)': (c) => { const o = OPS.resize.apply(c, src, { scale: 50 }, {}); o.delete(); },
    'geometry/rotate(A2)': (c) => { const o = OPS.rotate.apply(c, src, { angle: 30, expand: true }, {}); o.delete(); },
    'sharpen/custom_kernel': (c) => { const o = OPS.custom_kernel.apply(c, src, { preset: 'sharpen', amount: 1 }, {}); o.delete(); },
    'edge/laplacian_edge': (c) => { const o = OPS.laplacian_edge.apply(c, gray, { ksize: 3 }, {}); o.delete(); },
    'edge/scharr_edge': (c) => { const o = OPS.scharr_edge.apply(c, gray, { dir: 'both' }, {}); o.delete(); },
    'morphology/clear_border': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'clear_border', shape: 'rect', ksize: 3, iterations: 1, minArea: 150 }, {}); o.delete(); },
    'morphology/remove_small': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'remove_small', shape: 'rect', ksize: 3, iterations: 1, minArea: 150 }, {}); o.delete(); },
    'shape_analysis/distance_transform': (c) => { const o = OPS.distance_transform.apply(c, gray, { maskSize: '5' }, {}); o.delete(); },
    'threshold/multi_otsu_threshold': (c) => { const o = OPS.multi_otsu_threshold.apply(c, gray, {}, {}); o.delete(); },
    'colour/white_balance': (c) => { const o = OPS.white_balance.apply(c, src, {}, {}); o.delete(); },
  };

  for (const [label, run] of Object.entries(runOnce)) {
    assertLeakFree(cv, label, run, { iters: 200, warmup: 20 });
  }

  gray.delete();
  src.delete();
});

test('leak probe: B8 FFT ops do not leak Mats (640x480 gray, 100 iterations each, large sentinel + heap-size check)', () => {
  const gray = new cv.Mat(480, 640, cv.CV_8UC1);
  for (let i = 0; i < gray.data.length; i++) gray.data[i] = (i * 29) % 256;

  assertLeakFree(cv, 'fft_magnitude', (c) => { const o = OPS.fft_magnitude.apply(c, gray, {}, {}); o.delete(); }, { iters: 100, warmup: 10 });
  assertLeakFree(cv, 'fft_filter', (c) => { const o = OPS.fft_filter.apply(c, gray, { type: 'low', shape: 'gaussian', cutoff: 30 }, {}); o.delete(); }, { iters: 100, warmup: 10 });

  gray.delete();
});

// ===================================================================
// §12 — Palette wiring sanity
// ===================================================================

test('wiring: every new grouped op is present in its FILTERS mode select options', () => {
  const groupedNewOps = [
    ['sharpen', 'custom_kernel'],
    ['edge_detection', 'laplacian_edge'],
    ['edge_detection', 'scharr_edge'],
    ['shape_analysis', 'distance_transform'],
    ['threshold', 'multi_otsu_threshold'],
    ['colour', 'white_balance'],
    ['frequency', 'fft_magnitude'],
    ['frequency', 'fft_filter'],
  ];
  for (const [groupId, opId] of groupedNewOps) {
    const modeParam = FILTERS[groupId].params.find((p) => p.key === 'mode');
    assert.ok(modeParam, `${groupId} must have a mode select`);
    assert.ok(modeParam.options.some((o) => o.value === opId), `${groupId}'s mode select must include ${opId}`);
  }
});

test('wiring: clear_border/remove_small are present in morphology\'s own flat op select', () => {
  const opParam = FILTERS.morphology.params.find((p) => p.key === 'op');
  assert.ok(opParam, 'morphology must have an op select');
  assert.ok(opParam.options.some((o) => o.value === 'clear_border'));
  assert.ok(opParam.options.some((o) => o.value === 'remove_small'));
});

test('wiring: all new ops appear in TOOL_SECTIONS (palette) and resolve to a registered FILTERS group', () => {
  const paletteOps = flattenPaletteOps();
  const newOps = [
    'custom_kernel', 'laplacian_edge', 'scharr_edge', 'distance_transform',
    'multi_otsu_threshold', 'white_balance', 'fft_magnitude', 'fft_filter',
  ];
  for (const opId of newOps) {
    assert.ok(paletteOps.includes(opId), `expected ${opId} to be listed in some TOOL_SECTIONS entry`);
  }
  assert.ok(FILTERS.frequency, 'expected a registered "frequency" FILTERS group');
  assert.equal(FILTERS.frequency.group, 'Frequency Domain');
});

// ===================================================================
// CV-engineer final recheck fixes (qa's 3 open questions from Tier 1.5)
// ===================================================================

test('qa-Q(a). Custom Kernel "outline" preset hint correctly states the kernel sums to 1 (not 0) — matches the actual kernel matrix', () => {
  const outlineOption = OPS.custom_kernel.params
    .find((p) => p.key === 'preset').options
    .find((o) => o.value === 'outline');
  assert.ok(outlineOption, 'expected an "outline" preset option');
  // The hint text must not claim "sum = 0" since the real kernel [-1x8, center 9] sums to 1.
  assert.doesNotMatch(outlineOption.hint, /ผลรวม\s*0/, 'hint must not claim the outline kernel sums to 0');
  assert.match(outlineOption.hint, /ผลรวม.*=?\s*1|เพื่อนบ้านครบ 8/, 'hint should describe the kernel accurately (sum=1 / uses all 8 neighbors)');

  // Lock in the actual kernel math itself, independently of the hint text, so this test still catches
  // a regression even if someone only edits the matrix later without touching the hint again.
  const src = new cv.Mat(20, 20, cv.CV_8UC1, new cv.Scalar(100)); // flat/constant image
  const ctx = {};
  const out = OPS.custom_kernel.apply(cv, src, { preset: 'outline', amount: 1 }, ctx);
  // A kernel that sums to 1 leaves a perfectly flat/constant image unchanged (no edges to detect,
  // and no brightness drift since the weights sum to 1) — this is the empirical proof the sum is 1, not 0
  // (a sum=0 kernel would instead flatten a constant image toward 0, not preserve its value).
  const center = out.ucharPtr(10, 10)[0];
  assert.equal(center, 100, 'a sum=1 kernel must preserve a flat image\'s value exactly; sum=0 would push it toward 0');
  out.delete();
  src.delete();
});

test('qa-Q(c). fft_filter cutoff wording no longer says "px" (registry ctx.info + ui-meta OP_PRIMARY) — cutoff is a frequency-domain radius, not spatial pixels', () => {
  const src = new cv.Mat.zeros(100, 100, cv.CV_8UC1);
  cv.rectangle(src, new cv.Point(20, 20), new cv.Point(80, 80), new cv.Scalar(255), -1);
  const ctx = {};
  const out = OPS.fft_filter.apply(cv, src, { type: 'low', shape: 'gaussian', cutoff: 30 }, ctx);
  assert.doesNotMatch(ctx.info, /\d+px/, 'ctx.info must not word the cutoff value with a "px" suffix (misleading — it is a frequency-domain radius, not spatial pixels)');
  assert.match(ctx.info, /หน่วยความถี่/, 'ctx.info should describe the cutoff as a frequency-domain unit');
  out.delete();
  src.delete();

  const primaryText = OP_PRIMARY.fft_filter({ type: 'low', shape: 'gaussian', cutoff: 30 });
  assert.doesNotMatch(primaryText, /\d+px/, 'ui-meta OP_PRIMARY.fft_filter must not word the cutoff with a "px" suffix either');
});

test('qa-Q(b). fft_magnitude/fft_filter generated Python mirrors the app\'s >640px analysis-downscale, so live-app and generated-code results match at any resolution (not just <=640px)', () => {
  const magCode = buildPythonCode([{ id: 'frequency', params: { mode: 'fft_magnitude' } }]);
  assert.match(magCode, /640 \/ max\(rows, cols\)/, 'fft_magnitude codegen must mirror the app\'s maxAnalysisSide=640 downscale');
  // Must use round-half-up (int(x + 0.5)), matching JS Math.round exactly — Python's round() is
  // banker's rounding (round-half-to-even) and disagrees with Math.round on x.5 boundaries
  // (e.g. 1280x721 -> 360.5: round()=360 but Math.round=361), which shifted the analysis size
  // by 1px and threw off the padded DFT size (measured mean diff 17.4 vs the live app).
  assert.match(magCode, /cv2\.resize\(img, \(max\(1, int\(cols \* scale \+ 0\.5\)\), max\(1, int\(rows \* scale \+ 0\.5\)\)\), interpolation=cv2\.INTER_AREA\)/);
  assert.doesNotMatch(magCode, /round\(cols \* scale\)/, 'fft_magnitude codegen must not use round() for the downscale (banker\'s rounding disagrees with JS Math.round)');
  // JS pads the optimal DFT size up to even before the quadrant-swap/mask centring
  // (`if (optRows % 2 !== 0) optRows += 1;`) — Python must mirror this or odd optimal sizes
  // get a mask/spectrum centred half a pixel off (measured max diff 4).
  assert.match(magCode, /opt_rows \+= opt_rows % 2/, 'fft_magnitude codegen must pad opt_rows to even, like JS');
  assert.match(magCode, /opt_cols \+= opt_cols % 2/, 'fft_magnitude codegen must pad opt_cols to even, like JS');

  const filterCode = buildPythonCode([{ id: 'frequency', params: { mode: 'fft_filter', fft_filter__type: 'low', fft_filter__shape: 'gaussian', fft_filter__cutoff: 30 } }]);
  assert.match(filterCode, /640 \/ max\(rows, cols\)/, 'fft_filter codegen must mirror the same downscale');
  assert.match(filterCode, /cv2\.resize\(img, \(max\(1, int\(cols \* scale \+ 0\.5\)\), max\(1, int\(rows \* scale \+ 0\.5\)\)\), interpolation=cv2\.INTER_AREA\)/);
  assert.doesNotMatch(filterCode, /round\(cols \* scale\)/, 'fft_filter codegen must not use round() for the downscale either');
  assert.match(filterCode, /opt_rows \+= opt_rows % 2/, 'fft_filter codegen must pad opt_rows to even, like JS');
  assert.match(filterCode, /opt_cols \+= opt_cols % 2/, 'fft_filter codegen must pad opt_cols to even, like JS');
  assert.match(filterCode, /img = cv2\.resize\(inv, \(cols, rows\)\)/, 'fft_filter must upscale the filtered result back to the original size, like the app does');
});

test('CV-recheck finding: fft_filter desc documents border/wraparound ringing as an inherent FFT limitation (found during learner-pipeline recheck — a smooth gradient with a large edge-to-edge discontinuity produces ringing at the image border even with the Gaussian shape, due to DFT periodic-boundary assumption, not a code bug)', () => {
  assert.match(OPS.fft_filter.desc, /periodic|วนซ้ำ/, 'desc should mention the FFT periodic-boundary caveat');
  assert.match(OPS.fft_filter.desc, /ริมภาพ|ขอบ.*ภาพ/, 'desc should mention the border/edge nature of this artifact');

  // Empirical regression proof of the phenomenon itself (documents expected behavior, not a bug):
  // a pure linear gradient with a large left-right discontinuity produces near-perfect smooth output
  // in the interior but visible ringing confined to a handful of pixels at the left/right border.
  const w = 128, h = 32;
  const src = new cv.Mat(h, w, cv.CV_8UC1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) src.ucharPtr(y, x)[0] = Math.round((x / w) * 200 + 20);
  const out = OPS.fft_filter.apply(cv, src, { type: 'low', shape: 'gaussian', cutoff: 15 }, {});
  // interior (columns 10..117) must be monotonically non-decreasing (a genuine low-pass of a ramp stays a ramp)
  let interiorMonotonic = true;
  for (let x = 10; x < w - 10; x++) {
    if (out.ucharPtr(16, x + 1)[0] < out.ucharPtr(16, x)[0] - 2) { interiorMonotonic = false; break; }
  }
  assert.ok(interiorMonotonic, 'interior of the gradient should stay smooth/monotonic — border ringing must not leak into the interior');
  out.delete();
  src.delete();
});

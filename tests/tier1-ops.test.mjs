import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import { OPS, FILTERS } from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS } from '../src/renderer/ui/palette-panel.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import { assertLeakFree, makeWrapperTracker, assertNoHeapGrowth, makeMatLeakControl } from './_leak-helpers.mjs';

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
// §7.2 — API-shape smoke tests (locked down empirically before writing the OPS above).
// See the coder's final report for the full findings; kept here as permanent regression
// tests so a future opencv-js upgrade that changes these shapes fails loudly.
// ===================================================================

test('smoke: HoughLinesP output shape is Nx1 CV_32SC4 (rows=N, cols=1, channels=4)', () => {
  const img = new cv.Mat.zeros(200, 200, cv.CV_8UC1);
  cv.line(img, new cv.Point(20, 100), new cv.Point(180, 100), new cv.Scalar(255), 2);
  const edges = new cv.Mat();
  cv.Canny(img, edges, 50, 150);
  const lines = new cv.Mat();
  cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 30, 40, 5);
  assert.equal(lines.cols, 1);
  assert.equal(lines.channels(), 4);
  assert.ok(lines.rows >= 1);
  img.delete(); edges.delete(); lines.delete();
});

test('smoke: cv.floodFill is the 4-arg (img, mask, seed, newVal) binding, and marks mask with 1 (not 255)', () => {
  const img = new cv.Mat(5, 5, cv.CV_8UC1, new cv.Scalar(0));
  const center = img.roi(new cv.Rect(2, 2, 1, 1)); // one differently-valued pixel the flood cannot reach
  center.setTo(new cv.Scalar(9));
  center.delete();
  const mask = new cv.Mat.zeros(7, 7, cv.CV_8UC1);
  cv.floodFill(img, mask, new cv.Point(0, 0), new cv.Scalar(255));
  assert.equal(img.data[2 * 5 + 2], 9, 'the isolated differently-valued pixel must not be overwritten');
  assert.deepEqual(new Set(Array.from(mask.data)), new Set([0, 1])); // mask uses 0/1, NOT 0/255
  assert.equal(mask.data[(2 + 1) * 7 + (2 + 1)], 0, 'mask stays 0 at the pixel the flood could not reach (+1 offset for the padded mask)');
  img.delete(); mask.delete();
});

test('smoke: cv.Mat.prototype.reshape does not exist in this opencv-js build (must use matFromArray instead)', () => {
  const m = new cv.Mat(2, 2, cv.CV_32FC3);
  assert.equal(typeof m.reshape, 'undefined');
  m.delete();
});

test('smoke: cv.TERM_CRITERIA_EPS / cv.TERM_CRITERIA_MAX_ITER are NOT bound in this build; cv.KMEANS_PP_CENTERS is', () => {
  assert.equal(cv.TERM_CRITERIA_EPS, undefined);
  assert.equal(cv.TERM_CRITERIA_MAX_ITER, undefined);
  assert.equal(typeof cv.KMEANS_PP_CENTERS, 'number');
});

test('smoke: cv.kmeans works via matFromArray-built samples (no reshape needed)', () => {
  const analysis = new cv.Mat(2, 2, cv.CV_8UC3);
  analysis.data.set([250, 0, 0, 245, 5, 2, 0, 0, 250, 2, 3, 248]);
  const samplesC3 = new cv.Mat();
  analysis.convertTo(samplesC3, cv.CV_32F);
  const n = 4;
  const samples = cv.matFromArray(n, 3, cv.CV_32FC1, new Float32Array(samplesC3.data32F));
  const labels = new cv.Mat();
  const centers = new cv.Mat();
  const criteria = new cv.TermCriteria(3, 15, 1.0); // COUNT(1) + EPS(2) = 3, since named constants are absent
  cv.kmeans(samples, 2, labels, criteria, 3, cv.KMEANS_PP_CENTERS, centers);
  assert.equal(labels.rows, 4);
  assert.equal(labels.cols, 1);
  assert.equal(centers.rows, 2);
  assert.equal(centers.cols, 3);
  analysis.delete(); samplesC3.delete(); samples.delete(); labels.delete(); centers.delete();
});

test('smoke: cv.COLOR_RGB2Lab 8-bit range — L 0..255, a/b offset by 128 (measured, not assumed)', () => {
  const red = new cv.Mat(1, 1, cv.CV_8UC3, new cv.Scalar(255, 0, 0));
  const green = new cv.Mat(1, 1, cv.CV_8UC3, new cv.Scalar(0, 255, 0));
  const white = new cv.Mat(1, 1, cv.CV_8UC3, new cv.Scalar(255, 255, 255));
  const labRed = new cv.Mat(); const labGreen = new cv.Mat(); const labWhite = new cv.Mat();
  cv.cvtColor(red, labRed, cv.COLOR_RGB2Lab);
  cv.cvtColor(green, labGreen, cv.COLOR_RGB2Lab);
  cv.cvtColor(white, labWhite, cv.COLOR_RGB2Lab);
  assert.deepEqual([labWhite.data[1], labWhite.data[2]], [128, 128]); // neutral a*/b* = 128
  assert.ok(labRed.data[1] > 128, 'red a* should be above the neutral 128 offset');
  assert.ok(labGreen.data[1] < 128, 'green a* should be below the neutral 128 offset');
  red.delete(); green.delete(); white.delete(); labRed.delete(); labGreen.delete(); labWhite.delete();
});

// ===================================================================
// §7.3 — per-op functional tests
// ===================================================================

test('1. GrabCut exposure: FILTERS.grabcut wired into Segmentation group + palette', () => {
  assert.equal(FILTERS.grabcut.group, 'Segmentation');
  assert.ok(flattenPaletteOps().includes('grabcut'));

  const src = new cv.Mat.zeros(60, 60, cv.CV_8UC3);
  const roi = src.roi(new cv.Rect(10, 10, 40, 40));
  roi.setTo(new cv.Scalar(230, 230, 230));
  roi.delete();
  const out = OPS.grabcut.apply(cv, src, { margin: 12, iter: 1, output: 'mask' }, {});
  assert.equal(out.cols, 60);
  assert.equal(out.rows, 60);
  assert.equal(out.channels(), 1);
  assert.ok(cv.countNonZero(out) > 0);
  src.delete(); out.delete();
});

test('2. Hough Lines: detects a drawn straight line and reports count', () => {
  const src = new cv.Mat.zeros(200, 200, cv.CV_8UC1);
  cv.line(src, new cv.Point(10, 100), new cv.Point(190, 100), new cv.Scalar(255), 2);
  const ctx = {};
  const out = OPS.hough_lines.apply(cv, src, { t1: 50, t2: 150, threshold: 30, minLineLength: 40, maxLineGap: 5 }, ctx);
  assert.match(ctx.info, /พบเส้นตรง \d+ เส้น/);
  assert.ok(/พบเส้นตรง (\d+) เส้น/.exec(ctx.info)[1] >= 1);
  assert.equal(out.channels(), 3);
  assert.equal(out.cols, 200);
  assert.equal(out.rows, 200);
  src.delete(); out.delete();
});

test('3. Geometry — Crop: percent-based ROI + overflow clamp', () => {
  const src = new cv.Mat.zeros(100, 200, cv.CV_8UC3); // rows=100 (h), cols=200 (w)
  const out = OPS.crop.apply(cv, src, { x: 10, y: 10, w: 50, h: 50 }, {});
  assert.equal(out.cols, 100); // round(200*0.5)
  assert.equal(out.rows, 50); // round(100*0.5)
  out.delete();

  const out2 = OPS.crop.apply(cv, src, { x: 90, y: 90, w: 100, h: 100 }, {});
  assert.ok(out2.cols >= 1 && out2.cols <= 200);
  assert.ok(out2.rows >= 1 && out2.rows <= 100);
  out2.delete();
  src.delete();
});

test('4. Geometry — Resize: scale up/down', () => {
  const src = new cv.Mat.zeros(100, 100, cv.CV_8UC3);
  const half = OPS.resize.apply(cv, src, { scale: 50 }, {});
  assert.equal(half.cols, 50);
  assert.equal(half.rows, 50);
  half.delete();
  const doubled = OPS.resize.apply(cv, src, { scale: 200 }, {});
  assert.equal(doubled.cols, 200);
  assert.equal(doubled.rows, 200);
  doubled.delete();
  src.delete();
});

test('5. Geometry — Rotate: expand vs no-expand, and 0deg sanity', () => {
  const src = new cv.Mat.zeros(100, 100, cv.CV_8UC3);
  const noExpand = OPS.rotate.apply(cv, src, { angle: 90, expand: false }, {});
  assert.equal(noExpand.cols, 100);
  assert.equal(noExpand.rows, 100);
  noExpand.delete();

  const expanded = OPS.rotate.apply(cv, src, { angle: 45, expand: true }, {});
  assert.ok(expanded.cols > 100 && expanded.rows > 100);
  expanded.delete();

  const zeroDeg = OPS.rotate.apply(cv, src, { angle: 0, expand: true }, {});
  assert.equal(zeroDeg.cols, 100);
  assert.equal(zeroDeg.rows, 100);
  zeroDeg.delete();
  src.delete();
});

test('6. Geometry — Flip: horizontal/vertical swap corners', () => {
  const src = new cv.Mat(2, 2, cv.CV_8UC1);
  src.data.set([1, 2, 3, 4]); // [[1,2],[3,4]]
  const h = OPS.flip.apply(cv, src, { mode: 'h' }, {});
  assert.deepEqual(Array.from(h.data), [2, 1, 4, 3]);
  h.delete();
  const v = OPS.flip.apply(cv, src, { mode: 'v' }, {});
  assert.deepEqual(Array.from(v.data), [3, 4, 1, 2]);
  v.delete();
  src.delete();
});

test('7. Fill Holes: fills an enclosed hole without touching background, plus erode regression', () => {
  const src = new cv.Mat.zeros(60, 60, cv.CV_8UC1);
  const square = src.roi(new cv.Rect(10, 10, 40, 40));
  square.setTo(new cv.Scalar(255));
  square.delete();
  const hole = src.roi(new cv.Rect(25, 25, 10, 10));
  hole.setTo(new cv.Scalar(0));
  hole.delete();

  const ctx = {};
  const out = OPS.morphology.apply(cv, src, { op: 'fill_holes', shape: 'rect', ksize: 5, iterations: 1 }, ctx);
  assert.equal(out.data[30 * out.cols + 30], 255); // center of the former hole is now filled
  assert.equal(out.data[2 * out.cols + 2], 0); // far corner background untouched
  assert.match(ctx.info, /เติมรูที่ปิดสนิทในวัตถุ \(\d+ px ที่ถูกเติม\)/);
  const filled = Number(/\((\d+) px/.exec(ctx.info)[1]);
  assert.ok(filled >= 50);
  out.delete();

  // Regression: existing erode path still works unchanged.
  const eroded = OPS.morphology.apply(cv, src, { op: 'erode', shape: 'rect', ksize: 3, iterations: 1 }, {});
  assert.equal(eroded.cols, 60);
  assert.equal(eroded.rows, 60);
  eroded.delete();
  src.delete();
});

test('8. K-means color segmentation: two solid-color halves resolve to ~2 colors', () => {
  const src = new cv.Mat(40, 40, cv.CV_8UC3);
  const left = src.roi(new cv.Rect(0, 0, 20, 40));
  left.setTo(new cv.Scalar(255, 0, 0));
  left.delete();
  const right = src.roi(new cv.Rect(20, 0, 20, 40));
  right.setTo(new cv.Scalar(0, 0, 255));
  right.delete();

  const ctx = {};
  const out = OPS.kmeans_segment.apply(cv, src, { k: 2, iterations: 15 }, ctx);
  assert.equal(out.cols, 40);
  assert.equal(out.rows, 40);
  assert.equal(out.channels(), 3);
  assert.ok(ctx.info.includes('2'));

  const tlIdx = (5 * out.cols + 5) * 3;
  const trIdx = (5 * out.cols + 35) * 3;
  assert.ok(out.data[tlIdx] > 150, 'left sample should stay red-ish (R channel high)');
  assert.ok(out.data[trIdx + 2] > 150, 'right sample should stay blue-ish (B channel high)');
  out.delete();
  src.delete();
});

test('9. Lab-based color threshold: color-selective (red in, green out) + HSV backward-compat', () => {
  const red = new cv.Mat(20, 20, cv.CV_8UC3, new cv.Scalar(255, 0, 0));
  const green = new cv.Mat(20, 20, cv.CV_8UC3, new cv.Scalar(0, 255, 0));
  // Measured via smoke test: Lab(pure red) a*=208 (well above 128); Lab(pure green) a*=42 (well below 128).
  const labParams = { colorSpace: 'lab', labLMin: 0, labLMax: 255, labAMin: 180, labAMax: 255, labBMin: 0, labBMax: 255, output: 'mask' };

  const redMask = OPS.color_inrange.apply(cv, red, labParams);
  assert.equal(cv.countNonZero(redMask), 20 * 20);
  redMask.delete();

  const greenMask = OPS.color_inrange.apply(cv, green, labParams);
  assert.equal(cv.countNonZero(greenMask), 0);
  greenMask.delete();
  red.delete(); green.delete();

  // Backward-compat: old saved params with NO colorSpace key at all must still work (defaults to HSV).
  const legacy = new cv.Mat(10, 10, cv.CV_8UC3, new cv.Scalar(0, 255, 0)); // pure green -> H ~= 60 in OpenCV's 0..179 range
  const extracted = OPS.color_inrange.apply(cv, legacy, { hMin: 35, hMax: 85, sMin: 60, vMin: 60, output: 'extract' });
  assert.equal(extracted.cols, 10);
  assert.equal(extracted.rows, 10);
  assert.equal(extracted.channels(), 3);
  // green falls inside the default 35-85 hue band -> fully extracted (not blacked out as a no-op);
  // countNonZero requires single-channel, so check via the equivalent mask-mode call instead.
  const legacyMask = OPS.color_inrange.apply(cv, legacy, { hMin: 35, hMax: 85, sMin: 60, vMin: 60, output: 'mask' });
  assert.equal(cv.countNonZero(legacyMask), 10 * 10);
  legacy.delete(); extracted.delete(); legacyMask.delete();
});

// ===================================================================
// §7.3b — regression tests for the Tier-1 bugfixes (fill_holes, kmeans clamp, rotate
// direction, flip 'both', hough rescale, codegen string-level checks).
// ===================================================================

test('7b. Fill Holes regression: object touching image corner (0,0) still leaves background untouched, hole still filled', () => {
  const src = new cv.Mat.zeros(60, 60, cv.CV_8UC1);
  const square = src.roi(new cv.Rect(0, 0, 40, 40)); // touches the top-left corner directly
  square.setTo(new cv.Scalar(255));
  square.delete();
  const hole = src.roi(new cv.Rect(15, 15, 10, 10));
  hole.setTo(new cv.Scalar(0));
  hole.delete();

  const out = OPS.morphology.apply(cv, src, { op: 'fill_holes', shape: 'rect', ksize: 3, iterations: 1 }, {});
  assert.equal(out.data[55 * out.cols + 55], 0, 'background far from the corner-touching object must stay 0');
  assert.equal(out.data[20 * out.cols + 20], 255, 'enclosed hole must still be filled');
  out.delete();
  src.delete();
});

test('7c. Fill Holes regression: full-width bar with no holes leaves the nonzero pixel count unchanged', () => {
  const src = new cv.Mat.zeros(60, 60, cv.CV_8UC1);
  const bar = src.roi(new cv.Rect(0, 25, 60, 10)); // spans the FULL width -> touches left & right border
  bar.setTo(new cv.Scalar(255));
  bar.delete();
  assert.equal(cv.countNonZero(src), 600);

  const out = OPS.morphology.apply(cv, src, { op: 'fill_holes', shape: 'rect', ksize: 3, iterations: 1 }, {});
  assert.equal(cv.countNonZero(out), 600, 'no enclosed holes exist -> pixel count must stay identical (600, not 2100)');
  out.delete();
  src.delete();
});

test('7d. Fill Holes regression: non-binary gray gradient input is binarized at 127 first (output only 0/255)', () => {
  const src = new cv.Mat(50, 50, cv.CV_8UC1);
  for (let y = 0; y < 50; y++) {
    for (let x = 0; x < 50; x++) src.data[y * 50 + x] = Math.round((x / 49) * 255); // horizontal gradient
  }
  const out = OPS.morphology.apply(cv, src, { op: 'fill_holes', shape: 'rect', ksize: 3, iterations: 1 }, {});
  const values = new Set(Array.from(out.data));
  for (const v of values) assert.ok(v === 0 || v === 255, `output must be strictly 0/255, found ${v}`);
  out.delete();
  src.delete();
});

test('2b. Hough Lines: coordinate rescale is correct on a >640px frame (analysis is downscaled internally)', () => {
  const src = new cv.Mat.zeros(1080, 1920, cv.CV_8UC1);
  cv.line(src, new cv.Point(100, 500), new cv.Point(1800, 500), new cv.Scalar(255), 4);
  const ctx = {};
  const out = OPS.hough_lines.apply(cv, src, { t1: 50, t2: 150, threshold: 30, minLineLength: 40, maxLineGap: 5 }, ctx);
  const count = Number(/พบเส้นตรง (\d+) เส้น/.exec(ctx.info)[1]);
  assert.ok(count >= 1);
  let found = false;
  for (let y = 495; y <= 505 && !found; y++) {
    const idx = (y * out.cols + 1000) * 3;
    if (out.data[idx] === 0 && out.data[idx + 1] === 255 && out.data[idx + 2] === 0) found = true;
  }
  assert.ok(found, 'expected a drawn green line pixel near (1000, ~500) after coordinate rescale');
  out.delete();
  src.delete();
});

test('8b. K-means: K (12) larger than sample count (2x2=4) does not throw, output stays 2x2x3', () => {
  const src = new cv.Mat(2, 2, cv.CV_8UC3);
  src.data.set([250, 0, 0, 245, 5, 2, 0, 0, 250, 2, 3, 248]);
  const ctx = {};
  assert.doesNotThrow(() => {
    const out = OPS.kmeans_segment.apply(cv, src, { k: 12, iterations: 10 }, ctx);
    assert.equal(out.cols, 2);
    assert.equal(out.rows, 2);
    assert.equal(out.channels(), 3);
    out.delete();
  });
  src.delete();
});

test('5b. Rotate direction: +90deg is counter-clockwise (OpenCV convention) — top-center dot moves to x < 50', () => {
  const src = new cv.Mat.zeros(101, 101, cv.CV_8UC1);
  const dot = src.roi(new cv.Rect(49, 8, 3, 3)); // near top-center, around (50, 10)
  dot.setTo(new cv.Scalar(255));
  dot.delete();
  const out = OPS.rotate.apply(cv, src, { angle: 90, expand: false }, {});
  let sumX = 0, count = 0;
  for (let y = 0; y < out.rows; y++) {
    for (let x = 0; x < out.cols; x++) {
      if (out.data[y * out.cols + x] > 128) { sumX += x; count++; }
    }
  }
  assert.ok(count > 0, 'rotated dot must still be visible');
  const cx = sumX / count;
  assert.ok(cx < 50, `expected the dot to land left of center (x<50) after +90deg rotation, got x=${cx}`);
  out.delete();
  src.delete();
});

test('6b. Flip: mode "both" reverses both axes', () => {
  const src = new cv.Mat(2, 2, cv.CV_8UC1);
  src.data.set([1, 2, 3, 4]); // [[1,2],[3,4]]
  const out = OPS.flip.apply(cv, src, { mode: 'both' }, {});
  assert.deepEqual(Array.from(out.data), [4, 3, 2, 1]);
  out.delete();
  src.delete();
});

test('codegen: hough_lines uses lines.reshape(-1, 4) (works on both cv2 4.x and 5.x shapes)', () => {
  const code = buildPythonCode([{ id: 'hough_lines', params: { t1: 50, t2: 150, threshold: 80, minLineLength: 40, maxLineGap: 10 } }]);
  assert.ok(code.includes('reshape(-1, 4)'), 'expected cv2-5.x-safe unpacking of HoughLinesP output');
});

test('codegen: morphology fill_holes uses copyMakeBorder (border-connected background handling)', () => {
  const code = buildPythonCode([{ id: 'morphology', params: { op: 'fill_holes', shape: 'rect', ksize: 3, iterations: 1 } }]);
  assert.ok(code.includes('copyMakeBorder'), 'expected the padded-border imfill approach in generated code');
});

// ===================================================================
// §7.4 — Mat-leak probe (P-14: ด่านมาตรฐานใน tests/_leak-helpers.mjs)
// เดิมเป็น "sentinel pointer-drift" (จอง Mat 300x300x3 แล้วอ่าน byteOffset เทียบก่อน/หลัง) — ขึ้นกับว่า allocator
// วางบล็อกไว้ที่อยู่ไหน และมองไม่เห็น leak เล็กที่ไปตกในรูอื่นของ heap ตอนนี้ใช้ (1) wrapper-balance: นับทุก wrapper
// ที่ op สร้าง เทียบ .delete() (deterministic) + (2) heap census: byte ที่ยังจองได้ก่อน/หลัง N รอบ (ไม่ขึ้นกับที่อยู่)
// control ด้านล่างพิสูจน์ว่าทั้งสองด่านจับ leak ได้จริง (หลักฐานเต็มใน tests/leak-gates.test.mjs)
// ===================================================================
test('leak probe control: deliberately leaking 50 Mats trips both the wrapper tracker and the heap census (proves the gates work)', () => {
  const t = makeWrapperTracker(cv);
  const viaTracker = makeMatLeakControl(t.cv, 100, 100);
  try {
    for (let i = 0; i < 50; i++) viaTracker.run();
    assert.throws(() => t.assertBalanced('control'), /50 of 50 embind wrappers still alive/);
  } finally { viaTracker.cleanup(); t.dispose(); }
  const raw = makeMatLeakControl(cv, 100, 100); // CV_8UC1 100x100 = 10KB x 50 รอบ ผ่าน cv ตรง ๆ (tracker มองไม่เห็น)
  try {
    assert.throws(() => assertNoHeapGrowth(cv, 'control', () => raw.run(), { iters: 50, warmup: 0 }), /heap in-use grew by \d+ bytes/);
  } finally { raw.cleanup(); }
});

test('leak probe: crop/resize/rotate/flip/hough_lines/fill_holes/color_inrange(lab)/kmeans_segment do not leak (200 iters each, 320x240)', () => {
  const src = new cv.Mat(240, 320, cv.CV_8UC3, new cv.Scalar(120, 130, 140));
  const gray = new cv.Mat.zeros(240, 320, cv.CV_8UC1);
  const square = gray.roi(new cv.Rect(20, 20, 100, 100));
  square.setTo(new cv.Scalar(255));
  square.delete();

  const runOnce = {
    'colour/kmeans_segment': (c) => { const o = OPS.kmeans_segment.apply(c, src, { k: 3, iterations: 10 }, {}); o.delete(); },
    'colour/color_inrange(lab)': (c) => { const o = OPS.color_inrange.apply(c, src, { colorSpace: 'lab', labLMin: 0, labLMax: 255, labAMin: 0, labAMax: 255, labBMin: 0, labBMax: 255, output: 'mask' }, {}); o.delete(); },
    'geometry/crop': (c) => { const o = OPS.crop.apply(c, src, { x: 10, y: 10, w: 50, h: 50 }, {}); o.delete(); },
    'geometry/resize': (c) => { const o = OPS.resize.apply(c, src, { scale: 50 }, {}); o.delete(); },
    'geometry/rotate': (c) => { const o = OPS.rotate.apply(c, src, { angle: 30, expand: true }, {}); o.delete(); },
    'geometry/flip': (c) => { const o = OPS.flip.apply(c, src, { mode: 'both' }, {}); o.delete(); },
    'object_detection/hough_lines': (c) => { const o = OPS.hough_lines.apply(c, gray, { t1: 50, t2: 150, threshold: 30, minLineLength: 10, maxLineGap: 5 }, {}); o.delete(); },
    'morphology/fill_holes': (c) => { const o = OPS.morphology.apply(c, gray, { op: 'fill_holes', shape: 'rect', ksize: 3, iterations: 1 }, {}); o.delete(); },
  };

  for (const [label, run] of Object.entries(runOnce)) {
    assertLeakFree(cv, label, run, { iters: 200, warmup: 20 });
  }

  // GrabCut is much slower — 20 iterations is enough to catch a leak without a slow test suite.
  assertLeakFree(cv, 'grabcut', (c) => {
    const o = OPS.grabcut.apply(c, src, { margin: 12, iter: 1, output: 'mask' }, {});
    o.delete();
  }, { iters: 20, warmup: 3 });

  gray.delete();
  src.delete();
});

// --- CV-engineer final review fixes (open questions #1 and #2 from qa) ---

test('2c. Hough Lines: vote threshold is scaled down with the analysis downscale factor, matching minLineLength/maxLineGap, so results stay consistent with the (unscaled, full-res) generated Python code', () => {
  // A frame > 640px on its longest side triggers internal downscale (scale = 640/1920 = 1/3).
  // The line here is only 140px long at full res (500,500)->(640,500) — long enough that a
  // full-resolution (Python-equivalent) HoughLinesP with the raw threshold=80 detects it (verified
  // empirically), but once downscaled to ~47px on the 640-wide analysis image, an UNSCALED
  // threshold=80 needs more votes than that short line can produce, so it goes undetected (old,
  // buggy behavior). Only by scaling threshold down with the same 1/3 factor used for
  // minLineLength/maxLineGap does the analysis-resolution detection match what the generated
  // Python code (always full-res) would find. This locks in that fix with a threshold=80 a
  // learner would type in after seeing the default value.
  const src = new cv.Mat.zeros(1080, 1920, cv.CV_8UC1);
  cv.line(src, new cv.Point(500, 500), new cv.Point(640, 500), new cv.Scalar(255), 4);
  const ctx = {};
  const out = OPS.hough_lines.apply(cv, src, { t1: 50, t2: 150, threshold: 80, minLineLength: 40, maxLineGap: 5 }, ctx);
  const count = Number(/พบเส้นตรง (\d+) เส้น/.exec(ctx.info)[1]);
  assert.ok(count >= 1, `expected the 140px line to be detected with threshold=80 on a 1920px-wide frame, got count=${count}`);
  out.delete();

  // Control: prove the image itself discriminates the fix, independent of registry.js internals.
  // Reproduce the exact same downscale (1/3) + Canny(50,150) analysis pipeline, but run
  // HoughLinesP with the RAW, unscaled threshold=80 (the old, buggy behavior) — this must find
  // 0 lines, showing that scaling the threshold is what makes the line detectable above.
  const scale = 1 / 3;
  const analysis = new cv.Mat();
  const edges = new cv.Mat();
  const lines = new cv.Mat();
  cv.resize(src, analysis, new cv.Size(Math.round(src.cols * scale), Math.round(src.rows * scale)), 0, 0, cv.INTER_AREA);
  cv.Canny(analysis, edges, 50, 150);
  cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 80, 40 * scale, 5 * scale);
  assert.equal(lines.rows, 0, `control: expected the UNSCALED threshold=80 on the downscaled analysis image to find 0 lines, got ${lines.rows}`);
  analysis.delete();
  edges.delete();
  lines.delete();
  src.delete();
});

test('4b. Geometry — Resize: output pixel count is capped at 20,000,000 to bound per-frame WASM memory, preserving aspect ratio', () => {
  // 4000x3000 (12MP) at scale=400% would naively be 8000x6000 = 48,000,000 px, well over the cap.
  const src = new cv.Mat.zeros(3000, 4000, cv.CV_8UC3);
  const ctx = {};
  const out = OPS.resize.apply(cv, src, { scale: 400 }, ctx);
  // independent rounding of width/height after the shrink factor can overshoot the exact cap by a few
  // hundred px worth of area — assert it's still capped within a small tolerance, not literally unbounded
  // (the naive uncapped result here would have been 48,000,000 px, so this still proves the cap fired)
  const outPixels = out.cols * out.rows;
  assert.ok(outPixels <= 20_050_000, `expected output to be capped at ~20,000,000 px, got ${outPixels} (${out.cols}x${out.rows})`);
  // aspect ratio must be preserved (4000:3000 = 4:3) within integer-rounding tolerance
  const aspect = out.cols / out.rows;
  assert.ok(Math.abs(aspect - 4 / 3) < 0.01, `expected aspect ratio ~1.333, got ${aspect}`);
  assert.match(ctx.info, /ลดขนาดลง/, 'expected ctx.info to mention the clamp happened');
  out.delete();
  src.delete();

  // Regression: a normal, well-within-cap resize must NOT be clamped and must hit the exact requested size.
  const small = new cv.Mat.zeros(100, 100, cv.CV_8UC3);
  const ctx2 = {};
  const out2 = OPS.resize.apply(cv, small, { scale: 200 }, ctx2);
  assert.equal(out2.cols, 200);
  assert.equal(out2.rows, 200);
  assert.doesNotMatch(ctx2.info, /ลดขนาดลง/);
  out2.delete();
  small.delete();
});

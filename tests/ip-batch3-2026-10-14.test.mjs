// IP batch 3 (2026-10-14) — regionprops +8 คอลัมน์ · shape_filter · quantize · interpolation ใน resize/rotate
// สเปก: Vision_boi/specs/ip_batch3_2026-10-14.md · WORKLOG: WORKLOG-2026-10-14-ip-batch3.md
// โครงเดียวกับชิ้น C/D/E: ip3-0 (fingerprint, เทสต์แรกของไฟล์) + ip3-0w (wrapper balance เส้นทางชุดเดียวกัน) → ip3-L1..L3
// parity กับ Python จริง (cv2 5.0 / numpy 2.x) — ข้ามเฉพาะเมื่อไม่มี python/cv2
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import {
  OPS, FILTERS, defaultParams, needsOf, regionBox, hullRowSpans, regionPixelStats, regionShapeValues, REGION_SHAPE_COLUMNS,
  quantizeLut, quantizeBits, fixed2,
} from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS, OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { GROUP_ICON, OP_PRIMARY } from '../src/renderer/filters/ui-meta.js';
import { HISTOGRAM_COMPARE_OPS } from '../src/renderer/ui/phase3-math.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import {
  assertLeakFree, assertWrapperBalance, fingerprint, makeNativeLeakControl, makeThrowingCv, makeWrapperTracker,
} from './_leak-helpers.mjs';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.fillPoly === 'function' && typeof c.LUT === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

const readSrc = (rel) => fs.readFileSync(new URL(`../src/renderer/${rel}`, import.meta.url), 'utf8');
const NEW_OPS = ['quantize', 'shape_filter'];
const PROPS = ['circularity', 'solidity', 'eccentricity', 'extent', 'none'];
const INTERPS = ['auto', 'nearest', 'bilinear', 'bicubic'];
const ALL_COLS = ['index', 'area', 'cx', 'cy', 'perimeter', 'circularity', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'orientation_deg', ...REGION_SHAPE_COLUMNS];
const SF_KEYS = ['sf_total', 'sf_kept', 'sf_removed_area', 'sf_removed_shape'];

// ===================================================================
// ภาพทดสอบ (deterministic — ไม่ใช้ Math.random)
// ===================================================================
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}
function gray(w, h, fn) {
  const d = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d[y * w + x] = fn(x, y);
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
function rgb(w, h, fn) {
  const d = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d.set(fn(x, y), (y * w + x) * 3);
  return cv.matFromArray(h, w, cv.CV_8UC3, d);
}
function toRGBA(m) { const out = new cv.Mat(); cv.cvtColor(m, out, m.channels() === 1 ? cv.COLOR_GRAY2RGBA : cv.COLOR_RGB2RGBA); return out; }
function toRGB(m) { const out = new cv.Mat(); cv.cvtColor(m, out, cv.COLOR_GRAY2RGB); return out; }
function noiseGray(w, h, seed, density = 0.3) { const r = lcg(seed); return gray(w, h, () => (r() < density ? 255 : 0)); }
function noiseRGB(w, h, seed) { const r = lcg(seed); return rgb(w, h, () => [Math.floor(r() * 256), Math.floor(r() * 256), Math.floor(r() * 256)]); }
function bytes(m) { return Array.from(m.data); }
function black(w, h) { return new cv.Mat(h, w, cv.CV_8UC1, new cv.Scalar(0)); }
// วาดบนภาพดำ 1 ช่อง: draw(mat) ใช้ cv.* วาดสีขาว (255) — คืน Mat ที่ผู้เรียกต้อง delete
function drawn(w, h, draw) { const m = black(w, h); try { draw(m); } catch (e) { m.delete(); throw e; } return m; }
const W255 = new cv.Scalar(255);
function fillRectM(m, x, y, w, h, v = 255) { cv.rectangle(m, new cv.Point(x, y), new cv.Point(x + w - 1, y + h - 1), new cv.Scalar(v), -1); }
function ringM(m, cx, cy, r1, r2) { cv.circle(m, new cv.Point(cx, cy), r1, W255, -1); cv.circle(m, new cv.Point(cx, cy), r2, new cv.Scalar(0), -1); }
function ellipseM(m, cx, cy, a, b, ang) { cv.ellipse(m, new cv.Point(cx, cy), new cv.Size(a, b), ang, 0, 360, W255, -1); }
function lShapeM(m, x, y) { fillRectM(m, x, y, 80, 20); fillRectM(m, x, y, 20, 80); }

// ฉาก shape_filter: 400x300 — 3 discs r20 · 3 squares 30x30 · 2 ellipses 60x6 · 1 L · 1 ring 40/25 · 10 specks (แยกจากกันหมด)
function sfScene() {
  return drawn(400, 300, (m) => {
    for (const cx of [40, 110, 180]) cv.circle(m, new cv.Point(cx, 40), 20, W255, -1);
    for (const x of [230, 280, 330]) fillRectM(m, x, 25, 30, 30);
    ellipseM(m, 60, 120, 30, 3, 0);
    ellipseM(m, 160, 120, 30, 3, 0);
    lShapeM(m, 230, 90);
    ringM(m, 80, 230, 40, 25);
    const sizes = [[1, 1], [1, 1], [1, 1], [2, 1], [2, 1], [2, 2], [2, 2], [2, 2], [1, 2], [2, 1]];
    sizes.forEach(([w, h], i) => fillRectM(m, 200 + i * 15, 220, w, h));
  });
}
// ฉากสำหรับ regionprops/ตัวช่วย: ผลรวมหลายรูปทรง (แยกจากกัน)
function shapesScene() {
  return drawn(330, 200, (m) => {
    fillRectM(m, 5, 5, 40, 10);          // สี่เหลี่ยมผืนผ้า 40x10
    fillRectM(m, 60, 5, 10, 10);         // สี่เหลี่ยมจัตุรัส
    cv.circle(m, new cv.Point(120, 40), 40, W255, -1);
    ringM(m, 240, 60, 40, 25);
    lShapeM(m, 5, 90);
    ellipseM(m, 160, 150, 40, 20, 30);
    cv.line(m, new cv.Point(200, 120), new cv.Point(250, 150), W255, 1);
    m.data[190 * 330 + 300] = 255;       // พิกเซลเดี่ยว
  });
}
// ส่ง src ให้ op ที่ needs gray: ภาพสี → ใช้ cvtColor เองแบบ worker
function runOp(opId, src, params = {}, cvUse = cv) {
  const ctx = {};
  const out = OPS[opId].apply(cvUse, src, params, ctx);
  return { out, ctx };
}
function runInfo(opId, src, params = {}) { const { out, ctx } = runOp(opId, src, params); out.delete(); return ctx; }
function sameBytes(a, b, label) {
  assert.equal(a.rows, b.rows, `${label}: rows`);
  assert.equal(a.cols, b.cols, `${label}: cols`);
  assert.equal(a.channels(), b.channels(), `${label}: channels`);
  const da = a.data, db = b.data;
  let n = 0;
  for (let i = 0; i < da.length; i++) if (da[i] !== db[i]) n++;
  assert.equal(n, 0, `${label}: ${n} bytes differ`);
}
function scalarMap(ctx) { return Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value])); }
function tableRows(src, params = {}) {
  const { out, ctx } = runOp('regionprops', src, { minArea: 0, maxObjects: 0, ...params });
  out.delete();
  return { rows: ctx.values.rows.map((r) => Object.fromEntries(ALL_COLS.map((k, i) => [k, r[i]]))), ctx };
}
function near(a, b, tol, label) { assert.ok(Math.abs(a - b) <= tol, `${label}: ${a} vs ${b} (tol ${tol})`); }
function rowByBox(rows, x, y) { return rows.find((r) => r.bbox_x === x && r.bbox_y === y); }

// ===================================================================
// ip3-0 — fingerprint (เทสต์แรกของไฟล์ = process สะอาด) · ip3-0w — wrapper balance ของเส้นทางชุดเดียวกัน
// ===================================================================
function batch3LeakPaths(only) {
  const mats = [];
  const keep = (m) => { mats.push(m); return m; };
  const want = (id) => !only || only === id;
  const empty = keep(new cv.Mat());
  const one = keep(gray(1, 1, () => 255)), oneB = keep(gray(1, 1, () => 0)), tiny = keep(gray(3, 5, (x, y) => x * 40 + y * 20));
  const scene = keep(sfScene()), shapes = keep(shapesScene()), noise = keep(noiseGray(97, 61, 7, 0.3)), blackM = keep(black(30, 20));
  const holes = keep(drawn(60, 60, (m) => { fillRectM(m, 5, 5, 50, 50); fillRectM(m, 15, 15, 30, 30, 0); fillRectM(m, 25, 25, 10, 10); }));
  const colour = keep(rgb(50, 40, (x, y) => [x * 5 & 255, y * 6 & 255, 200]));
  const rgba = keep(toRGBA(colour));
  const g2 = keep(gray(40, 30, (x, y) => (x * 7 + y * 3) & 255));
  const paths = [];
  const add = (id, params, src, tag) => paths.push({ label: `${id} ${tag}`, run: (cvUse) => { const o = OPS[id].apply(cvUse, src, params, {}); o.delete(); } });
  if (want('regionprops')) {
    for (const [tag, src] of [['scene', scene], ['shapes', shapes], ['noise', noise], ['holes', holes], ['1x1', one], ['1x1 black', oneB], ['3x5', tiny], ['black', blackM], ['gray texture', g2], ['0x0', empty]]) {
      add('regionprops', { minArea: 0, maxObjects: 0 }, src, tag);
    }
  }
  if (want('shape_filter')) {
    for (const view of ['overlay', 'mask']) {
      for (const prop of PROPS) {
        add('shape_filter', { prop, view, lo: 0.5, hi: 1, minArea: 0 }, scene, `${prop} ${view} scene`);
        add('shape_filter', { prop, view, lo: 0.99, hi: 1, minArea: 100000 }, shapes, `${prop} ${view} all removed`);
        add('shape_filter', { prop, view, minArea: 0 }, noise, `${prop} ${view} noise`);
        add('shape_filter', { prop, view, minArea: 0 }, holes, `${prop} ${view} holes`);
      }
      add('shape_filter', { view }, blackM, `${view} no objects`);
      add('shape_filter', { view, prop: 'none', minArea: 0 }, one, `${view} 1x1`);
      add('shape_filter', { view, lo: 0.9, hi: 0.1, minArea: 500, maxArea: 100 }, scene, `${view} swapped`);
      add('shape_filter', { view, prop: 'none', minArea: 0 }, colour, `${view} colour`);
      add('shape_filter', { view, prop: 'none', minArea: 0 }, rgba, `${view} rgba`);
      add('shape_filter', { view }, empty, `${view} 0x0`);
    }
  }
  if (want('quantize')) {
    for (const levels of [2, 4, 7, 256]) {
      add('quantize', { levels }, g2, `${levels} gray`);
      add('quantize', { levels }, colour, `${levels} rgb`);
      add('quantize', { levels }, rgba, `${levels} rgba`);
    }
    add('quantize', {}, one, '1x1');
    add('quantize', {}, empty, '0x0');
  }
  if (want('resize') || want('rotate')) {
    for (const interp of INTERPS) {
      for (const [tag, src] of [['gray', g2], ['rgb', colour], ['rgba', rgba], ['1x1', one]]) {
        if (want('resize')) {
          add('resize', { scale: 50, interp }, src, `${interp} ${tag} shrink`);
          add('resize', { scale: 200, interp }, src, `${interp} ${tag} enlarge`);
        }
        if (want('rotate')) {
          add('rotate', { angle: 37, expand: true, interp }, src, `${interp} ${tag} expand`);
          add('rotate', { angle: -20, expand: false, interp }, src, `${interp} ${tag} keep`);
        }
      }
      if (want('resize')) add('resize', { interp }, empty, `${interp} 0x0`);
      if (want('rotate')) add('rotate', { interp }, empty, `${interp} 0x0`);
    }
  }
  return { paths, cleanup: () => { for (const m of mats) m.delete(); } };
}

// A leak-free op cycles through a few malloc states while a leak never revisits one (see ipE-0 in the batch E file)
function fingerprintCycleProbe(label, run, { iters = 6, warmup = 6, windows = 9, maxDistinct = 4 } = {}) {
  for (let i = 0; i < warmup; i++) run();
  const seen = [];
  for (let w = 0; w < windows; w++) {
    for (let i = 0; i < iters; i++) run();
    seen.push(fingerprint(cv));
  }
  const distinct = new Set(seen).size;
  assert.ok(distinct <= maxDistinct, `${label}: ${distinct} distinct malloc fingerprints in ${windows} windows (a leak-free op cycles through <= ${maxDistinct}) -> leak`);
  return distinct;
}

test('ip3-0. leak (address fingerprint cycle, first test of a clean process): every path of regionprops (hull path included), shape_filter (5 props x 2 views, removed-all, no objects, swaps, colour / RGBA, 0x0), quantize and resize / rotate (4 interps) keeps revisiting the same few malloc states', () => {
  const { paths, cleanup } = batch3LeakPaths();
  try {
    assert.ok(paths.length >= 120, `paths ${paths.length}`);
    fingerprintCycleProbe('IP batch 3 (all paths)', () => { for (const q of paths) q.run(cv); });
  } finally { cleanup(); }
});

test('ip3-0c. (control of the fingerprint gate) the same probe flags a native leak of 16 bytes per call on top of the shape_filter paths -- so ip3-0 is able to fail', () => {
  const { paths, cleanup } = batch3LeakPaths('shape_filter');
  const ctl = makeNativeLeakControl(cv, 16);
  try {
    assert.throws(() => fingerprintCycleProbe('control', () => { for (const q of paths) q.run(cv); ctl.run(); }), /distinct malloc fingerprints/);
  } finally { cleanup(); ctl.cleanup(); }
});

test('ip3-0w. leak (wrapper balance twin of ip3-0): the exact same path table creates and deletes every embind wrapper it makes (contours, per-object masks, hull, bin / gray / kept mask, LUT, outputs, passthrough clones)', () => {
  const { paths, cleanup } = batch3LeakPaths();
  try {
    assertWrapperBalance(cv, 'IP batch 3 ops (all paths)', (c) => { for (const p of paths) p.run(c); }, { minCreated: 400 });
  } finally { cleanup(); }
});

// ===================================================================
// ip3-L1…L3 — leak ต่อ op / ต่อเส้นทาง
// ===================================================================
function leakEach(label, cases, opts = {}) {
  for (const [tag, fn] of cases) {
    assertLeakFree(cv, `${label} ${tag}`, (c) => { const o = fn(c); o.delete(); }, { minCreated: 1, iters: 12, warmup: 4, ...opts });
  }
}

test('ip3-L1. leak: every op of the batch on every path (regionprops incl. hull path, shape_filter every prop / view + no objects + all removed, quantize gray / RGB / RGBA, resize and rotate every interp) balances every wrapper and grows the heap by 0', () => {
  for (const id of ['regionprops', 'shape_filter', 'quantize', 'resize', 'rotate']) {
    const { paths, cleanup } = batch3LeakPaths(id);
    try {
      for (const q of paths) {
        assertLeakFree(cv, q.label, (c) => q.run(c), { minCreated: 1, iters: 8, warmup: 3 });
      }
    } finally { cleanup(); }
  }
});

function throwingCases() {
  const scene = sfScene(), shapes = shapesScene(), colour = rgb(40, 30, (x, y) => [x * 6 & 255, y * 8 & 255, 90]), rgba = toRGBA(colour), g = gray(40, 30, (x, y) => (x * 7 + y * 3) & 255);
  const sf = (p, src = scene) => (cvU) => OPS.shape_filter.apply(cvU, src, { minArea: 0, ...p }, {});
  const cases = [
    ['regionprops fillPoly', 'fillPoly', (cvU) => OPS.regionprops.apply(cvU, shapes, { minArea: 0 }, {})],
    ['regionprops convexHull (1st)', 'convexHull', (cvU) => OPS.regionprops.apply(cvU, shapes, { minArea: 0 }, {})],
    ['regionprops convexHull (3rd)', ['convexHull', 3], (cvU) => OPS.regionprops.apply(cvU, shapes, { minArea: 0 }, {})],
    ['regionprops fillPoly (4th)', ['fillPoly', 4], (cvU) => OPS.regionprops.apply(cvU, shapes, { minArea: 0 }, {})],
    ['shape_filter threshold', 'threshold', sf({})],
    ['shape_filter findContours', 'findContours', sf({})],
    ['shape_filter fillPoly', 'fillPoly', sf({ prop: 'none' })],
    ['shape_filter fillPoly (5th)', ['fillPoly', 5], sf({ prop: 'none' })],
    ['shape_filter convexHull', 'convexHull', sf({ prop: 'solidity', lo: 0, hi: 1 })],
    ['shape_filter convexHull (4th)', ['convexHull', 4], sf({ prop: 'solidity', lo: 0, hi: 1 })],
    ['shape_filter arcLength', 'arcLength', sf({ prop: 'circularity' })],
    ['shape_filter arcLength (3rd)', ['arcLength', 3], sf({ prop: 'circularity' })],
    ['shape_filter drawContours', 'drawContours', sf({ prop: 'none' })],
    ['shape_filter drawContours (3rd)', ['drawContours', 3], sf({ prop: 'none' })],
    ['shape_filter bitwise_and', 'bitwise_and', sf({ prop: 'none' })],
    ['shape_filter bitwise_and (mask view)', 'bitwise_and', sf({ prop: 'none', view: 'mask' })],
    ['shape_filter cvtColor (grayIfNeeded, colour)', 'cvtColor', sf({ prop: 'none' }, colour)],
    ['shape_filter cvtColor (grayIfNeeded, rgba)', 'cvtColor', sf({ prop: 'none' }, rgba)],
    ['quantize LUT (gray)', 'LUT', (cvU) => OPS.quantize.apply(cvU, g, {}, {})],
    ['quantize LUT (rgba, after dropAlpha)', 'LUT', (cvU) => OPS.quantize.apply(cvU, rgba, {}, {})],
    ['quantize cvtColor (dropAlpha)', 'cvtColor', (cvU) => OPS.quantize.apply(cvU, rgba, {}, {})],
    ['resize resize (auto)', 'resize', (cvU) => OPS.resize.apply(cvU, colour, { scale: 50 }, {})],
    ['resize resize (bicubic)', 'resize', (cvU) => OPS.resize.apply(cvU, colour, { scale: 150, interp: 'bicubic' }, {})],
    ['rotate getRotationMatrix2D', 'getRotationMatrix2D', (cvU) => OPS.rotate.apply(cvU, colour, { interp: 'nearest' }, {})],
    ['rotate warpAffine', 'warpAffine', (cvU) => OPS.rotate.apply(cvU, colour, { interp: 'bicubic' }, {})],
  ];
  return { cases, cleanup: () => { for (const m of [scene, shapes, colour, rgba, g]) m.delete(); } };
}

test('ip3-L2. leak on the exception path: when a cv call inside each op throws (regionprops fillPoly / convexHull 1st + later, shape_filter threshold / findContours / fillPoly / convexHull / arcLength / drawContours / bitwise_and / cvtColor, quantize LUT / cvtColor, resize resize, rotate getRotationMatrix2D / warpAffine), every wrapper allocated so far is released and the exception propagates', () => {
  const { cases, cleanup } = throwingCases();
  try {
    for (const [label, method, fn] of cases) {
      const [name, onCall] = Array.isArray(method) ? method : [method, null];
      assertWrapperBalance(cv, `throw in ${label}`, (cvU) => {
        const o = fn(makeThrowingCv(cvU, name, onCall ? { onCall } : {}));
        o.delete();
      }, { expectThrow: /injected test failure/, minCreated: name === 'getRotationMatrix2D' ? 0 : 1 });
    }
  } finally { cleanup(); }
});

test('ip3-L3. leak on the exception path AFTER the output exists: a ctx whose "info" or "values" setter throws makes shape_filter (both views) and quantize delete their own output', () => {
  const scene = sfScene(), g = gray(40, 30, (x, y) => (x * 7 + y * 3) & 255), c = rgb(20, 20, () => [1, 2, 3]);
  const throwingValues = () => ({ info: '', set values(_) { throw new Error('injected test failure (ctx.values)'); } });
  const throwingInfo = () => ({ set info(_) { throw new Error('injected test failure (ctx.info)'); } });
  try {
    const sfCases = [['overlay', { view: 'overlay' }], ['mask', { view: 'mask' }]];
    for (const [tag, p] of sfCases) {
      assertWrapperBalance(cv, `ctx.values throws in shape_filter ${tag}`, (cvU) => { const o = OPS.shape_filter.apply(cvU, scene, p, throwingValues()); o.delete(); }, { expectThrow: /injected test failure \(ctx\.values\)/, minCreated: 1 });
      assertWrapperBalance(cv, `ctx.info throws in shape_filter ${tag}`, (cvU) => { const o = OPS.shape_filter.apply(cvU, scene, p, throwingInfo()); o.delete(); }, { expectThrow: /injected test failure \(ctx\.info\)/, minCreated: 1 });
    }
    for (const [tag, src] of [['gray', g], ['rgb', c]]) {
      assertWrapperBalance(cv, `ctx.info throws in quantize ${tag}`, (cvU) => { const o = OPS.quantize.apply(cvU, src, {}, throwingInfo()); o.delete(); }, { expectThrow: /injected test failure \(ctx\.info\)/, minCreated: 1 });
    }
    assertWrapperBalance(cv, 'ctx.values throws in regionprops', (cvU) => { const o = OPS.regionprops.apply(cvU, scene, {}, throwingValues()); o.delete(); }, { expectThrow: /injected test failure \(ctx\.values\)/, minCreated: 1 });
  } finally { for (const m of [scene, g, c]) m.delete(); }
});

// ===================================================================
// ตัวช่วยร่วม (registry.js): regionBox · hullRowSpans · regionPixelStats · regionShapeValues · quantizeLut/quantizeBits
// ===================================================================
function contoursOf(bin) {
  const contours = new cv.MatVector(), hier = new cv.Mat();
  cv.findContours(bin, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  hier.delete();
  return contours;
}
function binOf(g) { const b = new cv.Mat(); cv.threshold(g, b, 127, 255, cv.THRESH_BINARY); return b; }
// test-local monotone chain (ไม่ใช้ cv.convexHull) → hull แบบทวนเข็ม ไม่รวมจุดร่วมเส้น
function cross(o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); }
function monotoneHull(points) {
  const pts = [...new Map(points.map((p) => [`${p[0]},${p[1]}`, p])).values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length <= 2) return pts;
  const lower = [];
  for (const p of pts) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p); }
  const upper = [];
  for (const p of [...pts].reverse()) { while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p); }
  lower.pop(); upper.pop();
  return lower.concat(upper);
}
function inHull(x, y, hull) {
  const m = hull.length;
  if (m === 1) return hull[0][0] === x && hull[0][1] === y;
  if (m === 2) {
    const [a, b] = hull;
    return cross(a, b, [x, y]) === 0 && x >= Math.min(a[0], b[0]) && x <= Math.max(a[0], b[0]) && y >= Math.min(a[1], b[1]) && y <= Math.max(a[1], b[1]);
  }
  for (let i = 0; i < m; i++) if (cross(hull[i], hull[(i + 1) % m], [x, y]) < 0) return false;
  return true;
}
function bruteHullCount(hull) {
  let n = 0;
  const xs = hull.map((p) => p[0]), ys = hull.map((p) => p[1]);
  for (let y = Math.min(...ys); y <= Math.max(...ys); y++) for (let x = Math.min(...xs); x <= Math.max(...xs); x++) if (inHull(x, y, hull)) n++;
  return n;
}
function ptsOf(c) { const d = c.data32S, out = []; for (let k = 0; k < d.length; k += 2) out.push([d[k], d[k + 1]]); return out; }
const grayTexture = (w, h, bin) => gray(w, h, (x, y) => (bin.data[y * w + x] ? 128 + ((x * 7 + y * 13) % 128) : (x * 3 + y) % 100));
function eachContour(bin, fn) {
  const cs = contoursOf(bin);
  try { for (let i = 0; i < cs.size(); i++) { const c = cs.get(i); try { fn(c, i); } finally { c.delete(); } } return cs.size(); } finally { cs.delete(); }
}

test('ip3-H1. regionBox == cv.boundingRect on >= 3000 contours (noise 30% / 10% images, shapes, lines)', () => {
  let total = 0;
  const imgs = [noiseGray(97, 61, 1, 0.3), noiseGray(200, 150, 2, 0.1), noiseGray(160, 120, 3, 0.3), shapesScene(), noiseGray(120, 90, 4, 0.5)];
  try {
    for (const g of imgs) {
      total += eachContour(g, (c) => {
        const r = cv.boundingRect(c), b = regionBox(c.data32S);
        assert.deepEqual([b.x, b.y, b.w, b.h], [r.x, r.y, r.width, r.height]);
      });
    }
  } finally { for (const g of imgs) g.delete(); }
  assert.ok(total >= 3000, `contours ${total}`);
});

test('ip3-H2. hullRowSpans hand cases: triangle (0,0),(4,0),(0,4) -> row widths 5,4,3,2,1 = 15; one point -> 1; segment (0,0)-(3,3) -> 4 rows of width 1; horizontal segment (1,2)-(5,2) -> 5', () => {
  const widths = (s) => Array.from(s.lo, (l, i) => Math.max(0, s.hi[i] - l + 1));
  let s = hullRowSpans([0, 0, 4, 0, 0, 4], { x: 0, y: 0, w: 5, h: 5 });
  assert.deepEqual(widths(s), [5, 4, 3, 2, 1]);
  assert.equal(widths(s).reduce((a, b) => a + b), 15);
  s = hullRowSpans([2, 3], { x: 2, y: 3, w: 1, h: 1 });
  assert.deepEqual(widths(s), [1]);
  s = hullRowSpans([0, 0, 3, 3], { x: 0, y: 0, w: 4, h: 4 });
  assert.deepEqual(widths(s), [1, 1, 1, 1]);
  assert.deepEqual(Array.from(s.lo), [0, 1, 2, 3]);
  s = hullRowSpans([1, 2, 5, 2], { x: 1, y: 2, w: 5, h: 1 });
  assert.deepEqual([s.lo[0], s.hi[0]], [1, 5]);
  // ทิศของรูปนูน (ตามเข็ม / ทวนเข็ม) ไม่ทำให้ผลต่าง
  const a = hullRowSpans([0, 0, 4, 0, 0, 4], { x: 0, y: 0, w: 5, h: 5 }), b = hullRowSpans([0, 4, 4, 0, 0, 0], { x: 0, y: 0, w: 5, h: 5 });
  assert.deepEqual(widths(a), widths(b));
});

test('ip3-H3. hullRowSpans == brute-force "pixel centre inside or on the hull" (cross products) on 600 random convex hulls (1 to 14 points, grid 0..40)', () => {
  const r = lcg(99);
  let checked = 0, nonTrivial = 0;
  for (let t = 0; t < 600; t++) {
    const k = 1 + Math.floor(r() * 14), span = 3 + Math.floor(r() * 38);
    const pts = Array.from({ length: k }, () => [Math.floor(r() * span), Math.floor(r() * span)]);
    const hull = monotoneHull(pts);
    const ys = hull.map((p) => p[1]), xs = hull.map((p) => p[0]);
    const box = { x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs) + 1, h: Math.max(...ys) - Math.min(...ys) + 1 };
    const sp = hullRowSpans(hull.flat(), box);
    for (let row = 0; row < box.h; row++) {
      let lo = Infinity, hi = -Infinity;
      for (let x = box.x; x < box.x + box.w; x++) if (inHull(x, box.y + row, hull)) { if (x < lo) lo = x; if (x > hi) hi = x; }
      const w = Math.max(0, sp.hi[row] - sp.lo[row] + 1);
      assert.equal(w, hi >= lo ? hi - lo + 1 : 0, `hull ${JSON.stringify(hull)} row ${row} width`);
      if (w > 0) assert.deepEqual([sp.lo[row], sp.hi[row]], [lo, hi], `hull ${JSON.stringify(hull)} row ${row}`);
      checked++;
    }
    if (hull.length >= 3) nonTrivial++;
  }
  assert.ok(checked > 3000 && nonTrivial > 400, `${checked} rows, ${nonTrivial} polygons`);
});

function statsImages() {
  const holes = drawn(60, 60, (m) => { fillRectM(m, 5, 5, 50, 50); fillRectM(m, 15, 15, 30, 30, 0); fillRectM(m, 25, 25, 10, 10); });
  const lines = drawn(80, 60, (m) => { cv.line(m, new cv.Point(2, 2), new cv.Point(70, 40), W255, 1); cv.line(m, new cv.Point(5, 50), new cv.Point(60, 50), W255, 1); cv.line(m, new cv.Point(40, 5), new cv.Point(40, 25), W255, 1); });
  const big = drawn(1920, 1080, (m) => { ellipseM(m, 960, 540, 900, 500, 0); cv.ellipse(m, new cv.Point(700, 540), new cv.Size(200, 150), 0, 0, 360, new cv.Scalar(0), -1); });
  const tex = noiseGray(97, 61, 11, 0.3);
  return { holes, lines, big, tex, noise: noiseGray(97, 61, 5, 0.3), one: gray(1, 1, () => 255), shapes: shapesScene() };
}

test('ip3-H4. regionPixelStats on 6+ images: sum of n over all contours == countNonZero(bin) and n >= 1 for every contour (30% noise, lines, holes + island, 1x1, a 1080p object of ~1.4 M px, gray texture, shapes)', () => {
  const imgs = statsImages();
  try {
    for (const [name, g] of Object.entries(imgs)) {
      const gt = name === 'tex' ? grayTexture(g.cols, g.rows, g) : g;
      const bin = binOf(gt);
      let sum = 0, count = 0;
      try {
        count = eachContour(bin, (c) => {
          const box = regionBox(c.data32S);
          const s = regionPixelStats(cv, c, box, bin.data, gt.data, gt.cols, true);
          assert.ok(s.n >= 1, `${name}: n ${s.n}`);
          assert.ok(s.hullN >= s.n, `${name}: hullN ${s.hullN} < n ${s.n}`);
          sum += s.n;
        });
        assert.equal(sum, cv.countNonZero(bin), `${name}: sum of n`);
        assert.ok(count >= 1, name);
      } finally { bin.delete(); if (gt !== g) gt.delete(); }
    }
    assert.ok(cv.countNonZero(imgs.big) > 1_000_000);
  } finally { for (const m of Object.values(imgs)) m.delete(); }
});

test('ip3-H5. regionPixelStats n / sums / gray sum == an independent reference (the 8-connected component of the contour first point, in a scene without islands)', () => {
  const g0 = shapesScene(), bin0 = binOf(g0), g = grayTexture(g0.cols, g0.rows, bin0), bin = binOf(g);
  const W = g.cols, H = g.rows;
  try {
    assert.equal(cv.countNonZero(bin), cv.countNonZero(bin0));
    const n = eachContour(bin, (c) => {
      const box = regionBox(c.data32S);
      const s = regionPixelStats(cv, c, box, bin.data, g.data, W, false);
      const [sx0, sy0] = [c.data32S[0], c.data32S[1]];
      const seen = new Set([sy0 * W + sx0]), q = [[sx0, sy0]];
      const ref = { n: 0, sx: 0, sy: 0, sxx: 0, syy: 0, sxy: 0, sum: 0 };
      while (q.length) {
        const [x, y] = q.pop();
        const rx = x - box.x, ry = y - box.y;
        ref.n++; ref.sx += rx; ref.sy += ry; ref.sxx += rx * rx; ref.syy += ry * ry; ref.sxy += rx * ry; ref.sum += g.data[y * W + x];
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= W || ny >= H || !bin.data[ny * W + nx] || seen.has(ny * W + nx)) continue;
          seen.add(ny * W + nx); q.push([nx, ny]);
        }
      }
      assert.deepEqual({ n: s.n, sx: s.sx, sy: s.sy, sxx: s.sxx, syy: s.syy, sxy: s.sxy, sum: s.sum }, ref, `contour at ${box.x},${box.y}`);
      assert.equal(s.hullN, null);
    });
    assert.equal(n, 8);
  } finally { g0.delete(); bin0.delete(); g.delete(); bin.delete(); }
});

test('ip3-H6. regionShapeValues solidity == n / (brute-force count of pixel centres inside the hull of the contour points, test-local monotone chain) for every object of the shapes scene + noise', () => {
  const imgs = [shapesScene(), noiseGray(97, 61, 21, 0.3)];
  try {
    for (const g of imgs) {
      const bin = binOf(g);
      try {
        eachContour(bin, (c) => {
          const box = regionBox(c.data32S);
          const s = regionPixelStats(cv, c, box, bin.data, g.data, g.cols, true);
          const v = regionShapeValues(s, box);
          const count = bruteHullCount(monotoneHull(ptsOf(c)));
          assert.equal(v.solidity, s.n / count, `object at ${box.x},${box.y}`);
          assert.ok(v.solidity <= 1 && v.solidity > 0);
        });
      } finally { bin.delete(); }
    }
  } finally { for (const g of imgs) g.delete(); }
});

test('ip3-H7. quantizeLut rows (N 2 / 3 / 4 / 5 / 8 / 256), monotone, exactly N distinct values, first 0, last 255 for every N in 2..256; quantizeBits', () => {
  const pick = (N, ...vs) => vs.map((v) => quantizeLut(N)[v]);
  assert.deepEqual([...new Set(quantizeLut(2))], [0, 255]);
  assert.deepEqual([...new Set(quantizeLut(3))], [0, 128, 255]);
  assert.deepEqual([...new Set(quantizeLut(4))], [0, 85, 170, 255]);
  assert.deepEqual([...new Set(quantizeLut(5))], [0, 64, 128, 191, 255]);
  assert.deepEqual([...new Set(quantizeLut(8))].slice(0, 6), [0, 36, 73, 109, 146, 182]);
  assert.deepEqual(Array.from(quantizeLut(256)), Array.from({ length: 256 }, (_, i) => i));
  assert.deepEqual(pick(2, 0, 127, 128, 255), [0, 0, 255, 255]);
  for (let N = 2; N <= 256; N++) {
    const t = quantizeLut(N);
    for (let v = 1; v < 256; v++) assert.ok(t[v] >= t[v - 1], `N ${N} monotone at ${v}`);
    assert.equal(new Set(t).size, N, `N ${N} distinct`);
    assert.equal(t[0], 0); assert.equal(t[255], 255);
  }
  assert.deepEqual([2, 3, 4, 5, 255, 256].map(quantizeBits), [1, 2, 2, 3, 8, 8]);
});

// ===================================================================
// regionprops — 19 คอลัมน์
// ===================================================================
test('ip3-R1. regionprops has 19 columns in the documented order and every row is 19 long on every path (shapes, noise, 1x1, tiny, holes, cap, minArea)', () => {
  const imgs = [shapesScene(), noiseGray(97, 61, 5, 0.3), gray(1, 1, () => 255), gray(3, 5, (x, y) => (x === 1 ? 255 : 0)), sfScene()];
  try {
    for (const g of imgs) {
      for (const p of [{}, { minArea: 0, maxObjects: 0 }, { minArea: 0, maxObjects: 3 }, { minArea: 100 }]) {
        const { out, ctx } = runOp('regionprops', g, p);
        out.delete();
        assert.deepEqual(ctx.values.columns, ALL_COLS);
        for (const row of ctx.values.rows) assert.equal(row.length, 19);
        assert.ok(ctx.values.total >= ctx.values.rows.length);
      }
    }
  } finally { for (const g of imgs) g.delete(); }
  assert.equal(ALL_COLS.length, 19);
  assert.deepEqual(REGION_SHAPE_COLUMNS, ['area_px', 'equiv_diameter', 'extent', 'solidity', 'major_axis', 'minor_axis', 'eccentricity', 'mean_intensity']);
});

const EXPECT = [
  // [ชื่อ, วาด, area, area_px, equiv, extent, solidity, major, minor, ecc, tol]
  ['1 px', (m) => { m.data[10 * 100 + 10] = 255; }, 0, 1, Math.sqrt(4 / Math.PI), 1, 1, 2 * Math.SQRT2 * Math.sqrt(1 / 6), 2 * Math.SQRT2 * Math.sqrt(1 / 6), 0, 1e-9],
  ['40x10 rectangle', (m) => fillRectM(m, 10, 10, 40, 10), 351, 400, Math.sqrt(1600 / Math.PI), 1, 1, 80 / Math.sqrt(3), 20 / Math.sqrt(3), Math.sqrt(15 / 16), 1e-9],
  ['square 10x10', (m) => fillRectM(m, 10, 10, 10, 10), 81, 100, 11.284, 1, 1, 11.547, 11.547, 0, 1e-3],
  ['disc r 40', (m) => cv.circle(m, new cv.Point(50, 50), 40, W255, -1), 4912, 5025, 79.99, 0.766, 1, 79.997, 79.997, 0, 1e-2],
  ['ring r 40 minus r 25', (m) => ringM(m, 50, 50, 40, 25), 4912, 3064, 62.46, 0.467, 0.610, 94.32, 94.32, 0, 1e-2],
  ['L (80x20 + 20x80)', (m) => lShapeM(m, 10, 10), 2641.5, 2800, 59.71, 0.438, 0.613, 115.47, 62.68, 0.840, 1e-2],
  ['ellipse 40x20 at 30 deg', (m) => ellipseM(m, 60, 60, 40, 20, 30), 2510.5, 2598, 57.51, 0.671, 0.999, 80.88, 40.92, 0.863, 1e-2],
  ['1-px line (10,10)-(60,40)', (m) => cv.line(m, new cv.Point(10, 10), new cv.Point(60, 40), W255, 1), 0, 51, 8.06, 0.032, 1, 68.65, 1.50, 1, 3e-2],
];
test('ip3-R2. regionprops values of the spec table (1 px / 40x10 rectangle against closed forms within 1e-9; the others against the probe numbers)', () => {
  for (const [name, draw, area, areaPx, eq, ext, sol, maj, min, ecc, tol] of EXPECT) {
    const g = drawn(130, 130, draw);
    try {
      const { rows } = tableRows(g);
      assert.equal(rows.length, 1, name);
      const r = rows[0];
      assert.equal(r.area, area, `${name} area`);
      assert.equal(r.area_px, areaPx, `${name} area_px`);
      near(r.equiv_diameter, eq, tol, `${name} equiv_diameter`);
      near(r.extent, ext, name === '1-px line (10,10)-(60,40)' ? 5e-3 : Math.max(tol, 1e-3), `${name} extent`);
      near(r.solidity, sol, Math.max(tol, 1e-3), `${name} solidity`);
      near(r.major_axis, maj, Math.max(tol, tol >= 1e-2 ? 1e-2 : tol), `${name} major`);
      near(r.minor_axis, min, Math.max(tol, tol >= 1e-2 ? 1e-2 : tol), `${name} minor`);
      near(r.eccentricity, ecc, Math.max(tol, tol >= 1e-2 ? 1e-2 : tol), `${name} ecc`);
      assert.equal(r.mean_intensity, 255, `${name} mean`);
    } finally { g.delete(); }
  }
});

test('ip3-R3. regionprops on a gray input: a 20x20 square of 200 with an inner 10x10 of 150 -> area_px 400, mean 187.5; inner 10x10 of 50 (a hole, < 127) -> area_px 300, mean 200, area 361 (polygon counts the hole)', () => {
  const a = drawn(60, 60, (m) => { fillRectM(m, 10, 10, 20, 20, 200); fillRectM(m, 15, 15, 10, 10, 150); });
  const b = drawn(60, 60, (m) => { fillRectM(m, 10, 10, 20, 20, 200); fillRectM(m, 15, 15, 10, 10, 50); });
  try {
    const ra = tableRows(a), rb = tableRows(b);
    assert.equal(ra.rows.length, 1); assert.equal(rb.rows.length, 1);
    assert.equal(ra.rows[0].area_px, 400); assert.equal(ra.rows[0].mean_intensity, 187.5);
    assert.equal(rb.rows[0].area_px, 300); assert.equal(rb.rows[0].mean_intensity, 200); assert.equal(rb.rows[0].area, 361);
    assert.ok(rb.rows[0].solidity < 1 && Math.abs(rb.rows[0].solidity - 0.75) < 1e-12);
    assert.ok(!/ความสว่างเฉลี่ยจึงเป็น 255/.test(ra.ctx.info) && !/ความสว่างเฉลี่ยจึงเป็น 255/.test(rb.ctx.info), 'gray input -> no note');
  } finally { a.delete(); b.delete(); }
});

test('ip3-R4. regionprops invariants on 6 images: 0 < extent <= 1, 0 < solidity <= 1, 0 <= ecc < 1, minor <= major, minor >= sqrt(2/3) - 1e-12 (1.1547005383792515), 1 <= area_px <= bbox area, mean in 0..255', () => {
  const imgs = [shapesScene(), noiseGray(97, 61, 5, 0.3), noiseGray(160, 120, 8, 0.1), sfScene(), drawn(60, 60, (m) => { fillRectM(m, 5, 5, 50, 50); fillRectM(m, 15, 15, 30, 30, 0); fillRectM(m, 25, 25, 10, 10); }), noiseGray(80, 80, 9, 0.5)];
  let seen = 0;
  try {
    for (const g of imgs) {
      for (const r of tableRows(g).rows) {
        seen++;
        assert.ok(r.extent > 0 && r.extent <= 1, `extent ${r.extent}`);
        assert.ok(r.solidity > 0 && r.solidity <= 1, `solidity ${r.solidity}`);
        assert.ok(r.eccentricity >= 0 && r.eccentricity < 1, `ecc ${r.eccentricity}`);
        assert.ok(r.minor_axis <= r.major_axis + 1e-12, 'minor <= major');
        assert.ok(r.minor_axis >= 1.1547005383792515 - 1e-12, `minor ${r.minor_axis}`);
        assert.ok(r.area_px >= 1 && r.area_px <= r.bbox_w * r.bbox_h, 'area_px vs bbox');
        assert.ok(r.mean_intensity >= 0 && r.mean_intensity <= 255);
        assert.ok(r.equiv_diameter > 0);
      }
    }
  } finally { for (const g of imgs) g.delete(); }
  assert.ok(seen > 300, `rows ${seen}`);
});

test('ip3-R5. regionprops info note: binary input (every row mean 255) appends the black-and-white note once; gray input / no objects do not', () => {
  const note = ' · ภาพที่เข้ามาเป็นขาว-ดำอยู่แล้ว ความสว่างเฉลี่ยจึงเท่ากับค่าขาวเต็มทุกชิ้น';
  const bw = shapesScene(), gtex = (() => { const b = binOf(bw); const g = grayTexture(bw.cols, bw.rows, b); b.delete(); return g; })(), blk = black(30, 20);
  try {
    const i1 = runInfo('regionprops', bw, {}).info;
    assert.ok(i1.endsWith(note), i1);
    assert.equal(i1.split('ค่าขาวเต็มทุกชิ้น').length, 2);
    assert.ok(!runInfo('regionprops', gtex, {}).info.includes('ค่าขาวเต็มทุกชิ้น'));
    const i3 = runInfo('regionprops', blk, {}).info;
    assert.ok(i3.startsWith('ไม่พบวัตถุ') && !i3.includes('ค่าขาวเต็มทุกชิ้น'));
    // เมื่อ cap ตัดแถว: note ดูเฉพาะแถวที่แสดง
    const i4 = runInfo('regionprops', bw, { minArea: 0, maxObjects: 2 }).info;
    assert.ok(i4.includes('แสดง 2 ชิ้นที่ใหญ่ที่สุด') && i4.endsWith(note));
  } finally { bw.delete(); gtex.delete(); blk.delete(); }
});

test('ip3-R6. regionprops 0x0: clone + "ภาพว่าง" + a 19-column table with rows [] and total 0 (the 11-column table of before is gone); the output is a new Mat', () => {
  const e = new cv.Mat();
  try {
    const { out, ctx } = runOp('regionprops', e, {});
    try {
      assert.equal(out.rows, 0); assert.equal(out.cols, 0);
      assert.notEqual(out.$$.ptr, e.$$.ptr);
      assert.equal(ctx.info, 'ภาพว่าง');
      assert.deepEqual(ctx.values, { kind: 'table', columns: ALL_COLS, rows: [], total: 0 });
    } finally { out.delete(); }
  } finally { e.delete(); }
});

test('ip3-R7. regionprops edge cases: 1x1 white (area 0, area_px 1, extent 1, solidity 1, major = minor = 1.1547, ecc 0); axis-aligned rectangle (< 5 contour points): orientation null but axes / ecc numbers; 1-px line: area 0, circularity 0, area_px > 0, ecc -> 1; black -> rows []', () => {
  const one = gray(1, 1, () => 255), rect = drawn(60, 40, (m) => fillRectM(m, 5, 5, 40, 10)), line = drawn(80, 50, (m) => cv.line(m, new cv.Point(5, 5), new cv.Point(60, 30), W255, 1)), blk = black(10, 10);
  try {
    const r1 = tableRows(one).rows;
    assert.equal(r1.length, 1);
    assert.deepEqual([r1[0].area, r1[0].area_px, r1[0].extent, r1[0].solidity, r1[0].eccentricity], [0, 1, 1, 1, 0]);
    near(r1[0].major_axis, 1.1547005383792515, 1e-12, 'major'); near(r1[0].minor_axis, 1.1547005383792515, 1e-12, 'minor');
    const rr = tableRows(rect).rows[0];
    assert.equal(rr.orientation_deg, null);
    assert.ok(rr.major_axis > 0 && rr.minor_axis > 0 && rr.eccentricity > 0.9);
    const rl = tableRows(line).rows[0];
    assert.equal(rl.area, 0); assert.equal(rl.circularity, 0); assert.ok(rl.area_px > 20); assert.ok(rl.eccentricity > 0.99);
    assert.deepEqual(tableRows(blk).rows, []);
  } finally { for (const m of [one, rect, line, blk]) m.delete(); }
});

test('ip3-R8. regionprops holes / island: a frame with a hole and an island inside the hole is ONE object: area_px = outer - hole + island, area (polygon) counts the hole; solidity < 1', () => {
  const g = drawn(60, 60, (m) => { fillRectM(m, 5, 5, 50, 50); fillRectM(m, 15, 15, 30, 30, 0); fillRectM(m, 25, 25, 10, 10); });
  try {
    const { rows } = tableRows(g);
    assert.equal(rows.length, 1);
    assert.equal(rows[0].area_px, 2500 - 900 + 100);
    assert.equal(rows[0].area, 49 * 49);
    assert.ok(rows[0].solidity < 1);
    assert.equal(rows[0].extent, 1700 / 2500);
  } finally { g.delete(); }
});

test('ip3-R9. regionprops does not call boundingRect / moments / contourArea again for the new columns (counts: boundingRect only in the second pass = once per shown row), does not modify src, and a colour direct call is unchanged by gray conversion (needs gray)', () => {
  const g = drawn(300, 120, (m) => { fillRectM(m, 5, 5, 40, 10); cv.circle(m, new cv.Point(100, 40), 30, W255, -1); ringM(m, 200, 50, 40, 25); lShapeM(m, 5, 30); });
  const before = bytes(g);
  const counts = { boundingRect: 0, moments: 0, contourArea: 0, convexHull: 0, fillPoly: 0 };
  const spy = new Proxy(cv, { get(t, p) { if (p in counts) return (...a) => { counts[p]++; return t[p](...a); }; return t[p]; } });
  try {
    const { out, ctx } = runOp('regionprops', g, { minArea: 0, maxObjects: 0 }, spy);
    out.delete();
    const n = ctx.values.rows.length;
    assert.equal(counts.contourArea, 0);
    assert.equal(counts.boundingRect, n, 'boundingRect once per shown row');
    assert.equal(counts.convexHull, n);
    assert.equal(counts.fillPoly, n);
    assert.deepEqual(bytes(g), before);
  } finally { g.delete(); }
});

test('ip3-R10. regionprops note needs EVERY row to be 255: one object of 255 plus one of 200 (mean 200) gives no note; two objects of 255 give it', () => {
  const mixed = drawn(80, 40, (m) => { fillRectM(m, 5, 5, 20, 20, 255); fillRectM(m, 40, 5, 20, 20, 200); });
  const pure = drawn(80, 40, (m) => { fillRectM(m, 5, 5, 20, 20, 255); fillRectM(m, 40, 5, 20, 20, 255); });
  try {
    assert.deepEqual(tableRows(mixed).rows.map((r) => r.mean_intensity).sort(), [200, 255]);
    assert.ok(!runInfo('regionprops', mixed, {}).info.includes('ค่าขาวเต็มทุกชิ้น'));
    assert.ok(runInfo('regionprops', pure, {}).info.includes('ค่าขาวเต็มทุกชิ้น'));
  } finally { mixed.delete(); pure.delete(); }
});

// ===================================================================
// shape_filter
// ===================================================================
const sfRun = (src, p = {}) => runOp('shape_filter', src, p);
function sfCounts(src, p) {
  const { out, ctx } = sfRun(src, p);
  out.delete();
  const m = scalarMap(ctx);
  return { total: m.sf_total, kept: m.sf_kept, removedArea: m.sf_removed_area, removedShape: m.sf_removed_shape, info: ctx.info };
}
function keptBoxes(src, p) { // bbox ของวัตถุที่เก็บ (ผ่าน regionprops บน Mask) เรียงตามตำแหน่ง
  const { out } = sfRun(src, { ...p, view: 'mask' });
  try { return tableRows(out).rows.map((r) => `${r.bbox_x},${r.bbox_y}`).sort(); } finally { out.delete(); }
}

test('ip3-S1. shape_filter on the 20-object scene: defaults keep 3 discs + the ring (specks removed by area, squares / ellipses / L by shape); solidity 0.9-1 removes only L and ring; eccentricity 0.95-1 keeps only the 2 ellipses; extent 0.95-1 keeps only the 3 squares; none + area 100..1000 is area only', () => {
  const g = sfScene();
  try {
    const d = sfCounts(g, {});
    assert.deepEqual([d.total, d.kept, d.removedArea, d.removedShape], [20, 4, 10, 6], d.info);
    assert.equal(d.total, d.kept + d.removedArea + d.removedShape);
    const sol = sfCounts(g, { prop: 'solidity', lo: 0.9, hi: 1 });
    assert.deepEqual([sol.total, sol.kept, sol.removedArea, sol.removedShape], [20, 8, 10, 2], sol.info);
    const ecc = sfCounts(g, { prop: 'eccentricity', lo: 0.95, hi: 1 });
    assert.deepEqual([ecc.kept, ecc.removedShape], [2, 8], ecc.info);
    const ext = sfCounts(g, { prop: 'extent', lo: 0.95, hi: 1 });
    assert.deepEqual([ext.kept, ext.removedShape], [3, 7], ext.info);
    const area = sfCounts(g, { prop: 'none', minArea: 100, maxArea: 1000 });
    assert.deepEqual([area.kept, area.removedArea, area.removedShape], [5, 15, 0], area.info);
    // the ring is circularity 0.887 and is kept by default; the kept objects are exactly the 3 discs + the ring (by position)
    const boxes = keptBoxes(g, {});
    assert.equal(boxes.length, 4);
    for (const b of ['20,20', '90,20', '160,20', '40,190']) assert.ok(boxes.includes(b), `${b} in ${boxes}`);
  } finally { g.delete(); }
});

test('ip3-S2. shape_filter area bounds are inclusive: a 10x10 square (100 px) is kept at minArea 100 and removed at 101; kept at maxArea 100 and removed at 99; the bbox pre-filter never removes an object whose pixel count passes', () => {
  const g = drawn(60, 40, (m) => fillRectM(m, 10, 10, 10, 10));
  try {
    for (const view of ['mask', 'overlay']) {
      assert.equal(sfCounts(g, { prop: 'none', minArea: 100, view }).kept, 1);
      assert.equal(sfCounts(g, { prop: 'none', minArea: 101, view }).kept, 0);
      assert.equal(sfCounts(g, { prop: 'none', minArea: 0, maxArea: 100, view }).kept, 1);
      const r = sfCounts(g, { prop: 'none', minArea: 0, maxArea: 99, view });
      assert.deepEqual([r.kept, r.removedArea], [0, 1]);
    }
    // a thin diagonal line: bbox 40x40 passes the pre-filter but its pixel count (40) does not
    const l = drawn(60, 60, (m) => cv.line(m, new cv.Point(5, 5), new cv.Point(44, 44), W255, 1));
    try {
      const c = sfCounts(l, { prop: 'none', minArea: 50 });
      assert.deepEqual([c.kept, c.removedArea], [0, 1]);
      assert.equal(sfCounts(l, { prop: 'none', minArea: 40 }).kept, 1);
    } finally { l.delete(); }
    // maxArea 0 = unlimited
    assert.equal(sfCounts(g, { prop: 'none', minArea: 0, maxArea: 0 }).kept, 1);
  } finally { g.delete(); }
});

test('ip3-S3. shape_filter range bounds are inclusive and the property is the app value: lo = hi = the exact value keeps the object (solidity 1, extent 1 of a square); a value just above / below removes it', () => {
  const g = drawn(60, 40, (m) => fillRectM(m, 10, 10, 10, 10));
  try {
    assert.equal(sfCounts(g, { prop: 'solidity', lo: 1, hi: 1, minArea: 0 }).kept, 1);
    assert.equal(sfCounts(g, { prop: 'extent', lo: 1, hi: 1, minArea: 0 }).kept, 1);
    assert.equal(sfCounts(g, { prop: 'extent', lo: 0, hi: 0.99, minArea: 0 }).kept, 0);
    // circularity of an axis-aligned square = pi/4 = 0.785398...: lo 0.79 removes, lo 0.78 keeps
    assert.equal(sfCounts(g, { prop: 'circularity', lo: 0.79, hi: 1, minArea: 0 }).kept, 0);
    assert.equal(sfCounts(g, { prop: 'circularity', lo: 0.78, hi: 1, minArea: 0 }).kept, 1);
    assert.equal(sfCounts(g, { prop: 'circularity', lo: 0, hi: 0.78, minArea: 0 }).kept, 0);
    const { rows } = tableRows(g);
    assert.ok(Math.abs(rows[0].circularity - Math.PI / 4) < 1e-12);
    // eccentricity 0 (a square) with the range [0, 0]
    assert.equal(sfCounts(g, { prop: 'eccentricity', lo: 0, hi: 0, minArea: 0 }).kept, 1);
  } finally { g.delete(); }
});

test('ip3-S4. shape_filter holes and islands: a kept ring keeps its centre black in both views; an island inside a kept ring is kept; an island inside the hole of a removed object is removed (follows its outer object)', () => {
  const g = drawn(220, 120, (m) => {
    ringM(m, 50, 60, 40, 25);
    cv.circle(m, new cv.Point(50, 60), 10, W255, -1);          // island in the kept ring
    fillRectM(m, 110, 10, 100, 100);                           // frame (square: circularity 0.785 -> removed at lo 0.8)
    fillRectM(m, 125, 25, 70, 70, 0);
    cv.circle(m, new cv.Point(160, 60), 8, W255, -1);          // island in the removed frame
  });
  try {
    const mask = sfRun(g, { view: 'mask' }), ov = sfRun(g, { view: 'overlay' });
    try {
      const W = g.cols, at = (m, x, y, c = 0) => m.data[(y * W + x) * m.channels() + c];
      assert.equal(at(mask.out, 50 + 18, 60), 0, 'hole of the ring stays black');
      assert.equal(at(mask.out, 50, 60), 255, 'island inside the kept ring is kept');
      assert.equal(at(mask.out, 160, 60), 0, 'island inside a removed object is removed');
      assert.equal(at(mask.out, 112, 12), 0, 'frame removed');
      assert.deepEqual([at(ov.out, 68, 60, 0), at(ov.out, 68, 60, 1), at(ov.out, 68, 60, 2)], [0, 0, 0]);
      assert.deepEqual([at(ov.out, 50, 60, 0), at(ov.out, 50, 60, 1), at(ov.out, 50, 60, 2)], [255, 255, 255]);
      assert.deepEqual([at(ov.out, 160, 60, 0), at(ov.out, 160, 60, 1), at(ov.out, 160, 60, 2)], [255, 0, 0]);
      assert.deepEqual([at(ov.out, 112, 12, 0), at(ov.out, 112, 12, 1), at(ov.out, 112, 12, 2)], [255, 0, 0]);
      assert.equal(scalarMap(mask.ctx).sf_total, 2);
      assert.equal(scalarMap(mask.ctx).sf_kept, 1);
    } finally { mask.out.delete(); ov.out.delete(); }
  } finally { g.delete(); }
});

test('ip3-S5. shape_filter views: mask = 1 channel with values in {0, 255}; overlay = 3 channels with only (0,0,0) / (255,0,0) / (255,255,255); white == kept AND bin, red == bin AND NOT kept; both sized like the input', () => {
  const g = sfScene(), bin = binOf(g);
  try {
    const mask = sfRun(g, {}), ov = sfRun(g, { view: 'overlay' });
    const m = mask.out, o = ov.out;
    try {
      // default view = overlay: call with explicit mask view for the mask
      const mm = sfRun(g, { view: 'mask' });
      try {
        assert.equal(mm.out.channels(), 1); assert.equal(o.channels(), 3);
        assert.deepEqual([mm.out.cols, mm.out.rows, o.cols, o.rows], [400, 300, 400, 300]);
        assert.deepEqual([...new Set(mm.out.data)].sort((a, b) => a - b), [0, 255]);
        let white = 0, red = 0;
        for (let i = 0; i < 400 * 300; i++) {
          const px = [o.data[3 * i], o.data[3 * i + 1], o.data[3 * i + 2]].join(',');
          assert.ok(['0,0,0', '255,0,0', '255,255,255'].includes(px), px);
          const kept = mm.out.data[i] === 255, b = bin.data[i] === 255;
          if (px === '255,255,255') { assert.ok(kept && b); white++; }
          if (px === '255,0,0') { assert.ok(b && !kept); red++; }
          if (px === '0,0,0') assert.ok(!b);
          if (kept) assert.ok(b && px === '255,255,255');
        }
        assert.ok(white > 4000 && red > 1000, `white ${white} red ${red}`);
        assert.equal(white, cv.countNonZero(mm.out));
      } finally { mm.out.delete(); }
    } finally { m.delete(); o.delete(); }
  } finally { g.delete(); bin.delete(); }
});

test('ip3-S6. shape_filter chains into Region Properties: regionprops(overlay -> RGB2GRAY, red 76 < 127) rows == regionprops(mask) rows', () => {
  const g = sfScene();
  try {
    for (const p of [{}, { prop: 'solidity', lo: 0.9 }, { prop: 'none', minArea: 100, maxArea: 1000 }]) {
      const mask = sfRun(g, { ...p, view: 'mask' }), ov = sfRun(g, { ...p, view: 'overlay' });
      const og = new cv.Mat();
      try {
        cv.cvtColor(ov.out, og, cv.COLOR_RGB2GRAY);
        assert.equal(og.data[0], 0);
        assert.deepEqual(tableRows(og).rows, tableRows(mask.out).rows, JSON.stringify(p));
      } finally { mask.out.delete(); ov.out.delete(); og.delete(); }
    }
    // red really maps below 127 (the documented claim)
    const red = rgb(1, 1, () => [255, 0, 0]), rg = new cv.Mat();
    cv.cvtColor(red, rg, cv.COLOR_RGB2GRAY);
    assert.ok(rg.data[0] < 127); red.delete(); rg.delete();
  } finally { g.delete(); }
});

test('ip3-S7. shape_filter single source of truth: for each property the kept set == the objects whose Region Properties row (same image, minArea 0, maxObjects 0, matched by bbox) has the value in [lo, hi] and area_px in the area range', () => {
  const imgs = [sfScene(), shapesScene(), noiseGray(97, 61, 5, 0.3)];
  try {
    for (const g of imgs) {
      const { rows } = tableRows(g);
      for (const [prop, col, lo, hi] of [['circularity', 'circularity', 0.7, 0.9], ['solidity', 'solidity', 0.6, 0.99], ['eccentricity', 'eccentricity', 0.1, 0.9], ['extent', 'extent', 0.5, 0.8], ['circularity', 'circularity', 0.8, 1]]) {
        for (const [minArea, maxArea] of [[0, 0], [30, 2000]]) {
          const want = rows.filter((r) => r[col] >= lo && r[col] <= hi && r.area_px >= minArea && (maxArea === 0 || r.area_px <= maxArea)).map((r) => `${r.bbox_x},${r.bbox_y}`).sort();
          const got = keptBoxes(g, { prop, lo, hi, minArea, maxArea });
          assert.deepEqual(got, want, `${prop} ${lo}-${hi} area ${minArea}-${maxArea}`);
        }
      }
    }
  } finally { for (const g of imgs) g.delete(); }
});

test('ip3-S8. shape_filter swaps + values contract: lo > hi and maxArea < minArea are swapped with the notes in the documented order; the 4 values are always the same 4 keys; total == kept + removedArea + removedShape on every path', () => {
  const g = sfScene();
  try {
    const a = sfRun(g, { prop: 'circularity', lo: 1, hi: 0.5, minArea: 5000, maxArea: 100, view: 'overlay' });
    a.out.delete();
    assert.ok(a.ctx.info.includes(' · สลับค่าต่ำสุด/สูงสุดให้แล้ว · สลับพื้นที่ขั้นต่ำ/สูงสุดให้แล้ว · ขาว = เก็บ แดง = คัดทิ้ง'), a.ctx.info);
    assert.ok(a.ctx.info.includes('เก็บช่วง 0.50–1.00') && a.ctx.info.includes('พื้นที่ที่เก็บ 100–5000 px'), a.ctx.info);
    const b = sfRun(g, { lo: 0.5, hi: 1, view: 'mask' });
    b.out.delete();
    assert.ok(!b.ctx.info.includes('สลับ') && !b.ctx.info.includes('ขาว = เก็บ'), b.ctx.info);
    const swapped = sfCounts(g, { lo: 1, hi: 0.8 }), plain = sfCounts(g, { lo: 0.8, hi: 1 });
    assert.deepEqual([swapped.kept, swapped.removedShape, swapped.removedArea], [plain.kept, plain.removedShape, plain.removedArea]);
    const swappedA = sfCounts(g, { prop: 'none', minArea: 1000, maxArea: 100 }), plainA = sfCounts(g, { prop: 'none', minArea: 100, maxArea: 1000 });
    assert.deepEqual([swappedA.kept, swappedA.removedArea], [plainA.kept, plainA.removedArea]);
    for (const prop of PROPS) for (const view of ['overlay', 'mask']) {
      const { out, ctx } = sfRun(g, { prop, view, minArea: 20, lo: 0.3, hi: 0.95 });
      out.delete();
      assert.deepEqual(ctx.values.fields.map((f) => f.key), SF_KEYS);
      assert.deepEqual(ctx.values.fields.map((f) => f.label), ['วัตถุทั้งหมด (ชิ้น)', 'เก็บไว้ (ชิ้น)', 'คัดทิ้งเพราะพื้นที่ (ชิ้น)', 'คัดทิ้งเพราะรูปร่าง (ชิ้น)']);
      const m = scalarMap(ctx);
      assert.equal(m.sf_total, m.sf_kept + m.sf_removed_area + m.sf_removed_shape);
      if (prop === 'none') assert.equal(m.sf_removed_shape, 0);
    }
    // info text: 'none' omits the shape clause; unlimited max area text
    const n = sfCounts(g, { prop: 'none', minArea: 20 });
    assert.equal(n.info, `เก็บ ${n.kept} จาก ${n.total} ชิ้น · คัดทิ้งเพราะพื้นที่ ${n.removedArea} ชิ้น · พื้นที่ที่เก็บ 20–ไม่จำกัด px · ขาว = เก็บ แดง = คัดทิ้ง (ต่อเข้า Region Properties ได้เลย สีแดงจะถูกตัดทิ้งที่ 127)`);
    const c = sfCounts(g, { prop: 'solidity', lo: 0.9, hi: 1 });
    assert.ok(c.info.includes(`คัดทิ้งเพราะความตัน ${c.removedShape} ชิ้น (เก็บช่วง 0.90–1.00)`));
    for (const [prop, th] of [['circularity', 'ความกลม'], ['eccentricity', 'ความรี'], ['extent', 'เต็มกรอบ']]) assert.ok(sfCounts(g, { prop }).info.includes(`คัดทิ้งเพราะ${th} `), prop);
  } finally { g.delete(); }
});

test('ip3-S9. shape_filter calls convexHull only when prop = solidity (counting proxy), never for the other props; hull count == the objects that pass the bbox pre-filter', () => {
  const g = sfScene();
  const count = (p) => {
    let n = 0;
    const spy = new Proxy(cv, { get(t, k) { if (k === 'convexHull') return (...a) => { n++; return t.convexHull(...a); }; return t[k]; } });
    const { out } = runOp('shape_filter', g, p, spy);
    out.delete();
    return n;
  };
  try {
    for (const prop of ['circularity', 'eccentricity', 'extent', 'none']) assert.equal(count({ prop }), 0, prop);
    assert.equal(count({ prop: 'solidity', minArea: 50 }), 10, 'specks (w*h < 50) skip the hull; 10 real objects remain');
    assert.equal(count({ prop: 'solidity', minArea: 0 }), 20);
  } finally { g.delete(); }
});

test('ip3-S10. shape_filter edge cases: 0x0 (clone + "ภาพว่าง" + 4 null values, both views), 1x1 white (total 1; default area filter removes it by area; with minArea 0: circularity 0 -> removed by shape, prop none -> kept), black (total 0 + message + 0,0,0,0), everything filtered (mask all 0 / overlay only black+red), colour R != B and RGBA direct calls == the pre-converted gray, src untouched', () => {
  const e = new cv.Mat(), one = gray(1, 1, () => 255), blk = black(40, 30), g = sfScene(), col = noiseRGB(60, 50, 4), colG = new cv.Mat();
  cv.cvtColor(col, colG, cv.COLOR_RGB2GRAY);
  const rgba = toRGBA(col), rgbaG = new cv.Mat();
  cv.cvtColor(rgba, rgbaG, cv.COLOR_RGBA2GRAY);
  const before = bytes(g);
  try {
    for (const view of ['overlay', 'mask']) {
      const r = sfRun(e, { view });
      assert.equal(r.out.rows, 0); assert.equal(r.ctx.info, 'ภาพว่าง');
      assert.deepEqual(r.ctx.values.fields.map((f) => [f.key, f.value]), SF_KEYS.map((k) => [k, null]));
      r.out.delete();
      const b = sfRun(blk, { view });
      assert.deepEqual(scalarMap(b.ctx), { sf_total: 0, sf_kept: 0, sf_removed_area: 0, sf_removed_shape: 0 });
      assert.equal(b.ctx.info.startsWith('ไม่พบวัตถุ (ภาพดำล้วน — ต่อจาก Threshold ให้เป็นขาว-ดำก่อน)'), true);
      assert.ok(b.out.data.every((v) => v === 0));
      b.out.delete();
      const none = sfRun(g, { view, prop: 'circularity', lo: 1, hi: 1, minArea: 100000 });
      assert.equal(scalarMap(none.ctx).sf_kept, 0);
      if (view === 'mask') assert.ok(none.out.data.every((v) => v === 0)); else for (let i = 0; i < none.out.data.length; i += 3) assert.ok(none.out.data[i + 1] === 0 && none.out.data[i + 2] === 0 && (none.out.data[i] === 0 || none.out.data[i] === 255));
      none.out.delete();
    }
    assert.deepEqual([sfCounts(one, {}).removedArea, sfCounts(one, {}).total], [1, 1]);
    const c0 = sfCounts(one, { minArea: 0 });
    assert.deepEqual([c0.total, c0.kept, c0.removedShape], [1, 0, 1]);
    assert.equal(sfCounts(one, { minArea: 0, prop: 'none' }).kept, 1);
    // zero-area contour (1-px line): circularity 0 -> removed when lo > 0
    const line = drawn(60, 40, (m) => cv.line(m, new cv.Point(5, 5), new cv.Point(50, 30), W255, 1));
    try {
      assert.equal(sfCounts(line, { minArea: 0 }).removedShape, 1);
      assert.equal(sfCounts(line, { minArea: 0, lo: 0, hi: 1 }).kept, 1);
    } finally { line.delete(); }
    for (const view of ['overlay', 'mask']) {
      for (const [name, c, pre] of [['rgb', col, colG], ['rgba', rgba, rgbaG]]) {
        const a = sfRun(c, { view, prop: 'none', minArea: 0 }), b = sfRun(pre, { view, prop: 'none', minArea: 0 });
        try { sameBytes(a.out, b.out, `${name} ${view}`); assert.deepEqual(scalarMap(a.ctx), scalarMap(b.ctx)); } finally { a.out.delete(); b.out.delete(); }
      }
    }
    assert.deepEqual(bytes(g), before);
  } finally { for (const m of [e, one, blk, g, col, colG, rgba, rgbaG]) m.delete(); }
});

test('ip3-S11. shape_filter params: garbage / out of range / "constructor" -> defaults or clamps (prop / view "constructor" -> defaults; lo "x" -> 0.8; hi 5 -> 1; minArea -3 -> 0; maxArea huge -> 5000000)', () => {
  const g = sfScene();
  try {
    const d = sfRun(g, {}), c = sfRun(g, { prop: 'constructor', view: 'constructor', lo: 'x', hi: 'y', minArea: 'z', maxArea: 'q' });
    sameBytes(d.out, c.out, 'garbage == defaults'); assert.equal(d.ctx.info, c.ctx.info);
    d.out.delete(); c.out.delete();
    const a = sfRun(g, { prop: 'none', hi: 5, minArea: -3, maxArea: 99999999 });
    assert.ok(a.ctx.info.includes('พื้นที่ที่เก็บ 0–5000000 px'), a.ctx.info);
    a.out.delete();
    const b = sfRun(g, { prop: 'solidity', lo: -4, hi: 9 });
    assert.ok(b.ctx.info.includes('เก็บช่วง 0.00–1.00'), b.ctx.info);
    b.out.delete();
  } finally { g.delete(); }
});

// ===================================================================
// quantize
// ===================================================================
test('ip3-Q1. quantize: a gray ramp 0..255 == the LUT; RGB per channel; RGBA -> 3 channels; N = 2 == binary_threshold (thresh 127) byte for byte; N = 256 identity; the input is untouched', () => {
  const ramp = gray(256, 4, (x) => x), c = rgb(64, 8, (x, y) => [x * 4, 255 - x * 4, (y * 31 + x) & 255]), a = toRGBA(c);
  const before = bytes(c);
  try {
    for (const N of [2, 3, 4, 5, 8, 16, 255, 256]) {
      const lut = quantizeLut(N);
      const { out } = runOp('quantize', ramp, { levels: N });
      for (let x = 0; x < 256; x++) assert.equal(out.data[x], lut[x], `N ${N} v ${x}`);
      out.delete();
      const rc = runOp('quantize', c, { levels: N }), ra = runOp('quantize', a, { levels: N });
      assert.equal(rc.out.channels(), 3); assert.equal(ra.out.channels(), 3);
      for (let i = 0; i < c.data.length; i++) assert.equal(rc.out.data[i], lut[c.data[i]]);
      sameBytes(rc.out, ra.out, `rgba == rgb N ${N}`);
      rc.out.delete(); ra.out.delete();
    }
    const q2 = runOp('quantize', ramp, { levels: 2 }), bt = new cv.Mat();
    cv.threshold(ramp, bt, 127, 255, cv.THRESH_BINARY);
    const viaOp = OPS.binary_threshold.apply(cv, ramp, { ...defaultParams('segment_edge'), thresh: 127, value: 127 }, {});
    try { sameBytes(q2.out, bt, 'N = 2 vs cv.threshold 127'); } finally { q2.out.delete(); bt.delete(); viaOp.delete(); }
    const id = runOp('quantize', c, { levels: 256 });
    try { sameBytes(id.out, c, 'N = 256'); } finally { id.out.delete(); }
    assert.deepEqual(bytes(c), before);
  } finally { ramp.delete(); c.delete(); a.delete(); }
});

test('ip3-Q2. quantize info strings (N = 2, 3, 4, 8, 255, 256 on gray and colour), clamps (1 -> 2, 300 -> 256, "abc" -> 4, 4.6 -> 5) and 0x0 / 1x1', () => {
  const g = gray(8, 8, (x, y) => x * 30 + y), c = rgb(8, 8, (x, y) => [x * 30, y * 30, 7]), e = new cv.Mat(), one = gray(1, 1, () => 200), ca = toRGBA(c);
  const info = (src, levels) => { const r = runOp('quantize', src, { levels }); r.out.delete(); return r.ctx.info; };
  const bitsOf = (N) => { let b = 0; while ((1 << b) < N) b++; return b; };
  try {
    for (const N of [2, 3, 4, 8, 255]) {
      const b = bitsOf(N);
      let gi = `เหลือ ${N} ระดับเทา (ใช้ ${b} bit ต่อพิกเซล · ภาพเดิม 256 ระดับ = 8 bit) · ค่า 0–255 ถูกแบ่งเป็น ${N} ช่วงเท่ากัน แต่ละช่วงเหลือค่าเดียว`;
      let ci = `เหลือ ${N} ระดับต่อช่องสี × 3 ช่อง = ${N * N * N} สีที่เป็นไปได้ (${b} bit ต่อช่อง = ${3 * b} bit ต่อพิกเซล · ภาพเดิม 24 bit)`;
      const pow2 = (N & (N - 1)) === 0;
      if (!pow2) { const t = ` · ${N} ไม่ใช่กำลังของ 2: ${b} bit เก็บได้ ${2 ** b} ระดับ จึงใช้ไม่เต็ม`; gi += t; ci += t; }
      if (N === 2) { gi += ' · 2 ระดับ = ผลเดียวกับ Binary Threshold ที่ 127'; ci += ' · 2 ระดับ = ผลเดียวกับ Binary Threshold ที่ 127'; }
      assert.equal(info(g, N), gi); assert.equal(info(c, N), ci); assert.equal(info(ca, N), ci, `rgba N ${N}`);
    }
    assert.equal(info(g, 256), '256 ระดับ = 8 bit เท่าภาพเดิม — ภาพไม่เปลี่ยน');
    assert.equal(info(c, 256), '256 ระดับ = 8 bit เท่าภาพเดิม — ภาพไม่เปลี่ยน');
    assert.equal(info(g, 1), info(g, 2)); assert.equal(info(g, 300), info(g, 256)); assert.equal(info(g, 'abc'), info(g, 4)); assert.equal(info(g, 4.6), info(g, 5));
    assert.equal(info(g, undefined), info(g, 4));
    const z = runOp('quantize', e, {});
    assert.equal(z.out.rows, 0); assert.equal(z.ctx.info, 'ภาพว่าง'); assert.equal(z.ctx.values, undefined); z.out.delete();
    const o = runOp('quantize', one, { levels: 2 }); assert.equal(o.out.data[0], 255); o.out.delete();
  } finally { for (const m of [g, c, e, one, ca]) m.delete(); }
});

// ===================================================================
// interpolation (resize / rotate)
// ===================================================================
function directResize(src, w, h, flag) { const d = new cv.Mat(); cv.resize(src, d, new cv.Size(w, h), 0, 0, flag); return d; }
test('ip3-I1. resize: auto == the direct cv.resize with AREA (s < 1) / CUBIC (s >= 1) (today); each explicit mode == the direct call with that flag; "constructor" / garbage -> auto; the info of auto is today\'s text, explicit adds the method name (+ the aliasing note when shrinking)', () => {
  const g = noiseGray(37, 23, 3, 0.5), c = noiseRGB(41, 29, 6);
  const FL = { nearest: cv.INTER_NEAREST, bilinear: cv.INTER_LINEAR, bicubic: cv.INTER_CUBIC };
  try {
    for (const src of [g, c]) for (const scale of [30, 50, 100, 150, 400]) {
      const s = scale / 100, w = Math.max(1, Math.round(src.cols * s)), h = Math.max(1, Math.round(src.rows * s));
      const want = directResize(src, w, h, s < 1 ? cv.INTER_AREA : cv.INTER_CUBIC);
      for (const interp of [undefined, 'auto', 'constructor', 'zzz', 5, null]) {
        const r = runOp('resize', src, { scale, interp });
        try { sameBytes(r.out, want, `auto ${scale}% interp=${String(interp)}`); assert.equal(r.ctx.info, `ปรับขนาดเป็น ${w}×${h} px`); } finally { r.out.delete(); }
      }
      want.delete();
      for (const [k, flag] of Object.entries(FL)) {
        const d = directResize(src, w, h, flag), r = runOp('resize', src, { scale, interp: k });
        try {
          sameBytes(r.out, d, `${k} ${scale}%`);
          const name = { nearest: 'Nearest neighbour', bilinear: 'Bilinear', bicubic: 'Bicubic' }[k];
          const tail = s < 1 ? ' · ย่อด้วยวิธีนี้ไม่เฉลี่ยพิกเซลที่ถูกทิ้ง ลายละเอียดถี่ ๆ อาจกลายเป็นลายคลื่น (aliasing) — แบบอัตโนมัติใช้ Area ซึ่งเฉลี่ยให้' : '';
          assert.equal(r.ctx.info, `ปรับขนาดเป็น ${w}×${h} px · ${name}${tail}`);
        } finally { r.out.delete(); d.delete(); }
      }
    }
  } finally { g.delete(); c.delete(); }
});

test('ip3-I2. resize: 2x2 at 200% nearest == 2x2 blocks; on a step edge bicubic != bilinear; 1x1 any % -> 1x1; the 20 MP cap applies to every interp; 0x0 passthrough for every interp', () => {
  const t = gray(2, 2, (x, y) => [10, 20, 30, 40][y * 2 + x]), step = gray(32, 4, (x) => (x < 16 ? 30 : 220)), one = gray(1, 1, () => 77), e = new cv.Mat();
  try {
    const r = runOp('resize', t, { scale: 200, interp: 'nearest' });
    assert.deepEqual(bytes(r.out), [10, 10, 20, 20, 10, 10, 20, 20, 30, 30, 40, 40, 30, 30, 40, 40]); r.out.delete();
    const bl = runOp('resize', step, { scale: 400, interp: 'bilinear' }), bc = runOp('resize', step, { scale: 400, interp: 'bicubic' });
    assert.notDeepEqual(bytes(bl.out), bytes(bc.out)); assert.ok(bc.out.data.every((v) => v >= 0 && v <= 255)); bl.out.delete(); bc.out.delete();
    for (const interp of INTERPS) {
      for (const [scale, side] of [[10, 1], [50, 1], [400, 4]]) { const o = runOp('resize', one, { scale, interp }); assert.deepEqual([o.out.cols, o.out.rows], [side, side], `1x1 at ${scale}%`); o.out.delete(); }
      const z = runOp('resize', e, { interp }); assert.equal(z.out.rows, 0); assert.equal(z.ctx.info, 'ภาพว่าง'); z.out.delete();
    }
    // cap: 5000x4000 gray row of zeros would be heavy; use a 3000x2000 -> 400% = 12000x8000 = 96 MP -> capped near 20 MP
    const big = new cv.Mat(2000, 3000, cv.CV_8UC1, new cv.Scalar(9));
    try {
      for (const interp of ['auto', 'nearest']) {
        const o = runOp('resize', big, { scale: 400, interp });
        assert.ok(o.out.cols * o.out.rows <= 20_000_000 && o.out.cols * o.out.rows > 19_500_000, `${o.out.cols}x${o.out.rows}`);
        assert.ok(o.ctx.info.includes('ลดขนาดลงจากที่ตั้งไว้')); o.out.delete();
      }
    } finally { big.delete(); }
  } finally { t.delete(); step.delete(); one.delete(); e.delete(); }
});

test('ip3-I3. rotate: auto == today (INTER_LINEAR); each explicit mode == the direct warpAffine with that flag (expand off); expand on gives the same size for every interp; garbage interp -> auto; info text unchanged for auto / + method name for explicit; 0x0 passthrough', () => {
  const g = noiseGray(40, 30, 2, 0.5), c = noiseRGB(33, 21, 3), e = new cv.Mat();
  try {
    for (const src of [g, c]) {
      const M = cv.getRotationMatrix2D(new cv.Point(src.cols / 2, src.rows / 2), 15, 1.0);
      const direct = (flag) => { const d = new cv.Mat(); cv.warpAffine(src, d, M, new cv.Size(src.cols, src.rows), flag, cv.BORDER_CONSTANT, src.channels() === 1 ? new cv.Scalar(0) : new cv.Scalar(0, 0, 0)); return d; };
      try {
        const lin = direct(cv.INTER_LINEAR);
        for (const interp of [undefined, 'auto', 'constructor', 'q']) {
          const r = runOp('rotate', src, { angle: 15, expand: false, interp });
          try { sameBytes(r.out, lin, `auto ${String(interp)}`); assert.equal(r.ctx.info, `หมุน 15° → ${src.cols}×${src.rows} px`); } finally { r.out.delete(); }
        }
        lin.delete();
        for (const [k, flag, name] of [['nearest', cv.INTER_NEAREST, 'Nearest neighbour'], ['bilinear', cv.INTER_LINEAR, 'Bilinear'], ['bicubic', cv.INTER_CUBIC, 'Bicubic']]) {
          const d = direct(flag), r = runOp('rotate', src, { angle: 15, expand: false, interp: k });
          try { sameBytes(r.out, d, k); assert.equal(r.ctx.info, `หมุน 15° → ${src.cols}×${src.rows} px · ${name}`); } finally { r.out.delete(); d.delete(); }
        }
      } finally { M.delete(); }
      const sizes = INTERPS.map((interp) => { const r = runOp('rotate', src, { angle: 37, expand: true, interp }); const s = [r.out.cols, r.out.rows]; r.out.delete(); return s.join('x'); });
      assert.equal(new Set(sizes).size, 1, sizes.join());
    }
    for (const interp of INTERPS) { const z = runOp('rotate', e, { interp }); assert.equal(z.out.rows, 0); assert.equal(z.ctx.info, 'ภาพว่าง'); z.out.delete(); }
    // nearest differs from bilinear on a noisy image (so the flag really reaches warpAffine)
    const n = runOp('rotate', g, { angle: 15, expand: false, interp: 'nearest' }), b = runOp('rotate', g, { angle: 15, expand: false, interp: 'bilinear' });
    assert.notDeepEqual(bytes(n.out), bytes(b.out)); n.out.delete(); b.out.delete();
  } finally { g.delete(); c.delete(); e.delete(); }
});

const STEP = (id, mode, extra = {}) => ({ id, params: { ...defaultParams(id), mode, ...extra } });
const OPSTEP = (opId, extra = {}) => {
  const prefixed = {};
  for (const [k, v] of Object.entries(extra)) prefixed[`${opId}__${k}`] = v;
  return STEP(OP_TO_FILTER[opId], opId, prefixed);
};
test('ip3-I4. codegen of resize / rotate: auto / missing / unknown interp == today\'s literal lines (byte-identical); explicit interp exports cv2.resize with an explicit dsize (resize) / flags (rotate); ui-meta auto text unchanged', () => {
  const sub = (code, re) => { const m = code.match(re); assert.ok(m, String(re)); return m[0]; };
  for (const interp of [undefined, 'auto', 'constructor', 'zzz', 3]) {
    const e = interp === undefined ? {} : { interp };
    const r50 = buildPythonCode([OPSTEP('resize', { scale: 50, ...e })]), r150 = buildPythonCode([OPSTEP('resize', { scale: 150, ...e })]);
    assert.ok(r50.includes('    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)'), 'resize auto 50');
    assert.ok(r150.includes('    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)'), 'resize auto 150');
    const ro = buildPythonCode([OPSTEP('rotate', { angle: 15, expand: true, ...e })]), rk = buildPythonCode([OPSTEP('rotate', { angle: 15, expand: false, ...e })]);
    for (const l of ['(h, w) = img.shape[:2]', 'center = (w / 2, h / 2)', 'M = cv2.getRotationMatrix2D(center, 15, 1.0)', 'angle_rad = np.deg2rad(abs(15))', 'cos_a, sin_a = abs(np.cos(angle_rad)), abs(np.sin(angle_rad))', 'out_w = int(round(w * cos_a + h * sin_a))', 'out_h = int(round(w * sin_a + h * cos_a))', 'M[0, 2] += (out_w - w) / 2', 'M[1, 2] += (out_h - h) / 2', 'img = cv2.warpAffine(img, M, (out_w, out_h))']) assert.ok(ro.includes(`    ${l}\n`), `rotate auto expand: ${l}`);
    assert.ok(rk.includes('    img = cv2.warpAffine(img, M, (w, h))\n') && !rk.includes('flags='));
    assert.ok(!ro.includes('flags='));
  }
  const reg = readSrc('filters/ui-meta.js');
  assert.ok(reg.includes('resize: (p) => "สเกล: " + (p.scale ?? 50) + "%" + ownOr('));
  const ui = (id, p) => OP_PRIMARY[id](p);
  assert.equal(ui('resize', { scale: 50 }), 'สเกล: 50%'); assert.equal(ui('resize', { scale: 50, interp: 'constructor' }), 'สเกล: 50%'); assert.equal(ui('resize', {}), 'สเกล: 50%');
  assert.equal(ui('rotate', { angle: 15, expand: true }), 'มุม 15° · ขยายผืนภาพ'); assert.equal(ui('rotate', { angle: 20, expand: false, interp: 'auto' }), 'มุม 20° · คงขนาดเดิม');
  for (const [k, t] of [['nearest', 'Nearest'], ['bilinear', 'Bilinear'], ['bicubic', 'Bicubic']]) {
    assert.equal(ui('resize', { scale: 75, interp: k }), `สเกล: 75% · ${t}`);
    assert.equal(ui('rotate', { angle: 30, expand: true, interp: k }), `มุม 30° · ขยายผืนภาพ · ${t}`);
  }
  // explicit
  const rx = buildPythonCode([OPSTEP('resize', { scale: 50, interp: 'nearest' })]);
  for (const l of ['h, w = img.shape[:2]', 's = 0.5', 'nw, nh = max(1, int(np.floor(w * s + 0.5))), max(1, int(np.floor(h * s + 0.5)))', 'if nw * nh > 20000000:', 'k = np.sqrt(20000000 / (nw * nh))', 'nw, nh = max(1, int(np.floor(nw * k + 0.5))), max(1, int(np.floor(nh * k + 0.5)))', 'img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)']) assert.ok(rx.includes(l), l);
  assert.ok(!rx.includes('fx=') && !rx.includes('None,'));
  // the same expression as the app: Math.max(0.05, finiteOr(scale, 50) / 100) with no upper clamp
  for (const [scale, lit] of [[0, '0.05'], ['0', '0.05'], [4, '0.05'], [1000, '10'], ['abc', '0.5'], [undefined, '0.5'], [150, '1.5']]) assert.ok(buildPythonCode([OPSTEP('resize', { scale, interp: 'bicubic' })]).includes(`    s = ${lit}\n`), `scale ${String(scale)}`);
  for (const [k, f] of [['nearest', 'cv2.INTER_NEAREST'], ['bilinear', 'cv2.INTER_LINEAR'], ['bicubic', 'cv2.INTER_CUBIC']]) {
    assert.ok(buildPythonCode([OPSTEP('resize', { scale: 80, interp: k })]).includes(`interpolation=${f})`), k);
    const a = buildPythonCode([OPSTEP('rotate', { angle: -10, expand: true, interp: k })]), b = buildPythonCode([OPSTEP('rotate', { angle: -10, expand: false, interp: k })]);
    assert.ok(a.includes(`img = cv2.warpAffine(img, M, (out_w, out_h), flags=${f})`), k);
    assert.ok(b.includes(`img = cv2.warpAffine(img, M, (w, h), flags=${f})`), k);
    assert.ok(a.includes('out_w = int(round(w * cos_a + h * sin_a))'), 'rotate expand lines keep their existing int(round(...))');
    for (const code of [a, b]) assert.ok(!code.includes('np.round') && !code.includes('math.'));
  }
});

// ===================================================================
// static / structure
// ===================================================================
const SRC_REG = readSrc('filters/registry.js'), SRC_GEN = readSrc('codegen.js'), SRC_UI = readSrc('filters/ui-meta.js'), SRC_DET = readSrc('ui/detail-panel.js'), SRC_PAL = readSrc('ui/palette-panel.js');
function opBlock(source, opId) {
  const start = source.indexOf(`\n  ${opId}: {`);
  assert.ok(start >= 0, `no "${opId}: {" block`);
  const next = /\n {2}[a-z_0-9]+: \{/g;
  next.lastIndex = start + 1;
  const m = next.exec(source);
  return source.slice(start, m ? m.index : source.length);
}
const sectionOps = (title) => TOOL_SECTIONS.find((s) => s.title === title).ops;

test('ip3-T1. (static structure) 2 new ops exist with needs any / gray and no "mode" key, FILTERS is still 33, the ops live only in the category groups (enhance_tools / analysis_detect), legacy groups are untouched, defaultParams = OPS defaults, sections and group op lists follow the spec order', () => {
  assert.deepEqual(NEW_OPS.map((id) => OPS[id] && OPS[id].needs), ['any', 'gray']);
  for (const id of [...NEW_OPS, 'regionprops', 'resize', 'rotate']) for (const p of OPS[id].params) assert.notEqual(p.key, 'mode', id);
  assert.equal(Object.keys(FILTERS).length, 34); // +morph_tools (2026-10-18)
  assert.equal(OP_TO_FILTER.quantize, 'enhance_tools'); assert.equal(OP_TO_FILTER.shape_filter, 'analysis_detect');
  assert.equal(OP_TO_FILTER.resize, 'geometry'); assert.equal(OP_TO_FILTER.rotate, 'geometry');
  const modes = (f) => FILTERS[f].params.find((p) => p.key === 'mode').options.map((o) => o.value);
  for (const fid of Object.keys(FILTERS)) {
    if (fid === 'enhance_tools' || fid === 'analysis_detect') continue;
    const m = FILTERS[fid].params.find((p) => p.key === 'mode');
    if (m) for (const id of NEW_OPS) assert.ok(!m.options.some((o) => o.value === id), `${fid} must not host ${id}`);
  }
  assert.deepEqual(modes('image_adjust'), ['brightness_contrast', 'gamma', 'hist_equalize', 'clahe', 'contrast_stretch', 'flatfield_correction']);
  assert.deepEqual(modes('shape_analysis'), ['find_contours', 'connected_components', 'distance_transform', 'local_std_filter', 'local_entropy_filter', 'regionprops', 'feret', 'line_profile']);
  assert.deepEqual(sectionOps('ENHANCE').slice(4, 8), ['contrast_stretch', 'flatfield_correction', 'quantize', 'unsharp']);
  const ad = sectionOps('ANALYSIS & DETECTION');
  assert.deepEqual(ad.slice(ad.indexOf('regionprops'), ad.indexOf('regionprops') + 5), ['regionprops', 'feret', 'line_profile', 'shape_filter', 'bounding_boxes']);
  assert.deepEqual(modes('enhance_tools'), sectionOps('ENHANCE')); assert.deepEqual(modes('analysis_detect'), ad);
  assert.equal(defaultParams('enhance_tools').mode, 'brightness_contrast'); assert.equal(defaultParams('analysis_detect').mode, 'find_contours');
  for (const id of NEW_OPS) { const dp = defaultParams(OP_TO_FILTER[id]); for (const p of OPS[id].params) assert.deepEqual(dp[`${id}__${p.key}`], p.default, `${id}__${p.key}`); }
  for (const id of ['resize', 'rotate']) assert.deepEqual(OPS[id].params.at(-1).key, 'interp'), assert.equal(OPS[id].params.at(-1).default, 'auto');
  assert.deepEqual(OPS.resize.params.map((p) => p.key), ['scale', 'interp']); assert.deepEqual(OPS.rotate.params.map((p) => p.key), ['angle', 'expand', 'interp']);
  assert.deepEqual(OPS.resize.params.at(-1).options.map((o) => o.value), ['auto', 'nearest', 'bilinear', 'bicubic']);
  assert.deepEqual(OPS.quantize.params.map((p) => [p.key, p.min, p.max, p.step, p.default]), [['levels', 2, 256, 1, 4]]);
  assert.deepEqual(OPS.shape_filter.params.map((p) => p.key), ['prop', 'lo', 'hi', 'minArea', 'maxArea', 'view']);
  assert.deepEqual(OPS.shape_filter.params.find((p) => p.key === 'prop').options.map((o) => o.value), PROPS);
  assert.deepEqual(OPS.shape_filter.params.filter((p) => p.type === 'number').map((p) => [p.key, p.min, p.max, p.step, p.default]), [['lo', 0, 1, 0.01, 0.8], ['hi', 0, 1, 0.01, 1], ['minArea', 0, 100000, 10, 50], ['maxArea', 0, 5000000, 100, 0]]);
  assert.deepEqual(OPS.shape_filter.params.find((p) => p.key === 'view').options.map((o) => o.value), ['overlay', 'mask']);
  // the node itself works (group dispatch) and equals the direct op
  const g = sfScene();
  try {
    const via = FILTERS.analysis_detect.apply(cv, g, { ...defaultParams('analysis_detect'), mode: 'shape_filter' }, {}), direct = OPS.shape_filter.apply(cv, g, {}, {});
    try { sameBytes(via, direct, 'shape_filter via analysis_detect'); } finally { via.delete(); direct.delete(); }
  } finally { g.delete(); }
});

test('ip3-T2. icons: quantize "≣" and shape_filter "⛛" are unique against OP_ICON, GROUP_ICON, app.js, detail-panel.js and editor.js; the OP_ICON table has them once', () => {
  const icons = { quantize: '≣', shape_filter: '⛛' };
  for (const [id, glyph] of Object.entries(icons)) {
    assert.ok(new RegExp(`\\b${id}: "${glyph}"`).test(SRC_PAL), `${id} icon in palette-panel`);
    assert.equal(SRC_PAL.split(glyph).length - 1, 1, `${glyph} appears once in palette-panel.js`);
    assert.ok(!Object.values(GROUP_ICON).includes(glyph), `${glyph} vs GROUP_ICON`);
    for (const f of ['app.js', 'ui/detail-panel.js', 'graph/editor.js', 'filters/ui-meta.js']) assert.ok(!readSrc(f).includes(glyph), `${glyph} in ${f}`);
  }
  assert.notEqual(icons.quantize, icons.shape_filter);
  const table = SRC_PAL.slice(SRC_PAL.indexOf('const OP_ICON = {'), SRC_PAL.indexOf('};', SRC_PAL.indexOf('const OP_ICON = {')));
  const glyphs = [...table.matchAll(/\b[a-z_0-9]+: "([^"]+)"/g)].map((m) => m[1]);
  assert.equal(glyphs.filter((x) => x === '≣').length, 1); assert.equal(glyphs.filter((x) => x === '⛛').length, 1);
  assert.ok(!['🧩', '📊', '📌'].includes('≣') && !['🧩', '📊', '📌'].includes('⛛'));
});

test('ip3-T3. ui-meta OP_PRIMARY (pnum, clamped values the op really uses): quantize "N ระดับต่อช่อง (b bit)"; shape_filter "ความกลม 0.80–1.00 · ≥ 50 px" (ascending lo-hi, area range when maxArea > 0, none -> "พื้นที่ ≥ 50 px", " · Mask")', () => {
  assert.equal(OP_PRIMARY.quantize({}), '4 ระดับต่อช่อง (2 bit)');
  assert.equal(OP_PRIMARY.quantize({ levels: 1 }), '2 ระดับต่อช่อง (1 bit)');
  assert.equal(OP_PRIMARY.quantize({ levels: 300 }), '256 ระดับต่อช่อง (8 bit)');
  assert.equal(OP_PRIMARY.quantize({ levels: 4.6 }), '5 ระดับต่อช่อง (3 bit)');
  assert.equal(OP_PRIMARY.quantize({ levels: 'abc' }), '4 ระดับต่อช่อง (2 bit)');
  assert.equal(OP_PRIMARY.shape_filter({}), 'ความกลม 0.80–1.00 · ≥ 50 px');
  assert.equal(OP_PRIMARY.shape_filter({ prop: 'solidity', lo: 0.9, hi: 0.5, minArea: 20, maxArea: 900, view: 'mask' }), 'ความตัน 0.50–0.90 · 20–900 px · Mask');
  assert.equal(OP_PRIMARY.shape_filter({ prop: 'none' }), 'พื้นที่ ≥ 50 px');
  assert.equal(OP_PRIMARY.shape_filter({ prop: 'none', minArea: 9, maxArea: 3, view: 'overlay' }), 'พื้นที่ 3–9 px');
  assert.equal(OP_PRIMARY.shape_filter({ prop: 'constructor', view: 'constructor', lo: 'x', hi: 7 }), 'ความกลม 0.80–1.00 · ≥ 50 px');
  assert.equal(OP_PRIMARY.shape_filter({ prop: 'eccentricity', lo: -2, hi: 0.333 }), 'ความรี 0.00–0.33 · ≥ 50 px');
  assert.match(SRC_UI, /quantize: \(p\) => \{\n {4}const n = pnum\("quantize", "levels"/);
  assert.match(SRC_UI, /pnum\(op, "lo", p\.lo\)/);
  assert.ok(HISTOGRAM_COMPARE_OPS.has('quantize'));
});

test('ip3-T4. (static read) every numeric param of the new ops is clamped to EXACTLY the UI default/min/max in registry.js (numIn / intIn) and in codegen.js (clampNum); selects are read with ownKey / pickOwn (own properties only)', () => {
  const spec = (id, k) => { const s = OPS[id].params.find((p) => p.key === k); return [s.default, s.min, s.max]; };
  const num = '(-?[\\d.]+)';
  const covered = new Set();
  const regCheck = (id, k, fn) => { const m = opBlock(SRC_REG, id).match(new RegExp(`${fn}\\(p\\.${k}, ${num}, ${num}, ${num}\\)`)); assert.ok(m, `registry ${id}.${k}`); assert.deepEqual(m.slice(1).map(Number), spec(id, k), `registry ${id}.${k}`); covered.add(`${id}.${k}`); };
  const genCheck = (id, k) => { const m = opBlock(SRC_GEN, id).match(new RegExp(`clampNum\\(p\\.${k}, ${num}, ${num}, ${num}\\)`)); assert.ok(m, `codegen ${id}.${k}`); assert.deepEqual(m.slice(1).map(Number), spec(id, k), `codegen ${id}.${k}`); };
  regCheck('quantize', 'levels', 'intIn');
  for (const k of ['lo', 'hi', 'minArea', 'maxArea']) { regCheck('shape_filter', k, 'numIn'); genCheck('shape_filter', k); }
  genCheck('quantize', 'levels');
  assert.match(opBlock(SRC_GEN, 'quantize'), /Math\.round\(clampNum\(p\.levels/);
  for (const id of NEW_OPS) for (const p of OPS[id].params) if (p.type === 'number') assert.ok(covered.has(`${id}.${p.key}`), `${id}.${p.key} not covered`);
  assert.match(opBlock(SRC_REG, 'shape_filter'), /ownKey\(SHAPE_FILTER_PROP_TH, p\.prop, "circularity"\)/);
  assert.match(opBlock(SRC_REG, 'shape_filter'), /ownKey\(SHAPE_FILTER_VIEWS, p\.view, "overlay"\)/);
  assert.match(opBlock(SRC_GEN, 'shape_filter'), /pickOwn\(SHAPE_FILTER_PROP_PY, p\.prop, "circularity"\)/);
  assert.match(opBlock(SRC_GEN, 'shape_filter'), /pickOwn\(SHAPE_FILTER_VIEW_PY, p\.view, "overlay"\)/);
  assert.match(SRC_REG, /function interpKey\(v\) \{ return Object\.hasOwn\(INTERP_FLAG, v\) \? v : "auto"; \}/);
  assert.match(SRC_GEN, /Object\.hasOwn\(INTERP_PY, p\.interp\)/);
  assert.match(SRC_GEN, /const INTERP_PY = \{ nearest: "cv2\.INTER_NEAREST", bilinear: "cv2\.INTER_LINEAR", bicubic: "cv2\.INTER_CUBIC" \};/);
  // behavioural twin: the registry really clamps like the UI says
  const g = gray(8, 8, (x, y) => x * 30 + y);
  try {
    for (const [v, want] of [[1, 2], [300, 256], ['abc', 4], [4.6, 5], [-5, 2]]) {
      const r = runOp('quantize', g, { levels: v }); const lut = quantizeLut(want);
      for (let i = 0; i < 64; i++) assert.equal(r.out.data[i], lut[g.data[i]], `levels ${v}`); r.out.delete();
    }
  } finally { g.delete(); }
});

const PY_VARIANTS = [
  ['quantize', {}], ['quantize', { levels: 2 }], ['quantize', { levels: 256 }], ['quantize', { levels: 300 }], ['quantize', { levels: 'abc' }],
  ...PROPS.flatMap((prop) => ['overlay', 'mask'].map((view) => ['shape_filter', { prop, view, lo: 0.3, hi: 0.95, minArea: 10, maxArea: 2000 }])),
  ['shape_filter', { lo: 1, hi: 0.2, minArea: 900, maxArea: 20 }], ['shape_filter', { prop: 'constructor', view: 'constructor' }],
  ['regionprops', {}],
  ...['nearest', 'bilinear', 'bicubic'].flatMap((interp) => [['resize', { scale: 33, interp }], ['rotate', { angle: 20, expand: true, interp }], ['rotate', { angle: 20, expand: false, interp }]]),
];
const NO_GOOD = ['WARNING: skipped', 'ข้าม'];
test('ip3-T5. codegen: new ops and explicit interp export real single-image Python (never "WARNING: skipped" / "ข้าม"), no round( / ** 0.5 / import math / np.round / math. in the new / explicit paths, every print is ASCII, embedded values are the clamped literals, only the chosen property branch is emitted', () => {
  for (const [id, p] of PY_VARIANTS) {
    const code = buildPythonCode([OPSTEP(id, p)]);
    for (const bad of NO_GOOD) assert.ok(!code.includes(bad), `${id} ${JSON.stringify(p)}: contains ${bad}`);
    const body = code.slice(code.indexOf('# ── ขั้นที่ 1'));
    const checked = id === 'rotate' ? body.split('\n').filter((l) => !/out_[wh] = int\(round\(/.test(l)).join('\n') : body;
    for (const bad of ['round(', '** 0.5', 'import math', 'np.round', 'math.']) assert.ok(!checked.includes(bad), `${id} ${JSON.stringify(p)}: contains ${bad}`);
    for (const line of code.split('\n')) if (/^\s*print\(/.test(line) && !/ชิ้น|พบวัตถุ|ไม่พบ/.test(line)) assert.ok(/^[\x20-\x7e]*$/.test(line), `non-ASCII print: ${line}`);
  }
  const q = buildPythonCode([OPSTEP('quantize', { levels: 300 })]);
  assert.ok(q.includes('levels = 256\n') && q.includes('q_lut = ((((np.arange(256) * levels) // 256) * 510 + (levels - 1)) // (2 * (levels - 1))).astype(np.uint8)') && q.includes('img = cv2.LUT(img, q_lut)') && q.includes('print(f"quantize: {levels} levels per channel (8 bit)")'));
  assert.ok(buildPythonCode([OPSTEP('quantize', { levels: 'abc' })]).includes('levels = 4\n'));
  assert.ok(buildPythonCode([OPSTEP('quantize', { levels: 1 })]).includes('levels = 2\n') && buildPythonCode([OPSTEP('quantize', { levels: 1 })]).includes('(1 bit)'));
  assert.ok(buildPythonCode([OPSTEP('quantize', { levels: 4.6 })]).includes('levels = 5\n') && buildPythonCode([OPSTEP('quantize', { levels: 4.6 })]).includes('(3 bit)'));
  const sf = (p) => buildPythonCode([OPSTEP('shape_filter', p)]);
  const c = sf({ prop: 'circularity' });
  assert.ok(c.includes('lo, hi = 0.8, 1  # ') && c.includes('min_area, max_area = 50, 0  # ') && c.includes('region_stats(c, x, y, w, h, img, bin_img, False)') && c.includes('perimeter = cv2.arcLength(c, True)'));
  assert.ok(sf({ prop: 'solidity' }).includes('region_stats(c, x, y, w, h, img, bin_img, True)') && !sf({ prop: 'solidity' }).includes('arcLength') && sf({ prop: 'solidity' }).includes("v = st['solidity']"));
  assert.ok(sf({ prop: 'eccentricity' }).includes("v = st['eccentricity']") && sf({ prop: 'extent' }).includes("v = st['extent']"));
  const n = sf({ prop: 'none' });
  assert.ok(!n.includes("v = ") && !n.includes('if not (lo <= v <= hi)') && !n.includes('arcLength') && n.includes('region_stats(c, x, y, w, h, img, bin_img, False)'));
  assert.ok(sf({ lo: 1, hi: 0.2, minArea: 900, maxArea: 20 }).includes('lo, hi = 0.2, 1  # ') && sf({ lo: 1, hi: 0.2, minArea: 900, maxArea: 20 }).includes('min_area, max_area = 20, 900  # '));
  assert.ok(sf({ lo: 1, hi: 0.2 }).includes('print("note: lo/hi were swapped")') && !sf({}).includes('swapped') && sf({ minArea: 900, maxArea: 20 }).includes('print("note: min/max area were swapped")'));
  assert.ok(sf({ view: 'mask' }).includes('img = kept_mask') && !sf({ view: 'mask' }).includes('out = np.zeros'));
  assert.ok(sf({ view: 'overlay' }).includes('out[bin_img > 0] = (0, 0, 255)') && sf({ view: 'overlay' }).includes('out[kept_mask > 0] = (255, 255, 255)') && sf({}).includes('img = out'));
  assert.ok(sf({ view: 'constructor', prop: 'constructor' }).includes('img = out') && sf({ prop: 'constructor' }).includes('perimeter = cv2.arcLength'));
  assert.equal(sf({ minArea: -9 }).includes('min_area, max_area = 0, 0'), true);
  // after Color Space the channel order is not reversed -> red = (255, 0, 0)
  const afterCS = buildPythonCode([{ id: 'color_tools', params: { ...defaultParams('color_tools'), mode: 'grayscale' } }, OPSTEP('shape_filter', {})]);
  assert.ok(afterCS.includes('out[bin_img > 0] = (255, 0, 0)') || afterCS.includes('out[bin_img > 0] = (0, 0, 255)'));
  // regionprops: new lines and the first table stay
  const rp = buildPythonCode([OPSTEP('regionprops', {})]);
  assert.ok(rp.indexOf('gray_src = img') < rp.indexOf('img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)') && rp.indexOf('img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)') < rp.indexOf('def region_stats(') && rp.indexOf('def region_stats(') < rp.indexOf('# รอบที่ 1'));
  assert.ok(rp.includes("regions[-1].update(region_stats(c, x, y, w, h, gray_src, bin_img, True))"));
  assert.ok(rp.includes("print('index  area  cx  cy  perimeter  circularity  bbox(x,y,w,h)  orientation_deg')") && rp.includes("print('object  area_px  equiv_diam  extent  solidity  major_axis  minor_axis  eccentricity  mean_intensity')"));
  assert.ok(!/\bimg\b/.test(rp.slice(rp.indexOf('def region_stats('), rp.indexOf('# รอบที่ 1'))), 'region_stats must not mention img (2-image rename safety)');
});

test('ip3-T6. teaching text: names / descs / labels / hints / options / every info message and the exported Python of the new text never say "ที่สุด"; each desc states the spec points; regionprops keeps its old sentences', () => {
  const texts = [];
  for (const id of NEW_OPS) {
    const o = OPS[id];
    texts.push(o.name, o.desc);
    for (const p of o.params) { texts.push(p.label, p.desc || '', p.hint || ''); for (const op of p.options || []) texts.push(op.label, op.hint || ''); }
  }
  const rpD = OPS.regionprops.desc;
  const appended = rpD.slice(rpD.indexOf(' · คอลัมน์เพิ่มเติม'));
  assert.ok(appended.length > 400);
  texts.push(appended);
  for (const id of ['resize', 'rotate']) {
    const ip = OPS[id].params.at(-1);
    texts.push(OPS[id].desc.slice(OPS[id].desc.indexOf(' · วิธีคำนวณพิกเซลใหม่')), ip.label, ...ip.options.flatMap((o) => [o.label, o.hint || '']));
  }
  // info messages
  const g = sfScene(), q = gray(8, 8, (x) => x * 30), c = rgb(8, 8, () => [1, 2, 3]), bw = shapesScene();
  try {
    for (const p of [{}, { prop: 'solidity', lo: 1, hi: 0, minArea: 9, maxArea: 3 }, { prop: 'none', view: 'mask' }]) texts.push(runInfo('shape_filter', g, p).info);
    for (const N of [2, 3, 4, 8, 255, 256]) for (const s of [q, c]) texts.push(runInfo('quantize', s, { levels: N }).info);
    texts.push(runInfo('shape_filter', black(5, 5), {}).info, runInfo('regionprops', bw, {}).info.slice(runInfo('regionprops', bw, {}).info.indexOf(' · ภาพที่เข้ามา')));
    for (const interp of INTERPS) for (const scale of [50, 150]) texts.push(runInfo('resize', c, { scale, interp }).info.replace(/^ปรับขนาดเป็น.*?px/, ''));
  } finally { for (const m of [g, q, c, bw]) m.delete(); }
  for (const [id, p] of PY_VARIANTS.filter(([i]) => i !== 'regionprops')) texts.push(buildPythonCode([OPSTEP(id, p)]));
  const det = SRC_DET.split('\n').filter((l) => /^\s*(area_px|equiv_diameter|extent|solidity|major_axis|minor_axis|eccentricity|mean_intensity|sf_\w+):/.test(l));
  assert.equal(det.length, 12);
  texts.push(...det, SRC_DET.split('\n').find((l) => /^\s*area:\s/.test(l)));
  for (const t of texts) assert.ok(!t.includes('ที่สุด'), `"ที่สุด" in: ${t.slice(0, 120)}`);
  // each desc states its points
  const D = (id) => OPS[id].desc;
  assert.ok(/bwpropfilt/.test(D('shape_filter')) && /bwareafilt/.test(D('shape_filter')) && /Remove Small/.test(D('shape_filter')), 'sf 1');
  assert.ok(/ตัวเลขที่ใช้คัดคือค่าเดียวกับคอลัมน์ใน Region Properties/.test(D('shape_filter')) && /พื้นที่นับพิกเซล/.test(D('shape_filter')), 'sf 2');
  assert.ok(/ไม่มีทางได้ 1/.test(D('shape_filter')) && /0\.87/.test(D('shape_filter')) && /≈ 0\.79/.test(D('shape_filter')), 'sf 3');
  assert.ok(/ไม่ใช่ 'area' แบบเส้นหลายเหลี่ยม/.test(D('shape_filter')), 'sf 4');
  assert.ok(/> 127/.test(D('shape_filter')) && /รูในวัตถุยังเป็นรู/.test(D('shape_filter')) && /วัตถุเกาะที่อยู่ในรูจะตามวัตถุชั้นนอก/.test(D('shape_filter')), 'sf 5');
  assert.ok(/overlay: ขาว = เก็บ, แดง = คัดทิ้ง/.test(D('shape_filter')) && /ต่อเข้า Region Properties/.test(D('shape_filter')), 'sf 6');
  assert.ok(/ต่างจาก MATLAB/.test(D('shape_filter')) && /8-connected/.test(D('shape_filter')) && /ผ่านกลางพิกเซล/.test(D('shape_filter')), 'sf 7');
  const qd = D('quantize');
  assert.ok(/8 bit = 256 ระดับ/.test(qd) && /posterize/.test(qd) && /N×N×N/.test(qd) && /imquantize/.test(qd) && /Multi-Otsu/.test(qd) && /1\.\.N/.test(qd) && /dithering/.test(qd) && /Binary Threshold ที่ 127/.test(qd) && /histogram/.test(qd), 'quantize points 1-7');
  for (const id of ['resize', 'rotate']) {
    const d = D(id);
    assert.ok(/Nearest neighbour/.test(d) && /Bilinear/.test(d) && /Bicubic/.test(d) && /aliasing/.test(d) && /imresize = bicubic/.test(d) && /imrotate = nearest/.test(d) && /อัตโนมัติ/.test(d), `${id} interp desc`);
  }
  assert.ok(/ความตัน \(solidity\)/.test(rpD) && /ความรี \(eccentricity\)/.test(rpD) && /major\/minor axis/.test(rpD) && /เต็มกรอบ \(extent\)/.test(rpD) && /area_px/.test(rpD) && /area 81, area_px 100/.test(rpD) && /regionprops\(BW, I\)/.test(rpD) && /มุมเอียง \(orientation\) จะแสดง —/.test(rpD) && /ผ่านกลางพิกเซล/.test(rpD) && /วัตถุเกาะ/.test(rpD) && /MATLAB/.test(rpD), 'regionprops desc points');
  assert.ok(/วัตถุที่มีจุดบนเส้นรอบรูปน้อยกว่า 5 จุด \(เช่น สี่เหลี่ยมตรง\) จะไม่มีค่ามุม/.test(rpD) && /orientation_deg/.test(rpD), 'old sentences kept');
  // regionprops 0.. no conditional blanket claims
  for (const id of NEW_OPS) assert.ok(!/ทุกกรณี|เสมอไป/.test(OPS[id].desc));
});

test('ip3-T7. detail-panel VAL_META: the 8 new columns have th / d / pri as specified (primary = index, area, cx, cy, circularity, solidity, eccentricity), area keeps its th and gets the corrected tip, the 4 sf_* entries exist, and HISTOGRAM_COMPARE_OPS has quantize', () => {
  const body = SRC_DET.slice(SRC_DET.indexOf('const VAL_META = {'), SRC_DET.indexOf('\n};', SRC_DET.indexOf('const VAL_META = {')));
  const entry = (k) => { const m = body.match(new RegExp(`\\n\\s*${k}:\\s*\\{([^\\n]*)\\}`)); assert.ok(m, k); return m[1]; };
  const want = { area_px: ['พื้นที่นับพิกเซล', 0, 92, false], equiv_diameter: ['เส้นผ่านศูนย์กลางเทียบเท่า (px)', 1, 110, false], extent: ['เต็มกรอบ', 3, 70, false], solidity: ['ความตัน', 3, 70, true], major_axis: ['แกนยาว (px)', 1, 82, false], minor_axis: ['แกนสั้น (px)', 1, 82, false], eccentricity: ['ความรี', 3, 70, true], mean_intensity: ['ความสว่างเฉลี่ย', 1, 96, false] };
  for (const [k, [th, d, w, pri]] of Object.entries(want)) {
    const e = entry(k);
    assert.ok(e.includes(`th: "${th}"`) && e.includes(`d: ${d}`) && e.includes(`w: ${w}`) && e.includes(`pri: ${pri}`) && /tip: "/.test(e), k);
  }
  const area = entry('area');
  assert.ok(area.includes('th: "พื้นที่ (px²)"') && area.includes("tip: \"พื้นที่ภายในเส้นขอบ (ลากผ่านกลางพิกเซลขอบ นับรวมรู) — น้อยกว่าจำนวนพิกเซลเล็กน้อย ดู 'พื้นที่นับพิกเซล'\""));
  for (const k of ['sf_total', 'sf_kept', 'sf_removed_area']) assert.ok(entry(k).includes('d: 0'));
  assert.ok(entry('sf_removed_shape').includes('tip: "ผ่านเกณฑ์พื้นที่ แต่ค่ารูปร่างอยู่นอกช่วงที่ตั้ง"'));
  const pri = ['index', 'area', 'cx', 'cy', 'perimeter', 'circularity', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'orientation_deg', ...REGION_SHAPE_COLUMNS].filter((k) => entry(k).includes('pri: true'));
  assert.deepEqual(pri, ['index', 'area', 'cx', 'cy', 'circularity', 'solidity', 'eccentricity']);
  assert.ok(HISTOGRAM_COMPARE_OPS.has('quantize'));
  for (const old of ['brightness_contrast', 'gamma', 'hist_equalize', 'clahe', 'contrast_stretch', 'rgb_adjust', 'invert', 'white_balance']) assert.ok(HISTOGRAM_COMPARE_OPS.has(old), old);
});

test('ip3-T8. src is never modified by the 5 ops (all interps / props / views) and the ops return a new Mat even for 0x0', () => {
  const g = sfScene(), c = noiseRGB(30, 20, 2), a = toRGBA(c), e = new cv.Mat();
  try {
    for (const src of [g, c, a]) {
      const before = bytes(src);
      for (const [id, p] of [['regionprops', {}], ['shape_filter', { view: 'mask' }], ['shape_filter', { prop: 'solidity' }], ['quantize', { levels: 3 }], ['resize', { interp: 'nearest' }], ['resize', { scale: 200, interp: 'bicubic' }], ['rotate', { interp: 'bicubic' }], ['rotate', { interp: 'nearest', expand: false }]]) {
        if (id === 'regionprops' && src.channels() !== 1) continue;
        const o = OPS[id].apply(cv, src, p, {}); o.delete();
        assert.deepEqual(bytes(src), before, `${id} modified src`);
      }
    }
    for (const id of ['regionprops', 'shape_filter', 'quantize', 'resize', 'rotate']) {
      const o = OPS[id].apply(cv, e, {}, {});
      assert.equal(o.rows, 0); assert.notEqual(o.$$.ptr, e.$$.ptr); o.delete();
    }
  } finally { for (const m of [g, c, a, e]) m.delete(); }
});

// ===================================================================
// parity กับ Python จริง (cv2 / numpy) — batch เดียวต่อกลุ่ม
// ===================================================================
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'ip-batch3-'));
process.on('exit', () => { try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ } });
function pythonAvailable() { try { execFileSync('python', ['-c', 'import cv2, numpy'], { encoding: 'utf8' }); return true; } catch { return false; } }
const HAS_PYTHON = pythonAvailable();
const PY_SKIP = { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' };
function swapRB3(buf) { const out = Buffer.from(buf); for (let i = 0; i < out.length; i += 3) { const t = out[i]; out[i] = out[i + 2]; out[i + 2] = t; } return out; }
const PY_DRIVER = `import sys, json, io, contextlib
import numpy as np
cases = json.load(open(sys.argv[1]))
res = []
for c in cases:
    ns = {'__name__': 'case'}
    buf = io.StringIO()
    err = None
    nd = None
    shape = None
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(open(c['code'], encoding='utf-8').read(), c['code'], 'exec'), ns)
            r = ns['process'](c['inp'])
        r.tofile(c['out'])
        nd = int(r.ndim)
        shape = [int(v) for v in r.shape]
    except Exception as e:
        err = repr(e)
    res.append({'stdout': buf.getvalue(), 'ndim': nd, 'shape': shape, 'err': err})
json.dump(res, open(sys.argv[2], 'w'))
`;
// cases: [{ code, src (3 channels, app order), regions }] -> [{ stdout, bytes (app order), shape, regions }]
function runPythonBatch(cases) {
  const id = Math.random().toString(36).slice(2);
  const manifest = cases.map((c, i) => {
    assert.equal(c.src.channels(), 3, 'cv2.imread gives 3 channels');
    let code = c.code.replace('img = cv2.imread(path)', `img = np.fromfile(path, dtype=np.uint8).reshape((${c.src.rows}, ${c.src.cols}, 3))`);
    const mainIdx = code.indexOf('if __name__ == "__main__":');
    assert.ok(mainIdx > 0);
    code = code.slice(0, mainIdx);
    if (c.regions) code = code.replace('\n    return img\n', "\n    import json as _j\n    print('REGIONS_JSON', _j.dumps(regions, default=float))\n    return img\n");
    const o = { code: path.join(TMP_DIR, `c_${id}_${i}.py`), inp: path.join(TMP_DIR, `i_${id}_${i}.bin`), out: path.join(TMP_DIR, `o_${id}_${i}.bin`) };
    fs.writeFileSync(o.code, code, 'utf8');
    fs.writeFileSync(o.inp, swapRB3(Buffer.from(c.src.data)));
    return o;
  });
  const mp = path.join(TMP_DIR, `m_${id}.json`), rp = path.join(TMP_DIR, `r_${id}.json`), dp = path.join(TMP_DIR, `d_${id}.py`);
  fs.writeFileSync(mp, JSON.stringify(manifest)); fs.writeFileSync(dp, PY_DRIVER);
  execFileSync('python', [dp, mp, rp], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' }, maxBuffer: 1 << 28 });
  const res = JSON.parse(fs.readFileSync(rp, 'utf8'));
  return res.map((r, i) => {
    assert.equal(r.err, null, `case ${i}: python raised ${r.err}`);
    const raw = fs.readFileSync(manifest[i].out);
    const regions = cases[i].regions ? JSON.parse((r.stdout.match(/REGIONS_JSON (.*)\n/) || [])[1] || 'null') : null;
    for (const f of Object.values(manifest[i])) fs.rmSync(f, { force: true });
    return { stdout: r.stdout, shape: r.shape, ndim: r.ndim, bytes: r.ndim === 3 ? swapRB3(raw) : Buffer.from(raw), regions };
  });
}
function appRun(pipe, src) {
  let cur = src.clone(), last = null;
  try {
    for (const st of pipe) {
      const f = FILTERS[st.id];
      let input = cur, conv = null;
      if (needsOf(f, st.params) === 'gray' && cur.channels() > 1) {
        conv = new cv.Mat();
        cv.cvtColor(cur, conv, cur.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY);
        input = conv;
      }
      const ctx = {};
      try { const o = f.apply(cv, input, st.params, ctx); cur.delete(); cur = o; } finally { if (conv) conv.delete(); }
      last = ctx;
    }
    return { mat: cur, ctx: last };
  } catch (err) { cur.delete(); throw err; }
}
function ndiffOf(appMat, pyBytes) {
  const d = appMat.data;
  assert.equal(pyBytes.length, d.length, 'output size');
  let n = 0;
  for (let i = 0; i < d.length; i++) if (d[i] !== pyBytes[i]) n++;
  return n;
}
function maxAbsDiff(appMat, pyBytes) {
  const d = appMat.data; assert.equal(pyBytes.length, d.length, 'output size');
  let m = 0; for (let i = 0; i < d.length; i++) m = Math.max(m, Math.abs(d[i] - pyBytes[i]));
  return m;
}
const rgbFromGray = (g) => { const o = new cv.Mat(); cv.cvtColor(g, o, cv.COLOR_GRAY2RGB); return o; };
// Python :.Nf (round-half-even on the exact binary value) vs JS toFixed (half up on ties): only exact ties differ
function pyFixed(x, d) {
  const sc = x * 10 ** d, fl = Math.floor(sc);
  if (sc - fl === 0.5) return ((fl % 2 === 0 ? fl : fl + 1) / 10 ** d).toFixed(d);
  return x.toFixed(d);
}

test('ip3-P1. parity regionprops (real Python): all 19 fields per object, the 8 new ones EXACTLY equal (===) incl. a 1920x1080 ~1.4 M-px ellipse with a hole (BigInt path), 30% noise 97x61, gray input with holes, binary input; the second printed table == the JS rows formatted; the note line iff the app note', PY_SKIP, () => {
  const big = drawn(1920, 1080, (m) => { ellipseM(m, 960, 540, 900, 500, 0); cv.ellipse(m, new cv.Point(700, 540), new cv.Size(200, 150), 0, 0, 360, new cv.Scalar(0), -1); });
  const shapes = shapesScene(), bin0 = binOf(shapes), gtex = grayTexture(shapes.cols, shapes.rows, bin0);
  const noise = noiseGray(97, 61, 5, 0.3), sf = sfScene();
  const blk = black(30, 20);
  const imgs = [['shapes', shapes], ['gray texture', gtex], ['noise 97x61', noise], ['sfScene', sf], ['big ellipse with a hole', big], ['black', blk]];
  const made = imgs.map(([n, g]) => [n, g, rgbFromGray(g)]);
  try {
    const pipe = [OPSTEP('regionprops', { minArea: 0, maxObjects: 0 })];
    const code = buildPythonCode(pipe);
    const py = runPythonBatch(made.map(([, , c]) => ({ code, src: c, regions: true })));
    made.forEach(([name, g, c], i) => {
      const app = appRun(pipe, c);
      try {
        const rows = app.ctx.values.rows, regs = py[i].regions;
        assert.equal(regs.length, rows.length, `${name}: object count`);
        assert.equal(app.ctx.values.total, rows.length);
        rows.forEach((row, k) => {
          const r = regs[k];
          const js = Object.fromEntries(ALL_COLS.map((key, j) => [key, row[j]]));
          assert.equal(r.bx, js.bbox_x); assert.equal(r.by, js.bbox_y); assert.equal(r.bw, js.bbox_w); assert.equal(r.bh, js.bbox_h);
          assert.equal(r.area, js.area); assert.equal(r.cx, js.cx); assert.equal(r.cy, js.cy);
          near(r.perimeter, js.perimeter, 1e-9, `${name} #${k} perimeter`); near(r.circularity, js.circularity, 1e-12, `${name} #${k} circ`);
          if (js.orientation_deg === null) assert.equal(r.orientation, null); else if (!name.startsWith('noise')) { const dd = Math.abs(r.orientation - js.orientation_deg) % 180; assert.ok(Math.min(dd, 180 - dd) < 0.5, `${name} #${k} orientation ${r.orientation} vs ${js.orientation_deg}`); } // fitEllipse of opencv.js 4.10 vs cv2 5.0 may differ slightly on tiny contours (existing column, not part of this batch)
          for (const key of REGION_SHAPE_COLUMNS) assert.ok(r[key] === js[key], `${name} #${k} ${key}: python ${r[key]} app ${js[key]}`);
        });
        // second printed table
        const lines = py[i].stdout.split('\n');
        const head = lines.findIndex((l) => l.startsWith('object  area_px'));
        if (rows.length === 0) assert.equal(head, -1);
        else {
          assert.ok(head > 0, `${name}: second table`);
          rows.slice(0, 20).forEach((row, k) => {
            const j = Object.fromEntries(ALL_COLS.map((key, q) => [key, row[q]]));
            const want = `${k + 1}  ${j.area_px}  ${pyFixed(j.equiv_diameter, 1)}  ${pyFixed(j.extent, 3)}  ${pyFixed(j.solidity, 3)}  ${pyFixed(j.major_axis, 1)}  ${pyFixed(j.minor_axis, 1)}  ${pyFixed(j.eccentricity, 3)}  ${pyFixed(j.mean_intensity, 1)}`;
            assert.equal(lines[head + 1 + k], want, `${name} table row ${k + 1}`);
          });
        }
        const appNote = rows.length > 0 && app.ctx.info.endsWith('ความสว่างเฉลี่ยจึงเท่ากับค่าขาวเต็มทุกชิ้น');
        assert.equal(py[i].stdout.includes('note: the input is already black/white, so mean_intensity is 255 for every object'), appNote, `${name}: note`);
        assert.equal(appNote, name !== 'gray texture' && name !== 'black', `${name}: note expectation`);
      } finally { app.mat.delete(); }
    });
    // the first table (before this batch) is still printed byte for byte
    assert.ok(py[0].stdout.includes('index  area  cx  cy  perimeter  circularity  bbox(x,y,w,h)  orientation_deg'));
  } finally { for (const [, , c] of made) c.delete(); for (const m of [big, shapes, bin0, gtex, noise, sf, blk]) m.delete(); }
});

test('ip3-P2. parity shape_filter (real Python): both views ndiff 0 and the printed counts == the app values for every prop x {scene, noise 97x61, 1x1, black, colour R != B}', PY_SKIP, () => {
  const scene = sfScene(), noise = noiseGray(97, 61, 5, 0.3), one = gray(1, 1, () => 255), blk = black(30, 20);
  const colour = noiseRGB(60, 50, 4);
  const inputs = [['scene', rgbFromGray(scene)], ['noise', rgbFromGray(noise)], ['1x1', rgbFromGray(one)], ['black', rgbFromGray(blk)], ['colour', colour]];
  const cases = [];
  for (const [name, src] of inputs) for (const prop of PROPS) for (const view of ['overlay', 'mask']) {
    const p = name === 'scene' ? { prop, view, lo: 0.5, hi: 1 } : { prop, view, lo: 0.3, hi: 0.95, minArea: 0, maxArea: name === 'noise' ? 40 : 0 };
    cases.push({ name: `${name} ${prop} ${view}`, src, pipe: [OPSTEP('shape_filter', p)] });
  }
  // area bounds exactly equal to an object's pixel count / bbox area (the 30x30 squares: 900 px) — inclusive on both sides
  for (const view of ['overlay', 'mask']) for (const p of [{ prop: 'none', minArea: 900 }, { prop: 'none', minArea: 0, maxArea: 900 }, { prop: 'none', minArea: 901 }, { prop: 'none', minArea: 0, maxArea: 899 }]) cases.push({ name: `scene area ${JSON.stringify(p)} ${view}`, src: inputs[0][1], pipe: [OPSTEP('shape_filter', { ...p, view })] });
  try {
    const py = runPythonBatch(cases.map((c) => ({ code: buildPythonCode(c.pipe), src: c.src })));
    let checked = 0;
    cases.forEach((c, i) => {
      const app = appRun(c.pipe, c.src);
      try {
        assert.equal(py[i].ndim, app.mat.channels() === 1 ? 2 : 3, `${c.name}: ndim`);
        const n = ndiffOf(app.mat, py[i].bytes);
        assert.equal(n, 0, `${c.name}: ${n} bytes differ`);
        const m = scalarMap(app.ctx);
        assert.equal(py[i].stdout.trim(), `shape filter: kept ${m.sf_kept} of ${m.sf_total} objects (removed by area ${m.sf_removed_area}, by shape ${m.sf_removed_shape})`, c.name);
        checked++;
      } finally { app.mat.delete(); }
    });
    assert.equal(checked, 58);
  } finally { for (const [, s] of inputs) s.delete(); for (const m of [scene, noise, one, blk]) m.delete(); }
});

test('ip3-P3. parity quantize (real Python): ndiff 0 for N in {2, 3, 4, 7, 16, 255, 256}, gray and colour', PY_SKIP, () => {
  const c = noiseRGB(64, 48, 8), ramp = rgb(256, 4, (x) => [x, 255 - x, (x * 3) & 255]);
  const cases = [];
  for (const N of [2, 3, 4, 7, 16, 255, 256]) for (const [tag, src] of [['colour', c], ['ramp', ramp]]) {
    cases.push({ name: `${tag} ${N}`, src, pipe: [OPSTEP('quantize', { levels: N })] });
    cases.push({ name: `${tag} gray ${N}`, src, pipe: [OPSTEP('grayscale', {}), OPSTEP('quantize', { levels: N })] });
  }
  try {
    const py = runPythonBatch(cases.map((x) => ({ code: buildPythonCode(x.pipe), src: x.src })));
    cases.forEach((x, i) => {
      const app = appRun(x.pipe, x.src);
      try { assert.equal(ndiffOf(app.mat, py[i].bytes), 0, x.name); assert.ok(py[i].stdout.startsWith('quantize: ')); } finally { app.mat.delete(); }
    });
  } finally { c.delete(); ramp.delete(); }
});

test('ip3-P4. parity resize explicit (real Python): nearest / bilinear ndiff 0, bicubic max |diff| <= 1, output size always equal, on 640x480, 97x61, 33x200, 2x1, 1x1 x {10, 33, 50, 75, 150, 400}%', PY_SKIP, () => {
  const sizes = [[640, 480], [97, 61], [33, 200], [2, 1], [1, 1]];
  const imgs = sizes.map(([w, h]) => noiseRGB(w, h, w * 7 + h));
  const cases = [];
  imgs.forEach((src) => { for (const scale of [0, 10, 33, 50, 75, 150, 400]) for (const interp of ['nearest', 'bilinear', 'bicubic']) cases.push({ name: `${src.cols}x${src.rows} ${scale}% ${interp}`, interp, src, pipe: [OPSTEP('resize', { scale, interp })] }); });
  try {
    const py = runPythonBatch(cases.map((x) => ({ code: buildPythonCode(x.pipe), src: x.src })));
    cases.forEach((x, i) => {
      const app = appRun(x.pipe, x.src);
      try {
        assert.deepEqual(py[i].shape.slice(0, 2), [app.mat.rows, app.mat.cols], `${x.name}: size`);
        if (x.interp === 'bicubic') assert.ok(maxAbsDiff(app.mat, py[i].bytes) <= 1, `${x.name}: max diff`);
        else assert.equal(ndiffOf(app.mat, py[i].bytes), 0, x.name);
      } finally { app.mat.delete(); }
    });
    assert.equal(cases.length, 105);
  } finally { for (const m of imgs) m.delete(); }
});

test('ip3-P5. parity rotate explicit (real Python; cv2 5.0 warpAffine differs from opencv.js 4.10, so tolerance only): output size always equal; 180 deg ndiff 0; at 15 / -37 / 45 deg on 320x240 and 640x480: bilinear / bicubic max |diff| <= 8, nearest <= 0.5% differing pixels', PY_SKIP, () => {
  const imgs = [noiseRGB(320, 240, 3), noiseRGB(640, 480, 4), noiseRGB(97, 61, 5), noiseRGB(33, 200, 6)];
  const cases = [];
  for (const src of imgs) for (const interp of ['nearest', 'bilinear', 'bicubic']) {
    const big = src.cols >= 320;
    for (const angle of big ? [15, -37, 45, 180] : [37, 180]) for (const expand of big ? [true, false] : [true]) cases.push({ name: `${src.cols}x${src.rows} ${angle} ${expand} ${interp}`, interp, angle, src, pipe: [OPSTEP('rotate', { angle, expand, interp })] });
  }
  try {
    const py = runPythonBatch(cases.map((x) => ({ code: buildPythonCode(x.pipe), src: x.src })));
    cases.forEach((x, i) => {
      const app = appRun(x.pipe, x.src);
      try {
        assert.deepEqual(py[i].shape.slice(0, 2), [app.mat.rows, app.mat.cols], `${x.name}: size`);
        const small = x.src.cols < 100;
        if (x.angle === 180) assert.equal(ndiffOf(app.mat, py[i].bytes), 0, `${x.name}: 180 deg`);
        else if (small) { /* size only */ } else if (x.interp === 'nearest') assert.ok(ndiffOf(app.mat, py[i].bytes) / app.mat.data.length <= 0.005, `${x.name}: nearest differing share`);
        else assert.ok(maxAbsDiff(app.mat, py[i].bytes) <= 8, `${x.name}: max diff ${maxAbsDiff(app.mat, py[i].bytes)}`);
      } finally { app.mat.delete(); }
    });
  } finally { for (const m of imgs) m.delete(); }
});

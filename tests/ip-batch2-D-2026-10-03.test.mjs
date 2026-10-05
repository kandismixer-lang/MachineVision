// IP batch 2 ชิ้น D (2026-10-03) — เครื่องมือวัด: feret (bwferet) · line_profile (improfile) · roi_polygon (roipoly)
// สเปก: Vision_boi/specs/ip_batch2_D_2026-10-03.md (§6 edge cases · §7 เกณฑ์ตัวเลข) · WORKLOG: WORKLOG-2026-10-03-ip-batch2-D-measurement.md
// โครงเดียวกับชิ้น C: ipD-0 (fingerprint, เทสต์แรกของไฟล์) + ipD-0w (wrapper balance เส้นทางชุดเดียวกัน) → ipD-L1..L5 → ตัวเลข → static →
// parity กับ Python จริง (cv2 5.0) — ภาพที่มีตัวอักษร (putText ของ cv2 5.0 วาดต่างจาก opencv.js 4.10) เทียบนอก "กล่องข้อความ" เท่านั้น
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { OPS, FILTERS, defaultParams, needsOf, FERET_COLUMNS } from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS, OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { GROUP_ICON, OP_PRIMARY } from '../src/renderer/filters/ui-meta.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import {
  assertLeakFree, assertWrapperBalance, fingerprintLeakProbe, makeThrowingCv,
} from './_leak-helpers.mjs';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.convexHull === 'function' && typeof c.fillPoly === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

const NEW_OPS = ['feret', 'line_profile', 'roi_polygon'];
const HOST = { feret: 'shape_analysis', line_profile: 'shape_analysis', roi_polygon: 'geometry' };
const LP_KEYS = ['line_length_px', 'samples', 'profile_min', 'profile_max', 'profile_mean', 'profile_contrast', 'edge_jump', 'edge_pos_px', 'edge_pair_px'];
const ROI_KEYS = ['roi_vertices', 'roi_area_px', 'roi_area_pct', 'roi_perimeter_px', 'roi_mean', 'roi_std', 'roi_crossed'];
const YELLOW = [255, 255, 0], GREEN = [0, 255, 0], RED = [255, 0, 0], MAGENTA = [255, 0, 255], GREY = [128, 128, 128];

// ===================================================================
// ภาพทดสอบ (deterministic ทั้งหมด — ไม่ใช้ Math.random)
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
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const v = fn(x, y); d.set(v, (y * w + x) * 3); }
  return cv.matFromArray(h, w, cv.CV_8UC3, d);
}
function toRGBA(m) {
  const out = new cv.Mat();
  cv.cvtColor(m, out, m.channels() === 1 ? cv.COLOR_GRAY2RGBA : cv.COLOR_RGB2RGBA);
  return out;
}
function toRGB(m) { const out = new cv.Mat(); cv.cvtColor(m, out, cv.COLOR_GRAY2RGB); return out; }
function toGray(m) { const out = new cv.Mat(); cv.cvtColor(m, out, m.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY); return out; }
function noiseGray(w, h, seed) { const r = lcg(seed); return gray(w, h, () => Math.floor(r() * 256)); }
function noiseRGB(w, h, seed) { const r = lcg(seed); return rgb(w, h, () => [Math.floor(r() * 256), Math.floor(r() * 256), Math.floor(r() * 256)]); }
function sceneGray(w, h, seed = 3) {
  const r = lcg(seed);
  return gray(w, h, (x, y) => {
    const base = x < w / 2 ? 60 : 190;
    const box = x > w / 4 && x < (3 * w) / 4 && y > h / 3 && y < (2 * h) / 3 ? 40 : 0;
    return Math.max(0, Math.min(255, base + box + Math.floor((r() - 0.5) * 30)));
  });
}
function sceneRGB(w, h, seed = 5) {
  const r = lcg(seed);
  return rgb(w, h, (x, y) => [
    (40 + 3 * x + Math.floor(r() * 20)) & 255,
    (90 + 2 * y + Math.floor(r() * 20)) & 255,
    (200 - x - y + Math.floor(r() * 10)) & 255,
  ]);
}
// วัตถุขาวบนพื้นดำหลายชิ้น (ภาพ 40×30): วงกลม, แท่งนอน, ก้อนแตะมุมขวาล่าง, พิกเซลเดี่ยว, แท่งตั้ง
function blobsGray(w = 40, h = 30) {
  return gray(w, h, (x, y) => ((x - 10) ** 2 + (y - 10) ** 2 < 36 || (x >= 20 && x < 31 && y >= 3 && y < 8) || (x >= w - 6 && y >= h - 6)
    || (x === 2 && y === h - 3) || (x >= 15 && x < 18 && y >= 18 && y < 28) ? 255 : 0));
}
function fillPolys(w, h, polys, channels = 1) {
  const m = cv.Mat.zeros(h, w, channels === 1 ? cv.CV_8UC1 : cv.CV_8UC3);
  const vec = new cv.MatVector();
  try {
    for (const poly of polys) {
      const pm = cv.matFromArray(poly.length, 1, cv.CV_32SC2, poly.flat());
      vec.push_back(pm);
      pm.delete();
    }
    cv.fillPoly(m, vec, channels === 1 ? new cv.Scalar(255) : new cv.Scalar(255, 255, 255));
  } finally { vec.delete(); }
  return m;
}
function bytes(m) { return Array.from(m.data); }
function run(opId, src, params = {}, cvUse = cv) {
  const ctx = {};
  const out = OPS[opId].apply(cvUse, src, params, ctx);
  return { out, ctx };
}
function runValues(opId, src, params = {}) {
  const { out, ctx } = run(opId, src, params);
  out.delete();
  return ctx;
}
function sameBytes(a, b, label) {
  assert.equal(a.rows, b.rows, `${label}: rows`);
  assert.equal(a.cols, b.cols, `${label}: cols`);
  assert.equal(a.channels(), b.channels(), `${label}: channels`);
  const da = a.data, db = b.data;
  let ndiff = 0;
  for (let i = 0; i < da.length; i++) if (da[i] !== db[i]) ndiff++;
  assert.equal(ndiff, 0, `${label}: ${ndiff} bytes differ`);
}
function px(m, x, y) { const c = m.channels(), i = (y * m.cols + x) * c; return Array.from(m.data.slice(i, i + c)); }
// % ที่ pctToPx (floor(pct/100·(n−1) + 0.5)) แปลงกลับเป็นพิกเซล p พอดี
function pctFor(p, n) { return n === 1 ? 0 : (p / (n - 1)) * 100; }
function field(ctx, key) { const f = ctx.values.fields.find((x) => x.key === key); assert.ok(f, `no field ${key}`); return f.value; }
function scalarMap(ctx) { return Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value])); }
function rowsAsObjects(values) { return values.rows.map((r) => Object.fromEntries(values.columns.map((c, i) => [c, r[i]]))); }
function lpParams(ax, ay, bx, by, w, h, extra = {}) {
  return { x1: pctFor(ax, w), y1: pctFor(ay, h), x2: pctFor(bx, w), y2: pctFor(by, h), ...extra };
}
function roiParams(points, w, h, extra = {}) {
  const p = { points: points.length, ...extra };
  points.forEach(([x, y], i) => { p[`p${i + 1}x`] = pctFor(x, w); p[`p${i + 1}y`] = pctFor(y, h); });
  return p;
}

// ===================================================================
// ipD-0 / ipD-0w — ด่าน leak หลัก (P-14 ข้อ 1 + คู่บังคับข้อ 2) — ทุกแถวของสเปก §6, ทุก view, n 3..6, cap ทำงาน
// ===================================================================
function batchDLeakPaths() {
  const blobs = blobsGray(), blobsC = toRGB(blobs), blobsA = toRGBA(blobs);
  const scene = sceneGray(80, 80), sceneC = sceneRGB(80, 80), sceneA = toRGBA(sceneC), small = sceneGray(30, 20);
  const oneW = gray(1, 1, () => 255), oneB = gray(1, 1, () => 0), black = gray(30, 20, () => 0), empty = new cv.Mat();
  const paths = [];
  const add = (id, params, src, tag) => paths.push({
    label: `${id} ${tag}`,
    run: (cvUse) => { const o = OPS[id].apply(cvUse, src, params, {}); o.delete(); },
  });
  add('feret', {}, blobs, 'gray');
  add('feret', { minArea: 0 }, blobs, 'gray minArea 0 (1-px object + border object)');
  add('feret', { minArea: 0 }, blobsC, 'colour (direct call)');
  add('feret', { minArea: 0 }, blobsA, 'rgba (direct call)');
  add('feret', { minArea: 0, maxObjects: 2 }, blobs, 'cap active');
  add('feret', { minArea: 0 }, oneW, '1x1 white');
  add('feret', {}, oneB, '1x1 black (none)');
  add('feret', {}, black, 'all black (none)');
  add('feret', {}, empty, '0x0');
  add('line_profile', {}, scene, 'overlay');
  add('line_profile', { view: 'plot' }, scene, 'plot');
  add('line_profile', {}, sceneC, 'colour overlay');
  add('line_profile', { view: 'plot' }, sceneA, 'rgba plot');
  add('line_profile', { y1: 95, y2: 95 }, scene, 'band at top');
  add('line_profile', { x1: 50, y1: 50, x2: 50, y2: 50 }, scene, 'A = B with graph');
  add('line_profile', { view: 'plot', x1: 50, y1: 50, x2: 50, y2: 50 }, scene, 'A = B plot');
  add('line_profile', {}, small, 'small overlay');
  add('line_profile', { view: 'plot' }, small, 'small plot');
  add('line_profile', {}, oneW, '1x1');
  add('line_profile', {}, empty, '0x0');
  for (const view of ['apply', 'mask', 'overlay']) {
    add('roi_polygon', { view }, scene, `${view} gray`);
    add('roi_polygon', { view }, sceneC, `${view} colour`);
    add('roi_polygon', { view }, sceneA, `${view} rgba`);
  }
  for (const n of [3, 5, 6]) add('roi_polygon', { points: n, view: 'overlay' }, sceneC, `n ${n}`);
  add('roi_polygon', { p1x: 10, p1y: 10, p2x: 90, p2y: 90, p3x: 90, p3y: 10, p4x: 10, p4y: 90 }, scene, 'bowtie');
  add('roi_polygon', { points: 3, p1x: 50, p1y: 50, p2x: 50, p2y: 50, p3x: 50, p3y: 50, view: 'overlay' }, scene, 'stacked');
  add('roi_polygon', {}, oneW, '1x1');
  add('roi_polygon', { view: 'mask' }, empty, '0x0');
  return {
    paths,
    cleanup: () => { for (const m of [blobs, blobsC, blobsA, scene, sceneC, sceneA, small, oneW, oneB, black, empty]) m.delete(); },
  };
}

// the fingerprint is checked per op (one probe per op, same path table filtered by op id): the combined probe was sensitive to the
// allocator layout of the whole directory it ran from (not a leak: ipD-0w + the census of L1-L3 stay green) -- reviewer MINOR-3
for (const [suffix, opId, what] of [
  ['', 'feret', 'feret gray / colour / RGBA / cap / 1-px object / border object / none / 0x0'],
  ['b', 'line_profile', 'line_profile overlay / plot / band at top / A = B / small image / colour / RGBA / 1x1 / 0x0'],
  ['c', 'roi_polygon', 'roi_polygon apply / mask / overlay on gray / colour / RGBA, n 3..6, bowtie, stacked points, 1x1, 0x0'],
]) {
  test(`ipD-0${suffix}. leak (address fingerprint, per op, ${suffix ? 'after ipD-0' : 'first test of a clean process'}): every branch of ${what} leaves the malloc fingerprint unchanged`, () => {
    const { paths, cleanup } = batchDLeakPaths();
    try {
      const mine = paths.filter((q) => q.label.startsWith(`${opId} `));
      assert.ok(mine.length >= 4, `${opId}: paths ${mine.length}`);
      fingerprintLeakProbe(cv, `IP batch 2 chunk D ${opId} (all paths)`, () => {
        for (const q of mine) q.run(cv);
      }, { iters: 25, warmup: 10 });
    } finally { cleanup(); }
  });
}

test('ipD-0w. leak (wrapper balance twin of ipD-0): the exact same path table creates and deletes every embind wrapper it makes (per-contour Mats, the corner/hull Mats of every object, the graph band ROI view, the polyline Mat + MatVector, the ROI mask / polygon / RGB temporaries, the passthrough clones)', () => {
  const { paths, cleanup } = batchDLeakPaths();
  try {
    assertWrapperBalance(cv, 'IP batch 2 chunk D ops (all paths)', (c) => {
      for (const p of paths) p.run(c);
    }, { minCreated: 150 });
  } finally { cleanup(); }
});

// ===================================================================
// ipD-L1…L5 — leak ต่อ op / ต่อเส้นทาง
// ===================================================================
function leakEach(label, cases, opts = {}) {
  for (const [tag, fn] of cases) {
    assertLeakFree(cv, `${label} ${tag}`, (c) => { const o = fn(c); o.delete(); }, { minCreated: 1, iters: 40, warmup: 10, ...opts });
  }
}

test('ipD-L1. leak: feret on gray / colour / RGBA, minArea filter, cap active, a 1-px object, an object touching the border, nothing found and the 0x0 passthrough balances every wrapper and grows the heap by 0', () => {
  const b = blobsGray(), c = toRGB(b), a = toRGBA(b), black = gray(20, 10, () => 0), empty = new cv.Mat();
  try {
    leakEach('feret', [
      ['gray', (cvU) => OPS.feret.apply(cvU, b, {}, {})],
      ['gray minArea 0', (cvU) => OPS.feret.apply(cvU, b, { minArea: 0 }, {})],
      ['colour', (cvU) => OPS.feret.apply(cvU, c, { minArea: 0 }, {})],
      ['rgba', (cvU) => OPS.feret.apply(cvU, a, { minArea: 0 }, {})],
      ['cap 1', (cvU) => OPS.feret.apply(cvU, b, { minArea: 0, maxObjects: 1 }, {})],
      ['none', (cvU) => OPS.feret.apply(cvU, black, {}, {})],
      ['0x0', (cvU) => OPS.feret.apply(cvU, empty, {}, {})],
    ]);
  } finally { for (const m of [b, c, a, black, empty]) m.delete(); }
});

test('ipD-L6. leak: feret when the sort has to break a tie with the centroid (two zero-area line objects with the same area and the same bounding-box corner -> compareRegionCandidates calls centroidOf, which re-reads the contour) balances every wrapper and grows the heap by 0', () => {
  // B = 3 px on row 10 (x 10..12); A = an L-shaped 1-px line (14,10)-(14,12)-(13,13)-(10,13): both area 0, bbox corner (10,10), 8-connected components apart
  const m = gray(30, 20, (x, y) => ((y === 10 && x >= 10 && x <= 12) || (x === 14 && y >= 10 && y <= 12) || (y === 13 && x >= 10 && x <= 13) ? 255 : 0));
  try {
    const r = run('feret', m, { minArea: 0 });
    try { assert.equal(r.ctx.values.total, 2, 'two objects (a tie on area / y / x)'); } finally { r.out.delete(); }
    leakEach('feret centroid tie-break', [['two lines', (cvU) => OPS.feret.apply(cvU, m, { minArea: 0 }, {})]]);
  } finally { m.delete(); }
});

test('ipD-L2. leak: line_profile overlay / plot / band at top / A = B / small image (graph skipped) / colour / RGBA / 1x1 and the 0x0 passthrough balances every wrapper (gray temp, RGB copy, band ROI view, polyline Mat + MatVector) and grows the heap by 0', () => {
  const g = sceneGray(80, 70), c = sceneRGB(80, 70), a = toRGBA(c), s = sceneGray(30, 20), one = gray(1, 1, () => 9), empty = new cv.Mat();
  try {
    leakEach('line_profile', [
      ['overlay', (cvU) => OPS.line_profile.apply(cvU, g, {}, {})],
      ['plot', (cvU) => OPS.line_profile.apply(cvU, g, { view: 'plot' }, {})],
      ['band top', (cvU) => OPS.line_profile.apply(cvU, c, { y1: 99, y2: 90 }, {})],
      ['A = B', (cvU) => OPS.line_profile.apply(cvU, a, { x1: 5, y1: 5, x2: 5, y2: 5 }, {})],
      ['A = B plot', (cvU) => OPS.line_profile.apply(cvU, g, { view: 'plot', x1: 5, y1: 5, x2: 5, y2: 5 }, {})],
      ['small', (cvU) => OPS.line_profile.apply(cvU, s, { view: 'plot' }, {})],
      ['rgba', (cvU) => OPS.line_profile.apply(cvU, a, {}, {})],
      ['1x1', (cvU) => OPS.line_profile.apply(cvU, one, {}, {})],
      ['0x0', (cvU) => OPS.line_profile.apply(cvU, empty, {}, {})],
    ]);
  } finally { for (const m of [g, c, a, s, one, empty]) m.delete(); }
});

test('ipD-L3. leak: roi_polygon apply / mask (the mask itself becomes the output) / overlay on gray / colour / RGBA, n 3 and 6, bowtie, stacked points, 1x1 and the 0x0 passthrough balances every wrapper (mask, polygon Mat, MatVector, gray temp, RGB temp, overlay base) and grows the heap by 0', () => {
  const g = sceneGray(50, 40), c = sceneRGB(50, 40), a = toRGBA(c), one = gray(1, 1, () => 9), empty = new cv.Mat();
  try {
    const cases = [];
    for (const view of ['apply', 'mask', 'overlay']) {
      cases.push([`${view} gray`, (cvU) => OPS.roi_polygon.apply(cvU, g, { view }, {})]);
      cases.push([`${view} colour`, (cvU) => OPS.roi_polygon.apply(cvU, c, { view, points: 6 }, {})]);
      cases.push([`${view} rgba`, (cvU) => OPS.roi_polygon.apply(cvU, a, { view, points: 3 }, {})]);
    }
    cases.push(['bowtie', (cvU) => OPS.roi_polygon.apply(cvU, g, { p1x: 10, p1y: 10, p2x: 90, p2y: 90, p3x: 90, p3y: 10, p4x: 10, p4y: 90 }, {})]);
    cases.push(['stacked', (cvU) => OPS.roi_polygon.apply(cvU, c, { points: 3, p1x: 1, p1y: 1, p2x: 1, p2y: 1, p3x: 1, p3y: 1, view: 'overlay' }, {})]);
    cases.push(['1x1', (cvU) => OPS.roi_polygon.apply(cvU, one, {}, {})]);
    cases.push(['0x0', (cvU) => OPS.roi_polygon.apply(cvU, empty, {}, {})]);
    leakEach('roi_polygon', cases);
  } finally { for (const m of [g, c, a, one, empty]) m.delete(); }
});

// เปลี่ยน method ของ Mat (copyTo/roi) ให้ throw ชั่วคราว — makeThrowingCv ครอบได้เฉพาะฟังก์ชันบน cv
function withThrowingMatMethod(name, fn) {
  const proto = cv.Mat.prototype;
  const orig = proto[name];
  proto[name] = function () { throw new Error(`injected test failure (Mat.${name})`); };
  try { return fn(); } finally { proto[name] = orig; }
}

test('ipD-L4. leak on the exception path: when a cv call inside each op throws (convexHull / matFromArray (corners, polyline, polygon) / findContours / putText / polylines / fillPoly / convertScaleAbs / cvtColor / line / circle / Mat.copyTo / Mat.roi), every wrapper allocated so far is still released and the exception propagates', () => {
  const b = blobsGray(), c = sceneRGB(80, 70), a = toRGBA(c), g = sceneGray(80, 70);
  const throwing = [
    ['feret convexHull', 'convexHull', (cvU) => OPS.feret.apply(cvU, b, { minArea: 0 }, {})],
    ['feret 2nd convexHull', ['convexHull', 2], (cvU) => OPS.feret.apply(cvU, b, { minArea: 0 }, {})],
    ['feret matFromArray', 'matFromArray', (cvU) => OPS.feret.apply(cvU, b, {}, {})],
    ['feret findContours', 'findContours', (cvU) => OPS.feret.apply(cvU, b, {}, {})],
    ['feret putText', ['putText', 2], (cvU) => OPS.feret.apply(cvU, b, { minArea: 0 }, {})],
    ['feret line', 'line', (cvU) => OPS.feret.apply(cvU, b, {}, {})],
    ['feret cvtColor (RGBA direct call)', 'cvtColor', (cvU) => OPS.feret.apply(cvU, a, {}, {})],
    ['feret GRAY2RGB', ['cvtColor', 2], (cvU) => OPS.feret.apply(cvU, a, {}, {})],
    ['line_profile cvtColor', 'cvtColor', (cvU) => OPS.line_profile.apply(cvU, a, {}, {})],
    ['line_profile 2nd cvtColor (RGBA->RGB)', ['cvtColor', 2], (cvU) => OPS.line_profile.apply(cvU, a, {}, {})],
    ['line_profile convertScaleAbs', 'convertScaleAbs', (cvU) => OPS.line_profile.apply(cvU, c, {}, {})],
    ['line_profile polylines', 'polylines', (cvU) => OPS.line_profile.apply(cvU, g, { view: 'plot' }, {})],
    ['line_profile matFromArray', 'matFromArray', (cvU) => OPS.line_profile.apply(cvU, c, {}, {})],
    ['line_profile line', 'line', (cvU) => OPS.line_profile.apply(cvU, c, {}, {})],
    ['line_profile magenta line', ['line', 4], (cvU) => OPS.line_profile.apply(cvU, c, {}, {})],
    ['line_profile circle', 'circle', (cvU) => OPS.line_profile.apply(cvU, g, {}, {})],
    ['roi fillPoly', 'fillPoly', (cvU) => OPS.roi_polygon.apply(cvU, c, {}, {})],
    ['roi matFromArray', 'matFromArray', (cvU) => OPS.roi_polygon.apply(cvU, c, {}, {})],
    ['roi cvtColor (gray)', 'cvtColor', (cvU) => OPS.roi_polygon.apply(cvU, a, { view: 'overlay' }, {})],
    ['roi dropAlpha cvtColor', ['cvtColor', 2], (cvU) => OPS.roi_polygon.apply(cvU, a, {}, {})],
    ['roi overlay RGBA->RGB', ['cvtColor', 2], (cvU) => OPS.roi_polygon.apply(cvU, a, { view: 'overlay' }, {})],
    ['roi convertScaleAbs', 'convertScaleAbs', (cvU) => OPS.roi_polygon.apply(cvU, c, { view: 'overlay' }, {})],
    ['roi polylines', 'polylines', (cvU) => OPS.roi_polygon.apply(cvU, c, { view: 'overlay' }, {})],
    ['roi putText', ['putText', 3], (cvU) => OPS.roi_polygon.apply(cvU, g, { view: 'overlay' }, {})],
    ['roi circle', 'circle', (cvU) => OPS.roi_polygon.apply(cvU, g, { view: 'overlay' }, {})],
  ];
  const matThrowing = [
    ['roi apply copyTo', 'copyTo', (cvU) => OPS.roi_polygon.apply(cvU, c, {}, {})],
    ['roi overlay copyTo', 'copyTo', (cvU) => OPS.roi_polygon.apply(cvU, g, { view: 'overlay' }, {})],
    ['line_profile band roi', 'roi', (cvU) => OPS.line_profile.apply(cvU, c, {}, {})],
    ['line_profile copyTo (motionRGB of RGB)', 'copyTo', (cvU) => OPS.line_profile.apply(cvU, c, {}, {})],
  ];
  try {
    for (const [label, method, fn] of throwing) {
      const [name, onCall] = Array.isArray(method) ? method : [method, null];
      assertWrapperBalance(cv, `throw in ${label}`, (cvU) => {
        const o = fn(makeThrowingCv(cvU, name, onCall ? { onCall } : {}));
        o.delete();
      }, { expectThrow: /injected test failure/, minCreated: 1 });
    }
    for (const [label, name, fn] of matThrowing) {
      assertWrapperBalance(cv, `throw in ${label}`, (cvU) => withThrowingMatMethod(name, () => { const o = fn(cvU); o.delete(); }),
        { expectThrow: /injected test failure \(Mat\./, minCreated: 1 });
    }
  } finally { for (const m of [b, c, a, g]) m.delete(); }
});

test('ipD-L5. leak on the exception path AFTER the output exists: a ctx whose "values" setter (the last statement of every successful path) or "info" setter throws makes each op delete its own output -- feret found / none, line_profile overlay / plot / A = B, roi_polygon apply / mask / overlay', () => {
  const b = blobsGray(), black = gray(20, 10, () => 0), g = sceneGray(80, 70), c = sceneRGB(80, 70);
  const throwingValues = () => ({ info: '', set values(_) { throw new Error('injected test failure (ctx.values)'); } });
  const throwingInfo = () => ({ set info(_) { throw new Error('injected test failure (ctx.info)'); } });
  const cases = [
    ['feret found', (cvU, ctx) => OPS.feret.apply(cvU, b, { minArea: 0 }, ctx)],
    ['feret none', (cvU, ctx) => OPS.feret.apply(cvU, black, {}, ctx)],
    ['line_profile overlay', (cvU, ctx) => OPS.line_profile.apply(cvU, c, {}, ctx)],
    ['line_profile plot', (cvU, ctx) => OPS.line_profile.apply(cvU, g, { view: 'plot' }, ctx)],
    ['line_profile A = B', (cvU, ctx) => OPS.line_profile.apply(cvU, g, { x1: 3, x2: 3, y1: 3, y2: 3 }, ctx)],
    ['roi apply', (cvU, ctx) => OPS.roi_polygon.apply(cvU, c, {}, ctx)],
    ['roi mask', (cvU, ctx) => OPS.roi_polygon.apply(cvU, g, { view: 'mask' }, ctx)],
    ['roi overlay', (cvU, ctx) => OPS.roi_polygon.apply(cvU, c, { view: 'overlay' }, ctx)],
  ];
  try {
    for (const [label, fn] of cases) {
      assertWrapperBalance(cv, `ctx.values throws in ${label}`, (cvU) => { const o = fn(cvU, throwingValues()); o.delete(); },
        { expectThrow: /injected test failure \(ctx\.values\)/, minCreated: 1 });
      assertWrapperBalance(cv, `ctx.info throws in ${label}`, (cvU) => { const o = fn(cvU, throwingInfo()); o.delete(); },
        { expectThrow: /injected test failure \(ctx\.info\)/, minCreated: 1 });
    }
  } finally { for (const m of [b, black, g, c]) m.delete(); }
});

// ===================================================================
// ipD-1…6 — feret
// ===================================================================
const close = (a, b, tol, label) => assert.ok(Math.abs(a - b) <= tol, `${label}: ${a} vs ${b} (tol ${tol})`);

test('ipD-1. feret reference shapes (spec 7): a single pixel -> (max sqrt2, min 1, len 1, aspect 1, maxDeg 45, minDeg 0); a 10x3 horizontal bar -> max sqrt(109), min 3, len 10, minDeg 90, maxDeg atan2(3, 10); a 10x10 square -> max 10*sqrt2 at 45 deg (45/135 tie -> 45), min 10 at 0 deg (0/90 tie -> 0); a ring has the same numbers as the disc of the same outer size; a 10x1 vertical line -> max sqrt(101), min 1 at 0 deg, len 10', () => {
  const cases = [
    ['1 px', gray(9, 9, (x, y) => (x === 4 && y === 4 ? 255 : 0)), { d2: 8, max: Math.SQRT2, min: 1, len: 1, aspect: 1, maxDeg: 45, minDeg: 0 }],
    ['10x3 bar', gray(40, 20, (x, y) => (x >= 10 && x < 20 && y >= 5 && y < 8 ? 255 : 0)),
      { d2: 436, max: Math.sqrt(109), min: 3, len: 10, aspect: 10 / 3, maxDeg: (Math.atan2(3, 10) * 180) / Math.PI, minDeg: 90 }],
    ['10x10 square', gray(30, 30, (x, y) => (x >= 5 && x < 15 && y >= 8 && y < 18 ? 255 : 0)), { d2: 800, max: Math.sqrt(200), min: 10, len: 10, aspect: 1, maxDeg: 45, minDeg: 0 }],
    ['1x10 vertical line', gray(20, 30, (x, y) => (x === 7 && y >= 3 && y < 13 ? 255 : 0)), { d2: 404, max: Math.sqrt(101), min: 1, len: 10, aspect: 10, maxDeg: (Math.atan2(10, 1) * 180) / Math.PI, minDeg: 0 }],
  ];
  try {
    for (const [label, img, e] of cases) {
      const ctx = runValues('feret', img, { minArea: 0 });
      assert.equal(ctx.values.total, 1, `${label}: one object`);
      const r = rowsAsObjects(ctx.values)[0];
      assert.equal(r.feret_max, Math.sqrt(e.d2) / 2, `${label}: max (exact, doubled-corner d2 = ${e.d2})`);
      close(r.feret_max, e.max, 1e-12, `${label} max`);
      assert.equal(r.feret_min, e.min, `${label}: min`);
      assert.equal(r.feret_len, e.len, `${label}: len`);
      close(r.feret_aspect, e.aspect, 1e-12, `${label} aspect`);
      close(r.feret_max_deg, e.maxDeg, 1e-9, `${label} maxDeg`);
      assert.equal(r.feret_min_deg, e.minDeg, `${label}: minDeg`);
      assert.ok(!Object.is(r.feret_min_deg, -0) && !Object.is(r.feret_max_deg, -0), `${label}: no -0 angles`);
    }
  } finally { for (const [, m] of cases) m.delete(); }
  const disc = gray(50, 50, (x, y) => ((x - 24) ** 2 + (y - 24) ** 2 <= 225 ? 255 : 0));
  const ring = gray(50, 50, (x, y) => { const d = (x - 24) ** 2 + (y - 24) ** 2; return d <= 225 && d > 64 ? 255 : 0; });
  try {
    const a = rowsAsObjects(runValues('feret', disc, { minArea: 0 }).values)[0];
    const b = rowsAsObjects(runValues('feret', ring, { minArea: 0 }).values)[0];
    for (const k of ['feret_max', 'feret_min', 'feret_len', 'feret_aspect', 'feret_max_deg', 'feret_min_deg']) assert.equal(b[k], a[k], `ring = disc: ${k}`);
    assert.ok(a.feret_min > 30 && a.feret_min <= 31, `disc of radius 15 is ~31 px wide on pixel edges (${a.feret_min})`);
  } finally { disc.delete(); ring.delete(); }
});

// พิกเซลขาวทั้งหมด → มุมพิกเซล ×2 (ไม่ใช้ hull) — ตัวเทียบอิสระ
function whiteCorners(m, boundaryOnly) {
  const out = [];
  const d = m.data, w = m.cols, h = m.rows;
  const on = (x, y) => x >= 0 && y >= 0 && x < w && y < h && d[y * w + x] !== 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!on(x, y)) continue;
    if (boundaryOnly && on(x - 1, y) && on(x + 1, y) && on(x, y - 1) && on(x, y + 1)) continue;
    out.push([2 * x - 1, 2 * y - 1], [2 * x + 1, 2 * y - 1], [2 * x + 1, 2 * y + 1], [2 * x - 1, 2 * y + 1]);
  }
  return out;
}

test('ipD-2. feret vs an independent brute force on >= 20 random star-shaped blobs: Max Feret is EXACTLY the largest distance between any two pixel corners (all corners, no hull), Min Feret lies in [sampled_min - 0.01, sampled_min + 1e-9] where sampled_min = smallest caliper width over a 0..180 deg sweep in 0.005 deg steps; feret_len lies between Min and Max, aspect = len/min', () => {
  const r = lcg(77);
  let checked = 0;
  for (let t = 0; t < 30 && checked < 22; t++) {
    const cx = 20 + Math.floor(r() * 4), cy = 20 + Math.floor(r() * 4), k = 5 + Math.floor(r() * 4), rot = r() * Math.PI;
    const poly = [];
    for (let i = 0; i < k; i++) {
      const ang = rot + (2 * Math.PI * i) / k, rad = 4 + r() * 12;
      poly.push([Math.round(cx + rad * Math.cos(ang)), Math.round(cy + rad * Math.sin(ang))]);
    }
    const img = fillPolys(44, 44, [poly]);
    try {
      const ctx = runValues('feret', img, { minArea: 0 });
      if (ctx.values.total !== 1) continue; // fillPoly อาจได้ 2 ชิ้นที่แตะกันแค่มุม — ใช้เฉพาะรูปที่เป็นชิ้นเดียว
      const row = rowsAsObjects(ctx.values)[0];
      const all = whiteCorners(img, false);
      let best = 0;
      for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) {
        const dx = all[j][0] - all[i][0], dy = all[j][1] - all[i][1];
        const d2 = dx * dx + dy * dy;
        if (d2 > best) best = d2;
      }
      assert.equal(row.feret_max, Math.sqrt(best) / 2, `blob ${t}: max exactly`);
      const bc = whiteCorners(img, true);
      let sampled = Infinity;
      for (let s = 0; s <= 36000; s++) {
        const th = (s * 0.005 * Math.PI) / 180, ux = Math.cos(th), uy = Math.sin(th);
        let lo = Infinity, hi = -Infinity;
        for (const [x, y] of bc) { const v = ux * x + uy * y; if (v < lo) lo = v; if (v > hi) hi = v; }
        if ((hi - lo) / 2 < sampled) sampled = (hi - lo) / 2;
      }
      assert.ok(row.feret_min >= sampled - 0.01 && row.feret_min <= sampled + 1e-9, `blob ${t}: min ${row.feret_min} vs sampled ${sampled}`);
      assert.ok(row.feret_len >= row.feret_min - 1e-9 && row.feret_len <= row.feret_max + 1e-9, `blob ${t}: min <= len <= max`);
      assert.equal(row.feret_aspect, row.feret_len / row.feret_min);
      assert.ok(row.feret_aspect >= 1 - 1e-12);
      checked++;
    } finally { img.delete(); }
  }
  assert.ok(checked >= 20, `only ${checked} single-object blobs checked`);
});

test('ipD-3. feret on bars rotated 30 and 60 deg (fillPoly of a 60x12 rectangle): Min Feret within the true width +- 1.5 (true width of the rasterised bar on pixel edges = polygon width 12 + the unit pixel square seen across the bar, |cos| + |sin|), Max Feret in [diagonal sqrt(60^2+12^2), +3], len within 60 + |cos| + |sin| +- 1.5, the min angle within 4 deg of the bar normal (120 / 150 deg) and the max angle near the bar direction; an upright bar gives a Bounding-Box-sized result', () => {
  for (const deg of [30, 60]) {
    const th = (deg * Math.PI) / 180, c = 50, L = 30, T = 6;
    const ux = Math.cos(th), uy = Math.sin(th), nx = -uy, ny = ux;
    const poly = [[-L, -T], [L, -T], [L, T], [-L, T]].map(([a, b]) => [Math.round(c + a * ux + b * nx), Math.round(c + a * uy + b * ny)]);
    const img = fillPolys(100, 100, [poly]);
    try {
      const row = rowsAsObjects(runValues('feret', img, { minArea: 0 }).values)[0];
      close(row.feret_min, 12 + Math.abs(Math.cos(th)) + Math.abs(Math.sin(th)), 1.5, `${deg} deg min`);
      assert.ok(row.feret_max >= Math.hypot(60, 12) && row.feret_max <= Math.hypot(60, 12) + 3, `${deg} deg max ${row.feret_max}`);
      close(row.feret_min_deg, deg + 90, 4, `${deg} deg min angle`);
      close(row.feret_len, 60 + Math.abs(Math.cos(th)) + Math.abs(Math.sin(th)), 1.5, `${deg} deg len`);
      close(row.feret_max_deg, deg, 15, `${deg} deg max angle (diagonal of the bar)`);
    } finally { img.delete(); }
  }
});

test('ipD-4. feret numbers objects exactly like Region Properties: with the same minArea (0 and 50) and both caps above the object count, index / area / cx / cy are equal row by row (incl. 1-px objects whose centroid falls back to the bounding-box centre, equal-area ties broken by y then x, an object touching the border)', () => {
  const img = gray(120, 90, (x, y) => {
    if ((x >= 10 && x < 20 && y >= 10 && y < 20) || (x >= 40 && x < 50 && y >= 10 && y < 20)) return 255; // ขนาดเท่ากัน y เท่ากัน → เรียงตาม x
    if ((x >= 70 && x < 80 && y >= 5 && y < 15)) return 255; // ขนาดเท่ากัน y น้อยกว่า → มาก่อน
    if ((x - 30) ** 2 + (y - 60) ** 2 < 150) return 255;
    if (x === 100 && y === 50) return 255; // 1 px
    if (x === 5 && y === 80) return 255;
    if (x >= 110 && y >= 80) return 255; // แตะมุมขวาล่าง
    if (x >= 60 && x < 95 && y === 70) return 255; // เส้นบาง (m00 = 0)
    return 0;
  });
  try {
    for (const minArea of [0, 50]) {
      const rp = rowsAsObjects(runValues('regionprops', img, { minArea, maxObjects: 0 }).values);
      const fv = runValues('feret', img, { minArea, maxObjects: 500 }).values;
      const fr = rowsAsObjects(fv);
      assert.equal(fr.length, rp.length, `minArea ${minArea}: same object count`);
      assert.equal(fv.total, rp.length);
      assert.ok(rp.length >= (minArea === 0 ? 8 : 3), `minArea ${minArea}: enough objects (${rp.length})`);
      for (let i = 0; i < rp.length; i++) for (const k of ['index', 'area', 'cx', 'cy']) assert.equal(fr[i][k], rp[i][k], `minArea ${minArea} row ${i + 1}: ${k}`);
    }
  } finally { img.delete(); }
});

test('ipD-5. feret cap and messages: 7 objects with maxObjects 3 -> 3 rows (the 3 largest, same order as Region Properties), total 7, info "พบวัตถุ 7 ชิ้น วัด 3 ชิ้นแรกเรียงจากใหญ่ไปเล็ก"; without the cap info "วัด Feret ของวัตถุ 7 ชิ้น · ชิ้น #1 ยาวสุด X.X px · แคบสุด Y.Y px"; nothing found -> rows [] + the hint; minArea is "area < minArea" (an object of exactly minArea is kept)', () => {
  const img = gray(140, 30, (x, y) => { for (let k = 0; k < 7; k++) { const x0 = 5 + k * 19, s = 3 + k * 2; if (x >= x0 && x < x0 + Math.min(s, 15) && y >= 5 && y < 5 + s) return 255; } return 0; });
  const black = gray(30, 20, () => 0);
  try {
    const capped = runValues('feret', img, { minArea: 0, maxObjects: 3 });
    assert.equal(capped.values.rows.length, 3);
    assert.equal(capped.values.total, 7);
    assert.match(capped.info, /^พบวัตถุ 7 ชิ้น วัด 3 ชิ้นแรกเรียงจากใหญ่ไปเล็ก \(ปรับ 'วัดวัตถุสูงสุด' ได้\) · ชิ้น #1 ยาวสุด [\d.]+ px · แคบสุด [\d.]+ px/);
    const rp = rowsAsObjects(runValues('regionprops', img, { minArea: 0, maxObjects: 3 }).values);
    assert.deepEqual(rowsAsObjects(capped.values).map((r) => r.area), rp.map((r) => r.area));
    const all = runValues('feret', img, { minArea: 0 });
    assert.equal(all.values.rows.length, 7);
    const r1 = rowsAsObjects(all.values)[0];
    const f1 = (v) => (Math.floor(v * 10 + 0.5) / 10).toFixed(1);
    assert.equal(all.info, `วัด Feret ของวัตถุ 7 ชิ้น · ชิ้น #1 ยาวสุด ${f1(r1.feret_max)} px · แคบสุด ${f1(r1.feret_min)} px (กรอบเขียว = กรอบที่แคบสุด, เส้นแดง = แนวที่ยาวสุด)`);
    const areas = rowsAsObjects(all.values).map((r) => r.area).sort((a, b) => a - b);
    const keepEq = runValues('feret', img, { minArea: areas[2] });
    assert.equal(keepEq.values.total, 5, 'area == minArea is kept (area < minArea drops)');
    const none = runValues('feret', black, {});
    assert.deepEqual(none.values, { kind: 'table', columns: FERET_COLUMNS, rows: [], total: 0 });
    assert.equal(none.info, 'ไม่พบวัตถุ (ลองลดค่าพื้นที่ขั้นต่ำ หรือทำ Threshold ให้เป็นขาว-ดำก่อน)');
  } finally { img.delete(); black.delete(); }
});

test('ipD-6. feret drawing (spec 4.1 step 4) on a 10x3 bar at (20..29, 20..22): the green narrowest box runs through pixels (20,20)-(30,20)-(30,23)-(20,23), the red Max Feret line goes from (20,20) to (30,23) (11 px, drawn after the box), output = GRAY2RGB of the input elsewhere; the input Mat is not modified by any of the 3 ops; RGB / RGBA direct calls equal the gray call', () => {
  const img = gray(50, 40, (x, y) => (x >= 20 && x < 30 && y >= 20 && y < 23 ? 200 : 0));
  const c = toRGB(img), a = toRGBA(img);
  try {
    const before = bytes(img);
    const { out, ctx } = run('feret', img, { minArea: 0 });
    try {
      assert.deepEqual(bytes(img), before, 'src unchanged');
      assert.equal(out.channels(), 3);
      assert.deepEqual(px(out, 20, 20), RED);
      assert.deepEqual(px(out, 30, 23), RED);
      for (const [x, y] of [[25, 20], [30, 21], [30, 22], [25, 23], [20, 22], [22, 23], [27, 20]]) assert.deepEqual(px(out, x, y), GREEN, `green at ${x},${y}`);
      let reds = 0;
      for (let y = 16; y < 40; y++) for (let x = 0; x < 50; x++) { const p = px(out, x, y); if (p[0] === 255 && p[1] === 0 && p[2] === 0) reds++; }
      assert.equal(reds, 11, 'one red 1-px line of 11 pixels');
      assert.ok([RED, [200, 200, 200]].some((c) => JSON.stringify(c) === JSON.stringify(px(out, 24, 21))), 'inside the bar: red line or GRAY2RGB(200)');
      assert.deepEqual(px(out, 45, 35), [0, 0, 0]);
      assert.equal(ctx.values.rows.length, 1);
      for (const src of [c, a]) {
        const o2 = OPS.feret.apply(cv, src, { minArea: 0 }, {});
        try { sameBytes(o2, out, `feret ${src.channels()}-channel direct call`); } finally { o2.delete(); }
      }
    } finally { out.delete(); }
    for (const [id, params, src] of [['line_profile', {}, c], ['line_profile', { view: 'plot' }, a], ['roi_polygon', { view: 'overlay' }, c], ['roi_polygon', {}, a], ['roi_polygon', { view: 'mask' }, img]]) {
      const b0 = bytes(src);
      const o = OPS[id].apply(cv, src, params, {});
      o.delete();
      assert.deepEqual(bytes(src), b0, `${id} ${JSON.stringify(params)}: src unchanged`);
    }
  } finally { img.delete(); c.delete(); a.delete(); }
});

// ===================================================================
// ipD-7…12 — line_profile
// ===================================================================
// ตัวเทียบ: จุดที่อ่านตามสูตรสเปก §4.2 ข้อ 4 + ค่าทั้ง 9 จากโปรไฟล์ (เขียนแยกจาก registry — argmax ตัวแรก, คู่ขอบขึ้น/ลง)
function refSamples(ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, N = Math.max(Math.abs(dx), Math.abs(dy)) + 1;
  const pts = [];
  for (let k = 0; k < N; k++) pts.push(N === 1 ? [ax, ay] : [Math.floor(ax + (dx * k) / (N - 1) + 0.5), Math.floor(ay + (dy * k) / (N - 1) + 0.5)]);
  return pts;
}
function grayData(m) {
  if (m.channels() === 1) return m.data;
  const g = toGray(m);
  try { return Uint8Array.from(g.data); } finally { g.delete(); }
}
function refProfile(m, ax, ay, bx, by) {
  const d = grayData(m);
  return refSamples(ax, ay, bx, by).map(([x, y]) => d[y * m.cols + x]);
}
function refValues(prof, ax, ay, bx, by) {
  const N = prof.length, L = Math.sqrt((bx - ax) ** 2 + (by - ay) ** 2);
  const v = {
    line_length_px: L, samples: N, profile_min: Math.min(...prof), profile_max: Math.max(...prof),
    profile_mean: prof.reduce((a, b) => a + b, 0) / N, profile_contrast: Math.max(...prof) - Math.min(...prof),
    edge_jump: null, edge_pos_px: null, edge_pair_px: null,
  };
  if (N >= 2) {
    const step = L / (N - 1), diff = prof.slice(1).map((x, i) => x - prof[i]);
    const absd = diff.map(Math.abs), iStar = absd.indexOf(Math.max(...absd));
    v.edge_jump = diff[iStar];
    v.edge_pos_px = (iStar + 0.5) * step;
    const mx = Math.max(...diff), mn = Math.min(...diff);
    if (mx > 0 && mn < 0) v.edge_pair_px = Math.abs(diff.indexOf(mx) - diff.indexOf(mn)) * step;
  }
  return v;
}
function checkProfile(img, ax, ay, bx, by, label, extra = {}) {
  const ctx = runValues('line_profile', img, lpParams(ax, ay, bx, by, img.cols, img.rows, extra));
  const want = refValues(refProfile(img, ax, ay, bx, by), ax, ay, bx, by);
  assert.deepEqual(scalarMap(ctx), want, `${label}: 9 values`);
  return ctx;
}
const isColour = (p, c) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2];

test('ipD-7. line_profile reads exactly the spec samples: a v = x ramp along a horizontal line gives the profile x1p..x2p (min/max/mean/contrast/edge exact, monotonic -> edge_pair null); A<->B reversed reverses it (edge_jump -1); a vertical v = y ramp and a diagonal line on a v = (3x + 7y) & 255 pattern match the reference sampler on all 9 values; in the plot view every sample k is drawn as a yellow polyline vertex at (X_k, Y(v_k))', () => {
  const ramp = gray(200, 100, (x) => x), vramp = gray(100, 200, (x, y) => y), pat = gray(100, 100, (x, y) => (3 * x + 7 * y) & 255);
  try {
    const h = checkProfile(ramp, 20, 40, 179, 40, 'horizontal');
    assert.deepEqual([field(h, 'samples'), field(h, 'profile_min'), field(h, 'profile_max'), field(h, 'profile_mean'), field(h, 'edge_jump'), field(h, 'edge_pos_px'), field(h, 'edge_pair_px')],
      [160, 20, 179, 99.5, 1, 0.5, null]);
    const r = checkProfile(ramp, 179, 40, 20, 40, 'reversed');
    assert.equal(field(r, 'edge_jump'), -1);
    checkProfile(vramp, 40, 20, 40, 179, 'vertical');
    checkProfile(pat, 10, 10, 70, 50, 'diagonal (dx > dy)');
    checkProfile(pat, 90, 5, 12, 97, 'diagonal (dy > dx, reversed x)');
    checkProfile(pat, 0, 99, 99, 0, 'anti-diagonal');
    const { out, ctx } = run('line_profile', ramp, lpParams(20, 40, 179, 40, 200, 100, { view: 'plot' }));
    try {
      assert.equal(field(ctx, 'samples'), 160);
      const N = 160, gw = 200, gh = 100;
      const xe = Math.floor((1 * (gw - 1)) / (N - 1));
      let checked = 0;
      for (let k = 0; k < N; k++) {
        const X = Math.floor((k * (gw - 1)) / (N - 1)), v = 20 + k, Y = gh - 1 - Math.floor((2 * v * (gh - 1) + 255) / 510);
        if (X === xe) continue;
        assert.deepEqual(px(out, X, Y), YELLOW, `vertex k=${k} at (${X},${Y})`);
        checked++;
      }
      assert.ok(checked >= 158);
    } finally { out.delete(); }
  } finally { ramp.delete(); vramp.delete(); pat.delete(); }
});

test('ipD-8. line_profile rounding: % -> px is floor(pct/100*(n-1) + 0.5) (w = 4, 50% -> 1.5 -> pixel 2, not 1); samples use floor(x + 0.5) (A (0,0) -> B (1,2): sample 1 is pixel (1,1), not (0,1) as trunc would give); 0% / 100% are the first / last pixel on both axes', () => {
  const four = gray(4, 4, (x) => x * 50);
  const hit = gray(10, 10, (x, y) => (x === 1 && y === 1 ? 255 : 0)), miss = gray(10, 10, (x, y) => (x === 0 && y === 1 ? 255 : 0));
  const ramp = gray(50, 30, (x, y) => (x * 5) ^ y), edges = gray(50, 30, (x, y) => (x === 0 || x === 49 || y === 0 || y === 29 ? 255 : 0));
  try {
    const v = runValues('line_profile', four, { x1: 50, y1: 0, x2: 50, y2: 100 });
    assert.equal(field(v, 'profile_min'), 100, '50% of w = 4 is pixel 2 (value 100)');
    assert.equal(field(v, 'profile_max'), 100);
    assert.equal(field(v, 'samples'), 4);
    assert.equal(field(runValues('line_profile', hit, lpParams(0, 0, 1, 2, 10, 10)), 'profile_max'), 255, 'sample 1 = (1,1)');
    assert.equal(field(runValues('line_profile', miss, lpParams(0, 0, 1, 2, 10, 10)), 'profile_max'), 0, '(0,1) is never read');
    const full = runValues('line_profile', ramp, { x1: 0, y1: 0, x2: 100, y2: 100 });
    assert.deepEqual(scalarMap(full), refValues(refProfile(ramp, 0, 0, 49, 29), 0, 0, 49, 29), '0% -> 0, 100% -> n-1');
    assert.match(full.info, /^เส้น A\(0,0\)→B\(49,29\) /);
    for (const [p, label] of [[{ x1: 0, y1: 0, x2: 0, y2: 100 }, 'left column'], [{ x1: 100, y1: 0, x2: 100, y2: 100 }, 'right column'], [{ x1: 0, y1: 100, x2: 100, y2: 100 }, 'bottom row'], [{ x1: 0, y1: 0, x2: 100, y2: 0 }, 'top row']]) {
      const c = runValues('line_profile', edges, p);
      assert.equal(field(c, 'profile_min'), 255, `${label}: every sample on the border`);
    }
  } finally { for (const m of [four, hit, miss, ramp, edges]) m.delete(); }
});

test('ipD-9. line_profile on a 60|190 step (step between x = 99 and 100): edge_jump +130 at edge_pos exactly 79.5 px from A (i* = 79, step 1), contrast 130, edge_pair null; reversed: -130 at 79.5; the first of equal jumps wins (two equal steps -> the one nearer A); info carries A/B in px, length, samples, range and the signed jump', () => {
  const step = gray(200, 100, (x) => (x < 100 ? 60 : 190));
  const twin = gray(200, 100, (x) => (x < 60 ? 0 : x < 120 ? 100 : 200));
  try {
    const a = checkProfile(step, 20, 40, 179, 40, 'step');
    assert.deepEqual([field(a, 'edge_jump'), field(a, 'edge_pos_px'), field(a, 'profile_contrast'), field(a, 'edge_pair_px')], [130, 79.5, 130, null]);
    assert.equal(a.info, 'เส้น A(20,40)→B(179,40) ยาว 159.0 px · อ่าน 160 จุด · ค่า 60–190 · ขอบแรงสุดห่างจาก A 79.5 px (กระโดด +130)');
    // Q11: the magenta ring (radius 4) is centred on sample i* + 1 = x 100 (not i* = 99)
    const ring = run('line_profile', step, lpParams(20, 40, 179, 40, 200, 100));
    try {
      assert.deepEqual([px(ring.out, 104, 40), px(ring.out, 96, 40), px(ring.out, 100, 44), px(ring.out, 100, 36)], [MAGENTA, MAGENTA, MAGENTA, MAGENTA], 'ring around (100, 40)');
      assert.ok(!isColour(px(ring.out, 95, 40), MAGENTA) && !isColour(px(ring.out, 103, 40), MAGENTA), 'not centred on 99');
    } finally { ring.out.delete(); }
    const b = checkProfile(step, 179, 40, 20, 40, 'step reversed');
    assert.deepEqual([field(b, 'edge_jump'), field(b, 'edge_pos_px')], [-130, 79.5]);
    assert.match(b.info, /\(กระโดด -130\)$/);
    const t = checkProfile(twin, 20, 40, 179, 40, 'two equal jumps');
    assert.equal(field(t, 'edge_pos_px'), 39.5, 'first occurrence (x 59 -> 60), not the later one (x 119 -> 120)');
  } finally { step.delete(); twin.delete(); }
});

test('ipD-10. line_profile edge pair = stripe width: a 10-px bright bar crossed horizontally -> exactly 10; crossed by a 45 deg diagonal -> 10 * step = 10 * sqrt(2) (spec: slanted lines read longer); a monotonic profile -> null; only-rising / only-falling -> null', () => {
  const bar = gray(200, 100, (x) => (x >= 100 && x < 110 ? 200 : 20));
  const vbar = gray(140, 140, (x) => (x >= 60 && x < 70 ? 200 : 20));
  const ramp = gray(200, 100, (x) => x);
  try {
    assert.equal(field(checkProfile(bar, 20, 40, 179, 40, 'bar'), 'edge_pair_px'), 10);
    const d = checkProfile(vbar, 20, 20, 120, 120, 'diagonal');
    assert.equal(field(d, 'edge_pair_px'), 10 * (Math.sqrt(20000) / 100));
    close(field(d, 'edge_pair_px'), 10 * Math.SQRT2, 1e-12, 'diagonal edge pair');
    assert.equal(field(checkProfile(ramp, 20, 40, 179, 40, 'ramp'), 'edge_pair_px'), null);
    assert.equal(field(checkProfile(bar, 20, 40, 104, 40, 'rising only'), 'edge_pair_px'), null);
    assert.equal(field(checkProfile(bar, 105, 40, 179, 40, 'falling only'), 'edge_pair_px'), null);
  } finally { bar.delete(); vbar.delete(); ramp.delete(); }
});

test('ipD-11. line_profile A = B (and a 1x1 image): samples 1, length 0, every edge field null, info "จุด A กับ B อยู่ตำแหน่งเดียวกัน — ปรับให้ห่างกันก่อน"; with a graph the single sample is a yellow dot at the middle column x = floor((gw-1)/2), row Y(v); a 1x1 image adds the small-image note and draws no graph', () => {
  const img = gray(80, 80, () => 255), one = gray(1, 1, () => 77);
  try {
    const { out, ctx } = run('line_profile', img, { x1: 30, y1: 30, x2: 30, y2: 30 });
    try {
      assert.deepEqual(scalarMap(ctx), { line_length_px: 0, samples: 1, profile_min: 255, profile_max: 255, profile_mean: 255, profile_contrast: 0, edge_jump: null, edge_pos_px: null, edge_pair_px: null });
      assert.equal(ctx.info, 'จุด A กับ B อยู่ตำแหน่งเดียวกัน — ปรับให้ห่างกันก่อน');
      const gh = Math.floor(0.35 * 80), gy0 = 80 - gh;
      assert.deepEqual(px(out, 39, gy0), YELLOW, 'dot at (floor(79/2), Y(255) = gy0)');
    } finally { out.delete(); }
    const o = run('line_profile', one, {});
    try {
      assert.deepEqual(scalarMap(o.ctx), { line_length_px: 0, samples: 1, profile_min: 77, profile_max: 77, profile_mean: 77, profile_contrast: 0, edge_jump: null, edge_pos_px: null, edge_pair_px: null });
      assert.equal(o.ctx.info, 'จุด A กับ B อยู่ตำแหน่งเดียวกัน — ปรับให้ห่างกันก่อน · ภาพเล็กกว่า 64×64 px จึงไม่วาดกราฟ (ตัวเลขยังคำนวณครบ)');
      assert.equal(o.out.channels(), 3);
    } finally { o.out.delete(); }
  } finally { img.delete(); one.delete(); }
});

test('ipD-12. line_profile graph: (a) constant 255 -> the profile polyline lies on row Y(255) = gy0 (bottom band, gh = floor(0.35 H)); (b) the band is darkened by convertScaleAbs alpha 0.3 and nothing above it changes; (c) both A and B inside the bottom band -> the band moves to the top, only one of them -> stays at the bottom; (d) plot view = input size, dark (24,24,24) background, no A->B line; (e) an image < 64 px on either side gets no graph (+ info note) and the plot view falls back to the overlay picture', () => {
  const W = 200, H = 100, gh = Math.floor(0.35 * H);
  const white = gray(W, H, () => 255), scene = sceneGray(W, H), small = sceneGray(100, 63), smallW = sceneGray(63, 100);
  const sceneRGBm = toRGB(scene);
  const darkRef = new cv.Mat();
  cv.convertScaleAbs(sceneRGBm, darkRef, 0.3, 0);
  const bandCheck = (out, y0, label) => {
    let same = 0, total = 0;
    for (let y = y0; y < y0 + gh; y++) {
      if (y === y0 || y === y0 + gh - 1) continue; // เส้นเทา Y(255) / Y(0)
      for (let x = 0; x < W; x++) {
        const p = px(out, x, y);
        if (p[0] !== p[1] || p[1] !== p[2]) continue; // จุดที่ถูกวาดสี
        total++;
        if (p[0] === px(darkRef, x, y)[0]) same++;
      }
    }
    assert.ok(total > 0.8 * W * (gh - 2), `${label}: most band pixels are undrawn (${total})`);
    assert.equal(same, total, `${label}: band = convertScaleAbs(src, 0.3)`);
  };
  try {
    const a = run('line_profile', white, {});
    try {
      let yellow = 0;
      for (let x = 0; x < W; x++) if (isColour(px(a.out, x, H - gh), YELLOW)) yellow++;
      assert.ok(yellow >= W - 2, `polyline on row Y(255) = ${H - gh} (${yellow} yellow)`);
      assert.deepEqual(px(a.out, 0, H - 1), GREY, 'grey grid line at Y(0) = bottom row of the band');
    } finally { a.out.delete(); }
    const b = run('line_profile', scene, {});
    try {
      bandCheck(b.out, H - gh, 'bottom band');
      for (let y = 0; y < 40; y++) for (let x = 0; x < W; x++) assert.equal(px(b.out, x, y)[0], px(scene, x, y)[0], `above the band unchanged at ${x},${y}`);
    } finally { b.out.delete(); }
    const c = run('line_profile', scene, { y1: 95, y2: 95 });
    try {
      bandCheck(c.out, 0, 'top band (A and B low)');
      for (let x = 0; x < W; x++) assert.deepEqual(px(c.out, x, 50), [px(scene, x, 50)[0], px(scene, x, 50)[0], px(scene, x, 50)[0]], 'middle rows untouched');
    } finally { c.out.delete(); }
    const c2 = run('line_profile', scene, { y1: 95, y2: 50 });
    try { bandCheck(c2.out, H - gh, 'only A low -> band stays at the bottom'); } finally { c2.out.delete(); }
    const c3 = run('line_profile', scene, lpParams(10, H - gh, 190, H - gh, W, H)); // Q8: exactly on the first row of the bottom band
    try { assert.equal(c3.ctx.values.fields.length, 9); bandCheck(c3.out, 0, 'A and B exactly on row H - gh -> band moves to the top'); } finally { c3.out.delete(); }
    const d = run('line_profile', scene, { view: 'plot' });
    try {
      assert.deepEqual([d.out.rows, d.out.cols, d.out.channels()], [H, W, 3]);
      let bg = 0;
      for (let i = 0; i < d.out.data.length; i += 3) if (d.out.data[i] === 24 && d.out.data[i + 1] === 24 && d.out.data[i + 2] === 24) bg++;
      assert.ok(bg > 0.9 * W * H, `plot background (24,24,24): ${bg}`);
      assert.deepEqual(px(d.out, 0, 0), GREY, 'Y(255) = row 0 in the plot view');
      assert.deepEqual(px(d.out, 0, H - 1), GREY, 'Y(0) = last row');
      assert.ok(!isColour(px(d.out, 20, 50), GREEN), 'no A marker in the plot view');
    } finally { d.out.delete(); }
    const edge64 = gray(64, 64, () => 200); // 64×64 พอดี = วาดกราฟ (เกณฑ์คือ "เล็กกว่า 64")
    const g64 = run('line_profile', edge64, {});
    try {
      assert.doesNotMatch(g64.ctx.info, /ภาพเล็กกว่า/);
      assert.deepEqual(px(g64.out, 10, 50), [60, 60, 60], '64x64: band darkened (200 * 0.3)');
    } finally { g64.out.delete(); edge64.delete(); }
    // H = 70 -> 0.35 * H = 24.5: the band height is floor (24, top row 46), not round (25, top row 45) -- mutation L11b
    const h70 = gray(100, 70, () => 200);
    const g70 = run('line_profile', h70, {});
    try {
      assert.deepEqual(px(g70.out, 50, 45), [200, 200, 200], 'H = 70: row 45 is above the band (untouched)');
      assert.deepEqual(px(g70.out, 50, 46), GREY, 'H = 70: the band starts at row H - floor(0.35 H) = 46 (grey Y(255) line on its top row)');
    } finally { g70.out.delete(); h70.delete(); }
    for (const sm of [small, smallW]) {
      const e = run('line_profile', sm, {});
      const f = run('line_profile', sm, { view: 'plot' });
      try {
        assert.match(e.ctx.info, / · ภาพเล็กกว่า 64×64 px จึงไม่วาดกราฟ \(ตัวเลขยังคำนวณครบ\)$/);
        sameBytes(f.out, e.out, `small ${sm.cols}x${sm.rows}: plot falls back to the overlay picture`);
        const ref = toRGB(sm);
        try {
          const ay = Math.floor(0.5 * (sm.rows - 1) + 0.5);
          let changed = 0;
          for (let y = 0; y < sm.rows; y++) for (let x = 0; x < sm.cols; x++) {
            if (Math.abs(y - ay) <= 5) continue; // เส้น A→B + วงกลม
            if (px(e.out, x, y)[0] !== px(ref, x, y)[0] || px(e.out, x, y)[2] !== px(ref, x, y)[2]) changed++;
          }
          assert.equal(changed, 0, 'no darkened band / no graph');
        } finally { ref.delete(); }
      } finally { e.out.delete(); f.out.delete(); }
    }
  } finally { for (const m of [white, scene, small, smallW, sceneRGBm, darkRef]) m.delete(); }
});

test('ipD-26. labels near the top edge stay readable (Math.max(10, y - 5)): a feret object whose bbox top is y = 3 gets its yellow number inside rows 0..14; a roi_polygon overlay vertex at y = 3 gets its white number there too', () => {
  const m = gray(60, 40, (x, y) => (x >= 20 && x < 30 && y >= 3 && y < 13 ? 255 : 0));
  try {
    const r = run('feret', m, {});
    try {
      assert.equal(r.ctx.values.rows[0][3] > 0, true);
      let yel = 0;
      for (let y = 0; y < 15; y++) for (let x = 20; x < 31; x++) if (isColour(px(r.out, x, y), YELLOW)) yel++;
      assert.ok(yel >= 5, `feret label pixels in [20..30] x [0..14]: ${yel}`);
    } finally { r.out.delete(); }
  } finally { m.delete(); }
  const src = gray(101, 101, () => 50);
  try {
    const r = run('roi_polygon', src, roiParams([[40, 3], [90, 3], [90, 60], [40, 60]], 101, 101, { view: 'overlay' }));
    try {
      let wh = 0;
      for (let y = 0; y < 15; y++) for (let x = 45; x < 58; x++) if (isColour(px(r.out, x, y), [255, 255, 255])) wh++;
      assert.ok(wh >= 5, `vertex 1 label pixels in [45..57] x [0..14]: ${wh}`);
    } finally { r.out.delete(); }
  } finally { src.delete(); }
});

test('ipD-27. line_profile on a flat line (every sample equal, N >= 2): edge_jump / edge_pos_px / edge_pair_px are null, no magenta ring or graph line is drawn (overlay and plot), the info says the values are equal everywhere instead of the edge clause; a profile that differs by just 1 still reports its edge', () => {
  const flat = gray(200, 100, () => 120);
  const near = gray(200, 100, (x) => (x >= 100 ? 121 : 120));
  try {
    for (const view of ['overlay', 'plot']) {
      const r = run('line_profile', flat, { view });
      try {
        const f = scalarMap(r.ctx);
        assert.deepEqual([f.edge_jump, f.edge_pos_px, f.edge_pair_px], [null, null, null], `${view}: edge fields null`);
        assert.equal(f.samples, 160);
        assert.equal(f.profile_contrast, 0);
        assert.match(r.ctx.info, / · ค่า 120–120 · ค่าบนเส้นเท่ากันทุกจุด ไม่มีขอบ$/);
        assert.doesNotMatch(r.ctx.info, /ขอบแรงสุด/);
        let mg = 0;
        for (let i = 0; i < r.out.data.length; i += 3) if (r.out.data[i] === 255 && r.out.data[i + 1] === 0 && r.out.data[i + 2] === 255) mg++;
        assert.equal(mg, 0, `${view}: no magenta pixel`);
      } finally { r.out.delete(); }
    }
    const n = run('line_profile', near, {});
    try {
      assert.deepEqual([field(n.ctx, 'edge_jump'), field(n.ctx, 'edge_pos_px')], [1, 79.5]);
    } finally { n.out.delete(); }
  } finally { flat.delete(); near.delete(); }
});

// ===================================================================
// ipD-13…17 — roi_polygon
// ===================================================================
const WHOLE = [[0, 0], [100, 0], [100, 100], [0, 100]];
function wholeParams(extra = {}) { const p = { points: 4, ...extra }; WHOLE.forEach(([x, y], i) => { p[`p${i + 1}x`] = x; p[`p${i + 1}y`] = y; }); return p; }

test('ipD-13. roi_polygon over the whole frame (0,0)-(100,0)-(100,100)-(0,100): the mask is 255 everywhere and apply returns the input byte for byte, area = w*h, 100% -- at 640x480, 97x61, 1x1, 2x1 and 1x50 (vertex 100% = last pixel)', () => {
  for (const [w, h] of [[640, 480], [97, 61], [1, 1], [2, 1], [1, 50]]) {
    const src = noiseRGB(w, h, w + h);
    try {
      const m = run('roi_polygon', src, wholeParams({ view: 'mask' }));
      const a = run('roi_polygon', src, wholeParams());
      try {
        assert.equal(m.out.channels(), 1);
        assert.ok(m.out.data.every((v) => v === 255), `${w}x${h}: mask all 255`);
        sameBytes(a.out, src, `${w}x${h}: apply = input`);
        assert.equal(field(a.ctx, 'roi_area_px'), w * h);
        assert.equal(field(a.ctx, 'roi_area_pct'), 100);
      } finally { m.out.delete(); a.out.delete(); }
    } finally { src.delete(); }
  }
});

function segDist(px0, py0, [ax, ay], [bx, by]) {
  const vx = bx - ax, vy = by - ay, t = Math.max(0, Math.min(1, ((px0 - ax) * vx + (py0 - ay) * vy) / (vx * vx + vy * vy)));
  return Math.hypot(px0 - (ax + t * vx), py0 - (ay + t * vy));
}
function insidePoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

test('ipD-14. roi_polygon mask = the filled polygon: on a triangle (10,10)-(90,15)-(40,80) px every pixel strictly inside (point-in-polygon, > 1 px from every edge) is 255 and every pixel > 1 px outside is 0; the same for the default 4..6-point polygons on 97x61', () => {
  const cases = [[100, 100, [[10, 10], [90, 15], [40, 80]]]];
  for (const n of [4, 5, 6]) {
    const def = OPS.roi_polygon.params;
    const pts = [];
    for (let i = 1; i <= n; i++) {
      const dx = def.find((p) => p.key === `p${i}x`).default, dy = def.find((p) => p.key === `p${i}y`).default;
      pts.push([Math.floor((dx / 100) * 96 + 0.5), Math.floor((dy / 100) * 60 + 0.5)]);
    }
    cases.push([97, 61, pts]);
  }
  for (const [w, h, poly] of cases) {
    const src = gray(w, h, () => 0);
    try {
      const { out } = run('roi_polygon', src, roiParams(poly, w, h, { view: 'mask' }));
      try {
        let nin = 0, nout = 0;
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
          const dmin = Math.min(...poly.map((p, i) => segDist(x, y, p, poly[(i + 1) % poly.length])));
          if (dmin <= 1) continue;
          const v = out.data[y * w + x];
          if (insidePoly(x, y, poly)) { assert.equal(v, 255, `${w}x${h} inside ${x},${y}`); nin++; } else { assert.equal(v, 0, `${w}x${h} outside ${x},${y}`); nout++; }
        }
        assert.ok(nin > 100 && nout > 100);
      } finally { out.delete(); }
    } finally { src.delete(); }
  }
});

test('ipD-15. roi_polygon views: apply = src inside / 0 outside (gray stays 1 channel, RGBA -> 3 channels equal to its RGB version); mask = 1 channel of only 0/255; overlay = src inside and round(0.4 * src) outside except the yellow outline / vertex dots / white labels (checked outside a 3-px band around the outline and the label boxes)', () => {
  const W = 120, H = 90;
  const c = sceneRGB(W, H), a = toRGBA(c), g = sceneGray(W, H);
  try {
    const mask = run('roi_polygon', c, { view: 'mask' });
    try {
      assert.equal(mask.out.channels(), 1);
      assert.ok(mask.out.data.every((v) => v === 0 || v === 255));
      const md = mask.out.data;
      const ap = run('roi_polygon', c, {});
      const apA = run('roi_polygon', a, {});
      const apG = run('roi_polygon', g, {});
      try {
        assert.equal(ap.out.channels(), 3);
        assert.equal(apG.out.channels(), 1);
        sameBytes(apA.out, ap.out, 'RGBA apply = RGB apply');
        for (let i = 0; i < W * H; i++) {
          for (let k = 0; k < 3; k++) assert.equal(ap.out.data[3 * i + k], md[i] ? c.data[3 * i + k] : 0);
          assert.equal(apG.out.data[i], md[i] ? g.data[i] : 0);
        }
      } finally { ap.out.delete(); apA.out.delete(); apG.out.delete(); }
      const ov = run('roi_polygon', c, { view: 'overlay', points: 5 });
      try {
        const pts = [];
        for (let i = 1; i <= 5; i++) pts.push([Math.floor((OPS.roi_polygon.params.find((p) => p.key === `p${i}x`).default / 100) * (W - 1) + 0.5), Math.floor((OPS.roi_polygon.params.find((p) => p.key === `p${i}y`).default / 100) * (H - 1) + 0.5)]);
        const m5 = run('roi_polygon', c, { view: 'mask', points: 5 });
        try {
          let ins = 0, outs = 0;
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            if (Math.min(...pts.map((p, i) => segDist(x, y, p, pts[(i + 1) % 5]))) <= 3) continue;
            if (pts.some(([vx, vy]) => { const ax = vx + 5, ay = Math.max(10, vy - 5); return x >= ax - 3 && x <= ax + 15 && y >= ay - 14 && y <= ay + 4; })) continue;
            const i = y * W + x;
            for (let k = 0; k < 3; k++) {
              const s = c.data[3 * i + k];
              assert.equal(ov.out.data[3 * i + k], m5.out.data[i] ? s : Math.round(0.4 * s), `overlay ${x},${y}`);
            }
            if (m5.out.data[i]) ins++; else outs++;
          }
          assert.ok(ins > 500 && outs > 500);
          assert.deepEqual(px(ov.out, pts[0][0], pts[0][1]), YELLOW, 'vertex dot');
        } finally { m5.out.delete(); }
      } finally { ov.out.delete(); }
    } finally { mask.out.delete(); }
  } finally { c.delete(); a.delete(); g.delete(); }
});

test('ipD-16. roi_polygon shape checks: the default polygons for n = 3..6 are simple (roi_crossed false, area > 0, no warning) on 640x480 and 97x61; a bowtie -> crossed true + the crossing warning (and NOT the collinear one even though its shoelace area is 0); collinear points -> the collinear warning (not the crossing one); all points stacked -> 1 px; perimeter includes the closing edge (3-4-5 triangle 30/40/50 px -> 120)', () => {
  for (const [w, h] of [[640, 480], [97, 61]]) {
    const src = gray(w, h, () => 100);
    try {
      for (let n = 3; n <= 6; n++) {
        const ctx = runValues('roi_polygon', src, { points: n });
        assert.equal(field(ctx, 'roi_crossed'), false, `${w}x${h} n=${n}`);
        assert.ok(field(ctx, 'roi_area_px') > 0);
        assert.doesNotMatch(ctx.info, /⚠️/);
        assert.equal(field(ctx, 'roi_vertices'), n);
      }
    } finally { src.delete(); }
  }
  const src = gray(101, 101, () => 50);
  try {
    const bow = runValues('roi_polygon', src, { p1x: 10, p1y: 10, p2x: 90, p2y: 90, p3x: 90, p3y: 10, p4x: 10, p4y: 90 });
    assert.equal(field(bow, 'roi_crossed'), true);
    assert.match(bow.info, / · ⚠️ ลำดับจุดไขว้กัน รูปจะไม่ใช่รูปที่ตั้งใจ — เรียงจุดตามเข็มหรือทวนเข็มนาฬิกา$/);
    assert.doesNotMatch(bow.info, /เส้นตรง/);
    assert.ok(field(bow, 'roi_area_px') > 2000, 'fillPoly fills both lobes');
    const col = runValues('roi_polygon', src, { points: 3, p1x: 10, p1y: 10, p2x: 50, p2y: 50, p3x: 90, p3y: 90 });
    assert.match(col.info, / · ⚠️ ทุกจุดเรียงเป็นเส้นตรง\/ซ้อนกัน ROI จึงเหลือแค่เส้นบาง ๆ$/);
    assert.equal(field(col, 'roi_area_px'), 81, 'only the diagonal line (x 10..90)');
    const col4 = runValues('roi_polygon', src, { points: 4, p1x: 10, p1y: 10, p2x: 30, p2y: 30, p3x: 60, p3y: 60, p4x: 90, p4y: 90 });
    assert.match(col4.info, /เรียงเป็นเส้นตรง/);
    assert.doesNotMatch(col4.info, /ไขว้/);
    const st = runValues('roi_polygon', src, { points: 6, p1x: 50, p1y: 50, p2x: 50, p2y: 50, p3x: 50, p3y: 50, p4x: 50, p4y: 50, p5x: 50, p5y: 50, p6x: 50, p6y: 50 });
    assert.equal(field(st, 'roi_area_px'), 1);
    assert.equal(field(st, 'roi_perimeter_px'), 0);
    assert.match(st.info, /เรียงเป็นเส้นตรง/);
    const tri = runValues('roi_polygon', src, roiParams([[0, 0], [30, 0], [0, 40]], 101, 101));
    assert.equal(field(tri, 'roi_perimeter_px'), 120);
    const sq = runValues('roi_polygon', src, roiParams([[0, 0], [10, 0], [10, 10], [0, 10]], 101, 101));
    assert.equal(field(sq, 'roi_perimeter_px'), 40);
    // MINOR-1: a repeated point is a zero-length edge and must not make its two neighbours "touch" (false crossed)
    const simple = (pts, label) => {
      const c = runValues('roi_polygon', src, roiParams(pts, 101, 101));
      assert.equal(field(c, 'roi_crossed'), false, `${label}: crossed false`);
      assert.doesNotMatch(c.info, /⚠️/, `${label}: no warning`);
      assert.ok(field(c, 'roi_area_px') > 100, `${label}: has area`);
    };
    simple([[30, 20], [70, 20], [90, 85], [90, 85]], 'P4 = P3');
    simple([[10, 10], [50, 10], [50, 50], [10, 10]], 'P4 = P1 (closing duplicate)');
    simple([[10, 10], [10, 10], [60, 20], [30, 70]], 'P1 = P2 (P3, P4 not collinear: not degenerate)');
    simple([[10, 10], [60, 20], [60, 20], [30, 70]], 'P2 = P3');
    const five = gray(5, 5, () => 100);
    try {
      const c5 = runValues('roi_polygon', five, { points: 6 }); // P6 rounds onto P1 on 5x5
      assert.equal(field(c5, 'roi_crossed'), false, '5x5 default n = 6 (P6 -> P1)');
      assert.doesNotMatch(c5.info, /⚠️/);
    } finally { five.delete(); }
    // Q12: concave polygons that do NOT cross (the line through one edge cuts the other edge's line but not the segment)
    const variants = (pts) => { const out = []; for (let sh = 0; sh < pts.length; sh++) for (const rev of [false, true]) { const a = pts.map((_, i) => pts[(i + sh) % pts.length]); out.push(rev ? a.reverse() : a); } return out; };
    for (const pts of [[[10, 80], [50, 20], [90, 80], [50, 50]], [[10, 50], [40, 10], [40, 30], [90, 30], [90, 70], [40, 70]]]) {
      for (const v of variants(pts)) simple(v, `concave ${pts.length}-gon ${JSON.stringify(v)}`);
    }
    // Q5: a vertex touching the MIDDLE of a non-adjacent edge counts as crossed (every cyclic shift / direction = every clause of the touch test)
    for (const v of variants([[10, 10], [90, 10], [90, 90], [50, 10]])) {
      const c = runValues('roi_polygon', src, roiParams(v, 101, 101));
      assert.equal(field(c, 'roi_crossed'), true, `vertex on an edge: ${JSON.stringify(v)}`);
      assert.match(c.info, /ไขว้กัน/);
    }
    // two edges overlapping along one line (collinear overlap) also touch
    assert.equal(field(runValues('roi_polygon', src, roiParams([[10, 10], [60, 10], [60, 50], [30, 10], [30, 80]], 101, 101)), 'roi_crossed'), true, 'collinear overlap');
  } finally { src.delete(); }
});

test('ipD-17. roi_polygon numbers: mean / std (population) equal integer brute-force sums over the mask on gray and colour (colour measured on RGB2GRAY), area_pct = 100*a/(w*h); clamping like the UI -- points 2 == 3, 9 == 6, "5.4" == 5, garbage == 4; coordinates -10 == 0, 150 == 100, garbage == default; an unknown view ("constructor") == apply; info "ROI n จุด · N px (X.X% ของภาพ) · ความสว่างเฉลี่ย M.M" + the overlay hint outside the overlay view', () => {
  const c = noiseRGB(97, 61, 31), g = noiseGray(97, 61, 32);
  try {
    for (const src of [c, g]) for (const n of [3, 5, 6]) {
      const ctx = runValues('roi_polygon', src, { points: n });
      const m = run('roi_polygon', src, { points: n, view: 'mask' });
      try {
        const gd = grayData(src);
        let a = 0, s = 0, s2 = 0;
        for (let i = 0; i < gd.length; i++) if (m.out.data[i]) { a++; s += gd[i]; s2 += gd[i] * gd[i]; }
        const mean = s / a;
        assert.equal(field(ctx, 'roi_area_px'), a);
        assert.equal(field(ctx, 'roi_mean'), mean);
        assert.equal(field(ctx, 'roi_std'), Math.sqrt(Math.max(0, s2 / a - mean * mean)));
        assert.equal(field(ctx, 'roi_area_pct'), (100 * a) / (97 * 61));
        const f1 = (v) => (Math.floor(v * 10 + 0.5) / 10).toFixed(1);
        assert.equal(ctx.info, `ROI ${n} จุด · ${a} px (${f1((100 * a) / (97 * 61))}% ของภาพ) · ความสว่างเฉลี่ย ${f1(mean)} (เลือก 'แสดงผลเป็น: ดูตำแหน่ง' เพื่อเห็นเส้นขอบและเลขจุด)`);
      } finally { m.out.delete(); }
    }
    const same = (p1, p2, label) => {
      const a = run('roi_polygon', c, p1), b = run('roi_polygon', c, p2);
      try { sameBytes(a.out, b.out, label); assert.deepEqual(a.ctx.values, b.ctx.values, `${label}: values`); } finally { a.out.delete(); b.out.delete(); }
    };
    same({ points: 2 }, { points: 3 }, 'points 2 -> 3');
    same({ points: 9 }, { points: 6 }, 'points 9 -> 6');
    same({ points: '5.4' }, { points: 5 }, 'points "5.4" -> 5');
    same({ points: 'abc' }, { points: 4 }, 'points garbage -> 4');
    same({ p1x: -10, p2y: 150 }, { p1x: 0, p2y: 100 }, 'coords clamp');
    same({ p3x: 'x', p4y: 'abc' }, {}, 'coords garbage -> defaults');
    same({ view: 'constructor' }, { view: 'apply' }, 'unknown view -> apply');
    same({ view: '__proto__' }, {}, '__proto__ view -> apply');
    assert.doesNotMatch(runValues('roi_polygon', c, { view: 'overlay' }).info, /ดูตำแหน่ง/);
    const lp1 = run('line_profile', c, { view: 'constructor' }), lp2 = run('line_profile', c, {});
    try { sameBytes(lp1.out, lp2.out, 'line_profile unknown view -> overlay'); } finally { lp1.out.delete(); lp2.out.delete(); }
    const lpc = (p, q, label) => { const a = runValues('line_profile', c, p), b = runValues('line_profile', c, q); assert.deepEqual(a.values, b.values, label); };
    lpc({ x1: -5, x2: 900 }, { x1: 0, x2: 100 }, 'line_profile coords clamp');
    lpc({ x1: 'abc', y2: undefined }, {}, 'line_profile garbage -> defaults');
    const fv = (p, q, label) => { const b0 = blobsGray(); try { assert.deepEqual(runValues('feret', b0, p).values, runValues('feret', b0, q).values, label); } finally { b0.delete(); } };
    fv({ minArea: -5, maxObjects: 0 }, { minArea: 0, maxObjects: 1 }, 'feret clamp low');
    fv({ minArea: 1e9, maxObjects: 9999 }, { minArea: 5000, maxObjects: 500 }, 'feret clamp high');
    fv({ minArea: 'x', maxObjects: '2.6' }, { minArea: 50, maxObjects: 3 }, 'feret garbage / rounding');
  } finally { c.delete(); g.delete(); }
});

// ===================================================================
// ipD-18 — สัญญา ctx.values (ทุกเส้นทางส่ง key/คอลัมน์ชุดเดียวกัน — detail-panel staleWatch พึ่งข้อนี้)
// ===================================================================
test('ipD-18. values contract: every path of all 3 ops sets ctx.values with the same keys / columns in the same order -- 0x0 (feret rows [] total 0; line_profile all 9 null; roi all null except roi_vertices = n), nothing found, A = B, small image, every view, 1x1 -- every feret row is columns.length long, a fresh object every call, and the 0x0 path returns a new 0x0 Mat with info "ภาพว่าง"', () => {
  const empty = new cv.Mat(), b = blobsGray(), black = gray(20, 10, () => 0), one = gray(1, 1, () => 255), s = sceneGray(80, 80), sm = sceneGray(30, 20), c = sceneRGB(80, 80);
  try {
    const feretCases = [[empty, {}], [b, {}], [b, { minArea: 0 }], [b, { minArea: 0, maxObjects: 1 }], [black, {}], [one, { minArea: 0 }], [c, {}]];
    for (const [src, p] of feretCases) {
      const ctx = runValues('feret', src, p);
      assert.equal(ctx.values.kind, 'table');
      assert.deepEqual(ctx.values.columns, FERET_COLUMNS);
      assert.equal(typeof ctx.values.total, 'number');
      for (const r of ctx.values.rows) assert.equal(r.length, FERET_COLUMNS.length);
    }
    const lpCases = [[empty, {}], [s, {}], [s, { view: 'plot' }], [s, { x1: 4, x2: 4, y1: 4, y2: 4 }], [sm, {}], [sm, { view: 'plot' }], [one, {}], [c, { y1: 99, y2: 99 }]];
    for (const [src, p] of lpCases) assert.deepEqual(runValues('line_profile', src, p).values.fields.map((f) => f.key), LP_KEYS);
    const roiCases = [[empty, {}], [empty, { points: 6, view: 'overlay' }], [s, {}], [s, { view: 'mask' }], [c, { view: 'overlay' }], [one, {}], [s, { points: 3, p1x: 5, p1y: 5, p2x: 5, p2y: 5, p3x: 5, p3y: 5 }]];
    for (const [src, p] of roiCases) assert.deepEqual(runValues('roi_polygon', src, p).values.fields.map((f) => f.key), ROI_KEYS);
    const e1 = run('feret', empty, {}), e2 = run('line_profile', empty, {}), e3 = run('roi_polygon', empty, { points: 5 });
    try {
      for (const e of [e1, e2, e3]) { assert.equal(e.ctx.info, 'ภาพว่าง'); assert.equal(e.out.rows, 0); assert.notEqual(e.out, empty); }
      assert.deepEqual(e1.ctx.values, { kind: 'table', columns: FERET_COLUMNS, rows: [], total: 0 });
      assert.ok(e2.ctx.values.fields.every((f) => f.value === null));
      assert.deepEqual(e3.ctx.values.fields.map((f) => f.value), [5, null, null, null, null, null, null]);
    } finally { e1.out.delete(); e2.out.delete(); e3.out.delete(); }
    const v1 = runValues('feret', b, {}).values, v2 = runValues('feret', b, {}).values;
    assert.notEqual(v1, v2);
    assert.notEqual(v1.columns, FERET_COLUMNS, 'columns copied, not the shared constant');
    const r1 = runValues('roi_polygon', one, {});
    assert.equal(field(r1, 'roi_area_px'), 1);
    assert.equal(field(r1, 'roi_area_pct'), 100);
    const f1 = runValues('feret', one, { minArea: 0 });
    assert.equal(f1.values.total, 1);
    assert.equal(rowsAsObjects(f1.values)[0].feret_max, Math.SQRT2);
  } finally { for (const m of [empty, b, black, one, s, sm, c]) m.delete(); }
});

// ===================================================================
// ipD-19…25 — static (โครงสร้าง / UI / codegen / ข้อความ)
// ===================================================================
const readSrc = (rel) => fs.readFileSync(new URL(`../src/renderer/${rel}`, import.meta.url), 'utf8');
const CS = (space) => ({ id: 'color_space', params: { mode: 'colorspace', colorspace__space: space } });
const STEP = (id, mode, extra = {}) => ({ id, params: { ...defaultParams(id), mode, ...extra } });
const OPSTEP = (opId, extra = {}) => {
  const prefixed = {};
  for (const [k, v] of Object.entries(extra)) prefixed[`${opId}__${k}`] = v;
  return STEP(HOST[opId], opId, prefixed);
};

test('ipD-19. (static structure) the 3 ops exist with needs gray / any / any (no "mode" param key), FILTERS is still 26, shape_analysis keeps its 6 modes + default find_contours (+ legacy watershed) and appends feret, line_profile; geometry keeps its 7 modes + default resize and appends roi_polygon; defaultParams() = OPS defaults (prefixed); the merged nodes dispatch to the new ops; the legacy segmentation group does not offer them', () => {
  assert.deepEqual(NEW_OPS.map((id) => OPS[id] && OPS[id].needs), ['gray', 'any', 'any']);
  for (const id of NEW_OPS) for (const p of OPS[id].params) assert.notEqual(p.key, 'mode');
  assert.equal(Object.keys(FILTERS).length, 26);
  const modes = (f) => FILTERS[f].params.find((p) => p.key === 'mode').options.map((o) => o.value);
  assert.deepEqual(modes('shape_analysis').slice(0, 6), ['find_contours', 'connected_components', 'distance_transform', 'local_std_filter', 'local_entropy_filter', 'regionprops']);
  assert.deepEqual(modes('shape_analysis').filter((m) => NEW_OPS.includes(m)), ['feret', 'line_profile']);
  assert.equal(modes('shape_analysis').indexOf('feret'), modes('shape_analysis').indexOf('regionprops') + 1);
  assert.deepEqual(modes('geometry'), ['crop', 'resize', 'rotate', 'flip', 'perspective_warp', 'undistort', 'translate_shear', 'roi_polygon']);
  assert.equal(defaultParams('shape_analysis').mode, 'find_contours');
  assert.equal(defaultParams('geometry').mode, 'resize');
  for (const legacy of ['segmentation', 'restoration', 'enhancement']) for (const id of NEW_OPS) assert.ok(!modes(legacy).includes(id));
  for (const id of NEW_OPS) {
    const dp = defaultParams(HOST[id]);
    for (const p of OPS[id].params) assert.deepEqual(dp[`${id}__${p.key}`], p.default, `${HOST[id]}.${id}__${p.key}`);
  }
  assert.equal(needsOf(FILTERS.shape_analysis, { mode: 'feret' }), 'gray');
  assert.equal(needsOf(FILTERS.shape_analysis, { mode: 'line_profile' }), 'any');
  const g = blobsGray(), c = sceneRGB(80, 80);
  try {
    for (const [id, src] of [['feret', g], ['line_profile', c], ['roi_polygon', c]]) {
      const viaNode = FILTERS[HOST[id]].apply(cv, src, { ...defaultParams(HOST[id]), mode: id }, {});
      const direct = OPS[id].apply(cv, src, {}, {});
      try { sameBytes(viaNode, direct, `${id} through FILTERS.${HOST[id]}`); } finally { viaNode.delete(); direct.delete(); }
    }
  } finally { g.delete(); c.delete(); }
  const pts = OPS.roi_polygon.params.filter((p) => /^p\d[xy]$/.test(p.key));
  assert.deepEqual(pts.map((p) => p.default), [30, 20, 70, 20, 90, 85, 10, 85, 5, 55, 15, 30], 'spec 4.3 default points');
  for (const k of [4, 5, 6]) assert.match(pts.find((p) => p.key === `p${k}x`).label, new RegExp(`ใช้เมื่อจำนวนจุด ≥ ${k}`));
  assert.deepEqual(OPS.line_profile.params.find((p) => p.key === 'view').options.map((o) => o.value), ['overlay', 'plot']);
  assert.deepEqual(OPS.roi_polygon.params.find((p) => p.key === 'view').options.map((o) => o.value), ['apply', 'mask', 'overlay']);
});

test('ipD-20. palette-panel.js: ANALYSIS & DETECTION has feret, line_profile right after regionprops; GEOMETRY = crop, roi_polygon, resize, rotate, flip, perspective_warp, undistort, translate_shear; OP_TO_FILTER resolves each new op to its host node; each new tile is in exactly one section; the 3 glyphs (⌀ 〽 ⬠) differ from every other OP_ICON, every GROUP_ICON and the 📊/📌 markers, and appear nowhere in app.js / detail-panel.js', () => {
  const ana = TOOL_SECTIONS.find((s) => s.title === 'ANALYSIS & DETECTION').ops;
  assert.deepEqual(ana.slice(ana.indexOf('regionprops'), ana.indexOf('regionprops') + 3), ['regionprops', 'feret', 'line_profile']);
  assert.deepEqual(TOOL_SECTIONS.find((s) => s.title === 'GEOMETRY').ops, ['crop', 'roi_polygon', 'resize', 'rotate', 'flip', 'perspective_warp', 'undistort', 'translate_shear']);
  for (const id of NEW_OPS) {
    assert.equal(OP_TO_FILTER[id], HOST[id]);
    assert.equal(TOOL_SECTIONS.filter((s) => s.ops.includes(id)).length, 1);
  }
  const src = readSrc('ui/palette-panel.js');
  const body = src.slice(src.indexOf('const OP_ICON = {'), src.indexOf('};', src.indexOf('const OP_ICON = {')));
  const icons = {};
  for (const m of body.matchAll(/([a-z_0-9]+): "([^"]+)"/g)) icons[m[1]] = m[2];
  assert.deepEqual([icons.feret, icons.line_profile, icons.roi_polygon], ['⌀', '〽', '⬠']);
  const groupGlyphs = new Set(Object.values(GROUP_ICON));
  const app = readSrc('app.js'), dp = readSrc('ui/detail-panel.js');
  for (const id of NEW_OPS) {
    const others = Object.entries(icons).filter(([k]) => k !== id).map(([, v]) => v);
    assert.ok(!others.includes(icons[id]), `${id} collides with another OP_ICON`);
    assert.ok(!groupGlyphs.has(icons[id]), `${id} collides with a GROUP_ICON`);
    assert.ok(!['📊', '📌'].includes(icons[id]));
    assert.ok(!app.includes(icons[id]) && !dp.includes(icons[id]), `${id} glyph used by app.js / detail-panel.js`);
  }
});

test('ipD-21. ui-meta.js OP_PRIMARY uses the clamped values the ops really use (pnum): "Feret · พื้นที่ขั้นต่ำ 50px · สูงสุด 50 ชิ้น", "เส้น (10%,50%) → (90%,50%)" (+ " · กราฟเต็มภาพ" for the plot view), "ROI 4 จุด · ตัดนอกกรอบ" / "· Mask" / "· ดูตำแหน่ง"; out-of-range / garbage / "constructor" values show what the op does', () => {
  assert.equal(OP_PRIMARY.feret({}), 'Feret · พื้นที่ขั้นต่ำ 50px · สูงสุด 50 ชิ้น');
  assert.equal(OP_PRIMARY.feret({ minArea: -4, maxObjects: 9999 }), 'Feret · พื้นที่ขั้นต่ำ 0px · สูงสุด 500 ชิ้น');
  assert.equal(OP_PRIMARY.feret({ minArea: 'x', maxObjects: '2.6' }), 'Feret · พื้นที่ขั้นต่ำ 50px · สูงสุด 3 ชิ้น');
  assert.equal(OP_PRIMARY.line_profile({}), 'เส้น (10%,50%) → (90%,50%)');
  assert.equal(OP_PRIMARY.line_profile({ x1: 0, y1: 20, x2: 150, y2: 'x', view: 'plot' }), 'เส้น (0%,20%) → (100%,50%) · กราฟเต็มภาพ');
  assert.equal(OP_PRIMARY.line_profile({ view: 'constructor' }), 'เส้น (10%,50%) → (90%,50%)');
  assert.equal(OP_PRIMARY.roi_polygon({}), 'ROI 4 จุด · ตัดนอกกรอบ');
  assert.equal(OP_PRIMARY.roi_polygon({ points: 9, view: 'mask' }), 'ROI 6 จุด · Mask');
  assert.equal(OP_PRIMARY.roi_polygon({ points: '5.4', view: 'overlay' }), 'ROI 5 จุด · ดูตำแหน่ง');
  assert.equal(OP_PRIMARY.roi_polygon({ points: 2, view: 'constructor' }), 'ROI 3 จุด · ตัดนอกกรอบ');
  const meta = readSrc('filters/ui-meta.js');
  for (const id of NEW_OPS) {
    const start = meta.indexOf(`\n  ${id}: `);
    assert.ok(start > 0);
    const next = /\n {2}(\/\/|[a-z_0-9]+: )/g;
    next.lastIndex = start + 1;
    const entry = meta.slice(start, next.exec(meta).index);
    assert.match(entry, /pnum\(/, `${id} card uses pnum`);
  }
});

// [opId, key, registry pattern, default, min, max]
test('ipD-22. (static read) every numeric param of the 3 ops is clamped to EXACTLY the UI default/min/max in registry.js (numIn/intIn, pctToPx = numIn(pct, d, 0, 100)) and in codegen.js (clampNum); P1..P6 defaults come from one table on each side (ROI_DEFAULTS builds the params; codegen reads OPS defaults); every numeric param is covered', () => {
  const reg = readSrc('filters/registry.js'), gen = readSrc('codegen.js');
  const block = (source, opId) => {
    const start = source.indexOf(`\n  ${opId}: {`);
    assert.ok(start >= 0, `no "${opId}: {" block`);
    const next = /\n {2}[a-z_0-9]+: \{/g;
    next.lastIndex = start + 1;
    const m = next.exec(source);
    return source.slice(start, m ? m.index : source.length);
  };
  const spec = (id, k) => { const s = OPS[id].params.find((p) => p.key === k); return [s.default, s.min, s.max]; };
  const num = '(-?[\\d.]+)';
  const covered = new Set();
  const regCheck = (id, k, re, vals) => { const m = block(reg, id).match(re); assert.ok(m, `registry ${id}.${k}: ${re}`); assert.deepEqual(m.slice(1).map(Number), vals, `registry ${id}.${k}`); };
  const genCheck = (id, k, want) => { const m = block(gen, id).match(new RegExp(`clampNum\\(p\\.${k}, ${num}, ${num}, ${num}\\)`)); assert.ok(m, `codegen ${id}.${k}`); assert.deepEqual(m.slice(1).map(Number), want, `codegen ${id}.${k}`); };
  regCheck('feret', 'minArea', new RegExp(`numIn\\(p\\.minArea, ${num}, ${num}, ${num}\\)`), spec('feret', 'minArea'));
  regCheck('feret', 'maxObjects', new RegExp(`intIn\\(p\\.maxObjects, ${num}, ${num}, ${num}\\)`), spec('feret', 'maxObjects'));
  genCheck('feret', 'minArea', spec('feret', 'minArea'));
  genCheck('feret', 'maxObjects', spec('feret', 'maxObjects'));
  assert.match(block(gen, 'feret'), /Math\.round\(clampNum\(p\.maxObjects/);
  covered.add('feret.minArea').add('feret.maxObjects');
  assert.match(reg, /function pctToPx\(pct, d, n\) \{\n {2}return Math\.floor\(\(numIn\(pct, d, 0, 100\) \/ 100\) \* \(n - 1\) \+ 0\.5\);/);
  for (const k of ['x1', 'y1', 'x2', 'y2']) {
    const [d, lo, hi] = spec('line_profile', k);
    assert.deepEqual([lo, hi], [0, 100]);
    regCheck('line_profile', k, new RegExp(`pctToPx\\(p\\.${k}, ${num}, [WH]\\)`), [d]);
    genCheck('line_profile', k, [d, 0, 100]);
    covered.add(`line_profile.${k}`);
  }
  regCheck('roi_polygon', 'points', new RegExp(`intIn\\(p\\.points, ${num}, ${num}, ${num}\\)`), spec('roi_polygon', 'points'));
  genCheck('roi_polygon', 'points', spec('roi_polygon', 'points'));
  assert.match(block(gen, 'roi_polygon'), /Math\.round\(clampNum\(p\.points/);
  covered.add('roi_polygon.points');
  assert.match(block(reg, 'roi_polygon'), /pctToPx\(p\[`p\$\{i \+ 1\}x`\], ROI_DEFAULTS\[i\]\[0\], W\)/);
  assert.match(block(reg, 'roi_polygon'), /pctToPx\(p\[`p\$\{i \+ 1\}y`\], ROI_DEFAULTS\[i\]\[1\], H\)/);
  assert.match(block(gen, 'roi_polygon'), /clampNum\(p\[`p\$\{i\}x`\], paramDefault\("roi_polygon", `p\$\{i\}x`\), 0, 100\)/);
  assert.match(block(gen, 'roi_polygon'), /clampNum\(p\[`p\$\{i\}y`\], paramDefault\("roi_polygon", `p\$\{i\}y`\), 0, 100\)/);
  for (const p of OPS.roi_polygon.params) if (/^p\d[xy]$/.test(p.key)) { assert.deepEqual([p.min, p.max], [0, 100]); covered.add(`roi_polygon.${p.key}`); }
  assert.match(reg, /const ROI_DEFAULTS = \[\[30, 20\], \[70, 20\], \[90, 85\], \[10, 85\], \[5, 55\], \[15, 30\]\];/);
  for (const id of NEW_OPS) for (const p of OPS[id].params) if (p.type === 'number') assert.ok(covered.has(`${id}.${p.key}`), `${id}.${p.key} not covered`);
});

const PY_VARIANTS = [
  ['feret', {}], ['feret', { minArea: 0, maxObjects: 3 }],
  ['line_profile', {}], ['line_profile', { view: 'plot' }], ['line_profile', { x1: 50, y1: 50, x2: 50, y2: 50 }],
  ['roi_polygon', {}], ['roi_polygon', { view: 'mask', points: 6 }], ['roi_polygon', { view: 'overlay', points: 3 }],
];

test('ipD-23. codegen: all three export real single-image Python (never "WARNING: skipped" / "ข้าม" / round( / ** 0.5 / import math), embed the clamped values, use int(pct / 100.0 * (n - 1) + 0.5) for % -> px, np.sqrt for every root, the regionprops sort key / candidate lines character for character (no edit of GEN.regionprops), and keep the gray/colour state (feret out colour; line_profile colour; roi_polygon mask -> gray, overlay -> colour, apply -> same)', () => {
  for (const [id, p] of PY_VARIANTS) {
    const code = buildPythonCode([OPSTEP(id, p)]);
    for (const bad of ['WARNING: skipped', 'ข้าม', 'round(', '** 0.5', 'import math', 'np.round', 'math.']) assert.ok(!code.includes(bad), `${id} ${JSON.stringify(p)}: contains ${bad}`);
    assert.match(code, new RegExp(`# ── ขั้นที่ 1: ${OPS[id].name.replace(/[()]/g, '\\$&')} \\(${id}\\) ──`));
  }
  const lp = buildPythonCode([OPSTEP('line_profile', { x1: -3, y1: 33.3, x2: 250, y2: 'x' })]);
  assert.match(lp, /x1, y1 = int\(0 \/ 100\.0 \* \(w - 1\) \+ 0\.5\), int\(33\.3 \/ 100\.0 \* \(h - 1\) \+ 0\.5\)/);
  assert.match(lp, /x2, y2 = int\(100 \/ 100\.0 \* \(w - 1\) \+ 0\.5\), int\(50 \/ 100\.0 \* \(h - 1\) \+ 0\.5\)/);
  const roi = buildPythonCode([OPSTEP('roi_polygon', { points: 9, p1x: -1, p6y: 'x' })]);
  assert.match(roi, /pcts = \[\(0, 20\), \(70, 20\), \(90, 85\), \(10, 85\), \(5, 55\), \(15, 30\)\]/);
  assert.match(buildPythonCode([OPSTEP('feret', { minArea: -5, maxObjects: 0 })]), /if area < 0:[\s\S]*max_objects = 1\n/);
  // คีย์เรียง + บรรทัดคัดวัตถุ = GEN.regionprops ทุกตัวอักษร
  const cand = (code) => { const a = code.indexOf('    cands = []'); return code.slice(a, code.indexOf('    total_found = len(cands)', a)); };
  const rp = buildPythonCode([STEP('shape_analysis', 'regionprops', { regionprops__minArea: 50 })]);
  const fr = buildPythonCode([OPSTEP('feret', { minArea: 50 })]);
  assert.ok(cand(rp).length > 200);
  assert.equal(cand(fr), cand(rp), 'candidate selection + sort key identical to regionprops');
  assert.match(cand(fr), /cands\.append\(\(\(-area, y, x, cx, cy, tx\), c, area, cx, cy\)\)/);
  // ctx.gray ต่อจากแต่ละ view: needs:"gray" ขั้นถัดไปแทรก BGR2GRAY เฉพาะตอนภาพเป็นสี
  const after = (opId, p) => buildPythonCode([OPSTEP(opId, p), STEP('shape_analysis', 'regionprops')]).includes('COLOR_BGR2GRAY)  # (แปลงขาวดำอัตโนมัติ)');
  assert.equal(after('feret', {}), true, 'feret output is colour');
  assert.equal(after('line_profile', {}), true);
  assert.equal(after('roi_polygon', { view: 'overlay' }), true);
  assert.equal(after('roi_polygon', { view: 'mask' }), false, 'mask is gray');
  assert.equal(after('roi_polygon', { view: 'apply' }), true, 'apply keeps colour');
  assert.equal(buildPythonCode([STEP('color_space', 'grayscale'), OPSTEP('roi_polygon', {}), STEP('shape_analysis', 'regionprops')]).split('(แปลงขาวดำอัตโนมัติ)').length - 1, 0, 'apply keeps gray');
});

test('ipD-24. teaching text: names / descs / labels / hints / options / every info message of the 3 ops, the exported Python of every variant, and the new VAL_META lines never say "ที่สุด"; each desc states the spec 4 points (feret 1-6, line_profile 1-5, roi_polygon); VAL_META has th for all 6 feret columns', () => {
  const texts = [];
  for (const id of NEW_OPS) {
    const o = OPS[id];
    texts.push(o.name, o.desc);
    for (const p of o.params) { texts.push(p.label, p.desc || '', p.hint || ''); for (const op of p.options || []) texts.push(op.label, op.hint || ''); }
  }
  const b = blobsGray(), s = sceneGray(80, 80), sm = sceneGray(20, 20), black = gray(10, 10, () => 0), empty = new cv.Mat();
  try {
    for (const [id, src, p] of [['feret', b, {}], ['feret', b, { minArea: 0, maxObjects: 1 }], ['feret', black, {}], ['feret', empty, {}], ['line_profile', s, {}], ['line_profile', sm, { x1: 5, x2: 5, y1: 5, y2: 5 }],
      ['roi_polygon', s, {}], ['roi_polygon', s, { p1x: 10, p1y: 10, p2x: 90, p2y: 90, p3x: 90, p3y: 10, p4x: 10, p4y: 90 }], ['roi_polygon', s, { points: 3, p1x: 1, p1y: 1, p2x: 2, p2y: 2, p3x: 3, p3y: 3 }]]) texts.push(runValues(id, src, p).info);
  } finally { for (const m of [b, s, sm, black, empty]) m.delete(); }
  for (const [id, p] of PY_VARIANTS) texts.push(buildPythonCode([OPSTEP(id, p)]));
  const dp = readSrc('ui/detail-panel.js');
  const vm = dp.slice(dp.indexOf('const VAL_META = {'), dp.indexOf('};', dp.indexOf('const VAL_META = {')));
  texts.push(vm);
  for (const t of texts) assert.ok(!t.includes('ที่สุด'), `"ที่สุด" in: ${t.slice(0, 120)}`);
  const fd = OPS.feret.desc, ld = OPS.line_profile.desc, rd = OPS.roi_polygon.desc;
  for (const re of [/คาลิปเปอร์/, /Max Feret/, /Min Feret/, /ไม่ขึ้นกับการหมุน/, /Bounding Box/, /ขอบพิกเซล/, /10 px วัดได้ 10/, /Min = 1 และ Max = 1\.41/, /เส้นทแยงมุม/, /ยาวตามกรอบแคบสุด/, /มีรู/, /convex hull/, /Region Properties/, /bwferet/, /ไม่รับประกัน/]) assert.match(fd, re, `feret desc ${re}`);
  for (const re of [/แกนนอน = ระยะจาก A/, /แกนตั้ง = ความสว่าง 0–255/, /กราฟจะแบน/, /กระโดด/, /noise ทำให้กราฟหยัก/, /ภาพสี/, /เทา/, /nearest neighbour/, /improfile/, /ระยะระหว่างขอบขึ้นกับขอบลง/, /ตั้งฉาก/, /เฉียง/, /%/, /ลากเส้นบนภาพโดยตรงไม่ได้/]) assert.match(ld, re, `line_profile desc ${re}`);
  for (const re of [/ดูเฉพาะบริเวณนี้/, /Image Arithmetic → Mask/, /roipoly/, /%/, /คลิกวาด/]) assert.match(rd, re, `roi desc ${re}`);
  for (const k of ['feret_max', 'feret_min', 'feret_len', 'feret_aspect', 'feret_max_deg', 'feret_min_deg']) assert.match(vm, new RegExp(`\\n {2}${k}: +\\{ th: "`), `VAL_META.${k} has th`);
  assert.match(vm, /feret_aspect: +\{ th: "ยาว÷แคบ", +d: 2, w: 70, pri: true, +tip: "1 = กว้างเท่ากันทุกทิศ · ยิ่งมากยิ่งยาวรี" \}/);
});

test('ipD-25. (static) channel order of the exported Python follows ctx.mirrored through isMirrored only: plain colour input -> line_profile / roi_polygon convert with COLOR_BGR2GRAY and draw reversed tuples (yellow (0, 255, 255)); right after Color Space (and through a colour -> colour step after it) -> COLOR_RGB2GRAY and app-order tuples (255, 255, 0); a gray step before -> mirrored again; feret always draws reversed tuples (its auto BGR2GRAY makes the image BGR again)', () => {
  const lpLine = (code) => code.split('\n').find((l) => l.includes('gray = img if img.ndim == 2 else cv2.cvtColor(img,'));
  const has = (code, s) => code.includes(s);
  const plain = buildPythonCode([OPSTEP('line_profile')]);
  assert.match(lpLine(plain), /COLOR_BGR2GRAY/);
  assert.ok(has(plain, 'cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 1)'));
  assert.ok(has(plain, 'cv2.circle(out, (x2, y2), 3, (0, 0, 255), -1)'), 'B red reversed');
  for (const pre of [[CS('hsv')], [CS('lab'), STEP('blur', 'gaussian_blur')]]) {
    const code = buildPythonCode([...pre, OPSTEP('line_profile')]);
    assert.match(lpLine(code), /COLOR_RGB2GRAY/);
    assert.ok(has(code, 'cv2.line(out, (x1, y1), (x2, y2), (255, 255, 0), 1)'));
    assert.ok(has(code, 'cv2.circle(out, (x2, y2), 3, (255, 0, 0), -1)'));
    const r = buildPythonCode([...pre, OPSTEP('roi_polygon', { view: 'overlay' })]);
    assert.ok(has(r, 'gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)'));
    assert.ok(has(r, 'True, (255, 255, 0), 2)'));
  }
  const gthen = buildPythonCode([STEP('color_space', 'grayscale'), OPSTEP('line_profile')]);
  assert.ok(has(gthen, '(0, 255, 255), 1)'));
  const rplain = buildPythonCode([OPSTEP('roi_polygon', { view: 'overlay' })]);
  assert.ok(has(rplain, 'cv2.COLOR_BGR2GRAY)') && has(rplain, 'True, (0, 255, 255), 2)'));
  for (const pre of [[], [CS('hsv')]]) {
    const f = buildPythonCode([...pre, OPSTEP('feret')]);
    assert.ok(has(f, 'cv2.line(img, box[q], box[(q + 1) % 4], (0, 255, 0), 1)') && has(f, '(hull[bj][1] + 1) // 2), (0, 0, 255), 1)') && has(f, '0.5, (0, 255, 255), 1)'), `feret after ${pre.length ? 'hsv' : 'nothing'}: reversed tuples`);
  }
});

// ===================================================================
// ipD-P1…P4 — parity กับ Python จริง (cv2 5.0 / numpy 2.x)
// ===================================================================
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'ip-batch2-D-2026-10-03-'));
process.on('exit', () => {
  try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ }
});
function pythonAvailable() {
  try { execFileSync('python', ['-c', 'import cv2, numpy'], { encoding: 'utf8' }); return true; } catch { return false; }
}
const HAS_PYTHON = pythonAvailable();
const PY_SKIP = { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' };
function swapRB3(buf) {
  const out = Buffer.from(buf);
  for (let i = 0; i < out.length; i += 3) { const t = out[i]; out[i] = out[i + 2]; out[i + 2] = t; }
  return out;
}
// ภาพสีของแอป (RGB) ส่งให้ Python เป็น BGR · ผลสีกลับเป็น RGB ก่อนเทียบ (ยกเว้นหลัง Color Space — swapOut false)
function runPython(code, src, { swapOut = true, extra = '' } = {}) {
  assert.equal(src.channels(), 3, 'cv2.imread gives 3 channels');
  let patched = code.replace('img = cv2.imread(path)', `img = np.fromfile(path, dtype=np.uint8).reshape((${src.rows}, ${src.cols}, 3))`);
  if (extra) patched = patched.replace('    return img\n\n\nif __name__', `${extra}\n    return img\n\n\nif __name__`);
  const mainIdx = patched.indexOf('if __name__ == "__main__":');
  assert.ok(mainIdx > 0);
  patched = patched.slice(0, mainIdx) + '\nimport sys\nresult = process(sys.argv[1])\nprint("NDIM", result.ndim)\nresult.tofile(sys.argv[2])\n';
  const id = Math.random().toString(36).slice(2);
  const scriptPath = path.join(TMP_DIR, `s_${id}.py`), inPath = path.join(TMP_DIR, `i_${id}.bin`), outPath = path.join(TMP_DIR, `o_${id}.bin`);
  fs.writeFileSync(scriptPath, patched, 'utf8');
  fs.writeFileSync(inPath, swapRB3(Buffer.from(src.data)));
  const stdout = execFileSync('python', [scriptPath, inPath, outPath], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
  const ndim = Number((stdout.match(/NDIM (\d)/) || [])[1]);
  const raw = fs.readFileSync(outPath);
  for (const f of [scriptPath, inPath, outPath]) fs.rmSync(f, { force: true });
  return { stdout, ndim, bytes: ndim === 3 && swapOut ? swapRB3(raw) : Buffer.from(raw) };
}
// รัน pipeline แบบ worker (needs "gray" → RGB2GRAY อัตโนมัติ) — คืนผลสุดท้าย + ctx ของขั้นสุดท้าย
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
// เทียบภาพทุก byte — mask(x, y) = true คือข้าม (กล่องข้อความ)
function diffImages(appMat, pyBytes, skip = () => false) {
  const ch = appMat.channels(), w = appMat.cols, d = appMat.data;
  assert.equal(pyBytes.length, d.length, 'output size');
  let ndiff = 0, inSkip = 0;
  for (let i = 0; i < d.length; i++) {
    if (d[i] === pyBytes[i]) continue;
    const p = Math.floor(i / ch);
    if (skip(p % w, Math.floor(p / w))) { inSkip++; continue; }
    ndiff++;
  }
  return { ndiff, inSkip };
}
function parsePyValues(stdout, keys) {
  const out = {};
  for (const k of keys) {
    const m = stdout.match(new RegExp(`^${k} = (.*)$`, 'm'));
    assert.ok(m, `python printed ${k}`);
    const raw = m[1].trim();
    out[k] = raw === 'None' ? null : raw === 'True' ? true : raw === 'False' ? false : Number(raw);
  }
  return out;
}
// กล่องข้อความ (สเปก §7): anchor (ax, ay) → [ax−3, ay−14] .. [ax+3+12·len, ay+4]
function labelBoxes(anchors) {
  return (x, y) => anchors.some(([ax, ay, len]) => x >= ax - 3 && x <= ax + 3 + 12 * len && y >= ay - 14 && y <= ay + 4);
}
const FERET_DUMP = '    import json\n    print("FERET_JSON", json.dumps(feret_rows))';

function feretParity(src, p, label, pre = []) {
  const pipe = [...pre, OPSTEP('feret', p)];
  const py = runPython(buildPythonCode(pipe), src, { extra: FERET_DUMP });
  const app = appRun(pipe, src);
  try {
    const pyRows = JSON.parse(py.stdout.match(/^FERET_JSON (.*)$/m)[1]);
    const v = app.ctx.values;
    assert.equal(pyRows.length, v.rows.length, `${label}: row count`);
    assert.ok(py.stdout.split(/\r?\n/).includes(app.ctx.info), `${label}: Python prints the app info line ("${app.ctx.info}")`);
    for (let i = 0; i < v.rows.length; i++) {
      for (let k = 0; k < 8; k++) assert.equal(pyRows[i][k], v.rows[i][k], `${label} row ${i + 1} ${FERET_COLUMNS[k]}`);
      for (let k = 8; k < 10; k++) close(pyRows[i][k], v.rows[i][k], 1e-9, `${label} row ${i + 1} ${FERET_COLUMNS[k]}`);
    }
    // กล่องเลขวัตถุจากคอลัมน์ bbox ของ Region Properties (minArea/cap เดียวกัน — ยืนยันลำดับไปด้วย)
    const rpPipe = [...pre, STEP('shape_analysis', 'regionprops', { regionprops__minArea: p.minArea ?? 50, regionprops__maxObjects: p.maxObjects ?? 50 })];
    const rp = appRun(rpPipe, src);
    let anchors;
    try {
      const rr = rowsAsObjects(rp.ctx.values);
      assert.deepEqual(rr.map((r) => [r.area, r.cx, r.cy]), rowsAsObjects(v).map((r) => [r.area, r.cx, r.cy]), `${label}: same objects/order as regionprops`);
      anchors = rr.map((r) => [r.bbox_x, Math.max(10, r.bbox_y - 5), String(r.index).length]);
    } finally { rp.mat.delete(); }
    const d = diffImages(app.mat, py.bytes, labelBoxes(anchors));
    assert.equal(d.ndiff, 0, `${label}: ${d.ndiff} bytes differ outside the label boxes`);
    return { rows: v.rows.length, inSkip: d.inSkip };
  } finally { app.mat.delete(); }
}

test('ipD-P1. codegen parity feret (real Python): index / area / cx / cy / max / min / len / aspect EXACTLY equal, angles within 1e-9 deg, the info line identical, and the image ndiff 0 outside the object-label boxes (anchored at the Region Properties bbox of the same minArea/cap, which also pins the ordering) -- blobs, rotated bars, 1-px lines, noise with cap 40, gray and colour, minArea 0 / 50', PY_SKIP, () => {
  const rotBars = fillPolys(160, 120, [[[20, 30], [70, 10], [75, 22], [25, 42]], [[100, 20], [110, 15], [150, 95], [140, 100]], [[30, 80], [60, 70], [62, 76], [32, 86]]], 3);
  const lines = rgb(90, 60, (x, y) => ((y === 10 && x > 5 && x < 80) || (x === 40 && y > 15 && y < 55) || (x === y + 20 && y > 20 && y < 50) || (x === 5 && y === 50) ? [255, 255, 255] : [0, 0, 0]));
  const blobsC = rgb(140, 100, (x, y) => (((x - 30) ** 2 + (y - 30) ** 2 < 300) || ((x - 90) ** 2 / 4 + (y - 60) ** 2 < 200) || (x > 110 && y < 20) ? [220, 90, 160] : [10, 40, 20]));
  const noise = noiseRGB(320, 240, 9);
  const blobs = toRGB(blobsGray());
  try {
    let total = 0;
    for (const [src, p, label] of [[blobs, { minArea: 0 }, 'blobs gray'], [blobs, {}, 'blobs minArea 50'], [rotBars, { minArea: 0 }, 'rotated bars'], [lines, { minArea: 0 }, '1-px lines'],
      [blobsC, {}, 'colour blobs'], [noise, { minArea: 0, maxObjects: 40 }, 'noise cap 40']]) {
      total += feretParity(src, p, label).rows;
    }
    assert.ok(total >= 55, `rows compared: ${total}`);
  } finally { for (const m of [rotBars, lines, blobsC, noise, blobs]) m.delete(); }
});

function lineProfileParity(src, p, label, { pre = [], mirroredOut = true } = {}) {
  const pipe = [...pre, OPSTEP('line_profile', p)];
  const py = runPython(buildPythonCode(pipe), src, { swapOut: mirroredOut });
  const app = appRun(pipe, src);
  try {
    const d = diffImages(app.mat, py.bytes);
    assert.equal(d.ndiff, 0, `${label}: image ndiff ${d.ndiff}`);
    assert.deepEqual(parsePyValues(py.stdout, LP_KEYS), scalarMap(app.ctx), `${label}: 9 values`);
    assert.ok(py.stdout.split(/\r?\n/).includes(app.ctx.info), `${label}: info line`);
    return { stdout: py.stdout, ctx: app.ctx };
  } finally { app.mat.delete(); }
}

test('ipD-P2. codegen parity line_profile (real Python): ndiff 0 over the WHOLE image in both views, the 9 values and the info line identical, and the full printed profile = the reference sampler -- horizontal / vertical / diagonal / reversed / A = B / < 64 px / band at the top, gray (after Grayscale) and colour, 97x61 / 320x240 / 700x500', PY_SKIP, () => {
  // 64x64 pins the boundary of the Python 'small' test (< 64, not <= 64) and 100x70 pins int(0.35 * h) = 24 (24.5 -> floor)
  const imgs = [sceneRGB(97, 61), sceneRGB(320, 240, 8), noiseRGB(700, 500, 3), sceneRGB(60, 40, 2), sceneRGB(64, 64, 4), sceneRGB(100, 70, 6)];
  try {
    const lines = [
      [{}, 'default'], [{ x1: 50, y1: 0, x2: 50, y2: 100 }, 'vertical'], [{ x1: 3, y1: 7, x2: 96, y2: 88 }, 'diagonal'], [{ x1: 90, y1: 80, x2: 12.5, y2: 5 }, 'reversed diagonal'],
      [{ x1: 40, y1: 40, x2: 40, y2: 40 }, 'A = B'], [{ x1: 5, y1: 95, x2: 95, y2: 99 }, 'band at top'], [{ x1: 0, y1: 0, x2: 100, y2: 100 }, 'corner to corner'],
    ];
    let n = 0;
    for (const src of imgs) for (const [p, label] of lines) for (const view of ['overlay', 'plot']) {
      if (src.cols === 700 && label !== 'diagonal' && label !== 'band at top') continue;
      const r = lineProfileParity(src, { ...p, view }, `${label} ${view} ${src.cols}x${src.rows}`);
      const prof = JSON.parse(r.stdout.match(/^profile = (.*)$/m)[1]);
      const W = src.cols, H = src.rows;
      const q = (v, d, n2) => Math.floor((Math.min(100, Math.max(0, v ?? d)) / 100) * (n2 - 1) + 0.5);
      assert.deepEqual(prof, refProfile(src, q(p.x1, 10, W), q(p.y1, 50, H), q(p.x2, 90, W), q(p.y2, 50, H)), `${label} ${view}: full profile`);
      n++;
    }
    for (const view of ['overlay', 'plot']) lineProfileParity(imgs[0], { view, x1: 10, y1: 20, x2: 80, y2: 70 }, `gray ${view}`, { pre: [STEP('color_space', 'grayscale')] });
    assert.ok(n >= 30);
    // flat line (every sample equal): no edge, no magenta, Python prints the same info / None values
    const flatG = gray(120, 100, () => 120), flat = toRGB(flatG);
    try { for (const view of ['overlay', 'plot']) lineProfileParity(flat, { view }, `flat ${view}`); } finally { flatG.delete(); flat.delete(); }
  } finally { for (const m of imgs) m.delete(); }
});

function roiParity(src, p, label, { pre = [], mirroredOut = true } = {}) {
  const pipe = [...pre, OPSTEP('roi_polygon', p)];
  const py = runPython(buildPythonCode(pipe), src, { swapOut: mirroredOut });
  const app = appRun(pipe, src);
  try {
    let skip = () => false;
    if (p.view === 'overlay') {
      const n = Math.round(Math.min(6, Math.max(3, Number(p.points ?? 4))));
      const anchors = [];
      for (let i = 1; i <= n; i++) {
        const vx = Math.floor((Math.min(100, Math.max(0, p[`p${i}x`] ?? OPS.roi_polygon.params.find((s) => s.key === `p${i}x`).default)) / 100) * (src.cols - 1) + 0.5);
        const vy = Math.floor((Math.min(100, Math.max(0, p[`p${i}y`] ?? OPS.roi_polygon.params.find((s) => s.key === `p${i}y`).default)) / 100) * (src.rows - 1) + 0.5);
        anchors.push([vx + 5, Math.max(10, vy - 5), String(i).length]);
      }
      skip = labelBoxes(anchors);
    }
    const d = diffImages(app.mat, py.bytes, skip);
    assert.equal(d.ndiff, 0, `${label}: image ndiff ${d.ndiff}`);
    assert.equal(py.ndim === 3 ? 3 : 1, app.mat.channels(), `${label}: channels`);
    assert.deepEqual(parsePyValues(py.stdout, ROI_KEYS), scalarMap(app.ctx), `${label}: 7 values`);
    assert.ok(py.stdout.split(/\r?\n/).includes(app.ctx.info), `${label}: info line`);
  } finally { app.mat.delete(); }
}

test('ipD-P3. codegen parity roi_polygon (real Python): apply / mask ndiff 0, overlay ndiff 0 outside the vertex-label boxes, the 7 values and the info line identical -- n 3..6, bowtie, collinear, stacked, gray (after Grayscale) and colour, 97x61 and 700x500', PY_SKIP, () => {
  const imgs = [sceneRGB(97, 61), noiseRGB(700, 500, 4)];
  try {
    for (const src of imgs) {
      for (const view of ['apply', 'mask', 'overlay']) {
        for (let n = 3; n <= 6; n++) roiParity(src, { points: n, view }, `n ${n} ${view} ${src.cols}x${src.rows}`);
        roiParity(src, { view, p1x: 10, p1y: 10, p2x: 90, p2y: 90, p3x: 90, p3y: 10, p4x: 10, p4y: 90 }, `bowtie ${view}`);
        roiParity(src, { view, points: 3, p1x: 10, p1y: 10, p2x: 50, p2y: 50, p3x: 90, p3y: 90 }, `collinear ${view}`);
        roiParity(src, { view, points: 5, p1x: 40, p1y: 40, p2x: 40, p2y: 40, p3x: 40, p3y: 40, p4x: 40, p4y: 40, p5x: 40, p5y: 40 }, `stacked ${view}`);
        // MINOR-1: repeated points (zero-length edges) -> crossed false / not degenerate in Python as in the app; concave non-crossing; vertex on an edge
        roiParity(src, { view, points: 4, p1x: 30, p1y: 20, p2x: 70, p2y: 20, p3x: 90, p3y: 85, p4x: 90, p4y: 85 }, `P4 = P3 ${view}`);
        roiParity(src, { view, points: 5, p1x: 10, p1y: 10, p2x: 10, p2y: 10, p3x: 80, p3y: 10, p4x: 80, p4y: 60, p5x: 10, p5y: 60 }, `P1 = P2 ${view}`); // axis-aligned: the polygon raster of cv2 5.0 and opencv.js 4.10 differs by 1-3 px on some slanted triangles (not related to the duplicate)
        roiParity(src, { view, points: 4, p1x: 10, p1y: 10, p2x: 50, p2y: 10, p3x: 50, p3y: 50, p4x: 10, p4y: 10 }, `P4 = P1 ${view}`);
        roiParity(src, { view, points: 4, p1x: 10, p1y: 80, p2x: 50, p2y: 20, p3x: 90, p3y: 80, p4x: 50, p4y: 50 }, `chevron ${view}`);
        roiParity(src, { view, points: 5, p1x: 10, p1y: 10, p2x: 80, p2y: 10, p3x: 80, p3y: 60, p4x: 50, p4y: 60, p5x: 50, p5y: 10 }, `vertex on edge ${view}`); // rectilinear (raster agrees)
      }
    }
    for (const view of ['apply', 'mask', 'overlay']) roiParity(imgs[0], { view, points: 6 }, `gray ${view}`, { pre: [STEP('color_space', 'grayscale')] });
    // vertex on the middle of a non-adjacent edge: every cyclic shift and both directions (each clause of the touch test is the deciding one in some variant)
    const tee = [[10, 10], [80, 10], [80, 60], [50, 60], [50, 10]];
    for (let sh = 0; sh < tee.length; sh++) for (const rev of [false, true]) {
      const v = tee.map((_, i) => tee[(i + sh) % tee.length]);
      if (rev) v.reverse();
      const p = { view: 'mask', points: v.length };
      v.forEach(([x, y], i) => { p[`p${i + 1}x`] = x; p[`p${i + 1}y`] = y; });
      roiParity(imgs[0], p, `tee shift ${sh}${rev ? ' reversed' : ''}`);
    }
  } finally { for (const m of imgs) m.delete(); }
});

test('ipD-P4. codegen parity AFTER a Color Space (HSV / Lab) step (isMirrored): line_profile overlay + plot and roi_polygon apply / mask / overlay give ndiff 0 (no channel swap of the output) with the same values / info -- the internal colour -> gray uses the app channel order (RGB2GRAY on H,S,V) and drawn colours are not reversed; HSV -> Gaussian Blur -> line_profile keeps that; the same chains without Color Space stay at ndiff 0 with the BGR order', PY_SKIP, () => {
  const src = sceneRGB(97, 61), big = noiseRGB(320, 240, 12);
  try {
    for (const s of [src, big]) for (const space of ['hsv', 'lab']) {
      for (const view of ['overlay', 'plot']) lineProfileParity(s, { view, x1: 5, y1: 30, x2: 95, y2: 70 }, `${space} -> line_profile ${view}`, { pre: [CS(space)], mirroredOut: false });
      for (const view of ['apply', 'mask', 'overlay']) roiParity(s, { view, points: 5 }, `${space} -> roi ${view}`, { pre: [CS(space)], mirroredOut: false });
    }
    lineProfileParity(src, {}, 'hsv -> blur -> line_profile', { pre: [CS('hsv'), STEP('blur', 'gaussian_blur')], mirroredOut: false });
    lineProfileParity(src, {}, 'blur -> line_profile', { pre: [STEP('blur', 'gaussian_blur')], mirroredOut: true });
    roiParity(src, { view: 'overlay' }, 'blur -> roi overlay', { pre: [STEP('blur', 'gaussian_blur')], mirroredOut: true });
  } finally { src.delete(); big.delete(); }
});

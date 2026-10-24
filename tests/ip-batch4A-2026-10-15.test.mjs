// IP batch 4A (2026-10-15) — 5 op ใหม่ (guided_filter, focus_score, pseudocolor, delta_e, roi_apply) + 8 โหมด/param ใหม่ใน op เดิม
// (custom_kernel motion/disk · gaussian_noise speckle · gamma log · otsu triangle · fft_filter butterworth · white_balance white_patch/ref_point ·
//  colorspace xyz/luv/yuv · mmPerPx ใน regionprops/feret/line_profile)
// สเปก: Vision_boi/specs/ip_batch4A_2026-10-15.md · WORKLOG: WORKLOG-2026-10-15-ip-batch4.md
// โครงเดียวกับ batch 3: ip4a-0 (fingerprint, เทสต์แรกของไฟล์) + ip4a-0w (wrapper balance เส้นทางชุดเดียวกัน) → L1..L3 · golden hash (ip4a-G)
// พิสูจน์ว่า pipeline เดิม byte-identical · parity กับ Python จริง (cv2 5.0 / numpy 2.x) — ข้ามเฉพาะเมื่อไม่มี python/cv2
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import {
  OPS, FILTERS, defaultParams, needsOf, psfKernel, PSF_SS, guidedFilterF32, PSEUDOCOLOR_MAPS, seededGaussian, MM_COLUMNS, mmValue, deltaE76, deltaE2000,
} from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS, OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { GROUP_ICON, OP_PRIMARY } from '../src/renderer/filters/ui-meta.js';
import { thresholdMarker } from '../src/renderer/ui/phase3-math.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import {
  assertLeakFree, assertWrapperBalance, fingerprint, makeThrowingCv, makeWrapperTracker,
} from './_leak-helpers.mjs';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.LUT === 'function' && typeof c.dft === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

const readSrc = (rel) => fs.readFileSync(new URL(`../src/renderer/${rel}`, import.meta.url), 'utf8');
const NEW_OPS = ['guided_filter', 'focus_score', 'pseudocolor', 'delta_e', 'roi_apply'];
const HOST = { guided_filter: 'filter_denoise', focus_score: 'analysis_detect', pseudocolor: 'color_tools', delta_e: 'image_quality', roi_apply: 'image_arithmetic' };
const NEW_ICONS = { guided_filter: '≀', pseudocolor: '🌡', focus_score: '⦿', delta_e: '◬', roi_apply: '⊡' };
const DE_KEYS = ['de_mean', 'de_max', 'de_over_pct', 'de_a_L', 'de_a_a', 'de_a_b', 'de_b_L', 'de_b_a', 'de_b_b'];
const RA_KEYS = ['ra_px', 'ra_pct'];

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
function rgbFromGray(g) { const o = new cv.Mat(); cv.cvtColor(g, o, cv.COLOR_GRAY2RGB); return o; }
function noiseRGB(w, h, seed) { const r = lcg(seed); return rgb(w, h, () => [Math.floor(r() * 256), Math.floor(r() * 256), Math.floor(r() * 256)]); }
function noiseGray(w, h, seed) { const r = lcg(seed); return gray(w, h, () => Math.floor(r() * 256)); }
// ภาพสีเรียบ ๆ ที่มีหลายเฉด (ใช้กับ parity ที่ต้องการเนื้อภาพไม่ใช่ noise ล้วน)
function smoothRGB(w, h) { return rgb(w, h, (x, y) => [(x * 255 / Math.max(1, w - 1)) | 0, (y * 255 / Math.max(1, h - 1)) | 0, ((x + y) * 5) & 255]); }
function bytes(m) { return Array.from(m.data); }
function black(w, h) { return new cv.Mat(h, w, cv.CV_8UC1, new cv.Scalar(0)); }
const W255 = new cv.Scalar(255);
function fillRectM(m, x, y, w, h, v = 255) { cv.rectangle(m, new cv.Point(x, y), new cv.Point(x + w - 1, y + h - 1), new cv.Scalar(v), -1); }
function blobGray() {
  const m = new cv.Mat(240, 320, cv.CV_8UC1, new cv.Scalar(20));
  cv.circle(m, new cv.Point(80, 80), 40, new cv.Scalar(220), -1);
  cv.rectangle(m, new cv.Point(150, 40), new cv.Point(230, 120), new cv.Scalar(180), -1);
  cv.ellipse(m, new cv.Point(240, 180), new cv.Size(50, 20), 30, 0, 360, new cv.Scalar(250), -1);
  cv.circle(m, new cv.Point(60, 190), 15, new cv.Scalar(100), -1);
  return m;
}
function blobColor() {
  const g = blobGray(), o = new cv.Mat();
  cv.cvtColor(g, o, cv.COLOR_GRAY2RGB); g.delete();
  for (let i = 0; i < o.data.length; i += 3) o.data[i + 2] = Math.min(255, o.data[i + 2] * 0.8);
  return o;
}
// ภาพขาวดำสำหรับ regionprops / feret / line_profile (หลายวัตถุแยกจากกัน)
function objectsScene() {
  const m = black(200, 140);
  fillRectM(m, 10, 10, 40, 20);
  cv.circle(m, new cv.Point(100, 40), 25, W255, -1);
  cv.ellipse(m, new cv.Point(160, 100), new cv.Size(30, 12), 30, 0, 360, W255, -1);
  fillRectM(m, 20, 80, 30, 30);
  return m;
}
function runOp(opId, src, params = {}, ctx = {}, cvUse = cv) {
  const out = OPS[opId].apply(cvUse, src, params, ctx);
  return { out, ctx };
}
function runInfo(opId, src, params = {}, ctx = {}) { const { out } = runOp(opId, src, params, ctx); out.delete(); return ctx; }
function scalarMap(ctx) { return Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value])); }
function near(a, b, tol, label) { assert.ok(Math.abs(a - b) <= tol, `${label}: ${a} vs ${b} (tol ${tol})`); }
function sameBytes(a, b, label) {
  assert.equal(a.rows, b.rows, `${label}: rows`);
  assert.equal(a.cols, b.cols, `${label}: cols`);
  assert.equal(a.channels(), b.channels(), `${label}: channels`);
  let n = 0;
  const da = a.data, db = b.data;
  for (let i = 0; i < da.length; i++) if (da[i] !== db[i]) n++;
  assert.equal(n, 0, `${label}: ${n} bytes differ`);
}
const STEP = (id, mode, extra = {}) => ({ id, params: { ...defaultParams(id), mode, ...extra } });
const OPSTEP = (opId, extra = {}) => {
  const prefixed = {};
  for (const [k, v] of Object.entries(extra)) prefixed[`${opId}__${k}`] = v;
  const step = STEP(OP_TO_FILTER[opId], opId, prefixed);
  if (FILTERS[step.id].inputs === 2) step.src2 = { pipeline: [] };
  return step;
};
const lcgBytes = (seed, n) => { const r = lcg(seed); const a = new Uint8Array(n); for (let i = 0; i < n; i++) a[i] = Math.floor(r() * 256); return a; };

// ===================================================================
// ip4a-0 — fingerprint (เทสต์แรกของไฟล์ = process สะอาด) · ip4a-0w — wrapper balance เส้นทางชุดเดียวกัน
// ===================================================================
function batch4ALeakPaths(only) {
  const mats = [];
  const keep = (m) => { mats.push(m); return m; };
  const want = (id) => !only || only === id;
  const empty = keep(new cv.Mat());
  const one = keep(gray(1, 1, () => 128)), oneC = keep(rgb(1, 1, () => [200, 100, 50]));
  const g = keep(gray(40, 30, (x, y) => (x * 7 + y * 3) & 255));
  const c = keep(rgb(40, 30, (x, y) => [x * 6 & 255, y * 8 & 255, 90 + ((x * y) & 63)]));
  const a = keep(toRGBA(c));
  const gB = keep(gray(30, 20, (x, y) => (x * 9 + y * 5) & 255));
  const cB = keep(rgb(40, 30, (x, y) => [(x * 5 + 40) & 255, (y * 7 + 10) & 255, 120]));
  const cBsmall = keep(rgb(7, 5, (x, y) => [x * 30, y * 40, 60]));
  const scene = keep(objectsScene());
  const paths = [];
  const add = (id, params, src, tag, B) => paths.push({
    label: `${id} ${tag}`,
    run: (cvUse) => { const o = OPS[id].apply(cvUse, src, params, B === undefined ? {} : { src2: B }); o.delete(); },
  });
  const srcs = [['gray', g], ['rgb', c], ['rgba', a], ['1x1', one], ['0x0', empty]];
  if (want('custom_kernel')) {
    for (const preset of ['motion', 'disk']) for (const psfSize of [1, 9, 50]) for (const [t, s] of srcs) add('custom_kernel', { preset, psfSize, psfAngle: 30 }, s, `${preset} ${psfSize} ${t}`);
    for (const [t, s] of srcs) add('custom_kernel', { preset: 'motion', psfSize: 9, amount: 0.5 }, s, `motion amount .5 ${t}`);
    add('custom_kernel', { preset: 'constructor' }, g, 'constructor');
  }
  if (want('guided_filter')) for (const radius of [1, 2, 20]) for (const [t, s] of srcs) add('guided_filter', { radius }, s, `r${radius} ${t}`);
  if (want('focus_score')) {
    for (const method of ['lap_var', 'tenengrad']) for (const view of ['lap', 'pass']) for (const roiPct of [100, 50]) for (const [t, s] of srcs) add('focus_score', { method, view, roiPct }, s, `${method} ${view} ${roiPct} ${t}`);
  }
  if (want('pseudocolor')) {
    for (const range of ['full', 'minmax']) for (const [t, s] of srcs) add('pseudocolor', { map: 'jet', range }, s, `jet ${range} ${t}`);
    for (const map of ['hot', 'cool']) for (const range of ['full', 'minmax']) add('pseudocolor', { map, range }, g, `${map} ${range} gray`);
  }
  if (want('delta_e')) {
    for (const formula of ['cie76', 'ciede2000']) for (const compare of ['pixel', 'mean_b']) for (const view of ['heatmap', 'over']) add('delta_e', { formula, compare, view }, c, `${formula} ${compare} ${view}`, cB);
    for (const formula of ['cie76', 'ciede2000']) {
      for (const [t, A, B] of [['gray/rgb', g, c], ['rgb/gray-small', c, gB], ['rgba/rgba', a, a], ['1x1/1x1', one, oneC], ['rgb/resized', c, cBsmall]]) {
        add('delta_e', { formula }, A, `${formula} ${t}`, B);
        add('delta_e', { formula, compare: 'mean_b' }, A, `${formula} mean_b ${t}`, B);
      }
    }
    add('delta_e', {}, c, 'no B'); add('delta_e', {}, c, 'B 0x0', empty); add('delta_e', {}, empty, 'A 0x0', cB);
  }
  if (want('roi_apply')) {
    for (const shape of ['rect', 'ellipse']) for (const invert of [false, true]) for (const feather of [0, 5]) for (const view of ['result', 'outline']) add('roi_apply', { shape, invert, feather, view }, c, `${shape} inv${invert} f${feather} ${view}`, cB);
    for (const feather of [0, 3]) {
      for (const [t, A, B] of [['gray/gray', g, gB], ['gray/rgb', g, c], ['rgba/rgb', a, c], ['rgb/resized', c, cBsmall], ['1x1/1x1', one, oneC]]) add('roi_apply', { feather, view: 'outline', shape: 'ellipse' }, A, `f${feather} ${t}`, B);
    }
    add('roi_apply', {}, c, 'no B'); add('roi_apply', {}, c, 'B 0x0', empty); add('roi_apply', {}, empty, 'A 0x0', cB);
  }
  if (want('white_balance')) {
    for (const method of ['white_patch', 'ref_point']) for (const [t, s] of srcs) add('white_balance', { method }, s, `${method} ${t}`);
    add('white_balance', { method: 'ref_point', refX: 0, refY: 100, refSize: 50 }, c, 'ref corner'); add('white_balance', { method: 'ref_point', refX: 100, refY: 0, refSize: 1 }, a, 'ref corner rgba');
  }
  if (want('fft_filter')) for (const type of ['low', 'high']) for (const order of [1, 10]) add('fft_filter', { type, shape: 'butterworth', order }, g, `butterworth ${type} n${order}`);
  if (want('gamma')) for (const [t, s] of srcs) add('gamma', { curve: 'log' }, s, `log ${t}`);
  if (want('otsu_threshold')) for (const invert of [false, true]) for (const [t, s] of [['gray', g], ['1x1', one], ['0x0', empty]]) add('otsu_threshold', { method: 'triangle', invert }, s, `triangle inv${invert} ${t}`);
  if (want('gaussian_noise')) for (const [t, s] of srcs) add('gaussian_noise', { kind: 'speckle' }, s, `speckle ${t}`);
  if (want('colorspace')) for (const space of ['xyz', 'luv', 'yuv']) for (const [t, s] of [['rgb', c], ['gray', g]]) add('colorspace', { space }, s, `${space} ${t}`);
  if (want('mm')) {
    add('regionprops', { minArea: 0, mmPerPx: 0.25 }, scene, 'regionprops mm'); add('regionprops', { mmPerPx: 0.25 }, empty, 'regionprops mm 0x0');
    add('feret', { minArea: 0, mmPerPx: 0.25 }, scene, 'feret mm'); add('feret', { mmPerPx: 0.25 }, empty, 'feret mm 0x0');
    add('line_profile', { mmPerPx: 0.25 }, c, 'line_profile mm'); add('line_profile', { mmPerPx: 0.25 }, empty, 'line_profile mm 0x0');
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

test('ip4a-0. leak (address fingerprint cycle, first test of a clean process): every path of the 5 new ops and the new modes of custom_kernel / gamma / otsu / gaussian_noise / fft_filter / white_balance / colorspace / mm columns (all selects, gray / RGB / RGBA / 1x1 / 0x0, no B / B 0x0 / B resized) keeps revisiting the same few malloc states', () => {
  const { paths, cleanup } = batch4ALeakPaths();
  try {
    assert.ok(paths.length >= 200, `paths ${paths.length}`);
    fingerprintCycleProbe('IP batch 4A (all paths)', () => { for (const q of paths) q.run(cv); });
  } finally { cleanup(); }
});

test('ip4a-0w. leak (wrapper balance twin of ip4a-0): the exact same path table creates and deletes every embind wrapper it makes (box / split / merge temporaries, ROI clones, Lab Mats, masks, outputs, passthrough clones)', () => {
  const { paths, cleanup } = batch4ALeakPaths();
  try {
    assertWrapperBalance(cv, 'IP batch 4A ops (all paths)', (c) => { for (const p of paths) p.run(c); }, { minCreated: 600 });
  } finally { cleanup(); }
});

// ===================================================================
// ip4a-L1…L3 — leak ต่อ op / ต่อเส้นทาง
// ===================================================================
test('ip4a-L1. leak: every path of every new op / mode balances every wrapper and grows the heap by 0', () => {
  for (const id of ['custom_kernel', 'guided_filter', 'focus_score', 'pseudocolor', 'delta_e', 'roi_apply', 'white_balance', 'fft_filter', 'gamma', 'otsu_threshold', 'gaussian_noise', 'colorspace', 'mm']) {
    const { paths, cleanup } = batch4ALeakPaths(id);
    try {
      assert.ok(paths.length > 0, id);
      for (const q of paths) assertLeakFree(cv, q.label, (c) => q.run(c), { minCreated: 0, iters: 6, warmup: 3 });
    } finally { cleanup(); }
  }
});

// ฉีด error ใน method ของ Mat (cv.Mat.prototype) ชั่วคราว — makeThrowingCv ทำได้เฉพาะ function ของ cv · onCall = ครั้งที่ n (ไม่ใส่ = ทุกครั้ง)
function withThrowingMatMethod(name, onCall, fn) {
  const proto = cv.Mat.prototype;
  const desc = Object.getOwnPropertyDescriptor(proto, name);
  assert.ok(desc && typeof desc.value === 'function', `Mat.${name}`);
  let count = 0;
  const orig = desc.value;
  const patched = function (...args) {
    count++;
    if (onCall == null || count === onCall) throw new Error(`injected test failure (Mat.${name}${onCall == null ? '' : ` call #${onCall}`})`);
    return orig.apply(this, args);
  };
  Object.assign(patched, orig);
  Object.defineProperty(proto, name, { ...desc, value: patched });
  try { return fn(); } finally { Object.defineProperty(proto, name, desc); }
}

function throwingCases() {
  const g = gray(40, 30, (x, y) => (x * 7 + y * 3) & 255), c = rgb(40, 30, (x, y) => [x * 6 & 255, y * 8 & 255, 90]), a = toRGBA(c);
  const cB = rgb(40, 30, (x, y) => [(x * 5 + 40) & 255, (y * 7 + 10) & 255, 120]), small = rgb(7, 5, (x, y) => [x * 30, y * 40, 60]), scene = objectsScene(), gB = gray(40, 30, () => 1);
  const one = (id, p, src, B) => (cvU) => OPS[id].apply(cvU, src, p, B === undefined ? {} : { src2: B });
  const cases = [
    // [label, cv function | [name, onCall] | { mat: name, onCall }, runner]
    ['custom_kernel matFromArray (motion)', 'matFromArray', one('custom_kernel', { preset: 'motion' }, c)],
    ['custom_kernel matFromArray (disk)', 'matFromArray', one('custom_kernel', { preset: 'disk' }, c)],
    ['custom_kernel filter2D (motion)', 'filter2D', one('custom_kernel', { preset: 'motion' }, c)],
    ['custom_kernel addWeighted (motion amount .5)', 'addWeighted', one('custom_kernel', { preset: 'motion', amount: 0.5 }, c)],
    ['guided_filter boxFilter (1st)', 'boxFilter', one('guided_filter', {}, g)],
    ['guided_filter boxFilter (3rd = box(II), before A/B)', ['boxFilter', 3], one('guided_filter', {}, g)],
    ['guided_filter boxFilter (5th = box(A), A and B exist)', ['boxFilter', 5], one('guided_filter', {}, g)],
    ['guided_filter boxFilter (6th = box(B))', ['boxFilter', 6], one('guided_filter', {}, g)],
    ['guided_filter boxFilter (8th = 2nd channel of rgb)', ['boxFilter', 8], one('guided_filter', {}, c)],
    ['guided_filter multiply (1st)', 'multiply', one('guided_filter', {}, g)],
    ['guided_filter multiply (2nd)', ['multiply', 2], one('guided_filter', {}, g)],
    ['guided_filter split (rgb)', 'split', one('guided_filter', {}, c)],
    ['guided_filter merge (rgb)', 'merge', one('guided_filter', {}, c)],
    ['guided_filter cvtColor (rgba dropAlpha)', 'cvtColor', one('guided_filter', {}, a)],
    ['guided_filter convertTo (Mat method)', { mat: 'convertTo' }, one('guided_filter', {}, g)],
    ['guided_filter create #1 (A)', { mat: 'create', onCall: 1 }, one('guided_filter', {}, g)],
    ['guided_filter create #2 (B)', { mat: 'create', onCall: 2 }, one('guided_filter', {}, g)],
    ['guided_filter create #3 (q, after A / B exist)', { mat: 'create', onCall: 3 }, one('guided_filter', {}, g)],
    ['focus_score Laplacian (1st, lap_var)', 'Laplacian', one('focus_score', {}, g)],
    ['focus_score Laplacian (2nd, view lap)', ['Laplacian', 2], one('focus_score', {}, g)],
    ['focus_score Sobel (1st)', 'Sobel', one('focus_score', { method: 'tenengrad' }, g)],
    ['focus_score Sobel (2nd)', ['Sobel', 2], one('focus_score', { method: 'tenengrad' }, g)],
    ['focus_score convertScaleAbs', 'convertScaleAbs', one('focus_score', {}, g)],
    ['focus_score cvtColor (grayIfNeeded, rgb)', 'cvtColor', one('focus_score', {}, c)],
    ['focus_score roi (Mat method)', { mat: 'roi' }, one('focus_score', { roiPct: 50 }, g)],
    ['focus_score clone (Mat method)', { mat: 'clone' }, one('focus_score', { roiPct: 50 }, g)],
    ['pseudocolor cvtColor (grayIfNeeded, rgb)', 'cvtColor', one('pseudocolor', {}, c)],
    ['delta_e cvtColor (rgbIfNeeded gray A)', 'cvtColor', one('delta_e', {}, g, cB)],
    ['delta_e cvtColor (Lab of A, rgb)', 'cvtColor', one('delta_e', {}, c, cB)],
    ['delta_e cvtColor (Lab of B)', ['cvtColor', 2], one('delta_e', {}, c, cB)],
    ['delta_e resize (matchSize)', 'resize', one('delta_e', {}, c, small)],
    ['delta_e convertTo #1 (Mat method)', { mat: 'convertTo', onCall: 1 }, one('delta_e', {}, c, cB)],
    ['delta_e convertTo #2 (Mat method)', { mat: 'convertTo', onCall: 2 }, one('delta_e', {}, c, cB)],
    ['roi_apply copyTo (Mat method, feather 0)', { mat: 'copyTo' }, one('roi_apply', {}, c, cB)],
    ['roi_apply matFromArray (mask, feather 0)', 'matFromArray', one('roi_apply', {}, c, cB)],
    ['roi_apply matFromArray (mf, feather 5)', 'matFromArray', one('roi_apply', { feather: 5 }, c, cB)],
    ['roi_apply GaussianBlur', 'GaussianBlur', one('roi_apply', { feather: 5 }, c, cB)],
    ['roi_apply resize', 'resize', one('roi_apply', {}, c, small)],
    ['roi_apply cvtColor (gray A, rgb B)', 'cvtColor', one('roi_apply', {}, g, cB)],
    ['roi_apply cvtColor (outline of gray result)', 'cvtColor', one('roi_apply', { view: 'outline' }, g, gB)],
    ['white_balance cvtColor (rgba dropAlpha, white_patch)', 'cvtColor', one('white_balance', { method: 'white_patch' }, a)],
    ['white_balance cvtColor (rgba dropAlpha, ref_point)', 'cvtColor', one('white_balance', { method: 'ref_point' }, a)],
    ['fft_filter dft (1st, butterworth)', 'dft', one('fft_filter', { shape: 'butterworth' }, g)],
    ['fft_filter dft (2nd, butterworth)', ['dft', 2], one('fft_filter', { shape: 'butterworth' }, g)],
    ['gamma LUT (log)', 'LUT', one('gamma', { curve: 'log' }, c)],
    ['gamma matFromArray (log)', 'matFromArray', one('gamma', { curve: 'log' }, c)],
    ['otsu threshold (triangle)', 'threshold', one('otsu_threshold', { method: 'triangle' }, g)],
    ['gaussian_noise cvtColor (speckle rgba)', 'cvtColor', one('gaussian_noise', { kind: 'speckle' }, a)],
    ['regionprops (mm) findContours', 'findContours', one('regionprops', { mmPerPx: 0.2 }, scene)],
    ['feret (mm) findContours', 'findContours', one('feret', { mmPerPx: 0.2 }, scene)],
  ];
  return { cases, cleanup: () => { for (const m of [g, c, a, cB, small, scene, gB]) m.delete(); } };
}

test('ip4a-L2. leak on the exception path: a throw inside every cv call (and Mat method) the new paths make still gives every wrapper back (injected failures; each case must throw)', () => {
  const { cases, cleanup } = throwingCases();
  try {
    for (const [label, how, fn] of cases) {
      const run = (cvU) => {
        if (how && how.mat) return withThrowingMatMethod(how.mat, how.onCall ?? null, () => { const o = fn(cvU); o.delete(); });
        const [name, onCall] = Array.isArray(how) ? how : [how, null];
        const o = fn(makeThrowingCv(cvU, name, onCall ? { onCall } : {}));
        o.delete();
        return undefined;
      };
      assertWrapperBalance(cv, `throw in ${label}`, run, { expectThrow: /injected test failure/, minCreated: 0 });
    }
  } finally { cleanup(); }
});

test('ip4a-L3. leak on the exception path AFTER the output exists: a ctx whose "info" / "values" setter throws makes every new op / mode delete its own output', () => {
  const g = gray(40, 30, (x, y) => (x * 7 + y * 3) & 255), c = rgb(40, 30, (x, y) => [x * 6 & 255, y * 8 & 255, 90]), cB = rgb(40, 30, () => [10, 200, 30]), scene = objectsScene(), a = toRGBA(c);
  const tInfo = (B) => ({ ...(B ? { src2: B } : {}), set info(_) { throw new Error('injected test failure (ctx.info)'); } });
  const tValues = (B) => ({ ...(B ? { src2: B } : {}), info: '', set values(_) { throw new Error('injected test failure (ctx.values)'); } });
  const check = (label, id, p, src, ctxMaker, B) => assertWrapperBalance(cv, label, (cvU) => { const o = OPS[id].apply(cvU, src, p, ctxMaker(B)); o.delete(); }, { expectThrow: /injected test failure/, minCreated: 1 });
  try {
    check('info guided gray', 'guided_filter', {}, g, tInfo); check('info guided rgb', 'guided_filter', {}, c, tInfo);
    check('info focus', 'focus_score', {}, g, tInfo); check('values focus', 'focus_score', {}, g, tValues); check('values focus pass', 'focus_score', { view: 'pass', method: 'tenengrad' }, g, tValues);
    check('info pseudocolor', 'pseudocolor', { range: 'minmax' }, g, tInfo);
    check('info delta_e', 'delta_e', {}, c, tInfo, cB); check('values delta_e', 'delta_e', { formula: 'ciede2000', compare: 'mean_b' }, c, tValues, cB);
    for (const view of ['result', 'outline']) for (const feather of [0, 4]) {
      check(`info roi_apply ${view} f${feather}`, 'roi_apply', { view, feather }, c, tInfo, cB);
      check(`values roi_apply ${view} f${feather}`, 'roi_apply', { view, feather }, g, tValues, cB);
    }
    check('info custom_kernel motion', 'custom_kernel', { preset: 'motion' }, c, tInfo); check('info custom_kernel disk', 'custom_kernel', { preset: 'disk', amount: 0.5 }, c, tInfo);
    check('info gamma log', 'gamma', { curve: 'log' }, c, tInfo); check('info otsu triangle', 'otsu_threshold', { method: 'triangle' }, g, tInfo);
    check('info speckle', 'gaussian_noise', { kind: 'speckle' }, c, tInfo); check('info butterworth', 'fft_filter', { shape: 'butterworth' }, g, tInfo);
    check('info white_patch', 'white_balance', { method: 'white_patch' }, c, tInfo); check('info ref_point', 'white_balance', { method: 'ref_point' }, a, tInfo);
    check('values regionprops mm', 'regionprops', { minArea: 0, mmPerPx: 0.3 }, scene, tValues); check('values feret mm', 'feret', { minArea: 0, mmPerPx: 0.3 }, scene, tValues);
    check('values line_profile mm', 'line_profile', { mmPerPx: 0.3 }, c, tValues);
  } finally { for (const m of [g, c, cB, scene, a]) m.delete(); }
});

// ===================================================================
// ip4a-G — golden: op เดิม 10 ตัว × params เดิม ให้ผล byte-identical (ภาพ + info + values + Python ที่ export + ข้อความการ์ด)
// hash วัดบนโค้ดเดิม (HEAD 1755233) ก่อนแก้โค้ดบรรทัดแรก · 3 กรณี: (1) params เดิม (2) ส่ง param ใหม่เป็นค่าเริ่มต้นตรง ๆ (3) select ใหม่เป็นค่าแปลก / "constructor" / ตัวเลขนอกช่วง
// ===================================================================
const GOLDEN = {
  "custom_kernel:{\"preset\":\"identity\"}|color97": "912220aba888da50",
  "custom_kernel:{\"preset\":\"identity\"}|gray97": "22f4de4c526abb01",
  "custom_kernel:{\"preset\":\"identity\"}|blobG": "e280b17f55818b2d",
  "custom_kernel:{\"preset\":\"identity\"}|blobC": "24f92d11b43f36ae",
  "custom_kernel:{\"preset\":\"identity\"}|one1": "f64060e2c0c8eb45",
  "custom_kernel:{\"preset\":\"identity\"}|one3": "d689fddd02930a62",
  "custom_kernel:{\"preset\":\"identity\"}|PY": "454c73bd54bfa7a1",
  "custom_kernel:{\"preset\":\"identity\"}|UI": "3b9f431326bccafe",
  "custom_kernel:{\"preset\":\"identity\",\"amount\":0.5}|color97": "912220aba888da50",
  "custom_kernel:{\"preset\":\"identity\",\"amount\":0.5}|gray97": "22f4de4c526abb01",
  "custom_kernel:{\"preset\":\"identity\",\"amount\":0.5}|blobG": "e280b17f55818b2d",
  "custom_kernel:{\"preset\":\"identity\",\"amount\":0.5}|blobC": "24f92d11b43f36ae",
  "custom_kernel:{\"preset\":\"identity\",\"amount\":0.5}|one1": "f64060e2c0c8eb45",
  "custom_kernel:{\"preset\":\"identity\",\"amount\":0.5}|one3": "d689fddd02930a62",
  "custom_kernel:{\"preset\":\"identity\",\"amount\":0.5}|PY": "de1a9d30a696b6bb",
  "custom_kernel:{\"preset\":\"identity\",\"amount\":0.5}|UI": "ce2652cf4f823887",
  "custom_kernel:{\"preset\":\"sharpen\"}|color97": "23919b9f473907a4",
  "custom_kernel:{\"preset\":\"sharpen\"}|gray97": "11d7444fe70bbd71",
  "custom_kernel:{\"preset\":\"sharpen\"}|blobG": "62313270a4862a83",
  "custom_kernel:{\"preset\":\"sharpen\"}|blobC": "476b6a68792c0b0e",
  "custom_kernel:{\"preset\":\"sharpen\"}|one1": "a61b3308e534debd",
  "custom_kernel:{\"preset\":\"sharpen\"}|one3": "47c8eccde7f2a668",
  "custom_kernel:{\"preset\":\"sharpen\"}|PY": "6b6c50a85860fbf0",
  "custom_kernel:{\"preset\":\"sharpen\"}|UI": "b5aabb79ddfa754f",
  "custom_kernel:{\"preset\":\"sharpen\",\"amount\":0.5}|color97": "829a1ad5bfaf914e",
  "custom_kernel:{\"preset\":\"sharpen\",\"amount\":0.5}|gray97": "8b1ee4cdbce0e345",
  "custom_kernel:{\"preset\":\"sharpen\",\"amount\":0.5}|blobG": "9dac7c1934ae99d2",
  "custom_kernel:{\"preset\":\"sharpen\",\"amount\":0.5}|blobC": "afa17435c8bb18df",
  "custom_kernel:{\"preset\":\"sharpen\",\"amount\":0.5}|one1": "a61b3308e534debd",
  "custom_kernel:{\"preset\":\"sharpen\",\"amount\":0.5}|one3": "47c8eccde7f2a668",
  "custom_kernel:{\"preset\":\"sharpen\",\"amount\":0.5}|PY": "1640c70d472554d7",
  "custom_kernel:{\"preset\":\"sharpen\",\"amount\":0.5}|UI": "4b20d04e7dfdfabc",
  "custom_kernel:{\"preset\":\"edge\"}|color97": "b1e1d28e4bb7f331",
  "custom_kernel:{\"preset\":\"edge\"}|gray97": "e61e20b78ee68fb1",
  "custom_kernel:{\"preset\":\"edge\"}|blobG": "74efe8b99259bf0a",
  "custom_kernel:{\"preset\":\"edge\"}|blobC": "f6c3f8a74dd49fba",
  "custom_kernel:{\"preset\":\"edge\"}|one1": "81d85617186e9a67",
  "custom_kernel:{\"preset\":\"edge\"}|one3": "a07f5cfe41abd8a4",
  "custom_kernel:{\"preset\":\"edge\"}|PY": "1889763637e5e6db",
  "custom_kernel:{\"preset\":\"edge\"}|UI": "4f1c2db41e65acf5",
  "custom_kernel:{\"preset\":\"edge\",\"amount\":0.5}|color97": "c68b26f73ea444dc",
  "custom_kernel:{\"preset\":\"edge\",\"amount\":0.5}|gray97": "957df296c50f5e04",
  "custom_kernel:{\"preset\":\"edge\",\"amount\":0.5}|blobG": "6aa075c6efe16faf",
  "custom_kernel:{\"preset\":\"edge\",\"amount\":0.5}|blobC": "f542d09f37cdda23",
  "custom_kernel:{\"preset\":\"edge\",\"amount\":0.5}|one1": "0dbcb737f3204a12",
  "custom_kernel:{\"preset\":\"edge\",\"amount\":0.5}|one3": "83ad965caca3ebc4",
  "custom_kernel:{\"preset\":\"edge\",\"amount\":0.5}|PY": "664aa3306b755707",
  "custom_kernel:{\"preset\":\"edge\",\"amount\":0.5}|UI": "876ccbc921845fa0",
  "custom_kernel:{\"preset\":\"emboss\"}|color97": "fd06bce17a8f13d9",
  "custom_kernel:{\"preset\":\"emboss\"}|gray97": "29ee3c47334665ca",
  "custom_kernel:{\"preset\":\"emboss\"}|blobG": "4b49bf1eea0886f6",
  "custom_kernel:{\"preset\":\"emboss\"}|blobC": "aa3bd0336f6380d0",
  "custom_kernel:{\"preset\":\"emboss\"}|one1": "955fc3e401acfa05",
  "custom_kernel:{\"preset\":\"emboss\"}|one3": "089492be5e99f7f8",
  "custom_kernel:{\"preset\":\"emboss\"}|PY": "a03ccc23d24e522a",
  "custom_kernel:{\"preset\":\"emboss\"}|UI": "fbfa7c6e0f07f923",
  "custom_kernel:{\"preset\":\"emboss\",\"amount\":0.5}|color97": "a30f5c36be245acf",
  "custom_kernel:{\"preset\":\"emboss\",\"amount\":0.5}|gray97": "bec9fc249ed136a9",
  "custom_kernel:{\"preset\":\"emboss\",\"amount\":0.5}|blobG": "32220c0977296abb",
  "custom_kernel:{\"preset\":\"emboss\",\"amount\":0.5}|blobC": "9a57e09522057a17",
  "custom_kernel:{\"preset\":\"emboss\",\"amount\":0.5}|one1": "955fc3e401acfa05",
  "custom_kernel:{\"preset\":\"emboss\",\"amount\":0.5}|one3": "089492be5e99f7f8",
  "custom_kernel:{\"preset\":\"emboss\",\"amount\":0.5}|PY": "17a183f95794116f",
  "custom_kernel:{\"preset\":\"emboss\",\"amount\":0.5}|UI": "1970bcc427eff92a",
  "custom_kernel:{\"preset\":\"outline\"}|color97": "a2b257dc819d8aed",
  "custom_kernel:{\"preset\":\"outline\"}|gray97": "8d9d3e1477d4c308",
  "custom_kernel:{\"preset\":\"outline\"}|blobG": "e20a87642ad6d863",
  "custom_kernel:{\"preset\":\"outline\"}|blobC": "e1e6c29a94420cd7",
  "custom_kernel:{\"preset\":\"outline\"}|one1": "b2ab9382845cdc5b",
  "custom_kernel:{\"preset\":\"outline\"}|one3": "c399287af71ff777",
  "custom_kernel:{\"preset\":\"outline\"}|PY": "bbc362bdf0e38a00",
  "custom_kernel:{\"preset\":\"outline\"}|UI": "72fb2f15c50a5345",
  "custom_kernel:{\"preset\":\"outline\",\"amount\":0.5}|color97": "381399db6168332a",
  "custom_kernel:{\"preset\":\"outline\",\"amount\":0.5}|gray97": "894aad71f009b9d4",
  "custom_kernel:{\"preset\":\"outline\",\"amount\":0.5}|blobG": "19784dcbee838557",
  "custom_kernel:{\"preset\":\"outline\",\"amount\":0.5}|blobC": "53cc73f2007a8272",
  "custom_kernel:{\"preset\":\"outline\",\"amount\":0.5}|one1": "b2ab9382845cdc5b",
  "custom_kernel:{\"preset\":\"outline\",\"amount\":0.5}|one3": "c399287af71ff777",
  "custom_kernel:{\"preset\":\"outline\",\"amount\":0.5}|PY": "52df339c1403d8d0",
  "custom_kernel:{\"preset\":\"outline\",\"amount\":0.5}|UI": "6db724a4c972e272",
  "custom_kernel:{\"preset\":\"box3\"}|color97": "12dd2da08583ad38",
  "custom_kernel:{\"preset\":\"box3\"}|gray97": "80b2d6618e283a5f",
  "custom_kernel:{\"preset\":\"box3\"}|blobG": "8e56caeb2cd6e6ad",
  "custom_kernel:{\"preset\":\"box3\"}|blobC": "e6c122024f444c4d",
  "custom_kernel:{\"preset\":\"box3\"}|one1": "1cd8558aa0b8f0a0",
  "custom_kernel:{\"preset\":\"box3\"}|one3": "7283bb84265f6ef6",
  "custom_kernel:{\"preset\":\"box3\"}|PY": "ffaee2d9f0f8332b",
  "custom_kernel:{\"preset\":\"box3\"}|UI": "d6cd5a52cceae6e1",
  "custom_kernel:{\"preset\":\"box3\",\"amount\":0.5}|color97": "32fd5fd2232149e4",
  "custom_kernel:{\"preset\":\"box3\",\"amount\":0.5}|gray97": "1888324c0cccc730",
  "custom_kernel:{\"preset\":\"box3\",\"amount\":0.5}|blobG": "f414a62c2c4d38f0",
  "custom_kernel:{\"preset\":\"box3\",\"amount\":0.5}|blobC": "3afc57eebb03c7b4",
  "custom_kernel:{\"preset\":\"box3\",\"amount\":0.5}|one1": "1cd8558aa0b8f0a0",
  "custom_kernel:{\"preset\":\"box3\",\"amount\":0.5}|one3": "7283bb84265f6ef6",
  "custom_kernel:{\"preset\":\"box3\",\"amount\":0.5}|PY": "07fdebeff8f3449f",
  "custom_kernel:{\"preset\":\"box3\",\"amount\":0.5}|UI": "22d3c533c5f65505",
  "custom_kernel:{\"preset\":\"box5\"}|color97": "319a00ef7eb5e685",
  "custom_kernel:{\"preset\":\"box5\"}|gray97": "e0ffcc1997da23c3",
  "custom_kernel:{\"preset\":\"box5\"}|blobG": "616cd6c46251a2de",
  "custom_kernel:{\"preset\":\"box5\"}|blobC": "72fa23c1012f60f1",
  "custom_kernel:{\"preset\":\"box5\"}|one1": "5ae0084555f11c4e",
  "custom_kernel:{\"preset\":\"box5\"}|one3": "147b1eba8da86b58",
  "custom_kernel:{\"preset\":\"box5\"}|PY": "ac527e42024abfcc",
  "custom_kernel:{\"preset\":\"box5\"}|UI": "bfac7bf479d7e450",
  "custom_kernel:{\"preset\":\"box5\",\"amount\":0.5}|color97": "b74e0483f7e6f02c",
  "custom_kernel:{\"preset\":\"box5\",\"amount\":0.5}|gray97": "ea3d519d92bbeb3c",
  "custom_kernel:{\"preset\":\"box5\",\"amount\":0.5}|blobG": "79fa0452abc8b493",
  "custom_kernel:{\"preset\":\"box5\",\"amount\":0.5}|blobC": "d491880ceb11ffb5",
  "custom_kernel:{\"preset\":\"box5\",\"amount\":0.5}|one1": "5ae0084555f11c4e",
  "custom_kernel:{\"preset\":\"box5\",\"amount\":0.5}|one3": "147b1eba8da86b58",
  "custom_kernel:{\"preset\":\"box5\",\"amount\":0.5}|PY": "7b5b93b11de0b112",
  "custom_kernel:{\"preset\":\"box5\",\"amount\":0.5}|UI": "e68766e7603b7271",
  "gaussian_noise:{}|color97": "990f0856e735a827",
  "gaussian_noise:{}|gray97": "bd8ef807dff3f271",
  "gaussian_noise:{}|blobG": "f491e561f4069461",
  "gaussian_noise:{}|blobC": "d36cd60f5955a63d",
  "gaussian_noise:{}|one1": "978edf7a99a39717",
  "gaussian_noise:{}|one3": "fb8510a529569040",
  "gaussian_noise:{}|PY": "3b65a38a9b137281",
  "gaussian_noise:{}|UI": "5547a86ad0d18465",
  "gaussian_noise:{\"sigma\":0}|color97": "904afeb875fce76a",
  "gaussian_noise:{\"sigma\":0}|gray97": "cf2ec2a6e4740b23",
  "gaussian_noise:{\"sigma\":0}|blobG": "653573917b4a7512",
  "gaussian_noise:{\"sigma\":0}|blobC": "b31811bc98c18247",
  "gaussian_noise:{\"sigma\":0}|one1": "2b328be4a4dbb894",
  "gaussian_noise:{\"sigma\":0}|one3": "99f5cd7384f2498e",
  "gaussian_noise:{\"sigma\":0}|PY": "c12561ab36e6e225",
  "gaussian_noise:{\"sigma\":0}|UI": "187bc592bdc1d901",
  "gaussian_noise:{\"sigma\":40,\"seed\":7}|color97": "e89d375e61070eab",
  "gaussian_noise:{\"sigma\":40,\"seed\":7}|gray97": "281df6a1930aa0f6",
  "gaussian_noise:{\"sigma\":40,\"seed\":7}|blobG": "ff9283cc470da453",
  "gaussian_noise:{\"sigma\":40,\"seed\":7}|blobC": "f1846ae54fee9d17",
  "gaussian_noise:{\"sigma\":40,\"seed\":7}|one1": "bcf340b12a4ce635",
  "gaussian_noise:{\"sigma\":40,\"seed\":7}|one3": "37db7967ca0123f0",
  "gaussian_noise:{\"sigma\":40,\"seed\":7}|PY": "0b8db12e5a7b90d0",
  "gaussian_noise:{\"sigma\":40,\"seed\":7}|UI": "6054a0794797d05a",
  "gamma:{}|color97": "8be1fa2a6a05d4b6",
  "gamma:{}|gray97": "4e06a77a533f15ee",
  "gamma:{}|blobG": "734ec98e4495cd2d",
  "gamma:{}|blobC": "af3a758a3841fbe6",
  "gamma:{}|one1": "0faf96a5f63556d4",
  "gamma:{}|one3": "0e57e90600243d20",
  "gamma:{}|PY": "5f49345c043ffbd6",
  "gamma:{}|UI": "b7a1a60f50cff665",
  "gamma:{\"g\":0.5}|color97": "c9a00e4627a539bb",
  "gamma:{\"g\":0.5}|gray97": "d44b74b5e933038f",
  "gamma:{\"g\":0.5}|blobG": "0d707f4bc7c970d3",
  "gamma:{\"g\":0.5}|blobC": "91ea690b5adbba89",
  "gamma:{\"g\":0.5}|one1": "4b0421d34d9298d9",
  "gamma:{\"g\":0.5}|one3": "03150eeb39e00b97",
  "gamma:{\"g\":0.5}|PY": "46dc9360e5710d13",
  "gamma:{\"g\":0.5}|UI": "a8fcb97db69d8782",
  "otsu_threshold:{}|color97": "4dbb7181057580fc",
  "otsu_threshold:{}|gray97": "0b18d5c9a307a79d",
  "otsu_threshold:{}|blobG": "14e1293b77d46f95",
  "otsu_threshold:{}|blobC": "5511dc0d7b26decf",
  "otsu_threshold:{}|one1": "5cb439a4d7271eb1",
  "otsu_threshold:{}|one3": "5cb439a4d7271eb1",
  "otsu_threshold:{}|PY": "7256fe7e5a863361",
  "otsu_threshold:{}|UI": "bf78ce7fb806bd3f",
  "otsu_threshold:{\"invert\":true}|color97": "2d9fa73ba6c20f4e",
  "otsu_threshold:{\"invert\":true}|gray97": "25655023454fb278",
  "otsu_threshold:{\"invert\":true}|blobG": "f3a682ab65dc04e1",
  "otsu_threshold:{\"invert\":true}|blobC": "d83d232966de1365",
  "otsu_threshold:{\"invert\":true}|one1": "4477d65138dabbc4",
  "otsu_threshold:{\"invert\":true}|one3": "4477d65138dabbc4",
  "otsu_threshold:{\"invert\":true}|PY": "c1d3819ce3d4dbec",
  "otsu_threshold:{\"invert\":true}|UI": "bf78ce7fb806bd3f",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\"}|color97": "39b34651c82d4a55",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\"}|gray97": "6660bc72ee78f6cf",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\"}|blobG": "d3df518fe41227a5",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\"}|blobC": "cda499f44f32595f",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\"}|one1": "6486f737894952ea",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\"}|one3": "6486f737894952ea",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\"}|PY": "3c413265bae7bb5c",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\"}|UI": "f976c14e2cfa86e6",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\",\"cutoff\":10}|color97": "023b1fe0d6a02ed7",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\",\"cutoff\":10}|gray97": "9d8031b909aea58b",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\",\"cutoff\":10}|blobG": "41daf39c1551b375",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\",\"cutoff\":10}|blobC": "486d56d2c351d930",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\",\"cutoff\":10}|one1": "169f5e4efa6d8429",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\",\"cutoff\":10}|one3": "169f5e4efa6d8429",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\",\"cutoff\":10}|PY": "c7dc93b7f270204c",
  "fft_filter:{\"type\":\"low\",\"shape\":\"gaussian\",\"cutoff\":10}|UI": "ec4c31c38d6486f2",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\"}|color97": "207c4211165910f7",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\"}|gray97": "d768875eeac2c979",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\"}|blobG": "5b22e4606a2ab426",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\"}|blobC": "3373b000fbe7e973",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\"}|one1": "a8193255da9dc0ae",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\"}|one3": "a8193255da9dc0ae",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\"}|PY": "78321b744eebf45d",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\"}|UI": "8c6107727353b2f8",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\",\"cutoff\":10}|color97": "25dcf8d077f1bcdf",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\",\"cutoff\":10}|gray97": "aa99fe69943ed629",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\",\"cutoff\":10}|blobG": "be95f08c6a26868f",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\",\"cutoff\":10}|blobC": "677d91213b4bd6c0",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\",\"cutoff\":10}|one1": "86f3804a38a4a5b3",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\",\"cutoff\":10}|one3": "86f3804a38a4a5b3",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\",\"cutoff\":10}|PY": "90dab0061d3524d4",
  "fft_filter:{\"type\":\"low\",\"shape\":\"ideal\",\"cutoff\":10}|UI": "f2848bd1b33439c2",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\"}|color97": "f0439ee1299bc00a",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\"}|gray97": "720fedb312353886",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\"}|blobG": "c1f9305cc8e3b060",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\"}|blobC": "75036fcd318b83fa",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\"}|one1": "3e9cea53e1d6b7c0",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\"}|one3": "3e9cea53e1d6b7c0",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\"}|PY": "1f321f8f2c547f51",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\"}|UI": "c114ddc8aaf6ca06",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\",\"cutoff\":10}|color97": "733d47c894532609",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\",\"cutoff\":10}|gray97": "d33c1ea50f2c9c96",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\",\"cutoff\":10}|blobG": "955445d4f6840e90",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\",\"cutoff\":10}|blobC": "5e0a955f9defcb4b",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\",\"cutoff\":10}|one1": "236849404dd7ad72",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\",\"cutoff\":10}|one3": "236849404dd7ad72",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\",\"cutoff\":10}|PY": "851d460002ed832c",
  "fft_filter:{\"type\":\"high\",\"shape\":\"gaussian\",\"cutoff\":10}|UI": "52c32f6f41d97aab",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\"}|color97": "b4ae29a15fa66ac1",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\"}|gray97": "7efd820d687f11d8",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\"}|blobG": "b2c10c6284382126",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\"}|blobC": "22cfd099f30e62ee",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\"}|one1": "60ebeb9d3025de0a",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\"}|one3": "60ebeb9d3025de0a",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\"}|PY": "ab29aaa73fa4eeeb",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\"}|UI": "26456dde715e21eb",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\",\"cutoff\":10}|color97": "87ae2c36cc64b568",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\",\"cutoff\":10}|gray97": "818e6c391a9acfcc",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\",\"cutoff\":10}|blobG": "7c03da0a2d6d6722",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\",\"cutoff\":10}|blobC": "e0e810a4d90750d1",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\",\"cutoff\":10}|one1": "a7d1687249871826",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\",\"cutoff\":10}|one3": "a7d1687249871826",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\",\"cutoff\":10}|PY": "d59e98ca8bcac9dc",
  "fft_filter:{\"type\":\"high\",\"shape\":\"ideal\",\"cutoff\":10}|UI": "ed9c2818aa931399",
  "white_balance:{}|color97": "c952f7ef979e9f29",
  "white_balance:{}|gray97": "dff1cbe451f29f48",
  "white_balance:{}|blobG": "d4a375c0a3c29a94",
  "white_balance:{}|blobC": "f0e1ae0668d50ca5",
  "white_balance:{}|one1": "01610b417de80e67",
  "white_balance:{}|one3": "42a09f934169c8af",
  "white_balance:{}|PY": "26dbfc104d7cd784",
  "white_balance:{}|UI": "8b2e1b923160068f",
  "colorspace:{\"space\":\"hsv\"}|color97": "aba8aeaccc5ccfcd",
  "colorspace:{\"space\":\"hsv\"}|gray97": "c7840dd94212ea29",
  "colorspace:{\"space\":\"hsv\"}|blobG": "9cb36797c45edb51",
  "colorspace:{\"space\":\"hsv\"}|blobC": "4e606ea5a35df412",
  "colorspace:{\"space\":\"hsv\"}|one1": "b318f067f1227e50",
  "colorspace:{\"space\":\"hsv\"}|one3": "377e8cc2179dcfa6",
  "colorspace:{\"space\":\"hsv\"}|PY": "c8fb8ab0542a133b",
  "colorspace:{\"space\":\"hsv\"}|UI": "c4745dab6dff505f",
  "colorspace:{\"space\":\"hls\"}|color97": "fb1211e65939b34d",
  "colorspace:{\"space\":\"hls\"}|gray97": "f936719eaf569359",
  "colorspace:{\"space\":\"hls\"}|blobG": "d672ee6de70bb9fb",
  "colorspace:{\"space\":\"hls\"}|blobC": "4d81ce8592a2caf2",
  "colorspace:{\"space\":\"hls\"}|one1": "d43f28115c3e4421",
  "colorspace:{\"space\":\"hls\"}|one3": "2e42eabaa8f8d520",
  "colorspace:{\"space\":\"hls\"}|PY": "3665d9139a6ac1ee",
  "colorspace:{\"space\":\"hls\"}|UI": "ec1daa0d187adce9",
  "colorspace:{\"space\":\"lab\"}|color97": "bef9e567792e6493",
  "colorspace:{\"space\":\"lab\"}|gray97": "a95077df7ff8da54",
  "colorspace:{\"space\":\"lab\"}|blobG": "3c870d323874e31a",
  "colorspace:{\"space\":\"lab\"}|blobC": "630a3e50b33c873d",
  "colorspace:{\"space\":\"lab\"}|one1": "f08599c739faefdc",
  "colorspace:{\"space\":\"lab\"}|one3": "6ae8f5eaa8cff465",
  "colorspace:{\"space\":\"lab\"}|PY": "c13bdacb494b1c70",
  "colorspace:{\"space\":\"lab\"}|UI": "a262062d8ccb9424",
  "colorspace:{\"space\":\"ycrcb\"}|color97": "8e4598812de47152",
  "colorspace:{\"space\":\"ycrcb\"}|gray97": "eaee76509a751c72",
  "colorspace:{\"space\":\"ycrcb\"}|blobG": "e2d2bc6087bdbda3",
  "colorspace:{\"space\":\"ycrcb\"}|blobC": "31246104658b4e0e",
  "colorspace:{\"space\":\"ycrcb\"}|one1": "70c1823fdaee128a",
  "colorspace:{\"space\":\"ycrcb\"}|one3": "b6a397479ea1fe0e",
  "colorspace:{\"space\":\"ycrcb\"}|PY": "0362b0b5e9487e10",
  "colorspace:{\"space\":\"ycrcb\"}|UI": "ed4043a0e970fb26",
  "regionprops:{}|color97": "6b9afb267e3b4c35",
  "regionprops:{}|gray97": "73933f82664d0e8a",
  "regionprops:{}|blobG": "ec2af324677e91cf",
  "regionprops:{}|blobC": "9705b712b5f8ce44",
  "regionprops:{}|one1": "72fb6644f2bf1069",
  "regionprops:{}|one3": "e075ead2855006ce",
  "regionprops:{}|PY": "41b5895ba0c9b6cc",
  "regionprops:{}|UI": "5a08ee8d291d6939",
  "regionprops:{\"minArea\":0,\"maxObjects\":0}|color97": "44046b19ca644f81",
  "regionprops:{\"minArea\":0,\"maxObjects\":0}|gray97": "e08c6746f17aef44",
  "regionprops:{\"minArea\":0,\"maxObjects\":0}|blobG": "ec2af324677e91cf",
  "regionprops:{\"minArea\":0,\"maxObjects\":0}|blobC": "9705b712b5f8ce44",
  "regionprops:{\"minArea\":0,\"maxObjects\":0}|one1": "7625e2f3a5f5be14",
  "regionprops:{\"minArea\":0,\"maxObjects\":0}|one3": "e075ead2855006ce",
  "regionprops:{\"minArea\":0,\"maxObjects\":0}|PY": "1661458e6eddda6b",
  "regionprops:{\"minArea\":0,\"maxObjects\":0}|UI": "74f5448d1e9d4422",
  "feret:{}|color97": "dad33441fa59eeb2",
  "feret:{}|gray97": "8e97793908bffc34",
  "feret:{}|blobG": "cbf688f48a9c9cd3",
  "feret:{}|blobC": "1d10bdcec22ccefe",
  "feret:{}|one1": "e745c743914a2a9c",
  "feret:{}|one3": "819be35c1c023d9d",
  "feret:{}|PY": "f429ee07c3834afe",
  "feret:{}|UI": "827f45eef7854845",
  "feret:{\"minArea\":0}|color97": "c7587fba4838a194",
  "feret:{\"minArea\":0}|gray97": "2eb1b1003bf9de7e",
  "feret:{\"minArea\":0}|blobG": "cbf688f48a9c9cd3",
  "feret:{\"minArea\":0}|blobC": "1d10bdcec22ccefe",
  "feret:{\"minArea\":0}|one1": "7a5aef87ea963b7f",
  "feret:{\"minArea\":0}|one3": "819be35c1c023d9d",
  "feret:{\"minArea\":0}|PY": "87eb7b3221ff1e1d",
  "feret:{\"minArea\":0}|UI": "4b42c020520b0c59",
  "line_profile:{}|color97": "365f3e76869d6788",
  "line_profile:{}|gray97": "cdcee2f5c63778e6",
  "line_profile:{}|blobG": "3ba1bdd24c14a14a",
  "line_profile:{}|blobC": "644247cbc0e67b38",
  "line_profile:{}|one1": "0bcf79f260d0007a",
  "line_profile:{}|one3": "537d403a28771bb3",
  "line_profile:{}|PY": "e0838d74aa136acf",
  "line_profile:{}|UI": "1b12220733dcf182",
  "line_profile:{\"view\":\"plot\"}|color97": "365f3e76869d6788",
  "line_profile:{\"view\":\"plot\"}|gray97": "cdcee2f5c63778e6",
  "line_profile:{\"view\":\"plot\"}|blobG": "d7d6a3cc16486491",
  "line_profile:{\"view\":\"plot\"}|blobC": "59524f118a18c1b9",
  "line_profile:{\"view\":\"plot\"}|one1": "0bcf79f260d0007a",
  "line_profile:{\"view\":\"plot\"}|one3": "537d403a28771bb3",
  "line_profile:{\"view\":\"plot\"}|PY": "a00d09f49154319d",
  "line_profile:{\"view\":\"plot\"}|UI": "ed2c1a22d017f0db",
};


const G_KS = ['identity', 'sharpen', 'edge', 'emboss', 'outline', 'box3', 'box5'];
const G_CASES = [];
for (const k of G_KS) { G_CASES.push(['custom_kernel', { preset: k }]); G_CASES.push(['custom_kernel', { preset: k, amount: 0.5 }]); }
G_CASES.push(['gaussian_noise', {}], ['gaussian_noise', { sigma: 0 }], ['gaussian_noise', { sigma: 40, seed: 7 }]);
G_CASES.push(['gamma', {}], ['gamma', { g: 0.5 }]);
G_CASES.push(['otsu_threshold', {}], ['otsu_threshold', { invert: true }]);
for (const t of ['low', 'high']) for (const s of ['gaussian', 'ideal']) { G_CASES.push(['fft_filter', { type: t, shape: s }]); G_CASES.push(['fft_filter', { type: t, shape: s, cutoff: 10 }]); }
G_CASES.push(['white_balance', {}]);
for (const s of ['hsv', 'hls', 'lab', 'ycrcb']) G_CASES.push(['colorspace', { space: s }]);
G_CASES.push(['regionprops', {}], ['regionprops', { minArea: 0, maxObjects: 0 }], ['feret', {}], ['feret', { minArea: 0 }], ['line_profile', {}], ['line_profile', { view: 'plot' }]);
const G_DEF = { custom_kernel: { psfSize: 9, psfAngle: 0 }, gaussian_noise: { kind: 'additive' }, gamma: { curve: 'power' }, otsu_threshold: { method: 'otsu' }, fft_filter: { order: 2 }, white_balance: { method: 'gray_world', refX: 50, refY: 50, refSize: 5 }, regionprops: { mmPerPx: 0 }, feret: { mmPerPx: 0 }, line_profile: { mmPerPx: 0 } };
const G_WEIRD = { custom_kernel: { psfSize: 'x', psfAngle: -9 }, gaussian_noise: { kind: 'constructor' }, gamma: { curve: 'constructor' }, otsu_threshold: { method: 'constructor' }, fft_filter: { order: 'abc' }, white_balance: { method: 'constructor', refX: 'q', refY: -3, refSize: 999 }, regionprops: { mmPerPx: -5 }, feret: { mmPerPx: 'abc' }, line_profile: { mmPerPx: null } };
function goldenImages() {
  const r1 = lcg(11), r2 = lcg(12);
  const mk = (w, h, ch, fn) => { const d = new Uint8Array(w * h * ch); for (let i = 0; i < d.length; i++) d[i] = fn(i); return cv.matFromArray(h, w, ch === 1 ? cv.CV_8UC1 : cv.CV_8UC3, d); };
  return { color97: mk(97, 61, 3, () => Math.floor(r1() * 256)), gray97: mk(97, 61, 1, () => Math.floor(r2() * 256)), blobG: blobGray(), blobC: blobColor(), one1: mk(1, 1, 1, () => 128), one3: mk(1, 1, 3, () => 100) };
}
const gsha = (...parts) => { const h = crypto.createHash('sha256'); for (const p of parts) h.update(p); return h.digest('hex').slice(0, 16); };
function goldenRun(op, params, src) {
  let input = src, conv = null;
  if (OPS[op].needs === 'gray' && src.channels() > 1) { conv = new cv.Mat(); cv.cvtColor(src, conv, cv.COLOR_RGB2GRAY); input = conv; }
  const ctx = {};
  try {
    const out = OPS[op].apply(cv, input, params, ctx);
    const h = gsha(Buffer.from(out.data), `${out.rows}x${out.cols}t${out.type()}`, JSON.stringify(ctx.info ?? null), JSON.stringify(ctx.values ?? null));
    out.delete(); return h;
  } catch (e) { return 'ERR:' + (e && e.message ? e.message : typeof e === 'number' ? 'cvexc' : String(e)).slice(0, 60); } finally { if (conv) conv.delete(); }
}
function goldenPy(op, params) {
  const pp = { mode: op }; for (const [k, v] of Object.entries(params)) pp[`${op}__${k}`] = v;
  return gsha(buildPythonCode([{ id: OP_TO_FILTER[op], params: pp, nodeId: 1 }]));
}
function goldenAll(variant) {
  const imgs = goldenImages(), res = {};
  try {
    for (const [op, p0] of G_CASES) {
      const p = variant === 'A' ? p0 : variant === 'B' ? { ...p0, ...G_DEF[op] } : { ...p0, ...G_WEIRD[op] };
      const key = `${op}:${JSON.stringify(p0)}`;
      for (const [n, m] of Object.entries(imgs)) res[`${key}|${n}`] = goldenRun(op, p, m);
      res[`${key}|PY`] = goldenPy(op, p);
      const pr = OP_PRIMARY[op]; res[`${key}|UI`] = pr ? gsha(String(pr(p))) : 'none';
    }
  } finally { for (const m of Object.values(imgs)) m.delete(); }
  return res;
}
for (const [variant, what] of [['A', 'the old params'], ['B', 'every new param passed explicitly at its default'], ['C', 'every new select set to an unknown value / "constructor" and every new number to junk']]) {
  test(`ip4a-G${variant}. golden: the 10 existing ops give byte-identical image + ctx.info + ctx.values + exported Python + card text on 4 deterministic images (${what}) -- 320 hashes measured on HEAD 1755233 before any edit`, () => {
    const cur = goldenAll(variant);
    assert.equal(Object.keys(GOLDEN).length, 320);
    const bad = Object.keys(GOLDEN).filter((k) => cur[k] !== GOLDEN[k]);
    assert.deepEqual(bad, [], `${bad.length} of 320 differ: ${bad.slice(0, 5).join(' ; ')}`);
    assert.ok(!Object.values(GOLDEN).some((v) => v.startsWith('ERR:')), 'no golden case throws on the old code');
  });
}

// ===================================================================
// psfKernel (ตัวช่วยกลาง) + custom_kernel motion / disk
// ===================================================================
test('ip4a-psf1. psfKernel: sum = 1 (±1e-12), point symmetric (k[i] === k[n*n-1-i]), odd size, centre >= every other cell on the line, no all-zero outer ring -- every motion L 1..50 x theta {0,1,15,30,45,60,89,90,91,135,179} and every disk R 1..50', () => {
  const THETAS = [0, 1, 15, 30, 45, 60, 89, 90, 91, 135, 179];
  const chk = (psf, label) => {
    const { n, k } = psf;
    assert.equal(n % 2, 1, `${label}: odd`);
    assert.equal(k.length, n * n);
    near(k.reduce((s, v) => s + v, 0), 1, 1e-12, `${label}: sum`);
    for (let i = 0; i < k.length; i++) assert.equal(k[i], k[n * n - 1 - i], `${label}: symmetry at ${i}`);
    if (n > 1) {
      let ring = 0;
      for (let i = 0; i < n; i++) ring += k[i] + k[(n - 1) * n + i] + k[i * n] + k[i * n + n - 1];
      assert.ok(ring > 0, `${label}: outer ring is all zero (should have been trimmed)`);
    }
  };
  for (let L = 1; L <= 50; L++) for (const t of THETAS) chk(psfKernel('motion', L, t), `motion L${L} ${t}`);
  for (let R = 1; R <= 50; R++) {
    const p = psfKernel('disk', R, 0);
    chk(p, `disk R${R}`);
    assert.equal(p.n, 2 * Math.ceil(R - 0.5) + 1, `disk R${R}: size 2*ceil(R-0.5)+1`);
  }
  assert.equal(PSF_SS, 16);
});

test('ip4a-psf2. psfKernel: L = 1 -> [1] (1x1) for every angle · theta 0 has values only in the centre row, theta 90 only in the centre column · disk R = 5 is 11x11 like fspecial · a longer line spreads over a bigger kernel · 45 degrees equals the mirrored 135 degrees', () => {
  for (const t of [0, 33, 90, 179]) { const p = psfKernel('motion', 1, t); assert.equal(p.n, 1); assert.deepEqual(Array.from(p.k), [1]); }
  const h = psfKernel('motion', 9, 0), mid = (h.n - 1) / 2;
  for (let y = 0; y < h.n; y++) for (let x = 0; x < h.n; x++) if (y !== mid) assert.equal(h.k[y * h.n + x], 0, `theta 0 off-row (${x},${y})`);
  assert.equal(h.n, 9, 'a 9-long horizontal line -> 9 cells wide (fspecial motion 9 0 is 1x9 too)');
  let rowSum = 0; for (let x = 0; x < h.n; x++) rowSum += h.k[mid * h.n + x];
  near(rowSum, 1, 1e-12, 'row sum');
  for (let x = 0; x < h.n; x++) assert.ok(h.k[mid * h.n + x] > 0, `centre row cell ${x} is covered`);
  const v = psfKernel('motion', 9, 90);
  for (let y = 0; y < v.n; y++) for (let x = 0; x < v.n; x++) if (x !== (v.n - 1) / 2) assert.equal(v.k[y * v.n + x], 0, `theta 90 off-column (${x},${y})`);
  assert.equal(psfKernel('disk', 5, 0).n, 11);
  assert.ok(psfKernel('motion', 30, 0).n > psfKernel('motion', 10, 0).n);
  const a = psfKernel('motion', 15, 45), b = psfKernel('motion', 15, 135);
  assert.equal(a.n, b.n);
  for (let y = 0; y < a.n; y++) for (let x = 0; x < a.n; x++) near(a.k[y * a.n + x], b.k[y * a.n + (a.n - 1 - x)], 1e-12, `45 vs mirrored 135 at ${x},${y}`);
});

test('ip4a-psf3. custom_kernel motion / disk on a single white dot on black: the result around the dot is the kernel x 255 (filter2D rounding, +-1) at the right positions and 0 everywhere else; info text; preset "constructor" no longer throws', () => {
  for (const [preset, psfSize, psfAngle] of [['motion', 9, 0], ['motion', 7, 30], ['disk', 5, 0], ['motion', 1, 0]]) {
    const dot = gray(41, 41, (x, y) => (x === 20 && y === 20 ? 255 : 0));
    const { out, ctx } = runOp('custom_kernel', dot, { preset, psfSize, psfAngle, amount: 1 });
    try {
      const psf = psfKernel(preset, psfSize, psfAngle), h = (psf.n - 1) / 2;
      let maxd = 0, cells = 0;
      for (let y = 0; y < 41; y++) for (let x = 0; x < 41; x++) {
        const dx = x - 20, dy = y - 20;
        const inK = Math.abs(dx) <= h && Math.abs(dy) <= h;
        // filter2D is a correlation: out(x) = sum k(i) src(x + i - c) -> with a point source out(20 + d) = k(c - d) ; every kernel here is point symmetric
        const want = inK ? Math.fround(Math.fround(psf.k[(h - dy) * psf.n + (h - dx)]) * 255) : 0;
        const got = out.data[y * 41 + x];
        maxd = Math.max(maxd, Math.abs(got - want)); if (want > 0) cells++;
      }
      assert.ok(maxd <= 1, `${preset} ${psfSize}/${psfAngle}: max |got - kernel*255| = ${maxd}`);
      assert.ok(cells >= 1);
      const infoWant = preset === 'motion' ? `Kernel: motion (${psf.n}x${psf.n}) · ยาว ${psfSize} px มุม ${psfAngle}°` : `Kernel: disk (${psf.n}x${psf.n}) · รัศมี ${psfSize} px`;
      assert.equal(ctx.info, infoWant);
    } finally { out.delete(); dot.delete(); }
  }
  const g = noiseGray(20, 15, 3);
  try {
    const sharp = runOp('custom_kernel', g, { preset: 'sharpen' }), weird = runOp('custom_kernel', g, { preset: 'constructor' });
    sameBytes(sharp.out, weird.out, 'constructor = sharpen');
    assert.equal(sharp.ctx.info, 'Kernel: sharpen (3x3)');
    for (const r of [sharp, weird]) r.out.delete();
    for (const preset of ['toString', '__proto__', 'hasOwnProperty']) { const r = runOp('custom_kernel', g, { preset }); r.out.delete(); }
  } finally { g.delete(); }
});

test('ip4a-psf4. custom_kernel motion / disk: amount mixes with the original (0 = original, 1 = filtered, 0.5 = between); psfSize / psfAngle are clamped to 1..50 / 0..179 (junk -> defaults 9 / 0); 0x0 -> clone + "ภาพว่าง" for the PSF presets only; 1x1 with L=50 does not throw; src untouched; RGBA works', () => {
  const g = noiseGray(30, 20, 9), e = new cv.Mat(), one = gray(1, 1, () => 77), a = toRGBA(noiseRGB(16, 12, 4));
  try {
    const before = bytes(g);
    const r0 = runOp('custom_kernel', g, { preset: 'motion', amount: 0 }), r1 = runOp('custom_kernel', g, { preset: 'motion', amount: 1 }), rh = runOp('custom_kernel', g, { preset: 'motion', amount: 0.5 });
    sameBytes(r0.out, g, 'amount 0 = original');
    for (let i = 0; i < g.data.length; i++) { const lo = Math.min(g.data[i], r1.out.data[i]), hi = Math.max(g.data[i], r1.out.data[i]); assert.ok(rh.out.data[i] >= lo - 1 && rh.out.data[i] <= hi + 1, 'amount 0.5 lies between'); }
    assert.deepEqual(bytes(g), before, 'src untouched');
    for (const r of [r0, r1, rh]) r.out.delete();
    const dflt = runOp('custom_kernel', g, { preset: 'motion' }), junk = runOp('custom_kernel', g, { preset: 'motion', psfSize: 'x', psfAngle: 'y' });
    sameBytes(dflt.out, junk.out, 'junk psfSize / psfAngle = defaults 9 / 0'); assert.equal(dflt.ctx.info, junk.ctx.info);
    assert.match(dflt.ctx.info, /ยาว 9 px มุม 0°/);
    const big = runOp('custom_kernel', g, { preset: 'motion', psfSize: 999, psfAngle: 999 }); assert.match(big.ctx.info, /ยาว 50 px มุม 179°/);
    const neg = runOp('custom_kernel', g, { preset: 'disk', psfSize: -4 }); assert.match(neg.ctx.info, /รัศมี 1 px/);
    for (const r of [dflt, junk, big, neg]) r.out.delete();
    for (const preset of ['motion', 'disk']) {
      const z = runOp('custom_kernel', e, { preset }); assert.equal(z.out.rows, 0); assert.notEqual(z.out.$$.ptr, e.$$.ptr); assert.equal(z.ctx.info, 'ภาพว่าง'); z.out.delete();
      const o1 = runOp('custom_kernel', one, { preset, psfSize: 50 }); assert.equal(o1.out.rows, 1); o1.out.delete();
      const ra = runOp('custom_kernel', a, { preset }); assert.equal(ra.out.channels(), 4); ra.out.delete();
    }
    assert.throws(() => runOp('custom_kernel', e, { preset: 'sharpen' }), 'the old path on 0x0 keeps its old behaviour (no new guard)');
  } finally { for (const m of [g, e, one, a]) m.delete(); }
});

// ===================================================================
// gaussian_noise speckle
// ===================================================================
test('ip4a-sp1. gaussian_noise speckle: black pixels never change; per pixel exactly floor(d + d*(sigma/255)*z + 0.5) with the same z as the additive kind (seededGaussian, same order); same seed -> same image; different seed -> different; info text; sigma 0 = original + the additive text', () => {
  const src = rgb(23, 17, (x, y) => [x * 11 & 255, (y * 15) & 255, x === 3 ? 0 : 200]);
  const dark = black(30, 20);
  try {
    const { out, ctx } = runOp('gaussian_noise', src, { kind: 'speckle', sigma: 40, seed: 5 });
    const n = src.data.length, z = seededGaussian(5, n), s = 40 / 255;
    let clipped = 0;
    for (let k = 0; k < n; k++) {
      const v = Math.floor(src.data[k] + src.data[k] * s * z[k] + 0.5);
      const want = v < 0 ? 0 : v > 255 ? 255 : v;
      if (v < 0 || v > 255) clipped++;
      assert.equal(out.data[k], want, `pixel ${k}`);
      if (src.data[k] === 0) assert.equal(out.data[k], 0, 'a black value stays black');
    }
    assert.equal(ctx.info, `เติม speckle noise (แรงตามความสว่าง) σ=40 ที่ความสว่าง 255 (seed 5) · ชนขอบ 0/255 ${(Math.floor(clipped * 100 / n * 10 + 0.5) / 10).toFixed(1)}% ของค่า`);
    const again = runOp('gaussian_noise', src, { kind: 'speckle', sigma: 40, seed: 5 }), other = runOp('gaussian_noise', src, { kind: 'speckle', sigma: 40, seed: 6 });
    sameBytes(out, again.out, 'same seed'); assert.notDeepEqual(bytes(out), bytes(other.out));
    const add = runOp('gaussian_noise', src, { kind: 'additive', sigma: 40, seed: 5 });
    assert.notDeepEqual(bytes(out), bytes(add.out), 'speckle differs from additive');
    const zero = runOp('gaussian_noise', src, { kind: 'speckle', sigma: 0 }), zeroAdd = runOp('gaussian_noise', src, { sigma: 0 });
    sameBytes(zero.out, src, 'sigma 0 = original'); assert.equal(zero.ctx.info, zeroAdd.ctx.info); assert.equal(zero.ctx.info, 'σ = 0 — ไม่เติม noise (ภาพเดิม)');
    const bl = runOp('gaussian_noise', dark, { kind: 'speckle', sigma: 100 }); sameBytes(bl.out, dark, 'an all-black image is untouched at sigma 100');
    for (const r of [again, other, add, zero, zeroAdd, bl]) r.out.delete();
    out.delete();
    for (const kind of ['constructor', 'zzz', undefined, 3]) { const r = runOp('gaussian_noise', src, { kind, sigma: 40, seed: 5 }); const base = runOp('gaussian_noise', src, { sigma: 40, seed: 5 }); sameBytes(r.out, base.out, `kind ${String(kind)} = additive`); assert.equal(r.ctx.info, base.ctx.info); r.out.delete(); base.out.delete(); }
  } finally { src.delete(); dark.delete(); }
});

// ===================================================================
// gamma log
// ===================================================================
test('ip4a-log1. gamma curve log: LUT 0 -> 0, 255 -> 255, strictly rising (255 distinct levels... non-decreasing, and above the identity for 1..254), every value == floor(log1p(i) * 255/ln(256) + 0.5); the info reports lut[10] / lut[128]; gamma value ignored; power default unchanged', () => {
  const ramp = gray(256, 1, (x) => x);
  try {
    const { out, ctx } = runOp('gamma', ramp, { curve: 'log', g: 2.5 });
    const lut = Array.from(out.data);
    assert.equal(lut[0], 0); assert.equal(lut[255], 255);
    for (let i = 0; i < 256; i++) {
      assert.equal(lut[i], Math.floor(Math.log1p(i) * (255 / Math.log(256)) + 0.5), `lut[${i}]`);
      if (i > 0) assert.ok(lut[i] >= lut[i - 1], 'non-decreasing');
      if (i > 0 && i < 255) assert.ok(lut[i] > i, `log lifts the dark and mid range: lut[${i}] = ${lut[i]}`);
    }
    assert.equal(ctx.info, `Log transform: ใหม่ = 255·ln(1+เดิม)/ln(256) — เดิม 10 → ${lut[10]}, 128 → ${lut[128]} (ค่า γ ไม่ถูกใช้)`);
    const other = runOp('gamma', ramp, { curve: 'log', g: 0.3 }); sameBytes(out, other.out, 'g ignored'); other.out.delete();
    const pw = runOp('gamma', ramp, {}), pw2 = runOp('gamma', ramp, { curve: 'power' }), pw3 = runOp('gamma', ramp, { curve: 'constructor' });
    sameBytes(pw.out, pw2.out, 'power default'); sameBytes(pw.out, pw3.out, 'constructor = power');
    assert.notDeepEqual(bytes(pw.out), lut);
    for (const r of [pw, pw2, pw3]) r.out.delete();
    out.delete();
    const c = noiseRGB(10, 8, 2), a = toRGBA(c), e = new cv.Mat();
    for (const src of [c, a]) { const r = runOp('gamma', src, { curve: 'log' }); assert.equal(r.out.channels(), src.channels()); for (let i = 0; i < src.data.length; i++) if (i % src.channels() !== 3 || src.channels() === 3) assert.equal(r.out.data[i], lut[src.data[i]]); r.out.delete(); }
    const z = runOp('gamma', e, { curve: 'log' }); assert.equal(z.out.rows, 0); assert.equal(z.ctx.info, 'ภาพว่าง'); z.out.delete();
    for (const m of [c, a, e]) m.delete();
  } finally { ramp.delete(); }
});

// ===================================================================
// otsu triangle
// ===================================================================
test('ip4a-tri1. otsu_threshold method triangle: output and threshold == cv.threshold(THRESH_TRIANGLE) exactly; thresholdMarker reads the same number from the info; invert; constant images 0 / 128 / 255 (threshold = value - 1, no throw); 1x1; unknown method = otsu; 0x0 clone', () => {
  const imgs = [noiseGray(60, 40, 3), blobGray(), gray(50, 40, (x) => (x < 5 ? 255 : 30))];
  try {
    for (const g of imgs) for (const invert of [false, true]) {
      const { out, ctx } = runOp('otsu_threshold', g, { method: 'triangle', invert });
      const ref = new cv.Mat();
      const t = cv.threshold(g, ref, 0, 255, (invert ? cv.THRESH_BINARY_INV : cv.THRESH_BINARY) + cv.THRESH_TRIANGLE);
      sameBytes(out, ref, `triangle invert=${invert}`);
      assert.equal(ctx.info, `ค่าเกณฑ์ที่ Triangle คำนวณได้ = ${Math.round(t)}`);
      assert.equal(thresholdMarker('otsu_threshold', {}, [ctx.info]), Math.round(t));
      out.delete(); ref.delete();
    }
    const tri = runOp('otsu_threshold', imgs[2], { method: 'triangle' }), ots = runOp('otsu_threshold', imgs[2], {});
    assert.notEqual(tri.ctx.info, ots.ctx.info); tri.out.delete(); ots.out.delete();
    // single-valued image: cv.threshold(TRIANGLE) returns v - 1 (v = 0 gives 1 and v = 1 gives -1 -- OpenCV quirks, identical in cv2); never throws
    const TRI_CONST = { 0: 1, 1: -1, 128: 127, 254: 253, 255: 254 };
    for (const v of [0, 1, 128, 254, 255]) {
      const c = gray(10, 8, () => v);
      const r = runOp('otsu_threshold', c, { method: 'triangle' });
      assert.equal(r.ctx.info, `ค่าเกณฑ์ที่ Triangle คำนวณได้ = ${TRI_CONST[v]}`, `constant ${v}`);
      assert.equal(thresholdMarker('otsu_threshold', {}, [r.ctx.info]), Math.max(0, TRI_CONST[v]), `marker ${v}`);
      const ref = new cv.Mat(); cv.threshold(c, ref, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE); sameBytes(r.out, ref, `constant ${v}`);
      r.out.delete(); ref.delete(); c.delete();
    }
    const one = gray(1, 1, () => 100), r1 = runOp('otsu_threshold', one, { method: 'triangle' }); assert.equal(r1.out.rows, 1); r1.out.delete(); one.delete();
    const weird = runOp('otsu_threshold', imgs[0], { method: 'constructor' }), plain = runOp('otsu_threshold', imgs[0], {});
    sameBytes(weird.out, plain.out, 'constructor = otsu'); assert.equal(weird.ctx.info, plain.ctx.info); assert.match(plain.ctx.info, /^ค่าเกณฑ์ที่ Otsu คำนวณได้ = \d+$/);
    weird.out.delete(); plain.out.delete();
    const e = new cv.Mat(), z = runOp('otsu_threshold', e, { method: 'triangle' }); assert.equal(z.out.rows, 0); assert.equal(z.ctx.info, 'ภาพว่าง'); z.out.delete(); e.delete();
  } finally { for (const m of imgs) m.delete(); }
});

// ===================================================================
// fft_filter butterworth
// ===================================================================
// ตัวคำนวณอ้างอิงอิสระในเทสต์: ทำ pipeline เดียวกับ op (pad -> dft -> fftshift -> คูณ mask -> ifftshift -> idft -> normalize 0..255) ด้วยสูตร mask ของเทสต์เอง
function refFftFilter(g, type, cutoff, maskFn) {
  const R = g.rows, C = g.cols;
  assert.ok(R % 2 === 0 && C % 2 === 0 && cv.getOptimalDFTSize(R) === R && cv.getOptimalDFTSize(C) === C, 'reference needs an optimal even size');
  const f = new cv.Mat(); g.convertTo(f, cv.CV_32F);
  const zeros = cv.Mat.zeros(R, C, cv.CV_32F), planes = new cv.MatVector(); planes.push_back(f); planes.push_back(zeros);
  const cplx = new cv.Mat(), dftM = new cv.Mat(), inv = new cv.Mat(), normed = new cv.Mat(), u8 = new cv.Mat();
  cv.merge(planes, cplx); cv.dft(cplx, dftM, cv.DFT_COMPLEX_OUTPUT);
  const d = dftM.data32F, cy = R / 2, cx = C / 2, out = new Float32Array(d.length);
  for (let y = 0; y < R; y++) for (let x = 0; x < C; x++) {
    // fftshift then mask(dist from centre) then ifftshift == multiply the un-shifted spectrum by the mask evaluated at the shifted position
    const sx = (x + C / 2) % C, sy = (y + R / 2) % R;
    const low = maskFn(Math.sqrt((sx - cx) * (sx - cx) + (sy - cy) * (sy - cy)), cutoff);
    const m = Math.fround(type === 'high' ? 1 - low : low);
    out[(y * C + x) * 2] = d[(y * C + x) * 2] * m; out[(y * C + x) * 2 + 1] = d[(y * C + x) * 2 + 1] * m;
  }
  const filt = cv.matFromArray(R, C, cv.CV_32FC2, out);
  cv.dft(filt, inv, cv.DFT_INVERSE + cv.DFT_SCALE + cv.DFT_REAL_OUTPUT);
  cv.normalize(inv, normed, 0, 255, cv.NORM_MINMAX); normed.convertTo(u8, cv.CV_8U);
  for (const m of [f, zeros, planes, cplx, dftM, inv, normed, filt]) m.delete();
  return u8;
}
test('ip4a-bw1. fft_filter butterworth: matches an independent reference of the same pipeline with H = 1/(1+(D/D0)^2n) (<= 1 level; both low and high; n = 1, 2, 5, 10; cutoff 12 and 30); H(D0) = 0.5; info "(butterworth n=2)"; n=1 vs n=10 differ; butterworth differs from gaussian and ideal; unknown order = 2; src untouched', () => {
  const g = noiseGray(64, 64, 8), smooth = gray(64, 64, (x, y) => (x < 32 ? 40 : 200) + ((y * 3) & 15));
  try {
    for (const img of [g, smooth]) for (const type of ['low', 'high']) for (const [order, cutoff] of [[1, 12], [2, 30], [5, 12], [10, 30]]) {
      const { out, ctx } = runOp('fft_filter', img, { type, shape: 'butterworth', order, cutoff });
      const ref = refFftFilter(img, type, cutoff, (dd, c0) => 1 / (1 + Math.pow(dd / c0, 2 * order)));
      let md = 0; for (let i = 0; i < ref.data.length; i++) md = Math.max(md, Math.abs(ref.data[i] - out.data[i]));
      assert.ok(md <= 1, `${type} n=${order} D0=${cutoff}: max diff ${md}`);
      assert.equal(ctx.info, `${type === 'high' ? 'High' : 'Low'}-pass (butterworth n=${order}), รัศมี ${cutoff} (หน่วยความถี่ นับจากกลางสเปกตรัม — ไม่ใช่พิกเซลของภาพจริง)`);
      out.delete(); ref.delete();
    }
    near(1 / (1 + Math.pow(1, 2 * 3)), 0.5, 0, 'H(D0) = 0.5 for any n');
    const before = bytes(g);
    const a = runOp('fft_filter', g, { shape: 'butterworth', order: 1, cutoff: 12 }), b = runOp('fft_filter', g, { shape: 'butterworth', order: 10, cutoff: 12 });
    assert.notDeepEqual(bytes(a.out), bytes(b.out));
    const ga = runOp('fft_filter', g, { shape: 'gaussian', cutoff: 12 }), id = runOp('fft_filter', g, { shape: 'ideal', cutoff: 12 });
    assert.notDeepEqual(bytes(a.out), bytes(ga.out)); assert.notDeepEqual(bytes(b.out), bytes(id.out));
    const dflt = runOp('fft_filter', g, { shape: 'butterworth', cutoff: 12 }), two = runOp('fft_filter', g, { shape: 'butterworth', order: 2, cutoff: 12 }), junk = runOp('fft_filter', g, { shape: 'butterworth', order: 'zz', cutoff: 12 });
    sameBytes(dflt.out, two.out, 'default order 2'); sameBytes(junk.out, two.out, 'junk order = 2'); assert.match(dflt.ctx.info, /butterworth n=2\)/);
    const hi = runOp('fft_filter', g, { shape: 'butterworth', order: 99, cutoff: 12 }), ten = runOp('fft_filter', g, { shape: 'butterworth', order: 10, cutoff: 12 }); sameBytes(hi.out, ten.out, 'order clamped to 10'); assert.match(hi.ctx.info, /n=10\)/);
    assert.deepEqual(bytes(g), before);
    for (const r of [a, b, ga, id, dflt, two, junk, hi, ten]) r.out.delete();
    const weird = runOp('fft_filter', g, { shape: 'constructor', cutoff: 12 }), gauss = runOp('fft_filter', g, { shape: 'gaussian', cutoff: 12 }); sameBytes(weird.out, gauss.out, 'unknown shape = gaussian'); weird.out.delete(); gauss.out.delete();
  } finally { g.delete(); smooth.delete(); }
});

// ===================================================================
// white_balance white_patch / ref_point
// ===================================================================
test('ip4a-wb1. white_balance white_patch: an image whose B channel was scaled by 0.8 has its 99th-percentile value 255 in every channel afterwards; gains / info exact; gray_world default and unknown method stay the old path; gray input keeps the old path for every method; RGBA -> 3 channels; src untouched', () => {
  const r = lcg(5);
  const base = rgb(60, 40, () => [Math.floor(r() * 256), Math.floor(r() * 256), Math.floor(r() * 256)]);
  const tinted = new cv.Mat(); base.copyTo(tinted);
  for (let i = 2; i < tinted.data.length; i += 3) tinted.data[i] = Math.floor(tinted.data[i] * 0.8);
  try {
    const before = bytes(tinted);
    const { out, ctx } = runOp('white_balance', tinted, { method: 'white_patch' });
    assert.deepEqual(bytes(tinted), before, 'src untouched');
    const N = 60 * 40, k = Math.floor((N * 99 + 99) / 100);
    for (let c = 0; c < 3; c++) {
      const h = new Array(256).fill(0); for (let i = 0; i < N; i++) h[out.data[i * 3 + c]]++;
      // the new 99% value of every channel is (almost exactly) 255
      let cum = 0, v99 = 0; for (; v99 < 255; v99++) { cum += h[v99]; if (cum >= k) break; }
      assert.ok(v99 >= 254, `channel ${c}: p99 after = ${v99}`);
    }
    const v = [0, 1, 2].map((c) => { const h = new Array(256).fill(0); for (let i = 0; i < N; i++) h[tinted.data[i * 3 + c]]++; let cum = 0, q = 0; for (; q < 255; q++) { cum += h[q]; if (cum >= k) break; } return q; });
    const g = v.map((x) => (x > 0 ? 255 / x : 1));
    const f2 = (x) => (Math.floor(x * 100 + 0.5) / 100).toFixed(2);
    assert.equal(ctx.info, `White Patch: ค่าที่ 99% R/G/B = ${v[0]}/${v[1]}/${v[2]} → เกน R×${f2(g[0])} G×${f2(g[1])} B×${f2(g[2])}`);
    for (let i = 0; i < tinted.data.length; i++) assert.equal(out.data[i], Math.min(255, Math.floor(tinted.data[i] * g[i % 3] + 0.5)), `pixel ${i}`);
    out.delete();
    const gw = runOp('white_balance', tinted, {}), gw2 = runOp('white_balance', tinted, { method: 'gray_world' }), gw3 = runOp('white_balance', tinted, { method: 'constructor' });
    sameBytes(gw.out, gw2.out, 'gray_world'); sameBytes(gw.out, gw3.out, 'unknown method'); assert.equal(gw.ctx.info, gw3.ctx.info); assert.match(gw.ctx.info, /^เกน R×/);
    for (const q of [gw, gw2, gw3]) q.out.delete();
    const mono = noiseGray(20, 10, 3);
    for (const method of ['gray_world', 'white_patch', 'ref_point']) { const m = runOp('white_balance', mono, { method }); assert.equal(m.out.channels(), 3); assert.equal(m.ctx.info, 'ภาพขาวดำ ไม่มีสีให้ปรับสมดุล'); for (let i = 0; i < 200; i++) { assert.equal(m.out.data[i * 3], mono.data[i]); } m.out.delete(); }
    mono.delete();
    const a = toRGBA(tinted), ra = runOp('white_balance', a, { method: 'white_patch' }), rb = runOp('white_balance', tinted, { method: 'white_patch' });
    assert.equal(ra.out.channels(), 3); sameBytes(ra.out, rb.out, 'RGBA = RGB'); assert.equal(ra.ctx.info, rb.ctx.info);
    for (const q of [ra, rb]) q.out.delete(); a.delete();
    // percentile hits k exactly: 99 px at 10 and 1 px at 200 -> k = 99, cumsum(10) = 99 >= k -> v99 = 10 (a '>' instead of '>=' would give 200)
    const exact = rgb(10, 10, (x, y) => (x === 9 && y === 9 ? [200, 200, 200] : [10, 10, 10])), re = runOp('white_balance', exact, { method: 'white_patch' });
    assert.equal(re.ctx.info, 'White Patch: ค่าที่ 99% R/G/B = 10/10/10 → เกน R×25.50 G×25.50 B×25.50'); assert.equal(re.out.data[0], 255); re.out.delete(); exact.delete();
    const dark = rgb(10, 10, () => [0, 0, 0]), rd = runOp('white_balance', dark, { method: 'white_patch' }); assert.match(rd.ctx.info, /ค่าที่ 99% R\/G\/B = 0\/0\/0 → เกน R×1\.00 G×1\.00 B×1\.00/); rd.out.delete(); dark.delete();
    const e = new cv.Mat(); for (const method of ['white_patch', 'ref_point']) { const z = runOp('white_balance', e, { method }); assert.equal(z.out.rows, 0); assert.equal(z.ctx.info, 'ภาพว่าง'); z.out.delete(); } e.delete();
  } finally { base.delete(); tinted.delete(); }
});

test('ip4a-wb2. white_balance ref_point: a grey card dyed by a colour cast has its reference region averaged back to grey (|R-G|, |G-B| <= 1); box clipped at the image edge; centre / gains / info exact; dark reference warns; src untouched; refSize / refX clamps', () => {
  // scene: noisy colourful background + a 20x20 grey card (value 120) dyed by gains (1.3, 1.0, 0.7)
  const r = lcg(2);
  const scene = rgb(80, 60, (x, y) => (x >= 30 && x < 50 && y >= 20 && y < 40 ? [Math.min(255, Math.round(120 * 1.3)), 120, Math.round(120 * 0.7)] : [Math.floor(r() * 256), Math.floor(r() * 256), Math.floor(r() * 256)]));
  try {
    const before = bytes(scene);
    const { out, ctx } = runOp('white_balance', scene, { method: 'ref_point', refX: 50, refY: 50, refSize: 5 });
    assert.deepEqual(bytes(scene), before);
    const cx = Math.floor(50 / 100 * 79 + 0.5), cy = Math.floor(50 / 100 * 59 + 0.5);
    let sr = 0, sg = 0, sb = 0, n = 0;
    for (let y = cy - 5; y <= cy + 5; y++) for (let x = cx - 5; x <= cx + 5; x++) { const o = (y * 80 + x) * 3; sr += out.data[o]; sg += out.data[o + 1]; sb += out.data[o + 2]; n++; }
    const m = [sr / n, sg / n, sb / n];
    assert.ok(Math.abs(m[0] - m[1]) <= 1 && Math.abs(m[1] - m[2]) <= 1, `reference region after: ${m}`);
    const f1 = (x) => (Math.floor(x * 10 + 0.5) / 10).toFixed(1), f2 = (x) => (Math.floor(x * 100 + 0.5) / 100).toFixed(2);
    const mean = [156, 120, 84], t = (156 + 120 + 84) / 3, gains = mean.map((v) => t / v);
    assert.equal(ctx.info, `จุดอ้างอิง (${cx}, ${cy}) px รัศมี 5 · สีเฉลี่ย R/G/B = ${f1(mean[0])}/${f1(mean[1])}/${f1(mean[2])} → เกน R×${f2(gains[0])} G×${f2(gains[1])} B×${f2(gains[2])}`);
    for (let i = 0; i < scene.data.length; i++) assert.equal(out.data[i], Math.min(255, Math.floor(scene.data[i] * gains[i % 3] + 0.5)));
    out.delete();
    // corner reference: the box is clipped at the image edge (4 x 4 pixels for refSize 3 at (0,0)) · refSize clamps 1..50
    const c0 = runOp('white_balance', scene, { method: 'ref_point', refX: 0, refY: 0, refSize: 3 });
    const mm = [0, 1, 2].map((c) => { let s = 0; for (let y = 0; y <= 3; y++) for (let x = 0; x <= 3; x++) s += scene.data[(y * 80 + x) * 3 + c]; return s / 16; });
    assert.match(c0.ctx.info, new RegExp(`^จุดอ้างอิง \\(0, 0\\) px รัศมี 3 · สีเฉลี่ย R/G/B = ${f1(mm[0])}/${f1(mm[1])}/${f1(mm[2])}`));
    c0.out.delete();
    const big = runOp('white_balance', scene, { method: 'ref_point', refSize: 9999 }), tiny = runOp('white_balance', scene, { method: 'ref_point', refSize: -3 });
    assert.match(big.ctx.info, /รัศมี 50/); assert.match(tiny.ctx.info, /รัศมี 1 /);
    for (const q of [big, tiny]) q.out.delete();
    const edge = runOp('white_balance', scene, { method: 'ref_point', refX: 100, refY: 100, refSize: 4 }); assert.match(edge.ctx.info, /^จุดอ้างอิง \(79, 59\) px/);
    { // box clipped at the last row / column: x 75..79, y 55..59 (25 px), not a 9x9 box reaching outside
      const me = [0, 1, 2].map((c) => { let s = 0; for (let y = 55; y <= 59; y++) for (let x = 75; x <= 79; x++) s += scene.data[(y * 80 + x) * 3 + c]; return s / 25; }), te = (me[0] + me[1] + me[2]) / 3, ge = me.map((v) => te / v);
      assert.equal(edge.ctx.info.split(' → ')[0], `จุดอ้างอิง (79, 59) px รัศมี 4 · สีเฉลี่ย R/G/B = ${f1(me[0])}/${f1(me[1])}/${f1(me[2])}`);
      assert.ok(edge.ctx.info.includes(`เกน R×${f2(ge[0])} G×${f2(ge[1])} B×${f2(ge[2])}`));
    }
    edge.out.delete();
    const junk = runOp('white_balance', scene, { method: 'ref_point', refX: 'q', refY: 'abc' }), dflt = runOp('white_balance', scene, { method: 'ref_point' }); assert.equal(junk.ctx.info, dflt.ctx.info); junk.out.delete(); dflt.out.delete();
    const dark = rgb(20, 20, () => [4, 6, 3]), rd = runOp('white_balance', dark, { method: 'ref_point' });
    assert.ok(rd.ctx.info.endsWith(' · จุดอ้างอิงมืดมาก ผลอาจเพี้ยน — เลือกบริเวณเทาหรือขาว')); rd.out.delete();
    const black0 = rgb(20, 20, () => [0, 0, 0]), rb = runOp('white_balance', black0, { method: 'ref_point' }); assert.match(rb.ctx.info, /เกน R×1\.00 G×1\.00 B×1\.00/); sameBytes(rb.out, black0, 'black stays black'); rb.out.delete();
    dark.delete(); black0.delete();
  } finally { scene.delete(); }
});

// ===================================================================
// colorspace xyz / luv / yuv
// ===================================================================
test('ip4a-cs1. colorspace xyz / luv / yuv == cv.cvtColor(RGB2XYZ / RGB2Luv / RGB2YUV) exactly (colour and gray input); the 4 old spaces unchanged; the card text is XYZ / LUV / YUV; unknown space still throws like before', () => {
  const c = noiseRGB(33, 21, 6), g = noiseGray(33, 21, 7);
  try {
    for (const [space, code] of [['xyz', cv.COLOR_RGB2XYZ], ['luv', cv.COLOR_RGB2Luv], ['yuv', cv.COLOR_RGB2YUV]]) {
      for (const src of [c, g]) {
        const rgbSrc = src.channels() === 1 ? rgbFromGray(src) : src, ref = new cv.Mat(); cv.cvtColor(rgbSrc, ref, code);
        const { out } = runOp('colorspace', src, { space }); sameBytes(out, ref, `${space} ${src.channels()}ch`);
        out.delete(); ref.delete(); if (rgbSrc !== src) rgbSrc.delete();
      }
      assert.equal(OP_PRIMARY.colorspace({ space }), `BGR → ${space.toUpperCase()}`);
    }
    assert.deepEqual(OPS.colorspace.params[0].options.map((o) => o.value), ['hsv', 'hls', 'lab', 'ycrcb', 'xyz', 'luv', 'yuv']);
    assert.throws(() => runOp('colorspace', c, { space: 'constructor' }));
  } finally { c.delete(); g.delete(); }
});

// ===================================================================
// mmPerPx
// ===================================================================
test('ip4a-mm1. mmPerPx: MM_COLUMNS / mmValue contract (area x f^2, length x f, null stays null); regionprops / feret / line_profile with f > 0 add exactly the new columns / fields at the end, every value = px value x f (area x f^2) exactly, info suffix; f = 0 / junk / negative leave columns + info unchanged; 0x0 and no-object paths carry the same keys', () => {
  assert.equal(mmValue(null, 0.5, 1), null); assert.equal(mmValue(undefined, 0.5, 2), null); assert.equal(mmValue(10, 0.5, 1), 5); assert.equal(mmValue(10, 0.5, 2), 2.5); assert.equal(mmValue(0, 0.5, 2), 0);
  assert.deepEqual(MM_COLUMNS.regionprops.map((x) => x[0]), ['area_mm2', 'perimeter_mm', 'equiv_diameter_mm', 'major_axis_mm', 'minor_axis_mm']);
  assert.deepEqual(MM_COLUMNS.feret.map((x) => x[0]), ['feret_max_mm', 'feret_min_mm', 'feret_len_mm']);
  assert.deepEqual(MM_COLUMNS.line_profile.map((x) => x[0]), ['line_length_mm', 'edge_pos_mm', 'edge_pair_mm']);
  const scene = objectsScene(), e = new cv.Mat(), c = noiseRGB(60, 40, 3), flat = rgb(60, 40, () => [90, 90, 90]), blk = black(40, 30);
  try {
    // regionprops
    const base = runOp('regionprops', scene, { minArea: 0 }), mm = runOp('regionprops', scene, { minArea: 0, mmPerPx: 0.25 });
    const bc = base.ctx.values.columns, mc = mm.ctx.values.columns;
    assert.equal(mc.length, bc.length + 5); assert.deepEqual(mc.slice(0, bc.length), bc); assert.deepEqual(mc.slice(bc.length), MM_COLUMNS.regionprops.map((x) => x[0]));
    assert.equal(mm.ctx.values.total, base.ctx.values.total);
    mm.ctx.values.rows.forEach((row, i) => {
      const br = base.ctx.values.rows[i];
      assert.deepEqual(row.slice(0, br.length), br);
      MM_COLUMNS.regionprops.forEach(([, src, pw], j) => assert.equal(row[br.length + j], br[bc.indexOf(src)] * (pw === 2 ? 0.25 * 0.25 : 0.25), `${src} -> mm`));
    });
    assert.equal(mm.ctx.info, base.ctx.info + ' · 1 px = 0.25 มม.');
    sameBytes(mm.out, base.out, 'image unchanged by mmPerPx');
    for (const f of [0, -2, 'abc', null, undefined, NaN]) { const q = runOp('regionprops', scene, { minArea: 0, mmPerPx: f }); assert.deepEqual(q.ctx.values, base.ctx.values); assert.equal(q.ctx.info, base.ctx.info); q.out.delete(); }
    const big = runOp('regionprops', scene, { minArea: 0, mmPerPx: 999 }); assert.equal(big.ctx.info, base.ctx.info + ' · 1 px = 100 มม.'); big.out.delete();
    const none = runOp('regionprops', blk, { mmPerPx: 0.5 }), noneBase = runOp('regionprops', blk, {}); assert.equal(none.ctx.values.columns.length, noneBase.ctx.values.columns.length + 5); assert.deepEqual(none.ctx.values.rows, []); assert.match(none.ctx.info, /^ไม่พบวัตถุ.* · 1 px = 0\.5 มม\.$/);
    const z = runOp('regionprops', e, { mmPerPx: 0.5 }); assert.equal(z.ctx.info, 'ภาพว่าง'); assert.equal(z.ctx.values.columns.length, bc.length + 5); assert.equal(z.ctx.values.total, 0);
    for (const q of [base, mm, none, noneBase, z]) q.out.delete();
    // feret
    const fb = runOp('feret', scene, { minArea: 0 }), fm = runOp('feret', scene, { minArea: 0, mmPerPx: 0.1 });
    assert.deepEqual(fm.ctx.values.columns, [...fb.ctx.values.columns, 'feret_max_mm', 'feret_min_mm', 'feret_len_mm']);
    fm.ctx.values.rows.forEach((row, i) => { const br = fb.ctx.values.rows[i]; assert.deepEqual(row.slice(0, br.length), br); [4, 5, 6].forEach((ix, j) => assert.equal(row[br.length + j], br[ix] * 0.1)); });
    assert.equal(fm.ctx.info, fb.ctx.info + ' · 1 px = 0.1 มม.');
    const fz = runOp('feret', e, { mmPerPx: 0.1 }); assert.equal(fz.ctx.values.columns.length, 13); assert.equal(fz.ctx.info, 'ภาพว่าง');
    const fn = runOp('feret', blk, { mmPerPx: 0.1 }); assert.equal(fn.ctx.values.columns.length, 13); assert.deepEqual(fn.ctx.values.rows, []);
    for (const q of [fb, fm, fz, fn]) q.out.delete();
    // line_profile
    const lb = runOp('line_profile', c, {}), lm = runOp('line_profile', c, { mmPerPx: 0.2 });
    assert.deepEqual(lm.ctx.values.fields.map((f) => f.key), [...lb.ctx.values.fields.map((f) => f.key), 'line_length_mm', 'edge_pos_mm', 'edge_pair_mm']);
    assert.deepEqual(lm.ctx.values.fields.slice(0, 9), lb.ctx.values.fields);
    const sb = scalarMap(lb.ctx), sm = scalarMap(lm.ctx);
    assert.equal(sm.line_length_mm, sb.line_length_px * 0.2); assert.equal(sm.edge_pos_mm, mmValue(sb.edge_pos_px, 0.2, 1)); assert.equal(sm.edge_pair_mm, mmValue(sb.edge_pair_px, 0.2, 1));
    assert.deepEqual(lm.ctx.values.fields.slice(9).map((f) => f.label), ['ความยาวเส้น (มม.)', 'ตำแหน่งขอบที่กระโดดแรง นับจาก A (มม.)', 'ระยะระหว่างขอบขึ้นกับขอบลง (มม.)']);
    assert.equal(lm.ctx.info, lb.ctx.info + ' · 1 px = 0.2 มม.');
    const lf = runOp('line_profile', flat, { mmPerPx: 0.2 }), sf = scalarMap(lf.ctx); assert.equal(sf.edge_pos_mm, null); assert.equal(sf.edge_pair_mm, null); assert.equal(sf.edge_pos_px, null);
    const lz = runOp('line_profile', e, { mmPerPx: 0.2 }); assert.equal(lz.ctx.values.fields.length, 12); assert.ok(lz.ctx.values.fields.every((f) => f.value === null)); assert.equal(lz.ctx.info, 'ภาพว่าง');
    for (const q of [lb, lm, lf, lz]) q.out.delete();
  } finally { for (const m of [scene, e, c, flat, blk]) m.delete(); }
});

// ===================================================================
// guided_filter
// ===================================================================
// อ้างอิงอิสระ (double ล้วน, box แบบ brute force ขอบสะท้อน REFLECT_101) ของ self-guided filter บนภาพ 1 ช่อง — ใช้เทียบ op (ต่าง <= 1 ระดับ)
function refGuided(data, W, H, r, eps255) {
  const refl = (i, n) => { if (n === 1) return 0; while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i; return i; };
  const box = (arr) => {
    const o = new Float64Array(W * H), cnt = (2 * r + 1) * (2 * r + 1);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let s = 0;
      for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) s += arr[refl(y + dy, H) * W + refl(x + dx, W)];
      o[y * W + x] = s / cnt;
    }
    return o;
  };
  const I = Float64Array.from(data), II = I.map((v) => v * v);
  const mI = box(I), cI = box(II);
  const a = new Float64Array(W * H), b = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) { const v = cI[i] - mI[i] * mI[i]; a[i] = v / (v + eps255); b[i] = mI[i] - a[i] * mI[i]; }
  const mA = box(a), mB = box(b);
  return Array.from(I, (v, i) => mA[i] * v + mB[i]);
}
test('ip4a-gf1. guided_filter: gray output == an independent float64 reference (box by brute force with reflect-101 edges), max diff <= 1 -- r 1 / 2 / 5 / 20 x eps 0.001 / 0.01 / 0.2 / 1 on noise + step images, incl. r larger than the image (1x1, 3x2); constant image -> unchanged; guidedFilterF32 returns a fresh CV_32FC1 Mat', () => {
  const imgs = [['noise', noiseGray(23, 17, 4)], ['step', gray(30, 20, (x) => (x < 15 ? 50 : 200))], ['3x2', gray(3, 2, (x, y) => 40 * x + 90 * y)], ['1x1', gray(1, 1, () => 77)], ['flat', gray(12, 9, () => 123)]];
  try {
    for (const [name, g] of imgs) for (const r of [1, 2, 5, 20]) for (const eps of [0.001, 0.01, 0.2, 1]) {
      const { out } = runOp('guided_filter', g, { radius: r, eps });
      const ref = refGuided(g.data, g.cols, g.rows, r, eps * 65025);
      let md = 0, nd = 0;
      for (let i = 0; i < ref.length; i++) { const want = Math.min(255, Math.max(0, Math.floor(ref[i] + 0.5))); const e = Math.abs(out.data[i] - want); md = Math.max(md, e); if (e) nd++; }
      assert.ok(md <= 1, `${name} r${r} eps${eps}: max diff to the float64 reference ${md}`);
      assert.ok(nd <= Math.max(1, ref.length * 0.001), `${name} r${r} eps${eps}: ${nd} of ${ref.length} pixels differ (rounding must be floor(q + 0.5))`);
      if (name === 'flat') sameBytes(out, g, 'constant image unchanged');
      out.delete();
    }
    const f = new cv.Mat(); imgs[0][1].convertTo(f, cv.CV_32F);
    const q = guidedFilterF32(cv, f, f, 2, 650.25);
    assert.equal(q.type(), cv.CV_32FC1); assert.equal(q.rows, 17); assert.equal(q.cols, 23); assert.notEqual(q.$$.ptr, f.$$.ptr);
    q.delete(); f.delete();
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4a-gf2. guided_filter: a 50 -> 200 step at eps 0.01 keeps the edge (the pixel next to the edge stays within 12 levels of its side, where a box blur is ~60 off) while a box blur of the same r smears it; eps 1 is much closer to the box blur than eps 0.01; colour = the 3 channels each filtered alone; RGBA -> 3 channels; 0x0 clone; info text; clamps; src untouched', () => {
  const step = gray(40, 20, (x) => (x < 20 ? 50 : 200)), c = noiseRGB(15, 11, 3), a = toRGBA(c), e = new cv.Mat();
  try {
    const sharp = runOp('guided_filter', step, { radius: 2, eps: 0.01 }), box = new cv.Mat(), soft = runOp('guided_filter', step, { radius: 2, eps: 1 });
    cv.boxFilter(step, box, -1, new cv.Size(5, 5), new cv.Point(-1, -1), true, cv.BORDER_REFLECT_101);
    const row = 10;
    assert.ok(Math.abs(sharp.out.data[row * 40 + 19] - 50) <= 12 && Math.abs(sharp.out.data[row * 40 + 20] - 200) <= 12, `edge pixels ${sharp.out.data[row * 40 + 19]} / ${sharp.out.data[row * 40 + 20]}`);
    assert.ok(box.data[row * 40 + 19] > 100 && box.data[row * 40 + 20] < 150, 'the box blur smears the same edge');
    let dSharp = 0, dSoft = 0; for (let i = 0; i < step.data.length; i++) { dSharp = Math.max(dSharp, Math.abs(sharp.out.data[i] - box.data[i])); dSoft = Math.max(dSoft, Math.abs(soft.out.data[i] - box.data[i])); }
    assert.ok(dSoft < dSharp / 2, `eps 1 (${dSoft}) is closer to the box blur than eps 0.01 (${dSharp})`);
    for (const m of [sharp.out, soft.out, box]) m.delete();
    const before = bytes(c);
    const rc = runOp('guided_filter', c, { radius: 3, eps: 0.05 });
    assert.equal(rc.out.channels(), 3); assert.deepEqual(bytes(c), before);
    const vec = new cv.MatVector(); cv.split(c, vec);
    for (let k = 0; k < 3; k++) {
      const ch = vec.get(k), one = runOp('guided_filter', ch, { radius: 3, eps: 0.05 });
      for (let i = 0; i < ch.data.length; i++) assert.equal(rc.out.data[i * 3 + k], one.out.data[i], `channel ${k} pixel ${i}`);
      one.out.delete(); ch.delete();
    }
    vec.delete();
    const ra = runOp('guided_filter', a, { radius: 3, eps: 0.05 }); assert.equal(ra.out.channels(), 3); sameBytes(ra.out, rc.out, 'RGBA = RGB'); assert.equal(ra.ctx.info, rc.ctx.info);
    assert.equal(rc.ctx.info, `Guided filter r=3 (หน้าต่าง 7×7) · ε=0.05 — ขอบที่ต่างกันเกินราว ${(Math.floor(Math.sqrt(0.05) * 255 * 10 + 0.5) / 10).toFixed(1)} ระดับสีถูกรักษาไว้`);
    const d = runOp('guided_filter', c, {}); assert.match(d.ctx.info, /^Guided filter r=2 \(หน้าต่าง 5×5\) · ε=0\.01 — ขอบที่ต่างกันเกินราว 25\.5 ระดับสีถูกรักษาไว้$/);
    const junk = runOp('guided_filter', c, { radius: 'x', eps: 'y' }); sameBytes(junk.out, d.out, 'junk -> defaults');
    const hi = runOp('guided_filter', c, { radius: 999, eps: 99 }); assert.match(hi.ctx.info, /^Guided filter r=20 \(หน้าต่าง 41×41\) · ε=1 /);
    const lo = runOp('guided_filter', c, { radius: -3, eps: 0 }); assert.match(lo.ctx.info, /^Guided filter r=1 \(หน้าต่าง 3×3\) · ε=0\.001 /);
    const z = runOp('guided_filter', e, {}); assert.equal(z.out.rows, 0); assert.notEqual(z.out.$$.ptr, e.$$.ptr); assert.equal(z.ctx.info, 'ภาพว่าง');
    for (const q of [rc, ra, d, junk, hi, lo, z]) q.out.delete();
  } finally { for (const m of [step, c, a, e]) m.delete(); }
});

// ===================================================================
// focus_score
// ===================================================================
const refl101 = (i, n) => { if (n === 1) return 0; while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i; return i; };
function refFocus(g, method, x0 = 0, y0 = 0, w = g.cols, h = g.rows) {
  // คำนวณจากภาพ crop แยก (ขอบสะท้อนในตัวของ crop เอง) ด้วยมือ ไม่ใช้ cv
  const px = (x, y) => g.data[(y0 + refl101(y, h)) * g.cols + x0 + refl101(x, w)];
  let s1 = 0, s2 = 0, st = 0, sm = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const l = px(x - 1, y) + px(x + 1, y) + px(x, y - 1) + px(x, y + 1) - 4 * px(x, y);
    s1 += l; s2 += l * l;
    const gx = (px(x + 1, y - 1) + 2 * px(x + 1, y) + px(x + 1, y + 1)) - (px(x - 1, y - 1) + 2 * px(x - 1, y) + px(x - 1, y + 1));
    const gy = (px(x - 1, y + 1) + 2 * px(x, y + 1) + px(x + 1, y + 1)) - (px(x - 1, y - 1) + 2 * px(x, y - 1) + px(x + 1, y - 1));
    st += gx * gx + gy * gy; sm += px(x, y);
  }
  const n = w * h;
  return { score: method === 'tenengrad' ? st / n : s2 / n - (s1 / n) * (s1 / n), mean: sm / n };
}
test('ip4a-fs1. focus_score: both methods == a brute-force hand computation EXACTLY (===) on noise / odd sizes / 1x1 / 2x1 (reflect-101 edges); roiPct 50 == the score of the separately cropped image (proves the ROI is cloned); focus_mean; values order; box size in the info', () => {
  const imgs = [['noise 31x23', noiseGray(31, 23, 5)], ['blob', blobGray()], ['1x1', gray(1, 1, () => 90)], ['2x1', gray(2, 1, (x) => 10 + 100 * x)], ['3x3', noiseGray(3, 3, 8)]];
  try {
    for (const [name, g] of imgs) for (const method of ['lap_var', 'tenengrad']) {
      const { out, ctx } = runOp('focus_score', g, { method });
      const ref = refFocus(g, method), m = scalarMap(ctx);
      assert.equal(m.focus_score, ref.score, `${name} ${method} score`); assert.equal(m.focus_mean, ref.mean, `${name} focus_mean`);
      assert.deepEqual(ctx.values.fields.map((f) => f.key), ['focus_score', 'focus_mean']); assert.equal(ctx.values.kind, 'scalars');
      assert.match(ctx.info, new RegExp(`^คะแนนความคม \\(${method === 'lap_var' ? 'variance of Laplacian' : 'Tenengrad'}\\) = `));
      assert.ok(ctx.info.includes(`วัดในกล่อง ${g.cols}×${g.rows} px กลางภาพ · ใช้เทียบภาพขนาดเดียวกัน ฉากเดียวกันเท่านั้น`));
      out.delete();
    }
    const g = imgs[1][1];
    for (const method of ['lap_var', 'tenengrad']) for (const pct of [50, 20, 75]) {
      const { out, ctx } = runOp('focus_score', g, { method, roiPct: pct });
      const rw = Math.max(1, Math.floor(g.cols * pct / 100 + 0.5)), rh = Math.max(1, Math.floor(g.rows * pct / 100 + 0.5)), x0 = Math.floor((g.cols - rw) / 2), y0 = Math.floor((g.rows - rh) / 2);
      // the independent check: copy the sub-image into its own Mat and score that whole image
      const cropData = new Uint8Array(rw * rh); for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) cropData[y * rw + x] = g.data[(y0 + y) * g.cols + x0 + x];
      const crop = cv.matFromArray(rh, rw, cv.CV_8UC1, cropData);
      const sep = runOp('focus_score', crop, { method, roiPct: 100 });
      assert.equal(scalarMap(ctx).focus_score, scalarMap(sep.ctx).focus_score, `${method} ${pct}%: roi == separate crop`);
      assert.equal(scalarMap(ctx).focus_mean, scalarMap(sep.ctx).focus_mean);
      assert.equal(scalarMap(ctx).focus_score, refFocus(g, method, x0, y0, rw, rh).score);
      assert.ok(ctx.info.includes(`วัดในกล่อง ${rw}×${rh} px กลางภาพ`));
      for (const m of [out, sep.out, crop]) m.delete();
    }
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4a-fs2. focus_score: a flat image scores 0 (+ the flat-image text); a sharp image scores higher than the same image after a 5x5 Gaussian blur; view lap = |Laplacian| of the whole image, pass = the input gray image; colour / RGBA input goes through gray; unknown method / view = defaults; clamps; 0x0 -> clone + null values; src untouched', () => {
  const flat = gray(30, 20, () => 100), sharp = blobGray(), blurred = new cv.Mat(), c = noiseRGB(20, 14, 2), a = toRGBA(c), e = new cv.Mat();
  try {
    cv.GaussianBlur(sharp, blurred, new cv.Size(5, 5), 0);
    const f0 = runOp('focus_score', flat, {}); assert.equal(scalarMap(f0.ctx).focus_score, 0); assert.ok(f0.ctx.info.endsWith(' · ภาพเรียบสนิท (ไม่มีขอบเลย)')); f0.out.delete();
    const s = runOp('focus_score', sharp, {}), b = runOp('focus_score', blurred, {});
    assert.ok(scalarMap(s.ctx).focus_score > scalarMap(b.ctx).focus_score * 2, `sharp ${scalarMap(s.ctx).focus_score} vs blurred ${scalarMap(b.ctx).focus_score}`);
    assert.ok(!s.ctx.info.includes('ภาพเรียบสนิท'));
    const lap = new cv.Mat(), absLap = new cv.Mat(); cv.Laplacian(sharp, lap, cv.CV_16S, 1); cv.convertScaleAbs(lap, absLap);
    sameBytes(s.out, absLap, 'view lap');
    const pass = runOp('focus_score', sharp, { view: 'pass' }); sameBytes(pass.out, sharp, 'view pass');
    const pass50 = runOp('focus_score', sharp, { view: 'pass', roiPct: 50 }); sameBytes(pass50.out, sharp, 'view pass keeps the whole image even for roiPct 50');
    const lap50 = runOp('focus_score', sharp, { roiPct: 50 }); sameBytes(lap50.out, absLap, 'view lap is the whole image even for roiPct 50');
    const before = bytes(c);
    const rc = runOp('focus_score', c, {}), g = new cv.Mat(); cv.cvtColor(c, g, cv.COLOR_RGB2GRAY);
    const rg = runOp('focus_score', g, {}); assert.equal(rc.out.channels(), 1); assert.equal(scalarMap(rc.ctx).focus_score, scalarMap(rg.ctx).focus_score); sameBytes(rc.out, rg.out, 'colour = via gray'); assert.deepEqual(bytes(c), before);
    const ra = runOp('focus_score', a, {}), g2 = new cv.Mat(); cv.cvtColor(a, g2, cv.COLOR_RGBA2GRAY); const rg2 = runOp('focus_score', g2, {}); assert.equal(scalarMap(ra.ctx).focus_score, scalarMap(rg2.ctx).focus_score);
    const weird = runOp('focus_score', sharp, { method: 'constructor', view: 'constructor', roiPct: 'x' }); assert.equal(scalarMap(weird.ctx).focus_score, scalarMap(s.ctx).focus_score); sameBytes(weird.out, s.out, 'unknown -> lap');
    const lo = runOp('focus_score', sharp, { roiPct: 1 }); assert.ok(lo.ctx.info.includes('วัดในกล่อง 32×24 px')); const hi = runOp('focus_score', sharp, { roiPct: 500 }); assert.ok(hi.ctx.info.includes('วัดในกล่อง 320×240 px'));
    const z = runOp('focus_score', e, {}); assert.equal(z.out.rows, 0); assert.notEqual(z.out.$$.ptr, e.$$.ptr); assert.equal(z.ctx.info, 'ภาพว่าง'); assert.deepEqual(z.ctx.values.fields.map((f) => [f.key, f.value]), [['focus_score', null], ['focus_mean', null]]);
    for (const q of [s, b, pass, pass50, lap50, rc, rg, ra, rg2, weird, lo, hi, z]) q.out.delete();
    for (const m of [lap, absLap, g, g2]) m.delete();
  } finally { for (const m of [flat, sharp, blurred, c, a, e]) m.delete(); }
});

// ===================================================================
// pseudocolor
// ===================================================================
// สูตรอ้างอิงเขียนใหม่ในเทสต์ (ไม่ import PSEUDOCOLOR_MAPS มาเทียบกับตัวเอง)
const clamp255 = (v) => Math.floor(Math.min(255, Math.max(0, v)) + 0.5);
const REF_MAPS = {
  jet: (i) => [clamp255(382.5 - Math.abs(4 * i - 765)), clamp255(382.5 - Math.abs(4 * i - 510)), clamp255(382.5 - Math.abs(4 * i - 255))],
  hot: (i) => [clamp255(3 * i), clamp255(3 * i - 255), clamp255(3 * i - 510)],
  cool: (i) => [i, 255 - i, 255],
};
test('ip4a-pc1. pseudocolor: every map at every level 0..255 == the formula (re-written in the test); PSEUDOCOLOR_MAPS agrees; jet 0 = (0,0,128) and 255 = (128,0,0); hot 0 = black, 255 = white; cool ends; output is RGB 3-channel same size; info per map; minmax on a 50..150 ramp maps the two ends to index 0 / 255 and follows the integer formula; single-valued image -> colour of index 0 + the extra info; colour / RGBA input via gray; 0x0 clone', () => {
  const ramp = gray(256, 2, (x) => x);
  try {
    for (const map of ['jet', 'hot', 'cool']) {
      const { out, ctx } = runOp('pseudocolor', ramp, { map });
      assert.equal(out.channels(), 3); assert.equal(out.rows, 2); assert.equal(out.cols, 256);
      for (let i = 0; i < 256; i++) { assert.deepEqual([out.data[i * 3], out.data[i * 3 + 1], out.data[i * 3 + 2]], REF_MAPS[map](i), `${map}[${i}]`); assert.deepEqual(PSEUDOCOLOR_MAPS[map](i), REF_MAPS[map](i)); }
      out.delete();
      assert.equal(ctx.info.includes('สีเป็นแค่การแสดงผล ข้อมูลยังเป็นค่าเทาเดิม'), true);
    }
    assert.deepEqual(PSEUDOCOLOR_MAPS.jet(0), [0, 0, 128]); assert.deepEqual(PSEUDOCOLOR_MAPS.jet(255), [128, 0, 0]);
    assert.deepEqual(PSEUDOCOLOR_MAPS.hot(0), [0, 0, 0]); assert.deepEqual(PSEUDOCOLOR_MAPS.hot(255), [255, 255, 255]);
    assert.deepEqual(PSEUDOCOLOR_MAPS.cool(0), [0, 255, 255]); assert.deepEqual(PSEUDOCOLOR_MAPS.cool(255), [255, 0, 255]);
    for (const i of [0, 64, 128, 191, 255]) for (const m of ['jet', 'hot', 'cool']) assert.deepEqual(PSEUDOCOLOR_MAPS[m](i), REF_MAPS[m](i));
    const j = runOp('pseudocolor', ramp, {}), j2 = runOp('pseudocolor', ramp, { map: 'constructor', range: 'constructor' }); sameBytes(j.out, j2.out, 'unknown = jet / full'); assert.equal(j.ctx.info, j2.ctx.info);
    assert.equal(j.ctx.info, 'Jet: ค่า 0 = น้ำเงินเข้ม → 255 = แดงเข้ม · สีเป็นแค่การแสดงผล ข้อมูลยังเป็นค่าเทาเดิม');
    j.out.delete(); j2.out.delete();
    // minmax
    const r2 = gray(101, 1, (x) => 50 + x), mm = runOp('pseudocolor', r2, { range: 'minmax' });
    for (let x = 0; x < 101; x++) { const t = Math.floor(((x) * 510 + 100) / (2 * 100)); assert.deepEqual([mm.out.data[x * 3], mm.out.data[x * 3 + 1], mm.out.data[x * 3 + 2]], REF_MAPS.jet(t), `minmax x${x} -> index ${t}`); }
    assert.deepEqual([mm.out.data[0], mm.out.data[1], mm.out.data[2]], REF_MAPS.jet(0)); assert.deepEqual([mm.out.data[300], mm.out.data[301], mm.out.data[302]], REF_MAPS.jet(255));
    assert.ok(mm.ctx.info.endsWith(' · ยืดช่วง 50–150 ให้เต็มแถบสี')); mm.out.delete(); r2.delete();
    const one = gray(6, 4, () => 77), o1 = runOp('pseudocolor', one, { range: 'minmax', map: 'hot' });
    for (let i = 0; i < 24; i++) assert.deepEqual([o1.out.data[i * 3], o1.out.data[i * 3 + 1], o1.out.data[i * 3 + 2]], REF_MAPS.hot(0));
    assert.ok(o1.ctx.info.endsWith(' · ยืดช่วง 77–77 ให้เต็มแถบสี · ภาพมีค่าเดียว (77) จึงได้สีต้นแถบทั้งภาพ')); o1.out.delete(); one.delete();
    const c = noiseRGB(10, 8, 1), a = toRGBA(c), g = new cv.Mat(); cv.cvtColor(c, g, cv.COLOR_RGB2GRAY);
    const rc = runOp('pseudocolor', c, {}), rg = runOp('pseudocolor', g, {}); sameBytes(rc.out, rg.out, 'colour via gray');
    const ra = runOp('pseudocolor', a, {}), g2 = new cv.Mat(); cv.cvtColor(a, g2, cv.COLOR_RGBA2GRAY); const rg2 = runOp('pseudocolor', g2, {}); sameBytes(ra.out, rg2.out, 'RGBA via gray');
    const e = new cv.Mat(), z = runOp('pseudocolor', e, {}); assert.equal(z.out.rows, 0); assert.equal(z.ctx.info, 'ภาพว่าง');
    for (const q of [rc, rg, ra, rg2, z]) q.out.delete();
    for (const m of [c, a, g, g2, e]) m.delete();
  } finally { ramp.delete(); }
});

// ===================================================================
// delta_e
// ===================================================================
// Sharma, Wu, Dalal (2005) "The CIEDE2000 colour-difference formula", Table 1 (L1 a1 b1 | L2 a2 b2 | dE00) — ชุดทดสอบมาตรฐาน 34 คู่ · คู่ #1 #7 #17 คัดจากสเปก
const SHARMA = [
  [50, 2.6772, -79.7751, 50, 0, -82.7485, 2.0425], [50, 3.1571, -77.2803, 50, 0, -82.7485, 2.8615], [50, 2.8361, -74.0200, 50, 0, -82.7485, 3.4412],
  [50, -1.3802, -84.2814, 50, 0, -82.7485, 1.0000], [50, -1.1848, -84.8006, 50, 0, -82.7485, 1.0000], [50, -0.9009, -85.5211, 50, 0, -82.7485, 1.0000],
  [50, 0, 0, 50, -1, 2, 2.3669], [50, -1, 2, 50, 0, 0, 2.3669], [50, 2.49, -0.001, 50, -2.49, 0.0009, 7.1792], [50, 2.49, -0.001, 50, -2.49, 0.001, 7.1792],
  [50, 2.49, -0.001, 50, -2.49, 0.0011, 7.2195], [50, 2.49, -0.001, 50, -2.49, 0.0012, 7.2195], [50, -0.001, 2.49, 50, 0.0009, -2.49, 4.8045], [50, -0.001, 2.49, 50, 0.001, -2.49, 4.8045],
  [50, -0.001, 2.49, 50, 0.0011, -2.49, 4.7461], [50, 2.5, 0, 50, 0, -2.5, 4.3065], [50, 2.5, 0, 73, 25, -18, 27.1492], [50, 2.5, 0, 61, -5, 29, 22.8977],
  [50, 2.5, 0, 56, -27, -3, 31.9030], [50, 2.5, 0, 58, 24, 15, 19.4535], [50, 2.5, 0, 50, 3.1736, 0.5854, 1.0000], [50, 2.5, 0, 50, 3.2972, 0, 1.0000],
  [50, 2.5, 0, 50, 1.8634, 0.5757, 1.0000], [50, 2.5, 0, 50, 3.2592, 0.3350, 1.0000], [60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644],
  [63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630], [61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731], [35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645],
  [22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373], [36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146], [90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441],
  [90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381], [6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377], [2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082],
];
test('ip4a-de1. deltaE2000 == the 34 test pairs of Sharma, Wu, Dalal (2005) Table 1 (|diff| <= 5e-5), symmetric in its arguments; deltaE76 is the plain Lab distance; identical colours -> 0', () => {
  assert.equal(SHARMA.length, 34);
  SHARMA.forEach(([L1, a1, b1, L2, a2, b2, want], i) => {
    near(deltaE2000(L1, a1, b1, L2, a2, b2), want, 5e-5, `pair #${i + 1}`);
    near(deltaE2000(L2, a2, b2, L1, a1, b1), want, 5e-5, `pair #${i + 1} swapped`);
  });
  assert.equal(deltaE76(50, 10, -20, 50, 10, -20), 0); assert.equal(deltaE2000(50, 10, -20, 50, 10, -20), 0);
  near(deltaE76(10, 0, 0, 13, 4, 0), 5, 1e-12, 'dE76 3-4-5'); near(deltaE76(0, 3, 0, 0, 0, 4), 5, 1e-12, 'dE76');
  assert.equal(deltaE2000(0, 0, 0, 0, 0, 0), 0);
});

function labOf(rgbTriple) { // อ้างอิงอิสระ: Lab (float32 pipeline ของ cv) ของสีเดียว
  const m = rgb(1, 1, () => rgbTriple), f = new cv.Mat(), lab = new cv.Mat();
  m.convertTo(f, cv.CV_32F, 1 / 255); cv.cvtColor(f, lab, cv.COLOR_RGB2Lab);
  const v = Array.from(lab.data32F); for (const x of [m, f, lab]) x.delete();
  return v;
}
test('ip4a-de2. delta_e: the same image -> every value 0 and a black heatmap; A red vs B blue: CIE76 == the Lab distance of the two colours (computed with cv on 1x1 images); values: 9 scalars in order with the right labels; heatmap / over formulas; CIEDE2000 on the same pair; src and src2 untouched', () => {
  const A = rgb(20, 12, () => [220, 30, 30]), B = rgb(20, 12, () => [30, 30, 220]), same = noiseRGB(15, 11, 4);
  try {
    const bA = bytes(A), bB = bytes(B);
    for (const formula of ['cie76', 'ciede2000']) {
      const z = runOp('delta_e', same, { formula }, { src2: same });
      const m = scalarMap(z.ctx);
      assert.equal(m.de_mean, 0); assert.equal(m.de_max, 0); assert.equal(m.de_over_pct, 0);
      assert.ok(z.out.data.every((v) => v === 0)); assert.equal(z.out.channels(), 1);
      assert.deepEqual(z.ctx.values.fields.map((f) => f.key), DE_KEYS); assert.equal(z.ctx.values.kind, 'scalars');
      assert.deepEqual(z.ctx.values.fields.map((f) => f.label), ['ΔE เฉลี่ย', 'ΔE สูงสุด', 'พื้นที่ที่ ΔE เกินเกณฑ์ (%)', 'A: L* เฉลี่ย', 'A: a* เฉลี่ย', 'A: b* เฉลี่ย', 'B: L* เฉลี่ย', 'B: a* เฉลี่ย', 'B: b* เฉลี่ย']);
      assert.equal(z.ctx.info, `ΔE (${formula === 'cie76' ? 'CIE76' : 'CIEDE2000'}) เฉลี่ย 0.00 · สูงสุด 0.00 · เกินเกณฑ์ 2.3 อยู่ 0.0% ของภาพ`);
      z.out.delete();
    }
    const la = labOf([220, 30, 30]), lb = labOf([30, 30, 220]);
    const r76 = runOp('delta_e', A, { formula: 'cie76', scaleMax: 100, tol: 10 }, { src2: B }), m76 = scalarMap(r76.ctx);
    const want76 = Math.sqrt((la[0] - lb[0]) ** 2 + (la[1] - lb[1]) ** 2 + (la[2] - lb[2]) ** 2);
    near(m76.de_mean, want76, 1e-9, 'CIE76 mean'); near(m76.de_max, want76, 1e-9, 'CIE76 max'); assert.equal(m76.de_over_pct, 100);
    near(m76.de_a_L, la[0], 1e-9, 'A L'); near(m76.de_a_a, la[1], 1e-9, 'A a'); near(m76.de_a_b, la[2], 1e-9, 'A b'); near(m76.de_b_L, lb[0], 1e-9, 'B L'); near(m76.de_b_a, lb[1], 1e-9, 'B a'); near(m76.de_b_b, lb[2], 1e-9, 'B b');
    assert.ok(want76 > 50 && want76 < 150, `a red / blue pair is far apart (dE76 = ${want76})`);
    assert.ok(r76.out.data.every((v) => v === Math.min(255, Math.floor(want76 * 255 / 100 + 0.5))), 'heatmap formula');
    const r00 = runOp('delta_e', A, { formula: 'ciede2000', view: 'over', tol: 10 }, { src2: B });
    near(scalarMap(r00.ctx).de_mean, deltaE2000(la[0], la[1], la[2], lb[0], lb[1], lb[2]), 1e-9, 'CIEDE2000 mean'); assert.ok(r00.out.data.every((v) => v === 255), 'over mask');
    const rNo = runOp('delta_e', A, { formula: 'ciede2000', view: 'over', tol: 50 }, { src2: B }); assert.ok(rNo.out.data.every((v) => v === 0)); assert.equal(scalarMap(rNo.ctx).de_over_pct, 0);
    assert.deepEqual(bytes(A), bA); assert.deepEqual(bytes(B), bB);
    for (const q of [r76, r00, rNo]) q.out.delete();
  } finally { for (const m of [A, B, same]) m.delete(); }
});

test('ip4a-de3. delta_e: compare mean_b with a B of any size (no resize, no resize text) uses B\'s mean Lab for every pixel; pixel compare with a different-size B resizes it (+ RESIZED text) and equals running on the pre-resized B; gray / RGBA inputs; heatmap / over per-pixel formulas on a small image; no B / B 0x0 -> clone + info + 9 null values; A 0x0; unknown selects and junk numbers; clamps', () => {
  const A = rgb(16, 10, (x, y) => [x * 15, y * 25, 100]), Bs = rgb(5, 4, (x, y) => [x * 50, 200 - y * 40, 80]), Bsame = rgb(16, 10, (x, y) => [(x * 7 + 20) & 255, y * 20, 60]), e = new cv.Mat();
  try {
    const mb = runOp('delta_e', A, { compare: 'mean_b' }, { src2: Bs });
    assert.ok(!mb.ctx.info.includes('ปรับขนาด B ให้เท่า A')); assert.equal(mb.out.cols, 16); assert.equal(mb.out.rows, 10);
    const f = new cv.Mat(), lab = new cv.Mat(); Bs.convertTo(f, cv.CV_32F, 1 / 255); cv.cvtColor(f, lab, cv.COLOR_RGB2Lab);
    let sL = 0, sa = 0, sb = 0; for (let i = 0; i < 20; i++) { sL += lab.data32F[i * 3]; sa += lab.data32F[i * 3 + 1]; sb += lab.data32F[i * 3 + 2]; }
    const m = scalarMap(mb.ctx); near(m.de_b_L, sL / 20, 1e-9, 'mean B L'); near(m.de_b_a, sa / 20, 1e-9, 'mean B a'); near(m.de_b_b, sb / 20, 1e-9, 'mean B b');
    const f1 = (x) => (Math.floor(x * 10 + 0.5) / 10).toFixed(1);
    assert.ok(mb.ctx.info.includes(` · เทียบกับสีเฉลี่ยของ B (L* ${f1(sL / 20)} a* ${f1(sa / 20)} b* ${f1(sb / 20)})`));
    // per-pixel check of the heatmap against the formula on A's own Lab
    const fa = new cv.Mat(), laba = new cv.Mat(); A.convertTo(fa, cv.CV_32F, 1 / 255); cv.cvtColor(fa, laba, cv.COLOR_RGB2Lab);
    for (let i = 0; i < 160; i++) { const d = deltaE76(laba.data32F[i * 3], laba.data32F[i * 3 + 1], laba.data32F[i * 3 + 2], sL / 20, sa / 20, sb / 20); assert.equal(mb.out.data[i], Math.min(255, Math.floor(d * 255 / 20 + 0.5)), `heatmap pixel ${i}`); }
    for (const x of [f, lab, fa, laba]) x.delete();
    const px = runOp('delta_e', A, {}, { src2: Bs });
    assert.ok(px.ctx.info.endsWith(' (ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)'));
    const pre = new cv.Mat(); cv.resize(Bs, pre, new cv.Size(16, 10), 0, 0, cv.INTER_LINEAR);
    const px2 = runOp('delta_e', A, {}, { src2: pre }); sameBytes(px.out, px2.out, 'pixel compare on resized B'); assert.deepEqual(scalarMap(px.ctx), scalarMap(px2.ctx)); assert.ok(!px2.ctx.info.includes('ปรับขนาด'));
    const ov = runOp('delta_e', A, { view: 'over', tol: 8 }, { src2: Bsame });
    const fb = new cv.Mat(), fa2 = new cv.Mat(), lA = new cv.Mat(), lB = new cv.Mat(); A.convertTo(fa2, cv.CV_32F, 1 / 255); Bsame.convertTo(fb, cv.CV_32F, 1 / 255); cv.cvtColor(fa2, lA, cv.COLOR_RGB2Lab); cv.cvtColor(fb, lB, cv.COLOR_RGB2Lab);
    let over = 0; for (let i = 0; i < 160; i++) { const d = deltaE76(lA.data32F[i * 3], lA.data32F[i * 3 + 1], lA.data32F[i * 3 + 2], lB.data32F[i * 3], lB.data32F[i * 3 + 1], lB.data32F[i * 3 + 2]); const want = d > 8 ? 255 : 0; if (d > 8) over++; assert.equal(ov.out.data[i], want, `over pixel ${i}`); }
    assert.equal(scalarMap(ov.ctx).de_over_pct, over * 100 / 160); for (const x of [fb, fa2, lA, lB]) x.delete();
    const gA = noiseGray(12, 9, 2), aA = toRGBA(noiseRGB(12, 9, 3));
    for (const [x, y] of [[gA, Bsame], [aA, gA], [aA, aA], [gA, gA]]) { const r = runOp('delta_e', x, {}, { src2: y }); assert.equal(r.out.channels(), 1); assert.equal(r.out.rows, 9); r.out.delete(); }
    const rg = runOp('delta_e', gA, {}, { src2: gA }); assert.equal(scalarMap(rg.ctx).de_mean, 0); rg.out.delete();
    const noB = runOp('delta_e', A, {}), emptyB = runOp('delta_e', A, {}, { src2: e });
    assert.equal(noB.ctx.info, "ยังไม่ได้ต่อสาย 'เข้า B' — ลากเส้นจากอีกจุดหนึ่งมาต่อที่ช่องอินพุตที่สองของกล่องนี้"); assert.match(emptyB.ctx.info, /^ภาพที่ต่อเข้า 'เข้า B' ว่าง/);
    for (const r of [noB, emptyB]) { assert.deepEqual(r.ctx.values.fields.map((q) => [q.key, q.value]), DE_KEYS.map((k) => [k, null])); sameBytes(r.out, A, 'passthrough clone'); assert.notEqual(r.out.$$.ptr, A.$$.ptr); r.out.delete(); }
    const aE = runOp('delta_e', e, {}, { src2: Bsame }); assert.equal(aE.out.rows, 0); assert.equal(aE.ctx.info, 'ภาพว่าง'); assert.deepEqual(aE.ctx.values.fields.map((q) => q.value), DE_KEYS.map(() => null)); aE.out.delete();
    const base = runOp('delta_e', A, {}, { src2: Bsame }), weird = runOp('delta_e', A, { formula: 'constructor', compare: 'constructor', view: 'constructor', tol: 'x', scaleMax: 'y' }, { src2: Bsame });
    sameBytes(base.out, weird.out, 'unknown selects / junk numbers = defaults'); assert.equal(base.ctx.info, weird.ctx.info);
    const hiN = runOp('delta_e', A, { tol: 999, scaleMax: 999 }, { src2: Bsame }), loN = runOp('delta_e', A, { tol: -4, scaleMax: -9 }, { src2: Bsame });
    assert.ok(hiN.ctx.info.includes('เกินเกณฑ์ 50 อยู่')); assert.ok(loN.ctx.info.includes('เกินเกณฑ์ 0 อยู่'));
    for (const q of [mb, px, px2, ov, rg, base, weird, hiN, loN]) { if (q.out && !q.out.isDeleted()) q.out.delete(); }
    for (const m2 of [gA, aA, pre]) m2.delete();
  } finally { for (const m of [A, Bs, Bsame, e]) m.delete(); }
});

// ===================================================================
// roi_apply
// ===================================================================
test('ip4a-ra1. roi_apply: rect 25/25/50/50 on 100x80 -> box (25,20)-(74,59); outside = A exactly, inside = B exactly (feather 0); ellipse pixel set == a brute-force double check of the integer formula; invert = the complement; values (count, %) after invert; info text; B of another size is resized; A gray + B colour -> 3 channels; src and src2 untouched', () => {
  const A = rgb(100, 80, (x, y) => [x, y, 7]), B = rgb(100, 80, (x, y) => [255 - x, 200, y]), small = rgb(10, 8, (x, y) => [x * 20, y * 20, 90]), g = gray(100, 80, (x, y) => (x + y) & 255);
  try {
    const bA = bytes(A), bB = bytes(B);
    const { out, ctx } = runOp('roi_apply', A, {}, { src2: B });
    for (let y = 0; y < 80; y++) for (let x = 0; x < 100; x++) {
      const inside = x >= 25 && x <= 74 && y >= 20 && y <= 59, src = inside ? B : A;
      for (let c = 0; c < 3; c++) assert.equal(out.data[(y * 100 + x) * 3 + c], src.data[(y * 100 + x) * 3 + c], `(${x},${y}) c${c}`);
    }
    assert.deepEqual(scalarMap(ctx), { ra_px: 2000, ra_pct: 25 }); assert.deepEqual(ctx.values.fields.map((f) => f.key), RA_KEYS);
    assert.equal(ctx.info, 'ใช้ภาพ B ในกรอบสี่เหลี่ยม (25, 20)–(74, 59) px · 2000 px (25.0%)');
    out.delete();
    // ellipse vs brute force (double)
    const el = runOp('roi_apply', A, { shape: 'ellipse' }, { src2: B });
    let cnt = 0;
    for (let y = 0; y < 80; y++) for (let x = 0; x < 100; x++) {
      const ins = x >= 25 && x < 75 && y >= 20 && y < 60 && ((x + 0.5 - 50) / 25) ** 2 + ((y + 0.5 - 40) / 20) ** 2 <= 1 + 1e-12;
      const got = el.out.data[(y * 100 + x) * 3] === B.data[(y * 100 + x) * 3] && el.out.data[(y * 100 + x) * 3 + 2] === B.data[(y * 100 + x) * 3 + 2];
      if (ins) cnt++;
      if (ins) assert.ok(got, `ellipse inside (${x},${y})`); else assert.ok(!got || (A.data[(y * 100 + x) * 3] === B.data[(y * 100 + x) * 3] && A.data[(y * 100 + x) * 3 + 2] === B.data[(y * 100 + x) * 3 + 2]), `ellipse outside (${x},${y})`);
    }
    assert.equal(scalarMap(el.ctx).ra_px, cnt); assert.ok(cnt > 1500 && cnt < 2500, `ellipse area ${cnt} ~ pi*25*20 = 1571`); assert.match(el.ctx.info, /^ใช้ภาพ B ในกรอบวงรี \(25, 20\)–\(74, 59\) px · /);
    const inv = runOp('roi_apply', A, { shape: 'ellipse', invert: true }, { src2: B });
    assert.equal(scalarMap(inv.ctx).ra_px, 8000 - cnt); near(scalarMap(inv.ctx).ra_pct, (8000 - cnt) * 100 / 8000, 1e-12, 'ra_pct'); assert.match(inv.ctx.info, /^ใช้ภาพ B นอกกรอบวงรี/);
    for (let i = 0; i < 8000; i++) { if (A.data[i * 3] !== B.data[i * 3]) assert.notEqual(el.out.data[i * 3] === B.data[i * 3], inv.out.data[i * 3] === B.data[i * 3], `pixel ${i}: invert flips which image is used`); }
    for (const q of [el, inv]) q.out.delete();
    const rs = runOp('roi_apply', A, {}, { src2: small }); assert.ok(rs.ctx.info.endsWith(' (ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)')); assert.equal(rs.out.cols, 100); rs.out.delete();
    const mix = runOp('roi_apply', g, {}, { src2: B }); assert.equal(mix.out.channels(), 3); mix.out.delete();
    const mix2 = runOp('roi_apply', A, {}, { src2: g }); assert.equal(mix2.out.channels(), 3); mix2.out.delete();
    const gg = runOp('roi_apply', g, {}, { src2: g }); assert.equal(gg.out.channels(), 1); sameBytes(gg.out, g, 'identical gray images'); gg.out.delete();
    const a4 = toRGBA(A), ra = runOp('roi_apply', a4, {}, { src2: B }); assert.equal(ra.out.channels(), 3); ra.out.delete(); a4.delete();
    assert.deepEqual(bytes(A), bA); assert.deepEqual(bytes(B), bB);
  } finally { for (const m of [A, B, small, g]) m.delete(); }
});

test('ip4a-ra2. roi_apply: feather 0 is a hard mask; feather > 0 blends smoothly (the box centre is B, far outside is A, the seam pixels are in between and the values stay between A and B); outline paints yellow only on the box border; invert; clamps (the box is always >= 1 px, even x = 100 on a 1x1); no B / B 0x0 / A 0x0; unknown selects', () => {
  const A = gray(60, 40, () => 0), B = gray(60, 40, () => 200), Ac = rgb(60, 40, () => [0, 0, 0]), Bc = rgb(60, 40, () => [200, 200, 200]), e = new cv.Mat(), one = gray(1, 1, () => 9), oneB = gray(1, 1, () => 99);
  try {
    const hard = runOp('roi_apply', A, {}, { src2: B });
    assert.equal(hard.out.data[20 * 60 + 15], 200); assert.equal(hard.out.data[20 * 60 + 14], 0);
    const soft = runOp('roi_apply', A, { feather: 6 }, { src2: B });
    assert.equal(soft.out.data[20 * 60 + 30], 200, 'centre of the box = B'); assert.equal(soft.out.data[2 * 60 + 2], 0, 'far outside = A');
    let mid = 0; for (let x = 8; x < 22; x++) { const v = soft.out.data[20 * 60 + x]; assert.ok(v >= 0 && v <= 200); if (v > 0 && v < 200) mid++; }
    assert.ok(mid >= 3, `feather leaves a gradient across the seam (${mid} mid pixels)`);
    for (let x = 9; x < 21; x++) assert.ok(soft.out.data[20 * 60 + x] <= soft.out.data[20 * 60 + x + 1] + 0, 'monotone across the left seam');
    assert.equal(scalarMap(soft.ctx).ra_px, scalarMap(hard.ctx).ra_px, 'feather does not change the counter'); assert.ok(soft.ctx.info.includes(' · ขอบนุ่ม 6 px'));
    const w = (x, y) => hard.out.data[y * 60 + x];
    const ol = runOp('roi_apply', Ac, { view: 'outline' }, { src2: Bc });
    const x0 = 15, y0 = 10, x1 = 45, y1 = 30; // 60 * 25 / 100 = 15 · 40 * 25 / 100 = 10 · width 50% = 30 · height 50% = 20
    let yellow = 0;
    for (let y = 0; y < 40; y++) for (let x = 0; x < 60; x++) {
      const px = [ol.out.data[(y * 60 + x) * 3], ol.out.data[(y * 60 + x) * 3 + 1], ol.out.data[(y * 60 + x) * 3 + 2]];
      const inside = x >= x0 && x < x1 && y >= y0 && y < y1, border = inside && (x === x0 || x === x1 - 1 || y === y0 || y === y1 - 1);
      if (border) { assert.deepEqual(px, [255, 255, 0], `border (${x},${y})`); yellow++; }
      else assert.deepEqual(px, inside ? [200, 200, 200] : [0, 0, 0], `(${x},${y})`);
    }
    assert.equal(yellow, 2 * 30 + 2 * 20 - 4);
    const olg = runOp('roi_apply', A, { view: 'outline' }, { src2: B }); assert.equal(olg.out.channels(), 3); assert.deepEqual([olg.out.data[(10 * 60 + 15) * 3], olg.out.data[(10 * 60 + 15) * 3 + 1], olg.out.data[(10 * 60 + 15) * 3 + 2]], [255, 255, 0]);
    const tiny = runOp('roi_apply', one, { x: 100, y: 100, w: 100, h: 100 }, { src2: oneB }); assert.equal(tiny.out.data[0], 99); assert.equal(scalarMap(tiny.ctx).ra_px, 1); assert.ok(tiny.ctx.info.includes('(0, 0)–(0, 0) px · 1 px (100.0%)'));
    const edge = runOp('roi_apply', A, { x: 100, y: 100, w: 1, h: 1 }, { src2: B }); assert.equal(scalarMap(edge.ctx).ra_px, 1); assert.equal(edge.out.data[39 * 60 + 59], 200);
    const junk = runOp('roi_apply', A, { shape: 'constructor', view: 'constructor', x: 'q', y: 'q', w: 'q', h: 'q', feather: 'q', invert: 'yes' }, { src2: B }), base = runOp('roi_apply', A, {}, { src2: B });
    sameBytes(junk.out, base.out, 'unknown selects / junk numbers / non-boolean invert = defaults'); assert.equal(junk.ctx.info, base.ctx.info);
    const big = runOp('roi_apply', A, { x: -5, y: -5, w: 999, h: 999, feather: 999 }, { src2: B }); assert.ok(big.ctx.info.includes('(0, 0)–(59, 39) px') && big.ctx.info.includes('ขอบนุ่ม 50 px'));
    const noB = runOp('roi_apply', A, {}), eB = runOp('roi_apply', A, {}, { src2: e }), eA = runOp('roi_apply', e, {}, { src2: B });
    for (const r of [noB, eB]) { assert.deepEqual(r.ctx.values.fields.map((f) => [f.key, f.value]), [['ra_px', null], ['ra_pct', null]]); sameBytes(r.out, A, 'passthrough'); }
    assert.equal(eA.out.rows, 0); assert.equal(eA.ctx.info, 'ภาพว่าง'); assert.deepEqual(eA.ctx.values.fields.map((f) => f.value), [null, null]);
    for (const q of [hard, soft, ol, olg, tiny, edge, junk, base, big, noB, eB, eA]) q.out.delete();
    void w;
  } finally { for (const m of [A, B, Ac, Bc, e, one, oneB]) m.delete(); }
});

// ===================================================================
// detail-panel: เปลี่ยน mmPerPx แล้วชุดคอลัมน์ของตารางเปลี่ยนถูกต้อง (สเปก §4.8 [ต้องยืนยัน]) — DOM stub ขนาดเล็ก (ไม่มี jsdom)
// ===================================================================
class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.children = []; this.parentNode = null; this._text = ''; this._cls = ''; this.style = {}; this._attrs = {}; this.hidden = false; this.title = ''; this._listeners = {};
    const self = this;
    this.classList = {
      add(c) { const s = new Set(self._cls.split(' ').filter(Boolean)); s.add(c); self._cls = [...s].join(' '); },
      remove(c) { const s = new Set(self._cls.split(' ').filter(Boolean)); s.delete(c); self._cls = [...s].join(' '); },
      contains(c) { return self._cls.split(' ').includes(c); },
      toggle(c, on) { const want = on === undefined ? !this.contains(c) : !!on; if (want) this.add(c); else this.remove(c); },
    };
  }
  get className() { return this._cls; }
  set className(v) { this._cls = String(v); }
  get textContent() { return this.children.length ? this.children.map((c) => c.textContent).join('') : this._text; }
  set textContent(v) { this._text = String(v); this.children = []; }
  set innerHTML(v) { this._text = String(v).replace(/<[^>]*>/g, ''); this.children = []; }
  get innerHTML() { return this._text; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  append(...cs) { for (const c of cs) this.appendChild(c); }
  setAttribute(k, v = '') { this._attrs[k] = String(v); }
  getAttribute(k) { return Object.prototype.hasOwnProperty.call(this._attrs, k) ? this._attrs[k] : null; }
  removeAttribute(k) { delete this._attrs[k]; }
  addEventListener(t, fn) { (this._listeners[t] ||= []).push(fn); }
  blur() {} focus() {}
  all(cls, out = []) { for (const c of this.children) { if (c._cls.split(' ').includes(cls)) out.push(c); c.all(cls, out); } return out; }
}
test('ip4a-mm2. detail panel: switching mmPerPx 0 -> 0.25 -> 0 rebuilds the Region Properties table with 19 -> 24 -> 19 columns (new headers in Thai, no ghost columns); same for the Feret table (10 -> 13 -> 10) and the Line Profile scalar list (9 -> 12 -> 9)', async () => {
  const { createDetailPanel } = await import('../src/renderer/ui/detail-panel.js');
  const saved = { document: globalThis.document, hadDocument: Object.prototype.hasOwnProperty.call(globalThis, 'document'), st: globalThis.setTimeout, ct: globalThis.clearTimeout, now: Date.now };
  let fakeNow = 5_000_000, timers = [], seq = 0;
  globalThis.document = { createElement: (t) => new El(t) };
  globalThis.setTimeout = (fn, ms) => { const id = ++seq; timers.push({ id, at: fakeNow + (Number(ms) || 0), fn }); return id; };
  globalThis.clearTimeout = (id) => { timers = timers.filter((t) => t.id !== id); };
  Date.now = () => fakeNow;
  const advance = (ms) => { const end = fakeNow + ms; for (;;) { let due = null; for (const t of timers) if (t.at <= end && (!due || t.at < due.at)) due = t; if (!due) break; timers = timers.filter((t) => t !== due); fakeNow = due.at; due.fn(); } fakeNow = end; };
  const scene = objectsScene(), c = noiseRGB(60, 40, 3);
  try {
    for (const [mode, src, headerKey, mmHeaders] of [['regionprops', scene, 'dv-th', ['พื้นที่ (มม.²)', 'เส้นรอบรูป (มม.)', 'แกนยาว (มม.)']], ['feret', scene, 'dv-th', ['Feret ยาวสุด (มม.)', 'Feret แคบสุด (มม.)', 'ยาวตามกรอบแคบสุด (มม.)']], ['line_profile', c, 'dv-k', null]]) {
      const filterId = 'analysis_detect';
      const step = { id: filterId, nodeId: 3, params: { ...defaultParams(filterId), mode } };
      const container = new El('div');
      const panel = createDetailPanel(container, { onParamChange: () => {} });
      const notesFor = (mm) => {
        step.params[`${mode}__mmPerPx`] = mm;
        const local = { mmPerPx: mm, ...(mode === 'regionprops' ? { minArea: 0 } : mode === 'feret' ? { minArea: 0 } : {}) };
        const { out, ctx } = runOp(mode, src, local); out.delete();
        const view = FILTERS[filterId].subtitle(step.params);
        return [{ id: filterId, nodeId: 3, view, info: ctx.info }, { id: filterId, nodeId: 3, view, values: ctx.values }];
      };
      const apply = (notes) => { if (!panel.applyNotes(notes)) panel.render(step, notes); advance(300); };
      const shown = () => (mode === 'line_profile' ? container.all('dv-row').length || container.all('dv-n').length : container.all('dv-th').length);
      const header = () => container.all('dv-th').map((e) => e.textContent);
      step.params[`${mode}__mmPerPx`] = 0;
      const n0 = notesFor(0); panel.render(step, n0); advance(300);
      const base = mode === 'line_profile' ? n0[1].values.fields.length : n0[1].values.columns.length;
      assert.equal(base, mode === 'regionprops' ? 19 : mode === 'feret' ? 10 : 9);
      if (mode !== 'line_profile') assert.equal(shown(), base, `${mode}: columns before`);
      apply(notesFor(0.25));
      const n1 = notesFor(0.25), mmCount = mode === 'line_profile' ? n1[1].values.fields.length : n1[1].values.columns.length;
      assert.equal(mmCount, base + (mode === 'regionprops' ? 5 : 3));
      apply(n1);
      if (mode !== 'line_profile') {
        assert.equal(shown(), mmCount, `${mode}: columns after switching mmPerPx on`);
        for (const h of mmHeaders) assert.ok(header().includes(h), `${mode}: header "${h}" missing`);
      } else {
        const txt = container.textContent; for (const lab of ['ความยาวเส้น (มม.)', 'ตำแหน่งขอบที่กระโดดแรง นับจาก A (มม.)', 'ระยะระหว่างขอบขึ้นกับขอบลง (มม.)']) assert.ok(txt.includes(lab), `line_profile: "${lab}" missing`);
      }
      const n2 = notesFor(0); apply(n2);
      if (mode !== 'line_profile') { assert.equal(shown(), base, `${mode}: columns after switching back`); for (const h of mmHeaders) assert.ok(!header().includes(h), `${mode}: ghost header "${h}"`); }
      else for (const lab of ['ความยาวเส้น (มม.)']) assert.ok(!container.textContent.includes(lab), 'line_profile: ghost mm field');
      void headerKey; void shown;
    }
  } finally {
    globalThis.setTimeout = saved.st; globalThis.clearTimeout = saved.ct; Date.now = saved.now;
    if (saved.hadDocument) globalThis.document = saved.document; else delete globalThis.document;
    scene.delete(); c.delete();
  }
});

// ===================================================================
// static: โครงสร้าง / palette / icon / clamp / ข้อความ
// ===================================================================
test('ip4a-S1. (static structure) the 5 new ops exist with the right needs and host groups; OPS = 87, FILTERS stays 33; no select key is "mode"; defaultParams() == the OPS defaults (prefixed); legacy groups are untouched and FILTERS.psnr has no params', () => {
  assert.ok(Object.keys(OPS).length >= 87, 'batch 4B adds 8 more (95 — pinned in ip4b-S1)');
  assert.equal(Object.keys(FILTERS).length, 35); // +morph_tools (2026-10-18) +image_registration (2026-10-24)
  const NEEDS = { guided_filter: 'any', focus_score: 'gray', pseudocolor: 'gray', delta_e: 'any', roi_apply: 'any' };
  for (const id of NEW_OPS) {
    assert.ok(OPS[id], id); assert.equal(OPS[id].needs, NEEDS[id], `${id}.needs`);
    assert.equal(OP_TO_FILTER[id], HOST[id], `${id} host`);
    for (const p of OPS[id].params) assert.notEqual(p.key, 'mode', `${id}.${p.key}`);
    const host = FILTERS[HOST[id]], pre = defaultParams(HOST[id]);
    assert.ok(host.params.find((p) => p.key === 'mode').options.some((o) => o.value === id), `${id} is a mode of ${HOST[id]}`);
    for (const p of OPS[id].params) assert.deepEqual(pre[`${id}__${p.key}`], p.default, `${id}.${p.key} default`);
    assert.equal(needsOf(host, { mode: id }), NEEDS[id]);
  }
  assert.equal(FILTERS.image_quality.inputs, 2); assert.equal(FILTERS.image_arithmetic.inputs, 2);
  assert.equal(FILTERS.psnr.params.length, 0);
  const modes = (id) => FILTERS[id].params.find((p) => p.key === 'mode').options.map((o) => o.value);
  // legacy groups (op list + default) are exactly what they were before this batch
  assert.deepEqual(modes('colour'), ['color_inrange', 'kmeans_segment', 'white_balance', 'invert', 'slic']);
  assert.deepEqual(modes('blur'), ['median_blur', 'gaussian_blur', 'box_blur', 'bilateral', 'wiener_filter']);
  assert.deepEqual(modes('noise'), ['salt_pepper_noise', 'gaussian_noise']);
  assert.deepEqual(modes('image_adjust'), ['brightness_contrast', 'gamma', 'hist_equalize', 'clahe', 'contrast_stretch', 'flatfield_correction']);
  assert.deepEqual(modes('sharpen'), ['unsharp', 'laplacian_sharpen', 'custom_kernel']);
  assert.deepEqual(modes('threshold'), ['binary_threshold', 'otsu_threshold', 'adaptive_threshold', 'multi_otsu_threshold']);
  assert.deepEqual(modes('shape_analysis'), ['find_contours', 'connected_components', 'distance_transform', 'local_std_filter', 'local_entropy_filter', 'regionprops', 'feret', 'line_profile']);
  assert.deepEqual(modes('frequency').slice(0, 3), ['fft_magnitude', 'fft_filter', 'bandpass_notch']); // batch 4B appends dct_jpeg
  assert.deepEqual(modes('color_space'), ['grayscale', 'colorspace', 'channel_split', 'rgb_adjust', 'color_inrange', 'invert']);
  // the category groups = the palette sections, in order (cm-1) · defaults unchanged
  for (const [fid, title, dflt] of [['color_tools', 'COLOR', 'grayscale'], ['filter_denoise', 'FILTER & DENOISE', 'salt_pepper_noise'], ['analysis_detect', 'ANALYSIS & DETECTION', 'find_contours'], ['image_quality', 'IMAGE QUALITY', 'psnr'], ['image_arithmetic', 'IMAGE ARITHMETIC', 'imabsdiff']]) {
    assert.deepEqual(modes(fid), TOOL_SECTIONS.find((s) => s.title === title).ops, `${fid} == ${title}`);
    assert.equal(FILTERS[fid].params.find((p) => p.key === 'mode').default, dflt);
  }
  const sec = (t) => TOOL_SECTIONS.find((s) => s.title === t).ops;
  assert.deepEqual(sec('COLOR').slice(-2), ['invert', 'pseudocolor']);
  assert.deepEqual(sec('FILTER & DENOISE').slice(6, 8), ['wiener_filter', 'guided_filter']); // batch 4B appends nlm_denoise / deconv_wiener / deconv_lucy after them
  assert.ok(sec('ANALYSIS & DETECTION').indexOf('focus_score') > sec('ANALYSIS & DETECTION').indexOf('local_entropy_filter')); // batch 4B puts gabor_filter / glcm between them
  assert.equal(sec('ANALYSIS & DETECTION').indexOf('regionprops'), sec('ANALYSIS & DETECTION').indexOf('focus_score') + 1);
  assert.deepEqual(sec('IMAGE ARITHMETIC').slice(6, 8), ['hist_match', 'roi_apply']); // batch 4B appended ecc_register; 2026-10-24 it moved to IMAGE REGISTRATION
  assert.deepEqual(sec('IMAGE QUALITY'), ['psnr', 'ssim', 'delta_e']);
  for (const id of NEW_OPS) assert.equal(TOOL_SECTIONS.filter((s) => s.ops.includes(id)).length, 1, `${id} in exactly one section`);
});

test('ip4a-S2. (static) the 5 new OP_ICON glyphs are distinct from every other OP_ICON / GROUP_ICON glyph and from the markers 🧩 / 📊 / 📌, and appear nowhere in app.js / detail-panel.js / editor.js; OP_PRIMARY card text exists for every new op and reads the right values', () => {
  const src = readSrc('ui/palette-panel.js');
  const body = src.slice(src.indexOf('const OP_ICON = {'), src.indexOf('};', src.indexOf('const OP_ICON = {')));
  const icons = {}; for (const m of body.matchAll(/([a-z_0-9]+): "([^"]+)"/g)) icons[m[1]] = m[2];
  const groupGlyphs = new Set(Object.values(GROUP_ICON));
  for (const id of NEW_OPS) {
    assert.equal(icons[id], NEW_ICONS[id], `OP_ICON.${id}`);
    assert.ok(!Object.entries(icons).some(([k, v]) => k !== id && v === icons[id]), `${id} glyph collides with another OP_ICON`);
    assert.ok(!groupGlyphs.has(icons[id]), `${id} glyph collides with a GROUP_ICON`);
    assert.ok(!['🧩', '📊', '📌'].includes(icons[id]));
    for (const f of ['app.js', 'ui/detail-panel.js', 'graph/editor.js']) assert.ok(!readSrc(f).includes(icons[id]), `${id} glyph appears in ${f}`);
  }
  assert.equal(new Set(Object.values(NEW_ICONS)).size, 5);
  assert.equal(OP_PRIMARY.guided_filter({}), 'r=2 · ε 0.01'); assert.equal(OP_PRIMARY.guided_filter({ radius: 99, eps: 7 }), 'r=20 · ε 1'); assert.equal(OP_PRIMARY.guided_filter({ radius: 'x', eps: 'y' }), 'r=2 · ε 0.01');
  assert.equal(OP_PRIMARY.focus_score({}), 'Laplacian var · กลาง 100%'); assert.equal(OP_PRIMARY.focus_score({ method: 'tenengrad', roiPct: 50 }), 'Tenengrad · กลาง 50%'); assert.equal(OP_PRIMARY.focus_score({ method: 'constructor', roiPct: 1 }), 'Laplacian var · กลาง 10%');
  assert.equal(OP_PRIMARY.pseudocolor({}), 'jet · 0–255'); assert.equal(OP_PRIMARY.pseudocolor({ map: 'hot', range: 'minmax' }), 'hot · ยืดช่วง'); assert.equal(OP_PRIMARY.pseudocolor({ map: 'constructor' }), 'jet · 0–255');
  assert.equal(OP_PRIMARY.delta_e({}), 'CIE76 · ทีละพิกเซล · เกณฑ์ 2.3'); assert.equal(OP_PRIMARY.delta_e({ formula: 'ciede2000', compare: 'mean_b', tol: 5 }), 'CIEDE2000 · เทียบสีเฉลี่ย B · เกณฑ์ 5');
  assert.equal(OP_PRIMARY.roi_apply({}), 'สี่เหลี่ยม 25%,25% 50×50%'); assert.equal(OP_PRIMARY.roi_apply({ shape: 'ellipse', invert: true, feather: 4, x: 10, y: 20, w: 30, h: 40 }), 'วงรี 10%,20% 30×40% · กลับด้าน · ขอบนุ่ม 4px');
  // the modes of the old ops
  assert.equal(OP_PRIMARY.custom_kernel({ preset: 'motion', psfSize: 15, psfAngle: 30, amount: 1 }), 'motion 15px 30° · 1x'); assert.equal(OP_PRIMARY.custom_kernel({ preset: 'disk', psfSize: 4 }), 'disk r4 · 1x'); assert.equal(OP_PRIMARY.custom_kernel({ preset: 'box3' }), 'box3 · 1x');
  assert.equal(OP_PRIMARY.gaussian_noise({ kind: 'speckle', sigma: 30, seed: 4 }), 'speckle σ 30 · seed 4'); assert.equal(OP_PRIMARY.gaussian_noise({}), 'σ 20 · seed 1');
  assert.equal(OP_PRIMARY.gamma({ curve: 'log' }), 'log: c·log(1+r)'); assert.equal(OP_PRIMARY.gamma({}), 'gamma: 1.5');
  assert.equal(OP_PRIMARY.otsu_threshold({ method: 'triangle' }), 'วิธี: Triangle (อัตโนมัติ)'); assert.equal(OP_PRIMARY.otsu_threshold({}), 'วิธี: Otsu (อัตโนมัติ)');
  assert.equal(OP_PRIMARY.fft_filter({ shape: 'butterworth', order: 3, cutoff: 20 }), 'Low-pass · r=20 หน่วยความถี่ (butterworth n=3)'); assert.equal(OP_PRIMARY.fft_filter({}), 'Low-pass · r=30 หน่วยความถี่ (gaussian)');
  assert.equal(OP_PRIMARY.white_balance({ method: 'white_patch' }), 'ปรับสมดุลสี (White Patch)'); assert.equal(OP_PRIMARY.white_balance({ method: 'ref_point', refX: 30, refY: 70 }), 'ปรับสมดุลสี (จุดอ้างอิง 30%, 70%)'); assert.equal(OP_PRIMARY.white_balance({}), 'ปรับสมดุลสีอัตโนมัติ (Gray World)');
  assert.equal(OP_PRIMARY.regionprops({ minArea: 10, maxObjects: 5, mmPerPx: 0.2 }), 'วัดค่าต่อวัตถุ · พื้นที่ขั้นต่ำ 10px · แสดงสูงสุด 5 · 0.2 มม./px'); assert.ok(!OP_PRIMARY.regionprops({ mmPerPx: 0 }).includes('มม.'));
  assert.ok(OP_PRIMARY.feret({ mmPerPx: 1.5 }).endsWith(' · 1.5 มม./px')); assert.ok(!OP_PRIMARY.feret({}).includes('มม.')); assert.ok(OP_PRIMARY.line_profile({ mmPerPx: 0.5 }).endsWith(' · 0.5 มม./px')); assert.ok(!OP_PRIMARY.line_profile({}).includes('มม.'));
  assert.equal(OP_PRIMARY.colorspace({ space: 'luv' }), 'BGR → LUV');
});

test('ip4a-S3. (static read) every new numeric param is clamped to EXACTLY the UI default / min / max in registry.js (numIn / intIn / pctToPx) and in codegen.js (clampNum)', () => {
  const reg = readSrc('filters/registry.js'), gen = readSrc('codegen.js');
  const num = '(-?[\\d.]+)';
  const block = (source, opId) => {
    const start = source.indexOf(`\n  ${opId}: {`);
    assert.ok(start >= 0, `no "${opId}: {" block`);
    const next = /\n {2}[a-z_0-9]+: \{/g; next.lastIndex = start + 1;
    const m = next.exec(source);
    return source.slice(start, m ? m.index : source.length);
  };
  const spec = (id, k) => { const s = OPS[id].params.find((p) => p.key === k); assert.ok(s, `${id}.${k}`); return [s.default, s.min, s.max]; };
  const regRe = (k, fn = '(?:numIn|intIn)') => new RegExp(`${fn}\\(p\\.${k}, ${num}, ${num}, ${num}\\)`);
  // [opId, key, registry text to search, fn]
  const REG_WHOLE = reg; // helpers (whiteBalanceByGain) live outside the op block
  const checks = [
    ['custom_kernel', 'psfSize'], ['custom_kernel', 'psfAngle'], ['guided_filter', 'radius'], ['guided_filter', 'eps'], ['focus_score', 'roiPct'],
    ['delta_e', 'tol'], ['delta_e', 'scaleMax'], ['roi_apply', 'x'], ['roi_apply', 'y'], ['roi_apply', 'w'], ['roi_apply', 'h'], ['roi_apply', 'feather'],
    ['regionprops', 'mmPerPx'], ['feret', 'mmPerPx'], ['line_profile', 'mmPerPx'],
  ];
  for (const [id, k] of checks) {
    const m = block(reg, id).match(regRe(k)); assert.ok(m, `registry ${id}.${k}`);
    assert.deepEqual(m.slice(1).map(Number), spec(id, k), `registry ${id}.${k}`);
    const g = block(gen, id).match(new RegExp(`clampNum\\(p\\.${k}, ${num}, ${num}, ${num}\\)`)); assert.ok(g, `codegen ${id}.${k}`);
    assert.deepEqual(g.slice(1).map(Number), spec(id, k), `codegen ${id}.${k}`);
  }
  // fft_filter order · white_balance ref* live in a helper / a wrapper outside the original blocks
  assert.deepEqual(REG_WHOLE.match(regRe('order')).slice(1).map(Number), spec('fft_filter', 'order'));
  assert.deepEqual(gen.match(new RegExp(`clampNum\\(p\\.order, ${num}, ${num}, ${num}\\)`)).slice(1).map(Number), spec('fft_filter', 'order'));
  assert.deepEqual(REG_WHOLE.match(regRe('refSize')).slice(1).map(Number), spec('white_balance', 'refSize'));
  assert.deepEqual(gen.match(new RegExp(`clampNum\\(p\\.refSize, ${num}, ${num}, ${num}\\)`)).slice(1).map(Number), spec('white_balance', 'refSize'));
  for (const k of ['refX', 'refY']) {
    const [d, lo, hi] = spec('white_balance', k); assert.deepEqual([lo, hi], [0, 100]);
    assert.ok(new RegExp(`pctToPx\\(p\\.${k}, ${d}, [WH]\\)`).test(REG_WHOLE), `registry ${k}`);
    assert.deepEqual(gen.match(new RegExp(`clampNum\\(p\\.${k}, ${num}, ${num}, ${num}\\)`)).slice(1).map(Number), [d, 0, 100], `codegen ${k}`);
  }
  // the UI declares the numeric params that exist (no stray ones)
  const numeric = (id) => OPS[id].params.filter((p) => p.type === 'number').map((p) => p.key);
  assert.deepEqual(numeric('guided_filter'), ['radius', 'eps']); assert.deepEqual(numeric('focus_score'), ['roiPct']); assert.deepEqual(numeric('delta_e'), ['tol', 'scaleMax']); assert.deepEqual(numeric('roi_apply'), ['x', 'y', 'w', 'h', 'feather']);
  for (const id of ['regionprops', 'feret', 'line_profile']) assert.equal(OPS[id].params.at(-1).key, 'mmPerPx', `${id}: mmPerPx is the last param`);
  assert.deepEqual(OPS.custom_kernel.params.map((p) => p.key), ['preset', 'amount', 'psfSize', 'psfAngle']);
  assert.deepEqual(OPS.gaussian_noise.params.map((p) => p.key), ['sigma', 'seed', 'kind']); assert.deepEqual(OPS.gamma.params.map((p) => p.key), ['g', 'curve']); assert.deepEqual(OPS.otsu_threshold.params.map((p) => p.key), ['invert', 'method']);
  assert.deepEqual(OPS.fft_filter.params.map((p) => p.key), ['type', 'shape', 'cutoff', 'order']); assert.deepEqual(OPS.white_balance.params.map((p) => p.key), ['method', 'refX', 'refY', 'refSize']);
  assert.deepEqual(OPS.custom_kernel.params[0].options.map((o) => o.value), ['identity', 'sharpen', 'edge', 'emboss', 'outline', 'box3', 'box5', 'motion', 'disk']);
  assert.deepEqual(OPS.fft_filter.params[1].options.map((o) => o.value), ['gaussian', 'ideal', 'butterworth']);
});

test('ip4a-S4. (static) Thai text: no "ที่สุด" in any new name / desc / label / hint / option / info / VAL_META line; the required desc points are present; the select defaults are the old behaviour', () => {
  const texts = [];
  const walk = (o) => { if (typeof o === 'string') texts.push(o); else if (Array.isArray(o)) o.forEach(walk); else if (o && typeof o === 'object') Object.values(o).forEach(walk); };
  for (const id of NEW_OPS) walk({ name: OPS[id].name, desc: OPS[id].desc, params: OPS[id].params });
  const NEWPARAMS = { custom_kernel: ['psfSize', 'psfAngle'], gaussian_noise: ['kind'], gamma: ['curve'], otsu_threshold: ['method'], fft_filter: ['order'], white_balance: ['method', 'refX', 'refY', 'refSize'], regionprops: ['mmPerPx'], feret: ['mmPerPx'], line_profile: ['mmPerPx'] };
  for (const [id, keys] of Object.entries(NEWPARAMS)) for (const k of keys) walk(OPS[id].params.find((p) => p.key === k));
  for (const id of ['custom_kernel']) walk(OPS[id].params[0].options.slice(-2));
  walk(OPS.fft_filter.params[1].options.at(-1)); walk(OPS.colorspace.params[0].options.slice(-3));
  // the appended desc sentences of the old ops
  for (const [id, marker] of [['custom_kernel', 'Motion/Disk คือ PSF'], ['gaussian_noise', 'แบบ Speckle'], ['gamma', 'Log: ใหม่'], ['otsu_threshold', 'Triangle (Zack'], ['fft_filter', 'Butterworth: H'], ['white_balance', 'มี 3 สมมติฐาน'], ['colorspace', 'XYZ/Luv/YUV'], ['regionprops', 'แปลงเป็นหน่วยจริง'], ['feret', 'แปลงเป็นหน่วยจริง'], ['line_profile', 'แปลงเป็นหน่วยจริง']]) {
    const i = OPS[id].desc.indexOf(marker); assert.ok(i >= 0, `${id} desc tail "${marker}"`); texts.push(OPS[id].desc.slice(i));
  }
  const dp = readSrc('ui/detail-panel.js'); texts.push(dp.slice(dp.indexOf('// IP batch 4A — หน่วยมิลลิเมตร'), dp.indexOf('const META_FALLBACK')));
  for (const t of texts) assert.ok(!t.includes('ที่สุด'), `"ที่สุด" in: ${t.slice(0, 80)}`);
  // required desc points
  const d = (id) => OPS[id].desc;
  for (const re of [/ภาพนำทาง/, /self-guided/, /Bilateral/, /box filter/, /ทีละช่อง R\/G\/B/, /สะท้อน/, /Dehaze/]) assert.match(d('guided_filter'), re);
  for (const re of [/variance|แปรปรวน/, /ไม่มีหน่วย/, /เทียบ/, /autofocus/, /สว่าง/, /noise/, /Gaussian Blur 3×3/, /IPT/]) assert.match(d('focus_score'), re);
  for (const re of [/Distance Transform/, /ไม่ได้เพิ่มข้อมูล/, /ยืดช่วง/, /ind2rgb/, /applyColorMap/]) assert.match(d('pseudocolor'), re);
  for (const re of [/Lab/, /< ?1|ต่ำกว่า 1/, /CIEDE2000/, /Crop/, /White Balance/, /imcolordiff/, /บีบสเกล/]) assert.match(d('delta_e'), re);
  for (const re of [/เข้า A|เข้า B|ภาพเดียวกันแตกสองทาง/, /Mask/, /roifilt2/, /ROI Polygon/, /ย่อ\/ขยาย/]) assert.match(d('roi_apply'), re);
  assert.match(d('custom_kernel'), /fspecial/); assert.match(d('white_balance'), /chromadapt/); assert.match(d('white_balance'), /Bradford/); assert.match(d('gaussian_noise'), /uniform/);
  assert.match(d('regionprops'), /Undistort/); assert.match(d('otsu_threshold'), /ค่านั้น − 1/);
  // the selects default to the old behaviour
  const dflt = (id, k) => OPS[id].params.find((p) => p.key === k).default;
  assert.equal(dflt('gaussian_noise', 'kind'), 'additive'); assert.equal(dflt('gamma', 'curve'), 'power'); assert.equal(dflt('otsu_threshold', 'method'), 'otsu'); assert.equal(dflt('white_balance', 'method'), 'gray_world'); assert.equal(dflt('custom_kernel', 'preset'), 'sharpen');
  assert.equal(dflt('fft_filter', 'shape'), 'gaussian'); assert.equal(dflt('regionprops', 'mmPerPx'), 0);
  // every new key has a VAL_META line with decimals
  const vm = dp.slice(dp.indexOf('const VAL_META = {'), dp.indexOf('const META_FALLBACK'));
  for (const k of ['area_mm2', 'perimeter_mm', 'equiv_diameter_mm', 'major_axis_mm', 'minor_axis_mm', 'feret_max_mm', 'feret_min_mm', 'feret_len_mm', 'line_length_mm', 'edge_pos_mm', 'edge_pair_mm', 'focus_score', 'focus_mean', ...DE_KEYS, ...RA_KEYS]) assert.match(vm, new RegExp(`\\n  ${k}:\\s+\\{[^\\n]*d: \\d`), `VAL_META.${k}`);
  assert.match(vm, /area_mm2:\s+\{ th: "พื้นที่ \(มม\.²\)", d: 2, w: 92, pri: true, tip: "พื้นที่นับพิกเซล × \(มม\.\/px\)²" \}/);
  assert.match(vm, /de_over_pct:\s+\{ d: 1 \}/);
});

test('ip4a-S5. (static) codegen of the new paths: no "WARNING: skipped" / "ข้าม" / round( / ** 0.5 / import math / np.round / math. ; print() text is ASCII only; every new select reads own-property only ("constructor" = default); Python of the default paths has no new lines', () => {
  const OLD_OPS = new Set(['custom_kernel', 'gaussian_noise', 'gamma', 'otsu_threshold', 'fft_filter', 'white_balance', 'colorspace', 'regionprops', 'feret', 'line_profile']);
  const FORBID = ['WARNING: skipped', 'ข้าม', 'round(', '** 0.5', 'import math', 'np.round', 'math.'];
  const V = [
    ['custom_kernel', { preset: 'motion' }], ['custom_kernel', { preset: 'motion', psfSize: 21, psfAngle: 45, amount: 0.5 }], ['custom_kernel', { preset: 'disk', psfSize: 7 }],
    ['gaussian_noise', { kind: 'speckle' }], ['gaussian_noise', { kind: 'speckle', sigma: 0 }], ['gamma', { curve: 'log' }], ['otsu_threshold', { method: 'triangle' }], ['otsu_threshold', { method: 'triangle', invert: true }],
    ['fft_filter', { shape: 'butterworth' }], ['fft_filter', { shape: 'butterworth', type: 'high', order: 7, cutoff: 12 }],
    ['white_balance', { method: 'white_patch' }], ['white_balance', { method: 'ref_point', refX: 20, refY: 80, refSize: 9 }], ['colorspace', { space: 'xyz' }], ['colorspace', { space: 'luv' }], ['colorspace', { space: 'yuv' }],
    ['regionprops', { mmPerPx: 0.2 }], ['feret', { mmPerPx: 0.2 }], ['line_profile', { mmPerPx: 0.2 }],
    ['guided_filter', {}], ['guided_filter', { radius: 7, eps: 0.2 }], ['focus_score', {}], ['focus_score', { method: 'tenengrad', roiPct: 40, view: 'pass' }],
    ['pseudocolor', {}], ['pseudocolor', { map: 'hot', range: 'minmax' }], ['pseudocolor', { map: 'cool' }], ['delta_e', {}], ['delta_e', { formula: 'ciede2000', compare: 'mean_b', view: 'over', tol: 4 }],
    ['roi_apply', {}], ['roi_apply', { shape: 'ellipse', invert: true, feather: 5, view: 'outline' }],
  ];
  for (const [id, p] of V) {
    const code = buildPythonCode([OPSTEP(id, p)]);
    for (const bad of FORBID) assert.ok(!code.includes(bad), `${id} ${JSON.stringify(p)}: contains ${bad}`);
    // print() lines that are new in this batch are ASCII only (the old ops keep their old Thai print lines as they were)
    const oldLines = OLD_OPS.has(id) ? new Set(buildPythonCode([OPSTEP(id, {})]).split('\n')) : new Set();
    for (const line of code.split('\n')) { const m = line.match(/^\s*print\((.*)\)\s*(#.*)?$/); if (m && !oldLines.has(line)) assert.ok(/^[\x00-\x7f]*$/.test(m[1]), `${id}: non-ASCII print: ${line.slice(0, 100)}`); }
    assert.match(code, new RegExp(`# ── ขั้นที่ 1: ${OPS[id].name.replace(/[()]/g, '\\$&')} \\(${id}\\) ──`));
  }
  // default / unknown selects emit the old lines (golden covers the exact hashes; here the markers)
  const k = (id, p) => buildPythonCode([OPSTEP(id, p)]);
  assert.ok(!k('custom_kernel', { preset: 'constructor' }).includes('psf_kernel')); assert.ok(k('custom_kernel', { preset: 'constructor' }).includes('[[0,-1,0],[-1,5,-1],[0,-1,0]]'));
  assert.ok(!k('gaussian_noise', { kind: 'constructor' }).includes('speckle')); assert.ok(!k('gamma', { curve: 'constructor' }).includes('log1p')); assert.ok(!k('otsu_threshold', { method: 'constructor' }).includes('TRIANGLE'));
  assert.ok(!k('fft_filter', { shape: 'constructor' }).includes('Butterworth')); assert.ok(!k('white_balance', { method: 'constructor' }).includes('searchsorted')); assert.ok(k('white_balance', { method: 'constructor' }).includes('gain_r = gray_mean / r.mean()'));
  for (const id of ['regionprops', 'feret', 'line_profile']) for (const f of [0, -1, 'abc', undefined]) assert.ok(!k(id, f === undefined ? {} : { mmPerPx: f }).includes('mm_per_px'), `${id} mmPerPx ${f}`);
  assert.ok(k('pseudocolor', { map: 'constructor' }).includes('382.5 - np.abs(4 * i - 765)'));
  assert.ok(k('roi_apply', { shape: 'constructor', view: 'constructor' }).includes('inside = (ii >= x0)') && !k('roi_apply', { shape: 'constructor' }).includes('ww, hh'));
  // roi_apply Python keeps the box inside the image (x0 / y0 clamp to w_i - 1 / h_i - 1) -- JS-only guard for the twin
  { const c = k('roi_apply', {}); assert.ok(c.includes('x0 = min(w_i - 1, int(np.floor(w_i * ') && c.includes('y0 = min(h_i - 1, int(np.floor(h_i * ')); }
  assert.ok(!k('delta_e', { formula: 'constructor' }).includes('delta_e2000')); assert.ok(k('delta_e', { compare: 'constructor' }).includes('b_bgr = cv2.resize'));
  // the codegen mirrors: after Color Space the white-balance LUT channel order follows the app (no reversal)
  const afterCs = buildPythonCode([STEP('color_tools', 'colorspace', { colorspace__space: 'lab' }), OPSTEP('white_balance', { method: 'white_patch' })]);
  assert.ok(afterCs.includes('for ch in (0, 1, 2):') && !afterCs.includes('for ch in (2, 1, 0):'));
  assert.ok(k('white_balance', { method: 'white_patch' }).includes('for ch in (2, 1, 0):'));
  const gray2 = buildPythonCode([STEP('color_tools', 'grayscale'), OPSTEP('white_balance', { method: 'ref_point' })]); assert.ok(gray2.includes('gray image, nothing to adjust') && !gray2.includes('patch = img'));
  // 2-image scripts: delta_e / roi_apply use img_b and check .ndim at run time
  for (const id of ['delta_e', 'roi_apply']) { const c = k(id, {}); assert.ok(c.includes('img_b.ndim == 3') && c.includes('def process(path_a, path_b)')); }
  // psf_kernel / guided_f32 / delta_e2000 contain no bare `img` token (the B-chain rewrite replaces \bimg\b)
  for (const [id, p, fn] of [['custom_kernel', { preset: 'motion' }, 'def psf_kernel'], ['guided_filter', {}, 'def guided_f32'], ['delta_e', { formula: 'ciede2000' }, 'def delta_e2000']]) {
    const code = k(id, p), i = code.indexOf(fn), body = code.slice(i, code.indexOf('\n    \n', i) > 0 ? code.indexOf('\n    \n', i) : i + 3000);
    const fnLines = body.split('\n'); const end = fnLines.findIndex((l, j) => j > 0 && l.length > 0 && !l.startsWith('        ') && !l.startsWith('    def') && !/^\s{4}\s+/.test(l));
    assert.ok(!/\bimg\b/.test(fnLines.slice(0, end > 0 ? end : fnLines.length).join('\n')), `${fn} contains a bare img token`);
  }
});

// ===================================================================
// parity กับ Python จริง (cv2 5.0 / numpy 2.x) — หลายกรณีต่อ process เดียว
// ===================================================================
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'ip-batch4A-'));
process.on('exit', () => { try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ } });
function pythonAvailable() { try { execFileSync('python', ['-c', 'import cv2, numpy'], { encoding: 'utf8' }); return true; } catch { return false; } }
const HAS_PYTHON = !process.env.IP4A_NO_PY && pythonAvailable(); // IP4A_NO_PY=1 จำลองเครื่องที่ไม่มี cv2 (ใช้ตอน mutation แบบ JS-only)
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
            r = ns['process'](*c['inps'])
        r.tofile(c['out'])
        nd = int(r.ndim)
        shape = [int(v) for v in r.shape]
    except Exception as e:
        err = repr(e)
    res.append({'stdout': buf.getvalue(), 'ndim': nd, 'shape': shape, 'err': err})
json.dump(res, open(sys.argv[2], 'w'))
`;
// cases: [{ code, srcs: [Mat RGB 3ch (app order)], tail?: 'python lines before return' }] -> [{ stdout, bytes (app order for 3ch), shape, ndim }]
function runPythonBatch(cases, { swap = true } = {}) {
  const id = Math.random().toString(36).slice(2);
  const manifest = cases.map((c, i) => {
    let code = c.code;
    const [A, B] = c.srcs;
    assert.equal(A.channels(), 3, 'cv2.imread gives 3 channels');
    const rd = (name, m) => `np.fromfile(${name}, dtype=np.uint8).reshape((${m.rows}, ${m.cols}, 3))`;
    if (c.srcs.length === 1) code = code.replace('img = cv2.imread(path)', `img = ${rd('path', A)}`);
    else code = code.replace('img = cv2.imread(path_a)', `img = ${rd('path_a', A)}`).replace('img_b = cv2.imread(path_b)', `img_b = ${rd('path_b', B)}`);
    const mainIdx = code.indexOf('if __name__ == "__main__":');
    assert.ok(mainIdx > 0);
    code = code.slice(0, mainIdx);
    if (c.tail) code = code.replace('\n    return img\n', `\n${c.tail.split('\n').map((l) => '    ' + l).join('\n')}\n    return img\n`);
    const o = { code: path.join(TMP_DIR, `c_${id}_${i}.py`), inps: c.srcs.map((_, k) => path.join(TMP_DIR, `i_${id}_${i}_${k}.bin`)), out: path.join(TMP_DIR, `o_${id}_${i}.bin`) };
    fs.writeFileSync(o.code, code, 'utf8');
    c.srcs.forEach((m, k) => fs.writeFileSync(o.inps[k], swapRB3(Buffer.from(m.data))));
    return o;
  });
  const mp = path.join(TMP_DIR, `m_${id}.json`), rp = path.join(TMP_DIR, `r_${id}.json`), dp = path.join(TMP_DIR, `d_${id}.py`);
  fs.writeFileSync(mp, JSON.stringify(manifest)); fs.writeFileSync(dp, PY_DRIVER);
  execFileSync('python', [dp, mp, rp], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' }, maxBuffer: 1 << 28 });
  const res = JSON.parse(fs.readFileSync(rp, 'utf8'));
  return res.map((r, i) => {
    assert.equal(r.err, null, `case ${i}: python raised ${r.err}`);
    const raw = fs.readFileSync(manifest[i].out);
    for (const f of [...manifest[i].inps, manifest[i].out, manifest[i].code]) fs.rmSync(f, { force: true });
    return { stdout: r.stdout, shape: r.shape, ndim: r.ndim, bytes: r.ndim === 3 && swap ? swapRB3(raw) : Buffer.from(raw) };
  });
}
function appRun(pipe, src, srcB) {
  let cur = src.clone(), last = null;
  try {
    for (const st of pipe) {
      const f = FILTERS[st.id];
      let input = cur, conv = null;
      if (needsOf(f, st.params) === 'gray' && cur.channels() > 1) {
        conv = new cv.Mat(); cv.cvtColor(cur, conv, cur.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY); input = conv;
      }
      const ctx = f.inputs === 2 ? { src2: srcB } : {};
      try { const o = f.apply(cv, input, st.params, ctx); cur.delete(); cur = o; } finally { if (conv) conv.delete(); }
      last = ctx;
    }
    return { mat: cur, ctx: last };
  } catch (err) { cur.delete(); throw err; }
}
function diffStats(appMat, pyBytes) {
  const d = appMat.data;
  assert.equal(pyBytes.length, d.length, 'output size');
  let n = 0, mx = 0;
  for (let i = 0; i < d.length; i++) { const e = Math.abs(d[i] - pyBytes[i]); if (e) n++; if (e > mx) mx = e; }
  return { n, mx };
}
const PY_IMGS = () => [['640x480', smoothRGB(640, 480)], ['97x61', noiseRGB(97, 61, 21)], ['1x1', rgb(1, 1, () => [200, 90, 40])]];
const pyFmt = (v, d) => (Math.floor(v * 10 ** d + 0.5) / 10 ** d).toFixed(d);

test('ip4a-P1. parity psf_kernel: the Python twin gives EXACTLY the same kernel values (===, all elements) as psfKernel for every motion L 1..50 x theta {0,1,15,30,45,60,89,90,91,135,179} and every disk R 1..50 (V8 vs libm cos/sin agree on all of these)', PY_SKIP, () => {
  const THETAS = [0, 1, 15, 30, 45, 60, 89, 90, 91, 135, 179];
  const code = buildPythonCode([OPSTEP('custom_kernel', { preset: 'motion' })]);
  const i0 = code.indexOf('    def psf_kernel'), i1 = code.indexOf('    return cnt / cnt.sum()') + '    return cnt / cnt.sum()'.length;
  assert.ok(i0 > 0 && i1 > i0);
  const def = code.slice(i0, i1).split('\n').map((l) => l.slice(4)).join('\n');
  const script = `import numpy as np, json\n${def}\nout = []\nfor L in range(1, 51):\n    for t in ${JSON.stringify(THETAS)}:\n        k = psf_kernel("motion", L, t)\n        out.append([k.shape[0]] + [float(v) for v in k.ravel()])\nfor R in range(1, 51):\n    k = psf_kernel("disk", R, 0)\n    out.append([k.shape[0]] + [float(v) for v in k.ravel()])\nprint(json.dumps(out))\n`;
  const f = path.join(TMP_DIR, `psf_${Math.random().toString(36).slice(2)}.py`);
  fs.writeFileSync(f, script, 'utf8');
  const res = JSON.parse(execFileSync('python', [f], { encoding: 'utf8', maxBuffer: 1 << 28 }));
  let idx = 0, bad = [];
  for (let L = 1; L <= 50; L++) for (const t of THETAS) {
    const js = psfKernel('motion', L, t), py = res[idx++];
    if (py[0] !== js.n || py.length !== 1 + js.k.length || js.k.some((v, j) => v !== py[1 + j])) bad.push(`motion L${L} theta${t}`);
  }
  for (let R = 1; R <= 50; R++) { const js = psfKernel('disk', R, 0), py = res[idx++]; if (py[0] !== js.n || js.k.some((v, j) => v !== py[1 + j])) bad.push(`disk R${R}`); }
  assert.deepEqual(bad, [], `${bad.length} kernels differ between JS and Python: ${bad.slice(0, 6).join(', ')}`);
});

test('ip4a-P2. parity custom_kernel motion / disk (exported Python == app) on 640x480 / 97x61 / 1x1 colour: max |diff| <= 1 level', PY_SKIP, () => {
  const imgs = PY_IMGS();
  try {
    const params = [{ preset: 'motion' }, { preset: 'motion', psfSize: 21, psfAngle: 30, amount: 0.5 }, { preset: 'disk', psfSize: 6 }, { preset: 'motion', psfSize: 15, psfAngle: 135 }];
    const jobs = [];
    for (const [name, m] of imgs) for (const p of params) jobs.push({ name, m, p, pipe: [OPSTEP('custom_kernel', p)] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try { const { mx, n } = diffStats(app.mat, py[i].bytes); assert.ok(mx <= 1, `${j.name} ${JSON.stringify(j.p)}: max diff ${mx} (${n} px)`); } finally { app.mat.delete(); }
    });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4a-P3. parity speckle noise: ndiff 0 on 640x480 / 97x61 / 1x1 colour; the printed clipped % equals the app info', PY_SKIP, () => {
  const imgs = PY_IMGS();
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const p of [{ kind: 'speckle', sigma: 30, seed: 3 }, { kind: 'speckle', sigma: 100, seed: 77 }, { kind: 'speckle', sigma: 0 }]) jobs.push({ name, m, p, pipe: [OPSTEP('gaussian_noise', p)] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        assert.equal(diffStats(app.mat, py[i].bytes).n, 0, `${j.name} ${JSON.stringify(j.p)}`);
        if (j.p.sigma > 0) {
          const want = app.ctx.info.match(/ชนขอบ 0\/255 ([\d.]+)% ของค่า/)[1], got = py[i].stdout.match(/clipped at 0\/255 ([\d.]+)% of values/)[1];
          assert.equal(got, want, `${j.name}: clipped %`);
          assert.match(py[i].stdout, new RegExp(`^speckle noise sigma=${j.p.sigma} \\(seed ${j.p.seed}\\)`));
        }
      } finally { app.mat.delete(); }
    });
    // speckle != additive in Python too
    const add = runPythonBatch([{ code: buildPythonCode([OPSTEP('gaussian_noise', { sigma: 30, seed: 3 })]), srcs: [imgs[1][1]] }]);
    assert.notDeepEqual(Array.from(add[0].bytes), Array.from(py[3].bytes));
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4a-P4. parity gamma log: ndiff 0 on 640x480 / 97x61 / 1x1 colour and after a grayscale step', PY_SKIP, () => {
  const imgs = PY_IMGS();
  try {
    const jobs = [];
    for (const [name, m] of imgs) { jobs.push({ name, m, pipe: [OPSTEP('gamma', { curve: 'log' })] }); jobs.push({ name: `${name} gray`, m, pipe: [STEP('color_tools', 'grayscale'), OPSTEP('gamma', { curve: 'log' })] }); }
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => { const app = appRun(j.pipe, j.m); try { assert.equal(diffStats(app.mat, py[i].bytes).n, 0, j.name); assert.match(py[i].stdout, /log transform: s = 255\*ln\(1\+r\)\/ln\(256\)/); } finally { app.mat.delete(); } });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4a-P5. parity otsu triangle (+ invert): ndiff 0 on 640x480 / 97x61 / 1x1 / a single-peak scene', PY_SKIP, () => {
  const imgs = [...PY_IMGS(), ['blob', blobColor()], ['flat 0', rgb(20, 10, () => [0, 0, 0])], ['flat 128', rgb(20, 10, () => [128, 128, 128])], ['flat 255', rgb(20, 10, () => [255, 255, 255])]];
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const invert of [false, true]) jobs.push({ name: `${name} inv${invert}`, m, pipe: [OPSTEP('otsu_threshold', { method: 'triangle', invert })] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => { const app = appRun(j.pipe, j.m); try { assert.equal(diffStats(app.mat, py[i].bytes).n, 0, j.name); } finally { app.mat.delete(); } });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4a-P6. parity fft_filter butterworth (low / high, n 1 / 2 / 6 / 10): max |diff| <= 1 level on 640x480 / 97x61 gray (and the Python script is the old one for gaussian / ideal)', PY_SKIP, () => {
  const imgs = [['640x480', smoothRGB(640, 480)], ['97x61', noiseRGB(97, 61, 5)]];
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const [type, order, cutoff] of [['low', 2, 30], ['high', 2, 30], ['low', 1, 12], ['low', 6, 40], ['high', 10, 20]]) jobs.push({ name: `${name} ${type} n${order}`, m, pipe: [OPSTEP('fft_filter', { shape: 'butterworth', type, order, cutoff })] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => { const app = appRun(j.pipe, j.m); try { const { mx, n } = diffStats(app.mat, py[i].bytes); assert.ok(mx <= 1, `${j.name}: max diff ${mx} (${n} px)`); } finally { app.mat.delete(); } });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4a-P7. parity white_balance white_patch / ref_point: ndiff 0 on 640x480 / 97x61 / 1x1 / a tinted grey-card scene (RGB) and after a Color Space step (no channel reversal); the printed p99 / gains equal the app info', PY_SKIP, () => {
  const r = lcg(2);
  const card = rgb(80, 60, (x, y) => (x >= 30 && x < 50 && y >= 20 && y < 40 ? [156, 120, 84] : [Math.floor(r() * 256), Math.floor(r() * 256), Math.floor(r() * 256)]));
  const imgs = [...PY_IMGS(), ['card', card]];
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const p of [{ method: 'white_patch' }, { method: 'ref_point' }, { method: 'ref_point', refX: 0, refY: 100, refSize: 30 }, { method: 'ref_point', refX: 100, refY: 0, refSize: 50 }]) jobs.push({ name: `${name} ${JSON.stringify(p)}`, m, pipe: [OPSTEP('white_balance', p)] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        assert.equal(diffStats(app.mat, py[i].bytes).n, 0, j.name);
        const g = (s, re) => s.match(re).slice(1).map(Number);
        if (j.pipe[0].params.white_balance__method === 'white_patch') {
          assert.deepEqual(g(py[i].stdout, /p99 R\/G\/B = (\d+)\/(\d+)\/(\d+) /), g(app.ctx.info, /ค่าที่ 99% R\/G\/B = (\d+)\/(\d+)\/(\d+) /), `${j.name}: p99`);
        }
        assert.deepEqual(g(py[i].stdout, /gain R x([\d.]+) G x([\d.]+) B x([\d.]+)/), g(app.ctx.info, /เกน R×([\d.]+) G×([\d.]+) B×([\d.]+)/), `${j.name}: gains`);
      } finally { app.mat.delete(); }
    });
    // channel order after Color Space (the app keeps R,G,B -> cv2 sees the converted planes in the same order)
    const chained = [{ name: 'after hsv, white_patch', pipe: [STEP('color_tools', 'colorspace', { colorspace__space: 'hsv' }), OPSTEP('white_balance', { method: 'white_patch' })] }, { name: 'after lab, ref_point', pipe: [STEP('color_tools', 'colorspace', { colorspace__space: 'lab' }), OPSTEP('white_balance', { method: 'ref_point' })] }];
    const py2 = runPythonBatch(chained.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [imgs[1][1]] })), { swap: false });
    chained.forEach((j, i) => { const app = appRun(j.pipe, imgs[1][1]); try { assert.equal(diffStats(app.mat, py2[i].bytes).n, 0, j.name); } finally { app.mat.delete(); } });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4a-P8. parity colorspace xyz / luv / yuv: ndiff 0 on 640x480 / 97x61 / 1x1 colour (planes keep the app order, no channel reversal)', PY_SKIP, () => {
  const imgs = PY_IMGS();
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const space of ['xyz', 'luv', 'yuv', 'lab', 'hsv']) jobs.push({ name: `${name} ${space}`, m, pipe: [OPSTEP('colorspace', { space })] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })), { swap: false });
    jobs.forEach((j, i) => { const app = appRun(j.pipe, j.m); try { assert.equal(diffStats(app.mat, py[i].bytes).n, 0, j.name); } finally { app.mat.delete(); } });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4a-P9. parity mmPerPx: the Python mm table / lines print the same numbers (2 decimals, floor(v*100+0.5)/100) as the app columns / fields for regionprops, feret and line_profile; the old tables / lines are unchanged and still first', PY_SKIP, () => {
  const scene = objectsScene(), sceneC = rgbFromGray(scene), noise = noiseRGB(120, 80, 9);
  try {
    const f = 0.37;
    const jobs = [
      { name: 'regionprops', m: sceneC, pipe: [OPSTEP('regionprops', { minArea: 0, mmPerPx: f })] },
      { name: 'feret', m: sceneC, pipe: [OPSTEP('feret', { minArea: 0, mmPerPx: f })] },
      { name: 'line_profile edge', m: sceneC, pipe: [OPSTEP('line_profile', { x1: 5, y1: 20, x2: 95, y2: 20, mmPerPx: f })] },
      { name: 'line_profile noise', m: noise, pipe: [OPSTEP('line_profile', { mmPerPx: f })] },
      { name: 'line_profile flat', m: rgb(80, 70, () => [90, 90, 90]), pipe: [OPSTEP('line_profile', { mmPerPx: f })] },
    ];
    const plain = jobs.map((j) => ({ ...j, pipe: [{ ...j.pipe[0], params: { ...j.pipe[0].params, [`${j.pipe[0].params.mode}__mmPerPx`]: 0 } }] }));
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    const py0 = runPythonBatch(plain.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        assert.ok(py[i].stdout.startsWith(py0[i].stdout), `${j.name}: the old output is a prefix of the mm output`);
        const extra = py[i].stdout.slice(py0[i].stdout.length).split('\n').filter(Boolean);
        const V = app.ctx.values;
        if (j.name === 'regionprops' || j.name === 'feret') {
          const spec = MM_COLUMNS[j.name], cols = V.columns;
          const hdr = j.name === 'regionprops' ? 'object  area_mm2  perimeter_mm  equiv_diam_mm  major_axis_mm  minor_axis_mm' : 'object  feret_max_mm  feret_min_mm  feret_len_mm';
          assert.equal(extra[0], hdr);
          V.rows.slice(0, 20).forEach((row, k) => {
            const want = `${k + 1}  ${spec.map(([key]) => pyFmt(row[cols.indexOf(key)], 2)).join('  ')}`;
            assert.equal(extra[1 + k], want, `${j.name} row ${k + 1}`);
          });
          assert.ok(!extra[0].startsWith('index  area'));
        } else {
          const m = scalarMap(app.ctx), line = (key) => (m[key] === null ? '-' : `${pyFmt(m[key], 2)} mm`);
          assert.deepEqual(extra, [`line length: ${line('line_length_mm')}`, `edge position from A: ${line('edge_pos_mm')}`, `edge pair distance: ${line('edge_pair_mm')}`]);
        }
      } finally { app.mat.delete(); }
    });
    jobs[4].m.delete();
  } finally { for (const m of [scene, sceneC, noise]) m.delete(); }
});

test('ip4a-P10. parity guided_filter (float32 twin): max |diff| <= 1 level and <= 0.1% of the pixels differ -- gray and colour, r 2 / 8, eps 0.01 / 0.3, on 640x480 / 97x61 / 1x1', PY_SKIP, () => {
  const imgs = PY_IMGS();
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const gray of [false, true]) for (const p of [{}, { radius: 8, eps: 0.3 }]) jobs.push({ name: `${name} ${gray ? 'gray' : 'rgb'} ${JSON.stringify(p)}`, m, pipe: [...(gray ? [STEP('color_tools', 'grayscale')] : []), OPSTEP('guided_filter', p)] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        const { mx, n } = diffStats(app.mat, py[i].bytes);
        assert.ok(mx <= 1, `${j.name}: max diff ${mx}`); assert.ok(n <= Math.max(1, app.mat.data.length * 0.001), `${j.name}: ${n} of ${app.mat.data.length} bytes differ`);
        assert.match(py[i].stdout, /^guided filter r=\d+ eps=[\d.]+/);
      } finally { app.mat.delete(); }
    });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4a-P11. parity focus_score: the score is EXACTLY equal (===) in app and Python for both methods and roiPct 100 / 50 / 20; the |Laplacian| image has ndiff 0; the printed text equals the app info', PY_SKIP, () => {
  const imgs = [...PY_IMGS(), ['blob', blobColor()], ['flat', rgb(30, 20, () => [100, 100, 100])]];
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const p of [{}, { method: 'tenengrad' }, { roiPct: 50 }, { method: 'tenengrad', roiPct: 20, view: 'pass' }]) jobs.push({ name: `${name} ${JSON.stringify(p)}`, m, p, pipe: [OPSTEP('focus_score', p)] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m], tail: "print('SCORE_REPR', repr(float(score)), repr(float(mean_gray)))" })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        const [, sc, mn] = py[i].stdout.match(/SCORE_REPR (\S+) (\S+)/);
        const m = scalarMap(app.ctx);
        assert.equal(Number(sc), m.focus_score, `${j.name}: score`); assert.equal(Number(mn), m.focus_mean, `${j.name}: mean`);
        assert.equal(diffStats(app.mat, py[i].bytes).n, 0, `${j.name}: image`);
        const pyTxt = py[i].stdout.match(/focus score \((.*?)\) = ([\d.]+) in (\d+)x(\d+) px/);
        assert.equal(pyTxt[1], j.p.method === 'tenengrad' ? 'Tenengrad' : 'variance of Laplacian');
        assert.equal(pyTxt[2], pyFmt(m.focus_score, 1)); assert.ok(app.ctx.info.includes(`${pyTxt[3]}×${pyTxt[4]} px`));
      } finally { app.mat.delete(); }
    });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4a-P12. parity pseudocolor: ndiff 0 for jet / hot / cool x full / minmax on 640x480 / 97x61 / 1x1 / a two-level image', PY_SKIP, () => {
  const imgs = [...PY_IMGS(), ['two-level', rgb(30, 20, (x) => (x < 10 ? [60, 60, 60] : [180, 180, 180]))], ['flat', rgb(10, 6, () => [77, 77, 77])]];
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const map of ['jet', 'hot', 'cool']) for (const range of ['full', 'minmax']) jobs.push({ name: `${name} ${map} ${range}`, m, pipe: [OPSTEP('pseudocolor', { map, range })] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => { const app = appRun(j.pipe, j.m); try { assert.equal(diffStats(app.mat, py[i].bytes).n, 0, j.name); assert.equal(py[i].ndim, 3); } finally { app.mat.delete(); } });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4a-P13. parity delta_e: CIE76 image ndiff 0 and every printed / returned number equal (<= 1e-9); CIEDE2000 values <= 1e-9 and image <= 1 level -- pixel and mean_b compare, B of the same / another size, over / heatmap', PY_SKIP, () => {
  const A640 = smoothRGB(640, 480), B640 = noiseRGB(640, 480, 5), A = noiseRGB(97, 61, 3), Bsame = noiseRGB(97, 61, 4), Bsmall = noiseRGB(30, 20, 8), one = rgb(1, 1, () => [10, 200, 30]), oneB = rgb(1, 1, () => [200, 10, 90]);
  try {
    const cases = [];
    for (const formula of ['cie76', 'ciede2000']) {
      for (const [name, a, b] of [['640x480', A640, B640], ['97x61', A, Bsame], ['97x61 vs small B', A, Bsmall], ['1x1', one, oneB]]) {
        for (const compare of ['pixel', 'mean_b']) for (const view of ['heatmap', 'over']) cases.push({ name: `${formula} ${name} ${compare} ${view}`, a, b, formula, p: { formula, compare, view, tol: 5, scaleMax: 40 } });
      }
    }
    const jobs = cases.map((c) => ({ ...c, pipe: [OPSTEP('delta_e', c.p)] }));
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.a, j.b], tail: "print('VALS', repr(float(de_mean)), repr(float(de.max())), repr(float(over_pct)), *[repr(float(v)) for v in mean_a], *[repr(float(v)) for v in mean_b])" })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.a, j.b);
      try {
        const { mx, n } = diffStats(app.mat, py[i].bytes);
        if (j.formula === 'cie76') assert.equal(n, 0, `${j.name}: image ndiff`); else assert.ok(mx <= 1, `${j.name}: max diff ${mx}`);
        const vals = py[i].stdout.match(/VALS (.*)/)[1].split(' ').map(Number), m = scalarMap(app.ctx);
        DE_KEYS.forEach((k, q) => near(vals[q], m[k], 1e-9, `${j.name}: ${k}`));
        assert.match(py[i].stdout, new RegExp(`delta E \\(${j.formula === 'cie76' ? 'CIE76' : 'CIEDE2000'}\\): mean ${pyFmt(m.de_mean, 2)} max ${pyFmt(m.de_max, 2)} over 5: ${pyFmt(m.de_over_pct, 1)}%`));
      } finally { app.mat.delete(); }
    });
  } finally { for (const m of [A640, B640, A, Bsame, Bsmall, one, oneB]) m.delete(); }
});

test('ip4a-P14. parity roi_apply: feather 0 ndiff 0 (rect / ellipse / invert / outline, B of the same or another size, 640x480 / 97x61 / 1x1); feather > 0 <= 1 level; the printed count equals the app value', PY_SKIP, () => {
  const A640 = smoothRGB(640, 480), B640 = noiseRGB(640, 480, 5), A = noiseRGB(97, 61, 3), Bsame = noiseRGB(97, 61, 4), Bsmall = noiseRGB(30, 20, 8), one = rgb(1, 1, () => [10, 200, 30]), oneB = rgb(1, 1, () => [200, 10, 90]);
  try {
    const jobs = [];
    for (const [name, a, b] of [['640x480', A640, B640], ['97x61', A, Bsame], ['97x61 small B', A, Bsmall], ['1x1', one, oneB]]) {
      for (const p of [{}, { shape: 'ellipse' }, { invert: true }, { shape: 'ellipse', invert: true, view: 'outline' }, { view: 'outline', x: 0, y: 0, w: 100, h: 100 }, { x: 80, y: 90, w: 60, h: 70 }, { feather: 5 }, { shape: 'ellipse', feather: 12, invert: true, view: 'outline' }, { x: 100, y: 100, w: 10, h: 10 }]) jobs.push({ name: `${name} ${JSON.stringify(p)}`, a, b, p, pipe: [OPSTEP('roi_apply', p)] });
    }
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.a, j.b] })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.a, j.b);
      try {
        const { mx, n } = diffStats(app.mat, py[i].bytes);
        if (!j.p.feather) assert.equal(n, 0, `${j.name}: ndiff`); else assert.ok(mx <= 1, `${j.name}: max diff ${mx}`);
        const m = scalarMap(app.ctx);
        assert.equal(Number(py[i].stdout.match(/roi apply: (\d+) px use B/)[1]), m.ra_px, `${j.name}: count`);
        assert.equal(py[i].stdout.match(/\(([\d.]+)%\)/)[1], pyFmt(m.ra_pct, 1));
      } finally { app.mat.delete(); }
    });
  } finally { for (const m of [A640, B640, A, Bsame, Bsmall, one, oneB]) m.delete(); }
});

// ===================================================================
// ข้อกำหนดร่วม: ห้ามแก้ src / ctx.src2 · คืน Mat ใหม่เสมอ (แม้ 0×0) · เส้นทาง "ไม่มีค่า" คืน values ชุด key เดิมเสมอ
// ===================================================================
test('ip4a-T8. no new op / mode ever modifies src or ctx.src2, and each returns a NEW Mat (also for 0x0 and for the passthrough paths)', () => {
  const g = noiseGray(24, 18, 1), c = noiseRGB(24, 18, 2), a = toRGBA(c), one = gray(1, 1, () => 5), e = new cv.Mat(), B = noiseRGB(24, 18, 7), Bs = noiseRGB(8, 6, 9), scene = objectsScene();
  const jobs = [];
  const ADD = (id, p, srcs, withB) => { for (const s of srcs) jobs.push({ id, p, s, withB }); };
  const ALL = [g, c, a, one, e];
  ADD('custom_kernel', { preset: 'motion', psfSize: 11, psfAngle: 20 }, ALL); ADD('custom_kernel', { preset: 'disk', psfSize: 4, amount: 0.5 }, ALL);
  ADD('guided_filter', {}, ALL); ADD('focus_score', { roiPct: 50 }, ALL); ADD('focus_score', { view: 'pass', method: 'tenengrad' }, ALL); ADD('pseudocolor', { range: 'minmax' }, ALL);
  ADD('gaussian_noise', { kind: 'speckle' }, ALL); ADD('gamma', { curve: 'log' }, ALL); ADD('otsu_threshold', { method: 'triangle' }, [g, one, e]);
  ADD('white_balance', { method: 'white_patch' }, ALL); ADD('white_balance', { method: 'ref_point' }, ALL); ADD('colorspace', { space: 'luv' }, [c, g]);
  ADD('regionprops', { minArea: 0, mmPerPx: 0.5 }, [scene, e]); ADD('feret', { minArea: 0, mmPerPx: 0.5 }, [scene, e]); ADD('line_profile', { mmPerPx: 0.5 }, [c, e]);
  ADD('delta_e', { formula: 'ciede2000' }, [g, c, a, one, e], B); ADD('delta_e', { compare: 'mean_b' }, [c, a], Bs); ADD('roi_apply', { feather: 3, view: 'outline' }, [g, c, a, one, e], B); ADD('roi_apply', {}, [c], Bs);
  ADD('delta_e', {}, [c], null); ADD('roi_apply', {}, [c], null); ADD('delta_e', {}, [c], e); ADD('roi_apply', {}, [c], e);
  try {
    for (const j of jobs) {
      const label = `${j.id} ${JSON.stringify(j.p)} ${j.s.rows}x${j.s.cols}x${j.s.channels()}`;
      const sb = bytes(j.s), bb = j.withB ? bytes(j.withB) : null;
      const ctx = j.withB === undefined ? {} : j.withB === null ? {} : { src2: j.withB };
      const gin = OPS[j.id].needs === 'gray' && j.s.channels() > 1 ? (() => { const t = new cv.Mat(); cv.cvtColor(j.s, t, j.s.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY); return t; })() : null;
      const input = gin || j.s;
      let out;
      try { out = OPS[j.id].apply(cv, input, j.p, ctx); } catch (err) { if (j.s.rows === 0) { assert.ok(true); continue; } throw err; }
      try {
        assert.notEqual(out.$$.ptr, input.$$.ptr, `${label}: returned the input Mat itself`);
        assert.deepEqual(bytes(j.s), sb, `${label}: src modified`);
        if (bb) assert.deepEqual(bytes(j.withB), bb, `${label}: ctx.src2 modified`);
        if (j.withB) assert.ok(!j.withB.isDeleted(), `${label}: ctx.src2 was deleted`);
        // every path of the values-bearing ops carries the same keys
        if (ctx.values && ctx.values.kind === 'scalars') {
          const want = { focus_score: ['focus_score', 'focus_mean'], delta_e: DE_KEYS, roi_apply: RA_KEYS }[j.id];
          if (want) assert.deepEqual(ctx.values.fields.map((f) => f.key), want, `${label}: keys`);
        }
      } finally { out.delete(); if (gin) gin.delete(); }
    }
  } finally { for (const m of [g, c, a, one, e, B, Bs, scene]) m.delete(); }
});

test('ip4a-T9. the 0x0 contract of the new paths: every new op / mode returns a clone + the info "ภาพว่าง" (or the passthrough text) and ctx.values with every key present and null; no old-op default path gained a guard (the old 0x0 behaviour is unchanged)', () => {
  const e = new cv.Mat(), B = noiseRGB(6, 5, 3);
  try {
    const cases = [
      ['guided_filter', {}, undefined], ['focus_score', {}, undefined], ['pseudocolor', {}, undefined], ['delta_e', {}, B], ['roi_apply', {}, B],
      ['custom_kernel', { preset: 'motion' }, undefined], ['custom_kernel', { preset: 'disk' }, undefined], ['gamma', { curve: 'log' }, undefined], ['otsu_threshold', { method: 'triangle' }, undefined],
      ['white_balance', { method: 'white_patch' }, undefined], ['white_balance', { method: 'ref_point' }, undefined], ['fft_filter', { shape: 'butterworth' }, undefined],
    ];
    for (const [id, p, b] of cases) {
      const ctx = b ? { src2: b } : {};
      const out = OPS[id].apply(cv, e, p, ctx);
      assert.equal(out.rows, 0, id); assert.equal(out.cols, 0, id); assert.notEqual(out.$$.ptr, e.$$.ptr, `${id}: must not return the input`); assert.equal(ctx.info, 'ภาพว่าง', id);
      if (ctx.values) assert.ok(ctx.values.fields.every((f) => f.value === null), `${id}: values must all be null`);
      out.delete();
    }
    // the OLD default paths on 0x0 behave as before (no new guard): colorspace / fft_filter / gamma power / otsu / white_balance gray_world
    for (const [id, p] of [['gamma', {}], ['otsu_threshold', {}], ['white_balance', {}], ['colorspace', {}], ['custom_kernel', { preset: 'sharpen' }]]) {
      let threw = false, out = null;
      try { out = OPS[id].apply(cv, e, p, {}); } catch { threw = true; }
      if (out) { assert.equal(out.rows, 0); out.delete(); }
      // measured on HEAD 1755233: only Otsu returns an empty Mat on 0x0; the others throw a C++ exception (kept exactly as before)
      assert.equal(threw, id !== 'otsu_threshold', `${id}: old 0x0 behaviour (threw=${threw})`);
    }
  } finally { e.delete(); B.delete(); }
});

test('ip4a-T10. the psf direction follows MATLAB (angle counter-clockwise on screen): a 45 degree motion kernel weighs the upper-right quadrant more than the lower-right one; 135 degrees the opposite; disk is symmetric under both flips', () => {
  const quad = (psf) => { const n = psf.n, h = (n - 1) / 2; let ur = 0, lr = 0, ul = 0, ll = 0; for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const v = psf.k[y * n + x], dx = x - h, dy = y - h; if (dx > 0 && dy < 0) ur += v; if (dx > 0 && dy > 0) lr += v; if (dx < 0 && dy < 0) ul += v; if (dx < 0 && dy > 0) ll += v; } return { ur, lr, ul, ll }; };
  const a = quad(psfKernel('motion', 15, 45)), b = quad(psfKernel('motion', 15, 135));
  assert.ok(a.ur > 10 * a.lr && a.ll > 10 * a.ul, `45 deg: ${JSON.stringify(a)}`);
  assert.ok(b.ul > 10 * b.ll && b.lr > 10 * b.ur, `135 deg: ${JSON.stringify(b)}`);
  const d = psfKernel('disk', 6, 0);
  for (let y = 0; y < d.n; y++) for (let x = 0; x < d.n; x++) { assert.equal(d.k[y * d.n + x], d.k[x * d.n + y], 'disk transpose symmetric'); assert.equal(d.k[y * d.n + x], d.k[y * d.n + (d.n - 1 - x)], 'disk mirror symmetric'); }
});

test('ip4a-psf5. psfKernel is pinned (measured at SS = 16): sha256 of every kernel (motion L 1..50 x 11 angles, disk R 1..50) and exact literals of four small kernels -- a JS-only guard for the Python-exact twin', () => {
  const h = crypto.createHash('sha256');
  for (let L = 1; L <= 50; L++) for (const t of [0, 1, 15, 30, 45, 60, 89, 90, 91, 135, 179]) { const p = psfKernel('motion', L, t); h.update(p.n + ':'); h.update(Buffer.from(p.k.buffer, p.k.byteOffset, p.k.byteLength)); }
  for (let R = 1; R <= 50; R++) { const p = psfKernel('disk', R, 0); h.update(p.n + ':'); h.update(Buffer.from(p.k.buffer, p.k.byteOffset, p.k.byteLength)); }
  assert.equal(h.digest('hex'), '54c0e848188c8df4d909307033f9150252a20b66c8f337cc486294eeacff01ae');
  assert.deepEqual(Array.from(psfKernel('motion', 3, 0).k), [0, 0, 0, 0.32222222222222224, 0.35555555555555557, 0.32222222222222224, 0, 0, 0]);
  assert.deepEqual(Array.from(psfKernel('motion', 2, 0).k), [0, 0, 0, 0.22413793103448276, 0.5517241379310345, 0.22413793103448276, 0, 0, 0]);
  assert.deepEqual(Array.from(psfKernel('disk', 1, 0).k), [0.02586206896551724, 0.14532019704433496, 0.02586206896551724, 0.14532019704433496, 0.31527093596059114, 0.14532019704433496, 0.02586206896551724, 0.14532019704433496, 0.02586206896551724]);
  assert.deepEqual(Array.from(psfKernel('motion', 5, 45).k), [0, 0, 0, 0.04260450160771704, 0.024115755627009645, 0, 0, 0.05305466237942122, 0.18971061093247588, 0.04260450160771704, 0, 0.05305466237942122, 0.18971061093247588, 0.05305466237942122, 0, 0.04260450160771704, 0.18971061093247588, 0.05305466237942122, 0, 0, 0.024115755627009645, 0.04260450160771704, 0, 0, 0]);
});

// ===================================================================
// พิสูจน์ "mutant เทียบเท่า" ที่พบตอนทำ mutation testing (บันทึกไว้เป็นเทสต์ ไม่ใช่ช่องโหว่)
// ===================================================================
test('ip4a-eq1. (equivalent mutants documented) (a) the PSF boundary test <= 0.25 vs < 0.25 never matters: for every motion L 1..50 x the 11 angles no sub-sample lands at distance^2 === 0.25 exactly; (b) the integer ellipse test <= vs < never matters: dx^2*hh^2 + dy^2*ww^2 === ww^2*hh^2 has no solution with the parities the pixel grid forces (ww, hh 1..100)', () => {
  const SS = 16;
  for (let L = 1; L <= 50; L++) for (const th0 of [0, 1, 15, 30, 45, 60, 89, 90, 91, 135, 179]) {
    const th = (th0 * Math.PI) / 180, c = Math.cos(th), s = -Math.sin(th), hl = (L - 1) / 2, h = Math.ceil(hl * Math.max(Math.abs(c), Math.abs(s)) + 0.5), n = 2 * h + 1;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let sy = 0; sy < SS; sy++) for (let sx = 0; sx < SS; sx++) {
      const px = (x - h) + (2 * sx + 1 - SS) / (2 * SS), py = (y - h) + (2 * sy + 1 - SS) / (2 * SS);
      const t = Math.max(-hl, Math.min(hl, px * c + py * s)), dx = px - t * c, dy = py - t * s;
      if (dx * dx + dy * dy === 0.25) assert.fail(`exact boundary sample at L${L} theta${th0} (${x},${y}) (${sx},${sy})`);
    }
  }
  for (let ww = 1; ww <= 100; ww++) for (let hh = 1; hh <= 100; hh++) {
    // pixel (i, j) in a box starting at x0: dx = 2i + 1 - (2*x0 + ww) has the parity of (1 + ww); dy that of (1 + hh)
    for (let dx = -ww; dx <= ww; dx++) { if ((((dx % 2) + 2) % 2) !== ((1 + ww) % 2)) continue; for (let dy = -hh; dy <= hh; dy++) { if ((((dy % 2) + 2) % 2) !== ((1 + hh) % 2)) continue; assert.notEqual(dx * dx * hh * hh + dy * dy * ww * ww, ww * ww * hh * hh, `boundary pixel for ww ${ww} hh ${hh}`); } }
  }
  // (c) jet 382.5 vs 383 and disk ceil(R - 0.5) vs ceil(R) are equal for every integer input
  for (let i = 0; i < 256; i++) assert.equal(clamp255(382.5 - Math.abs(4 * i - 765)), clamp255(383 - Math.abs(4 * i - 765)));
  for (let R = 1; R <= 50; R++) assert.equal(Math.ceil(R - 0.5), Math.ceil(R));
});


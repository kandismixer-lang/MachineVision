// IP batch 4B (2026-10-15) — 8 op ใหม่ (deconv_wiener, deconv_lucy, ecc_register, gabor_filter, glcm, nlm_denoise, dehaze, dct_jpeg)
// + 5 operation ใหม่ใน Morphology (hit_or_miss, regional_max, regional_min, h_maxima, extended_max) + 3 param ใหม่ (hmPattern, hmRotate, hHeight)
// สเปก: Vision_boi/specs/ip_batch4B_2026-10-15.md · WORKLOG: WORKLOG-2026-10-15-ip-batch4.md
// โครงเดียวกับ batch 4A: ip4b-0 (fingerprint, เทสต์แรกของไฟล์) + ip4b-0w (wrapper balance เส้นทางชุดเดียวกัน) → L1..L4 · golden hash (ip4b-G) พิสูจน์ว่า Morphology 14 op เดิม byte-identical
// parity กับ Python จริง (cv2 5.0 / numpy 2.x) — ข้ามเฉพาะเมื่อไม่มี python/cv2 · IP4B_NO_PY=1 จำลองเครื่องที่ไม่มี cv2 (ใช้ตอน mutation แบบ JS-only)
// ทุกบั๊กที่เทสต์ Python จับได้ ต้องมีเทสต์ JS ล้วนที่จับได้เช่นกัน (บทเรียนจากรีวิว 4A) — เทสต์ JS ล้วนใช้ตัวอ้างอิงที่เขียนแยกในไฟล์นี้ (brute force / นิยามตรง ๆ)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import {
  OPS, FILTERS, defaultParams, needsOf, psfKernel, guidedFilterF32, fixed2, fixed3, seededGaussian,
  NLM_MAX_SIDE, LUCY_MAX_ITERS, ECC_DEFAULT_SIDE, psfParamDefs, psfInfo, HITMISS_PATTERNS, rotate90cw, hitMissKernels, reconstructDilate, regionalMaxMask, hMaximaImage,
  wienerDeconvolve, lucyDeconvolve, gaborKernel, gaborSigma, gaborHalf, glcmCounts, glcmProps, nlmCore, dct8Table, dct8Forward, dct8Inverse, jpegQTable,
} from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS, OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { GROUP_ICON, OP_PRIMARY } from '../src/renderer/filters/ui-meta.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import {
  assertLeakFree, assertWrapperBalance, assertNoExtraHeapGrowth, fingerprint, makeThrowingCv,
} from './_leak-helpers.mjs';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.LUT === 'function' && typeof c.dft === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

const readSrc = (rel) => fs.readFileSync(new URL(`../src/renderer/${rel}`, import.meta.url), 'utf8');
const NEW_OPS = ['deconv_wiener', 'deconv_lucy', 'ecc_register', 'gabor_filter', 'glcm', 'nlm_denoise', 'dehaze', 'dct_jpeg'];
const HOST = { deconv_wiener: 'filter_denoise', deconv_lucy: 'filter_denoise', nlm_denoise: 'filter_denoise', ecc_register: 'image_arithmetic', gabor_filter: 'analysis_detect', glcm: 'analysis_detect', dehaze: 'enhance_tools', dct_jpeg: 'frequency' };
const NEEDS = { deconv_wiener: 'any', deconv_lucy: 'any', ecc_register: 'any', gabor_filter: 'gray', glcm: 'gray', nlm_denoise: 'any', dehaze: 'any', dct_jpeg: 'gray' };
const NEW_ICONS = { deconv_wiener: '⟲', deconv_lucy: '⌾', ecc_register: '⧉', gabor_filter: '⋈', glcm: '⧈', nlm_denoise: '✣', dehaze: '🌫', dct_jpeg: '⬗' };
const MORPH_NEW = ['hit_or_miss', 'regional_max', 'regional_min', 'h_maxima', 'extended_max'];
const MORPH_OLD = ['erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'blackhat', 'fill_holes', 'clear_border', 'remove_small', 'skeleton', 'reconstruct_open', 'convex_hull', 'endpoints_branchpoints'];
const ECC_KEYS = ['ecc_ok', 'ecc_rho', 'ecc_dx', 'ecc_dy', 'ecc_angle_deg', 'ecc_scale'];
const GLCM_KEYS = ['glcm_contrast', 'glcm_correlation', 'glcm_energy', 'glcm_homogeneity', 'glcm_pairs'];
const DEHAZE_KEYS = ['dh_a_r', 'dh_a_g', 'dh_a_b', 'dh_t_mean'];
const DCT_KEYS = ['dct_psnr', 'dct_zero_pct', 'dct_blocks'];
const GABOR_KEYS = ['gabor_mean', 'gabor_max'];

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
function noiseGray(w, h, seed) { const r = lcg(seed); return gray(w, h, () => Math.floor(r() * 256)); }
function noiseRGB(w, h, seed) { const r = lcg(seed); return rgb(w, h, () => [Math.floor(r() * 256), Math.floor(r() * 256), Math.floor(r() * 256)]); }
function smoothRGB(w, h) { return rgb(w, h, (x, y) => [(x * 255 / Math.max(1, w - 1)) | 0, (y * 255 / Math.max(1, h - 1)) | 0, ((x + y) * 5) & 255]); }
// ภาพเรียบมีลวดลาย (sin/cos) — ECC ลู่เข้าได้ · ค่าคงที่ของฟังก์ชันตายตัว
const wave = (x, y) => Math.floor(128 + 60 * Math.sin(x / 9) + 50 * Math.cos(y / 7) + 20 * Math.sin((x + y) / 5));
function waveGray(w, h) { return gray(w, h, wave); }
function bytes(m) { return Array.from(m.data); }
function black(w, h) { return new cv.Mat(h, w, cv.CV_8UC1, new cv.Scalar(0)); }
function fillRectM(m, x, y, w, h, v = 255) { cv.rectangle(m, new cv.Point(x, y), new cv.Point(x + w - 1, y + h - 1), new cv.Scalar(v), -1); }
function warpedBy(m, M2x3, flags = cv.INTER_LINEAR) { // forward warp (dst(M·p) = src(p)) ด้วยขอบ replicate
  const M = cv.matFromArray(2, 3, cv.CV_64FC1, M2x3), out = new cv.Mat();
  cv.warpAffine(m, out, M, new cv.Size(m.cols, m.rows), flags, cv.BORDER_REPLICATE, new cv.Scalar());
  M.delete(); return out;
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
const mae = (a, b) => { const da = a.data, db = b.data; let s = 0; for (let i = 0; i < da.length; i++) s += Math.abs(da[i] - db[i]); return s / da.length; };
const mse = (a, b) => { const da = a.data, db = b.data; let s = 0; for (let i = 0; i < da.length; i++) s += (da[i] - db[i]) ** 2; return s / da.length; };
const STEP = (id, mode, extra = {}) => ({ id, params: { ...defaultParams(id), mode, ...extra } });
const OPSTEP = (opId, extra = {}) => {
  const prefixed = {};
  for (const [k, v] of Object.entries(extra)) prefixed[`${opId}__${k}`] = v;
  const step = opId === 'morphology' ? { id: 'morphology', params: { ...defaultParams('morphology'), ...extra } } : STEP(OP_TO_FILTER[opId], opId, prefixed);
  if (FILTERS[step.id].inputs === 2) step.src2 = { pipeline: [] };
  return step;
};
const MORPH = (op, extra = {}) => ({ ...defaultParams('morphology'), op, ...extra });
const defaultsOf = (id) => Object.fromEntries(OPS[id].params.map((p) => [p.key, p.default]));
// ภาพเบลอแบบที่ Custom Kernel ทำ (app เองเรียก) — ใช้ตรวจ Wiener / Lucy แบบวงจรปิด
function resizeTo(m, w, h) { const o = new cv.Mat(); cv.resize(m, o, new cv.Size(w, h), 0, 0, cv.INTER_AREA); return o; }
function blurWith(src, preset, psfSize = 9, psfAngle = 0) {
  const out = OPS.custom_kernel.apply(cv, src, { preset, amount: 1, psfSize, psfAngle }, {});
  return out;
}
function checkerboard(w, h, tile = 10) { return gray(w, h, (x, y) => (((x / tile) | 0) + ((y / tile) | 0)) & 1 ? 230 : 30); }

// ===================================================================
// ip4b-0 — fingerprint (เทสต์แรกของไฟล์ = process สะอาด) · ip4b-0w — wrapper balance เส้นทางชุดเดียวกัน
// ===================================================================
// ECC บนภาพลวดลายเรียบที่เลื่อนจริง (ลู่เข้า → ไม่ throw C++) · ภาพที่ "ไม่ลู่เข้า" ถูกทดสอบแยก (ip4b-L4) เพราะ C++ exception รั่ว ~760 B/ครั้ง (WORK-TRACKER §1) แก้ในแอปไม่ได้
function batch4BLeakPaths(only) {
  const mats = [];
  const keep = (m) => { mats.push(m); return m; };
  const want = (id) => !only || only === id;
  const empty = keep(new cv.Mat());
  const one = keep(gray(1, 1, () => 128)), oneC = keep(rgb(1, 1, () => [200, 100, 50]));
  const g = keep(gray(40, 30, (x, y) => (x * 7 + y * 3) & 255));
  const c = keep(rgb(40, 30, (x, y) => [x * 6 & 255, y * 8 & 255, 90 + ((x * y) & 63)]));
  const a = keep(toRGBA(c));
  const flat = keep(gray(40, 30, () => 90)), flatC = keep(rgb(40, 30, () => [90, 90, 90]));
  const wide = keep(gray(700, 12, (x, y) => (x * 3 + y * 5) & 255));
  const wA = keep(waveGray(40, 30)), wAc = keep(toRGBA(keep(rgb(40, 30, (x, y) => [wave(x, y), wave(x, y), wave(x, y)]))));
  const wB = keep(warpedBy(wA, [1, 0, 2.5, 0, 1, -1.5]));
  const wBc = keep(rgb(40, 30, (x, y) => { const v = wave(x + 2.5, y - 1.5); return [v, v, v]; }));
  const wBsmall = keep(resizeTo(wA, 20, 15)), tiny = keep(gray(10, 10, (x, y) => (x * 25 + y * 11) & 255));
  const bigWave = keep(waveGray(80, 60));
  const paths = [];
  const add = (id, params, src, tag, B) => paths.push({
    label: `${id} ${tag}`,
    run: (cvUse) => { const o = OPS[id].apply(cvUse, src, params, B === undefined ? {} : { src2: B }); o.delete(); },
  });
  const morph = (op, params, src, tag) => paths.push({
    label: `morphology ${op} ${tag}`,
    run: (cvUse) => { const o = OPS.morphology.apply(cvUse, src, { ...defaultsOf('morphology'), op, ...params }, {}); o.delete(); },
  });
  const srcs = [['gray', g], ['rgb', c], ['rgba', a], ['1x1', one], ['0x0', empty], ['flat', flat]];
  if (want('deconv_wiener')) {
    for (const psf of ['motion', 'disk']) for (const psfSize of [1, 9, 20]) for (const [t, s] of srcs) add('deconv_wiener', { psf, psfSize, psfAngle: 30 }, s, `${psf} ${psfSize} ${t}`);
    for (const nsr of [0, 0.5]) for (const [t, s] of [['gray', g], ['rgb', c]]) add('deconv_wiener', { nsr }, s, `nsr ${nsr} ${t}`);
    add('deconv_wiener', { psf: 'constructor', psfSize: 'x', nsr: 'q' }, g, 'weird');
  }
  if (want('deconv_lucy')) {
    for (const psf of ['motion', 'disk']) for (const iterations of [1, 3]) for (const [t, s] of srcs) add('deconv_lucy', { psf, iterations, psfSize: 7 }, s, `${psf} ${iterations} ${t}`);
    add('deconv_lucy', { psf: 'constructor', iterations: 'x' }, g, 'weird');
  }
  if (want('ecc_register')) {
    for (const motion of ['translation', 'euclidean', 'affine', 'homography']) for (const view of ['aligned', 'overlay', 'diff']) add('ecc_register', { motion, view, iterations: 50 }, wA, `${motion} ${view} gray`, wB);
    add('ecc_register', { view: 'overlay' }, wAc, 'rgba/rgb', wBc); add('ecc_register', { view: 'diff' }, wAc, 'rgba/rgb diff', wBc); add('ecc_register', {}, wAc, 'rgba/rgb aligned', wBc);
    add('ecc_register', { view: 'overlay' }, wA, 'gray/rgb resized B', keep(toRGBA(wBsmall)));
    add('ecc_register', { maxSide: 64, view: 'overlay' }, bigWave, 'shrunk to 64', keep(warpedBy(bigWave, [1, 0, 3, 0, 1, 2])));
    add('ecc_register', { maxSide: 64, view: 'diff' }, bigWave, 'shrunk diff', keep(warpedBy(bigWave, [1, 0, 3, 0, 1, 2])));
    add('ecc_register', {}, flat, 'flat A', wB); add('ecc_register', {}, wA, 'flat B', flat); add('ecc_register', {}, tiny, 'tiny', tiny); add('ecc_register', {}, one, '1x1', one);
    add('ecc_register', {}, wA, 'no B'); add('ecc_register', {}, wA, 'B 0x0', empty); add('ecc_register', {}, empty, 'A 0x0', wB);
    add('ecc_register', { motion: 'constructor', view: 'constructor', iterations: 'x', maxSide: 'x' }, wA, 'weird', wB);
  }
  if (want('gabor_filter')) {
    for (const output of ['magnitude', 'real']) for (const [t, s] of srcs) add('gabor_filter', { output, wavelength: 6, orientation: 30 }, s, `${output} ${t}`);
    add('gabor_filter', { wavelength: 64, aspect: 0.2, bandwidth: 0.5 }, g, 'k=99'); add('gabor_filter', { wavelength: 2, bandwidth: 3 }, g, 'small');
  }
  if (want('glcm')) {
    for (const view of ['matrix', 'quantized']) for (const [t, s] of srcs) add('glcm', { view }, s, `${view} ${t}`);
    for (const levels of ['2', '64']) for (const angle of ['0', '45', '90', '135', 'avg']) for (const symmetric of [false, true]) add('glcm', { levels, angle, symmetric, distance: 3 }, g, `L${levels} ${angle} sym${symmetric}`);
    add('glcm', { distance: 10, angle: 'avg' }, one, '1x1 d10'); add('glcm', { distance: 10 }, gray(5, 5, () => 7), 'd > size');
  }
  if (want('nlm_denoise')) {
    for (const [t, s] of srcs) add('nlm_denoise', { search: '7' }, s, `s7 ${t}`);
    for (const patch of ['3', '7']) add('nlm_denoise', { patch, search: '7', h: 30 }, c, `patch ${patch} rgb`);
    add('nlm_denoise', { search: '7' }, wide, 'wide 700 (shrunk to 640)'); add('nlm_denoise', { patch: 'constructor', search: 'constructor', h: 'x' }, g, 'weird');
  }
  if (want('dehaze')) {
    for (const view of ['result', 'transmission', 'dark']) for (const refine of [true, false]) for (const [t, s] of srcs) add('dehaze', { view, refine, patch: 7 }, s, `${view} r${refine} ${t}`);
    add('dehaze', { patch: 31, amount: 1 }, c, 'patch 31'); add('dehaze', { patch: 3, amount: 0.1 }, c, 'patch 3'); add('dehaze', {}, keep(rgb(40, 30, () => [0, 0, 0])), 'black');
  }
  if (want('dct_jpeg')) {
    for (const method of ['jpeg', 'zonal']) for (const view of ['result', 'coeffs', 'error']) for (const [t, s] of srcs) add('dct_jpeg', { method, view }, s, `${method} ${view} ${t}`);
    add('dct_jpeg', { quality: 1 }, gray(13, 7, (x, y) => (x * 20 + y * 9) & 255), '13x7 q1'); add('dct_jpeg', { quality: 100, keep: 8, method: 'zonal' }, g, 'k8');
  }
  if (want('morphology')) {
    for (const hmPattern of Object.keys(HITMISS_PATTERNS)) for (const hmRotate of [true, false]) for (const [t, s] of [['gray', g], ['flat', flat], ['1x1', one], ['0x0', empty], ['rgb', c], ['rgba', a]]) morph('hit_or_miss', { hmPattern, hmRotate }, s, `${hmPattern} rot${hmRotate} ${t}`);
    morph('hit_or_miss', { hmPattern: 'constructor', hmRotate: 'x' }, g, 'weird');
    for (const op of ['regional_max', 'regional_min', 'h_maxima', 'extended_max']) {
      for (const [t, s] of [['gray', g], ['flat', flat], ['1x1', one], ['0x0', empty], ['rgb', c]]) morph(op, { hHeight: 20 }, s, t);
      morph(op, { hHeight: 'junk' }, g, 'junk h'); morph(op, { hHeight: 255 }, g, 'h255');
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

test('ip4b-0. leak (address fingerprint cycle, first test of a clean process): every path of the 8 new ops and the 5 new Morphology operations (all selects, gray / RGB / RGBA / 1x1 / 0x0 / flat, no B / B 0x0 / B resized, ECC guards) keeps revisiting the same few malloc states', () => {
  const { paths, cleanup } = batch4BLeakPaths();
  try {
    assert.ok(paths.length >= 250, `paths ${paths.length}`);
    fingerprintCycleProbe('IP batch 4B (all paths)', () => { for (const q of paths) q.run(cv); });
  } finally { cleanup(); }
});

test('ip4b-0w. leak (wrapper balance twin of ip4b-0): the exact same path table creates and deletes every embind wrapper it makes (OTF / DFT temporaries, per-channel Mats, Lucy buffers, ECC warp / mask / resized pair, Gabor kernels, dark-channel Mats, DCT pad, hit-or-miss kernels, regional masks)', () => {
  const { paths, cleanup } = batch4BLeakPaths();
  try {
    assertWrapperBalance(cv, 'IP batch 4B ops (all paths)', (c) => { for (const p of paths) p.run(c); }, { minCreated: 600 });
  } finally { cleanup(); }
});

// ===================================================================
// ip4b-L1…L4 — leak ต่อ op / ต่อเส้นทาง
// ===================================================================
test('ip4b-L1. leak: every path of every new op / operation balances every wrapper and grows the heap by 0', () => {
  for (const id of [...NEW_OPS, 'morphology']) {
    const { paths, cleanup } = batch4BLeakPaths(id);
    try {
      assert.ok(paths.length > 0, id);
      for (const q of paths) assertLeakFree(cv, q.label, (c) => q.run(c), { minCreated: 0, iters: 4, warmup: 2 });
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
  const wA = waveGray(40, 30), wB = warpedBy(wA, [1, 0, 2.5, 0, 1, -1.5]), wBsmall = waveGray(20, 15), wAc = rgb(40, 30, (x, y) => [wave(x, y), wave(x, y), wave(x, y)]);
  const bigWave = waveGray(80, 60), bigB = warpedBy(bigWave, [1, 0, 3, 0, 1, 2]), wide = gray(700, 12, (x, y) => (x * 3 + y * 5) & 255);
  const mk = (id, p, src, B) => (cvU) => OPS[id].apply(cvU, src, p, B === undefined ? {} : { src2: B });
  const mo = (op, p, src) => (cvU) => OPS.morphology.apply(cvU, src, { ...defaultsOf('morphology'), op, ...p }, {});
  const cases = [
    // [label, cv function | [name, onCall] | { mat: name, onCall }, runner]
    ['wiener getOptimalDFTSize', 'getOptimalDFTSize', mk('deconv_wiener', {}, g)],
    ['wiener matFromArray #1 (OTF mat)', ['matFromArray', 1], mk('deconv_wiener', {}, g)],
    ['wiener dft #1 (OTF)', ['dft', 1], mk('deconv_wiener', {}, g)],
    ['wiener dft #2 (channel spectrum)', ['dft', 2], mk('deconv_wiener', {}, g)],
    ['wiener dft #3 (inverse)', ['dft', 3], mk('deconv_wiener', {}, g)],
    ['wiener dft #5 (2nd channel spectrum, rgb)', ['dft', 5], mk('deconv_wiener', {}, c)],
    ['wiener dft #6 (2nd channel inverse, rgb)', ['dft', 6], mk('deconv_wiener', {}, c)],
    ['wiener copyMakeBorder #1', ['copyMakeBorder', 1], mk('deconv_wiener', {}, g)],
    ['wiener copyMakeBorder #3 (3rd channel)', ['copyMakeBorder', 3], mk('deconv_wiener', {}, c)],
    ['wiener matFromArray #2 (result)', ['matFromArray', 2], mk('deconv_wiener', {}, g)],
    ['wiener split', 'split', mk('deconv_wiener', {}, c)],
    ['wiener cvtColor (rgba dropAlpha)', 'cvtColor', mk('deconv_wiener', {}, a)],
    ['wiener convertTo (Mat method)', { mat: 'convertTo' }, mk('deconv_wiener', {}, g)],
    ['lucy matFromArray #1 (K)', ['matFromArray', 1], mk('deconv_lucy', {}, g)],
    ['lucy matFromArray #2 (result)', ['matFromArray', 2], mk('deconv_lucy', {}, g)],
    ['lucy flip', 'flip', mk('deconv_lucy', {}, g)],
    ['lucy filter2D #1', ['filter2D', 1], mk('deconv_lucy', {}, g)],
    ['lucy filter2D #2 (adjoint, 1st round)', ['filter2D', 2], mk('deconv_lucy', {}, g)],
    ['lucy filter2D #9 (5th round, blur)', ['filter2D', 9], mk('deconv_lucy', {}, g)],
    ['lucy filter2D #10 (5th round, adjoint)', ['filter2D', 10], mk('deconv_lucy', {}, g)],
    ['lucy filter2D #3 (2nd channel, rgb 1 round)', ['filter2D', 3], mk('deconv_lucy', { iterations: 1 }, c)],
    ['lucy split', 'split', mk('deconv_lucy', {}, c)],
    ['lucy cvtColor (rgba dropAlpha)', 'cvtColor', mk('deconv_lucy', {}, a)],
    ['lucy convertTo (Mat method)', { mat: 'convertTo' }, mk('deconv_lucy', {}, g)],
    ['lucy clone (Mat method)', { mat: 'clone' }, mk('deconv_lucy', {}, g)],
    ['ecc matchSize resize', 'resize', mk('ecc_register', {}, wA, wBsmall)],
    ['ecc resize #1 (A shrunk)', ['resize', 1], mk('ecc_register', { maxSide: 64 }, bigWave, bigB)],
    ['ecc resize #2 (B shrunk)', ['resize', 2], mk('ecc_register', { maxSide: 64 }, bigWave, bigB)],
    ['ecc cvtColor (A rgba to gray)', 'cvtColor', mk('ecc_register', {}, a, wB)],
    ['ecc cvtColor (B gray conversion)', 'cvtColor', mk('ecc_register', {}, wA, c)],
    ['ecc meanStdDev #1 (A)', ['meanStdDev', 1], mk('ecc_register', {}, wA, wB)],
    ['ecc meanStdDev #2 (B, A stats exist)', ['meanStdDev', 2], mk('ecc_register', {}, wA, wB)],
    ['ecc matFromArray #1 (warp)', ['matFromArray', 1], mk('ecc_register', {}, wA, wB)],
    ['ecc warpAffine (aligned)', 'warpAffine', mk('ecc_register', {}, wA, wB)],
    ['ecc warpPerspective (homography)', 'warpPerspective', mk('ecc_register', { motion: 'homography' }, wA, wB)],
    ['ecc absdiff (diff view)', 'absdiff', mk('ecc_register', { view: 'diff' }, wA, wB)],
    ['ecc matFromArray #2 (overlay result)', ['matFromArray', 2], mk('ecc_register', { view: 'overlay' }, wA, wB)],
    ['ecc cvtColor (overlay of rgb A)', ['cvtColor', 2], mk('ecc_register', { view: 'overlay' }, wAc, wB)],
    ['gabor cvtColor (rgb)', 'cvtColor', mk('gabor_filter', {}, c)],
    ['gabor convertTo (Mat method)', { mat: 'convertTo' }, mk('gabor_filter', {}, g)],
    ['gabor matFromArray #1 (even kernel)', ['matFromArray', 1], mk('gabor_filter', {}, g)],
    ['gabor matFromArray #2 (odd kernel)', ['matFromArray', 2], mk('gabor_filter', {}, g)],
    ['gabor matFromArray #3 (result)', ['matFromArray', 3], mk('gabor_filter', {}, g)],
    ['gabor filter2D #1', ['filter2D', 1], mk('gabor_filter', {}, g)],
    ['gabor filter2D #2', ['filter2D', 2], mk('gabor_filter', {}, g)],
    ['gabor magnitude', 'magnitude', mk('gabor_filter', {}, g)],
    ['glcm cvtColor (rgb)', 'cvtColor', mk('glcm', {}, c)],
    ['glcm matFromArray (matrix view)', 'matFromArray', mk('glcm', {}, g)],
    ['glcm matFromArray (quantized view)', 'matFromArray', mk('glcm', { view: 'quantized' }, g)],
    ['nlm resize #1 (shrink)', ['resize', 1], mk('nlm_denoise', { search: '7' }, wide)],
    ['nlm resize #2 (enlarge back)', ['resize', 2], mk('nlm_denoise', { search: '7' }, wide)],
    ['nlm matFromArray', 'matFromArray', mk('nlm_denoise', { search: '7' }, g)],
    ['nlm cvtColor (rgba dropAlpha)', 'cvtColor', mk('nlm_denoise', { search: '7' }, a)],
    ['dehaze getStructuringElement', 'getStructuringElement', mk('dehaze', {}, c)],
    ['dehaze matFromArray #1 (dark source)', ['matFromArray', 1], mk('dehaze', {}, c)],
    ['dehaze erode #1 (dark channel)', ['erode', 1], mk('dehaze', {}, c)],
    ['dehaze matFromArray #2 (coarse map)', ['matFromArray', 2], mk('dehaze', {}, c)],
    ['dehaze erode #2 (coarse map)', ['erode', 2], mk('dehaze', {}, c)],
    ['dehaze cvtColor (guide of rgb)', 'cvtColor', mk('dehaze', {}, c)],
    ['dehaze cvtColor (rgba dropAlpha)', 'cvtColor', mk('dehaze', {}, a)],
    ['dehaze convertTo (guide, Mat method)', { mat: 'convertTo' }, mk('dehaze', {}, g)],
    ['dehaze matFromArray #3 (P)', ['matFromArray', 3], mk('dehaze', {}, c)],
    ['dehaze boxFilter #1 (guided)', ['boxFilter', 1], mk('dehaze', {}, c)],
    ['dehaze boxFilter #6 (guided, box(B))', ['boxFilter', 6], mk('dehaze', {}, c)],
    ['dehaze matFromArray (transmission view)', ['matFromArray', 3], mk('dehaze', { view: 'transmission', refine: false }, c)],
    ['dct copyMakeBorder', 'copyMakeBorder', mk('dct_jpeg', {}, g)],
    ['dct matFromArray', 'matFromArray', mk('dct_jpeg', {}, g)],
    ['dct cvtColor (rgb)', 'cvtColor', mk('dct_jpeg', {}, c)],
    ['hit_or_miss threshold', 'threshold', mo('hit_or_miss', {}, g)],
    ['hit_or_miss matFromArray #1', ['matFromArray', 1], mo('hit_or_miss', { hmPattern: 'corner' }, g)],
    ['hit_or_miss morphologyEx #1', ['morphologyEx', 1], mo('hit_or_miss', { hmPattern: 'corner' }, g)],
    ['hit_or_miss morphologyEx #3 (3rd kernel)', ['morphologyEx', 3], mo('hit_or_miss', { hmPattern: 'corner' }, g)],
    ['hit_or_miss bitwise_or #1', ['bitwise_or', 1], mo('hit_or_miss', { hmPattern: 'corner' }, g)],
    ['hit_or_miss bitwise_or #3', ['bitwise_or', 3], mo('hit_or_miss', { hmPattern: 'corner' }, g)],
    ['hit_or_miss countNonZero', 'countNonZero', mo('hit_or_miss', { hmPattern: 'corner' }, g)],
    ['regional_max matFromArray #1 (result)', ['matFromArray', 1], mo('regional_max', {}, g)],
    ['regional_max matFromArray #2 (component mask)', ['matFromArray', 2], mo('regional_max', {}, g)],
    ['regional_max connectedComponents', 'connectedComponents', mo('regional_max', {}, g)],
    ['regional_min connectedComponents', 'connectedComponents', mo('regional_min', {}, g)],
    ['h_maxima matFromArray #2 (component mask)', ['matFromArray', 2], mo('h_maxima', {}, g)],
    ['h_maxima connectedComponents', 'connectedComponents', mo('h_maxima', {}, g)],
    ['extended_max connectedComponents', 'connectedComponents', mo('extended_max', {}, g)],
    ['regional cvtColor (rgb)', 'cvtColor', mo('regional_max', {}, c)],
    ['hit_or_miss cvtColor (rgb)', 'cvtColor', mo('hit_or_miss', {}, c)],
  ];
  return { cases, cleanup: () => { for (const m of [g, c, a, wA, wB, wBsmall, wAc, bigWave, bigB, wide]) m.delete(); } };
}

test('ip4b-L2. leak on the exception path: a throw inside every cv call (and Mat method) the new paths make still gives every wrapper back (injected failures; each case must throw)', () => {
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

test('ip4b-L3. leak on the exception path AFTER the output exists: a ctx whose "info" / "values" setter throws makes every new op / operation delete its own output', () => {
  const g = gray(40, 30, (x, y) => (x * 7 + y * 3) & 255), c = rgb(40, 30, (x, y) => [x * 6 & 255, y * 8 & 255, 90]), a = toRGBA(c);
  const wA = waveGray(40, 30), wB = warpedBy(wA, [1, 0, 2.5, 0, 1, -1.5]);
  const tInfo = (B) => ({ ...(B ? { src2: B } : {}), set info(_) { throw new Error('injected test failure (ctx.info)'); } });
  const tValues = (B) => ({ ...(B ? { src2: B } : {}), info: '', set values(_) { throw new Error('injected test failure (ctx.values)'); } });
  const check = (label, id, p, src, ctxMaker, B) => assertWrapperBalance(cv, label, (cvU) => { const o = OPS[id].apply(cvU, src, p, ctxMaker(B)); o.delete(); }, { expectThrow: /injected test failure/, minCreated: 1 });
  const checkM = (label, op, p, src, ctxMaker) => assertWrapperBalance(cv, label, (cvU) => { const o = OPS.morphology.apply(cvU, src, { ...defaultsOf('morphology'), op, ...p }, ctxMaker()); o.delete(); }, { expectThrow: /injected test failure/, minCreated: 1 });
  try {
    check('info wiener gray', 'deconv_wiener', {}, g, tInfo); check('info wiener rgb', 'deconv_wiener', {}, c, tInfo);
    check('info lucy gray', 'deconv_lucy', {}, g, tInfo); check('info lucy rgb', 'deconv_lucy', {}, a, tInfo);
    for (const view of ['aligned', 'overlay', 'diff']) { check(`info ecc ${view}`, 'ecc_register', { view }, wA, tInfo, wB); check(`values ecc ${view}`, 'ecc_register', { view }, wA, tValues, wB); }
    { const flat0 = black(40, 30); try { check('values ecc flat (guard path)', 'ecc_register', {}, flat0, tValues, wB); } finally { flat0.delete(); } }
    check('info gabor', 'gabor_filter', {}, g, tInfo); check('values gabor', 'gabor_filter', { output: 'real' }, g, tValues);
    for (const view of ['matrix', 'quantized']) { check(`info glcm ${view}`, 'glcm', { view }, g, tInfo); check(`values glcm ${view}`, 'glcm', { view }, g, tValues); }
    check('info nlm', 'nlm_denoise', { search: '7' }, c, tInfo);
    for (const view of ['result', 'transmission', 'dark']) { check(`info dehaze ${view}`, 'dehaze', { view }, c, tInfo); check(`values dehaze ${view}`, 'dehaze', { view, refine: false }, g, tValues); }
    for (const view of ['result', 'coeffs', 'error']) { check(`info dct ${view}`, 'dct_jpeg', { view }, g, tInfo); check(`values dct ${view}`, 'dct_jpeg', { view, method: 'zonal' }, g, tValues); }
    for (const op of ['hit_or_miss', 'regional_max', 'regional_min', 'h_maxima', 'extended_max']) checkM(`info morphology ${op}`, op, {}, g, () => tInfo());
  } finally { for (const m of [g, c, a, wA, wB]) m.delete(); }
});

// ip4b-L4 — ECC: (a) findTransformECC ที่ throw เป็น JS error → path catch คืน A เดิม + ecc_ok 0 ไม่รั่ว (b) C++ exception จริง (สองภาพสุ่มคนละชุดที่ผ่าน guard แต่ไม่ลู่เข้า) ไม่รั่วเกิน baseline ที่เรียก cv.findTransformECC ตรง ๆ
test('ip4b-L4. leak: ECC non-convergence -- (a) an injected JS error in findTransformECC takes the catch path (A clone, ecc_ok 0, every wrapper deleted); (b) a REAL C++ exception (two unrelated noise images that pass the guards) leaks no more than the raw cv.findTransformECC baseline (the unavoidable ~760 B per throw of opencv.js)', () => {
  const wA = waveGray(40, 30), wB = warpedBy(wA, [1, 0, 2.5, 0, 1, -1.5]);
  try {
    // (a) injected failure
    assertWrapperBalance(cv, 'ecc catch path (injected JS error)', (cvU) => {
      const ctx = { src2: wB };
      const o = OPS.ecc_register.apply(makeThrowingCv(cvU, 'findTransformECC'), wA, {}, ctx);
      try {
        sameBytes(o, wA, 'A returned unchanged');
        assert.equal(scalarMap(ctx).ecc_ok, 0);
        assert.match(ctx.info, /ไม่ลู่เข้า/);
        for (const k of ECC_KEYS.slice(1)) assert.equal(scalarMap(ctx)[k], null, k);
      } finally { o.delete(); }
    }, { minCreated: 5 });
    // (b) real C++ exception — find a pair that really does not converge at these settings
    const nA = noiseGray(40, 30, 5), nB = noiseGray(40, 30, 6);
    const rawRun = () => {
      const warp = cv.matFromArray(2, 3, cv.CV_32FC1, [1, 0, 0, 0, 1, 0]), mask = new cv.Mat();
      try { cv.findTransformECC(nB, nA, warp, cv.MOTION_EUCLIDEAN, new cv.TermCriteria(cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 10, 1e-5), mask, 5); } catch (e) { /* expected C++ exception */ } finally { warp.delete(); mask.delete(); }
    };
    let threw = false;
    { const warp = cv.matFromArray(2, 3, cv.CV_32FC1, [1, 0, 0, 0, 1, 0]), mask = new cv.Mat(); try { cv.findTransformECC(nB, nA, warp, cv.MOTION_EUCLIDEAN, new cv.TermCriteria(cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 10, 1e-5), mask, 5); } catch (e) { threw = true; } finally { warp.delete(); mask.delete(); } }
    assert.ok(threw, 'unrelated noise images must make findTransformECC throw (otherwise this test proves nothing)');
    const ctx0 = { src2: nB };
    const o0 = OPS.ecc_register.apply(cv, nA, { iterations: 10 }, ctx0);
    sameBytes(o0, nA, 'non-convergence returns A'); o0.delete();
    assert.equal(scalarMap(ctx0).ecc_ok, 0);
    assertWrapperBalance(cv, 'ecc real C++ non-convergence', (cvU) => { const o = OPS.ecc_register.apply(cvU, nA, { iterations: 10 }, { src2: nB }); o.delete(); }, { minCreated: 3 });
    assertNoExtraHeapGrowth(cv, 'ecc real C++ non-convergence vs raw baseline', () => { const o = OPS.ecc_register.apply(cv, nA, { iterations: 10 }, { src2: nB }); o.delete(); }, rawRun, { iters: 40, warmup: 5 });
    nA.delete(); nB.delete();
  } finally { wA.delete(); wB.delete(); }
});

// ===================================================================
// ip4b-G — golden: Morphology 14 op เดิม (+3 กรณีที่ตั้ง shape/ksize/iterations/minArea เอง) ให้ผล byte-identical (ภาพ + info + Python ที่ export + ข้อความการ์ด)
// hash วัดบนโค้ดเดิม (HEAD ffce85b) ก่อนแก้โค้ดบรรทัดแรก · 4 ภาพ: สุ่มขาว-ดำ 97×61, blob 320×240 (มีรู + วัตถุชิดขอบ), เส้นบาง 120×90, 1×1
// 3 กรณี: (A) params เดิม (B) ส่ง param ใหม่ (hmPattern / hmRotate / hHeight) ที่ค่าเริ่มต้น (C) param ใหม่เป็นค่าขยะ / "constructor" — ทั้งสามต้องได้ hash เดิม
// ===================================================================
const GOLDEN = {
  "{\"op\":\"erode\"}|noise97": "4e06a77a533f15ee",
  "{\"op\":\"erode\"}|blob320": "8d28cc748593428e",
  "{\"op\":\"erode\"}|lines120": "e25ceab59f62dab4",
  "{\"op\":\"erode\"}|one1": "439d5b97ae88bc7c",
  "{\"op\":\"erode\"}|PY": "0478779123a38da0",
  "{\"op\":\"erode\"}|UI": "f390dfddd597c463",
  "{\"op\":\"dilate\"}|noise97": "b95ef0d553feeaba",
  "{\"op\":\"dilate\"}|blob320": "029e81207b85f9ef",
  "{\"op\":\"dilate\"}|lines120": "7f2688f21773b186",
  "{\"op\":\"dilate\"}|one1": "439d5b97ae88bc7c",
  "{\"op\":\"dilate\"}|PY": "26428fd7df588637",
  "{\"op\":\"dilate\"}|UI": "78f7e9204792b986",
  "{\"op\":\"open\"}|noise97": "4e06a77a533f15ee",
  "{\"op\":\"open\"}|blob320": "c469021c59bee91c",
  "{\"op\":\"open\"}|lines120": "e25ceab59f62dab4",
  "{\"op\":\"open\"}|one1": "439d5b97ae88bc7c",
  "{\"op\":\"open\"}|PY": "7059989ad1f403ad",
  "{\"op\":\"open\"}|UI": "d761fd923694231c",
  "{\"op\":\"close\"}|noise97": "b95ef0d553feeaba",
  "{\"op\":\"close\"}|blob320": "912141420b1f2fe6",
  "{\"op\":\"close\"}|lines120": "34ad1b8d31f23798",
  "{\"op\":\"close\"}|one1": "439d5b97ae88bc7c",
  "{\"op\":\"close\"}|PY": "8356198c3fa826a6",
  "{\"op\":\"close\"}|UI": "92274d96e682b13e",
  "{\"op\":\"gradient\"}|noise97": "b95ef0d553feeaba",
  "{\"op\":\"gradient\"}|blob320": "3345fb76bf561fa7",
  "{\"op\":\"gradient\"}|lines120": "7f2688f21773b186",
  "{\"op\":\"gradient\"}|one1": "0faf96a5f63556d4",
  "{\"op\":\"gradient\"}|PY": "4173100672c9164e",
  "{\"op\":\"gradient\"}|UI": "103d22bb46f47b4a",
  "{\"op\":\"tophat\"}|noise97": "432e471bac4d2d8b",
  "{\"op\":\"tophat\"}|blob320": "e8e7dcb4e79f1c4c",
  "{\"op\":\"tophat\"}|lines120": "d6368e140e5c4b58",
  "{\"op\":\"tophat\"}|one1": "0faf96a5f63556d4",
  "{\"op\":\"tophat\"}|PY": "86b2759914ab2363",
  "{\"op\":\"tophat\"}|UI": "8002239df6bac6de",
  "{\"op\":\"blackhat\"}|noise97": "0aac33488701fd73",
  "{\"op\":\"blackhat\"}|blob320": "091369a518917bed",
  "{\"op\":\"blackhat\"}|lines120": "c38254b0f6ac8766",
  "{\"op\":\"blackhat\"}|one1": "0faf96a5f63556d4",
  "{\"op\":\"blackhat\"}|PY": "a181741738ab9de6",
  "{\"op\":\"blackhat\"}|UI": "994f81144749945d",
  "{\"op\":\"fill_holes\"}|noise97": "61e6de0050bd4113",
  "{\"op\":\"fill_holes\"}|blob320": "eb8797839e5da59c",
  "{\"op\":\"fill_holes\"}|lines120": "75489686cc507b78",
  "{\"op\":\"fill_holes\"}|one1": "b1624c8023072b68",
  "{\"op\":\"fill_holes\"}|PY": "b7dd20fac8fed11f",
  "{\"op\":\"fill_holes\"}|UI": "8fe458b61572ca37",
  "{\"op\":\"clear_border\"}|noise97": "9d8554585d0f2ede",
  "{\"op\":\"clear_border\"}|blob320": "c95c0eb6d0ee1225",
  "{\"op\":\"clear_border\"}|lines120": "143d6741a2c64a76",
  "{\"op\":\"clear_border\"}|one1": "778f51f8e016b68f",
  "{\"op\":\"clear_border\"}|PY": "ed4debec9a1fdef9",
  "{\"op\":\"clear_border\"}|UI": "f3270eac5d6ed425",
  "{\"op\":\"remove_small\"}|noise97": "d836a0c68ba1986b",
  "{\"op\":\"remove_small\"}|blob320": "d53db8774a362494",
  "{\"op\":\"remove_small\"}|lines120": "992b01a7b21abc14",
  "{\"op\":\"remove_small\"}|one1": "08c663b2bc674cff",
  "{\"op\":\"remove_small\"}|PY": "87ce0c7181698e2e",
  "{\"op\":\"remove_small\"}|UI": "e5f156d368b3ecd1",
  "{\"op\":\"skeleton\"}|noise97": "e3f742518763ffb1",  // 2026-10-21: skeleton เต็มความละเอียด + ขอบดำ 1 px (ตั้งใจเปลี่ยน)
  "{\"op\":\"skeleton\"}|blob320": "d9b7f99c680d4d9d",
  "{\"op\":\"skeleton\"}|lines120": "7e295a2255bc7c23",
  "{\"op\":\"skeleton\"}|one1": "cb36710e9190ee31",
  "{\"op\":\"skeleton\"}|PY": "37629d01e8550cfe",
  "{\"op\":\"skeleton\"}|UI": "0b2385e2c351af74",
  "{\"op\":\"reconstruct_open\"}|noise97": "2db702da647f85af",
  "{\"op\":\"reconstruct_open\"}|blob320": "e0f19ccf0ddab173",
  "{\"op\":\"reconstruct_open\"}|lines120": "0f6bcd3ab39ae3b4",
  "{\"op\":\"reconstruct_open\"}|one1": "2d5ed03cc9e0ff1d",
  "{\"op\":\"reconstruct_open\"}|PY": "91955cad95ef03ca",
  "{\"op\":\"reconstruct_open\"}|UI": "c68da7530d10c9d7",
  "{\"op\":\"convex_hull\"}|noise97": "65a633dfcd75d3b9",
  "{\"op\":\"convex_hull\"}|blob320": "244bf209dfe70c88",
  "{\"op\":\"convex_hull\"}|lines120": "12c83018560a4809",
  "{\"op\":\"convex_hull\"}|one1": "0ae5c167e33153bb",
  "{\"op\":\"convex_hull\"}|PY": "c134ce727412459a",
  "{\"op\":\"convex_hull\"}|UI": "6ba2b3ff98b1f526",
  "{\"op\":\"endpoints_branchpoints\"}|noise97": "0fd1d6cec68eb55e",
  "{\"op\":\"endpoints_branchpoints\"}|blob320": "b5239b762ff91590",
  "{\"op\":\"endpoints_branchpoints\"}|lines120": "7a51db2fc2b180cb",
  "{\"op\":\"endpoints_branchpoints\"}|one1": "08e1c7392a1b5774",
  "{\"op\":\"endpoints_branchpoints\"}|PY": "787ed2de1f29a919",
  "{\"op\":\"endpoints_branchpoints\"}|UI": "bce7958997d69b02",
  "{\"op\":\"erode\",\"shape\":\"diamond\",\"ksize\":7,\"iterations\":2}|noise97": "4e06a77a533f15ee",
  "{\"op\":\"erode\",\"shape\":\"diamond\",\"ksize\":7,\"iterations\":2}|blob320": "4f75db3fa997c9d8",
  "{\"op\":\"erode\",\"shape\":\"diamond\",\"ksize\":7,\"iterations\":2}|lines120": "e25ceab59f62dab4",
  "{\"op\":\"erode\",\"shape\":\"diamond\",\"ksize\":7,\"iterations\":2}|one1": "439d5b97ae88bc7c",
  "{\"op\":\"erode\",\"shape\":\"diamond\",\"ksize\":7,\"iterations\":2}|PY": "1db2daa516a15859",
  "{\"op\":\"erode\",\"shape\":\"diamond\",\"ksize\":7,\"iterations\":2}|UI": "54d64ca3c81d687a",
  "{\"op\":\"close\",\"shape\":\"line\",\"ksize\":9,\"lineAngle\":30}|noise97": "6c792c3522a7a73a",
  "{\"op\":\"close\",\"shape\":\"line\",\"ksize\":9,\"lineAngle\":30}|blob320": "01004116a6ee748c",
  "{\"op\":\"close\",\"shape\":\"line\",\"ksize\":9,\"lineAngle\":30}|lines120": "5feb72515f37d626",
  "{\"op\":\"close\",\"shape\":\"line\",\"ksize\":9,\"lineAngle\":30}|one1": "439d5b97ae88bc7c",
  "{\"op\":\"close\",\"shape\":\"line\",\"ksize\":9,\"lineAngle\":30}|PY": "192d962a785ee1bf",
  "{\"op\":\"close\",\"shape\":\"line\",\"ksize\":9,\"lineAngle\":30}|UI": "0bde8f0a7f36f669",
  "{\"op\":\"remove_small\",\"minArea\":40}|noise97": "d836a0c68ba1986b",
  "{\"op\":\"remove_small\",\"minArea\":40}|blob320": "d53db8774a362494",
  "{\"op\":\"remove_small\",\"minArea\":40}|lines120": "992b01a7b21abc14",
  "{\"op\":\"remove_small\",\"minArea\":40}|one1": "08c663b2bc674cff",
  "{\"op\":\"remove_small\",\"minArea\":40}|PY": "2680354fc6586335",
  "{\"op\":\"remove_small\",\"minArea\":40}|UI": "73479856a5893df0",
};

const M_CASES = MORPH_OLD.map((op) => ({ op }));
M_CASES.push({ op: 'erode', shape: 'diamond', ksize: 7, iterations: 2 }, { op: 'close', shape: 'line', ksize: 9, lineAngle: 30 }, { op: 'remove_small', minArea: 40 });
function goldenImages() {
  const r1 = lcg(31);
  const noise = new Uint8Array(97 * 61); for (let i = 0; i < noise.length; i++) noise[i] = r1() < 0.5 ? 255 : 0;
  const blob = new cv.Mat(240, 320, cv.CV_8UC1, new cv.Scalar(0));
  cv.circle(blob, new cv.Point(80, 80), 40, new cv.Scalar(255), -1);
  cv.rectangle(blob, new cv.Point(150, 40), new cv.Point(230, 120), new cv.Scalar(255), -1);
  cv.ellipse(blob, new cv.Point(240, 180), new cv.Size(50, 20), 30, 0, 360, new cv.Scalar(255), -1);
  cv.circle(blob, new cv.Point(60, 190), 15, new cv.Scalar(255), -1);
  cv.circle(blob, new cv.Point(80, 80), 12, new cv.Scalar(0), -1);
  cv.rectangle(blob, new cv.Point(300, 200), new cv.Point(319, 239), new cv.Scalar(255), -1);
  const lines = new cv.Mat(90, 120, cv.CV_8UC1, new cv.Scalar(0));
  cv.line(lines, new cv.Point(10, 10), new cv.Point(100, 10), new cv.Scalar(255), 1);
  cv.line(lines, new cv.Point(50, 10), new cv.Point(50, 80), new cv.Scalar(255), 1);
  cv.line(lines, new cv.Point(20, 60), new cv.Point(110, 85), new cv.Scalar(255), 1);
  return { noise97: cv.matFromArray(61, 97, cv.CV_8UC1, noise), blob320: blob, lines120: lines, one1: cv.matFromArray(1, 1, cv.CV_8UC1, [255]) };
}
const gsha = (...parts) => { const h = crypto.createHash('sha256'); for (const p of parts) h.update(p); return h.digest('hex').slice(0, 16); };
function goldenRun(params, src) {
  const ctx = {};
  try {
    const out = OPS.morphology.apply(cv, src, params, ctx);
    const h = gsha(Buffer.from(out.data), `${out.rows}x${out.cols}t${out.type()}`, JSON.stringify(ctx.info ?? null), JSON.stringify(ctx.values ?? null));
    out.delete(); return h;
  } catch (e) { return 'ERR:' + (e && e.message ? e.message : typeof e === 'number' ? 'cvexc' : String(e)).slice(0, 60); }
}
function goldenAll(variant) {
  const imgs = goldenImages(), res = {};
  const defs = {}; for (const s of OPS.morphology.params) defs[s.key] = s.default;
  const NEW_DEF = { hmPattern: 'isolated', hmRotate: true, hHeight: 10 };
  const NEW_WEIRD = { hmPattern: 'constructor', hmRotate: 'x', hHeight: 'junk' };
  try {
    for (const c of M_CASES) {
      const p0 = { ...defs, ...c };
      for (const k of Object.keys(NEW_DEF)) delete p0[k];
      const p = variant === 'A' ? p0 : variant === 'B' ? { ...p0, ...NEW_DEF } : { ...p0, ...NEW_WEIRD };
      const key = JSON.stringify(c);
      for (const [n, m] of Object.entries(imgs)) res[`${key}|${n}`] = goldenRun(p, m);
      const pp = {}; for (const [k, v] of Object.entries(p)) pp[k] = v;
      res[`${key}|PY`] = gsha(buildPythonCode([{ id: 'morphology', params: pp, nodeId: 1 }]));
      res[`${key}|UI`] = gsha(String(OP_PRIMARY.morphology(p)));
    }
  } finally { for (const m of Object.values(imgs)) m.delete(); }
  return res;
}
for (const [variant, what] of [['A', 'the old params'], ['B', 'every new param passed explicitly at its default'], ['C', 'every new param set to junk / "constructor"']]) {
  test(`ip4b-G${variant}. golden: the 14 existing Morphology operations (+3 custom-kernel cases) give byte-identical image + ctx.info + exported Python + card text on 4 deterministic images (${what}) -- 102 hashes measured on HEAD ffce85b before any edit`, () => {
    const cur = goldenAll(variant);
    assert.equal(Object.keys(GOLDEN).length, 102);
    const bad = Object.keys(GOLDEN).filter((k) => cur[k] !== GOLDEN[k]);
    assert.deepEqual(bad, [], `${bad.length} of 102 differ: ${bad.slice(0, 5).join(' ; ')}`);
    assert.ok(!Object.values(GOLDEN).some((v) => v.startsWith('ERR:')), 'no golden case throws on the old code');
  });
}

// ===================================================================
// ip4b-S — static structure / text / clamps / codegen
// ===================================================================
test('ip4b-S1. (static structure) the 8 new ops exist with the right needs and host groups; OPS = 115 (96 + 19 morph_* from 2026-10-18), FILTERS = 34 (33 + morph_tools); Morphology has 19 operations (the old 14 first, in order); no select key is "mode"; defaultParams() == the OPS defaults (prefixed); legacy groups untouched; the category groups = the palette sections in order', () => {
  assert.equal(Object.keys(OPS).length, 115); // 96 + 19 morph_* (Morphology แยกเครื่องมือ, 2026-10-18)
  assert.equal(Object.keys(FILTERS).length, 34); // +morph_tools (2026-10-18)
  for (const id of NEW_OPS) {
    assert.ok(OPS[id], id); assert.equal(OPS[id].needs, NEEDS[id], `${id}.needs`);
    assert.equal(OP_TO_FILTER[id], HOST[id], `${id} host`);
    for (const p of OPS[id].params) assert.notEqual(p.key, 'mode', `${id}.${p.key}`);
    const host = FILTERS[HOST[id]], pre = defaultParams(HOST[id]);
    assert.ok(host.params.find((p) => p.key === 'mode').options.some((o) => o.value === id), `${id} is a mode of ${HOST[id]}`);
    for (const p of OPS[id].params) assert.deepEqual(pre[`${id}__${p.key}`], p.default, `${id}.${p.key} default`);
    assert.equal(needsOf(host, { mode: id }), NEEDS[id]);
  }
  assert.equal(FILTERS.image_arithmetic.inputs, 2);
  for (const id of ['deconv_wiener', 'deconv_lucy', 'nlm_denoise', 'gabor_filter', 'glcm', 'dehaze', 'dct_jpeg']) assert.equal(FILTERS[HOST[id]].inputs, undefined, `${id} host is single-input`);
  // Morphology: 19 options, the first 14 unchanged and in order, the new 5 appended
  const opOptions = OPS.morphology.params.find((p) => p.key === 'op').options.map((o) => o.value);
  assert.equal(opOptions.length, 19);
  assert.deepEqual(opOptions.slice(0, 14), MORPH_OLD);
  assert.deepEqual(opOptions.slice(14), MORPH_NEW);
  assert.equal(OPS.morphology.params.find((p) => p.key === 'op').default, 'erode');
  assert.deepEqual(OPS.morphology.params.slice(-3).map((p) => p.key), ['hmPattern', 'hmRotate', 'hHeight']);
  assert.deepEqual(OPS.morphology.params.slice(0, 6).map((p) => p.key), ['op', 'shape', 'ksize', 'iterations', 'minArea', 'lineAngle']);
  assert.equal(defaultParams('morphology').hmPattern, 'isolated'); assert.equal(defaultParams('morphology').hmRotate, true); assert.equal(defaultParams('morphology').hHeight, 10);
  assert.deepEqual(OPS.morphology.params.find((p) => p.key === 'hmPattern').options.map((o) => o.value), Object.keys(HITMISS_PATTERNS));
  const modes = (id) => FILTERS[id].params.find((p) => p.key === 'mode').options.map((o) => o.value);
  // legacy groups are exactly what they were before this batch
  assert.deepEqual(modes('colour'), ['color_inrange', 'kmeans_segment', 'white_balance', 'invert', 'slic']);
  assert.deepEqual(modes('blur'), ['median_blur', 'gaussian_blur', 'box_blur', 'bilateral', 'wiener_filter']);
  assert.deepEqual(modes('sharpen'), ['unsharp', 'laplacian_sharpen', 'custom_kernel']);
  assert.deepEqual(modes('image_adjust'), ['brightness_contrast', 'gamma', 'hist_equalize', 'clahe', 'contrast_stretch', 'flatfield_correction']);
  assert.deepEqual(modes('shape_analysis'), ['find_contours', 'connected_components', 'distance_transform', 'local_std_filter', 'local_entropy_filter', 'regionprops', 'feret', 'line_profile']);
  assert.deepEqual(modes('color_space'), ['grayscale', 'colorspace', 'channel_split', 'rgb_adjust', 'color_inrange', 'invert']);
  // the category groups = the palette sections, in order (cm-1) · defaults unchanged
  for (const [fid, title, dflt] of [['enhance_tools', 'ENHANCE', 'brightness_contrast'], ['filter_denoise', 'FILTER & DENOISE', 'salt_pepper_noise'], ['analysis_detect', 'ANALYSIS & DETECTION', 'find_contours'], ['frequency', 'FREQUENCY DOMAIN', 'fft_magnitude'], ['image_arithmetic', 'IMAGE ARITHMETIC', 'imabsdiff']]) {
    assert.deepEqual(modes(fid), TOOL_SECTIONS.find((s) => s.title === title).ops, `${fid} == ${title}`);
    assert.equal(FILTERS[fid].params.find((p) => p.key === 'mode').default, dflt);
  }
  const sec = (t) => TOOL_SECTIONS.find((s) => s.title === t).ops;
  assert.deepEqual(sec('ENHANCE').slice(-2), ['custom_kernel', 'dehaze']);
  assert.deepEqual(sec('FILTER & DENOISE').slice(-5), ['wiener_filter', 'guided_filter', 'nlm_denoise', 'deconv_wiener', 'deconv_lucy']);
  const ad = sec('ANALYSIS & DETECTION');
  assert.deepEqual(ad.slice(ad.indexOf('local_entropy_filter'), ad.indexOf('local_entropy_filter') + 5), ['local_entropy_filter', 'gabor_filter', 'glcm', 'focus_score', 'regionprops']);
  assert.deepEqual(sec('FREQUENCY DOMAIN').slice(-2), ['bandpass_notch', 'dct_jpeg']);
  assert.deepEqual(sec('IMAGE ARITHMETIC').slice(-2), ['roi_apply', 'ecc_register']);
  // 2026-10-18: MORPHOLOGY = 19 เครื่องมือ morph_* (ลำดับ/ความหมายล็อกใน tests/morph-split.test.mjs) · ตัวเก่า 'morphology' เป็น legacy ไม่อยู่บน palette
  assert.deepEqual(sec('MORPHOLOGY'), modes('morph_tools'));
  assert.equal(sec('MORPHOLOGY').length, 19);
  for (const id of NEW_OPS) assert.equal(TOOL_SECTIONS.filter((s) => s.ops.includes(id)).length, 1, `${id} in exactly one section`);
  // slow-mode hint only on nlm_denoise and deconv_lucy, and it lands in the mode dropdown
  for (const id of Object.keys(OPS)) {
    const has = typeof OPS[id].modeHint === 'string' && OPS[id].modeHint.includes('ช้า — เหมาะกับภาพนิ่ง');
    assert.equal(has, id === 'nlm_denoise' || id === 'deconv_lucy', `${id} slow modeHint`);
  }
  assert.ok(FILTERS.filter_denoise.params[0].options.find((o) => o.value === 'nlm_denoise').hint.includes('ช้า'));
  assert.ok(FILTERS.filter_denoise.params[0].options.find((o) => o.value === 'deconv_lucy').hint.includes('ช้า'));
  // psfParamDefs() == the PSF params of custom_kernel (range + default)
  const ck = Object.fromEntries(OPS.custom_kernel.params.map((p) => [p.key, p]));
  const pd = psfParamDefs();
  assert.deepEqual(pd.map((p) => p.key), ['psf', 'psfSize', 'psfAngle']);
  for (const k of ['psfSize', 'psfAngle']) { const q = pd.find((p) => p.key === k); for (const f of ['min', 'max', 'step', 'default']) assert.equal(q[f], ck[k][f], `${k}.${f}`); }
  assert.deepEqual(pd[0].options.map((o) => o.value), ['motion', 'disk']); assert.equal(pd[0].default, 'motion');
  for (const id of ['deconv_wiener', 'deconv_lucy']) assert.deepEqual(OPS[id].params.slice(0, 3).map((p) => p.key), ['psf', 'psfSize', 'psfAngle']);
  assert.equal(OPS.deconv_lucy.params.find((p) => p.key === 'iterations').max, LUCY_MAX_ITERS); assert.equal(LUCY_MAX_ITERS, 50);
  assert.equal(OPS.ecc_register.params.find((p) => p.key === 'maxSide').default, ECC_DEFAULT_SIDE); assert.equal(ECC_DEFAULT_SIDE, 320); assert.equal(NLM_MAX_SIDE, 640);
  assert.equal(psfInfo('motion', 9, 0, 9), 'motion ยาว 9 px มุม 0° (9x9)'); assert.equal(psfInfo('disk', 9, 0, 19), 'disk รัศมี 9 px (19x19)');
});

test('ip4b-S2. (static) the 8 new OP_ICON glyphs are distinct from every other OP_ICON / GROUP_ICON glyph and from the markers 🧩 / 📊 / 📌, and appear nowhere in app.js / detail-panel.js / editor.js; OP_PRIMARY card text exists for every new op / operation and reads the clamped values', () => {
  const src = readSrc('ui/palette-panel.js');
  const body = src.slice(src.indexOf('const OP_ICON = {'), src.indexOf('};', src.indexOf('const OP_ICON = {')));
  const icons = {}; for (const m of body.matchAll(/([a-z_0-9]+): "([^"]+)"/g)) icons[m[1]] = m[2];
  for (const id of NEW_OPS) assert.equal(icons[id], NEW_ICONS[id], `${id} icon`);
  const all = Object.entries(icons);
  assert.equal(new Set(all.map(([, v]) => v)).size, all.length, 'every OP_ICON glyph is unique');
  for (const [id, g] of Object.entries(NEW_ICONS)) {
    for (const [k, v] of Object.entries(GROUP_ICON)) assert.notEqual(v, g, `${id} vs GROUP_ICON.${k}`);
    for (const m of ['🧩', '📊', '📌']) assert.notEqual(g, m);
    for (const f of ['app.js', 'ui/detail-panel.js', 'graph/editor.js']) assert.ok(!readSrc(f).includes(g), `${g} appears in ${f}`);
  }
  assert.equal(OP_PRIMARY.deconv_wiener({}), 'motion 9px 0° · NSR 0.01'); assert.equal(OP_PRIMARY.deconv_wiener({ psf: 'disk', psfSize: 7, nsr: 0.05 }), 'disk r7 · NSR 0.05');
  assert.equal(OP_PRIMARY.deconv_wiener({ psf: 'constructor', psfSize: 999, psfAngle: -5, nsr: 9 }), 'motion 50px 0° · NSR 0.5');
  assert.equal(OP_PRIMARY.deconv_lucy({}), '10 รอบ · motion 9px 0°'); assert.equal(OP_PRIMARY.deconv_lucy({ iterations: 999, psf: 'disk', psfSize: 5 }), '50 รอบ · disk r5');
  assert.equal(OP_PRIMARY.ecc_register({}), 'เลื่อน + หมุน · ≤ 50 รอบ · ย่อ 320px'); assert.equal(OP_PRIMARY.ecc_register({ motion: 'homography', iterations: 5, maxSide: 5000 }), 'มุมมองเปลี่ยน · ≤ 10 รอบ · ย่อ 1024px');
  assert.equal(OP_PRIMARY.ecc_register({ motion: 'constructor' }), 'เลื่อน + หมุน · ≤ 50 รอบ · ย่อ 320px');
  assert.equal(OP_PRIMARY.gabor_filter({}), 'λ 8px · θ 0°'); assert.equal(OP_PRIMARY.gabor_filter({ wavelength: 999, orientation: 500 }), 'λ 64px · θ 179°');
  assert.equal(OP_PRIMARY.glcm({}), '8 ระดับ · d=1 · 0°'); assert.equal(OP_PRIMARY.glcm({ levels: '64', distance: 99, angle: 'avg' }), '64 ระดับ · d=10 · เฉลี่ย 4 ทิศ'); assert.equal(OP_PRIMARY.glcm({ levels: 'constructor', angle: 'constructor' }), '8 ระดับ · d=1 · 0°');
  assert.equal(OP_PRIMARY.nlm_denoise({}), 'h 10 · แผ่น 5 · ค้นหา 11'); assert.equal(OP_PRIMARY.nlm_denoise({ h: 99, patch: '7', search: 'constructor' }), 'h 50 · แผ่น 7 · ค้นหา 11');
  assert.equal(OP_PRIMARY.dehaze({}), 'ω 0.95 · หน้าต่าง 15'); assert.equal(OP_PRIMARY.dehaze({ patch: 4, refine: false }), 'ω 0.95 · หน้าต่าง 5 · ไม่เกลี่ย'); assert.equal(OP_PRIMARY.dehaze({ amount: 7, patch: 99 }), 'ω 1 · หน้าต่าง 31');
  assert.equal(OP_PRIMARY.dct_jpeg({}), 'JPEG q50'); assert.equal(OP_PRIMARY.dct_jpeg({ method: 'zonal', keep: 3 }), 'ความถี่ต่ำ 3×3'); assert.equal(OP_PRIMARY.dct_jpeg({ quality: 500 }), 'JPEG q100');
  // morphology card text: the 14 old strings are pinned by ip4b-G (UI hash) -- the new ones here
  assert.equal(OP_PRIMARY.morphology({ op: 'hit_or_miss' }), 'Hit-or-Miss · จุดเดี่ยว'); assert.equal(OP_PRIMARY.morphology({ op: 'hit_or_miss', hmPattern: 'corner' }), 'Hit-or-Miss · มุมนูน');
  assert.equal(OP_PRIMARY.morphology({ op: 'hit_or_miss', hmPattern: 'constructor' }), 'Hit-or-Miss · จุดเดี่ยว');
  assert.equal(OP_PRIMARY.morphology({ op: 'regional_max' }), 'Regional Maxima'); assert.equal(OP_PRIMARY.morphology({ op: 'regional_min' }), 'Regional Minima');
  assert.equal(OP_PRIMARY.morphology({ op: 'h_maxima' }), 'H-maxima h=10'); assert.equal(OP_PRIMARY.morphology({ op: 'extended_max', hHeight: 999 }), 'Extended Maxima h=255'); assert.equal(OP_PRIMARY.morphology({ op: 'h_maxima', hHeight: 'junk' }), 'H-maxima h=10');
});

test('ip4b-S3. (static read) every new numeric param is clamped to EXACTLY the UI default / min / max in registry.js (numIn / intIn) and in codegen.js (clampNum); the speed constants are used from registry in codegen (not hard-coded again)', () => {
  const reg = readSrc('filters/registry.js'), gen = readSrc('codegen.js');
  const CONST = { iterations: { deconv_lucy: 'LUCY_MAX_ITERS' }, maxSide: { ecc_register: 'ECC_DEFAULT_SIDE' } };
  const INT = { // int = the registry rounds (intIn) · codegen rounds too
    deconv_wiener: { psfSize: 1, psfAngle: 1, nsr: 0 }, deconv_lucy: { psfSize: 1, psfAngle: 1, iterations: 1 }, ecc_register: { iterations: 1, maxSide: 1 },
    gabor_filter: { wavelength: 1, orientation: 1, bandwidth: 0, aspect: 0 }, glcm: { distance: 1 }, nlm_denoise: { h: 1 }, dehaze: { amount: 0, patch: 1 }, dct_jpeg: { quality: 1, keep: 1 },
  };
  for (const [id, keys] of Object.entries(INT)) {
    for (const [k, isInt] of Object.entries(keys)) {
      const sp = OPS[id].params.find((p) => p.key === k) || psfParamDefs().find((p) => p.key === k);
      assert.ok(sp, `${id}.${k}`);
      const cname = (CONST[k] || {})[id];
      const lo = sp.min, hi = cname === 'LUCY_MAX_ITERS' ? 'LUCY_MAX_ITERS' : sp.max, d = cname === 'ECC_DEFAULT_SIDE' ? 'ECC_DEFAULT_SIDE' : sp.default;
      const call = `${isInt ? 'intIn' : 'numIn'}(p.${k}, ${d}, ${lo}, ${hi})`;
      if (id === 'deconv_wiener' || id === 'deconv_lucy') {
        if (k === 'psfSize' || k === 'psfAngle') assert.ok(reg.includes(`intIn(p.${k}, ${sp.default}, ${sp.min}, ${sp.max})`), `registry psf ${k}`);
        else assert.ok(reg.includes(call), `registry ${id}.${k}: ${call}`);
      } else if (id === 'dehaze' && k === 'patch') assert.ok(reg.includes(`ensureOdd(intIn(p.patch, ${sp.default}, ${sp.min}, ${sp.max}))`), 'registry dehaze patch (odd)');
      else assert.ok(reg.includes(call), `registry ${id}.${k}: ${call}`);
      const gcall = `clampNum(p.${k}, ${d}, ${lo}, ${hi})`;
      assert.ok(gen.includes(gcall), `codegen ${id}.${k}: ${gcall}`);
    }
  }
  for (const k of ['hHeight']) assert.ok(reg.includes('intIn(p.hHeight, 10, 1, 255)') && gen.includes('clampNum(p.hHeight, 10, 1, 255)'), k);
  const hh = OPS.morphology.params.find((p) => p.key === 'hHeight'); assert.deepEqual([hh.min, hh.max, hh.default], [1, 255, 10]);
  // speed constants: imported from registry, never re-typed
  assert.match(gen, /import \{[^}]*NLM_MAX_SIDE[^}]*LUCY_MAX_ITERS[^}]*ECC_DEFAULT_SIDE[^}]*\} from "\.\/filters\/registry\.js"/);
  { const nlmGen = gen.slice(gen.indexOf('  nlm_denoise: { needs'), gen.indexOf('  dehaze: { needs')); assert.ok(nlmGen.includes('${NLM_MAX_SIDE}') && !/640/.test(nlmGen), 'codegen nlm uses NLM_MAX_SIDE'); const eccGen = gen.slice(gen.indexOf('  ecc_register: { needs'), gen.indexOf('  gabor_filter: { needs')); assert.ok(!/320/.test(eccGen), 'codegen ecc does not re-type 320'); }
  assert.ok(reg.includes('analysisScale(W, H, NLM_MAX_SIDE)') && reg.includes('analysisScale(W, H, maxSide)'));
  // UI: the card shows the value the op really uses (clamped)
  assert.equal(OP_PRIMARY.gabor_filter({ wavelength: -4 }), 'λ 2px · θ 0°');
});

test('ip4b-S4. (static) Thai text: no "ที่สุด" in any new name / desc / label / hint / option / info / VAL_META line; the required desc points of every op are present; always-visible Morphology params say which operation they belong to', () => {
  const texts = [];
  const walk = (o) => { if (typeof o === 'string') texts.push(o); else if (Array.isArray(o)) o.forEach(walk); else if (o && typeof o === 'object') Object.values(o).forEach(walk); };
  for (const id of NEW_OPS) walk({ name: OPS[id].name, desc: OPS[id].desc, params: OPS[id].params, hint: OPS[id].modeHint });
  walk(OPS.morphology.params.slice(-3)); walk(OPS.morphology.params[0].options.slice(14)); texts.push(OPS.morphology.desc.slice(OPS.morphology.desc.indexOf('Hit-or-Miss หา')));
  const dp = readSrc('ui/detail-panel.js'); texts.push(dp.slice(dp.indexOf('// IP batch 4B'), dp.indexOf('const META_FALLBACK')));
  for (const t of texts) assert.ok(!t.includes('ที่สุด'), `"ที่สุด" in: ${t.slice(0, 80)}`);
  const d = (id) => OPS[id].desc;
  for (const re of [/PSF/, /หน้าตาของจุดเดียว/, /H\*\/\(\|H\|² \+ NSR\)/, /Custom Kernel/, /ringing|รอยคลื่น/, /edgetaper/, /รอยปัดเศษ/, /deconvwnr/, /blind deconvolution/]) assert.match(d('deconv_wiener'), re);
  for (const re of [/เดาภาพคมทีละรอบ/, /ไม่ติดลบ/, /Wiener/, /5–10 รอบ/, /ครึ่งวินาที/, /deconvlucy/, /acceleration/, /damping/]) assert.match(d('deconv_lucy'), re);
  for (const re of [/ECC/, /Absolute Difference/, /10%/, /ลวดลาย/, /ไม่ลู่เข้า/, /มุมซ้ายบนของ B/, /imregister/, /0\.05 px/]) assert.match(d('ecc_register'), re);
  for (const re of [/cos คูณระฆังเกาส์/, /0\/45\/90\/135/, /magnitude/, /imgaborfilt/, /ลบค่าเฉลี่ย/, /getGaborKernel/, /ปรับความสว่างอัตโนมัติ/]) assert.match(d('gabor_filter'), re);
  for (const re of [/\(i, j\)/, /Contrast/, /Energy/, /Homogeneity/, /Correlation/, /ตั้งค่าเดียวกัน/, /Crop/, /graycomatrix/, /graycoprops/]) assert.match(d('glcm'), re);
  for (const re of [/ลายรอบตัวคล้ายกัน/, /h คุม/, /1 วินาที/, /imnlmfilt/, /fastNlMeansDenoising/]) assert.match(d('nlm_denoise'), re);
  for (const re of [/dark channel/, /A = สีของหมอก/, /t = สัดส่วนแสง/, /ท้องฟ้า/, /guided filter/, /imreducehaze/]) assert.match(d('dehaze'), re);
  for (const re of [/8×8/, /ความถี่/, /quantization/, /Huffman/, /blocking/, /16\/64/, /Y\/Cb\/Cr/, /dct2/, /cv\.dct/, /ไม่ใช่งานตรวจสอบ/]) assert.match(d('dct_jpeg'), re);
  assert.match(OPS.morphology.desc, /bwhitmiss/); assert.match(OPS.morphology.desc, /ไม่สน/); assert.match(OPS.morphology.desc, /marker/); assert.match(OPS.morphology.desc, /8-connected/); assert.match(OPS.morphology.desc, /Endpoints & Branchpoints/);
  for (const k of ['hmPattern', 'hmRotate']) assert.match(OPS.morphology.params.find((p) => p.key === k).label, /ใช้กับ Hit-or-Miss เท่านั้น/);
  assert.match(OPS.morphology.params.find((p) => p.key === 'hHeight').label, /ใช้กับ H-maxima \/ Extended Maxima เท่านั้น/);
  // psf params shown only for the op (merged group showIf) but still say who they are for
  assert.match(psfParamDefs()[2].label, /ใช้กับ Motion เท่านั้น/);
  // wiener / lucy: psfSize says it must match the real blur
  assert.match(psfParamDefs()[1].label, /ต้องตรงกับที่ภาพเบลอจริง/);
  // the select defaults
  const dflt = (id, k) => OPS[id].params.find((p) => p.key === k).default;
  assert.equal(dflt('ecc_register', 'motion'), 'euclidean'); assert.equal(dflt('ecc_register', 'view'), 'aligned'); assert.equal(dflt('gabor_filter', 'output'), 'magnitude');
  assert.equal(dflt('glcm', 'levels'), '8'); assert.equal(dflt('glcm', 'angle'), '0'); assert.equal(dflt('glcm', 'symmetric'), false); assert.equal(dflt('nlm_denoise', 'patch'), '5'); assert.equal(dflt('nlm_denoise', 'search'), '11');
  assert.equal(dflt('dehaze', 'refine'), true); assert.equal(dflt('dct_jpeg', 'method'), 'jpeg'); assert.equal(dflt('deconv_wiener', 'nsr'), 0.01); assert.equal(dflt('deconv_lucy', 'iterations'), 10);
  // VAL_META: every new scalar key has a line with decimals
  const vm = dp.slice(dp.indexOf('const VAL_META = {'), dp.indexOf('const META_FALLBACK'));
  for (const k of [...ECC_KEYS, ...GABOR_KEYS, ...GLCM_KEYS, ...DEHAZE_KEYS, ...DCT_KEYS]) assert.match(vm, new RegExp(`\\n  ${k}:\\s+\\{[^\\n]*d: \\d`), `VAL_META.${k}`);
  assert.match(vm, /glcm_contrast:\s+\{ d: 4/); assert.match(vm, /glcm_pairs:\s+\{ d: 0/); assert.match(vm, /dct_psnr:\s+\{ d: 2[^\n]*infTip/); assert.match(vm, /gabor_max:\s+\{ d: 1 \}/); assert.match(vm, /dh_t_mean:\s+\{ d: 3 \}/);
  assert.match(vm, /dct_blocks:\s+\{ d: 0 \}/); assert.match(vm, /dct_zero_pct:\s+\{ d: 1 \}/);
});

test('ip4b-S5. (static) codegen of the new paths: no "WARNING: skipped" / "ข้าม" / round( / ** 0.5 / import math / np.round / math. ; print() text is ASCII only; every new select reads own-property only ("constructor" = default); numbers are printed with the same floor(x*10^d+0.5) rounding as the app; the default Morphology paths have no new lines', () => {
  const FORBID = ['WARNING: skipped', 'ข้าม', 'round(', '** 0.5', 'import math', 'np.round', 'math.'];
  const V = [
    ['deconv_wiener', {}], ['deconv_wiener', { psf: 'disk', psfSize: 7, nsr: 0 }], ['deconv_wiener', { psfAngle: 45, psfSize: 15, nsr: 0.2 }],
    ['deconv_lucy', {}], ['deconv_lucy', { psf: 'disk', psfSize: 5, iterations: 3 }], ['ecc_register', {}], ['ecc_register', { motion: 'translation', view: 'diff', maxSide: 128 }], ['ecc_register', { motion: 'homography', view: 'overlay', iterations: 100 }], ['ecc_register', { motion: 'affine' }],
    ['gabor_filter', {}], ['gabor_filter', { wavelength: 5, orientation: 45, bandwidth: 2, aspect: 0.8, output: 'real' }], ['glcm', {}], ['glcm', { levels: '16', angle: 'avg', symmetric: true, distance: 3, view: 'quantized' }], ['glcm', { angle: '45' }],
    ['nlm_denoise', {}], ['nlm_denoise', { h: 20, patch: '7', search: '15' }], ['dehaze', {}], ['dehaze', { refine: false, view: 'transmission', patch: 9, amount: 0.5 }], ['dehaze', { view: 'dark' }],
    ['dct_jpeg', {}], ['dct_jpeg', { method: 'zonal', keep: 3, view: 'coeffs' }], ['dct_jpeg', { quality: 20, view: 'error' }],
    ['morphology', { op: 'hit_or_miss' }], ['morphology', { op: 'hit_or_miss', hmPattern: 'corner', hmRotate: false }], ['morphology', { op: 'regional_max' }], ['morphology', { op: 'regional_min' }], ['morphology', { op: 'h_maxima', hHeight: 25 }], ['morphology', { op: 'extended_max', hHeight: 3 }],
  ];
  for (const [id, p] of V) {
    const code = buildPythonCode([OPSTEP(id, p)]);
    for (const bad of FORBID) assert.ok(!code.includes(bad), `${id} ${JSON.stringify(p)}: contains ${bad}`);
    for (const line of code.split('\n')) { const m = line.match(/^\s*print\((.*)\)\s*(#.*)?$/); if (m && !line.includes('ลดเหลือ {iters_used}')) assert.ok(/^[\x00-\x7f]*$/.test(m[1]), `${id}: non-ASCII print: ${line.slice(0, 100)}`); }
    if (id !== 'morphology') assert.match(code, new RegExp(`# ── ขั้นที่ 1: ${OPS[id].name.replace(/[()]/g, '\\$&')} \\(${id}\\) ──`));
  }
  const k = (id, p) => buildPythonCode([OPSTEP(id, p)]);
  // unknown selects emit the default path
  assert.ok(k('deconv_wiener', { psf: 'constructor' }).includes('psf_kernel("motion"'));
  assert.ok(k('ecc_register', { motion: 'constructor', view: 'constructor' }).includes('cv2.MOTION_EUCLIDEAN') && !k('ecc_register', { view: 'constructor' }).includes('cv2.absdiff'));
  assert.ok(!k('gabor_filter', { output: 'constructor' }).includes('resp = re') && k('glcm', { levels: 'constructor', angle: 'constructor', view: 'constructor' }).includes('levels = 8'));
  assert.ok(k('nlm_denoise', { patch: 'constructor', search: 'constructor' }).includes('pr = 2') && k('nlm_denoise', { patch: 'constructor', search: 'constructor' }).includes('sr = 5'));
  assert.ok(k('dehaze', { view: 'constructor' }).includes('img = res if nc > 1 else res[:, :, 0]') && k('dct_jpeg', { method: 'constructor', view: 'constructor' }).includes('q50 = np.array') && k('dct_jpeg', { view: 'constructor' }).includes('img = rec'));
  assert.ok(k('morphology', { op: 'hit_or_miss', hmPattern: 'constructor' }).includes('[[-1,-1,-1],[-1,1,-1],[-1,-1,-1]]'));
  // number printing = the app's rounding (np.floor(x * 10^d + 0.5) / 10^d), never a bare :.2f of the raw value
  for (const [id, p] of [['ecc_register', {}], ['gabor_filter', {}], ['glcm', {}], ['dehaze', {}], ['dct_jpeg', {}]]) {
    for (const line of k(id, p).split('\n')) { const m = line.match(/^\s*print\((.*)\)\s*(#.*)?$/); if (!m) continue; for (const f of m[1].matchAll(/\{([^{}]*?):\.(\d)f\}/g)) assert.match(f[1], /np\.floor\(.*\* (10|100|1000|10 \*\* d) \+ 0\.5\) \/ (10|100|1000|10 \*\* d)|np\.floor\(v \* 10 \*\* d/, `${id}: bare format in ${line.slice(0, 120)}`); }
  }
  // helper defs: no bare `img` token (the src2-chain rewrite replaces \bimg\b in every line of the B chain)
  for (const [id, p, fn, end] of [['deconv_wiener', {}, 'def wiener_channel', '    return np.where(np.isfinite(v)'], ['deconv_lucy', {}, 'def lucy_channel', '    return np.clip('], ['glcm', {}, 'def glcm_counts', '    return [contrast, corr, energy, homog, n]']]) {
    const code = k(id, p), i = code.indexOf(fn), j = code.indexOf(end, i); assert.ok(i > 0 && j > i, `${id} def span`);
    assert.ok(!/\bimg\b/.test(code.slice(i, j)), `${id}: a helper def mentions img`);
  }
  // the 2-image builder is used for ecc_register (reads img_b at run time)
  const ecc = k('ecc_register', {}); assert.ok(ecc.includes('def process(path_a, path_b)') && ecc.includes('img_b.shape[:2] == img.shape[:2]'));
  // default Morphology op path has none of the new lines
  for (const op of MORPH_OLD) { const c = k('morphology', { op }); for (const t of ['hit-or-miss', 'recon(', 'MORPH_HITMISS', 'regional_max']) assert.ok(!c.includes(t), `${op} Python contains ${t}`); }
  // ECC constants are the ones in registry
  assert.ok(ecc.includes(`scale = min(1.0, ${ECC_DEFAULT_SIDE} / max(ww, hh))`));
  assert.ok(k('nlm_denoise', {}).includes(`scale = min(1.0, ${NLM_MAX_SIDE} / max(ww0, hh0))`));
  assert.ok(k('deconv_lucy', { iterations: 999 }).includes(`iters = ${LUCY_MAX_ITERS}`));
});

// ===================================================================
// ตัวอ้างอิงในไฟล์นี้ (เขียนแยกจากโค้ดในแอป — นิยามตรง ๆ / brute force) ใช้เทียบทั้งในเทสต์ JS ล้วนและเทสต์ Python
// ===================================================================
const clampI = (v, hi) => (v < 0 ? 0 : v > hi ? hi : v);
// reconstruction by dilation ตามนิยาม: วน dilate 3×3 + min(mask) จนไม่เปลี่ยน (ช้าแต่ผลมีหนึ่งเดียว)
function reconRef(J0, I, W, H) {
  let J = Int32Array.from(J0);
  for (;;) {
    const nj = new Int32Array(J.length);
    let changed = false;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const p = y * W + x;
        let m = J[p];
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy >= 0 && yy < H && xx >= 0 && xx < W && J[yy * W + xx] > m) m = J[yy * W + xx];
        }
        nj[p] = Math.min(m, I[p]);
        if (nj[p] !== J[p]) changed = true;
      }
    }
    J = nj;
    if (!changed) return J;
  }
}
const i32 = (m) => Int32Array.from(m.data);
function regMaxRef(I, W, H) { const R = reconRef(I.map((v) => v - 1), I, W, H); return Uint8Array.from(I, (v, i) => (v - R[i] >= 1 ? 255 : 0)); }
function hMaxRef(I, W, H, h) { return reconRef(I.map((v) => Math.max(v - h, 0)), I, W, H); }
function spiralMask(n) { // เส้นทางเกลียวกว้าง 1 px เว้นผนัง 1 px ระหว่างขด (8-connected ต่อกันทางเดียวตลอดเส้น)
  const m = new Uint8Array(n * n);
  let x = 0, y = 0, d = 0, turns = 0;
  const dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]];
  const inb = (a, b) => a >= 0 && b >= 0 && a < n && b < n;
  m[0] = 255;
  while (turns < 2) {
    const nx = x + dirs[d][0], ny = y + dirs[d][1], n2x = nx + dirs[d][0], n2y = ny + dirs[d][1];
    if (inb(nx, ny) && !m[ny * n + nx] && !(inb(n2x, n2y) && m[n2y * n + n2x])) { x = nx; y = ny; m[y * n + x] = 255; turns = 0; } else { d = (d + 1) % 4; turns++; }
  }
  return m;
}
// Hit-or-Miss ตามนิยาม (นอกภาพ = ไม่สน) บนภาพ 0/255
function hitMissRef(bin, W, H, k3) {
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let ok = true;
    for (let j = -1; j <= 1 && ok; j++) for (let i = -1; i <= 1; i++) {
      const v = k3[j + 1][i + 1];
      if (v === 0) continue;
      const yy = y + j, xx = x + i;
      if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
      const px = bin[yy * W + xx] !== 0;
      if ((v === 1 && !px) || (v === -1 && px)) { ok = false; break; }
    }
    out[y * W + x] = ok ? 255 : 0;
  }
  return out;
}
// NLM brute force (ไม่ใช้ integral image) — ลำดับ offset / การบวกเดียวกับ nlmCore → ต้องได้ byte เท่ากัน
function nlmRef(data, W, H, ch, h, pr, sr) {
  const out = new Float64Array(W * H * ch), wsum = new Float64Array(W * H);
  for (let dy = -sr; dy <= sr; dy++) for (let dx = -sr; dx <= sr; dx++) {
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const y0 = Math.max(0, y - pr), y1 = Math.min(H - 1, y + pr), x0 = Math.max(0, x - pr), x1 = Math.min(W - 1, x + pr);
      let S = 0;
      for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) {
        const b = clampI(yy + dy, H - 1) * W + clampI(xx + dx, W - 1);
        for (let c = 0; c < ch; c++) { const t = data[(yy * W + xx) * ch + c] - data[b * ch + c]; S += t * t; }
      }
      const n = (y1 - y0 + 1) * (x1 - x0 + 1) * ch, w = Math.exp(-(S / n) / (h * h)), p = y * W + x, bp = clampI(y + dy, H - 1) * W + clampI(x + dx, W - 1);
      wsum[p] += w;
      for (let c = 0; c < ch; c++) out[p * ch + c] += w * data[bp * ch + c];
    }
  }
  return Uint8Array.from(out, (v, i) => Math.min(255, Math.max(0, Math.floor(v / wsum[Math.floor(i / ch)] + 0.5))));
}
// Wiener ตามสูตรใน double ด้วย DFT แบบตรง ๆ (separable, O(N²)) — ขอบ BORDER_REFLECT เขียนเอง
const reflectIdx = (i, s) => { const m = ((i % (2 * s)) + 2 * s) % (2 * s); return m < s ? m : 2 * s - 1 - m; };
function dft2(re, im, Hh, Ww, inverse) {
  const sg = inverse ? 1 : -1;
  const r1 = new Float64Array(Hh * Ww), i1 = new Float64Array(Hh * Ww);
  for (let y = 0; y < Hh; y++) for (let k = 0; k < Ww; k++) {
    let sr = 0, si = 0;
    for (let x = 0; x < Ww; x++) { const a = sg * 2 * Math.PI * k * x / Ww, c = Math.cos(a), s = Math.sin(a); sr += re[y * Ww + x] * c - im[y * Ww + x] * s; si += re[y * Ww + x] * s + im[y * Ww + x] * c; }
    r1[y * Ww + k] = sr; i1[y * Ww + k] = si;
  }
  const r2 = new Float64Array(Hh * Ww), i2 = new Float64Array(Hh * Ww);
  for (let x = 0; x < Ww; x++) for (let k = 0; k < Hh; k++) {
    let sr = 0, si = 0;
    for (let y = 0; y < Hh; y++) { const a = sg * 2 * Math.PI * k * y / Hh, c = Math.cos(a), s = Math.sin(a); sr += r1[y * Ww + x] * c - i1[y * Ww + x] * s; si += r1[y * Ww + x] * s + i1[y * Ww + x] * c; }
    r2[k * Ww + x] = sr; i2[k * Ww + x] = si;
  }
  if (inverse) for (let i = 0; i < r2.length; i++) { r2[i] /= Hh * Ww; i2[i] /= Hh * Ww; }
  return [r2, i2];
}
function wienerRef(src, psf, nsr) { // src = Mat 8UC1
  const W = src.cols, H = src.rows, n = psf.n, hc = (n - 1) / 2, d = src.data;
  const Wp = cv.getOptimalDFTSize(W + 2 * n), Hp = cv.getOptimalDFTSize(H + 2 * n);
  const big = new Float64Array(Hp * Wp), ph = new Float64Array(Hp * Wp), zero = new Float64Array(Hp * Wp);
  for (let y = 0; y < Hp; y++) for (let x = 0; x < Wp; x++) big[y * Wp + x] = d[reflectIdx(y - n, H) * W + reflectIdx(x - n, W)];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) ph[((y - hc + Hp) % Hp) * Wp + ((x - hc + Wp) % Wp)] = psf.k[y * n + x];
  const [gr, gi] = dft2(big, zero, Hp, Wp, false), [hr, hi] = dft2(ph, zero, Hp, Wp, false);
  const fr = new Float64Array(Hp * Wp), fi = new Float64Array(Hp * Wp);
  for (let i = 0; i < fr.length; i++) { const den = hr[i] * hr[i] + hi[i] * hi[i] + nsr; fr[i] = (gr[i] * hr[i] + gi[i] * hi[i]) / den; fi[i] = (gi[i] * hr[i] - gr[i] * hi[i]) / den; }
  const [rr] = dft2(fr, fi, Hp, Wp, true);
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) out[y * W + x] = Math.min(255, Math.max(0, Math.floor(rr[(y + n) * Wp + x + n] + 0.5)));
  return out;
}
const reflect101 = (i, s) => { if (s === 1) return 0; const p = 2 * s - 2, m = ((i % p) + p) % p; return m < s ? m : p - m; };
// Lucy-Richardson ตามนิยามใน double (convolution จริงด้วย K, adjoint = correlation) — PSF ใดก็ได้ (รวมที่ไม่สมมาตร)
function lucyRef(src, psf, iters) {
  const W = src.cols, H = src.rows, n = psf.n, hc = (n - 1) / 2, g = Float64Array.from(src.data);
  let est = Float64Array.from(g);
  const at = (a, y, x) => a[reflect101(y, H) * W + reflect101(x, W)];
  for (let it = 0; it < iters; it++) {
    const r = new Float64Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let c = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c += psf.k[i * n + j] * at(est, y + hc - i, x + hc - j);
      r[y * W + x] = c > 1e-6 ? g[y * W + x] / c : 0;
    }
    const ne = new Float64Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let corr = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) corr += psf.k[i * n + j] * at(r, y + i - hc, x + j - hc);
      ne[y * W + x] = est[y * W + x] * corr;
    }
    est = ne;
  }
  return Uint8Array.from(est, (v) => Math.min(255, Math.max(0, Math.floor(v + 0.5))));
}
// Dehaze (refine = false) ตามสูตรเดิมเขียนใหม่ด้วยลูปตรง ๆ + Math.fround ที่จุดเดียวกับแอป
function dehazeRef(m, omega, ps) {
  const W = m.cols, H = m.rows, ch = m.channels(), N = W * H, x = m.data, r = (ps - 1) / 2, f = Math.fround;
  const minWin = (a) => { const o = new Float64Array(N); for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) { let v = Infinity; for (let j = Math.max(0, y - r); j <= Math.min(H - 1, y + r); j++) for (let i = Math.max(0, xx - r); i <= Math.min(W - 1, xx + r); i++) if (a[j * W + i] < v) v = a[j * W + i]; o[y * W + xx] = v; } return o; };
  const d8 = new Float64Array(N); for (let i = 0; i < N; i++) { let v = Infinity; for (let c = 0; c < ch; c++) v = Math.min(v, x[i * ch + c]); d8[i] = v; }
  const dark = minWin(d8);
  const k = Math.max(1, Math.floor(N / 1000));
  const sorted = Array.from(dark).sort((a, b) => b - a), T = sorted[k - 1];
  const A = new Array(ch).fill(0); let cnt = 0;
  for (let i = 0; i < N; i++) if (dark[i] >= T) { cnt++; for (let c = 0; c < ch; c++) A[c] += x[i * ch + c]; }
  for (let c = 0; c < ch; c++) A[c] = Math.max(A[c] / cnt, 1);
  const mn = new Float64Array(N); for (let i = 0; i < N; i++) { let v = Infinity; for (let c = 0; c < ch; c++) v = Math.min(v, x[i * ch + c] / A[c]); mn[i] = f(v); }
  const dn = minWin(mn);
  const tCoarse = Float32Array.from(dn, (v) => 1 - omega * v);
  const out = new Uint8Array(N * ch); let tsum = 0;
  for (let i = 0; i < N; i++) {
    const t = Math.max(f(1 - omega * dn[i]), 0.1); tsum += t;
    for (let c = 0; c < ch; c++) { const v = Math.floor((x[i * ch + c] - A[c]) / t + A[c] + 0.5); out[i * ch + c] = v < 0 ? 0 : v > 255 ? 255 : v; }
  }
  return { out, A, tMean: tsum / N, dark, tCoarse };
}
// ฟังก์ชันวิเคราะห์สำหรับ ECC: A(x, y) = f(x, y) และ B(p) = f(W·p) (ไม่มีการ interpolate ของภาพ จึงรู้ warp จริงเป๊ะ)
const fWave = (x, y) => 128 + 50 * Math.sin(x / 17 + y / 29) + 40 * Math.cos(y / 13 - x / 37) + 25 * Math.sin((x + 2 * y) / 23);
const grayF = (w, h, fn) => gray(w, h, (x, y) => Math.floor(fn(x, y) + 0.5));
function eccPair(w, h, mapper) { return [grayF(w, h, fWave), grayF(w, h, (x, y) => { const [u, v] = mapper(x, y); return fWave(u, v); })]; }
const eucMap = (deg, tx, ty) => { const th = deg * Math.PI / 180, c = Math.cos(th), s = Math.sin(th); return (x, y) => [c * x - s * y + tx, s * x + c * y + ty]; };
// ตัวนับการเรียก findTransformECC (พิสูจน์ว่า guard กันไม่ให้เรียก)
function countingCv(name) { const st = { n: 0 }; const c = new Proxy(cv, { get(t, p) { if (p !== name) return t[p]; return (...a) => { st.n++; return t[p](...a); }; } }); return { cv: c, st }; }
// ค่า k ของ kernel ที่ปักไว้ (วัดจาก cv2.getGaborKernel 5.0.0, float32) — sha256[:16] ของ byte ทั้ง kernel
const GABOR_PINS = [[11, 2.5, 30, 7, 0.6, 0, '352b9614cbf51114'], [9, 1.7, 0, 4, 1, Math.PI / 2, '97d0846a5235ee33'], [21, 4.5, 135, 8, 0.5, 0, '7cd949e7b54faa1d'], [55, 4.5, 0, 8, 0.5, 0, '5148fe92af0b3acd']];

// ===================================================================
// Morphology: reconstructDilate / Regional Max / Min / H-maxima / Extended Maxima
// ===================================================================
test('ip4b-rc1. reconstructDilate == the definition (iterate dilate-3x3 and min(mask) until stable) on noise / plateaus / a constant image / 1x1 / 1xN / Nx1 / smooth waves up to 200x150 (marker = I-1 and I-h)', () => {
  const imgs = [];
  for (const seed of [1, 2, 3]) { const m = noiseGray(17, 13, seed); imgs.push([`noise${seed}`, m]); }
  imgs.push(['plateau', gray(30, 20, (x, y) => ((x >> 3) + (y >> 2)) * 25 & 255)], ['constant', gray(9, 7, () => 77)], ['1x1', gray(1, 1, () => 200)], ['1x12', gray(12, 1, (x) => (x * 37) & 255)], ['12x1', gray(1, 12, (_, y) => (y * 53) & 255)]);
  imgs.push(['wave200', waveGray(200, 150)]);
  imgs.push(['blocks', gray(40, 30, (x, y) => (((x * 7) ^ (y * 13)) & 3) * 60)]);
  try {
    for (const [name, m] of imgs) {
      const W = m.cols, H = m.rows, I = i32(m);
      for (const off of [1, 10, 60]) {
        const J0 = I.map((v) => Math.max(v - off, off === 1 ? -1 : 0)), J = reconstructDilate(Int32Array.from(J0), I, W, H);
        assert.deepEqual(Array.from(J), Array.from(reconRef(J0, I, W, H)), `${name} marker I-${off}`);
      }
    }
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4b-rc2. reconstructDilate: a single seed on a long 1-px spiral must flood the whole spiral (a raster / anti-raster scan alone cannot -- this needs the FIFO queue) and equals the definition; a long snake in a 200x200 image runs in O(N)', () => {
  for (const n of [11, 25, 41]) {
    const I = Int32Array.from(spiralMask(n)), J = new Int32Array(n * n);
    J[0] = 255;
    const R = reconstructDilate(Int32Array.from(J), I, n, n);
    assert.deepEqual(Array.from(R), Array.from(I), `spiral ${n}: whole spiral reached`);
    assert.deepEqual(Array.from(R), Array.from(reconRef(J, I, n, n)));
    assert.ok(I.filter((v) => v).length > n * n / 4, 'the spiral really is long');
  }
  const n = 301, I = Int32Array.from(spiralMask(n)), J = new Int32Array(n * n); J[0] = 255;
  const t0 = performance.now(); const R = reconstructDilate(J, I, n, n); const dt = performance.now() - t0;
  assert.deepEqual(Array.from(R), Array.from(I)); assert.ok(dt < 500, `301x301 spiral took ${dt} ms`);
  // reversed / flipped traversal directions (the seed at the far end)
  const m = 41, Ix = Int32Array.from(spiralMask(m)), Jx = new Int32Array(m * m); let last = 0; for (let i = 0; i < Ix.length; i++) if (Ix[i]) last = i; Jx[last] = 255;
  assert.deepEqual(Array.from(reconstructDilate(Jx, Ix, m, m)), Array.from(Ix));
});

test('ip4b-rc3. reconstructDilate on SEEDED RANDOM multi-level plateau images (few grey levels, random blocks) == the definition, for the h-maxima marker max(I-h, 0) and for arbitrary markers <= I: the queue de-duplication (inq cleared on pop) is pinned here without Python', () => {
  const r = lcg(2024);
  const t0 = performance.now();
  for (let k = 0; k < 120; k++) {
    const W = 6 + Math.floor(r() * 18), H = 5 + Math.floor(r() * 14), levels = 2 + Math.floor(r() * 4), mul = 1 + Math.floor(r() * 60), blk = 1 + Math.floor(r() * 3);
    const lv = new Int32Array(W * H); const rnd = new Int32Array(Math.ceil(W / blk) * Math.ceil(H / blk)).map(() => Math.floor(r() * levels));
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) lv[y * W + x] = rnd[Math.floor(y / blk) * Math.ceil(W / blk) + Math.floor(x / blk)] * mul;
    const h = 1 + Math.floor(r() * 80);
    const m1 = lv.map((v) => Math.max(v - h, 0));
    const m2 = lv.map((v) => (r() < 0.2 ? v : Math.max(0, v - Math.floor(r() * 3 * mul))));
    for (const [name, mk] of [['h-marker', m1], ['random marker', m2]]) assert.deepEqual(Array.from(reconstructDilate(Int32Array.from(mk), lv, W, H)), Array.from(reconRef(mk, lv, W, H)), `case ${k} ${W}x${H} levels ${levels} mul ${mul} ${name}`);
    assert.deepEqual(Array.from(hMaximaImage(lv, W, H, h)), Array.from(hMaxRef(lv, W, H, h)), `hMaximaImage case ${k}`);
  }
  // full-range 0-255 noise: a pixel is raised again AFTER it left the queue — the few-level blocks above never do this, so only these cases pin `inq[p] = 0` on pop without Python
  for (const [W, H] of [[97, 61], [40, 30]]) for (const seed of [11, 12, 13]) {
    const n = lcg(seed), I = Int32Array.from({ length: W * H }, () => Math.floor(n() * 256));
    for (const h of [1, 10, 60]) {
      const mk = I.map((v) => Math.max(v - h, 0));
      assert.deepEqual(Array.from(reconstructDilate(Int32Array.from(mk), I, W, H)), Array.from(reconRef(mk, I, W, H)), `noise ${W}x${H} seed ${seed} h ${h}`);
      assert.deepEqual(Array.from(hMaximaImage(I, W, H, h)), Array.from(hMaxRef(I, W, H, h)), `hMaximaImage noise ${W}x${H} seed ${seed} h ${h}`);
    }
  }
  assert.ok(performance.now() - t0 < 4000, 'fast enough');
});

test('ip4b-rg1. regional_max / regional_min: 3 peaks of different heights -> 3 regions (+info with the region and pixel count); regional_min of an image == regional_max of its inverse; the result is a 0/255 gray image of the same size; the image is NOT cut at 127', () => {
  const m = gray(60, 40, () => 10);
  fillRectM(m, 5, 5, 8, 8, 100); fillRectM(m, 25, 10, 6, 6, 150); fillRectM(m, 45, 25, 4, 4, 200);
  fillRectM(m, 28, 12, 1, 1, 150); // inside a plateau — still one region
  try {
    const ctx = {}; const o = OPS.morphology.apply(cv, m, MORPH('regional_max'), ctx);
    assert.equal(o.type(), cv.CV_8UC1); assert.equal(o.rows, 40); assert.equal(o.cols, 60);
    assert.ok(Array.from(o.data).every((v) => v === 0 || v === 255));
    assert.equal(cv.countNonZero(o), 64 + 36 + 16);
    assert.match(ctx.info, /พบ 3 บริเวณ \(116 พิกเซล\)/);
    // all 3 plateaus are exactly the white pixels (8-connected regions are the plateaus)
    for (const [x, y, w, h] of [[5, 5, 8, 8], [25, 10, 6, 6], [45, 25, 4, 4]]) for (let j = y; j < y + h; j++) for (let i = x; i < x + w; i++) assert.equal(o.data[j * 60 + i], 255);
    o.delete();
    // a dim peak (values < 127, e.g. 40 on 30) is found too: no threshold at 127
    const dim = gray(20, 10, () => 30); fillRectM(dim, 3, 3, 2, 2, 40);
    const od = OPS.morphology.apply(cv, dim, MORPH('regional_max'), {}); assert.equal(cv.countNonZero(od), 4); od.delete(); dim.delete();
    // regional min = regional max of the inverted image
    const inv = new cv.Mat(); cv.bitwise_not(m, inv);
    const c1 = {}, c2 = {}; const a = OPS.morphology.apply(cv, m, MORPH('regional_min'), c1), b = OPS.morphology.apply(cv, inv, MORPH('regional_max'), c2);
    sameBytes(a, b, 'regional_min(I) == regional_max(255 - I)'); assert.match(c1.info, /Regional Minima · พบ 1 บริเวณ/);
    a.delete(); b.delete(); inv.delete();
    // kernel params are ignored: the same output for another shape / ksize / iterations
    const o1 = OPS.morphology.apply(cv, m, MORPH('regional_max', { shape: 'cross', ksize: 15, iterations: 4 }), {}), o2 = OPS.morphology.apply(cv, m, MORPH('regional_max'), {}); sameBytes(o1, o2, 'ignores shape/ksize/iterations'); o1.delete(); o2.delete();
  } finally { m.delete(); }
});

test('ip4b-rg2. regional_max / regional_min / h_maxima / extended_max == the definition-based reference (iterated reconstruction) on noise, waves and plateaus; h_maxima keeps the right heights; a peak lower than h disappears; extended_max = regional max of h_maxima', () => {
  const imgs = [noiseGray(23, 17, 5), noiseGray(23, 17, 6), waveGray(64, 48), gray(30, 20, (x, y) => ((x >> 2) * 31 + (y >> 2) * 17) & 255), gray(9, 9, () => 0)];
  try {
    for (const m of imgs) {
      const W = m.cols, H = m.rows, I = i32(m);
      for (const op of ['regional_max', 'regional_min', 'h_maxima', 'extended_max']) for (const h of [1, 10, 40]) {
        const o = OPS.morphology.apply(cv, m, MORPH(op, { hHeight: h }), {});
        const hm = hMaxRef(I, W, H, h);
        const exp = op === 'regional_max' ? regMaxRef(I, W, H) : op === 'regional_min' ? regMaxRef(I.map((v) => 255 - v), W, H) : op === 'h_maxima' ? Uint8Array.from(hm) : regMaxRef(hm, W, H);
        assert.deepEqual(Array.from(o.data), Array.from(exp), `${op} h${h} on ${W}x${H}`); o.delete();
      }
    }
    // h-maxima by hand: 3 plateaus 100 / 150 / 200 on a flat 10 background (heights above it 90 / 140 / 190), h = 100
    const m = gray(60, 40, () => 10); fillRectM(m, 5, 5, 8, 8, 100); fillRectM(m, 25, 10, 6, 6, 150); fillRectM(m, 45, 25, 4, 4, 200);
    const ctx = {}; const o = OPS.morphology.apply(cv, m, MORPH('h_maxima', { hHeight: 100 }), ctx);
    // imhmax semantics: a peak shorter than h is levelled to the surrounding level (10), the others are lowered by h (and the surround rises to the lowered 150 peak's reach, capped by the original 10)
    assert.equal(o.data[7 * 60 + 7], 10, 'the 100 peak (90 above the background) is levelled to the background'); assert.equal(o.data[12 * 60 + 27], 50, '150 - 100'); assert.equal(o.data[26 * 60 + 46], 100, '200 - 100'); assert.equal(o.data[0], 10);
    assert.match(ctx.info, /กดยอดลง 100 ระดับ · ยอดที่เหลือ 2 บริเวณ/);
    o.delete();
    const c2 = {}; const e = OPS.morphology.apply(cv, m, MORPH('extended_max', { hHeight: 100 }), c2);
    assert.equal(cv.countNonZero(e), 36 + 16); assert.match(c2.info, /Extended Maxima \(h=100\) · พบ 2 บริเวณ \(52 พิกเซล\)/); e.delete();
    const c3 = {}; const e3 = OPS.morphology.apply(cv, m, MORPH('extended_max', { hHeight: 20 }), c3); assert.equal(cv.countNonZero(e3), 64 + 36 + 16); e3.delete();
    m.delete();
    // h clamps: 0 -> 1, 999 -> 255, junk -> 10
    const q = noiseGray(15, 11, 9);
    for (const [given, used] of [[0, 1], [-5, 1], [999, 255], ['junk', 10]]) { const a = OPS.morphology.apply(cv, q, MORPH('h_maxima', { hHeight: given }), {}), b = OPS.morphology.apply(cv, q, MORPH('h_maxima', { hHeight: used }), {}); sameBytes(a, b, `h ${given} -> ${used}`); a.delete(); b.delete(); }
    q.delete();
  } finally { for (const m of imgs) m.delete(); }
});

test('ip4b-rg3. regional operations on a constant image: regional max / min = all white (+ the one-value text, MATLAB does the same); 1x1 works; h_maxima of a constant image 5 with h 10 is 0 and a constant 200 is 190; a big natural-looking image (640x480 waves) matches the definition', () => {
  const flat = gray(12, 8, () => 77), one = gray(1, 1, () => 5);
  try {
    for (const op of ['regional_max', 'regional_min', 'extended_max']) {
      const ctx = {}; const o = OPS.morphology.apply(cv, flat, MORPH(op, { hHeight: 10 }), ctx);
      assert.equal(cv.countNonZero(o), 96, op); o.delete();
      if (op !== 'extended_max') assert.match(ctx.info, /ภาพมีค่าเดียว ทั้งภาพจึงเป็นยอดเดียว/);
      assert.match(ctx.info, /พบ 1 บริเวณ \(96 พิกเซล\)/);
      const o1 = OPS.morphology.apply(cv, one, MORPH(op), {}); assert.equal(o1.data[0], 255); o1.delete();
    }
    const h5 = gray(6, 6, () => 5), h200 = gray(6, 6, () => 200);
    const a = OPS.morphology.apply(cv, h5, MORPH('h_maxima', { hHeight: 10 }), {}), b = OPS.morphology.apply(cv, h200, MORPH('h_maxima', { hHeight: 10 }), {});
    assert.ok(Array.from(a.data).every((v) => v === 0)); assert.ok(Array.from(b.data).every((v) => v === 190)); a.delete(); b.delete(); h5.delete(); h200.delete();
    const big = waveGray(640, 480), I = i32(big), t0 = performance.now(); const o = OPS.morphology.apply(cv, big, MORPH('regional_max'), {}); const dt = performance.now() - t0;
    assert.deepEqual(Array.from(o.data), Array.from(regMaxRef(I, 640, 480))); assert.ok(dt < 1500, `640x480 regional_max ${dt} ms`);
    o.delete(); big.delete();
  } finally { flat.delete(); one.delete(); }
});

// ===================================================================
// Hit-or-Miss
// ===================================================================
test('ip4b-hm1. hit_or_miss: every pattern on an image built to contain EXACTLY N matches (including pixels at the image border -- outside the image counts as "do not care", as OpenCV does) and the exact positions; rotate on / off', () => {
  const W = 30, H = 30;
  // isolated: 3 inner dots + 2 on the border (corner / edge) + a 2-px blob (no match)
  const iso = black(W, H); for (const [x, y] of [[5, 5], [15, 10], [25, 25], [0, 0], [29, 14]]) iso.data[y * W + x] = 255; iso.data[20 * W + 5] = 255; iso.data[20 * W + 6] = 255;
  // hole: white image with single black pixels (inner x3, corner, edge) and a 2-px black blob (no match)
  const hole = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255)); for (const [x, y] of [[5, 5], [15, 10], [25, 25], [0, 0], [29, 14]]) hole.data[y * W + x] = 0; hole.data[20 * W + 5] = 0; hole.data[20 * W + 6] = 0;
  // corner: a white 10x8 box away from the border -> its 4 convex corners; plus a thin horizontal line (no convex corner pattern)
  const box = black(W, H); fillRectM(box, 8, 10, 10, 8);
  // line_end: a horizontal 1-px line and a vertical 1-px line, apart from each other and from the border
  const lines = black(W, H); for (let x = 5; x <= 20; x++) lines.data[8 * W + x] = 255; for (let y = 14; y <= 26; y++) lines.data[y * W + 24] = 255;
  const at = (o) => { const r = []; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (o.data[y * W + x]) r.push(`${x},${y}`); return r.sort(); };
  try {
    const run = (m, pat, rot) => { const ctx = {}; const o = OPS.morphology.apply(cv, m, MORPH('hit_or_miss', { hmPattern: pat, hmRotate: rot }), ctx); return { o, ctx }; };
    for (const rot of [true, false]) {
      let r = run(iso, 'isolated', rot); assert.deepEqual(at(r.o), ['0,0', '15,10', '25,25', '29,14', '5,5'], `isolated rot${rot}`); assert.match(r.ctx.info, new RegExp(`Hit-or-Miss \\(จุดเดี่ยว${rot ? ' · 4 ทิศ' : ''}\\) พบ 5 ตำแหน่ง`)); r.o.delete();
      r = run(hole, 'hole', rot); assert.deepEqual(at(r.o), ['0,0', '15,10', '25,25', '29,14', '5,5'], `hole rot${rot}`); r.o.delete();
    }
    let r = run(box, 'corner', true); assert.deepEqual(at(r.o), ['17,10', '17,17', '8,10', '8,17'], 'corner x4 directions'); assert.match(r.ctx.info, /มุมนูน · 4 ทิศ\) พบ 4 ตำแหน่ง/); r.o.delete();
    r = run(box, 'corner', false); assert.deepEqual(at(r.o), ['17,10'], 'corner top-right only'); assert.match(r.ctx.info, /พบ 1 ตำแหน่ง/); r.o.delete();
    r = run(lines, 'line_end', true); assert.deepEqual(at(r.o), ['20,8', '24,14', '24,26', '5,8'], 'line ends x4 directions'); r.o.delete();
    r = run(lines, 'line_end', false); assert.deepEqual(at(r.o), ['24,14'], 'line end (top) only'); r.o.delete();
    // the default image of a gray (non 0/255) input is cut at 127 first
    const soft = gray(W, H, () => 100); soft.data[5 * W + 5] = 130; r = run(soft, 'isolated', true); assert.deepEqual(at(r.o), ['5,5']); r.o.delete(); soft.delete();
  } finally { for (const m of [iso, hole, box, lines]) m.delete(); }
});

test('ip4b-hm2. hit_or_miss == the definition-based brute force AND == cv.morphologyEx(MORPH_HITMISS) with the same kernels (OR of the rotations), every pattern x rotate on 97x61 / 20x15 random binary images; rotate90cw / hitMissKernels helpers; the pattern table', () => {
  const rnd = [noiseGray(97, 61, 11), noiseGray(20, 15, 12), noiseGray(3, 3, 13), noiseGray(1, 1, 14)];
  try {
    for (const m of rnd) {
      const W = m.cols, H = m.rows, bin = Uint8Array.from(m.data, (v) => (v > 127 ? 255 : 0));
      for (const pat of Object.keys(HITMISS_PATTERNS)) for (const rot of [true, false]) {
        const o = OPS.morphology.apply(cv, m, MORPH('hit_or_miss', { hmPattern: pat, hmRotate: rot }), {});
        let exp = new Uint8Array(W * H), k = HITMISS_PATTERNS[pat].k;
        for (let r = 0; r < (rot ? 4 : 1); r++) { const part = hitMissRef(bin, W, H, k); exp = exp.map((v, i) => v | part[i]); k = rotate90cw(k); }
        assert.deepEqual(Array.from(o.data), Array.from(exp), `${pat} rot${rot} ${W}x${H} vs brute force`);
        // direct OpenCV
        const b = new cv.Mat(); cv.threshold(m, b, 127, 255, cv.THRESH_BINARY); const acc = cv.Mat.zeros(H, W, cv.CV_8UC1);
        for (const flat of hitMissKernels(pat, rot)) { const kk = cv.matFromArray(3, 3, cv.CV_32SC1, flat), t = new cv.Mat(); cv.morphologyEx(b, t, cv.MORPH_HITMISS, kk); cv.bitwise_or(acc, t, acc); t.delete(); kk.delete(); }
        sameBytes(o, acc, `${pat} rot${rot} vs cv.morphologyEx`); acc.delete(); b.delete(); o.delete();
      }
    }
  } finally { for (const m of rnd) m.delete(); }
  // helpers
  assert.deepEqual(rotate90cw([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [[7, 4, 1], [8, 5, 2], [9, 6, 3]]);
  let k = HITMISS_PATTERNS.corner.k; for (let i = 0; i < 4; i++) k = rotate90cw(k); assert.deepEqual(k, HITMISS_PATTERNS.corner.k);
  assert.equal(hitMissKernels('isolated', true).length, 1); assert.equal(hitMissKernels('corner', true).length, 4); assert.equal(hitMissKernels('corner', false).length, 1);
  assert.deepEqual(HITMISS_PATTERNS.isolated.k, [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]); assert.deepEqual(HITMISS_PATTERNS.hole.k, [[1, 1, 1], [1, -1, 1], [1, 1, 1]]);
  assert.deepEqual(HITMISS_PATTERNS.corner.k, [[0, -1, -1], [1, 1, -1], [0, 1, 0]]); assert.deepEqual(HITMISS_PATTERNS.line_end.k, [[-1, -1, -1], [-1, 1, -1], [0, 1, 0]]);
  // weird pattern / rotate values fall back to the defaults (isolated / rotate on)
  const g = noiseGray(30, 20, 15), a = OPS.morphology.apply(cv, g, MORPH('hit_or_miss', { hmPattern: 'constructor', hmRotate: 'x' }), {}), b = OPS.morphology.apply(cv, g, MORPH('hit_or_miss', { hmPattern: 'isolated', hmRotate: true }), {});
  sameBytes(a, b, 'weird pattern'); a.delete(); b.delete(); g.delete();
});

// ===================================================================
// Wiener / Lucy-Richardson
// ===================================================================
test('ip4b-w1. deconv_wiener closes the loop with Custom Kernel: a 200x150 checkerboard blurred by motion 9 px 0 deg (the Custom Kernel default) comes back with at least 30% lower MAE at the default settings; same for a disk PSF; a WRONG angle (90 deg) is clearly worse than the right one; the exact blur size is better than a wrong one', () => {
  const sharp = checkerboard(200, 150, 10);
  try {
    const blurred = blurWith(sharp, 'motion', 9, 0), base = mae(blurred, sharp);
    const ctx = {}; const fixed = OPS.deconv_wiener.apply(cv, blurred, defaultsOf('deconv_wiener'), ctx);
    assert.ok(mae(fixed, sharp) < base * 0.7, `motion: MAE ${mae(fixed, sharp)} vs blurred ${base}`);
    assert.match(ctx.info, /Wiener deconvolution · PSF motion ยาว 9 px มุม 0° \(9x9\) · NSR 0\.01/);
    const wrong = OPS.deconv_wiener.apply(cv, blurred, { psfAngle: 90 }, {}), size3 = OPS.deconv_wiener.apply(cv, blurred, { psfSize: 15 }, {});
    assert.ok(mae(wrong, sharp) > mae(fixed, sharp) * 1.5, `wrong angle ${mae(wrong, sharp)} vs right ${mae(fixed, sharp)}`);
    assert.ok(mae(size3, sharp) > mae(fixed, sharp), 'wrong size is worse');
    for (const m of [fixed, wrong, size3, blurred]) m.delete();
    const dblur = blurWith(sharp, 'disk', 6), dbase = mae(dblur, sharp), dfix = OPS.deconv_wiener.apply(cv, dblur, { psf: 'disk', psfSize: 6 }, {});
    assert.ok(mae(dfix, sharp) < dbase * 0.7, `disk: MAE ${mae(dfix, sharp)} vs blurred ${dbase}`); dfix.delete(); dblur.delete();
  } finally { sharp.delete(); }
});

test('ip4b-w2. deconv_wiener == a double-precision DFT reference of F = G conj(H) / (|H|^2 + NSR) (own reflect padding, psf2otf shift, crop) within 1 level for the real PSFs AND for an asymmetric PSF through the core (a symmetric PSF has a real OTF, so conj / shift errors hide there); NSR matters; colour == channel by channel', () => {
  const m = noiseGray(11, 9, 21);
  const asym = { n: 3, k: [0, 0.1, 0, 0.5, 0.3, 0, 0, 0.1, 0] }; // ไม่สมมาตรรอบจุดกลาง (sum = 1)
  try {
    for (const [name, psf] of [['motion 5 / 0 deg', psfKernel('motion', 5, 0)], ['motion 5 / 30 deg', psfKernel('motion', 5, 30)], ['disk 2', psfKernel('disk', 2, 0)], ['asymmetric', asym]]) {
      for (const nsr of [0.001, 0.01, 0.2]) {
        const o = wienerDeconvolveForTest(m, psf, nsr), exp = wienerRef(m, psf, nsr);
        let mx = 0; for (let i = 0; i < exp.length; i++) mx = Math.max(mx, Math.abs(o.data[i] - exp[i]));
        assert.ok(mx <= 1, `${name} nsr ${nsr}: max diff ${mx}`); o.delete();
      }
    }
    // through the op (default PSF, nsr 0.05)
    const op = OPS.deconv_wiener.apply(cv, m, { psfSize: 5, nsr: 0.05 }, {}), ref = wienerRef(m, psfKernel('motion', 5, 0), 0.05);
    let mx = 0; for (let i = 0; i < ref.length; i++) mx = Math.max(mx, Math.abs(op.data[i] - ref[i])); assert.ok(mx <= 1, `op vs reference ${mx}`); op.delete();
    // NSR changes the answer (the term is really in the denominator)
    const a = OPS.deconv_wiener.apply(cv, m, { nsr: 0.001 }, {}), b = OPS.deconv_wiener.apply(cv, m, { nsr: 0.3 }, {}); assert.ok(mae(a, b) > 1, 'NSR has an effect'); a.delete(); b.delete();
    // colour = channel by channel
    const c = noiseRGB(13, 10, 22), vec = new cv.MatVector(); cv.split(c, vec);
    const col = OPS.deconv_wiener.apply(cv, c, { psfSize: 5, psf: 'disk' }, {});
    for (let ch = 0; ch < 3; ch++) { const one = vec.get(ch), r = OPS.deconv_wiener.apply(cv, one, { psfSize: 5, psf: 'disk' }, {}); for (let i = 0; i < r.data.length; i++) assert.equal(col.data[i * 3 + ch], r.data[i], `channel ${ch} px ${i}`); r.delete(); one.delete(); }
    vec.delete(); col.delete(); c.delete();
  } finally { m.delete(); }
});
function wienerDeconvolveForTest(m, psf, nsr) { return wienerDeconvolve(cv, m, psf, nsr); }

test('ip4b-w3. deconv_wiener edge cases: NSR 0 never throws / never NaN (the output stays 0..255); 1x1 and 2x3 images keep their size; a flat image stays flat (+-1); weird / out-of-range params clamp; RGBA gives 3 channels; 0x0 -> clone + "ภาพว่าง"; src is not modified', () => {
  const g = noiseGray(40, 30, 3), flat = gray(30, 20, () => 120), one = gray(1, 1, () => 99), two = noiseGray(2, 3, 5);
  try {
    const o = OPS.deconv_wiener.apply(cv, g, { nsr: 0 }, {}); assert.equal(o.rows, 30); assert.ok(Array.from(o.data).every((v) => v >= 0 && v <= 255)); o.delete();
    const ctx0 = {}; const o0 = OPS.deconv_wiener.apply(cv, g, { nsr: 0 }, ctx0); assert.match(ctx0.info, /NSR = 0 คือหารตรง \(inverse filter\)/); o0.delete();
    const ctx1 = {}; const o1 = OPS.deconv_wiener.apply(cv, g, {}, ctx1); assert.ok(!/NSR = 0 คือ/.test(ctx1.info)); o1.delete();
    for (const m of [one, two]) { const r = OPS.deconv_wiener.apply(cv, m, {}, {}); assert.equal(r.rows, m.rows); assert.equal(r.cols, m.cols); r.delete(); }
    const f = OPS.deconv_wiener.apply(cv, flat, {}, {}); assert.ok(Array.from(f.data).every((v) => Math.abs(v - 120) <= 1), 'flat stays flat'); f.delete();
    const before = bytes(g);
    const w = OPS.deconv_wiener.apply(cv, g, { psf: 'constructor', psfSize: 'x', psfAngle: -9, nsr: 'q' }, {}), d = OPS.deconv_wiener.apply(cv, g, {}, {}); sameBytes(w, d, 'junk params = defaults'); w.delete(); d.delete();
    const big = OPS.deconv_wiener.apply(cv, g, { psfSize: 999, psfAngle: 999, nsr: 9 }, {}), ref = OPS.deconv_wiener.apply(cv, g, { psfSize: 50, psfAngle: 179, nsr: 0.5 }, {}); sameBytes(big, ref, 'clamps'); big.delete(); ref.delete();
    assert.deepEqual(bytes(g), before, 'src untouched');
    const a = toRGBA(noiseRGB(20, 15, 4)), ra = OPS.deconv_wiener.apply(cv, a, {}, {}); assert.equal(ra.channels(), 3); ra.delete(); a.delete();
    const e = new cv.Mat(), ce = {}; const re = OPS.deconv_wiener.apply(cv, e, {}, ce); assert.equal(re.rows, 0); assert.equal(ce.info, 'ภาพว่าง'); re.delete(); e.delete();
  } finally { for (const m of [g, flat, one, two]) m.delete(); }
});

test('ip4b-l1. deconv_lucy: the error falls with the rounds 1 -> 5 -> 10 on a noise-free blurred image (and 10 -> 30 keeps falling); iterations clamp 0 -> 1 and 99 -> 50; the info says the rounds and the colour note; weird PSF params = defaults', () => {
  const sharp = checkerboard(120, 90, 10);
  try {
    const blurred = blurWith(sharp, 'motion', 9, 0);
    const e = [0, 1, 5, 10, 30].map((it) => { if (it === 0) return mae(blurred, sharp); const o = OPS.deconv_lucy.apply(cv, blurred, { iterations: it }, {}); const v = mae(o, sharp); o.delete(); return v; });
    for (let i = 1; i < e.length; i++) assert.ok(e[i] < e[i - 1], `MAE must fall: ${e.join(' > ')}`);
    assert.ok(e[3] < e[0] * 0.7, '10 rounds at least 30% better than the blur');
    const c0 = {}, a = OPS.deconv_lucy.apply(cv, blurred, { iterations: 0 }, c0), b = OPS.deconv_lucy.apply(cv, blurred, { iterations: 1 }, {}); sameBytes(a, b, 'iterations 0 -> 1'); assert.match(c0.info, /Lucy-Richardson 1 รอบ/); a.delete(); b.delete();
    const c99 = {}, a2 = OPS.deconv_lucy.apply(cv, blurred, { iterations: 99 }, c99), b2 = OPS.deconv_lucy.apply(cv, blurred, { iterations: LUCY_MAX_ITERS }, {}); sameBytes(a2, b2, 'iterations 99 -> 50'); assert.match(c99.info, /Lucy-Richardson 50 รอบ/); a2.delete(); b2.delete();
    const cc = {}, col = noiseRGB(20, 15, 3), oc = OPS.deconv_lucy.apply(cv, col, {}, cc); assert.match(cc.info, /ภาพสีใช้เวลาราว 3 เท่าของภาพเทา/); assert.match(cc.info, /PSF motion ยาว 9 px มุม 0° \(9x9\)/); oc.delete(); col.delete();
    const g = noiseGray(20, 15, 3), w = OPS.deconv_lucy.apply(cv, g, { psf: 'constructor', psfSize: 'x', iterations: 'q' }, {}), d = OPS.deconv_lucy.apply(cv, g, {}, {}); sameBytes(w, d, 'junk = defaults'); w.delete(); d.delete(); g.delete();
    blurred.delete();
  } finally { sharp.delete(); }
});

test('ip4b-l2. deconv_lucy == the definition in double precision (true convolution with K, correlation with K as the adjoint, reflect-101 borders) within 1 level -- with the real PSFs and with an ASYMMETRIC PSF through the core (the flip matters only there); an unflipped kernel is measurably different on the asymmetric PSF', () => {
  const m = noiseGray(14, 11, 31), asym = { n: 3, k: [0, 0.1, 0, 0.5, 0.3, 0, 0, 0.1, 0] };
  try {
    for (const [name, psf] of [['motion 5', psfKernel('motion', 5, 20)], ['disk 3', psfKernel('disk', 3, 0)], ['asymmetric', asym]]) {
      for (const it of [1, 3]) {
        const o = lucyDeconvolve(cv, m, psf, it), exp = lucyRef(m, psf, it);
        let mx = 0; for (let i = 0; i < exp.length; i++) mx = Math.max(mx, Math.abs(o.data[i] - exp[i]));
        assert.ok(mx <= 1, `${name} x${it}: max diff ${mx}`); o.delete();
      }
    }
    // the flip: an (incorrect) reference that skips it must differ from the app on the asymmetric PSF
    const o = lucyDeconvolve(cv, m, asym, 3), wrongPsf = { n: 3, k: [0, 0.1, 0, 0, 0.3, 0.5, 0, 0.1, 0] }; // PSF กลับด้านซ้าย-ขวา
    const wrong = lucyRef(m, wrongPsf, 3); let diff = 0; for (let i = 0; i < wrong.length; i++) diff += Math.abs(o.data[i] - wrong[i]); assert.ok(diff / wrong.length > 1, `the flipped PSF gives a clearly different result (${diff / wrong.length})`); o.delete();
    // through the op (3 channels)
    const c = noiseRGB(12, 10, 32), oc = OPS.deconv_lucy.apply(cv, c, { psf: 'disk', psfSize: 3, iterations: 2 }, {}), vec = new cv.MatVector(); cv.split(c, vec);
    for (let ch = 0; ch < 3; ch++) { const one = vec.get(ch), exp = lucyRef(one, psfKernel('disk', 3, 0), 2); let mx = 0; for (let i = 0; i < exp.length; i++) mx = Math.max(mx, Math.abs(oc.data[i * 3 + ch] - exp[i])); assert.ok(mx <= 1, `channel ${ch}: ${mx}`); one.delete(); }
    vec.delete(); oc.delete(); c.delete();
  } finally { m.delete(); }
});

test('ip4b-l3. deconv_lucy edge cases: a flat image stays flat; 1x1 / 2x3 keep their size; RGBA -> 3 channels; 0x0 -> clone + "ภาพว่าง"; src not modified; a black image stays black (no 0/0); output always 0..255', () => {
  const flat = gray(30, 20, () => 120), one = gray(1, 1, () => 99), two = noiseGray(2, 3, 5), blk = gray(20, 10, () => 0), g = noiseGray(25, 20, 8);
  try {
    const f = OPS.deconv_lucy.apply(cv, flat, {}, {}); assert.ok(Array.from(f.data).every((v) => Math.abs(v - 120) <= 1)); f.delete();
    for (const m of [one, two]) { const r = OPS.deconv_lucy.apply(cv, m, {}, {}); assert.equal(r.rows, m.rows); assert.equal(r.cols, m.cols); r.delete(); }
    const b = OPS.deconv_lucy.apply(cv, blk, {}, {}); assert.ok(Array.from(b.data).every((v) => v === 0)); b.delete();
    const before = bytes(g); const o = OPS.deconv_lucy.apply(cv, g, { iterations: 20 }, {}); assert.ok(Array.from(o.data).every((v) => v >= 0 && v <= 255)); o.delete(); assert.deepEqual(bytes(g), before);
    const a = toRGBA(noiseRGB(20, 15, 4)), ra = OPS.deconv_lucy.apply(cv, a, {}, {}); assert.equal(ra.channels(), 3); ra.delete(); a.delete();
    const e = new cv.Mat(), ce = {}; const re = OPS.deconv_lucy.apply(cv, e, {}, ce); assert.equal(re.rows, 0); assert.equal(ce.info, 'ภาพว่าง'); re.delete(); e.delete();
  } finally { for (const m of [flat, one, two, blk, g]) m.delete(); }
});

// ===================================================================
// ECC
// ===================================================================
test('ip4b-ecc1. ecc_register recovers a KNOWN euclidean warp (A = f(x, y), B(p) = f(W p) with W = rotate 2.9 deg + (5.5, -3.25)): dx / dy within 0.1 px, angle within 0.05 deg, scale 1, rho ~ 1, the aligned A matches B (mean |diff| < 1 in the interior), and the info text carries the same numbers; the sign convention (A moves to B) is pinned', () => {
  const [A, B] = eccPair(320, 240, eucMap(2.9, 5.5, -3.25));
  try {
    const ctx = { src2: B }; const o = OPS.ecc_register.apply(cv, A, {}, ctx), v = scalarMap(ctx);
    assert.deepEqual(ctx.values.fields.map((f) => f.key), ECC_KEYS);
    assert.equal(v.ecc_ok, 1); near(v.ecc_dx, 5.5, 0.1, 'dx'); near(v.ecc_dy, -3.25, 0.1, 'dy'); near(v.ecc_angle_deg, 2.9, 0.05, 'angle'); near(v.ecc_scale, 1, 1e-3, 'scale'); assert.ok(v.ecc_rho > 0.99 && v.ecc_rho <= 1, `rho ${v.ecc_rho}`);
    assert.equal(ctx.info.startsWith(`จัดภาพแบบเลื่อน + หมุน · A เลื่อน (${fixed2(v.ecc_dx)}, ${fixed2(v.ecc_dy)}) px · หมุน ${fixed2(v.ecc_angle_deg)}° · ขยาย ×${fixed3(v.ecc_scale)} · ρ = ${fixed3(v.ecc_rho)}`), true, ctx.info);
    assert.ok(!/จับคู่บนภาพย่อ/.test(ctx.info) && !/ปรับขนาด B/.test(ctx.info));
    assert.equal(o.rows, 240); assert.equal(o.cols, 320); assert.equal(o.channels(), 1);
    let s = 0, n = 0; for (let y = 20; y < 220; y++) for (let x = 20; x < 300; x++) { s += Math.abs(o.data[y * 320 + x] - B.data[y * 320 + x]); n++; }
    assert.ok(s / n < 1, `aligned vs B mean abs diff ${s / n}`);
    let s0 = 0; for (let y = 20; y < 220; y++) for (let x = 20; x < 300; x++) s0 += Math.abs(A.data[y * 320 + x] - B.data[y * 320 + x]);
    assert.ok(s0 / n > 8 * (s / n), 'the unaligned pair differs much more (the test is not trivial)');
    o.delete();
  } finally { A.delete(); B.delete(); }
});

test('ip4b-ecc2. ecc_register: translation (pure shift), affine and homography recover their known warps; maxSide 160 gives close values to 320 (the warp is scaled back by 1/s), aligned images agree with B at both sizes (mean |diff| < 1; a homography whose perspective terms were scaled the wrong way would not)', () => {
  const w = 320, h = 240;
  const cases = [
    ['translation', (x, y) => [x + 5.5, y - 3.25], { dx: 5.5, dy: -3.25, ang: 0, sc: 1 }],
    ['affine', eucMap(2.9, 5.5, -3.25), { dx: 5.5, dy: -3.25, ang: 2.9, sc: 1 }],
    ['homography', (x, y) => { const d = 4e-4 * x - 3e-4 * y + 1; return [(x + 3) / d, (y - 2) / d]; }, { dx: 3, dy: -2, ang: 0, sc: 1 }],
  ];
  for (const [motion, mapper, exp] of cases) {
    const [A, B] = eccPair(w, h, mapper);
    try {
      for (const maxSide of [320, 160]) {
        const ctx = { src2: B }, p = { motion, maxSide, iterations: 100 }; const o = OPS.ecc_register.apply(cv, A, p, ctx), v = scalarMap(ctx);
        assert.equal(v.ecc_ok, 1, `${motion}@${maxSide}`);
        near(v.ecc_dx, exp.dx, 0.15, `${motion}@${maxSide} dx`); near(v.ecc_dy, exp.dy, 0.15, `${motion}@${maxSide} dy`); near(v.ecc_angle_deg, exp.ang, 0.06, `${motion}@${maxSide} angle`); near(v.ecc_scale, exp.sc, 2e-3, `${motion}@${maxSide} scale`);
        if (maxSide === 160) assert.match(ctx.info, /จับคู่บนภาพย่อ 160×120/); else assert.ok(!/ภาพย่อ/.test(ctx.info));
        let s = 0, n = 0; for (let y = 20; y < h - 20; y++) for (let x = 20; x < w - 20; x++) { s += Math.abs(o.data[y * w + x] - B.data[y * w + x]); n++; }
        assert.ok(s / n < 1, `${motion}@${maxSide}: aligned vs B ${s / n}`);
        o.delete();
      }
    } finally { A.delete(); B.delete(); }
  }
});

test('ip4b-ecc3. ecc_register guards: a flat A, a flat B, an image under 16 px after shrinking (10x10; a 400x10 strip shrunk to 64 px wide) NEVER reach findTransformECC (call counter = 0: the call would throw a C++ exception that leaks ~760 B), give A back unchanged + ecc_ok 0 + the reason; a textured pair reaches it exactly once', () => {
  const wv = waveGray(40, 30), flat = gray(40, 30, () => 90), tiny = waveGray(10, 10), strip = waveGray(400, 10), nearFlat = gray(40, 30, (x) => (x % 2 ? 90 : 91));
  try {
    const cases = [
      ['flat A', flat, wv, {}, /ภาพเรียบเกินไป ไม่มีลวดลายให้จับคู่/], ['flat B', wv, flat, {}, /ภาพเรียบเกินไป/], ['10x10', tiny, tiny, {}, /ภาพเล็กเกินไปสำหรับจัดภาพ \(ต้องกว้าง\/สูงอย่างน้อย 16 px หลังย่อ\)/],
      ['strip 400x10 at maxSide 64 (6x1 after shrinking)', strip, strip, { maxSide: 64 }, /ภาพเล็กเกินไป/], ['near-flat (std 0.5 < 1)', nearFlat, wv, {}, /ภาพเรียบเกินไป/],
    ];
    for (const [label, A, B, p, re] of cases) {
      const { cv: cc, st } = countingCv('findTransformECC'), ctx = { src2: B };
      const o = OPS.ecc_register.apply(cc, A, p, ctx);
      assert.equal(st.n, 0, `${label}: findTransformECC must not be called`);
      sameBytes(o, A, `${label}: A returned unchanged`); assert.notEqual(o, A); assert.match(ctx.info, re, label);
      const v = scalarMap(ctx); assert.equal(v.ecc_ok, 0); for (const k of ECC_KEYS.slice(1)) assert.equal(v[k], null, `${label} ${k}`);
      o.delete();
    }
    const wb = warpedBy(wv, [1, 0, 1.5, 0, 1, -1]), { cv: c2, st: s2 } = countingCv('findTransformECC'), ctx2 = { src2: wb }, o2 = OPS.ecc_register.apply(c2, wv, {}, ctx2);
    assert.equal(s2.n, 1, 'a textured pair calls it exactly once'); assert.equal(scalarMap(ctx2).ecc_ok, 1); o2.delete(); wb.delete();
    // 1x1 pairs: guard < 16
    const one = gray(1, 1, () => 5), { cv: c3, st: s3 } = countingCv('findTransformECC'), o3 = OPS.ecc_register.apply(c3, one, {}, { src2: one }); assert.equal(s3.n, 0); o3.delete(); one.delete();
  } finally { for (const m of [wv, flat, tiny, strip, nearFlat]) m.delete(); }
});

test('ip4b-ecc4. ecc_register non-convergence (two unrelated noise images that pass the guards): no throw, A returned unchanged, ecc_ok 0, rho / dx / dy / angle / scale null, and the "ไม่ลู่เข้า" text; ecc_ok = 0 -> the diff / overlay views are not built (the output is A)', () => {
  const a = noiseGray(40, 30, 5), b = noiseGray(40, 30, 6);
  try {
    for (const view of ['aligned', 'overlay', 'diff']) {
      const ctx = { src2: b }, o = OPS.ecc_register.apply(cv, a, { iterations: 10, view }, ctx);
      sameBytes(o, a, `${view}: A returned unchanged`); assert.match(ctx.info, /จัดภาพไม่สำเร็จ \(ไม่ลู่เข้า\) — ลองเพิ่มจำนวนรอบ ใช้แบบเลื่อนอย่างเดียว หรือให้สองภาพเหลื่อมกันน้อยลง · ถ้าล้มเหลวซ้ำ ๆ อย่าปล่อยให้รันต่อเนื่องบนกล้องสด \(หน่วยความจำจะรั่วทีละน้อยทุกเฟรม\) — ใช้กับภาพนิ่งแทน/);
      const v = scalarMap(ctx); assert.equal(v.ecc_ok, 0); for (const k of ECC_KEYS.slice(1)) assert.equal(v[k], null);
      o.delete();
    }
  } finally { a.delete(); b.delete(); }
});

test('ip4b-ecc5. ecc_register inputs: no B / B 0x0 -> passthrough text + the 6 value keys (ecc_ok 0, the rest null); A 0x0 -> clone + "ภาพว่าง"; a B of another size is resized with RESIZED_B_INFO appended; gray / rgb / rgba mixes work and the aligned image keeps A\'s channel count; src / src2 are never modified; weird params fall back (motion -> euclidean, iterations / maxSide clamp)', () => {
  const [A, B] = eccPair(120, 90, eucMap(1.5, 3.25, -2.5)), Ac = rgbFromGray(A), Aa = toRGBA(Ac), Bc = rgbFromGray(B), empty = new cv.Mat(), Bsmall = resizeTo(B, 60, 45);
  function rgbFromGray(g) { const o = new cv.Mat(); cv.cvtColor(g, o, cv.COLOR_GRAY2RGB); return o; }
  try {
    for (const [label, B2, re] of [['no B', undefined, /ยังไม่ได้ต่อสาย 'เข้า B'/], ['B 0x0', empty, /ว่าง \(0×0 px\)/]]) {
      const ctx = B2 === undefined ? {} : { src2: B2 }, o = OPS.ecc_register.apply(cv, A, {}, ctx); sameBytes(o, A, label); assert.match(ctx.info, re);
      assert.deepEqual(ctx.values.fields.map((f) => f.key), ECC_KEYS); assert.equal(scalarMap(ctx).ecc_ok, 0); for (const k of ECC_KEYS.slice(1)) assert.equal(scalarMap(ctx)[k], null); o.delete();
    }
    { const ctx = { src2: B }, o = OPS.ecc_register.apply(cv, empty, {}, ctx); assert.equal(o.rows, 0); assert.equal(ctx.info, 'ภาพว่าง'); assert.deepEqual(ctx.values.fields.map((f) => f.key), ECC_KEYS); assert.equal(scalarMap(ctx).ecc_ok, 0); o.delete(); }
    // B of another size -> resized (bilinear) and says so
    { const ctx = { src2: Bsmall }, o = OPS.ecc_register.apply(cv, A, {}, ctx); assert.ok(ctx.info.endsWith(' (ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)'), ctx.info); assert.equal(Bsmall.cols, 60, 'src2 untouched'); o.delete(); }
    // channel mixes
    for (const [label, a, b, ch] of [['gray/gray', A, B, 1], ['rgb/gray', Ac, B, 3], ['gray/rgb', A, Bc, 1], ['rgba/rgb', Aa, Bc, 4], ['rgba/gray', Aa, B, 4]]) {
      const ctx = { src2: b }, ba = bytes(a), bb = bytes(b), o = OPS.ecc_register.apply(cv, a, {}, ctx);
      assert.equal(o.channels(), ch, label); assert.equal(scalarMap(ctx).ecc_ok, 1, `${label} ecc_ok`); assert.deepEqual(bytes(a), ba); assert.deepEqual(bytes(b), bb); o.delete();
    }
    // weird params
    const ctxW = { src2: B }, w = OPS.ecc_register.apply(cv, A, { motion: 'constructor', view: 'constructor', iterations: 'x', maxSide: 'x' }, ctxW), ctxD = { src2: B }, d = OPS.ecc_register.apply(cv, A, {}, ctxD);
    sameBytes(w, d, 'weird = defaults'); assert.equal(ctxW.info, ctxD.info); w.delete(); d.delete();
    const c1 = { src2: B }, big = OPS.ecc_register.apply(cv, A, { maxSide: 99999, iterations: 9999 }, c1), c2 = { src2: B }, ref = OPS.ecc_register.apply(cv, A, { maxSide: 1024, iterations: 300 }, c2); sameBytes(big, ref, 'clamps'); big.delete(); ref.delete();
    const c3 = { src2: B }, lo = OPS.ecc_register.apply(cv, A, { maxSide: -5, iterations: -5 }, c3); assert.match(c3.info, /จับคู่บนภาพย่อ 64×48/); lo.delete();
  } finally { for (const m of [A, B, Ac, Aa, Bc, empty, Bsmall]) m.delete(); }
});

test('ip4b-ecc6. ecc_register views: aligned = A warped (A\'s channels); overlay = [gray(aligned), gray(B), gray(aligned)] as RGB; diff = |gray(aligned) - gray(B)| as 1 channel; on an already-aligned pair (A == B) the overlay is gray everywhere (R = G = B) and the diff is ~0', () => {
  const [A, B] = eccPair(160, 120, eucMap(1.2, 2.5, -1.75)), Ac = new cv.Mat();
  cv.cvtColor(A, Ac, cv.COLOR_GRAY2RGB);
  try {
    const run = (view, a = A, b = B) => { const ctx = { src2: b }; const o = OPS.ecc_register.apply(cv, a, { view }, ctx); return { o, ctx }; };
    const al = run('aligned'), ov = run('overlay'), df = run('diff');
    assert.equal(ov.o.channels(), 3); assert.equal(df.o.channels(), 1);
    for (let i = 0; i < 160 * 120; i++) {
      assert.equal(ov.o.data[3 * i], al.o.data[i], `R = aligned @${i}`); assert.equal(ov.o.data[3 * i + 1], B.data[i], `G = B @${i}`); assert.equal(ov.o.data[3 * i + 2], al.o.data[i], `B = aligned @${i}`);
      assert.equal(df.o.data[i], Math.abs(al.o.data[i] - B.data[i]), `diff @${i}`);
    }
    const rgbA = run('overlay', Ac, B); assert.equal(rgbA.o.channels(), 3); const alC = run('aligned', Ac, B); assert.equal(alC.o.channels(), 3);
    for (let i = 0; i < 160 * 120; i++) { const gi = Math.floor((299 * alC.o.data[3 * i] + 587 * alC.o.data[3 * i + 1] + 114 * alC.o.data[3 * i + 2] + 500) / 1000); assert.ok(Math.abs(rgbA.o.data[3 * i] - gi) <= 1, 'overlay R = gray of the aligned colour A'); }
    for (const r of [al, ov, df, rgbA, alC]) r.o.delete();
    // an aligned pair: A vs A
    const same = run('overlay', A, A), diffSame = run('diff', A, A);
    for (let i = 0; i < 160 * 120; i++) { assert.equal(same.o.data[3 * i], same.o.data[3 * i + 1]); assert.equal(same.o.data[3 * i + 1], same.o.data[3 * i + 2]); }
    assert.ok(Math.max(...diffSame.o.data) <= 2, 'diff of an aligned pair ~ 0'); assert.equal(scalarMap(same.ctx).ecc_ok, 1); near(scalarMap(same.ctx).ecc_dx, 0, 0.02, 'dx of an aligned pair');
    same.o.delete(); diffSame.o.delete();
  } finally { A.delete(); B.delete(); Ac.delete(); }
});

// ===================================================================
// Gabor
// ===================================================================
test('ip4b-gb1. gaborKernel == cv2.getGaborKernel (float32, every value): the byte hashes of 4 kernels measured from cv2 5.0.0 are pinned; sigma / half formulas; k <= 99 at lambda 64, gamma 0.2, bandwidth 0.5; the op reports sigma and the kernel size', () => {
  for (const [k, s, t, l, g, p, hash] of GABOR_PINS) {
    const ke = gaborKernel(k, s, t, l, g, p);
    assert.equal(crypto.createHash('sha256').update(Buffer.from(ke.buffer)).digest('hex').slice(0, 16), hash, `kernel ${k} sigma ${s} theta ${t}`);
  }
  near(gaborSigma(8, 1), 8 / Math.PI * Math.sqrt(Math.LN2 / 2) * 3, 1e-12, 'sigma(8, 1)'); near(gaborSigma(8, 1), 0.5622 * 8, 0.01, 'b = 1 -> 0.5622 lambda');
  assert.equal(gaborHalf(gaborSigma(8, 1), 0.5), 27); assert.equal(gaborHalf(gaborSigma(64, 0.5), 0.2), 49);
  const ctx = {}; const g = noiseGray(30, 30, 1), o = OPS.gabor_filter.apply(cv, g, { wavelength: 64, aspect: 0.2, bandwidth: 0.5 }, ctx);
  assert.match(ctx.info, /kernel 99×99/); assert.match(ctx.info, /Gabor λ=64 px θ=0° \(σ=69\.9\d? px/); o.delete();
  const c2 = {}, o2 = OPS.gabor_filter.apply(cv, g, {}, c2); assert.ok(c2.info.startsWith('Gabor λ=8 px θ=0° (σ=4.50 px, kernel 55×55) · สว่าง = มีลายที่ความถี่และทิศนี้ · ปรับให้ค่าสูงสุดเป็น 255 ทุกภาพ (ค่าจริงดูที่แผงตัวเลข)'), c2.info); o2.delete(); g.delete();
  // the kernel's centre / symmetry (cos phase 0: even kernel is point-symmetric; psi = pi/2 is antisymmetric)
  const ke = gaborKernel(15, 3, 20, 6, 0.7, 0), ko = gaborKernel(15, 3, 20, 6, 0.7, Math.PI / 2);
  for (let i = 0; i < ke.length; i++) { assert.equal(ke[i], ke[ke.length - 1 - i]); near(ko[i], -ko[ko.length - 1 - i], 1e-7, 'odd kernel antisymmetric'); }
  assert.equal(ke[7 * 15 + 7], 1);
});

test('ip4b-gb2. gabor_filter: vertical stripes of period 8 respond at least 5x stronger at theta 0 (lambda 8) than at theta 90; a wrong wavelength responds less; "real" shows the phase (shift by half a period flips the sign) while "magnitude" does not care; the response mean / max are in the values (d: 1)', () => {
  const stripes = (shift, period = 8) => gray(96, 96, (x) => Math.floor(128 + 60 * Math.cos(2 * Math.PI * (x - shift) / period) + 0.5));
  const s0 = stripes(0), s4 = stripes(4), s16 = stripes(0, 16);
  try {
    const run = (m, p) => { const ctx = {}; const o = OPS.gabor_filter.apply(cv, m, p, ctx); return { o, v: scalarMap(ctx), ctx }; };
    const a = run(s0, { wavelength: 8, orientation: 0 }), b = run(s0, { wavelength: 8, orientation: 90 }), c = run(s16, { wavelength: 8, orientation: 0 });
    assert.deepEqual(a.ctx.values.fields.map((f) => f.key), GABOR_KEYS);
    assert.ok(a.v.gabor_max >= 5 * b.v.gabor_max, `theta 0 max ${a.v.gabor_max} vs theta 90 ${b.v.gabor_max}`); assert.ok(a.v.gabor_mean >= 5 * b.v.gabor_mean, 'mean too');
    assert.ok(a.v.gabor_max > 1.3 * c.v.gabor_max, `lambda 8 on period-8 stripes ${a.v.gabor_max} vs on period-16 stripes ${c.v.gabor_max}`);
    assert.equal(Math.max(...a.o.data), 255); assert.equal(a.o.channels(), 1);
    // magnitude ignores the phase (interior only: away from the reflect borders), real flips sign when the stripes shift by half a period
    const m4 = run(s4, { wavelength: 8, orientation: 0 }); let md = 0; for (let y = 30; y < 66; y++) for (let x = 30; x < 66; x++) md = Math.max(md, Math.abs(a.o.data[y * 96 + x] - m4.o.data[y * 96 + x]));
    assert.ok(md <= 8, `magnitude phase-invariant in the interior (max ${md})`);
    const r0 = run(s0, { wavelength: 8, orientation: 0, output: 'real' }), r4 = run(s4, { wavelength: 8, orientation: 0, output: 'real' });
    let flip = 0, n = 0; for (let y = 30; y < 66; y++) for (let x = 30; x < 66; x++) { const u = r0.o.data[y * 96 + x] - 128, w = r4.o.data[y * 96 + x] - 128; if (Math.abs(u) > 40) { n++; if (u * w < 0) flip++; } }
    assert.ok(n > 100 && flip / n > 0.95, `real flips with the half-period shift (${flip}/${n})`);
    assert.ok(Math.min(...r0.o.data) < 40 && Math.max(...r0.o.data) > 215, 'real spans both signs');
    for (const r of [a, b, c, m4, r0, r4]) r.o.delete();
  } finally { s0.delete(); s4.delete(); s16.delete(); }
});

test('ip4b-gb3. gabor_filter on a flat image: magnitude all 0 / real all 128 / mean = max = 0 + the "no texture" text (the float32 residue of the mean-removed kernel is NOT amplified to white -- measured up to 3e-3 at k = 99; a flat image is detected from its pixels, min == max); a tiny +-1 level texture IS found; RGB goes through gray; weird params clamp', () => {
  const flat = gray(64, 48, () => 100), faint = gray(64, 48, (x) => (Math.floor(x / 4) % 2 ? 101 : 99)), c = noiseRGB(30, 20, 3), g = noiseGray(30, 20, 3);
  try {
    for (const prm of [{}, { wavelength: 64, aspect: 0.2, bandwidth: 0.5 }, { wavelength: 2, bandwidth: 3 }]) for (const [output, val] of [['magnitude', 0], ['real', 128]]) {
      const ctx = {}; const o = OPS.gabor_filter.apply(cv, flat, { ...prm, output }, ctx);
      assert.ok(Array.from(o.data).every((v) => v === val), `${JSON.stringify(prm)} ${output}`); assert.equal(scalarMap(ctx).gabor_max, 0); assert.equal(scalarMap(ctx).gabor_mean, 0); assert.match(ctx.info, /ไม่พบลายที่ความถี่และทิศนี้/); o.delete();
    }
    // worst measured residue of a flat 255 image (lambda 61, theta 45, bandwidth 0.5, gamma 0.85) and other flat grey levels: detected as flat directly (min == max), not by the size of the residue
    for (const v of [0, 1, 128, 255]) for (const output of ['magnitude', 'real']) { const fl = gray(70, 50, () => v), ctx = {}; const o = OPS.gabor_filter.apply(cv, fl, { wavelength: 61, orientation: 45, bandwidth: 0.5, aspect: 0.85, output }, ctx); assert.ok(Array.from(o.data).every((x) => x === (output === 'real' ? 128 : 0)), `flat ${v} ${output}`); assert.equal(scalarMap(ctx).gabor_max, 0); assert.match(ctx.info, /ภาพเรียบ ทุกพิกเซลเท่ากัน/); o.delete(); fl.delete(); }
    const ctxF = {}; const of = OPS.gabor_filter.apply(cv, faint, { wavelength: 8 }, ctxF); assert.ok(scalarMap(ctxF).gabor_max > 0.05, `${scalarMap(ctxF).gabor_max}`); assert.equal(Math.max(...of.data), 255); assert.ok(!/ไม่พบลาย/.test(ctxF.info)); of.delete();
    const gc = new cv.Mat(); cv.cvtColor(c, gc, cv.COLOR_RGB2GRAY); const a = OPS.gabor_filter.apply(cv, c, {}, {}), b = OPS.gabor_filter.apply(cv, gc, {}, {}); sameBytes(a, b, 'rgb == its gray'); a.delete(); b.delete(); gc.delete();
    const w = OPS.gabor_filter.apply(cv, g, { wavelength: 999, orientation: 999, bandwidth: 99, aspect: 99, output: 'constructor' }, {}), r = OPS.gabor_filter.apply(cv, g, { wavelength: 64, orientation: 179, bandwidth: 3, aspect: 1 }, {}); sameBytes(w, r, 'clamps + unknown output = magnitude'); w.delete(); r.delete();
    const w2 = OPS.gabor_filter.apply(cv, g, { wavelength: 0, orientation: -5, bandwidth: 0, aspect: 0 }, {}), r2 = OPS.gabor_filter.apply(cv, g, { wavelength: 2, orientation: 0, bandwidth: 0.5, aspect: 0.2 }, {}); sameBytes(w2, r2, 'low clamps'); w2.delete(); r2.delete();
  } finally { for (const m of [flat, faint, c, g]) m.delete(); }
});

// ===================================================================
// GLCM
// ===================================================================
// MATLAB graycomatrix doc image (4x5, levels 1..8) — pairs (right neighbour) listed by hand
const GL_IMG = [[1, 1, 5, 6, 8], [2, 3, 5, 7, 1], [4, 5, 7, 1, 2], [8, 5, 1, 2, 5]];
const glMat = () => gray(5, 4, (x, y) => (GL_IMG[y][x] - 1) * 32 + 16);
const GL_PAIRS_0 = [[1, 1, 1], [1, 2, 2], [1, 5, 1], [2, 3, 1], [2, 5, 1], [3, 5, 1], [4, 5, 1], [5, 1, 1], [5, 6, 1], [5, 7, 2], [6, 8, 1], [7, 1, 2], [8, 5, 1]]; // [i, j, count], levels 1-based
function glProps(C, L) { // independent re-statement of graycoprops
  let N = 0; for (const c of C) N += c;
  const P = Array.from(C, (c) => c / N); let con = 0, en = 0, hom = 0, mi = 0, mj = 0;
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) { const p = P[i * L + j]; con += (i - j) ** 2 * p; en += p * p; hom += p / (1 + Math.abs(i - j)); mi += i * p; mj += j * p; }
  let vi = 0, vj = 0, cov = 0; for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) { const p = P[i * L + j]; vi += (i - mi) ** 2 * p; vj += (j - mj) ** 2 * p; cov += (i - mi) * (j - mj) * p; }
  return { N, contrast: con, energy: en, homogeneity: hom, correlation: cov / Math.sqrt(vi * vj) };
}
test('ip4b-gl1. glcm on the MATLAB graycomatrix example image (4x5): the counts equal the hand-listed pair table, contrast / energy / homogeneity / correlation equal an independent restatement of graycoprops, the pair count is 16, the 256x256 table has the log-scaled 32x32 blocks, the info line carries the numbers', () => {
  const m = glMat();
  try {
    const q = Uint8Array.from(m.data, (v) => Math.min(7, Math.floor(v * 8 / 255)));
    assert.deepEqual(Array.from(q), GL_IMG.flat().map((v) => v - 1), 'the test image quantizes to levels 0..7 (= MATLAB 1..8 minus 1)');
    const C = glcmCounts(q, 5, 4, 8, 1, 0), exp = new Float64Array(64); for (const [i, j, c] of GL_PAIRS_0) exp[(i - 1) * 8 + j - 1] = c;
    assert.deepEqual(Array.from(C), Array.from(exp)); assert.equal(C.reduce((a, b) => a + b, 0), 16);
    const ref = glProps(C, 8), p = glcmProps(C, 8);
    for (const k of ['contrast', 'energy', 'homogeneity', 'correlation']) near(p[k], ref[k], 1e-12, k); assert.equal(p.pairs, 16);
    const ctx = {}; const o = OPS.glcm.apply(cv, m, {}, ctx), v = scalarMap(ctx);
    assert.deepEqual(ctx.values.fields.map((f) => f.key), GLCM_KEYS);
    near(v.glcm_contrast, ref.contrast, 1e-12, 'op contrast'); near(v.glcm_energy, ref.energy, 1e-12, 'op energy'); near(v.glcm_homogeneity, ref.homogeneity, 1e-12, 'op homogeneity'); near(v.glcm_correlation, ref.correlation, 1e-12, 'op correlation'); assert.equal(v.glcm_pairs, 16);
    assert.equal(ctx.info, `GLCM 8 ระดับ · ระยะ 1 px ทิศ 0° · contrast ${fixed2(ref.contrast)} · energy ${fixed3(ref.energy)} · homogeneity ${fixed3(ref.homogeneity)} · correlation ${fixed3(ref.correlation)} · แถว = ระดับของพิกเซล คอลัมน์ = ระดับของเพื่อนบ้าน · เส้นทแยงสว่าง = ภาพเรียบ`);
    assert.equal(o.rows, 256); assert.equal(o.cols, 256); assert.equal(o.channels(), 1);
    const cmax = Math.max(...exp);
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { const c = exp[i * 8 + j], want = c === 0 ? 0 : Math.floor(255 * Math.log1p(c) / Math.log1p(cmax) + 0.5); for (const [y, x] of [[i * 32, j * 32], [i * 32 + 31, j * 32 + 31], [i * 32 + 7, j * 32 + 20]]) assert.equal(o.data[y * 256 + x], want, `block (${i},${j})`); }
    o.delete();
    const c2 = {}; const o2 = OPS.glcm.apply(cv, m, { view: 'quantized', levels: '8' }, c2); assert.equal(o2.rows, 4); assert.equal(o2.cols, 5);
    for (let i = 0; i < 20; i++) assert.equal(o2.data[i], Math.floor(q[i] * 255 / 7 + 0.5)); assert.ok(!/แถว = ระดับ/.test(c2.info)); o2.delete();
  } finally { m.delete(); }
});

test('ip4b-gl2. glcm other directions / symmetric / average / constant / checkerboard / distance: offsets are (x right, y down): 0 = (d, 0), 45 = (d, -d), 90 = (0, -d), 135 = (-d, -d); symmetric = C + C^T; avg = mean of the four directions\' values (pairs = their sum); a flat image gives energy 1 / contrast 0 / homogeneity 1 / correlation null (+ the text); a 1-px checkerboard has the maximal contrast (L-1)^2; a distance bigger than the image gives 0 pairs + the text', () => {
  const m = glMat(), q = Uint8Array.from(m.data, (v) => Math.min(7, Math.floor(v * 8 / 255)));
  try {
    // directions on a hand-checkable 3x3: values 1..9 row by row
    const u = gray(3, 3, (x, y) => (y * 3 + x) * 28), qu = Uint8Array.from(u.data, (v) => Math.min(63, Math.floor(v * 64 / 255)));
    const cnt = (dx, dy) => { const C = glcmCounts(qu, 3, 3, 64, dx, dy); const r = []; for (let i = 0; i < 4096; i++) for (let k = 0; k < C[i]; k++) r.push([Math.floor(i / 64), i % 64]); return r; };
    assert.equal(cnt(1, 0).length, 6); assert.equal(cnt(1, -1).length, 4); assert.equal(cnt(0, -1).length, 6); assert.equal(cnt(-1, -1).length, 4); assert.equal(cnt(2, 0).length, 3);
    const pairAt = (dx, dy) => { const set = new Set(cnt(dx, dy).map(([a, b]) => `${a},${b}`)); return (x, y) => set.has(`${qu[y * 3 + x]},${qu[(y + dy) * 3 + x + dx]}`); };
    assert.ok(pairAt(1, -1)(0, 2)); assert.ok(pairAt(1, -1)(1, 1)); assert.ok(pairAt(0, -1)(1, 2)); assert.ok(pairAt(-1, -1)(2, 2)); assert.ok(pairAt(-1, -1)(1, 1)); assert.ok(pairAt(1, 0)(0, 0));
    u.delete();
    const sepVals = {}; for (const a of ['0', '45', '90', '135']) { const ctx = {}; OPS.glcm.apply(cv, m, { angle: a }, ctx).delete(); sepVals[a] = scalarMap(ctx); }
    const ctxA = {}; OPS.glcm.apply(cv, m, { angle: 'avg' }, ctxA).delete(); const va = scalarMap(ctxA);
    for (const k of ['glcm_contrast', 'glcm_energy', 'glcm_homogeneity']) near(va[k], (sepVals[0][k] + sepVals[45][k] + sepVals[90][k] + sepVals[135][k]) / 4, 1e-12, `avg ${k}`);
    assert.equal(va.glcm_pairs, sepVals[0].glcm_pairs + sepVals[45].glcm_pairs + sepVals[90].glcm_pairs + sepVals[135].glcm_pairs); assert.equal(va.glcm_pairs, 16 + 12 + 15 + 12);
    assert.match(ctxA.info, /ทิศ เฉลี่ย 4 ทิศ/);
    // symmetric
    const cs = {}; OPS.glcm.apply(cv, m, { symmetric: true }, cs).delete(); assert.equal(scalarMap(cs).glcm_pairs, 32);
    const C = glcmCounts(q, 5, 4, 8, 1, 0), S = new Float64Array(64); for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) S[i * 8 + j] = C[i * 8 + j] + C[j * 8 + i];
    const ref = glProps(S, 8); near(scalarMap(cs).glcm_contrast, ref.contrast, 1e-12, 'symmetric contrast'); near(scalarMap(cs).glcm_correlation, ref.correlation, 1e-12, 'symmetric correlation');
    // flat
    const flat = gray(20, 10, () => 90), cf = {}; const of = OPS.glcm.apply(cv, flat, {}, cf), vf = scalarMap(cf);
    assert.equal(vf.glcm_energy, 1); assert.equal(vf.glcm_contrast, 0); assert.equal(vf.glcm_homogeneity, 1); assert.equal(vf.glcm_correlation, null); assert.match(cf.info, /correlation — · ภาพมีระดับเดียว correlation หาไม่ได้|correlation — .*ภาพมีระดับเดียว correlation หาไม่ได้/); of.delete(); flat.delete();
    // checkerboard 1 px, every level count: contrast (L-1)^2, correlation -1
    const chk = gray(16, 16, (x, y) => ((x + y) & 1 ? 255 : 0));
    for (const L of [2, 4, 8, 16, 32, 64]) { const ctx = {}; OPS.glcm.apply(cv, chk, { levels: String(L) }, ctx).delete(); const v = scalarMap(ctx); near(v.glcm_contrast, (L - 1) ** 2, 1e-9, `contrast L${L}`); near(v.glcm_energy, 0.5, 1e-12, 'energy'); near(v.glcm_homogeneity, 1 / L, 1e-12, 'homogeneity'); near(v.glcm_correlation, -1, 1e-9, 'correlation'); }
    chk.delete();
    // distance bigger than the image
    const small = gray(4, 3, (x, y) => x * 40 + y), cd = {}; const od = OPS.glcm.apply(cv, small, { distance: 10 }, cd), vd = scalarMap(od ? cd : cd);
    assert.equal(cd.info, 'ภาพเล็กกว่าระยะ 10 px — ไม่มีคู่พิกเซลให้นับ'); assert.equal(vd.glcm_pairs, 0); for (const k of GLCM_KEYS.slice(0, 4)) assert.equal(vd[k], null); assert.equal(od.rows, 256); assert.ok(Array.from(od.data).every((v) => v === 0)); od.delete();
    const cd2 = {}; OPS.glcm.apply(cv, small, { distance: 10, angle: 'avg' }, cd2).delete(); assert.equal(scalarMap(cd2).glcm_pairs, 0);
    const cd3 = {}; OPS.glcm.apply(cv, gray(4, 1, (x) => x * 50), { distance: 2, angle: '90' }, cd3).delete(); assert.equal(scalarMap(cd3).glcm_pairs, 0, 'one row: no vertical pairs');
    // weird params
    const w = OPS.glcm.apply(cv, m, { levels: 'constructor', angle: 'constructor', distance: 99, view: 'constructor' }, {}), r = OPS.glcm.apply(cv, m, { levels: '8', angle: '0', distance: 10, view: 'matrix' }, {}); sameBytes(w, r, 'weird'); w.delete(); r.delete();
    const l3 = {}, o3 = OPS.glcm.apply(cv, m, { levels: 8, angle: 45 }, l3); assert.match(l3.info, /ทิศ 45°/); o3.delete(); // saved projects may hold numbers
  } finally { m.delete(); }
});

test('ip4b-gl3. glcm directions THROUGH THE OP: for distance 1 / 2 / 3 and every angle, symmetric on / off, the contrast / energy / homogeneity / correlation / pairs equal a brute-force enumeration of the MATLAB offsets ([row col] = [0 D] [-D D] [-D 0] [-D -D], i.e. 45 deg is up-right and 135 deg is up-left) on the MATLAB example image and a random 9x7 image', () => {
  const MAT = { 0: (d) => [0, d], 45: (d) => [-d, d], 90: (d) => [-d, 0], 135: (d) => [-d, -d] };
  const imgs = [glMat(), noiseGray(9, 7, 4)];
  try {
    for (const m of imgs) {
      const W = m.cols, H = m.rows, q = Uint8Array.from(m.data, (v) => Math.min(7, Math.floor(v * 8 / 255)));
      for (const angle of ['0', '45', '90', '135']) for (const d of [1, 2, 3]) for (const sym of [false, true]) {
        const [dr, dc] = MAT[angle](d); let C = new Float64Array(64);
        for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) { const r2 = r + dr, c2 = c + dc; if (r2 >= 0 && r2 < H && c2 >= 0 && c2 < W) C[q[r * W + c] * 8 + q[r2 * W + c2]]++; }
        if (sym) { const T = new Float64Array(64); for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) T[i * 8 + j] = C[i * 8 + j] + C[j * 8 + i]; C = T; }
        const ctx = {}; OPS.glcm.apply(cv, m, { angle, distance: d, symmetric: sym }, ctx).delete(); const v = scalarMap(ctx), N = C.reduce((a, b) => a + b, 0);
        assert.equal(v.glcm_pairs, N, `${angle} d${d} sym${sym}: pairs`);
        if (N === 0) continue;
        const ref = glProps(C, 8);
        for (const [k, rk] of [['glcm_contrast', 'contrast'], ['glcm_energy', 'energy'], ['glcm_homogeneity', 'homogeneity']]) near(v[k], ref[rk], 1e-12, `${angle} d${d} sym${sym} ${rk}`);
        if (Number.isFinite(ref.correlation)) near(v.glcm_correlation, ref.correlation, 1e-12, `${angle} d${d} sym${sym} correlation`);
      }
    }
  } finally { for (const m of imgs) m.delete(); }
});

test('ip4b-gl4. glcm correlation is null when EITHER side has a single level (rows only / columns only), with finite contrast / energy / homogeneity; "avg" with a direction that has no pairs (1-px-wide image) gives all-null values with the CORRECT reason (not "one level")', () => {
  const run = (m, p) => { const ctx = {}; OPS.glcm.apply(cv, m, p, ctx).delete(); return { v: scalarMap(ctx), info: ctx.info }; };
  // angle 90: pair = (bottom pixel, top pixel): constant top row -> only one column level; constant bottom row -> only one row level
  const colsSingle = gray(3, 2, (x, y) => (y === 0 ? 0 : x * 100)), rowsSingle = gray(3, 2, (x, y) => (y === 0 ? x * 100 : 0));
  for (const m of [colsSingle, rowsSingle]) {
    const { v, info } = run(m, { angle: '90' });
    assert.equal(v.glcm_correlation, null); assert.ok(Number.isFinite(v.glcm_contrast) && v.glcm_contrast > 0); assert.ok(Number.isFinite(v.glcm_energy) && Number.isFinite(v.glcm_homogeneity)); assert.equal(v.glcm_pairs, 3);
    assert.match(info, /correlation —.*ภาพมีระดับเดียว correlation หาไม่ได้/);
  }
  colsSingle.delete(); rowsSingle.delete();
  const narrow = gray(1, 6, (x, y) => y * 40), { v, info } = run(narrow, { angle: 'avg' });
  assert.ok(v.glcm_pairs > 0); for (const k of GLCM_KEYS.slice(0, 4)) assert.equal(v[k], null, k);
  assert.match(info, /บางทิศไม่มีคู่พิกเซล \(ภาพเล็กกว่าระยะ\) จึงหาค่าเฉลี่ย 4 ทิศไม่ได้/); assert.ok(!/ภาพมีระดับเดียว/.test(info), info);
  narrow.delete();
});

test('ip4b-nl4. nlmCore == the brute force on a 300x300 RGB noise image whose integral image REALLY exceeds 2^31 (the Int32 wrap + `| 0` must give the exact window sums)', () => {
  const m = noiseRGB(300, 300, 77), W = 300, H = 300, d = m.data;
  let total = 0; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const p = (y * W + x) * 3, b = (y * W + Math.min(W - 1, x + 1)) * 3; for (let c = 0; c < 3; c++) { const t = d[p + c] - d[b + c]; total += t * t; } }
  assert.ok(total > 2 ** 31, `the d2 integral of one offset (${total}) must exceed 2^31`);
  // h = 100: at h = 10 every off-centre weight on full noise is ~e^-108, so the output == the input and the window sums are never checked
  try {
    const o = nlmCore(d, W, H, 3, 100, 1, 1);
    assert.ok(o.some((v, i) => v !== d[i]), 'h = 100 must actually change the image (the window sums matter)');
    assert.deepEqual(Array.from(o), Array.from(nlmRef(d, W, H, 3, 100, 1, 1)));
  } finally { m.delete(); }
});

test('ip4b-dh4. dehaze with the guided filter ON == a direct guidedFilterF32 call with radius max(4, 2 * patch) on the coarse map (pins the radius 2 * ps; gray guide for colour)', () => {
  const imgs = [noiseRGB(40, 30, 3), noiseGray(33, 21, 5)];
  try {
    for (const m of imgs) for (const ps of [3, 7, 15]) {
      const ref = dehazeRef(m, 0.95, ps), W = m.cols, H = m.rows, N = W * H, ch = m.channels();
      const gm = ch === 3 ? (() => { const t = new cv.Mat(); cv.cvtColor(m, t, cv.COLOR_RGB2GRAY); return t; })() : m, G = new cv.Mat(); gm.convertTo(G, cv.CV_32F);
      const P = cv.matFromArray(H, W, cv.CV_32FC1, Float32Array.from(ref.tCoarse, (t) => t * 255)), q = guidedFilterF32(cv, G, P, Math.max(4, 2 * ps), 65.025), qd = q.data32F;
      const t = Float64Array.from({ length: N }, (_, i) => Math.max(Math.fround(qd[i] / 255), 0.1)), A = ref.A, exp = new Uint8Array(N * ch);
      for (let i = 0; i < N; i++) for (let c = 0; c < ch; c++) { const v = Math.floor((m.data[i * ch + c] - A[c]) / t[i] + A[c] + 0.5); exp[i * ch + c] = v < 0 ? 0 : v > 255 ? 255 : v; }
      const o = OPS.dehaze.apply(cv, m, { patch: ps }, {}); assert.deepEqual(Array.from(o.data), Array.from(exp), `${W}x${H}x${ch} patch ${ps}`);
      for (const x of [o, q, P, G]) x.delete(); if (gm !== m) gm.delete();
    }
  } finally { for (const m of imgs) m.delete(); }
});

test('ip4b-w4. deconv_wiener / deconv_lucy: an image above 640x480 px gets a Thai "ภาพใหญ่ ... ช้า" note in the info (output unchanged -- no resizing); at or below 640x480 the info text is exactly the old text', () => {
  const big = noiseGray(700, 500, 3), ok = noiseGray(640, 480, 3);
  try {
    for (const [id, p] of [['deconv_wiener', {}], ['deconv_lucy', { iterations: 1 }]]) {
      const cb = {}, ob = OPS[id].apply(cv, big, p, cb); assert.equal(ob.cols, 700); assert.equal(ob.rows, 500); assert.match(cb.info, / · ภาพใหญ่ \(700×500 px\) ช้า/); assert.match(cb.info, /ควรย่อภาพ \(Resize\) ก่อน/); ob.delete();
      const co = {}, oo = OPS[id].apply(cv, ok, p, co); assert.ok(!/ภาพใหญ่/.test(co.info), co.info); oo.delete();
    }
    const cl = {}, ol = OPS.deconv_lucy.apply(cv, big, { iterations: 3 }, cl); assert.match(cl.info, /Lucy 3 รอบบนภาพขนาดนี้อาจใช้เวลาหลายวินาที/); ol.delete();
  } finally { big.delete(); ok.delete(); }
});

// ===================================================================
// Non-local Means
// ===================================================================
test('ip4b-nl1. nlmCore == the brute force (no integral image, per-window sums; byte-equal) on 12x9 gray / RGB images for every patch x search, a 1-px wide image, and 1x1; the exact window cut at the image border and replicate clamping of the neighbour are pinned', () => {
  const imgs = [noiseGray(12, 9, 1), noiseRGB(12, 9, 2), noiseGray(1, 7, 3), noiseGray(7, 1, 4), gray(1, 1, () => 77)];
  try {
    for (const m of imgs) {
      const ch = m.channels(), W = m.cols, H = m.rows;
      for (const [pr, sr, h] of [[1, 3, 10], [2, 5, 20], [3, 7, 5], [1, 5, 50]]) assert.deepEqual(Array.from(nlmCore(m.data, W, H, ch, h, pr, sr)), Array.from(nlmRef(m.data, W, H, ch, h, pr, sr)), `${W}x${H}x${ch} pr${pr} sr${sr} h${h}`);
    }
  } finally { for (const m of imgs) m.delete(); }
});

test('ip4b-nl2. nlm_denoise: smooth image + seeded gaussian noise -> MSE falls by at least 30%; a larger h smooths more (and a constant image / the 1x1 image come back unchanged); a texture of repeating dots keeps more contrast than a plain mean filter; colour works and keeps channels; the info line and the patch / search selects', () => {
  const clean = gray(48, 48, (x, y) => Math.floor(60 + 2 * x + 1.5 * y)), r = seededGaussian(7, 48 * 48);
  const noisy = gray(48, 48, (x, y) => Math.min(255, Math.max(0, Math.floor(60 + 2 * x + 1.5 * y + 15 * r[y * 48 + x] + 0.5))));
  try {
    const ctx = {}; const o = OPS.nlm_denoise.apply(cv, noisy, { search: '7' }, ctx);
    assert.ok(mse(o, clean) < 0.7 * mse(noisy, clean), `MSE ${mse(o, clean)} vs noisy ${mse(noisy, clean)}`);
    assert.equal(ctx.info, 'Non-local Means h=10 · แผ่น 5×5 · ค้นหา 7×7 · ช้า: เหมาะกับภาพนิ่ง');
    const h1 = OPS.nlm_denoise.apply(cv, noisy, { h: 1, search: '7' }, {}), h40 = OPS.nlm_denoise.apply(cv, noisy, { h: 40, search: '7' }, {});
    const rough = (m) => { let s = 0; for (let y = 0; y < 47; y++) for (let x = 0; x < 47; x++) s += Math.abs(m.data[y * 48 + x] - m.data[y * 48 + x + 1]) + Math.abs(m.data[y * 48 + x] - m.data[(y + 1) * 48 + x]); return s; };
    assert.ok(rough(h40) < rough(h1) * 0.8, `h 40 is smoother than h 1 (${rough(h40)} vs ${rough(h1)})`); assert.ok(mae(h1, noisy) < 1, 'h 1 barely changes the image');
    for (const m of [o, h1, h40]) m.delete();
    const flat = gray(20, 15, () => 66), of = OPS.nlm_denoise.apply(cv, flat, {}, {}); sameBytes(of, flat, 'flat'); of.delete(); flat.delete();
    const one = gray(1, 1, () => 200), o1 = OPS.nlm_denoise.apply(cv, one, {}, {}); sameBytes(o1, one, '1x1'); o1.delete(); one.delete();
    const c = noiseRGB(16, 12, 5), oc = OPS.nlm_denoise.apply(cv, c, { patch: '3', search: '7' }, {}); assert.equal(oc.channels(), 3); assert.equal(oc.rows, 12); oc.delete(); c.delete();
    const a = toRGBA(noiseRGB(16, 12, 5)), oa = OPS.nlm_denoise.apply(cv, a, { search: '7' }, {}); assert.equal(oa.channels(), 3); oa.delete(); a.delete();
    const wei = OPS.nlm_denoise.apply(cv, noisy, { patch: 'constructor', search: 'constructor', h: 'x' }, {}), dflt = OPS.nlm_denoise.apply(cv, noisy, {}, {}); sameBytes(wei, dflt, 'weird = defaults'); wei.delete(); dflt.delete();
    const big = OPS.nlm_denoise.apply(cv, noisy, { h: 999, patch: '7', search: '7' }, {}), ref = OPS.nlm_denoise.apply(cv, noisy, { h: 50, patch: '7', search: '7' }, {}); sameBytes(big, ref, 'h clamp'); big.delete(); ref.delete();
    const op = OPS.nlm_denoise.apply(cv, noisy, { h: 12, patch: '3', search: '7' }, {}), exp = nlmRef(noisy.data, 48, 48, 1, 12, 1, 3); assert.deepEqual(Array.from(op.data), Array.from(exp), 'the op == the brute force'); op.delete();
  } finally { clean.delete(); noisy.delete(); }
});

test('ip4b-nl3. nlm_denoise on an image whose long side is above NLM_MAX_SIDE (700x30): processed at 640x29 and enlarged back (same size as the input, the text says so); a 640-wide image is NOT shrunk (no text); the constant NLM_MAX_SIDE is 640', () => {
  const wide = gray(700, 30, (x, y) => (x * 3 + y * 7) & 255), exact = gray(640, 3, (x) => (x * 5) & 255);
  try {
    const ctx = {}; const o = OPS.nlm_denoise.apply(cv, wide, { search: '7', patch: '3' }, ctx);
    assert.equal(o.cols, 700); assert.equal(o.rows, 30); assert.match(ctx.info, /ประมวลผลบนภาพย่อ 640×27 แล้วขยายกลับ \(ภาพใหญ่ช้ามาก\)/); assert.ok(ctx.info.endsWith(' · ช้า: เหมาะกับภาพนิ่ง')); o.delete();
    const c2 = {}; const o2 = OPS.nlm_denoise.apply(cv, exact, { search: '7', patch: '3' }, c2); assert.equal(o2.cols, 640); assert.ok(!/ภาพย่อ/.test(c2.info)); o2.delete();
    assert.equal(NLM_MAX_SIDE, 640);
  } finally { wide.delete(); exact.delete(); }
});

// ===================================================================
// Dehaze
// ===================================================================
function hazyScene() { // ฉากมืด (ทุกพิกเซลมีหนึ่งช่องเป็น 0) · ซ้าย 80% t = 0.5 · แถบขวา 20% = ท้องฟ้า t = 0 (I = A)
  const W = 64, H = 48, A = [230, 235, 240], r = lcg(77), J = new Uint8Array(W * H * 3), I = new Uint8Array(W * H * 3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const zero = Math.floor(r() * 3), t = x >= 52 ? 0 : 0.5;
    for (let c = 0; c < 3; c++) { const j = c === zero ? 0 : Math.floor(r() * 41); J[(y * W + x) * 3 + c] = j; I[(y * W + x) * 3 + c] = Math.floor(j * t + A[c] * (1 - t) + 0.5); }
  }
  return { J: cv.matFromArray(H, W, cv.CV_8UC3, J), I: cv.matFromArray(H, W, cv.CV_8UC3, I), A };
}
test('ip4b-dh1. dehaze on a synthetic hazy scene (dark scene J, haze colour A = (230, 235, 240), t = 0.5 with a t = 0 sky strip): the estimated A is within 3 of the real haze colour, the mean t is near the true mean, the dehazed image is much closer to J than the hazy image (outside the sky), with and without the guided filter', () => {
  const { J, I, A } = hazyScene();
  try {
    const trueT = (52 * 0.5 + 12 * 0) / 64;
    for (const refine of [false, true]) {
      const ctx = {}; const o = OPS.dehaze.apply(cv, I, { refine, patch: 7 }, ctx), v = scalarMap(ctx);
      assert.deepEqual(ctx.values.fields.map((f) => f.key), DEHAZE_KEYS);
      near(v.dh_a_r, A[0], 3, 'A red'); near(v.dh_a_g, A[1], 3, 'A green'); near(v.dh_a_b, A[2], 3, 'A blue'); near(v.dh_t_mean, trueT, 0.1, 'mean t');
      let eh = 0, ed = 0, n = 0; for (let y = 0; y < 48; y++) for (let x = 0; x < 44; x++) for (let c = 0; c < 3; c++) { const i = (y * 64 + x) * 3 + c; eh += Math.abs(I.data[i] - J.data[i]); ed += Math.abs(o.data[i] - J.data[i]); n++; }
      assert.ok(ed / n < 0.5 * eh / n, `refine ${refine}: dehazed MAE ${ed / n} vs hazy ${eh / n}`);
      assert.equal(o.channels(), 3); o.delete();
    }
  } finally { J.delete(); I.delete(); }
});

test('ip4b-dh2. dehaze (refine off) == a brute-force re-implementation of the dark-channel steps (explicit window min, float32 rounding at the same points): the bytes, A, mean t and the dark / transmission views are EXACTLY equal, for colour and gray images and patch 3 / 7 / 15 / 31', () => {
  const imgs = [noiseRGB(40, 30, 3), smoothRGB(33, 21), noiseGray(25, 18, 5), hazyScene().I];
  try {
    for (const m of imgs) for (const ps of [3, 7, 15, 31]) for (const omega of [0.95, 0.4]) {
      const ref = dehazeRef(m, omega, ps), ctx = {}, o = OPS.dehaze.apply(cv, m, { refine: false, patch: ps, amount: omega }, ctx);
      assert.deepEqual(Array.from(o.data), Array.from(ref.out), `${m.cols}x${m.rows}x${m.channels()} patch ${ps} omega ${omega}`);
      const v = scalarMap(ctx); const a = m.channels() === 1 ? [ref.A[0], ref.A[0], ref.A[0]] : ref.A;
      assert.equal(v.dh_a_r, a[0]); assert.equal(v.dh_a_g, a[1]); assert.equal(v.dh_a_b, a[2]); near(v.dh_t_mean, ref.tMean, 1e-12, 't mean');
      assert.equal(ctx.info, `Dehaze ω=${omega} หน้าต่าง ${ps}×${ps} · แสงหมอก A (R/G/B) = ${fixed1(a[0])}/${fixed1(a[1])}/${fixed1(a[2])} · ความโปร่งใสเฉลี่ย ${fixed2(ref.tMean)} · ไม่เกลี่ยแผนที่หมอก (เห็นเป็นบล็อกรอบขอบวัตถุ)`);
      o.delete();
    }
    const m = imgs[0], ref = dehazeRef(m, 0.95, 7);
    const od = OPS.dehaze.apply(cv, m, { refine: false, patch: 7, view: 'dark' }, {}); assert.equal(od.channels(), 1); assert.deepEqual(Array.from(od.data), Array.from(ref.dark)); od.delete();
    const ot = OPS.dehaze.apply(cv, m, { refine: false, patch: 7, view: 'transmission' }, {}); assert.equal(ot.channels(), 1);
    for (let i = 0; i < 40 * 30; i++) assert.ok(ot.data[i] >= Math.floor(0.1 * 255 + 0.5) && ot.data[i] <= 255, `t >= 0.1 at ${i}`); ot.delete();
  } finally { for (const m of imgs) m.delete(); }
});
function fixed1(x) { return (Math.floor(x * 10 + 0.5) / 10).toFixed(1); }

test('ip4b-dh3. dehaze: refine on differs from off (and says nothing about "ไม่เกลี่ย"); the minimum t is 0.1 (a black / white image does not blow up: A = max(A, 1), no 0 division); gray images give A three times; RGBA -> 3 channels; weird params clamp; 0x0', () => {
  const c = noiseRGB(40, 30, 3), blk = new cv.Mat(20, 20, cv.CV_8UC3, new cv.Scalar(0, 0, 0)), wht = new cv.Mat(20, 20, cv.CV_8UC3, new cv.Scalar(255, 255, 255)), g = noiseGray(30, 20, 6);
  try {
    const cr = {}, cn = {}; const a = OPS.dehaze.apply(cv, c, { patch: 7 }, cr), b = OPS.dehaze.apply(cv, c, { patch: 7, refine: false }, cn); assert.ok(mae(a, b) > 0.05, 'refine changes the result'); assert.ok(!/ไม่เกลี่ย/.test(cr.info)); assert.match(cn.info, /ไม่เกลี่ยแผนที่หมอก/); a.delete(); b.delete();
    const ck = {}; const ob = OPS.dehaze.apply(cv, blk, {}, ck); assert.ok(Array.from(ob.data).every((v) => v === 0)); assert.equal(scalarMap(ck).dh_a_r, 1); assert.ok(Number.isFinite(scalarMap(ck).dh_t_mean)); ob.delete();
    const cw = {}; const ow = OPS.dehaze.apply(cv, wht, {}, cw); assert.ok(Array.from(ow.data).every((v) => v === 255), 'white stays white'); ow.delete();
    const cg = {}; const og = OPS.dehaze.apply(cv, g, { patch: 5 }, cg), vg = scalarMap(cg); assert.equal(vg.dh_a_r, vg.dh_a_g); assert.equal(vg.dh_a_g, vg.dh_a_b); assert.equal(og.channels(), 1); og.delete();
    const ra = toRGBA(c), oa = OPS.dehaze.apply(cv, ra, { patch: 7 }, {}), ref = OPS.dehaze.apply(cv, c, { patch: 7 }, {}); assert.equal(oa.channels(), 3); sameBytes(oa, ref, 'RGBA == RGB'); oa.delete(); ref.delete(); ra.delete();
    const w1 = OPS.dehaze.apply(cv, c, { amount: 9, patch: 99, view: 'constructor', refine: 'x' }, {}), r1 = OPS.dehaze.apply(cv, c, { amount: 1, patch: 31, view: 'result', refine: true }, {}); sameBytes(w1, r1, 'clamps / unknown view / non-false refine'); w1.delete(); r1.delete();
    const w2 = OPS.dehaze.apply(cv, c, { amount: 0, patch: 4 }, {}), r2 = OPS.dehaze.apply(cv, c, { amount: 0.1, patch: 5 }, {}); sameBytes(w2, r2, 'low clamp + even patch -> odd'); w2.delete(); r2.delete();
    const e = new cv.Mat(), ce = {}; const oe = OPS.dehaze.apply(cv, e, {}, ce); assert.equal(oe.rows, 0); assert.equal(ce.info, 'ภาพว่าง'); assert.deepEqual(ce.values.fields.map((f) => f.key), DEHAZE_KEYS); for (const f of ce.values.fields) assert.equal(f.value, null); oe.delete(); e.delete();
  } finally { for (const m of [c, blk, wht, g]) m.delete(); }
});

// ===================================================================
// DCT / JPEG
// ===================================================================
test('ip4b-dc1. dct8Table is orthonormal (C C^T = I within 1e-12); dct8Forward of a block == the textbook double sum within 1e-9; dct8Inverse(dct8Forward(B)) == B within 1e-9; a constant block v has F00 = 8 (v - 128); the first basis row is constant 1/sqrt(8)', () => {
  const C = dct8Table();
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { let s = 0; for (let k = 0; k < 8; k++) s += C[i * 8 + k] * C[j * 8 + k]; near(s, i === j ? 1 : 0, 1e-12, `C C^T [${i}][${j}]`); }
  for (let x = 0; x < 8; x++) near(C[x], Math.sqrt(1 / 8), 1e-15, 'row 0');
  const r = lcg(5), B = new Float64Array(64); for (let i = 0; i < 64; i++) B[i] = Math.floor(r() * 256) - 128;
  const F = dct8Forward(B, new Float64Array(64));
  for (let u = 0; u < 8; u++) for (let v = 0; v < 8; v++) {
    let s = 0; for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) s += B[y * 8 + x] * Math.cos((2 * y + 1) * u * Math.PI / 16) * Math.cos((2 * x + 1) * v * Math.PI / 16);
    const ref = (u === 0 ? Math.sqrt(1 / 8) : Math.sqrt(2 / 8)) * (v === 0 ? Math.sqrt(1 / 8) : Math.sqrt(2 / 8)) * s; near(F[u * 8 + v], ref, 1e-9, `F[${u}][${v}]`);
  }
  const back = dct8Inverse(F, new Float64Array(64)); for (let i = 0; i < 64; i++) near(back[i], B[i], 1e-9, `round trip ${i}`);
  const K = new Float64Array(64).fill(100 - 128), FK = dct8Forward(K, new Float64Array(64)); near(FK[0], 8 * (100 - 128), 1e-9, 'DC'); for (let i = 1; i < 64; i++) near(FK[i], 0, 1e-9, 'AC of a constant block');
});

test('ip4b-dc2. jpegQTable: quality 50 is the standard Annex K.1 luminance table; other qualities follow the IJG scaling S = 5000/q (q < 50) or 200 - 2q, Qs = max(1, floor((Q S + 50) / 100)) (re-stated here, and pinned for q = 10 / 90 / 100 / 1); the table is not capped at 255 (as the spec says); the op uses it (the DC step of a constant block)', () => {
  const STD = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99];
  assert.deepEqual(Array.from(jpegQTable(50)), STD);
  for (let q = 1; q <= 100; q++) { const S = q < 50 ? Math.floor(5000 / q) : 200 - 2 * q; assert.deepEqual(Array.from(jpegQTable(q)), STD.map((v) => Math.max(1, Math.floor((v * S + 50) / 100))), `q ${q}`); }
  assert.deepEqual(Array.from(jpegQTable(10)).slice(0, 8), [80, 55, 50, 80, 120, 200, 255, 305]); assert.deepEqual(Array.from(jpegQTable(90)).slice(0, 8), [3, 2, 2, 3, 5, 8, 10, 12]);
  assert.ok(Array.from(jpegQTable(100)).every((v) => v === 1)); assert.equal(jpegQTable(1)[0], 800);
  // the op: a constant block 80 has F00 = 8 * (80 - 128) = -384 -> at quality 50 qf = -384/16 = -24 exactly -> reconstructs 80
  const m = gray(8, 8, () => 80), o = OPS.dct_jpeg.apply(cv, m, {}, {}); assert.ok(Array.from(o.data).every((v) => v === 80)); o.delete(); m.delete();
});

test('ip4b-dc3. dct_jpeg: a flat 128 image comes back unchanged at every quality / zonal K / view (PSNR infinity, zero% 100, text "∞ (เหมือนเดิมทุกพิกเซล)"); zonal K = 8 returns the original EXACTLY for any image; K = 1 gives constant 8x8 blocks; PSNR rises with K and with quality; quality 100 beats quality 10 (PSNR higher, fewer zero coefficients); 13x7 keeps its size and has 2 blocks', () => {
  const flat = gray(24, 16, () => 128), noise = noiseGray(32, 24, 4), tex = gray(32, 24, (x, y) => Math.floor(128 + 70 * Math.sin(x / 3) * Math.cos(y / 4)));
  try {
    for (const prm of [{ quality: 1 }, { quality: 50 }, { quality: 100 }, { method: 'zonal', keep: 1 }, { method: 'zonal', keep: 8 }]) {
      const ctx = {}; const o = OPS.dct_jpeg.apply(cv, flat, prm, ctx), v = scalarMap(ctx);
      assert.ok(Array.from(o.data).every((x) => x === 128), JSON.stringify(prm)); assert.equal(v.dct_psnr, Infinity); assert.ok(ctx.info.includes('PSNR ∞ (เหมือนเดิมทุกพิกเซล)'), ctx.info); assert.equal(v.dct_blocks, 6);
      if (!prm.method || prm.method === 'jpeg') assert.equal(v.dct_zero_pct, 100); o.delete();
    }
    const k8 = {}; const o8 = OPS.dct_jpeg.apply(cv, noise, { method: 'zonal', keep: 8 }, k8); sameBytes(o8, noise, 'zonal K = 8'); assert.equal(scalarMap(k8).dct_psnr, Infinity); assert.equal(scalarMap(k8).dct_zero_pct, 0); o8.delete();
    const k1 = {}; const o1 = OPS.dct_jpeg.apply(cv, tex, { method: 'zonal', keep: 1 }, k1);
    for (let by = 0; by < 3; by++) for (let bx = 0; bx < 4; bx++) { const v0 = o1.data[by * 8 * 32 + bx * 8]; for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) assert.equal(o1.data[(by * 8 + y) * 32 + bx * 8 + x], v0, `block (${bx},${by}) is constant`); }
    assert.equal(scalarMap(k1).dct_zero_pct, 63 / 64 * 100); assert.match(k1.info, /เก็บความถี่ต่ำ 1×1 จาก 8×8 ต่อบล็อก \(1\/64\) · PSNR \d+\.\d\d dB/); o1.delete();
    const ps = [1, 2, 4, 8].map((K) => { const c = {}; OPS.dct_jpeg.apply(cv, tex, { method: 'zonal', keep: K }, c).delete(); return scalarMap(c).dct_psnr; });
    for (let i = 1; i < ps.length; i++) assert.ok(ps[i] > ps[i - 1], `PSNR rises with K: ${ps}`);
    const hi = {}, lo = {}; OPS.dct_jpeg.apply(cv, tex, { quality: 100 }, hi).delete(); OPS.dct_jpeg.apply(cv, tex, { quality: 10 }, lo).delete();
    assert.ok(scalarMap(hi).dct_psnr > scalarMap(lo).dct_psnr + 5); assert.ok(scalarMap(hi).dct_zero_pct < scalarMap(lo).dct_zero_pct);
    const odd = gray(13, 7, (x, y) => (x * 20 + y * 9) & 255), co = {}, oo = OPS.dct_jpeg.apply(cv, odd, {}, co); assert.equal(oo.cols, 13); assert.equal(oo.rows, 7); assert.equal(scalarMap(co).dct_blocks, 2); oo.delete(); odd.delete();
    const one = gray(1, 1, () => 77), c1 = {}, o11 = OPS.dct_jpeg.apply(cv, one, {}, c1); assert.equal(o11.rows, 1); assert.equal(scalarMap(c1).dct_blocks, 1); o11.delete(); one.delete();
  } finally { for (const m of [flat, noise, tex]) m.delete(); }
});

test('ip4b-dc4. dct_jpeg == an independent re-statement of the pipeline (definition-based forward DCT, quantize with sign x floor(|x| + 0.5), inverse, +128, round) on random / textured images (jpeg at q 10 / 50 / 90 and zonal K 3), within 1 level in at most 0.1% of the pixels; the PSNR / zero% in the values equal the ones computed from the result image', () => {
  const imgs = [noiseGray(24, 16, 8), gray(21, 11, (x, y) => Math.floor(128 + 70 * Math.sin(x / 3) * Math.cos(y / 4)))];
  const refRun = (m, quality, zonalK) => {
    const W = m.cols, H = m.rows, Wp = Math.ceil(W / 8) * 8, Hp = Math.ceil(H / 8) * 8, out = new Uint8Array(W * H), Q = jpegQTable(quality ?? 50); let zeros = 0;
    const tieBlocks = new Set(); // blocks with a coefficient within 1e-6 of a rounding tie: FP noise (1e-14) decides them, so two correct implementations may differ there
    const px = (y, x) => m.data[Math.min(y, H - 1) * W + Math.min(x, W - 1)];
    for (let by = 0; by < Hp / 8; by++) for (let bx = 0; bx < Wp / 8; bx++) {
      const F = new Float64Array(64);
      for (let u = 0; u < 8; u++) for (let v = 0; v < 8; v++) {
        let s = 0; for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) s += (px(by * 8 + y, bx * 8 + x) - 128) * Math.cos((2 * y + 1) * u * Math.PI / 16) * Math.cos((2 * x + 1) * v * Math.PI / 16);
        let f = (u === 0 ? Math.sqrt(1 / 8) : Math.sqrt(2 / 8)) * (v === 0 ? Math.sqrt(1 / 8) : Math.sqrt(2 / 8)) * s;
        if (zonalK) { if (u >= zonalK || v >= zonalK) { f = 0; zeros++; } } else { const z = f / Q[u * 8 + v], qf = Math.sign(z) * Math.floor(Math.abs(z) + 0.5); if (Math.abs(Math.abs(z) % 1 - 0.5) < 1e-6) tieBlocks.add(`${bx},${by}`); f = qf * Q[u * 8 + v]; if (qf === 0) zeros++; }
        F[u * 8 + v] = f;
      }
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
        if (by * 8 + y >= H || bx * 8 + x >= W) continue;
        let s = 0; for (let u = 0; u < 8; u++) for (let v = 0; v < 8; v++) s += F[u * 8 + v] * (u === 0 ? Math.sqrt(1 / 8) : Math.sqrt(2 / 8)) * (v === 0 ? Math.sqrt(1 / 8) : Math.sqrt(2 / 8)) * Math.cos((2 * y + 1) * u * Math.PI / 16) * Math.cos((2 * x + 1) * v * Math.PI / 16);
        out[(by * 8 + y) * W + bx * 8 + x] = Math.min(255, Math.max(0, Math.floor(s + 128 + 0.5)));
      }
    }
    return { out, zeroPct: zeros * 100 / (Hp / 8 * (Wp / 8) * 64), tieBlocks };
  };
  try {
    for (const m of imgs) for (const [prm, q, k] of [[{ quality: 10 }, 10, 0], [{ quality: 50 }, 50, 0], [{ quality: 90 }, 90, 0], [{ method: 'zonal', keep: 3 }, 50, 3]]) {
      const ctx = {}, o = OPS.dct_jpeg.apply(cv, m, prm, ctx), ref = refRun(m, q, k); let bad = 0, mx = 0;
      let cmp = 0; for (let i = 0; i < ref.out.length; i++) { const x = i % m.cols, y = Math.floor(i / m.cols); if (ref.tieBlocks.has(`${x >> 3},${y >> 3}`)) continue; cmp++; const d = Math.abs(o.data[i] - ref.out[i]); if (d) bad++; mx = Math.max(mx, d); }
      assert.ok(cmp > ref.out.length / 2, 'most blocks are compared');
      assert.ok(mx <= 1 && bad <= Math.ceil(cmp * 0.001), `${JSON.stringify(prm)} ${m.cols}x${m.rows}: ${bad} px differ (max ${mx})`);
      const v = scalarMap(ctx); if (ref.tieBlocks.size === 0) near(v.dct_zero_pct, ref.zeroPct, 0.05, 'zero %');
      let sse = 0; for (let i = 0; i < ref.out.length; i++) sse += (m.data[i] - o.data[i]) ** 2; const ps = sse === 0 ? Infinity : 10 * Math.log10(255 * 255 / (sse / ref.out.length)); near(v.dct_psnr, ps, 1e-9, 'PSNR from the result image');
      o.delete();
    }
  } finally { for (const m of imgs) m.delete(); }
});

test('ip4b-dc5. dct_jpeg rounding: coefficients that land exactly on -x.5 round AWAY from zero (sign x floor(|x| + 0.5)), not toward +infinity like Math.round -- constant blocks whose DC / 16 = -x.5 at quality 50 (17 such grey levels) reconstruct to the away-from-zero answer; the views coeffs / error follow their formulas', () => {
  const TIES = [31, 41, 47, 51, 57, 61, 63, 77, 85, 89, 95, 99, 105, 109, 111, 117, 125];
  for (const v of TIES) {
    const m = gray(8, 8, () => v), o = OPS.dct_jpeg.apply(cv, m, {}, {});
    const z = 8 * (v - 128) / 16, qf = Math.sign(z) * Math.floor(Math.abs(z) + 0.5), expect = Math.floor(qf * 16 / 8 + 128 + 0.5), mathRound = Math.floor(Math.round(z) * 16 / 8 + 128 + 0.5);
    assert.equal(Math.abs(z) % 1, 0.5, `${v}: a real tie`); assert.notEqual(expect, mathRound, `${v}: Math.round would give another answer`);
    assert.ok(Array.from(o.data).every((x) => x === expect), `grey ${v}: expected ${expect}, got ${o.data[0]}`); o.delete(); m.delete();
  }
  const m = noiseGray(16, 8, 2), res = OPS.dct_jpeg.apply(cv, m, { quality: 30 }, {}), err = OPS.dct_jpeg.apply(cv, m, { quality: 30, view: 'error' }, {}), cf = OPS.dct_jpeg.apply(cv, m, { quality: 30, view: 'coeffs' }, {});
  for (let i = 0; i < 128; i++) assert.equal(err.data[i], Math.min(255, Math.abs(m.data[i] - res.data[i]) * 8), `error view @${i}`);
  // coeffs view: the DC cell (top-left of each block) is the brightest for a bright block; every cell is a log-scaled |F'| <= 255
  assert.ok(cf.data[0] > 150 || m.data[0] < 150); assert.ok(Array.from(cf.data).every((x) => x <= 255)); assert.equal(cf.rows, 8); assert.equal(cf.cols, 16);
  const ctx = {}; const co = OPS.dct_jpeg.apply(cv, m, { view: 'coeffs' }, ctx); assert.ok(ctx.info.endsWith(' · มุมซ้ายบนของทุกบล็อก = ความถี่ต่ำ (สว่าง)')); co.delete();
  const mid = gray(8, 8, () => 40), cdc = OPS.dct_jpeg.apply(cv, mid, { view: 'coeffs', method: 'zonal', keep: 8 }, {});
  assert.equal(cdc.data[0], Math.min(255, Math.floor(255 * Math.log1p(Math.abs(8 * (40 - 128))) / Math.log1p(1024) + 0.5)), 'DC cell = 255 log1p(|F00|) / log1p(1024)'); assert.ok(Array.from(cdc.data).slice(1).every((x) => x === 0), 'a constant block has no AC energy'); cdc.delete(); mid.delete();
  for (const x of [m, res, err, cf]) x.delete();
  // params: weird -> defaults, clamps; RGB goes through gray
  const g = noiseGray(16, 16, 3), w = OPS.dct_jpeg.apply(cv, g, { method: 'constructor', view: 'constructor', quality: 'x', keep: 'x' }, {}), d = OPS.dct_jpeg.apply(cv, g, {}, {}); sameBytes(w, d, 'weird = defaults'); w.delete(); d.delete();
  const big = OPS.dct_jpeg.apply(cv, g, { quality: 999 }, {}), q100 = OPS.dct_jpeg.apply(cv, g, { quality: 100 }, {}); sameBytes(big, q100, 'quality clamp'); big.delete(); q100.delete();
  const k9 = OPS.dct_jpeg.apply(cv, g, { method: 'zonal', keep: 99 }, {}), k8 = OPS.dct_jpeg.apply(cv, g, { method: 'zonal', keep: 8 }, {}); sameBytes(k9, k8, 'keep clamp'); k9.delete(); k8.delete();
  const c = noiseRGB(16, 16, 3), gc = new cv.Mat(); cv.cvtColor(c, gc, cv.COLOR_RGB2GRAY); const a = OPS.dct_jpeg.apply(cv, c, {}, {}), b = OPS.dct_jpeg.apply(cv, gc, {}, {}); sameBytes(a, b, 'rgb == its gray'); a.delete(); b.delete(); gc.delete(); c.delete(); g.delete();
});

// ===================================================================
// ip4b-T — the contract of every new op / operation
// ===================================================================
function allNewRuns() { // [label, run(srcMat, B) -> out] over the 8 ops + 5 operations
  const runs = [];
  for (const id of NEW_OPS) runs.push([id, (m, B, cvU = cv, ctx = {}) => { if (B) ctx.src2 = B; return OPS[id].apply(cvU, m, id === 'nlm_denoise' ? { search: '7' } : {}, ctx); }]);
  for (const op of MORPH_NEW) runs.push([`morphology ${op}`, (m, B, cvU = cv, ctx = {}) => OPS.morphology.apply(cvU, m, MORPH(op), ctx)]);
  return runs;
}
test('ip4b-T8. no new op / operation ever modifies src or ctx.src2, and each returns a NEW Mat (also for 0x0, 1x1, flat and RGBA inputs)', () => {
  const ins = [['gray', noiseGray(30, 20, 1)], ['rgb', noiseRGB(30, 20, 2)], ['rgba', toRGBA(noiseRGB(30, 20, 3))], ['flat', gray(30, 20, () => 50)], ['1x1', gray(1, 1, () => 9)], ['0x0', new cv.Mat()]];
  const B = waveGray(30, 20);
  try {
    for (const [label, run] of allNewRuns()) for (const [t, m] of ins) {
      const before = Array.from(m.data), bB = Array.from(B.data), o = run(m, B);
      try { assert.notEqual(o, m, `${label} ${t}: a new Mat`); assert.deepEqual(Array.from(m.data), before, `${label} ${t}: src modified`); assert.deepEqual(Array.from(B.data), bB, `${label} ${t}: src2 modified`); assert.equal(m.rows, t === '0x0' ? 0 : m.rows); } finally { o.delete(); }
    }
  } finally { for (const [, m] of ins) m.delete(); B.delete(); }
});

test('ip4b-T9. the 0x0 contract: every new op / operation returns a clone (0x0) + the info "ภาพว่าง"; the ops with values still return their keys (ecc_ok 0 / the rest null) and never throw', () => {
  const e = new cv.Mat();
  try {
    for (const [label, run] of allNewRuns()) {
      const ctx = {}; const Bw = waveGray(8, 8); const o = run(e, Bw, cv, ctx); Bw.delete();
      try { assert.equal(o.rows, 0, label); assert.equal(o.cols, 0, label); assert.equal(ctx.info, 'ภาพว่าง', label); } finally { o.delete(); }
    }
    const KEYS = { ecc_register: ECC_KEYS, gabor_filter: GABOR_KEYS, glcm: GLCM_KEYS, dehaze: DEHAZE_KEYS, dct_jpeg: DCT_KEYS };
    for (const [id, keys] of Object.entries(KEYS)) {
      const c2 = { src2: waveGray(8, 8) }; const o2 = OPS[id].apply(cv, e, {}, c2); assert.deepEqual(c2.values.fields.map((f) => f.key), keys, id);
      for (const f of c2.values.fields) assert.equal(f.value, id === 'ecc_register' && f.key === 'ecc_ok' ? 0 : null, `${id}.${f.key}`); o2.delete(); c2.src2.delete();
    }
  } finally { e.delete(); }
});

test('ip4b-T10. the edge-case table: 1x1 / flat / RGBA behave as specified for every new op (wiener / lucy work and keep the size; ecc guards "< 16 px"; gabor works; glcm: 0 pairs + text; nlm = the original; dehaze works; dct works (padded to 8x8); hit-or-miss isolated on a lone white pixel matches it (outside = do not care); regional on one pixel = white); RGBA -> 3 channels for wiener / lucy / nlm / dehaze; "constructor" in every new select = the default', () => {
  const one = gray(1, 1, () => 120), oneW = gray(1, 1, () => 255), flat = gray(20, 12, () => 77), rgba = toRGBA(noiseRGB(24, 16, 1));
  try {
    for (const id of ['deconv_wiener', 'deconv_lucy', 'gabor_filter', 'dehaze', 'dct_jpeg']) { const o = OPS[id].apply(cv, one, {}, {}); assert.equal(o.rows, 1, id); assert.equal(o.cols, 1); o.delete(); }
    { const ctx = { src2: one }, o = OPS.ecc_register.apply(cv, one, {}, ctx); assert.match(ctx.info, /ภาพเล็กเกินไป/); sameBytes(o, one, 'ecc 1x1'); o.delete(); }
    { const ctx = {}, o = OPS.glcm.apply(cv, one, {}, ctx); assert.equal(scalarMap(ctx).glcm_pairs, 0); assert.match(ctx.info, /ภาพเล็กกว่าระยะ 1 px/); assert.equal(o.rows, 256); o.delete(); }
    { const o = OPS.nlm_denoise.apply(cv, one, {}, {}); sameBytes(o, one, 'nlm 1x1'); o.delete(); }
    { const ctx = {}, o = OPS.morphology.apply(cv, oneW, MORPH('hit_or_miss'), ctx); assert.equal(o.data[0], 255); assert.match(ctx.info, /พบ 1 ตำแหน่ง/); o.delete(); }
    { const ctx = {}, o = OPS.morphology.apply(cv, one, MORPH('regional_max'), ctx); assert.equal(o.data[0], 255); o.delete(); }
    for (const id of ['deconv_wiener', 'deconv_lucy', 'nlm_denoise', 'dehaze']) { const o = OPS[id].apply(cv, rgba, id === 'nlm_denoise' ? { search: '7' } : {}, {}); assert.equal(o.channels(), 3, id); o.delete(); }
    // flat: wiener / lucy / nlm keep the picture (+-1)
    for (const id of ['deconv_wiener', 'deconv_lucy', 'nlm_denoise']) { const o = OPS[id].apply(cv, flat, id === 'nlm_denoise' ? { search: '7' } : {}, {}); assert.ok(Array.from(o.data).every((v) => Math.abs(v - 77) <= 1), id); o.delete(); }
    // "constructor" in every new select = the default (same output as the default params)
    const g = noiseGray(24, 18, 4), B = waveGray(24, 18);
    for (const [id, junk] of [['deconv_wiener', { psf: 'constructor' }], ['deconv_lucy', { psf: 'constructor' }], ['ecc_register', { motion: 'constructor', view: 'constructor' }], ['gabor_filter', { output: 'constructor' }], ['glcm', { levels: 'constructor', angle: 'constructor', view: 'constructor' }], ['nlm_denoise', { patch: 'constructor', search: 'constructor' }], ['dehaze', { view: 'constructor' }], ['dct_jpeg', { method: 'constructor', view: 'constructor' }]]) {
      const a = OPS[id].apply(cv, id === 'ecc_register' ? B : g, junk, { src2: B }), b = OPS[id].apply(cv, id === 'ecc_register' ? B : g, {}, { src2: B }); sameBytes(a, b, `${id} constructor`); a.delete(); b.delete();
    }
    g.delete(); B.delete();
  } finally { for (const m of [one, oneW, flat, rgba]) m.delete(); }
});

test('ip4b-T11. needsOf: the merged groups report gray for gabor / glcm / dct and any for the others; OP_PRIMARY exists for all 8 ops; the worker-facing contract: a gray op never receives a colour image through the merged FILTERS apply (the app converts) but still works on one when called directly', () => {
  for (const id of NEW_OPS) assert.equal(needsOf(FILTERS[HOST[id]], { mode: id }), NEEDS[id], id);
  for (const id of NEW_OPS) assert.equal(typeof OP_PRIMARY[id], 'function', `OP_PRIMARY.${id}`);
  const g = noiseGray(20, 20, 1), c = noiseRGB(20, 20, 1);
  for (const id of ['gabor_filter', 'glcm', 'dct_jpeg']) { const a = OPS[id].apply(cv, c, {}, {}); assert.ok(a.rows > 0); a.delete(); }
  const merged = FILTERS.analysis_detect.apply(cv, g, { ...defaultParams('analysis_detect'), mode: 'gabor_filter', gabor_filter__wavelength: 6 }, {}), direct = OPS.gabor_filter.apply(cv, g, { wavelength: 6 }, {}); sameBytes(merged, direct, 'merged apply routes the prefixed params'); merged.delete(); direct.delete();
  const mm = FILTERS.image_arithmetic.apply(cv, g, { ...defaultParams('image_arithmetic'), mode: 'ecc_register', ecc_register__view: 'diff' }, { src2: g }); assert.equal(mm.channels(), 1); mm.delete();
  g.delete(); c.delete();
});

// ===================================================================
// parity กับ Python จริง (cv2 5.0 / numpy 2.x) — หลายกรณีต่อ process เดียว · ข้ามเมื่อไม่มี python/cv2 (IP4B_NO_PY=1 จำลอง)
// ===================================================================
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'ip-batch4B-'));
process.on('exit', () => { try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ } });
function pythonAvailable() { try { execFileSync('python', ['-c', 'import cv2, numpy'], { encoding: 'utf8' }); return true; } catch { return false; } }
const HAS_PYTHON = !process.env.IP4B_NO_PY && pythonAvailable();
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
const grayRGB = (g) => { const o = new cv.Mat(); cv.cvtColor(g, o, cv.COLOR_GRAY2RGB); return o; };
const pyVals = (stdout, tag) => { const line = stdout.split('\n').find((l) => l.startsWith(tag + ' ')); assert.ok(line, `no ${tag} line in: ${stdout.slice(0, 300)}`); return line.slice(tag.length + 1).trim().split(/\s+/).map((s) => (s === 'None' ? null : s === 'inf' ? Infinity : Number(s))); };
const PYNUM = (e) => `repr(float(${e}))`;
// 8x8 blocks (key "bx,by") of the gray version of an RGB image where some quantized coefficient z = F / Qs is within 1e-6 of a rounding tie
function tieBlocksOf(m, quality) {
  const g = new cv.Mat(); cv.cvtColor(m, g, cv.COLOR_RGB2GRAY);
  const W = g.cols, H = g.rows, Q = jpegQTable(quality), out = new Set(), B = new Float64Array(64), F = new Float64Array(64);
  for (let by = 0; by < Math.ceil(H / 8); by++) for (let bx = 0; bx < Math.ceil(W / 8); bx++) {
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) B[y * 8 + x] = g.data[Math.min(H - 1, by * 8 + y) * W + Math.min(W - 1, bx * 8 + x)] - 128;
    dct8Forward(B, F);
    for (let i = 0; i < 64; i++) if (Math.abs(Math.abs(F[i] / Q[i]) % 1 - 0.5) < 1e-6) out.add(`${bx},${by}`);
  }
  g.delete(); return out;
}
const PY_IMGS = () => [['97x61', noiseRGB(97, 61, 21)], ['64x48', smoothRGB(64, 48)], ['1x1', (() => rgb(1, 1, () => [200, 90, 40]))()]];

test('ip4b-P1. parity deconv_wiener (exported Python == app): max |diff| <= 2 levels and >= 99% of the pixels equal on 97x61 noise / 200x150 blurred checkerboard / 1x1, motion + disk PSF, NSR 0.01 / 0.001 / 0.2, colour and gray; NSR = 0 only has to run without NaN / error (the DFT of cv2 5.0 and 4.10 differ by ulps which Wiener amplifies)', PY_SKIP, () => {
  const sharp = grayRGB(checkerboard(200, 150, 10)), blurred = (() => { const b = blurWith(sharp, 'motion', 9, 0); return b; })();
  const imgs = [...PY_IMGS(), ['blurred200', blurred]];
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const p of [{}, { psf: 'disk', psfSize: 5, nsr: 0.001 }, { psfSize: 15, psfAngle: 45, nsr: 0.2 }, { nsr: 0 }]) jobs.push({ name, m, p, pipe: [OPSTEP('deconv_wiener', p)] });
    jobs.push({ name: 'gray97', m: imgs[0][1], p: { nsr: 0.01 }, pipe: [STEP('color_tools', 'grayscale'), OPSTEP('deconv_wiener', { nsr: 0.01 })] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        if (j.p.nsr === 0) { assert.ok(py[i].bytes.length === app.mat.data.length); assert.ok(Array.from(py[i].bytes).every((v) => v >= 0 && v <= 255), `${j.name} nsr 0 stays in range`); return; }
        const { n, mx } = diffStats(app.mat, py[i].bytes);
        assert.ok(mx <= 2 && n <= app.mat.data.length * 0.01, `${j.name} ${JSON.stringify(j.p)}: ${n}/${app.mat.data.length} differ (max ${mx})`);
        assert.match(py[i].stdout, /wiener deconvolution: psf (motion \d+ px \d+ deg|disk \d+ px), NSR [\d.]+/);
      } finally { app.mat.delete(); }
    });
  } finally { for (const [, m] of imgs) m.delete(); sharp.delete(); }
});

test('ip4b-P2. parity deconv_lucy: max |diff| <= 2 levels and >= 99% equal on 97x61 noise / 64x48 smooth / 1x1, motion + disk, 1 / 10 / 20 rounds, colour and gray (filter2D of 5.0 vs 4.10 differ by ulps that accumulate over the rounds)', PY_SKIP, () => {
  const imgs = PY_IMGS();
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const p of [{}, { psf: 'disk', psfSize: 4, iterations: 20 }, { iterations: 1, psfAngle: 30, psfSize: 11 }]) jobs.push({ name, m, p, pipe: [OPSTEP('deconv_lucy', p)] });
    jobs.push({ name: 'gray97', m: imgs[0][1], p: {}, pipe: [STEP('color_tools', 'grayscale'), OPSTEP('deconv_lucy', {})] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        const { n, mx } = diffStats(app.mat, py[i].bytes);
        assert.ok(mx <= 2 && n <= app.mat.data.length * 0.01, `${j.name} ${JSON.stringify(j.p)}: ${n}/${app.mat.data.length} differ (max ${mx})`);
        assert.match(py[i].stdout, /lucy-richardson: \d+ iterations, psf (motion \d+ px \d+ deg|disk \d+ px)/);
      } finally { app.mat.delete(); }
    });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4b-P3. parity ecc_register: on analytic image pairs with a KNOWN warp the exported Python agrees with the app: |d dx|, |d dy| <= 0.05 px, |d angle| <= 0.01 deg, |d scale| and |d rho| <= 1e-3, ecc_ok equal, the aligned image differs by more than 2 levels in <= 2% of the pixels (all 4 motions; maxSide 160 shrinking); flat / tiny / non-convergent pairs fail the same way in both (A unchanged)', PY_SKIP, () => {
  const mkpair = (w, h, mapper) => { const [a, b] = eccPair(w, h, mapper); const A = grayRGB(a), B = grayRGB(b); a.delete(); b.delete(); return [A, B]; };
  const cases = [
    ['euclid320', ...mkpair(320, 240, eucMap(2.9, 5.5, -3.25)), { motion: 'euclidean' }],
    ['affine', ...mkpair(200, 150, eucMap(1.5, 3.25, -2.5)), { motion: 'affine' }],
    ['translation', ...mkpair(200, 150, (x, y) => [x + 4.5, y - 2.75]), { motion: 'translation' }],
    ['homography', ...mkpair(200, 150, (x, y) => { const d = 4e-4 * x - 3e-4 * y + 1; return [(x + 3) / d, (y - 2) / d]; }), { motion: 'homography', iterations: 100 }],
    ['euclid shrunk 160', ...mkpair(320, 240, eucMap(2.9, 5.5, -3.25)), { motion: 'euclidean', maxSide: 160 }],
    ['shrunk to a non-whole size (320x237 -> 160x119 vs 160x118)', ...mkpair(320, 237, eucMap(2.9, 5.5, -3.25)), { motion: 'euclidean', maxSide: 160 }],
    ['overlay', ...mkpair(160, 120, eucMap(1.2, 2.5, -1.75)), { view: 'overlay' }],
    ['diff', ...mkpair(160, 120, eucMap(1.2, 2.5, -1.75)), { view: 'diff' }],
    ['flat', grayRGB(gray(40, 30, () => 90)), grayRGB(waveGray(40, 30)), {}],
    ['tiny', grayRGB(waveGray(10, 10)), grayRGB(waveGray(10, 10)), {}],
    ['noise', grayRGB(noiseGray(40, 30, 5)), grayRGB(noiseGray(40, 30, 6)), { iterations: 10 }],
    ['resized B', ...(() => { const [a, b] = mkpair(120, 90, eucMap(1.5, 3.25, -2.5)); const bs = resizeTo(b, 60, 45); b.delete(); return [a, bs]; })(), {}],
  ];
  try {
    const tail = "print('ECC', ecc_ok, *( [repr(float(rho)), repr(float(warp[0, 2])), repr(float(warp[1, 2])), repr(float(np.arctan2(warp[1, 0], warp[0, 0]) * 180 / np.pi)), repr(float(np.sqrt(abs(float(warp[0, 0]) * float(warp[1, 1]) - float(warp[0, 1]) * float(warp[1, 0])))))] if ecc_ok else [] ))";
    const jobs = cases.map(([name, A, B, p]) => ({ name, A, B, p, pipe: [OPSTEP('ecc_register', p)] }));
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.A, j.B], tail })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.A, j.B);
      try {
        const v = scalarMap(app.ctx), pv = py[i].stdout.split('\n').find((l) => l.startsWith('ECC ')).split(' ').slice(1);
        assert.equal(pv[0] === 'True' ? 1 : 0, v.ecc_ok, `${j.name}: ecc_ok`);
        if (v.ecc_ok) {
          const [rho, dx, dy, ang, sc] = pv.slice(1).map(Number);
          near(dx, v.ecc_dx, 0.05, `${j.name} dx`); near(dy, v.ecc_dy, 0.05, `${j.name} dy`); near(ang, v.ecc_angle_deg, 0.01, `${j.name} angle`); near(sc, v.ecc_scale, 1e-3, `${j.name} scale`); near(rho, v.ecc_rho, 1e-3, `${j.name} rho`);
          assert.match(py[i].stdout, /ecc (translation|euclidean|affine|homography): dx -?[\d.]+ dy -?[\d.]+ angle -?[\d.]+ scale [\d.]+ rho [\d.]+/);
        } else assert.match(py[i].stdout, /ecc \w+: failed \((image too small|image too flat|did not converge)/);
        // image
        const A3 = app.mat.channels() === 3 ? app.mat : (() => { const t = new cv.Mat(); cv.cvtColor(app.mat, t, cv.COLOR_GRAY2RGB); return t; })();
        const d = A3.data, pb = py[i].bytes; assert.equal(pb.length, d.length, `${j.name}: size`);
        let big = 0; for (let k = 0; k < d.length; k++) if (Math.abs(d[k] - pb[k]) > 2) big++;
        assert.ok(big <= d.length * 0.02, `${j.name}: ${big}/${d.length} pixels differ by > 2 levels`);
        if (A3 !== app.mat) A3.delete();
      } finally { app.mat.delete(); }
    });
  } finally { for (const j of cases) { j[1].delete(); j[2].delete(); } }
});

test('ip4b-P4. parity gabor_filter: cv2.getGaborKernel == gaborKernel for 40 parameter sets (every float32 value), and the exported Python agrees with the app: image <= 1 level (max / mean of the response within 1e-4 relative); a flat image is all 0 (magnitude) / 128 (real) in both', PY_SKIP, () => {
  // (a) kernel equality with the real cv2
  const sets = []; for (const k of [5, 11, 21, 55, 99]) for (const [s, t, l, g, p] of [[2, 0, 6, 0.5, 0], [4.5, 30, 8, 0.7, Math.PI / 2], [7, 135, 12, 0.25, 0], [3.3, 77, 5, 1, Math.PI / 2]]) sets.push([k, s, t, l, g, p]);
  const script = `import cv2, json, numpy as np\nout = []\nfor k, s, t, l, g, p in ${JSON.stringify(sets)}:\n    ke = cv2.getGaborKernel((k, k), s, t * np.pi / 180, l, g, p, ktype=cv2.CV_32F)\n    out.append([float(v) for v in ke.ravel()])\nprint(json.dumps(out))\n`;
  const f = path.join(TMP_DIR, `gk_${Math.random().toString(36).slice(2)}.py`); fs.writeFileSync(f, script, 'utf8');
  const res = JSON.parse(execFileSync('python', [f], { encoding: 'utf8', maxBuffer: 1 << 28 }));
  sets.forEach(([k, s, t, l, g, p], i) => { const js = gaborKernel(k, s, t, l, g, p); assert.equal(js.length, res[i].length); for (let j = 0; j < js.length; j++) assert.equal(js[j], res[i][j], `kernel ${k}/${s}/${t}/${l}/${g}/${p} value ${j}`); });
  // (b) op parity
  const imgs = [['97x61', noiseRGB(97, 61, 21)], ['64x48', smoothRGB(64, 48)], ['flat', rgb(30, 20, () => [100, 100, 100])]];
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const p of [{}, { wavelength: 5, orientation: 45, bandwidth: 2, aspect: 0.8, output: 'real' }, { wavelength: 12, orientation: 120, output: 'real' }, { wavelength: 64, aspect: 0.2, bandwidth: 0.5 }]) jobs.push({ name, m, p, pipe: [OPSTEP('gabor_filter', p)] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m], tail: "print('GB', repr(float(mean_resp)), repr(float(mx)))" })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        const { mx } = diffStats(app.mat, py[i].bytes); assert.ok(mx <= 1, `${j.name} ${JSON.stringify(j.p)}: max diff ${mx}`);
        const [pm, px] = pyVals(py[i].stdout, 'GB'), v = scalarMap(app.ctx);
        if (j.name === 'flat') { assert.equal(v.gabor_max, 0); assert.equal(px, 0); assert.ok(Array.from(py[i].bytes).every((b) => b === (j.p.output === 'real' ? 128 : 0))); }
        else { near(px, v.gabor_max, Math.abs(v.gabor_max) * 1e-4, 'max'); near(pm, v.gabor_mean, Math.abs(v.gabor_mean) * 1e-4, 'mean'); }
      } finally { app.mat.delete(); }
    });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4b-P5. parity glcm: the pair COUNTS are exactly equal (every direction / distance / symmetric / levels, incl. avg), the properties agree within 1e-12 relative, the 256x256 table within 1 level (log1p), the quantized view exactly; a distance bigger than the image gives 0 pairs in both', PY_SKIP, () => {
  const imgs = [['97x61', noiseRGB(97, 61, 21)], ['64x48', smoothRGB(64, 48)], ['5x4', rgb(5, 4, (x, y) => { const v = (GL_IMG[y][x] - 1) * 32 + 16; return [v, v, v]; })], ['3x3', noiseRGB(3, 3, 9)]];
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const p of [{}, { angle: '45', distance: 3, levels: '16' }, { angle: '90', symmetric: true, levels: '64' }, { angle: '135', distance: 2, levels: '2' }, { angle: 'avg', symmetric: true, distance: 2 }, { view: 'quantized', levels: '32' }, { distance: 10, angle: 'avg' }]) jobs.push({ name, m, p, pipe: [OPSTEP('glcm', p)] });
    const tail = "print('CNT', json.dumps([mm.astype(np.int64).ravel().tolist() for mm in mats]))\nprint('PROPS', json.dumps([None if pr is None else [None if v is None else float(v) for v in pr] for pr in props]))";
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe).replace('import sys\n', 'import sys\nimport json\n'), srcs: [j.m], tail })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        const g = (() => { const t = new cv.Mat(); cv.cvtColor(j.m, t, cv.COLOR_RGB2GRAY); return t; })();
        const L = Number(j.p.levels ?? 8), d = j.p.distance ?? 1, angle = j.p.angle ?? '0', q = Uint8Array.from(g.data, (v) => Math.min(L - 1, Math.floor(v * L / 255)));
        const OFF = { 0: [d, 0], 45: [d, -d], 90: [0, -d], 135: [-d, -d] };
        const mine = (angle === 'avg' ? [0, 45, 90, 135] : [Number(angle)]).map((a) => { let C = glcmCounts(q, g.cols, g.rows, L, OFF[a][0], OFF[a][1]); if (j.p.symmetric) { const T = new Float64Array(L * L); for (let x = 0; x < L; x++) for (let y = 0; y < L; y++) T[x * L + y] = C[x * L + y] + C[y * L + x]; C = T; } return C; });
        const cnt = JSON.parse(py[i].stdout.split('\n').find((l) => l.startsWith('CNT ')).slice(4)), props = JSON.parse(py[i].stdout.split('\n').find((l) => l.startsWith('PROPS ')).slice(6));
        assert.deepEqual(cnt, mine.map((C) => Array.from(C)), `${j.name} ${JSON.stringify(j.p)} counts`);
        mine.forEach((C, k) => { const pj = glcmProps(C, L), pp = props[k]; if (!pj) { assert.equal(pp, null); return; } [['contrast', 0], ['correlation', 1], ['energy', 2], ['homogeneity', 3]].forEach(([key, idx]) => { if (pj[key] === null) assert.equal(pp[idx], null, key); else near(pj[key], pp[idx], Math.abs(pj[key]) * 1e-12 + 1e-15, `${j.name} ${key}`); }); });
        const { mx } = diffStats(app.mat, py[i].bytes); assert.ok(mx <= (j.p.view === 'quantized' ? 0 : 1), `${j.name} ${JSON.stringify(j.p)} image max diff ${mx}`);
        const v = scalarMap(app.ctx), total = mine.reduce((s, C) => s + C.reduce((a, b) => a + b, 0), 0); assert.equal(v.glcm_pairs, total);
        if (total === 0) assert.match(py[i].stdout, /glcm: no pixel pairs/); else assert.match(py[i].stdout, /glcm levels \d+ dist \d+ angle (0|45|90|135|avg): contrast/);
        g.delete();
      } finally { app.mat.delete(); }
    });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4b-P6. parity nlm_denoise (the app formula in Python): >= 99.9% of the pixels equal and max |diff| <= 1 on 64x48 colour / gray / a 20x12 patch 3 search 7 case / an image shrunk from 700 px; 1x1 equal', PY_SKIP, () => {
  const wide = (() => { const d = new Uint8Array(700 * 6 * 3); const r = lcg(3); for (let i = 0; i < d.length; i++) d[i] = (Math.floor(r() * 40) + (i % 53)) & 255; return cv.matFromArray(6, 700, cv.CV_8UC3, d); })();
  const imgs = [['64x48', smoothRGB(64, 48)], ['noise 40x30', noiseRGB(40, 30, 5)], ['1x1', rgb(1, 1, () => [200, 90, 40])], ['wide', wide]];
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const p of [{ search: '7' }, { h: 25, patch: '3', search: '7' }, { patch: '7', search: '11' }]) { if (name === 'wide' && p.search === '11') continue; jobs.push({ name, m, p, pipe: [OPSTEP('nlm_denoise', p)] }); }
    jobs.push({ name: 'gray64', m: imgs[0][1], p: { search: '7' }, pipe: [STEP('color_tools', 'grayscale'), OPSTEP('nlm_denoise', { search: '7' })] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        const { n, mx } = diffStats(app.mat, py[i].bytes);
        assert.ok(mx <= 1 && n <= app.mat.data.length * 0.001, `${j.name} ${JSON.stringify(j.p)}: ${n}/${app.mat.data.length} differ (max ${mx})`);
        assert.match(py[i].stdout, /non-local means: h \d+, patch [357], search (7|11|15)/);
      } finally { app.mat.delete(); }
    });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4b-P7. parity dehaze: with the guided filter OFF the image is byte-identical (ndiff 0) and A equal exactly; with it ON <= 1 level in <= 0.5% of the pixels; transmission / dark views the same; the printed A / t_mean carry the app rounding', PY_SKIP, () => {
  const hz = hazyScene();
  const imgs = [['hazy 64x48', hz.I], ['noise 40x30', noiseRGB(40, 30, 5)], ['smooth', smoothRGB(33, 21)], ['1x1', rgb(1, 1, () => [200, 90, 40])]];
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const p of [{ refine: false }, { refine: false, patch: 7, amount: 0.5 }, { refine: true, patch: 7 }, { view: 'transmission', refine: false }, { view: 'dark', refine: false, patch: 5 }, { view: 'transmission' }]) jobs.push({ name, m, p, pipe: [OPSTEP('dehaze', p)] });
    jobs.push({ name: 'gray', m: imgs[0][1], p: { refine: false }, pipe: [STEP('color_tools', 'grayscale'), OPSTEP('dehaze', { refine: false })] }, { name: 'gray refine', m: imgs[0][1], p: {}, pipe: [STEP('color_tools', 'grayscale'), OPSTEP('dehaze', {})] });
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m], tail: "print('DH', *[repr(float(v)) for v in a3], repr(float(t_mean)))" })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        const { n, mx } = diffStats(app.mat, py[i].bytes), exact = j.p.refine === false;
        if (exact) assert.equal(n, 0, `${j.name} ${JSON.stringify(j.p)}: ${n} px differ (max ${mx})`); else assert.ok(mx <= 1 && n <= app.mat.data.length * 0.005, `${j.name} ${JSON.stringify(j.p)}: ${n}/${app.mat.data.length} differ (max ${mx})`);
        const [ar, ag, ab, tm] = pyVals(py[i].stdout, 'DH'), v = scalarMap(app.ctx);
        if (exact) { assert.equal(ar, v.dh_a_r); assert.equal(ag, v.dh_a_g); assert.equal(ab, v.dh_a_b); near(tm, v.dh_t_mean, 1e-12, 't mean'); } else { near(ar, v.dh_a_r, 1e-9, 'A r'); near(tm, v.dh_t_mean, 2e-3, 't mean (refined)'); }
        assert.ok(py[i].stdout.includes(`dehaze: A (R/G/B) ${fixed1(v.dh_a_r)}/${fixed1(v.dh_a_g)}/${fixed1(v.dh_a_b)}`), `A text: ${py[i].stdout.slice(0, 120)}`);
        if (exact) assert.ok(py[i].stdout.includes(`t_mean ${fixed2(v.dh_t_mean)}`), 't_mean text');
      } finally { app.mat.delete(); }
    });
  } finally { hz.J.delete(); for (const [, m] of imgs) m.delete(); }
});

test('ip4b-P8. parity dct_jpeg: every pixel equal except inside 8x8 blocks that hold a coefficient within 1e-6 of an exact rounding tie (the order of the floating-point sums decides those), max |diff| <= 3, on 97x61 noise / 64x48 smooth / 13x7 / 1x1 at quality 10 / 50 / 90 and zonal K 1 / 4 / 8; dct_zero_pct within 0.05, PSNR within 0.05 dB; zonal K = 8 equals the input +-1; the views coeffs / error agree', PY_SKIP, () => {
  const imgs = [['97x61', noiseRGB(97, 61, 21)], ['64x48', smoothRGB(64, 48)], ['13x7', rgb(13, 7, (x, y) => [x * 20 & 255, y * 30 & 255, 90])], ['1x1', rgb(1, 1, () => [200, 90, 40])]];
  try {
    const jobs = [];
    for (const [name, m] of imgs) for (const p of [{ quality: 10 }, { quality: 50 }, { quality: 90 }, { method: 'zonal', keep: 1 }, { method: 'zonal', keep: 4 }, { method: 'zonal', keep: 8 }, { view: 'coeffs' }, { view: 'error', quality: 25 }]) jobs.push({ name, m, p, pipe: [OPSTEP('dct_jpeg', p)] });
    const tail = "print('DCT', repr(float(psnr)), repr(float(n_zero * 100 / ((hp // 8) * (wp // 8) * 64))))";
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m], tail })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        const tot = app.mat.data.length, W = app.mat.cols, ties = j.p.method === 'zonal' || j.p.view === 'coeffs' ? new Set() : tieBlocksOf(j.m, j.p.quality ?? 50), pb = py[i].bytes;
        let outside = 0, inside = 0, mx = 0; for (let k = 0; k < tot; k++) { const dd = Math.abs(app.mat.data[k] - pb[k]); if (!dd) continue; mx = Math.max(mx, dd); if (ties.has(`${(k % W) >> 3},${Math.floor(k / W) >> 3}`) && j.p.view !== 'error') inside++; else outside++; }
        // a coefficient that lands within an ulp of an exact rounding tie (z = k + 0.5) is decided by the order of the floating-point sums, so a differing pixel is only acceptable inside such a block
        assert.ok(outside === 0 && mx <= 3 && inside <= tot * 0.02, `${j.name} ${JSON.stringify(j.p)}: ${outside} px differ outside tie blocks (+${inside} inside, max ${mx})`);
        const [pp, pz] = pyVals(py[i].stdout, 'DCT'), v = scalarMap(app.ctx);
        near(pz, v.dct_zero_pct, 0.05, `${j.name} zero %`); if (Number.isFinite(v.dct_psnr)) near(pp, v.dct_psnr, 0.05, `${j.name} PSNR`); else assert.equal(pp, Infinity);
        if (j.p.method === 'zonal' && j.p.keep === 8) { const g = new cv.Mat(); cv.cvtColor(j.m, g, cv.COLOR_RGB2GRAY); const d = diffStats(g, py[i].bytes); assert.ok(d.mx <= 1, `zonal K = 8 returns the input (max ${d.mx})`); g.delete(); }
      } finally { app.mat.delete(); }
    });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4b-P9. parity Morphology hit_or_miss / regional_max / regional_min / h_maxima / extended_max: ndiff 0 on 97x61 / 40x30 / 1x1 images (random binary, waves, plateaus) for every pattern x rotate and h 1 / 10 / 60; the printed counts equal the app\'s', PY_SKIP, () => {
  const wv = (() => { const g = waveGray(48, 36), o = grayRGB(g); g.delete(); return o; })();
  const plateau = (() => { const g = gray(40, 30, (x, y) => ((x >> 2) * 31 + (y >> 2) * 17) & 255), o = grayRGB(g); g.delete(); return o; })();
  const imgs = [['97x61', noiseRGB(97, 61, 21)], ['wave', wv], ['plateau', plateau], ['1x1', rgb(1, 1, () => [200, 90, 40])]];
  try {
    const jobs = [];
    for (const [name, m] of imgs) {
      for (const hmPattern of Object.keys(HITMISS_PATTERNS)) for (const hmRotate of [true, false]) jobs.push({ name, m, p: { op: 'hit_or_miss', hmPattern, hmRotate }, pipe: [OPSTEP('morphology', { op: 'hit_or_miss', hmPattern, hmRotate })] });
      for (const op of ['regional_max', 'regional_min']) jobs.push({ name, m, p: { op }, pipe: [OPSTEP('morphology', { op })] });
      for (const op of ['h_maxima', 'extended_max']) for (const hHeight of [1, 10, 60]) jobs.push({ name, m, p: { op, hHeight }, pipe: [OPSTEP('morphology', { op, hHeight })] });
    }
    const py = runPythonBatch(jobs.map((j) => ({ code: buildPythonCode(j.pipe), srcs: [j.m] })));
    jobs.forEach((j, i) => {
      const app = appRun(j.pipe, j.m);
      try {
        const { n, mx } = diffStats(app.mat, py[i].bytes); assert.equal(n, 0, `${j.name} ${JSON.stringify(j.p)}: ${n} px differ (max ${mx})`);
        const cnt = (re) => { const m = py[i].stdout.match(re); assert.ok(m, `stdout: ${py[i].stdout.slice(0, 120)}`); return m; };
        if (j.p.op === 'hit_or_miss') assert.equal(Number(cnt(/hit-or-miss \w+: (\d+) matches/)[1]), cv.countNonZero(app.mat));
        else if (j.p.op === 'regional_max' || j.p.op === 'regional_min' || j.p.op === 'extended_max') { const m = cnt(/regions \((\d+) px\)/); assert.equal(Number(m[1]), cv.countNonZero(app.mat)); }
        else cnt(/h-maxima h=\d+: \d+ regions left/);
        if (j.p.op !== 'hit_or_miss' && j.p.op !== 'h_maxima') { // the region count printed by Python == the count in the app's info
          const r = Number(cnt(/: (\d+) regions/)[1]), g = new cv.Mat(); cv.cvtColor(j.m, g, cv.COLOR_RGB2GRAY); const c = {}; const o = OPS.morphology.apply(cv, g, MORPH(j.p.op, { hHeight: j.p.hHeight ?? 10 }), c); o.delete(); g.delete();
          assert.ok(c.info.includes(`พบ ${r} บริเวณ`), `regions ${r} vs ${c.info}`);
        }
      } finally { app.mat.delete(); }
    });
  } finally { for (const [, m] of imgs) m.delete(); }
});

test('ip4b-P10. JS-only twins of the Python parity bugs (these must fail even when python / cv2 is missing): the Python source of every new op is checked line by line against the JS constants it must mirror (rounding, clamps, orders, PSF / guard text), and the helper twins agree with the JS helper on small cases by an independent re-implementation of the Python formulas in JS', () => {
  const k = (id, p) => buildPythonCode([OPSTEP(id, p)]);
  // wiener: psf2otf roll amount, reflect border, conj formula, denominators, rounding
  const w = k('deconv_wiener', { psfSize: 9, nsr: 0.05 });
  for (const t of ['ph = np.roll(ph, (-hc, -hc), axis=(0, 1))', 'cv2.copyMakeBorder(ch, n, hp - hh - n, n, wp - ww - n, cv2.BORDER_REFLECT)', 'den = hr * hr + hi * hi + nsr', 're = np.where(ok, (gr * hr + gi * hi) / safe, 0.0)', 'im = np.where(ok, (gi * hr - gr * hi) / safe, 0.0)', 'nsr = 0.05', 'cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT', 'inv[n:n + hh, n:n + ww]', 'np.floor(', '+ 0.5)']) assert.ok(w.includes(t), `wiener python: ${t}`);
  // lucy: flipped kernel for the convolution step, plain kernel for the adjoint, ratio g / c, reflect-101, the same threshold
  const l = k('deconv_lucy', { iterations: 7 });
  for (const t of ['kf = np.ascontiguousarray(np.flip(psf, (0, 1)).astype(np.float32))', 'kk = psf.astype(np.float32)', 'c = cv2.filter2D(est, -1, kf, borderType=cv2.BORDER_REFLECT_101)', 'corr = cv2.filter2D(r, -1, kk, borderType=cv2.BORDER_REFLECT_101)', 'ok = c > 1e-6', 'g / np.where(ok, c, np.float32(1))', 'est = (est * corr).astype(np.float32)', 'iters = 7']) assert.ok(l.includes(t), `lucy python: ${t}`);
  // ecc: warp order template = B (sb), input = A (sa), 7 arguments, INVERSE_MAP, scale-back, guards
  const e = k('ecc_register', { motion: 'homography' });
  for (const t of ['cv2.findTransformECC(sb, sa, warp, cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-5), None, 5)', 'warp[0, 2] = np.float32(float(warp[0, 2]) / scale)', 'warp[1, 2] = np.float32(float(warp[1, 2]) / scale)', 'warp[2, 0] = np.float32(float(warp[2, 0]) * scale)', 'warp[2, 1] = np.float32(float(warp[2, 1]) * scale)', 'flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP', 'cv2.warpPerspective(img, warp, (ww, hh)', 'min(sa.shape) < 16', 'sa.std() < 1 or sb.std() < 1', 'except cv2.error:', 'cv2.INTER_AREA']) assert.ok(e.includes(t), `ecc python: ${t}`);
  const e2 = k('ecc_register', { motion: 'euclidean' }); assert.ok(e2.includes('warp[0, 2] = np.float32(float(warp[0, 2]) / scale)') && !e2.includes('warp[2, 0]') && e2.includes('cv2.warpAffine(img, warp, (ww, hh)'));
  // gabor: formulas in the same order as JS (sigma, half, mean removal in double with an ordered sum, psi = pi / 2, magnitude), the noise floor
  const g = k('gabor_filter', { wavelength: 6, orientation: 30, bandwidth: 1.5, aspect: 0.6 });
  for (const t of ['theta = 30 * np.pi / 180', 'sigma = lambd / np.pi * np.sqrt(np.log(2) / 2) * (2 ** bw + 1) / (2 ** bw - 1)', 'half = min(49, int(np.ceil(3 * sigma / gamma)))', 'np.cumsum(ke.astype(np.float64).ravel())[-1] / ke.size', 'cv2.getGaborKernel((k, k), sigma, theta, lambd, gamma, np.pi / 2, ktype=cv2.CV_32F)', 'resp = cv2.magnitude(re, im)', 'none = (not (mx > 0)) or int(img.min()) == int(img.max())', 'np.floor(resp64 * 255 / mx + 0.5)']) assert.ok(g.includes(t), `gabor python: ${t}`);
  assert.ok(k('gabor_filter', { output: 'real' }).includes('np.floor(128 + resp64 * 127 / mx + 0.5)'));
  // the old 0.05 noise floor is gone: a flat image is detected from its pixels (min == max) on both sides
  const JS = readSrc('filters/registry.js'); assert.ok(!JS.includes('GABOR_NOISE_FLOOR'));
  // lucy: the final round-half-up (a mutant dropping `+ 0.5` survived without cv2)
  assert.ok(k('deconv_lucy', {}).includes('return np.clip(np.floor(est.astype(np.float64) + 0.5), 0, 255).astype(np.uint8)'));
  // JS-only pins of Python lines whose mutants survived without cv2 (reviewer M13-M16): NLM patch-sum corners, DCT block un-transpose, GLCM contrast, ECC rounded shrink size
  assert.ok(k('nlm_denoise', {}).includes('s = integ[y_hi + 1][:, x_hi + 1] - integ[y_lo][:, x_hi + 1] - integ[y_hi + 1][:, x_lo] + integ[y_lo][:, x_lo]'));
  assert.ok(k('dct_jpeg', {}).includes('.transpose(0, 2, 1, 3).reshape(hp, wp).astype(np.uint8)[:hh, :ww]') && k('dct_jpeg', {}).includes('blocks = (pad.astype(np.float64) - 128).reshape(hp // 8, 8, wp // 8, 8).transpose(0, 2, 1, 3)'));
  assert.ok(k('glcm', {}).includes('contrast = ((i - j) ** 2 * pr).sum()') && k('glcm', {}).includes('energy = (pr * pr).sum()') && k('glcm', {}).includes('homog = (pr / (1 + np.abs(i - j))).sum()'));
  assert.ok(k('ecc_register', {}).includes('aw, ah = max(1, int(np.floor(ww * scale + 0.5))), max(1, int(np.floor(hh * scale + 0.5)))'));
  // glcm: offsets in (dx, dy) form, quantization, symmetric transpose, avg of the same four values
  const gl = k('glcm', { angle: 'avg', distance: 2, symmetric: true });
  for (const t of ['offsets = [(2, 0), (2, -2), (0, -2), (-2, -2)]', 'q = np.minimum(levels - 1, (img.astype(np.int64) * levels) // 255)', 'c = c + c.T', 'a = q[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]', 'b = q[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]', 'pairs = sum(float(m.sum()) for m in mats)', 'np.log1p(max(cmax, 1))']) assert.ok(gl.includes(t), `glcm python: ${t}`);
  // nlm: the same loop nest and expression order as nlmCore
  const n = k('nlm_denoise', { h: 12, patch: '7', search: '15' });
  for (const t of ['h = 12', 'pr = 3', 'sr = 7', 'for dy in range(-sr, sr + 1):', 'for dx in range(-sr, sr + 1):', 'w = np.exp(-(s / area) / (h * h))', 'acc += w[:, :, None] * nb', 'np.clip(yy0 + dy, 0, hh - 1)', 'np.cumsum(np.cumsum(d2, axis=1), axis=0)', 'area = ((y_hi - y_lo + 1)[:, None] * (x_hi - x_lo + 1)[None, :] * nc).astype(np.float64)']) assert.ok(n.includes(t), `nlm python: ${t}`);
  assert.ok(n.indexOf('for dy in range') < n.indexOf('for dx in range'), 'dy is the OUTER loop (the accumulation order of nlmCore)');
  // dehaze
  const d = k('dehaze', { refine: true, patch: 9, amount: 0.8 });
  for (const t of ['omega = 0.8', 'cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))', 'top = max(1, n_px // 1000)', 't_dark = 255 - int(np.argmax(np.cumsum(hist[::-1]) >= top))', 'np.maximum(x[sel].sum(axis=0, dtype=np.int64) / int(sel.sum()), 1)', 'mn = (x.astype(np.float64) / atm).min(axis=2).astype(np.float32)', 't = (1 - omega * dn.astype(np.float64)).astype(np.float32)', 'guided_f32(gray.astype(np.float32), pp, 18, np.float32(65.025))', 't = np.maximum(t.astype(np.float64), 0.1)', 'np.floor((x.astype(np.float64) - atm) / t[:, :, None] + atm + 0.5)', 'def guided_f32(']) assert.ok(d.includes(t), `dehaze python: ${t}`);
  assert.ok(!k('dehaze', { refine: false }).includes('def guided_f32') && k('dehaze', { refine: false }).includes('t = np.maximum(t.astype(np.float64), 0.1)'));
  // dct: the IJG scale / ordered rounding / table values
  const c = k('dct_jpeg', { quality: 30 });
  for (const t of ['s = 5000 // quality if quality < 50 else 200 - 2 * quality', 'qs = np.maximum(1, (q50 * s + 50) // 100)', 'qf = np.sign(z) * np.floor(np.abs(z) + 0.5)', 'quality = 30', "np.einsum('ui,abij,vj->abuv', c, blocks, c)", "np.einsum('ui,abuv,vj->abij', c, fq, c) + 128", 'cv2.BORDER_REPLICATE', '16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55']) assert.ok(c.includes(t), `dct python: ${t}`);
  assert.ok(k('dct_jpeg', { method: 'zonal', keep: 3 }).includes('keep = 3') && k('dct_jpeg', { method: 'zonal', keep: 3 }).includes('n_zero = (64 - keep * keep) * (hp // 8) * (wp // 8)'));
  // morphology: kernels / rotation direction / OR / int32 / the reconstruction by definition
  const hm = k('morphology', { op: 'hit_or_miss', hmPattern: 'corner' });
  for (const t of ['k = np.array([[0,-1,-1],[1,1,-1],[0,1,0]], np.int32)', 'ks.append(np.rot90(ks[-1], -1))', 'cv2.morphologyEx(b, cv2.MORPH_HITMISS, np.ascontiguousarray(kk))', 'np.bitwise_or(out,', '_, b = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)']) assert.ok(hm.includes(t), `hit-or-miss python: ${t}`);
  assert.ok(!k('morphology', { op: 'hit_or_miss', hmRotate: false }).includes('np.rot90'));
  const rg = k('morphology', { op: 'h_maxima', hHeight: 25 });
  for (const t of ['nxt = np.minimum(cv2.dilate(m, k3), mask)', 'np.array_equal(nxt, m)', 'a - recon(a - 1, a) >= 1', 'hm = recon(np.maximum(i16 - 25, 0), i16)', 'cv2.connectedComponents(regional_max(hm), connectivity=8)']) assert.ok(rg.includes(t), `regional python: ${t}`);
  assert.ok(k('morphology', { op: 'regional_min' }).includes('out = regional_max(255 - i16)') && k('morphology', { op: 'extended_max', hHeight: 3 }).includes('out = regional_max(hm)'));
  // JS-side numbers the Python must reproduce: the GLCM props of the MATLAB example with the Python formulas re-typed in JS (sums in another order)
  const q = Uint8Array.from(GL_IMG.flat(), (v) => v - 1), C = glcmCounts(q, 5, 4, 8, 1, 0), N = 16, P = Array.from(C, (x) => x / N);
  let contrast = 0, energy = 0, homog = 0; for (let i = 7; i >= 0; i--) for (let j = 7; j >= 0; j--) { const p = P[i * 8 + j]; contrast += (i - j) ** 2 * p; energy += p * p; homog += p / (1 + Math.abs(i - j)); }
  const pj = glcmProps(C, 8); near(pj.contrast, contrast, 1e-12, 'contrast (reversed sum order)'); near(pj.energy, energy, 1e-12, 'energy'); near(pj.homogeneity, homog, 1e-12, 'homogeneity');
});

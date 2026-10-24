// IP batch 2 ชิ้น E (2026-10-04) — slic (superpixels) · bandpass_notch (band-pass / band-reject / notch)
// สเปก: Vision_boi/specs/ip_batch2_E_2026-10-03.md (§7 edge cases · §8 เกณฑ์ตัวเลข) · WORKLOG: WORKLOG-2026-10-04-ip-batch2-E.md
// โครงเดียวกับชิ้น C/D: ipE-0 (fingerprint, เทสต์แรกของไฟล์) + ipE-0w (wrapper balance เส้นทางชุดเดียวกัน) → ipE-L1..L4 → ตัวเลข → static →
// parity กับ Python จริง (cv2 5.0 / numpy 2.x)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { OPS, FILTERS, defaultParams, needsOf, slicLabels, signedText } from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS, OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { GROUP_ICON, OP_PRIMARY } from '../src/renderer/filters/ui-meta.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import {
  assertLeakFree, assertWrapperBalance, fingerprint, makeNativeLeakControl, makeThrowingCv, makeWrapperTracker,
} from './_leak-helpers.mjs';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.dft === 'function' && typeof c.getOptimalDFTSize === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

const NEW_OPS = ['slic', 'bandpass_notch'];
const HOST = { slic: 'colour', bandpass_notch: 'frequency' };
const SLIC_KEYS = ['sp_count', 'sp_requested', 'sp_grid', 'sp_step_px', 'sp_mean_area_px'];
const YELLOW = [255, 255, 0];

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
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d.set(fn(x, y), (y * w + x) * 3);
  return cv.matFromArray(h, w, cv.CV_8UC3, d);
}
function toRGBA(m) { const out = new cv.Mat(); cv.cvtColor(m, out, m.channels() === 1 ? cv.COLOR_GRAY2RGBA : cv.COLOR_RGB2RGBA); return out; }
function toRGB(m) { const out = new cv.Mat(); cv.cvtColor(m, out, cv.COLOR_GRAY2RGB); return out; }
function toGray(m) { const out = new cv.Mat(); cv.cvtColor(m, out, m.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY); return out; }
function noiseGray(w, h, seed) { const r = lcg(seed); return gray(w, h, () => Math.floor(r() * 256)); }
function noiseRGB(w, h, seed) { const r = lcg(seed); return rgb(w, h, () => [Math.floor(r() * 256), Math.floor(r() * 256), Math.floor(r() * 256)]); }
// ภาพสี R ≠ B มีโครงสร้าง (ไล่สี + ก้อนสี + noise เล็กน้อย) — ให้ SLIC มีขอบสีให้เกาะ
function sceneRGB(w, h, seed = 5) {
  const r = lcg(seed);
  return rgb(w, h, (x, y) => {
    const blob = (x - 0.3 * w) ** 2 + (y - 0.4 * h) ** 2 < (0.2 * Math.min(w, h)) ** 2 ? 70 : 0;
    return [
      (40 + Math.floor((200 * x) / w) + blob + Math.floor(r() * 12)) & 255,
      (90 + Math.floor((120 * y) / h) + Math.floor(r() * 12)) & 255,
      (210 - Math.floor((100 * (x + y)) / (w + h)) - blob + Math.floor(r() * 8)) & 255,
    ];
  });
}
function sceneGray(w, h, seed = 3) {
  const r = lcg(seed);
  return gray(w, h, (x, y) => {
    const base = x < w / 2 ? 60 : 190;
    const box = x > w / 4 && x < (3 * w) / 4 && y > h / 3 && y < (2 * h) / 3 ? 40 : 0;
    return Math.max(0, Math.min(255, base + box + Math.floor((r() - 0.5) * 24)));
  });
}
function bytes(m) { return Array.from(m.data); }
function run(opId, src, params = {}, cvUse = cv) {
  const ctx = {};
  const out = OPS[opId].apply(cvUse, src, params, ctx);
  return { out, ctx };
}
function runInfo(opId, src, params = {}) {
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
function scalarMap(ctx) { return Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value])); }
function fieldKeys(ctx) { return ctx.values.fields.map((f) => f.key); }
function infoGrid(info) { const m = info.match(/ตารางเริ่ม (\d+)×(\d+)/); return [Number(m[1]), Number(m[2])]; }
function infoCount(info) { return Number(info.match(/แบ่งภาพเป็น (\d+) superpixel/)[1]); }
function infoMerged(info) { return Number(info.match(/รวมเศษชิ้นเล็ก (\d+) ชิ้น/)[1]); }
const NOT_FOUND = Symbol('x');
// cvtColor RGB2Lab ของ opencv.js (เหมือนที่ op ใช้) → Uint8Array interleave 3 ช่อง สำหรับเรียก slicLabels ตรง ๆ
function labOf(m) {
  const lab = new cv.Mat();
  try { cv.cvtColor(m, lab, cv.COLOR_RGB2Lab); return Uint8Array.from(lab.data); } finally { lab.delete(); }
}
function featOf(m) { return m.channels() === 1 ? { F: Uint8Array.from(m.data), nch: 1 } : { F: labOf(m), nch: 3 }; }

// ===================================================================
// ipE-0 / ipE-0w — ด่าน leak หลัก (P-14 ข้อ 1 + คู่บังคับข้อ 2) — ทุกแถวของสเปก §7, ทุก view/kind/shape, ทั้งเส้นทางย่อและไม่ย่อ
// ===================================================================
// only = 'slic' | 'bandpass_notch' | undefined (both): the fingerprint probe is sensitive to which Mats are alive in the heap
// (a pair of 64x48 Mats alive during the slic probe moved the allocator state; not a leak — census + wrapper balance stay exact) so
// each op's probe builds only the images that op's paths use
function batchELeakPaths(only) {
  const mats = [];
  const keep = (m) => { mats.push(m); return m; };
  const wantS = !only || only === 'slic', wantB = !only || only === 'bandpass_notch';
  const empty = keep(new cv.Mat());
  const one = keep(gray(1, 1, () => 77)), two = keep(gray(2, 1, () => 5)), line = keep(gray(1, 50, (x, y) => y * 4)), tiny = keep(gray(3, 5, (x, y) => x * 40 + y * 20));
  const flatG = keep(gray(40, 7, () => 128));
  const paths = [];
  const add = (id, params, src, tag) => paths.push({ label: `${id} ${tag}`, run: (cvUse) => { const o = OPS[id].apply(cvUse, src, params, {}); o.delete(); } });
  if (wantS) {
    const g = keep(sceneGray(60, 40)), c = keep(sceneRGB(60, 40)), a = keep(toRGBA(c)), wideC = keep(sceneRGB(340, 20)), wideG = keep(sceneGray(340, 20));
    const oneC = keep(rgb(1, 1, () => [10, 200, 30])), flatC = keep(rgb(40, 7, () => [9, 99, 199]));
    for (const view of ['boundary', 'mean']) {
      add('slic', { view }, g, `${view} gray`);
      add('slic', { view }, c, `${view} colour`);
      add('slic', { view }, a, `${view} rgba`);
      add('slic', { view }, wideC, `${view} colour shrink path`);
      add('slic', { view }, wideG, `${view} gray shrink path`);
      add('slic', { view, count: 1000 }, tiny, `${view} 3x5`);
      add('slic', { view }, one, `${view} 1x1 gray`);
      add('slic', { view }, oneC, `${view} 1x1 colour`);
      add('slic', { view }, two, `${view} 2x1`);
      add('slic', { view }, line, `${view} 1x50`);
      add('slic', { view, count: 1000 }, flatG, `${view} 40x7 K 1000`);
      add('slic', { view, count: 12 }, flatC, `${view} constant`);
    }
    add('slic', {}, empty, '0x0');
  }
  if (wantB) {
    const wideBp = keep(gray(2000, 100, (x, y) => 128 + ((x >> 3) & 1) * 40));
    const bpG = keep(sceneGray(64, 48)), bpC = keep(sceneRGB(64, 48)), bpA = keep(toRGBA(bpC));
    for (const kind of ['bandpass', 'bandreject', 'notch']) {
      for (const shape of ['gaussian', 'ideal']) {
        for (const view of ['result', 'spectrum']) add('bandpass_notch', { kind, shape, view }, bpG, `${kind} ${shape} ${view}`);
      }
    }
    add('bandpass_notch', {}, bpC, 'colour (direct call)');
    add('bandpass_notch', { kind: 'notch' }, bpA, 'rgba (direct call)');
    add('bandpass_notch', { rIn: 50, rOut: 10 }, bpG, 'swapped');
    add('bandpass_notch', { kind: 'bandreject', rIn: 0 }, flatG, 'constant + DC warning');
    add('bandpass_notch', {}, flatG, 'constant bandpass flat');
    add('bandpass_notch', {}, one, '1x1');
    add('bandpass_notch', { view: 'spectrum' }, one, '1x1 spectrum');
    add('bandpass_notch', {}, two, '2x1');
    add('bandpass_notch', { kind: 'notch' }, line, '1x50');
    add('bandpass_notch', { kind: 'notch', view: 'spectrum' }, tiny, '3x5 spectrum');
    add('bandpass_notch', {}, wideBp, '2000x100 shrink path');
    add('bandpass_notch', { kind: 'notch', view: 'spectrum' }, wideBp, '2000x100 shrink path spectrum');
    add('bandpass_notch', {}, empty, '0x0');
  }
  return { paths, cleanup: () => { for (const m of mats) m.delete(); } };
}

// The two-point fingerprint of fingerprintLeakProbe (A before / B after N calls) turned out to flip with unrelated things in this file
// (the file's name, the tests defined after it) although the census and the wrapper balance stay exact: the allocator's free lists of a
// leak-free op cycle through 1-3 states (measured sequences 0000000000 / 0101010101 / 012012012012), never settling on A after exactly
// N calls. A leak never revisits a state: every window moves the next-malloc addresses (measured: 16 bytes per call -> 0123456789).
// So the probe samples the fingerprint after 9 windows of `iters` calls and requires that states repeat (<= 4 distinct of 9)
function fingerprintCycleProbe(label, run, { iters = 12, warmup = 10, windows = 9, maxDistinct = 4 } = {}) {
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

for (const [suffix, opId, what] of [
  ['', 'slic', 'slic boundary / mean on gray / colour / RGBA, the shrink path, 1x1 / 2x1 / 1x50 / 3x5 / K 1000, constant, 0x0'],
  ['b', 'bandpass_notch', 'bandpass_notch 3 kinds x 2 shapes x 2 views, colour / RGBA direct calls, swapped rings, constants, 1x1 / 2x1 / 1x50 / 3x5, the 2000x100 shrink path, 0x0'],
]) {
  test(`ipE-0${suffix}. leak (address fingerprint cycle, per op, ${suffix ? 'after ipE-0' : 'first test of a clean process'}): every branch of ${what} keeps revisiting the same few malloc states (a leak never does)`, () => {
    const { paths, cleanup } = batchELeakPaths(opId);
    try {
      const mine = paths.filter((q) => q.label.startsWith(`${opId} `));
      assert.ok(mine.length >= 20, `${opId}: paths ${mine.length}`);
      fingerprintCycleProbe(`IP batch 2 chunk E ${opId} (all paths)`, () => { for (const q of mine) q.run(cv); });
    } finally { cleanup(); }
  });
}

test('ipE-0c. (control of the fingerprint gate) the same probe flags a native leak of 16 bytes per call on top of the slic paths (9 distinct states in 9 windows) -- so ipE-0 / ipE-0b are able to fail', () => {
  const { paths, cleanup } = batchELeakPaths('slic');
  const ctl = makeNativeLeakControl(cv, 16);
  try {
    const mine = paths.filter((q) => q.label.startsWith('slic '));
    assert.throws(() => fingerprintCycleProbe('control', () => { for (const q of mine) q.run(cv); ctl.run(); }), /distinct malloc fingerprints/);
  } finally { cleanup(); ctl.cleanup(); }
});

test('ipE-0d. (control of the fingerprint gate, bandpass_notch twin of ipE-0c) the same probe flags a native leak of 16 bytes per call on top of the bandpass_notch paths -- so ipE-0b is able to fail', () => {
  const { paths, cleanup } = batchELeakPaths('bandpass_notch');
  const ctl = makeNativeLeakControl(cv, 16);
  try {
    const mine = paths.filter((q) => q.label.startsWith('bandpass_notch '));
    assert.throws(() => fingerprintCycleProbe('control', () => { for (const q of mine) q.run(cv); ctl.run(); }), /distinct malloc fingerprints/);
  } finally { cleanup(); ctl.cleanup(); }
});

test('ipE-0w. leak (wrapper balance twin of ipE-0): the exact same path table creates and deletes every embind wrapper it makes (RGB copy of RGBA, shrunk copy, Lab Mat, output, gray temp, padded / float / spectrum / inverse Mats, 8-bit view, passthrough clones)', () => {
  const { paths, cleanup } = batchELeakPaths();
  try {
    assertWrapperBalance(cv, 'IP batch 2 chunk E ops (all paths)', (c) => {
      for (const p of paths) p.run(c);
    }, { minCreated: 150 });
  } finally { cleanup(); }
});

// ===================================================================
// ipE-L1…L4 — leak ต่อ op / ต่อเส้นทาง
// ===================================================================
function leakEach(label, cases, opts = {}) {
  for (const [tag, fn] of cases) {
    assertLeakFree(cv, `${label} ${tag}`, (c) => { const o = fn(c); o.delete(); }, { minCreated: 1, iters: 12, warmup: 4, ...opts });
  }
}

test('ipE-L1. leak: slic boundary / mean on gray / colour / RGBA, the shrink path, tiny images, constant and the 0x0 passthrough balances every wrapper and grows the heap by 0', () => {
  const g = sceneGray(60, 40), c = sceneRGB(60, 40), a = toRGBA(c), wide = sceneRGB(340, 30), one = gray(1, 1, () => 9), flat = rgb(30, 20, () => [4, 5, 6]), empty = new cv.Mat();
  try {
    leakEach('slic', [
      ['boundary gray', (cvU) => OPS.slic.apply(cvU, g, {}, {})],
      ['mean gray', (cvU) => OPS.slic.apply(cvU, g, { view: 'mean' }, {})],
      ['boundary colour', (cvU) => OPS.slic.apply(cvU, c, { count: 50 }, {})],
      ['mean colour', (cvU) => OPS.slic.apply(cvU, c, { view: 'mean', count: 50 }, {})],
      ['boundary rgba', (cvU) => OPS.slic.apply(cvU, a, { count: 50 }, {})],
      ['mean rgba', (cvU) => OPS.slic.apply(cvU, a, { view: 'mean', count: 50 }, {})],
      ['shrink boundary', (cvU) => OPS.slic.apply(cvU, wide, { count: 40, iterations: 2 }, {})],
      ['shrink mean', (cvU) => OPS.slic.apply(cvU, wide, { view: 'mean', count: 40, iterations: 2 }, {})],
      ['1x1', (cvU) => OPS.slic.apply(cvU, one, {}, {})],
      ['constant', (cvU) => OPS.slic.apply(cvU, flat, { count: 12 }, {})],
      ['0x0', (cvU) => OPS.slic.apply(cvU, empty, {}, {})],
    ]);
  } finally { for (const m of [g, c, a, wide, one, flat, empty]) m.delete(); }
});

test('ipE-L2. leak: bandpass_notch result / spectrum for every kind, colour / RGBA direct calls, the shrink path, 1x1 and the 0x0 passthrough balances every wrapper (gray temp, shrunk copy, padded, float, spectrum, inverse, 8-bit view) and grows the heap by 0', () => {
  const g = sceneGray(64, 48), c = sceneRGB(64, 48), a = toRGBA(c), one = gray(1, 1, () => 9), wide = gray(2000, 40, (x) => 100 + ((x >> 2) & 1) * 50), empty = new cv.Mat();
  try {
    const cases = [];
    for (const kind of ['bandpass', 'bandreject', 'notch']) for (const view of ['result', 'spectrum']) cases.push([`${kind} ${view}`, (cvU) => OPS.bandpass_notch.apply(cvU, g, { kind, view }, {})]);
    cases.push(['colour', (cvU) => OPS.bandpass_notch.apply(cvU, c, {}, {})]);
    cases.push(['rgba notch spectrum', (cvU) => OPS.bandpass_notch.apply(cvU, a, { kind: 'notch', view: 'spectrum' }, {})]);
    cases.push(['shrink result', (cvU) => OPS.bandpass_notch.apply(cvU, wide, {}, {})]);
    cases.push(['shrink notch', (cvU) => OPS.bandpass_notch.apply(cvU, wide, { kind: 'notch' }, {})]);
    cases.push(['1x1', (cvU) => OPS.bandpass_notch.apply(cvU, one, {}, {})]);
    cases.push(['0x0', (cvU) => OPS.bandpass_notch.apply(cvU, empty, {}, {})]);
    leakEach('bandpass_notch', cases);
  } finally { for (const m of [g, c, a, one, wide, empty]) m.delete(); }
});

function withThrowingMatMethod(name, fn) {
  const proto = cv.Mat.prototype;
  const orig = proto[name];
  proto[name] = function () { throw new Error(`injected test failure (Mat.${name})`); };
  try { return fn(); } finally { proto[name] = orig; }
}

test('ipE-L3. leak on the exception path: when a cv call inside each op throws (cvtColor 1st / 2nd, resize, copyMakeBorder, dft forward / inverse, getOptimalDFTSize, spectrum resize, Mat.copyTo, Mat.convertTo), every wrapper allocated so far is still released and the exception propagates', () => {
  const g = sceneGray(60, 40), c = sceneRGB(60, 40), a = toRGBA(c), wideC = sceneRGB(340, 20), wideG = sceneGray(340, 20);
  const bg = sceneGray(64, 48), bc = sceneRGB(64, 48), wideBp = gray(2000, 40, (x) => 100 + ((x >> 2) & 1) * 50);
  const throwing = [
    ['slic Lab cvtColor', 'cvtColor', (cvU) => OPS.slic.apply(cvU, c, {}, {})],
    ['slic RGBA->RGB cvtColor', 'cvtColor', (cvU) => OPS.slic.apply(cvU, a, {}, {})],
    ['slic Lab cvtColor after RGBA->RGB', ['cvtColor', 2], (cvU) => OPS.slic.apply(cvU, a, {}, {})],
    ['slic GRAY2RGB (boundary gray)', 'cvtColor', (cvU) => OPS.slic.apply(cvU, g, {}, {})],
    ['slic shrink resize colour', 'resize', (cvU) => OPS.slic.apply(cvU, wideC, {}, {})],
    ['slic shrink resize gray', 'resize', (cvU) => OPS.slic.apply(cvU, wideG, { view: 'mean' }, {})],
    ['bandpass colour cvtColor', 'cvtColor', (cvU) => OPS.bandpass_notch.apply(cvU, bc, {}, {})],
    ['bandpass shrink resize', 'resize', (cvU) => OPS.bandpass_notch.apply(cvU, wideBp, {}, {})],
    ['bandpass spectrum resize', 'resize', (cvU) => OPS.bandpass_notch.apply(cvU, bg, { view: 'spectrum' }, {})],
    ['bandpass shrink 2nd resize (result)', ['resize', 2], (cvU) => OPS.bandpass_notch.apply(cvU, wideBp, {}, {})],
    ['bandpass getOptimalDFTSize (colour: gray temp exists)', 'getOptimalDFTSize', (cvU) => OPS.bandpass_notch.apply(cvU, bc, {}, {})],
    ['bandpass 2nd getOptimalDFTSize (colour: gray temp exists)', ['getOptimalDFTSize', 2], (cvU) => OPS.bandpass_notch.apply(cvU, bc, {}, {})],
    ['bandpass copyMakeBorder', 'copyMakeBorder', (cvU) => OPS.bandpass_notch.apply(cvU, bg, {}, {})],
    ['bandpass forward dft', 'dft', (cvU) => OPS.bandpass_notch.apply(cvU, bg, {}, {})],
    ['bandpass inverse dft', ['dft', 2], (cvU) => OPS.bandpass_notch.apply(cvU, bg, {}, {})],
  ];
  const matThrowing = [
    ['slic boundary copyTo (colour)', 'copyTo', (cvU) => OPS.slic.apply(cvU, c, {}, {})],
    ['bandpass convertTo', 'convertTo', (cvU) => OPS.bandpass_notch.apply(cvU, bg, {}, {})],
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
  } finally { for (const m of [g, c, a, wideC, wideG, bg, bc, wideBp]) m.delete(); }
});

test('ipE-L4. leak on the exception path AFTER the output exists: a ctx whose "values" setter (the last statement of slic) or "info" setter throws makes each op delete its own output -- slic boundary / mean, bandpass_notch result (no shrink / shrink) / spectrum', () => {
  const g = sceneGray(60, 40), c = sceneRGB(60, 40), bg = sceneGray(64, 48), wideBp = gray(2000, 40, (x) => 100 + ((x >> 2) & 1) * 50);
  const throwingValues = () => ({ info: '', set values(_) { throw new Error('injected test failure (ctx.values)'); } });
  const throwingInfo = () => ({ set info(_) { throw new Error('injected test failure (ctx.info)'); } });
  const cases = [
    ['slic boundary colour', (cvU, ctx) => OPS.slic.apply(cvU, c, {}, ctx)],
    ['slic mean gray', (cvU, ctx) => OPS.slic.apply(cvU, g, { view: 'mean' }, ctx)],
  ];
  const infoOnly = [
    ['bandpass result', (cvU, ctx) => OPS.bandpass_notch.apply(cvU, bg, {}, ctx)],
    ['bandpass shrink result', (cvU, ctx) => OPS.bandpass_notch.apply(cvU, wideBp, {}, ctx)],
    ['bandpass spectrum', (cvU, ctx) => OPS.bandpass_notch.apply(cvU, bg, { view: 'spectrum' }, ctx)],
  ];
  try {
    for (const [label, fn] of cases) {
      assertWrapperBalance(cv, `ctx.values throws in ${label}`, (cvU) => { const o = fn(cvU, throwingValues()); o.delete(); }, { expectThrow: /injected test failure \(ctx\.values\)/, minCreated: 1 });
    }
    for (const [label, fn] of [...cases, ...infoOnly]) {
      assertWrapperBalance(cv, `ctx.info throws in ${label}`, (cvU) => { const o = fn(cvU, throwingInfo()); o.delete(); }, { expectThrow: /injected test failure \(ctx\.info\)/, minCreated: 1 });
    }
  } finally { for (const m of [g, c, bg, wideBp]) m.delete(); }
});

test('ipE-L5. peak memory of bandpass_notch (P-14 "delete intermediates early"): the padded Mat is gone before the forward DFT result is used, the float Mat before the inverse DFT, the spectrum Mat before the 8-bit view is expanded -- measured as the number of live wrappers at each cv call (no shrink: forward dft {f, spec} = 2, inverse dft {spec, inv} = 2; shrink path: the expanding resize sees {small, inv, u8, dst} = 4)', () => {
  const g = sceneGray(64, 48), wide = gray(2000, 40, (x) => 100 + ((x >> 2) & 1) * 50);
  const probe = (src) => {
    const t = makeWrapperTracker(cv);
    const seen = { dft: [], resize: [] };
    const hooked = new Proxy(t.cv, {
      get(target, prop) {
        if (prop !== 'dft' && prop !== 'resize') return target[prop];
        return (...args) => { seen[prop].push(t.live.size); return target[prop](...args); };
      },
    });
    try {
      const o = OPS.bandpass_notch.apply(hooked, src, {}, {});
      o.delete();
      t.assertBalanced('peak probe', { minCreated: 1 });
    } finally { t.dispose(); }
    return seen;
  };
  try {
    const a = probe(g);
    assert.deepEqual(a.dft, [2, 2], `live wrappers at the forward / inverse dft: ${a.dft}`);
    const b = probe(wide);
    assert.deepEqual(b.dft, [3, 3], `shrink path live at dft (small + the above): ${b.dft}`); // small stays alive until the end: f + spec + small, spec + inv + small
    assert.deepEqual(b.resize, [1, 4], `live wrappers at the shrink / expand resize: ${b.resize}`); // 1st: {small}; 2nd: {small, inv, u8, dst}
  } finally { g.delete(); wide.delete(); }
});

// ===================================================================
// slic — ตัวเลข
// ===================================================================
// ป้ายกำกับ → ตรวจสมบัติ: 0..count−1 เรียงตามพิกเซลแรกที่พบแบบ raster · ทุกป้ายเชื่อมกัน 4 ทิศ · ทุกก้อนใหญ่กว่า limit (ยกเว้นก้อนของพิกเซล 0)
function checkLabelProps(res, w, h, label) {
  const { labels, count, nx, ny } = res;
  const N = w * h;
  assert.equal(labels.length, N);
  let next = 0;
  const sizes = new Int32Array(count);
  for (let i = 0; i < N; i++) {
    const l = labels[i];
    assert.ok(l >= 0 && l < count, `${label}: label ${l} out of 0..${count - 1}`);
    if (l === next) next++;
    else assert.ok(l < next, `${label}: label ${l} first appears out of raster order at pixel ${i} (next new = ${next})`);
    sizes[l]++;
  }
  assert.equal(next, count, `${label}: every label 0..${count - 1} appears`);
  // 4-connected: flood fill each label from its first pixel
  const seen = new Uint8Array(N), reach = new Int32Array(count);
  for (let i0 = 0; i0 < N; i0++) {
    const l = labels[i0];
    if (seen[i0]) continue;
    assert.equal(reach[l], 0, `${label}: label ${l} is split into 2+ 4-connected pieces`);
    const stack = [i0]; seen[i0] = 1;
    while (stack.length) {
      const i = stack.pop(); reach[l]++;
      const x = i % w, y = (i - x) / w;
      for (const [ok, j] of [[x > 0, i - 1], [y > 0, i - w], [x < w - 1, i + 1], [y < h - 1, i + w]]) if (ok && !seen[j] && labels[j] === l) { seen[j] = 1; stack.push(j); }
    }
    assert.equal(reach[l], sizes[l], `${label}: label ${l} 4-connected size`);
  }
  const limit = Math.floor(N / (nx * ny)) >> 2;
  for (let l = 1; l < count; l++) assert.ok(sizes[l] > limit, `${label}: label ${l} has ${sizes[l]} <= limit ${limit} pixels (should have been merged)`);
}

test('ipE-1. slicLabels hand case: a 4x2 gray image [0,0,255,255] x2, K 2, m 1, 1 iteration -> labels [0,0,1,1,0,0,1,1], count 2, grid 2x1, S 2 (and the arguments are not modified)', () => {
  const F = Uint8Array.from([0, 0, 255, 255, 0, 0, 255, 255]);
  const before = Array.from(F);
  const r = slicLabels(F, 1, 4, 2, 2, 1, 1);
  assert.deepEqual(Array.from(r.labels), [0, 0, 1, 1, 0, 0, 1, 1]);
  assert.equal(r.count, 2); assert.equal(r.nx, 2); assert.equal(r.ny, 1); assert.equal(r.S, 2); assert.equal(r.merged, 0);
  assert.deepEqual(Array.from(F), before);
  // 3-channel call (Lab-like data) with a colour edge gives the same split
  const F3 = new Uint8Array(24);
  for (let i = 0; i < 8; i++) { F3[i * 3] = i % 4 < 2 ? 0 : 255; F3[i * 3 + 1] = 128; F3[i * 3 + 2] = 128; }
  assert.deepEqual(Array.from(slicLabels(F3, 3, 4, 2, 2, 1, 1).labels), [0, 0, 1, 1, 0, 0, 1, 1]);
});

test('ipE-2. slicLabels invariants on 7 images (gray / Lab, noise / scene / ramp, several K and m): labels are exactly 0..count-1 in raster order of first appearance, every label is 4-connected, every segment has more than limit = floor(N / (nx*ny)) >> 2 pixels except the one holding pixel 0, count <= nx*ny, grid = round(w/S) x round(h/S)', () => {
  const cases = [
    [noiseGray(61, 47, 1), 100, 10], [sceneGray(64, 64), 30, 20], [sceneRGB(80, 60), 200, 10], [sceneRGB(97, 61, 7), 1000, 5], [noiseRGB(40, 30, 4), 20, 1],
    [gray(50, 40, (x, y) => (x * 5 + y * 3) & 255), 60, 40], [sceneRGB(33, 90, 2), 80, 10],
  ];
  try {
    for (const [m, K, comp] of cases) {
      const { F, nch } = featOf(m);
      const r = slicLabels(F, nch, m.cols, m.rows, K, comp, 10);
      const label = `${m.cols}x${m.rows} ch${nch} K${K} m${comp}`;
      checkLabelProps(r, m.cols, m.rows, label);
      assert.ok(r.count >= 1 && r.count <= r.nx * r.ny, `${label}: count ${r.count} vs grid ${r.nx * r.ny}`);
      const S = Math.sqrt((m.cols * m.rows) / K);
      assert.equal(r.nx, Math.max(1, Math.min(m.cols, Math.floor(m.cols / S + 0.5))), `${label}: nx`);
      assert.equal(r.ny, Math.max(1, Math.min(m.rows, Math.floor(m.rows / S + 0.5))), `${label}: ny`);
      assert.equal(r.S, S);
    }
  } finally { for (const [m] of cases) m.delete(); }
});

test('ipE-3. slic on left red / right blue 320x240 (defaults): the mean view reproduces the input byte for byte and the count is exactly the grid (192 = 16 x 12, nothing merged)', () => {
  const m = rgb(320, 240, (x) => (x < 160 ? [220, 20, 20] : [20, 20, 220]));
  try {
    const { out, ctx } = run('slic', m, { view: 'mean' });
    try { sameBytes(out, m, 'mean of two flat colours'); } finally { out.delete(); }
    assert.equal(scalarMap(ctx).sp_count, 192);
    assert.equal(scalarMap(ctx).sp_grid, 192);
    assert.equal(infoMerged(ctx.info), 0);
    assert.deepEqual(infoGrid(ctx.info), [16, 12]);
  } finally { m.delete(); }
});

test('ipE-4. slic on a constant 128 image with K 12 (80x60): count 12 (a 4 x 3 grid), the mean view is all 128 (gray and colour), the boundary view has yellow lines only between the grid cells', () => {
  const g = gray(80, 60, () => 128), c = rgb(80, 60, () => [128, 128, 128]);
  try {
    for (const m of [g, c]) {
      const { out, ctx } = run('slic', m, { count: 12, view: 'mean' });
      try {
        assert.equal(scalarMap(ctx).sp_count, 12);
        assert.deepEqual(infoGrid(ctx.info), [4, 3]);
        assert.ok(out.data.every((v) => v === 128), 'mean all 128');
        assert.equal(out.channels(), m.channels());
      } finally { out.delete(); }
    }
    const { out } = run('slic', g, { count: 12 });
    try {
      assert.equal(out.channels(), 3);
      let yellow = 0;
      for (let i = 0; i < 80 * 60; i++) if (out.data[i * 3] === 255 && out.data[i * 3 + 1] === 255 && out.data[i * 3 + 2] === 0) yellow++;
      assert.ok(yellow > 0 && yellow < 80 * 60 / 4, `yellow pixels ${yellow}`);
    } finally { out.delete(); }
  } finally { g.delete(); c.delete(); }
});

test('ipE-5. slic compactness: the number of boundary pixels falls as compactness rises (m = 1 > m = 10 > m = 40) on a textured scene -- low m follows colour edges (ragged, long boundaries), high m gives regular cells', () => {
  const m = sceneRGB(320, 240, 11);
  const boundary = (p) => {
    const { out } = run('slic', m, { count: 200, ...p });
    try {
      let n = 0;
      for (let i = 0; i < 320 * 240; i++) if (out.data[i * 3] === 255 && out.data[i * 3 + 1] === 255 && out.data[i * 3 + 2] === 0) n++;
      return n;
    } finally { out.delete(); }
  };
  try {
    const a = boundary({ compactness: 1 }), b = boundary({ compactness: 10 }), c = boundary({ compactness: 40 });
    assert.ok(a > b && b > c, `boundary pixels m1 ${a} m10 ${b} m40 ${c}`);
  } finally { m.delete(); }
});

test('ipE-6. slic is deterministic (two runs give identical bytes and values) and never modifies its input (gray / colour / RGBA / shrink path, both views)', () => {
  const imgs = [sceneGray(70, 50), sceneRGB(70, 50), toRGBA(sceneRGB(70, 50)), sceneRGB(400, 300)];
  try {
    for (const m of imgs) for (const view of ['boundary', 'mean']) {
      const before = bytes(m);
      const a = run('slic', m, { view, count: 60, iterations: 4 }), b = run('slic', m, { view, count: 60, iterations: 4 });
      try {
        sameBytes(a.out, b.out, `${m.cols}x${m.rows} ch${m.channels()} ${view}`);
        assert.deepEqual(a.ctx.values, b.ctx.values);
        assert.equal(a.ctx.info, b.ctx.info);
        assert.deepEqual(bytes(m), before, 'input untouched');
      } finally { a.out.delete(); b.out.delete(); }
    }
  } finally { for (const m of imgs) m.delete(); }
});

test('ipE-7. slic mean view = an independent recompute (labels from slicLabels on cv.cvtColor(RGB2Lab) of the input, per-label integer sums over the image, floor(sum / count + 0.5)) at scale 1 -- gray, colour, RGBA (RGB part), several K; the boundary view marks exactly the pixels whose right / down neighbour has another label (identity full-res map at scale 1)', () => {
  const imgs = [sceneGray(90, 60), sceneRGB(90, 60, 3), toRGBA(sceneRGB(64, 48, 9))];
  try {
    for (const m of imgs) for (const K of [30, 150]) {
      const rgbPart = m.channels() === 4 ? (() => { const t = new cv.Mat(); cv.cvtColor(m, t, cv.COLOR_RGBA2RGB); return t; })() : m;
      try {
        const nch = rgbPart.channels();
        const { F } = featOf(rgbPart);
        const r = slicLabels(F, nch, m.cols, m.rows, K, 10, 10);
        const sums = new Float64Array(r.count * nch), cnt = new Float64Array(r.count);
        for (let i = 0; i < m.cols * m.rows; i++) { cnt[r.labels[i]]++; for (let c = 0; c < nch; c++) sums[r.labels[i] * nch + c] += rgbPart.data[i * nch + c]; }
        const mean = run('slic', m, { count: K, view: 'mean' }), bound = run('slic', m, { count: K, view: 'boundary' });
        try {
          assert.equal(mean.ctx.values.fields[0].value, r.count);
          for (let i = 0; i < m.cols * m.rows; i++) for (let c = 0; c < nch; c++) {
            const want = Math.floor(sums[r.labels[i] * nch + c] / cnt[r.labels[i]] + 0.5);
            if (mean.out.data[i * nch + c] !== want) assert.fail(`mean pixel ${i} ch ${c}: ${mean.out.data[i * nch + c]} vs ${want}`);
          }
          const W = m.cols, H = m.rows;
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x;
            const edge = (x + 1 < W && r.labels[i + 1] !== r.labels[i]) || (y + 1 < H && r.labels[i + W] !== r.labels[i]);
            const o = i * 3, got = bound.out.data[o] === 255 && bound.out.data[o + 1] === 255 && bound.out.data[o + 2] === 0;
            if (edge) assert.ok(got, `boundary pixel (${x},${y}) not yellow`);
            else for (let c = 0; c < 3; c++) if (bound.out.data[o + c] !== (nch === 1 ? rgbPart.data[i] : rgbPart.data[i * 3 + c])) assert.fail(`non-boundary pixel (${x},${y}) changed`);
          }
        } finally { mean.out.delete(); bound.out.delete(); }
      } finally { if (rgbPart !== m) rgbPart.delete(); }
    }
  } finally { for (const m of imgs) m.delete(); }
});

test('ipE-8. slic shrink path (> 320 px): analysis size = round(side * 320 / max) via INTER_AREA, the label map is expanded with floor(((2x + 1) * aw) / (2W)) (centred) -- at 700x500 (analysis 320x229, a ratio that is NOT an integer, where centred and left-aligned maps differ) the boundary pixels are exactly the label changes of the 320x229 area-shrunk copy sampled with that formula; the info says the analysis size; sp_step_px = S / scale', () => {
  const m = sceneRGB(700, 500, 21);
  const small = new cv.Mat();
  try {
    cv.resize(m, small, new cv.Size(320, 229), 0, 0, cv.INTER_AREA);
    const r = slicLabels(labOf(small), 3, 320, 229, 200, 10, 10);
    const { out, ctx } = run('slic', m, { view: 'boundary' });
    try {
      assert.match(ctx.info, /คำนวณบนภาพย่อ 320×229 แล้วขยายผลกลับ/);
      const W = 700, H = 500;
      const L = (x, y) => r.labels[Math.floor(((2 * y + 1) * 229) / (2 * H)) * 320 + Math.floor(((2 * x + 1) * 320) / (2 * W))];
      let bad = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const edge = (x + 1 < W && L(x + 1, y) !== L(x, y)) || (y + 1 < H && L(x, y + 1) !== L(x, y));
        const o = (y * W + x) * 3, got = out.data[o] === 255 && out.data[o + 1] === 255 && out.data[o + 2] === 0;
        if (edge !== got) bad++;
      }
      assert.equal(bad, 0, 'boundary pixels = label change of the centred map');
      assert.equal(scalarMap(ctx).sp_count, r.count);
      assert.equal(scalarMap(ctx).sp_step_px, r.S / (320 / 700));
      assert.equal(scalarMap(ctx).sp_mean_area_px, (700 * 500) / r.count);
    } finally { out.delete(); }
    // not shrunk when the long side is exactly 320 (and no note), shrunk from 321
    const m320 = sceneRGB(320, 100), m321 = sceneRGB(321, 100);
    try {
      const e = run('slic', m320, {});
      try { assert.ok(!e.ctx.info.includes('คำนวณบนภาพย่อ')); } finally { e.out.delete(); }
      const f = run('slic', m321, {});
      try { assert.match(f.ctx.info, /ภาพย่อ 320×100 /); } finally { f.out.delete(); }
    } finally { m320.delete(); m321.delete(); }
  } finally { m.delete(); small.delete(); }
});

test('ipE-9. slic values contract: always the same 5 keys in the same order (sp_count, sp_requested, sp_grid, sp_step_px, sp_mean_area_px) on every path (gray / colour / RGBA, shrink, tiny, constant, 0x0 -- where only sp_requested = K is set); sp_count / sp_grid / sp_requested equal the numbers in the info; a fresh object every call', () => {
  const empty = new cv.Mat();
  const cases = [[sceneGray(60, 40), {}], [sceneRGB(60, 40), { count: 50 }], [toRGBA(sceneRGB(60, 40)), { view: 'mean' }], [sceneRGB(500, 100), { count: 30 }], [gray(1, 1, () => 7), {}],
    [gray(40, 7, () => 3), { count: 1000 }], [rgb(30, 20, () => [1, 2, 3]), { count: 12 }], [gray(3, 40, (x, y) => y), { count: 1000 }]];
  try {
    const seen = new Set();
    for (const [m, p] of cases) {
      const ctx = runInfo('slic', m, p);
      assert.equal(ctx.values.kind, 'scalars');
      assert.deepEqual(fieldKeys(ctx), SLIC_KEYS);
      assert.deepEqual(ctx.values.fields.map((f) => f.label), ['จำนวน superpixel ที่ได้', 'จำนวนที่ตั้งไว้', 'จุดเริ่มบนตาราง', 'ระยะห่างจุดเริ่ม (px ของภาพจริง)', 'พื้นที่เฉลี่ยต่อก้อน (px)']);
      const v = scalarMap(ctx), [nx, ny] = infoGrid(ctx.info);
      assert.equal(v.sp_count, infoCount(ctx.info));
      assert.equal(v.sp_grid, nx * ny);
      assert.equal(v.sp_requested, Math.round(p.count ?? 200));
      assert.equal(v.sp_mean_area_px, (m.cols * m.rows) / v.sp_count);
      assert.ok(!seen.has(ctx.values)); seen.add(ctx.values);
    }
    const e = run('slic', empty, { count: 77 });
    try {
      assert.equal(e.ctx.info, 'ภาพว่าง');
      assert.deepEqual(fieldKeys(e.ctx), SLIC_KEYS);
      assert.deepEqual(scalarMap(e.ctx), { sp_count: null, sp_requested: 77, sp_grid: null, sp_step_px: null, sp_mean_area_px: null });
      assert.equal(e.out.rows, 0); assert.equal(e.out.cols, 0);
    } finally { e.out.delete(); }
    assert.equal(scalarMap(runInfo('slic', empty, { count: 'x' })).sp_requested, 200);
  } finally { empty.delete(); for (const [m] of cases) m.delete(); }
});

test('ipE-10. slic edge cases (spec 7): 1x1 -> count 1, no boundary, mean = input; 2x1 / 1x50 / 3x5 / 3x40 / 40x7 never throw and keep size; 40x7 with K 1000 -> 280 superpixels (every pixel its own); RGBA -> 3 channels; gray boundary -> 3 channels, gray mean -> 1; a constant image gives count = grid and a byte-identical mean', () => {
  const one = gray(1, 1, () => 77), oneC = rgb(1, 1, () => [10, 200, 30]);
  try {
    for (const m of [one, oneC]) {
      const b = run('slic', m, {}), mn = run('slic', m, { view: 'mean' });
      try {
        assert.equal(scalarMap(b.ctx).sp_count, 1);
        assert.deepEqual(bytes(mn.out), bytes(m));
        const exp = m.channels() === 1 ? [77, 77, 77] : [10, 200, 30];
        assert.deepEqual(bytes(b.out), exp, 'no boundary on 1x1');
      } finally { b.out.delete(); mn.out.delete(); }
    }
  } finally { one.delete(); oneC.delete(); }
  for (const [w, h, K] of [[2, 1, 200], [1, 50, 200], [3, 5, 1000], [3, 40, 200], [40, 7, 200], [40, 7, 1000]]) {
    const m = sceneRGB(w, h, 2);
    try {
      for (const view of ['boundary', 'mean']) {
        const { out, ctx } = run('slic', m, { count: K, view });
        try {
          assert.equal(out.cols, w); assert.equal(out.rows, h);
          if (w === 40 && h === 7 && K === 1000) { assert.equal(scalarMap(ctx).sp_count, 280); assert.equal(scalarMap(ctx).sp_grid, 280); }
          if (view === 'mean' && w === 40 && h === 7 && K === 1000) assert.deepEqual(bytes(out), bytes(m), 'every pixel its own superpixel');
        } finally { out.delete(); }
      }
    } finally { m.delete(); }
  }
  const a = toRGBA(sceneRGB(30, 20)), g = sceneGray(30, 20);
  const flat = rgb(60, 40, () => [30, 60, 90]);
  try {
    const ra = run('slic', a, {}), rm = run('slic', a, { view: 'mean' }), gb = run('slic', g, {}), gm = run('slic', g, { view: 'mean' });
    try { assert.deepEqual([ra.out.channels(), rm.out.channels(), gb.out.channels(), gm.out.channels()], [3, 3, 3, 1]); } finally { for (const r of [ra, rm, gb, gm]) r.out.delete(); }
    const f = run('slic', flat, { view: 'mean', count: 24 });
    try {
      sameBytes(f.out, flat, 'constant mean');
      const [nx, ny] = infoGrid(f.ctx.info);
      assert.equal(scalarMap(f.ctx).sp_count, nx * ny);
    } finally { f.out.delete(); }
  } finally { a.delete(); g.delete(); flat.delete(); }
});

test('ipE-11. slic params: garbage / out of range / "constructor" -> defaults or clamps (count 5 -> 10, 5000 -> 1000, "x" -> 200; compactness 0 -> 1, 99 -> 40; iterations 0 -> 1, 99 -> 20, "7.4" -> 7; view "constructor" / "toString" -> boundary); the input Mat is never changed', () => {
  const m = sceneRGB(60, 40);
  const before = bytes(m);
  const same = (p, q, label) => {
    const a = run('slic', m, p), b = run('slic', m, q);
    try { sameBytes(a.out, b.out, label); assert.equal(a.ctx.info, b.ctx.info, label); } finally { a.out.delete(); b.out.delete(); }
  };
  try {
    same({ count: 5 }, { count: 10 }, 'count 5 -> 10');
    same({ count: 5000 }, { count: 1000 }, 'count 5000 -> 1000');
    same({ count: 'x' }, {}, 'count garbage');
    same({ count: '60.4' }, { count: 60 }, 'count string');
    same({ compactness: 0 }, { compactness: 1 }, 'm 0 -> 1');
    same({ compactness: 99 }, { compactness: 40 }, 'm 99 -> 40');
    same({ compactness: NaN }, {}, 'm NaN');
    same({ iterations: 0 }, { iterations: 1 }, 'it 0 -> 1');
    same({ iterations: 99 }, { iterations: 20 }, 'it 99 -> 20');
    same({ iterations: '7.4' }, { iterations: 7 }, 'it 7.4 -> 7');
    for (const bad of ['constructor', 'toString', '__proto__', 'hasOwnProperty', 7, null]) same({ view: bad }, { view: 'boundary' }, `view ${String(bad)}`);
    assert.deepEqual(bytes(m), before);
    assert.equal(runInfo('slic', m, { count: 'x' }).values.fields[1].value, 200);
  } finally { m.delete(); }
});

// ===================================================================
// bandpass_notch — ตัวเลข
// ===================================================================
const sinusoid256 = () => gray(256, 256, (x) => Math.round(128 + 100 * Math.sin((2 * Math.PI * x) / 16)));
function corr(a, b) {
  let sa = 0, sb = 0; const n = a.length;
  for (let i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
  const ma = sa / n, mb = sb / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return sab / Math.sqrt(saa * sbb);
}

test('ipE-12. bandpass_notch on 256x256 128 + 100 sin(2 pi x / 16) (12.5% of Nyquist): ideal band-pass 10-15 keeps the sinusoid (correlation > 0.999), ideal band-pass 20-40 -> an all black image with the "ภาพเรียบ" note, ideal band-reject 10-15 -> max |out - 128| <= 1 (the wave is gone, the mean stays)', () => {
  const m = sinusoid256();
  try {
    const a = run('bandpass_notch', m, { kind: 'bandpass', shape: 'ideal', rIn: 10, rOut: 15 });
    try {
      assert.ok(corr(Array.from(a.out.data), Array.from(m.data)) > 0.999, 'correlation');
      assert.ok(!a.ctx.info.includes('ไม่มีรายละเอียด'));
      assert.match(a.ctx.info, /Band-pass \(Ideal\) เก็บวงแหวนความถี่ 10–15% ของ Nyquist · สเปกตรัม 256×256 · ยืดค่าผลลัพธ์ให้เต็มช่วง 0–255 เพื่อให้มองเห็น/);
      assert.equal(a.out.rows, 256); assert.equal(a.out.cols, 256); assert.equal(a.out.channels(), 1);
    } finally { a.out.delete(); }
    const b = run('bandpass_notch', m, { kind: 'bandpass', shape: 'ideal', rIn: 20, rOut: 40 });
    try {
      assert.ok(b.out.data.every((v) => v === 0), 'black');
      assert.match(b.ctx.info, /· ไม่มีรายละเอียดในช่วงความถี่นี้ \(ภาพเรียบในช่วงนี้\) จึงได้ภาพดำ/);
    } finally { b.out.delete(); }
    const c = run('bandpass_notch', m, { kind: 'bandreject', shape: 'ideal', rIn: 10, rOut: 15 });
    try {
      let worst = 0;
      for (const v of c.out.data) worst = Math.max(worst, Math.abs(v - 128));
      assert.ok(worst <= 1, `max |out - 128| = ${worst}`);
      assert.ok(!c.ctx.info.includes('⚠️'), 'DC kept: no darkening warning');
    } finally { c.out.delete(); }
  } finally { m.delete(); }
});

test('ipE-13. bandpass_notch spectrum view of the sinusoid image: the brightest pixels are the centre (DC, 128 128) and (128 +- 16, 128); an ideal notch (12.5, 0, r 2) darkens exactly those two spots to 0 and leaves DC bright; the view is 1 channel at the source size', () => {
  const m = sinusoid256();
  try {
    const s = run('bandpass_notch', m, { view: 'spectrum', kind: 'bandreject', shape: 'ideal', rIn: 0, rOut: 1 });
    try {
      // an ideal band-reject of 0-1% only removes the centre bin — the rest of the spectrum is untouched
      assert.equal(s.out.channels(), 1); assert.equal(s.out.cols, 256); assert.equal(s.out.rows, 256);
      assert.match(s.ctx.info, /· มุมมองสเปกตรัม: จุดกลาง = ความถี่ต่ำ ส่วนที่มืดคือส่วนที่ถูกตัด$/);
      assert.ok(!s.ctx.info.includes('⚠️'), 'no darkening warning in the spectrum view');
    } finally { s.out.delete(); }
    const plain = run('bandpass_notch', m, { view: 'spectrum', kind: 'notch', shape: 'ideal', nx: 100, ny: 100, nr: 1 }); // notch far away: spectrum untouched near the wave
    try {
      const d = plain.out.data, W = 256;
      const top = Array.from({ length: d.length }, (_, i) => i).sort((p, q) => d[q] - d[p] || p - q).slice(0, 3).map((i) => [i % W, Math.floor(i / W)]);
      const key = (p) => p.join(',');
      assert.deepEqual(new Set(top.map(key)), new Set(['128,128', '112,128', '144,128']), 'brightest 3 pixels');
    } finally { plain.out.delete(); }
    const n = run('bandpass_notch', m, { view: 'spectrum', kind: 'notch', shape: 'ideal', nx: 12.5, ny: 0, nr: 2 });
    try {
      assert.equal(n.out.data[128 * 256 + 112], 0); assert.equal(n.out.data[128 * 256 + 144], 0);
      assert.ok(n.out.data[128 * 256 + 128] > 200, 'DC still bright');
      assert.match(n.ctx.info, /Notch \(Ideal\) ตัดจุดคู่ที่ \(\+12\.5%, 0%\) และ \(−12\.5%, 0%\) รัศมี 2%/);
    } finally { n.out.delete(); }
  } finally { m.delete(); }
});

// ภาพสะอาดเรียบ + ลายซ้ำ sin(2 pi (0.15 x + 0.05 y)) แอมพลิจูด 30 — 0.15 รอบ/px = 30% ของ Nyquist, 0.05 = 10% (ตรง bin พอดีที่ 640x480 และ 400x400)
function periodicPair(w, h) {
  const clean = gray(w, h, (x, y) => Math.round(110 + 40 * Math.exp(-(((x - w / 2) / (w / 3)) ** 2 + ((y - h / 2) / (h / 3)) ** 2)) + (30 * x) / w));
  const noisy = gray(w, h, (x, y) => Math.max(0, Math.min(255, Math.round(clean.data[y * w + x] + 30 * Math.sin(2 * Math.PI * (0.15 * x + 0.05 * y))))));
  return { clean, noisy };
}
function interiorMae(a, b, w, h) {
  const x0 = Math.floor(0.1 * w), x1 = Math.ceil(0.9 * w), y0 = Math.floor(0.1 * h), y1 = Math.ceil(0.9 * h);
  let s = 0, n = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { s += Math.abs(a.data[y * w + x] - b.data[y * w + x]); n++; }
  return s / n;
}

test('ipE-14. bandpass_notch removes periodic stripes: 640x480 (and 400x400) clean + 30 sin(2 pi (0.15 x + 0.05 y)) -> notch (30, 10, r 4) brings the interior-80% mean abs error vs the clean image from > 15 down to <= 0.6, and the mirror-image setting (30, -10) removes nothing (error >= 15: pins "positive ny = below the centre" / y axis down)', () => {
  for (const [w, h] of [[640, 480], [400, 400]]) {
    const { clean, noisy } = periodicPair(w, h);
    try {
      const before = interiorMae(noisy, clean, w, h);
      assert.ok(before > 15, `noise MAE ${before}`);
      const good = run('bandpass_notch', noisy, { kind: 'notch', shape: 'gaussian', nx: 30, ny: 10, nr: 4 });
      const bad = run('bandpass_notch', noisy, { kind: 'notch', shape: 'gaussian', nx: 30, ny: -10, nr: 4 });
      const neg = run('bandpass_notch', noisy, { kind: 'notch', shape: 'gaussian', nx: -30, ny: -10, nr: 4 }); // the same pair written the other way round
      try {
        const e = interiorMae(good.out, clean, w, h), eb = interiorMae(bad.out, clean, w, h);
        assert.ok(e <= 0.6, `${w}x${h}: notch MAE ${e}`);
        assert.ok(eb >= 15, `${w}x${h}: wrong ny sign MAE ${eb}`);
        sameBytes(good.out, neg.out, 'the pair (+nx,+ny)/(-nx,-ny) is symmetric');
        assert.match(good.ctx.info, /Notch \(Gaussian\) ตัดจุดคู่ที่ \(\+30%, \+10%\) และ \(−30%, −10%\) รัศมี 4%/);
      } finally { good.out.delete(); bad.out.delete(); neg.out.delete(); }
    } finally { clean.delete(); noisy.delete(); }
  }
});

test('ipE-15. bandpass_notch constant images: a notch / ideal notch away from DC and a band-reject that keeps DC give a byte-identical image; a band-pass is all black with the "ภาพเรียบ" note; a band-reject starting at 0% removes DC -> dark image + the darkening warning (result view only, never for band-pass, never in the spectrum view)', () => {
  const m = gray(80, 60, () => 100);
  try {
    for (const p of [{ kind: 'notch' }, { kind: 'notch', shape: 'ideal' }, { kind: 'bandreject', rIn: 10, rOut: 30 }, { kind: 'bandreject', shape: 'ideal', rIn: 10, rOut: 30 }]) {
      const { out, ctx } = run('bandpass_notch', m, p);
      try { sameBytes(out, m, JSON.stringify(p)); assert.ok(!ctx.info.includes('⚠️')); } finally { out.delete(); }
    }
    const bp = run('bandpass_notch', m, {});
    try { assert.ok(bp.out.data.every((v) => v === 0)); assert.match(bp.ctx.info, /ภาพเรียบ/); assert.ok(!bp.ctx.info.includes('⚠️')); } finally { bp.out.delete(); }
    for (const shape of ['gaussian', 'ideal']) {
      const br = run('bandpass_notch', m, { kind: 'bandreject', shape, rIn: 0, rOut: 30 });
      try {
        assert.ok(br.out.data.every((v) => v === 0), `bandreject from 0% (${shape}) is dark`);
        assert.match(br.ctx.info, /· ⚠️ ตัวกรองนี้ตัดค่าเฉลี่ยความสว่าง \(จุดกลางสเปกตรัม\) ทิ้งด้วย ภาพจึงมืดลง/);
      } finally { br.out.delete(); }
    }
    const sp = run('bandpass_notch', m, { kind: 'bandreject', rIn: 0, rOut: 30, view: 'spectrum' });
    try { assert.ok(!sp.ctx.info.includes('⚠️')); } finally { sp.out.delete(); }
    const nt = run('bandpass_notch', m, { kind: 'notch', nx: 0, ny: 0, nr: 5 }); // a notch sitting on DC also removes the mean
    try { assert.ok(nt.out.data.every((v) => v === 0)); assert.match(nt.ctx.info, /⚠️/); } finally { nt.out.delete(); }
    // the threshold is DC gain < 0.5: a Gaussian notch r 5 at nx 7 has DC gain (1 - exp(-49/50))^2 = 0.39 -> warns, at nx 8 it is 0.52 -> silent
    for (const [nx, warns] of [[7, true], [8, false]]) {
      const w = run('bandpass_notch', m, { kind: 'notch', nx, ny: 0, nr: 5 });
      try { assert.equal(w.ctx.info.includes('⚠️'), warns, `notch nx ${nx}`); } finally { w.out.delete(); }
    }
  } finally { m.delete(); }
});

test('ipE-16. bandpass_notch rIn > rOut: swapped silently with the note "สลับขอบใน/ขอบนอกให้แล้ว" -- bytes identical to the written-the-right-way-round call; rOut 0 -> 1 (the minimum); the note never appears for a notch or when the ring is not swapped; numbers clamp (rIn 150 -> 100, nr 0 -> 1, nr 99 -> 30, nx 500 -> 100) and strings / NaN fall back to the defaults', () => {
  const m = sceneGray(96, 64);
  const same = (p, q, label) => {
    const a = run('bandpass_notch', m, p), b = run('bandpass_notch', m, q);
    try { sameBytes(a.out, b.out, label); return [a.ctx.info, b.ctx.info]; } finally { a.out.delete(); b.out.delete(); }
  };
  try {
    for (const kind of ['bandpass', 'bandreject']) {
      const [ia, ib] = same({ kind, rIn: 40, rOut: 10 }, { kind, rIn: 10, rOut: 40 }, `${kind} swap`);
      assert.ok(ia.includes('· สลับขอบใน/ขอบนอกให้แล้ว') && !ib.includes('สลับ'));
      assert.match(ia, /ความถี่ 10–40% ของ Nyquist/);
    }
    const [ieq] = same({ rIn: 30, rOut: 30 }, { rIn: 30, rOut: 30, kind: 'bandpass' }, 'rIn = rOut');
    assert.ok(!ieq.includes('สลับ'), 'rIn = rOut is not swapped');
    const [i0] = same({ rOut: 0, rIn: 0 }, { rOut: 1, rIn: 0 }, 'rOut 0 -> 1');
    assert.match(i0, /ความถี่ 0–1% ของ Nyquist/);
    assert.ok(!same({ kind: 'notch', rIn: 90, rOut: 3 }, { kind: 'notch' }, 'notch ignores the ring')[0].includes('สลับ'));
    same({ kind: 'notch', nr: 0 }, { kind: 'notch', nr: 1 }, 'nr 0 -> 1');
    same({ kind: 'notch', nr: 99 }, { kind: 'notch', nr: 30 }, 'nr 99 -> 30');
    same({ kind: 'notch', nx: 500, ny: -500 }, { kind: 'notch', nx: 100, ny: -100 }, 'nx/ny clamp');
    same({ rIn: 150 }, { rIn: 100 }, 'rIn 150');
    same({ rIn: 'x', rOut: NaN, nx: 'y', kind: 'notch' }, { kind: 'notch' }, 'garbage = defaults');
    same({ rIn: '12', rOut: '33' }, { rIn: 12, rOut: 33 }, 'numeric strings');
    for (const bad of ['constructor', 'toString', '__proto__', 7, null]) same({ kind: bad, shape: bad, view: bad }, {}, `selects ${String(bad)} -> defaults`);
  } finally { m.delete(); }
});

test('ipE-17. bandpass_notch size paths: 2000x100 shrinks to 1920x96 (INTER_AREA) and the result is expanded back to 2000x100 (INTER_LINEAR) with the info note; exactly 1920 wide is NOT shrunk; the spectrum view is 1 channel at the source size; a colour / RGBA direct call equals the call on its gray version; the input is never modified; the 1x1 / 2x1 / 3x5 / 1x50 images keep their size; 1x1 uses a 2x2 spectrum', () => {
  const wide = gray(2000, 100, (x, y) => 128 + ((x >> 3) & 1) * 40 + (y & 1) * 5);
  const exact = gray(1920, 20, (x) => 100 + ((x >> 2) & 1) * 30);
  try {
    const a = run('bandpass_notch', wide, { kind: 'bandreject', rIn: 5, rOut: 20 });
    try {
      assert.equal(a.out.cols, 2000); assert.equal(a.out.rows, 100);
      assert.match(a.ctx.info, /· ภาพใหญ่กว่า 1920 px จึงย่อเหลือ 1920×96 ก่อนกรอง แล้วขยายกลับ/);
      assert.match(a.ctx.info, /สเปกตรัม 1920×96/);
    } finally { a.out.delete(); }
    const s = run('bandpass_notch', wide, { view: 'spectrum' });
    try { assert.equal(s.out.channels(), 1); assert.equal(s.out.cols, 2000); assert.equal(s.out.rows, 100); } finally { s.out.delete(); }
    const e = run('bandpass_notch', exact, {});
    try { assert.ok(!e.ctx.info.includes('จึงย่อเหลือ')); assert.equal(e.out.cols, 1920); } finally { e.out.delete(); }
  } finally { wide.delete(); exact.delete(); }
  // non-integer shrink: 2500x1333 -> 1920 x round(1333 * 1920 / 2500 = 1023.74) = 1024 (a floor would give 1023)
  const odd = gray(2500, 1333, (x, y) => 100 + ((x >> 3) & 1) * 30 + (y & 3));
  try {
    const { out, ctx } = run('bandpass_notch', odd, { kind: 'notch' });
    try {
      assert.match(ctx.info, /ย่อเหลือ 1920×1024 ก่อนกรอง/);
      assert.match(ctx.info, /สเปกตรัม 1920×1024/);
      assert.equal(out.cols, 2500); assert.equal(out.rows, 1333);
    } finally { out.delete(); }
  } finally { odd.delete(); }
  const c = sceneRGB(64, 48), a = toRGBA(c), g = toGray(c);
  const before = bytes(c);
  try {
    const rc = run('bandpass_notch', c, { kind: 'notch', view: 'spectrum' }), ra = run('bandpass_notch', a, { kind: 'notch', view: 'spectrum' }), rg = run('bandpass_notch', g, { kind: 'notch', view: 'spectrum' });
    try { sameBytes(rc.out, rg.out, 'colour = gray'); sameBytes(ra.out, rg.out, 'rgba = gray'); assert.deepEqual(bytes(c), before); } finally { rc.out.delete(); ra.out.delete(); rg.out.delete(); }
  } finally { c.delete(); a.delete(); g.delete(); }
  for (const [w, h, P, Q] of [[1, 1, 2, 2], [2, 1, 2, 2], [3, 5, 4, 6], [1, 50, 2, 50]]) {
    const m = gray(w, h, (x, y) => 40 + x * 30 + y * 7);
    try {
      for (const view of ['result', 'spectrum']) for (const kind of ['bandpass', 'bandreject', 'notch']) {
        const { out, ctx } = run('bandpass_notch', m, { view, kind });
        try {
          assert.equal(out.cols, w); assert.equal(out.rows, h); assert.equal(out.channels(), 1);
          assert.ok(ctx.info.includes(`สเปกตรัม ${P}×${Q}`), `${w}x${h}: ${ctx.info}`);
        } finally { out.delete(); }
      }
    } finally { m.delete(); }
  }
  // 1x1: DFT of a single pixel round-trips (a notch away from DC keeps the value 77)
  const one = gray(1, 1, () => 77);
  try { const { out } = run('bandpass_notch', one, { kind: 'bandreject', rIn: 10, rOut: 30 }); try { assert.equal(out.data[0], 77); } finally { out.delete(); } } finally { one.delete(); }
});

test('ipE-18. bandpass_notch values contract: ctx.values is NEVER set (every path: result / spectrum, shrink, tiny, 0x0), the 0x0 path returns a new 0x0 Mat with info "ภาพว่าง", and the info of every kind states P x Q', () => {
  const empty = new cv.Mat(), m = sceneGray(40, 30), wide = gray(2000, 8, (x) => x & 255);
  try {
    for (const [src, p] of [[m, {}], [m, { view: 'spectrum' }], [m, { kind: 'notch' }], [m, { kind: 'bandreject' }], [wide, {}], [gray(2, 1, () => 3), {}]]) {
      const ctx = {};
      const out = OPS.bandpass_notch.apply(cv, src, p, ctx);
      out.delete();
      assert.ok(!('values' in ctx), `values set: ${JSON.stringify(p)}`);
      assert.ok(/สเปกตรัม \d+×\d+/.test(ctx.info));
    }
    const ctx = {};
    const e = OPS.bandpass_notch.apply(cv, empty, {}, ctx);
    try { assert.equal(ctx.info, 'ภาพว่าง'); assert.ok(!('values' in ctx)); assert.equal(e.rows, 0); assert.equal(e.cols, 0); } finally { e.delete(); }
  } finally { empty.delete(); m.delete(); wide.delete(); }
});

test('ipE-19. bandpass_notch gain shapes (white-box through impulse images): a Gaussian band-pass attenuates DC to 0 and passes the ring centre; the ring edges use "r = sqrt(fx^2 + fy^2) * 100" in % of Nyquist for both axes (a wave along y at 25% passes the same ring as a wave along x at 25%); the ideal ring is inclusive at both ends (lo and hi are kept)', () => {
  const sx = gray(128, 128, (x) => Math.round(128 + 90 * Math.sin((2 * Math.PI * x) / 8))); // 25% along x
  const sy = gray(128, 128, (x, y) => Math.round(128 + 90 * Math.sin((2 * Math.PI * y) / 8))); // 25% along y
  try {
    const run1 = (m, p) => { const r = run('bandpass_notch', m, { shape: 'ideal', kind: 'bandpass', ...p }); const o = Array.from(r.out.data); r.out.delete(); return o; };
    for (const p of [{ rIn: 24, rOut: 26 }, { rIn: 25, rOut: 26 }, { rIn: 24, rOut: 25 }]) {
      assert.ok(corr(run1(sx, p), Array.from(sx.data)) > 0.999, `x wave kept by ${JSON.stringify(p)}`);
      assert.ok(corr(run1(sy, p), Array.from(sy.data)) > 0.999, `y wave kept by ${JSON.stringify(p)}`);
    }
    const out = run1(sx, { rIn: 30, rOut: 60 });
    assert.ok(out.every((v) => v === 0), 'wave outside the ring removed');
  } finally { sx.delete(); sy.delete(); }
});

// ตัวเลขทองของ slicLabels / bandpass_notch — ค่าเหล่านี้ตรวจกับ numpy / cv2 จริง (Python ของ op ที่ export) ตอนเขียนเทสต์: ทุกแถว slic ได้
// count / grid / merged / sha ของ labels ตรงกันทุกค่า · ทุกแถว bandpass_notch คือ (ภาพ, ค่า) ชุดเดียวกับที่ ipE-P3 เทียบ ndiff 0 กับ Python —
// ใช้ปกป้องอัลกอริทึมเมื่อไม่มี Python (ipE-P* ถูกข้าม) · ถ้าเปลี่ยนอัลกอริทึมโดยตั้งใจ ต้องรันเทียบ Python ใหม่แล้วค่อยอัปเดตตาราง
const sha16 = (buf) => crypto.createHash('sha256').update(buf).digest('hex').slice(0, 16);
const EMPTY_CLUSTER_F = [255, 200, 0, 0, 40, 75, 162, 200, 200, 0, 152, 255, 200, 128, 40, 118, 141, 255, 40, 200, 255, 100, 255, 194, 255, 255, 255, 255, 40, 40, 255, 200, 0, 0, 0, 200, 200, 0, 255, 34, 200, 40, 0, 40, 200, 54, 200, 200, 40, 40, 21, 255, 103, 255, 255, 200, 255, 255, 106, 181, 255, 40, 255, 41, 225, 40, 0, 40, 255, 255, 200, 0, 200, 0, 40, 0, 200, 40];

test('ipE-29. golden digests: slicLabels on 7 fixed images (gray / Lab; noise, scene, ramp; K 20..1000; m 1..40; 5 / 10 iterations; one image where a cluster goes empty) gives exactly the count / grid / merged / labels digest checked against the numpy twin, and bandpass_notch gives exactly the output digest of 10 fixed settings (incl. the exact-radius ideal notch bin and the 2000x100 shrink path) checked against cv2 (ipE-P3)', () => {
  const SLIC_GOLD = [
    ['noise gray 61x47 K100 m10', () => noiseGray(61, 47, 1), 100, 10, 10, '72 11 9 1133 98d57e6a22478026'],
    ['scene 80x60 K200 m10', () => sceneRGB(80, 60), 200, 10, 10, '192 16 12 94 ad76ea93a84b75e1'],
    ['scene 97x61 K1000 m5 it5', () => sceneRGB(97, 61, 7), 1000, 5, 5, '997 40 25 305 bee46817c84f9414'],
    ['noise rgb 40x30 K20 m1', () => noiseRGB(40, 30, 4), 20, 1, 10, '6 5 4 498 cfcb780126202a69'],
    ['ramp gray 50x40 K60 m40', () => gray(50, 40, (x, y) => (x * 5 + y * 3) & 255), 60, 40, 10, '63 9 7 0 85058de0b5282694'],
    ['empty cluster 13x6 K28 m5', () => cv.matFromArray(6, 13, cv.CV_8UC1, Uint8Array.from(EMPTY_CLUSTER_F)), 28, 5, 10, '38 8 4 0 72d85e81265e3499'],
    ['scene 33x90 K80 m10', () => sceneRGB(33, 90, 2), 80, 10, 10, '75 5 15 60 0ddf4d869a966142'],
  ];
  for (const [name, mk, K, m, it, want] of SLIC_GOLD) {
    const img = mk();
    try {
      const { F, nch } = featOf(img);
      const r = slicLabels(F, nch, img.cols, img.rows, K, m, it);
      assert.equal(`${r.count} ${r.nx} ${r.ny} ${r.merged} ${sha16(Buffer.from(r.labels.buffer))}`, want, name);
    } finally { img.delete(); }
  }
  const BP_GOLD = [
    [97, 61, {}, 'e64ba5d919291cc8'], [97, 61, { kind: 'bandreject', shape: 'ideal', rIn: 10, rOut: 40 }, 'd8407c7bfebbc3bb'], [97, 61, { kind: 'notch', view: 'spectrum' }, '8fba59e1a52e4e7f'],
    [97, 61, { kind: 'notch', shape: 'ideal', nx: -15, ny: 33.5, nr: 2.5 }, '0c7f592eb30459d1'], [97, 61, { kind: 'bandpass', shape: 'ideal', rIn: 12, rOut: 2, view: 'spectrum' }, 'dbaacee9843a680c'],
    [640, 480, {}, 'c65be690f1817a36'], [640, 480, { kind: 'notch', nx: 30, ny: 10, nr: 4, view: 'spectrum' }, '774467364a4e81ef'], [33, 200, { kind: 'bandreject', rIn: 0, rOut: 25 }, '3c321294c266b1e4'],
    [640, 480, { kind: 'notch', shape: 'ideal' }, '1b5689b3d4fc11b7'], [2000, 100, { kind: 'bandreject', rIn: 10, rOut: 40 }, '95a6f18e702ad880'],
  ];
  for (const [w, h, p, want] of BP_GOLD) {
    const c = sceneRGB(w, h, w * 3 + h), g = toGray(c);
    try {
      const { out } = run('bandpass_notch', g, p);
      try { assert.equal(sha16(Buffer.from(out.data)), want, `${w}x${h} ${JSON.stringify(p)}`); } finally { out.delete(); }
    } finally { c.delete(); g.delete(); }
  }
});

test('ipE-30. the label an SLIC pixel is given BEFORE the first iteration is never used: every pixel lies inside the window of at least one initial centre (x and y checked separately) for sizes up to 70 x 70 and K in {10, 37, 200, 1000} -- so the initial grid-label formula cannot change a result (documents why dropping its x term is an equivalent change)', () => {
  let checked = 0;
  for (let w = 1; w <= 70; w += (w < 12 ? 1 : 7)) for (let h = 1; h <= 70; h += (h < 12 ? 1 : 5)) for (const K of [10, 37, 200, 1000]) {
    const S = Math.sqrt((w * h) / K);
    const nx = Math.max(1, Math.min(w, Math.floor(w / S + 0.5))), ny = Math.max(1, Math.min(h, Math.floor(h / S + 0.5)));
    const covered = (n, size, i) => {
      for (let j = 0; j < n; j++) { const c = Math.floor(((2 * j + 1) * size) / (2 * n)); if (Math.ceil(c - S) <= i && i <= Math.floor(c + S)) return true; }
      return false;
    };
    for (let x = 0; x < w; x++) assert.ok(covered(nx, w, x), `x ${x} of ${w}x${h} K ${K} not covered`);
    for (let y = 0; y < h; y++) assert.ok(covered(ny, h, y), `y ${y} of ${w}x${h} K ${K} not covered`);
    checked++;
  }
  assert.ok(checked > 300);
});

test('ipE-31. slic empty cluster: a seed whose pixels are all taken by neighbours keeps its old centre (cnt > 0 guard) instead of becoming NaN -- 13x6 gray with K 28 m 5 and 10 iterations ends with 38 superpixels (36 without the guard), reached through the op too (K >= 10); the 5 values / info stay consistent', () => {
  const r = slicLabels(Uint8Array.from(EMPTY_CLUSTER_F), 1, 13, 6, 28, 5, 10);
  assert.equal(r.count, 38);
  const m = cv.matFromArray(6, 13, cv.CV_8UC1, Uint8Array.from(EMPTY_CLUSTER_F));
  try {
    const { out, ctx } = run('slic', m, { count: 28, compactness: 5, iterations: 10, view: 'mean' });
    try {
      assert.equal(scalarMap(ctx).sp_count, 38);
      assert.equal(infoCount(ctx.info), 38);
      assert.ok(Number.isFinite(scalarMap(ctx).sp_mean_area_px));
    } finally { out.delete(); }
  } finally { m.delete(); }
});

test('ipE-32. bandpass flat threshold is exactly 1: 256x256 with 100 / 101 stripes (period 16, 12.5%) only in the top 11 rows -> the ideal 10-15 band-pass is NOT flat (stretched to 0..255); with only 10 rows its range falls just below 1 -> flat: all black + the "ภาพเรียบ" note', () => {
  for (const [rows, flat] of [[11, false], [10, true]]) {
    const m = gray(256, 256, (x, y) => 100 + (y < rows ? (x >> 3) & 1 : 0));
    try {
      const { out, ctx } = run('bandpass_notch', m, { kind: 'bandpass', shape: 'ideal', rIn: 10, rOut: 15 });
      try {
        assert.equal(ctx.info.includes('ไม่มีรายละเอียดในช่วงความถี่นี้'), flat, `${rows} rows`);
        assert.equal(Math.max(...out.data), flat ? 0 : 255, `${rows} rows: max`);
      } finally { out.delete(); }
    } finally { m.delete(); }
  }
});

test('ipE-33. bandpass_notch spectrum view stretches ranges below 1: a single impulse (value 1) at (5, 5) of a 32x32 black image has |F| = 1 everywhere; an ideal band-reject 30-60% keeps ~part of it -> log(1 + |F|) = 0.693 where kept and 0 where cut (range 0.69, between 1e-3 and 1) -> the 8-bit view is stretched to 0 / 255 (not all zero); a perfectly flat spectrum (a ring between two bin radii cuts nothing) gives an all-zero view', () => {
  const m = gray(32, 32, (x, y) => (x === 5 && y === 5 ? 1 : 0));
  try {
    const a = run('bandpass_notch', m, { kind: 'bandreject', shape: 'ideal', rIn: 30, rOut: 60, view: 'spectrum' });
    try {
      const vals = new Set(a.out.data);
      assert.ok(vals.has(255) && vals.has(0), `values ${[...vals].join(',')}`);
    } finally { a.out.delete(); }
    const b = run('bandpass_notch', m, { kind: 'bandreject', shape: 'ideal', rIn: 2, rOut: 2.4, view: 'spectrum' });
    try { assert.ok(b.out.data.every((v) => v === 0), 'flat spectrum -> zeros'); } finally { b.out.delete(); }
  } finally { m.delete(); }
});

// ===================================================================
// static — โครงสร้าง / UI / codegen / ข้อความ
// ===================================================================
const readSrc = (rel) => fs.readFileSync(new URL(`../src/renderer/${rel}`, import.meta.url), 'utf8');
const CS = (space) => ({ id: 'color_space', params: { mode: 'colorspace', colorspace__space: space } });
const STEP = (id, mode, extra = {}) => ({ id, params: { ...defaultParams(id), mode, ...extra } });
const OPSTEP = (opId, extra = {}) => {
  const prefixed = {};
  for (const [k, v] of Object.entries(extra)) prefixed[`${opId}__${k}`] = v;
  return STEP(HOST[opId], opId, prefixed);
};

test('ipE-20. (static structure) the 2 ops exist with needs any / gray (no "mode" param key), FILTERS is still 26, colour = color_inrange, kmeans_segment, white_balance, invert, slic (default color_inrange) and frequency = fft_magnitude, fft_filter, bandpass_notch (default fft_magnitude); defaultParams() = OPS defaults (prefixed); the merged nodes dispatch to the new ops; needsOf follows the mode; the legacy groups do not offer them; select options / defaults match the spec', () => {
  assert.deepEqual(NEW_OPS.map((id) => OPS[id] && OPS[id].needs), ['any', 'gray']);
  for (const id of NEW_OPS) for (const p of OPS[id].params) assert.notEqual(p.key, 'mode');
  assert.equal(Object.keys(FILTERS).length, 35); // 26 + 7 กลุ่มระดับหมวด (2026-10-07) + morph_tools (2026-10-18) + image_registration (2026-10-24)
  const modes = (f) => FILTERS[f].params.find((p) => p.key === 'mode').options.map((o) => o.value);
  assert.deepEqual(modes('colour'), ['color_inrange', 'kmeans_segment', 'white_balance', 'invert', 'slic']);
  // op ใหม่ batch 4 ต่อท้าย — ห้ามสลับลำดับเดิม
  assert.deepEqual(modes('frequency').slice(0, 3), ['fft_magnitude', 'fft_filter', 'bandpass_notch']);
  assert.equal(defaultParams('colour').mode, 'color_inrange');
  assert.equal(defaultParams('frequency').mode, 'fft_magnitude');
  for (const legacy of ['segmentation', 'restoration', 'enhancement']) for (const id of NEW_OPS) assert.ok(!modes(legacy).includes(id));
  for (const id of NEW_OPS) {
    const dp = defaultParams(HOST[id]);
    for (const p of OPS[id].params) assert.deepEqual(dp[`${id}__${p.key}`], p.default, `${HOST[id]}.${id}__${p.key}`);
  }
  assert.equal(needsOf(FILTERS.colour, { mode: 'slic' }), 'any');
  assert.equal(needsOf(FILTERS.frequency, { mode: 'bandpass_notch' }), 'gray');
  const c = sceneRGB(80, 60), g = sceneGray(80, 60);
  try {
    for (const [id, src] of [['slic', c], ['bandpass_notch', g]]) {
      const viaNode = FILTERS[HOST[id]].apply(cv, src, { ...defaultParams(HOST[id]), mode: id }, {});
      const direct = OPS[id].apply(cv, src, {}, {});
      try { sameBytes(viaNode, direct, `${id} through FILTERS.${HOST[id]}`); } finally { viaNode.delete(); direct.delete(); }
    }
  } finally { c.delete(); g.delete(); }
  const spec = (id, k) => OPS[id].params.find((p) => p.key === k);
  assert.deepEqual(OPS.slic.params.map((p) => p.key), ['count', 'compactness', 'iterations', 'view']);
  assert.deepEqual(['count', 'compactness', 'iterations'].map((k) => [spec('slic', k).min, spec('slic', k).max, spec('slic', k).step, spec('slic', k).default]), [[10, 1000, 10, 200], [1, 40, 1, 10], [1, 20, 1, 10]]);
  assert.deepEqual(spec('slic', 'view').options.map((o) => o.value), ['boundary', 'mean']);
  assert.equal(spec('slic', 'view').default, 'boundary');
  assert.deepEqual(OPS.bandpass_notch.params.map((p) => p.key), ['kind', 'shape', 'rIn', 'rOut', 'nx', 'ny', 'nr', 'view']);
  assert.deepEqual(['rIn', 'rOut', 'nx', 'ny', 'nr'].map((k) => [spec('bandpass_notch', k).min, spec('bandpass_notch', k).max, spec('bandpass_notch', k).step, spec('bandpass_notch', k).default]),
    [[0, 100, 1, 5], [1, 100, 1, 30], [-100, 100, 0.5, 25], [-100, 100, 0.5, 0], [1, 30, 0.5, 5]]);
  assert.deepEqual(spec('bandpass_notch', 'kind').options.map((o) => o.value), ['bandpass', 'bandreject', 'notch']);
  assert.deepEqual(spec('bandpass_notch', 'shape').options.map((o) => o.value), ['gaussian', 'ideal']);
  assert.deepEqual(spec('bandpass_notch', 'view').options.map((o) => o.value), ['result', 'spectrum']);
  assert.deepEqual(['kind', 'shape', 'view'].map((k) => spec('bandpass_notch', k).default), ['bandpass', 'gaussian', 'result']);
  assert.equal(signedText(25), '+25'); assert.equal(signedText(-25), '−25'); assert.equal(signedText(0), '0'); assert.equal(signedText(-0), '0'); assert.equal(signedText(12.5), '+12.5');
});

test('ipE-21. palette-panel.js: COLOR = grayscale, colorspace, channel_split, rgb_adjust, color_inrange, kmeans_segment, slic, white_balance, invert; FREQUENCY DOMAIN = fft_magnitude, fft_filter, bandpass_notch; 12 sections; OP_TO_FILTER resolves each new op to its host node; each new tile is in exactly one section; the 2 glyphs (⊞ ◙) differ from every other OP_ICON, every GROUP_ICON and the 📊/📌 markers and appear nowhere in app.js / detail-panel.js / editor.js', () => {
  assert.equal(TOOL_SECTIONS.length, 12);
  // op ใหม่ batch 4 ต่อท้าย — ห้ามสลับลำดับเดิม (IP batch 4A เพิ่ม pseudocolor)
  assert.deepEqual(TOOL_SECTIONS.find((s) => s.title === 'COLOR').ops.slice(0, 9), ['grayscale', 'colorspace', 'channel_split', 'rgb_adjust', 'color_inrange', 'kmeans_segment', 'slic', 'white_balance', 'invert']);
  assert.deepEqual(TOOL_SECTIONS.find((s) => s.title === 'FREQUENCY DOMAIN').ops.slice(0, 3), ['fft_magnitude', 'fft_filter', 'bandpass_notch']);
  for (const id of NEW_OPS) {
    // 2026-10-07: palette ชี้กลุ่มระดับหมวด (HOST = กลุ่มเดิม ยังคงไว้ใน FILTERS)
    assert.equal(OP_TO_FILTER[id], { slic: 'color_tools', bandpass_notch: 'frequency' }[id]);
    assert.equal(TOOL_SECTIONS.filter((s) => s.ops.includes(id)).length, 1);
  }
  const src = readSrc('ui/palette-panel.js');
  const body = src.slice(src.indexOf('const OP_ICON = {'), src.indexOf('};', src.indexOf('const OP_ICON = {')));
  const icons = {};
  for (const m of body.matchAll(/([a-z_0-9]+): "([^"]+)"/g)) icons[m[1]] = m[2];
  assert.deepEqual([icons.slic, icons.bandpass_notch], ['⊞', '◙']);
  const groupGlyphs = new Set(Object.values(GROUP_ICON));
  const app = readSrc('app.js'), dp = readSrc('ui/detail-panel.js'), ed = readSrc('graph/editor.js');
  for (const id of NEW_OPS) {
    const others = Object.entries(icons).filter(([k]) => k !== id).map(([, v]) => v);
    assert.ok(!others.includes(icons[id]), `${id} collides with another OP_ICON`);
    assert.ok(!groupGlyphs.has(icons[id]), `${id} collides with a GROUP_ICON`);
    assert.ok(!['📊', '📌'].includes(icons[id]));
    for (const [name, text] of [['app.js', app], ['detail-panel.js', dp], ['editor.js', ed]]) assert.ok(!text.includes(icons[id]), `${id} glyph used by ${name}`);
  }
});

test('ipE-22. ui-meta.js OP_PRIMARY uses the clamped values the ops really use (pnum): slic "~200 ก้อน · compactness 10" (+ " · สีเฉลี่ย" for mean), bandpass_notch "Band-pass 5–30% · Gaussian" / "Band-reject 25–35% · Ideal" / "Notch (+25%, 0%) r 5%" (+ " · สเปกตรัม" for the spectrum view); out-of-range / garbage / "constructor" values show what the op does (rIn > rOut shown swapped)', () => {
  assert.equal(OP_PRIMARY.slic({}), '~200 ก้อน · compactness 10');
  assert.equal(OP_PRIMARY.slic({ count: 5000, compactness: 0, view: 'mean' }), '~1000 ก้อน · compactness 1 · สีเฉลี่ย');
  assert.equal(OP_PRIMARY.slic({ count: '55.6', compactness: 'x', view: 'constructor' }), '~56 ก้อน · compactness 10');
  assert.equal(OP_PRIMARY.bandpass_notch({}), 'Band-pass 5–30% · Gaussian');
  assert.equal(OP_PRIMARY.bandpass_notch({ kind: 'bandreject', shape: 'ideal', rIn: 25, rOut: 35 }), 'Band-reject 25–35% · Ideal');
  assert.equal(OP_PRIMARY.bandpass_notch({ kind: 'bandreject', shape: 'ideal', rIn: 35, rOut: 25 }), 'Band-reject 25–35% · Ideal');
  assert.equal(OP_PRIMARY.bandpass_notch({ kind: 'notch' }), 'Notch (+25%, 0%) r 5%');
  assert.equal(OP_PRIMARY.bandpass_notch({ kind: 'notch', nx: -12.5, ny: 40, nr: 99, view: 'spectrum' }), 'Notch (−12.5%, +40%) r 30% · สเปกตรัม');
  assert.equal(OP_PRIMARY.bandpass_notch({ view: 'spectrum', rIn: 0, rOut: 0 }), 'Band-pass 0–1% · Gaussian · สเปกตรัม');
  assert.equal(OP_PRIMARY.bandpass_notch({ kind: 'constructor', shape: 'constructor', view: 'constructor' }), 'Band-pass 5–30% · Gaussian');
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

test('ipE-23. (static read) every numeric param of the 2 ops is clamped to EXACTLY the UI default/min/max in registry.js (numIn / intIn) and in codegen.js (clampNum); every numeric param is covered; selects are read with ownKey / pickOwn (own properties only)', () => {
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
  const regCheck = (id, k, fn) => { const m = block(reg, id).match(new RegExp(`${fn}\\(p\\.${k}, ${num}, ${num}, ${num}\\)`)); assert.ok(m, `registry ${id}.${k}`); assert.deepEqual(m.slice(1).map(Number), spec(id, k), `registry ${id}.${k}`); covered.add(`${id}.${k}`); };
  const genCheck = (id, k) => { const m = block(gen, id).match(new RegExp(`clampNum\\(p\\.${k}, ${num}, ${num}, ${num}\\)`)); assert.ok(m, `codegen ${id}.${k}`); assert.deepEqual(m.slice(1).map(Number), spec(id, k), `codegen ${id}.${k}`); };
  regCheck('slic', 'count', 'intIn'); regCheck('slic', 'compactness', 'numIn'); regCheck('slic', 'iterations', 'intIn');
  for (const k of ['rIn', 'rOut', 'nx', 'ny', 'nr']) regCheck('bandpass_notch', k, 'numIn');
  for (const k of ['count', 'compactness', 'iterations']) genCheck('slic', k);
  for (const k of ['rIn', 'rOut', 'nx', 'ny', 'nr']) genCheck('bandpass_notch', k);
  assert.match(block(gen, 'slic'), /Math\.round\(clampNum\(p\.count/);
  assert.match(block(gen, 'slic'), /Math\.round\(clampNum\(p\.iterations/);
  for (const id of NEW_OPS) for (const p of OPS[id].params) if (p.type === 'number') assert.ok(covered.has(`${id}.${p.key}`), `${id}.${p.key} not covered`);
  assert.match(block(reg, 'slic'), /ownKey\(SLIC_VIEWS, p\.view, "boundary"\)/);
  for (const [k, d] of [['kind', 'bandpass'], ['shape', 'gaussian'], ['view', 'result']]) assert.match(block(reg, 'bandpass_notch'), new RegExp(`ownKey\\(BANDPASS_${k.toUpperCase()}S, p\\.${k}, "${d}"\\)`));
  assert.match(block(gen, 'slic'), /pickOwn\(SLIC_VIEW_PY, p\.view, "boundary"\)/);
  assert.match(block(gen, 'bandpass_notch'), /pickOwn\(BANDPASS_KIND_PY, p\.kind, "bandpass"\)/);
  assert.match(reg, /const SLIC_MAX_SIDE = 320;/);
  assert.match(reg, /const BANDPASS_MAX_SIDE = 1920;/);
});

const PY_VARIANTS = [
  ['slic', {}], ['slic', { view: 'mean', count: 50, compactness: 20, iterations: 5 }], ['slic', { count: 1000, compactness: 5 }],
  ['bandpass_notch', {}], ['bandpass_notch', { kind: 'bandreject', shape: 'ideal', rIn: 20, rOut: 10 }], ['bandpass_notch', { kind: 'notch', view: 'spectrum' }],
  ['bandpass_notch', { kind: 'notch', shape: 'ideal', nx: -12.5, ny: 7.5, nr: 2.5 }], ['bandpass_notch', { kind: 'bandpass', shape: 'ideal', view: 'spectrum' }],
];

test('ipE-24. codegen: both export real single-image Python (never "WARNING: skipped" / "ข้าม" / round( / ** 0.5 / import math), embed the clamped values as literals, use np.floor(x + 0.5) / np.sqrt / np.ceil, emit the gain only for the chosen kind / shape with the spec expression text, and keep the gray / colour state (slic boundary -> colour, mean -> same; bandpass_notch -> gray)', () => {
  for (const [id, p] of PY_VARIANTS) {
    const code = buildPythonCode([OPSTEP(id, p)]);
    for (const bad of ['WARNING: skipped', 'ข้าม', 'round(', '** 0.5', 'import math', 'np.round', 'math.']) assert.ok(!code.includes(bad), `${id} ${JSON.stringify(p)}: contains ${bad}`);
    assert.match(code, new RegExp(`# ── ขั้นที่ 1: ${OPS[id].name.replace(/[()]/g, '\\$&')} \\(${id}\\) ──`));
    for (const line of code.split('\n')) if (/^\s*print\(/.test(line)) assert.ok(/^[\x20-\x7e]*$/.test(line), `non-ASCII print: ${line}`);
  }
  const s = buildPythonCode([OPSTEP('slic', { count: 5, compactness: 99, iterations: 0, view: 'constructor' })]);
  assert.match(s, /slic_labels\(feat, a_cols, a_rows, 10, 40, 1\)/);
  assert.ok(s.includes('np.ceil(cx[k] - S)') && s.includes('np.floor(cx[k] + S)') && s.includes('d < sub') && s.includes('((2 * np.arange(rows) + 1) * a_rows) // (2 * rows)'));
  assert.ok(s.includes('def slic_labels(feat, w, h, K, m, iters):') && s.includes('limit = (n_px // nc) >> 2'));
  assert.ok(s.includes('d = dl * dl + da * da + db * db + (dx * dx + dy * dy) * w_s') && s.includes('analysis.ndim == 2'));
  assert.ok(s.includes('np.bincount(') && s.includes('weights='));
  const m = buildPythonCode([OPSTEP('slic', { view: 'mean' })]);
  assert.ok(m.includes('np.floor(np.bincount(') && m.includes('/ cnt + 0.5)'));
  const bp = buildPythonCode([OPSTEP('bandpass_notch', { rIn: 90, rOut: 3, kind: 'bandreject', shape: 'ideal' })]);
  assert.ok(bp.includes('lo, hi = 3, 90') && bp.includes('bp = ((r >= lo) & (r <= hi)).astype(np.float64)') && bp.includes('H = 1 - bp'));
  assert.ok(!bp.includes('np.exp(') && !bp.includes('d1 <= r2'), 'only the chosen kind / shape');
  const gb = buildPythonCode([OPSTEP('bandpass_notch', {})]);
  assert.ok(gb.includes('bp = (1 - np.exp(-(r * r) / (2 * lo * lo)) if lo > 0 else 1) * np.exp(-(r * r) / (2 * hi * hi))') && gb.includes('lo, hi = 5, 30') && gb.includes('H = bp'));
  const nt = buildPythonCode([OPSTEP('bandpass_notch', { kind: 'notch', nx: 'x', ny: 700, nr: 0 })]);
  assert.ok(nt.includes('nx, ny, nr = 25, 100, 1') && nt.includes('H = (1 - np.exp(-d1 / (2 * r2))) * (1 - np.exp(-d2 / (2 * r2)))'));
  assert.ok(nt.includes('a, b = fx * 100 - nx, fy * 100 - ny') && nt.includes('c, d = fx * 100 + nx, fy * 100 + ny'));
  assert.ok(nt.includes('fx = (np.where(xi < hp, xi, xi - P) / hp)[None, :]') && nt.includes('spec = spec * H[:, :, None]'));
  assert.ok(nt.includes('cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT') && nt.includes('dc_gain = float(H[0, 0])') && nt.includes('np.broadcast_to(H, (Q, P))'));
  assert.ok(nt.includes('scale = min(1.0, 1920 / max(rows, cols))') && nt.includes('flags=cv2.DFT_COMPLEX_OUTPUT') && nt.includes('padded.astype(np.float64)'));
  assert.ok(nt.includes('np.clip(np.floor(inv + 0.5), 0, 255)'));
  const sp = buildPythonCode([OPSTEP('bandpass_notch', { view: 'spectrum' })]);
  assert.ok(sp.includes('np.fft.fftshift(np.log(1 + np.sqrt(') && sp.includes('mx - mn < 1e-3') && !sp.includes('cv2.DFT_INVERSE'));
  assert.ok(buildPythonCode([OPSTEP('bandpass_notch', {})]).includes('flat = bool(mx - mn < 1)'));
  // ctx.gray / mirrored state: slic boundary -> colour, mean -> same, bandpass -> gray
  const autoGray = (pipe) => buildPythonCode(pipe).split('(แปลงขาวดำอัตโนมัติ)').length - 1;
  assert.equal(autoGray([OPSTEP('slic', {}), STEP('shape_analysis', 'regionprops')]), 1, 'boundary is colour');
  assert.equal(autoGray([OPSTEP('slic', { view: 'mean' }), STEP('shape_analysis', 'regionprops')]), 1, 'mean of a colour image is colour');
  assert.equal(autoGray([OPSTEP('bandpass_notch', {}), STEP('shape_analysis', 'regionprops')]), 1, 'bandpass output is gray: only the conversion BEFORE it, none after');
  assert.equal(buildPythonCode([STEP('color_space', 'grayscale'), OPSTEP('slic', { view: 'mean' }), STEP('shape_analysis', 'regionprops')]).split('(แปลงขาวดำอัตโนมัติ)').length - 1, 0, 'mean of gray stays gray');
  assert.equal(buildPythonCode([STEP('color_space', 'grayscale'), OPSTEP('slic', {}), STEP('shape_analysis', 'regionprops')]).split('(แปลงขาวดำอัตโนมัติ)').length - 1, 1, 'boundary of gray is colour');
});

test('ipE-25. teaching text: names / descs / labels / hints / options / every info message of the 2 ops and the exported Python of every variant never say "ที่สุด"; each desc states the spec points (slic 1-6, bandpass_notch 1-8)', () => {
  const texts = [];
  for (const id of NEW_OPS) {
    const o = OPS[id];
    texts.push(o.name, o.desc);
    for (const p of o.params) { texts.push(p.label, p.desc || '', p.hint || ''); for (const op of p.options || []) texts.push(op.label, op.hint || ''); }
  }
  const c = sceneRGB(60, 40), g = sceneGray(60, 40), wide = gray(2000, 10, () => 9), flat = gray(30, 30, () => 5), empty = new cv.Mat(), wc = sceneRGB(400, 20);
  try {
    for (const [id, src, p] of [['slic', c, {}], ['slic', g, { view: 'mean' }], ['slic', wc, {}], ['slic', empty, {}],
      ['bandpass_notch', g, {}], ['bandpass_notch', g, { kind: 'bandreject', rIn: 50, rOut: 5, view: 'spectrum' }], ['bandpass_notch', g, { kind: 'notch', shape: 'ideal' }], ['bandpass_notch', wide, {}],
      ['bandpass_notch', flat, {}], ['bandpass_notch', flat, { kind: 'bandreject', rIn: 0 }], ['bandpass_notch', empty, {}]]) texts.push(runInfo(id, src, p).info);
  } finally { for (const m of [c, g, wide, flat, empty, wc]) m.delete(); }
  for (const [id, p] of PY_VARIANTS) texts.push(buildPythonCode([OPSTEP(id, p)]));
  texts.push(readSrc('filters/ui-meta.js').slice(readSrc('filters/ui-meta.js').indexOf('slic: (p)'), readSrc('filters/ui-meta.js').indexOf('fft_magnitude: () =>')));
  for (const t of texts) assert.ok(!t.includes('ที่สุด'), `"ที่สุด" in: ${t.slice(0, 120)}`);
  const sd = OPS.slic.desc, bd = OPS.bandpass_notch.desc;
  for (const re of [/superpixel/, /เชื่อมต่อกัน/, /สีและตำแหน่ง/, /K-means/, /ดูแค่สี/, /กระจัดกระจาย/, /compactness/, /สี่เหลี่ยมสม่ำเสมอ/, /เกาะตามขอบสี/, /ไม่ได้บอกว่าอะไรคือวัตถุ/, /ขั้นเตรียมข้อมูล/, /ไม่เท่ากับที่ตั้งไว้/, /ปัดเศษ/, /เศษชิ้นเล็ก/, /MATLAB/, /320 px/, /ขั้นบันได/, /Achanta 2012/, /ไม่ตรงกันทุกพิกเซล/]) assert.match(sd, re, `slic desc ${re}`);
  for (const re of [/Band-pass/, /High-pass/, /Low-pass/, /Band-reject/, /Notch/, /คู่สมมาตร/, /ลายซ้ำ/, /เบลอ/, /Nyquist/, /200\/T/, /แกน y/, /ต่ำกว่าจุดกลาง/, /สเปกตรัมหลังกรอง/, /FFT Magnitude/, /640 px/, /1920 px/, /ringing/, /Gaussian/, /วนซ้ำ/, /0–255/, /0\/255/, /ขาวดำก่อน/, /หลายคู่/]) assert.match(bd, re, `bandpass desc ${re}`);
});

test('ipE-26. (static) channel order of the exported Python follows ctx.mirrored through isMirrored only: slic alone -> COLOR_BGR2Lab and the reversed yellow (0, 255, 255); right after Color Space (and through a colour -> colour step after it) -> COLOR_RGB2Lab and the runtime yellow choice `(0, 255, 255) if img.ndim == 2 else (255, 255, 0)`; after Grayscale -> BGR2Lab path with ndim check (the analysis is chosen by analysis.ndim at runtime); the 2-image path (P-21) is untouched', () => {
  const plain = buildPythonCode([OPSTEP('slic')]);
  assert.ok(plain.includes('cv2.cvtColor(analysis, cv2.COLOR_BGR2Lab)') && !plain.includes('cv2.COLOR_RGB2Lab'));
  assert.ok(plain.includes('edge_color = (0, 255, 255)  #') && !plain.includes('(255, 255, 0)'));
  for (const pre of [[CS('hsv')], [CS('lab'), STEP('blur', 'gaussian_blur')]]) {
    const code = buildPythonCode([...pre, OPSTEP('slic')]);
    assert.ok(code.includes('feat = analysis if analysis.ndim == 2 else cv2.cvtColor(analysis, cv2.COLOR_RGB2Lab)'));
    assert.ok(code.includes('edge_color = (0, 255, 255) if img.ndim == 2 else (255, 255, 0)'));
  }
  const g = buildPythonCode([STEP('color_space', 'grayscale'), OPSTEP('slic')]);
  assert.ok(g.includes('feat = analysis if analysis.ndim == 2 else cv2.cvtColor(analysis, cv2.COLOR_BGR2Lab)') && g.includes('edge_color = (0, 255, 255)  #'));
  const t = buildPythonCode([STEP('threshold', 'binary_threshold'), OPSTEP('slic')]);
  assert.ok(t.includes('analysis.ndim == 2') && t.includes('cv2.COLOR_GRAY2BGR'));
  // the 2-image path never emits our GEN with a mirrored flag — a pipeline without slic stays as before
  const two = buildPythonCode([STEP('image_arithmetic', 'imabsdiff')]);
  assert.ok(!two.includes('slic_labels'));
});

// Python ของ op เดิมต้องไม่เปลี่ยนแม้แต่ตัวอักษร — hash ของข้อความบล็อก GEN จาก codegen.js ก่อนชิ้น E (backups/2026-10-04-before-ip-batch2-E)
test('ipE-27. the exported Python of the existing ops is byte-identical to before this batch: the GEN blocks of kmeans_segment, fft_magnitude + fft_filter, and the generated code of a pipeline using them match the pre-E snapshot (sha256 of the block text)', () => {
  const gen = readSrc('codegen.js');
  const cut = (a, b) => { const i = gen.indexOf(a); assert.ok(i >= 0, a); const j = gen.indexOf(b, i); assert.ok(j > i, b); return gen.slice(i, j); };
  const sha = (s) => crypto.createHash('sha256').update(s).digest('hex');
  assert.equal(sha(cut('\n  kmeans_segment: {', '\n  white_balance: {')), EXPECT_SHA.kmeans);
  assert.equal(sha(cut('\n  fft_magnitude: {', '\n  // ---- Image Arithmetic (Tier 2, 18a) ----')), EXPECT_SHA.fft);
  const k = buildPythonCode([STEP('colour', 'kmeans_segment')]);
  assert.ok(k.includes('Z = img.reshape((-1, 3)).astype(np.float32)') && !k.includes('slic'));
  const f = buildPythonCode([STEP('frequency', 'fft_filter')]);
  assert.ok(f.includes('scale = min(1.0, 640 / max(rows, cols))') && !f.includes('1920'));
});
// B12 (2026-10-17): re-pin ของ fft — เอา wrapper GEN.fft_filter.gen (batch 4A) เข้ามาอยู่ในบล็อก · Python ที่ export ไม่เปลี่ยนแม้ตัวอักษรเดียว (พิสูจน์ด้วยเทสต์ bl-B12 ใน tests/backlog-2026-10-17.test.mjs)
const EXPECT_SHA = { kmeans: 'c3f022d7be8e5c1061069868d7b01fb2b52caaad28ed9da9c40c6508d29b0c22', fft: '08ffc912d2b8d266b1bbf15a611a1c0e950d5c21d84acd3953e10a8eb02184f1' };

// ===================================================================
// parity กับ Python จริง (cv2 5.0 / numpy 2.x)
// ===================================================================
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'ip-batch2-E-2026-10-04-'));
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
function runPython(code, src, { swapOut = true } = {}) {
  assert.equal(src.channels(), 3, 'cv2.imread gives 3 channels');
  let patched = code.replace('img = cv2.imread(path)', `img = np.fromfile(path, dtype=np.uint8).reshape((${src.rows}, ${src.cols}, 3))`);
  const mainIdx = patched.indexOf('if __name__ == "__main__":');
  assert.ok(mainIdx > 0);
  patched = patched.slice(0, mainIdx) + '\nimport sys\nresult = process(sys.argv[1])\nprint("NDIM", result.ndim)\nresult.tofile(sys.argv[2])\n';
  const id = Math.random().toString(36).slice(2);
  const scriptPath = path.join(TMP_DIR, `s_${id}.py`), inPath = path.join(TMP_DIR, `i_${id}.bin`), outPath = path.join(TMP_DIR, `o_${id}.bin`);
  fs.writeFileSync(scriptPath, patched, 'utf8');
  fs.writeFileSync(inPath, swapRB3(Buffer.from(src.data)));
  const stdout = execFileSync('python', [scriptPath, inPath, outPath], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' }, maxBuffer: 1 << 26 });
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
function ndiffOf(appMat, pyBytes) {
  const d = appMat.data;
  assert.equal(pyBytes.length, d.length, 'output size');
  let n = 0;
  for (let i = 0; i < d.length; i++) if (d[i] !== pyBytes[i]) n++;
  return n;
}
function pyLine(stdout, re) { const m = stdout.match(re); assert.ok(m, `python stdout lacks ${re}`); return m; }

function slicParity(src, p, label, { pre = [], mirroredOut = true } = {}) {
  const pipe = [...pre, OPSTEP('slic', p)];
  const py = runPython(buildPythonCode(pipe), src, { swapOut: mirroredOut });
  const app = appRun(pipe, src);
  try {
    const n = ndiffOf(app.mat, py.bytes);
    assert.equal(n, 0, `${label}: ${n} bytes differ`);
    assert.equal(py.ndim === 3 ? 3 : 1, app.mat.channels(), `${label}: channels`);
    const m = pyLine(py.stdout, /^SLIC: (\d+) superpixels \(requested (\d+), grid (\d+)x(\d+), merged (\d+) small fragments\), analysis (\d+)x(\d+)$/m);
    const [count, req, nx, ny, merged, aw, ah] = m.slice(1).map(Number);
    const v = scalarMap(app.ctx), [inx, iny] = infoGrid(app.ctx.info);
    assert.equal(count, v.sp_count, `${label}: count`);
    assert.equal(req, v.sp_requested, `${label}: requested`);
    assert.deepEqual([nx, ny], [inx, iny], `${label}: grid`);
    assert.equal(nx * ny, v.sp_grid);
    assert.equal(merged, infoMerged(app.ctx.info), `${label}: merged`);
    const sh = app.ctx.info.match(/ภาพย่อ (\d+)×(\d+)/);
    const scale = Math.min(1, 320 / Math.max(src.cols, src.rows));
    assert.deepEqual([aw, ah], sh ? [Number(sh[1]), Number(sh[2])] : [src.cols, src.rows], `${label}: analysis size`);
    if (scale < 1) assert.ok(sh, `${label}: shrink note`);
    return { count };
  } finally { app.mat.delete(); }
}

test('ipE-P1. codegen parity slic (real Python): ndiff 0 over the whole image, count / requested / grid / merged / analysis size in the stdout = the app -- both views x gray (after Grayscale) / colour x 97x61, 320x240, 640x480, 1280x720 (gray), 700x500, 3x40, 1x1 x {defaults; K 50 m 20 it 5; K 1000 m 5}', PY_SKIP, () => {
  const sizes = [[97, 61], [320, 240], [640, 480], [700, 500], [3, 40], [1, 1]];
  const settings = [{}, { count: 50, compactness: 20, iterations: 5 }, { count: 1000, compactness: 5 }];
  let n = 0, total = 0;
  for (const [w, h] of sizes) {
    const src = sceneRGB(w, h, w + h);
    try {
      for (const view of ['boundary', 'mean']) for (const s of settings) {
        if ((w >= 640) && s.count === 1000 && view === 'mean') continue; // keep the Python time down: K 1000 at 640+ is run in boundary view only
        total += slicParity(src, { ...s, view }, `colour ${w}x${h} ${view} ${JSON.stringify(s)}`).count; n++;
      }
      if (w <= 320 || w === 700) for (const view of ['boundary', 'mean']) {
        total += slicParity(src, { view, ...settings[1] }, `gray ${w}x${h} ${view}`, { pre: [STEP('color_space', 'grayscale')] }).count; n++;
      }
    } finally { src.delete(); }
  }
  const big = sceneRGB(1280, 720, 4);
  try {
    for (const view of ['boundary', 'mean']) { total += slicParity(big, { view }, `gray 1280x720 ${view}`, { pre: [STEP('color_space', 'grayscale')] }).count; n++; }
  } finally { big.delete(); }
  assert.ok(n >= 40 && total > 500, `cases ${n}, superpixels ${total}`);
});

test('ipE-P2. codegen parity slic AFTER a Color Space (HSV / Lab) step (isMirrored): both views give ndiff 0 with no channel swap of the output (RGB2Lab on the H,S,V / L,a,b bytes, yellow (255, 255, 0)); HSV -> Gaussian Blur -> slic keeps that; Grayscale -> slic and Threshold -> slic run the ndim == 2 path; the same chains without Color Space stay at ndiff 0 with the BGR order', PY_SKIP, () => {
  const a = sceneRGB(97, 61, 3), b = noiseRGB(200, 120, 5);
  try {
    for (const s of [a, b]) for (const space of ['hsv', 'lab']) for (const view of ['boundary', 'mean']) {
      slicParity(s, { view, count: 60 }, `${space} -> slic ${view} ${s.cols}x${s.rows}`, { pre: [CS(space)], mirroredOut: false });
    }
    slicParity(a, { count: 40 }, 'hsv -> blur -> slic', { pre: [CS('hsv'), STEP('blur', 'gaussian_blur')], mirroredOut: false });
    slicParity(a, { count: 40 }, 'blur -> slic', { pre: [STEP('blur', 'gaussian_blur')], mirroredOut: true });
    slicParity(a, { count: 40, view: 'mean' }, 'threshold -> slic mean', { pre: [STEP('threshold', 'binary_threshold')], mirroredOut: true });
    slicParity(a, { count: 40 }, 'threshold -> slic boundary', { pre: [STEP('threshold', 'binary_threshold')], mirroredOut: true });
    slicParity(a, { count: 40 }, 'grayscale -> slic', { pre: [STEP('color_space', 'grayscale')], mirroredOut: true });
    // a cluster that goes empty (K 28 -> 38 superpixels): the numpy twin keeps its old centre exactly as the app does
    const eg = rgb(13, 6, (x, y) => { const v = EMPTY_CLUSTER_F[y * 13 + x]; return [v, v, v]; });
    try { assert.equal(slicParity(eg, { count: 28, compactness: 5, iterations: 10 }, 'empty cluster', { pre: [STEP('color_space', 'grayscale')] }).count, 38); } finally { eg.delete(); }
    // 2 slic boxes in a row (the second sees the first's colour output)
    const two = [OPSTEP('slic', { count: 40, view: 'mean' }), OPSTEP('slic', { count: 20 })];
    const py = runPython(buildPythonCode(two), a), app = appRun(two, a);
    try { assert.equal(ndiffOf(app.mat, py.bytes), 0, 'slic -> slic'); } finally { app.mat.delete(); }
  } finally { a.delete(); b.delete(); }
});

function bpParity(src, p, label, { pre = [] } = {}) {
  const pipe = [...pre, OPSTEP('bandpass_notch', p)];
  const py = runPython(buildPythonCode(pipe), src);
  const app = appRun(pipe, src);
  try {
    assert.equal(py.ndim, 2, `${label}: gray output`);
    const n = ndiffOf(app.mat, py.bytes);
    assert.equal(n, 0, `${label}: ${n} bytes differ`);
    const info = app.ctx.info;
    const head = pyLine(py.stdout, /^BANDPASS: (\w+)\/(\w+) spectrum (\d+)x(\d+)$/m);
    assert.ok(info.includes(`สเปกตรัม ${head[3]}×${head[4]}`), `${label}: P x Q in info (${info})`);
    const flag = (k) => pyLine(py.stdout, new RegExp(`^${k} = (True|False)$`, 'm'))[1] === 'True';
    assert.equal(flag('swapped'), info.includes('สลับขอบใน/ขอบนอกให้แล้ว'), `${label}: swapped`);
    assert.equal(flag('dc_warning'), info.includes('⚠️'), `${label}: dc warning`);
    assert.equal(flag('flat'), info.includes('ไม่มีรายละเอียดในช่วงความถี่นี้'), `${label}: flat`);
    assert.equal(pyLine(py.stdout, /^view = (\w+)$/m)[1] === 'spectrum', info.includes('มุมมองสเปกตรัม'), `${label}: view`);
    const sc = pyLine(py.stdout, /^scaled = (True|False) analysis (\d+)x(\d+)$/m);
    const sh = info.match(/ย่อเหลือ (\d+)×(\d+) ก่อนกรอง/);
    assert.equal(sc[1] === 'True', !!sh, `${label}: scaled`);
    if (sh) assert.deepEqual([Number(sc[2]), Number(sc[3])], [Number(sh[1]), Number(sh[2])], `${label}: analysis`);
    assert.ok(!('values' in app.ctx), 'no values');
    return { info };
  } finally { app.mat.delete(); }
}

test('ipE-P3. codegen parity bandpass_notch (real Python, 64-bit DFT): ndiff 0 for 3 kinds x 2 shapes x 2 views on 97x61, 640x480, 1280x720, 1920x1080 (one setting per kind), 2000x100 (shrink path), 33x200, 1x1, 3x5, plus a colour input and constant cases; the stdout P x Q, swapped / scaled / dc_warning / flat / view flags = the app info', PY_SKIP, () => {
  const sizes = [[97, 61], [640, 480], [33, 200], [1, 1], [3, 5], [2000, 100], [1280, 720]];
  const settings = {
    bandpass: [{ rIn: 5, rOut: 30 }, { rIn: 12, rOut: 2 }], bandreject: [{ rIn: 10, rOut: 40 }, { rIn: 0, rOut: 25 }], notch: [{ nx: 25, ny: 0, nr: 5 }, { nx: -15, ny: 33.5, nr: 2.5 }],
  };
  let n = 0;
  for (const [w, h] of sizes) {
    const src = sceneRGB(w, h, w * 3 + h);
    try {
      for (const kind of ['bandpass', 'bandreject', 'notch']) for (const shape of ['gaussian', 'ideal']) for (const view of ['result', 'spectrum']) {
        const list = w >= 1280 ? [settings[kind][0]] : settings[kind];
        if (w >= 1280 && (shape === 'ideal' && view === 'spectrum')) continue;
        for (const s of list) { bpParity(src, { kind, shape, view, ...s }, `${kind} ${shape} ${view} ${w}x${h} ${JSON.stringify(s)}`); n++; }
      }
    } finally { src.delete(); }
  }
  const odd = sceneRGB(2500, 1333, 8);
  try { bpParity(odd, { kind: 'notch', shape: 'gaussian' }, '2500x1333 (non-integer shrink to 1920x1024)'); n++; } finally { odd.delete(); }
  const big = sceneRGB(1920, 1080, 6);
  try {
    for (const [kind, shape, view, s] of [['bandpass', 'gaussian', 'result', {}], ['bandreject', 'ideal', 'result', { rIn: 20, rOut: 30 }], ['notch', 'gaussian', 'spectrum', {}], ['notch', 'ideal', 'result', { nx: 20, ny: 10, nr: 3 }]]) {
      bpParity(big, { kind, shape, view, ...s }, `1920x1080 ${kind} ${shape} ${view}`); n++;
    }
  } finally { big.delete(); }
  const flatC = rgb(60, 40, () => [100, 100, 100]);
  try {
    for (const p of [{ kind: 'notch' }, {}, { kind: 'bandreject', rIn: 0 }, { kind: 'notch', nx: 0, ny: 0 }, { kind: 'bandreject', rIn: 10, rOut: 30, view: 'spectrum' }, { kind: 'notch', nx: 7 }, { kind: 'notch', nx: 8 }]) { bpParity(flatC, p, `constant ${JSON.stringify(p)}`); n++; }
  } finally { flatC.delete(); }
  const imp = rgb(32, 32, (x, y) => (x === 5 && y === 5 ? [1, 1, 1] : [0, 0, 0]));
  try { bpParity(imp, { kind: 'bandreject', shape: 'ideal', rIn: 30, rOut: 60, view: 'spectrum' }, 'impulse spectrum (range < 1)'); n++; } finally { imp.delete(); }
  for (const rows of [10, 11]) { // the flat threshold of the band-pass (range just below / above 1)
    const st = rgb(256, 256, (x, y) => { const v = 100 + (y < rows ? (x >> 3) & 1 : 0); return [v, v, v]; });
    try { bpParity(st, { kind: 'bandpass', shape: 'ideal', rIn: 10, rOut: 15 }, `stripes in ${rows} rows`); n++; } finally { st.delete(); }
  }
  const periodic = periodicPair(640, 480);
  try {
    const rgbN = toRGB(periodic.noisy);
    try { bpParity(rgbN, { kind: 'notch', nx: 30, ny: 10, nr: 4 }, 'periodic notch'); bpParity(rgbN, { kind: 'notch', nx: 30, ny: -10, nr: 4 }, 'periodic notch wrong sign'); n += 2; } finally { rgbN.delete(); }
  } finally { periodic.clean.delete(); periodic.noisy.delete(); }
  assert.ok(n >= 100, `cases ${n}`);
});

test('ipE-P4. codegen parity bandpass_notch after a Color Space step and as a colour input (auto BGR2GRAY of the exported script = RGB2GRAY of the app), and after another gray step (Grayscale -> bandpass_notch, Blur -> bandpass_notch): ndiff 0, same info flags', PY_SKIP, () => {
  const src = sceneRGB(97, 61, 8), big = sceneRGB(320, 240, 9);
  try {
    for (const s of [src, big]) {
      bpParity(s, { kind: 'notch', view: 'spectrum' }, `colour ${s.cols}x${s.rows}`);
      bpParity(s, { kind: 'bandreject', rIn: 10, rOut: 35 }, 'colour bandreject');
      bpParity(s, { rIn: 3, rOut: 40 }, 'grayscale -> bandpass', { pre: [STEP('color_space', 'grayscale')] });
      bpParity(s, { kind: 'notch', nx: 18, ny: 9 }, 'blur -> notch', { pre: [STEP('blur', 'gaussian_blur')] });
    }
    // the output of bandpass_notch is gray: a later needs-gray step adds no conversion and a later colour step sees a 2-D image
    const pipe = [OPSTEP('bandpass_notch', {}), STEP('shape_analysis', 'regionprops')];
    assert.ok(!buildPythonCode(pipe).split('\n').filter((l) => l.includes('COLOR_BGR2GRAY')).slice(1).length, 'no second BGR2GRAY after bandpass_notch');
  } finally { src.delete(); big.delete(); }
});

test('ipE-28. timing (informational, never asserted tightly): slic 640x480 / 1080p and bandpass_notch result 640x480 / 1080p stay in a sane range so a real-time graph is usable', () => {
  const c = sceneRGB(640, 480, 2), cb = sceneRGB(1920, 1080, 2), g = sceneGray(640, 480), gb = sceneGray(1920, 1080);
  const time = (id, src, p) => {
    run(id, src, p).out.delete();
    const t0 = performance.now();
    for (let i = 0; i < 3; i++) run(id, src, p).out.delete();
    return (performance.now() - t0) / 3;
  };
  try {
    const t = { slic480: time('slic', c, {}), slic1080: time('slic', cb, {}), bp480: time('bandpass_notch', g, {}), bp1080: time('bandpass_notch', gb, {}), sp1080: time('bandpass_notch', gb, { view: 'spectrum' }) };
    console.log('# timing ms', JSON.stringify(Object.fromEntries(Object.entries(t).map(([k, v]) => [k, Math.round(v)]))));
    assert.ok(t.slic480 < 500 && t.slic1080 < 800 && t.bp480 < 400 && t.bp1080 < 1500 && t.sp1080 < 1500, JSON.stringify(t));
  } finally { for (const m of [c, cb, g, gb]) m.delete(); }
});

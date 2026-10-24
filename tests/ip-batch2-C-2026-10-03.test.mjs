// 2026-10-03 — acceptance tests ของ IP batch 2 ชิ้น C (ภาพเดียว 3 ตัว: gaussian_noise ใน FILTERS.noise, wiener_filter ใน
// FILTERS.blur, translate_shear ใน FILTERS.geometry)
// สเปก: Vision_boi/specs/ip_batch2_C_2026-10-03.md — ค่าอ้างอิง PRNG/Box-Muller ในสเปก §4.1 qa คำนวณไว้ก่อน และ coder รันยืนยันซ้ำ
// ถ้าเทสต์ล้มให้ถือว่าโค้ดผิด ไม่ใช่ตัวเลขผิด
//
// นโยบาย leak (P-14, WORK-TRACKER §1): address-fingerprint ใช้ได้เฉพาะเทสต์แรกของไฟล์ (ipC-0) พร้อมคู่แฝด
// wrapper-balance (ipC-0w) — เทสต์อื่นใช้ assertLeakFree / assertWrapperBalance
// ไม่ปักจำนวน OPS รวม (batch D–E จะเพิ่มอีก) เช็คเฉพาะว่า op ของ batch นี้มีอยู่ และ FILTERS ยังเป็น 26
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { OPS, FILTERS, defaultParams, needsOf, mulberry32, seededGaussian } from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS, OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { GROUP_ICON, OP_PRIMARY } from '../src/renderer/filters/ui-meta.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import {
  assertLeakFree, assertWrapperBalance, fingerprintLeakProbe, makeThrowingCv,
} from './_leak-helpers.mjs';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.warpAffine === 'function' && typeof c.boxFilter === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

const NEW_OPS = ['gaussian_noise', 'wiener_filter', 'translate_shear'];
const HOST = { gaussian_noise: 'noise', wiener_filter: 'blur', translate_shear: 'geometry' };

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
// fn(x, y) -> [r, g, b]
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
function noiseGray(w, h, seed) { const r = lcg(seed); return gray(w, h, () => Math.floor(r() * 256)); }
// ภาพสีที่ R, G, B ต่างกันทุกพิกเซล (ใช้จับลำดับช่องสีของ gaussian_noise ใน Python)
function noiseRGB(w, h, seed) { const r = lcg(seed); return rgb(w, h, () => [Math.floor(r() * 256), Math.floor(r() * 256), Math.floor(r() * 256)]); }
// ภาพมีโครงสร้าง (ขอบ + พื้นเรียบ + texture) ใช้กับ wiener/shear
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
function constGray(w, h, v) { return gray(w, h, () => v); }
function bytes(m) { return Array.from(m.data); }
function run(opId, src, params = {}, cvUse = cv) {
  const ctx = {};
  const out = OPS[opId].apply(cvUse, src, params, ctx);
  return { out, ctx };
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

// ===================================================================
// ipC-0 / ipC-0w — ด่าน leak หลัก (P-14 ข้อ 1 + คู่บังคับข้อ 2)
// ===================================================================
function batchCLeakPaths() {
  const g = noiseGray(24, 18, 1), c = noiseRGB(24, 18, 2), rgba = toRGBA(c);
  const one = noiseGray(1, 1, 3), oneC = noiseRGB(1, 1, 4), thin = noiseGray(3, 40, 5), odd = noiseGray(3, 5, 6);
  const flat = constGray(24, 18, 77);
  const empty = new cv.Mat();
  const paths = [];
  const add = (id, params, src, tag) => paths.push({
    label: `${id} ${tag}`,
    run: (cvUse) => { const o = OPS[id].apply(cvUse, src, params, {}); o.delete(); },
  });
  for (const id of NEW_OPS) {
    add(id, {}, g, 'gray');
    add(id, {}, c, 'colour');
    add(id, {}, rgba, 'rgba');
    add(id, {}, empty, '0x0');
    add(id, {}, one, '1x1 gray');
    add(id, {}, oneC, '1x1 colour');
  }
  add('gaussian_noise', { sigma: 0 }, c, 'sigma 0');
  add('gaussian_noise', { sigma: 100, seed: 99999 }, odd, 'odd sample count');
  add('wiener_filter', { ksize: 3 }, g, 'k3');
  add('wiener_filter', { ksize: 15 }, c, 'k15 colour');
  add('wiener_filter', { ksize: 15 }, thin, 'k15 on 3x40');
  add('wiener_filter', { noiseVar: 0 }, flat, 'flat image (den = 0)');
  add('wiener_filter', { noiseVar: 100 }, c, 'noiseVar 100');
  add('translate_shear', { tx: -30, ty: 20, shearX: 0, shearY: 0 }, c, 'translate only');
  add('translate_shear', { shearX: 0.3, shearY: -0.2 }, g, 'shear');
  add('translate_shear', { shearX: 1, shearY: 1 }, c, 'singular guard');
  add('translate_shear', { tx: 100, ty: -100 }, g, 'shifted out of frame');
  return {
    paths,
    cleanup: () => { for (const m of [g, c, rgba, one, oneC, thin, odd, flat, empty]) m.delete(); },
  };
}

test('ipC-0. leak (address fingerprint, first test of a clean process): every branch of the three new ops (gray / colour / RGBA / 0x0 / 1x1, sigma 0, an odd sample count, ksize 3 and 15, a flat image (den = 0), noiseVar 0/100, translate only / shear / singular guard / shifted out of frame) leaves the malloc fingerprint unchanged', () => {
  const { paths, cleanup } = batchCLeakPaths();
  try {
    fingerprintLeakProbe(cv, 'IP batch 2 chunk C ops (all paths)', () => {
      for (const p of paths) p.run(cv);
    }, { iters: 25, warmup: 10 });
  } finally { cleanup(); }
});

test('ipC-0w. leak (wrapper balance twin of ipC-0): the exact same path table creates and deletes every embind wrapper it makes (RGBA->RGB temporaries, the noise copy, the float32 sums / squares of wiener, the 2x3 affine matrix, the passthrough clones)', () => {
  const { paths, cleanup } = batchCLeakPaths();
  try {
    assertWrapperBalance(cv, 'IP batch 2 chunk C ops (all paths)', (c) => {
      for (const p of paths) p.run(c);
    }, { minCreated: 70 });
  } finally { cleanup(); }
});

// ===================================================================
// ipC-L1…L5 — leak ต่อ op / ต่อเส้นทาง (P-14 ข้อ 2 + 3 ในคำสั่งเดียว)
// ===================================================================
function leakEach(label, cases, opts = {}) {
  for (const [tag, fn] of cases) {
    assertLeakFree(cv, `${label} ${tag}`, (c) => { const o = fn(c); o.delete(); }, { minCreated: 1, iters: 40, warmup: 10, ...opts });
  }
}

test('ipC-L1. leak: gaussian_noise on gray / colour / RGBA (the RGBA->RGB Mat becomes the output), sigma 0, 1x1 and the 0x0 passthrough balances every wrapper and grows the heap by 0', () => {
  const g = noiseGray(20, 14, 11), c = noiseRGB(20, 14, 12), rgba = toRGBA(c), one = noiseGray(1, 1, 13), empty = new cv.Mat();
  try {
    leakEach('gaussian_noise', [
      ['gray', (cvU) => OPS.gaussian_noise.apply(cvU, g, { sigma: 30, seed: 5 }, {})],
      ['colour', (cvU) => OPS.gaussian_noise.apply(cvU, c, {}, {})],
      ['rgba', (cvU) => OPS.gaussian_noise.apply(cvU, rgba, {}, {})],
      ['sigma 0', (cvU) => OPS.gaussian_noise.apply(cvU, c, { sigma: 0 }, {})],
      ['1x1', (cvU) => OPS.gaussian_noise.apply(cvU, one, {}, {})],
      ['0x0', (cvU) => OPS.gaussian_noise.apply(cvU, empty, {}, {})],
    ]);
  } finally { for (const m of [g, c, rgba, one, empty]) m.delete(); }
});

test('ipC-L2. leak: wiener_filter with ksize 3/5/15, auto and given noise variance, a flat image (den = 0 path), RGBA, 1x1, 3x40 at k=15 and the 0x0 passthrough balances every wrapper (s1, f, f*f, s2, RGB temp, output) and grows the heap by 0', () => {
  const g = sceneGray(20, 14), c = sceneRGB(20, 14), rgba = toRGBA(c), flat = constGray(20, 14, 200);
  const one = noiseGray(1, 1, 21), thin = noiseGray(3, 40, 22), empty = new cv.Mat();
  try {
    leakEach('wiener_filter', [
      ['gray k3', (cvU) => OPS.wiener_filter.apply(cvU, g, { ksize: 3 }, {})],
      ['colour k5', (cvU) => OPS.wiener_filter.apply(cvU, c, {}, {})],
      ['colour k15 noise 100', (cvU) => OPS.wiener_filter.apply(cvU, c, { ksize: 15, noiseVar: 100 }, {})],
      ['flat', (cvU) => OPS.wiener_filter.apply(cvU, flat, {}, {})],
      ['rgba', (cvU) => OPS.wiener_filter.apply(cvU, rgba, {}, {})],
      ['1x1 k15', (cvU) => OPS.wiener_filter.apply(cvU, one, { ksize: 15 }, {})],
      ['3x40 k15', (cvU) => OPS.wiener_filter.apply(cvU, thin, { ksize: 15 }, {})],
      ['0x0', (cvU) => OPS.wiener_filter.apply(cvU, empty, {}, {})],
    ]);
  } finally { for (const m of [g, c, rgba, flat, one, thin, empty]) m.delete(); }
});

test('ipC-L3. leak: translate_shear translate-only / with shear / singular guard (returns a clone from inside the try) / shifted out of frame / RGBA / 1x1 / 0x0 balances every wrapper (RGB temp, 2x3 matrix, output) and grows the heap by 0', () => {
  const g = sceneGray(20, 14), c = sceneRGB(20, 14), rgba = toRGBA(c), one = noiseRGB(1, 1, 31), empty = new cv.Mat();
  try {
    leakEach('translate_shear', [
      ['translate', (cvU) => OPS.translate_shear.apply(cvU, g, { tx: -25, ty: 40 }, {})],
      ['shear', (cvU) => OPS.translate_shear.apply(cvU, c, { shearX: 0.4, shearY: -0.3 }, {})],
      ['singular', (cvU) => OPS.translate_shear.apply(cvU, c, { shearX: -1, shearY: -1 }, {})],
      ['singular rgba', (cvU) => OPS.translate_shear.apply(cvU, rgba, { shearX: 1, shearY: 1 }, {})],
      ['out of frame', (cvU) => OPS.translate_shear.apply(cvU, g, { tx: 100, ty: 0 }, {})],
      ['rgba', (cvU) => OPS.translate_shear.apply(cvU, rgba, {}, {})],
      ['1x1', (cvU) => OPS.translate_shear.apply(cvU, one, { shearX: 0.5 }, {})],
      ['0x0', (cvU) => OPS.translate_shear.apply(cvU, empty, {}, {})],
    ]);
  } finally { for (const m of [g, c, rgba, one, empty]) m.delete(); }
});

test('ipC-L4. leak on the exception path: when a cv call inside each op throws (RGBA->RGB cvtColor of all three, the 1st and 2nd boxFilter and the multiply of wiener, matFromArray and warpAffine of translate_shear), every wrapper allocated so far is still released and the exception propagates', () => {
  const g = sceneGray(16, 12), c = sceneRGB(16, 12), rgba = toRGBA(c);
  const throwing = [
    ['gaussian_noise RGBA cvtColor', 'cvtColor', (cvU) => OPS.gaussian_noise.apply(cvU, rgba, {}, {})],
    ['wiener RGBA cvtColor', 'cvtColor', (cvU) => OPS.wiener_filter.apply(cvU, rgba, {}, {})],
    ['wiener 1st boxFilter', ['boxFilter', 1], (cvU) => OPS.wiener_filter.apply(cvU, rgba, {}, {})],
    ['wiener 2nd boxFilter', ['boxFilter', 2], (cvU) => OPS.wiener_filter.apply(cvU, rgba, {}, {})],
    ['wiener multiply', 'multiply', (cvU) => OPS.wiener_filter.apply(cvU, c, {}, {})],
    ['translate RGBA cvtColor', 'cvtColor', (cvU) => OPS.translate_shear.apply(cvU, rgba, {}, {})],
    ['translate matFromArray', 'matFromArray', (cvU) => OPS.translate_shear.apply(cvU, rgba, { shearX: 0.2 }, {})],
    ['translate warpAffine', 'warpAffine', (cvU) => OPS.translate_shear.apply(cvU, rgba, { shearX: 0.2 }, {})],
    ['translate warpAffine gray', 'warpAffine', (cvU) => OPS.translate_shear.apply(cvU, g, {}, {})],
  ];
  try {
    for (const [label, method, fn] of throwing) {
      const [name, onCall] = Array.isArray(method) ? method : [method, null];
      assertWrapperBalance(cv, `throw in ${label}`, (cvU) => {
        const o = fn(makeThrowingCv(cvU, name, onCall ? { onCall } : {}));
        o.delete();
      }, { expectThrow: /injected test failure/, minCreated: 1 });
    }
  } finally { for (const m of [g, c, rgba]) m.delete(); }
});

test('ipC-L5. leak on the exception path AFTER the output exists: a ctx whose "info" setter throws (the last statement of every successful path) makes each op delete its own output in catch -- gaussian_noise normal + sigma 0, wiener auto + given noise, translate_shear with and without shear', () => {
  const g = sceneGray(16, 12), c = sceneRGB(16, 12), rgba = toRGBA(c);
  const throwingCtx = () => ({ set info(_) { throw new Error('injected test failure (ctx.info)'); } });
  const cases = [
    ['gaussian_noise colour', (cvU) => OPS.gaussian_noise.apply(cvU, c, {}, throwingCtx())],
    ['gaussian_noise rgba', (cvU) => OPS.gaussian_noise.apply(cvU, rgba, {}, throwingCtx())],
    ['gaussian_noise sigma 0', (cvU) => OPS.gaussian_noise.apply(cvU, g, { sigma: 0 }, throwingCtx())],
    ['wiener auto', (cvU) => OPS.wiener_filter.apply(cvU, rgba, {}, throwingCtx())],
    ['wiener given', (cvU) => OPS.wiener_filter.apply(cvU, g, { noiseVar: 50 }, throwingCtx())],
    ['translate', (cvU) => OPS.translate_shear.apply(cvU, rgba, {}, throwingCtx())],
    ['translate shear', (cvU) => OPS.translate_shear.apply(cvU, g, { shearX: 0.3 }, throwingCtx())],
  ];
  try {
    for (const [label, fn] of cases) {
      assertWrapperBalance(cv, `ctx.info throws in ${label}`, (cvU) => { const o = fn(cvU); o.delete(); },
        { expectThrow: /injected test failure \(ctx\.info\)/, minCreated: 1 });
    }
  } finally { for (const m of [g, c, rgba]) m.delete(); }
});

// ===================================================================
// ipC-1…3 — gaussian_noise
// ===================================================================
test('ipC-1. seeded PRNG (spec 4.1 reference values): mulberry32(1) -> 2693262067, 11749833, 2265367787, 4213581821 · mulberry32(42) -> 2581720956, 1925393290 · seededGaussian(1, 2) -> z0 = 0.96597405863096397, z1 = 0.016605805250894896 exactly (Box-Muller with u1 = (r+1)/2^32, cos for even, sin for odd) · an odd count drops the sin of the last pair (prefix-stable)', () => {
  let g = mulberry32(1);
  assert.deepEqual([g(), g(), g(), g()], [2693262067, 11749833, 2265367787, 4213581821]);
  g = mulberry32(42);
  assert.deepEqual([g(), g()], [2581720956, 1925393290]);
  const z2 = seededGaussian(1, 2);
  assert.ok(z2 instanceof Float64Array);
  assert.equal(z2[0], 0.96597405863096397);
  assert.equal(z2[1], 0.016605805250894896);
  // หาค่าเองจาก uint32 ที่ปักไว้ (กัน mutation ที่เปลี่ยน +1 ของ u1 หรือสลับ cos/sin)
  const u1 = (2693262067 + 1) / 4294967296, u2 = 11749833 / 4294967296;
  assert.equal(z2[0], Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
  assert.equal(z2[1], Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2));
  const z3 = seededGaussian(1, 3), z4 = seededGaussian(1, 4), z1 = seededGaussian(1, 1);
  assert.equal(z3.length, 3);
  assert.equal(z1.length, 1);
  assert.equal(z1[0], z2[0]);
  assert.deepEqual(Array.from(z3), Array.from(z4).slice(0, 3), 'odd count = the even sequence with its last value dropped');
  assert.equal(seededGaussian(7, 0).length, 0);
  // seed เดิม = ค่าเดิม, seed ต่าง = ค่าต่าง
  assert.deepEqual(Array.from(seededGaussian(42, 50)), Array.from(seededGaussian(42, 50)));
  assert.notDeepEqual(Array.from(seededGaussian(42, 50)), Array.from(seededGaussian(43, 50)));
});

test('ipC-2. gaussian_noise: every output byte is exactly clamp(floor(src + sigma*z[k] + 0.5), 0, 255) with z = seededGaussian(seed, rows*cols*channels) in memory order (row -> column -> R,G,B) -- on gray and colour, several sigma/seed; sigma 0 returns a byte-identical NEW Mat with a message; the same seed twice is identical and another seed differs', () => {
  const g = noiseGray(37, 23, 41), c = noiseRGB(29, 17, 42);
  try {
    for (const [src, sigma, seed] of [[g, 20, 1], [g, 7.5, 314], [c, 25, 42], [c, 100, 99999]]) {
      const { out, ctx } = run('gaussian_noise', src, { sigma, seed });
      try {
        assert.equal(out.channels(), src.channels());
        const z = seededGaussian(seed, src.data.length);
        const s = src.data, o = out.data;
        let clipped = 0;
        for (let k = 0; k < s.length; k++) {
          const v = Math.floor(s[k] + sigma * z[k] + 0.5);
          if (v < 0 || v > 255) clipped++;
          assert.equal(o[k], Math.min(255, Math.max(0, v)), `sigma ${sigma} seed ${seed} byte ${k}`);
        }
        const pct = (Math.floor(((clipped * 100) / s.length) * 10 + 0.5) / 10).toFixed(1);
        assert.equal(ctx.info, `เติม noise แบบเกาส์ σ=${sigma} (seed ${seed}) · ชนขอบ 0/255 ${pct}% ของค่า`);
      } finally { out.delete(); }
    }
    const z0 = run('gaussian_noise', c, { sigma: 0, seed: 5 });
    try {
      assert.notEqual(z0.out, c, 'a new Mat, never src itself');
      sameBytes(z0.out, c, 'sigma 0');
      assert.match(z0.ctx.info, /σ = 0 — ไม่เติม noise/);
    } finally { z0.out.delete(); }
    const a = run('gaussian_noise', c, { seed: 9 }), b = run('gaussian_noise', c, { seed: 9 }), d = run('gaussian_noise', c, { seed: 10 });
    try {
      sameBytes(a.out, b.out, 'same seed twice');
      assert.notDeepEqual(bytes(a.out), bytes(d.out), 'another seed must give another pattern');
    } finally { a.out.delete(); b.out.delete(); d.out.delete(); }
  } finally { g.delete(); c.delete(); }
});

test('ipC-3. gaussian_noise statistics: a flat 128 gray 640x480 with sigma 20 has mean(out-128) within +-0.3 and std within 20 +- 0.4 (no clipping at 128); an all-black image with sigma 20 keeps ~50% (+-2%) of its pixels at 0 (half the noise is negative and is clipped) and the info reports the clipped share', () => {
  const flat = constGray(640, 480, 128), black = constGray(200, 150, 0);
  try {
    const { out } = run('gaussian_noise', flat, { sigma: 20, seed: 1 });
    try {
      let s = 0, s2 = 0;
      const d = out.data;
      for (let i = 0; i < d.length; i++) { const e = d[i] - 128; s += e; s2 += e * e; }
      const mean = s / d.length, std = Math.sqrt(s2 / d.length - mean * mean);
      assert.ok(Math.abs(mean) <= 0.3, `mean ${mean}`);
      assert.ok(Math.abs(std - 20) <= 0.4, `std ${std}`);
    } finally { out.delete(); }
    const r = run('gaussian_noise', black, { sigma: 20, seed: 3 });
    try {
      let zeros = 0;
      for (const v of r.out.data) if (v === 0) zeros++;
      const frac = zeros / r.out.data.length;
      assert.ok(Math.abs(frac - 0.5) <= 0.02, `zeros ${frac}`);
      const pct = Number(r.ctx.info.match(/ชนขอบ 0\/255 ([\d.]+)%/)[1]);
      assert.ok(Math.abs(pct - 50) <= 2, `clipped share ${pct}%`);
    } finally { r.out.delete(); }
  } finally { flat.delete(); black.delete(); }
});

test('ipC-3b. gaussian_noise params are clamped like the UI (sigma numIn [0,100], seed intIn [1,99999]): sigma 999 == 100, sigma -3 == 0 (unchanged copy), seed 0 == 1, seed "7.4" == 7, seed 1e9 == 99999, garbage == defaults (sigma 20 seed 1); an RGBA input gives a 3-channel result equal to running on its RGB version; a 1x1 image works', () => {
  const c = noiseRGB(17, 11, 51), rgba = toRGBA(c), one = noiseGray(1, 1, 52);
  const same = (pa, pb, label, src = c) => {
    const a = run('gaussian_noise', src, pa), b = run('gaussian_noise', src, pb);
    try { sameBytes(a.out, b.out, label); } finally { a.out.delete(); b.out.delete(); }
  };
  try {
    same({ sigma: 999, seed: 4 }, { sigma: 100, seed: 4 }, 'sigma 999 -> 100');
    same({ sigma: -3 }, { sigma: 0 }, 'sigma -3 -> 0');
    same({ seed: 0 }, { seed: 1 }, 'seed 0 -> 1');
    same({ seed: '7.6' }, { seed: 8 }, 'seed "7.6" -> 8 (intIn rounds; a bare `seed >>> 0` would truncate to 7)');
    same({ seed: 1e9 }, { seed: 99999 }, 'seed 1e9 -> 99999');
    same({ sigma: 'abc', seed: 'x' }, { sigma: 20, seed: 1 }, 'garbage -> defaults');
    const a = run('gaussian_noise', rgba, { seed: 8 }), b = run('gaussian_noise', c, { seed: 8 });
    try {
      assert.equal(a.out.channels(), 3, 'RGBA -> RGB result');
      sameBytes(a.out, b.out, 'rgba == rgb');
    } finally { a.out.delete(); b.out.delete(); }
    const o = run('gaussian_noise', one, { sigma: 50, seed: 2 });
    try {
      assert.equal(o.out.data[0], Math.min(255, Math.max(0, Math.floor(one.data[0] + 50 * seededGaussian(2, 1)[0] + 0.5))));
    } finally { o.out.delete(); }
  } finally { c.delete(); rgba.delete(); one.delete(); }
});

// รอบแก้หลังรีวิว (2026-10-03, TEST GAP ของ debugger): ค่า % ที่ลงครึ่งพอดี — toFixed(1) ของ 3.55 (double = 3.54999…) ได้ "3.5"
// แต่ fixed1 (ปัดครึ่งขึ้นแบบ np.floor(x*10+0.5)/10 ของ Python) ได้ "3.6" · เคสเดิมทั้งหมดไม่มีค่าที่ลงครึ่งพอดี mutant fixed1 → toFixed จึงรอด
test('ipC-3c. gaussian_noise info rounds a half-way share up like the exported Python (fixed1, not toFixed): a flat 250 gray 50x40 with sigma 3 seed 2 clips exactly 71 of 2000 values = 3.55% -> "3.6%" (toFixed(1) would print "3.5")', () => {
  const g = constGray(50, 40, 250);
  try {
    const { out, ctx } = run('gaussian_noise', g, { sigma: 3, seed: 2 });
    try {
      const z = seededGaussian(2, 2000);
      let clipped = 0;
      for (let k = 0; k < 2000; k++) { const v = Math.floor(250 + 3 * z[k] + 0.5); if (v > 255 || v < 0) clipped++; }
      assert.equal(clipped, 71, 'fixture: 71 clipped values');
      assert.equal(((clipped * 100) / 2000).toFixed(1), '3.5', 'fixture: toFixed would round this tie down');
      assert.equal(ctx.info, 'เติม noise แบบเกาส์ σ=3 (seed 2) · ชนขอบ 0/255 3.6% ของค่า');
    } finally { out.delete(); }
  } finally { g.delete(); }
});

// ===================================================================
// ipC-4… — wiener_filter
// ===================================================================
// ตัวอ้างอิงแบบเขียนเองทั้งหมด (ไม่ใช้ OpenCV): ผลรวมในหน้าต่างด้วยการสะท้อนแบบ BORDER_REFLECT_101 (gfedcb|abcdefgh|gfedcba)
// เป็นจำนวนเต็มแม่นเป๊ะ แล้วตามสูตรในสเปก §4.2 ทีละขั้น — op ต้องได้ทุก byte เท่ากัน
function reflect101(i, n) {
  if (n === 1) return 0;
  for (;;) {
    if (i < 0) i = -i;
    else if (i >= n) i = 2 * n - 2 - i;
    else return i;
  }
}
function windowSums(mat, k) {
  const w = mat.cols, h = mat.rows, ch = mat.channels(), d = mat.data, r = (k - 1) / 2;
  const S1 = new Float64Array(w * h * ch), S2 = new Float64Array(w * h * ch);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < ch; c++) {
    let a = 0, b = 0;
    for (let dy = -r; dy <= r; dy++) {
      const yy = reflect101(y + dy, h);
      for (let dx = -r; dx <= r; dx++) {
        const v = d[(yy * w + reflect101(x + dx, w)) * ch + c];
        a += v; b += v * v;
      }
    }
    S1[(y * w + x) * ch + c] = a; S2[(y * w + x) * ch + c] = b;
  }
  return { S1, S2 };
}
function wienerRef(mat, k, noiseVar) {
  const { S1, S2 } = windowSums(mat, k);
  const ch = mat.channels(), total = mat.rows * mat.cols, n = k * k, x = mat.data;
  const noise = new Float64Array(ch);
  if (noiseVar > 0) noise.fill(noiseVar);
  else {
    for (let i = 0; i < S1.length; i++) { const m = S1[i] / n; noise[i % ch] += S2[i] / n - m * m; }
    for (let c = 0; c < ch; c++) noise[c] /= total;
  }
  const out = new Uint8Array(S1.length);
  for (let i = 0; i < S1.length; i++) {
    const m = S1[i] / n, v = S2[i] / n - m * m, nz = noise[i % ch];
    const den = Math.max(v, nz), g = Math.max(v - nz, 0);
    const o = den > 0 ? m + ((x[i] - m) / den) * g : m;
    out[i] = Math.min(255, Math.max(0, Math.floor(o + 0.5)));
  }
  return { out, noise };
}

test('ipC-4. wiener_filter equals an independent brute-force wiener2 (integer window sums with BORDER_REFLECT_101, local mean/variance in double, noise = in-order mean of the local variance per channel, den = max(v, noise), gain = max(v - noise, 0), out = m + (x - m)/den*gain, floor(+0.5)) byte for byte -- gray and colour, k 3/5/15, noise auto/100/2500, plus 1x1 and 3x40 at k 15; the info shows the estimate (R/G/B for colour)', () => {
  const imgs = [sceneGray(31, 23, 61), sceneRGB(29, 19, 62), noiseGray(13, 9, 63), noiseRGB(1, 1, 64), noiseGray(3, 40, 65), noiseRGB(40, 3, 66)];
  try {
    for (const src of imgs) {
      for (const [ksize, noiseVar] of [[3, 0], [5, 0], [15, 0], [5, 100], [7, 2500]]) {
        const { out, ctx } = run('wiener_filter', src, { ksize, noiseVar });
        try {
          const ref = wienerRef(src, ksize, noiseVar);
          const label = `${src.cols}x${src.rows}x${src.channels()} k${ksize} noise ${noiseVar}`;
          assert.equal(out.channels(), src.channels(), label);
          assert.deepEqual(Array.from(out.data), Array.from(ref.out), label);
          const fx = (v) => (Math.floor(v * 10 + 0.5) / 10).toFixed(1);
          const expected = noiseVar > 0
            ? `Wiener ${ksize}×${ksize} · noise variance = ${noiseVar} (กำหนดเอง)`
            : `Wiener ${ksize}×${ksize} · noise variance ${src.channels() === 1 ? '' : 'R/G/B '}= ${Array.from(ref.noise, fx).join('/')} (ประมาณจากภาพ)`;
          assert.equal(ctx.info, expected, label);
        } finally { out.delete(); }
      }
    }
  } finally { for (const m of imgs) m.delete(); }
});

test('ipC-4b. wiener_filter on a constant image (gray 77 and colour (10,200,255)) returns it unchanged through the den = 0 path (local variance 0 = noise estimate 0); an all-white image at k = 15 is unchanged, and its float32 window sum of squares is exactly 14,630,625 (= 15^2 * 255^2 < 2^24) at every pixel -- while k = 17 would need 18,792,225 which float32 cannot hold (that is why the UI max is 15)', () => {
  const g = constGray(23, 17, 77), c = rgb(19, 13, () => [10, 200, 255]), white = constGray(40, 30, 255);
  try {
    for (const src of [g, c]) {
      const { out, ctx } = run('wiener_filter', src, {});
      try {
        sameBytes(out, src, `constant ${src.channels()}ch`);
        assert.match(ctx.info, /= 0\.0(\/0\.0\/0\.0)? \(ประมาณจากภาพ\)/);
      } finally { out.delete(); }
    }
    const w = run('wiener_filter', white, { ksize: 15 });
    try { sameBytes(w.out, white, 'white k15'); } finally { w.out.delete(); }
    const f = new cv.Mat(), sq = new cv.Mat(), s2 = new cv.Mat();
    try {
      white.convertTo(f, cv.CV_32F);
      cv.multiply(f, f, sq);
      cv.boxFilter(sq, s2, -1, new cv.Size(15, 15), new cv.Point(-1, -1), false, cv.BORDER_REFLECT_101);
      assert.ok(Array.from(s2.data32F).every((v) => v === 14630625), 'k=15 sum of squares exact in float32');
      assert.notEqual(Math.fround(17 * 17 * 255 * 255), 17 * 17 * 255 * 255, 'k=17 would not be exact in float32');
    } finally { f.delete(); sq.delete(); s2.delete(); }
    assert.equal(OPS.wiener_filter.params.find((p) => p.key === 'ksize').max, 15);
  } finally { g.delete(); c.delete(); white.delete(); }
});

test('ipC-4c. wiener_filter with noiseVar 10000 (above every local variance of an image whose values stay in 50..200, max variance 75^2) gives gain 0 everywhere -> every pixel is exactly floor(s1/n + 0.5) of the reflect-101 window sum (the plain local mean)', () => {
  const r = lcg(71);
  const g = gray(27, 19, () => 50 + Math.floor(r() * 151));
  const c = rgb(21, 15, () => [50 + Math.floor(r() * 151), 50 + Math.floor(r() * 151), 50 + Math.floor(r() * 151)]);
  try {
    for (const src of [g, c]) for (const k of [5, 9]) {
      const { out } = run('wiener_filter', src, { ksize: k, noiseVar: 10000 });
      try {
        const { S1 } = windowSums(src, k);
        assert.deepEqual(Array.from(out.data), Array.from(S1, (s) => Math.floor(s / (k * k) + 0.5)), `${src.channels()}ch k${k}`);
      } finally { out.delete(); }
    }
  } finally { g.delete(); c.delete(); }
});

test('ipC-4d. wiener_filter keeps a step edge sharper than a Gaussian Blur of the same size: on a 60|190 step with sigma-12 Gaussian noise, the jump between the two columns at the edge stays clearly larger after Wiener 5x5 than after GaussianBlur 5x5, while the flat areas are still smoothed (row-wise std inside the flat parts drops)', () => {
  const step = gray(80, 60, (x) => (x < 40 ? 60 : 190));
  const noisy = OPS.gaussian_noise.apply(cv, step, { sigma: 12, seed: 5 }, {});
  const wie = OPS.wiener_filter.apply(cv, noisy, { ksize: 5 }, {});
  const gb = new cv.Mat();
  cv.GaussianBlur(noisy, gb, new cv.Size(5, 5), 0);
  const colMean = (m, x) => { let s = 0; for (let y = 0; y < m.rows; y++) s += m.data[y * m.cols + x]; return s / m.rows; };
  const flatStd = (m) => { let s = 0, s2 = 0, n = 0; for (let y = 0; y < m.rows; y++) for (let x = 5; x < 30; x++) { const v = m.data[y * m.cols + x]; s += v; s2 += v * v; n++; } return Math.sqrt(s2 / n - (s / n) ** 2); };
  try {
    const jumpW = colMean(wie, 40) - colMean(wie, 39);
    const jumpG = colMean(gb, 40) - colMean(gb, 39);
    assert.ok(jumpW > jumpG + 20, `edge jump wiener ${jumpW.toFixed(1)} vs gaussian ${jumpG.toFixed(1)}`);
    assert.ok(flatStd(wie) < flatStd(noisy) * 0.8, `flat std ${flatStd(wie).toFixed(2)} vs noisy ${flatStd(noisy).toFixed(2)}`);
  } finally { step.delete(); noisy.delete(); wie.delete(); gb.delete(); }
});

test('ipC-4e. wiener_filter params are clamped like the UI and ksize is forced odd (ensureOdd(intIn(ksize, 5, 3, 15)), numIn(noiseVar, 0, 0, 10000)): 99 == 15, 1 == 3, 4 == 5, 14 == 15, "6.6" == 7, noiseVar -5 == auto, 1e9 == 10000, garbage == defaults; an RGBA input gives a 3-channel result equal to its RGB version', () => {
  const c = sceneRGB(23, 17, 81), rgba = toRGBA(c);
  const same = (pa, pb, label) => {
    const a = run('wiener_filter', c, pa), b = run('wiener_filter', c, pb);
    try { sameBytes(a.out, b.out, label); assert.equal(a.ctx.info, b.ctx.info, `${label} info`); } finally { a.out.delete(); b.out.delete(); }
  };
  try {
    same({ ksize: 99 }, { ksize: 15 }, 'ksize 99 -> 15');
    same({ ksize: 1 }, { ksize: 3 }, 'ksize 1 -> 3');
    same({ ksize: 4 }, { ksize: 5 }, 'ksize 4 -> 5');
    same({ ksize: 14 }, { ksize: 15 }, 'ksize 14 -> 15');
    same({ ksize: '6.6' }, { ksize: 7 }, 'ksize "6.6" -> 7');
    same({ noiseVar: -5 }, { noiseVar: 0 }, 'noiseVar -5 -> auto');
    same({ noiseVar: 1e9 }, { noiseVar: 10000 }, 'noiseVar 1e9 -> 10000');
    same({ ksize: 'x', noiseVar: undefined }, { ksize: 5, noiseVar: 0 }, 'garbage -> defaults');
    const a = run('wiener_filter', rgba, {}), b = run('wiener_filter', c, {});
    try { assert.equal(a.out.channels(), 3); sameBytes(a.out, b.out, 'rgba == rgb'); } finally { a.out.delete(); b.out.delete(); }
  } finally { c.delete(); rgba.delete(); }
});

// ===================================================================
// ipC-5… — translate_shear
// ===================================================================
function shiftRef(mat, dx, dy) {
  const w = mat.cols, h = mat.rows, ch = mat.channels(), s = mat.data, out = new Uint8Array(s.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const sx = x - dx, sy = y - dy;
    if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
    for (let c = 0; c < ch; c++) out[(y * w + x) * ch + c] = s[(sy * w + sx) * ch + c];
  }
  return out;
}
const pxOf = (pct, size) => Math.floor((pct / 100) * size + 0.5);

test('ipC-5. translate_shear without shear equals shifting the array by hand (empty area = 0) byte for byte -- gray and colour, positive / negative / mixed shifts, the .5 rounding cases (w = 97: tx 50 -> 48.5 -> 49, tx -50 -> -48.5 -> -48), +-100% (everything leaves the frame, all black, info says so) and 1x1 / 3x40; the info reports the pixel shift', () => {
  const g = noiseGray(97, 61, 91), c = noiseRGB(97, 61, 92), one = noiseRGB(1, 1, 93), thin = noiseGray(3, 40, 94);
  try {
    const cases = [[g, 10, 5], [g, -10, -5], [c, 50, -50], [c, -50, 50], [g, 33, 0], [c, 0, -17], [g, 100, 0], [c, 0, -100], [g, -100, 100], [one, 60, 60], [thin, 34, -13]];
    for (const [src, tx, ty] of cases) {
      const { out, ctx } = run('translate_shear', src, { tx, ty, shearX: 0, shearY: 0 });
      try {
        const dx = pxOf(tx, src.cols), dy = pxOf(ty, src.rows);
        const label = `${src.cols}x${src.rows}x${src.channels()} tx ${tx} ty ${ty} -> (${dx}, ${dy})`;
        assert.deepEqual(Array.from(out.data), Array.from(shiftRef(src, dx, dy)), label);
        const s = (v) => (v >= 0 ? '+' : '') + v;
        const gone = Math.abs(dx) >= src.cols || Math.abs(dy) >= src.rows;
        assert.equal(ctx.info, `เลื่อน (${s(dx)}, ${s(dy)}) px · shear x=0.00 y=0.00${gone ? ' (ภาพเลื่อนออกนอกกรอบหมด)' : ''}`, label);
        if (gone) assert.ok(out.data.every((v) => v === 0), `${label}: all black`);
      } finally { out.delete(); }
    }
    assert.equal(pxOf(50, 97), 49);
    assert.equal(pxOf(-50, 97), -48);
    const d = run('translate_shear', g, {}); // ค่าเริ่มต้น 10%, 5% เลื่อนให้เห็นผลทันที
    try { assert.match(d.ctx.info, /^เลื่อน \(\+10, \+3\) px/); } finally { d.out.delete(); }
  } finally { g.delete(); c.delete(); one.delete(); thin.delete(); }
});

// จุดขาวจุดเดียวที่ (x0, y0) บนภาพดำ 41x31 (cx = 20, cy = 15) — ค่าที่เลือกทำให้ x' / y' ลงจำนวนเต็มพอดี จุดนั้นต้องได้ 255 และเป็นจุดสว่างสุดจุดเดียว
const POINT_W = 41, POINT_H = 31;
const POINT_CASES = [
  // [x0, y0, tx%, ty%, kx, ky]
  [10, 25, 0, 0, 0.5, 0],
  [30, 5, 0, 0, 0, 0.5],
  [30, 25, 10, -10, 0.5, -0.5],
  [8, 3, -12, 20, -0.25, 0.75],
  [20, 15, 30, 30, 0.9, 0.5],
];
function pointImage(x0, y0) { return gray(POINT_W, POINT_H, (x, y) => (x === x0 && y === y0 ? 255 : 0)); }
function expectedPoint(x0, y0, tx, ty, kx, ky) {
  const cx = (POINT_W - 1) / 2, cy = (POINT_H - 1) / 2;
  return [x0 + kx * (y0 - cy) + pxOf(tx, POINT_W), y0 + ky * (x0 - cx) + pxOf(ty, POINT_H)];
}
function argmax(bytesArr, w) {
  let best = -1, at = -1, count = 0;
  for (let i = 0; i < bytesArr.length; i++) {
    if (bytesArr[i] > best) { best = bytesArr[i]; at = i; count = 1; } else if (bytesArr[i] === best) count++;
  }
  return { x: at % w, y: Math.floor(at / w), value: best, count };
}

test('ipC-5b. translate_shear maps a single white point (x0, y0) to x\' = x0 + kx*(y0 - cy) + txp, y\' = y0 + ky*(x0 - cx) + typ with cx = (w-1)/2, cy = (h-1)/2 (shear about the image centre) -- 5 cases incl. both shears at once and negative values, the point keeps value 255 at exactly that pixel and nowhere else (pins the sign of -kx*cy, the centre, and which shear goes with which axis)', () => {
  for (const [x0, y0, tx, ty, kx, ky] of POINT_CASES) {
    const src = pointImage(x0, y0);
    const { out, ctx } = run('translate_shear', src, { tx, ty, shearX: kx, shearY: ky });
    try {
      const [ex, ey] = expectedPoint(x0, y0, tx, ty, kx, ky);
      assert.ok(Number.isInteger(ex) && Number.isInteger(ey) && ex >= 0 && ey >= 0 && ex < POINT_W && ey < POINT_H, 'fixture lands on a pixel');
      const m = argmax(out.data, POINT_W);
      assert.deepEqual([m.x, m.y, m.value, m.count], [ex, ey, 255, 1], `point (${x0},${y0}) tx ${tx} ty ${ty} kx ${kx} ky ${ky}`);
      assert.match(ctx.info, new RegExp(`shear x=${kx.toFixed(2)} y=${ky.toFixed(2)}`));
    } finally { out.delete(); src.delete(); }
  }
});

test('ipC-5c. translate_shear singular guard: |1 - kx*ky| < 0.05 returns an unchanged copy + a message to lower the shear (kx = ky = 1, kx = ky = -1, kx 0.96 * ky 1 = 0.04) while 1 - 0.95*1 = 0.050000000000000044 (not below 0.05) still warps; shear values are clamped to [-1, 1] and tx/ty to [-100, 100]; an RGBA input gives a 3-channel result (also on the guard path); garbage -> defaults (10%, 5%, no shear)', () => {
  const c = sceneRGB(33, 21, 101), rgba = toRGBA(c);
  try {
    for (const [kx, ky] of [[1, 1], [-1, -1], [0.96, 1], [1, 0.96]]) {
      const { out, ctx } = run('translate_shear', c, { shearX: kx, shearY: ky });
      try {
        sameBytes(out, c, `guard ${kx},${ky}`);
        assert.notEqual(out, c);
        assert.match(ctx.info, /^ไม่เลื่อน\/เอียง: shear x × shear y ใกล้ 1 เกินไป .* ลด shear x หรือ shear y ลง$/, `guard ${kx},${ky}`);
        // รอบแก้หลังรีวิว (NIT-1): ข้อความแสดง kx/ky ด้วยกติกาปัดเดียวกับ info ปกติ (fixed2) และไม่มีตัวเลข 1 − kx·ky
        const f2 = (v) => (Math.floor(v * 100 + 0.5) / 100).toFixed(2);
        assert.equal(ctx.info, `ไม่เลื่อน/เอียง: shear x × shear y ใกล้ 1 เกินไป (shear x=${f2(kx)} y=${f2(ky)}) ภาพจะยุบเป็นเส้น — ลด shear x หรือ shear y ลง`, `guard ${kx},${ky}`);
      } finally { out.delete(); }
    }
    const ok = run('translate_shear', c, { tx: 0, ty: 0, shearX: 0.95, shearY: 1 });
    try { assert.match(ok.ctx.info, /^เลื่อน \(\+0, \+0\) px · shear x=0\.95 y=1\.00$/); assert.notDeepEqual(bytes(ok.out), bytes(c)); } finally { ok.out.delete(); }
    const same = (pa, pb, label, src = c) => {
      const a = run('translate_shear', src, pa), b = run('translate_shear', src, pb);
      try { sameBytes(a.out, b.out, label); assert.equal(a.ctx.info, b.ctx.info, `${label} info`); } finally { a.out.delete(); b.out.delete(); }
    };
    same({ shearX: 5, shearY: -7 }, { shearX: 1, shearY: -1 }, 'shear clamp');
    same({ tx: 500, ty: -500 }, { tx: 100, ty: -100 }, 'tx/ty clamp');
    same({ tx: 'a', ty: 'b', shearX: 'c', shearY: undefined }, { tx: 10, ty: 5, shearX: 0, shearY: 0 }, 'garbage -> defaults');
    for (const p of [{}, { shearX: 1, shearY: 1 }]) {
      const a = run('translate_shear', rgba, p), b = run('translate_shear', c, p);
      try { assert.equal(a.out.channels(), 3); sameBytes(a.out, b.out, `rgba ${JSON.stringify(p)}`); } finally { a.out.delete(); b.out.delete(); }
    }
  } finally { c.delete(); rgba.delete(); }
});

// รอบแก้หลังรีวิว (MINOR-1 ของ debugger): "(ภาพเลื่อนออกนอกกรอบหมด)" จริงเฉพาะตอนไม่มี shear — tx 100 + shear x −1 บนภาพ 640×480
// ยังเห็นภาพราว 9% (แถวล่างถูกเอียงกลับเข้ากรอบ) ข้อความเดิมจึงบอกผิด
test('ipC-5d. translate_shear says "everything left the frame" only when it is true: {tx 100, shear x -1} on 640x480 still shows part of the image (non-zero output, no "หมด" in the info) and so do other shifts sheared on the same axis; a horizontal shift >= w with shear x = 0 (any shear y) or a vertical shift >= h with shear y = 0 (any shear x) keeps the suffix and is all black; the tx hint says the +-100 rule holds when shear x = 0', () => {
  const src = constGray(640, 480, 200), c = sceneRGB(97, 61, 131);
  try {
    for (const [s, p] of [[src, { tx: 100, shearX: -1 }], [src, { tx: -100, ty: 0, shearX: 0.5 }], [c, { tx: 0, ty: 100, shearY: 0.6 }]]) {
      const { out, ctx } = run('translate_shear', s, p);
      try {
        let lit = 0;
        for (const v of out.data) if (v) lit++;
        assert.ok(lit > 0, `${JSON.stringify(p)}: part of the image is still visible`);
        assert.ok(!ctx.info.includes('หมด'), `${JSON.stringify(p)}: info must not claim the image is gone: "${ctx.info}"`);
      } finally { out.delete(); }
    }
    const share = run('translate_shear', src, { tx: 100, shearX: -1 });
    try {
      let lit = 0;
      for (const v of share.out.data) if (v) lit++;
      assert.ok(lit / share.out.data.length > 0.05, `debugger case: ${(100 * lit / share.out.data.length).toFixed(1)}% visible`);
    } finally { share.out.delete(); }
    // กฎแบบตรง (debugger delta): เลื่อนแนวนอน ≥ w โดย shear x = 0 หายหมดไม่ว่า shear y เท่าไร (x' = x0 + txp) — แนวตั้งเช่นเดียวกัน
    for (const p of [{ tx: 100, ty: 0, shearX: 0, shearY: 0 }, { tx: 0, ty: -100, shearX: 0, shearY: 0 }, { tx: 100, ty: 0, shearX: 0, shearY: 0.6 },
      { tx: -100, ty: 0, shearX: 0, shearY: -1 }, { tx: 0, ty: 100, shearX: 1, shearY: 0 }, { tx: 0, ty: -100, shearX: -0.5, shearY: 0 }]) {
      const { out, ctx } = run('translate_shear', src, p);
      try {
        assert.ok(ctx.info.endsWith(' (ภาพเลื่อนออกนอกกรอบหมด)'), JSON.stringify(p));
        assert.ok(out.data.every((v) => v === 0), `${JSON.stringify(p)}: all black`);
      } finally { out.delete(); }
    }
    assert.match(OPS.translate_shear.params.find((q) => q.key === 'tx').desc, /±100 = เลื่อนออกนอกกรอบทั้งภาพ \(เมื่อ shear แนวนอน = 0/);
  } finally { src.delete(); c.delete(); }
});

// รอบแก้หลังรีวิว (NIT-1): ตัวเลข shear ใน info ใช้กติกาปัดเดียวกับสคริปต์ Python (fixed2 = np.floor(x*100+0.5)/100 แล้ว :.2f)
// toFixed(2) ของ −0.125 ได้ "-0.13" ส่วน Python :.2f ดิบของ 0.125 ได้ "0.12" — สองฝั่งเคยปัดค่าที่ลงครึ่งพอดีต่างกัน
test('ipC-5e. translate_shear shear numbers use one half-up rounding rule (fixed2): 0.125 -> "0.13", -0.125 -> "-0.12" (toFixed would give "-0.13"), 0.375 -> "0.38", 0.15000000000000002 -> "0.15", 1e-7 -> "0.00", -1e-7 -> "0.00" (never "-0.00")', () => {
  const g = sceneGray(31, 23, 141);
  try {
    for (const [kx, ky, ex, ey] of [[0.125, 0, '0.13', '0.00'], [-0.125, 0.375, '-0.12', '0.38'], [0.15000000000000002, 1e-7, '0.15', '0.00'], [0, -1e-7, '0.00', '0.00']]) {
      const { out, ctx } = run('translate_shear', g, { tx: 0, ty: 0, shearX: kx, shearY: ky });
      try { assert.equal(ctx.info, `เลื่อน (+0, +0) px · shear x=${ex} y=${ey}`, `${kx}, ${ky}`); } finally { out.delete(); }
    }
  } finally { g.delete(); }
});

// ===================================================================
// ipC-6… — โครงสร้าง (static) / ui-meta / palette / codegen string
// ===================================================================
test('ipC-6. (static structure) the 3 ops exist in OPS (the OPS total is NOT pinned), needs "any", FILTERS is still 26, noise/blur/geometry keep their original modes + defaults and append the new op last, defaultParams() matches the OPS defaults, and the merged nodes dispatch to the new ops', () => {
  for (const id of NEW_OPS) {
    assert.ok(OPS[id], `OPS.${id} must exist`);
    assert.equal(OPS[id].needs, 'any', `${id}.needs`);
    for (const p of OPS[id].params) assert.notEqual(p.key, 'mode', `${id}: param key "mode" would collide with MODE_KEY`);
  }
  assert.equal(Object.keys(FILTERS).length, 35, 'FILTERS = 26 + 7 category-level groups (2026-10-07) + morph_tools (2026-10-18) + image_registration (2026-10-24)');
  const modes = (f) => FILTERS[f].params.find((p) => p.key === 'mode').options.map((o) => o.value);
  assert.deepEqual(modes('noise'), ['salt_pepper_noise', 'gaussian_noise']);
  assert.deepEqual(modes('blur'), ['median_blur', 'gaussian_blur', 'box_blur', 'bilateral', 'wiener_filter']);
  // ชิ้น D (2026-10-03) ต่อท้าย roi_polygon หลัง translate_shear → ปักแค่ 7 ตัวแรก (prefix) แบบเดียวกับ ipA-13 ของชิ้น B
  // 2026-10-07: dropdown ของ geometry เรียงตาม palette (roi_polygon ต่อจาก crop) → เช็คลำดับสัมพัทธ์ของ 7 ตัวเดิมโดยไม่นับ roi_polygon
  assert.deepEqual(modes('geometry').filter((m) => m !== 'roi_polygon').slice(0, 7), ['crop', 'resize', 'rotate', 'flip', 'perspective_warp', 'undistort', 'translate_shear']);
  for (const legacy of ['restoration', 'enhancement', 'segmentation']) for (const id of NEW_OPS) assert.ok(!modes(legacy).includes(id), `${id} must not enter legacy ${legacy}`);
  assert.equal(defaultParams('noise').mode, 'salt_pepper_noise');
  assert.equal(defaultParams('blur').mode, 'gaussian_blur');
  assert.equal(defaultParams('geometry').mode, 'resize');
  const expectDefaults = {
    noise: { gaussian_noise__sigma: 20, gaussian_noise__seed: 1 },
    blur: { wiener_filter__ksize: 5, wiener_filter__noiseVar: 0 },
    geometry: { translate_shear__tx: 10, translate_shear__ty: 5, translate_shear__shearX: 0, translate_shear__shearY: 0 },
  };
  for (const [f, exp] of Object.entries(expectDefaults)) {
    const dp = defaultParams(f);
    for (const [k, v] of Object.entries(exp)) assert.equal(dp[k], v, `${f}.${k}`);
  }
  const src = noiseRGB(12, 9, 111);
  try {
    for (const id of NEW_OPS) {
      const p = { ...defaultParams(HOST[id]), mode: id };
      const viaNode = FILTERS[HOST[id]].apply(cv, src, p, {});
      const direct = OPS[id].apply(cv, src, {}, {});
      try { sameBytes(viaNode, direct, `${id} through FILTERS.${HOST[id]}`); } finally { viaNode.delete(); direct.delete(); }
    }
  } finally { src.delete(); }
});

// [opId, key, helper, default, min, max] — clamp ใน registry.js และ codegen.js ต้องเป็นตัวเลขเดียวกับ UI
const CLAMPS = [
  ['gaussian_noise', 'sigma', 'numIn'],
  ['gaussian_noise', 'seed', 'intIn'],
  ['wiener_filter', 'ksize', 'intIn'],
  ['wiener_filter', 'noiseVar', 'numIn'],
  ['translate_shear', 'tx', 'numIn'],
  ['translate_shear', 'ty', 'numIn'],
  ['translate_shear', 'shearX', 'numIn'],
  ['translate_shear', 'shearY', 'numIn'],
];

test('ipC-7. (static read) every numeric param of the 3 ops is clamped in registry.js (numIn/intIn) AND in codegen.js (clampNum) to EXACTLY the UI default/min/max; wiener ksize is ensureOdd(intIn(...)) / odd(clampNum(...)); the singular-guard threshold 0.05 is the same on both sides; every numeric param is in the table', () => {
  const reg = fs.readFileSync(new URL('../src/renderer/filters/registry.js', import.meta.url), 'utf8');
  const gen = fs.readFileSync(new URL('../src/renderer/codegen.js', import.meta.url), 'utf8');
  // ค้นเฉพาะในก้อนของ op นั้นเอง (ตั้งแต่ "  opId: {" ถึง key ระดับบนสุดตัวถัดไป) — log_edge ก็มี numIn(p.sigma, …) / clampNum(p.sigma, …)
  const block = (source, opId) => {
    const start = source.indexOf(`\n  ${opId}: {`);
    assert.ok(start >= 0, `no "${opId}: {" block`);
    const next = /\n {2}[a-z_0-9]+: \{/g;
    next.lastIndex = start + 1;
    const m = next.exec(source);
    return source.slice(start, m ? m.index : source.length);
  };
  const num = '(-?[\\d.]+)';
  for (const [opId, key, fn] of CLAMPS) {
    const spec = OPS[opId].params.find((p) => p.key === key);
    assert.ok(spec, `${opId}.${key}: no such param`);
    const want = [spec.default, spec.min, spec.max];
    const m = block(reg, opId).match(new RegExp(`${fn}\\(p\\.${key}, ${num}, ${num}, ${num}\\)`));
    assert.ok(m, `registry.js: expected ${fn}(p.${key}, default, min, max)`);
    assert.deepEqual([Number(m[1]), Number(m[2]), Number(m[3])], want, `registry.js ${opId}.${key}`);
    const g = block(gen, opId).match(new RegExp(`clampNum\\(p\\.${key}, ${num}, ${num}, ${num}\\)`));
    assert.ok(g, `codegen.js: expected clampNum(p.${key}, default, min, max)`);
    assert.deepEqual([Number(g[1]), Number(g[2]), Number(g[3])], want, `codegen.js ${opId}.${key}`);
  }
  assert.match(reg, /ensureOdd\(intIn\(p\.ksize, 5, 3, 15\)\)/);
  assert.match(gen, /odd\(clampNum\(p\.ksize, 5, 3, 15\)\)/);
  assert.match(gen, /Math\.round\(clampNum\(p\.seed, 1, 1, 99999\)\)/, 'seed is an integer on both sides (intIn rounds)');
  assert.match(reg, /const SHEAR_SINGULAR_LIMIT = 0\.05;/);
  assert.match(gen, /if abs\(1 - kx \* ky\) < 0\.05:/);
  for (const id of NEW_OPS) for (const p of OPS[id].params) {
    if (p.type !== 'number') continue;
    assert.ok(CLAMPS.some(([o, k]) => o === id && k === p.key), `${id}.${p.key}: a numeric param with no clamp test`);
  }
});

test('ipC-8. ui-meta.js OP_PRIMARY: card text uses the clamped values the op really uses (pnum) -- "σ 20 · seed 1", "Wiener 5×5 · noise auto" / "noise 100" (ksize forced odd like ensureOdd: 14 -> 15), "เลื่อน 10%, 5%" / "เลื่อน 10%, 5% · shear 0.2, 0"', () => {
  assert.equal(OP_PRIMARY.gaussian_noise({ sigma: 20, seed: 1 }), 'σ 20 · seed 1');
  assert.equal(OP_PRIMARY.gaussian_noise({}), 'σ 20 · seed 1');
  assert.equal(OP_PRIMARY.gaussian_noise({ sigma: 999, seed: '7.4' }), 'σ 100 · seed 7');
  assert.equal(OP_PRIMARY.wiener_filter({ ksize: 5, noiseVar: 0 }), 'Wiener 5×5 · noise auto');
  assert.equal(OP_PRIMARY.wiener_filter({ ksize: 5, noiseVar: 100 }), 'Wiener 5×5 · noise 100');
  assert.equal(OP_PRIMARY.wiener_filter({ ksize: 14 }), 'Wiener 15×15 · noise auto');
  assert.equal(OP_PRIMARY.wiener_filter({ ksize: 99, noiseVar: -3 }), 'Wiener 15×15 · noise auto');
  assert.equal(OP_PRIMARY.wiener_filter({}), 'Wiener 5×5 · noise auto');
  assert.equal(OP_PRIMARY.translate_shear({ tx: 10, ty: 5, shearX: 0, shearY: 0 }), 'เลื่อน 10%, 5%');
  assert.equal(OP_PRIMARY.translate_shear({}), 'เลื่อน 10%, 5%');
  assert.equal(OP_PRIMARY.translate_shear({ tx: 10, ty: 5, shearX: 0.2, shearY: 0 }), 'เลื่อน 10%, 5% · shear 0.2, 0');
  assert.equal(OP_PRIMARY.translate_shear({ tx: 999, ty: -999, shearX: 3, shearY: -3 }), 'เลื่อน 100%, -100% · shear 1, -1');
  // รอบแก้หลังรีวิว (NIT-1): shear ปัด 2 ตำแหน่งด้วยกติกาเดียวกับ info (fixed2) แล้วตัดศูนย์ท้าย — ค่าที่สะสมจาก slider step 0.05
  // ไม่โชว์เศษ float และค่าที่ปัดแล้วเป็น 0 ทั้งคู่ถือว่าไม่เอียง (info ของแอปก็แสดง x=0.00 y=0.00)
  assert.equal(OP_PRIMARY.translate_shear({ shearX: 0.15000000000000002, shearY: 0 }), 'เลื่อน 10%, 5% · shear 0.15, 0');
  assert.equal(OP_PRIMARY.translate_shear({ shearX: 0.1 + 0.2, shearY: -0.05 }), 'เลื่อน 10%, 5% · shear 0.3, -0.05');
  assert.equal(OP_PRIMARY.translate_shear({ shearX: 1e-7, shearY: -1e-7 }), 'เลื่อน 10%, 5%');
  assert.equal(OP_PRIMARY.translate_shear({ shearX: 0.125, shearY: -0.125 }), 'เลื่อน 10%, 5% · shear 0.13, -0.12');
  assert.equal(OP_PRIMARY.translate_shear({ shearX: 0.5, shearY: 0 }), 'เลื่อน 10%, 5% · shear 0.5, 0');
});

test('ipC-9. palette-panel.js: FILTER & DENOISE = salt_pepper_noise, gaussian_noise, median_blur, gaussian_blur, box_blur, bilateral, wiener_filter; GEOMETRY ends with translate_shear; OP_TO_FILTER resolves each new op to its host node; each new op appears in exactly one section; the 3 new OP_ICON glyphs are distinct from every other OP_ICON and every GROUP_ICON glyph', () => {
  // op ใหม่ batch 4 ต่อท้าย — ห้ามสลับลำดับเดิม (IP batch 4A เพิ่ม guided_filter)
  assert.deepEqual(TOOL_SECTIONS.find((s) => s.title === 'FILTER & DENOISE').ops.slice(0, 7),
    ['salt_pepper_noise', 'gaussian_noise', 'median_blur', 'gaussian_blur', 'box_blur', 'bilateral', 'wiener_filter']);
  // ชิ้น D (2026-10-03) แทรก roi_polygon ต่อจาก crop บน palette (สเปก D §4.4) → ปัก 7 ตัวเดิมตามลำดับเดิม + translate_shear ต่อจาก undistort และอยู่ท้ายสุด
  const geo = TOOL_SECTIONS.find((s) => s.title === 'GEOMETRY').ops;
  const GEO7 = ['crop', 'resize', 'rotate', 'flip', 'perspective_warp', 'undistort', 'translate_shear'];
  assert.deepEqual(geo.filter((id) => GEO7.includes(id)), GEO7);
  assert.equal(geo.indexOf('translate_shear'), geo.indexOf('undistort') + 1);
  assert.equal(geo.at(-1), 'translate_shear');
  for (const id of NEW_OPS) {
    // 2026-10-07: palette ชี้กลุ่มระดับหมวด — HOST (กลุ่มเดิม) ยังเป็นที่อยู่ legacy ของ op ใน FILTERS
    assert.equal(OP_TO_FILTER[id], { gaussian_noise: 'filter_denoise', wiener_filter: 'filter_denoise', translate_shear: 'geometry' }[id]);
    assert.equal(TOOL_SECTIONS.filter((s) => s.ops.includes(id)).length, 1, `${id} must appear in exactly one section`);
  }
  const src = fs.readFileSync(new URL('../src/renderer/ui/palette-panel.js', import.meta.url), 'utf8');
  const body = src.slice(src.indexOf('const OP_ICON = {'), src.indexOf('};', src.indexOf('const OP_ICON = {')));
  const icons = {};
  for (const m of body.matchAll(/([a-z_0-9]+): "([^"]+)"/g)) icons[m[1]] = m[2];
  const groupGlyphs = new Set(Object.values(GROUP_ICON));
  for (const id of NEW_OPS) {
    assert.ok(icons[id], `OP_ICON.${id} missing`);
    const others = Object.entries(icons).filter(([k]) => k !== id).map(([, v]) => v);
    assert.ok(!others.includes(icons[id]), `OP_ICON.${id} (${icons[id]}) collides with another OP_ICON`);
    assert.ok(!groupGlyphs.has(icons[id]), `OP_ICON.${id} (${icons[id]}) collides with a GROUP_ICON`);
  }
});

test('ipC-10. codegen: all three export real single-image Python (never "WARNING: skipped" / "ข้าม" / round() / np.random / cv2.randn), carry the decisive pieces (uint32 mulberry32 constants, the BGR reversal [:, :, ::-1], boxFilter normalize=False + REFLECT_101, cumsum, int(np.floor(x + 0.5)), warpAffine INTER_LINEAR/BORDER_CONSTANT), embed clamped values, keep the gray/colour state (out "same"); the KEYPOINT_SORT_KEY line has its space back', () => {
  const one = (id, params) => buildPythonCode([{ id, params }], { displayName: 'X' });
  const all = [
    one('gaussian_noise', {}), one('gaussian_noise', { sigma: 0 }), one('wiener_filter', {}), one('wiener_filter', { noiseVar: 100 }),
    one('translate_shear', {}), one('translate_shear', { shearX: 0.2 }),
  ];
  for (const code of all) {
    assert.match(code, /def process\(path\):/);
    assert.doesNotMatch(code, /WARNING: skipped/);
    assert.doesNotMatch(code, /\(ข้าม: ยังไม่รองรับการ gen โค้ดของ/);
    assert.doesNotMatch(code, /\bround\(/);
    assert.doesNotMatch(code, /np\.random|cv2\.randn/);
    assert.doesNotMatch(code, /COLOR_BGR2GRAY/, 'needs "any" -> no forced grayscale');
  }
  const gn = one('gaussian_noise', { sigma: 999, seed: 0 });
  assert.match(gn, /np\.uint64\(1\) \+ np\.arange\(1, 2 \* n_pairs \+ 1, dtype=np\.uint64\) \* np\.uint64\(0x6D2B79F5\)/, 'seed 0 clamped to 1');
  for (const c of ['np.uint32(15)', 'np.uint32(1)', 'np.uint32(7)', 'np.uint32(61)', 'np.uint32(14)']) assert.ok(gn.includes(c), c);
  assert.match(gn, /\[:, :, ::-1\]/);
  // รอบแก้หลังรีวิว (TEST GAP): u1 ฝั่ง Python ต้องเป็น (r + 1) / 2^32 ทุกตัวอักษร (ช่วง (0, 1] กัน log 0 — เหมือน seededGaussian)
  assert.ok(gn.includes('rad = np.sqrt(-2 * np.log((rnd[0::2] + 1) / 4294967296))'), 'python u1 = (rnd[0::2] + 1) / 4294967296');
  assert.ok(gn.includes('ang = 2 * np.pi * (rnd[1::2] / 4294967296)'), 'python u2 = rnd[1::2] / 4294967296');
  assert.match(gn, /np\.floor\(img\.astype\(np\.float64\) \+ 100 \* z \+ 0\.5\)/, 'sigma 999 clamped to 100');
  assert.match(one('gaussian_noise', { seed: '7.6' }), /np\.uint64\(8\)/);
  assert.match(one('gaussian_noise', { sigma: -1 }), /sigma = 0, image unchanged/);
  const wn = one('wiener_filter', { ksize: 99, noiseVar: -5 });
  assert.match(wn, /k = 15\n/);
  assert.match(wn, /cv2\.boxFilter\(img, cv2\.CV_32F, \(k, k\), normalize=False, borderType=cv2\.BORDER_REFLECT_101\)/);
  assert.match(wn, /cv2\.boxFilter\(f32 \* f32, -1, \(k, k\), normalize=False, borderType=cv2\.BORDER_REFLECT_101\)/);
  assert.match(wn, /np\.cumsum\(/);
  assert.doesNotMatch(wn, /np\.mean/);
  assert.match(one('wiener_filter', { ksize: 14 }), /k = 15\n/);
  assert.match(one('wiener_filter', { noiseVar: 1e9 }), /noise = 10000  #/);
  const ts = one('translate_shear', { tx: 500, ty: -7, shearX: 3, shearY: -0.25 });
  assert.match(ts, /kx, ky = 1, -0\.25/);
  assert.match(ts, /tx = int\(np\.floor\(100 \/ 100 \* w \+ 0\.5\)\)/);
  assert.match(ts, /ty = int\(np\.floor\(-7 \/ 100 \* h \+ 0\.5\)\)/);
  assert.match(ts, /cx, cy = \(w - 1\) \/ 2, \(h - 1\) \/ 2/);
  assert.match(ts, /M = np\.array\(\[\[1, kx, -kx \* cy \+ tx\], \[ky, 1, -ky \* cx \+ ty\]\], dtype=np\.float64\)/);
  assert.match(ts, /cv2\.warpAffine\(img, M, \(w, h\), flags=cv2\.INTER_LINEAR, borderMode=cv2\.BORDER_CONSTANT, borderValue=0\)/);
  // out "same": หลัง op ทั้งสามบนภาพสี ขั้นที่ต้องการขาวดำยังต้องแปลง 1 ครั้ง · หลัง Grayscale ไม่ต้องแปลงซ้ำ
  const nGray = (code) => (code.match(/COLOR_BGR2GRAY\)  # \(แปลงขาวดำอัตโนมัติ\)/g) || []).length;
  for (const id of NEW_OPS) {
    const thr = { id: 'threshold', params: { mode: 'binary_threshold', binary_threshold__thresh: 127 } };
    assert.equal(nGray(buildPythonCode([{ id: HOST[id], params: { mode: id } }, thr])), 1, `${id} keeps colour`);
    assert.equal(nGray(buildPythonCode([{ id: 'color_space', params: { mode: 'grayscale' } }, { id: HOST[id], params: { mode: id } }, thr])), 0, `${id} keeps gray`);
  }
  const gsrc = fs.readFileSync(new URL('../src/renderer/codegen.js', import.meta.url), 'utf8');
  assert.match(gsrc, /^const KEYPOINT_SORT_KEY = "key=lambda k: \(-k\.response, k\.pt\[1\], k\.pt\[0\], -k\.size, k\.angle, k\.octave\)";$/m);
});

test('ipC-10b. teaching text: names/descs/hints of the 3 ops never say "ที่สุด"; gaussian says the seed fixes the pattern (unlike MATLAB) and gives the var 0.01 ~ sigma 25.5 hint; wiener says flat areas are smoothed hard and edges little, colour is per channel (MATLAB wiener2 is 2-D only) and the border is reflected unlike MATLAB zero padding; translate says % (MATLAB uses pixels) and shear is about the centre', () => {
  for (const id of NEW_OPS) {
    const texts = [OPS[id].name, OPS[id].desc, ...OPS[id].params.flatMap((p) => [p.label, p.desc, p.hint].filter(Boolean))];
    for (const t of texts) assert.ok(!t.includes('ที่สุด'), `${id}: "ที่สุด" in "${t}"`);
  }
  const gd = OPS.gaussian_noise.desc;
  assert.match(gd, /seed เดิม = ลาย noise เดิมทุกเฟรม/);
  assert.match(gd, /MATLAB/);
  assert.equal(OPS.gaussian_noise.params.find((p) => p.key === 'sigma').hint, 'MATLAB imnoise ค่าเริ่มต้น var 0.01 ≈ σ 25.5');
  const wd = OPS.wiener_filter.desc;
  assert.match(wd, /บริเวณเรียบ .*ถูกเกลี่ย.*แรง/);
  assert.match(wd, /ขอบ\/รายละเอียด .*ถูกแตะน้อย/);
  assert.match(wd, /ภาพสีทำแยกทีละช่อง/);
  assert.match(wd, /MATLAB wiener2 รับเฉพาะภาพ 2 มิติ/);
  assert.match(wd, /สะท้อน.*MATLAB ที่เติม 0/);
  assert.equal(OPS.wiener_filter.params.find((p) => p.key === 'noiseVar').hint, '0 = ประมาณจากภาพเองแบบ MATLAB');
  const td = OPS.translate_shear.desc;
  assert.match(td, /MATLAB imtranslate ใช้หน่วยพิกเซล/);
  assert.match(td, /รอบกึ่งกลางภาพ/);
});

// ===================================================================
// ipC-11 — codegen parity: the exported Python really produces the same pixels as the app (cv2 5.0.0 + numpy)
// ===================================================================
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'ip-batch2-C-2026-10-03-'));
process.on('exit', () => {
  try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ }
});
function pythonAvailable() {
  try { execFileSync('python', ['-c', 'import cv2, numpy'], { encoding: 'utf8' }); return true; } catch { return false; }
}
const HAS_PYTHON = pythonAvailable();
const PY_SKIP = { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' };

// ภาพสีของแอปเป็น RGB แต่ cv2.imread ให้ BGR — ส่งภาพเดียวกัน "ทางกายภาพ" ให้ Python เป็น BGR (สลับ R/B) แล้วสลับผลกลับก่อนเทียบ
// (ถ้าสคริปต์ลืมกลับลำดับค่าสุ่มด้วย [:, :, ::-1] ช่อง R กับ B จะได้ noise สลับกัน และภาพ R≠B ทุกพิกเซลจะ mismatch ทันที)
function swapRB3(buf) {
  const out = Buffer.from(buf);
  for (let i = 0; i < out.length; i += 3) { const t = out[i]; out[i] = out[i + 2]; out[i + 2] = t; }
  return out;
}
// swapOut: สลับ R/B ของผลกลับก่อนเทียบ (ค่าเริ่มต้น) — ปิดเมื่อขั้นสุดท้ายอยู่หลัง Color Space (ช่องของ cv2 เรียงเหมือนแอปแล้ว ดู ipC-13)
function runPython(code, src, { swapOut = true } = {}) {
  const ch = src.channels();
  assert.ok(ch === 1 || ch === 3, 'parity fixtures are 1 or 3 channels (cv2.imread never gives RGBA)');
  const shape = ch === 1 ? `(${src.rows}, ${src.cols})` : `(${src.rows}, ${src.cols}, 3)`;
  assert.ok(code.includes('img = cv2.imread(path)'), 'single-image loader line');
  let patched = code.replace('img = cv2.imread(path)', `img = np.fromfile(path, dtype=np.uint8).reshape(${shape})`);
  const mainIdx = patched.indexOf('if __name__ == "__main__":');
  assert.ok(mainIdx > 0);
  patched = patched.slice(0, mainIdx) + '\nimport sys\nresult = process(sys.argv[1])\nprint("NDIM", result.ndim)\nresult.tofile(sys.argv[2])\n';
  const id = Math.random().toString(36).slice(2);
  const scriptPath = path.join(TMP_DIR, `s_${id}.py`), inPath = path.join(TMP_DIR, `i_${id}.bin`), outPath = path.join(TMP_DIR, `o_${id}.bin`);
  fs.writeFileSync(scriptPath, patched, 'utf8');
  fs.writeFileSync(inPath, ch === 3 ? swapRB3(Buffer.from(src.data)) : Buffer.from(src.data));
  const stdout = execFileSync('python', [scriptPath, inPath, outPath], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
  const ndim = Number((stdout.match(/NDIM (\d)/) || [])[1]);
  const raw = fs.readFileSync(outPath);
  return { stdout, ndim, bytes: ndim === 3 && swapOut ? swapRB3(raw) : Buffer.from(raw) };
}
// คืนสถิติความต่างระหว่าง Python กับแอป (ทุกไบต์) — ndiff / mean abs / สัดส่วนที่ต่าง > 1 / max
function compareWithApp(opId, params, src, patch = (code) => code) {
  const label = `${opId} ${JSON.stringify(params)} ${src.cols}x${src.rows}x${src.channels()}`;
  const py = runPython(patch(buildPythonCode([{ id: opId, params }])), src);
  const ctx = {};
  const app = OPS[opId].apply(cv, src, params, ctx);
  try {
    assert.equal(py.ndim === 3 ? 3 : 1, app.channels(), `${label}: channel layout`);
    const js = app.data;
    assert.equal(py.bytes.length, js.length, `${label}: output size`);
    let ndiff = 0, over1 = 0, sum = 0, max = 0;
    for (let i = 0; i < js.length; i++) {
      const d = Math.abs(js[i] - py.bytes[i]);
      if (d) { ndiff++; sum += d; if (d > 1) over1++; if (d > max) max = d; }
    }
    return { label, ndiff, meanAbs: sum / js.length, over1Frac: over1 / js.length, max, stdout: py.stdout, info: ctx.info, pyBytes: py.bytes };
  } finally { app.delete(); }
}
function assertParity(opId, params, src, { expectStdout = null, patch } = {}) {
  const r = compareWithApp(opId, params, src, patch);
  assert.equal(r.ndiff, 0, `${r.label}: ndiff=${r.ndiff} max=${r.max}; python said "${r.stdout.trim()}", app said "${r.info}"`);
  if (expectStdout) assert.match(r.stdout, typeof expectStdout === 'function' ? expectStdout(r) : expectStdout, `${r.label}: python stdout`);
  return r;
}

test('ipC-11a. codegen parity gaussian_noise: ndiff 0 on gray and colour (R != B at every pixel, pins the [:, :, ::-1] reversal) at 97x61, 320x240 and 700x500 with non-default sigma/seed, plus 1x1 / 3x40 / an odd sample count, and sigma 0; the clipped share printed by Python equals the app info', PY_SKIP, () => {
  const imgs = [noiseGray(97, 61, 201), noiseRGB(97, 61, 202), noiseGray(320, 240, 203), noiseRGB(320, 240, 204), noiseRGB(700, 500, 205), noiseGray(700, 500, 206),
    noiseGray(1, 1, 207), noiseRGB(1, 1, 208), noiseRGB(3, 40, 209), noiseGray(3, 5, 210)];
  const settings = [{ sigma: 5, seed: 7 }, { sigma: 25, seed: 42 }, { sigma: 100, seed: 99999 }];
  try {
    imgs.forEach((src, i) => {
      const p = settings[i % settings.length];
      assertParity('gaussian_noise', p, src, {
        expectStdout: (r) => new RegExp(`clipped at 0/255 ${r.info.match(/ชนขอบ 0\/255 ([\d.]+)%/)[1].replace('.', '\\.')}% of values`),
      });
    });
    assertParity('gaussian_noise', { sigma: 60, seed: 3 }, imgs[1]);
    assertParity('gaussian_noise', { sigma: 0, seed: 3 }, imgs[1], { expectStdout: /sigma = 0, image unchanged/ });
    // รอบแก้หลังรีวิว (TEST GAP): ค่า % ที่ลงครึ่งพอดี (71/2000 = 3.55%) ต้องได้ "3.6%" ทั้งสองฝั่ง (ดู ipC-3c)
    const tie = constGray(50, 40, 250);
    try {
      const r = assertParity('gaussian_noise', { sigma: 3, seed: 2 }, tie, { expectStdout: /clipped at 0\/255 3\.6% of values/ });
      assert.match(r.info, /ชนขอบ 0\/255 3\.6% ของค่า/);
    } finally { tie.delete(); }
  } finally { for (const m of imgs) m.delete(); }
});

test('ipC-11b. codegen parity wiener_filter: ndiff 0 on gray and colour (scene + pure noise) at 97x61, 320x240 and 700x500 with k 3/5/15 and noise auto/100/2500, plus 1x1 / 3x40 at k 15 and a constant image (den = 0); the noise estimate printed by Python equals the app info (R/G/B order)', PY_SKIP, () => {
  const imgs = [sceneGray(97, 61, 211), sceneRGB(97, 61, 212), noiseRGB(320, 240, 213), sceneGray(320, 240, 214), sceneRGB(700, 500, 215),
    noiseGray(1, 1, 216), noiseRGB(3, 40, 217), noiseGray(40, 3, 218), constGray(30, 20, 99)];
  const settings = [{ ksize: 3, noiseVar: 0 }, { ksize: 5, noiseVar: 0 }, { ksize: 15, noiseVar: 0 }, { ksize: 7, noiseVar: 100 }, { ksize: 11, noiseVar: 2500 }];
  const noiseOf = (r) => {
    const m = r.info.match(/= ([\d./]+) \((ประมาณจากภาพ|กำหนดเอง)\)$/);
    return new RegExp(`noise variance = ${m[1].replace(/\./g, '\\.')} \\(${m[2] === 'กำหนดเอง' ? 'given' : 'estimated'}\\)`);
  };
  // ค่า noise ที่ประมาณได้ต้องตรงทุก bit (ไม่ใช่แค่ทศนิยม 1 ตำแหน่ง): เทสต์แทรกบรรทัดพิมพ์ repr ของ noise ลงสคริปต์ (ฝั่งเทสต์เท่านั้น)
  // แล้วเทียบกับผลรวมแบบบวกตามลำดับของ wienerRef — np.mean (pairwise) แทน cumsum จะต่างที่หลักท้าย ๆ และถูกจับที่นี่
  const OUT_LINE = 'img = np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)';
  const printNoise = (code) => {
    assert.ok(code.includes(OUT_LINE));
    return code.replace(OUT_LINE, `${OUT_LINE}\n    print("NOISE_REPR", repr([float(v) for v in np.atleast_1d(noise)[::-1]]))`);
  };
  try {
    for (const src of imgs.slice(0, 5)) for (const p of settings) {
      const exact = p.noiseVar === 0 && src.cols * src.rows <= 320 * 240;
      const r = assertParity('wiener_filter', p, src, { expectStdout: noiseOf, patch: exact ? printNoise : undefined });
      if (exact) {
        const pyNoise = JSON.parse(r.stdout.match(/NOISE_REPR (\[.*\])/)[1]);
        assert.deepEqual(pyNoise, Array.from(wienerRef(src, p.ksize, 0).noise), `${r.label}: noise estimate must match bit for bit`);
      }
    }
    for (const src of imgs.slice(5)) assertParity('wiener_filter', { ksize: 15, noiseVar: 0 }, src, { expectStdout: noiseOf });
    // รอบแก้หลังรีวิว (TEST GAP): noiseVar ที่ไม่ใช่จำนวนเต็ม — บรรทัด print "(given)" ของ Python ต้องพิมพ์ค่าเดียวกับ info ของแอป (12.25)
    for (const src of [imgs[0], imgs[1]]) {
      const r = assertParity('wiener_filter', { ksize: 5, noiseVar: 12.25 }, src, { expectStdout: noiseOf });
      assert.match(r.info, /noise variance = 12\.25 \(กำหนดเอง\)$/);
      assert.match(r.stdout, /wiener 5x5: noise variance = 12\.25 \(given\)/);
    }
  } finally { for (const m of imgs) m.delete(); }
});

test('ipC-11c. codegen parity translate_shear WITHOUT shear: ndiff 0 for positive / negative / mixed shifts, the .5 rounding cases (w = 97: tx 50 -> 48.5, tx -50 -> -48.5; h = 61: ty 50 -> 30.5), +-100% (out of frame), gray and colour, 1x1 / 3x40; Python prints the same pixel shift as the app', PY_SKIP, () => {
  const g = noiseGray(97, 61, 221), c = noiseRGB(97, 61, 222), big = noiseRGB(700, 500, 223), one = noiseRGB(1, 1, 224), thin = noiseGray(3, 40, 225);
  try {
    const cases = [[g, 10, 5], [c, -10, -5], [g, 50, 50], [c, -50, -50], [g, 33, -67], [c, 100, 0], [g, 0, -100], [big, 17, -23], [one, 50, -50], [thin, 50, 49]];
    for (const [src, tx, ty] of cases) {
      const dx = pxOf(tx, src.cols), dy = pxOf(ty, src.rows);
      const s = (v) => (v >= 0 ? '+' : '') + v;
      assertParity('translate_shear', { tx, ty, shearX: 0, shearY: 0 }, src, { expectStdout: new RegExp(`shifted \\(\\${s(dx)}, \\${s(dy)}\\) px, shear x=0\\.00 y=0\\.00`) });
    }
  } finally { for (const m of [g, c, big, one, thin]) m.delete(); }
});

// ภาพ "ธรรมชาติ" (ความสว่างเปลี่ยนต่อเนื่อง + ขอบระดับกลาง) — เกณฑ์ค่าเผื่อของสเปก §5 ใช้กับภาพแบบนี้
function smoothRGB(w, h) {
  return rgb(w, h, (x, y) => [
    Math.round(127 + 100 * Math.sin(x / 17) * Math.cos(y / 23)),
    Math.round(127 + 90 * Math.sin((x + y) / 29)),
    Math.round(60 + (150 * x) / w),
  ]);
}
// bilinear แบบ double ล้วนตาม matrix ในสเปก §4.3 (ไม่ใช้ OpenCV): dst(x, y) = src(M⁻¹·(x, y)) · นอกภาพ = 0
function exactWarp(src, tx, ty, kx, ky) {
  const w = src.cols, h = src.rows, ch = src.channels(), s = src.data;
  const c0 = -kx * ((h - 1) / 2) + pxOf(tx, w), c1 = -ky * ((w - 1) / 2) + pxOf(ty, h);
  const det = 1 - kx * ky;
  const ia = 1 / det, ib = -kx / det, id = -ky / det, ie = 1 / det;
  const ic = -(ia * c0 + ib * c1), iff = -(id * c0 + ie * c1);
  const px = (X, Y, k) => (X < 0 || Y < 0 || X >= w || Y >= h ? 0 : s[(Y * w + X) * ch + k]);
  const out = new Float64Array(w * h * ch);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const sx = ia * x + ib * y + ic, sy = id * x + ie * y + iff;
    const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
    for (let k = 0; k < ch; k++) {
      out[(y * w + x) * ch + k] = (1 - fx) * (1 - fy) * px(x0, y0, k) + fx * (1 - fy) * px(x0 + 1, y0, k)
        + (1 - fx) * fy * px(x0, y0 + 1, k) + fx * fy * px(x0 + 1, y0 + 1, k);
    }
  }
  return out;
}

test('ipC-11d. codegen parity translate_shear WITH shear (spec 5 tolerance -- cv2 5.0 warpAffine is exact bilinear, opencv.js 4.10 snaps source coordinates to 1/32 px): on natural-looking images (smooth colour, gray scene) at 97x61 / 320x240 / 700x500 x 6 shear settings, mean abs <= 0.25, share of bytes differing by > 1 <= 2%, max <= 6; the single white point lands on the same pixel in Python as in the app; the singular guard is skipped identically', PY_SKIP, (t) => {
  const imgs = [smoothRGB(97, 61), sceneGray(97, 61, 231), smoothRGB(320, 240), sceneGray(320, 240, 234), smoothRGB(700, 500), sceneGray(700, 500, 236)];
  const shears = [[0.2, 0, 10, 5], [0, -0.35, -20, 15], [0.5, 0.3, 0, 0], [-1, 0.5, 5, -5], [0.05, 0.05, 50, 50], [0.95, 1, 0, 0]];
  let worst = { meanAbs: 0, over1Frac: 0, max: 0 };
  try {
    for (const src of imgs) for (const [kx, ky, tx, ty] of shears) {
      const r = compareWithApp('translate_shear', { tx, ty, shearX: kx, shearY: ky }, src);
      assert.ok(r.meanAbs <= 0.25 && r.over1Frac <= 0.02 && r.max <= 6, `${r.label}: mean ${r.meanAbs.toFixed(4)} >1 ${(r.over1Frac * 100).toFixed(2)}% max ${r.max}`);
      assert.match(r.stdout, /shifted \([+-]\d+, [+-]\d+\) px/, `${r.label}: python warped`);
      worst = { meanAbs: Math.max(worst.meanAbs, r.meanAbs), over1Frac: Math.max(worst.over1Frac, r.over1Frac), max: Math.max(worst.max, r.max) };
    }
    t.diagnostic(`ipC-11d worst over ${imgs.length * shears.length} natural cases: mean abs ${worst.meanAbs.toFixed(4)}, >1 ${(worst.over1Frac * 100).toFixed(3)}%, max ${worst.max}`);
    for (const [x0, y0, tx, ty, kx, ky] of POINT_CASES) {
      const src = pointImage(x0, y0);
      try {
        const r = compareWithApp('translate_shear', { tx, ty, shearX: kx, shearY: ky }, src);
        const [ex, ey] = expectedPoint(x0, y0, tx, ty, kx, ky);
        const m = argmax(r.pyBytes, POINT_W);
        assert.deepEqual([m.x, m.y, m.value, m.count], [ex, ey, 255, 1], `python point (${x0},${y0}) kx ${kx} ky ${ky}`);
      } finally { src.delete(); }
    }
    for (const [kx, ky] of [[1, 1], [-1, -1], [0.96, 1]]) {
      assertParity('translate_shear', { shearX: kx, shearY: ky }, imgs[0], { expectStdout: /translate\/shear skipped: shear x \* shear y is too close to 1/ });
    }
  } finally { for (const m of imgs) m.delete(); }
});

test('ipC-11e. translate_shear WITH shear on worst-case images (random noise, full-contrast 0/255 stripes, colour noise): both sides are checked against an exact double-precision bilinear warp of the spec matrix -- Python (cv2 5.0) within 0.5 of it everywhere (= exact + rounding), the app within 8.5 (opencv.js snaps the source position to 1/32 px per axis: error <= (|dI/dx| + |dI/dy|)/64 <= 2*255/64 + 0.5). This pins the matrix of BOTH sides independently (a wrong sign / centre / swapped shear moves pixels by whole steps and blows far past 8.5) and documents why the spec tolerance only applies to natural images', PY_SKIP, (t) => {
  const r0 = lcg(241);
  const imgs = [noiseGray(61, 47, 242), gray(64, 48, (x, y) => ((x + y) % 10 < 5 ? 0 : 255)), noiseRGB(53, 41, 243), rgb(40, 30, () => [r0() < 0.5 ? 0 : 255, r0() < 0.5 ? 0 : 255, r0() < 0.5 ? 0 : 255])];
  const shears = [[0.2, 0, 10, 5], [0, -0.35, -20, 15], [0.5, 0.3, 0, 0], [-1, 0.5, 5, -5], [0.05, 0.05, 50, 50]];
  let worstApp = 0, worstPy = 0, worstPair = 0;
  try {
    for (const src of imgs) for (const [kx, ky, tx, ty] of shears) {
      const r = compareWithApp('translate_shear', { tx, ty, shearX: kx, shearY: ky }, src);
      const app = OPS.translate_shear.apply(cv, src, { tx, ty, shearX: kx, shearY: ky }, {});
      try {
        const ex = exactWarp(src, tx, ty, kx, ky);
        let dApp = 0, dPy = 0;
        for (let i = 0; i < ex.length; i++) {
          dApp = Math.max(dApp, Math.abs(app.data[i] - ex[i]));
          dPy = Math.max(dPy, Math.abs(r.pyBytes[i] - ex[i]));
        }
        assert.ok(dPy <= 0.5 + 1e-6, `${r.label}: python vs exact bilinear max ${dPy}`);
        assert.ok(dApp <= 8.5, `${r.label}: app vs exact bilinear max ${dApp}`);
        worstApp = Math.max(worstApp, dApp); worstPy = Math.max(worstPy, dPy); worstPair = Math.max(worstPair, r.max);
      } finally { app.delete(); }
    }
    t.diagnostic(`ipC-11e worst-case images: max |app - exact| ${worstApp.toFixed(2)}, max |python - exact| ${worstPy.toFixed(2)}, max |app - python| ${worstPair}`);
  } finally { for (const m of imgs) m.delete(); }
});

// ===================================================================
// รอบแก้หลังรีวิว (2026-10-03) — NIT-1 ฝั่ง Python · MAJOR-1 ลำดับช่องสีหลัง Color Space (ctx.mirrored ของ buildPythonCode)
// ===================================================================
test('ipC-11f. codegen parity of the shear numbers (NIT-1): Python prints shear x/y with the same half-up 2-decimal rule as the app info (np.floor(x*100 + 0.5)/100 then :.2f) -- 0.125 / -0.125 / 0.375 / 0.625 / -0.875 / 0.15000000000000002 give the same text on both sides (a bare {kx:.2f} rounds 0.125 -> 0.12 and 0.625 -> 0.62)', PY_SKIP, () => {
  const src = smoothRGB(41, 31);
  try {
    for (const [kx, ky] of [[0.125, 0], [-0.125, 0.375], [0.625, -0.875], [0.15000000000000002, -0.05]]) {
      const r = compareWithApp('translate_shear', { tx: 0, ty: 0, shearX: kx, shearY: ky }, src);
      const appShear = r.info.match(/shear x=(-?[\d.]+) y=(-?[\d.]+)/);
      const pyShear = r.stdout.match(/shear x=(-?[\d.]+) y=(-?[\d.]+)/);
      assert.ok(appShear && pyShear, `${r.label}: both sides print the shear ("${r.info}" / "${r.stdout.trim()}")`);
      assert.deepEqual(pyShear.slice(1), appShear.slice(1), `${r.label}: python vs app shear text`);
    }
  } finally { src.delete(); }
});

// ขั้น pipeline แบบที่ editor ส่งให้ worker/codegen (FILTERS id + mode + param ที่มี prefix)
const CS = (space) => ({ id: 'color_space', params: { mode: 'colorspace', colorspace__space: space } });
const STEP = (id, mode, extra = {}) => ({ id, params: { ...defaultParams(id), mode, ...extra } });
const SINGLE = (id, extra = {}) => ({ id, params: { ...defaultParams(id), ...extra } }); // FILTERS ที่มี op เดียว (ไม่มี key mode)
const NOISE = (sigma = 20, seed = 3) => STEP('noise', 'gaussian_noise', { gaussian_noise__sigma: sigma, gaussian_noise__seed: seed });
const WIENER = () => STEP('blur', 'wiener_filter', { wiener_filter__ksize: 5, wiener_filter__noiseVar: 0 });
const REVERSE_Z = 'z = z.reshape(img.shape[0], img.shape[1], 3)[:, :, ::-1]';
const REVERSE_NOISE = 'np.atleast_1d(noise)[::-1]';

test('ipC-12. (static) codegen tracks the channel order at compile time (ctx.mirrored, MAJOR-1): the BGR reversal of gaussian_noise / the R,G,B print order of wiener are emitted only while the Python channels are the app channels mirrored -- dropped right after Color Space (and through colour -> colour steps after it), restored by any gray step (incl. the automatic BGR2GRAY of a needs:"gray" step) and by gray -> colour steps; Color Space from a gray image is unmirrored again; the unmirrored wiener print does not claim R/G/B', () => {
  const gen = (pipe) => buildPythonCode(pipe, { displayName: 'X' });
  const cases = [
    // [label, pipeline, mirrored (reversal expected)]
    ['noise alone', [NOISE()], true],
    ['colorspace hsv -> noise', [CS('hsv'), NOISE()], false],
    ['colorspace hls -> noise', [CS('hls'), NOISE()], false],
    ['colorspace lab -> noise', [CS('lab'), NOISE()], false],
    ['colorspace ycrcb -> noise', [CS('ycrcb'), NOISE()], false],
    ['colorspace -> gaussian blur (same) -> noise', [CS('hsv'), STEP('blur', 'gaussian_blur'), NOISE()], false],
    ['colorspace -> translate_shear (same) -> noise', [CS('lab'), STEP('geometry', 'translate_shear'), NOISE()], false],
    ['colorspace -> rgb_adjust (colour -> colour) -> noise', [CS('hsv'), STEP('color_space', 'rgb_adjust'), NOISE()], false],
    ['colorspace -> white_balance (colour -> colour) -> noise', [CS('ycrcb'), STEP('colour', 'white_balance'), NOISE()], false],
    ['colorspace -> grayscale -> noise', [CS('hsv'), STEP('color_space', 'grayscale'), NOISE()], true],
    ['colorspace -> binary threshold (auto BGR2GRAY) -> noise', [CS('hsv'), STEP('threshold', 'binary_threshold'), NOISE()], true],
    ['colorspace -> grayscale -> rgb_adjust (gray -> colour) -> noise', [CS('hsv'), STEP('color_space', 'grayscale'), STEP('color_space', 'rgb_adjust'), NOISE()], true],
    ['colorspace -> find_contours (gray in, colour drawing out) -> noise', [CS('lab'), STEP('shape_analysis', 'find_contours'), NOISE()], true],
    ['grayscale -> colorspace (GRAY2BGR then BGR2HSV) -> noise', [STEP('color_space', 'grayscale'), CS('hsv'), NOISE()], false],
    ['colorspace -> grayscale -> colorspace -> noise', [CS('hsv'), STEP('color_space', 'grayscale'), CS('lab'), NOISE()], false],
    // debugger delta: grabcut แบบ mask ได้ภาพ 2 มิติทั้งที่ out = "color" (ของเดิม) — grayOutOf ทำให้ flag กลับเป็น BGR
    ['colorspace -> grabcut(mask) -> noise', [CS('hsv'), SINGLE('grabcut', { output: 'mask' }), NOISE()], true],
    ['colorspace -> grabcut(mask) -> watershed (ndim 2 -> BGR) -> noise', [CS('hsv'), SINGLE('grabcut', { output: 'mask' }), SINGLE('watershed'), NOISE()], true],
    ['colorspace -> grabcut(extract) -> noise', [CS('hsv'), SINGLE('grabcut', { output: 'extract' }), NOISE()], false],
  ];
  for (const [label, pipe, mirrored] of cases) {
    const code = gen(pipe);
    assert.equal(code.includes(REVERSE_Z), mirrored, `${label}: gaussian_noise reversal ${mirrored ? 'expected' : 'must be absent'}`);
    if (!mirrored) assert.ok(code.includes('z = z.reshape(img.shape[0], img.shape[1], 3)  #'), `${label}: unreversed reshape`);
    const w = gen([...pipe.slice(0, -1), WIENER()]);
    assert.equal(w.includes(REVERSE_NOISE), mirrored, `${label}: wiener print reversal ${mirrored ? 'expected' : 'must be absent'}`);
    assert.equal(w.includes('noise variance per channel (image order) ='), !mirrored, `${label}: neutral wiener label only when unmirrored`);
    assert.equal(w.includes('noise variance = {noise_txt}'), mirrored, `${label}: the usual wiener print only when mirrored`);
  }
  // σ = 0 และ noiseVar ที่กำหนดเองไม่ขึ้นกับลำดับช่อง — ได้บรรทัดเดิมไม่ว่าจะอยู่หลัง Color Space หรือไม่
  const tail = (code) => code.slice(code.indexOf('# ── ขั้นที่ 2'));
  assert.equal(tail(gen([CS('hsv'), NOISE(0)])), tail(gen([STEP('blur', 'gaussian_blur'), NOISE(0)])));
  const given = STEP('blur', 'wiener_filter', { wiener_filter__noiseVar: 100 });
  assert.equal(tail(gen([CS('hsv'), given])), tail(gen([STEP('blur', 'gaussian_blur'), given])));
  // path 2 ภาพ (buildPythonCode2Image — P-21 นอกขอบเขต ไม่ได้ track ctx.mirrored): ภาพเริ่มเป็น BGR จึงต้องยังกลับลำดับเหมือนก่อนแก้
  // ทั้งสาย A และสาย B (ctx ที่ไม่มี key mirrored = ถือว่ากลับด้าน)
  const two = gen([NOISE(), { ...STEP('image_arithmetic', 'imabsdiff'), src2: { pipeline: [NOISE(15, 2)] } }]);
  assert.match(two, /def process\(path_a, path_b\):/);
  assert.equal(two.split(REVERSE_Z).length - 1, 1, 'chain A keeps the reversal');
  assert.equal(two.split(REVERSE_Z.replace(/\bimg\b/g, 'img_b')).length - 1, 1, 'chain B keeps the reversal');
});

// รันทั้ง pipeline แบบที่ worker ทำ (needs "gray" → RGB2GRAY อัตโนมัติ แล้ว FILTERS[id].apply) — คืนผลสุดท้าย + info ทุกขั้น
function appPipeline(pipe, src) {
  let cur = src.clone();
  const infos = [];
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
      try {
        const o = f.apply(cv, input, st.params, ctx);
        cur.delete();
        cur = o;
      } finally { if (conv) conv.delete(); }
      infos.push(ctx.info);
    }
    return { mat: cur, infos };
  } catch (err) { cur.delete(); throw err; }
}
// mirroredOut = ผลสุดท้ายของ Python เป็นภาพกลับช่อง (BGR) ของแอปหรือไม่ — หลัง Color Space cv2 เก็บช่องเรียงเหมือนแอปแล้ว (HSV/HSV)
function pipelineDiff(pipe, src, { mirroredOut }) {
  const py = runPython(buildPythonCode(pipe), src, { swapOut: mirroredOut });
  const app = appPipeline(pipe, src);
  const label = `${pipe.map((s) => s.params.mode + (s.params.colorspace__space ? `(${s.params.colorspace__space})` : '')).join(' -> ')} ${src.cols}x${src.rows}`;
  try {
    assert.equal(py.bytes.length, app.mat.data.length, `${label}: output size`);
    let ndiff = 0, max = 0;
    for (let i = 0; i < py.bytes.length; i++) { const d = Math.abs(py.bytes[i] - app.mat.data[i]); if (d) { ndiff++; if (d > max) max = d; } }
    return { ndiff, max, stdout: py.stdout, info: app.infos.at(-1), label };
  } finally { app.mat.delete(); }
}
function assertPipelineParity(pipe, src, opts) {
  const r = pipelineDiff(pipe, src, opts);
  assert.equal(r.ndiff, 0, `${r.label}: ndiff=${r.ndiff} max=${r.max}; python said "${r.stdout.trim()}", app said "${r.info}"`);
  return r;
}

test('ipC-13. codegen parity across a Color Space step (MAJOR-1 regression): colorspace (hsv / hls / lab / ycrcb) -> gaussian_noise gives ndiff 0 with no channel swap and the same clipped share; colorspace -> wiener gives ndiff 0 and Python prints the noise estimates in the same order as the app info (with a label that does not claim R/G/B); the flag survives a "same" step (colorspace -> gaussian blur -> noise); gray -> colour -> gaussian (grayscale -> rgb_adjust with R != B -> noise) and plain noise / wiener keep the BGR order and stay at ndiff 0', PY_SKIP, () => {
  // ภาพที่ 1 = ภาพที่ debugger ใช้ repro (Color Space ทั้ง 4 แบบตรงกับ cv2 ทุก byte) · ภาพที่ 2 = noise สี R≠B — HLS ของ opencv.js 4.10
  // กับ cv2 5.0 ปัดต่างกันเองบนภาพนี้ 2 byte (±1 — ของเดิม ไม่เกี่ยวกับรอบนี้ ดู WORKLOG §5) จึงเช็ค HLS ของภาพที่ 2 แบบเทียบฐานแยกด้านล่าง
  const imgs = [rgb(40, 30, (x, y) => [(x * 6) & 255, (y * 8) & 255, (x * y) & 255]), noiseRGB(97, 61, 251)];
  try {
    const base = pipelineDiff([CS('hls')], imgs[1], { mirroredOut: false });
    const hls = pipelineDiff([CS('hls'), NOISE(20, 3)], imgs[1], { mirroredOut: false });
    assert.ok(base.ndiff <= 4 && base.max <= 1, `hls baseline on the noise image: ndiff ${base.ndiff} max ${base.max}`);
    assert.ok(hls.ndiff <= base.ndiff && hls.max <= 1, `hls -> noise must add no mismatch of its own: ndiff ${hls.ndiff} (baseline ${base.ndiff}) max ${hls.max}`);
    for (const [i, src] of imgs.entries()) {
      for (const space of (i === 0 ? ['hsv', 'hls', 'lab', 'ycrcb'] : ['hsv', 'lab', 'ycrcb'])) {
        assertPipelineParity([CS(space)], src, { mirroredOut: false }); // ตัวเทียบเอง: Color Space ขั้นเดียวต้อง ndiff 0 แบบไม่สลับช่อง
        const n = assertPipelineParity([CS(space), NOISE(20, 3)], src, { mirroredOut: false });
        const pct = n.info.match(/ชนขอบ 0\/255 ([\d.]+)%/)[1];
        assert.match(n.stdout, new RegExp(`clipped at 0/255 ${pct.replace('.', '\\.')}% of values`), `${n.label}: clipped share`);
        const w = assertPipelineParity([CS(space), WIENER()], src, { mirroredOut: false });
        const appNoise = w.info.match(/= ([\d./]+) \(ประมาณจากภาพ\)$/)[1];
        assert.match(w.stdout, new RegExp(`wiener 5x5: noise variance per channel \\(image order\\) = ${appNoise.replace(/\./g, '\\.')} \\(estimated\\)`), `${w.label}: python noise order = app info order`);
        assert.doesNotMatch(w.stdout, /R\/G\/B/);
      }
      assertPipelineParity([CS('hsv'), STEP('blur', 'gaussian_blur'), NOISE(15, 9)], src, { mirroredOut: false });
      const adj = STEP('color_space', 'rgb_adjust', { rgb_adjust__rOffset: 40, rgb_adjust__bOffset: -30 });
      assertPipelineParity([STEP('color_space', 'grayscale'), adj, NOISE(20, 4)], src, { mirroredOut: true });
      assertPipelineParity([NOISE(25, 11)], src, { mirroredOut: true });
      if (src === imgs[0]) { // debugger delta repro: grabcut แบบ mask (ภาพ 2 มิติ) → watershed แปลงกลับเป็น BGR → noise ต้องกลับลำดับอีกครั้ง
        const discs = rgb(80, 60, (x, y) => ((x - 30) ** 2 + (y - 30) ** 2 < 225 || (x - 50) ** 2 + (y - 30) ** 2 < 225 ? [200, 60, 40] : [20, 30, 50]));
        try {
          assertPipelineParity([CS('hsv'), SINGLE('grabcut', { output: 'mask' })], discs, { mirroredOut: false });
          const r = assertPipelineParity([CS('hsv'), SINGLE('grabcut', { output: 'mask' }), SINGLE('watershed'), NOISE(20, 3)], discs, { mirroredOut: true });
          assert.match(r.stdout, new RegExp(`clipped at 0/255 ${r.info.match(/ชนขอบ 0\/255 ([\d.]+)%/)[1].replace('.', '\\.')}% of values`));
        } finally { discs.delete(); }
      }
      const w2 = assertPipelineParity([WIENER()], src, { mirroredOut: true });
      assert.match(w2.stdout, new RegExp(`noise variance = ${w2.info.match(/R\/G\/B = ([\d./]+)/)[1].replace(/\./g, '\\.')} \\(estimated\\)`), 'plain wiener: python R/G/B order = app');
    }
  } finally { for (const m of imgs) m.delete(); }
});

// 2026-10-02 — acceptance tests ของ IP batch 2 ชิ้น B (หมวด IMAGE ARITHMETIC สองภาพ: imarith, logical_ops, inpaint,
// hist_match — ทั้ง 4 ตัวอยู่ใน node เดิม FILTERS.image_arithmetic)
// สเปก: Vision_boi/specs/ip_batch2_B_2026-10-02.md — ตัวเลขเกณฑ์ผ่านทุกตัวยืนยันด้วยการรันจริงก่อนลงโค้ด (คำนวณมือ +
// เทียบ cv2 5.0.0 / numpy 2.5.1 ที่ติดตั้งในเครื่องนี้) ถ้าเทสต์ล้มให้ถือว่าโค้ดผิด ไม่ใช่ตัวเลขผิด
//
// นโยบาย leak (P-14, WORK-TRACKER §1): address-fingerprint ใช้ได้เฉพาะเทสต์แรกของไฟล์ (ipB-0) พร้อมคู่แฝด
// wrapper-balance (ipB-0w) — เทสต์อื่นใช้ assertLeakFree / assertWrapperBalance
// ไม่ปักจำนวน OPS รวม (สเปก §6 — batch C–E จะเพิ่มอีก) เช็คเฉพาะว่า op ของ batch นี้มีอยู่ และ FILTERS ยังเป็น 26
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { OPS, FILTERS, defaultParams } from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS, OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { GROUP_ICON, OP_PRIMARY } from '../src/renderer/filters/ui-meta.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import {
  assertLeakFree, assertWrapperBalance, fingerprintLeakProbe, makeThrowingCv,
} from './_leak-helpers.mjs';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.inpaint === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

const NEW_OPS = ['imarith', 'logical_ops', 'inpaint', 'hist_match'];
const IMARITH_MODES = ['add', 'subtract', 'multiply', 'divide'];
const LOGICAL_MODES = ['and', 'or', 'xor', 'a_and_not_b'];
const INPAINT_METHODS = ['telea', 'ns'];
const NO_SRC2_TEXT = "ยังไม่ได้ต่อสาย 'เข้า B' — ลากเส้นจากอีกจุดหนึ่งมาต่อที่ช่องอินพุตที่สองของกล่องนี้";

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
function row(arr) { return cv.matFromArray(1, arr.length, cv.CV_8UC1, arr); }
function noiseGray(w, h, seed) { const r = lcg(seed); return gray(w, h, () => Math.floor(r() * 256)); }
function noiseRGB(w, h, seed) { const r = lcg(seed); return rgb(w, h, () => [Math.floor(r() * 256), Math.floor(r() * 256), Math.floor(r() * 256)]); }
// ภาพสีที่มีทั้งแนวโน้มและ texture (ใช้กับ inpaint/hist_match)
function sceneRGB(w, h, seed = 5) {
  const r = lcg(seed);
  return rgb(w, h, (x, y) => [
    (40 + 3 * x + Math.floor(r() * 20)) & 255,
    (90 + 2 * y + Math.floor(r() * 20)) & 255,
    (200 - x - y + Math.floor(r() * 10)) & 255,
  ]);
}
// mask: n พิกเซลแรก (row-major) เป็นขาว — คุมเปอร์เซ็นต์ได้แม่นยำ
function firstN(w, h, n, v = 255) { return gray(w, h, (x, y) => (y * w + x < n ? v : 0)); }
// mask ก้อนสี่เหลี่ยมเล็ก (ขาว) ในกรอบ [x0,x1)×[y0,y1)
function blobMask(w, h, x0, y0, x1, y1) { return gray(w, h, (x, y) => (x >= x0 && x < x1 && y >= y0 && y < y1 ? 255 : 0)); }
function bytes(m) { return Array.from(m.data); }
function run(opId, A, B, params = {}, cvUse = cv) {
  const ctx = B === undefined ? {} : { src2: B };
  const out = OPS[opId].apply(cvUse, A, params, ctx);
  return { out, ctx };
}

// ===================================================================
// ipB-0 / ipB-0w — ด่าน leak หลัก (P-14 ข้อ 1 + คู่บังคับข้อ 2)
// ===================================================================
function batchBLeakPaths() {
  const cA = noiseRGB(24, 18, 1), cB = noiseRGB(24, 18, 2), gA = noiseGray(24, 18, 3), gB = noiseGray(24, 18, 4);
  const cBsmall = noiseRGB(10, 7, 5), gBsmall = noiseGray(10, 7, 6);
  const rgbaA = toRGBA(cA), rgbaB = toRGBA(cB);
  const scene = sceneRGB(24, 18), sceneG = gray(24, 18, (x, y) => (60 + 4 * x + y) & 255), sceneRGBA = toRGBA(scene);
  const blob = blobMask(24, 18, 8, 6, 13, 10), blobC = (() => { const m = new cv.Mat(); cv.cvtColor(blob, m, cv.COLOR_GRAY2RGB); return m; })();
  const blobSmall = blobMask(12, 9, 4, 3, 7, 5);
  const noMask = firstN(24, 18, 0), bigMask = firstN(24, 18, Math.ceil(24 * 18 * 0.3));
  const empty = new cv.Mat();
  const paths = [];
  const add = (id, params, A, B, tag) => paths.push({
    label: `${id} ${tag}`,
    run: (c) => { const o = OPS[id].apply(c, A, params, B === undefined ? {} : { src2: B }); o.delete(); },
  });
  for (const id of NEW_OPS) {
    add(id, {}, cA, undefined, 'no src2');
    add(id, {}, cA, empty, 'src2 0x0');
    add(id, {}, empty, cB, 'A 0x0');
  }
  for (const operation of IMARITH_MODES) {
    add('imarith', { operation }, cA, cB, `${operation} colour+colour`);
    add('imarith', { operation }, gA, gB, `${operation} gray+gray`);
  }
  add('imarith', { operation: 'add' }, cA, gB, 'colour+gray');
  add('imarith', { operation: 'add' }, gA, cB, 'gray+colour');
  add('imarith', { operation: 'subtract' }, cA, cBsmall, 'size mismatch colour');
  add('imarith', { operation: 'divide' }, gA, gBsmall, 'size mismatch gray');
  add('imarith', { operation: 'multiply' }, rgbaA, rgbaB, 'rgba+rgba');
  for (const operation of LOGICAL_MODES) {
    add('logical_ops', { operation }, gA, gB, `${operation} gray+gray`);
    add('logical_ops', { operation }, cA, cB, `${operation} colour+colour`);
  }
  add('logical_ops', { operation: 'xor' }, gA, cBsmall, 'size mismatch');
  add('logical_ops', { operation: 'or' }, rgbaA, gB, 'rgba A');
  for (const method of INPAINT_METHODS) {
    add('inpaint', { method, radius: 3 }, scene, blob, `${method} colour`);
    add('inpaint', { method, radius: 2 }, sceneG, blob, `${method} gray`);
  }
  add('inpaint', { method: 'telea' }, scene, blobC, 'colour mask B');
  add('inpaint', { method: 'ns' }, sceneRGBA, blob, 'rgba A');
  add('inpaint', { method: 'telea' }, scene, blobSmall, 'size mismatch');
  add('inpaint', {}, scene, noMask, 'mask empty');
  add('inpaint', {}, scene, bigMask, 'mask > 25%');
  add('hist_match', {}, gA, gB, 'gray+gray');
  add('hist_match', {}, cA, cB, 'colour+colour');
  add('hist_match', {}, cA, gB, 'colour A + single-channel B');
  add('hist_match', {}, gA, cB, 'gray A + colour B');
  add('hist_match', {}, rgbaA, rgbaB, 'rgba A + rgba B');
  add('hist_match', {}, gA, rgbaB, 'gray A + rgba B');
  add('hist_match', {}, cA, cBsmall, 'B of different size');
  return {
    paths,
    cleanup: () => {
      for (const m of [cA, cB, gA, gB, cBsmall, gBsmall, rgbaA, rgbaB, scene, sceneG, sceneRGBA, blob, blobC, blobSmall, noMask, bigMask, empty]) m.delete();
    },
  };
}

test('ipB-0. leak (address fingerprint, first test of a clean process): every branch of all four new two-image ops (every operation/method, no src2, src2 0x0, A 0x0, size mismatch, gray+gray, colour+gray, RGBA, empty mask, mask > 25%, single-channel B) leaves the malloc fingerprint unchanged', () => {
  const { paths, cleanup } = batchBLeakPaths();
  try {
    fingerprintLeakProbe(cv, 'IP batch 2 chunk B ops (all paths)', () => {
      for (const p of paths) p.run(cv);
    }, { iters: 25, warmup: 10 });
  } finally { cleanup(); }
});

test('ipB-0w. leak (wrapper balance twin of ipB-0): the exact same path table creates and deletes every embind wrapper it makes (RGB/gray temporaries, the resized B, threshold masks, NOT(B), MatVectors/hist/LUT of hist_match, the passthrough clones)', () => {
  const { paths, cleanup } = batchBLeakPaths();
  try {
    assertWrapperBalance(cv, 'IP batch 2 chunk B ops (all paths)', (c) => {
      for (const p of paths) p.run(c);
    }, { minCreated: 120 });
  } finally { cleanup(); }
});

// ===================================================================
// ipB-L1…L5 — leak ต่อ op / ต่อเส้นทาง (P-14 ข้อ 2 + 3 ในคำสั่งเดียว)
// ===================================================================
function leakEach(label, cases, opts = {}) {
  for (const [tag, fn] of cases) {
    assertLeakFree(cv, `${label} ${tag}`, (c) => { const o = fn(c); o.delete(); }, { minCreated: 1, iters: 40, warmup: 10, ...opts });
  }
}

test('ipB-L1. leak: imarith -- all four operations on colour+colour and gray+gray (1-channel fast path), colour+gray, size mismatch (resized B), RGBA, and the no-src2 / src2 0x0 / A 0x0 passthroughs balance every wrapper and grow the heap by 0', () => {
  const cA = noiseRGB(20, 14, 11), cB = noiseRGB(20, 14, 12), gA = noiseGray(20, 14, 13), gB = noiseGray(20, 14, 14), small = noiseRGB(7, 5, 15);
  const rgbaA = toRGBA(cA), empty = new cv.Mat();
  try {
    const cases = [];
    for (const operation of IMARITH_MODES) {
      cases.push([`${operation} colour`, (c) => OPS.imarith.apply(c, cA, { operation }, { src2: cB })]);
      cases.push([`${operation} gray`, (c) => OPS.imarith.apply(c, gA, { operation }, { src2: gB })]);
    }
    cases.push(['colour+gray', (c) => OPS.imarith.apply(c, cA, { operation: 'add' }, { src2: gB })]);
    cases.push(['size mismatch', (c) => OPS.imarith.apply(c, gA, { operation: 'add' }, { src2: small })]);
    cases.push(['rgba', (c) => OPS.imarith.apply(c, rgbaA, { operation: 'add' }, { src2: cB })]);
    cases.push(['no src2', (c) => OPS.imarith.apply(c, cA, {}, {})]);
    cases.push(['src2 0x0', (c) => OPS.imarith.apply(c, cA, {}, { src2: empty })]);
    cases.push(['A 0x0', (c) => OPS.imarith.apply(c, empty, {}, { src2: cB })]);
    leakEach('imarith', cases);
  } finally { for (const m of [cA, cB, gA, gB, small, rgbaA, empty]) m.delete(); }
});

test('ipB-L2. leak: logical_ops -- all four operations (a_and_not_b allocates an extra NOT(B) Mat), colour inputs, size mismatch, and the passthroughs balance every wrapper and grow the heap by 0', () => {
  const cA = noiseRGB(20, 14, 21), cB = noiseRGB(20, 14, 22), gA = noiseGray(20, 14, 23), gB = noiseGray(20, 14, 24), small = noiseGray(9, 4, 25);
  const empty = new cv.Mat();
  try {
    const cases = [];
    for (const operation of LOGICAL_MODES) {
      cases.push([`${operation} gray`, (c) => OPS.logical_ops.apply(c, gA, { operation }, { src2: gB })]);
      cases.push([`${operation} colour`, (c) => OPS.logical_ops.apply(c, cA, { operation }, { src2: cB })]);
    }
    cases.push(['size mismatch', (c) => OPS.logical_ops.apply(c, cA, { operation: 'and' }, { src2: small })]);
    cases.push(['no src2', (c) => OPS.logical_ops.apply(c, gA, {}, undefined)]);
    cases.push(['src2 0x0', (c) => OPS.logical_ops.apply(c, gA, {}, { src2: empty })]);
    cases.push(['A 0x0', (c) => OPS.logical_ops.apply(c, empty, {}, { src2: gB })]);
    leakEach('logical_ops', cases);
  } finally { for (const m of [cA, cB, gA, gB, small, empty]) m.delete(); }
});

test('ipB-L3. leak: inpaint -- TELEA and NS on colour and gray, colour mask B, RGBA A (RGBA->RGB temp), size mismatch, the empty-mask and >25% guards (both return early from inside the try), and the passthroughs balance every wrapper and grow the heap by 0', () => {
  const scene = sceneRGB(20, 14), sceneG = gray(20, 14, (x, y) => (50 + 5 * x + y) & 255), sceneRGBA = toRGBA(scene);
  const blob = blobMask(20, 14, 7, 5, 11, 8), small = blobMask(10, 7, 3, 2, 6, 4);
  const blobC = new cv.Mat(); cv.cvtColor(blob, blobC, cv.COLOR_GRAY2RGB);
  const noMask = firstN(20, 14, 0), bigMask = firstN(20, 14, 100), empty = new cv.Mat();
  try {
    const cases = [];
    for (const method of INPAINT_METHODS) {
      cases.push([`${method} colour`, (c) => OPS.inpaint.apply(c, scene, { method, radius: 3 }, { src2: blob })]);
      cases.push([`${method} gray`, (c) => OPS.inpaint.apply(c, sceneG, { method, radius: 7 }, { src2: blob })]);
    }
    cases.push(['colour mask', (c) => OPS.inpaint.apply(c, scene, {}, { src2: blobC })]);
    cases.push(['rgba A', (c) => OPS.inpaint.apply(c, sceneRGBA, {}, { src2: blob })]);
    cases.push(['size mismatch', (c) => OPS.inpaint.apply(c, scene, {}, { src2: small })]);
    cases.push(['mask empty', (c) => OPS.inpaint.apply(c, scene, {}, { src2: noMask })]);
    cases.push(['mask > 25%', (c) => OPS.inpaint.apply(c, scene, {}, { src2: bigMask })]);
    cases.push(['no src2', (c) => OPS.inpaint.apply(c, scene, {}, {})]);
    cases.push(['src2 0x0', (c) => OPS.inpaint.apply(c, scene, {}, { src2: empty })]);
    cases.push(['A 0x0', (c) => OPS.inpaint.apply(c, empty, {}, { src2: blob })]);
    leakEach('inpaint', cases);
  } finally { for (const m of [scene, sceneG, sceneRGBA, blob, small, blobC, noMask, bigMask, empty]) m.delete(); }
});

test('ipB-L4. leak: hist_match -- gray+gray, colour+colour (channel by channel), colour A + single-channel B, gray A + colour B, RGBA A and RGBA B, B of a different size, and the passthroughs balance every wrapper (2 MatVectors, hist, mask, LUT, conversions) and grow the heap by 0', () => {
  const cA = noiseRGB(20, 14, 31), cB = noiseRGB(13, 9, 32), gA = noiseGray(20, 14, 33), gB = noiseGray(20, 14, 34);
  const rgbaA = toRGBA(cA), rgbaB = toRGBA(cB), empty = new cv.Mat();
  try {
    leakEach('hist_match', [
      ['gray+gray', (c) => OPS.hist_match.apply(c, gA, {}, { src2: gB })],
      ['colour+colour', (c) => OPS.hist_match.apply(c, cA, {}, { src2: cB })],
      ['colour A + gray B', (c) => OPS.hist_match.apply(c, cA, {}, { src2: gB })],
      ['gray A + colour B', (c) => OPS.hist_match.apply(c, gA, {}, { src2: cB })],
      ['rgba A + rgba B', (c) => OPS.hist_match.apply(c, rgbaA, {}, { src2: rgbaB })],
      ['colour A + rgba B', (c) => OPS.hist_match.apply(c, cA, {}, { src2: rgbaB })],
      ['no src2', (c) => OPS.hist_match.apply(c, cA, {}, {})],
      ['src2 0x0', (c) => OPS.hist_match.apply(c, cA, {}, { src2: empty })],
      ['A 0x0', (c) => OPS.hist_match.apply(c, empty, {}, { src2: cB })],
    ]);
  } finally { for (const m of [cA, cB, gA, gB, rgbaA, rgbaB, empty]) m.delete(); }
});

test('ipB-L5. leak on the exception path: when the core cv call of each op throws (cv.add/subtract/multiply/divide, the 2nd cv.threshold + bitwise ops, cv.inpaint, cv.LUT, and the resize inside matchSize), every wrapper allocated so far is still released and the exception propagates', () => {
  const cA = noiseRGB(16, 12, 41), cB = noiseRGB(16, 12, 42), gA = noiseGray(16, 12, 43), gB = noiseGray(16, 12, 44), small = noiseRGB(5, 4, 45);
  const scene = sceneRGB(16, 12), sceneRGBA = toRGBA(scene), blob = blobMask(16, 12, 5, 4, 9, 7), rgbaA = toRGBA(cA), rgbaB = toRGBA(cB);
  const throwing = [
    ['imarith cvtColor inside rgbIfNeeded', 'cvtColor', (c) => OPS.imarith.apply(c, cA, { operation: 'add' }, { src2: gB })],
    ['imarith 2nd cvtColor (A converted, B throws)', ['cvtColor', 2], (c) => OPS.imarith.apply(c, rgbaA, { operation: 'add' }, { src2: gB })],
    ['inpaint RGBA->RGB cvtColor', 'cvtColor', (c) => OPS.inpaint.apply(c, sceneRGBA, {}, { src2: blob })],
    ['hist_match RGBA A cvtColor', 'cvtColor', (c) => OPS.hist_match.apply(c, rgbaA, {}, { src2: cB })],
    ['hist_match RGBA B cvtColor (rgbIfNeeded)', 'cvtColor', (c) => OPS.hist_match.apply(c, cA, {}, { src2: rgbaB })],
    ['imarith add', 'add', (c) => OPS.imarith.apply(c, cA, { operation: 'add' }, { src2: gB })],
    ['imarith subtract', 'subtract', (c) => OPS.imarith.apply(c, gA, { operation: 'subtract' }, { src2: gB })],
    ['imarith multiply', 'multiply', (c) => OPS.imarith.apply(c, cA, { operation: 'multiply' }, { src2: small })],
    ['imarith divide', 'divide', (c) => OPS.imarith.apply(c, cA, { operation: 'divide' }, { src2: cB })],
    ['imarith resize', 'resize', (c) => OPS.imarith.apply(c, cA, { operation: 'add' }, { src2: small })],
    ['logical 2nd threshold', ['threshold', 2], (c) => OPS.logical_ops.apply(c, cA, { operation: 'and' }, { src2: small })],
    ['logical a_and_not_b bitwise_and', 'bitwise_and', (c) => OPS.logical_ops.apply(c, cA, { operation: 'a_and_not_b' }, { src2: cB })],
    ['logical countNonZero', 'countNonZero', (c) => OPS.logical_ops.apply(c, gA, { operation: 'xor' }, { src2: cB })],
    ['inpaint', 'inpaint', (c) => OPS.inpaint.apply(c, sceneRGBA, { method: 'ns' }, { src2: blob })],
    ['inpaint countNonZero', 'countNonZero', (c) => OPS.inpaint.apply(c, scene, {}, { src2: blob })],
    ['hist_match LUT', 'LUT', (c) => OPS.hist_match.apply(c, cA, {}, { src2: gB })],
    ['hist_match 2nd calcHist', ['calcHist', 2], (c) => OPS.hist_match.apply(c, rgbaA, {}, { src2: cB })],
    ['hist_match matFromArray', 'matFromArray', (c) => OPS.hist_match.apply(c, gA, {}, { src2: cB })],
  ];
  try {
    for (const [label, method, fn] of throwing) {
      const [name, onCall] = Array.isArray(method) ? method : [method, null];
      assertWrapperBalance(cv, `throw in ${label}`, (c) => {
        const o = fn(makeThrowingCv(c, name, onCall ? { onCall } : {}));
        o.delete();
      }, { expectThrow: /injected test failure/, minCreated: 1 });
    }
  } finally { for (const m of [cA, cB, gA, gB, small, scene, sceneRGBA, blob, rgbaA, rgbaB]) m.delete(); }
});

// ===================================================================
// ipB-1…ipB-3 — imarith
// ===================================================================
// เวกเตอร์ที่ qa วัด (สเปก §4.1) + เวกเตอร์เสริมที่แยก scale 1/255 กับ 1/256 และ 255 กับ 256 ได้จริง
const QA_A = [0, 100, 200, 250, 10, 255];
const QA_B = [0, 100, 100, 10, 50, 0];
const QA_EXPECT = {
  add: [0, 200, 255, 255, 60, 255],
  subtract: [0, 0, 100, 240, 0, 255],
  multiply: [0, 39, 78, 10, 2, 0],
  divide: [0, 255, 255, 255, 51, 0],
};
const SCALE_A = [255, 200, 128, 7];
const SCALE_B = [255, 255, 255, 255];
const SCALE_EXPECT = { multiply: [255, 200, 128, 7], divide: [255, 200, 128, 7] };

test('ipB-1. imarith: the qa vectors (spec 4.1) give exactly add [0,200,255,255,60,255] / subtract [0,0,100,240,0,255] / multiply [0,39,78,10,2,0] / divide [0,255,255,255,51,0] on the 1-channel fast path, and the very same numbers on every channel of the colour path (A colour, B colour)', () => {
  const a = row(QA_A), b = row(QA_B);
  const aC = new cv.Mat(), bC = new cv.Mat();
  cv.cvtColor(a, aC, cv.COLOR_GRAY2RGB); cv.cvtColor(b, bC, cv.COLOR_GRAY2RGB);
  try {
    for (const operation of IMARITH_MODES) {
      const g = run('imarith', a, b, { operation });
      try {
        assert.equal(g.out.channels(), 1, `${operation}: gray+gray stays 1 channel`);
        assert.deepEqual(bytes(g.out), QA_EXPECT[operation], `${operation} gray`);
        assert.match(g.ctx.info, /\(ขาวดำ\)/);
      } finally { g.out.delete(); }
      const c = run('imarith', aC, bC, { operation });
      try {
        assert.equal(c.out.channels(), 3);
        const d = bytes(c.out);
        for (let ch = 0; ch < 3; ch++) assert.deepEqual(d.filter((_, i) => i % 3 === ch), QA_EXPECT[operation], `${operation} colour channel ${ch}`);
        assert.doesNotMatch(c.ctx.info, /\(ขาวดำ\)/);
      } finally { c.out.delete(); }
    }
  } finally { a.delete(); b.delete(); aC.delete(); bC.delete(); }
});

test('ipB-1b. imarith: the scale factors are exactly 1/255 (multiply) and 255 (divide) -- with B = 255 everywhere, multiply and divide must both give back A unchanged ([255,200,128,7]); 1/256 would give [254,199,128,7] and x256 would give [255,201,128,7]', () => {
  const a = row(SCALE_A), b = row(SCALE_B);
  try {
    for (const operation of ['multiply', 'divide']) {
      const r = run('imarith', a, b, { operation });
      try { assert.deepEqual(bytes(r.out), SCALE_EXPECT[operation], operation); } finally { r.out.delete(); }
    }
  } finally { a.delete(); b.delete(); }
});

test('ipB-2. imarith: mixing colour and gray converts the gray image with GRAY2RGB first (3-channel result); a B of another size is resized to A (INTER_LINEAR, info says so); unknown / prototype-key operations ("constructor", "__proto__", undefined) fall back to add; RGBA inputs are reduced to RGB', () => {
  const cA = rgb(4, 3, (x, y) => [10 * x, 20 * y, 200]);
  const gB = gray(4, 3, (x) => 30 * x);
  const smallB = gray(2, 2, () => 50);
  const rgbaA = toRGBA(cA);
  try {
    const r = run('imarith', cA, gB, { operation: 'add' });
    try {
      assert.equal(r.out.channels(), 3);
      const d = bytes(r.out);
      for (let y = 0; y < 3; y++) for (let x = 0; x < 4; x++) {
        const i = (y * 4 + x) * 3;
        assert.deepEqual([d[i], d[i + 1], d[i + 2]], [10 * x + 30 * x, 20 * y + 30 * x, Math.min(255, 200 + 30 * x)]);
      }
      assert.doesNotMatch(r.ctx.info, /ปรับขนาด/);
    } finally { r.out.delete(); }

    const s = run('imarith', cA, smallB, { operation: 'subtract' });
    try {
      assert.equal(s.out.cols, 4); assert.equal(s.out.rows, 3);
      assert.match(s.ctx.info, /ปรับขนาด B ให้เท่า A ก่อนคำนวณ/);
      const d = bytes(s.out);
      assert.equal(d[2], 150, 'B resized to a constant 50 everywhere -> 200 - 50');
    } finally { s.out.delete(); }

    for (const operation of ['constructor', '__proto__', undefined, 'nope']) {
      const u = run('imarith', cA, gB, { operation });
      const ref = run('imarith', cA, gB, { operation: 'add' });
      try { assert.deepEqual(bytes(u.out), bytes(ref.out), `operation=${String(operation)} must behave as add`); } finally { u.out.delete(); ref.out.delete(); }
    }

    const q = run('imarith', rgbaA, gB, { operation: 'add' });
    const ref = run('imarith', cA, gB, { operation: 'add' });
    try {
      assert.equal(q.out.channels(), 3);
      assert.deepEqual(bytes(q.out), bytes(ref.out));
    } finally { q.out.delete(); ref.out.delete(); }
  } finally { cA.delete(); gB.delete(); smallB.delete(); rgbaA.delete(); }
});

// ===================================================================
// ipB-3 — logical_ops
// ===================================================================
test('ipB-3. logical_ops: the full 2x2 truth table for AND / OR / XOR / A AND NOT B (A=[0,0,255,255], B=[0,255,0,255]), with 128 counted as white and 127 as black on BOTH inputs (threshold 127, strict >), result 1 channel 0/255 and an info line with the white-pixel count', () => {
  const TABLE = {
    and: [0, 0, 0, 255],
    or: [0, 255, 255, 255],
    xor: [0, 255, 255, 0],
    a_and_not_b: [0, 0, 255, 0],
  };
  const fixtures = [
    ['0/255', [0, 0, 255, 255], [0, 255, 0, 255]],
    ['127/128', [127, 127, 128, 128], [127, 128, 127, 128]],
    ['mixed', [0, 127, 128, 255], [127, 200, 0, 128]],
  ];
  for (const [tag, av, bv] of fixtures) {
    const a = row(av), b = row(bv);
    try {
      for (const operation of LOGICAL_MODES) {
        const r = run('logical_ops', a, b, { operation });
        try {
          assert.equal(r.out.channels(), 1);
          assert.deepEqual(bytes(r.out), TABLE[operation], `${operation} on ${tag}`);
          const white = TABLE[operation].filter((v) => v === 255).length;
          assert.match(r.ctx.info, new RegExp(`· ขาว ${white} px$`), `${operation} info`);
        } finally { r.out.delete(); }
      }
    } finally { a.delete(); b.delete(); }
  }
});

test('ipB-3b. logical_ops: colour A is converted RGB->GRAY then cut at 127 (pure red 255,0,0 -> gray 76 -> black; pure green 0,255,0 -> 150 -> white); a colour B goes gray -> resize -> threshold (order of op mask); unknown/prototype-key operation falls back to AND; info names the operation', () => {
  const a = rgb(2, 1, (x) => (x === 0 ? [255, 0, 0] : [0, 255, 0]));
  const b = rgb(2, 1, () => [255, 255, 255]);
  const bSmall = rgb(1, 1, () => [255, 255, 255]);
  try {
    const r = run('logical_ops', a, b, { operation: 'and' });
    try { assert.deepEqual(bytes(r.out), [0, 255]); assert.match(r.ctx.info, /^A AND B — ขาวเฉพาะที่ขาวทั้งคู่/); } finally { r.out.delete(); }
    const s = run('logical_ops', a, bSmall, { operation: 'or' });
    try { assert.deepEqual(bytes(s.out), [255, 255]); assert.match(s.ctx.info, /^A OR B .*ปรับขนาด B ให้เท่า A/); } finally { s.out.delete(); }
    for (const operation of ['constructor', 'toString', undefined]) {
      const u = run('logical_ops', a, b, { operation });
      try { assert.match(u.ctx.info, /^A AND B/, `operation=${String(operation)}`); } finally { u.out.delete(); }
    }
    for (const [operation, re] of [['xor', /^A XOR B/], ['a_and_not_b', /^A AND NOT B — หัก B ออกจาก A/]]) {
      const u = run('logical_ops', a, b, { operation });
      try { assert.match(u.ctx.info, re); } finally { u.out.delete(); }
    }
  } finally { a.delete(); b.delete(); bSmall.delete(); }
});

test('ipB-3c. logical_ops + Color InRange (debugger 2026-10-03): the desc example "red OR yellow" works when both InRange boxes output "Mask ขาวดำ" (2 white px of [red, yellow, blue, gray]); with the InRange DEFAULT output "extract" the saturated red (gray ~81 < 127) is lost (1 px) -- the info must then warn that colour input was cut at 127 and tell the user to switch InRange to Mask; gray (mask) inputs carry no warning', () => {
  const img = rgb(4, 1, (x) => [[220, 30, 30], [230, 220, 30], [30, 30, 220], [128, 128, 128]][x]);
  const red = { colorSpace: 'hsv', hMin: 170, hMax: 10, sMin: 60, vMin: 60 };
  const yellow = { colorSpace: 'hsv', hMin: 20, hMax: 35, sMin: 60, vMin: 60 };
  const mats = [];
  const inr = (pp, output) => { const m = OPS.color_inrange.apply(cv, img, { ...pp, output }, {}); mats.push(m); return m; };
  try {
    const rM = inr(red, 'mask'), yM = inr(yellow, 'mask');
    assert.equal(rM.channels(), 1);
    const ok = run('logical_ops', rM, yM, { operation: 'or' });
    try {
      assert.deepEqual(bytes(ok.out), [255, 255, 0, 0]);
      assert.equal(ok.ctx.info, 'A OR B — ขาวถ้าขาวอย่างน้อยหนึ่งภาพ · ขาว 2 px', 'mask inputs: no colour warning');
    } finally { ok.out.delete(); }
    const rE = inr(red, 'extract'), yE = inr(yellow, 'extract');
    assert.equal(rE.channels(), 3, "InRange's default output is a colour image");
    const bad = run('logical_ops', rE, yE, { operation: 'or' });
    try {
      assert.deepEqual(bytes(bad.out), [0, 255, 0, 0], 'the pitfall is real: red (gray ~81) is cut away');
      assert.match(bad.ctx.info, /· ขาว 1 px · ภาพสีถูกแปลงเป็นขาวดำแล้วตัดที่ 127 — ถ้ามาจาก Color InRange ให้ตั้ง 'แสดงผลเป็น' = Mask ขาวดำ$/);
    } finally { bad.out.delete(); }
    assert.match(OPS.logical_ops.desc, /ตั้ง 'แสดงผลเป็น' = Mask ขาวดำ/);
    assert.match(OPS.inpaint.desc, /ตั้ง 'แสดงผลเป็น' = Mask ขาวดำ/);
    // inpaint: a colour mask B carries the same warning on every info path (empty-mask path included); a gray mask does not
    const scene = sceneRGB(4, 1);
    mats.push(scene);
    const e = run('inpaint', scene, rE, {});
    try { assert.match(e.ctx.info, /^mask ว่าง .*ตัดที่ 127 — ถ้ามาจาก Color InRange/); } finally { e.out.delete(); }
    const g = run('inpaint', scene, rM, {});
    try { assert.doesNotMatch(g.ctx.info, /ตัดที่ 127/); } finally { g.out.delete(); }
  } finally { img.delete(); for (const m of mats) m.delete(); }
});

// ===================================================================
// ipB-4 — inpaint
// ===================================================================
test('ipB-4. inpaint: an empty mask returns A unchanged byte-for-byte with the "mask ว่าง" message; pixels of B at 127 are NOT part of the mask but 128 are (threshold 127)', () => {
  const scene = sceneRGB(20, 10);
  const none = firstN(20, 10, 0);
  const at127 = firstN(20, 10, 30, 127);
  const at128 = firstN(20, 10, 30, 128);
  try {
    for (const B of [none, at127]) {
      const r = run('inpaint', scene, B, {});
      try {
        assert.deepEqual(bytes(r.out), bytes(scene));
        assert.match(r.ctx.info, /^mask ว่าง ไม่มีบริเวณให้ซ่อม/);
      } finally { r.out.delete(); }
    }
    const r = run('inpaint', scene, at128, {});
    try { assert.match(r.ctx.info, /^ซ่อม 30 px \(15\.0% ของภาพ\) ด้วย Telea r=3$/); } finally { r.out.delete(); }
  } finally { scene.delete(); none.delete(); at127.delete(); at128.delete(); }
});

test('ipB-4b. inpaint: the 25% guard compares integers (nz*100 > 25*total) -- on a 20x10 image (200 px) a mask of 52 px (26%) is refused (A returned unchanged + the Threshold hint), 50 px (exactly 25%) and 48 px (24%) are repaired', () => {
  const scene = sceneRGB(20, 10);
  try {
    for (const [n, repaired] of [[52, false], [51, false], [50, true], [48, true]]) {
      const B = firstN(20, 10, n);
      const r = run('inpaint', scene, B, {});
      try {
        if (repaired) {
          assert.match(r.ctx.info, new RegExp(`^ซ่อม ${n} px`), `${n} px must be repaired`);
          assert.notDeepEqual(bytes(r.out), bytes(scene));
        } else {
          assert.match(r.ctx.info, /^ไม่ซ่อม: mask ครอบ 2[56](\.\d)?% ของภาพ \(เกิน 25%\) — Inpaint ใช้ซ่อมรอยเล็ก ๆ ให้ใส่ Threshold ในสาย B/, `${n} px must be refused`);
          assert.deepEqual(bytes(r.out), bytes(scene));
        }
      } finally { r.out.delete(); B.delete(); }
    }
  } finally { scene.delete(); }
});

test('ipB-4c. inpaint: a bright blob drawn on a flat background is repaired -- every pixel inside the mask changes (and ends up close to the background), every pixel outside the mask is untouched; TELEA and NS give exactly cv.inpaint(..., INPAINT_TELEA / INPAINT_NS) and differ from each other on this fixture', () => {
  const w = 30, h = 20;
  const bgFn = (x, y) => [80 + x, 120 + y, 60 + ((x + y) & 7)];
  const scene = rgb(w, h, (x, y) => (x >= 12 && x < 16 && y >= 8 && y < 11 ? [255, 255, 255] : bgFn(x, y)));
  const B = blobMask(w, h, 12, 8, 16, 11);
  try {
    const outs = {};
    for (const method of INPAINT_METHODS) {
      const r = run('inpaint', scene, B, { method, radius: 3 });
      outs[method] = r.out;
      assert.match(r.ctx.info, new RegExp(`ด้วย ${method === 'ns' ? 'Navier-Stokes' : 'Telea'} r=3$`));
      const d = bytes(r.out), s = bytes(scene);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const inside = x >= 12 && x < 16 && y >= 8 && y < 11;
        for (let c = 0; c < 3; c++) {
          const i = (y * w + x) * 3 + c;
          if (inside) {
            assert.notEqual(d[i], s[i], `${method}: masked pixel (${x},${y}) ch${c} must change`);
            assert.ok(Math.abs(d[i] - bgFn(x, y)[c]) <= 12, `${method}: (${x},${y}) ch${c}=${d[i]} should be near the background ${bgFn(x, y)[c]}`);
          } else {
            assert.equal(d[i], s[i], `${method}: pixel (${x},${y}) outside the mask must not change`);
          }
        }
      }
      const direct = new cv.Mat();
      try {
        cv.inpaint(scene, B, direct, 3, method === 'ns' ? cv.INPAINT_NS : cv.INPAINT_TELEA);
        assert.deepEqual(d, bytes(direct), `${method} must be exactly cv.inpaint with the matching flag`);
      } finally { direct.delete(); }
    }
    assert.notDeepEqual(bytes(outs.telea), bytes(outs.ns), 'TELEA and NS must differ on this fixture (otherwise the flag check above is vacuous)');
    outs.telea.delete(); outs.ns.delete();
  } finally { scene.delete(); B.delete(); }
});

test('ipB-4d. inpaint: radius is clamped to [1, 7] with intIn (P-13) -- 99 behaves exactly like 7, 0 and -5 like 1, "4.6" like 5; an RGBA A is reduced to RGB (3-channel result equal to inpainting the RGB image); a colour B is converted to gray first; a smaller B is resized', () => {
  const scene = sceneRGB(26, 18, 9);
  const B = blobMask(26, 18, 8, 6, 14, 10);
  const Bc = new cv.Mat(); cv.cvtColor(B, Bc, cv.COLOR_GRAY2RGB);
  const Bsmall = blobMask(13, 9, 4, 3, 7, 5);
  const rgba = toRGBA(scene);
  const outOf = (A, M, params) => { const r = run('inpaint', A, M, params); const d = bytes(r.out); const info = r.ctx.info; const ch = r.out.channels(); r.out.delete(); return { d, info, ch }; };
  try {
    for (const method of INPAINT_METHODS) {
      const r7 = outOf(scene, B, { method, radius: 7 });
      const r1 = outOf(scene, B, { method, radius: 1 });
      const r5 = outOf(scene, B, { method, radius: 5 });
      assert.notDeepEqual(r7.d, r1.d, 'radius 7 and 1 must differ (else the clamp checks are vacuous)');
      assert.deepEqual(outOf(scene, B, { method, radius: 99 }).d, r7.d, 'radius 99 -> 7');
      assert.match(outOf(scene, B, { method, radius: 99 }).info, /r=7$/);
      assert.deepEqual(outOf(scene, B, { method, radius: 0 }).d, r1.d, 'radius 0 -> 1');
      assert.deepEqual(outOf(scene, B, { method, radius: -5 }).d, r1.d, 'radius -5 -> 1');
      assert.deepEqual(outOf(scene, B, { method, radius: '4.6' }).d, r5.d, 'radius "4.6" -> 5');
      assert.deepEqual(outOf(scene, B, { method }).d, outOf(scene, B, { method, radius: 3 }).d, 'missing radius -> default 3');
      const q = outOf(rgba, B, { method });
      assert.equal(q.ch, 3);
      assert.deepEqual(q.d, outOf(scene, B, { method }).d, 'RGBA A == RGB A');
    }
    assert.deepEqual(outOf(scene, Bc, {}).d, outOf(scene, B, {}).d, 'colour mask B == gray mask B');
    const s = outOf(scene, Bsmall, {});
    assert.match(s.info, /ปรับขนาด B ให้เท่า A ก่อนคำนวณ/);
    assert.match(outOf(scene, B, { method: 'ns' }).info, /Navier-Stokes/);
    assert.match(outOf(scene, B, { method: 'whatever' }).info, /Telea/, 'unknown method -> Telea (the default)');
  } finally { scene.delete(); B.delete(); Bc.delete(); Bsmall.delete(); rgba.delete(); }
});

// ===================================================================
// ipB-5 — hist_match
// ===================================================================
function float32TrapA() { return gray(60, 61, (x, y) => (y * 60 + x < 2173 ? 0 : 255)); }
function float32TrapB() { return gray(113, 89, (x, y) => (y * 113 + x < 5971 ? 10 : 200)); }

test('ipB-5. hist_match: B = A gives the identity (gray and colour); a ramp matched to a constant image of value v gives v at every pixel; the info names the channel pairing', () => {
  const g = noiseGray(40, 30, 51), c = noiseRGB(40, 30, 52);
  const ramp = gray(256, 2, (x) => x);
  try {
    for (const m of [g, c]) {
      const r = run('hist_match', m, m, {});
      try { assert.deepEqual(bytes(r.out), bytes(m), 'B = A must be the identity'); } finally { r.out.delete(); }
    }
    for (const v of [0, 37, 128, 255]) {
      const B = gray(5, 5, () => v);
      const r = run('hist_match', ramp, B, {});
      try {
        assert.ok(bytes(r.out).every((x) => x === v), `ramp vs constant ${v}: every pixel must become ${v}`);
        assert.equal(r.ctx.info, 'ปรับ histogram ของ A ให้เหมือน B (ขาวดำ)');
      } finally { r.out.delete(); B.delete(); }
    }
  } finally { g.delete(); c.delete(); ramp.delete(); }
});

test('ipB-5b. hist_match: the mapping is monotonic (never decreasing) for a full 0..255 ramp against several random references, and a B of a different size but the SAME normalised histogram (B tiled 2x2 / 3x1) gives exactly the same result (B is never resized; info never mentions resizing)', () => {
  const ramp = gray(256, 3, (x) => x);
  try {
    for (const seed of [61, 62, 63]) {
      const B = noiseGray(17, 11, seed);
      const r = run('hist_match', ramp, B, {});
      try {
        const d = bytes(r.out);
        for (let i = 1; i < 256; i++) assert.ok(d[i] >= d[i - 1], `seed ${seed}: LUT must not decrease at ${i} (${d[i - 1]} -> ${d[i]})`);
      } finally { r.out.delete(); B.delete(); }
    }
    const A = noiseRGB(30, 20, 64), B = noiseRGB(9, 7, 65);
    const tile = (m, nx, ny) => { const out = new cv.Mat(); const v = new cv.MatVector(); for (let i = 0; i < nx; i++) v.push_back(m); const h = new cv.Mat(); cv.hconcat(v, h); v.delete(); const v2 = new cv.MatVector(); for (let i = 0; i < ny; i++) v2.push_back(h); cv.vconcat(v2, out); v2.delete(); h.delete(); return out; };
    const B22 = tile(B, 2, 2), B31 = tile(B, 3, 1);
    try {
      const base = run('hist_match', A, B, {});
      for (const Bx of [B22, B31]) {
        const r = run('hist_match', A, Bx, {});
        try {
          assert.deepEqual(bytes(r.out), bytes(base.out), `B ${Bx.cols}x${Bx.rows} vs ${B.cols}x${B.rows}`);
          assert.doesNotMatch(r.ctx.info, /ปรับขนาด/);
        } finally { r.out.delete(); }
      }
      assert.equal(base.ctx.info, 'ปรับ histogram ของ A ให้เหมือน B (3 ช่อง แยกช่อง)');
      base.out.delete();
    } finally { A.delete(); B.delete(); B22.delete(); B31.delete(); }
  } finally { ramp.delete(); }
});

test('ipB-5c. hist_match channel pairing: colour A + colour B is matched channel by channel (B = constant (50,100,200) -> output exactly (50,100,200) everywhere); colour A + single-channel B uses that one histogram for all 3 channels; gray A + colour B converts B RGB->GRAY first (constant (255,0,0) -> 76); RGBA inputs are reduced to RGB', () => {
  const A = noiseRGB(16, 12, 71);
  const Bc = rgb(6, 6, () => [50, 100, 200]);
  const Bg = gray(6, 6, () => 77);
  const gA = noiseGray(16, 12, 72);
  const red = rgb(6, 6, () => [255, 0, 0]);
  const rgbaA = toRGBA(A), rgbaB = toRGBA(Bc);
  try {
    const check = (A_, B_, expectPx, infoRe, ch) => {
      const r = run('hist_match', A_, B_, {});
      try {
        assert.equal(r.out.channels(), ch);
        const d = bytes(r.out);
        for (let i = 0; i < d.length; i++) assert.equal(d[i], expectPx[i % ch], `byte ${i}`);
        assert.match(r.ctx.info, infoRe);
      } finally { r.out.delete(); }
    };
    check(A, Bc, [50, 100, 200], /\(3 ช่อง แยกช่อง\)$/, 3);
    check(A, Bg, [77, 77, 77], /\(3 ช่อง ใช้ histogram ขาวดำของ B กับทุกช่อง\)$/, 3);
    check(gA, red, [76], /\(ขาวดำ\)$/, 1);
    check(rgbaA, rgbaB, [50, 100, 200], /\(3 ช่อง แยกช่อง\)$/, 3);
    check(A, rgbaB, [50, 100, 200], /\(3 ช่อง แยกช่อง\)$/, 3);
  } finally { for (const m of [A, Bc, Bg, gA, red, rgbaA, rgbaB]) m.delete(); }
});

test('ipB-5d. hist_match: the CDF must be compared in float64 with ">=" -- A (60x61: 2173 px of 0, rest 255) has cdf_A[0] = 2173/3660 = 0.5937158470 and B (113x89: 5971 px of 10, rest 200) has cdf_B[10..199] = 5971/10057 = 0.5937158198 (smaller by 2.7e-8, the SAME float32 value) -> float64 maps 0 -> 200 (every output pixel = 200) while a float32 CDF would map 0 -> 10', () => {
  const A = float32TrapA(), B = float32TrapB();
  try {
    assert.ok(5971 / 10057 < 2173 / 3660, 'fixture: float64 strictly smaller');
    assert.equal(Math.fround(5971 / 10057), Math.fround(2173 / 3660), 'fixture: equal in float32');
    const r = run('hist_match', A, B, {});
    try { assert.ok(bytes(r.out).every((v) => v === 200), 'every pixel must map to 200 (float64 CDF)'); } finally { r.out.delete(); }
  } finally { A.delete(); B.delete(); }
});

// ===================================================================
// ipB-6 — โครงร่วม: passthroughNoSrc2 / ภาพว่าง / ctx.src2 เป็นของยืม
// ===================================================================
test('ipB-6. shared skeleton (all four ops): no src2 -> an exact copy of A (a NEW Mat, not src itself) with exactly the same message as imabsdiff; ctx missing entirely -> copy without throwing; src2 0x0 -> copy + a message that says B is empty (not "not connected"); A 0x0 -> "ภาพว่าง"; ctx.src2 is borrowed and never deleted', () => {
  const A = noiseRGB(12, 9, 81), B = noiseRGB(12, 9, 82), empty = new cv.Mat(), emptyA = new cv.Mat();
  try {
    const absCtx = {};
    const absOut = OPS.imabsdiff.apply(cv, A, {}, absCtx); absOut.delete();
    assert.equal(absCtx.info, NO_SRC2_TEXT, 'reference text from imabsdiff');
    for (const id of NEW_OPS) {
      const ctx = {};
      const o = OPS[id].apply(cv, A, {}, ctx);
      try {
        assert.notEqual(o, A, `${id}: must return a new Mat`);
        assert.deepEqual(bytes(o), bytes(A), `${id}: copy of A`);
        assert.equal(ctx.info, absCtx.info, `${id}: same message as imabsdiff`);
      } finally { o.delete(); }
      const o2 = OPS[id].apply(cv, A, {}, undefined);
      try { assert.deepEqual(bytes(o2), bytes(A)); } finally { o2.delete(); }
      const ctx3 = { src2: empty };
      const o3 = OPS[id].apply(cv, A, {}, ctx3);
      try {
        assert.deepEqual(bytes(o3), bytes(A));
        assert.match(ctx3.info, /^ภาพที่ต่อเข้า 'เข้า B' ว่าง \(0×0 px\)/, `${id}: empty B message`);
      } finally { o3.delete(); }
      const ctx4 = { src2: B };
      const o4 = OPS[id].apply(cv, emptyA, {}, ctx4);
      try {
        assert.equal(o4.rows, 0);
        assert.equal(ctx4.info, 'ภาพว่าง');
      } finally { o4.delete(); }
      const ctx5 = { src2: B };
      const o5 = OPS[id].apply(cv, A, {}, ctx5);
      o5.delete();
      assert.equal(B.isDeleted(), false, `${id}: ctx.src2 must not be deleted by the op`);
      assert.equal(A.isDeleted(), false, `${id}: src must not be deleted by the op`);
    }
  } finally { A.delete(); B.delete(); empty.delete(); emptyA.delete(); }
});

// ===================================================================
// ipB-7…ipB-10 — static: registry / ui-meta / palette / codegen
// ===================================================================
test('ipB-7. (static structure) the 4 ops exist in OPS (the OPS total is NOT pinned -- later batches add more), FILTERS is still 26, image_arithmetic keeps its 3 original modes first and appends the 4 new ones in order, the select keys are "operation"/"method"/"radius" (never "mode" -- that is mergeGroup\'s MODE_KEY), and defaultParams() matches the OPS defaults', () => {
  for (const id of NEW_OPS) assert.ok(OPS[id], `OPS.${id} must exist`);
  assert.equal(Object.keys(FILTERS).length, 33, 'FILTERS = 26 + 7 category-level groups (2026-10-07) -- all four still live in image_arithmetic');
  const ia = FILTERS.image_arithmetic;
  assert.equal(ia.inputs, 2);
  const modes = ia.params.find((p) => p.key === 'mode').options.map((o) => o.value);
  // op ใหม่ batch 4 ต่อท้าย — ห้ามสลับลำดับเดิม (IP batch 4A เพิ่ม roi_apply)
  assert.deepEqual(modes.slice(0, 7), ['imabsdiff', 'imfuse', 'mask', 'imarith', 'logical_ops', 'inpaint', 'hist_match']);
  for (const id of NEW_OPS) {
    assert.equal(OPS[id].needs, 'any', `${id}.needs`);
    for (const p of OPS[id].params) assert.notEqual(p.key, 'mode', `${id}: param key "mode" would collide with MODE_KEY`);
  }
  assert.deepEqual(OPS.imarith.params.find((p) => p.key === 'operation').options.map((o) => o.value), IMARITH_MODES);
  assert.deepEqual(OPS.logical_ops.params.find((p) => p.key === 'operation').options.map((o) => o.value), LOGICAL_MODES);
  assert.deepEqual(OPS.inpaint.params.find((p) => p.key === 'method').options.map((o) => o.value), INPAINT_METHODS);
  assert.deepEqual(OPS.hist_match.params, []);
  const dp = defaultParams('image_arithmetic');
  assert.equal(dp.mode, 'imabsdiff');
  assert.equal(dp.imarith__operation, 'add');
  assert.equal(dp.logical_ops__operation, 'and');
  assert.equal(dp.inpaint__method, 'telea');
  assert.equal(dp.inpaint__radius, 3);
  // through the merged node (what the worker actually calls)
  const A = row([200, 100]), B = row([100, 100]);
  try {
    const ctx = { src2: B };
    const o = FILTERS.image_arithmetic.apply(cv, A, { mode: 'imarith', imarith__operation: 'subtract' }, ctx);
    try { assert.deepEqual(bytes(o), [100, 0]); } finally { o.delete(); }
  } finally { A.delete(); B.delete(); }
});

test('ipB-8. (static read) inpaint radius is clamped in registry.js to EXACTLY the UI min/max/default (intIn(p.radius, 3, 1, 7)) and codegen.js clamps with the same numbers (clampNum(p.radius, 3, 1, 7)); the 25% ceiling is the same number on both sides; every numeric param of the 4 ops is in the clamp table', () => {
  const reg = fs.readFileSync(new URL('../src/renderer/filters/registry.js', import.meta.url), 'utf8');
  const gen = fs.readFileSync(new URL('../src/renderer/codegen.js', import.meta.url), 'utf8');
  const spec = OPS.inpaint.params.find((p) => p.key === 'radius');
  const m = reg.match(/intIn\(p\.radius, (-?[\d.]+), (-?[\d.]+), (-?[\d.]+)\)/);
  assert.ok(m, 'registry.js: expected intIn(p.radius, default, min, max)');
  assert.deepEqual([Number(m[1]), Number(m[2]), Number(m[3])], [spec.default, spec.min, spec.max]);
  const g = gen.match(/Math\.round\(clampNum\(p\.radius, (-?[\d.]+), (-?[\d.]+), (-?[\d.]+)\)\)/);
  assert.ok(g, 'codegen.js: expected Math.round(clampNum(p.radius, default, min, max))');
  assert.deepEqual([Number(g[1]), Number(g[2]), Number(g[3])], [spec.default, spec.min, spec.max]);
  assert.equal(spec.max, 7, 'max radius 7 (measured: 640x480 r=7 25% mask <= ~0.29 s, under the 500 ms limit)');
  const pct = reg.match(/const INPAINT_MAX_MASK_PCT = (\d+);/);
  assert.ok(pct && Number(pct[1]) === 25, 'registry.js: INPAINT_MAX_MASK_PCT = 25');
  assert.match(gen, /elif nz \* 100 > 25 \* total:/, 'codegen.js mirrors the same integer comparison');
  assert.match(spec.hint, /640×480 ≈ [\d.]+–[\d.]+ วินาที/, 'hint carries the measured numbers');
  for (const id of NEW_OPS) for (const p of OPS[id].params) {
    if (p.type !== 'number') continue;
    assert.ok(id === 'inpaint' && p.key === 'radius', `${id}.${p.key}: a numeric param with no clamp test`);
  }
});

test('ipB-9. ui-meta.js OP_PRIMARY: card text for every operation/method, clamped radius (pnum), and unknown / prototype-key operations render the default exactly like the op behaves', () => {
  assert.equal(OP_PRIMARY.imarith({ operation: 'add' }), 'A + B');
  assert.equal(OP_PRIMARY.imarith({ operation: 'subtract' }), 'A − B');
  assert.equal(OP_PRIMARY.imarith({ operation: 'multiply' }), 'A × B ÷ 255');
  assert.equal(OP_PRIMARY.imarith({ operation: 'divide' }), 'A ÷ B × 255');
  assert.equal(OP_PRIMARY.imarith({}), 'A + B');
  assert.equal(OP_PRIMARY.imarith({ operation: 'constructor' }), 'A + B');
  assert.equal(OP_PRIMARY.logical_ops({ operation: 'and' }), 'A AND B');
  assert.equal(OP_PRIMARY.logical_ops({ operation: 'or' }), 'A OR B');
  assert.equal(OP_PRIMARY.logical_ops({ operation: 'xor' }), 'A XOR B');
  assert.equal(OP_PRIMARY.logical_ops({ operation: 'a_and_not_b' }), 'A AND NOT B');
  assert.equal(OP_PRIMARY.logical_ops({ operation: '__proto__' }), 'A AND B');
  assert.equal(OP_PRIMARY.inpaint({ method: 'telea', radius: 3 }), 'Telea · r=3');
  assert.equal(OP_PRIMARY.inpaint({ method: 'ns', radius: 99 }), 'Navier-Stokes · r=7');
  assert.equal(OP_PRIMARY.inpaint({ radius: '4.6' }), 'Telea · r=5');
  assert.equal(OP_PRIMARY.inpaint({}), 'Telea · r=3');
  assert.equal(OP_PRIMARY.hist_match({}), 'ปรับ histogram ของ A ตาม B');
});

test('ipB-10. palette-panel.js: the IMAGE ARITHMETIC section lists the 3 original ops then the 4 new ones, each resolves to the image_arithmetic node through OP_TO_FILTER, and the 4 new OP_ICON glyphs are distinct from every other OP_ICON glyph and every GROUP_ICON glyph', () => {
  const sec = TOOL_SECTIONS.find((s) => s.title === 'IMAGE ARITHMETIC');
  // op ใหม่ batch 4 ต่อท้าย — ห้ามสลับลำดับเดิม (IP batch 4A เพิ่ม roi_apply)
  assert.deepEqual(sec.ops.slice(0, 7), ['imabsdiff', 'imfuse', 'mask', 'imarith', 'logical_ops', 'inpaint', 'hist_match']);
  for (const id of NEW_OPS) assert.equal(OP_TO_FILTER[id], 'image_arithmetic');
  for (const id of NEW_OPS) assert.equal(TOOL_SECTIONS.filter((s) => s.ops.includes(id)).length, 1, `${id} must appear in exactly one section`);
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

function pipe2(opId, local = {}, src2Pipeline = [], before = []) {
  const params = { mode: opId };
  for (const [k, v] of Object.entries(local)) params[`${opId}__${k}`] = v;
  return [...before, { id: 'image_arithmetic', params, enabled: true, src2: { pipeline: src2Pipeline } }];
}

test('ipB-11. codegen: all four ops export real two-image Python (process(path_a, path_b), never "WARNING: skipped" / "ข้าม"), use cv2 functions (never numpy + - * / on uint8 images, never round()), embed clamped values, and track the colour/gray state for the next step (imarith -> colour, logical_ops -> gray, inpaint/hist_match -> unchanged)', () => {
  const expectations = [
    ['imarith', { operation: 'add' }, /img = cv2\.add\(img, arith_b\)/],
    ['imarith', { operation: 'subtract' }, /img = cv2\.subtract\(img, arith_b\)/],
    ['imarith', { operation: 'multiply' }, /img = cv2\.multiply\(img, arith_b, scale=1\/255\)/],
    ['imarith', { operation: 'divide' }, /img = cv2\.divide\(img, arith_b, scale=255\)/],
    ['imarith', { operation: 'constructor' }, /img = cv2\.add\(img, arith_b\)/],
    ['logical_ops', { operation: 'and' }, /img = cv2\.bitwise_and\(a_bin, b_bin\)/],
    ['logical_ops', { operation: 'or' }, /img = cv2\.bitwise_or\(a_bin, b_bin\)/],
    ['logical_ops', { operation: 'xor' }, /img = cv2\.bitwise_xor\(a_bin, b_bin\)/],
    ['logical_ops', { operation: 'a_and_not_b' }, /img = cv2\.bitwise_and\(a_bin, cv2\.bitwise_not\(b_bin\)\)/],
    ['logical_ops', { operation: 'toString' }, /img = cv2\.bitwise_and\(a_bin, b_bin\)/],
    ['inpaint', { method: 'telea', radius: 3 }, /cv2\.inpaint\(img, b_bin, 3, cv2\.INPAINT_TELEA\)/],
    ['inpaint', { method: 'ns', radius: 99 }, /cv2\.inpaint\(img, b_bin, 7, cv2\.INPAINT_NS\)/],
    ['inpaint', { radius: 0 }, /cv2\.inpaint\(img, b_bin, 1, cv2\.INPAINT_TELEA\)/],
    ['inpaint', { radius: '4.6' }, /cv2\.inpaint\(img, b_bin, 5, cv2\.INPAINT_TELEA\)/],
    ['hist_match', {}, /np\.searchsorted\(cdf_b, cdf_a, side='left'\)/],
  ];
  for (const [opId, local, re] of expectations) {
    const code = buildPythonCode(pipe2(opId, local));
    const label = `${opId} ${JSON.stringify(local)}`;
    assert.match(code, /def process\(path_a, path_b\):/, label);
    assert.doesNotMatch(code, /WARNING: skipped/, label);
    assert.doesNotMatch(code, /\(ข้าม: ยังไม่รองรับการ gen โค้ดของ/, label);
    assert.match(code, re, label);
    assert.doesNotMatch(code, /\bround\(/, `${label}: int(x+0.5)/clamped ints only, never round()`);
    assert.doesNotMatch(code, /img (\+|-|\*|\/) img_b|img_b (\+|-|\*|\/) img/, `${label}: numpy arithmetic on uint8 wraps -- use cv2`);
  }
  assert.match(buildPythonCode(pipe2('hist_match')), /\.astype\(np\.float64\)/);
  assert.match(buildPythonCode(pipe2('inpaint')), /elif nz \* 100 > 25 \* total:/);
  const nGray = (code) => (code.match(/COLOR_BGR2GRAY\)  # \(แปลงขาวดำอัตโนมัติ\)/g) || []).length;
  const after = (opId, local) => buildPythonCode([...pipe2(opId, local), { id: 'threshold', params: { mode: 'binary_threshold', binary_threshold__thresh: 127 } }]);
  assert.equal(nGray(after('imarith', { operation: 'add' })), 1, 'imarith outputs colour -> the next gray step must convert');
  assert.equal(nGray(after('logical_ops', { operation: 'and' })), 0, 'logical_ops outputs gray -> no second conversion');
  assert.equal(nGray(after('inpaint', {})), 1, 'inpaint keeps the (colour) input state');
  assert.equal(nGray(after('hist_match', {})), 1, 'hist_match keeps the (colour) input state');
  const grayFirst = buildPythonCode([{ id: 'color_space', params: { mode: 'grayscale' } }, ...pipe2('hist_match'), { id: 'threshold', params: { mode: 'binary_threshold', binary_threshold__thresh: 127 } }]);
  assert.equal(nGray(grayFirst), 0, 'hist_match after grayscale stays gray -> no conversion before threshold');
});

// ===================================================================
// ipB-12 — codegen parity: the exported Python really produces the same pixels as the app (cv2 5.0.0 + numpy)
// ===================================================================
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'ip-batch2-B-2026-10-02-'));
process.on('exit', () => {
  try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ }
});
function pythonAvailable() {
  try { execFileSync('python', ['-c', 'import cv2, numpy'], { encoding: 'utf8' }); return true; } catch { return false; }
}
const HAS_PYTHON = pythonAvailable();

// ภาพสีของแอปเป็น RGB แต่ cv2.imread ให้ BGR — ส่งให้ Python เป็น BGR (สลับ R/B) ทุกครั้ง ไม่งั้น RGB2GRAY ของแอป
// กับ BGR2GRAY ของ Python จะใช้น้ำหนักสลับกันแล้ว mismatch ปลอม (สเปก §5) · ภาพขาวดำส่งเป็น array 2 มิติ (ndim 2 —
// แบบเดียวกับที่ภาพผ่าน Grayscale มาแล้ว) ทดสอบ branch ndim ของโค้ดที่ export ได้ตรง ๆ
function swapRB3(buf) {
  const out = Buffer.from(buf);
  for (let i = 0; i < out.length; i += 3) { const t = out[i]; out[i] = out[i + 2]; out[i + 2] = t; }
  return out;
}
function toPyRaw(mat) {
  const ch = mat.channels();
  assert.ok(ch === 1 || ch === 3, 'parity fixtures are 1 or 3 channels (cv2.imread never gives RGBA)');
  const data = ch === 3 ? swapRB3(Buffer.from(mat.data)) : Buffer.from(mat.data);
  const shape = ch === 1 ? `(${mat.rows}, ${mat.cols})` : `(${mat.rows}, ${mat.cols}, 3)`;
  return { data, shape };
}
function runPython2(code, A, B) {
  assert.ok(code.includes('img = cv2.imread(path_a)') && code.includes('img_b = cv2.imread(path_b)'), 'two-image loader lines');
  const ra = toPyRaw(A), rb = toPyRaw(B);
  let patched = code
    .replace('img = cv2.imread(path_a)', `img = np.fromfile(path_a, dtype=np.uint8).reshape(${ra.shape})`)
    .replace('img_b = cv2.imread(path_b)', `img_b = np.fromfile(path_b, dtype=np.uint8).reshape(${rb.shape})`);
  const mainIdx = patched.indexOf('if __name__ == "__main__":');
  assert.ok(mainIdx > 0);
  patched = patched.slice(0, mainIdx) + '\nimport sys\nresult = process(sys.argv[1], sys.argv[2])\nprint("NDIM", result.ndim)\nresult.tofile(sys.argv[3])\n';
  const id = Math.random().toString(36).slice(2);
  const scriptPath = path.join(TMP_DIR, `s_${id}.py`);
  const aPath = path.join(TMP_DIR, `a_${id}.bin`), bPath = path.join(TMP_DIR, `b_${id}.bin`), outPath = path.join(TMP_DIR, `o_${id}.bin`);
  fs.writeFileSync(scriptPath, patched, 'utf8');
  fs.writeFileSync(aPath, ra.data);
  fs.writeFileSync(bPath, rb.data);
  const stdout = execFileSync('python', [scriptPath, aPath, bPath, outPath], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
  const ndim = Number((stdout.match(/NDIM (\d)/) || [])[1]);
  return { stdout, ndim, bytes: new Uint8Array(fs.readFileSync(outPath)) };
}
// เทียบผลของ Python กับ OPS[opId].apply ทุกไบต์ · Python 3 ช่อง (BGR) สลับกลับเป็น RGB ก่อนเทียบ · ถ้าแอปได้ 1 ช่อง
// แต่ Python ได้ 3 ช่อง (imarith ตอนทั้งคู่ขาวดำ — precedent ของ imabsdiff) ต้องเท่ากันทั้ง 3 ช่อง
function assertParity2(opId, local, A, B, { expectStdout = null, expectInfo = null, before = [], appA = A, appB = B } = {}) {
  const label = `${opId} ${JSON.stringify(local)} A=${A.cols}x${A.rows}x${A.channels()} B=${B.cols}x${B.rows}x${B.channels()}`;
  const code = buildPythonCode(pipe2(opId, local, [], before));
  const py = runPython2(code, A, B);
  const ctx = { src2: appB };
  const expected = OPS[opId].apply(cv, appA, local, ctx);
  try {
    const js = Buffer.from(expected.data);
    const jsCh = expected.channels();
    const pyBytes = py.ndim === 3 ? swapRB3(py.bytes) : Buffer.from(py.bytes);
    const pyCh = py.ndim === 3 ? 3 : 1;
    let ndiff = 0, maxdiff = 0;
    if (jsCh === pyCh) {
      assert.equal(pyBytes.length, js.length, `${label}: output size`);
      for (let i = 0; i < js.length; i++) { const d = Math.abs(js[i] - pyBytes[i]); if (d) { ndiff++; maxdiff = Math.max(maxdiff, d); } }
    } else {
      assert.ok(jsCh === 1 && pyCh === 3, `${label}: channel layout app=${jsCh} python=${pyCh}`);
      assert.equal(pyBytes.length, js.length * 3, `${label}: output size`);
      for (let i = 0; i < js.length; i++) for (let c = 0; c < 3; c++) { const d = Math.abs(js[i] - pyBytes[i * 3 + c]); if (d) { ndiff++; maxdiff = Math.max(maxdiff, d); } }
    }
    assert.equal(ndiff, 0, `${label}: ndiff=${ndiff} maxdiff=${maxdiff}; python said "${py.stdout.trim()}", app said "${ctx.info}"`);
    if (expectStdout) assert.match(py.stdout, expectStdout, `${label}: python stdout`);
    if (expectInfo) assert.match(ctx.info, expectInfo, `${label}: app ctx.info`);
  } finally { expected.delete(); }
}

const PY_SKIP = { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' };

test('ipB-12a. codegen parity imarith: every operation x (colour+colour, gray+gray [app 1ch vs python 3ch], colour A + gray B of another size, gray A + colour B) is byte-identical to cv2 -- plus the qa vectors and the scale vectors', PY_SKIP, () => {
  const cA = noiseRGB(23, 17, 101), cB = noiseRGB(23, 17, 102), gA = noiseGray(23, 17, 103), gB = noiseGray(23, 17, 104);
  const gSmall = noiseGray(9, 6, 105), cSmall = noiseRGB(11, 13, 106);
  const qa = row([...QA_A, ...SCALE_A]), qb = row([...QA_B, ...SCALE_B]);
  try {
    for (const operation of IMARITH_MODES) {
      assertParity2('imarith', { operation }, cA, cB);
      assertParity2('imarith', { operation }, gA, gB);
      assertParity2('imarith', { operation }, cA, gSmall, { expectInfo: /ปรับขนาด B/ });
      assertParity2('imarith', { operation }, gA, cSmall);
      assertParity2('imarith', { operation }, qa, qb);
    }
  } finally { for (const m of [cA, cB, gA, gB, gSmall, cSmall, qa, qb]) m.delete(); }
});

test('ipB-12a2. codegen parity imarith inside a real pipeline: Grayscale on both lines then subtract (the compile-time ctx.gray path of the exporter) matches the app run step by step', PY_SKIP, () => {
  const cA = noiseRGB(19, 14, 111), cB = noiseRGB(19, 14, 112);
  const before = [{ id: 'color_space', params: { mode: 'grayscale' } }];
  const code = buildPythonCode([
    ...before,
    { id: 'image_arithmetic', params: { mode: 'imarith', imarith__operation: 'subtract' }, enabled: true, src2: { pipeline: [{ id: 'color_space', params: { mode: 'grayscale' } }] } },
  ]);
  const py = runPython2(code, cA, cB);
  const ga = OPS.grayscale.apply(cv, cA, {}, {}), gb = OPS.grayscale.apply(cv, cB, {}, {});
  const out = OPS.imarith.apply(cv, ga, { operation: 'subtract' }, { src2: gb });
  try {
    assert.equal(out.channels(), 1);
    assert.equal(py.ndim, 3, 'exported imarith is always colour (precedent of imabsdiff)');
    const js = out.data;
    let ndiff = 0;
    for (let i = 0; i < js.length; i++) for (let c = 0; c < 3; c++) if (js[i] !== py.bytes[i * 3 + c]) ndiff++;
    assert.equal(ndiff, 0);
  } finally { ga.delete(); gb.delete(); out.delete(); cA.delete(); cB.delete(); }
});

test('ipB-12b. codegen parity logical_ops: every operation x (colour A + colour B, gray A + colour B of another size, values 127/128 on both inputs) is byte-identical to cv2 and the white-pixel count printed by Python equals the app\'s', PY_SKIP, () => {
  const cA = noiseRGB(21, 15, 121), cB = noiseRGB(21, 15, 122), gA = noiseGray(21, 15, 123), cSmall = noiseRGB(8, 9, 124);
  const edgeA = row([127, 128, 127, 128, 0, 255]), edgeB = row([127, 127, 128, 128, 255, 0]);
  try {
    for (const operation of LOGICAL_MODES) {
      assertParity2('logical_ops', { operation }, cA, cB);
      assertParity2('logical_ops', { operation }, gA, cSmall);
      const ctx = { src2: edgeB };
      const o = OPS.logical_ops.apply(cv, edgeA, { operation }, ctx);
      const white = cv.countNonZero(o); o.delete();
      assertParity2('logical_ops', { operation }, edgeA, edgeB, { expectStdout: new RegExp(`white pixels: ${white}\\b`) });
    }
  } finally { for (const m of [cA, cB, gA, cSmall, edgeA, edgeB]) m.delete(); }
});

test('ipB-12c. codegen parity inpaint: TELEA and NS x (colour, gray, colour mask B, smaller mask B, radius 1/7) and every guard (empty mask, 24% / 25% repaired, 26% refused) decide and paint exactly like the app', PY_SKIP, () => {
  const scene = sceneRGB(20, 10, 131), sceneG = gray(20, 10, (x, y) => (40 + 7 * x + 3 * y) & 255);
  const blob = blobMask(20, 10, 6, 3, 11, 7), blobSmall = blobMask(10, 5, 3, 1, 6, 4);
  const blobC = new cv.Mat(); cv.cvtColor(blob, blobC, cv.COLOR_GRAY2RGB);
  try {
    for (const method of INPAINT_METHODS) {
      for (const radius of [1, 3, 7]) assertParity2('inpaint', { method, radius }, scene, blob, { expectStdout: /inpainted 20 px/ });
      assertParity2('inpaint', { method }, sceneG, blob);
      assertParity2('inpaint', { method }, scene, blobC);
      assertParity2('inpaint', { method }, scene, blobSmall, { expectInfo: /ปรับขนาด B/ });
    }
    for (const [n, re] of [[0, /inpaint skipped: mask is empty/], [48, /inpainted 48 px/], [50, /inpainted 50 px/], [52, /inpaint skipped: mask covers 26\.0%/]]) {
      const B = firstN(20, 10, n);
      try { assertParity2('inpaint', { method: 'telea' }, scene, B, { expectStdout: re }); } finally { B.delete(); }
    }
    // % ในข้อความต้องตรงกันสองฝั่ง (debugger 2026-10-03): 23x87 = 2001 px, 501 px = 25.037% -> ปัดขึ้น "25.1" ทั้งคู่
    // (เดิม Python :.1f พิมพ์ "25.0% (> 25%)") · 20x20, 49 px = 12.25% พอดี -> ปัดครึ่งขึ้น "12.3" ทั้งคู่ (toFixed เดิมได้ 12.3 แต่ :.1f ได้ 12.2)
    // 40x50 = 2000 px, 247 px = 12.35% ซึ่งเก็บเป็นเลขฐานสองได้ต่ำกว่า 12.35 นิดหนึ่ง -> toFixed/:.1f ตรง ๆ ได้ "12.3" แต่ปัดครึ่งขึ้นจาก 123.5 ได้ "12.4" ทั้งสองฝั่ง
    const big = sceneRGB(23, 87, 132), b501 = firstN(23, 87, 501), s400 = sceneRGB(20, 20, 133), b49 = firstN(20, 20, 49);
    const s2000 = sceneRGB(40, 50, 134), b247 = firstN(40, 50, 247);
    try {
      assertParity2('inpaint', {}, s2000, b247, { expectStdout: /inpainted 247 px \(12\.4%\)/, expectInfo: /^ซ่อม 247 px \(12\.4% ของภาพ\)/ });
      assertParity2('inpaint', {}, big, b501, { expectStdout: /mask covers 25\.1% of the image/, expectInfo: /^ไม่ซ่อม: mask ครอบ 25\.1% ของภาพ/ });
      assertParity2('inpaint', {}, s400, b49, { expectStdout: /inpainted 49 px \(12\.3%\)/, expectInfo: /^ซ่อม 49 px \(12\.3% ของภาพ\)/ });
    } finally { big.delete(); b501.delete(); s400.delete(); b49.delete(); s2000.delete(); b247.delete(); }
  } finally { for (const m of [scene, sceneG, blob, blobSmall, blobC]) m.delete(); }
});

test('ipB-12d. codegen parity hist_match: gray+gray, colour+colour, colour A + single-channel B, gray A + colour B (B converted with the matching gray weights), B of another size, and the float64 trap (5971/10057 vs 2173/3660) are byte-identical to the numpy searchsorted version', PY_SKIP, () => {
  const gA = noiseGray(37, 23, 141), gB = noiseGray(19, 29, 142), cA = sceneRGB(37, 23, 143), cB = noiseRGB(16, 12, 144);
  const trapA = float32TrapA(), trapB = float32TrapB();
  try {
    assertParity2('hist_match', {}, gA, gB);
    assertParity2('hist_match', {}, cA, cB);
    assertParity2('hist_match', {}, cA, gB);
    assertParity2('hist_match', {}, gA, cB);
    assertParity2('hist_match', {}, cB, cA);
    assertParity2('hist_match', {}, trapA, trapB, { expectStdout: /histogram of A matched to B/ });
  } finally { for (const m of [gA, gB, cA, cB, trapA, trapB]) m.delete(); }
});

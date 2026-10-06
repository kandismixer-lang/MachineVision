// Tier 2, 18a — infra tests: 2nd image input (src2), ctx.values plumbing, per-node persistent
// state (getNodeState/sweepNodeState), และ op ทดสอบจริงตัวแรก (imabsdiff)
// ครอบคลุม 11 ข้อใน spec §C.1 "18a acceptance tests" ทีละข้อ (คอมเมนต์ tier2-N บอกไว้ตรงแต่ละเทสต์)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { OPS, FILTERS, needsOf } from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS } from '../src/renderer/ui/palette-panel.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import { assertLeakFree, makeThrowingCv } from './_leak-helpers.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cvtColor === 'function'; }
async function waitForCv() {
  if (isReady(cv)) return;
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}
await waitForCv();

// cv-worker.js เป็น Web Worker script จริง (self.onmessage = ... ที่ top-level ของโมดูล) — Node ไม่มี
// `self`/`ImageData` เป็น global ตามธรรมชาติ ต้อง polyfill "ก่อน" import (dynamic import เพื่อไม่ให้
// static-import hoisting รันโมดูลก่อนที่ polyfill จะพร้อม) แค่พอให้โมดูล evaluate ผ่าน — ไม่กระทบโค้ด
// production เลย (self.onmessage ไม่เคยถูกเรียกจริงในเทสต์พวกนี้ เราเรียกฟังก์ชัน exported ตรง ๆ)
globalThis.self = globalThis;
if (typeof globalThis.ImageData === 'undefined') {
  globalThis.ImageData = class ImageData {
    constructor(data, width, height) { this.data = data; this.width = width; this.height = height; }
  };
}
const cvWorker = await import('../src/renderer/cv/cv-worker.js');
const { runPipeline, computeFrame, getNodeState, sweepNodeState, collectNodeIds } = cvWorker;

function flattenPaletteOps() {
  const ops = [];
  for (const section of TOOL_SECTIONS) ops.push(...section.ops);
  return ops;
}

let __uid = 1;
function uniqueNodeId() { return 900000 + __uid++; } // ช่วงเลขที่ไม่ชนกับ nodeId จริงใด ๆ ในแอป

// ===================================================================
// ภาพทดสอบสังเคราะห์
// ===================================================================
function solidRGB(w, h, r, g, b) {
  return new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(r, g, b));
}
function solidGray(w, h, v) {
  return new cv.Mat(h, w, cv.CV_8UC1, new cv.Scalar(v));
}
// ภาพสีพื้น + สี่เหลี่ยม "ต่าง" อยู่ตรงกลาง — ใช้ทดสอบความถูกต้องของ imabsdiff (บริเวณนอกสี่เหลี่ยม
// ต้องได้ diff=0 บริเวณในสี่เหลี่ยมต้องได้ diff=|255-0|=255 ทุกช่องสี)
function withRect(w, h, bg, rect, fg) {
  const m = solidRGB(w, h, bg, bg, bg);
  cv.rectangle(m, new cv.Point(rect.x, rect.y), new cv.Point(rect.x + rect.w, rect.y + rect.h), new cv.Scalar(fg, fg, fg), -1);
  return m;
}

// ===================================================================
// leak-test (P-14 — นโยบายเต็มใน tests/_leak-helpers.mjs): ไฟล์นี้ "ไม่มี" address-fingerprint แล้ว — ลองย้าย tier2-10 ขึ้นเป็น
// เทสต์แรกของไฟล์ (process สะอาด) แต่ fingerprint ยัง DIFF ตั้งแต่ path แรก (no-src2) แม้ warm-up 200 รอบ + fingerprint เปล่า
// 2 ครั้งก่อนวัด (small-bin ของ allocator จัดเรียงใหม่ช่วงต้น process หลัง import cv-worker.js) ทั้งที่ tracker/census ยืนยันว่า
// ไม่รั่ว -> tier2-3d, tier2-4, tier2-10 ทั้งหมดใช้ wrapper-balance + heap census (assertLeakFree)
// ===================================================================

// ทำ cv wrapper ที่ cv.absdiff throw (Proxy) — ใช้ทดสอบ error-path cleanup ของ imabsdiff.apply()
// (ใช้ makeThrowingCv กลาง ข้อความเดิมเป๊ะ: `injected test failure (absdiff)`)
function makeThrowingAbsdiffCv(realCv) {
  return makeThrowingCv(realCv, 'absdiff');
}

// ===================================================================
// tier2-1. imabsdiff ไม่ต่อสาย B -> graceful message, ไม่ throw
// ===================================================================
test('tier2-1. imabsdiff: no src2 connected -> graceful info message, output unchanged, no throw', () => {
  const src = withRect(40, 30, 10, { x: 10, y: 8, w: 15, h: 12 }, 200);
  const ctx = {};
  let out;
  assert.doesNotThrow(() => { out = OPS.imabsdiff.apply(cv, src, {}, ctx); });
  assert.equal(out.cols, 40);
  assert.equal(out.rows, 30);
  assert.match(ctx.info, /ยังไม่ได้ต่อสาย 'เข้า B'/);
  // ต้องคืนภาพเดิม ไม่ใช่ภาพดำ/ว่างเปล่า
  assert.equal(out.data[0], 10);
  out.delete();
  src.delete();
});

// ===================================================================
// tier2-2. size mismatch -> matchSize resize path fires, ctx.info บอก
// ===================================================================
test('tier2-2. imabsdiff: src2 with a different size -> matchSize resize path fires, ctx.info mentions it, output size = A', () => {
  const a = solidRGB(50, 40, 100, 100, 100);
  const b = solidRGB(20, 15, 50, 50, 50); // ขนาดต่างจาก a
  const ctx = { src2: b };
  const out = OPS.imabsdiff.apply(cv, a, {}, ctx);
  assert.equal(out.cols, 50);
  assert.equal(out.rows, 40);
  assert.match(ctx.info, /ปรับขนาด B ให้เท่า A/);
  out.delete();
  a.delete();
  b.delete();
});

// ===================================================================
// tier2-3. correctness: ภาพสังเคราะห์รู้คำตอบล่วงหน้า -> nonzero เฉพาะบริเวณที่ต่างจริง
// ===================================================================
test('tier2-3. imabsdiff correctness: known differing rectangle region -> output is 0 outside it, 255 inside it (per channel)', () => {
  const w = 40, h = 30;
  const rect = { x: 10, y: 8, w: 15, h: 12 };
  const a = solidRGB(w, h, 0, 0, 0);
  const b = withRect(w, h, 0, rect, 255);
  const ctx = { src2: b };
  let out;
  try {
    out = OPS.imabsdiff.apply(cv, a, {}, ctx);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        // cv.rectangle(pt1, pt2, ..., -1) วาดแบบ inclusive ทั้งสองมุม (พิกเซลที่ x===rect.x+rect.w หรือ
        // y===rect.y+rect.h ก็ถูกวาดด้วย) — ใช้ <= ให้ตรงกับพฤติกรรมจริงของ OpenCV ไม่ใช่ half-open range
        const inRect = x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h;
        const idx = (y * w + x) * 3;
        const expected = inRect ? 255 : 0;
        assert.equal(out.data[idx], expected, `pixel (${x},${y}) R channel`);
        assert.equal(out.data[idx + 1], expected, `pixel (${x},${y}) G channel`);
        assert.equal(out.data[idx + 2], expected, `pixel (${x},${y}) B channel`);
      }
    }
  } finally {
    if (out) out.delete();
    a.delete();
    b.delete();
  }
});

test('tier2-3b. CV-eng recheck: imabsdiff on two gray (1-channel) inputs stays gray (1-channel) output, not upconverted to RGB — matches the classic Grayscale-vs-Grayscale teaching pipeline a learner would build, and avoids a (h,w,3) surprise when they export the Python code', () => {
  const w = 20, h = 16;
  const a = solidGray(w, h, 0);
  const rect = { x: 4, y: 3, w: 8, h: 6 };
  const b = solidGray(w, h, 0);
  cv.rectangle(b, new cv.Point(rect.x, rect.y), new cv.Point(rect.x + rect.w, rect.y + rect.h), new cv.Scalar(255), -1);
  const ctx = { src2: b };
  let out;
  try {
    out = OPS.imabsdiff.apply(cv, a, {}, ctx);
    assert.equal(out.channels(), 1, 'expected a gray-vs-gray diff to stay 1-channel, not be upconverted to RGB');
    assert.match(ctx.info, /ขาวดำ/, 'ctx.info should mention this took the gray fast-path');
    // spot-check correctness is unaffected by the new branch
    assert.equal(out.ucharPtr(0, 0)[0], 0, 'outside the differing rect must be 0');
    assert.equal(out.ucharPtr(rect.y + 1, rect.x + 1)[0], 255, 'inside the differing rect must be 255');
  } finally {
    if (out) out.delete();
    a.delete();
    b.delete();
  }
});

test('tier2-3c. CV-eng recheck regression: imabsdiff with one gray + one color input still upconverts to RGB (mixed-channel case unchanged)', () => {
  const w = 20, h = 16;
  const a = solidGray(w, h, 100); // gray primary
  const b = solidRGB(w, h, 100, 50, 200); // color secondary
  const ctx = { src2: b };
  let out;
  try {
    out = OPS.imabsdiff.apply(cv, a, {}, ctx);
    assert.equal(out.channels(), 3, 'expected a gray-vs-color diff to still upconvert to RGB (mixed-channel case)');
    assert.doesNotMatch(ctx.info, /ขาวดำ/, 'ctx.info must not claim the gray fast-path for a mixed-channel case');
  } finally {
    if (out) out.delete();
    a.delete();
    b.delete();
  }
});

test('tier2-3d. leak probe: imabsdiff gray-fast-path (both-gray same size, both-gray mismatched size, both-gray thrown-error) does not leak', () => {
  const aGray = solidGray(20, 16, 50);
  const bGraySame = solidGray(20, 16, 90);
  const bGrayMismatch = solidGray(30, 24, 90); // exercises matchSize's resize path inside the new gray branch too

  // P-14: เดิมเป็น malloc-fingerprint กลางไฟล์ -> wrapper-balance + heap census
  assertLeakFree(cv, 'imabsdiff (gray fast-path, same size)', (c) => {
    const o = OPS.imabsdiff.apply(c, aGray, {}, { src2: bGraySame });
    o.delete();
  });
  assertLeakFree(cv, 'imabsdiff (gray fast-path, mismatched size)', (c) => {
    const o = OPS.imabsdiff.apply(c, aGray, {}, { src2: bGrayMismatch });
    o.delete();
  }, { minCreated: 2 });
  // cleanup happens in the catch/finally of the new gray branch itself
  assertLeakFree(cv, 'imabsdiff (gray fast-path, JS-exception mid-absdiff)', (c) => {
    const o = OPS.imabsdiff.apply(makeThrowingAbsdiffCv(c), aGray, {}, { src2: bGraySame });
    o.delete();
  }, { expectThrow: /injected test failure \(absdiff\)/ });

  aGray.delete();
  bGraySame.delete();
  bGrayMismatch.delete();
});

// ===================================================================
// tier2-4. runPipeline: src2 ที่ตัวมันเองมี 2-image step ซ้อนอีกชั้น (nested) -> ไม่ throw, ไม่รั่ว
// ===================================================================
function buildNestedSrc2Pipeline() {
  // outer: image_arithmetic(imabsdiff), src2 = [ inner: image_arithmetic(imabsdiff) (ไม่มี src2 ของตัวเอง
  //   -> เดิน pipeline:[] ว่างเปล่า -> runPipeline คืน clone ของ baseRGB เป็น ctx.src2 ของ inner) ]
  const innerStep = {
    id: 'image_arithmetic',
    params: { mode: 'imabsdiff' },
    enabled: true,
    nodeId: uniqueNodeId(),
    src2: { pipeline: [] }, // เดินแล้วไม่เจอ SourceNode จริง -> runPipeline([]) คืน clone ของ baseRGB เอง (ดู runPipeline)
  };
  const outerStep = {
    id: 'image_arithmetic',
    params: { mode: 'imabsdiff' },
    enabled: true,
    nodeId: uniqueNodeId(),
    src2: { pipeline: [innerStep] },
  };
  return [outerStep];
}

test('tier2-4. runPipeline: nested src2-of-src2 (2-image op feeding another 2-image op) does not throw and does not leak', () => {
  const w = 24, h = 18;
  const run = () => {
    const base = solidRGB(w, h, 77, 88, 99);
    const pipeline = buildNestedSrc2Pipeline();
    const r = runPipeline(base, pipeline, new Map(), new Set());
    assert.equal(r.mat.cols, w);
    assert.equal(r.mat.rows, h);
    r.mat.delete();
    base.delete();
  };
  assert.doesNotThrow(run);
  // P-14: runPipeline ใช้ cv ระดับโมดูลของ cv-worker.js (ไม่รับ cv เป็นพารามิเตอร์) -> tracker โหมด global (สลับ cv ตัวจริงชั่วคราว)
  // + heap census แทน fingerprint — base ถูกสร้างใน run จึงถูกนับด้วย (รวม clone ของ base ที่เป็น src2 ของ inner)
  assertLeakFree(cv, 'tier2-4 nested src2-of-src2', () => run(), { global: true, minCreated: 4, iters: 100 });
});

// ===================================================================
// tier2-5/6/7. getNodeState / sweepNodeState — dispose-once, params-fingerprint reset, resolution reset
// ===================================================================
test('tier2-5. getNodeState/sweepNodeState: node present frames 1-2, absent frame 3 -> __dispose called exactly once, fresh state after', () => {
  const id = uniqueNodeId();
  let disposeCalls = 0;
  const s1 = getNodeState(id, { a: 1 }, 10, 10);
  s1.__dispose = () => { disposeCalls++; };

  // frame 1: node ยังอยู่ -> sweep ต้องไม่ dispose
  sweepNodeState(new Set([id]));
  assert.equal(disposeCalls, 0);
  const s1b = getNodeState(id, { a: 1 }, 10, 10);
  assert.equal(s1b, s1, 'same params/dims + still active -> same state object (no reset)');

  // frame 2: ยังอยู่อีก -> ยังไม่ dispose
  sweepNodeState(new Set([id]));
  assert.equal(disposeCalls, 0);

  // frame 3: node หายจาก activeNodeIds -> ต้อง dispose ครั้งเดียว แล้วลบออกจาก map จริง
  sweepNodeState(new Set()); // id ไม่อยู่ในนี้
  assert.equal(disposeCalls, 1, '__dispose must be called exactly once');

  const s2 = getNodeState(id, { a: 1 }, 10, 10); // เรียกซ้ำด้วย params/dims เดิมเป๊ะ
  assert.notEqual(s2, s1, 'after sweep removed the entry, a fresh state object must be created');

  // sweep อีกครั้งโดยไม่มี id นี้ ไม่ควร dispose ซ้ำ (ป้องกัน double-dispose ของ s1 เดิม)
  sweepNodeState(new Set());
  assert.equal(disposeCalls, 1, 'sweeping again after removal must not call the old __dispose a second time');
});

test('tier2-6. getNodeState: params fingerprint change -> __dispose called, fresh state object returned', () => {
  const id = uniqueNodeId();
  let disposeCalls = 0;
  const s1 = getNodeState(id, { k: 1 }, 20, 20);
  s1.__dispose = () => { disposeCalls++; };
  const s2 = getNodeState(id, { k: 2 }, 20, 20); // params เปลี่ยน, ขนาดเท่าเดิม
  assert.equal(disposeCalls, 1);
  assert.notEqual(s2, s1);
  assert.equal(s2.__paramsFingerprint, JSON.stringify({ k: 2 }));
  sweepNodeState(new Set()); // cleanup กันค้าง
});

test('tier2-7. getNodeState: resolution change (same nodeId, different width/height) -> __dispose called, fresh state', () => {
  const id = uniqueNodeId();
  let disposeCalls = 0;
  const s1 = getNodeState(id, { k: 1 }, 100, 80);
  s1.__dispose = () => { disposeCalls++; };
  const s2 = getNodeState(id, { k: 1 }, 640, 480); // params เท่าเดิม แต่ขนาดเปลี่ยน
  assert.equal(disposeCalls, 1);
  assert.notEqual(s2, s1);
  assert.equal(s2.__lastW, 640);
  assert.equal(s2.__lastH, 480);
  sweepNodeState(new Set()); // cleanup กันค้าง
});

// ===================================================================
// tier2-8. Backward-compat: pipeline เก่า (1-input ล้วน) ผ่าน runPipeline signature ใหม่ (4 args) ต้อง
// ได้ผลลัพธ์ byte-identical กับของเดิม (2 args) — เทสต์สำคัญที่สุดของรอบนี้
// ===================================================================
// สำเนาตรงตัวของ runPipeline ก่อนแก้ไข Tier 2 (คัดลอกจาก cv-worker.js ก่อนแก้ตรง ๆ — ดู worklog สำหรับ
// diff เทียบกับของจริง) ใช้เป็น ground-truth อ้างอิงอิสระ ไม่พึ่งพาโค้ดที่กำลังทดสอบเลย
function toGrayRef(cvArg, mat) {
  const g = new cvArg.Mat();
  if (mat.channels() === 1) mat.copyTo(g);
  else cvArg.cvtColor(mat, g, mat.channels() === 4 ? cvArg.COLOR_RGBA2GRAY : cvArg.COLOR_RGB2GRAY);
  return g;
}
function oldRunPipelineRef(cvArg, baseRGB, pipeline) {
  let current = baseRGB;
  let ownsCurrent = false;
  const notes = [];
  for (const step of pipeline) {
    if (!step || step.enabled === false) continue;
    const f = FILTERS[step.id];
    if (!f) continue;
    let input = current;
    let converted = null;
    if (needsOf(f, step.params || {}) === 'gray' && input.channels() > 1) {
      converted = toGrayRef(cvArg, input);
      input = converted;
      notes.push({ id: step.id, autoGray: true });
    }
    const ctx = {};
    let out;
    try {
      out = f.apply(cvArg, input, step.params || {}, ctx);
    } catch (err) {
      if (converted) converted.delete();
      if (ownsCurrent) current.delete();
      throw err;
    }
    if (ctx.info) notes.push({ id: step.id, info: ctx.info });
    if (converted) converted.delete();
    if (ownsCurrent) current.delete();
    current = out;
    ownsCurrent = true;
  }
  if (!ownsCurrent) {
    const clone = new cvArg.Mat();
    current.copyTo(clone);
    current = clone;
  }
  return { mat: current, notes };
}

function representativeOldPipeline() {
  return [
    { id: 'blur', params: { mode: 'gaussian_blur', gaussian_blur__ksize: 5 }, enabled: true, nodeId: uniqueNodeId() },
    { id: 'threshold', params: { mode: 'binary_threshold', binary_threshold__thresh: 100 }, enabled: true, nodeId: uniqueNodeId() },
    { id: 'edge_detection', params: { mode: 'canny', canny__t1: 50, canny__t2: 150 }, enabled: true, nodeId: uniqueNodeId() },
  ];
}

test('tier2-8. Backward-compat: new 4-arg runPipeline produces byte-identical output+notes to the old 2-arg logic for a pre-existing 1-input pipeline', () => {
  const w = 96, h = 72;
  const baseA = withRect(w, h, 30, { x: 20, y: 15, w: 40, h: 30 }, 210);
  const baseB = withRect(w, h, 30, { x: 20, y: 15, w: 40, h: 30 }, 210);
  const pipeline = representativeOldPipeline();

  const oldResult = oldRunPipelineRef(cv, baseA, pipeline);
  const newResult = runPipeline(baseB, pipeline, new Map(), new Set(pipeline.map((s) => s.nodeId)));

  assert.equal(newResult.mat.cols, oldResult.mat.cols);
  assert.equal(newResult.mat.rows, oldResult.mat.rows);
  assert.equal(newResult.mat.channels(), oldResult.mat.channels());
  assert.deepEqual(Array.from(newResult.mat.data), Array.from(oldResult.mat.data), 'pixel bytes must be identical');
  // [nodeId round] note ใหม่ได้ field เพิ่ม (nodeId/view — additive) → เทียบเนื้อหาเดิมโดยตัด 2 field นี้ออก
  // และยืนยันว่าทุก note พก nodeId ของ step จริง (สัญญาใหม่)
  const stripNew = (ns) => ns.map(({ nodeId, view, ...rest }) => rest);
  assert.deepEqual(stripNew(newResult.notes), oldResult.notes, 'notes array must be identical (ยกเว้น nodeId/view ที่เพิ่มใหม่)');
  for (const n of newResult.notes) assert.ok(pipeline.some((st) => st.nodeId === n.nodeId && st.id === n.id), 'note ทุกตัวต้องมี nodeId ของ step ที่สร้างมัน');

  oldResult.mat.delete();
  newResult.mat.delete();
});

// Revert-proof (per spec §E): prove this test actually detects a regression, not just "doesn't throw".
// Simulate a broken new runPipeline (e.g. one that skips the auto-gray note or mangles output) and confirm
// deepEqual/pixel comparison above WOULD fail on it — done by direct inspection rather than a live revert:
// oldRunPipelineRef pushes {id,autoGray:true} for canny's auto-gray-conversion step; the real cv-worker.js
// runPipeline must too, or this exact deepEqual on `notes` fails. Locks that in as a concrete assertion:
test('tier2-8b. revert-proof: the auto-gray note for canny (edge_detection) actually fires and is captured identically', () => {
  const w = 60, h = 48;
  const base = withRect(w, h, 40, { x: 10, y: 10, w: 20, h: 20 }, 200);
  const pipeline = [{ id: 'edge_detection', params: { mode: 'canny', canny__t1: 50, canny__t2: 150 }, enabled: true, nodeId: uniqueNodeId() }];
  const r = runPipeline(base, pipeline, new Map(), new Set(pipeline.map((s) => s.nodeId)));
  assert.ok(r.notes.some((n) => n.id === 'edge_detection' && n.autoGray === true), 'expected an autoGray note — if this is missing, the gray-conversion path silently broke');
  r.mat.delete();
});

// ===================================================================
// tier2-9. editor.js: traceChainFrom (refactored backward walk) — ต้องได้ pipeline/connected shape
// เดิมเป๊ะสำหรับสาย 1-input แบบเก่า และรองรับสาย 2-input (src2) ถูกต้องสำหรับสายใหม่
//
// หมายเหตุขอบเขต (บันทึกไว้ตรง ๆ ให้ qa เห็น): createGraphEditor(canvasEl, cb) ต้องการ LGraphCanvas จริง
// ซึ่งต้องการ DOM (`document`/`canvas.getContext('2d')`/event listeners ฯลฯ) ที่ไม่มีใน Node test
// runner นี้ และไม่อนุญาตติดตั้ง jsdom (package.json ห้ามแตะ, ห้าม npm install) — จึงทดสอบฟังก์ชัน
// `traceChainFrom` ที่ "เอาซอร์สจริง" จาก editor.js มาสกัดด้วยการนับวงเล็บปีกกาให้สมดุล (ไม่ใช่การคัดลอก
// ด้วยมือที่เสี่ยง drift จากของจริง) แล้ว eval รันด้วย mock graph/FILTERS/LiteGraph ธรรมดา (ไม่พึ่ง DOM
// เลยเพราะ traceChainFrom เองไม่แตะ canvas/document — อ่านแค่ graph.links/graph.getNodeById + node
// properties เท่านั้น) — แนะนำให้ qa/ผู้ใช้เปิดโปรเจกต์เก่าจริงในแอป (Electron live) เป็นการยืนยันขั้น
// สุดท้ายอีกชั้นสำหรับจุดนี้โดยเฉพาะ เนื่องจากเป็นทางเดียวที่ทดสอบผ่าน LGraphCanvas จริง
function extractBalancedFunctionSource(fileText, signatureStart) {
  const idx = fileText.indexOf(signatureStart);
  assert.ok(idx >= 0, `function signature not found in source: ${signatureStart}`);
  // เริ่มหา "{" ของ function BODY หลังจบ signatureStart เท่านั้น (ไม่ใช่ตั้งแต่ idx) — signatureStart
  // ที่ส่งเข้ามารวม default param "opts = {}" ซึ่งมี "{"/"}" ของตัวเองอยู่แล้ว ถ้าเริ่มหาจาก idx ตรง ๆ
  // จะไปเจอ "{" ของ "opts = {}" ก่อน ทำให้ brace-counting จบเร็วเกินไปตั้งแต่ยังไม่ถึง body จริง
  let i = fileText.indexOf('{', idx + signatureStart.length);
  assert.ok(i > idx, 'opening brace not found');
  let depth = 0;
  for (; i < fileText.length; i++) {
    if (fileText[i] === '{') depth++;
    else if (fileText[i] === '}') { depth--; if (depth === 0) { i++; break; } }
  }
  return fileText.slice(idx, i);
}

const editorSrc = fs.readFileSync(path.join(__dirname, '../src/renderer/graph/editor.js'), 'utf8');
const traceChainFromSrc = extractBalancedFunctionSource(editorSrc, 'function traceChainFrom(startNode, startInputIdx, opts = {})');

// mock graph/node builder — เลียนแบบ shape ที่ traceChainFrom ใช้จริงเท่านั้น (links[id]={origin_id,
// origin_slot,target_id,target_slot}, getNodeById, node.{id,type,filterId,properties,mode,inputs})
function makeMockGraph() {
  const nodes = new Map();
  const links = {};
  let nextLinkId = 1;
  let nextNodeId = 1;
  function addNode(props) {
    const id = nextNodeId++;
    const node = { id, inputs: [], ...props };
    nodes.set(id, node);
    return node;
  }
  function connect(originNode, originSlot, targetNode, targetSlotIdx) {
    const linkId = nextLinkId++;
    links[linkId] = { id: linkId, origin_id: originNode.id, origin_slot: originSlot, target_id: targetNode.id, target_slot: targetSlotIdx };
    targetNode.inputs[targetSlotIdx] = { link: linkId };
    return linkId;
  }
  return { nodes, links, getNodeById: (id) => nodes.get(id), addNode, connect };
}

const SOURCE_TYPE_T = 'mv/source';
const DL_TYPE_T = 'mv/dl-yolo';
const LG_NEVER = 2; // ยืนยันแล้วจาก LiteGraph.NEVER จริง (node_modules/litegraph.js)
const LiteGraphMock = { NEVER: LG_NEVER };

function buildTraceChainFrom(g) {
  return new Function('graph', 'FILTERS', 'LiteGraph', 'SOURCE_TYPE', 'DL_TYPE',
    traceChainFromSrc + '\nreturn traceChainFrom;'
  )(g, FILTERS, LiteGraphMock, SOURCE_TYPE_T, DL_TYPE_T);
}

test('tier2-9a. traceChainFrom: old-style single-input chain (Source -> filterA -> filterB -> target) traces identically to the pre-refactor shape', () => {
  const g = makeMockGraph();
  const source = g.addNode({ type: SOURCE_TYPE_T });
  const filterA = g.addNode({ filterId: 'blur', properties: { mode: 'gaussian_blur', gaussian_blur__ksize: 5 }, mode: 0 });
  const filterB = g.addNode({ filterId: 'threshold', properties: { mode: 'binary_threshold', binary_threshold__thresh: 90 }, mode: 0 });
  const target = g.addNode({}); // เช่น DisplayNode (ไม่มี filterId)
  g.connect(source, 0, filterA, 0);
  g.connect(filterA, 0, filterB, 0);
  g.connect(filterB, 0, target, 0);

  const trace = buildTraceChainFrom(g);
  const r = trace(target, 0, { stopAtDL: false, collectPathNodes: true });
  assert.equal(r.connected, true);
  assert.equal(r.hasDL, false);
  assert.deepEqual(r.pipeline, [
    { id: 'blur', params: { mode: 'gaussian_blur', gaussian_blur__ksize: 5 }, enabled: true, nodeId: filterA.id },
    { id: 'threshold', params: { mode: 'binary_threshold', binary_threshold__thresh: 90 }, enabled: true, nodeId: filterB.id },
  ]);
  assert.deepEqual(r.pathNodes, [filterA, filterB]);
});

test('tier2-9b. traceChainFrom: unconnected chain (no path to SourceNode) -> connected:false, pipeline:[]', () => {
  const g = makeMockGraph();
  const filterA = g.addNode({ filterId: 'blur', properties: { mode: 'gaussian_blur' }, mode: 0 });
  const target = g.addNode({});
  g.connect(filterA, 0, target, 0); // filterA ไม่ได้ต่อกับ source เลย
  const trace = buildTraceChainFrom(g);
  const r = trace(target, 0, { stopAtDL: false, collectPathNodes: true });
  assert.equal(r.connected, false);
  assert.deepEqual(r.pipeline, []);
});

test('tier2-9c. traceChainFrom: main walk continues through a DL node (stopAtDL:false) — matches tracePipeline\'s original hasDL-tracking behavior', () => {
  const g = makeMockGraph();
  const source = g.addNode({ type: SOURCE_TYPE_T });
  const dl = g.addNode({ type: DL_TYPE_T });
  const filterA = g.addNode({ filterId: 'blur', properties: { mode: 'gaussian_blur' }, mode: 0 });
  const target = g.addNode({});
  g.connect(source, 0, dl, 0);
  g.connect(dl, 0, filterA, 0);
  g.connect(filterA, 0, target, 0);
  const trace = buildTraceChainFrom(g);
  const r = trace(target, 0, { stopAtDL: false, collectPathNodes: true });
  assert.equal(r.connected, true, 'must keep walking through the DL node and still reach source');
  assert.equal(r.hasDL, true);
  assert.deepEqual(r.pipeline, [{ id: 'blur', params: { mode: 'gaussian_blur' }, enabled: true, nodeId: filterA.id }]);
  assert.deepEqual(r.pathNodes, [dl, filterA]);
});

test('tier2-9d. traceChainFrom: stopAtDL:true (traceNodePipeline/src2 semantics) stops at a DL node -> not connected', () => {
  const g = makeMockGraph();
  const source = g.addNode({ type: SOURCE_TYPE_T });
  const dl = g.addNode({ type: DL_TYPE_T });
  const filterA = g.addNode({ filterId: 'blur', properties: { mode: 'gaussian_blur' }, mode: 0 });
  g.connect(source, 0, dl, 0);
  g.connect(dl, 0, filterA, 0);
  const trace = buildTraceChainFrom(g);
  const r = trace(filterA, 0, { stopAtDL: true, collectPathNodes: false });
  assert.equal(r.connected, false, 'stopAtDL must break at the DL node without reaching source');
  assert.deepEqual(r.pipeline, []);
});

test('tier2-9e. traceChainFrom: a 2-image (inputs===2) node picks up its own src2 recursively, including nested src2-of-src2', () => {
  const g = makeMockGraph();
  const source = g.addNode({ type: SOURCE_TYPE_T });
  const sourceB = g.addNode({ type: SOURCE_TYPE_T }); // แอปจริงมี source เดียว แต่ mock graph ไม่บังคับ — พิสูจน์ recursion ทั่วไป
  const filterB1 = g.addNode({ filterId: 'blur', properties: { mode: 'gaussian_blur' }, mode: 0 });
  const outer = g.addNode({ filterId: 'image_arithmetic', properties: { mode: 'imabsdiff' }, mode: 0 });
  const target = g.addNode({});
  g.connect(source, 0, outer, 0); // input A ของ outer
  g.connect(sourceB, 0, filterB1, 0);
  g.connect(filterB1, 0, outer, 1); // input B ของ outer (slot index 1)
  g.connect(outer, 0, target, 0);

  const trace = buildTraceChainFrom(g);
  const r = trace(target, 0, { stopAtDL: false, collectPathNodes: true });
  assert.equal(r.connected, true);
  assert.equal(r.pipeline.length, 1);
  const outerStep = r.pipeline[0];
  assert.equal(outerStep.id, 'image_arithmetic');
  assert.ok(outerStep.src2, 'outer step must carry a src2 sub-pipeline (FILTERS.image_arithmetic.inputs===2)');
  assert.equal(outerStep.src2.connected, true);
  assert.deepEqual(outerStep.src2.pipeline, [
    { id: 'blur', params: { mode: 'gaussian_blur' }, enabled: true, nodeId: filterB1.id },
  ]);
});

// ===================================================================
// tier2-10. Leak probe เต็มรูปแบบ (150-200 iterations) สำหรับ imabsdiff: normal / no-src2 / mismatched-size / thrown-error
// ===================================================================
test('tier2-10. imabsdiff leak probe (wrapper-balance + heap census): normal, no-src2, mismatched-size, and thrown-error paths', () => {
  const a = withRect(48, 36, 20, { x: 5, y: 5, w: 10, h: 10 }, 220);
  const bSame = withRect(48, 36, 20, { x: 5, y: 5, w: 10, h: 10 }, 220);
  const bMismatch = solidRGB(24, 18, 60, 60, 60);
  // P-14: เดิมเป็น malloc-fingerprint กลางไฟล์ — ย้ายขึ้นบนสุดก็ยังไม่นิ่ง (ดูหัวไฟล์) จึงใช้ wrapper-balance + heap census
  assertLeakFree(cv, 'imabsdiff (no src2)', (c) => {
    const o = OPS.imabsdiff.apply(c, a, {}, {});
    o.delete();
  });
  assertLeakFree(cv, 'imabsdiff (normal, same size)', (c) => {
    const o = OPS.imabsdiff.apply(c, a, {}, { src2: bSame });
    o.delete();
  });
  assertLeakFree(cv, 'imabsdiff (mismatched size -> matchSize resize path)', (c) => {
    const o = OPS.imabsdiff.apply(c, a, {}, { src2: bMismatch });
    o.delete();
  }, { minCreated: 2 });
  // cleanup happens in the catch/finally of imabsdiff.apply itself
  assertLeakFree(cv, 'imabsdiff (JS-exception mid-absdiff)', (c) => {
    const o = OPS.imabsdiff.apply(makeThrowingAbsdiffCv(c), a, {}, { src2: bSame });
    o.delete();
  }, { expectThrow: /injected test failure \(absdiff\)/ });

  a.delete();
  bSame.delete();
  bMismatch.delete();
});

// ===================================================================
// tier2-11. Codegen: buildPythonCode(pipeline) สำหรับ imabsdiff -> two-arg process(path_a, path_b) form,
// รันจริงด้วย cv2 บนภาพ raw สังเคราะห์ (ไม่ใช้ JPEG) เทียบตัวเลขกับผลลัพธ์ฝั่ง JS (absdiff)
// ===================================================================
for (const name of fs.readdirSync(os.tmpdir())) {
  if (name.startsWith('tier2-parity-')) {
    try { fs.rmSync(path.join(os.tmpdir(), name), { recursive: true, force: true }); } catch { /* best-effort */ }
  }
}
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'tier2-parity-'));
process.on('exit', () => {
  try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ }
});

function matToRaw(mat) {
  return { data: Buffer.from(mat.data), width: mat.cols, height: mat.rows, channels: mat.channels() };
}

// รัน python script ที่ patch แล้วให้อ่าน/เขียนเป็น raw bytes (np.fromfile/tofile) แทน JPEG (กัน lossy
// compression บิดผลตัวเลข) — เทคนิคเดียวกับไฟล์เทสต์อื่น ๆ ในโปรเจกต์นี้ (groupA/keypoints)
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
  execFileSync('python', [scriptPath, aPath, bPath, outPath], { encoding: 'utf8' });
  return new Uint8Array(fs.readFileSync(outPath));
}

test('tier2-11a. codegen: buildPythonCode on an imabsdiff pipeline emits process(path_a, path_b), executed against real cv2 matches JS absdiff exactly (raw, non-JPEG images)', () => {
  const w = 32, h = 24;
  const rect = { x: 6, y: 5, w: 10, h: 8 };
  const matA = withRect(w, h, 40, rect, 210);
  const matB = withRect(w, h, 40, rect, 60); // ต่างจาก A เฉพาะในสี่เหลี่ยม (210 vs 60 -> diff=150)

  const pipeline = [{ id: 'image_arithmetic', params: { mode: 'imabsdiff' }, enabled: true, nodeId: uniqueNodeId(), src2: { pipeline: [] } }];
  const code = buildPythonCode(pipeline);
  assert.match(code, /def process\(path_a, path_b\):/);

  const jsOut = OPS.imabsdiff.apply(cv, matA, {}, { src2: matB });
  const pyOut = runGeneratedPython2Image(code, matToRaw(matA), matToRaw(matB));

  assert.equal(pyOut.length, jsOut.data.length);
  // cv2.imread/imwrite ธรรมเนียม BGR แต่เรารันแบบ raw bytes ตรง ๆ (ไม่ผ่าน imread ที่สลับช่อง) จึง
  // เทียบ byte ต่อ byte ได้ตรง ๆ (absdiff เป็น per-channel ไม่สนใจว่าช่องไหนคือช่องอะไร)
  for (let i = 0; i < jsOut.data.length; i++) {
    assert.equal(pyOut[i], jsOut.data[i], `byte ${i} mismatch between JS and Python absdiff output`);
  }

  jsOut.delete();
  matA.delete();
  matB.delete();
});

// ยืนยันเชิงประจักษ์ (spec §G.1) ว่า trick \bimg\b -> img_b ปลอดภัยกับ GEN ที่ซับซ้อนสุดในไฟล์ (ตัวแปร
// local หลายตัว: f32/mean/mean_sq/variance/std/normed) — เรนเดอร์ local_std_filter เป็น src2 sub-pipeline
// ของ imabsdiff แล้วตรวจทั้ง (a) เนื้อโค้ดที่ generate ออกมา ต้อง reference "img_b" ไม่ใช่ "img" เปล่า ๆ
// ในจุดที่ควรเป็นตัวแปรภาพ B และ (b) รันจริงผ่าน python ไม่ error (proves ตัวแปร local ไม่ชนกัน)
test('tier2-11b. codegen \\bimg\\b->img_b replace trick is safe against local_std_filter\'s multi-local-variable GEN body (runs correctly via real cv2)', () => {
  const w = 32, h = 24;
  const matA = solidRGB(w, h, 100, 100, 100);
  const matB = withRect(w, h, 50, { x: 8, y: 6, w: 12, h: 10 }, 200);

  const src2Step = { id: 'shape_analysis', params: { mode: 'local_std_filter', local_std_filter__windowSize: 9 }, enabled: true, nodeId: uniqueNodeId() };
  const pipeline = [{ id: 'image_arithmetic', params: { mode: 'imabsdiff' }, enabled: true, nodeId: uniqueNodeId(), src2: { pipeline: [src2Step] } }];
  const code = buildPythonCode(pipeline);

  // เนื้อโค้ดต้องมี "img_b = cv2.cvtColor(img_b, ...)" (gray auto-convert ของ src2 chain) และตัวแปร
  // ท้ายสุดของ local_std_filter ("img = np.rint(normed)...") ต้องถูกแทนที่เป็น "img_b = np.rint(normed)"
  assert.match(code, /img_b = cv2\.cvtColor\(img_b, cv2\.COLOR_BGR2GRAY\)/);
  assert.match(code, /img_b = np\.rint\(normed\)\.astype\(np\.uint8\)/);
  // ตัวแปร local ของ op (f32/mean/mean_sq/variance/std/normed) ต้อง "ไม่ถูกแทนที่" (ไม่มี suffix _b ผิด ๆ)
  assert.match(code, /mean_sq = cv2\.boxFilter\(f32 \* f32/);
  assert.doesNotMatch(code, /mean_sq_b|f32_b|variance_b|std_b|normed_b/);

  // รันจริงผ่าน python เพื่อพิสูจน์ไม่ NameError/SyntaxError จาก replace trick (ความถูกต้องเชิงตัวเลข
  // เต็มรูปแบบของ local_std_filter เองมีเทสต์ parity แยกอยู่แล้วในรอบก่อนหน้า — ที่นี่เจาะจงพิสูจน์แค่
  // ว่า "การซ้อนมันเข้าไปใน src2 ของ imabsdiff" ไม่ทำให้โค้ดพัง)
  const pyOut = runGeneratedPython2Image(code, matToRaw(matA), matToRaw(matB));
  assert.equal(pyOut.length, w * h * 3);

  matA.delete();
  matB.delete();
});

test('tier2-11c. CV-eng recheck: GEN.imabsdiff intentionally stays "always BGR" even though registry.js\'s imabsdiff.apply() now has a gray-preserving fast-path — documents+proves this deliberate JS-vs-Python divergence is safe (same numeric values, no downstream crash) rather than an oversight', () => {
  const w = 32, h = 24;
  const rect = { x: 6, y: 5, w: 10, h: 8 };
  // ภาพต้นฉบับเป็นสี (cv2.imread ให้ 3-channel เสมอ) แต่ pipeline มี Grayscale ก่อนถึง imabsdiff ทั้งสองสาย
  // เพื่อให้ทั้ง A และ B เป็นขาวดำตอนถึง imabsdiff จริง ๆ (trigger fast-path ฝั่ง JS)
  const matA = withRect(w, h, 40, rect, 210);
  const matB = withRect(w, h, 40, rect, 60);

  const pipeline = [
    { id: 'color_space', params: { mode: 'grayscale' }, enabled: true, nodeId: uniqueNodeId() },
    {
      id: 'image_arithmetic', params: { mode: 'imabsdiff' }, enabled: true, nodeId: uniqueNodeId(),
      src2: { pipeline: [{ id: 'color_space', params: { mode: 'grayscale' }, enabled: true, nodeId: uniqueNodeId() }] },
    },
    // downstream gray-requiring step — this is exactly the scenario that would crash if GEN.imabsdiff
    // had been changed to emit a 1-channel output while ctx.gray stayed compile-time "color"
    { id: 'threshold', params: { mode: 'binary_threshold', binary_threshold__thresh: 127, binary_threshold__invert: false }, enabled: true, nodeId: uniqueNodeId() },
  ];
  const code = buildPythonCode(pipeline);
  assert.doesNotThrow(() => {
    const pyOut = runGeneratedPython2Image(code, matToRaw(matA), matToRaw(matB));
    assert.equal(pyOut.length, w * h, 'binary_threshold output should be 1-channel (proves the chain ran end-to-end without a channel-mismatch crash)');
  }, 'expected the generated Python (imabsdiff always-BGR + a downstream gray-requiring step) to run without a cv2.cvtColor channel-count error');

  // separately: confirm the JS side's gray fast-path and the Python side's always-BGR path agree on the
  // *values* (R=G=B in Python must equal the JS gray value) — proves the divergence is representation-only
  const grayA = OPS.grayscale.apply(cv, matA);
  const grayB = OPS.grayscale.apply(cv, matB);
  const jsOut = OPS.imabsdiff.apply(cv, grayA, {}, { src2: grayB });
  assert.equal(jsOut.channels(), 1, 'sanity: JS fast-path should indeed produce 1-channel here');

  const codeNoThreshold = buildPythonCode([
    { id: 'color_space', params: { mode: 'grayscale' }, enabled: true, nodeId: uniqueNodeId() },
    {
      id: 'image_arithmetic', params: { mode: 'imabsdiff' }, enabled: true, nodeId: uniqueNodeId(),
      src2: { pipeline: [{ id: 'color_space', params: { mode: 'grayscale' }, enabled: true, nodeId: uniqueNodeId() }] },
    },
  ]);
  const pyOutColor = runGeneratedPython2Image(codeNoThreshold, matToRaw(matA), matToRaw(matB));
  assert.equal(pyOutColor.length, w * h * 3, 'sanity: Python side should indeed still be 3-channel here (the documented divergence)');
  for (let i = 0; i < w * h; i++) {
    const jsVal = jsOut.data[i];
    const pyR = pyOutColor[i * 3], pyG = pyOutColor[i * 3 + 1], pyB = pyOutColor[i * 3 + 2];
    assert.equal(pyR, jsVal, `pixel ${i}: Python R channel must equal JS gray value`);
    assert.equal(pyG, jsVal, `pixel ${i}: Python G channel must equal JS gray value`);
    assert.equal(pyB, jsVal, `pixel ${i}: Python B channel must equal JS gray value`);
  }

  jsOut.delete(); grayA.delete(); grayB.delete();
  matA.delete(); matB.delete();
});

// ===================================================================
// §wiring — palette/registry wiring sanity (เหมือนแนวเดียวกับไฟล์เทสต์อื่น ๆ ในโปรเจกต์)
// ===================================================================
test('tier2-wiring. imabsdiff/image_arithmetic wired into FILTERS, TOOL_SECTIONS, GROUP_COLORS/GROUP_ORDER', async () => {
  assert.ok(FILTERS.image_arithmetic, 'expected a registered "image_arithmetic" FILTERS group');
  assert.equal(FILTERS.image_arithmetic.group, 'Image Arithmetic');
  assert.equal(FILTERS.image_arithmetic.inputs, 2);
  const modeParam = FILTERS.image_arithmetic.params.find((p) => p.key === 'mode');
  assert.ok(modeParam && modeParam.options.some((o) => o.value === 'imabsdiff'));
  assert.ok(flattenPaletteOps().includes('imabsdiff'), 'expected imabsdiff listed in some TOOL_SECTIONS entry');

  const { GROUP_COLORS, GROUP_ORDER } = await import('../src/renderer/filters/registry.js');
  assert.ok(GROUP_COLORS['Image Arithmetic'], 'expected a GROUP_COLORS entry for "Image Arithmetic"');
  assert.ok(GROUP_ORDER.includes('Image Arithmetic'));
});

test('tier2-wiring2. every other pre-existing FILTERS entry is unaffected by the mergeGroup opts/inputs change (inputs is undefined everywhere except the 2-image groups)', () => {
  // อัปเดตรอบ 18b: image_arithmetic (18a) + template_matching/psnr/ssim (18b ใหม่) เป็น inputs:2 โดยตั้งใจ
  // — ทุก FILTERS entry อื่นต้อง inputs undefined เหมือนเดิมไม่เปลี่ยน (ยืนยันว่า opts ไม่รั่วไปโดนกลุ่มอื่น)
  const TWO_IMAGE_GROUPS = new Set(['image_arithmetic', 'template_matching', 'psnr', 'ssim']);
  for (const [id, def] of Object.entries(FILTERS)) {
    if (TWO_IMAGE_GROUPS.has(id)) continue;
    assert.equal(def.inputs, undefined, `${id}.inputs must remain undefined (only image_arithmetic/template_matching/psnr/ssim get inputs:2)`);
  }
});

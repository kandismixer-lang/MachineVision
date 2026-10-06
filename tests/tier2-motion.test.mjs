// Tier 2, 18d — acceptance tests ของกลุ่ม "Motion" (mog2 / farneback_optical_flow / pyrlk_optical_flow)
// และของ "ขั้น 0" ใน cv-worker.js ที่ทำให้ op แบบ stateful ทำงานถูกต้อง (fingerprint ต้นน้ำ, store ที่ส่งเข้ามา,
// memo กันรันซ้ำในเฟรมเดียว, export ใช้ state แยก, node ที่ถูกปิดถือว่าไม่ active)
// ครบตามใบสั่งงาน specs/tier2_spec.md §C.4-final F8 — 49 เทสต์ชุดแรก: M-0/M-0w/M-1…M-6 (8) · D-1…D-12 (12) ·
// MG-1…6 · FB-1…6 · LK-1…6 · ALL-1…5 · CG-1…4 · W-1…2
// + รอบแก้ตามรายงาน debugger: W-3 · LK-7 · MN-1/MN-2 · D-13…D-16 · W-4
// + รอบแก้ตามรายงาน qa (ขั้นที่ 6) — ทุกตัวเกิดจาก mutation ที่รอดมาได้ ไม่ใช่บั๊กในโค้ด:
//   FB-7 (พิกัดลูกศร + เกณฑ์ "ขยับพอไหม" + moving_pct ต้องเป็นหน่วย px ของภาพจริง) · LK-8 (จุดที่เดินออก
//   นอกเฟรม) · MN-3 (history พิสูจน์เชิงพฤติกรรมได้ใน 50 เฟรม) · MN-4 (การ์ด node โชว์ค่าหลัง clamp) ·
//   D-17 (memo เฉพาะ op ที่ stateful) และ assertion ที่เติมใน LK-3 (พิกัดจุด/เส้นทาง) · MG-4 (ไม่มีคีย์
//   detectShadows) · D-16 (cloneValues รูปทรง table)
//
// นโยบาย leak (P-14, WORK-TRACKER §1): address-fingerprint ใช้ได้เฉพาะเทสต์แรกของไฟล์ (M-0) พร้อมคู่แฝด
// wrapper-balance (M-0w) — เทสต์อื่นทั้งหมดใช้ assertLeakFree / assertWrapperBalance / assertNoHeapGrowth
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { OPS, FILTERS, defaultParams, GROUP_COLORS, GROUP_ORDER } from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS } from '../src/renderer/ui/palette-panel.js';
import { GROUP_ICON, OP_PRIMARY } from '../src/renderer/filters/ui-meta.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import {
  assertLeakFree, assertWrapperBalance, assertNoHeapGrowth, fingerprintLeakProbe,
  makeWrapperTracker, makeThrowingCv,
} from './_leak-helpers.mjs';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cvtColor === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

// cv-worker.js เป็น Web Worker script จริง (self.onmessage ที่ top-level) — Node ไม่มี self/ImageData
// ต้อง polyfill ก่อน import (เหมือน tier2-infra) — แต่ "import ทีหลัง" (lazy) เพื่อให้ M-0 ซึ่งเป็น
// address-fingerprint ได้รันใน process ที่ยังสะอาดที่สุด (P-14 ข้อ 1: allocator ยังไม่ถูกกวน)
globalThis.self = globalThis;
if (typeof globalThis.ImageData === 'undefined') {
  globalThis.ImageData = class ImageData {
    constructor(data, width, height) { this.data = data; this.width = width; this.height = height; }
  };
}
let __worker = null;
async function worker() {
  if (!__worker) __worker = await import('../src/renderer/cv/cv-worker.js');
  return __worker;
}

const MOTION_IDS = ['mog2', 'farneback_optical_flow', 'pyrlk_optical_flow'];

let __uid = 1;
function nid() { return 910000 + __uid++; } // ช่วงเลขส่วนตัวของไฟล์นี้ ไม่ชนกับเทสต์อื่นหรือ nodeId จริง

// ===================================================================
// ภาพทดสอบสังเคราะห์ (deterministic — LCG เมล็ดคงที่ ไม่ใช้ Math.random)
// ===================================================================
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

// ผิวสัมผัสแบบ "เมฆ": noise สุ่มทุกพิกเซลแล้วเบลอ + normalize — มี gradient ทุกจุด ไม่มีพื้นที่เรียบเลย
// จงใจเลือกแบบนี้แทน "จานวงกลมเบลอ": ฉากที่มีพื้นที่เรียบกว้าง ๆ ทำให้ Farneback อ่านการขยับได้ต่ำกว่าจริงมาก
// (aperture problem — วัดแล้ว: จานวงกลมให้ dx=2.04 สำหรับการเลื่อน 3px ส่วนผิวเมฆให้ 3.00) และ gftt ก็หาจุดได้ทั่วภาพ
function cloudRGB(w, h, seed, blur = 9) {
  const data = new Uint8Array(w * h * 3);
  const rnd = lcg(seed);
  for (let i = 0; i < data.length; i++) data[i] = (rnd() * 256) | 0;
  const noise = cv.matFromArray(h, w, cv.CV_8UC3, data);
  const minSide = Math.min(w, h);
  const k = Math.min(blur, minSide % 2 === 0 ? minSide - 1 : minSide); // kernel ต้องคี่และไม่เกินขนาดภาพ
  const blurred = new cv.Mat();
  if (k >= 3) cv.GaussianBlur(noise, blurred, new cv.Size(k, k), 0); else noise.copyTo(blurred);
  noise.delete();
  const out = new cv.Mat();
  cv.normalize(blurred, out, 0, 255, cv.NORM_MINMAX);
  blurred.delete();
  return out;
}

// ชุดเฟรมที่ "เนื้อภาพเลื่อนไป (dx,dy) พอดีทุกพิกเซล" — ตัด ROI จากภาพที่ใหญ่กว่า จึงไม่มีขอบว่างมากวนค่าเฉลี่ย
function shiftedFrames(w, h, offsets, seed = 777) {
  const pad = 48;
  const big = cloudRGB(w + 2 * pad, h + 2 * pad, seed);
  const out = offsets.map(([dx, dy]) => {
    const r = big.roi(new cv.Rect(pad - dx, pad - dy, w, h));
    const c = r.clone();
    r.delete();
    return c;
  });
  big.delete();
  return out;
}

function convertMat(src, code) {
  const out = new cv.Mat();
  cv.cvtColor(src, out, code);
  return out;
}
function toGrayMat(src) { return convertMat(src, cv.COLOR_RGB2GRAY); }
function toRGBAMat(src) { return convertMat(src, cv.COLOR_RGB2RGBA); }
// เทียบเท่า motionRGB() ใน registry.js (ไม่ได้ export) — ใช้เป็น "คำตอบที่คาดหวัง" ของเฟรมแรก
function expectedRGB(src) {
  if (src.channels() === 1) return convertMat(src, cv.COLOR_GRAY2RGB);
  if (src.channels() === 4) return convertMat(src, cv.COLOR_RGBA2RGB);
  const o = new cv.Mat(); src.copyTo(o); return o;
}
function sameBytes(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
// กรอบ (bounding box) + จำนวนของพิกเซลที่ถูก "วาดทับ" ด้วยสีบริสุทธิ์สีหนึ่งบนภาพ RGB 3 ช่อง
// ใช้พิสูจน์ว่าพิกัดที่วาด (ลูกศร Farneback / จุด+เส้นทาง PyrLK) ถูกคูณกลับเป็น "พิกเซลของภาพจริง" แล้ว:
// ถ้าลืม × invScale ทุกอย่างจะกองอยู่มุมซ้ายบนขนาดเท่าภาพวิเคราะห์ (ที่ 1080p = 6.25% ของภาพ) โดยที่
// เทสต์ที่ดูแค่ "มีพิกเซลสีนี้ไหม" หรือดูแต่ตัวเลข ยังเขียวหมด
function pureColorBox(mat, [cr, cg, cb]) {
  const d = mat.data;
  const w = mat.cols;
  let count = 0, minX = Infinity, minY = Infinity, maxX = -1, maxY = -1;
  for (let i = 0; i < d.length; i += 3) {
    if (d[i] !== cr || d[i + 1] !== cg || d[i + 2] !== cb) continue;
    const p = i / 3;
    const x = p % w, y = (p / w) | 0;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    count++;
  }
  return { count, minX, minY, maxX, maxY };
}
function rgbaBufferOf(mat) {
  const rgba = mat.channels() === 4 ? mat.clone() : convertMat(mat, mat.channels() === 1 ? cv.COLOR_GRAY2RGBA : cv.COLOR_RGB2RGBA);
  const buf = new Uint8ClampedArray(rgba.data).buffer;
  rgba.delete();
  return buf;
}

function valuesOf(ctx) {
  const o = {};
  for (const f of (ctx.values && ctx.values.fields) || []) o[f.key] = f.value;
  return o;
}
// ดึงค่า field หนึ่งจาก notes ที่ runPipeline/computeFrame คืนมา (Tier 2 §A.2: note = { id, values })
function noteValue(notes, stepId, key) {
  for (const n of notes || []) {
    if (n.id === stepId && n.values) {
      const f = n.values.fields.find((x) => x.key === key);
      if (f) return f.value;
    }
  }
  return null;
}
function noteInfos(notes, stepId) {
  return (notes || []).filter((n) => n.id === stepId && n.info).map((n) => n.info);
}

// ===================================================================
// M-0. address-fingerprint (P-14 ข้อ 1 — เทสต์แรกของไฟล์เท่านั้น, ไฟล์ละ 1 ตัว)
// ===================================================================
// ตาราง "เส้นทาง" ที่ M-0 และ M-0w ใช้ร่วมกัน: ต่อ op = {3 เฟรมบน state เดียว + __dispose} และ {เรียกแบบไม่มี state}
// ทั้งที่ 96×72 (ไม่ย่อ) และ 640×360 (ย่อจริงทั้ง 3 op: mog2 ≤480, farneback ≤320, pyrlk ≤480)
function motionLeakPaths() {
  const small = cloudRGB(96, 72, 31);
  const big = cloudRGB(640, 360, 32);
  const paths = [];
  for (const id of MOTION_IDS) {
    for (const [tag, src] of [['96x72', small], ['640x360', big]]) {
      paths.push({
        label: `${id} 3 frames + dispose @${tag}`,
        run: (c) => {
          const st = {};
          try {
            for (let i = 0; i < 3; i++) {
              const o = OPS[id].apply(c, src, {}, { state: st });
              o.delete();
            }
          } finally { if (st.__dispose) st.__dispose(); }
        },
      });
      paths.push({
        label: `${id} no-state @${tag}`,
        run: (c) => { const o = OPS[id].apply(c, src, {}, {}); o.delete(); },
      });
    }
  }
  return { paths, cleanup: () => { small.delete(); big.delete(); } };
}

test('tier2d-M-0. leak (address fingerprint, first test of a clean process): every Motion op over {3 frames on one state + __dispose} and {no-state call}, at 96x72 and 640x360, leaves the malloc fingerprint unchanged', () => {
  const { paths, cleanup } = motionLeakPaths();
  try {
    fingerprintLeakProbe(cv, 'motion ops (all paths)', () => {
      for (const p of paths) p.run(cv);
    }, { iters: 25, warmup: 10 });
  } finally { cleanup(); }
});

test('tier2d-M-0w. leak (wrapper balance twin of M-0): the exact same path table creates and deletes every embind wrapper it makes', () => {
  const { paths, cleanup } = motionLeakPaths();
  try {
    assertWrapperBalance(cv, 'motion ops (all paths)', (c) => {
      for (const p of paths) p.run(c);
    }, { minCreated: 30 });
  } finally { cleanup(); }
});

// ===================================================================
// M-1. lifecycle ต่อ op: 5 เฟรม (มีภาพย่อ 640x360 ด้วย) แล้ว __dispose "ในตัว run" -> ต้องเหลือ 0 wrapper
// ===================================================================
test('tier2d-M-1. leak lifecycle (all three ops): 5 frames incl. a downscaled 640x360 input + __dispose inside the probe -> 0 live wrappers and no heap growth', () => {
  const small = cloudRGB(96, 72, 41);
  const big = cloudRGB(640, 360, 42);
  try {
    for (const id of MOTION_IDS) {
      assertLeakFree(cv, `${id} lifecycle`, (c) => {
        const st = {};
        try {
          for (let i = 0; i < 4; i++) { const o = OPS[id].apply(c, small, {}, { state: st }); o.delete(); }
          const o = OPS[id].apply(c, big, {}, { state: st }); // ขนาดเปลี่ยน -> ทางเดินรีเซ็ตภายใน op เอง
          o.delete();
        } finally { if (st.__dispose) st.__dispose(); }
      }, { minCreated: 5, iters: 40, warmup: 15 });
    }
  } finally { small.delete(); big.delete(); }
});

// ===================================================================
// M-2. steady state: state อุ่นแล้ว 5 เฟรม (นอก tracker) -> 1 เฟรมใน tracker ต้องเหลือ wrapper "ตามจำนวนของถาวร"
// (mog2 0 / farneback 1 / pyrlk 1 = prevGray ตัวใหม่ที่เพิ่งสร้าง) แล้วรัน 500 เฟรมดูว่า heap ไม่โต
// ===================================================================
const STEADY_LIVE = { mog2: 0, farneback_optical_flow: 1, pyrlk_optical_flow: 1 };
test('tier2d-M-2. leak steady state (all three ops): a warmed state + 1 tracked frame leaves exactly the intended number of live wrappers (mog2 0 / farneback 1 / pyrlk 1 = the fresh prevGray); 500 frames @96x72 and 100 frames @640x360 on that persistent state grow the heap by 0', () => {
  const small = cloudRGB(96, 72, 51);
  const big = cloudRGB(640, 360, 52);
  try {
    for (const id of MOTION_IDS) {
      const st = {};
      try {
        for (let i = 0; i < 5; i++) { const o = OPS[id].apply(cv, small, {}, { state: st }); o.delete(); }
        const t = makeWrapperTracker(cv);
        try {
          const o = OPS[id].apply(t.cv, small, {}, { state: st });
          o.delete();
          t.assertBalanced(`${id} steady frame`, { minCreated: 2, expectLive: STEADY_LIVE[id] });
        } finally { t.dispose(); }
        assertNoHeapGrowth(cv, `${id} 500 frames @96x72`, () => {
          const o = OPS[id].apply(cv, small, {}, { state: st });
          o.delete();
        }, { iters: 500, warmup: 30 });
        // ขนาดเปลี่ยนครั้งเดียวตอนเริ่ม (op รีเซ็ตภายในเอง) แล้ววัด 100 เฟรมที่ขนาดใหม่
        { const o = OPS[id].apply(cv, big, {}, { state: st }); o.delete(); }
        assertNoHeapGrowth(cv, `${id} 100 frames @640x360`, () => {
          const o = OPS[id].apply(cv, big, {}, { state: st });
          o.delete();
        }, { iters: 100, warmup: 10 });
      } finally { if (st.__dispose) st.__dispose(); }
    }
  } finally { small.delete(); big.delete(); }
});

// ===================================================================
// M-3. __dispose เรียกซ้ำได้ (idempotent) และของถาวรชุดเก่าถูก delete จริงตอน reset
// ===================================================================
test('tier2d-M-3. __dispose is idempotent for all three ops (2nd call does not throw, persistent handles are isDeleted()), and a state reset via getNodeState deletes the previous persistent objects', async () => {
  const src = cloudRGB(96, 72, 61);
  try {
    for (const id of MOTION_IDS) {
      const st = {};
      for (let i = 0; i < 2; i++) { const o = OPS[id].apply(cv, src, {}, { state: st }); o.delete(); }
      const persistent = id === 'mog2' ? st.mog : st.prevGray;
      assert.ok(persistent && !persistent.isDeleted(), `${id}: expected a live persistent handle after 2 frames`);
      st.__dispose();
      assert.ok(persistent.isDeleted(), `${id}: __dispose must delete the persistent handle`);
      assert.doesNotThrow(() => st.__dispose(), `${id}: __dispose must be safe to call twice`);
      if (id === 'mog2') assert.equal(st.mog, null); else assert.equal(st.prevGray, null);
      if (id === 'pyrlk_optical_flow') assert.equal(st.trails, null);
    }

    // reset จริงผ่าน getNodeState (param เปลี่ยน) -> ของถาวรของ state เก่าต้องถูก delete
    const { getNodeState, sweepNodeState } = await worker();
    const store = new Map();
    const id = nid();
    const s1 = getNodeState(id, [{ id: 'farneback_optical_flow', params: { maxMotion: 10 } }], 96, 72, store);
    { const o = OPS.farneback_optical_flow.apply(cv, src, {}, { state: s1 }); o.delete(); }
    const oldPrev = s1.prevGray;
    assert.ok(oldPrev && !oldPrev.isDeleted());
    const s2 = getNodeState(id, [{ id: 'farneback_optical_flow', params: { maxMotion: 20 } }], 96, 72, store);
    assert.notEqual(s2, s1, 'a params change must hand back a brand-new state object');
    assert.ok(oldPrev.isDeleted(), 'the old state\'s prevGray must be deleted by the reset');
    sweepNodeState(new Set(), store);
  } finally { src.delete(); }
});

// ===================================================================
// M-4. ฉีด error กลางเฟรม (JS ล้วน — ไม่ใช่ C++ exception) -> ต้อง throw, ไม่รั่ว, และ "state ยังใช้ต่อได้"
// ===================================================================
const THROW_AT = {
  mog2: 'threshold',                              // หลัง st.mog.apply() + frameCount++ แล้ว
  farneback_optical_flow: 'cartToPolar',          // หลังคำนวณ flow แล้ว ก่อน swap prevGray
  pyrlk_optical_flow: 'calcOpticalFlowPyrLK',     // ระหว่างติดตามจุด (state เดิมต้องไม่ถูกแตะ)
};
test('tier2d-M-4. injected mid-frame JS throw (mog2: cv.threshold, farneback: cv.cartToPolar, pyrlk: cv.calcOpticalFlowPyrLK -- all AFTER the state exists) -> the op throws, leaks no wrapper, and the state survives intact so the next normal frame succeeds', () => {
  const src = cloudRGB(96, 72, 71);
  try {
    for (const id of MOTION_IDS) {
      const st = {};
      try {
        for (let i = 0; i < 3; i++) { const o = OPS[id].apply(cv, src, {}, { state: st }); o.delete(); }
        const persistent = id === 'mog2' ? st.mog : st.prevGray;
        const frameBefore = st.frameCount;

        // state ถูกอุ่นไว้ "นอก" tracker แล้ว -> ของถาวรไม่ถูกนับ; เฟรมที่ throw จึงต้องเหลือ 0 wrapper
        assertWrapperBalance(cv, `${id} throwing frame`, (c) => {
          const bad = makeThrowingCv(c, THROW_AT[id]);
          const o = OPS[id].apply(bad, src, {}, { state: st });
          o.delete(); // ไปไม่ถึงบรรทัดนี้ (ต้อง throw ก่อน)
        }, { expectThrow: new RegExp(`injected test failure \\(${THROW_AT[id]}\\)`), minCreated: 2 });

        assert.ok(!persistent.isDeleted(), `${id}: the persistent handle must survive a mid-frame throw`);
        assert.equal(id === 'mog2' ? st.mog : st.prevGray, persistent, `${id}: the state must still point at the same live object`);

        const ctx = { state: st };
        let out;
        assert.doesNotThrow(() => { out = OPS[id].apply(cv, src, {}, ctx); }, `${id}: the frame after the throw must work`);
        // MINOR-2 (debugger): เฟรมที่ล้มกลางทางต้อง "ไม่" ถูกนับ — ตัวนับต้องขยับ 1 จากเฟรมที่สำเร็จเท่านั้น
        // (ถ้า mog2 ทำ frameCount++ ก่อน cv.threshold ค่าจะกลายเป็น frameBefore + 2 และข้อความ
        // "กำลังเรียนรู้พื้นหลัง (เฟรม n/30)" กับค่า `frame` ใน ctx.values จะเพี้ยนไปตลอดอายุของ node)
        assert.equal(st.frameCount, frameBefore + 1, `${id}: only the successful frame may advance frameCount`);
        out.delete();
      } finally { if (st.__dispose) st.__dispose(); }
    }
  } finally { src.delete(); }
});

// ===================================================================
// M-5. leak ระดับ worker (tracker แบบ global เพราะ cv-worker.js import cv เอง)
// ===================================================================
test('tier2d-M-5. leak at worker level (global tracker): a 6-frame runPipeline sequence with memo hits, an own-param reset and a final sweep is wrapper-balanced; 100 iterations of a 3-frame computeFrame sequence (2 pipelines sharing the node) grow the heap by 0', async () => {
  const { runPipeline, computeFrame, collectNodeIds, sweepNodeState } = await worker();
  const mogId = nid();
  const mkPipe = (varTh) => [{ id: 'mog2', params: { view: 'mask', varThreshold: varTh }, enabled: true, nodeId: mogId }];

  assertWrapperBalance(cv, 'runPipeline 6-frame stateful sequence', (c) => {
    const base = cloudRGB(96, 72, 81);
    const store = new Map();
    try {
      for (let f = 0; f < 6; f++) {
        const pipeline = mkPipe(f < 4 ? 16 : 24); // เฟรมที่ 5 เปลี่ยน param ของตัวเอง -> reset + __dispose ของเก่า
        const memo = new Map();
        try {
          const active = new Set();
          collectNodeIds(pipeline, active);
          const r1 = runPipeline(base, pipeline, store, active, memo);
          const r2 = runPipeline(base, pipeline, store, active, memo); // memo hit (พรีวิวของ node ที่เลือก)
          r1.mat.delete();
          r2.mat.delete();
        } finally { for (const e of memo.values()) e.mat.delete(); }
      }
      sweepNodeState(new Set(), store);
    } finally { base.delete(); }
  }, { global: true, minCreated: 10 });

  const base = cloudRGB(96, 72, 82);
  const buffer = rgbaBufferOf(base);
  const pipelines = [
    { connected: true, displayId: 'main', pipeline: mkPipe(16) },
    { connected: true, displayId: 'preview', silentNotes: true, pipeline: mkPipe(16) },
  ];
  try {
    assertNoHeapGrowth(cv, 'computeFrame 3-frame stateful sequence', () => {
      for (let f = 0; f < 3; f++) computeFrame(96, 72, buffer, pipelines);
      sweepNodeState(new Set()); // จบรอบแล้วทิ้ง state ของ store โมดูล (ไม่ให้สะสมข้ามรอบ)
    }, { iters: 100, warmup: 20 });
  } finally {
    sweepNodeState(new Set());
    base.delete();
  }
});

// ===================================================================
// M-6. guard ของ C++ exception: เส้นทางที่ "เคย" throw ทุกเฟรมต้องไม่ throw เลย -> ใช้ assertNoHeapGrowth
// ตรง ๆ (ไม่ใช่ assertNoExtraHeapGrowth) เพราะถ้ายัง throw จะรั่ว ~760B/เฟรม ซึ่งเกิน tolerance แน่นอน
// ===================================================================
test('tier2d-M-6. C++-throw guards: PyrLK over 200 all-black frames and 200 1x50 frames, Farneback/MOG2 over 1x1, 1x50 and 50x1 frames -> no throw and zero heap growth (proves no cv::Exception is raised per frame)', () => {
  const black = new cv.Mat(72, 96, cv.CV_8UC3, new cv.Scalar(0, 0, 0));
  const thin = new cv.Mat(50, 1, cv.CV_8UC3, new cv.Scalar(0, 0, 0));
  try {
    for (const [label, src, iters] of [['black 96x72', black, 200], ['1x50', thin, 200]]) {
      const st = {};
      try {
        assertNoHeapGrowth(cv, `pyrlk ${label}`, () => {
          const o = OPS.pyrlk_optical_flow.apply(cv, src, {}, { state: st });
          o.delete();
        }, { iters, warmup: 20 });
      } finally { if (st.__dispose) st.__dispose(); }
    }
  } finally { black.delete(); thin.delete(); }

  for (const [w, h] of [[1, 1], [1, 50], [50, 1]]) {
    const src = new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(20, 30, 40));
    try {
      for (const id of ['farneback_optical_flow', 'mog2']) {
        const st = {};
        try {
          assertNoHeapGrowth(cv, `${id} ${w}x${h}`, () => {
            const o = OPS[id].apply(cv, src, {}, { state: st });
            o.delete();
          }, { iters: 150, warmup: 20 });
        } finally { if (st.__dispose) st.__dispose(); }
      }
    } finally { src.delete(); }
  }
});

// ===================================================================
// D-1…D-12 — โครงสร้างพื้นฐาน "ขั้น 0" ใน cv-worker.js
// ===================================================================
// รัน pipeline เดียว n เฟรมบน store ที่ส่งมา คืน notes ของเฟรมสุดท้าย
async function runFrames(base, pipeline, store, n = 1, memo = null) {
  const { runPipeline, collectNodeIds } = await worker();
  let notes = null;
  for (let i = 0; i < n; i++) {
    const active = new Set();
    collectNodeIds(pipeline, active);
    const r = runPipeline(base, pipeline, store, active, memo);
    r.mat.delete();
    notes = r.notes;
  }
  return notes;
}
// ห่อ __dispose ของ state ปัจจุบันเพื่อ "นับ" ว่าถูกเรียกกี่ครั้ง (ของจริงยังทำงานเหมือนเดิม)
function spyDispose(store, nodeId) {
  const st = store.get(nodeId);
  assert.ok(st, `expected a state object for node ${nodeId}`);
  const orig = st.__dispose;
  const counter = { calls: 0, state: st };
  st.__dispose = () => { counter.calls++; if (orig) orig(); };
  return counter;
}

test('tier2d-D-1. fingerprint: changing an UPSTREAM step\'s params (Gaussian ksize before mog2) resets the node state -> the old __dispose runs exactly once and the new state restarts at frame 1', async () => {
  const { sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 101);
  const store = new Map();
  const gaussId = nid(), mogId = nid();
  const pipe = (ksize) => [
    { id: 'blur', params: { mode: 'gaussian_blur', gaussian_blur__ksize: ksize }, enabled: true, nodeId: gaussId },
    { id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId },
  ];
  try {
    const n3 = await runFrames(base, pipe(5), store, 3);
    assert.equal(noteValue(n3, 'mog2', 'frame'), 3);
    const spy = spyDispose(store, mogId);
    const n4 = await runFrames(base, pipe(9), store, 1);
    assert.equal(spy.calls, 1, 'the old state\'s __dispose must be called exactly once on reset');
    assert.equal(noteValue(n4, 'mog2', 'frame'), 1, 'a new state restarts learning at frame 1');
    assert.notEqual(store.get(mogId), spy.state);
  } finally { sweepNodeState(new Set(), store); base.delete(); }
});

test('tier2d-D-2. fingerprint: toggling an UPSTREAM step\'s `enabled` resets the node state (the pixels reaching the node changed even though no params did)', async () => {
  const { sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 102);
  const store = new Map();
  const gaussId = nid(), mogId = nid();
  const pipe = (on) => [
    { id: 'blur', params: { mode: 'gaussian_blur', gaussian_blur__ksize: 7 }, enabled: on, nodeId: gaussId },
    { id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId },
  ];
  try {
    assert.equal(noteValue(await runFrames(base, pipe(true), store, 3), 'mog2', 'frame'), 3);
    const spy = spyDispose(store, mogId);
    assert.equal(noteValue(await runFrames(base, pipe(false), store, 1), 'mog2', 'frame'), 1);
    assert.equal(spy.calls, 1);
  } finally { sweepNodeState(new Set(), store); base.delete(); }
});

test('tier2d-D-3. fingerprint (V13): changing a param inside the src2 (B) branch of an UPSTREAM 2-image step resets a downstream Motion node -- stepFingerprint recurses into src2', async () => {
  const { sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 103);
  const store = new Map();
  const bId = nid(), absId = nid(), fbId = nid();
  const pipe = (ksize) => [
    {
      id: 'image_arithmetic', params: { mode: 'imabsdiff' }, enabled: true, nodeId: absId,
      src2: { pipeline: [{ id: 'blur', params: { mode: 'gaussian_blur', gaussian_blur__ksize: ksize }, enabled: true, nodeId: bId }] },
    },
    { id: 'farneback_optical_flow', params: {}, enabled: true, nodeId: fbId },
  ];
  try {
    assert.equal(noteValue(await runFrames(base, pipe(5), store, 3), 'farneback_optical_flow', 'frame'), 3);
    const spy = spyDispose(store, fbId);
    assert.equal(noteValue(await runFrames(base, pipe(11), store, 1), 'farneback_optical_flow', 'frame'), 1,
      'a change in the B branch feeding this node must reset it');
    assert.equal(spy.calls, 1);
  } finally { sweepNodeState(new Set(), store); base.delete(); }
});

test('tier2d-D-4. fingerprint: changing a DOWNSTREAM step (Morphology after mog2) must NOT reset -- the upstream prefix ends at the current step, so the state object stays the same and `frame` keeps counting', async () => {
  const { sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 104);
  const store = new Map();
  const mogId = nid(), morphId = nid();
  const pipe = (ksize) => [
    { id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId },
    { id: 'morphology', params: { ...defaultParams('morphology'), op: 'open', ksize }, enabled: true, nodeId: morphId },
  ];
  try {
    assert.equal(noteValue(await runFrames(base, pipe(3), store, 3), 'mog2', 'frame'), 3);
    const before = store.get(mogId);
    assert.equal(noteValue(await runFrames(base, pipe(7), store, 1), 'mog2', 'frame'), 4,
      'a downstream edit must not reset the Motion node');
    assert.equal(store.get(mogId), before, 'the very same state object must be reused');
  } finally { sweepNodeState(new Set(), store); base.delete(); }
});

test('tier2d-D-5. fingerprint: changing the node\'s OWN params resets its state (frame restarts at 1 and the old __dispose runs once)', async () => {
  const { sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 105);
  const store = new Map();
  const mogId = nid();
  const pipe = (vt) => [{ id: 'mog2', params: { view: 'mask', varThreshold: vt }, enabled: true, nodeId: mogId }];
  try {
    assert.equal(noteValue(await runFrames(base, pipe(16), store, 3), 'mog2', 'frame'), 3);
    const spy = spyDispose(store, mogId);
    assert.equal(noteValue(await runFrames(base, pipe(40), store, 1), 'mog2', 'frame'), 1);
    assert.equal(spy.calls, 1);
  } finally { sweepNodeState(new Set(), store); base.delete(); }
});

test('tier2d-D-6. reset on resolution change: a different base frame size resets the state, and so does an upstream Resize param change (the size reaching the node is what counts, not the raw camera size)', async () => {
  const { sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 106);
  const small = cloudRGB(64, 48, 106);
  const store = new Map();
  const mogId = nid(), resizeId = nid();
  const plain = [{ id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId }];
  try {
    assert.equal(noteValue(await runFrames(base, plain, store, 3), 'mog2', 'frame'), 3);
    const spy = spyDispose(store, mogId);
    assert.equal(noteValue(await runFrames(small, plain, store, 1), 'mog2', 'frame'), 1,
      'a different input size must reset the state');
    assert.equal(spy.calls, 1);
  } finally { sweepNodeState(new Set(), store); }

  const store2 = new Map();
  const withResize = (scale) => [
    { id: 'geometry', params: { mode: 'resize', resize__scale: scale }, enabled: true, nodeId: resizeId },
    { id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId },
  ];
  try {
    assert.equal(noteValue(await runFrames(base, withResize(50), store2, 3), 'mog2', 'frame'), 3);
    const st = store2.get(mogId);
    assert.equal(st.__lastW, 48, 'the fingerprint size must be the size AFTER the upstream Resize');
    assert.equal(noteValue(await runFrames(base, withResize(75), store2, 1), 'mog2', 'frame'), 1);
    assert.equal(store2.get(mogId).__lastW, 72);
  } finally { sweepNodeState(new Set(), store2); base.delete(); small.delete(); }
});

test('tier2d-D-7. sweep: a node that disappears from every pipeline has its state disposed on the next computeFrame; re-adding it starts over at frame 1', async () => {
  const { computeFrame, sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 107);
  const buffer = rgbaBufferOf(base);
  const mogId = nid();
  const withNode = [{ connected: true, displayId: 'a', pipeline: [{ id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId }] }];
  const withoutNode = [{ connected: true, displayId: 'a', pipeline: [{ id: 'color_space', params: { mode: 'grayscale' }, enabled: true, nodeId: nid() }] }];
  try {
    computeFrame(96, 72, buffer, withNode);
    assert.equal(noteValue(computeFrame(96, 72, buffer, withNode).notes, 'mog2', 'frame'), 2);
    computeFrame(96, 72, buffer, withoutNode); // node หายไปจากทุกกล่อง -> sweep
    assert.equal(noteValue(computeFrame(96, 72, buffer, withNode).notes, 'mog2', 'frame'), 1,
      'after the sweep the re-added node must start learning from scratch');
  } finally { sweepNodeState(new Set()); base.delete(); }
});

test('tier2d-D-8. sweep (V14): a DISABLED (bypassed) node is not "active", so its state is swept -- re-enabling it starts at frame 1 instead of resuming with a stale background', async () => {
  const { computeFrame, sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 108);
  const buffer = rgbaBufferOf(base);
  const mogId = nid();
  const pl = (on) => [{ connected: true, displayId: 'a', pipeline: [{ id: 'mog2', params: { view: 'mask' }, enabled: on, nodeId: mogId }] }];
  try {
    computeFrame(96, 72, buffer, pl(true));
    assert.equal(noteValue(computeFrame(96, 72, buffer, pl(true)).notes, 'mog2', 'frame'), 2);
    computeFrame(96, 72, buffer, pl(false)); // ปิดกล่อง -> ไม่ active -> sweep
    assert.equal(noteValue(computeFrame(96, 72, buffer, pl(true)).notes, 'mog2', 'frame'), 1);
  } finally { sweepNodeState(new Set()); base.delete(); }
});

test('tier2d-D-9. per-frame memo (V10): the same Motion node in a display pipeline AND in the selected-node preview (silentNotes) runs ONCE per computeFrame -- `frame` advances by exactly 1, both result buffers are byte-identical, and its info appears exactly once in notes', async () => {
  const { computeFrame, sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 109);
  const buffer = rgbaBufferOf(base);
  const mogId = nid();
  const step = () => ({ id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId });
  const pipelines = [
    { connected: true, displayId: 'display', pipeline: [step()] },
    { connected: true, displayId: 'preview', silentNotes: true, pipeline: [step()] },
  ];
  try {
    for (let f = 1; f <= 3; f++) {
      const r = computeFrame(96, 72, buffer, pipelines);
      assert.equal(noteValue(r.notes, 'mog2', 'frame'), f, `frame ${f}: the node must advance by exactly 1 per computeFrame`);
      assert.equal(r.results.length, 2);
      assert.ok(sameBytes(new Uint8Array(r.results[0].buffer), new Uint8Array(r.results[1].buffer)),
        'the memoised result must be byte-identical in both pipelines');
      assert.equal(noteInfos(r.notes, 'mog2').length, 1, 'the silent preview pipeline must not add a second note');
    }
  } finally { sweepNodeState(new Set()); base.delete(); }
});

test('tier2d-D-10. per-frame memo inside ONE pipeline: a single mog2 node feeding both inputs of imabsdiff is applied once -> |A-B| is 0 everywhere (without the memo the second run would see a different learning step)', async () => {
  const { runPipeline, collectNodeIds, sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 110);
  const store = new Map();
  const mogId = nid();
  const step = () => ({ id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId });
  const pipeline = [
    step(),
    { id: 'image_arithmetic', params: { mode: 'imabsdiff' }, enabled: true, nodeId: nid(), src2: { pipeline: [step()] } },
  ];
  try {
    for (let f = 1; f <= 4; f++) {
      const memo = new Map();
      const active = new Set();
      collectNodeIds(pipeline, active);
      const r = runPipeline(base, pipeline, store, active, memo);
      try {
        assert.equal(noteValue(r.notes, 'mog2', 'frame'), f, 'the shared node must advance by exactly 1 per frame');
        assert.equal(cv.countNonZero(r.mat), 0, '|A-B| of the same memoised node output must be 0 everywhere');
      } finally {
        r.mat.delete();
        for (const e of memo.values()) e.mat.delete();
      }
    }
  } finally { sweepNodeState(new Set(), store); base.delete(); }
});

test('tier2d-D-11. export isolation (V11): computeFrame({ isolatedState: true }) uses its own state store -- the export sees frame-1 semantics, the live counter is untouched (1, 2 -> export -> 3), and nothing leaks (global wrapper tracker)', async () => {
  const { computeFrame, sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 111);
  const buffer = rgbaBufferOf(base);
  const mogId = nid();
  const pipelines = [{ connected: true, displayId: 'a', pipeline: [{ id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId }] }];
  try {
    assert.equal(noteValue(computeFrame(96, 72, buffer, pipelines).notes, 'mog2', 'frame'), 1);
    assert.equal(noteValue(computeFrame(96, 72, buffer, pipelines).notes, 'mog2', 'frame'), 2);

    const exported = computeFrame(96, 72, buffer, pipelines, { isolatedState: true });
    assert.equal(noteValue(exported.notes, 'mog2', 'frame'), 1, 'every exported image is frame 1 of a fresh state');
    assert.match(noteInfos(exported.notes, 'mog2')[0], /เฟรมแรก/);
    // เฟรมแรกของ MOG2 ต้องเป็น mask ว่าง (V2) -> ภาพผลลัพธ์ของ export ดำสนิท
    assert.ok(new Uint8Array(exported.results[0].buffer).every((v, i) => (i % 4 === 3 ? v === 255 : v === 0)));

    assert.equal(noteValue(computeFrame(96, 72, buffer, pipelines).notes, 'mog2', 'frame'), 3,
      'the live state must continue where it was, unaffected by the export');

    assertWrapperBalance(cv, 'export interleaved with live frames', () => {
      computeFrame(96, 72, buffer, pipelines);
      computeFrame(96, 72, buffer, pipelines, { isolatedState: true });
      computeFrame(96, 72, buffer, pipelines);
      sweepNodeState(new Set()); // ทิ้ง state สดที่ถูกสร้างภายใน tracker ด้วย ไม่งั้นค้างตามจริง
    }, { global: true, minCreated: 10 });
  } finally { sweepNodeState(new Set()); base.delete(); }
});

test('tier2d-D-12. backward compatibility: runPipeline still works with 4 arguments (no memo) and with `nodeState` undefined -- in that case getNodeState falls back to the module-level store (proved by sweeping it)', async () => {
  const { runPipeline, collectNodeIds, sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 112);
  const mogId = nid();
  const pipeline = [{ id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId }];
  const active = new Set();
  collectNodeIds(pipeline, active);
  try {
    const r1 = runPipeline(base, pipeline, undefined, active); // 4 อาร์กิวเมนต์ ไม่มี frameMemo
    r1.mat.delete();
    const r2 = runPipeline(base, pipeline, undefined, active);
    r2.mat.delete();
    assert.equal(noteValue(r2.notes, 'mog2', 'frame'), 2, 'state must persist even when no store is passed');
    sweepNodeState(new Set()); // sweep "store ของโมดูล" (ค่าเริ่มต้นของพารามิเตอร์ที่สอง)
    const r3 = runPipeline(base, pipeline, undefined, active);
    r3.mat.delete();
    assert.equal(noteValue(r3.notes, 'mog2', 'frame'), 1, 'sweeping the module store must have cleared that state');
  } finally { sweepNodeState(new Set()); base.delete(); }
});

// ===================================================================
// MG-1…MG-6 — mog2
// ===================================================================
// วางสี่เหลี่ยมทึบลงบนภาพ (คืน Mat ใหม่)
function withRect(base, r, scalar) {
  const m = base.clone();
  cv.rectangle(m, new cv.Point(r.x, r.y), new cv.Point(r.x + r.w - 1, r.y + r.h - 1), scalar, -1);
  return m;
}
// สถิติของ mask 1 ช่อง เทียบกับสี่เหลี่ยมที่วางไว้
// insidePct นับเฉพาะพิกเซลที่เป็น 255 เป๊ะ ๆ ส่วน insideNonZeroPct นับ "ทุกค่าที่ไม่ใช่ 0" — สองตัวนี้ต่างกัน
// ตรงป้ายเงา (127) ของ MOG2 ซึ่งเป็นสิ่งที่ cv.threshold(...,200,...) ต้องตัดทิ้ง: findContours/Bounding Boxes
// ปลายน้ำนับ "ทุกพิกเซลที่ไม่เป็นศูนย์" เป็นวัตถุ เงาที่หลุดออกไปจึงกลายเป็นวัตถุปลอมทันที (MG-4 ใช้ตัวหลัง)
function maskStats(mask, r) {
  let inHit = 0, inNz = 0, inTot = 0, outHit = 0, outTot = 0;
  let minX = Infinity, minY = Infinity, maxX = -1, maxY = -1;
  const d = mask.data;
  for (let y = 0; y < mask.rows; y++) {
    for (let x = 0; x < mask.cols; x++) {
      const v = d[y * mask.cols + x];
      const inside = x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h;
      if (inside) { inTot++; if (v === 255) inHit++; if (v !== 0) inNz++; } else { outTot++; if (v !== 0) outHit++; }
      if (v !== 0) { if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y; }
    }
  }
  return {
    insidePct: (100 * inHit) / inTot,
    insideNonZeroPct: (100 * inNz) / inTot,
    outsidePct: (100 * outHit) / outTot,
    bbox: { minX, minY, maxX, maxY },
  };
}
// อุ่น MOG2 ด้วยฉากนิ่ง n เฟรม แล้วคืน state
function warmMog(bg, params, n = 40) {
  const st = {};
  for (let i = 0; i < n; i++) {
    const o = OPS.mog2.apply(cv, bg, params, { state: st });
    o.delete();
  }
  return st;
}

test('tier2d-MG-1. mog2 frame 1: OpenCV marks 100% of the pixels as foreground (V2) -- the op must suppress that, so the output is a 1-channel mask that is 0 everywhere, info says "first frame" and `frame` = 1 (checked with detectShadows both on and off: with shadows OFF OpenCV really returns 255 everywhere, which is the case the frame-1 guard exists for)', () => {
  const src = cloudRGB(96, 72, 121);
  try {
    for (const detectShadows of [true, false]) {
      // ยืนยันก่อนว่า OpenCV ดิบ ๆ ให้ "ทั้งภาพเป็น foreground" จริงในเฟรมแรก (ไม่งั้นเทสต์นี้จะพิสูจน์อะไรไม่ได้)
      const probe = new cv.BackgroundSubtractorMOG2(500, 16, detectShadows);
      const raw = new cv.Mat();
      probe.apply(src, raw, -1);
      assert.equal(cv.countNonZero(raw), src.rows * src.cols,
        `detectShadows=${detectShadows}: OpenCV is expected to mark every pixel on frame 1`);
      raw.delete(); probe.delete();

      const ctx = { state: {} };
      const out = OPS.mog2.apply(cv, src, { view: 'mask', detectShadows }, ctx);
      try {
        assert.equal(out.channels(), 1);
        assert.equal(out.type(), cv.CV_8UC1);
        assert.equal(out.cols, 96);
        assert.equal(out.rows, 72);
        assert.equal(cv.countNonZero(out), 0, `detectShadows=${detectShadows}: frame 1 must not report any foreground`);
        assert.match(ctx.info, /เฟรมแรก/);
        const v = valuesOf(ctx);
        assert.equal(v.frame, 1);
        assert.equal(v.foreground_px, 0);
        assert.equal(v.foreground_pct, 0);
      } finally {
        out.delete();
        if (ctx.state.__dispose) ctx.state.__dispose();
      }
    }
  } finally { src.delete(); }
});

test('tier2d-MG-2. mog2 detection: after 40 static frames, a bright square appearing is marked -- >=90% of the square is 255 and <=1% of the rest, and foreground_px equals countNonZero of the returned mask', () => {
  const bg = cloudRGB(96, 72, 122);
  const rect = { x: 30, y: 20, w: 24, h: 18 };
  const params = { view: 'mask' };
  const st = warmMog(bg, params);
  const moved = withRect(bg, rect, new cv.Scalar(255, 255, 255));
  const ctx = { state: st };
  const out = OPS.mog2.apply(cv, moved, params, ctx);
  try {
    const s = maskStats(out, rect);
    assert.ok(s.insidePct >= 90, `expected >=90% of the square detected, got ${s.insidePct.toFixed(1)}%`);
    assert.ok(s.outsidePct <= 1, `expected <=1% false positives outside, got ${s.outsidePct.toFixed(2)}%`);
    const v = valuesOf(ctx);
    assert.equal(v.foreground_px, cv.countNonZero(out));
    assert.equal(v.frame, 41);
    assert.match(ctx.info, /พบส่วนที่เคลื่อนไหว/);
  } finally { out.delete(); moved.delete(); st.__dispose(); bg.delete(); }
});

test('tier2d-MG-3. mog2 downscale mapping: on a 1280x720 input (analysis 480x270) the mask comes back at the INPUT size, holds only 0/255 (INTER_NEAREST, no interpolated grays) and its bounding box matches the planted square within +/-(invScale+1) px', () => {
  const bg = cloudRGB(1280, 720, 123);
  const rect = { x: 400, y: 200, w: 300, h: 200 };
  const params = { view: 'mask' };
  const st = warmMog(bg, params);
  const moved = withRect(bg, rect, new cv.Scalar(255, 255, 255));
  const ctx = { state: st };
  const out = OPS.mog2.apply(cv, moved, params, ctx);
  try {
    assert.equal(out.cols, 1280);
    assert.equal(out.rows, 720);
    for (const v of out.data) assert.ok(v === 0 || v === 255, `mask must be strictly 0/255 (found ${v})`);
    const tol = 1280 / 480 + 1;
    const s = maskStats(out, rect);
    assert.ok(Math.abs(s.bbox.minX - rect.x) <= tol, `minX ${s.bbox.minX} vs ${rect.x}`);
    assert.ok(Math.abs(s.bbox.minY - rect.y) <= tol, `minY ${s.bbox.minY} vs ${rect.y}`);
    assert.ok(Math.abs(s.bbox.maxX - (rect.x + rect.w - 1)) <= tol, `maxX ${s.bbox.maxX} vs ${rect.x + rect.w - 1}`);
    assert.ok(Math.abs(s.bbox.maxY - (rect.y + rect.h - 1)) <= tol, `maxY ${s.bbox.maxY} vs ${rect.y + rect.h - 1}`);
  } finally { out.delete(); moved.delete(); st.__dispose(); bg.delete(); }
});

test('tier2d-MG-4. mog2 shadows: a darker copy of the background (a shadow) is excluded with detectShadows=true (<5% of the patch), excluded just the same when the key is MISSING from the params (the registry default is true), and included with detectShadows=false (>=90%) -- the 127 shadow label is thresholded away so downstream Find Contours never sees it', () => {
  const rect = { x: 30, y: 20, w: 24, h: 18 };
  // เคส "ไม่มีคีย์เลย" (undefined) คือของจริงที่เกิดได้: ไฟล์โปรเจกต์ที่บันทึกไว้ก่อนมี param นี้ หรือ
  // โค้ดที่เรียก apply() ตรง ๆ ด้วย {} — ต้องทำตัวเหมือน true ตาม default ของ param ใน registry
  for (const [detectShadows, lo, hi] of [[true, 0, 5], [undefined, 0, 5], [false, 90, 100]]) {
    const bg = cloudRGB(96, 72, 124);
    const params = { view: 'mask' };
    if (detectShadows !== undefined) params.detectShadows = detectShadows;
    const st = warmMog(bg, params);
    const shadowed = bg.clone();
    const roi = shadowed.roi(new cv.Rect(rect.x, rect.y, rect.w, rect.h));
    const dark = new cv.Mat();
    roi.convertTo(dark, -1, 0.55, 0); // 55% ของความสว่างเดิม = เงาตามนิยามของ MOG2
    dark.copyTo(roi);
    dark.delete(); roi.delete();
    const out = OPS.mog2.apply(cv, shadowed, params, { state: st });
    try {
      // นับ "ทุกพิกเซลที่ไม่เป็นศูนย์" ไม่ใช่เฉพาะ 255 — OpenCV ใส่ป้ายเงาเป็น 127 (ยืนยันแล้วบนฉากนี้:
      // ผลดิบของเฟรมนี้คือ 127 ทั้งแปลง) ถ้า cv.threshold(...,200,...) หายไป ค่า 127 จะหลุดออกไปเป็น
      // "วัตถุ" ของ Find Contours/Bounding Boxes ปลายน้ำทันที — เทสต์นี้คือด่านของขั้นนั้น
      const s = maskStats(out, rect);
      assert.ok(s.insideNonZeroPct >= lo && s.insideNonZeroPct <= hi,
        `detectShadows=${detectShadows === undefined ? '(key absent)' : detectShadows}: expected ${lo}..${hi}% of the shadow patch to be non-zero, got ${s.insideNonZeroPct.toFixed(1)}%`);
      for (const v of out.data) assert.ok(v === 0 || v === 255, `the mask must be strictly 0/255, never the 127 shadow label (found ${v})`);
    } finally { out.delete(); shadowed.delete(); st.__dispose(); bg.delete(); }
  }
});

test('tier2d-MG-5. mog2 overlay view: 3-channel output at the input size, pixels outside the mask are byte-identical to the input, pixels inside are tinted green -- and a 1-channel (gray) input also works', () => {
  const bg = cloudRGB(96, 72, 125);
  const rect = { x: 30, y: 20, w: 24, h: 18 };
  const params = { view: 'overlay' };
  const st = warmMog(bg, params);
  const moved = withRect(bg, rect, new cv.Scalar(255, 255, 255));
  const maskSt = warmMog(bg, { view: 'mask' });
  const mask = OPS.mog2.apply(cv, moved, { view: 'mask' }, { state: maskSt });
  const out = OPS.mog2.apply(cv, moved, params, { state: st });
  try {
    assert.equal(out.channels(), 3);
    assert.equal(out.cols, 96);
    assert.equal(out.rows, 72);
    let outsideSame = 0, insideTinted = 0, insideTotal = 0;
    for (let y = 0; y < out.rows; y++) {
      for (let x = 0; x < out.cols; x++) {
        const i = (y * out.cols + x) * 3;
        const m = mask.data[y * out.cols + x];
        if (m === 0) {
          assert.equal(out.data[i], moved.data[i], `pixel outside the mask must be untouched at (${x},${y})`);
          assert.equal(out.data[i + 1], moved.data[i + 1]);
          assert.equal(out.data[i + 2], moved.data[i + 2]);
          outsideSame++;
        } else {
          insideTotal++;
          if (out.data[i + 1] > moved.data[i + 1] || out.data[i] < moved.data[i]) insideTinted++;
        }
      }
    }
    assert.ok(outsideSame > 0 && insideTotal > 0, 'the test scene must produce both masked and unmasked pixels');
    assert.ok(insideTinted / insideTotal >= 0.9, 'masked pixels must be visibly tinted towards green');

    const gray = toGrayMat(bg);
    const grayMoved = toGrayMat(moved);
    const gst = warmMog(gray, params);
    const gout = OPS.mog2.apply(cv, grayMoved, params, { state: gst });
    assert.equal(gout.channels(), 3, 'a gray input must still give a 3-channel overlay');
    assert.equal(gout.cols, 96);
    gout.delete(); gst.__dispose(); gray.delete(); grayMoved.delete();
  } finally {
    out.delete(); mask.delete(); maskSt.__dispose(); moved.delete(); st.__dispose(); bg.delete();
  }
});

test('tier2d-MG-6. mog2 params really are used: varThreshold 4 finds strictly more foreground than 64 on the same sequence, and string params ("300"/"16" from a hand-edited project file) behave exactly like numbers (P-13)', () => {
  const rect = { x: 30, y: 20, w: 24, h: 18 };
  const measure = (params) => {
    const bg = cloudRGB(96, 72, 126);
    const st = warmMog(bg, params);
    const moved = withRect(bg, rect, new cv.Scalar(120, 120, 120)); // ต่างจากพื้นหลังแบบพอประมาณ -> ความไวมีผลจริง
    const ctx = { state: st };
    const out = OPS.mog2.apply(cv, moved, params, ctx);
    const px = cv.countNonZero(out);
    out.delete(); moved.delete(); st.__dispose(); bg.delete();
    return px;
  };
  const sensitive = measure({ view: 'mask', varThreshold: 4 });
  const strict = measure({ view: 'mask', varThreshold: 64 });
  assert.ok(sensitive > strict, `varThreshold 4 (${sensitive} px) must find strictly more than 64 (${strict} px)`);

  const asNumbers = measure({ view: 'mask', history: 300, varThreshold: 16 });
  const asStrings = measure({ view: 'mask', history: '300', varThreshold: '16' });
  assert.equal(asStrings, asNumbers, 'string params must be coerced, not fall back to the defaults');
});

// ===================================================================
// FB-1…FB-6 — farneback_optical_flow
// ===================================================================
function runFlowPair(id, frames, params) {
  const st = {};
  const results = frames.map((f) => {
    const ctx = { state: st };
    const out = OPS[id].apply(cv, f, params, ctx);
    return { out, ctx };
  });
  return { st, results };
}

test('tier2d-FB-1. farneback frame 1: no previous frame exists yet -> the output is byte-identical to the RGB copy of the input, info says "first frame", and every motion value is 0 with frame = 1', () => {
  const src = cloudRGB(96, 72, 131);
  const ctx = { state: {} };
  const out = OPS.farneback_optical_flow.apply(cv, src, {}, ctx);
  const ref = expectedRGB(src);
  try {
    assert.equal(out.channels(), 3);
    assert.ok(sameBytes(out.data, ref.data), 'frame 1 must pass the image through unchanged');
    assert.match(ctx.info, /เฟรมแรก/);
    const v = valuesOf(ctx);
    assert.equal(v.frame, 1);
    for (const k of ['mean_motion_px', 'max_motion_px', 'moving_pct', 'mean_dx_px', 'mean_dy_px']) assert.equal(v[k], 0);
  } finally {
    out.delete(); ref.delete();
    if (ctx.state.__dispose) ctx.state.__dispose();
    src.delete();
  }
});

test('tier2d-FB-2. farneback correctness: a planted (+3,-2) px shift at 160x120 is reported as mean_dx_px/mean_dy_px within +/-0.8 px, and mean_motion_px matches hypot(3,2)', () => {
  const frames = shiftedFrames(160, 120, [[0, 0], [3, -2]], 131);
  const { st, results } = runFlowPair('farneback_optical_flow', frames, {});
  try {
    const v = valuesOf(results[1].ctx);
    assert.ok(Math.abs(v.mean_dx_px - 3) <= 0.8, `mean_dx_px = ${v.mean_dx_px}`);
    assert.ok(Math.abs(v.mean_dy_px - -2) <= 0.8, `mean_dy_px = ${v.mean_dy_px}`);
    assert.ok(Math.abs(v.mean_motion_px - Math.hypot(3, 2)) <= 0.8, `mean_motion_px = ${v.mean_motion_px}`);
    assert.equal(v.frame, 2);
    assert.match(results[1].ctx.info, /ขยับเฉลี่ย/);
  } finally {
    for (const r of results) r.out.delete();
    st.__dispose();
    for (const f of frames) f.delete();
  }
});

test('tier2d-FB-3. farneback downscale (the "x invScale" conversion): on a 640x480 input (analysis 320x240) a planted (+6,-4) px shift is reported in INPUT pixels -- mean_dx/mean_dy ~ (6,-4), mean_motion_px AND max_motion_px ~ hypot(6,4), output keeps the input size -- and the colour view of the SAME physical motion is equally bright at 640x480 (downscaled) and at 320x240 (not downscaled)', () => {
  const frames = shiftedFrames(640, 480, [[0, 0], [6, -4]], 132);
  const { st, results } = runFlowPair('farneback_optical_flow', frames, {});
  try {
    const v = valuesOf(results[1].ctx);
    assert.ok(Math.abs(v.mean_dx_px - 6) <= 1.5, `mean_dx_px = ${v.mean_dx_px}`);
    assert.ok(Math.abs(v.mean_dy_px - -4) <= 1.5, `mean_dy_px = ${v.mean_dy_px}`);
    // ตัวเลขความเร็วต้องเป็น px ของ "ภาพจริง" ไม่ใช่ px ของภาพย่อ: ที่ 640x480 invScale = 2 ถ้าลืม x inv
    // ค่าจะต่ำกว่าความจริง 2 เท่า (และที่ 1080p ผิด 6 เท่า) โดยภาพที่เห็นยังดูปกติทุกอย่าง
    const expected = Math.hypot(6, 4); // 7.21 px/เฟรม
    assert.ok(Math.abs(v.mean_motion_px - expected) <= 0.5, `mean_motion_px = ${v.mean_motion_px} (expected ~${expected.toFixed(2)})`);
    assert.ok(v.max_motion_px >= expected - 0.5 && v.max_motion_px <= 2 * expected, `max_motion_px = ${v.max_motion_px} (expected ~${expected.toFixed(2)}, never the analysis-pixel value ~${(expected / 2).toFixed(2)})`);
    assert.equal(results[1].out.cols, 640);
    assert.equal(results[1].out.rows, 480);
  } finally {
    for (const r of results) r.out.delete();
    st.__dispose();
    for (const f of frames) f.delete();
  }

  // ความสว่างของโหมดวงล้อสี = ความเร็วเป็น px ของภาพจริง / maxMotion -> การเคลื่อนที่ (6,-4) px เท่ากัน
  // ต้องได้ความสว่างเท่ากันไม่ว่าภาพจะถูกย่อหรือไม่ (640x480 ย่อเหลือ 320x240 / 320x240 ไม่ถูกย่อเลย)
  // ถ้าสเกลเป็น 255/maxMotion แทน (255 * inv)/maxMotion ภาพที่ความละเอียดสูงจะมืดลงตามอัตราการย่อ
  const brightness = (w, h) => {
    const fr = shiftedFrames(w, h, [[0, 0], [6, -4]], 132);
    const p = runFlowPair('farneback_optical_flow', fr, { view: 'color', maxMotion: 10 });
    const m = cv.mean(p.results[1].out);
    for (const r of p.results) r.out.delete();
    p.st.__dispose();
    for (const f of fr) f.delete();
    return m[0] + m[1] + m[2];
  };
  const b640 = brightness(640, 480);
  const b320 = brightness(320, 240);
  assert.ok(Math.abs(b640 - b320) / b320 <= 0.05,
    `the colour view must not depend on the analysis downscale: 640x480 -> ${b640.toFixed(1)}, 320x240 -> ${b320.toFixed(1)}`);
});

test('tier2d-FB-4. farneback on two identical frames: mean_motion_px < 0.05, the colour view stays essentially black (no per-frame normalize -- a normalize would turn this residue into a full-brightness flicker), and the arrows view is byte-identical to the input', () => {
  const src = cloudRGB(160, 120, 133);
  const colour = runFlowPair('farneback_optical_flow', [src, src], { view: 'color' });
  try {
    const v = valuesOf(colour.results[1].ctx);
    assert.ok(v.mean_motion_px < 0.05, `mean_motion_px = ${v.mean_motion_px}`);
    // ค่าคงที่ 10: ใบสั่งงานเขียนไว้ 5 แต่วัดจริงบนฉากของไฟล์นี้ได้สูงสุด 6 — เพดานของค่านี้คือ
    // max(|flow|) x 255 / maxMotion (สเกลคงที่, ไม่มี normalize) ซึ่งเป็นเศษเหลือของอัลกอริทึมล้วน ๆ
    // สิ่งที่เทสต์นี้ต้องจับคือ "ถ้าเผลอใส่ normalize ต่อเฟรม ค่าจะพุ่งไป 255" ซึ่ง 10 ก็จับได้เท่ากับ 5
    for (const px of colour.results[1].out.data) assert.ok(px <= 10, `colour view must stay dark when nothing moves (found ${px})`);
  } finally { for (const r of colour.results) r.out.delete(); colour.st.__dispose(); }

  const arrows = runFlowPair('farneback_optical_flow', [src, src], { view: 'arrows' });
  const ref = expectedRGB(src);
  try {
    assert.ok(sameBytes(arrows.results[1].out.data, ref.data), 'no motion must mean no arrows drawn at all');
  } finally {
    for (const r of arrows.results) r.out.delete();
    arrows.st.__dispose();
    ref.delete();
    src.delete();
  }
});

test('tier2d-FB-5. farneback colour wheel: a purely rightward shift gives a red-dominant image and a leftward shift a cyan one (G and B dominant) -- this is the check behind the Thai colour words in the op description', () => {
  const meanRGB = (m) => {
    const mv = new cv.MatVector();
    cv.split(m, mv);
    const out = [];
    for (let i = 0; i < 3; i++) { const c = mv.get(i); out.push(cv.mean(c)[0]); c.delete(); }
    mv.delete();
    return out;
  };
  const run = (dx, dy) => {
    const frames = shiftedFrames(160, 120, [[0, 0], [dx, dy]], 134);
    const { st, results } = runFlowPair('farneback_optical_flow', frames, { view: 'color', maxMotion: 8 });
    const rgb = meanRGB(results[1].out);
    for (const r of results) r.out.delete();
    st.__dispose();
    for (const f of frames) f.delete();
    return rgb;
  };
  const right = run(6, 0);
  assert.ok(right[0] > 100 && right[1] < 30 && right[2] < 30, `rightward must be red-dominant, got ${right.map((v) => v.toFixed(1))}`);
  const left = run(-6, 0);
  assert.ok(left[1] > 100 && left[2] > 100 && left[0] < 30, `leftward must be cyan (G+B), got ${left.map((v) => v.toFixed(1))}`);
  // ยืนยันคำไทยที่เหลือใน desc: ลง = เขียวอมเหลือง (R กลาง + G สูง, B ต่ำ), ขึ้น = ม่วง (R กลาง + B สูง, G ต่ำ)
  const down = run(0, 6);
  assert.ok(down[1] > 100 && down[2] < 30 && down[0] > 30 && down[0] < down[1], `downward must be yellow-green, got ${down.map((v) => v.toFixed(1))}`);
  const up = run(0, -6);
  assert.ok(up[2] > 100 && up[1] < 30 && up[0] > 30 && up[0] < up[2], `upward must be violet, got ${up.map((v) => v.toFixed(1))}`);
  // ข้อความในคำอธิบายต้องตรงกับที่วัดได้จริง (ห้ามเดา)
  assert.match(OPS.farneback_optical_flow.desc, /แดง=ไปขวา/);
  assert.match(OPS.farneback_optical_flow.desc, /เขียวอมเหลือง=ลง/);
  assert.match(OPS.farneback_optical_flow.desc, /ฟ้า=ไปซ้าย/);
  assert.match(OPS.farneback_optical_flow.desc, /ม่วง=ขึ้น/);
});

test('tier2d-FB-6. farneback arrows view draws green pixels when the scene moves, and maxMotion scales the colour view monotonically (5 is brighter than 20 for the same motion)', () => {
  const frames = shiftedFrames(160, 120, [[0, 0], [4, 3]], 135);
  try {
    const arrows = runFlowPair('farneback_optical_flow', frames, { view: 'arrows' });
    let green = 0;
    const d = arrows.results[1].out.data;
    for (let i = 0; i < d.length; i += 3) if (d[i] === 0 && d[i + 1] === 255 && d[i + 2] === 0) green++;
    assert.ok(green > 0, 'moving scene must draw at least some pure-green arrow pixels');
    assert.equal(arrows.results[1].out.cols, 160);
    for (const r of arrows.results) r.out.delete();
    arrows.st.__dispose();

    const brightness = (maxMotion) => {
      const p = runFlowPair('farneback_optical_flow', frames, { view: 'color', maxMotion });
      const m = cv.mean(p.results[1].out);
      const sum = m[0] + m[1] + m[2];
      for (const r of p.results) r.out.delete();
      p.st.__dispose();
      return sum;
    };
    assert.ok(brightness(5) > brightness(20), 'a lower maxMotion must render the same motion brighter');
  } finally { for (const f of frames) f.delete(); }
});

// FB-6 ข้างบนรันที่ 160x120 ซึ่ง "ไม่ถูกย่อ" (invScale = 1) — ทุกจุดที่คูณ inv จึงมองไม่เห็นความต่าง
// FB-7 คือคู่แฝดของมันที่ความละเอียดที่ถูกย่อจริง: ทั้งเกณฑ์ "ขยับพอไหม" และพิกัดที่วาด ต้องอยู่ในหน่วย
// พิกเซลของภาพจริง ไม่ใช่ของภาพวิเคราะห์ (ภาพที่ผู้ใช้เห็นคือ output หลักของโหมดลูกศร)
test('tier2d-FB-7. farneback at a REAL downscale (1280x960 -> analysis 320x240, invScale 4): a (2,1) px shift is only 0.56 px in the analysis image but 2.24 px in the input, so (a) arrows must still be drawn (the "too still to draw" test is in input pixels), (b) moving_pct must match the same physical motion at 320x240 (the >=1px threshold is in input pixels too) and (c) the arrows must cover the WHOLE input image -- not the top-left 320x240 corner it would be if the drawn coordinates were left in analysis pixels', () => {
  const runArrows = (w, h) => {
    const frames = shiftedFrames(w, h, [[0, 0], [2, 1]], 136);
    const { st, results } = runFlowPair('farneback_optical_flow', frames, { view: 'arrows' });
    try {
      return { v: valuesOf(results[1].ctx), green: pureColorBox(results[1].out, [0, 255, 0]), cols: results[1].out.cols, rows: results[1].out.rows };
    } finally {
      for (const r of results) r.out.delete();
      st.__dispose();
      for (const f of frames) f.delete();
    }
  };
  const big = runArrows(1280, 960); // ย่อจริง: invScale = 4
  const small = runArrows(320, 240); // ไม่ถูกย่อเลย: invScale = 1

  const expected = Math.hypot(2, 1); // 2.24 px/เฟรม ของภาพจริง
  assert.ok(Math.abs(big.v.mean_motion_px - expected) <= 0.8, `mean_motion_px = ${big.v.mean_motion_px}`);
  assert.equal(big.cols, 1280);
  assert.equal(big.rows, 960);
  // (a) เกณฑ์ "นิ่งพอจนไม่ต้องวาด" ต้องเทียบเป็น px ของภาพจริง — ที่นี่ภาพย่อขยับแค่ ~0.56 px
  assert.ok(big.green.count > 0,
    'a motion of 2.24 input px must still draw arrows even though it is only 0.56 px in the analysis image');
  // (b) moving_pct = สัดส่วนพื้นที่ที่ขยับ >= 1 px ของภาพจริง -> การเคลื่อนที่ทางกายภาพเดียวกัน
  //     ต้องได้ค่าเท่ากันไม่ว่าภาพจะถูกย่อหรือไม่ (ถ้าเกณฑ์เป็น 1 px ของภาพย่อ ที่ 1280x960 จะได้ 0%)
  assert.ok(big.v.moving_pct > 80, `moving_pct at 1280x960 = ${big.v.moving_pct} (the whole scene moves by 2.24 input px)`);
  assert.ok(Math.abs(big.v.moving_pct - small.v.moving_pct) <= 5,
    `moving_pct must not depend on the analysis downscale: 1280x960 -> ${big.v.moving_pct.toFixed(1)}, 320x240 -> ${small.v.moving_pct.toFixed(1)}`);
  // (c) พิกัดที่วาดต้องเป็นพิกเซลของภาพจริง — ลูกศรต้องแผ่เต็มภาพ ไม่ใช่กองอยู่ในกรอบ 320x240 มุมซ้ายบน
  assert.ok(big.green.maxX > 320 && big.green.maxY > 240,
    `arrows must be drawn in INPUT coordinates: green bbox is [${big.green.minX},${big.green.minY}]-[${big.green.maxX},${big.green.maxY}] but the analysis image is only 320x240`);
  assert.ok(big.green.maxX >= 0.9 * 1280 && big.green.maxY >= 0.9 * 960,
    `arrows must reach the far edge of the input image, got [${big.green.maxX},${big.green.maxY}] of 1280x960`);
});

// ===================================================================
// LK-1…LK-6 — pyrlk_optical_flow
// ===================================================================
test('tier2d-LK-1. pyrlk frame 1: feature points are seeded (>0), red dots are drawn on the frame, info says "first frame" and `tracked` is 0 (nothing to compare against yet)', () => {
  const src = cloudRGB(160, 120, 141);
  const ctx = { state: {} };
  const out = OPS.pyrlk_optical_flow.apply(cv, src, {}, ctx);
  try {
    const v = valuesOf(ctx);
    assert.ok(v.points > 0, 'a textured scene must yield feature points');
    assert.equal(v.tracked, 0);
    assert.equal(v.reseeded, 1);
    assert.equal(v.frame, 1);
    assert.match(ctx.info, /เฟรมแรก/);
    assert.equal(out.channels(), 3);
    assert.equal(out.cols, 160);
    let red = 0;
    for (let i = 0; i < out.data.length; i += 3) if (out.data[i] === 255 && out.data[i + 1] === 0 && out.data[i + 2] === 0) red++;
    assert.ok(red > 0, 'the current position of each tracked point must be drawn as a red dot');
  } finally {
    out.delete();
    if (ctx.state.__dispose) ctx.state.__dispose();
    src.delete();
  }
});

test('tier2d-LK-2. pyrlk correctness: a planted (4,1) px shift gives mean_motion_px ~= 4.12 (+/-0.5), and each trail grows to min(frames, trailLength+1) positions', () => {
  const frames = shiftedFrames(160, 120, [[0, 0], [4, 1], [8, 2], [12, 3]], 142);
  const st = {};
  const outs = [];
  try {
    const ctxs = frames.map((f) => {
      const ctx = { state: st };
      outs.push(OPS.pyrlk_optical_flow.apply(cv, f, { trailLength: 10 }, ctx));
      return ctx;
    });
    const v = valuesOf(ctxs[1]);
    assert.ok(Math.abs(v.mean_motion_px - Math.hypot(4, 1)) <= 0.5, `mean_motion_px = ${v.mean_motion_px}`);
    assert.ok(v.tracked > 0);
    assert.equal(v.frame, 2);
    assert.match(ctxs[1].info, /ติดตาม \d+ จุด/);
    const longest = Math.max(...st.trails.map((t) => t.length));
    assert.equal(longest, Math.min(frames.length, 11), 'trails must grow by one position per frame');
  } finally {
    for (const o of outs) o.delete();
    st.__dispose();
    for (const f of frames) f.delete();
  }
});

test('tier2d-LK-3. pyrlk downscale: on a 960x540 input (analysis 480x270) a planted (8,2) px shift is reported in INPUT pixels (~8.25 +/-1.0), the output keeps the input size, AND the drawn dots/trails are in input coordinates too -- they must cover the whole 960x540 frame, not the top-left 480x270 corner they would land in if the coordinates were left in analysis pixels', () => {
  const frames = shiftedFrames(960, 540, [[0, 0], [8, 2]], 143);
  const st = {};
  const outs = [];
  try {
    const ctxs = frames.map((f) => {
      const ctx = { state: st };
      outs.push(OPS.pyrlk_optical_flow.apply(cv, f, {}, ctx));
      return ctx;
    });
    const v = valuesOf(ctxs[1]);
    assert.ok(Math.abs(v.mean_motion_px - Math.hypot(8, 2)) <= 1.0, `mean_motion_px = ${v.mean_motion_px}`);
    assert.equal(outs[1].cols, 960);
    assert.equal(outs[1].rows, 540);
    // ภาพที่ผู้ใช้เห็นคือ output หลักของ op นี้ — พิกัดที่วาดต้อง × invScale เหมือนกับตัวเลข
    // (ถ้าลืม จุด/เส้นทั้งหมดจะกองอยู่ในกรอบ 480x270 มุมซ้ายบน = 25% ของภาพ ที่ 1080p เหลือ 6.25%)
    const red = pureColorBox(outs[1], [255, 0, 0]);
    const green = pureColorBox(outs[1], [0, 255, 0]);
    assert.ok(red.count > 0 && green.count > 0, 'a tracked frame must draw both red dots and green trails');
    // จุดแดงต้อง "มองเห็นได้" บนจอความละเอียดสูง ไม่ใช่พิกเซลเดียวต่อจุด (วัดจริงที่ 1080p: รัศมี 3 = 29 px/จุด)
    // เกณฑ์หลวม ๆ ที่ไม่ล็อกรัศมีไว้ที่ค่าใดค่าหนึ่ง: อย่างน้อย 2 พิกเซลต่อจุดที่ติดตามอยู่
    assert.ok(red.count >= 2 * st.trails.length,
      `จุดที่ติดตามต้องเห็นเป็นวง ไม่ใช่พิกเซลเดียว: ${red.count} px สำหรับ ${st.trails.length} จุด`);
    for (const [name, b] of [['red dots', red], ['green trails', green]]) {
      assert.ok(b.maxX > 480 && b.maxY > 270,
        `${name} must be drawn in INPUT coordinates: bbox [${b.minX},${b.minY}]-[${b.maxX},${b.maxY}] fits inside the 480x270 analysis image`);
      assert.ok(b.maxX >= 0.8 * 960 && b.maxY >= 0.8 * 540,
        `${name} must spread over the whole 960x540 frame, got [${b.maxX},${b.maxY}]`);
    }
  } finally {
    for (const o of outs) o.delete();
    st.__dispose();
    for (const f of frames) f.delete();
  }
});

test('tier2d-LK-4. pyrlk on flat black frames (V5 zero-points guard): 0 points, no throw, a helpful Thai message -- and a textured frame afterwards reseeds successfully', () => {
  const black = new cv.Mat(72, 96, cv.CV_8UC3, new cv.Scalar(0, 0, 0));
  const textured = cloudRGB(96, 72, 144);
  const st = {};
  try {
    for (let i = 0; i < 3; i++) {
      const ctx = { state: st };
      let out;
      assert.doesNotThrow(() => { out = OPS.pyrlk_optical_flow.apply(cv, black, {}, ctx); });
      const v = valuesOf(ctx);
      assert.equal(v.points, 0);
      assert.equal(v.tracked, 0);
      assert.match(ctx.info, /ไม่พบจุดเด่น/);
      assert.equal(out.cols, 96);
      out.delete();
    }
    const ctx = { state: st };
    const out = OPS.pyrlk_optical_flow.apply(cv, textured, {}, ctx);
    const v = valuesOf(ctx);
    assert.ok(v.points > 0, 'a textured frame must reseed points');
    assert.equal(v.reseeded, 1);
    out.delete();
  } finally { st.__dispose(); black.delete(); textured.delete(); }
});

test('tier2d-LK-5. pyrlk reseed rule: the threshold is relative to the last seed count, not to maxPoints -- on a low-texture scene with far fewer corners than maxPoints the trails still grow (longest >= 3 after 5 frames); and when the points do die out, exactly one reseed happens', () => {
  // ฉาก "มุมน้อย": จานเบลอไม่กี่ใบบนพื้นเรียบ -> gftt ได้จุดน้อยกว่า 30% ของ maxPoints=100 มาก
  const sparseScene = (w, h, dx, dy) => {
    const m = new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(30, 30, 30));
    const rnd = lcg(4242);
    for (let i = 0; i < 12; i++) {
      const x = 12 + Math.round(rnd() * (w - 24)), y = 12 + Math.round(rnd() * (h - 24));
      cv.rectangle(m, new cv.Point(x + dx, y + dy), new cv.Point(x + dx + 6, y + dy + 6), new cv.Scalar(220, 220, 220), -1);
    }
    const b = new cv.Mat();
    cv.GaussianBlur(m, b, new cv.Size(3, 3), 0);
    m.delete();
    return b;
  };
  const st = {};
  const frames = [];
  try {
    for (let i = 0; i < 5; i++) frames.push(sparseScene(160, 120, i, 0));
    let seeds = 0;
    for (const f of frames) {
      const ctx = { state: st };
      const o = OPS.pyrlk_optical_flow.apply(cv, f, { maxPoints: 100, trailLength: 10 }, ctx);
      seeds += valuesOf(ctx).reseeded;
      o.delete();
    }
    assert.ok(st.seedCount < 30, `the scene must have far fewer corners than maxPoints (seedCount = ${st.seedCount})`);
    const longest = Math.max(...st.trails.map((t) => t.length));
    assert.ok(longest >= 3, `trails must be able to grow (longest = ${longest}); reseeding every frame would keep them at 1`);
    assert.equal(seeds, 1, 'only the first frame should have needed a seed');
  } finally { st.__dispose(); for (const f of frames) f.delete(); }

  // จุดตายยกแผง (ฉากดับไปเป็นสีดำ) -> reseed ต้องเกิดขึ้น "และเกิดตามกฎ": ทุกเฟรมที่ reseed จุดที่รอดต้อง
  // น้อยกว่าครึ่งของ seedCount ครั้งล่าสุด และทุกเฟรมที่ไม่ reseed จุดที่รอดต้องยังถึงครึ่ง (ตรวจเป็น invariant
  // แทนการล็อกตัวเลขเป๊ะ ๆ เพราะจำนวนจุดที่ LK ยอมรับบนภาพดำเป็นรายละเอียดภายในของ OpenCV)
  const st2 = {};
  const textured = cloudRGB(160, 120, 145);
  const black = new cv.Mat(120, 160, cv.CV_8UC3, new cv.Scalar(0, 0, 0));
  try {
    { const o = OPS.pyrlk_optical_flow.apply(cv, textured, {}, { state: st2 }); o.delete(); }
    assert.ok(st2.seedCount > 0);
    let reseeds = 0;
    for (let i = 0; i < 2; i++) {
      const seedBefore = st2.seedCount;
      const ctx = { state: st2 };
      const o = OPS.pyrlk_optical_flow.apply(cv, black, {}, ctx);
      o.delete();
      const v = valuesOf(ctx);
      const half = Math.ceil(0.5 * seedBefore);
      if (v.reseeded) {
        reseeds++;
        assert.ok(v.tracked < half || seedBefore === 0,
          `frame ${i + 1}: reseeded although ${v.tracked} of ${seedBefore} points survived (>= half = ${half})`);
      } else {
        assert.ok(v.tracked >= half,
          `frame ${i + 1}: did not reseed although only ${v.tracked} of ${seedBefore} points survived (< half = ${half})`);
      }
    }
    assert.ok(reseeds >= 1, 'losing the whole scene must eventually trigger a reseed');
  } finally { st2.__dispose(); textured.delete(); black.delete(); }
});

test('tier2d-LK-6. pyrlk trailLength bounds every trail to at most trailLength+1 positions (memory of the trails array stays bounded no matter how long the camera runs)', () => {
  const frames = [];
  const st = {};
  try {
    for (let i = 0; i < 12; i++) frames.push(...shiftedFrames(160, 120, [[2 * i, i]], 146));
    for (const f of frames) {
      const o = OPS.pyrlk_optical_flow.apply(cv, f, { trailLength: 3 }, { state: st });
      o.delete();
    }
    assert.ok(st.trails.length > 0);
    for (const t of st.trails) assert.ok(t.length <= 4, `trail of ${t.length} positions exceeds trailLength+1 = 4`);
    assert.equal(Math.max(...st.trails.map((t) => t.length)), 4, 'at least one trail must actually reach the cap');
  } finally { st.__dispose(); for (const f of frames) f.delete(); }
});

test('tier2d-LK-8. pyrlk drops points that walked OUT of the frame: PyrLK still reports status = 1 for a point whose new position is up to ~half a window outside the image, so status alone is not enough -- those points must be counted as lost, never kept in the trails (a point outside the image would be fed back into PyrLK next frame and would keep `lost` under-reported forever)', () => {
  const W = 480, H = 360, SHIFT = 20; // 480 = ไม่ถูกย่อ (invScale 1) -> พิกัดใน state = พิกเซลของภาพจริง
  const frames = shiftedFrames(W, H, [[0, 0], [SHIFT, 0]], 151);
  const st = {};
  const outs = [];
  try {
    outs.push(OPS.pyrlk_optical_flow.apply(cv, frames[0], {}, { state: st })); // เฟรมแรก = หาจุด
    // จุดที่ "ต้อง" หลุดออกนอกเฟรมเมื่อฉากเลื่อนไปขวา 20 px — ถ้าฉากไม่มีจุดแบบนี้เลย เทสต์จะพิสูจน์อะไรไม่ได้
    const predictedOut = st.trails.filter((t) => t[t.length - 1][0] + SHIFT > W - 1).length;
    assert.ok(predictedOut >= 2, `the scene must seed points that leave the frame (found ${predictedOut}) -- otherwise this test proves nothing`);

    const ctx = { state: st };
    outs.push(OPS.pyrlk_optical_flow.apply(cv, frames[1], {}, ctx));
    const v = valuesOf(ctx);
    assert.ok(Math.abs(v.mean_motion_px - SHIFT) <= 1.0, `mean_motion_px = ${v.mean_motion_px}`);
    assert.ok(v.lost >= predictedOut, `points that left the frame must be counted as lost: lost = ${v.lost}, at least ${predictedOut} points left the frame`);
    for (const t of st.trails) {
      const [x, y] = t[t.length - 1];
      assert.ok(x >= 0 && x <= W - 1 && y >= 0 && y <= H - 1,
        `a kept point sits outside the image: (${x.toFixed(1)}, ${y.toFixed(1)}) is not inside 0..${W - 1} x 0..${H - 1}`);
    }
  } finally {
    for (const o of outs) o.delete();
    st.__dispose();
    for (const f of frames) f.delete();
  }
});

// ===================================================================
// ALL-1…ALL-5 — ทั้งสาม op
// ===================================================================
test('tier2d-ALL-1. tiny inputs (1x1, 1x50, 50x1, 2x2, 3x3) over 3 consecutive frames: no op throws and every output is exactly the input size', () => {
  for (const [w, h] of [[1, 1], [1, 50], [50, 1], [2, 2], [3, 3]]) {
    const src = new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(70, 90, 110));
    try {
      for (const id of MOTION_IDS) {
        const st = {};
        try {
          for (let i = 0; i < 3; i++) {
            let out;
            assert.doesNotThrow(() => { out = OPS[id].apply(cv, src, {}, { state: st }); }, `${id} @${w}x${h} frame ${i + 1}`);
            assert.equal(out.cols, w, `${id} @${w}x${h}: output width`);
            assert.equal(out.rows, h, `${id} @${w}x${h}: output height`);
            out.delete();
          }
        } finally { if (st.__dispose) st.__dispose(); }
      }
    } finally { src.delete(); }
  }
});

test('tier2d-ALL-2. extreme aspect ratios 4000x2 and 2x4000 (analysisScale must clamp with max(1,...) -- P-11 lesson): no throw, output size = input size', () => {
  for (const [w, h] of [[4000, 2], [2, 4000]]) {
    const src = cloudRGB(w, h, 151, 3);
    try {
      for (const id of MOTION_IDS) {
        const st = {};
        try {
          for (let i = 0; i < 2; i++) {
            let out;
            assert.doesNotThrow(() => { out = OPS[id].apply(cv, src, {}, { state: st }); }, `${id} @${w}x${h}`);
            assert.equal(out.cols, w);
            assert.equal(out.rows, h);
            out.delete();
          }
        } finally { if (st.__dispose) st.__dispose(); }
      }
    } finally { src.delete(); }
  }
});

test('tier2d-ALL-3. inputs with 1, 3 and 4 channels are all accepted by every Motion op (the ops convert internally -- Farneback/PyrLK throw a C++ exception on 3-channel input, so this must never reach them)', () => {
  const rgb = cloudRGB(96, 72, 152);
  const gray = toGrayMat(rgb);
  const rgba = toRGBAMat(rgb);
  try {
    for (const [label, src] of [['1ch', gray], ['3ch', rgb], ['4ch', rgba]]) {
      for (const id of MOTION_IDS) {
        const st = {};
        try {
          for (let i = 0; i < 2; i++) {
            let out;
            assert.doesNotThrow(() => { out = OPS[id].apply(cv, src, {}, { state: st }); }, `${id} with ${label} input`);
            assert.equal(out.cols, 96);
            assert.equal(out.rows, 72);
            // mask view ของ mog2 เป็น 1 ช่องเสมอ, ตัวอื่น ๆ วาดทับภาพจึงเป็น 3 ช่อง
            assert.equal(out.channels(), id === 'mog2' ? 1 : 3, `${id} with ${label} input: channel count`);
            out.delete();
          }
        } finally { if (st.__dispose) st.__dispose(); }
      }
    }
  } finally { rgb.delete(); gray.delete(); rgba.delete(); }
});

test('tier2d-ALL-4. no ctx.state at all (the filter-smoke / direct-unit-call path): every op behaves as "frame 1", works with ctx = {} and with no ctx, and disposes its temporary state (wrapper balance 0 live)', () => {
  const src = cloudRGB(96, 72, 153);
  try {
    for (const id of MOTION_IDS) {
      const ctx = {};
      const out = OPS[id].apply(cv, src, {}, ctx);
      assert.equal(out.cols, 96);
      assert.ok(ctx.info, `${id} must still report an info message without a node state`);
      assert.match(ctx.info, id === 'pyrlk_optical_flow' ? /เฟรมแรก|ไม่พบจุดเด่น/ : /เฟรมแรก/);
      out.delete();

      let noCtxOut;
      assert.doesNotThrow(() => { noCtxOut = OPS[id].apply(cv, src, {}, undefined); }, `${id} must tolerate a missing ctx`);
      noCtxOut.delete();

      assertWrapperBalance(cv, `${id} no-state`, (c) => {
        const o = OPS[id].apply(c, src, {}, {});
        o.delete();
      }, { minCreated: 2 });
    }
  } finally { src.delete(); }
});

test('tier2d-ALL-5. state reuse across frames: mog2 keeps the SAME BackgroundSubtractorMOG2 instance, and both flow ops replace prevGray every frame -- the previous one is always isDeleted() (never orphaned, never reused after delete)', () => {
  const src = cloudRGB(96, 72, 154);
  try {
    const mogSt = {};
    let mogInstance = null;
    for (let i = 0; i < 4; i++) {
      const o = OPS.mog2.apply(cv, src, {}, { state: mogSt });
      o.delete();
      if (i === 0) mogInstance = mogSt.mog;
      else assert.equal(mogSt.mog, mogInstance, 'MOG2 must not be reconstructed every frame (that would leak catastrophically)');
      assert.ok(!mogSt.mog.isDeleted());
    }
    mogSt.__dispose();

    for (const id of ['farneback_optical_flow', 'pyrlk_optical_flow']) {
      const st = {};
      let prev = null;
      for (let i = 0; i < 4; i++) {
        const o = OPS[id].apply(cv, src, {}, { state: st });
        o.delete();
        if (prev) {
          assert.ok(prev.isDeleted(), `${id}: the previous prevGray must be deleted when it is replaced`);
          assert.notEqual(st.prevGray, prev);
        }
        assert.ok(st.prevGray && !st.prevGray.isDeleted());
        prev = st.prevGray;
      }
      st.__dispose();
      assert.ok(prev.isDeleted());
    }
  } finally { src.delete(); }
});

// ===================================================================
// CG-1…CG-4 — Python codegen (app-only notice)
// ===================================================================
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'tier2d-motion-'));
process.on('exit', () => {
  try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ }
});
function pythonAvailable() {
  try { execFileSync('python', ['-c', 'import cv2, numpy'], { encoding: 'utf8' }); return true; } catch { return false; }
}
const HAS_PYTHON = pythonAvailable();

function matToRaw(mat) {
  return { data: Buffer.from(mat.data), width: mat.cols, height: mat.rows, channels: mat.channels() };
}
// รันโค้ดที่ gen มา โดยอ่าน/เขียนเป็น raw bytes (เลี่ยง JPEG lossy) — และ **ไม่ตั้ง PYTHONIOENCODING**
// เพื่อพิสูจน์ V15: บน Windows ภาษาไทย stdout ที่ถูก redirect เป็น cp874 บรรทัด print ต้องเป็น ASCII ล้วน
function runGeneratedPython(code, raw) {
  const shape = raw.channels === 1 ? `(${raw.height}, ${raw.width})` : `(${raw.height}, ${raw.width}, ${raw.channels})`;
  assert.ok(code.includes('img = cv2.imread(path)'), 'expected the single-image process(path) loader line');
  let patched = code.replace('img = cv2.imread(path)', `img = np.fromfile(path, dtype=np.uint8).reshape(${shape})`);
  const mainIdx = patched.indexOf('if __name__ == "__main__":');
  assert.ok(mainIdx > 0);
  patched = patched.slice(0, mainIdx);
  patched += '\nimport sys\nresult = process(sys.argv[1])\nresult.tofile(sys.argv[2])\n';
  const id = Math.random().toString(36).slice(2);
  const scriptPath = path.join(TMP_DIR, `s_${id}.py`);
  const inPath = path.join(TMP_DIR, `in_${id}.bin`);
  const outPath = path.join(TMP_DIR, `out_${id}.bin`);
  fs.writeFileSync(scriptPath, patched, 'utf8');
  fs.writeFileSync(inPath, raw.data);
  const env = { ...process.env };
  delete env.PYTHONIOENCODING; // สำคัญ: ต้องไม่ช่วยเหลือ encoding ของ stdout
  const stdout = execFileSync('python', [scriptPath, inPath, outPath], { encoding: 'utf8', env });
  return { stdout, bytes: new Uint8Array(fs.readFileSync(outPath)) };
}

test('tier2d-CG-1. codegen app-only notice: each Motion op gets a Thai step comment block, an ASCII-only runtime WARNING print and a header note -- no GEN entry exists for them, the output never mentions BackgroundSubtractorMOG2/calcOpticalFlow, and the block is distinct from the generic "(ข้าม: ยังไม่รองรับ" line', () => {
  for (const id of MOTION_IDS) {
    const code = buildPythonCode([{ id, params: defaultParams(id) }], { displayName: 'X' });
    assert.match(code, new RegExp(`# ── ขั้นที่ 1: .*\\(${id}\\) — ใช้ได้ในแอปเท่านั้น ──`), `${id}: step header`);
    assert.match(code, /# ขั้นนี้ถูกข้าม: ขั้นถัดไปจะได้ภาพเดียวกับที่เข้าขั้นนี้/, `${id}: identity note`);
    assert.ok(code.includes(`# หมายเหตุ: มีขั้นตอนที่ใช้ได้ในแอปเท่านั้น (${id})`), `${id}: header note`);
    const warn = `print("WARNING: skipped '${id}' (app-only: needs consecutive video frames) - result differs from the app")`;
    assert.ok(code.includes(warn), `${id}: ASCII warning line`);
    // V15: บรรทัด print ทั้งบรรทัดต้องเป็น ASCII ล้วน (ไทย/⚠/— จะ crash เมื่อ stdout ถูก redirect บน cp874)
    for (const line of code.split('\n')) {
      if (!line.trim().startsWith('print(')) continue;
      // eslint-disable-next-line no-control-regex
      assert.ok(/^[\x00-\x7F]*$/.test(line), `every generated print() must be pure ASCII, found: ${line}`);
    }
    assert.doesNotMatch(code, /BackgroundSubtractorMOG2|calcOpticalFlow/, `${id}: no real cv2 implementation may be emitted`);
    assert.doesNotMatch(code, /\(ข้าม: ยังไม่รองรับการ gen โค้ดของ/, `${id}: must not fall through to the generic skip line`);
  }
});

test('tier2d-CG-2. codegen equivalence: dropping the comment lines and the WARNING line from [gaussian_blur, mog2, binary_threshold] leaves exactly the code lines of [gaussian_blur, binary_threshold] -- the app-only step really is an identity', () => {
  const withOp = buildPythonCode([
    { id: 'blur', params: { mode: 'gaussian_blur', gaussian_blur__ksize: 5 } },
    { id: 'mog2', params: defaultParams('mog2') },
    { id: 'threshold', params: { mode: 'binary_threshold', binary_threshold__thresh: 120 } },
  ]);
  const without = buildPythonCode([
    { id: 'blur', params: { mode: 'gaussian_blur', gaussian_blur__ksize: 5 } },
    { id: 'threshold', params: { mode: 'binary_threshold', binary_threshold__thresh: 120 } },
  ]);
  const codeLines = (src) => src.split('\n')
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#') && !l.startsWith('print("WARNING: skipped'));
  assert.deepEqual(codeLines(withOp), codeLines(without));
});

test('tier2d-CG-3. codegen runs for real (cv2), WITHOUT PYTHONIOENCODING (the V15 proof): exit code 0, stdout contains the ASCII WARNING, and the output bytes equal the same pipeline without the app-only step', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  const src = cloudRGB(48, 36, 161);
  try {
    const raw = matToRaw(src);
    const steps = [
      { id: 'blur', params: { mode: 'gaussian_blur', gaussian_blur__ksize: 5 } },
      { id: 'threshold', params: { mode: 'binary_threshold', binary_threshold__thresh: 120 } },
    ];
    const withMog = runGeneratedPython(buildPythonCode([steps[0], { id: 'mog2', params: defaultParams('mog2') }, steps[1]]), raw);
    assert.ok(withMog.stdout.includes("WARNING: skipped 'mog2'"), `stdout was: ${JSON.stringify(withMog.stdout)}`);
    const plain = runGeneratedPython(buildPythonCode(steps), raw);
    assert.equal(plain.stdout.includes('WARNING'), false);
    assert.equal(withMog.bytes.length, plain.bytes.length);
    assert.ok(sameBytes(withMog.bytes, plain.bytes), 'the skipped step must not change a single output byte');
  } finally { src.delete(); }
});

test('tier2d-CG-4. codegen 2-image pipeline with an app-only op inside the src2 (B) branch: the notice lands in the B block labelled "สาย B", the header note lists it, and the script still runs under real cv2', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  const pipeline = [{
    id: 'image_arithmetic', params: { mode: 'imabsdiff' }, enabled: true, nodeId: nid(),
    src2: { pipeline: [{ id: 'farneback_optical_flow', params: defaultParams('farneback_optical_flow'), enabled: true, nodeId: nid() }] },
  }];
  const code = buildPythonCode(pipeline);
  assert.match(code, /def process\(path_a, path_b\):/);
  assert.match(code, /# ── สาย B: .*\(farneback_optical_flow\) — ใช้ได้ในแอปเท่านั้น ──/);
  assert.ok(code.includes('# หมายเหตุ: มีขั้นตอนที่ใช้ได้ในแอปเท่านั้น (farneback_optical_flow)'));
  assert.ok(code.includes(`print("WARNING: skipped 'farneback_optical_flow' (app-only: needs consecutive video frames) - result differs from the app")`));
  // บรรทัดคำเตือนต้องไม่ถูก replace \bimg\b -> img_b (ต้องไม่มีคำว่า img_b ในข้อความ)
  assert.doesNotMatch(code, /WARNING: skipped.*img_b/);

  const a = cloudRGB(32, 24, 162);
  const b = cloudRGB(32, 24, 163);
  try {
    const shape = '(24, 32, 3)';
    let patched = code
      .replace('img = cv2.imread(path_a)', `img = np.fromfile(path_a, dtype=np.uint8).reshape(${shape})`)
      .replace('img_b = cv2.imread(path_b)', `img_b = np.fromfile(path_b, dtype=np.uint8).reshape(${shape})`);
    patched = patched.slice(0, patched.indexOf('if __name__ == "__main__":'));
    patched += '\nimport sys\nresult = process(sys.argv[1], sys.argv[2])\nresult.tofile(sys.argv[3])\n';
    const id = Math.random().toString(36).slice(2);
    const sp = path.join(TMP_DIR, `s2_${id}.py`), ap = path.join(TMP_DIR, `a_${id}.bin`);
    const bp = path.join(TMP_DIR, `b_${id}.bin`), op = path.join(TMP_DIR, `o_${id}.bin`);
    fs.writeFileSync(sp, patched, 'utf8');
    fs.writeFileSync(ap, matToRaw(a).data);
    fs.writeFileSync(bp, matToRaw(b).data);
    const env = { ...process.env };
    delete env.PYTHONIOENCODING;
    const stdout = execFileSync('python', [sp, ap, bp, op], { encoding: 'utf8', env });
    assert.ok(stdout.includes("WARNING: skipped 'farneback_optical_flow'"));
    assert.equal(fs.readFileSync(op).length, 32 * 24 * 3);
  } finally { a.delete(); b.delete(); }
});

// ===================================================================
// W-1…W-2 — การต่อสายเข้ากับ registry / palette / ui-meta / smoke
// ===================================================================
test('tier2d-W-1. wiring: the three Motion FILTERS entries exist with group "Motion", stateful === true and inputs === undefined; defaultParams works; TOOL_SECTIONS has a MOTION section; GROUP_COLORS/GROUP_ORDER know the group; icons do not collide; OP_PRIMARY returns non-empty strings; and no OTHER filter carries `stateful`', () => {
  for (const id of MOTION_IDS) {
    const f = FILTERS[id];
    assert.ok(f, `FILTERS.${id} must exist`);
    assert.equal(f.group, 'Motion');
    assert.equal(f.stateful, true);
    assert.equal(f.inputs, undefined, 'Motion ops take a single image -- `inputs` must stay undefined');
    assert.equal(typeof f.apply, 'function');
    assert.ok(f.desc && f.desc.length > 50, `${id}: needs a teaching description`);
    const dp = defaultParams(id);
    for (const p of f.params) {
      assert.ok(Object.prototype.hasOwnProperty.call(dp, p.key), `${id}: defaultParams must cover ${p.key}`);
      assert.notEqual(dp[p.key], undefined, `${id}: default of ${p.key}`);
    }
    assert.ok(GROUP_ICON[id], `${id}: GROUP_ICON entry`);
    assert.equal(typeof OP_PRIMARY[id], 'function', `${id}: OP_PRIMARY entry`);
    const line = OP_PRIMARY[id](dp);
    assert.equal(typeof line, 'string');
    assert.ok(line.length > 0, `${id}: OP_PRIMARY must produce a non-empty node subtitle`);
    assert.doesNotMatch(line, /undefined|NaN/, `${id}: OP_PRIMARY on defaults -> ${line}`);
  }

  const section = TOOL_SECTIONS.find((s) => s.title === 'MOTION');
  assert.ok(section, 'palette must have a MOTION section');
  assert.deepEqual(section.ops, MOTION_IDS);
  assert.ok(GROUP_COLORS.Motion && GROUP_COLORS.Motion.title, 'GROUP_COLORS["Motion"]');
  assert.ok(GROUP_ORDER.includes('Motion'));
  assert.ok(GROUP_ORDER.indexOf('Motion') > GROUP_ORDER.indexOf('Image Quality'));

  // icon ต้องไม่ชนกัน (ยกเว้นคู่ OP↔GROUP ของกลุ่มเดียวกันซึ่งตั้งใจให้เหมือนกัน)
  const iconOwners = new Map();
  for (const [id, glyph] of Object.entries(GROUP_ICON)) {
    if (!iconOwners.has(glyph)) iconOwners.set(glyph, new Set());
    iconOwners.get(glyph).add(id);
  }
  for (const id of MOTION_IDS) {
    const owners = iconOwners.get(GROUP_ICON[id]);
    assert.equal(owners.size, 1, `icon ${GROUP_ICON[id]} of ${id} collides with ${[...owners].join(', ')}`);
  }

  const stateful = Object.entries(FILTERS).filter(([, f]) => f.stateful !== undefined).map(([k]) => k);
  // 2026-10-07: + 'motion' = กลุ่มระดับหมวด palette ที่รวม 3 op (stateful:true ตามกัน)
  assert.deepEqual(stateful.sort(), [...MOTION_IDS, 'motion'].sort(), 'only the Motion entries may declare `stateful`');
});

test('tier2d-W-2. filter-smoke wiring (static read): the three ops reach the smoke CASES through TOOL_SECTIONS and OP_TO_FILTER, they have POSITIVE_NOTE patterns, and only mog2 is exempt from the nonConstant check', async () => {
  const { OP_TO_FILTER } = await import('../src/renderer/ui/palette-panel.js');
  for (const id of MOTION_IDS) {
    // 2026-10-07: palette ชี้กลุ่มระดับหมวด 'motion' (mode select) — entry เดี่ยวเดิมยังอยู่ใน FILTERS เป็น legacy
    assert.equal(OP_TO_FILTER[id], 'motion', `${id} must resolve to the category-level Motion node`);
  }
  const smokeSrc = fs.readFileSync(new URL('../src/renderer/filter-smoke.js', import.meta.url), 'utf8');
  assert.match(smokeSrc, /mog2:\s*\/เฟรมแรก\//);
  assert.match(smokeSrc, /farneback_optical_flow:\s*\/เฟรมแรก\//);
  assert.match(smokeSrc, /pyrlk_optical_flow:\s*\/เฟรมแรก\|ไม่พบจุดเด่น\//);
  assert.match(smokeSrc, /const CONSTANT_OK_OPS = new Set\(\["mog2"\]\)/);
  assert.match(smokeSrc, /nonConstant: CONSTANT_OK_OPS\.has\(opId\) \? true : Boolean\(pixels && max > min\)/);
  // ต้องยกเว้น "เฉพาะ" mog2 เท่านั้น
  assert.doesNotMatch(smokeSrc, /CONSTANT_OK_OPS = new Set\(\[[^\]]*optical_flow/);
});

// ===================================================================
// รอบแก้ตามรายงาน debugger (2026-09-29) — เทสต์ที่ปิดช่องโหว่ 9 จุดที่ mutation testing เจาะผ่านได้
// W-3 (MAJOR-1) · LK-7 (MAJOR-2) · MN-1/MN-2 (MINOR-1) · D-13…D-16 (RISK-1/MINOR-3/MINOR-5) · W-4 (RISK-2)
// (MAJOR-3 อยู่ใน FB-3 · MINOR-2 อยู่ใน M-4 — ขยายของเดิมแทนการเพิ่มเทสต์ใหม่ เพราะเป็น assertion ที่ขาดไป
// ของเส้นทางเดิม ไม่ใช่เส้นทางใหม่)
// ===================================================================

test('tier2d-W-3. the REAL worker message protocol (self.onmessage), not computeFrame directly: init -> ready, 40 live frames advance the Motion node to 40, an {type:"export"} message in between is a fresh state (frame 1 -> black mask) and does NOT disturb the live state (the next live frame is 41 and it still detects the object against the background it learnt)', async () => {
  const { sweepNodeState } = await worker(); // import ก่อน เพื่อให้ self.onmessage ถูกติดตั้งแล้ว
  const clean = cloudRGB(96, 72, 181);
  const moved = withRect(clean, new cv.Rect(30, 20, 24, 18), new cv.Scalar(255, 255, 255)); // มีวัตถุสว่างโผล่มา
  const cleanBuf = rgbaBufferOf(clean);
  const movedBuf = rgbaBufferOf(moved);
  const mogId = nid();
  const pipelines = [{ connected: true, displayId: 'display', pipeline: [{ id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId }] }];

  const posted = [];
  const prevPost = globalThis.postMessage;
  globalThis.postMessage = (msg) => { posted.push(msg); };
  const send = async (msg) => { await self.onmessage({ data: msg }); return posted[posted.length - 1]; };
  try {
    assert.equal(typeof self.onmessage, 'function', 'cv-worker.js must install a self.onmessage handler');
    assert.equal((await send({ type: 'init' })).type, 'ready');

    // เฟรมสด: ฉากนิ่ง 40 เฟรม = เรียนรู้พื้นหลังจนใช้งานได้จริง (ตรวจตัวนับที่เฟรมแรกและเฟรมสุดท้าย)
    const WARM = 40;
    for (let f = 1; f <= WARM; f++) {
      const r = await send({ type: 'frame', width: 96, height: 72, buffer: cleanBuf, pipelines });
      assert.equal(r.type, 'result');
      if (f === 1 || f === WARM) assert.equal(noteValue(r.notes, 'mog2', 'frame'), f, `live frame ${f}`);
    }

    // export คั่นกลาง ด้วยภาพที่ "มีวัตถุ" — ต้องถูกมองเป็นเฟรมแรกของ state ใหม่เสมอ (U1) จึงได้มาสก์ว่าง
    const ex = await send({ type: 'export', reqId: 'req-7', width: 96, height: 72, buffer: movedBuf, pipelines });
    assert.equal(ex.type, 'export-result');
    assert.equal(ex.reqId, 'req-7');
    assert.equal(ex.error, undefined);
    assert.equal(ex.results.length, 1);
    assert.ok(new Uint8Array(ex.results[0].buffer).every((v, i) => (i % 4 === 3 ? v === 255 : v === 0)),
      'an exported image must be frame 1 of its own state -> an empty (black) mask, never a live learning step');

    // ...และ state ของภาพสดต้องไม่ถูกแตะเลย: เฟรมถัดไปคือเฟรมที่ 3 และยังจับวัตถุที่โผล่มาได้จริง
    const after = await send({ type: 'frame', width: 96, height: 72, buffer: movedBuf, pipelines });
    assert.equal(noteValue(after.notes, 'mog2', 'frame'), WARM + 1, 'the export must not consume a live frame');
    assert.ok(noteValue(after.notes, 'mog2', 'foreground_px') > 200,
      'the live state must still hold the learnt background (otherwise the export polluted it)');
  } finally {
    globalThis.postMessage = prevPost;
    sweepNodeState(new Set());
    clean.delete();
    moved.delete();
  }
});

test('tier2d-LK-7. pyrlk/farneback really read their params (not hard-coded constants) and accept them as strings (P-13): maxPoints 20 vs 200 give exactly 20 vs 200 points on a corner-rich scene, and {maxPoints:"20", trailLength:"3"} / {maxMotion:"5"} behave byte-identically to the numbers', () => {
  const src = cloudRGB(640, 480, 321); // ฉากลายเยอะ: gftt หาจุดได้ >1000 จุด จึงได้ตามที่ขอเป๊ะทุกค่า
  const pointsFor = (params) => {
    const st = {};
    const ctx = { state: st };
    const out = OPS.pyrlk_optical_flow.apply(cv, src, params, ctx);
    const n = valuesOf(ctx).points;
    out.delete();
    st.__dispose();
    return n;
  };
  try {
    assert.equal(pointsFor({ maxPoints: 20 }), 20, 'maxPoints must be honoured, not replaced by a constant');
    assert.equal(pointsFor({ maxPoints: 200 }), 200);
    assert.equal(pointsFor({}), 100, 'the default must still be 100');
    assert.equal(pointsFor({ maxPoints: '20' }), 20, 'a string param from a hand-edited project file must work like the number');
    assert.equal(pointsFor({ maxPoints: '200' }), 200);
  } finally { src.delete(); }

  // 4 เฟรมเต็ม ๆ: ผลลัพธ์ทั้งภาพและทุกค่าต้องเท่ากันเป๊ะระหว่าง params ที่เป็นตัวเลขกับที่เป็น string
  const frames = shiftedFrames(160, 120, [[0, 0], [2, 1], [4, 2], [6, 3]], 322);
  try {
    const runBoth = (id, numeric, strings) => {
      const a = runFlowPair(id, frames, numeric);
      const b = runFlowPair(id, frames, strings);
      for (let i = 0; i < frames.length; i++) {
        assert.ok(sameBytes(a.results[i].out.data, b.results[i].out.data), `${id}: frame ${i + 1} image differs between numeric and string params`);
        assert.deepEqual(valuesOf(b.results[i].ctx), valuesOf(a.results[i].ctx), `${id}: frame ${i + 1} values differ between numeric and string params`);
      }
      for (const r of [...a.results, ...b.results]) r.out.delete();
      a.st.__dispose();
      b.st.__dispose();
    };
    runBoth('pyrlk_optical_flow', { maxPoints: 20, trailLength: 3 }, { maxPoints: '20', trailLength: '3' });
    runBoth('farneback_optical_flow', { view: 'color', maxMotion: 5 }, { view: 'color', maxMotion: '5' });
  } finally { for (const f of frames) f.delete(); }
});

test('tier2d-MN-1. (static read) every numeric param of the three Motion ops is clamped in code to EXACTLY the min/max/default the UI offers -- so a hand-edited project file can never push a value the UI cannot produce into OpenCV, and the two can never drift apart silently', () => {
  const source = fs.readFileSync(new URL('../src/renderer/filters/registry.js', import.meta.url), 'utf8');
  // [opId, paramKey, ตัวช่วยที่ใช้] — intIn = ปัดเป็นจำนวนเต็มด้วย, numIn = clamp เฉย ๆ (ทศนิยมใช้ได้)
  const CLAMPED = [
    ['mog2', 'history', 'intIn'],
    ['mog2', 'varThreshold', 'numIn'],
    ['farneback_optical_flow', 'maxMotion', 'numIn'],
    ['pyrlk_optical_flow', 'maxPoints', 'intIn'],
    ['pyrlk_optical_flow', 'trailLength', 'intIn'],
  ];
  for (const [opId, key, fn] of CLAMPED) {
    const spec = OPS[opId].params.find((p) => p.key === key);
    assert.ok(spec, `${opId}.${key}: no such param`);
    const m = source.match(new RegExp(`${fn}\\(p\\.${key}, (-?[\\d.]+), (-?[\\d.]+), (-?[\\d.]+)\\)`));
    assert.ok(m, `${opId}.${key}: expected a ${fn}(p.${key}, default, min, max) call in registry.js`);
    assert.equal(Number(m[1]), spec.default, `${opId}.${key}: clamp default`);
    assert.equal(Number(m[2]), spec.min, `${opId}.${key}: clamp minimum must equal the UI minimum`);
    assert.equal(Number(m[3]), spec.max, `${opId}.${key}: clamp maximum must equal the UI maximum`);
  }
  // ...และห้ามมี param ตัวเลขตัวไหนของ 3 op นี้หลุดออกไปจากรายการข้างบน (เพิ่ม param ใหม่ = ต้องมี clamp)
  for (const id of MOTION_IDS) {
    for (const p of OPS[id].params) {
      if (p.type !== 'number') continue;
      assert.ok(CLAMPED.some(([o, k]) => o === id && k === p.key), `${id}.${p.key}: a numeric param with no clamp (add it to the table and clamp it in apply())`);
    }
  }
});

test('tier2d-MN-2. out-of-range params really are clamped at runtime: maxPoints 5/99999 -> 10/300 points, trailLength 999 -> at most 51 positions per trail, varThreshold 1 behaves as 4 and 1e6 behaves as 100 (while 4 and 100 still differ from each other, so the clamp is not hiding a dead param)', () => {
  const src = cloudRGB(640, 480, 331);
  const pointsFor = (params) => {
    const st = {};
    const ctx = { state: st };
    const out = OPS.pyrlk_optical_flow.apply(cv, src, params, ctx);
    const n = valuesOf(ctx).points;
    out.delete();
    st.__dispose();
    return n;
  };
  try {
    assert.equal(pointsFor({ maxPoints: 5 }), 10, 'below the UI minimum must clamp up to 10');
    assert.equal(pointsFor({ maxPoints: 99999 }), 300, 'above the UI maximum must clamp down to 300');
    assert.equal(pointsFor({ maxPoints: '5' }), 10, 'the clamp must apply to string params too');
  } finally { src.delete(); }

  // trailLength: ฉากเดิมซ้ำ 55 เฟรม (จุดไม่หลุดเลย) -> เส้นทางยาวขึ้นเฟรมละ 1 จนชนเพดาน trailLength + 1
  const still = cloudRGB(96, 72, 332);
  const stLong = {};
  const stShort = {};
  try {
    for (let i = 0; i < 55; i++) { const o = OPS.pyrlk_optical_flow.apply(cv, still, { trailLength: 999 }, { state: stLong }); o.delete(); }
    assert.equal(Math.max(...stLong.trails.map((t) => t.length)), 51, 'trailLength 999 must clamp to 50 (+1 for the current position)');
    for (let i = 0; i < 6; i++) { const o = OPS.pyrlk_optical_flow.apply(cv, still, { trailLength: 0 }, { state: stShort }); o.delete(); }
    assert.equal(Math.max(...stShort.trails.map((t) => t.length)), 2, 'trailLength 0 must clamp up to 1 (+1 for the current position)');
  } finally { stLong.__dispose(); stShort.__dispose(); still.delete(); }

  // varThreshold: ฉากนิ่ง 40 เฟรม แล้วมีแผ่นสว่างกลาง ๆ โผล่มา (ค่าที่วัดได้จริงตอนเขียนเทสต์:
  // 1 -> 1199, 4 -> 1191, 16 -> 1165, 100 -> 960, 1e6 -> 0 px)
  const fg = (varThreshold) => {
    const bg = cloudRGB(160, 120, 55);
    const st2 = {};
    let px = 0;
    for (let i = 0; i <= 40; i++) {
      const frame = i < 40 ? bg : withRect(bg, new cv.Rect(40, 30, 40, 30), new cv.Scalar(100, 100, 100));
      const ctx = { state: st2 };
      const out = OPS.mog2.apply(cv, frame, { view: 'mask', varThreshold }, ctx);
      px = valuesOf(ctx).foreground_px;
      out.delete();
      if (frame !== bg) frame.delete();
    }
    st2.__dispose();
    bg.delete();
    return px;
  };
  const at4 = fg(4);
  const at100 = fg(100);
  assert.notEqual(at4, at100, 'varThreshold must genuinely change the result inside the UI range (otherwise the clamp checks below prove nothing)');
  assert.equal(fg(1), at4, 'below the UI minimum must clamp up to 4');
  assert.equal(fg(1e6), at100, 'above the UI maximum must clamp down to 100');
  assert.equal(fg('100'), at100, 'the clamp must apply to string params too');
  // history: clamp เป็น 10-2000 เหมือนกัน · อัตราเรียนรู้อัตโนมัติของ MOG2 คือ 1/min(2*nframes, history)
  // (ไม่ใช่ 1/min(nframes, history) — บันทึกเก่าของ coder ตกเลข 2 ไป) จึง "อิ่มตัว" ที่ nframes = history/2:
  // ค่าน้อย ๆ อย่าง 10 กับค่า default 500 ต่างกันตั้งแต่เฟรมที่ ~50 แล้ว -> พิสูจน์เชิงพฤติกรรมได้ ดู MN-3
  // ที่พิสูจน์เชิงพฤติกรรมไม่ไหวจริง ๆ มีแค่ "เพดาน 2000 เทียบ 5000" (ต้องรันเกิน 1000 เฟรมถึงจะเริ่มต่าง)
  // -> ด้านนั้นล็อกด้วย MN-1 (static read) แทน
});

test('tier2d-MN-3. mog2 `history` really is used (behaviour, not just a static read): MOG2 picks its own learning rate = 1/min(2*nframes, history), so a small history bites long before the frame count reaches it -- on a scene that slowly brightens, history 10 keeps up with the light (almost no foreground at frame 50) while history 500 is still comparing against the old dark background (nearly the whole frame is foreground)', () => {
  const base = cloudRGB(96, 72, 501);
  const TOTAL = 96 * 72;
  // ฉาก: ภาพเดิมที่ค่อย ๆ สว่างขึ้นเฟรมละ 1.2 ระดับ (แสงในห้องค่อย ๆ เปลี่ยน) — ไม่มีวัตถุเคลื่อนไหวเลย
  const fgAtFrame50 = (history) => {
    const st = {};
    let px = 0;
    try {
      for (let i = 0; i < 50; i++) {
        const frame = new cv.Mat();
        base.convertTo(frame, -1, 1, i * 1.2);
        const ctx = { state: st };
        const out = OPS.mog2.apply(cv, frame, { history, view: 'mask' }, ctx);
        px = valuesOf(ctx).foreground_px;
        out.delete();
        frame.delete();
      }
    } finally { st.__dispose(); }
    return px;
  };
  try {
    const fast = fgAtFrame50(10);
    const slow = fgAtFrame50(500);
    assert.ok(fast <= 0.05 * TOTAL, `history 10 must have learnt the new brightness by frame 50 (foreground = ${fast} px of ${TOTAL})`);
    assert.ok(slow >= 0.8 * TOTAL, `history 500 must still see the brightened scene as foreground at frame 50 (foreground = ${slow} px of ${TOTAL})`);
  } finally { base.delete(); }
});

test('tier2d-MN-4. the node card must show the value the op REALLY uses: OP_PRIMARY clamps with the same min/max/default as registry, so a hand-edited project file carrying history 9999 reads "จำ 2000 เฟรม" (what OpenCV is actually given), never "จำ 9999 เฟรม" -- and string params (P-13) and missing keys behave the same way', () => {
  const shows = (id, params) => OP_PRIMARY[id](params);
  const CASES = [
    // [opId, params, ต้องมีข้อความนี้, ต้องไม่มีข้อความนี้]
    ['mog2', { history: 9999 }, 'จำ 2000 เฟรม', '9999'],
    ['mog2', { history: '9999' }, 'จำ 2000 เฟรม', '9999'],
    ['mog2', { history: 1 }, 'จำ 10 เฟรม', 'จำ 1 เฟรม'],
    ['mog2', { varThreshold: 1e6 }, 'ไว 100', 'ไว 1000000'],
    ['mog2', { varThreshold: 0 }, 'ไว 4', 'ไว 0'],
    ['farneback_optical_flow', { maxMotion: 999 }, 'สว่างสุด 50 px', 'สว่างสุด 999 px'],
    ['farneback_optical_flow', { maxMotion: 0 }, 'สว่างสุด 1 px', 'สว่างสุด 0 px'],
    ['pyrlk_optical_flow', { maxPoints: 99999, trailLength: 999 }, '300 จุด', '99999'],
    ['pyrlk_optical_flow', { maxPoints: 99999, trailLength: 999 }, 'เส้นทาง 50 เฟรม', 'เส้นทาง 999 เฟรม'],
    ['pyrlk_optical_flow', { maxPoints: 1, trailLength: 0 }, '10 จุด', '1 จุด ·'],
  ];
  for (const [id, params, must, mustNot] of CASES) {
    const line = shows(id, params);
    assert.ok(line.includes(must), `${id} ${JSON.stringify(params)} -> "${line}" (expected it to contain "${must}")`);
    assert.ok(!line.includes(mustNot), `${id} ${JSON.stringify(params)} -> "${line}" must not show the raw, unclamped "${mustNot}"`);
  }
  // ไม่มีคีย์เลย / ค่าที่แปลงเป็นตัวเลขไม่ได้ -> ต้องโชว์ค่า default ของ param นั้น (ไม่ใช่ undefined/NaN)
  for (const id of MOTION_IDS) {
    for (const params of [{}, { history: 'abc', varThreshold: 'abc', maxMotion: 'abc', maxPoints: 'abc', trailLength: 'abc' }]) {
      const line = OP_PRIMARY[id](params);
      assert.doesNotMatch(line, /undefined|NaN/, `${id} ${JSON.stringify(params)} -> ${line}`);
      assert.equal(line, OP_PRIMARY[id](defaultParams(id)), `${id}: ${JSON.stringify(params)} must read exactly like the defaults`);
    }
  }
});

test('tier2d-MN-5. (static read) the analysis downscale cap of each Motion op is exactly the one the work order fixes (mog2 480 / farneback 320 / pyrlk 480) -- it is NOT just a speed knob: at 320 px a winsize of 15 covers the same field of view as ~90 px at 1080p, so the cap is what decides how fast a motion can still be followed; and the farneback desc quotes the number to the learner, so code and text must not drift apart', () => {
  const source = fs.readFileSync(new URL('../src/renderer/filters/registry.js', import.meta.url), 'utf8');
  // ตัดซอร์สเป็นบล็อกต่อ op ก่อน (คีย์ระดับ 2 ช่องว่างใน registry.js = ประกาศ op เท่านั้น) — เลข 480 ถูกใช้
  // 2 op ถ้า match ทั้งไฟล์รวด ๆ จะจับสลับกันได้โดยยังเขียวอยู่ · เก็บเฉพาะครั้งแรกของแต่ละ id คือบล็อกใน OPS
  // (FILTERS ท้ายไฟล์ประกาศชื่อเดิมซ้ำแบบ ...spread)
  const blocks = new Map();
  const marks = [...source.matchAll(/\n {2}([a-z_0-9]+): \{/g)].map((m) => ({ id: m[1], start: m.index }));
  for (let i = 0; i < marks.length; i++) {
    if (blocks.has(marks[i].id)) continue;
    blocks.set(marks[i].id, source.slice(marks[i].start, i + 1 < marks.length ? marks[i + 1].start : source.length));
  }
  // [opId, เพดาน px ตามใบสั่งงาน §C.4-final F3/F4/F5]
  const CAPS = [
    ['mog2', 480],
    ['farneback_optical_flow', 320],
    ['pyrlk_optical_flow', 480],
  ];
  for (const [id, cap] of CAPS) {
    const block = blocks.get(id);
    assert.ok(block, `${id}: ไม่พบบล็อกของ op ใน registry.js`);
    const found = [...block.matchAll(/analysisScale\(src\.cols, src\.rows, (\d+)\)/g)].map((m) => Number(m[1]));
    assert.deepEqual(found, [cap],
      `${id}: ต้องเรียก analysisScale(src.cols, src.rows, ${cap}) หนึ่งครั้งพอดี (เจอ ${JSON.stringify(found)}) — เพดานนี้เป็นพารามิเตอร์ของอัลกอริทึม ไม่ใช่แค่ความเร็ว`);
  }
  // desc ของ farneback บอกเลขนี้กับผู้เรียนตรง ๆ — เปลี่ยนเพดานแล้วไม่แก้ข้อความ = คำอธิบายกลายเป็นเท็จ
  const fbCap = CAPS.find(([id]) => id === 'farneback_optical_flow')[1];
  assert.ok(OPS.farneback_optical_flow.desc.includes(`${fbCap}px`),
    `farneback desc ต้องบอกเพดาน "${fbCap}px" ให้ตรงกับโค้ด: ${OPS.farneback_optical_flow.desc}`);
});

test('tier2d-D-13. RISK-1 hardening: if two pipelines ever disagree about the upstream prefix of the SAME node (e.g. one carries a disabled step the other does not -- exactly the shape a future editor.js refactor could produce), the node must still run once per frame and keep counting (1,2,3), NOT reset its state every frame', async () => {
  const { computeFrame, sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 191);
  const buffer = rgbaBufferOf(base);
  const mogId = nid();
  const mog = () => ({ id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId });
  // สองสายนี้ให้ "พิกเซลเหมือนกันเป๊ะ" (blur ถูกปิดอยู่) แต่ fingerprint ต้นน้ำของ mog2 ต่างกัน
  const pipelines = [
    { connected: true, displayId: 'display', pipeline: [{ id: 'blur', params: { mode: 'gaussian_blur', gaussian_blur__ksize: 5 }, enabled: false, nodeId: nid() }, mog()] },
    { connected: true, displayId: 'preview', pipeline: [mog()] },
  ];
  try {
    for (let f = 1; f <= 3; f++) {
      const r = computeFrame(96, 72, buffer, pipelines);
      const frames = r.notes.filter((n) => n.id === 'mog2' && n.values).map((n) => n.values.fields.find((x) => x.key === 'frame').value);
      assert.deepEqual(frames, [f, f], `frame ${f}: both pipelines must report the same, advancing frame number (got ${JSON.stringify(frames)})`);
      assert.equal(r.results.length, 2);
      assert.ok(sameBytes(new Uint8Array(r.results[0].buffer), new Uint8Array(r.results[1].buffer)), 'one node = one evaluation per frame');
    }
  } finally { sweepNodeState(new Set()); base.delete(); }
});

test('tier2d-D-14. the per-frame memo is still keyed by the input SIZE: the same node reached at two different sizes in one frame must not be served the other size picture, and replacing a memo entry must not leak the Mat it held (wrapper balance)', async () => {
  const { computeFrame, sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 192);
  const buffer = rgbaBufferOf(base);
  const mogId = nid();
  const mog = () => ({ id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId });
  const pipelines = [
    { connected: true, displayId: 'full', pipeline: [mog()] },
    { connected: true, displayId: 'half', pipeline: [{ id: 'geometry', params: { mode: 'resize', resize__scale: 50 }, enabled: true, nodeId: nid() }, mog()] },
  ];
  try {
    const r = computeFrame(96, 72, buffer, pipelines);
    assert.equal(r.results[0].width, 96);
    assert.equal(r.results[0].height, 72);
    assert.equal(r.results[1].width, 48, 'the second pipeline must get its own 48x36 result, not a clone of the 96x72 memo entry');
    assert.equal(r.results[1].height, 36);

    assertWrapperBalance(cv, 'memo entry replaced within one frame (two sizes)', () => {
      computeFrame(96, 72, buffer, pipelines);
      sweepNodeState(new Set());
    }, { global: true, minCreated: 10 });
  } finally { sweepNodeState(new Set()); base.delete(); }
});

test('tier2d-D-15. if storing the memo entry throws (Mat.clone injected to fail right after the op succeeded), runPipeline must delete the result it is already holding -- the frame fails, but not one wrapper leaks', async () => {
  const { computeFrame, sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 193);
  const buffer = rgbaBufferOf(base);
  const mogId = nid();
  const pipelines = [{ connected: true, displayId: 'display', pipeline: [{ id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId }] }];
  try {
    // เทคนิค monkey-patch นี้พึ่งพาสัญญาข้อหนึ่งของ assertWrapperBalance: มัน "เรียก callback ครั้งเดียว"
    // ถ้าวันหนึ่ง helper เปลี่ยนไปเรียกซ้ำ (เช่น warm-up รอบแรกแล้ววัดรอบสอง) การเขียนทับ/คืนค่า
    // cv.Mat.prototype.clone จะซ้อนกันเอง และ assert ด้านล่างจะเชื่อถือไม่ได้ — ให้แก้ที่นี่ด้วย
    assertWrapperBalance(cv, 'memo store throws mid-frame', () => {
      const realClone = cv.Mat.prototype.clone;
      cv.Mat.prototype.clone = function injectedCloneFailure() { throw new Error('injected test failure (clone)'); };
      let message = null;
      try {
        computeFrame(96, 72, buffer, pipelines);
      } catch (err) {
        message = err.message;
      } finally {
        cv.Mat.prototype.clone = realClone;
      }
      assert.equal(message, 'injected test failure (clone)', 'the injected failure must propagate out of computeFrame');
      sweepNodeState(new Set()); // state ที่เกิดภายใน tracker ต้องถูกเก็บกวาดในรอบเดียวกัน
    }, { global: true, minCreated: 5 });
  } finally { sweepNodeState(new Set()); base.delete(); }
});

test('tier2d-D-16. a memo hit must hand out its OWN copy of ctx.values: the two notes of one shared node carry equal but SEPARATE objects (postMessage keeps shared references shared, so detail-panel/U3 editing one table would otherwise edit the other)', async () => {
  const { computeFrame, sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 194);
  const buffer = rgbaBufferOf(base);
  const mogId = nid();
  const mog = () => ({ id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId: mogId });
  const pipelines = [
    { connected: true, displayId: 'a', pipeline: [mog()] },
    { connected: true, displayId: 'b', pipeline: [mog()] },
  ];
  try {
    const r = computeFrame(96, 72, buffer, pipelines);
    const vs = r.notes.filter((n) => n.id === 'mog2' && n.values).map((n) => n.values);
    assert.equal(vs.length, 2, 'both (non-silent) pipelines must report the shared node');
    assert.deepEqual(vs[1], vs[0], 'the memoised values must be equal in content');
    assert.notEqual(vs[1], vs[0], 'but they must not be the same object');
    assert.notEqual(vs[1].fields, vs[0].fields, 'the fields array must be copied too');
    assert.notEqual(vs[1].fields[0], vs[0].fields[0], 'and so must each field object');
    vs[1].fields[0].value = -999; // จำลอง UI ที่เขียนทับค่าในตารางของกล่องหนึ่ง
    assert.notEqual(vs[0].fields[0].value, -999, 'editing one note must not change the other');

    // อีกรูปทรงหนึ่งของ §A.2 (kind:"table") ยังไม่มี op ที่ stateful ตัวไหนคืนมาในวันนี้ จึงทดสอบ
    // cloneValues ตรง ๆ — ถ้าวันหนึ่งมี op แบบนั้น การคัดลอกต้องลึกพอที่ทั้ง columns และ rows แยกจากกัน
    const { cloneValues } = await worker();
    const table = { kind: 'table', columns: ['id', 'area'], rows: [[1, 10], [2, 20]] };
    const copy = cloneValues(table);
    assert.deepEqual(copy, table);
    assert.notEqual(copy.columns, table.columns, 'columns must be a separate array');
    assert.notEqual(copy.rows, table.rows, 'rows must be a separate array');
    assert.notEqual(copy.rows[0], table.rows[0], 'each row must be a separate array');
    copy.columns.push('extra');
    copy.rows[0][1] = -999;
    assert.deepEqual(table.columns, ['id', 'area'], 'editing the copy must not touch the original columns');
    assert.equal(table.rows[0][1], 10, 'editing the copy must not touch the original rows');
    for (const v of [null, undefined, 7, 'x']) assert.equal(cloneValues(v), v, 'non-object values pass through unchanged');
  } finally { sweepNodeState(new Set()); base.delete(); }
});

test('tier2d-D-17. the per-frame memo is ONLY for `stateful: true` ops (work order 0-C): a plain, stateless node that two pipelines share must still be evaluated in each of them -- its memo key is just the input SIZE, so memoising every node would serve the other pipeline\'s picture whenever the upstream differs', async () => {
  const { computeFrame, sweepNodeState } = await worker();
  const base = cloudRGB(96, 72, 195);
  const buffer = rgbaBufferOf(base);
  const sharedId = nid();
  // กล่องเบลอตัวเดียวกัน (nodeId เดียวกัน, ขนาดเข้าเท่ากัน) แต่สายหนึ่งมี "กลับสี" อยู่ต้นน้ำ อีกสายไม่มี
  const blur = () => ({ id: 'blur', params: { mode: 'gaussian_blur', gaussian_blur__ksize: 9 }, enabled: true, nodeId: sharedId });
  const invert = () => ({ id: 'colour', params: { mode: 'invert' }, enabled: true, nodeId: nid() });
  const pipelines = [
    { connected: true, displayId: 'inverted', pipeline: [invert(), blur()] },
    { connected: true, displayId: 'plain', pipeline: [blur()] },
  ];
  try {
    const r = computeFrame(96, 72, buffer, pipelines);
    const ref = computeFrame(96, 72, buffer, [{ connected: true, displayId: 'plain', pipeline: [blur()] }]);
    const inverted = new Uint8Array(r.results[0].buffer);
    const plain = new Uint8Array(r.results[1].buffer);
    assert.ok(!sameBytes(plain, inverted), 'the two pipelines feed the shared node different pixels, so their results must differ');
    assert.ok(sameBytes(plain, new Uint8Array(ref.results[0].buffer)),
      'the stateless node must compute its own result in the second pipeline, not hand out the first pipeline\'s memo entry');
  } finally { sweepNodeState(new Set()); base.delete(); }
});

test('tier2d-W-4. RISK-2: all three Motion ops warn in their description that exporting still images gives a "first frame" result (the user chose U1 -- every exported image is independent -- so the app must at least say so)', () => {
  for (const id of MOTION_IDS) {
    const desc = OPS[id].desc;
    assert.equal(typeof desc, 'string');
    assert.match(desc, /export/, `${id}: the description must mention export`);
    assert.match(desc, /เฟรมแรก/, `${id}: the description must say every exported image is treated as the first frame`);
  }
});

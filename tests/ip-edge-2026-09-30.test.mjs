// 2026-09-30 — acceptance tests ของชุด SEGMENTATION & EDGE (prewitt_edge / roberts_edge / log_edge /
// region_growing) ตามใบสั่งงาน Vision_boi/specs/ip_batch_2026-09-30.md §5
// ตัวเลขเกณฑ์ผ่านทุกตัวมาจากใบสั่งงาน (CV eng รันจริงและยืนยัน parity กับ Python cv2 = 0 pixel diff แล้ว)
// — ถ้าเทสต์ล้ม ให้ถือว่าโค้ดผิด ไม่ใช่ตัวเลขผิด
//
// นโยบาย leak (P-14, WORK-TRACKER §1): address-fingerprint ใช้ได้เฉพาะเทสต์แรกของไฟล์ (ipe-0) พร้อมคู่แฝด
// wrapper-balance (ipe-0w) — เทสต์อื่นใช้ assertLeakFree / assertNoHeapGrowth
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { OPS, FILTERS, defaultParams } from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS, OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { GROUP_ICON, OP_PRIMARY, currentOpId, localParamsFor } from '../src/renderer/filters/ui-meta.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import {
  assertLeakFree, assertWrapperBalance, assertNoHeapGrowth, fingerprintLeakProbe,
} from './_leak-helpers.mjs';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cvtColor === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

const NEW_OPS = ['prewitt_edge', 'roberts_edge', 'log_edge', 'region_growing'];

// ===================================================================
// ภาพทดสอบ (deterministic ทั้งหมด — ไม่ใช้ Math.random)
// ===================================================================
// ขอบตั้งคม 1 เส้น: x < 10 -> 0, x >= 10 -> 255  (ภาพอ้างอิงของ §3.1/§3.2/§3.3)
function stepEdge(w = 20, h = 20, at = 10) {
  const d = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d[y * w + x] = x < at ? 0 : 255;
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
function flatGray(v = 128, w = 20, h = 20) {
  return cv.matFromArray(h, w, cv.CV_8UC1, new Uint8Array(w * h).fill(v));
}
// พื้นหลัง 40 + สี่เหลี่ยม 200 ที่ x,y ∈ [5,14] = 100 พิกเซลพอดี (ภาพอ้างอิงของ §3.4)
function squareOnBg(w = 20, h = 20) {
  const d = new Uint8Array(w * h).fill(40);
  for (let y = 5; y <= 14; y++) for (let x = 5; x <= 14; x++) d[y * w + x] = 200;
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
// รูปทรงคม ๆ บนพื้นเรียบ ไม่มี noise — ใช้เทียบ "Roberts บางกว่า Prewitt" ให้ผลชัดเจน
function shapes(w, h) {
  const d = new Uint8Array(w * h).fill(40);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 40;
      if (x >= w * 0.15 && x < w * 0.45 && y >= h * 0.2 && y < h * 0.6) v = 200;
      const dx = x - w * 0.72; const dy = y - h * 0.5; const rr = Math.min(w, h) * 0.2;
      if (dx * dx + dy * dy < rr * rr) v = 140;
      d[y * w + x] = v;
    }
  }
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}
// ramp ไล่ระดับ + สี่เหลี่ยม + วงกลม + noise deterministic — ภาพแบบเดียวกับที่ใบสั่งงานใช้วัด parity
// (ramp จำเป็นสำหรับ floating vs fixed: บนภาพค่าคงที่ทั้งสองโหมดให้ผลเท่ากันจึงจับความต่างไม่ได้)
function rampScene(w, h, seed = 7, noise = 10) {
  const d = new Uint8Array(w * h);
  const rnd = lcg(seed);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 30 + Math.round((120 * x) / (w - 1));
      if (x >= w * 0.2 && x < w * 0.45 && y >= h * 0.2 && y < h * 0.6) v = 200;
      const dx = x - w * 0.72; const dy = y - h * 0.4; const rr = Math.min(w, h) * 0.15;
      if (dx * dx + dy * dy < rr * rr) v = 80;
      v += Math.round((rnd() * 2 - 1) * noise);
      d[y * w + x] = Math.max(0, Math.min(255, v));
    }
  }
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
// ภาพสีที่ทุกช่องเท่ากัน (R=G=B) — แปลงเป็นเทาแล้วได้ค่าเดิมเป๊ะทั้ง RGB2GRAY และ BGR2GRAY
function grayAsChannels(gray, channels) {
  const out = new cv.Mat();
  cv.cvtColor(gray, out, channels === 4 ? cv.COLOR_GRAY2RGBA : cv.COLOR_GRAY2RGB);
  return out;
}
function pixels(mat) { return Array.from(mat.data); }
function rowOf(mat, y) { return pixels(mat).slice(y * mat.cols, (y + 1) * mat.cols); }
function countValue(mat, v) { return pixels(mat).filter((x) => x === v).length; }
function columnsWith(mat, v) {
  const set = new Set();
  const d = mat.data;
  for (let y = 0; y < mat.rows; y++) for (let x = 0; x < mat.cols; x++) if (d[y * mat.cols + x] === v) set.add(x);
  return [...set].sort((a, b) => a - b);
}
// seed เป็น % ที่ทำให้ตกที่พิกเซล (px, py) พอดี (ใช้สูตรเดียวกับใน registry: pct/100*(cols-1))
function seedPct(px, size) { return (px / (size - 1)) * 100; }

// ===================================================================
// ipe-0 / ipe-0w — ด่าน leak หลัก (P-14 ข้อ 1 + คู่บังคับข้อ 2)
// ===================================================================
// ตาราง "เส้นทาง" ที่ ipe-0 และ ipe-0w ใช้ร่วมกัน: ทุก branch ของทั้ง 4 op + เส้นทางภาพว่าง + ภาพหลายช่อง
function edgeLeakPaths() {
  const step = stepEdge();
  const scene = rampScene(64, 48, 21);
  const rgba = grayAsChannels(scene, 4);
  const empty = new cv.Mat();
  const paths = [];
  const add = (id, params, src, tag) => paths.push({
    label: `${id} ${tag}`,
    run: (c) => { const o = OPS[id].apply(c, src, params, {}); o.delete(); },
  });
  for (const dir of ['x', 'y', 'both', undefined]) add('prewitt_edge', { dir }, step, `dir=${dir}`);
  for (const dir of ['d1', 'd2', 'both', undefined]) add('roberts_edge', { dir }, step, `dir=${dir}`);
  for (const view of ['zerocross', 'response']) add('log_edge', { view }, scene, `view=${view}`);
  add('log_edge', { view: 'zerocross' }, rgba, 'rgba input');
  for (const view of ['mask', 'overlay']) {
    for (const rangeMode of ['fixed', 'floating']) {
      add('region_growing', { view, rangeMode }, scene, `${view}/${rangeMode}`);
    }
  }
  add('region_growing', { view: 'overlay' }, rgba, 'rgba input');
  for (const id of NEW_OPS) add(id, {}, empty, 'empty 0x0');
  return { paths, cleanup: () => { step.delete(); scene.delete(); rgba.delete(); empty.delete(); } };
}

test('ipe-0. leak (address fingerprint, first test of a clean process): every branch of all four new ops (prewitt x/y/both, roberts d1/d2/both, log zerocross/response, region growing mask|overlay x fixed|floating) plus the 0x0 and multi-channel paths leaves the malloc fingerprint unchanged', () => {
  const { paths, cleanup } = edgeLeakPaths();
  try {
    fingerprintLeakProbe(cv, 'segmentation & edge ops (all paths)', () => {
      for (const p of paths) p.run(cv);
    }, { iters: 25, warmup: 10 });
  } finally { cleanup(); }
});

test('ipe-0w. leak (wrapper balance twin of ipe-0): the exact same path table creates and deletes every embind wrapper it makes (kernels, blur/lap, mask, roi, the internal gray copy)', () => {
  const { paths, cleanup } = edgeLeakPaths();
  try {
    assertWrapperBalance(cv, 'segmentation & edge ops (all paths)', (c) => {
      for (const p of paths) p.run(c);
    }, { minCreated: 40 });
  } finally { cleanup(); }
});

// ===================================================================
// ipe-L1…L6 — leak ต่อ op / ต่อเส้นทาง (P-14 ข้อ 2 + 3 ในคำสั่งเดียว)
// ===================================================================
test('ipe-L1. leak: prewitt_edge over dir x / y / both (one kernel per used direction, CV_16S buffer + convertScaleAbs result) balances every wrapper and grows the heap by 0', () => {
  const src = stepEdge(64, 48, 32);
  try {
    for (const dir of ['x', 'y', 'both']) {
      assertLeakFree(cv, `prewitt_edge dir=${dir}`, (c) => {
        const o = OPS.prewitt_edge.apply(c, src, { dir }, {});
        o.delete();
      }, { minCreated: 3, iters: 60, warmup: 20 });
    }
  } finally { src.delete(); }
});

test('ipe-L2. leak: roberts_edge over dir d1 / d2 / both balances every wrapper and grows the heap by 0', () => {
  const src = stepEdge(64, 48, 32);
  try {
    for (const dir of ['d1', 'd2', 'both']) {
      assertLeakFree(cv, `roberts_edge dir=${dir}`, (c) => {
        const o = OPS.roberts_edge.apply(c, src, { dir }, {});
        o.delete();
      }, { minCreated: 3, iters: 60, warmup: 20 });
    }
  } finally { src.delete(); }
});

test('ipe-L3. leak: log_edge over view zerocross / response, on a 1-channel and on a 4-channel input (the internal gray copy is only allocated for the multi-channel case and must be released either way)', () => {
  const gray = rampScene(64, 48, 31);
  const rgba = grayAsChannels(gray, 4);
  try {
    for (const view of ['zerocross', 'response']) {
      for (const [tag, src] of [['gray', gray], ['rgba', rgba]]) {
        assertLeakFree(cv, `log_edge view=${view} (${tag})`, (c) => {
          const o = OPS.log_edge.apply(c, src, { view }, {});
          o.delete();
        }, { minCreated: 3, iters: 50, warmup: 20 });
      }
    }
  } finally { gray.delete(); rgba.delete(); }
});

test('ipe-L4. leak: region_growing over view mask|overlay x rangeMode fixed|floating, on 1-channel and 4-channel inputs (Mat.zeros mask + mask.roi() wrapper + optional gray copy all released)', () => {
  const gray = rampScene(64, 48, 41);
  const rgba = grayAsChannels(gray, 4);
  try {
    for (const view of ['mask', 'overlay']) {
      for (const rangeMode of ['fixed', 'floating']) {
        for (const [tag, src] of [['gray', gray], ['rgba', rgba]]) {
          assertLeakFree(cv, `region_growing ${view}/${rangeMode} (${tag})`, (c) => {
            const o = OPS.region_growing.apply(c, src, { view, rangeMode }, {});
            o.delete();
          }, { minCreated: 3, iters: 50, warmup: 20 });
        }
      }
    }
  } finally { gray.delete(); rgba.delete(); }
});

test('ipe-L5. leak + proof the 0x0 guard is necessary: calling cv.filter2D / cv.GaussianBlur / cv.floodFill directly on an empty Mat really does throw a C++ exception (which leaks ~760 B inside opencv.js every time), while all four ops guard before that point -- no throw, no leak, 0 live wrappers', () => {
  const empty = new cv.Mat();
  try {
    // ถ้าวันหนึ่งถอด guard ออก เทสต์ ipe-23 จะล้มทันที — สามข้อนี้พิสูจน์ว่า guard ไม่ใช่โค้ดส่วนเกิน
    const k = cv.matFromArray(3, 3, cv.CV_32FC1, [-1, 0, 1, -1, 0, 1, -1, 0, 1]);
    const dst = new cv.Mat();
    assert.throws(() => cv.filter2D(empty, dst, cv.CV_16S, k, new cv.Point(-1, -1), 0, cv.BORDER_DEFAULT),
      'cv.filter2D on a 0x0 Mat must throw (that is why prewitt/roberts guard first)');
    assert.throws(() => cv.GaussianBlur(empty, dst, new cv.Size(0, 0), 2, 2, cv.BORDER_DEFAULT),
      'cv.GaussianBlur on a 0x0 Mat must throw (that is why log_edge guards first)');
    const mask = new cv.Mat.zeros(2, 2, cv.CV_8UC1);
    assert.throws(() => cv.floodFill(empty, mask, new cv.Point(0, 0), new cv.Scalar(0), new cv.Rect(),
      new cv.Scalar(0, 0, 0, 0), new cv.Scalar(0, 0, 0, 0), 4 | (255 << 8) | cv.FLOODFILL_MASK_ONLY),
    'cv.floodFill on a 0x0 Mat must throw (that is why region_growing guards first)');
    k.delete(); dst.delete(); mask.delete();

    for (const id of NEW_OPS) {
      assertLeakFree(cv, `${id} on an empty 0x0 Mat`, (c) => {
        const o = OPS[id].apply(c, empty, {}, {});
        o.delete();
      }, { minCreated: 1, iters: 60, warmup: 20 });
    }
  } finally { empty.delete(); }
});

test('ipe-L6. leak: region_growing with the seed pinned at both extremes (0% and 100% in both axes) for 100 rounds grows the heap by 0 -- if the clamp ever let a seed out of the frame, floodFill would throw a C++ exception and leak ~760 B per round', () => {
  const src = rampScene(48, 36, 51);
  try {
    for (const [sx, sy] of [[0, 0], [100, 100], [0, 100], [100, 0]]) {
      // assertNoHeapGrowth เรียก run() โดยไม่ส่ง cv (ต่างจาก assertLeakFree) — ใช้ cv จาก closure
      assertNoHeapGrowth(cv, `region_growing seed (${sx}%, ${sy}%)`, () => {
        const o = OPS.region_growing.apply(cv, src, { seedX: sx, seedY: sy }, {});
        o.delete();
      }, { iters: 100, warmup: 30 });
    }
  } finally { src.delete(); }
});

// ===================================================================
// ipe-1…4 — prewitt_edge (เกณฑ์ผ่าน §3.1)
// ===================================================================
test('ipe-1. prewitt_edge dir=x on a 20x20 vertical step edge: row y=10 is exactly [0 x9, 255, 255, 0 x9] -- and on the mirrored (bright -> dark) edge the answer is identical, which only holds because the intermediate buffer is signed (CV_16S + convertScaleAbs); CV_8U would clip the negative gradient to 0 and the falling edge would disappear', () => {
  const expected = [...Array(9).fill(0), 255, 255, ...Array(9).fill(0)];
  const rising = stepEdge();
  try {
    const out = OPS.prewitt_edge.apply(cv, rising, { dir: 'x' }, {});
    try {
      assert.deepEqual(rowOf(out, 10), expected);
      assert.equal(out.channels(), 1);
      assert.equal(out.rows, 20);
      assert.equal(out.cols, 20);
    } finally { out.delete(); }
  } finally { rising.delete(); }

  // ขอบขาลง (สว่าง -> มืด): ความชันเป็นลบ ถ้า buffer กลางเป็น CV_8U ค่าลบจะถูกตัดเป็น 0 แล้วแถวนี้จะกลายเป็นดำหมด
  const falling = new cv.Mat();
  const inv = stepEdge();
  try {
    cv.bitwise_not(inv, falling);
    for (const id of ['prewitt_edge', 'roberts_edge']) {
      const dir = id === 'prewitt_edge' ? 'x' : 'd1';
      const a = OPS[id].apply(cv, inv, { dir }, {});
      const b = OPS[id].apply(cv, falling, { dir }, {});
      try {
        assert.ok(cv.countNonZero(b) > 0, `${id}: a falling edge must be detected too (a signed intermediate buffer is required)`);
        assert.ok(Buffer.compare(Buffer.from(a.data), Buffer.from(b.data)) === 0,
          `${id}: a rising and a falling edge of the same magnitude must give the same picture`);
      } finally { a.delete(); b.delete(); }
    }
  } finally { falling.delete(); inv.delete(); }
});

test('ipe-2. prewitt_edge dir=y on the same vertical step edge: every pixel is 0 (there is no horizontal edge at all) -- proves kx and ky are not swapped', () => {
  const src = stepEdge();
  try {
    const out = OPS.prewitt_edge.apply(cv, src, { dir: 'y' }, {});
    try {
      assert.equal(cv.countNonZero(out), 0);
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipe-3. prewitt_edge dir=both: row y=10 is exactly [0 x9, 128, 128, 0 x9] -- the 0.5/0.5 weights of addWeighted are what turn 255 into 128 (weights of 1/1 would saturate back to 255); an unknown dir value falls back to "both"', () => {
  const src = stepEdge();
  try {
    const expected = [...Array(9).fill(0), 128, 128, ...Array(9).fill(0)];
    for (const dir of ['both', 'nonsense', undefined]) {
      const out = OPS.prewitt_edge.apply(cv, src, { dir }, {});
      try {
        assert.deepEqual(rowOf(out, 10), expected, `dir=${dir}`);
      } finally { out.delete(); }
    }
  } finally { src.delete(); }
});

test('ipe-4. prewitt_edge on a flat 128 image: maxVal is exactly 0 in all three directions (a constant image has no gradient anywhere)', () => {
  const src = flatGray(128);
  try {
    for (const dir of ['x', 'y', 'both']) {
      const out = OPS.prewitt_edge.apply(cv, src, { dir }, {});
      try {
        assert.equal(cv.minMaxLoc(out).maxVal, 0, `dir=${dir}`);
      } finally { out.delete(); }
    }
  } finally { src.delete(); }
});

// ===================================================================
// ipe-5…8 — roberts_edge (เกณฑ์ผ่าน §3.2)
// ===================================================================
test('ipe-5. roberts_edge dir=d1 on the 20x20 step edge: only x=10 is 255 on row y=10 (a single column, unlike Prewitt which lights up two) -- this is the anchor (-1,-1) result; an anchor of (0,0) shifts the response by one pixel', () => {
  const src = stepEdge();
  try {
    const out = OPS.roberts_edge.apply(cv, src, { dir: 'd1' }, {});
    try {
      const expected = [...Array(10).fill(0), 255, ...Array(9).fill(0)];
      assert.deepEqual(rowOf(out, 10), expected);
      assert.deepEqual(columnsWith(out, 255), [10], 'the whole image must only light up column 10');
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipe-6. roberts_edge dir=both (and an unknown dir): row y=10 has 255 at x=10 only', () => {
  const src = stepEdge();
  try {
    const expected = [...Array(10).fill(0), 255, ...Array(9).fill(0)];
    for (const dir of ['both', 'nonsense', undefined]) {
      const out = OPS.roberts_edge.apply(cv, src, { dir }, {});
      try {
        assert.deepEqual(rowOf(out, 10), expected, `dir=${dir}`);
        assert.deepEqual(columnsWith(out, 255), [10], `dir=${dir}: only column 10`);
      } finally { out.delete(); }
    }
  } finally { src.delete(); }

  // ขอบทแยง 45°: kernel ↘ ไม่เห็นขอบเลย (0) ส่วน ↗ เห็นเต็ม (255) -> "รวมสองแนว" ต้องได้ 128 พอดี
  // (นี่คือจุดที่พิสูจน์น้ำหนัก 0.5/0.5 ของ Roberts — บนขอบตั้ง/ขอบนอน ทั้งสองแนวให้ค่าเท่ากันจึงจับไม่ได้)
  const diag = (() => {
    const d = new Uint8Array(400);
    for (let y = 0; y < 20; y++) for (let x = 0; x < 20; x++) d[y * 20 + x] = x > y ? 255 : 0;
    return cv.matFromArray(20, 20, cv.CV_8UC1, d);
  })();
  try {
    const d1 = OPS.roberts_edge.apply(cv, diag, { dir: 'd1' }, {});
    const d2 = OPS.roberts_edge.apply(cv, diag, { dir: 'd2' }, {});
    const both = OPS.roberts_edge.apply(cv, diag, { dir: 'both' }, {});
    try {
      const A = d1.data; const B = d2.data; const C = both.data;
      let checked = 0;
      for (let i = 0; i < A.length; i++) {
        if (A[i] === 0 && B[i] === 255) {
          assert.equal(C[i], 128, `pixel ${i}: d1=0 and d2=255 must average to 128, not ${C[i]}`);
          checked++;
        }
      }
      assert.ok(checked > 5, `expected several pixels where only one diagonal responds, found ${checked}`);
    } finally { d1.delete(); d2.delete(); both.delete(); }
  } finally { diag.delete(); }
});

test('ipe-7. roberts_edge on a flat 128 image: maxVal is exactly 0 in all three directions', () => {
  const src = flatGray(128);
  try {
    for (const dir of ['d1', 'd2', 'both']) {
      const out = OPS.roberts_edge.apply(cv, src, { dir }, {});
      try {
        assert.equal(cv.minMaxLoc(out).maxVal, 0, `dir=${dir}`);
      } finally { out.delete(); }
    }
  } finally { src.delete(); }
});

test('ipe-8. roberts_edge vs prewitt_edge on the same 320x240 scene: Roberts marks strictly fewer non-zero pixels (a 2x2 kernel gives thinner edges than a 3x3 one) -- the teaching point of having both in the same box', () => {
  const src = shapes(320, 240);
  try {
    const pre = OPS.prewitt_edge.apply(cv, src, { dir: 'both' }, {});
    const rob = OPS.roberts_edge.apply(cv, src, { dir: 'both' }, {});
    try {
      const nPre = cv.countNonZero(pre);
      const nRob = cv.countNonZero(rob);
      assert.ok(nPre > 0 && nRob > 0, `both must find edges (prewitt ${nPre}, roberts ${nRob})`);
      assert.ok(nRob < nPre, `roberts (${nRob}) must be thinner than prewitt (${nPre})`);
    } finally { pre.delete(); rob.delete(); }
  } finally { src.delete(); }
});

// ===================================================================
// ipe-9…14 — log_edge (เกณฑ์ผ่าน §3.3)
// ===================================================================
test('ipe-9. log_edge sigma=2 zeroThresh=8 view=zerocross on the 20x20 step edge: exactly 19 pixels are 255 and all of them sit in column x=9 (rows 0..18 -- the last row and last column are 0 by design), and ctx.info reports that same count', () => {
  const src = stepEdge();
  try {
    const ctx = {};
    const out = OPS.log_edge.apply(cv, src, { sigma: 2, zeroThresh: 8, view: 'zerocross' }, ctx);
    try {
      assert.equal(countValue(out, 255), 19);
      assert.deepEqual(columnsWith(out, 255), [9]);
      assert.equal(rowOf(out, 19).filter((v) => v !== 0).length, 0, 'the last row is 0 by design');
      assert.equal(pixels(out).filter((v, i) => i % 20 === 19 && v !== 0).length, 0, 'the last column is 0 by design');
      assert.match(ctx.info, /พบขอบ \(จุดตัดศูนย์\) 19 พิกเซล · sigma 2 · เกณฑ์ 8/);
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipe-9b. log_edge on a 320x240 scene: the last row and the last column are always 0 (there is no right/bottom neighbour to compare with) -- the loop must stop at w-1 / h-1, and the sign test must be strict (> 0, not >= 0)', () => {
  const src = rampScene(320, 240, 9);
  try {
    const out = OPS.log_edge.apply(cv, src, { sigma: 2, zeroThresh: 8 }, {});
    try {
      const d = out.data;
      let lastCol = 0; let lastRow = 0;
      for (let y = 0; y < out.rows; y++) if (d[y * out.cols + (out.cols - 1)] !== 0) lastCol++;
      for (let x = 0; x < out.cols; x++) if (d[(out.rows - 1) * out.cols + x] !== 0) lastRow++;
      assert.equal(lastCol, 0, 'the last column must stay 0');
      assert.equal(lastRow, 0, 'the last row must stay 0');
      assert.ok(cv.countNonZero(out) > 100, 'the scene must actually produce edges (otherwise the check above is vacuous)');
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipe-10. log_edge zeroThresh boundaries on the step edge (the exact |c-r| values at the sign changes are 4.0 and 48.0): thr 0 and 3.9 give 38 pixels in columns {2, 9}, thr 4.0 gives 19 in column {9}, thr 47.9 gives 19 and thr 48.0 gives 0 -- the comparison must be strictly greater-than, and 0 must be usable as a real value (not swallowed as "falsy")', () => {
  const src = stepEdge();
  try {
    const run = (zeroThresh) => {
      const out = OPS.log_edge.apply(cv, src, { sigma: 2, zeroThresh }, {});
      const result = { n: countValue(out, 255), cols: columnsWith(out, 255) };
      out.delete();
      return result;
    };
    assert.deepEqual(run(0), { n: 38, cols: [2, 9] }, 'zeroThresh 0 must keep every zero-crossing');
    assert.deepEqual(run(3.9), { n: 38, cols: [2, 9] });
    assert.deepEqual(run(4), { n: 19, cols: [9] }, 'a difference of exactly 4.0 must NOT pass a threshold of 4.0');
    assert.deepEqual(run(47.9), { n: 19, cols: [9] });
    assert.deepEqual(run(48), { n: 0, cols: [] }, 'a difference of exactly 48.0 must NOT pass a threshold of 48.0');
  } finally { src.delete(); }
});

test('ipe-11. log_edge sigma=1 zeroThresh=8: still exactly 19 pixels, still in column x=9 -- changing sigma does not move where the edge is', () => {
  const src = stepEdge();
  try {
    const out = OPS.log_edge.apply(cv, src, { sigma: 1, zeroThresh: 8 }, {});
    try {
      assert.equal(countValue(out, 255), 19);
      assert.deepEqual(columnsWith(out, 255), [9]);
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipe-12. log_edge view=zerocross on a flat 128 image: 0 edge pixels and ctx.info says 0', () => {
  const src = flatGray(128);
  try {
    const ctx = {};
    const out = OPS.log_edge.apply(cv, src, { view: 'zerocross' }, ctx);
    try {
      assert.equal(cv.countNonZero(out), 0);
      assert.match(ctx.info, /จุดตัดศูนย์\) 0 พิกเซล/);
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipe-13. log_edge view=response on a flat image: every pixel is exactly 128 (128 = "LoG value zero"; a beta of 0 would make the whole image black and the mexican-hat lesson meaningless), and the alpha guard keeps it from dividing by zero', () => {
  for (const level of [0, 128, 255]) {
    const src = flatGray(level);
    try {
      const ctx = {};
      const out = OPS.log_edge.apply(cv, src, { view: 'response' }, ctx);
      try {
        const mm = cv.minMaxLoc(out);
        assert.equal(mm.minVal, 128, `flat ${level}: min`);
        assert.equal(mm.maxVal, 128, `flat ${level}: max`);
        assert.equal(out.channels(), 1);
        assert.match(ctx.info, /128 = ศูนย์/);
      } finally { out.delete(); }
    } finally { src.delete(); }
  }
  // ...และบนภาพจริง โหมด response ต้องมีทั้งค่าสว่างกว่าและมืดกว่า 128 (หมวกเม็กซิกัน 2 ข้าง)
  const scene = stepEdge();
  try {
    const out = OPS.log_edge.apply(cv, scene, { view: 'response' }, {});
    try {
      const mm = cv.minMaxLoc(out);
      assert.ok(mm.minVal < 128 && mm.maxVal > 128, `expected values on both sides of 128, got ${mm.minVal}..${mm.maxVal}`);
    } finally { out.delete(); }
  } finally { scene.delete(); }
});

// ===================================================================
// ipe-15…21 — region_growing (เกณฑ์ผ่าน §3.4)
// ===================================================================
test('ipe-15. region_growing seeded in the middle of the 10x10 square (tolerance 20): exactly 100 selected pixels, the mask is 0/255 (max must be 255 -- without the (255 << 8) flag OpenCV writes 1 and the picture would look black), and ctx.info carries the same 100', () => {
  const src = squareOnBg();
  try {
    const ctx = {};
    const out = OPS.region_growing.apply(cv, src, {
      seedX: seedPct(10, 20), seedY: seedPct(10, 20), tolerance: 20,
    }, ctx);
    try {
      assert.equal(cv.countNonZero(out), 100);
      assert.equal(cv.minMaxLoc(out).maxVal, 255, 'the mask must be 255, not 1');
      assert.equal(cv.minMaxLoc(out).minVal, 0);
      assert.equal(out.channels(), 1);
      assert.equal(countValue(out, 255), 100, 'exactly 100 pixels must carry the value 255');
      assert.match(ctx.info, /เลือกได้ 100 พิกเซล \(25\.0% ของภาพ\) · จุดเริ่ม \(10, 10\) · ความต่างไม่เกิน 20/);
    } finally { out.delete(); }
  } finally { src.delete(); }

  // 4-connectivity เป็นการตัดสินใจที่ตั้งใจ (ไม่เปิดเป็นพารามิเตอร์) — สองบล็อกที่แตะกันแค่มุมทแยงต้อง
  // ไม่ลามถึงกัน ถ้าวันหนึ่งกลายเป็น 8 จะได้ 98 แทน 49
  const corners = (() => {
    const d = new Uint8Array(400);
    for (let y = 2; y <= 8; y++) for (let x = 2; x <= 8; x++) d[y * 20 + x] = 200;
    for (let y = 9; y <= 15; y++) for (let x = 9; x <= 15; x++) d[y * 20 + x] = 200;
    return cv.matFromArray(20, 20, cv.CV_8UC1, d);
  })();
  try {
    const out = OPS.region_growing.apply(cv, corners, {
      seedX: seedPct(5, 20), seedY: seedPct(5, 20), tolerance: 20,
    }, {});
    try {
      assert.equal(cv.countNonZero(out), 49, 'with 4-connectivity the flood must stop at the diagonal corner contact');
    } finally { out.delete(); }
  } finally { corners.delete(); }
});

test('ipe-16. region_growing with tolerance 0 on the same seed: still exactly 100 (the square is one single value) -- 0 must behave as a real tolerance, not fall back to the default of 12', () => {
  const src = squareOnBg();
  try {
    const ctx = {};
    const out = OPS.region_growing.apply(cv, src, {
      seedX: seedPct(10, 20), seedY: seedPct(10, 20), tolerance: 0,
    }, ctx);
    try {
      assert.equal(cv.countNonZero(out), 100);
      assert.match(ctx.info, /ความต่างไม่เกิน 0$/);
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipe-17. region_growing seeded on the background corner (0%, 0%) with tolerance 20: exactly 300 pixels (400 - the 100 of the square) -- the region stops at the object boundary', () => {
  const src = squareOnBg();
  try {
    const ctx = {};
    const out = OPS.region_growing.apply(cv, src, { seedX: 0, seedY: 0, tolerance: 20 }, ctx);
    try {
      assert.equal(cv.countNonZero(out), 300);
      assert.match(ctx.info, /จุดเริ่ม \(0, 0\)/);
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipe-18. region_growing tolerance clamp: 100 (the UI maximum) and 500 (only reachable by hand-editing a project file) both give exactly 100 pixels, not 400 -- and the test is not vacuous: calling cv.floodFill directly with the unclamped 160 really does flood all 400 pixels (the background is 160 grey levels away)', () => {
  const src = squareOnBg();
  try {
    for (const tolerance of [100, 500, '500', 1e9]) {
      const ctx = {};
      const out = OPS.region_growing.apply(cv, src, {
        seedX: seedPct(10, 20), seedY: seedPct(10, 20), tolerance,
      }, ctx);
      try {
        assert.equal(cv.countNonZero(out), 100, `tolerance ${tolerance} must stay inside the square`);
        assert.match(ctx.info, /ความต่างไม่เกิน 100$/, `tolerance ${tolerance} must be reported as clamped to 100`);
      } finally { out.delete(); }
    }
    // ค่าที่ UI ผลิตไม่ได้ (160) ทำให้ลามทั้งภาพจริง — พิสูจน์ว่าการ clamp ที่ 100 มีผลจริง ไม่ใช่ no-op
    const mask = new cv.Mat.zeros(22, 22, cv.CV_8UC1);
    try {
      const flags = 4 | (255 << 8) | cv.FLOODFILL_MASK_ONLY | cv.FLOODFILL_FIXED_RANGE;
      const filled = cv.floodFill(src, mask, new cv.Point(10, 10), new cv.Scalar(0), new cv.Rect(),
        new cv.Scalar(160, 160, 160, 160), new cv.Scalar(160, 160, 160, 160), flags);
      assert.equal(filled, 400, 'with tolerance 160 the flood really would take the whole image');
    } finally { mask.delete(); }
  } finally { src.delete(); }
});

test('ipe-18b. region_growing seed mapping uses (size - 1), not size: on a 40x30 image 0% -> (0,0), 100% -> (39,29) and 95% -> (37,28) exactly -- with "size" instead of "size - 1" the 95% seed would land on (38,29) and 100% would fall outside the frame', () => {
  const src = rampScene(40, 30, 61);
  try {
    const seedAt = (sx, sy) => {
      const ctx = {};
      const out = OPS.region_growing.apply(cv, src, { seedX: sx, seedY: sy, tolerance: 5 }, ctx);
      out.delete();
      const m = ctx.info.match(/จุดเริ่ม \((\d+), (\d+)\)/);
      return [Number(m[1]), Number(m[2])];
    };
    assert.deepEqual(seedAt(0, 0), [0, 0]);
    assert.deepEqual(seedAt(100, 100), [39, 29]);
    assert.deepEqual(seedAt(95, 95), [37, 28]);
    assert.deepEqual(seedAt(50, 50), [20, 15]);
  } finally { src.delete(); }
});

test('ipe-19. region_growing view=overlay: the output has 3 channels, the selected area is painted pure green (0,255,0) in RGB order, the seed marker is drawn in red, and the picture outside the selection still shows the original grey', () => {
  const src = squareOnBg();
  try {
    const ctx = {};
    const out = OPS.region_growing.apply(cv, src, {
      seedX: seedPct(10, 20), seedY: seedPct(10, 20), tolerance: 20, view: 'overlay',
    }, ctx);
    try {
      assert.equal(out.channels(), 3);
      assert.equal(out.rows, 20);
      assert.equal(out.cols, 20);
      const d = out.data;
      const at = (x, y) => [d[(y * 20 + x) * 3], d[(y * 20 + x) * 3 + 1], d[(y * 20 + x) * 3 + 2]];
      assert.deepEqual(at(10, 10), [0, 255, 0], 'the seed pixel itself is inside the selection -> green');
      assert.deepEqual(at(6, 6), [0, 255, 0], 'a corner of the square is selected -> green');
      assert.deepEqual(at(0, 0), [40, 40, 40], 'background outside the selection keeps the original grey');
      // จุดเริ่มถูกวาดเป็นวงกลมสีแดง (RGB) — หาพิกเซลแดงอย่างน้อย 1 จุดรอบ ๆ จุดเริ่ม
      let red = 0;
      for (let y = 0; y < 20; y++) for (let x = 0; x < 20; x++) {
        const [r, g, b] = at(x, y);
        if (r === 255 && g === 0 && b === 0) red++;
      }
      assert.ok(red > 0, 'the seed marker circle must be visible in red');
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipe-20. region_growing rangeMode on a scene with a brightness ramp: "floating" (compare with the neighbour) selects strictly more pixels than "fixed" (compare with the seed) -- on a flat object both are identical, which is why this needs a ramp', () => {
  const src = rampScene(32, 24, 11);
  try {
    const run = (rangeMode) => {
      const ctx = {};
      const out = OPS.region_growing.apply(cv, src, { seedX: 50, seedY: 50, tolerance: 10, rangeMode }, ctx);
      const n = cv.countNonZero(out);
      // เลขที่รายงานใน ctx.info (= ค่าที่ floodFill คืนมา) ต้องเท่ากับจำนวนพิกเซลในหน้ากากเป๊ะทุกโหมด —
      // filter-smoke ใช้เลขนี้เป็นด่าน POSITIVE_NOTE จึงต้องเชื่อถือได้ ไม่ใช่แค่ "มีเลขอะไรสักตัว"
      assert.equal(Number(ctx.info.match(/เลือกได้ (\d+) พิกเซล/)[1]), n,
        `${rangeMode}: the count reported in ctx.info must equal the mask`);
      out.delete();
      return n;
    };
    const fixed = run('fixed');
    const floating = run('floating');
    assert.ok(floating > fixed, `floating (${floating}) must reach further than fixed (${fixed})`);
    // ...และค่าที่ไม่รู้จัก/ไม่ได้ตั้ง ต้องตกมาที่ fixed (ค่าปลอดภัยที่คาดเดาได้)
    assert.equal(run(undefined), fixed, 'an unset rangeMode must behave as "fixed"');
    assert.equal(run('nonsense'), fixed, 'an unknown rangeMode must behave as "fixed"');
    // บนภาพค่าคงที่ ทั้งสองโหมดต้องเท่ากัน (เหตุผลที่เทสต์นี้ต้องใช้ภาพ ramp)
    const flat = squareOnBg();
    try {
      const p = { seedX: seedPct(10, 20), seedY: seedPct(10, 20), tolerance: 20 };
      const a = OPS.region_growing.apply(cv, flat, { ...p, rangeMode: 'fixed' }, {});
      const b = OPS.region_growing.apply(cv, flat, { ...p, rangeMode: 'floating' }, {});
      assert.equal(cv.countNonZero(a), cv.countNonZero(b), 'on a flat object both modes agree');
      a.delete(); b.delete();
    } finally { flat.delete(); }
  } finally { src.delete(); }
});

test('ipe-21. region_growing must never write into its input image: after every view/rangeMode combination the source bytes are identical (without FLOODFILL_MASK_ONLY, floodFill paints the source in place and would corrupt the Mat that runPipeline shares with the other boxes in the graph)', () => {
  const src = squareOnBg();
  const before = Buffer.from(src.data);
  try {
    for (const view of ['mask', 'overlay']) {
      for (const rangeMode of ['fixed', 'floating']) {
        for (const tolerance of [0, 20, 100]) {
          const out = OPS.region_growing.apply(cv, src, {
            seedX: seedPct(10, 10), seedY: seedPct(10, 10), tolerance, view, rangeMode,
          }, {});
          out.delete();
          assert.ok(Buffer.compare(before, Buffer.from(src.data)) === 0,
            `${view}/${rangeMode}/tol ${tolerance}: the source image was modified in place`);
        }
      }
    }
  } finally { src.delete(); }
  // ...และ op อื่นอีก 3 ตัวก็ต้องไม่แตะภาพต้นทางเช่นกัน
  const scene = rampScene(48, 36, 71);
  const snapshot = Buffer.from(scene.data);
  try {
    for (const id of ['prewitt_edge', 'roberts_edge', 'log_edge']) {
      const out = OPS[id].apply(cv, scene, {}, {});
      out.delete();
      assert.ok(Buffer.compare(snapshot, Buffer.from(scene.data)) === 0, `${id} modified its input`);
    }
  } finally { scene.delete(); }
});

// ===================================================================
// ipe-22…26 — เคสขอบ / พารามิเตอร์
// ===================================================================
test('ipe-22. tiny inputs (1x1, 2x2, 1x50, 50x1, 3x3): none of the four ops throws, and every output keeps the input size (log_edge simply produces no zero-crossings when there is no right/bottom neighbour)', () => {
  for (const [w, h] of [[1, 1], [2, 2], [1, 50], [50, 1], [3, 3]]) {
    const d = new Uint8Array(w * h);
    for (let i = 0; i < d.length; i++) d[i] = (i * 37) % 256;
    const src = cv.matFromArray(h, w, cv.CV_8UC1, d);
    try {
      for (const id of NEW_OPS) {
        const ctx = {};
        const out = OPS[id].apply(cv, src, {}, ctx);
        try {
          assert.equal(out.rows, h, `${id} @${w}x${h}: rows`);
          assert.equal(out.cols, w, `${id} @${w}x${h}: cols`);
        } finally { out.delete(); }
      }
      // region_growing โหมด overlay ก็ต้องรอดบนภาพเล็กกว่ารัศมีวงกลมจุดเริ่ม (cv.circle ตัดขอบให้เอง)
      const ov = OPS.region_growing.apply(cv, src, { view: 'overlay' }, {});
      try {
        assert.equal(ov.channels(), 3, `overlay @${w}x${h}`);
        assert.equal(ov.rows, h);
        assert.equal(ov.cols, w);
      } finally { ov.delete(); }
    } finally { src.delete(); }
  }
});

test('ipe-23. empty 0x0 input: all four ops return a Mat instead of throwing (a C++ exception would leak ~760 B inside opencv.js every frame), the output stays 0x0 and ctx.info says so', () => {
  const empty = new cv.Mat();
  try {
    for (const id of NEW_OPS) {
      const ctx = {};
      const out = OPS[id].apply(cv, empty, {}, ctx);
      try {
        assert.equal(out.rows, 0, `${id}: rows`);
        assert.equal(out.cols, 0, `${id}: cols`);
        assert.equal(ctx.info, 'ภาพว่าง', `${id}: info`);
      } finally { out.delete(); }
    }
  } finally { empty.delete(); }
});

test('ipe-24. 3-channel and 4-channel inputs handed straight to log_edge / region_growing (the direct unit-call path, no runPipeline auto-gray): no throw, and the result is byte-for-byte the same as converting to grey first -- without that guard log_edge would read a 3-channel float buffer as if it were 1-channel and be silently wrong, and region_growing would throw on RGBA', () => {
  const gray = rampScene(48, 36, 81);
  try {
    const cases = [
      ['log_edge', { view: 'zerocross' }],
      ['log_edge', { view: 'response' }],
      ['region_growing', { view: 'mask' }],
      ['region_growing', { view: 'overlay' }],
    ];
    for (const [id, params] of cases) {
      const expected = OPS[id].apply(cv, gray, params, {});
      try {
        for (const channels of [3, 4]) {
          const colour = grayAsChannels(gray, channels);
          try {
            const got = OPS[id].apply(cv, colour, params, {});
            try {
              assert.equal(got.rows, expected.rows, `${id} ${JSON.stringify(params)} @${channels}ch: rows`);
              assert.equal(got.cols, expected.cols);
              assert.equal(got.channels(), expected.channels(), `${id} @${channels}ch: output channels`);
              assert.ok(Buffer.compare(Buffer.from(got.data), Buffer.from(expected.data)) === 0,
                `${id} ${JSON.stringify(params)} @${channels}ch: pixels differ from the grey path`);
            } finally { got.delete(); }
          } finally { colour.delete(); }
        }
      } finally { expected.delete(); }
    }
  } finally { gray.delete(); }
});

test('ipe-25. params arriving as strings (a hand-edited or imported project file): "2"/"0" behave exactly like 2/0 for every new op, and "0" never falls back to the default', () => {
  const step = stepEdge();
  const square = squareOnBg();
  try {
    const same = (id, src, a, b, label) => {
      const x = OPS[id].apply(cv, src, a, {});
      const y = OPS[id].apply(cv, src, b, {});
      try {
        assert.ok(Buffer.compare(Buffer.from(x.data), Buffer.from(y.data)) === 0, label);
      } finally { x.delete(); y.delete(); }
    };
    same('log_edge', step, { sigma: 2, zeroThresh: 0 }, { sigma: '2', zeroThresh: '0' }, 'log_edge string params');
    same('region_growing', square,
      { seedX: 0, seedY: 0, tolerance: 0 }, { seedX: '0', seedY: '0', tolerance: '0' }, 'region_growing string params');
    // "0" ต้องไม่กลายเป็นค่า default: zeroThresh "0" ให้ 38 จุด (default 8 ให้ 19)
    const zeroThr = OPS.log_edge.apply(cv, step, { sigma: 2, zeroThresh: '0' }, {});
    try {
      assert.equal(countValue(zeroThr, 255), 38, 'zeroThresh "0" must mean 0, not the default 8');
    } finally { zeroThr.delete(); }
    // seedX "0" ต้องเป็นมุมซ้าย (300 พิกเซลของพื้นหลัง) ไม่ใช่ default 50% (กลางสี่เหลี่ยม = 100)
    const ctx = {};
    const corner = OPS.region_growing.apply(cv, square, { seedX: '0', seedY: '0', tolerance: 20 }, ctx);
    try {
      assert.equal(cv.countNonZero(corner), 300, 'seedX "0" must mean 0%, not the default 50%');
      assert.match(ctx.info, /จุดเริ่ม \(0, 0\)/);
    } finally { corner.delete(); }
  } finally { step.delete(); square.delete(); }
});

test('ipe-26. out-of-range params (sigma 99, sigma 0, sigma -1, zeroThresh -5, tolerance -5, seedX 500, seedY -200, NaN, undefined): every op clamps into the UI range, never throws, and never returns a broken image', () => {
  const scene = rampScene(48, 36, 91);
  try {
    const wild = [
      { sigma: 99, zeroThresh: -5 },
      { sigma: 0, zeroThresh: 1e9 },
      { sigma: -1, zeroThresh: NaN },
      { sigma: 'abc', zeroThresh: undefined },
    ];
    for (const params of wild) {
      const out = OPS.log_edge.apply(cv, scene, params, {});
      try {
        assert.equal(out.rows, 36);
        assert.equal(out.cols, 48);
        assert.equal(out.channels(), 1);
      } finally { out.delete(); }
    }
    for (const params of [
      { seedX: 500, seedY: -200, tolerance: -5 },
      { seedX: -1, seedY: 1e9, tolerance: 1e9 },
      { seedX: NaN, seedY: 'abc', tolerance: undefined },
    ]) {
      const ctx = {};
      const out = OPS.region_growing.apply(cv, scene, params, ctx);
      try {
        assert.equal(out.rows, 36);
        assert.equal(out.cols, 48);
        const m = ctx.info.match(/จุดเริ่ม \((\d+), (\d+)\)/);
        assert.ok(m, `ctx.info must report the seed: ${ctx.info}`);
        assert.ok(Number(m[1]) >= 0 && Number(m[1]) <= 47, `seed x out of frame: ${m[1]}`);
        assert.ok(Number(m[2]) >= 0 && Number(m[2]) <= 35, `seed y out of frame: ${m[2]}`);
      } finally { out.delete(); }
    }
    // sigma ที่ต่ำกว่า/สูงกว่าช่วง ต้องให้ผลเท่ากับค่าที่ขอบช่วงพอดี (พิสูจน์ว่า clamp ทำงาน ไม่ใช่แค่ไม่ throw)
    const a = OPS.log_edge.apply(cv, scene, { sigma: -1 }, {});
    const b = OPS.log_edge.apply(cv, scene, { sigma: 0.5 }, {});
    const c = OPS.log_edge.apply(cv, scene, { sigma: 99 }, {});
    const d = OPS.log_edge.apply(cv, scene, { sigma: 5 }, {});
    try {
      assert.ok(Buffer.compare(Buffer.from(a.data), Buffer.from(b.data)) === 0, 'sigma -1 must clamp to the minimum 0.5');
      assert.ok(Buffer.compare(Buffer.from(c.data), Buffer.from(d.data)) === 0, 'sigma 99 must clamp to the maximum 5');
      assert.ok(Buffer.compare(Buffer.from(b.data), Buffer.from(d.data)) !== 0, 'sigma 0.5 and 5 must actually differ (otherwise the check above is vacuous)');
    } finally { a.delete(); b.delete(); c.delete(); d.delete(); }
  } finally { scene.delete(); }
});

// ===================================================================
// ipe-27 — static read: ช่วงที่โค้ด clamp ต้องตรงกับ min/max/default ของ UI เป๊ะ
// ===================================================================
test('ipe-27. (static read) every numeric param of the four new ops is clamped in registry.js to EXACTLY the min/max/default the UI offers -- a hand-edited project file can never push a value the UI cannot produce into OpenCV, and the two can never drift apart silently', () => {
  const source = fs.readFileSync(new URL('../src/renderer/filters/registry.js', import.meta.url), 'utf8');
  // [opId, paramKey, ตัวช่วยที่ใช้] — intIn = ปัดเป็นจำนวนเต็มด้วย, numIn = clamp เฉย ๆ (ทศนิยมใช้ได้)
  const CLAMPED = [
    ['log_edge', 'sigma', 'numIn'],
    ['log_edge', 'zeroThresh', 'numIn'],
    ['region_growing', 'seedX', 'numIn'],
    ['region_growing', 'seedY', 'numIn'],
    ['region_growing', 'tolerance', 'intIn'],
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
  // ...และห้ามมี param ตัวเลขตัวไหนของ 4 op นี้หลุดออกไปจากรายการข้างบน (เพิ่ม param ใหม่ = ต้องมี clamp)
  for (const id of NEW_OPS) {
    for (const p of OPS[id].params) {
      if (p.type !== 'number') continue;
      assert.ok(CLAMPED.some(([o, k]) => o === id && k === p.key),
        `${id}.${p.key}: a numeric param with no clamp (add it to the table and clamp it in apply())`);
    }
  }
  // param แบบ select: ค่า default ต้องเป็นหนึ่งในตัวเลือกจริง และทุกตัวเลือกต้องมี label
  for (const id of NEW_OPS) {
    for (const p of OPS[id].params) {
      if (p.type !== 'select') continue;
      const values = p.options.map((o) => o.value);
      assert.ok(values.includes(p.default), `${id}.${p.key}: default "${p.default}" is not one of ${values}`);
      for (const o of p.options) assert.ok(o.label, `${id}.${p.key}: option ${o.value} has no label`);
    }
  }
  // defaultParams() ของกล่องต้องให้ค่าตรงกับ default ใน OPS (ค่าเริ่มต้นที่ผู้ใช้เห็นตอนลากกล่องลงมา)
  assert.equal(defaultParams('region_growing').tolerance, 12);
  const edgeDefaults = defaultParams('edge_detection');
  assert.equal(edgeDefaults.log_edge__sigma, 2);
  assert.equal(edgeDefaults.log_edge__zeroThresh, 8);
  assert.equal(edgeDefaults.prewitt_edge__dir, 'both');
  assert.equal(edgeDefaults.roberts_edge__dir, 'both');
});

// ===================================================================
// ipe-28 — codegen (ต้อง export Python ได้จริงทั้ง 4 ตัว)
// ===================================================================
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'ip-edge-2026-09-30-'));
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
// รันโค้ดที่ gen มาแบบอ่าน/เขียน raw bytes (เลี่ยง JPEG lossy) — แบบเดียวกับ tier2d-CG-3
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
  const stdout = execFileSync('python', [scriptPath, inPath, outPath], { encoding: 'utf8' });
  return { stdout, bytes: new Uint8Array(fs.readFileSync(outPath)) };
}

test('ipe-28. codegen: all four new ops really are exported as Python (they are NOT app-only) -- the generated script carries the expected cv2 calls, never prints "WARNING: skipped", never falls through to the generic "not supported yet" line, and region_growing tells the following steps whether the image is now grey (mask) or colour (overlay)', () => {
  const expectations = [
    ['prewitt_edge', { dir: 'both' }, [/kx = np\.array/, /ky = np\.array/, /cv2\.filter2D\(img, cv2\.CV_16S, kx\)/, /cv2\.addWeighted\(gx, 0\.5, gy, 0\.5, 0\)/]],
    ['prewitt_edge', { dir: 'x' }, [/cv2\.filter2D\(img, cv2\.CV_16S, kx\)/]],
    ['roberts_edge', { dir: 'both' }, [/k1 = np\.array\(\[\[1,0\],\[0,-1\]\], np\.float32\)/, /cv2\.addWeighted\(g1, 0\.5, g2, 0\.5, 0\)/]],
    ['roberts_edge', { dir: 'd2' }, [/cv2\.filter2D\(img, cv2\.CV_16S, k2\)/]],
    ['log_edge', { sigma: 2, zeroThresh: 8, view: 'zerocross' }, [/cv2\.GaussianBlur\(img, \(0, 0\), 2\)/, /cv2\.Laplacian\(blur, cv2\.CV_32F, ksize=3\)/, /np\.abs\(c - r\) > 8/, /zc\[:-1, :-1\]/]],
    ['log_edge', { view: 'response' }, [/cv2\.convertScaleAbs\(lap, alpha=alpha, beta=128\)/]],
    ['region_growing', { view: 'mask' }, [/cv2\.FLOODFILL_MASK_ONLY \| cv2\.FLOODFILL_FIXED_RANGE/, /\(255 << 8\)/, /filled, _, mask, _ = cv2\.floodFill/, /img = region/]],
    ['region_growing', { view: 'overlay', rangeMode: 'floating' }, [/cv2\.cvtColor\(img, cv2\.COLOR_GRAY2BGR\)/, /img\[region > 0\] = \(0, 255, 0\)/, /cv2\.circle\(img, \(sx, sy\), 4, \(0, 0, 255\), 2\)/]],
  ];
  for (const [opId, params, patterns] of expectations) {
    const code = buildPythonCode([{ id: opId, params }], { displayName: 'X' });
    assert.doesNotMatch(code, /WARNING: skipped/, `${opId}: must not be treated as app-only`);
    assert.doesNotMatch(code, /\(ข้าม: ยังไม่รองรับการ gen โค้ดของ/, `${opId}: must not fall through to the generic skip line`);
    assert.ok(code.includes(`(${opId}) ──`), `${opId}: step header`);
    assert.ok(code.includes('img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (แปลงขาวดำอัตโนมัติ)'), `${opId}: needs gray`);
    for (const re of patterns) assert.match(code, re, `${opId} ${JSON.stringify(params)}`);
  }
  // rangeMode floating ต้องไม่มี FIXED_RANGE
  const floating = buildPythonCode([{ id: 'region_growing', params: { rangeMode: 'floating' } }]);
  assert.doesNotMatch(floating, /FLOODFILL_FIXED_RANGE/, 'floating mode must not set FIXED_RANGE');
  // ค่าที่นอกช่วงต้องถูก clamp ในโค้ดที่ export เหมือนกับที่ registry clamp (ไม่งั้นแอปกับสคริปต์ให้ผลต่างกัน)
  const clamped = buildPythonCode([{ id: 'log_edge', params: { sigma: 99, zeroThresh: -5 } },
    { id: 'region_growing', params: { tolerance: 500, seedX: 500 } }]);
  assert.match(clamped, /cv2\.GaussianBlur\(img, \(0, 0\), 5\)/, 'sigma must be clamped to 5 in the exported code');
  assert.match(clamped, /np\.abs\(c - r\) > 0\)/, 'zeroThresh must be clamped to 0 in the exported code');
  assert.match(clamped, /\(100,\) \* 3/, 'tolerance must be clamped to 100 in the exported code');
  assert.match(clamped, /100 \/ 100\.0 \* \(w - 1\)/, 'seedX must be clamped to 100 in the exported code');
  // overlay ทำให้ภาพเป็นสี -> ขั้นถัดไปที่ต้องการภาพขาวดำ ต้องมีบรรทัดแปลงกลับ (mask ไม่ต้องมี)
  const afterOverlay = buildPythonCode([{ id: 'region_growing', params: { view: 'overlay' } }, { id: 'canny', params: { t1: 50, t2: 150 } }]);
  assert.equal((afterOverlay.match(/COLOR_BGR2GRAY/g) || []).length, 2, 'overlay output is colour -> Canny needs a second auto-gray line');
  const afterMask = buildPythonCode([{ id: 'region_growing', params: { view: 'mask' } }, { id: 'canny', params: { t1: 50, t2: 150 } }]);
  assert.equal((afterMask.match(/COLOR_BGR2GRAY/g) || []).length, 1, 'mask output is already grey -> no second conversion');
});

test('ipe-28b. codegen parity: the exported Python really produces the same pixels as the app for all four ops (cv2 + numpy)', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  // ภาพ 3 ช่องที่ทุกช่องเท่ากัน — cv2.cvtColor(BGR2GRAY) จึงให้ค่าเดิมเป๊ะเท่ากับภาพขาวดำตั้งต้น
  const gray = rampScene(320, 240, 101);
  const colour = grayAsChannels(gray, 3);
  try {
    const raw = matToRaw(colour);
    const cases = [
      ['prewitt_edge', { dir: 'both' }],
      ['prewitt_edge', { dir: 'x' }],
      ['roberts_edge', { dir: 'both' }],
      ['roberts_edge', { dir: 'd1' }],
      ['log_edge', { sigma: 2, zeroThresh: 8, view: 'zerocross' }],
      ['log_edge', { sigma: 2, zeroThresh: 0, view: 'zerocross' }],
      ['log_edge', { sigma: 2, view: 'response' }],
      ['region_growing', { seedX: 50, seedY: 50, tolerance: 12, view: 'mask' }],
      ['region_growing', { seedX: 20, seedY: 30, tolerance: 12, rangeMode: 'floating', view: 'mask' }],
    ];
    for (const [opId, params] of cases) {
      const code = buildPythonCode([{ id: opId, params }]);
      const { bytes } = runGeneratedPython(code, raw);
      const expected = OPS[opId].apply(cv, gray, params, {});
      try {
        const js = Buffer.from(expected.data);
        assert.equal(bytes.length, js.length, `${opId} ${JSON.stringify(params)}: output size`);
        let ndiff = 0;
        let maxdiff = 0;
        for (let i = 0; i < js.length; i++) {
          const d = Math.abs(js[i] - bytes[i]);
          if (d) { ndiff++; maxdiff = Math.max(maxdiff, d); }
        }
        assert.equal(ndiff, 0, `${opId} ${JSON.stringify(params)}: ndiff=${ndiff} maxdiff=${maxdiff} of ${js.length}`);
      } finally { expected.delete(); }
    }
  } finally { gray.delete(); colour.delete(); }
});

// ===================================================================
// ipe-28c/28d — rounding parity ของเมล็ด region_growing (MINOR-1)
// ปัญหาเดิม: แอปใช้ Math.round ของ JS (ปัดครึ่ง "ขึ้น") แต่โค้ดที่ export ใช้ round() ของ Python
// (ปัดครึ่งเข้าหา "เลขคู่") → ภาพกว้าง 22 ที่ 50% แอปได้ x=11 แต่สคริปต์ได้ 10 = flood fill คนละพื้นที่
// ===================================================================

// ขนาดที่กวาด: 1/2 = ขอบสุด (w - 1 = 0 และ 1), 22 กับ 26 = ขนาดที่พิสูจน์แล้วว่าเดิมให้ผลต่าง
const SEED_DIMS = [1, 2, 22, 26, 320, 640, 1920];
const SEED_PCTS = [0, 10, 25, 33, 50, 75, 95, 100];
// ipe-28c กวาดหนักกว่า: ทุกค่าจำนวนเต็มที่ slider ใน UI กดได้จริง (0–100) บวกค่าทศนิยมที่ไฟล์โปรเจกต์
// ที่แก้มือใส่มาได้ — ทุกค่าต้อง parity หมด ไม่ใช่แค่ค่ากลม ๆ ที่เลือกมาโชว์
const SEED_PCTS_FULL = [...Array.from({ length: 101 }, (_, i) => i), 0.5, 12.5, 33.333333, 66.666667, 99.9];

// จำลอง round() ของ Python (round-half-to-even) — ใส่ไว้เพื่อให้เทสต์ยังจับได้ถูกต้องแม้โค้ดจะถอยกลับไปใช้
// round() อีกครั้ง (ถ้าจำลองเป็นปัดขึ้น เทสต์จะกลายเป็น vacuous ทันที)
function pyRound(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// แปลง "บรรทัดที่ generate ออกมาจริง" ให้เป็นนิพจน์ JS ที่มี semantics แบบ Python เป๊ะ แล้ว eval
// (int() ของ Python = ตัดเศษเข้าหาศูนย์ = Math.trunc · min/max = Math.min/Math.max · round() = pyRound)
// ตั้งใจไม่เขียนสูตรซ้ำในเทสต์ — ค่าที่ได้ต้องมาจาก string ที่ buildPythonCode คืนมาเท่านั้น
// w กับ h ต้องแทนด้วยคนละค่าเสมอ ถ้าแทนด้วยเลขตัวเดียวกัน เทสต์จะมองไม่เห็นบรรทัดที่ copy-paste
// มาแล้วลืมเปลี่ยน w -> h (sy ที่ยังอ้าง w) เพราะบนภาพจัตุรัสค่ามันเท่ากันพอดี
function evalPythonSeedLine(line, w, h) {
  const expr = line.trim().replace(/^s[xy] = /, '')
    .replace(/\bmin\(/g, 'Math.min(')
    .replace(/\bmax\(/g, 'Math.max(')
    .replace(/\bint\(/g, 'Math.trunc(')
    .replace(/\bround\(/g, 'pyRound(')
    .replace(/\bw\b/g, String(w))
    .replace(/\bh\b/g, String(h));
  // กันกรณีบรรทัดถูกเขียนใหม่ด้วยฟังก์ชัน Python ตัวอื่น (เช่น math.floor) แล้วเทสต์ eval มั่ว ๆ ผ่านไปเฉย ๆ
  const leftover = expr.replace(/Math\.min|Math\.max|Math\.trunc|pyRound/g, '');
  assert.doesNotMatch(leftover, /[A-Za-z_]/,
    `unsupported python construct in the generated seed line (cannot be simulated safely): ${line.trim()}`);
  return Function('pyRound', `"use strict"; return (${expr});`)(pyRound);
}

// ดึงบรรทัด sx/sy ของขั้น region_growing ออกจากสคริปต์ที่ generate จริง
function seedLinesFor(pct) {
  const code = buildPythonCode([{ id: 'region_growing', params: { seedX: pct, seedY: pct } }], { displayName: 'X' });
  const sx = code.split('\n').map((l) => l.trim()).find((l) => l.startsWith('sx = '));
  const sy = code.split('\n').map((l) => l.trim()).find((l) => l.startsWith('sy = '));
  assert.ok(sx && sy, `seedX/seedY ${pct}: generated script must contain both an "sx = " and an "sy = " line`);
  return { code, sx, sy };
}

const REGISTRY_SRC_FOR_SEED = fs.readFileSync(new URL('../src/renderer/filters/registry.js', import.meta.url), 'utf8');

// สูตรฝั่งแอป — mirror มาจาก registry.js (region_growing, ราวบรรทัด 2449) ตรง ๆ
// กันเทสต์กลายเป็น vacuous: ipe-28c มี static-read assertion ว่า registry ยังเป็นสูตรนี้อยู่จริง
function appSeed(pct, dim) {
  return Math.min(dim - 1, Math.max(0, Math.round(pct / 100 * (dim - 1))));
}

test('ipe-28c. region_growing seed rounding parity (no python needed): the sx/sy lines in the exported script land on EXACTLY the same pixel as the app for every size x percentage -- evaluated with real Python semantics (int() = trunc, round() = half-to-even) straight off the generated text, never from a formula rewritten in the test', () => {
  // (static read) ถ้า registry เปลี่ยนสูตรเมล็ด เทสต์นี้ต้องพังทันที ไม่ใช่ผ่านไปเงียบ ๆ กับสูตรเก่าที่ mirror ไว้
  assert.match(REGISTRY_SRC_FOR_SEED,
    /const sx = Math\.min\(cols - 1, Math\.max\(0, Math\.round\(numIn\(p\.seedX, 50, 0, 100\) \/ 100 \* \(cols - 1\)\)\)\);/,
    'registry.js region_growing seedX formula changed -- update appSeed() in this test to match it');
  assert.match(REGISTRY_SRC_FOR_SEED,
    /const sy = Math\.min\(rows - 1, Math\.max\(0, Math\.round\(numIn\(p\.seedY, 50, 0, 100\) \/ 100 \* \(rows - 1\)\)\)\);/,
    'registry.js region_growing seedY formula changed -- update appSeed() in this test to match it');

  for (const pct of SEED_PCTS_FULL) {
    const { code, sx, sy } = seedLinesFor(pct);
    // ข้อความ: ห้ามกลับไปใช้ round() และต้องมี + 0.5 (ปัดครึ่งขึ้นแบบเดียวกับ Math.round)
    assert.doesNotMatch(sx, /round\(/, `seedX ${pct}: the sx line must not use Python round() (banker's rounding)`);
    assert.doesNotMatch(sy, /round\(/, `seedY ${pct}: the sy line must not use Python round() (banker's rounding)`);
    assert.match(sx, /\+ 0\.5\)/, `seedX ${pct}: the sx line must round half up via "+ 0.5"`);
    assert.match(sy, /\+ 0\.5\)/, `seedY ${pct}: the sy line must round half up via "+ 0.5"`);
    // ห้ามแอบเติม import math เข้า header (จะพ่วงไปทุก op ที่ export)
    assert.doesNotMatch(code, /^import math$/m, `seedX ${pct}: the generated script must not import math`);
    assert.doesNotMatch(code, /\bmath\./, `seedX ${pct}: the generated script must not call into the math module`);

    for (const w of SEED_DIMS) {
      for (const h of SEED_DIMS) {
        // ป้อน w กับ h แยกกันทั้งสองบรรทัด: ถ้า sy ไปอ้าง w (หรือ sx ไปอ้าง h) ค่าจะหลุดทันทีที่ w != h
        const py = { sx: evalPythonSeedLine(sx, w, h), sy: evalPythonSeedLine(sy, w, h) };
        assert.equal(py.sx, appSeed(pct, w), `seed ${pct}% on w=${w}: app=${appSeed(pct, w)} but the exported script computes sx=${py.sx}`);
        assert.equal(py.sy, appSeed(pct, h), `seed ${pct}% on h=${h}: app=${appSeed(pct, h)} but the exported script computes sy=${py.sy}`);
        // เมล็ดต้องอยู่ในกรอบภาพเสมอ (นอกกรอบ = floodFill throw ทั้งสองฝั่ง)
        assert.ok(py.sx >= 0 && py.sx <= w - 1, `seed ${pct}% on w=${w}: sx=${py.sx} is outside the image`);
        assert.ok(py.sy >= 0 && py.sy <= h - 1, `seed ${pct}% on h=${h}: sy=${py.sy} is outside the image`);
      }
    }
  }
  // ตัวอย่างที่เคยพังจริง (จาก qa รอบรันจริง): ถ้าหลุดกลับไปใช้ round() ค่าพวกนี้จะต่างทันที
  assert.equal(appSeed(50, 22), 11, 'guard: 50% of a 22px-wide image is x=11 in the app');
  assert.equal(appSeed(10, 26), 3, 'guard: 10% of a 26px-wide image is x=3 in the app');
});

test('ipe-28d. region_growing seed rounding parity, verified by really running python: the sx/sy lines from the exported script, executed by the python interpreter itself, give the same pixel as the app for every size x percentage', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  const sizes = JSON.stringify(SEED_DIMS.map((d) => [d, d]));
  const lines = ['import json', 'out = []'];
  for (const pct of SEED_PCTS) {
    const { sx, sy } = seedLinesFor(pct);
    lines.push(`for w, h in ${sizes}:`, `    ${sx}`, `    ${sy}`, `    out.append([${pct}, w, h, sx, sy])`);
    // ขนาดที่ w != h ด้วย เพื่อพิสูจน์ว่า sx ใช้ w และ sy ใช้ h ไม่สลับกัน
    lines.push('for w, h in [[22, 26], [26, 22], [1920, 2], [2, 1920]]:', `    ${sx}`, `    ${sy}`,
      `    out.append([${pct}, w, h, sx, sy])`);
  }
  lines.push('print(json.dumps(out))');
  const scriptPath = path.join(TMP_DIR, `seedparity_${Math.random().toString(36).slice(2)}.py`);
  fs.writeFileSync(scriptPath, lines.join('\n'), 'utf8');
  const rows = JSON.parse(execFileSync('python', [scriptPath], { encoding: 'utf8' }));
  assert.equal(rows.length, SEED_PCTS.length * (SEED_DIMS.length + 4), 'every case must have been evaluated by python');
  for (const [pct, w, h, sx, sy] of rows) {
    assert.equal(sx, appSeed(pct, w), `python says sx=${sx} for ${pct}% of w=${w}, the app says ${appSeed(pct, w)}`);
    assert.equal(sy, appSeed(pct, h), `python says sy=${sy} for ${pct}% of h=${h}, the app says ${appSeed(pct, h)}`);
  }
});

// ===================================================================
// ipe-29 — palette / ui-meta / registry wiring
// ===================================================================
test('ipe-29. wiring: every new op is on the palette, resolves to a registered filter, has a unique icon glyph and an OP_PRIMARY line -- and the three edge ops joined the existing Edge Detection box while region_growing became its own Segmentation node (no new group, no reordering)', () => {
  const paletteSource = fs.readFileSync(new URL('../src/renderer/ui/palette-panel.js', import.meta.url), 'utf8');
  const uiMetaSource = fs.readFileSync(new URL('../src/renderer/filters/ui-meta.js', import.meta.url), 'utf8');
  const segSection = TOOL_SECTIONS.find((s) => s.title === 'SEGMENTATION & EDGE');
  assert.ok(segSection, 'the SEGMENTATION & EDGE row must exist');

  for (const id of NEW_OPS) {
    assert.ok(OPS[id], `${id}: missing from OPS`);
    assert.ok(segSection.ops.includes(id), `${id}: not on the SEGMENTATION & EDGE palette row`);
    const filterId = OP_TO_FILTER[id];
    assert.ok(filterId && FILTERS[filterId], `${id}: does not resolve to a registered filter`);
    assert.ok(typeof OP_PRIMARY[id] === 'function', `${id}: no OP_PRIMARY line`);
    assert.ok(OP_PRIMARY[id](defaultParams(filterId === id ? id : filterId)).length > 0, `${id}: OP_PRIMARY returned nothing`);
    assert.ok(OPS[id].name && OPS[id].desc, `${id}: needs a name and a Thai description`);
    assert.equal(OPS[id].needs, 'gray', `${id}: must declare needs "gray"`);
  }
  // 2026-10-07: palette ชี้กลุ่มระดับหมวด segment_edge (edge_detection/region_growing เดิมยังอยู่ใน FILTERS เป็น legacy)
  assert.equal(OP_TO_FILTER.prewitt_edge, 'segment_edge');
  assert.equal(OP_TO_FILTER.roberts_edge, 'segment_edge');
  assert.equal(OP_TO_FILTER.log_edge, 'segment_edge');
  assert.equal(OP_TO_FILTER.region_growing, 'segment_edge');
  assert.equal(FILTERS.region_growing.group, 'Segmentation');
  assert.equal(FILTERS.edge_detection.group, 'Edge Detection');
  assert.equal(GROUP_ICON.region_growing, '🪄');

  // การ์ดบนกระดานต้องอ่านค่าผ่าน pnum: ค่าที่แก้มือเกินช่วงต้องโชว์ "ค่าหลัง clamp" ไม่ใช่ค่าดิบ
  assert.match(OP_PRIMARY.log_edge({ sigma: 99, zeroThresh: -5 }), /sigma 5 · เกณฑ์ 0/);
  assert.match(OP_PRIMARY.region_growing({ seedX: 500, seedY: -3, tolerance: 999 }), /จุดเริ่ม \(100%, 0%\) · ต่างได้ 100/);
  assert.match(OP_PRIMARY.log_edge({}), /sigma 2 · เกณฑ์ 8/, 'no params -> the registry defaults');
  assert.match(OP_PRIMARY.prewitt_edge({}), /BOTH/);
  assert.match(OP_PRIMARY.roberts_edge({ dir: 'd1' }), /D1/);

  // glyph ต้องไม่ซ้ำกับตัวใดในไฟล์ palette-panel.js/ui-meta.js (อ่านแบบ static เพราะ OP_ICON ไม่ได้ export)
  const icons = {};
  for (const m of paletteSource.matchAll(/^\s{2}([a-z0-9_]+): "([^"]+)",/gm)) icons[m[1]] = m[2];
  for (const m of paletteSource.matchAll(/([a-z0-9_]+): "([^"]+)"/g)) if (!icons[m[1]]) icons[m[1]] = m[2];
  for (const id of NEW_OPS) assert.ok(icons[id], `${id}: no OP_ICON entry in palette-panel.js`);
  const glyphs = Object.values(icons);
  for (const id of NEW_OPS) {
    const g = icons[id];
    assert.equal(glyphs.filter((x) => x === g).length, 1, `${id}: glyph "${g}" is used more than once in palette-panel.js`);
    const inUiMeta = [...uiMetaSource.matchAll(/: "([^"]+)",/g)].map((m) => m[1]).filter((x) => x === g);
    // ui-meta มี GROUP_ICON.region_growing = glyph เดียวกันโดยตั้งใจ (single-op group: filterId = opId)
    assert.ok(inUiMeta.length <= (id === 'region_growing' ? 1 : 0), `${id}: glyph "${g}" collides with a GROUP_ICON in ui-meta.js`);
  }

  // กล่อง Edge Detection ต้องมีครบ 7 วิธี และ mode ที่เลือกได้ต้องครอบ op ใหม่ทั้ง 3
  const modeParam = FILTERS.edge_detection.params.find((p) => p.key === 'mode');
  assert.deepEqual(modeParam.options.map((o) => o.value),
    ['canny', 'sobel', 'laplacian_edge', 'scharr_edge', 'prewitt_edge', 'roberts_edge', 'log_edge']);
  assert.equal(modeParam.default, 'canny', 'the existing default must not change');
  // currentOpId/localParamsFor ต้องแกะ sub-param ของ op ใหม่ได้ (เส้นทางที่ detail panel ใช้จริง)
  const props = { mode: 'log_edge', log_edge__sigma: 3, scharr_edge__dir: 'x' };
  assert.equal(currentOpId('edge_detection', props), 'log_edge');
  assert.deepEqual(localParamsFor('edge_detection', props), { sigma: 3 });
});

test('ipe-29b. the 18d Motion ops and the pre-existing edge ops are untouched by this round: their palette row, icons and OPS entries are exactly where they were', () => {
  const motion = TOOL_SECTIONS.find((s) => s.title === 'MOTION');
  assert.deepEqual(motion.ops, ['mog2', 'farneback_optical_flow', 'pyrlk_optical_flow']);
  for (const id of ['canny', 'sobel', 'laplacian_edge', 'scharr_edge', 'watershed', 'grabcut']) {
    assert.ok(OPS[id], `${id} must still exist`);
  }
  assert.equal(TOOL_SECTIONS.length, 12, 'no new palette row was added this round');
});

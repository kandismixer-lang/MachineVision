// 2026-10-01 — acceptance tests ของ IP batch 2 ชิ้น A (หมวด ENHANCE: contrast_stretch, flatfield_correction ·
// หมวด MORPHOLOGY: convex_hull, endpoints_branchpoints เป็น p.op ใหม่ใน OPS.morphology)
// ใบสั่งงาน: ข้อความงานจาก qa ตรง ๆ (ไม่มีไฟล์ spec แยก) — ตัวเลขเกณฑ์ผ่านทุกตัวยืนยันด้วยการรันจริงใน
// scratchpad ก่อนลงโค้ด (คำนวณมือ + เทียบ cv2 5.0.0 / numpy 2.5.1 ที่ติดตั้งอยู่ในเครื่องนี้) — ถ้าเทสต์ล้ม
// ให้ถือว่าโค้ดผิด ไม่ใช่ตัวเลขผิด
//
// นโยบาย leak (P-14, WORK-TRACKER §1): address-fingerprint ใช้ได้เฉพาะเทสต์แรกของไฟล์ (ipA-0) พร้อมคู่แฝด
// wrapper-balance (ipA-0w) — เทสต์อื่นใช้ assertLeakFree / assertNoHeapGrowth
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

const NEW_ENHANCE_OPS = ['contrast_stretch', 'flatfield_correction'];
const NEW_MORPH_MODES = ['convex_hull', 'endpoints_branchpoints'];

// ===================================================================
// ภาพทดสอบ (deterministic ทั้งหมด — ไม่ใช้ Math.random)
// ===================================================================
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}
// ไล่ระดับ 60..200 ซ้ายไปขวา + noise deterministic (ใช้ทดสอบ contrast_stretch percentile cut)
function noisyRamp(w, h, seed = 777, noise = 30) {
  const d = new Uint8Array(w * h);
  const rnd = lcg(seed);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const base = 60 + (140 * x) / (w - 1);
      const n = (rnd() - 0.5) * noise;
      d[y * w + x] = Math.max(0, Math.min(255, Math.round(base + n)));
    }
  }
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
// ramp 0..255 เป๊ะ: แถวเดียวกัน, ทุกระดับความสว่างปรากฏ "rowsPerLevel" ครั้งพอดี (คุม histogram ได้แม่นยำ)
function exactRamp(levels = 256, rowsPerLevel = 100) {
  const w = levels, h = rowsPerLevel;
  const d = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d[y * w + x] = Math.round((x * 255) / (w - 1));
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
function flatGray(v = 128, w = 20, h = 20) {
  return cv.matFromArray(h, w, cv.CV_8UC1, new Uint8Array(w * h).fill(v));
}
// แสงไม่สม่ำเสมอ (ซ้ายมืด-ขวาสว่าง) + วัตถุสี่เหลี่ยมเล็กตรงกลางที่สะท้อนแสงมากกว่าพื้นหลัง — ใช้ทดสอบ
// flatfield_correction (ภาพนี้ "เอียง" ความสว่างจริง จึงพิสูจน์ว่า correction ทำงานจริง ไม่ใช่ identity)
function unevenLightGray(w, h) {
  const d = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const lighting = 40 + (220 - 40) * (x / (w - 1));
      let v = lighting * 0.5;
      if (x > w * 0.4 && x < w * 0.5 && y > h * 0.4 && y < h * 0.5) v = lighting * 0.9;
      d[y * w + x] = Math.max(0, Math.min(255, Math.round(v)));
    }
  }
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
function unevenLightColour(w, h) {
  const d = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const lighting = 40 + (220 - 40) * (x / (w - 1));
      let r = lighting * 0.5; let g = lighting * 0.6; let b = lighting * 0.4;
      if (x > w * 0.4 && x < w * 0.5 && y > h * 0.4 && y < h * 0.5) { r = lighting * 0.9; g = lighting * 0.85; b = lighting * 0.8; }
      const i = (y * w + x) * 3;
      d[i] = Math.max(0, Math.min(255, Math.round(r)));
      d[i + 1] = Math.max(0, Math.min(255, Math.round(g)));
      d[i + 2] = Math.max(0, Math.min(255, Math.round(b)));
    }
  }
  return cv.matFromArray(h, w, cv.CV_8UC3, d);
}
// สี่เหลี่ยม 10..40 ที่มีรอยบากเว้าเข้าไปจากขอบขวา (x>=25,y 20..30 ถูกเจาะ) = concave silhouette จริง
// (ต่างจาก "รู" ภายในที่ไม่ถูกเจาะถึงขอบ ซึ่ง RETR_EXTERNAL จะไม่เห็นเป็นความเว้า) + noise speck 1px และ 2px
// (ทดสอบ guard "ข้าม contour < 3 จุด")
function concaveShapeWithSpecks(w = 60, h = 60) {
  const d = new Uint8Array(w * h).fill(0);
  const set = (x, y) => { if (x >= 0 && x < w && y >= 0 && y < h) d[y * w + x] = 255; };
  for (let y = 10; y < 40; y++) {
    for (let x = 10; x < 40; x++) {
      if (x >= 25 && x < 40 && y >= 20 && y < 30) continue; // บากเว้าเปิดออกขอบขวา
      set(x, y);
    }
  }
  set(50, 5); // speck 1 พิกเซล
  set(55, 50); set(56, 50); // speck 2 พิกเซล
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
// โดนัท: สี่เหลี่ยมทึบมีรูสี่เหลี่ยมกลวงตรงกลาง (รูไม่แตะขอบภาพ) -- RETR_EXTERNAL ต้องเห็นแค่ 1 contour
// (ขอบนอก) แต่ RETR_LIST จะเห็น 2 (ขอบนอก + ขอบรู) ใช้ยืนยันว่า convex_hull ใช้ RETR_EXTERNAL จริง
function donutShape(w = 40, h = 40) {
  const d = new Uint8Array(w * h).fill(0);
  for (let y = 5; y < 35; y++) for (let x = 5; x < 35; x++) d[y * w + x] = 255;
  for (let y = 15; y < 25; y++) for (let x = 15; x < 25; x++) d[y * w + x] = 0;
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
// skeleton เส้นเดียวไม่มีกิ่ง (2 endpoints, 0 branchpoints)
function straightLine(w = 60, h = 40) {
  const d = new Uint8Array(w * h).fill(0);
  for (let x = 10; x <= 50; x++) d[20 * w + x] = 255;
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
// กิ่ง 'Y' (ก้านตั้ง + สองแขนทแยง) + เส้นตรงแยกต่างหาก = endpoints/branchpoints ที่ยืนยันค่าด้วยมือแล้ว
function yBranchPlusLine(w = 60, h = 40) {
  const d = new Uint8Array(w * h).fill(0);
  const set = (x, y) => { if (x >= 0 && x < w && y >= 0 && y < h) d[y * w + x] = 255; };
  for (let y = 5; y <= 25; y++) set(20, y);
  for (let i = 0; i <= 10; i++) { set(20 + i, 25 + i); set(20 - i, 25 + i); }
  for (let x = 35; x <= 45; x++) set(x, 10);
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
// เครื่องหมายบวก 21x21 (เส้นนอน y=10 x=2..18 + เส้นตั้ง x=10 y=2..18): จุดตัดจริงมี 1 จุดที่ (10,10) แต่พิกเซลข้างจุดตัด
// 4 จุด ((9,10),(11,10),(10,9),(10,11)) ก็มีเพื่อนบ้านขาว >= 3 เหมือนกัน — กฎเก่า "n>=3" จะนับจุดแยกเป็น 5 (ผิด)
// กฎ Crossing Number ต้องได้ 1 (qa วัดแล้ว: จุดปลาย 4 จุด · จุดแยก 1 จุด) -- fixture นี้คือตัวแยกสองกฎออกจากกัน
// (ส่วน Y fixture ด้านบนแยกไม่ได้เพราะจุดแยกมีเพื่อนบ้านพอดี 3 และพิกเซลข้างเคียงมีแค่ 2)
function plusSign(w = 21, h = 21) {
  const d = new Uint8Array(w * h).fill(0);
  for (let x = 2; x <= 18; x++) d[10 * w + x] = 255;
  for (let y = 2; y <= 18; y++) d[y * w + 10] = 255;
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
// กากบาท X 21x21 (ทแยงสองเส้นตัดกันที่ (10,10)) -- จุดตัดมีเพื่อนบ้านทแยง 4 จุด สลับขาว-ดำรอบวงแหวน (CN=4)
function crossX(w = 21, h = 21) {
  const d = new Uint8Array(w * h).fill(0);
  for (let i = 2; i <= 18; i++) { d[i * w + i] = 255; d[i * w + (20 - i)] = 255; }
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
// 60x40: เส้นนอนบาง 1px ยาว 46px (y=5, x=5..50) + เส้นทแยงบาง 1px ยาว 20px + บล็อก 15x20 (qa พิสูจน์: contour ของเส้น
// บาง 1px ทุกความยาวมีแค่ 2 จุดหลัง CHAIN_APPROX_SIMPLE → กฎเก่า "ข้าม contour <3 จุด" ทำให้เส้นหายจากผลลัพธ์/จำนวน)
function linesAndBlock(w = 60, h = 40) {
  const d = new Uint8Array(w * h).fill(0);
  for (let x = 5; x <= 50; x++) d[5 * w + x] = 255; // 46px
  for (let i = 0; i < 20; i++) d[(12 + i) * w + (5 + i)] = 255; // 20px ทแยง
  for (let y = 15; y < 35; y++) for (let x = 35; x < 50; x++) d[y * w + x] = 255; // บล็อก 15x20
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
// ramp ค่า 50..200 (151 คอลัมน์ x rows แถว) -- มืดสุด 50 / สว่างสุด 200 ไม่มีพิกเซลที่ค่า 0 เลย
function rampBetween(lo = 50, hi = 200, rows = 10) {
  const w = hi - lo + 1;
  const d = new Uint8Array(w * rows);
  for (let y = 0; y < rows; y++) for (let x = 0; x < w; x++) d[y * w + x] = lo + x;
  return cv.matFromArray(rows, w, cv.CV_8UC1, d);
}
// 399x601 (239799 พิกเซล): 238600 พิกเซลแรก (ลำดับ raster) = 10 ที่เหลือ = 200 — highPct=0.5 → highCount=238600.005
// ซึ่ง cum ที่ระดับ 10 (=238600) ต้อง "ยังไม่ถึง" ในการเทียบแบบ float64 แต่ float32 ปัด 238600.005 เป็น 238600.0 แล้วนับว่าถึง
// (ต้นเหตุของ MINOR-1: codegen ฝั่ง Python สะสมใน float32 ภายใต้ NumPy 2)
function float32TrapImage() {
  const w = 399; const h = 601;
  const d = new Uint8Array(w * h).fill(200);
  d.fill(10, 0, 238600);
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
function grayAsChannels(gray, channels) {
  const out = new cv.Mat();
  cv.cvtColor(gray, out, channels === 4 ? cv.COLOR_GRAY2RGBA : cv.COLOR_GRAY2RGB);
  return out;
}

// ===================================================================
// ipA-0 / ipA-0w — ด่าน leak หลัก (P-14 ข้อ 1 + คู่บังคับข้อ 2)
// ===================================================================
function batchALeakPaths() {
  const ramp = noisyRamp(48, 36, 11);
  const rampRgba = grayAsChannels(ramp, 4);
  const flat = flatGray(128, 20, 20);
  const uneven = unevenLightGray(48, 36);
  const unevenRgba = grayAsChannels(uneven, 4);
  const concave = concaveShapeWithSpecks();
  const noObjects = flatGray(0, 20, 20);
  const yBranch = yBranchPlusLine();
  const noLine = flatGray(0, 20, 20);
  const unevenColour = unevenLightColour(48, 36);
  const line = straightLine();
  const plus = plusSign();
  const lines = linesAndBlock();
  const ff320 = unevenLightGray(320, 240); // ด้านยาว > 240 → เส้นทาง downsample (ภาพจากกล้องจริงทุกเฟรมเข้าทางนี้)
  const ff320c = unevenLightColour(320, 240);
  const empty = new cv.Mat();
  const paths = [];
  const add = (id, params, src, tag) => paths.push({
    label: `${id} ${tag}`,
    run: (c) => { const o = OPS[id].apply(c, src, params, {}); o.delete(); },
  });
  for (const [lowPct, highPct] of [[1, 1], [0, 0], [5, 5]]) add('contrast_stretch', { lowPct, highPct }, ramp, `lo=${lowPct} hi=${highPct}`);
  add('contrast_stretch', { lowPct: 1, highPct: 1 }, flat, 'uniform image (guard)');
  add('contrast_stretch', { lowPct: 1, highPct: 1 }, rampRgba, 'rgba input');
  add('contrast_stretch', {}, empty, 'empty 0x0');
  for (const blurSize of [15, 61, 121]) add('flatfield_correction', { blurSize }, uneven, `blurSize=${blurSize}`);
  add('flatfield_correction', { blurSize: 61 }, unevenColour, 'colour input');
  add('flatfield_correction', { blurSize: 61 }, unevenRgba, 'rgba input');
  add('flatfield_correction', { blurSize: 121 }, ff320, 'downsample path 320x240 gray');
  add('flatfield_correction', { blurSize: 15 }, ff320c, 'downsample path 320x240 colour');
  add('flatfield_correction', {}, empty, 'empty 0x0');
  add('morphology', { op: 'convex_hull' }, concave, 'normal shapes + specks');
  add('morphology', { op: 'convex_hull' }, lines, 'thin lines + block');
  add('morphology', { op: 'convex_hull' }, noObjects, 'no objects');
  add('morphology', { op: 'convex_hull' }, empty, 'empty 0x0');
  add('morphology', { op: 'endpoints_branchpoints' }, yBranch, 'Y branch + line');
  add('morphology', { op: 'endpoints_branchpoints' }, plus, 'plus sign (crossing number)');
  add('morphology', { op: 'endpoints_branchpoints' }, line, 'straight line only');
  add('morphology', { op: 'endpoints_branchpoints' }, noLine, 'no line');
  add('morphology', { op: 'endpoints_branchpoints' }, empty, 'empty 0x0');
  return {
    paths,
    cleanup: () => {
      ramp.delete(); rampRgba.delete(); flat.delete(); uneven.delete(); unevenRgba.delete();
      concave.delete(); noObjects.delete(); yBranch.delete(); noLine.delete();
      unevenColour.delete(); line.delete(); plus.delete(); lines.delete(); ff320.delete(); ff320c.delete(); empty.delete();
    },
  };
}

test('ipA-0. leak (address fingerprint, first test of a clean process): every branch of all four new ops (contrast_stretch normal/uniform-guard/rgba/empty, flatfield_correction several blur sizes/colour/rgba/empty, convex_hull normal/no-object/empty, endpoints_branchpoints normal/no-line/empty) leaves the malloc fingerprint unchanged', () => {
  const { paths, cleanup } = batchALeakPaths();
  try {
    fingerprintLeakProbe(cv, 'IP batch 2 chunk A ops (all paths)', () => {
      for (const p of paths) p.run(cv);
    }, { iters: 25, warmup: 10 });
  } finally { cleanup(); }
});

test('ipA-0w. leak (wrapper balance twin of ipA-0): the exact same path table creates and deletes every embind wrapper it makes (hist/mask/lut, the float Mats + epsilon/mean constant Mats, contours/hull, the drawn BGR overlay)', () => {
  const { paths, cleanup } = batchALeakPaths();
  try {
    assertWrapperBalance(cv, 'IP batch 2 chunk A ops (all paths)', (c) => {
      for (const p of paths) p.run(c);
    }, { minCreated: 80 });
  } finally { cleanup(); }
});

// ===================================================================
// ipA-L1…L5 — leak ต่อ op / ต่อเส้นทาง (P-14 ข้อ 2 + 3 ในคำสั่งเดียว)
// ===================================================================
test('ipA-L1. leak: contrast_stretch over several lowPct/highPct, the uniform-image guard path, and a multi-channel input balances every wrapper and grows the heap by 0', () => {
  const ramp = noisyRamp(48, 36, 22);
  const flat = flatGray(128, 20, 20);
  const rgba = grayAsChannels(ramp, 4);
  try {
    for (const [lowPct, highPct] of [[1, 1], [0, 0], [2.5, 0.5], [10, 10]]) {
      assertLeakFree(cv, `contrast_stretch lo=${lowPct} hi=${highPct}`, (c) => {
        const o = OPS.contrast_stretch.apply(c, ramp, { lowPct, highPct }, {});
        o.delete();
      }, { minCreated: 3, iters: 60, warmup: 20 });
    }
    assertLeakFree(cv, 'contrast_stretch uniform image (guard returns clone)', (c) => {
      const o = OPS.contrast_stretch.apply(c, flat, { lowPct: 1, highPct: 1 }, {});
      o.delete();
    }, { minCreated: 3, iters: 60, warmup: 20 });
    assertLeakFree(cv, 'contrast_stretch rgba input', (c) => {
      const o = OPS.contrast_stretch.apply(c, rgba, { lowPct: 1, highPct: 1 }, {});
      o.delete();
    }, { minCreated: 3, iters: 60, warmup: 20 });
  } finally { ramp.delete(); flat.delete(); rgba.delete(); }
});

test('ipA-L2. leak: flatfield_correction over several blur sizes, a colour (3ch) input, and an all-black input (epsilon guard against divide-by-zero) balances every wrapper and grows the heap by 0', () => {
  const gray = unevenLightGray(48, 36);
  const colour = unevenLightColour(48, 36);
  const black = flatGray(0, 20, 20);
  try {
    for (const blurSize of [15, 61, 121, 201]) {
      assertLeakFree(cv, `flatfield_correction blurSize=${blurSize}`, (c) => {
        const o = OPS.flatfield_correction.apply(c, gray, { blurSize }, {});
        o.delete();
      }, { minCreated: 8, iters: 40, warmup: 15 });
    }
    assertLeakFree(cv, 'flatfield_correction colour input', (c) => {
      const o = OPS.flatfield_correction.apply(c, colour, { blurSize: 61 }, {});
      o.delete();
    }, { minCreated: 8, iters: 40, warmup: 15 });
    assertLeakFree(cv, 'flatfield_correction all-black input (background would be 0 without +epsilon)', (c) => {
      const o = OPS.flatfield_correction.apply(c, black, { blurSize: 15 }, {});
      o.delete();
    }, { minCreated: 8, iters: 40, warmup: 15 });
  } finally { gray.delete(); colour.delete(); black.delete(); }
});

test('ipA-L2b. leak (P-14, the main REAL path): flatfield_correction on images whose long side is > 240 px (every real camera frame takes the downsample branch: resize INTER_AREA -> GaussianBlur -> resize INTER_LINEAR) -- gray and colour, blurSize 15 / 121 (default) / 401 -- balances every wrapper (including the extra `analysis` and `bgSmall` Mats) and grows the heap by 0', () => {
  const gray = unevenLightGray(320, 240);
  const colour = unevenLightColour(320, 240);
  const tall = unevenLightGray(7, 1000); // thin image: aw/ah are clamped by Math.max(1, ...)
  try {
    for (const blurSize of [15, 121, 401]) {
      assertLeakFree(cv, `flatfield_correction 320x240 gray blurSize=${blurSize} (downsample path)`, (c) => {
        const o = OPS.flatfield_correction.apply(c, gray, { blurSize }, {});
        o.delete();
      }, { minCreated: 10, iters: 25, warmup: 8 });
      assertLeakFree(cv, `flatfield_correction 320x240 colour blurSize=${blurSize} (downsample path)`, (c) => {
        const o = OPS.flatfield_correction.apply(c, colour, { blurSize }, {});
        o.delete();
      }, { minCreated: 10, iters: 25, warmup: 8 });
    }
    assertLeakFree(cv, 'flatfield_correction 7x1000 gray (thin, downsample path)', (c) => {
      const o = OPS.flatfield_correction.apply(c, tall, { blurSize: 121 }, {});
      o.delete();
    }, { minCreated: 10, iters: 25, warmup: 8 });
  } finally { gray.delete(); colour.delete(); tall.delete(); }
});

test('ipA-L3. leak: morphology convex_hull over normal shapes+specks (1px/2px specks are hulled, drawn and counted like any other shape), thin 1px lines + a block (2-point contours), and a no-object image balances every wrapper (contours MatVector, each contour wrapper, each hull Mat + its 1-item MatVector) and grows the heap by 0', () => {
  const concave = concaveShapeWithSpecks();
  const noObjects = flatGray(0, 20, 20);
  const lines = linesAndBlock();
  try {
    assertLeakFree(cv, 'convex_hull normal shapes + 1px/2px specks', (c) => {
      const o = OPS.morphology.apply(c, concave, { op: 'convex_hull' }, {});
      o.delete();
    }, { minCreated: 5, iters: 50, warmup: 20 });
    assertLeakFree(cv, 'convex_hull thin lines + block (2-point contours)', (c) => {
      const o = OPS.morphology.apply(c, lines, { op: 'convex_hull' }, {});
      o.delete();
    }, { minCreated: 5, iters: 50, warmup: 20 });
    assertLeakFree(cv, 'convex_hull no objects found', (c) => {
      const o = OPS.morphology.apply(c, noObjects, { op: 'convex_hull' }, {});
      o.delete();
    }, { minCreated: 3, iters: 50, warmup: 20 });
  } finally { concave.delete(); noObjects.delete(); lines.delete(); }
});

test('ipA-L3b. leak on the exception path inside the per-contour loop: if hullVec.push_back() throws (debugger-found bug -- `hullVec` used to be created+push_back\'d BEFORE the inner try opened, so a throwing push_back skipped its own cleanup and leaked the MatVector every time), hullVec/hull/contour/bin/contours/hier are still all released and the exception still propagates', () => {
  const square = cv.matFromArray(20, 20, cv.CV_8UC1, (() => {
    const d = new Uint8Array(400).fill(0);
    for (let y = 5; y < 15; y++) for (let x = 5; x < 15; x++) d[y * 20 + x] = 255;
    return d;
  })());
  const realPushBack = cv.MatVector.prototype.push_back;
  try {
    let calls = 0;
    // throw on the first (and only, for this 1-shape fixture) push_back call -- restored in finally below
    cv.MatVector.prototype.push_back = function patchedPushBack(...args) {
      calls++;
      if (calls === 1) throw new Error('injected test failure (push_back)');
      return realPushBack.apply(this, args);
    };
    assertWrapperBalance(cv, 'convex_hull with push_back throwing on the hullVec that wraps the one real shape', (c) => {
      OPS.morphology.apply(c, square, { op: 'convex_hull' }, {});
    }, { expectThrow: /injected test failure \(push_back\)/, minCreated: 5 });
  } finally {
    cv.MatVector.prototype.push_back = realPushBack;
    square.delete();
  }
});

test('ipA-L4. leak: morphology endpoints_branchpoints over a branching skeleton, a single straight line, and an empty (no-line) image balances every wrapper and grows the heap by 0', () => {
  const yBranch = yBranchPlusLine();
  const line = straightLine();
  const noLine = flatGray(0, 20, 20);
  const plus = plusSign();
  try {
    assertLeakFree(cv, 'endpoints_branchpoints plus sign (crossing-number path)', (c) => {
      const o = OPS.morphology.apply(c, plus, { op: 'endpoints_branchpoints' }, {});
      o.delete();
    }, { minCreated: 2, iters: 50, warmup: 20 });
    assertLeakFree(cv, 'endpoints_branchpoints Y branch + separate line', (c) => {
      const o = OPS.morphology.apply(c, yBranch, { op: 'endpoints_branchpoints' }, {});
      o.delete();
    }, { minCreated: 2, iters: 50, warmup: 20 });
    assertLeakFree(cv, 'endpoints_branchpoints straight line only', (c) => {
      const o = OPS.morphology.apply(c, line, { op: 'endpoints_branchpoints' }, {});
      o.delete();
    }, { minCreated: 2, iters: 50, warmup: 20 });
    assertLeakFree(cv, 'endpoints_branchpoints no line found', (c) => {
      const o = OPS.morphology.apply(c, noLine, { op: 'endpoints_branchpoints' }, {});
      o.delete();
    }, { minCreated: 2, iters: 50, warmup: 20 });
  } finally { yBranch.delete(); line.delete(); noLine.delete(); plus.delete(); }
});

test('ipA-L5. leak + proof the 0x0 guard is necessary for two of the four ops: cv.GaussianBlur and cv.cvtColor(...,GRAY2RGB) both throw a C++ exception on an empty Mat (that is why flatfield_correction and endpoints_branchpoints guard first) -- cv.calcHist and cv.findContours do NOT throw on an empty/empty-derived Mat (verified directly below), so contrast_stretch/convex_hull\'s guard is a defensive convention rather than a throw-preventer, but is kept for consistency with every other op in this file. All four paths still leak-test clean either way', () => {
  const empty = new cv.Mat();
  try {
    const hist = new cv.Mat();
    const mask = new cv.Mat();
    const srcVec = new cv.MatVector();
    srcVec.push_back(empty);
    assert.doesNotThrow(() => cv.calcHist(srcVec, [0], mask, hist, [256], [0, 256]),
      'cv.calcHist on a 0x0 Mat does not throw (all-zero histogram) -- contrast_stretch\'s own empty guard is defensive, not a throw-preventer');
    srcVec.delete(); hist.delete(); mask.delete();
    const bg = new cv.Mat();
    assert.throws(() => cv.GaussianBlur(empty, bg, new cv.Size(15, 15), 0, 0, cv.BORDER_DEFAULT),
      'cv.GaussianBlur on a 0x0 Mat must throw (that is why flatfield_correction guards first)');
    bg.delete();
    const rgbOut = new cv.Mat();
    assert.throws(() => cv.cvtColor(empty, rgbOut, cv.COLOR_GRAY2RGB),
      'cv.cvtColor(...,GRAY2RGB) on a 0x0 Mat must throw (that is why endpoints_branchpoints guards first -- it calls this exact conversion)');
    rgbOut.delete();
    const emptyContours = new cv.MatVector();
    const emptyHier = new cv.Mat();
    assert.doesNotThrow(() => cv.findContours(empty, emptyContours, emptyHier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE),
      'cv.findContours on a 0x0 Mat does not throw (0 contours) -- convex_hull\'s own empty guard is defensive, not a throw-preventer');
    assert.equal(emptyContours.size(), 0);
    emptyContours.delete(); emptyHier.delete();

    for (const [label, params] of [
      ['contrast_stretch', {}],
      ['flatfield_correction', {}],
    ]) {
      assertLeakFree(cv, `${label} on an empty 0x0 Mat`, (c) => {
        const o = OPS[label].apply(c, empty, params, {});
        o.delete();
      }, { minCreated: 1, iters: 60, warmup: 20 });
    }
    for (const op of NEW_MORPH_MODES) {
      assertLeakFree(cv, `morphology ${op} on an empty 0x0 Mat`, (c) => {
        const o = OPS.morphology.apply(c, empty, { op }, {});
        o.delete();
      }, { minCreated: 1, iters: 60, warmup: 20 });
    }
  } finally { empty.delete(); }
});

// ===================================================================
// ipA-1…ipA-2 — contrast_stretch: ค่าตัวเลขยืนยันด้วยมือ (histogram cumulative ของ exactRamp ควบคุมได้เป๊ะ)
// ===================================================================
test('ipA-1. contrast_stretch: on a 256-level ramp with exactly 100 rows per level (25600 px total), lowPct=1/highPct=1 cuts at the first level whose cumulative count reaches 1%/99% of all pixels -- low=2 (cum 300 >= 256), high=253 (cum 25400 >= 25344), matching a by-hand histogram walk', () => {
  const ramp = exactRamp(256, 100);
  try {
    const ctx = {};
    const out = OPS.contrast_stretch.apply(cv, ramp, { lowPct: 1, highPct: 1 }, ctx);
    try {
      assert.equal(ctx.info, 'ยืดช่วง [2, 253] → [0, 255]');
      // level 2 (the "low" cut) must now map to 0, level 253 ("high") must map to 255
      assert.equal(out.data[0 * 256 + 2], 0);
      assert.equal(out.data[0 * 256 + 253], 255);
      // a mid-range level must land strictly between 0 and 255 (not clamped at an edge)
      const mid = out.data[0 * 256 + 127];
      assert.ok(mid > 0 && mid < 255, `expected a strictly interior value, got ${mid}`);
    } finally { out.delete(); }
  } finally { ramp.delete(); }
});

test('ipA-2. contrast_stretch: lowPct=0/highPct=0 is the identity transform (low=0, high=255 -> LUT is i -> i exactly) on a ramp that already spans the full range; a perfectly uniform image triggers the "no range to stretch" guard and returns an unmodified clone (not a divide-by-zero crash)', () => {
  const ramp = exactRamp(256, 100);
  try {
    const ctx = {};
    const out = OPS.contrast_stretch.apply(cv, ramp, { lowPct: 0, highPct: 0 }, ctx);
    try {
      assert.equal(ctx.info, 'ยืดช่วง [0, 255] → [0, 255]');
      assert.equal(Buffer.compare(Buffer.from(out.data), Buffer.from(ramp.data)), 0, 'identity transform must reproduce the input exactly, byte for byte');
    } finally { out.delete(); }
  } finally { ramp.delete(); }

  const flat = flatGray(128, 10, 10);
  try {
    const ctx = {};
    const out = OPS.contrast_stretch.apply(cv, flat, { lowPct: 1, highPct: 1 }, ctx);
    try {
      assert.equal(ctx.info, 'ภาพมีความสว่างสม่ำเสมอ ไม่มีช่วงให้ยืด');
      assert.equal(Buffer.compare(Buffer.from(out.data), Buffer.from(flat.data)), 0, 'guard path must return an unmodified clone');
    } finally { out.delete(); }
  } finally { flat.delete(); }
});

test('ipA-3. contrast_stretch: lowPct/highPct are clamped to [0, 10] via numIn (P-13) -- an out-of-range value gives the exact same result as the nearest edge of the range, and the two edges actually differ (so the check is not vacuous)', () => {
  const ramp = noisyRamp(64, 48, 303);
  try {
    const a = OPS.contrast_stretch.apply(cv, ramp, { lowPct: -5, highPct: -5 }, {});
    const b = OPS.contrast_stretch.apply(cv, ramp, { lowPct: 0, highPct: 0 }, {});
    const c = OPS.contrast_stretch.apply(cv, ramp, { lowPct: 999, highPct: 999 }, {});
    const d = OPS.contrast_stretch.apply(cv, ramp, { lowPct: 10, highPct: 10 }, {});
    try {
      assert.equal(Buffer.compare(Buffer.from(a.data), Buffer.from(b.data)), 0, 'lowPct/highPct=-5 must clamp to the minimum 0');
      assert.equal(Buffer.compare(Buffer.from(c.data), Buffer.from(d.data)), 0, 'lowPct/highPct=999 must clamp to the maximum 10');
      assert.notEqual(Buffer.compare(Buffer.from(b.data), Buffer.from(d.data)), 0, '0 and 10 must actually give different results (otherwise the clamp check above is vacuous)');
    } finally { a.delete(); b.delete(); c.delete(); d.delete(); }
  } finally { ramp.delete(); }
});

test('ipA-3b. contrast_stretch: the cumulative-count comparison for the HIGH cut uses ">=" (not ">") -- on a 100-level ramp with exactly 16 rows per level (1600 px total), highPct=10 (the maximum the UI allows) makes the cumulative count land on EXACTLY 1440 at level 89 (= highCount), so ">=" must cut there while ">" would instead roll over to level 90. (A 256-level/100-rows-per-level ramp, as used elsewhere in this file, never lands on an exact tie anywhere inside the valid highPct<=10 range -- this fixture is sized specifically to produce one)', () => {
  const w = 100; const h = 16;
  const d = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d[y * w + x] = x; // levels 0..99, 16 rows each
  const ramp = cv.matFromArray(h, w, cv.CV_8UC1, d);
  try {
    const ctx = {};
    const out = OPS.contrast_stretch.apply(cv, ramp, { lowPct: 0, highPct: 10 }, ctx);
    try {
      assert.equal(ctx.info, 'ยืดช่วง [0, 89] → [0, 255]', 'exact-tie cumulative count (1440) must satisfy ">=" at level 89, not roll over to 90');
    } finally { out.delete(); }
  } finally { ramp.delete(); }
});

test('ipA-3c. contrast_stretch: the LOW cut uses strict ">" (like MATLAB stretchlim: find(cdf > tol_low, 1)) -- on a ramp whose darkest pixel is 50 (151 columns, values 50..200) lowPct=0/highPct=0 must stretch the dark end too ([50, 200] -> [0, 255], output min 0 and max 255). With ">=" lowCount=0 is already "reached" at the EMPTY bin 0, so low would stay 0 (info [0, 200], output min 64) while lowPct=0.5 gave [50, 200] -- an asymmetric, wrong result (qa-found MAJOR-3). The HIGH cut keeps ">=" (highPct=0 -> exactly the brightest level, ipA-3b covers the tie)', () => {
  const ramp = rampBetween(50, 200, 10);
  try {
    for (const [lowPct, highPct] of [[0, 0], [0.5, 0.5]]) {
      const ctx = {};
      const out = OPS.contrast_stretch.apply(cv, ramp, { lowPct, highPct }, ctx);
      try {
        assert.equal(ctx.info, 'ยืดช่วง [50, 200] → [0, 255]', `lowPct=${lowPct} highPct=${highPct}`);
        const mm = cv.minMaxLoc(out);
        assert.equal(mm.minVal, 0, `lowPct=${lowPct}: the darkest input level must map to 0`);
        assert.equal(mm.maxVal, 255, `highPct=${highPct}: the brightest input level must map to 255`);
        // the darkest column (value 50) -> 0 and the brightest (value 200) -> 255 exactly
        assert.equal(out.data[0], 0);
        assert.equal(out.data[ramp.cols - 1], 255);
      } finally { out.delete(); }
    }
  } finally { ramp.delete(); }
});

// ===================================================================
// ipA-4…ipA-6 — flatfield_correction
// ===================================================================
test('ipA-4. flatfield_correction: on an image with a strong left-to-right lighting gradient, the corrected output has a much smaller brightness range than the input (the lighting trend is flattened) while NOT collapsing to a single constant value (the small bright square in the middle must still stand out from its surroundings)', () => {
  const src = unevenLightGray(96, 72);
  try {
    const mmIn = cv.minMaxLoc(src);
    const ctx = {};
    const out = OPS.flatfield_correction.apply(cv, src, { blurSize: 61 }, ctx);
    try {
      const mmOut = cv.minMaxLoc(out);
      assert.ok(mmOut.maxVal - mmOut.minVal < mmIn.maxVal - mmIn.minVal,
        `corrected range (${mmOut.minVal}-${mmOut.maxVal}) should be narrower than the input range (${mmIn.minVal}-${mmIn.maxVal})`);
      assert.ok(mmOut.maxVal - mmOut.minVal > 5, 'must not collapse to a constant image (the embedded square must still be visible)');
      assert.match(ctx.info, /^ปรับแสงด้วยพื้นหลังเบลอขนาด \d+px$/);
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipA-5. flatfield_correction: an all-black image (background would be exactly 0 without the +1 epsilon, which would divide by zero) is handled without throwing and produces an all-zero output (0 / epsilon * mean(0) = 0 everywhere, not NaN/garbage)', () => {
  const black = flatGray(0, 20, 20);
  try {
    const ctx = {};
    const out = OPS.flatfield_correction.apply(cv, black, { blurSize: 15 }, ctx);
    try {
      const mm = cv.minMaxLoc(out);
      assert.equal(mm.minVal, 0);
      assert.equal(mm.maxVal, 0);
    } finally { out.delete(); }
  } finally { black.delete(); }
});

test('ipA-6. flatfield_correction: blurSize is forced odd via ensureOdd() and clamped to [15, 401] via intIn (P-13) -- an even value and an out-of-range value both give the exact same result as the correctly-rounded/clamped odd value', () => {
  const src = unevenLightGray(64, 48);
  try {
    const even = OPS.flatfield_correction.apply(cv, src, { blurSize: 60 }, {}); // must behave like 61 (ensureOdd)
    const odd61 = OPS.flatfield_correction.apply(cv, src, { blurSize: 61 }, {});
    const tooSmall = OPS.flatfield_correction.apply(cv, src, { blurSize: 1 }, {}); // must clamp to 15 (then stay odd)
    const atMin = OPS.flatfield_correction.apply(cv, src, { blurSize: 15 }, {});
    const tooBig = OPS.flatfield_correction.apply(cv, src, { blurSize: 9999 }, {}); // must clamp to 401
    const atMax = OPS.flatfield_correction.apply(cv, src, { blurSize: 401 }, {});
    try {
      assert.equal(Buffer.compare(Buffer.from(even.data), Buffer.from(odd61.data)), 0, 'blurSize=60 must round to the same odd kernel as 61');
      assert.equal(Buffer.compare(Buffer.from(tooSmall.data), Buffer.from(atMin.data)), 0, 'blurSize=1 must clamp to the minimum 15');
      assert.equal(Buffer.compare(Buffer.from(tooBig.data), Buffer.from(atMax.data)), 0, 'blurSize=9999 must clamp to the maximum 401');
      assert.notEqual(Buffer.compare(Buffer.from(atMin.data), Buffer.from(atMax.data)), 0, '15 and 401 must actually differ (otherwise the clamp checks above are vacuous)');
    } finally { even.delete(); odd61.delete(); tooSmall.delete(); atMin.delete(); tooBig.delete(); atMax.delete(); }
  } finally { src.delete(); }
});

// reference model for the ipA-6b test: the SAME flat-field maths but with the background blurred at full resolution
// (what the downsample shortcut approximates)
function flatfieldFullRes(src, k) {
  const bg = new cv.Mat(); const srcF = new cv.Mat(); const bgF = new cv.Mat(); const bgEps = new cv.Mat();
  const ratio = new cv.Mat(); const scaled = new cv.Mat(); const dst = new cv.Mat();
  const eps = new cv.Mat(src.rows, src.cols, [0, cv.CV_32FC1, cv.CV_32FC2, cv.CV_32FC3, cv.CV_32FC4][src.channels()], new cv.Scalar(1, 1, 1, 1));
  try {
    cv.GaussianBlur(src, bg, new cv.Size(k, k), 0, 0, cv.BORDER_DEFAULT);
    const mean = cv.mean(bg);
    src.convertTo(srcF, cv.CV_32F); bg.convertTo(bgF, cv.CV_32F);
    cv.add(bgF, eps, bgEps); cv.divide(srcF, bgEps, ratio);
    const meanMat = new cv.Mat(ratio.rows, ratio.cols, ratio.type(), new cv.Scalar(mean[0], mean[1], mean[2], mean[3]));
    try { cv.multiply(ratio, meanMat, scaled); } finally { meanMat.delete(); }
    scaled.convertTo(dst, src.type());
    return dst;
  } finally { bg.delete(); srcF.delete(); bgF.delete(); bgEps.delete(); ratio.delete(); scaled.delete(); eps.delete(); }
}
function diffStats(a, b) {
  assert.equal(a.data.length, b.data.length, 'sizes must match');
  let ndiff = 0; let maxdiff = 0;
  for (let i = 0; i < a.data.length; i++) { const d = Math.abs(a.data[i] - b.data[i]); if (d) { ndiff++; maxdiff = Math.max(maxdiff, d); } }
  return { ndiff, maxdiff };
}

test('ipA-6b. flatfield_correction: the downsample path (long side > 240 -- every real camera frame) is real and pinned: ctx.info states the analysed size (320x240 and 640x480 -> 240x180, 321x241 -> 240x180, 1000x7 -> 240x2), the result is NOT bit-equal to a full-resolution blur (the shortcut is really taken) yet stays within a few grey levels of it (the shortcut is faithful), the output keeps the input size/type, and an image whose long side is exactly 240 takes the full-resolution path (bit-equal to the reference, no "ภาพย่อ" text)', () => {
  const cases = [[320, 240, '240x180'], [640, 480, '240x180'], [321, 241, '240x180'], [1000, 7, '240x2']];
  for (const [w, h, analysed] of cases) {
    for (const colour of [false, true]) {
      const src = colour ? unevenLightColour(w, h) : unevenLightGray(w, h);
      const ctx = {};
      const out = OPS.flatfield_correction.apply(cv, src, { blurSize: 121 }, ctx);
      const ref = flatfieldFullRes(src, 121);
      try {
        const tag = `${w}x${h} ${colour ? 'colour' : 'gray'}`;
        assert.ok(ctx.info.includes(`ประมาณพื้นหลังที่ภาพย่อ ${analysed}px แล้วขยายกลับ`), `${tag}: info must state the analysed size ${analysed}; got "${ctx.info}"`);
        assert.equal(out.rows, h); assert.equal(out.cols, w); assert.equal(out.type(), src.type());
        const { ndiff, maxdiff } = diffStats(out, ref);
        assert.ok(ndiff > 0, `${tag}: result must differ from the full-resolution blur (otherwise the downsample branch was not taken)`);
        assert.ok(maxdiff <= 10, `${tag}: downsampled background must stay within 10 grey levels of the full-resolution one, got maxdiff=${maxdiff}`);
      } finally { src.delete(); out.delete(); ref.delete(); }
    }
  }
  // boundary: long side exactly 240 (scale === 1) -> full-resolution blur, bit-equal to the reference
  const edge = unevenLightGray(240, 180);
  try {
    const ctx = {};
    const out = OPS.flatfield_correction.apply(cv, edge, { blurSize: 121 }, ctx);
    const ref = flatfieldFullRes(edge, 121);
    try {
      assert.equal(ctx.info, 'ปรับแสงด้วยพื้นหลังเบลอขนาด 121px');
      assert.equal(diffStats(out, ref).ndiff, 0, '240x180 must use the full-resolution path exactly');
    } finally { out.delete(); ref.delete(); }
  } finally { edge.delete(); }
});

// ===================================================================
// ipA-7…ipA-9 — morphology convex_hull
// ===================================================================
test('ipA-7. morphology convex_hull: on a square with a notch cut into its right edge (a genuine concave silhouette, not an enclosed hole), the hull fills in the notch (hull area > original area) and reports a solidity < 1; the 1px and 2px specks are NOT dropped any more (every contour is hulled, drawn and counted -- 3 shapes) and, having area 0, do not change the solidity (still 0.80)', () => {
  const src = concaveShapeWithSpecks();
  try {
    const ctx = {};
    const out = OPS.morphology.apply(cv, src, { op: 'convex_hull' }, ctx);
    try {
      assert.match(ctx.info, /ห่อวัตถุ 3 ชิ้นด้วยรูปนูน/, 'three shapes must be wrapped (the square + the 1px and 2px specks)');
      assert.match(ctx.info, /solidity รวม \(ถ่วงตามพื้นที่\) 0\.80/, 'hand-verified solidity for this fixture is 0.80 (foreground area / hull area; zero-area specks do not change it)');
      // the notch (x in [30,40), y in [20,30)) must now be filled white by the hull (it was 0 in the input)
      const w = out.cols;
      let notchNowFilled = false;
      for (let y = 20; y < 30 && !notchNowFilled; y++) {
        for (let x = 30; x < 40; x++) if (out.data[y * w + x] === 255) { notchNowFilled = true; break; }
      }
      assert.ok(notchNowFilled, 'the concave notch must be covered by the convex hull');
      // the specks survive as the same isolated pixels (their hull is themselves)
      assert.equal(out.data[5 * w + 50], 255, '1px speck must be drawn (its hull is the pixel itself)');
      assert.equal(out.data[50 * w + 55], 255, '2px speck must be drawn');
      assert.equal(out.data[50 * w + 56], 255, '2px speck must be drawn');
      assert.equal(cv.countNonZero(out), cv.countNonZero(src) + 150 /* notch filled by the hull: 15x10 */, 'hull adds exactly the notch (150 px) and nothing else');
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipA-8. morphology convex_hull: an image with no foreground at all reports "no objects found" (with the threshold hint) and returns an all-black mask the same size as the input, instead of a divide-by-zero or an empty MatVector crash', () => {
  const src = flatGray(0, 30, 24);
  try {
    const ctx = {};
    const out = OPS.morphology.apply(cv, src, { op: 'convex_hull' }, ctx);
    try {
      assert.equal(ctx.info, 'ไม่พบวัตถุ (ลองทำ Threshold ให้เป็นขาว-ดำมาก่อน)');
      assert.equal(out.rows, src.rows);
      assert.equal(out.cols, src.cols);
      assert.equal(cv.countNonZero(out), 0);
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipA-9. morphology convex_hull: a non-binary grayscale input is binarized at 127 first (same convention as fill_holes/clear_border) -- values above/below the threshold are treated identically to a true 0/255 image', () => {
  const grayish = cv.matFromArray(20, 20, cv.CV_8UC1, (() => {
    const d = new Uint8Array(400).fill(60); // below 127 -> background
    for (let y = 5; y < 15; y++) for (let x = 5; x < 15; x++) d[y * 20 + x] = 200; // above 127 -> foreground
    return d;
  })());
  const trueBinary = cv.matFromArray(20, 20, cv.CV_8UC1, (() => {
    const d = new Uint8Array(400).fill(0);
    for (let y = 5; y < 15; y++) for (let x = 5; x < 15; x++) d[y * 20 + x] = 255;
    return d;
  })());
  try {
    const a = OPS.morphology.apply(cv, grayish, { op: 'convex_hull' }, {});
    const b = OPS.morphology.apply(cv, trueBinary, { op: 'convex_hull' }, {});
    try {
      assert.equal(Buffer.compare(Buffer.from(a.data), Buffer.from(b.data)), 0, 'binarizing at 127 first must make the two inputs give an identical result');
    } finally { a.delete(); b.delete(); }
  } finally { grayish.delete(); trueBinary.delete(); }
});

test('ipA-9b. morphology convex_hull: findContours is called with RETR_EXTERNAL (not RETR_LIST/RETR_TREE) -- a "donut" (solid square with an enclosed hole that does not touch the image border) must be treated as exactly 1 shape (the outer boundary only), never 2 (outer + the hole\'s own boundary, which RETR_LIST would also return)', () => {
  const src = donutShape();
  try {
    const ctx = {};
    const out = OPS.morphology.apply(cv, src, { op: 'convex_hull' }, ctx);
    try {
      assert.match(ctx.info, /ห่อวัตถุ 1 ชิ้นด้วยรูปนูน/, 'a donut has exactly 1 external contour -- RETR_LIST would wrongly report 2 (outer + hole)');
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipA-9c. morphology convex_hull: thin 1px lines are real objects (qa-found MINOR-2): a 46px horizontal line + a 20px diagonal line + a 15x20 block are wrapped as 3 shapes, every input pixel is still white in the output (a hull always contains its object -- the lines must NOT vanish: CHAIN_APPROX_SIMPLE reduces any straight 1px line to 2 contour points, so skipping contours with < 3 points deleted them), and the result has the same white-pixel count as the input (lines and a convex block are their own hulls)', () => {
  const src = linesAndBlock();
  try {
    const ctx = {};
    const out = OPS.morphology.apply(cv, src, { op: 'convex_hull' }, ctx);
    try {
      assert.match(ctx.info, /ห่อวัตถุ 3 ชิ้นด้วยรูปนูน/, 'line + diagonal + block = 3 shapes');
      assert.match(ctx.info, /solidity รวม \(ถ่วงตามพื้นที่\) 1\.00/);
      let lost = 0;
      for (let i = 0; i < src.data.length; i++) if (src.data[i] && !out.data[i]) lost++;
      assert.equal(lost, 0, 'no input pixel may disappear from the hull output');
      assert.equal(cv.countNonZero(src), 366);
      assert.equal(cv.countNonZero(out), 366, '46 + 20 + 300 = 366 white pixels, unchanged');
      const w = out.cols;
      assert.equal(out.data[5 * w + 5], 255); assert.equal(out.data[5 * w + 50], 255); // line ends
      assert.equal(out.data[12 * w + 5], 255); assert.equal(out.data[31 * w + 24], 255); // diagonal ends
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipA-9d. morphology convex_hull: solidity is the AREA-WEIGHTED total (sum of object areas / sum of hull areas), not a mean of per-object solidities -- a 50x50 solid square (solidity 1.0) next to a small 10x10 square with a 5x5 notch (solidity ~0.82) reads ~0.995, nowhere near the unweighted mean ~0.91; the info text says so ("รวม (ถ่วงตามพื้นที่)", never "เฉลี่ย")', () => {
  const w = 90; const h = 60;
  const d = new Uint8Array(w * h).fill(0);
  for (let y = 5; y < 55; y++) for (let x = 5; x < 55; x++) d[y * w + x] = 255; // area (contour) 49*49 = 2401
  for (let y = 5; y < 15; y++) for (let x = 70; x < 80; x++) { if (x >= 75 && y >= 10) continue; d[y * w + x] = 255; } // notched L
  const src = cv.matFromArray(h, w, cv.CV_8UC1, d);
  try {
    const ctx = {};
    const out = OPS.morphology.apply(cv, src, { op: 'convex_hull' }, ctx);
    try {
      assert.match(ctx.info, /ห่อวัตถุ 2 ชิ้น/);
      const m = ctx.info.match(/solidity รวม \(ถ่วงตามพื้นที่\) ([\d.]+)/);
      assert.ok(m, `info must use the area-weighted wording, got "${ctx.info}"`);
      assert.ok(Number(m[1]) > 0.95, `area-weighted solidity must be dominated by the big square (>0.95), got ${m[1]}`);
      assert.doesNotMatch(ctx.info, /เฉลี่ย/);
    } finally { out.delete(); }
  } finally { src.delete(); }
});

// ===================================================================
// ipA-10…ipA-12 — morphology endpoints_branchpoints
// ===================================================================
test('ipA-10. morphology endpoints_branchpoints: a hand-built "Y" skeleton (one branchpoint where 3 strokes meet) plus a separate straight line (2 endpoints, 0 branchpoints) reports exactly 5 endpoints and 1 branchpoint in total, and draws a green dot at an endpoint pixel + a red dot at the branchpoint pixel (BGR/RGB-safe: checks the specific channel that must be 255)', () => {
  const src = yBranchPlusLine();
  try {
    const ctx = {};
    const out = OPS.morphology.apply(cv, src, { op: 'endpoints_branchpoints' }, ctx);
    try {
      assert.equal(ctx.info, 'พบจุดปลาย 5 จุด · จุดแยก 1 จุด');
      assert.equal(out.channels(), 3, 'output must always be a 3-channel overlay');
      // the branch junction is at (20, 25) -- must be painted red: channel 0 (R) = 255, channel 1 (G) = 0
      const w = out.cols;
      const px = (x, y) => Array.from(out.data.slice((y * w + x) * 3, (y * w + x) * 3 + 3));
      assert.deepEqual(px(20, 25), [255, 0, 0], 'the branchpoint pixel must be painted red (R,G,B)=(255,0,0)');
      // the top tip of the vertical stem (20, 5) is an endpoint -- must be painted green
      assert.deepEqual(px(20, 5), [0, 255, 0], 'an endpoint pixel must be painted green (R,G,B)=(0,255,0)');
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipA-11. morphology endpoints_branchpoints: an image with no white pixels at all reports the dedicated "no line found" message (distinct from "0 endpoints / 0 branchpoints") and returns an all-black 3-channel image', () => {
  const src = flatGray(0, 20, 20);
  try {
    const ctx = {};
    const out = OPS.morphology.apply(cv, src, { op: 'endpoints_branchpoints' }, ctx);
    try {
      assert.equal(ctx.info, 'ไม่พบเส้น (ลองทำ Skeleton ก่อน)');
      assert.equal(out.channels(), 3);
      assert.equal(cv.countNonZero !== undefined, true); // sanity: countNonZero exists in this build
      let sum = 0;
      for (const b of out.data) sum += b;
      assert.equal(sum, 0, 'every byte of the output must be 0 (all black)');
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipA-12. morphology endpoints_branchpoints: white pixels that are present but have 0 or 2 neighbours (neither an endpoint, which needs exactly 1 neighbour, nor a branchpoint, which needs Crossing Number >= 3) are correctly NOT counted as either -- a lone isolated pixel (0 neighbours) and the middle of a straight run (exactly 2 neighbours) both increment neither counter', () => {
  const w = 20; const h = 20;
  const d = new Uint8Array(w * h).fill(0);
  d[10 * w + 2] = 255; // isolated pixel, 0 neighbours
  for (let x = 10; x <= 14; x++) d[5 * w + x] = 255; // straight run of 5 -> 2 endpoints (x=10,14) + 3 "middle" pixels with 2 neighbours
  const src = cv.matFromArray(h, w, cv.CV_8UC1, d);
  try {
    const ctx = {};
    const out = OPS.morphology.apply(cv, src, { op: 'endpoints_branchpoints' }, ctx);
    try {
      assert.equal(ctx.info, 'พบจุดปลาย 2 จุด · จุดแยก 0 จุด', 'only the two ends of the 5px run count as endpoints; the isolated pixel and the 3 middle pixels count as neither');
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ipA-12c. morphology endpoints_branchpoints: the Crossing-Number branchpoint rule (the hint\'s headline claim: NOT just counting neighbours) -- a plus sign has ONE real crossing at (10,10) but its 4 side pixels also have >= 3 white neighbours, so a plain "neighbours >= 3" rule would report 5 branchpoints; Crossing Number reports exactly 1 (+ 4 endpoints) and paints red ONLY the centre pixel. An X of two diagonals also gives 4 endpoints / 1 branchpoint', () => {
  const plus = plusSign();
  try {
    const ctx = {};
    const out = OPS.morphology.apply(cv, plus, { op: 'endpoints_branchpoints' }, ctx);
    try {
      assert.equal(ctx.info, 'พบจุดปลาย 4 จุด · จุดแยก 1 จุด', 'n>=3 would give 5 branchpoints here');
      const px = (x, y) => Array.from(out.data.slice((y * out.cols + x) * 3, (y * out.cols + x) * 3 + 3));
      assert.deepEqual(px(10, 10), [255, 0, 0], 'the real crossing must be red');
      // (the 4 side pixels of the crossing have 4 white neighbours but only 2 white runs on the ring -> not branchpoints;
      //  they are overpainted by the radius-3 red circle of the real one, so the count above is what proves it)
      assert.deepEqual(px(2, 10), [0, 255, 0]);
      assert.deepEqual(px(18, 10), [0, 255, 0]);
      assert.deepEqual(px(10, 2), [0, 255, 0]);
      assert.deepEqual(px(10, 18), [0, 255, 0]);
    } finally { out.delete(); }
  } finally { plus.delete(); }
  const x = crossX();
  try {
    const ctx = {};
    const out = OPS.morphology.apply(cv, x, { op: 'endpoints_branchpoints' }, ctx);
    try { assert.equal(ctx.info, 'พบจุดปลาย 4 จุด · จุดแยก 1 จุด'); } finally { out.delete(); }
  } finally { x.delete(); }
});

test('ipA-12b. morphology endpoints_branchpoints: the same "threshold at 127" convention as fill_holes/clear_border/convex_hull -- a value of 150 (above 127, below 255) must be treated as foreground (binarized to white), so a run built from value 150 reports the same endpoint/branchpoint counts as the equivalent true-binary (0/255) run', () => {
  const w = 20; const h = 20;
  const mk = (fgVal) => {
    const d = new Uint8Array(w * h).fill(0);
    for (let x = 10; x <= 14; x++) d[5 * w + x] = fgVal;
    return cv.matFromArray(h, w, cv.CV_8UC1, d);
  };
  const src150 = mk(150);
  const src255 = mk(255);
  try {
    const out150 = OPS.morphology.apply(cv, src150, { op: 'endpoints_branchpoints' }, {});
    const out255 = OPS.morphology.apply(cv, src255, { op: 'endpoints_branchpoints' }, {});
    try {
      assert.equal(Buffer.compare(Buffer.from(out150.data), Buffer.from(out255.data)), 0, 'value 150 (above the 127 threshold) must binarize identically to 255');
    } finally { out150.delete(); out255.delete(); }
  } finally { src150.delete(); src255.delete(); }
});

// ===================================================================
// ipA-13 — registry structure: OPS/FILTERS count, image_adjust mode list, morphology op list, defaultParams
// ===================================================================
// spec ip_batch2_B §6: ไม่ปักจำนวน OPS รวมอีกต่อไป (batch B–E เพิ่ม op ทุกรอบ) — เช็คว่า op ของ batch นี้มีอยู่ + ไม่ลดลงต่ำกว่า 68
test('ipA-13. (static structure) OPS contains contrast_stretch/flatfield_correction and has >= 68 entries (later batches add more -- the exact total is no longer pinned) and FILTERS is unchanged at 26 -- both new ops are reachable through FILTERS.image_adjust\'s mode select, and the two new morphology modes are reachable through FILTERS.morphology\'s "op" select; defaultParams() matches the OPS defaults exactly', () => {
  for (const id of NEW_ENHANCE_OPS) assert.ok(OPS[id], `OPS must contain "${id}"`);
  assert.ok(Object.keys(OPS).length >= 68, `OPS must have at least 68 entries (66 + the 2 from batch A), got ${Object.keys(OPS).length}`);
  assert.equal(Object.keys(FILTERS).length, 33, 'FILTERS = 26 legacy/host groups + 7 category-level groups (2026-10-07) -- both new ops still live inside existing nodes');

  const modeParam = FILTERS.image_adjust.params.find((p) => p.key === 'mode');
  const modeValues = modeParam.options.map((o) => o.value);
  for (const id of NEW_ENHANCE_OPS) assert.ok(modeValues.includes(id), `image_adjust mode select must offer "${id}"`);
  // the 4 original modes must still be present and in their original relative order (no project-file breakage)
  assert.deepEqual(modeValues.slice(0, 4), ['brightness_contrast', 'gamma', 'hist_equalize', 'clahe']);

  const opParam = OPS.morphology.params.find((p) => p.key === 'op');
  const opValues = opParam.options.map((o) => o.value);
  for (const op of NEW_MORPH_MODES) assert.ok(opValues.includes(op), `morphology "op" select must offer "${op}"`);
  assert.equal(opValues.length, 14, 'morphology must now have exactly 14 op choices (12 + 2 new)');

  const dp = defaultParams('image_adjust');
  assert.equal(dp.contrast_stretch__lowPct, 1);
  assert.equal(dp.contrast_stretch__highPct, 1);
  assert.equal(dp.flatfield_correction__blurSize, 121);
});

test('ipA-14. (static read) every numeric param of the two new ENHANCE ops is clamped in registry.js to EXACTLY the min/max/default the UI offers (P-13) -- a hand-edited project file can never push a value the UI cannot produce into OpenCV', () => {
  const source = fs.readFileSync(new URL('../src/renderer/filters/registry.js', import.meta.url), 'utf8');
  const CLAMPED = [
    ['contrast_stretch', 'lowPct', 'numIn'],
    ['contrast_stretch', 'highPct', 'numIn'],
    ['flatfield_correction', 'blurSize', 'intIn'],
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
  for (const id of NEW_ENHANCE_OPS) {
    for (const p of OPS[id].params) {
      if (p.type !== 'number') continue;
      assert.ok(CLAMPED.some(([o, k]) => o === id && k === p.key),
        `${id}.${p.key}: a numeric param with no clamp (add it to the table and clamp it in apply())`);
    }
  }
});

// ===================================================================
// ipA-15 — ui-meta.js (OP_PRIMARY card text must read clamped values via pnum, not raw properties)
// ===================================================================
test('ipA-15. ui-meta.js OP_PRIMARY renders clamped/sane card text for all four new op choices, including the morphology dropdown sub-cases', () => {
  assert.equal(OP_PRIMARY.contrast_stretch({ lowPct: 1, highPct: 1 }), 'ตัดมืด 1% · ตัดสว่าง 1%');
  // out-of-range input must render the clamped value (10), not the raw out-of-range one (999)
  assert.equal(OP_PRIMARY.contrast_stretch({ lowPct: 999, highPct: -5 }), 'ตัดมืด 10% · ตัดสว่าง 0%');
  assert.equal(OP_PRIMARY.flatfield_correction({ blurSize: 121 }), 'เบลอพื้นหลัง: 121px');
  assert.equal(OP_PRIMARY.flatfield_correction({ blurSize: 9999 }), 'เบลอพื้นหลัง: 401px');
  assert.equal(OP_PRIMARY.morphology({ op: 'convex_hull' }), 'Convex Hull (รูปนูน)');
  assert.equal(OP_PRIMARY.morphology({ op: 'endpoints_branchpoints' }), 'จุดปลาย/จุดแยก (ต่อจาก Skeleton)');
  // GROUP_ICON of the two host groups must still exist (no regression from touching this file)
  assert.ok(GROUP_ICON.image_adjust);
  assert.ok(GROUP_ICON.morphology);
});

// ===================================================================
// ipA-16 — palette-panel.js (ENHANCE section must expose both new ops as distinct tiles)
// ===================================================================
test('ipA-16. palette-panel.js: both new ENHANCE ops are listed in the ENHANCE section and resolve to the image_adjust node through OP_TO_FILTER; the morphology dropdown additions do NOT need their own palette tile (MORPHOLOGY section stays a single "morphology" entry)', () => {
  const enhance = TOOL_SECTIONS.find((s) => s.title === 'ENHANCE');
  assert.ok(enhance, 'ENHANCE section must exist');
  for (const id of NEW_ENHANCE_OPS) assert.ok(enhance.ops.includes(id), `ENHANCE section must list "${id}"`);
  for (const id of NEW_ENHANCE_OPS) assert.equal(OP_TO_FILTER[id], 'enhance_tools'); // 2026-10-07: palette ชี้กลุ่มระดับหมวด (image_adjust เป็น legacy)

  const morph = TOOL_SECTIONS.find((s) => s.title === 'MORPHOLOGY');
  assert.deepEqual(morph.ops, ['morphology'], 'MORPHOLOGY section must stay a single node (sub-modes live in the dropdown, not the palette)');
});

// ===================================================================
// ipA-17…ipA-19 — codegen (must export real Python, never app-only, correct needs/out tracking)
// ===================================================================
test('ipA-17. codegen: contrast_stretch and flatfield_correction are exported as real Python (never "WARNING: skipped" / the generic "not supported yet" line) and carry the expected cv2 calls', () => {
  const csCode = buildPythonCode([{ id: 'contrast_stretch', params: { lowPct: 1, highPct: 1 } }], { displayName: 'X' });
  assert.doesNotMatch(csCode, /WARNING: skipped/);
  assert.doesNotMatch(csCode, /\(ข้าม: ยังไม่รองรับการ gen โค้ดของ/);
  assert.match(csCode, /cv2\.calcHist\(\[img\], \[0\], None, \[256\], \[0, 256\]\)/);
  assert.match(csCode, /cv2\.LUT\(img, lut\)/);
  assert.match(csCode, /img = cv2\.cvtColor\(img, cv2\.COLOR_BGR2GRAY\)  # \(แปลงขาวดำอัตโนมัติ\)/, 'needs gray -> auto grayscale line');

  const ffCode = buildPythonCode([{ id: 'flatfield_correction', params: { blurSize: 61 } }], { displayName: 'X' });
  assert.doesNotMatch(ffCode, /WARNING: skipped/);
  assert.doesNotMatch(ffCode, /\(ข้าม: ยังไม่รองรับการ gen โค้ดของ/);
  assert.match(ffCode, /cv2\.GaussianBlur\(img, \(k, k\), 0\)/);
  assert.match(ffCode, /k = 61/);
  assert.match(ffCode, /cv2\.mean\(bg\)/);
  assert.match(ffCode, /cv2\.convertScaleAbs\(scaled\)/);
  assert.doesNotMatch(ffCode, /COLOR_BGR2GRAY/, 'flatfield_correction needs "any" -> no forced grayscale conversion');
});

test('ipA-18. codegen: morphology convex_hull / endpoints_branchpoints are exported as real Python with the correct output colour-tracking (endpoints_branchpoints turns the image into colour, so a gray-needing step right after it must insert a second auto-gray line; convex_hull stays gray)', () => {
  const chCode = buildPythonCode([{ id: 'morphology', params: { op: 'convex_hull' } }], { displayName: 'X' });
  assert.doesNotMatch(chCode, /WARNING: skipped/);
  assert.match(chCode, /cv2\.findContours\(bin_img, cv2\.RETR_EXTERNAL, cv2\.CHAIN_APPROX_SIMPLE\)/);
  assert.match(chCode, /cv2\.convexHull\(c\)/);
  assert.doesNotMatch(chCode, /len\(c\) < 3/, 'every contour is hulled -- thin 1px lines have only 2 contour points and must not be skipped');
  assert.match(chCode, /area-weighted solidity/);

  const ebCode = buildPythonCode([{ id: 'morphology', params: { op: 'endpoints_branchpoints' } }], { displayName: 'X' });
  assert.match(ebCode, /cv2\.cvtColor\(bin_img, cv2\.COLOR_GRAY2BGR\)/);
  assert.match(ebCode, /\(0, 255, 0\), -1\)  # เขียว \(BGR\) = จุดปลาย/);
  assert.match(ebCode, /\(0, 0, 255\), -1\)  # แดง \(BGR\) = จุดแยก/);

  // after endpoints_branchpoints (colour output) a gray-needing step must re-convert
  const chained = buildPythonCode([
    { id: 'morphology', params: { op: 'endpoints_branchpoints' } },
    { id: 'binary_threshold', params: { thresh: 127 } },
  ]);
  assert.equal((chained.match(/COLOR_BGR2GRAY/g) || []).length, 2, 'colour output after endpoints_branchpoints must force a second auto-gray conversion');
  // after convex_hull (still gray) a gray-needing step must NOT re-convert a second time
  const chainedGray = buildPythonCode([
    { id: 'morphology', params: { op: 'convex_hull' } },
    { id: 'binary_threshold', params: { thresh: 127 } },
  ]);
  assert.equal((chainedGray.match(/COLOR_BGR2GRAY/g) || []).length, 1, 'convex_hull stays gray -> only the first auto-gray conversion is needed');
});

test('ipA-19. codegen: out-of-range values are clamped in the exported code exactly like registry.js (lowPct/highPct to [0,10], blurSize to an odd value in [15,401]) -- the app and the exported script can never silently diverge', () => {
  const csClamped = buildPythonCode([{ id: 'contrast_stretch', params: { lowPct: 999, highPct: -5 } }]);
  assert.match(csClamped, /low_count = total \* 10 \/ 100/, 'lowPct must be clamped to 10 in the exported code');
  assert.match(csClamped, /high_count = total \* \(100 - 0\) \/ 100/, 'highPct must be clamped to 0 in the exported code');

  const ffClamped = buildPythonCode([{ id: 'flatfield_correction', params: { blurSize: 9999 } }]);
  assert.match(ffClamped, /k = 401/, 'blurSize must be clamped to the maximum 401 (already odd)');
  const ffEven = buildPythonCode([{ id: 'flatfield_correction', params: { blurSize: 60 } }]);
  assert.match(ffEven, /k = 61/, 'blurSize=60 must be rounded up to the nearest odd value (61), matching ensureOdd() in registry.js');
});

// ===================================================================
// ipA-20 — codegen parity: the exported Python really produces the same pixels as the app (cv2 + numpy)
// ===================================================================
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'ip-batch2-A-2026-10-01-'));
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


// swap R<->B of every 3-byte pixel. endpoints_branchpoints always outputs a colour overlay -- the app draws in RGB
// (cv.cvtColor(...,GRAY2RGB) + cv.Scalar(255,0,0) for red) while the exported Python follows this whole file's BGR
// convention (cv2.cvtColor(...,GRAY2BGR) + (0,0,255) for red) -- documented at the top of codegen.js and already the
// reason ipe-28b in the previous batch only parity-tests region_growing's "mask" (grayscale) view. Swapping the R/B
// byte before comparing proves the two are the *same* image modulo that known channel-order convention -- not a silent
// pass (R!=B at every marker pixel, so an unswapped compare fails loudly).
function swapRB3(buf) {
  const out = Buffer.from(buf);
  for (let i = 0; i < out.length; i += 3) { const t = out[i]; out[i] = out[i + 2]; out[i + 2] = t; }
  return out;
}
// run the exported Python for `opId`/`params` on `src` and compare it byte-for-byte with OPS[opId].apply.
//  asGray3: "needs gray" ops are driven through a 3-channel input whose channels are all equal so the auto-inserted
//  `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` reproduces the 1-channel fixture exactly (same convention as ipe-28b)
//  expectStdout: regexp the Python `print` line must match (pins the *decision* both sides took, not only pixels)
//  expectInfo: regexp the app's ctx.info must match
function assertParity(opId, params, src, { asGray3 = false, swapRB = false, expectStdout = null, expectInfo = null } = {}) {
  const input = asGray3 ? grayAsChannels(src, 3) : src;
  const label = `${opId} ${JSON.stringify(params)} ${src.cols}x${src.rows}x${src.channels()}`;
  try {
    const code = buildPythonCode([{ id: opId, params }]);
    const { bytes: rawBytes, stdout } = runGeneratedPython(code, matToRaw(input));
    const bytes = swapRB ? swapRB3(rawBytes) : rawBytes;
    const ctx = {};
    const expected = OPS[opId].apply(cv, src, params, ctx);
    try {
      const js = Buffer.from(expected.data);
      assert.equal(bytes.length, js.length, `${label}: output size`);
      let ndiff = 0; let maxdiff = 0;
      for (let i = 0; i < js.length; i++) {
        const d = Math.abs(js[i] - bytes[i]);
        if (d) { ndiff++; maxdiff = Math.max(maxdiff, d); }
      }
      assert.equal(ndiff, 0, `${label}: ndiff=${ndiff} maxdiff=${maxdiff} of ${js.length}${swapRB ? ' (after R/B channel swap)' : ''}; python said "${stdout.trim()}", app said "${ctx.info}"`);
      if (expectStdout) assert.match(stdout, expectStdout, `${label}: python stdout`);
      if (expectInfo) assert.match(ctx.info, expectInfo, `${label}: app ctx.info`);
    } finally { expected.delete(); }
  } finally { if (asGray3) input.delete(); }
}

test('ipA-20. codegen parity: the exported Python really produces the same pixels as the app for all four new op choices (cv2 5.0.0 + numpy, run for real -- not just string matching), including the plus/X crossings (Crossing-Number rule), a ramp whose darkest pixel is 50 (strict-">" low cut), and thin 1px lines (every contour hulled)', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  const gray = noisyRamp(83, 59, 909);
  const uneven = unevenLightGray(97, 71);
  const unevenColour = unevenLightColour(97, 71); // a genuine 3-channel case for flatfield_correction (needs:"any")
  const concave = concaveShapeWithSpecks();
  const yBranch = yBranchPlusLine();
  const plus = plusSign();
  const x = crossX();
  const lines = linesAndBlock();
  const ramp50 = rampBetween(50, 200, 10);
  try {
    assertParity('contrast_stretch', { lowPct: 1, highPct: 1 }, gray, { asGray3: true });
    assertParity('contrast_stretch', { lowPct: 0, highPct: 10 }, gray, { asGray3: true });
    assertParity('contrast_stretch', { lowPct: 0, highPct: 0 }, ramp50, { asGray3: true, expectStdout: /stretched \[50, 200\] -> \[0, 255\]/, expectInfo: /ยืดช่วง \[50, 200\]/ });
    assertParity('morphology', { op: 'convex_hull' }, concave, { asGray3: true, expectStdout: /wrapped 3 object\(s\), area-weighted solidity 0\.80/ });
    assertParity('morphology', { op: 'convex_hull' }, lines, { asGray3: true, expectStdout: /wrapped 3 object\(s\), area-weighted solidity 1\.00/ });
    assertParity('morphology', { op: 'endpoints_branchpoints' }, yBranch, { asGray3: true, swapRB: true });
    assertParity('morphology', { op: 'endpoints_branchpoints' }, plus, { asGray3: true, swapRB: true, expectStdout: /endpoints=4 branchpoints=1/, expectInfo: /จุดแยก 1 จุด/ });
    assertParity('morphology', { op: 'endpoints_branchpoints' }, x, { asGray3: true, swapRB: true, expectStdout: /endpoints=4 branchpoints=1/ });
    // flatfield_correction (images <= 240px: full-resolution path): the real 3-channel path directly, then gray
    for (const blurSize of [41, 61, 121]) assertParity('flatfield_correction', { blurSize }, unevenColour);
    for (const blurSize of [41, 61]) assertParity('flatfield_correction', { blurSize }, uneven);
  } finally {
    gray.delete(); uneven.delete(); unevenColour.delete(); concave.delete(); yBranch.delete();
    plus.delete(); x.delete(); lines.delete(); ramp50.delete();
  }
});

test('ipA-20b. codegen parity of the flatfield_correction DOWNSAMPLE path (long side > 240 -- every real camera frame): resize INTER_AREA -> GaussianBlur(bgK=ensureOdd(max(3,round(k*scale)))) -> resize INTER_LINEAR must match cv2 byte-for-byte at several sizes (321x241 colour, 640x480 gray + colour, thin 7x1000 / 1000x7) and kernel sizes (15 / 121 / 401), so the interpolation flags, the 240 limit and the scaled-kernel formula are all pinned', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  const cases = [
    [321, 241, true, 121], [321, 241, false, 15], [640, 480, false, 401], [640, 480, true, 121], [7, 1000, false, 121], [1000, 7, false, 15],
  ];
  for (const [w, h, colour, blurSize] of cases) {
    const src = colour ? unevenLightColour(w, h) : unevenLightGray(w, h);
    try {
      assertParity('flatfield_correction', { blurSize }, src, { expectInfo: /ประมาณพื้นหลังที่ภาพย่อ \d+x\d+px/ });
    } finally { src.delete(); }
  }
});

test('ipA-20c. codegen parity (MINOR-1): the exported Python accumulates the histogram in float64 exactly like JS -- on a 399x601 image whose first 238600 pixels are 10 and the rest 200 with lowPct=0/highPct=0.5, highCount = 238600.005; float32 (what NumPy 2 gives `0.0 + np.float32`) rounds it to 238600.0 and wrongly stops the high cut at level 10 ([10, 10] -> "flat", ndiff 238600) while float64 correctly continues to 200 -- both sides must say [10, 200] and agree on every pixel', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  const trap = float32TrapImage();
  try {
    assertParity('contrast_stretch', { lowPct: 0, highPct: 0.5 }, trap, {
      asGray3: true, expectStdout: /stretched \[10, 200\] -> \[0, 255\]/, expectInfo: /^ยืดช่วง \[10, 200\] → \[0, 255\]$/,
    });
  } finally { trap.delete(); }
});

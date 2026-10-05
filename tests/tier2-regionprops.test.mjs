// Tier 2, 18c — regionprops: วัดคุณสมบัติต่อวัตถุ (ตัวแรกที่ใช้ ctx.values แบบ "table" แถวต่อวัตถุ)
// ครอบคลุมรายการ acceptance tests ใน spec §C.3 + รายการ watch-out ที่ qa ระบุไว้ทีละข้อ (คอมเมนต์
// regionprops-N บอกไว้ตรงแต่ละเทสต์)
//
// นโยบาย leak-test (P-14 — รายละเอียดใน tests/_leak-helpers.mjs): (1) address-fingerprint ใช้ได้เฉพาะเทสต์แรกสุดของไฟล์
// (regionprops-24a อยู่บนสุด ก่อนเทสต์หนักอย่าง 23) (2) ด่านหลักคือ wrapper-balance ทุกเส้นทาง (14a-d, 24b, 24c)
// (3) leak ฝั่ง native ตรวจด้วย heap census (assertNoHeapGrowth) ไม่ใช่การเทียบที่อยู่
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { OPS, FILTERS, regionScan, compareRegionCandidates } from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS } from '../src/renderer/ui/palette-panel.js';
import { OP_PRIMARY } from '../src/renderer/filters/ui-meta.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import { fingerprintLeakProbe, assertWrapperBalance, assertNoHeapGrowth, makeThrowingCv } from './_leak-helpers.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cvtColor === 'function'; }
async function waitForCv() {
  if (isReady(cv)) return;
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}
await waitForCv();

// ===================================================================
// ภาพทดสอบสังเคราะห์
// ===================================================================
function blank(w, h) { return new cv.Mat.zeros(h, w, cv.CV_8UC1); }

function withCircle(w, h, cx, cy, r) {
  const m = blank(w, h);
  cv.circle(m, new cv.Point(cx, cy), r, new cv.Scalar(255), -1);
  return m;
}
function withRect(w, h, x, y, rw, rh) {
  const m = blank(w, h);
  cv.rectangle(m, new cv.Point(x, y), new cv.Point(x + rw, y + rh), new cv.Scalar(255), -1);
  return m;
}
// วงรียาวรี ปลูกมุมเอียงที่รู้ค่าแน่นอนด้วย cv.ellipse (major axis ยาวกว่า minor axis มากพอให้ fitEllipse
// วัดมุมได้เสถียร ไม่ใช่เกือบกลม) — angleDeg ตรงกับธรรมเนียมของ cv.ellipse เอง (มุมของแกน "major" จาก
// แกน x บวก, วัดตามเข็มนาฬิกาเพราะแกน y ของภาพชี้ลง)
function withEllipse(w, h, cx, cy, majorHalf, minorHalf, angleDeg) {
  const m = blank(w, h);
  cv.ellipse(m, new cv.Point(cx, cy), new cv.Size(majorHalf, minorHalf), angleDeg, 0, 360, new cv.Scalar(255), -1);
  return m;
}

// ===================================================================
// regionprops-24a — address-fingerprint (เทสต์แรกสุดของไฟล์โดยตั้งใจ: process สะอาด)
// ===================================================================
// fingerprint แบบเทียบที่อยู่ "ใช้ได้เฉพาะเป็นเทสต์แรกของไฟล์" — ย้ายมาไว้บนสุดก่อนเทสต์หนักทุกตัว (เช่น regionprops-23 ภาพ
// 1080p noise) เพราะหลัง workload หนัก free-list ของ allocator หมุนที่อยู่ ทำให้ fingerprint ต่างกันทั้งที่ไม่มี leak
// (พิสูจน์ซ้ำได้ใน tests/leak-gates.test.mjs และ WORKLOG P-14) ชื่อคงเป็น 24a เพื่อให้ชื่อเทสต์เดิมคงที่ (ธรรมเนียม X-0 ของไฟล์อื่น)
test('regionprops-24a. leak probe: cap path (many objects, truncation fires)', () => {
  const src = manyObjectsImage();
  fingerprintLeakProbe(cv, 'regionprops cap path', () => {
    const d = OPS.regionprops.apply(cv, src, { minArea: 0, maxObjects: 10 }, {});
    d.delete();
  }, { iters: 180, warmup: 200 });
  src.delete();
});

// ===================================================================
// regionprops-1. wiring: อยู่ใน OPS, shape_analysis group, palette section, OP_PRIMARY
// ===================================================================
test('regionprops-1. wired into OPS, shape_analysis mergeGroup, ANALYSIS & DETECTION palette section, and OP_PRIMARY', () => {
  assert.ok(OPS.regionprops, 'OPS.regionprops must exist');
  assert.equal(OPS.regionprops.needs, 'gray');
  const modeOptions = FILTERS.shape_analysis.params.find((p) => p.key === 'mode').options.map((o) => o.value);
  assert.ok(modeOptions.includes('regionprops'), 'shape_analysis mergeGroup must list regionprops as a mode option');
  const section = TOOL_SECTIONS.find((s) => s.title === 'ANALYSIS & DETECTION');
  assert.ok(section && section.ops.includes('regionprops'), 'palette-panel.js ANALYSIS & DETECTION section must include regionprops');
  assert.equal(typeof OP_PRIMARY.regionprops, 'function', 'ui-meta.js OP_PRIMARY must have a regionprops entry');
  assert.match(OP_PRIMARY.regionprops({ minArea: 50 }), /50/);
});

// ===================================================================
// regionprops-2. known-shape ground truth: circle of known radius
// ===================================================================
test('regionprops-2. synthetic circle r=50: area~pi*r^2, perimeter~2*pi*r, circularity high (within empirically-measured pixel-discretization tolerance)', () => {
  const r = 50;
  const src = withCircle(160, 160, 80, 80, r);
  const ctx = {};
  const dst = OPS.regionprops.apply(cv, src, { minArea: 50 }, ctx);
  assert.equal(ctx.values.rows.length, 1);
  const [, area, cx, cy, perimeter, circularity, , , , , orientation] = ctx.values.rows[0];
  const expectedArea = Math.PI * r * r;
  const expectedPer = 2 * Math.PI * r;
  // ยืนยันเชิงประจักษ์ก่อนเขียนเทสต์นี้ (scratch script, ดู worklog): cv.contourArea/cv.arcLength บนเส้นรอบรูป
  // จากพิกเซล (staircase) เบี่ยงจากค่าวงกลมเรียบจริงประมาณ 2-5% เป็นปกติ (ไม่ใช่บั๊ก) — ใช้ tolerance กว้าง
  // พอครอบคลุม แต่แคบพอจะจับความผิดพลาดจริงได้
  assert.ok(Math.abs(area - expectedArea) / expectedArea < 0.06, `area ${area} vs expected ~${expectedArea}`);
  assert.ok(Math.abs(perimeter - expectedPer) / expectedPer < 0.10, `perimeter ${perimeter} vs expected ~${expectedPer}`);
  assert.ok(circularity > 0.85 && circularity <= 1.0, `circularity ${circularity} should be high (near-circular) and clamped <=1`);
  assert.ok(Math.abs(cx - 80) <= 1 && Math.abs(cy - 80) <= 1, `centroid (${cx},${cy}) should be near (80,80)`);
  dst.delete();
  src.delete();
});

// ===================================================================
// regionprops-3. synthetic rectangle -> circularity noticeably < 1
// ===================================================================
test('regionprops-3. synthetic rectangle: circularity noticeably below 1 (unlike a circle)', () => {
  const src = withRect(200, 200, 50, 60, 90, 40); // ยาวรี ไม่ใช่สี่เหลี่ยมจัตุรัส
  const ctx = {};
  const dst = OPS.regionprops.apply(cv, src, { minArea: 50 }, ctx);
  assert.equal(ctx.values.rows.length, 1);
  const [, area, , , , circularity, bx, by, bw, bh] = ctx.values.rows[0];
  // หมายเหตุ (ยืนยันด้วย scratch probe): cv.rectangle(p1,p2,...,-1) วาดรวมทั้งสองมุม (inclusive) จึง
  // ได้ bounding box กว้าง/สูงมากกว่าค่าที่ส่งเข้า withRect ไป 1px ต่อด้าน (91x41 ไม่ใช่ 90x40 เป๊ะ) แต่
  // cv.contourArea (ที่ trace ขอบพิกเซล ไม่ใช่นับพิกเซลตรง ๆ) กลับให้ 90*40=3600 พอดี (ธรรมชาติของ
  // contourArea ที่วัด area ของ "เส้นรอบรูป" ซึ่งเยื้องจากขอบพิกเซลจริงไปประมาณครึ่งพิกเซลรอบด้าน)
  assert.equal(area, 90 * 40);
  assert.ok(circularity < 0.85, `rectangle circularity ${circularity} should be noticeably < 1`);
  assert.equal(bx, 50); assert.equal(by, 60); assert.equal(bw, 91); assert.equal(bh, 41);
  dst.delete();
  src.delete();
});

// ===================================================================
// regionprops-4. elongated ellipse, known planted angle -> orientation_deg matches within a few degrees
// (มุม convention ยืนยันเชิงประจักษ์แล้วก่อนเขียนโค้ด production: rot.angle เป็นมุมแกน "width" ที่ตั้งฉาก
// กับแกนหลัก จึงแปลงกลับด้วย (rot.angle+90)%180 — ดู comment เต็มใน registry.js — เทสต์นี้ยืนยันซ้ำอีกชั้น
// ผ่าน public API จริง (OPS.regionprops.apply) ไม่ใช่แค่ scratch script ที่ลบไปแล้ว)
// ===================================================================
test('regionprops-4. elongated ellipse with a known planted angle -> orientation_deg matches within a few degrees, for several angles', () => {
  for (const angleDeg of [0, 30, 45, 60, 90, 120, 150]) {
    const src = withEllipse(220, 220, 110, 110, 70, 20, angleDeg);
    const ctx = {};
    const dst = OPS.regionprops.apply(cv, src, { minArea: 50 }, ctx);
    assert.equal(ctx.values.rows.length, 1, `angle ${angleDeg}: expected exactly 1 object`);
    const orientation = ctx.values.rows[0][10];
    assert.notEqual(orientation, null, `angle ${angleDeg}: orientation_deg should not be null for a >=5-point elongated ellipse contour`);
    // ระยะห่างเชิงมุมแบบวนรอบ (0 == 180 ในความหมายของแกน ไม่มีทิศทาง) กันเคส wrap-around เช่น planted=0 vs measured=179.5
    let diff = Math.abs(orientation - angleDeg) % 180;
    if (diff > 90) diff = 180 - diff;
    assert.ok(diff <= 5, `angle ${angleDeg}: measured orientation_deg=${orientation}, diff=${diff} > 5deg tolerance`);
    dst.delete();
    src.delete();
  }
});

// ===================================================================
// regionprops-5. minArea filtering correctness: below excluded, at/above included
// ===================================================================
test('regionprops-5. minArea filtering: object below threshold excluded, object at/above threshold included', () => {
  const w = 200, h = 200;
  const small = withCircle(w, h, 40, 40, 8); // area ~ pi*64 ~ 201 (rasterized ~174, see earlier probe)
  const big = withCircle(w, h, 150, 150, 30); // area ~ 2736 rasterized
  const merged = new cv.Mat();
  cv.bitwise_or(small, big, merged);

  // เลือก minArea ให้อยู่ระหว่างพื้นที่จริงสองก้อน (วัดจากการรันจริงด้านล่างเพื่อไม่เดา threshold)
  const ctxProbe = {};
  const dstProbe = OPS.regionprops.apply(cv, merged, { minArea: 0 }, ctxProbe);
  assert.equal(ctxProbe.values.rows.length, 2, 'sanity: both objects present at minArea=0');
  const areas = ctxProbe.values.rows.map((r) => r[1]).sort((a, b) => a - b);
  const [smallArea, bigArea] = areas;
  assert.ok(smallArea < bigArea);
  dstProbe.delete();

  const cutoff = Math.round((smallArea + bigArea) / 2);
  const ctx = {};
  const dst = OPS.regionprops.apply(cv, merged, { minArea: cutoff }, ctx);
  assert.equal(ctx.values.rows.length, 1, 'only the bigger object should survive the cutoff');
  assert.ok(ctx.values.rows[0][1] >= cutoff);
  dst.delete();

  // "at/above threshold included": minArea ตรงเป๊ะกับพื้นที่จริงของ big -> ยังต้องรวมอยู่ (>= ไม่ใช่ >)
  const ctxAt = {};
  const dstAt = OPS.regionprops.apply(cv, merged, { minArea: bigArea }, ctxAt);
  assert.equal(ctxAt.values.rows.length, 1, 'object exactly at minArea threshold must still be included (>=, not >)');
  dstAt.delete();

  merged.delete(); small.delete(); big.delete();
});

// ===================================================================
// regionprops-6. ctx.values table shape validation
// ===================================================================
test('regionprops-6. ctx.values table shape: column count matches every row length, for multiple objects', () => {
  const w = 300, h = 300;
  const m1 = withCircle(w, h, 50, 50, 20);
  const m2 = withRect(w, h, 150, 40, 60, 30);
  const m3 = withCircle(w, h, 100, 220, 35);
  const merged = new cv.Mat();
  cv.bitwise_or(m1, m2, merged);
  cv.bitwise_or(merged, m3, merged);
  const ctx = {};
  const dst = OPS.regionprops.apply(cv, merged, { minArea: 50 }, ctx);
  assert.equal(ctx.values.kind, 'table');
  assert.equal(ctx.values.rows.length, 3);
  for (const row of ctx.values.rows) {
    assert.equal(row.length, ctx.values.columns.length, `row ${JSON.stringify(row)} length must match columns.length`);
  }
  dst.delete(); merged.delete(); m1.delete(); m2.delete(); m3.delete();
});

// ===================================================================
// regionprops-7. zero-object case handled gracefully
// ===================================================================
test('regionprops-7. blank image (zero objects): empty rows array, sensible ctx.info, no throw', () => {
  const src = blank(100, 100);
  const ctx = {};
  let dst;
  assert.doesNotThrow(() => { dst = OPS.regionprops.apply(cv, src, { minArea: 50 }, ctx); });
  assert.equal(ctx.values.kind, 'table');
  assert.deepEqual(ctx.values.rows, []);
  assert.equal(typeof ctx.info, 'string');
  assert.ok(ctx.info.length > 0);
  assert.doesNotMatch(ctx.info, /NaN|undefined/);
  dst.delete(); src.delete();
});

// ===================================================================
// regionprops-8. non-binary (grayscale, not pre-thresholded) input handled via auto-binarize
// ===================================================================
test('regionprops-8. non-binary grayscale input: auto-binarizes at 127 (a photo-like gradient still yields a sensible, non-throwing result)', () => {
  const w = 200, h = 100;
  const grad = new cv.Mat(h, w, cv.CV_8UC1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) grad.ucharPtr(y, x)[0] = Math.min(255, Math.round((x / w) * 255));
  const ctx = {};
  let dst;
  assert.doesNotThrow(() => { dst = OPS.regionprops.apply(cv, grad, { minArea: 50 }, ctx); });
  // เกิน 127 คือ ~roughly x > w*127/255 -> ควรได้วัตถุ 1 ก้อนเป็นแถบทางขวา (ไม่ throw, ไม่ empty เพราะ
  // แถบขวาใหญ่กว่า minArea แน่นอน)
  assert.equal(ctx.values.rows.length, 1);
  const [, , , , , , bx] = ctx.values.rows[0];
  const expectedX = Math.floor((127 / 255) * w) - 2; // เผื่อ threshold boundary คลาดเคลื่อน 1-2px
  assert.ok(bx >= expectedX - 3 && bx <= w, `bbox_x=${bx} should start near the 127-threshold boundary (~${expectedX})`);
  dst.delete(); grad.delete();
});

// ===================================================================
// regionprops-9. contour with fewer than 5 points -> orientation_deg is null, no crash
// ===================================================================
test('regionprops-9. axis-aligned rectangle (4-point contour via CHAIN_APPROX_SIMPLE) -> orientation_deg is null, no crash', () => {
  const src = withRect(150, 150, 30, 30, 60, 50);
  const ctx = {};
  const dst = OPS.regionprops.apply(cv, src, { minArea: 50 }, ctx);
  assert.equal(ctx.values.rows.length, 1);
  assert.equal(ctx.values.rows[0][10], null, 'orientation_deg must be null for a <5-point contour, not a crash/garbage value');
  dst.delete(); src.delete();
});

// ===================================================================
// regionprops-10. near-zero-area / 1-pixel object -> no crash, sensible (not NaN/Infinity) values
// ===================================================================
test('regionprops-10. 1-pixel object with minArea=0: no crash on centroid/circularity division, finite non-NaN values', () => {
  const src = blank(60, 60);
  src.ucharPtr(30, 30)[0] = 255; // 1 พิกเซลเดียว -> contourArea=0, arcLength=0, m00=0 (ยืนยันแล้วด้วย scratch probe)
  const ctx = {};
  const dst = OPS.regionprops.apply(cv, src, { minArea: 0 }, ctx); // minArea:0 ต้องไม่ตกเป็น falsy fallback (finiteOr rule)
  assert.equal(ctx.values.rows.length, 1, 'minArea:0 must legitimately mean "no filtering", including the degenerate area=0 object');
  const [, area, cx, cy, perimeter, circularity, , , , , orientation] = ctx.values.rows[0];
  for (const v of [area, cx, cy, perimeter, circularity]) {
    assert.ok(Number.isFinite(v), `expected finite value, got ${v}`);
  }
  assert.equal(area, 0);
  assert.equal(perimeter, 0);
  assert.equal(circularity, 0, 'perimeter=0 must guard the division, not produce NaN/Infinity');
  assert.ok(Math.abs(cx - 30) <= 1 && Math.abs(cy - 30) <= 1, 'centroid falls back to bbox center when m00=0');
  assert.equal(orientation, null, '1-point contour is far below the 5-point fitEllipse minimum');
  dst.delete(); src.delete();
});

// ===================================================================
// regionprops-11. circularity clamp: documented finding + defensive-clamp correctness
// ===================================================================
// สำคัญ (อ่านก่อนแก้เทสต์นี้): พยายามหา "รูปทรงจริง" ที่ทำให้ 4*pi*A/P^2 > 1.0 ก่อนไม่ ก่อนเขียนโค้ด
// production ด้วย scratch script (ดู worklog) — ทดสอบวงกลม rasterized r=1..1000, regular polygon เทียม
// (จุดลอยตัวแม่นยำสูง ไม่ rasterize) ถึง 5000 เหลี่ยม, เศษเสี้ยว/รูปเสื่อมสภาพหลายแบบ, พิกัดไกลจากจุดกำเนิด
// (กัน floating-point cancellation) — ไม่พบสักกรณีเดียวที่เกิน 1.0 เพราะ area=cv.contourArea(c) กับ
// perimeter=cv.arcLength(c,true) มาจาก "polygon เดียวกัน" เป๊ะ (จุดเดียวกันทุกจุด) ซึ่ง isoperimetric
// inequality (4*pi*A <= P^2) เป็นจริงเสมอทางคณิตศาสตร์สำหรับ simple closed polygon ใด ๆ ที่ area/perimeter
// สอดคล้องกัน (ไม่ใช่แค่ "ปกติมักจะ" แต่ "พิสูจน์ได้ว่าเป็นจริงเสมอ") — สถานการณ์ที่สเปกอธิบาย (circularity
// เกิน 1.0 จาก pixel discretization) มักเกิดจากการ "ผสม" ตัวชี้วัดคนละแบบ (เช่น area จากการนับพิกเซลตรง ๆ
// แต่ perimeter จากเส้นรอบรูป) ซึ่งไม่ตรงกับการออกแบบในไฟล์นี้ (ดู comment เหตุผลข้อ 2 ที่ registry.js ว่าทำไม
// เลือก contourArea ให้สอดคล้องกับ arcLength ของ contour เดียวกัน) — คง clamp ไว้ตามที่สเปกสั่งชัดเจน (ฟรี,
// กัน edge case ในอนาคตถ้ามีคนเปลี่ยนนิยาม area/perimeter ภายหลัง) แต่ไม่สามารถสร้างเทสต์แบบ "revert clamp
// แล้วภาพจริงให้ผลเกิน 1.0" ได้อย่างสุจริตด้วยนิยามปัจจุบัน — รายงานให้ qa ทราบแล้วในรายงานส่งงาน
test('regionprops-11a. circularity is mathematically bounded in [0,1] across many near-circular shapes (isoperimetric invariant, documents why the clamp is unreachable-but-defensive with this area/perimeter pairing)', () => {
  for (const r of [3, 5, 8, 12, 20, 30, 50, 80, 120]) {
    const size = r * 2 + 30;
    const src = withCircle(size, size, size / 2, size / 2, r);
    const ctx = {};
    const dst = OPS.regionprops.apply(cv, src, { minArea: 0 }, ctx);
    if (ctx.values.rows.length > 0) {
      const circularity = ctx.values.rows[0][5];
      assert.ok(circularity >= 0 && circularity <= 1, `r=${r}: circularity ${circularity} out of [0,1]`);
    }
    dst.delete(); src.delete();
  }
});
test('regionprops-11b. clamp formula itself: Math.max(0, Math.min(1, x)) — the exact expression used in registry.js — correctly clamps out-of-range inputs', () => {
  // เทสต์นี้ตรวจ "สูตร" clamp ที่ใช้จริงในโค้ด (คัดลอก expression มาตรง ๆ) เนื่องจากไม่สามารถหาเคสภาพจริงที่
  // ทำให้ over 1.0 ได้อย่างสุจริต (ดู comment ด้านบน) — เป็นเทสต์เสริมยืนยันความถูกต้องของสูตร ไม่ใช่การยืนยัน
  // ว่า apply() จะ hit เคส >1 จริงในทางปฏิบัติ
  const clamp = (x) => Math.max(0, Math.min(1, x));
  assert.equal(clamp(1.37), 1);
  assert.equal(clamp(-0.02), 0);
  assert.equal(clamp(0.5), 0.5);
});

// ===================================================================
// regionprops-12. deterministic row ordering, including a tie-break on the primary sort key
// ===================================================================
test('regionprops-12. deterministic row ordering: area DESC, then bbox y ASC, then bbox x ASC tie-break', () => {
  const w = 300, h = 300;
  // 3 วัตถุพื้นที่ต่างกันชัดเจน + 2 วัตถุพื้นที่เท่ากันเป๊ะ (สี่เหลี่ยมขนาดเดียวกัน) ที่ตำแหน่งต่างกัน
  // เพื่อบังคับให้ tie-break ทำงาน (ไม่ใช่แค่ sort ตาม area เพียงอย่างเดียว)
  const big = withCircle(w, h, 250, 250, 40); // area ใหญ่สุด
  const tieA = withRect(w, h, 10, 100, 30, 30); // area เท่ากับ tieB เป๊ะ (900px) แต่ y น้อยกว่า
  const tieB = withRect(w, h, 200, 10, 30, 30); // area เท่ากับ tieA เป๊ะ แต่ y มากกว่า -> ต้องมาหลัง tieA
  const small = withCircle(w, h, 50, 250, 10); // area เล็กสุด
  let merged = new cv.Mat();
  cv.bitwise_or(big, tieA, merged);
  cv.bitwise_or(merged, tieB, merged);
  cv.bitwise_or(merged, small, merged);

  const ctx = {};
  const dst = OPS.regionprops.apply(cv, merged, { minArea: 0 }, ctx);
  assert.equal(ctx.values.rows.length, 4);
  const areas = ctx.values.rows.map((r) => r[1]);
  // area ต้องไม่เพิ่มขึ้นเลย (non-increasing) ตลอดทั้ง list -> DESC โดยรวม
  for (let i = 1; i < areas.length; i++) assert.ok(areas[i] <= areas[i - 1], `areas not sorted DESC: ${areas}`);
  // สองแถวที่มี area เท่ากันเป๊ะ (tieA/tieB, พื้นที่ 900) ต้องเรียง y ASC (tieA ที่ y=100 มาก่อน tieB ที่ y=10...
  // รอ: tieA y=100 > tieB y=10 ดังนั้น tieB (y=10) ต้องมาก่อน tieA (y=100) ตามกติกา "y ASC"
  const tieRows = ctx.values.rows.filter((r) => r[1] === 900);
  assert.equal(tieRows.length, 2, 'expected exactly 2 rows with the tied area 900px (30x30 rectangles)');
  assert.ok(tieRows[0][7] < tieRows[1][7], `tie-break must order by bbox_y ASC: got y=${tieRows[0][7]} before y=${tieRows[1][7]}`);
  assert.equal(tieRows[0][7], 10); // tieB (y=10) ต้องมาก่อน
  assert.equal(tieRows[1][7], 100); // tieA (y=100) ต้องมาหลัง

  // index column ต้องตรงกับตำแหน่งหลัง sort (1-based) ให้ label ที่วาดตรงกับแถวในตาราง
  ctx.values.rows.forEach((row, i) => assert.equal(row[0], i + 1));

  dst.delete(); merged.delete(); big.delete(); tieA.delete(); tieB.delete(); small.delete();
});

// ===================================================================
// regionprops-13. no row cap: spec §E explicitly documents "regionprops does not truncate" for this round
// (design decision — see handback) — verify many small objects above minArea are ALL kept, not truncated
// ===================================================================
test('regionprops-13. no row cap: many small objects above minArea all appear in rows (design decision, matches spec §E)', () => {
  const w = 400, h = 400;
  let merged = blank(w, h);
  const n = 30;
  for (let i = 0; i < n; i++) {
    const cx = 15 + (i % 10) * 38;
    const cy = 15 + Math.floor(i / 10) * 38;
    const one = withCircle(w, h, cx, cy, 10);
    cv.bitwise_or(merged, one, merged);
    one.delete();
  }
  const ctx = {};
  const dst = OPS.regionprops.apply(cv, merged, { minArea: 50 }, ctx);
  assert.equal(ctx.values.rows.length, n, `expected all ${n} objects kept, no truncation`);
  dst.delete(); merged.delete();
});

// ===================================================================
// regionprops-14. leak probes — wrapper-balance (ด่านหลัก) + heap census (native) ทุกเส้นทาง
// (เดิมเป็น malloc-fingerprint: หลังเทสต์หนักอย่าง 23 free-list หมุนที่อยู่จนผลไม่นิ่ง — ดู _leak-helpers.mjs / WORKLOG P-14)
// ===================================================================
test('regionprops-14a. leak probe: normal multi-object path', () => {
  const w = 200, h = 200;
  const m1 = withCircle(w, h, 50, 50, 20);
  const m2 = withRect(w, h, 120, 30, 50, 30);
  const merged = new cv.Mat();
  cv.bitwise_or(m1, m2, merged);
  assertWrapperBalance(cv, 'regionprops multi-object', (c) => {
    const dst = OPS.regionprops.apply(c, merged, { minArea: 50 }, {});
    dst.delete();
  }, { minCreated: 5 });
  assertNoHeapGrowth(cv, 'regionprops multi-object', () => {
    const dst = OPS.regionprops.apply(cv, merged, { minArea: 50 }, {});
    dst.delete();
  });
  merged.delete(); m1.delete(); m2.delete();
});

test('regionprops-14b. leak probe: zero-object (blank) path', () => {
  const src = blank(100, 100);
  assertWrapperBalance(cv, 'regionprops zero-object', (c) => {
    const dst = OPS.regionprops.apply(c, src, { minArea: 50 }, {});
    dst.delete();
  }, { minCreated: 3 });
  assertNoHeapGrowth(cv, 'regionprops zero-object', () => {
    const dst = OPS.regionprops.apply(cv, src, { minArea: 50 }, {});
    dst.delete();
  });
  src.delete();
});

test('regionprops-14c. leak probe: minArea filters everything out', () => {
  const src = withCircle(100, 100, 50, 50, 5); // area เล็กมาก
  assertWrapperBalance(cv, 'regionprops minArea-filters-all', (c) => {
    const dst = OPS.regionprops.apply(c, src, { minArea: 100000 }, {});
    dst.delete();
  }, { minCreated: 3 });
  assertNoHeapGrowth(cv, 'regionprops minArea-filters-all', () => {
    const dst = OPS.regionprops.apply(cv, src, { minArea: 100000 }, {});
    dst.delete();
  });
  src.delete();
});

test('regionprops-14d. leak probe: thrown-error path (injected cv.moments failure) leaks nothing, including per-contour MatVector.get() wrapper', () => {
  const m1 = withCircle(200, 200, 50, 50, 20);
  const m2 = withRect(200, 200, 120, 30, 50, 30);
  const merged = new cv.Mat();
  cv.bitwise_or(m1, m2, merged);
  assertWrapperBalance(cv, 'regionprops thrown-error (moments)', (c) => {
    OPS.regionprops.apply(makeThrowingCv(c, 'moments'), merged, { minArea: 50 }, {});
  }, { expectThrow: /injected test failure/, minCreated: 5 });
  assertNoHeapGrowth(cv, 'regionprops thrown-error (moments)', () => {
    assert.throws(() => OPS.regionprops.apply(makeThrowingCv(cv, 'moments'), merged, { minArea: 50 }, {}), /injected test failure/);
  });
  merged.delete(); m1.delete(); m2.delete();
});

// ===================================================================
// Python codegen: per-field numeric parity against real cv2 5.0.0 (raw-bytes technique, matToRaw,
// reused from tests/tier2-infra.test.mjs / keypoints-ops.test.mjs's runGeneratedPythonKeypoints pattern)
// ===================================================================
for (const name of fs.readdirSync(os.tmpdir())) {
  if (name.startsWith('regionprops-parity-')) {
    try { fs.rmSync(path.join(os.tmpdir(), name), { recursive: true, force: true }); } catch { /* best-effort */ }
  }
}
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'regionprops-parity-'));
process.on('exit', () => {
  try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ }
});

function matToRaw(mat) {
  return { data: Buffer.from(mat.data), width: mat.cols, height: mat.rows, channels: mat.channels() };
}

function pythonAvailable() {
  try {
    execFileSync('python', ['-c', 'import cv2, numpy'], { encoding: 'utf8' });
    return true;
  } catch {
    return false;
  }
}
const HAS_PYTHON = pythonAvailable();

// dump ตัวแปร `regions` (สร้างไว้แล้วใน GEN.regionprops ก่อนบรรทัด `return img`) เป็น JSON — เทคนิคเดียวกับ
// runGeneratedPythonKeypoints ใน tests/keypoints-ops.test.mjs (dump ตัวแปรภายใน process() ก่อน return)
function runGeneratedPythonRegions(pipeline, inputRaw) {
  const code = buildPythonCode(pipeline);
  const shape = inputRaw.channels === 1
    ? `(${inputRaw.height}, ${inputRaw.width})`
    : `(${inputRaw.height}, ${inputRaw.width}, ${inputRaw.channels})`;
  const loaderLine = 'img = cv2.imread(path)  # อ่านภาพ (BGR)';
  assert.ok(code.includes(loaderLine), 'generated code must contain the expected cv2.imread line to patch');
  let patched = code.replace(loaderLine, `img = np.fromfile(path, dtype=np.uint8).reshape(${shape})`);
  const returnIdx = patched.lastIndexOf('    return img');
  assert.ok(returnIdx > 0, 'generated code must end with "    return img" inside process() to patch for regions dump');
  patched = patched.slice(0, returnIdx)
    + "    with open(sys.argv[2], 'w') as _f:\n"
    + "        json.dump([[idx, float(r['area']), float(r['cx']), float(r['cy']), float(r['perimeter']), float(r['circularity']), int(r['bx']), int(r['by']), int(r['bw']), int(r['bh']), (float(r['orientation']) if r['orientation'] is not None else None)] for idx, r in enumerate(regions, start=1)], _f)\n"
    + '    return img\n'
    + patched.slice(returnIdx + '    return img'.length);
  const mainIdx = patched.indexOf('if __name__ == "__main__":');
  assert.ok(mainIdx > 0, 'generated code must contain a __main__ guard to strip');
  patched = patched.slice(0, mainIdx);
  patched += `
import sys, json
process(sys.argv[1])
`;
  const id = Math.random().toString(36).slice(2);
  const scriptPath = path.join(TMP_DIR, `rpscript_${id}.py`);
  const inPath = path.join(TMP_DIR, `rpin_${id}.bin`);
  const rpPath = path.join(TMP_DIR, `rpout_${id}.json`);
  fs.writeFileSync(scriptPath, patched, 'utf8');
  fs.writeFileSync(inPath, inputRaw.data);
  const stdout = execFileSync('python', [scriptPath, inPath, rpPath], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
  const rows = JSON.parse(fs.readFileSync(rpPath, 'utf8'));
  rows.stdout = stdout; // แนบ stdout เป็น property (ผู้เรียกเดิมใช้เป็น array ตามเดิมได้ ไม่กระทบ)
  return rows;
}

function colorRawFromGray(grayMat) {
  const c = new cv.Mat();
  cv.cvtColor(grayMat, c, cv.COLOR_GRAY2RGB);
  const raw = matToRaw(c);
  c.delete();
  return raw;
}

test('regionprops-15. Python codegen: buildPythonCode emits process(path), executed against real cv2 5.0.0', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  const src = withCircle(150, 150, 75, 75, 30);
  const code = buildPythonCode([{ id: 'regionprops', params: { minArea: 50 } }]);
  assert.match(code, /def process\(path\):/);
  assert.match(code, /cv2\.contourArea/);
  assert.match(code, /cv2\.RETR_EXTERNAL/);
  const raw = colorRawFromGray(src);
  const pyRows = runGeneratedPythonRegions([{ id: 'regionprops', params: { minArea: 50 } }], raw);
  assert.equal(pyRows.length, 1);
  src.delete();
});

test('regionprops-16. Python codegen per-field numeric parity (real cv2): area/perimeter/circularity/orientation/bbox/centroid match JS field-by-field on a synthetic multi-object image', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  const w = 300, h = 300;
  const m1 = withCircle(w, h, 70, 70, 35);
  const m2 = withRect(w, h, 180, 40, 70, 35);
  const m3 = withEllipse(w, h, 100, 220, 55, 18, 35);
  let merged = new cv.Mat();
  cv.bitwise_or(m1, m2, merged);
  cv.bitwise_or(merged, m3, merged);

  const params = { minArea: 50 };
  const ctx = {};
  const dst = OPS.regionprops.apply(cv, merged, params, ctx);
  const jsRows = ctx.values.rows;
  assert.equal(jsRows.length, 3, 'sanity: expected 3 distinct objects');

  const raw = colorRawFromGray(merged);
  const pyRows = runGeneratedPythonRegions([{ id: 'regionprops', params }], raw);
  assert.equal(pyRows.length, jsRows.length, 'JS and Python must find the same object count');

  // ลำดับต้องตรงกันเป๊ะเพราะ sort key เดียวกันทั้งสองฝั่ง (deterministic — ดู regionprops-12) จึงเทียบแบบ
  // positional ได้ตรง ๆ ไม่ต้องจับคู่ตามตำแหน่ง/nearest-neighbor
  for (let i = 0; i < jsRows.length; i++) {
    const [jIdx, jArea, jCx, jCy, jPer, jCirc, jBx, jBy, jBw, jBh, jOri] = jsRows[i];
    const [pIdx, pArea, pCx, pCy, pPer, pCirc, pBx, pBy, pBw, pBh, pOri] = pyRows[i];
    assert.equal(jIdx, pIdx, `row ${i}: index mismatch`);
    // pixel-integer fields: bbox ต้องตรงเป๊ะ (ไม่มีช่องว่างให้คลาดเคลื่อนเลย เพราะมาจาก boundingRect บน
    // contour ที่ควรเหมือนกันเป๊ะระหว่าง build ถ้า input pixels เหมือนกันเป๊ะ — ส่งแบบ raw bytes ไม่ใช่ JPEG)
    assert.equal(jBx, pBx, `row ${i}: bbox_x mismatch`);
    assert.equal(jBy, pBy, `row ${i}: bbox_y mismatch`);
    assert.equal(jBw, pBw, `row ${i}: bbox_w mismatch`);
    assert.equal(jBh, pBh, `row ${i}: bbox_h mismatch`);
    // area/perimeter: ควรตรงเป๊ะหรือใกล้เป๊ะมาก (ทั้งคู่มาจาก contour จุดเดียวกัน, cv.contourArea/cv2.contourArea
    // ทั้งคู่ผูกกับ core C++ เดียวกัน) — ใช้ tolerance เล็กน้อยกันความต่างของลำดับการบวกจุดทศนิยม
    assert.ok(Math.abs(jArea - pArea) <= Math.max(1, jArea * 0.001), `row ${i}: area ${jArea} vs ${pArea}`);
    assert.ok(Math.abs(jPer - pPer) <= Math.max(0.5, jPer * 0.005), `row ${i}: perimeter ${jPer} vs ${pPer}`);
    // centroid: floating point, tolerance หลวมกว่าเล็กน้อย (คำนวณจาก moments ที่สะสมค่าจากทุกจุด)
    assert.ok(Math.abs(jCx - pCx) < 0.5, `row ${i}: cx ${jCx} vs ${pCx}`);
    assert.ok(Math.abs(jCy - pCy) < 0.5, `row ${i}: cy ${jCy} vs ${pCy}`);
    // circularity: derived จาก area/perimeter ที่มี tolerance ของตัวเองแล้ว -> ต้องมี tolerance หลวมกว่านั้นอีกนิด
    assert.ok(Math.abs(jCirc - pCirc) < 0.01, `row ${i}: circularity ${jCirc} vs ${pCirc}`);
    // orientation: null ต้องเป็น null ทั้งคู่ หรือค่าต้องใกล้กันภายในไม่กี่องศา (floating-point fitEllipse)
    if (jOri === null || pOri === null) {
      assert.equal(jOri, pOri, `row ${i}: orientation null-mismatch (JS=${jOri}, Python=${pOri})`);
    } else {
      let diff = Math.abs(jOri - pOri) % 180;
      if (diff > 90) diff = 180 - diff;
      assert.ok(diff < 1, `row ${i}: orientation_deg ${jOri} vs ${pOri} (diff ${diff})`);
    }
  }

  dst.delete(); merged.delete(); m1.delete(); m2.delete(); m3.delete();
});

// ===================================================================
// qa 18c รอบ 1 (FAIL) — fixes: total-order tie-break, cheap-pass + cap (maxObjects), orientation convention desc
// ===================================================================

// เส้น 1px สองเส้นที่มี area=0 เท่ากัน และ bbox_x=20, bbox_y=10 เท่ากันเป๊ะ (3 key เดิมเสมอกันจริง) — A เป็น
// เส้นทแยง (30,10)->(20,20) (ไม่มีพิกเซล (20,10)), B เป็นเส้นตั้ง (20,10)->(20,16) — ไม่แตะกัน (ห่าง >=4px)
// ยืนยันแล้ว: findContours คืน A ก่อน B ดังนั้น sort 3 key แบบ stable จะได้ A ก่อน (ผิด) ส่วน total order
// (cx ASC: B=20.5 < A=25.5) ต้องได้ B ก่อน
function threeKeyTieImage() {
  const m = blank(50, 40);
  cv.line(m, new cv.Point(30, 10), new cv.Point(20, 20), new cv.Scalar(255), 1, cv.LINE_8);
  cv.line(m, new cv.Point(20, 10), new cv.Point(20, 16), new cv.Scalar(255), 1, cv.LINE_8);
  return m;
}

test('regionprops-17. tie-break is total: two objects tied on area, bbox_y AND bbox_x are ordered by cx ASC (a naive 3-key stable sort gets this wrong)', () => {
  const src = threeKeyTieImage();
  // sanity: the construction really ties on the naive 3 keys, and findContours returns the "wrong" one first
  const cs = new cv.MatVector(); const h = new cv.Mat();
  cv.findContours(src, cs, h, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  assert.equal(cs.size(), 2);
  const raw = [];
  for (let i = 0; i < 2; i++) { const c = cs.get(i); raw.push({ area: cv.contourArea(c), r: cv.boundingRect(c) }); c.delete(); }
  cs.delete(); h.delete();
  assert.equal(raw[0].area, raw[1].area);
  assert.equal(raw[0].r.y, raw[1].r.y);
  assert.equal(raw[0].r.x, raw[1].r.x);
  assert.equal(raw[0].r.width, 11, 'findContours returns the diagonal (wider, larger cx) first -> naive stable sort would keep it first');

  const ctx = {};
  const dst = OPS.regionprops.apply(cv, src, { minArea: 0 }, ctx);
  const rows = ctx.values.rows;
  assert.equal(rows.length, 2);
  assert.equal(rows[0][8], 1, 'vertical line (bbox_w=1, cx=20.5) must come first by cx ASC');
  assert.equal(rows[1][8], 11);
  assert.ok(rows[0][2] < rows[1][2]);
  dst.delete(); src.delete();
});

test('regionprops-18. comparator unit: cy ASC after cx, then the unique top-left point tx; lazy centroid only consulted on 3-key ties', () => {
  let calls = 0;
  const centroidOf = (r) => { calls++; return r; };
  const base = { area: 5, by: 3, bx: 4 };
  assert.ok(compareRegionCandidates({ ...base, cx: 1, cy: 9, tx: 9 }, { ...base, cx: 2, cy: 0, tx: 0 }, centroidOf) < 0);
  assert.ok(compareRegionCandidates({ ...base, cx: 1, cy: 2, tx: 9 }, { ...base, cx: 1, cy: 3, tx: 0 }, centroidOf) < 0);
  assert.ok(compareRegionCandidates({ ...base, cx: 1, cy: 2, tx: 7 }, { ...base, cx: 1, cy: 2, tx: 6 }, centroidOf) > 0);
  // area DESC / by / bx decide first: centroid must NOT be requested (proves laziness)
  calls = 0;
  assert.ok(compareRegionCandidates({ area: 9, by: 0, bx: 0, tx: 0 }, { area: 5, by: 0, bx: 0, tx: 0 }, centroidOf) < 0);
  assert.ok(compareRegionCandidates({ area: 5, by: 1, bx: 0, tx: 0 }, { area: 5, by: 2, bx: 0, tx: 0 }, centroidOf) < 0);
  assert.ok(compareRegionCandidates({ area: 5, by: 1, bx: 3, tx: 0 }, { area: 5, by: 1, bx: 2, tx: 0 }, centroidOf) > 0);
  assert.equal(calls, 0, 'moments (centroid) must only be computed for 3-key ties');
});

test('regionprops-19. regionScan area/bbox_x/bbox_y equal cv.contourArea / cv.boundingRect exactly on thousands of real contours; (by, tx) unique per object', () => {
  const w = 400, h = 300;
  const src = blank(w, h);
  let s = 7;
  for (let i = 0; i < w * h; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; if (s % 100 < 8) src.data[i] = 255; }
  cv.circle(src, new cv.Point(200, 150), 40, new cv.Scalar(255), -1);
  cv.ellipse(src, new cv.Point(80, 80), new cv.Size(50, 15), 33, 0, 360, new cv.Scalar(255), -1);
  const cs = new cv.MatVector(); const hier = new cv.Mat();
  cv.findContours(src, cs, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  const n = cs.size();
  assert.ok(n > 1000, `expected many contours, got ${n}`);
  const topLefts = new Set();
  for (let i = 0; i < n; i++) {
    const c = cs.get(i);
    const sc = regionScan(c.data32S);
    const r = cv.boundingRect(c);
    assert.equal(sc.area, cv.contourArea(c), `contour ${i}: area`);
    assert.equal(sc.bx, r.x); assert.equal(sc.by, r.y);
    const d = c.data32S; let found = false;
    for (let k = 0; k < d.length; k += 2) if (d[k] === sc.tx && d[k + 1] === sc.by) found = true;
    assert.ok(found, 'top-left key must be an actual contour point');
    topLefts.add(sc.by * 100000 + sc.tx);
    c.delete();
  }
  assert.equal(topLefts.size, n, '(by, tx) must be unique across distinct objects (final sort key is unique by construction)');
  cs.delete(); hier.delete(); src.delete();
});

// 60 วงกลมหลายขนาด (ขนาดซ้ำกันเป็นชุด -> มี tie บน area ให้ tie-break ทำงาน) ใช้ทดสอบ cap
function manyObjectsImage() {
  const w = 640, h = 480;
  const m = blank(w, h);
  let k = 0;
  for (let gy = 0; gy < 6; gy++) for (let gx = 0; gx < 10; gx++) {
    const r = 4 + (k % 7) * 2; k++;
    cv.circle(m, new cv.Point(30 + gx * 62, 35 + gy * 75), r, new cv.Scalar(255), -1);
  }
  return m;
}
function gridBlobs(nBlobs) {
  const m = blank(800, 600);
  for (let i = 0; i < nBlobs; i++) {
    const x = 5 + (i % 40) * 20, y = 5 + Math.floor(i / 40) * 20;
    cv.circle(m, new cv.Point(x + 4, y + 4), 3 + (i % 3), new cv.Scalar(255), -1);
  }
  return m;
}

test('regionprops-20. maxObjects cap: rows capped, survivors are exactly the top-N of the full (uncapped) total order, total exact, Thai truncation note present', () => {
  const src = manyObjectsImage();
  const ctxAll = {};
  const dAll = OPS.regionprops.apply(cv, src, { minArea: 0, maxObjects: 0 }, ctxAll);
  assert.equal(ctxAll.values.rows.length, 60);
  assert.equal(ctxAll.values.total, 60);
  assert.doesNotMatch(ctxAll.info, /แสดง/);
  const ctx = {};
  const d = OPS.regionprops.apply(cv, src, { minArea: 0, maxObjects: 25 }, ctx);
  assert.equal(ctx.values.rows.length, 25);
  assert.equal(ctx.values.total, 60, 'total object count must stay exact when the cap fires');
  assert.deepEqual(ctx.values.rows, ctxAll.values.rows.slice(0, 25), 'capped rows must equal the first 25 rows of the uncapped run');
  assert.match(ctx.info, /พบวัตถุ 60 ชิ้น แสดง 25 ชิ้นที่ใหญ่ที่สุด \(ปรับ 'แสดงวัตถุสูงสุด' ได้\)/);
  assert.equal(ctx.info.match(/พื้นที่เฉลี่ย (\d+)px²/)[1], ctxAll.info.match(/พื้นที่เฉลี่ย (\d+)px²/)[1], 'average area is over ALL found objects');
  dAll.delete(); d.delete(); src.delete();
});

test('regionprops-21. maxObjects defaults: missing key (older saved project) -> 500; 0 = unlimited (not falsy); cap >= count -> no truncation note', () => {
  const src = manyObjectsImage();
  const ctxMissing = {};
  const d1 = OPS.regionprops.apply(cv, src, { minArea: 0 }, ctxMissing);
  assert.equal(ctxMissing.values.rows.length, 60);
  assert.doesNotMatch(ctxMissing.info, /แสดง/);
  d1.delete();
  const ctxZero = {};
  const d2 = OPS.regionprops.apply(cv, src, { minArea: 0, maxObjects: 0 }, ctxZero);
  assert.equal(ctxZero.values.rows.length, 60, 'maxObjects:0 must mean unlimited');
  d2.delete(); src.delete();
  assert.equal(OPS.regionprops.params.find((p) => p.key === 'maxObjects').default, 500);
  const big = gridBlobs(600);
  const ctxDef = {};
  const d3 = OPS.regionprops.apply(cv, big, { minArea: 0 }, ctxDef);
  assert.equal(ctxDef.values.total, 600);
  assert.equal(ctxDef.values.rows.length, 500, 'missing maxObjects key -> default cap 500');
  assert.match(ctxDef.info, /พบวัตถุ 600 ชิ้น แสดง 500 ชิ้นที่ใหญ่ที่สุด/);
  d3.delete(); big.delete();
});

// Proxy นับจำนวนครั้งที่เรียก cv.* — ใช้พิสูจน์ว่า cap ตัด "ก่อน" งานแพง ไม่ใช่หลัง
function makeCountingCv(realCv, names) {
  const counts = Object.fromEntries(names.map((n) => [n, 0]));
  const proxy = new Proxy(realCv, {
    get(target, prop) {
      if (names.includes(prop)) return (...args) => { counts[prop]++; return target[prop](...args); };
      return target[prop];
    },
  });
  return { proxy, counts };
}

test('regionprops-22. expensive work only for survivors: arcLength/putText/boundingRect called exactly shown-count times, fitEllipse/moments <= shown-count, no per-contour contourArea', () => {
  const big = gridBlobs(600);
  const { proxy, counts } = makeCountingCv(cv, ['arcLength', 'fitEllipse', 'putText', 'boundingRect', 'moments', 'contourArea']);
  const ctx = {};
  const d = OPS.regionprops.apply(proxy, big, { minArea: 0, maxObjects: 40 }, ctx);
  assert.equal(ctx.values.total, 600);
  assert.equal(ctx.values.rows.length, 40);
  assert.equal(counts.arcLength, 40);
  assert.equal(counts.putText, 40);
  assert.equal(counts.boundingRect, 40);
  assert.ok(counts.fitEllipse <= 40, `fitEllipse called ${counts.fitEllipse} times`);
  assert.ok(counts.moments <= 40, `moments called ${counts.moments} times; must be lazy (<= shown when no 3-key ties)`);
  assert.equal(counts.contourArea, 0, 'cheap pass uses regionScan, not a per-contour embind call');
  d.delete(); big.delete();
});

test('regionprops-23. cap is a real perf win: 1080p salt-and-pepper 5% at minArea 0 with the default cap runs far faster than uncapped', () => {
  const w = 1920, h = 1080;
  const src = blank(w, h);
  let s = 1;
  for (let i = 0; i < w * h; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; if (s % 10000 < 500) src.data[i] = 255; }
  const run = (params) => { const t = performance.now(); const c = {}; const d = OPS.regionprops.apply(cv, src, params, c); d.delete(); return { ms: performance.now() - t, c }; };
  run({ minArea: 0 }); // warm-up
  const capped = run({ minArea: 0 });
  const uncapped = run({ minArea: 0, maxObjects: 0 });
  assert.ok(capped.c.values.total > 50000);
  assert.equal(capped.c.values.total, uncapped.c.values.total);
  // วัดจริงบนเครื่อง dev: capped ~118ms vs uncapped ~1018ms — เกณฑ์หลวม (3x) กันเครื่องช้า/ผันผวน
  assert.ok(capped.ms * 3 < uncapped.ms, `capped ${capped.ms.toFixed(0)}ms should be far below uncapped ${uncapped.ms.toFixed(0)}ms`);
  src.delete();
});

// ===================================================================
// regionprops-24b/24c — wrapper-balance ทุกเส้นทาง (ใช้ tracker กลางใน tests/_leak-helpers.mjs)
// เหตุผลที่ไม่ใช้ malloc fingerprint (ตรวจแล้ว ดู worklog): หลัง workload ก่อนหน้าที่ใช้บล็อกขนาดต่างกัน free list ของ
// allocator "หมุน" ที่อยู่ของ chunk ขนาดเท่ากัน — control ที่ทำแค่ cvtColor alloc+free ก็ให้ fingerprint ต่างกัน 10/10 รอบ
// tracker นับ wrapper JS ทุกตัว (new Mat/MatVector, Mat จาก MatVector.get() ต่อ contour, roi/clone, factory) เทียบ .delete()
// ครบทุก path รวม error path แบบ deterministic
// ===================================================================
function bal(label, src, params, { throwOn = null, minCreated = 3 } = {}) {
  assertWrapperBalance(cv, label, (c) => {
    const d = OPS.regionprops.apply(throwOn ? makeThrowingCv(c, throwOn) : c, src, params, {});
    d.delete();
  }, { expectThrow: throwOn ? /injected test failure/ : false, minCreated });
}

test('regionprops-24b. wrapper-balance probe: lazy-centroid path inside the sort comparator (3-key tie) — normal, and cv.moments throwing inside the comparator', () => {
  const src = threeKeyTieImage();
  bal('lazy centroid normal', src, { minArea: 0 });
  bal('lazy centroid, moments throws in comparator', src, { minArea: 0 }, { throwOn: 'moments' });
  src.delete();
});

test('regionprops-24c. wrapper-balance probe: every path — cap firing, uncapped, zero objects, minArea filters all, and throws in the cheap pass / expensive pass / drawing', () => {
  const many = manyObjectsImage();
  const empty = blank(60, 40);
  bal('cap fires', many, { minArea: 0, maxObjects: 10 });
  bal('uncapped', many, { minArea: 0, maxObjects: 0 });
  bal('zero objects', empty, { minArea: 0 });
  bal('minArea filters all', many, { minArea: 100000 });
  bal('expensive pass: fitEllipse throws (after cap)', many, { minArea: 0, maxObjects: 10 }, { throwOn: 'fitEllipse' });
  bal('expensive pass: arcLength throws', many, { minArea: 0, maxObjects: 10 }, { throwOn: 'arcLength' });
  bal('drawing: putText throws', many, { minArea: 0, maxObjects: 10 }, { throwOn: 'putText' });
  bal('findContours throws', many, { minArea: 0 }, { throwOn: 'findContours' });
  many.delete(); empty.delete();
});

test('regionprops-25. Thai desc states the orientation convention (0° horizontal, clockwise because image y points down, 0–180° range), Python has the matching comment', () => {
  const d = OPS.regionprops.desc;
  assert.match(d, /0° = แนวนอน/);
  assert.match(d, /ตามเข็มนาฬิกา/);
  assert.match(d, /แกน y ของภาพชี้ลง/);
  assert.match(d, /0–180°/);
  const code = buildPythonCode([{ id: 'regionprops', params: { minArea: 50, maxObjects: 500 } }]);
  assert.match(code, /0 = แนวนอน, เพิ่มตามเข็มนาฬิกาเพราะแกน y ของภาพชี้ลง/);
  assert.match(code, /max_objects = 500/);
  assert.match(buildPythonCode([{ id: 'regionprops', params: { minArea: 50 } }]), /max_objects = 500/, 'missing key -> 500 in Python too');
  assert.match(buildPythonCode([{ id: 'regionprops', params: { minArea: 50, maxObjects: 0 } }]), /max_objects = 0 /, '0 must stay 0 (unlimited), not fall back');
});

test('regionprops-27. string-typed params (e.g. from an older saved project) are coerced exactly like codegen finiteOr: "0" means 0 in both JS and the generated Python', () => {
  const src = manyObjectsImage();
  const ctx = {};
  const d = OPS.regionprops.apply(cv, src, { minArea: '0', maxObjects: '0' }, ctx);
  assert.equal(ctx.values.rows.length, 60, 'maxObjects "0" must mean unlimited (not fall back to 500) and minArea "0" must filter nothing');
  d.delete();
  const ctx2 = {};
  const d2 = OPS.regionprops.apply(cv, src, { minArea: '0', maxObjects: '25' }, ctx2);
  assert.equal(ctx2.values.rows.length, 25);
  d2.delete();
  const code = buildPythonCode([{ id: 'regionprops', params: { minArea: '0', maxObjects: '0' } }]);
  assert.match(code, /if area < 0:/);
  assert.match(code, /max_objects = 0 /);
  // non-numeric garbage falls back to the default on BOTH sides
  const ctx3 = {};
  const d3 = OPS.regionprops.apply(cv, src, { minArea: 'abc', maxObjects: 'abc' }, ctx3);
  const code3 = buildPythonCode([{ id: 'regionprops', params: { minArea: 'abc', maxObjects: 'abc' } }]);
  assert.match(code3, /if area < 50:/);
  assert.match(code3, /max_objects = 500 /);
  assert.ok(ctx3.values.rows.every((r) => r[1] >= 50));
  d3.delete(); src.delete();
});

test('regionprops-26. Python parity (real cv2) with the cap firing AND a 3-key tie present: same rows in the same order', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  const src = manyObjectsImage();
  // the 3-key-tie pair (area 0, same bbox_x/bbox_y) in a free corner
  cv.line(src, new cv.Point(630, 440), new cv.Point(620, 450), new cv.Scalar(255), 1, cv.LINE_8);
  cv.line(src, new cv.Point(620, 440), new cv.Point(620, 446), new cv.Scalar(255), 1, cv.LINE_8);
  for (const maxObjects of [0, 30]) {
    const params = { minArea: 0, maxObjects };
    const ctx = {};
    const d = OPS.regionprops.apply(cv, src, params, ctx);
    const pyRows = runGeneratedPythonRegions([{ id: 'regionprops', params }], colorRawFromGray(src));
    const js = ctx.values.rows;
    assert.equal(pyRows.length, js.length, `maxObjects=${maxObjects}: row count`);
    for (let i = 0; i < js.length; i++) {
      assert.equal(pyRows[i][0], js[i][0]);
      assert.equal(pyRows[i][1], js[i][1], `maxObjects=${maxObjects} row ${i}: area`);
      assert.deepEqual(pyRows[i].slice(6, 10), js[i].slice(6, 10), `maxObjects=${maxObjects} row ${i}: bbox (same object in the same position => same order)`);
      assert.ok(Math.abs(pyRows[i][2] - js[i][2]) < 1e-6 && Math.abs(pyRows[i][3] - js[i][3]) < 1e-6, `maxObjects=${maxObjects} row ${i}: centroid`);
    }
    d.delete();
  }
  src.delete();
});

// ===================================================================
// regionprops-28: ตารางที่ Python พิมพ์ออก stdout (CV eng เพิ่ม 6 บรรทัดใน GEN.regionprops ก่อน draw loop —
// header + สูงสุด 20 แถว) — qa ขอ test coverage
// ===================================================================
// จำลอง f"{x:.Nf}" ของ Python เป๊ะ: Python ปัดค่า binary จริงแบบ "ครึ่งไปหาเลขคู่" (round-half-even) ส่วน
// JS toFixed ปัดครึ่งขึ้นเสมอ — ต่างกันเฉพาะเมื่อค่าเป็น tie ทศนิยมพอดีจริง (เช่น cx=30.25 ซึ่ง binary แทนได้
// เป๊ะ, Python ".1f" ได้ "30.2" แต่ JS toFixed(1) ได้ "30.3") — toFixed(100) ให้ทศนิยมจริงของค่า binary ครบ
// (double ที่ใช้ในเทสต์นี้มีทศนิยมไม่เกิน ~60 หลัก) จึงปัดเองจากสตริงได้ตรงกับ Python ทุกกรณี
function pyFixed(x, d) {
  const neg = x < 0;
  const [ip, fp] = Math.abs(x).toFixed(100).split('.');
  const keep = fp.slice(0, d), rest = fp.slice(d);
  let n = BigInt(ip + keep);
  const restIsHalf = /^50*$/.test(rest), restAboveHalf = rest > '5' && !restIsHalf;
  if (restAboveHalf || (restIsHalf && n % 2n === 1n)) n += 1n;
  let s = n.toString().padStart(d + 1, '0');
  s = d > 0 ? `${s.slice(0, -d)}.${s.slice(-d)}` : s;
  return (neg && /[1-9]/.test(s) ? '-' : '') + s;
}
function expectedPrintedRow(row) {
  const [idx, area, cx, cy, per, circ, bx, by, bw, bh, ori] = row;
  const o = ori === null ? '-' : pyFixed(ori, 1);
  return `${idx}  ${pyFixed(area, 1)}  ${pyFixed(cx, 1)}  ${pyFixed(cy, 1)}  ${pyFixed(per, 1)}  ${pyFixed(circ, 2)}  (${bx},${by},${bw},${bh})  ${o}`;
}
function printedTable(stdout) {
  const lines = stdout.split(/\r?\n/);
  const h = lines.findIndex((l) => l.startsWith('index  area'));
  if (h < 0) return null;
  const rows = [];
  for (let i = h + 1; i < lines.length && /^\d+  /.test(lines[i]); i++) rows.push(lines[i]);
  return rows;
}

test('regionprops-28a. pyFixed matches Python f-string rounding on exact binary ties (sanity for the formatter used by 28b)', () => {
  assert.equal(pyFixed(30.25, 1), '30.2'); // Python: f"{30.25:.1f}" == "30.2" (ties-to-even on an exact binary value)
  assert.equal(pyFixed(30.75, 1), '30.8');
  assert.equal(pyFixed(0.125, 2), '0.12');
  assert.equal(pyFixed(0.1, 1), '0.1');
  assert.equal(pyFixed(2.675, 2), '2.67'); // 2.675 is really 2.67499999... in binary
  assert.equal(pyFixed(0.995, 2), '0.99');
  assert.equal(pyFixed(9.96, 1), '10.0');
});

test('regionprops-28b. Python stdout table: header present, min(20, rows shown) rows, each row equals the JS row formatted (1 dp / circularity 2 dp / int bbox / "-" for null orientation); blank image prints no header', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  if (HAS_PYTHON) {
    const ok = execFileSync('python', ['-c', 'print(f"{30.25:.1f} {0.125:.2f} {2.675:.2f}")'], { encoding: 'utf8' }).trim();
    assert.equal(ok, '30.2 0.12 2.67', 'Python rounding assumption behind pyFixed');
  }
  // 60 วงกลม + วงรีเอียง (มีค่ามุม) + สี่เหลี่ยมตรง (4 จุด -> มุม null -> "-") วางในแถบว่างระหว่างแถววงกลม
  // ทั้งสองชิ้นใหญ่กว่าวงกลมทุกวงจึงอยู่ใน 10 อันดับแรกเสมอ (อยู่ในตารางที่พิมพ์ทั้งกรณี cap 10 และไม่ cap)
  const src = manyObjectsImage();
  cv.ellipse(src, new cv.Point(150, 72), new cv.Size(40, 10), 20, 0, 360, new cv.Scalar(255), -1);
  cv.rectangle(src, new cv.Point(300, 60), new cv.Point(360, 85), new cv.Scalar(255), -1);
  const raw = colorRawFromGray(src);
  for (const maxObjects of [500, 10]) {
    const params = { minArea: 0, maxObjects };
    const ctx = {};
    const d = OPS.regionprops.apply(cv, src, params, ctx);
    d.delete();
    const jsRows = ctx.values.rows;
    assert.equal(ctx.values.total, 62, 'sanity: 60 circles + ellipse + rectangle');
    const py = runGeneratedPythonRegions([{ id: 'regionprops', params }], raw);
    const printed = printedTable(py.stdout);
    assert.ok(printed, `maxObjects=${maxObjects}: header line starting with "index  area" must be printed`);
    assert.equal(printed.length, Math.min(20, jsRows.length), `maxObjects=${maxObjects}: printed row count`);
    for (let i = 0; i < printed.length; i++) {
      assert.equal(printed[i], expectedPrintedRow(jsRows[i]), `maxObjects=${maxObjects} printed row ${i + 1}`);
    }
    assert.ok(printed.some((l) => l.endsWith('  -')), `maxObjects=${maxObjects}: the rectangle row must print "-" for orientation`);
    assert.ok(printed.some((l) => /  \d+\.\d$/.test(l)), `maxObjects=${maxObjects}: the ellipse/circle rows must print a numeric orientation`);
  }
  src.delete();

  const empty = blank(80, 60);
  const pyBlank = runGeneratedPythonRegions([{ id: 'regionprops', params: { minArea: 0 } }], colorRawFromGray(empty));
  assert.equal(pyBlank.length, 0);
  assert.equal(printedTable(pyBlank.stdout), null, 'blank image must print no table header');
  assert.match(pyBlank.stdout, /ไม่พบวัตถุ/);
  empty.delete();
});

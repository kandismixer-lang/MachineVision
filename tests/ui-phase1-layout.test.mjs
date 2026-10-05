// UI Phase 1 fix round — คณิตศาสตร์กระดาน (graph/layout-math.js): fit / จุดวางกล่องใหม่ / buffer canvas / ข้อความ status
import test from 'node:test';
import assert from 'node:assert/strict';
import { computeFit, findFreeSpot, canvasBufferSize, statusText, sourceDetail, applyFit, isViewUntouched } from '../src/renderer/graph/layout-math.js';

const TH = 28;
const card = (x, y, w, h) => ({ x0: x, y0: y - TH, x1: x + w, y1: y + h }); // เหมือน nodeRects() ใน editor.js
const overlap = (a, b) => a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;
const defaultPair = () => [card(40, 80, 192, 56), card(260, 80, 192, 96)];

test('p1-a1. computeFit: กราฟตั้งต้น ~803×293 ได้ scale=1 และอยู่กึ่งกลาง', () => {
  const f = computeFit(defaultPair(), 803, 293);
  assert.equal(f.scale, 1);
  // bbox x 40..452 กึ่งกลาง 246 → เมื่อ scale=1 ตรงกลาง viewport (401.5) พอดี
  assert.ok(Math.abs((246 + f.ox) - 803 / 2) < 1e-9);
  assert.ok(Math.abs((114 + f.oy) - 293 / 2) < 1e-9); // y กึ่งกลาง bbox = (52+176)/2
});

test('p1-a2. computeFit: แถว 8 ขั้นที่กว้าง 744 → scale=0.6 และชิดซ้าย (ไม่ตัดต้นทาง)', () => {
  const rects = [card(40, 80, 192, 56)];
  for (let i = 1; i <= 8; i++) rects.push(card(40 + i * 220, 80, 192, 56));
  const f = computeFit(rects, 744, 300);
  assert.equal(f.scale, 0.6);
  assert.ok(Math.abs(f.ox - (40 / 0.6 - 40)) < 1e-9, 'ชิดซ้าย: ขอบซ้ายของ bbox อยู่ที่ pad');
  assert.ok((40 + f.ox) * f.scale >= 0, 'Source ไม่หลุดขอบซ้าย');
});

test('p1-a2b. computeFit: สูงเกินแม้ scale ต่ำสุด → ชิดบน (ไม่ตัดทั้งบนและล่าง)', () => {
  const rects = [card(40, 80, 192, 56), card(40, 900, 192, 56)];
  const f = computeFit(rects, 800, 300);
  assert.equal(f.scale, 0.6);
  assert.ok(Math.abs(f.oy - (40 / 0.6 - (80 - TH))) < 1e-9);
});

test('p1-a3. computeFit: กล่องเดียว + viewport ว่าง → null', () => {
  const f = computeFit([card(100, 100, 192, 56)], 800, 300);
  assert.equal(f.scale, 1);
  assert.ok(Math.abs((100 + 96) + f.ox - 400) < 1e-9);
  assert.equal(computeFit([card(0, 0, 10, 10)], 0, 300), null);
  assert.equal(computeFit([card(0, 0, 10, 10)], 800, 0), null);
  assert.equal(computeFit([], 800, 300), null);
});

test('p1-b1. findFreeSpot: กราฟตั้งต้น → ขวาของกราฟก่อน (แถวกลางเดียวกัน) และไม่ทับกล่องใด ๆ (รวมแถบหัว)', () => {
  const rects = defaultPair();
  const size = [192, 56];
  const centre = [246, 114]; // กึ่งกลางของกราฟตั้งต้นหลัง fit
  const [px, py] = findFreeSpot(rects, centre, size, TH);
  const mine = { x0: px, y0: py - TH, x1: px + size[0], y1: py + size[1] };
  for (const r of rects) assert.ok(!overlap(mine, r), 'ทับกล่องเดิม');
  assert.equal(px, 452 + 28, 'ขวาสุด + 28');
  assert.equal(py, 114 - 28, 'แถวกลางเดียวกัน (centre.y − h/2)');
});

test('p1-b1b. findFreeSpot: เพิ่ม 6 กล่องติดกันบนจอ 803×293 → ไม่ทับกัน และอยู่ในขอบเขตที่มองเห็นที่ scale .6 ทุกกล่อง', () => {
  const rects = defaultPair();
  const size = [192, 56];
  const bounds = { x1: 40 + (803 - 80) / 0.6, y1: 52 + (293 - 80) / 0.6 };
  for (let i = 0; i < 6; i++) {
    const [px, py] = findFreeSpot(rects, [246, 114], size, TH, 16, bounds);
    const mine = { x0: px, y0: py - TH, x1: px + size[0], y1: py + size[1] };
    for (const r of rects) assert.ok(!overlap(mine, r), `add #${i + 1} ทับ`);
    assert.ok(px + size[0] <= bounds.x1 && py + size[1] <= bounds.y1, `add #${i + 1} หลุดขอบเขต`);
    rects.push(mine);
  }
});

test('p1-b2. findFreeSpot: ที่กลางว่างอยู่แล้ว → วางตรงกลางเป๊ะ · แน่นทุกแถว → ใต้ bbox', () => {
  assert.deepEqual(findFreeSpot([], [300, 200], [192, 56], TH), [204, 172]);
  const wall = [{ x0: -1e5, y0: -1e5, x1: 1e5, y1: 1e5 }];
  const [wx] = findFreeSpot(wall, [0, 0], [192, 56], TH);
  assert.ok(wx >= 1e5 + 16, 'กำแพงใหญ่ → ออกไปขวาสุดของกราฟ ไม่ทับ');
  // ขอบเขตที่มองเห็นเล็กจนไม่มีที่ → ยังคืนที่ว่างที่ไม่ทับ (ผู้เรียก pan เอง)
  const [bx, by] = findFreeSpot(defaultPair(), [246, 114], [192, 56], TH, 16, { x1: 100, y1: 100 });
  assert.ok(!overlap({ x0: bx, y0: by - TH, x1: bx + 192, y1: by + 56 }, defaultPair()[1]));
});

test('p1-c1. canvasBufferSize: = ขนาด CSS ปัดเศษเป๊ะ ไม่มี floor 200×150', () => {
  assert.deepEqual(canvasBufferSize({ width: 803.4, height: 116.2 }), { w: 803, h: 116 });
  assert.deepEqual(canvasBufferSize({ width: 120, height: 40 }), { w: 120, h: 40 });
});

test('p1-c2. canvasBufferSize: ซ่อนอยู่ (0 / ไม่มี rect) → null', () => {
  assert.equal(canvasBufferSize({ width: 0, height: 300 }), null);
  assert.equal(canvasBufferSize({ width: 800, height: 0.4 }), null);
  assert.equal(canvasBufferSize(null), null);
});

test('p1-d1. statusText: ใช้ชื่อขั้นตอน, ภาพต้นฉบับ, ชื่อ Display, จำนวนที่ไม่ต่อสาย', () => {
  const lab = (s) => s.label;
  const dps = [
    { connected: true, name: null, autoNum: 1, pipeline: [{ label: 'Grayscale' }, { label: 'Otsu Threshold' }] },
    { connected: true, name: 'ผลสุดท้าย', autoNum: 2, pipeline: [] },
    { connected: false, pipeline: [] },
  ];
  assert.equal(statusText(dps, lab), 'Display 1: Grayscale → Otsu Threshold   |   ผลสุดท้าย: ภาพต้นฉบับ   (อีก 1 กล่องยังไม่ต่อสาย)');
  assert.equal(statusText([], lab), 'ยังไม่มีกล่องผลลัพธ์');
  assert.match(statusText([{ connected: false, pipeline: [] }], lab), /^⚠/);
});

test('p1-d2. sourceDetail: ไม่เคยมี "undefined" — ขนาดยังไม่รู้ก็แสดงแค่ชื่อ', () => {
  assert.equal(sourceDetail({ name: 'parts.png', w: 1280, h: 720 }), 'parts.png · 1280×720');
  assert.equal(sourceDetail({ name: 'parts.png', w: undefined, h: undefined }), 'parts.png');
  assert.equal(sourceDetail({ w: 1280, h: 720, port: 1 }), '1280×720 · Port 1');
  assert.equal(sourceDetail({}), '');
});

test('p1-e1. applyFit/isViewUntouched: ds.offset เป็น Float32Array (330.8 → 330.79998779) แล้วยังนับว่า "ยังไม่ถูกแตะ"', () => {
  const ds = { scale: 1, offset: new Float32Array(2) };
  const snap = applyFit(ds, { scale: 0.6, ox: 330.8, oy: 12.3456789 });
  assert.notEqual(ds.offset[0], 330.8, 'ยืนยันว่า Float32 ปัดเศษจริง');
  assert.equal(isViewUntouched(ds, snap), true);
  // snapshot แบบเก่า (เก็บ double ก่อนปัดเศษ) ต้องถูกมองว่า "ถูกแตะ" ทั้งที่ไม่มีใครแตะ = บั๊กเดิม
  assert.equal(isViewUntouched(ds, { s: 0.6, ox: 330.8, oy: 12.3456789 }), false);
});

test('p1-e2. isViewUntouched: ซูม/เลื่อน/ยังไม่เคย fit → false', () => {
  const ds = { scale: 1, offset: new Float32Array(2) };
  const snap = applyFit(ds, { scale: 1, ox: 100.1, oy: 50.7 });
  assert.equal(isViewUntouched(ds, snap), true);
  ds.offset[0] += 1; assert.equal(isViewUntouched(ds, snap), false);
  ds.offset[0] -= 1; ds.scale = 1.2; assert.equal(isViewUntouched(ds, snap), false);
  assert.equal(isViewUntouched(ds, null), false);
});

// 2026-10-09 — เทสต์ของ UI เฟส 3 (nice to have)
//   3.1 ซูม/เลื่อน + แปลงพิกัดเมาส์ → พิกัดภาพ · 3.2 เส้นเกณฑ์ T / แผน histogram · 3.3 ใช้ล่าสุด · 3.4 พื้นหลังจุด
//   3.5 สายวิ่ง (เงื่อนไขเปิด/ปิด) · 3.6 error ราย node (worker → UI) · 3.7 คีย์ 1–9/F · 3.8 ขนาด buffer ที่ DPR ≠ 1
// ตรรกะล้วนอยู่ใน src/renderer/ui/phase3-math.js · DOM stub เขียนเองในไฟล์ (ไม่มี jsdom) ติดตั้ง/คืนในแต่ละเทสต์ด้วย try/finally
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import cv from '@techstark/opencv-js';
import {
  fitRect, FIT_VIEW, maxZoom, displayScale, clampView, zoomAt, panBy, wheelFactor, screenToImage, visiblePixelRange,
  readRGB, formatPixel, formatPos, formatZoomPct, PX_GRID_SCALE, PX_MAX_SCALE,
  histogramPlan, thresholdMarker, markerX, otsuFromHistogram,
  dotGridRange, flowActive, flowT, flowDotCount,
  routeWorkerError, shortErrorText, ERROR_FALLBACK_TEXT,
  nthStepNodeId, parseNavKey, selectedIdOf, buildStepIds,
  canvasBufferSizeDpr, histogramBufferSize,
  SELECTED_PREVIEW_PREFIX, SELECTED_INPUT_PREFIX, buildInputPipeline, pickWorkerPipelines, classifyResultId, resolveThresholdT,
  visibleAreaOf, isFlowLink, createFlowLoop,
} from '../src/renderer/ui/phase3-math.js';
import { createTileInspector } from '../src/renderer/ui/pixel-inspector.js';
import { resolveRecent, pushRecent, createPalettePanel, PALETTE_RECENT_KEY, PALETTE_RECENT_MAX, OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { stepAndNotesOf, infoTextsOf } from '../src/renderer/ui/detail-panel.js';
import { localParamsFor } from '../src/renderer/filters/ui-meta.js';
import { FILTERS, OPS } from '../src/renderer/filters/registry.js';

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `expected ${a} ≈ ${b}`);

// ===================================================================
// 3.1 ซูม/เลื่อน
// ===================================================================
const G = { vw: 800, vh: 400, iw: 640, ih: 400 }; // fit: f = 1 (สูงพอดี) · ภาพกว้าง 640 วางกลาง → x เริ่ม 80
const G2 = { vw: 784, vh: 283, iw: 640, ih: 400 }; // f = 0.7075 (เคสจริงจากแอป)

test('p3-a1 fitRect: contain + กึ่งกลาง · ขนาดไม่ถูกต้อง → null', () => {
  const r = fitRect(800, 400, 640, 400);
  assert.deepEqual([r.f, r.x, r.y, r.w, r.h], [1, 80, 0, 640, 400]);
  const r2 = fitRect(300, 600, 640, 400);
  near(r2.f, 300 / 640); near(r2.x, 0); near(r2.y, (600 - 400 * (300 / 640)) / 2);
  assert.equal(fitRect(0, 100, 10, 10), null);
  assert.equal(fitRect(100, 100, 0, 10), null);
});

test('p3-a2 screenToImage ที่ fit: มุม/กลาง/นอกภาพ (แถบดำข้าง) = inside false · พิกัดไม่หลุดช่วง', () => {
  const v = FIT_VIEW;
  let p = screenToImage(v, 80, 0, G);
  assert.deepEqual([p.x, p.y, p.inside], [0, 0, true]);
  p = screenToImage(v, 80 + 639.9, 399.9, G);
  assert.deepEqual([p.x, p.y, p.inside], [639, 399, true]);
  p = screenToImage(v, 400, 200, G);
  assert.deepEqual([p.x, p.y, p.inside], [320, 200, true]);
  p = screenToImage(v, 10, 200, G); // แถบดำซ้าย
  assert.equal(p.inside, false);
  assert.equal(p.x, 0, 'clamp เข้าช่วงภาพ');
  p = screenToImage(v, 790, 200, G); // แถบดำขวา
  assert.equal(p.inside, false);
  assert.equal(p.x, 639);
});

test('p3-a3 screenToImage ใช้อ็อบเจ็กต์ out ซ้ำ (mousemove ไม่สร้างของใหม่)', () => {
  const out = { x: 0, y: 0, inside: false };
  const r1 = screenToImage(FIT_VIEW, 400, 200, G, out);
  assert.equal(r1, out);
  const r2 = screenToImage(FIT_VIEW, 100, 10, G, out);
  assert.equal(r2, out);
  assert.deepEqual([out.x, out.y], [20, 10]);
});

test('p3-a4 zoomAt ตรึงพิกเซลใต้เคอร์เซอร์ — กลางภาพ (ไม่ชนขอบ) ทุกระดับซูม · นอกกลางเมื่อภาพเต็มจอแล้ว', () => {
  const cx = 392.2, cy = 141.7; // ใกล้กึ่งกลาง viewport ของ G2 (เลี่ยงขอบเขตพิกเซลพอดี)
  const before = screenToImage(FIT_VIEW, cx, cy, G2);
  let v = FIT_VIEW;
  for (const f of [1.3, 1.3, 1.5, 2, 3, 2]) {
    v = zoomAt(v, f, cx, cy, G2);
    const p = screenToImage(v, cx, cy, G2);
    assert.deepEqual([p.x, p.y], [before.x, before.y], `หลังซูม z=${v.z.toFixed(2)}`);
  }
  assert.ok(v.z > 10);
  // ซูมเข้ามากแล้ว: เคอร์เซอร์นอกกลางก็ตรึงพิกเซลเดิมได้ (ไม่ชนขอบ)
  const ox = 600, oy = 90;
  const pin = screenToImage(v, ox, oy, G2);
  const v2 = zoomAt(v, 1.2, ox, oy, G2);
  const q = screenToImage(v2, ox, oy, G2);
  assert.deepEqual([q.x, q.y], [pin.x, pin.y]);
});

test('p3-a4b zoomAt ขณะภาพยังเล็กกว่าจอ: ขอบภาพห้ามหลุดจอ (ยอมให้จุดใต้เคอร์เซอร์ขยับเพื่อชดเชยพื้นว่าง)', () => {
  const v = zoomAt(FIT_VIEW, 1.69, 453, 160, G2);
  const r = fitRect(G2.vw, G2.vh, G2.iw, G2.ih);
  const left = v.tx + v.z * r.x;
  assert.ok(left >= -1e-9 && left + v.z * r.w <= G2.vw + 1e-9);
});

test('p3-a5 zoomAt จำกัด z ∈ [1, maxZoom] · ซูมออกสุด = fit เป๊ะ', () => {
  let v = zoomAt(FIT_VIEW, 1e9, 400, 200, G);
  near(v.z, maxZoom(G));
  near(displayScale(v, G), PX_MAX_SCALE, 1e-6);
  v = zoomAt(v, 1e-9, 400, 200, G);
  assert.deepEqual(v, { z: 1, tx: 0, ty: 0 });
  v = zoomAt(FIT_VIEW, 0.5, 400, 200, G); // ซูมออกจาก fit = อยู่ที่ fit
  assert.deepEqual(v, { z: 1, tx: 0, ty: 0 });
});

test('p3-a6 clampView: ภาพที่ใหญ่กว่าจอห้ามเห็นพื้นว่างข้างใน · ภาพยังเล็กกว่าจอห้ามหลุดจอ', () => {
  const big = { z: 4, tx: 9999, ty: -9999 };
  const c = clampView(big, G);
  // แกน x: ภาพกว้าง 640*4 = 2560 > 800 → ขอบซ้ายของภาพ (tx + z*80) ต้อง ≤ 0 และขอบขวา ≥ 800
  assert.ok(c.tx + 4 * 80 <= 1e-9);
  assert.ok(c.tx + 4 * 80 + 2560 >= 800 - 1e-9);
  assert.ok(c.ty <= 1e-9 && c.ty + 400 * 4 >= 400 - 1e-9);
  // z = 1.1: ภาพกว้าง 704 < 800 → ต้องอยู่ในจอทั้งภาพ
  const s = clampView({ z: 1.1, tx: -5000, ty: 0 }, G);
  const left = s.tx + 1.1 * 80;
  assert.ok(left >= -1e-9 && left + 704 <= 800 + 1e-9);
  // ค่าพัง (NaN) ไม่ทำให้ตก
  assert.deepEqual(clampView({ z: NaN, tx: NaN, ty: NaN }, G), { z: 1, tx: 0, ty: 0 });
});

test('p3-a7 panBy เลื่อนตามนิ้ว (ขอบไม่ทะลุ) · ที่ z=1 ไม่ขยับ', () => {
  let v = zoomAt(FIT_VIEW, 4, 400, 200, G);
  const v2 = panBy(v, 30, -20, G);
  near(v2.tx, clampView({ z: v.z, tx: v.tx + 30, ty: v.ty - 20 }, G).tx);
  assert.equal(panBy(FIT_VIEW, 50, 50, G).tx, 0);
  const far = panBy(v, 1e6, 1e6, G);
  assert.ok(far.tx + v.z * 80 <= 1e-9, 'ขอบซ้ายชิดสุดไม่หลุดเข้าข้างใน');
});

test('p3-a8 wheelFactor: เลื่อนขึ้น (deltaY<0) = ซูมเข้า · จำกัดแรง', () => {
  assert.ok(wheelFactor(-100) > 1);
  assert.ok(wheelFactor(100) < 1);
  near(wheelFactor(-100) * wheelFactor(100), 1);
  assert.equal(wheelFactor(-1e6), wheelFactor(-300));
  assert.equal(wheelFactor(0), 1);
  assert.equal(wheelFactor(NaN), 1);
});

test('p3-a9 ตารางพิกเซลเริ่มที่ 800% พอดี (>=) · visiblePixelRange ตรงกับหน้าต่างที่เห็น', () => {
  // สเกลจอ = f*z; G f=1 → z=8 คือ 800%
  assert.equal(PX_GRID_SCALE, 8);
  assert.ok(displayScale({ z: 7.99, tx: 0, ty: 0 }, G) < PX_GRID_SCALE);
  near(displayScale({ z: 8, tx: 0, ty: 0 }, G), 8);
  const v = zoomAt(FIT_VIEW, 8, 400, 200, G);
  const r = visiblePixelRange(v, G);
  near(r.s, 8);
  // จอกว้าง 800 / 8 = 100 พิกเซลภาพ · สูง 400 / 8 = 50
  assert.ok(r.x1 - r.x0 >= 100 && r.x1 - r.x0 <= 101);
  assert.ok(r.y1 - r.y0 >= 50 && r.y1 - r.y0 <= 51);
  // ตำแหน่งจอของเส้นแบ่งพิกเซล x0 ต้องอยู่ที่ขอบจอหรือก่อนหน้านั้น
  assert.ok(r.ox + r.x0 * r.s <= 1e-6);
  assert.ok(r.ox + r.x1 * r.s >= 800 - 1e-6);
  // ที่ fit ครอบทั้งภาพ
  const f = visiblePixelRange(FIT_VIEW, G);
  assert.deepEqual([f.x0, f.x1, f.y0, f.y1], [0, 640, 0, 400]);
});

test('p3-a10 readRGB/formatPixel: เทา vs สี · นอกช่วง = null', () => {
  const data = new Uint8ClampedArray([10, 10, 10, 255, 200, 100, 50, 255]); // 2×1
  const out = [0, 0, 0];
  assert.equal(readRGB(data, 2, 1, 0, 0, out), out);
  assert.equal(formatPixel(out), 'เทา 10');
  readRGB(data, 2, 1, 1, 0, out);
  assert.equal(formatPixel(out), 'R 200  G 100  B 50');
  assert.equal(readRGB(data, 2, 1, 2, 0), null);
  assert.equal(readRGB(data, 2, 1, 0, 1), null);
  assert.equal(readRGB(data, 2, 1, -1, 0), null);
  assert.equal(readRGB(new Uint8ClampedArray(3), 2, 1, 1, 0), null, 'ข้อมูลสั้นกว่าขนาด → null');
  assert.equal(formatPixel(null), '');
  assert.equal(formatPos(12, 34), 'x 12  y 34');
  assert.equal(formatZoomPct(8.004), '800%');
});

test('p3-a11 แผนที่ยกพิกเซลที่ชี้ → ค่าตรงภาพจริงหลังซูม+เลื่อน (เส้นทางครบ: view → พิกเซล → ค่า)', () => {
  // ภาพ 4×2 ไล่เฉด: ค่า R = x*10 + y
  const w = 4, h = 2; const d = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = (y * w + x) * 4; d[i] = x * 10 + y; d[i + 1] = 1; d[i + 2] = 2; d[i + 3] = 255; }
  const g = { vw: 400, vh: 200, iw: w, ih: h }; // f = 100
  let v = zoomAt(FIT_VIEW, 2, 150, 60, g);
  v = panBy(v, -37, 11, g);
  for (const [px, py] of [[10, 10], [200, 100], [390, 190], [123, 45]]) {
    const p = screenToImage(v, px, py, g);
    const rgb = readRGB(d, w, h, p.x, p.y);
    assert.equal(rgb[0], p.x * 10 + p.y);
  }
});

// ===================================================================
// 3.2 histogram
// ===================================================================
test('p3-b1 histogramPlan ตามวิธีที่เลือกจริง (ไม่ใช่ชื่อกล่องหมวด)', () => {
  for (const op of ['binary_threshold', 'otsu_threshold', 'adaptive_threshold']) assert.equal(histogramPlan(op), 'threshold', op);
  for (const op of ['brightness_contrast', 'gamma', 'hist_equalize', 'clahe', 'contrast_stretch', 'rgb_adjust', 'invert', 'white_balance']) assert.equal(histogramPlan(op), 'compare', op);
  for (const op of ['canny', 'segment_edge', 'enhance_tools', 'grayscale', '', undefined]) assert.equal(histogramPlan(op), 'plain', String(op));
});

test('p3-b2 วิธีใน histogram plan มีอยู่จริงใน registry (กันพิมพ์ id ผิด)', () => {
  for (const op of ['binary_threshold', 'otsu_threshold', 'adaptive_threshold', 'brightness_contrast', 'gamma', 'hist_equalize', 'clahe', 'contrast_stretch', 'rgb_adjust', 'invert', 'white_balance']) {
    assert.ok(OPS[op], op + ' ต้องมีใน OPS');
  }
});

test('p3-b3 thresholdMarker: binary = ค่าที่ตั้ง (clamp 0..255, กล่องรวมใช้ params ตัด prefix แล้ว)', () => {
  const props = { mode: 'binary_threshold', binary_threshold__thresh: 90, binary_threshold__invert: false, otsu_threshold__invert: true };
  const local = localParamsFor('segment_edge', props);
  assert.equal(thresholdMarker('binary_threshold', local, []), 90);
  assert.equal(thresholdMarker('binary_threshold', { thresh: 300 }, []), 255);
  assert.equal(thresholdMarker('binary_threshold', { thresh: -4 }, []), 0);
  assert.equal(thresholdMarker('binary_threshold', { thresh: '127' }, []), 127);
  assert.equal(thresholdMarker('binary_threshold', {}, []), null);
  assert.equal(thresholdMarker('binary_threshold', { thresh: 'abc' }, []), null);
});

test('p3-b4 thresholdMarker: adaptive/อื่น ๆ = null (ไม่มีเส้น)', () => {
  assert.equal(thresholdMarker('adaptive_threshold', { blockSize: 11, C: 2 }, ['ค่าเกณฑ์ที่ Otsu คำนวณได้ = 99']), null);
  assert.equal(thresholdMarker('canny', {}, ['x = 5']), null);
});

test('p3-b5 thresholdMarker: otsu อ่านจากข้อความจริงที่ registry สร้าง (ผูกกับเทมเพลตจริง ไม่ใช่สตริงเดา)', () => {
  const fakeCv = {
    Mat: class { }, THRESH_BINARY: 0, THRESH_BINARY_INV: 1, THRESH_OTSU: 8,
    threshold: () => 136.6,
  };
  const ctx = {};
  OPS.otsu_threshold.apply(fakeCv, new fakeCv.Mat(), { invert: false }, ctx);
  assert.ok(ctx.info, 'otsu ต้องรายงาน info');
  assert.equal(thresholdMarker('otsu_threshold', {}, [ctx.info]), 137);
  assert.equal(thresholdMarker('otsu_threshold', {}, []), null);
  assert.equal(thresholdMarker('otsu_threshold', {}, ['ไม่มีตัวเลข']), null);
  assert.equal(thresholdMarker('otsu_threshold', {}, ['ไม่เกี่ยว', 'ค่า = 12.4']), 12);
});

test('p3-b6 เส้นทาง note ราย nodeId: กล่อง Otsu สองกล่องไม่ปนกัน (stepAndNotesOf → infoTextsOf → thresholdMarker)', () => {
  const sel = { id: 11, filterId: 'segment_edge', getParams: () => ({ mode: 'otsu_threshold', otsu_threshold__invert: false }) };
  const notes = [
    { nodeId: 10, id: 'segment_edge', view: 'Otsu Threshold', info: 'ค่าเกณฑ์ที่ Otsu คำนวณได้ = 50' },
    { nodeId: 11, id: 'segment_edge', view: 'Otsu Threshold', info: 'ค่าเกณฑ์ที่ Otsu คำนวณได้ = 137' },
    { nodeId: 11, id: 'segment_edge', view: 'Binary Threshold', info: 'ค่าเกณฑ์ที่ Otsu คำนวณได้ = 9' }, // วิธีเก่าของกล่องเดียวกัน
  ];
  const { notes: mine } = stepAndNotesOf(sel, notes);
  assert.equal(thresholdMarker('otsu_threshold', {}, infoTextsOf(mine)), 137);
});

test('p3-b7 otsuFromHistogram: สองกลุ่มแยกชัด → เกณฑ์อยู่ระหว่างกลุ่ม · ภาพสีเดียว/ว่าง = 0 · ตรงกับ OpenCV', () => {
  const bins = new Uint32Array(256);
  for (let i = 40; i < 60; i++) bins[i] = 100;
  for (let i = 180; i < 200; i++) bins[i] = 100;
  const t = otsuFromHistogram(bins);
  assert.ok(t >= 59 && t < 180, `t=${t}`);
  assert.equal(otsuFromHistogram(new Uint32Array(256)), 0);
  const one = new Uint32Array(256); one[77] = 500;
  assert.equal(otsuFromHistogram(one), 0, 'กลุ่มเดียว → 0');
});

test('p3-b8 otsuFromHistogram = ค่าที่ OpenCV คำนวณจริงบนภาพสุ่ม (±1)', async () => {
  await new Promise((resolve) => { const poll = setInterval(() => { if (cv && typeof cv.Mat === 'function' && typeof cv.threshold === 'function') { clearInterval(poll); resolve(); } }, 30); });
  let seed = 12345; const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const w = 64, h = 48; const m = new cv.Mat(h, w, cv.CV_8UC1);
  const bins = new Uint32Array(256);
  for (let i = 0; i < w * h; i++) { const v = rnd() < 0.4 ? Math.round(60 + rnd() * 40) : Math.round(150 + rnd() * 60); m.data[i] = v; bins[v]++; }
  const dst = new cv.Mat();
  const thr = cv.threshold(m, dst, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  m.delete(); dst.delete();
  assert.ok(Math.abs(otsuFromHistogram(bins) - thr) <= 1, `ours=${otsuFromHistogram(bins)} opencv=${thr}`);
});

test('p3-b9 markerX: 0→ซ้าย 255→ขวา · นอกช่วง clamp', () => {
  assert.equal(markerX(0, 10, 310), 10);
  assert.equal(markerX(255, 10, 310), 310);
  near(markerX(127.5, 10, 310), 160);
  assert.equal(markerX(-20, 10, 310), 10);
  assert.equal(markerX(999, 10, 310), 310);
});

// ===================================================================
// 3.3 ใช้ล่าสุด
// ===================================================================
test('p3-c1 pushRecent: ใหม่สุดก่อน · ไม่ซ้ำ (ย้ายขึ้น) · ไม่เกิน 5 · ไม่แก้ของเดิม', () => {
  const a = ['a', 'b', 'c'];
  const r = pushRecent(a, 'c');
  assert.deepEqual(r, ['c', 'a', 'b']);
  assert.deepEqual(a, ['a', 'b', 'c'], 'ไม่แก้ array เดิม');
  let l = [];
  for (const id of ['1', '2', '3', '4', '5', '6', '7']) l = pushRecent(l, id);
  assert.deepEqual(l, ['7', '6', '5', '4', '3']);
  assert.equal(PALETTE_RECENT_MAX, 5);
  assert.deepEqual(pushRecent(['x'], ''), ['x']);
  assert.deepEqual(pushRecent(['x'], null), ['x']);
});

test('p3-c2 resolveRecent: ค่าพัง/ซ้ำ/ไม่รู้จัก/ไม่ใช่สตริงถูกทิ้ง · จำกัด 5', () => {
  const known = (id) => id.startsWith('k');
  assert.deepEqual(resolveRecent(null, known), []);
  assert.deepEqual(resolveRecent('{bad json', known), []);
  assert.deepEqual(resolveRecent('{"a":1}', known), []);
  assert.deepEqual(resolveRecent(JSON.stringify(['k1', 'zz', 'k1', 5, null, 'k2']), known), ['k1', 'k2']);
  assert.deepEqual(resolveRecent(JSON.stringify(['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7']), known), ['k1', 'k2', 'k3', 'k4', 'k5']);
});

class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.children = []; this.parentNode = null; this._text = ''; this._cls = '';
    this.style = { setProperty() { } }; this._attrs = {}; this.hidden = false; this.title = ''; this.value = ''; this._listeners = {}; this._root = false;
    const self = this;
    this.classList = {
      add(c) { const s = new Set(self._cls.split(' ').filter(Boolean)); s.add(c); self._cls = [...s].join(' '); },
      remove(c) { const s = new Set(self._cls.split(' ').filter(Boolean)); s.delete(c); self._cls = [...s].join(' '); },
      contains(c) { return self._cls.split(' ').includes(c); },
    };
  }
  get className() { return this._cls; }
  set className(v) { this._cls = String(v); }
  get textContent() { return this.children.length ? this.children.map((c) => c.textContent).join('') : this._text; }
  set textContent(v) { this._text = String(v); this.children = []; }
  set innerHTML(v) { this._text = String(v).replace(/<[^>]*>/g, ''); this.children = []; }
  get isConnected() { return this._root || (!!this.parentNode && this.parentNode.isConnected); }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  append(...cs) { for (const c of cs) this.appendChild(c); }
  replaceWith(n) { const p = this.parentNode; const i = p.children.indexOf(this); p.children[i] = n; n.parentNode = p; this.parentNode = null; }
  setAttribute(k, v = '') { this._attrs[k] = String(v); }
  addEventListener(t, fn) { (this._listeners[t] ||= []).push(fn); }
  fire(t, ev = {}) { for (const fn of this._listeners[t] || []) fn(ev); }
  click() { this.fire('click'); }
  focus() { } select() { }
  all(cls, out = []) { for (const c of this.children) { if (c._cls.split(' ').includes(cls)) out.push(c); c.all(cls, out); } return out; }
  querySelector(sel) { return sel.startsWith('.') ? this.all(sel.slice(1))[0] || null : null; }
}
function withDom(fn, store) {
  const had = { doc: Object.prototype.hasOwnProperty.call(globalThis, 'document'), ls: Object.prototype.hasOwnProperty.call(globalThis, 'localStorage') };
  const prev = { doc: globalThis.document, ls: globalThis.localStorage };
  globalThis.document = { createElement: (t) => new El(t) };
  if (store) globalThis.localStorage = store;
  try { return fn(); } finally {
    if (had.doc) globalThis.document = prev.doc; else delete globalThis.document;
    if (store) { if (had.ls) globalThis.localStorage = prev.ls; else delete globalThis.localStorage; }
  }
}
const makeStore = (init) => { const m = new Map(init || []); return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => { m.set(k, String(v)); }, _m: m }; };
function mkPalette(store) {
  const container = new El('div'); container._root = true;
  const added = [];
  const pal = createPalettePanel(container, { onAdd: (f, o) => added.push([f, o]), onAddDisplay() { }, onTemplates() { }, onSaveTemplate() { }, onAddDL() { } });
  pal.render();
  const recent = () => container.all('palette-recent')[0];
  const recentNames = () => (recent() ? recent().all('af-name').map((e) => e.textContent) : []);
  const row = (name) => container.all('tool-row').find((b) => b.all('af-name')[0]?.textContent === name && b.parentNode.className !== 'palette-recent');
  return { container, added, pal, recent, recentNames, row };
}

test('p3-c3 palette: ว่าง = ซ่อน · คลิกเครื่องมือ → ขึ้น "ใช้ล่าสุด" + บันทึก + เรียก onAdd พร้อม opId เดิม', () => {
  const store = makeStore();
  withDom(() => {
    const t = mkPalette(store);
    assert.equal(t.recent().hidden, true, 'ว่าง → ซ่อน');
    const gray = t.row('Grayscale');
    gray.click();
    assert.deepEqual(t.added, [[OP_TO_FILTER.grayscale, 'grayscale']]);
    assert.equal(t.recent().hidden, false);
    assert.deepEqual(t.recentNames(), ['Grayscale']);
    assert.equal(store._m.get(PALETTE_RECENT_KEY), JSON.stringify(['grayscale']));
    t.row('Invert').click();
    t.row('Grayscale').click(); // ซ้ำ → ย้ายขึ้นหน้าสุด ไม่เพิ่ม
    assert.deepEqual(t.recentNames(), ['Grayscale', 'Invert']);
    // คลิกจากแถวใน "ใช้ล่าสุด" ใช้ทางเดียวกัน (opId เดียวกัน)
    const before = t.added.length;
    t.recent().all('tool-row')[1].click(); // Invert
    assert.deepEqual(t.added[before], [OP_TO_FILTER.invert, 'invert']);
    assert.deepEqual(t.recentNames()[0], 'Invert');
  }, store);
});

test('p3-c4 palette: อ่านค่าที่บันทึกไว้ตอนสร้าง · id ที่ไม่รู้จักถูกทิ้ง · localStorage พัง (throw) ไม่ทำแอปล้ม', () => {
  const store = makeStore([[PALETTE_RECENT_KEY, JSON.stringify(['invert', 'ไม่มีอยู่จริง', 'grayscale'])]]);
  withDom(() => {
    const t = mkPalette(store);
    assert.deepEqual(t.recentNames(), ['Invert', 'Grayscale']);
  }, store);
  const broken = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('quota'); } };
  withDom(() => {
    const t = mkPalette(broken);
    assert.equal(t.recent().hidden, true);
    t.row('Invert').click(); // setItem throw ต้องถูกกลืน
    assert.deepEqual(t.recentNames(), ['Invert']);
  }, broken);
});

test('p3-c5 palette: "ใช้ล่าสุด" ถูกกรองด้วยคำค้นเหมือนแถวอื่น (ไม่ตรง → ซ่อนทั้งก้อน)', () => {
  const store = makeStore([[PALETTE_RECENT_KEY, JSON.stringify(['invert', 'grayscale'])]]);
  withDom(() => {
    const t = mkPalette(store);
    const search = t.container.all('palette-search')[0];
    search.value = 'gray'; search.fire('input');
    const rows = t.recent().all('tool-row');
    assert.equal(rows[0].hidden, true, 'Invert ไม่ตรง');
    assert.equal(rows[1].hidden, false, 'Grayscale ตรง');
    assert.equal(t.recent().hidden, false);
    search.value = 'zzzzzz'; search.fire('input');
    assert.equal(t.recent().hidden, true, 'ไม่มีตัวที่ตรง → ซ่อนทั้งก้อน');
    search.value = ''; search.fire('input');
    assert.equal(t.recent().hidden, false);
    assert.ok(rows.every((r) => r.hidden === false));
  }, store);
});

// ===================================================================
// 3.4 พื้นหลังจุด
// ===================================================================
test('p3-d1 dotGridRange: เฉพาะ scale > 0.6 (0.6 เป๊ะ = ไม่วาด) · จุดตรงพิกัดคูณ 24 อยู่ในพื้นที่ที่เห็น', () => {
  const vis = [-100, -50, 400, 300];
  assert.equal(dotGridRange(vis, 0.6), null);
  assert.equal(dotGridRange(vis, 0.59), null);
  assert.ok(dotGridRange(vis, 0.61));
  const r = dotGridRange(vis, 1);
  assert.equal(r.spacing, 24);
  assert.equal(r.x0 % 24 === 0 || r.x0 === 0, true);
  assert.ok(r.x0 >= -100 && r.x0 < -100 + 24);
  assert.ok(r.y0 >= -50 && r.y0 < -50 + 24);
  const lastX = r.x0 + (r.cols - 1) * 24, lastY = r.y0 + (r.rows - 1) * 24;
  assert.ok(lastX <= 300 && lastX + 24 > 300, 'จุดสุดท้ายซ้ายสุดของขอบขวา');
  assert.ok(lastY <= 250 && lastY + 24 > 250);
});

test('p3-d2 dotGridRange: พื้นที่ว่าง/พังไม่วาด · เพดานจำนวนจุด', () => {
  assert.equal(dotGridRange(null, 1), null);
  assert.equal(dotGridRange([0, 0, 0, 100], 1), null);
  assert.equal(dotGridRange([0, 0, 100, -5], 1), null);
  assert.equal(dotGridRange([0, 0, 1e9, 1e9], 1), null, 'ใหญ่เกินแม้เว้น 16 เท่า → ไม่วาด');
  const tiny = dotGridRange([25, 25, 5, 5], 1); // ไม่มีจุดตรง grid ในกรอบเล็ก
  assert.equal(tiny, null);
});

// ===================================================================
// 3.5 สายวิ่ง
// ===================================================================
test('p3-e1 flowActive: เฉพาะกล้องสดโหมดต้นฉบับที่กำลังเดิน ไม่ pause และไม่ขอลดแอนิเมชัน', () => {
  const on = { source: 'camera', cameraMode: 'original', cameraStarted: true, paused: false, reducedMotion: false };
  assert.equal(flowActive(on), true);
  assert.equal(flowActive({ ...on, source: 'image' }), false, 'ภาพนิ่ง');
  assert.equal(flowActive({ ...on, source: 'video' }), false);
  assert.equal(flowActive({ ...on, cameraMode: 'pose' }), false);
  assert.equal(flowActive({ ...on, cameraStarted: false }), false);
  assert.equal(flowActive({ ...on, paused: true }), false);
  assert.equal(flowActive({ ...on, reducedMotion: true }), false);
  assert.equal(flowActive(), false);
});

test('p3-e2 flowT/flowDotCount: อยู่ใน [0,1) เลื่อนไปข้างหน้าตามเวลา · จำนวนจุดตามความยาว', () => {
  for (const t of [0, 1, 799, 1600, 123456, -50]) for (let i = 0; i < 4; i++) {
    const v = flowT(t, i, 4);
    assert.ok(v >= 0 && v < 1, `t=${t} i=${i} v=${v}`);
  }
  near(flowT(0, 0, 4), 0); near(flowT(0, 1, 4), 0.25);
  near(flowT(400, 0, 4, 1600), 0.25);
  assert.ok(flowT(800, 0, 2) > flowT(400, 0, 2));
  assert.equal(flowDotCount(10), 1);
  assert.equal(flowDotCount(210), 3);
  assert.equal(flowDotCount(99999), 6);
});

// ===================================================================
// 3.6 error ราย node
// ===================================================================
test('p3-f1 routeWorkerError/shortErrorText', () => {
  assert.deepEqual(routeWorkerError({ type: 'error', message: 'boom', nodeId: 7 }), { nodeId: 7, text: 'boom' });
  assert.deepEqual(routeWorkerError({ type: 'error', message: 'boom' }), { nodeId: null, text: 'boom' });
  assert.deepEqual(routeWorkerError({ type: 'error', message: 'x', nodeId: undefined }), { nodeId: null, text: 'x' });
  assert.equal(routeWorkerError({ message: 'x', nodeId: '7' }).nodeId, null, 'ไม่ใช่ตัวเลข → ไม่รู้ว่า node ไหน');
  assert.equal(routeWorkerError(null).text, ERROR_FALLBACK_TEXT);
  assert.equal(routeWorkerError({ nodeId: 0, message: 'zero' }).nodeId, 0, 'nodeId = 0 ใช้ได้');
  assert.equal(shortErrorText(undefined), ERROR_FALLBACK_TEXT);
  assert.equal(shortErrorText('undefined'), ERROR_FALLBACK_TEXT);
  assert.equal(shortErrorText('6980656'), ERROR_FALLBACK_TEXT, 'เลขพอยน์เตอร์ดิบของ OpenCV ไม่ใช่ข้อความ');
  assert.equal(shortErrorText(' 42 '), ERROR_FALLBACK_TEXT);
  assert.equal(shortErrorText('code 42'), 'code 42');
  assert.equal(shortErrorText('  a\n  b\t c '), 'a b c');
  const long = 'x'.repeat(500);
  const s = shortErrorText(long, 50);
  assert.equal(s.length, 50);
  assert.ok(s.endsWith('…'));
});

let cvReady;
function waitCv() {
  cvReady ||= new Promise((resolve) => { const poll = setInterval(() => { if (cv && typeof cv.Mat === 'function' && typeof cv.cvtColor === 'function') { clearInterval(poll); resolve(); } }, 30); });
  return cvReady;
}
async function loadWorker() {
  await waitCv();
  globalThis.self = globalThis;
  if (typeof globalThis.ImageData === 'undefined') globalThis.ImageData = class ImageData { constructor(data, width, height) { this.data = data; this.width = width; this.height = height; } };
  return import('../src/renderer/cv/cv-worker.js');
}
function runFail(runPipeline, pipeline) {
  const base = new cv.Mat(20, 30, cv.CV_8UC3, new cv.Scalar(10, 20, 30));
  try {
    runPipeline(base, pipeline, new Map(), new Set(), null);
  } catch (e) { return e; } finally { base.delete(); }
  return null;
}

test('p3-f2 cv-worker: step ที่พังแนบ nodeId ไปกับ error (ข้อความเดิมคงอยู่)', async () => {
  const { runPipeline } = await loadWorker();
  const e = runFail(runPipeline, [
    { id: 'color_tools', params: { mode: 'invert' }, enabled: true, nodeId: 101 },
    { id: 'segment_edge', params: { mode: 'zzz_no_such_op' }, enabled: true, nodeId: 202 },
  ]);
  assert.ok(e, 'ต้องพัง');
  assert.equal(e.nodeId, 202);
  assert.equal(typeof e.message, 'string');
});

test('p3-f3 cv-worker: error ของสาย B (src2) คง nodeId ของ step ชั้นในสุด ไม่ถูกทับด้วยกล่องที่ครอบ', async () => {
  const { runPipeline } = await loadWorker();
  const e = runFail(runPipeline, [{
    id: 'image_arithmetic', params: { mode: 'imabsdiff' }, enabled: true, nodeId: 300,
    src2: { pipeline: [{ id: 'segment_edge', params: { mode: 'zzz_no_such_op' }, enabled: true, nodeId: 301 }], connected: true },
  }]);
  assert.ok(e);
  assert.equal(e.nodeId, 301);
});

test('p3-f4 cv-worker: ค่าที่ throw ไม่ใช่ Error (เช่นตัวเลขจาก WASM) ถูกห่อเป็น Error พร้อม nodeId · step ที่ไม่มี nodeId ไม่พังเพิ่ม', async () => {
  const { runPipeline } = await loadWorker();
  const orig = FILTERS.color_tools.apply;
  try {
    FILTERS.color_tools.apply = () => { throw 12345678; };
    const e = runFail(runPipeline, [{ id: 'color_tools', params: { mode: 'invert' }, enabled: true, nodeId: 404 }]);
    assert.ok(e instanceof Error);
    assert.equal(e.nodeId, 404);
    assert.equal(e.message, '12345678');
    FILTERS.color_tools.apply = () => { throw 7; };
    const e2 = runFail(runPipeline, [{ id: 'color_tools', params: { mode: 'invert' }, enabled: true }]);
    assert.equal(e2, 7, 'ไม่มี nodeId → ส่งต่อค่าเดิมตามเดิม');
  } finally { FILTERS.color_tools.apply = orig; }
});

test('p3-f5 cv-worker: ข้อความ error ของ frame ส่ง nodeId ไปด้วย (อ่านจากซอร์ส — ห้ามหลุดฟิลด์)', () => {
  const src = fs.readFileSync(new URL('../src/renderer/cv/cv-worker.js', import.meta.url), 'utf8');
  assert.match(src, /postMessage\(\{ type: "error", message: err\.message, nodeId: err\.nodeId \}\)/);
});

// ===================================================================
// 3.7 คีย์ลัด
// ===================================================================
const P = (stepIds, extra = {}) => ({ displayId: stepIds[stepIds.length - 1], connected: true, pipeline: [], stepIds, ...extra });

test('p3-g1 nthStepNodeId: เลขตามการ์ด (1=แหล่งภาพ, 2..=Function, สุดท้าย=Display)', () => {
  const pls = [P([1, 5, 6, 7, 2])];
  assert.equal(nthStepNodeId(pls, 1, null), 1);
  assert.equal(nthStepNodeId(pls, 2, null), 5);
  assert.equal(nthStepNodeId(pls, 4, null), 7);
  assert.equal(nthStepNodeId(pls, 5, null), 2);
  assert.equal(nthStepNodeId(pls, 6, null), null, 'เกินจำนวน');
  assert.equal(nthStepNodeId(pls, 0, null), null);
  assert.equal(nthStepNodeId(pls, 1.5, null), null);
  assert.equal(nthStepNodeId(pls, NaN, null), null);
});

test('p3-g2 nthStepNodeId: หลาย Display → สายที่มีกล่องที่เลือก · ไม่เลือก/ไม่อยู่ในสาย = สายแรกที่ต่อสำเร็จ · ข้ามสายที่ไม่ต่อ', () => {
  const a = P([1, 5, 6, 2]);
  const b = P([1, 5, 8, 9, 3]);
  const off = { displayId: 4, connected: false, pipeline: [], stepIds: [] };
  assert.equal(nthStepNodeId([off, a, b], 3, null), 6, 'สายแรกที่ต่อ');
  assert.equal(nthStepNodeId([off, a, b], 3, 9), 8, 'เลือกกล่อง 9 (อยู่สาย b)');
  assert.equal(nthStepNodeId([off, a, b], 3, 6), 6, 'เลือกกล่อง 6 (อยู่สาย a)');
  assert.equal(nthStepNodeId([off, a, b], 3, 999), 6, 'ไม่อยู่ในสายใด → สายแรก');
  assert.equal(nthStepNodeId([off], 2, null), null);
  assert.equal(nthStepNodeId([], 2, null), null);
  assert.equal(nthStepNodeId(undefined, 2, null), null);
  assert.equal(nthStepNodeId([{ displayId: 1, connected: true, pipeline: [] }], 1, null), null, 'ไม่มี stepIds');
  assert.equal(nthStepNodeId([P([0, 1])], 1, 0), 0, 'id = 0 ใช้ได้');
});

test('p3-g3 parseNavKey: F / 1–9 / 0 เมื่อไม่มี modifier · มี modifier หรือกดค้าง = ไม่ทำ', () => {
  const k = (key, extra = {}) => parseNavKey({ key, ...extra });
  assert.deepEqual(k('f'), { type: 'fit' });
  assert.deepEqual(k('F'), { type: 'fit' });
  assert.deepEqual(k('1'), { type: 'step', n: 1 });
  assert.deepEqual(k('9'), { type: 'step', n: 9 });
  assert.deepEqual(k('0'), { type: 'reset-view' });
  for (const m of ['ctrlKey', 'altKey', 'metaKey', 'shiftKey', 'repeat']) assert.equal(k('f', { [m]: true }), null, m);
  for (const key of ['a', ' ', 'm', 'c', 'e', 'ArrowLeft', 'ArrowRight', 'Enter', '10', '', undefined]) assert.equal(k(key), null, String(key));
  assert.equal(parseNavKey(null), null);
});

test('p3-g4 คีย์ใหม่ไม่ชนคีย์เดิมใน app.js (Space/M/C/E/ลูกศร) — และ app.js เรียกหลังคีย์ที่ผู้ใช้ผูกเอง + ข้ามเมื่อโหมดตีกรอบ', () => {
  const src = fs.readFileSync(new URL('../src/renderer/app.js', import.meta.url), 'utf8');
  const m = /const DEFAULT_KEYMAP = \{([^}]*)\}/.exec(src);
  assert.ok(m);
  const bound = [...m[1].matchAll(/:\s*"([^"]+)"/g)].map((x) => x[1].toLowerCase());
  for (const key of ['f', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) assert.ok(!bound.includes(key), `${key} ชนคีย์เดิม`);
  const iLoop = src.indexOf('for (const a of KEY_ACTIONS)');
  const iNav = src.indexOf('parseNavKey(e)');
  const iAnn = src.indexOf('if (annotateCanvas.isActive()) return;', iLoop);
  assert.ok(iLoop > 0 && iAnn > iLoop && iNav > iAnn, 'ลำดับ: keymap ผู้ใช้ → ข้ามโหมดตีกรอบ → คีย์นำทาง');
  // ช่องพิมพ์ถูกตัดก่อนถึงที่นี่ (ด่านเดียวกับ keymap เดิม)
  const iTyping = src.indexOf('t.tagName === "INPUT" || t.tagName === "SELECT"');
  assert.ok(iTyping > 0 && iTyping < iLoop);
});

// ===================================================================
// 3.8 DPR
// ===================================================================
test('p3-h1 canvasBufferSizeDpr: DPR 1 / 1.25 / 1.5 → buffer = css × dpr (ปัดเศษ) · kx/ky คืนพิกัดตรงเป๊ะ', () => {
  for (const dpr of [1, 1.25, 1.5, 2]) {
    for (const rect of [{ width: 747.6, height: 116.575 }, { width: 1013.6, height: 480 }, { width: 400, height: 300 }]) {
      const s = canvasBufferSizeDpr(rect, dpr);
      assert.equal(s.w, Math.round(rect.width * dpr));
      assert.equal(s.h, Math.round(rect.height * dpr));
      // ปลาย CSS ไปตกที่ปลาย buffer พอดี (ภาพที่วาดไม่คลาดจากเมาส์)
      near(rect.width * s.kx, s.w, 1e-9);
      near(rect.height * s.ky, s.h, 1e-9);
      assert.ok(Math.abs(s.kx - dpr) < 0.5 / rect.width + 1e-9, 'kx ≈ dpr');
    }
  }
  const one = canvasBufferSizeDpr({ width: 400, height: 300 }, 1);
  assert.deepEqual([one.w, one.h, one.kx, one.ky], [400, 300, 1, 1], 'DPR 1 = พฤติกรรมเดิมเป๊ะ');
});

test('p3-h2 canvasBufferSizeDpr: ซ่อน/เล็กเกิน = null · dpr ผิดปกติ clamp [1,3] · ไม่มี floor ที่ host เตี้ย', () => {
  assert.equal(canvasBufferSizeDpr({ width: 0, height: 100 }, 1.5), null);
  assert.equal(canvasBufferSizeDpr({ width: 100, height: 0.5 }, 1.5), null);
  assert.equal(canvasBufferSizeDpr(null, 1), null);
  assert.equal(canvasBufferSizeDpr({ width: 100, height: 100 }, 0.75).w, 100, 'ซูมออก (dpr<1) ใช้ buffer = CSS');
  assert.equal(canvasBufferSizeDpr({ width: 100, height: 100 }, 99).w, 300, 'เพดาน 3');
  assert.equal(canvasBufferSizeDpr({ width: 100, height: 100 }, NaN).w, 100);
  assert.equal(canvasBufferSizeDpr({ width: 700, height: 116.6 }, 1.5).h, 175, 'host เตี้ยสุดไม่ถูกดันขึ้นด้วย floor');
});

test('p3-h3 histogramBufferSize: เรขาคณิตตรรกะ 320×172 · ขนาด buffer ตาม CSS×DPR · ซ่อน = ใช้ 320', () => {
  const a = histogramBufferSize(320, 1);
  assert.deepEqual([a.w, a.h, a.scale], [320, 172, 1]);
  const b = histogramBufferSize(400, 1.5);
  assert.equal(b.w, 600); assert.equal(b.h, Math.round(600 * 172 / 320)); near(b.scale, 600 / 320);
  const c = histogramBufferSize(0, 1.25);
  assert.equal(c.w, 400);
  const d = histogramBufferSize(200, 1); // panel แคบกว่า 320 → ไม่ต่ำกว่าเรขาคณิตตรรกะ
  assert.equal(d.w, 320);
});

test('p3-h4 editor.js ใช้ขนาด CSS (ไม่ใช่ buffer) กับ fit/วางกล่อง/ซูมกลางจอ และไม่เหลือ canvasEl.width ในตรรกะ viewport', () => {
  const src = fs.readFileSync(new URL('../src/renderer/graph/editor.js', import.meta.url), 'utf8');
  const uses = [...src.matchAll(/canvasEl\.(width|height)/g)].length;
  assert.equal(uses, 2, 'เหลือเฉพาะ fallback ใน viewW()/viewH()');
  assert.match(src, /ctx\.scale\(this\.scale \* kx, this\.scale \* ky\)/);
  assert.match(src, /canvasBufferSizeDpr\(rect, window\.devicePixelRatio\)/);
  assert.match(src, /matchMedia\(`\(resolution: \$\{window\.devicePixelRatio\}dppx\)`\)/, 'ฟัง DPR เปลี่ยนเมื่อย้ายจอ');
});

// ===================================================================
// รอบแก้ (reviewer): ท่อ preview/ขาเข้า, error clear, Thai layout, flow/DPR ของ editor, inspector, dot thinning, focus
// ===================================================================
test('p3-r1 dotGridRange: พื้นที่กว้างจน > 8000 จุด → เว้นเป็น 2×/4× (เซตย่อยของตารางเดิม) ไม่หายทั้งหมด', () => {
  const vis = [-5000, -5000, 5000, 4000]; // ที่ระยะ 24 = ~208×167 = 34k จุด
  const r = dotGridRange(vis, 0.62);
  assert.ok(r, 'ต้องยังวาด');
  assert.ok(r.cols * r.rows <= 8000);
  assert.ok(r.spacing >= 48 && r.spacing % 24 === 0, 'spacing เป็นพหุคูณของ 24 → จุดตรงตารางเดิม');
  assert.equal(r.x0 % r.spacing === 0 || r.x0 === 0, true);
  const small = dotGridRange([0, 0, 600, 400], 1);
  assert.equal(small.spacing, 24, 'พื้นที่ปกติไม่เว้น');
});

test('p3-r2 parseNavKey: แป้นพิมพ์ไทย — ใช้ e.code (KeyF / Digit0–9 / Numpad0–9) เมื่อ e.key เป็นอักษรไทย', () => {
  const k = (key, code, extra = {}) => parseNavKey({ key, code, ...extra });
  assert.deepEqual(k('ด', 'KeyF'), { type: 'fit' });
  assert.deepEqual(k('ๅ', 'Digit1'), { type: 'step', n: 1 });
  assert.deepEqual(k('๙', 'Digit9'), { type: 'step', n: 9 });
  assert.deepEqual(k('ข', 'Digit0'), { type: 'reset-view' });
  assert.deepEqual(k('5', 'Numpad5'), { type: 'step', n: 5 });
  assert.deepEqual(k('Unidentified', 'Numpad0'), { type: 'reset-view' });
  assert.equal(k('ก', 'KeyA'), null);
  assert.equal(k('ก', 'Digit10'), null);
  assert.equal(k('ด', 'KeyF', { ctrlKey: true }), null, 'modifier ยังกันอยู่');
  assert.equal(k('ด', 'KeyF', { shiftKey: true }), null);
  assert.equal(k('ด', 'KeyF', { repeat: true }), null);
  assert.equal(k('Enter', 'Enter'), null);
  assert.equal(k(' ', 'Space'), null);
});

test('p3-r3 selectedIdOf/buildStepIds: Source อยู่หน้าสุด · ไม่มีแหล่งภาพ → []', () => {
  assert.deepEqual(buildStepIds(1, [5, 6], 2), [1, 5, 6, 2]);
  assert.deepEqual(buildStepIds(1, [], 2), [1, 2]);
  assert.deepEqual(buildStepIds(null, [5], 2), []);
  assert.deepEqual(buildStepIds(1, [5], null), []);
  assert.deepEqual(buildStepIds(0, [], 3), [0, 3], 'id = 0 ใช้ได้');
  assert.equal(selectedIdOf({ selectedNode: { id: 7 }, selectedDisplay: { id: 2 } }), 7);
  assert.equal(selectedIdOf({ selectedNode: null, selectedDisplay: { id: 2 } }), 2);
  assert.equal(selectedIdOf({ selectedNode: null, selectedDisplay: null }), null);
  assert.equal(selectedIdOf({ selectedNode: { id: 0 } }), 0);
  assert.equal(selectedIdOf(null), null);
  // ต่อกับ nthStepNodeId: เลือกกล่องในสาย b → คีย์ 3 ไปสาย b
  const a = { connected: true, stepIds: buildStepIds(1, [5, 6], 2) };
  const b = { connected: true, stepIds: buildStepIds(1, [5, 8, 9], 3) };
  assert.equal(nthStepNodeId([a, b], 3, selectedIdOf({ selectedNode: { id: 9 } })), 8);
  assert.equal(nthStepNodeId([a, b], 3, selectedIdOf({ selectedNode: { id: 6 } })), 6);
  assert.equal(nthStepNodeId([a, b], 1, null), 1);
});

test('p3-r4 buildInputPipeline = preview ตัดขั้นสุดท้าย (ไม่แก้ของเดิม) · ขั้นเดียว → ว่าง (ภาพต้นฉบับ)', () => {
  const prev = { displayId: 'p', pipeline: [{ nodeId: 1 }, { nodeId: 2 }, { nodeId: 3 }] };
  const inp = buildInputPipeline(prev, 'i');
  assert.deepEqual(inp.pipeline.map((x) => x.nodeId), [1, 2]);
  assert.equal(prev.pipeline.length, 3);
  assert.deepEqual([inp.displayId, inp.connected, inp.dl, inp.silentNotes], ['i', true, false, true]);
  assert.deepEqual(buildInputPipeline({ pipeline: [{ nodeId: 1 }] }, 'i').pipeline, []);
  assert.equal(buildInputPipeline(null, 'i'), null);
});

test('p3-r5 pickWorkerPipelines: ลำดับ จอ → ขาเข้า → preview · ภาพนิ่งขอทุกครั้ง · กล้องสด throttle 300 ms', () => {
  const D = [{ displayId: 'd1' }, { displayId: 'd2' }];
  const P = { displayId: 'p' }, I = { displayId: 'i' };
  const ids = (a) => a.map((x) => x.displayId).join(',');
  const base = { displays: D, preview: P, input: I, source: 'image', inputReady: true, inputAt: 1000, now: 1001 };
  assert.equal(ids(pickWorkerPipelines(base)), 'd1,d2,i,p', 'ขาเข้าก่อน preview');
  assert.equal(ids(pickWorkerPipelines({ ...base, source: 'camera', now: 1100 })), 'd1,d2,p', 'กล้อง <300ms → ไม่ขอ');
  assert.equal(ids(pickWorkerPipelines({ ...base, source: 'camera', now: 1300 })), 'd1,d2,i,p', 'กล้อง =300ms → ขอ');
  assert.equal(ids(pickWorkerPipelines({ ...base, source: 'camera', now: 1299 })), 'd1,d2,p');
  assert.equal(ids(pickWorkerPipelines({ ...base, source: 'camera', inputReady: false, now: 1001 })), 'd1,d2,i,p', 'ยังไม่เคยได้ → ขอทันที');
  assert.equal(ids(pickWorkerPipelines({ ...base, input: null })), 'd1,d2,p');
  assert.equal(ids(pickWorkerPipelines({ ...base, preview: null })), 'd1,d2', 'ไม่มี preview → ไม่ส่งขาเข้าลอย ๆ');
  assert.equal(ids(pickWorkerPipelines({ displays: [], preview: null, input: null, source: 'camera', now: 0 })), '');
  const dd = [{ displayId: 'x' }];
  pickWorkerPipelines({ ...base, displays: dd });
  assert.equal(dd.length, 1, 'ไม่แก้ array ของ displays');
});

test('p3-r6 classifyResultId: ขาเข้า/preview/จอ/stale — ขาเข้าห้ามหลุดไปเส้นทางจอ', () => {
  const curI = SELECTED_INPUT_PREFIX + ':5:3', curP = SELECTED_PREVIEW_PREFIX + ':5:3';
  assert.equal(classifyResultId(curI, curI, curP), 'input');
  assert.equal(classifyResultId(curP, curI, curP), 'preview');
  assert.equal(classifyResultId(SELECTED_INPUT_PREFIX + ':5:2', curI, curP), 'stale', 'generation เก่า');
  assert.equal(classifyResultId(SELECTED_PREVIEW_PREFIX + ':5:2', curI, curP), 'stale');
  assert.equal(classifyResultId(curI, undefined, curP), 'stale', 'เลิกขอขาเข้าแล้ว → ทิ้ง ไม่ใช่ display');
  assert.equal(classifyResultId(curP, curI, undefined), 'stale');
  assert.equal(classifyResultId(12, curI, curP), 'display');
  assert.equal(classifyResultId('Display-9', curI, curP), 'display');
  assert.notEqual(SELECTED_INPUT_PREFIX, SELECTED_PREVIEW_PREFIX);
  assert.ok(!SELECTED_INPUT_PREFIX.startsWith(SELECTED_PREVIEW_PREFIX) && !SELECTED_PREVIEW_PREFIX.startsWith(SELECTED_INPUT_PREFIX));
});

test('p3-r7 resolveThresholdT: note ก่อน · Otsu ไม่มี note → คำนวณจาก histogram ขาเข้า · adaptive/binary ไม่ใช้ fallback', () => {
  const bins = new Uint32Array(256);
  for (let i = 40; i < 60; i++) bins[i] = 100;
  for (let i = 180; i < 200; i++) bins[i] = 100;
  const fallback = otsuFromHistogram(bins);
  assert.ok(fallback > 0);
  assert.equal(resolveThresholdT('otsu_threshold', {}, ['ค่า = 77'], bins), 77, 'note ชนะ');
  assert.equal(resolveThresholdT('otsu_threshold', {}, [], bins), fallback);
  assert.equal(resolveThresholdT('otsu_threshold', {}, [], null), null);
  assert.equal(resolveThresholdT('adaptive_threshold', {}, [], bins), null);
  assert.equal(resolveThresholdT('binary_threshold', {}, [], bins), null, 'binary ไม่มีค่าตั้ง → ไม่ใช้ Otsu');
  assert.equal(resolveThresholdT('binary_threshold', { thresh: 90 }, [], bins), 90);
});

const appSrc = () => fs.readFileSync(new URL('../src/renderer/app.js', import.meta.url), 'utf8');
const edSrc = () => fs.readFileSync(new URL('../src/renderer/graph/editor.js', import.meta.url), 'utf8');

test('p3-r8 กฎล้าง error + ลำดับ notes ใน app.js: ล้างเมื่อ result สำเร็จ / กราฟเปลี่ยน · error ตั้งหลังล้าง · notes ตั้งก่อน drawResults', () => {
  const src = appSrc();
  const res = src.indexOf('msg.type === "result"');
  const err = src.indexOf('msg.type === "error"');
  assert.ok(res > 0 && err > res);
  const resBlock = src.slice(res, err);
  const iClear = resBlock.indexOf('editor.clearNodeErrors()');
  const iNotes = resBlock.indexOf('state.notes = msg.notes');
  const iDraw = resBlock.indexOf('drawResults(msg.results');
  assert.ok(iClear >= 0 && iNotes >= 0 && iDraw >= 0);
  assert.ok(iNotes < iDraw, 'notes ต้องตั้งก่อน drawResults (T ของ Otsu ไม่ช้า 1 เฟรม)');
  const errBlock = src.slice(err, err + 1200);
  const iC = errBlock.indexOf('editor.clearNodeErrors()');
  const iS = errBlock.indexOf('editor.setNodeError(');
  assert.ok(iC >= 0 && iS > iC, 'error เก่าถูกล้างก่อนตั้งของใหม่');
  const pc = src.indexOf('onPipelineChange: (pipelines) => {');
  assert.ok(pc > 0 && src.slice(pc, pc + 400).includes('editor.clearNodeErrors()'), 'กราฟเปลี่ยน → ล้าง');
});

test('p3-r9 app.js ต่อสายใช้ helper: pickWorkerPipelines / classifyResultId / selectedIdOf / resolveThresholdT / buildInputPipeline', () => {
  const src = appSrc();
  for (const needle of ['pickWorkerPipelines({', 'classifyResultId(displayId', 'nthStepNodeId(state.pipelines, nav.n, selectedIdOf(state))', 'resolveThresholdT(', 'buildInputPipeline(preview']) {
    assert.ok(src.includes(needle), needle);
  }
  assert.ok(!src.includes('.slice(0, -1)'), 'ไม่มี slice เองใน app.js แล้ว');
  assert.ok(!/const SELECTED_(INPUT|PREVIEW)_PREFIX =/.test(src), 'prefix มีที่เดียว (phase3-math.js)');
});

test('p3-r10 editor.js ใช้ helper: visibleAreaOf / isFlowLink / createFlowLoop / buildStepIds + วาดใหม่หลัง resize', () => {
  const src = edSrc();
  assert.match(src, /visibleAreaOf\(this\.offset, this\.element\.width, this\.element\.height, kx, ky, this\.scale, this\.visible_area\)/);
  assert.match(src, /isFlowLink\(graph\.getNodeById\(link\.origin_id\), graph\.getNodeById\(link\.target_id\)\)/);
  assert.match(src, /createFlowLoop\(\{/);
  assert.match(src, /buildStepIds\(sourceNode\.id, pathNodes\.map/);
  assert.match(src, /canvas\.resize\(w, h\);\s*canvas\.setDirty\(true, true\)/, 'resize แล้ววาดใหม่ (kx/ky อาจเปลี่ยนทั้งที่ buffer เท่าเดิม)');
  assert.match(src, /if \(!link \|\| flow \|\| !flowOn\(\) \|\| !linkInPipeline\(link\)\) return;/, 'จุดวิ่งเฉพาะสายใน pipeline');
});

test('p3-r11 visibleAreaOf: ความสูงหารอัตราส่วน DPR ด้วย (ไม่ใช่แค่ความกว้าง)', () => {
  const v = visibleAreaOf([10, 20], 1500, 750, 1.5, 1.25, 2);
  assert.deepEqual(v, [-10, -20, 500, 300]);
  const out = [0, 0, 0, 0];
  assert.equal(visibleAreaOf([0, 0], 100, 100, 1, 1, 1, out), out);
  assert.deepEqual(visibleAreaOf([0, 0], 1000, 500, 1, 1, 0.5), [-0, -0, 2000, 1000]);
});

test('p3-r12 isFlowLink: เฉพาะเมื่อปลายทั้งสองอยู่ใน pipeline', () => {
  assert.equal(isFlowLink({ __mvInPipeline: true }, { __mvInPipeline: true }), true);
  assert.equal(isFlowLink({ __mvInPipeline: true }, { __mvInPipeline: false }), false);
  assert.equal(isFlowLink({ __mvInPipeline: false }, { __mvInPipeline: true }), false);
  assert.equal(isFlowLink(null, { __mvInPipeline: true }), false);
  assert.equal(isFlowLink({ __mvInPipeline: true }, undefined), false);
});

test('p3-r13 createFlowLoop: เปิด = ขอ RAF + วาดไม่ถี่กว่า minGap · ปิด = ยกเลิก RAF ทันที ไม่มี loop ค้าง', () => {
  let on = true, work = true, draws = 0, stops = 0, nextId = 1; const pending = new Map();
  const loop = createFlowLoop({
    raf: (fn) => { const id = nextId++; pending.set(id, fn); return id; },
    caf: (id) => { pending.delete(id); },
    isOn: () => on, hasWork: () => work, redraw: () => { draws++; }, onStop: () => { stops++; }, minGapMs: 40,
  });
  const step = (ts) => { const [id, fn] = [...pending][0]; pending.delete(id); fn(ts); };
  assert.equal(loop.running(), false);
  loop.sync(); assert.equal(loop.running(), true); assert.equal(pending.size, 1);
  loop.sync(); assert.equal(pending.size, 1, 'sync ซ้ำไม่ขอซ้อน');
  step(100); assert.equal(draws, 1);
  step(120); assert.equal(draws, 1, 'ถี่เกิน → ข้าม');
  step(141); assert.equal(draws, 2);
  work = false; step(300); assert.equal(draws, 2, 'ไม่มีสายใน pipeline → ไม่วาด');
  assert.equal(pending.size, 1, 'ยังวน (รอสายมา)');
  on = false; loop.sync();
  assert.equal(loop.running(), false); assert.equal(pending.size, 0, 'ปิด → ไม่มี RAF ค้าง'); assert.equal(stops, 1);
  on = true; loop.sync(); on = false; step(500);
  assert.equal(pending.size, 0, 'tick เจอว่าปิดแล้ว → ไม่ขอต่อ');
  loop.sync(); assert.equal(stops, 1, 'ปิดอยู่แล้วไม่เรียก onStop ซ้ำ');
});

// ---- createTileInspector (DOM stub) ----
function mkInspector({ w = 4, h = 2, getImage = null, fallback = [9, 8, 7, 255], onFallback = () => {} } = {}) {
  const el = () => ({
    _l: {}, addEventListener(t, f) { this._l[t] = f; }, style: {}, dataset: {}, hidden: false, textContent: '',
    classList: { toggle() {}, add() {}, remove() {}, contains: () => false }, contains: () => false, firstElementChild: { textContent: '' },
    setPointerCapture() {}, releasePointerCapture() {},
  });
  const rect = { left: 0, top: 0, width: 400, height: 200 };
  const viewEl = el(); viewEl.getBoundingClientRect = () => ({ ...rect });
  const canvas = el(); canvas.width = w; canvas.height = h;
  canvas.getContext = () => ({ getImageData: () => { onFallback(); return { data: fallback }; } });
  const gridCanvas = el(); gridCanvas.getContext = () => new Proxy({}, { get: () => () => {}, set: () => true });
  const spans = { '.px-pos': el(), '.px-val': el(), '.px-swatch': el() };
  const readout = el(); readout.hidden = true; readout.querySelector = (q) => spans[q];
  const badge = el(); badge.hidden = true;
  const insp = createTileInspector({ view: viewEl, canvas, gridCanvas, readout, badge }, { getImage: () => getImage && getImage() });
  const ev = (t, e) => viewEl._l[t]({ pointerId: 1, button: 0, preventDefault() {}, target: viewEl, ...e });
  return { insp, viewEl, canvas, readout, spans, ev, rect };
}

test('p3-r14 inspector: เฟรมเปลี่ยนขนาด → มุมมองกลับพอดีจอ (ไม่ค้างซูมเก่า)', () => {
  const t = mkInspector({ w: 40, h: 20 });
  t.insp.onFrame();
  t.ev('pointerenter', {});
  t.ev('wheel', { deltaY: -300, clientX: 200, clientY: 100 });
  assert.ok(t.insp.getView().z > 1);
  assert.notEqual(t.canvas.style.transform, '');
  t.insp.onFrame();
  assert.ok(t.insp.getView().z > 1, 'ขนาดเท่าเดิม → คงซูม');
  t.canvas.width = 80; t.canvas.height = 40;
  t.insp.onFrame();
  assert.deepEqual(t.insp.getView(), { z: 1, tx: 0, ty: 0 });
  assert.equal(t.canvas.style.transform, '');
});

test('p3-r15 inspector: ImageData ที่ขนาดไม่ตรง canvas (ค้างจากเฟรมเก่า) ห้ามถูกอ่าน', () => {
  let fallbacks = 0;
  const stale = { width: 4, height: 2, data: new Uint8ClampedArray(4 * 2 * 4).fill(50) };
  const t = mkInspector({ w: 8, h: 4, getImage: () => stale, onFallback: () => { fallbacks++; } });
  t.insp.onFrame(); t.ev('pointerenter', {});
  t.ev('pointermove', { clientX: 100, clientY: 50 });
  assert.equal(fallbacks, 1, 'ขนาดไม่ตรง → อ่านจาก canvas จริง ไม่ใช่ข้อมูลเก่า');
  assert.equal(t.spans['.px-val'].textContent, 'R 9  G 8  B 7');
  const fresh = { width: 8, height: 4, data: new Uint8ClampedArray(8 * 4 * 4).fill(50) };
  const u = mkInspector({ w: 8, h: 4, getImage: () => fresh, onFallback: () => { throw new Error('ไม่ควรอ่าน canvas'); } });
  u.insp.onFrame(); u.ev('pointerenter', {});
  u.ev('pointermove', { clientX: 100, clientY: 50 });
  assert.equal(u.spans['.px-val'].textContent, 'เทา 50');
  assert.equal(u.readout.hidden, false);
});

test('p3-r16 inspector: ภาพเปลี่ยนขนาดขณะชี้อยู่ → พิกเซลใต้เมาส์คำนวณใหม่จากตำแหน่งล่าสุด', () => {
  const t = mkInspector({ getImage: () => null });
  t.insp.onFrame(); t.ev('pointerenter', {});
  t.ev('pointermove', { clientX: 150, clientY: 50 }); // 4×2, f=100 → x=1,y=0
  assert.equal(t.spans['.px-pos'].textContent, 'x 1  y 0');
  t.canvas.width = 8; t.canvas.height = 4; // f=50 → x=3,y=1
  t.insp.onFrame();
  assert.equal(t.spans['.px-pos'].textContent, 'x 3  y 1');
});

test('p3-r17 inspector: จอที่ไม่มี ImageData (DL) ไม่ getImageData ทุกเฟรม — อ่านเมื่อพิกเซลเปลี่ยนหรือเกิน ~250ms', () => {
  let reads = 0;
  const t = mkInspector({ getImage: () => null, onFallback: () => { reads++; } });
  t.insp.onFrame(); t.ev('pointerenter', {});
  t.ev('pointermove', { clientX: 150, clientY: 50 });
  assert.equal(reads, 1);
  for (let i = 0; i < 30; i++) t.insp.onFrame();
  assert.equal(reads, 1, '30 เฟรมติดกันที่พิกเซลเดิม → ไม่อ่านซ้ำ');
  t.ev('pointermove', { clientX: 250, clientY: 50 });
  assert.equal(reads, 2, 'พิกเซลเปลี่ยน → อ่านทันที');
});

test('p3-r18 palette: เพิ่มเครื่องมือด้วยแถวที่โฟกัสอยู่ → โฟกัสกลับมาที่แถว opId เดียวกันหลังสร้างบล็อกใหม่', () => {
  const store = makeStore([[PALETTE_RECENT_KEY, JSON.stringify(['invert', 'grayscale'])]]);
  withDom(() => {
    let focused = null;
    globalThis.document = { createElement: (t) => { const e = new El(t); e.focus = () => { focused = e; }; return e; }, get activeElement() { return focused; } };
    const t = mkPalette(store);
    const rowsBefore = t.recent().all('tool-row');
    focused = rowsBefore[1]; // Grayscale
    rowsBefore[1].click(); // เพิ่ม → ย้ายขึ้นหน้าสุด → สร้างบล็อกใหม่
    const rowsAfter = t.recent().all('tool-row');
    assert.notEqual(rowsAfter[0], rowsBefore[1], 'สร้างใหม่');
    assert.equal(focused, rowsAfter[0], 'โฟกัสอยู่ที่ Grayscale (ตัวใหม่)');
    assert.equal(rowsAfter[0].all('af-name')[0].textContent, 'Grayscale');
    focused = null; // ไม่มีโฟกัสในบล็อก → ไม่แย่งโฟกัส
    rowsAfter[1].click();
    assert.equal(focused, null);
  }, store);
});

test('p3-r19 cv-worker: สาย B (src2) พัง → คืน Mat ของ step นี้ (converted + current) ไม่รั่ว · nodeId = กล่องในสาย B', async () => {
  const { runPipeline } = await loadWorker();
  const base = new cv.Mat(20, 30, cv.CV_8UC3, new cv.Scalar(10, 20, 30));
  // ขั้นแรกสร้าง current (clone ที่ worker เป็นเจ้าของ) — ดักไว้เพื่อเช็กว่าถูกลบ
  const seen = [];
  const orig = FILTERS.color_tools.apply;
  FILTERS.color_tools.apply = function (cvx, src, p, ctx) { const out = orig.call(this, cvx, src, p, ctx); seen.push(out); return out; };
  let err = null;
  try {
    runPipeline(base, [
      { id: 'color_tools', params: { mode: 'invert' }, enabled: true, nodeId: 100 },
      {
        id: 'image_arithmetic', params: { mode: 'imabsdiff' }, enabled: true, nodeId: 300,
        src2: { pipeline: [{ id: 'segment_edge', params: { mode: 'zzz_no_such_op' }, enabled: true, nodeId: 301 }], connected: true },
      },
    ], new Map(), new Set(), null);
  } catch (e) { err = e; } finally { FILTERS.color_tools.apply = orig; }
  assert.ok(err, 'ต้องพัง');
  assert.equal(err.nodeId, 301, 'กล่องในสาย B ชนะกล่องที่ครอบ');
  assert.equal(seen.length, 1);
  assert.equal(seen[0].isDeleted(), true, 'current ของขั้นก่อนหน้าต้องถูกคืนเมื่อสาย B พัง (เดิมรั่วทุกเฟรม)');
  assert.equal(base.isDeleted(), false, 'baseRGB เป็นของผู้เรียก ห้ามลบ');
  base.delete();
});

test('p3-r20 cv-worker: ข้อยกเว้นตัวเลขของ OpenCV ถอดด้วย exceptionFromPtr เมื่อมี · ไม่มี/พัง → ตัวเลขเดิม (UI จะใช้ข้อความสำรอง)', async () => {
  const { runPipeline } = await loadWorker();
  const orig = FILTERS.color_tools.apply;
  const had = Object.prototype.hasOwnProperty.call(cv, 'exceptionFromPtr');
  const prevEx = cv.exceptionFromPtr;
  try {
    FILTERS.color_tools.apply = () => { throw 6980656; };
    cv.exceptionFromPtr = (p) => ({ msg: 'OpenCV: bad thing #' + p });
    const step = [{ id: 'color_tools', params: { mode: 'invert' }, enabled: true, nodeId: 9 }];
    assert.equal(runFail(runPipeline, step).message, 'OpenCV: bad thing #6980656');
    cv.exceptionFromPtr = () => { throw new Error('no'); };
    assert.equal(runFail(runPipeline, step).message, '6980656');
    delete cv.exceptionFromPtr;
    assert.equal(runFail(runPipeline, step).message, '6980656');
    assert.equal(shortErrorText('6980656'), ERROR_FALLBACK_TEXT, 'แล้ว UI ไม่โชว์ตัวเลขเปล่า');
  } finally {
    FILTERS.color_tools.apply = orig;
    if (had) cv.exceptionFromPtr = prevEx; else delete cv.exceptionFromPtr;
  }
});

test('p3-r21 inspector: tile ขยับตำแหน่งโดยไม่เปลี่ยนขนาด → วัดตำแหน่งใหม่เป็นระยะ (ไม่ใช้ left/top เก่าตลอด)', () => {
  const realNow = Date.now; let now = 1_000_000; Date.now = () => now;
  try {
    const t = mkInspector({ getImage: () => null });
    t.insp.onFrame(); t.ev('pointerenter', {});
    now += 300; t.ev('pointermove', { clientX: 150, clientY: 50 });
    assert.equal(t.spans['.px-pos'].textContent, 'x 1  y 0');
    t.rect.left = 100; // tile ถูกดันไปทางขวา 100px
    now += 100; t.ev('pointermove', { clientX: 250, clientY: 50 });
    assert.equal(t.spans['.px-pos'].textContent, 'x 2  y 0', 'ยังไม่ถึงรอบวัดซ้ำ → ใช้ตำแหน่งเก่า (ไม่วัดทุก mousemove)');
    now += 300; t.ev('pointermove', { clientX: 250, clientY: 50 });
    assert.equal(t.spans['.px-pos'].textContent, 'x 1  y 0', 'ถึงรอบวัดซ้ำ → ใช้ตำแหน่งใหม่');
  } finally { Date.now = realNow; }
});

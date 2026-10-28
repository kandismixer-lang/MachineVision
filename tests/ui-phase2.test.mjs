// 2026-10-05 — เทสต์ของ UI Redesign Phase 2 (usability)
//   2.2 Properties แบบพารามิเตอร์ก่อน (splitReadout / selectStyle / ↺ reset / ตำแหน่งบล็อกค่า / empty state)
//   2.3 Palette (สถานะพับ-กางที่บันทึกไว้ / ค้นหา)
//   2.4 ป้ายซูม + preset มุมมอง · 2.5 planInsert (แทรกหลังกล่องที่เลือก) · carry-over: statusLevel / sourceChipLabel
//
// DOM stub เขียนเองในไฟล์ (ห้ามลง dependency ใหม่ → ไม่มี jsdom) — ติดตั้งใน callback ของแต่ละเทสต์และคืนด้วย try/finally
// (node --test อาจรันหลายไฟล์ในโปรเซสเดียวกัน — ห้ามทิ้ง global รั่ว)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createDetailPanel, splitReadout, selectStyle, differsFromDefault, SEG_MAX_LABEL } from '../src/renderer/ui/detail-panel.js';
import { createPalettePanel, toolMatches, TOOL_SECTIONS, OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { STAGE_PRESETS, WIDTH_PRESETS } from '../src/renderer/ui/layout-resize.js';
import { planInsert, formatZoom, statusLevel, sourceChipLabel } from '../src/renderer/graph/layout-math.js';
import { FILTERS, defaultParams } from '../src/renderer/filters/registry.js';
import fs from 'node:fs';

// ===================================================================
// DOM stub ขั้นต่ำ
// ===================================================================
class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.parentNode = null;
    this._text = '';
    this._cls = '';
    this.style = { setProperty() {} };
    this._attrs = {};
    this.hidden = false;
    this.disabled = false;
    this.title = '';
    this.value = '';
    this.selectedIndex = 0;
    this._listeners = {};
    const self = this;
    this.classList = {
      add(c) { const s = new Set(self._cls.split(' ').filter(Boolean)); s.add(c); self._cls = [...s].join(' '); },
      remove(c) { const s = new Set(self._cls.split(' ').filter(Boolean)); s.delete(c); self._cls = [...s].join(' '); },
      contains(c) { return self._cls.split(' ').includes(c); },
      toggle(c, on) { const want = on === undefined ? !this.contains(c) : !!on; if (want) this.add(c); else this.remove(c); },
    };
  }
  get className() { return this._cls; }
  set className(v) { this._cls = String(v); }
  get textContent() { return this.children.length ? this.children.map((c) => c.textContent).join('') : this._text; }
  set textContent(v) { this._text = String(v); this.children = []; }
  set innerHTML(v) { this._text = String(v).replace(/<[^>]*>/g, ''); this.children = []; }
  get innerHTML() { return this._text; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  append(...cs) { for (const c of cs) this.appendChild(c); }
  setAttribute(k, v = '') { this._attrs[k] = String(v); }
  getAttribute(k) { return Object.prototype.hasOwnProperty.call(this._attrs, k) ? this._attrs[k] : null; }
  addEventListener(t, fn) { (this._listeners[t] ||= []).push(fn); }
  fire(t, ev = {}) { for (const fn of this._listeners[t] || []) fn(ev); }
  click() { this.fire('click'); }
  focus() { env.focused = this; }
  select() {}
  all(cls, out = []) {
    for (const c of this.children) {
      if (c._cls.split(' ').includes(cls)) out.push(c);
      c.all(cls, out);
    }
    return out;
  }
  one(cls) { return this.all(cls)[0] || null; }
  querySelector(sel) { return sel.startsWith('.') ? this.one(sel.slice(1)) : null; }
}

const env = { focused: null, docListeners: [], modal: null };

// เรียกฟังก์ชันภายใต้ document/localStorage ปลอม แล้วคืนของจริงเสมอ
function withDom(fn, store) {
  const had = { doc: Object.prototype.hasOwnProperty.call(globalThis, 'document'), ls: Object.prototype.hasOwnProperty.call(globalThis, 'localStorage') };
  const prev = { doc: globalThis.document, ls: globalThis.localStorage, now: Date.now };
  env.focused = null;
  env.docListeners = [];
  env.modal = null;
  globalThis.document = { createElement: (t) => new El(t), querySelector: (sel) => sel === '.modal-overlay' ? env.modal : null, addEventListener: (t, fn, capture) => { env.docListeners.push([t, fn, capture]); }, get activeElement() { return env.focused; } };
  if (store) globalThis.localStorage = store;
  // เวลาเดินหน้าเอง 1 วินาทีต่อการเรียก → throttle ของ detail-panel ทาสีทันที ไม่ทิ้ง setTimeout จริงค้าง
  let t = 5_000_000;
  Date.now = () => (t += 1000);
  try {
    return fn();
  } finally {
    Date.now = prev.now;
    if (had.doc) globalThis.document = prev.doc; else delete globalThis.document;
    if (store) { if (had.ls) globalThis.localStorage = prev.ls; else delete globalThis.localStorage; }
  }
}

function makeStore(initial) {
  const m = new Map(initial || []);
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => { m.set(k, String(v)); }, _m: m };
}

function makeRig(step, notes, extra = {}) {
  const container = new El('div');
  const rig = { container, step, changes: [], templates: 0 };
  rig.panel = createDetailPanel(container, {
    onParamChange: (k, v) => { rig.changes.push([k, v]); rig.step.params[k] = v; },
    ...(extra.noTemplates ? {} : { onTemplates: () => { rig.templates++; } }),
  });
  rig.render = (n = notes) => rig.panel.render(rig.step, n || []);
  return rig;
}
const topSeq = (rig) => rig.container.children.map((c) => c.className.split(' ')[0]);

// ===================================================================
// 2.2 splitReadout
// ===================================================================
test('p2-a1 splitReadout: รูปแบบ "ป้าย = ตัวเลข"', () => {
  assert.deepEqual(splitReadout('ค่าเกณฑ์ที่ Otsu คำนวณได้ = 137'), { label: 'ค่าเกณฑ์ที่ Otsu คำนวณได้', value: '137', unit: '' });
  assert.deepEqual(splitReadout('ระยะไกลสุดจากขอบ = 12.3 px'), { label: 'ระยะไกลสุดจากขอบ', value: '12.3', unit: 'px' });
  assert.deepEqual(splitReadout('ค่าเทียบ: -4'), { label: 'ค่าเทียบ', value: '-4', unit: '' });
});

test('p2-a2 splitReadout: รูปแบบ "ป้าย ตัวเลข หน่วย"', () => {
  assert.deepEqual(splitReadout('พบวัตถุ 5 ชิ้น'), { label: 'พบวัตถุ', value: '5', unit: 'ชิ้น' });
  assert.deepEqual(splitReadout('ภาพขยับ 3.6 px'), { label: 'ภาพขยับ', value: '3.6', unit: 'px' });
  assert.deepEqual(splitReadout('พบเส้นตรง 12 เส้น'), { label: 'พบเส้นตรง', value: '12', unit: 'เส้น' });
});

test('p2-a3 splitReadout: ไม่เข้ารูปแบบ → null (ให้แสดงแถวข้อความเดิม)', () => {
  assert.equal(splitReadout('PSNR'), null);
  assert.equal(splitReadout(''), null);
  assert.equal(splitReadout(null), null);
  assert.equal(splitReadout('ภาพว่าง'), null);
  assert.equal(splitReadout('= 137'), null, 'ไม่มีป้าย → ไม่ทำ tile');
  assert.equal(splitReadout('ค่า = . ชิ้น'), null, 'ไม่มีหลักตัวเลขจริง');
  assert.equal(splitReadout('เกณฑ์ที่คำนวณได้ = 12, 200'), null, 'หลายค่าปนในหน่วย → ไม่ใช่ tile');
  assert.equal(splitReadout('ห่อวัตถุ 5 ชิ้นด้วยรูปนูน · solidity รวม 0.98 (1.0 = นูนเต็มที่)'), null);
  assert.equal(splitReadout('ข้อความที่เปลี่ยนทุกเฟรม #7'), null);
});

// ===================================================================
// 2.2 selectStyle / differsFromDefault
// ===================================================================
test('p2-b1 selectStyle: mode เป็น <select> เสมอ · ≤3 ตัวเลือกสั้นเป็น segmented · อื่น ๆ <select>', () => {
  const o = (...labels) => labels.map((l, i) => ({ value: 'v' + i, label: l }));
  assert.equal(selectStyle({ key: 'mode', options: o('ก', 'ข') }), 'select');
  assert.equal(selectStyle({ key: 'x', options: o('ก', 'ข', 'ค') }), 'seg');
  assert.equal(selectStyle({ key: 'x', options: o('ก', 'ข', 'ค', 'ง') }), 'select', '> 3 ตัวเลือก');
  assert.equal(selectStyle({ key: 'x', options: o('ก', 'x'.repeat(SEG_MAX_LABEL + 1)) }), 'select', 'ป้ายยาว');
  assert.equal(selectStyle({ key: 'x', options: o('ก', 'x'.repeat(SEG_MAX_LABEL)) }), 'seg', 'ป้ายยาวพอดีเพดาน');
  assert.equal(selectStyle({ key: 'x', options: [] }), 'select');
  // ของจริงใน registry: ทิศ Sobel (3 ตัวเลือกสั้น) → seg · mode ของ edge_detection → select
  const ed = FILTERS.edge_detection.params;
  assert.equal(selectStyle(ed.find((p) => p.key === 'sobel__dir')), 'seg');
  assert.equal(selectStyle(ed.find((p) => p.key === 'mode')), 'select');
});

test('p2-b2 differsFromDefault: ทน float · ไม่มี default → false', () => {
  assert.equal(differsFromDefault(5, 5), false);
  assert.equal(differsFromDefault('5', 5), false);
  assert.equal(differsFromDefault(0.1 + 0.2, 0.3), false);
  assert.equal(differsFromDefault(6, 5), true);
  assert.equal(differsFromDefault(5, undefined), false);
  assert.equal(differsFromDefault(5, null), false);
});

// ===================================================================
// 2.2 detail-panel (DOM stub)
// ===================================================================
const otsuFilter = OP_TO_FILTER.otsu_threshold;
const otsuStep = () => ({ id: otsuFilter, params: { ...defaultParams(otsuFilter), mode: 'otsu_threshold' } });

test('p2-c1 note-info → readout tile (ป้าย/ตัวเลข/หน่วย) + อัปเดตในที่ไม่ rebuild · ไม่เข้ารูป → แถว 📊 เดิม', () => withDom(() => {
  const notes = (t) => [{ id: otsuFilter, info: t }];
  const rig = makeRig(otsuStep(), notes('ค่าเกณฑ์ที่ Otsu คำนวณได้ = 137'));
  rig.render();
  const tile = rig.container.one('detail-note-info');
  assert.ok(tile.className.includes('readout'));
  assert.equal(tile.one('rt-label').textContent, 'ค่าเกณฑ์ที่ Otsu คำนวณได้');
  assert.equal(tile.one('rt-value').textContent, '137');
  // อัปเดตในที่: ไม่ rebuild (applyNotes คืน true) และ DOM เดิมถูกเขียนทับ
  const valueEl = tile.one('rt-value');
  assert.equal(rig.panel.applyNotes(notes('ค่าเกณฑ์ที่ Otsu คำนวณได้ = 98')), true);
  assert.equal(valueEl.textContent, '98', 'tile ต้องอัปเดตในที่');
  // สลับเป็นข้อความที่ไม่เข้ารูป → แถวข้อความเดิม (📊) บนองค์ประกอบเดิม
  rig.panel.updateInfo(['ภาพว่าง']);
  assert.equal(tile.textContent, '📊 ภาพว่าง');
  assert.ok(!tile.className.includes('readout'));
  // และกลับเป็น tile ได้อีก
  rig.panel.updateInfo(['พบวัตถุ 5 ชิ้น']);
  assert.equal(tile.one('rt-value').textContent, '5');
  assert.equal(tile.one('rt-unit').textContent, 'ชิ้น');
}));

test('p2-c2 บล็อก "ค่าที่วัดได้" อยู่ใต้พารามิเตอร์ ทั้งทางที่มี param และทางไม่มี param (SSIM/PSNR ไม่หาย)', () => withDom(() => {
  // ทางมี param: regionprops (table)
  const rp = makeRig({ id: 'shape_analysis', params: { ...defaultParams('shape_analysis'), mode: 'regionprops' } }, [
    { id: 'shape_analysis', info: 'พบวัตถุ 2 ชิ้น' },
    { id: 'shape_analysis', values: { kind: 'table', columns: ['index', 'area'], rows: [[1, 10], [2, 20]], total: 2 } },
  ]);
  rp.render();
  const a = topSeq(rp);
  assert.ok(a.includes('dv'), 'ต้องมีบล็อกค่า');
  assert.ok(a.indexOf('params-head') < a.indexOf('readouts'), 'tile อยู่ใต้หัว "พารามิเตอร์"');
  assert.ok(a.lastIndexOf('param') < a.indexOf('dv'), 'บล็อกค่าอยู่ใต้ทุกพารามิเตอร์');
  assert.ok(a.indexOf('readouts') < a.indexOf('dv'));
  // ทางไม่มี param: PSNR
  assert.equal(FILTERS.psnr.params.length, 0);
  const ps = makeRig({ id: 'psnr', params: {} }, [
    { id: 'psnr', values: { kind: 'scalars', fields: [{ key: 'psnr_db', label: 'PSNR (dB)', value: 31.2 }] } },
  ]);
  ps.render();
  const b = topSeq(ps);
  assert.ok(b.includes('dv') && b.includes('param-hint'));
  assert.ok(b.indexOf('param-hint') < b.indexOf('dv'));
  // SSIM (มี param แต่ไม่มี showIf)
  const ss = makeRig({ id: 'ssim', params: { ...defaultParams('ssim') } }, [
    { id: 'ssim', values: { kind: 'scalars', fields: [{ key: 'ssim', label: 'SSIM', value: 0.9 }] } },
  ]);
  ss.render();
  assert.ok(topSeq(ss).includes('dv'));
}));

test('p2-c3 หัว "ค่าที่วัดได้" ของ tile ซ่อนเมื่อมีบล็อกค่า (ไม่ซ้ำ) และโชว์เมื่อไม่มีบล็อกค่า', () => withDom(() => {
  const only = makeRig(otsuStep(), [{ id: otsuFilter, info: 'ค่าเกณฑ์ที่ Otsu คำนวณได้ = 137' }]);
  only.render();
  assert.equal(only.container.one('readouts-head').hidden, false);
  const both = makeRig({ id: 'shape_analysis', params: { ...defaultParams('shape_analysis'), mode: 'regionprops' } }, [
    { id: 'shape_analysis', info: 'พบวัตถุ 1 ชิ้น' },
    { id: 'shape_analysis', values: { kind: 'table', columns: ['index'], rows: [[1]], total: 1 } },
  ]);
  both.render();
  assert.equal(both.container.one('readouts-head').hidden, true);
}));

// หา param ตัวเลขตัวแรกที่ไม่มี showIf ใน filter ที่ไม่มี mode (ไม่ผูกกับ filter ตัวใดตัวหนึ่ง)
function findNumberParam() {
  for (const [id, f] of Object.entries(FILTERS)) {
    if (f.params.some((p) => p.key === 'mode')) continue;
    const p = f.params.find((q) => (!q.type || q.type === 'number') && !q.showIf && typeof q.min === 'number' && typeof q.max === 'number' && q.max > q.min);
    if (p) return { id, p };
  }
  throw new Error('ไม่พบ param ตัวเลขใน registry');
}

test('p2-c4 ↺ รีเซ็ต: ซ่อนเมื่อเท่า default · โชว์เมื่อต่าง · คลิกแล้วคืนค่า + แจ้ง onParamChange', () => withDom(() => {
  const { id, p } = findNumberParam();
  const def = defaultParams(id)[p.key];
  const rig = makeRig({ id, params: { ...defaultParams(id) } }, []);
  rig.render();
  const wrap = rig.container.all('param-num').find((w) => w.one('pn-label').textContent === p.label);
  assert.ok(wrap, 'ไม่พบแถวพารามิเตอร์ตัวเลข');
  const reset = wrap.one('pn-reset'), slider = wrap.one('pn-slider'), box = wrap.one('pn-value');
  assert.equal(reset.disabled, true);
  assert.ok(reset.className.includes('is-off'));
  // ขยับสไลเดอร์ไปค่าอื่น
  const other = def === p.max ? p.min : p.max;
  slider.value = String(other);
  slider.fire('input');
  assert.equal(reset.disabled, false);
  assert.ok(!reset.className.includes('is-off'));
  assert.deepEqual(rig.changes.at(-1), [p.key, other]);
  // คลิก ↺
  reset.click();
  assert.deepEqual(rig.changes.at(-1), [p.key, def]);
  assert.equal(Number(slider.value), def);
  assert.equal(Number(box.value), def);
  assert.equal(reset.disabled, true);
  // min/max ปลายสไลเดอร์
  const range = wrap.one('pn-range');
  assert.equal(range.children[0].textContent, String(p.min));
  assert.equal(range.children[1].textContent, String(p.max));
}));

test('p2-c5 "วิธี" (mode) เป็น <select> · เลือกแล้วแจ้ง onParamChange + คืนโฟกัสให้ select ตัวใหม่', () => withDom(() => {
  const rig = makeRig({ id: 'shape_analysis', params: { ...defaultParams('shape_analysis') } }, []);
  rig.render();
  const sel = rig.container.one('field-select');
  assert.ok(sel, 'ต้องมี select วิธี');
  assert.equal(rig.container.all('choice').length, 0, 'ไม่มีรายการปุ่ม choice แล้ว');
  const modeParam = FILTERS.shape_analysis.params.find((p) => p.key === 'mode');
  assert.equal(sel.children.length, modeParam.options.length);
  const curIdx = Math.max(0, modeParam.options.findIndex((o) => o.value === rig.step.params.mode));
  assert.equal(sel.children[curIdx].selected, true, 'ตัวเลือกปัจจุบันต้องถูกเลือก');
  const next = (curIdx + 1) % modeParam.options.length;
  sel.focus(); // ผู้ใช้โฟกัส select อยู่ (activeElement)
  sel.value = sel.children[next].value;
  sel.fire('change');
  assert.deepEqual(rig.changes.at(-1), ['mode', modeParam.options[next].value]);
  const sel2 = rig.container.one('field-select');
  assert.notEqual(sel2, sel, 'แผงต้องถูกสร้างใหม่');
  assert.equal(env.focused, sel2, 'โฟกัสต้องกลับมาที่ select ตัวใหม่');
  assert.equal(sel2.children[next].selected, true);
}));

test('p2-c6 segmented control (ทิศ Sobel): 3 ปุ่ม · aria-pressed ตามค่า · คลิกแล้วแจ้ง onParamChange', () => withDom(() => {
  const rig = makeRig({ id: 'edge_detection', params: { ...defaultParams('edge_detection'), mode: 'sobel' } }, []);
  rig.render();
  const seg = rig.container.one('seg-control');
  assert.ok(seg, 'ต้องมี segmented control');
  const dirParam = FILTERS.edge_detection.params.find((p) => p.key === 'sobel__dir');
  const btns = seg.all('seg-btn');
  assert.equal(btns.length, dirParam.options.length);
  const cur = btns.findIndex((b) => b.getAttribute('aria-pressed') === 'true');
  assert.equal(dirParam.options[cur].value, rig.step.params.sobel__dir);
  const other = btns.findIndex((_, i) => i !== cur);
  btns[other].focus(); // คลิกแล้วปุ่มได้โฟกัส
  btns[other].click();
  assert.deepEqual(rig.changes.at(-1), ['sobel__dir', dirParam.options[other].value]);
  const seg2 = rig.container.one('seg-control');
  const pressed2 = seg2.all('seg-btn').findIndex((b) => b.getAttribute('aria-pressed') === 'true');
  assert.equal(pressed2, other);
  assert.equal(env.focused, seg2.all('seg-btn')[other], 'โฟกัสไปที่ปุ่มที่เลือกใหม่');
}));

test('p2-c7 empty state: มีปุ่ม "เริ่มจากเทมเพลต" เมื่อส่ง callback · ไม่ส่ง = ไม่มีปุ่ม', () => withDom(() => {
  const rig = makeRig(null, []);
  rig.panel.render(null, null);
  const btn = rig.container.one('de-template');
  assert.ok(btn, 'ต้องมีปุ่มเทมเพลต');
  assert.ok(btn.textContent.includes('เริ่มจากเทมเพลต'));
  btn.click();
  assert.equal(rig.templates, 1);
  const bare = makeRig(null, [], { noTemplates: true });
  bare.panel.render(null, null);
  assert.equal(bare.container.one('de-template'), null);
}));

// ===================================================================
// 2.3 Palette
// ===================================================================
const TITLES = TOOL_SECTIONS.map((s) => s.title);

test('p2-d3 toolMatches: ไม่สนตัวพิมพ์ · ค้นจากชื่อและรหัส · ว่าง = ตรงทั้งหมด', () => {
  assert.equal(toolMatches('Otsu Threshold', 'otsu_threshold', ''), true);
  assert.equal(toolMatches('Otsu Threshold', 'otsu_threshold', '  '), true);
  assert.equal(toolMatches('Otsu Threshold', 'otsu_threshold', 'OTSU'), true);
  assert.equal(toolMatches('Gaussian Blur', 'gaussian_blur', 'blur'), true);
  assert.equal(toolMatches('Binary Threshold', 'binary_threshold', 'binary_th'), true, 'ค้นจากรหัสภายได้');
  assert.equal(toolMatches('Canny Edge', 'canny', 'otsu'), false);
});

// ---- palette แบบ popup (แทน accordion เดิม) ----
const DOC_ESC = { key: 'Escape', preventDefault() {}, stopImmediatePropagation() {} };
function mkPal() {
  const container = new El('div');
  const added = [];
  const pal = createPalettePanel(container, { onAdd: (f, o) => added.push([f, o]), onAddDisplay() {}, onTemplates() {}, onSaveTemplate() {}, onAddDL() {} });
  pal.render();
  const secs = container.all('pal-section');
  const popup = () => container.one('pal-popover'); // ไม่มี document.body ใน DOM ปลอม → popup อยู่ใต้ container
  const popRows = () => popup().one('pal-list').all('tool-row');
  const popNames = () => popRows().map((r) => r.one('af-name').textContent);
  const fireDoc = (type, ev) => env.docListeners.filter(([t]) => t === type).forEach(([, fn]) => fn(ev));
  return { container, added, pal, secs, popup, popRows, popNames, fireDoc };
}
// จำนวนเครื่องมือจริงของหมวด (op ที่ resolve ได้บน palette)
const toolCount = (sec) => sec.ops.filter((id) => OP_TO_FILTER[id]).length;

test('p2-d4 palette DOM: เริ่มเหลือแต่หัวหมวด (ไม่มี popup/รายการ inline) · คลิกหัวหมวดเปิด popup ของหมวดนั้นพร้อมเครื่องมือครบทุกตัว · คลิกซ้ำปิด', () => {
  withDom(() => {
    const t = mkPal();
    assert.equal(t.secs.length, TITLES.length);
    assert.ok(t.secs.every((s) => s.one('pal-list').hidden), 'ไม่ค้น → ไม่มีรายการ inline');
    assert.equal(t.popup().hidden, true, 'popup เริ่มปิด (ไม่จำข้ามการเปิดแอป)');
    assert.ok(t.secs.every((s) => s.one('pal-head').getAttribute('aria-expanded') === 'false'));
    // จำนวนในหัวหมวด = จำนวนเครื่องมือของหมวด
    TOOL_SECTIONS.forEach((sec, i) => assert.equal(t.secs[i].one('pal-count').textContent, String(toolCount(sec))));

    t.secs[1].one('pal-head').click(); // ENHANCE
    assert.equal(t.popup().hidden, false);
    assert.equal(t.popup().one('pal-pop-title').textContent, 'ENHANCE');
    assert.equal(t.popup().one('pal-pop-count').textContent, String(toolCount(TOOL_SECTIONS[1])));
    assert.equal(t.popRows().length, toolCount(TOOL_SECTIONS[1]), 'ครบทุกตัว ไม่มี Show all');
    assert.equal(t.popup().all('tool-more').length, 0);
    assert.equal(t.secs[1].one('pal-head').getAttribute('aria-expanded'), 'true');
    assert.equal(t.secs[0].one('pal-head').getAttribute('aria-expanded'), 'false');

    t.secs[1].one('pal-head').click(); // คลิกซ้ำ → ปิด
    assert.equal(t.popup().hidden, true);
    assert.equal(t.secs[1].one('pal-head').getAttribute('aria-expanded'), 'false');
  }, makeStore());
});

test('p2-d4b popup: เปิดทีละหมวด (คลิกหมวดอื่น = สลับ) · Esc / กดนอก popup+palette / ปุ่ม × ปิด · กดใน palette ไม่ปิด', () => {
  withDom(() => {
    const t = mkPal();
    t.secs[0].one('pal-head').click();
    t.secs[3].one('pal-head').click(); // สลับไป FILTER & DENOISE
    assert.equal(t.popup().hidden, false);
    assert.equal(t.popup().one('pal-pop-title').textContent, 'FILTER & DENOISE');
    assert.equal(t.popRows().length, toolCount(TOOL_SECTIONS[3]));
    assert.deepEqual(t.secs.map((s) => s.one('pal-head').getAttribute('aria-expanded')), TOOL_SECTIONS.map((s) => String(s.title === 'FILTER & DENOISE')));

    t.fireDoc('keydown', { key: 'a' }); // ปุ่มอื่นไม่ปิด
    assert.equal(t.popup().hidden, false);
    t.fireDoc('keydown', DOC_ESC);
    assert.equal(t.popup().hidden, true, 'Esc ปิด');

    t.secs[2].one('pal-head').click();
    t.fireDoc('pointerdown', { target: t.secs[5].one('pal-head') }); // กดในพื้นที่ palette → ไม่ปิด
    t.fireDoc('pointerdown', { target: t.popup().one('af-name') }); // กดใน popup → ไม่ปิด
    assert.equal(t.popup().hidden, false);
    t.fireDoc('pointerdown', { target: new El('canvas') }); // กดนอก → ปิด
    assert.equal(t.popup().hidden, true);

    t.secs[2].one('pal-head').click();
    t.popup().one('pal-pop-close').click();
    assert.equal(t.popup().hidden, true, 'ปุ่ม × ปิด');
  }, makeStore());
});

test('p2-d4c popup: คลิก tile = onAdd(filterId, opId) แล้ว popup ยังเปิด เพิ่มได้หลายตัว', () => {
  withDom(() => {
    const t = mkPal();
    t.secs[1].one('pal-head').click();
    t.popRows()[0].click();
    t.popRows()[1].click();
    assert.deepEqual(t.added, [['enhance_tools', 'brightness_contrast'], ['enhance_tools', 'gamma']]);
    assert.equal(t.popup().hidden, false, 'เพิ่มแล้วต้องไม่ปิด');
  }, makeStore());
});

test('p2-d4d popup: เปิดด้วยคีย์บอร์ด → โฟกัสไปที่ tile แรกใน popup · Esc กลับไปที่หัวหมวด · เมาส์ไม่ย้ายโฟกัส · aria ครบ', () => {
  withDom(() => {
    const t = mkPal();
    const head = t.secs[1].one('pal-head');
    assert.equal(head.getAttribute('aria-haspopup'), 'dialog');
    assert.equal(head.getAttribute('aria-controls'), t.popup().id);
    head.click(); // เมาส์ (ไม่มี detail 0)
    assert.equal(env.focused, null);
    head.click(); // ปิด
    head.fire('click', { detail: 0 }); // คีย์บอร์ด
    assert.equal(env.focused, t.popRows()[0]);
    t.fireDoc('keydown', DOC_ESC);
    assert.equal(t.popup().hidden, true);
    assert.equal(env.focused, head, 'Esc คืนโฟกัสให้หัวหมวด');
  }, makeStore());
});

test('p2-d4d2 Esc: listener เดียวใน capture · ปิดเฉพาะชั้น palette ที่มีโฟกัส · modal ได้สิทธิ์ก่อน', () => {
  withDom(() => {
    const t = mkPal();
    t.pal.render(); // render ซ้ำไม่ผูก listener เพิ่ม
    const heads = t.container.all('pal-head');
    assert.deepEqual(env.docListeners.filter(([type]) => type === 'keydown').map(([, , capture]) => capture), [true]);
    assert.ok(heads.every((head) => !head._listeners.keydown));
    assert.ok(!t.popup()._listeners.keydown);
    const head = heads[1];
    const escape = () => {
      const calls = [];
      t.fireDoc('keydown', { key: 'Escape', preventDefault() { calls.push('prevent'); }, stopImmediatePropagation() { calls.push('stop'); } });
      return calls;
    };
    assert.deepEqual(escape(), [], 'popup ปิดอยู่ → ไม่กิน Esc');
    head.click();
    new El('canvas').focus();
    assert.deepEqual(escape(), [], 'โฟกัสนอก palette → ให้ชั้นอื่นรับ Esc');
    assert.equal(t.popup().hidden, false);
    head.focus();
    env.modal = new El('div');
    assert.deepEqual(escape(), [], 'มี modal → ไม่กิน Esc');
    assert.equal(t.popup().hidden, false);
    env.modal = null;
    // โฟกัสในหัวหมวด / ช่องค้นหา / popup / body / ไม่มีโฟกัส → ปิดและหยุด Esc
    document.body = new El('body');
    head.click(); // ปิด popup ก่อนเริ่มแต่ละกรณี
    for (const focus of [() => head, () => t.container.one('palette-search'), () => t.popRows()[0], () => document.body, () => null]) {
      head.click();
      const active = focus();
      env.focused = active;
      assert.deepEqual(escape(), ['prevent', 'stop']);
      assert.equal(t.popup().hidden, true);
      if (active === head || active?.className === 'tool-row') assert.equal(env.focused, head);
    }
  }, makeStore());
});

test('p2-d4e ค้นหา: ปิด popup แล้วแสดงเครื่องมือที่ตรง inline ใต้หัวหมวด · หัวหมวดคลิกไม่เปิด popup · ล้างคำค้น → กลับเหลือแต่หัวหมวด · pal-count = จำนวนที่ตรง', () => {
  withDom(() => {
    const t = mkPal();
    t.secs[0].one('pal-head').click(); // เปิด popup ก่อนค้น
    assert.equal(t.popup().hidden, false);

    const input = t.container.one('palette-search');
    input.value = 'otsu';
    input.fire('input');
    assert.equal(t.popup().hidden, true, 'เริ่มค้น → ปิด popup');
    const segIdx = TITLES.indexOf('SEGMENTATION & EDGE');
    const seg = t.secs[segIdx];
    assert.equal(seg.hidden, false);
    assert.equal(seg.one('pal-list').hidden, false, 'ผลค้นหาแสดง inline');
    const visible = seg.one('pal-list').all('tool-row').filter((r) => !r.hidden);
    assert.ok(visible.length >= 1 && visible.length < seg.one('pal-list').all('tool-row').length);
    assert.ok(visible.every((r) => r.one('af-name').textContent.toLowerCase().includes('otsu')));
    assert.equal(seg.one('pal-count').textContent, String(visible.length), 'ค้นอยู่ → นับเฉพาะที่ตรง');
    assert.ok(t.secs.filter((s) => s !== seg).every((s) => s.hidden), 'หมวดที่ไม่ตรงต้องซ่อน');
    assert.equal(t.container.one('palette-empty').hidden, true);
    // ค้นอยู่ → คลิกหัวหมวดไม่เปิด popup
    assert.equal(seg.one('pal-head').getAttribute('aria-disabled'), 'true');
    seg.one('pal-head').click();
    assert.equal(t.popup().hidden, true);

    // คลิก tile inline = onAdd (ต้องส่งทั้ง filterId ระดับหมวดและ opId ที่คลิก)
    visible[0].click();
    assert.equal(t.added.length, 1);
    assert.equal(t.added[0][0], 'segment_edge');

    // ไม่พบ
    input.value = 'zzzz-ไม่มีแน่นอน';
    input.fire('input');
    assert.equal(t.container.one('palette-empty').hidden, false);
    assert.equal(t.container.one('palette-empty').textContent, 'ไม่พบเครื่องมือ');
    assert.ok(t.secs.every((s) => s.hidden));

    // ล้างคำค้น → เหลือแต่หัวหมวด · popup ยังปิด · เปิดใหม่ได้
    input.value = '';
    input.fire('input');
    assert.equal(t.container.one('palette-empty').hidden, true);
    assert.ok(t.secs.every((s) => !s.hidden && s.one('pal-list').hidden));
    assert.equal(seg.one('pal-count').textContent, String(toolCount(TOOL_SECTIONS[segIdx])));
    assert.equal(seg.one('pal-head').getAttribute('aria-disabled'), 'false');
    seg.one('pal-head').click();
    assert.equal(t.popup().hidden, false);
  }, makeStore());
});

test('p2-d5 palette: ไม่จำหมวดที่เปิดข้ามการเปิดแอป — localStorage เก่า (mv-palette-open) ถูกเมิน popup เริ่มปิดเสมอ · render ซ้ำปิด popup', () => {
  const stored = JSON.stringify({ COLOR: true, MOTION: true });
  const store = makeStore([['mv-palette-open', stored]]);
  withDom(() => {
    const t = mkPal();
    assert.equal(t.popup().hidden, true);
    assert.ok(t.secs.every((s) => s.one('pal-head').getAttribute('aria-expanded') === 'false'));
    t.secs[0].one('pal-head').click();
    t.pal.render();
    assert.equal(t.container.one('pal-popover').hidden, true, 'render ใหม่ → popup ปิด');
    assert.equal(store.getItem('mv-palette-open'), stored, 'ไม่เขียนทับ/ไม่บันทึกสถานะเปิดอีก');
  }, store);
});

// ===================================================================
// 2.4 ป้ายซูม + preset มุมมอง + carry-over
// ===================================================================
test('p2-e1 formatZoom: scale → "86%"', () => {
  assert.equal(formatZoom(0.86), '86%');
  assert.equal(formatZoom(1), '100%');
  assert.equal(formatZoom(0.6), '60%');
  assert.equal(formatZoom(1.2 * 1.2), '144%');
  assert.equal(formatZoom(1 / 1.1), '91%', 'ปัดเศษ ไม่ตัดทิ้ง (90.9 → 91)');
  assert.equal(formatZoom(0.996), '100%');
  assert.equal(formatZoom(NaN), '—');
  assert.equal(formatZoom(undefined), '—');
});

test('p2-e2 preset มุมมอง: สัดส่วน 0.70/0.56/0.35 และ preset ความกว้างแคบ < กว้าง', () => {
  assert.deepEqual(STAGE_PRESETS.map((p) => [p.label, p.ratio]), [['ภาพใหญ่', 0.70], ['สมดุล', 0.56], ['กระดานใหญ่', 0.35]]);
  for (const k of ['palette', 'props']) assert.ok(WIDTH_PRESETS[k].narrow < WIDTH_PRESETS[k].wide);
  // ต้องอยู่ในขอบเขตเดียวกับการลาก (LIMITS ใน layout-resize.js: palette 200–440 · props 280–560)
  assert.ok(WIDTH_PRESETS.palette.narrow >= 200 && WIDTH_PRESETS.palette.wide <= 440);
  assert.ok(WIDTH_PRESETS.props.narrow >= 280 && WIDTH_PRESETS.props.wide <= 560);
});

test('p2-e3 statusLevel / sourceChipLabel', () => {
  assert.equal(statusLevel('Display 1: Grayscale → Otsu Threshold'), 'ok');
  assert.equal(statusLevel('✕ ประมวลผลไม่สำเร็จ — ลองปรับค่า'), 'error');
  assert.equal(statusLevel('✕ ตัวประมวลผลภาพขัดข้อง'), 'error');
  assert.equal(statusLevel('⚠ ยังไม่มีกล่องผลลัพธ์ใดต่อสายถึงแหล่งภาพ'), 'warn');
  assert.equal(statusLevel('ยังไม่มีกล่องผลลัพธ์'), 'warn');
  assert.equal(statusLevel('พร้อมใช้งาน'), 'ok');
  assert.equal(statusLevel(''), 'ok');
  assert.equal(sourceChipLabel('camera'), 'กล้อง');
  assert.equal(sourceChipLabel('video'), 'วิดีโอ');
  assert.equal(sourceChipLabel('image'), 'ภาพนิ่ง');
});

// ===================================================================
// 2.5 planInsert
// ===================================================================
const W = 192, TH = 28;
const filterSel = { kind: 'filter', x: 260, y: 80, w: W, h: 56 };
const newF = { w: W, h: 56, inputs: 1 };
const dn = (id, x, y = 80) => ({ id, x, y, w: W, h: 56 });

test('p2-f1 planInsert: แทรกกลางสายปกติ → วางขวาของ selected +28 · เลื่อนปลายน้ำให้พ้น (dx = ขอบขวากล่องใหม่ + 28 − x ปลายน้ำ)', () => {
  const plan = planInsert({ sel: filterSel, newNode: newF, fanOut: 1, downstream: [dn(7, 480), dn(8, 700)], TH });
  assert.equal(plan.mode, 'splice');
  assert.deepEqual(plan.pos, [260 + W + 28, 80]);
  // กล่องใหม่ x0=480 → x1=672 → ปลายน้ำต้องเริ่ม ≥ 700 → ปลายน้ำที่ใกล้สุด (480) เลื่อน +220
  assert.equal(plan.dx, 220);
  assert.deepEqual(plan.shiftIds, [7, 8]);
});

test('p2-f2 planInsert: ปลายน้ำห่างพออยู่แล้ว → ไม่เลื่อน (dx = 0) · ไม่มีปลายน้ำ → ต่อท้าย dx = 0', () => {
  assert.equal(planInsert({ sel: filterSel, newNode: newF, fanOut: 1, downstream: [dn(7, 900)], TH }).dx, 0);
  const tail = planInsert({ sel: filterSel, newNode: newF, fanOut: 0, downstream: [], TH });
  assert.equal(tail.mode, 'splice');
  assert.equal(tail.dx, 0);
  assert.deepEqual(tail.shiftIds, []);
});

test('p2-f3 planInsert: Source ก็แทรกได้', () => {
  const plan = planInsert({ sel: { kind: 'source', x: 40, y: 80, w: W, h: 56 }, newNode: newF, fanOut: 1, downstream: [dn(2, 260)], TH });
  assert.equal(plan.mode, 'splice');
  assert.deepEqual(plan.pos, [260, 80]);
});

test('p2-f4 planInsert: ข้าม (วางที่ว่างแบบเดิม) เมื่อ selected เป็น Display/DL · กล่องใหม่ 2 อินพุต · fan-out > 1', () => {
  for (const kind of ['display', 'dl', undefined]) {
    const r = planInsert({ sel: { ...filterSel, kind }, newNode: newF, fanOut: 1, downstream: [dn(7, 480)], TH });
    assert.equal(r.mode, 'free', `kind=${kind}`);
    assert.equal(r.reason, 'selected-kind');
  }
  assert.equal(planInsert({ sel: null, newNode: newF, TH }).mode, 'free');
  const two = planInsert({ sel: filterSel, newNode: { ...newF, inputs: 2 }, fanOut: 1, downstream: [dn(7, 480)], TH });
  assert.deepEqual([two.mode, two.reason], ['free', 'two-inputs']);
  const fan = planInsert({ sel: filterSel, newNode: newF, fanOut: 2, downstream: [dn(7, 480), dn(8, 480, 200)], TH });
  assert.deepEqual([fan.mode, fan.reason], ['free', 'fan-out']);
  // fan-out = 1 ยังแทรกได้
  assert.equal(planInsert({ sel: filterSel, newNode: newF, fanOut: 1, downstream: [dn(7, 480)], TH }).mode, 'splice');
});

test('p2-f5 planInsert: กล่องอื่นขวางตำแหน่งใหม่ หรือขวางปลายน้ำที่เลื่อน → ข้าม (ไม่ทับ)', () => {
  const blocker = { x0: 470, y0: 80 - TH, x1: 662, y1: 136 }; // ทับตำแหน่งกล่องใหม่ (480,80)
  const r1 = planInsert({ sel: filterSel, newNode: newF, fanOut: 1, downstream: [dn(7, 480)], obstacles: [blocker], TH });
  assert.deepEqual([r1.mode, r1.reason], ['free', 'blocked']);
  // ปลายน้ำ x=480 จะถูกเลื่อน +220 ไป 700..892 → มีของขวางตรงนั้น (แต่ไม่ขวางกล่องใหม่ที่ 480..672 ที่ y ต่างกัน)
  const far = { x0: 760, y0: 80 - TH, x1: 900, y1: 136 };
  const r2 = planInsert({ sel: filterSel, newNode: newF, fanOut: 1, downstream: [dn(7, 480)], obstacles: [far], TH });
  assert.deepEqual([r2.mode, r2.reason], ['free', 'blocked-downstream']);
  // ของที่อยู่ไกลออกไปไม่ขวาง
  const away = { x0: 2000, y0: 0, x1: 2200, y1: 100 };
  assert.equal(planInsert({ sel: filterSel, newNode: newF, fanOut: 1, downstream: [dn(7, 480)], obstacles: [away], TH }).mode, 'splice');
});

test('p2-f6 planInsert: Display ใต้ตัวสุดท้าย (เทมเพลต) → เลื่อน Display ไปขวา ไม่ทับกล่องใหม่', () => {
  // ตัวสุดท้ายอยู่ x=260 y=80 · Display อยู่ใต้ [260, 188]
  const plan = planInsert({ sel: filterSel, newNode: newF, fanOut: 1, downstream: [dn(9, 260, 188)], TH });
  assert.equal(plan.mode, 'splice');
  assert.equal(plan.dx, 260 + W + 28 + W + 28 - 260);
  const newBoxRight = plan.pos[0] + W;
  assert.ok(260 + plan.dx >= newBoxRight + 28);
});

test('p2-f7 planInsert: เลื่อนวัดจาก "เป้าหมายตรง ๆ" — ปลายน้ำที่อยู่ซ้ายไกลกว่าไม่ทำให้เลื่อนเกิน', () => {
  // เป้าหมายตรง ๆ อยู่ x=260 · ปลายน้ำอีกตัว (เช่น กล่องที่เคยวางไว้ซ้ายสุด) อยู่ x=40 — ถ้าวัดจากซ้ายสุด dx จะเป็น 440
  const ds = [{ ...dn(7, 260), direct: true }, dn(8, 40, 248)];
  const plan = planInsert({ sel: { kind: 'source', x: 40, y: 80, w: W, h: 56 }, newNode: newF, fanOut: 1, downstream: ds, TH });
  assert.equal(plan.mode, 'splice');
  assert.equal(plan.dx, 220, 'dx = 452 + 28 − 260');
  assert.deepEqual(plan.shiftIds, [7, 8], 'ปลายน้ำทั้งก้อนยังเลื่อนไปด้วยกัน');
});

// ===================================================================
// รอบแก้ (reviewer): splitReadout corpus จริง · โฟกัสหลัง render ซ้ำ · ↺ · ลำดับ · statusLevel · planInsert · select placeholder
// ===================================================================
const REG_SRC = fs.readFileSync(new URL('../src/renderer/filters/registry.js', import.meta.url), 'utf8');

test('p2-g1 splitReadout: ข้อความ ctx.info จริงที่มีเลขหลายตัว → null (ไม่โชว์เลขผิด)', () => {
  const multi = [
    'ลบวัตถุเล็ก 3 ชิ้น (เหลือ 5 ชิ้น)',
    'ลบวัตถุที่แตะขอบภาพ 2 ชิ้น (เหลือ 6 ชิ้น)',
    'พบจุดปลาย 3 จุด · จุดแยก 2 จุด',
    'พบจุดปลาย 3 จุด · จุดแยก 2',
    'พบขอบ (จุดตัดศูนย์) 1234 พิกเซล · sigma 2 · เกณฑ์ 5',
    'Entropy filter 5x5, 16 ระดับ (ประมาณ ไม่ใช่ 256 ระดับเต็ม)',
    'ผสมโปร่งแสง A 50% / B 50%',
    'ซ้อนแบบตารางหมากรุก (ขนาดตาราง 32px)',
    'ซ่อม 120 px (1.2% ของภาพ) ด้วย Telea r=3',
    'พบวัตถุ 5 ชิ้น — พื้นที่เฉลี่ย 4428px²',
    'พบวัตถุ 9 ชิ้น แสดง 5 ชิ้นที่ใหญ่ที่สุด (ปรับ \'แสดงวัตถุสูงสุด\' ได้) — พื้นที่เฉลี่ย 4428px²',
    'SSIM = 0.9123 (1 = เหมือนกันทุกประการ)',
    'PSNR = 31.20 dB (MSE = 12.00)',
    'เกณฑ์ที่คำนวณได้ = 80, 160',
    'เลือกได้ 1200 พิกเซล (3.1% ของภาพ) · จุดเริ่ม (10, 20) · ความต่างไม่เกิน 10',
    'พบจุดเด่น 500 จุด — แสดง 300 จุดที่คะแนนดีที่สุด (ORB)',
  ];
  for (const t of multi) assert.equal(splitReadout(t), null, t);
});

test('p2-g2 splitReadout: ข้อความเลขตัวเดียว (Otsu / พบวัตถุ …) ยังได้ tile ถูกค่า', () => {
  assert.deepEqual(splitReadout('ค่าเกณฑ์ที่ Otsu คำนวณได้ = 137'), { label: 'ค่าเกณฑ์ที่ Otsu คำนวณได้', value: '137', unit: '' });
  assert.deepEqual(splitReadout('พบวัตถุ 5 ชิ้น'), { label: 'พบวัตถุ', value: '5', unit: 'ชิ้น' });
  assert.deepEqual(splitReadout('ระยะไกลสุดจากขอบ = 12.3 px'), { label: 'ระยะไกลสุดจากขอบ', value: '12.3', unit: 'px' });
  assert.deepEqual(splitReadout('พบวงกลม 4 วง'), { label: 'พบวงกลม', value: '4', unit: 'วง' });
});

test('p2-g3 splitReadout: ข้อความ ctx.info ทุกแบบใน registry.js — ถ้าเป็น tile ต้องมีเลขตัวเดียวทั้งข้อความ', () => {
  const tpls = [...REG_SRC.matchAll(/ctx\.info = `([^`\n]*)`/g)].map((m) => m[1]);
  assert.ok(tpls.length > 30, 'อ่านเทมเพลต info จาก registry ได้น้อยเกินไป — regex เสีย?');
  let tiles = 0;
  for (const t of tpls) {
    const rendered = t.replace(/\$\{[^}]*\}/g, '7');
    const r = splitReadout(rendered);
    if (!r) continue;
    tiles++;
    const nums = rendered.match(/\d+/g) || [];
    assert.equal(nums.length, 1, `tile จากข้อความหลายเลข: ${rendered}`);
    assert.equal(r.value, '7');
  }
  assert.ok(tiles >= 3, 'ควรมีข้อความที่เป็น tile ได้อย่างน้อยไม่กี่แบบ (กัน guard เข้มจน vacuous)');
});

test('p2-g4 โฟกัสไม่หลุดเมื่อแผงถูกสร้างใหม่ซ้ำ (signature เปลี่ยนหลังเปลี่ยนวิธี)', () => withDom(() => {
  const rig = makeRig({ id: 'shape_analysis', params: { ...defaultParams('shape_analysis') } }, []);
  rig.render();
  const modeParam = FILTERS.shape_analysis.params.find((p) => p.key === 'mode');
  const sel = rig.container.one('field-select');
  sel.focus();
  const cur = modeParam.options.findIndex((o) => o.value === rig.step.params.mode);
  sel.value = sel.children[(cur + 1) % modeParam.options.length].value;
  sel.fire('change');
  const sel2 = rig.container.one('field-select');
  assert.equal(env.focused, sel2);
  // note ชุดใหม่ทำให้ app.js เรียก render() เต็มอีกครั้ง (activeElement = select ตัวใหม่)
  rig.render([{ id: 'shape_analysis', info: 'พบวัตถุ 5 ชิ้น' }]);
  const sel3 = rig.container.one('field-select');
  assert.notEqual(sel3, sel2);
  assert.equal(env.focused, sel3, 'โฟกัสต้องตามไปที่ select ตัวใหม่ ไม่ใช่ BODY');
  // ไม่มีโฟกัสในแผง (เช่น คลิกที่ canvas) → ไม่ไปแย่งโฟกัส
  env.focused = null;
  rig.render([]);
  assert.equal(env.focused, null);
}));

test('p2-g5 ↺ แล้วโฟกัสย้ายไปช่องค่า (ไม่หลุดตอนปุ่มปิดตัวเอง) + แผงสร้างใหม่ยังคืนโฟกัสช่องค่า/สไลเดอร์', () => withDom(() => {
  const { id, p } = findNumberParam();
  const rig = makeRig({ id, params: { ...defaultParams(id) } }, []);
  rig.render();
  const wrap = rig.container.all('param-num').find((w) => w.one('pn-label').textContent === p.label);
  const slider = wrap.one('pn-slider');
  slider.value = String(p.max === defaultParams(id)[p.key] ? p.min : p.max);
  slider.fire('input');
  wrap.one('pn-reset').click();
  assert.equal(env.focused, wrap.one('pn-value'));
  // render ซ้ำขณะโฟกัสสไลเดอร์ → ได้สไลเดอร์ตัวใหม่
  slider.focus();
  rig.render();
  const slider2 = rig.container.all('param-num').find((w) => w.one('pn-label').textContent === p.label).one('pn-slider');
  assert.equal(env.focused, slider2);
}));

test('p2-g6 ลำดับ: หัว → วิธี (mode) → "พารามิเตอร์" → ตัวที่เหลือ · วิธีที่ไม่มีตัวอื่น → ข้อความ "ไม่มีค่าให้ปรับ"', () => withDom(() => {
  const rig = makeRig({ id: 'shape_analysis', params: { ...defaultParams('shape_analysis'), mode: 'regionprops' } }, []);
  rig.render();
  const a = topSeq(rig);
  assert.ok(a.indexOf('param') >= 0 && a.indexOf('param') < a.indexOf('params-head'), 'ตัวเลือกวิธีต้องมาก่อนหัว "พารามิเตอร์"');
  assert.ok(rig.container.children[a.indexOf('param')].one('field-select'), 'param ตัวแรกคือ select วิธี');
  assert.ok(a.indexOf('params-head') < a.lastIndexOf('param'));
  // Grayscale (color_space mode=grayscale): มีแต่ mode → ไม่มีหัว "พารามิเตอร์" แต่มีข้อความไม่มีค่าให้ปรับ
  const gs = makeRig({ id: 'color_space', params: { ...defaultParams('color_space'), mode: 'grayscale' } }, []);
  gs.render();
  const b = topSeq(gs);
  const rest = FILTERS.color_space.params.filter((q) => q.key !== 'mode' && (!q.showIf || q.showIf.value === 'grayscale'));
  if (rest.length === 0) {
    assert.ok(!b.includes('params-head'));
    assert.ok(b.includes('param-hint'));
  }
}));

test('p2-g7 select ที่ค่าไม่อยู่ในตัวเลือก: มีตัวเลือกว่างซ่อนไว้และเลือกอยู่ · เลือกตัวแรกแล้ว commit ได้', () => withDom(() => {
  const rig = makeRig({ id: 'shape_analysis', params: { ...defaultParams('shape_analysis'), mode: 'ค่าที่ไม่มีอยู่จริง' } }, []);
  rig.render();
  const modeParam = FILTERS.shape_analysis.params.find((p) => p.key === 'mode');
  const sel = rig.container.one('field-select');
  assert.equal(sel.children.length, modeParam.options.length + 1);
  assert.equal(sel.children[0].hidden, true);
  assert.equal(sel.children[0].selected, true, 'โชว์ตัวเลือกว่าง ไม่ใช่ตัวเลือกแรกของจริง');
  assert.equal(sel.children[0].value, '', 'placeholder value ว่าง (แยกจาก index 0)');
  sel.value = sel.children[1].value; // ตัวเลือกจริงตัวแรก
  sel.fire('change');
  assert.deepEqual(rig.changes.at(-1), ['mode', modeParam.options[0].value]);
}));

test('p2-g8 statusLevel: ข้อความล้มเหลวที่ไม่มี ✕ → error · ⚠ ยังเป็น warn · ข้อความปกติยัง ok', () => {
  for (const t of ['โหลดโมเดลไม่สำเร็จ: x', 'เปิด Pose ไม่สำเร็จ: x', 'ไม่พบไฟล์ภาพในสิ่งที่เลือก', 'นำเข้า Dataset ไม่สำเร็จ: x', 'ตัวประมวลผลขัดข้อง']) {
    assert.equal(statusLevel(t), 'error', t);
  }
  assert.equal(statusLevel('⚠ ตรวจไม่พบร่างกายในภาพนี้'), 'warn', '⚠ มาก่อน — ยังเป็นส้ม');
  for (const t of ['Display 1: Grayscale → Otsu Threshold', 'ภาพ 2/5 — a.png', '📋 วางเทมเพลตแล้ว — ปรับค่าที่แผงขวาต่อได้เลย', 'พร้อมใช้งาน']) {
    assert.equal(statusLevel(t), 'ok', t);
  }
});

test('p2-g9 planInsert: ปลายน้ำที่ไม่ใช่เป้าหมายตรง ๆ ชนกล่องใหม่ → free', () => {
  // เป้าหมายตรง ๆ อยู่ไกล (dx = 0) แต่ปลายน้ำอีกตัวอยู่ที่ (260,110) ทับตำแหน่งกล่องใหม่ (260,80)
  const ds = [{ ...dn(7, 700), direct: true }, dn(8, 260, 110)];
  const r = planInsert({ sel: { kind: 'source', x: 40, y: 80, w: W, h: 56 }, newNode: newF, fanOut: 1, downstream: ds, TH });
  assert.deepEqual([r.mode, r.reason], ['free', 'blocked-downstream']);
});

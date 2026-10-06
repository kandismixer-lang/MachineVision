// 2026-10-01 — เทสต์ถาวรของงาน U3 "ค่าที่วัดได้" ในแผงขวา (src/renderer/ui/detail-panel.js)
//
// ทำไมต้องมีไฟล์นี้: การรับประกันของ U3 (โดยเฉพาะ P-16 = ห้าม rebuild แผงทุกเฟรม) เคยอยู่ใน harness
// ชั่วคราวเท่านั้น ถ้าไม่ตรึงเป็นเทสต์ถาวร โค้ดจะ regress เงียบ ๆ (แผงกลับไป rebuild 30 ครั้ง/วินาที,
// ตารางผีของวิธีเก่าค้างจอ, DOM งอก, new Intl.NumberFormat ทุกเซลล์ทุกเฟรม)
//
// ข้อจำกัดของสนามทดสอบ: ห้ามลง dependency ใหม่ → ไม่มี jsdom ต้องใช้ DOM stub ที่เขียนเองในไฟล์นี้
// (stub ทำเท่าที่ detail-panel.js ใช้จริง: createElement / appendChild / append / classList /
//  textContent / innerHTML / setAttribute / addEventListener+click / style / hidden / title)
//
// นโยบาย global: ไฟล์นี้ patch globalThis.document, globalThis.setTimeout/clearTimeout, Date.now และ
// Intl.NumberFormat — ทั้งหมดติดตั้ง "ภายใน callback ของเทสต์แต่ละตัว" (ดู u3test()) และคืนของจริงด้วย
// try/finally เสมอ · ห้ามใช้ beforeEach/afterEach ระดับไฟล์เด็ดขาด — ดูเหตุผลที่ u3test()
// (node --test อาจรันหลายไฟล์ในโปรเซสเดียวกัน — global ที่รั่วจะทำให้ไฟล์เทสต์อื่นพังแบบสุ่ม)
// เทสต์ตัวสุดท้ายของไฟล์ (u3-24) ตรวจซ้ำว่า restoreEnv() คืน global ได้ครบจริง
//
// ทุกเทสต์ deterministic 100%: ไม่มี Math.random ไม่พึ่งเวลาจริง (clock stub) ไม่พึ่งกล้อง/เครือข่าย
import assert from 'node:assert/strict';
import { test, after } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createDetailPanel, splitReadout } from '../src/renderer/ui/detail-panel.js';
import { FILTERS } from '../src/renderer/filters/registry.js';

// ค่าคงที่ที่ "สะท้อน" ของจริงใน detail-panel.js — ถ้าของจริงเปลี่ยน เทสต์ที่พึ่งมันจะฟ้องทันที
const PAINT_MS = 200;       // throttle การทาสีค่า
const ROWS_DEFAULT = 20;    // แถวที่สร้างไว้ใน pool ตอนแรก
const FRAME_MS = 1000 / 30; // กล้องสด 30 fps
const DASH = '—';      // em dash
const INF = '∞';

const SRC_DIR = fileURLToPath(new URL('../src/renderer/', import.meta.url));
const PANEL_SRC = fs.readFileSync(path.join(SRC_DIR, 'ui', 'detail-panel.js'), 'utf8');

// ===================================================================
// DOM stub + clock stub + ตัวนับ
// ===================================================================
const stats = { created: 0, textWrites: 0, nfNew: 0 };

class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.parentNode = null;
    this._text = '';
    this._cls = '';
    this.style = {};
    this._attrs = {};
    this.hidden = false;
    this.title = '';
    this._listeners = {};
    stats.created++;
    const self = this;
    this.classList = {
      add(c) { const s = new Set(self._cls.split(' ').filter(Boolean)); s.add(c); self._cls = [...s].join(' '); },
      remove(c) { const s = new Set(self._cls.split(' ').filter(Boolean)); s.delete(c); self._cls = [...s].join(' '); },
      contains(c) { return self._cls.split(' ').includes(c); },
      toggle(c, on) {
        const want = on === undefined ? !this.contains(c) : !!on;
        if (want) this.add(c); else this.remove(c);
      },
    };
  }
  get className() { return this._cls; }
  set className(v) { this._cls = String(v); }
  // DOM จริง: textContent = ข้อความรวมของลูกทุกตัว · การ set จะแทนที่ลูกทั้งหมดด้วย text node เดียว
  get textContent() {
    if (this.children.length) return this.children.map((c) => c.textContent).join('');
    return this._text;
  }
  set textContent(v) { stats.textWrites++; this._text = String(v); this.children = []; }
  // detail-panel ใช้ innerHTML แค่ 2 แบบ: ล้าง ("") และฝัง markup คงที่ → เก็บเฉพาะข้อความ (ตรงกับ textContent จริง)
  set innerHTML(v) { this._text = String(v).replace(/<[^>]*>/g, ''); this.children = []; }
  get innerHTML() { return this._text; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  append(...cs) { for (const c of cs) this.appendChild(c); }
  // DOM จริง: setAttribute(k) ที่ไม่ส่งค่า → ได้ "" ไม่ใช่ "undefined" — ต้องตรงกันไม่งั้น stub โกหก
  setAttribute(k, v = '') { this._attrs[k] = String(v); }
  getAttribute(k) { return Object.prototype.hasOwnProperty.call(this._attrs, k) ? this._attrs[k] : null; }
  removeAttribute(k) { delete this._attrs[k]; }
  addEventListener(t, fn) { (this._listeners[t] ||= []).push(fn); }
  click() { for (const fn of this._listeners.click || []) fn({}); }
  blur() {}
  focus() {}
  // ---- helper ของเทสต์ (ไม่ใช่ API ของ DOM) ----
  all(cls, out = []) {
    for (const c of this.children) {
      if (c._cls.split(' ').includes(cls)) out.push(c);
      c.all(cls, out);
    }
    return out;
  }
  one(cls) { return this.all(cls)[0] || null; }
  count() { let n = this.children.length; for (const c of this.children) n += c.count(); return n; }
}

let fakeNow = 1_000_000;
let timers = [];
let timerSeq = 0;
const clock = {
  now() { return fakeNow; },
  advance(ms) {
    const end = fakeNow + ms;
    for (;;) {
      let due = null;
      for (const t of timers) {
        if (t.at <= end && (!due || t.at < due.at || (t.at === due.at && t.id < due.id))) due = t;
      }
      if (!due) break;
      timers = timers.filter((t) => t !== due);
      fakeNow = due.at;
      due.fn();
    }
    fakeNow = end;
  },
  pending() { return timers.length; },
};

// ของจริงที่ต้องคืนให้ครบ — จับภาพ "ครั้งเดียว" ตอนโหลดไฟล์ ก่อนแตะอะไรทั้งนั้น
const ORIG = {
  hadDocument: Object.prototype.hasOwnProperty.call(globalThis, 'document'),
  document: globalThis.document,
  setTimeout: globalThis.setTimeout,
  clearTimeout: globalThis.clearTimeout,
  dateNow: Date.now,
  NumberFormat: Intl.NumberFormat,
};

function installEnv() {
  // ถ้าไฟล์เทสต์อื่นรันก่อนเราแล้วทิ้ง stub ค้างไว้ ORIG จะจับภาพ "ของปลอม" มาตั้งแต่ตอน import
  // → restoreEnv() จะคืนของปลอมให้ไฟล์ถัดไป ลามเป็นลูกโซ่ · ดักให้ดังตั้งแต่ตรงนี้ ไม่ใช่ปล่อยให้เงียบ
  assert.ok(ORIG.dateNow() > 1.7e12, 'Date.now ที่ ORIG จับภาพไว้ไม่ใช่ของจริง — มี global รั่วจากไฟล์เทสต์อื่น');
  assert.equal(ORIG.NumberFormat.name, 'NumberFormat', 'Intl.NumberFormat ที่ ORIG จับภาพไว้ถูกห่อไว้แล้ว — มี global รั่วจากไฟล์เทสต์อื่น');
  stats.created = 0;
  stats.textWrites = 0;
  stats.nfNew = 0;
  timers = [];
  timerSeq = 0;
  fakeNow = 1_000_000;
  globalThis.document = { createElement: (t) => new El(t) };
  globalThis.setTimeout = (fn, ms) => {
    const id = ++timerSeq;
    timers.push({ id, at: fakeNow + (Number(ms) || 0), fn });
    return id;
  };
  globalThis.clearTimeout = (id) => { timers = timers.filter((t) => t.id !== id); };
  Date.now = () => fakeNow;
  const Orig = ORIG.NumberFormat;
  function NumberFormatSpy(...a) { stats.nfNew++; return new Orig(...a); }
  NumberFormatSpy.prototype = Orig.prototype;
  NumberFormatSpy.supportedLocalesOf = (...a) => Orig.supportedLocalesOf(...a);
  Intl.NumberFormat = NumberFormatSpy;
}

function restoreEnv() {
  Intl.NumberFormat = ORIG.NumberFormat;
  Date.now = ORIG.dateNow;
  globalThis.setTimeout = ORIG.setTimeout;
  globalThis.clearTimeout = ORIG.clearTimeout;
  if (ORIG.hadDocument) globalThis.document = ORIG.document;
  else delete globalThis.document;
  timers = [];
}

// ทุกเทสต์ของไฟล์นี้ต้องผ่าน u3test() เท่านั้น — ติดตั้ง stub ตอนเข้า คืนของจริงตอนออกด้วย try/finally
//
// ทำไมไม่ใช้ beforeEach/afterEach ระดับไฟล์: node --test --experimental-test-isolation=none รันทุกไฟล์
// ในโปรเซสเดียว และ hook ระดับ root ของไฟล์หนึ่งจะไปครอบเทสต์ของ "ไฟล์อื่น" ด้วย → เทสต์ไฟล์อื่นที่ await
// setTimeout จริงจะค้างตลอดกาลเพราะโดน fake clock ของเราครอบ (ยืนยันแล้วว่าค้างจริง ไม่ใช่ทฤษฎี)
// ขอบเขตของ stub จึงต้องแคบที่สุดเท่าที่ทำได้ = อยู่แค่ใน callback ของเทสต์ตัวเอง
function u3test(name, fn) {
  test(name, () => {
    installEnv();
    try {
      fn();
    } finally {
      restoreEnv();
    }
  });
}
after(restoreEnv); // เข็มขัดนิรภัยชั้นที่ 2 (idempotent — เขียนทับด้วยของจริงเสมอ)

// ===================================================================
// Rig — จำลอง loop ของ app.js เป๊ะ ๆ
//   worker.onmessage: if (!detailPanel.applyNotes(state.notes)) renderDetail();   (app.js:298)
//   renderDetail():   detailPanel.render(sel, state.notes.filter(n => n.id === sel.filterId))  (app.js:1238)
// rig.renders = จำนวนครั้งที่แผงถูก "rebuild ทั้งแผง" = ตัวเลขที่ P-16 ต้องกดให้เป็น 0
// ===================================================================
function makeRig(step, notes) {
  const container = new El('div');
  const rig = {
    container,
    step,
    notes: notes || [],
    renders: 0,
    paramChanges: [],
    panel: null,
    mine() { return (rig.notes || []).filter((n) => n.id === rig.step.id); },
    render() { rig.renders++; rig.panel.render(rig.step, rig.mine()); },
    frame(ns) {
      if (ns !== undefined) rig.notes = ns;
      if (!rig.panel.applyNotes(rig.notes)) rig.render();
      clock.advance(FRAME_MS);
    },
    // ปล่อยให้ trailing timeout ของ throttle ทำงานจนหมด (โหมดภาพนิ่ง/หลังเฟรมสุดท้าย)
    flush(ms = PAINT_MS + 50) { clock.advance(ms); },
    has(cls) { return container.all(cls).length > 0; },
  };
  rig.panel = createDetailPanel(container, {
    onParamChange: (k, v) => { rig.paramChanges.push([k, v]); if (rig.step) rig.step.params[k] = v; },
  });
  return rig;
}

// ---- ข้อมูล note จำลอง (deterministic: ขึ้นกับ tick ที่ส่งเข้ามาเท่านั้น) ----
const RP_COLS = ['index', 'area', 'cx', 'cy', 'perimeter', 'circularity',
  'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'orientation_deg'];

function rpRows(n, t) {
  return Array.from({ length: n }, (_, i) => [
    i + 1,
    4000 + ((i * 37 + t * 13) % 900),
    120.4 + i + (t % 7) / 10,
    88.1 + i,
    233.5 + (t % 11),
    0.912,
    10, 20, 30, 40,
    i % 3 === 0 ? null : 12.5 + (t % 5),
  ]);
}
// regionprops: จำนวนวัตถุกระพริบทุกเฟรม + ข้อความ info เปลี่ยนทุกเฟรม (เคสจริงที่ทำให้ P-16 เกิด)
// [nodeId round] note จริงจาก worker พก view = f.subtitle(params) ของวิธีที่ใช้คำนวณ → ทดสอบด้วยรูปเดียวกัน
const VIEW_RP = FILTERS.shape_analysis.subtitle({ mode: 'regionprops' });
const VIEW_FC = FILTERS.shape_analysis.subtitle({ mode: 'find_contours' });
function rpNotes(n, t = 0) {
  return [
    { id: 'shape_analysis', view: VIEW_RP, info: `พบวัตถุ ${n} ชิ้น (เฟรม ${t})` },
    { id: 'shape_analysis', view: VIEW_RP, values: { kind: 'table', columns: RP_COLS, rows: rpRows(n, t), total: n } },
  ];
}
// find_contours: ส่งแต่ info ไม่มี values เลย
const fcNotes = (t = 0) => [{ id: 'shape_analysis', view: VIEW_FC, info: `พบเส้นขอบ ${3 + (t % 4)} เส้น` }];

// Motion (Farneback): scalars 6 ตัวที่เปลี่ยนค่าทุกเฟรม + info เปลี่ยนทุกเฟรม
function fbNotes(t) {
  return [
    { id: 'farneback_optical_flow', info: `ภาพขยับ ${((t * 7) % 97) / 10} px` },
    {
      id: 'farneback_optical_flow',
      values: {
        kind: 'scalars',
        fields: [
          { key: 'mean_motion_px', label: 'ระยะขยับเฉลี่ย (px)', value: ((t * 31) % 311) / 100 },
          { key: 'max_motion_px', label: 'ระยะขยับสูงสุด (px)', value: ((t * 17) % 733) / 10 },
          { key: 'moving_pct', label: 'พื้นที่ที่ขยับ (%)', value: ((t * 5) % 97) / 10 },
          { key: 'mean_dx_px', label: 'ขยับแนวนอน (px)', value: -1.25 + (t % 3) },
          { key: 'mean_dy_px', label: 'ขยับแนวตั้ง (px)', value: 0.5 + (t % 4) },
          { key: 'frame', label: 'เฟรมที่', value: t },
        ],
      },
    },
  ];
}

const psnrNotes = (db = 31.2, mse = 12) => [
  { id: 'psnr', info: 'PSNR' },
  {
    id: 'psnr',
    values: {
      kind: 'scalars',
      fields: [
        { key: 'psnr_db', label: 'PSNR (dB)', value: db },
        { key: 'mse', label: 'MSE', value: mse },
      ],
    },
  },
];

// ---- คลิกปุ่ม "เลือกวิธี" ของกล่องรวม (choice-list ที่ registry สร้าง) ----
const SHAPE_SEL = FILTERS.shape_analysis.params.find((p) => p.type === 'select');
const modeLabel = (v) => SHAPE_SEL.options.find((o) => o.value === v).label;
// [Phase 2.2 แก้เทสต์] "วิธี" (mode) เป็น <select class="field-select"> แทนรายการปุ่ม choice — เลือกตัวเลือกแล้วยิง change
function clickMode(rig, value) {
  const want = modeLabel(value);
  const sel = rig.container.one('field-select');
  assert.ok(sel, 'ไม่พบ select "วิธี" ในแผง');
  const idx = sel.children.findIndex((o) => o.textContent === want);
  assert.ok(idx >= 0, `ไม่พบตัวเลือกวิธี "${want}" ใน select`);
  sel.selectedIndex = idx;
  for (const fn of sel._listeners.change || []) fn({});
}

// อ่านไฟล์ .js ทั้งหมดใน src/renderer/ (ใช้กับเทสต์ static)
function walkJs(dir, out = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walkJs(p, out);
    else if (e.name.endsWith('.js')) out.push(p);
  }
  return out;
}
// ตัดคอมเมนต์ออกก่อนค้นหา — คอมเมนต์ใน app.js พูดถึง "เลิก JSON.stringify notes" ซึ่งไม่ใช่โค้ดจริง
// ข้อจำกัดที่รู้ตัว: นี่คือ regex ไม่ใช่ JS parser จริง (ห้ามลง dependency ใหม่ → ไม่มี acorn/babel ให้ใช้)
// จึงอาจพลาดถ้าโค้ดมี string literal ที่มี "/*" หรือ "//" อยู่ข้างใน — ตรวจแล้วว่า src/renderer/ ไม่มีเคสนั้น
// ถ้าวันหนึ่งมี ผลคือ "ตัดเกิน/ตัดขาด" ซึ่งทำให้เทสต์ static ดังผิด ๆ (คนรีวิวเห็น) ไม่ใช่เงียบ
function stripComments(src) {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:"'`\\])\/\/[^\n]*/g, '$1');
}

// ===================================================================
// ข้อ 1 — P-16: กล้องสดต้องไม่ rebuild แผงเลย
// ===================================================================

u3test('u3-01 P-16: Motion (scalars + info เปลี่ยนทุกเฟรม) 300 เฟรม → render() ถูกเรียก 0 ครั้ง', () => {
  const rig = makeRig({ id: 'farneback_optical_flow', params: {}, enabled: true }, fbNotes(0));
  rig.render();
  for (let t = 1; t <= 10; t++) rig.frame(fbNotes(t)); // warm-up
  const base = rig.renders;
  const writes0 = stats.textWrites;
  for (let t = 11; t <= 310; t++) rig.frame(fbNotes(t));
  assert.equal(rig.renders - base, 0, 'แผงถูก rebuild ระหว่างกล้องสด — P-16 regress');

  // throttle ต้องยังคุมงานอยู่: PAINT_MS = 200 → ทาสีไม่เกิน 5 ครั้ง/วินาที
  // 300 เฟรม = 10 วินาที · ต่อการทา 1 ครั้งมีข้อความได้มากสุด 7 ชิ้น (scalars 6 + info 1)
  const writes = stats.textWrites - writes0;
  const cap = 7 * 10 * (1000 / PAINT_MS);
  assert.ok(writes <= cap, `เขียนข้อความ ${writes} ครั้งใน 10 วินาที (เพดาน ${cap}) — throttle PAINT_MS หลุด`);
  assert.ok(writes > 100, `เขียนข้อความแค่ ${writes} ครั้ง — ค่าแทบไม่ถูกทาเลย เทสต์นี้จะ vacuous`);

  // ไม่ vacuous: ตัวเลขต้องอัปเดตจริงในที่ (in-place) หลัง flush trailing timer
  rig.flush();
  const shown = rig.container.all('dv-v').map((e) => e.textContent);
  assert.equal(shown[5], '310', 'ค่า scalars ไม่ถูกทาใหม่ — เทสต์นี้จะ vacuous');
  // [Phase 2.2 แก้เทสต์] note-info ที่เข้ารูป "ป้าย ตัวเลข หน่วย" เป็น readout tile (3 span) แทนข้อความ "📊 …" บรรทัดเดียว
  // — พฤติกรรมที่ปักไว้ (อัปเดตในที่ + ค่าสุดท้ายถูกต้อง) คงเดิม เปลี่ยนแค่รูปมาร์กอัป
  const want = splitReadout(fbNotes(310)[0].info);
  assert.ok(want, 'ข้อความ info ของ Motion ควรเข้ารูป readout');
  assert.equal(rig.container.one('rt-label').textContent, want.label);
  assert.equal(rig.container.one('rt-value').textContent, want.value);
  assert.equal(rig.container.one('rt-unit').textContent, want.unit);
});

u3test('u3-02 P-16: regionprops (ตาราง + จำนวนแถวกระพริบ) 300 เฟรม → render() ถูกเรียก 0 ครั้ง', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops', min_area: 50 } }, rpNotes(11, 0));
  rig.render();
  for (let t = 1; t <= 10; t++) rig.frame(rpNotes(8 + (t % 7), t));
  const base = rig.renders;
  for (let t = 11; t <= 310; t++) rig.frame(rpNotes(8 + (t % 7), t));
  assert.equal(rig.renders - base, 0, 'ตารางกระพริบทำให้แผง rebuild — P-16 regress');

  rig.flush();
  const rows = 8 + (310 % 7);
  assert.match(rig.container.one('dv-count').textContent, new RegExp('^' + rows + ' แถว$'),
    'จำนวนแถวท้ายตารางไม่ตามค่าเฟรมล่าสุด — เทสต์นี้จะ vacuous');
});

u3test('u3-03 P-16 (static): ไม่มี JSON.stringify ของ notes หลงเหลือใน src/renderer/', () => {
  const offenders = [];
  for (const file of walkJs(SRC_DIR)) {
    const code = stripComments(fs.readFileSync(file, 'utf8'));
    // จับทั้ง JSON.stringify(notes) / JSON.stringify(state.notes) / JSON.stringify(msg.notes) ฯลฯ
    if (/JSON\.stringify\s*\(\s*[A-Za-z_$][\w$.]*\.?notes\b/.test(code)) offenders.push(path.relative(SRC_DIR, file));
  }
  assert.deepEqual(offenders, [], 'กลับไป stringify notes ทุกเฟรมแล้ว — P-16 regress');
  assert.ok(!/JSON\.stringify/.test(stripComments(PANEL_SRC)), 'detail-panel.js ไม่ควรมี JSON.stringify เลย');
});

// ===================================================================
// ข้อ 2 — กฎ "อ่าน values ตัวแรกเสมอ" (dedupe) · เกราะกัน qa 8.2 / P-19
// ===================================================================

u3test('u3-04 dedupe: note id เดียวกัน 2 ตัว ตอน render() → ต้องอ่าน values ของ "ตัวแรก"', () => {
  const dupe = [
    { id: 'psnr', info: 'PSNR' },
    { id: 'psnr', values: { kind: 'scalars', fields: [{ key: 'psnr_db', label: 'PSNR (dB)', value: 11 }] } },
    { id: 'psnr', values: { kind: 'scalars', fields: [{ key: 'psnr_db', label: 'PSNR (dB)', value: 99 }] } },
  ];
  const rig = makeRig({ id: 'psnr', params: {} }, dupe);
  rig.render();
  assert.equal(rig.container.one('dv-v').textContent, '11.00', 'ไม่ได้อ่าน values ของ note ตัวแรก');
  assert.ok(rig.container.one('dv-dup'), 'มี note ค่าซ้ำ 2 ตัวแต่ไม่เตือนผู้ใช้');
});

u3test('u3-05 dedupe: note id เดียวกัน 2 ตัว ตอน paint ทุกเฟรม → ต้องอ่าน values ของ "ตัวแรก"', () => {
  const dup = (a, b) => [
    { id: 'psnr', info: 'PSNR' },
    { id: 'psnr', values: { kind: 'scalars', fields: [{ key: 'psnr_db', label: 'PSNR (dB)', value: a }] } },
    { id: 'psnr', values: { kind: 'scalars', fields: [{ key: 'psnr_db', label: 'PSNR (dB)', value: b }] } },
  ];
  const rig = makeRig({ id: 'psnr', params: {} }, dup(11, 99));
  rig.render();
  const before = rig.renders;
  for (let t = 0; t < 30; t++) rig.frame(dup(22, 88)); // เส้นทาง applyNotes → doPaint → firstValues
  rig.flush();
  assert.equal(rig.renders - before, 0, 'โครงไม่ควรเปลี่ยน — signature ต้องนิ่ง');
  assert.equal(rig.container.one('dv-v').textContent, '22.00', 'paint อ่าน values ตัวสุดท้ายแทนตัวแรก');
});

// ===================================================================
// ข้อ 3 — guard "ผีค้าง": สลับวิธีแล้ว filter หยุดส่ง note ทั้งหมด
// ===================================================================

u3test('u3-06 guard: สลับ regionprops → find_contours แล้วยังมี note (info อย่างเดียว) → ตารางผีต้องหายใน 1 เฟรม', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops', min_area: 50 } }, rpNotes(6, 0));
  rig.render();
  for (let t = 1; t <= 6; t++) rig.frame(rpNotes(6, t));
  assert.ok(rig.has('dv'), 'ก่อนสลับวิธีต้องมีตาราง');

  clickMode(rig, 'find_contours');
  assert.equal(rig.has('dv'), false, 'กดสลับวิธีแล้วยังมีตารางของวิธีเก่าโผล่ทันที');

  rig.frame(rpNotes(6, 7)); // เฟรมค้างในท่อ คำนวณด้วยวิธีเก่า
  let ghost = 0;
  for (let t = 8; t < 68; t++) { rig.frame(fcNotes(t)); if (rig.has('dv')) ghost++; }
  assert.equal(ghost, 0, `ตารางของวิธีเก่าค้างอยู่ ${ghost} เฟรมหลังสลับวิธี`);
});

u3test('u3-07 guard (MAJOR): สลับวิธีพร้อมกับ filter หยุดส่ง note ทั้งหมด → เฟรมค้างของวิธีเก่าถูกทิ้งด้วย view tag + ห้ามมีตารางผีถาวร', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops', min_area: 50 } }, rpNotes(6, 0));
  rig.render();
  for (let t = 1; t <= 6; t++) rig.frame(rpNotes(6, t));

  clickMode(rig, 'find_contours');
  rig.frame(rpNotes(6, 7)); // เฟรมค้างที่ยังพกค่าของวิธีเก่ามา (view = regionprops) → ต้องถูกทิ้งทันที ไม่สร้างบล็อก
  assert.equal(rig.has('dv'), false, 'note ของวิธีเก่าที่ view ไม่ตรงยังสร้างตารางผีได้');

  // ตั้งแต่นี้กล่องไม่ส่ง note ออกมาเลย (หลุดจากสายผลลัพธ์ / เฟรมหยุด / worker error)
  for (let t = 0; t < 120; t++) rig.frame([]);
  assert.equal(rig.has('dv'), false, 'ตารางของวิธีเก่าค้างถาวร (MAJOR regress)');
});

u3test('u3-08 guard ต้องไม่ over-fire: note หายชั่วคราว (psnr ถอดสาย B) → บล็อกต้องอยู่ + ขึ้น "กำลังรอค่า"', () => {
  const rig = makeRig({ id: 'psnr', params: {} }, psnrNotes());
  rig.render();
  for (let t = 0; t < 5; t++) rig.frame(psnrNotes());
  for (let t = 0; t < 90; t++) rig.frame([{ id: 'psnr', info: 'ยังไม่ได้ต่อภาพเข้าช่อง B' }]);
  assert.ok(rig.has('dv'), 'บล็อกค่าหายตอนถอดสาย B — ผู้ใช้เสียบริบท (แผงกระโดด)');
  assert.ok(rig.container.one('dv').className.includes('is-waiting'), 'ไม่เข้าสถานะ "กำลังรอค่า"');
});

// [nodeId round] เดิมเทสต์นี้ตรึงขอบเขตของ heuristic staleWatch — ตอนนี้ staleWatch ถูกถอดแล้ว (ใช้ view tag แทน)
// ยังคงไว้เป็นเกราะ: note ที่ไม่พก view (ข้อความ info ล้วน) ห้ามทำให้บล็อก/การตรึงของผู้ใช้หาย (G3 หายไปด้วย)
u3test('u3-09 guard ต้องไม่ over-fire: note ที่ไม่พก view (info ล้วน) มาแทน → บล็อก + การตรึงต้องอยู่ครบ', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'find_contours' } }, fcNotes(0));
  rig.render();
  rig.frame(fcNotes(1));
  clickMode(rig, 'regionprops');
  // เดินหลายเฟรมให้ได้ค่าของ regionprops หลายชุด — throttle ทำให้ต้องเดินหลายเฟรม
  for (let t = 2; t <= 22; t++) rig.frame(rpNotes(9, t));
  rig.flush();
  assert.ok(rig.has('dv'), 'สลับมา regionprops แล้วต้องมีตาราง');

  rig.container.one('dv-freeze').click();
  assert.ok(rig.container.one('dv').className.includes('is-frozen'), 'กดตรึงแล้วต้องติดสถานะตรึง');

  // กล่องนี้เงียบ แต่กล่องชนิดเดียวกันอีกกล่อง (คนละวิธี) ยังส่ง note มาแทน
  for (let t = 0; t < 60; t++) rig.frame([{ id: 'shape_analysis', info: 'พบวัตถุ 3 ชิ้น' }]);
  assert.ok(rig.has('dv'), 'บล็อกหายไปเพราะกล่องชนิดเดียวกันอีกกล่องรายงาน — guard ยิงเกินขอบเขต');
  assert.ok(rig.container.one('dv').className.includes('is-frozen'), 'การตรึงของผู้ใช้หายไป');
  assert.ok(rig.container.all('dv-td')[1].textContent.length > 0, 'ค่าที่ตรึงไว้หายไปจากเซลล์');
});

// ===================================================================
// ข้อ 4 — signature ต้องไม่มี rows.length และไม่มีเนื้อข้อความ info
// ===================================================================

u3test('u3-10 signature: จำนวนแถวเปลี่ยน 0↔30 → ห้าม rebuild (ห้ามเอา rows.length เข้า signature)', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops' } }, rpNotes(5, 0));
  rig.render();
  const base = rig.renders;
  const counts = [0, 1, 30, 2, 0, 19, 20, 21, 7, 0, 30];
  for (let t = 0; t < 220; t++) rig.frame(rpNotes(counts[t % counts.length], t));
  assert.equal(rig.renders - base, 0, 'จำนวนแถวที่เปลี่ยนทำให้ rebuild — rows.length หลุดเข้า signature');

  // ตรึงเชิงโครงสร้างอีกชั้น: ตัวสร้าง signature ห้ามแตะ rows เลย
  const body = PANEL_SRC.match(/function structSig\(mine\)\s*\{[\s\S]*?\n\}/);
  assert.ok(body, 'ไม่พบ function structSig — โครงไฟล์เปลี่ยน ต้องรีวิวเทสต์ชุดนี้ใหม่');
  // ตัดคอมเมนต์ก่อน — ในตัวฟังก์ชันมีคอมเมนต์อธิบายว่า "ไม่เอา rows.length" ซึ่งไม่ใช่โค้ด
  // ใช้ \brows\b (ไม่ใช่ /rows/ ดิบ) เพื่อไม่ให้ดังใส่ชื่อตัวแปรอื่นที่บังเอิญลงท้ายด้วย rows
  assert.ok(!/\brows\b/.test(stripComments(body[0])),
    'structSig อ้างถึง rows — จะ rebuild ทุกครั้งที่จำนวนวัตถุกระพริบ');
});

u3test('u3-11 signature: เนื้อข้อความ info เปลี่ยนทุกเฟรม → ห้าม rebuild แต่ข้อความต้องถูกเขียนทับในที่', () => {
  const info = (t) => [{ id: 'shape_analysis', info: `ข้อความที่เปลี่ยนทุกเฟรม #${t}` },
    { id: 'shape_analysis', values: { kind: 'table', columns: RP_COLS, rows: rpRows(4, t), total: 4 } }];
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops' } }, info(0));
  rig.render();
  const base = rig.renders;
  const created = stats.created;
  for (let t = 1; t <= 200; t++) rig.frame(info(t));
  assert.equal(rig.renders - base, 0, 'ข้อความ info ที่เปลี่ยนทำให้ rebuild — เนื้อ info หลุดเข้า signature');
  assert.equal(stats.created - created, 0, 'สร้าง DOM ใหม่ระหว่างที่ข้อความเปลี่ยน');

  rig.flush();
  assert.equal(rig.container.one('detail-note-info').textContent, '📊 ' + info(200)[0].info,
    'ข้อความ 📊 ไม่ถูกอัปเดตในที่ — ผู้ใช้เห็นข้อความค้าง');
});

u3test('u3-12 signature ต้องไวจริง (คุมไม่ให้เทสต์ข้างบน vacuous): โครงเปลี่ยนจริง → ต้อง rebuild', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops' } }, rpNotes(5, 0));
  rig.render();
  const base = rig.renders;
  // เปลี่ยนชุดคอลัมน์ = โครงเปลี่ยนจริง
  rig.frame([{ id: 'shape_analysis', info: 'x' },
    { id: 'shape_analysis', values: { kind: 'table', columns: ['index', 'area'], rows: [[1, 2]], total: 1 } }]);
  assert.equal(rig.renders - base, 1, 'คอลัมน์เปลี่ยนแล้วไม่ rebuild — applyNotes หลวมเกินไป');
  // เปลี่ยน kind = โครงเปลี่ยนจริง
  rig.frame([{ id: 'shape_analysis', values: { kind: 'scalars', fields: [{ key: 'area', value: 1 }] } }]);
  assert.equal(rig.renders - base, 2, 'kind เปลี่ยนแล้วไม่ rebuild');
});

// ===================================================================
// ข้อ 5 — Intl.NumberFormat ต้องถูก cache
// ===================================================================

u3test('u3-13 perf: Intl.NumberFormat ถูก cache — 300 เฟรม × 11 คอลัมน์ สร้างเพิ่ม 0 ตัว', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops' } }, rpNotes(20, 0));
  rig.render();
  for (let t = 1; t <= 30; t++) rig.frame(rpNotes(20, t)); // warm-up ให้ครบทุกจำนวนทศนิยมที่ใช้
  rig.flush();
  const warm = stats.nfNew;
  for (let t = 31; t <= 330; t++) rig.frame(rpNotes(20, t));
  rig.flush();
  assert.equal(stats.nfNew - warm, 0,
    `new Intl.NumberFormat เพิ่มขึ้น ${stats.nfNew - warm} ตัวใน 300 เฟรม — cache NF หลุด`);
  // จำนวนที่สร้างตลอดชีวิตต้องเท่ากับ "จำนวนทศนิยมที่ต่างกัน" เท่านั้น ไม่โตตามเซลล์/เฟรม
  assert.ok(warm <= 8, `สร้าง Intl.NumberFormat ไปแล้ว ${warm} ตัวตั้งแต่ warm-up — ควร ≤ 8 (1 ตัวต่อ 1 จำนวนทศนิยม)`);
});

// ===================================================================
// ข้อ 6 — ไม่มี DOM node งอก (pool คงที่)
// ===================================================================

u3test('u3-14 soak 5 นาที (9000 เฟรม): จำนวน DOM node คงที่ ไม่มีการสร้าง element ใหม่', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops' } }, rpNotes(12, 0));
  rig.render();
  for (let t = 1; t <= 20; t++) rig.frame(rpNotes(3 + (t % 18), t));
  const dom0 = rig.container.count();
  const created0 = stats.created;
  const base = rig.renders;
  for (let t = 21; t <= 9020; t++) rig.frame(rpNotes(3 + (t % 25), t)); // 3..27 แถว คร่อม ROWS_DEFAULT
  assert.equal(rig.renders - base, 0, 'มี rebuild ระหว่าง soak');
  assert.equal(stats.created - created0, 0, `สร้าง element ใหม่ ${stats.created - created0} ตัวระหว่าง soak`);
  assert.equal(rig.container.count(), dom0, `DOM โตจาก ${dom0} เป็น ${rig.container.count()}`);
  assert.ok(clock.pending() <= 1, 'timer ค้างสะสม');
  assert.equal(rig.container.all('dv-tr').length, ROWS_DEFAULT, 'pool แถวต้องคงที่ที่ ROWS_DEFAULT');
});

// ===================================================================
// ข้อ 7 — resetValuesView() ต้องถูกเรียกใน renderDisplay() / renderDL()
// ===================================================================

u3test('u3-15 renderDisplay(): ต้องล้างบล็อกค่า + ยกเลิก trailing timer (ห้ามเขียนลง DOM ที่หลุดต้นไม้แล้ว)', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops' } }, rpNotes(4, 0));
  rig.render();
  // ยังไม่ขยับนาฬิกา → schedulePaint ต้องตั้ง trailing timer (ไม่ทาทันที)
  rig.panel.applyNotes(rpNotes(4, 99));
  assert.equal(clock.pending(), 1, 'ไม่ได้ตั้ง trailing timer — เทสต์นี้จะ vacuous');

  const td = rig.container.all('dv-td')[1];
  const before = td.textContent;
  rig.panel.renderDisplay({ getName: () => '', setName: () => {} });
  assert.equal(clock.pending(), 0, 'renderDisplay ไม่ได้ล้าง timer ที่ค้าง (resetValuesView หาย)');
  clock.advance(5000);
  assert.equal(td.textContent, before, 'timer ที่ค้างไปเขียนลงเซลล์ที่หลุดจากต้นไม้แล้ว');
  assert.equal(rig.container.all('dv').length, 0, 'แผงกล่องผลลัพธ์ยังมีบล็อกค่าเก่าค้าง');
});

u3test('u3-16 renderDL(): ต้องล้างบล็อกค่า + ยกเลิก trailing timer เช่นเดียวกัน', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops' } }, rpNotes(4, 0));
  rig.render();
  rig.panel.applyNotes(rpNotes(4, 77));
  assert.equal(clock.pending(), 1, 'ไม่ได้ตั้ง trailing timer — เทสต์นี้จะ vacuous');

  const td = rig.container.all('dv-td')[1];
  const before = td.textContent;
  rig.panel.renderDL({ getStatus: () => 'ยังไม่ได้โหลดโมเดล', onBrowseModel: () => {}, onBrowseLabels: () => {} });
  assert.equal(clock.pending(), 0, 'renderDL ไม่ได้ล้าง timer ที่ค้าง (resetValuesView หาย)');
  clock.advance(5000);
  assert.equal(td.textContent, before, 'timer ที่ค้างไปเขียนลงเซลล์ที่หลุดจากต้นไม้แล้ว');
  assert.equal(rig.container.all('dv').length, 0, 'แผง Model Detect ยังมีบล็อกค่าเก่าค้าง');
});

u3test('u3-17 (static): renderDisplay/renderDL/render เรียก resetValuesView() ก่อนล้าง container', () => {
  for (const fn of ['render(step, notes)', 'renderDisplay(info)', 'renderDL(info)']) {
    const name = fn.slice(0, fn.indexOf('('));
    const re = new RegExp('function ' + name + '\\([^)]*\\)\\s*\\{([\\s\\S]*?)container\\.innerHTML\\s*=\\s*""');
    const m = PANEL_SRC.match(re);
    assert.ok(m, `ไม่พบหัวฟังก์ชัน ${name}() ที่ล้าง container — โครงไฟล์เปลี่ยน ต้องรีวิวเทสต์ใหม่`);
    assert.ok(/resetValuesView\(\);/.test(m[1]),
      `${name}() ล้าง container.innerHTML โดยไม่เรียก resetValuesView() ก่อน → DOM/timer เก่าจะค้าง`);
  }
});

// ===================================================================
// ข้อ 8 — บล็อกค่าต้องถูกแทรก "ก่อน" early-return ของ "วิธีนี้ไม่มีค่าให้ปรับ"
// ===================================================================

u3test('u3-18 filter ที่ไม่มี param เลย (psnr) ต้องยังได้ตารางค่า + อยู่ใต้ข้อความ "ไม่มีค่าให้ปรับ"', () => {
  assert.equal(FILTERS.psnr.params.length, 0, 'psnr มี param แล้ว — ต้องเลือก filter อื่นที่ไม่มี param มาคุมเคสนี้');
  const rig = makeRig({ id: 'psnr', params: {} }, psnrNotes(Infinity, 0));
  rig.render();
  const seq = rig.container.children.map((c) => c.className.split(' ')[0]);
  assert.ok(seq.includes('dv'), 'filter ที่ไม่มี param ไม่ได้บล็อก "ค่าที่วัดได้" เลย (early-return กินไป)');
  const hint = rig.container.children.find((c) => c.className === 'param-hint');
  assert.ok(hint && hint.textContent.includes('ไม่มีค่าให้ปรับ'), 'ข้อความ "ไม่มีค่าให้ปรับ" หายไป');
  assert.ok(seq.indexOf('dv') > seq.indexOf('detail-note'), 'บล็อกค่าต้องอยู่ใต้ note');
  // [Phase 2.2 แก้เทสต์] spec §6.2 ย้ายบล็อกค่าไปอยู่ "ใต้" ส่วนพารามิเตอร์ (ข้อความ "ไม่มีค่าให้ปรับ" คือส่วนนั้นของทางไม่มี param) — ยังต้องไม่หายเพราะ early-return
  assert.ok(seq.indexOf('dv') > seq.indexOf('param-hint'), 'บล็อกค่าต้องอยู่ใต้ข้อความ "ไม่มีค่าให้ปรับ"');
  assert.equal(rig.container.one('dv-v').textContent, INF, 'Infinity ต้องแสดงเป็นสัญลักษณ์อนันต์');
  assert.ok(rig.container.one('dv-v').className.includes('dv-inf'));
});

u3test('u3-19 SSIM (มี param แต่ไม่มี showIf) ก็ต้องได้บล็อกค่าเช่นกัน', () => {
  const notes = [{ id: 'ssim', values: { kind: 'scalars', fields: [{ key: 'ssim', label: 'SSIM', value: 0.9123456 }] } }];
  const rig = makeRig({ id: 'ssim', params: {} }, notes);
  rig.render();
  assert.ok(rig.has('dv'), 'SSIM ไม่ได้บล็อกค่า');
  assert.equal(rig.container.one('dv-v').textContent, '0.9123', 'ssim ต้องแสดง 4 ตำแหน่ง (VAL_META.ssim.d = 4)');
});

// ===================================================================
// เสริม — ตัวป้องกันรอบข้างที่ทำให้ข้อ 1–8 ยังมีความหมาย
// ===================================================================

u3test('u3-20 การจัดรูปค่า: หลักพัน / boolean / NaN / null / key ที่ไม่รู้จัก / เลขมหึมา', () => {
  const notes = [{
    id: 'pyrlk_optical_flow',
    values: {
      kind: 'scalars',
      fields: [
        { key: 'points', label: 'จุดที่ติดตาม', value: 12345 },
        { key: 'reseeded', label: 'เลือกจุดใหม่', value: true },
        { key: 'lost', label: 'หลุด', value: NaN },
        { key: 'tracked', label: 'ตามได้', value: null },
        { key: 'brand_new_key_from_future_op', value: 3.14159 },
        { key: 'big', label: 'ใหญ่', value: 2.5e9 },
      ],
    },
  }];
  const rig = makeRig({ id: 'pyrlk_optical_flow', params: {} }, notes);
  rig.render();
  const v = rig.container.all('dv-v').map((e) => e.textContent);
  assert.equal(v[0], '12,345');
  assert.equal(v[1], 'ใช่');
  assert.equal(v[2], DASH);
  assert.equal(v[3], DASH);
  assert.equal(v[4], '3.14', 'key ที่ไม่รู้จักต้องยังแสดงได้ (heuristic 2 ตำแหน่ง)');
  assert.equal(v[5], '2.50e+9');
  assert.equal(rig.container.all('dv-k')[4].textContent, 'brand_new_key_from_future_op',
    'key ที่ไม่รู้จักต้อง fallback เป็น key ดิบ ไม่ใช่ว่าง');
});

u3test('u3-21 ตาราง: หัวคอลัมน์ไทย + ความกว้างตายตัว + mark คอลัมน์รอง + em dash ตอนค่าเป็น null', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops' } }, rpNotes(4, 0));
  rig.render();
  const ths = rig.container.all('dv-th');
  assert.equal(ths.length, RP_COLS.length);
  assert.ok(ths.every((t) => /^\d+px$/.test(t.style.width || '')), 'ทุกคอลัมน์ต้องมีความกว้างตายตัว (table-layout:fixed)');
  assert.ok(ths.every((t) => t.title && t.title.length > 0), 'ทุกคอลัมน์ต้องมี tooltip ให้มือใหม่');
  assert.equal(ths[1].textContent, 'พื้นที่ (px²)');
  // คอลัมน์รอง (pri: false ใน VAL_META) = perimeter, bbox_x, bbox_y, bbox_w, bbox_h, orientation_deg
  const secondary = ths.filter((t) => t.getAttribute('data-dv-secondary') !== null).map((t) => t.textContent);
  assert.deepEqual(secondary,
    ['เส้นรอบรูป (px)', 'กรอบ X', 'กรอบ Y', 'กรอบ กว้าง', 'กรอบ สูง', 'มุมเอียง (°)'],
    'ชุดคอลัมน์รองเปลี่ยนไป — มือใหม่จะเห็นคอลัมน์ผิดชุดตั้งแต่แรก');
  // เซลล์ในแถวก็ต้อง mark ตรงกับหัวคอลัมน์ ไม่งั้น CSS ซ่อน/โชว์ไม่ตรงกัน
  const tds = rig.container.all('dv-td').slice(0, RP_COLS.length);
  assert.deepEqual(
    tds.map((t) => t.getAttribute('data-dv-secondary') !== null),
    ths.map((t) => t.getAttribute('data-dv-secondary') !== null),
    'การ mark คอลัมน์รองของ <td> ไม่ตรงกับ <th>');
  assert.equal(rig.container.all('dv-td')[10].textContent, DASH, 'orientation_deg = null ต้องเป็น em dash');
});

u3test('u3-22 ปุ่ม "ตรึงค่า" / "ทุกคอลัมน์": สลับสถานะโดยไม่สร้าง DOM ใหม่ และตรึงแล้วตัวเลขต้องหยุดจริง', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops' } }, rpNotes(4, 0));
  rig.render();
  for (let t = 1; t <= 5; t++) rig.frame(rpNotes(4, t));
  rig.flush();

  const created = stats.created;
  const colsBtn = rig.container.one('dv-cols');
  colsBtn.click();
  assert.equal(stats.created, created, 'ปุ่มสลับคอลัมน์ไปสร้าง DOM ใหม่');
  assert.ok(rig.container.one('dv-table').className.includes('show-all'));
  colsBtn.click();
  assert.ok(!rig.container.one('dv-table').className.includes('show-all'));

  const fb = rig.container.one('dv-freeze');
  fb.click();
  const frozen = rig.container.all('dv-td')[1].textContent;
  for (let t = 6; t <= 60; t++) rig.frame(rpNotes(4, t));
  rig.flush();
  assert.equal(rig.container.all('dv-td')[1].textContent, frozen, 'ตรึงแล้วตัวเลขยังขยับ');
  assert.equal(fb.getAttribute('aria-pressed'), 'true');
  assert.ok(rig.container.one('dv-hint').textContent.includes('ตรึงค่าไว้อยู่'));
  fb.click();
  assert.equal(fb.getAttribute('aria-pressed'), 'false');
  assert.notEqual(rig.container.all('dv-td')[1].textContent, frozen, 'ปลดตรึงแล้วต้องทาค่าล่าสุดทันที');
});

u3test('u3-23 โหมดภาพนิ่ง: เฟรมเดียวแล้วเงียบ → trailing timer ต้องทาค่าสุดท้ายเสมอ (ไม่ตกหล่น)', () => {
  const rig = makeRig({ id: 'shape_analysis', params: { mode: 'regionprops' } }, rpNotes(4, 0));
  rig.render();
  const before = rig.container.all('dv-td')[1].textContent;
  rig.panel.applyNotes(rpNotes(4, 123)); // ภายใน throttle → เข้า trailing
  assert.equal(rig.container.all('dv-td')[1].textContent, before, 'ควรถูก throttle ไว้ก่อน');
  clock.advance(PAINT_MS + 10);
  assert.notEqual(rig.container.all('dv-td')[1].textContent, before, 'trailing timer ไม่ได้ทาค่าสุดท้าย');
  assert.equal(clock.pending(), 0, 'timer ไม่ถูกเคลียร์หลังทำงาน');
});

u3test('u3-24 global stub ไม่รั่ว: หลังคืนค่าแล้ว document/timers/Date.now/Intl.NumberFormat ต้องเป็นของจริง', () => {
  // จุดนี้ยังอยู่ในสถานะ "ติดตั้ง stub" (u3test ติดตั้งให้ตอนเข้า) → พิสูจน์ว่า restoreEnv() คืนได้ครบจริง
  assert.notEqual(Intl.NumberFormat, ORIG.NumberFormat, 'stub ต้องถูกติดตั้งอยู่ ณ จุดนี้ ไม่งั้นเทสต์นี้ vacuous');
  restoreEnv();
  assert.equal(Intl.NumberFormat, ORIG.NumberFormat);
  assert.equal(Date.now, ORIG.dateNow);
  assert.equal(globalThis.setTimeout, ORIG.setTimeout);
  assert.equal(globalThis.clearTimeout, ORIG.clearTimeout);
  assert.equal(Object.prototype.hasOwnProperty.call(globalThis, 'document'), ORIG.hadDocument);
  assert.ok(typeof Date.now() === 'number' && Date.now() > 1.7e12, 'Date.now ต้องกลับมาเป็นเวลาจริง');
  assert.equal(new Intl.NumberFormat('en-US').format(1234), '1,234');
  installEnv(); // คืนสภาพให้ afterEach ทำงานตามปกติ
});

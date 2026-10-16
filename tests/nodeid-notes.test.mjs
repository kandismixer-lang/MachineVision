// 2026-10-06 — รอบ nodeId: note ผูกกับ "กล่อง" (nodeId) ไม่ใช่ชนิด filter + view tag แทน staleWatch
// ครอบคลุม: (1) worker ใส่ nodeId/view ให้ทุก note (autoGray/info/values · สาย B · memo hit ของ stateful op)
// (2) detail-panel/app.js กรองด้วย nodeId + ทิ้ง note ของ "วิธีเก่า" แบบ deterministic (3) P-16 ไม่ถอย
// DOM stub/clock stub ก๊อปจาก u3-detail-panel.test.mjs — ติดตั้งใน ntest() แต่ละตัวด้วย try/finally เท่านั้น
// (ห้ามใช้ beforeEach/afterEach ระดับไฟล์ — ดูเหตุผลใน u3-detail-panel.test.mjs)
import assert from 'node:assert/strict';
import { test, after } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import cv from '@techstark/opencv-js';
import { createDetailPanel, notesOfStep, stepAndNotesOf, infoTextsOf } from '../src/renderer/ui/detail-panel.js';
import { FILTERS } from '../src/renderer/filters/registry.js';

const PAINT_MS = 200;
const FRAME_MS = 1000 / 30;
const SRC_DIR = fileURLToPath(new URL('../src/renderer/', import.meta.url));

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cvtColor === 'function'; }
await new Promise((resolve) => {
  if (isReady(cv)) return resolve();
  const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
});
globalThis.self = globalThis;
if (typeof globalThis.ImageData === 'undefined') {
  globalThis.ImageData = class ImageData {
    constructor(data, width, height) { this.data = data; this.width = width; this.height = height; }
  };
}
const { runPipeline } = await import('../src/renderer/cv/cv-worker.js');

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

// ทุกเทสต์ของไฟล์นี้ต้องผ่าน ntest() เท่านั้น — ติดตั้ง stub ตอนเข้า คืนของจริงตอนออกด้วย try/finally
//
// ทำไมไม่ใช้ beforeEach/afterEach ระดับไฟล์: node --test --experimental-test-isolation=none รันทุกไฟล์
// ในโปรเซสเดียว และ hook ระดับ root ของไฟล์หนึ่งจะไปครอบเทสต์ของ "ไฟล์อื่น" ด้วย → เทสต์ไฟล์อื่นที่ await
// setTimeout จริงจะค้างตลอดกาลเพราะโดน fake clock ของเราครอบ (ยืนยันแล้วว่าค้างจริง ไม่ใช่ทฤษฎี)
// ขอบเขตของ stub จึงต้องแคบที่สุดเท่าที่ทำได้ = อยู่แค่ใน callback ของเทสต์ตัวเอง
function ntest(name, fn) {
  test(name, () => {
    installEnv();
    try {
      fn();
    } finally {
      restoreEnv();
    }
  });
}
after(restoreEnv);

// ===================================================================
// ตัวช่วย
// ===================================================================
let __uid = 1;
const nid = () => 700000 + __uid++;

// ภาพ RGB พื้นดำ + สี่เหลี่ยมขาว 2 อัน (ใหญ่ 30x30 พื้นที่ ~841, เล็ก 12x12 พื้นที่ ~121)
function twoBlobs() {
  const m = new cv.Mat(72, 96, cv.CV_8UC3, new cv.Scalar(0, 0, 0));
  cv.rectangle(m, new cv.Point(10, 10), new cv.Point(39, 39), new cv.Scalar(255, 255, 255), -1);
  cv.rectangle(m, new cv.Point(60, 40), new cv.Point(71, 51), new cv.Scalar(255, 255, 255), -1);
  return m;
}
const rp = (nodeId, minArea) => ({ id: 'shape_analysis', params: { mode: 'regionprops', regionprops__minArea: minArea }, enabled: true, nodeId });
const fcStep = (nodeId) => ({ id: 'shape_analysis', params: { mode: 'find_contours' }, enabled: true, nodeId });

// รัน pipeline เดียว คืน notes (ลบ mat เอง)
function runNotes(base, pipeline, memo = null, store = new Map()) {
  const r = runPipeline(base, pipeline, store, new Set(), memo);
  r.mat.delete();
  return r.notes;
}

// Rig = จำลอง loop ของ app.js: applyNotes ทุกเฟรม ถ้า false → render (นับ rebuild)
// rig.nodeParams = params ของ "node จริง" (เหมือน node.properties) แยกเป็นสำเนาคนละก้อนกับ step.params ที่แผงถือ
// — onParamChange เขียนลง node ไม่ใช่ลง curStep ของแผง ไม่งั้นลำดับ "อัปเดต params ก่อน render" ใน changeChoice เทสต์ไม่ได้
function makeRig(step, notes) {
  const container = new El('div');
  step = { ...step, params: { ...step.params } };
  const rig = {
    container, step, notes: notes || [], renders: 0, nodeParams: { ...step.params }, ownStale: [],
    render() { rig.renders++; rig.panel.render(rig.step, rig.notes); }, // ส่ง note ดิบ — panel กรองเอง
    frame(ns) {
      if (ns !== undefined) rig.notes = ns;
      if (!rig.panel.applyNotes(rig.notes)) rig.render();
      clock.advance(FRAME_MS);
    },
    flush() { clock.advance(PAINT_MS + 50); },
    has(cls) { return container.all(cls).length > 0; },
    count() { const c = container.one('dv-count'); return c ? c.textContent : null; },
  };
  rig.panel = createDetailPanel(container, {
    onParamChange: (k, v) => {
      rig.nodeParams[k] = v;
      // ระหว่างแผงเปลี่ยน param เอง (editor เรียก onPipelineChange ซ้อนข้างใน) isViewStale ต้องเป็น false เสมอ
      rig.ownStale.push(rig.panel.isViewStale({ ...rig.step, params: { ...rig.nodeParams } }));
    },
  });
  return rig;
}
const SHAPE_SEL = FILTERS.shape_analysis.params.find((p) => p.type === 'select');
function clickMode(rig, value) {
  const want = SHAPE_SEL.options.find((o) => o.value === value).label;
  const sel = rig.container.one('field-select');
  assert.ok(sel, 'ไม่พบ select "วิธี" ในแผง');
  // <option value> = index ใน param.options (detail-panel อ่านจาก sel.value) — shape_analysis ไม่มี optgroup จึงเป็นลูกตรง
  sel.value = sel.children.find((o) => o.textContent === want).value;
  for (const fn of sel._listeners.change || []) fn({});
}
const stepFor = (s) => ({ id: s.id, params: { ...s.params }, enabled: true, nodeId: s.nodeId });

// ===================================================================
// (1) worker: note ทุกตัวมี nodeId / view
// ===================================================================
ntest('nodeid-01 worker: note ทุกตัว (autoGray / info / values) มี nodeId ของ step · merged node มี view = subtitle', () => {
  const base = twoBlobs();
  try {
    const a = nid();
    const notes = runNotes(base, [rp(a, 10)]);
    assert.ok(notes.some((n) => n.autoGray), 'ต้องมี note autoGray (shape_analysis ต้องการภาพขาวดำ)');
    assert.ok(notes.some((n) => n.info) && notes.some((n) => n.values), 'ต้องมีทั้ง info และ values');
    for (const n of notes) {
      assert.equal(n.nodeId, a, 'note ทุกตัวต้องมี nodeId = step.nodeId: ' + JSON.stringify(Object.keys(n)));
      assert.equal(n.view, FILTERS.shape_analysis.subtitle({ mode: 'regionprops' }), 'view ต้องเป็น subtitle ของวิธีที่ใช้คำนวณ');
    }
  } finally { base.delete(); }
});

ntest('nodeid-02 worker: step ไม่มี nodeId → note ไม่มี key nodeId', () => {
  const base = twoBlobs();
  try {
    const notes = runNotes(base, [{ id: 'shape_analysis', params: { mode: 'regionprops', minArea: 10 }, enabled: true }]);
    assert.ok(notes.length > 0);
    for (const n of notes) assert.ok(!('nodeId' in n));
  } finally { base.delete(); }
});

ntest('nodeid-03 worker สาย B: note ของ node ในสาย src2 คง nodeId ของ node ต้นทาง (ไม่ใช่ของ op ที่รับ src2) และแผงของ node นั้นได้ค่า', () => {
  const base = twoBlobs();
  try {
    const outer = nid();
    const inner = nid();
    const pipeline = [{ id: 'image_arithmetic', params: { mode: 'imabsdiff' }, enabled: true, nodeId: outer, src2: { pipeline: [rp(inner, 10)] } }];
    const notes = runNotes(base, pipeline);
    assert.ok(notes.filter((n) => n.nodeId === inner).some((n) => n.values), 'ไม่มี values ของ node สาย B ที่ nodeId ของมันเอง');
    assert.ok(!notes.some((n) => n.nodeId === outer && n.values), 'values ของสาย B ห้ามถูกติด nodeId ของ op ที่รับ src2');
    // บั๊กเดิม: id "<op>(B):<filterId>" ไม่ตรง → แผง "รอค่า" ตลอดกาล
    const rig = makeRig(stepFor(rp(inner, 10)), notes);
    rig.render();
    assert.ok(rig.has('dv'), 'เลือก node สาย B แล้วไม่มีบล็อกค่า');
    assert.ok(!rig.container.one('dv').className.includes('is-waiting'), 'ค้างสถานะ "รอค่า"');
    assert.match(rig.count(), /^2 แถว$/);
  } finally { base.delete(); }
});

ntest('nodeid-04 worker: memo hit ของ stateful op ยังส่ง info/values (และ nodeId) ให้กล่องผลลัพธ์ที่ 2', () => {
  const base = twoBlobs();
  const store = new Map();
  try {
    const n = nid();
    const pl = [{ id: 'farneback_optical_flow', params: {}, enabled: true, nodeId: n }];
    const frame = () => {
      const m = new Map();
      try {
        const first = runNotes(base, pl, m, store);
        const second = runNotes(base, pl, m, store); // memo hit
        return { first, second };
      } finally { for (const e of m.values()) e.mat.delete(); }
    };
    frame();
    const { first, second } = frame();
    assert.ok(first.some((x) => x.values) && first.some((x) => x.info), 'เฟรมที่ 2 ต้องมี info+values ที่กล่องแรก');
    assert.equal(second.filter((x) => x.info).length, first.filter((x) => x.info).length, 'memo hit ทำ info หาย (qa 8.2)');
    assert.equal(second.filter((x) => x.values).length, first.filter((x) => x.values).length, 'memo hit ทำ values หาย (qa 8.2)');
    for (const x of second) assert.equal(x.nodeId, n);
    assert.notEqual(second.find((x) => x.values).values, first.find((x) => x.values).values, 'values ต้องเป็นสำเนา ไม่แชร์ object');
  } finally { base.delete(); store.clear(); }
});

// ===================================================================
// (2) UI: กล่องชนิดเดียวกัน 2 กล่องไม่ปนกัน
// ===================================================================
ntest('nodeid-05 กล่องชนิดเดียวกัน 2 กล่อง (min_area ต่างกัน) → แผงของแต่ละกล่องโชว์ค่าของตัวเอง ไม่ปนกัน', () => {
  const base = twoBlobs();
  try {
    const a = nid(), b = nid();
    const notes = [...runNotes(base, [rp(a, 200)]), ...runNotes(base, [rp(b, 10)])]; // A = 1 วัตถุ, B = 2 วัตถุ
    const rigA = makeRig(stepFor(rp(a, 200)), notes);
    rigA.render();
    const rigB = makeRig(stepFor(rp(b, 10)), notes);
    rigB.render();
    assert.match(rigA.count(), /^1 แถว$/, 'กล่อง A ต้องเห็น 1 แถว (ไม่ใช่ของ B)');
    assert.match(rigB.count(), /^2 แถว$/, 'กล่อง B ต้องเห็น 2 แถว (ไม่ใช่ตัวแรกในลิสต์)');
    // เส้นทาง paint ทุกเฟรม (applyNotes) ก็ต้องไม่ปน
    for (let t = 0; t < 20; t++) rigB.frame(notes);
    rigB.flush();
    assert.match(rigB.count(), /^2 แถว$/);
    // info (readout) ก็ต้องไม่ปน
    const txt = (rig) => rig.container.all('detail-note-info').map((e) => e.textContent).join('|');
    assert.ok(/1/.test(txt(rigA)) && !/2/.test(txt(rigA)), 'info ของ A ปน: ' + txt(rigA));
    assert.ok(/2/.test(txt(rigB)), 'info ของ B ไม่ใช่ของ B: ' + txt(rigB));
  } finally { base.delete(); }
});

ntest('nodeid-06 กล่องอื่นชนิดเดียวกันมี values แต่กล่องที่เลือกเงียบ → ห้ามยืมค่าของกล่องอื่น (G3)', () => {
  const base = twoBlobs();
  try {
    const a = nid(), b = nid();
    const notes = runNotes(base, [rp(b, 10)]); // มีเฉพาะของ B
    const rig = makeRig(stepFor(rp(a, 10)), notes);
    rig.render();
    for (let t = 0; t < 10; t++) rig.frame(notes);
    assert.equal(rig.has('dv'), false, 'แผงของ A โชว์ตารางของ B');
  } finally { base.delete(); }
});

// ===================================================================
// (3) สลับวิธี: note ของวิธีเก่าไม่ถูกวาด (deterministic ไม่พึ่งเวลา)
// ===================================================================
ntest('nodeid-07 สลับวิธี regionprops → find_contours: เฟรมค้างของวิธีเก่า (view ไม่ตรง) ไม่ถูกวาด · batch เดียวของวิธีใหม่ถูกวาด', () => {
  const base = twoBlobs();
  try {
    const a = nid();
    const oldNotes = runNotes(base, [rp(a, 10)]);
    const rig = makeRig(stepFor(rp(a, 10)), oldNotes);
    rig.render();
    assert.ok(rig.has('dv'), 'ก่อนสลับต้องมีตาราง');

    clickMode(rig, 'find_contours');
    assert.equal(rig.has('dv'), false, 'สลับแล้วตารางเก่ายังอยู่ทันที');
    rig.frame(oldNotes); // เฟรมค้างจากวิธีเก่า (ซ้ำหลายเฟรมก็ต้องไม่วาด)
    rig.frame(oldNotes);
    rig.flush();
    assert.equal(rig.has('dv'), false, 'note ของวิธีเก่าถูกวาดเป็นตารางผี');
    assert.equal(rig.container.all('detail-note-info').length, 0, 'info ของวิธีเก่าถูกวาด');

    rig.frame(runNotes(base, [fcStep(a)])); // ภาพนิ่ง: batch เดียวของวิธีใหม่
    assert.ok(rig.container.all('detail-note-info').length > 0, 'info ของวิธีใหม่ไม่ถูกวาด');
  } finally { base.delete(); }
});

ntest('nodeid-08 สลับไป-กลับ: batch ของวิธีที่เลือกอยู่ได้รับการวาดเสมอ', () => {
  const base = twoBlobs();
  try {
    const a = nid();
    const rig = makeRig(stepFor(rp(a, 10)), runNotes(base, [rp(a, 10)]));
    rig.render();
    clickMode(rig, 'find_contours');
    rig.frame(runNotes(base, [fcStep(a)]));
    clickMode(rig, 'regionprops');
    rig.frame(runNotes(base, [rp(a, 10)]));
    rig.flush();
    assert.match(rig.count() || '', /^2 แถว$/);
  } finally { base.delete(); }
});

// ===================================================================
// (4) notesOfStep หน่วย + "ตัวแรกชนะ"
// ===================================================================
ntest('nodeid-09 notesOfStep: กรอง nodeId · note ไม่มี view ผ่าน · view ไม่ตรงทิ้ง · step/notes ว่าง → []', () => {
  const step = { id: 'shape_analysis', params: { mode: 'regionprops' }, nodeId: 5 };
  const v = FILTERS.shape_analysis.subtitle({ mode: 'regionprops' });
  const notes = [
    { id: 'shape_analysis', nodeId: 5, view: v, info: 'ok' },
    { id: 'shape_analysis', nodeId: 6, view: v, info: 'กล่องอื่น' },
    { id: 'shape_analysis', nodeId: 5, view: 'Find Contours', info: 'วิธีเก่า' },
    { id: 'shape_analysis', nodeId: 5, info: 'ไม่มี view' },
    { id: 'shape_analysis', info: 'ไม่มี nodeId' },
  ];
  assert.deepEqual(notesOfStep(step, notes).map((n) => n.info), ['ok', 'ไม่มี view']);
  assert.deepEqual(notesOfStep(null, notes), []);
  assert.deepEqual(notesOfStep(step, null), []);
});

ntest('nodeid-10 N4: node เดียวป้อน 2 กล่องผลลัพธ์ → note ซ้ำ nodeId เดียวกัน → แผงอ่านตัวแรก + เตือนค่าซ้ำ', () => {
  const mk = (val) => ({ id: 'psnr', nodeId: 9, values: { kind: 'scalars', fields: [{ key: 'psnr_db', label: 'PSNR (dB)', value: val }] } });
  const rig = makeRig({ id: 'psnr', params: {}, enabled: true, nodeId: 9 }, [{ id: 'psnr', nodeId: 9, info: 'PSNR' }, mk(11), mk(99)]);
  rig.render();
  assert.equal(rig.container.one('dv-v').textContent, '11.00');
  assert.ok(rig.container.one('dv-dup'));
});

// ===================================================================
// (5) P-16: ไม่ rebuild ทุกเฟรม แม้มี note ของกล่องอื่นปนในลิสต์
// ===================================================================
ntest('nodeid-11 P-16: note ของกล่องอื่น (โครงต่างกัน) ปนเข้ามาทุกเฟรม → แผงของกล่องที่เลือกไม่ rebuild', () => {
  const base = twoBlobs();
  try {
    const a = nid(), b = nid();
    const mine = runNotes(base, [rp(a, 10)]);
    const other = [...runNotes(base, [fcStep(b)]), { id: 'shape_analysis', nodeId: b, info: 'x' }];
    const rig = makeRig(stepFor(rp(a, 10)), [...mine, ...other]);
    rig.render();
    const r0 = rig.renders;
    for (let t = 0; t < 120; t++) rig.frame(t % 2 ? [...mine, ...other] : [...other, ...mine]);
    rig.flush();
    assert.equal(rig.renders - r0, 0, 'แผง rebuild ทุกเฟรม — P-16 ถอย');
    assert.match(rig.count(), /^2 แถว$/);
  } finally { base.delete(); }
});

// ===================================================================
// (6) static: จุดบริโภคทั้งหมดไม่พึ่ง n.id อีก
// ===================================================================
ntest('nodeid-12 (static) ฝั่ง UI กรองด้วย notesOfStep — ไม่เหลือ n.id === ..., ไม่มี staleWatch, step พก nodeId', () => {
  const strip = (s) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '').replace(/\s\/\/.*$/gm, '');
  const panel = strip(fs.readFileSync(path.join(SRC_DIR, 'ui', 'detail-panel.js'), 'utf8'));
  const app = strip(fs.readFileSync(path.join(SRC_DIR, 'app.js'), 'utf8'));
  assert.ok(!/\bn\.id\s*===/.test(panel), 'detail-panel.js ยังเทียบ n.id');
  assert.ok(!/\bn\.id\s*===\s*sel\.filterId/.test(app), 'app.js ยังกรองด้วย filterId');
  assert.ok(!/staleWatch|staleSeen/.test(panel), 'ยังเหลือ staleWatch/staleSeen');
  assert.ok(/stepAndNotesOf\(sel, state\.notes\)/.test(app), 'app.js renderDetail ไม่ได้กรองด้วย stepAndNotesOf');
  });

// ===================================================================
// (7) step/notes จาก sel (app.js renderDetail + ทางลาก slider) — ทดสอบพฤติกรรมผ่านฟังก์ชัน pure
// ===================================================================
const fakeSel = (s) => ({ kind: 'filter', id: s.nodeId, filterId: s.id, getParams: () => ({ ...s.params }) });

ntest('nodeid-13 stepAndNotesOf: sel อยู่วิธีที่ไม่ใช่ default ของกล่องรวม + 2 กล่องชนิดเดียวกัน → step พก params/nodeId จริง และ note ไม่ปน', () => {
  const base = twoBlobs();
  try {
    assert.equal(FILTERS.shape_analysis.subtitle({}), FILTERS.shape_analysis.subtitle({ mode: 'find_contours' }), 'ข้อสมมติ: default = find_contours');
    const a = nid(), b = nid();
    const all = [...runNotes(base, [rp(a, 200)]), ...runNotes(base, [rp(b, 10)]), ...runNotes(base, [fcStep(nid())])];
    const { step, notes } = stepAndNotesOf(fakeSel(rp(b, 10)), all);
    assert.equal(step.id, 'shape_analysis');
    assert.equal(step.nodeId, b);
    assert.equal(step.params.mode, 'regionprops', 'params ต้องมาจาก sel.getParams() (ไม่งั้น view ตกเป็น default)');
    assert.ok(notes.length >= 3 && notes.every((n) => n.nodeId === b), 'ต้องได้ note ของกล่อง B ครบ ไม่ปนกล่องอื่น');
    assert.equal(notes.find((n) => n.values).values.rows.length, 2);
    // และเมื่อป้อนเข้าแผงจริง ต้องเห็นตารางของ B
    const rig = makeRig(step, notes);
    rig.render();
    assert.match(rig.count(), /^2 แถว$/);
  } finally { base.delete(); }
});

ntest('nodeid-14 infoTextsOf (ทางลาก slider): ได้เฉพาะ info ของกล่องที่เลือก ไม่มี info ของกล่องอื่นชนิดเดียวกัน/วิธีเก่า', () => {
  const base = twoBlobs();
  try {
    const a = nid(), b = nid();
    const all = [...runNotes(base, [rp(a, 200)]), ...runNotes(base, [rp(b, 10)]), ...runNotes(base, [fcStep(nid())])];
    const infoA = runNotes(base, [rp(a, 200)]).find((n) => n.info).info;
    const infoB = runNotes(base, [rp(b, 10)]).find((n) => n.info).info;
    assert.notEqual(infoA, infoB, 'ข้อสมมติของเทสต์: info ของสองกล่องต้องต่างกัน');
    assert.deepEqual(infoTextsOf(stepAndNotesOf(fakeSel(rp(a, 200)), all).notes), [infoA]);
    assert.deepEqual(infoTextsOf(stepAndNotesOf(fakeSel(rp(b, 10)), all).notes), [infoB]);
    assert.deepEqual(infoTextsOf(null), []);
  } finally { base.delete(); }
});

// ===================================================================
// (8) เปลี่ยนวิธีนอกแผง (เมนู Properties ของ LiteGraph) → isViewStale → app.js renderDetail()
// ===================================================================
ntest('nodeid-15 isViewStale: วิธีเปลี่ยนนอกแผง → true · วิธีเดิม → false · แผงเปลี่ยนเอง (select) → false ตลอด · ไม่มี step → false', () => {
  const base = twoBlobs();
  try {
    const a = nid();
    const rig = makeRig(stepFor(rp(a, 10)), runNotes(base, [rp(a, 10)]));
    assert.equal(rig.panel.isViewStale(stepFor(fcStep(a))), false, 'ยังไม่ render → ไม่มี curStep');
    rig.render();
    assert.equal(rig.panel.isViewStale(stepFor(rp(a, 10))), false);
    assert.equal(rig.panel.isViewStale(stepFor(rp(a, 99))), false, 'ปรับ param อื่นที่ไม่เปลี่ยนวิธีต้องไม่ stale');
    assert.equal(rig.panel.isViewStale(stepFor(fcStep(a))), true, 'วิธีเปลี่ยนนอกแผง ต้อง stale');
    assert.equal(rig.panel.isViewStale(null), false);
    clickMode(rig, 'find_contours');
    assert.ok(rig.ownStale.length > 0 && rig.ownStale.every((x) => x === false), 'แผงเปลี่ยนวิธีเอง ต้องไม่ถูกมองเป็น stale (กัน render ซ้อน)');
  } finally { base.delete(); }
});

ntest('nodeid-16 เปลี่ยนวิธีนอกแผง: ถ้า renderDetail() ตาม isViewStale แผงเห็น info ของวิธีใหม่ · ถ้าไม่ทำ แผงค้าง (พิสูจน์ว่ามีเงื่อนไขนี้จริง)', () => {
  const base = twoBlobs();
  try {
    const a = nid();
    const newNotes = runNotes(base, [fcStep(a)]);
    const mk = () => makeRig(stepFor(fcStep(a)), runNotes(base, [rp(a, 10)])); // เริ่มที่ regionprops ด้วย step เดิม
    const start = (rig) => { rig.step = stepFor(rp(a, 10)); rig.render(); };

    const stuck = mk(); start(stuck);
    for (let t = 0; t < 5; t++) stuck.frame(newNotes); // ไม่ rebuild → note วิธีใหม่ถูกกรองทิ้งทั้งหมด
    assert.equal(stuck.container.all('detail-note-info').length, 0, 'ข้อสมมติ: ถ้าไม่ rebuild แผงต้องค้าง');

    const fixed = mk(); start(fixed);
    const cur = stepFor(fcStep(a)); // step ใหม่จาก sel.getParams()
    if (fixed.panel.isViewStale(cur)) { fixed.step = cur; fixed.render(); } // = app.js onPipelineChange → renderDetail()
    for (let t = 0; t < 5; t++) fixed.frame(newNotes);
    assert.ok(fixed.container.all('detail-note-info').length > 0, 'rebuild แล้วยังไม่เห็น info ของวิธีใหม่');
  } finally { base.delete(); }
});

ntest('nodeid-17 (static) app.js: renderDetail ใช้ stepAndNotesOf/infoTextsOf · onPipelineChange เรียก isViewStale → renderDetail', () => {
  const strip = (s) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '').replace(/\s\/\/.*$/gm, '');
  const app = strip(fs.readFileSync(path.join(SRC_DIR, 'app.js'), 'utf8'));
  assert.ok(/const \{ step, notes \} = stepAndNotesOf\(sel, state\.notes\)/.test(app));
  assert.ok(/updateInfo\(infoTextsOf\(notes\)\)/.test(app));
  assert.ok(/isViewStale\([^)]*\)[^;]*\)\s*renderDetail\(\)/.test(app), 'onPipelineChange ไม่ rebuild เมื่อวิธีเปลี่ยนนอกแผง');
});

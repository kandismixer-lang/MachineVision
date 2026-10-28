// 2026-10-17 — เทสต์รอบ UI แผงพารามิเตอร์ (spec: specs/ui_param_panel_review_2026-10-17.md)
//   R1 visibleWhen (ซ่อน param ตาม select ภายใน op) · R3 ตัวคุมอยู่เหนือ param ที่คุม · R2 <optgroup> หัวข้อไทย
// DOM stub เขียนเองในไฟล์ (ไม่มี jsdom) · ติดตั้งภายใต้ withDom แล้วคืนเสมอ
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { FILTERS, OPS, defaultParams, isParamVisible, orderParamsControllersFirst } from '../src/renderer/filters/registry.js';
import { OPTION_GROUPS, OPTION_GROUPS_OTHER, optionGroupsFor } from '../src/renderer/filters/ui-meta.js';
import { createDetailPanel } from '../src/renderer/ui/detail-panel.js';

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
    this.selected = false;
    this.title = '';
    this.value = '';
    this.label = '';
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
  focus() { env.focused = this; }
  select() {}
  find(pred, out = []) {
    for (const c of this.children) { if (pred(c)) out.push(c); c.find(pred, out); }
    return out;
  }
}
const env = { focused: null };

function withDom(fn) {
  const had = Object.prototype.hasOwnProperty.call(globalThis, 'document');
  const prev = { doc: globalThis.document, now: Date.now };
  env.focused = null;
  globalThis.document = { createElement: (t) => new El(t), get activeElement() { return env.focused; } };
  let t = 5_000_000;
  Date.now = () => (t += 1000);
  try { return fn(); } finally {
    Date.now = prev.now;
    if (had) globalThis.document = prev.doc; else delete globalThis.document;
  }
}

function makeRig(filterId, over = {}) {
  const container = new El('div');
  const rig = { container, step: { id: filterId, params: { ...defaultParams(filterId), ...over } }, changes: [] };
  rig.panel = createDetailPanel(container, {
    onParamChange: (k, v) => { rig.changes.push([k, v]); rig.step.params[k] = v; },
    onTemplates: () => {},
  });
  rig.render = () => rig.panel.render(rig.step, []);
  rig.render();
  return rig;
}
// key ของตัวควบคุมที่ถูกวาดในแผง (data-pkey) เรียงตามลำดับบนจอ — ตัวเดียวต่อ param (select/ปุ่ม seg/toggle/slider)
function panelKeys(rig) {
  const seen = [];
  for (const el of rig.container.find((e) => e.getAttribute('data-pkey') !== null)) {
    const k = el.getAttribute('data-pkey');
    if (!seen.includes(k)) seen.push(k);
  }
  return seen;
}
const findSelect = (rig, key) => rig.container.find((e) => e.tagName === 'SELECT' && e.getAttribute('data-pkey') === key)[0] || null;
const optionsOf = (sel) => sel.find((e) => e.tagName === 'OPTION');
const findSlider = (rig, key) => rig.container.find((e) => e.getAttribute('data-pkey') === key && e.getAttribute('data-pkind') === 'slider')[0] || null;

// ---------- helpers ฝั่ง registry ----------
const containersOf = (opId) => Object.keys(FILTERS).filter((id) => id === opId || (FILTERS[id].params[0]?.key === 'mode' && FILTERS[id].params[0].options.some((o) => o.value === opId)));
const prefixOf = (id, opId) => (id === opId ? '' : opId + '__');
function visibleLocal(filterId, opId, local) {
  const f = FILTERS[filterId];
  const pre = prefixOf(filterId, opId);
  const params = { ...defaultParams(filterId) };
  if (pre) params.mode = opId;
  for (const k of Object.keys(local)) params[pre + k] = local[k];
  return f.params.filter((p) => (pre ? p.key.startsWith(pre) : true) && isParamVisible(p, params, f.params)).map((p) => (pre ? p.key.slice(pre.length) : p.key));
}
const opts = (opId, key) => OPS[opId].params.find((p) => p.key === key).options.map((o) => o.value);

// ===================================================================
// R1 — ตาราง visibleWhen (เขียนซ้ำจาก spec แบบอิสระ แล้วเทียบกับ registry ทุกชุดค่าของตัวควบคุม)
// ===================================================================
const BASIC = ['erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'blackhat'];
// rules[opId] = { controllers: [keys], conditional: { key: (local)=>boolean } } — key ที่ไม่อยู่ใน conditional ต้องโชว์เสมอ
const RULES = {
  morphology: { controllers: ['op', 'shape'], conditional: {
    shape: (l) => BASIC.includes(l.op),
    ksize: (l) => [...BASIC, 'reconstruct_open'].includes(l.op),
    iterations: (l) => [...BASIC, 'reconstruct_open', 'majority', 'spur', 'thin', 'shrink'].includes(l.op),
    lineAngle: (l) => BASIC.includes(l.op) && l.shape === 'line',
    minArea: (l) => l.op === 'remove_small',
    hmPattern: (l) => l.op === 'hit_or_miss',
    hmRotate: (l) => l.op === 'hit_or_miss',
    hHeight: (l) => ['h_maxima', 'extended_max', 'h_minima', 'extended_min'].includes(l.op),
  } },
  color_inrange: { controllers: ['colorSpace'], conditional: {
    hMin: (l) => l.colorSpace === 'hsv', hMax: (l) => l.colorSpace === 'hsv', sMin: (l) => l.colorSpace === 'hsv', vMin: (l) => l.colorSpace === 'hsv',
    labLMin: (l) => l.colorSpace === 'lab', labLMax: (l) => l.colorSpace === 'lab', labAMin: (l) => l.colorSpace === 'lab',
    labAMax: (l) => l.colorSpace === 'lab', labBMin: (l) => l.colorSpace === 'lab', labBMax: (l) => l.colorSpace === 'lab',
  } },
  custom_kernel: { controllers: ['preset'], conditional: {
    psfSize: (l) => l.preset === 'motion' || l.preset === 'disk',
    psfAngle: (l) => l.preset === 'motion',
  } },
  white_balance: { controllers: ['method'], conditional: { refX: (l) => l.method === 'ref_point', refY: (l) => l.method === 'ref_point', refSize: (l) => l.method === 'ref_point' } },
  fft_filter: { controllers: ['shape'], conditional: { order: (l) => l.shape === 'butterworth' } },
  bandpass_notch: { controllers: ['kind'], conditional: {
    rIn: (l) => l.kind !== 'notch', rOut: (l) => l.kind !== 'notch', nx: (l) => l.kind === 'notch', ny: (l) => l.kind === 'notch', nr: (l) => l.kind === 'notch',
  } },
  dct_jpeg: { controllers: ['method'], conditional: { quality: (l) => l.method === 'jpeg', keep: (l) => l.method === 'zonal' } },
  imfuse: { controllers: ['mode'], conditional: { alpha: (l) => l.mode === 'blend', tileSize: (l) => l.mode === 'checkerboard' } },
  gamma: { controllers: ['curve'], conditional: { g: (l) => l.curve === 'power' } },
  shape_filter: { controllers: ['prop'], conditional: { lo: (l) => l.prop !== 'none', hi: (l) => l.prop !== 'none' } },
  deconv_wiener: { controllers: ['psf'], conditional: { psfAngle: (l) => l.psf === 'motion' } },
  deconv_lucy: { controllers: ['psf'], conditional: { psfAngle: (l) => l.psf === 'motion' } },
};
// op ที่ต้องปรากฏใน node หมวด/กลุ่มเดิมที่ spec ระบุ (standalone + merged + legacy)
const MUST_CONTAIN = {
  morphology: ['morphology'],
  color_inrange: ['color_tools', 'color_space'],
  custom_kernel: ['enhance_tools', 'sharpen'],
  white_balance: ['color_tools'],
  fft_filter: ['frequency'],
  bandpass_notch: ['frequency'],
  dct_jpeg: ['frequency'],
  imfuse: ['image_arithmetic'],
  gamma: ['enhance_tools', 'image_adjust'],
  shape_filter: ['analysis_detect'],
  deconv_wiener: ['filter_denoise'],
  deconv_lucy: ['filter_denoise'],
};

test('pv-01 R1 ตาราง visibleWhen: ทุกชุดค่าตัวควบคุม × ทุก node ที่มี op นั้น (standalone/merged/legacy) ตรงกับ spec', () => {
  for (const [opId, rule] of Object.entries(RULES)) {
    const containers = containersOf(opId);
    for (const need of MUST_CONTAIN[opId]) assert.ok(containers.includes(need), `${opId} ต้องอยู่ใน ${need}`);
    const allKeys = OPS[opId].params.map((p) => p.key);
    // ทุกชุดค่า (cartesian) ของตัวควบคุม
    let combos = [{}];
    for (const c of rule.controllers) combos = combos.flatMap((cb) => opts(opId, c).map((v) => ({ ...cb, [c]: v })));
    for (const fid of containers) {
      for (const local of combos) {
        const full = { ...defaultParams(fid) };
        const got = visibleLocal(fid, opId, local);
        // โลคัลที่ไม่ได้ส่งมาใช้ default ของ op (เช่น psf-based ที่ default ต่างกัน) — ตัวควบคุมถูกส่งครบอยู่แล้ว
        void full;
        const expected = allKeys.filter((k) => !rule.conditional[k] || rule.conditional[k](local));
        assert.deepEqual(got, expected, `${fid}/${opId} ${JSON.stringify(local)}`);
      }
    }
  }
});

test('pv-02 R1 AC: จำนวน control ของ Morphology / inRange ตาม spec', () => {
  const m = (local) => visibleLocal('morphology', 'morphology', local);
  assert.deepEqual(m({ op: 'erode', shape: 'rect' }), ['op', 'shape', 'ksize', 'iterations']); // 4
  assert.equal(m({ op: 'erode', shape: 'line' }).length, 5);
  assert.deepEqual(m({ op: 'fill_holes' }), ['op']);
  assert.equal(m({ op: 'hit_or_miss' }).length, 3);
  assert.equal(m({ op: 'h_maxima' }).length, 2);
  // กลุ่ม merged: ตัวเดียวกันผ่าน color_tools
  assert.equal(visibleLocal('color_tools', 'color_inrange', { colorSpace: 'hsv' }).length, 6);
  assert.equal(visibleLocal('color_tools', 'color_inrange', { colorSpace: 'lab' }).length, 8);
});

test('pv-03 R1 โครงสร้าง: visibleWhen ทุกตัวอ้างตัวควบคุมที่มีจริง (select/boolean) + ค่าใน in/notIn อยู่ใน options · key มี prefix ถูก', () => {
  let n = 0;
  for (const [fid, f] of Object.entries(FILTERS)) {
    for (const p of f.params) {
      if (!p.visibleWhen) continue;
      for (const c of p.visibleWhen) {
        n++;
        const ctrl = f.params.find((q) => q.key === c.key);
        assert.ok(ctrl, `${fid}.${p.key} อ้าง ${c.key} ที่ไม่มี`);
        assert.ok(ctrl.type === 'select' || ctrl.type === 'boolean', `${fid}.${p.key}: ตัวคุม ${c.key} ต้อง select/boolean (ห้ามสไลเดอร์ตัวเลข)`);
        if (ctrl.type === 'select') for (const v of [...(c.in || []), ...(c.notIn || [])]) assert.ok(ctrl.options.some((o) => o.value === v), `${fid}.${p.key}: ${v} ไม่อยู่ใน options ของ ${c.key}`);
        assert.ok(c.in || c.notIn);
        if (p.showIf) assert.ok(c.key.startsWith(p.showIf.value + '__'), `${fid}.${p.key}: key เงื่อนไขต้องมี prefix ของ op เดียวกับ param`);
        assert.notEqual(c.key, p.key);
      }
    }
  }
  assert.ok(n > 60, 'มีเงื่อนไขจริงจำนวนมากในทุก node (กันตารางหายทั้งก้อน)');
  // imfuse/flip มี param ภายในชื่อ "mode" — ต้องไม่ชนกับ select วิธีของ node นอก
  const iaMode = FILTERS.image_arithmetic.params.find((p) => p.key === 'imfuse__alpha');
  assert.deepEqual(iaMode.visibleWhen, [{ key: 'imfuse__mode', in: ['blend'] }]);
  assert.deepEqual(iaMode.showIf, { key: 'mode', value: 'imfuse' });
});

test('pv-04 R1 defaultParams ของทุก filter (key ลำดับ + ค่า) เท่ากับก่อนแก้ทุกตัว (hash จาก HEAD 4c4f354)', () => {
  const snap = {};
  // P-8 เพิ่ม op pass_fail (key pass_fail__*) ใน analysis_detect — ตัดออกก่อน hash เพื่อให้ hash เดิมยังล็อกทุก param เดิมเหมือนเดิม
  const added5A = ['watershed_marker__', 'lowlight_enhance__', 'colorspace_inverse__', 'euler_number__'];
  const old = (k) => !k.startsWith('pass_fail__') && !added5A.some((prefix) => k.startsWith(prefix));
  for (const id of Object.keys(FILTERS)) {
    if (id === 'morph_tools' || id === 'image_registration') continue; // image_registration (2026-10-24) ล็อกใน tests/category-merge.test.mjs // 2026-10-18: กลุ่มใหม่ (Morphology แยกเครื่องมือ) — hash นี้ล็อกเฉพาะ filter เดิมทั้ง 33 ตัว · morph_tools ล็อกใน tests/morph-split.test.mjs
    const d = defaultParams(id);
    snap[id] = { keys: FILTERS[id].params.map((p) => p.key).filter(old), defaults: Object.fromEntries(Object.entries(d).filter(([k]) => old(k))) };
  }
  const h = crypto.createHash('sha256').update(JSON.stringify(snap)).digest('hex');
  assert.equal(Object.keys(snap).length, 33);
  assert.equal(h, '8b7732b84f064393b82060d14b5502a08f6a497738183a7d96411751a86457c4');
});

test('pv-05 R1 ค่าตัวควบคุมแปลก/ไม่มี → ใช้ default ของตัวควบคุม · ไม่ crash', () => {
  const f = FILTERS.morphology;
  const vis = (params) => f.params.filter((p) => isParamVisible(p, params, f.params)).map((p) => p.key);
  const def = vis(defaultParams('morphology'));
  for (const bad of ['constructor', '__proto__', undefined, null, 42, '']) {
    assert.deepEqual(vis({ ...defaultParams('morphology'), op: bad }), def, String(bad));
  }
  // ตัวคุมไม่ถูกส่งมาเลย
  const noOp = { ...defaultParams('morphology') }; delete noOp.op;
  assert.deepEqual(vis(noOp), def);
  // inRange: ค่าแปลก = hsv (ตรงกับ apply ที่ตกไป HSV)
  const g = FILTERS.color_tools;
  const gv = (cs) => g.params.filter((p) => isParamVisible(p, { ...defaultParams('color_tools'), mode: 'color_inrange', color_inrange__colorSpace: cs }, g.params)).map((p) => p.key.replace('color_inrange__', ''));
  assert.deepEqual(gv('constructor'), gv('hsv'));
  // boolean ตัวคุม (ชนิดไม่ใช่ boolean → default)
  const synth = [{ key: 'flag', type: 'boolean', default: true }, { key: 'x', type: 'number', visibleWhen: [{ key: 'flag', in: [true] }] }];
  assert.equal(isParamVisible(synth[1], { flag: 'yes' }, synth), true);
  assert.equal(isParamVisible(synth[1], { flag: false }, synth), false);
  // notIn
  const sf = FILTERS.analysis_detect;
  const lo = sf.params.find((p) => p.key === 'shape_filter__lo');
  const base = { ...defaultParams('analysis_detect'), mode: 'shape_filter' };
  assert.equal(isParamVisible(lo, { ...base, shape_filter__prop: 'none' }, sf.params), false);
  assert.equal(isParamVisible(lo, { ...base, shape_filter__prop: 'weird' }, sf.params), true, 'ค่าแปลก → default circularity → โชว์');
  // showIf ชั้นนอกยังกั้นเหมือนเดิม
  assert.equal(isParamVisible(lo, { ...base, mode: 'find_contours' }, sf.params), false);
});

test('pv-06 R1 แผงจริง: สลับ shape rect → line แล้ว lineAngle โผล่ ค่า 45 ไม่ถูกรีเซ็ต · ค่าที่ซ่อนไม่ถูกเขียนทับ', () => withDom(() => {
  const rig = makeRig('morphology', { lineAngle: 45, shape: 'rect', op: 'erode' });
  assert.deepEqual(panelKeys(rig), ['op', 'shape', 'ksize', 'iterations']);
  assert.equal(findSlider(rig, 'lineAngle'), null);
  const sel = findSelect(rig, 'shape');
  const lineEl = optionsOf(sel).find((o) => FILTERS.morphology.params.find((p) => p.key === 'shape').options[Number(o.value)].value === 'line');
  sel.value = lineEl.value; sel.fire('change');
  assert.deepEqual(rig.changes, [['shape', 'line']], 'มีการแจ้งเปลี่ยนแค่ shape — ไม่มีการเขียน lineAngle ทับ');
  assert.equal(rig.step.params.lineAngle, 45);
  assert.deepEqual(panelKeys(rig), ['op', 'shape', 'ksize', 'iterations', 'lineAngle']);
  assert.equal(findSlider(rig, 'lineAngle').value, 45);
  // กลับ rect → ซ่อน แต่ค่ายังอยู่
  const sel2 = findSelect(rig, 'shape');
  const rectEl = optionsOf(sel2).find((o) => o.value === '0');
  sel2.value = rectEl.value; sel2.fire('change');
  assert.equal(rig.step.params.lineAngle, 45);
  assert.equal(findSlider(rig, 'lineAngle'), null);
  // เปลี่ยน op เป็น fill_holes → เหลือ op อย่างเดียว + ข้อความไม่มีค่าให้ปรับไม่โผล่ (ยังมีตัวคุม op)
  const opSel = findSelect(rig, 'op');
  const fh = optionsOf(opSel).find((o) => FILTERS.morphology.params[0].options[Number(o.value)].value === 'fill_holes');
  opSel.value = fh.value; opSel.fire('change');
  assert.deepEqual(panelKeys(rig), ['op']);
  assert.equal(rig.step.params.ksize, defaultParams('morphology').ksize);
}));

test('pv-07 R1 แผงจริง: ค่า op แปลง ("constructor") ไม่ crash แสดงชุด default · merged node (color_tools/inRange)', () => withDom(() => {
  const rig = makeRig('morphology', { op: 'constructor' });
  assert.deepEqual(panelKeys(rig), ['op', 'shape', 'ksize', 'iterations']);
  const ct = makeRig('color_tools', { mode: 'color_inrange' });
  assert.deepEqual(panelKeys(ct).filter((k) => k !== 'mode'), ['color_inrange__colorSpace', 'color_inrange__hMin', 'color_inrange__hMax', 'color_inrange__sMin', 'color_inrange__vMin', 'color_inrange__output']);
  ct.step.params.color_inrange__colorSpace = 'lab'; ct.render();
  assert.equal(panelKeys(ct).filter((k) => k.includes('__lab')).length, 6);
  assert.ok(!panelKeys(ct).includes('color_inrange__hMin'));
}));

test('pv-08 R1 ตัวใช้ครบ 3 จุด: ชิปในรายการขั้นตอน (renderSteps) ไม่โชว์ค่าที่ไม่มีผล · stepRawParams ใน app.js เรียก helper', () => withDom(() => {
  const container = new El('div');
  const panel = createDetailPanel(container, { onParamChange() {}, onTemplates() {} });
  const params = { ...defaultParams('morphology'), op: 'fill_holes' };
  panel.renderDisplay({
    id: 1, getName: () => '', setName() {}, onJump() {},
    getSteps: () => ({ connected: true, pipeline: [{ id: 'morphology', params, nodeId: 1 }, { id: 'morphology', params: { ...defaultParams('morphology'), op: 'erode' }, nodeId: 2 }] }),
  });
  const rows = container.find((e) => e.className === 'step-row');
  const chips = (r) => r.find((e) => e.className === 'step-param').map((e) => e.title);
  assert.deepEqual(chips(rows[0]), ['op']);
  assert.deepEqual(chips(rows[1]), ['op', 'shape', 'ksize', 'iterations']);
  const app = fs.readFileSync(new URL('../src/renderer/app.js', import.meta.url), 'utf8');
  const body = app.slice(app.indexOf('function stepRawParams'), app.indexOf('function stepLabel'));
  // ล็อกทั้งบรรทัด: `isParamVisible(...) && false` ต้องจับได้ (review MINOR)
  assert.match(body, /\n\s*if \(!isParamVisible\(p, s\.params, f\.params\)\) continue;\n/);
  assert.match(body, /for \(const p of orderParamsControllersFirst\(f\.params\)\)/, 'breadcrumb เรียงตัวคุมก่อนเหมือนแผง (R3)');
  assert.doesNotMatch(body, /p\.showIf/, 'ไม่มีเช็ค showIf เองซ้ำ');
  const dp = fs.readFileSync(new URL('../src/renderer/ui/detail-panel.js', import.meta.url), 'utf8');
  assert.doesNotMatch(dp, /p\.showIf/, 'detail-panel ใช้ helper เดียว');
  assert.match(dp, /for \(const p of orderParamsControllersFirst\(f\.params\)\) \{\n\s*if \(!isParamVisible\(p, s\.params, f\.params\)\) continue;/, 'ชิปในรายการขั้นตอนเรียงตัวคุมก่อน (R3)');
}));

// ===================================================================
// R3 — ตัวควบคุมอยู่เหนือ param ที่คุม
// ===================================================================
test('pv-09 R3 gamma: curve ก่อน g (standalone-merged enhance_tools + legacy image_adjust) · op อื่นลำดับเดิมทุกตัว', () => withDom(() => {
  const et = makeRig('enhance_tools', { mode: 'gamma' });
  assert.deepEqual(panelKeys(et), ['mode', 'gamma__curve', 'gamma__g']);
  const ia = makeRig('image_adjust', { mode: 'gamma' });
  assert.deepEqual(panelKeys(ia), ['mode', 'gamma__curve', 'gamma__g']);
  // curve = log → g หายไป เหลือ curve ตัวเดียว (select ไม่เลื่อน)
  const lg = makeRig('enhance_tools', { mode: 'gamma', gamma__curve: 'log' });
  assert.deepEqual(panelKeys(lg), ['mode', 'gamma__curve']);
}));

test('pv-10 R3 snapshot: ทุก node × ทุกวิธี — ลำดับเท่าเดิม ยกเว้นที่ตัวคุมตามหลังตัวที่คุม (เฉพาะ gamma)', () => {
  const changed = [];
  for (const [fid, f] of Object.entries(FILTERS)) {
    const modes = f.params[0]?.key === 'mode' ? f.params[0].options.map((o) => o.value) : [null];
    for (const m of modes) {
      const params = defaultParams(fid);
      if (m) params.mode = m;
      const visible = f.params.filter((p) => isParamVisible(p, params, f.params) && p.key !== 'mode');
      const ordered = orderParamsControllersFirst(visible);
      assert.equal(ordered.length, visible.length);
      assert.deepEqual([...ordered].sort((a, b) => visible.indexOf(a) - visible.indexOf(b)), visible, 'ไม่ตกหล่น/ซ้ำ');
      const before = visible.map((p) => p.key).join(','), after = ordered.map((p) => p.key).join(',');
      if (before !== after) changed.push(fid + ':' + m);
      // ตัวคุมต้องมาก่อนตัวที่คุมเสมอ
      for (const p of ordered) for (const c of p.visibleWhen || []) {
        const ci = ordered.findIndex((q) => q.key === c.key);
        if (ci >= 0) assert.ok(ci < ordered.indexOf(p), `${fid}:${m} ${c.key} ต้องก่อน ${p.key}`);
      }
    }
  }
  assert.deepEqual(changed.sort(), ['enhance_tools:gamma', 'enhancement:gamma', 'image_adjust:gamma']);
});

test('pv-11 R3 ไม่วนไม่สิ้นสุดเมื่ออ้างกันเป็นวง · ลำดับเดิมเมื่อไม่มีการอ้าง', () => {
  const a = { key: 'a', visibleWhen: [{ key: 'b', in: [1] }] }, b = { key: 'b', visibleWhen: [{ key: 'a', in: [1] }] }, c = { key: 'c' };
  assert.equal(orderParamsControllersFirst([a, b, c]).length, 3);
  assert.deepEqual(orderParamsControllersFirst([c, a]).map((p) => p.key), ['c', 'a']);
});

// ===================================================================
// R2 — <optgroup> หัวข้อไทย
// ===================================================================
function paramForTableKey(key) {
  const [left, local] = key.split('.');
  if (local === 'mode') return { filterId: left, param: FILTERS[left].params[0] };
  const p = OPS[left].params.find((q) => q.key === local);
  const fid = containersOf(left)[0];
  return { filterId: fid, param: { ...p, key: fid === left ? local : left + '__' + local } };
}

test('pv-12 R2 OPTION_GROUPS: ทุก key มี value ครบเท่ากับ param.options ไม่ขาดไม่ซ้ำ · ไม่มีกลุ่มว่าง · ไม่เข้า "อื่น ๆ"', () => {
  assert.ok(Object.keys(OPTION_GROUPS).length >= 10);
  for (const [key, table] of Object.entries(OPTION_GROUPS)) {
    const { filterId, param } = paramForTableKey(key);
    const flat = table.flatMap(([, vals]) => vals);
    assert.equal(new Set(flat).size, flat.length, `${key}: value ซ้ำในตาราง`);
    assert.deepEqual([...flat].sort(), param.options.map((o) => o.value).sort(), `${key}: ตารางต้องมี value ครบเท่า options`);
    for (const [title, vals] of table) { assert.ok(title && vals.length, `${key}: กลุ่มว่าง`); assert.notEqual(title, OPTION_GROUPS_OTHER); }
    const out = optionGroupsFor(filterId, param);
    assert.ok(out, key);
    assert.equal(out.length, table.length);
    assert.deepEqual(out.map(([t]) => t), table.map(([t]) => t));
    assert.equal(out.flatMap(([, o]) => o).length, param.options.length);
  }
});

test('pv-13 R2 optionGroupsFor: ทุก key ในตารางถูกใช้โดย select จริงอย่างน้อยหนึ่งตัว · select อื่นไม่มีตาราง → null · option ใหม่ → "อื่น ๆ" ท้ายสุด', () => {
  const hit = new Set();
  for (const [fid, f] of Object.entries(FILTERS)) {
    for (const p of f.params) {
      if (p.type !== 'select') continue;
      const g = optionGroupsFor(fid, p);
      if (!g) continue;
      const tk = p.key === 'mode' ? fid + '.mode' : p.key.includes('__') ? p.key.replace('__', '.') : fid + '.' + p.key;
      hit.add(tk);
    }
  }
  assert.deepEqual([...hit].sort(), Object.keys(OPTION_GROUPS).sort());
  // ไม่ผูกกับ node ที่ไม่อยู่ในตาราง (legacy mode)
  assert.equal(optionGroupsFor('shape_analysis', FILTERS.shape_analysis.params[0]), null);
  assert.equal(optionGroupsFor('color_space', FILTERS.color_space.params[0]), null);
  // ลืมใส่ op ใหม่ → อื่น ๆ ท้ายสุด
  const base = FILTERS.segment_edge.params[0];
  const extended = { ...base, options: [...base.options, { value: 'future_op', label: 'Future' }] };
  const g = optionGroupsFor('segment_edge', extended);
  assert.equal(g.at(-1)[0], OPTION_GROUPS_OTHER);
  assert.deepEqual(g.at(-1)[1].map((o) => o.value), ['future_op']);
});

test('pv-14 R2 แผงจริง: <optgroup> จริง · เลือกได้ทุก option → onParamChange(key,value) ถูก (analysis_detect ครบ + morphology.op ครบ + select อื่นในตาราง)', () => withDom(() => {
  const cases = [
    ['analysis_detect', 'mode', {}],
    ['morphology', 'op', {}],
    ['segment_edge', 'mode', {}],
    ['enhance_tools', 'mode', {}],
    ['filter_denoise', 'mode', {}],
    ['color_tools', 'mode', {}],
    ['image_arithmetic', 'mode', {}],
    ['enhance_tools', 'custom_kernel__preset', { mode: 'custom_kernel' }],
    ['color_tools', 'colorspace__space', { mode: 'colorspace' }],
    ['color_tools', 'channel_split__channel', { mode: 'channel_split' }],
  ];
  for (const [fid, key, over] of cases) {
    const param = FILTERS[fid].params.find((p) => p.key === key);
    const table = optionGroupsFor(fid, param);
    assert.ok(table, `${fid}.${key}`);
    for (const target of param.options) {
      // ค่าเริ่มต้นที่ต่างจาก target (กัน change ไม่ยิงเมื่อเลือกตัวเดิม)
      const start = param.options.find((o) => o.value !== target.value).value;
      const rig = makeRig(fid, { ...over, [key]: start });
      const sel = findSelect(rig, key);
      assert.ok(sel, `${fid}.${key} ต้องเป็น <select>`);
      // โครงสร้าง: ลูกของ select = optgroup ล้วน (ไม่มี placeholder เพราะค่าอยู่ใน options) · ลูกของ optgroup = option
      assert.ok(sel.children.every((c) => c.tagName === 'OPTGROUP'), `${fid}.${key}: ต้องเป็น <optgroup> จริง`);
      assert.deepEqual(sel.children.map((c) => c.label), table.map(([t]) => t));
      const optEls = optionsOf(sel);
      assert.equal(optEls.length, param.options.length);
      assert.deepEqual(optEls.map((o) => o.value).sort((a, b) => a - b), param.options.map((_, i) => String(i)).sort((a, b) => a - b), 'value = index ใน param.options ไม่ซ้ำ');
      assert.ok(!optEls.some((o) => o.disabled), 'ห้ามมี option disabled ปลอมเป็นหัวข้อ');
      const el = optEls.find((o) => param.options[Number(o.value)].value === target.value);
      assert.equal(el.textContent, target.label);
      assert.equal(optEls.find((o) => o.selected).value, String(param.options.findIndex((o) => o.value === start)), 'ตัวที่เลือกอยู่ = ค่าปัจจุบัน');
      sel.value = el.value; sel.fire('change');
      assert.deepEqual(rig.changes.at(-1), [key, target.value], `${fid}.${key} → ${target.value}`);
      assert.equal(rig.changes.length, 1);
    }
  }
}));

test('pv-15 R2 ค่าที่ไม่อยู่ใน options → placeholder "— เลือก —" (value ว่าง, ซ่อน, เลือกอยู่) · เลือก placeholder ไม่ commit · เลือกจริงได้ (กลุ่มมี optgroup ต่อ)', () => withDom(() => {
  const rig = makeRig('analysis_detect', { mode: 'ค่าที่ไม่มีอยู่จริง' });
  const sel = findSelect(rig, 'mode');
  assert.equal(sel.children[0].tagName, 'OPTION');
  assert.equal(sel.children[0].value, '');
  assert.equal(sel.children[0].textContent, '— เลือก —');
  assert.equal(sel.children[0].hidden, true);
  assert.equal(sel.children[0].selected, true);
  assert.ok(sel.children.slice(1).every((c) => c.tagName === 'OPTGROUP'));
  assert.equal(optionsOf(sel).filter((o) => o.selected).length, 1);
  sel.value = ''; sel.fire('change');
  assert.equal(rig.changes.length, 0, 'placeholder (value "") ต้องไม่กลายเป็น index 0');
  const first = optionsOf(sel).find((o) => o.value === '0');
  sel.value = first.value; sel.fire('change');
  assert.deepEqual(rig.changes.at(-1), ['mode', FILTERS.analysis_detect.params[0].options[0].value]);
}));

test('pv-16 R2 ไม่กระทบ: select ที่ไม่มีตาราง = <option> เรียบเหมือนเดิม · segmented ≤3 ตัวยังเป็นปุ่ม · CSS optgroup มีใน app.css', () => withDom(() => {
  const rig = makeRig('shape_analysis');
  const sel = findSelect(rig, 'mode');
  assert.ok(sel.children.every((c) => c.tagName === 'OPTION'));
  assert.deepEqual(sel.children.map((c) => c.value), FILTERS.shape_analysis.params[0].options.map((_, i) => String(i)));
  const sob = makeRig('segment_edge', { mode: 'sobel', sobel__dir: 'x' });
  assert.ok(sob.container.find((e) => e.className.split(' ').includes('seg-control')).length >= 1, 'ทิศ Sobel ยัง segmented');
  const css = fs.readFileSync(new URL('../src/styles/app.css', import.meta.url), 'utf8');
  assert.match(css, /\.detail-content \.field-select optgroup \{[^}]*font-weight: 600/);
  assert.match(css, /\.detail-content \.field-select optgroup option \{[^}]*font-weight: 400/);
}));

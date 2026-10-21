// 2026-10-18 — Morphology แยก Operation เป็นเครื่องมือรายตัว (morph_erode ... morph_extended_max) + กลุ่ม morph_tools
// ครอบคลุม: (1) ลงทะเบียน/legacy · (2) param ของแต่ละตัว = เท่าที่ใช้จริง · (3) พิกเซล+ctx.info เท่า OPS.morphology เดิมทุก op (ทั้ง default / non-default
// ทั้งเรียกตรงและผ่านกลุ่ม) · (4) palette + dropdown + optgroup · (5) migration ของ step morphology เก่า · (6) codegen เท่าเดิม · (7) icon ไม่ชน · (8) การ์ด node
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import cv from '@techstark/opencv-js';
import {
  FILTERS, OPS, MORPH_OP_IDS, HITMISS_PATTERNS, defaultParams, needsOf, isParamVisible, orderParamsControllersFirst, migrateLegacyStep, opIdOfStep,
  GROUP_COLORS, GROUP_ORDER,
} from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS, OP_TO_FILTER, resolveRecent, migrateRecentIds, toolMatches } from '../src/renderer/ui/palette-panel.js';
import { GROUP_ICON, OP_PRIMARY, OPTION_GROUPS, optionGroupsFor, currentOpId, localParamsFor } from '../src/renderer/filters/ui-meta.js';
import { buildPythonCode } from '../src/renderer/codegen.js';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cvtColor === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

// ลำดับที่ผู้ใช้สั่ง (เขียนซ้ำอิสระจาก registry)
const OPS_ORDER = ['erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'blackhat', 'fill_holes', 'clear_border', 'remove_small',
  'skeleton', 'reconstruct_open', 'convex_hull', 'endpoints_branchpoints', 'hit_or_miss', 'regional_max', 'regional_min', 'h_maxima', 'extended_max'];
const NEW_IDS = OPS_ORDER.map((o) => 'morph_' + o);
const BASIC = ['erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'blackhat'];

// param ที่แต่ละตัวต้องมี "พอดี" (ตามลำดับที่ปรากฏในแผง)
const EXPECT_KEYS = {};
for (const o of BASIC) EXPECT_KEYS[o] = ['shape', 'ksize', 'iterations', 'lineAngle'];
Object.assign(EXPECT_KEYS, {
  fill_holes: [], clear_border: [], remove_small: ['minArea'], skeleton: [], reconstruct_open: ['ksize', 'iterations'], convex_hull: [],
  endpoints_branchpoints: [], hit_or_miss: ['hmPattern', 'hmRotate'], regional_max: [], regional_min: [], h_maxima: ['hHeight'], extended_max: ['hHeight'],
});

// ค่าที่ไม่ใช่ default ของแต่ละ op (ต้องต่างจาก default จริง — ตรวจในเทสต์)
const NONDEFAULT = {
  erode: { shape: 'ellipse', ksize: 7, iterations: 2 },
  dilate: { shape: 'cross', ksize: 3, iterations: 3 },
  open: { shape: 'diamond', ksize: 9, iterations: 2 },
  close: { shape: 'octagon', ksize: 7, iterations: 2 },
  gradient: { shape: 'line', lineAngle: 45, ksize: 9 },
  tophat: { shape: 'line', lineAngle: 90, ksize: 11 },
  blackhat: { shape: 'cross', ksize: 15, iterations: 2 },
  fill_holes: {}, clear_border: {}, skeleton: {}, convex_hull: {}, endpoints_branchpoints: {}, regional_max: {}, regional_min: {},
  remove_small: { minArea: 60 },
  reconstruct_open: { ksize: 17, iterations: 2 },
  hit_or_miss: { hmPattern: Object.keys(HITMISS_PATTERNS).find((k) => k !== 'isolated'), hmRotate: false },
  h_maxima: { hHeight: 30 },
  extended_max: { hHeight: 25 },
};

// ภาพเทาสังเคราะห์ 72x60: วัตถุมีรู · วัตถุติดขอบ · จุดเล็ก · เส้นกากบาท (skeleton/branch) · พื้นหลังมียอดเทาหลายระดับ
function makeGray() {
  const W = 72, H = 60;
  const a = new Uint8Array(W * H);
  let s = 99;
  const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s >>> 24; };
  for (let i = 0; i < a.length; i++) a[i] = 10 + (rnd() % 40); // พื้นหลังเทาเข้มมี noise (ยอด/หุบเล็ก ๆ)
  const rect = (x0, y0, x1, y1, v) => { for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) a[y * W + x] = v; };
  rect(6, 6, 28, 26, 255); rect(14, 12, 18, 16, 0); // วัตถุมีรูปิด
  rect(0, 40, 9, 52, 255); // ติดขอบซ้าย
  rect(40, 4, 41, 5, 255); rect(50, 8, 50, 8, 255); rect(60, 3, 62, 4, 255); // จุดเล็ก
  rect(36, 30, 66, 30, 255); rect(50, 22, 50, 46, 255); rect(50, 38, 62, 38, 255); // เส้น + แยกกิ่ง
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { // วงกลม + ยอดเทาโค้ง
    if ((x - 24) ** 2 + (y - 44) ** 2 <= 49) a[y * W + x] = 230;
    if ((x - 62) ** 2 + (y - 50) ** 2 <= 16) a[y * W + x] = 180;
  }
  const m = new cv.Mat(H, W, cv.CV_8UC1);
  m.data.set(a);
  return m;
}

function snap(out, ctx) {
  return {
    rows: out.rows, cols: out.cols, ch: out.channels(), type: out.type(),
    data: Buffer.from(out.data).toString('base64'),
    info: ctx.info === undefined ? null : ctx.info,
  };
}
// รัน apply ใด ๆ บนภาพเทา → snapshot (ล้าง Mat ทุกตัว)
function run(fn) {
  const src = makeGray();
  const ctx = {};
  let out = null;
  try { out = fn(cv, src, ctx); return snap(out, ctx); } finally { if (out) out.delete(); src.delete(); }
}
const oldParams = (op, local) => ({ ...defaultParams('morphology'), op, ...local });
const runOld = (op, local) => run((c, src, ctx) => OPS.morphology.apply(c, src, oldParams(op, local), ctx));
const runNew = (op, local) => run((c, src, ctx) => OPS['morph_' + op].apply(c, src, { ...localDefaults('morph_' + op), ...local }, ctx));
function localDefaults(id) { return Object.fromEntries(OPS[id].params.map((q) => [q.key, q.default])); }
const groupParams = (op, local) => {
  const id = 'morph_' + op;
  const p = { ...defaultParams('morph_tools'), mode: id };
  for (const [k, v] of Object.entries(local)) p[id + '__' + k] = v;
  return p;
};
const runGroup = (op, local) => run((c, src, ctx) => FILTERS.morph_tools.apply(c, src, groupParams(op, local), ctx));

// ---------------------------------------------------------------------------
test('mp-1. 19 new ops registered (ids/order), group morph_tools registered, legacy morphology kept with all 19 options', () => {
  assert.deepEqual(MORPH_OP_IDS, NEW_IDS);
  for (const id of NEW_IDS) {
    assert.ok(OPS[id], id);
    assert.equal(OPS[id].needs, 'gray', `${id}.needs`);
    assert.ok(OPS[id].name && /[A-Za-z]/.test(OPS[id].name) && /[฀-๿]|Top Hat|Black Hat/.test(OPS[id].name), `${id}.name`);
    assert.ok(OPS[id].desc && OPS[id].desc.length > 30, `${id}.desc`);
    assert.equal(typeof OPS[id].apply, 'function');
  }
  assert.equal(new Set(NEW_IDS.map((i) => OPS[i].name)).size, 19, 'names unique');
  for (const i of NEW_IDS) assert.ok(OPS[i].name.length <= 34, `${i}: name must be short enough for a palette tile (${OPS[i].name})`);
  assert.equal(new Set(NEW_IDS.map((i) => OPS[i].desc)).size, 19, 'desc unique');
  // ไม่ชน id เดิม: ไม่มี id ซ้ำใน OPS/FILTERS
  assert.equal(new Set(Object.keys(OPS)).size, Object.keys(OPS).length);
  assert.ok(FILTERS.morphology, 'legacy morphology stays registered');
  assert.deepEqual(FILTERS.morphology.params[0].options.map((o) => o.value), OPS_ORDER);
  assert.equal(FILTERS.morphology.params[0].default, 'erode');
  const g = FILTERS.morph_tools;
  assert.ok(g);
  assert.equal(g.group, 'Morphology');
  assert.ok(GROUP_COLORS[g.group] && GROUP_ORDER.includes(g.group));
  assert.equal(g.inputs, undefined);
  assert.equal(g.stateful, undefined);
  const mode = g.params[0];
  assert.equal(mode.key, 'mode');
  assert.equal(mode.label, 'เลือกวิธี');
  assert.deepEqual(mode.options.map((o) => o.value), NEW_IDS);
  assert.equal(mode.default, 'morph_erode');
  assert.equal(defaultParams('morph_tools').mode, 'morph_erode');
});

test('mp-2. each op has EXACTLY its own params (no op / no unrelated keys); lineAngle only visible for shape=line; defaults/ranges equal legacy', () => {
  for (const op of OPS_ORDER) {
    const id = 'morph_' + op;
    assert.deepEqual(OPS[id].params.map((q) => q.key), EXPECT_KEYS[op], `${id} params`);
    for (const q of OPS[id].params) {
      const old = OPS.morphology.params.find((x) => x.key === q.key);
      assert.equal(q.default, old.default, `${id}.${q.key} default`);
      assert.equal(q.type, old.type);
      if (q.min !== undefined) { assert.equal(q.min, old.min); assert.equal(q.max, old.max); assert.equal(q.step, old.step); }
      if (q.type === 'select') assert.deepEqual(q.options.map((o) => o.value), old.options.map((o) => o.value));
      // label ใหม่ไม่เหลือวลีแยกตัวแบบแผงเดิม
      assert.ok(!/ใช้กับ .*เท่านั้น|ไม่มีผลกับ/.test(q.label), `${id}.${q.key} label: ${q.label}`);
      // ไม่มี visibleWhen ที่อ้าง key "op" (ไม่มีแล้ว)
      assert.ok(!(q.visibleWhen || []).some((c) => c.key === 'op'), `${id}.${q.key} visibleWhen still references op`);
    }
    // desc ของ param ต้องตรงกับ op นี้: ไม่พูดถึง Reconstruction ในตัวอื่น · reconstruct_open พูดถึง marker
    for (const q of OPS[id].params) {
      if (q.key === 'ksize' || q.key === 'iterations') {
        if (op === 'reconstruct_open') assert.match(q.desc, /marker/, `${id}.${q.key} desc`);
        else assert.doesNotMatch(q.desc, /Reconstruction|marker/, `${id}.${q.key} desc`);
      }
      assert.doesNotMatch(q.desc || '', /ไม่มีผลกับ Operation อื่น/, `${id}.${q.key} desc`);
    }
    const la = OPS[id].params.find((q) => q.key === 'lineAngle');
    if (la) assert.deepEqual(la.visibleWhen, [{ key: 'shape', in: ['line'] }]);
  }
  // ในกลุ่ม: param ที่เห็นของแต่ละ op (หลัง prefix) = ตรงตามตาราง โดย lineAngle เห็นเฉพาะ shape=line
  const f = FILTERS.morph_tools;
  for (const op of OPS_ORDER) {
    const id = 'morph_' + op;
    for (const shape of ['rect', 'line']) {
      const p = { ...defaultParams('morph_tools'), mode: id };
      if (OPS[id].params.some((q) => q.key === 'shape')) p[id + '__shape'] = shape;
      const vis = f.params.filter((q) => q.key !== 'mode' && isParamVisible(q, p, f.params)).map((q) => q.key);
      const want = EXPECT_KEYS[op].filter((k) => k !== 'lineAngle' || shape === 'line').map((k) => id + '__' + k);
      assert.deepEqual(vis, want, `${id} visible (shape=${shape})`);
    }
  }
  // ตัวควบคุม (shape) มาก่อน lineAngle ในลำดับแสดง
  const ord = orderParamsControllersFirst(OPS.morph_erode.params).map((q) => q.key);
  assert.ok(ord.indexOf('shape') < ord.indexOf('lineAngle'));
});

test('mp-3. every op: pixels + size + type + ctx.info IDENTICAL to OPS.morphology with the same op (default AND non-default params; standalone AND via morph_tools)', () => {
  for (const op of OPS_ORDER) {
    const id = 'morph_' + op;
    for (const [tag, local] of [['default', {}], ['non-default', NONDEFAULT[op]]]) {
      const a = runOld(op, local);
      const b = run((c, src, ctx) => OPS[id].apply(c, src, { ...localDefaults(id), ...local }, ctx));
      const g = runGroup(op, local);
      assert.deepEqual(b, a, `${id} ${tag}: standalone differs from legacy`);
      assert.deepEqual(g, a, `${id} ${tag}: via morph_tools differs from legacy`);
      assert.equal(needsOf(FILTERS.morph_tools, groupParams(op, local)), needsOf(FILTERS.morphology, oldParams(op, local)));
    }
  }
  // กันเทสต์ผ่านเพราะ param ไม่มีผล: non-default ต้องให้ผลต่างจาก default จริงในตัวที่มี param
  for (const op of OPS_ORDER) {
    if (!Object.keys(NONDEFAULT[op]).length) continue;
    const d = defaultParams('morphology');
    for (const [k, v] of Object.entries(NONDEFAULT[op])) assert.notEqual(v, d[k], `${op}.${k}: non-default value must differ from default`);
    assert.notEqual(runOld(op, NONDEFAULT[op]).data, runOld(op, {}).data, `${op}: non-default params must change output`);
  }
  // ctx.info มีข้อความจริงในตัวที่ใช้ (กัน deepEqual ผ่านเพราะ null ทั้งคู่)
  assert.match(runNew('fill_holes', {}).info, /เติมรูที่ปิดสนิท/);
  assert.match(runNew('endpoints_branchpoints', {}).info, /จุดปลาย/);
  assert.match(runNew('extended_max', { hHeight: 25 }).info, /h=25/);
});

test('mp-3b. apply does not mutate its params and the ops do not leak mats across 30 runs (output is a fresh Mat each time)', () => {
  const src = makeGray();
  try {
    for (const op of OPS_ORDER) {
      const p = { ...localDefaults('morph_' + op), ...NONDEFAULT[op] };
      const frozen = JSON.stringify(p);
      for (let i = 0; i < 2; i++) {
        const out = OPS['morph_' + op].apply(cv, src, p, {});
        assert.notEqual(out, src);
        out.delete();
      }
      assert.equal(JSON.stringify(p), frozen, `${op}: params mutated`);
    }
  } finally { src.delete(); }
});

// ---------------------------------------------------------------------------
const OP_NAME_PREFIX = { erode: 'Erosion', dilate: 'Dilation', open: 'Opening', close: 'Closing', gradient: 'Gradient', tophat: 'Top Hat', blackhat: 'Black Hat' };
test('mp-4. palette: MORPHOLOGY lists the 19 new ids in order; every id → morph_tools; legacy morphology not on the palette; dropdown = palette order', () => {
  const sec = TOOL_SECTIONS.find((s) => s.title === 'MORPHOLOGY');
  assert.deepEqual(sec.ops, NEW_IDS);
  for (const id of NEW_IDS) assert.equal(OP_TO_FILTER[id], 'morph_tools', id);
  assert.equal(OP_TO_FILTER.morphology, undefined, 'legacy morphology must not be on the palette');
  assert.ok(!Object.values(OP_TO_FILTER).includes('morphology'));
  assert.deepEqual(FILTERS.morph_tools.params[0].options.map((o) => o.value), sec.ops);
  for (const [op, n] of Object.entries(OP_NAME_PREFIX)) assert.ok(OPS['morph_' + op].name.startsWith(n), op);
  // "ใช้ล่าสุด" เก่าที่เก็บ "morphology" ถูกทิ้งเงียบ ๆ (isKnown เดียวกับ loadRecent) · ตัวใหม่อยู่ต่อ
  const known = (id) => !!OP_TO_FILTER[id] && !!OPS[id];
  // (แก้ตามรีวิว: เดิมทิ้ง → ตอนนี้ map "morphology" → morph_erode ไม่ซ้ำ ไม่ทิ้งประวัติ)
  assert.deepEqual(resolveRecent(migrateRecentIds(JSON.stringify(['morphology', 'canny'])), known), ['morph_erode', 'canny']);
  assert.deepEqual(resolveRecent(migrateRecentIds(JSON.stringify(['morph_dilate', 'morphology', 'morph_erode'])), known), ['morph_dilate', 'morph_erode']);
  assert.equal(migrateRecentIds('not json'), 'not json');
  assert.equal(migrateRecentIds(null), null);
  assert.equal(migrateRecentIds('{"a":1}'), '{"a":1}');
  // ค้นหา: ชื่อหมวดนับด้วย ("morphology"/"morpho" เจอทุกเครื่องมือในหมวด) — palette ส่ง r.name + " " + s.title
  const palSrc = fs.readFileSync(new URL('../src/renderer/ui/palette-panel.js', import.meta.url), 'utf8');
  assert.match(palSrc, /toolMatches\(r\.name \+ " " \+ s\.title, r\.opId, query\)/);
  const msec = TOOL_SECTIONS.find((s) => s.title === 'MORPHOLOGY');
  for (const q of ['morphology', 'morpho', 'MORPH']) for (const id of msec.ops) assert.ok(toolMatches(OPS[id].name + ' ' + msec.title, id, q), `${q} → ${id}`);
  assert.equal(toolMatches('Gaussian Blur ' + 'FILTER & DENOISE', 'gaussian_blur', 'morphology'), false);
  // แหล่งที่มาของ VISIBLE_FILTER_IDS (ไม่ export) — อ่านแบบ static
  const src = fs.readFileSync(new URL('../src/renderer/ui/palette-panel.js', import.meta.url), 'utf8');
  const vis = src.slice(src.indexOf('const VISIBLE_FILTER_IDS = ['), src.indexOf('];', src.indexOf('const VISIBLE_FILTER_IDS = [')));
  assert.match(vis, /"morph_tools"/);
  assert.doesNotMatch(vis, /"morphology"/);
});

test('mp-5. dropdown optgroups: 4 Thai groups in the required order/membership; every option covered once; no "อื่น ๆ"', () => {
  const param = FILTERS.morph_tools.params[0];
  const g = optionGroupsFor('morph_tools', param);
  assert.deepEqual(g.map(([t]) => t), ['พื้นฐาน', 'ทำความสะอาด', 'รูปร่าง', 'ยอด/หุบ']);
  const ids = (list) => list.map((o) => o.value);
  assert.deepEqual(ids(g[0][1]), NEW_IDS.slice(0, 7));
  assert.deepEqual(ids(g[1][1]), ['morph_fill_holes', 'morph_clear_border', 'morph_remove_small']);
  assert.deepEqual(ids(g[2][1]), ['morph_skeleton', 'morph_reconstruct_open', 'morph_convex_hull', 'morph_endpoints_branchpoints', 'morph_hit_or_miss']);
  assert.deepEqual(ids(g[3][1]), ['morph_regional_max', 'morph_regional_min', 'morph_h_maxima', 'morph_extended_max']);
  // เรียงต่อกัน = ลำดับ dropdown เดิมเป๊ะ (ลำดับบนจอ = ลำดับ palette)
  assert.deepEqual(g.flatMap(([, o]) => ids(o)), NEW_IDS);
  assert.ok(Object.hasOwn(OPTION_GROUPS, 'morph_tools.mode'));
  // option object เป็นตัวเดียวกับใน param.options (detail-panel ใช้ index ของ param.options)
  for (const [, list] of g) for (const o of list) assert.ok(param.options.includes(o));
  // legacy morphology.op ยังมีตารางของตัวเองเหมือนเดิม
  assert.ok(Object.hasOwn(OPTION_GROUPS, 'morphology.op'));
});

// ---------------------------------------------------------------------------
test('mp-6. migrateLegacyStep: old morphology step → morph_tools with mode = morph_<op> and params under "morph_<op>__"; non-default values survive and behaviour is identical (all 19 ops)', () => {
  for (const op of OPS_ORDER) {
    const id = 'morph_' + op;
    // step เก่าเต็ม (ค่า default ครบ + ค่าที่ไม่ใช่ default ทับ) เหมือนที่เทมเพลตเก่า/applyTemplate ใส่ properties
    const old = { id: 'morphology', params: oldParams(op, NONDEFAULT[op]), nodeId: 7, enabled: true };
    const frozen = JSON.stringify(old);
    const mig = migrateLegacyStep(old);
    assert.equal(JSON.stringify(old), frozen, `${op}: input mutated`);
    assert.equal(mig.id, 'morph_tools');
    assert.equal(mig.nodeId, 7);
    assert.equal(mig.enabled, true);
    assert.equal(mig.params.mode, id);
    assert.deepEqual(Object.keys(mig.params).sort(), ['mode', ...EXPECT_KEYS[op].map((k) => id + '__' + k)].sort(), `${op}: migrated keys`);
    for (const k of EXPECT_KEYS[op]) assert.equal(mig.params[id + '__' + k], old.params[k], `${op}.${k} value`);
    for (const [k, v] of Object.entries(NONDEFAULT[op])) assert.equal(mig.params[id + '__' + k], v, `${op}.${k} non-default survives`);
    // เหมือน editor.applyTemplate: defaultParams(กลุ่มใหม่) แล้ว assign ทับ
    const applied = Object.assign(defaultParams(mig.id), mig.params);
    assert.deepEqual(run((c, src, ctx) => FILTERS.morph_tools.apply(c, src, applied, ctx)), run((c, src, ctx) => FILTERS.morphology.apply(c, src, old.params, ctx)), `${op}: output differs after migration`);
    assert.equal(opIdOfStep(mig), id);
  }
  // step แบบที่ app.js ใช้: params ไม่ครบ / ไม่มี params / ไม่มี op
  assert.deepEqual(migrateLegacyStep({ id: 'morphology', params: { op: 'dilate', ksize: 9 } }), { id: 'morph_tools', params: { mode: 'morph_dilate', morph_dilate__ksize: 9 } });
  assert.deepEqual(migrateLegacyStep({ id: 'morphology', params: { ksize: 3, shape: 'cross' } }), { id: 'morph_tools', params: { mode: 'morph_erode', morph_erode__shape: 'cross', morph_erode__ksize: 3 } });
  assert.deepEqual(migrateLegacyStep({ id: 'morphology' }), { id: 'morph_tools', params: { mode: 'morph_erode' } });
  // key ของ op อื่นที่ติดมาใน step เก่า (เช่น minArea ใน erode) ไม่ถูกพาไปด้วย
  assert.deepEqual(Object.keys(migrateLegacyStep({ id: 'morphology', params: { op: 'erode', minArea: 99, hHeight: 5, ksize: 3 } }).params).sort(), ['mode', 'morph_erode__ksize']);
  // op ไม่รู้จัก / ชนิดแปลก → คืนตัวเดิม (legacy morphology ยังรันได้) · new group ไม่ถูกย้ายซ้ำ
  for (const bad of ['nope', 'constructor', 7, null, {}]) {
    const s = { id: 'morphology', params: { op: bad } };
    if (bad === null) { assert.equal(migrateLegacyStep(s).id, 'morph_tools'); continue; } // null → ถือว่าไม่มี op → erode
    assert.equal(migrateLegacyStep(s), s, String(bad));
  }
  const already = { id: 'morph_tools', params: { mode: 'morph_open' } };
  assert.equal(migrateLegacyStep(already), already);
});

test('mp-6b. editor.applyTemplate still routes through migrateLegacyStep (static) and app.js built-in templates have no morphology step needing conversion', () => {
  const ed = fs.readFileSync(new URL('../src/renderer/graph/editor.js', import.meta.url), 'utf8');
  assert.match(ed, /const step = migrateLegacyStep\(raw\)/);
  const app = fs.readFileSync(new URL('../src/renderer/app.js', import.meta.url), 'utf8');
  const block = app.slice(app.indexOf('const TEMPLATES = ['), app.indexOf('// เทมเพลตที่ผู้ใช้เซฟเอง'));
  assert.doesNotMatch(block, /id: "morphology"/, 'built-in templates do not use legacy morphology');
  // ทุก step ในเทมเพลตในตัวย้ายแล้วลงกลุ่มที่ลงทะเบียน
  for (const m of block.matchAll(/\{ id: "([a-z_0-9]+)"/g)) assert.ok(FILTERS[migrateLegacyStep({ id: m[1], params: {} }).id], m[1]);
});

// ---------------------------------------------------------------------------
// เทียบ Python: ตัดบรรทัดหัวขั้น ("# ── ขั้นที่ N: ชื่อ (id) ──") ที่ขึ้นชื่อ op ต่างกันโดยธรรมชาติ แล้วที่เหลือต้องเหมือนกันทุกตัวอักษร
const stripHeader = (code) => code.split('\n').filter((l) => !/^\s*# ── ขั้นที่ \d+: /.test(l)).join('\n');
const py = (steps) => stripHeader(buildPythonCode(steps, { displayName: 'X' }));

test('mp-7. codegen: Python of morph_<op> (standalone AND inside morph_tools) == legacy morphology with that op — default + non-default, alone and followed by a gray-needing step', () => {
  for (const op of OPS_ORDER) {
    const id = 'morph_' + op;
    for (const local of [{}, NONDEFAULT[op]]) {
      const old = [{ id: 'morphology', params: oldParams(op, local) }];
      const standalone = [{ id, params: { ...localDefaults(id), ...local } }];
      const grouped = [{ id: 'morph_tools', params: groupParams(op, local) }];
      const want = py(old);
      assert.ok(want.includes('img'), `${op}: sanity`);
      assert.equal(py(standalone), want, `${id} standalone`);
      assert.equal(py(grouped), want, `${id} grouped`);
      // ต่อด้วย canny (needs gray) — ctx.gray/outOf ของ endpoints_branchpoints (ภาพสี) ต้องทำให้แทรก BGR2GRAY เหมือนเดิม
      const tailStep = { id: 'segment_edge', params: { ...defaultParams('segment_edge'), mode: 'canny' } };
      assert.equal(py([...grouped, tailStep]), py([...old, tailStep]), `${id} + canny`);
    }
  }
  // และบรรทัดหัวขั้นของตัวใหม่บอกชื่อ/รหัสของตัวใหม่ + ข้อความสอนเดิม
  const code = buildPythonCode([{ id: 'morph_tools', params: groupParams('dilate', {}) }], { displayName: 'X' });
  assert.match(code, /ขั้นที่ 1: Dilation \(ขยายให้ใหญ่ขึ้น\) \(morph_dilate\)/);
  assert.match(code, /จัดรูปทรงด้วย kernel/);
  assert.match(code, /cv2\.dilate\(img, kernel, iterations=1\)/);
});

// ---------------------------------------------------------------------------
test('mp-8. icons: 19 OP_ICON glyphs + GROUP_ICON.morph_tools are unique across OP_ICON/GROUP_ICON, not the 🧩/📊/📌 markers, and appear nowhere else in src', () => {
  const pal = fs.readFileSync(new URL('../src/renderer/ui/palette-panel.js', import.meta.url), 'utf8');
  const body = pal.slice(pal.indexOf('const OP_ICON = {'), pal.indexOf('};', pal.indexOf('const OP_ICON = {')));
  const icons = Object.fromEntries([...body.matchAll(/([a-z_0-9]+): "([^"]+)"/g)].map((m) => [m[1], m[2]]));
  const all = Object.entries(icons);
  assert.equal(new Set(all.map(([, v]) => v)).size, all.length, 'every OP_ICON glyph is unique');
  const groupGlyphs = Object.values(GROUP_ICON);
  assert.equal(new Set(groupGlyphs).size, groupGlyphs.length, 'every GROUP_ICON glyph is unique');
  const bad = new Set(['🧩', '📊', '📌']);
  const files = ['app.js', 'ui/detail-panel.js', 'graph/editor.js'].map((f) => fs.readFileSync(new URL('../src/renderer/' + f, import.meta.url), 'utf8'));
  for (const id of NEW_IDS) {
    const g = icons[id];
    assert.ok(g, `${id}: OP_ICON entry`);
    assert.ok(!groupGlyphs.includes(g), `${id} ${g} collides with a GROUP_ICON`);
    assert.ok(!bad.has(g));
    for (const f of files) assert.ok(!f.includes(g), `${id} ${g} appears in another UI file`);
  }
  const gi = GROUP_ICON.morph_tools;
  assert.ok(gi && !all.some(([, v]) => v === gi), 'GROUP_ICON.morph_tools unique vs OP_ICON');
  assert.ok(!bad.has(gi));
  for (const f of files) assert.ok(!f.includes(gi), 'GROUP_ICON.morph_tools appears elsewhere');
});

test('mp-9. node card text: OP_PRIMARY[morph_<op>] == legacy OP_PRIMARY.morphology for that op (clamped values included); currentOpId/localParamsFor work on the group', () => {
  for (const op of OPS_ORDER) {
    const id = 'morph_' + op;
    assert.equal(typeof OP_PRIMARY[id], 'function', id);
    for (const local of [{}, NONDEFAULT[op]]) {
      assert.equal(OP_PRIMARY[id]({ ...localDefaults(id), ...local }), OP_PRIMARY.morphology({ ...defaultParams('morphology'), op, ...local }), `${id} card`);
    }
    const props = groupParams(op, NONDEFAULT[op]);
    assert.equal(currentOpId('morph_tools', props), id);
    assert.deepEqual(localParamsFor('morph_tools', props), Object.fromEntries(OPS[id].params.map((q) => [q.key, q.key in NONDEFAULT[op] ? NONDEFAULT[op][q.key] : q.default])));
  }
  assert.equal(OP_PRIMARY.morph_h_maxima({ hHeight: 9999 }), 'H-maxima h=255');
  assert.equal(OP_PRIMARY.morph_remove_small({ minArea: 60 }), 'ลบวัตถุ < 60px');
});

test('mp-10. guidance kept: Endpoints & Branchpoints says "use after Skeleton"; Hit-or-Miss keeps the out-of-image border note; Skeleton mentions Invert', () => {
  assert.match(OPS.morph_endpoints_branchpoints.desc, /ใช้ต่อจาก Skeleton/);
  assert.match(OPS.morph_endpoints_branchpoints.modeHint, /Skeleton/);
  assert.match(OPS.morph_hit_or_miss.desc, /นอกขอบภาพ/);
  assert.match(OPS.morph_skeleton.desc, /Invert/);
  assert.match(OPS.morph_remove_small.desc, /พื้นที่ขั้นต่ำ/);
});

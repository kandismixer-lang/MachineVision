// P-8 รอบแรก (2026-10-17) — C-A (valuesToCsv + ปุ่ม "📋 คัดลอก" / "💾 CSV") + PF-A (op pass_fail 1 rule)
// WORKLOG: WORKLOG-2026-10-17-p8-passfail-csv.md
// โครงเดียวกับ tests/backlog-2026-10-17.test.mjs: cv จาก @techstark/opencv-js · Python จริงเมื่อมี python+cv2 (P8_NO_PY=1 จำลองเครื่องไม่มี Python)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import {
  OPS, FILTERS, defaultParams, isParamVisible, passFailVerdict, passFailParams, PASS_FAIL_METRICS, opIdOfStep,
} from '../src/renderer/filters/registry.js';
import { OP_TO_FILTER, TOOL_SECTIONS } from '../src/renderer/ui/palette-panel.js';
import { OP_PRIMARY, OPTION_GROUPS } from '../src/renderer/filters/ui-meta.js';
import { buildPythonCode, PF_VALUE_OPS } from '../src/renderer/codegen.js';
import { valuesToCsv } from '../src/renderer/ui/values-csv.js';
import { createDetailPanel } from '../src/renderer/ui/detail-panel.js';
import { assertLeakFree, assertWrapperBalance, makeThrowingCv } from './_leak-helpers.mjs';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cvtColor === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}
globalThis.self = globalThis;
if (typeof globalThis.ImageData === 'undefined') globalThis.ImageData = class { constructor(d, w, h) { this.data = d; this.width = w; this.height = h; } };
const { runPipeline } = await import('../src/renderer/cv/cv-worker.js');

// params ของ op pass_fail เปล่า ๆ (pass_fail ไม่ใช่ FILTERS เดี่ยว จึงใช้ defaultParams() ตรง ๆ ไม่ได้)
const PF_DEFAULTS = Object.fromEntries(OPS.pass_fail.params.map((q) => [q.key, q.default]));
const readSrc = (rel) => fs.readFileSync(new URL(`../src/renderer/${rel}`, import.meta.url), 'utf8');

// ---------- ภาพ/ตัวช่วย ----------
// 3 วงกลมขาวบนพื้นดำ (รัศมี 10/12/14) — ภาพมาตรฐานของ parity
function circlesRGB() {
  const m = new cv.Mat.zeros(90, 120, cv.CV_8UC3);
  for (const [x, y, r] of [[25, 25, 10], [70, 30, 12], [60, 65, 14]]) cv.circle(m, new cv.Point(x, y), r, new cv.Scalar(255, 255, 255), -1);
  return m;
}
function grayMat(w, h, fn) {
  const d = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d[y * w + x] = fn(x, y);
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
function rgbMat(w, h, fn) {
  const d = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d.set(fn(x, y), (y * w + x) * 3);
  return cv.matFromArray(h, w, cv.CV_8UC3, d);
}
const px = (m, x, y) => Array.from(m.data.slice((y * m.cols + x) * m.channels(), (y * m.cols + x) * m.channels() + m.channels()));

// step ของกล่องรวม: step('regionprops', { minArea: 10 }, 1) → { id: 'analysis_detect', params: { mode, regionprops__minArea }, nodeId }
const step = (opId, extra = {}, nodeId = 1, more = {}) => {
  const host = OP_TO_FILTER[opId];
  if (!host || host === opId) return { id: opId, params: { ...defaultParams(opId), ...extra }, enabled: true, nodeId, ...more };
  const prefixed = {};
  for (const [k, v] of Object.entries(extra)) prefixed[`${opId}__${k}`] = v;
  return { id: host, params: { ...defaultParams(host), mode: opId, ...prefixed }, enabled: true, nodeId, ...more };
};
const pf = (extra = {}, nodeId = 9, more = {}) => step('pass_fail', extra, nodeId, more);
const off = (s) => ({ ...s, enabled: false });

// รัน runPipeline จริง (worker) → { notes, mat } · ล้าง Mat ของ memo/ผลลัพธ์ให้เอง
function runApp(pipeline, base, memo = new Map()) {
  const store = new Map();
  const r = runPipeline(base, pipeline, store, new Set(pipeline.map((s) => s.nodeId)), memo);
  const out = { notes: r.notes, mat: r.mat };
  return out;
}
function release(r, memo) { r.mat.delete(); if (memo) for (const e of memo.values()) e.mat.delete(); }
const fieldsOf = (notes, nodeId) => {
  const n = notes.find((x) => x.nodeId === nodeId && x.values);
  return n ? Object.fromEntries(n.values.fields.map((f) => [f.key, f.value])) : null;
};
const infoOf = (notes, nodeId) => (notes.find((x) => x.nodeId === nodeId && x.info) || {}).info;

// ===================================================================
// C-A — valuesToCsv
// ===================================================================
test('p8-csv-1. table: UTF-8 BOM first, English keys as header, CRLF lines, one row per object, raw numbers (no rounding)', () => {
  const t = { kind: 'table', columns: ['index', 'area', 'cx'], rows: [[1, 314.5, 25.123456789], [2, 452, 70], [3, 615, 60.0000001]], total: 3 };
  const s = valuesToCsv(t);
  assert.equal(s.charCodeAt(0), 0xFEFF, 'BOM');
  assert.equal(s, '\uFEFFindex,area,cx\r\n1,314.5,25.123456789\r\n2,452,70\r\n3,615,60.0000001\r\n');
  assert.equal(Buffer.from(s, 'utf8').subarray(0, 3).toString('hex'), 'efbbbf', 'the encoded file starts with EF BB BF');
});

test('p8-csv-2. escaping: comma / quote / CR / LF in a cell are quoted and " doubled · Thai text kept as is · bool → true/false', () => {
  const t = { kind: 'table', columns: ['a', 'b,c', 'say "hi"'], rows: [['x,y', 'he said "ok"', 'line1\nline2'], ['ก ข', 'cr\rhere', true], [false, 'plain', 0]] };
  const s = valuesToCsv(t, { bom: false });
  assert.equal(s, 'a,"b,c","say ""hi"""\r\n"x,y","he said ""ok""","line1\nline2"\r\nก ข,"cr\rhere",true\r\nfalse,plain,0\r\n');
});

test('p8-csv-3. null/undefined/NaN → empty · Infinity → inf · -Infinity → -inf · 0 and negative numbers stay · short rows are padded', () => {
  const t = { kind: 'table', columns: ['a', 'b', 'c', 'd', 'e', 'f'], rows: [[null, undefined, NaN, Infinity, -Infinity, 0], [-1.5, 2], []] };
  assert.equal(valuesToCsv(t, { bom: false }), 'a,b,c,d,e,f\r\n,,,inf,-inf,0\r\n-1.5,2,,,,\r\n,,,,,\r\n');
});

test('p8-csv-4. scalars: header = keys of every field (not Thai labels), one data row · Thai string values and null', () => {
  const v = { kind: 'scalars', fields: [{ key: 'result', label: 'ผลตัดสิน', value: 'NG' }, { key: 'measured', label: 'ค่า', value: null }, { key: 'reason', label: 'เหตุผล', value: 'ไม่มีค่าจากกล่องก่อนหน้า, ตรวจสาย' }] };
  assert.equal(valuesToCsv(v), '\uFEFFresult,measured,reason\r\nNG,,"ไม่มีค่าจากกล่องก่อนหน้า, ตรวจสาย"\r\n');
});

test('p8-csv-5. empty table → header only · weird input (null / unknown kind / no fields) never throws', () => {
  assert.equal(valuesToCsv({ kind: 'table', columns: ['index', 'area'], rows: [], total: 0 }, { bom: false }), 'index,area\r\n');
  for (const bad of [null, undefined, 5, 'x', {}, { kind: 'nope' }]) assert.equal(valuesToCsv(bad), null);
  assert.equal(valuesToCsv({ kind: 'scalars' }, { bom: false }), '\r\n\r\n', 'scalars without fields: empty header + empty row (does not throw)');
  assert.equal(valuesToCsv({ kind: 'table' }, { bom: false }), '\r\n');
});

test('p8-csv-6. real regionprops values (worker) → CSV: header is REGION columns, row count = rows, every line has the same number of cells', () => {
  const base = circlesRGB();
  const r = runApp([step('regionprops', {}, 1)], base);
  try {
    const v = r.notes.find((n) => n.values).values;
    const lines = valuesToCsv(v, { bom: false }).trimEnd().split('\r\n');
    assert.equal(lines.length, v.rows.length + 1);
    assert.equal(lines[0], v.columns.join(','));
    for (const l of lines) assert.equal(l.split(',').length, v.columns.length);
    assert.equal(v.rows.length, 3);
  } finally { release(r); base.delete(); }
});

// ===================================================================
// PF-A — ตารางความจริงของ passFailVerdict (ล้วน ๆ ไม่มี cv)
// ===================================================================
const tbl = (rows, extraCols = [], total = rows.length) => ({ kind: 'table', columns: ['index', 'area_px', ...extraCols], rows, total });
const RP = (areas, mm) => ({
  op: 'regionprops',
  values: mm
    ? { kind: 'table', columns: ['index', 'area_px', 'area_mm2'], rows: areas.map((a, i) => [i + 1, a, a * mm * mm]), total: areas.length }
    : tbl(areas.map((a, i) => [i + 1, a]), [], areas.length),
});
const SC = (op, key, value) => ({ op, values: { kind: 'scalars', fields: [{ key, label: key, value }] } });
const V = (p, up) => passFailVerdict(p, up);

test('p8-pf-1. range [lo,hi] is inclusive on both edges; just outside = NG; measured/lo/hi are reported', () => {
  const up = SC('focus_score', 'focus_score', 10);
  const p = (lo, hi) => ({ metric: 'focus', cond: 'range', lo, hi });
  assert.deepEqual(V(p(10, 20), up), { result: 'OK', measured: 10, lo: 10, hi: 20, reason: null });
  assert.equal(V(p(0, 10), up).result, 'OK');
  assert.equal(V(p(10.0001, 20), up).result, 'NG');
  assert.equal(V(p(0, 9.9999), up).result, 'NG');
  assert.equal(V(p(10, 10), up).result, 'OK', 'lo = hi = measured');
  const bad = V(p(20, 10), up);
  assert.equal(bad.result, 'NG'); assert.match(bad.reason, /ต่ำสุดมากกว่า/);
});

test('p8-pf-2. ge / le / eq: edges inclusive · unused bound is null · eq tolerates 1e-9 relative', () => {
  const up = SC('focus_score', 'focus_score', 5);
  const ge = (lo) => V({ metric: 'focus', cond: 'ge', lo, hi: 1 }, up);
  assert.deepEqual(ge(5), { result: 'OK', measured: 5, lo: 5, hi: null, reason: null });
  assert.equal(ge(5.0000001).result, 'NG'); assert.equal(ge(4).result, 'OK');
  const le = (hi) => V({ metric: 'focus', cond: 'le', lo: 99, hi }, up);
  assert.deepEqual(le(5), { result: 'OK', measured: 5, lo: null, hi: 5, reason: null });
  assert.equal(le(4.9999999).result, 'NG'); assert.equal(le(6).result, 'OK');
  const eq = (lo, m = 5) => V({ metric: 'focus', cond: 'eq', lo, hi: 1 }, SC('focus_score', 'focus_score', m));
  assert.equal(eq(5).result, 'OK'); assert.equal(eq(5.1).result, 'NG'); assert.equal(eq(4.9).result, 'NG');
  assert.equal(eq(0.1 + 0.2, 0.3).result, 'OK', 'float noise tolerated');
  assert.deepEqual([eq(5).lo, eq(5).hi], [5, null]);
});

test('p8-pf-3. missing value → NG + reason (fail-safe): no upstream, null/undefined values, NaN, ±Infinity, non-number, missing field', () => {
  const p = { metric: 'focus', cond: 'ge', lo: 0, hi: 1 };
  for (const up of [null, undefined, {}, { op: 'focus_score' }, { op: 'focus_score', values: null }]) {
    const r = V(p, up); assert.equal(r.result, 'NG', JSON.stringify(up)); assert.match(r.reason, /ไม่มีค่าจากกล่องก่อนหน้า/); assert.equal(r.measured, null);
  }
  for (const val of [null, undefined, NaN, Infinity, -Infinity, '12', true, {}]) {
    const r = V(p, SC('focus_score', 'focus_score', val));
    assert.equal(r.result, 'NG', String(val)); assert.equal(r.measured, null); assert.match(r.reason, /ว่างหรือไม่ใช่ตัวเลข/);
  }
  const noField = V(p, { op: 'focus_score', values: { kind: 'scalars', fields: [{ key: 'other', value: 1 }] } });
  assert.equal(noField.result, 'NG'); assert.ok(noField.reason);
  const noFields = V(p, { op: 'focus_score', values: { kind: 'scalars' } });
  assert.equal(noFields.result, 'NG');
  // ค่าที่ตัดสิน NG เพราะไม่ผ่านเกณฑ์ (ไม่ใช่ไม่มีค่า) → reason = null
  assert.equal(V(p, SC('focus_score', 'focus_score', -1)).reason, null);
});

test('p8-pf-4. values from another box (wrong op) → NG + reason, even when a field with the same key exists', () => {
  const r = V({ metric: 'roi_mean', cond: 'ge', lo: 0 }, SC('focus_score', 'roi_mean', 100));
  assert.equal(r.result, 'NG'); assert.match(r.reason, /กล่องอื่น/); assert.match(r.reason, /ROI Polygon/);
  assert.equal(V({ metric: 'count', cond: 'ge', lo: 0 }, SC('focus_score', 'focus_score', 3)).result, 'NG');
});

test('p8-pf-5. regionprops metrics: count uses `total` (uncapped), area_sum / area_max use the rows (capped by maxObjects) · count 0 is a real 0', () => {
  const up = { op: 'regionprops', values: { ...tbl([[1, 100], [2, 50]]), total: 7 } }; // cap: 2 shown of 7 found
  assert.equal(V({ metric: 'count', cond: 'eq', lo: 7 }, up).result, 'OK');
  assert.equal(V({ metric: 'count', cond: 'eq', lo: 7 }, up).measured, 7);
  assert.equal(V({ metric: 'area_sum', cond: 'eq', lo: 150 }, up).measured, 150);
  assert.equal(V({ metric: 'area_max', cond: 'eq', lo: 100 }, up).measured, 100);
  // total ไม่มี → ใช้จำนวนแถว
  assert.equal(V({ metric: 'count', cond: 'ge', lo: 0 }, { op: 'regionprops', values: { kind: 'table', columns: ['index', 'area_px'], rows: [[1, 5]] } }).measured, 1);
  const none = RP([]);
  const c0 = V({ metric: 'count', cond: 'eq', lo: 0 }, none);
  assert.deepEqual([c0.result, c0.measured, c0.reason], ['OK', 0, null], 'zero objects is a valid count of 0');
  assert.equal(V({ metric: 'count', cond: 'range', lo: 1, hi: 100 }, none).result, 'NG');
});

test('p8-pf-6. EMPTY table: area metrics → NG with "ตารางว่าง" reason (not 0) · also columns missing / ragged rows / non-number cell', () => {
  const none = RP([]);
  for (const metric of ['area_sum', 'area_max']) {
    const r = V({ metric, cond: 'ge', lo: 0 }, none);
    assert.equal(r.result, 'NG', metric); assert.match(r.reason, /ตารางว่าง/); assert.equal(r.measured, null);
  }
  assert.equal(V({ metric: 'area_sum', cond: 'ge', lo: 0 }, { op: 'regionprops', values: { kind: 'table', columns: ['index', 'perimeter'], rows: [[1, 5]] } }).result, 'NG');
  assert.equal(V({ metric: 'area_sum', cond: 'ge', lo: 0 }, { op: 'regionprops', values: { kind: 'table', columns: ['index', 'area_px'], rows: [[1, 5], [2, null]] } }).result, 'NG', 'a null cell is NOT silently summed as 0');
  assert.equal(V({ metric: 'area_max', cond: 'ge', lo: 0 }, { op: 'regionprops', values: { kind: 'table', columns: ['index', 'area_px'], rows: [[1, 5], 'x'] } }).result, 'NG');
  assert.equal(V({ metric: 'area_sum', cond: 'ge', lo: 0 }, { op: 'regionprops', values: { kind: 'scalars', fields: [] } }).result, 'NG', 'wrong kind');
});

test('p8-pf-7. mm: with a mm/px scale on regionprops the area metrics use area_mm2 (and px otherwise)', () => {
  const px = RP([100, 400]), mm = RP([100, 400], 0.5);
  assert.equal(V({ metric: 'area_sum', cond: 'ge', lo: 0 }, px).measured, 500);
  assert.equal(V({ metric: 'area_sum', cond: 'ge', lo: 0 }, mm).measured, 125);
  assert.equal(V({ metric: 'area_max', cond: 'ge', lo: 0 }, mm).measured, 100);
  assert.equal(V({ metric: 'area_max', cond: 'range', lo: 99, hi: 101 }, mm).result, 'OK');
});

test('p8-pf-8. every metric reads its own field: roi_mean / focus / de_mean (scalars) and the six metric ids exist', () => {
  assert.deepEqual(Object.keys(PASS_FAIL_METRICS), ['count', 'area_sum', 'area_max', 'roi_mean', 'focus', 'de_mean']);
  assert.equal(V({ metric: 'roi_mean', cond: 'range', lo: 100, hi: 120 }, SC('roi_polygon', 'roi_mean', 110.5)).result, 'OK');
  assert.equal(V({ metric: 'de_mean', cond: 'le', hi: 2.3 }, SC('delta_e', 'de_mean', 2.3)).result, 'OK');
  assert.equal(V({ metric: 'de_mean', cond: 'le', hi: 2.3 }, SC('delta_e', 'de_mean', 2.31)).result, 'NG');
});

test('p8-pf-9. params are sanitised: junk metric/cond → defaults (own-property) · NaN/strings/out-of-range numbers clamp · numeric strings accepted', () => {
  for (const junk of ['constructor', '__proto__', 'toString', undefined, null, 5, '']) {
    const q = passFailParams({ metric: junk, cond: junk });
    assert.deepEqual([q.metric, q.cond], ['count', 'range'], String(junk));
  }
  assert.deepEqual(passFailParams({ lo: 'abc', hi: NaN }), { metric: 'count', cond: 'range', lo: 1, hi: 100 });
  assert.deepEqual(passFailParams({ lo: -5, hi: 1e12 }), { metric: 'count', cond: 'range', lo: 0, hi: 10000000 });
  assert.deepEqual(passFailParams({ lo: '2.5', hi: ' 7 ' }), { metric: 'count', cond: 'range', lo: 2.5, hi: 7 });
  assert.deepEqual(passFailParams(undefined), { metric: 'count', cond: 'range', lo: 1, hi: 100 });
  // default params of the box work out of the box (3 circles: count 3 in [1,100])
  assert.deepEqual(PF_DEFAULTS, { metric: 'count', cond: 'range', lo: 1, hi: 100 });
  assert.equal(V(PF_DEFAULTS, RP([1, 2, 3])).result, 'OK');
});

// ===================================================================
// registry / ui-meta / palette
// ===================================================================
test('p8-reg-1. pass_fail is a member of ANALYSIS & DETECTION (last), host analysis_detect, select/number params only, no "mode" key', () => {
  assert.ok(OPS.pass_fail);
  assert.equal(OPS.pass_fail.needs, 'any');
  assert.equal(OP_TO_FILTER.pass_fail, 'analysis_detect');
  const sec = TOOL_SECTIONS.find((s) => s.title === 'ANALYSIS & DETECTION');
  assert.equal(sec.ops.at(-1), 'pass_fail');
  assert.deepEqual(FILTERS.analysis_detect.params[0].options.map((o) => o.value), sec.ops);
  assert.deepEqual(OPS.pass_fail.params.map((q) => q.key), ['metric', 'cond', 'lo', 'hi']);
  for (const q of OPS.pass_fail.params) assert.ok(['select', 'number'].includes(q.type), q.key);
  assert.equal(Object.keys(FILTERS).length, 34, 'no new FILTERS entry from P-8 (34 = 33 + morph_tools from 2026-10-18)');
  assert.deepEqual(OPS.pass_fail.params[0].options.map((o) => o.value), Object.keys(PASS_FAIL_METRICS), 'metric options = metrics table');
  assert.match(OPS.pass_fail.desc, /maxObjects|แสดงวัตถุสูงสุด/, 'desc warns that regionprops is capped');
  assert.match(OPS.pass_fail.desc, /NG/);
  for (const q of OPS.pass_fail.params) assert.match(q.label, /[ก-๙]/, `${q.key} has a Thai label`);
});

test('p8-reg-2. visibleWhen: lo/hi shown only when the chosen condition uses them (range: both · ge/eq: lo · le: hi) — merged prefix and standalone', () => {
  const f = FILTERS.analysis_detect;
  const vis = (cond) => {
    const params = { ...defaultParams('analysis_detect'), mode: 'pass_fail', pass_fail__cond: cond };
    return f.params.filter((q) => q.key.startsWith('pass_fail__') && isParamVisible(q, params, f.params)).map((q) => q.key.slice(11));
  };
  assert.deepEqual(vis('range'), ['metric', 'cond', 'lo', 'hi']);
  assert.deepEqual(vis('ge'), ['metric', 'cond', 'lo']);
  assert.deepEqual(vis('eq'), ['metric', 'cond', 'lo']);
  assert.deepEqual(vis('le'), ['metric', 'cond', 'hi']);
  assert.deepEqual(vis('junk'), ['metric', 'cond', 'lo', 'hi'], 'junk controller → default (range) like apply()');
  const loP = f.params.find((q) => q.key === 'pass_fail__lo');
  assert.deepEqual(loP.visibleWhen, [{ key: 'pass_fail__cond', notIn: ['le'] }]);
});

test('p8-reg-3. ui-meta: optgroup table lists pass_fail (not under "อื่น ๆ") · card summary text · icon exists', () => {
  const all = OPTION_GROUPS['analysis_detect.mode'].flatMap(([, ids]) => ids);
  assert.ok(all.includes('pass_fail'));
  assert.equal(OP_PRIMARY.pass_fail(PF_DEFAULTS), 'จำนวนวัตถุ ผ่านเมื่อ 1–100');
  assert.equal(OP_PRIMARY.pass_fail({ metric: 'focus', cond: 'ge', lo: 80 }), 'ความคม ผ่านเมื่อ ≥ 80');
  assert.equal(OP_PRIMARY.pass_fail({ metric: 'de_mean', cond: 'le', hi: 2.3 }), 'ΔE เฉลี่ย ผ่านเมื่อ ≤ 2.3');
  assert.equal(OP_PRIMARY.pass_fail({ metric: 'constructor', cond: 'eq', lo: 3 }), 'จำนวนวัตถุ ผ่านเมื่อ = 3');
  assert.match(readSrc('ui/palette-panel.js'), /pass_fail: "✔"/);
});

test('p8-reg-4. opIdOfStep: merged node → mode (or the default when absent) · single node → id', () => {
  assert.equal(opIdOfStep({ id: 'analysis_detect', params: { mode: 'regionprops' } }), 'regionprops');
  assert.equal(opIdOfStep({ id: 'analysis_detect', params: {} }), 'find_contours');
  assert.equal(opIdOfStep({ id: 'analysis_detect' }), 'find_contours');
  assert.equal(opIdOfStep({ id: 'morphology', params: { op: 'erode' } }), 'morphology');
  assert.equal(opIdOfStep({ id: 'geometry', params: { mode: 'roi_polygon' } }), 'roi_polygon');
  assert.equal(opIdOfStep({ id: 'nope', params: { mode: 'x' } }), 'nope');
});

// ===================================================================
// op pass_fail บน cv (ภาพ + ctx.values)
// ===================================================================
const UP_OK = { op: 'regionprops', values: { kind: 'table', columns: ['index', 'area_px'], rows: [[1, 100]], total: 1 } };

test('p8-op-1. image: input + green border (OK) / red border (NG) + RGB output for gray / RGB / RGBA inputs · size unchanged · centre pixel untouched', () => {
  const g = grayMat(60, 40, () => 77), c = rgbMat(60, 40, () => [10, 20, 30]);
  const a = new cv.Mat(); cv.cvtColor(c, a, cv.COLOR_RGB2RGBA);
  try {
    for (const [label, src] of [['gray', g], ['rgb', c], ['rgba', a]]) {
      for (const [verdictOk, want] of [[true, [0, 200, 0]], [false, [255, 0, 0]]]) {
        const ctx = { upstream: UP_OK };
        const o = OPS.pass_fail.apply(cv, src, { metric: 'count', cond: 'ge', lo: verdictOk ? 1 : 5 }, ctx);
        try {
          assert.equal(o.channels(), 3, label); assert.equal(o.rows, 40); assert.equal(o.cols, 60);
          assert.deepEqual(px(o, 1, 20), want, `${label} ${verdictOk ? 'OK' : 'NG'} border`);
          assert.deepEqual(px(o, 30, 35), label === 'gray' ? [77, 77, 77] : [10, 20, 30], `${label} interior untouched`);
          assert.equal(ctx.values.fields[0].value, verdictOk ? 'OK' : 'NG');
        } finally { o.delete(); }
      }
    }
  } finally { g.delete(); c.delete(); a.delete(); }
});

test('p8-op-2. ctx.values = scalars [result, measured, lo, hi, reason] with a STABLE key list (also for NG-no-value) · info says OK/NG and why', () => {
  const g = grayMat(30, 20, () => 0);
  try {
    const run = (p, up) => { const ctx = { upstream: up }; OPS.pass_fail.apply(cv, g, p, ctx).delete(); return ctx; };
    const ok = run({ metric: 'count', cond: 'range', lo: 1, hi: 5 }, UP_OK);
    const no = run({ metric: 'count', cond: 'range', lo: 1, hi: 5 }, null);
    for (const c of [ok, no]) { assert.equal(c.values.kind, 'scalars'); assert.deepEqual(c.values.fields.map((f) => f.key), ['result', 'measured', 'lo', 'hi', 'reason']); }
    assert.deepEqual(ok.values.fields.map((f) => f.value), ['OK', 1, 1, 5, null]);
    assert.deepEqual(no.values.fields.map((f) => f.value), ['NG', null, 1, 5, 'ไม่มีค่าจากกล่องก่อนหน้า (ต้องต่อหลังกล่องที่วัดค่า)']);
    assert.match(ok.info, /^OK: .*1\.00.*1–5/); assert.match(no.info, /^NG: ไม่มีค่าจากกล่องก่อนหน้า/);
    // ctx ไม่มี upstream key เลย (เรียกตรงจากเทสต์เก่า ๆ) และ ctx = undefined ก็ไม่ crash
    const bare = {}; OPS.pass_fail.apply(cv, g, {}, bare).delete(); assert.equal(bare.values.fields[0].value, 'NG');
    OPS.pass_fail.apply(cv, g, {}, undefined).delete();
  } finally { g.delete(); }
});

test('p8-op-3. empty (0×0) input does not throw and does not draw · junk params fall back to defaults', () => {
  const e = new cv.Mat(); const ctx = { upstream: UP_OK };
  try {
    const o = OPS.pass_fail.apply(cv, e, { metric: 'constructor', cond: '__proto__', lo: 'x', hi: null }, ctx);
    try { assert.equal(o.rows, 0); } finally { o.delete(); }
    assert.equal(ctx.values.fields[0].value, 'OK');
  } finally { e.delete(); }
});

test('p8-op-4. tiny images (1x1, 2x3, 5x200) do not throw and keep their size', () => {
  for (const [w, h] of [[1, 1], [2, 3], [5, 200], [200, 5]]) {
    const m = grayMat(w, h, () => 9);
    try { const o = OPS.pass_fail.apply(cv, m, {}, { upstream: UP_OK }); try { assert.deepEqual([o.cols, o.rows, o.channels()], [w, h, 3]); } finally { o.delete(); } } finally { m.delete(); }
  }
});

test('p8-op-5. leak gates: pass_fail (gray / rgb / rgba / empty, OK and NG) balances every embind wrapper and the heap is flat · injected throw in rectangle/putText still frees the output', () => {
  const g = grayMat(40, 30, (x) => x), c = rgbMat(40, 30, () => [1, 2, 3]);
  const a = new cv.Mat(); cv.cvtColor(c, a, cv.COLOR_RGB2RGBA);
  const e = new cv.Mat();
  try {
    for (const [label, m] of [['gray', g], ['rgb', c], ['rgba', a], ['empty', e]]) {
      for (const lo of [1, 9]) {
        assertLeakFree(cv, `pass_fail ${label} lo=${lo}`, (cvU) => { OPS.pass_fail.apply(cvU, m, { lo }, { upstream: UP_OK }).delete(); }, { minCreated: 0, iters: 8, warmup: 3 });
      }
    }
    for (const method of ['rectangle', 'putText']) {
      for (const m of [g, c, a]) {
        assertWrapperBalance(cv, `throw in ${method}`, (cvU) => { OPS.pass_fail.apply(makeThrowingCv(cvU, method), m, {}, { upstream: UP_OK }).delete(); }, { expectThrow: /injected test failure/, minCreated: 1 });
      }
    }
    assertWrapperBalance(cv, 'throw in cvtColor (gray)', (cvU) => { OPS.pass_fail.apply(makeThrowingCv(cvU, 'cvtColor'), g, {}, { upstream: UP_OK }).delete(); }, { expectThrow: /injected test failure/, minCreated: 0 });
  } finally { for (const m of [g, c, a, e]) m.delete(); }
});

// ===================================================================
// worker path (runPipeline จริง)
// ===================================================================
test('p8-wk-1. regionprops → pass_fail: upstream carries the real table (3 circles) · OK/NG by count · box ids on the notes · measured = total', () => {
  const base = circlesRGB();
  try {
    for (const [p, want, measured] of [
      [{ metric: 'count', cond: 'eq', lo: 3 }, 'OK', 3],
      [{ metric: 'count', cond: 'ge', lo: 4 }, 'NG', 3],
      [{ metric: 'count', cond: 'range', lo: 1, hi: 3 }, 'OK', 3],
      [{ metric: 'count', cond: 'le', hi: 2 }, 'NG', 3],
    ]) {
      const r = runApp([step('regionprops', {}, 1), pf(p, 9)], base);
      try {
        const f = fieldsOf(r.notes, 9);
        assert.equal(f.result, want, JSON.stringify(p)); assert.equal(f.measured, measured);
        assert.ok(r.notes.find((n) => n.nodeId === 9 && n.values).view, 'pass_fail note carries its own view tag');
        assert.equal(r.mat.channels(), 3); assert.equal(r.mat.rows, 90);
        // pixel on the border column: green for OK, red for NG
        assert.deepEqual(px(r.mat, 1, 45), want === 'OK' ? [0, 200, 0] : [255, 0, 0]);
      } finally { release(r); }
    }
    // area metrics = same numbers as the regionprops table the app shows
    const r = runApp([step('regionprops', {}, 1), pf({ metric: 'area_sum', cond: 'ge', lo: 0 }, 9), pf({ metric: 'area_max', cond: 'ge', lo: 0 }, 10)], base);
    try {
      const rp = r.notes.find((n) => n.nodeId === 1 && n.values).values;
      const ci = rp.columns.indexOf('area_px');
      const areas = rp.rows.map((row) => row[ci]);
      assert.equal(fieldsOf(r.notes, 9).measured, areas.reduce((s, x) => s + x, 0));
      assert.equal(fieldsOf(r.notes, 10).measured, Math.max(...areas), 'two pass_fail boxes in a row both read the regionprops box (pass_fail does not become upstream)');
    } finally { release(r); }
  } finally { base.delete(); }
});

test('p8-wk-2. no measuring box before it → NG with reason (also as the first box) · info explains', () => {
  const base = circlesRGB();
  const r = runApp([pf({}, 9)], base);
  try {
    const f = fieldsOf(r.notes, 9);
    assert.equal(f.result, 'NG'); assert.match(f.reason, /ไม่มีค่าจากกล่องก่อนหน้า/); assert.equal(f.measured, null);
    assert.match(infoOf(r.notes, 9), /^NG: ไม่มีค่าจากกล่องก่อนหน้า/);
    assert.deepEqual(px(r.mat, 1, 45), [255, 0, 0]);
  } finally { release(r); base.delete(); }
  // a box without values (gaussian blur) before it does not create upstream
  const b2 = circlesRGB();
  const r2 = runApp([step('gaussian_blur', {}, 1), pf({}, 9)], b2);
  try { assert.equal(fieldsOf(r2.notes, 9).result, 'NG'); } finally { release(r2); b2.delete(); }
});

test('p8-wk-3. wrong measuring box in between: the LATEST box with values wins (regionprops → focus_score → pass_fail count = NG "กล่องอื่น")', () => {
  const base = circlesRGB();
  const r = runApp([step('regionprops', {}, 1), step('focus_score', {}, 2), pf({ metric: 'count', cond: 'ge', lo: 0 }, 9)], base);
  try {
    const f = fieldsOf(r.notes, 9);
    assert.equal(f.result, 'NG'); assert.match(f.reason, /กล่องอื่น/);
    // …and the focus metric now works on the same pipeline
  } finally { release(r); }
  const r2 = runApp([step('regionprops', {}, 1), step('focus_score', {}, 2), pf({ metric: 'focus', cond: 'ge', lo: 0 }, 9)], base);
  try {
    const f = fieldsOf(r2.notes, 9), fs0 = fieldsOf(r2.notes, 2);
    assert.equal(f.result, 'OK'); assert.equal(f.measured, fs0.focus_score);
  } finally { release(r2); base.delete(); }
});

test('p8-wk-4. bypass: a disabled box is skipped — the value from the box BEFORE it is used (regionprops → [focus_score off] → pass_fail count OK) · a disabled pass_fail produces nothing', () => {
  const base = circlesRGB();
  const r = runApp([step('regionprops', {}, 1), off(step('focus_score', {}, 2)), pf({ metric: 'count', cond: 'eq', lo: 3 }, 9)], base);
  try { assert.equal(fieldsOf(r.notes, 9).result, 'OK'); assert.equal(fieldsOf(r.notes, 2), null); } finally { release(r); }
  const r2 = runApp([off(step('regionprops', {}, 1)), pf({}, 9)], base);
  try { assert.equal(fieldsOf(r2.notes, 9).result, 'NG', 'bypassed measuring box gives no values'); } finally { release(r2); }
  const r3 = runApp([step('regionprops', {}, 1), off(pf({}, 9))], base);
  try { assert.equal(fieldsOf(r3.notes, 9), null); assert.equal(r3.mat.channels(), 3); } finally { release(r3); base.delete(); }
});

test('p8-wk-5. line B values do NOT cross: (a) main line has regionprops, pass_fail on line B → NG no-upstream · (b) regionprops only on line B, pass_fail on main → NG', () => {
  const base = circlesRGB();
  // (a) psnr node with src2 = [pass_fail] ; main line: regionprops → psnr
  const lineB = [pf({}, 20)];
  const a = runApp([step('regionprops', {}, 1), step('psnr', {}, 2, { src2: { pipeline: lineB } })], base);
  try {
    const nb = a.notes.find((n) => String(n.id).includes('(B)') && n.values);
    assert.ok(nb, 'pass_fail of line B reported');
    const f = Object.fromEntries(nb.values.fields.map((x) => [x.key, x.value]));
    assert.equal(f.result, 'NG'); assert.match(f.reason, /ไม่มีค่าจากกล่องก่อนหน้า/);
  } finally { release(a); }
  // (b) regionprops only on line B; pass_fail on the main line
  const b = runApp([step('psnr', {}, 2, { src2: { pipeline: [step('regionprops', {}, 21)] } }), pf({}, 9)], base);
  try {
    const f = fieldsOf(b.notes, 9);
    assert.equal(f.result, 'NG', 'psnr values are the latest on the main line, regionprops of line B never crosses'); assert.ok(f.reason);
  } finally { release(b); }
  // (c) B line has its own measuring box then pass_fail: works inside the line
  const c = runApp([step('psnr', {}, 2, { src2: { pipeline: [step('regionprops', {}, 21), pf({ metric: 'count', cond: 'eq', lo: 3 }, 22)] } })], base);
  try {
    const nb = c.notes.find((n) => String(n.id).includes('(B)') && n.nodeId === 22 && n.values);
    assert.equal(nb.values.fields[0].value, 'OK');
  } finally { release(c); base.delete(); }
});

test('p8-wk-6. memo hit: a stateful box served from the frame memo still becomes upstream (reason says "กล่องอื่น", not "ไม่มีค่า") and gives the same verdict both times · no Mat leaks from the memo path', () => {
  const frame = rgbMat(48, 40, (x, y) => [(x * 5) & 255, (y * 7) & 255, 90]);
  const memo = new Map();
  const pipeline = [step('mog2', {}, 3), pf({ metric: 'focus', cond: 'ge', lo: 0 }, 9)];
  const r1 = runApp(pipeline, frame, memo);
  const r2 = runApp(pipeline, frame, memo); // memo hit for mog2
  try {
    const a = fieldsOf(r1.notes, 9), b = fieldsOf(r2.notes, 9);
    assert.equal(a.result, 'NG'); assert.match(a.reason, /กล่องอื่น/);
    assert.deepEqual(b, a);
    assert.ok(r2.notes.some((n) => n.nodeId === 3 && n.values), 'memo hit still reports the values');
  } finally { r1.mat.delete(); r2.mat.delete(); for (const e of memo.values()) e.mat.delete(); frame.delete(); }
});

test('p8-wk-7. roi_polygon (roi_mean) / focus_score (focus) / delta_e (de_mean, with line B) feed pass_fail through the real worker', () => {
  const base = circlesRGB();
  try {
    const roi = runApp([step('roi_polygon', {}, 1), pf({ metric: 'roi_mean', cond: 'ge', lo: 0 }, 9)], base);
    try { assert.equal(fieldsOf(roi.notes, 9).measured, fieldsOf(roi.notes, 1).roi_mean); assert.equal(fieldsOf(roi.notes, 9).result, 'OK'); } finally { release(roi); }
    const fo = runApp([step('focus_score', {}, 1), pf({ metric: 'focus', cond: 'ge', lo: 0 }, 9)], base);
    try { assert.equal(fieldsOf(fo.notes, 9).measured, fieldsOf(fo.notes, 1).focus_score); } finally { release(fo); }
    // delta_e of the image with itself (B = same image) → mean 0 → ΔE ≤ 2.3 passes
    const de = runApp([step('delta_e', {}, 1, { src2: { pipeline: [] } }), pf({ metric: 'de_mean', cond: 'le', hi: 2.3 }, 9)], base);
    try { const f = fieldsOf(de.notes, 9); assert.equal(f.measured, 0); assert.equal(f.result, 'OK'); } finally { release(de); }
    // delta_e without line B is a passthrough with null values → NG (fail-safe), not a crash
    const de2 = runApp([step('delta_e', {}, 1), pf({ metric: 'de_mean', cond: 'le', hi: 2.3 }, 9)], base);
    try { assert.equal(fieldsOf(de2.notes, 9).result, 'NG'); } finally { release(de2); }
  } finally { base.delete(); }
});

test('p8-wk-8. regionprops cap: maxObjects=1 → count still 3 (total), area_sum only the single biggest shown (documented in desc)', () => {
  const base = circlesRGB();
  const r = runApp([step('regionprops', { maxObjects: 1 }, 1), pf({ metric: 'count', cond: 'eq', lo: 3 }, 9), pf({ metric: 'area_sum', cond: 'ge', lo: 0 }, 10)], base);
  try {
    assert.equal(fieldsOf(r.notes, 9).result, 'OK');
    const rp = r.notes.find((n) => n.nodeId === 1 && n.values).values;
    assert.equal(rp.rows.length, 1);
    assert.equal(fieldsOf(r.notes, 10).measured, rp.rows[0][rp.columns.indexOf('area_px')]);
  } finally { release(r); base.delete(); }
});

test('p8-wk-9. mm: regionprops with mmPerPx > 0 → pass_fail area metrics are in mm² (area_mm2 column)', () => {
  const base = circlesRGB();
  const r0 = runApp([step('regionprops', {}, 1), pf({ metric: 'area_max', cond: 'ge', lo: 0 }, 9)], base);
  const r1 = runApp([step('regionprops', { mmPerPx: 0.5 }, 1), pf({ metric: 'area_max', cond: 'ge', lo: 0 }, 9)], base);
  try {
    const px0 = fieldsOf(r0.notes, 9).measured, mm1 = fieldsOf(r1.notes, 9).measured;
    assert.equal(mm1, px0 * (0.5 * 0.5));
  } finally { release(r0); release(r1); base.delete(); }
});

test('p8-wk-10. worker source: ctx.upstream is wired with a plain `upstream` variable local to runPipeline (so line B, a recursive call, starts empty)', () => {
  const w = readSrc('cv/cv-worker.js');
  assert.match(w, /let upstream = null;/);
  assert.match(w, /const ctx = \{ upstream \};/);
  assert.match(w, /if \(opId !== "pass_fail"\) upstream = \{ op: opId, values: ctx\.values \};/);
  assert.match(w, /opIdOfStep/);
});

// ===================================================================
// DOM stub — ปุ่ม "📋 คัดลอก" / "💾 CSV"
// ===================================================================
class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.children = []; this.parentNode = null; this._text = ''; this._cls = '';
    this.style = { setProperty() {} }; this._attrs = {}; this.hidden = false; this.disabled = false; this.selected = false;
    this.title = ''; this.value = ''; this.label = ''; this._listeners = {};
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
  removeAttribute(k) { delete this._attrs[k]; }
  addEventListener(t, fn) { (this._listeners[t] ||= []).push(fn); }
  async click() { for (const fn of this._listeners.click || []) await fn({}); }
  focus() {} blur() {} select() {} remove() {}
  all(cls, out = []) { for (const c of this.children) { if (c._cls.split(' ').includes(cls)) out.push(c); c.all(cls, out); } return out; }
  one(cls) { return this.all(cls)[0] || null; }
}
// stub ขอบเขตแคบ: document / setTimeout / clearTimeout / navigator ติดตั้งในเทสต์เดียว คืนของจริงเสมอ
async function withDom(fn) {
  const had = Object.prototype.hasOwnProperty.call(globalThis, 'document');
  const prev = { doc: globalThis.document, st: globalThis.setTimeout, ct: globalThis.clearTimeout, now: Date.now, nav: Object.getOwnPropertyDescriptor(globalThis, 'navigator') };
  const clip = { writes: [], fail: false };
  const timers = [];
  globalThis.document = { createElement: (t) => new El(t), body: new El('body'), execCommand: () => true };
  globalThis.setTimeout = (fn2, ms) => { timers.push({ fn: fn2, ms }); return timers.length; };
  globalThis.clearTimeout = () => {};
  let t = 5_000_000; Date.now = () => (t += 1000);
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { clipboard: { writeText: async (s) => { if (clip.fail) throw new Error('denied'); clip.writes.push(s); } } } });
  try { return await fn({ clip, timers }); } finally {
    Date.now = prev.now; globalThis.setTimeout = prev.st; globalThis.clearTimeout = prev.ct;
    if (had) globalThis.document = prev.doc; else delete globalThis.document;
    if (prev.nav) Object.defineProperty(globalThis, 'navigator', prev.nav); else delete globalThis.navigator;
  }
}
function makePanel(opId, params, cbExtra = {}) {
  const container = new El('div');
  const stepObj = { id: 'analysis_detect', params: { ...defaultParams('analysis_detect'), mode: opId, ...params }, enabled: true, nodeId: 5 };
  const view = FILTERS.analysis_detect.subtitle(stepObj.params);
  const downloads = [];
  const panel = createDetailPanel(container, { onParamChange() {}, onDownloadCsv: (text, name) => downloads.push([text, name]), ...cbExtra });
  const notes = (values) => [{ id: 'analysis_detect', nodeId: 5, view, values }];
  return { container, panel, step: stepObj, notes, downloads };
}
const TBL = { kind: 'table', columns: ['index', 'area', 'note'], rows: [[1, 314, 'a,b'], [2, 452.5, null]], total: 2 };
const SCL = { kind: 'scalars', fields: [{ key: 'focus_score', label: 'คะแนน', value: 12.5 }, { key: 'focus_mean', label: 'เฉลี่ย', value: Infinity }] };

test('p8-ui-1. both buttons sit in .dv-head right after "📌 ตรึงค่า" (table and scalars): labels, classes, button type', () => withDom(async () => {
  for (const [opId, values] of [['regionprops', TBL], ['focus_score', SCL]]) {
    const rig = makePanel(opId, {});
    rig.panel.render(rig.step, rig.notes(values));
    const head = rig.container.one('dv-head');
    assert.ok(head, opId);
    const btns = head.children.filter((c) => c.tagName === 'BUTTON');
    const iFreeze = btns.findIndex((b) => b.classList.contains('dv-freeze'));
    assert.ok(iFreeze >= 0);
    assert.deepEqual(btns.slice(iFreeze + 1).map((b) => b.textContent), ['📋 คัดลอก', '💾 CSV'], opId);
    assert.deepEqual(btns.slice(iFreeze + 1).map((b) => b.className), ['dv-btn dv-copy', 'dv-btn dv-csv']);
    assert.equal(btns[iFreeze].textContent, '📌 ตรึงค่า', 'freeze button unchanged');
    assert.ok(btns.every((b) => b.getAttribute('type') === null ? b.type === 'button' : true));
  }
}));

test('p8-ui-2. "📋 คัดลอก": writes the CSV of the shown values WITHOUT BOM to the clipboard, then shows "✓ คัดลอกแล้ว" and restores the label via a timer', () => withDom(async ({ clip, timers }) => {
  const rig = makePanel('regionprops', {});
  rig.panel.render(rig.step, rig.notes(TBL));
  const copy = rig.container.one('dv-copy');
  await copy.click();
  assert.deepEqual(clip.writes, ['index,area,note\r\n1,314,"a,b"\r\n2,452.5,\r\n']);
  assert.equal(copy.textContent, '✓ คัดลอกแล้ว');
  assert.ok(timers.some((t) => t.ms === 1500));
  timers.find((t) => t.ms === 1500).fn();
  assert.equal(copy.textContent, '📋 คัดลอก');
  assert.equal(rig.downloads.length, 0, 'copy does not download');
}));

test('p8-ui-3. copy falls back to execCommand when the clipboard API rejects (no throw)', () => withDom(async ({ clip }) => {
  clip.fail = true;
  let used = 0;
  globalThis.document.execCommand = (c) => { if (c === 'copy') used++; return true; };
  const rig = makePanel('focus_score', {});
  rig.panel.render(rig.step, rig.notes(SCL));
  await rig.container.one('dv-copy').click();
  assert.equal(used, 1);
  assert.equal(rig.container.one('dv-copy').textContent, '✓ คัดลอกแล้ว');
}));

test('p8-ui-4. "💾 CSV": cb.onDownloadCsv(text with BOM, "<opId>-YYYYMMDD-HHMMSS.csv") — scalars Infinity → inf · table escapes', () => withDom(async () => {
  const t = makePanel('regionprops', {});
  t.panel.render(t.step, t.notes(TBL));
  await t.container.one('dv-csv').click();
  assert.equal(t.downloads.length, 1);
  assert.equal(t.downloads[0][0], '\uFEFFindex,area,note\r\n1,314,"a,b"\r\n2,452.5,\r\n');
  assert.match(t.downloads[0][1], /^regionprops-\d{8}-\d{6}\.csv$/);
  const s = makePanel('focus_score', {});
  s.panel.render(s.step, s.notes(SCL));
  await s.container.one('dv-csv').click();
  assert.equal(s.downloads[0][0], '\uFEFFfocus_score,focus_mean\r\n12.5,inf\r\n');
  assert.match(s.downloads[0][1], /^focus_score-\d{8}-\d{6}\.csv$/);
}));

test('p8-ui-5. frozen: the buttons export the FROZEN snapshot, not the newer frame · unfreezing exports the latest again', () => withDom(async () => {
  const rig = makePanel('focus_score', {});
  rig.panel.render(rig.step, rig.notes(SCL));
  await rig.container.one('dv-freeze').click(); // freeze
  const newer = { kind: 'scalars', fields: [{ key: 'focus_score', label: 'คะแนน', value: 99 }, { key: 'focus_mean', label: 'เฉลี่ย', value: 1 }] };
  rig.panel.applyNotes(rig.notes(newer));
  await rig.container.one('dv-csv').click();
  assert.equal(rig.downloads.at(-1)[0], '\uFEFFfocus_score,focus_mean\r\n12.5,inf\r\n');
  await rig.container.one('dv-freeze').click(); // unfreeze
  await rig.container.one('dv-csv').click();
  assert.equal(rig.downloads.at(-1)[0], '\uFEFFfocus_score,focus_mean\r\n99,1\r\n');
}));

test('p8-ui-6. no onDownloadCsv in cb → clicking "💾 CSV" does nothing (no throw) · op without values → no buttons at all', () => withDom(async () => {
  const rig = makePanel('focus_score', {}, { onDownloadCsv: undefined });
  rig.panel.render(rig.step, rig.notes(SCL));
  await rig.container.one('dv-csv').click();
  const plain = makePanel('find_contours', {});
  plain.panel.render(plain.step, [{ id: 'analysis_detect', nodeId: 5, view: FILTERS.analysis_detect.subtitle(plain.step.params), info: 'พบเส้นขอบ 3 เส้น' }]);
  assert.equal(plain.container.one('dv-csv'), null);
  assert.equal(plain.container.one('dv-copy'), null);
}));

test('p8-ui-7. pass_fail box itself shows the values block with result / measured / lo / hi / reason + buttons; the CSV carries the verdict', () => withDom(async () => {
  const rig = makePanel('pass_fail', {});
  const g = grayMat(20, 20, () => 0); const ctx = { upstream: UP_OK };
  OPS.pass_fail.apply(cv, g, { metric: 'count', cond: 'range', lo: 1, hi: 5 }, ctx).delete(); g.delete();
  rig.panel.render(rig.step, rig.notes(ctx.values));
  await rig.container.one('dv-csv').click();
  assert.equal(rig.downloads[0][0], '\uFEFFresult,measured,lo,hi,reason\r\nOK,1,1,5,\r\n');
  assert.match(rig.downloads[0][1], /^pass_fail-/);
}));

test('p8-ui-8. app.js wires cb.onDownloadCsv to the existing downloadBlob (text/csv, UTF-8) and does not touch electron/main.js (static)', () => {
  const app = readSrc('app.js');
  assert.match(app, /onDownloadCsv: \(text, filename\) => downloadBlob\(new Blob\(\[text\], \{ type: "text\/csv;charset=utf-8" \}\), filename\)/);
  const main = fs.readFileSync(new URL('../electron/main.js', import.meta.url), 'utf8');
  assert.ok(!/csv/i.test(main), 'electron/main.js has no CSV code');
});

// ===================================================================
// Python codegen + parity
// ===================================================================
const code = (pipe) => buildPythonCode(pipe);
const ascii = (s) => /^[\x00-\x7f]*$/.test(s);
const printLines = (py) => py.split('\n').filter((l) => /^\s*print\(/.test(l));

test('p8-py-1. regionprops → pass_fail: snapshot `pf_up` right after the measuring step, verdict block, ASCII RESULT line, every print() ASCII', () => {
  const py = code([step('regionprops', {}), pf({ metric: 'count', cond: 'range', lo: 1, hi: 5 })]);
  assert.match(py, /pf_up = \{'count': total_found, 'areas': \[r\['area_px'\] for r in regions\]\}/);
  assert.ok(py.indexOf("pf_up = {'count'") < py.indexOf('pf_measured = pf_up'), 'snapshot comes before the verdict');
  assert.match(py, /pf_measured = pf_up\['count'\]/);
  assert.match(py, /pf_ok = pf_measured is not None and \(pf_lo <= pf_measured <= pf_hi\)/);
  assert.match(py, /print\('RESULT: OK' if pf_ok else 'RESULT: NG'\)/);
  assert.match(py, /print\(f'measured: \{pf_measured!r\}'\)/);
  for (const l of printLines(py).filter((x) => /pf_|RESULT|measured|reason/.test(x))) assert.ok(ascii(l), l);
  assert.ok(py.includes('# ── ขั้นที่ 2: Pass / Fail (ตัดสิน OK / NG) (pass_fail) ──'));
});

test('p8-py-2. no pass_fail in the pipeline → the generated Python of every other op is byte-identical (no pf_up lines anywhere)', () => {
  for (const s of [step('regionprops', {}), step('roi_polygon', {}), step('focus_score', {}), step('shape_filter', {})]) {
    assert.ok(!code([s]).includes('pf_'), s.id);
  }
  // bypassed pass_fail → also nothing
  assert.ok(!code([step('regionprops', {}), off(pf({}))]).includes('pf_'));
});

test('p8-py-3. conditions → Python expressions (range / ge / le / eq) · numbers printed exactly (1e-7, 10000000)', () => {
  const body = (cond, extra = {}) => code([step('regionprops', {}), pf({ metric: 'count', cond, ...extra })]);
  assert.match(body('ge', { lo: 4 }), /pf_lo, pf_hi = 4, 100/); assert.match(body('ge', { lo: 4 }), /\(pf_measured >= pf_lo\)/);
  assert.match(body('le', { hi: 9 }), /\(pf_measured <= pf_hi\)/);
  assert.match(body('eq', { lo: 3 }), /abs\(pf_measured - pf_lo\) <= 1e-9 \* max\(1\.0, abs\(pf_lo\)\)/);
  assert.match(body('range', { lo: 0.0000001, hi: 1e7 }), /pf_lo, pf_hi = 1e-7, 10000000/);
});

test('p8-py-4. metrics: area_sum = explicit loop (no sum()), area_max = max(), roi_mean / focus snapshots, mm → area × (mm_per_px²)', () => {
  const a = code([step('regionprops', {}), pf({ metric: 'area_sum', cond: 'ge', lo: 0 })]);
  assert.match(a, /for pf_a in pf_up\['areas'\]:/); assert.ok(!/\bsum\(/.test(a.split('pf_up')[2] || ''), 'no sum() in the verdict block');
  assert.match(code([step('regionprops', {}), pf({ metric: 'area_max', cond: 'ge', lo: 0 })]), /pf_measured = max\(pf_up\['areas'\]\) if pf_up\['areas'\] else None/);
  assert.match(code([step('regionprops', { mmPerPx: 0.5 }), pf({ metric: 'area_sum', cond: 'ge', lo: 0 })]), /'areas': \[r\['area_px'\] \* \(mm_per_px \* mm_per_px\) for r in regions\]/);
  const r = code([step('roi_polygon', {}), pf({ metric: 'roi_mean', cond: 'ge', lo: 0 })]);
  assert.match(r, /pf_up = \{'value': roi_mean\}/); assert.match(r, /pf_measured = pf_up\['value'\]/);
  const f = code([step('focus_score', {}), pf({ metric: 'focus', cond: 'ge', lo: 0 })]);
  assert.match(f, /pf_up = \{'value': score\}/);
});

test('p8-py-5. no upstream / wrong box → NG lines (like the app) with ASCII reasons · bypassed measuring box is skipped (no snapshot, NG)', () => {
  const none = code([pf({})]);
  assert.match(none, /pf_reason = 'no upstream values'/); assert.ok(!none.includes('pf_up'));
  const wrong = code([step('regionprops', {}), step('focus_score', {}), pf({ metric: 'count' })]);
  assert.match(wrong, /pf_reason = 'values come from another box'/);
  const bypass = code([off(step('regionprops', {})), pf({})]);
  assert.match(bypass, /no upstream values/);
  const skipped = code([step('regionprops', {}), off(step('focus_score', {})), pf({ metric: 'count' })]);
  assert.match(skipped, /pf_measured = pf_up\['count'\]/, 'a bypassed box between does not replace the source');
});

test('p8-py-6. unsupported pair (delta_e → de_mean) → WARNING comment + ASCII print, image passes through (gray→BGR only), no RESULT line · works in the 2-image script too', () => {
  const de = { ...step('delta_e', {}), src2: { pipeline: [] } };
  const py = code([de, pf({ metric: 'de_mean', cond: 'le', hi: 2.3 })]);
  assert.match(py, /# metric 'de_mean' มาจาก Color Difference ΔE .* ซึ่งยังไม่รองรับ/);
  assert.match(py, /print\("WARNING: skipped 'pass_fail' \(metric from 'delta_e' is app-only in the Python export\) - no RESULT, differs from the app"\)/);
  assert.ok(!py.includes('RESULT:'));
  assert.ok(py.includes('def process(path_a, path_b):'), 'the 2-image builder is used');
  for (const l of printLines(py).filter((x) => /WARNING: skipped 'pass_fail'/.test(x))) assert.ok(ascii(l));
});

test('p8-py-7. pass_fail on line B → WARNING skip (values never cross lines) · the main line is unaffected', () => {
  const de = { ...step('delta_e', {}), src2: { pipeline: [pf({})] } };
  const py = code([de]);
  assert.match(py, /print\("WARNING: skipped 'pass_fail' in line B \(not supported in the Python export\)"\)/);
});

test('p8-py-8. PF_VALUE_OPS = exactly the ops of registry.js that assign ctx.values (static scan, minus pass_fail itself)', () => {
  const src = readSrc('filters/registry.js');
  const start = src.indexOf('export const OPS = {');
  const end = src.indexOf('\n};\n\n// ---- ตัวช่วย/ตารางของ IP batch 4B');
  assert.ok(start > 0 && end > start);
  const found = new Set();
  let cur = null;
  for (const line of src.slice(start, end).split('\n')) {
    const m = /^  ([a-z_0-9]+): \{$/.exec(line);
    if (m) cur = m[1];
    if (cur && /ctx\.values\s*=/.test(line)) found.add(cur);
  }
  found.delete('pass_fail');
  assert.deepEqual([...found].sort(), [...PF_VALUE_OPS].sort());
});

// ---- Python จริง ----
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'p8-1017-'));
process.on('exit', () => { try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ } });
function pythonAvailable() { try { execFileSync('python', ['-c', 'import cv2, numpy'], { encoding: 'utf8' }); return true; } catch { return false; } }
const HAS_PYTHON = !process.env.P8_NO_PY && pythonAvailable();
const PY_SKIP = { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' };
const swapRB3 = (buf) => { const out = Buffer.from(buf); for (let i = 0; i < out.length; i += 3) { const t = out[i]; out[i] = out[i + 2]; out[i + 2] = t; } return out; };
const PY_DRIVER = `import sys, json, io, contextlib
import numpy as np
c = json.load(open(sys.argv[1]))
ns = {'__name__': 'case'}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(open(c['code'], encoding='utf-8').read(), c['code'], 'exec'), ns)
    r = ns['process'](*c['inps'])
r.tofile(c['out'])
json.dump({'stdout': buf.getvalue(), 'ndim': int(r.ndim), 'shape': [int(v) for v in r.shape]}, open(sys.argv[2], 'w'))
`;
let pyCounter = 0;
function runPython(src, m) { // m = Mat RGB · script 2 ภาพ (process(path_a, path_b)) ใช้ภาพเดียวกันเป็นทั้ง A และ B
  const id = `${process.pid}_${pyCounter++}`;
  const rd = (name) => `np.fromfile(${name}, dtype=np.uint8).reshape((${m.rows}, ${m.cols}, 3))`;
  const two = src.includes('def process(path_a, path_b):');
  let c = two ? src.replace('img = cv2.imread(path_a)', `img = ${rd('path_a')}`).replace('img_b = cv2.imread(path_b)', `img_b = ${rd('path_b')}`) : src.replace('img = cv2.imread(path)', `img = ${rd('path')}`);
  c = c.slice(0, c.indexOf('if __name__ == "__main__":'));
  const o = { code: path.join(TMP_DIR, `c_${id}.py`), inps: two ? [path.join(TMP_DIR, `i_${id}.bin`), path.join(TMP_DIR, `i_${id}.bin`)] : [path.join(TMP_DIR, `i_${id}.bin`)], out: path.join(TMP_DIR, `o_${id}.bin`) };
  fs.writeFileSync(o.code, c, 'utf8');
  fs.writeFileSync(o.inps[0], swapRB3(Buffer.from(m.data)));
  const mp = path.join(TMP_DIR, `m_${id}.json`), rp = path.join(TMP_DIR, `r_${id}.json`), dp = path.join(TMP_DIR, `d_${id}.py`);
  fs.writeFileSync(mp, JSON.stringify(o)); fs.writeFileSync(dp, PY_DRIVER);
  execFileSync('python', [dp, mp, rp], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' }, maxBuffer: 1 << 28 });
  const r = JSON.parse(fs.readFileSync(rp, 'utf8'));
  const raw = fs.readFileSync(o.out);
  return { stdout: r.stdout, shape: r.shape, ndim: r.ndim, bytes: r.ndim === 3 ? swapRB3(raw) : Buffer.from(raw) };
}
const pyResult = (stdout) => {
  const res = /^RESULT: (OK|NG)$/m.exec(stdout);
  const mea = /^measured: (.*)$/m.exec(stdout);
  return { result: res ? res[1] : null, measured: mea ? (mea[1] === 'None' ? null : Number(mea[1])) : undefined };
};

// ตารางเคส: [ป้าย, pipeline (ไม่รวม pass_fail), พารามิเตอร์ pass_fail] — ภาพ 3 วงกลมทุกเคส
const PARITY = [
  ['count range OK', [step('regionprops', {})], { metric: 'count', cond: 'range', lo: 1, hi: 5 }],
  ['count eq 3 OK', [step('regionprops', {})], { metric: 'count', cond: 'eq', lo: 3 }],
  ['count ge 4 NG', [step('regionprops', {})], { metric: 'count', cond: 'ge', lo: 4 }],
  ['count le 2 NG', [step('regionprops', {})], { metric: 'count', cond: 'le', hi: 2 }],
  ['count capped (maxObjects 1) still 3', [step('regionprops', { maxObjects: 1 })], { metric: 'count', cond: 'eq', lo: 3 }],
  ['area_sum', [step('regionprops', {})], { metric: 'area_sum', cond: 'ge', lo: 1000 }],
  ['area_sum capped', [step('regionprops', { maxObjects: 2 })], { metric: 'area_sum', cond: 'le', hi: 99999 }],
  ['area_max', [step('regionprops', {})], { metric: 'area_max', cond: 'range', lo: 100, hi: 5000 }],
  ['area_max mm', [step('regionprops', { mmPerPx: 0.25 })], { metric: 'area_max', cond: 'ge', lo: 1 }],
  ['area_sum minArea kills all (empty table) NG', [step('regionprops', { minArea: 5000 })], { metric: 'area_sum', cond: 'ge', lo: 0 }],
  ['count 0 objects, eq 0 OK', [step('regionprops', { minArea: 5000 })], { metric: 'count', cond: 'eq', lo: 0 }],
  ['roi_mean', [step('roi_polygon', {})], { metric: 'roi_mean', cond: 'range', lo: 0, hi: 255 }],
  ['roi_mean NG', [step('roi_polygon', {})], { metric: 'roi_mean', cond: 'ge', lo: 255 }],
  ['focus', [step('focus_score', {})], { metric: 'focus', cond: 'ge', lo: 0 }],
  ['focus tenengrad le 1 NG', [step('focus_score', { method: 'tenengrad' })], { metric: 'focus', cond: 'le', hi: 1 }],
  ['no upstream NG', [], { metric: 'count', cond: 'ge', lo: 0 }],
  ['wrong box NG', [step('regionprops', {}), step('focus_score', {})], { metric: 'count', cond: 'ge', lo: 0 }],
  ['bypassed box in between', [step('regionprops', {}), off(step('focus_score', {}))], { metric: 'count', cond: 'eq', lo: 3 }],
];

test('p8-parity-1. app verdict == exported Python verdict (RESULT line + measured value, exact) and the border colour pixel matches, on the 3-circle image', PY_SKIP, () => {
  const base = circlesRGB();
  try {
    for (const [label, pre, p] of PARITY) {
      const pipeline = [...pre, pf(p, 9)];
      const r = runApp(pipeline, base);
      let app;
      try {
        const f = fieldsOf(r.notes, 9);
        app = { result: f.result, measured: f.measured, edge: px(r.mat, 1, 45) };
      } finally { release(r); }
      const py = runPython(code(pipeline), base);
      const got = pyResult(py.stdout);
      assert.equal(got.result, app.result, `${label}: verdict`);
      assert.equal(got.measured, app.measured, `${label}: measured`);
      assert.deepEqual([py.ndim, py.shape], [3, [90, 120, 3]], label);
      assert.deepEqual(Array.from(py.bytes.subarray((45 * 120 + 1) * 3, (45 * 120 + 1) * 3 + 3)), app.edge, `${label}: border pixel`);
      assert.deepEqual(app.edge, app.result === 'OK' ? [0, 200, 0] : [255, 0, 0], label);
    }
  } finally { base.delete(); }
});

test('p8-parity-2. unsupported pair in Python (delta_e → de_mean): the 2-image script still runs, prints the ASCII WARNING and no RESULT, returns a 3-channel image · after Threshold the pipeline works too', PY_SKIP, () => {
  const base = circlesRGB();
  try {
    const de = step('delta_e', {}, 1, { src2: { pipeline: [] } });
    const out = runPython(code([de, pf({ metric: 'de_mean', cond: 'le', hi: 2.3 }, 9)]), base);
    assert.ok(!out.stdout.includes('RESULT:'));
    assert.match(out.stdout, /^WARNING: skipped 'pass_fail' \(metric from 'delta_e' is app-only in the Python export\)/m);
    assert.deepEqual([out.ndim, out.shape], [3, [90, 120, 3]]);
    const thr = runPython(code([step('binary_threshold', {}), step('regionprops', {}), pf({ metric: 'count', cond: 'eq', lo: 3 })]), base);
    assert.match(thr.stdout, /^RESULT: OK$/m);
  } finally { base.delete(); }
});

// ===================================================================
// รอบแก้หลังรีวิว (M1 / m1 / m2 / nits)
// ===================================================================
test('p8-fix-1. M1: an app-only box (mog2) between the measuring box and pass_fail becomes the latest source in Python too → "another box" NG, single and 2-image scripts, and the unsupported (no GEN) branch', () => {
  for (const mid of ['mog2', 'farneback_optical_flow', 'pyrlk_optical_flow']) {
    const py = code([step('regionprops', {}), step(mid, {}, 2), pf({ metric: 'count', cond: 'ge', lo: 1 })]);
    assert.match(py, /pf_reason = 'values come from another box'/, mid);
    assert.ok(!py.includes("pf_measured = pf_up['count']"), mid);
  }
  // 2-image script
  const two = code([{ ...step('delta_e', {}, 1), src2: { pipeline: [] } }, step('regionprops', {}, 2), step('mog2', {}, 3), pf({ metric: 'count', cond: 'ge', lo: 1 })]);
  assert.ok(two.includes('def process(path_a, path_b):'));
  assert.match(two, /pf_reason = 'values come from another box'/);
  // without pass_fail nothing changes
  assert.ok(!code([step('regionprops', {}), step('mog2', {}, 2)]).includes('pf_'));
  // an app-only box that is NOT a values box would not matter; and a bypassed mog2 is skipped entirely
  assert.match(code([step('regionprops', {}), off(step('mog2', {}, 2)), pf({ metric: 'count', cond: 'ge', lo: 1 })]), /pf_measured = pf_up\['count'\]/);
});

test('p8-fix-2. M1 parity with the app: regionprops → mog2 → pass_fail is NG in the worker and in the exported Python', PY_SKIP, () => {
  const base = circlesRGB();
  try {
    const pipeline = [step('regionprops', {}, 1), step('mog2', {}, 2), pf({ metric: 'count', cond: 'ge', lo: 1 }, 9)];
    const r = runApp(pipeline, base);
    try { assert.equal(fieldsOf(r.notes, 9).result, 'NG'); } finally { release(r); }
    assert.match(runPython(code(pipeline), base).stdout, /^RESULT: NG$/m);
  } finally { base.delete(); }
});

test('p8-fix-3. lo > hi: reason in the app AND in Python (ASCII) · Python verdict NG', PY_SKIP, () => {
  const base = circlesRGB();
  try {
    const pipeline = [step('regionprops', {}), pf({ metric: 'count', cond: 'range', lo: 10, hi: 2 })];
    const out = runPython(code(pipeline), base).stdout;
    assert.match(out, /^RESULT: NG$/m); assert.match(out, /^reason: low limit is greater than high limit$/m);
    assert.ok(out.split(String.fromCharCode(10)).filter((l) => /^(RESULT|measured|reason)/.test(l)).every((l) => ascii(l)));
  } finally { base.delete(); }
});

test('p8-fix-4. CSV: rows are trimmed/padded to the header · capped table keeps the real count as "# total,<n>" (only when total > rows) · BOM written as an escape', () => {
  assert.equal(valuesToCsv({ kind: 'table', columns: ['a', 'b'], rows: [[1, 2, 3], [4]] }, { bom: false }), 'a,b\r\n1,2\r\n4,\r\n');
  const capped = valuesToCsv({ kind: 'table', columns: ['a'], rows: [[1], [2]], total: 7 }, { bom: false });
  assert.equal(capped, 'a\r\n1\r\n2\r\n# total,7\r\n');
  assert.equal(valuesToCsv({ kind: 'table', columns: ['a'], rows: [[1], [2]], total: 2 }, { bom: false }), 'a\r\n1\r\n2\r\n');
  assert.equal(valuesToCsv({ kind: 'table', columns: ['a'], rows: [], total: 0 }, { bom: false }), 'a\r\n');
  assert.equal(valuesToCsv({ kind: 'table', columns: ['a'], rows: [[1]], total: NaN }, { bom: false }), 'a\r\n1\r\n');
  assert.ok(valuesToCsv({ kind: 'scalars', fields: [] }).startsWith('\uFEFF'));
  const csvSrc = readSrc('ui/values-csv.js');
  assert.ok(csvSrc.includes(String.fromCharCode(34, 92) + 'uFEFF' + String.fromCharCode(34)), 'escape, not the literal character');
  assert.ok(!csvSrc.includes(String.fromCharCode(0xFEFF)));
});

test('p8-fix-5. opIdOfStep: unknown mode falls back to the default mode like the group apply()', () => {
  assert.equal(opIdOfStep({ id: 'analysis_detect', params: { mode: 'nonsense' } }), 'find_contours');
  assert.equal(opIdOfStep({ id: 'analysis_detect', params: { mode: 'constructor' } }), 'find_contours');
});

test('p8-fix-6. copy: "✓ คัดลอกแล้ว" only when the fallback execCommand returns true, otherwise a failure label', () => withDom(async ({ clip }) => {
  clip.fail = true;
  for (const [ret, want] of [[true, '✓ คัดลอกแล้ว'], [false, '✗ คัดลอกไม่สำเร็จ']]) {
    globalThis.document.execCommand = () => ret;
    const rig = makePanel('focus_score', {});
    rig.panel.render(rig.step, rig.notes(SCL));
    await rig.container.one('dv-copy').click();
    assert.equal(rig.container.one('dv-copy').textContent, want);
  }
}));

test('p8-fix-7. floats in info and card summary are trimmed (0.1+0.2 → 0.3, never 0.30000000000000004)', () => {
  const g = grayMat(20, 20, () => 0); const ctx = { upstream: SC('focus_score', 'focus_score', 1) };
  OPS.pass_fail.apply(cv, g, { metric: 'focus', cond: 'range', lo: 0.1 + 0.2, hi: 5 }, ctx).delete(); g.delete();
  assert.ok(!/0\.3000/.test(ctx.info)); assert.match(ctx.info, /0\.3–5/);
  assert.equal(OP_PRIMARY.pass_fail({ metric: 'focus', cond: 'ge', lo: 0.1 + 0.2 }), 'ความคม ผ่านเมื่อ ≥ 0.3');
  assert.match(OPS.pass_fail.apply.toString(), /pfNum/);
});

test('p8-fix-8. stale values: while waiting the hint says the numbers are old (ค่าเก่า) and the block is greyed (is-waiting) · fresh values clear it', () => withDom(async () => {
  const rig = makePanel('focus_score', {});
  rig.panel.render(rig.step, rig.notes(SCL));
  const hint = rig.container.one('dv-hint');
  assert.ok(hint);
  assert.match(readSrc('ui/detail-panel.js'), /\(ค่าเก่า — ยังไม่ใช่ผลของเฟรมล่าสุด\) กำลังรอค่า/);
  assert.match(readSrc('../styles/app.css'), /\.dv\.is-waiting \.dv-scroll, \.dv\.is-waiting \.dv-list \{ opacity: \.45; \}/);
}));

// P-8 C-B (2026-10-17) — batch CSV: ค่าที่วัดได้ของทุกภาพในโฟลเดอร์ → <stem>-summary.csv (+ <stem>-objects.csv เมื่อมีตาราง)
// WORKLOG: WORKLOG-2026-10-17-p8-batch-csv.md
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import { defaultParams } from '../src/renderer/filters/registry.js';
import { OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { createBatchCsv, saveBatchCsv, finishBatchCsv, MAX_OBJECT_ROWS } from '../src/renderer/ui/batch-csv.js';
import { assertNoHeapGrowth } from './_leak-helpers.mjs';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cvtColor === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}
globalThis.self = globalThis;
if (typeof globalThis.ImageData === 'undefined') globalThis.ImageData = class { constructor(d, w, h) { this.data = d; this.width = w; this.height = h; } };
const { computeFrame } = await import('../src/renderer/cv/cv-worker.js');
const readSrc = (rel) => fs.readFileSync(new URL(`../src/renderer/${rel}`, import.meta.url), 'utf8');

const BOM = String.fromCharCode(0xfeff);
const lines = (text) => text.replace(BOM, '').split('\r\n').slice(0, -1);
const cells = (line) => line.split(',');
const sc = (...kv) => ({ kind: 'scalars', fields: kv.map(([key, value]) => ({ key, label: key, value })) });
const tb = (columns, rows, total = rows.length) => ({ kind: 'table', columns, rows, total });
const ent = (op, nodeId, values) => ({ op, id: op, nodeId, values });

// ===================================================================
// builder
// ===================================================================
test('p8-bc-1. summary: BOM + CRLF, one row per image, header = file + "<op>.<key>" · table → "<op>.total" · objects file only when a table exists', () => {
  const b = createBatchCsv();
  b.add('a.png', [ent('focus_score', 1, sc(['focus_score', 12.5], ['focus_mean', 3]))]);
  b.add('b.png', [ent('focus_score', 1, sc(['focus_score', 7], ['focus_mean', 2]))]);
  const { files } = b.build('batch-x');
  assert.deepEqual(files.map((f) => f.name), ['batch-x-summary.csv']); // ไม่มีตาราง = ไม่มี objects
  const t = files[0].text;
  assert.equal(t.charCodeAt(0), 0xfeff);
  assert.ok(t.endsWith('\r\n'));
  assert.deepEqual(lines(t), ['file,focus_score.focus_score,focus_score.focus_mean', 'a.png,12.5,3', 'b.png,7,2']);
});

test('p8-bc-2. column union across images with different keys: missing = empty, first-seen order kept', () => {
  const b = createBatchCsv();
  b.add('a.png', [ent('focus_score', 1, sc(['x', 1]))]);
  b.add('b.png', [ent('focus_score', 1, sc(['x', 2], ['y', 5]))]);
  b.add('c.png', []); // ภาพที่ไม่มีค่า → แถวว่าง
  const l = lines(b.build('s').files[0].text);
  assert.deepEqual(l, ['file,focus_score.x,focus_score.y', 'a.png,1,', 'b.png,2,5', 'c.png,,']);
});

test('p8-bc-3. same op twice (2 boxes) → "<op>" and "<op>#2" prefixes, stable across images; same node seen from two output lines counts once', () => {
  const b = createBatchCsv();
  const e1 = ent('roi_polygon', 10, sc(['roi_mean', 100]));
  const e2 = ent('roi_polygon', 11, sc(['roi_mean', 200]));
  b.add('a.png', [e1, e2, e1, e2]); // สายผลลัพธ์ 2 สายผ่านกล่องเดียวกัน
  b.add('b.png', [ent('roi_polygon', 11, sc(['roi_mean', 5]))]); // ภาพนี้มีแค่กล่องที่ 2 → ยังเป็น #2
  assert.deepEqual(lines(b.build('s').files[0].text), ['file,roi_polygon.roi_mean,roi_polygon#2.roi_mean', 'a.png,100,200', 'b.png,,5']);
});

test('p8-bc-4. pass_fail: verdict + reason columns after file; the box keeps measured/lo/hi (result/reason not repeated); NG carries the reason; Thai text kept', () => {
  const b = createBatchCsv();
  const pf = (result, measured, reason) => ent('pass_fail', 9, sc(['result', result], ['measured', measured], ['lo', 1], ['hi', 100], ['reason', reason]));
  b.add('ok.png', [ent('regionprops', 1, tb(['index', 'area_px'], [[1, 50]], 1)), pf('OK', 1, null)]);
  b.add('ng.png', [ent('regionprops', 1, tb(['index', 'area_px'], [], 0)), pf('NG', 0, 'จำนวนวัตถุ 0 ต่ำกว่า 1, "มาก"')]);
  const t = b.build('s').files[0].text;
  const l = lines(t);
  assert.equal(l[0], 'file,verdict,reason,regionprops.total,pass_fail.measured,pass_fail.lo,pass_fail.hi');
  assert.equal(l[1], 'ok.png,OK,,1,1,1,100');
  assert.equal(l[2], 'ng.png,NG,"จำนวนวัตถุ 0 ต่ำกว่า 1, ""มาก""",0,0,1,100');
});

test('p8-bc-5. two pass_fail boxes: verdict = NG if any NG (reasons joined) · both boxes keep their own result/reason columns', () => {
  const b = createBatchCsv();
  const pf = (id, result, reason) => ent('pass_fail', id, sc(['result', result], ['measured', 1], ['reason', reason]));
  b.add('a.png', [pf(1, 'OK', null), pf(2, 'NG', 'เกินช่วง')]);
  b.add('b.png', [pf(1, 'OK', null), pf(2, 'OK', null)]);
  const l = lines(b.build('s').files[0].text);
  assert.equal(l[0], 'file,verdict,reason,pass_fail.result,pass_fail.measured,pass_fail.reason,pass_fail#2.result,pass_fail#2.measured,pass_fail#2.reason');
  assert.equal(l[1], 'a.png,NG,เกินช่วง,OK,1,,NG,1,เกินช่วง');
  assert.equal(l[2], 'b.png,OK,,OK,1,,OK,1,');
});

test('p8-bc-6. table: summary gets <op>.total (true total, not capped rows) · objects file = file, box, row, columns (union) · 1-based row', () => {
  const b = createBatchCsv();
  b.add('a.png', [ent('regionprops', 1, tb(['index', 'area_px'], [[1, 10], [2, 20]], 7))]);
  b.add('b.png', [ent('regionprops', 1, tb(['index', 'area_px', 'cx'], [[1, 5, 3.5]], 1))]);
  const { files } = b.build('s');
  assert.deepEqual(files.map((f) => f.name), ['s-summary.csv', 's-objects.csv']);
  assert.deepEqual(lines(files[0].text), ['file,regionprops.total', 'a.png,7', 'b.png,1']);
  assert.deepEqual(lines(files[1].text), ['file,box,row,index,area_px,cx', 'a.png,regionprops,1,1,10,', 'a.png,regionprops,2,2,20,', 'b.png,regionprops,1,1,5,3.5', '# truncated,3,5']); // total 7 แต่ส่งมา 2 แถว
  assert.equal(files[1].text.charCodeAt(0), 0xfeff);
});

test('p8-bc-7. a table box with zero objects in every image still produces the objects file (header only) and total 0', () => {
  const b = createBatchCsv();
  b.add('a.png', [ent('regionprops', 1, tb(['index'], [], 0))]);
  const { files } = b.build('s');
  assert.equal(files.length, 2);
  assert.deepEqual(lines(files[1].text), ['file,box,row,index']);
  assert.deepEqual(lines(files[0].text), ['file,regionprops.total', 'a.png,0']);
});

test('p8-bc-8. error images: row with file + error column (batch continues) · error column only when some image failed · escaping/inf/null rules from valuesToCsv', () => {
  const b = createBatchCsv();
  b.add('a.png', [ent('focus_score', 1, sc(['v', Infinity], ['w', null], ['n', NaN]))]);
  b.addError('bad,name.png', 'เปิดภาพไม่ได้');
  b.add('c.png', [ent('focus_score', 1, sc(['v', -Infinity], ['w', 'x"y'], ['n', 0]))]);
  const l = lines(b.build('s').files[0].text);
  assert.deepEqual(l, ['file,focus_score.v,focus_score.w,focus_score.n,error', 'a.png,inf,,,', '"bad,name.png",,,,เปิดภาพไม่ได้', 'c.png,-inf,"x""y",0,']);
  // ไม่มีภาพพัง → ไม่มีคอลัมน์ error
  const g = createBatchCsv(); g.add('a.png', [ent('focus_score', 1, sc(['v', 1]))]);
  assert.ok(!g.build('s').files[0].text.includes('error'));
});

test('p8-bc-9. no measuring box at all → no files (even when images errored) · empty batch → no files · errors alongside a box still get rows', () => {
  const e = createBatchCsv(); e.addError('a.png', 'หมดเวลา');
  assert.deepEqual(e.build('s').files, []);
  const m = createBatchCsv(); m.addError('a.png', 'x'); m.add('b.png', [ent('focus_score', 1, sc(['v', 1]))]);
  assert.deepEqual(lines(m.build('s').files[0].text), ['file,focus_score.v,error', 'a.png,,x', 'b.png,1,']);
  const n = createBatchCsv(); n.add('a.png', []); n.add('b.png', undefined);
  assert.deepEqual(n.build('s').files, []);
  assert.deepEqual(createBatchCsv().build('s').files, []);
});

test('p8-bc-10. object-row cap: rows beyond maxObjectRows are dropped (counted in stats) but <op>.total stays true · default cap constant', () => {
  assert.equal(MAX_OBJECT_ROWS, 50000);
  const b = createBatchCsv({ maxObjectRows: 3 });
  b.add('a.png', [ent('regionprops', 1, tb(['i'], [[1], [2]], 2))]);
  b.add('b.png', [ent('regionprops', 1, tb(['i'], [[1], [2]], 2))]);
  const { files, stats } = b.build('s');
  assert.equal(stats.objectRowsKept, 3);
  assert.equal(stats.objectRowsDropped, 1);
  assert.deepEqual(lines(files[0].text), ['file,regionprops.total', 'a.png,2', 'b.png,2']);
  assert.equal(lines(files[1].text).length, 1 + 3 + 1); // หัว + 3 แถว + # truncated,3,1
  assert.equal(lines(files[1].text).at(-1), '# truncated,3,1');
});

test('p8-bc-11. junk input never throws: values without kind / fields not array / null entries / ragged rows', () => {
  const b = createBatchCsv();
  b.add('a.png', [null, undefined, { op: 'x' }, ent('x', 1, { kind: 'weird' }), ent('y', 2, { kind: 'scalars' }), ent('z', 3, tb(['a', 'b'], [[1], [1, 2, 3], 'bad']))]);
  const { files } = b.build('s');
  assert.deepEqual(lines(files[1].text), ['file,box,row,a,b', 'a.png,z,1,1,', 'a.png,z,2,1,2', 'a.png,z,3,,']);
});

// ===================================================================
// worker export path
// ===================================================================
function circlesBuf(w = 120, h = 90) {
  const m = new cv.Mat.zeros(h, w, cv.CV_8UC3);
  for (const [x, y, r] of [[25, 25, 10], [70, 30, 12], [60, 65, 14]]) cv.circle(m, new cv.Point(x, y), r, new cv.Scalar(255, 255, 255), -1);
  const rgba = new cv.Mat(); cv.cvtColor(m, rgba, cv.COLOR_RGB2RGBA);
  const b = new Uint8ClampedArray(rgba.data).buffer.slice(0);
  m.delete(); rgba.delete();
  return b;
}
const step = (opId, extra = {}, nodeId = 1) => {
  const host = OP_TO_FILTER[opId];
  if (!host || host === opId) return { id: opId, params: { ...defaultParams(opId), ...extra }, enabled: true, nodeId };
  const prefixed = {};
  for (const [k, v] of Object.entries(extra)) prefixed[`${opId}__${k}`] = v;
  return { id: host, params: { ...defaultParams(host), mode: opId, ...prefixed }, enabled: true, nodeId };
};
const PIPE = () => [{ connected: true, displayId: 'out1', pipeline: [step('regionprops', {}, 1), step('pass_fail', { lo: 1, hi: 100 }, 2)] }];

test('p8-bc-12. worker export: result carries values of every box that produced them (op = real op, nodeId, table + pass_fail scalars) · image bytes identical with/without collectValues', () => {
  const buf = circlesBuf();
  const plain = computeFrame(120, 90, buf.slice(0), PIPE(), { isolatedState: true });
  const withV = computeFrame(120, 90, buf.slice(0), PIPE(), { isolatedState: true, collectValues: true });
  assert.equal(plain.results[0].values, undefined);
  assert.deepEqual(Object.keys(plain.results[0]).sort(), ['buffer', 'displayId', 'height', 'width']);
  assert.deepEqual(Buffer.from(plain.results[0].buffer), Buffer.from(withV.results[0].buffer), 'image bytes must not change');
  const vals = withV.results[0].values;
  assert.deepEqual(vals.map((v) => [v.op, v.nodeId]), [['regionprops', 1], ['pass_fail', 2]]);
  assert.equal(vals[0].values.kind, 'table');
  assert.equal(vals[0].values.total, 3);
  assert.equal(vals[0].values.rows.length, 3);
  const pf = Object.fromEntries(vals[1].values.fields.map((f) => [f.key, f.value]));
  assert.equal(pf.result, 'OK');
  assert.equal(pf.measured, 3);
  // notes เดิมของ live ยังเหมือนเดิม (op เป็นฟิลด์เพิ่มเฉย ๆ)
  assert.ok(withV.notes.some((n) => n.values && n.op === 'regionprops'));
});

test('p8-bc-13. worker → builder end to end: 2 images, one with 3 circles one empty → summary + objects with the real columns, verdict OK / NG', () => {
  const b = createBatchCsv();
  const empty = (() => { const m = new Uint8ClampedArray(120 * 90 * 4); for (let i = 3; i < m.length; i += 4) m[i] = 255; return m.buffer; })();
  for (const [name, buf] of [['circles.png', circlesBuf()], ['blank.png', empty]]) {
    const r = computeFrame(120, 90, buf, PIPE(), { isolatedState: true, collectValues: true });
    b.add(name, r.results.flatMap((x) => x.values || []));
  }
  const { files } = b.build('batch');
  const sum = lines(files[0].text);
  assert.equal(sum[0], 'file,verdict,reason,regionprops.total,pass_fail.measured,pass_fail.lo,pass_fail.hi');
  assert.equal(sum[1], 'circles.png,OK,,3,3,1,100');
  assert.ok(sum[2].startsWith('blank.png,NG,'), sum[2]);
  assert.ok(sum[2].includes(',0,0,1,100'), sum[2]);
  const obj = lines(files[1].text);
  assert.equal(obj.length, 1 + 3); // blank.png ไม่มีวัตถุ
  assert.ok(obj[0].startsWith('file,box,row,'));
  assert.ok(obj.slice(1).every((r, i) => r.startsWith(`circles.png,regionprops,${i + 1},`)));
  // ทุกแถวมีจำนวนช่องเท่าหัว
  assert.ok(obj.every((r) => cells(r).length === cells(obj[0]).length));
});

test('p8-bc-14. worker export path leaks nothing (heap flat) with collectValues', () => {
  const buf = circlesBuf();
  assertNoHeapGrowth(cv, 'computeFrame collectValues', () => { computeFrame(120, 90, buf.slice(0), PIPE(), { isolatedState: true, collectValues: true }); }, { iters: 30, warmup: 8 });
});

// ===================================================================
// save / app wiring
// ===================================================================
function sampleCsv() {
  const b = createBatchCsv();
  b.add('a.png', [ent('regionprops', 1, tb(['i'], [[1]], 1))]);
  return b;
}

test('p8-bc-15. Electron path: files written into the chosen dir via writeFile(dir, name, bytes) · names "<stem>-summary.csv" / "-objects.csv" · bytes start with EF BB BF · no download', async () => {
  const writes = [], downloads = [];
  const note = await finishBatchCsv(sampleCsv(), { dir: 'C:/out', writeFile: async (d, n, bytes) => writes.push([d, n, bytes]), download: (...a) => downloads.push(a), stem: 'batch-20261017-101500' });
  assert.deepEqual(writes.map((w) => [w[0], w[1]]), [['C:/out', 'batch-20261017-101500-summary.csv'], ['C:/out', 'batch-20261017-101500-objects.csv']]);
  assert.ok(writes.every((w) => w[2] instanceof Uint8Array && w[2][0] === 0xef && w[2][1] === 0xbb && w[2][2] === 0xbf));
  assert.equal(downloads.length, 0);
  assert.equal(note, ' · CSV 2 ไฟล์');
  assert.ok(new TextDecoder().decode(writes[0][2]).includes('a.png'));
});

test('p8-bc-16. browser fallback (no dir): downloadBlob(blob text/csv, name) for each file · writeFile never called', async () => {
  const writes = [], downloads = [];
  const note = await finishBatchCsv(sampleCsv(), { dir: null, writeFile: null, download: (blob, name) => downloads.push([blob, name]), stem: 's' });
  assert.deepEqual(downloads.map((d) => d[1]), ['s-summary.csv', 's-objects.csv']);
  assert.ok(downloads.every((d) => d[0].type === 'text/csv;charset=utf-8'));
  assert.equal(writes.length, 0);
  assert.equal(note, ' · CSV 2 ไฟล์');
  const bytes = new Uint8Array(await downloads[0][0].arrayBuffer());
  assert.deepEqual(Array.from(bytes.slice(0, 3)), [0xef, 0xbb, 0xbf]);
});

test('p8-bc-17. option off (csv = null) → nothing written, no status text · no values and no errors → no files + explanatory text · write failure does not throw', async () => {
  const writes = [];
  assert.equal(await finishBatchCsv(null, { dir: 'd', writeFile: async () => writes.push(1), download: () => writes.push(2), stem: 's' }), '');
  assert.equal(writes.length, 0);
  const none = createBatchCsv(); none.add('a.png', []);
  const note = await finishBatchCsv(none, { dir: 'd', writeFile: async () => writes.push(1), download: () => writes.push(2), stem: 's' });
  assert.equal(writes.length, 0);
  assert.equal(note, ''); // ไม่มีกล่องที่ให้ค่า → เงียบ
  const fail = await finishBatchCsv(sampleCsv(), { dir: 'd', writeFile: async () => { throw new Error('disk'); }, download: null, stem: 's' });
  assert.equal(fail, ' · บันทึก CSV ไม่สำเร็จ');
  assert.equal(await saveBatchCsv([{ name: 'x.csv', text: 'a' }], { dir: null, writeFile: null, download: null }), 0);
});

test('p8-bc-18. note text mentions error images and dropped object rows', async () => {
  const b = createBatchCsv({ maxObjectRows: 1 });
  b.add('a.png', [ent('regionprops', 1, tb(['i'], [[1], [2], [3]], 3))]);
  b.addError('b.png', 'x');
  const note = await finishBatchCsv(b, { dir: null, download: () => {}, stem: 's' });
  assert.equal(note, ' · CSV 2 ไฟล์ (ภาพพลาด 1) · แถววัตถุเกิน 1 แถวถูกตัดทิ้ง 2');
});

test('p8-bc-19. app.js wiring: option default ON, off → csv = null · per-image error rows (load fail / empty results) · uses finishBatchCsv + downloadBlob · export:writeFile IPC only (main.js has no CSV code, package.json untouched by name)', () => {
  const app = readSrc('app.js');
  assert.match(app, /let batchCsvOn = true;/);
  assert.match(app, /const csv = csvOn \? createBatchCsv\(\) : null;/);
  assert.match(app, /csv\.addError\(rec\.name, "เปิดภาพไม่ได้"\)/);
  assert.match(app, /results\.length === 0\) csv\.addError\(rec\.name, results\.error/);
  assert.match(app, /csv\.add\(rec\.name, results\.flatMap\(\(r\) => r\.values \|\| \[\]\)\)/);
  assert.match(app, /finishBatchCsv\(csv, \{[\s\S]*?window\.mvExport\.writeFile\(d, n, b\)[\s\S]*?download: downloadBlob/);
  assert.match(app, /getBatchCsv: \(\) => batchCsvOn/);
  assert.match(app, /onBatchCsv: \(on\) => \{ batchCsvOn = !!on; \}/);
  assert.match(app, /exportWaiters\.set[\s\S]*?Object\.assign\(msg\.results \|\| \[\], \{ error: msg\.error \}\)/);
  const main = fs.readFileSync(new URL('../electron/main.js', import.meta.url), 'utf8');
  assert.ok(!/csv/i.test(main), 'electron/main.js must not know about CSV');
  // batchCsvOn ต้องประกาศก่อน createSourceBar (render ครั้งแรกเรียก getBatchCsv)
  assert.ok(app.indexOf('let batchCsvOn') < app.indexOf('const sourceBar = createSourceBar('));
});

test('p8-bc-20. worker source: export message asks for collectValues; live frame does not', () => {
  const w = readSrc('cv/cv-worker.js');
  assert.match(w, /isolatedState: true, collectValues: true/);
  assert.match(w, /computeFrame\(msg\.width, msg\.height, msg\.buffer, msg\.pipelines \|\| \[\]\);/); // live ไม่ส่ง opts
});

// ---- checkbox ใน source-bar (DOM stub เล็ก ๆ) ----
function makeEl(tag) {
  const el = {
    tagName: tag.toUpperCase(), children: [], className: '', textContent: '', title: '', style: {}, listeners: {}, checked: false, type: '',
    classList: { contains() { return false; }, add() {}, remove() {} },
    appendChild(c) { this.children.push(c); return c; }, append(...cs) { this.children.push(...cs); },
    setAttribute() {}, addEventListener(t, f) { this.listeners[t] = f; },
    set innerHTML(v) { this.children = []; }, get innerHTML() { return ''; },
    querySelector() { return null; },
  };
  return el;
}
function walk(el, out = []) { out.push(el); for (const c of el.children) walk(c, out); return out; }

async function renderBar(cbExtra) {
  const prev = globalThis.document;
  globalThis.document = { createElement: makeEl, body: { classList: { contains() { return false; } } }, getElementById: () => null, addEventListener() {} };
  try {
    const { createSourceBar } = await import('../src/renderer/ui/source-bar.js');
    const container = makeEl('div');
    const bar = createSourceBar(container, {
      getSource: () => 'image', onSource() {}, getCameraMode: () => 'original', onCameraMode() {}, getCameras: () => [], getCameraId: () => '',
      onSelectCamera() {}, onImportClick() {}, levels: [{ value: 'a', label: 'a' }], getLevel: () => 'a', onLevel() {},
      onSaveImages() {}, getShortcut: () => '', ...cbExtra,
    });
    if (bar && bar.render) bar.render();
    return walk(container);
  } finally { if (prev) globalThis.document = prev; else delete globalThis.document; }
}

test('p8-bc-21. source-bar: checkbox "📊 CSV ค่าที่วัด" next to Save Picture only for image batch > 1 · reflects getBatchCsv · change → onBatchCsv', async () => {
  const calls = [];
  const many = await renderBar({ getBatch: () => ({ index: 0, total: 3, name: 'a.png' }), getBatchCsv: () => true, onBatchCsv: (v) => calls.push(v) });
  const wrap = many.find((e) => e.className === 'batch-csv-opt');
  assert.ok(wrap, 'checkbox label exists');
  const box = wrap.children.find((c) => c.type === 'checkbox');
  assert.equal(box.checked, true);
  assert.ok(wrap.children.some((c) => c.textContent === '📊 CSV ค่าที่วัด'));
  box.checked = false; box.listeners.change();
  assert.deepEqual(calls, [false]);
  // ภาพเดี่ยว / ไม่มี callback → ไม่โผล่
  const single = await renderBar({ getBatch: () => ({ index: 0, total: 1, name: 'a.png' }), getBatchCsv: () => true, onBatchCsv() {} });
  assert.ok(!single.some((e) => e.className === 'batch-csv-opt'));
  const noCb = await renderBar({ getBatch: () => ({ index: 0, total: 3, name: 'a.png' }) });
  assert.ok(!noCb.some((e) => e.className === 'batch-csv-opt'));
});

test('p8-bc-22. objects CSV: trailing "# truncated,<kept>,<dropped>" when rows were dropped by the batch cap OR by the op own cap (total > rows) · absent otherwise', () => {
  const b = createBatchCsv({ maxObjectRows: 3 });
  b.add('a.png', [ent('regionprops', 1, tb(['i'], [[1], [2]], 2))]);
  b.add('b.png', [ent('regionprops', 1, tb(['i'], [[1], [2]], 2))]); // เพดานชุดตัด 1 แถว
  const t = b.build('s').files[1].text;
  assert.ok(t.endsWith('\r\n# truncated,3,1\r\n'), t);
  const o = createBatchCsv(); // op ตัดเอง: total 5 แต่ส่งมา 2 แถว
  o.add('a.png', [ent('regionprops', 1, tb(['i'], [[1], [2]], 5))]);
  assert.ok(o.build('s').files[1].text.endsWith('\r\n# truncated,2,3\r\n'));
  const n = createBatchCsv(); n.add('a.png', [ent('regionprops', 1, tb(['i'], [[1]], 1))]);
  assert.ok(!n.build('s').files[1].text.includes('truncated'));
});

test('p8-bc-23. column order: grouped by box then key (not first-seen across images)', () => {
  const b = createBatchCsv();
  b.add('a.png', [ent('focus_score', 1, sc(['a', 1]))]); // ภาพแรกมีแต่กล่อง 1
  b.add('b.png', [ent('focus_score', 1, sc(['a', 1], ['b', 2])), ent('roi_polygon', 2, sc(['m', 3]))]);
  b.add('c.png', [ent('focus_score', 1, sc(['a', 1], ['b', 2]))]);
  assert.equal(lines(b.build('s').files[0].text)[0], 'file,focus_score.a,focus_score.b,roi_polygon.m');
  // กล่อง 2 เจอก่อนใน "ภาพ" แต่กล่อง 1 อยู่ลำดับแรกของชุด → กล่อง 1 ก่อนเสมอ
  const c = createBatchCsv();
  c.add('a.png', [ent('focus_score', 1, sc(['a', 1]))]);
  c.add('b.png', [ent('roi_polygon', 2, sc(['m', 3])), ent('focus_score', 1, sc(['z', 9]))]);
  assert.equal(lines(c.build('s').files[0].text)[0], 'file,focus_score.a,focus_score.z,roi_polygon.m');
});

test('p8-bc-24. app.js: a failed image save is counted + csv error row and the loop continues; finishBatchCsv still runs after an unexpected throw; status says partial; confirm text does not promise 2 CSV files', () => {
  const app = readSrc('app.js');
  assert.match(app, /csv\.addError\(rec\.name, "บันทึกภาพไม่สำเร็จ"\)/);
  assert.match(app, /catch \{\s*saveFailed = true;\s*failed\+\+;/);
  assert.match(app, /catch \{\s*aborted = true;[^\n]*\n\s*\}\s*\/\/ csv = null[\s\S]*?await finishBatchCsv\(csv/);
  assert.match(app, /ผลบางส่วน/);
  assert.ok(!/\(csvOn \? 2 : 0\)/.test(app));
  assert.match(app, /\+CSV ถ้ามีค่าที่วัด/);
});

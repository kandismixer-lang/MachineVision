// RF-DETR ไม่ฝังชื่อคลาสใน .onnx → เซลล์ RFDETR_NAMES_BLOCK ใน notebook ฝัง metadata "names" ให้ แล้วแอปอ่านด้วย extractOnnxNames
// เทสต์ end-to-end: สร้าง .onnx เล็ก ๆ ที่มี output แบบ RF-DETR (dets [1,300,4] + labels [1,300,N+1]) → รันเซลล์จริงด้วย Python → อ่านชื่อด้วย JS
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { extractOnnxNames } from '../src/renderer/dl-labels.js';
// export-dialog.js import app.js (DOM/CSS) — ตัดบรรทัด import ทิ้งแบบเดียวกับ training-code.test.mjs
const dialogSrc = fs.readFileSync(new URL('../src/renderer/ui/export-dialog.js', import.meta.url), 'utf8').replace(/^import .*;\r?$/gm, '');
const { RFDETR_NAMES_BLOCK, buildNotebook, buildTrainingCode } = await import(`data:text/javascript;base64,${Buffer.from(dialogSrc).toString('base64')}`);

let onnxPy = false;
try { execFileSync('python', ['-c', 'import onnx, yaml'], { stdio: 'pipe' }); onnxPy = true; } catch {}

const enc = (s) => new TextEncoder().encode(s);

test('extractOnnxNames: ultralytics-style dict, quotes, Thai, escapes', () => {
  assert.deepEqual(extractOnnxNames(enc('\x0a\x05names\x12\x2a{0: \'person\', 1: "it\'s", 2: \'รอยร้าว\', 3: \'a\\\\b\'}')), ['person', "it's", 'รอยร้าว', 'a\\b']);
});

test('extractOnnxNames: skips an earlier "names" that is not followed by a dict, finds the metadata one', () => {
  const bytes = enc('graph node class_names_concat {weird} ...... \x0a\x05names\x12\x10{0: \'Defect\'}');
  assert.deepEqual(extractOnnxNames(bytes), ['Defect']);
});

test('extractOnnxNames: no metadata → null', () => {
  assert.equal(extractOnnxNames(enc('dets labels output0')), null);
  assert.equal(extractOnnxNames(enc('names\x12\x02{}')), null);
});

test('RF-DETR notebook + copy-paste code include the names cell after export', () => {
  const nb = JSON.parse(buildNotebook({ framework: 'rfdetr', epochs: 1, batch: 2, rfModel: 'RFDETRNano' }));
  const src = nb.cells.map((c) => (Array.isArray(c.source) ? c.source.join('') : c.source));
  const exp = src.findIndex((s) => s.includes('model.export()'));
  const names = src.findIndex((s) => s.includes('metadata_props.add(key="names"'));
  assert.ok(exp >= 0 && names === exp + 1);
  const code = buildTrainingCode({ framework: 'rfdetr', epochs: 1, batch: 2, rfModel: 'RFDETRNano' });
  assert.ok(code.indexOf('model.export()') < code.indexOf('metadata_props.add(key="names"'));
  const yolo = buildNotebook({ framework: 'yolo', epochs: 1, batch: 2, model: 'yolo11n' });
  assert.ok(!yolo.includes('metadata_props'));
});

// สร้าง .onnx ปลอม + dataset แล้วรันเซลล์จริง (แทน DATASET และไม่มี google.colab → ข้ามการดาวน์โหลด)
function runNamesCell(dir, { slots, dataset }) {
  const ds = path.join(dir, 'dataset');
  fs.mkdirSync(path.join(ds, 'train'), { recursive: true });
  if (dataset.coco) fs.writeFileSync(path.join(ds, 'train', '_annotations.coco.json'), JSON.stringify(dataset.coco));
  if (dataset.yaml) fs.writeFileSync(path.join(ds, 'data.yaml'), dataset.yaml);
  fs.mkdirSync(path.join(dir, 'output'), { recursive: true });
  const mk = [
    'import onnx, sys',
    'from onnx import helper, TensorProto',
    `n = ${slots + 1}`,
    'x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])',
    'd = helper.make_tensor_value_info("dets", TensorProto.FLOAT, [1, 300, 4])',
    'l = helper.make_tensor_value_info("labels", TensorProto.FLOAT, [1, 300, n])',
    'z4 = helper.make_tensor("z4", TensorProto.FLOAT, [1, 300, 4], [0.0] * 1200)',
    'zn = helper.make_tensor("zn", TensorProto.FLOAT, [1, 300, n], [0.0] * 300 * n)',
    'g = helper.make_graph([helper.make_node("Identity", ["z4"], ["dets"]), helper.make_node("Identity", ["zn"], ["labels"])], "g", [x], [d, l], [z4, zn])',
    'onnx.save(helper.make_model(g), sys.argv[1])',
  ].join('\n');
  execFileSync('python', ['-c', mk, path.join(dir, 'output', 'inference_model.onnx')], { stdio: 'pipe' });
  const cell = RFDETR_NAMES_BLOCK.replace('DATASET = "/content/dataset"', `DATASET = ${JSON.stringify(ds)}`);
  assert.notEqual(cell, RFDETR_NAMES_BLOCK);
  const log = execFileSync('python', ['-c', cell], { cwd: dir, encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
  return { names: extractOnnxNames(fs.readFileSync(path.join(dir, 'output', 'inference_model.onnx'))), log };
}

const withDir = (fn) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rf-names-'));
  try { return fn(dir); } finally { fs.rmSync(dir, { recursive: true, force: true }); }
};
const ann = (...ids) => ids.map((category_id, i) => ({ id: i + 1, image_id: 0, category_id, bbox: [0, 0, 1, 1] }));

test('cell: COCO exported by the app (root "objects") → names embedded, app reads them', { skip: !onnxPy }, () => withDir((dir) => {
  const coco = { categories: [{ id: 0, name: 'objects', supercategory: 'none' }, { id: 1, name: 'Defect', supercategory: 'objects' }, { id: 2, name: 'รอยร้าว', supercategory: 'objects' }], annotations: ann(1, 2) };
  assert.deepEqual(runNamesCell(dir, { slots: 2, dataset: { coco } }).names, ['Defect', 'รอยร้าว']);
}));

test('cell: Roboflow COCO with a project-named root → root dropped', { skip: !onnxPy }, () => withDir((dir) => {
  const coco = { categories: [{ id: 0, name: 'pcb-qc', supercategory: 'none' }, { id: 1, name: 'short', supercategory: 'pcb-qc' }, { id: 2, name: "it's open", supercategory: 'pcb-qc' }], annotations: ann(1, 2) };
  assert.deepEqual(runNamesCell(dir, { slots: 2, dataset: { coco } }).names, ['short', "it's open"]);
}));

test('cell: model kept the root as a class (older rfdetr) → slot count picks the full list', { skip: !onnxPy }, () => withDir((dir) => {
  const coco = { categories: [{ id: 0, name: 'pcb-qc', supercategory: 'none' }, { id: 1, name: 'short', supercategory: 'pcb-qc' }], annotations: ann(1) };
  assert.deepEqual(runNamesCell(dir, { slots: 2, dataset: { coco } }).names, ['pcb-qc', 'short']);
}));

test('cell: YOLO dataset.zip (data.yaml) → names from yaml', { skip: !onnxPy }, () => withDir((dir) => {
  assert.deepEqual(runNamesCell(dir, { slots: 2, dataset: { yaml: 'nc: 2\nnames: ["ok", "ng"]\n' } }).names, ['ok', 'ng']);
}));

test('cell: count mismatch → warns and pads so every slot has a name', { skip: !onnxPy }, () => withDir((dir) => {
  const r = runNamesCell(dir, { slots: 3, dataset: { yaml: 'names: [a]\n' } });
  assert.deepEqual(r.names, ['a', 'class_1', 'class_2']);
  assert.match(r.log, /⚠️/);
}));

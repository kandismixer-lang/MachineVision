// ชื่อคลาสจาก COCO json ต้องเรียงตรงกับ index ที่ RF-DETR ใช้ตอนเทรน (rfdetr filter_parent_categories + cat2label)
// ก่อนแก้: แอปตัดรากทิ้งเฉพาะ id 0 ที่ชื่อ objects/none/background → dataset จาก Roboflow ที่รากชื่อตามโปรเจกต์ ชื่อเลื่อนไป 1 ช่อง
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { cocoClassNames, parseLabels } from '../src/renderer/dl-labels.js';

let rfdetr = false;
try { execFileSync('python', ['-c', 'from rfdetr.datasets.coco import filter_parent_categories'], { stdio: 'pipe' }); rfdetr = true; } catch {}

const ann = (...ids) => ids.map((category_id, i) => ({ id: i + 1, image_id: 0, category_id, bbox: [0, 0, 1, 1] }));

const FIXTURES = {
  // รูปแบบที่ Argus export (export-coco.js) — ตรงกับ dataset จริงใน Testpcb
  argus: { categories: [{ id: 0, name: 'objects', supercategory: 'none' }, { id: 1, name: 'Defect', supercategory: 'objects' }], annotations: ann(1, 1) },
  // Roboflow.com: รากชื่อตามโปรเจกต์ (ตัวอย่างใน docstring ของ rfdetr)
  roboflowProject: {
    categories: [{ id: 0, name: 'eggmasses', supercategory: 'none' }, { id: 1, name: 'stake', supercategory: 'eggmasses' }, { id: 2, name: 'tree', supercategory: 'eggmasses' }],
    annotations: ann(1, 2),
  },
  // รากที่มี annotation จริง = คลาสจริง ไม่ตัด
  annotatedRoot: {
    categories: [{ id: 0, name: 'eggmasses', supercategory: 'none' }, { id: 1, name: 'stake', supercategory: 'eggmasses' }, { id: 2, name: 'tree', supercategory: 'eggmasses' }],
    annotations: ann(0, 1, 2),
  },
  // COCO ต้นฉบับ (id กระโดด, supercategory เป็นกลุ่มที่ไม่ใช่ชื่อคลาส, person เป็นแม่ของตัวเอง) — ใส่สลับลำดับ
  cocoOriginal: {
    categories: [
      { id: 18, name: 'dog', supercategory: 'animal' }, { id: 1, name: 'person', supercategory: 'person' },
      { id: 3, name: 'car', supercategory: 'vehicle' }, { id: 2, name: 'bicycle', supercategory: 'vehicle' },
    ],
    annotations: ann(1, 2, 3, 18),
  },
  // id เป็นสตริง + ไม่มี supercategory
  stringIds: { categories: [{ id: '2', name: 'b' }, { id: '1', name: 'a' }], annotations: ann('1') },
};

test('Argus export: root "objects" dropped → Defect is class 0 (matches the trained model: Defect fires on channel 0)', () => {
  assert.deepEqual(cocoClassNames(FIXTURES.argus), ['Defect']);
});

test('Roboflow project-named root is dropped (old code kept it and shifted every label by one)', () => {
  assert.deepEqual(cocoClassNames(FIXTURES.roboflowProject), ['stake', 'tree']);
});

test('a root that has annotations is a real class and is kept', () => {
  assert.deepEqual(cocoClassNames(FIXTURES.annotatedRoot), ['eggmasses', 'stake', 'tree']);
});

test('original COCO: nothing dropped, sorted by id (sparse ids → contiguous)', () => {
  assert.deepEqual(cocoClassNames(FIXTURES.cocoOriginal), ['person', 'bicycle', 'car', 'dog']);
});

test('string ids and missing supercategory', () => {
  assert.deepEqual(cocoClassNames(FIXTURES.stringIds), ['a', 'b']);
});

test('parseLabels: json / yaml / txt / broken json', () => {
  assert.deepEqual(parseLabels(JSON.stringify(FIXTURES.roboflowProject), '_annotations.coco.json'), ['stake', 'tree']);
  assert.deepEqual(parseLabels('names: [ok, ng]\n', 'data.yaml'), ['ok', 'ng']);
  assert.deepEqual(parseLabels('path: x\nnames:\n  0: ok\n  1: "ng"\nnc: 2\n', 'data.yml'), ['ok', 'ng']);
  assert.deepEqual(parseLabels('ok\r\n\r\nng\n', 'classes.txt'), ['ok', 'ng']);
  assert.deepEqual(parseLabels('{oops', 'a.json'), []);
});

test('same order as rfdetr itself (filter_parent_categories + annotated ids)', { skip: !rfdetr }, () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dl-labels-'));
  try {
    const inp = path.join(dir, 'in.json');
    fs.writeFileSync(inp, JSON.stringify(FIXTURES));
    const py = [
      'import json, sys',
      'from rfdetr.datasets.coco import filter_parent_categories, annotated_category_ids',
      'fx = json.load(open(sys.argv[1], encoding="utf-8"))',
      'print(json.dumps({k: [c["name"] for c in filter_parent_categories(v["categories"], annotated_category_ids(v))] for k, v in fx.items()}))',
    ].join('\n');
    const out = JSON.parse(execFileSync('python', ['-c', py, inp], { encoding: 'utf8' }));
    for (const [k, v] of Object.entries(FIXTURES)) assert.deepEqual(cocoClassNames(v), out[k], k);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

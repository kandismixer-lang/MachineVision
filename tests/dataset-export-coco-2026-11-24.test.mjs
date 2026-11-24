// Export Dataset → รูปแบบ COCO (เปิดกลับมา 2026-11-24): โครงแบบ Roboflow ที่ rfdetr อ่านได้ + ชื่อคลาสตรง index ของโมเดล
import assert from 'node:assert/strict';
import { after, beforeEach, test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import JSZip from 'jszip';

const storageDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: () => null, setItem: () => {} } });
const annotations = await import('../src/renderer/dataset/annotations.js');
const { exportCocoDataset } = await import('../src/renderer/dataset/export-coco.js');
const { cocoClassNames } = await import('../src/renderer/dl-labels.js');

let rfdetr = false;
try { execFileSync('python', ['-c', 'from rfdetr.datasets.coco import filter_parent_categories'], { stdio: 'pipe' }); rfdetr = true; } catch {}

beforeEach(() => {
  while (annotations.getClasses().length) annotations.removeClass(0);
  annotations.addClass('part');
  annotations.addClass('รอยร้าว');
});
after(() => {
  annotations.save();
  if (storageDescriptor) Object.defineProperty(globalThis, 'localStorage', storageDescriptor);
  else delete globalThis.localStorage;
});

const box = { type: 'bbox', classIdx: 1, cx: 0.3, cy: 0.4, w: 0.2, h: 0.6 };
const polygon = { type: 'polygon', classIdx: 0, points: [{ x: 0.1, y: 0.2 }, { x: 0.7, y: 0.3 }, { x: 0.4, y: 0.8 }] };
const line = { type: 'line', classIdx: 0, points: [{ x: 0.1, y: 0.2 }, { x: 0.8, y: 0.9 }] };

function entry(key, labels, split = 'train') {
  annotations.setAnnotations(key, labels);
  annotations.setSplit(key, split);
  return { key, getCanvas: async () => ({ width: 800, height: 400, toBlob: (cb, mime) => cb(new Blob([key], { type: mime })) }) };
}

async function exportZip(entries) {
  const { blob, stats } = await exportCocoDataset(entries);
  return { zip: await JSZip.loadAsync(await blob.arrayBuffer()), stats };
}
const json = async (zip, f) => JSON.parse(await zip.file(f).async('string'));

test('Roboflow layout: <split>/_annotations.coco.json + images, only for splits that have images', async () => {
  const { zip, stats } = await exportZip([entry('a.png', [box, polygon, line]), entry('b.png', [box], 'valid'), entry('empty.png', [])]);
  assert.ok(zip.file('train/_annotations.coco.json') && zip.file('train/a.jpg'));
  assert.ok(zip.file('valid/_annotations.coco.json') && zip.file('valid/b.jpg'));
  assert.equal(zip.file('test/_annotations.coco.json'), null);
  assert.deepEqual(stats.perSplit, { train: 1, valid: 1, test: 0 });
  assert.equal(stats.skipped, 1);
  assert.equal(stats.skippedLines, 1);
});

test('categories: root "objects" + classes 1..N · names read back in model order', async () => {
  const { zip } = await exportZip([entry('a.png', [box, polygon])]);
  const d = await json(zip, 'train/_annotations.coco.json');
  assert.deepEqual(d.categories.map((c) => [c.id, c.name]), [[0, 'objects'], [1, 'part'], [2, 'รอยร้าว']]);
  assert.deepEqual(cocoClassNames(d), ['part', 'รอยร้าว']);
  const [b, p] = d.annotations;
  assert.equal(b.category_id, 2);
  assert.deepEqual(b.bbox, [160, 40, 160, 240]);
  assert.equal(p.category_id, 1);
  assert.equal(p.segmentation[0].length, 6);
  assert.deepEqual(d.images, [{ id: 0, file_name: 'a.jpg', width: 800, height: 400 }]);
});

test('rfdetr itself maps the exported categories to labels 0..N-1 (part=0, รอยร้าว=1)', { skip: !rfdetr }, async () => {
  const { zip } = await exportZip([entry('a.png', [box, polygon])]);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'coco-exp-'));
  try {
    fs.mkdirSync(path.join(dir, 'train'));
    fs.writeFileSync(path.join(dir, 'train', '_annotations.coco.json'), await zip.file('train/_annotations.coco.json').async('string'));
    const py = [
      'import sys, json',
      'from rfdetr.datasets import detect_roboflow_format',
      'from rfdetr.datasets.coco import filter_parent_categories, annotated_category_ids',
      'from pathlib import Path',
      'root = Path(sys.argv[1])',
      'd = json.load(open(root / "train" / "_annotations.coco.json", encoding="utf-8"))',
      '# สูตรเดียวกับ CocoDetection(remap_category_ids=True) / _train_split_cat2label ของ rfdetr',
      'kept = filter_parent_categories(d["categories"], annotated_category_ids(d))',
      'print(json.dumps({"format": detect_roboflow_format(root), "cat2label": {str(c["id"]): i for i, c in enumerate(kept)}}))',
    ].join('\n');
    const out = JSON.parse(execFileSync('python', ['-c', py, dir], { encoding: 'utf8' }).trim().split('\n').pop());
    assert.equal(out.format, 'coco');
    assert.deepEqual(out.cat2label, { 1: 0, 2: 1 });
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

import assert from 'node:assert/strict';
import { test } from 'node:test';
import JSZip from 'jszip';
import { importYoloDataset, parseYoloNames } from '../src/renderer/dataset/import-yolo.js';

test('Roboflow names lists and indexed YAML maps', () => {
  for (const yaml of ["names: ['part', 'defect']\nnc: 2", 'names:\n  - part\n  - defect\nnc: 2', 'names:\n  0: part\n  1: defect', 'names: {0: part, 1: defect}']) {
    assert.deepEqual(parseYoloNames(yaml), ['part', 'defect']);
  }
  assert.throws(() => parseYoloNames('names: [part]\nnc: 2'), /nc/);
});

async function archive(label = '0 0.5 0.5 0.2 0.4') {
  const zip = new JSZip();
  zip.file('wrapped/data.yaml', 'names: [part]\nnc: 1\ntrain: ../train/images');
  zip.file('wrapped/train/images/part.jpg', 'pixels');
  zip.file('wrapped/train/labels/part.txt', label);
  zip.file('wrapped/valid/images/part.jpg', 'other pixels');
  zip.file('wrapped/valid/labels/part.txt', '');
  return zip.generateAsync({ type: 'uint8array' });
}
test('ZIP preserves split/labels and deduplicates image names', async () => {
  const result = await importYoloDataset(await archive());
  assert.equal(result.images.length, 2);
  assert.deepEqual(result.images.map(i => i.split), ['train', 'valid']);
  assert.notEqual(result.images[0].key, result.images[1].key);
  assert.equal(result.images[0].annotations[0].w, 0.2);
  assert.equal(result.images[1].annotations.length, 0);
  assert.equal(result.files[0].type, 'image/jpeg');
});
test('polygon import retains points', async () => {
  const result = await importYoloDataset(await archive('0 0.1 0.1 0.8 0.1 0.8 0.9'));
  assert.equal(result.images[0].annotations[0].type, 'polygon');
  assert.equal(result.images[0].annotations[0].points.length, 3);
});
test('invalid label aborts before app state mutation', async () => {
  await assert.rejects(importYoloDataset(await archive('3 0.5 0.5 0.2 0.4')), /class ID/);
  await assert.rejects(importYoloDataset(await archive('0 NaN 0.5 0.2 0.4')), /coordinate/);
});

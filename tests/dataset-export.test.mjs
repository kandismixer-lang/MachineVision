import assert from 'node:assert/strict';
import { after, beforeEach, test } from 'node:test';
import path from 'node:path';
import JSZip from 'jszip';

// Exercise the real annotation store without reading or writing browser storage.
const storageDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: { getItem: () => null, setItem: () => {} },
});
const annotations = await import('../src/renderer/dataset/annotations.js');
const { exportYoloDataset } = await import('../src/renderer/dataset/export-yolo.js');

beforeEach(() => {
  while (annotations.getClasses().length) annotations.removeClass(0);
  annotations.addClass('part');
  annotations.addClass('defect, "small"');
});
after(() => {
  annotations.save(); // Flush the store's pending debounce into the memory stub.
  if (storageDescriptor) Object.defineProperty(globalThis, 'localStorage', storageDescriptor);
  else delete globalThis.localStorage;
});

const box = { type: 'bbox', classIdx: 1, cx: 0.3, cy: 0.4, w: 0.2, h: 0.6 };
const polygon = {
  type: 'polygon', classIdx: 0,
  points: [{ x: 0.1, y: 0.2 }, { x: 0.7, y: 0.3 }, { x: 0.4, y: 0.8 }],
};
const line = { type: 'line', classIdx: 0, points: [{ x: 0.1, y: 0.2 }, { x: 0.8, y: 0.9 }] };

function entry(key, labels, split = 'train', payload = key) {
  annotations.setAnnotations(key, labels);
  annotations.setSplit(key, split);
  return {
    key,
    getCanvas: async () => ({
      width: 800,
      height: 400,
      toBlob(callback, mime, quality) {
        assert.equal(mime, 'image/jpeg');
        assert.ok(quality > 0 && quality <= 1);
        // Distinct bytes expose accidental overwrites; no image decoder is needed.
        callback(new Blob([payload], { type: mime }));
      },
    }),
  };
}

async function exportZip(entries) {
  const { blob, stats } = await exportYoloDataset(entries);
  assert.ok(blob instanceof Blob);
  return { zip: await JSZip.loadAsync(await blob.arrayBuffer()), stats };
}

async function text(zip, filename) {
  const file = zip.file(filename);
  assert.ok(file, `Missing ZIP entry: ${filename}`);
  return file.async('string');
}

test('polygon exports an enclosing detection box and preserves source points', async () => {
  const image = entry('polygon.jpg', [polygon]);
  const before = annotations.getAnnotations(image.key);
  const { zip, stats } = await exportZip([image]);
  assert.equal(stats.task, 'detect');
  assert.equal(await text(zip, 'train/labels/polygon.txt'), '0 0.400000 0.500000 0.600000 0.600000\n');
  assert.deepEqual(annotations.getAnnotations(image.key), before);
  assert.equal(await text(zip, 'train/images/polygon.jpg'), 'polygon.jpg');
});

test('mixed boxes, polygons and lines stay detection labels with lines omitted', async () => {
  const { zip, stats } = await exportZip([entry('mixed.png', [box, polygon, line])]);
  assert.equal(await text(zip, 'train/labels/mixed.txt'),
    '1 0.300000 0.400000 0.200000 0.600000\n0 0.400000 0.500000 0.600000 0.600000\n');
  assert.equal(stats.task, 'detect');
  assert.equal(stats.boxes, 2);
  assert.equal(stats.skippedLines, 1);
  assert.equal(stats.exported, 1);
});

test('line-only images are omitted without requesting their canvas', async () => {
  const skipped = entry('line-only.png', [line]);
  skipped.getCanvas = async () => assert.fail('Line-only image should not be decoded');
  const { zip, stats } = await exportZip([skipped, entry('box.jpg', [box])]);
  assert.equal(zip.file('train/images/line-only.jpg'), null);
  assert.equal(zip.file('train/labels/line-only.txt'), null);
  assert.equal(stats.skipped, 1);
  assert.equal(stats.skippedLines, 1);
  assert.equal(stats.exported, 1);
  await assert.rejects(exportYoloDataset([skipped]), /ไม่มีภาพ/);
});

test('colliding basenames and existing suffixes preserve every image-label pair', async () => {
  const { zip, stats } = await exportZip([
    entry('photo.jpg', [box], 'train', 'first'),
    entry('photo_2.jpg', [polygon], 'train', 'existing suffix'),
    entry('photo.png', [polygon], 'train', 'same basename'),
    entry('photo.jpg', [box], 'train', 'same filename'),
  ]);
  const expected = [
    ['photo', 'first', '1 0.300000 0.400000 0.200000 0.600000\n'],
    ['photo_2', 'existing suffix', '0 0.400000 0.500000 0.600000 0.600000\n'],
    ['photo_3', 'same basename', '0 0.400000 0.500000 0.600000 0.600000\n'],
    ['photo_4', 'same filename', '1 0.300000 0.400000 0.200000 0.600000\n'],
  ];
  for (const [base, bytes, label] of expected) {
    assert.equal(await text(zip, `train/images/${base}.jpg`), bytes);
    assert.equal(await text(zip, `train/labels/${base}.txt`), label);
  }
  assert.equal(Object.values(zip.files).filter(file => !file.dir).length, 9);
  assert.equal(stats.exported, 4);
});

test('standard split layout and YAML paths remain portable with escaped class names', async () => {
  const { zip, stats } = await exportZip([
    entry('shared.jpg', [box], 'train'),
    entry('shared.png', [polygon], 'valid'),
    entry('test.jpg', [box], 'test'),
  ]);
  const yaml = await text(zip, 'data.yaml');
  assert.equal(yaml, 'train: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames: ["part", "defect, \\"small\\""]\n'.replaceAll('\\\\', '\\'));
  assert.doesNotMatch(yaml, /^path:|\/content\/|[A-Za-z]:[\\/]/m);
  for (const [, relative] of yaml.matchAll(/^(?:train|val|test): (.+)$/gm)) {
    assert.equal(path.posix.isAbsolute(relative), false);
    assert.ok(Object.keys(zip.files).some(file => file.startsWith(`${relative}/`) && file.endsWith('.jpg')));
  }
  assert.ok(zip.file('train/images/shared.jpg'));
  assert.ok(zip.file('valid/images/shared.jpg')); // Names are scoped to each split.
  assert.ok(zip.file('valid/labels/shared.txt'));
  assert.deepEqual(stats.perSplit, { train: 1, valid: 1, test: 1 });
});

test('train-only YAML points validation at an existing folder and omits absent test', async () => {
  const { zip } = await exportZip([entry('only.jpg', [box])]);
  const yaml = await text(zip, 'data.yaml');
  assert.match(yaml, /^train: train\/images\nval: train\/images\n/);
  assert.doesNotMatch(yaml, /^test:|^path:/m);
  assert.ok(zip.file('train/images/only.jpg'));
});

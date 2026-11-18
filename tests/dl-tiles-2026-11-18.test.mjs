import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { tilePositions, tileGrid, mergeTileBoxes } from '../src/renderer/dl-tiles.js';

test('tilePositions: step = tile×(1−overlap), last tile flush with the edge', () => {
  assert.deepEqual(tilePositions(1920, 320, 0.2), [0, 256, 512, 768, 1024, 1280, 1536, 1600]);
  assert.deepEqual(tilePositions(1004, 320, 0.2), [0, 256, 512, 684]);
  assert.deepEqual(tilePositions(1000, 500, 0), [0, 500]); // exact fit: no duplicate last tile
});

test('tilePositions: image not larger than the tile → one tile at 0', () => {
  assert.deepEqual(tilePositions(300, 320, 0.2), [0]);
  assert.deepEqual(tilePositions(320, 320, 0.2), [0]);
});

test('tilePositions: overlap clamped, never a zero step', () => {
  const p = tilePositions(100, 10, 5); // overlap 5 → clamp 0.9 → step 1
  assert.equal(p[1] - p[0], 1);
  assert.equal(p.at(-1), 90);
  assert.deepEqual(tilePositions(30, 10, -1), [0, 10, 20]);
});

test('tileGrid: covers the whole image, every tile inside it', () => {
  const W = 1920, H = 1004;
  const rects = tileGrid(W, H, 320, 0.2);
  assert.equal(rects.length, 32);
  for (const r of rects) {
    assert.ok(r.x >= 0 && r.y >= 0 && r.x + r.w <= W && r.y + r.h <= H);
    assert.equal(r.w, 320);
    assert.equal(r.h, 320);
  }
  assert.ok(rects.some((r) => r.x + r.w === W && r.y + r.h === H));
});

test('tileGrid: side shorter than the tile uses the image length', () => {
  assert.deepEqual(tileGrid(800, 200, 320, 0.2).map((r) => [r.w, r.h]), [[320, 200], [320, 200], [320, 200]]);
  assert.deepEqual(tileGrid(100, 80, 640, 0.2), [{ x: 0, y: 0, w: 100, h: 80 }]);
});

test('mergeTileBoxes: half box cut by a tile edge is dropped (IoU alone would keep it)', () => {
  const full = { x: 100, y: 100, w: 30, h: 30, score: 0.9, cls: 0 };
  const half = { x: 100, y: 100, w: 12, h: 30, score: 0.6, cls: 0 }; // IoU 0.4, IoS 1.0
  assert.deepEqual(mergeTileBoxes([half, full]), [full]);
});

test('mergeTileBoxes: keeps separate objects and other classes, highest score first', () => {
  const a = { x: 0, y: 0, w: 20, h: 20, score: 0.5, cls: 0 };
  const b = { x: 200, y: 0, w: 20, h: 20, score: 0.8, cls: 0 };
  const c = { x: 2, y: 2, w: 18, h: 18, score: 0.7, cls: 1 }; // overlaps a but another class
  assert.deepEqual(mergeTileBoxes([a, b, c]), [b, c, a]);
});

test('mergeTileBoxes: duplicate from the overlap strip keeps the higher score', () => {
  const lo = { x: 300, y: 50, w: 25, h: 25, score: 0.7, cls: 0 };
  const hi = { x: 302, y: 51, w: 24, h: 25, score: 0.92, cls: 0 };
  assert.deepEqual(mergeTileBoxes([lo, hi]), [hi]);
});

test('app wiring: Model Detect uses detectTiled when the tile switch is on', async () => {
  const app = await readFile(new URL('../src/renderer/app.js', import.meta.url), 'utf8');
  assert.match(app, /t\.on \? dlDetectTiled\(src, w, h, \{ \.\.\.detOpts, tile: t\.size, overlap: t\.overlap \/ 100 \}\)/);
  const yolo = await readFile(new URL('../src/renderer/dl-yolo.js', import.meta.url), 'utf8');
  assert.match(yolo, /export function detectTiled\(/);
  // tile offsets are added back so boxes land on full-image coordinates
  assert.match(yolo, /x: b\.x \+ r\.x, y: b\.y \+ r\.y/);
});

test('DL result canvas resizes when only the height changes (no strip of the previous image)', async () => {
  const app = await readFile(new URL('../src/renderer/app.js', import.meta.url), 'utf8');
  assert.match(app, /if \(oc\.width !== w \|\| oc\.height !== h\)/);
  assert.match(app, /if \(cell\.canvas\.width !== w \|\| cell\.canvas\.height !== h\)/);
});

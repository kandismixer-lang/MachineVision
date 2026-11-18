import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { decodeOutputs, kindFromOutputRanks } from '../src/renderer/dl-decode.js';

test('kindFromOutputRanks: 1 output = yolo, 2 with a 4-D proto = yoloseg, 2 of 3-D = rfdetr', () => {
  assert.equal(kindFromOutputRanks([3]), 'yolo');
  assert.equal(kindFromOutputRanks([3, 4]), 'yoloseg');
  assert.equal(kindFromOutputRanks([3, 3]), 'rfdetr');
});

test('decodeOutputs rfdetr: picks the last-dim-4 output as boxes and scales to the original size', () => {
  const logit = (p) => Math.log(p / (1 - p));
  const out = {
    labels: { dims: [1, 2, 2], data: Float32Array.from([logit(0.9), logit(0.1), logit(0.05), logit(0.1)]) },
    dets: { dims: [1, 2, 4], data: Float32Array.from([0.5, 0.25, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1]) },
  };
  const boxes = decodeOutputs('rfdetr', ['labels', 'dets'], out, { origW: 640, origH: 400, conf: 0.25 }, 384);
  assert.equal(boxes.length, 1);
  const b = boxes[0];
  assert.equal(b.cls, 0);
  assert.ok(Math.abs(b.score - 0.9) < 1e-6);
  assert.ok(Math.abs(b.x - (0.5 - 0.05) * 640) < 1e-3 && Math.abs(b.y - (0.25 - 0.1) * 400) < 1e-3);
  assert.ok(Math.abs(b.w - 64) < 1e-3 && Math.abs(b.h - 80) < 1e-3);
});

test('worker and native share one decoder (dl-decode.js), worker no longer has its own copy', async () => {
  const w = await readFile(new URL('../src/renderer/dl-worker.js', import.meta.url), 'utf8');
  assert.match(w, /import \{ guessKind, decodeOutputs \} from "\.\/dl-decode\.js";/);
  assert.doesNotMatch(w, /function decodeRfdetr\(/);
  const y = await readFile(new URL('../src/renderer/dl-yolo.js', import.meta.url), 'utf8');
  assert.match(y, /window\.mvDL\.nativeRun\(data, sizeNow\)/);
  assert.match(y, /decodeOutputs\(kindNow, namesNow, res\.outputs, meta, sizeNow\)/);
});

test('native backend: GPU (DirectML) first then CPU, env override, skipped runs after a new load', async () => {
  const n = await readFile(new URL('../electron/dl-native.js', import.meta.url), 'utf8');
  assert.match(n, /: \["dml", "cpu"\];/);
  assert.match(n, /MV_DL_EP/);
  assert.match(n, /if \(!session \|\| myGen !== gen\) return \{ skipped: true \};/);
  const main = await readFile(new URL('../electron/main.js', import.meta.url), 'utf8');
  assert.match(main, /ipcMain\.handle\("dl:nativeLoad"/);
  assert.match(main, /ipcMain\.handle\("dl:nativeRun"/);
  const pre = await readFile(new URL('../electron/preload.cjs', import.meta.url), 'utf8');
  assert.match(pre, /nativeRun: \(data, size\) => ipcRenderer\.invoke\("dl:nativeRun"/);
});

test('packaging keeps only the win32/x64 native ORT binaries in the Windows build', async () => {
  const pkg = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
  assert.ok(pkg.dependencies['onnxruntime-node']);
  for (const pat of ['linux', 'darwin', 'win32/arm64']) assert.ok(pkg.build.files.some((f) => f.startsWith('!') && f.includes(pat)), pat);
});

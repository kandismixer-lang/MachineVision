import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';

// dl-yolo.js needs a DOM at import time; pull the metadata helpers out verbatim.
const source = await readFile(new URL('../src/renderer/dl-yolo.js', import.meta.url), 'utf8');
const grab = (name) => new Function(`${source.match(new RegExp(`function ${name}\\(bytes\\) \\{[\\s\\S]*?\\n\\}`))[0]}; return ${name};`)();
const extractOnnxNames = grab('extractOnnxNames');
const extractOnnxImgsz = grab('extractOnnxImgsz');

// Mimic an ultralytics .onnx: binary graph bytes, then UTF-8 metadata props written as str(dict).
function fakeOnnx(meta) {
  const graph = Uint8Array.from({ length: 512 }, (_, i) => (i * 37 + 11) & 0xff);
  const tail = new TextEncoder().encode(meta);
  const out = new Uint8Array(graph.length + tail.length);
  out.set(graph);
  out.set(tail, graph.length);
  return out;
}

test('Thai class names survive (metadata is UTF-8, not latin1)', () => {
  const bytes = fakeOnnx(`\x12imgsz\x1a\n[640, 640]\x12\x05names\x1a\x20{0: 'รอยขีด', 1: 'dent'}`);
  assert.deepEqual(extractOnnxNames(bytes), ['รอยขีด', 'dent']);
  assert.equal(extractOnnxImgsz(bytes), 640);
});

test('names quoted with " or containing quotes/braces keep their index', () => {
  const bytes = fakeOnnx(`names\x1a{0: "it's", 1: 'a "b"', 2: 'x}y', 3: 'z\\'q'}`);
  assert.deepEqual(extractOnnxNames(bytes), ["it's", 'a "b"', 'x}y', "z'q"]);
});

test('no names metadata or an empty dict → null', () => {
  assert.equal(extractOnnxNames(fakeOnnx('dets labels')), null);
  assert.equal(extractOnnxNames(fakeOnnx('names {}')), null);
});

// Junctions must survive either pixel alignment of a diagonal X.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import cv from '@techstark/opencv-js';
import { OPS } from '../src/renderer/filters/registry.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import { assertLeakFree, assertWrapperBalance, makeThrowingCv } from './_leak-helpers.mjs';

while (typeof cv.Mat !== 'function') await new Promise((r) => setTimeout(r, 20));

const fixtures = [
  ['X on-pixel', 4, 1, [[20, 20]]],
  ['X between-pixel', 4, 1, [[20, 20]]],
  ['+', 4, 1, [[20, 20]]],
  ['T', 3, 1, [[20, 12]]],
  ['Y', 3, 1, [[20, 20]]],
  ['H', 4, 2, [[12, 20], [28, 20]]],
  ['ring', 0, 0, []],
  ['plain line', 2, 0, []],
  ['thick X', 4, null, null],
];

function fixture(name) {
  const src = new cv.Mat(48, 48, cv.CV_8UC1, new cv.Scalar(0));
  const line = (x1, y1, x2, y2, width = 1) => cv.line(src, new cv.Point(x1, y1), new cv.Point(x2, y2), new cv.Scalar(255), width);
  if (name === 'X on-pixel' || name === 'thick X') {
    const width = name === 'thick X' ? 9 : 1;
    line(6, 6, 34, 34, width); line(34, 6, 6, 34, width);
  } else if (name === 'X between-pixel') {
    // y=x and y=-x+41 cross at (20.5,20.5), leaving a white 2x2 core.
    line(6, 6, 35, 35); line(35, 6, 6, 35);
  } else if (name === '+') {
    line(6, 20, 34, 20); line(20, 6, 20, 34);
  } else if (name === 'T') {
    line(6, 12, 34, 12); line(20, 12, 20, 34);
  } else if (name === 'Y') {
    line(6, 6, 20, 20); line(34, 6, 20, 20); line(20, 20, 20, 35);
  } else if (name === 'H') {
    line(12, 6, 12, 34); line(28, 6, 28, 34); line(12, 20, 28, 20);
  } else if (name === 'ring') {
    // Diamond ring avoids the extra diagonal adjacency at square corners.
    line(20, 5, 35, 20); line(35, 20, 20, 35); line(20, 35, 5, 20); line(5, 20, 20, 5);
  } else if (name === 'plain line') line(6, 20, 35, 20);
  return src;
}

function run(src, skeleton = false, c = cv) {
  const ctx = {};
  const thin = skeleton ? OPS.morphology.apply(c, src, { op: 'skeleton' }, {}) : null;
  try {
    const out = OPS.morphology.apply(c, thin || src, { op: 'endpoints_branchpoints' }, ctx);
    return { out, info: ctx.info };
  } finally { if (thin) thin.delete(); }
}

function counts(info) {
  const match = /^พบจุดปลาย (\d+) จุด · จุดแยก (\d+) จุด$/.exec(info);
  assert.ok(match, info);
  return match.slice(1).map(Number);
}

for (const [name, endpoints, branches, centers] of fixtures) {
  test(`endpoints-X app: ${name}`, (t) => {
    const src = fixture(name);
    try {
      const { out, info } = run(src, name === 'thick X');
      try {
        t.diagnostic(info);
        const got = counts(info);
        assert.equal(got[0], endpoints);
        if (branches === null) assert.ok(got[1] >= 1, 'thick X -> app Skeleton must retain a junction');
        else assert.equal(got[1], branches);
        if (centers) {
          // Compare the whole overlay with independently chosen marker centers.
          const expected = new cv.Mat();
          try {
            cv.cvtColor(src, expected, cv.COLOR_GRAY2RGB);
            const tips = {
              'X on-pixel': [[6, 6], [34, 6], [6, 34], [34, 34]],
              'X between-pixel': [[6, 6], [35, 6], [6, 35], [35, 35]],
              '+': [[20, 6], [6, 20], [34, 20], [20, 34]],
              'T': [[6, 12], [34, 12], [20, 34]],
              'Y': [[6, 6], [34, 6], [20, 35]],
              'H': [[12, 6], [28, 6], [12, 34], [28, 34]],
              'ring': [], 'plain line': [[6, 20], [35, 20]],
            }[name];
            for (const [x, y] of tips) cv.circle(expected, new cv.Point(x, y), 3, new cv.Scalar(0, 255, 0), -1);
            for (const [x, y] of centers) cv.circle(expected, new cv.Point(x, y), 3, new cv.Scalar(255, 0, 0), -1);
            assert.deepEqual(Buffer.from(out.data), Buffer.from(expected.data), 'deterministic overlay centers');
          } finally { expected.delete(); }
        }
      } finally { out.delete(); }
    } finally { src.delete(); }
  });
}

test('endpoints-X empty and all-black retain their dedicated outputs', () => {
  const empty = new cv.Mat(), black = fixture('black');
  try {
    for (const [src, expectedInfo, channels] of [[empty, 'ภาพว่าง', 1], [black, 'ไม่พบเส้น (ลองทำ Skeleton ก่อน)', 3]]) {
      const { out, info } = run(src);
      try {
        assert.equal(info, expectedInfo);
        assert.equal(out.rows, src.rows); assert.equal(out.cols, src.cols);
        assert.equal(out.channels(), channels);
        assert.ok(out.data.every((v) => v === 0));
      } finally { out.delete(); }
    }
  } finally { empty.delete(); black.delete(); }
});

test('endpoints-X border junction and a separate capped blob retain the small junction', () => {
  const border = fixture('black'), mixed = new cv.Mat(96, 160, cv.CV_8UC1, new cv.Scalar(0));
  try {
    cv.line(border, new cv.Point(6, 0), new cv.Point(34, 0), new cv.Scalar(255), 1);
    cv.line(border, new cv.Point(20, 0), new cv.Point(20, 30), new cv.Scalar(255), 1);
    cv.rectangle(mixed, new cv.Point(0, 0), new cv.Point(15, 15), new cv.Scalar(255), -1); // exactly 256 candidates
    cv.line(mixed, new cv.Point(100, 20), new cv.Point(141, 61), new cv.Scalar(255), 1);
    cv.line(mixed, new cv.Point(141, 20), new cv.Point(100, 61), new cv.Scalar(255), 1);
    for (const [src, expected] of [[border, [3, 1]], [mixed, [4, 1]]]) {
      const before = Buffer.from(src.data), { out, info } = run(src);
      try {
        assert.deepEqual(counts(info), expected);
        assert.deepEqual(Buffer.from(src.data), before, 'analysis must not mutate input');
      } finally { out.delete(); }
    }
  } finally { border.delete(); mixed.delete(); }
});

test('endpoints-X large blobs are bounded and leave no native allocations', () => {
  const src = new cv.Mat(64, 64, cv.CV_8UC1, new cv.Scalar(255));
  try {
    assertLeakFree(cv, 'endpoints large candidate cluster', (c) => {
      const { out, info } = run(src, false, c);
      try { assert.deepEqual(counts(info), [0, 0]); } finally { out.delete(); }
    });
    assertWrapperBalance(cv, 'endpoints circle failure', (c) => {
      const x = fixture('X between-pixel');
      try { assert.throws(() => run(x, false, makeThrowingCv(c, 'circle', { onCall: 1 })), /injected test failure/); }
      finally { x.delete(); }
    });
    assertWrapperBalance(cv, 'endpoints labeling failure', (c) => {
      assert.throws(() => run(src, false, makeThrowingCv(c, 'connectedComponentsWithStats')), /injected test failure/);
    });
  } finally { src.delete(); }
});

test('endpoints-X native heap growth during labeling retains the junction', () => {
  const src = fixture('X between-pixel');
  const growing = new Proxy(cv, {
    get(target, key) {
      if (key !== 'connectedComponentsWithStats') return target[key];
      return (...args) => {
        const count = target[key](...args), before = cv.HEAP8.buffer;
        const allocation = cv._malloc(before.byteLength);
        try {
          assert.ok(allocation);
          assert.notEqual(cv.HEAP8.buffer, before, 'exercise real WASM view invalidation');
        } finally { cv._free(allocation); }
        return count;
      };
    },
  });
  try {
    const { out, info } = run(src, false, growing);
    try { assert.deepEqual(counts(info), [4, 1]); } finally { out.delete(); }
  } finally { src.delete(); }
});

test('endpoints-X app timings: sparse 720p and all-white 1080p', (t) => {
  const thin = new cv.Mat(720, 1280, cv.CV_8UC1, new cv.Scalar(0));
  const white = new cv.Mat(1080, 1920, cv.CV_8UC1, new cv.Scalar(255));
  cv.line(thin, new cv.Point(10, 10), new cv.Point(1260, 700), new cv.Scalar(255), 1);
  cv.line(thin, new cv.Point(10, 700), new cv.Point(1260, 10), new cv.Scalar(255), 1);
  try {
    for (const [label, src] of [['thin 1280x720', thin], ['white 1920x1080', white]]) {
      const times = [];
      for (let i = 0; i < 9; i++) {
        const start = performance.now(), { out } = run(src);
        times.push(performance.now() - start); out.delete();
      }
      const measured = times.slice(2).sort((a, b) => a - b);
      t.diagnostic(`${label}: median=${measured[3].toFixed(2)} ms range=${measured[0].toFixed(2)}..${measured[6].toFixed(2)} ms (7 runs, 2 warmups)`);
    }
  } finally { thin.delete(); white.delete(); }
});

const pipe = (skeleton = false) => [...(skeleton ? [{ id: 'morphology', params: { op: 'skeleton' } }] : []), { id: 'morphology', params: { op: 'endpoints_branchpoints' } }];
let python = false, pyAvailable = false;
try {
  execFileSync('python', ['-c', 'pass'], { stdio: 'pipe' }); python = true;
  try { execFileSync('python', ['-c', 'import cv2,numpy'], { stdio: 'pipe' }); pyAvailable = true; } catch {}
} catch {}

test('endpoints-X Python exports compile, including Skeleton chain', { skip: !python && 'python unavailable' }, () => {
  execFileSync('python', ['-c', "import sys,json; [compile(c, 'endpoints_x', 'exec') for c in json.loads(sys.stdin.buffer.read())]"], { input: JSON.stringify([buildPythonCode(pipe()), buildPythonCode(pipe(true))]), stdio: 'pipe' });
});

// Same raw-input / process(path) helper pattern as ip-batch2-A. Execute the
// generated export, then swap BGR -> RGB for the app's overlay convention.
function pythonCase(src, skeleton) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'endpoints-x-'));
  try {
    const input = path.join(dir, 'input.bin'), output = path.join(dir, 'output.bin'), script = path.join(dir, 'case.py');
    const rgb = Buffer.alloc(src.rows * src.cols * 3);
    for (let i = 0; i < src.data.length; i++) rgb.fill(src.data[i], i * 3, i * 3 + 3);
    fs.writeFileSync(input, rgb);
    let code = buildPythonCode(pipe(skeleton));
    assert.ok(code.includes('img = cv2.imread(path)'));
    code = code.replace('img = cv2.imread(path)', `img = np.fromfile(path, dtype=np.uint8).reshape((${src.rows}, ${src.cols}, 3))`);
    const main = code.indexOf('if __name__ == "__main__":'); assert.ok(main > 0);
    code = code.slice(0, main) + '\nimport sys\nprocess(sys.argv[1]).tofile(sys.argv[2])\n';
    fs.writeFileSync(script, code);
    const stdout = execFileSync('python', [script, input, output], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
    const bytes = fs.readFileSync(output);
    for (let i = 0; i < bytes.length; i += 3) [bytes[i], bytes[i + 2]] = [bytes[i + 2], bytes[i]];
    return { stdout, bytes };
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}

for (const [name, endpoints, branches] of [...fixtures, ['black', null, null]]) {
  test(`endpoints-X Python counts and exact pixels: ${name}`, { skip: !pyAvailable && 'python/cv2/numpy unavailable' }, () => {
    const src = fixture(name);
    try {
      const skeleton = name === 'thick X', { out, info } = run(src, skeleton);
      try {
        const py = pythonCase(src, skeleton);
        assert.deepEqual(py.bytes, Buffer.from(out.data));
        if (name === 'black') assert.match(py.stdout, /no line found \(try Skeleton first\)/);
        else {
          const got = /endpoints=(\d+) branchpoints=(\d+)/.exec(py.stdout); assert.ok(got, py.stdout);
          const actual = got.slice(1).map(Number);
          assert.deepEqual(actual, counts(info));
          assert.equal(actual[0], endpoints);
          if (branches === null) assert.ok(actual[1] >= 1);
          else assert.equal(actual[1], branches);
        }
      } finally { out.delete(); }
    } finally { src.delete(); }
  });
}

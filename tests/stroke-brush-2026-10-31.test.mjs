import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { FILTERS, defaultParams, migrateLegacyStep } from "../src/renderer/filters/registry.js";
import {
  STROKE_MAX_POINTS, STROKE_MAX_STROKES, STROKE_MAX_TOTAL, STROKE_Q,
  diskHalfWidths, forEachCenter, normalizeStrokes, pixelToStrokeQ,
  rasterizeStrokes, strokeAspectChanged, strokePixel, strokeRadiusPx,
  summarizeStrokes,
} from "../src/renderer/filters/stroke-mask.js";
import { strokeRasterPyLines, strokesPyLiteral } from "../src/renderer/codegen-strokes.js";
import { createStrokeModel } from "../src/renderer/ui/stroke-model.js";
import { createStrokeCanvas, normalizeStrokeRadiusPreference } from "../src/renderer/ui/stroke-canvas.js";
import { createDetailPanel, formatParamValue } from "../src/renderer/ui/detail-panel.js";

const LABEL_PARAM = {
  key: "strokes", type: "strokes", label: "เส้นขีดบนภาพ", default: null,
  labels: [
    { value: 1, name: "วัตถุ", color: [230, 30, 30], hint: "ลากเส้นทับตัววัตถุ" },
    { value: 2, name: "พื้นหลัง", color: [30, 80, 240], hint: "ลากเส้นบนพื้นหลัง" },
  ],
  eraser: true,
  desc: "ขีดเส้นบนภาพ",
};
const ALLOWED = new Set([0, 1, 2]);

function stroke(label, rr, ...xy) { return [label, rr, ...xy]; }
function value(strokes, w = 100, h = 80) { return { v: 1, w, h, s: strokes }; }
function qAtPixel(x, y, w, h) { return [pixelToStrokeQ(x, w), pixelToStrokeQ(y, h)]; }
function referenceRasterize(w, h, input) {
  const norm = normalizeStrokes(input, ALLOWED);
  if (!norm) return null;
  const out = new Uint8Array(w * h);
  for (const line of norm.s) {
    const radius = strokeRadiusPx(line[1], w, h), halfWidths = diskHalfWidths(radius);
    forEachCenter(line, w, h, (cx, cy) => {
      for (let dy = -radius; dy <= radius; dy++) {
        const y = cy + dy;
        if (y < 0 || y >= h) continue;
        const half = halfWidths[Math.abs(dy)];
        const left = Math.max(0, cx - half), right = Math.min(w - 1, cx + half);
        out.fill(line[0], y * w + left, y * w + right + 1);
      }
    });
  }
  return out;
}

test("sb-off-image-q: extreme stroke coordinates clip disks at both image edges", () => {
  const w = 13, h = 9;
  const labels = rasterizeStrokes(w, h, value([stroke(1, 12, -100000, -100000, 100000, 100000)], w, h), ALLOWED).labels;
  assert.equal(labels[0], 1);
  assert.equal(labels[w * h - 1], 1);
});

function makeRandomStrokes(seed, count, w, h, radius = 12, compact = false, pointCount = 30) {
  let state = seed >>> 0;
  const rand = () => { state = (1664525 * state + 1013904223) >>> 0; return state; };
  const centerX = Math.floor(STROKE_Q / 2), centerY = Math.floor(STROKE_Q / 2);
  return Array.from({ length: count }, (_, i) => {
    const label = i % 3 === 0 ? 1 : i % 3 === 1 ? 2 : 0;
    const coords = [];
    for (let p = 0; p < pointCount; p++) {
      const x = compact ? centerX + (rand() % 31) - 15 : rand() % (STROKE_Q + 1);
      const y = compact ? centerY + (rand() % 31) - 15 : rand() % (STROKE_Q + 1);
      coords.push(x, y);
    }
    return stroke(label, radius, ...coords);
  });
}

test("sb-norm: rejects malformed values and ignores inherited keys", () => {
  for (const bad of [null, undefined, 123, "x", [], { v: 2 }, { v: 1, s: "x" }]) {
    assert.equal(normalizeStrokes(bad), null);
  }
  assert.equal(normalizeStrokes(JSON.parse('{"__proto__":{"v":1,"s":[[1,12,0,0]]}}')), null);
  const inherited = Object.create({ v: 1, s: [[1, 12, 0, 0]] });
  assert.equal(normalizeStrokes(inherited), null);
});

test("sb-norm: rounds, clamps, truncates odd tails and drops invalid or disallowed lines", () => {
  const out = normalizeStrokes(value([
    [1.2, 0.5, -5, 10007, 12.5, 4.1, 999],
    [2, 12, 0, 0, NaN, 4],
    [9, 12, 0, 0],
    [2, 500, 0, Infinity],
    [2, 500, 0, 0, 8, 8],
  ]), ALLOWED);
  assert.deepEqual(out.s, [[1, 1, 0, 10000, 13, 4], [2, 100, 0, 0, 8, 8]]);
  assert.equal(normalizeStrokes(value([[9, 12, 0, 0]]), ALLOWED), null);
  const sparse = [1, 12, 0, 0, , 10];
  assert.equal(normalizeStrokes(value([sparse]), ALLOWED), null);
});

test("sb-norm: enforces stroke and point limits without mutating input", () => {
  const raw = value(Array.from({ length: STROKE_MAX_STROKES + 20 }, (_, i) => stroke(i % 2 + 1, 12, 0, 0)));
  const before = structuredClone(raw);
  const out = normalizeStrokes(raw, new Set([1, 2]));
  assert.equal(out.s.length, STROKE_MAX_STROKES);
  assert.deepEqual(raw, before);

  const pointsPerLine = STROKE_MAX_POINTS + 5;
  const many = Array.from({ length: 11 }, () => [1, 12, ...Array.from({ length: pointsPerLine * 2 }, (_, i) => i % STROKE_Q)]);
  const capped = normalizeStrokes(value(many), new Set([1]));
  assert.equal(capped.s[0].length, 2 + STROKE_MAX_POINTS * 2);
  assert.equal(capped.s.reduce((n, s) => n + (s.length - 2) / 2, 0), STROKE_MAX_TOTAL);
});

test("sb-norm: keeps the partial remainder at the total point cap", () => {
  const line = (points) => [1, 12, ...Array.from({ length: points * 2 }, (_, i) => i % (STROKE_Q + 1))];
  const input = value([
    ...Array.from({ length: 9 }, () => line(1000)),
    line(999), line(4), line(1),
  ]);
  const normalized = normalizeStrokes(input, new Set([1]));
  assert.equal(normalized.s.length, 11);
  assert.equal(normalized.s[10].length, 4);
  assert.equal(normalized.s.reduce((n, item) => n + (item.length - 2) / 2, 0), 10000);
});

test("sb-norm: defaults a missing width to zero", () => {
  assert.deepEqual(normalizeStrokes({ v: 1, h: 80, s: [[1, 12, 0, 0]] }, new Set([1])), {
    v: 1, w: 0, h: 80, s: [[1, 12, 0, 0]],
  });
});

test("sb-radius: uses half-up rounding and the specified disk shape", () => {
  assert.equal(strokeRadiusPx(12, 640, 480), 6);
  assert.equal(strokeRadiusPx(1, 500, 700), 1);
  assert.equal(strokeRadiusPx(1, 499, 700), 0);
  assert.deepEqual([0, 1, 2, 3].map((r) => diskHalfWidths(r).reduce((n, x, k) => n + (k ? 2 : 1) * (2 * x + 1), 0)), [1, 9, 21, 37]);
  assert.deepEqual([null, "", "bad", "12", "0", "101"].map(normalizeStrokeRadiusPreference), [12, 12, 12, 12, 1, 100]);
});

test("sb-coordinates: maps endpoints and round-trips pixel centers through 5000 px", () => {
  assert.equal(strokePixel(STROKE_Q, 640), 639);
  assert.equal(strokePixel(0, 640), 0);
  for (const w of [1, 2, 3, 7, 640, 1920, 4999, 5000]) {
    for (let px = 0; px < w; px++) assert.equal(strokePixel(pixelToStrokeQ(px, w), w), px, `w=${w}, px=${px}`);
  }
});

test("sb-centers: interpolates with floor-half-up in both directions", () => {
  const [x0, y0] = qAtPixel(0, 0, 5, 3);
  const [x1, y1] = qAtPixel(4, 2, 5, 3);
  const centers = [];
  forEachCenter(stroke(1, 1, x0, y0, x1, y1), 5, 3, (x, y) => centers.push([x, y]));
  assert.deepEqual(centers, [[0, 0], [1, 1], [2, 1], [3, 2], [4, 2]]);
  const reverse = [];
  forEachCenter(stroke(1, 1, x1, y1, x0, y0), 5, 3, (x, y) => reverse.push([x, y]));
  assert.deepEqual(new Set(reverse.map((p) => p.join(","))), new Set(centers.map((p) => p.join(","))));
  let count = 0;
  forEachCenter(stroke(1, 1, x0, y0, x0, y0), 5, 3, () => count++);
  assert.equal(count, 1);
});

test("sb-raster: later labels replace earlier ones and eraser writes zero", () => {
  const [x, y] = qAtPixel(3, 2, 7, 5);
  const result = rasterizeStrokes(7, 5, value([
    stroke(1, 1, x, y), stroke(2, 1, x, y), stroke(0, 1, x, y),
  ]), ALLOWED);
  assert.equal(result.labels[2 * 7 + 3], 0);
  assert.equal(result.strokes, 3);
  assert.equal(result.points, 3);
  const reversed = rasterizeStrokes(7, 5, value([
    stroke(2, 1, x, y), stroke(1, 1, x, y),
  ]), ALLOWED);
  assert.equal(reversed.labels[2 * 7 + 3], 1);
  const counts = new Int32Array(8);
  for (const label of result.labels) counts[label]++;
  assert.deepEqual(result.counts, counts);
});

test("sb-raster: clips at each edge, handles degenerate images, and preserves aspect-scaled positions", () => {
  const [x, y] = qAtPixel(4, 0, 5, 5);
  const edge = rasterizeStrokes(5, 5, value([stroke(1, 400, x, y)]), new Set([1]));
  assert.equal(edge.labels[1 * 5], 0, "the stamp must not wrap to the next row");
  assert.equal(edge.labels[0 * 5 + 4], 1);
  for (const [w, h] of [[1, 1], [1, 9], [9, 1], [0, 0]]) {
    assert.doesNotThrow(() => rasterizeStrokes(w, h, value([stroke(1, 1, 0, 0)]), new Set([1])));
  }
  assert.ok(rasterizeStrokes(0, 0, value([stroke(1, 1, 0, 0)]), new Set([1])).labels instanceof Uint8Array);

  const [qx, qy] = qAtPixel(31, 23, 64, 48);
  const small = rasterizeStrokes(64, 48, value([stroke(1, 12, qx, qy)]), new Set([1]));
  const large = rasterizeStrokes(128, 96, value([stroke(1, 12, qx, qy)]), new Set([1]));
  const centroid = (labels, w) => {
    let xSum = 0, ySum = 0, n = 0;
    for (let i = 0; i < labels.length; i++) if (labels[i] === 1) { xSum += i % w; ySum += Math.floor(i / w); n++; }
    return [xSum / n, ySum / n];
  };
  const a = centroid(small.labels, 64), b = centroid(large.labels, 128);
  assert.ok(Math.abs(a[0] - b[0] / 2) <= 1);
  assert.ok(Math.abs(a[1] - b[1] / 2) <= 1);
});

test("sb-raster-runs: scanline unions equal direct disk stamping on seeded cases", () => {
  let seed = 0x1badb002;
  for (const [w, h] of [[9, 7], [29, 21], [100, 80]]) {
    for (const radius of [1, 12, 100]) {
      const strokes = makeRandomStrokes(seed++, 8, w, h, radius, false, 10);
      const input = value(strokes, w, h);
      const actual = rasterizeStrokes(w, h, input, ALLOWED).labels, expected = referenceRasterize(w, h, input);
      let mismatch = -1;
      for (let i = 0; i < actual.length; i++) if (actual[i] !== expected[i]) { mismatch = i; break; }
      assert.equal(mismatch, -1, `${w}x${h} rr=${radius} at ${mismatch % w},${Math.floor(mismatch / w)}: ${actual[mismatch]} vs ${expected[mismatch]}`);
    }
  }
});

test("sb-aspect-summary: warns only after a real aspect change and counts allowed labels", () => {
  const v = value([stroke(1, 12, 10, 10), stroke(1, 12, 20, 20), stroke(2, 12, 30, 30), stroke(0, 12, 40, 40)], 640, 480);
  assert.equal(strokeAspectChanged(v, 1280, 960), false);
  assert.equal(strokeAspectChanged(v, 640, 500), true);
  assert.deepEqual(summarizeStrokes(v, LABEL_PARAM), {
    text: "วัตถุ 2 เส้น · พื้นหลัง 1 เส้น · ยางลบ 1 ครั้ง",
    short: "วัตถุ 2 · พื้นหลัง 1 · ยางลบ 1",
    empty: false,
  });
  assert.deepEqual(summarizeStrokes(null, LABEL_PARAM), { text: "ยังไม่มีเส้น", short: "ยังไม่มีเส้น", empty: true });
});

test("sb-py-literal: compiles normalized and empty values and contains only data literals", (t) => {
  const py = spawnSync("python", ["-c", "import sys; compile(sys.stdin.buffer.read(), '<stroke>', 'exec')"], {
    input: strokesPyLiteral(value([[1.5, 12.5, -5, 10007, 8, 9, 123]], 640, 480), "strokes5g", LABEL_PARAM).join("\n") + "\n" +
      strokesPyLiteral(null, "strokesEmpty", LABEL_PARAM).join("\n"),
    encoding: "utf8", env: { ...process.env, PYTHONIOENCODING: "utf8" },
  });
  if (py.error?.code === "ENOENT") return t.skip("Python is not installed");
  assert.equal(py.status, 0, py.stderr);
  const helperCompile = spawnSync("python", ["-c", "import sys; compile(sys.stdin.read(), '<stroke-helper>', 'exec')"], {
    input: strokeRasterPyLines().join("\n") + "\n", encoding: "utf8",
  });
  assert.equal(helperCompile.status, 0, helperCompile.stderr);
  const lines = strokesPyLiteral(value([[1, 12, 0, 0, 1, 1]], 4, 4), "strokes5g", LABEL_PARAM);
  const dataLines = lines.filter((line) => !line.trimStart().startsWith("#")).join("\n").replaceAll("strokes5g", "");
  assert.match(dataLines, /^[\d,\s()[\]=]*$/);
  assert.deepEqual(strokesPyLiteral(null, "strokesEmpty", LABEL_PARAM), ["strokesEmpty = []"]);
  assert.ok(strokeRasterPyLines().some((line) => line.includes("def stroke_labels")));
});

test("sb-py-literal: wraps exactly eight and sixteen point pairs", () => {
  const eight = strokesPyLiteral(value([stroke(1, 12, ...Array.from({ length: 16 }, (_, i) => i))]), "mask_2", LABEL_PARAM);
  const sixteen = strokesPyLiteral(value([stroke(1, 12, ...Array.from({ length: 32 }, (_, i) => i))]), "mask_2", LABEL_PARAM);
  assert.equal(eight.filter((line) => line.includes("(1, 12,")).length, 1);
  assert.equal(eight.filter((line) => /^\s+\d/.test(line)).length, 0);
  assert.match(eight.at(-2), /\]\),$/);
  assert.equal(sixteen.filter((line) => line.includes("(1, 12,")).length, 1);
  assert.equal(sixteen.filter((line) => /^\s+\d/.test(line)).length, 1);
  assert.match(sixteen.at(-2), /\]\),$/);
});

test("sb-py-literal: sanitizes invalid Python variable names", () => {
  assert.deepEqual(strokesPyLiteral(null, "mask_2", LABEL_PARAM), ["mask_2 = []"]);
  assert.deepEqual(strokesPyLiteral(null, "2bad-name\nimport os", LABEL_PARAM), ["strokes5g = []"]);
  assert.deepEqual(strokesPyLiteral(null, "class", LABEL_PARAM), ["strokes5g = []"]);
});

const PY_READY = process.env.STROKE_NO_PY !== "1" && spawnSync("python", ["-c", "import cv2, numpy"], { encoding: "utf8" }).status === 0;
test("sb-parity: Python and JavaScript label maps match byte for byte", { skip: PY_READY ? false : "Python/cv2/numpy unavailable or STROKE_NO_PY=1" }, () => {
  const dims = [[45, 37], [37, 45], [1, 9], [9, 1], [1, 1], [640, 480], [1919, 1079]];
  const cases = [];
  let seed = 0x5a17;
  for (const [w, h] of dims) {
    const [cx, cy] = qAtPixel(Math.floor(w / 2), Math.floor(h / 2), Math.max(1, w), Math.max(1, h));
    const [qx1, qy1] = qAtPixel(w - 1, h - 1, Math.max(1, w), Math.max(1, h));
    const [qx2, qy2] = qAtPixel(0, h - 1, Math.max(1, w), Math.max(1, h));
    const patterns = [
      [stroke(1, 12, cx, cy)],
      [stroke(1, 1, cx, cy)],
      [stroke(2, 100, cx, cy)],
      [stroke(1, 12, 0, 0, qx1, qy1)],
      [stroke(1, 12, -100000, -100000, 100000, 100000)],
      [stroke(1, 12, 0, 0, STROKE_Q, 0), stroke(2, 12, 0, STROKE_Q, STROKE_Q, STROKE_Q), stroke(1, 12, 0, 0, 0, STROKE_Q), stroke(2, 12, STROKE_Q, 0, STROKE_Q, STROKE_Q)],
      [stroke(1, 20, cx, cy), stroke(2, 20, cx, cy), stroke(0, 20, cx, cy)],
      makeRandomStrokes(seed++ * 101, 30, w, h, 12),
    ];
    for (let i = 0; i < patterns.length; i++) cases.push({ w, h, v: value(patterns[i], w, h), name: `p${cases.length}` });
  }

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "stroke-brush-"));
  try {
    const literals = cases.map((c) => strokesPyLiteral(c.v, `strokes${c.name}`, LABEL_PARAM).join("\n")).join("\n");
    const rows = cases.map((c, i) => `    (${c.w}, ${c.h}, strokes${c.name}, ${i}),`).join("\n");
    const script = [
      "import os, sys",
      "import cv2, numpy as np",
      ...strokeRasterPyLines(),
      literals,
      "cases = [",
      rows,
      "]",
      "for w, h, strokes, i in cases:",
      "    stroke_labels(w, h, strokes).tofile(os.path.join(sys.argv[1], str(i) + '.bin'))",
    ].join("\n");
    const pyPath = path.join(dir, "parity.py");
    fs.writeFileSync(pyPath, script, "utf8");
    const run = spawnSync("python", [pyPath, dir], { encoding: "utf8", env: { ...process.env, PYTHONIOENCODING: "utf8" }, maxBuffer: 16 * 1024 * 1024 });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    cases.forEach((c, i) => {
      const js = rasterizeStrokes(c.w, c.h, c.v, ALLOWED).labels;
      const py = fs.readFileSync(path.join(dir, `${i}.bin`));
      assert.deepEqual(py, Buffer.from(js), `${c.w}x${c.h} ${c.name}`);
    });
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("sb-py-distance-transform-rasterizer: Python uses precise Euclidean coverage and bounds off-image centres", () => {
  const source = strokeRasterPyLines().join("\n");
  assert.match(source, /cv2\.distanceTransform\(src, cv2\.DIST_L2, cv2\.DIST_MASK_PRECISE\)/);
  assert.match(source, /cov = \(d \* d\) <= \(r \* r \+ r \+ 0\.5\)/);
  assert.match(source, /keep = \(cx >= -r - 1\) & \(cx <= w \+ r\) & \(cy >= -r - 1\) & \(cy <= h \+ r\)/);
  assert.doesNotMatch(source, /np\.add\.at|np\.bincount|for dy in range\(-r, r \+ 1\)/);
});

test("sb-py-worst-parity: vector Python rasterizer stays exact and fast on dense strokes", { skip: PY_READY ? false : "Python/cv2/numpy unavailable or STROKE_NO_PY=1" }, () => {
  const tiny = [];
  for (const [w, h] of [[1, 9], [9, 1]]) {
    const points = w === 1
      ? [qAtPixel(0, 0, w, h), qAtPixel(0, h - 1, w, h)]
      : [qAtPixel(0, 0, w, h), qAtPixel(w - 1, 0, w, h), qAtPixel(w - 1, h - 1, w, h)];
    tiny.push({ name: `border-${w}x${h}`, w, h, v: value([stroke(1, 1, ...points.flat())], w, h) });
  }
  const cases = [
    ...tiny,
    { name: "worst", w: 1920, h: 1080, v: value(makeRandomStrokes(0x5eed, 100, 1920, 1080, 100, false, 100), 1920, 1080) },
    { name: "realistic", w: 1920, h: 1080, v: value(makeRandomStrokes(0x51a7, 8, 1920, 1080, 12, true, 30), 1920, 1080) },
    { name: "rr12-full", w: 1920, h: 1080, v: value(makeRandomStrokes(0x12a5, 100, 1920, 1080, 12, false, 100), 1920, 1080) },
  ];
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "stroke-py-fast-"));
  try {
    const literals = cases.map((c, i) => strokesPyLiteral(c.v, `strokes_${i}`, LABEL_PARAM).join("\n")).join("\n");
    const rows = cases.map((c, i) => `    (${c.w}, ${c.h}, strokes_${i}, ${i}, ${JSON.stringify(c.name)}),`).join("\n");
    const script = [
      "import os, sys, time",
      "import cv2, numpy as np",
      ...strokeRasterPyLines(),
      literals,
      "cases = [", rows, "]",
      "for w, h, strokes, i, name in cases:",
      "    start = time.perf_counter()",
      "    labels = stroke_labels(w, h, strokes)",
      "    elapsed = time.perf_counter() - start",
      "    labels.tofile(os.path.join(sys.argv[1], str(i) + '.bin'))",
      "    print('TIMING', name, format(elapsed, '.6f'))",
    ].join("\n");
    const pyPath = path.join(dir, "fast.py");
    fs.writeFileSync(pyPath, script, "utf8");
    const run = spawnSync("python", [pyPath, dir], { encoding: "utf8", env: { ...process.env, PYTHONIOENCODING: "utf8" }, maxBuffer: 2 * 1024 * 1024, timeout: 180000 });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    for (const [i, c] of cases.entries()) {
      const expected = rasterizeStrokes(c.w, c.h, c.v, ALLOWED).labels;
      assert.deepEqual(fs.readFileSync(path.join(dir, `${i}.bin`)), Buffer.from(expected), c.name);
    }
    const timings = Object.fromEntries(run.stdout.trim().split(/\r?\n/).map((line) => {
      const [, name, seconds] = line.split(/\s+/);
      return [name, Number(seconds)];
    }));
    console.log(`sb-py-timing ${JSON.stringify(timings)}`);
    // เป้าตอนรันเดี่ยว ≤ 10 s (วัดได้ ~4.4 s) · ตอนรันทั้งชุดไฟล์เทสต์วิ่งขนานกัน Python ถูกแย่ง CPU (~15 s) จึงเผื่อเป็น 30 s
    assert.ok(timings.worst <= 30, `worst case took ${timings.worst}s`);
    assert.ok(timings.realistic <= 0.3, `realistic case took ${timings.realistic}s`);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("sb-serialize: JSON round-trip preserves strokes and legacy migration leaves them alone", () => {
  const source = value([stroke(1, 12, 120, 340, 560, 780), stroke(2, 5, 9000, 1000)], 640, 480);
  const norm = normalizeStrokes(source, ALLOWED);
  assert.deepEqual(normalizeStrokes(JSON.parse(JSON.stringify(norm)), ALLOWED), norm);
  const step = { id: "segment_edge", params: { mode: "geodesic_distance", scribble_segment__strokes: norm } };
  const copied = JSON.parse(JSON.stringify(step));
  assert.deepEqual(rasterizeStrokes(64, 48, copied.params.scribble_segment__strokes, ALLOWED).labels,
    rasterizeStrokes(64, 48, step.params.scribble_segment__strokes, ALLOWED).labels);
  assert.deepEqual(migrateLegacyStep(step), step);
  assert.equal(rasterizeStrokes(8, 8, undefined, ALLOWED), null);
  const old = FILTERS.__stroke_default_fixture;
  FILTERS.__stroke_default_fixture = { name: "Stroke default fixture", params: [LABEL_PARAM] };
  try {
    const defaults = defaultParams("__stroke_default_fixture");
    assert.equal(defaults.strokes, null);
    assert.equal(rasterizeStrokes(8, 8, defaults.strokes, ALLOWED), null);
  } finally {
    if (old === undefined) delete FILTERS.__stroke_default_fixture;
    else FILTERS.__stroke_default_fixture = old;
  }
});

test("sb-model: begin, decimate, append the final point, and preserve immutable snapshots", () => {
  const model = createStrokeModel({ allowed: new Set([1, 2]), initial: null });
  assert.deepEqual(model.begin(1, 12, ...qAtPixel(0, 0, 640, 480), 640, 480), { ok: true });
  assert.equal(model.extend(...qAtPixel(2, 0, 640, 480), 640, 480), "skipped");
  assert.equal(model.extend(...qAtPixel(3, 0, 640, 480), 640, 480), "appended");
  assert.equal(model.extend(...qAtPixel(8, 0, 640, 480), 640, 480), "appended");
  const result = model.end();
  assert.deepEqual(result.s[0].slice(2), [...qAtPixel(0, 0, 640, 480), ...qAtPixel(3, 0, 640, 480), ...qAtPixel(8, 0, 640, 480)]);
  assert.equal(result.w, 640);
  assert.equal(result.h, 480);
});

test("sb-model: undo, redo, clear, cancel, and reset clear redo", () => {
  const model = createStrokeModel({ allowed: new Set([1]), initial: null });
  model.begin(1, 12, 0, 0, 20, 20); model.end();
  const first = model.end();
  assert.ok(first);
  model.begin(1, 12, STROKE_Q, STROKE_Q, 20, 20); model.end();
  const second = model.undo();
  assert.equal(second.s.length, 1);
  assert.equal(model.canRedo, true);
  assert.equal(model.redo().s.length, 2);
  assert.equal(model.clear(), null);
  assert.equal(model.canUndo, true);
  assert.equal(model.undo().s.length, 2);
  model.undo();
  model.reset(null);
  assert.equal(model.canRedo, false);
  model.begin(1, 12, 0, 0, 20, 20);
  model.cancel();
  assert.equal(model.end(), null);
});

test("sb-model: reset keeps redo for the same value and clears it for an external change", () => {
  const model = createStrokeModel({ allowed: new Set([1]) });
  model.begin(1, 12, 0, 0, 20, 20); model.end();
  model.begin(1, 12, STROKE_Q, STROKE_Q, 20, 20); model.end();
  model.undo();
  const unchanged = model.value;
  model.reset(unchanged);
  assert.equal(model.canRedo, true);
  model.reset(value([stroke(1, 12, 5000, 5000)]));
  assert.equal(model.canRedo, false);
});

test("sb-model: retains at most one hundred undo snapshots", () => {
  const model = createStrokeModel({ allowed: new Set([1]) });
  for (let i = 0; i < 51; i++) {
    model.begin(1, 12, i * 100, i * 100, 100, 100);
    model.end();
    model.clear();
  }
  let undos = 0;
  while (model.canUndo) { model.undo(); undos++; }
  assert.equal(undos, 100);
  assert.equal(model.canUndo, false);
});

test("sb-model: enforces stroke, per-stroke point, and total point limits", () => {
  const fullStrokes = value(Array.from({ length: STROKE_MAX_STROKES }, () => stroke(1, 12, 0, 0)));
  const byStrokes = createStrokeModel({ allowed: new Set([1]), initial: fullStrokes });
  assert.deepEqual(byStrokes.begin(1, 12, 0, 0, 100, 100), { ok: false, reason: "จำนวนเส้นครบแล้ว" });

  const byPoints = createStrokeModel({ allowed: new Set([1]), initial: null });
  assert.deepEqual(byPoints.begin(1, 1, 0, 0, 100000, 100000), { ok: true });
  for (let i = 1; i < STROKE_MAX_POINTS; i++) {
    assert.equal(byPoints.extend(pixelToStrokeQ(i * 60, 100000), 0, 100000, 100000), "appended");
  }
  assert.equal(byPoints.extend(pixelToStrokeQ(STROKE_MAX_POINTS * 60, 100000), 0, 100000, 100000), "full");
  byPoints.cancel();

  const totalPoints = value(Array.from({ length: 10 }, () => [1, 1, ...Array.from({ length: 1000 * 2 }, (_, i) => i % STROKE_Q)]));
  const byTotal = createStrokeModel({ allowed: new Set([1]), initial: totalPoints });
  assert.deepEqual(byTotal.begin(1, 1, 0, 0, 100, 100), { ok: false, reason: "จำนวนจุดรวมครบแล้ว" });
});

class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.children = []; this.parentNode = null;
    this._text = ""; this._className = ""; this.attrs = {}; this.style = {}; this.hidden = false; this.listeners = {};
    const self = this;
    this.classList = {
      add(c) { if (!this.contains(c)) self._className += (self._className ? " " : "") + c; },
      remove(c) { self._className = self._className.split(" ").filter((x) => x !== c).join(" "); },
      contains(c) { return self._className.split(" ").includes(c); },
      toggle(c, on) { const yes = on === undefined ? !this.contains(c) : !!on; if (yes) this.add(c); else this.remove(c); },
    };
  }
  get className() { return this._className; }
  set className(v) { this._className = String(v); }
  get textContent() { return this.children.length ? this.children.map((c) => c.textContent).join("") : this._text; }
  set textContent(v) { this._text = String(v); this.children = []; }
  set innerHTML(v) { this._text = String(v).replace(/<[^>]*>/g, ""); this.children = []; }
  get innerHTML() { return this._text; }
  appendChild(child) { child.parentNode = this; this.children.push(child); return child; }
  append(...children) { children.forEach((child) => this.appendChild(child)); }
  setAttribute(key, value = "") { this.attrs[key] = String(value); }
  getAttribute(key) { return Object.hasOwn(this.attrs, key) ? this.attrs[key] : null; }
  removeAttribute(key) { delete this.attrs[key]; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter((item) => item !== fn); }
  click() { if (!this.disabled) for (const fn of this.listeners.click || []) fn({}); }
  focus() { if (globalThis.document) globalThis.document.activeElement = this; }
  blur() {}
  dispatch(type, event = {}) { for (const fn of this.listeners[type] || []) fn({ target: this, ...event }); }
  all(cls, out = []) { for (const child of this.children) { if (child.className.split(" ").includes(cls)) out.push(child); child.all(cls, out); } return out; }
  one(cls) { return this.all(cls)[0] || null; }
}

class CanvasEl extends El {
  constructor() {
    super("canvas");
    this.width = 0;
    this.height = 0;
    this.context = {
      putImages: [], texts: [],
      createImageData: (width, height) => ({ width, height, data: new Uint8ClampedArray(width * height * 4) }),
      putImageData(image) { this.putImages.push(image); },
      clearRect() {}, drawImage() {}, fillRect() {}, save() {}, restore() {}, setLineDash() {},
      beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, arc() {},
      fillText(text) { this.texts.push(text); },
    };
  }
  getContext() { return this.context; }
  getBoundingClientRect() { return { left: 0, top: 0, width: this.width || 100, height: this.height || 100 }; }
  setPointerCapture() {}
  releasePointerCapture() {}
}

function installStrokeDom(t) {
  const names = ["document", "window", "localStorage", "requestAnimationFrame"];
  const originals = new Map(names.map((name) => [name, { exists: Object.hasOwn(globalThis, name), value: globalThis[name] }]));
  const created = [], docListeners = {}, windowListeners = {};
  globalThis.document = {
    activeElement: null,
    createElement(tag) { const el = tag === "canvas" ? new CanvasEl() : new El(tag); created.push(el); return el; },
    querySelector() { return null; },
    addEventListener(type, fn) { (docListeners[type] ||= []).push(fn); },
    removeEventListener(type, fn) { docListeners[type] = (docListeners[type] || []).filter((item) => item !== fn); },
    dispatch(type, event) { for (const fn of docListeners[type] || []) fn(event); },
  };
  globalThis.window = {
    addEventListener(type, fn) { (windowListeners[type] ||= []).push(fn); },
    removeEventListener(type, fn) { windowListeners[type] = (windowListeners[type] || []).filter((item) => item !== fn); },
  };
  globalThis.localStorage = { getItem() { return null; }, setItem() {} };
  globalThis.requestAnimationFrame = (callback) => { callback(); return 1; };
  t.after(() => {
    for (const [name, original] of originals) {
      if (original.exists) globalThis[name] = original.value;
      else delete globalThis[name];
    }
  });
  return { created, docListeners };
}

function setupStrokeCanvas(t, initial = null) {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { created } = installStrokeDom(t);
  const stage = new El("div"), canvas = new CanvasEl(), toolbar = new El("div");
  stage.appendChild(canvas);
  const commits = [], statuses = [];
  const brush = createStrokeCanvas({
    canvas, toolbar,
    getValue: () => initial,
    commit: (key, next) => commits.push({ key, value: next }),
    setStatus: (message) => statuses.push(message),
  });
  const target = { nodeId: "node-a", paramKey: "strokes", opName: "ทดสอบ", param: LABEL_PARAM };
  brush.enter(target);
  brush.onInputImage({ width: 100, height: 100, data: new Uint8ClampedArray(100 * 100 * 4) });
  return { brush, canvas, toolbar, commits, statuses, created };
}

function drawCanvasDab(canvas, x, y, pointerId = 1) {
  const event = { pointerType: "mouse", button: 0, pointerId, clientX: x, clientY: y, preventDefault() {} };
  canvas.dispatch("pointerdown", event);
  canvas.dispatch("pointerup", event);
}

function toolbarButton(toolbar, text) {
  return toolbarElements(toolbar).find((el) => el.tagName === "BUTTON" && el.textContent.includes(text)) || null;
}

function toolbarElements(toolbar) {
  const out = [], stack = [...toolbar.children];
  while (stack.length) { const el = stack.shift(); out.push(el); stack.unshift(...el.children); }
  return out;
}

function domSnapshot(root) {
  const items = [];
  const walk = (el) => { items.push([el.tagName, el.className, el.textContent]); el.children.forEach(walk); };
  walk(root);
  return crypto.createHash("sha256").update(JSON.stringify(items)).digest("hex");
}

test("sb-detail: renders summaries, invokes callbacks, disables blocked editing, and clears strokes", () => {
  const old = FILTERS.__stroke_fixture;
  FILTERS.__stroke_fixture = { name: "Stroke fixture", group: "Test", params: [LABEL_PARAM] };
  const hadDocument = Object.hasOwn(globalThis, "document"), originalDocument = globalThis.document;
  try {
    globalThis.document = { activeElement: null, createElement: (tag) => new El(tag) };
    const root = new El("div"), changes = [], edits = [], statuses = [];
    let blocked = null, editing = false;
    const panel = createDetailPanel(root, {
      onParamChange: (key, v) => { changes.push([key, v]); },
      onEditStrokes: (key, param) => edits.push([key, param]),
      strokeEditBlocked: () => blocked,
      isEditingStrokes: () => editing,
      setStatus: (msg) => statuses.push(msg),
    });
    const existing = value([stroke(1, 12, 10, 10), stroke(2, 12, 20, 20), stroke(0, 12, 30, 30)]);
    const step = { id: "__stroke_fixture", params: { strokes: existing }, enabled: true };
    panel.render(step, []);
    assert.equal(root.one("stroke-summary").textContent, "วัตถุ 1 เส้น · พื้นหลัง 1 เส้น · ยางลบ 1 ครั้ง");
    assert.equal(formatParamValue(LABEL_PARAM, existing), "วัตถุ 1 เส้น · พื้นหลัง 1 เส้น · ยางลบ 1 ครั้ง");
    for (const id of ["blur", "segment_edge", "image_arithmetic"]) {
      const params = FILTERS[id].params;
      const hasType = (type) => params.find((p) => p.type === type);
      const number = hasType("number"), select = hasType("select"), bool = hasType("boolean");
      if (number) assert.equal(formatParamValue(number, 12.5), "12.5");
      if (select) assert.equal(formatParamValue(select, "test"), "test");
      if (bool) assert.equal(formatParamValue(bool, true), "true");
    }

    blocked = "ต่อสายกล่องนี้กับแหล่งภาพก่อน ถึงจะวาดเส้นได้";
    panel.render(step, []);
    const edit = root.one("stroke-edit");
    assert.equal(edit.disabled, true);
    assert.equal(root.one("stroke-blocked").textContent, blocked);
    blocked = null;
    editing = true;
    panel.render(step, []);
    assert.equal(root.one("stroke-edit").textContent, "✓ วาดเสร็จแล้ว");
    editing = false;
    root.one("stroke-edit").click();
    assert.equal(edits.length, 1);
    assert.equal(edits[0][0], "strokes");
    const clear = root.one("stroke-clear");
    assert.equal(clear.disabled, false);
    clear.click();
    assert.deepEqual(changes, [["strokes", null]]);
    assert.deepEqual(statuses, ["ล้างเส้นแล้ว — กด Ctrl+Z เพื่อเอาคืน"]);
    assert.equal(root.one("stroke-summary").textContent, "ยังไม่มีเส้น");
    assert.equal(root.one("stroke-clear").disabled, true);
    assert.equal(formatParamValue({ type: "number" }, 0), "0");
    assert.equal(formatParamValue({ type: "select" }, "x"), "x");
    assert.equal(formatParamValue({ type: "boolean" }, false), "false");
  } finally {
    if (old === undefined) delete FILTERS.__stroke_fixture; else FILTERS.__stroke_fixture = old;
    if (hadDocument) globalThis.document = originalDocument; else delete globalThis.document;
  }
});

test("sb-detail-legacy: blur, segment_edge and image_arithmetic DOM snapshots stay identical", () => {
  const hadDocument = Object.hasOwn(globalThis, "document"), originalDocument = globalThis.document;
  try {
    globalThis.document = { activeElement: null, createElement: (tag) => new El(tag) };
    const expected = {
      blur: "d3121237a4b0f37ddfd36dc1fae9a4b95debf0f33c71137e3a10ca65323469e1",
      segment_edge: "d68fef298846b2eba2c6821e00d699671a1faee8af3ec5df1bb8a91e1fd8c6c8",
      image_arithmetic: "9c5dff5a6dab32215950fc1f13d2ad19182cef05cbda9ff134d79228be110589",
    };
    for (const [id, hash] of Object.entries(expected)) {
      const root = new El("div");
      createDetailPanel(root, { onParamChange() {} }).render({ id, params: defaultParams(id), enabled: true }, []);
      assert.equal(domSnapshot(root), hash, `${id} DOM structure changed`);
    }
  } finally {
    if (hadDocument) globalThis.document = originalDocument; else delete globalThis.document;
  }
});

test("sb-canvas-clear: a debounced clear commits null once and reports the status", (t) => {
  const initial = value([stroke(1, 12, 1000, 1000)]);
  const { toolbar, commits, statuses } = setupStrokeCanvas(t, initial);
  toolbarButton(toolbar, "ล้าง").click();
  assert.equal(commits.length, 0);
  assert.deepEqual(statuses, ["ล้างเส้นแล้ว — กด ↶ ย้อน เพื่อเอาคืน"]);
  t.mock.timers.tick(120);
  assert.equal(commits.length, 1);
  assert.deepEqual(commits[0], { key: "strokes", value: null });
});

test("sb-canvas-clear: undoing the only new stroke commits null", (t) => {
  const { canvas, commits } = setupStrokeCanvas(t);
  drawCanvasDab(canvas, 10, 10);
  t.mock.timers.tick(120);
  assert.ok(commits[0].value);
  document.dispatch("keydown", {
    target: canvas, code: "KeyZ", key: "z", ctrlKey: true, metaKey: false, shiftKey: false,
    preventDefault() {}, stopPropagation() {},
  });
  t.mock.timers.tick(120);
  assert.equal(commits.length, 2);
  assert.equal(commits[1].value, null);
});

test("sb-canvas-clear: exiting immediately flushes a pending clear", (t) => {
  const { brush, toolbar, commits } = setupStrokeCanvas(t, value([stroke(1, 12, 1000, 1000)]));
  toolbarButton(toolbar, "ล้าง").click();
  brush.exit();
  assert.deepEqual(commits, [{ key: "strokes", value: null }]);
  t.mock.timers.tick(120);
  assert.equal(commits.length, 1);
});

test("sb-canvas-external: a stale notify cannot replace a newer pending stroke", (t) => {
  const { brush, canvas, commits } = setupStrokeCanvas(t);
  drawCanvasDab(canvas, 10, 10);
  t.mock.timers.tick(120);
  const committedA = commits[0].value;
  brush.onExternalValue(committedA);
  t.mock.timers.tick(20);
  drawCanvasDab(canvas, 80, 80);
  brush.onExternalValue(committedA);
  t.mock.timers.tick(120);
  assert.equal(commits.length, 2);
  assert.equal(commits[1].value.s.length, 2);
});

test("sb-canvas-images: ignores frames while inactive and releases image canvases on exit", (t) => {
  const { brush, created } = setupStrokeCanvas(t);
  const [input, result] = created.filter((el) => el.tagName === "CANVAS");
  brush.onResultImage({ width: 64, height: 32, data: new Uint8ClampedArray(64 * 32 * 4) });
  assert.equal(input.context.putImages.length, 1);
  assert.equal(result.context.putImages.length, 1);
  brush.exit();
  assert.deepEqual([input.width, input.height, result.width, result.height], [0, 0, 0, 0]);
  brush.onInputImage({ width: 50, height: 50, data: new Uint8ClampedArray(50 * 50 * 4) });
  brush.onResultImage({ width: 50, height: 50, data: new Uint8ClampedArray(50 * 50 * 4) });
  assert.equal(input.context.putImages.length, 1);
  assert.equal(result.context.putImages.length, 1);
  assert.deepEqual([input.width, input.height, result.width, result.height], [0, 0, 0, 0]);
});

test("sb-canvas-enter: re-enter waits for its own input image before allowing drawing", (t) => {
  const { brush, canvas, commits } = setupStrokeCanvas(t, value([stroke(1, 12, 1000, 1000)]));
  brush.exit();
  brush.enter({ nodeId: "node-b", paramKey: "strokes", opName: "ภาพใหม่", param: LABEL_PARAM });
  assert.ok(canvas.context.texts.includes("กำลังโหลดภาพที่เข้ากล่องนี้…"));
  drawCanvasDab(canvas, 40, 40);
  t.mock.timers.tick(120);
  assert.equal(commits.length, 0);
});

test("sb-canvas-pointer: a second pointer press during a drag is ignored silently", (t) => {
  const { canvas, commits, statuses } = setupStrokeCanvas(t);
  const down = (pointerId, clientX) => ({ pointerType: "mouse", button: 0, pointerId, clientX, clientY: 20, preventDefault() {} });
  canvas.dispatch("pointerdown", down(1, 10));
  canvas.dispatch("pointerdown", down(2, 80));
  assert.deepEqual(statuses, []);
  canvas.dispatch("pointerup", down(1, 10));
  t.mock.timers.tick(120);
  assert.equal(commits.length, 1);
  assert.equal(commits[0].value.s.length, 1);
});

test("sb-canvas-keys: toolbar keys and control shortcuts pass through; slider arrows stay local", (t) => {
  const { brush, toolbar, canvas } = setupStrokeCanvas(t);
  const fire = (target, code, { ctrlKey = false, shiftKey = false } = {}) => {
    let prevented = false, stopped = false;
    document.dispatch("keydown", {
      target, code, key: code === "Space" ? " " : code, ctrlKey, shiftKey,
      preventDefault() { prevented = true; }, stopPropagation() { stopped = true; },
    });
    return { prevented, stopped };
  };
  for (const code of ["KeyC", "KeyV", "KeyI", "KeyR"]) {
    assert.deepEqual(fire(canvas, code, { ctrlKey: true, shiftKey: code === "KeyI" }), { prevented: false, stopped: false });
  }
  assert.deepEqual(fire(canvas, "Tab"), { prevented: false, stopped: false });
  const button = toolbarButton(toolbar, "เสร็จ");
  assert.deepEqual(fire(button, "Space"), { prevented: false, stopped: false });
  assert.deepEqual(fire(button, "Enter"), { prevented: false, stopped: false });
  const slider = toolbarElements(toolbar).find((el) => el.tagName === "INPUT" && el.type === "range");
  assert.deepEqual(fire(slider, "ArrowRight"), { prevented: false, stopped: true });
  assert.equal(fire(canvas, "Enter").prevented, true);
  assert.equal(brush.isActive(), false);
});

test("sb-aspect-warning: an exact one-percent ratio change is not over the boundary", () => {
  const v = value([stroke(1, 12, 0, 0)], 100, 100);
  assert.equal(strokeAspectChanged(v, 100, 101), false);
  assert.equal(strokeAspectChanged(v, 100, 102), true);
  assert.equal(strokeAspectChanged(v, 101, 100), false);
  assert.equal(strokeAspectChanged(v, 99, 100), false);
  assert.equal(strokeAspectChanged(v, 102, 100), true);
  assert.equal(strokeAspectChanged(v, 98, 100), true);
});

test("sb-performance: rasterizes 100 full-range seeded strokes with 100 points under three seconds", () => {
  const strokes = makeRandomStrokes(0x5eed, 100, 1920, 1080, 100, false, 100);
  const start = performance.now();
  const result = rasterizeStrokes(1920, 1080, value(strokes, 1920, 1080), ALLOWED);
  const elapsed = performance.now() - start;
  console.log(`sb-performance 1920x1080: ${elapsed.toFixed(1)} ms`);
  assert.ok(result && result.strokes === 100);
  assert.ok(elapsed < 3000, `rasterization took ${elapsed.toFixed(1)} ms`);
});

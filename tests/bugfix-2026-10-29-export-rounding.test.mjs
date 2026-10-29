// Exact byte parity on tie-free maps, using the raw Python parity harness.
import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import cv from "@techstark/opencv-js";
import { OPS } from "../src/renderer/filters/registry.js";
import { OP_TO_FILTER } from "../src/renderer/ui/palette-panel.js";
import { buildPythonCode } from "../src/renderer/codegen.js";

while (typeof cv.Mat !== "function") await new Promise((r) => setTimeout(r, 20));
globalThis.self = { postMessage() {} };
const { runPipeline } = await import("../src/renderer/cv/cv-worker.js");
const defaults = (id) => Object.fromEntries(OPS[id].params.map((q) => [q.key, q.default]));
function step(id, p = {}) {
  const fid = OP_TO_FILTER[id] || id, local = { ...defaults(id), ...p };
  const st = fid === id ? { id, params: local } : { id: fid, params: { mode: id, ...Object.fromEntries(Object.entries(local).map(([k, v]) => [id + "__" + k, v])) } };
  if (id === "ssim") st.src2 = { pipeline: [] };
  return st;
}
function fixture(w = 33, h = 31, second = false) {
  const data = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set(second
    ? [(x * 13 + y * 7 + 31) % 256, (x * 19 + y * 3 + 51) % 256, (x * 11 + y * 17) % 256]
    : [(x * 17 + y * 3) % 256, (x * 5 + y * 23) % 256, (x * 29 + y * 11 + 91) % 256], (y * w + x) * 3);
  return cv.matFromArray(h, w, cv.CV_8UC3, data);
}
// Repeated rows keep the float map's distinct values few enough to select a
// fixture away from half ties. Seeded bytes are explicit and platform-stable.
function stripes(w, h, second = false, seed = 0) {
  let state = seed + 1;
  const row = Array.from({ length: w }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return ((state >>> 24) + (second ? 51 : 0)) % 256;
  });
  return cv.matFromArray(h, w, cv.CV_8UC3, Uint8Array.from({ length: w * h * 3 }, (_, i) => row[Math.floor(i / 3) % w]));
}
const tieDistance = (v) => Math.abs(v - Math.floor(v) - 0.5);
let pyAvailable = false;
try { execFileSync("python", ["-c", "import cv2,numpy"], { stdio: "pipe" }); pyAvailable = true; } catch {}
const PY_SKIP = { skip: !pyAvailable && "python/cv2/numpy unavailable" };

// The only controlled dependency is kmeans' RNG-dependent clustering result.
// Exercise the real app/export centers -> labels -> image mapping with identical
// float32 centers and labels, independent of prior native RNG state or version.
const CENTERS = [
  [0.4999999701976776, 0.5, 2.5], // promoting before +0.5 matters; ties round up
  [12.5, 31.75, 254.5],
  [100.25, 127.5, 200.875],
  [255, 64.5, 1.25],
];
const COLORS = [[0, 1, 3], [13, 32, 255], [100, 128, 201], [255, 65, 1]];
function mappedApp(src, params, analysisSize = [src.cols, src.rows], trace = {}) {
  const k = params.k, ctx = {};
  const controlled = new Proxy(cv, { get(target, key) {
    if (key !== "kmeans") return target[key];
    return (samples, K, labels, criteria, attempts, flags, centers) => {
      assert.equal(K, k);
      assert.equal(samples.type(), cv.CV_32FC1);
      assert.equal(samples.cols, 3);
      assert.equal(samples.rows, analysisSize[0] * analysisSize[1], "cluster the analysis image, not the full input");
      trace.samples = new Float32Array(samples.data32F);
      labels.create(samples.rows, 1, cv.CV_32SC1);
      centers.create(k, 3, cv.CV_32FC1);
      for (let i = 0; i < samples.rows; i++) labels.data32S[i] = i % k;
      centers.data32F.set(CENTERS.slice(0, k).flat());
      return 0;
    };
  } });
  return OPS.kmeans_segment.apply(controlled, src, params, ctx);
}
function pythonCase(id, params, src, src2, mutate = (code) => code, analysisSize = [src.cols, src.rows]) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "export-rounding-parity-"));
  const bgr = new cv.Mat();
  try {
    const input = path.join(dir, "input.bin"), inputB = path.join(dir, "input-b.bin"), output = path.join(dir, "output.bin");
    cv.cvtColor(src, bgr, cv.COLOR_RGB2BGR); fs.writeFileSync(input, Buffer.from(bgr.data));
    cv.cvtColor(src2, bgr, cv.COLOR_RGB2BGR); fs.writeFileSync(inputB, Buffer.from(bgr.data));
    let code = mutate(buildPythonCode([step(id, params)]));
    code = code.slice(0, code.indexOf('if __name__ == "__main__":'));
    for (const name of ["path", "path_a", "path_b"]) code = code.replace(`cv2.imread(${name})`, `np.fromfile(${name}, dtype=np.uint8).reshape((${src.rows}, ${src.cols}, 3))`);
    const floatOutput = path.join(dir, "pre-round.bin"), samplesOutput = path.join(dir, "samples.bin"), maskOutput = path.join(dir, "mask.bin");
    const normalized = {
      distance_transform: "cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)",
      fft_magnitude: "cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)",
      fft_filter: "cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX)",
      ssim: "cv2.normalize(ssim_map, None, 0, 255, cv2.NORM_MINMAX)",
    }[id];
    if (normalized) {
      assert.equal(code.split(normalized).length - 1, 1, "capture exactly the pre-rounding float map");
      code = code.replace(normalized, `capture(${normalized})`);
      code += `\ndef capture(values):\n    values.astype('<f4').tofile(${JSON.stringify(floatOutput)})\n    print('MAP', *values.shape)\n    return values\n`;
    }
    if (id === "fft_filter") {
      code = code.replace("complex_i[:, :, 0] *= mask", `mask.astype('<f4').tofile(${JSON.stringify(maskOutput)})\n    complex_i[:, :, 0] *= mask`);
    }
    if (id === "kmeans_segment") {
      // Python starts in BGR; reverse the controlled RGB centers correspondingly.
      code += `\ndef controlled_kmeans(Z, K, *args):\n    assert K == ${params.k}\n    assert Z.shape == (${analysisSize[0] * analysisSize[1]}, 3), Z.shape\n    Z.astype('<f4').tofile(${JSON.stringify(samplesOutput)})\n    centers = np.array(${JSON.stringify(CENTERS.slice(0, params.k).map((c) => [...c].reverse()))}, dtype=np.float32)\n    labels = (np.arange(Z.shape[0]) % K).astype(np.int32).reshape(-1, 1)\n    return 0, labels, centers\ncv2.kmeans = controlled_kmeans\n`;
    }
    const args = code.includes("def process(path_a, path_b)") ? `${JSON.stringify(input)}, ${JSON.stringify(inputB)}` : JSON.stringify(input);
    code += `\nr = process(${args})\nr.tofile(${JSON.stringify(output)})\nprint('SHAPE', *r.shape)\n`;
    const file = path.join(dir, "case.py"); fs.writeFileSync(file, code);
    const stdout = execFileSync("python", [file], { encoding: "utf8", env: { ...process.env, PYTHONIOENCODING: "utf8" } });
    const bytes = fs.readFileSync(output), shape = stdout.match(/SHAPE (.*)/)[1].split(" ").map(Number);
    if (shape.length === 3) for (let i = 0; i < bytes.length; i += 3) [bytes[i], bytes[i + 2]] = [bytes[i + 2], bytes[i]];
    const floats = (file) => {
      const raw = fs.readFileSync(file);
      return Array.from({ length: raw.length / 4 }, (_, i) => raw.readFloatLE(i * 4));
    };
    return { bytes, shape, map: normalized && floats(floatOutput), samples: id === "kmeans_segment" && floats(samplesOutput), mask: id === "fft_filter" && floats(maskOutput) };
  } finally { bgr.delete(); fs.rmSync(dir, { recursive: true, force: true }); }
}
function appCase(id, params, src, src2, analysisSize, trace) {
  if (id === "kmeans_segment") return mappedApp(src, { ...defaults(id), ...params }, analysisSize, trace);
  if (id !== "ssim" && id !== "fft_filter") return runPipeline(src, [step(id, params)], new Map(), new Set()).mat;
  const gray = new cv.Mat();
  try {
    cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
    if (id === "fft_filter") {
      const controlled = new Proxy(cv, { get(target, key) {
        if (key !== "matFromArray") return target[key];
        return (rows, cols, type, data) => {
          if (type === cv.CV_32FC1) trace.mask = new Float32Array(data);
          return target.matFromArray(rows, cols, type, data);
        };
      } });
      return OPS.fft_filter.apply(controlled, gray, { ...defaults(id), ...params }, {});
    }
    return OPS.ssim.apply(cv, gray, { ...defaults(id), ...params }, { src2 });
  } finally { gray.delete(); }
}
function parity(id, params = {}, w = 33, h = 31, mutate, { seed, analysisSize = [w, h] } = {}) {
  const make = seed === undefined ? fixture : (w, h, second) => stripes(w, h, second, seed);
  const src = make(w, h), src2 = make(w, h, true), trace = {}; let out;
  try {
    out = appCase(id, params, src, src2, analysisSize, trace);
    const py = pythonCase(id, { ...defaults(id), ...params }, src, src2, mutate, analysisSize);
    if (py.map) {
      assert.ok(py.map.length > 0 && py.map.every(Number.isFinite), `${id}: finite pre-rounding map required`);
      // OpenCV.js and cv2 DFTs (and SSIM float arithmetic) can straddle x.5
      // through float noise. Exact byte parity is meaningful only away from ties.
      // Any future >640 FFT case must use a 0.05 margin instead of 0.01.
      const margin = id.startsWith("fft_") && Math.max(w, h) > 640 ? 0.05 : 0.01;
      const nearTies = py.map.filter((v) => tieDistance(v) <= margin);
      assert.equal(nearTies.length, 0, `${id}: ${nearTies.length} pre-rounding values within ${margin} of x.5`);
    }
    if (py.samples) {
      assert.equal(py.samples.length, trace.samples.length);
      for (let i = 0; i < py.samples.length; i += 3) {
        assert.deepEqual(py.samples.slice(i, i + 3).reverse(), Array.from(trace.samples.slice(i, i + 3)), `analysis sample ${i / 3}: INTER_AREA bytes`);
      }
    }
    if (py.mask) {
      // The final image can hide tiny mask errors. Compare the real float32
      // masks too, so casting low before 1-low fails even on a byte-stable image.
      assert.deepEqual(py.mask, Array.from(trace.mask), "FFT mask must retain double precision through high-pass subtraction");
    }
    assert.deepEqual(py.shape, out.channels() === 1 ? [out.rows, out.cols] : [out.rows, out.cols, out.channels()]);
    const different = py.bytes.reduce((n, v, i) => n + (v !== out.data[i]), 0);
    assert.equal(py.bytes.length, out.data.length);
    assert.equal(different, 0, `${id}: ${different} bytes different`);
  } finally { if (out) out.delete(); src.delete(); src2.delete(); }
}

// Literal HEAD expressions are retained so mutation tests restore the actual
// truncation, including its position before resize, rather than a no-op edit.
const CONVERSIONS = {
  distance_transform: ["img = np.clip(np.rint(cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)), 0, 255).astype(np.uint8)", "img = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)"],
  fft_magnitude: ["img = cv2.resize(np.clip(np.rint(mag), 0, 255).astype(np.uint8), (cols, rows))", "img = cv2.resize(mag.astype(np.uint8), (cols, rows))"],
  fft_filter: ["inv = np.clip(np.rint(cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX)), 0, 255).astype(np.uint8)", "inv = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)"],
  ssim: ["img = cv2.resize(np.clip(np.rint(normed), 0, 255).astype(np.uint8), (cols, rows))", "img = cv2.resize(normed.astype(np.uint8), (cols, rows))"],
  kmeans_segment: ["centers = np.floor(centers.astype(np.float64) + 0.5).astype(np.uint8)", "centers = np.uint8(centers)"],
};
function truncate(id, code) {
  const [rounded, old] = CONVERSIONS[id];
  assert.equal(code.split(rounded).length - 1, 1, `${id}: mutation must replace exactly one conversion`);
  const mutated = code.replace(rounded, old);
  assert.notEqual(mutated, code, `${id}: mutation must change exported Python`);
  return mutated;
}
for (const [id, [rounded]] of Object.entries(CONVERSIONS)) {
  test(`export-rounding. ${id} emits the app's byte conversion in place`, () => {
    const code = buildPythonCode([step(id)]);
    assert.ok(code.includes(rounded), id);
    truncate(id, code);
  });
}
test("export-rounding. native float32 convertTo saturates and rounds ties to even", () => {
  const src = cv.matFromArray(1, 8, cv.CV_32FC1, [-0.6, 0.5, 1.5, 2.5, 3.5, 254.5, 255.6, 300]), dst = new cv.Mat();
  try {
    src.convertTo(dst, cv.CV_8U);
    assert.deepEqual(Array.from(dst.data), [0, 0, 2, 2, 4, 254, 255, 255]);
  } finally { src.delete(); dst.delete(); }
});
test("export-rounding. high-pass mask casts only after double-precision subtraction", () => {
  for (const shape of ["gaussian", "ideal", "butterworth"]) {
    const code = buildPythonCode([step("fft_filter", { shape, type: "high", cutoff: 100, order: 5 })]);
    assert.ok(code.includes("mask = (1 - low).astype(np.float32)"), shape);
    assert.ok(!code.match(/low = .*astype\(np\.float32\)/), `${shape}: preserve low precision until subtraction`);
  }
});
test("export-rounding. kmeans emits analysis resize and nearest color-result upscale", () => {
  const code = buildPythonCode([step("kmeans_segment")]);
  assert.ok(code.includes("scale = min(1.0, 320 / max(rows, cols))"));
  assert.ok(code.includes("Z = analysis.reshape((-1, 3)).astype(np.float32)"));
  assert.ok(code.includes("img = centers[labels.flatten()].reshape(analysis.shape)"));
  assert.ok(code.includes("img = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_NEAREST)"));
});
for (const k of [4, 3]) {
  test(`export-rounding. kmeans k=${k} maps float32 centers with half-up ties`, () => {
    const src = fixture(); let out;
    try {
      out = mappedApp(src, { ...defaults("kmeans_segment"), k });
      for (let i = 0; i < src.rows * src.cols; i++) assert.deepEqual(Array.from(out.data.slice(i * 3, i * 3 + 3)), COLORS[i % k]);
      assert.ok(out.data.some((v, i) => v !== Math.trunc(Math.fround(CENTERS[Math.floor(i / 3) % k][i % 3]))), "the old center cast must fail");
    } finally { if (out) out.delete(); src.delete(); }
  });
}

for (const maskSize of ["5", "3"]) test(`export-rounding-P. distance mask=${maskSize}: 0 bytes different`, PY_SKIP, () => parity("distance_transform", { maskSize }));
// No magnitude params exist. Keep padding + resize; omit >640 exact parity:
// DFT float noise crosses half ties on that analysis path for every tried seed.
test("export-rounding-P. magnitude 33x31 tie-free: 0 bytes different", PY_SKIP, () => parity("fft_magnitude", {}, 33, 31, undefined, { seed: 15 }));
for (const [shape, type, cutoff, order, seed] of [
  ["gaussian", "low", 30, 5, 22], ["ideal", "low", 30, 5, 29], ["butterworth", "low", 30, 5, 26],
  ["gaussian", "high", 100, 5, 27], ["ideal", "high", 10, 3, 41], ["butterworth", "high", 100, 5, 29],
]) {
  test(`export-rounding-P. FFT ${shape}/${type} cutoff=${cutoff} order=${order} tie-free: 0 bytes different`, PY_SKIP, () => parity("fft_filter", { shape, type, cutoff, order }, 96, 80, undefined, { seed }));
}
for (const speed of ["full", "fast"]) test(`export-rounding-P. SSIM ${speed} 33x31 tie-free: 0 bytes different`, PY_SKIP, () => parity("ssim", { speed }, 33, 31, undefined, { seed: 0 }));
for (const k of [4, 3]) test(`export-rounding-P. controlled kmeans k=${k}: 0 bytes different`, PY_SKIP, () => parity("kmeans_segment", { k, ...(k === 3 ? { iterations: 30 } : {}) }));
// 201x640 also catches Python round()'s ties-to-even: 201 * 0.5 must be 101.
for (const [w, h, analysisSize] of [[400, 300, [320, 240]], [300, 641, [150, 320]], [201, 640, [101, 320]]]) {
  test(`export-rounding-P. controlled kmeans ${w}x${h} -> ${analysisSize.join("x")}: 0 bytes different`, PY_SKIP, () => parity("kmeans_segment", {}, w, h, undefined, { analysisSize }));
}
test("export-rounding-P. casting low before high-pass subtraction fails mask parity", PY_SKIP, () => {
  const castEarly = (code) => {
    const line = "mask = (1 - low).astype(np.float32)";
    assert.equal(code.split(line).length - 1, 1);
    return code.replace(line, "mask = 1 - low.astype(np.float32)");
  };
  assert.throws(() => parity("fft_filter", { shape: "butterworth", type: "high", cutoff: 100, order: 5 }, 96, 80, castEarly, { seed: 29 }),
    (err) => err.code === "ERR_ASSERTION" && /FFT mask must retain double precision/.test(err.message));
});
test("export-rounding-P. full-resolution kmeans clustering fails the analysis-size contract", PY_SKIP, () => {
  const fullSize = (code) => {
    const line = "Z = analysis.reshape((-1, 3)).astype(np.float32)";
    assert.equal(code.split(line).length - 1, 1);
    return code.replace(line, "Z = img.reshape((-1, 3)).astype(np.float32)");
  };
  assert.throws(() => parity("kmeans_segment", {}, 400, 300, fullSize, { analysisSize: [320, 240] }),
    (err) => /AssertionError: \(120000, 3\)/.test(String(err.stderr)));
});
for (const id of Object.keys(CONVERSIONS)) {
  test(`export-rounding-P. ${id}: actual HEAD truncation fails byte parity`, PY_SKIP, () => {
    const [params, w, h, options] = id === "fft_filter" ? [{ order: 5 }, 96, 80, { seed: 22 }]
      : id === "fft_magnitude" ? [{}, 33, 31, { seed: 15 }]
      : id === "ssim" ? [{}, 33, 31, { seed: 0 }] : [{}, 33, 31, {}];
    assert.throws(() => parity(id, params, w, h, (code) => truncate(id, code), options), (err) => err.code === "ERR_ASSERTION" && /[1-9]\d* bytes different/.test(err.message));
  });
}

// Large-image companion to export-rounding: float noise may flip half ties.
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
const defaults = (id) => Object.fromEntries(OPS[id].params.map((q) => [q.key, q.default]));
// Minimal step/raw-I/O helpers copied from export-rounding; importing that test
// would register its tests too. Keep the real two-image export for SSIM.
function step(id, p = {}) {
  const fid = OP_TO_FILTER[id] || id, local = { ...defaults(id), ...p };
  const st = fid === id ? { id, params: local } : { id: fid, params: { mode: id, ...Object.fromEntries(Object.entries(local).map(([k, v]) => [id + "__" + k, v])) } };
  if (id === "ssim") st.src2 = { pipeline: [] };
  return st;
}
function fixture(w, h, second = false) {
  let state = second ? 0x9abcdef0 : 0x12345678;
  const data = new Uint8Array(w * h * 3);
  for (let i = 0; i < data.length; i += 3) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    // RGB triplets with equal channels isolate FFT/SSIM arithmetic from colour
    // conversion differences. Both independent images vary in x AND y.
    data.fill(state >>> 24, i, i + 3);
  }
  return cv.matFromArray(h, w, cv.CV_8UC3, data);
}
const CASES = [
  { name: "magnitude 800x600", id: "fft_magnitude", w: 800, h: 600, params: {}, analysis: [480, 640] },
  { name: "magnitude 641x480", id: "fft_magnitude", w: 641, h: 480, params: {}, analysis: [479, 640] },
  { name: "Gaussian low-pass 800x600", id: "fft_filter", w: 800, h: 600, params: { shape: "gaussian", type: "low", cutoff: 30 }, analysis: [480, 640] },
  { name: "Butterworth high-pass 800x600", id: "fft_filter", w: 800, h: 600, params: { shape: "butterworth", type: "high", cutoff: 100, order: 5 }, analysis: [480, 640] },
  { name: "SSIM full 800x600", id: "ssim", w: 800, h: 600, params: { speed: "full" }, analysis: [600, 800] },
  { name: "SSIM fast 800x600", id: "ssim", w: 800, h: 600, params: { speed: "fast" }, analysis: [480, 640] },
];
let pyAvailable = false;
try { execFileSync("python", ["-c", "import cv2,numpy"], { stdio: "pipe", timeout: 10000 }); pyAvailable = true; } catch {}
const PY_SKIP = { skip: !pyAvailable && "python/cv2/numpy unavailable" };
const FAST_NOTE = "คำนวณจากภาพที่ย่อขนาดลงก่อนเพื่อความเร็ว";
const appResults = new Map();

function appCase(c) {
  if (appResults.has(c.name)) return appResults.get(c.name);
  const src = fixture(c.w, c.h), src2 = fixture(c.w, c.h, true), gray = new cv.Mat();
  const analysis = [], resizes = [], ctx = c.id === "ssim" ? { src2 } : {};
  // Observe real calls without replacing their results or arithmetic.
  const observed = new Proxy(cv, { get(target, key) {
    if (key === "resize") return (...args) => {
      const [input, , size, , , interpolation] = args;
      resizes.push({ from: [input.rows, input.cols], to: [size.height, size.width], interpolation });
      return target.resize(...args);
    };
    if ((c.id === "ssim" && key === "GaussianBlur") || (c.id !== "ssim" && key === "copyMakeBorder")) return (...args) => {
      analysis.push([args[0].rows, args[0].cols]);
      return target[key](...args);
    };
    return target[key];
  } });
  let out;
  try {
    cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
    out = OPS[c.id].apply(observed, gray, { ...defaults(c.id), ...c.params }, ctx);
    assert.equal(out.channels(), 1, `${c.name}: grayscale output`);
    assert.deepEqual([out.rows, out.cols], [c.h, c.w], `${c.name}: original output size`);
    assert.ok(analysis.length > 0, `${c.name}: observe actual analysis input`);
    for (const shape of analysis) assert.deepEqual(shape, c.analysis, `${c.name}: analysis size`);
    const area = resizes.filter((r) => r.interpolation === cv.INTER_AREA);
    const count = c.params.speed === "full" ? 0 : c.id === "ssim" ? 2 : 1;
    assert.equal(area.length, count, `${c.name}: both SSIM inputs downscale only in fast mode`);
    for (const r of area) {
      assert.deepEqual(r.from, [c.h, c.w]);
      assert.deepEqual(r.to, c.analysis);
    }
    assert.deepEqual(resizes.at(-1).to, [c.h, c.w], `${c.name}: resize result back to original size`);
    assert.equal(resizes.at(-1).interpolation, cv.INTER_LINEAR);
    assert.ok(cv.minMaxLoc(out).maxVal > cv.minMaxLoc(out).minVal, `${c.name}: nonconstant result`);
    assert.equal(typeof ctx.info, "string");
    if (c.id === "ssim") {
      assert.match(ctx.info, /SSIM = -?\d+\.\d{4}/);
      assert.equal(ctx.info.includes(FAST_NOTE), c.params.speed === "fast");
      assert.notDeepEqual(src.data, src2.data, "SSIM must receive two distinct images");
    } else if (c.id === "fft_filter") {
      assert.match(ctx.info, new RegExp(`รัศมี ${c.params.cutoff} \\(`));
      if (c.params.order) assert.match(ctx.info, new RegExp(`n=${c.params.order}\\)`));
    }
    const result = { bytes: Buffer.from(out.data), shape: [out.rows, out.cols], info: ctx.info, analysis: c.analysis };
    appResults.set(c.name, result);
    return result;
  } finally { if (out) out.delete(); gray.delete(); src.delete(); src2.delete(); }
}

function replaceOnce(code, before, after) {
  assert.equal(code.split(before).length - 1, 1, `harness/mutation must replace exactly one ${before}`);
  return code.replace(before, after);
}
function pythonCase(c, mutate = (code) => code) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "large-fft-ssim-parity-"));
  const src = fixture(c.w, c.h), src2 = fixture(c.w, c.h, true), bgr = new cv.Mat();
  try {
    const input = path.join(dir, "input.bin"), inputB = path.join(dir, "input-b.bin"), output = path.join(dir, "output.bin");
    cv.cvtColor(src, bgr, cv.COLOR_RGB2BGR); fs.writeFileSync(input, Buffer.from(bgr.data));
    cv.cvtColor(src2, bgr, cv.COLOR_RGB2BGR); fs.writeFileSync(inputB, Buffer.from(bgr.data));
    let code = mutate(buildPythonCode([step(c.id, c.params)]));
    const main = code.indexOf('if __name__ == "__main__":');
    assert.ok(main > 0, "remove only the interactive export entry point");
    code = code.slice(0, main);
    for (const name of c.id === "ssim" ? ["path_a", "path_b"] : ["path"]) {
      code = replaceOnce(code, `cv2.imread(${name})`, `np.fromfile(${name}, dtype=np.uint8).reshape((${c.h}, ${c.w}, 3))`);
    }
    // Print real analysis shapes, leaving resize/DFT/SSIM/conversion untouched.
    if (c.id === "ssim") {
      code = replaceOnce(code, "af = a_a.astype(np.float32)", "print('ANALYSIS', *a_a.shape)\n    print('ANALYSIS', *a_b.shape)\n    af = a_a.astype(np.float32)");
    } else {
      code = replaceOnce(code, "a_rows, a_cols = analysis.shape", "a_rows, a_cols = analysis.shape\n    print('ANALYSIS', *analysis.shape)");
    }
    const args = c.id === "ssim" ? `${JSON.stringify(input)}, ${JSON.stringify(inputB)}` : JSON.stringify(input);
    code += `\nr = process(${args})\nr.tofile(${JSON.stringify(output)})\nprint('SHAPE', *r.shape)\n`;
    const file = path.join(dir, "case.py"); fs.writeFileSync(file, code);
    const stdout = execFileSync("python", [file], { encoding: "utf8", timeout: 15000, env: { ...process.env, PYTHONIOENCODING: "utf8", PYTHONDONTWRITEBYTECODE: "1" } });
    const shape = stdout.match(/^SHAPE (.+)$/m);
    assert.ok(shape, stdout);
    return {
      bytes: fs.readFileSync(output), shape: shape[1].trim().split(/\s+/).map(Number), stdout,
      analysis: Array.from(stdout.matchAll(/^ANALYSIS (.+)$/gm), (m) => m[1].trim().split(/\s+/).map(Number)),
    };
  } finally { bgr.delete(); src.delete(); src2.delete(); fs.rmSync(dir, { recursive: true, force: true }); }
}

function compare(c, py) {
  const app = appCase(c);
  assert.deepEqual(py.shape, app.shape, `${c.name}: app/Python output shape`);
  assert.equal(py.bytes.length, app.bytes.length);
  assert.equal(py.analysis.length, c.id === "ssim" ? 2 : 1, "capture both SSIM inputs");
  for (const shape of py.analysis) assert.deepEqual(shape, app.analysis, `${c.name}: app/Python analysis size`);
  if (c.id === "ssim") {
    const appScore = Number(app.info.match(/SSIM = (-?\d+\.\d+)/)[1]);
    const pyScore = py.stdout.match(/SSIM = (-?\d+\.\d+)/);
    assert.ok(pyScore, py.stdout);
    assert.equal(Number(pyScore[1]).toFixed(3), appScore.toFixed(3), `${c.name}: score text at 3 decimals`);
    assert.equal(py.stdout.includes(FAST_NOTE), app.info.includes(FAST_NOTE), "downscale disclosure matches");
  }
  let maxDiff = 0, different = 0;
  for (let i = 0; i < app.bytes.length; i++) {
    const diff = Math.abs(app.bytes[i] - py.bytes[i]);
    maxDiff = Math.max(maxDiff, diff);
    if (diff !== 0) different++;
  }
  // All outputs are single-channel: one output byte equals one output pixel.
  return { maxDiff, different, pixels: c.w * c.h, percent: 100 * different / (c.w * c.h) };
}
function assertParity(c, stats) {
  assert.ok(stats.maxDiff <= 1 && stats.different <= stats.pixels * 0.005,
    `${c.name}: tolerant byte parity failed: max diff ${stats.maxDiff}; ${stats.different}/${stats.pixels} bytes different (${stats.percent.toFixed(6)}%); limits 1 and 0.5%`);
}
function diagnostic(t, c, stats) {
  t.diagnostic(`${c.name}: max diff=${stats.maxDiff}; different=${stats.different}/${stats.pixels} (${stats.percent.toFixed(6)}%)`);
}

// These checks still execute on machines without cv2. They catch a removed
// analysis resize, resizing only SSIM A, or accidentally downscaling full SSIM.
for (const c of CASES) test(`large-fft-ssim. ${c.name}: app shape and analysis path`, (t) => {
  const app = appCase(c);
  t.diagnostic(`output=${app.shape.join("x")}; analysis=${app.analysis.join("x")} (rows x cols); info=${app.info}`);
});
// Regressions caught: truncation, incorrect resize dimensions/interpolation,
// changed FFT filtering, SSIM formula or score text on the >640px branch.
for (const c of CASES) test(`large-fft-ssim-P. ${c.name}: max diff <=1, differing pixels <=0.5%`, PY_SKIP, (t) => {
  const stats = compare(c, pythonCase(c));
  diagnostic(t, c, stats);
  assertParity(c, stats);
});
test("large-fft-ssim-P. floor instead of rint fails the same tolerant byte parity", PY_SKIP, (t) => {
  const c = CASES[2];
  const stats = compare(c, pythonCase(c, (code) => replaceOnce(code, "np.rint(", "np.floor(")));
  diagnostic(t, { name: `${c.name} FLOOR MUTATION` }, stats);
  // Require the real image comparison to fail, not a shape/score/harness error.
  assert.throws(() => assertParity(c, stats), (err) => err.code === "ERR_ASSERTION" && /tolerant byte parity failed/.test(err.message));
});

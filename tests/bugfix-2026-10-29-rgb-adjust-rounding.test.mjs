// Reuse the two-image-order builders, fixture and raw-byte Python parity harness.
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
function step(id, p = {}) {
  const fid = OP_TO_FILTER[id] || id;
  const local = { ...Object.fromEntries(OPS[id].params.map((q) => [q.key, q.default])), ...p };
  return fid === id ? { id, params: local } : { id: fid, params: { mode: id, ...Object.fromEntries(Object.entries(local).map(([k, v]) => [id + "__" + k, v])) } };
}
function fixture() {
  const data = new Uint8Array(33 * 31 * 3);
  for (let y = 0; y < 31; y++) for (let x = 0; x < 33; x++) data.set([(x * 17 + y * 3) % 256, (x * 5 + y * 23) % 256, (x * 29 + y * 11 + 91) % 256], (y * 33 + x) * 3);
  return cv.matFromArray(31, 33, cv.CV_8UC3, data);
}
function ramp() {
  return cv.matFromArray(1, 256, cv.CV_8UC3, Uint8Array.from({ length: 256 * 3 }, (_, i) => Math.floor(i / 3)));
}
let pyAvailable = false;
try { execFileSync("python", ["-c", "import cv2,numpy"], { stdio: "pipe" }); pyAvailable = true; } catch {}
const PY_SKIP = { skip: !pyAvailable && "python/cv2/numpy unavailable" };
function pythonCase(pipe, src, mirrored = true, mutate = (code) => code) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "rgb-adjust-rounding-parity-"));
  const bgr = new cv.Mat();
  try {
    cv.cvtColor(src, bgr, cv.COLOR_RGB2BGR);
    const input = path.join(dir, "input.bin"), output = path.join(dir, "output.bin");
    fs.writeFileSync(input, Buffer.from(bgr.data));
    let code = mutate(buildPythonCode(pipe));
    code = code.slice(0, code.indexOf('if __name__ == "__main__":'));
    for (const name of ["path", "path_a", "path_b"]) code = code.replace(`cv2.imread(${name})`, `np.fromfile(${name}, dtype=np.uint8).reshape((${src.rows}, ${src.cols}, 3))`);
    const args = code.includes("def process(path_a, path_b)") ? `${JSON.stringify(input)}, ${JSON.stringify(input)}` : JSON.stringify(input);
    code += `\nr = process(${args})\nr.tofile(${JSON.stringify(output)})\nprint('SHAPE', *r.shape)\n`;
    const file = path.join(dir, "case.py"); fs.writeFileSync(file, code);
    const stdout = execFileSync("python", [file], { encoding: "utf8", env: { ...process.env, PYTHONIOENCODING: "utf8" } });
    const bytes = fs.readFileSync(output), shape = stdout.match(/SHAPE (.*)/)[1].split(" ").map(Number);
    if (mirrored && shape.length === 3) for (let i = 0; i < bytes.length; i += 3) [bytes[i], bytes[i + 2]] = [bytes[i + 2], bytes[i]];
    return { bytes, shape };
  } finally { bgr.delete(); fs.rmSync(dir, { recursive: true, force: true }); }
}
function parity(pipe, mirrored, makeSource = fixture) {
  const src = makeSource(); let out;
  try {
    out = runPipeline(src, pipe, new Map(), new Set());
    const py = pythonCase(pipe, src, mirrored);
    assert.deepEqual(py.shape, [out.mat.rows, out.mat.cols, out.mat.channels()]);
    assert.deepEqual(py.bytes, Buffer.from(out.mat.data), "0 bytes different");
  } finally { if (out) out.mat.delete(); src.delete(); }
}
const SETUPS = [["plain mirrored", [], true], ["after colorspace hsv", [step("colorspace", { space: "hsv" })], false]];

test("rgb-adjust-rounding. export rounds each float32 gain + offset before saturating", () => {
  const params = { rGain: 1.05, gGain: 0.5, bGain: 1.5, rOffset: -19, gOffset: 7.5, bOffset: 31 };
  for (const [, pre] of SETUPS) {
    const code = buildPythonCode([...pre, step("rgb_adjust", params)]);
    for (const ch of ["r", "g", "b"]) {
      assert.ok(code.includes(`${ch} = np.clip(np.rint(${ch}.astype(np.float32) * np.float32(${params[ch + "Gain"]}) + np.float32(${params[ch + "Offset"]})), 0, 255).astype(np.uint8)`), `${ch}: float32 multiply, float32 add, ties-to-even, saturation`);
    }
  }
});

test("rgb-adjust-rounding. app ties go to even and 1.05 uses float32 rather than float64", () => {
  const src = ramp(); let out;
  try {
    out = runPipeline(src, [step("rgb_adjust", { rGain: 0.5, gGain: 1.5, bGain: 1.05 })], new Map(), new Set());
    for (const [input, expected] of [[1, [0, 2, 1]], [3, [2, 4, 3]], [5, [2, 8, 5]], [30, [15, 45, 31]], [110, [55, 165, 115]], [190, [95, 255, 199]]]) {
      assert.deepEqual(Array.from(out.mat.data.slice(input * 3, input * 3 + 3)), expected, `input ${input}`);
    }
  } finally { if (out) out.mat.delete(); src.delete(); }
});

test("rgb-adjust-rounding. truncation fails on the shared fixture at gain 0.5", () => {
  const src = fixture(); let out;
  try {
    out = runPipeline(src, [step("rgb_adjust", { rGain: 0.5 })], new Map(), new Set());
    let ndiff = 0;
    for (let i = 0; i < src.data.length; i += 3) {
      const truncated = Math.trunc(Math.fround(src.data[i] * 0.5));
      if (truncated !== out.mat.data[i]) ndiff++;
    }
    assert.equal(ndiff, 256, "truncation differs at 256 red bytes; parity must catch it");
  } finally { if (out) out.mat.delete(); src.delete(); }
});

for (const gain of [0.5, 1.5, 0.95, 1.05, 2]) {
  for (const [name, pre, mirrored] of SETUPS) {
    test(`rgb-adjust-rounding-P. rGain=${gain} ${name}: 0 bytes different`, PY_SKIP,
      () => parity([...pre, step("rgb_adjust", { rGain: gain })], mirrored));
  }
}
for (const [label, params] of [
  ["all channels with signed offsets", { rGain: 0.5, gGain: 1.05, bGain: 1.5, rOffset: -19, gOffset: 7, bOffset: 31 }],
  ["fractional offsets and saturation", { rGain: 2, gGain: 0.95, bGain: 0.5, rOffset: -99.5, gOffset: 7.5, bOffset: 31.5 }],
]) {
  for (const [name, pre, mirrored] of SETUPS) {
    test(`rgb-adjust-rounding-P. ${label} ${name}: 0 bytes different`, PY_SKIP,
      () => parity([...pre, step("rgb_adjust", params)], mirrored));
  }
}
for (const gain of [0.35, 1.05, 1.45, 2.55]) {
  test(`rgb-adjust-rounding-P. full byte ramp gain=${gain}: 0 bytes different`, PY_SKIP,
    () => parity([step("rgb_adjust", { rGain: gain, gGain: gain, bGain: gain })], true, ramp));
}
// คูณแล้วบวกแยกสองขั้น (แอป) ≠ fused multiply-add ของ cv2.convertScaleAbs/addWeighted เช่น x=250 gain 0.05 offset −10 → แอป 2, cv2 3
test("rgb-adjust-rounding-P. full byte ramp gain=0.05 offset=-10 (separate multiply then add): 0 bytes different", PY_SKIP,
  () => parity([step("rgb_adjust", { rGain: 0.05, gGain: 0.05, bGain: 0.05, rOffset: -10, gOffset: -10, bOffset: -10 })], true, ramp));
test("rgb-adjust-rounding-P. removing rounding makes pixel parity fail", PY_SKIP, () => {
  const pipe = [step("rgb_adjust", { rGain: 0.5 })], src = fixture(); let out;
  try {
    out = runPipeline(src, pipe, new Map(), new Set());
    const truncated = pythonCase(pipe, src, true, (code) => code.replaceAll("np.rint(", "("));
    assert.throws(() => assert.deepEqual(truncated.bytes, Buffer.from(out.mat.data), "0 bytes different"), { code: "ERR_ASSERTION" });
    assert.equal(truncated.bytes.reduce((n, v, i) => n + (v !== out.mat.data[i]), 0), 256);
  } finally { if (out) out.mat.delete(); src.delete(); }
});

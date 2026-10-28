// Two-image export order regression; builders/fixture/parity harness reused from gray-order.
import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
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
  return fid === id ? { id, params: local } : { id: fid, params: { mode: id, ...Object.fromEntries(Object.entries(local).map(([k, v]) => [id + "__" + k, v])) } };
}
function two(id, a = [], b = []) {
  return [...a, { ...step(id), src2: { pipeline: b } }];
}
const color = (space = "hsv") => step("colorspace", { space });
function fixture() {
  const data = new Uint8Array(33 * 31 * 3);
  for (let y = 0; y < 31; y++) for (let x = 0; x < 33; x++) data.set([(x * 17 + y * 3) % 256, (x * 5 + y * 23) % 256, (x * 29 + y * 11 + 91) % 256], (y * 33 + x) * 3);
  return cv.matFromArray(31, 33, cv.CV_8UC3, data);
}
const COLOR_OPS = ["imabsdiff", "imfuse", "imarith", "hist_match", "delta_e", "roi_apply"];
const SETUPS = [
  ["A=hsv B=plain", [color()], [], false],
  ["A=plain B=hsv", [], [color()], true],
  ["A=hsv B=lab", [color()], [color("lab")], false],
];
const align = (name = "img_b") => `if ${name}.ndim == 3 and ${name}.shape[2] == 3:\n        ${name} = cv2.cvtColor(${name}, cv2.COLOR_BGR2RGB)`;

for (const id of COLOR_OPS) test(`two-image-order. ${id} swaps only mismatched three-channel B`, () => {
  for (const [, a, b] of SETUPS.slice(0, 2)) {
    const code = buildPythonCode(two(id, a, b));
    // hist_match's alignment belongs inside its color branch, after the gray reference path.
    const expected = id === "hist_match" ? align().replace("\n        ", "\n            ") : align();
    assert.ok(code.includes(expected), id + ": guarded B alignment");
    assert.equal(code.match(/img_b = cv2.cvtColor\(img_b, cv2.COLOR_BGR2RGB\)/g)?.length, 1);
  }
  assert.ok(!buildPythonCode(two(id)).includes("cv2.COLOR_BGR2RGB"));
  assert.ok(!buildPythonCode(two(id, [color()], [color("lab")])).includes("cv2.COLOR_BGR2RGB"));
});

test("two-image-order. gray consumers retain B's own gray conversion without a color swap", () => {
  for (const id of ["mask", "logical_ops", "inpaint", "psnr", "ssim", "template_matching", "seg_score", "ecc_register"]) {
    for (const [a, b, want] of [[[color()], [], "BGR"], [[], [color()], "RGB"]]) {
      const code = buildPythonCode(two(id, a, b));
      assert.match(code, new RegExp(`cvtColor\\((?:img_b|b_src), cv2.COLOR_${want}2GRAY\\)`), id);
      assert.ok(!code.includes("cv2.COLOR_BGR2RGB"), id);
    }
  }
  const gray = buildPythonCode(two("hist_match", [step("grayscale")], [color()]));
  assert.ok(gray.indexOf("hm_ref = img_b if") < gray.indexOf("cv2.COLOR_BGR2RGB"));
  assert.ok(gray.includes("else:\n        if img_b.ndim == 3 and img_b.shape[2] == 3:"));
});

test("two-image-order. nested B swaps its own reference and each parent starts from raw B", () => {
  const pipe = [...two("imabsdiff", [], two("imfuse", [color()])), ...two("hist_match", [color()], [])];
  const code = buildPythonCode(pipe);
  assert.ok(code.includes(align("img_b2")));
  assert.ok(code.includes(align()));
  assert.equal(code.match(/img_b = img_b_raw.copy\(\)/g)?.length, 2);
  assert.equal(code.match(/img_b = cv2.cvtColor\(img_b, cv2.COLOR_BGR2RGB\)/g)?.length, 2);
});

test("two-image-order. color outputs retain A order through downstream gray and nested consumers", () => {
  for (const id of COLOR_OPS.filter((id) => id !== "delta_e")) {
    for (const [, a, b, mirrored] of SETUPS) {
      const code = buildPythonCode([...two(id, a, b), step("grayscale")]);
      assert.ok(code.includes(`img = cv2.cvtColor(img, cv2.COLOR_${mirrored ? "BGR" : "RGB"}2GRAY)`), id);
    }
    const nested = buildPythonCode(two("imabsdiff", [], two(id, [color()])));
    assert.ok(nested.includes(align()), id + ": nested result remains in app order");
  }
});

// SHA-256 pins from clean d7fb708, except the five corrected app-order outputs
// authorized in Round 2 (four delta_e variants and the ROI outline).
const PINNED = [
  ["imabsdiff", {}, "8c67d363c4fcbdf05b82539418deab590eade21fc9aad70361e5350240009519", "7966df6c7f5d95f67a36a985cd53c7527939df0e853f6916e4aa89241caccaf9"],
  ["imfuse", {}, "3bc26da901cf5b157500c5a68aa75c06849e338d0b10df653e3216fe657b8126", "966c2e6b308bd86d40252e8f4dbf9b26b0a2dc60d99730d9058bc2eed6634ef2"],
  ["imfuse", { mode: "checkerboard", tileSize: 8 }, "9da3f194c3867f845d72957aed111e0899db72a9dcd01fbd4949de6be67a161b", "6fed653e4585db9a51a7fbf3f6a090ff2be4300dd54a0fe8d8545abf63f414fe"],
  ["imarith", {}, "9cbff083ee6a2e701b9782660dad23ebc05603ed0f648a65433dec594573d8d5", "55554a5f932c8066ecadd1bc4e9354eda64425c2f4ef8ad57cf3a35fb52aeb5c"],
  ["imarith", { operation: "subtract" }, "1143511461cefa3baaa3dd4e61b467884eead9b8d357195a96b2dd84d566223d", "f2b26b138ac2ac6b89d412e8d190b3be87461ea501d9d3cee1bf526157510875"],
  ["imarith", { operation: "multiply" }, "01a81ef0a0d4178af187d2ee4f64cc8a9b7e049c173e19a33a7680ecfb93129e", "ab2ace4bd964a65c7b3adff89c8600de8fbca521d3ae30966bb4809c25a8e132"],
  ["imarith", { operation: "divide" }, "ee5ba5a078e8fb4ece355f4804489eebb09616c9d32df55295ea4f59439bfa59", "335a552b361a9f638013af96246cbb4a5e5be381bee3729d97e153777ea940ba"],
  ["hist_match", {}, "64d783f2a30f7549cf457e35167e771c83c6e96ab681bb2ead735f0815d669ec", "db70c7635b737aef363853542b1ab8606f529613a51a19aa4f87ccf761d28de3"],
  ["delta_e", {}, "59b09bd7999b2a3924ff2d4cd94a38b73d1bd68439da3a6f842fe1efc16a6ae5", "9bb9abe8e9d28f968d98dc8b3b5ddd9c58f13f728887236fa37661ea4c528438"],
  ["delta_e", { scaleMax: 100 }, "a0d39c5ebf1a798726c470abe2ba5d914a4d7d244d371272f74e44ec36e96db9", "fd6ac7e72839aebcb0f4ba258aaed1033f70a17bfaea92a9c418b2a5c16d858c"],
  ["delta_e", { formula: "ciede2000", compare: "mean_b", scaleMax: 100 }, "bc0dbed4c09ae714b97abbe2bc1234a65b7052c7d40ae0562ab1356a370f9c99", "ad5c0d0bd9fd601490cad470399941ab9145e2c37a720add9876f64cb61adc3b"],
  ["delta_e", { formula: "ciede2000", view: "over", tol: 50 }, "d2ac40f00b55bbe52a2a98db123ed33a2b2634cfb02bcfe925e7508454a10d08", "f339ac9af4aa4f94f2607cfa2b67b033fd0a237ec1af499c5a358a3c0e97125e"],
  ["roi_apply", {}, "41871c2f30161a851ad16bfd4ad3300ad63466b5e729b570312acf56a070f9f0", "8638b3f8d7b93084898a6445d7264cf221e096a2c1f28c5c409b8c37ece55b22"],
  ["roi_apply", { view: "outline" }, "14748d47fd5cc698647b9bdd62973c36293fb2eb89aeffa92296329643558217", "60061ca58e6de39d475ac243357ef6e80fe91310e2e26a30eba3765714df1992"],
  ["roi_apply", { shape: "ellipse", invert: true, feather: 4 }, "1a588da91edce0cc200b23a16e2c0d82978658a9d1dc39ad112540707b768560", "46cd5723e9fde5835d2089a12f82940ad613797068c3fc1e63ae7f827c31748c"],
];
function variant(id, p, a = [], b = []) {
  const pipe = two(id, a, b);
  pipe[pipe.length - 1] = { ...step(id, p), src2: { pipeline: b } };
  return pipe;
}
test("two-image-order. delta_e converts both aligned operands using A's tracked order", () => {
  for (const [id, p] of PINNED.filter(([id]) => id === "delta_e")) {
    for (const [, a, b, mirrored] of [["plain", [], [], true], ...SETUPS]) {
      const code = buildPythonCode(variant(id, p, a, b));
      for (const operand of ["a", "b"]) {
        assert.ok(code.includes(`lab_${operand} = cv2.cvtColor(${operand}_bgr.astype(np.float32) / 255, cv2.COLOR_${mirrored ? "BGR" : "RGB"}2Lab)`));
      }
    }
    const nested = buildPythonCode(two("imabsdiff", [], variant(id, p, [color()])));
    assert.ok(nested.includes("lab_a = cv2.cvtColor(a_bgr.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)"));
  }
});
test("two-image-order. ROI outline is yellow in A's tracked order, including nested B", () => {
  for (const [, a, b, mirrored] of [["plain", [], [], true], ...SETUPS]) {
    const code = buildPythonCode(variant("roi_apply", { view: "outline" }, a, b));
    assert.ok(code.includes(`img[edge] = ${mirrored ? "(0, 255, 255)" : "(255, 255, 0)"}`));
  }
  const nested = buildPythonCode(two("imabsdiff", [], variant("roi_apply", { view: "outline" }, [color()])));
  assert.ok(nested.includes("img_b[edge] = (255, 255, 0)"));
});

const BGR_AUDIT = [
  ...[["hsv", "HSV"], ["hls", "HLS"], ["lab", "Lab"], ["ycrcb", "YCrCb"], ["xyz", "XYZ"], ["luv", "Luv"], ["yuv", "YUV"]]
    .map(([space, target]) => ["colorspace", { space }, "img", target]),
  ...["h", "s", "v"].map((channel) => ["channel_split", { channel }, "img", "HSV"]),
  ["color_inrange", { colorSpace: "hsv" }, "img", "HSV"],
  ["color_inrange", { colorSpace: "lab" }, "img", "Lab"],
  ["slic", {}, "analysis", "Lab"],
];
for (const [id, p, input, target] of BGR_AUDIT) test(`two-image-order-audit. ${id} ${JSON.stringify(p)} follows tracked order in single, main and nested B chains`, () => {
  const op = step(id, p);
  for (const [wrap, image] of [
    [(a) => a, "img"],
    [(a) => two("imabsdiff", a), "img"],
    [(b) => two("imabsdiff", [], b), "img_b"],
  ]) {
    for (const [pre, order] of [[[], "BGR"], [[color()], "RGB"], [[color(), step("grayscale")], "BGR"]]) {
      const code = buildPythonCode(wrap([...pre, op]));
      const source = input === "img" ? image : input;
      const conversions = code.match(new RegExp(`cv2\\.cvtColor\\(${source}, cv2\\.COLOR_(?:BGR|RGB)2${target}\\)`, "g"));
      assert.equal(conversions?.at(-1), `cv2.cvtColor(${source}, cv2.COLOR_${order}2${target})`, `${id}: ${image} ${order}`);
      if (id === "slic" && order === "RGB") {
        assert.ok(code.includes(`edge_color = (0, 255, 255) if ${image}.ndim == 2 else (255, 255, 0)`));
      }
    }
  }
});
for (const [id, p, mirrored, app] of PINNED) test(`two-image-order-pin. ${id} ${JSON.stringify(p)} preserves mirrored HEAD and approved app-order baseline`, () => {
  for (const [pipe, want] of [[variant(id, p), mirrored], ...(app ? [[variant(id, p, [color()], [color("lab")]), app]] : [])]) {
    assert.equal(crypto.createHash("sha256").update(buildPythonCode(pipe)).digest("hex"), want);
  }
});

// Each literal checks emitted Python behavior, including legacy two-image contexts.
const ORDER_AUDIT = [
  ["inverse HSV", [step("colorspace_inverse", { space: "hsv" })], true, ["encoded = np.ascontiguousarray(img[:, :, :3])"]],
  ...[["r", 0], ["g", 1], ["b", 2]].map(([channel, idx]) =>
    [`split ${channel}`, [step("channel_split", { channel })], true, [`img = cv2.split(img)[${idx}]`]]),
  ...[{ rGain: 2 }, { bGain: 2 }, { rGain: 2, gGain: 3, bGain: 0, rOffset: -19, gOffset: 7, bOffset: 31 }].map((p) =>
    [`adjust ${JSON.stringify(p)}`, [step("rgb_adjust", p)], false, ["r, g, b = cv2.split(img)", "img = cv2.merge([r, g, b])"]]),
  ...["additive", "speckle"].map((kind) =>
    [`noise ${kind}`, [step("gaussian_noise", { kind })], false, ["z = z.reshape(img.shape[0], img.shape[1], 3)  #"]]),
  ["wiener image-order estimates", [step("wiener_filter")], false, ['for v in np.atleast_1d(noise))  #', 'noise variance per channel (image order)']],
  ["white patch channel order", [step("white_balance", { method: "white_patch" })], false, ["for ch in (0, 1, 2):", "for ch, g in zip((0, 1, 2), gains):"]],
  ["reference point channel order", [step("white_balance", { method: "ref_point" })], false, ["for ch in (0, 1, 2)]", "for ch, g in zip((0, 1, 2), gains):"]],
  ["segmentation fresh BGR then gray", [...two("seg_score", [], [step("otsu_threshold")]), step("grayscale")], true,
    ["img[aa & ~bb] = (60, 60, 230)", "img[~aa & bb] = (255, 140, 60)", "img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"]],
];
const orderContexts = (chain) => [
  ["single", [color(), ...chain], "img"],
  ["main", [...two("imfuse"), color(), ...chain], "img"],
  ["nested B", two("imabsdiff", [], [color(), ...chain]), "img_b"],
];
for (const [name, chain, , checks] of ORDER_AUDIT) test(`two-image-order-round3. ${name} follows tracked order`, () => {
  for (const [context, pipe, image] of orderContexts(chain)) {
    const code = buildPythonCode(pipe);
    for (const check of checks) assert.ok(code.includes(check.replace(/\bimg\b/g, image)), `${name} ${context}: ${check}`);
  }
});
test("two-image-order-round3. drawing helpers use the canvas order and gray producers reset it", () => {
  for (const [id, p, check] of [
    ["line_profile", { view: "overlay" }, "cv2.line(out, (x1, y1), (x2, y2), (255, 255, 0), 1)"],
    ["roi_polygon", { view: "overlay" }, "(255, 255, 0)"],
    ["watershed", {}, "img[markers == -1] = (255, 0, 0)"],
    ["pass_fail", {}, "pf_color = (0, 200, 0) if pf_ok else (255, 0, 0)"],
    ["feret", {}, "(0, 0, 255)"],
    ["shape_filter", { view: "overlay" }, "out[bin_img > 0] = (0, 0, 255)"],
  ]) {
    for (const [context, pipe, image] of orderContexts([step(id, p)])) {
      // Pass / Fail is explicitly skipped in B chains by the existing exporter.
      if (id === "pass_fail" && context === "nested B") continue;
      assert.ok(buildPythonCode(pipe).includes(check.replace(/\bimg\b/g, image)), `${id} ${context}`);
    }
  }
});

let pyAvailable = false;
let syntaxPython = false;
try {
  execFileSync("python", ["-c", "pass"], { stdio: "pipe" }); syntaxPython = true;
  try { execFileSync("python", ["-c", "import cv2,numpy"], { stdio: "pipe" }); pyAvailable = true; } catch {}
} catch {}
const PY_SKIP = { skip: !pyAvailable && "python/cv2/numpy unavailable" };
test("two-image-order-P-syntax. mixed, same-order, gray and nested exports compile", { skip: !syntaxPython && "python unavailable" }, () => {
  const pipes = PINNED.flatMap(([id, p]) => [variant(id, p), ...SETUPS.map(([, a, b]) => variant(id, p, a, b))]);
  for (const id of COLOR_OPS) pipes.push(two("imabsdiff", [], two(id, [color()])));
  pipes.push(two("hist_match", [step("grayscale")], [color()]));
  for (const [id, p] of BGR_AUDIT) {
    const op = step(id, p);
    pipes.push([color(), op], [...two("imfuse"), color(), op], two("imabsdiff", [], [color(), op]));
  }
  for (const [, chain] of ORDER_AUDIT) pipes.push(...orderContexts(chain).map(([, pipe]) => pipe));
  execFileSync("python", ["-c", "import sys,json; [compile(c, 'two_image_order', 'exec') for c in json.loads(sys.stdin.buffer.read())]"], { input: JSON.stringify(pipes.map((p) => buildPythonCode(p))), stdio: "pipe" });
});
function pythonCase(pipe, src, mirrored = true) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "two-image-order-parity-"));
  const bgr = new cv.Mat();
  try {
    cv.cvtColor(src, bgr, cv.COLOR_RGB2BGR);
    const input = path.join(dir, "input.bin"), output = path.join(dir, "output.bin");
    fs.writeFileSync(input, Buffer.from(bgr.data));
    let code = buildPythonCode(pipe);
    code = code.slice(0, code.indexOf('if __name__ == "__main__":'));
    for (const name of ["path", "path_a", "path_b"]) code = code.replace(`cv2.imread(${name})`, `np.fromfile(${name}, dtype=np.uint8).reshape((${src.rows}, ${src.cols}, 3))`);
    const args = code.includes("def process(path_a, path_b)") ? `${JSON.stringify(input)}, ${JSON.stringify(input)}` : JSON.stringify(input);
    code += `\nr = process(${args})\nr.tofile(${JSON.stringify(output)})\nprint('SHAPE', *r.shape)\n`;
    const file = path.join(dir, "case.py"); fs.writeFileSync(file, code);
    const stdout = execFileSync("python", [file], { encoding: "utf8", env: { ...process.env, PYTHONIOENCODING: "utf8" } });
    const bytes = fs.readFileSync(output), shape = stdout.match(/SHAPE (.*)/)[1].split(" ").map(Number);
    // Unlike the gray-only cases, a color output can already be in app order.
    if (mirrored && shape.length === 3) for (let i = 0; i < bytes.length; i += 3) [bytes[i], bytes[i + 2]] = [bytes[i + 2], bytes[i]];
    return { bytes, shape, stdout };
  } finally { bgr.delete(); fs.rmSync(dir, { recursive: true, force: true }); }
}
function parity(pipe, mirrored = true, maxDiff = 0, maxNdiff = Infinity) {
  const src = fixture(); let out;
  try {
    out = runPipeline(src, pipe, new Map(), new Set());
    const py = pythonCase(pipe, src, mirrored);
    assert.deepEqual(py.shape, out.mat.channels() === 1 ? [out.mat.rows, out.mat.cols] : [out.mat.rows, out.mat.cols, out.mat.channels()]);
    if (maxDiff === 0) assert.deepEqual(py.bytes, Buffer.from(out.mat.data), "0 bytes different");
    else {
      assert.equal(py.bytes.length, out.mat.data.length);
      let ndiff = 0;
      for (let i = 0; i < py.bytes.length; i++) {
        const diff = Math.abs(py.bytes[i] - out.mat.data[i]);
        assert.ok(diff <= maxDiff, `byte ${i}: ${py.bytes[i]} vs ${out.mat.data[i]}`);
        if (diff) ndiff++;
      }
      assert.ok(ndiff <= maxNdiff, `${ndiff} bytes different (limit ${maxNdiff})`);
    }
  } finally { if (out) out.mat.delete(); src.delete(); }
}
for (const [id, p] of PINNED) {
  // Existing ip4a-P13/P14 permit one level for CIEDE2000 and feathered ROI only.
  const maxDiff = p.formula === "ciede2000" || p.feather > 0 ? 1 : 0;
  for (const [name, a, b, mirrored] of [["plain control", [], [], true], ...SETUPS]) {
    test(`two-image-order-P. ${id} ${JSON.stringify(p)} ${name}: ${maxDiff ? "existing <= 1 level" : "0 bytes different"}`, PY_SKIP,
      () => parity(variant(id, p, a, b), mirrored, maxDiff));
  }
}
for (const [name, pipe, mirrored] of [
  ["nested B result into mirrored A", two("imabsdiff", [], two("imfuse", [color()])), true],
  ["nested B result into app-order A", two("roi_apply", [color()], two("hist_match", [], [color()])), false],
  ["raw B reset between operations", [...two("imabsdiff", [color()]), ...two("imfuse")], false],
  ["gray A and color B", two("imabsdiff", [step("grayscale")], [color()]), true],
  ["color A and gray B", two("roi_apply", [color()], [step("grayscale")]), false],
  ["histogram gray A keeps independent B conversion", two("hist_match", [step("grayscale")], [color()]), true],
  ["downstream gray reads A order", [...two("imarith", [color()]), step("grayscale")], true],
]) test(`two-image-order-P. ${name}: 0 bytes different`, PY_SKIP, () => parity(pipe, mirrored));

for (const [id, p] of BGR_AUDIT) {
  const op = step(id, p), grayOut = id === "channel_split";
  // Existing HLS build-rounding allowance: tests/ip-batch2-C-2026-10-03.test.mjs:1210.
  const hls = id === "colorspace" && p.space === "hls";
  for (const [name, pipe, mirrored] of [
    ["single", [color(), op], grayOut],
    ["main", [...two("imfuse"), color(), op], grayOut],
    ["nested B", two("imabsdiff", [], [color(), op]), true],
  ]) test(`two-image-order-audit-P. ${id} ${JSON.stringify(p)} ${name}: ${hls ? "ndiff <= 4 and max <= 1" : "0 bytes different"}`, PY_SKIP,
    () => parity(pipe, mirrored, hls ? 1 : 0, hls ? 4 : 0));
}

for (const [name, chain, mirrored] of ORDER_AUDIT) {
  for (const [context, pipe] of orderContexts(chain)) {
    test(`two-image-order-round3-P. ${name} ${context}: 0 bytes different`, PY_SKIP,
      () => parity(pipe, context === "nested B" ? true : mirrored));
  }
}

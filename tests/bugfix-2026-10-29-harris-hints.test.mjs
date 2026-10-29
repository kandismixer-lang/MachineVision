import assert from "node:assert/strict";
import { test } from "node:test";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import cv from "@techstark/opencv-js";
import { OPS, FILTERS } from "../src/renderer/filters/registry.js";
import { OP_TO_FILTER } from "../src/renderer/ui/palette-panel.js";
import { buildPythonCode } from "../src/renderer/codegen.js";

while (typeof cv.Mat !== "function") await new Promise((r) => setTimeout(r, 20));
const defaults = { blockSize: 2, ksize: 3, k: 0.04, thresh: 1 };
function gray(fn, w = 160, h = 160) {
  const data = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = fn(x, y);
  return cv.matFromArray(h, w, cv.CV_8UC1, data);
}
const fixtures = {
  checkerboard: () => gray((x, y) => ((Math.floor(x / 20) + Math.floor(y / 20)) % 2) * 255),
  "L-corner": () => gray((x, y) => x >= 60 && y >= 60 ? 255 : 0),
  square: () => gray((x, y) => x >= 40 && x <= 119 && y >= 40 && y <= 119 ? 255 : 0),
  "rotated square": () => gray((x, y) => Math.abs(x - 80) + Math.abs(y - 80) <= 40 ? 255 : 0),
  "low contrast": () => gray((x, y) => x >= 40 && x <= 119 && y >= 40 && y <= 119 ? 132 : 128),
  flat: () => gray(() => 128),
  "pure line": () => gray((x) => x === 80 ? 255 : 0), // spans the image: no endpoints/corners
  "mixed contrast": () => gray((x, y) => x >= 20 && x <= 60 && y >= 20 && y <= 60 ? 200
    : x >= 100 && x <= 140 && y >= 100 && y <= 140 ? 60 : 0),
  "downscaled checkerboard": () => gray((x, y) => ((Math.floor(x / 80) + Math.floor(y / 80)) % 2) * 255, 640, 640),
};
function appPoints(src, extra = {}, palette = false) {
  const points = [], ctx = {}, observed = Object.create(cv);
  observed.circle = (...args) => { points.push([args[1].x, args[1].y]); return cv.circle(...args); };
  const p = { ...defaults, ...extra };
  const id = palette ? OP_TO_FILTER.harris_corners : "harris_corners";
  const params = palette ? { mode: "harris_corners", ...Object.fromEntries(Object.entries(p).map(([k, v]) => ["harris_corners__" + k, v])) } : p;
  const out = (palette ? FILTERS : OPS)[id].apply(observed, src, params, ctx);
  try {
    assert.equal(ctx.info, `พบมุม ${points.length} จุด`);
    return points;
  } finally { out.delete(); }
}
function assertNear(points, expected, tolerance = 3) {
  assert.equal(points.length, expected.length);
  const remaining = [...points];
  for (const [x, y] of expected) {
    const i = remaining.findIndex(([px, py]) => Math.hypot(px - x, py - y) <= tolerance);
    assert.ok(i >= 0, `missing corner near (${x}, ${y}); got ${JSON.stringify(points)}`);
    remaining.splice(i, 1);
  }
}
function assertSeparated(points, nmsRadius) {
  for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++) {
    const distance = Math.max(Math.abs(points[i][0] - points[j][0]), Math.abs(points[i][1] - points[j][1]));
    assert.ok(distance > nmsRadius, `points ${points[i]} and ${points[j]} overlap the radius-${nmsRadius} suppression window`);
  }
}

test("harris-29. checkerboard has exactly 49 inner corners, without four detections per corner", (t) => {
  const src = fixtures.checkerboard();
  try {
    for (const blockSize of [2, 3, 5]) {
      const points = appPoints(src, { blockSize });
      t.diagnostic(`8x8 checkerboard blockSize=${blockSize}: ${points.length} points; true inner corners=49`);
      const expected = [];
      for (let y = 20; y < 160; y += 20) for (let x = 20; x < 160; x += 20) expected.push([x, y]);
      assertNear(points, expected);
      assert.deepEqual(appPoints(src, { blockSize }), points, "ties must be deterministic");
    }
  } finally { src.delete(); }
});

test("harris-29. symmetric 12px checkerboard ties keep the first maximum at the export's float32 input depth", () => {
  const src = gray((x, y) => ((Math.floor(x / 12) + Math.floor(y / 12)) % 2) ? 220 : 35, 200, 150);
  try {
    const expected = [];
    for (let y = 11; y < 150; y += 12) for (let x = 11; x < 200; x += 12) expected.push([x, y]);
    const points = appPoints(src, { thresh: 60 });
    assert.equal(points.length, 192);
    assert.deepEqual(points, expected, "one-ULP differences from uint8 normalization must not move symmetric ties by 2px");
    assertSeparated(points, defaults.blockSize);
  } finally { src.delete(); }
});

test("harris-29. every reported peak is maximal throughout the blockSize-based window", () => {
  const src = gray((x, y) => (x * 17 + y * 31 + x * y * 7) & 255, 73, 61), analysis = new cv.Mat(), resp = new cv.Mat();
  try {
    src.convertTo(analysis, cv.CV_32F); // Canonical response depth shared by the app and Python export.
    for (const blockSize of [2, 5]) {
      cv.cornerHarris(analysis, resp, blockSize, 3, 0.04, cv.BORDER_DEFAULT);
      const points = appPoints(src, { blockSize });
      assert.ok(points.length > 0);
      for (const [x, y] of points) for (let yy = Math.max(0, y - blockSize); yy <= Math.min(src.rows - 1, y + blockSize); yy++) {
        for (let xx = Math.max(0, x - blockSize); xx <= Math.min(src.cols - 1, x + blockSize); xx++) {
          assert.ok(resp.data32F[y * src.cols + x] >= resp.data32F[yy * src.cols + xx] * (1 - 1e-5), `non-maximal peak (${x}, ${y}) for blockSize=${blockSize}`);
        }
      }
    }
  } finally { src.delete(); analysis.delete(); resp.delete(); }
});

test("harris-29. a broad plateau is sampled deterministically beyond the NMS suppression radius", () => {
  // Exact ties in this repeating texture exercise the app's window rule, not cross-library float equality.
  const src = gray((x, y) => ((Math.floor(x / 2) + Math.floor(y / 2)) % 2) * 255, 64, 64);
  try {
    for (const blockSize of [2, 3, 5]) {
      const points = appPoints(src, { blockSize });
      const interior = points.filter(([x, y]) => x >= 6 && x <= 58 && y >= 6 && y <= 58);
      assert.ok(interior.length > 1, "a plateau wider than the window must not swallow distant maxima");
      assertSeparated(points, blockSize);
      assert.deepEqual(appPoints(src, { blockSize }), points, "raster-order ties must be deterministic");
    }
  } finally { src.delete(); }
});

for (const [name, expected] of [
  ["L-corner", [[60, 60]]],
  ["square", [[40, 40], [119, 40], [40, 119], [119, 119]]],
  ["rotated square", [[80, 40], [40, 80], [120, 80], [80, 120]]],
  ["low contrast", [[40, 40], [119, 40], [40, 119], [119, 119]]],
  ["flat", []], ["pure line", []],
]) test(`harris-29. ${name}: true corners within 3px, no false corners on flat/line inputs`, () => {
  const src = fixtures[name]();
  try { assertNear(appPoints(src), expected); } finally { src.delete(); }
});

test("harris-29. fractional threshold survives app and palette paths and reveals weaker corners", () => {
  const src = fixtures["mixed contrast"]();
  try {
    const half = appPoints(src, { thresh: 0.5 }), one = appPoints(src, { thresh: 1 });
    assert.equal(half.length, 8);
    assert.equal(one.length, 4);
    assert.ok(half.length > one.length);
    assert.deepEqual(appPoints(src, { thresh: "0.5" }), half, "saved numeric strings preserve decimals");
    assert.deepEqual(appPoints(src, { thresh: 0.5 }, true), half, "palette dispatch preserves decimals");
    for (const thresh of [1, 2, 20, 100]) {
      assert.deepEqual(appPoints(src, { thresh }, true), appPoints(src, { thresh }), `saved integer ${thresh}`);
    }
  } finally { src.delete(); }
  for (const q of [OPS.harris_corners.params.find((p) => p.key === "thresh"),
    FILTERS[OP_TO_FILTER.harris_corners].params.find((p) => p.key === "harris_corners__thresh")]) {
    assert.equal(q.min, 0.1);
    assert.equal(q.step, 0.1);
    assert.equal(q.default, 1);
    assert.equal(q.max, 100);
  }
});

test("harris-29. Skeleton select and palette descriptions explain noise, cleanup and round blobs concisely", () => {
  const select = OPS.morphology.params.find((p) => p.key === "op").options.find((q) => q.value === "skeleton").hint;
  assert.ok(select.length + OPS.morph_skeleton.desc.length <= 443 + 402 + 120, "combined hint growth stays within the requested 120 characters of HEAD");
  for (const text of [select, OPS.morph_skeleton.desc]) {
    for (const phrase of ["10%", "142", "Median Blur", "Opening", "ก้อนกลม", "จุด", "เส้นสั้น"]) assert.ok(text.includes(phrase), phrase);
  }
});

test("harris-29. Laplacian hint explains noise gain, cleanup and RGB colour fringing", () => {
  const text = OPS.laplacian_sharpen.desc + " " + OPS.laplacian_sharpen.params[0].desc;
  for (const phrase of ["5.4", "14", "เบลอ", "Unsharp Mask", "RGB", "เหลื่อม"]) assert.ok(text.includes(phrase), phrase);
});

let pyAvailable = false, syntaxPython = false;
try {
  execFileSync("python", ["-c", "pass"], { stdio: "pipe" }); syntaxPython = true;
  try { execFileSync("python", ["-c", "import cv2,numpy"], { stdio: "pipe" }); pyAvailable = true; } catch {}
} catch {}
const PY_SKIP = { skip: !pyAvailable && "python/cv2/numpy unavailable" };
function pythonPoints(src, extra) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "harris-hints-parity-"));
  try {
    const input = path.join(dir, "input.bin"), output = path.join(dir, "points.json"), script = path.join(dir, "case.py");
    const rgb = Buffer.alloc(src.data.length * 3);
    for (let i = 0; i < src.data.length; i++) rgb.fill(src.data[i], i * 3, i * 3 + 3);
    fs.writeFileSync(input, rgb);
    let code = buildPythonCode([{ id: "harris_corners", params: { ...defaults, ...extra } }]);
    const loader = "img = cv2.imread(path)  # อ่านภาพ (BGR)";
    assert.ok(code.includes(loader));
    code = code.replace(loader, `img = np.fromfile(path, dtype=np.uint8).reshape((${src.rows}, ${src.cols}, 3))`);
    const end = code.indexOf('if __name__ == "__main__":');
    assert.ok(end > 0);
    code = code.slice(0, end);
    assert.ok(code.includes("\n    return img\n"));
    code = code.replace("\n    return img\n", "\n    with open(sys.argv[2], 'w') as f:\n        json.dump([[float(x) * inv_scale, float(y) * inv_scale] for x, y in pts], f)\n    return img\n");
    code += "\nimport sys, json\nprocess(sys.argv[1])\n";
    fs.writeFileSync(script, code);
    const stdout = execFileSync("python", [script, input, output], { encoding: "utf8", env: { ...process.env, PYTHONIOENCODING: "utf8" } });
    const points = JSON.parse(fs.readFileSync(output, "utf8"));
    assert.ok(stdout.includes(`พบมุม ${points.length} จุด`));
    return points;
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}

test("harris-29. generated Python compiles for fractional and saved integer thresholds", { skip: !syntaxPython && "python unavailable" }, () => {
  const codes = [0.1, 0.5, 1, 2, 100, "0.5"].map((thresh) => buildPythonCode([{ id: "harris_corners", params: { ...defaults, thresh } }]));
  assert.ok(codes[1].includes("thr = 0.5 / 100 * max_r"));
  for (const blockSize of [1, 2, 3, 5]) codes.push(buildPythonCode([{ id: "harris_corners", params: { ...defaults, blockSize } }]));
  execFileSync("python", ["-c", "import sys,json; [compile(c, 'harris_export', 'exec') for c in json.loads(sys.stdin.buffer.read())]"], { input: JSON.stringify(codes), stdio: "pipe" });
});

for (const name of Object.keys(fixtures)) test(`harris-29-P. ${name}: exact count and all positions within keypoints' existing 2px tolerance`, PY_SKIP, () => {
  const src = fixtures[name]();
  try {
    for (const blockSize of [2, 3, 5]) for (const thresh of [0.5, 1, 2, 100]) {
      const params = { blockSize, thresh }, app = appPoints(src, params), py = pythonPoints(src, params);
      assertSeparated(app, blockSize);
      assertSeparated(py, blockSize);
      assertNear(py, app, 2);
    }
  } finally { src.delete(); }
});

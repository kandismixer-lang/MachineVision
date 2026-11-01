// IP batch 5C: behavioral fixtures, frozen clean-tree goldens and exported Python parity.
import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import cv from "@techstark/opencv-js";
import { OPS, FILTERS, defaultParams, needsOf, passFailVerdict } from "../src/renderer/filters/registry.js";
import { TOOL_SECTIONS, OP_TO_FILTER } from "../src/renderer/ui/palette-panel.js";
import { OP_ICON, GROUP_ICON, OP_PRIMARY, OPTION_GROUPS } from "../src/renderer/filters/ui-meta.js";
import { buildPythonCode, PF_VALUE_OPS } from "../src/renderer/codegen.js";
import { assertLeakFree, assertWrapperBalance, makeThrowingCv } from "./_leak-helpers.mjs";

while (typeof cv.Mat !== "function") await new Promise((r) => setTimeout(r, 20));
const IDS = ["bf_score", "morph_ult_erode", "local_contrast", "mask_overlay"];
const defaults = (id) => Object.fromEntries(OPS[id].params.map((q) => [q.key, q.default]));
const hash = (x) => crypto.createHash("sha256").update(x).digest("hex");
const vals = (ctx) => Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));
function image(w, h, fn, ch = 1) {
  const data = new Uint8Array(w * h * ch);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = fn(x, y);
    if (ch === 1) data[y * w + x] = v;
    else data.set(v, (y * w + x) * ch);
  }
  return cv.matFromArray(h, w, ch === 1 ? cv.CV_8UC1 : ch === 3 ? cv.CV_8UC3 : cv.CV_8UC4, data);
}
function run(id, src, p = {}, ctx = {}, c = cv) { return OPS[id].apply(c, src, { ...defaults(id), ...p }, ctx); }
function step(id, p = {}) {
  const fid = OP_TO_FILTER[id] || id, local = { ...defaults(id), ...p };
  return fid === id ? { id, params: local } : { id: fid, params: { mode: id, ...Object.fromEntries(Object.entries(local).map(([k, v]) => [id + "__" + k, v])) } };
}
function same(a, b) {
  assert.deepEqual([a.cols, a.rows, a.channels()], [b.cols, b.rows, b.channels()]);
  assert.deepEqual(Buffer.from(a.data), Buffer.from(b.data));
}
const circle = (cx, cy, r) => (x, y) => (x - cx) ** 2 + (y - cy) ** 2 <= r * r ? 255 : 0;

test("ip5c-S. defaults, ranges, host placement, unique icons and Pass/Fail source", () => {
  for (const [id, host, after] of [["bf_score", "image_quality", "seg_score"], ["local_contrast", "enhance_tools", "lowlight_enhance"], ["mask_overlay", "image_arithmetic", "mask"], ["morph_ult_erode", "morph_tools", "morph_thicken"]]) {
    assert.ok(OPS[id], id);
    assert.equal(OP_TO_FILTER[id], host);
    const list = TOOL_SECTIONS.find((s) => s.ops.includes(id)).ops;
    assert.equal(list[list.indexOf(after) + 1], id);
    assert.deepEqual(FILTERS[host].params[0].options.map((o) => o.value), list);
    assert.ok(OPS[id].desc && OPS[id].modeHint);
    assert.ok(!/ที่สุด/.test(JSON.stringify(OPS[id])));
    assert.ok(/[ก-๙]/.test(OP_PRIMARY[id](defaults(id))));
    assert.equal(Object.values(OP_ICON).filter((g) => g === OP_ICON[id]).length, 1);
    assert.ok(![...Object.values(GROUP_ICON), "🧩", "📊", "📌"].includes(OP_ICON[id]));
    for (const q of OPS[id].params) {
      assert.ok(q.label && (q.desc || q.hint));
      assert.equal(defaultParams(host)[id + "__" + q.key], q.default);
    }
    if (OPTION_GROUPS[host + ".mode"]) assert.ok(OPTION_GROUPS[host + ".mode"].flatMap((g) => g[1]).includes(id));
  }
  assert.deepEqual(defaults("bf_score"), { tolPct: 0.75 });
  assert.deepEqual(defaults("local_contrast"), { edgeThreshold: 0.3, amount: 0.5 });
  assert.deepEqual(defaults("mask_overlay"), { style: "fill", color: "red", opacity: 50, thickness: 2 });
  assert.deepEqual(defaults("morph_ult_erode"), {});
  for (const [id, key, expected] of [["bf_score", "tolPct", [0.1, 5, 0.05]], ["local_contrast", "edgeThreshold", [0.01, 1, 0.01]], ["local_contrast", "amount", [-1, 1, 0.05]], ["mask_overlay", "opacity", [0, 100, 5]], ["mask_overlay", "thickness", [1, 10, 1]], ["shape_filter", "keepN", [0, 1000, 1]]]) {
    const q = OPS[id].params.find((q) => q.key === key);
    assert.deepEqual([q.min, q.max, q.step], expected);
  }
  assert.equal(defaults("shape_filter").keepN, 0);
  assert.equal(defaults("shape_filter").keepRank, "largest");
  assert.equal(OPS.morphology.params[0].options.at(-4).value, "ult_erode");
  assert.ok(PF_VALUE_OPS.has("bf_score"));
  assert.match(OP_PRIMARY.pass_fail({ metric: "bf", cond: "ge", lo: 0.9 }), /Boundary F1/);
  for (const id of IDS.slice(1)) assert.ok(!PF_VALUE_OPS.has(id));
});

test("ip5c-BF. exact boundaries, tolerant translations, empty conventions and missing B", () => {
  const a = image(200, 200, circle(80, 90, 3)), b = image(200, 200, circle(81, 90, 3));
  const partialA = image(200, 200, circle(80, 90, 2)), shifted = image(200, 200, circle(85, 90, 2));
  const far = image(200, 200, circle(110, 90, 3)), empty = image(200, 200, () => 0);
  try {
    for (const [aa, bb, expected] of [[a, a, [1, 1, 1]], [a, b, [1, 1, 1]], [a, far, [0, 0, 0]], [empty, empty, [1, 1, 1]], [a, empty, [0, 0, 1]], [empty, a, [0, 1, 0]]]) {
      const ctx = { src2: bb }, out = run("bf_score", aa, { tolPct: 0.53 }, ctx);
      assert.deepEqual([vals(ctx).bf, vals(ctx).precision, vals(ctx).recall], expected);
      assert.match(ctx.info, /BF .*precision .*recall .*1\.50 px/);
      out.delete();
    }
    const partialCtx = { src2: shifted }, partial = run("bf_score", partialA, { tolPct: 0.53 }, partialCtx);
    try {
      assert.match(partialCtx.info, /1\.50 px/);
      assert.ok(vals(partialCtx).bf > 0 && vals(partialCtx).bf < 0.2, `5 px shift: BF=${vals(partialCtx).bf}`);
    } finally { partial.delete(); }
    const square = image(3, 3, () => 255), ctx = { src2: square };
    const out = run("bf_score", square, {}, ctx);
    assert.deepEqual(Array.from(out.data.slice(12, 15)), [0, 0, 0]); // center has four foreground neighbors
    assert.deepEqual(Array.from(out.data.slice(0, 3)), [60, 200, 90]); // outside is background
    out.delete(); square.delete();
    for (const src2 of [undefined, new cv.Mat()]) {
      const ctx = { src2 }, out = run("bf_score", a, {}, ctx);
      assert.ok(!ctx.values); assert.match(ctx.info, /เข้า B|ว่าง/); out.delete(); src2?.delete();
    }
    const ctxPF = { src2: a }; run("bf_score", a, {}, ctxPF).delete();
    assert.equal(passFailVerdict({ metric: "bf", cond: "ge", lo: 0.9 }, { op: "bf_score", values: ctxPF.values }).result, "OK");
  } finally { for (const m of [a, b, partialA, shifted, far, empty]) m.delete(); }
});

test("ip5c-BF-distance. squared integer distance rejects a float32-root false match", () => {
  const a = image(20, 20, (x, y) => x === 4 && y === 4 ? 255 : 0);
  const b = image(20, 20, (x, y) => x === 5 && y === 5 ? 255 : 0);
  try {
    for (const [tolPct, expected] of [[4.99999995, 0], [5, 1]]) {
      const ctx = { src2: b }, out = run("bf_score", a, { tolPct }, ctx);
      assert.equal(vals(ctx).bf, expected);
      assert.deepEqual(Array.from(out.data.slice((4 * 20 + 4) * 3, (4 * 20 + 4) * 3 + 3)), expected ? [60, 200, 90] : [230, 60, 60]);
      out.delete();
    }
  } finally { a.delete(); b.delete(); }
});

test("ip5c-ultimate. disk center, overlapping disks, rectangular plateau and blank foreground", () => {
  const disk = image(65, 65, circle(32, 32, 12));
  const overlap = image(80, 60, (x, y) => circle(28, 30, 15)(x, y) || circle(50, 30, 15)(x, y));
  const rect = image(60, 40, (x, y) => x >= 10 && x <= 49 && y >= 10 && y <= 29 ? 255 : 0);
  const border = image(97, 65, circle(0, 32, 15));
  const blank = image(30, 30, () => 0), threshold = image(30, 30, () => 127);
  try {
    for (const [src, n] of [[disk, 1], [overlap, 2], [rect, 1], [border, 1], [blank, 0], [threshold, 0]]) {
      const ctx = {}, out = run("morph_ult_erode", src, {}, ctx), labels = new cv.Mat();
      assert.equal(cv.connectedComponents(out, labels, 8, cv.CV_32S) - 1, n);
      assert.equal(ctx.info, `พบจุดกลาง ${n} บริเวณ` + (n === 0 ? " · ไม่มีวัตถุ (ค่า > 127)" : ""));
      if (src === disk) { assert.equal(out.data[32 * 65 + 32], 255); assert.equal(cv.countNonZero(out), 1); }
      if (src === rect) { assert.ok(cv.countNonZero(out) >= 20); assert.equal(out.data[19 * 60 + 30], 255); }
      if (src === border) { assert.equal(cv.countNonZero(out), 1); assert.equal(out.data[32 * 97], 255); }
      assert.ok(out.data.every((v) => v === 0 || v === 255));
      out.delete(); labels.delete();
    }
  } finally { for (const m of [disk, overlap, rect, border, blank, threshold]) m.delete(); }
});

test("ip5c-local-depth. 16–31 px images have one detail level; below 16 stays unchanged", () => {
  for (const [size, levels] of [[15, 1], [16, 1], [20, 1], [31, 1], [32, 2]]) {
    const src = image(size, size, (x, y) => 80 + (x * 7 + y * 11) % 101);
    let out;
    try {
      const ctx = {}; out = run("local_contrast", src, { amount: 1 }, ctx);
      assert.match(ctx.info, new RegExp(` · ${levels} ชั้น`));
      if (size < 16) { same(out, src); assert.match(ctx.info, /ภาพเล็กกว่า 16 px/); }
      else assert.notDeepEqual(Buffer.from(out.data), Buffer.from(src.data), `${size}×${size} needs a detail band`);
    } finally { out?.delete(); src.delete(); }
  }
});

const localGoldenImage = () => image(64, 48, (x, y) => 32 + (3 * x + 5 * y) % 160 + (((x >> 2) ^ (y >> 2)) & 1) * 16);
test("ip5c-local-golden. positive and negative hat-weight outputs on fixed 64×48 luminance", () => {
  // Checked independently by interpolating the two bracketing remap samples at each pixel.
  const src = localGoldenImage();
  try {
    for (const [amount, expected] of [
      [0.5, "28a9ee52854b88037b4687ea61f7dcfc15e8d2362adcb73e8b21cf3e6308dba3"],
      [-0.5, "07c9eb44fa12b5ed51301385939cded2bcfaae1432999d2ea38abf226f4b932e"],
    ]) {
      const out = run("local_contrast", src, { amount, edgeThreshold: 0.3 });
      try { assert.equal(hash(out.data), expected, `amount ${amount}`); }
      finally { out.delete(); }
    }
  } finally { src.delete(); }
});

test("ip5c-local. identity, weak-detail enhancement/smoothing, strong edge and 640x480 timing", (t) => {
  const texture = image(129, 97, (x, y) => Math.floor(125 + 4 * Math.sin(x * 1.1) + 3 * Math.cos(y * 0.9) + 0.5));
  const edge = image(128, 96, (x) => x < 64 ? 0 : 200);
  const color = image(640, 480, (x, y) => [50 + x % 100, 80 + y % 100, 100 + (x + y) % 100], 3);
  const std = (m) => { const a = Array.from(m.data), mean = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - mean) ** 2, 0) / a.length); };
  try {
    for (const src of [texture, color]) { const out = run("local_contrast", src, { amount: 0 }); same(out, src); out.delete(); }
    const plus = run("local_contrast", texture), minus = run("local_contrast", texture, { amount: -0.5 });
    assert.ok(std(plus) > std(texture)); assert.ok(std(minus) < std(texture));
    const out = run("local_contrast", edge);
    // spec author 2026-10-29: tolerance 4 levels after nLev became the detail-level count (K = 10 discretisation; measured max 204)
    assert.ok(Math.max(...out.data) <= 204);
    assert.ok(out.data.every((v, i) => Math.abs(v - edge.data[i]) <= 4));
    const started = performance.now(), timed = run("local_contrast", color);
    t.diagnostic(`local_contrast 640x480 RGB: ${(performance.now() - started).toFixed(2)} ms`);
    for (const m of [plus, minus, out, timed]) m.delete();
  } finally { for (const m of [texture, edge, color]) m.delete(); }
});

test("ip5c-shape. rank after existing filters, stable ties, removed overlay and unchanged disabled export", () => {
  const centers = [15, 45, 80, 120, 165], radii = [3, 5, 8, 11, 14];
  const src = image(190, 60, (x, y) => centers.some((cx, i) => circle(cx, 30, radii[i])(x, y)) ? 255 : 0);
  const tied = image(60, 30, (x, y) => circle(15, 15, 5)(x, y) || circle(45, 15, 5)(x, y));
  try {
    for (const [keepRank, selected] of [["largest", [3, 4]], ["smallest", [0, 1]]]) {
      const ctx = {}, out = run("shape_filter", src, { prop: "none", minArea: 0, keepN: 2, keepRank }, ctx);
      assert.equal(ctx.values.fields[1].value, 2); assert.match(ctx.info, /เก็บ 2 ชิ้น/);
      for (let i = 0; i < centers.length; i++) assert.deepEqual(Array.from(out.data.slice((30 * 190 + centers[i]) * 3, (30 * 190 + centers[i]) * 3 + 3)), selected.includes(i) ? [255, 255, 255] : [255, 0, 0]);
      out.delete();
    }
    const base = run("shape_filter", src, { minArea: 0 }), large = run("shape_filter", src, { minArea: 0, keepN: 100 }); same(base, large);
    const ctx = {}, ranked = run("shape_filter", src, { prop: "circularity", lo: 0.8, minArea: 0, keepN: 1 }, ctx);
    assert.equal(ctx.values.fields[1].value, 1);
    const tie = run("shape_filter", tied, { prop: "none", minArea: 0, keepN: 1, view: "mask" });
    assert.equal(tie.data[15 * 60 + 45], 255); assert.equal(tie.data[15 * 60 + 15], 0); // RETR_EXTERNAL returns rightmost first here
    const oldStep = step("shape_filter"); delete oldStep.params.shape_filter__keepN; delete oldStep.params.shape_filter__keepRank;
    assert.equal(buildPythonCode([oldStep]), buildPythonCode([step("shape_filter", { keepN: 0 })]));
    for (const m of [base, large, ranked, tie]) m.delete();
  } finally { src.delete(); tied.delete(); }
});

test("ip5c-overlay. integer half-up blend, six colors, three labels, inward outline and resized B", () => {
  const src = image(40, 30, () => [11, 21, 31], 3);
  const mask = image(40, 30, (x, y) => x >= 3 && x <= 36 && y >= 3 && y <= 26 ? 255 : 0);
  const islands = image(40, 30, (x, y) => y >= 10 && y <= 15 && [4, 16, 28].some((a) => x >= a && x < a + 5) ? 255 : 0);
  const small = image(20, 15, () => 255);
  try {
    const zero = run("mask_overlay", src, { opacity: 0 }, { src2: mask }); same(zero, src); zero.delete();
    for (const [color, rgb] of Object.entries({ red: [230, 60, 60], green: [60, 200, 90], blue: [60, 140, 255], yellow: [240, 210, 40], cyan: [40, 210, 220], magenta: [220, 70, 200] })) {
      const out = run("mask_overlay", src, { opacity: 100, color }, { src2: mask });
      assert.deepEqual(Array.from(out.data.slice((15 * 40 + 20) * 3, (15 * 40 + 20) * 3 + 3)), rgb); out.delete();
    }
    const half = run("mask_overlay", src, {}, { src2: mask });
    assert.deepEqual(Array.from(half.data.slice((15 * 40 + 20) * 3, (15 * 40 + 20) * 3 + 3)), [121, 41, 46]); half.delete();
    const ctx = { src2: islands }, labels = run("mask_overlay", src, { style: "labels", opacity: 100 }, ctx);
    const colors = [4, 16, 28].map((x) => Array.from(labels.data.slice((12 * 40 + x) * 3, (12 * 40 + x) * 3 + 3)));
    assert.equal(new Set(colors.map((rgb) => rgb.join(","))).size, 3);
    // Component 1 -> markerLabelRGB(2), then markerLabelRGB(3) and markerLabelRGB(4).
    assert.deepEqual(colors, [[255, 55, 55], [55, 255, 113], [172, 55, 255]]);
    assert.match(ctx.info, /3 ชิ้น/); labels.delete();
    const outline = run("mask_overlay", src, { style: "outline", opacity: 100 }, { src2: mask });
    assert.deepEqual(Array.from(outline.data.slice((15 * 40 + 20) * 3, (15 * 40 + 20) * 3 + 3)), [11, 21, 31]);
    assert.deepEqual(Array.from(outline.data.slice((3 * 40 + 20) * 3, (3 * 40 + 20) * 3 + 3)), [230, 60, 60]); outline.delete();
    const rc = { src2: small }, resized = run("mask_overlay", src, {}, rc); assert.match(rc.info, /ปรับขนาด B/); resized.delete();
    // Missing/empty B keeps the RGB convention of bf_score/seg_score, including gray A.
    const gray = image(7, 5, () => 42), empty = new cv.Mat();
    try {
      for (const src2 of [undefined, empty]) {
        const missingCtx = { src2 }, out = run("mask_overlay", gray, {}, missingCtx);
        try {
          assert.equal(out.channels(), 3); assert.ok(out.data.every((v) => v === 42));
          assert.match(missingCtx.info, /เข้า B|ว่าง/);
        } finally { out.delete(); }
      }
    } finally { gray.delete(); empty.delete(); }
  } finally { for (const m of [src, mask, islands, small]) m.delete(); }
});

for (const id of [...IDS, "shape_filter"]) test(`ip5c-edge-${id}. empty, singleton, gray, RGBA and malformed params`, () => {
  const inputs = [new cv.Mat(), image(1, 1, () => 0), image(1, 1, () => 255), image(19, 17, (x, y) => (x + y) % 2 * 255), image(19, 17, () => [40, 80, 160, 37], 4)];
  try {
    for (const src of inputs) for (const p of [{}, { amount: null, edgeThreshold: "bad", tolPct: Infinity, opacity: null, thickness: -99, style: "constructor", color: "__proto__", keepN: NaN, keepRank: "constructor" }]) {
      const ctx = { src2: src }, out = run(id, src, p, ctx);
      assert.deepEqual([out.cols, out.rows], [src.cols, src.rows]); assert.ok(ctx.info); out.delete();
    }
  } finally { for (const m of inputs) m.delete(); }
});

test("ip5c-export. new exports execute measurements and ranking rather than skipping tools", () => {
  const bf = step("bf_score"); bf.src2 = { pipeline: [] };
  const overlay = step("mask_overlay", { style: "labels" }); overlay.src2 = { pipeline: [] };
  assert.match(buildPythonCode([bf]), /distanceTransform/);
  assert.match(buildPythonCode([step("local_contrast")]), /pyrDown/);
  assert.match(buildPythonCode([overlay]), /connectedComponents/);
  assert.match(buildPythonCode([step("morph_ult_erode")]), /distanceTransform/);
  assert.notEqual(buildPythonCode([step("shape_filter", { keepN: 2 })]), buildPythonCode([step("shape_filter", { keepN: 0 })]));
});

test("ip5c-export-labels. connected-B labels follow both BGR and app-order contexts", () => {
  const overlay = step("mask_overlay", { style: "labels" }); overlay.src2 = { pipeline: [] };
  assert.match(buildPythonCode([overlay]), /colors\[label\] = rgb\[::-1\]/);
  assert.match(buildPythonCode([step("colorspace", { space: "lab" }), overlay]), /colors\[label\] = rgb\n/);
});

test("ip5c-L. native heap, wrappers, injected calls and throwing context setters", () => {
  const src = image(65, 49, (x, y) => [x * 3 & 255, y * 5 & 255, (x + y) * 7 & 255], 3);
  const mask = image(65, 49, circle(32, 24, 15)), empty = new cv.Mat();
  const small = image(31, 23, circle(15, 11, 7));
  const jobs = [["bf_score", {}], ["morph_ult_erode", {}], ["local_contrast", {}], ["local_contrast", { amount: 0 }], ["mask_overlay", {}], ["mask_overlay", { style: "labels" }], ["mask_overlay", { style: "outline" }], ["shape_filter", { prop: "none", keepN: 2 }]];
  try {
    for (const [id, p] of jobs) for (const input of [src, empty]) {
      assertLeakFree(cv, id + JSON.stringify(p), (cc) => run(id, input, p, { src2: mask }, cc).delete(), { iters: 3, warmup: 1, minCreated: 0 });
    }
    for (const id of ["bf_score", "mask_overlay"]) for (const src2 of [undefined, empty]) assertLeakFree(cv, `${id}/missing B`, (cc) => run(id, src, {}, { src2 }, cc).delete(), { iters: 3, warmup: 1, minCreated: 0 });
    for (const [id, p, names] of [["bf_score", {}, ["cvtColor", "resize", "matFromArray", "distanceTransform"]], ["morph_ult_erode", {}, ["threshold", "distanceTransform", "connectedComponents"]], ["local_contrast", {}, ["convertTo", "pyrDown", "pyrUp"]], ["mask_overlay", { style: "outline" }, ["cvtColor", "threshold", "erode"]], ["mask_overlay", { style: "labels" }, ["connectedComponents"]], ["shape_filter", { prop: "none", keepN: 2 }, ["findContours", "drawContours"]]]) {
      for (const name of names) {
        // resize needs a different-sized B; convertTo is a Mat method, not a cv factory.
        if (name === "convertTo") continue;
        const src2 = name === "resize" ? small : name === "cvtColor" ? src : mask;
        assertWrapperBalance(cv, `${id}/${name}`, (cc) => run(id, src, p, { src2 }, makeThrowingCv(cc, name)).delete(), { expectThrow: /injected test failure/, minCreated: 0 });
      }
    }
    for (const [id, p] of jobs) for (const key of ["info", ...(id === "bf_score" || id === "shape_filter" ? ["values"] : [])]) {
      const ctx = { src2: mask };
      Object.defineProperty(ctx, key, { set() { throw Error("setter"); } });
      assertWrapperBalance(cv, `${id}/${key}`, (cc) => run(id, src, p, ctx, cc).delete(), { expectThrow: /setter/, minCreated: 1 });
    }
  } finally { for (const m of [src, mask, empty, small]) m.delete(); }
});

// Availability is the only parity gate. The exported operation bodies are never replaced.
let syntaxPython = false, pyAvailable = false;
try {
  execFileSync("python", ["-c", "pass"], { stdio: "pipe" }); syntaxPython = true;
  if (!process.env.IP5C_NO_PY) { execFileSync("python", ["-c", "import cv2,numpy"], { stdio: "pipe" }); pyAvailable = true; }
} catch {}
const PY_SKIP = { skip: !pyAvailable && "python/cv2 unavailable (or IP5C_NO_PY=1)" };
function variants(id) {
  if (id === "bf_score") return [{}, { tolPct: 0.1 }, { tolPct: 5 }];
  if (id === "morph_ult_erode") return [{}];
  if (id === "local_contrast") return [{}, { amount: 0 }, { amount: -0.5 }, { amount: 1 }, { amount: -1, edgeThreshold: 0.01 }, { edgeThreshold: 1 }];
  if (id === "mask_overlay") return [
    ...["fill", "labels", "outline"].flatMap((style) => [0, 50, 100].map((opacity) => ({ style, opacity }))),
    ...["red", "green", "blue", "yellow", "cyan", "magenta"].map((color) => ({ color, opacity: 35 })),
    { style: "outline", thickness: 10 },
  ];
  return ["none", "circularity", "solidity", "eccentricity", "extent"].flatMap((prop) => ["largest", "smallest"].flatMap((keepRank) => ["overlay", "mask"].map((view) => ({ prop, keepN: 2, keepRank, view, minArea: 0, lo: 0, hi: 1 }))));
}
test("ip5c-P-syntax. every branch, malformed params, sequences and BF Pass/Fail compile", { skip: !syntaxPython && "python unavailable" }, (t) => {
  const codes = [];
  for (const id of [...IDS, "shape_filter"]) for (const p of [...variants(id), { amount: null, tolPct: 999, edgeThreshold: "bad", opacity: -9, thickness: 999, style: "constructor", keepRank: "constructor", keepN: "bad" }]) {
    const st = step(id, p); if (["bf_score", "mask_overlay"].includes(id)) st.src2 = { pipeline: [] };
    codes.push(buildPythonCode([st]));
  }
  const bf = step("bf_score"); bf.src2 = { pipeline: [] };
  codes.push(buildPythonCode([bf, step("pass_fail", { metric: "bf", cond: "ge", lo: 0.9 })]));
  codes.push(buildPythonCode([step("colorspace", { space: "lab" }), step("local_contrast"), step("morph_ult_erode")]));
  execFileSync("python", ["-c", "import sys,json; codes=json.loads(sys.stdin.buffer.read()); [compile(c, 'exported_'+str(i), 'exec') for i,c in enumerate(codes)]"], { input: JSON.stringify(codes), stdio: "pipe" });
  t.diagnostic(`${codes.length} exported scripts compiled`);
});

function pythonCases(jobs) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ip5c-parity-"));
  function bytes(m) {
    const rgb = new cv.Mat(), bgr = new cv.Mat();
    try {
      // imread always supplies three channels, including for gray source fixtures.
      if (m.channels() === 1) cv.cvtColor(m, rgb, cv.COLOR_GRAY2RGB);
      else if (m.channels() === 4) cv.cvtColor(m, rgb, cv.COLOR_RGBA2RGB);
      else m.copyTo(rgb);
      cv.cvtColor(rgb, bgr, cv.COLOR_RGB2BGR); return Buffer.from(bgr.data);
    } finally { rgb.delete(); bgr.delete(); }
  }
  try {
    const manifest = jobs.map((j, i) => {
      let pipe = j.pipe || [step(j.id, j.p)];
      if (j.b) pipe.find((st) => st.id === OP_TO_FILTER[j.id]).src2 = { pipeline: [] };
      if (j.src.channels() === 1) pipe = [step("grayscale"), ...pipe];
      let code = buildPythonCode(pipe);
      code = code.slice(0, code.indexOf('if __name__ == "__main__":'));
      const inputs = [j.src, ...(j.b ? [j.b] : [])].map((m, k) => {
        const input = path.join(dir, `i${i}_${k}.bin`); fs.writeFileSync(input, bytes(m));
        const arg = j.b ? k ? "path_b" : "path_a" : "path";
        code = code.replace(`cv2.imread(${arg})`, `np.fromfile(${arg}, dtype=np.uint8).reshape((${m.rows}, ${m.cols}, 3))`);
        return input;
      });
      if (j.tail) code = code.replace("\n    return img\n", "\n    " + j.tail + "\n    return img\n");
      const file = path.join(dir, `c${i}.py`), out = path.join(dir, `o${i}.bin`);
      fs.writeFileSync(file, code); return { file, inputs, out };
    });
    fs.writeFileSync(path.join(dir, "manifest.json"), JSON.stringify(manifest));
    fs.writeFileSync(path.join(dir, "driver.py"), `import json,io,contextlib,sys
results=[]
for c in json.load(open(sys.argv[1])):
    ns={'__name__':'case'}
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(open(c['file'],encoding='utf8').read(),c['file'],'exec'),ns)
        result=ns['process'](*c['inputs'])
    result.tofile(c['out'])
    results.append({'shape':result.shape,'stdout':buf.getvalue()})
json.dump(results,open(sys.argv[2],'w'))
`);
    execFileSync("python", [path.join(dir, "driver.py"), path.join(dir, "manifest.json"), path.join(dir, "results.json")], { stdio: "pipe", env: { ...process.env, PYTHONIOENCODING: "utf8" }, maxBuffer: 1 << 25 });
    return JSON.parse(fs.readFileSync(path.join(dir, "results.json"), "utf8")).map((r, i) => {
      const data = fs.readFileSync(manifest[i].out);
      if (r.shape.length === 3) for (let k = 0; k < data.length; k += 3) [data[k], data[k + 2]] = [data[k + 2], data[k]];
      return { ...r, data };
    });
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}

for (const id of [...IDS, "shape_filter"]) test(`ip5c-P-${id}. real Python parity, all branches, RGB/gray/singleton/odd sizes`, PY_SKIP, (t) => {
  const src = image(97, 65, (x, y) => [80 + x % 80, 90 + y % 80, 40 + (x + y) % 150], 3);
  const gray = image(97, 65, (x, y) => Math.floor(120 + 5 * Math.sin(x * 1.1) + 4 * Math.cos(y * 0.9) + 0.5));
  const binary = image(97, 65, (x, y) => [15, 45, 75].some((cx, i) => circle(cx, 32, 5 + i * 3)(x, y)) ? 255 : 0);
  const one = image(1, 1, () => 255), small = image(31, 25, circle(15, 12, 8));
  const blank = image(97, 65, () => 0), full = image(19, 17, () => 255);
  const tied = image(97, 65, (x, y) => circle(25, 32, 8)(x, y) || circle(70, 32, 8)(x, y));
  const border = image(97, 65, circle(0, 32, 15));
  const detail = image(20, 20, (x, y) => 80 + (x * 7 + y * 11) % 101), golden = localGoldenImage();
  try {
    const jobs = [src, gray, binary, one, full].flatMap((input) => variants(id).map((p) => ({ id, src: input, p, ...["bf_score", "mask_overlay"].includes(id) ? { b: binary } : {} })));
    if (id === "shape_filter") for (const p of variants(id)) jobs.push({ id, src: tied, p: { ...p, keepN: 1 } });
    if (id === "morph_ult_erode") for (const input of [border, blank]) jobs.push({ id, src: input, p: {} });
    if (id === "local_contrast") for (const input of [detail, golden]) for (const amount of [0.5, -0.5, 1]) jobs.push({ id, src: input, p: { amount } });
    if (["bf_score", "mask_overlay"].includes(id)) for (const b of [blank, small, one]) for (const p of variants(id)) jobs.push({ id, src, b, p });
    if (id === "bf_score") for (const job of jobs) job.tail = "print('SCALARS', bf, precision, recall)";
    const results = pythonCases(jobs); let worst = 0, most = 0;
    for (let n = 0; n < jobs.length; n++) {
      const j = jobs[n], ctx = { src2: j.b }, out = run(id, j.src, j.p, ctx), py = results[n];
      try {
        assert.deepEqual(py.shape, out.channels() === 1 ? [out.rows, out.cols] : [out.rows, out.cols, out.channels()]);
        let max = 0, changed = 0;
        for (let i = 0; i < out.data.length; i += out.channels()) {
          let different = false;
          for (let c = 0; c < out.channels(); c++) { const d = Math.abs(out.data[i + c] - py.data[i + c]); max = Math.max(max, d); if (d) different = true; }
          if (different) changed++;
        }
        const fraction = changed / (out.rows * out.cols); worst = Math.max(worst, max); most = Math.max(most, fraction);
        assert.ok(max <= (id === "local_contrast" ? 1 : 0), `${id} ${JSON.stringify(j.p)}: max=${max}`);
        assert.ok(fraction <= (id === "local_contrast" ? 0.005 : 0), `${id} ${JSON.stringify(j.p)}: changed=${changed}`);
        if (id === "bf_score") {
          const line = py.stdout.split("\n").find((s) => s.startsWith("SCALARS "));
          const v = vals(ctx); assert.deepEqual(line.split(" ").slice(1).map(Number), [v.bf, v.precision, v.recall]);
        }
        if (id === "morph_ult_erode") {
          if (j.src === blank) assert.match(py.stdout, /no objects.*> 127/);
          else assert.doesNotMatch(py.stdout, /no objects/);
          if (j.src === border) {
            assert.equal(cv.countNonZero(out), 1);
            assert.equal(out.data[32 * 97], 255); // Maximum lies on the image border.
          }
        }
      } finally { out.delete(); }
    }
    t.diagnostic(`${jobs.length} cases; max delta ${worst}; differing pixels ${(most * 100).toFixed(4)}%`);
  } finally { for (const m of [src, gray, binary, one, small, blank, full, tied, border, detail, golden]) m.delete(); }
});

test("ip5c-PF. real Python BF Score to Pass/Fail OK and NG", PY_SKIP, () => {
  const src = image(97, 65, circle(45, 30, 15)), blank = image(97, 65, () => 0);
  try {
    const jobs = [src, blank].map((b) => ({ id: "bf_score", src, b, pipe: [step("bf_score"), step("pass_fail", { metric: "bf", cond: "ge", lo: 0.9 })] }));
    const results = pythonCases(jobs);
    assert.match(results[0].stdout, /RESULT: OK/); assert.match(results[1].stdout, /RESULT: NG/);
  } finally { src.delete(); blank.delete(); }
});

// Captured from normalized clean HEAD 402b967 before any edits: every old mode/select variant.
test("ip5c-G. 615 clean-tree cases preserve pixels, info, values and Python bytes", () => {
const ids={"image_quality":["psnr","ssim","delta_e","seg_score","piqe"],"enhance_tools":["brightness_contrast","gamma","hist_equalize","clahe","contrast_stretch","flatfield_correction","quantize","unsharp","laplacian_sharpen","custom_kernel","dehaze","lowlight_enhance"],"image_arithmetic":["imabsdiff","imfuse","mask","imarith","logical_ops","inpaint","hist_match","roi_apply"],"morph_tools":["morph_erode","morph_dilate","morph_open","morph_close","morph_gradient","morph_tophat","morph_blackhat","morph_fill_holes","morph_clear_border","morph_remove_small","morph_clean","morph_majority","morph_skeleton","morph_spur","morph_thin","morph_shrink","morph_reconstruct_open","morph_convex_hull","morph_endpoints_branchpoints","morph_hit_or_miss","morph_regional_max","morph_regional_min","morph_h_maxima","morph_h_minima","morph_extended_max","morph_extended_min"],"analysis_detect":["shape_filter"]};
const result={};
for(const [kind,w,h] of [['rgb',37,29],['gray',37,29],['rgba',37,29],['rgb',1,1],['empty',0,0]]) {
 const data=Uint8Array.from({length:w*h*3},(_,i)=>(Math.imul(i+11,73)^(i>>4)*17)&255);
 const c=w?cv.matFromArray(h,w,cv.CV_8UC3,data):new cv.Mat(),src=new cv.Mat();
 if(kind==='gray')cv.cvtColor(c,src,cv.COLOR_RGB2GRAY); else if(kind==='rgba')cv.cvtColor(c,src,cv.COLOR_RGB2RGBA); else c.copyTo(src);
 try { for(const [fid,ops] of Object.entries(ids)) for(const id of ops) {
 const defs=Object.fromEntries(OPS[id].params.filter(q=>!['keepN','keepRank'].includes(q.key)&&!(id==='inpaint'&&['patch','search'].includes(q.key))).map(q=>[q.key,q.default]));
 const variants=[{}];
 for(const q of OPS[id].params.filter(q=>q.type==='select'&&q.key!=='keepRank'&&!(id==='inpaint'&&q.key==='patch')))for(const {value}of q.options.filter(o=>!["cie94","angle"].includes(o.value)&&!(id==='inpaint'&&['laplace','exemplar'].includes(o.value))&&!(id.startsWith('morph_')&&q.key==='shape'&&o.value==='ball')))if(value!==q.default)variants.push({[q.key]:value});
 if(id==='shape_filter')variants.push({keepN:0},{prop:'none',minArea:0},{view:'mask',minArea:0});
 for(const extra of variants){
 const local={...defs,...extra},params={mode:id,...Object.fromEntries(Object.entries(local).map(([k,v])=>[id+'__'+k,v]))},st={id:fid,params};
 const key=JSON.stringify([kind,w,h,fid,id,extra]),ctx={src2:c};let g=null;
 try{if(w&&needsOf(FILTERS[fid],params)==='gray'&&src.channels()>1){g=new cv.Mat();cv.cvtColor(src,g,src.channels()===4?cv.COLOR_RGBA2GRAY:cv.COLOR_RGB2GRAY);}
 const out=FILTERS[fid].apply(cv,g||src,params,ctx);result[key]={image:hash(Buffer.from(out.data)),rows:out.rows,cols:out.cols,ch:out.channels(),info:ctx.info,values:ctx.values,python:hash(buildPythonCode([st]))};out.delete();
 }catch(e){result[key]={error:typeof e==='number'?'OpenCV native exception':String(e),python:hash(buildPythonCode([st]))};}finally{if(g)g.delete();}
 }
 }}finally{src.delete();c.delete();}
}

assert.equal(Object.keys(result).length,615);
assert.equal(hash(JSON.stringify(result)),"34be796098b3e5b8f59ce25ef9e07b5a876286e77e8ef685ee01833a87658553");
});

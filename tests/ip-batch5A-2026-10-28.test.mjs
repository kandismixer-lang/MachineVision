// IP batch 5A: independent numeric fixtures, old-output goldens, leak gates and real exported-Python parity.
// สเปก 2026-10-28 · รอบแก้รีวิว: snapshot ขอบ shrink, สูตร HSV ร่วม และทดสอบ Python จริง
import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import cv from "@techstark/opencv-js";
import { OPS, FILTERS, MORPH_OP_IDS, MORPH_5A_LUT, defaultParams, needsOf, passFailVerdict } from "../src/renderer/filters/registry.js";
import { TOOL_SECTIONS, OP_TO_FILTER } from "../src/renderer/ui/palette-panel.js";
import { GROUP_ICON, OP_PRIMARY, OPTION_GROUPS } from "../src/renderer/filters/ui-meta.js";
import { buildPythonCode, PF_VALUE_OPS } from "../src/renderer/codegen.js";
import { assertLeakFree, assertWrapperBalance, makeThrowingCv } from "./_leak-helpers.mjs";
while (typeof cv.Mat !== "function") await new Promise((r) => setTimeout(r, 20));
const IDS = ["watershed_marker", "seg_score", "lowlight_enhance", "colorspace_inverse", "euler_number"];
const MORPH = ["clean", "majority", "spur", "thin", "shrink", "h_minima", "extended_min"];
const ALL = [...IDS, ...MORPH.map((x) => "morph_" + x)];
const OLD_MORPH = ["erode", "dilate", "open", "close", "gradient", "tophat", "blackhat", "fill_holes", "clear_border", "remove_small", "skeleton", "reconstruct_open", "convex_hull", "endpoints_branchpoints", "hit_or_miss", "regional_max", "regional_min", "h_maxima", "extended_max"];
const defaults = (id) => Object.fromEntries(OPS[id].params.map((q) => [q.key, q.default]));
// ภาพ deterministic และตัวช่วยรัน op — ทุก Mat ที่สร้างต้องล้างใน finally
function gray(w, h, fn) {
  const data = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = fn(x, y);
  return cv.matFromArray(h, w, cv.CV_8UC1, data);
}
function rgb(w, h, fn) {
  const data = new Uint8Array(w * h * 3);
  // Keep the original per-channel callback sequence so seeded fixtures do not change.
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    for (let channel = 0; channel < 3; channel++) data[(y * w + x) * 3 + channel] = fn(x, y)[channel];
  }
  return cv.matFromArray(h, w, cv.CV_8UC3, data);
}
function noise(w, h, lo = 0, span = 256) {
  let s = 71;
  return rgb(w, h, () => Array.from({ length: 3 }, () => {
    s = Math.imul(s, 1664525) + 1013904223 >>> 0;
    return lo + Math.floor(s / 4294967296 * span);
  }));
}
const vals = (ctx) => Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));
function run(id, src, p = {}, ctx = {}, c = cv) {
  return OPS[id].apply(c, src, { ...defaults(id), ...p }, ctx);
}
function same(a, b, label = "bytes") {
  assert.equal(a.rows, b.rows, label);
  assert.equal(a.cols, b.cols, label);
  assert.equal(a.channels(), b.channels(), label);
  assert.deepEqual(Buffer.from(a.data), Buffer.from(b.data), label);
}
const hash = (x) => crypto.createHash("sha256").update(x).digest("hex");
function step(id, p = {}) {
  const fid = OP_TO_FILTER[id] || id;
  return fid === id ? { id, params: { ...defaults(id), ...p } } : { id: fid, params: { ...defaultParams(fid), mode: id, ...Object.fromEntries(Object.entries(p).map(([k, v]) => [id + "__" + k, v])) } };
}
function topology(src, conn = "8") {
  const ctx = {};
  const out = run("euler_number", src, { conn }, ctx);
  out.delete();
  return vals(ctx);
}
function rings() {
  return gray(45, 20, (x, y) => {
    if (x >= 2 && x <= 12 && y >= 2 && y <= 16) return x >= 5 && x <= 9 && y >= 6 && y <= 12 ? 0 : 255;
    if (x >= 17 && x <= 23 && y >= 4 && y <= 13) return 255;
    if (x >= 28 && x <= 40 && y >= 2 && y <= 16) return x >= 32 && x <= 36 && (y >= 5 && y <= 7 || y >= 11 && y <= 13) ? 0 : 255;
    return 0;
  });
}
function touching() {
  return rgb(95, 65, (x, y) => {
    const v = (x - 33) ** 2 + (y - 32) ** 2 <= 20 ** 2 || (x - 61) ** 2 + (y - 32) ** 2 <= 20 ** 2 ? 20 : 240;
    return [v, v, v];
  });
}
// Mat lifetime — เส้นทางปกติ, ภาพขอบกรณี และ injected exception
test("ip5a-L1. all new ops: wrapper balance and heap census across gray / RGB / RGBA / 1x1 / 0x0, selects and new morphology branches", () => {
  const g = rings(), c = noise(15, 11), rgba = new cv.Mat(), one = gray(1, 1, () => 255), empty = new cv.Mat(), touch = touching();
  cv.cvtColor(c, rgba, cv.COLOR_RGB2RGBA);
  const jobs = [];
  for (const id of ALL) for (const src of [g, c, rgba, one, empty]) jobs.push([id, src, {}, id === "seg_score" ? { src2: c } : {}]);
  for (const view of ["boundaries", "labels"]) for (const polarity of ["dark", "bright"]) jobs.push(["watershed_marker", touch, { view, polarity }, {}]);
  for (const refine of [true, false]) for (const alphaBlend of [true, false]) jobs.push(["lowlight_enhance", rgba, { refine, alphaBlend }, {}]);
  for (const conn of ["4", "8"]) jobs.push(["euler_number", g, { conn }, {}]);
  for (const space of OPS.colorspace.params[0].options) jobs.push(["colorspace_inverse", c, { space: space.value }, {}]);
  for (const id of ["morph_spur", "morph_thin", "morph_shrink"]) jobs.push([id, g, { iterations: 0 }, {}]);
  jobs.push(["seg_score", g, {}, {}], ["seg_score", g, {}, { src2: empty }]);
  try {
    for (const [id, src, p, ctx] of jobs) assertLeakFree(cv, id, (cc) => run(id, src, p, { ...ctx }, cc).delete(), { iters: 4, warmup: 2, minCreated: 0 });
  } finally {
    for (const m of [g, c, rgba, one, empty, touch]) m.delete();
  }
});
test("ip5a-L2. injected cv failures and throwing info / values setters reclaim every Mat, including an already-created output", () => {
  const src = touching(), b = rings();
  try {
    const calls = { watershed_marker: ["cvtColor", "threshold", "getStructuringElement", "morphologyEx", "distanceTransform", "matFromArray", "connectedComponents", "dilate", "watershed"], seg_score: ["cvtColor", "resize"], lowlight_enhance: ["bitwise_not", "getStructuringElement", "erode", "matFromArray", "boxFilter", "multiply"], colorspace_inverse: ["cvtColor"], euler_number: ["cvtColor", "threshold", "connectedComponents", "bitwise_not"] };
    for (const [id, names] of Object.entries(calls)) for (const name of names) assertWrapperBalance(cv, id + " " + name, (cc) => run(id, src, id === "colorspace_inverse" ? { space: "hls" } : {}, { src2: b }, makeThrowingCv(cc, name)).delete(), { expectThrow: /injected test failure/, minCreated: 0 });
    for (const id of ALL) {
      assertWrapperBalance(cv, id + " info", (cc) => run(id, src, {}, { src2: b, set info(v) {
        throw Error("setter");
      } }, cc).delete(), { expectThrow: /setter/, minCreated: 1 });
      if (["watershed_marker", "seg_score", "euler_number"].includes(id)) assertWrapperBalance(cv, id + " values", (cc) => run(id, src, {}, { src2: b, set values(v) {
        throw Error("setter");
      } }, cc).delete(), { expectThrow: /setter/, minCreated: 1 });
    }
    for (const id of MORPH.map((x) => "morph_" + x)) assertWrapperBalance(cv, id + " factory", (cc) => run(id, src, {}, {}, makeThrowingCv(cc, "matFromArray")).delete(), { expectThrow: /injected test failure/, minCreated: 0 });
  } finally {
    src.delete();
    b.delete();
  }
});
test("ip5a-S1. registration, placement, shared inverse options, params, groups, icons and Pass/Fail sources", () => {
  const host = { watershed_marker: "segment_edge", seg_score: "image_quality", lowlight_enhance: "enhance_tools", colorspace_inverse: "color_tools", euler_number: "analysis_detect" };
  for (const id of ALL) {
    assert.equal(OP_TO_FILTER[id], host[id] || "morph_tools");
    assert.ok(OPS[id].desc.length > 30);
    assert.ok(OPS[id].modeHint);
    assert.equal(typeof OP_PRIMARY[id], "function");
    assert.equal(TOOL_SECTIONS.filter((s) => s.ops.includes(id)).length, 1);
    assert.ok(!/ที่สุด/.test(JSON.stringify({ name: OPS[id].name, desc: OPS[id].desc, hint: OPS[id].modeHint, params: OPS[id].params })));
    for (const q of OPS[id].params) {
      assert.notEqual(q.key, "mode");
      assert.equal(defaultParams(OP_TO_FILTER[id])[id + "__" + q.key], q.default);
    }
    const py = buildPythonCode([step(id)]);
    assert.ok(!/WARNING: skipped|np\.round|\bround\(|import math|math\./.test(py));
    for (const line of py.split("\n").filter((l) => l.trimStart().startsWith("print("))) assert.ok(/^[\x00-\x7f]*$/.test(line), line);
  }
  assert.equal(OPS.colorspace_inverse.params[0].options, OPS.colorspace.params[0].options);
  assert.equal(OPTION_GROUPS["colorspace_inverse.space"], OPTION_GROUPS["colorspace.space"]);
  const after = (title, a, b) => {
    const xs = TOOL_SECTIONS.find((s) => s.title === title).ops;
    assert.equal(xs[xs.indexOf(a) + 1], b);
  };
  after("COLOR", "colorspace", "colorspace_inverse");
  after("ENHANCE", "dehaze", "lowlight_enhance");
  after("SEGMENTATION & EDGE", "watershed", "watershed_marker");
  after("ANALYSIS & DETECTION", "regionprops", "euler_number");
  for (const [a, b] of [["remove_small", "clean"], ["clean", "majority"], ["skeleton", "spur"], ["spur", "thin"], ["thin", "shrink"], ["h_maxima", "h_minima"], ["extended_max", "extended_min"]]) after("MORPHOLOGY", "morph_" + a, "morph_" + b);
  assert.deepEqual(OPS.morphology.params[0].options.map((o) => o.value), [...OLD_MORPH, ...MORPH]);
  assert.equal(FILTERS.image_quality.inputs, 2);
  for (const id of ["watershed_marker", "seg_score", "euler_number"]) assert.ok(PF_VALUE_OPS.has(id));
  assert.ok(!PF_VALUE_OPS.has("lowlight_enhance"));
  const src = fs.readFileSync(new URL("../src/renderer/filters/ui-meta.js", import.meta.url), "utf8");
  const body = src.slice(src.indexOf("const OP_ICON = {"), src.indexOf("};", src.indexOf("const OP_ICON = {")));
  const icons = Object.fromEntries([...body.matchAll(/([a-z_0-9]+): "([^"]+)"/g)].map((m) => [m[1], m[2]]));
  const files = [];
  function walk(dir) {
    for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, d.name);
      if (d.isDirectory()) walk(p);
      else if (d.name !== "ui-meta.js" && /\.(js|html|css)$/.test(d.name)) files.push(fs.readFileSync(p, "utf8"));
    }
  }
  walk(fileURLToPath(new URL("../src", import.meta.url)));
  for (const id of ALL) {
    const glyph = icons[id];
    assert.ok(glyph);
    assert.equal(Object.values(icons).filter((x) => x === glyph).length, 1);
    assert.ok(![...Object.values(GROUP_ICON), "🧩", "📊", "📌"].includes(glyph));
    for (const f of files) assert.ok(!f.includes(glyph), id + " icon appears elsewhere");
  }
  assert.equal(defaults("watershed_marker").hHeight, 2);
  assert.equal(defaults("lowlight_enhance").amount, 1);
  assert.equal(defaults("euler_number").conn, "8");
  for (const id of ["morph_spur", "morph_thin", "morph_shrink"]) {
    const q = OPS[id].params[0];
    assert.deepEqual([q.key, q.min, q.max, q.default], ["iterations", 0, 100, id === "morph_spur" ? 1 : 0]);
  }
});
test("ip5a-E1. empty / singleton / color inputs are safe, outputs are new, sources immutable, numeric and unknown-select fallbacks agree", () => {
  const empty = new cv.Mat(), one = gray(1, 1, () => 128), c = noise(9, 7), rgba = new cv.Mat();
  cv.cvtColor(c, rgba, cv.COLOR_RGB2RGBA);
  try {
    for (const id of ALL) for (const src of [empty, one, c, rgba]) {
      const before = Buffer.from(src.data), ctx = { src2: one }, out = run(id, src, {}, ctx);
      assert.notEqual(out, src);
      assert.deepEqual(Buffer.from(src.data), before);
      assert.equal(out.rows, src.rows);
      assert.equal(out.cols, src.cols);
      if (src === empty) {
        assert.equal(ctx.info, "ภาพว่าง");
        if (ctx.values) assert.ok(Object.values(vals(ctx)).every((v) => v === null));
      } else if (id.startsWith("morph_")) assert.equal(out.channels(), 1);
      else if (id !== "lowlight_enhance") assert.equal(out.channels(), 3);
      out.delete();
    }
    for (const [id, a, b] of [["watershed_marker", { hHeight: 900, view: "constructor", polarity: "constructor" }, { hHeight: 50 }], ["colorspace_inverse", { space: "constructor" }, {}], ["euler_number", { conn: "constructor" }, {}], ["lowlight_enhance", { amount: 9, patch: -9, refine: "junk" }, { amount: 1, patch: 3 }], ["morph_thin", { iterations: -3 }, { iterations: 0 }], ["morph_h_minima", { hHeight: 999 }, { hHeight: 255 }]]) {
      const x = run(id, c, a), y = run(id, c, b);
      same(x, y, id);
      x.delete();
      y.delete();
    }
  } finally {
    empty.delete();
    one.delete();
    c.delete();
    rgba.delete();
  }
});
test("ip5a-seg. TP=2 FP=1 FN=1 TN=2, exact overlay, empty conventions, nearest B resize, missing B and IoU/Dice Pass/Fail", () => {
  const a = gray(6, 1, (x) => [255, 255, 255, 0, 0, 127][x]), b = gray(6, 1, (x) => [255, 255, 0, 255, 0, 127][x]), zero = gray(6, 1, () => 0), small = gray(2, 1, (x) => x ? 0 : 255);
  try {
    const ctx = { src2: b }, out = run("seg_score", a, {}, ctx);
    assert.deepEqual(vals(ctx), { iou: 0.5, dice: 2 / 3, precision: 2 / 3, recall: 2 / 3, tp: 2, fp: 1, fn: 1 });
    assert.deepEqual(Array.from(out.data), [255, 255, 255, 255, 255, 255, 230, 60, 60, 60, 140, 255, 0, 0, 0, 0, 0, 0]);
    out.delete();
    assert.match(ctx.info, /IoU 0.5000 · Dice 0.6667/);
    const ec = { src2: zero };
    run("seg_score", zero, {}, ec).delete();
    assert.deepEqual(vals(ec), { iou: 1, dice: 1, precision: 1, recall: 1, tp: 0, fp: 0, fn: 0 });
    const onlyB = { src2: b };
    run("seg_score", zero, {}, onlyB).delete();
    assert.equal(vals(onlyB).precision, 1);
    assert.equal(vals(onlyB).recall, 0);
    assert.match(OPS.seg_score.desc, /ตัวหารเป็น 0 → ถือว่า 1/);
    const emptyB = new cv.Mat();
    try {
      const ctx2 = { src2: emptyB };
      run("seg_score", a, {}, ctx2).delete();
      assert.match(ctx2.info, /ภาพที่ต่อเข้า 'เข้า B' ว่าง \(0×0 px\)/);
      assert.equal(ctx2.values, void 0);
    } finally {
      emptyB.delete();
    }
    const rc = { src2: small };
    run("seg_score", a, {}, rc).delete();
    assert.equal(vals(rc).iou, 1);
    assert.match(rc.info, /ปรับขนาด B/);
    const missing = { values: { stale: true } };
    const mo = run("seg_score", a, {}, missing);
    assert.equal(mo.channels(), 3);
    assert.ok(!("values" in missing));
    assert.match(missing.info, /เข้า B/);
    mo.delete();
    for (const metric of ["iou", "dice"]) {
      assert.equal(passFailVerdict({ metric, cond: "ge", lo: 0.6 }, { op: "seg_score", values: ctx.values }).result, metric === "iou" ? "NG" : "OK");
      const ss = step("seg_score");
      ss.src2 = { pipeline: [] };
      const code = buildPythonCode([ss, step("pass_fail", { metric, cond: "ge", lo: 0.6 })]);
      assert.ok(!code.includes("WARNING: skipped"));
      assert.ok(code.includes(`pf_measured = pf_up['${metric}']`));
    }
  } finally {
    for (const m of [a, b, zero, small]) m.delete();
  }
});
test("ip5a-euler. ring + solid + figure eight = 3 objects, 3 holes, Euler 0; edge-connected background and dual connectivity", () => {
  const src = rings(), diagonal = gray(2, 2, (x, y) => x === y ? 255 : 0), edge = gray(7, 7, (x, y) => x === 0 || y === 0 || x === 6 || y === 6 ? x === 3 && y === 0 ? 0 : 255 : 0);
  try {
    assert.deepEqual(topology(src), { objects: 3, holes: 3, euler: 0 });
    assert.deepEqual(topology(diagonal, "8"), { objects: 1, holes: 0, euler: 1 });
    assert.deepEqual(topology(diagonal, "4"), { objects: 2, holes: 0, euler: 2 });
    assert.equal(topology(edge).holes, 0);
    const o = run("euler_number", src);
    assert.deepEqual(Array.from(o.data.slice((7 * 45 + 6) * 3, (7 * 45 + 6) * 3 + 3)), [230, 60, 60]);
    assert.deepEqual(Array.from(o.data.slice((2 * 45 + 2) * 3, (2 * 45 + 2) * 3 + 3)), [200, 200, 200]);
    o.delete();
    const dual = gray(3, 3, (x, y) => x === 1 && y === 1 || x === 0 && y === 0 ? 0 : 255);
    assert.equal(topology(dual, "8").holes, 1);
    assert.equal(topology(dual, "4").holes, 0);
    dual.delete();
  } finally {
    src.delete();
    diagonal.delete();
    edge.delete();
  }
});
test("ip5a-lowlight. exact inverse/dehaze/inverse identity, luma blending, gray preservation, no values, and measured improvement on a dark scene", (t) => {
  const src = noise(37, 29, 0, 75), g = gray(29, 17, (x, y) => (x * 3 + y * 7) % 80);
  try {
    for (const x of [src, g]) for (const refine of [false, true]) for (const alphaBlend of [false, true]) {
      const inv = new cv.Mat();
      cv.bitwise_not(x, inv);
      const dh = run("dehaze", inv, { amount: 1, patch: 7, refine });
      const expected = new cv.Mat();
      cv.bitwise_not(dh, expected);
      const lum = new cv.Mat();
      if (x.channels() === 1) x.copyTo(lum);
      else cv.cvtColor(x, lum, cv.COLOR_RGB2GRAY);
      if (alphaBlend) {
        const ld = lum.data, xd = x.data, ed = expected.data, ch = x.channels();
        for (let i = 0; i < ld.length; i++) for (let c = 0; c < ch; c++) {
          const j = i * ch + c, a = ld[i] / 255;
          ed[j] = Math.floor(a * xd[j] + (1 - a) * ed[j] + 0.5);
        }
      }
      const ctx = {}, o = run("lowlight_enhance", x, { patch: 7, refine, alphaBlend }, ctx);
      same(o, expected);
      assert.equal(o.channels(), x.channels());
      assert.equal(ctx.values, void 0);
      if (x === src && !alphaBlend && !refine) {
        const mean = (a) => Array.from(a.data).reduce((u, v) => u + v, 0) / a.data.length;
        t.diagnostic(`lowlight mean ${mean(x).toFixed(3)} -> ${mean(o).toFixed(3)}`);
        assert.ok(mean(o) > mean(x) + 10);
      }
      for (const m of [o, inv, dh, expected, lum]) m.delete();
    }
  } finally {
    src.delete();
    g.delete();
  }
});
// Round-trip errors วัดจริงจาก fixture เดิมทั้ง full range และ RGB 32..223
test("ip5a-inverse. every forward/inverse color pair, measured full-range and interior-gamut round-trip maxima, gray hint and regenerated channel order", (t) => {
  const src = noise(128, 96), interior = noise(128, 96, 32, 192), g = gray(3, 2, () => 70);
  try {
    const measured = {};
    for (const x of [src, interior]) for (const { value: space } of OPS.colorspace.params[0].options) {
      const enc = run("colorspace", x, { space }), out = run("colorspace_inverse", enc, { space });
      let max = 0;
      for (let i = 0; i < x.data.length; i++) max = Math.max(max, Math.abs(x.data[i] - out.data[i]));
      (measured[x === src ? "full" : "interior"] ||= {})[space] = max;
      assert.ok(max <= { hsv: 5, hls: 5, lab: 24, ycrcb: 2, xyz: 24, luv: 30, yuv: 40 }[space], space + " max " + max);
      enc.delete();
      out.delete();
    }
    t.diagnostic("roundtrip " + JSON.stringify(measured));
    const ctx = {}, o = run("colorspace_inverse", g, {}, ctx);
    assert.match(ctx.info, /ภาพเข้าเป็นภาพเทา/);
    assert.equal(o.channels(), 3);
    o.delete();
    const code = buildPythonCode([step("colorspace", { space: "lab" }), step("colorspace_inverse", { space: "lab" }), step("lowlight_enhance", { refine: true })]);
    assert.match(code, /cv2.COLOR_Lab2BGR/);
    assert.ok(code.includes("cv2.COLOR_BGR2GRAY"));
  } finally {
    src.delete();
    interior.delete();
    g.delete();
  }
});
test("ip5a-hsv-formula. six sectors, hue wrap, zero saturation and half-up rounding have exact RGB bytes", () => {
  const hsv = [
    [0, 255, 255],
    [30, 255, 255],
    [60, 255, 255],
    [90, 255, 255],
    [120, 255, 255],
    [150, 255, 255],
    [180, 255, 255],
    [255, 0, 73],
    [15, 255, 255]
  ];
  const src = rgb(hsv.length, 1, (x) => hsv[x]);
  const out = run("colorspace_inverse", src, { space: "hsv" });
  try {
    assert.deepEqual(Array.from(out.data), [
      255,
      0,
      0,
      255,
      255,
      0,
      0,
      255,
      0,
      0,
      255,
      255,
      0,
      0,
      255,
      255,
      0,
      255,
      255,
      0,
      0,
      73,
      73,
      73,
      255,
      128,
      0
    ]);
  } finally {
    src.delete();
    out.delete();
  }
});
test("ip5a-marker. two touching discs yield two markers, polarity mirrors, labels deterministic, h suppresses weak peaks, no foreground passes RGB through", (t) => {
  const src = touching(), inv = new cv.Mat(), blank = rgb(7, 7, () => [240, 240, 240]);
  cv.bitwise_not(src, inv);
  try {
    for (const view of ["boundaries", "labels"]) {
      const ctx2 = {}, out = run("watershed_marker", src, { view, hHeight: 2 }, ctx2);
      assert.equal(vals(ctx2).objects, 2);
      assert.match(ctx2.info, /แยกได้ 2 วัตถุ/);
      const other = run("watershed_marker", inv, { view, polarity: "bright" });
      if (view === "labels") same(out, other);
      out.delete();
      other.delete();
    }
    const counts = [];
    for (const hHeight of [1, 2, 10, 50]) {
      const ctx2 = {};
      run("watershed_marker", src, { hHeight }, ctx2).delete();
      counts.push(vals(ctx2).objects);
    }
    t.diagnostic("marker counts h=1/2/10/50 " + counts.join("/"));
    assert.ok(counts.at(-1) <= counts[0]);
    const ctx = {}, o = run("watershed_marker", blank, {}, ctx);
    same(o, blank);
    assert.equal(vals(ctx).objects, 0);
    assert.match(ctx.info, /ไม่พบวัตถุ/);
    o.delete();
  } finally {
    src.delete();
    inv.delete();
    blank.delete();
  }
});
// Independent 3x3 component counter for checking every one of the 256 topology cases.
function components3(a, value, conn, cardinalOnly = false) {
  const seen = new Set();
  let n = 0;
  for (let i = 0; i < 9; i++) {
    if (i === 4 || a[i] !== value || seen.has(i)) continue;
    const q = [i];
    seen.add(i);
    let cardinal = false;
    for (let k = 0; k < q.length; k++) {
      const j = q[k], x = j % 3, y = Math.floor(j / 3);
      cardinal ||= Math.abs(x - 1) + Math.abs(y - 1) === 1;
      for (let v = 0; v < 9; v++) {
        if (v === 4 || seen.has(v) || a[v] !== value) continue;
        const dx = Math.abs(v % 3 - x), dy = Math.abs(Math.floor(v / 3) - y);
        if (conn === 8 ? Math.max(dx, dy) === 1 : dx + dy === 1) {
          seen.add(v);
          q.push(v);
        }
      }
    }
    if (!cardinalOnly || cardinal) n++;
  }
  return n;
}
test("ip5a-morph-LUT. exhaustive independent Lam–Lee–Suen conditions and simple-point 8/4 topology for all 256 neighborhoods", () => {
  for (let mask = 0; mask < 256; mask++) {
    const x = Array.from({ length: 8 }, (_, i) => mask >> i & 1), sum = x.reduce((a2, b) => a2 + b, 0);
    assert.equal(MORPH_5A_LUT.count[mask], sum);
    const a = [x[3], x[2], x[1], x[4], 1, x[0], x[5], x[6], x[7]];
    assert.equal(MORPH_5A_LUT.simple[mask], +(components3(a, 1, 8) === 1 && components3(a, 0, 4, true) === 1));
    const X = [0, 2, 4, 6].filter((i) => !x[i] && (x[(i + 1) % 8] || x[(i + 2) % 8])).length, n1 = [0, 2, 4, 6].filter((i) => x[i] || x[(i + 1) % 8]).length, n2 = [1, 3, 5, 7].filter((i) => x[i] || x[(i + 1) % 8]).length, base = X === 1 && Math.min(n1, n2) >= 2 && Math.min(n1, n2) <= 3;
    assert.equal(MORPH_5A_LUT.thin[0][mask], +(base && (!x[0] || !x[1] && !x[2] && x[7])));
    assert.equal(MORPH_5A_LUT.thin[1][mask], +(base && (!x[4] || !x[5] && !x[6] && x[3])));
  }
});
test("ip5a-morph. clean / majority / simultaneous spur / thin endpoints / subfield shrink topology / grayscale minima duals", () => {
  const isolated = gray(9, 5, (x, y) => y === 2 && [1, 4, 5, 6].includes(x) ? 255 : 0), block = gray(9, 9, (x, y) => x >= 3 && x <= 5 && y >= 1 && y <= 7 ? 255 : 0), src = rings();
  try {
    const clean = run("morph_clean", isolated);
    assert.equal(cv.countNonZero(clean), 3);
    clean.delete();
    const sp = run("morph_spur", isolated, { iterations: 1 });
    assert.equal(cv.countNonZero(sp), 2);
    assert.equal(sp.data[2 * 9 + 1], 255);
    assert.equal(sp.data[2 * 9 + 5], 255);
    sp.delete();
    const line = gray(9, 1, (x) => x >= 2 && x <= 6 ? 255 : 0);
    for (const iterations of [1, 0]) {
      const o = run("morph_spur", line, { iterations });
      assert.equal(cv.countNonZero(o), iterations ? 3 : 1);
      o.delete();
    }
    line.delete();
    const maj = run("morph_majority", block);
    assert.equal(cv.countNonZero(maj), 17);
    maj.delete();
    const thin = run("morph_thin", block, { iterations: 0 });
    assert.equal(topology(thin).objects, 1);
    assert.equal(cv.countNonZero(thin), 5);
    for (let y = 0; y < 9; y++) {
      let n = 0;
      for (let x = 0; x < 9; x++) n += thin.data[y * 9 + x] ? 1 : 0;
      assert.ok(n <= 1);
    }
    const thin2 = run("morph_thin", thin, { iterations: 0 });
    same(thin, thin2);
    thin2.delete();
    thin.delete();
    const singleLine = gray(9, 9, (x, y) => x === 4 && y >= 1 && y <= 7 ? 255 : 0), kept = run("morph_thin", singleLine, { iterations: 0 });
    same(kept, singleLine);
    kept.delete();
    singleLine.delete();
    for (const x of [src, block, isolated, gray(2, 1, () => 255)]) {
      const before = topology(x), o = run("morph_shrink", x, { iterations: 0 });
      assert.deepEqual(topology(o), before);
      if (before.holes === 0) assert.equal(cv.countNonZero(o), before.objects);
      o.delete();
      if (x !== src && x !== block && x !== isolated) x.delete();
    }
    const ramp = gray(19, 13, (x, y) => (x * 23 + y * 37) % 256), inv = new cv.Mat();
    cv.bitwise_not(ramp, inv);
    for (const hHeight of [1, 10, 255]) for (const [op, dual] of [["h_minima", "h_maxima"], ["extended_min", "extended_max"]]) {
      const expected = run("morphology", inv, { op: dual, hHeight });
      if (op === "h_minima") cv.bitwise_not(expected, expected);
      const out = run("morph_" + op, ramp, { hHeight });
      same(out, expected);
      expected.delete();
      out.delete();
    }
    ramp.delete();
    inv.delete();
  } finally {
    isolated.delete();
    block.delete();
    src.delete();
  }
});
// Deliberately slow reference: explicit neighbors and flood-fill definitions, no production LUT.
function referenceMorph(src, op, iterations) {
  const w = src.cols, h = src.rows;
  let a = Array.from(src.data, (v) => v > 127 ? 1 : 0);
  const at = (x, y) => x < 0 || x >= w || y < 0 || y >= h ? 0 : a[y * w + x];
  const cap = op === "clean" ? 1 : iterations || (op === "spur" ? Math.max(w, h) : w * h);
  for (let round = 0; round < cap; round++) {
    let changed = false;
    for (const sub of op === "thin" ? [0, 1] : op === "shrink" ? [0, 1, 2, 3] : [0]) {
      // Freeze the directional candidates across all four parity subfields.
      const [dx, dy] = [[0, -1], [0, 1], [1, 0], [-1, 0]][sub];
      const candidates = op === "shrink" ? a.map((v, i) => v && !at(i % w + dx, Math.floor(i / w) + dy)) : null;
      for (const field of op === "shrink" ? [0, 1, 2, 3] : [0]) {
        const next = a.slice();
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
          if (candidates && (!candidates[y * w + x] || ((y + 1) & 1) * 2 + ((x + 1) & 1) !== field)) continue;
          const ring = [[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]].map(([dx2, dy2]) => at(x + dx2, y + dy2));
          const center = at(x, y), n = ring.reduce((u, v) => u + v, 0);
          let value = center;
          if (op === "majority") value = +(n + center >= 5);
          else if (center) {
            if (op === "clean" && n === 0 || op === "spur" && n === 1) value = 0;
            if (op === "thin") {
              const cyclic = (i) => ring[i % 8];
              let crossings = 0, n1 = 0, n2 = 0;
              for (let i = 0; i < 8; i += 2) {
                if (!cyclic(i) && (cyclic(i + 1) || cyclic(i + 2))) crossings++;
                if (cyclic(i) || cyclic(i + 1)) n1++;
                if (cyclic(i + 1) || cyclic(i + 2)) n2++;
              }
              const direction = sub === 0 ? !(ring[0] && (ring[1] || ring[2] || !ring[7])) : !(ring[4] && (ring[5] || ring[6] || !ring[3]));
              if (crossings === 1 && Math.min(n1, n2) >= 2 && Math.min(n1, n2) <= 3 && direction) value = 0;
            }
            if (op === "shrink") {
              const neighborhood = [at(x - 1, y - 1), at(x, y - 1), at(x + 1, y - 1), at(x - 1, y), 1, at(x + 1, y), at(x - 1, y + 1), at(x, y + 1), at(x + 1, y + 1)];
              if (n >= 1 && components3(neighborhood, 1, 8) === 1 && components3(neighborhood, 0, 4, true) === 1) value = 0;
            }
          }
          if (value !== center) changed = true;
          next[y * w + x] = value;
        }
        a = next;
      }
    }
    if (!changed) break;
  }
  return Buffer.from(a.map((v) => v * 255));
}
test("ip5a-shrink-border. one round keeps the 7x7 core; stable square is central and a stable ring stays between its hole and outer edge", () => {
  const solid = gray(13, 13, (x, y) => x >= 2 && x <= 10 && y >= 2 && y <= 10 ? 255 : 0);
  const holed = gray(17, 17, (x, y) => x >= 2 && x <= 14 && y >= 2 && y <= 14 && !(x >= 7 && x <= 9 && y >= 7 && y <= 9) ? 255 : 0);
  const once = run("morph_shrink", solid, { iterations: 1 }), point = run("morph_shrink", solid, { iterations: 0 }), ring = run("morph_shrink", holed, { iterations: 0 });
  try {
    assert.ok(cv.countNonZero(once) >= 49);
    for (let y = 3; y <= 9; y++) for (let x = 3; x <= 9; x++) assert.equal(once.data[y * 13 + x], 255, `core ${x},${y}`);
    assert.equal(cv.countNonZero(point), 1);
    const i = point.data.findIndex((v) => v !== 0);
    assert.ok(Math.abs(i % 13 - 6) <= 1 && Math.abs(Math.floor(i / 13) - 6) <= 1, `central point ${i % 13},${Math.floor(i / 13)}`);
    assert.deepEqual(topology(ring), topology(holed));
    const white = Array.from(ring.data, (v, i2) => v ? [i2 % 17, Math.floor(i2 / 17)] : null).filter(Boolean);
    const xs = white.map(([x]) => x), ys = white.map(([, y]) => y);
    assert.ok(Math.min(...xs) > 2 && Math.max(...xs) < 14 && Math.min(...ys) > 2 && Math.max(...ys) < 14, "ring moves inward from every outer edge");
    assert.ok(Math.min(...xs) < 6 && Math.max(...xs) > 10 && Math.min(...ys) < 6 && Math.max(...ys) > 10, "ring remains separated from every hole edge");
  } finally {
    for (const m of [solid, holed, once, point, ring]) m.delete();
  }
});
test("ip5a-shrink-subfields. one round preserves thin lines and stable elongated objects stay central", () => {
  const fixtures = [
    ["horizontal", 45, 5, (x, y) => y === 2 && x >= 2 && x <= 42, 30, [22, 2]],
    ["vertical", 5, 45, (x, y) => x === 2 && y >= 2 && y <= 42, 20, [2, 22]],
    ["diagonal41", 45, 45, (x, y) => x === y && x >= 2 && x <= 42, 20, [22, 22]],
    ["diagonal30", 34, 34, (x, y) => x === y && x >= 2 && x <= 31, null, [16, 16]],
    ["rectangle", 45, 13, (x, y) => x >= 2 && x <= 42 && y >= 2 && y <= 10, null, [22, 6]],
  ];
  for (const [name, w, h, fn, minimum, center] of fixtures) {
    const src = gray(w, h, (x, y) => fn(x, y) ? 255 : 0);
    try {
      for (const iterations of [1, 0]) {
        const ctx = {}, out = run("morph_shrink", src, { iterations }, ctx);
        try {
          const kept = cv.countNonZero(out);
          assert.deepEqual(Buffer.from(out.data), referenceMorph(src, "shrink", iterations), `${name}/${iterations}`);
          assert.deepEqual(topology(out), topology(src), `${name}/${iterations} topology`);
          assert.match(ctx.info, new RegExp(`ลบ ${cv.countNonZero(src) - kept} พิกเซล`));
          if (iterations === 1 && minimum !== null) assert.ok(kept >= minimum, `${name}: kept ${kept}, need >= ${minimum}`);
          if (iterations === 0) {
            assert.equal(kept, 1, name);
            const i = out.data.findIndex((v) => v !== 0), x = i % w, y = Math.floor(i / w);
            assert.ok(Math.abs(x - center[0]) <= 2 && Math.abs(y - center[1]) <= 2, `${name}: point ${x},${y}`);
          }
        } finally {
          out.delete();
        }
      }
    } finally {
      src.delete();
    }
  }
});
test("ip5a-review-text. Thai morphology info, additions/removals, stable card text, minima depth and inverse warnings", () => {
  const src = gray(5, 5, (x, y) => !(x === 2 && y === 2) ? 255 : 0);
  try {
    const ctx = {};
    run("morph_majority", src, { iterations: 1 }, ctx).delete();
    assert.match(ctx.info, /ลบ 4 พิกเซล · เพิ่ม 1 พิกเซล/);
    for (const op of ["clean", "majority", "spur", "thin", "shrink"]) {
      const ctx2 = {};
      run("morph_" + op, src, {}, ctx2).delete();
      assert.doesNotMatch(ctx2.info, new RegExp(op + " ·"));
    }
    for (const op of ["spur", "thin", "shrink"]) {
      const name = { spur: "ตัดกิ่งเส้น", thin: "ทำให้บาง", shrink: "หดเป็นจุด" }[op];
      assert.equal(OP_PRIMARY["morph_" + op]({ iterations: 0 }), `${name} · จนนิ่ง`);
      assert.equal(OP_PRIMARY["morph_" + op]({ iterations: 2 }), `${name} · 2 รอบ`);
      assert.match(OP_PRIMARY["morph_" + op]({ iterations: 0 }), /จนนิ่ง$/);
      assert.doesNotMatch(OP_PRIMARY["morph_" + op]({ iterations: 0 }), /จนนิ่ง รอบ/);
      assert.match(OP_PRIMARY["morph_" + op]({ iterations: 2 }), /2 รอบ$/);
    }
    assert.equal(OP_PRIMARY.morph_h_minima({ hHeight: 10 }), "ถมหุบขึ้น h=10");
    assert.equal(OP_PRIMARY.morph_extended_min({ hHeight: 10 }), "บริเวณหุบต่ำ h=10");
    for (const op of ["h_minima", "extended_min"]) assert.equal(OPS["morph_" + op].params[0].label, "ความลึก h");
    assert.match(OPS.morphology.params.find((q) => q.key === "hHeight").label, /H-minima \/ Extended Minima/);
    assert.match(OPS.colorspace_inverse.desc, /Lab/);
    assert.equal(OPS.euler_number.needs, "any");
  } finally {
    src.delete();
  }
});
test("ip5a-morph-reference. all five binary paths equal independent definitions on seeded random / border / 1D / singleton masks, preserving shrink topology", () => {
  let seed = 91;
  const rand = () => {
    seed = Math.imul(seed, 1664525) + 1013904223 >>> 0;
    return seed / 4294967296;
  };
  for (const [w, h] of [[1, 1], [1, 13], [15, 1], [17, 13]]) for (const density of [0.25, 0.55, 0.85, 1]) {
    const src = gray(w, h, () => rand() < density ? 255 : 0), before = topology(src);
    try {
      for (const op of ["clean", "majority", "spur", "thin", "shrink"]) for (const iterations of [1, 3, 0]) {
        const used = op === "majority" && iterations === 0 ? 1 : iterations, out = run("morph_" + op, src, { iterations });
        try {
          assert.deepEqual(Buffer.from(out.data), referenceMorph(src, op, used), `${w}x${h} ${density} ${op}/${iterations}`);
          if (op === "shrink") assert.deepEqual(topology(out), before);
        } finally {
          out.delete();
        }
      }
    } finally {
      src.delete();
    }
  }
});
test("ip5a-perf. measured thin / shrink at 640x480 on solid and holed masks; topology and convergence retained", (t) => {
  const solid = gray(640, 480, (x, y) => x >= 80 && x < 560 && y >= 80 && y < 400 ? 255 : 0);
  const holed = gray(640, 480, (x, y) => x >= 80 && x < 560 && y >= 80 && y < 400 && !(x >= 250 && x < 390 && y >= 180 && y < 300) ? 255 : 0);
  try {
    for (const [name, src] of [["solid", solid], ["holed", holed]]) {
      const before = topology(src);
      for (const op of ["thin", "shrink"]) {
        const times = [];
        for (let k = 0; k < 3; k++) {
          const ctx = {}, start = performance.now(), out = run("morph_" + op, src, { iterations: 0 }, ctx);
          times.push(performance.now() - start);
          assert.deepEqual(topology(out), before);
          if (name === "solid" && op === "shrink") assert.equal(cv.countNonZero(out), 1);
          out.delete();
          if (k === 0) t.diagnostic(name + " " + op + " " + ctx.info);
        }
        times.sort((a, b) => a - b);
        t.diagnostic(`${name} ${op} 640x480 median ${times[1].toFixed(2)} ms (3 runs)`);
      }
    }
  } finally {
    solid.delete();
    holed.delete();
  }
});
test("ip5a-G. 99 pre-edit goldens: dehaze, all 19 old morphology operations, all 7 colorspaces and watershed retain pixels / info / values / generated Python", () => {
  const result = {};
  for (const [w, h] of [[1, 1], [37, 29], [97, 61]]) {
    const c = rgb(w, h, (x, y) => {
      const i = (y * w + x) * 3;
      return [0, 1, 2].map((k) => (Math.imul(i + k + 11, 73) ^ (i + k >> 4) * 17) & 255);
    }), g = new cv.Mat();
    cv.cvtColor(c, g, cv.COLOR_RGB2GRAY);
    const jobs = OLD_MORPH.map((op) => ["morphology", { op }]);
    for (const { value: space } of OPS.colorspace.params[0].options) jobs.push(["colorspace", { space }]);
    for (const refine of [true, false]) for (const view of ["result", "dark", "transmission"]) jobs.push(["dehaze", { refine, view }]);
    jobs.push(["watershed", {}]);
    for (const [id, extra] of jobs) {
      const p = { ...defaults(id), ...extra }, ctx = {}, out = run(id, id === "morphology" ? g : c, p, ctx);
      const fid = OP_TO_FILTER[id] || id;
      const st = id === "morphology" ? { id, params: p } : { id: fid, params: { mode: id, ...Object.fromEntries(Object.entries(p).map(([k, v]) => [id + "__" + k, v])) } };
      result[JSON.stringify([w, h, id, extra])] = { image: hash(Buffer.from(out.data)), info: ctx.info, values: ctx.values, python: hash(buildPythonCode([st])) };
      out.delete();
    }
    c.delete();
    g.delete();
  }
  assert.equal(hash(JSON.stringify(result)), "3e796d082a59d924996d267adc0ecedd70bf5d2ee7c8ae5116d22ef75119301f");
});
// Exported Python is executed, not reimplemented by the test. Only availability gates it.
let pyAvailable = false;
try {
  if (!process.env.IP5A_NO_PY) {
    execFileSync("python", ["-c", "import cv2,numpy"], { stdio: "pipe" });
    pyAvailable = true;
  }
} catch {
}
const PY_SKIP = { skip: !pyAvailable && "python/cv2 unavailable (or IP5A_NO_PY=1)" };
let syntaxPython = false;
try {
  execFileSync("python", ["-c", "pass"], { stdio: "pipe" });
  syntaxPython = true;
} catch {
}
test("ip5a-P-syntax. compile every exported branch with real Python (does not require cv2)", { skip: !syntaxPython && "python unavailable" }, () => {
  const codes = [];
  for (const id of ALL) for (const p of [{}, { iterations: 0, hHeight: 50, space: "constructor", view: "labels", alphaBlend: true }, { refine: false, polarity: "bright", conn: "4" }]) codes.push(buildPythonCode([step(id, p)]));
  for (const { value: space } of OPS.colorspace.params[0].options) codes.push(buildPythonCode([step("colorspace", { space }), step("colorspace_inverse", { space }), step("lowlight_enhance")]));
  const ss = step("seg_score");
  ss.src2 = { pipeline: [] };
  codes.push(buildPythonCode([ss, step("pass_fail", { metric: "iou" })]));
  execFileSync("python", ["-c", "import sys,json; codes=json.loads(sys.stdin.buffer.read()); [compile(c, 'exported_'+str(i), 'exec') for i,c in enumerate(codes)]; print(len(codes))"], { input: JSON.stringify(codes), encoding: "utf8", stdio: ["pipe", "pipe", "pipe"] });
});
function pythonCases(jobs) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ip5a-"));
  // cv2.imread always returns BGR. Gray fixtures must enter through three channels;
  // a real Grayscale step below recreates the app's upstream gray input.
  function imageBytes(m) {
    const rgb2 = new cv.Mat(), bgr = new cv.Mat();
    try {
      if (m.channels() === 1) cv.cvtColor(m, rgb2, cv.COLOR_GRAY2RGB);
      else if (m.channels() === 4) cv.cvtColor(m, rgb2, cv.COLOR_RGBA2RGB);
      else m.copyTo(rgb2);
      cv.cvtColor(rgb2, bgr, cv.COLOR_RGB2BGR);
      return Buffer.from(bgr.data);
    } finally {
      rgb2.delete();
      bgr.delete();
    }
  }
  try {
    const manifest = jobs.map((j, i) => {
      let pipe = j.pipe || [step(j.id, j.p)];
      if (j.b) pipe.find((st) => st.id === OP_TO_FILTER.seg_score).src2 = { pipeline: [] };
      if (j.src.channels() === 1) pipe = [step("grayscale"), ...pipe];
      let code = buildPythonCode(pipe);
      code = code.slice(0, code.indexOf('if __name__ == "__main__":'));
      const inputs = [j.src, ...j.b ? [j.b] : []].map((m, k) => {
        const p = path.join(dir, `i${i}_${k}.bin`);
        fs.writeFileSync(p, imageBytes(m));
        const expr = `np.fromfile(${j.b ? k ? "path_b" : "path_a" : "path"}, dtype=np.uint8).reshape((${m.rows}, ${m.cols}, 3))`;
        code = code.replace(`cv2.imread(${j.b ? k ? "path_b" : "path_a" : "path"})`, expr);
        return p;
      });
      if (j.tail) code = code.replace("\n    return img\n", "\n    " + j.tail + "\n    return img\n");
      const file = path.join(dir, `c${i}.py`), out = path.join(dir, `o${i}.bin`);
      fs.writeFileSync(file, code);
      return { file, inputs, out };
    });
    fs.writeFileSync(path.join(dir, "manifest.json"), JSON.stringify(manifest));
    fs.writeFileSync(path.join(dir, "driver.py"), `import json, io, contextlib, sys
cases=json.load(open(sys.argv[1]))
results=[]
for c in cases:
    ns={'__name__':'case'}
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(open(c['file'],encoding='utf8').read(),c['file'],'exec'),ns)
        r=ns['process'](*c['inputs'])
    r.tofile(c['out'])
    results.append({'shape':r.shape,'stdout':buf.getvalue()})
json.dump(results,open(sys.argv[2],'w'))
`);
    execFileSync("python", [path.join(dir, "driver.py"), path.join(dir, "manifest.json"), path.join(dir, "results.json")], { stdio: "pipe", env: { ...process.env, PYTHONIOENCODING: "utf8" }, maxBuffer: 1 << 25 });
    return JSON.parse(fs.readFileSync(path.join(dir, "results.json"), "utf8")).map((r, i) => {
      const raw = fs.readFileSync(manifest[i].out);
      if (r.shape.length === 3) for (let k = 0; k < raw.length; k += 3) [raw[k], raw[k + 2]] = [raw[k + 2], raw[k]];
      return { ...r, bytes: raw };
    });
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}
function appPipe(pipe, src, b) {
  let cur = src.clone();
  try {
    for (const st of pipe) {
      const f = FILTERS[st.id], ctx = { src2: b };
      let g = null;
      try {
        if (needsOf(f, st.params) === "gray" && cur.channels() > 1) {
          g = new cv.Mat();
          cv.cvtColor(cur, g, cur.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY);
        }
        const o = f.apply(cv, g || cur, st.params, ctx);
        cur.delete();
        cur = o;
      } finally {
        if (g) g.delete();
      }
    }
    return cur;
  } catch (e) {
    cur.delete();
    throw e;
  }
}
// แต่ละกลุ่มใช้พารามิเตอร์ชุดเดียวกันทั้งแอปและ Python เพื่อเทียบทุก branch
function parityParams(id) {
  if (id === "watershed_marker") return [{ view: "boundaries" }, { view: "labels" }, { polarity: "bright", hHeight: 1 }];
  if (id === "colorspace_inverse") return OPS.colorspace.params[0].options.map(({ value: space }) => ({ space }));
  if (id === "lowlight_enhance") return [
    { refine: false },
    { refine: true },
    { refine: false, alphaBlend: true },
    { refine: true, alphaBlend: true }
  ];
  if (id === "euler_number") return [{ conn: "8" }, { conn: "4" }];
  if (id.startsWith("morph_")) return [
    { iterations: 1, hHeight: 1 },
    { iterations: 0, hHeight: 10 },
    { iterations: 3, hHeight: 255 }
  ];
  return [{}];
}
function scalarTail(id) {
  if (id === "seg_score") return "print('SCALARS', iou, dice, precision, recall, tp, fp, fn)";
  if (id === "euler_number") return "print('SCALARS', objects, holes, euler)";
  if (id === "watershed_marker") return "print('SCALARS', objects)";
  return void 0;
}
for (const id of ALL) test("ip5a-P. real Python pixel parity " + id, PY_SKIP, (t) => {
  const a = noise(37, 29), g = gray(31, 23, (x, y) => (x * 17 + y * 13) % 256), r = rings(), touch = touching(), one = rgb(1, 1, () => [30, 60, 90]);
  const jobs = [];
  try {
    const images = id === "watershed_marker" ? [touch, one] : [a, g, r, one];
    for (const src of images) {
      for (const p of parityParams(id)) jobs.push({ id, src, p, b: id === "seg_score" ? r : void 0, tail: scalarTail(id) });
    }
    if (id === "colorspace_inverse") for (const { value: space } of OPS.colorspace.params[0].options) jobs.push({ id, src: a, pipe: [step("colorspace", { space }), step(id, { space }), step("lowlight_enhance", { refine: false })] });
    const results = pythonCases(jobs);
    let differing = 0, total = 0, max = 0;
    jobs.forEach((j, i) => {
      const pipe = j.pipe || [step(j.id, j.p)], out = appPipe(pipe, j.src, j.b), py = results[i];
      try {
        assert.equal(py.bytes.length, out.data.length);
        let n = 0, mx = 0;
        for (let k = 0; k < out.data.length; k++) {
          const d = Math.abs(out.data[k] - py.bytes[k]);
          if (d) n++;
          mx = Math.max(mx, d);
        }
        if (id === "lowlight_enhance" && j.p?.refine !== false) {
          assert.ok(mx <= 1 && n <= out.data.length * 5e-3, `${id} n=${n} max=${mx}`);
        } else assert.equal(n, 0, `${id} ${JSON.stringify(j.p)} max=${mx}`);
        differing += n;
        total += out.data.length;
        max = Math.max(max, mx);
        if (j.tail) {
          const ctx = { src2: j.b }, direct = run(id, j.src, j.p, ctx);
          direct.delete();
          const fields = Object.values(vals(ctx)), line = py.stdout.split("\n").find((x) => x.startsWith("SCALARS "));
          assert.ok(line);
          assert.deepEqual(line.slice(8).trim().split(/\s+/).map(Number), fields);
        }
      } finally {
        out.delete();
      }
    });
    t.diagnostic(`${jobs.length} cases, ${differing}/${total} bytes differ, max ${max}`);
  } finally {
    for (const m of [a, g, r, touch, one]) m.delete();
  }
});
test("ip5a-PF. real Python seg_score -> Pass/Fail decisions", PY_SKIP, () => {
  const src = rgb(13, 9, (x, y) => x > 4 && y > 2 ? [255, 255, 255] : [0, 0, 0]);
  try {
    for (const metric of ["iou", "dice"]) {
      const pipe = [step("seg_score"), step("pass_fail", { metric, cond: "ge", lo: 0.9 })];
      const [py] = pythonCases([{ src, b: src, pipe }]);
      assert.match(py.stdout, /RESULT: OK/);
    }
  } finally {
    src.delete();
  }
});
// ลำดับช่องสีหลัง Color Space ต้องกลับเป็น BGR เมื่อสร้างภาพระบาย label ใหม่
test("ip5a-marker-order-contract. labels use BGR output after Color Space and downstream gray conversion tracks it", () => {
  const code = buildPythonCode([step("colorspace", { space: "lab" }), step("watershed_marker", { view: "labels" }), step("lowlight_enhance", { alphaBlend: true })]);
  assert.match(code, /img\[markers == label\] = rgb\[::-1\]/);
  assert.match(code, /ll_gray = cv2.cvtColor\(ll_input, cv2.COLOR_BGR2GRAY\)/);
});
test("ip5a-P-marker-order. labels after Color Space and a real upstream gray step match exported Python", PY_SKIP, (t) => {
  const src = touching(), g = new cv.Mat();
  cv.cvtColor(src, g, cv.COLOR_RGB2GRAY);
  try {
    const jobs = OPS.colorspace.params[0].options.map(({ value: space }) => ({ src, pipe: [step("colorspace", { space }), step("watershed_marker", { view: "labels" })] }));
    for (const view of ["boundaries", "labels"]) jobs.push({ src: g, pipe: [step("grayscale"), step("watershed_marker", { view })] });
    const results = pythonCases(jobs);
    let total = 0;
    jobs.forEach((j, i) => {
      const out = appPipe(j.pipe, j.src);
      try {
        assert.deepEqual(results[i].bytes, Buffer.from(out.data), `marker pipeline ${i}`);
        total += out.data.length;
      } finally {
        out.delete();
      }
    });
    t.diagnostic(`${jobs.length} cases, 0/${total} bytes differ`);
  } finally {
    src.delete();
    g.delete();
  }
});
test("ip5a-P-seg-colours. Python overlay includes exact white TP, red FP, blue FN and black TN", PY_SKIP, (t) => {
  const a = rgb(6, 1, (x) => [255, 255, 255, 0, 0, 127][x] === 255 ? [255, 255, 255] : [0, 0, 0]);
  const b = rgb(6, 1, (x) => [255, 255, 0, 255, 0, 127][x] === 255 ? [255, 255, 255] : [0, 0, 0]);
  try {
    const [py] = pythonCases([{ id: "seg_score", src: a, b }]), out = run("seg_score", a, {}, { src2: b });
    try {
      assert.deepEqual(py.bytes, Buffer.from(out.data));
      assert.deepEqual(Array.from(py.bytes), [255, 255, 255, 255, 255, 255, 230, 60, 60, 60, 140, 255, 0, 0, 0, 0, 0, 0]);
      t.diagnostic("1 case, 0/18 bytes differ");
    } finally {
      out.delete();
    }
  } finally {
    a.delete();
    b.delete();
  }
});

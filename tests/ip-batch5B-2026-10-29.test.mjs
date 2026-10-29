// IP batch 5B — spec fixtures, clean-tree goldens, lifetime gates and real exported-Python parity.
import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import cv from "@techstark/opencv-js";
import { OPS, FILTERS, defaultParams, needsOf, isParamVisible, passFailVerdict, seededGaussian } from "../src/renderer/filters/registry.js";
import { TOOL_SECTIONS, OP_TO_FILTER } from "../src/renderer/ui/palette-panel.js";
import { OP_PRIMARY, OP_ICON, GROUP_ICON, OPTION_GROUPS } from "../src/renderer/filters/ui-meta.js";
import { buildPythonCode, PF_VALUE_OPS } from "../src/renderer/codegen.js";
import { assertLeakFree, assertWrapperBalance, makeThrowingCv } from "./_leak-helpers.mjs";

while (typeof cv.Mat !== "function") await new Promise((r) => setTimeout(r, 20));
const IDS = ["image_pyramid", "frangi_vesselness", "anisotropic_diffusion", "debayer", "piqe"];
const defaults = (id) => Object.fromEntries(OPS[id].params.map((q) => [q.key, q.default]));
const hash = (x) => crypto.createHash("sha256").update(x).digest("hex");
const vals = (ctx) => Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));
function gray(w, h, fn) {
  const data = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = fn(x, y);
  return cv.matFromArray(h, w, cv.CV_8UC1, data);
}
function rgb(w, h, fn) {
  const data = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set(fn(x, y), (y * w + x) * 3);
  return cv.matFromArray(h, w, cv.CV_8UC3, data);
}
function run(id, src, p = {}, ctx = {}, c = cv) {
  return OPS[id].apply(c, src, { ...defaults(id), ...p }, ctx);
}
function step(id, p = {}) {
  const fid = OP_TO_FILTER[id] || id;
  const local = { ...defaults(id), ...p };
  return fid === id ? { id, params: local } : { id: fid, params: { mode: id, ...Object.fromEntries(Object.entries(local).map(([k, v]) => [id + "__" + k, v])) } };
}
function same(a, b) {
  assert.deepEqual([a.cols, a.rows, a.channels()], [b.cols, b.rows, b.channels()]);
  assert.deepEqual(Buffer.from(a.data), Buffer.from(b.data));
}
function mad(a, b, border = 0) {
  let total = 0;
  let n = 0;
  for (let y = border; y < a.rows - border; y++) for (let x = border; x < a.cols - border; x++) for (let c = 0; c < a.channels(); c++) {
    const i = (y * a.cols + x) * a.channels() + c;
    total += Math.abs(a.data[i] - b.data[i]);
    n++;
  }
  return total / n;
}
function sampleVariance(xs) {
  let sum = 0;
  for (const x of xs) sum += x;
  const mean = sum / xs.length;
  let ss = 0;
  for (const x of xs) ss += (x - mean) * (x - mean);
  return ss / (xs.length - 1);
}
// อ้างอิงอิสระของ PIQE: ใช้ตัวแปรสุ่ม N-1 จาก sampleVariance ในเทสต์เอง (v มีผลต่อคะแนน จึงจับ N แทน N-1 ได้)
// ขอบบล็อกเขียนตรง ๆ: แถว 0, แถว 15, คอลัมน์ 0, คอลัมน์ 15 · คอลัมน์กลาง 7–8
function piqeReference(m) {
  const w = m.cols, h = m.rows, input = new cv.Mat(), square = new cv.Mat(), mean = new cv.Mat(), meanSquare = new cv.Mat();
  try {
    m.convertTo(input, cv.CV_64F);
    square.create(h, w, cv.CV_64FC1);
    for (let i = 0; i < w * h; i++) square.data64F[i] = input.data64F[i] ** 2;
    cv.GaussianBlur(input, mean, new cv.Size(7, 7), 7 / 6, 7 / 6, cv.BORDER_REPLICATE);
    cv.GaussianBlur(square, meanSquare, new cv.Size(7, 7), 7 / 6, 7 / 6, cv.BORDER_REPLICATE);
    const mscn = (y, x) => { const i = y * w + x, mu = mean.data64F[i]; return (input.data64F[i] - mu) / (Math.sqrt(Math.abs(meanSquare.data64F[i] - mu * mu)) + 1); };
    let active = 0, artifact = 0, noise = 0, distortion = 0;
    for (let by = 0; by + 16 <= h; by += 16) for (let bx = 0; bx + 16 <= w; bx += 16) {
      const at = (y, x) => mscn(by + y, bx + x), all = [], center = [], surround = [];
      for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { all.push(at(y, x)); (x === 7 || x === 8 ? center : surround).push(at(y, x)); }
      const v = sampleVariance(all);
      if (v <= 0.1) continue;
      active++;
      const idx = Array.from({ length: 16 }, (_, i) => i);
      const edges = [idx.map((i) => at(0, i)), idx.map((i) => at(15, i)), idx.map((i) => at(i, 0)), idx.map((i) => at(i, 15))];
      let hasArtifact = false;
      for (const e of edges) for (let k = 0; k <= 10; k++) if (Math.sqrt(sampleVariance(e.slice(k, k + 6))) < 0.1) hasArtifact = true;
      const sc = Math.sqrt(sampleVariance(center)), ss = Math.sqrt(sampleVariance(surround)), top = Math.max(sc, ss);
      const hasNoise = Math.sqrt(v) > 2 * (top ? Math.abs(sc - ss) / top : 0);
      if (hasArtifact) { artifact++; distortion += 1 - v; }
      if (hasNoise) { noise++; distortion += v; }
    }
    return { piqe: Math.max(0, Math.min(100, (distortion + 1) / (active + 1) * 100)), piqe_active: active, piqe_artifact: artifact, piqe_noise: noise };
  } finally {
    for (const x of [input, square, mean, meanSquare]) x.delete();
  }
}
function texture(w = 128, h = 128) {
  return gray(w, h, (x, y) => Math.floor(110 + 35 * Math.sin(x * 0.6) + 30 * Math.sin(y * 0.72) + 0.15 * x + 0.1 * y + 0.5));
}

test("ip5b-S. five registered tools, exact defaults/ranges, placement, visibility, Thai metadata and values sources", () => {
  const placements = [["image_pyramid", "geometry", "resize"], ["frangi_vesselness", "segment_edge", "log_edge"], ["anisotropic_diffusion", "filter_denoise", "guided_filter"], ["debayer", "color_tools", "colorspace_inverse"], ["piqe", "image_quality", "seg_score"]];
  for (const [id, host, after] of placements) {
    assert.ok(OPS[id], id + " must be registered");
    assert.equal(OP_TO_FILTER[id], host);
    const list = TOOL_SECTIONS.find((s) => s.ops.includes(id)).ops;
    assert.equal(list[list.indexOf(after) + 1], id);
    assert.deepEqual(FILTERS[host].params[0].options.map((o) => o.value), list);
    assert.ok(OPS[id].desc.length > 70);
    assert.ok(OPS[id].modeHint);
    assert.ok(!/ที่สุด/.test(JSON.stringify(OPS[id])));
    assert.ok(/[ก-๙]/.test(OP_PRIMARY[id](defaults(id))));
    const glyph = OP_ICON[id];
    assert.equal(Object.values(OP_ICON).filter((x) => x === glyph).length, 1);
    assert.ok(![...Object.values(GROUP_ICON), "🧩", "📊", "📌"].includes(glyph));
    for (const q of OPS[id].params) {
      assert.ok(q.label);
      assert.ok(q.desc || q.hint);
      assert.equal(defaultParams(host)[id + "__" + q.key], q.default);
    }
    if (OPTION_GROUPS[host + ".mode"]) assert.ok(OPTION_GROUPS[host + ".mode"].flatMap((g) => g[1]).includes(id));
  }
  assert.deepEqual(defaults("image_pyramid"), { kind: "gaussian", direction: "reduce", levels: 1, view: "level" });
  assert.deepEqual(defaults("frangi_vesselness"), { polarity: "bright", sigmaMin: 1, sigmaMax: 3, beta: 0.5 });
  assert.deepEqual(defaults("anisotropic_diffusion"), { iterations: 10, kappa: 20, conduction: "exponential" });
  assert.deepEqual(defaults("debayer"), { mode: "demosaic", pattern: "rggb", method: "bilinear" });
  assert.deepEqual(defaults("piqe"), {});
  const range = (id, key) => {
    const q = OPS[id].params.find((q) => q.key === key);
    return [q.min, q.max, q.step];
  };
  assert.deepEqual(range("image_pyramid", "levels"), [1, 5, 1]);
  assert.deepEqual(range("frangi_vesselness", "sigmaMin"), [1, 8, 0.5]);
  assert.deepEqual(range("frangi_vesselness", "sigmaMax"), [1, 10, 0.5]);
  assert.deepEqual(range("frangi_vesselness", "beta"), [0.1, 2, 0.05]);
  assert.deepEqual(range("anisotropic_diffusion", "iterations"), [1, 50, 1]);
  assert.deepEqual(range("anisotropic_diffusion", "kappa"), [1, 100, 1]);
  for (const [id, key, p, visible] of [["image_pyramid", "direction", { kind: "laplacian" }, false], ["image_pyramid", "direction", { kind: "gaussian" }, true], ["debayer", "method", { mode: "mosaic" }, false], ["debayer", "method", { mode: "demosaic" }, true]]) {
    const host = OP_TO_FILTER[id];
    assert.equal(isParamVisible(FILTERS[host].params.find((q) => q.key === id + "__" + key), step(id, p).params, FILTERS[host].params), visible);
  }
  for (const id of IDS) assert.equal(PF_VALUE_OPS.has(id), ["image_pyramid", "frangi_vesselness", "piqe"].includes(id));
});

test("ip5b-pyramid. odd sizes, constant Laplacians, all mosaic steps, smooth reconstruction and dimension caps", () => {
  const src = rgb(641, 481, () => [30, 80, 150]);
  const smooth = gray(65, 49, (x, y) => 30 + x + y);
  const wide = gray(3000, 2, () => 50);
  const cap = gray(2049, 1, () => 50);
  const one = gray(1, 1, () => 77);
  const row7 = gray(7, 1, (x) => 10 * x);
  const col7 = gray(1, 7, (x, y) => 10 * y);
  try {
    for (const [p, dims, last] of [[{}, [321, 241], [321, 241]], [{ levels: 2, view: "mosaic" }, [1131, 481], [161, 121]], [{ kind: "laplacian", levels: 2 }, [321, 241], [321, 241]], [{ kind: "laplacian", levels: 2, view: "mosaic" }, [966, 481], [321, 241]]]) {
      const ctx = {};
      const out = run("image_pyramid", src, p, ctx);
      assert.deepEqual([out.cols, out.rows], dims);
      assert.deepEqual([vals(ctx).pyr_width, vals(ctx).pyr_height], last);
      if (p.kind === "laplacian") for (let y = 0; y < out.rows; y++) for (let x = 0; x < out.cols; x++) {
        const inTile = x < 641 || x >= 645 && y < 241;
        assert.equal(out.data[(y * out.cols + x) * 3], inTile ? 128 : 0);
      }
      out.delete();
    }
    const reduced = run("image_pyramid", smooth);
    const up = new cv.Mat();
    cv.pyrUp(reduced, up, new cv.Size(smooth.cols, smooth.rows));
    assert.ok(mad(smooth, up, 3) < 1);
    reduced.delete();
    up.delete();
    const ctx = {};
    const mosaic = run("image_pyramid", wide, { view: "mosaic" }, ctx);
    assert.equal(mosaic.cols, 3000);
    assert.match(ctx.info, /4096/);
    mosaic.delete();
    // กฎหยุด reduce คือ "ด้านใดด้านหนึ่ง < 2" (ไม่ใช่ทั้งสองด้าน) จึงต้องหยุดที่ระดับ 0 แม้อีกด้านยาว 7
    for (const [input, p] of [[cap, { direction: "expand" }], [one, {}], [one, { kind: "laplacian" }], [row7, {}], [col7, {}], [row7, { kind: "laplacian" }], [col7, { kind: "laplacian" }]]) {
      const stop = {};
      const out = run("image_pyramid", input, p, stop);
      assert.equal(vals(stop).pyr_levels, 0);
      assert.match(stop.info, /หยุด/);
      out.delete();
    }
    const expanded = run("image_pyramid", one, { direction: "expand", levels: 3 });
    assert.deepEqual([expanded.cols, expanded.rows], [8, 8]);
    assert.ok(expanded.data.every((v) => v === 77));
    expanded.delete();
  } finally {
    for (const m of [src, smooth, wide, cap, one, row7, col7]) m.delete();
  }
});

test("ip5b-frangi. bright/dark line polarity, blob suppression, constant black response and swapped scale endpoints", () => {
  const line = gray(65, 65, (x) => Math.abs(x - 32) <= 1 ? 255 : 0);
  const blob = gray(65, 65, (x, y) => (x - 32) ** 2 + (y - 32) ** 2 <= 4 ? 255 : 0);
  const constant = gray(31, 27, () => 111);
  try {
    const ctx = {};
    const bright = run("frangi_vesselness", line, {}, ctx);
    const dark = run("frangi_vesselness", line, { polarity: "dark" });
    const ball = run("frangi_vesselness", blob);
    assert.ok(bright.data[32 * 65 + 32] >= 200);
    assert.equal(bright.data[32 * 65 + 15], 0);
    assert.equal(dark.data[32 * 65 + 32], 0);
    assert.ok(ball.data[32 * 65 + 32] < bright.data[32 * 65 + 32]);
    assert.ok(vals(ctx).ridge_max > 0.5);
    const cc = {};
    const flat = run("frangi_vesselness", constant, {}, cc);
    assert.ok(flat.data.every((v) => v === 0));
    assert.equal(vals(cc).ridge_max, 0);
    assert.match(cc.info, /ไม่พบโครงสร้างเส้น/);
    assert.match(ctx.info, /เส้นสว่าง/);
    const dc = {};
    run("frangi_vesselness", line, { polarity: "dark" }, dc).delete();
    assert.match(dc.info, /เส้นมืด/);
    assert.ok(!/bright|dark/.test(ctx.info + dc.info));
    const a = run("frangi_vesselness", line, { sigmaMin: 3, sigmaMax: 1 });
    same(a, bright);
    for (const m of [bright, dark, ball, flat, a]) m.delete();
  } finally {
    for (const m of [line, blob, constant]) m.delete();
  }
});

test("ip5b-diffusion. simultaneous four-neighbor update, flat-region variance reduction and step-edge contrast", (t) => {
  const z = seededGaussian(77, 96 * 64);
  const src = gray(96, 64, (x, y) => Math.max(0, Math.min(255, Math.floor((x < 48 ? 40 : 210) + 10 * z[y * 96 + x] + 0.5))));
  const constant = rgb(19, 17, () => [21, 83, 177]);
  const impulse = gray(3, 3, (x, y) => x === 1 && y === 1 ? 4 : 0);
  const region = (m) => Array.from({ length: 32 * 44 }, (_, i) => m.data[(16 + Math.floor(i / 32)) * 96 + 4 + i % 32]);
  const contrast = (m) => {
    let left = 0;
    let right = 0;
    for (let y = 8; y < 56; y++) for (let d = 0; d < 3; d++) {
      left += m.data[y * 96 + 47 - d];
      right += m.data[y * 96 + 48 + d];
    }
    return (right - left) / 144;
  };
  try {
    for (const conduction of ["exponential", "quadratic"]) {
      const out = run("anisotropic_diffusion", src, { conduction });
      const ratio = sampleVariance(region(out)) / sampleVariance(region(src));
      const edge = contrast(out) / contrast(src);
      t.diagnostic(`${conduction}: variance ratio ${ratio}, edge contrast ratio ${edge}`);
      assert.ok(ratio <= 0.5);
      assert.ok(edge >= 0.8);
      const flat = run("anisotropic_diffusion", constant, { conduction });
      same(flat, constant);
      const one = run("anisotropic_diffusion", impulse, { iterations: 1, kappa: 100, conduction });
      assert.deepEqual(Array.from(one.data), [0, 1, 0, 1, 0, 1, 0, 1, 0]);
      for (const m of [out, flat, one]) m.delete();
    }
  } finally {
    for (const m of [src, constant, impulse]) m.delete();
  }
});

test("ip5b-debayer. every pattern/method round-trip, wrong pattern, pure red and removed VNG falls back to Bilinear", (t) => {
  const src = rgb(64, 48, (x, y) => [30 + 2 * x, 50 + 2 * y, 210 - x - y]);
  const red = rgb(32, 32, () => [200, 0, 0]);
  try {
    for (const pattern of ["rggb", "bggr", "grbg", "gbrg"]) {
      const raw = run("debayer", src, { mode: "mosaic", pattern });
      assert.equal(raw.channels(), 1);
      const rawRed = run("debayer", red, { mode: "mosaic", pattern });
      const want = { rggb: [30, 50, 52, 208], bggr: [210, 50, 52, 32], grbg: [50, 32, 209, 52], gbrg: [50, 209, 30, 52] }[pattern];
      assert.deepEqual([raw.data[0], raw.data[1], raw.data[64], raw.data[65]], want);
      for (const method of ["bilinear", "ea"]) {
        const out = run("debayer", raw, { pattern, method });
        const error = mad(src, out);
        t.diagnostic(`${pattern}/${method}: mean abs round-trip error ${error}`);
        assert.ok(error <= 3);
        const wrong = run("debayer", raw, { pattern: { rggb: "bggr", bggr: "rggb", grbg: "gbrg", gbrg: "grbg" }[pattern], method });
        assert.ok(mad(src, wrong) > error + 15);
        const redOut = run("debayer", rawRed, { pattern, method });
        assert.deepEqual(Array.from(redOut.data.slice((16 * 32 + 16) * 3, (16 * 32 + 16) * 3 + 3)), [200, 0, 0]);
        for (const m of [out, wrong, redOut]) m.delete();
      }
      raw.delete();
      rawRed.delete();
    }
    // VNG ถูกถอดออก: opencv.js 4.10 COLOR_Bayer*2RGB_VNG ให้ภาพเลื่อน 2 แถว + สีเพี้ยน (ต่างจาก cv2)
    // วัดจริง: round-trip ไล่ระดับ 1.93 (opencv.js) เทียบ 0.000 (cv2) · ลายวงแหวน 18.3 เทียบ 2.74
    // ค่า "vng" ที่ค้างมาจากโปรเจกต์เก่า/แก้มือ ต้องใช้ Bilinear เหมือนกันทั้งในแอปและ Python ที่ export
    const ctx = {};
    const fallback = run("debayer", src, { mode: "demosaic", method: "vng" }, ctx);
    const bilinear = run("debayer", src, { mode: "demosaic" });
    same(fallback, bilinear);
    assert.match(ctx.info, /Bilinear/);
    assert.equal(buildPythonCode([step("debayer", { method: "vng" })]), buildPythonCode([step("debayer", { method: "bilinear" })]));
    fallback.delete();
    bilinear.delete();
  } finally {
    for (const m of [src, red]) m.delete();
  }
});

test("ip5b-piqe. no-detail convention, small-image empty values, cropped remainder, formula and Pass/Fail", () => {
  const flat = rgb(32, 32, () => [40, 80, 120]);
  const small = gray(15, 32, () => 40);
  // Four 16x8 checkerboard bands have flat 6-sample boundary segments.
  // All four blocks are active and both artifact+noise: D = 4*((1-v)+v) = 4, score=100.
  const patches = gray(32, 32, (x, y) => y % 16 >= 4 && y % 16 < 12 ? (x + y) % 2 * 255 : 128);
  try {
    const ctx = { src2: small };
    const out = run("piqe", flat, {}, ctx);
    same(out, flat);
    assert.deepEqual(vals(ctx), { piqe: 100, piqe_active: 0, piqe_artifact: 0, piqe_noise: 0 });
    assert.match(ctx.info, /ไม่มีบริเวณที่มีรายละเอียดพอจะประเมิน/);
    assert.ok(!/เข้า B/.test(ctx.info));
    out.delete();
    const sc = {};
    run("piqe", small, {}, sc).delete();
    assert.ok(Object.values(vals(sc)).every((v) => v === null));
    assert.match(sc.info, /ภาพเล็กกว่า 16×16 ประเมินไม่ได้/);
    const pc = {};
    const po = run("piqe", patches, {}, pc);
    assert.deepEqual(vals(pc), { piqe: 100, piqe_active: 4, piqe_artifact: 4, piqe_noise: 4 });
    assert.deepEqual(Array.from(po.data.slice((8 * 32 + 8) * 3, (8 * 32 + 8) * 3 + 3)), [68, 32, 88]);
    po.delete();
    const remainder = gray(35, 34, (x, y) => x >= 32 || y >= 32 ? (x + y) % 2 * 255 : 128);
    const rc = {};
    const ro = run("piqe", remainder, {}, rc);
    for (let y = 0; y < 34; y++) for (let x = 0; x < 35; x++) if (x >= 32 || y >= 32) for (let c = 0; c < 3; c++) assert.equal(ro.data[(y * 35 + x) * 3 + c], remainder.data[y * 35 + x]);
    ro.delete();
    remainder.delete();
    assert.equal(passFailVerdict({ metric: "piqe", cond: "le", hi: 50 }, { op: "piqe", values: ctx.values }).result, "NG");
    assert.equal(passFailVerdict({ metric: "piqe", cond: "le", hi: 100 }, { op: "piqe", values: ctx.values }).result, "OK");
    const py = buildPythonCode([step("piqe"), step("pass_fail", { metric: "piqe", cond: "le", hi: 50 })]);
    assert.ok(!py.includes("WARNING: skipped"));
    assert.match(py, /pf_up = \{'value': piqe\}/);
  } finally {
    for (const m of [flat, small, patches]) m.delete();
  }
});

test("ip5b-piqe-ref. v matters: artifact-only, noise-only, centre columns 7-8 and bottom edge row 15 against an independent N-1 reference", () => {
  const z = seededGaussian(5, 32 * 32);
  // ลายริ้วคาบ 8 คอลัมน์ในแถว 0..T-1 ที่เหลือเรียบ: ขอบล่างแถว 15 เรียบพอ (std < 0.1) แต่แถว 14 ยังไม่เรียบ
  const stripes = (rows, amp) => gray(32, 32, (x, y) => y < rows ? Math.round(128 + amp * Math.sin(2 * Math.PI * x / 8) + ((x + y) % 2 ? 10 : -10)) : 128);
  // คอลัมน์ 5..10 มีแอมพลิจูดต่างกัน: ถ้าใช้คอลัมน์กลางผิด (6–7 หรือ 8–9) beta เปลี่ยนจน noise พลิก
  const amps = { 5: 60, 6: 60, 7: 60, 8: 0, 9: 60, 10: 0 };
  const cases = [
    ["noise-only", gray(32, 32, (x, y) => Math.max(0, Math.min(255, Math.floor(128 + 40 * z[y * 32 + x])))), { piqe_active: 4, piqe_artifact: 0, piqe_noise: 4 }],
    ["artifact-only bottom row 15 (rows<13)", stripes(13, 60), { piqe_active: 2, piqe_artifact: 2, piqe_noise: 0 }],
    ["artifact-only bottom row 15 (rows<14)", stripes(14, 127), { piqe_active: 2, piqe_artifact: 2, piqe_noise: 0 }],
    ["centre columns 7-8", gray(32, 32, (x, y) => { const a = amps[x % 16] || 0; return 128 + ((x + y) % 2 ? a / 2 : -a / 2) | 0; }), { piqe_active: 4, piqe_artifact: 4, piqe_noise: 4 }],
  ];
  for (const [name, m, want] of cases) {
    try {
      const ctx = {};
      run("piqe", m, {}, ctx).delete();
      const got = vals(ctx), ref = piqeReference(m);
      for (const key of ["piqe_active", "piqe_artifact", "piqe_noise"]) {
        assert.equal(got[key], want[key], `${name} app ${key}`);
        assert.equal(ref[key], want[key], `${name} ref ${key}`);
      }
      assert.ok(Math.abs(got.piqe - ref.piqe) <= 1e-9, `${name}: app ${got.piqe} ref ${ref.piqe}`);
    } finally {
      m.delete();
    }
  }
});

test("ip5b-E/L. immutable inputs, new outputs, numeric/select fallbacks, all edge types and Mat lifetime", () => {
  const c = rgb(19, 17, (x, y) => [x * 11, y * 13, (x + y) * 7]);
  const g = gray(19, 17, (x, y) => (x * 19 + y * 23) & 255);
  const rgba = new cv.Mat();
  cv.cvtColor(c, rgba, cv.COLOR_RGB2RGBA);
  const one = gray(1, 1, () => 128);
  const empty = new cv.Mat();
  try {
    for (const id of IDS) for (const src of [c, g, rgba, one, empty]) {
      const before = Buffer.from(src.data);
      const ctx = {};
      const out = run(id, src, {}, ctx);
      assert.notEqual(out, src);
      assert.deepEqual(Buffer.from(src.data), before);
      if (src === empty) assert.equal(ctx.info, "ภาพว่าง");
      out.delete();
      assertLeakFree(cv, `${id}/${src.channels()}/${src.cols}`, (cc) => run(id, src, {}, {}, cc).delete(), { iters: 3, warmup: 1, minCreated: 0 });
    }
    for (const [id, invalid, valid] of [["image_pyramid", { kind: "constructor", direction: "constructor", view: "constructor", levels: -8 }, {}], ["frangi_vesselness", { polarity: "constructor", beta: 9, sigmaMin: -2, sigmaMax: NaN }, { beta: 2 }], ["anisotropic_diffusion", { iterations: -4, kappa: 900, conduction: "constructor" }, { iterations: 1, kappa: 100 }], ["debayer", { mode: "constructor", pattern: "constructor", method: "constructor" }, {}]]) {
      const a = run(id, c, invalid);
      const b = run(id, c, valid);
      same(a, b);
      a.delete();
      b.delete();
    }
  } finally {
    for (const m of [c, g, rgba, one, empty]) m.delete();
  }
});

test("ip5b-L-branches. all selected branches and injected cv/info/values failures release every Mat", () => {
  const src = rgb(35, 33, (x, y) => [(x * 17 + y) & 255, (y * 19 + x) & 255, (x * y) & 255]);
  try {
    const jobs = [];
    for (const kind of ["gaussian", "laplacian"]) for (const direction of ["reduce", "expand"]) for (const view of ["level", "mosaic"]) jobs.push(["image_pyramid", { kind, direction, view, levels: 2 }]);
    for (const polarity of ["bright", "dark"]) jobs.push(["frangi_vesselness", { polarity, sigmaMin: 2, sigmaMax: 1 }]);
    jobs.push(["anisotropic_diffusion", { conduction: "quadratic" }], ["piqe", {}]);
    for (const mode of ["mosaic", "demosaic"]) for (const pattern of ["rggb", "bggr", "grbg", "gbrg"]) for (const method of ["bilinear", "ea"]) jobs.push(["debayer", { mode, pattern, method }]);
    for (const [id, p] of jobs) assertLeakFree(cv, id + JSON.stringify(p), (cc) => run(id, src, p, {}, cc).delete(), { iters: 3, warmup: 1, minCreated: 0 });
    const calls = { image_pyramid: ["pyrDown", "pyrUp", "subtract"], frangi_vesselness: ["cvtColor", "matFromArray", "GaussianBlur", "Sobel"], debayer: ["cvtColor"], piqe: ["cvtColor", "GaussianBlur"] };
    for (const [id, names] of Object.entries(calls)) for (const name of names) {
      const p = id === "image_pyramid" ? { kind: "laplacian" } : {};
      assertWrapperBalance(cv, `${id}/${name}`, (cc) => run(id, src, p, {}, makeThrowingCv(cc, name)).delete(), { expectThrow: /injected test failure/, minCreated: 0 });
    }
    for (const id of IDS) for (const key of ["info", ...["image_pyramid", "frangi_vesselness", "piqe"].includes(id) ? ["values"] : []]) {
      const ctx = {};
      Object.defineProperty(ctx, key, { set() {
        throw Error("setter");
      } });
      assertWrapperBalance(cv, `${id}/${key}`, (cc) => run(id, src, {}, ctx, cc).delete(), { expectThrow: /setter/, minCreated: 1 });
    }
  } finally {
    src.delete();
  }
});

test("ip5b-piqe-quality. clean texture scores below seeded Gaussian noise and strong DCT JPEG distortion", (t) => {
  const src = texture();
  const z = seededGaussian(129, src.data.length);
  const noisy = gray(src.cols, src.rows, (x, y) => Math.max(0, Math.min(255, Math.floor(src.data[y * src.cols + x] + z[y * src.cols + x] * 25 + 0.5))));
  const jpeg = run("dct_jpeg", src, { quality: 5 });
  try {
    const scores = {};
    for (const [name, input] of [["clean", src], ["noise25", noisy], ["jpeg5", jpeg]]) {
      const ctx = {};
      run("piqe", input, {}, ctx).delete();
      scores[name] = vals(ctx).piqe;
      t.diagnostic(`${name}: ${JSON.stringify(vals(ctx))}`);
    }
    assert.ok(scores.clean < scores.noise25);
    assert.ok(scores.clean < scores.jpeg5);
  } finally {
    for (const m of [src, noisy, jpeg]) m.delete();
  }
});

test("ip5b-perf. Frangi sigma 1-3 and diffusion 10 iterations, 640x480, median of three warm runs", (t) => {
  const graySrc = gray(640, 480, (x, y) => (x * 7 + y * 11 + x * y % 97) & 255);
  const colorSrc = rgb(640, 480, (x, y) => [(x + y) & 255, (x * 3 + y * 7) & 255, (x * 11 + y * 5) & 255]);
  try {
    for (const [id, src] of [["frangi_vesselness", graySrc], ["anisotropic_diffusion", graySrc], ["anisotropic_diffusion", colorSrc]]) {
      run(id, src).delete();
      const times = [];
      for (let i = 0; i < 3; i++) {
        const start = performance.now();
        run(id, src).delete();
        times.push(performance.now() - start);
      }
      times.sort((a, b) => a - b);
      t.diagnostic(`${id} ${src.channels()}ch 640x480 median ${times[1].toFixed(2)} ms`);
    }
  } finally {
    graySrc.delete();
    colorSrc.delete();
  }
});

// The availability gate is the only parity skip; emitted scripts are executed unchanged.
let pyAvailable = false;
let syntaxPython = false;
try {
  execFileSync("python", ["-c", "pass"], { stdio: "pipe" });
  syntaxPython = true;
  if (!process.env.IP5B_NO_PY) {
    execFileSync("python", ["-c", "import cv2,numpy"], { stdio: "pipe" });
    pyAvailable = true;
  }
} catch {
}
const PY_SKIP = { skip: !pyAvailable && "python/cv2 unavailable (or IP5B_NO_PY=1)" };
function parityParams(id) {
  if (id === "image_pyramid") return ["gaussian", "laplacian"].flatMap((kind) => ["reduce", "expand"].flatMap((direction) => ["level", "mosaic"].map((view) => ({ kind, direction, view, levels: 2 }))));
  if (id === "frangi_vesselness") return [{}, { polarity: "dark" }, { sigmaMin: 3, sigmaMax: 1, beta: 0.1 }, { sigmaMin: 1.5, sigmaMax: 2.5, beta: 2 }];
  if (id === "anisotropic_diffusion") return ["exponential", "quadratic"].flatMap((conduction) => [1, 10, 50].map((iterations) => ({ conduction, iterations, kappa: iterations === 1 ? 100 : 20 })));
  if (id === "debayer") return ["rggb", "bggr", "grbg", "gbrg"].flatMap((pattern) => ["mosaic", "demosaic"].flatMap((mode) => ["bilinear", "ea"].map((method) => ({ pattern, mode, method }))));
  return [{}];
}
test("ip5b-P-syntax. every generated branch, malformed params, sequences and Pass/Fail compile with Python", { skip: !syntaxPython && "python unavailable" }, () => {
  const codes = [];
  for (const id of IDS) for (const p of [...parityParams(id), { mode: "constructor", kind: "constructor", levels: 900, kappa: null, iterations: -9, sigmaMin: 900, sigmaMax: -8 }]) {
    const code = buildPythonCode([step(id, p)]);
    codes.push(code);
    assert.ok(!/WARNING: skipped|ยังไม่ได้ต่อสาย B/.test(code));
    for (const line of code.split("\n").filter((l) => l.trimStart().startsWith("print("))) assert.ok(/^[\x00-\x7f]*$/.test(line), line);
  }
  codes.push(buildPythonCode([step("piqe"), step("pass_fail", { metric: "piqe" })]));
  for (const pattern of ["rggb", "bggr", "grbg", "gbrg"]) codes.push(buildPythonCode([step("debayer", { mode: "mosaic", pattern }), step("debayer", { pattern }), step("frangi_vesselness"), step("piqe")]));
  execFileSync("python", ["-c", "import sys,json; codes=json.loads(sys.stdin.buffer.read()); [compile(c, 'exported_'+str(i), 'exec') for i,c in enumerate(codes)]; print(len(codes))"], { input: JSON.stringify(codes), stdio: "pipe" });
});
function pythonCases(jobs) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ip5b-parity-"));
  try {
    const manifest = jobs.map((job, i) => {
      let pipe = job.pipe || [step(job.id, job.p)];
      if (job.src.channels() === 1) pipe = [step("grayscale"), ...pipe];
      let code = buildPythonCode(pipe);
      code = code.slice(0, code.indexOf('if __name__ == "__main__":'));
      const rgbInput = new cv.Mat();
      const bgr = new cv.Mat();
      try {
        // Always feed three channels: cv2.imread is BGR, even for gray PNGs.
        if (job.src.channels() === 1) cv.cvtColor(job.src, rgbInput, cv.COLOR_GRAY2RGB);
        else if (job.src.channels() === 4) cv.cvtColor(job.src, rgbInput, cv.COLOR_RGBA2RGB);
        else job.src.copyTo(rgbInput);
        cv.cvtColor(rgbInput, bgr, cv.COLOR_RGB2BGR);
        const input = path.join(dir, `i${i}.bin`);
        fs.writeFileSync(input, Buffer.from(bgr.data));
        code = code.replace("cv2.imread(path)", `np.fromfile(path, dtype=np.uint8).reshape((${job.src.rows}, ${job.src.cols}, 3))`);
        if (job.tail) code = code.replace("\n    return img\n", "\n    " + job.tail + "\n    return img\n");
        const file = path.join(dir, `c${i}.py`);
        const out = path.join(dir, `o${i}.bin`);
        fs.writeFileSync(file, code);
        return { file, input, out };
      } finally {
        rgbInput.delete();
        bgr.delete();
      }
    });
    fs.writeFileSync(path.join(dir, "manifest.json"), JSON.stringify(manifest));
    fs.writeFileSync(path.join(dir, "driver.py"), `import json,io,contextlib,sys
results=[]
for c in json.load(open(sys.argv[1])):
    ns={'__name__':'case'}
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(open(c['file'],encoding='utf8').read(),c['file'],'exec'),ns)
        result=ns['process'](c['input'])
    result.tofile(c['out'])
    results.append({'shape':result.shape,'stdout':buf.getvalue()})
json.dump(results,open(sys.argv[2],'w'))
`);
    execFileSync("python", [path.join(dir, "driver.py"), path.join(dir, "manifest.json"), path.join(dir, "results.json")], { stdio: "pipe", env: { ...process.env, PYTHONIOENCODING: "utf8" }, maxBuffer: 1 << 25 });
    return JSON.parse(fs.readFileSync(path.join(dir, "results.json"), "utf8")).map((r, i) => {
      const bytes = fs.readFileSync(manifest[i].out);
      if (r.shape.length === 3) for (let k = 0; k < bytes.length; k += 3) [bytes[k], bytes[k + 2]] = [bytes[k + 2], bytes[k]];
      return { ...r, bytes };
    });
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}
function appPipe(pipe, src) {
  let current = src.clone();
  let context = {};
  try {
    for (const st of pipe) {
      const filter = FILTERS[st.id];
      let grayInput = null;
      context = {};
      try {
        if (needsOf(filter, st.params) === "gray" && current.channels() > 1) {
          grayInput = new cv.Mat();
          cv.cvtColor(current, grayInput, cv.COLOR_RGB2GRAY);
        }
        const next = filter.apply(cv, grayInput || current, st.params, context);
        current.delete();
        current = next;
      } finally {
        if (grayInput) grayInput.delete();
      }
    }
    return { image: current, context };
  } catch (e) {
    current.delete();
    throw e;
  }
}
function compareParity(job, result, t) {
  const app = appPipe(job.pipe || [step(job.id, job.p)], job.src);
  try {
    const out = app.image;
    assert.deepEqual(result.shape, out.channels() === 1 ? [out.rows, out.cols] : [out.rows, out.cols, out.channels()]);
    assert.equal(result.bytes.length, out.data.length);
    let max = 0;
    let differingBytes = 0;
    const pixels = new Set();
    for (let i = 0; i < result.bytes.length; i++) {
      const diff = Math.abs(result.bytes[i] - out.data[i]);
      if (diff) {
        differingBytes++;
        pixels.add(Math.floor(i / out.channels()));
      }
      max = Math.max(max, diff);
    }
    // วัดจริงแล้วตรงกันทุกไบต์ (รวม frangi/anisotropic) จึงไม่ยอมให้คลาดเคลื่อน
    assert.ok(max <= 0, `${job.id} max diff ${max}`);
    assert.ok(pixels.size <= 0, `${job.id} differing pixels ${pixels.size}`);
    t.diagnostic(`${job.id || "sequence"} ${JSON.stringify(job.p || {})}: max ${max}, differing bytes ${differingBytes}, pixels ${pixels.size}`);
    if (job.id === "piqe" && out.cols >= 16 && out.rows >= 16) {
      const printed = JSON.parse(result.stdout.match(/VALUES (\{.*\})/)[1]);
      const values = vals(app.context);
      assert.ok(Math.abs(printed.piqe - values.piqe) <= 1e-6);
      for (const key of ["piqe_active", "piqe_artifact", "piqe_noise"]) assert.equal(printed[key], values[key]);
    }
  } finally {
    app.image.delete();
  }
}
for (const id of IDS) test(`ip5b-P-${id}. real Python parity across every branch, singleton, odd RGB and gray fixtures`, PY_SKIP, (t) => {
  const c = rgb(37, 35, (x, y) => [(x * 17 + y * 7) & 255, (x * 11 + y * 3) & 255, (x * 5 + y * 23) & 255]);
  const g = texture(35, 33);
  const one = gray(1, 1, () => 113);
  const jpegSrc = id === "piqe" ? texture(128, 128) : null, jpeg = jpegSrc && run("dct_jpeg", jpegSrc, { quality: 5 });
  try {
    if (jpeg) {
      // texture(128,128) -> dct_jpeg q5: 64 active, 21 artifact บล็อก (ผ่านครบทุกกิ่งของ PIQE จริง)
      const jc = {};
      run("piqe", jpeg, {}, jc).delete();
      assert.equal(vals(jc).piqe_active, 64);
      assert.equal(vals(jc).piqe_artifact, 21);
    }
    const jobs = [c, g, one, ...(jpeg ? [jpeg] : [])].flatMap((src) => parityParams(id).map((p) => ({ id, src, p, tail: id === "piqe" ? "import json; print('VALUES ' + json.dumps({'piqe':piqe,'piqe_active':piqe_active,'piqe_artifact':piqe_artifact,'piqe_noise':piqe_noise}))" : undefined })));
    const results = pythonCases(jobs);
    for (let i = 0; i < jobs.length; i++) compareParity(jobs[i], results[i], t);
  } finally {
    for (const m of [c, g, one, jpegSrc, jpeg]) if (m) m.delete();
  }
});
test("ip5b-P-sequences. Bayer mosaic/demosaic, color-space channel order and downstream grayscale", PY_SKIP, (t) => {
  const src = rgb(33, 31, (x, y) => [(x * 7 + y * 11) & 255, (x * 3 + y * 5) & 255, (x * 13 + y * 17) & 255]);
  try {
    const pipes = [];
    for (const pattern of ["rggb", "bggr", "grbg", "gbrg"]) pipes.push([step("debayer", { mode: "mosaic", pattern }), step("debayer", { pattern }), step("grayscale")]);
    for (const id of IDS) pipes.push([step("colorspace", { space: "hsv" }), step(id), step("colorspace_inverse", { space: "hsv" })]);
    const jobs = pipes.map((pipe) => ({ src, pipe }));
    const results = pythonCases(jobs);
    for (let i = 0; i < jobs.length; i++) compareParity(jobs[i], results[i], t);
  } finally {
    src.delete();
  }
});

// Measured twice on clean HEAD 0129f4d before any production edit.
// Re-pinned for rgb_adjust rounding: only its five Python fields changed.
test("ip5b-G. 550 pre-edit cases retain pixels, info, values and Python except approved rgb_adjust rounding", () => {
  const result = {};
  const hosts = ['geometry', 'color_tools', 'segment_edge', 'filter_denoise', 'image_quality'];
  const oldIds = ["crop","roi_polygon","resize","rotate","flip","perspective_warp","undistort","translate_shear","grayscale","colorspace","colorspace_inverse","channel_split","rgb_adjust","color_inrange","kmeans_segment","slic","white_balance","invert","pseudocolor","binary_threshold","otsu_threshold","adaptive_threshold","multi_otsu_threshold","canny","sobel","laplacian_edge","scharr_edge","prewitt_edge","roberts_edge","log_edge","watershed","watershed_marker","grabcut","region_growing","salt_pepper_noise","gaussian_noise","median_blur","gaussian_blur","box_blur","bilateral","wiener_filter","guided_filter","nlm_denoise","deconv_wiener","deconv_lucy","psnr","ssim","delta_e","seg_score"];
  for (const [kind, w, h] of [['rgb', 37, 29], ['gray', 37, 29], ['rgba', 37, 29], ['rgb', 1, 1], ['empty', 0, 0]]) {
    const data = Uint8Array.from({length: w * h * 3}, (_, i) => (Math.imul(i + 11, 73) ^ (i >> 4) * 17) & 255);
    const c = w ? cv.matFromArray(h, w, cv.CV_8UC3, data) : new cv.Mat();
    const src = new cv.Mat();
    if (kind === 'gray') cv.cvtColor(c, src, cv.COLOR_RGB2GRAY);
    else if (kind === 'rgba') cv.cvtColor(c, src, cv.COLOR_RGB2RGBA);
    else c.copyTo(src);
    try {
      for (const fid of hosts) for (const {value: id} of FILTERS[fid].params.find(q => q.key === 'mode').options) {
        if (!OPS[id] || !oldIds.includes(id)) continue;
        const defs = Object.fromEntries(OPS[id].params.map(q => [q.key, q.default]));
        const variants = [{}];
        for (const q of OPS[id].params.filter(q => q.type === 'select')) for (const {value} of q.options) if (value !== q.default) variants.push({[q.key]: value});
        for (const extra of variants) {
          const local = {...defs, ...extra};
          const params = {mode: id, ...Object.fromEntries(Object.entries(local).map(([k,v]) => [id + '__' + k,v]))};
          const st = {id: fid, params};
          const key = JSON.stringify([kind,w,h,fid,id,extra]);
          const ctx = {src2: c};
          let g = null;
          const random = Math.random;
          let seed = 17;
          Math.random = () => ((seed = Math.imul(seed, 1664525) + 1013904223 >>> 0) / 4294967296);
          try {
            if (w && needsOf(FILTERS[fid], params) === 'gray' && src.channels() > 1) {
              g = new cv.Mat();
              cv.cvtColor(src,g,src.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY);
            }
            const out = FILTERS[fid].apply(cv,g || src,params,ctx);
            result[key] = {image: hash(Buffer.from(out.data)), rows:out.rows, cols:out.cols, ch:out.channels(), info:ctx.info, values:ctx.values, python:hash(buildPythonCode([st]))};
            out.delete();
          } catch (e) {
            result[key] = {error: typeof e === 'number' ? 'OpenCV native exception' : String(e), python:hash(buildPythonCode([st]))};
          } finally {
            Math.random = random;
            if (g) g.delete();
          }
        }
      }
    } finally {
      src.delete();
      c.delete();
    }
  }
  assert.equal(Object.keys(result).length, 550);
  assert.equal(hash(JSON.stringify(result)), "7803e35b87e619fea92117bd053dc224595a8c93965846a74400d0086480931c");
});

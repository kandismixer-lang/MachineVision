// IP batch 5D: behavioral fixtures, frozen clean-tree goldens and Python parity.
import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import cv from "@techstark/opencv-js";
import { OPS, FILTERS, defaultParams, needsOf, passFailVerdict, deltaE94, simplifyClosedContour5D } from "../src/renderer/filters/registry.js";
import { TOOL_SECTIONS, OP_TO_FILTER } from "../src/renderer/ui/palette-panel.js";
import { OP_ICON, GROUP_ICON, OP_PRIMARY, OPTION_GROUPS } from "../src/renderer/filters/ui-meta.js";
import { buildPythonCode, PF_VALUE_OPS } from "../src/renderer/codegen.js";
import { assertLeakFree, assertWrapperBalance, makeThrowingCv } from "./_leak-helpers.mjs";
while (typeof cv.Mat !== "function")
    await new Promise((r) => setTimeout(r, 20));
const IDS = ["rank_filter", "custom_lut", "image_stats", "corr2", "gradient_dir"];
const defaults = (id) => Object.fromEntries(OPS[id].params.map((q) => [q.key, q.default]));
const hash = (x) => crypto.createHash("sha256").update(x).digest("hex");
const vals = (ctx) => Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));
function image(w, h, fn, ch = 1) {
    const data = new Uint8Array(w * h * ch);
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            const v = fn(x, y);
            if (ch === 1)
                data[y * w + x] = v;
            else
                data.set(v, (y * w + x) * ch);
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
function legacyParams(id) {
    return OPS[id].params.filter(q => !(id === "hough_lines" && q.key === "view") && !(id === "find_contours" && q.key === "simplify"));
}
test("ip5d-G. 730 clean-tree cases freeze pixels, info, values and Python bytes", () => {
    const result = {};
    for (const [kind, w, h] of [["rgb", 37, 29], ["gray", 37, 29], ["rgba", 37, 29], ["rgb", 1, 1], ["empty", 0, 0]]) {
        const data = Uint8Array.from({ length: w * h * 3 }, (_, i) => (Math.imul(i + 11, 73) ^ (i >> 4) * 17) & 255);
        const c = w ? cv.matFromArray(h, w, cv.CV_8UC3, data) : new cv.Mat(), src = new cv.Mat();
        if (kind === "gray")
            cv.cvtColor(c, src, cv.COLOR_RGB2GRAY);
        else if (kind === "rgba")
            cv.cvtColor(c, src, cv.COLOR_RGB2RGBA);
        else
            c.copyTo(src);
        try {
            for (const [fid, ids] of Object.entries(OLD_HOSTS))
                for (const id of ids) {
                    const qs = legacyParams(id), defs = Object.fromEntries(qs.map(q => [q.key, q.default]));
                    const variants = [{}, ...qs.filter(q => q.type === "select").flatMap(q => q.options.filter(o => o.value !== q.default && !(id === "delta_e" && o.value === "cie94") && !(id === "pass_fail" && ["mean", "std", "entropy", "corr2"].includes(o.value))).map(o => ({ [q.key]: o.value })))];
                    for (const extra of variants) {
                        const local = { ...defs, ...extra }, params = { mode: id, ...Object.fromEntries(Object.entries(local).map(([k, v]) => [id + "__" + k, v])) }, st = { id: fid, params };
                        const key = JSON.stringify([kind, w, h, fid, id, extra]), ctx = { src2: c };
                        let g = null;
                        try {
                            if (w && needsOf(FILTERS[fid], params) === "gray" && src.channels() > 1) {
                                g = new cv.Mat();
                                cv.cvtColor(src, g, src.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY);
                            }
                            // Salt/pepper uses Math.random. Seed only that op; original 85eafc5 source
                            // was replayed from OS temp to freeze these five stochastic pixel cases.
                            const random = Math.random;
                            let seed = 12345, out;
                            try {
                                if (id === "salt_pepper_noise")
                                    Math.random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
                                out = FILTERS[fid].apply(cv, g || src, params, ctx);
                            }
                            finally {
                                Math.random = random;
                            }
                            try {
                                result[key] = { image: hash(Buffer.from(out.data)), rows: out.rows, cols: out.cols, ch: out.channels(), info: ctx.info, values: ctx.values, python: hash(buildPythonCode([st])) };
                            }
                            finally {
                                out.delete();
                            }
                        }
                        catch (e) {
                            result[key] = { error: typeof e === "number" ? "OpenCV native exception" : String(e), python: hash(buildPythonCode([st])) };
                        }
                        finally {
                            g?.delete();
                        }
                    }
                }
        }
        finally {
            src.delete();
            c.delete();
        }
    }
    assert.equal(Object.keys(result).length, 730);
    assert.equal(hash(JSON.stringify(result)), "34676ca578de0243d261b7dfa852c0336bdd49cca1b7cc303d8601d2fb28edd8");
});
test("ip5d-S. defaults, ranges, placement, icons, labels and Pass/Fail", () => {
    for (const [id, host, after] of [["rank_filter", "filter_denoise", "median_blur"], ["custom_lut", "enhance_tools", "gamma"], ["image_stats", "analysis_detect", "euler_number"], ["corr2", "image_quality", "bf_score"], ["gradient_dir", "segment_edge", "sobel"]]) {
        assert.ok(OPS[id], id);
        assert.equal(OP_TO_FILTER[id], host);
        const list = TOOL_SECTIONS.find((s) => s.ops.includes(id)).ops;
        assert.equal(list[list.indexOf(after) + 1], id);
        assert.deepEqual(FILTERS[host].params[0].options.map((o) => o.value), list);
        assert.ok(OPS[id].desc && OPS[id].modeHint);
        assert.ok(!/ที่สุด/.test(JSON.stringify(OPS[id])));
        assert.match(OP_PRIMARY[id](defaults(id)), /[ก-๙]/);
        assert.equal(Object.values(OP_ICON).filter((g) => g === OP_ICON[id]).length, 1);
        assert.ok(![...Object.values(GROUP_ICON), "🧩", "📊", "📌"].includes(OP_ICON[id]));
        for (const q of OPS[id].params) {
            assert.ok(q.label && (q.desc || q.hint));
            assert.equal(defaultParams(host)[id + "__" + q.key], q.default);
        }
        if (OPTION_GROUPS[host + ".mode"])
            assert.ok(OPTION_GROUPS[host + ".mode"].flatMap((g) => g[1]).includes(id));
    }
    assert.deepEqual(defaults("rank_filter"), { mode: "percentile", percentile: 50, size: 3 });
    assert.deepEqual(defaults("custom_lut"), { out0: 0, in1: 64, out1: 64, in2: 128, out2: 128, in3: 192, out3: 192, out255: 255, view: "result" });
    assert.deepEqual(defaults("gradient_dir"), { method: "sobel", view: "magnitude" });
    assert.deepEqual(defaults("image_stats"), {});
    assert.deepEqual(defaults("corr2"), {});
    for (const [id, key, range] of [["rank_filter", "percentile", [0, 100, 1]], ["rank_filter", "size", [3, 15, 2]], ["find_contours", "simplify", [0, 20, 0.5]]]) {
        const q = OPS[id].params.find((q) => q.key === key);
        assert.deepEqual([q.min, q.max, q.step], range);
    }
    assert.equal(OPS.hough_lines.params.at(-1).key, "view");
    assert.equal(OPS.find_contours.params.at(-1).key, "simplify");
    assert.equal(OPS.delta_e.params[0].options.at(-1).value, "cie94");
    for (const id of ["image_stats", "corr2"])
        assert.ok(PF_VALUE_OPS.has(id));
});
function reflect(i, n) {
    while (i < 0 || i >= n)
        i = i < 0 ? -i - 1 : 2 * n - 1 - i;
    return i;
}
function rankOracle(src, mode, p, size) {
    const data = [], ch = src.channels(), r = size >> 1;
    for (let y = 0; y < src.rows; y++)
        for (let x = 0; x < src.cols; x++)
            for (let c = 0; c < ch; c++) {
                const a = [];
                for (let dy = -r; dy <= r; dy++)
                    for (let dx = -r; dx <= r; dx++)
                        a.push(src.data[(reflect(y + dy, src.rows) * src.cols + reflect(x + dx, src.cols)) * ch + c]);
                a.sort((a, b) => a - b);
                let v = a[Math.floor(p / 100 * (a.length - 1) + 0.5)];
                if (mode === "min")
                    v = a[0];
                if (mode === "max")
                    v = a.at(-1);
                if (mode === "range")
                    v = a.at(-1) - a[0];
                if (mode === "mode") {
                    v = a[0];
                    let best = 0;
                    for (const n of new Set(a)) {
                        const count = a.filter((x) => x === n).length;
                        if (count > best) {
                            v = n;
                            best = count;
                        }
                    }
                }
                data.push(v);
            }
    return Buffer.from(data);
}
test("ip5d-rank. independent sorted-window oracle, reflect boundaries and RGB", (t) => {
    for (const ch of [1, 3]) {
        const src = image(7, 5, (x, y) => ch === 1 ? (x * 11 + y * 37) % 251 : [x * 17, y * 31, (x + y) * 9], ch);
        try {
            for (const size of [3, 7, 15])
                for (const mode of ["percentile", "min", "max", "range", "mode"])
                    for (const percentile of [0, 50, 67, 100]) {
                        const ctx = {}, out = run("rank_filter", src, { size, mode, percentile }, ctx);
                        try {
                            assert.deepEqual(Buffer.from(out.data), rankOracle(src, mode, percentile, size));
                            assert.equal(ctx.info, `Rank filter · โหมด ${mode} · หน้าต่าง ${size}×${size}`);
                        }
                        finally {
                            out.delete();
                        }
                    }
        }
        finally {
            src.delete();
        }
    }
    const src = image(640, 480, (x, y) => [x & 255, y & 255, (x * 7 + y * 11) & 255], 3);
    try {
        run("rank_filter", src, { size: 7 }).delete();
        const start = performance.now();
        run("rank_filter", src, { size: 7 }).delete();
        t.diagnostic(`rank_filter 640x480 RGB size7: ${(performance.now() - start).toFixed(2)} ms`);
    }
    finally {
        src.delete();
    }
});
test("ip5d-lut. identity, inversion, stable duplicate inputs and curve controls", () => {
    const src = image(256, 1, (x) => x);
    try {
        const identity = run("custom_lut", src);
        same(identity, src);
        identity.delete();
        const ctx = {}, inv = run("custom_lut", src, { out0: 255, out1: 191, out2: 127, out3: 63, out255: 0 }, ctx);
        assert.deepEqual(Array.from(inv.data), Array.from(src.data, v => 255 - v));
        assert.match(ctx.info, /มีช่วงกลับด้าน/);
        inv.delete();
        const dup = run("custom_lut", src, { in1: 128, out1: 20, in2: 128, out2: 70, in3: 128, out3: 100 });
        assert.equal(dup.data[128], 100);
        assert.equal(dup.data[64], 50);
        dup.delete();
        const graph = run("custom_lut", src, { view: "curve" });
        assert.deepEqual([graph.cols, graph.rows, graph.channels()], [256, 256, 3]);
        assert.deepEqual(Array.from(graph.data.slice((127 * 256 + 128) * 3, (127 * 256 + 128) * 3 + 3)), [255, 0, 0]);
        graph.delete();
    }
    finally {
        src.delete();
    }
});
test("ip5d-stats. sample std, ordered entropy, bwarea patterns and Pass/Fail", () => {
    for (const [data, expected] of [[[0], [0, 0, 0, 0, 0, 0]], [[255], [255, 0, 255, 255, 0, 1]], [[0, 255], [127.5, Math.sqrt(32512.5), 0, 255, 1, 1]]]) {
        const src = image(data.length, 1, (x) => data[x]), ctx = {}, out = run("image_stats", src, {}, ctx);
        try {
            assert.deepEqual(Object.values(vals(ctx)), expected);
            for (const metric of ["mean", "std", "entropy"])
                assert.equal(passFailVerdict({ metric, cond: "ge", lo: 0 }, { op: "image_stats", values: ctx.values }).result, "OK");
        }
        finally {
            out.delete();
            src.delete();
        }
    }
    for (const [bits, area] of [[[255, 0, 0, 255], 2.25], [[255, 255, 0, 0], 2], [[255, 255, 255, 0], 3.125], [[255, 255, 255, 255], 4]]) {
        const src = image(2, 2, (x, y) => bits[y * 2 + x]), ctx = {}, out = run("image_stats", src, {}, ctx);
        assert.equal(vals(ctx).bwarea, area);
        out.delete();
        src.delete();
    }
});
test("ip5d-corr. affine invariance, inversion, flat images and missing B", () => {
    const a = image(7, 5, (x, y) => 10 + x + y * 7), b = image(7, 5, (x, y) => 30 + 2 * (x + y * 7)), inv = image(7, 5, (x, y) => 245 - x - y * 7), flat = image(2, 2, () => 5);
    try {
        for (const [src2, r] of [[a, 1], [b, 1], [inv, -1]]) {
            const ctx = { src2 }, out = run("corr2", a, {}, ctx);
            assert.equal(vals(ctx).corr2, r);
            assert.equal(ctx.info, `corr2 = ${r.toFixed(4)}`);
            out.delete();
        }
        const ctx = { src2: flat }, out = run("corr2", a, {}, ctx);
        assert.ok(!ctx.values.fields.some(f => f.key === "corr2"));
        assert.match(ctx.info, /ภาพเรียบ คำนวณไม่ได้/);
        out.delete();
        const missing = {}, pass = run("corr2", a, {}, missing);
        assert.match(missing.info, /เข้า B/);
        assert.ok(!missing.values.fields.some(f => f.key === "corr2"));
        pass.delete();
    }
    finally {
        for (const m of [a, b, inv, flat])
            m.delete();
    }
});
test("ip5d-gradient. ramps, five methods, constant and direction convention", () => {
    const ramp = image(7, 5, (x) => x * 20), zero = image(1, 1, () => 80), down = image(5, 7, (x, y) => y * 20);
    try {
        for (const method of ["sobel", "prewitt", "central", "intermediate", "roberts"])
            for (const view of ["magnitude", "direction", "direction_gray"]) {
                const ctx = {}, out = run("gradient_dir", ramp, { method, view }, ctx), black = run("gradient_dir", zero, { method, view });
                assert.ok(black.data.every(v => v === 0));
                assert.ok(ctx.info.startsWith(`Gradient (${method})`));
                if (method !== "roberts" && view === "direction_gray")
                    assert.equal(out.data[2 * 7 + 3], 128);
                if (method !== "roberts" && view === "direction")
                    assert.deepEqual(Array.from(out.data.slice((2 * 7 + 3) * 3, (2 * 7 + 3) * 3 + 3)), [255, 0, 0]);
                out.delete();
                black.delete();
            }
        const out = run("gradient_dir", down, { view: "direction_gray" });
        assert.equal(out.data[3 * 5 + 2], 64);
        out.delete();
    }
    finally {
        ramp.delete();
        zero.delete();
        down.delete();
    }
});
test("ip5d-options. CIE94 asymmetry, blank accumulator and contour reduction", () => {
    assert.equal(deltaE94(50, 40, 0, 50, 40, 0), 0);
    assert.ok(Math.abs(deltaE94(50, 40, 0, 50, 20, 0) - 20 / 2.8) < 1e-12);
    assert.ok(Math.abs(deltaE94(50, 20, 0, 50, 40, 0) - 20 / 1.9) < 1e-12);
    assert.equal(deltaE94(30, 0, 0, 40, 0, 0), 10);
    // Equal chroma, nonzero hue: delta H^2 = 3200 and S_H = 1 + 0.015 * 40 = 1.6.
    assert.equal(deltaE94(50, 40, 0, 50, 0, 40), Math.sqrt(3200 / (1.6 * 1.6)));
    const a = image(20, 20, () => [255, 0, 0], 3), b = image(20, 20, () => [0, 255, 0], 3), blank = image(11, 9, () => 0), disk = image(61, 61, (x, y) => (x - 30) ** 2 + (y - 30) ** 2 < 20 ** 2 ? 255 : 0);
    try {
        const ac = { src2: b }, bc = { src2: a }, ab = run("delta_e", a, { formula: "cie94" }, ac), ba = run("delta_e", b, { formula: "cie94" }, bc);
        assert.ok(Math.abs(vals(ac).de_mean - vals(bc).de_mean) > 1);
        assert.match(ac.info, /CIE94/);
        ab.delete();
        ba.delete();
        const ctx = {}, acc = run("hough_lines", blank, { view: "accumulator" }, ctx), nr = 2 * Math.ceil(Math.sqrt(10 ** 2 + 8 ** 2)) + 1;
        assert.deepEqual([acc.cols, acc.rows, acc.channels()], [180, nr, 3]);
        assert.ok(acc.data.every(v => v === 0));
        assert.equal(ctx.info, `Accumulator ρ×θ = ${nr}×180 · ยอด 0 จุด`);
        acc.delete();
        const c = {}, simple = run("find_contours", disk, { minArea: 0, simplify: 3 }, c);
        assert.match(c.info, /ลดเหลือ \d+ จุด \(จาก \d+\)/);
        const ns = c.info.match(/ลดเหลือ (\d+) จุด \(จาก (\d+)\)/);
        assert.ok(+ns[1] < +ns[2]);
        simple.delete();
    }
    finally {
        for (const m of [a, b, blank, disk])
            m.delete();
    }
});
test("ip5d-contour-DP. closed anchors, infinite-line distance, first ties and degenerate contours", () => {
    for (const [points, epsilon, expected] of [
        [[], 1, []],
        [[2, 3], 1, [2, 3]],
        [[2, 3, 2, 3, 2, 3], 1, [2, 3]],
        [[0, 0, 4, 0], 1, [0, 0, 4, 0]],
        [[0, 0, 2, 0, 4, 0, 4, 4, 2, 4, 0, 4], 0.5, [0, 0, 4, 0, 4, 4, 0, 4]],
        // At exactly epsilon, drop the intermediate vertex.
        [[0, 0, 2, 0, 2, 2, 0, 2], Math.sqrt(2), [0, 0, 2, 2]],
        [[0, 0, 1, 1, 2, 0, 1, -1], 1, [0, 0, 2, 0]],
        // Equal deviations of (1,2)/(3,2): keep the first, then drop the second.
        [[0, 0, 1, 2, 3, 2, 4, 0, 0, -4], 1.9, [0, 0, 1, 2, 4, 0, 0, -4]],
        // (-2,0) projects before (0,0): distance to the line < 1.8, to the segment > 1.8.
        [[0, 0, -2, 0, 2, 3, 10, 0, 0, -10], 1.8, [0, 0, 2, 3, 10, 0, 0, -10]],
    ])
        assert.deepEqual(simplifyClosedContour5D(Int32Array.from(points), epsilon), expected);
});
// Measured before editing clean HEAD 85eafc5.
const OLD_HOSTS = { "filter_denoise": ["salt_pepper_noise", "gaussian_noise", "median_blur", "gaussian_blur", "box_blur", "bilateral", "wiener_filter", "guided_filter", "anisotropic_diffusion", "nlm_denoise", "deconv_wiener", "deconv_lucy"], "enhance_tools": ["brightness_contrast", "gamma", "hist_equalize", "clahe", "contrast_stretch", "flatfield_correction", "quantize", "unsharp", "laplacian_sharpen", "custom_kernel", "dehaze", "lowlight_enhance", "local_contrast"], "analysis_detect": ["find_contours", "connected_components", "distance_transform", "local_std_filter", "local_entropy_filter", "gabor_filter", "glcm", "focus_score", "regionprops", "euler_number", "feret", "line_profile", "shape_filter", "bounding_boxes", "hough_circles", "hough_lines", "harris_corners", "shi_tomasi_corners", "orb_keypoints", "akaze_keypoints", "fast_keypoints", "pass_fail"], "image_quality": ["psnr", "ssim", "delta_e", "seg_score", "bf_score", "piqe"], "segment_edge": ["binary_threshold", "otsu_threshold", "adaptive_threshold", "multi_otsu_threshold", "canny", "sobel", "laplacian_edge", "scharr_edge", "prewitt_edge", "roberts_edge", "log_edge", "frangi_vesselness", "watershed", "watershed_marker", "grabcut", "region_growing"] };
let syntaxPython = false, pyAvailable = false;
try {
    execFileSync("python", ["-c", "pass"], { stdio: "pipe" });
    syntaxPython = true;
    if (!process.env.IP5D_NO_PY) {
        execFileSync("python", ["-c", "import cv2,numpy"], { stdio: "pipe" });
        pyAvailable = true;
    }
}
catch { }
const PY_SKIP = { skip: !pyAvailable && "python/cv2 unavailable (or IP5D_NO_PY=1)" };
function variants(id) {
    if (id === "rank_filter")
        return ["percentile", "min", "max", "range", "mode"].flatMap(mode => [3, 7, 15].map(size => ({ mode, size, percentile: 67 })));
    if (id === "custom_lut")
        return ["result", "curve"].flatMap(view => [{ view }, { view, in1: 200, in2: 64, in3: 128, out1: 10, out2: 240, out3: 80 }, { view, in1: 0, in2: 0, in3: 255, out0: 90, out1: 13, out2: 77, out3: 250, out255: 1 }]);
    if (id === "gradient_dir")
        return ["sobel", "prewitt", "central", "intermediate", "roberts"].flatMap(method => ["magnitude", "direction", "direction_gray"].map(view => ({ method, view })));
    if (id === "delta_e")
        return ["pixel", "mean_b"].flatMap(compare => ["heatmap", "over"].map(view => ({ formula: "cie94", compare, view })));
    if (id === "hough_lines")
        return [{ view: "accumulator" }];
    if (id === "find_contours")
        return [{ simplify: 0.5, minArea: 0 }, { simplify: 3, minArea: 0 }, { simplify: 20 }];
    return [{}];
}
test("ip5d-P-syntax. all branches, invalid params, channel sequences and Pass/Fail compile", { skip: !syntaxPython && "python unavailable" }, (t) => {
    const codes = [];
    for (const id of [...IDS, "delta_e", "hough_lines", "find_contours"])
        for (const p of [...variants(id), { method: "constructor", view: "__proto__", mode: "constructor", size: null, percentile: Infinity, in1: "bad", in2: NaN, out0: -99, simplify: 999 }]) {
            const st = step(id, p);
            if (OP_TO_FILTER[id] === "image_quality")
                st.src2 = { pipeline: [] };
            codes.push(buildPythonCode([st]));
        }
    for (const metric of ["mean", "std", "entropy", "corr2"]) {
        const st = step(metric === "corr2" ? "corr2" : "image_stats");
        if (metric === "corr2")
            st.src2 = { pipeline: [] };
        codes.push(buildPythonCode([st, step("pass_fail", { metric, cond: "ge", lo: 0.5 })]));
    }
    for (const view of ["result", "curve"])
        codes.push(buildPythonCode([step("colorspace", { space: "lab" }), step("custom_lut", { view }), step("image_stats")]));
    execFileSync("python", ["-c", "import sys,json; codes=json.loads(sys.stdin.buffer.read()); [compile(c, 'exported_'+str(i), 'exec') for i,c in enumerate(codes)]"], { input: JSON.stringify(codes), stdio: "pipe" });
    t.diagnostic(`${codes.length} exported scripts compiled`);
});
test("ip5d-L. empty/gray/RGB/RGBA, heap, wrappers and throwing contexts", () => {
    const rgb = image(13, 11, (x, y) => [x * 17, y * 21, (x + y) * 9], 3), gray = image(13, 11, (x, y) => (x * 31 + y * 17) & 255), rgba = image(7, 5, (x, y) => [x * 11, y * 19, 200, 17], 4), one = image(1, 1, () => 70), empty = new cv.Mat();
    const cases = [...IDS.map(id => [id, {}]), ["hough_lines", { view: "accumulator" }], ["find_contours", { simplify: 3, minArea: 0 }], ["custom_lut", { view: "curve" }], ["gradient_dir", { view: "direction" }], ["delta_e", { formula: "cie94" }]];
    try {
        for (const [id, p] of cases) {
            for (const src of [empty, one, gray, rgb, rgba]) {
                const ctx = { src2: one }, out = run(id, src, p, ctx);
                assert.ok(typeof ctx.info === "string");
                if (src === empty)
                    assert.equal(out.rows * out.cols, 0);
                out.delete();
            }
            assertLeakFree(cv, id, c => run(id, rgba, p, { src2: rgb }, c).delete(), { iters: 30, warmup: 5 });
            const throwing = { src2: rgb, set info(v) { throw new Error("context write"); } };
            assertWrapperBalance(cv, id + " context", c => run(id, rgba, p, throwing, c), { expectThrow: /context write/ });
        }
        for (const [id, p, method] of [["hough_lines", { view: "accumulator" }, "Canny"], ["find_contours", { simplify: 3 }, "matFromArray"], ["find_contours", { simplify: 3, minArea: 0 }, "drawContours"], ["image_stats", {}, "cvtColor"], ["corr2", {}, "resize"]]) {
            assertWrapperBalance(cv, id + " injected " + method, c => run(id, rgb, p, { src2: one }, makeThrowingCv(c, method)), { expectThrow: /injected test failure/ });
        }
        for (const id of ["image_stats", "corr2"]) {
            const ctx = { set values(v) { throw new Error("values write"); }, src2: rgb };
            assertWrapperBalance(cv, id + " values setter", c => run(id, rgba, {}, ctx, c), { expectThrow: /values write/ });
        }
    }
    finally {
        for (const m of [rgb, gray, rgba, one, empty])
            m.delete();
    }
});
test("ip5d-defaults. missing/new defaults retain exact existing output and Python", () => {
    const src = image(47, 39, (x, y) => ((x - 24) ** 2 + (y - 20) ** 2 < 120 ? 255 : 0));
    try {
        for (const [id, key, value] of [["hough_lines", "view", "lines"], ["find_contours", "simplify", 0]]) {
            const p = defaults(id);
            delete p[key];
            const a = {}, b = {}, old = OPS[id].apply(cv, src, p, a), out = run(id, src, { [key]: value }, b);
            same(old, out);
            assert.deepEqual(a, b);
            const fid = OP_TO_FILTER[id], st = { id: fid, params: { mode: id, ...Object.fromEntries(Object.entries(p).map(([k, v]) => [id + "__" + k, v])) } };
            assert.equal(buildPythonCode([st]), buildPythonCode([step(id, { [key]: value })]));
            old.delete();
            out.delete();
        }
    }
    finally {
        src.delete();
    }
});
test("ip5d-corr-no-B. disconnected correlation clears earlier Pass/Fail measurements", () => {
    const connected = step("corr2");
    connected.src2 = { pipeline: [] };
    const code = buildPythonCode([connected, step("corr2"), step("pass_fail", { metric: "corr2", cond: "ge", lo: 0.9 })]);
    assert.equal((code.match(/corr2_value = None/g) || []).length, 2);
    assert.equal((code.match(/pf_up = \{'value': corr2_value\}/g) || []).length, 2);
    const afterGray = buildPythonCode([step("grayscale"), step("corr2")]);
    assert.match(afterGray, /img = cv2.cvtColor\(img, cv2.COLOR_GRAY2BGR\) if img.ndim == 2 else img/);
    const guarded = buildPythonCode([connected]);
    const guardIndex = guarded.indexOf("if img_b.size == 0:");
    assert.ok(guardIndex >= 0 && guardIndex < guarded.indexOf("b = img_b if img_b.ndim"));
});
test("ip5d-stats-RGB. measurement preserves RGB pixels and has valid host conversion", () => {
    const src = image(9, 7, (x, y) => [x * 27, y * 39, (x + y) * 13], 3), ctx = {};
    const st = step("image_stats"), out = FILTERS[st.id].apply(cv, src, st.params, ctx);
    try {
        assert.equal(needsOf(FILTERS[st.id], st.params), "any");
        same(out, src);
        assert.match(ctx.info, /^mean \d+\.\d{2} · std \d+\.\d{2} · min \d+ · max \d+ · entropy \d+\.\d{4} bit · bwarea \d+\.\d{2} px$/);
    }
    finally {
        out.delete();
        src.delete();
    }
});
test("ip5d-accumulator-votes. origin votes, row-major ties, ten peaks and clipped nonwrapping frames", () => {
    const src = image(3, 3, () => 0), ctx = {};
    const originEdges = new Proxy(cv, { get(target, key) {
            if (key !== "Canny")
                return target[key];
            return (input, edges) => { edges.create(input.rows, input.cols, cv.CV_8UC1); edges.data.fill(0); edges.data[0] = 255; };
        } });
    const out = run("hough_lines", src, { view: "accumulator" }, ctx, originEdges);
    try {
        assert.deepEqual([out.rows, out.cols], [7, 180]);
        assert.equal(ctx.info, "Accumulator ρ×θ = 7×180 · ยอด 10 จุด");
        const expected = new Uint8Array(7 * 180 * 3);
        for (let x = 0; x < 180; x++)
            expected.fill(255, (3 * 180 + x) * 3, (3 * 180 + x) * 3 + 3);
        for (let peak = 0; peak < 10; peak++)
            for (let dy = -2; dy <= 2; dy++)
                for (let dx = -2; dx <= 2; dx++) {
                    if (Math.abs(dx) !== 2 && Math.abs(dy) !== 2)
                        continue;
                    const x = peak * 3 + dx, y = 3 + dy;
                    if (x >= 0 && x < 180)
                        expected.set([255, 0, 0], (y * 180 + x) * 3);
                }
        assert.deepEqual(Buffer.from(out.data), Buffer.from(expected));
    }
    finally {
        out.delete();
        src.delete();
    }
});
test("ip5d-rank-heap. destination allocation can grow WASM without detaching input pixels", () => {
    const src = image(5, 5, () => [73, 91, 123], 3);
    let grew = false;
    const growingMat = new Proxy(cv.Mat, { construct(target, args) {
            if (!grew) {
                const old = cv.HEAPU8.buffer, ptr = cv._malloc(cv.HEAPU8.length);
                assert.ok(ptr);
                cv._free(ptr);
                assert.notEqual(cv.HEAPU8.buffer, old);
                grew = true;
            }
            return Reflect.construct(target, args);
        } });
    const growCv = new Proxy(cv, { get(target, key) { return key === "Mat" ? growingMat : target[key]; } });
    let out;
    try {
        out = run("rank_filter", src, {}, {}, growCv);
        assert.ok(grew);
        same(out, src);
    }
    finally {
        out?.delete();
        src.delete();
    }
});
function pythonCases(jobs) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ip5d-parity-"));
    function bytes(m) {
        if (m.rows * m.cols === 0)
            return Buffer.alloc(0);
        const rgb = new cv.Mat(), bgr = new cv.Mat();
        try {
            // imread always supplies three channels, including for gray source fixtures.
            if (m.channels() === 1)
                cv.cvtColor(m, rgb, cv.COLOR_GRAY2RGB);
            else if (m.channels() === 4)
                cv.cvtColor(m, rgb, cv.COLOR_RGBA2RGB);
            else
                m.copyTo(rgb);
            cv.cvtColor(rgb, bgr, cv.COLOR_RGB2BGR);
            return Buffer.from(bgr.data);
        }
        finally {
            rgb.delete();
            bgr.delete();
        }
    }
    try {
        const manifest = jobs.map((j, i) => {
            let pipe = j.pipe || [step(j.id, j.p)];
            if (j.b)
                pipe.find((st) => st.id === OP_TO_FILTER[j.id]).src2 = { pipeline: [] };
            if (j.src.channels() === 1)
                pipe = [step("grayscale"), ...pipe];
            let code = buildPythonCode(pipe);
            code = code.slice(0, code.indexOf('if __name__ == "__main__":'));
            const inputs = [j.src, ...(j.b ? [j.b] : [])].map((m, k) => {
                const input = path.join(dir, `i${i}_${k}.bin`);
                fs.writeFileSync(input, bytes(m));
                const arg = j.b ? k ? "path_b" : "path_a" : "path";
                code = code.replace(`cv2.imread(${arg})`, `np.fromfile(${arg}, dtype=np.uint8).reshape((${m.rows}, ${m.cols}, 3))`);
                return input;
            });
            if (j.tail)
                code = code.replace("\n    return img\n", "\n    " + j.tail + "\n    return img\n");
            const file = path.join(dir, `c${i}.py`), out = path.join(dir, `o${i}.bin`);
            fs.writeFileSync(file, code);
            return { file, inputs, out };
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
            if (r.shape.length === 3)
                for (let k = 0; k < data.length; k += 3)
                    [data[k], data[k + 2]] = [data[k + 2], data[k]];
            return { ...r, data };
        });
    }
    finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
}
function workerRun(id, src, p, ctx) {
    const st = step(id, p);
    let gray = null;
    try {
        if (needsOf(FILTERS[st.id], st.params) === "gray" && src.channels() > 1) {
            gray = new cv.Mat();
            cv.cvtColor(src, gray, src.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY);
        }
        return run(id, gray || src, p, ctx);
    }
    finally {
        gray?.delete();
    }
}
function noisyMask5D() {
    let seed = 0x5d2026;
    const noise = image(407, 301, () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed >>> 24;
    }), blurred = new cv.Mat(), mask = new cv.Mat();
    try {
        cv.GaussianBlur(noise, blurred, new cv.Size(7, 7), 1.5, 1.5, cv.BORDER_REFLECT_101);
        cv.threshold(blurred, mask, 127, 255, cv.THRESH_BINARY);
        return mask;
    }
    catch (error) { mask.delete(); throw error; }
    finally { noise.delete(); blurred.delete(); }
}
test("ip5d-contour-float64. generated Python and JS keep identical noisy contour vertices", { skip: !syntaxPython && "python unavailable" }, () => {
    const src = noisyMask5D(), contours = new cv.MatVector(), hierarchy = new cv.Mat();
    try {
        cv.findContours(src, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        const jobs = [];
        for (let i = 0; i < contours.size(); i++) {
            const c = contours.get(i);
            try { for (const epsilon of [0.5, 2]) jobs.push({ points: Array.from(c.data32S), epsilon }); }
            finally { c.delete(); }
        }
        const code = buildPythonCode([step("find_contours", { simplify: 2 })]);
        const match = code.match(/    def simplify_closed_contour\(points, epsilon\):\n[\s\S]*?(?=    contours, _ =)/);
        assert.ok(match, "export must contain the shared float64 simplifier");
        const helper = match[0].replace(/^    /gm, "");
        const script = helper + "\nimport json,sys\njobs=json.loads(sys.stdin.buffer.read())\njson.dump([simplify_closed_contour(list(zip(j['points'][::2],j['points'][1::2])),j['epsilon']) for j in jobs],sys.stdout)\n";
        const results = JSON.parse(execFileSync("python", ["-c", script], { input: JSON.stringify(jobs), maxBuffer: 1 << 24 }));
        for (let i = 0; i < jobs.length; i++)
            assert.deepEqual(results[i].flat(), simplifyClosedContour5D(jobs[i].points, jobs[i].epsilon), `contour ${i}`);
        assert.ok(jobs.some(j => j.points.length > 200), "fixture must exercise a long noisy contour");
    }
    finally { src.delete(); contours.delete(); hierarchy.delete(); }
});
test("ip5d-P-contour-noise. noisy binary simplify 0.5/2 has exact info and bytes", PY_SKIP, () => {
    const src = noisyMask5D();
    try {
        const jobs = [0.5, 2].map(simplify => ({ id: "find_contours", src, p: { simplify, minArea: 0 } }));
        const results = pythonCases(jobs);
        for (const [i, job] of jobs.entries()) {
            const ctx = {}, out = workerRun(job.id, src, job.p, ctx);
            try {
                assert.deepEqual(results[i].shape, [out.rows, out.cols, 3]);
                assert.deepEqual(results[i].data, Buffer.from(out.data), `simplify ${job.p.simplify}`);
                const m = results[i].stdout.match(/objects (\d+); simplified (\d+) points \(from (\d+)\)/);
                assert.ok(m, results[i].stdout);
                assert.equal(ctx.info, `พบวัตถุ ${m[1]} ชิ้น · ลดเหลือ ${m[2]} จุด (จาก ${m[3]})`);
            }
            finally { out.delete(); }
        }
    }
    finally { src.delete(); }
});
for (const id of [...IDS, "delta_e", "hough_lines", "find_contours"])
    test(`ip5d-P-${id}. real exported Python pixels and measurements`, PY_SKIP, (t) => {
        const rgb = image(19, 13, (x, y) => [(x * 11 + y * 3) % 256, (x * 5 + y * 17) % 256, (x * 19 + y * 7) % 256], 3);
        const gray = image(17, 11, (x, y) => (x * 37 + y * 19) % 256);
        const rgba = image(15, 9, (x, y) => [x * 17, y * 31, (x + y) * 9, 27], 4);
        const one = image(1, 1, () => [31, 70, 140], 3);
        const flat = image(13, 7, () => [90, 90, 90], 3);
        const mask = image(41, 39, (x, y) => (x - 20) ** 2 + (y - 19) ** 2 < 14 ** 2 ? 255 : 0);
        const line = image(61, 43, (x, y) => x === 20 || y === 10 ? 255 : 0);
        const wide = image(701, 3, (x, y) => x === 350 ? 255 : 0);
        const owned = [rgb, gray, rgba, one, flat, mask, line, wide], jobs = [];
        try {
            const inputs = id === "find_contours" ? [mask, one, flat] : id === "hough_lines" ? [line, flat, one, wide] : [rgb, gray, rgba, one, flat];
            for (const src of inputs)
                for (const p of variants(id))
                    jobs.push({ id, src, p, ...(["corr2", "delta_e"].includes(id) ? { b: src } : {}) });
            if (id === "corr2" || id === "delta_e") {
                for (const p of variants(id))
                    jobs.push({ id, src: rgb, b: gray, p });
                const inv = image(19, 13, (x, y) => 255 - (x * 5 + y * 7)), ramp = image(19, 13, (x, y) => x * 5 + y * 7);
                owned.push(inv, ramp);
                jobs.push({ id, src: ramp, b: inv, p: id === "delta_e" ? { formula: "cie94" } : {} });
            }
            if (id === "corr2") {
                const empty = new cv.Mat();
                owned.push(empty);
                jobs.push({ id, src: rgb, b: empty, p: {} });
            }
            if (id === "rank_filter") {
                // Several row bands at size 15, with channel differences and frequent mode ties.
                const bands = image(641, 83, (x, y) => [(x * 11 + y * 3) & 255, (x * 5 + y * 17) & 255, (x * 19 + y * 7) & 255], 3);
                owned.push(bands);
                for (const mode of ["percentile", "min", "max", "range", "mode"])
                    jobs.push({ id, src: bands, p: { mode, size: 15, percentile: 67 } });
                for (const percentile of [0, 100])
                    jobs.push({ id, src: bands, p: { mode: "percentile", size: 7, percentile } });
            }
            const results = pythonCases(jobs);
            let worst = 0, most = 0;
            for (let j = 0; j < jobs.length; j++) {
                const job = jobs[j], ctx = { src2: job.b }, out = workerRun(id, job.src, job.p, ctx), py = results[j];
                try {
                    assert.deepEqual(py.shape, out.channels() === 1 ? [out.rows, out.cols] : [out.rows, out.cols, out.channels()], `${id} shape case ${j}`);
                    let max = 0, count = 0;
                    for (let i = 0; i < out.data.length; i++) {
                        const d = Math.abs(out.data[i] - py.data[i]);
                        if (d)
                            count++;
                        if (d > max)
                            max = d;
                    }
                    worst = Math.max(worst, max);
                    most = Math.max(most, count);
                    const tolerance = id === "gradient_dir" && ["direction", "direction_gray"].includes(job.p.view) ? 1 : 0;
                    assert.ok(max <= tolerance, `${id} case ${j} ${JSON.stringify(job.p)} max=${max}, differing bytes=${count}`);
                    assert.ok(!/WARNING|skipped/.test(py.stdout));
                    if (id === "image_stats") {
                        const match = py.stdout.match(/mean ([\d.e+-]+) std ([\d.e+-]+) min (\d+) max (\d+) entropy ([\d.e+-]+) bwarea ([\d.e+-]+)/);
                        assert.ok(match, py.stdout);
                        for (const [k, key] of ["mean", "std", "min", "max", "entropy", "bwarea"].entries())
                            assert.ok(Math.abs(+match[k + 1] - vals(ctx)[key]) <= 1e-12 * Math.max(1, Math.abs(vals(ctx)[key])), `${key}: ${match[k + 1]} vs ${vals(ctx)[key]}`);
                    }
                    if (id === "corr2") {
                        const match = py.stdout.match(/corr2 = ([\d.e+-]+)/);
                        if (match)
                            assert.ok(Math.abs(+match[1] - vals(ctx).corr2) <= 1e-12);
                        else {
                            assert.match(py.stdout, job.b.rows * job.b.cols === 0 ? /empty B/ : /flat image/);
                            assert.ok(!("corr2" in vals(ctx)));
                        }
                    }
                }
                finally {
                    out.delete();
                }
            }
            t.diagnostic(`${jobs.length} cases; max byte difference ${worst}; most differing bytes ${most}`);
        }
        finally {
            for (const m of owned)
                m.delete();
        }
    });
test("ip5d-P-gradient-memory. direction preserves every byte without six RGB sector images", PY_SKIP, (t) => {
    const src = image(41, 37, (x, y) => (x * 37 + y * 19) & 255);
    try {
        // Frozen pre-R2 sector assembly on a small fixture; compare with zero tolerance.
        const tail = "old_rgb = np.stack([np.stack(c, axis=-1) for c in [(value,t,z),(q,value,z),(z,value,t),(z,q,value),(t,z,value),(value,z,q)]]); yy, xx = np.indices(mag.shape); assert np.array_equal(img, np.floor(old_rgb[sector, yy, xx, ::-1] * 255 + 0.5).astype(np.uint8))";
        pythonCases(["sobel", "prewitt", "central", "intermediate", "roberts"].map(method => ({ id: "gradient_dir", src, p: { method, view: "direction" }, tail })));
        const code = buildPythonCode([step("gradient_dir", { view: "direction" })]).replace("cv2.imread(path)", "path");
        const peak = +execFileSync("python", ["-c", `import json,sys,io,contextlib,tracemalloc
import numpy as np
ns={'__name__':'case'}
exec(compile(json.loads(sys.stdin.buffer.read()),'gradient_export','exec'),ns)
frame=np.random.default_rng(5).integers(0,256,(480,640,3),dtype=np.uint8)
tracemalloc.start()
with contextlib.redirect_stdout(io.StringIO()):
    result=ns['process'](frame)
peak=tracemalloc.get_traced_memory()[1]
assert result.shape==(480,640,3)
assert peak < 192 * 480 * 640 + 1048576, peak
print(peak)
`], { input: JSON.stringify(code), stdio: "pipe" }).toString().trim();
        t.diagnostic(`direction export peak traced allocation 640x480: ${(peak / 1048576).toFixed(2)} MiB`);
    }
    finally { src.delete(); }
});
test("ip5d-P-rank-timing. every mode 640x480 RGB size7 matches JS and reports Python time", PY_SKIP, (t) => {
    const src = image(640, 480, (x, y) => [(x * 11 + y * 3) & 255, (x * 5 + y * 17) & 255, (x * 19 + y * 7) & 255], 3);
    try {
        const jobs = ["percentile", "min", "max", "range", "mode"].map(mode => {
            const p = { mode, size: 7, percentile: 50 }, out = run("rank_filter", src, p);
            try {
                const bytes = Buffer.from(out.data);
                for (let i = 0; i < bytes.length; i += 3) [bytes[i], bytes[i + 2]] = [bytes[i + 2], bytes[i]];
                return { mode, hash: hash(bytes), code: buildPythonCode([step("rank_filter", p)]).replace("cv2.imread(path)", "path") };
            }
            finally { out.delete(); }
        });
        const results = JSON.parse(execFileSync("python", ["-c", `import json,sys,time,statistics,hashlib,io,contextlib,tracemalloc
import numpy as np
y,x=np.indices((480,640))
frame=np.stack(((x*19+y*7)&255,(x*5+y*17)&255,(x*11+y*3)&255),axis=-1).astype(np.uint8)
results=[]
for job in json.loads(sys.stdin.buffer.read()):
    ns={'__name__':'case'}
    exec(compile(job['code'],'rank_export','exec'),ns)
    with contextlib.redirect_stdout(io.StringIO()):
        ns['process'](frame)
        times=[]
        for _ in range(3):
            start=time.perf_counter()
            result=ns['process'](frame)
            times.append((time.perf_counter()-start)*1000)
    assert hashlib.sha256(result.tobytes()).hexdigest()==job['hash'], job['mode']
    record={'mode':job['mode'],'median_ms':statistics.median(times)}
    if job['mode'] in ('percentile','mode'):
        large=np.random.default_rng(5).integers(0,256,(1080,1920,3),dtype=np.uint8)
        tracemalloc.start()
        with contextlib.redirect_stdout(io.StringIO()):
            large_result=ns['process'](large)
        record['peak_1080p']=tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert large_result.shape==large.shape
        assert record['peak_1080p'] < 128 * 1024 * 1024, record
        del large, large_result
    results.append(record)
print(json.dumps(results))
`], { input: JSON.stringify(jobs), stdio: "pipe" }).toString());
        for (const result of results) {
            t.diagnostic(`Python rank_filter 640x480 RGB size7 ${result.mode}: ${result.median_ms.toFixed(2)} ms (median of 3, after warm-up)`);
            if (result.peak_1080p)
                t.diagnostic(`Python rank_filter 1080p ${result.mode} peak traced allocation: ${(result.peak_1080p / 1048576).toFixed(2)} MiB`);
        }
    }
    finally { src.delete(); }
});
test("ip5d-PF. real Python stats/correlation Pass/Fail OK, NG and missing values", PY_SKIP, () => {
    const src = image(47, 39, (x, y) => [x * 3, y * 4, (x + y) * 2], 3), flat = image(47, 39, () => [90, 90, 90], 3);
    try {
        const jobs = [];
        for (const metric of ["mean", "std", "entropy", "corr2"])
            for (const lo of [0, 1000]) {
                const id = metric === "corr2" ? "corr2" : "image_stats", measure = step(id);
                if (id === "corr2")
                    measure.src2 = { pipeline: [] };
                jobs.push({ id, src, ...(id === "corr2" ? { b: src } : {}), pipe: [measure, step("pass_fail", { metric, cond: "ge", lo })] });
            }
        const measure = step("corr2");
        measure.src2 = { pipeline: [] };
        jobs.push({ id: "corr2", src: flat, b: flat, pipe: [measure, step("pass_fail", { metric: "corr2", cond: "ge", lo: 0 })] });
        const connected = step("corr2");
        connected.src2 = { pipeline: [] };
        jobs.push({ id: "corr2", src, b: src, noBLast: true, pipe: [connected, step("corr2"), step("pass_fail", { metric: "corr2", cond: "ge", lo: 0.9 })] });
        const results = pythonCases(jobs);
        for (let i = 0; i < jobs.length; i++) {
            const job = jobs[i], ctx = { src2: job.noBLast ? undefined : job.b }, out = run(job.id, job.src, {}, ctx), pf = job.pipe.at(-1).params;
            const result = passFailVerdict({ metric: pf.pass_fail__metric, cond: "ge", lo: pf.pass_fail__lo }, { op: job.id, values: ctx.values });
            out.delete();
            assert.match(results[i].stdout, new RegExp(`RESULT: ${result.result}`));
            assert.ok(!/WARNING|skipped/.test(results[i].stdout));
        }
    }
    finally {
        src.delete();
        flat.delete();
    }
});

import assert from 'node:assert/strict';
import { test } from 'node:test';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import cv from '@techstark/opencv-js';
import { OPS, FILTERS, needsOf, defaultParams, LUCY_WORK_BUDGET } from '../src/renderer/filters/registry.js';
import { buildPythonCode, PF_VALUE_OPS } from '../src/renderer/codegen.js';
import { OP_TO_FILTER, TOOL_SECTIONS } from '../src/renderer/ui/palette-panel.js';
import { OP_ICON, OP_PRIMARY, OPTION_GROUPS, GROUP_ICON } from '../src/renderer/filters/ui-meta.js';
import * as kernels5G from '../src/renderer/filters/batch5g.js';
import { pixelToStrokeQ, rasterizeStrokes } from '../src/renderer/filters/stroke-mask.js';
import { assertWrapperBalance } from './_leak-helpers.mjs';

while (typeof cv.Mat !== 'function') await new Promise((resolve) => setTimeout(resolve, 20));
globalThis.self = globalThis;
if (typeof globalThis.ImageData === 'undefined') globalThis.ImageData = class { constructor(data, width, height) { this.data = data; this.width = width; this.height = height; } };
const { runPipeline } = await import('../src/renderer/cv/cv-worker.js');
let python = false, parity = false;
if (process.env.IP5G2_NO_PY !== '1') {
  try { execFileSync('python', ['-c', 'pass'], { stdio: 'pipe' }); python = true; } catch {}
  try { execFileSync('python', ['-c', 'import cv2,numpy'], { stdio: 'pipe', env: { ...process.env, PYTHONIOENCODING: 'utf8' } }); parity = python; } catch {}
}

const hash = (value) => crypto.createHash('sha256').update(value).digest('hex');
const defaults = (id) => Object.fromEntries(OPS[id].params.map((p) => [p.key, p.default]));

test('ip5g2-G: clean-tree host variants preserve pixels, info, values and Python bytes', () => {
  const hosts = ['color_tools', 'frequency', 'image_registration', 'image_arithmetic', 'segment_edge'];
  const addedOps = new Set(['phase_register', 'scribble_segment', 'isodata_segment', 'fir_design']);
  const result = {};
  let operationVariants = 0;
  const imageKinds = [['rgb', 37, 29], ['gray', 37, 29], ['rgba', 37, 29], ['rgb', 1, 1], ['empty', 0, 0]];
  for (const [kind, width, height] of imageKinds) {
    const data = Uint8Array.from({ length: width * height * 3 }, (_, i) => (Math.imul(i + 11, 73) ^ (i >> 4) * 17) & 255);
    const color = width ? cv.matFromArray(height, width, cv.CV_8UC3, data) : new cv.Mat();
    const src = new cv.Mat();
    if (kind === 'gray') cv.cvtColor(color, src, cv.COLOR_RGB2GRAY);
    else if (kind === 'rgba') cv.cvtColor(color, src, cv.COLOR_RGB2RGBA);
    else color.copyTo(src);
    try {
      const jobs = [];
      for (const host of hosts) {
        const mode = FILTERS[host].params.find((param) => param.key === 'mode');
        for (const id of mode.options.map((option) => option.value)) {
          if (addedOps.has(id) || !OPS[id]) continue;
          const params = (OPS[id].params || []).filter((param) => !(id === 'inpaint' && ['patch', 'search'].includes(param.key)));
          const extras = [{}, ...params.filter((param) => param.type === 'select').flatMap((param) => param.options
            .filter((option) => option.value !== param.default)
            .filter((option) => !(id === 'inpaint' && param.key === 'method' && option.value === 'exemplar'))
            .filter((option) => !(id === 'pass_fail' && param.key === 'metric' && ['contour_area', 'qt_blocks'].includes(option.value)))
            .map((option) => ({ [param.key]: option.value })))];
          for (const extra of extras) jobs.push({ host, id, params, extra });
        }
      }
      if (kind === 'rgb' && width === 37) operationVariants = jobs.length;
      for (const { host, id, params, extra } of jobs) {
        const local = Object.fromEntries(params.map((param) => [param.key, param.default]));
        Object.assign(local, extra);
        const filterId = OP_TO_FILTER[id] || id;
        const stepParams = filterId === id
          ? local
          : { mode: id, ...Object.fromEntries(Object.entries(local).map(([key, value]) => [`${id}__${key}`, value])) };
        const stepForPython = { id: filterId, params: stepParams };
        const key = JSON.stringify([kind, width, height, host, id, extra]);
        const ctx = { src2: color };
        let converted = null, output = null;
        try {
          if (width && needsOf(FILTERS[host], stepParams) === 'gray' && src.channels() > 1) {
            converted = new cv.Mat();
            cv.cvtColor(src, converted, src.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY);
          }
          let seed = 12345;
          const random = Math.random;
          try {
            Math.random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
            output = FILTERS[host].apply(cv, converted || src, stepParams, ctx);
          } finally { Math.random = random; }
          result[key] = { image: hash(Buffer.from(output.data)), rows: output.rows, cols: output.cols, ch: output.channels(), info: ctx.info, values: ctx.values, python: hash(buildPythonCode([stepForPython])) };
        } catch (error) {
          result[key] = { error: typeof error === 'number' ? 'OpenCV native exception' : String(error), python: hash(buildPythonCode([stepForPython])) };
        } finally { if (output) output.delete(); if (converted) converted.delete(); }
      }
    } finally { src.delete(); color.delete(); }
  }
  assert.equal(operationVariants, 160);
  assert.equal(Object.keys(result).length, 800);
  assert.equal(hash(JSON.stringify(result)), '85c5e9418799da836f86833b6de3f2401e5b0da9b2d9e78a66c1aa0c597949db');
});

test('ip5g2-9-kernels: phase window, cross power and wrap threshold follow the analytic rules', () => {
  assert.deepEqual(Array.from(kernels5G.phaseWindow5G(1)), [1]);
  const win = kernels5G.phaseWindow5G(5);
  for (const [i, expected] of [0, 0.5, 1, 0.5, 0].entries()) assert.ok(Math.abs(win[i] - expected) <= 1e-12);
  const cross = new Float64Array(2);
  kernels5G.phaseCrossPower5G(new Float64Array([1, 0]), new Float64Array([0, 1]), cross);
  assert.deepEqual(Array.from(cross), [0, -1]);
  const inverse = new Float64Array(64 * 48 * 2);
  inverse[2 * (4 * 64 + 7)] = 1;
  const peak = kernels5G.phasePeak5G(inverse, 64, 48, false);
  assert.deepEqual([peak.dx, peak.dy, peak.cx, peak.cy], [7, 4, -7, -4]);
  const half = new Float64Array(64 * 1 * 2);
  half[2 * 32] = 1;
  assert.equal(kernels5G.phasePeak5G(half, 64, 1, false).cx, -32);
});

test('ip5g2-9-generated-python-booleans: phase options emit valid Python booleans', () => {
  const generated = [];
  for (const window of [true, false]) for (const subpixel of [true, false]) {
    const job = buildStep('phase_register', { window, subpixel });
    job.src2 = { pipeline: [] };
    generated.push(buildPythonCode([job]));
  }
  for (const source of generated) {
    const code = source.split(/\r?\n/)
      .filter((line) => !line.trimStart().startsWith('#'))
      .map((line) => line.replace(/'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"/g, ''))
      .join('\n');
    assert.doesNotMatch(code, /\b(?:true|false|null|undefined)\b/);
  }
});

test('ip5g2-9-gray-aligned-python: aligned output indexes gray and color arrays safely', () => {
  const job = buildStep('phase_register', { view: 'aligned' });
  job.src2 = { pipeline: [] };
  const code = buildPythonCode([job]);
  assert.match(code, /ph5g_out = ph5g_sample\(ph5g_raw, ph5g_sx, ph5g_sy\)/);
  assert.match(code, /ph5g_out\[:, :, ph5g_c\] = ph5g_sample\(ph5g_raw, ph5g_sx, ph5g_sy, ph5g_c\)/);
  assert.match(code, /img = ph5g_out\n/);
  assert.doesNotMatch(code, /ph5g_out\[:, :, 0\]/);
});

test('ip5g2-9-vectorized-sampling-python: aligned, overlay and diff sample whole gray and color grids', () => {
  for (const view of ['aligned', 'overlay', 'diff']) {
    const job = buildStep('phase_register', { view });
    job.src2 = { pipeline: [] };
    const code = buildPythonCode([job]);
    assert.match(code, /ph5g_xs, ph5g_ys = np\.meshgrid\(np\.arange\(ph5g_w, dtype=np\.float64\), np\.arange\(ph5g_h, dtype=np\.float64\), indexing='xy'\)/);
    assert.match(code, /np\.floor\(ph5g_x \+ 0\.5\)/);
    assert.match(code, /ph5g_a\.ndim == 2/);
    assert.match(code, /ph5g_value = ph5g_top \+ \(ph5g_bottom - ph5g_top\) \* ph5g_fy/);
    assert.match(code, /np\.clip\(np\.floor\(ph5g_value \+ 0\.5\), 0, 255\)/);
    assert.doesNotMatch(code, /for ph5g_y in range\(ph5g_h\)/);
  }
});

test('ip5g2-9-shift-snap-threshold: tiny phase noise is zeroed and near-integer shifts snap', () => {
  const cross = new Float64Array(6);
  kernels5G.phaseCrossPower5G(new Float64Array([1, 0, 1e-12, 0, 0, 0]), new Float64Array([1, 0, 1, 0, 0, 0]), cross);
  assert.deepEqual(Array.from(cross), [1, 0, 0, 0, 0, 0]);
  const zero = new Float64Array(4);
  kernels5G.phaseCrossPower5G(new Float64Array(4), new Float64Array(4), zero);
  assert.deepEqual(Array.from(zero), [0, 0, 0, 0]);
  const inverse = new Float64Array(16 * 2);
  inverse[2 * 5] = 10;
  inverse[2 * 4] = 9.4999999999;
  inverse[2 * 6] = 9.5000000001;
  const peak = kernels5G.phasePeak5G(inverse, 16, 1, true);
  assert.equal(peak.cx, -5);
  assert.equal(peak.cy, 0);
});

test('ip5g2-13-exemplar-python-gradient-and-gray-ssd: Python keeps the best gradient and sums gray channels', () => {
  const job = buildStep('inpaint', { method: 'exemplar', patch: '5', search: 60 });
  job.src2 = { pipeline: [] };
  const code = buildPythonCode([job]);
  assert.match(code, /ex5g_tgx = int\(ex5g_igr\[ex5g_qy0, ex5g_qx0 \+ 1\] - ex5g_igr\[ex5g_qy0, ex5g_qx0 - 1\]\)/);
  assert.match(code, /if ex5g_g2 > ex5g_gbest: ex5g_gbest, ex5g_gx, ex5g_gy = ex5g_g2, ex5g_tgx, ex5g_tgy/);
  assert.match(code, /ex5g_ssd \+= \(ex5g_diff \* ex5g_diff\)\.sum\(axis=1\)/);
  assert.doesNotMatch(code, /if ex5g_gray: ex5g_ssd/);
});

test('ip5g2-10-scribble-python-info-numbers: missing-seed messages include the app count', () => {
  const code = buildPythonCode([buildStep('scribble_segment')]);
  assert.match(code, /Scribble: missing object strokes \(need at least 1\)/);
  assert.match(code, /Scribble: missing background strokes \(need at least 1\)/);
});

test('ip5g2-9-register: phase correlation reports the correction to apply to A', () => {
  const w = 64, h = 48;
  let seed = 4;
  const base = Uint8Array.from({ length: w * h }, () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) >>> 24));
  const shifted = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) shifted[y * w + x] = base[((y + 4 + h) % h) * w + ((x - 7 + w) % w)];
  const a = cv.matFromArray(h, w, cv.CV_8UC1, shifted), b = cv.matFromArray(h, w, cv.CV_8UC1, base);
  let out = null;
  try {
    const ctx = { src2: b };
    out = OPS.phase_register.apply(cv, a, { window: false, subpixel: false, view: 'aligned' }, ctx);
    assert.deepEqual([ctx.values.fields[0].value, ctx.values.fields[1].value], [-7, 4]);
    assert.match(ctx.info, /เลื่อน A \(-7\.00, 4\.00\) px/);
    assert.equal(out.channels(), 1);
  } finally { out?.delete(); a.delete(); b.delete(); }
});

test('ip5g2-9-identity-views: equal inputs report a zero shift and center the surface peak', () => {
  const w = 32, h = 24;
  let state = 0x905;
  const a = image(w, h, 1, () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) >>> 24));
  let aligned = null, surface = null;
  try {
    const alignedCtx = { src2: a };
    aligned = apply('phase_register', a, { window: false, subpixel: false, view: 'aligned' }, alignedCtx);
    assert.deepEqual(Array.from(aligned.data), Array.from(a.data));
    assert.deepEqual(alignedCtx.values.fields.slice(0, 2).map((field) => field.value), [0, 0]);
    assert.ok(alignedCtx.values.fields[2].value > 0.99);
    const surfaceCtx = { src2: a };
    surface = apply('phase_register', a, { window: false, subpixel: false, view: 'surface' }, surfaceCtx);
    assert.equal(surface.channels(), 1);
    assert.equal(surface.data[Math.floor(surface.rows / 2) * surface.cols + Math.floor(surface.cols / 2)], 255);
  } finally { aligned?.delete(); surface?.delete(); a.delete(); }
});

test('ip5g2-9-peak-sign-wrap-and-subpixel: first peak, strict half-wrap and parabolic direction are pinned', () => {
  const inv = new Float64Array(64 * 3 * 2);
  inv[2 * (1 * 64 + 32)] = 10;
  const half = kernels5G.phasePeak5G(inv, 64, 3, false);
  assert.deepEqual([half.px, half.py, half.cx, half.cy], [32, 1, -32, -1]);
  const tied = new Float64Array(8 * 2);
  tied[2] = 1;
  tied[10] = 1;
  assert.equal(kernels5G.phasePeak5G(tied, 8, 1, false).px, 1);
  const parabola = new Float64Array(16 * 2);
  parabola[2 * 5] = 7;
  parabola[2 * 6] = 10;
  parabola[2 * 7] = 8;
  const sub = kernels5G.phasePeak5G(parabola, 16, 1, true);
  assert.equal(sub.dx, 6.1);
  assert.equal(sub.cx, -6.1);
});

test('ip5g2-9-failure-layouts: missing B, empty A, small and flat inputs return the view shape and values', () => {
  for (const channels of [1, 3, 4]) for (const view of ['aligned', 'overlay', 'diff', 'surface']) {
    const a = image(7, 9, channels, (x, y) => channels === 1 ? (x * 29 + y * 17) & 255 : [x * 29 & 255, y * 17 & 255, (x * 7 + y * 13) & 255, 71]);
    const b = image(7, 9, 1, (x, y) => (x * 11 + y * 23) & 255);
    let out = null;
    try {
      const ctx = { src2: b };
      out = apply('phase_register', a, { view }, ctx);
      const expected = view === 'aligned' ? (channels === 4 ? 3 : channels) : view === 'overlay' ? 3 : 1;
      assert.equal(out.channels(), expected, `${channels}ch ${view}`);
      assert.deepEqual(ctx.values.fields.map((field) => field.value), [0, 0, 0]);
    } finally { out?.delete(); a.delete(); b.delete(); }
  }
  const a = image(12, 10, 1, (x, y) => (x * 11 + y * 19) & 255);
  const empty = new cv.Mat();
  try {
    const ctx = {};
    const out = apply('phase_register', a, {}, ctx);
    assert.equal(out.channels(), 1);
    assert.match(ctx.info, /missing|B/i);
    assert.ok(ctx.values.fields.every((field) => field.value === null));
    out.delete();
    const emptyB = new cv.Mat(), emptyBCtx = { src2: emptyB };
    const emptyBOut = apply('phase_register', a, {}, emptyBCtx);
    assert.deepEqual(emptyBCtx.values.fields.map((field) => field.value), [null, null, null]);
    assert.equal(emptyBOut.rows, a.rows); emptyBOut.delete(); emptyB.delete();
    const emptyCtx = { src2: empty };
    const emptyA = new cv.Mat();
    const emptyOut = apply('phase_register', emptyA, {}, emptyCtx);
    assert.equal(emptyOut.rows, 0);
    assert.equal(emptyOut.cols, 0);
    assert.ok(emptyCtx.values.fields.every((field) => field.value === null));
    emptyOut.delete();
    emptyA.delete();
    assert.ok(emptyCtx.info);
    const resizedB = image(16, 12, 1, (x, y) => x * 13 + y * 17), resizeCtx = { src2: resizedB };
    const resizeOut = apply('phase_register', a, { window: false, subpixel: false }, resizeCtx);
    assert.equal(resizeOut.rows, a.rows); assert.equal(resizeOut.cols, a.cols); assert.ok(resizeCtx.info);
    resizeOut.delete(); resizedB.delete();
  } finally { a.delete(); empty.delete(); }
  const flatA = image(12, 10, 3, () => [73, 73, 73]), flatB = image(12, 10, 1, (x, y) => (x * 17 + y * 31) & 255);
  try {
    for (const view of ['aligned', 'overlay', 'diff', 'surface']) {
      const ctx = { src2: flatB }, out = apply('phase_register', flatA, { view }, ctx);
      try {
        assert.equal(ctx.values.fields[2].value, 0);
        assert.equal(out.channels(), view === 'aligned' ? 3 : view === 'overlay' ? 3 : 1);
        assert.ok(Array.from(out.data).every((value) => value === 73), `${view} failure image is A in its view layout`);
        assert.match(ctx.info, /flat|เรียบ/);
      } finally { out.delete(); }
    }
  } finally { flatA.delete(); flatB.delete(); }
  const texturedA = image(12, 10, 3, (x, y) => [x * 11 & 255, y * 17 & 255, (x * 7 + y * 13) & 255]);
  const constantB = image(12, 10, 1, () => 101);
  try {
    for (const view of ['aligned', 'overlay', 'diff', 'surface']) {
      const ctx = { src2: constantB }, out = apply('phase_register', texturedA, { view }, ctx);
      try {
        assert.equal(ctx.values.fields[2].value, 0);
        assert.equal(out.channels(), view === 'aligned' || view === 'overlay' ? 3 : 1);
        assert.ok(ctx.info.includes('เรียบ'));
      } finally { out.delete(); }
    }
  } finally { texturedA.delete(); constantB.delete(); }
});

test('ip5g2-9-analytic-shifts: half-wrap, subpixel blob, cutoff translation and resized B are pinned', () => {
  let state = 0x9a17;
  const randomBytes = (width, height) => Uint8Array.from({ length: width * height }, () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) >>> 24));
  const shift = (data, width, height, dx, dy, wrap) => {
    const out = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      let sx = x - dx, sy = y - dy;
      if (wrap) { sx = (sx + width) % width; sy = (sy + height) % height; }
      out[y * width + x] = sx >= 0 && sx < width && sy >= 0 && sy < height ? data[sy * width + sx] : 0;
    }
    return out;
  };
  const w = 64, h = 48, baseData = randomBytes(w, h), base = cv.matFromArray(h, w, cv.CV_8UC1, baseData);
  const halfTurn = cv.matFromArray(h, w, cv.CV_8UC1, shift(baseData, w, h, 32, 0, true));
  try {
    const ctx = { src2: base }, out = apply('phase_register', halfTurn, { window: false, subpixel: false }, ctx);
    try { assert.deepEqual(ctx.values.fields.slice(0, 2).map((field) => field.value), [-32, 0]); }
    finally { out.delete(); }
  } finally { base.delete(); halfTurn.delete(); }

  const blob = (cx, cy) => image(w, h, 1, (x, y) => Math.floor(128 + 100 * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 36)) + 0.5));
  const blobB = blob(30, 24), blobA = blob(33.4, 24);
  try {
    const ctx = { src2: blobB }, out = apply('phase_register', blobA, { window: false, subpixel: true }, ctx);
    try { assert.ok(ctx.values.fields[0].value < -3.05 && ctx.values.fields[0].value > -3.9, `expected geometry-based correction near -3.4, got ${ctx.values.fields[0].value}`); }
    finally { out.delete(); }
  } finally { blobB.delete(); blobA.delete(); }

  const cutoffData = randomBytes(w, h), cutoffB = cv.matFromArray(h, w, cv.CV_8UC1, cutoffData);
  const cutoffA = cv.matFromArray(h, w, cv.CV_8UC1, shift(cutoffData, w, h, 7, -4, false));
  try {
    const ctx = { src2: cutoffB }, out = apply('phase_register', cutoffA, { window: true, subpixel: true }, ctx);
    try { assert.deepEqual(ctx.values.fields.slice(0, 2).map((field) => Math.round(field.value)), [-7, 4]); }
    finally { out.delete(); }
  } finally { cutoffB.delete(); cutoffA.delete(); }

  const smallW = 32, smallH = 24, smallData = Uint8Array.from({ length: smallW * smallH }, (_, i) => ((Math.imul(i + 13, 73) ^ Math.imul(i + 7, 29)) & 255));
  const smallB = cv.matFromArray(smallH, smallW, cv.CV_8UC1, smallData), fullB = new cv.Mat();
  cv.resize(smallB, fullB, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR);
  const fullData = Uint8Array.from(fullB.data), shiftedA = cv.matFromArray(h, w, cv.CV_8UC1, shift(fullData, w, h, 3, -2, true));
  try {
    const ctx = { src2: smallB }, out = apply('phase_register', shiftedA, { window: false, subpixel: false }, ctx);
    try {
      assert.deepEqual(ctx.values.fields.slice(0, 2).map((field) => field.value), [-3, 2]);
      assert.match(ctx.info, /B/);
      assert.match(ctx.info, /ขนาดไม่ตรงกัน/);
    } finally { out.delete(); }
  } finally { smallB.delete(); fullB.delete(); shiftedA.delete(); }
});

test('ip5g2-10-11-12-kernels: scribble ties go to background and FIR kernels stay symmetric', () => {
  const w = 40, h = 20, pixels = new Uint8Array(w * h), labels = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) pixels[y * w + x] = x < 20 ? 30 : 220;
  labels[10 * w + 30] = 1; labels[10 * w + 10] = 2;
  const seg = kernels5G.scribbleSegment5G(pixels, w, h, 1, labels, 1, 'chessboard', 'mask', 100);
  assert.equal(seg.area, 400);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) assert.equal(seg.data[y * w + x], x >= 20 ? 255 : 0);
  const iso = kernels5G.isodata5G(Uint8Array.from(Array.from({ length: 16 }, (_, i) => i < 8 ? 0 : 255)), 4, 4, 1, 2, 10, 0, 128, 0, 'mean', 10);
  assert.equal(iso.clusters, 2);
  assert.deepEqual(Array.from(iso.data), Array.from({ length: 16 }, (_, i) => i < 8 ? 0 : 255));
  const d = kernels5G.firDesign5G(5, 'sampling', 'lowpass', 0.5, 0.6, 'rect');
  assert.ok(Math.abs(d.sum - 1) <= 1e-9);
  assert.equal(kernels5G.reflect101_5G(-1, 4), 1);
  const delta = kernels5G.mcclellanSynth5G([1, 0, 0], 5);
  assert.equal(delta[2 * 5 + 2], 1);
  const p2 = kernels5G.mcclellanSynth5G([0, 0, 1], 5);
  assert.equal(p2[2 * 5 + 2], 0.125);
});

test('ip5g2-10-scribble-hand-cases: ties, edge-free distances, overlay rounding and seed colours match the spec', () => {
  const w = 11, h = 3, pix = new Uint8Array(w * h).fill(100), labels = new Uint8Array(w * h);
  labels[1 * w + 2] = 1;
  labels[1 * w + 8] = 2;
  const tied = kernels5G.scribbleSegment5G(pix, w, h, 1, labels, 0, 'chessboard', 'mask', 100);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) assert.equal(tied.data[y * w + x], x <= 4 ? 255 : 0);
  const overlay = kernels5G.scribbleSegment5G(pix, w, h, 1, labels, 0, 'chessboard', 'overlay', 100);
  assert.deepEqual(Array.from(overlay.data.slice(3 * (1 * w + 2), 3 * (1 * w + 2) + 3)), [230, 30, 30]);
  assert.deepEqual(Array.from(overlay.data.slice(3 * (1 * w + 8), 3 * (1 * w + 8) + 3)), [30, 80, 240]);
  const oneObject = new Uint8Array(9); oneObject[4] = 1;
  const oneBackground = new Uint8Array(9); oneBackground[0] = 2;
  const simpleLabels = Uint8Array.from(oneObject, (v, i) => v || oneBackground[i]);
  const flat = kernels5G.scribbleSegment5G(new Uint8Array(9).fill(100), 3, 3, 1, simpleLabels, 0, 'chessboard', 'overlay', 100);
  assert.deepEqual(Array.from(flat.data.slice(3 * 8, 3 * 8 + 3)), [50, 178, 50]);
  const quasi = kernels5G.scribbleSegment5G(new Uint8Array(4).fill(1), 2, 2, 1, Uint8Array.from([1, 0, 0, 2]), 0, 'quasi', 'distance', 100);
  assert.equal(quasi.data[0], 0);
  assert.deepEqual(Array.from(quasi.data.slice(3 * 3, 3 * 3 + 3)), [255, 255, 255]);
  const diagonal = kernels5G.scribbleSegment5G(new Uint8Array(9).fill(1), 3, 3, 1, Uint8Array.from([1, 0, 0, 0, 0, 0, 0, 0, 2]), 0, 'quasi', 'distance', 100);
  assert.deepEqual(Array.from(diagonal.data.slice(3 * 4, 3 * 4 + 3)), [128, 128, 128], 'one sqrt(2) diagonal over a two-sqrt(2) maximum is half-scale');
  assert.deepEqual(Array.from(diagonal.data.slice(3 * 8, 3 * 8 + 3)), [255, 255, 255], 'distance view does not paint a seed cross');
});

test('ip5g2-10-scribble-maze: repeated geodesic sweeps match Dijkstra and distance reaches 255', () => {
  const width = 11, height = 9, pix = new Uint8Array(width * height), labels = new Uint8Array(width * height);
  for (const y of [1, 3, 5, 7]) for (let x = 1; x <= 9; x++) pix[y * width + x] = 255;
  pix[2 * width + 9] = pix[4 * width + 1] = pix[6 * width + 9] = 255;
  labels[width + 1] = 1; labels[7 * width + 9] = 2;
  const cost = (from, to) => 1 + Math.abs(pix[from] - pix[to]);
  const reference = (seed) => {
    const d = new Float64Array(width * height), seen = new Uint8Array(width * height); d.fill(Infinity); d[seed] = 0;
    for (let count = 0; count < d.length; count++) {
      let at = -1;
      for (let i = 0; i < d.length; i++) if (!seen[i] && Number.isFinite(d[i]) && (at < 0 || d[i] < d[at])) at = i;
      if (at < 0) break;
      seen[at] = 1; const x = at % width, y = Math.floor(at / width);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const nx = x + dx, ny = y + dy; if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const next = ny * width + nx, candidate = d[at] + cost(at, next);
        if (candidate < d[next]) d[next] = candidate;
      }
    }
    return d;
  };
  const redRef = reference(width + 1), blueRef = reference(7 * width + 9);
  const oneSet = kernels5G.scribbleSegment5G(pix, width, height, 1, labels, 1, 'chessboard', 'distance', 1);
  assert.equal(oneSet.converged, false);
  const settled = kernels5G.scribbleSegment5G(pix, width, height, 1, labels, 1, 'chessboard', 'distance', 100);
  const mask = kernels5G.scribbleSegment5G(pix, width, height, 1, labels, 1, 'chessboard', 'mask', 100);
  assert.equal(settled.converged, true);
  assert.equal(settled.data.filter((value, i) => i % 3 === 0 && value === 255).length > 0, true);
  const maxD = Math.max(...redRef);
  for (let i = 0; i < redRef.length; i++) {
    const expected = Math.floor(redRef[i] / maxD * 255 + 0.5);
    assert.deepEqual(Array.from(settled.data.slice(i * 3, i * 3 + 3)), [expected, expected, expected], `Dijkstra distance ${i}`);
    assert.equal(mask.data[i] === 255, redRef[i] < blueRef[i], `Dijkstra classification ${i}`);
  }
  assert.ok(Math.max(...Array.from({ length: redRef.length }, (_, i) => settled.data[i * 3])) === 255);
});

test('ip5g2-10-seed-layouts-and-aspect: missing, one-sided and aspect-scaled strokes keep view layouts and values', () => {
  const src = image(40, 20, 4, () => [100, 100, 100, 27]);
  const emptyCases = [null, {}, scribbles(40, 20, []), scribbles(40, 20, [[1, 5, 10]]), scribbles(40, 20, [[2, 35, 10]])];
  try {
    for (const strokes of emptyCases) for (const view of ['overlay', 'mask', 'distance']) {
      const ctx = {}, out = apply('scribble_segment', src, { view, strokes }, ctx);
      assert.equal(out.channels(), view === 'mask' ? 1 : 3, `${view}/${String(strokes)}`);
      assert.equal(ctx.values.fields[0].value, null);
      assert.ok(ctx.info);
      if (strokes?.s?.length === 1 && [1, 2].includes(strokes.s[0][0])) assert.ok(ctx.info.includes('1'));
      if (strokes?.s?.length === 1 && strokes.s[0][0] === 1) assert.match(ctx.info, /น้ำเงิน/);
      if (strokes?.s?.length === 1 && strokes.s[0][0] === 2) assert.match(ctx.info, /แดง/);
      if (strokes == null || Object.keys(strokes || {}).length === 0 || strokes?.s?.length === 0) assert.match(ctx.info, /ยังไม่มีเส้นขีด/);
      out.delete();
    }
  } finally { src.delete(); }
  for (const channels of [1, 3, 4]) {
    const source = image(8, 6, channels, () => channels === 1 ? 90 : [90, 120, 150, 33]);
    try {
      for (const view of ['overlay', 'mask', 'distance']) {
        const ctx = {}, out = apply('scribble_segment', source, { view, strokes: null }, ctx);
        try {
          assert.equal(out.channels(), view === 'mask' ? 1 : 3, `${channels}ch/${view} no-seed layout`);
          assert.equal(ctx.values.fields[0].value, null);
        } finally { out.delete(); }
      }
    } finally { source.delete(); }
  }
  const run = (w, h, savedW = w, savedH = h) => {
    const src = image(w, h, 1, (x) => x < w / 2 ? 30 : 220);
    const q = (x, n) => pixelToStrokeQ(x, n);
    const strokes = { v: 1, w: savedW, h: savedH, s: [[1, 12, q(savedW * 3 / 4, savedW), q(savedH / 2, savedH)], [2, 12, q(savedW / 4, savedW), q(savedH / 2, savedH)]] };
    const ctx = {}, out = apply('scribble_segment', src, { view: 'mask', edge: 1, strokes }, ctx);
    const result = { area: ctx.values.fields[0].value, info: ctx.info, data: Uint8Array.from(out.data) };
    out.delete(); src.delete(); return result;
  };
  const small = run(40, 20), large = run(80, 40);
  assert.equal(small.area, 400);
  assert.equal(large.area, 1600);
  assert.notEqual(run(20, 20, 40, 20).info, run(20, 20).info);
});

test('ip5g2-10-scribble-budget: 640x480 noise reports the capped three sweep sets per colour', () => {
  let state = 0x71f2a5;
  const src = image(640, 480, 3, () => {
    const pixel = [];
    for (let c = 0; c < 3; c++) { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; pixel.push(state >>> 24); }
    return pixel;
  });
  const ctx = {}, strokes = { v: 1, w: 640, h: 480, s: [[1, 10, 900, 900], [2, 10, 9100, 9100]] };
  try {
    const out = apply('scribble_segment', src, { view: 'overlay', edge: 1, metric: 'chessboard', strokes }, ctx);
    try {
      assert.equal(out.channels(), 3);
      assert.match(ctx.info, /6 รอบกวาด/);
      assert.match(ctx.info, /สูงสุด 3 ชุดต่อสี/);
    } finally { out.delete(); }
  } finally { src.delete(); }
});

test('ip5g2-11-isodata-hand-cases: split, cut, merge, stable count and output palette are pinned', () => {
  const block = (colors) => {
    const out = new Uint8Array(colors.length * 3);
    colors.forEach((color, i) => out.set(color, 3 * i));
    return out;
  };
  const tri = block([
    ...Array.from({ length: 100 }, () => [0, 0, 0]),
    ...Array.from({ length: 100 }, () => [0, 0, 255]),
    ...Array.from({ length: 100 }, () => [255, 0, 0]),
  ]);
  const three = kernels5G.isodata5G(tri, 20, 15, 3, 3, 10, 1, 30, 30, 'mean', 10);
  assert.equal(three.clusters, 3);
  assert.equal(three.splits + three.merges + three.cuts, 0);
  assert.deepEqual(three.data, tri);
  const split = kernels5G.isodata5G(tri, 20, 15, 3, 2, 10, 1, 30, 30, 'mean', 10);
  assert.equal(split.clusters, 3);
  assert.equal(split.splits, 1);
  assert.equal(split.used, 3);
  const splitLabels = kernels5G.isodata5G(tri, 20, 15, 3, 2, 10, 1, 30, 30, 'labels', 10);
  assert.deepEqual(Array.from(splitLabels.data.slice(0, 3)), kernels5G.ISODATA_PALETTE_5G[0], 'the original centre keeps the lower split direction');
  assert.deepEqual(Array.from(splitLabels.data.slice(100 * 3, 103 * 3)), Array.from({ length: 3 }, () => kernels5G.ISODATA_PALETTE_5G[2]).flat(), 'blue joins the newly appended high-side centre');
  const gray = kernels5G.isodata5G(new Uint8Array(4).fill(77), 2, 2, 1, 4, 4, 1, 30, 30, 'labels', 4);
  assert.equal(gray.clusters, 1);
  assert.equal(gray.cuts, 3);
  assert.equal(gray.channels, 3);
  assert.deepEqual(Array.from(gray.data.slice(0, 3)), kernels5G.ISODATA_PALETTE_5G[0]);
  const merge = kernels5G.isodata5G(Uint8Array.from([0, 50, 100, 150]), 4, 1, 1, 4, 5, 0, 128, 255, 'labels', 5);
  assert.equal(merge.merges, 2);
  assert.equal(merge.clusters, 2);
  const mergeAtThreshold = kernels5G.isodata5G(Uint8Array.from([0, 30]), 2, 1, 1, 2, 3, 0, 128, 30, 'mean', 3);
  assert.equal(mergeAtThreshold.clusters, 2, 'distance equal to mergeDist is not merged');
  const tie = kernels5G.isodata5G(Uint8Array.from([0, 127, 254]), 3, 1, 1, 2, 1, 0, 128, 0, 'labels', 1);
  assert.deepEqual(Array.from(tie.data.slice(3, 6)), kernels5G.ISODATA_PALETTE_5G[0], 'the exactly equidistant middle value goes to the lower centre index');
  assert.deepEqual(Array.from(tie.data.slice(6, 9)), kernels5G.ISODATA_PALETTE_5G[1]);
  const cutData = new Uint8Array(400);
  cutData.fill(0, 0, 200); cutData.fill(100, 200, 201); cutData.fill(200, 201);
  const cut = kernels5G.isodata5G(cutData, 20, 20, 1, 3, 5, 1, 128, 0, 'mean', 5);
  assert.equal(cut.cuts, 1);
  assert.equal(cut.clusters, 2);
  const paletteColors = Uint8Array.from({ length: 24 }, (_, i) => i * 10);
  const wrapped = kernels5G.isodata5G(paletteColors, 24, 1, 1, 12, 2, 0, 4, 0, 'labels', 2);
  assert.equal(wrapped.clusters, 24);
  assert.deepEqual(Array.from(wrapped.data.slice(3, 6)), kernels5G.ISODATA_PALETTE_5G[11]);
  assert.deepEqual(Array.from(wrapped.data.slice(23 * 3, 23 * 3 + 3)), kernels5G.ISODATA_PALETTE_5G[11]);
});

test('ip5g2-11-isodata-oracle: unique-colour kernel matches a naive per-pixel implementation on 40 seeded cases', () => {
  let state = 0x5eeda11;
  const random = (n) => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state % n; };
  for (let trial = 0; trial < 40; trial++) {
    const width = 1 + random(12), height = 1 + random(12), ch = random(2) ? 1 : 3;
    const pix = Uint8Array.from({ length: width * height * ch }, () => random(8) * 31);
    const k = 2 + random(5), iterations = 1 + random(7), minSizePct = random(30) / 2;
    const splitStd = random(128), mergeDist = random(90), view = random(2) ? 'mean' : 'labels';
    const actual = kernels5G.isodata5G(pix, width, height, ch, k, iterations, minSizePct, splitStd, mergeDist, view, iterations);
    const expected = isodataOracle(pix, width, height, ch, k, iterations, minSizePct, splitStd, mergeDist, view);
    assert.deepEqual(actual.data, expected.data, `trial ${trial} bytes`);
    assert.deepEqual([actual.clusters, actual.used, actual.splits, actual.merges, actual.cuts, actual.converged],
      [expected.clusters, expected.used, expected.splits, expected.merges, expected.cuts, expected.converged], `trial ${trial} stats`);
  }
});

test('ip5g2-12-fir-oracle: design coefficients match a separately written direct-summation reference', () => {
  const cos12 = (a) => Math.floor(Math.cos(a) * 1e12 + 0.5) / 1e12;
  const round12 = (v) => Math.floor(v * 1e12 + 0.5) / 1e12;
  function oracle(n, method, response, c1, c2, win) {
    const m = (n - 1) / 2, lowEdge = Math.min(c1, c2), highEdge = Math.max(c1, c2), rw = m + 1;
    const windowAt = (r) => r >= rw ? 0 : win === 'hamming' ? 0.54 + 0.46 * cos12(Math.PI * r / rw) : win === 'hann' ? 0.5 + 0.5 * cos12(Math.PI * r / rw) : 1;
    const h = new Float64Array(n * n);
    if (method === 'mcclellan') {
      const low = (fc, q) => round12((fc * (q === 0 ? 1 : Math.sin(Math.PI * fc * q) / (Math.PI * fc * q))) * windowAt(Math.abs(q)));
      const a = [];
      for (let q = 0; q <= m; q++) {
        const d = q === 0 ? 1 : 0;
        const v = response === 'lowpass' ? low(c1, q) : response === 'highpass' ? d - low(c1, q) : response === 'bandpass' ? low(highEdge, q) - low(lowEdge, q) : d - (low(highEdge, q) - low(lowEdge, q));
        a.push(q === 0 ? v : 2 * v);
      }
      const t = [0.125, 0.25, 0.125, 0.25, -0.5, 0.25, 0.125, 0.25, 0.125];
      let p2 = null, p = new Float64Array(n * n); p[m * n + m] = 1;
      for (let i = 0; i < h.length; i++) h[i] += a[0] * p[i];
      for (let q = 1; q <= m; q++) {
        const next = new Float64Array(n * n);
        if (q === 1) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) next[(m + y - 1) * n + m + x - 1] = t[y * 3 + x];
        else for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
          let acc = 0;
          for (let ty = 0; ty < 3; ty++) for (let tx = 0; tx < 3; tx++) {
            const yy = y - ty + 1, xx = x - tx + 1;
            if (yy >= 0 && yy < n && xx >= 0 && xx < n) acc += t[ty * 3 + tx] * p[yy * n + xx];
          }
          next[y * n + x] = 2 * acc - p2[y * n + x];
        }
        for (let i = 0; i < h.length; i++) h[i] += a[q] * next[i];
        p2 = p; p = next;
      }
      for (let i = 0; i < h.length; i++) h[i] = round12(h[i]);
      return h;
    }
    const hd = new Float64Array(n * n), ctab = Float64Array.from({ length: n }, (_, a) => cos12((2 * Math.PI * a) / n));
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const f1 = ((x - m) * 2) / n, f2 = ((y - m) * 2) / n, r = Math.sqrt(f1 * f1 + f2 * f2);
      const lo = r <= c1 ? 1 : 0, band = r >= lowEdge && r <= highEdge ? 1 : 0;
      hd[y * n + x] = response === 'lowpass' ? lo : response === 'highpass' ? 1 - lo : response === 'bandpass' ? band : 1 - band;
    }
    for (let dy = -m; dy <= m; dy++) for (let dx = -m; dx <= m; dx++) {
      let acc = 0;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const index = (((x - m) * dx + (y - m) * dy) % n + n) % n;
        acc += hd[y * n + x] * ctab[index];
      }
      h[(dy + m) * n + dx + m] = round12(method === 'window' ? (acc / (n * n)) * windowAt(Math.sqrt(dx * dx + dy * dy)) : acc / (n * n));
    }
    return h;
  }
  for (const n of [3, 5, 11, 21]) for (const method of ['window', 'sampling', 'mcclellan']) for (const response of ['lowpass', 'highpass', 'bandpass', 'bandstop']) for (const win of ['hamming', 'hann', 'rect']) {
    const actual = kernels5G.firDesign5G(n, method, response, 0.27, 0.63, win).h;
    assert.deepEqual(actual, oracle(n, method, response, 0.27, 0.63, win), `${n}/${method}/${response}/${win}`);
  }
  assert.ok(Math.abs(kernels5G.firDesign5G(11, 'sampling', 'lowpass', 0.3, 0.6, 'rect').sum - 1) <= 1e-9);
  assert.ok(Math.abs(kernels5G.firDesign5G(11, 'sampling', 'highpass', 0.3, 0.6, 'rect').sum) <= 1e-9);
});

test('ip5g2-12-fir-views-boundaries: symmetry, quantization, reflect101, DC behavior, dimensions and budget match', () => {
  for (const method of ['window', 'sampling', 'mcclellan']) for (const response of ['lowpass', 'highpass', 'bandpass', 'bandstop']) {
    const { N, M, h } = kernels5G.firDesign5G(11, method, response, 0.31, 0.62, 'hann');
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      assert.ok(Math.abs(h[y * N + x] * 1e12 - Math.round(h[y * N + x] * 1e12)) < 1e-3);
      assert.equal(h[y * N + x], h[(N - 1 - y) * N + N - 1 - x]);
      assert.ok(Math.abs(h[y * N + x] - h[x * N + y]) <= 1e-12);
    }
    assert.ok(Number.isInteger(M));
  }
  assert.equal(kernels5G.firKernelSize5G(640, 480, 3, 21, LUCY_WORK_BUDGET), 21);
  assert.equal(kernels5G.firKernelSize5G(1920, 1080, 3, 21, LUCY_WORK_BUDGET), 7);
  assert.deepEqual(Array.from({ length: 5 }, (_, i) => kernels5G.reflect101_5G(i - 2, 1)), [0, 0, 0, 0, 0]);
  assert.deepEqual(Array.from({ length: 9 }, (_, i) => kernels5G.reflect101_5G(i - 2, 4)), [2, 1, 0, 1, 2, 3, 2, 1, 0]);
  const narrow = kernels5G.firFilter5G(new Uint8Array(9).fill(80), 1, 9, kernels5G.firDesign5G(21, 'sampling', 'lowpass', 0.3, 0.6, 'rect').h, 21, false);
  assert.ok(narrow.every(Number.isFinite));
  const reflect = (at, n) => {
    if (n <= 1) return 0;
    const period = 2 * n - 2, q = ((at % period) + period) % period;
    return q < n ? q : period - q;
  };
  const edgeInput = Uint8Array.from([0, 100, 200, 250]), edgeDesign = kernels5G.firDesign5G(5, 'sampling', 'lowpass', 0.3, 0.6, 'rect');
  const edgeOutput = kernels5G.firFilter5G(edgeInput, 4, 1, edgeDesign.h, 5, false);
  let edgeExpected = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) edgeExpected += edgeDesign.h[(dy + 2) * 5 + dx + 2] * edgeInput[reflect(dx, 4)];
  edgeExpected = Math.max(0, Math.min(255, Math.floor(edgeExpected + 0.5)));
  let replicateExpected = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) replicateExpected += edgeDesign.h[(dy + 2) * 5 + dx + 2] * edgeInput[Math.max(0, Math.min(3, dx))];
  replicateExpected = Math.max(0, Math.min(255, Math.floor(replicateExpected + 0.5)));
  assert.equal(edgeOutput[0], edgeExpected);
  assert.notEqual(edgeExpected, replicateExpected, 'the fixture distinguishes reflect101 from replicate padding');
  const constant = new Uint8Array(16).fill(73), low = kernels5G.firDesign5G(5, 'sampling', 'lowpass', 0.3, 0.6, 'rect');
  assert.deepEqual(kernels5G.firFilter5G(constant, 4, 4, low.h, 5, false), constant);
  const high = kernels5G.firDesign5G(5, 'sampling', 'highpass', 0.3, 0.6, 'rect');
  assert.deepEqual(Array.from(kernels5G.firFilter5G(constant, 4, 4, high.h, 5, true)), Array(16).fill(128));
  const response = kernels5G.firResponseImage5G(kernels5G.firResponse5G(low.h, 5));
  const mag = kernels5G.firResponse5G(low.h, 5), scale = Math.max(1, ...mag);
  assert.equal(response[125 * 256 + 128], Math.floor(mag[32 * 64 + 32] / scale * 255 + 0.5));
  const directMagnitude = new Float64Array(64 * 64);
  for (let g2 = 0; g2 < 64; g2++) for (let g1 = 0; g1 < 64; g1++) {
    const f1 = (g1 - 32) / 32, f2 = (g2 - 32) / 32;
    let re = 0, im = 0;
    for (let y = 0; y < low.N; y++) for (let x = 0; x < low.N; x++) {
      const m1 = x - low.M, m2 = y - low.M, phase = Math.PI * (f1 * m1 + f2 * m2), hv = low.h[y * low.N + x];
      re += hv * Math.cos(phase); im -= hv * Math.sin(phase);
    }
    directMagnitude[g2 * 64 + g1] = Math.sqrt(re * re + im * im);
  }
  const directScale = Math.max(1, ...directMagnitude);
  const directImage = kernels5G.firResponseImage5G(directMagnitude);
  for (let i = 0; i < response.length; i++) assert.ok(Math.abs(response[i] - directImage[i]) <= 1, `response view pixel ${i}`);
  const kernel = kernels5G.firKernelImage5G(low.h, 5);
  assert.equal(kernel.width, 5 * Math.floor(256 / 5));
});

test('ip5g2-12-fir-empty-views: 0x0 returns a clone with the empty-image note for every view', () => {
  const src = new cv.Mat();
  try {
    for (const view of ['result', 'response', 'kernel']) {
      const ctx = {}, out = apply('fir_design', src, { view }, ctx);
      try {
        assert.deepEqual([out.rows, out.cols], [0, 0]);
        assert.equal(ctx.info, 'ภาพว่าง');
      } finally { out.delete(); }
    }
  } finally { src.delete(); }
});

test('ip5g2-13-kernel: exemplar fills only masked pixels and falls back without a full patch', () => {
  const w = 12, h = 12, source = new Uint8Array(w * h).fill(100), mask = new Uint8Array(w * h);
  for (let y = 5; y < 7; y++) for (let x = 5; x < 7; x++) mask[y * w + x] = 255;
  const q = kernels5G.exemplar5G(source, w, h, 1, mask, 5, 10, 40 * 18432000);
  assert.equal(q.status, 'ok');
  assert.deepEqual(Array.from(q.data), Array.from(source));
  const small = kernels5G.exemplar5G(new Uint8Array(4 * 4).fill(70), 4, 4, 1, new Uint8Array(16).fill(1), 5, 10, 40 * 18432000);
  assert.equal(small.status, 'fallback');
});

test('ip5g2-13-exemplar-oracle: optimized exemplar agrees with a naive recomputation on 30 seeded holes', () => {
  let state = 0x13e7a;
  const random = (n) => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state % n; };
  for (let trial = 0; trial < 30; trial++) {
    const width = 8 + random(17), height = 8 + random(17), ch = random(2) ? 1 : 3;
    const data = Uint8Array.from({ length: width * height * ch }, () => random(256));
    const mask = new Uint8Array(width * height), holeW = 1 + random(Math.max(1, Math.floor(width / 3))), holeH = 1 + random(Math.max(1, Math.floor(height / 3)));
    const x0 = trial % 4 === 0 ? 0 : random(width - holeW + 1), y0 = trial % 4 === 1 ? 0 : random(height - holeH + 1);
    for (let y = y0; y < y0 + holeH; y++) for (let x = x0; x < x0 + holeW; x++) mask[y * width + x] = 255;
    const patch = [5, 7, 9][random(3)], search = [10, 20, 40][random(3)];
    const actual = kernels5G.exemplar5G(data, width, height, ch, mask, patch, search, Number.MAX_SAFE_INTEGER);
    const expected = exemplarOracle(data, width, height, ch, mask, patch, search);
    assert.equal(actual.status, expected.status, `trial ${trial} status ${width}x${height} P${patch}`);
    if (actual.status === 'ok') {
      assert.deepEqual(actual.data, expected.data, `trial ${trial} bytes ${width}x${height} P${patch}`);
      assert.equal(actual.iterations, expected.iterations, `trial ${trial} iterations`);
      assert.equal(actual.search, expected.search, `trial ${trial} search`);
      for (let i = 0; i < mask.length; i++) if (!mask[i]) for (let c = 0; c < ch; c++) assert.equal(actual.data[i * ch + c], data[i * ch + c], `trial ${trial} unchanged ${i}/${c}`);
    }
  }
});

test('ip5g2-13-exemplar-mutant-fixtures: fixed random holes pin priority, SSD, copy, ties, confidence and normal', () => {
  let state = 0x13e7a;
  const random = (n) => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state % n; };
  const fixtures = new Map();
  for (let trial = 0; trial <= 12; trial++) {
    const width = 8 + random(17), height = 8 + random(17), ch = random(2) ? 1 : 3;
    const data = Uint8Array.from({ length: width * height * ch }, () => random(256));
    const mask = new Uint8Array(width * height), holeW = 1 + random(Math.max(1, Math.floor(width / 3))), holeH = 1 + random(Math.max(1, Math.floor(height / 3)));
    const x0 = trial % 4 === 0 ? 0 : random(width - holeW + 1), y0 = trial % 4 === 1 ? 0 : random(height - holeH + 1);
    for (let y = y0; y < y0 + holeH; y++) for (let x = x0; x < x0 + holeW; x++) mask[y * width + x] = 255;
    const patch = [5, 7, 9][random(3)], search = [10, 20, 40][random(3)];
    if ([0, 6, 8, 12].includes(trial)) fixtures.set(trial, { width, height, ch, data, mask, patch, search });
  }
  assert.deepEqual([...fixtures].map(([trial, f]) => [trial, f.width, f.height, f.ch, f.patch, f.search]), [
    [0, 18, 12, 1, 7, 20], [6, 11, 22, 1, 9, 10], [8, 17, 24, 1, 9, 20], [12, 24, 18, 1, 9, 40],
  ]);
  for (const [trial, fixture] of fixtures) {
    const actual = kernels5G.exemplar5G(fixture.data, fixture.width, fixture.height, fixture.ch, fixture.mask, fixture.patch, fixture.search, Number.MAX_SAFE_INTEGER);
    const expected = exemplarOracle(fixture.data, fixture.width, fixture.height, fixture.ch, fixture.mask, fixture.patch, fixture.search);
    assert.equal(actual.status, expected.status, `fixture ${trial} status`);
    assert.deepEqual(actual.data, expected.data, `fixture ${trial} pixels`);
    assert.equal(actual.iterations, expected.iterations, `fixture ${trial} iterations`);
  }
});

test('ip5g2-13-exemplar-striped-pattern: a repeated vertical pattern is restored exactly', () => {
  const width = 40, height = 40, data = new Uint8Array(width * height), mask = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) data[y * width + x] = Math.floor(x / 4) % 2 ? 255 : 0;
  for (let y = 18; y < 22; y++) for (let x = 4; x < 8; x++) mask[y * width + x] = 255;
  const out = kernels5G.exemplar5G(data, width, height, 1, mask, 5, 60, Number.MAX_SAFE_INTEGER);
  assert.equal(out.status, 'ok');
  assert.deepEqual(out.data, data);
});

test('ip5g2-11-isodata-strict-split-threshold: std equal to splitStd does not split', () => {
  const data = new Uint8Array(80);
  data.fill(96, 0, 28); data.fill(104, 28, 56); data.fill(255, 56);
  const q = kernels5G.isodata5G(data, 10, 8, 1, 2, 1, 0, 4, 0, 'mean', 1);
  assert.equal(q.splits, 0);
  assert.equal(q.clusters, 2);
});

test('ip5g2-13-exemplar-search-window: edge source is used and the next pixel is excluded', () => {
  const fixture = exemplarFixture('edge');
  const q = kernels5G.exemplar5G(fixture.data, fixture.width, fixture.height, 1, fixture.mask, 5, 10, Number.MAX_SAFE_INTEGER);
  assert.equal(q.status, 'ok');
  assert.equal(q.data[fixture.holeY * fixture.width + fixture.holeX], 40, 'the slightly imperfect source at search distance 10 wins');
});

test('ip5g2-13-exemplar-search-window-y: vertical edge source is used and the next row is excluded', () => {
  const width = 64, height = 40, holeX = 30, holeY = 20;
  const data = new Uint8Array(width * height).fill(200), mask = new Uint8Array(width * height);
  mask[holeY * width + holeX] = 255;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) data[(holeY + dy) * width + holeX + dx] = 100;
  for (const [sourceY, center] of [[holeY - 10, 40], [holeY + 11, 80]]) {
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) data[(sourceY + dy) * width + holeX + dx] = 100;
    data[sourceY * width + holeX] = center;
  }
  data[(holeY - 12) * width + holeX] = 101;
  const q = kernels5G.exemplar5G(data, width, height, 1, mask, 5, 10, Number.MAX_SAFE_INTEGER);
  assert.equal(q.status, 'ok');
  assert.equal(q.data[holeY * width + holeX], 40, 'the source at row distance 10 wins; row distance 11 is excluded');
});

test('ip5g2-13-exemplar-isophote-tie: first equal-magnitude gradient in scan order wins', () => {
  const width = 40, height = 40, data = new Uint8Array(width * height), mask = new Uint8Array(width * height);
  let state = 0x77a;
  for (let i = 0; i < data.length; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    data[i] = 100 + state % 3;
  }
  for (let y = 10; y < 22; y++) for (let x = 12; x < 24; x++) mask[y * width + x] = 255;
  data[8 * width + 8] = 100; data[8 * width + 10] = 110;
  data[7 * width + 9] = 100; data[9 * width + 9] = 100;
  data[14 * width + 9] = 100; data[16 * width + 9] = 110;
  data[15 * width + 8] = 100; data[15 * width + 10] = 100;
  const firstGradient = [data[8 * width + 10] - data[8 * width + 8], data[9 * width + 9] - data[7 * width + 9]];
  const laterGradient = [data[15 * width + 10] - data[15 * width + 8], data[16 * width + 9] - data[14 * width + 9]];
  assert.deepEqual(firstGradient, [10, 0]);
  assert.deepEqual(laterGradient, [0, 10]);
  assert.equal(firstGradient[0] ** 2 + firstGradient[1] ** 2, laterGradient[0] ** 2 + laterGradient[1] ** 2);
  const q = kernels5G.exemplar5G(data, width, height, 1, mask, 9, 10, Number.MAX_SAFE_INTEGER);
  assert.equal(q.status, 'ok');
  assert.deepEqual([
    q.data[15 * width + 18], q.data[15 * width + 21], q.data[15 * width + 22],
    q.data[16 * width + 17], q.data[17 * width + 19], q.data[18 * width + 20],
  ], [101, 102, 100, 101, 100, 102]);
});

test('ip5g2-13-exemplar-tie-break: equal SSD selects the first top-to-bottom, left-to-right source', () => {
  const fixture = exemplarFixture('tie');
  const q = kernels5G.exemplar5G(fixture.data, fixture.width, fixture.height, 1, fixture.mask, 5, 10, Number.MAX_SAFE_INTEGER);
  assert.equal(q.status, 'ok');
  assert.equal(q.data[fixture.holeY * fixture.width + fixture.holeX], 40);
  const job = buildStep('inpaint', { method: 'exemplar', patch: '5', search: 10 });
  job.src2 = { pipeline: [] };
  const py = buildPythonCode([job]);
  assert.match(py, /ex5g_bq = int\(np\.argmin\(ex5g_ssd\)\)/, 'NumPy argmin keeps the first SSD tie');
});

test('ip5g2-13-exemplar-work-cap: a 4-to-40 budget workload still completes', () => {
  const width = 520, height = 520;
  const src = image(width, height, 3, (x, y) => [(x * 7 + y * 11) & 255, (x * 13 + y * 3) & 255, (x * 5 + y * 17) & 255]);
  const mask = image(width, height, 1, (x, y) => (x >= 20 && x < 23 && y >= 20 && y < 23) || (x >= 500 && x < 503 && y >= 500 && y < 504) ? 255 : 0);
  const ctx = { src2: mask };
  try {
    const out = apply('inpaint', src, { method: 'exemplar', patch: '9', search: 10 }, ctx);
    try {
      assert.match(ctx.info, /Exemplar/);
      const infoNumbers = (ctx.info.match(/\d+/g) || []).map(Number);
      assert.ok(infoNumbers.length >= 4 && infoNumbers[3] > 0, `round count in ${ctx.info}`);
      assert.equal(out.rows, height);
    } finally { out.delete(); }
  } finally { src.delete(); mask.delete(); }
});

test('ip5g2-13-exemplar-host-guards-and-fallback: early exits preserve layout and no-patch delegates to Laplace', () => {
  const w = 9, h = 9, a = image(w, h, 4, (x, y) => [x * 19, y * 23, (x * 13 + y * 11) & 255, 77]);
  const b = image(w, h, 1, (x, y) => x >= 3 && x <= 5 && y >= 3 && y <= 5 ? 255 : 0);
  const exemplarCtx = { src2: b }, laplaceCtx = { src2: b };
  let exemplar = null, laplace = null;
  try {
    exemplar = apply('inpaint', a, { method: 'exemplar', patch: '9' }, exemplarCtx);
    laplace = apply('inpaint', a, { method: 'laplace' }, laplaceCtx);
    assert.equal(exemplar.channels(), 3);
    assert.deepEqual(Array.from(exemplar.data), Array.from(laplace.data));
    assert.match(exemplarCtx.info, /Laplace/);
    assert.match(exemplarCtx.info, /mask/);
  } finally { exemplar?.delete(); laplace?.delete(); a.delete(); b.delete(); }

  const width = 640, height = 480, rgba = image(width, height, 4, (x, y) => [(x * 7) & 255, (y * 11) & 255, (x + y) & 255, 99]);
  const largeMask = image(width, height, 1, (_x, y) => y < Math.floor(height * 0.24) ? 255 : 0);
  const ctx = { src2: largeMask };
  try {
    const out = apply('inpaint', rgba, { method: 'exemplar', patch: '9' }, ctx);
    assert.equal(out.channels(), 3);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let c = 0; c < 3; c++) assert.equal(out.data[(y * width + x) * 3 + c], rgba.data[(y * width + x) * 4 + c]);
    assert.match(ctx.info, /mask/);
    out.delete();
  } finally { rgba.delete(); largeMask.delete(); }
});

test('ip5g2-13-exemplar-reduced-search-start: work cap halves a whole-image search from 60 to 30', () => {
  const width = 160, height = 160, data = new Uint8Array(width * height * 3).fill(73), mask = new Uint8Array(width * height);
  for (let y = 79; y <= 80; y++) for (let x = 79; x <= 80; x++) mask[y * width + x] = 1;
  const q = kernels5G.exemplar5G(data, width, height, 3, mask, 5, 60, 400000);
  assert.equal(q.status, 'ok');
  assert.equal(q.search, 30);
  assert.equal(q.requestedSearch, 60);
  assert.equal(q.reduced, true);
});

test('ip5g2-schema: host, palette, option groups, icons, labels and Pass/Fail registration agree', () => {
  assert.equal(Object.keys(OPS).length, 166);
  const mapping = [
    ['image_registration', 'IMAGE REGISTRATION', 'phase_register'],
    ['segment_edge', 'SEGMENTATION & EDGE', 'scribble_segment'],
    ['color_tools', 'COLOR', 'isodata_segment'],
    ['frequency', 'FREQUENCY DOMAIN', 'fir_design'],
  ];
  for (const [host, sectionTitle, id] of mapping) {
    const hostIds = FILTERS[host].params.find((p) => p.key === 'mode').options.map((o) => o.value);
    const paletteIds = TOOL_SECTIONS.find((s) => s.title === sectionTitle).ops;
    assert.deepEqual(paletteIds.filter((value) => hostIds.includes(value)), hostIds);
    assert.ok(hostIds.includes(id));
    if (id !== 'isodata_segment') assert.match(OP_PRIMARY[id](defaults(id)), /[\u0E00-\u0E7F]/);
    else assert.equal(OP_PRIMARY[id](defaults(id)), 'ISODATA k=4');
    assert.ok(OP_ICON[id]);
    assert.equal(OP_TO_FILTER[id], host);
  }
  const newIcons = ['phase_register', 'scribble_segment', 'isodata_segment', 'fir_design'].map((id) => OP_ICON[id]);
  const allIcons = [...Object.values(OP_ICON), ...Object.values(GROUP_ICON)];
  for (const icon of newIcons) assert.equal(allIcons.filter((value) => value === icon).length, 1);
  assert.deepEqual(OPTION_GROUPS['segment_edge.mode'].find((x) => x[0] === 'แยกวัตถุเป็นชิ้น')[1].at(-1), 'scribble_segment');
  assert.deepEqual(OPTION_GROUPS['color_tools.mode'].find((x) => x[0] === 'เลือก / แบ่งตามสี')[1], ['color_inrange', 'kmeans_segment', 'isodata_segment', 'slic']);
  for (const id of ['phase_register', 'scribble_segment', 'isodata_segment']) assert.ok(PF_VALUE_OPS.has(id));
  assert.ok(!PF_VALUE_OPS.has('fir_design'));
  const method = OPS.inpaint.params.find((p) => p.key === 'method');
  assert.equal(method.options.at(-1).value, 'exemplar');
  assert.ok(method.options.at(-1).hint.includes('ช้า'));
  for (const key of ['patch', 'search']) assert.deepEqual(OPS.inpaint.params.find((p) => p.key === key).visibleWhen, [{ key: 'method', in: ['exemplar'] }]);
  assert.deepEqual(OPS.inpaint.params.find((p) => p.key === 'radius').visibleWhen, [{ key: 'method', in: ['telea', 'ns'] }]);
  assert.equal(OPS.scribble_segment.params[0].type, 'strokes');
  assert.equal(OPS.scribble_segment.params[0].default, null);
  assert.match(OPS.scribble_segment.modeHint, /ช้า/);
  assert.match(OPS.isodata_segment.modeHint, /ช้า/);
});

function buildStep(id, params = {}, nodeId = 1) {
  const local = { ...defaults(id), ...params }, host = OP_TO_FILTER[id] || id;
  return { id: host, params: { ...defaultParams(host), mode: id, ...Object.fromEntries(Object.entries(local).map(([key, value]) => [`${id}__${key}`, value])) }, nodeId };
}

function image(width, height, channels, pixel) {
  if (!width || !height) return new cv.Mat();
  const values = new Uint8Array(width * height * channels);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const value = pixel(x, y);
    for (let c = 0; c < channels; c++) values[(y * width + x) * channels + c] = Array.isArray(value) ? value[c] : value;
  }
  const type = channels === 1 ? cv.CV_8UC1 : channels === 4 ? cv.CV_8UC4 : cv.CV_8UC3;
  return cv.matFromArray(height, width, type, values);
}

function apply(id, src, params = {}, ctx = {}) {
  return OPS[id].apply(cv, src, { ...defaults(id), ...params }, ctx);
}

function scribbles(w, h, entries) {
  return { v: 1, w, h, s: entries.map(([label, x, y, rr = 12]) => [label, rr, pixelToStrokeQ(x, w), pixelToStrokeQ(y, h)]) };
}

function exemplarFixture(kind) {
  const width = 64, height = 40, holeX = 30, holeY = 20;
  const data = new Uint8Array(width * height).fill(200);
  const mask = new Uint8Array(width * height);
  mask[holeY * width + holeX] = 255;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    data[(holeY + dy) * width + holeX + dx] = 100;
  }
  const candidates = kind === 'edge' ? [[holeX - 10, 40], [holeX + 11, 80]] : [[holeX - 10, 40], [holeX + 10, 160]];
  for (const [qx, center] of candidates) {
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) data[(holeY + dy) * width + qx + dx] = 100;
    data[holeY * width + qx] = center;
  }
  if (kind === 'edge') data[holeY * width + candidates[0][0] - 2] = 101;
  return { width, height, holeX, holeY, data, mask };
}

function isodataOracle(pix, width, height, ch, k, iterations, minSizePct, splitStd, mergeDist, view = 'mean') {
  const n = width * height, nf = ch === 1 ? 1 : 3;
  const features = Array.from({ length: n }, (_, i) => Array.from({ length: nf }, (_, c) => pix[i * ch + c]));
  const keys = Array.from({ length: n }, (_, i) => nf === 1 ? pix[i] : (pix[3 * i] << 16) | (pix[3 * i + 1] << 8) | pix[3 * i + 2]);
  const lum = (i) => nf === 1 ? features[i][0] : 299 * features[i][0] + 587 * features[i][1] + 114 * features[i][2];
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => lum(a) - lum(b) || keys[a] - keys[b] || a - b);
  let centers = Array.from({ length: k }, (_, j) => features[order[Math.floor(((2 * j + 1) * n) / (2 * k))]].slice());
  const assign = (cs) => Array.from({ length: n }, (_, i) => {
    let best = Infinity, winner = 0;
    for (let j = 0; j < cs.length; j++) {
      let d = (features[i][0] - cs[j][0]) * (features[i][0] - cs[j][0]);
      if (nf === 3) for (let c = 1; c < 3; c++) { const t = features[i][c] - cs[j][c]; d = d + t * t; }
      if (d < best) { best = d; winner = j; }
    }
    return winner;
  });
  const stats = (labels, count) => {
    const sizes = Array(count).fill(0), s1 = Array.from({ length: nf }, () => Array(count).fill(0)), s2 = Array.from({ length: nf }, () => Array(count).fill(0));
    for (let i = 0; i < n; i++) {
      const j = labels[i]; sizes[j]++;
      for (let c = 0; c < nf; c++) { s1[c][j] += features[i][c]; s2[c][j] += features[i][c] * features[i][c]; }
    }
    return { sizes, s1, s2 };
  };
  const dropEmpty = (labels, cs, st) => {
    const keep = st.sizes.map((v) => v > 0);
    if (keep.every(Boolean)) return [labels, cs, st, 0];
    const map = [], nextCenters = [];
    for (let j = 0; j < keep.length; j++) { map[j] = keep[j] ? nextCenters.length : -1; if (keep[j]) nextCenters.push(cs[j]); }
    const nextLabels = labels.map((v) => map[v]);
    return [nextLabels, nextCenters, stats(nextLabels, nextCenters.length), keep.length - nextCenters.length];
  };
  const minSize = Math.max(1, Math.floor((minSizePct * n) / 100)), maxK = 2 * k, floorK = Math.max(1, Math.floor(k / 2));
  let splits = 0, merges = 0, cuts = 0, used = 0, converged = false, prev = null, prevK = -1;
  for (let it = 1; it <= iterations; it++) {
    used = it;
    let labels = assign(centers), st = stats(labels, centers.length), changed = false, didSplit = false;
    let keep = st.sizes.map((v) => v >= minSize);
    if (!keep.some(Boolean)) { let best = 0; for (let j = 1; j < st.sizes.length; j++) if (st.sizes[j] > st.sizes[best]) best = j; keep[best] = true; }
    if (!keep.every(Boolean)) {
      cuts += keep.filter((v) => !v).length; changed = true;
      centers = centers.filter((_, j) => keep[j]);
      labels = assign(centers); st = stats(labels, centers.length);
      let removed; [labels, centers, st, removed] = dropEmpty(labels, centers, st); cuts += removed;
    }
    const kk = centers.length, means = [], sigma = [];
    for (let j = 0; j < kk; j++) {
      const mean = [], sig = [];
      for (let c = 0; c < nf; c++) { const m = st.s1[c][j] / st.sizes[j], v = st.s2[c][j] / st.sizes[j] - m * m; mean.push(m); sig.push(Math.sqrt(Math.max(0, v))); }
      means.push(mean); sigma.push(sig);
    }
    centers = means.map((m) => m.slice());
    if (centers.length < maxK) {
      const before = centers.length;
      for (let j = 0; j < before && centers.length < maxK; j++) {
        let cmax = 0; for (let c = 1; c < nf; c++) if (sigma[j][c] > sigma[j][cmax]) cmax = c;
        if (sigma[j][cmax] > splitStd && st.sizes[j] >= 2 * minSize) {
          const half = sigma[j][cmax] / 2, high = means[j].slice();
          centers[j][cmax] = means[j][cmax] - half; high[cmax] = means[j][cmax] + half;
          centers.push(high); splits++; changed = true; didSplit = true;
        }
      }
    }
    if (!didSplit && centers.length > floorK) {
      const pairs = [];
      for (let i = 0; i < kk; i++) for (let j = i + 1; j < kk; j++) {
        let d = (means[i][0] - means[j][0]) * (means[i][0] - means[j][0]);
        if (nf === 3) for (let c = 1; c < 3; c++) { const t = means[i][c] - means[j][c]; d = d + t * t; }
        const dist = Math.sqrt(d); if (dist < mergeDist) pairs.push([dist, i, j]);
      }
      pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
      const taken = new Set(), gone = new Set(); let done = 0;
      for (const [, i, j] of pairs) {
        if (done >= 2 || centers.length - gone.size <= floorK) break;
        if (taken.has(i) || taken.has(j)) continue;
        const ni = st.sizes[i], nj = st.sizes[j];
        centers[i] = means[i].map((v, c) => (ni * v + nj * means[j][c]) / (ni + nj));
        gone.add(j); taken.add(i); taken.add(j); done++; merges++; changed = true;
      }
      centers = centers.filter((_, j) => !gone.has(j));
    }
    const same = prev !== null && prevK === kk && labels.every((v, i) => v === prev[i]);
    if (!changed && same) { converged = true; break; }
    prev = labels; prevK = kk;
  }
  let labels = assign(centers), st = stats(labels, centers.length), removed;
  [labels, centers, st, removed] = dropEmpty(labels, centers, st); cuts += removed;
  const means = centers.map((_, j) => Array.from({ length: nf }, (_, c) => st.s1[c][j] / st.sizes[j]));
  const outCh = view === 'labels' ? 3 : ch, data = new Uint8Array(n * outCh);
  for (let i = 0; i < n; i++) {
    const j = labels[i];
    if (view === 'labels') data.set(kernels5G.ISODATA_PALETTE_5G[j % 12], i * 3);
    else for (let c = 0; c < nf; c++) data[i * outCh + c] = Math.floor(means[j][c] + 0.5);
  }
  return { data, clusters: centers.length, used, splits, merges, cuts, converged };
}

function exemplarOracle(data, width, height, ch, mask, patch, search) {
  const r = (patch - 1) / 2, n = width * height, cur = Uint8Array.from(data), om0 = Uint8Array.from(mask, (v) => v ? 1 : 0), om = Uint8Array.from(om0), confidence = Float64Array.from(om, (v) => v ? 0 : 1);
  let nz = 0, bx0 = width, bx1 = -1, by0 = height, by1 = -1;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (om0[y * width + x]) { nz++; bx0 = Math.min(bx0, x); bx1 = Math.max(bx1, x); by0 = Math.min(by0, y); by1 = Math.max(by1, y); }
  const candidates = (s) => {
    const out = [];
    for (let y = Math.max(r, by0 - s); y <= Math.min(height - 1 - r, by1 + s); y++) for (let x = Math.max(r, bx0 - s); x <= Math.min(width - 1 - r, bx1 + s); x++) {
      let clear = true;
      for (let dy = -r; dy <= r && clear; dy++) for (let dx = -r; dx <= r; dx++) if (om0[(y + dy) * width + x + dx]) { clear = false; break; }
      if (clear) out.push(y * width + x);
    }
    return out;
  };
  let usedSearch = search, Q = candidates(usedSearch);
  if (!Q.length) { usedSearch = Math.max(width, height); Q = candidates(usedSearch); }
  if (!Q.length) return { status: 'fallback', nz };
  const offsets = [];
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) offsets.push([dy, dx]);
  const igr = new Int32Array(n);
  for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) igr[i] += cur[i * ch + c];
  const inside = (x, y) => x >= 0 && x < width && y >= 0 && y < height;
  let left = nz, iterations = 0;
  while (left) {
    const front = [];
    for (let y = Math.max(0, by0 - 1); y <= Math.min(height - 1, by1 + 1); y++) for (let x = Math.max(0, bx0 - 1); x <= Math.min(width - 1, bx1 + 1); x++) {
      const i = y * width + x;
      if (om[i] && ((y > 0 && !om[i - width]) || (y + 1 < height && !om[i + width]) || (x > 0 && !om[i - 1]) || (x + 1 < width && !om[i + 1]))) front.push(i);
    }
    let bestPriority = -Infinity, bestIndex = -1, bestConfidence = 0;
    for (const i of front) {
      const x = i % width, y = Math.floor(i / width); let acc = 0, count = 0;
      for (const [dy, dx] of offsets) { const qx = x + dx, qy = y + dy; if (inside(qx, qy)) { acc += confidence[qy * width + qx]; count++; } }
      const cp = acc / count;
      const nx = (x + 1 < width && om[i + 1] ? 1 : 0) - (x > 0 && om[i - 1] ? 1 : 0);
      const ny = (y + 1 < height && om[i + width] ? 1 : 0) - (y > 0 && om[i - width] ? 1 : 0);
      const nlen = Math.sqrt(nx * nx + ny * ny);
      let bestG2 = -1, gx = 0, gy = 0;
      for (const [dy, dx] of offsets) {
        const qx = x + dx, qy = y + dy;
        if (!inside(qx, qy) || !inside(qx - 1, qy) || !inside(qx + 1, qy) || !inside(qx, qy - 1) || !inside(qx, qy + 1)) continue;
        const qi = qy * width + qx;
        if (om[qi] || om[qi - 1] || om[qi + 1] || om[qi - width] || om[qi + width]) continue;
        const dxg = igr[qi + 1] - igr[qi - 1], dyg = igr[qi + width] - igr[qi - width], g2 = dxg * dxg + dyg * dyg;
        if (g2 > bestG2) { bestG2 = g2; gx = dxg; gy = dyg; }
      }
      const D = nlen > 0 ? Math.abs((0 - gy) * nx + gx * ny) / (nlen * 2 * 255 * ch) : 0;
      const priority = cp * (D + 0.001);
      if (priority > bestPriority) { bestPriority = priority; bestIndex = i; bestConfidence = cp; }
    }
    assert.ok(bestIndex >= 0, 'oracle front has a fillable pixel');
    const px = bestIndex % width, py = Math.floor(bestIndex / width);
    let bestSsd = Infinity, bestQ = Q[0];
    for (const candidate of Q) {
      const qx = candidate % width, qy = Math.floor(candidate / width); let ssd = 0;
      for (const [dy, dx] of offsets) {
        const x = px + dx, y = py + dy;
        if (!inside(x, y) || om[y * width + x]) continue;
        const source = ((qy + dy) * width + qx + dx) * ch, target = (y * width + x) * ch;
        for (let c = 0; c < ch; c++) { const d = cur[target + c] - cur[source + c]; ssd += d * d; }
      }
      if (ssd < bestSsd) { bestSsd = ssd; bestQ = candidate; }
    }
    const qx = bestQ % width, qy = Math.floor(bestQ / width);
    for (const [dy, dx] of offsets) {
      const x = px + dx, y = py + dy, sx = qx + dx, sy = qy + dy;
      if (!inside(x, y)) continue;
      const i = y * width + x;
      if (!om[i]) continue;
      const target = i * ch, source = (sy * width + sx) * ch;
      for (let c = 0; c < ch; c++) cur[target + c] = cur[source + c];
      igr[i] = 0; for (let c = 0; c < ch; c++) igr[i] += cur[target + c];
      confidence[i] = bestConfidence; om[i] = 0; left--;
    }
    iterations++;
  }
  return { status: 'ok', data: cur, iterations, search: usedSearch, nz };
}

test('ip5g2-T5-Python-syntax-and-text: all variants sanitize, compile, and keep constrained codegen primitives', { skip: !python }, () => {
  const jobs = [];
  for (const view of ['aligned', 'overlay', 'diff', 'surface']) for (const window of [true, false]) for (const subpixel of [true, false]) jobs.push(buildStep('phase_register', { view, window, subpixel }));
  for (const view of ['overlay', 'mask', 'distance']) for (const metric of ['chessboard', 'quasi']) jobs.push(buildStep('scribble_segment', { view, metric, strokes: { v: 1, w: 10, h: 10, s: [[1, 5, 1000, 1000], [2, 5, 8000, 8000]] } }));
  for (const view of ['mean', 'labels']) for (const k of [2, 12]) jobs.push(buildStep('isodata_segment', { k, view }));
  for (const method of ['window', 'sampling', 'mcclellan']) for (const response of ['lowpass', 'highpass', 'bandpass', 'bandstop']) for (const view of ['result', 'response', 'kernel']) for (const add128 of [true, false]) jobs.push(buildStep('fir_design', { method, response, view, size: 21, add128 }));
  for (const patch of ['5', '7', '9']) for (const search of [10, 60, 200]) jobs.push(buildStep('inpaint', { method: 'exemplar', patch, search }));
  jobs.push(
    buildStep('scribble_segment', { strokes: {}, edge: Infinity, metric: '__proto__' }),
    buildStep('scribble_segment', { strokes: 'x' }),
    buildStep('scribble_segment', { strokes: { v: 1, s: [[1, 'a']] } }),
    buildStep('scribble_segment', { strokes: { v: 1, s: 'x' } }),
    buildStep('scribble_segment', { strokes: JSON.parse('{"__proto__":{"v":1}}') }),
    buildStep('isodata_segment', { k: 'constructor', iterations: NaN, minSizePct: Infinity, splitStd: null, mergeDist: -999, view: '__proto__' }),
    buildStep('fir_design', { method: 'constructor', response: '__proto__', cutoff: Infinity, cutoff2: NaN, size: 100, window: 'constructor' }),
    buildStep('inpaint', { method: 'exemplar', patch: '__proto__', search: Infinity }),
  );
  const compiled = [];
  for (const job of jobs) {
    const id = job.params.mode;
    if (id === 'phase_register' || id === 'inpaint') job.src2 = { pipeline: [] };
    compiled.push(buildPythonCode([job]));
    const chain = [buildStep('colorspace', { space: 'hsv' }, 1), { ...job, nodeId: 2 }];
    if (id === 'phase_register' || id === 'inpaint') chain[1].src2 = { pipeline: [] };
    chain.push(buildStep('pass_fail', { metric: 'qt_blocks', cond: 'ge', lo: 1 }, 3));
    compiled.push(buildPythonCode(chain));
  }
  const textJobs = [
    buildStep('scribble_segment', { strokes: { v: 1, w: 10, h: 10, s: [[1, 5, 1000, 1000], [2, 5, 8000, 8000]] } }),
    buildStep('fir_design', { method: 'sampling', response: 'bandpass', size: 7, view: 'result' }),
  ];
  const scribbleCode = buildPythonCode([textJobs[0]]);
  assert.match(scribbleCode, /stroke_labels\(/);
  assert.match(scribbleCode, /strokes5g/);
  assert.doesNotMatch(scribbleCode, /img_b/);
  const firCode = buildPythonCode([textJobs[1]]);
  assert.match(firCode, /reflect101_5g/);
  assert.doesNotMatch(firCode, /np\.pad\([^\n]*(?:reflect|edge)/i);
  assert.doesNotMatch(firCode, /np\.square|cv2\.filter2D|GaussianBlur|\*\*|np\.sum\(|np\.mean\(|np\.dot\(|\s@\s|kmeans\(/);
  const malformedCode = jobs.slice(-7).map((job) => {
    if (job.params.mode === 'inpaint') job.src2 = { pipeline: [] };
    return buildPythonCode([job]);
  });
  for (const code of malformedCode) assert.doesNotMatch(code, /__proto__|constructor|Infinity|NaN/);
  execFileSync('python', ['-c', 'import sys,json\nfor source in json.loads(sys.stdin.buffer.read()): compile(source,"<ip5g2>","exec")'], {
    input: Buffer.from(JSON.stringify(compiled), 'utf8'), stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env, PYTHONIOENCODING: 'utf8' },
  });
});

test('ip5g2-T6-parity: 5G-2 images, info numbers, values and late Pass/Fail match Python', { skip: !parity }, () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ip5g2-parity-'));
  const rgb = image(45, 37, 3, (x, y) => [(x * 13 + y * 7) & 255, (x * 3 + y * 17 + 31) & 255, (x * 11 + y * 5 + 93) & 255]);
  const rgba = image(45, 37, 4, (x, y) => [(x * 13 + y * 7) & 255, (x * 3 + y * 17 + 31) & 255, (x * 11 + y * 5 + 93) & 255, 177]);
  const gray = image(45, 37, 1, (x, y) => (x * 29 + y * 47 + x * y * 3) & 255);
  const tinyH = image(1, 9, 3, (_x, y) => [y * 23, y * 11, 255 - y * 19]);
  const tinyW = image(9, 1, 3, (x) => [x * 23, 255 - x * 17, x * 11]);
  const large = image(640, 480, 3, (x, y) => [(Math.imul(x + 19, 73) ^ Math.imul(y + 3, 29)) & 255, (Math.imul(x + 7, 41) ^ Math.imul(y + 11, 97)) & 255, (Math.imul(x + 13, 113) ^ Math.imul(y + 5, 17)) & 255]);
  let noiseState = 0x71f2a5;
  const largeNoise = image(640, 480, 3, () => {
    const pixel = [];
    for (let c = 0; c < 3; c++) { noiseState = (Math.imul(noiseState, 1664525) + 1013904223) >>> 0; pixel.push(noiseState >>> 24); }
    return pixel;
  });
  const largeB = image(320, 240, 1, (x, y) => (Math.imul(x + 3, 71) ^ Math.imul(y + 19, 43)) & 255);
  const srcs = [gray, rgb, rgba];
  const masks = new Map();
  const maskFor = (src, shape = 'hole') => {
    const key = `${src.cols}x${src.rows}/${shape}`;
    if (masks.has(key)) return masks.get(key);
    const mask = image(src.cols, src.rows, 1, (x, y) => shape === 'border'
      ? (x < 4 && y < 8 ? 255 : 0)
      : shape === 'band' ? (y >= Math.floor(src.rows * 0.2) && y < Math.floor(src.rows * 0.4) ? 255 : 0)
        : (x >= Math.floor(src.cols / 2) - 2 && x < Math.floor(src.cols / 2) + 2 && y >= Math.floor(src.rows / 2) - 2 && y < Math.floor(src.rows / 2) + 2 ? 255 : 0));
    masks.set(key, mask); return mask;
  };
  let strokeSeed = 0x5eed5;
  const rnd = (n) => { strokeSeed = (Math.imul(strokeSeed, 1664525) + 1013904223) >>> 0; return strokeSeed % n; };
  const strokeSets = Array.from({ length: 3 }, (_, set) => ({ v: 1, w: 45, h: 37, s: [
    [1, 8 + rnd(18), pixelToStrokeQ(4 + rnd(18), 45), pixelToStrokeQ(4 + rnd(25), 37)],
    [2, 8 + rnd(18), pixelToStrokeQ(23 + rnd(18), 45), pixelToStrokeQ(7 + rnd(25), 37)],
    ...(set === 1 ? [[0, 4, pixelToStrokeQ(22, 45), pixelToStrokeQ(18, 37)]] : []),
  ] }));
  const cases = [];
  const phaseSources = [];
  const fixtureSources = [];
  const add = (id, src, params = {}, b = null, extra = {}) => cases.push({ id, src, params, b, ...extra });
  const phasePair = (pixel, dx, dy) => {
    const base = image(64, 48, 1, pixel);
    const shifted = image(64, 48, 1, (x, y) => base.data[((y - dy + 48) % 48) * 64 + ((x - dx + 64) % 64)]);
    phaseSources.push(shifted);
    return [shifted, base];
  };
  const addExemplarFixture = (kind) => {
    const fixture = exemplarFixture(kind);
    const src = image(fixture.width, fixture.height, 3, (x, y) => {
      const value = fixture.data[y * fixture.width + x]; return [value, value, value];
    });
    const mask = image(fixture.width, fixture.height, 1, (x, y) => fixture.mask[y * fixture.width + x]);
    fixtureSources.push(src);
    add('inpaint', src, { method: 'exemplar', patch: '5', search: 10 }, mask);
  };
  for (const src of srcs) for (const view of ['aligned', 'overlay', 'diff', 'surface']) add('phase_register', src, { view, window: true, subpixel: true }, image(src.cols, src.rows, 1, (x, y) => (x * 17 + y * 31 + x * y) & 255));
  for (const src of [tinyH, tinyW]) for (const view of ['aligned', 'overlay', 'diff', 'surface']) add('phase_register', src, { view }, image(src.cols, src.rows, 1, (x, y) => x * 11 + y * 29));
  add('phase_register', rgb, { view: 'aligned', window: false, subpixel: false }, image(45, 37, 1, (x, y) => (x * 19 + y * 23 + x * y) & 255));
  add('phase_register', large, { view: 'aligned', window: false, subpixel: false }, largeB);
  add('phase_register', rgb, { view: 'aligned', window: false, subpixel: false }, image(45, 37, 1, (x, y) => (x * 7 + y * 29 + x * y * 3) & 255), { chain: true });
  for (const view of ['aligned', 'overlay', 'diff', 'surface']) add('phase_register', rgb, { view, window: false, subpixel: false }, image(45, 37, 1, (x, y) => (x * 23 + y * 11 + x * y) & 255), { preGray: true });
  for (const window of [false, true]) {
    const [a, b] = phasePair((x, y) => (Math.imul(x + 11, 73) ^ Math.imul(y + 5, 29) ^ Math.imul(x + 3, y + 7)) & 255, 7, -4);
    add('phase_register', a, { view: 'aligned', window, subpixel: true }, b);
  }
  for (const window of [false, true]) {
    const [a, b] = phasePair((x, y) => ((Math.floor(x / 8) + Math.floor(y / 8)) & 1) ? 255 : 0, -5, 4);
    add('phase_register', a, { view: 'aligned', window, subpixel: true }, b);
  }
  for (const [dx, dy] of [[32, 0], [0, 24]]) {
    const [a, b] = phasePair((x, y) => (Math.imul(x + 3, 71) ^ Math.imul(y + 11, 43) ^ Math.imul(x, y + 1)) & 255, dx, dy);
    add('phase_register', a, { view: 'aligned', window: false, subpixel: false }, b);
  }
  for (const src of srcs) for (const view of ['overlay', 'mask', 'distance']) add('scribble_segment', src, { view, strokes: strokeSets[0] });
  for (const src of [tinyH, tinyW]) add('scribble_segment', src, { view: 'mask', strokes: scribbles(src.cols, src.rows, [[1, 12, 0, 0], [2, 12, src.cols - 1, src.rows - 1]]) });
  for (const strokes of strokeSets) add('scribble_segment', rgb, { view: 'overlay', strokes, metric: 'quasi' });
  add('scribble_segment', rgb, { view: 'mask', strokes: scribbles(45, 37, [[2, 10, 10]]) });
  add('scribble_segment', rgba, { view: 'distance', strokes: scribbles(45, 37, [[1, 12, 12]]) });
  add('scribble_segment', rgb, { view: 'overlay', strokes: strokeSets[0] }, null, { chain: true });
  add('scribble_segment', largeNoise, { view: 'overlay', strokes: { v: 1, w: 640, h: 480, s: [[1, 10, 900, 900], [2, 10, 9100, 9100]] } });
  for (const src of srcs) for (const view of ['mean', 'labels']) add('isodata_segment', src, { k: 4, iterations: 12, view });
  for (const src of [tinyH, tinyW]) add('isodata_segment', src, { k: 2, iterations: 3, view: 'mean' });
  add('isodata_segment', rgb, { k: 2, view: 'mean' });
  add('isodata_segment', rgb, { k: 12, view: 'labels' });
  add('isodata_segment', largeNoise, { k: 4, iterations: 20, view: 'mean' });
  add('isodata_segment', rgb, { k: 4, view: 'mean' }, null, { chain: true });
  for (const method of ['window', 'sampling', 'mcclellan']) for (const response of ['lowpass', 'highpass', 'bandpass', 'bandstop']) for (const view of ['result', 'response', 'kernel']) add('fir_design', rgb, { method, response, view, size: 11 });
  for (const src of srcs) add('fir_design', src, { method: 'window', response: 'lowpass', size: 11, view: 'result' });
  add('fir_design', tinyW, { method: 'sampling', response: 'highpass', size: 21, view: 'result' });
  add('fir_design', tinyH, { method: 'sampling', response: 'highpass', size: 21, view: 'result' });
  add('fir_design', large, { method: 'sampling', response: 'lowpass', size: 21, view: 'result' });
  add('fir_design', rgb, { method: 'window', response: 'lowpass', size: 11, view: 'result' }, null, { chain: true });
  for (const [index, src] of srcs.entries()) add('inpaint', src, { method: 'exemplar', patch: ['5', '7', '9'][index], search: 60 }, maskFor(src));
  add('inpaint', rgb, { method: 'exemplar', patch: '7', search: 60 }, maskFor(rgb, 'border'));
  add('inpaint', tinyH, { method: 'exemplar', patch: '5', search: 10 }, image(1, 9, 1, (_x, y) => y >= 4 ? 255 : 0));
  add('inpaint', tinyW, { method: 'exemplar', patch: '5', search: 10 }, image(9, 1, 1, (x) => x >= 4 ? 255 : 0));
  add('inpaint', large, { method: 'exemplar', patch: '5', search: 60 }, image(640, 480, 1, (x, y) => x >= 300 && x < 340 && y >= 220 && y < 260 ? 255 : 0));
  add('inpaint', large, { method: 'exemplar', patch: '9', search: 60 }, maskFor(large, 'band'));
  add('inpaint', rgb, { method: 'exemplar', patch: '5', search: 60 }, image(31, 27, 1, (x, y) => x >= 14 && x < 18 && y >= 11 && y < 15 ? 255 : 0));
  add('inpaint', large, { method: 'exemplar', patch: '9', search: 60 }, image(640, 480, 1, (_x, y) => y < Math.ceil(0.26 * 480) ? 255 : 0));
  add('inpaint', rgb, { method: 'exemplar', patch: '5', search: 60 }, maskFor(rgb), { chain: true });
  for (const patch of ['5', '7']) add('inpaint', rgb, { method: 'exemplar', patch, search: 60 }, image(45, 37, 1, (x, y) => x >= 19 && x < 25 && y >= 15 && y < 21 ? 255 : 0), { preGray: true });
  addExemplarFixture('edge');
  addExemplarFixture('tie');
  for (const id of ['phase_register', 'scribble_segment', 'isodata_segment']) {
    const params = id === 'phase_register' ? { view: 'aligned', window: false, subpixel: false }
      : id === 'scribble_segment' ? { view: 'mask', strokes: strokeSets[0] } : { k: 4, view: 'mean' };
    add(id, rgb, params, id === 'phase_register' ? image(45, 37, 1, (x, y) => (x * 5 + y * 31 + 11) & 255) : null, { passFail: true });
  }

  const toBgrBytes = (mat) => {
    const rgbMat = new cv.Mat(), bgrMat = new cv.Mat();
    try {
      if (mat.channels() === 1) cv.cvtColor(mat, rgbMat, cv.COLOR_GRAY2RGB);
      else if (mat.channels() === 4) cv.cvtColor(mat, rgbMat, cv.COLOR_RGBA2RGB);
      else mat.copyTo(rgbMat);
      cv.cvtColor(rgbMat, bgrMat, cv.COLOR_RGB2BGR);
      return Buffer.from(bgrMat.data);
    } finally { rgbMat.delete(); bgrMat.delete(); }
  };
  const applyWithNeeds = (id, src, params, b) => {
    let input = src, converted = null;
    if (needsOf(OPS[id], { ...defaults(id), ...params }) === 'gray' && src.channels() > 1) {
      converted = new cv.Mat();
      cv.cvtColor(src, converted, src.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY);
      input = converted;
    }
    const ctx = b ? { src2: b } : {}, out = OPS[id].apply(cv, input, { ...defaults(id), ...params }, ctx);
    if (converted) converted.delete();
    return { out, ctx };
  };
  const runApp = (item) => {
    if (item.passFail) {
      const qt = applyWithNeeds('quadtree', item.src, { threshold: 0, view: 'mean' }, null);
      try {
        const current = applyWithNeeds(item.id, qt.out, item.params, item.b);
        try {
          const ctx = { upstream: { op: item.id, values: current.ctx.values } };
          const out = OPS.pass_fail.apply(cv, current.out, { ...defaults('pass_fail'), metric: 'qt_blocks', cond: 'ge', lo: 1 }, ctx);
          return { out, ctx, pf: ctx.info.startsWith('NG:') ? 'NG' : 'OK' };
        } finally { current.out.delete(); }
      } finally { qt.out.delete(); }
    }
    let input = item.src, temp = null, preGray = null;
    try {
      if (item.preGray) { preGray = OPS.grayscale.apply(cv, input, defaults('grayscale'), {}); input = preGray; }
      if (item.chain) {
        const hsv = OPS.colorspace.apply(cv, input, { ...defaults('colorspace'), space: 'hsv' }, {});
        temp = hsv;
        input = hsv;
      }
      const applied = applyWithNeeds(item.id, input, item.params, item.b);
      if (!item.chain) return { out: applied.out, ctx: applied.ctx };
      const grayOut = OPS.grayscale.apply(cv, applied.out, defaults('grayscale'), {});
      applied.out.delete();
      return { out: grayOut, ctx: applied.ctx };
    } finally { temp?.delete(); preGray?.delete(); }
  };
  const pyNumberList = (value) => (String(value).match(/-?\d+(?:\.\d+)?/g) || []).map(Number);
  const extractedInfo = (name) => {
    const source = fs.readFileSync(new URL('../src/renderer/filters/registry.js', import.meta.url), 'utf8');
    const match = source.match(new RegExp(`const ${name} = ("(?:\\\\.|[^"\\\\])*");`));
    return match ? JSON.parse(match[1]) : '';
  };
  const resizedInfo = extractedInfo('RESIZED_B_INFO'), colourInfo = extractedInfo('COLOUR_CUT_INFO');
  const manifests = [], appOutputs = [], mismatches = [], stats = {};
  try {
    for (const [index, item] of cases.entries()) {
      const app = runApp(item);
      appOutputs.push(app.out);
      const pythonSteps = [];
      if (item.preGray || item.src.channels() === 1) pythonSteps.push(buildStep('grayscale', {}, 1));
      if (item.chain) pythonSteps.push(buildStep('colorspace', { space: 'hsv' }, 1));
      if (item.passFail) pythonSteps.push(buildStep('quadtree', { threshold: 0, view: 'mean' }, 1));
      const operation = buildStep(item.id, item.params, item.passFail ? 2 : pythonSteps.length + 1);
      if (item.b) operation.src2 = { pipeline: [] };
      pythonSteps.push(operation);
      if (item.chain) pythonSteps.push(buildStep('grayscale', {}, pythonSteps.length + 1));
      if (item.passFail) pythonSteps.push(buildStep('pass_fail', { metric: 'qt_blocks', cond: 'ge', lo: 1 }, 3));
      let code = buildPythonCode(pythonSteps).slice(0, buildPythonCode(pythonSteps).indexOf('if __name__ == "__main__":'));
      if (item.passFail) {
        const before = code;
        code = code.replace('    return img\n', "    print('IP5G_PF', pf_measured, pf_ok)\n    return img\n");
        assert.notEqual(code, before, 'the Python Pass/Fail case marker was inserted');
      } else if (['phase_register', 'scribble_segment', 'isodata_segment'].includes(item.id)) {
        const marker = item.id === 'phase_register' ? "    print('IP5G_VALUES', locals().get('ph5g_cx', 0), locals().get('ph5g_cy', 0), locals().get('ph5g_pk', 0))\n"
          : item.id === 'scribble_segment' ? "    print('IP5G_VALUES', locals().get('sc5g_area', None))\n"
            : "    print('IP5G_VALUES', len(locals().get('iso5g_cent', [])))\n";
        const before = code;
        code = code.replace('    return img\n', marker + '    return img\n');
        assert.notEqual(code, before, `${item.id} Python values marker was inserted`);
      }
      const mats = [item.src, ...(item.b ? [item.b] : [])];
      const inputs = mats.map((mat, inputIndex) => {
        const file = path.join(dir, `i${index}_${inputIndex}.bin`);
        fs.writeFileSync(file, toBgrBytes(mat));
        const pathName = item.b ? (inputIndex ? 'path_b' : 'path_a') : 'path';
        const replacement = `np.fromfile(${pathName}, dtype=np.uint8).reshape((${mat.rows}, ${mat.cols}, 3))`;
        const before = code;
        code = code.replace(`cv2.imread(${pathName})`, replacement);
        assert.notEqual(code, before, `case ${index} replaced cv2.imread(${pathName})`);
        return file;
      });
      const script = path.join(dir, `c${index}.py`), output = path.join(dir, `o${index}.bin`);
      fs.writeFileSync(script, code, 'utf8');
      manifests.push({ index, id: item.id, script, inputs, output, shape: [app.out.rows, app.out.cols, app.out.channels()], info: app.ctx.info || '', values: app.ctx.values?.fields.map((field) => field.value) || [], pf: app.pf || null, phase: item.id === 'phase_register' });
    }
    const mf = path.join(dir, 'manifest.json'), rf = path.join(dir, 'results.json'), driver = path.join(dir, 'driver.py');
    fs.writeFileSync(mf, JSON.stringify(manifests.map(({ script, inputs, output, pf }) => ({ script, inputs, output, pf }))));
    fs.writeFileSync(driver, `import contextlib,io,json,sys\nresults=[]\nfor c in json.loads(sys.stdin.buffer.read()):\n ns={'__name__':'case'}\n log=io.StringIO()\n try:\n  with contextlib.redirect_stdout(log):\n   exec(compile(open(c['script'],encoding='utf8').read(),c['script'],'exec'),ns)\n   result=ns['process'](*c['inputs'])\n  result.tofile(c['output'])\n  results.append({'shape':list(result.shape),'log':log.getvalue(),'pf':c['pf']})\n except Exception as exc:\n  results.append({'error':type(exc).__name__+': '+str(exc),'log':log.getvalue(),'pf':c['pf']})\nwith open(sys.argv[1],'w',encoding='utf8') as f: json.dump(results,f)\n`, 'utf8');
    execFileSync('python', [driver, rf], { input: Buffer.from(fs.readFileSync(mf)), stdio: ['pipe', 'pipe', 'pipe'], maxBuffer: 1 << 25, env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
    const results = JSON.parse(fs.readFileSync(rf, 'utf8'));
    for (let i = 0; i < manifests.length; i++) {
      const m = manifests[i], py = results[i];
      if (py.error) { mismatches.push(`${m.id} case ${i}: Python crashed: ${py.error}`); continue; }
      const bytes = Buffer.from(fs.readFileSync(m.output));
      const expectedShape = m.shape[2] === 1 ? m.shape.slice(0, 2) : m.shape;
      if (JSON.stringify(py.shape) !== JSON.stringify(expectedShape)) mismatches.push(`${m.id} case ${i}: shape ${JSON.stringify(py.shape)} != ${JSON.stringify(expectedShape)}`);
      if (m.pf) {
        const marker = py.log.split(/\r?\n/).find((line) => line.startsWith('IP5G_PF '));
        if (!marker) mismatches.push(`${m.id} case ${i}: Pass/Fail marker missing`);
        else { const [, measured, verdict] = marker.split(/\s+/); if ((verdict === 'True' ? 'OK' : 'NG') !== m.pf || measured !== 'None') mismatches.push(`${m.id} case ${i}: Pass/Fail ${marker} != ${m.pf}/None`); }
        continue;
      }
      if (m.shape[2] === 3) for (let k = 0; k < bytes.length; k += 3) [bytes[k], bytes[k + 2]] = [bytes[k + 2], bytes[k]];
      const expected = Buffer.from(appOutputs[i].data);
      let differing = 0, maxDiff = 0;
      for (let k = 0; k < Math.min(bytes.length, expected.length); k++) { const d = Math.abs(bytes[k] - expected[k]); if (d) differing++; if (d > maxDiff) maxDiff = d; }
      const family = m.phase ? 'phase_register' : m.id;
      const row = stats[family] ||= { cases: 0, maxDiff: 0, differingBytes: 0, totalBytes: 0 };
      row.cases++; row.maxDiff = Math.max(row.maxDiff, maxDiff); row.differingBytes += differing; row.totalBytes += expected.length;
      if (bytes.length !== expected.length) mismatches.push(`${m.id} case ${i}: ${bytes.length} bytes != ${expected.length}`);
      else if (m.phase ? (maxDiff > 1 || differing > expected.length * 0.005) : !bytes.equals(expected)) mismatches.push(`${m.id} case ${i}: max diff ${maxDiff}, ${differing}/${expected.length} differing bytes`);
      const appInfo = m.id === 'inpaint' ? m.info.replace(resizedInfo, '').replace(colourInfo, '') : m.info;
      const markerlessLog = py.log.split(/\r?\n/).filter((line) => !line.startsWith('IP5G')).join('\n');
      const aNums = pyNumberList(appInfo), pNums = pyNumberList(markerlessLog);
      if (m.phase ? aNums.length !== pNums.length || aNums.some((v, j) => Math.abs(v - pNums[j]) > (j < 2 ? 0.01 : 0.01)) : JSON.stringify(aNums) !== JSON.stringify(pNums)) mismatches.push(`${m.id} case ${i}: info numbers ${JSON.stringify(aNums)} != ${JSON.stringify(pNums)}`);
      if (['phase_register', 'scribble_segment', 'isodata_segment'].includes(m.id)) {
        const marker = py.log.split(/\r?\n/).find((line) => line.startsWith('IP5G_VALUES '));
        const actual = marker ? marker.slice('IP5G_VALUES '.length).trim().split(/\s+/).map((v) => v === 'None' ? null : Number(v)) : null;
        if (!marker || actual.length !== m.values.length || actual.some((v, j) => v === null ? m.values[j] !== null : Math.abs(v - m.values[j]) > (m.phase ? 0.01 : 0))) mismatches.push(`${m.id} case ${i}: values ${JSON.stringify(actual)} != ${JSON.stringify(m.values)}`);
      }
    }
    console.log(`ip5g2-T6 parity families ${JSON.stringify(stats)}`);
    assert.equal(mismatches.length, 0, `${mismatches.length} parity mismatch(es):\n${mismatches.join('\n')}`);
  } finally {
    for (const mat of appOutputs) mat.delete();
    for (const mat of [rgb, rgba, gray, tinyH, tinyW, large, largeNoise, largeB, ...phaseSources, ...fixtureSources, ...masks.values()]) mat.delete();
    for (const item of cases) if (item.b && ![rgb, rgba, gray, tinyH, tinyW, large, largeB].includes(item.b) && ![...masks.values()].includes(item.b)) item.b.delete();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('ip5g2-T4-small-shapes: every new operation returns a Mat for empty, narrow, gray, RGB, RGBA and constant inputs', () => {
  const shapes = [
    [0, 0, 1, () => 0], [1, 1, 1, () => 90], [1, 9, 3, (_x, y) => [0, y * 23, 255 - y * 19]],
    [9, 1, 4, (x) => [x * 23, 255 - x * 17, x * 11, 77]], [12, 10, 1, (x, y) => (x * 13 + y * 29) & 255],
    [12, 10, 3, (x, y) => [x * 13 & 255, y * 29 & 255, (x * 7 + y * 11) & 255]],
    [12, 10, 4, (x, y) => [x * 13 & 255, y * 29 & 255, (x * 7 + y * 11) & 255, 83]],
    [8, 8, 3, () => [100, 100, 100]],
  ];
  const cases = [
    ['phase_register', ['aligned', 'overlay', 'diff', 'surface']],
    ['scribble_segment', ['overlay', 'mask', 'distance']],
    ['isodata_segment', ['mean', 'labels']],
    ['fir_design', ['result', 'response', 'kernel']],
  ];
  for (const [width, height, channels, pixel] of shapes) {
    const src = image(width, height, channels, pixel);
    const b = width && height ? image(width, height, 1, (x, y) => (x * 17 + y * 31) & 255) : new cv.Mat();
    const mask = width && height ? image(width, height, 1, (x, y) => width > 2 && x === Math.floor(width / 2) && y === Math.floor(height / 2) ? 255 : 0) : new cv.Mat();
    try {
      for (const [id, views] of cases) for (const view of views) {
        const ctx = id === 'phase_register' ? { src2: b } : {};
        const strokes = width && height ? scribbles(width, height, [[1, 0, 0], [2, width - 1, height - 1]]) : null;
        assertWrapperBalance(cv, `${id}/${view} ${width}x${height}`, (c) => {
          const out = OPS[id].apply(c, src, { ...defaults(id), view, strokes }, ctx);
          assert.ok(out && typeof out.delete === 'function', `${id}/${view} ${width}x${height}`);
          out.delete();
        }, { minCreated: width === 0 || height === 0 || (width === 1 && height === 1) || (id === 'phase_register' && Math.min(width, height) < 8) ? 0 : 1 });
      }
      assertWrapperBalance(cv, `inpaint exemplar ${width}x${height}`, (c) => {
        const out = OPS.inpaint.apply(c, src, { ...defaults('inpaint'), method: 'exemplar', patch: '5', search: 10 }, { src2: mask });
        assert.ok(out && typeof out.delete === 'function', `inpaint exemplar ${width}x${height}`);
        out.delete();
      }, { minCreated: width === 0 || height === 0 || (width === 1 && height === 1) ? 0 : 1 });
    } finally { src.delete(); b.delete(); mask.delete(); }
  }
});

test('ip5g2-T4-wrapper-balance: normal, failure and fallback paths release every OpenCV wrapper', () => {
  const src = image(24, 20, 3, (x, y) => [(x * 19 + y * 3) & 255, (x * 7 + y * 17) & 255, (x * 11 + y * 13) & 255]);
  const b = image(24, 20, 3, (x, y) => [(x * 5 + y * 11 + 31) & 255, (x * 17 + y * 3) & 255, (x * 13 + y * 7 + 91) & 255]);
  const mask = image(24, 20, 1, (x, y) => x >= 11 && x <= 12 && y >= 9 && y <= 10 ? 255 : 0);
  const scrub = scribbles(24, 20, [[1, 4, 10], [2, 20, 10]]);
  try {
    for (const view of ['aligned', 'overlay', 'diff', 'surface']) assertWrapperBalance(cv, `phase ${view}`, (c) => OPS.phase_register.apply(c, src, { ...defaults('phase_register'), view }, { src2: b }).delete());
    for (const view of ['overlay', 'mask', 'distance']) assertWrapperBalance(cv, `scribble ${view}`, (c) => OPS.scribble_segment.apply(c, src, { ...defaults('scribble_segment'), strokes: scrub, view }, {}).delete());
    for (const view of ['mean', 'labels']) assertWrapperBalance(cv, `isodata ${view}`, (c) => OPS.isodata_segment.apply(c, src, { ...defaults('isodata_segment'), view }, {}).delete());
    for (const view of ['result', 'response', 'kernel']) assertWrapperBalance(cv, `FIR ${view}`, (c) => OPS.fir_design.apply(c, src, { ...defaults('fir_design'), size: 5, view }, {}).delete());
    assertWrapperBalance(cv, 'inpaint exemplar', (c) => OPS.inpaint.apply(c, src, { ...defaults('inpaint'), method: 'exemplar', patch: '5', search: 10 }, { src2: mask }).delete());
    assertWrapperBalance(cv, 'inpaint exemplar missing B', (c) => OPS.inpaint.apply(c, src, { ...defaults('inpaint'), method: 'exemplar', patch: '5' }, {}).delete());
    const emptyMask = new cv.Mat();
    try { assertWrapperBalance(cv, 'inpaint exemplar empty B', (c) => OPS.inpaint.apply(c, src, { ...defaults('inpaint'), method: 'exemplar', patch: '5' }, { src2: emptyMask }).delete()); }
    finally { emptyMask.delete(); }
    const mismatchMask = image(7, 5, 1, (x, y) => x === 3 && y === 2 ? 255 : 0);
    try { assertWrapperBalance(cv, 'inpaint exemplar mismatched B', (c) => OPS.inpaint.apply(c, src, { ...defaults('inpaint'), method: 'exemplar', patch: '5', search: 10 }, { src2: mismatchMask }).delete()); }
    finally { mismatchMask.delete(); }
    const tiny = image(4, 4, 1, () => 50), fullMask = image(4, 4, 1, () => 255);
    try { assertWrapperBalance(cv, 'inpaint exemplar fallback', (c) => OPS.inpaint.apply(c, tiny, { ...defaults('inpaint'), method: 'exemplar', patch: '9' }, { src2: fullMask }).delete()); }
    finally { tiny.delete(); fullMask.delete(); }
    const refuseSrc = image(4, 4, 1, () => 50), overMask = image(4, 4, 1, (x, y) => x < 3 && y < 2 ? 255 : 0);
    try { assertWrapperBalance(cv, 'inpaint exemplar refuse', (c) => OPS.inpaint.apply(c, refuseSrc, { ...defaults('inpaint'), method: 'exemplar', patch: '5', search: 10 }, { src2: overMask }).delete()); }
    finally { refuseSrc.delete(); overMask.delete(); }
    const tinyNoSeeds = image(4, 4, 1, () => 50), emptyIso = new cv.Mat(), emptyFir = new cv.Mat();
    try {
      assertWrapperBalance(cv, 'scribble no seeds', (c) => OPS.scribble_segment.apply(c, tinyNoSeeds, { ...defaults('scribble_segment'), strokes: null }, {}).delete());
      assertWrapperBalance(cv, 'isodata empty', (c) => OPS.isodata_segment.apply(c, emptyIso, { ...defaults('isodata_segment') }, {}).delete());
      for (const view of ['result', 'response', 'kernel']) assertWrapperBalance(cv, `FIR empty ${view}`, (c) => OPS.fir_design.apply(c, emptyFir, { ...defaults('fir_design'), view }, {}).delete());
    } finally { tinyNoSeeds.delete(); emptyIso.delete(); emptyFir.delete(); }
    const small = image(7, 7, 1, (x, y) => (x * 31 + y * 7) & 255);
    try { assertWrapperBalance(cv, 'phase failure', (c) => OPS.phase_register.apply(c, small, { ...defaults('phase_register') }, { src2: b }).delete()); }
    finally { small.delete(); }
  } finally { src.delete(); b.delete(); mask.delete(); }
});

test('ip5g2-T4-runPipeline: each registered host dispatches its 5G-2 operation end to end', () => {
  const base = image(24, 20, 3, (x, y) => x >= 10 && x <= 12 && y >= 9 && y <= 11 ? [240, 240, 240] : [20, 20, 20]);
  const maskStep = buildStep('binary_threshold', { thresh: 200 }, 2);
  const jobs = [
    buildStep('phase_register', { window: false, subpixel: false, view: 'aligned' }, 1),
    buildStep('scribble_segment', { strokes: scribbles(24, 20, [[1, 4, 10], [2, 20, 10]]) }, 1),
    buildStep('isodata_segment', { k: 3 }, 1),
    buildStep('fir_design', { size: 5 }, 1),
    buildStep('inpaint', { method: 'exemplar', patch: '5', search: 10 }, 1),
  ];
  jobs[0].src2 = { pipeline: [] };
  jobs[4].src2 = { pipeline: [maskStep] };
  try {
    for (const job of jobs) {
      const result = runPipeline(base, [job], new Map(), new Set([job.nodeId]), new Map());
      assert.ok(result.mat, job.params.mode);
      assert.equal(result.mat.rows > 0, true, job.params.mode);
      assert.ok(result.notes.some((note) => note.nodeId === job.nodeId), job.params.mode);
      result.mat.delete();
    }
  } finally { base.delete(); }
});

test('ip5g2-T6-late-value-app: a later value producer makes qt_blocks Pass/Fail NG after quadtree', () => {
  const base = image(24, 20, 3, (x, y) => [(x * 17 + y * 3) & 255, (x * 7 + y * 19) & 255, (x * 11 + y * 23) & 255]);
  const producers = [
    buildStep('phase_register', { window: false, subpixel: false }, 2),
    buildStep('scribble_segment', { strokes: scribbles(24, 20, [[1, 4, 10], [2, 20, 10]]) }, 2),
    buildStep('isodata_segment', { k: 3, iterations: 2 }, 2),
  ];
  producers[0].src2 = { pipeline: [] };
  try {
    for (const producer of producers) {
      const pass = buildStep('pass_fail', { metric: 'qt_blocks', cond: 'ge', lo: 1 }, 3);
      const steps = [buildStep('quadtree', { threshold: 0, view: 'mean' }, 1), producer, pass];
      const result = runPipeline(base, steps, new Map(), new Set([1, 2, 3]), new Map());
      try {
        const note = result.notes.find((item) => item.nodeId === 3);
        assert.ok(note, `${producer.params.mode} Pass/Fail note`);
        assert.match(note.info, /^NG:/);
      } finally { result.mat?.delete(); }
    }
  } finally { base.delete(); }
});

test('ip5g2-T4-heap-growth: every 5G-2 path stays textured after a large allocation and balances wrappers', () => {
  const width = 2000, height = 1500;
  const dummy = new cv.Mat(height, width, cv.CV_8UC3);
  const src = image(width, height, 3, (x, y) => [
    (Math.imul(x + 3, 73) ^ Math.imul(y + 17, 29)) & 255,
    (Math.imul(x + 11, 41) ^ Math.imul(y + 5, 97)) & 255,
    (Math.imul(x + 7, 113) ^ Math.imul(y + 23, 13)) & 255,
  ]);
  const b = image(width, height, 3, (x, y) => [
    (Math.imul(x + 6, 73) ^ Math.imul(y + 17, 29)) & 255,
    (Math.imul(x + 14, 41) ^ Math.imul(y + 5, 97)) & 255,
    (Math.imul(x + 10, 113) ^ Math.imul(y + 23, 13)) & 255,
  ]);
  let noiseState = 0x13579bdf;
  const isoSource = image(width, height, 3, () => {
    const pixel = [];
    for (let c = 0; c < 3; c++) { noiseState = (Math.imul(noiseState, 1664525) + 1013904223) >>> 0; pixel.push(noiseState >>> 24); }
    return pixel;
  });
  const mask = image(width, height, 1, (x, y) => x >= 998 && x < 1002 && y >= 748 && y < 752 ? 255 : 0);
  const strokes = scribbles(width, height, [[1, 100, 100, 1], [2, 1900, 1400, 1]]);
  const assertTextured = (label, out) => {
    const data = out.data;
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i];
    assert.ok(data.length && sum / data.length > 10, `${label} mean ${sum / Math.max(1, data.length)}`);
  };
  try {
    const jobs = [
      ['phase_register', { view: 'aligned', window: false, subpixel: false }, { src2: b }],
      ['scribble_segment', { view: 'overlay', strokes }, {}],
      ['isodata_segment', { k: 4, iterations: 2 }, {}, isoSource],
      ['fir_design', { view: 'result', size: 5, method: 'sampling' }, {}],
      ['inpaint', { method: 'exemplar', patch: '5', search: 10 }, { src2: mask }],
    ];
    for (const [id, params, ctx, input = src] of jobs) {
      assertWrapperBalance(cv, `${id} heap-growth`, (c) => {
        const localCtx = { ...ctx };
        const out = OPS[id].apply(c, input, { ...defaults(id), ...params }, localCtx);
        assertTextured(id, out);
        if (id === 'isodata_segment') assert.match(localCtx.info, /ลดเหลือ 1 รอบ/);
        out.delete();
      });
    }
  } finally { src.delete(); b.delete(); isoSource.delete(); mask.delete(); dummy.delete(); }
});

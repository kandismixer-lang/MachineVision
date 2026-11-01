import assert from 'node:assert/strict';
import { test } from 'node:test';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import cv from '@techstark/opencv-js';
import { OPS, FILTERS, PASS_FAIL_METRICS, LUCY_WORK_BUDGET, BALL_WORK_BUDGET, BALL_LARGE_WORK_BUDGET, needsOf, defaultParams, passFailVerdict } from '../src/renderer/filters/registry.js';
import { buildPythonCode, PF_VALUE_OPS } from '../src/renderer/codegen.js';
import { OP_TO_FILTER, TOOL_SECTIONS } from '../src/renderer/ui/palette-panel.js';
import { OP_ICON, OP_PRIMARY, OPTION_GROUPS } from '../src/renderer/filters/ui-meta.js';
import { testPattern5G } from '../src/renderer/filters/batch5g.js';
import * as kernels5G from '../src/renderer/filters/batch5g.js';
import { assertWrapperBalance } from './_leak-helpers.mjs';

while (typeof cv.Mat !== 'function') await new Promise((resolve) => setTimeout(resolve, 20));
globalThis.self = globalThis;
if (typeof globalThis.ImageData === 'undefined') globalThis.ImageData = class { constructor(data, width, height) { this.data = data; this.width = width; this.height = height; } };
const { runPipeline } = await import('../src/renderer/cv/cv-worker.js');
let python = false, parity = false;
try {
  execFileSync('python', ['-c', 'pass'], { stdio: 'pipe' });
  python = true;
  if (!process.env.IP5G_NO_PY) { execFileSync('python', ['-c', 'import cv2,numpy'], { stdio: 'pipe' }); parity = true; }
} catch {}

const defaults = (id) => Object.fromEntries(OPS[id].params.map((q) => [q.key, q.default]));
const run = (id, src, p = {}, ctx = {}, c = cv) => OPS[id].apply(c, src, { ...defaults(id), ...p }, ctx);
function image(w, h, fn, channels = 1) {
  const data = Uint8Array.from(Array.from({ length: w * h }, (_, i) => fn(i % w, Math.floor(i / w))).flat());
  return cv.matFromArray(h, w, channels === 1 ? cv.CV_8UC1 : channels === 4 ? cv.CV_8UC4 : cv.CV_8UC3, data);
}
function step(id, p = {}, nodeId = 1) {
  const local = { ...defaults(id), ...p };
  const host = OP_TO_FILTER[id] || id;
  return {
    id: host,
    params: { ...defaultParams(host), mode: id, ...Object.fromEntries(Object.entries(local).map(([key, value]) => [id + '__' + key, value])) },
    nodeId,
  };
}
function runApp(pipeline, base) {
  return runPipeline(base, pipeline, new Map(), new Set(pipeline.map((s) => s.nodeId)), new Map());
}
const fieldsOf = (notes, nodeId) => {
  const note = notes.find((n) => n.nodeId === nodeId && n.values);
  return note ? Object.fromEntries(note.values.fields.map((field) => [field.key, field.value])) : null;
};
const hash = (value) => crypto.createHash('sha256').update(value).digest('hex');
function runPythonExport(code, inputs) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ip5g-focused-python-'));
  try {
    let source = code.slice(0, code.indexOf('if __name__ == "__main__":'));
    const inputFiles = inputs.map((mat, index) => {
      const rgb = new cv.Mat(), bgr = new cv.Mat();
      try {
        if (mat.channels() === 1) cv.cvtColor(mat, rgb, cv.COLOR_GRAY2RGB);
        else if (mat.channels() === 4) cv.cvtColor(mat, rgb, cv.COLOR_RGBA2RGB);
        else mat.copyTo(rgb);
        cv.cvtColor(rgb, bgr, cv.COLOR_RGB2BGR);
        const file = path.join(dir, `i${index}.bin`);
        fs.writeFileSync(file, Buffer.from(bgr.data));
        const arg = inputs.length > 1 ? (index === 0 ? 'path_a' : 'path_b') : 'path';
        const before = source;
        source = source.replace(`cv2.imread(${arg})`, `np.fromfile(${arg}, dtype=np.uint8).reshape((${mat.rows}, ${mat.cols}, 3))`);
        assert.notEqual(source, before, `expected Python imread replacement for ${arg}`);
        return file;
      } finally { rgb.delete(); bgr.delete(); }
    });
    const script = path.join(dir, 'case.py'), output = path.join(dir, 'output.bin'), resultFile = path.join(dir, 'result.json');
    fs.writeFileSync(script, source, 'utf8');
    const driver = path.join(dir, 'driver.py');
    fs.writeFileSync(driver, `import contextlib,io,json,sys\nns={'__name__':'case'}\nlog=io.StringIO()\nwith contextlib.redirect_stdout(log):\n exec(compile(open(sys.argv[1],encoding='utf8').read(),sys.argv[1],'exec'),ns)\n result=ns['process'](*json.loads(sys.argv[2]))\n result.tofile(sys.argv[3])\njson.dump({'shape':list(result.shape),'log':log.getvalue()},open(sys.argv[4],'w'))\n`, 'utf8');
    execFileSync('python', [driver, script, JSON.stringify(inputFiles), output, resultFile], { stdio: 'pipe', env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
    const result = JSON.parse(fs.readFileSync(resultFile, 'utf8'));
    return { ...result, data: Buffer.from(fs.readFileSync(output)) };
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}

test('ip5g-G: 527 clean-tree host variants preserve pixels, info, values and Python bytes', () => {
  const legacyHosts = ['geometry', 'frequency', 'segment_edge', 'analysis_detect', 'image_registration', 'morph_tools'];
  const addedOps = new Set(['test_pattern', 'similarity_transform', 'quadtree', 'contour_lines', 'mtb_register', 'geodesic_distance', 'radon']);
  const splitMorphOps = new Set(['morph_erode', 'morph_dilate', 'morph_open', 'morph_close', 'morph_gradient', 'morph_tophat', 'morph_blackhat']);
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
      for (const host of legacyHosts) {
        const mode = FILTERS[host].params.find((param) => param.key === 'mode');
        for (const id of mode.options.map((option) => option.value)) {
          if (addedOps.has(id) || !OPS[id]) continue;
          const params = OPS[id].params || [];
          const extras = [{}, ...params.filter((param) => param.type === 'select').flatMap((param) => param.options
            .filter((option) => option.value !== param.default)
            .filter((option) => !(splitMorphOps.has(id) && param.key === 'shape' && option.value === 'ball'))
            .filter((option) => !(id === 'pass_fail' && param.key === 'metric' && ['contour_area', 'qt_blocks'].includes(option.value)))
            .map((option) => ({ [param.key]: option.value })))];
          for (const extra of extras) jobs.push({ host, id, params, extra });
        }
      }
      const legacyMorphParams = OPS.morphology.params;
      const legacyOpParam = legacyMorphParams.find((param) => param.key === 'op');
      const legacyOtherSelects = legacyMorphParams.filter((param) => param.key !== 'op' && param.type === 'select');
      for (const opOption of legacyOpParam.options) {
        const morphExtras = [{ op: opOption.value }, ...legacyOtherSelects.flatMap((param) => param.options
          .filter((option) => option.value !== param.default)
          .filter((option) => !(param.key === 'shape' && option.value === 'ball'))
          .map((option) => ({ op: opOption.value, [param.key]: option.value })))];
        for (const extra of morphExtras) jobs.push({ host: 'morphology', id: 'morphology', params: legacyMorphParams, extra });
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
  assert.equal(operationVariants, 527);
  assert.equal(Object.keys(result).length, 2635);
  assert.equal(hash(JSON.stringify(result)), '4b24a9e6cb7bf3ed510f085db1a102c614c97dc9c0efc02352c7bb5c5f572aa9');
});

test('ip5g-0-active-contour: Pass/Fail measures the actual contour area', () => {
  const src = image(21, 19, (x, y) => x >= 6 && x <= 14 && y >= 5 && y <= 13 ? 220 : 30);
  let out = null;
  try {
    const upstream = {};
    out = OPS.active_contour.apply(cv, src, { ...defaults('active_contour'), init: 'otsu', smooth: 0 }, upstream);
    assert.equal(upstream.values.fields.find((field) => field.key === 'area').value, 81);
    assert.equal(PASS_FAIL_METRICS.contour_area.op, 'active_contour');
    assert.equal(PASS_FAIL_METRICS.contour_area.th, 'Active Contour');
    assert.ok(PF_VALUE_OPS.has('active_contour'));
    const verdict = passFailVerdict(
      { metric: 'contour_area', cond: 'range', lo: 81, hi: 81 },
      { op: 'active_contour', values: upstream.values },
    );
    assert.deepEqual([verdict.result, verdict.measured], ['OK', 81]);
    const ng = passFailVerdict(
      { metric: 'contour_area', cond: 'ge', lo: 82 },
      { op: 'active_contour', values: upstream.values },
    );
    assert.deepEqual([ng.result, ng.measured], ['NG', 81]);
  } finally {
    out?.delete();
    src.delete();
  }
});

test('ip5g-0-active-contour: the real worker returns OK and NG from the measured area', () => {
  const src = image(21, 19, (x, y) => x >= 6 && x <= 14 && y >= 5 && y <= 13 ? 220 : 30);
  let ok = null, ng = null;
  try {
    ok = runApp([
      step('active_contour', { init: 'otsu', smooth: 0 }, 1),
      step('pass_fail', { metric: 'contour_area', cond: 'range', lo: 81, hi: 81 }, 2),
    ], src);
    assert.equal(fieldsOf(ok.notes, 1).area, 81);
    assert.deepEqual([fieldsOf(ok.notes, 2).result, fieldsOf(ok.notes, 2).measured], ['OK', 81]);
    ng = runApp([
      step('active_contour', { init: 'otsu', smooth: 0 }, 1),
      step('pass_fail', { metric: 'contour_area', cond: 'ge', lo: 82 }, 2),
    ], src);
    assert.deepEqual([fieldsOf(ng.notes, 2).result, fieldsOf(ng.notes, 2).measured], ['NG', 81]);
  } finally {
    ok?.mat.delete();
    ng?.mat.delete();
    src.delete();
  }
});

test('ip5g-0-active-contour: exported Pass/Fail snapshots area5f', () => {
  const code = buildPythonCode([
    step('active_contour', { init: 'otsu', smooth: 0 }),
    step('pass_fail', { metric: 'contour_area', cond: 'range', lo: 81, hi: 81 }),
  ]);
  assert.match(code, /pf_up = \{'value': area5f\}/);
  assert.match(code, /pf_measured = pf_up\['value'\]/);
});

test('ip5g-parity: separate A/B files match worker inputs, info numbers and measured Pass/Fail values', { skip: !parity }, () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ip5g-parity-'));
  const rgb = image(45, 37, (x, y) => [(x * 13 + y * 3) & 255, (x * 3 + y * 17) & 255, (x * 7 + y * 11) & 255], 3);
  const rgba = image(45, 37, (x, y) => [(x * 13 + y * 3) & 255, (x * 3 + y * 17) & 255, (x * 7 + y * 11) & 255, 255], 4);
  const tinyH = image(1, 9, (_x, y) => [y * 23, y * 11, 255 - y * 19], 3);
  const tinyW = image(9, 1, (x) => [x * 23, 255 - x * 17, x * 11], 3);
  const gray = image(45, 37, (x, y) => (x * 13 + y * 17) & 255, 1);
  const sources = [rgb, rgba, tinyH, tinyW, gray];
  const large = cv.matFromArray(480, 640, cv.CV_8UC3, Uint8Array.from({ length: 640 * 480 * 3 }, (_, i) => (i * 37 + (i >> 8) * 19) & 255));
  const largeB = cv.matFromArray(240, 320, cv.CV_8UC3, Uint8Array.from({ length: 320 * 240 * 3 }, (_, i) => (i * 23 + (i >> 6) * 41 + 71) & 255));
  const wall = image(21, 17, (x, y) => x === 10 || y === 8 ? 0 : 200, 1);
  const black = image(17, 11, () => 0, 1);
  const pfSource = image(32, 24, (x, y) => [(x * 17 + y * 3) & 255, (x * 11 + y * 19) & 255, (x * 7 + y * 23) & 255], 3);
  const pfB = image(19, 13, (x, y) => [(x * 31 + y * 7 + 91) & 255, (x * 3 + y * 29) & 255, (x * 13 + y * 11 + 17) & 255], 3);
  const cases = [];
  const add = (id, src, params = {}, b = null, extra = {}) => cases.push({ id, src, params, b, ...extra });
  for (const src of sources) {
    for (const pattern of ['checkerboard', 'grid', 'dots', 'ramp_h', 'ramp_v', 'color_bars']) {
      add('test_pattern', src, { pattern, square: 8, matlabGray: pattern === 'checkerboard' });
    }
    add('similarity_transform', src, { angle: 33, scale: 1.2, tx: 7.5, ty: -3, interp: 'nearest', output: 'fit' });
    for (const op of ['erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'blackhat']) {
      add('morphology', src, { op, shape: 'ball', ksize: 3, iterations: 1, ballHeight: 7 });
    }
    const b = image(Math.max(1, src.cols - 3), Math.max(1, src.rows - 5), (x, y) => [(x * 11 + y * 7 + 30) & 255, (x * 5 + y * 13 + 80) & 255, (x * 17 + y * 3 + 10) & 255], 3);
    for (const view of ['aligned', 'overlay', 'diff', 'bitmap']) { add('mtb_register', src, { maxShift: '7', exclusion: 1, view }, b); cases.at(-1).ownedB = b; }
  }
  for (const pattern of ['checkerboard', 'grid', 'dots', 'ramp_h', 'ramp_v', 'color_bars']) {
    add('test_pattern', rgb, { pattern, size: '256', square: 8, matlabGray: pattern === 'checkerboard' });
  }
  for (const src of [tinyH, tinyW]) for (const op of ['erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'blackhat']) {
    add('morphology', src, { op, shape: 'ball', ksize: 21, iterations: 1, ballHeight: 7 });
  }
  for (const view of ['blocks', 'mean', 'size']) add('quadtree', rgb, { threshold: 10, maxSize: '32', minSize: '2', view });
  for (const view of ['overlay', 'plot']) add('contour_lines', rgb, { levels: 5, view });
  for (const method of ['similar', 'binary', 'gray']) for (const metric of ['chessboard', 'cityblock', 'quasi']) {
    add('geodesic_distance', rgb, { method, metric, weight: 0.7, seedX: 45, seedY: 52, view: 'distance' });
  }
  add('geodesic_distance', wall, { method: 'binary', metric: 'cityblock', seedX: 50, seedY: 50, view: 'distance' });
  add('geodesic_distance', rgb, { method: 'similar', metric: 'cityblock', weight: 0, seedX: 45, seedY: 52, view: 'segment', thresh: 0.3 });
  for (const thetaStep of ['2', '5']) for (const view of ['sinogram', 'peak']) add('radon', rgb, { thetaStep, view });
  add('radon', black, { thetaStep: '5', view: 'sinogram' });
  add('test_pattern', large, { pattern: 'checkerboard', square: 16 });
  add('morphology', large, { op: 'close', shape: 'ball', ksize: 21, iterations: 5, ballHeight: 10 });
  for (const view of ['blocks', 'mean', 'size']) add('quadtree', large, { threshold: 12, maxSize: '64', minSize: '2', view });
  add('contour_lines', large, { levels: 5, view: 'plot' });
  for (const view of ['aligned', 'overlay', 'diff', 'bitmap']) { add('mtb_register', large, { maxShift: '15', exclusion: 2, view }, largeB); cases.at(-1).ownedB = largeB; }
  add('geodesic_distance', large, { method: 'similar', metric: 'cityblock', weight: 0, view: 'segment', thresh: 0.25 });
  add('radon', large, { thetaStep: '1', view: 'sinogram' });
  add('radon', large, { thetaStep: '1', view: 'peak' });
  const activeSrc = image(21, 19, (x, y) => x >= 6 && x <= 14 && y >= 5 && y <= 13 ? [220, 220, 220] : [30, 30, 30], 3);
  for (const [lo, hi] of [[81, 81], [82, 82]]) add('active_contour', activeSrc, { lo, hi }, null, { passFail: 'active' });
  for (const id of ['mtb_register', 'geodesic_distance', 'radon']) {
    add(id, pfSource, { view: id === 'geodesic_distance' ? 'segment' : id === 'radon' ? 'peak' : 'aligned', maxShift: '7', thetaStep: '5', method: 'similar', metric: 'cityblock', weight: 0, seedX: 50, seedY: 50 }, id === 'mtb_register' ? pfB : null, { passFail: 'late_value' });
    cases.at(-1).ownedB = id === 'mtb_register' ? pfB : null;
  }

  const chainB = image(45, 37, (x, y) => [(x * 7 + y * 19 + 13) & 255, (x * 23 + y * 5) & 255, (x * 3 + y * 11 + 101) & 255], 3);
  add('mtb_register', rgb, { maxShift: '7', exclusion: 1, view: 'aligned' }, chainB, { chainMtbGray: true });
  cases.at(-1).ownedB = chainB;

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
  const applyWorkerNeeds = (id, src, params, b = null) => {
    const local = { ...defaults(id), ...params };
    let input = src, converted = null;
    if (needsOf(OPS[id], local) === 'gray' && src.channels() > 1) {
      converted = new cv.Mat();
      cv.cvtColor(src, converted, src.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY);
      input = converted;
    }
    const ctx = b ? { src2: b } : {};
    const out = OPS[id].apply(cv, input, local, ctx);
    if (converted) converted.delete();
    return { out, ctx };
  };
  const casesForPython = (item) => {
    const grayStep = item.src.channels() === 1 ? [step('grayscale')] : [];
    if (item.chainMtbGray) {
      const registration = step('mtb_register', item.params, 2);
      registration.src2 = { pipeline: [] };
      return [step('colorspace', { space: 'hsv' }, 1), registration, step('grayscale', {}, 3)];
    }
    if (item.passFail === 'active') return [...grayStep, step('active_contour', { init: 'otsu', view: 'mask', smooth: 0 }, 1), step('pass_fail', { metric: 'contour_area', cond: 'range', lo: item.params.lo, hi: item.params.hi }, 2)];
    if (item.passFail === 'late_value') {
      const producer = step(item.id, item.params, 2);
      if (item.b) producer.src2 = { pipeline: [] };
      return [...grayStep, step('quadtree', { threshold: 0, view: 'mean' }, 1), producer, step('pass_fail', { metric: 'qt_blocks', cond: 'ge', lo: 1 }, 3)];
    }
    const operation = step(item.id, item.params);
    if (item.b) operation.src2 = { pipeline: [] };
    return [...grayStep, operation];
  };
  const manifests = [], outputs = [], mismatches = [];
  try {
    for (const [index, item] of cases.entries()) {
      const pf = !!item.passFail;
      let appOut, expectedPf = null, info = null, values = null;
      if (item.passFail === 'active') {
        const active = applyWorkerNeeds('active_contour', item.src, { init: 'otsu', view: 'mask', smooth: 0 });
        try {
          const pfCtx = { upstream: { op: 'active_contour', values: active.ctx.values } }, parameters = { ...defaults('pass_fail'), metric: 'contour_area', cond: 'range', lo: item.params.lo, hi: item.params.hi };
          appOut = OPS.pass_fail.apply(cv, active.out, parameters, pfCtx);
          expectedPf = fieldsOf([{ nodeId: 2, values: pfCtx.values }], 2);
        } finally { active.out.delete(); }
      } else if (item.passFail === 'late_value') {
        const qt = applyWorkerNeeds('quadtree', item.src, { threshold: 0, view: 'mean' });
        try {
          const current = applyWorkerNeeds(item.id, qt.out, item.params, item.b);
          try {
            const pfCtx = { upstream: { op: item.id, values: current.ctx.values } }, parameters = { ...defaults('pass_fail'), metric: 'qt_blocks', cond: 'ge', lo: 1 };
            appOut = OPS.pass_fail.apply(cv, current.out, parameters, pfCtx);
            expectedPf = Object.fromEntries(pfCtx.values.fields.map((field) => [field.key, field.value]));
          } finally { current.out.delete(); }
        } finally { qt.out.delete(); }
      } else if (item.chainMtbGray) {
        const hsv = applyWorkerNeeds('colorspace', item.src, { space: 'hsv' });
        try {
          const registered = applyWorkerNeeds('mtb_register', hsv.out, item.params, item.b);
          try {
            const gray = applyWorkerNeeds('grayscale', registered.out, {});
            appOut = gray.out; info = registered.ctx.info;
          } finally { registered.out.delete(); }
        } finally { hsv.out.delete(); }
      } else {
        const result = applyWorkerNeeds(item.id, item.src, item.params, item.b);
        appOut = result.out; info = result.ctx.info; values = result.ctx.values;
      }
      outputs.push(appOut);
      const pipeline = casesForPython(item);
      const built = buildPythonCode(pipeline);
      let code = built.slice(0, built.indexOf('if __name__ == "__main__":'));
      if (pf) {
        const before = code;
        code = code.replace('    return img\n', "    print('IP5G_PF', pf_measured, pf_ok)\n    return img\n");
        assert.notEqual(code, before, 'inserted Python Pass/Fail measurement marker');
      }
      const mats = [item.src, ...(item.b ? [item.b] : [])];
      const inputFiles = mats.map((mat, inputIndex) => {
        const file = path.join(dir, `i${index}_${inputIndex}.bin`);
        fs.writeFileSync(file, toBgrBytes(mat));
        const arg = item.b ? (inputIndex ? 'path_b' : 'path_a') : 'path';
        const before = code;
        code = code.replace(`cv2.imread(${arg})`, `np.fromfile(${arg}, dtype=np.uint8).reshape((${mat.rows}, ${mat.cols}, 3))`);
        assert.notEqual(code, before, `expected Python imread replacement for ${arg}`);
        return file;
      });
      const script = path.join(dir, `c${index}.py`), output = path.join(dir, `o${index}.bin`);
      fs.writeFileSync(script, code, 'utf8');
      manifests.push({ script, inputs: inputFiles, output, shape: [appOut.rows, appOut.cols, appOut.channels()], pf, expectedPf, info, values, id: item.id });
    }
    const manifestFile = path.join(dir, 'manifest.json'), resultFile = path.join(dir, 'results.json');
    fs.writeFileSync(manifestFile, JSON.stringify(manifests.map(({ script, inputs, output, pf }) => ({ script, inputs, output, pf }))));
    const driver = path.join(dir, 'driver.py');
    fs.writeFileSync(driver, `import contextlib,io,json,sys\nresults=[]\nfor c in json.load(open(sys.argv[1])):\n ns={'__name__':'case'}\n log=io.StringIO()\n with contextlib.redirect_stdout(log):\n  exec(compile(open(c['script'],encoding='utf8').read(),c['script'],'exec'),ns)\n  result=ns['process'](*c['inputs'])\n result.tofile(c['output'])\n results.append({'shape':list(result.shape),'log':log.getvalue(),'pf':c['pf']})\njson.dump(results,open(sys.argv[2],'w'))\n`);
    execFileSync('python', [driver, manifestFile, resultFile], { stdio: 'pipe', maxBuffer: 1 << 25, env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
    const results = JSON.parse(fs.readFileSync(resultFile, 'utf8'));
    const infoNumbers = (text) => (String(text).match(/-?\d+(?:\.\d+)?/g) || []).map(Number);
    for (let i = 0; i < manifests.length; i++) {
      const manifest = manifests[i], result = results[i];
      if (manifest.pf) {
        const marker = result.log.split(/\r?\n/).find((line) => line.startsWith('IP5G_PF '));
        if (!marker) { mismatches.push(`case ${i} Pass/Fail marker missing: ${result.log}`); continue; }
        const [, measuredText, verdictText] = marker.split(/\s+/);
        const measured = measuredText === 'None' ? null : Number(measuredText);
        if (measured !== manifest.expectedPf.measured) mismatches.push(`case ${i} Pass/Fail measured ${measured}, expected ${manifest.expectedPf.measured}`);
        const verdict = verdictText === 'True' ? 'OK' : 'NG';
        if (verdict !== manifest.expectedPf.result) mismatches.push(`case ${i} Pass/Fail verdict ${verdict}, expected ${manifest.expectedPf.result}`);
        continue;
      }
      const data = Buffer.from(fs.readFileSync(manifest.output));
      const expectedShape = manifest.shape[2] === 1 ? manifest.shape.slice(0, 2) : manifest.shape;
      if (JSON.stringify(result.shape) !== JSON.stringify(expectedShape)) mismatches.push(`case ${i} shape ${JSON.stringify(result.shape)}, expected ${JSON.stringify(expectedShape)}`);
      if (manifest.shape[2] === 3) for (let k = 0; k < data.length; k += 3) [data[k], data[k + 2]] = [data[k + 2], data[k]];
      const expectedBytes = Buffer.from(outputs[i].data);
      if (!data.equals(expectedBytes)) {
        let differing = 0, first = -1;
        for (let k = 0; k < Math.min(data.length, expectedBytes.length); k++) if (data[k] !== expectedBytes[k]) { differing++; if (first < 0) first = k; }
        mismatches.push(`case ${i} pixels differ at ${differing} bytes (first ${first}; app ${expectedBytes.length} bytes, Python ${data.length})`);
      }
      const appNumbers = infoNumbers(manifest.info);
      if (manifest.id === 'radon' && manifest.values) {
        const peak = manifest.values.fields.find((field) => field.key === 'peak')?.value;
        if (peak !== null && peak !== undefined) appNumbers.push(Number(Number(peak).toFixed(3)));
      }
      const pythonNumbers = infoNumbers(result.log);
      if (JSON.stringify(appNumbers) !== JSON.stringify(pythonNumbers)) mismatches.push(`case ${i} info numbers ${JSON.stringify(appNumbers)} / ${JSON.stringify(pythonNumbers)}: ${manifest.info} / ${result.log}`);
    }
    assert.equal(mismatches.length, 0, `${mismatches.length} parity mismatch(es):\n${mismatches.join('\n')}`);
  } finally {
    for (const output of outputs) output.delete();
    for (const mat of [...sources, large, largeB, wall, black, pfSource, pfB, activeSrc].filter(Boolean)) mat.delete();
    for (const mat of new Set(cases.map((item) => item.ownedB).filter(Boolean))) if (![largeB, pfB].includes(mat)) mat.delete();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('ip5g-Python: every export compiles with malformed parameters and Pass/Fail after Active Contour', { skip: !python }, () => {
  const jobs = [
    ['test_pattern', { pattern: 'constructor', size: 'constructor', square: 999, matlabGray: 'no' }],
    ['similarity_transform', { angle: NaN, scale: Infinity, tx: -999, ty: '__proto__', interp: 'constructor', output: 'constructor' }],
    ['morphology', { op: 'dilate', shape: 'ball', ksize: 101, iterations: 999, ballHeight: 0 }],
    ['quadtree', { threshold: Infinity, maxSize: 'constructor', minSize: '16', view: 'constructor' }],
    ['contour_lines', { levels: NaN, view: 'constructor' }],
    ['mtb_register', { maxShift: 'constructor', exclusion: Infinity, view: 'constructor' }],
    ['geodesic_distance', { method: 'constructor', metric: 'constructor', weight: Infinity, seedX: -5, seedY: 101, view: 'constructor' }],
    ['radon', { thetaStep: 'constructor', view: 'constructor' }],
  ];
  const codes = jobs.map(([id, params]) => {
    const st = step(id, params);
    if (id === 'mtb_register') st.src2 = { pipeline: [] };
    return buildPythonCode([st]);
  });
  codes.push(buildPythonCode([
    step('active_contour', { init: 'otsu', smooth: 0 }),
    step('pass_fail', { metric: 'contour_area', cond: 'range', lo: 81, hi: 81 }),
  ]));
  execFileSync('python', ['-c', "import sys,json\nfor source in json.loads(sys.stdin.buffer.read()): compile(source,'export','exec')"], {
    input: Buffer.from(JSON.stringify(codes), 'utf8'), env: { ...process.env, PYTHONIOENCODING: 'utf8' },
  });
});

test('ip5g-1-pattern: checkerboard, gray half, ramps, grid, dots and RGB bars match their integer definitions', () => {
  assert.ok(OPS.test_pattern);
  const src = image(4, 4, () => 77);
  try {
    assert.deepEqual(Array.from(testPattern5G(4, 4, 'checkerboard', 1, false).data), [0, 255, 0, 255, 255, 0, 255, 0, 0, 255, 0, 255, 255, 0, 255, 0]);
    assert.deepEqual(Array.from(testPattern5G(4, 4, 'checkerboard', 1, true).data), [0, 255, 0, 179, 255, 0, 179, 0, 0, 255, 0, 179, 255, 0, 179, 0]);
    const check = run('test_pattern', src, { pattern: 'checkerboard', square: 2 });
    try { assert.deepEqual(Array.from(check.data), [0,0,255,255, 0,0,255,255, 255,255,0,0, 255,255,0,0]); } finally { check.delete(); }
    const rampSrc = image(3, 1, () => 0);
    try {
      const ramp = run('test_pattern', rampSrc, { pattern: 'ramp_h' });
      try { assert.deepEqual(Array.from(ramp.data), [0, 128, 255]); } finally { ramp.delete(); }
    } finally { rampSrc.delete(); }
    assert.deepEqual(Array.from(testPattern5G(2, 1, 'ramp_h', 32, false).data), [0, 255]);
    assert.deepEqual(Array.from(testPattern5G(8, 1, 'ramp_h', 32, false).data), [0, 36, 73, 109, 146, 182, 219, 255]);
    const ramp256Src = image(256, 1, () => 0);
    try {
      const ramp = run('test_pattern', ramp256Src, { pattern: 'ramp_h' });
      try { assert.deepEqual(Array.from(ramp.data), Array.from({ length: 256 }, (_, x) => x)); } finally { ramp.delete(); }
    } finally { ramp256Src.delete(); }
    assert.deepEqual(Array.from(testPattern5G(5, 1, 'ramp_h', 32, false).data), [0, 64, 128, 191, 255], 'the exact 127.5 midpoint rounds up');
    assert.equal(testPattern5G(1, 3, 'ramp_h', 32, false).data[0], 0);
    assert.deepEqual(Array.from(testPattern5G(1, 3, 'ramp_v', 32, false).data), [0, 128, 255]);
    const gridSrc = image(4, 3, () => 0);
    try {
      const grid = run('test_pattern', gridSrc, { pattern: 'grid', square: 2 });
      try { assert.deepEqual(Array.from(grid.data), [0,0,0,0, 0,255,0,0, 0,0,0,0]); } finally { grid.delete(); }
    } finally { gridSrc.delete(); }
    const dotSrc = image(8, 8, () => 0);
    try {
      const dots = run('test_pattern', dotSrc, { pattern: 'dots', square: 8 });
      try { assert.equal(Array.from(dots.data).filter((value) => value === 255).length, 13); } finally { dots.delete(); }
    } finally { dotSrc.delete(); }
    const barsSrc = image(8, 1, () => 0);
    const bars = run('test_pattern', barsSrc, { pattern: 'color_bars' });
    try {
      assert.deepEqual(Array.from(bars.data), [255,255,255, 255,255,0, 0,255,255, 0,255,0, 255,0,255, 255,0,0, 0,0,255, 0,0,0]);
      assert.deepEqual([bars.rows, bars.cols, bars.channels()], [1, 8, 3]);
    } finally { bars.delete(); }
    barsSrc.delete();
    const context = {};
    const fixed = run('test_pattern', src, { pattern: 'grid', size: '256' }, context);
    try {
      assert.deepEqual([fixed.rows, fixed.cols, fixed.channels()], [256, 256, 1]);
      assert.match(context.info, /256×256/);
      assert.match(context.info, /ไม่ใช้ค่าพิกเซลของภาพเข้า/);
      assert.equal(context.values, undefined);
    } finally { fixed.delete(); }
  } finally { src.delete(); }
});

test('ip5g-1-schema: Test Pattern follows geometry, has unique icon and beginner Thai label', () => {
  const section = TOOL_SECTIONS.find((item) => item.title === 'GEOMETRY');
  assert.deepEqual(section.ops.slice(-2), ['similarity_transform', 'test_pattern']);
  assert.equal(OP_TO_FILTER.test_pattern, 'geometry');
  assert.deepEqual(FILTERS.geometry.params[0].options.map((item) => item.value), section.ops);
  assert.ok(OP_ICON.test_pattern);
  assert.equal(Object.values(OP_ICON).filter((icon) => icon === OP_ICON.test_pattern).length, 1);
  assert.match(OP_PRIMARY.test_pattern(defaults('test_pattern')), /[ก-๙]/);
  assert.deepEqual(OPS.test_pattern.params.map((item) => item.key), ['pattern', 'size', 'square', 'matlabGray']);
  assert.deepEqual(OPS.test_pattern.params[2].visibleWhen, [{ key: 'pattern', in: ['checkerboard', 'grid', 'dots'] }]);
  assert.deepEqual(OPS.test_pattern.params[3].visibleWhen, [{ key: 'pattern', in: ['checkerboard'] }]);
  const code = buildPythonCode([step('test_pattern', { pattern: 'checkerboard', square: 8, matlabGray: true })]);
  assert.match(code, /yy, xx = np\.indices\(\(h, w\), dtype=np\.int64\)/);
  assert.match(code, /h, w = img\.shape\[:2\]/);
  execFileSync('python', ['-c', 'import ast,sys; ast.parse(sys.stdin.buffer.read().decode("utf-8"))'], { input: Buffer.from(code, 'utf8'), env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
});

test('ip5g-1-pattern: empty input passes through and a heap-grown 2000×1500 image still draws', () => {
  const empty = new cv.Mat(), ctx = {};
  try {
    const out = run('test_pattern', empty, {}, ctx);
    try { assert.notEqual(out, empty); assert.deepEqual([out.rows, out.cols], [0, 0]); assert.equal(ctx.info, 'ภาพว่าง'); } finally { out.delete(); }
  } finally { empty.delete(); }
  const small = image(3, 3, () => [40, 40, 40], 3);
  const large = new cv.Mat(1500, 2000, cv.CV_8UC3);
  try {
    const out = run('test_pattern', large, { pattern: 'checkerboard', square: 32 });
    try {
      assert.deepEqual([out.rows, out.cols, out.channels()], [1500, 2000, 1]);
      assert.ok(Array.from(out.data).some((value) => value > 10));
    } finally { out.delete(); }
    assertWrapperBalance(cv, 'geometry', (c) => { const result = run('test_pattern', large, {}, {}, c); result.delete(); });
  } finally { large.delete(); small.delete(); }
});

test('ip5g-1-pattern: all patterns use x/y axes correctly on non-square and one-pixel dimensions', () => {
  const cases = [[9, 1], [1, 9], [7, 5]];
  for (const [width, height] of cases) {
    for (const pattern of ['checkerboard', 'grid', 'dots', 'ramp_h', 'ramp_v', 'color_bars']) {
      const output = testPattern5G(width, height, pattern, 4, pattern === 'checkerboard');
      assert.deepEqual([output.data.length, output.channels], [width * height * (pattern === 'color_bars' ? 3 : 1), pattern === 'color_bars' ? 3 : 1], `${pattern} ${width}x${height}`);
      if (pattern === 'ramp_h' && width === 9) assert.deepEqual(Array.from(output.data), [0, 32, 64, 96, 128, 159, 191, 223, 255]);
      if (pattern === 'ramp_v' && height === 9) assert.deepEqual(Array.from(output.data), [0, 32, 64, 96, 128, 159, 191, 223, 255]);
      if (pattern === 'color_bars' && width === 1) assert.deepEqual(Array.from(output.data), Array(27).fill(255));
    }
  }
  const code = buildPythonCode([step('test_pattern', { pattern: 'color_bars' })]);
  assert.match(code, /yy, xx = np\.indices\(\(h, w\), dtype=np\.int64\)/);
  assert.match(code, /Test pattern แถบสี 8 สี/);
  const rampCode = buildPythonCode([step('test_pattern', { pattern: 'ramp_h' }), step('test_pattern', { pattern: 'ramp_v' }, 2)]);
  assert.match(rampCode, /510 \* xx \+ w - 1/);
  assert.match(rampCode, /510 \* yy \+ h - 1/);
  const ctx = {}, mat = image(9, 1, () => 0);
  try {
    const out = run('test_pattern', mat, { pattern: 'color_bars' }, ctx);
    try { assert.match(ctx.info, /Test pattern แถบสี 8 สี 9×1/); }
    finally { out.delete(); }
  } finally { mat.delete(); }
});

test('ip5g-3-ball: integer-height ball dilation, erosion and derived operations follow the specified offsets', () => {
  const src = image(11, 11, (x, y) => x === 5 && y === 5 ? 100 : 0);
  try {
    const dilated = run('morphology', src, { op: 'dilate', shape: 'ball', ksize: 5, iterations: 1, ballHeight: 10 }, {});
    try {
      assert.deepEqual([dilated.data[5 * 11 + 5], dilated.data[5 * 11 + 6], dilated.data[5 * 11 + 7], dilated.data[6 * 11 + 7]], [110, 109, 100, 10]);
    } finally { dilated.delete(); }
    for (const op of ['erode', 'open', 'close', 'gradient', 'tophat', 'blackhat']) {
      const constant = image(7, 5, () => 200);
      const ctx = {};
      const out = run('morphology', constant, { op, shape: 'ball', ksize: 5, iterations: 1, ballHeight: 10 }, ctx);
      try {
        if (op === 'erode') assert.ok(Array.from(out.data).every((v) => v === 190), 'edge pixels ignore out-of-image neighbors');
        if (op === 'tophat') assert.ok(Array.from(out.data).every((v) => v === 0));
        assert.ok(ctx.info.startsWith('Ball '));
        assert.ok(ctx.info.includes('2') && ctx.info.includes('10'));
      } finally { out.delete(); constant.delete(); }
    }
  } finally { src.delete(); }
});

test('ip5g-3-ball-schema: ball is last, height is conditional, and non-ball paths keep their prior output', () => {
  const morph = OPS.morphology;
  const shape = morph.params.find((item) => item.key === 'shape');
  const height = morph.params.find((item) => item.key === 'ballHeight');
  assert.equal(shape.options.at(-1).value, 'ball');
  assert.deepEqual(height.visibleWhen, [
    { key: 'op', in: ['erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'blackhat'] },
    { key: 'shape', in: ['ball'] },
  ]);
  assert.equal(morph.params.at(-1).key, 'ballHeight');
  assert.match(shape.options.at(-1).label, /Ball, ไม่แบน — offsetstrel/);
  assert.match(shape.options.at(-1).hint, /rolling ball/);
  assert.match(shape.options.at(-1).hint, /N = 8/);
  assert.match(shape.options.at(-1).hint, /ช้า/);
  const src = image(13, 9, (x, y) => (x * 31 + y * 17) & 255);
  try {
    const oldCtx = {}, newCtx = {};
    const old = morph.apply(cv, src, { op: 'open', shape: 'ellipse', ksize: 5, iterations: 2 }, oldCtx);
    const withUnusedKey = morph.apply(cv, src, { op: 'open', shape: 'ellipse', ksize: 5, iterations: 2, ballHeight: 88 }, newCtx);
    try { assert.deepEqual([Array.from(withUnusedKey.data), newCtx.info], [Array.from(old.data), oldCtx.info]); }
    finally { old.delete(); withUnusedKey.delete(); }
    const ignored = morph.apply(cv, src, { op: 'skeleton', shape: 'ball', ballHeight: 88 }, {});
    try { assert.ok(ignored.data.length > 0); }
    finally { ignored.delete(); }
  } finally { src.delete(); }
});

test('ip5g-3-ball-python: Python branch compiles and does not use OpenCV morphology primitives', () => {
  const code = buildPythonCode([step('morphology', { op: 'dilate', shape: 'ball', ksize: 5, iterations: 2, ballHeight: 12 })]);
  assert.match(code, /ball5g/);
  assert.match(code, /np\.int16/);
  assert.match(code, /if y1_5g <= y0_5g or x1_5g <= x0_5g:\n\s+continue/);
  assert.doesNotMatch(code, /cv2\.(?:erode|dilate|morphologyEx)\(img/);
  execFileSync('python', ['-c', 'import ast,sys; ast.parse(sys.stdin.buffer.read().decode("utf-8"))'], { input: Buffer.from(code, 'utf8'), env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
  assert.equal(kernels5G.ballKernelSize5G(640, 480, 'close', 21, 5, BALL_WORK_BUDGET), 21);
  assert.equal(kernels5G.ballKernelSize5G(1920, 1080, 'close', 21, 5, BALL_LARGE_WORK_BUDGET), 5);
  assert.match(code, /ball5g_large_budget if img\.shape\[0\] \* img\.shape\[1\] > 640 \* 480 else ball5g_small_budget/);
});

test('ip5g-3-ball-budget: keep k21 at 640×480 and report the reduced 1080p kernel', () => {
  for (const [width, height, expectedKernel] of [[640, 480, 21], [1920, 1080, 5]]) {
    const src = image(width, height, () => 0), ctx = {};
    try {
      const out = run('morphology', src, { op: 'close', shape: 'ball', ksize: 21, iterations: 5, ballHeight: 10 }, ctx);
      try {
        assert.equal(out.data.length, width * height);
        if (expectedKernel === 21) assert.doesNotMatch(ctx.info, /kernel เป็น/);
        else assert.match(ctx.info, /kernel เป็น 5 px \(ขอ 21 px\)/);
      } finally { out.delete(); }
    } finally { src.delete(); }
  }
});

test('ip5g-3-ball: 1xN, RGB auto-gray, iteration semantics and wrapper balance', () => {
  const row = image(9, 1, (x) => x === 4 ? 100 : 0);
  try {
    const out = run('morphology', row, { op: 'dilate', shape: 'ball', ksize: 5, iterations: 1, ballHeight: 10 });
    try { assert.deepEqual(Array.from(out.data), [10, 10, 100, 109, 110, 109, 100, 10, 10]); }
    finally { out.delete(); }
  } finally { row.delete(); }
  const color = image(5, 3, (x, y) => [x * 30, y * 40, 0], 3);
  try {
    const result = runApp([step('morphology', { op: 'dilate', shape: 'ball', ksize: 3, iterations: 2, ballHeight: 7 })], color);
    try { assert.equal(result.mat.channels(), 1); assert.ok(Array.from(result.mat.data).some((value) => value > 0)); }
    finally { result.mat.delete(); }
  } finally { color.delete(); }
  const tiny = image(3, 3, () => [20, 40, 60], 3);
  const rgbData = Uint8Array.from({ length: 2000 * 1500 * 3 }, (_, i) => i % 239);
  const large = cv.matFromArray(1500, 2000, cv.CV_8UC3, rgbData);
  try {
    const output = runApp([step('morphology', { op: 'close', shape: 'ball', ksize: 3, iterations: 1, ballHeight: 10 })], large);
    try { assert.deepEqual([output.mat.rows, output.mat.cols, output.mat.channels()], [1500, 2000, 1]); assert.ok(Array.from(output.mat.data).some((value) => value > 10)); }
    finally { output.mat.delete(); }
    assertWrapperBalance(cv, 'morphology ball', (c) => {
      const gray = new c.Mat(); c.cvtColor(large, gray, c.COLOR_RGB2GRAY);
      const out = OPS.morphology.apply(c, gray, { op: 'dilate', shape: 'ball', ksize: 3, iterations: 1, ballHeight: 10 }, {});
      out.delete(); gray.delete();
    });
  } finally { large.delete(); tiny.delete(); }
});

test('ip5g-4-quadtree: split counts, strict threshold, clipped roots, mean rounding and views', () => {
  assert.equal(typeof kernels5G.quadtree5G, 'function');
  const one = Uint8Array.from({ length: 64 }, (_, i) => i === 0 ? 255 : 0);
  const divided = kernels5G.quadtree5G(one, 8, 8, 10, 8, 1, 'blocks');
  assert.equal(divided.blocks, 10);
  assert.deepEqual([...divided.counts].sort((a, b) => b[0] - a[0]), [[4, 3], [2, 3], [1, 4]]);
  assert.equal(divided.channels, 3);
  assert.deepEqual(Array.from(divided.data.slice(0, 3)), [0, 255, 0]);
  const fixed = kernels5G.quadtree5G(new Uint8Array(64).fill(42), 8, 8, 0, 8, 1, 'mean');
  assert.equal(fixed.blocks, 1);
  assert.ok(Array.from(fixed.data).every((v) => v === 42));
  const cropped = kernels5G.quadtree5G(new Uint8Array(60).fill(80), 10, 6, 255, 8, 2, 'mean');
  assert.equal(cropped.blocks, 2);
  const equal = new Uint8Array(64); equal[0] = 10;
  assert.equal(kernels5G.quadtree5G(equal, 8, 8, 10, 8, 1, 'size').blocks, 1, 'range equal to T does not split');
  const strictPython = buildPythonCode([step('quadtree', { threshold: 10, maxSize: '8', minSize: '1', view: 'size' })]);
  assert.match(strictPython, /qt5g_split = qt5g_active & \(qt5g_size > qt5g_min\) & \(\(qt5g_hi - qt5g_lo\) > qt5g_threshold\)/);
  const tie = kernels5G.quadtree5G(Uint8Array.from([0, 1]), 2, 1, 255, 8, 1, 'mean');
  assert.deepEqual(Array.from(tie.data), [1, 1]);
  const onlyLeaf = kernels5G.quadtree5G(Uint8Array.from([80]), 1, 1, 0, 1, 1, 'size');
  assert.deepEqual(Array.from(onlyLeaf.data), [255]);
});

test('ip5g-4-quadtree: worker values feed Pass/Fail and Python export snapshots the count', () => {
  const src = image(8, 8, (x, y) => x === 0 && y === 0 ? 255 : 0);
  let app = null;
  try {
    const upstream = {};
    const out = OPS.quadtree.apply(cv, src, { threshold: 10, maxSize: '8', minSize: '1', view: 'mean' }, upstream);
    try {
      assert.equal(upstream.values.fields.find((field) => field.key === 'qt_blocks').value, 10);
      const verdict = passFailVerdict({ metric: 'qt_blocks', cond: 'eq', lo: 10 }, { op: 'quadtree', values: upstream.values });
      assert.deepEqual([verdict.result, verdict.measured], ['OK', 10]);
    } finally { out.delete(); }
    app = runApp([
      step('quadtree', { threshold: 10, maxSize: '8', minSize: '1', view: 'mean' }, 1),
      step('pass_fail', { metric: 'qt_blocks', cond: 'eq', lo: 10 }, 2),
    ], src);
    assert.deepEqual([fieldsOf(app.notes, 1).qt_blocks, fieldsOf(app.notes, 2).result], [10, 'OK']);
  } finally { app?.mat.delete(); src.delete(); }
  const code = buildPythonCode([
    step('quadtree', { threshold: 10, maxSize: '8', minSize: '1', view: 'mean' }),
    step('pass_fail', { metric: 'qt_blocks', cond: 'eq', lo: 10 }),
  ]);
  assert.match(code, /qt5g_grid\.max\(axis=\(1, 3\)\)/);
  assert.match(code, /qt5g_threshold, qt5g_max, qt5g_min, qt5g_view = 10, 8, 1, "mean"/);
  assert.match(code, /pf_up = \{'value': qt_blocks\}/);
  execFileSync('python', ['-c', 'import ast,sys; ast.parse(sys.stdin.buffer.read().decode("utf-8"))'], { input: Buffer.from(code, 'utf8'), env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
});

test('ip5g-4-quadtree: split children keep the lower rows on tall rectangular images', () => {
  const width = 6, height = 10, data = new Uint8Array(width * height);
  data[9 * width + 5] = 255;
  const result = kernels5G.quadtree5G(data, width, height, 0, 8, 1, 'mean');
  assert.equal(result.blocks, 6);
  assert.equal(result.data[9 * width + 5], 255);
  assert.equal(result.data[0], 0);
});

test('ip5g-4-quadtree-python: a block with range exactly T stays a single leaf', { skip: !parity }, () => {
  const data = new Uint8Array(8 * 8); data[0] = 10;
  const src = cv.matFromArray(8, 8, cv.CV_8UC1, data), params = { threshold: 10, maxSize: '8', minSize: '1', view: 'size' };
  try {
    const ctx = {}, app = run('quadtree', src, params, ctx);
    try {
      assert.equal(ctx.values.fields.find((field) => field.key === 'qt_blocks').value, 1);
      assert.ok(Array.from(app.data).every((value) => value === 255));
      const python = runPythonExport(buildPythonCode([step('quadtree', params)]), [src]);
      assert.deepEqual(python.shape, [8, 8]);
      assert.deepEqual(python.data, Buffer.from(app.data));
    } finally { app.delete(); }
  } finally { src.delete(); }
});

test('ip5g-5G-1-passfail: late value producers keep Pass/Fail on the wrong-source path', () => {
  const source = image(17, 13, (x, y) => (x * 37 + y * 19) & 255);
  const other = image(9, 7, (x, y) => (x * 11 + y * 43) & 255);
  try {
    const qtCtx = {}, qt = run('quadtree', source, { view: 'mean', threshold: 0 }, qtCtx);
    try {
      for (const id of ['mtb_register', 'geodesic_distance', 'radon']) {
        assert.ok(PF_VALUE_OPS.has(id), `${id} updates the latest values source`);
        const ctx = id === 'mtb_register' ? { src2: other } : {};
        const out = run(id, qt, { view: id === 'geodesic_distance' ? 'segment' : id === 'radon' ? 'peak' : 'aligned', maxShift: '7', thetaStep: '5' }, ctx);
        try {
          assert.ok(ctx.values, `${id} publishes values`);
          const verdict = passFailVerdict({ metric: 'qt_blocks', cond: 'ge', lo: 1 }, { op: id, values: ctx.values });
          assert.deepEqual([verdict.result, verdict.measured], ['NG', null]);
        } finally { out.delete(); }
        const producer = step(id, { view: id === 'geodesic_distance' ? 'segment' : id === 'radon' ? 'peak' : 'aligned', maxShift: '7', thetaStep: '5' }, 2);
        if (id === 'mtb_register') producer.src2 = { pipeline: [] };
        const code = buildPythonCode([
          step('quadtree', { view: 'mean', threshold: 0 }, 1),
          producer,
          step('pass_fail', { metric: 'qt_blocks', cond: 'ge', lo: 1 }, 3),
        ]);
        assert.match(code, /pf_measured = None/);
        assert.match(code, /values come from another box/);
      } 
    } finally { qt.delete(); }
  } finally { source.delete(); other.delete(); }
});

test('ip5g-4-quadtree-schema: analysis order, optgroup, metric and empty input', () => {
  const section = TOOL_SECTIONS.find((item) => item.title === 'ANALYSIS & DETECTION');
  assert.ok(section.ops.indexOf('quadtree') === section.ops.indexOf('focus_score') + 1);
  assert.equal(OP_TO_FILTER.quadtree, 'analysis_detect');
  assert.deepEqual(FILTERS.analysis_detect.params[0].options.map((item) => item.value), section.ops);
  assert.ok(OP_ICON.quadtree);
  assert.equal(Object.values(OP_ICON).filter((icon) => icon === OP_ICON.quadtree).length, 1);
  assert.match(OP_PRIMARY.quadtree(defaults('quadtree')), /[\u0e01-\u0e2e]/);
  assert.equal(PASS_FAIL_METRICS.qt_blocks.op, 'quadtree');
  const empty = new cv.Mat(), ctx = {};
  try {
    const out = run('quadtree', empty, {}, ctx);
    try { assert.deepEqual([out.rows, out.cols], [0, 0]); assert.deepEqual(ctx.values.fields[0].value, null); assert.equal(ctx.info, 'ภาพว่าง'); }
    finally { out.delete(); }
  } finally { empty.delete(); }
  const emptyCode = buildPythonCode([step('quadtree')]);
  assert.match(emptyCode, /print\('ภาพว่าง'\)/);
});

test('ip5g-5-contours: color mapping, bright-side pixels, flat input and views', () => {
  assert.equal(typeof kernels5G.contourLines5G, 'function');
  const ramp = Uint8Array.from({ length: 256 }, (_, x) => x);
  const contour = kernels5G.contourLines5G(ramp, 256, 1, 1, 'plot');
  assert.deepEqual(Array.from(contour.data.slice(128 * 3, 128 * 3 + 3)), [128, 255, 128]);
  assert.deepEqual(Array.from(contour.data.slice(127 * 3, 127 * 3 + 3)), [255, 255, 255]);
  const stepData = Uint8Array.from([0, 200, 200, 0]);
  const stepResult = kernels5G.contourLines5G(stepData, 4, 1, 3, 'plot');
  assert.deepEqual(Array.from(stepResult.data.slice(3, 6)), [128, 0, 0]);
  const overlay = kernels5G.contourLines5G(Uint8Array.from([7, 7, 7]), 3, 1, 5, 'overlay');
  assert.equal(overlay.flat, true);
  assert.deepEqual(Array.from(overlay.data), [7, 7, 7, 7, 7, 7, 7, 7, 7]);
  const rampMat = cv.matFromArray(1, 256, cv.CV_8UC1, ramp);
  try {
    const ctx = {}, out = run('contour_lines', rampMat, { levels: 1, view: 'overlay' }, ctx);
    try {
      assert.equal(out.channels(), 3);
      assert.deepEqual(Array.from(out.data.slice(128 * 3, 128 * 3 + 3)), [128, 255, 128]);
      assert.match(ctx.info, /127\.5/);
    } finally { out.delete(); }
  } finally { rampMat.delete(); }
});

test('ip5g-5-contours: empty input reports info and heap growth preserves pixels with balanced wrappers', () => {
  const empty = new cv.Mat(), emptyCtx = {};
  try {
    const out = run('contour_lines', empty, {}, emptyCtx);
    try { assert.equal(emptyCtx.info, 'ภาพว่าง ไม่มีเส้นชั้นความสว่าง'); assert.deepEqual([out.rows, out.cols], [0, 0]); }
    finally { out.delete(); }
  } finally { empty.delete(); }
  assert.match(buildPythonCode([step('contour_lines')]), /ภาพว่าง ไม่มีเส้นชั้นความสว่าง/);
  const source = cv.matFromArray(1500, 2000, cv.CV_8UC1, Uint8Array.from({ length: 3000000 }, (_, i) => (Math.floor(i / 2000) + i % 2000) & 255));
  try {
    for (const id of ['quadtree', 'contour_lines']) {
      const ctx = {}, out = run(id, source, id === 'quadtree' ? { view: 'mean', maxSize: '64', minSize: '2' } : { view: 'plot', levels: 5 }, ctx);
      try { assert.ok(out.data.some((value) => value > 10), `${id} output survived heap growth`); assert.ok(ctx.info); }
      finally { out.delete(); }
      assertWrapperBalance(cv, id, (c) => {
        const result = OPS[id].apply(c, source, id === 'quadtree' ? { ...defaults(id), view: 'mean' } : { ...defaults(id), view: 'plot' }, {});
        result.delete();
      });
    }
  } finally { source.delete(); }
});

test('ip5g-5-contours-schema: placed after line_profile and compiles BGR Python generator', () => {
  const section = TOOL_SECTIONS.find((item) => item.title === 'ANALYSIS & DETECTION');
  assert.equal(section.ops.indexOf('contour_lines'), section.ops.indexOf('line_profile') + 1);
  assert.equal(OP_TO_FILTER.contour_lines, 'analysis_detect');
  assert.deepEqual(FILTERS.analysis_detect.params[0].options.map((item) => item.value), section.ops);
  assert.ok(OP_ICON.contour_lines);
  assert.equal(Object.values(OP_ICON).filter((icon) => icon === OP_ICON.contour_lines).length, 1);
  assert.match(OP_PRIMARY.contour_lines(defaults('contour_lines')), /[\u0e01-\u0e2e]/);
  assert.deepEqual(OPS.contour_lines.params.map((item) => item.key), ['levels', 'view']);
  const code = buildPythonCode([step('contour_lines', { levels: 3, view: 'plot' })]);
  assert.match(code, /range\(1, 4\)/);
  assert.match(code, /out = np\.full\(\(h, w, 3\), 255, dtype=np\.uint8\)/);
  assert.match(code, /bright5g = img >= level5g/);
  assert.match(code, /colors5g = \[\[128,0,0\]/);
  execFileSync('python', ['-c', 'import ast,sys; ast.parse(sys.stdin.buffer.read().decode("utf-8"))'], { input: Buffer.from(code, 'utf8'), env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
});

test('ip5g-6-mtb: pyramid search recovers integer shift, ignores exposure and measures the median bitmap', () => {
  const width = 64, height = 64;
  let state = 123456789;
  const aData = Uint8Array.from({ length: width * height }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state >>> 24;
  });
  const bData = new Uint8Array(aData.length);
  const dx = 5, dy = -3;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const sx = x - dx, sy = y - dy;
    if (sx >= 0 && sx < width && sy >= 0 && sy < height) bData[y * width + x] = aData[sy * width + sx];
  }
  const a = cv.matFromArray(height, width, cv.CV_8UC1, aData), b = cv.matFromArray(height, width, cv.CV_8UC1, bData);
  try {
    const ctx = { src2: b }, out = run('mtb_register', a, { maxShift: '15', exclusion: 4, view: 'aligned' }, ctx);
    try {
      assert.deepEqual([ctx.values.fields[0].value, ctx.values.fields[1].value], [5, -3]);
      assert.deepEqual(Array.from(out.data), Array.from(b.data));
      assert.match(ctx.info, /MTB/);
    } finally { out.delete(); }
    const brightA = Uint8Array.from(aData, (v) => Math.floor(v / 2));
    const bright = cv.matFromArray(height, width, cv.CV_8UC1, brightA);
    try {
      const ctxBright = { src2: b }, result = run('mtb_register', bright, { maxShift: '15', exclusion: 4 }, ctxBright);
      try { assert.deepEqual([ctxBright.values.fields[0].value, ctxBright.values.fields[1].value], [5, -3]); }
      finally { result.delete(); }
    } finally { bright.delete(); }
    const halves = Uint8Array.from([10, 10, 20, 20]);
    const m = kernels5G.mtbRegister5G(halves, halves, halves, 4, 1, 1, 7, 0, 'bitmap');
    assert.deepEqual(Array.from(m.data), [128, 128, 255, 255], 'median is the lower value when exactly half the pixels are 10');
  assert.deepEqual(Array.from(kernels5G.mtbHalf5G(Uint8Array.from([1, 1, 0, 0]), 2, 2)), [1], 'a 0.5 average rounds up in the 2x2 pyramid');
  } finally { a.delete(); b.delete(); }
});

test('ip5g-6-mtb: overlap-normalized mismatch ratio chooses the lower-error shift', () => {
  let state = 123456789;
  const next = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296 < 0.5 ? 0 : 255);
  const width = 32, height = 24;
  const a = Uint8Array.from({ length: width * height }, next), b = Uint8Array.from({ length: width * height }, next);
  const result = kernels5G.mtbRegister5G(a, a, b, width, height, 1, 7, 0, 'aligned');
  assert.deepEqual([result.sx, result.sy], [0, -2]);
  assert.match(OPS.mtb_register.desc, /ลายซ้ำ.*maxShift.*ลด/);
  const registration = step('mtb_register', { maxShift: '7', exclusion: 0 });
  registration.src2 = { pipeline: [] };
  const code = buildPythonCode([registration]);
  assert.match(code, /mtb5g_candidate \* mtb5g_best_used < mtb5g_best_err \* mtb5g_candidate_used/);
});

test('ip5g-6-mtb: A/B views, no-B passthrough, Python source, and registered input wiring', () => {
  assert.equal(typeof kernels5G.mtbRegister5G, 'function');
  assert.equal(OPS.mtb_register.needs, 'any');
  assert.deepEqual(FILTERS.image_registration.params[0].options.map((item) => item.value), ['template_matching', 'ecc_register', 'mtb_register']);
  const color = image(9, 7, (x, y) => [x * 20, y * 30, (x + y) * 12, 255], 4);
  const ref = image(9, 7, (x, y) => (x * 20 + y * 30) & 255);
  try {
    const overlayCtx = { src2: ref }, overlay = run('mtb_register', color, { view: 'overlay', maxShift: '7' }, overlayCtx);
    try { assert.equal(overlay.channels(), 3); assert.equal(overlay.data.length, 9 * 7 * 3); }
    finally { overlay.delete(); }
    const diff = run('mtb_register', color, { view: 'diff', maxShift: '7' }, { src2: ref });
    try { assert.equal(diff.channels(), 1); }
    finally { diff.delete(); }
    const noBctx = {}, noB = run('mtb_register', color, {}, noBctx);
    try { assert.deepEqual([noB.rows, noB.cols, noB.channels()], [7, 9, 4]); assert.match(noBctx.info, /B/); }
    finally { noB.delete(); }
  } finally { color.delete(); ref.delete(); }
  const codeStep = step('mtb_register', { maxShift: '15', exclusion: 4, view: 'aligned' });
  codeStep.src2 = { pipeline: [] };
  const code = buildPythonCode([codeStep]);
  assert.match(code, /cv2\.imread\(path_a\)/);
  assert.match(code, /cv2\.imread\(path_b\)/);
  assert.match(code, /mtb5g_half/);
  assert.match(code, /mtb5g_max_shift, mtb5g_exclusion = 15, 4/);
  assert.match(code, /np\.bincount\(a\.ravel\(\), minlength=256\)/);
  assert.match(code, /mtb5g_sx \*= 2/);
  assert.doesNotMatch(code, /mtb_register:[\s\S]{0,180}bgrChannelOrder: true/);
  execFileSync('python', ['-c', 'import ast,sys; ast.parse(sys.stdin.buffer.read().decode("utf-8"))'], { input: Buffer.from(code, 'utf8'), env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
  const item = TOOL_SECTIONS.find((entry) => entry.title === 'IMAGE REGISTRATION');
  assert.equal(item.ops.at(-1), 'mtb_register');
  assert.equal(OP_TO_FILTER.mtb_register, 'image_registration');
  assert.ok(OP_ICON.mtb_register);
  assert.equal(Object.values(OP_ICON).filter((icon) => icon === OP_ICON.mtb_register).length, 1);
  assert.match(OP_PRIMARY.mtb_register(defaults('mtb_register')), /[\u0e01-\u0e2e]/);
});

test('ip5g-6-mtb: aligned view keeps the A channel order for Grayscale after HSV', () => {
  const registration = step('mtb_register', { view: 'aligned' }, 2);
  registration.src2 = { pipeline: [] };
  const code = buildPythonCode([
    step('colorspace', { space: 'hsv' }, 1),
    registration,
    step('grayscale', {}, 3),
  ]);
  const process = code.slice(code.indexOf('def process'));
  assert.match(process, /img = cv2\.cvtColor\(img, cv2\.COLOR_RGB2GRAY\)/);
});

test('ip5g-6-mtb: heap-grown RGB A/B inputs remain readable and wrappers balance', () => {
  const tiny = image(3, 3, () => [10, 20, 30], 3);
  const data = Uint8Array.from({ length: 2000 * 1500 * 3 }, (_, i) => i % 251);
  const a = cv.matFromArray(1500, 2000, cv.CV_8UC3, data), b = cv.matFromArray(1500, 2000, cv.CV_8UC3, data);
  try {
    const ctx = { src2: b }, out = run('mtb_register', a, { maxShift: '7', view: 'aligned' }, ctx);
    try { assert.deepEqual([out.rows, out.cols, out.channels()], [1500, 2000, 3]); assert.ok(out.data.some((value) => value > 10)); }
    finally { out.delete(); }
    assertWrapperBalance(cv, 'mtb_register', (c) => { const result = OPS.mtb_register.apply(c, a, { maxShift: '7' }, { src2: b }); result.delete(); });
  } finally { a.delete(); b.delete(); tiny.delete(); }
});

test('ip5g-7-geodesic: metrics, U-path distance, maze convergence, blocked seed and reusable sweep', () => {
  assert.equal(typeof kernels5G.geodesicSweep5G, 'function');
  assert.equal(typeof kernels5G.geodesicDistance5G, 'function');
  const open = new Uint8Array(49).fill(200);
  const distances = {};
  for (const metric of ['cityblock', 'chessboard', 'quasi']) {
    const q = kernels5G.geodesicDistance5G(open, 7, 7, 'similar', metric, 0, 3, 3, 'distance', 0.1, 100);
    distances[metric] = q.distance[0];
    assert.equal(q.converged, true);
    assert.deepEqual([q.seedX, q.seedY, q.reached], [3, 3, 49]);
  }
  assert.equal(distances.cityblock, 6);
  assert.equal(distances.chessboard, 3);
  assert.equal(distances.quasi, 3 * Math.SQRT2);

  const width = 7, height = 5, u = new Uint8Array(width * height);
  for (let x = 1; x <= 5; x++) { u[1 * width + x] = 255; u[3 * width + x] = 255; }
  u[2 * width + 5] = 255;
  const around = kernels5G.geodesicDistance5G(u, width, height, 'binary', 'cityblock', 1, 1, 1, 'distance', 0.1, 100);
  assert.equal(around.distance[3 * width + 1], 10, 'the path goes around the U instead of through the black gap');

  const mazeW = 11, mazeH = 9, maze = new Uint8Array(mazeW * mazeH);
  for (const y of [1, 3, 5, 7]) for (let x = 1; x <= 9; x++) maze[y * mazeW + x] = 255;
  maze[2 * mazeW + 9] = maze[4 * mazeW + 1] = maze[6 * mazeW + 9] = 255;
  const mazeCost = (_from, to) => maze[to] > 127 ? 1 : Infinity;
  const one = kernels5G.geodesicSweep5G(mazeW, mazeH, mazeCost, [mazeW + 1], 4, 1);
  const full = kernels5G.geodesicSweep5G(mazeW, mazeH, mazeCost, [mazeW + 1], 4, 100);
  const reference = new Float64Array(maze.length); reference.fill(Infinity); reference[mazeW + 1] = 0;
  const settled = new Uint8Array(maze.length);
  for (let iteration = 0; iteration < maze.length; iteration++) {
    let best = -1;
    for (let i = 0; i < reference.length; i++) if (!settled[i] && Number.isFinite(reference[i]) && (best < 0 || reference[i] < reference[best])) best = i;
    if (best < 0) break;
    settled[best] = 1;
    const x = best % mazeW, y = Math.floor(best / mazeW);
    for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= mazeW || ny < 0 || ny >= mazeH) continue;
      const next = ny * mazeW + nx;
      if (maze[next] > 127 && reference[best] + 1 < reference[next]) reference[next] = reference[best] + 1;
    }
  }
  assert.equal(one.converged, false, 'one set of sweeps is insufficient for this winding corridor');
  assert.equal(full.converged, true);
  assert.deepEqual(Array.from(full.distance), Array.from(reference));
  assert.equal(full.distance[7 * mazeW + 9], 30);

  const blocked = image(5, 5, (x, y) => x === 2 && y === 2 ? 0 : 255);
  try {
    const ctx = {}, out = run('geodesic_distance', blocked, { method: 'binary', seedX: 50, seedY: 50 }, ctx);
    try { assert.deepEqual([out.rows, out.cols, out.channels()], [5, 5, 3]); assert.deepEqual([...new Set(out.data)], [0], `${ctx.info}; values ${[...new Set(out.data)]}`); assert.match(ctx.info, /≤ 127/); assert.equal(ctx.values, undefined); }
    finally { out.delete(); }
  } finally { blocked.delete(); }
  const row = image(9, 1, (x) => x * 20);
  try {
    const out = run('geodesic_distance', row, { method: 'similar', weight: 0, metric: 'cityblock' });
    try { assert.deepEqual([out.rows, out.cols, out.channels()], [1, 9, 3]); assert.deepEqual(Array.from(out.data.slice(4 * 3, 4 * 3 + 3)), [255, 0, 0]); }
    finally { out.delete(); }
  } finally { row.delete(); }
  const formula = image(9, 1, () => 200), seedCtx = {};
  try {
    const out = run('geodesic_distance', formula, { method: 'similar', metric: 'cityblock', weight: 0, seedX: 8, seedY: 0, view: 'segment' }, seedCtx);
    try { assert.match(seedCtx.info, /จุดเริ่ม \(0, 0\)/); }
    finally { out.delete(); }
    const seed = kernels5G.geodesicDistance5G(Uint8Array.from(formula.data), 9, 1, 'similar', 'cityblock', 0, 0, 0, 'distance', 0.1, 100);
    assert.deepEqual([seed.distance[0], seed.distance[1]], [0, 1]);
    const code = buildPythonCode([step('geodesic_distance', { seedX: 8, seedY: 0 })]);
    assert.match(code, /min\(geo5g_w - 1, int\(np\.floor\(8 \/ 100 \* geo5g_w\)\)\)/);
  } finally { formula.delete(); }
  // แกน Y ต้องใช้สูตรเดียวกับ X: seedY 8 บนภาพสูง 9 → แถว floor(0.08·9) = 0 (สูตรเก่าปัดได้ 1)
  const column = image(1, 9, () => 200), seedYCtx = {};
  try {
    const out = run('geodesic_distance', column, { method: 'similar', metric: 'cityblock', weight: 0, seedX: 0, seedY: 8, view: 'segment' }, seedYCtx);
    try { assert.match(seedYCtx.info, /จุดเริ่ม \(0, 0\)/); }
    finally { out.delete(); }
  } finally { column.delete(); }
  const uniform = image(7, 7, () => 200), sweepCtx = {};
  try {
    const out = run('geodesic_distance', uniform, { method: 'similar', metric: 'cityblock', weight: 0 }, sweepCtx);
    try {
      const q = kernels5G.geodesicDistance5G(new Uint8Array(49).fill(200), 7, 7, 'similar', 'cityblock', 0, 3, 3, 'distance', 0.1, 100);
      const numbers = (sweepCtx.info.match(/-?\d+(?:\.\d+)?/g) || []).map(Number);
      assert.equal(numbers.at(-1), q.sweeps, `${sweepCtx.info}; expected ${q.sweeps} sweeps`);
    } finally { out.delete(); }
    const code = buildPythonCode([step('geodesic_distance', { metric: 'cityblock', weight: 0 })]);
    assert.match(code, /geo5g_sets \+= 1/);
    assert.match(code, /geo5g_sets\} sweeps/);
  } finally { uniform.delete(); }
});

test('ip5g-7-geodesic: the small maze reports exactly five sweeps and seeds the Y axis by height', () => {
  const width = 11, height = 9, maze = image(width, height, (x, y) => {
    if ([1, 3, 5, 7].includes(y) && x >= 1 && x <= 9) return 255;
    return (y === 2 && x === 9) || (y === 4 && x === 1) || (y === 6 && x === 9) ? 255 : 0;
  });
  try {
    const ctx = {}, out = run('geodesic_distance', maze, { method: 'binary', metric: 'cityblock', seedX: 10, seedY: 12 }, ctx);
    try { assert.match(ctx.info, /5 รอบกวาด/); }
    finally { out.delete(); }
  } finally { maze.delete(); }

  const tall = image(1, 9, (_x, y) => 20 + y * 20);
  try {
    const ctx = {}, out = run('geodesic_distance', tall, { method: 'similar', metric: 'cityblock', weight: 0, seedX: 0, seedY: 99 }, ctx);
    try { assert.match(ctx.info, /จุดเริ่ม \(0, 8\)/); }
    finally { out.delete(); }
    const code = buildPythonCode([step('geodesic_distance', { method: 'similar', metric: 'cityblock', seedX: 0, seedY: 99 })]);
    assert.match(code, /min\(geo5g_h - 1, int\(np\.floor\(99 \/ 100 \* geo5g_h\)\)\)/);
    assert.match(code, /geo5g_diag = math\.sqrt\(2\.0\) if geo5g_metric == 'quasi' else 1\.0/);
  } finally { tall.delete(); }
});

test('ip5g-7-geodesic-python: Y seed and maze sweep count match the app', { skip: !parity }, () => {
  const width = 11, height = 9, maze = image(width, height, (x, y) => {
    if ([1, 3, 5, 7].includes(y) && x >= 1 && x <= 9) return 255;
    return (y === 2 && x === 9) || (y === 4 && x === 1) || (y === 6 && x === 9) ? 255 : 0;
  });
  const tall = image(1, 9, (_x, y) => 20 + y * 20);
  try {
    const mazePython = runPythonExport(buildPythonCode([step('geodesic_distance', { method: 'binary', metric: 'cityblock', seedX: 10, seedY: 12 })]), [maze]);
    assert.match(mazePython.log, /5 sweeps/);
    const seedPython = runPythonExport(buildPythonCode([step('geodesic_distance', { method: 'similar', metric: 'cityblock', weight: 0, seedX: 0, seedY: 99 })]), [tall]);
    assert.match(seedPython.log, /start \(0, 8\)/);
  } finally { maze.delete(); tall.delete(); }
});

test('ip5g-7-geodesic: registered UI, Python sweeps compile, RGB heap growth and wrapper balance', () => {
  const segment = TOOL_SECTIONS.find((entry) => entry.title === 'SEGMENTATION & EDGE');
  assert.equal(segment.ops.at(-1), 'geodesic_distance');
  assert.deepEqual(FILTERS.segment_edge.params[0].options.map((item) => item.value).at(-1), 'geodesic_distance');
  assert.equal(OPTION_GROUPS['geodesic_distance.method'], undefined);
  assert.deepEqual(OPS.geodesic_distance.params[0].options.map((item) => item.value), ['similar', 'binary', 'gray']);
  assert.ok(OP_ICON.geodesic_distance);
  assert.match(OP_PRIMARY.geodesic_distance(defaults('geodesic_distance')), /[\u0e01-\u0e2e]/);
  const code = buildPythonCode([step('geodesic_distance', { method: 'similar', metric: 'quasi', weight: 0.7, view: 'segment', thresh: 0.2 })]);
  assert.match(code, /geo5g_relax/);
  assert.match(code, /np\.minimum\(geo5g_dst, geo5g_src \+ geo5g_edge, out=geo5g_dst\)/);
  assert.match(code, /geo5g_method, geo5g_metric, geo5g_weight = 'similar', 'quasi', 0\.7/);
  assert.match(code, /geo5g_img = np\.where\(np\.isfinite\(geo5g_dist\) & \(geo5g_dist <= geo5g_thresh \* geo5g_max_dist\), 255, 0\)/);
  assert.match(code, /Geodesic similar, quasi: start/);
  assert.match(code, /\{geo5g_sets\} sweeps, converged=\{geo5g_stable\}/);
  assert.match(code, /geo5g_w \* geo5g_h \* 8/);
  assert.match(code, /limited to fewer sweeps \(max \{geo5g_max_sets\}\)/);
  assert.match(code, /np\.float64/);
  execFileSync('python', ['-c', 'import ast,sys; ast.parse(sys.stdin.buffer.read().decode("utf-8"))'], { input: Buffer.from(code, 'utf8'), env: { ...process.env, PYTHONIOENCODING: 'utf8' } });

  const rgb = cv.matFromArray(1500, 2000, cv.CV_8UC3, Uint8Array.from({ length: 2000 * 1500 * 3 }, (_, i) => i % 251));
  try {
    const output = runApp([step('geodesic_distance', { method: 'similar', weight: 0, metric: 'chessboard' })], rgb);
    try { assert.deepEqual([output.mat.rows, output.mat.cols, output.mat.channels()], [1500, 2000, 3]); assert.ok(output.mat.data.some((value) => value > 0)); }
    finally { output.mat.delete(); }
    assertWrapperBalance(cv, 'geodesic_distance', (c) => {
      const result = OPS.geodesic_distance.apply(c, rgb, { ...defaults('geodesic_distance'), method: 'similar', weight: 0 }, {});
      result.delete();
    });
  } finally { rgb.delete(); }
});

test('ip5g-8-radon: subpixel sums, oriented peaks, output views, adaptive angle budget and black image', () => {
  assert.equal(typeof kernels5G.radon5G, 'function');
  const width = 64, height = 40, cx = Math.floor((width - 1) / 2), cy = Math.floor((height - 1) / 2);
  const horizontal = new Uint8Array(width * height);
  for (let x = 0; x < width; x++) horizontal[(cy - 2) * width + x] = 255;
  const hq = kernels5G.radon5G(horizontal, width, height, 1, 'sinogram', LUCY_WORK_BUDGET);
  assert.equal(hq.peakTheta, 90);
  assert.ok(hq.peakRho > 0);
  for (let theta = 0; theta < hq.width; theta++) {
    let sum = 0;
    for (let rho = 0; rho < hq.height; rho++) sum += hq.values[rho * hq.width + theta];
    assert.ok(Math.abs(sum - 255 * width) <= 1e-9, `projection sum differs at theta column ${theta}: ${sum}`);
  }
  const vertical = new Uint8Array(width * height), xLine = cx + 4;
  for (let y = 0; y < height; y++) vertical[y * width + xLine] = 255;
  const vq = kernels5G.radon5G(vertical, width, height, 1, 'sinogram', LUCY_WORK_BUDGET);
  assert.deepEqual([vq.peakTheta, vq.peakRho], [0, xLine - cx]);

  const single = new Uint8Array(11 * 9); single[4 * 11 + 5] = 4;
  const sq = kernels5G.radon5G(single, 11, 9, 1, 'sinogram', LUCY_WORK_BUDGET);
  for (let theta = 0; theta < sq.width; theta++) {
    let sum = 0;
    for (let rho = 0; rho < sq.height; rho++) sum += sq.values[rho * sq.width + theta];
    assert.ok(Math.abs(sum - 4) <= 1e-12);
  }
  const peak = kernels5G.radon5G(horizontal, width, height, 1, 'peak', LUCY_WORK_BUDGET);
  assert.deepEqual([peak.width, peak.height, peak.channels], [width, height, 3]);
  assert.ok(Array.from(peak.data).some((_, i) => i % 3 === 0 && peak.data[i] === 255 && peak.data[i + 1] === 0 && peak.data[i + 2] === 0));
  const black = kernels5G.radon5G(new Uint8Array(9), 3, 3, 5, 'sinogram', LUCY_WORK_BUDGET);
  assert.equal(black.maxValue, 0);
  assert.ok(black.data.every((value) => value === 0));
  assert.equal(black.peakRho, 0);
  assert.equal(black.values.length, 0);
  assert.equal(kernels5G.radonPeakBand5G(0, 0, 0, 299, 1, 0.5, 149), true, 'a pixel exactly half a bin from the peak line is included');

  const dense = kernels5G.radon5G(new Uint8Array(640 * 480).fill(1), 640, 480, 1, 'sinogram', LUCY_WORK_BUDGET);
  assert.equal(dense.degrees, 2);
  assert.deepEqual([dense.analysisWidth, dense.analysisHeight, dense.factor], [320, 240, 2]);
  for (const [side, factor] of [[512, 1], [1024, 2]]) {
    const boundary = new Uint8Array(side * side);
    boundary.fill(0); boundary[0] = boundary[1] = boundary[side] = boundary[side + 1] = 255;
    const q = kernels5G.radon5G(boundary, side, side, 5, 'sinogram', LUCY_WORK_BUDGET);
    assert.equal(q.factor, factor, `side ${side} uses factor ${factor}`);
    const src = cv.matFromArray(side, side, cv.CV_8UC1, boundary), ctx = {};
    try {
      const out = run('radon', src, { thetaStep: '5' }, ctx);
      try {
        if (factor === 1) assert.doesNotMatch(ctx.info, /ภาพใหญ่ย่อเหลือ/);
        else assert.match(ctx.info, /ภาพใหญ่ย่อเหลือ 512×512 ด้วยค่าเฉลี่ยบล็อก 2×2/);
      } finally { out.delete(); }
    } finally { src.delete(); }
  }
});

test('ip5g-8-radon-python: side 512 stays full size and side 1024 uses factor 2', { skip: !parity }, () => {
  const code = buildPythonCode([step('radon', { thetaStep: '5' })]);
  assert.match(code, /radon5g_factor = max\(1, \(max\(radon5g_original_h, radon5g_original_w\) \+ 511\) \/\/ 512\)/);
  for (const [side, factor] of [[512, 1], [1024, 2]]) {
    const data = new Uint8Array(side * side); data[0] = data[1] = data[side] = data[side + 1] = 255;
    const src = cv.matFromArray(side, side, cv.CV_8UC1, data);
    try {
      const python = runPythonExport(code, [src]);
      if (factor === 1) assert.doesNotMatch(python.log, /reduced to .* by \d+x\d+ block averaging/);
      else assert.match(python.log, /reduced to 512x512 by 2x2 block averaging/);
    } finally { src.delete(); }
  }
});

test('ip5g-8-radon: registered UI, generated Python compiles and 2000x1500 RGB heap growth balances wrappers', () => {
  const frequency = TOOL_SECTIONS.find((entry) => entry.title === 'FREQUENCY DOMAIN');
  assert.equal(frequency.ops.at(-1), 'radon');
  assert.equal(FILTERS.frequency.params[0].options.at(-1).value, 'radon');
  assert.ok(OP_ICON.radon);
  assert.match(OP_PRIMARY.radon(defaults('radon')), /[\u0e01-\u0e2e]/);
  for (const view of ['sinogram', 'peak']) {
    const code = buildPythonCode([step('radon', { thetaStep: '2', view })]);
    assert.match(code, /radon5g_requested = 2/);
    assert.match(code, /radon5g_factor = max\(1, \(max\(radon5g_original_h, radon5g_original_w\) \+ 511\) \/\/ 512\)/);
    assert.match(code, /np\.bincount\(radon5g_indices, weights=radon5g_weights, minlength=radon5g_bins\)/);
    execFileSync('python', ['-c', 'import ast,sys; ast.parse(sys.stdin.buffer.read().decode("utf-8"))'], { input: Buffer.from(code, 'utf8'), env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
  }
  const rgb = cv.matFromArray(1500, 2000, cv.CV_8UC3, Uint8Array.from({ length: 2000 * 1500 * 3 }, (_, i) => i % 241));
  try {
    const output = runApp([step('radon', { thetaStep: '1' })], rgb), appOut = output.mat;
    try { assert.deepEqual([appOut.rows, appOut.cols, appOut.channels()], [629, 45, 1]); assert.ok(appOut.data.some((value) => value > 0)); }
    finally { appOut.delete(); }
    const gray = new cv.Mat(); cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
    try {
      const ctx = {}, out = run('radon', gray, { thetaStep: '1' }, ctx);
      try { assert.deepEqual([out.rows, out.cols, out.channels()], [629, 45, 1]); assert.match(ctx.info, /ทีละ 4°/); assert.match(ctx.info, /ย่อเหลือ 500×375/); }
      finally { out.delete(); }
      assertWrapperBalance(cv, 'radon', (c) => { const result = OPS.radon.apply(c, gray, { thetaStep: '5', view: 'peak' }, {}); result.delete(); });
    }
    finally { gray.delete(); }
  } finally { rgb.delete(); }
});

test('ip5g-2-similarity: identity, 180°, 90° fit, integer translation and nearest scaling match pixel geometry', () => {
  assert.ok(OPS.similarity_transform);
  const src = image(7, 5, (x, y) => (x * 19 + y * 31) & 255);
  try {
    const identity = run('similarity_transform', src, { angle: 0, scale: 1, tx: 0, ty: 0 });
    try { assert.deepEqual([identity.rows, identity.cols, Array.from(identity.data)], [src.rows, src.cols, Array.from(src.data)]); } finally { identity.delete(); }
    const turned = run('similarity_transform', src, { angle: 180, scale: 1, tx: 0, ty: 0, interp: 'nearest' });
    try { assert.deepEqual(Array.from(turned.data), Array.from(src.data).reverse()); } finally { turned.delete(); }
    const fitted = run('similarity_transform', src, { angle: 90, scale: 1, tx: 0, ty: 0, interp: 'nearest', output: 'fit' });
    const expected = new cv.Mat();
    try {
      cv.rotate(src, expected, cv.ROTATE_90_COUNTERCLOCKWISE);
      assert.deepEqual([fitted.rows, fitted.cols, Array.from(fitted.data)], [expected.rows, expected.cols, Array.from(expected.data)]);
    } finally { expected.delete(); fitted.delete(); }
  } finally { src.delete(); }
  const moveSrc = image(5, 2, (x, y) => y * 5 + x + 1);
  try {
    const move = run('similarity_transform', moveSrc, { angle: 0, scale: 1, tx: 40, ty: 0 });
    try { assert.deepEqual(Array.from(move.data), [0,0,1,2,3, 0,0,6,7,8]); } finally { move.delete(); }
  } finally { moveSrc.delete(); }
  const scaledSrc = image(3, 3, (x, y) => y * 3 + x + 1);
  try {
    const scaled = run('similarity_transform', scaledSrc, { angle: 0, scale: 2, tx: 0, ty: 0, interp: 'nearest' });
    try { assert.deepEqual(Array.from(scaled.data), [5,5,6, 5,5,6, 8,8,9]); } finally { scaled.delete(); }
  } finally { scaledSrc.delete(); }
});

test('ip5g-2-sampler: exported bilinear and nearest helper follows half-up byte rounding', () => {
  assert.equal(typeof kernels5G.sampleImage5G, 'function');
  const data = Uint8Array.from([0, 100, 200, 255]);
  assert.equal(kernels5G.sampleImage5G(data, 2, 2, 1, 0.5, 0.5, 'bilinear', 0), 139);
  assert.equal(kernels5G.sampleImage5G(data, 2, 2, 1, 0.75, 0.75, 'nearest', 0), 255);
  assert.equal(kernels5G.sampleImage5G(data, 2, 2, 1, -0.01, 0, 'bilinear', 0), 0);
});

test('ip5g-2-schema: Similarity Transform is placed before Test Pattern and exports its Python branch', () => {
  const section = TOOL_SECTIONS.find((item) => item.title === 'GEOMETRY');
  assert.equal(section.ops.at(-2), 'similarity_transform');
  assert.equal(OP_TO_FILTER.similarity_transform, 'geometry');
  assert.deepEqual(FILTERS.geometry.params[0].options.map((item) => item.value), section.ops);
  assert.ok(OP_ICON.similarity_transform);
  assert.equal(Object.values(OP_ICON).filter((icon) => icon === OP_ICON.similarity_transform).length, 1);
  assert.match(OP_PRIMARY.similarity_transform(defaults('similarity_transform')), /[ก-๙]/);
  assert.deepEqual(OPS.similarity_transform.params.map((item) => item.key), ['angle', 'scale', 'tx', 'ty', 'interp', 'output']);
  const code = buildPythonCode([step('similarity_transform', { angle: 33, scale: 1.7, tx: 12.5, ty: -8.5, output: 'fit' })]);
  assert.match(code, /sim5g_sx = \(sim5g_c \* sim5g_dx - sim5g_s \* sim5g_dy\) \/ sim5g_scale/);
  assert.match(code, /np\.indices\(\(sim5g_out_h, sim5g_out_w\), dtype=np\.float64\)/);
  assert.match(code, /sim5g_angle, sim5g_scale, sim5g_tx_percent, sim5g_ty_percent = 33, 1\.7, 12\.5, -8\.5/);
  assert.match(code, /sim5g_corners =/);
  assert.match(code, /sim5g_fallback/);
  execFileSync('python', ['-c', "import sys; compile(sys.stdin.buffer.read().decode('utf-8'), 'export', 'exec')"], { input: Buffer.from(code, 'utf8'), stdio: 'pipe' });
});

test('ip5g-2-similarity: alpha is dropped, empty input passes through, and oversized fit keeps same dimensions', () => {
  const rgba = image(4, 3, (x, y) => [x * 20, y * 30, x + y, 255], 4);
  try {
    const ctx = {}, out = run('similarity_transform', rgba, { angle: 0, scale: 1, tx: 0, ty: 0 }, ctx);
    try {
      assert.deepEqual([out.rows, out.cols, out.channels()], [3, 4, 3]);
      assert.deepEqual(Array.from(out.data), Array.from(rgba.data).filter((_, i) => i % 4 !== 3));
      assert.match(ctx.info, /M = \[/);
    } finally { out.delete(); }
  } finally { rgba.delete(); }
  const empty = new cv.Mat(), emptyCtx = {};
  try {
    const out = run('similarity_transform', empty, {}, emptyCtx);
    try { assert.deepEqual([out.rows, out.cols], [0, 0]); assert.equal(emptyCtx.info, 'ภาพว่าง'); } finally { out.delete(); }
  } finally { empty.delete(); }
  const large = image(1000, 1000, () => 50), fitCtx = {};
  try {
    const out = run('similarity_transform', large, { angle: 15, scale: 4, output: 'fit' }, fitCtx);
    try {
      assert.deepEqual([out.rows, out.cols], [1000, 1000]);
      assert.match(fitCtx.info, /ผลใหญ่เกินไป จึงใช้ขนาดเท่าเดิม/);
    } finally { out.delete(); }
  } finally { large.delete(); }
});

test('ip5g-2-similarity: 2000×1500 RGB after heap growth remains readable and wrapper balanced', () => {
  const small = image(3, 3, () => [40, 80, 120], 3);
  const data = Uint8Array.from({ length: 2000 * 1500 * 3 }, (_, i) => i % 251);
  const large = cv.matFromArray(1500, 2000, cv.CV_8UC3, data);
  try {
    const out = run('similarity_transform', large, { angle: 0, scale: 1, tx: 0, ty: 0, interp: 'nearest' });
    try {
      assert.deepEqual([out.rows, out.cols, out.channels()], [1500, 2000, 3]);
      assert.deepEqual(Array.from(out.data.slice(0, 24)), Array.from(data.slice(0, 24)));
      assert.ok(Array.from(out.data).some((value) => value > 10));
    } finally { out.delete(); }
    assertWrapperBalance(cv, 'similarity_transform', (c) => { const result = run('similarity_transform', large, { angle: 0, scale: 1, tx: 0, ty: 0 }, {}, c); result.delete(); });
  } finally { large.delete(); small.delete(); }
});

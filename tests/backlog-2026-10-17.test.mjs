// Backlog bug-fix round (2026-10-17) — B1..B12 · WORKLOG: WORKLOG-2026-10-17-backlog-bugs.md
// โครงเดียวกับ tests/ip-batch4B-2026-10-15.test.mjs: โหลด cv จาก @techstark/opencv-js, รัน Python จริงเมื่อมี python+cv2 (BL_NO_PY=1 จำลองเครื่องที่ไม่มี)
// ทุกบั๊กที่มีเทสต์ Python ต้องมีเทสต์ JS ล้วนที่จับได้เช่นกัน
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { OPS, FILTERS, defaultParams, needsOf, LUCY_WORK_BUDGET, lucyBudgetIters } from '../src/renderer/filters/registry.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import { OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.LUT === 'function' && typeof c.dft === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

const readSrc = (rel) => fs.readFileSync(new URL(`../src/renderer/${rel}`, import.meta.url), 'utf8');
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}
function gray(w, h, fn) {
  const d = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d[y * w + x] = fn(x, y);
  return cv.matFromArray(h, w, cv.CV_8UC1, d);
}
function rgb(w, h, fn) {
  const d = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d.set(fn(x, y), (y * w + x) * 3);
  return cv.matFromArray(h, w, cv.CV_8UC3, d);
}
function noiseRGB(w, h, seed, scale = [1, 1, 1]) { const r = lcg(seed); return rgb(w, h, () => scale.map((k) => Math.floor(r() * 256 * k))); }
const code = (pipe) => buildPythonCode(pipe);
// step('gamma', { g: 1.5 }) — op ที่ถูกรวมกลุ่ม (merged filter) ใส่ prefix param ให้เอง เหมือน OPSTEP ในเทสต์ batch 4B
const step = (opId, extra = {}) => {
  const host = OP_TO_FILTER[opId];
  if (!host || host === opId) return { id: opId, params: { ...defaultParams(opId), ...extra } };
  const prefixed = {};
  for (const [k, v] of Object.entries(extra)) prefixed[`${opId}__${k}`] = v;
  const st = { id: host, params: { ...defaultParams(host), mode: opId, ...prefixed } };
  if (FILTERS[host].inputs === 2) st.src2 = { pipeline: [] };
  return st;
};

// ===================================================================
// Python จริง (cv2 / numpy 2) — ข้ามเมื่อไม่มี python/cv2
// ===================================================================
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'backlog-1017-'));
process.on('exit', () => { try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ } });
function pythonAvailable() { try { execFileSync('python', ['-c', 'import cv2, numpy'], { encoding: 'utf8' }); return true; } catch { return false; } }
const HAS_PYTHON = !process.env.BL_NO_PY && pythonAvailable();
const PY_SKIP = { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' };
function swapRB3(buf) { const out = Buffer.from(buf); for (let i = 0; i < out.length; i += 3) { const t = out[i]; out[i] = out[i + 2]; out[i + 2] = t; } return out; }
const PY_DRIVER = `import sys, json, io, contextlib
import numpy as np
c = json.load(open(sys.argv[1]))
ns = {'__name__': 'case'}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(open(c['code'], encoding='utf-8').read(), c['code'], 'exec'), ns)
    r = ns['process'](*c['inps'])
r.tofile(c['out'])
json.dump({'stdout': buf.getvalue(), 'ndim': int(r.ndim), 'shape': [int(v) for v in r.shape]}, open(sys.argv[2], 'w'))
`;
let pyCounter = 0;
// srcs: Mat RGB 3ch (app order) → รัน process() ของโค้ดที่ export โดยแทน cv2.imread ด้วยการอ่านไบต์ตรง ๆ
function runPython(src, srcs, { swap = true } = {}) {
  const id = `${process.pid}_${pyCounter++}`;
  let c = src;
  const [A, B] = srcs;
  const rd = (name, m) => `np.fromfile(${name}, dtype=np.uint8).reshape((${m.rows}, ${m.cols}, 3))`;
  if (srcs.length === 1) c = c.replace('img = cv2.imread(path)', `img = ${rd('path', A)}`);
  else c = c.replace('img = cv2.imread(path_a)', `img = ${rd('path_a', A)}`).replace('img_b = cv2.imread(path_b)', `img_b = ${rd('path_b', B)}`);
  const mainIdx = c.indexOf('if __name__ == "__main__":');
  assert.ok(mainIdx > 0);
  c = c.slice(0, mainIdx);
  const o = { code: path.join(TMP_DIR, `c_${id}.py`), inps: srcs.map((_, k) => path.join(TMP_DIR, `i_${id}_${k}.bin`)), out: path.join(TMP_DIR, `o_${id}.bin`) };
  fs.writeFileSync(o.code, c, 'utf8');
  srcs.forEach((m, k) => fs.writeFileSync(o.inps[k], swapRB3(Buffer.from(m.data))));
  const mp = path.join(TMP_DIR, `m_${id}.json`), rp = path.join(TMP_DIR, `r_${id}.json`), dp = path.join(TMP_DIR, `d_${id}.py`);
  fs.writeFileSync(mp, JSON.stringify(o)); fs.writeFileSync(dp, PY_DRIVER);
  execFileSync('python', [dp, mp, rp], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' }, maxBuffer: 1 << 28 });
  const r = JSON.parse(fs.readFileSync(rp, 'utf8'));
  const raw = fs.readFileSync(o.out);
  return { stdout: r.stdout, shape: r.shape, ndim: r.ndim, bytes: r.ndim === 3 && swap ? swapRB3(raw) : Buffer.from(raw) };
}
function appRunSteps(pipe, src, ctxOut = {}) {
  let cur = src.clone();
  try {
    for (const st of pipe) {
      const f = FILTERS[st.id];
      let input = cur, conv = null;
      if (needsOf(f, st.params) === 'gray' && cur.channels() > 1) { conv = new cv.Mat(); cv.cvtColor(cur, conv, cv.COLOR_RGB2GRAY); input = conv; }
      try { const o = f.apply(cv, input, st.params, ctxOut); cur.delete(); cur = o; } finally { if (conv) conv.delete(); }
    }
    return cur;
  } catch (e) { cur.delete(); throw e; }
}
function countDiff(appMat, pyBytes) {
  const d = appMat.data; assert.equal(pyBytes.length, d.length, 'output size');
  let n = 0, mx = 0;
  for (let i = 0; i < d.length; i++) { const e = Math.abs(d[i] - pyBytes[i]); if (e) n++; if (e > mx) mx = e; }
  return { n, mx };
}

// ===================================================================
// B1 (P-20) multi_otsu_threshold Python: histogram เป็น float64
// ===================================================================
test('bl-B1. multi_otsu_threshold Python casts the calcHist histogram to float64 (float32 accumulation of cumsum is exact only up to 16.7 MP under NumPy 2) — the cast is on the hist line, before cumsum', () => {
  const py = code([step('multi_otsu_threshold')]);
  const lines = py.split('\n');
  const iHist = lines.findIndex((l) => l.includes('hist = cv2.calcHist'));
  const iCum = lines.findIndex((l) => l.includes('P = np.concatenate'));
  assert.ok(iHist > 0 && iCum > iHist);
  assert.match(lines[iHist], /\.flatten\(\)\.astype\(np\.float64\)/);
});

// ===================================================================
// B2 Python gamma: ปัดครึ่งขึ้นเหมือน Math.round ของแอป
// ===================================================================
const GAMMAS = [0.2, 0.5, 0.7, 1, 1.2, 1.5, 2.2, 3];
function appGammaLut(g) {
  const ramp = gray(256, 1, (x) => x);
  const o = OPS.gamma.apply(cv, ramp, { g, curve: 'power' }, {});
  const r = Array.from(o.data); ramp.delete(); o.delete(); return r;
}
// สูตรเดียวกับบรรทัด Python ที่ export (np.minimum(255, np.floor(... + 0.5)) → uint8) เขียนใหม่เป็น JS
const pyGammaLut = (g) => Array.from({ length: 256 }, (_, i) => Math.min(255, Math.floor(Math.pow(i / 255.0, 1.0 / g) * 255 + 0.5)));
test('bl-B2a. gamma: the exported Python LUT formula (floor(x + 0.5)) equals the app LUT for gamma 0.2..3, and the old truncating formula would differ (>100 of 256 values at 1.5)', () => {
  for (const g of GAMMAS) assert.deepEqual(pyGammaLut(g), appGammaLut(g), `gamma ${g}`);
  const trunc = (g) => Array.from({ length: 256 }, (_, i) => Math.trunc(Math.pow(i / 255.0, 1.0 / g) * 255));
  const t = trunc(1.5);
  const bad = appGammaLut(1.5).filter((v, i) => v !== t[i]).length;
  assert.ok(bad > 100, `the old formula differed on ${bad} values — the test must be able to see the bug`);
  const py = code([step('gamma', { g: 1.5 })]);
  assert.ok(py.includes('np.floor(np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]) + 0.5)'), 'rounding line is pinned');
  assert.ok(py.includes('.astype(np.uint8)'));
});
test('bl-B2b. gamma: exported Python == app, byte for byte, for gamma 0.2 / 1.5 / 2.2 / 3 on a colour noise image', PY_SKIP, () => {
  const src = noiseRGB(33, 17, 5);
  try {
    for (const g of [0.2, 1.5, 2.2, 3]) {
      const st = [step('gamma', { g })];
      const app = appRunSteps(st, src);
      try { const py = runPython(code(st), [src]); assert.deepEqual(countDiff(app, py.bytes), { n: 0, mx: 0 }, `gamma ${g}`); } finally { app.delete(); }
    }
  } finally { src.delete(); }
});

// ===================================================================
// B3 Python Gray World: ปัดเศษเหมือน convertTo ของแอป
// ===================================================================
test('bl-B3a. white_balance gray_world Python: means come from the uint8 channels (float64, like cv.mean) and the gain is applied with cv2.convertScaleAbs (rounds, like convertTo) — no float32 / astype truncation', () => {
  const py = code([step('white_balance')]);
  assert.ok(py.includes('b, g, r = cv2.split(img)\n'), 'split on the uint8 image');
  assert.ok(py.includes('r = cv2.convertScaleAbs(r, alpha=gain_r)'));
  assert.ok(py.includes('b = cv2.convertScaleAbs(b, alpha=gain_b)'));
  assert.ok(!py.includes('np.float32'), 'no float32 in the Gray World lines');
  assert.ok(!/np\.clip\(r \* gain_r/.test(py));
});
test('bl-B3b. white_balance gray_world: exported Python == app byte for byte on tinted noise (3 images)', PY_SKIP, () => {
  const imgs = [noiseRGB(41, 23, 7, [1, 0.6, 0.3]), noiseRGB(64, 48, 8, [0.4, 1, 0.9]), noiseRGB(5, 5, 9)];
  try {
    for (const src of imgs) {
      const st = [step('white_balance', { method: 'gray_world' })];
      const app = appRunSteps(st, src);
      try { const py = runPython(code(st), [src]); assert.deepEqual(countDiff(app, py.bytes), { n: 0, mx: 0 }); } finally { app.delete(); }
    }
  } finally { imgs.forEach((m) => m.delete()); }
});

// ===================================================================
// B4 (P-11) Harris / Shi-Tomasi: ย่อภาพวิเคราะห์สัดส่วนสุดโต่งต้องไม่ error
// ===================================================================
const defaultsOf = (id) => Object.fromEntries(OPS[id].params.map((p) => [p.key, p.default]));
const cornerImage = (w, h, t = 12) => gray(w, h, (x, y) => ((((x / t) | 0) + ((y / t) | 0)) & 1 ? 220 : 35));
const cornerRGB = (w, h) => rgb(w, h, (x, y) => { const v = (((x / 12) | 0) + ((y / 12) | 0)) & 1 ? 220 : 35; return [v, v, v]; });
test('bl-B4a. harris_corners / shi_tomasi_corners: extreme aspect (1000x1, 1x1000, 3000x2 — Math.round gave a 0-px analysis side) no longer throw, keep the input size, and a normal image still finds corners', () => {
  for (const id of ['harris_corners', 'shi_tomasi_corners']) {
    for (const [w, h] of [[1000, 1], [1, 1000], [3000, 2], [2, 3000], [1, 1], [600, 450]]) {
      const src = cornerImage(w, h, w === 600 ? 75 : 12);
      const ctx = {};
      let out = null;
      try {
        out = OPS[id].apply(cv, src, { ...defaultsOf(id), ...(w === 600 ? { thresh: 95 } : {}) }, ctx);
        assert.equal(out.cols, w, `${id} ${w}x${h} cols`);
        assert.equal(out.rows, h, `${id} ${w}x${h} rows`);
        assert.equal(out.channels(), 3);
        assert.equal(typeof ctx.info, 'string');
        if (w === 600) assert.match(ctx.info, /[1-9]\d* จุด/, `${id} must still find corners on a normal image`);
      } finally { if (out) out.delete(); src.delete(); }
    }
  }
});
test('bl-B4b. harris_corners / shi_tomasi_corners: the analysis size goes through analysisScale (never 0 px) — source-level pin', () => {
  const reg = readSrc('filters/registry.js');
  for (const id of ['harris_corners', 'shi_tomasi_corners']) {
    const i = reg.indexOf(`  ${id}: {`);
    assert.ok(i > 0);
    const body = reg.slice(i, reg.indexOf('\n  },\n', i));
    assert.ok(body.includes('analysisScale(src.cols, src.rows, maxAnalysisSide)'), id);
    assert.ok(!body.includes('Math.round(src.cols * scale), Math.round(src.rows * scale)'), `${id} must not round to 0`);
  }
});

// ===================================================================
// B5 (P-10) Harris: Python ทำ dilate-then-count เหมือนแอป
// ===================================================================
test('bl-B5a. harris_corners Python does the app\'s convert -> threshold -> 3x3 ellipse dilate -> count-nonzero (no raw "resp > thresh" counting), and the 3000-corner cap like the app', () => {
  const py = code([step('harris_corners', { thresh: 60 })]);
  assert.ok(py.includes('mask8 = cv2.convertScaleAbs(resp)'));
  assert.ok(py.includes('cv2.threshold(mask8, 60 / 100 * 255, 255, cv2.THRESH_BINARY)'));
  assert.ok(py.includes('cv2.dilate(mask8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))'));
  assert.ok(py.includes('ys, xs = np.where(mask8 > 0)'));
  assert.ok(py.includes('if len(xs) > 3000:'));
  assert.ok(!py.includes('np.where(resp > thresh)'));
  // app side keeps the dilate (what learners see)
  assert.ok(/cv\.dilate\(mask8, mask8, kern\)/.test(readSrc('filters/registry.js')));
});
test('bl-B5b. harris_corners: exported Python image == app image (<= 40 px of circle-edge noise) and the printed count == the app info count, at 200x150 / 400x350 / 700x500 (downscale), blockSize 2 and 5, thresh 20 / 60', PY_SKIP, () => {
  for (const [w, h] of [[200, 150], [400, 350], [700, 500]]) {
    const src = cornerRGB(w, h);
    try {
      for (const [blockSize, thresh] of [[2, 60], [5, 20]]) {
        const st = [step('harris_corners', { blockSize, thresh })];
        const ctx = {};
        const app = appRunSteps(st, src, ctx);
        try {
          const py = runPython(code(st), [src]);
          const m = /(?:พบมุม (\d+) จุด|จุดมุมเยอะเกิน)/.exec(ctx.info);
          const pm = /(?:พบมุม (\d+) จุด|จุดมุมเยอะเกิน)/.exec(py.stdout);
          assert.ok(m && pm, `${w}x${h}: info ${ctx.info} / stdout ${py.stdout}`);
          assert.equal(pm[1], m[1], `${w}x${h} b${blockSize} t${thresh}: count`);
          assert.ok(countDiff(app, py.bytes).n <= 40, `${w}x${h} b${blockSize} t${thresh}: pixels differ ${countDiff(app, py.bytes).n}`);
        } finally { app.delete(); }
      }
    } finally { src.delete(); }
  }
});

// ===================================================================
// B6 Lucy-Richardson: งบงาน (พิกเซล × ช่อง × รอบ) — ภาพใหญ่ลดรอบ ไม่ย่อภาพ
// ===================================================================
test('bl-B6a. lucyBudgetIters: at/below 640x480 colour 20 rounds nothing changes (any rounds <= 50 on a smaller image too); above the budget the rounds drop to max(1, floor(budget / (W*H*ch))) and never exceed the request', () => {
  assert.equal(LUCY_WORK_BUDGET, 640 * 480 * 3 * 20);
  assert.equal(lucyBudgetIters(640, 480, 3, 20), 20);
  assert.equal(lucyBudgetIters(1280, 480, 1, 20), 20);   // gray 2x pixels = same work as colour 640x480... still within budget
  assert.equal(lucyBudgetIters(64, 48, 3, 50), 50);
  assert.equal(lucyBudgetIters(641, 480, 3, 20), 19);
  assert.equal(lucyBudgetIters(1920, 1080, 3, 10), 2);   // 18432000 / 6220800 = 2.96 -> 2
  assert.equal(lucyBudgetIters(1920, 1080, 3, 1), 1);
  assert.equal(lucyBudgetIters(8000, 6000, 3, 20), 1);   // floor = 0 -> at least 1
  assert.equal(lucyBudgetIters(1000, 1000, 1, 5), 5);    // never above the request
  for (const [w, h, ch, req] of [[300, 200, 3, 50], [2000, 1500, 1, 20], [641, 481, 3, 20]]) {
    const n = lucyBudgetIters(w, h, ch, req);
    assert.ok(n >= 1 && n <= req);
    assert.ok(n === req ? w * h * ch * req <= LUCY_WORK_BUDGET : w * h * ch * n <= LUCY_WORK_BUDGET || n === 1);
  }
});
test('bl-B6b. deconv_lucy on a big image: runs fewer rounds, says so in Thai in ctx.info, the image keeps its size (no resize) and equals the same op asked for exactly that many rounds; a small image is untouched (no note)', () => {
  const src = gray(1000, 1000, (x, y) => ((x * 7 + y * 13) ^ (x >> 3)) & 255);
  const small = gray(64, 48, (x, y) => (x * 5 + y * 3) & 255);
  let a = null, b = null, c = null;
  try {
    const ctxA = {};
    a = OPS.deconv_lucy.apply(cv, src, { psf: 'motion', psfSize: 5, iterations: 20 }, ctxA);
    assert.equal(a.cols, 1000); assert.equal(a.rows, 1000);
    const n = lucyBudgetIters(1000, 1000, 1, 20); // 18
    assert.equal(n, 18);
    assert.ok(ctxA.info.includes(`Lucy-Richardson ${n} รอบ`), ctxA.info);
    assert.ok(ctxA.info.includes(`ภาพใหญ่ จึงลดเหลือ ${n} รอบ (ขอ 20 รอบ)`), ctxA.info);
    const ctxB = {};
    b = OPS.deconv_lucy.apply(cv, src, { psf: 'motion', psfSize: 5, iterations: n }, ctxB);
    assert.deepEqual(Array.from(a.data), Array.from(b.data), 'reduced run == explicit run with the effective rounds');
    assert.ok(!ctxB.info.includes('ลดเหลือ'), 'asking for exactly the budget is not "reduced"');
    const ctxC = {};
    c = OPS.deconv_lucy.apply(cv, small, { psf: 'motion', psfSize: 5, iterations: 20 }, ctxC);
    assert.ok(!ctxC.info.includes('ลดเหลือ'), ctxC.info);
    assert.ok(ctxC.info.includes('Lucy-Richardson 20 รอบ'));
  } finally { for (const m of [src, small, a, b, c]) if (m) m.delete(); }
});
test('bl-B6c. deconv_lucy codegen imports the budget from the registry (single source of truth, no hardcoded literal) and the Python applies the same rule + prints the same note', () => {
  const gen = readSrc('codegen.js');
  assert.match(gen, /import \{[^}]*LUCY_WORK_BUDGET[^}]*\} from "\.\/filters\/registry\.js"/);
  assert.ok(!gen.includes('18432000'));
  const py = code([step('deconv_lucy', { iterations: 20 })]);
  assert.ok(py.includes(`iters_used = min(iters, max(1, ${LUCY_WORK_BUDGET} // (img.shape[0] * img.shape[1] * lucy_ch)))`));
  assert.ok(py.includes('ภาพใหญ่ จึงลดเหลือ {iters_used} รอบ (ขอ {iters} รอบ)'));
  assert.ok(py.indexOf('iters = iters_used') < py.indexOf('def lucy_channel'), 'rounds are settled before the loop');
});
test('bl-B6d. deconv_lucy: exported Python on a colour image over the budget runs the same number of rounds as the app (same Thai note in stdout / ctx.info) and the image agrees within 2 levels on >= 99% of the bytes', PY_SKIP, () => {
  const src = noiseRGB(700, 500, 3);
  const st = [step('deconv_lucy', { psfSize: 5, iterations: 20 })];
  const ctx = {};
  const app = appRunSteps(st, src, ctx);
  try {
    const n = lucyBudgetIters(700, 500, 3, 20);
    assert.equal(n, 17);
    const note = `ภาพใหญ่ จึงลดเหลือ ${n} รอบ (ขอ 20 รอบ)`;
    assert.ok(ctx.info.includes(note), ctx.info);
    const py = runPython(code(st), [src]);
    assert.ok(py.stdout.includes(note), py.stdout);
    assert.ok(py.stdout.includes(`lucy-richardson: ${n} iterations`), py.stdout);
    const d = countDiff(app, py.bytes);
    assert.ok(d.mx <= 2 && d.n / app.data.length <= 0.01, JSON.stringify(d));
  } finally { app.delete(); src.delete(); }
});

// ===================================================================
// B7 (P-13 + P-17) finiteOr / boolOr กลาง — แอปกับ Python ตรงกันแม้ param เป็น string
// ===================================================================
import { finiteOr, boolOr } from '../src/renderer/filters/registry.js';
test('bl-B7a. finiteOr: numbers pass, numeric strings become numbers, everything else (undefined/null/""/" "/NaN/Infinity/booleans/objects/non-numeric strings) falls back; boolOr: true/false/"true"/"false"/1/0 convert, everything else falls back', () => {
  assert.equal(finiteOr(5, 9), 5); assert.equal(finiteOr(0, 9), 0); assert.equal(finiteOr(-0.3, 9), -0.3);
  assert.equal(finiteOr('0', 9), 0); assert.equal(finiteOr('2.5', 9), 2.5); assert.equal(finiteOr(' 7 ', 9), 7); assert.equal(finiteOr('-1', 9), -1);
  for (const bad of [undefined, null, '', '  ', 'abc', '5px', NaN, Infinity, -Infinity, 'Infinity', true, false, {}, [], [1], () => 1]) assert.equal(finiteOr(bad, 9), 9, String(bad));
  for (const t of [true, 1, 'true']) assert.equal(boolOr(t, false), true, String(t));
  for (const f of [false, 0, 'false']) assert.equal(boolOr(f, true), false, String(f));
  for (const bad of [undefined, null, '', 'yes', 'no', 'TRUE', 'True', 2, -1, NaN, {}, [], '0', '1']) { assert.equal(boolOr(bad, true), true, String(bad)); assert.equal(boolOr(bad, false), false, String(bad)); }
});
test('bl-B7b. source pin: no Number.isFinite(p.x) left in registry.js; codegen has no private finiteOr (imports the registry one); no bare truthiness of the boolean params (invert / expand / useHarris / symmetric / refine / hmRotate / detectShadows) in registry.js, codegen.js, ui-meta.js', () => {
  const reg = readSrc('filters/registry.js'), gen = readSrc('codegen.js'), meta = readSrc('filters/ui-meta.js');
  assert.ok(!/Number\.isFinite\(p\./.test(reg));
  assert.ok(!/function finiteOr\(/.test(gen));
  assert.match(gen, /import \{[^}]*\bfiniteOr\b[^}]*\bboolOr\b[^}]*\} from "\.\/filters\/registry\.js"/);
  assert.ok((reg.match(/export function finiteOr\(/g) || []).length === 1 && (reg.match(/export function boolOr\(/g) || []).length === 1);
  const bare = /\bp\.(invert|expand|useHarris|symmetric|refine|hmRotate|detectShadows)\b(?!,)/;
  for (const [n, src] of [['registry', reg], ['codegen', gen], ['ui-meta', meta]]) {
    const hit = src.split('\n').findIndex((l) => bare.test(l));
    assert.equal(hit, -1, `${n}.js:${hit + 1} bare use of a boolean param`);
  }
});
// [op, key, ค่า string, ค่าจริง, ค่าที่ทำให้ผลต่าง (กันเทสต์ว่างเปล่า)]
const B7_CASES = [
  ['laplacian_edge', 'ksize', '5', 5, 3],
  ['local_std_filter', 'windowSize', '5', 5, 9],
  ['fft_filter', 'cutoff', '10', 10, 30],
  ['orb_keypoints', 'maxFeatures', '12', 12, null], // ภาพเล็กได้จุด ORB น้อยกว่า 12 → เทียบแค่ตรงกัน (ไม่เช็กว่าต่างจาก default)
  ['fast_keypoints', 'maxPoints', '7', 7, 300],
  ['binary_threshold', 'invert', 'false', false, true],
  ['binary_threshold', 'invert', 'true', true, false],
  ['otsu_threshold', 'invert', 'false', false, true],
  ['rotate', 'expand', 'false', false, true],
  ['shi_tomasi_corners', 'useHarris', 'true', true, false],
  ['glcm', 'symmetric', 'false', false, true],
  ['dehaze', 'refine', 'false', false, true],
];
test('bl-B7c. string params ("5" / "false" / "true") give the same app image AND the same exported Python as the real number / boolean, and a different image from the other value (so the old "falls back to default" / "truthy" behaviour would fail)', () => {
  const src = rgbTex();
  try {
    for (const [op, key, str, real, other] of B7_CASES) {
      const run = (v) => { const st = [step(op, { [key]: v })]; return { st, mat: appRunSteps(st, src) }; };
      const a = run(str), b = run(real), c = other === null ? null : run(other);
      try {
        assert.deepEqual(Array.from(a.mat.data), Array.from(b.mat.data), `${op}.${key}="${str}" app image == ${real}`);
        if (c) assert.ok(a.mat.rows !== c.mat.rows || a.mat.cols !== c.mat.cols || Array.from(a.mat.data).some((v, i) => v !== c.mat.data[i]), `${op}.${key}: "${str}" must differ from ${other} (non-vacuous)`);
        assert.equal(code(a.st), code(b.st), `${op}.${key}="${str}" Python text == ${real}`);
      } finally { a.mat.delete(); b.mat.delete(); if (c) c.mat.delete(); }
    }
  } finally { src.delete(); }
});
function rgbTex() { const r = lcg(11); return rgb(48, 36, (x, y) => { const v = (((x / 6) | 0) + ((y / 6) | 0)) & 1 ? 200 : 40; return [v, (v + Math.floor(r() * 40)) & 255, (x * 5 + y * 3) & 255]; }); }
test('bl-B7d. numbers in strings reach the exported Python like the app: a "0" k1 / amount stays 0 (not the default); junk ("abc", null, "") falls back to the default in both', () => {
  const g = (op, key, v) => code([step(op, { [key]: v })]);
  assert.equal(g('undistort', 'k1', '0'), g('undistort', 'k1', 0));
  assert.notEqual(g('undistort', 'k1', '0'), g('undistort', 'k1', -0.3), 'a string "0" is 0, not the default -0.3');
  for (const junk of ['abc', null, '']) assert.equal(g('undistort', 'k1', junk), g('undistort', 'k1', -0.3), `junk ${String(junk)} -> default`);
  const src = rgbTex();
  try {
    const app = (v) => { const m = appRunSteps([step('undistort', { k1: v })], src); const r = Array.from(m.data); m.delete(); return r; };
    assert.deepEqual(app('0'), app(0));
    assert.deepEqual(app(null), app(-0.3));
    assert.notDeepEqual(app('0'), app(-0.3));
  } finally { src.delete(); }
});

// ===================================================================
// B8 (P-21) buildPythonCode2Image: ภาพ B ของแต่ละ step · op สองภาพซ้อนในสาย B · ไม่มีสาย B
// ===================================================================
globalThis.self = globalThis;
if (typeof globalThis.ImageData === 'undefined') {
  globalThis.ImageData = class ImageData { constructor(data, width, height) { this.data = data; this.width = width; this.height = height; } };
}
const { runPipeline } = await import('../src/renderer/cv/cv-worker.js');
const two = (opId, extra, src2Pipeline) => { const st = step(opId, extra); if (src2Pipeline !== undefined) st.src2 = { pipeline: src2Pipeline }; else delete st.src2; return st; };
// แอปจริง: A และ B มาจากภาพต้นฉบับใบเดียวกัน (runPipeline รัน src2 จาก baseRGB เดียวกัน) → ส่งภาพเดียวกันให้ Python ทั้ง path_a และ path_b
function appTwo(pipe, src) {
  const r = runPipeline(src, pipe, new Map(), new Set());
  return r.mat;
}
function pyTwo(pipe, src) { return runPython(code(pipe), [src, src]); }
const count = (s, sub) => s.split(sub).length - 1;
const processBody = (py) => py.slice(py.indexOf('def process'), py.indexOf('if __name__ =='));

test('bl-B8a. two two-image steps: every step starts its B image from the raw B (img_b = img_b_raw.copy()) and runs only ITS OWN B chain, in order next to the step (structure)', () => {
  const pipe = [
    two('imarith', { operation: 'add' }, [step('gaussian_blur', { ksize: 9 })]),
    two('imabsdiff', {}, [step('invert')]),
  ];
  const body = processBody(code(pipe));
  assert.equal(count(body, 'img_b = img_b_raw.copy()'), 2);
  const i1 = body.indexOf('img_b = img_b_raw.copy()'), i2 = body.lastIndexOf('img_b = img_b_raw.copy()');
  const iBlur = body.indexOf('GaussianBlur'), iAdd = body.indexOf('cv2.add('), iInvert = body.indexOf('cv2.bitwise_not(img_b)'), iDiff = body.indexOf('absdiff');
  assert.ok(i1 < iBlur && iBlur < iAdd && iAdd < i2, 'step 1: B chain (blur) -> step 1');
  assert.ok(i2 < iInvert && iInvert < iDiff, 'step 2: B chain (invert) -> step 2');
  assert.ok(!body.slice(i2).includes('GaussianBlur(img_b'), "step 2's B image must not be blurred by step 1's chain");
  assert.ok(body.includes('img_b = cv2.imread(path_b)'), 'raw B is still read as img_b (tests / users patch this line)');
});
test('bl-B8b. two two-image steps: exported Python == app (same source for A and B) byte for byte — the 2nd step used to receive a B that already went through the 1st chain', PY_SKIP, () => {
  const src = noiseRGB(36, 28, 21);
  const pipe = [
    two('imarith', { operation: 'add' }, [step('gaussian_blur', { ksize: 9 })]),
    two('imabsdiff', {}, [step('invert')]),
  ];
  try {
    const app = appTwo(pipe, src);
    try { const d = countDiff(app, pyTwo(pipe, src).bytes); assert.deepEqual(d, { n: 0, mx: 0 }); } finally { app.delete(); }
  } finally { src.delete(); }
});
test('bl-B8c. a two-image op INSIDE the B chain uses its own src2 — STRUCTURE only (byte parity is bl-B8c2): the nested chain is built on its own variable img_b2 and the outer img_b is not touched by it', () => {
  const nested = two('imarith', { operation: 'add' }, [step('gaussian_blur', { ksize: 9 })]);
  const pipe = [two('imabsdiff', {}, [nested])];
  const body = processBody(code(pipe));
  assert.equal(count(body, 'img_b2 = img_b_raw.copy()'), 1, 'nested B image is its own variable');
  assert.ok(/arith_b = img_b2/.test(body), 'nested op reads img_b2 as its second operand');
  assert.ok(body.indexOf('img_b2 = img_b_raw.copy()') < body.indexOf('GaussianBlur(img_b2'), 'nested chain is built on img_b2');
  // img_b (the outer B) must not be blurred: the blur belongs to the nested chain only
  assert.ok(!body.includes('GaussianBlur(img_b,'), body);
});
test('bl-B8c2. nested two-image op: exported Python == app (A = B = same source) byte for byte, incl. a 3-deep nesting', PY_SKIP, () => {
  const src = noiseRGB(30, 22, 5);
  try {
    const nested = two('imarith', { operation: 'add' }, [step('gaussian_blur', { ksize: 9 })]);
    const deep = two('imarith', { operation: 'subtract' }, [two('imabsdiff', {}, [step('invert')])]);
    for (const pipe of [[two('imabsdiff', {}, [nested])], [two('imabsdiff', {}, [deep])]]) {
      const app = appTwo(pipe, src);
      try { assert.deepEqual(countDiff(app, pyTwo(pipe, src).bytes), { n: 0, mx: 0 }); } finally { app.delete(); }
    }
  } finally { src.delete(); }
});
test('bl-B8d. a two-image step WITHOUT any src2 (not connected): the app returns A unchanged, the Python skips the step (no img_b computation, no path_b maths) and returns A; a nested one without src2 is skipped too', PY_SKIP, () => {
  const src = noiseRGB(30, 22, 8);
  try {
    const pipe = [two('imabsdiff', {})];
    const py = code(pipe);
    const body = processBody(py);
    assert.ok(!body.includes('cv2.absdiff') && !body.includes('img_b_raw.copy()'), 'no B computation');
    assert.ok(body.includes('ยังไม่ได้ต่อสาย B'));
    const app = appTwo(pipe, src);
    try {
      assert.deepEqual(Array.from(app.data), Array.from(src.data), 'app returns A');
      assert.deepEqual(countDiff(app, runPython(py, [src, src]).bytes), { n: 0, mx: 0 });
    } finally { app.delete(); }
    // nested without src2 inside a B chain: skipped, the outer op still works on the B chain it has
    const pipe2 = [two('imabsdiff', {}, [two('imabsdiff', {}), step('invert')])];
    const app2 = appTwo(pipe2, src);
    try { assert.deepEqual(countDiff(app2, runPython(code(pipe2), [src, src]).bytes), { n: 0, mx: 0 }); } finally { app2.delete(); }
  } finally { src.delete(); }
});
test('bl-B8e. single two-image step with a normal B chain still works (regression): Python == app byte for byte for imabsdiff with [blur]', PY_SKIP, () => {
  const src = noiseRGB(30, 22, 3);
  const pipe = [step('gaussian_blur', { ksize: 5 }), two('imabsdiff', {}, [step('gaussian_blur', { ksize: 9 })])];
  try {
    const app = appTwo(pipe, src);
    try { assert.deepEqual(countDiff(app, pyTwo(pipe, src).bytes), { n: 0, mx: 0 }); } finally { app.delete(); }
  } finally { src.delete(); }
});

// ===================================================================
// B9 (P-19) cv-worker memo hit: กล่องผลลัพธ์ที่ 2 (ไม่ silent) ที่ใช้ node Motion ตัวเดียวกันต้องได้ข้อความ info + values ของตัวเอง
// ===================================================================
const { computeFrame, sweepNodeState } = await import('../src/renderer/cv/cv-worker.js');
function rgbaBuf(m) { const rgba = new cv.Mat(); cv.cvtColor(m, rgba, cv.COLOR_RGB2RGBA); const b = new Uint8ClampedArray(rgba.data).buffer.slice(0); rgba.delete(); return b; }
test('bl-B9a. one Motion node (mog2) shared by two NON-silent result boxes: the memo hit (box 2) still reports its info text and its own copy of values (before the fix nothing guarded this); a silent preview still adds no note', () => {
  const base = noiseRGB(48, 36, 4);
  const buffer = rgbaBuf(base);
  const nodeId = 987001;
  const mk = () => ({ id: 'mog2', params: { view: 'mask' }, enabled: true, nodeId });
  try {
    const two = computeFrame(48, 36, buffer, [
      { connected: true, displayId: 'a', pipeline: [mk()] },
      { connected: true, displayId: 'b', pipeline: [mk()] },
    ], { isolatedState: true });
    const infos = two.notes.filter((n) => n.id === 'mog2' && n.info);
    const vals = two.notes.filter((n) => n.id === 'mog2' && n.values);
    assert.equal(infos.length, 2, 'both boxes must carry the info text');
    assert.equal(infos[0].info, infos[1].info);
    assert.ok(infos[0].info.length > 0);
    assert.equal(vals.length, 2, 'both boxes must carry the values');
    assert.notEqual(vals[0].values, vals[1].values, 'values must not be one shared object (cloneValues)');
    assert.deepEqual(vals[0].values, vals[1].values);
    assert.equal(two.results.length, 2);
    assert.deepEqual(Array.from(new Uint8Array(two.results[0].buffer)), Array.from(new Uint8Array(two.results[1].buffer)), 'memoised result identical');
    const withSilent = computeFrame(48, 36, buffer, [
      { connected: true, displayId: 'a', pipeline: [mk()] },
      { connected: true, displayId: 'p', silentNotes: true, pipeline: [mk()] },
    ], { isolatedState: true });
    assert.equal(withSilent.notes.filter((n) => n.id === 'mog2' && n.info).length, 1, 'silent preview adds no note');
  } finally { sweepNodeState(new Set()); base.delete(); }
});
test('bl-B9b. source pin: the memo-hit branch of runPipeline copies info and values from the memo entry', () => {
  const w = fs.readFileSync(new URL('../src/renderer/cv/cv-worker.js', import.meta.url), 'utf8');
  assert.ok(w.includes('ctx.info = hit.info;'));
  assert.ok(w.includes('ctx.values = cloneValues(hit.values);'));
});

// ===================================================================
// B12 รวม wrapper GEN.fft_filter.gen เข้าบล็อกเดิม — Python ที่ export ต้อง byte-identical
// ===================================================================
import crypto from 'node:crypto';
const FFT_MATRIX_SHA = '519b59f4babba9978c37cfbe239b57348a341dc37754eb64ab0b6b7f8512672e'; // sha256 ของ JSON เมทริกซ์ด้านล่าง วัดบนโค้ดก่อน refactor (wrapper ยังอยู่)
function fftMatrix() {
  const out = {};
  const shapes = ['gaussian', 'ideal', 'butterworth', 'constructor', undefined, 'x', 'Butterworth'];
  const types = ['low', 'high', undefined, 'band'];
  const cutoffs = [0, 5, 30, 64.5, undefined, 'abc', '12', -3, NaN, null];
  const orders = [undefined, 1, 2, 3, 10, 11, 0, -2, 2.6, 'x', '4', NaN];
  for (const shape of shapes) for (const type of types) for (const cutoff of cutoffs) for (const order of orders) {
    const params = { ...defaultParams('frequency'), mode: 'fft_filter' };
    const set = (k, v) => { if (v !== undefined) params[`fft_filter__${k}`] = v; };
    set('shape', shape); set('type', type); set('cutoff', cutoff); set('order', order);
    out[JSON.stringify([shape ?? null, type ?? null, String(cutoff), String(order)])] = buildPythonCode([{ id: 'frequency', params }]);
  }
  for (const mode of ['fft_magnitude', 'bandpass_notch', 'dct_jpeg']) out['mode:' + mode] = buildPythonCode([{ id: 'frequency', params: { ...defaultParams('frequency'), mode } }]);
  return out;
}
test('bl-B12a. fft_filter exported Python is byte-identical to the pre-merge output for 3363 parameter combinations (shape x type x cutoff x order, incl. junk / undefined / strings) + the other frequency modes (sha256 measured before the refactor)', () => {
  const m = fftMatrix();
  assert.equal(Object.keys(m).length, 3363);
  assert.equal(crypto.createHash('sha256').update(JSON.stringify(m)).digest('hex'), FFT_MATRIX_SHA);
  // butterworth really differs from gaussian (the digest is not vacuous)
  const bw = m[JSON.stringify(['butterworth', 'low', '30', '2'])], ga = m[JSON.stringify(['gaussian', 'low', '30', '2'])];
  assert.ok(bw.includes('Butterworth ลำดับ n=2') && bw.includes('(1 / (1 + (d / 30) ** 4))') && !ga.includes('Butterworth'));
});
test('bl-B12b. the wrapper is gone: GEN.fft_filter.gen is defined inside the block (no post-hoc reassignment of GEN.fft_filter.gen, no fftFilterBase) and its own body carries the Butterworth line', () => {
  const gen = readSrc('codegen.js');
  assert.ok(!gen.includes('GEN.fft_filter.gen = '));
  assert.ok(!gen.includes('fftFilterBase'));
  const i = gen.indexOf('\n  fft_filter: { needs'), j = gen.indexOf('\n  // ---- Image Arithmetic (Tier 2, 18a) ----', i);
  assert.ok(i > 0 && j > i);
  assert.ok(gen.slice(i, j).includes('Butterworth ลำดับ n='));
});

// ===================================================================
// รอบแก้ตาม review
// ===================================================================
test('bl-R1a. a GRAY op inside the src2 of a nested two-image op (imabsdiff <- imarith <- otsu, and 2 more shapes): Python == app byte for byte (the BGR2GRAY line of a nested B chain must use that chain own variable)', PY_SKIP, () => {
  const src = noiseRGB(40, 30, 9);
  const pipes = [
    [two('imabsdiff', {}, [two('imarith', { operation: 'add' }, [step('otsu_threshold')])])],
    [two('imabsdiff', {}, [two('imarith', { operation: 'add' }, [step('gaussian_blur', { ksize: 5 }), step('binary_threshold', { thresh: 100 })]), step('invert')])],
    [step('median_blur', { ksize: 3 }), two('imarith', { operation: 'subtract' }, [step('otsu_threshold')]), two('imabsdiff', {}, [two('imabsdiff', {}, [step('invert')])])],
  ];
  try {
    for (const pipe of pipes) {
      const app = appTwo(pipe, src);
      try { assert.deepEqual(countDiff(app, pyTwo(pipe, src).bytes), { n: 0, mx: 0 }); } finally { app.delete(); }
    }
  } finally { src.delete(); }
});
test('bl-R1b. Lucy on a GRAY image over the budget (1 channel, 1100x1000, 30 rounds asked): the rounds in the Python stdout == the rounds in the app info (Python must count 1 channel, not 3)', PY_SKIP, () => {
  const src = noiseRGB(1100, 1000, 4);
  const pipe = [step('grayscale'), step('deconv_lucy', { iterations: 30 })];
  const ctx = {};
  const app = appRunSteps(pipe, src, ctx);
  try {
    const n = lucyBudgetIters(1100, 1000, 1, 30);
    assert.ok(n < 30 && n > 1);
    const re = /ลดเหลือ (\d+) รอบ \(ขอ 30 รอบ\)/;
    const m = re.exec(ctx.info), pm = re.exec(runPython(code(pipe), [src], { swap: false }).stdout);
    assert.ok(m && pm, ctx.info);
    assert.equal(Number(m[1]), n);
    assert.equal(pm[1], m[1]);
  } finally { app.delete(); src.delete(); }
});
test('bl-R1c. Lucy on a small image: the Python stdout has NO reduce note (asking for exactly what fits is not a reduction)', PY_SKIP, () => {
  const src = noiseRGB(60, 50, 4);
  try {
    for (const it of [50, 1]) assert.ok(!runPython(code([step('deconv_lucy', { iterations: it })]), [src]).stdout.includes('ลดเหลือ'));
  } finally { src.delete(); }
});

// hough_circles / hough_lines: ย่อภาพวิเคราะห์สัดส่วนสุดโต่งต้องไม่ error (เหมือน B4)
test('bl-R3a. hough_circles / hough_lines: extreme aspect (3000x2, 2x3000, 1000x1, 1x1) no longer throw and keep the input size; a normal 700x500 image still runs', () => {
  for (const id of ['hough_circles', 'hough_lines']) {
    for (const [w, h] of [[3000, 2], [2, 3000], [1000, 1], [1, 1], [700, 500]]) {
      const src = gray(w, h, (x, y) => ((((x - 350) ** 2 + (y - 250) ** 2) < 80 ** 2 && ((x - 350) ** 2 + (y - 250) ** 2) > 70 ** 2) || y === 100 ? 255 : 20));
      const ctx = {};
      let out = null;
      try {
        out = OPS[id].apply(cv, src, defaultsOf(id), ctx);
        assert.equal(out.cols, w, `${id} ${w}x${h}`);
        assert.equal(out.rows, h);
        assert.equal(typeof ctx.info, 'string');
      } finally { if (out) out.delete(); src.delete(); }
    }
  }
});
test('bl-R3b. hough_circles / hough_lines: the analysis size goes through analysisScale — source pin (the Python side has no downscale, so no max(1) is needed there)', () => {
  const reg = readSrc('filters/registry.js');
  for (const id of ['hough_circles', 'hough_lines']) {
    const i = reg.indexOf(`  ${id}: {`);
    const body = reg.slice(i, reg.indexOf('\n  },\n', i));
    assert.ok(body.includes('analysisScale(src.cols, src.rows, maxAnalysisSide)'), id);
    assert.ok(!body.includes('Math.round(src.cols * scale), Math.round(src.rows * scale)'), id);
  }
  const py = code([step('hough_circles'), step('hough_lines')]);
  assert.ok(!py.includes('cv2.resize'), 'Python hough runs at full size (known, unchanged)');
});

// step ที่ปิดไว้ (enabled === false): แอปข้าม (cv-worker runPipeline) — Python ต้องข้ามเหมือนกัน ทั้งสายหลักและสาย B
const off = (st) => ({ ...st, enabled: false });
test('bl-R7a. a disabled step is skipped by the exported Python exactly as the app does: single-image pipeline (structure + byte parity); a lone disabled two-image step does not switch the script to the 2-image form', PY_SKIP, () => {
  const src = noiseRGB(30, 22, 6);
  try {
    const pipe = [step('gaussian_blur', { ksize: 5 }), off(step('invert')), off(step('otsu_threshold')), step('gamma', { g: 2 })];
    const py = code(pipe);
    assert.ok(!py.includes('bitwise_not') && !py.includes('THRESH_OTSU'), 'disabled steps are not generated');
    assert.ok(!/ขั้นที่ 3/.test(py), 'step numbers count only the enabled steps');
    const app = appTwo(pipe, src);
    try { assert.deepEqual(countDiff(app, runPython(py, [src]).bytes), { n: 0, mx: 0 }); } finally { app.delete(); }
    const withOff2 = [step('gaussian_blur', { ksize: 5 }), off(two('imabsdiff', {}, [step('invert')]))];
    assert.ok(code(withOff2).includes('def process(path):'), 'only a disabled 2-image step -> single-image script');
  } finally { src.delete(); }
});
test('bl-R7b. disabled steps inside a B chain (and a disabled nested two-image op) are skipped like the app: Python == app byte for byte', PY_SKIP, () => {
  const src = noiseRGB(30, 22, 12);
  try {
    const pipes = [
      [two('imabsdiff', {}, [step('gaussian_blur', { ksize: 9 }), off(step('invert'))])],
      [two('imabsdiff', {}, [off(two('imarith', { operation: 'add' }, [step('invert')])), step('gaussian_blur', { ksize: 5 })])],
      [two('imarith', { operation: 'add' }, [step('invert')]), off(step('gamma', { g: 3 })), two('imabsdiff', {}, [off(step('invert'))])],
    ];
    for (const pipe of pipes) {
      const app = appTwo(pipe, src);
      try { assert.deepEqual(countDiff(app, pyTwo(pipe, src).bytes), { n: 0, mx: 0 }); } finally { app.delete(); }
    }
  } finally { src.delete(); }
});

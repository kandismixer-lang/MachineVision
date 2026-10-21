// แก้บั๊กจากการเทียบกับเอกสาร MATLAB IPT (2026-10-21) — B1 Harris · B2 Laplacian sharpen · B3 Brightness/Contrast · B4 Skeleton
// รายงาน: Vision_boi/reports/matlab-ipt-compare-2026-10-21/README.md
// โครงเดียวกับ tests/backlog-2026-10-17.test.mjs: รัน Python จริงเมื่อมี python+cv2 (BL_NO_PY=1 จำลองเครื่องที่ไม่มี)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { OPS, FILTERS, defaultParams, needsOf } from '../src/renderer/filters/registry.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import { OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cornerHarris === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

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
function toRGB(m) { const o = new cv.Mat(); cv.cvtColor(m, o, cv.COLOR_GRAY2RGB); return o; }
const HARRIS = { blockSize: 2, ksize: 3, k: 0.04, thresh: 1 };
const step = (opId, extra = {}) => {
  const host = OP_TO_FILTER[opId];
  if (!host || host === opId) return { id: opId, params: { ...defaultParams(opId), ...extra } };
  const prefixed = {};
  for (const [k, v] of Object.entries(extra)) prefixed[`${opId}__${k}`] = v;
  return { id: host, params: { ...defaultParams(host), mode: opId, ...prefixed } };
};

// ภาพทดสอบ: ก้อนสี่เหลี่ยม/วงกลมสุ่ม (บางก้อนแตะขอบภาพ) + noise เล็กน้อย
function blobs(w, h, seed, noise = 0) {
  const r = lcg(seed);
  const m = cv.Mat.zeros(h, w, cv.CV_8UC1);
  for (let k = 0; k < 10; k++) {
    const x = Math.floor(r() * w), y = Math.floor(r() * h);
    if (k % 2) cv.circle(m, new cv.Point(x, y), 3 + Math.floor(r() * 30), new cv.Scalar(255), -1);
    else cv.rectangle(m, new cv.Point(x, y), new cv.Point(x + Math.floor(r() * 80), y + Math.floor(r() * 50)), new cv.Scalar(255), -1);
  }
  for (let i = 0; i < w * h; i++) if (r() < noise) m.data[i] = 255 - m.data[i];
  return m;
}

// Zhang-Suen แบบไล่ตรวจทุกพิกเซลทุกรอบ (ช้าแต่ตรงตามนิยาม) บนภาพที่เติมขอบดำ 1 px — ใช้เป็นเฉลย
function zhangSuenReference(bin) {
  const w = bin.cols, h = bin.rows, W = w + 2;
  const im = new Uint8Array(W * (h + 2));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) im[(y + 1) * W + x + 1] = bin.data[y * w + x] > 127 ? 1 : 0;
  for (;;) {
    let changed = false;
    for (const st of [1, 2]) {
      const del = [];
      for (let y = 1; y <= h; y++) {
        for (let x = 1; x <= w; x++) {
          const i = y * W + x;
          if (!im[i]) continue;
          const n = [im[i - W], im[i - W + 1], im[i + 1], im[i + W + 1], im[i + W], im[i + W - 1], im[i - 1], im[i - W - 1]];
          const B = n.reduce((a, b) => a + b, 0);
          if (B < 2 || B > 6) continue;
          let A = 0;
          for (let k = 0; k < 8; k++) if (!n[k] && n[(k + 1) % 8]) A++;
          if (A !== 1) continue;
          const [p2, , p4, , p6, , p8] = n;
          if (st === 1 ? (p2 * p4 * p6 || p4 * p6 * p8) : (p2 * p4 * p8 || p2 * p6 * p8)) continue;
          del.push(i);
        }
      }
      if (del.length) { changed = true; for (const i of del) im[i] = 0; }
    }
    if (!changed) break;
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = im[(y + 1) * W + x + 1] ? 255 : 0;
  return out;
}

// ===================================================================
// Python จริง (cv2 / numpy) — ข้ามเมื่อไม่มี python/cv2
// ===================================================================
const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'matlab-fix-1021-'));
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
    r = ns['process'](c['inp'])
r.tofile(c['out'])
json.dump({'stdout': buf.getvalue(), 'ndim': int(r.ndim)}, open(sys.argv[2], 'w'))
`;
let pyCounter = 0;
// src: Mat RGB 3ch (ลำดับของแอป) → process() ของโค้ดที่ export โดยแทน cv2.imread ด้วยการอ่านไบต์ตรง ๆ
function runPython(pipe, src) {
  const id = `${process.pid}_${pyCounter++}`;
  let c = buildPythonCode(pipe).replace('img = cv2.imread(path)', `img = np.fromfile(path, dtype=np.uint8).reshape((${src.rows}, ${src.cols}, 3))`);
  const mainIdx = c.indexOf('if __name__ == "__main__":');
  assert.ok(mainIdx > 0);
  c = c.slice(0, mainIdx);
  const o = { code: path.join(TMP_DIR, `c_${id}.py`), inp: path.join(TMP_DIR, `i_${id}.bin`), out: path.join(TMP_DIR, `o_${id}.bin`) };
  fs.writeFileSync(o.code, c, 'utf8');
  fs.writeFileSync(o.inp, swapRB3(Buffer.from(src.data)));
  const mp = path.join(TMP_DIR, `m_${id}.json`), rp = path.join(TMP_DIR, `r_${id}.json`), dp = path.join(TMP_DIR, `d_${id}.py`);
  fs.writeFileSync(mp, JSON.stringify(o)); fs.writeFileSync(dp, PY_DRIVER);
  execFileSync('python', [dp, mp, rp], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' }, maxBuffer: 1 << 28 });
  const r = JSON.parse(fs.readFileSync(rp, 'utf8'));
  const raw = fs.readFileSync(o.out);
  return { stdout: r.stdout, bytes: r.ndim === 3 ? swapRB3(raw) : Buffer.from(raw) };
}
function appRun(pipe, src, ctx = {}) {
  let cur = src.clone();
  for (const st of pipe) {
    const f = FILTERS[st.id];
    let input = cur, conv = null;
    if (needsOf(f, st.params) === 'gray' && cur.channels() > 1) { conv = new cv.Mat(); cv.cvtColor(cur, conv, cv.COLOR_RGB2GRAY); input = conv; }
    try { const o = f.apply(cv, input, st.params, ctx); cur.delete(); cur = o; } finally { if (conv) conv.delete(); }
  }
  return cur;
}
function countDiff(appMat, pyBytes) {
  const d = appMat.data;
  assert.equal(pyBytes.length, d.length, 'output size');
  let n = 0;
  for (let i = 0; i < d.length; i++) if (d[i] !== pyBytes[i]) n++;
  return n;
}

// ===================================================================
// B3 Brightness / Contrast
// ===================================================================
test('mf-B3a. brightness_contrast: β<0 clips to 0 like imlincomb (convertScaleAbs used to flip negatives, so the darkest pixels got brighter)', () => {
  const src = cv.matFromArray(1, 4, cv.CV_8UC1, [20, 41, 100, 250]);
  const out = OPS.brightness_contrast.apply(cv, src, { alpha: 1.2, beta: -50 });
  assert.deepEqual(Array.from(out.data), [0, 0, 70, 250]); // 250·1.2−50 = 250 · 41·1.2−50 = −0.8 → 0 (เดิมได้ 1)
  out.delete(); src.delete();
});

test('mf-B3b. brightness_contrast: exported Python == app pixel-for-pixel (incl. negative β and fractional α)', PY_SKIP, () => {
  const r = lcg(31);
  const src = rgb(97, 61, () => [r() * 256, r() * 256, r() * 256].map(Math.floor));
  for (const [alpha, beta] of [[1.2, -50], [0.7, 30], [2.3, -100]]) {
    const pipe = [step('brightness_contrast', { alpha, beta })];
    const app = appRun(pipe, src);
    const py = runPython(pipe, src);
    assert.equal(countDiff(app, py.bytes), 0, `α=${alpha} β=${beta}`);
    app.delete();
  }
  src.delete();
});

test('mf-B3c. brightness_contrast: string params from a hand-edited project are read as numbers (same as num() in codegen), and app == Python', PY_SKIP, () => {
  const src = rgb(16, 8, (x, y) => [x * 16, y * 32, 200]);
  const strOut = OPS.brightness_contrast.apply(cv, src, { alpha: '1', beta: '-50' });
  const numOut = OPS.brightness_contrast.apply(cv, src, { alpha: 1, beta: -50 });
  assert.deepEqual(Array.from(strOut.data), Array.from(numOut.data));
  const pipe = [step('brightness_contrast', { alpha: '1', beta: '-50' })];
  const app = appRun(pipe, src);
  assert.equal(countDiff(app, runPython(pipe, src).bytes), 0);
  [src, strOut, numOut, app].forEach((m) => m.delete());
});

// ===================================================================
// B2 Laplacian sharpen
// ===================================================================
test('mf-B2a. laplacian_sharpen: a 50|200 step gets darker on the dark side and brighter on the bright side (g = f − ∇²f, signed)', () => {
  const src = gray(8, 5, (x) => (x < 4 ? 50 : 200));
  const out = OPS.laplacian_sharpen.apply(cv, src, { strength: 1 });
  const row = Array.from(out.data.slice(16, 24));
  // ∇² (4 ทิศ) ตรงขอบ = +150 ฝั่งมืด / −150 ฝั่งสว่าง → 50−150 → 0 · 200+150 → 255 (เดิมได้ 0 ทั้งสองฝั่ง)
  assert.deepEqual(row, [50, 50, 50, 0, 255, 200, 200, 200]);
  const half = OPS.laplacian_sharpen.apply(cv, src, { strength: 0.2 });
  assert.deepEqual(Array.from(half.data.slice(16, 24)), [50, 50, 50, 20, 230, 200, 200, 200]);
  out.delete(); half.delete(); src.delete();
});

test('mf-B2b. laplacian_sharpen: strength 0 is the identity, and a flat image is unchanged at any strength', () => {
  const r = lcg(5);
  const src = rgb(40, 30, () => [r() * 256, r() * 256, r() * 256].map(Math.floor));
  const id0 = OPS.laplacian_sharpen.apply(cv, src, { strength: 0 });
  assert.deepEqual(Array.from(id0.data), Array.from(src.data));
  const flat = new cv.Mat(20, 20, cv.CV_8UC3, new cv.Scalar(90, 140, 30));
  const f3 = OPS.laplacian_sharpen.apply(cv, flat, { strength: 3 });
  assert.deepEqual(Array.from(f3.data), Array.from(flat.data));
  [src, id0, flat, f3].forEach((m) => m.delete());
});

test('mf-B2c. laplacian_sharpen: exported Python == app pixel-for-pixel on RGB noise and on gray', PY_SKIP, () => {
  const r = lcg(77);
  const src = rgb(83, 57, () => [r() * 256, r() * 256, r() * 256].map(Math.floor));
  for (const strength of [1, 0.3, 2.7]) {
    const pipe = [step('laplacian_sharpen', { strength })];
    const app = appRun(pipe, src);
    assert.equal(countDiff(app, runPython(pipe, src).bytes), 0, `strength ${strength}`);
    app.delete();
  }
  src.delete();
});

// ===================================================================
// B1 Harris
// ===================================================================
test('mf-B1a. harris_corners: default thresh (1% of max R, like MATLAB MinQuality 0.01) finds exactly the 4 corners of a rectangle — flat areas no longer pass', () => {
  const src = cv.Mat.zeros(240, 320, cv.CV_8UC1);
  cv.rectangle(src, new cv.Point(80, 60), new cv.Point(240, 180), new cv.Scalar(200), -1);
  for (const thresh of [1, 10, 50, 90]) {
    const ctx = {};
    OPS.harris_corners.apply(cv, src, { ...HARRIS, thresh }, ctx).delete();
    assert.equal(ctx.info, 'พบมุม 4 จุด', `thresh ${thresh}`);
  }
  src.delete();
});

test('mf-B1b. harris_corners: a flat image has no corners; one corner = one point (non-max suppression), not a dilated blob of pixels', () => {
  const flat = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(128));
  const c1 = {};
  OPS.harris_corners.apply(cv, flat, HARRIS, c1).delete();
  assert.equal(c1.info, 'พบมุม 0 จุด');
  const sq = cv.Mat.zeros(100, 100, cv.CV_8UC1);
  cv.rectangle(sq, new cv.Point(30, 30), new cv.Point(99, 99), new cv.Scalar(255), -1); // แตะขอบภาพ → มีมุมเดียวที่ (30,30)
  const c2 = {};
  OPS.harris_corners.apply(cv, sq, HARRIS, c2).delete();
  assert.equal(c2.info, 'พบมุม 1 จุด');
  flat.delete(); sq.delete();
});

test('mf-B1d. harris_corners: thresh 100 (slider max) still keeps the strongest corner(s) — the threshold is ">= thresh% of max"', () => {
  const sq = cv.Mat.zeros(100, 100, cv.CV_8UC1);
  cv.rectangle(sq, new cv.Point(30, 30), new cv.Point(99, 99), new cv.Scalar(255), -1);
  const ctx = {};
  OPS.harris_corners.apply(cv, sq, { ...HARRIS, thresh: 100 }, ctx).delete();
  assert.equal(ctx.info, 'พบมุม 1 จุด');
  sq.delete();
});

test('mf-B1c. harris_corners: exported Python == app (image and printed count) at 300x240 and 700x500 (downscale)', PY_SKIP, () => {
  for (const [w, h] of [[300, 240], [700, 500]]) {
    const g = blobs(w, h, 11);
    const src = toRGB(g);
    for (const thresh of [1, 20]) {
      const pipe = [step('harris_corners', { ...HARRIS, thresh })];
      const ctx = {};
      const app = appRun(pipe, src, ctx);
      const py = runPython(pipe, src);
      assert.ok(py.stdout.includes(ctx.info), `${w}x${h} thresh ${thresh}: app "${ctx.info}" vs python "${py.stdout.trim()}"`);
      assert.match(ctx.info, /พบมุม \d+ จุด/);
      const n = countDiff(app, py.bytes);
      assert.ok(w <= 480 ? n === 0 : n <= 40, `${w}x${h} thresh ${thresh}: ${n} differing bytes`);
      app.delete();
    }
    g.delete(); src.delete();
  }
});

// ===================================================================
// B4 Skeleton
// ===================================================================
test('mf-B4a. morph_skeleton == full-scan Zhang-Suen on a zero-padded image (the active-pixel shortcut changes nothing), incl. noisy and border-touching inputs', () => {
  const r = lcg(99);
  for (let t = 0; t < 8; t++) {
    const w = 40 + Math.floor(r() * 200), h = 30 + Math.floor(r() * 150);
    const src = blobs(w, h, 1000 + t, t % 2 ? 0.06 : 0);
    const out = OPS.morph_skeleton.apply(cv, src, {}, {});
    const ref = zhangSuenReference(src);
    let diff = 0;
    for (let i = 0; i < ref.length; i++) if (ref[i] !== out.data[i]) diff++;
    assert.equal(diff, 0, `case ${t} ${w}x${h}`);
    out.delete(); src.delete();
  }
});

test('mf-B4b. morph_skeleton: full resolution (1px line on a 640x480 bar), border pixels thin too, output size = input size', () => {
  const bar = cv.Mat.zeros(480, 640, cv.CV_8UC1);
  cv.rectangle(bar, new cv.Point(0, 220), new cv.Point(639, 259), new cv.Scalar(255), -1); // แตะขอบซ้าย-ขวา
  const ctx = {};
  const out = OPS.morph_skeleton.apply(cv, bar, {}, ctx);
  assert.equal(out.cols, 640); assert.equal(out.rows, 480);
  let thick = 0;
  for (let y = 0; y < 480; y++) if (out.ucharAt(y, 320) > 0) thick++;
  assert.equal(thick, 1);
  let edgeCol = 0;
  for (let y = 0; y < 480; y++) if (out.ucharAt(y, 0) > 0) edgeCol++;
  assert.ok(edgeCol <= 1, `left border column should be thinned too, has ${edgeCol} white pixels`);
  assert.match(ctx.info, /ที่ 640x480px/);
  out.delete(); bar.delete();
});

test('mf-B4c. morph_skeleton: exported Python == app pixel-for-pixel (incl. >240px, border-touching and noisy inputs)', PY_SKIP, () => {
  for (const [w, h, seed, noise] of [[130, 90, 3, 0], [500, 300, 4, 0], [321, 241, 5, 0.04], [97, 61, 6, 0.5], [1, 1, 7, 0]]) {
    const g = blobs(w, h, seed, noise);
    const src = toRGB(g);
    const pipe = [step('morph_skeleton')];
    const app = appRun(pipe, src);
    assert.equal(countDiff(app, runPython(pipe, src).bytes), 0, `${w}x${h}`);
    app.delete(); g.delete(); src.delete();
  }
});

// ===================================================================
// ข้อความที่เคยอ้าง MATLAB ผิด
// ===================================================================
test('mf-T. corrected MATLAB claims in help texts', () => {
  assert.doesNotMatch(OPS.imarith.desc, /\(เหมือน imadd\/imsubtract\/immultiply\/imdivide/);
  assert.match(OPS.imarith.desc, /immultiply/);
  assert.doesNotMatch(OPS.delta_e.desc, /ค่าเริ่มต้นของ MATLAB คือ CIE76/);
  assert.match(OPS.delta_e.desc, /CIE94/);
  assert.doesNotMatch(OPS.guided_filter.desc, /MATLAB กับภาพสีใช้ทั้ง 3 ช่องร่วมกัน/);
  const amount = OPS.dehaze.params.find((p) => p.key === 'amount');
  assert.doesNotMatch(amount.hint, /MATLAB imreducehaze ค่าเริ่มต้น 0\.95/);
  const shape = FILTERS.morphology.params.find((p) => p.key === 'shape');
  assert.doesNotMatch(shape.options.find((o) => o.value === 'ellipse').label, /เท่ากับ/);
  assert.doesNotMatch(OPS.morph_hit_or_miss.desc, /bwhitmiss นับเป็นดำ/);
  assert.match(OPS.imarith.desc, /A > 0 กับ B = 0 ได้ 255/);
  assert.doesNotMatch(OPS.morph_skeleton.desc, /240px/);
});

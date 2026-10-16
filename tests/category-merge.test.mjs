// 2026-10-07 — "1 หมวดบน palette = 1 node": dropdown "เลือกวิธี" ของ node ที่หยิบจากหมวดต้องมีครบทุก op ของหมวด
// ตามลำดับ TOOL_SECTIONS (บั๊กเดิม: COLOR 9 ตัว แต่คลิก Grayscale ได้ node ที่ dropdown มี 6)
// ครอบคลุม: (1) palette ↔ dropdown ตรงกันทุกหมวด (กันกลับมาเมื่อเพิ่ม op ใหม่) · (2) กลุ่มใหม่ apply/needs/codegen
// เท่ากับ node เดิมทุก op · (3) กลุ่มเดิมยังลงทะเบียนครบ (legacy) · (4) migrateLegacyStep ของเทมเพลตเก่า
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import cv from '@techstark/opencv-js';
import {
  FILTERS, OPS, defaultParams, needsOf, GROUP_COLORS, GROUP_ORDER, CATEGORY_FILTER_IDS, migrateLegacyStep,
} from '../src/renderer/filters/registry.js';
import { TOOL_SECTIONS, OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { GROUP_ICON } from '../src/renderer/filters/ui-meta.js';
import { buildPythonCode } from '../src/renderer/codegen.js';

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cvtColor === 'function'; }
if (!isReady(cv)) {
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}

const modeOptions = (id) => FILTERS[id].params.find((p) => p.key === 'mode').options.map((o) => o.value);

// กลุ่มเดิม (ก่อน 2026-10-07) ทั้ง 26 ตัว — ต้องยังอยู่ครบและ op list/default เดิมไม่ขยับ
const LEGACY_IDS = [
  'color_space', 'image_adjust', 'sharpen', 'noise', 'blur', 'morphology', 'threshold', 'edge_detection', 'shape_analysis',
  'watershed', 'grabcut', 'region_growing', 'object_detection', 'colour', 'geometry', 'frequency', 'image_arithmetic',
  'template_matching', 'psnr', 'ssim', 'mog2', 'farneback_optical_flow', 'pyrlk_optical_flow', 'enhancement', 'restoration',
  'segmentation',
];

// ---------------------------------------------------------------------------
// (1) palette ↔ dropdown
// ---------------------------------------------------------------------------
test('cm-1. every TOOL_SECTIONS category: all ops map to ONE filter whose "เลือกวิธี" dropdown lists exactly those ops in order, default = first op', () => {
  for (const section of TOOL_SECTIONS) {
    const filterIds = new Set(section.ops.map((op) => OP_TO_FILTER[op]));
    assert.equal(filterIds.size, 1, `${section.title}: ops map to ${[...filterIds].join(', ')} (must be one filter)`);
    const fid = [...filterIds][0];
    assert.ok(FILTERS[fid], `${section.title}: ${fid} must be registered`);
    const modeParam = FILTERS[fid].params.find((p) => p.key === 'mode');
    if (!modeParam) {
      // กลุ่ม op เดียว (morphology / template_matching) — ไม่มี dropdown
      assert.deepEqual(section.ops, [fid], `${section.title}: single-op category must be the filter itself`);
      continue;
    }
    assert.deepEqual(modeOptions(fid), section.ops, `${section.title}: dropdown of ${fid} must equal the category ops in order`);
    // GEOMETRY คง default เดิม 'resize' (กลุ่มเดิมที่ไม่ได้ถูกแทน — ผู้ใช้เคยได้ Resize เป็นค่าเริ่มต้น) · ที่เหลือ = op แรกของหมวด
    const wantDefault = section.title === 'GEOMETRY' ? 'resize' : section.ops[0];
    assert.equal(modeParam.default, wantDefault, `${section.title}: default`);
  }
});

test('cm-2. each op appears in exactly one category; the 7 split categories use the new category-level groups; 12 categories = 12 palette nodes', () => {
  const seen = new Map();
  for (const s of TOOL_SECTIONS) for (const op of s.ops) {
    assert.ok(OPS[op], `${op} must exist in OPS`);
    assert.ok(!seen.has(op), `${op} listed in both ${seen.get(op)} and ${s.title}`);
    seen.set(op, s.title);
  }
  const EXPECT = {
    COLOR: 'color_tools', ENHANCE: 'enhance_tools', 'FILTER & DENOISE': 'filter_denoise',
    'SEGMENTATION & EDGE': 'segment_edge', 'ANALYSIS & DETECTION': 'analysis_detect', 'IMAGE QUALITY': 'image_quality', MOTION: 'motion',
  };
  for (const [title, fid] of Object.entries(EXPECT)) {
    const s = TOOL_SECTIONS.find((x) => x.title === title);
    for (const op of s.ops) assert.equal(OP_TO_FILTER[op], fid, `${op} (${title})`);
  }
  assert.equal(new Set(Object.values(OP_TO_FILTER)).size, TOOL_SECTIONS.length, 'one palette node per category');
  // กลุ่มเดิมที่ถูกแทนต้องไม่โผล่บน palette อีก
  const hidden = ['color_space', 'colour', 'image_adjust', 'sharpen', 'noise', 'blur', 'threshold', 'edge_detection',
    'shape_analysis', 'object_detection', 'watershed', 'grabcut', 'region_growing', 'psnr', 'ssim', 'mog2',
    'farneback_optical_flow', 'pyrlk_optical_flow', 'enhancement', 'restoration', 'segmentation'];
  const shown = new Set(Object.values(OP_TO_FILTER));
  for (const id of hidden) assert.ok(!shown.has(id), `${id} must be hidden from the palette (legacy)`);
});

test('cm-3. new groups: flags (psnr/ssim group inputs:2, motion stateful:true, nobody else changed), colors, icons, GROUP_ORDER', () => {
  assert.equal(CATEGORY_FILTER_IDS.length, 7);
  for (const id of CATEGORY_FILTER_IDS) {
    const f = FILTERS[id];
    assert.ok(f, id);
    assert.equal(f.inputs, id === 'image_quality' ? 2 : undefined, `${id}.inputs`);
    assert.equal(f.stateful, id === 'motion' ? true : undefined, `${id}.stateful`);
    assert.ok(GROUP_COLORS[f.group] && GROUP_COLORS[f.group].title, `${id}: GROUP_COLORS["${f.group}"]`);
    assert.ok(GROUP_ORDER.includes(f.group), `${id}: GROUP_ORDER has "${f.group}"`);
    assert.ok(GROUP_ICON[id], `${id}: GROUP_ICON`);
  }
  const icons = new Map();
  for (const [id, g] of Object.entries(GROUP_ICON)) {
    if (!icons.has(g)) icons.set(g, []);
    icons.get(g).push(id);
  }
  for (const id of CATEGORY_FILTER_IDS) assert.equal(icons.get(GROUP_ICON[id]).length, 1, `${id}: icon collides with ${icons.get(GROUP_ICON[id])}`);
  // เฉพาะกลุ่มเก่า psnr/ssim (+image_quality) / template_matching / image_arithmetic ที่เป็น inputs:2; stateful เฉพาะ motion + 3 ตัวเดี่ยว
  const two = Object.entries(FILTERS).filter(([, f]) => f.inputs !== undefined).map(([k]) => k).sort();
  assert.deepEqual(two, ['image_arithmetic', 'image_quality', 'psnr', 'ssim', 'template_matching']);
  const st = Object.entries(FILTERS).filter(([, f]) => f.stateful !== undefined).map(([k]) => k).sort();
  assert.deepEqual(st, ['farneback_optical_flow', 'mog2', 'motion', 'pyrlk_optical_flow']);
});

test('cm-4. legacy: all 26 old FILTERS still registered with their ORIGINAL op list/default (saved templates stay valid)', () => {
  for (const id of LEGACY_IDS) assert.ok(FILTERS[id], `${id} must stay registered`);
  assert.equal(Object.keys(FILTERS).length, LEGACY_IDS.length + CATEGORY_FILTER_IDS.length);
  const ORIG = {
    color_space: ['grayscale', 'colorspace', 'channel_split', 'rgb_adjust', 'color_inrange', 'invert'],
    colour: ['color_inrange', 'kmeans_segment', 'white_balance', 'invert', 'slic'],
    image_adjust: ['brightness_contrast', 'gamma', 'hist_equalize', 'clahe', 'contrast_stretch', 'flatfield_correction'],
    segmentation: ['binary_threshold', 'otsu_threshold', 'adaptive_threshold', 'canny', 'sobel', 'find_contours', 'connected_components', 'watershed'],
    restoration: ['salt_pepper_noise', 'median_blur', 'gaussian_blur', 'box_blur', 'bilateral'],
    enhancement: ['brightness_contrast', 'gamma', 'hist_equalize', 'clahe', 'unsharp', 'laplacian_sharpen'],
  };
  for (const [id, ops] of Object.entries(ORIG)) assert.deepEqual(modeOptions(id), ops, id);
  assert.equal(FILTERS.color_space.params[0].default, 'grayscale');
  assert.equal(FILTERS.segmentation.params[0].default, 'binary_threshold');
  assert.equal(FILTERS.restoration.params[0].default, 'gaussian_blur');
});

// ---------------------------------------------------------------------------
// (2) กลุ่มใหม่ = node เดิม (needs / apply / codegen) ทุก op ของ palette
// ---------------------------------------------------------------------------
// legacy filterId ของแต่ละ op (ตามที่ palette เคยชี้): ตัวอย่างคือ op ที่ palette เคย resolve
const OLD_HOST = {};
{
  const order = ['colour', 'color_space', 'image_adjust', 'sharpen', 'geometry', 'noise', 'blur', 'morphology', 'threshold',
    'edge_detection', 'shape_analysis', 'object_detection', 'watershed', 'grabcut', 'region_growing', 'frequency',
    'image_arithmetic', 'template_matching', 'psnr', 'ssim', 'mog2', 'farneback_optical_flow', 'pyrlk_optical_flow'];
  for (const fid of order) {
    const m = FILTERS[fid].params.find((p) => p.key === 'mode');
    if (m) for (const o of m.options) OLD_HOST[o.value] = fid; else OLD_HOST[fid] = fid;
  }
}
// op ใหม่หลัง 2026-10-07 อยู่เฉพาะกลุ่มระดับหมวด ไม่มี node เดิมให้เทียบ · ลำดับ = ลำดับใน CATEGORY_OPS (cm-5 deepEqual ตามลำดับ)
// IP batch 4A (2026-10-15) เพิ่ม pseudocolor / guided_filter / focus_score / delta_e (roi_apply อยู่ image_arithmetic ซึ่งไม่ใช่หมวดที่ re-group จึงไม่อยู่ในลิสต์นี้)
// IP batch 4B (2026-10-15) เพิ่ม dehaze / nlm_denoise / deconv_wiener / deconv_lucy / gabor_filter / glcm (dct_jpeg อยู่ frequency, ecc_register อยู่ image_arithmetic — ไม่ใช่หมวดที่ re-group)
const POST_MERGE_OPS = ['pseudocolor', 'quantize', 'dehaze', 'guided_filter', 'nlm_denoise', 'deconv_wiener', 'deconv_lucy', 'gabor_filter', 'glcm', 'focus_score', 'shape_filter', 'delta_e'];
const CATEGORY_OPS = TOOL_SECTIONS.filter((s) => CATEGORY_FILTER_IDS.includes(OP_TO_FILTER[s.ops[0]])).flatMap((s) => s.ops);

function paramsFor(fid, op) {
  const p = defaultParams(fid);
  if ('mode' in p) p.mode = op;
  return p;
}

function makeRGB(seed = 1) {
  const W = 48, H = 40;
  const data = new Uint8Array(W * H * 3);
  let s = seed >>> 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const i = (y * W + x) * 3;
    const blob = (x > 10 && x < 30 && y > 8 && y < 28) ? 150 : 0;
    data[i] = Math.min(255, 40 + x * 3 + blob);
    data[i + 1] = Math.min(255, 30 + y * 4 + (s >>> 28));
    data[i + 2] = (x * 7 + y * 5 + blob) & 255;
  }
  const m = new cv.Mat(H, W, cv.CV_8UC3);
  m.data.set(data);
  return m;
}
function toGrayIfNeeded(m, needs) {
  if (needs !== 'gray' || m.channels() === 1) return { mat: m, owned: false };
  const g = new cv.Mat();
  cv.cvtColor(m, g, cv.COLOR_RGB2GRAY);
  return { mat: g, owned: true };
}
// op ที่สุ่มจริงและ seed ไม่ได้: salt_pepper_noise ใช้ Math.random (stub ให้นิ่งด้านล่าง) · kmeans_segment ใช้ RNG ภายใน cv.kmeans
// (opencv.js ไม่มี setRNGSeed) → เทียบ "ทุกอย่างยกเว้นพิกเซล" (ขนาด/ชนิด/ข้อความ) — พิกเซลของ kmeans เทียบผ่านเส้นทางโค้ดเดียวกันอยู่แล้ว
const PIXELS_NONDETERMINISTIC = new Set(['kmeans_segment']);
function withFixedRandom(fn) {
  const saved = Math.random;
  let s = 123456789;
  Math.random = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
  try { return fn(); } finally { Math.random = saved; }
}
function runFilter(fid, params, withSrc2) {
  return withFixedRandom(() => runFilterRaw(fid, params, withSrc2));
}
const pix = (r, op) => (PIXELS_NONDETERMINISTIC.has(op) && r ? { ...r, data: null } : r);
// รัน filter หนึ่งตัวบนภาพสังเคราะห์ → { data, rows, cols, ch, info, values } (ล้าง Mat ทุกตัว)
function runFilterRaw(fid, params, withSrc2) {
  const base = makeRGB(7);
  const f = FILTERS[fid];
  const { mat: src, owned } = toGrayIfNeeded(base, needsOf(f, params));
  const src2 = withSrc2 ? makeRGB(99) : null;
  const ctx = {};
  if (src2) ctx.src2 = src2;
  let out = null;
  try {
    out = f.apply(cv, src, params, ctx);
    return {
      rows: out.rows, cols: out.cols, ch: out.channels(), type: out.type(),
      data: Buffer.from(out.data).toString('base64'),
      info: ctx.info === undefined ? null : ctx.info,
      values: JSON.stringify(ctx.values === undefined ? null : ctx.values),
    };
  } finally {
    if (out) out.delete();
    if (owned) src.delete();
    if (src2) src2.delete();
    base.delete();
  }
}

test('cm-5. needs(): new group == old node for every op of the 7 re-grouped categories', () => {
  assert.ok(CATEGORY_OPS.length >= 50, `expected the 7 categories to cover many ops, got ${CATEGORY_OPS.length}`);
  assert.deepEqual(CATEGORY_OPS.filter((op) => !OLD_HOST[op]), POST_MERGE_OPS);
  for (const op of CATEGORY_OPS) {
    if (POST_MERGE_OPS.includes(op)) continue;
    const nf = OP_TO_FILTER[op];
    const of = OLD_HOST[op];
    assert.equal(needsOf(FILTERS[nf], paramsFor(nf, op)), needsOf(FILTERS[of], paramsFor(of, op)), op);
    assert.equal(FILTERS[nf].subtitle(paramsFor(nf, op)), OPS[op].name, `${op}: subtitle`);
    assert.equal(FILTERS[nf].desc(paramsFor(nf, op)), OPS[op].desc, `${op}: desc`);
  }
});

test('cm-6. apply(): new group output == old node output (pixels, size, ctx.info, ctx.values) for every op of the 7 categories', () => {
  for (const op of CATEGORY_OPS) {
    if (POST_MERGE_OPS.includes(op)) continue;
    const nf = OP_TO_FILTER[op];
    const of = OLD_HOST[op];
    const two = FILTERS[nf].inputs === 2;
    assert.equal(FILTERS[of].inputs === 2, two, `${op}: inputs flag must match old node`);
    let a, b, errA = null, errB = null;
    try { a = runFilter(of, paramsFor(of, op), two); } catch (e) { errA = String(e && e.message || e); }
    try { b = runFilter(nf, paramsFor(nf, op), two); } catch (e) { errB = String(e && e.message || e); }
    assert.equal(errB, errA, `${op}: both must throw the same error (or neither)`);
    if (!errA) assert.deepEqual(pix(b, op), pix(a, op), `${op}: output differs between ${of} and ${nf}`);
  }
});

test('cm-7. codegen: Python code of the new group == old node for every op of the 7 categories', () => {
  for (const op of CATEGORY_OPS) {
    if (POST_MERGE_OPS.includes(op)) continue;
    const nf = OP_TO_FILTER[op];
    const of = OLD_HOST[op];
    const mk = (fid) => {
      const step = { id: fid, params: paramsFor(fid, op) };
      if (FILTERS[fid].inputs === 2) step.src2 = { pipeline: [] };
      return buildPythonCode([step], { displayName: 'X' });
    };
    assert.equal(mk(nf), mk(of), `${op}: codegen differs`);
  }
});

// ---------------------------------------------------------------------------
// (3) migration ของเทมเพลตเก่า
// ---------------------------------------------------------------------------
test('cm-8. migrateLegacyStep: merged legacy group → category group (mode kept, params copied, input not mutated)', () => {
  const src = { id: 'segmentation', params: { mode: 'canny', canny__low: 33, canny__high: 99 }, nodeId: 5 };
  const frozen = JSON.stringify(src);
  const m = migrateLegacyStep(src);
  assert.equal(JSON.stringify(src), frozen, 'input must not be mutated');
  assert.deepEqual(m, { id: 'segment_edge', params: { mode: 'canny', canny__low: 33, canny__high: 99 }, nodeId: 5 });
  // restoration gaussian_blur → filter_denoise · segmentation find_contours → analysis_detect (ข้ามหมวด ตาม op)
  assert.equal(migrateLegacyStep({ id: 'restoration', params: { mode: 'median_blur' } }).id, 'filter_denoise');
  assert.equal(migrateLegacyStep({ id: 'segmentation', params: { mode: 'find_contours' } }).id, 'analysis_detect');
  assert.equal(migrateLegacyStep({ id: 'colour', params: { mode: 'kmeans_segment' } }).id, 'color_tools');
  assert.equal(migrateLegacyStep({ id: 'enhancement', params: { mode: 'clahe' } }).id, 'enhance_tools');
  // legacy watershed ใน shape_analysis → segment_edge
  assert.equal(migrateLegacyStep({ id: 'shape_analysis', params: { mode: 'watershed' } }).id, 'segment_edge');
});

test('cm-9. migrateLegacyStep: no mode → default mode of the old group (color_space {} → color_tools/grayscale; template step shapes of app.js)', () => {
  assert.deepEqual(migrateLegacyStep({ id: 'color_space' }), { id: 'color_tools', params: { mode: 'grayscale' } });
  assert.deepEqual(migrateLegacyStep({ id: 'restoration', params: {} }), { id: 'filter_denoise', params: { mode: 'gaussian_blur' } });
});

test('cm-10. migrateLegacyStep: single-op legacy node gets mode + "op__" prefix (watershed, psnr, mog2)', () => {
  assert.deepEqual(migrateLegacyStep({ id: 'watershed', params: { a: 1, b: 2 } }),
    { id: 'segment_edge', params: { mode: 'watershed', watershed__a: 1, watershed__b: 2 } });
  assert.deepEqual(migrateLegacyStep({ id: 'psnr', params: {} }), { id: 'image_quality', params: { mode: 'psnr' } });
  assert.equal(migrateLegacyStep({ id: 'mog2', params: {} }).id, 'motion');
  assert.equal(migrateLegacyStep({ id: 'region_growing', params: {} }).id, 'segment_edge');
});

test('cm-11. migrateLegacyStep: leaves alone steps that need no change (new groups, morphology/geometry/template_matching, unknown id/mode, null)', () => {
  const same = [
    { id: 'color_tools', params: { mode: 'invert' } }, { id: 'morphology', params: { op: 'open' } },
    { id: 'geometry', params: { mode: 'resize' } }, { id: 'template_matching', params: {} },
    { id: 'frequency', params: { mode: 'fft_magnitude' } }, { id: 'image_arithmetic', params: { mode: 'imabsdiff' } },
    { id: 'nope', params: {} }, { id: 'toString', params: {} }, { id: 'segmentation', params: { mode: 'not_an_op' } },
  ];
  for (const s of same) assert.equal(migrateLegacyStep(s), s, JSON.stringify(s));
  assert.equal(migrateLegacyStep(null), null);
  assert.equal(migrateLegacyStep(undefined), undefined);
});

test('cm-12. migration preserves behaviour: for EVERY legacy node x every op, old node output == migrated node output; migrated mode is in the new dropdown', () => {
  let checked = 0;
  for (const id of LEGACY_IDS) {
    const m = FILTERS[id].params.find((p) => p.key === 'mode');
    const ops = m ? m.options.map((o) => o.value) : [id];
    if (id === 'shape_analysis') ops.push('watershed'); // legacyOpIds
    for (const op of ops) {
      const oldParams = paramsFor(id, op);
      const mig = migrateLegacyStep({ id, params: oldParams });
      if (mig.id === id) {
        // morphology/geometry/frequency/image_arithmetic/template_matching = ไม่ย้าย (ตรงกับหมวด palette อยู่แล้ว)
        assert.ok(['morphology', 'geometry', 'frequency', 'image_arithmetic', 'template_matching'].includes(id), `${id}/${op} unexpectedly not migrated`);
        continue;
      }
      assert.ok(CATEGORY_FILTER_IDS.includes(mig.id), `${id}/${op} → ${mig.id}`);
      assert.equal(mig.params.mode, op);
      assert.ok(modeOptions(mig.id).includes(op), `${op} must be in ${mig.id} dropdown`);
      const newParams = { ...defaultParams(mig.id), ...mig.params };
      const two = FILTERS[id].inputs === 2;
      assert.equal(FILTERS[mig.id].inputs === 2, two, `${id}/${op}: inputs`);
      assert.equal(Boolean(FILTERS[mig.id].stateful), Boolean(FILTERS[id].stateful), `${id}/${op}: stateful`);
      assert.equal(needsOf(FILTERS[mig.id], newParams), needsOf(FILTERS[id], oldParams), `${id}/${op}: needs`);
      let a, b, ea = null, eb = null;
      try { a = runFilter(id, oldParams, two); } catch (e) { ea = String(e && e.message || e); }
      try { b = runFilter(mig.id, newParams, two); } catch (e) { eb = String(e && e.message || e); }
      assert.equal(eb, ea, `${id}/${op}: error parity`);
      if (!ea) assert.deepEqual(pix(b, op), pix(a, op), `${id}/${op}: output differs after migration`);
      checked++;
    }
  }
  assert.ok(checked > 80, `expected >80 migrated (legacy node, op) pairs, got ${checked}`);
});

test('cm-13. migration keeps non-default param values: old node with custom sub-param → same output on migrated node', () => {
  const old = { ...defaultParams('restoration'), mode: 'median_blur', median_blur__ksize: 9 };
  const mig = migrateLegacyStep({ id: 'restoration', params: old });
  assert.equal(mig.params.median_blur__ksize, 9);
  const a = runFilter('restoration', old, false);
  const b = runFilter(mig.id, { ...defaultParams(mig.id), ...mig.params }, false);
  assert.deepEqual(b, a);
  // และต่างจากค่า default จริง (กันเทสต์ผ่านเพราะ param ไม่มีผล)
  const d = runFilter('restoration', { ...defaultParams('restoration'), mode: 'median_blur' }, false);
  assert.notDeepEqual(a.data, d.data);
});

test('cm-14. editor.applyTemplate wires migrateLegacyStep (static read — LiteGraph is not loadable in node)', () => {
  const src = fs.readFileSync(new URL('../src/renderer/graph/editor.js', import.meta.url), 'utf8');
  assert.match(src, /import \{[^}]*migrateLegacyStep[^}]*\} from "\.\.\/filters\/registry\.js"/);
  assert.match(src, /const step = migrateLegacyStep\(raw\)/);
  // ค่าที่ migrate แล้วต้องเป็นสิ่งที่ถูกใส่ลง node จริง (ไม่ใช่ raw.params)
  assert.match(src, /Object\.assign\(n\.properties, step\.params\)/);
  assert.doesNotMatch(src, /Object\.assign\(n\.properties, raw\.params\)/);
  const at = src.indexOf('function applyTemplate');
  assert.ok(at > 0 && src.indexOf('migrateLegacyStep(raw)', at) > at, 'migrateLegacyStep must be called inside applyTemplate');
  assert.match(src, /LiteGraph\.createNode\("filters\/" \+ step\.id\)/);
});

test('cm-15. app.js built-in templates still reference only registered ids, and each migrates to a registered category group', () => {
  const src = fs.readFileSync(new URL('../src/renderer/app.js', import.meta.url), 'utf8');
  const block = src.slice(src.indexOf('const TEMPLATES = ['), src.indexOf('// เทมเพลตที่ผู้ใช้เซฟเอง'));
  const ids = [...block.matchAll(/\{ id: "([a-z_0-9]+)"/g)].map((m) => m[1]);
  assert.ok(ids.length >= 7, `found ${ids.length} template steps`);
  for (const id of ids) {
    assert.ok(FILTERS[id], `template id ${id} must be registered`);
    const mig = migrateLegacyStep({ id, params: {} });
    assert.ok(FILTERS[mig.id], `${id} → ${mig.id}`);
  }
});

test('cm-16. migration of single-op legacy steps with NON-default values: params keep their value under the "op__" prefix and the migrated node behaves like the legacy one', () => {
  const cases = [
    { id: 'mog2', params: { view: 'overlay' } },
    { id: 'watershed', params: Object.fromEntries(OPS.watershed.params.slice(0, 1).map((q) => [q.key, q.type === 'select' ? q.options.find((o) => o.value !== q.default).value : (q.max !== undefined ? q.max : q.default)])) },
    { id: 'region_growing', params: { seedX: 20, seedY: 70, tolerance: 40 } },
  ];
  for (const c of cases) {
    const legacyParams = { ...defaultParams(c.id), ...c.params };
    for (const k of Object.keys(c.params)) assert.notEqual(legacyParams[k], defaultParams(c.id)[k], `${c.id}.${k}: test value must differ from default`);
    const mig = migrateLegacyStep({ id: c.id, params: c.params });
    for (const [k, v] of Object.entries(c.params)) assert.equal(mig.params[c.id + '__' + k], v, `${c.id}__${k}`);
    // เหมือน applyTemplate: defaultParams(กลุ่มใหม่) แล้ว assign ทับด้วย params ที่ migrate แล้ว
    const applied = Object.assign(defaultParams(mig.id), mig.params);
    for (const [k, v] of Object.entries(c.params)) assert.equal(applied[c.id + '__' + k], v);
    assert.equal(applied.mode, c.id);
    assert.deepEqual(runFilter(mig.id, applied, false), runFilter(c.id, legacyParams, false), `${c.id}: output differs`);
  }
});

test('cm-17. Motion node: switching method on ONE nodeId via the real runPipeline gives a fresh state, disposes the old one, and reports "เฟรมแรก"', async () => {
  globalThis.self = globalThis;
  if (typeof globalThis.ImageData === 'undefined') globalThis.ImageData = class { constructor(d, w, h) { this.data = d; this.width = w; this.height = h; } };
  const W = await import('../src/renderer/cv/cv-worker.js');
  const frame = (shift) => {
    const w = 64, h = 48, d = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 3, v = ((((x + shift) * 37) ^ (y * 53)) & 255);
      d[i] = v; d[i + 1] = (v * 3) & 255; d[i + 2] = (v + 80) & 255;
    }
    return cv.matFromArray(h, w, cv.CV_8UC3, d);
  };
  const store = new Map();
  const base = defaultParams('motion');
  const seq = ['mog2', 'mog2', 'mog2', 'farneback_optical_flow', 'farneback_optical_flow', 'pyrlk_optical_flow', 'pyrlk_optical_flow', 'mog2', 'mog2'];
  const wrapped = new WeakSet();
  let prevMode = null, prevState = null, disposedOld = 0, k = 0;
  for (const mode of seq) {
    const img = frame(k++);
    const pl = [{ id: 'motion', params: { ...base, mode }, enabled: true, nodeId: 4242 }];
    const memo = new Map();
    // สลับแล้วต้องเรียก __dispose ของ state เก่า (op ตั้ง __dispose เอง — ครอบด้วยตัวนับ)
    const before = disposedOld;
    const r1 = W.runPipeline(img, pl, store, new Set([4242]), memo);
    const r2 = W.runPipeline(img, pl, store, new Set([4242]), memo); // กล่องผลลัพธ์ + พรีวิวในเฟรมเดียว
    const st = store.get(4242);
    const switched = prevMode !== null && prevMode !== mode;
    if (switched) {
      assert.notEqual(st, prevState, `${mode}: fresh state object after switch`);
      assert.equal(disposedOld, before + 1, `${mode}: old state's __dispose must run exactly once`);
    } else if (prevMode !== null) {
      assert.equal(st, prevState, `${mode}: same method must keep its state`);
    }
    const info = r1.notes.find((n) => n.info)?.info || '';
    if (prevMode !== mode) assert.match(info, /เฟรมแรก/, `${mode}: first frame after switch -> ${info}`);
    else assert.doesNotMatch(info, /เฟรมแรก/, `${mode}: second frame must not be first-frame -> ${info}`);
    const fr = (r) => r.notes.find((n) => n.values)?.values?.fields?.find((f) => f.key === 'frame')?.value;
    assert.equal(fr(r1), fr(r2), 'memo: second pipeline in same frame must not re-run');
    prevMode = mode; prevState = st;
    // ครอบ __dispose หลังรันเฟรม (op ตั้ง __dispose ใหม่ทุกเฟรม — ครอบก่อนรันจะถูกเขียนทับ)
    // op ตั้ง __dispose แค่ครั้งแรก (`if (!st.__dispose)`) → ครอบครั้งเดียวต่อ state object
    if (!wrapped.has(st)) { wrapped.add(st); const orig = st.__dispose; st.__dispose = () => { disposedOld++; if (typeof orig === 'function') orig(); }; }
    r1.mat.delete(); r2.mat.delete(); for (const e of memo.values()) e.mat.delete(); img.delete();
  }
  assert.ok(disposedOld >= 3);
  W.sweepNodeState(new Set(), store);
  assert.equal(store.size, 0);
});

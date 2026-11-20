// โค้ด Python ของกล่องผลลัพธ์ที่ต่อผ่าน Model Detect (2026-11-20)
// บั๊กเดิม: กด "สร้างโค้ด Python" ที่จอที่ต่อผ่าน Model Detect → ได้โค้ดที่ไม่มีโมเดล (process_image คืนภาพเดิม)
// เทสต์นี้เทียบฟังก์ชัน Python ที่ gen ออกมากับโค้ดของแอปเอง (dl-decode.js / dl-tiles.js) บนข้อมูลเดียวกัน
// mutant ที่ต้องจับได้:
//   conf ">=" → ">"                 : dc-6 (คะแนนเท่า CONF พอดี), dc-8
//   nms "<" → "<="                  : dc-6 (คู่ที่ IoU เท่าเกณฑ์พอดี)
//   NMS ข้ามคลาส                     : dc-6 (กล่องต่างคลาสซ้อนกัน)
//   round() แทน floor(x+0.5)        : dc-9 (ค่า .5 ใน tile_positions)
//   สลับ dx/dy                       : dc-6, dc-11 (ภาพ 320×180 → dx=0, dy=140)
//   ไม่แปลง BGR → RGB               : dc-12
//   decodeSeg ไม่มี inputSize        : dc-13
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { buildDLPythonCode } from '../src/renderer/codegen-dl.js';
import { buildPythonCode } from '../src/renderer/codegen.js';
import { FILTERS, defaultParams } from '../src/renderer/filters/registry.js';
import { decodeRawHead, nms, decodeNmsBaked, decodeOutputs, decodeSeg } from '../src/renderer/dl-decode.js';
import { tileGrid, mergeTileBoxes } from '../src/renderer/dl-tiles.js';

let python = false;
try { execFileSync('python', ['-c', 'import cv2,numpy'], { stdio: 'pipe' }); python = true; } catch {}
let onnx = false;
try { execFileSync('python', ['-c', 'import onnx,onnxruntime'], { stdio: 'pipe' }); onnx = python; } catch {}

const DL = {
  modelPath: 'C:\\models\\best (8).onnx', modelName: 'best (8).onnx', loaded: true, kind: 'yolo', inputSize: 640,
  labels: ['L', 'Triangle', 'Box'], conf: 0.25, tile: { on: false, size: 640, overlap: 20 },
};
const step = (id, enabled = true) => ({ id, params: { ...defaultParams(id) }, enabled, nodeId: 1 });
const gen = (over = {}, pipeline = [], source) => buildDLPythonCode(pipeline, { displayName: 'Display 1', source, dl: { ...DL, ...over } });

// รันโค้ดที่ gen ด้วย __name__ อื่น (ไม่เข้า main) แล้วต่อด้วยสคริปต์ทดสอบ · ส่งข้อมูลเข้า/ออกเป็น JSON
function runPy(code, script, input, pre = '') {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dl-cg-'));
  try {
    fs.writeFileSync(path.join(dir, 'gen.py'), code, 'utf8');
    fs.writeFileSync(path.join(dir, 'in.json'), JSON.stringify(input ?? null), 'utf8');
    fs.writeFileSync(path.join(dir, 'drv.py'), [
      'import sys, json', pre,
      "ns = {'__name__': 'gen'}",
      "exec(compile(open(sys.argv[1], encoding='utf8').read(), sys.argv[1], 'exec'), ns)",
      "globals().update(ns)",
      "IN = json.load(open(sys.argv[2], encoding='utf8'))",
      script,
    ].join('\n'), 'utf8');
    const out = execFileSync('python', [path.join(dir, 'drv.py'), path.join(dir, 'gen.py'), path.join(dir, 'in.json')],
      { cwd: dir, encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
    return JSON.parse(out.trim().split('\n').pop());
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}

// สุ่มแบบกำหนด seed (mulberry32)
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const f32 = (arr) => Array.from(new Float32Array(arr)); // ข้อมูลจริงจากโมเดลเป็น float32
const sortBoxes = (bs) => [...bs].sort((a, b) => a.cls - b.cls || a.x - b.x || a.y - b.y || b.score - a.score);
function sameBoxes(py, js, tol, msg) {
  assert.equal(py.length, js.length, `${msg}: จำนวนกล่อง Python ${py.length} ≠ แอป ${js.length}`);
  const P = sortBoxes(py), J = sortBoxes(js);
  P.forEach((p, i) => {
    const j = J[i];
    assert.equal(p.cls, j.cls, `${msg}: คลาสกล่อง ${i}`);
    for (const k of ['x', 'y', 'w', 'h', 'score']) assert.ok(Math.abs(p[k] - j[k]) <= tol, `${msg}: ${k} กล่อง ${i} Python ${p[k]} แอป ${j[k]}`);
  });
}
// letterbox ของภาพ 320×180 ที่ 640 (สูตรเดียวกับ dl-yolo.js)
const LB = (() => { const r = Math.min(640 / 320, 640 / 180); const nw = Math.round(320 * r), nh = Math.round(180 * r); return { r, dx: Math.floor((640 - nw) / 2), dy: Math.floor((640 - nh) / 2) }; })();

// raw head [1, 4+3, N] ที่มีกรณีขอบ: คะแนน = CONF พอดี, IoU = 0.45 พอดี (ทั้งคลาสเดียวกัน/ต่างคลาส), คะแนนเท่ากัน
function rawHead(N, seed) {
  const R = rng(seed), C = 7;
  const d = new Array(C * N).fill(0);
  const set = (i, cx, cy, w, h, s) => { d[i] = cx; d[N + i] = cy; d[2 * N + i] = w; d[3 * N + i] = h; for (let c = 0; c < 3; c++) d[(4 + c) * N + i] = s[c]; };
  for (let i = 0; i < N; i++) set(i, 40 + R() * 560, 150 + R() * 340, 10 + R() * 120, 10 + R() * 120, [R(), R(), R()].map((v) => v * v));
  set(0, 300, 300, 100, 100, [0.25, 0.1, 0]); // คะแนน = CONF พอดี (float32 ของ 0.25 แม่นตรง)
  // คู่ IoU = 0.45 พอดี: กล่อง 100×100 กับกล่องกว้าง 45 ที่อยู่ข้างในชิดซ้าย → inter/union = 4500/10000 (ค่าแม่นทุกตัว)
  set(1, 200, 250, 100, 100, [0, 0.9, 0]);
  set(2, 172.5, 250, 45, 100, [0, 0.8, 0]);
  set(3, 200, 250, 100, 100, [0, 0, 0.7]); // ซ้อนกล่อง 1 พอดีแต่คนละคลาส → ต้องอยู่
  set(4, 500, 400, 60, 60, [0.6, 0, 0]);
  set(5, 500, 400, 60, 60, [0.6, 0, 0]); // คะแนนเท่ากัน กล่องเดียวกัน → เหลือกล่องเดียว
  return { dims: [1, C, N], data: f32(d) };
}

// ---------- ข้อความในโค้ด (ไม่ต้องมี Python) ----------
test('dc-1 ตั้งค่าจากกล่อง Model Detect ลงหัวไฟล์ และไม่มี true/false/null ของ JS', () => {
  const code = gen();
  assert.match(code, /^MODEL_PATH = "C:\\\\models\\\\best \(8\)\.onnx"/m);
  assert.match(code, /^NAMES = \["L", "Triangle", "Box"\]/m);
  assert.match(code, /^CONF = 0\.25 /m);
  assert.match(code, /^TILE_ON = False /m);
  assert.match(code, /^TILE_OVERLAP = 0\.2 /m);
  assert.match(code, /^MODEL_KIND = "yolo" /m);
  assert.match(gen({ tile: { on: true, size: 320, overlap: 35 } }), /^TILE_ON = True[\s\S]*^TILE_SIZE = 320[\s\S]*^TILE_OVERLAP = 0\.35 /m);
  assert.match(gen({ kind: null }), /^MODEL_KIND = None /m);
  for (const line of code.split('\n')) {
    const bare = line.replace(/#.*$/, '').replace(/"(?:[^"\\]|\\.)*"/g, '""');
    assert.doesNotMatch(bare, /\b(true|false|null|undefined)\b/, line);
  }
});

test('dc-2 ยังไม่โหลดโมเดล / ไม่รู้ path เต็ม → ช่องให้ใส่เองพร้อมคำอธิบาย', () => {
  const none = gen({ loaded: false, modelPath: null, modelName: null });
  assert.match(none, /^MODEL_PATH = ""   # << ยังไม่ได้โหลดโมเดลในโปรแกรม/m);
  assert.match(none, /ยังไม่ได้ระบุไฟล์โมเดล/);
  const nameOnly = gen({ modelPath: null, modelName: 'x.onnx' });
  assert.match(nameOnly, /^MODEL_PATH = "x\.onnx"   # << ไม่รู้ตำแหน่งไฟล์เต็ม/m);
});

test('dc-3 Box อื่นบนสาย DL ไม่ถูกใส่ (แอปตรวจบนภาพต้นฉบับ) และบอกชื่อไว้', () => {
  const other = Object.keys(FILTERS).find((k) => k !== 'blur' && FILTERS[k].inputs !== 2 && FILTERS[k].name !== FILTERS.blur.name);
  const code = gen({}, [step('blur'), step(other, false)]);
  const note = code.split(/\r?\n/).find((l) => l.startsWith('# หมายเหตุ: ในโปรแกรม กล่อง Model Detect'));
  assert.ok(note && note.includes(`(${FILTERS.blur.name})`), 'ต้องบอกชื่อ Box ที่ไม่ได้ใช้ (เฉพาะตัวที่เปิดอยู่)');
  assert.doesNotMatch(code, /cv2\.(GaussianBlur|blur)\(/);
  assert.match(code, /print\("NOTE: filter boxes on this Model Detect line are not applied/);
  assert.doesNotMatch(gen(), /NOTE: filter boxes/);
});

test('dc-4 แหล่งภาพ: วิดีโอ → ใช้ส่วนรันเดิมแต่ MAX_SIDE = 0 · ไม่มีแหล่ง → แม่แบบ input.jpg', () => {
  const code = gen({}, [], { kind: 'video', maxSide: 720, path: 'C:\\v\\a.mp4', name: 'a.mp4' });
  assert.match(code, /^MAX_SIDE = 0 /m);
  assert.match(code, /cap = cv2\.VideoCapture\(VIDEO_PATH\)/);
  assert.match(code, /^VIDEO_PATH = "C:\\\\v\\\\a\.mp4"/m);
  assert.match(code, /ตรวจบนภาพขนาดจริง \(ไม่ย่อ\)/);
  assert.match(gen(), /result = process\("input\.jpg"\)/);
});

test('dc-5 กล่องที่ไม่ผ่าน Model Detect ใช้ตัว gen เดิม (ไม่มีโมเดล) · app.js เลือกตัว gen ตาม dp.dl', () => {
  for (const p of [[], [step('blur')]]) {
    assert.doesNotMatch(buildPythonCode(p, { displayName: 'D' }), /MODEL_PATH/);
    assert.doesNotMatch(buildPythonCode(p, { displayName: 'D', source: { kind: 'image', paths: ['a.png'] } }), /MODEL_PATH/);
  }
  const app = fs.readFileSync(new URL('../src/renderer/app.js', import.meta.url), 'utf8');
  assert.match(app, /dp && dp\.dl\s*\?\s*buildDLPythonCode\(dp\.pipeline/);
  assert.match(app, /state\.dlModel = \{ path: onnxPath, name: onnxName, kind: modelKind, inputSize \}/);
});

// ---------- Python เทียบกับโค้ดของแอป ----------
test('dc-6 decode_raw_head + nms = decodeRawHead + nms ของแอป (รวมกรณีขอบ)', { skip: !python }, () => {
  const { dims, data } = rawHead(60, 7);
  const js = nms(decodeRawHead(dims, data, LB.r, LB.dx, LB.dy, 0.25), 0.45);
  const py = runPy(gen(), `
d = np.array(IN["data"], np.float32).reshape(IN["dims"])
print(json.dumps(nms(decode_raw_head(d, IN["r"], IN["dx"], IN["dy"], 0.25), 0.45)))`, { dims, data, ...LB });
  sameBoxes(py, js, 1e-4, 'raw head');
  assert.ok(js.some((b) => Math.abs(b.score - 0.25) < 1e-9), 'กล่องคะแนน = CONF ต้องอยู่');
  assert.ok(!js.some((b) => b.cls === 1 && Math.abs(b.w - 22.5) < 1e-9 && Math.abs(b.score - 0.8) < 1e-6), 'IoU = เกณฑ์พอดี ต้องถูกตัด');
  assert.ok(js.some((b) => b.cls === 2 && Math.abs(b.score - 0.7) < 1e-6), 'กล่องต่างคลาสที่ซ้อนกันต้องอยู่');
});

test('dc-7 decode_nms_baked = decodeNmsBaked ของแอป (YOLO26 / end-to-end)', { skip: !python }, () => {
  const R = rng(3), n = 10, d = [];
  for (let i = 0; i < n; i++) { const x = R() * 500, y = 140 + R() * 300; d.push(x, y, x + 10 + R() * 100, y + 10 + R() * 100, i === 0 ? 0.25 : R(), i % 3); }
  const data = f32(d);
  const js = decodeNmsBaked([1, n, 6], data, LB.r, LB.dx, LB.dy, 0.25);
  const py = runPy(gen(), `
o = np.array(IN["data"], np.float32).reshape(1, ${n}, 6)
print(json.dumps(decode_outputs("yolo", {"output0": o}, {"r": IN["r"], "dx": IN["dx"], "dy": IN["dy"], "conf": 0.25, "iou": 0.45}, 640)))`, { data, ...LB });
  sameBoxes(py, js, 1e-4, 'nms-baked');
});

test('dc-8 decode RF-DETR = decodeOutputs("rfdetr") ของแอป (เลือก output box/logit ถูก)', { skip: !python }, () => {
  const R = rng(11), N = 20, C = 3, box = [], logit = [];
  for (let i = 0; i < N; i++) { box.push(0.1 + R() * 0.8, 0.1 + R() * 0.8, 0.05 + R() * 0.3, 0.05 + R() * 0.3); for (let c = 0; c < C; c++) logit.push((R() - 0.6) * 8); }
  logit[0] = 0; logit[1] = -5; logit[2] = -5; // sigmoid(0) = 0.5 = CONF พอดี (ต้องนับ: >=)
  const out = { dets: { dims: [1, N, 4], data: f32(box) }, labels: { dims: [1, N, C], data: f32(logit) } };
  for (const names of [['dets', 'labels'], ['labels', 'dets']]) {
    const js = decodeOutputs('rfdetr', names, out, { origW: 320, origH: 180, conf: 0.5 }, 560);
    const py = runPy(gen({ kind: 'rfdetr' }), `
outs = {n: np.array(IN["out"][n]["data"], np.float32).reshape(IN["out"][n]["dims"]) for n in IN["names"]}
print(json.dumps(decode_outputs("rfdetr", outs, {"orig_w": 320, "orig_h": 180, "conf": 0.5}, 560)))`, { out, names });
    sameBoxes(py, js, 1e-4, `rfdetr ${names}`);
    assert.ok(js.some((b) => b.score === 0.5), 'คะแนน = CONF พอดีต้องนับ');
  }
});

test('dc-9 tile_grid / merge_tile_boxes = tileGrid / mergeTileBoxes ของแอป', { skip: !python }, () => {
  const cases = [[1920, 1080, 640, 0.2], [320, 180, 640, 0.2], [1000, 700, 300, 0.5], [641, 641, 640, 0], [1000, 1000, 400, 0.25]];
  // 1000/400/0.25 → step = round(300) · 0.5 ต้องปัดขึ้น: 300 × (1-0.5) ไม่มี .5 จึงเพิ่มกรณี tile 301 overlap 0.5 → 150.5
  cases.push([1200, 900, 301, 0.5]);
  const R = rng(5), boxes = [];
  for (let i = 0; i < 30; i++) { const x = R() * 600, y = R() * 400; boxes.push({ x, y, w: 20 + R() * 150, h: 20 + R() * 150, score: Math.round(R() * 20) / 20, cls: i % 2 }); }
  const py = runPy(gen(), `
grids = [[list(r) for r in tile_grid(*c)] for c in IN["cases"]]
print(json.dumps({"grids": grids, "merged": merge_tile_boxes([dict(b) for b in IN["boxes"]], 0.5)}))`, { cases, boxes });
  cases.forEach((c, i) => assert.deepEqual(py.grids[i], tileGrid(...c).map((r) => [r.x, r.y, r.w, r.h]), `tileGrid ${c}`));
  sameBoxes(py.merged, mergeTileBoxes(boxes, 0.5), 1e-9, 'merge');
});

test('dc-10 decode_seg = decodeSeg ของแอป (กล่อง + mask ทุกพิกเซล)', { skip: !python }, () => {
  const R = rng(9), N = 40, nc = 2, C = 4 + nc + 32, S = 160, mh = 40, mw = 40;
  const det = new Array(C * N).fill(0);
  for (let i = 0; i < N; i++) {
    det[i] = 20 + R() * 120; det[N + i] = 20 + R() * 120; det[2 * N + i] = 10 + R() * 60; det[3 * N + i] = 10 + R() * 60;
    for (let c = 0; c < nc; c++) det[(4 + c) * N + i] = R();
    for (let k = 0; k < 32; k++) det[(4 + nc + k) * N + i] = (R() - 0.5) * 2;
  }
  const proto = Array.from({ length: 32 * mh * mw }, () => (R() - 0.5) * 2);
  const d32 = f32(det), p32 = f32(proto);
  const js = decodeSeg([1, C, N], d32, [1, 32, mh, mw], p32, 1, 0, 0, 0.25, 0.45, S);
  const py = runPy(gen({ kind: 'yoloseg' }), `
det = np.array(IN["det"], np.float32).reshape(1, ${C}, ${N}); proto = np.array(IN["proto"], np.float32).reshape(1, 32, ${mh}, ${mw})
bs = decode_seg(det, proto, 1, 0, 0, 0.25, 0.45, ${S})
for b in bs: b["mask"] = {"mh": int(b["mask"].shape[0]), "mw": int(b["mask"].shape[1]), "data": b["mask"].ravel().tolist()}
print(json.dumps(bs))`, { det: d32, proto: p32 });
  sameBoxes(py, js, 1e-4, 'seg');
  const P = sortBoxes(py), J = sortBoxes(js);
  P.forEach((p, i) => {
    assert.equal(p.mask.mw, J[i].mask.mw); assert.equal(p.mask.mh, J[i].mask.mh);
    assert.deepEqual(p.mask.data, Array.from(J[i].mask.data), `mask กล่อง ${i}`);
  });
});

test('dc-13 decodeOutputs("yoloseg") ของแอปไม่พัง (เดิม decodeSeg ใช้ inputSize ที่ไม่ได้ประกาศ)', () => {
  const N = 4, C = 4 + 1 + 32;
  const det = new Float32Array(C * N);
  for (let i = 0; i < N; i++) { det[i] = 40 + i * 20; det[N + i] = 40; det[2 * N + i] = 20; det[3 * N + i] = 20; det[4 * N + i] = 0.9; det[(5) * N + i] = 1; }
  const proto = new Float32Array(32 * 40 * 40).fill(1);
  const out = { output0: { dims: [1, C, N], data: det }, output1: { dims: [1, 32, 40, 40], data: proto } };
  const bs = decodeOutputs('yoloseg', ['output0', 'output1'], out, { r: 1, dx: 0, dy: 0, conf: 0.25, iou: 0.45 }, 160);
  assert.equal(bs.length, N);
  assert.ok(bs.every((b) => b.mask && b.mask.mw === 6 && b.mask.mh === 6 && b.mask.data.every((v) => v === 255)));
});

// ---------- end-to-end: ไฟล์ .onnx จริง (โมเดลคืนค่าคงที่) ----------
// สร้างโมเดลด้วย onnx.helper: output0 [1,7,N] = CONST + k × mean(ช่องสีแรกของ input)
// ช่องแรกของ input = R (แอปส่ง RGB) → ภาพแดงล้วนต้องได้คะแนนตามสูตร R ไม่ใช่ B
const MAKE_MODEL = `
import onnx, numpy as np
from onnx import helper, TensorProto, numpy_helper
def make_model(path, const, k_index, k):
    C, N = const.shape[1], const.shape[2]
    kk = np.zeros_like(const); kk.flat[k_index] = k
    nodes = [
        helper.make_node("Slice", ["images", "s0", "s1", "ax"], ["ch0"]),
        helper.make_node("ReduceMean", ["ch0"], ["m"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Mul", ["m", "kk"], ["mk"]),
        helper.make_node("Add", ["cst", "mk"], ["output0"]),
    ]
    inits = [numpy_helper.from_array(np.array([0], np.int64), "s0"), numpy_helper.from_array(np.array([1], np.int64), "s1"),
             numpy_helper.from_array(np.array([1], np.int64), "ax"), numpy_helper.from_array(const.astype(np.float32), "cst"),
             numpy_helper.from_array(kk.astype(np.float32), "kk")]
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
                          [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, C, N])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    onnx.save(m, path)
`;

function e2e(force_cv2dnn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ทดสอบ model '));
  try {
    const modelPath = path.join(dir, 'โมเดล ทดสอบ.onnx');
    const { dims, data } = rawHead(100, 21);
    const code = gen({ modelPath });
    const pre = force_cv2dnn ? "sys.modules['onnxruntime'] = None" : '';
    const py = runPy(code, `${MAKE_MODEL}
make_model(IN["path"], np.array(IN["data"], np.float32).reshape(IN["dims"]), 4 * 100 + 6, 0.0)
img = np.zeros((180, 320, 3), np.uint8); img[:] = (40, 90, 160)
res = process_image(img)
assert res.shape == img.shape and res.dtype == np.uint8
bs = detect(img)
print(json.dumps({"boxes": bs, "engine": load_model()["engine"], "changed": int((res != img).any(axis=2).sum())}))`, { path: modelPath, dims, data }, pre);
    const js = nms(decodeRawHead(dims, data, LB.r, LB.dx, LB.dy, 0.25), 0.45);
    sameBoxes(py.boxes, js, 1e-3, force_cv2dnn ? 'cv2.dnn' : 'onnxruntime');
    assert.equal(py.engine, force_cv2dnn ? 'cv2.dnn' : 'onnxruntime');
    assert.ok(py.changed > 0, 'ต้องวาดกรอบลงภาพ');
    assert.ok(py.boxes.every((b) => ['L', 'Triangle', 'Box'][b.cls] === b.label));
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}
test('dc-11 end-to-end onnxruntime: path ภาษาไทย+เว้นวรรค · กล่องตรงกับแอป · วาดกรอบลงภาพ', { skip: !onnx }, () => e2e(false));
test('dc-11b end-to-end ไม่มี onnxruntime → ใช้ cv2.dnn ได้กล่องเหมือนกัน', { skip: !onnx }, () => e2e(true));

test('dc-12 ส่งภาพเข้าโมเดลเป็น RGB (ภาพแดงล้วน → ช่องแรก = แดง)', { skip: !onnx }, () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dl-rgb-'));
  try {
    const modelPath = path.join(dir, 'm.onnx');
    // anchor เดียว คลาส 0: คะแนน = 0 + 1 × mean(ช่องแรก) · ภาพแดงล้วน 320×180 → letterbox 640×360 + ขอบเทา 114
    const N = 1, d = [320, 320, 100, 100, 0, 0, 0];
    const py = runPy(gen({ modelPath, conf: 0.05 }), `${MAKE_MODEL}
make_model(IN["path"], np.array(IN["d"], np.float32).reshape(1, 7, 1), 4, 1.0)
img = np.zeros((180, 320, 3), np.uint8); img[:] = (0, 0, 255)
print(json.dumps(detect(img)))`, { path: modelPath, d }, '');
    const area = 640 * 640, inside = 640 * 360;
    const meanR = (inside * 1 + (area - inside) * (114 / 255)) / area; // ช่องแรก = R
    assert.equal(py.length, N);
    assert.ok(Math.abs(py[0].score - meanR) < 2e-3,  // ReduceMean ของ onnxruntime สะสมแบบ float32 · ถ้าส่ง BGR จะได้ ~0.196
      `score ${py[0].score} ต้อง = mean ช่อง R ${meanR}`);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

// ---------- เลือก CPU / GPU ตอนรัน (2026-11-20) ----------
test('dc-14 มีตัวเลือก DEVICE = "auto" พร้อมวิธีลง onnxruntime รุ่น GPU', () => {
  const code = gen();
  assert.match(code, /^DEVICE = "auto"   # "auto" = GPU ถ้ามี/m);
  assert.match(code, /onnxruntime-directml/);
  assert.doesNotMatch(code, /providers=\["CPUExecutionProvider"\]\)/, 'ห้ามล็อก CPU อย่างเดียวแบบเดิม');
});

// onnxruntime ปลอม: บอกว่ามี provider อะไร แล้วจดว่า InferenceSession ถูกเรียกด้วย provider/option อะไร
const FAKE_ORT = (avail) => `
import sys, types
ort = types.ModuleType("onnxruntime")
ort.get_available_providers = lambda: ${JSON.stringify(avail)}
class _SO: pass
ort.SessionOptions = _SO
ort.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL="SEQ")
CALLS = []
class _IO:
    def __init__(s, n, sh): s.name, s.shape = n, sh
class _Sess:
    def __init__(s, data, so=None, providers=None):
        CALLS.append({"providers": providers, "mem": getattr(so, "enable_mem_pattern", None), "mode": getattr(so, "execution_mode", None)})
        s._p = providers
    def get_inputs(s): return [_IO("images", [1, 3, 640, 640])]
    def get_outputs(s): return [_IO("output0", [1, 7, 8400])]
    def get_providers(s): return s._p
ort.InferenceSession = _Sess
sys.modules["onnxruntime"] = ort
`;
function device(choice, avail, cv2dnn = false) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dl-dev-'));
  try {
    const modelPath = path.join(dir, 'm.onnx');
    fs.writeFileSync(modelPath, 'x');
    const code = gen({ modelPath }).replace(/^DEVICE = "auto"/m, `DEVICE = "${choice}"`);
    return runPy(code, `
try:
    m = load_model()
    print(json.dumps({"engine": m["engine"], "calls": CALLS if "CALLS" in globals() else []}))
except RuntimeError as e:
    print(json.dumps({"raised": str(e)}))`, null, cv2dnn ? "sys.modules['onnxruntime'] = None\nCALLS = []" : FAKE_ORT(avail));
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}
test('dc-15 DEVICE: auto ใช้ GPU ถ้ามี (DirectML ตั้ง option ถูก) · cpu บังคับ CPU · gpu ไม่มี GPU = แจ้งชัด', { skip: !python }, () => {
  const DML = ['DmlExecutionProvider', 'CPUExecutionProvider'], CPU = ['CPUExecutionProvider'];
  const auto = device('auto', DML).calls[0];
  assert.deepEqual(auto.providers, ['DmlExecutionProvider', 'CPUExecutionProvider']);
  assert.equal(auto.mem, false); assert.equal(auto.mode, 'SEQ');
  assert.deepEqual(device('auto', CPU).calls[0].providers, CPU);
  assert.deepEqual(device('cpu', DML).calls[0].providers, CPU);
  assert.deepEqual(device('gpu', ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']).calls[0].providers, ['CUDAExecutionProvider', 'CPUExecutionProvider']);
  assert.match(device('gpu', CPU).raised, /ไม่มี GPU/);
  assert.match(device('gpu', null, true).raised, /cv2\.dnn รันบน CPU เท่านั้น/);
});

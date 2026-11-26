// กล่องหลัง Model Detect (2026-11-26): 📍 Centroid + 🔢 Object Count (เส้นนับ / โซนนับ)
// ตรวจ: ตรรกะในแอป (dl-post.js) ถูกตามนิยาม + โค้ด Python ที่ gen (codegen-dl.js) ได้ผลเดียวกันทุกเฟรม
// mutant ที่ต้องจับได้:
//   นับซ้ำชิ้นเดิมเมื่อแกว่งข้ามเส้น          : dp-3
//   สลับทิศ IN/OUT                           : dp-2, dp-8
//   นับข้ามเส้นตรงนอกช่วง A–B                 : dp-4
//   โซนนับชิ้นเดิมซ้ำทุกเฟรม                    : dp-5
//   ภาพนิ่งติดตามข้ามภาพ                       : dp-6
//   Otsu เลือกพื้นหลังเป็นชิ้นงาน (กลับสี)       : dp-7
//   Python ใช้ round()/cv2 gray ต่างจากแอป      : dp-8, dp-9
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { buildDLPythonCode } from '../src/renderer/codegen-dl.js';
import {
  createCounter, createTracker, computeCentroids, shapeCentroidRGBA, otsuThreshold, normCount, countKey, counterName, counterText,
} from '../src/renderer/dl-post.js';

let python = false;
try { execFileSync('python', ['-c', 'import cv2,numpy'], { stdio: 'pipe' }); python = true; } catch {}

const W = 320, H = 180;
const pt = (x, y, label = 'box', cls = 2) => ({ x, y, label, cls });
// ชิ้นเดียววิ่งตามเส้นทาง → คืน state สุดท้าย
function runPath(params, path, still = false) {
  const c = createCounter();
  let st;
  for (const p of path) st = c.update(p ? [p] : [], params, W, H, still);
  return st;
}
const VLINE = { mode: 'line', x1: 50, y1: 0, x2: 50, y2: 100 }; // เส้นตั้งกลางภาพ A บน → B ล่าง

test('dp-1 tracker: ชิ้นเดิมได้ id เดิม · ชิ้นใหม่ได้ id ใหม่ · หายเกิน maxMissed = ลืม', () => {
  const t = createTracker();
  t.update([pt(10, 10), pt(100, 100)], 30, 2);
  let tr = t.update([pt(105, 102), pt(14, 11)], 30, 2);
  assert.deepEqual(tr.map((x) => [x.id, Math.round(x.x)]), [[1, 14], [2, 105]]);
  t.update([], 30, 2); t.update([], 30, 2);
  tr = t.update([], 30, 2); // หาย 3 ครั้ง > 2
  assert.equal(tr.length, 0);
  tr = t.update([pt(14, 11)], 30, 2);
  assert.equal(tr[0].id, 3);
});

test('dp-2 เส้นนับ: ซ้าย→ขวา ข้ามเส้นตั้ง A บน→B ล่าง = IN (ลูกศรชี้ขวา) · ขวา→ซ้าย = OUT', () => {
  const lr = runPath(VLINE, [pt(100, 90), pt(150, 90), pt(170, 90)]);
  assert.equal(lr.in, 1); assert.equal(lr.out, 0); assert.equal(lr.total, 1);
  const rl = runPath(VLINE, [pt(220, 90), pt(170, 90), pt(150, 90)]);
  assert.equal(rl.in, 0); assert.equal(rl.out, 1);
  // dir = in → ไม่นับ OUT
  assert.equal(runPath({ ...VLINE, dir: 'in' }, [pt(180, 90), pt(150, 90)]).total, 0);
  assert.equal(runPath({ ...VLINE, dir: 'out' }, [pt(180, 90), pt(150, 90)]).total, 1);
});

test('dp-3 เส้นนับ: แกว่งข้ามเส้นไปมา = นับชิ้นละครั้ง · จุดที่ตกบนเส้นพอดีไม่ทำให้พลาด', () => {
  const st = runPath(VLINE, [pt(150, 90), pt(170, 90), pt(150, 90), pt(170, 90), pt(150, 90)]);
  assert.equal(st.total, 1);
  const on = runPath(VLINE, [pt(150, 90), pt(160, 90), pt(170, 90)]); // 160 = บนเส้น
  assert.equal(on.in, 1);
});

test('dp-4 เส้นนับ: ข้ามเส้นตรงแต่นอกช่วง A–B ไม่นับ · กรองคลาส', () => {
  const short = { mode: 'line', x1: 50, y1: 0, x2: 50, y2: 40 }; // เส้นครึ่งบน (y 0–72)
  assert.equal(runPath(short, [pt(150, 150), pt(170, 150)]).total, 0);
  assert.equal(runPath(short, [pt(150, 30), pt(170, 30)]).total, 1);
  assert.equal(runPath({ ...VLINE, cls: 'L' }, [pt(150, 90, 'box'), pt(170, 90, 'box')]).total, 0);
  const st = runPath({ ...VLINE, cls: 'L' }, [pt(150, 90, 'L', 0), pt(170, 90, 'L', 0)]);
  assert.deepEqual(st.byClass, { L: 1 });
});

test('dp-5 โซนนับ: NOW = ชิ้นในกรอบตอนนี้ · TOTAL = ชิ้นที่เคยเข้า (ชิ้นละครั้ง)', () => {
  const zone = { mode: 'zone', zx: 25, zy: 0, zw: 50, zh: 100 }; // x 80–240
  const c = createCounter();
  const seq = [
    [pt(40, 50), pt(40, 120, 'L', 0)],
    [pt(90, 50), pt(60, 120, 'L', 0)],
    [pt(110, 50), pt(85, 120, 'L', 0)],
    [pt(130, 50), pt(110, 120, 'L', 0)],
    [pt(260, 50), pt(130, 120, 'L', 0)],
  ];
  const nows = seq.map((s) => c.update(s, zone, W, H).now);
  assert.deepEqual(nows, [0, 1, 2, 2, 1]);
  assert.equal(c.state.total, 2);
  assert.deepEqual(c.state.byClass, { box: 1, L: 1 });
  assert.equal(counterText('Zone 1', zone, c.state), 'Zone 1: NOW 1  TOTAL 2');
});

test('dp-6 ภาพนิ่ง: นับใหม่ทุกภาพ (ไม่ติดตามข้ามภาพ) · เส้นนับในภาพนิ่ง = 0', () => {
  const zone = { mode: 'zone', zx: 0, zy: 0, zw: 100, zh: 100 };
  const c = createCounter();
  c.update([pt(10, 10), pt(50, 50)], zone, W, H, true);
  const st = c.update([pt(10, 10)], zone, W, H, true);
  assert.equal(st.now, 1);
  assert.equal(st.total, 1);
  assert.equal(runPath(VLINE, [pt(150, 90), pt(170, 90)], true).total, 0);
});

test('dp-6b ตรวจช้า (ข้ามเฟรม ~6–24 เฟรม เหมือน CPU/WASM): ทำนายตำแหน่งจากความเร็ว → ยังนับเท่าตรวจทุกเฟรม', () => {
  // 4 ชิ้นวิ่งซ้าย→ขวา 6 px/เฟรม ห่างกันแนวตั้ง · เส้นนับตั้งกลางภาพ
  const objs = [0, 1, 2, 3].map((i) => ({ t0: i * 40, y: 30 + i * 35 }));
  const at = (f) => objs.filter((o) => f >= o.t0).map((o) => pt(-20 + (f - o.t0) * 6, o.y)).filter((p) => p.x <= 330);
  const count = (gaps) => {
    const c = createCounter();
    let f = 0, i = 0, prev = 0;
    while (f < 260) { c.update(at(f), VLINE, W, H, false, (f - prev || 1) / 30); prev = f; f += gaps[i++ % gaps.length]; }
    return c.state.total;
  };
  assert.equal(count([1]), 4);
  assert.equal(count([6, 24, 7, 20, 6]), 4); // ช่วงตรวจ 200–800 ms ที่วัดได้จริงใน WASM
});

// ภาพ RGBA: พื้นสีเขียวเข้ม + รูปตัว L สีขาว
function lImage(w, h, invert = false) {
  const d = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const inL = (x >= 4 && x < 12 && y >= 4 && y < 36) || (x >= 4 && x < 30 && y >= 28 && y < 36);
    const fg = invert ? !inL : inL;
    const i = (y * w + x) * 4;
    d[i] = fg ? 235 : 20; d[i + 1] = fg ? 235 : 70; d[i + 2] = fg ? 230 : 50; d[i + 3] = 255;
  }
  return d;
}
function lTruth() {
  let n = 0, sx = 0, sy = 0;
  for (let y = 0; y < 40; y++) for (let x = 0; x < 34; x++) {
    if ((x >= 4 && x < 12 && y >= 4 && y < 36) || (x >= 4 && x < 30 && y >= 28 && y < 36)) { n++; sx += x + 0.5; sy += y + 0.5; }
  }
  return { x: sx / n, y: sy / n };
}

test('dp-7 centroid รูปทรง: ตัว L = จุดศูนย์ถ่วงของเนื้อจริง (ไม่ใช่กลางกรอบ) · ชิ้นงานมืดบนพื้นสว่างก็ได้', () => {
  const truth = lTruth();
  const c = shapeCentroidRGBA(lImage(34, 40), 34, 40);
  assert.ok(Math.abs(c.x - truth.x) < 1e-9 && Math.abs(c.y - truth.y) < 1e-9, JSON.stringify(c));
  // ชิ้นงานมืดบนพื้นสว่าง (กลับสี) → ยังได้เนื้อชิ้นงานเดิม (กลุ่มที่แตะขอบน้อยกว่า)
  const inv = new Uint8ClampedArray(34 * 40 * 4);
  const src = lImage(34, 40);
  for (let i = 0; i < src.length; i += 4) { inv[i] = 255 - src[i]; inv[i + 1] = 255 - src[i + 1]; inv[i + 2] = 255 - src[i + 2]; inv[i + 3] = 255; }
  const ci = shapeCentroidRGBA(inv, 34, 40);
  assert.ok(Math.abs(ci.x - truth.x) < 1e-9 && Math.abs(ci.y - truth.y) < 1e-9);
  // ภาพสีเดียว = แยกไม่ได้ → computeCentroids ถอยไปกลางกรอบ
  const flat = new Uint8ClampedArray(10 * 10 * 4).fill(128);
  const [p] = computeCentroids([{ x: 0, y: 0, w: 10, h: 10, cls: 0, label: 'L', score: 0.9 }], { method: 'shape' }, 10, 10, () => flat);
  assert.deepEqual([p.x, p.y, p.method], [5, 5, 'box']);
  // mask ของ seg ชนะ Otsu
  const mask = { mw: 2, mh: 2, data: new Uint8Array([255, 0, 0, 0]) };
  const [m] = computeCentroids([{ x: 0, y: 0, w: 10, h: 10, cls: 0, label: 'L', score: 0.9, mask }], { method: 'shape' }, 10, 10, () => flat);
  assert.deepEqual([m.x, m.y, m.method], [2.5, 2.5, 'mask']);
  // Otsu: histogram สองยอด → เกณฑ์อยู่ระหว่างยอด
  const hist = new Float64Array(256); hist[20] = 50; hist[200] = 50;
  const t = otsuThreshold(hist, 100);
  assert.ok(t >= 20 && t < 200);
});

// ---------- เทียบกับ Python ที่ gen ----------
function genCode(post, source) {
  return buildDLPythonCode([], {
    displayName: 'Display 1', source,
    dl: { modelPath: 'C:\\m\\best.onnx', loaded: true, kind: 'yolo', inputSize: 640, labels: ['L', 'Triangle', 'box'], conf: 0.25, tile: { on: false, size: 640, overlap: 20 } },
    post,
  });
}
function runPy(code, script, input) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dl-post-'));
  try {
    fs.writeFileSync(path.join(dir, 'gen.py'), code, 'utf8');
    fs.writeFileSync(path.join(dir, 'in.json'), JSON.stringify(input ?? null), 'utf8');
    fs.writeFileSync(path.join(dir, 'drv.py'), [
      'import sys, json, atexit',
      "ns = {'__name__': 'gen'}",
      "exec(compile(open(sys.argv[1], encoding='utf8').read(), sys.argv[1], 'exec'), ns)",
      "globals().update(ns)",
      "atexit.unregister(print_summary)",
      "IN = json.load(open(sys.argv[2], encoding='utf8'))",
      script,
    ].join('\n'), 'utf8');
    const out = execFileSync('python', [path.join(dir, 'drv.py'), path.join(dir, 'gen.py'), path.join(dir, 'in.json')],
      { cwd: dir, encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
    return JSON.parse(out.trim().split('\n').pop());
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

test('dp-8 Counter ของ Python = createCounter ของแอป ทุกเฟรม (เส้น 3 แบบ + โซน · สุ่มชิ้นงานวิ่ง/หาย/สะดุด)', { skip: !python }, () => {
  const counters = [
    { mode: 'line', x1: 50, y1: 0, x2: 50, y2: 100 },
    { mode: 'line', x1: 20, y1: 70, x2: 90, y2: 30, dir: 'in', maxDist: 8 },
    { mode: 'line', x1: 60, y1: 10, x2: 60, y2: 60, dir: 'out', cls: 'box', maxMissed: 3 },
    { mode: 'zone', zx: 30, zy: 20, zw: 35, zh: 50 },
  ];
  const r = rng(7);
  // ชิ้นงาน 12 ชิ้นวิ่งผ่านภาพ (บางชิ้นวิ่งย้อน) + ตรวจหลุดบางเฟรม + จุดสั่น
  const objs = Array.from({ length: 12 }, (_, i) => ({
    t0: i * 25, x: r() < 0.3 ? 330 : -10, y: 20 + r() * 140, vx: (2 + r() * 4) * (r() < 0.3 ? -1 : 1), vy: (r() - 0.5) * 1.5,
    label: ['L', 'Triangle', 'box'][i % 3], cls: i % 3,
  }));
  objs.forEach((o) => { if (o.x > 300) o.vx = -Math.abs(o.vx); else o.vx = Math.abs(o.vx); });
  const frames = [];
  for (let f = 0; f < 420; f++) {
    const pts = [];
    for (const o of objs) {
      const k = f - o.t0;
      if (k < 0) continue;
      const x = o.x + o.vx * k + (r() - 0.5) * 3, y = o.y + o.vy * k + (r() - 0.5) * 3;
      if (x < -20 || x > 340 || r() < 0.08) continue; // นอกภาพ / ตรวจหลุด
      pts.push({ x, y, cls: o.cls, label: o.label });
    }
    frames.push(pts);
  }
  // ครึ่งแรกตรวจทุกเฟรม · ครึ่งหลังตรวจข้ามเฟรมแบบสุ่ม (เหมือนเครื่องช้า) → dt ไม่คงที่
  const seq = [];
  for (let f = 0, prev = -1; f < frames.length;) {
    seq.push({ pts: frames[f], dt: (f - prev) / 30 }); // dt = เวลาตั้งแต่ตรวจครั้งก่อน
    prev = f;
    f += f < 210 ? 1 : 1 + Math.floor(r() * 8);
  }
  const post = counters.map((p, i) => ({ type: 'count', params: p, nodeId: 10 + i }));
  const js = counters.map(() => createCounter());
  const jsOut = seq.map(({ pts, dt }) => js.map((c, i) => {
    const st = c.update(pts.map((p) => ({ ...p })), counters[i], W, H, false, dt);
    return [st.in, st.out, st.total, st.now, st.events.map((e) => e.id)];
  }));
  const pyOut = runPy(genCode(post, { kind: 'video', maxSide: 0, path: 'v.mp4' }), [
    'res = []',
    'for fr in IN:',
    '    pts = fr["pts"]',
    '    row = []',
    '    for k in COUNTER_OBJS:',
    '        k.update([dict(p) for p in pts], 320, 180, False, fr["dt"])',
    '        row.append([k.n_in, k.n_out, k.total, k.now, [e[0] for e in k.events]])',
    '    res.append(row)',
    'print(json.dumps(res))',
  ].join('\n'), seq);
  assert.deepEqual(pyOut, jsOut);
  const final = jsOut[jsOut.length - 1];
  assert.ok(final[0][2] >= 5, 'เส้นกลางภาพควรนับได้หลายชิ้น: ' + JSON.stringify(final));
  assert.ok(final[3][2] >= 3, 'โซนควรนับได้หลายชิ้น');
});

test('dp-9 compute_centroids ของ Python = computeCentroids ของแอป (Otsu + moments ทีละพิกเซล, ภาพสุ่ม)', { skip: !python }, () => {
  const r = rng(11);
  const iw = 96, ih = 64;
  const rgba = new Uint8ClampedArray(iw * ih * 4);
  for (let y = 0; y < ih; y++) for (let x = 0; x < iw; x++) {
    const i = (y * iw + x) * 4;
    const blob = Math.hypot(x - 30, (y - 30) * 1.4) < 14 || (x > 55 && x < 85 && y > 10 && y < 22) || (x > 60 && x < 68 && y >= 22 && y < 55);
    const base = blob ? 200 : 40;
    rgba[i] = base + r() * 40; rgba[i + 1] = base - 10 + r() * 50; rgba[i + 2] = base + r() * 30; rgba[i + 3] = 255;
  }
  const boxes = [
    { x: 14.3, y: 18.6, w: 32.2, h: 24.9, cls: 0, label: 'L', score: 0.9 },
    { x: 52.7, y: 6.1, w: 36.5, h: 52.4, cls: 1, label: 'Triangle', score: 0.8 },
    { x: 90.2, y: 60.5, w: 20, h: 20, cls: 2, label: 'box', score: 0.7 }, // ล้นขอบ
    { x: 3, y: 3, w: 1.2, h: 30, cls: 2, label: 'box', score: 0.7 }, // แคบเกิน → กลางกรอบ
  ];
  const read = (rc) => {
    const out = new Uint8ClampedArray(rc.w * rc.h * 4);
    for (let y = 0; y < rc.h; y++) out.set(rgba.subarray(((rc.y0 + y) * iw + rc.x0) * 4, ((rc.y0 + y) * iw + rc.x0 + rc.w) * 4), y * rc.w * 4);
    return out;
  };
  const js = computeCentroids(boxes, { method: 'shape' }, iw, ih, read).map((p) => [p.x, p.y]);
  const bgr = [];
  for (let i = 0; i < rgba.length; i += 4) bgr.push(rgba[i + 2], rgba[i + 1], rgba[i]);
  const py = runPy(genCode([{ type: 'centroid', params: { method: 'shape' }, nodeId: 1 }]), [
    'img = np.array(IN["bgr"], np.uint8).reshape(IN["h"], IN["w"], 3)',
    'pts = compute_centroids(img, [dict(b) for b in IN["boxes"]])',
    'print(json.dumps([[p["x"], p["y"]] for p in pts]))',
  ].join('\n'), { bgr, w: iw, h: ih, boxes });
  assert.equal(py.length, js.length);
  py.forEach((p, i) => {
    assert.ok(Math.abs(p[0] - js[i][0]) < 1e-9 && Math.abs(p[1] - js[i][1]) < 1e-9, `box ${i}: py ${p} js ${js[i]}`);
  });
  assert.notDeepEqual(js[0], [14.3 + 16.1, 18.6 + 12.45]); // ใช้รูปทรงจริง ไม่ใช่กลางกรอบ
});

test('dp-10 โค้ดที่ gen: ตั้งค่าตรงกล่อง · ภาพนิ่ง STILL=True · ไม่มีกล่องหลัง DL = โค้ดเดิม (ไม่มีตัวนับ)', () => {
  const post = [
    { type: 'centroid', params: { method: 'box', showCoords: false }, nodeId: 1 },
    { type: 'count', params: { mode: 'zone', zx: 10, zy: 11, zw: 12, zh: 13, cls: 'L' }, nodeId: 2 },
    { type: 'count', params: { mode: 'line', x1: 1, y1: 2, x2: 3, y2: 4, dir: 'out' }, nodeId: 3 },
  ];
  const code = genCode(post, { kind: 'video', maxSide: 0, path: 'v.mp4' });
  assert.match(code, /CENTROID_ON = True/);
  assert.match(code, /CENTROID_METHOD = "box"/);
  assert.match(code, /SHOW_COORDS = False/);
  assert.match(code, /STILL = False/);
  assert.match(code, /"name": "Zone 1 \[L\]", "mode": "zone"/); // กรองคลาส → ชื่อมีชนิดต่อท้าย
  assert.match(code, /"zx": 10, "zy": 11, "zw": 12, "zh": 13/);
  assert.match(code, /"cls": "L"/);
  assert.match(code, /"name": "Line 1", "mode": "line"/);
  assert.match(code, /"x1": 1, "y1": 2, "x2": 3, "y2": 4, "dir": "out"/);
  assert.match(genCode(post, { kind: 'image', maxSide: 0, paths: ['a.jpg'], missing: 0 }), /STILL = True/);
  const plain = genCode([], { kind: 'video', maxSide: 0, path: 'v.mp4' });
  assert.doesNotMatch(plain, /COUNTERS|atexit|class Counter/);
  assert.match(plain, /def process_image\(img\):\n    # ตรวจด้วยโมเดล/);
  assert.equal(counterName(normCount({ mode: 'zone' }), 2), 'Zone 2');
});

test('dp-11 ขนาดตัวหนังสือ: เปลี่ยนแล้วไม่เริ่มนับใหม่ (countKey เท่าเดิม) · ติดไปในโค้ด Python · บีบช่วง 30–200%', { skip: !python }, () => {
  assert.equal(countKey({ ...VLINE, fontPct: 40 }), countKey(VLINE));
  assert.notEqual(countKey({ ...VLINE, x1: 40 }), countKey(VLINE));
  assert.equal(normCount({ fontPct: 5 }).fontPct, 30);
  assert.equal(normCount({ fontPct: 999 }).fontPct, 200);
  const code = genCode([
    { type: 'centroid', params: { fontPct: 60 }, nodeId: 1 },
    { type: 'count', params: { ...VLINE, fontPct: 50 }, nodeId: 2 },
  ], { kind: 'video', maxSide: 0, path: 'v.mp4' });
  assert.match(code, /CENTROID_FONT = 0\.6/);
  assert.match(code, /"font": 0\.5\}/);
  // วาดได้จริงทั้งขนาดเล็กสุด/ใหญ่สุด (ไม่ error ใน cv2.putText)
  const out = runPy(code, [
    'img = np.zeros((180, 320, 3), np.uint8)',
    'for f in (0.3, 2.0):',
    '    COUNTER_OBJS[0].cfg["font"] = f',
    '    globals()["CENTROID_FONT"] = f',
    '    COUNTER_OBJS[0].update([{"x": 100.0, "y": 90.0, "cls": 0, "label": "L"}], 320, 180)',
    '    draw_centroids(img, [{"x": 100.0, "y": 90.0, "label": "L"}], 1.0)',
    '    draw_counters(img, 1.0)',
    'print(json.dumps(int(img.sum() > 0)))',
  ].join('\n'));
  assert.equal(out, 1);
});

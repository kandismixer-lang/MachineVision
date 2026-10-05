// P-14 — พิสูจน์ว่า "ด่าน leak" ใน tests/_leak-helpers.mjs ทำงานจริง (ทั้งจับได้ และไม่ false-positive)
// ไฟล์นี้ไม่ได้ทดสอบ op ของแอป แต่ทดสอบ "เครื่องมือทดสอบ" เอง — ถ้าเทสต์ในไฟล์นี้ fail แปลว่า leak probe ทุกไฟล์
// ที่ใช้ helper เดียวกันเชื่อถือไม่ได้ ต้องแก้ helper ก่อน
//  gates-1  helper ไม่ถูก glob `tests/*.test.mjs` หยิบไปรันเป็นเทสต์
//  gates-2  tracker จับ Mat ที่ลืม .delete() (constructor / static factory / matFromArray / clone / roi)
//  gates-3  tracker จับ wrapper จาก MatVector.get() ที่ลืม .delete()
//  gates-4  tracker ครอบคลุม embind class ที่แอปใช้: MatVector, KeyPointVector, ORB, AKAZE, FastFeatureDetector, CLAHE,
//           BackgroundSubtractorMOG2 (รอบหน้าต้องใช้) — ทั้งเคสลืม delete (จับได้) และเคสครบ (ผ่าน)
//  gates-5  โหมด global เห็นโค้ดที่ใช้ cv ตัวจริง และ dispose คืนทุกอย่างเป๊ะ
//  gates-6  census จับ `_malloc` ที่ไม่ free (leak native ล้วน — tracker มองไม่เห็น)
//  gates-7  census จับ Mat ที่รั่วผ่าน cv ตรง ๆ (นอกสายตา tracker)
//  gates-8  control ไม่รั่ว + heavy workload ≥10 รอบ: census/tracker ไม่ trip (fingerprint แบบเทียบที่อยู่ trip — บันทึกเป็น diagnostic)
//  gates-11 C++ exception (cv::Exception) ที่ catch ฝั่ง JS รั่ว ~760B/ครั้งใน opencv.js เอง — census เทียบ baseline
//           (assertNoExtraHeapGrowth) ไม่ trip กับ op ที่เก็บกวาดครบ แต่ trip เมื่อ op รั่ว Mat เพิ่มบน throw path
//  gates-9  เคสจริง: ORB/AKAZE .detect() throw บนภาพ <3px ผ่าน cv ตรง ๆ — census จับได้, tracker มองไม่เห็น
//           (วางท้ายไฟล์โดยตั้งใจ: throw สะสมหลายร้อยครั้งทำ heap ของ opencv.js พังได้ — ดู WORKLOG P-14 measure)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  makeWrapperTracker, assertWrapperBalance, assertNoHeapGrowth, assertNoExtraHeapGrowth, assertLeakFree, heapCensus,
  makeNativeLeakControl, makeMatLeakControl, heavyWorkload, fingerprint, makeThrowingCv,
} from './_leak-helpers.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function isReady(c) { return c && typeof c.Mat === 'function' && typeof c.cvtColor === 'function'; }
async function waitForCv() {
  if (isReady(cv)) return;
  await new Promise((resolve) => {
    const poll = setInterval(() => { if (isReady(cv)) { clearInterval(poll); resolve(); } }, 50);
  });
}
await waitForCv();

// ภาพที่มีวัตถุหลายชิ้น (findContours ได้หลาย contour)
function blobs() {
  const m = new cv.Mat.zeros(60, 80, cv.CV_8UC1);
  for (let i = 0; i < 4; i++) cv.rectangle(m, new cv.Point(5 + i * 18, 10), new cv.Point(15 + i * 18, 30), new cv.Scalar(255), -1);
  return m;
}
// ลวดลายให้ ORB/AKAZE/FAST เจอ keypoint
function textured(w, h) {
  const m = new cv.Mat.zeros(h, w, cv.CV_8UC1);
  for (let y = 10; y < h - 10; y += 24) for (let x = 10; x < w - 10; x += 24) {
    cv.rectangle(m, new cv.Point(x, y), new cv.Point(x + 10, y + 8), new cv.Scalar(255), -1);
  }
  return m;
}
// assert ว่า gate "fail" ด้วย AssertionError ที่ข้อความตรง pattern (ไม่ใช่ error อื่นแบบบังเอิญ)
function assertGateTrips(fn, pattern) {
  assert.throws(fn, (err) => err instanceof assert.AssertionError && pattern.test(err.message));
}

test('gates-1. _leak-helpers.mjs exists and is NOT matched by the tests/*.test.mjs glob (never run as a test file)', () => {
  const files = fs.readdirSync(__dirname);
  assert.ok(files.includes('_leak-helpers.mjs'));
  const globbed = files.filter((f) => /^[^/\\]*\.test\.mjs$/.test(f));
  assert.ok(!globbed.includes('_leak-helpers.mjs'), 'helper must not match *.test.mjs');
  assert.ok(globbed.includes('leak-gates.test.mjs'));
});

test('gates-2. wrapper tracker catches a forgotten Mat .delete() — new Mat, new Mat.zeros, matFromArray, clone(), roi()', () => {
  const base = new cv.Mat(10, 10, cv.CV_8UC1, new cv.Scalar(7));
  const cases = {
    'new Mat': (c) => { new c.Mat(4, 4, c.CV_8UC1); },
    'Mat.zeros()': (c) => { new c.Mat.zeros(4, 4, c.CV_8UC1); },
    'cv.matFromArray()': (c) => { c.matFromArray(2, 2, c.CV_8UC1, [1, 2, 3, 4]); },
    'Mat.clone()': () => { base.clone(); },
    'Mat.roi()': (c) => { base.roi(new c.Rect(1, 1, 3, 3)); },
  };
  for (const [origin, leak] of Object.entries(cases)) {
    const t = makeWrapperTracker(cv);
    try {
      leak(t.cv);
      assertGateTrips(() => t.assertBalanced(origin), new RegExp(`1 of 1 embind wrappers still alive.*${origin.replace(/[.()]/g, '\\$&')}`));
    } finally {
      for (const w of [...t.live.keys()]) w.delete();
      t.dispose();
    }
  }
  // เคสที่ delete ครบ ต้องผ่าน (และ tracker ต้องเห็น wrapper จริง ไม่ใช่ผ่านเพราะนับไม่ได้)
  assertWrapperBalance(cv, 'all deleted', (c) => {
    new c.Mat(4, 4, c.CV_8UC1).delete();
    new c.Mat.zeros(4, 4, c.CV_8UC1).delete();
    c.matFromArray(2, 2, c.CV_8UC1, [1, 2, 3, 4]).delete();
    base.clone().delete();
    base.roi(new c.Rect(1, 1, 3, 3)).delete();
  }, { minCreated: 5 });
  // ลืม delete บน throw path: op ที่ throw กลางทางโดยไม่มี finally
  assertGateTrips(() => assertWrapperBalance(cv, 'throw path', (c) => {
    const tmp = new c.Mat(4, 4, c.CV_8UC1);
    makeThrowingCv(c, 'GaussianBlur').GaussianBlur(tmp, tmp, new c.Size(3, 3), 0);
    tmp.delete();
  }, { expectThrow: /injected test failure/ }), /still alive.*new Mat/);
  base.delete();
});

test('gates-3. wrapper tracker catches a forgotten .delete() of a MatVector.get() result (per-contour wrapper)', () => {
  const src = blobs();
  const opLike = (forgetGet) => (c) => {
    const contours = new c.MatVector();
    const hier = new c.Mat();
    try {
      c.findContours(src, contours, hier, c.RETR_EXTERNAL, c.CHAIN_APPROX_SIMPLE);
      for (let i = 0; i < contours.size(); i++) {
        const cnt = contours.get(i);
        c.contourArea(cnt);
        if (!(forgetGet && i === 2)) cnt.delete();
      }
    } finally { contours.delete(); hier.delete(); }
  };
  assertWrapperBalance(cv, 'every get() deleted', opLike(false), { minCreated: 6 });
  let leaked;
  const t = makeWrapperTracker(cv);
  try {
    opLike(true)(t.cv);
    leaked = [...t.live.keys()];
    assertGateTrips(() => t.assertBalanced('forgot one get()'), /1 of 6 embind wrappers still alive.*MatVector\.get\(\)/);
  } finally { t.dispose(); }
  for (const w of leaked) w.delete();
  src.delete();
});

test('gates-4. tracker covers every embind class the ops use (MatVector, KeyPointVector, ORB, AKAZE, FastFeatureDetector, CLAHE, BackgroundSubtractorMOG2): forgotten -> caught, deleted -> balanced', () => {
  const img = textured(96, 80);
  const color = new cv.Mat(40, 50, cv.CV_8UC3, new cv.Scalar(10, 20, 30));
  const detectWith = (C, ...args) => (c) => {
    const det = new c[C](...args);
    const kps = new c.KeyPointVector();
    det.detect(img, kps);
    assert.ok(kps.size() >= 0);
    return [det, kps];
  };
  const makers = {
    MatVector: (c) => [new c.MatVector()],
    KeyPointVector: (c) => [new c.KeyPointVector()],
    ORB: detectWith('ORB', 50),
    AKAZE: detectWith('AKAZE'),
    FastFeatureDetector: detectWith('FastFeatureDetector', 10, true),
    CLAHE: (c) => { const cl = new c.CLAHE(2.0, new c.Size(8, 8)); const out = new c.Mat(); cl.apply(img, out); return [cl, out]; },
    BackgroundSubtractorMOG2: (c) => {
      const bg = new c.BackgroundSubtractorMOG2(500, 16, true);
      const fg = new c.Mat();
      bg.apply(color, fg); bg.apply(color, fg);
      return [bg, fg];
    },
  };
  for (const [cls, make] of Object.entries(makers)) {
    // ครบ -> balanced
    assertWrapperBalance(cv, `${cls} deleted`, (c) => { for (const w of make(c)) w.delete(); }, { minCreated: 1 });
    // ลืมตัวแรก (ตัว class เอง) -> ต้องจับได้ และระบุที่มา `new <cls>`
    const t = makeWrapperTracker(cv);
    let ws;
    try {
      ws = make(t.cv);
      for (const w of ws.slice(1)) w.delete();
      assertGateTrips(() => t.assertBalanced(`${cls} forgotten`), new RegExp(`still alive.*1x new ${cls}\\b`));
    } finally { t.dispose(); }
    ws[0].delete();
  }
  img.delete(); color.delete();
});

test('gates-5. global mode sees code that uses the real cv module (like cv-worker.js runPipeline) and dispose() restores cv + prototypes exactly', () => {
  const before = { Mat: cv.Mat, MatVector: cv.MatVector, cvtColor: cv.cvtColor, zeros: cv.Mat.zeros, get: cv.MatVector.prototype.get, clone: cv.Mat.prototype.clone };
  const moduleLevelCode = (forget) => { // เขียนแบบโค้ดที่ import cv เอง (ไม่รับ cv เป็นพารามิเตอร์)
    const a = new cv.Mat(8, 8, cv.CV_8UC3, new cv.Scalar(1, 2, 3));
    const g = new cv.Mat();
    cv.cvtColor(a, g, cv.COLOR_RGB2GRAY);
    a.delete();
    if (!forget) g.delete();
    return g;
  };
  // โหมดปกติ: มองไม่เห็น (tracker.cv ไม่ถูกใช้) — นี่คือเหตุผลที่ต้องมีโหมด global
  const tNormal = makeWrapperTracker(cv);
  let g1;
  try { g1 = moduleLevelCode(true); assert.equal(tNormal.created(), 0); } finally { tNormal.dispose(); }
  g1.delete();
  // โหมด global: เห็น
  assertWrapperBalance(cv, 'global, all deleted', () => moduleLevelCode(false), { global: true, minCreated: 2 });
  let g2;
  const tGlobal = makeWrapperTracker(cv, { global: true });
  try {
    g2 = moduleLevelCode(true);
    assertGateTrips(() => tGlobal.assertBalanced('global forgot'), /1 of 2 embind wrappers still alive.*new Mat/);
  } finally { tGlobal.dispose(); }
  g2.delete();
  // คืนของเดิมเป๊ะ (identity เดิม ไม่ใช่ Proxy)
  assert.equal(cv.Mat, before.Mat); assert.equal(cv.MatVector, before.MatVector); assert.equal(cv.cvtColor, before.cvtColor);
  assert.equal(cv.Mat.zeros, before.zeros); assert.equal(cv.MatVector.prototype.get, before.get); assert.equal(cv.Mat.prototype.clone, before.clone);
  // tracker ที่ค้าง (เทสต์ก่อนหน้า fail โดยไม่ dispose) ถูก dispose อัตโนมัติเมื่อสร้างตัวใหม่
  makeWrapperTracker(cv, { global: true });
  makeWrapperTracker(cv).dispose();
  assert.equal(cv.Mat, before.Mat);
});

test('gates-6. heap census catches a synthetic native leak (_malloc without _free, 64B x 100) that the wrapper tracker cannot see', () => {
  const leak = makeNativeLeakControl(cv, 64);
  try {
    // tracker: ไม่มี wrapper ใด ๆ -> "balanced" ทั้งที่รั่ว (จุดบอดที่ census ต้องปิด)
    assertWrapperBalance(cv, 'native leak (tracker blind)', () => leak.run(), { minCreated: 0 });
    assertGateTrips(() => assertNoHeapGrowth(cv, 'native _malloc leak 64B', () => leak.run(), { iters: 100 }), /heap in-use grew by \d+ bytes/);
    assert.ok(leak.leakedBytes >= 64 * 131);
  } finally { leak.cleanup(); }
  // หลัง cleanup: ตัวเดิมที่ไม่รั่วต้องผ่าน (census ไม่ติดค้างสถานะ)
  assertNoHeapGrowth(cv, 'malloc+free control', () => { cv._free(cv._malloc(64)); });
});

test('gates-7. heap census catches a Mat leaked through the raw cv (outside the tracker), and the tracker catches it when it goes through tracker.cv', () => {
  const raw = makeMatLeakControl(cv, 4, 4);
  try {
    assertGateTrips(() => assertNoHeapGrowth(cv, 'raw Mat leak', () => raw.run(), { iters: 100 }), /heap in-use grew/);
  } finally { raw.cleanup(); }
  const t = makeWrapperTracker(cv);
  const viaTracker = makeMatLeakControl(t.cv, 4, 4);
  try {
    viaTracker.run();
    assertGateTrips(() => t.assertBalanced('tracked Mat leak'), /1 of 1 embind wrappers still alive/);
  } finally { viaTracker.cleanup(); t.dispose(); }
});

test('gates-8. leak-free control after a heavy workload (heap grown, allocator free-lists rotated): census + tracker do not trip over 12 rounds (the old address fingerprint does — recorded as diagnostic)', (t) => {
  const src = new cv.Mat(240, 320, cv.CV_8UC3, new cv.Scalar(90, 120, 150));
  const control = (c) => { const g = new c.Mat(); c.cvtColor(src, g, c.COLOR_RGB2GRAY); const b = new c.Mat(); c.GaussianBlur(g, b, new c.Size(5, 5), 0); g.delete(); b.delete(); };
  const contoursControl = (c) => {
    const bw = new c.Mat(); c.threshold(src, bw, 100, 255, c.THRESH_BINARY);
    const g = new c.Mat(); c.cvtColor(bw, g, c.COLOR_RGB2GRAY);
    const cs = new c.MatVector(); const h = new c.Mat();
    c.findContours(g, cs, h, c.RETR_LIST, c.CHAIN_APPROX_SIMPLE);
    for (let i = 0; i < cs.size(); i++) cs.get(i).delete();
    cs.delete(); h.delete(); g.delete(); bw.delete();
  };
  let fpDiff = 0;
  const grewAll = [];
  const ROUNDS = 12;
  for (let r = 0; r < ROUNDS; r++) {
    heavyWorkload(cv, { rounds: 1 });
    const fpA = fingerprint(cv);
    const a = assertLeakFree(cv, `round ${r} cvtColor+blur`, control, { minCreated: 2, iters: 100 });
    const fpB = fingerprint(cv);
    if (fpA !== fpB) fpDiff++;
    const b = assertLeakFree(cv, `round ${r} findContours+get`, contoursControl, { minCreated: 4, iters: 50 });
    grewAll.push(a.grew, b.grew);
  }
  assert.ok(cv.HEAP8.buffer.byteLength > 256 * 1024 * 1024, `heavy workload must actually grow the heap (got ${cv.HEAP8.buffer.byteLength})`);
  t.diagnostic(`census grew per probe over ${ROUNDS} heavy rounds: [${grewAll.join(',')}] bytes; old address-fingerprint DIFF on the same leak-free control: ${fpDiff}/${ROUNDS} rounds`);
  src.delete();
});

test('gates-10. census is stable back-to-back and reports used/free consistently (sanity for the measurement itself)', () => {
  const a = heapCensus(cv);
  const b = heapCensus(cv);
  assert.equal(b.used, a.used, 'two censuses with nothing in between must agree exactly');
  assert.ok(a.free > 0 && a.byteLength === cv.HEAP8.buffer.byteLength);
});

test('gates-11. a C++ cv::Exception caught in JS leaks ~760B inside opencv.js itself (plain census trips even for a clean path); assertNoExtraHeapGrowth vs a raw-cv baseline passes a clean throw path and still catches a Mat leaked on top of it', (t) => {
  const empty = new cv.Mat();
  const k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
  const rawThrow = () => { const d = new cv.Mat(); try { cv.erode(empty, d, k); } catch { /* cv::Exception (ตัวเลข pointer) */ } finally { d.delete(); } };
  const kept = [];
  const leakyThrow = () => { kept.push(new cv.Mat(32, 32, cv.CV_8UC1)); rawThrow(); }; // "op" ที่ลืม delete Mat 1KB บน throw path
  let msg = '';
  assertGateTrips(() => { try { assertNoHeapGrowth(cv, 'raw C++ throw', rawThrow, { iters: 50, warmup: 5 }); } catch (e) { msg = e.message; throw e; } }, /heap in-use grew/);
  t.diagnostic(`raw cv.erode(empty) C++ throw: ${/grew by (\d+)/.exec(msg)[1]} bytes over 50 throws (inherent to opencv.js)`);
  const r = assertNoExtraHeapGrowth(cv, 'clean throw path', rawThrow, rawThrow, { iters: 50 });
  t.diagnostic(`baseline-relative: grew ${r.grew} vs baseline ${r.baselineGrew}`);
  assertGateTrips(() => assertNoExtraHeapGrowth(cv, 'leaky throw path', leakyThrow, rawThrow, { iters: 50 }), /leak beyond the unavoidable C\+\+-exception cost/);
  for (const m of kept) m.delete();
  k.delete(); empty.delete();
});

test('gates-9. real native case via raw cv calls: ORB/AKAZE .detect() throwing on a <3px image leaks inside WASM — census catches it, the wrapper tracker cannot (the tooSmallForKeypoints guard exists for this)', (t) => {
  const tiny = new cv.Mat(1, 50, cv.CV_8UC1, new cv.Scalar(90));
  const ok = new cv.Mat(3, 50, cv.CV_8UC1, new cv.Scalar(90));
  const detectRaw = (C, img, ...args) => (c = cv) => {
    const det = new c[C](...args);
    const kps = new c.KeyPointVector();
    try { det.detect(img, kps); } catch { /* ของจริงใน opencv.js: throw เป็น exception pointer (number) */ } finally { kps.delete(); det.delete(); }
  };
  // ยืนยันก่อนว่า detect บนภาพ 1x50 throw จริง (ถ้า opencv.js รุ่นใหม่เลิก throw เทสต์นี้ต้องทบทวน)
  const det = new cv.ORB(50); const kps = new cv.KeyPointVector();
  let threw = false;
  try { det.detect(tiny, kps); } catch { threw = true; }
  kps.delete(); det.delete();
  assert.ok(threw, 'ORB.detect on a 1x50 image is expected to throw in opencv.js 4.10');

  for (const [C, args] of [['ORB', [50]], ['AKAZE', []]]) {
    // tracker: detector/KeyPointVector ถูก delete ครบ -> balanced (มองไม่เห็น leak ข้างใน WASM)
    assertWrapperBalance(cv, `${C} tiny (tracker)`, detectRaw(C, tiny, ...args), { minCreated: 2 });
    // census: ต้อง trip — จำนวน throw ต่ำ (warmup 2 + iters 10) เพราะ throw สะสมหลายร้อยครั้งทำ heap พัง
    let grewMsg = '';
    assertGateTrips(() => {
      try { assertNoHeapGrowth(cv, `${C} tiny (census)`, detectRaw(C, tiny, ...args), { iters: 10, warmup: 2 }); } catch (e) { grewMsg = e.message; throw e; }
    }, /heap in-use grew by \d+ bytes over 10 iterations/);
    t.diagnostic(`${C} detect-throw on 1x50: ${/grew by (\d+)/.exec(grewMsg)[1]} bytes over 10 throws`);
    // control: ภาพ 3x50 (ขอบเขตต่ำสุดที่ guard ยอมให้ผ่าน) ไม่ throw และไม่รั่ว
    assertNoHeapGrowth(cv, `${C} 3px control`, detectRaw(C, ok, ...args), { iters: 10, warmup: 3 });
  }
  tiny.delete(); ok.delete();
});

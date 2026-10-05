// tests/_leak-helpers.mjs — เครื่องมือกลางสำหรับ "leak gate" ของทุกไฟล์เทสต์ (P-14)
// ไฟล์นี้ชื่อขึ้นต้นด้วย "_" และไม่ลงท้าย .test.mjs จึง "ไม่ถูก" `node --test tests/*.test.mjs` หยิบไปรันเป็นเทสต์
// (มีเทสต์ยืนยันเรื่องนี้ใน tests/leak-gates.test.mjs) — ไม่มี dependency นอกจาก node:assert; รับ `cv` เป็นพารามิเตอร์
//
// ===================================================================
// นโยบาย leak-test (3 ข้อ — สรุปเดียวกับ WORK-TRACKER.md §1)
// ===================================================================
// (1) "address-fingerprint" (จองบล็อกหลายขนาด เรียงที่อยู่ เทียบก่อน/หลัง) ใช้ได้เฉพาะเป็นเทสต์แรกของไฟล์ (process
//     สะอาด) ไฟล์ละไม่เกิน 1 ตัว ตั้งชื่อ `X-0` — หลัง workload หนัก free-list ของ allocator "หมุน" ที่อยู่ของ chunk ขนาด
//     เท่ากัน จึงให้ DIFF ทั้งที่ไม่มี leak (วัดซ้ำได้: control ที่ทำแค่ cvtColor alloc+free ให้ DIFF 10/10 รอบหลัง heavy
//     workload, 0/6 บน process สะอาด — ดู WORKLOG P-14)  => `fingerprintLeakProbe`
// (2) ด่านหลักของทุก op ทุกเส้นทาง (ปกติ/ว่าง/ถูกกรอง/throw) คือ "wrapper-balance": นับทุก `new <embind class>` และทุก
//     wrapper ที่ method/factory คืนมา (MatVector.get, mat.roi, mat.clone, Mat.zeros, cv.matFromArray, ...) เทียบกับ
//     `.delete()`  => `makeWrapperTracker` / `assertWrapperBalance`  (deterministic 100%, ไม่ขึ้นกับ allocator)
// (3) leak "ฝั่ง native" (หน่วยความจำที่รั่วข้างใน WASM โดยไม่มี wrapper JS — เช่น throw จาก ORB/AKAZE .detect() บนภาพ <3px)
//     ตรวจด้วย "heap census" คือวัด byte ที่ยังจองได้ในหน่วยความจำปัจจุบัน ก่อน/หลัง N รอบ  => `assertNoHeapGrowth`
//     ไม่ใช้การเทียบที่อยู่ และ `HEAP8.buffer.byteLength` เดี่ยว ๆ ใช้ไม่ได้ (โตทีละ 128MB จึงไม่ขยับเลยแม้รั่วเป็น MB)
//
// จุดบอดที่ยอมรับ (บอกตรง ๆ):
//  - tracker มองเห็นเฉพาะ wrapper ที่เกิดผ่าน `tracker.cv` (constructor/static/factory) หรือ method ของ embind class
//    ที่ถูก patch ตอน tracker ทำงาน — Mat ที่เทสต์สร้างเองด้วย `cv` ตรง ๆ ไม่นับ (ตั้งใจ: มันเป็นของเทสต์)
//  - tracker ไม่เห็น leak native (ดู `makeNativeLeakControl` + เทสต์พิสูจน์ใน leak-gates.test.mjs)
//  - census เห็น leak ที่ "กินพื้นที่ว่าง" เกิน tolerance รวมตลอดการวน (ค่าเริ่มต้น 2KB ที่ 100 รอบ ≈ 20B/รอบ)
//    แต่ leak เล็กมากที่ตกลงในรูเล็ก ๆ (<64B) ของ heap ที่มีอยู่แล้วจะกลืนหายได้ และมองไม่เห็น leak ที่เกิด "ครั้งเดียว"
//    (เช่น cache แบบ lazy-init ของ OpenCV ที่เกิดหลัง warm-up) — ดังนั้นต้อง warm-up ก่อนเสมอ
//  - C++ exception (cv::Exception) ที่ถูก catch ฝั่ง JS รั่ว ~740–780B ต่อครั้งข้างใน opencv.js เสมอ (ไม่ใช่ความผิดของ op) —
//    เส้นทาง throw แบบ C++ ใช้ `assertNoExtraHeapGrowth` (เทียบกับ baseline ที่ throw แบบเดียวกันผ่าน cv ตรง ๆ) แทน
//    `assertNoHeapGrowth`; throw ที่ "ฉีด" ด้วย makeThrowingCv เป็น JS Error ล้วน ไม่มีค่านี้
import assert from 'node:assert/strict';

// ===================================================================
// 1. wrapper tracker
// ===================================================================
// method ของ ClassHandle (embind) ที่ไม่คืน wrapper ใหม่ และห้ามห่อ
const SKIP_METHODS = new Set(['constructor', 'delete', 'isDeleted', 'isAliasOf', 'deleteLater']);
// property ของโมดูล cv ที่โหมด global ไม่แตะ (hook ของ runtime emscripten / thenable ของโมดูล — ไม่มีทางคืน wrapper)
const GLOBAL_SKIP = new Set(['then', 'onRuntimeInitialized', 'onAbort', 'print', 'printErr', 'locateFile', 'instantiateWasm', 'preRun', 'postRun']);

function isEmbindClass(v) {
  return typeof v === 'function' && v.prototype != null && typeof v.prototype.delete === 'function';
}
// object ที่ "เจ้าของต้อง .delete() เอง": embind handle (มีทั้ง delete และ isDeleted) — value object อย่าง KeyPoint /
// Point / Rect / minMaxLoc ที่คืนเป็น plain object ไม่มี delete จึงไม่ถูกนับ
function isOwnedWrapper(o) {
  return o !== null && typeof o === 'object' && typeof o.delete === 'function' && typeof o.isDeleted === 'function';
}

let activeTracker = null;

/**
 * สร้าง tracker ที่นับ "ทุก" embind wrapper ที่สร้างระหว่างที่ tracker ทำงาน แล้วเทียบกับ `.delete()`
 *
 * ครอบคลุม (ทั้งหมดผ่าน `tracker.cv` — ส่งตัวนี้ให้ op แทน `cv`):
 *   - constructor ของ embind class ทุกตัวที่มี prototype.delete: Mat, MatVector, KeyPointVector, PointVector, ORB, AKAZE,
 *     FastFeatureDetector, CLAHE, BackgroundSubtractorMOG2, ... (สแกนจาก cv จริง ไม่ hard-code รายชื่อ)
 *   - static/factory ที่คืน wrapper: `new cv.Mat.zeros(...)`/`Mat.ones`/`Mat.eye`, `cv.matFromArray`, `cv.getStructuringElement`,
 *     `cv.getPerspectiveTransform`, `cv.imread` ... (ห่อทุก function บน cv และ static บน class — ตรวจผลลัพธ์ว่าเป็น
 *     wrapper หรือไม่ ไม่ต้องรู้ชื่อล่วงหน้า จึงครอบ factory ใหม่ ๆ ในอนาคตอัตโนมัติ)
 *   - method ที่คืน wrapper ใหม่ (patch ที่ prototype ระหว่าง tracker ทำงานเท่านั้น จึงเห็นแม้ Mat ต้นทางเป็นของเทสต์):
 *     MatVector.get, mat.roi/clone/row/col/rowRange/colRange/diag/t/inv/mul, DMatchVectorVector.get ฯลฯ
 *     (ห่อทุก method ของทุก class แล้วตรวจผลลัพธ์ — `clone()`/`roi()` คืน wrapper ที่ต้อง delete เอง)
 *   value object ธรรมดา (KeyPointVector.get → KeyPoint, cv.boundingRect → Rect ฯลฯ) ไม่มี .delete() จึงไม่นับ
 *
 * ต้อง `dispose()` เสมอ (ใช้ try/finally หรือ `assertWrapperBalance`/`withWrapperTracker`) — คืน prototype ที่ patch ไว้
 * มีได้ทีละ tracker เดียว (ถ้ามีตัวเก่าค้างจากเทสต์ที่ fail กลางคัน จะถูก dispose ให้อัตโนมัติ)
 *
 * `{ global: true }` — สำหรับโค้ดที่ "ไม่รับ cv เป็นพารามิเตอร์" แต่ import cv เองที่ระดับโมดูล (เช่น runPipeline ใน
 * cv-worker.js): สลับ property ของ `cv` ตัวจริงชั่วคราว (class → Proxy ที่นับ `new`, function → Proxy ที่นับผลลัพธ์)
 * แล้วคืนของเดิมเป๊ะตอน dispose (ทุก property ของโมดูล cv เป็น writable/configurable — ยืนยันแล้ว) ในโหมดนี้
 * `tracker.cv` กับ `cv` ตัวจริงนับเหมือนกัน — fixture ที่สร้าง "ก่อน" สร้าง tracker ไม่ถูกนับเหมือนเดิม
 *
 * @param {object} cv
 * @param {{global?: boolean}} [opts]
 * @returns {{cv: object, live: Map, created: () => number, createdBy: () => object, describeLive: () => string,
 *            assertBalanced: (label: string, opts?: {minCreated?: number, expectLive?: number}) => void, dispose: () => void}}
 */
export function makeWrapperTracker(cv, { global = false } = {}) {
  if (activeTracker) activeTracker.dispose(); // เทสต์ก่อนหน้า fail กลางทางแล้วไม่ได้ dispose

  const live = new Map();       // wrapper -> ที่มา ("new Mat", "MatVector.get()", ...)
  const seen = new WeakSet();   // กันนับซ้ำ (method คืน handle เดิม)
  const createdByOrigin = {};
  let created = 0;

  function adopt(obj, origin) {
    if (!isOwnedWrapper(obj) || seen.has(obj)) return obj;
    seen.add(obj);
    created++;
    createdByOrigin[origin] = (createdByOrigin[origin] || 0) + 1;
    live.set(obj, origin);
    const realDelete = obj.delete; // method ของ prototype (embind) — ผูกผ่าน call ตอนเรียก
    obj.delete = function trackedDelete(...a) { live.delete(obj); return realDelete.apply(obj, a); };
    return obj;
  }

  // ---- (ก) patch method ของทุก embind class ที่ prototype ----
  const restore = [];
  for (const key of Object.keys(cv)) {
    let C;
    try { C = cv[key]; } catch { continue; }
    if (!isEmbindClass(C)) continue;
    const proto = C.prototype;
    for (const name of Object.getOwnPropertyNames(proto)) {
      if (SKIP_METHODS.has(name)) continue;
      const desc = Object.getOwnPropertyDescriptor(proto, name);
      if (!desc || typeof desc.value !== 'function' || !desc.configurable) continue; // getter (เช่น Mat.data) ไม่แตะ
      const orig = desc.value;
      const origin = `${key}.${name}()`;
      const patched = function (...args) { return adopt(orig.apply(this, args), origin); };
      // embind สร้าง method แบบอ้าง `proto[name].overloadTable` ตอนเรียก — ต้องคัดลอก property ของตัวเดิมมาให้ตัวที่ห่อ
      Object.assign(patched, orig);
      Object.defineProperty(proto, name, { ...desc, value: patched });
      restore.push([proto, name, desc]);
    }
  }

  // ---- (ข) Proxy ของ cv: constructor, static, factory ----
  const fnCache = new Map();
  function wrapFn(fn, origin) {
    let w = fnCache.get(fn);
    if (!w) {
      w = new Proxy(fn, {
        apply(t, thisArg, args) { return adopt(Reflect.apply(t, thisArg, args), origin); },
        construct(t, args, nt) { return adopt(Reflect.construct(t, args, nt), origin); },
      });
      fnCache.set(fn, w);
    }
    return w;
  }
  const classCache = new Map();
  function wrapClass(C, key) {
    let w = classCache.get(C);
    if (!w) {
      w = new Proxy(C, {
        construct(t, args, nt) { return adopt(Reflect.construct(t, args, nt === w ? t : nt), `new ${key}`); },
        get(t, prop, receiver) {
          const v = Reflect.get(t, prop, receiver);
          // static factory (Mat.zeros / ones / eye ...) — `new cv.Mat.zeros(...)` ก็ผ่านทางนี้
          if (typeof v === 'function' && typeof prop === 'string' && prop !== 'constructor' && Object.prototype.hasOwnProperty.call(t, prop) && !isEmbindClass(v)) return wrapFn(v, `${key}.${prop}()`);
          return v;
        },
      });
      classCache.set(C, w);
    }
    return w;
  }
  const proxy = new Proxy(cv, {
    get(t, prop, receiver) {
      const v = Reflect.get(t, prop, receiver);
      if (typeof prop !== 'string' || prop.startsWith('_')) return v;
      if (isEmbindClass(v)) return wrapClass(v, prop);
      if (typeof v === 'function') return wrapFn(v, `cv.${prop}()`);
      return v;
    },
  });

  // ---- (ค) โหมด global: สลับ property ของ cv ตัวจริง (คืนใน dispose ตามลำดับย้อนกลับ) ----
  const restoreGlobal = [];
  if (global) {
    for (const key of Object.keys(cv)) {
      if (key.startsWith('_') || GLOBAL_SKIP.has(key)) continue;
      const desc = Object.getOwnPropertyDescriptor(cv, key);
      if (!desc || !('value' in desc) || typeof desc.value !== 'function' || !desc.writable) continue;
      const v = desc.value;
      Object.defineProperty(cv, key, { ...desc, value: isEmbindClass(v) ? wrapClass(v, key) : wrapFn(v, `cv.${key}()`) });
      restoreGlobal.push([key, desc]);
    }
  }

  let disposed = false;
  const tracker = {
    cv: proxy,
    live,
    created: () => created,
    createdBy: () => ({ ...createdByOrigin }),
    describeLive() {
      const by = {};
      for (const o of live.values()) by[o] = (by[o] || 0) + 1;
      return Object.entries(by).map(([o, n]) => `${n}x ${o}`).join(', ') || '(none)';
    },
    /**
     * @param {string} label
     * @param {{minCreated?: number, expectLive?: number}} [opts] minCreated: ต้องเห็น wrapper อย่างน้อยกี่ตัว (กัน probe "ตาบอด"
     *   แล้วผ่านเพราะไม่ได้นับอะไรเลย); expectLive: จำนวน wrapper ที่ตั้งใจให้ค้างไว้ (เช่น state ข้ามเฟรมของ op ที่มี ctx.state)
     */
    assertBalanced(label, { minCreated = 1, expectLive = 0 } = {}) {
      assert.ok(created >= minCreated, `${label}: wrapper probe must actually observe wrappers (saw ${created}, need >= ${minCreated})`);
      assert.equal(live.size, expectLive,
        `${label}: ${live.size} of ${created} embind wrappers still alive (expected ${expectLive}) -> never .delete()d: ${this.describeLive()}`);
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      for (let i = restoreGlobal.length - 1; i >= 0; i--) {
        const [key, desc] = restoreGlobal[i];
        Object.defineProperty(cv, key, desc);
      }
      for (let i = restore.length - 1; i >= 0; i--) {
        const [proto, name, desc] = restore[i];
        Object.defineProperty(proto, name, desc);
      }
      if (activeTracker === tracker) activeTracker = null;
    },
  };
  activeTracker = tracker;
  return tracker;
}

/** สร้าง tracker, ส่งให้ fn(tracker), dispose ให้เสมอ (try/finally) */
export function withWrapperTracker(cv, fn, opts = {}) {
  const t = makeWrapperTracker(cv, opts);
  try { return fn(t); } finally { t.dispose(); }
}

/**
 * ด่านหลัก: รัน `run(trackedCv)` หนึ่งครั้ง (run ต้องเรียก op ด้วย cv ที่ได้รับ และ .delete() ผลลัพธ์เอง) แล้วยืนยันว่า wrapper
 * ทุกตัวที่ถูกสร้างถูก .delete() ครบ ไม่ว่าจบปกติหรือ throw
 * @param {object} cv
 * @param {string} label
 * @param {(cvT: object) => void} run
 * @param {{expectThrow?: boolean|RegExp, minCreated?: number, expectLive?: number, global?: boolean}} [opts] expectThrow:
 *   true = ต้อง throw (อะไรก็ได้), RegExp = ต้อง throw และข้อความตรง pattern; global: ดู makeWrapperTracker
 */
export function assertWrapperBalance(cv, label, run, { expectThrow = false, minCreated = 1, expectLive = 0, global = false } = {}) {
  const t = makeWrapperTracker(cv, { global });
  try {
    if (expectThrow) assert.throws(() => run(t.cv), expectThrow instanceof RegExp ? expectThrow : /./, `${label}: expected a throw`);
    else run(t.cv);
    t.assertBalanced(label, { minCreated, expectLive });
  } finally {
    t.dispose();
  }
}

// ===================================================================
// 2. ตัวช่วยฉีด error (แทน makeThrowing*Cv ที่เคย copy ไว้ทุกไฟล์)
// ===================================================================
/**
 * Proxy ของ cv ที่ทำให้ function `methodName` throw — ข้อความ `injected test failure (<name>)` (หรือ `... call #n)`)
 * @param {object} baseCv cv จริง หรือ tracker.cv
 * @param {string} methodName
 * @param {{onCall?: number}} [opts] onCall = n: throw เฉพาะ "ครั้งที่ n" (นับใหม่ทุก proxy) ครั้งอื่นเรียกของจริง
 */
export function makeThrowingCv(baseCv, methodName, { onCall = null } = {}) {
  let count = 0;
  return new Proxy(baseCv, {
    get(target, prop) {
      if (prop !== methodName) return target[prop];
      if (onCall == null) return () => { throw new Error(`injected test failure (${methodName})`); };
      return (...args) => {
        count++;
        if (count === onCall) throw new Error(`injected test failure (${methodName} call #${onCall})`);
        return target[prop](...args);
      };
    },
  });
}

// ===================================================================
// 3. heap census — gate สำหรับ leak native / รวม Mat  (ด่านข้อ 3 ของนโยบาย)
// ===================================================================
// วัดอย่างไร: จอง `_malloc` เป็นบล็อกขนาดลดหลั่น (256KB → 16KB → 1KB → 64B) ไปเรื่อย ๆ จนที่อยู่ใกล้ปลาย heap (เหลือ reserve
// 1MB กันไม่ให้ heap โตเอง) แล้ว free ทั้งหมด — ผลรวม byte ที่จองได้ = "พื้นที่ว่างที่ใช้ได้จริง" ของ heap ปัจจุบัน ซึ่งลดลงทุก
// byte ที่รั่ว ไม่ว่ารั่วเป็น Mat, detector หรือ memory ที่หลุดข้างใน WASM และ *ไม่ขึ้นกับว่า allocator วางบล็อกไว้ที่อยู่ไหน*
// (ต่างจาก fingerprint) จึงไม่โดน free-list rotation ยืนยันเชิงประจักษ์: control ที่ไม่รั่ว 12 แบบ × 4 รอบ × (process สะอาด,
// หลัง heavy workload ที่ heap โต 128→512MB) ได้ drop = 0 ทุกค่า ส่วน leak 64B×100 รอบ ได้ drop 7,168B (เห็น) ใช้เวลา ~1ms
// ต่อครั้ง ตัวเลขทั้งหมดอยู่ใน WORKLOG-2026-09-28-tier2-multi-input.md (รายการ "P-14 measure")
export const CENSUS_LEVELS = [262144, 16384, 1024, 64];
export const CENSUS_RESERVE = 1 << 20;

/** @returns {{free: number, used: number, byteLength: number}} used = (byteLength - reserve) - free  (คงที่แม้ heap โตระหว่างวัด) */
export function heapCensus(cv, { levels = CENSUS_LEVELS, reserve = CENSUS_RESERVE } = {}) {
  const byteLength = cv.HEAP8.buffer.byteLength;
  const limit = byteLength - reserve;
  const held = [];
  let free = 0;
  for (const size of levels) {
    for (;;) {
      const p = cv._malloc(size);
      if (p === 0) break;
      if (p + size > limit) { cv._free(p); break; } // ถึงเขต reserve ท้าย heap แล้ว
      held.push(p);
      free += size;
    }
  }
  for (let i = held.length - 1; i >= 0; i--) cv._free(held[i]);
  return { free, used: limit - free, byteLength };
}

/**
 * ยืนยันว่ารัน `run` เพิ่ม `iters` รอบแล้ว heap ที่ใช้อยู่ "ไม่โต" เกิน tolerance (byte รวมทั้งลูป)
 * ต้อง warm-up ก่อนเสมอ (cache ของ OpenCV/allocator เกิดครั้งแรกจะดูเหมือน leak) — ค่าเริ่มต้น warmup=30
 * ความไว ≈ tolerance/iters byte ต่อรอบ (2048/100 ≈ 20B) — ต้องเลือก iters ให้ leak ที่กลัวสะสมเกิน tolerance
 * @returns {{grew: number, iters: number}}
 */
export function assertNoHeapGrowth(cv, label, run, { iters = 100, warmup = 30, tolerance = 2048 } = {}) {
  for (let i = 0; i < warmup; i++) run();
  const a = heapCensus(cv);
  for (let i = 0; i < iters; i++) run();
  const b = heapCensus(cv);
  const grew = b.used - a.used;
  assert.ok(grew <= tolerance,
    `${label}: heap in-use grew by ${grew} bytes over ${iters} iterations (tolerance ${tolerance}) -> native/Mat leak`
    + ` [census free ${a.free} -> ${b.free}, byteLength ${a.byteLength} -> ${b.byteLength}]`);
  return { grew, iters };
}

/**
 * census แบบ "เทียบ baseline" สำหรับเส้นทางที่ throw C++ exception (cv::Exception) — ทุกครั้งที่ C++ exception ถูก catch
 * ฝั่ง JS, opencv.js รั่ว exception object + ข้อความ ~740–780B ข้างใน WASM เอง (วัดแล้ว: cv.erode/cv.cvtColor บน Mat ว่าง
 * ผ่าน cv ตรง ๆ ก็รั่วเท่ากัน; opencv.js 4.10 ไม่ export ___cxa_free_exception/decrementExceptionRefcount จึงคืนไม่ได้) —
 * leak นี้แก้ในโค้ดแอปไม่ได้ (ทางเดียวคือ guard ไม่ให้ throw ตั้งแต่แรก แบบ tooSmallForKeypoints)
 * จึงวัด `baselineRun` (เรียก cv ตรง ๆ ให้ throw exception แบบเดียวกัน) กับ `run` (op จริง) ด้วยจำนวนรอบเท่ากัน แล้วยืนยันว่า
 * op "ไม่รั่วเพิ่ม" จาก baseline เกิน max(tolerance, relTolerance × baseline) — ยังจับ Mat/native leak ของ op บน throw path ได้
 * (พิสูจน์ใน leak-gates gates-11) ส่วน relTolerance (10%) มีเพราะ exception object ~760B/ครั้งไปตกในรูของ heap ต่างกันตามสภาพ
 * fragmentation: ปกติต่างจาก baseline ~700B/50 รอบ แต่ถ้า heap ถูกทำให้กระจัดกระจายด้วย leak ของเทสต์ก่อนหน้า (revert-proof: Mat
 * 72B รั่วหลายร้อยตัว) ต่างได้ ~3KB ทั้งที่ path นี้ไม่รั่ว — ความไวยังเหลือ ~75B/throw (Mat เล็กสุด + header ก็เกินนี้)
 * @returns {{grew: number, baselineGrew: number, iters: number}}
 */
export function assertNoExtraHeapGrowth(cv, label, run, baselineRun, { iters = 50, warmup = 5, tolerance = 2048, relTolerance = 0.1 } = {}) {
  const measure = (fn) => {
    for (let i = 0; i < warmup; i++) fn();
    const a = heapCensus(cv);
    for (let i = 0; i < iters; i++) fn();
    return heapCensus(cv).used - a.used;
  };
  const baselineGrew = measure(baselineRun);
  const grew = measure(run);
  const allowed = Math.max(tolerance, Math.round(baselineGrew * relTolerance));
  assert.ok(grew - baselineGrew <= allowed,
    `${label}: heap in-use grew by ${grew} bytes over ${iters} iterations vs ${baselineGrew} for the raw-cv baseline`
    + ` (extra ${grew - baselineGrew} > allowed ${allowed}) -> leak beyond the unavoidable C++-exception cost`);
  return { grew, baselineGrew, iters };
}

/**
 * ด่านมาตรฐานต่อ "หนึ่ง op หนึ่งเส้นทาง" (ใช้แทน sentinel/fingerprint เดิมในเทสต์ที่ไม่ใช่ตัวแรกของไฟล์):
 *   1) wrapper-balance หนึ่งรอบ (`run(tracker.cv)`) — ด่านหลัก, deterministic
 *   2) heap census `iters` รอบ (`run(cv)`) — จับ leak native / leak ที่ tracker มองไม่เห็น
 * `run(c)` ต้องเรียก op ด้วย `c` และ .delete() ผลลัพธ์เอง; ถ้า `expectThrow` ทุกรอบต้อง throw ตาม pattern
 * @param {{expectThrow?: boolean|RegExp, minCreated?: number, expectLive?: number, global?: boolean,
 *          iters?: number, warmup?: number, tolerance?: number}} [opts]
 */
export function assertLeakFree(cv, label, run, { expectThrow = false, minCreated = 1, expectLive = 0, global = false, iters = 100, warmup = 30, tolerance = 2048 } = {}) {
  assertWrapperBalance(cv, label, run, { expectThrow, minCreated, expectLive, global });
  const pattern = expectThrow instanceof RegExp ? expectThrow : /./;
  const once = expectThrow ? () => assert.throws(() => run(cv), pattern, `${label}: expected a throw`) : () => run(cv);
  return assertNoHeapGrowth(cv, label, once, { iters, warmup, tolerance });
}

// ===================================================================
// 4. address-fingerprint — "ใช้ได้เฉพาะเป็นเทสต์แรกของไฟล์ (process สะอาด)"
// ===================================================================
export const FP_SIZES = [8, 16, 24, 40, 56, 72, 104, 136, 200, 264, 520, 1032, 2056, 4104, 8200, 16392, 65544, 262152, 1048584];

export function fingerprint(cv, sizes = FP_SIZES) {
  const ptrs = [];
  for (const s of sizes) for (let r = 0; r < 4; r++) ptrs.push(cv._malloc(s));
  ptrs.sort((a, b) => a - b);
  const sig = ptrs.join(',');
  for (let i = ptrs.length - 1; i >= 0; i--) cv._free(ptrs[i]);
  return sig + '|' + cv.HEAP8.buffer.byteLength;
}

/**
 * !!! VALID ONLY AS THE FIRST TEST IN A CLEAN PROCESS (ไฟล์ละไม่เกิน 1 เทสต์ ชื่อ `X-0`, วางบนสุดของไฟล์) !!!
 * หลัง workload หนัก free-list ของ allocator หมุนที่อยู่ของ chunk ขนาดเท่ากัน ทำให้ fingerprint ต่างกันทั้งที่ไม่มี leak
 * (false positive) — ห้ามเรียกจากเทสต์ที่ไม่ใช่ตัวแรกของไฟล์ ใช้ `assertWrapperBalance` + `assertNoHeapGrowth` แทน
 * เก็บไว้เพราะไวต่อ leak เล็ก ๆ ของ chunk ขนาดเฉพาะได้ในบริบทสะอาด
 */
export function fingerprintLeakProbe(cv, label, run, { iters = 180, warmup = 30, sizes = FP_SIZES } = {}) {
  for (let i = 0; i < warmup; i++) run();
  const fpA = fingerprint(cv, sizes);
  for (let i = 0; i < iters; i++) run();
  const fpB = fingerprint(cv, sizes);
  assert.equal(fpB, fpA, `${label}: malloc fingerprint changed after ${iters} more calls -> leak`);
}

// ===================================================================
// 5. ตัวควบคุมสำหรับ "พิสูจน์ว่า gate ทำงานได้จริง" (leak โดยตั้งใจ)
// ===================================================================
/**
 * leak native โดยตั้งใจ: `_malloc(bytesPerCall)` ทุกครั้งที่ run() แล้วไม่ free — ไม่มี wrapper JS ใด ๆ
 * tracker จึง "มองไม่เห็น" (จุดบอดที่ตั้งใจให้ census ปิด) ต้องเรียก cleanup() ท้ายเทสต์
 */
export function makeNativeLeakControl(cv, bytesPerCall = 64) {
  const ptrs = [];
  return {
    run() { ptrs.push(cv._malloc(bytesPerCall)); },
    get leakedBytes() { return ptrs.length * bytesPerCall; },
    cleanup() { for (const p of ptrs) cv._free(p); ptrs.length = 0; },
  };
}

/** leak แบบ wrapper: สร้าง Mat เล็ก ๆ ผ่าน `cvUsed` (ส่ง tracker.cv ถ้าจะให้ tracker เห็น) แล้วไม่ delete — ต้อง cleanup() ท้ายเทสต์ */
export function makeMatLeakControl(cvUsed, rows = 4, cols = 4) {
  const mats = [];
  return {
    run() { mats.push(new cvUsed.Mat(rows, cols, cvUsed.CV_8UC1)); },
    cleanup() { for (const m of mats) if (!m.isDeleted()) m.delete(); mats.length = 0; },
  };
}

// ===================================================================
// 6. workload หนักสำหรับพิสูจน์ว่า gate ไม่ false-positive หลังงานหนัก
// ===================================================================
/**
 * workload หนักที่ทำให้ heap โต (128→512MB บนเครื่อง dev) และ free-list ของ allocator หมุนที่อยู่: findContours บนภาพ noise
 * 1080p + churn `_malloc` 20,000 บล็อกขนาดสุ่ม 16B–70KB แล้ว free สลับลำดับ — deterministic (PRNG ตายตัว) และไม่ leak
 * (free ครบ) นี่คือสถานการณ์ที่ทำให้ fingerprint แบบเทียบที่อยู่ DIFF ทั้งที่ไม่มี leak
 */
export function heavyWorkload(cv, { rounds = 2 } = {}) {
  let seed = 12345;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
  for (let r = 0; r < rounds; r++) {
    const noise = new cv.Mat(1080, 1920, cv.CV_8UC1);
    for (let i = 0; i < noise.data.length; i++) noise.data[i] = rnd() < 0.05 ? 255 : 0;
    const cs = new cv.MatVector();
    const h = new cv.Mat();
    cv.findContours(noise, cs, h, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    for (let i = 0; i < Math.min(cs.size(), 20000); i++) { const c = cs.get(i); cv.boundingRect(c); c.delete(); }
    cs.delete(); h.delete(); noise.delete();
    const ptrs = [];
    for (let i = 0; i < 20000; i++) ptrs.push(cv._malloc(16 + Math.floor(rnd() * rnd() * 70000)));
    for (let i = ptrs.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [ptrs[i], ptrs[j]] = [ptrs[j], ptrs[i]]; }
    for (const p of ptrs) cv._free(p);
  }
}

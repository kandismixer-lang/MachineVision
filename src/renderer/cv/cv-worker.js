// Web Worker: โหลด OpenCV.js + รัน pipeline ของ filter หลายชั้น แยกจาก main thread
import cv from "@techstark/opencv-js";
import { OPS, FILTERS, needsOf, opIdOfStep } from "../filters/registry.js";

let ready = false;

function isReady(c) {
  return c && typeof c.Mat === "function" && typeof c.cvtColor === "function";
}

function initOpenCV() {
  return new Promise((resolve) => {
    if (isReady(cv)) return resolve();
    const poll = setInterval(() => {
      if (isReady(cv)) {
        clearInterval(poll);
        resolve();
      }
    }, 100);
    cv.onRuntimeInitialized = () => {
      clearInterval(poll);
      resolve();
    };
  });
}

function toGray(mat) {
  const g = new cv.Mat();
  if (mat.channels() === 1) mat.copyTo(g);
  else cv.cvtColor(mat, g, mat.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY);
  return g;
}

// แปลงผลลัพธ์ (1/3/4 channel) → RGBA เพื่อส่งไปแสดงบน canvas
function toRGBA(mat) {
  const out = new cv.Mat();
  const ch = mat.channels();
  if (ch === 1) cv.cvtColor(mat, out, cv.COLOR_GRAY2RGBA);
  else if (ch === 3) cv.cvtColor(mat, out, cv.COLOR_RGB2RGBA);
  else mat.copyTo(out);
  return out;
}

// เก็บ state ที่คงอยู่ข้ามเฟรมของแต่ละ node (Tier 2 §A.3) — keyed by nodeId เดียวกับที่ tracePipeline
// ฝั่ง editor.js ใส่มาใน step.nodeId อยู่แล้ว ทุก field เป็นของ op เอง ยกเว้น __paramsFingerprint/
// __lastW/__lastH/__dispose ที่ getNodeState/sweepNodeState ใช้จัดการ reset เอง (ดู comment ด้านล่าง)
const nodeState = new Map(); // nodeId -> { ...op-defined fields..., __paramsFingerprint, __lastW, __lastH, __dispose }

// คืน state object ของ node นี้ — ถ้า params เปลี่ยน (fingerprint ไม่ตรง) หรือขนาดภาพที่ "เข้าถึง op นี้
// จริง ๆ" เปลี่ยน (ไม่ใช่ขนาดเฟรมกล้องดิบ — ดู call site ใน runPipeline ที่ส่ง input.cols/input.rows ไม่ใช่
// width/height ระดับบนสุด) จะเรียก __dispose() ของ state เก่าก่อน (op เป็นคนตั้งเอง ทำความสะอาด Mat/
// detector ของตัวเอง) แล้วคืน object ใหม่เปล่า ๆ ให้ op สร้างใหม่ทั้งหมด
// 18d (§C.4-final 0-A/0-B): `params` ที่ runPipeline ส่งมาคือ "fingerprint ของ pipeline ต้นน้ำจนถึง step นี้"
// (ดู stepFingerprint) ไม่ใช่แค่ params ของ node เอง — แก้อะไรก็ตามที่เปลี่ยนพิกเซลที่ไหลเข้า node นี้ = reset
// `store` = Map ที่เก็บ state (ค่าเริ่มต้น = Map ของโมดูล) — export ส่ง Map ชั่วคราวของตัวเองมา (0-D)
export function getNodeState(nodeId, params, width, height, store = nodeState) {
  const fp = JSON.stringify(params || {});
  let s = store.get(nodeId);
  if (s && (s.__paramsFingerprint !== fp || s.__lastW !== width || s.__lastH !== height)) {
    if (typeof s.__dispose === "function") s.__dispose(); // op คนละตัวเป็นคนเคลียร์ Mat/detector ของตัวเอง
    s = null;
  }
  if (!s) {
    s = { __paramsFingerprint: fp, __lastW: width, __lastH: height };
    store.set(nodeId, s);
  }
  return s;
}

// เรียกครั้งเดียวต่อเฟรม ก่อนวน pipelines ทั้งหมด — ทิ้ง state ของ node ที่ไม่อยู่ใน graph แล้ว (ลบ node/
// ถอดสายออกจากทุกกล่องผลลัพธ์) กันหน่วยความจำค้างของ node ที่ผู้ใช้ลบ/ถอดสายไปแล้วแต่ worker ไม่เคยรู้
export function sweepNodeState(activeNodeIds, store = nodeState) {
  for (const [nodeId, s] of store) {
    if (!activeNodeIds.has(nodeId)) {
      if (typeof s.__dispose === "function") s.__dispose();
      store.delete(nodeId);
    }
  }
}

// เก็บ nodeId ทุกตัวที่ "ยังใช้งานอยู่จริง" ในเฟรมนี้ — รวมของ src2 sub-pipeline แบบ recursive ด้วย
// (imabsdiff ซ้อน imabsdiff ก็ยังเก็บครบ) ใช้ก่อนเรียก sweepNodeState แต่ละเฟรม
// PIQE อยู่ใน host สองพอร์ตแต่ไม่ใช้ B: ไม่เดินสาย/เก็บ state/fingerprint ของสายที่ไม่ได้รัน
const usesSrc2 = (step) => OPS[opIdOfStep(step)]?.usesSrc2 !== false;
export function collectNodeIds(pipeline, set) {
  for (const step of pipeline || []) {
    if (!step || step.enabled === false) continue; // node ที่ถูกปิด (bypass) และ src2 ของมันไม่ได้รันเฟรมนี้ → sweep ทิ้ง state (18d, V14)
    if (step.nodeId != null) set.add(step.nodeId);
    if (step.src2 && usesSrc2(step)) collectNodeIds(step.src2.pipeline, set);
  }
}

// fingerprint ของ step หนึ่ง "รวม src2 แบบ recursive" (18d, V13) — ใช้สร้าง prefix ของ pipeline ต้นน้ำจนถึง
// step ปัจจุบัน ให้ getNodeState reset state เมื่อ "อะไรก็ตาม" ที่เปลี่ยนพิกเซลที่ไหลเข้า node นี้เปลี่ยน
// (params/เปิด-ปิด ของกล่องก่อนหน้า หรือกล่องในสาย B ของกล่อง 2 ภาพที่อยู่ต้นน้ำ)
function stepFingerprint(s) {
  if (!s) return null; // runPipeline ข้าม step ที่เป็น null อยู่แล้ว — fingerprint ต้องไม่ throw เพราะมัน
  return {
    id: s.id,
    params: s.params,
    enabled: s.enabled,
    src2: s.src2 && usesSrc2(s) ? (s.src2.pipeline || []).map(stepFingerprint) : undefined, // JSON.stringify ตัด undefined ทิ้ง
  };
}

// คัดลอก ctx.values แบบตื้น ๆ ตามรูปทรงที่ §A.2 กำหนดไว้ (kind:"scalars" มี fields[] · kind:"table" มี
// columns[]/rows[][]) — ใช้ตอน memo hit เท่านั้น: ถ้าคืน object เดิมให้ทั้งกล่องผลลัพธ์และพรีวิว note ทั้งสอง
// จะชี้ object เดียวกัน และ structuredClone ของ postMessage ก็รักษาความเป็น object เดียวกันข้ามไปฝั่ง UI ด้วย
// → detail-panel (U3) ที่เขียน/เพิ่ม field ลงตารางหนึ่ง จะไปแก้ของอีกกล่องด้วยโดยไม่ตั้งใจ
// export ไว้เพื่อให้เทสต์ล็อกได้ทั้ง 2 รูปทรงของ §A.2 โดยตรง — วันนี้ยังไม่มี op ที่ stateful ตัวไหนคืน
// kind:"table" (สาขา columns/rows จึงยังไม่ถูกเดินผ่านในการใช้งานจริง) แต่ §A.2 ประกาศรูปทรงนี้ไว้แล้ว
// ถ้าปล่อยให้ไม่มีเทสต์ วันที่มี op แบบนั้นเกิดขึ้นจริงจะไม่มีอะไรบอกว่าการคัดลอกตื้นไปครึ่งทาง
export function cloneValues(v) {
  if (!v || typeof v !== "object") return v; // undefined/null/ค่าพื้นฐาน — ไม่มีอะไรให้คัดลอก
  const c = { ...v };
  if (Array.isArray(v.fields)) c.fields = v.fields.map((f) => (f && typeof f === "object" ? { ...f } : f));
  if (Array.isArray(v.columns)) c.columns = v.columns.slice();
  if (Array.isArray(v.rows)) c.rows = v.rows.map((r) => (Array.isArray(r) ? r.slice() : r));
  return c;
}

// รัน pipeline: [{ id, params, enabled, nodeId, src2? }] ตามลำดับ
// คืน { mat, notes } โดย mat เป็นผลลัพธ์ (caller ต้อง delete)
// nodeState/activeNodeIds (Tier 2 §A.3) รับเข้ามาเป็นพารามิเตอร์ ส่งต่อ (thread) ลงไปยัง recursive
// call ของ src2 sub-pipeline ด้วย — ตัว runPipeline เองไม่ได้ใช้ activeNodeIds โดยตรง (ใช้ที่
// computeFrame ระดับบนสุดครั้งเดียวก่อนวน pipeline ทั้งหมด) แต่ต้อง thread ผ่านไว้เผื่อ recursion ต่อไป
//
// 18d (§C.4-final 0-B/0-C): `nodeState` = store ของ state ที่ใช้จริง (ถ้า undefined ใช้ Map ของโมดูล) ·
// `frameMemo` (optional, computeFrame สร้างให้ 1 ตัวต่อเฟรม) = nodeId -> { key, mat, info, values } กัน op ที่
// stateful:true รันซ้ำในเฟรมเดียวกัน (node เดียวอยู่ในกล่องผลลัพธ์ + พรีวิวของ node ที่เลือก หรือป้อนทั้ง 2 ช่อง
// ของกล่อง 2 ภาพ) — รันซ้ำ = MOG2 เรียนรู้เร็วขึ้น 2 เท่า / optical flow เทียบเฟรมกับตัวเอง (ขยับ = 0 เสมอ)
// เจ้าของ mat ใน memo คือ computeFrame (delete ใน finally) — ที่นี่ใช้แค่ clone ออกไป
export function runPipeline(baseRGB, pipeline, nodeState, activeNodeIds, frameMemo = null) {
  // note ทุกตัวผูกกับ "กล่อง" ไม่ใช่ชนิด filter: nodeId = step.nodeId (id ของ node ใน LiteGraph) — UI กรองด้วยตัวนี้
  // (กล่องชนิดเดียวกัน 2 กล่องจึงไม่ปนกัน) · view = f.subtitle(params) = "วิธี" ที่ใช้ตอนคำนวณ (เฉพาะ filter ที่มี
  // subtitle เช่นกล่องรวม) ให้ UI ทิ้ง note ของวิธีเก่าที่ตามมากับเฟรมค้างในท่อได้แบบแน่นอน ไม่ต้องเดาด้วยเวลา
  // id เดิม (step.id) คงไว้เป็น additive — UI ไม่พึ่งแล้ว
  const mkNote = (step, f, extra) => {
    const n = { id: step.id, ...extra };
    if (step.nodeId != null) n.nodeId = step.nodeId;
    if (typeof f.subtitle === "function") n.view = f.subtitle(step.params || {});
    return n;
  };
  let current = baseRGB;
  let ownsCurrent = false; // baseRGB เป็นของ caller — ยังไม่ต้อง delete
  const notes = [];
  let upstream = null; // ดู ctx.upstream ด้านล่าง

  for (let i = 0; i < pipeline.length; i++) {
    const step = pipeline[i];
    if (!step || step.enabled === false) continue;
    const f = FILTERS[step.id];
    if (!f) continue;

    let input = current;
    let converted = null;

    // auto-convert เป็นขาวดำถ้า filter ต้องการ แต่ input ยังเป็นภาพสี
    // needsOf รองรับ merged filter ที่ needs ขึ้นกับวิธีที่เลือก (params)
    if (needsOf(f, step.params || {}) === "gray" && input.channels() > 1) {
      converted = toGray(input);
      input = converted;
      notes.push(mkNote(step, f, { autoGray: true }));
    }

    // ctx ให้ filter รายงานผลกลับ (เช่น จำนวนวัตถุที่พบ, ค่าเกณฑ์ Otsu) — Tier 2: เพิ่ม ctx.src2 (ภาพที่
    // สองสำหรับ op ที่ def.inputs===2) และ ctx.state (per-node persistent state, ดู getNodeState)
    // ctx.upstream (P-8) = { op, values } ของกล่อง "ล่าสุดที่มี ctx.values" บนสายหลักนี้ — ให้ pass_fail อ่านค่าจากกล่องวัดก่อนหน้า
    // (กล่องที่ปิด/bypass ถูกข้ามไปแล้วด้านบน จึงไม่ทับ · สาย B เป็น runPipeline แยก ค่าไม่ข้ามสาย)
    const ctx = { upstream };
    let src2Mat = null;
    let src2Owned = false;
    let out = null;
    let memoKey = null;
    try {
      if (step.src2 && usesSrc2(step)) {
        // src2 ประมวลผลจาก baseRGB เดียวกัน (กล้องมีแหล่งเดียวในแอปนี้ — ดู §A.3) ไม่ใช่จาก `current` ของ
        // chain หลัก เพราะ chain B เป็นสายแยกที่ย้อนกลับไปหา SourceNode ของตัวเอง ไม่ใช่ต่อจาก chain A
        // อยู่ "ใน" try: สาย B throw → catch ด้านล่างต้องคืน converted/current ของ step นี้ (เดิมอยู่นอก try = รั่วทุกเฟรม)
        const r2 = runPipeline(baseRGB, step.src2.pipeline || [], nodeState, activeNodeIds, frameMemo);
        src2Mat = r2.mat;
        src2Owned = true;
        for (const n of r2.notes) notes.push({ ...n, id: `${step.id}(B):${n.id}` }); // prefix กัน id ชนกับ note หลัก (UI ไม่พึ่ง id) — nodeId/view ของ node ต้นทางในสาย B คงไว้ตามเดิมจาก ...n
        ctx.src2 = src2Mat;
      }
      // ขนาดที่ใช้ fingerprint ต้อง = ขนาดภาพที่ "เข้าถึง op นี้จริง" (input, หลัง auto-gray/Crop/Resize
      // ของสายของ op เอง) ไม่ใช่ width/height ของเฟรมกล้องดิบระดับบนสุด (ดู §A.3 ตาราง reset rules) —
      // ต้องอยู่ "ใน" try นี้ ไม่ใช่ก่อนหน้า: ถ้า getNodeState throw (เช่น __dispose ของ state เก่า
      // throw เอง) ต้องยัง fall เข้า catch ด้านล่างเพื่อ delete converted/src2Mat/current ไม่งั้นรั่ว
      if (step.nodeId != null) {
        if (f.stateful === true && frameMemo) {
          // key ของ memo = "ขนาดภาพที่เข้ากล่องนี้" เท่านั้น ไม่ผูกกับ fingerprint ต้นน้ำ (RISK-1 ของ debugger):
          // หนึ่ง nodeId = กล่องเดียวในกราฟ จึงมีสายต้นน้ำเดียวเสมอ "หนึ่งกล่อง = รันครั้งเดียวต่อเฟรม" คือสัญญา
          // ที่ถูกต้องอยู่แล้ว · ถ้าวันหนึ่ง editor.js สกัด prefix ของกล่องผลลัพธ์กับของพรีวิวไม่ตรงกัน ผลเสียจะเป็น
          // แค่ "ใช้ผลของสายที่รันก่อน" ไม่ใช่ state ถูก reset ทุกเฟรมจนกล่องพังเงียบ ๆ (frame ค้างที่ 1 ตลอดกาล
          // และ mean_motion_px = 0 ตลอด) · ขนาดยังต้องอยู่ในคีย์ ไม่งั้นกล่องปลายน้ำอาจได้ Mat ผิดขนาด
          memoKey = input.cols + "x" + input.rows;
          const hit = frameMemo.get(step.nodeId);
          if (hit && hit.key === memoKey) {
            out = hit.mat.clone(); // เฟรมนี้ node นี้รันไปแล้ว (อีกกล่องผลลัพธ์/พรีวิว) — ห้ามรันซ้ำ ห้ามแตะ state
            ctx.info = hit.info; // string — ไม่ต้องคัดลอก
            ctx.values = cloneValues(hit.values); // ห้ามแชร์ object เดียวกันข้าม note (ดู cloneValues)
            memoKey = null; // ไม่ต้องบันทึกซ้ำ
          }
        }
        // prefix ต้นน้ำ "จบที่ step นี้" (รวมตัวเอง) — แก้กล่องที่อยู่ปลายน้ำต้องไม่ทำให้ reset (0-A)
        if (!out) ctx.state = getNodeState(step.nodeId, pipeline.slice(0, i + 1).map(stepFingerprint), input.cols, input.rows, nodeState);
      }
      if (!out) out = f.apply(cv, input, step.params || {}, ctx);
      if (memoKey) {
        const old = frameMemo.get(step.nodeId);
        // ลบ entry ออกจาก Map ก่อน แล้วค่อย delete Mat — ถ้า delete throw ต้องไม่เหลือ entry ที่ชี้ Mat ที่ตายแล้ว
        // (finally ของ computeFrame วนลบทุก entry จะกลายเป็น double-delete)
        if (old) { frameMemo.delete(step.nodeId); old.mat.delete(); }
        frameMemo.set(step.nodeId, { key: memoKey, mat: out.clone(), info: ctx.info, values: ctx.values });
      }
    } catch (err) {
      // ถ้า filter ใด throw ต้องคืน Mat ที่ worker ถืออยู่ก่อนส่ง error กลับ
      // ไม่เช่นนั้น error ซ้ำใน live camera จะกิน WASM heap จนแอปค้างจริง
      if (out) out.delete(); // 18d: out มีแล้วได้ ถ้า clone ของ memo throw
      if (converted) converted.delete();
      if (src2Owned && src2Mat) src2Mat.delete();
      if (ownsCurrent) current.delete();
      // UI เฟส 3.6: บอกว่า step ไหนพัง (nodeId เดียวกับ note) — ชั้นในสุด (สาย B) ตั้งก่อน ชั้นนอกห้ามทับ
      // ค่า throw ที่ไม่ใช่ object (ข้อยกเว้นของ WASM เป็นตัวเลข) ห่อเป็น Error เพื่อแนบ nodeId ได้
      if (step.nodeId != null) {
        // ข้อยกเว้นของ OpenCV.js เป็นเลขพอยน์เตอร์ — ถอดเป็นข้อความจริงถ้า cv มี exceptionFromPtr
        let text = String(err);
        if (typeof err === "number" && typeof cv.exceptionFromPtr === "function") {
          try { text = cv.exceptionFromPtr(err).msg || text; } catch { /* ใช้ตัวเลขเดิม */ }
        }
        const e = err !== null && typeof err === "object" ? err : new Error(text);
        if (e.nodeId === undefined) { try { e.nodeId = step.nodeId; } catch { /* object ที่ freeze — ข้ามไป */ } }
        throw e;
      }
      throw err;
    }
    if (ctx.info) notes.push(mkNote(step, f, { info: ctx.info }));
    if (ctx.values) {
      // pass_fail เองไม่เป็น upstream ของใคร — กล่อง pass_fail ซ้อนกันจึงอ่านกล่องวัดตัวเดียวกันได้
      const opId = opIdOfStep(step);
      // op = id ของ "วิธี" จริง (กล่องรวม → mode) — batch CSV (P-8 C-B) ใช้ตั้งชื่อคอลัมน์ · เพิ่มแบบ additive UI เดิมไม่อ่าน
      notes.push(mkNote(step, f, { values: ctx.values, op: opId })); // Tier 2 §A.2 — ช่องผลลัพธ์ตัวเลข
      if (opId !== "pass_fail") upstream = { op: opId, values: ctx.values };
    }

    if (converted) converted.delete();
    if (src2Owned && src2Mat) src2Mat.delete(); // ctx.src2 เป็นแค่ view ที่ op ยืมใช้ชั่วคราว ไม่ใช่เจ้าของ
    if (ownsCurrent) current.delete();
    current = out;
    ownsCurrent = true;
  }

  // ถ้าไม่มี filter ทำงานเลย ให้ clone ฐานไว้ (จะได้ delete ได้สม่ำเสมอ)
  if (!ownsCurrent) {
    const clone = new cv.Mat();
    current.copyTo(clone);
    current = clone;
  }
  return { mat: current, notes };
}

// ประมวลผล 1 เฟรม → ผลลัพธ์ของทุกกล่อง (ใช้ทั้ง live และ export)
// opts.isolatedState (18d, 0-D) = export: ใช้ Map state ชั่วคราวของตัวเอง (ภาพ export แต่ละภาพ = "เฟรมแรก"
// ของ state ใหม่เสมอ) ห้ามแตะ/sweep state ของภาพสด — ถ้าวันหนึ่งจะให้ export หลายภาพเป็น "ลำดับเฟรม" (U1)
// ให้ caller ถือ Map ข้ามภาพแล้วส่งเข้ามาแทน new Map() ตรงนี้ (แก้ที่ computeFrame จุดเดียว)
export function computeFrame(width, height, buffer, pipelines, opts = {}) {
  const isolated = opts.isolatedState === true;
  const store = isolated ? new Map() : nodeState; // export ห้ามแตะ state ของภาพสด
  const frameMemo = new Map(); // nodeId -> ผลของ op stateful ในเฟรมนี้ (กันรันซ้ำ, ดู runPipeline)
  const imageData = new ImageData(new Uint8ClampedArray(buffer), width, height);
  const srcRGBA = cv.matFromImageData(imageData);
  let baseRGB = null;

  const results = [];
  const transfer = [];
  const notes = [];
  try {
    baseRGB = new cv.Mat();
    cv.cvtColor(srcRGBA, baseRGB, cv.COLOR_RGBA2RGB);

    // Tier 2 §A.3 — เก็บ nodeId ที่ยังใช้งานจริงในเฟรมนี้ (ทุกกล่องผลลัพธ์ รวม src2 sub-pipeline) แล้ว
    // sweep ทิ้ง state ของ node ที่หายไปแล้ว ครั้งเดียวก่อนวน pipeline ทั้งหมด (ไม่ใช่ต่อ pipeline) —
    // 18d: อยู่ใน try แล้ว (ถ้า __dispose throw ต้องไม่ทำ srcRGBA/baseRGB รั่ว) และ export ไม่ sweep store สด
    const activeNodeIds = new Set();
    for (const pl of pipelines || []) collectNodeIds(pl && pl.pipeline, activeNodeIds);
    if (!isolated) sweepNodeState(activeNodeIds, store);

    for (const pl of pipelines || []) {
      if (!pl || !pl.connected) continue;
      let piped = null;
      let rgba = null;
      try {
        const r = runPipeline(baseRGB, pl.pipeline || [], store, activeNodeIds, frameMemo);
        piped = r.mat;
        if (!pl.silentNotes && r.notes) for (const n of r.notes) notes.push(n);
        rgba = toRGBA(piped);
        const outData = new Uint8ClampedArray(rgba.data);
        const res = { displayId: pl.displayId, width: rgba.cols, height: rgba.rows, buffer: outData.buffer };
        // export (P-8 C-B): แนบ ctx.values ของทุกกล่องที่มีค่า (ชุดเดียวกับแผง "ค่าที่วัดได้") — ไม่แตะ buffer ภาพ
        if (opts.collectValues === true && r.notes) {
          res.values = r.notes.filter((n) => n.values).map((n) => ({ op: n.op, id: n.id, nodeId: n.nodeId, values: n.values }));
        }
        results.push(res);
        transfer.push(outData.buffer);
      } finally {
        if (piped) piped.delete();
        if (rgba) rgba.delete();
      }
    }
  } finally {
    srcRGBA.delete();
    if (baseRGB) baseRGB.delete();
    for (const e of frameMemo.values()) e.mat.delete();
    if (isolated) sweepNodeState(new Set(), store); // ทิ้ง state ชั่วคราวของ export ทั้งหมด
  }
  return { results, transfer, notes };
}

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === "init") {
    await initOpenCV();
    ready = true;
    self.postMessage({ type: "ready" });
    return;
  }

  if (msg.type === "frame") {
    if (!ready) return;
    try {
      const { results, transfer, notes } = computeFrame(msg.width, msg.height, msg.buffer, msg.pipelines || []);
      self.postMessage({ type: "result", results, notes }, transfer);
    } catch (err) {
      self.postMessage({ type: "error", message: err.message, nodeId: err.nodeId });
    }
    return;
  }

  // ประมวลผลภาพเดี่ยวสำหรับ export (batch) — ตอบกลับพร้อม reqId ไม่ยุ่งกับจอ live
  if (msg.type === "export") {
    if (!ready) {
      self.postMessage({ type: "export-result", reqId: msg.reqId, results: [], error: "ยังไม่พร้อม" });
      return;
    }
    try {
      // 18d (0-D): export ใช้ state แยกของตัวเอง — ภาพ export ต้องไม่ไหลเข้า MOG2/optical flow ของกล้องสด
      const { results, transfer } = computeFrame(msg.width, msg.height, msg.buffer, msg.pipelines || [], { isolatedState: true, collectValues: true });
      self.postMessage({ type: "export-result", reqId: msg.reqId, results }, transfer);
    } catch (err) {
      self.postMessage({ type: "export-result", reqId: msg.reqId, results: [], error: err.message });
    }
    return;
  }
};

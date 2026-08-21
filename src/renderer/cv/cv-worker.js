// Web Worker: โหลด OpenCV.js + รัน pipeline ของ filter หลายชั้น แยกจาก main thread
import cv from "@techstark/opencv-js";
import { FILTERS, needsOf } from "../filters/registry.js";

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

// รัน pipeline: [{ id, params, enabled }] ตามลำดับ
// คืน { mat, notes } โดย mat เป็นผลลัพธ์ (caller ต้อง delete)
function runPipeline(baseRGB, pipeline) {
  let current = baseRGB;
  let ownsCurrent = false; // baseRGB เป็นของ caller — ยังไม่ต้อง delete
  const notes = [];

  for (const step of pipeline) {
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
      notes.push({ id: step.id, autoGray: true });
    }

    // ctx ให้ filter รายงานผลกลับ (เช่น จำนวนวัตถุที่พบ, ค่าเกณฑ์ Otsu)
    const ctx = {};
    let out;
    try {
      out = f.apply(cv, input, step.params || {}, ctx);
    } catch (err) {
      // ถ้า filter ใด throw ต้องคืน Mat ที่ worker ถืออยู่ก่อนส่ง error กลับ
      // ไม่เช่นนั้น error ซ้ำใน live camera จะกิน WASM heap จนแอปค้างจริง
      if (converted) converted.delete();
      if (ownsCurrent) current.delete();
      throw err;
    }
    if (ctx.info) notes.push({ id: step.id, info: ctx.info });

    if (converted) converted.delete();
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
function computeFrame(width, height, buffer, pipelines) {
  const imageData = new ImageData(new Uint8ClampedArray(buffer), width, height);
  const srcRGBA = cv.matFromImageData(imageData);
  const baseRGB = new cv.Mat();
  cv.cvtColor(srcRGBA, baseRGB, cv.COLOR_RGBA2RGB);

  const results = [];
  const transfer = [];
  const notes = [];
  try {
    for (const pl of pipelines || []) {
      if (!pl || !pl.connected) continue;
      let piped = null;
      let rgba = null;
      try {
        const r = runPipeline(baseRGB, pl.pipeline || []);
        piped = r.mat;
        if (!pl.silentNotes && r.notes) for (const n of r.notes) notes.push(n);
        rgba = toRGBA(piped);
        const outData = new Uint8ClampedArray(rgba.data);
        results.push({ displayId: pl.displayId, width: rgba.cols, height: rgba.rows, buffer: outData.buffer });
        transfer.push(outData.buffer);
      } finally {
        if (piped) piped.delete();
        if (rgba) rgba.delete();
      }
    }
  } finally {
    srcRGBA.delete();
    baseRGB.delete();
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
      self.postMessage({ type: "error", message: err.message });
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
      const { results, transfer } = computeFrame(msg.width, msg.height, msg.buffer, msg.pipelines || []);
      self.postMessage({ type: "export-result", reqId: msg.reqId, results }, transfer);
    } catch (err) {
      self.postMessage({ type: "export-result", reqId: msg.reqId, results: [], error: err.message });
    }
    return;
  }
};

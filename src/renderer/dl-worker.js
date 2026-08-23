// Web Worker: รัน YOLO inference (onnxruntime-web) แยกจาก main thread — กัน UI ค้าง/แลค
// main ทำ preprocess (letterbox) แล้วส่ง tensor มา, worker รัน session + decode + NMS แล้วคืนกล่อง
import * as ort from "onnxruntime-web";

let session = null;
let inputName = "";
let outputName = "";
let inputSize = 640;

function iouOf(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const uni = a.w * a.h + b.w * b.h - inter;
  return uni > 0 ? inter / uni : 0;
}
function nms(boxes, iouThresh) {
  boxes.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const b of boxes) {
    if (kept.every((k) => k.cls !== b.cls || iouOf(k, b) < iouThresh)) kept.push(b);
  }
  return kept;
}
// ถอดผลลัพธ์แบบ end-to-end / NMS-baked export (เช่น model.export(..., nms=True)):
// [1, จำนวนกล่องสูงสุด, 6] โดยแต่ละแถวคือ [x1, y1, x2, y2, score, class] พิกัดพิกเซลจริงบนภาพ
// letterboxed (ก่อนหาร r/ลบ dx,dy กลับ) — ผ่าน NMS มาจากในโมเดลแล้ว ไม่ต้องทำ NMS ซ้ำ
function decodeNmsBaked(dims, d, r, dx, dy, conf) {
  const n = dims[1];
  const boxes = [];
  for (let i = 0; i < n; i++) {
    const base = i * 6;
    const x1 = d[base];
    const y1 = d[base + 1];
    const x2 = d[base + 2];
    const y2 = d[base + 3];
    const score = d[base + 4];
    const cls = d[base + 5] | 0;
    if (score < conf) continue;
    boxes.push({ x: (x1 - dx) / r, y: (y1 - dy) / r, w: (x2 - x1) / r, h: (y2 - y1) / r, score, cls });
  }
  return boxes;
}

// ถอดผลลัพธ์ YOLOv8 แบบ raw detection head: [1, 4+nc, n] → กล่องบนพิกัดภาพจริง (ใช้ r/dx/dy จาก letterbox)
// ยังไม่ผ่าน NMS จากในโมเดล ต้องทำ NMS เอง (ดู nms() ด้านบน)
function decodeRawHead(dims, d, r, dx, dy, conf) {
  const ch = dims[1];
  const n = dims[2];
  const nc = ch - 4;
  const boxes = [];
  for (let i = 0; i < n; i++) {
    let best = 0;
    let bi = 0;
    for (let c = 0; c < nc; c++) {
      const s = d[(4 + c) * n + i];
      if (s > best) {
        best = s;
        bi = c;
      }
    }
    if (best < conf) continue;
    const cx = d[i];
    const cy = d[n + i];
    const w = d[2 * n + i];
    const h = d[3 * n + i];
    boxes.push({ x: (cx - w / 2 - dx) / r, y: (cy - h / 2 - dy) / r, w: w / r, h: h / r, score: best, cls: bi });
  }
  return boxes;
}

function decode(dims, d, r, dx, dy, conf, iou) {
  // แยกสองรูปแบบ output ด้วย "ขนาดช่องสุดท้าย": โมเดล NMS-baked จะมีเสมอ 6 ค่า/แถว
  // (x1,y1,x2,y2,score,class) ส่วน raw head ช่องกลาง (4+nc) แทบไม่มีทางเท่ากับ 6 พอดี
  // (nc=2 เท่านั้นถึงจะชนกัน ซึ่งเป็นไปได้ยากมากในทางปฏิบัติ)
  if (dims.length === 3 && dims[2] === 6) {
    return decodeNmsBaked(dims, d, r, dx, dy, conf); // ผ่าน NMS จากโมเดลแล้ว ไม่ต้อง nms() ซ้ำ
  }
  return nms(decodeRawHead(dims, d, r, dx, dy, conf), iou);
}

self.onmessage = async (e) => {
  const m = e.data;

  if (m.type === "load") {
    try {
      ort.env.wasm.wasmPaths = m.wasmBase;
      ort.env.wasm.numThreads = 1; // เลี่ยง SharedArrayBuffer
      session = await ort.InferenceSession.create(m.bytes, { executionProviders: ["wasm"] });
      inputName = session.inputNames[0];
      outputName = session.outputNames[0];
      const dims = session.inputMetadata?.[0]?.dimensions || session.inputMetadata?.[0]?.shape;
      inputSize = Array.isArray(dims) && typeof dims[2] === "number" && dims[2] > 0 ? dims[2] : 640;
      self.postMessage({ type: "loaded", inputSize });
    } catch (err) {
      self.postMessage({ type: "loaded", error: err.message });
    }
    return;
  }

  if (m.type === "detect") {
    if (!session) {
      self.postMessage({ type: "result", reqId: m.reqId, boxes: [] });
      return;
    }
    try {
      const t = new ort.Tensor("float32", m.data, [1, 3, inputSize, inputSize]);
      const out = await session.run({ [inputName]: t });
      const o = out[outputName];
      const boxes = decode(o.dims, o.data, m.r, m.dx, m.dy, m.conf, m.iou);
      self.postMessage({ type: "result", reqId: m.reqId, boxes });
    } catch (err) {
      self.postMessage({ type: "result", reqId: m.reqId, boxes: [], error: err.message });
    }
    return;
  }
};

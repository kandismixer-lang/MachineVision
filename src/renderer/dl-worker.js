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
// ถอดผลลัพธ์ YOLOv8: [1, 4+nc, n] → กล่องบนพิกัดภาพจริง (ใช้ r/dx/dy จาก letterbox)
function decode(dims, d, r, dx, dy, conf, iou) {
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
  return nms(boxes, iou);
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

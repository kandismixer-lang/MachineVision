// Deep Learning — YOLO detect (proxy ไป Web Worker เพื่อไม่ให้ inference บล็อก UI)
// main thread: letterbox (เบา) + วาดกล่อง · worker: session.run + decode + NMS (หนัก)
let worker = null;
let ready = false;
let inputSize = 640;
let seq = 0;
const waiters = new Map();
const loadResolvers = []; // คิว resolver ของ loadModel (FIFO) — กันโหลดซ้อนแล้ว promise สลับกัน

const pre = document.createElement("canvas"); // canvas สำหรับ letterbox

// ชื่อคลาสมาตรฐาน COCO 80
export const COCO = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
  "hair drier", "toothbrush",
];

// ดึงชื่อคลาสจาก metadata ที่ ultralytics ฝังในไฟล์ .onnx (ใช้ภายในไฟล์นี้)
function extractOnnxNames(bytes) {
  try {
    const s = new TextDecoder("latin1").decode(bytes);
    const i = s.indexOf("names");
    if (i < 0) return null;
    const brace = s.indexOf("{", i);
    const end = s.indexOf("}", brace);
    if (brace < 0 || end < 0) return null;
    const dict = s.slice(brace, end + 1);
    const names = [...dict.matchAll(/['"]([^'"]+)['"]/g)].map((m) => m[1]);
    return names.length ? names : null;
  } catch {
    return null;
  }
}

function ensureWorker() {
  if (worker) return;
  worker = new Worker(new URL("./dl-worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "loaded") {
      if (!m.error) {
        ready = true;
        inputSize = m.inputSize;
      }
      const r = loadResolvers.shift(); // ตอบตามลำดับที่ส่ง (worker โหลดทีละไฟล์)
      if (r) r(m);
    } else if (m.type === "result") {
      const w = waiters.get(m.reqId);
      if (w) {
        waiters.delete(m.reqId);
        w(m.boxes || []);
      }
    }
  };
  // worker พังแบบ uncaught → ทิ้ง worker ที่ตายแล้ว (รอบหน้า ensureWorker สร้างใหม่)
  // + ปลดล็อกโหลดที่ค้างทั้งคิว ไม่ให้ UI ค้าง "กำลังโหลด" ตลอด
  worker.onerror = () => {
    try { worker.terminate(); } catch { /* ignore */ }
    worker = null;
    ready = false;
    while (loadResolvers.length) {
      loadResolvers.shift()({ error: "โหลดโมเดลล้มเหลว (worker ขัดข้อง) — ลองใหม่อีกครั้ง" });
    }
    // ปลด detect() ที่ค้าง → คืน [] ไม่งั้น detBusy ค้าง true จับภาพหยุดถาวรทั้ง session
    for (const w of waiters.values()) w([]);
    waiters.clear();
  };
}

// โหลดโมเดลจาก bytes (Uint8Array ของ .onnx) — ส่งเข้า worker
// serialize โหลดทีละไฟล์: worker ใช้ session global ตัวเดียว + reply มาไม่ตรงลำดับ postMessage
// ได้ (async) → ถ้าโหลดซ้อนต้องเข้าคิว ไม่งั้นผลจับคู่ผิดไฟล์
let loadChain = Promise.resolve();
export function loadModel(bytes) {
  ensureWorker();
  const names = extractOnnxNames(bytes); // ดึงชื่อคลาสก่อน transfer buffer
  const task = async () => {
    const wasmBase = new URL("./onnx/", location.href).href; // path wasm (absolute — worker ใช้ได้)
    const p = new Promise((res) => loadResolvers.push(res));
    worker.postMessage({ type: "load", bytes, wasmBase }, [bytes.buffer]);
    const m = await p;
    ready = false; // ตั้งใหม่จากผล
    if (m.error) throw new Error(m.error);
    ready = true;
    inputSize = m.inputSize;
    return { inputSize, names };
  };
  const result = loadChain.then(task, task); // ต่อคิว: โหลดก่อนหน้าเสร็จค่อยเริ่มตัวถัดไป
  loadChain = result.catch(() => {}); // กัน chain ขาดถ้าตัวก่อนหน้า error
  return result;
}

export function isLoaded() {
  return ready;
}

// letterbox: ย่อภาพให้พอดี inputSize คงอัตราส่วน เติมสีเทา 114 — คืน Float32 CHW + meta
function letterbox(src, W, H) {
  const r = Math.min(inputSize / W, inputSize / H);
  const nw = Math.round(W * r);
  const nh = Math.round(H * r);
  const dx = Math.floor((inputSize - nw) / 2);
  const dy = Math.floor((inputSize - nh) / 2);
  pre.width = inputSize;
  pre.height = inputSize;
  const ctx = pre.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, inputSize, inputSize);
  ctx.drawImage(src, 0, 0, W, H, dx, dy, nw, nh);
  const img = ctx.getImageData(0, 0, inputSize, inputSize).data;
  const area = inputSize * inputSize;
  const data = new Float32Array(area * 3);
  for (let i = 0; i < area; i++) {
    data[i] = img[i * 4] / 255;
    data[i + area] = img[i * 4 + 1] / 255;
    data[i + 2 * area] = img[i * 4 + 2] / 255;
  }
  return { data, r, dx, dy };
}

// ตรวจจับ — คืน Promise ของ [{ x, y, w, h, score, cls, label }]
export function detect(src, W, H, { conf = 0.25, iou = 0.45, labels } = {}) {
  if (!ready) return Promise.resolve([]);
  const { data, r, dx, dy } = letterbox(src, W, H);
  const id = ++seq;
  return new Promise((res) => {
    waiters.set(id, (boxes) =>
      res(boxes.map((b) => ({ ...b, label: (labels && labels[b.cls]) || String(b.cls) })))
    );
    worker.postMessage({ type: "detect", reqId: id, data, r, dx, dy, conf, iou }, [data.buffer]);
  });
}

// วาดกล่อง detect ลง ctx
export function drawDetections(ctx, boxes, scale = 1) {
  ctx.lineWidth = 2 * scale;
  ctx.font = `${16 * scale}px sans-serif`;
  ctx.textBaseline = "alphabetic";
  for (const b of boxes) {
    ctx.strokeStyle = "#00e0ff";
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    const text = `${b.label} ${(b.score * 100) | 0}%`;
    const tw = ctx.measureText(text).width + 6 * scale;
    const th = 20 * scale;
    ctx.fillStyle = "#00e0ff";
    ctx.fillRect(b.x, b.y - th, tw, th);
    ctx.fillStyle = "#001018";
    ctx.fillText(text, b.x + 3 * scale, b.y - 5 * scale);
  }
}

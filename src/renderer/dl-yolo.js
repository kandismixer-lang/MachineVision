// Deep Learning — YOLO detect (proxy ไป Web Worker เพื่อไม่ให้ inference บล็อก UI)
// main thread: letterbox (เบา) + วาดกล่อง · worker: session.run + decode + NMS (หนัก)
import { tileGrid, mergeTileBoxes } from "./dl-tiles.js";
import { decodeOutputs, kindFromOutputRanks, guessKind } from "./dl-decode.js";
let worker = null;
let ready = false;
let inputSize = 640;
let segModel = false; // โมเดลที่โหลดล่าสุดเป็น segmentation ไหม
let modelKind = "yolo"; // 'yolo' | 'yoloseg' | 'rfdetr'
// ตัวรัน: "native" = onnxruntime-node ใน Electron main (GPU DirectML → CPU, เร็วกว่ามาก) · "worker" = WASM ใน Web Worker (สำรอง)
let backend = "worker";
let nativeOutputNames = [];
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
// metadata เป็น str(dict) ของ Python เก็บแบบ UTF-8 → ต้องถอด utf-8 (latin1 ทำชื่อไทยเพี้ยนเป็น "à¸£à¸...")
// Python ใส่ ' หรือ " แล้วแต่ชื่อ (เช่น "it's") → อ่านทีละคู่ key: 'value' ตาม index จริง
function extractOnnxNames(bytes) {
  try {
    const s = new TextDecoder("utf-8").decode(bytes);
    const i = s.indexOf("names");
    if (i < 0) return null;
    const brace = s.indexOf("{", i);
    if (brace < 0) return null;
    const re = /\s*(\d+)\s*:\s*(['"])((?:\\.|(?!\2)[^\\])*)\2\s*([,}])/y;
    re.lastIndex = brace + 1;
    const names = [];
    let m;
    while ((m = re.exec(s))) {
      names[Number(m[1])] = m[3].replace(/\\(.)/g, "$1");
      if (m[4] === "}") break;
    }
    return names.length ? Array.from(names, (n) => n ?? "") : null;
  } catch {
    return null;
  }
}

// ดึง imgsz จาก metadata ที่ ultralytics ฝังในไฟล์ .onnx (เช่น "imgsz": [1280, 1280])
// จำเป็นเพราะ session.inputMetadata ไม่มีใน onnxruntime-web (undefined เสมอ) — อ่านจากไบต์แทน
function extractOnnxImgsz(bytes) {
  try {
    const s = new TextDecoder("latin1").decode(bytes);
    const i = s.indexOf("imgsz");
    if (i < 0) return null;
    const bracket = s.indexOf("[", i);
    if (bracket < 0) return null;
    const m = /-?\d+/.exec(s.slice(bracket, bracket + 32));
    if (!m) return null;
    const v = parseInt(m[0], 10);
    return Number.isFinite(v) && v > 0 ? v : null;
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
        segModel = !!m.seg;
        modelKind = m.kind || "yolo";
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
  const imgsz = extractOnnxImgsz(bytes); // ดึง input size จริง (เช่น 1280) ก่อน transfer buffer
  const task = async () => {
    // ลอง native ก่อน (มีเฉพาะในแอป Electron) — ไม่สำเร็จ (ไม่มีแพ็กเกจ / GPU+CPU โหลดไม่ได้) ค่อยใช้ WASM worker
    if (typeof window !== "undefined" && window.mvDL?.nativeLoad) {
      let r;
      try { r = await window.mvDL.nativeLoad(bytes, imgsz); } catch (e) { r = { ok: false, error: e?.message || String(e) }; }
      if (r && r.ok) {
        backend = "native";
        nativeOutputNames = r.outputNames;
        inputSize = r.inputSize;
        modelKind = r.outputRanks.every((x) => x > 0) ? kindFromOutputRanks(r.outputRanks) : guessKind(r.outputNames);
        segModel = modelKind === "yoloseg";
        ready = true;
        return { inputSize, names, seg: segModel, kind: modelKind, threads: 0, engine: r.ep === "dml" ? "GPU (DirectML)" : "CPU native" };
      }
      console.warn("[Model Detect] native ใช้ไม่ได้ → ใช้ WASM:", r?.error);
    }
    backend = "worker";
    const wasmBase = new URL("./onnx/", location.href).href; // path wasm (absolute — worker ใช้ได้)
    const p = new Promise((res) => loadResolvers.push(res));
    worker.postMessage({ type: "load", bytes, wasmBase, imgsz }, [bytes.buffer]);
    const m = await p;
    ready = false; // ตั้งใหม่จากผล
    if (m.error) throw new Error(m.error);
    ready = true;
    inputSize = m.inputSize;
    segModel = !!m.seg;
    modelKind = m.kind || "yolo";
    const threads = m.threads || 1;
    return { inputSize, names, seg: segModel, kind: modelKind, threads, engine: `WASM ${threads} thread` };
  };
  const result = loadChain.then(task, task); // ต่อคิว: โหลดก่อนหน้าเสร็จค่อยเริ่มตัวถัดไป
  loadChain = result.catch(() => {}); // กัน chain ขาดถ้าตัวก่อนหน้า error
  return result;
}

export function isLoaded() {
  return ready;
}

export function isSegModel() {
  return segModel;
}

export function isRfdetr() {
  return modelKind === "rfdetr";
}

// RF-DETR preprocess: resize ตรง ๆ เป็น inputSize×inputSize (ไม่ letterbox — บิดสัดส่วนได้ กล่อง normalize
// เทียบภาพเต็มอยู่แล้ว) + normalize แบบ ImageNet (mean/std) ตามที่ DETR ใช้ ไม่ใช่แค่หาร 255
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
function rfdetrPreprocess(src, W, H) {
  pre.width = inputSize;
  pre.height = inputSize;
  const ctx = pre.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(src, 0, 0, W, H, 0, 0, inputSize, inputSize); // resize เต็ม (distort)
  const img = ctx.getImageData(0, 0, inputSize, inputSize).data;
  const area = inputSize * inputSize;
  const data = new Float32Array(area * 3);
  for (let i = 0; i < area; i++) {
    data[i] = (img[i * 4] / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    data[i + area] = (img[i * 4 + 1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    data[i + 2 * area] = (img[i * 4 + 2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
  return { data };
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
  const id = ++seq;
  const resolveMap = (res) =>
    waiters.set(id, (boxes) =>
      res(boxes.map((b) => ({ ...b, label: (labels && labels[b.cls]) || String(b.cls) })))
    );
  if (backend === "native") {
    // native: preprocess เหมือนเดิม → main รัน → ถอดผลที่นี่ด้วยโค้ดชุดเดียวกับ worker (dl-decode.js)
    let data, meta;
    if (modelKind === "rfdetr") {
      ({ data } = rfdetrPreprocess(src, W, H));
      meta = { origW: W, origH: H, conf };
    } else {
      const lb = letterbox(src, W, H);
      data = lb.data;
      meta = { r: lb.r, dx: lb.dx, dy: lb.dy, conf, iou };
    }
    const kindNow = modelKind, namesNow = nativeOutputNames, sizeNow = inputSize;
    return window.mvDL.nativeRun(data, sizeNow).then((res) => {
      const boxes = res.skipped ? [] : decodeOutputs(kindNow, namesNow, res.outputs, meta, sizeNow);
      return boxes.map((b) => ({ ...b, label: (labels && labels[b.cls]) || String(b.cls) }));
    });
  }
  if (modelKind === "rfdetr") {
    const { data } = rfdetrPreprocess(src, W, H);
    return new Promise((res) => {
      resolveMap(res);
      // RF-DETR: กล่อง normalize เทียบภาพเต็ม → ส่งขนาดภาพจริงไป map กลับ (ไม่มี letterbox r/dx/dy)
      worker.postMessage({ type: "detect", reqId: id, data, origW: W, origH: H, conf }, [data.buffer]);
    });
  }
  const { data, r, dx, dy } = letterbox(src, W, H);
  return new Promise((res) => {
    resolveMap(res);
    worker.postMessage({ type: "detect", reqId: id, data, r, dx, dy, conf, iou }, [data.buffer]);
  });
}

// ตรวจแบบแบ่ง tile — หั่นภาพเป็นชิ้น tile×tile (เหลื่อมกัน overlap 0..0.9) ตรวจทุกชิ้น แล้วรวมกล่องกลับเป็นพิกัดภาพเต็ม
// ทุกชิ้นถูก preprocess ใน loop เดียวแบบ synchronous (detect() อ่านพิกเซลก่อน return) → ทุกชิ้นมาจากเฟรมเดียวกัน
// แม้ src เป็นวิดีโอสด · worker รันทีละชิ้นตามคิว
const tileCanvas = typeof document !== "undefined" ? document.createElement("canvas") : null;
export function detectTiled(src, W, H, { tile = 640, overlap = 0.2, onProgress = null, ...opts } = {}) {
  const rects = tileGrid(W, H, tile, overlap);
  if (!ready || rects.length <= 1) return detect(src, W, H, opts);
  const ctx = tileCanvas.getContext("2d", { willReadFrequently: true });
  let done = 0;
  if (onProgress) onProgress(0, rects.length);
  const jobs = rects.map((r) => {
    tileCanvas.width = r.w;
    tileCanvas.height = r.h;
    ctx.drawImage(src, r.x, r.y, r.w, r.h, 0, 0, r.w, r.h);
    return detect(tileCanvas, r.w, r.h, opts).then((bs) => {
      done += 1;
      if (onProgress) onProgress(done, rects.length);
      return bs.map((b) => ({ ...b, x: b.x + r.x, y: b.y + r.y }));
    });
  });
  return Promise.all(jobs).then((lists) => mergeTileBoxes(lists.flat()));
}

// canvas ชั่วคราวสำหรับ blit mask (ความละเอียด proto) ก่อน scale ทับกล่อง
const maskTmp = typeof document !== "undefined" ? document.createElement("canvas") : null;

// สีทึบ (hex) ต่อ index คลาส — วน 10 สีเหมือน palette ของ dataset (mask segmentation)
const MASK_COLORS = [
  [59, 130, 246], [239, 68, 68], [34, 197, 94], [234, 179, 8], [168, 85, 247],
  [6, 182, 212], [249, 115, 22], [236, 72, 153], [132, 204, 22], [20, 184, 166],
];

// วาด mask ของกล่องหนึ่ง (ครอปความละเอียด proto) ทับพื้นที่กล่องบน ctx
function drawMask(ctx, b) {
  if (!maskTmp || !b.mask) return;
  const { mw, mh, data } = b.mask;
  if (!mw || !mh) return;
  const [r, g, bl] = MASK_COLORS[b.cls % MASK_COLORS.length];
  maskTmp.width = mw;
  maskTmp.height = mh;
  const mctx = maskTmp.getContext("2d");
  const img = mctx.createImageData(mw, mh);
  for (let i = 0; i < data.length; i++) {
    const on = data[i] > 0;
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = bl;
    img.data[i * 4 + 3] = on ? 120 : 0; // โปร่งแสง ~47% เห็นภาพใต้ mask
  }
  mctx.putImageData(img, 0, 0);
  const prev = ctx.imageSmoothingEnabled;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(maskTmp, b.x, b.y, b.w, b.h); // scale จาก proto-crop → พื้นที่กล่องในภาพจริง
  ctx.imageSmoothingEnabled = prev;
}

// วาดกล่อง detect (+ mask ถ้าเป็น segmentation) ลง ctx
export function drawDetections(ctx, boxes, scale = 1) {
  // วาด mask ก่อน (ชั้นล่าง) แล้วค่อยกล่อง+ป้าย (ชั้นบน)
  for (const b of boxes) if (b.mask) drawMask(ctx, b);
  ctx.lineWidth = 2 * scale;
  ctx.font = `${16 * scale}px sans-serif`;
  ctx.textBaseline = "alphabetic";
  for (const b of boxes) {
    ctx.strokeStyle = "#00e0ff";
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    const text = `${b.label} ${(b.score * 100) | 0}%`;
    const tw = ctx.measureText(text).width + 6 * scale;
    const th = 20 * scale;
    // ป้ายไม่ล้นขอบภาพ: ชิดขวาถ้าเกินขอบขวา · ย้ายลงใต้กรอบถ้าเกินขอบบน
    const cw = ctx.canvas?.width || Infinity;
    const lx = Math.max(0, Math.min(b.x, cw - tw));
    const ly = b.y - th >= 0 ? b.y - th : b.y + b.h;
    ctx.fillStyle = "#00e0ff";
    ctx.fillRect(lx, ly, tw, th);
    ctx.fillStyle = "#001018";
    ctx.fillText(text, lx + 3 * scale, ly + th - 5 * scale);
  }
}

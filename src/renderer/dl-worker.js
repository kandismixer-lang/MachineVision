// Web Worker: รัน YOLO inference (onnxruntime-web) แยกจาก main thread — กัน UI ค้าง/แลค
// main ทำ preprocess (letterbox) แล้วส่ง tensor มา, worker รัน session + decode + NMS แล้วคืนกล่อง
// "onnxruntime-web/wasm" (ไม่ใช่ตัว bundle ที่ฝัง glue ไว้ในตัว): glue ort-wasm-simd-threaded.mjs โหลดจาก wasmPaths (public/onnx)
// → ตอนรันหลาย thread, emscripten สร้าง pthread worker จากไฟล์ glue นั้นได้ถูกตัว — ถ้าใช้ตัว bundle มันจะเอา
// ไฟล์ worker นี้เองไปเป็น pthread แล้วค้างตอนโหลดโมเดล
import * as ort from "onnxruntime-web/wasm";
import { guessKind, decodeOutputs } from "./dl-decode.js";

let session = null;
let inputName = "";
let outputName = "";
let outputNames = [];
let inputSize = 640;
let isSeg = false; // โมเดล segmentation (มี 2 output: detection head + mask prototypes)
let kind = "yolo"; // 'yolo' | 'yoloseg' | 'rfdetr'

// probe โมเดล 2-output (onnxruntime-web ไม่บอก input/output shape ก่อนรัน): ลองรัน dummy หลายขนาด
// อันแรกที่ session.run ไม่ throw = ขนาด input จริง — คืน { size, out } เพื่อดู "รูปร่าง output" ต่อ
// (แยก yoloseg vs rfdetr ด้วย rank ของ output แทนชื่อ ซึ่งเชื่อถือได้กว่า — ดู qa MAJOR)
async function probeModel() {
  const candidates = [640, 384, 512, 560, 1280, 960, 576, 448, 336, 728];
  for (const s of candidates) {
    try {
      const t = new ort.Tensor("float32", new Float32Array(3 * s * s), [1, 3, s, s]);
      const out = await session.run({ [inputName]: t });
      return { size: s, out };
    } catch { /* ขนาดไม่ตรง — ลองอันถัดไป */ }
  }
  return null;
}

// รับทีละข้อความ: onmessage เป็น async — ถ้าไม่ต่อคิว ข้อความ detect หลายอันที่มาติดกัน (เช่น ตรวจแบบแบ่ง tile
// ส่งมาทีเดียว 8 ชิ้น) จะเรียก session.run ซ้อนกันบน session เดียว → ผลลัพธ์ปนกันระหว่างชิ้น/หาย
// โหลดโมเดลใหม่ = รุ่น (gen) ใหม่: detect ของโมเดลเก่าที่ยังค้างคิว (เช่น tile ที่เหลือ) ตอบกล่องว่างทันทีไม่ต้องรัน
// → โหลดโมเดลไม่ต้องรอ tile เก่าทั้งชุด (รอแค่ชิ้นที่กำลังรันอยู่)
let queue = Promise.resolve();
let gen = 0;
self.onmessage = (e) => {
  const m = e.data;
  if (m.type === "load") gen += 1;
  const myGen = gen;
  queue = queue.then(() => handleMessage(m, myGen), () => handleMessage(m, myGen));
};

async function handleMessage(m, myGen) {

  if (m.type === "load") {
    try {
      ort.env.wasm.wasmPaths = m.wasmBase;
      // หลาย thread ได้เมื่อมี SharedArrayBuffer (main.js เปิดสวิตช์ให้) — ใช้ราวจำนวนคอร์จริง (ครึ่งของ logical) ไม่เกิน 8
      // ไม่มี SAB (เช่น เปิดในเบราว์เซอร์ธรรมดา) → 1 thread เหมือนเดิม · ort อ่านค่านี้ครั้งเดียวตอน init wasm ครั้งแรก
      ort.env.wasm.numThreads = typeof SharedArrayBuffer !== "undefined"
        ? Math.min(8, Math.max(1, Math.floor((self.navigator?.hardwareConcurrency || 2) / 2)))
        : 1;
      session = await ort.InferenceSession.create(m.bytes, { executionProviders: ["wasm"] });
      inputName = session.inputNames[0];
      outputNames = session.outputNames.slice();
      outputName = outputNames[0];
      if (outputNames.length >= 2) {
        // 2 output = yoloseg หรือ rfdetr — แยกด้วย "รูปร่าง output จริง" (probe รันครั้งเดียว) ไม่ใช่ชื่อ
        // yoloseg มี prototypes เป็น tensor 4 มิติ [1,32,mh,mw] · rfdetr เป็น 3 มิติทั้งคู่ ([1,N,4]+[1,N,C])
        const probe = await probeModel();
        if (probe) {
          const has4d = outputNames.some((n) => probe.out[n] && probe.out[n].dims.length === 4);
          kind = has4d ? "yoloseg" : "rfdetr";
          inputSize = probe.size; // ขนาดที่โมเดลรับจริง (ที่รันผ่าน)
        } else {
          // probe ล้มทุกขนาด → เดาจากชื่อเป็น fallback
          kind = guessKind(outputNames);
          inputSize = kind === "rfdetr" ? 560 : typeof m.imgsz === "number" && m.imgsz > 0 ? m.imgsz : 640;
        }
      } else {
        kind = "yolo";
        // YOLO: session.inputMetadata undefined ใน onnxruntime-web → ใช้ imgsz ที่ main แกะจากไบต์ .onnx
        inputSize = typeof m.imgsz === "number" && m.imgsz > 0 ? m.imgsz : 640;
      }
      isSeg = kind === "yoloseg";
      self.postMessage({ type: "loaded", inputSize, seg: isSeg, kind, threads: ort.env.wasm.numThreads || 1 });
    } catch (err) {
      self.postMessage({ type: "loaded", error: err.message });
    }
    return;
  }

  if (m.type === "detect") {
    if (!session || myGen !== gen) {
      self.postMessage({ type: "result", reqId: m.reqId, boxes: [] });
      return;
    }
    try {
      const t = new ort.Tensor("float32", m.data, [1, 3, inputSize, inputSize]);
      const out = await session.run({ [inputName]: t });
      const boxes = decodeOutputs(kind, outputNames, out, m, inputSize);
      const transfer = boxes.filter((b) => b.mask).map((b) => b.mask.data.buffer);
      self.postMessage({ type: "result", reqId: m.reqId, boxes }, transfer);
    } catch (err) {
      self.postMessage({ type: "result", reqId: m.reqId, boxes: [], error: err.message });
    }
    return;
  }
}

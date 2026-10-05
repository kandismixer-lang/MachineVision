// Web Worker: รัน YOLO inference (onnxruntime-web) แยกจาก main thread — กัน UI ค้าง/แลค
// main ทำ preprocess (letterbox) แล้วส่ง tensor มา, worker รัน session + decode + NMS แล้วคืนกล่อง
import * as ort from "onnxruntime-web";

let session = null;
let inputName = "";
let outputName = "";
let outputNames = [];
let inputSize = 640;
let isSeg = false; // โมเดล segmentation (มี 2 output: detection head + mask prototypes)
let kind = "yolo"; // 'yolo' | 'yoloseg' | 'rfdetr'

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// ตรวจชนิดโมเดลจากชื่อ output — RF-DETR export ออก 2 output ชื่อแนว "dets"/"labels"
// (YOLO ใช้ "output0"/"output1") — heuristic นี้แยก rfdetr ออกจาก yolo/yolo-seg
function guessKind(names) {
  if (names.length >= 2) {
    const lc = names.map((n) => String(n).toLowerCase());
    const hasBox = lc.some((n) => /det|box|bbox|pred_box|point/.test(n));
    const hasCls = lc.some((n) => /label|logit|score|class|prob|cls/.test(n));
    if (hasBox && hasCls) return "rfdetr";
    return "yoloseg";
  }
  return "yolo";
}

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

// ถอดผล RF-DETR (DETR แบบ NMS-free):
//   box   [1, N, 4] = cx,cy,w,h normalized 0..1 (เทียบภาพเต็ม — resize ตรง ๆ ไม่ letterbox)
//   logit [1, N, C] = logit ต่อคลาส → sigmoid เป็นคะแนน (ไม่ใช่ softmax)
// คืนกล่องพิกัดภาพจริง (คูณ origW/origH) — ไม่ต้อง NMS
function decodeRfdetr(boxDims, box, logitDims, logit, origW, origH, conf) {
  // ผู้เรียกระบุ box/logit มาถูกแล้ว (ดู handler) — ไม่ต้องสลับเองอีก
  const bDims = boxDims, bData = box, lDims = logitDims, lData = logit;
  const N = bDims[1], C = lDims[lDims.length - 1];
  // rfdetr remap คลาสเป็น 0-indexed ตามลำดับคลาสจริง (ยืนยันจาก inference: Defect = channel 0)
  // → cls = bi ตรง ๆ map เข้ากับ state.labels ของแอปได้เลย
  const boxes = [];
  for (let i = 0; i < N; i++) {
    let best = 0, bi = 0;
    for (let c = 0; c < C; c++) {
      const s = sigmoid(lData[i * C + c]);
      if (s > best) { best = s; bi = c; }
    }
    if (best < conf) continue;
    const cx = bData[i * 4], cy = bData[i * 4 + 1], w = bData[i * 4 + 2], h = bData[i * 4 + 3];
    boxes.push({ x: (cx - w / 2) * origW, y: (cy - h / 2) * origH, w: w * origW, h: h * origH, score: best, cls: bi });
  }
  return boxes;
}

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

// ถอดผล YOLO-seg: 2 output
//   det   [1, 4+nc+32, N] = กล่อง(4) + คะแนนคลาส(nc) + สัมประสิทธิ์ mask(32) ต่อ anchor
//   proto [1, 32, mh, mw] = แม่แบบ mask 32 แผ่น (ความละเอียด ~ inputSize/4)
// mask ของแต่ละกล่อง = sigmoid(Σ coeff_k · proto_k) แล้ว threshold 0.5 · ครอปเฉพาะในกรอบ (ลดข้อมูลที่ส่งกลับ)
// คืน box พิกัดภาพจริง + mask ครอป (ความละเอียด proto) เพื่อให้ main วาดทับได้เลย
function decodeSeg(detDims, det, protoDims, proto, r, dx, dy, conf, iou) {
  const C = detDims[1], N = detDims[2];
  const NM = 32; // จำนวน mask prototype (คงที่ใน ultralytics)
  const nc = C - 4 - NM;
  if (nc <= 0) return [];
  const protoC = protoDims[1], mh = protoDims[2], mw = protoDims[3];
  const sx = mw / inputSize, sy = mh / inputSize; // input px → proto px
  const cand = [];
  for (let i = 0; i < N; i++) {
    let best = 0, bi = 0;
    for (let c = 0; c < nc; c++) {
      const s = det[(4 + c) * N + i];
      if (s > best) { best = s; bi = c; }
    }
    if (best < conf) continue;
    const cx = det[i], cy = det[N + i], w = det[2 * N + i], h = det[3 * N + i];
    const coeffs = new Float32Array(NM);
    for (let k = 0; k < NM; k++) coeffs[k] = det[(4 + nc + k) * N + i];
    cand.push({
      x: (cx - w / 2 - dx) / r, y: (cy - h / 2 - dy) / r, w: w / r, h: h / r, score: best, cls: bi,
      // เก็บกรอบใน input px ไว้คำนวณ proto ต่อ
      ix0: cx - w / 2, iy0: cy - h / 2, ix1: cx + w / 2, iy1: cy + h / 2, coeffs,
    });
  }
  const kept = nms(cand, iou);
  const protoStride = mh * mw;
  for (const b of kept) {
    // กรอบในพิกัด proto (clamp ในขอบ)
    let px0 = Math.max(0, Math.floor(b.ix0 * sx));
    let py0 = Math.max(0, Math.floor(b.iy0 * sy));
    let px1 = Math.min(mw, Math.ceil(b.ix1 * sx));
    let py1 = Math.min(mh, Math.ceil(b.iy1 * sy));
    const cw = Math.max(1, px1 - px0), ch = Math.max(1, py1 - py0);
    const mask = new Uint8Array(cw * ch);
    const co = b.coeffs;
    for (let my = 0; my < ch; my++) {
      for (let mx = 0; mx < cw; mx++) {
        const gp = (py0 + my) * mw + (px0 + mx);
        let acc = 0;
        for (let k = 0; k < NM; k++) acc += co[k] * proto[k * protoStride + gp];
        if (sigmoid(acc) > 0.5) mask[my * cw + mx] = 255;
      }
    }
    b.mask = { mw: cw, mh: ch, data: mask };
    delete b.coeffs; delete b.ix0; delete b.iy0; delete b.ix1; delete b.iy1;
  }
  return kept;
}

self.onmessage = async (e) => {
  const m = e.data;

  if (m.type === "load") {
    try {
      ort.env.wasm.wasmPaths = m.wasmBase;
      ort.env.wasm.numThreads = 1; // เลี่ยง SharedArrayBuffer
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
      self.postMessage({ type: "loaded", inputSize, seg: isSeg, kind });
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
      let boxes;
      const transfer = [];
      if (kind === "rfdetr") {
        // RF-DETR: 2 output — box (last-dim 4) + logit (last-dim = จำนวนคลาส)
        // ระบุ box ให้ชัด: เลือก output ที่ last-dim=4 · ถ้าทั้งคู่ =4 (คลาส=4 พอดี) ใช้ชื่อช่วย แล้ว fallback ลำดับ
        const outs = outputNames.map((n) => ({ n, o: out[n] }));
        const last = (o) => o.dims[o.dims.length - 1];
        const four = outs.filter((x) => last(x.o) === 4);
        let boxX;
        if (four.length === 1) boxX = four[0];
        else boxX = outs.find((x) => /det|box|bbox|point/i.test(x.n)) || outs[0];
        const logitX = outs.find((x) => x !== boxX);
        boxes = decodeRfdetr(boxX.o.dims, boxX.o.data, logitX.o.dims, logitX.o.data, m.origW || inputSize, m.origH || inputSize, m.conf);
      } else if (isSeg) {
        // แยก output: detection head = 3 มิติ (มี anchor), prototypes = 4 มิติ [1,32,mh,mw]
        let detO = null, protoO = null;
        for (const name of outputNames) {
          const o = out[name];
          if (o.dims.length === 4) protoO = o;
          else if (o.dims.length === 3) detO = o;
        }
        if (detO && protoO) {
          boxes = decodeSeg(detO.dims, detO.data, protoO.dims, protoO.data, m.r, m.dx, m.dy, m.conf, m.iou);
          for (const b of boxes) if (b.mask) transfer.push(b.mask.data.buffer);
        } else {
          boxes = [];
        }
      } else {
        const o = out[outputName];
        boxes = decode(o.dims, o.data, m.r, m.dx, m.dy, m.conf, m.iou);
      }
      self.postMessage({ type: "result", reqId: m.reqId, boxes }, transfer);
    } catch (err) {
      self.postMessage({ type: "result", reqId: m.reqId, boxes: [], error: err.message });
    }
    return;
  }
};

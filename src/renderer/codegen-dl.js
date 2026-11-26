// โค้ด Python ของกล่องผลลัพธ์ที่ต่อสายผ่านกล่อง Model Detect (2026-11-20)
// เดิม buildPythonCode ไม่รู้จักกล่อง DL (editor ไม่ใส่ลง pipeline แค่ตั้ง dl: true) → โค้ดที่ได้ไม่มีโมเดลเลย
// ไฟล์นี้สร้างสคริปต์ที่รันโมเดล .onnx ตัวเดียวกับในแอป ด้วยขั้นตอนเดียวกันทุกอย่าง:
//   letterbox / RF-DETR preprocess (dl-yolo.js) · ถอดผล + NMS (dl-decode.js) · แบ่ง tile + รวมกล่อง (dl-tiles.js)
//   · วาดกรอบ/ป้าย/mask (drawDetections ใน dl-yolo.js)
// แอปตรวจบน "ภาพต้นฉบับขนาดจริง" (renderDLDisplays ส่ง src เต็ม, Box อื่นบนสาย DL ไม่ถูกรัน) → สคริปต์ทำเหมือนกัน
import { FILTERS } from "./filters/registry.js";
import { normalizeSource5S, runSection5S } from "./codegen.js";
import { normCentroid, normCount, counterName } from "./dl-post.js";

const KINDS = new Set(["yolo", "yoloseg", "rfdetr"]);
const pyStr = (s) => JSON.stringify(String(s)); // สตริง JSON = สตริง Python ที่ถูกต้อง (escape \ " อักขระควบคุมแล้ว)
const pyNum = (v, d) => (Number.isFinite(Number(v)) ? String(Number(v)) : String(d));
const oneLine = (t) => String(t ?? "").replace(/[\r\n\x85\p{Zl}\p{Zp}]+/gu, " ").replace(/[\u0000-\u001f\u007f]/g, "");

function settingsLines(dl) {
  const lines = [];
  if (!dl.loaded) {
    lines.push('MODEL_PATH = ""   # << ยังไม่ได้โหลดโมเดลในโปรแกรม ใส่ path ไฟล์ .onnx');
  } else if (dl.modelPath) {
    lines.push(`MODEL_PATH = ${pyStr(dl.modelPath)}   # โมเดลที่โหลดในกล่อง Model Detect`);
  } else {
    lines.push(`MODEL_PATH = ${pyStr(dl.modelName || "model.onnx")}   # << ไม่รู้ตำแหน่งไฟล์เต็ม ใส่ path ของไฟล์ .onnx เอง`);
  }
  const kind = KINDS.has(dl.kind) ? pyStr(dl.kind) : "None";
  const size = Number.isFinite(Number(dl.inputSize)) && Number(dl.inputSize) > 0 ? Math.round(Number(dl.inputSize)) : 640;
  const labels = Array.isArray(dl.labels) ? dl.labels.map((s) => String(s ?? "")) : [];
  const tile = dl.tile || {};
  const conf = Number.isFinite(Number(dl.conf)) ? Number(dl.conf) : 0.25;
  const tileSize = Number.isFinite(Number(tile.size)) && Number(tile.size) > 0 ? Math.round(Number(tile.size)) : 640;
  const overlap = Number.isFinite(Number(tile.overlap)) ? Number(tile.overlap) / 100 : 0.2;
  const device = ["auto", "gpu", "cpu"].includes(dl.device) ? dl.device : "auto"; // อุปกรณ์ที่เลือกในกล่อง Model Detect
  lines.push(
    `MODEL_KIND = ${kind}   # yolo | yoloseg | rfdetr (ชนิดที่โปรแกรมตรวจได้ตอนโหลดโมเดล) · None = เดาจาก output ของโมเดล`,
    `INPUT_SIZE = ${size}   # ขนาด input ของโมเดล (ใช้เมื่ออ่านจากไฟล์โมเดลไม่ได้)`,
    `NAMES = [${labels.map(pyStr).join(", ")}]   # ชื่อคลาสตามลำดับ index ของโมเดล`,
    `CONF = ${pyNum(conf, 0.25)}   # ค่า "ความไว" ของกล่อง Model Detect ในโปรแกรม (ต่ำกว่านี้ไม่นับ)`,
    "IOU = 0.45   # NMS: กล่องคลาสเดียวกันที่ซ้อนกันเกินนี้ เก็บไว้แค่กล่องคะแนนสูงสุด",
    `TILE_ON = ${tile.on ? "True" : "False"}   # ตรวจแบบแบ่ง tile (เปิด/ปิดในกล่อง Model Detect)`,
    `TILE_SIZE = ${tileSize}`,
    `TILE_OVERLAP = ${pyNum(Math.min(0.9, Math.max(0, overlap)), 0.2)}   # สัดส่วนเหลื่อมกันของ tile (${pyNum(tile.overlap, 20)}%)`,
    `DEVICE = "${device}"   # "auto" = GPU ถ้ามี ไม่มีใช้ CPU · "gpu" = บังคับ GPU · "cpu" = บังคับ CPU (ตั้งจากกล่อง Model Detect)`,
    "# GPU ต้องลง onnxruntime รุ่น GPU แทนตัวธรรมดา: pip uninstall onnxruntime แล้ว pip install onnxruntime-directml (การ์ดทุกยี่ห้อบน Windows)",
    "#   หรือ onnxruntime-gpu (NVIDIA + CUDA) · cv2.dnn ใช้ CPU เสมอ",
  );
  return lines;
}

// ส่วนรันโมเดล — ฟังก์ชันตรงกับ dl-yolo.js / dl-decode.js / dl-tiles.js ทีละบรรทัด (ชื่อแบบ snake_case)
const CORE = String.raw`
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
BOX_COLOR = (255, 224, 0)    # #00e0ff (BGR) สีกรอบเดียวกับในโปรแกรม
TEXT_COLOR = (24, 16, 0)     # #001018
MASK_COLORS = [(59, 130, 246), (239, 68, 68), (34, 197, 94), (234, 179, 8), (168, 85, 247),
               (6, 182, 212), (249, 115, 22), (236, 72, 153), (132, 204, 22), (20, 184, 166)]  # RGB


def js_round(x):
    # Math.round ของ JavaScript (.5 ปัดขึ้นเสมอ) — round() ของ Python ปัดเข้าเลขคู่ ผลจะต่างจากโปรแกรม
    return int(math.floor(x + 0.5))


_MODELS = {}


def load_model(path=None):
    # โหลดโมเดลครั้งเดียวแล้วเก็บไว้ (ไม่โหลดใหม่ทุกเฟรม)
    path = MODEL_PATH if path is None else path
    if path in _MODELS:
        return _MODELS[path]
    if not path:
        raise FileNotFoundError("ยังไม่ได้ระบุไฟล์โมเดล — ใส่ path ของไฟล์ .onnx ที่ MODEL_PATH ด้านบน")
    if not os.path.isfile(path):
        raise FileNotFoundError("ไม่เจอไฟล์โมเดล: " + path)
    data = np.fromfile(path, dtype=np.uint8)   # อ่านเป็น bytes เอง → path ภาษาไทยก็โหลดได้
    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
    if DEVICE not in ("auto", "gpu", "cpu"):
        raise ValueError('DEVICE ต้องเป็น "auto", "gpu" หรือ "cpu"')
    if ort is not None:
        avail = ort.get_available_providers()
        gpu = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in avail]
        if DEVICE == "gpu" and not gpu:
            raise RuntimeError("ไม่มี GPU ให้ใช้ — ลง onnxruntime-directml หรือ onnxruntime-gpu แทน onnxruntime ก่อน")
        providers = (gpu[:1] if DEVICE != "cpu" else []) + ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        if "DmlExecutionProvider" in providers:
            so.enable_mem_pattern = False   # DirectML ต้องปิดสองค่านี้
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(data.tobytes(), so, providers=providers)
        inp = sess.get_inputs()[0]
        shape = inp.shape
        size = shape[2] if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0 else INPUT_SIZE
        names = [o.name for o in sess.get_outputs()]

        def run(blob):
            return dict(zip(names, sess.run(names, {inp.name: blob})))
        engine = "onnxruntime"
        used = sess.get_providers()[0]
        print("model runs on:", "GPU (" + used + ")" if used != "CPUExecutionProvider" else "CPU")
    else:
        if DEVICE == "gpu":
            raise RuntimeError("ไม่มี GPU ให้ใช้ — cv2.dnn รันบน CPU เท่านั้น ลง onnxruntime-directml หรือ onnxruntime-gpu ก่อน")
        print("onnxruntime not installed - using cv2.dnn on CPU (pip install onnxruntime for YOLO26 / RF-DETR / seg models)")
        net = cv2.dnn.readNetFromONNX(data)
        names = list(net.getUnconnectedOutLayersNames())
        size = INPUT_SIZE

        def run(blob):
            net.setInput(blob)
            outs = net.forward(names)
            return dict(zip(names, outs))
        engine = "cv2.dnn"
    model = {"run": run, "size": int(size), "names": names, "engine": engine}
    _MODELS[path] = model
    return model


def letterbox(img, size):
    # ย่อคงอัตราส่วนให้พอดี size × size วางกลาง เติมสีเทา 114 → RGB / 255, CHW (เหมือน letterbox ใน dl-yolo.js)
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = max(1, js_round(w * r)), max(1, js_round(h * r))
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))[None], r, dx, dy


def rfdetr_preprocess(img, size):
    # RF-DETR: resize ตรง ๆ เป็น size × size (ไม่ letterbox) + normalize แบบ ImageNet
    rgb = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)[None]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def best_class(scores):
    # scores = [คลาส, จำนวน] → (คะแนนสูงสุด, index) ต่อคอลัมน์ · เท่ากันเอาคลาสแรก · ไม่มีค่า > 0 → (0, 0) เหมือน JS
    m = scores.max(axis=0)
    cls = np.where(m > 0, scores.argmax(axis=0), 0)
    return np.maximum(m, 0.0), cls


def decode_raw_head(out, r, dx, dy, conf):
    # YOLO แบบ raw head [1, 4+nc, N]: cx, cy, w, h (px บนภาพ letterbox) + คะแนนต่อคลาส
    d = np.asarray(out, np.float64)[0]
    if d.shape[0] <= 4:
        return []
    best, cls = best_class(d[4:])
    boxes = []
    for i in np.nonzero(best >= conf)[0]:
        cx, cy, w, h = d[0, i], d[1, i], d[2, i], d[3, i]
        boxes.append({"x": (cx - w / 2 - dx) / r, "y": (cy - h / 2 - dy) / r, "w": w / r, "h": h / r,
                      "score": float(best[i]), "cls": int(cls[i])})
    return boxes


def iou_of(a, b):
    x1, y1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x2, y2 = min(a["x"] + a["w"], b["x"] + b["w"]), min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    uni = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / uni if uni > 0 else 0.0


def nms(boxes, iou_thresh):
    # เรียงคะแนนมาก → น้อย · ตัดเฉพาะกล่องคลาสเดียวกันที่ IoU ไม่ต่ำกว่าเกณฑ์
    kept = []
    for b in sorted(boxes, key=lambda b: -b["score"]):
        if all(k["cls"] != b["cls"] or iou_of(k, b) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def decode_nms_baked(out, r, dx, dy, conf):
    # end-to-end / NMS ในโมเดล (เช่น YOLO26): [1, n, 6] = x1, y1, x2, y2, score, class — ไม่ต้อง NMS ซ้ำ
    boxes = []
    for x1, y1, x2, y2, score, c in np.asarray(out, np.float64)[0]:
        if score < conf:
            continue
        boxes.append({"x": (x1 - dx) / r, "y": (y1 - dy) / r, "w": (x2 - x1) / r, "h": (y2 - y1) / r,
                      "score": float(score), "cls": int(c)})
    return boxes


def decode_rfdetr(box, logit, orig_w, orig_h, conf):
    # RF-DETR: box [1, N, 4] = cx, cy, w, h (0..1 เทียบภาพเต็ม) · logit [1, N, C] → sigmoid · ไม่ต้อง NMS
    bd = np.asarray(box, np.float64)[0]
    best, cls = best_class(sigmoid(np.asarray(logit, np.float64)[0]).T)
    boxes = []
    for i in np.nonzero(best >= conf)[0]:
        cx, cy, w, h = bd[i]
        boxes.append({"x": (cx - w / 2) * orig_w, "y": (cy - h / 2) * orig_h, "w": w * orig_w, "h": h * orig_h,
                      "score": float(best[i]), "cls": int(cls[i])})
    return boxes


def decode_seg(det, proto, r, dx, dy, conf, iou, size):
    # YOLO-seg: det [1, 4+nc+32, N] + proto [1, 32, mh, mw] → กล่อง + mask (ครอปเฉพาะในกรอบ ความละเอียด proto)
    d = np.asarray(det, np.float64)[0]
    p = np.asarray(proto, np.float64)[0]
    NM = 32
    nc = d.shape[0] - 4 - NM
    if nc <= 0:
        return []
    mh, mw = p.shape[1], p.shape[2]
    sx, sy = mw / size, mh / size
    best, cls = best_class(d[4:4 + nc])
    cand = []
    for i in np.nonzero(best >= conf)[0]:
        cx, cy, w, h = d[0, i], d[1, i], d[2, i], d[3, i]
        cand.append({"x": (cx - w / 2 - dx) / r, "y": (cy - h / 2 - dy) / r, "w": w / r, "h": h / r,
                     "score": float(best[i]), "cls": int(cls[i]),
                     "_in": (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), "_co": d[4 + nc:4 + nc + NM, i]})
    kept = nms(cand, iou)
    for b in kept:
        ix0, iy0, ix1, iy1 = b.pop("_in")
        co = b.pop("_co")
        px0, py0 = max(0, math.floor(ix0 * sx)), max(0, math.floor(iy0 * sy))
        px1, py1 = min(mw, math.ceil(ix1 * sx)), min(mh, math.ceil(iy1 * sy))
        cw, ch = max(1, px1 - px0), max(1, py1 - py0)
        mask = np.zeros((ch, cw), np.uint8)
        vh, vw = max(0, min(ch, mh - py0)), max(0, min(cw, mw - px0))   # ส่วนที่อยู่ในแผ่น proto จริง
        if vh and vw:
            acc = np.tensordot(co, p[:, py0:py0 + vh, px0:px0 + vw], axes=1)
            mask[:vh, :vw] = np.where(sigmoid(acc) > 0.5, 255, 0).astype(np.uint8)
        b["mask"] = mask
    return kept


def kind_of(outs):
    # เดาชนิดโมเดลจาก output (กติกาเดียวกับโปรแกรม): 1 output = yolo · มีตัว 4 มิติ = yoloseg · นอกนั้น rfdetr
    if len(outs) < 2:
        return "yolo"
    return "yoloseg" if any(np.ndim(o) == 4 for o in outs.values()) else "rfdetr"


def decode_outputs(kind, outs, m, size):
    # outs = {ชื่อ output: array} · m = {orig_w, orig_h, r, dx, dy, conf, iou}
    names = list(outs.keys())
    if kind == "rfdetr":
        four = [n for n in names if np.shape(outs[n])[-1] == 4]
        if len(four) == 1:
            box_name = four[0]
        else:
            box_name = next((n for n in names if re.search(r"det|box|bbox|point", n, re.I)), names[0])
        logit_name = next(n for n in names if n != box_name)
        return decode_rfdetr(outs[box_name], outs[logit_name], m["orig_w"], m["orig_h"], m["conf"])
    if kind == "yoloseg":
        det = next((outs[n] for n in names if np.ndim(outs[n]) == 3), None)
        proto = next((outs[n] for n in names if np.ndim(outs[n]) == 4), None)
        if det is None or proto is None:
            return []
        return decode_seg(det, proto, m["r"], m["dx"], m["dy"], m["conf"], m["iou"], size)
    o = outs[names[0]]
    if np.ndim(o) == 3 and np.shape(o)[2] == 6:
        return decode_nms_baked(o, m["r"], m["dx"], m["dy"], m["conf"])
    return nms(decode_raw_head(o, m["r"], m["dx"], m["dy"], m["conf"]), m["iou"])


def label_of(c):
    return NAMES[c] if 0 <= c < len(NAMES) and NAMES[c] else str(c)


def detect(img):
    # ตรวจภาพ 1 ภาพ (BGR) → list ของกล่อง {x, y, w, h, score, cls, label} บนพิกัดภาพนี้
    model = load_model()
    size = model["size"]
    h, w = img.shape[:2]
    if MODEL_KIND == "rfdetr":
        outs = model["run"](rfdetr_preprocess(img, size))
        m = {"orig_w": w, "orig_h": h, "conf": CONF}
    else:
        blob, r, dx, dy = letterbox(img, size)
        outs = model["run"](blob)
        m = {"r": r, "dx": dx, "dy": dy, "conf": CONF, "iou": IOU}
    kind = MODEL_KIND or kind_of(outs)
    if kind == "rfdetr" and MODEL_KIND != "rfdetr":   # เดาได้ว่าเป็น RF-DETR → ต้อง preprocess แบบ RF-DETR ใหม่
        outs = model["run"](rfdetr_preprocess(img, size))
        m = {"orig_w": w, "orig_h": h, "conf": CONF}
    boxes = decode_outputs(kind, outs, m, size)
    for b in boxes:
        b["label"] = label_of(b["cls"])
    return boxes


def tile_positions(length, tile, overlap):
    # ตำแหน่งเริ่ม tile ตามแกนเดียว: ก้าวทีละ tile × (1 − overlap) ชิ้นสุดท้ายชิดขอบพอดี
    L, T = max(1, js_round(length)), max(1, js_round(tile))
    if L <= T:
        return [0]
    ov = min(0.9, max(0.0, float(overlap or 0)))
    step = max(1, js_round(T * (1 - ov)))
    out, p = [], 0
    while p + T < L:
        out.append(p)
        p += step
    if out[-1] != L - T:
        out.append(L - T)
    return out


def tile_grid(w, h, tile, overlap):
    tw, th = min(js_round(tile), js_round(w)), min(js_round(tile), js_round(h))
    return [(x, y, tw, th) for y in tile_positions(h, tile, overlap) for x in tile_positions(w, tile, overlap)]


def intersect_over_smaller(a, b):
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    small = min(a["w"] * a["h"], b["w"] * b["h"])
    return ix * iy / small if small > 0 else 0.0


def merge_tile_boxes(boxes, thr=0.5):
    # รวมกล่องจากทุก tile: กล่องคลาสเดียวกันที่ทับกัน (intersection / กล่องเล็ก ≥ thr) → ขยายให้ครอบทั้งคู่
    keep = []
    for b in sorted(boxes, key=lambda b: -b["score"]):
        k = next((q for q in keep if q["cls"] == b["cls"] and intersect_over_smaller(q, b) >= thr), None)
        if k is None:
            keep.append(dict(b))
            continue
        x2, y2 = max(k["x"] + k["w"], b["x"] + b["w"]), max(k["y"] + k["h"], b["y"] + b["h"])
        k["x"], k["y"] = min(k["x"], b["x"]), min(k["y"], b["y"])
        k["w"], k["h"] = x2 - k["x"], y2 - k["y"]
    return keep


def detect_tiled(img):
    # หั่นภาพเป็น tile เหลื่อมกัน ตรวจทีละชิ้น แล้วรวมกล่องกลับเป็นพิกัดภาพเต็ม
    h, w = img.shape[:2]
    rects = tile_grid(w, h, TILE_SIZE, TILE_OVERLAP)
    if len(rects) <= 1:
        return detect(img)
    found = []
    for x, y, tw, th in rects:
        for b in detect(img[y:y + th, x:x + tw]):
            b["x"] += x
            b["y"] += y
            found.append(b)
    return merge_tile_boxes(found, 0.5)


def draw_mask(img, b):
    mask = b.get("mask")
    if mask is None or mask.size == 0:
        return
    x0, y0 = js_round(b["x"]), js_round(b["y"])
    bw, bh = max(1, js_round(b["w"])), max(1, js_round(b["h"]))
    a = cv2.resize(mask, (bw, bh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0 * (120 / 255.0)
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + bw), min(H, y0 + bh)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    a = a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0, None]
    rr, gg, bb = MASK_COLORS[b["cls"] % len(MASK_COLORS)]
    roi = img[cy0:cy1, cx0:cx1].astype(np.float32)
    img[cy0:cy1, cx0:cx1] = (roi * (1 - a) + np.array([bb, gg, rr], np.float32) * a + 0.5).astype(np.uint8)


def draw_detections(img, boxes):
    # วาดเหมือนจอผลลัพธ์ในโปรแกรม: mask (ถ้ามี) → กรอบฟ้า + ป้าย "ชื่อ xx%"
    # (cv2.putText เขียนภาษาไทยไม่ได้ — ชื่อคลาสภาษาไทยจะขึ้นเป็น ???)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H, W = img.shape[:2]
    scale = max(1.0, W / 640.0)
    for b in boxes:
        draw_mask(img, b)
    lw = max(1, js_round(2 * scale))
    fs, ft = 0.5 * scale, max(1, js_round(scale))
    th = 20 * scale
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cv2.rectangle(img, (js_round(x), js_round(y)), (js_round(x + w), js_round(y + h)), BOX_COLOR, lw)
        text = "%s %d%%" % (b.get("label", b["cls"]), int(b["score"] * 100))
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)[0][0] + 6 * scale
        lx = max(0.0, min(x, W - tw))
        ly = y - th if y - th >= 0 else y + h
        cv2.rectangle(img, (js_round(lx), js_round(ly)), (js_round(lx + tw), js_round(ly + th)), BOX_COLOR, -1)
        cv2.putText(img, text, (js_round(lx + 3 * scale), js_round(ly + th - 5 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, TEXT_COLOR, ft, cv2.LINE_AA)
    return img


`;

const PROCESS_PLAIN = String.raw`
def process_image(img):
    # ตรวจด้วยโมเดลบนภาพนี้ แล้ววาดกรอบ + ชื่อคลาส เหมือนจอผลลัพธ์ในโปรแกรม
    boxes = detect_tiled(img) if TILE_ON else detect(img)
    return draw_detections(img.copy(), boxes)
`;

const PROCESS_TAIL = String.raw`
def process(path):
    img = cv2.imread(path)  # อ่านภาพ (BGR)
    if img is None:
        raise FileNotFoundError(path)
    return process_image(img)
`;

// กล่อง 📍 Centroid + 🔢 Object Count — ตรงกับ dl-post.js ทีละขั้น
const POST_CORE = String.raw`
def otsu_threshold(hist, total):
    # เกณฑ์ Otsu (ค่าที่ > เกณฑ์ = กลุ่มสว่าง) · เท่ากันเอาเกณฑ์ตัวแรก — เหมือนในโปรแกรม
    sum_all = float(np.dot(np.arange(256), hist))
    w_b, sum_b, best, thr = 0.0, 0.0, -1.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, thr = between, t
    return thr


def shape_centroid(roi):
    # จุดศูนย์ถ่วงของชิ้นงานในกรอบ (พิกัดในกรอบ): gray → Otsu → กลุ่มที่แตะขอบกรอบน้อยกว่า = ชิ้นงาน → moments
    h, w = roi.shape[:2]
    if w < 2 or h < 2:
        return None
    b, g, r = (roi[:, :, i].astype(np.int64) for i in range(3))
    gray = (299 * r + 587 * g + 114 * b + 500) // 1000
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    t = otsu_threshold(hist, float(w * h))
    bright = gray > t
    border = np.concatenate([bright[0, :], bright[-1, :], bright[1:-1, 0], bright[1:-1, -1]])
    fg = bright if border.sum() * 2 <= border.size else ~bright
    m00 = int(fg.sum())
    if m00 == 0 or m00 == w * h:
        return None
    ys, xs = np.nonzero(fg)
    return float(xs.mean()) + 0.5, float(ys.mean()) + 0.5


def mask_centroid_frac(mask):
    # mask ของโมเดล seg (ครอปในกรอบ) → จุดศูนย์ถ่วงเป็นสัดส่วน 0..1 ของกรอบ
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.nonzero(mask >= 128)
    if xs.size == 0:
        return None
    mh, mw = mask.shape[:2]
    return float((xs + 0.5).mean()) / mw, float((ys + 0.5).mean()) / mh


def box_centers(boxes):
    return [{"x": float(b["x"] + b["w"] / 2), "y": float(b["y"] + b["h"] / 2), "cls": b["cls"], "label": b.get("label", str(b["cls"]))}
            for b in boxes]


def compute_centroids(img, boxes):
    # จุดของแต่ละชิ้น: "shape" = mask (ถ้ามี) หรือ Otsu ในกรอบ · "box" หรือหาไม่ได้ = กึ่งกลางกรอบ
    H, W = img.shape[:2]
    pts = box_centers(boxes)
    if CENTROID_METHOD != "shape":
        return pts
    for p, b in zip(pts, boxes):
        m = mask_centroid_frac(b.get("mask"))
        if m is not None:
            p["x"], p["y"] = b["x"] + m[0] * b["w"], b["y"] + m[1] * b["h"]
            continue
        x0, y0 = min(W, max(0, math.floor(b["x"]))), min(H, max(0, math.floor(b["y"])))
        x1, y1 = min(W, max(0, math.ceil(b["x"] + b["w"]))), min(H, max(0, math.ceil(b["y"] + b["h"])))
        c = shape_centroid(img[y0:y1, x0:x1]) if x1 - x0 >= 2 and y1 - y0 >= 2 else None
        if c is not None:
            p["x"], p["y"] = x0 + c[0], y0 + c[1]
    return pts


FRAME_DT = 1.0 / 30   # ช่วงเวลาอ้างอิงของ max_dist (1 เฟรมที่ 30 fps) · สคริปต์นี้ตรวจทุกเฟรม = ห่างกันเฟรมละ FRAME_DT
MAX_GATE_SCALE = 4    # ตรวจห่างมาก → ขยายระยะจับคู่ได้ไม่เกินกี่เท่า


class Tracker:
    # จับคู่จุดของเฟรมนี้กับตำแหน่งที่ทำนายของชิ้นเดิม (ล่าสุด + ความเร็ว × เวลา) คู่ที่ใกล้สุดก่อน (ไม่เกินระยะ)
    # ชิ้นที่หายเกิน max_missed ครั้ง = ลืม
    def __init__(self):
        self.tracks, self.next_id = [], 1

    def update(self, pts, max_dist, max_missed, dt=FRAME_DT):
        step = dt if dt and dt > 0 else 0.0
        pairs = []
        for ti, t in enumerate(self.tracks):
            since = t["since"] + step
            ex, ey = t["x"] + t["vx"] * since, t["y"] + t["vy"] * since
            gate = max_dist * min(MAX_GATE_SCALE, max(1.0, since / FRAME_DT))
            for pi, p in enumerate(pts):
                d = math.hypot(p["x"] - ex, p["y"] - ey)
                if d <= gate:
                    pairs.append((d, ti, pi))
        pairs.sort()
        used_t, used_p = set(), set()
        for _, ti, pi in pairs:
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            t, p = self.tracks[ti], pts[pi]
            since = t["since"] + step
            if since > 0:   # ความเร็ว (px/วินาที) เฉลี่ยกับค่าเดิมครึ่งต่อครึ่ง
                vx, vy = (p["x"] - t["x"]) / since, (p["y"] - t["y"]) / since
                if t["hits"] > 1:
                    t["vx"], t["vy"] = (t["vx"] + vx) / 2, (t["vy"] + vy) / 2
                else:
                    t["vx"], t["vy"] = vx, vy
            t.update(x=p["x"], y=p["y"], cls=p["cls"], label=p["label"], missed=0, since=0.0, hits=t["hits"] + 1)
            t["votes"][p["label"]] = t["votes"].get(p["label"], 0) + 1
        kept = []
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t["missed"] += 1
                t["since"] += step
            if t["missed"] <= max_missed:
                kept.append(t)
        for pi, p in enumerate(pts):
            if pi not in used_p:
                kept.append({"id": self.next_id, "x": p["x"], "y": p["y"], "vx": 0.0, "vy": 0.0, "since": 0.0,
                             "cls": p["cls"], "label": p["label"], "missed": 0, "hits": 1, "votes": {p["label"]: 1}})
                self.next_id += 1
        self.tracks = kept
        return kept


def track_label(t):
    # คลาสของชิ้นงาน = คลาสที่โมเดลทายบ่อยสุดตลอดช่วงที่ติดตาม (ทายสลับบางเฟรม → ไม่นับผิดคลาส)
    # เท่ากัน → คลาสของเฟรมล่าสุด ถ้าไม่อยู่ในกลุ่มที่เท่ากัน → ชื่อที่เรียงก่อน
    cur = t["label"]
    votes = t.get("votes", {})
    best, n = cur, votes.get(cur, 0)
    for k, c in votes.items():
        if c > n or (c == n and best != cur and k < best):
            best, n = k, c
    return best


def side_of(ax, ay, bx, by, x, y):
    v = float((bx - ax) * (y - ay) - (by - ay) * (x - ax))
    return (v > 0) - (v < 0)


def crosses_segment(ax, ay, bx, by, x0, y0, x1, y1):
    s0 = side_of(x0, y0, x1, y1, ax, ay)
    s1 = side_of(x0, y0, x1, y1, bx, by)
    return s0 == 0 or s1 == 0 or s0 != s1


class Counter:
    # line: นับเมื่อชิ้นงานข้ามเส้น A→B (ชิ้นละครั้ง) · จากฝั่ง + ไปฝั่ง − = IN (ทิศลูกศรบนภาพ)
    # zone: NOW = ชิ้นในกรอบตอนนี้ · TOTAL = ชิ้นที่เคยเข้ากรอบ (ชิ้นละครั้ง)
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.tracker = Tracker()
        self.n_in = self.n_out = self.total = self.now = 0
        self.by_class, self.counted, self.ref, self.events = {}, set(), {}, []

    def _slot(self, label):
        return self.by_class.setdefault(label, {"in": 0, "out": 0, "total": 0, "now": 0})

    def _bump(self, t, key):
        if key == "in":
            self.n_in += 1
        else:
            self.n_out += 1
        self.total += 1
        label = track_label(t)
        b = self._slot(label)
        b[key] += 1
        b["total"] += 1
        self.events.append({"counter": self.cfg["name"], "id": t["id"], "label": label, "kind": key,
                            "x": float(t["x"]), "y": float(t["y"]), "total": self.total, "class_total": b["total"]})

    def update(self, pts, W, H, still=False, dt=FRAME_DT):
        c = self.cfg
        if still:
            self.reset()
        self.events = []
        tracks = self.tracker.update(pts, c["max_dist"] / 100.0 * math.hypot(W, H), c["max_missed"], dt)

        def want(t):
            return not c["cls"] or track_label(t) == c["cls"]
        if c["mode"] == "zone":
            zx, zy = c["zx"] / 100.0 * W, c["zy"] / 100.0 * H
            zx2, zy2 = zx + c["zw"] / 100.0 * W, zy + c["zh"] / 100.0 * H
            now = 0
            for b in self.by_class.values():
                b["now"] = 0
            for t in tracks:
                if t["missed"] != 0 or not want(t):
                    continue
                if t["x"] < zx or t["x"] > zx2 or t["y"] < zy or t["y"] > zy2:
                    continue
                now += 1
                if t["id"] not in self.counted:
                    self.counted.add(t["id"])
                    self._bump(t, "in")
                self._slot(track_label(t))["now"] += 1
            self.now = now
        else:
            ax, ay = c["x1"] / 100.0 * W, c["y1"] / 100.0 * H
            bx, by = c["x2"] / 100.0 * W, c["y2"] / 100.0 * H
            alive = set()
            for t in tracks:
                alive.add(t["id"])
                if t["missed"] != 0:
                    continue
                s = side_of(ax, ay, bx, by, t["x"], t["y"])
                if s == 0:
                    continue
                r = self.ref.get(t["id"])
                self.ref[t["id"]] = (s, t["x"], t["y"])
                if r is None or r[0] == s or t["id"] in self.counted or not want(t):
                    continue
                if not crosses_segment(ax, ay, bx, by, r[1], r[2], t["x"], t["y"]):
                    continue
                key = "in" if r[0] > 0 else "out"
                if c["dir"] != "both" and c["dir"] != key:
                    continue
                self.counted.add(t["id"])
                self._bump(t, key)
            self.ref = {k: v for k, v in self.ref.items() if k in alive}
            self.now = 0

    def text(self):
        c, name = self.cfg, self.cfg["name"]
        if c["mode"] == "zone":
            return "%s: NOW %d  TOTAL %d" % (name, self.now, self.total)
        if c["dir"] == "in":
            return "%s: IN %d" % (name, self.n_in)
        if c["dir"] == "out":
            return "%s: OUT %d" % (name, self.n_out)
        return "%s: %d  (IN %d / OUT %d)" % (name, self.total, self.n_in, self.n_out)

    def class_rows(self):
        # ยอดแยกคลาส (เรียงชื่อคลาส) — แถวละคลาส
        c, rows = self.cfg, []
        for k in sorted(self.by_class):
            b = self.by_class[k]
            if c["mode"] == "zone":
                rows.append("%s: NOW %d  TOTAL %d" % (k, b["now"], b["total"]))
            elif c["dir"] == "in":
                rows.append("%s: IN %d" % (k, b["in"]))
            elif c["dir"] == "out":
                rows.append("%s: OUT %d" % (k, b["out"]))
            else:
                rows.append("%s: %d  (IN %d / OUT %d)" % (k, b["total"], b["in"], b["out"]))
        return rows


COUNTER_OBJS = [Counter(c) for c in COUNTERS]
LINE_BGR, ZONE_BGR, CENTROID_BGR = (10, 214, 255), (153, 211, 52), (92, 59, 255)


def draw_centroids(img, pts, scale):
    for p in pts:
        x, y = js_round(p["x"]), js_round(p["y"])
        cv2.circle(img, (x, y), max(2, js_round(4 * scale)), CENTROID_BGR, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), max(2, js_round(4 * scale)), (255, 255, 255), 1, cv2.LINE_AA)
        if SHOW_COORDS:
            fs = 0.4 * scale * CENTROID_FONT
            text = "(%d, %d)" % (x, y)
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
            tx = p["x"] + 6 * scale
            if tx + tw > img.shape[1]:   # ชิดขอบขวา → เขียนทางซ้ายของจุด
                tx = p["x"] - 6 * scale - tw
            org = (js_round(tx), js_round(p["y"] - 6 * scale))
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)


def draw_counters(img, scale):
    H, W = img.shape[:2]
    for k in COUNTER_OBJS:
        c = k.cfg
        flash = bool(k.events)
        if c["mode"] == "zone":
            p0 = (js_round(c["zx"] / 100.0 * W), js_round(c["zy"] / 100.0 * H))
            p1 = (js_round((c["zx"] + c["zw"]) / 100.0 * W), js_round((c["zy"] + c["zh"]) / 100.0 * H))
            over = img.copy()
            cv2.rectangle(over, p0, p1, ZONE_BGR, -1)
            cv2.addWeighted(over, 0.35 if flash else 0.15, img, 0.65 if flash else 0.85, 0, img)
            cv2.rectangle(img, p0, p1, ZONE_BGR, max(1, js_round(2 * scale)))
        else:
            a = (js_round(c["x1"] / 100.0 * W), js_round(c["y1"] / 100.0 * H))
            b = (js_round(c["x2"] / 100.0 * W), js_round(c["y2"] / 100.0 * H))
            cv2.line(img, a, b, (136, 255, 34) if flash else LINE_BGR, max(1, js_round((4 if flash else 2.5) * scale)), cv2.LINE_AA)
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            if L > 1:   # ลูกศรทิศ IN (ตั้งฉากกับเส้น ชี้ไปฝั่ง −)
                nx, ny = (b[1] - a[1]) / L, -(b[0] - a[0]) / L
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                tail = (js_round(mx - nx * 8 * scale), js_round(my - ny * 8 * scale))
                head = (js_round(mx + nx * 16 * scale), js_round(my + ny * 16 * scale))
                cv2.arrowedLine(img, tail, head, LINE_BGR, max(1, js_round(2 * scale)), cv2.LINE_AA, tipLength=0.35)
    if COUNTER_OBJS:   # เลข id ของชิ้นที่ติดตามอยู่ (ของกล่องนับกล่องแรก)
        ts = scale * COUNTER_OBJS[0].cfg.get("font", 1.0)
        for t in COUNTER_OBJS[0].tracker.tracks:
            if t["missed"] == 0:
                org = (js_round(t["x"] + 6 * scale), js_round(t["y"] + 14 * ts))
                cv2.putText(img, "#%d" % t["id"], org, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ts, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, "#%d" % t["id"], org, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ts, LINE_BGR, 1, cv2.LINE_AA)
    y = js_round(6 * scale)
    for k in COUNTER_OBJS:   # ป้ายยอดนับมุมซ้ายบน (ขนาดตาม "font" ของแต่ละกล่อง) + แถวแยกคลาส (split_class)
        ts = scale * k.cfg.get("font", 1.0)
        row = max(6, js_round(20 * ts))
        fs = 0.45 * ts
        color = ZONE_BGR if k.cfg["mode"] == "zone" else LINE_BGR
        rows = [(k.text(), 0)]
        if k.cfg.get("split_class"):
            rows += [(r, js_round(12 * ts)) for r in k.class_rows()]
        for text, indent in rows:
            x0 = js_round(6 * scale) + indent
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] + js_round(12 * ts)
            over = img.copy()
            cv2.rectangle(over, (x0, y), (x0 + tw, y + row), (0, 0, 0), -1)
            cv2.addWeighted(over, 0.6, img, 0.4, 0, img)
            cv2.putText(img, text, (x0 + js_round(6 * ts), y + row - js_round(6 * ts)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255) if indent else color, 1, cv2.LINE_AA)
            y += row + js_round((2 if indent else 4) * ts)


_FRAME = [0]


def on_count(event):
    # ★ เรียกทุกครั้งที่นับได้ 1 ชิ้น — ใส่โค้ดสั่งงานต่อตรงนี้ (ส่ง Serial ไป Dobot / Modbus ไป PLC / เขียน CSV ...)
    # event = {"frame": 310, "counter": "Line 1", "id": 12, "label": "Triangle", "kind": "in",
    #          "x": 412.5, "y": 230.1, "total": 10, "class_total": 4}
    #   label = คลาสของชิ้น (ที่โมเดลทายบ่อยสุดตลอดช่วงที่ติดตาม) · kind = "in" / "out" (โซน = "in" เสมอ)
    #   x, y = พิกัดจุดของชิ้นตอนนับ (พิกเซลบนภาพ) · total / class_total = ยอดรวม / ยอดของคลาสนี้ หลังนับชิ้นนี้
    # ยอดทั้งหมดอ่านได้ตลอดจาก COUNTER_OBJS[i].total / .n_in / .n_out / .now / .by_class["Triangle"]["total"]
    # ตัวอย่าง:
    #   if event["label"] == "Triangle":
    #       robot.move_to(event["x"], event["y"])
    pass


def process_image(img):
    # ตรวจด้วยโมเดล → วาดกรอบ → หา centroid → อัปเดตตัวนับ → วาดเส้น/โซน + ยอดนับ (เหมือนจอผลลัพธ์ในโปรแกรม)
    _FRAME[0] += 1
    boxes = detect_tiled(img) if TILE_ON else detect(img)
    out = draw_detections(img.copy(), boxes)
    H, W = img.shape[:2]
    scale = max(1.0, W / 640.0)
    pts = compute_centroids(img, boxes) if CENTROID_ON else box_centers(boxes)
    for k in COUNTER_OBJS:
        k.update(pts, W, H, STILL)
        for ev in k.events:
            ev["frame"] = _FRAME[0]
            print("[frame %d] %s: +1 %s (#%d, %s) -> %s" % (ev["frame"], ev["counter"], ev["label"], ev["id"], ev["kind"].upper(), k.text()))
            on_count(ev)
    if CENTROID_ON:
        draw_centroids(out, pts, scale)
        if STILL or not COUNTER_OBJS:
            for p in pts:
                print("centroid %s: (%.1f, %.1f)" % (p["label"], p["x"], p["y"]))
    draw_counters(out, scale)
    return out


def print_summary():
    # สรุปยอดตอนจบโปรแกรม (ปิดหน้าต่าง / วิดีโอจบ)
    for k in COUNTER_OBJS:
        print("SUMMARY %s | by class: %s" % (k.text(), " · ".join(k.class_rows()) or "-"))


atexit.register(print_summary)
`;

// post = [{ type: "centroid" | "count", params }] ตามลำดับบนสาย (ต้นน้ำ → ปลายน้ำ)
function postSettingsLines(post, still) {
  const cen = post.find((s) => s.type === "centroid");
  const cp = normCentroid(cen?.params);
  const counters = post.filter((s) => s.type === "count");
  const idx = { line: 0, zone: 0 };
  const lines = [
    "",
    "# ── ตั้งค่า 📍 Centroid + 🔢 Object Count (ดึงจากกล่องในโปรแกรม) ──",
    `CENTROID_ON = ${cen ? "True" : "False"}   # มีกล่อง Centroid บนสาย (ไม่มี = นับด้วยจุดกึ่งกลางกรอบ)`,
    `CENTROID_METHOD = ${pyStr(cp.method)}   # "shape" = จุดศูนย์ถ่วงของรูปทรงชิ้นงานในกรอบ · "box" = กึ่งกลางกรอบ`,
    `SHOW_COORDS = ${cp.showCoords ? "True" : "False"}   # เขียนพิกัด (x, y) ข้างจุด`,
    `CENTROID_FONT = ${cp.fontPct / 100}   # ขนาดตัวหนังสือพิกัด (1 = ปกติ · 0.5 = ครึ่งหนึ่ง)`,
    `STILL = ${still ? "True" : "False"}   # True = ภาพนิ่ง (นับใหม่ทุกภาพ ไม่ติดตามข้ามภาพ) · วิดีโอ/กล้อง = False`,
    "# พิกัดเส้น/โซนเป็น % ของกว้าง/สูงภาพ · max_dist = ระยะจับคู่ชิ้นเดิมข้ามเฟรม (% ของเส้นทแยงภาพ)",
    "COUNTERS = [",
  ];
  for (const s of counters) {
    const p = normCount(s.params);
    const name = counterName(p, ++idx[p.mode]);
    lines.push(
      `    {"name": ${pyStr(name)}, "mode": ${pyStr(p.mode)},   # ${p.mode === "zone" ? "โซนนับ" : "เส้นนับ"}`,
      `     "x1": ${p.x1}, "y1": ${p.y1}, "x2": ${p.x2}, "y2": ${p.y2}, "dir": ${pyStr(p.dir)},   # เส้น A(x1,y1) → B(x2,y2) · dir = "both" | "in" | "out"`,
      `     "zx": ${p.zx}, "zy": ${p.zy}, "zw": ${p.zw}, "zh": ${p.zh},   # โซน: มุมซ้ายบน + กว้าง/สูง`,
      `     "cls": ${pyStr(p.cls)}, "max_dist": ${p.maxDist}, "max_missed": ${p.maxMissed},   # cls "" = ทุกคลาส`,
      `     "font": ${p.fontPct / 100}, "split_class": ${p.splitClass ? "True" : "False"}},   # font = ขนาดตัวหนังสือป้ายยอดนับ + เลข # (1 = ปกติ) · split_class = แสดงแถวยอดแยกคลาสบนภาพ`,
    );
  }
  lines.push("]");
  return lines;
}

export function buildDLPythonCode(pipeline, opts = {}) {
  const name = opts.displayName || "ผลลัพธ์";
  const dl = opts.dl || {};
  const src = normalizeSource5S(opts.source);
  const ignored = (pipeline || []).filter((st) => st && st.enabled !== false);
  const post = (opts.post || []).filter((s) => s && (s.type === "centroid" || s.type === "count"));

  const out = [
    "# -*- coding: utf-8 -*-",
    `# โค้ดนี้สร้างอัตโนมัติจากลำดับ Box ในกล่องผลลัพธ์: "${oneLine(name)}"`,
    "# กล่องนี้ต่อผ่าน Model Detect — รันโมเดล .onnx แบบเดียวกับในโปรแกรม (onnxruntime หรือ cv2.dnn)",
    "# ต้องมี: pip install opencv-python numpy onnxruntime   (ไม่มี onnxruntime จะใช้ cv2.dnn แทน)",
    "import cv2",
    "import numpy as np",
    "import sys",
    "import os",
    "import math",
    "import re",
    ...(post.length ? ["import atexit"] : []),
    "",
    "# บังคับให้ข้อความที่ print() ออกมาเป็น UTF-8 เสมอ — บนวินโดวส์ เวลาส่งผลไปเก็บในไฟล์",
    "# (เช่น python code.py > log.txt) ระบบจะใช้รหัสอักขระเดิมของเครื่อง ซึ่งเขียนภาษาไทยหรือ",
    "# สัญลักษณ์อย่าง px² / ∞ ไม่ได้ แล้วโปรแกรมจะหยุดกลางคันด้วย UnicodeEncodeError",
    "if hasattr(sys.stdout, \"reconfigure\"):",
    "    sys.stdout.reconfigure(encoding=\"utf-8\", errors=\"replace\")",
    "",
    "# ── ตั้งค่า (ดึงจากกล่อง Model Detect ตอนกดสร้างโค้ด) ──",
    ...settingsLines(dl),
    ...(post.length ? postSettingsLines(post, !src || src.kind === "image") : []),
  ];
  if (ignored.length) {
    const names = ignored.map((st) => FILTERS[st.id]?.name || st.id).join(", ");
    out.push(
      "",
      `# หมายเหตุ: ในโปรแกรม กล่อง Model Detect ตรวจบน "ภาพต้นฉบับ" — Box อื่นบนสายนี้ (${oneLine(names)}) ไม่มีผลกับจอนี้ จึงไม่ได้ใส่ในโค้ด`,
      'print("NOTE: filter boxes on this Model Detect line are not applied (same as the app)")',
    );
  }
  out.push("", CORE.trim(), "", "", (post.length ? POST_CORE : PROCESS_PLAIN).trim(), "", "", PROCESS_TAIL.trim(), "", "");
  if (src) {
    out.push("# กล่อง Model Detect ในโปรแกรมตรวจบนภาพขนาดจริง (ไม่ย่อ) จึงตั้ง MAX_SIDE = 0");
    out.push(runSection5S({ ...src, maxSide: 0 }, false));
  } else {
    out.push(
      "if __name__ == \"__main__\":",
      "    result = process(\"input.jpg\")   # << ใส่ path ภาพของคุณ",
      "    cv2.imwrite(\"output.png\", result)",
      "    cv2.imshow(\"result\", result)",
      "    cv2.waitKey(0)",
      "    cv2.destroyAllWindows()",
      "",
    );
  }
  return out.join("\n");
}

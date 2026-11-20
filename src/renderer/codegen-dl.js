// โค้ด Python ของกล่องผลลัพธ์ที่ต่อสายผ่านกล่อง Model Detect (2026-11-20)
// เดิม buildPythonCode ไม่รู้จักกล่อง DL (editor ไม่ใส่ลง pipeline แค่ตั้ง dl: true) → โค้ดที่ได้ไม่มีโมเดลเลย
// ไฟล์นี้สร้างสคริปต์ที่รันโมเดล .onnx ตัวเดียวกับในแอป ด้วยขั้นตอนเดียวกันทุกอย่าง:
//   letterbox / RF-DETR preprocess (dl-yolo.js) · ถอดผล + NMS (dl-decode.js) · แบ่ง tile + รวมกล่อง (dl-tiles.js)
//   · วาดกรอบ/ป้าย/mask (drawDetections ใน dl-yolo.js)
// แอปตรวจบน "ภาพต้นฉบับขนาดจริง" (renderDLDisplays ส่ง src เต็ม, Box อื่นบนสาย DL ไม่ถูกรัน) → สคริปต์ทำเหมือนกัน
import { FILTERS } from "./filters/registry.js";
import { normalizeSource5S, runSection5S } from "./codegen.js";

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
  lines.push(
    `MODEL_KIND = ${kind}   # yolo | yoloseg | rfdetr (ชนิดที่โปรแกรมตรวจได้ตอนโหลดโมเดล) · None = เดาจาก output ของโมเดล`,
    `INPUT_SIZE = ${size}   # ขนาด input ของโมเดล (ใช้เมื่ออ่านจากไฟล์โมเดลไม่ได้)`,
    `NAMES = [${labels.map(pyStr).join(", ")}]   # ชื่อคลาสตามลำดับ index ของโมเดล`,
    `CONF = ${pyNum(conf, 0.25)}   # ค่า "ความไว" ของกล่อง Model Detect ในโปรแกรม (ต่ำกว่านี้ไม่นับ)`,
    "IOU = 0.45   # NMS: กล่องคลาสเดียวกันที่ซ้อนกันเกินนี้ เก็บไว้แค่กล่องคะแนนสูงสุด",
    `TILE_ON = ${tile.on ? "True" : "False"}   # ตรวจแบบแบ่ง tile (เปิด/ปิดในกล่อง Model Detect)`,
    `TILE_SIZE = ${tileSize}`,
    `TILE_OVERLAP = ${pyNum(Math.min(0.9, Math.max(0, overlap)), 0.2)}   # สัดส่วนเหลื่อมกันของ tile (${pyNum(tile.overlap, 20)}%)`,
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
    if ort is not None:
        sess = ort.InferenceSession(data.tobytes(), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shape = inp.shape
        size = shape[2] if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0 else INPUT_SIZE
        names = [o.name for o in sess.get_outputs()]

        def run(blob):
            return dict(zip(names, sess.run(names, {inp.name: blob})))
        engine = "onnxruntime"
    else:
        print("onnxruntime not installed - using cv2.dnn (pip install onnxruntime for YOLO26 / RF-DETR / seg models)")
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


def process_image(img):
    # ตรวจด้วยโมเดลบนภาพนี้ แล้ววาดกรอบ + ชื่อคลาส เหมือนจอผลลัพธ์ในโปรแกรม
    boxes = detect_tiled(img) if TILE_ON else detect(img)
    return draw_detections(img.copy(), boxes)


def process(path):
    img = cv2.imread(path)  # อ่านภาพ (BGR)
    if img is None:
        raise FileNotFoundError(path)
    return process_image(img)
`;

export function buildDLPythonCode(pipeline, opts = {}) {
  const name = opts.displayName || "ผลลัพธ์";
  const dl = opts.dl || {};
  const src = normalizeSource5S(opts.source);
  const ignored = (pipeline || []).filter((st) => st && st.enabled !== false);

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
    "",
    "# บังคับให้ข้อความที่ print() ออกมาเป็น UTF-8 เสมอ — บนวินโดวส์ เวลาส่งผลไปเก็บในไฟล์",
    "# (เช่น python code.py > log.txt) ระบบจะใช้รหัสอักขระเดิมของเครื่อง ซึ่งเขียนภาษาไทยหรือ",
    "# สัญลักษณ์อย่าง px² / ∞ ไม่ได้ แล้วโปรแกรมจะหยุดกลางคันด้วย UnicodeEncodeError",
    "if hasattr(sys.stdout, \"reconfigure\"):",
    "    sys.stdout.reconfigure(encoding=\"utf-8\", errors=\"replace\")",
    "",
    "# ── ตั้งค่า (ดึงจากกล่อง Model Detect ตอนกดสร้างโค้ด) ──",
    ...settingsLines(dl),
  ];
  if (ignored.length) {
    const names = ignored.map((st) => FILTERS[st.id]?.name || st.id).join(", ");
    out.push(
      "",
      `# หมายเหตุ: ในโปรแกรม กล่อง Model Detect ตรวจบน "ภาพต้นฉบับ" — Box อื่นบนสายนี้ (${oneLine(names)}) ไม่มีผลกับจอนี้ จึงไม่ได้ใส่ในโค้ด`,
      'print("NOTE: filter boxes on this Model Detect line are not applied (same as the app)")',
    );
  }
  out.push("", CORE.trim(), "", "");
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

// คณิตศาสตร์/ตรรกะล้วนของ UI เฟส 3 (ไม่ import DOM/LiteGraph/registry) — แยกออกมาเพื่อเทสต์ได้
// ใช้โดย pixel-inspector.js (3.1) · app.js (3.2 / 3.5 / 3.6 / 3.7) · editor.js (3.4 / 3.5 / 3.8)

// =====================================================================
// 3.1 ซูม/เลื่อนในจอผลลัพธ์ (viewer tile) + แปลงพิกัดเมาส์ → พิกัดภาพ
// =====================================================================
// โมเดล: <canvas> ถูกยืดเต็มกล่อง viewport (object-fit: contain) แล้วใช้ CSS transform ซูม/เลื่อนทั้งกล่อง
//   view = { z, tx, ty }  · z = 1, tx = ty = 0 คือ "พอดีจอ" (fit)
//   พิกัดบน viewport = (tx, ty) + z * พิกัดในกล่อง canvas
//   geom = { vw, vh, iw, ih } = ขนาด viewport (CSS px) และขนาดภาพจริง (px)
export const PX_MAX_SCALE = 64; // ซูมสูงสุด: 1 พิกเซลภาพ = 64 พิกเซลจอ
export const PX_GRID_SCALE = 8; // ≥ 800% → วาดตารางพิกเซล + ปิด smoothing

export function fitRect(vw, vh, iw, ih) {
  if (!(vw > 0) || !(vh > 0) || !(iw > 0) || !(ih > 0)) return null;
  const f = Math.min(vw / iw, vh / ih);
  const w = iw * f, h = ih * f;
  return { f, x: (vw - w) / 2, y: (vh - h) / 2, w, h };
}

export const FIT_VIEW = Object.freeze({ z: 1, tx: 0, ty: 0 });

// ซูมสูงสุดที่อนุญาต (เทียบกับ fit) — ไม่ต่ำกว่า 1 เสมอ
export function maxZoom(geom) {
  const r = fitRect(geom.vw, geom.vh, geom.iw, geom.ih);
  return r ? Math.max(1, PX_MAX_SCALE / r.f) : 1;
}

// สเกลจริงบนจอ = พิกเซลจอต่อ 1 พิกเซลภาพ (800% = 8)
export function displayScale(view, geom) {
  const r = fitRect(geom.vw, geom.vh, geom.iw, geom.ih);
  return r ? r.f * view.z : 0;
}

// ขอบภาพ (left = t + z*fpos) ต้องอยู่ในช่วงที่ภาพไม่หลุดจอ: ภาพใหญ่กว่าจอ → left ∈ [vsize − size, 0] (ไม่เห็นพื้นว่างข้างใน)
// ภาพยังเล็กกว่าจอ → left ∈ [0, vsize − size] (อยู่ในจอทั้งภาพ) — ไม่บังคับกึ่งกลาง เพื่อให้ซูมรอบเคอร์เซอร์ตรึงจุดได้จริงตั้งแต่ก้าวแรก
function clampAxis(t, z, fpos, fsize, vsize) {
  const size = z * fsize;
  const lo = Math.min(0, vsize - size) - z * fpos;
  const hi = Math.max(0, vsize - size) - z * fpos;
  return Math.min(hi, Math.max(lo, t));
}

// บังคับ z ∈ [1, maxZoom] และไม่ให้ภาพถูกลากจนเห็นพื้นว่างข้างใน
export function clampView(view, geom) {
  const r = fitRect(geom.vw, geom.vh, geom.iw, geom.ih);
  if (!r) return { z: 1, tx: 0, ty: 0 };
  const z = Math.min(maxZoom(geom), Math.max(1, Number.isFinite(view.z) ? view.z : 1));
  if (z <= 1.0001) return { z: 1, tx: 0, ty: 0 }; // ซูมออกจนสุด = พอดีจอเป๊ะ (กลับกึ่งกลาง)
  return {
    z,
    tx: clampAxis(Number.isFinite(view.tx) ? view.tx : 0, z, r.x, r.w, geom.vw),
    ty: clampAxis(Number.isFinite(view.ty) ? view.ty : 0, z, r.y, r.h, geom.vh),
  };
}

// ซูมคูณ factor โดยตรึงจุด (cx, cy) บน viewport ไว้ใต้เคอร์เซอร์
export function zoomAt(view, factor, cx, cy, geom) {
  const z0 = view.z;
  const z1 = Math.min(maxZoom(geom), Math.max(1, z0 * factor));
  const k = z1 / z0;
  return clampView({ z: z1, tx: cx - (cx - view.tx) * k, ty: cy - (cy - view.ty) * k }, geom);
}

export function panBy(view, dx, dy, geom) {
  return clampView({ z: view.z, tx: view.tx + dx, ty: view.ty + dy }, geom);
}

// ตัวคูณซูมจากล้อเมาส์ (deltaY > 0 = เลื่อนลง = ซูมออก) — จำกัดแรงต่อ event กันแทร็กแพดกระโดด
export function wheelFactor(deltaY) {
  const d = Math.max(-300, Math.min(300, Number(deltaY) || 0));
  return Math.exp(-d * 0.002);
}

// พิกัดบน viewport (px, py = ระยะจากมุมซ้ายบนของ viewport) → พิกเซลภาพ (จำนวนเต็ม) · inside = อยู่ในภาพจริง
// out (ไม่บังคับ) = object ที่ใช้ซ้ำ — mousemove จึงไม่ต้องสร้าง object ใหม่ทุกครั้ง
export function screenToImage(view, px, py, geom, out) {
  const o = out || { x: 0, y: 0, inside: false };
  const r = fitRect(geom.vw, geom.vh, geom.iw, geom.ih);
  if (!r) { o.x = 0; o.y = 0; o.inside = false; return o; }
  const fx = ((px - view.tx) / view.z - r.x) / r.f;
  const fy = ((py - view.ty) / view.z - r.y) / r.f;
  o.inside = fx >= 0 && fy >= 0 && fx < geom.iw && fy < geom.ih;
  o.x = Math.min(geom.iw - 1, Math.max(0, Math.floor(fx)));
  o.y = Math.min(geom.ih - 1, Math.max(0, Math.floor(fy)));
  return o;
}

// ช่วงพิกเซลภาพที่มองเห็นอยู่ (ครึ่งเปิด [x0, x1) × [y0, y1)) — ใช้วาดตารางเฉพาะส่วนที่เห็น
export function visiblePixelRange(view, geom) {
  const r = fitRect(geom.vw, geom.vh, geom.iw, geom.ih);
  if (!r) return { x0: 0, y0: 0, x1: 0, y1: 0 };
  const s = r.f * view.z;
  const ix = (v) => ((v - view.tx) / view.z - r.x) / r.f;
  const iy = (v) => ((v - view.ty) / view.z - r.y) / r.f;
  return {
    x0: Math.max(0, Math.floor(ix(0))), x1: Math.min(geom.iw, Math.ceil(ix(geom.vw))),
    y0: Math.max(0, Math.floor(iy(0))), y1: Math.min(geom.ih, Math.ceil(iy(geom.vh))),
    s,
    // พิกัดจอของเส้นแบ่งพิกเซลที่ index i: gx(i) = ox + i * s
    ox: view.tx + view.z * r.x, oy: view.ty + view.z * r.y,
  };
}

// อ่านพิกเซล RGBA ที่ (x, y) จาก data (Uint8ClampedArray) → [r,g,b] ใส่ใน out · นอกช่วง/ข้อมูลไม่พอ → null
export function readRGB(data, w, h, x, y, out) {
  if (!data || x < 0 || y < 0 || x >= w || y >= h) return null;
  const i = (y * w + x) * 4;
  if (i + 2 >= data.length) return null;
  const o = out || [0, 0, 0];
  o[0] = data[i]; o[1] = data[i + 1]; o[2] = data[i + 2];
  return o;
}

// ข้อความ readout: ภาพเทา (r = g = b) → "เทา 128" · สี → "R 12  G 34  B 56"
export function formatPixel(rgb) {
  if (!rgb) return "";
  const [r, g, b] = rgb;
  return r === g && g === b ? `เทา ${r}` : `R ${r}  G ${g}  B ${b}`;
}
export function formatPos(x, y) {
  return `x ${x}  y ${y}`;
}
export function formatZoomPct(scale) {
  return Number.isFinite(scale) ? `${Math.round(scale * 100)}%` : "—";
}

// =====================================================================
// 3.2 Histogram: เส้นเกณฑ์ T + ขาเข้า/ขาออก
// =====================================================================
// op ที่ "ตัดสินด้วยเกณฑ์" — ดู histogram ขาเข้าพร้อมเส้น T (ขาออกเป็นขาว/ดำล้วน 2 แท่ง ไม่มีอะไรให้เทียบ)
export const HISTOGRAM_THRESHOLD_OPS = new Set(["binary_threshold", "otsu_threshold", "adaptive_threshold"]);
// op ปรับภาพ/เพิ่มคุณภาพ — ดู histogram ขาเข้า (จาง) ซ้อนขาออก
export const HISTOGRAM_COMPARE_OPS = new Set([
  "brightness_contrast", "gamma", "hist_equalize", "clahe", "contrast_stretch", "rgb_adjust", "invert", "white_balance",
]);

// opId = วิธีที่เลือกอยู่จริง (กล่องหมวดรวมใช้ params.mode) → "threshold" | "compare" | "plain"
export function histogramPlan(opId) {
  if (HISTOGRAM_THRESHOLD_OPS.has(opId)) return "threshold";
  if (HISTOGRAM_COMPARE_OPS.has(opId)) return "compare";
  return "plain";
}

// ค่าเกณฑ์ที่จะขีดเส้น: binary = ค่าที่ตั้ง · otsu = ค่าที่ worker รายงานใน note.info ("... = 137") · adaptive/อื่น ๆ = null
// localParams = พารามิเตอร์ของ op นั้นที่ตัด prefix แล้ว (localParamsFor) · infoTexts = note.info ของกล่องนี้ (infoTextsOf)
export function thresholdMarker(opId, localParams, infoTexts) {
  const clamp = (v) => (Number.isFinite(v) ? Math.max(0, Math.min(255, Math.round(v))) : null);
  if (opId === "binary_threshold") {
    const v = localParams && localParams.thresh;
    return v == null || v === "" ? null : clamp(Number(v));
  }
  if (opId === "otsu_threshold") {
    for (const t of infoTexts || []) {
      const m = /=\s*(-?\d+(?:\.\d+)?)/.exec(String(t));
      if (m) return clamp(Number(m[1]));
    }
    return null;
  }
  return null;
}

// ตำแหน่ง x ของเส้น T บนกราฟ (ซ้ายสุด left ↔ ขวาสุด right สำหรับค่า 0..255)
export function markerX(t, left, right) {
  return left + (Math.max(0, Math.min(255, t)) / 255) * (right - left);
}

// =====================================================================
// 3.4 พื้นหลังแบบจุด · 3.5 สายวิ่ง
// =====================================================================
export const DOT_SPACING = 24;
export const DOT_MIN_SCALE = 0.6;
export const DOT_MAX_STRIDE = 16;
export const DOT_MAX_COUNT = 8000; // กันซูมออกจนจุดเป็นหมื่น (scale > 0.6 ทำให้ไม่เกิดจริง แต่กัน viewport ใหญ่ผิดปกติ)

// พื้นที่ที่มองเห็น (พิกัด graph) → { x0, y0, cols, rows } ของจุดที่ตรงพิกัดคูณ spacing · null = ไม่ต้องวาด
export function dotGridRange(visible, scale, spacing = DOT_SPACING) {
  if (!visible || !(scale > DOT_MIN_SCALE)) return null;
  const [vx, vy, vw, vh] = visible;
  if (!(vw > 0) || !(vh > 0)) return null;
  // พื้นที่กว้างมาก (จอใหญ่ ๆ ที่ scale เพิ่งพ้น 0.6) → ขยายระยะเป็น 2×, 4×, … (เป็นเซตย่อยของตารางเดิม) แทนที่จะไม่วาดเลย
  for (let stride = 1; stride <= DOT_MAX_STRIDE; stride *= 2) {
    const sp = spacing * stride;
    const x0 = Math.ceil(vx / sp) * sp;
    const y0 = Math.ceil(vy / sp) * sp;
    const cols = x0 > vx + vw ? 0 : Math.floor((vx + vw - x0) / sp) + 1;
    const rows = y0 > vy + vh ? 0 : Math.floor((vy + vh - y0) / sp) + 1;
    if (cols <= 0 || rows <= 0) return null;
    if (cols * rows <= DOT_MAX_COUNT) return { x0, y0, cols, rows, spacing: sp };
  }
  return null;
}

// สายวิ่งทำงานเมื่อ: กล้องสด (โหมดต้นฉบับ) กำลังเดิน ไม่ pause และผู้ใช้ไม่ได้ขอลดแอนิเมชัน
export function flowActive({ source, cameraMode, cameraStarted, paused, reducedMotion } = {}) {
  return source === "camera" && cameraMode === "original" && !!cameraStarted && !paused && !reducedMotion;
}

// ตำแหน่ง 0..1 ของจุดที่ i จาก n บนสาย ณ เวลา timeMs (ไล่ไปทางปลายสาย)
export function flowT(timeMs, i, n, periodMs = 1600) {
  const t = (timeMs / periodMs + i / n) % 1;
  return t < 0 ? t + 1 : t;
}

// จำนวนจุดต่อสายตามความยาว (สายสั้น 1–2 จุด · ยาวสุด 6)
export function flowDotCount(dist) {
  return Math.max(1, Math.min(6, Math.round(dist / 70)));
}

// =====================================================================
// 3.6 สถานะ error ราย node
// =====================================================================
export const ERROR_FALLBACK_TEXT = "ประมวลผลไม่สำเร็จ";

// ข้อความ error สั้น ๆ บรรทัดเดียว (ตัดตามความกว้างจริงอีกชั้นตอนวาด) · ว่าง/ไม่ใช่ข้อความ → ข้อความสำรอง
export function shortErrorText(message, max = 120) {
  const s = typeof message === "string" ? message.replace(/\s+/g, " ").trim() : "";
  if (!s || s === "undefined" || /^[0-9]+$/.test(s)) return ERROR_FALLBACK_TEXT; // เลขพอยน์เตอร์ดิบของ OpenCV.js ไม่ใช่ข้อความ
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}

// msg จาก worker {type:"error", message, nodeId?} → { nodeId | null, text }
export function routeWorkerError(msg) {
  const id = msg ? msg.nodeId : null;
  return {
    nodeId: Number.isFinite(id) ? id : null,
    text: shortErrorText(msg && msg.message),
  };
}

// =====================================================================
// 3.7 คีย์ลัด 1–9 / F
// =====================================================================
// เลขที่ผู้ใช้เห็นบนการ์ด: 1 = แหล่งภาพ, 2.. = Function ตามลำดับ (รวมกล่อง Model Detect), สุดท้าย = Display
// dp.stepIds = [id แหล่งภาพ, ...id กล่องบนสาย, id Display] (editor สร้างใน tracePipeline) · ขอบเขต = สายที่มีกล่องที่เลือกอยู่
// ไม่มีกล่องที่เลือก/ไม่อยู่ในสายไหน → สายแรกที่ต่อสำเร็จ · คืน id ของ node หรือ null
export function nthStepNodeId(pipelines, n, selectedId) {
  if (!(n >= 1) || !Number.isInteger(n)) return null;
  const conn = (pipelines || []).filter((d) => d && d.connected && Array.isArray(d.stepIds) && d.stepIds.length);
  if (!conn.length) return null;
  const scope = (selectedId != null && conn.find((d) => d.stepIds.includes(selectedId))) || conn[0];
  const id = scope.stepIds[n - 1];
  return id == null ? null : id;
}

// คีย์เดี่ยว (ไม่มี modifier) ที่เราตีความ: "f"/"F" → { type: "fit" } · "1".."9" → { type: "step", n } · "0" → { type: "reset-view" }
export function parseNavKey(e) {
  if (!e || e.ctrlKey || e.altKey || e.metaKey || e.shiftKey || e.repeat) return null;
  const k = e.key;
  if (k === "f" || k === "F") return { type: "fit" };
  if (typeof k === "string" && k.length === 1 && k >= "1" && k <= "9") return { type: "step", n: Number(k) };
  if (k === "0") return { type: "reset-view" };
  // แป้นพิมพ์ไทย (เกษมณี) e.key เป็นอักษรไทย/เลขไทย → ใช้ตำแหน่งแป้นจริง (e.code) แทน · ไม่ทับกรณีที่ e.key ถูกจัดการแล้วข้างบน
  const c = e.code;
  if (typeof c === "string") {
    if (c === "KeyF") return { type: "fit" };
    const m = /^(?:Digit|Numpad)([0-9])$/.exec(c);
    if (m) return m[1] === "0" ? { type: "reset-view" } : { type: "step", n: Number(m[1]) };
  }
  return null;
}

// id ของกล่องที่เลือกอยู่ (กล่อง filter หรือกล่อง Display) — ใช้เลือกสายสำหรับคีย์ 1–9
export function selectedIdOf(st) {
  if (!st) return null;
  if (st.selectedNode && st.selectedNode.id != null) return st.selectedNode.id;
  if (st.selectedDisplay && st.selectedDisplay.id != null) return st.selectedDisplay.id;
  return null;
}

// id ของกล่องตามเลขบนการ์ด: [แหล่งภาพ, ...กล่องบนสาย, Display] · ไม่มีแหล่งภาพ (ยังไม่ต่อสาย) → []
export function buildStepIds(sourceId, pathIds, displayId) {
  if (sourceId == null || displayId == null) return [];
  return [sourceId, ...(pathIds || []), displayId];
}

// =====================================================================
// 3.2 เลือก pipeline ที่ส่ง worker + แยกประเภทผลลัพธ์
// =====================================================================
export const SELECTED_PREVIEW_PREFIX = "__selected_function_preview__";
export const SELECTED_INPUT_PREFIX = "__selected_function_input__";
export const INPUT_HIST_INTERVAL_MS = 300;

// pipeline ขาเข้า = สาย preview ตัดขั้นสุดท้าย (ตัวกล่องที่เลือก) ออก
export function buildInputPipeline(preview, displayId) {
  if (!preview || !Array.isArray(preview.pipeline)) return null;
  return { displayId, connected: true, dl: false, silentNotes: true, pipeline: preview.pipeline.slice(0, -1) };
}

// รายการ pipeline ที่ส่ง worker: จอปกติ → (ขาเข้า) → preview · ขาเข้าต้องมาก่อน preview (ผลมาตามลำดับ → histogram ของ preview เจอขาเข้าพร้อมแล้ว)
// ภาพนิ่ง = ขอทุกครั้ง · กล้องสด = ขอเมื่อยังไม่เคยได้ หรือผ่านมา ≥ intervalMs
export function pickWorkerPipelines({ displays, preview, input, source, inputReady, inputAt, now, intervalMs = INPUT_HIST_INTERVAL_MS }) {
  const out = [...(displays || [])];
  if (preview && input && (source === "image" || !inputReady || now - inputAt >= intervalMs)) out.push(input);
  if (preview) out.push(preview);
  return out;
}

// ผลลัพธ์จาก worker ชิ้นนี้เป็นของอะไร: "input" | "preview" | "display" | "stale" (รุ่น generation เก่า/ไม่ได้ขอแล้ว → ทิ้ง)
// ผลขาเข้าห้ามหลุดไปเส้นทาง display เด็ดขาด (จะไปวาดทับจอผลลัพธ์)
export function classifyResultId(id, curInputId, curPreviewId) {
  if (typeof id === "string") {
    if (id.startsWith(SELECTED_INPUT_PREFIX)) return curInputId != null && id === curInputId ? "input" : "stale";
    if (id.startsWith(SELECTED_PREVIEW_PREFIX)) return curPreviewId != null && id === curPreviewId ? "preview" : "stale";
  }
  return "display";
}

// เส้นเกณฑ์ T: ค่าจาก note/ค่าตั้ง ก่อน · Otsu ที่ไม่มี note → คำนวณจาก histogram luma ขาเข้า
export function resolveThresholdT(opId, localParams, infoTexts, lumaBins) {
  const t = thresholdMarker(opId, localParams, infoTexts);
  if (t == null && opId === "otsu_threshold" && lumaBins) return otsuFromHistogram(lumaBins);
  return t;
}

// =====================================================================
// 3.5/3.8 ชิ้นส่วนของ editor ที่แยกมาเทสต์ได้
// =====================================================================
// พื้นที่ที่มองเห็น (พิกัด graph) จาก buffer (ew × eh), อัตราส่วน buffer/CSS (kx, ky) และซูม
export function visibleAreaOf(offset, ew, eh, kx, ky, scale, out) {
  const o = out || [0, 0, 0, 0];
  o[0] = -offset[0]; o[1] = -offset[1];
  o[2] = ew / kx / scale; o[3] = eh / ky / scale;
  return o;
}

// สายนี้อยู่ใน pipeline? (ปลายทั้งสองถูกตั้ง __mvInPipeline)
export function isFlowLink(from, to) {
  return !!(from && to && from.__mvInPipeline && to.__mvInPipeline);
}

// ตัวคุม loop สายวิ่ง: เปิด → ขอ RAF และวาดไม่ถี่กว่า minGapMs (เมื่อ hasWork) · ปิด → ยกเลิก RAF ทันที + เรียก onStop
export function createFlowLoop({ raf, caf, isOn, hasWork, redraw, onStop, minGapMs = 40 }) {
  let id = 0;
  let last = 0;
  function tick(ts) {
    id = 0;
    if (!isOn()) return;
    if (ts - last >= minGapMs && hasWork()) { last = ts; redraw(); }
    id = raf(tick);
  }
  return {
    sync() {
      if (isOn()) { if (!id) id = raf(tick); }
      else if (id) { caf(id); id = 0; if (onStop) onStop(); }
    },
    running: () => id !== 0,
  };
}

// =====================================================================
// 3.8 ขนาด buffer ของ canvas ที่ DPR ≠ 1
// =====================================================================
// rect = ขนาด CSS จริง (getBoundingClientRect) · dpr = devicePixelRatio
// คืน { w, h = ขนาด buffer (จำนวนเต็ม), kx, ky = buffer px ต่อ CSS px (≈ dpr; คิดจากอัตราส่วนจริงของ buffer ที่ปัดเศษแล้ว
// เพื่อให้เมาส์กับภาพตรงกันเป๊ะ), cw, ch = ขนาด CSS) · ซ่อนอยู่/เล็กเกิน → null
// dpr < 1 (ซูมออก) ใช้ 1 (buffer = CSS เหมือนเดิม) · เพดาน 3
export function canvasBufferSizeDpr(rect, dpr) {
  if (!rect || !(rect.width >= 1) || !(rect.height >= 1)) return null;
  const d = Math.min(3, Math.max(1, Number.isFinite(dpr) ? dpr : 1));
  const w = Math.round(rect.width * d);
  const h = Math.round(rect.height * d);
  return { w, h, kx: w / rect.width, ky: h / rect.height, cw: rect.width, ch: rect.height };
}

// ขนาด buffer ของ histogram canvas (geometry ตรรกะคงที่ lw×lh) ให้คมที่ DPR · cssW ≤ 0 (ซ่อน) → ใช้ lw
export function histogramBufferSize(cssW, dpr, lw = 320, lh = 172) {
  const d = Math.min(3, Math.max(1, Number.isFinite(dpr) ? dpr : 1));
  const w = Math.max(lw, Math.round((cssW > 0 ? cssW : lw) * d));
  return { w, h: Math.round((w * lh) / lw), scale: w / lw };
}

// ค่าเกณฑ์ Otsu จาก histogram 256 ช่อง (สำรองเมื่อกล่องยังไม่ต่อ Display จึงไม่มี note จาก worker) — สูตรเดียวกับ OpenCV:
// เลือก t ที่ทำให้ความแปรปรวนระหว่างกลุ่ม (w0·w1·(μ0−μ1)²) มากสุด · ภาพสีเดียว/ว่าง → 0
export function otsuFromHistogram(bins) {
  let total = 0, sum = 0;
  for (let i = 0; i < 256; i++) { total += bins[i]; sum += i * bins[i]; }
  if (!(total > 0)) return 0;
  let wB = 0, sumB = 0, best = -1, thr = 0;
  for (let t = 0; t < 256; t++) {
    wB += bins[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * bins[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) { best = between; thr = t; }
  }
  return thr;
}

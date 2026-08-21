// จุดเริ่มของ renderer: node-graph สไตล์ Node-RED + worker + แหล่งภาพ (กล้อง/ภาพนิ่ง)
// การไหล: [แหล่งภาพ] → filter nodes → [ผลลัพธ์] — กราฟเปลี่ยนเมื่อไร ประมวลผลใหม่ทันที
import { createGraphEditor } from "./graph/editor.js";
import { createPalettePanel } from "./ui/palette-panel.js";
import { createDetailPanel } from "./ui/detail-panel.js";
import { createSourceBar } from "./ui/source-bar.js";
import { createExportDialog } from "./ui/export-dialog.js";
import { startCamera, listCameras } from "./camera.js";
import { startPose, stopPose, pausePose, resumePose, renderStillErgonomics } from "./mediapipe-pose.js";
import { loadModel as dlLoadModel, isLoaded as dlLoaded, detect as dlDetect, drawDetections as dlDraw, COCO } from "./dl-yolo.js";
import { FILTERS } from "./filters/registry.js";
import { buildPythonCode } from "./codegen.js";
import { showCodeDialog } from "./ui/code-dialog.js";
import { openSettings } from "./ui/settings-dialog.js";
import { openVideoFrames } from "./ui/video-frames-dialog.js";
import { showTemplates } from "./ui/onboarding.js";

const loadingEl = document.getElementById("loading");
const loadingText = document.getElementById("loading-text");
const appEl = document.getElementById("app");
// เติมเลขเวอร์ชันจริงจาก package.json (ผ่าน preload) — กันข้อความ "vX.X.X" ค้างของเก่าไม่ตรง build จริง
{
  const verEl = document.getElementById("status-ver");
  const v = window.appInfo?.version;
  if (verEl && v) verEl.textContent = `Machine Vision by IRiSH · v${v}`;
}
const videoEl = document.getElementById("camera");
const gridEl = document.getElementById("stage-grid");
const statusEl = document.getElementById("status");
const cameraError = document.getElementById("camera-error");
const fileInput = document.getElementById("file-input");
const folderInput = document.getElementById("folder-input");
const graphCanvasEl = document.getElementById("graph-canvas");
const poseCanvas = document.getElementById("pose-canvas");
const detCanvas = document.getElementById("det-canvas");
const imageEmptyEl = document.getElementById("image-empty");
const videoSourceEl = document.getElementById("video-source");
const sourceVideoEl = document.getElementById("source-video");
const modelInput = document.getElementById("model-input");
const labelsInput = document.getElementById("labels-input");
const selectedPreviewCanvas = document.getElementById("selected-preview-canvas");
const selectedPreviewHistogram = document.getElementById("selected-preview-histogram");
const selectedPreviewMetrics = document.getElementById("selected-preview-metrics");
const selectedPreviewEmpty = document.getElementById("selected-preview-empty");
const selectedPreviewName = document.getElementById("selected-preview-name");
const selectedPreviewTabs = [...document.querySelectorAll("[data-preview-view]")];
const SELECTED_PREVIEW_PREFIX = "__selected_function_preview__";
let selectedPreviewGeneration = 0;
let selectedPreviewView = "output";
let selectedPreviewLatestImage = null;
let selectedPreviewAnalysisAt = 0;
const selectedPreviewHistogramBins = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
const HISTOGRAM_DEFAULT_OPS = new Set([
  "grayscale", "colorspace", "channel_split", "color_inrange", "invert",
  "brightness_contrast", "gamma", "hist_equalize", "clahe",
  "binary_threshold", "otsu_threshold", "adaptive_threshold",
]);

const scratchCanvas = document.createElement("canvas");

// ---------- ระดับความละเอียดประมวลผล ----------
// ย่อภาพก่อนส่งเข้า worker เพื่อลดงานต่อเฟรม (ลด delay) — ยิ่งเล็กยิ่งเร็ว
// ค่า = ความยาวด้านที่ยาวสุด (px); 0 = ไม่ย่อ (เต็มความละเอียดต้นฉบับ)
export const PROC_LEVELS = [
  { value: "fast", label: "เร็ว (480p)", max: 480 },
  { value: "balanced", label: "สมดุล (720p)", max: 720 },
  { value: "detail", label: "ละเอียด (1080p)", max: 1080 },
  { value: "full", label: "ภาพเต็ม(ต้นฉบับ)", max: 0 },
];

// ---------- state ----------
const state = {
  pipelines: [], // หนึ่งรายการต่อหนึ่งกล่องผลลัพธ์: [{ displayId, name, pipeline, connected }]
  source: "camera", // 'camera' | 'image'
  cameraMode: "original", // โหมดย่อยของกล้อง: 'original' | 'pose'
  cameras: [], // กล้องที่ต่ออยู่: [{ deviceId, label }]
  cameraId: null, // กล้องที่กำลังใช้ (deviceId)
  labels: COCO, // ชื่อคลาสของโมเดล DL (ค่าเริ่ม = COCO)
  images: [], // ชุดภาพนำเข้า (batch): [{ canvas, w, h, name }]
  imageIndex: 0, // ภาพที่กำลังดูอยู่ในชุด
  procLevel: "balanced", // ระดับความละเอียดประมวลผล (ดู PROC_LEVELS)
  notes: [], // จาก worker เช่น auto-gray
  selectedNode: null, // filter node ที่เลือก: { filterId, getParams, setParam }
  selectedDisplay: null, // กล่องผลลัพธ์ที่เลือก: { id, getName, setName }
  selectedDL: false, // เลือกกล่อง YOLO Detect อยู่ไหม
  captures: [], // ภาพที่แคปไว้: [{ canvas, w, h, name }]
  selectedCapture: -1, // index ภาพแคปที่เลือกในถาด
  paused: false, // หยุดประมวลผลสดชั่วคราว (Run/Pause)
};

// สร้างคำอธิบาย "ภาพผ่านอะไรมา" จาก pipeline ของกล่องผลลัพธ์
function autoLabel(pipeline) {
  if (!pipeline || pipeline.length === 0) return "แหล่งภาพ (ต้นฉบับ)";
  const parts = pipeline.map((s) => {
    const f = FILTERS[s.id];
    if (!f) return s.id;
    return typeof f.subtitle === "function" ? f.subtitle(s.params) : f.name;
  });
  return "แหล่งภาพ → " + parts.join(" → ");
}

// chip ตัวตนของกล่อง: ชื่อที่ผู้ใช้ตั้ง > "Display N" (auto)
function displayChip(dp) {
  return dp.name || "Display " + (dp.autoNum || 1);
}
// ใช้กับชื่อไฟล์ export (สั้น สื่อความหมาย)
function captionFor(dp) {
  return displayChip(dp);
}

// canvas สำหรับย่อภาพนิ่งก่อนส่ง (แยกจาก scratchCanvas ที่กล้องใช้)
const procCanvas = document.createElement("canvas");

// คืนค่า max px ของระดับที่เลือก
function procMax() {
  return (PROC_LEVELS.find((l) => l.value === state.procLevel) || PROC_LEVELS[1]).max;
}

// คำนวณขนาดหลังย่อ (คงอัตราส่วน) — ถ้าไม่เกินเกณฑ์ หรือระดับ = เต็ม คืนขนาดเดิม
function processDims(w, h) {
  const max = procMax();
  if (!max || Math.max(w, h) <= max) return { w, h };
  const s = max / Math.max(w, h);
  return { w: Math.round(w * s), h: Math.round(h * s) };
}

// offscreen canvas ต่อกล่องผลลัพธ์ (ใช้ blit เข้าไปวาดในกล่องบนกราฟ)
const displayCanvases = new Map();
function displayCanvasFor(id) {
  let c = displayCanvases.get(id);
  if (!c) {
    c = document.createElement("canvas");
    displayCanvases.set(id, c);
  }
  return c;
}

// ---------- grid จอผลลัพธ์ (1 กล่อง = 1 จอ + หัวข้อกำกับ) ----------
const cells = new Map(); // displayId -> { root, canvas, ctx, cap }
let gridKey = "";
// สร้าง/ปรับ grid ให้ตรงกับชุดกล่องผลลัพธ์ปัจจุบัน (rebuild เฉพาะเมื่อชุดเปลี่ยน)
function ensureGrid(order) {
  const key = order.join(",");
  if (key === gridKey) return;
  gridKey = key;
  closeAllChainMenus();
  gridEl.innerHTML = "";
  cells.clear();
  const cols = Math.min(order.length, Math.ceil(Math.sqrt(order.length)) || 1);
  gridEl.style.setProperty("--cols", cols || 1);
  for (const id of order) {
    const root = document.createElement("div");
    root.className = "stage-cell";

    // หัวจอ: [chip ตัวตน] + สายทอด (breadcrumb) ของ step แต่ละอันเป็น dropdown
    const cap = document.createElement("div");
    cap.className = "stage-cap";
    const chip = document.createElement("span");
    chip.className = "stage-chip";
    const chain = document.createElement("span"); // ภาพ › step ▾ › step ▾
    chain.className = "stage-chain";
    cap.appendChild(chip);
    cap.appendChild(chain);

    const cvs = document.createElement("canvas");
    cvs.className = "stage-canvas";

    root.appendChild(cap);
    root.appendChild(cvs);
    gridEl.appendChild(root);
    cells.set(id, { root, canvas: cvs, ctx: cvs.getContext("2d"), chip, chain });
  }
}

let workerReady = false;
let busy = false;
let procFrames = 0; // ตัวนับเฟรมที่ประมวลผล (สำหรับ FPS)
let pendingProcess = false; // มีคำขอประมวลผลค้างระหว่าง worker busy (โหมดภาพนิ่ง)
let cameraStarted = false;

// ---------- worker ----------
const worker = new Worker(new URL("./cv/cv-worker.js", import.meta.url), {
  type: "module",
});

worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "ready") {
    workerReady = true;
    onOpenCVReady();
  } else if (msg.type === "result") {
    procFrames++; // นับเฟรมที่ประมวลผลจริง → คำนวณ FPS
    drawResults(msg.results || []);
    const changed = JSON.stringify(msg.notes) !== JSON.stringify(state.notes);
    state.notes = msg.notes || [];
    busy = false;
    if (changed) renderDetail();
    // มีค่าที่เลื่อนใหม่ระหว่าง worker ยังคำนวณอยู่ → ประมวลผลค่าล่าสุดทันที (โหมดภาพนิ่ง)
    if (pendingProcess && state.source === "image") {
      pendingProcess = false;
      requestProcess();
    }
  } else if (msg.type === "export-result") {
    const waiter = exportWaiters.get(msg.reqId);
    if (waiter) {
      exportWaiters.delete(msg.reqId);
      waiter(msg);
    }
  } else if (msg.type === "error") {
    statusEl.textContent = "✕ ประมวลผลไม่สำเร็จ — ลองปรับค่าหรือเปลี่ยนภาพดู (" + msg.message + ")";
    busy = false;
    // ค่าล่าสุดที่ค้างไว้ควรลองใหม่ ไม่งั้นภาพค้างที่ค่าเก่าจนกว่าจะขยับ slider อีก
    if (pendingProcess && state.source === "image") {
      pendingProcess = false;
      requestProcess();
    }
  }
};

// worker ล่ม (WASM OOM/abort) → onmessage เงียบตลอด: busy จะค้าง true (live แช่แข็ง)
// และ exportWaiters จะไม่ resolve (exportBatch await ค้าง) — ปลดล็อกทุกอย่างแล้วแจ้งผู้ใช้
worker.onerror = (e) => {
  busy = false;
  pendingProcess = false;
  for (const waiter of exportWaiters.values()) waiter({ error: e.message || "worker error" });
  exportWaiters.clear();
  if (statusEl) statusEl.textContent = "✕ ตัวประมวลผลภาพขัดข้อง — ลองปิด-เปิดโปรแกรมใหม่";
};

// รอผลจาก worker สำหรับ export ทีละภาพ (คู่กับ reqId)
const exportWaiters = new Map();
let exportSeq = 0;

// ---------- error surfacing ----------
window.addEventListener("error", (e) => {
  loadingText.textContent = "เกิดข้อผิดพลาด: " + (e.message || e.error);
});
window.addEventListener("unhandledrejection", (e) => {
  loadingText.textContent = "เกิดข้อผิดพลาด (promise): " + (e.reason?.message || e.reason);
});

// ---------- graph editor ----------
const editor = createGraphEditor(graphCanvasEl, {
  onPipelineChange: (pipelines) => {
    state.pipelines = pipelines;
    // เก็บกวาด canvas ของกล่องผลลัพธ์ที่ถูกลบไปแล้ว
    const alive = new Set(pipelines.map((d) => d.displayId));
    for (const id of displayCanvases.keys()) {
      if (!alive.has(id)) displayCanvases.delete(id);
    }
    updateStatus();
    updateCaptions(); // ชื่อ/สายเปลี่ยน → อัปเดตหัวจอทันที
    requestProcess();
  },
  onSelect: (sel) => {
    selectedPreviewGeneration++;
    selectedPreviewLatestImage = null;
    selectedPreviewAnalysisAt = 0;
    selectedPreviewCanvas.classList.add("hidden");
    selectedPreviewHistogram.classList.add("hidden");
    selectedPreviewMetrics.classList.add("hidden");
    delete selectedPreviewCanvas.dataset.ready;
    delete selectedPreviewHistogram.dataset.ready;
    delete selectedPreviewMetrics.dataset.ready;
    state.selectedNode = null;
    state.selectedDisplay = null;
    state.selectedDL = false;
    if (sel && sel.kind === "filter") state.selectedNode = sel;
    else if (sel && sel.kind === "display") {
      // ให้แผงขวาดึงขั้นตอนล่าสุดของกล่องนี้ + กระโดดไป node ได้
      sel.getSteps = () => state.pipelines.find((d) => d.displayId === sel.id) || null;
      sel.onJump = (nid) => editor.selectNodeById(nid);
      // สร้างโค้ด Python ตามลำดับ Box ที่ต่อเข้ากล่องนี้
      sel.onGenCode = () => {
        const dp = state.pipelines.find((d) => d.displayId === sel.id);
        const name = dp ? displayChip(dp) : "ผลลัพธ์";
        const code = buildPythonCode(dp ? dp.pipeline : [], { displayName: name });
        showCodeDialog(code, { title: `🐍 โค้ด Python — ${name}`, filename: "image_processing.py" });
      };
      state.selectedDisplay = sel;
    } else if (sel && sel.kind === "dl") state.selectedDL = true;
    if (state.selectedNode) setSelectedPreviewView(defaultSelectedPreviewView(), false);
    renderDetail();
    renderSelectedPreviewState();
    // ภาพนิ่งต้องสั่งรันทันที; กล้องจะได้ preview ใน pump เฟรมถัดไป
    if (state.selectedNode) requestProcess();
  },
});

function selectedPreviewPipeline() {
  const sel = state.selectedNode;
  if (!sel?.getPreviewPipeline) return null;
  if (state.source === "image" && state.images.length === 0) return null;
  if (state.source === "camera" && (!cameraStarted || state.cameraMode !== "original")) return null;
  if (state.source !== "image" && state.source !== "camera") return null;
  const traced = sel.getPreviewPipeline();
  if (!traced?.connected || !traced.pipeline?.length) return null;
  return {
    displayId: `${SELECTED_PREVIEW_PREFIX}:${sel.id}:${selectedPreviewGeneration}`,
    connected: true,
    dl: false,
    silentNotes: true,
    pipeline: traced.pipeline,
  };
}

function selectedPreviewOpId() {
  if (!state.selectedNode) return "";
  const params = state.selectedNode.getParams?.() || {};
  return params.mode || state.selectedNode.filterId || "";
}

function defaultSelectedPreviewView() {
  return HISTOGRAM_DEFAULT_OPS.has(selectedPreviewOpId()) ? "histogram" : "metrics";
}

function setSelectedPreviewView(view, userInitiated = true) {
  selectedPreviewView = view;
  for (const tab of selectedPreviewTabs) {
    const active = tab.dataset.previewView === view;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  }
  const ready = view === "output"
    ? selectedPreviewCanvas.dataset.ready
    : view === "histogram"
      ? selectedPreviewHistogram.dataset.ready
      : selectedPreviewMetrics.dataset.ready;
  selectedPreviewCanvas.classList.toggle("hidden", view !== "output" || !selectedPreviewCanvas.dataset.ready);
  selectedPreviewHistogram.classList.toggle("hidden", view !== "histogram" || !selectedPreviewHistogram.dataset.ready);
  selectedPreviewMetrics.classList.toggle("hidden", view !== "metrics" || !selectedPreviewMetrics.dataset.ready);
  if (userInitiated && state.selectedNode) selectedPreviewEmpty.classList.toggle("hidden", Boolean(ready));
  if (userInitiated && view !== "output" && selectedPreviewLatestImage) {
    renderSelectedPreviewAnalysis(
      selectedPreviewLatestImage.img,
      selectedPreviewLatestImage.width,
      selectedPreviewLatestImage.height,
    );
    selectedPreviewEmpty.classList.add("hidden");
    setSelectedPreviewView(view, false);
  }
}

for (const tab of selectedPreviewTabs) {
  tab.addEventListener("click", () => setSelectedPreviewView(tab.dataset.previewView));
}

function invalidateSelectedPreview() {
  // ปรับพารามิเตอร์ → ผลเก่าใน worker ถือว่าใช้ไม่ได้ (bump generation ให้ผลเก่าที่ค้างมาถูกทิ้ง)
  // แต่ "ไม่ซ่อน canvas / ไม่โชว์ Processing…" — คงภาพเดิมไว้จนผลใหม่มาทับในที่ กันแผงขวากระพริบตอนลากค่า
  selectedPreviewGeneration++;
  selectedPreviewAnalysisAt = 0;
}

function renderSelectedPreviewState() {
  const preview = selectedPreviewPipeline();
  if (!preview) {
    selectedPreviewCanvas.classList.add("hidden");
    selectedPreviewEmpty.classList.remove("hidden");
    selectedPreviewName.textContent = state.selectedNode ? "Not connected" : "No selection";
    selectedPreviewEmpty.textContent = state.selectedNode
      ? "Connect this Function to Source to preview its result"
      : "Select a Function box to preview its output";
    return;
  }
  const f = FILTERS[state.selectedNode.filterId];
  const params = state.selectedNode.getParams();
  selectedPreviewName.textContent = typeof f?.subtitle === "function" ? f.subtitle(params) : (f?.name || "Selected Function");
  selectedPreviewEmpty.textContent = "Processing preview…";
  selectedPreviewEmpty.classList.remove("hidden");
  // canvas จะแสดงหลัง Worker ส่งเฟรมล่าสุดกลับมา
}

function renderSelectedPreviewAnalysis(img, width, height) {
  const hist = selectedPreviewHistogramBins;
  for (const bins of hist) bins.fill(0);
  const data = img.data;
  const pixelCount = width * height;
  const sampleStep = Math.max(1, Math.ceil(pixelCount / 160000));
  let samples = 0;
  let sum = 0;
  let sumSq = 0;
  let min = 255;
  let max = 0;
  let active = 0;
  let detail = 0;
  let previous = null;
  let grayscale = true;
  for (let p = 0; p < pixelCount; p += sampleStep) {
    const i = p * 4;
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const y = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    hist[0][r]++; hist[1][g]++; hist[2][b]++;
    grayscale &&= r === g && g === b;
    sum += y; sumSq += y * y;
    if (y < min) min = y;
    if (y > max) max = y;
    if (y > 8) active++;
    if (previous !== null) detail += Math.abs(y - previous);
    previous = y;
    samples++;
  }

  const ctx = selectedPreviewHistogram.getContext("2d");
  const w = selectedPreviewHistogram.width;
  const h = selectedPreviewHistogram.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#050b14";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(126, 160, 198, .13)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= 4; x++) {
    const gx = 10 + ((w - 20) * x) / 4;
    ctx.beginPath(); ctx.moveTo(gx, 10); ctx.lineTo(gx, h - 22); ctx.stroke();
  }
  for (let y = 0; y <= 3; y++) {
    const gy = 10 + ((h - 32) * y) / 3;
    ctx.beginPath(); ctx.moveTo(10, gy); ctx.lineTo(w - 10, gy); ctx.stroke();
  }
  const channels = grayscale ? [[hist[0], "#38d9ff"]] : [
    [hist[0], "#ff5577"], [hist[1], "#45e59a"], [hist[2], "#4aa3ff"],
  ];
  let peak = 1;
  for (const [bins] of channels) {
    for (let i = 0; i < 256; i++) if (bins[i] > peak) peak = bins[i];
  }
  for (const [bins, color] of channels) {
    ctx.strokeStyle = color;
    ctx.lineWidth = grayscale ? 2 : 1.35;
    ctx.beginPath();
    for (let i = 0; i < 256; i++) {
      const x = 10 + (i / 255) * (w - 20);
      const normalized = Math.log1p(bins[i]) / Math.log1p(peak);
      const y = h - 22 - normalized * (h - 34);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.fillStyle = "#71849d";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("0", 10, h - 7);
  ctx.fillText("128", w / 2 - 9, h - 7);
  ctx.fillText("255", w - 30, h - 7);
  selectedPreviewHistogram.dataset.ready = "true";

  const mean = samples ? sum / samples : 0;
  const variance = samples ? Math.max(0, sumSq / samples - mean * mean) : 0;
  const cards = [
    ["Resolution", `${width} × ${height}`],
    ["Mean", mean.toFixed(1)],
    ["Range", `${min}–${max}`],
    ["Contrast", Math.sqrt(variance).toFixed(1)],
    ["Active pixels", `${((active / Math.max(1, samples)) * 100).toFixed(1)}%`],
    ["Detail", (detail / Math.max(1, samples - 1)).toFixed(1)],
  ];
  if (selectedPreviewMetrics.children.length !== cards.length) {
    selectedPreviewMetrics.replaceChildren(...cards.map(([label, value]) => {
      const item = document.createElement("div");
      item.className = "selected-preview-metric";
      const lab = document.createElement("span"); lab.textContent = label;
      const val = document.createElement("strong"); val.textContent = value;
      item.append(lab, val);
      return item;
    }));
  } else {
    cards.forEach(([label, value], index) => {
      const item = selectedPreviewMetrics.children[index];
      item.firstElementChild.textContent = label;
      item.lastElementChild.textContent = value;
    });
  }
  selectedPreviewMetrics.dataset.ready = "true";
  selectedPreviewAnalysisAt = performance.now();
}

// ---------- panels ----------
const palettePanel = createPalettePanel(document.getElementById("palette-panel"), {
  onAdd: (filterId, opId) => editor.addFilterNode(filterId, opId),
  onAddDisplay: () => editor.addDisplayNode(),
  onAddDL: () => editor.addDLNode(),
  onTemplates: () => openTemplates(),
  onSaveTemplate: () => saveCurrentAsTemplate(),
});

const detailPanel = createDetailPanel(document.getElementById("detail-panel"), {
  onParamChange: (key, value) => {
    if (state.selectedNode) {
      invalidateSelectedPreview();
      state.selectedNode.setParam(key, value);
      if (key === "mode") {
        setSelectedPreviewView(defaultSelectedPreviewView(), false);
        renderSelectedPreviewState();
      }
    }
  },
});

const sourceBar = createSourceBar(document.getElementById("source-bar"), {
  getSource: () => state.source,
  onSource: (s) => setSource(s),
  getCameraMode: () => state.cameraMode,
  onCameraMode: (m) => setCameraMode(m),
  onCapture: () => captureFrame(),
  getCameras: () => state.cameras,
  getCameraId: () => state.cameraId,
  onSelectCamera: (id) => selectCamera(id),
  onImportClick: () => fileInput.click(),
  onImportFolder: () => folderInput.click(),
  levels: PROC_LEVELS,
  getLevel: () => state.procLevel,
  onLevel: (v) => {
    state.procLevel = v;
    requestProcess(); // ภาพนิ่ง: re-process ทันที (กล้องจะใช้ค่าใหม่เฟรมถัดไปเอง)
  },
  onExportCode: () => openExportCode(),
  onSaveImages: () => saveImages(),
  isPaused: () => state.paused,
  onTogglePause: () => togglePause(),
  // ตัวนำทางชุดภาพ (batch): จำนวน + ชื่อ + Prev/Next
  getBatch: () => {
    if (state.images.length === 0) return null;
    return {
      index: state.imageIndex,
      total: state.images.length,
      name: state.images[state.imageIndex]?.name || "",
    };
  },
  onPrev: () => setImageIndex(state.imageIndex - 1),
  onNext: () => setImageIndex(state.imageIndex + 1),
  onOpenSettings: () => openSettingsDialog(),
  getShortcut: (id) => state.keymap[id],
});

// Right-panel Camera Control reuses the existing camera/process controls; it adds no fake hardware settings.
const cameraControl = {
  panel: document.getElementById("camera-control-panel"),
  empty: document.getElementById("camera-control-empty"),
  body: document.getElementById("camera-control-body"),
  status: document.getElementById("camera-control-status"),
  device: document.getElementById("camera-control-device"),
  level: document.getElementById("camera-control-level"),
  resolution: document.getElementById("camera-control-resolution"),
  mode: document.getElementById("camera-control-mode"),
  capture: document.getElementById("camera-control-capture"),
  pause: document.getElementById("camera-control-pause"),
};

function renderCameraControl() {
  const active = state.source === "camera";
  const connected = active && cameraStarted;
  cameraControl.empty.classList.toggle("hidden", active);
  cameraControl.body.classList.toggle("hidden", !active);
  cameraControl.status.textContent = !active ? "Offline" : !connected ? "Not connected" : state.paused ? "Paused" : "● Live";
  cameraControl.status.classList.toggle("live", connected && !state.paused);
  if (!active) return;

  const selected = state.cameraId;
  const current = cameraControl.device.value;
  if (current !== selected || cameraControl.device.options.length !== state.cameras.length) {
    cameraControl.device.innerHTML = "";
    state.cameras.forEach((camera, i) => {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.textContent = `Port ${i + 1} — ${camera.label}`;
      option.selected = camera.deviceId === selected;
      cameraControl.device.appendChild(option);
    });
  }
  if (!cameraControl.level.options.length) {
    PROC_LEVELS.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.value;
      option.textContent = item.label;
      cameraControl.level.appendChild(option);
    });
  }
  cameraControl.level.value = state.procLevel;
  cameraControl.resolution.textContent = connected && videoEl.videoWidth ? `${videoEl.videoWidth}×${videoEl.videoHeight}` : "Not connected";
  cameraControl.mode.textContent = !connected ? "—" : state.cameraMode === "pose" ? "Pose" : (state.paused ? "Paused" : "Live");
  cameraControl.pause.textContent = state.paused ? "▶ Run" : "⏸ Pause";
  cameraControl.pause.classList.toggle("running", !state.paused);
  cameraControl.capture.disabled = !connected;
  cameraControl.pause.disabled = !connected;
}

cameraControl.device.addEventListener("change", () => selectCamera(cameraControl.device.value));
cameraControl.level.addEventListener("change", () => {
  state.procLevel = cameraControl.level.value;
  requestProcess();
  sourceBar.render();
});
cameraControl.capture.addEventListener("click", () => captureFrame());
cameraControl.pause.addEventListener("click", () => togglePause());

// ถอดรหัสภาพแบบ lazy — เก็บแค่ไฟล์ในชุด ถอดเป็น canvas เฉพาะเท่าที่ต้องใช้
// (กัน OOM เวลาโหลดภาพจำนวนมาก เช่น 200 ภาพ)
const decodeCache = new Map(); // index -> { canvas, w, h }
const DECODE_CACHE_MAX = 6;

async function ensureDecoded(i) {
  if (decodeCache.has(i)) return decodeCache.get(i);
  const rec = state.images[i];
  if (!rec) return null;
  // ภาพจากคลังแคป: ถอดไว้แล้ว (canvas อยู่ในหน่วยความจำ) ใช้ตรง ๆ
  if (rec.canvas) return { canvas: rec.canvas, w: rec.w, h: rec.h };
  const im = await loadImageFile(rec.file);
  if (!im) return null;
  // เก็บมิติไว้บน record ด้วย เพื่อให้แถบข้อมูลภาพอ่านได้หลัง lazy decode
  // (ก่อนหน้านี้ record มีแค่ file/name จึงแสดง undefined×undefined)
  rec.w = im.w;
  rec.h = im.h;
  decodeCache.set(i, im);
  if (decodeCache.size > DECODE_CACHE_MAX) {
    const oldest = decodeCache.keys().next().value; // ตัวเก่าสุด
    decodeCache.delete(oldest);
  }
  return im;
}

// เลื่อนดูภาพในชุด (วนรอบ) แล้ว re-process
function setImageIndex(i) {
  if (state.images.length === 0) return;
  const n = state.images.length;
  state.imageIndex = ((i % n) + n) % n;
  state.source = "image";
  sourceBar.render();
  updateImageEmpty();
  statusEl.textContent = `ภาพ ${state.imageIndex + 1}/${n} — ${state.images[state.imageIndex].name}`;
  requestProcess();
}

// export ผลลัพธ์แต่ละกล่อง (ที่ต่อสายครบ) เป็นไฟล์ PNG
// หมายเหตุ: รูปแบบ export ปรับเปลี่ยนได้ภายหลัง (ตอนนี้ default = PNG ต่อกล่อง)
function sanitizeName(s) {
  return (s || "result").replace(/[\\/:*?"<>|]+/g, "_").trim().slice(0, 60) || "result";
}
// ปุ่ม "โค้ด YOLO": เปิดหน้าต่างโค้ดเทรนอย่างเดียว (ไม่เซฟภาพ)
const exportDialog = createExportDialog();
function openExportCode() {
  exportDialog.open(() => {}); // โชว์/ดาวน์โหลดโค้ดเท่านั้น — เซฟภาพแยกเป็นอีกปุ่ม
}
// ปุ่ม "เซฟภาพ": บันทึกภาพผลลัพธ์เป็น PNG (เลือกโฟลเดอร์)
function saveImages() {
  const connected = state.pipelines.filter((d) => d.connected);
  if (connected.length === 0) {
    statusEl.textContent = "ยังไม่มีผลลัพธ์ให้เซฟ — ต่อสายให้ครบก่อน";
    return;
  }
  exportDataset();
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

const canExportFolder = !!(window.mvExport && window.mvExport.available);

function canvasToBlob(cv) {
  return new Promise((res) => cv.toBlob((b) => res(b), "image/png"));
}
// บันทึกไฟล์ 1 ไฟล์: โหมดโฟลเดอร์ (Electron) เขียนลง dir; ไม่งั้น fallback ดาวน์โหลด
async function saveOne(dir, name, blob) {
  if (!blob) return false;
  if (dir) {
    const bytes = new Uint8Array(await blob.arrayBuffer());
    await window.mvExport.writeFile(dir, name, bytes);
  } else {
    downloadBlob(blob, name);
  }
  return true;
}

// Export เฉพาะภาพ dataset (ผลลัพธ์ที่ผ่าน filter) — เลือกโฟลเดอร์ครั้งเดียว บันทึกทั้งหมด
async function exportDataset() {
  const connected = state.pipelines.filter((d) => d.connected);
  if (connected.length === 0) {
    statusEl.textContent = "ยังไม่มีผลลัพธ์ให้ export — ต่อสายให้ครบก่อน";
    return;
  }
  // เลือกโฟลเดอร์ปลายทางครั้งเดียว (ถ้าอยู่ใน Electron)
  let dir = null;
  if (canExportFolder) {
    dir = await window.mvExport.chooseDir();
    if (!dir) {
      statusEl.textContent = "ยกเลิกการบันทึก dataset";
      return;
    }
  }

  if (state.source === "image" && state.images.length > 1) {
    await exportBatch(connected, dir);
  } else {
    // เดี่ยว/กล้อง: บันทึกภาพผลลัพธ์ปัจจุบันของแต่ละกล่อง (จาก cache)
    let n = 0;
    let i = 0;
    for (const dp of connected) {
      const oc = displayCanvases.get(dp.displayId);
      if (!oc || !oc.width) continue;
      i++;
      const blob = await canvasToBlob(oc);
      if (await saveOne(dir, `${i}_${sanitizeName(captionFor(dp))}.png`, blob)) n++;
    }
    statusEl.textContent = n > 0
      ? `บันทึก ${n} ภาพ${dir ? " ลงโฟลเดอร์ที่เลือกแล้ว" : "แล้ว"}`
      : "ยังไม่มีภาพผลลัพธ์ (รอประมวลผลสักครู่)";
  }
}

// canvas แยกสำหรับ export (ไม่ชนกับ procCanvas ของ live)
const expCanvas = document.createElement("canvas");
function imageToProcData(imgObj) {
  const d = processDims(imgObj.w, imgObj.h);
  expCanvas.width = d.w;
  expCanvas.height = d.h;
  const ctx = expCanvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(imgObj.canvas, 0, 0, d.w, d.h);
  return ctx.getImageData(0, 0, d.w, d.h);
}
// ส่งภาพหนึ่งไป worker แล้วรอผลของทุกกล่อง (ไม่ยุ่งกับจอ live)
function processExport(imageData, pipelines) {
  return new Promise((resolve) => {
    const id = ++exportSeq;
    exportWaiters.set(id, (msg) => resolve(msg.results || []));
    const copy = new Uint8ClampedArray(imageData.data);
    worker.postMessage(
      { type: "export", reqId: id, width: imageData.width, height: imageData.height, buffer: copy.buffer, pipelines },
      [copy.buffer]
    );
  });
}
function resultToPng(result) {
  const c = document.createElement("canvas");
  c.width = result.width;
  c.height = result.height;
  c.getContext("2d").putImageData(
    new ImageData(new Uint8ClampedArray(result.buffer), result.width, result.height),
    0,
    0
  );
  return new Promise((res) => c.toBlob((b) => res(b), "image/png"));
}
function pad(n, width) {
  return String(n).padStart(width, "0");
}

let exporting = false;
async function exportBatch(connected, dir) {
  if (exporting) return;
  const total = state.images.length;
  // โหมด fallback (ไม่มีโฟลเดอร์) ถ้าไฟล์เยอะจะเด้ง dialog รัว — เตือนก่อน
  const files = total * connected.length;
  if (!dir && files > 60 && !window.confirm(`จะบันทึกทั้งหมด ${files} ไฟล์ ดำเนินการต่อ?`)) return;
  exporting = true;
  const width = String(total).length;
  let saved = 0;
  try {
    for (let i = 0; i < total; i++) {
      statusEl.textContent = `กำลัง Export ภาพ ${i + 1}/${total} …`;
      // ภาพจากไฟล์ = ถอดทีละภาพ (memory คงที่); ภาพจากคลังแคป = ถอดไว้แล้ว
      const rec = state.images[i];
      const im = rec.canvas ? { canvas: rec.canvas, w: rec.w, h: rec.h } : await loadImageFile(rec.file);
      if (!im) continue;
      const data = imageToProcData(im);
      const results = await processExport(data, state.pipelines);
      for (const dp of connected) {
        const r = results.find((x) => x.displayId === dp.displayId);
        if (!r) continue;
        const blob = await resultToPng(r);
        const base = sanitizeName(state.images[i].name.replace(/\.[^.]+$/, ""));
        const tag = connected.length > 1 ? "_" + sanitizeName(captionFor(dp)) : "";
        if (await saveOne(dir, `${pad(i + 1, width)}_${base}${tag}.png`, blob)) saved++;
      }
    }
    statusEl.textContent = `Export เสร็จ ${saved} ไฟล์ จาก ${total} ภาพ${dir ? " (ลงโฟลเดอร์ที่เลือก)" : ""}`;
  } finally {
    exporting = false;
  }
}

const detailPanelEl = document.getElementById("detail-panel");

function renderDetail() {
  // เลือกกล่อง YOLO Detect → แสดงปุ่ม browse โมเดล
  if (state.selectedDL) {
    detailPanel.renderDL({
      getStatus: () =>
        dlLoaded()
          ? `โมเดลพร้อม · ${state.labels.length} คลาส`
          : "ยังไม่ได้โหลดโมเดล — กดเลือกไฟล์ด้านล่าง",
      onBrowseModel: () => modelInput.click(),
      onBrowseLabels: () => labelsInput.click(),
    });
    return;
  }
  // เลือกกล่องผลลัพธ์ → แสดงช่องตั้งชื่อ
  if (state.selectedDisplay) {
    detailPanel.renderDisplay(state.selectedDisplay);
    return;
  }
  const sel = state.selectedNode;
  if (!sel) {
    detailPanel.render(null, null);
    return;
  }
  const notes = state.notes.filter((n) => n.id === sel.filterId);

  // ถ้าผู้ใช้กำลังลาก slider อยู่ อย่า rebuild ทั้ง panel (จะหลุดมือ) —
  // อัปเดตเฉพาะข้อความรายงานผล (เช่น จำนวนวัตถุ) แทน
  const active = document.activeElement;
  if (active && detailPanelEl.contains(active) && (active.tagName === "INPUT" || active.tagName === "SELECT")) {
    const infoEls = detailPanelEl.querySelectorAll(".detail-note-info");
    const infoNotes = notes.filter((n) => n.info);
    infoEls.forEach((el, i) => {
      if (infoNotes[i]) el.textContent = "📊 " + infoNotes[i].info;
    });
    return;
  }

  const step = { id: sel.filterId, params: sel.getParams(), enabled: true };
  detailPanel.render(step, notes);
}

function updateStatus() {
  const dps = state.pipelines;
  const connected = dps.filter((d) => d.connected);
  if (dps.length === 0) {
    statusEl.textContent = "ยังไม่มีกล่องผลลัพธ์";
    return;
  }
  if (connected.length === 0) {
    statusEl.textContent =
      "⚠ ยังไม่มีกล่องผลลัพธ์ใดต่อสายถึงแหล่งภาพ — ลากสายจากแหล่งภาพ ผ่าน filter ไปยังผลลัพธ์ให้ครบ";
    return;
  }
  // อธิบายทีละกล่องผลลัพธ์ (แยกให้เห็นว่าแต่ละกล่องแสดงอะไร)
  const parts = connected.map((d, i) => {
    const chain = d.pipeline.length
      ? d.pipeline.map((s) => s.id).join(" → ")
      : "(ภาพต้นฉบับ)";
    return `กล่อง ${i + 1}: ${chain}`;
  });
  const extra = dps.length - connected.length;
  statusEl.textContent =
    parts.join("   |   ") + (extra > 0 ? `   (อีก ${extra} กล่องยังไม่ต่อสาย)` : "");
}

// ---------- init ----------
async function main() {
  loadingText.textContent = "กำลังโหลด OpenCV.js (wasm ~10MB) ในเบื้องหลัง …";
  worker.postMessage({ type: "init" });
}

async function onOpenCVReady() {
  loadingText.textContent = "กำลังเปิดกล้อง …";
  palettePanel.render();
  renderDetail();
  showApp();
  editor.resize();
  requestAnimationFrame(pump);
  // เดินเส้นทางเดียวกับตอนกดปุ่ม "กล้อง" — เปิดกล้อง + จัดจอ Original ให้ติดทันที
  await setSource("camera");
}

// เทมเพลตสำเร็จรูป (Design §9)
const TEMPLATES = [
  { name: "ภาพขาวดำ → แยกส่วน", level: "Beginner", desc: "Grayscale แล้ว Threshold แยกวัตถุออกจากพื้นหลัง",
    objective: "เข้าใจว่า threshold ตัดสินขาว-ดำจากอะไร", minutes: 3,
    steps: [{ id: "color_space" }, { id: "segmentation", params: { mode: "binary_threshold" } }] },
  { name: "หาขอบภาพ (Canny)", level: "Beginner", desc: "Grayscale → เบลอตัด noise ก่อน → ตรวจจับขอบด้วย Canny (เบลอก่อนช่วยกันขอบปลอมจาก noise)",
    objective: "เข้าใจว่าทำไมต้องเบลอก่อนหาขอบ", minutes: 5,
    steps: [{ id: "color_space" }, { id: "restoration", params: { mode: "gaussian_blur" } }, { id: "segmentation", params: { mode: "canny" } }] },
  { name: "ลด Noise (Median Blur)", level: "Beginner", desc: "ลบเม็ด noise เกลือ-พริกไทยด้วย Median",
    objective: "เทียบ Median กับ Gaussian บน noise เม็ด", minutes: 4,
    steps: [{ id: "restoration", params: { mode: "median_blur" } }] },
  { name: "นับวัตถุ (Otsu → Contours)", level: "Intermediate", desc: "หาเกณฑ์อัตโนมัติแล้วหาเส้นรอบวัตถุเพื่อนับ",
    objective: "จากภาพสี ไปถึงตัวเลขจำนวนวัตถุ", minutes: 8,
    steps: [{ id: "color_space" }, { id: "segmentation", params: { mode: "otsu_threshold" } }, { id: "segmentation", params: { mode: "find_contours" } }] },
];
// เทมเพลตที่ผู้ใช้เซฟเอง (localStorage) — รวมกับเทมเพลตในตัว
function loadCustomTemplates() {
  try {
    const a = JSON.parse(localStorage.getItem("mv-templates") || "[]");
    return Array.isArray(a) ? a : [];
  } catch { return []; }
}
function saveCustomTemplates(list) {
  try { localStorage.setItem("mv-templates", JSON.stringify(list)); } catch { /* ignore */ }
}

function openTemplates() {
  // เก็บลิสต์ custom ชุดเดียวไว้ใน closure — ลบด้วย identity ของตัวเทมเพลต ไม่ใช้ index
  const customs = loadCustomTemplates().map((t) => ({ ...t, custom: true }));
  const all = [...TEMPLATES, ...customs];
  showTemplates(all, (steps) => {
    editor.applyTemplate(steps);
    statusEl.textContent = "📋 วางเทมเพลตแล้ว — ปรับค่าที่แผงขวาต่อได้เลย";
  }, { onDelete: (tmpl) => {
    // ลบได้เฉพาะเทมเพลตที่เซฟเอง — หาโดยตัวอ้างอิงจริง ไม่พึ่ง index (กันลบผิดตัว)
    const pos = customs.indexOf(tmpl);
    if (pos < 0) return false; // built-in หรือหาไม่เจอ → ลบไม่ได้
    customs.splice(pos, 1);
    saveCustomTemplates(customs.map(({ custom, ...t }) => t));
    return true;
  }});
}

// เซฟ pipeline ปัจจุบันเป็นเทมเพลตใหม่ (โผล่ปุ่มเมื่อมี filter บนกระดาน)
// ใช้ modal กรอกชื่อเอง — window.prompt ใช้ไม่ได้ใน Electron
function saveCurrentAsTemplate() {
  const steps = editor.exportTemplateSteps();
  if (!steps.length) {
    statusEl.textContent = "ยังไม่มี pipeline ให้เซฟ — เพิ่มเครื่องมือแล้วต่อสายถึงกล่องผลลัพธ์ก่อน";
    return;
  }
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  const modal = document.createElement("div");
  modal.className = "modal";
  overlay.appendChild(modal);
  const h = document.createElement("h2");
  h.className = "modal-title";
  h.textContent = "💾 เซฟเป็นเทมเพลต";
  modal.appendChild(h);
  const sub = document.createElement("div");
  sub.className = "modal-sub";
  sub.textContent = `บันทึก ${steps.length} ขั้นตอนบนกระดานเป็นเทมเพลต — เปิดใช้ซ้ำได้ที่ปุ่มเทมเพลต`;
  modal.appendChild(sub);
  const input = document.createElement("input");
  input.type = "text";
  input.className = "name-input";
  input.placeholder = "ชื่อเทมเพลต";
  input.value = "เทมเพลตของฉัน";
  modal.appendChild(input);
  const actions = document.createElement("div");
  actions.className = "modal-actions";
  const cancel = document.createElement("button");
  cancel.className = "btn";
  cancel.textContent = "ยกเลิก";
  cancel.setAttribute("data-close", "");
  const ok = document.createElement("button");
  ok.className = "btn btn-primary";
  ok.textContent = "เซฟ";
  const close = () => overlay.remove();
  const doSave = () => {
    const name = input.value.trim();
    if (!name) { input.focus(); return; }
    const cust = loadCustomTemplates();
    cust.push({ name, level: "Custom", desc: steps.length + " ขั้นตอน", custom: true, steps });
    saveCustomTemplates(cust);
    statusEl.textContent = `💾 เซฟเทมเพลต "${name}" แล้ว — เปิดได้ที่ปุ่มเทมเพลต`;
    close();
  };
  cancel.addEventListener("click", close);
  ok.addEventListener("click", doSave);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") doSave(); });
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  actions.appendChild(cancel);
  actions.appendChild(ok);
  modal.appendChild(actions);
  document.body.appendChild(overlay);
  input.focus();
  input.select();
}

async function ensureCamera() {
  if (cameraStarted) return true;
  try {
    const stream = await startCamera(videoEl, state.cameraId);
    if (!stream) return true; // ถูกแซงด้วยการเปิดกล้องใหม่ — รุ่นล่าสุดจัดการเอง
    cameraStarted = true;
    cameraError.classList.add("hidden");
    gridEl.classList.remove("hidden");
    // จดจำ deviceId ที่ใช้จริง + โหลดรายชื่อกล้อง (label มีค่าหลังได้สิทธิ์แล้ว)
    const track = stream.getVideoTracks()[0];
    if (track && !state.cameraId) state.cameraId = track.getSettings().deviceId || null;
    await refreshCameras();
    return true;
  } catch (err) {
    showCameraError(err.message);
    return false;
  }
}

// โหลดรายชื่อกล้องที่ต่ออยู่ แล้วอัปเดตแถบเลือก
async function refreshCameras() {
  state.cameras = await listCameras();
  if (!state.cameraId && state.cameras[0]) state.cameraId = state.cameras[0].deviceId;
  sourceBar.render();
}

// สลับไปใช้กล้องตัวที่เลือก (Port อื่น) — สตรีมใหม่ใช้ videoEl เดิม (โหมด Pose ก็ยังทำงานต่อ)
async function selectCamera(id) {
  // เดิม: ถ้า id ตรงกับตัวที่เลือกอยู่แล้วจะ return เงียบ ๆ ทันที —
  // แต่ถ้ารอบก่อนเปิดกล้องนี้ "ล้มเหลว" (cameraStarted=false) ผู้ใช้กด port เดิมซ้ำเพื่อ retry
  // จะโดนเงียบใส่ ไม่ retry ไม่โชว์ error อะไรเลย (บั๊ค: กดแล้วไม่มีอะไรเกิดขึ้น)
  if (id === state.cameraId && cameraStarted) return;
  state.cameraId = id;
  try {
    const stream = await startCamera(videoEl, id);
    if (!stream) return; // ถูกแซงด้วยการสลับกล้องใหม่ — รุ่นล่าสุดจัดการเอง
    cameraStarted = true;
    cameraError.classList.add("hidden");
    // สำคัญ: ถ้า port ก่อนหน้าล้มเหลวมา showCameraError() จะซ่อน gridEl (Live View) ไว้ —
    // ต้องเปิดกลับตอนสลับ port แล้วสำเร็จ ไม่งั้นจอหลักจะดำค้างทั้งที่กล้องทำงานจริง (FPS/thumbnail ขึ้นปกติ)
    gridEl.classList.remove("hidden");
    updateStatus(); // ล้างข้อความ "✕ กล้องไม่พร้อมใช้งาน" ที่ค้างมาจาก error รอบก่อน
    sourceBar.render();
  } catch (err) {
    showCameraError(err.message);
  }
}

// เสียบ/ถอดกล้องสด → อัปเดตรายชื่อ
if (navigator.mediaDevices) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    if (cameraStarted) refreshCameras();
  });
}

// ---------- source switching ----------
async function setSource(s) {
  if (s !== "video" && videoURL) closeVideoSource(); // ออกจากโหมดวิดีโอ
  state.source = s;
  state.paused = false; // paused มีอายุเฉพาะ session ปัจจุบัน — สลับ source แล้วกลับมารันสด
  pendingProcess = false; // กัน pendingProcess ค้างข้ามโหมด (เช่น กล้อง→ภาพนิ่ง) ประมวลผลเกิน
  invalidateSelectedPreview();
  renderSelectedPreviewState(); // สลับ source → ถ้า preview ใช้ไม่ได้ ให้เคลียร์/โชว์ข้อความ ไม่ค้างภาพเก่า
  sourceBar.render();

  if (s === "camera") {
    cameraError.classList.add("hidden");
    imageEmptyEl.classList.add("hidden");
    const ok = await ensureCamera();
    if (!ok) return;
    applyCameraView(); // แสดงตามโหมดย่อย (original/pose)
  } else {
    // image: หยุด pose ถ้าเปิดอยู่
    stopPose();
    poseCanvas.classList.add("hidden");
    gridEl.classList.remove("hidden");
    cameraError.classList.add("hidden");
    updateImageEmpty();
    if (state.images.length === 0) {
      statusEl.textContent = "ยังไม่มีภาพ — เลือกนำเข้าภาพเดี่ยว หรือทั้งโฟลเดอร์";
    } else {
      updateStatus();
      requestProcess();
    }
  }
}

// สลับโหมดย่อยของกล้อง: Original (ผ่าน node-graph) / Ergonomics Pose
function setCameraMode(m) {
  state.cameraMode = m;
  state.paused = false; // สลับโหมดย่อย → เริ่มรันสดเสมอ (กัน paused ค้างขัดกับสถานะจริง)
  invalidateSelectedPreview();
  renderSelectedPreviewState(); // เข้าโหมด Pose (ไม่มี preview) → เคลียร์ ไม่ค้างภาพเก่า
  sourceBar.render();
  if (state.source === "camera") applyCameraView();
}

// จัดหน้าจอกล้องตามโหมดย่อยปัจจุบัน
function applyCameraView() {
  if (state.cameraMode === "pose") {
    poseCanvas.classList.remove("hidden");
    gridEl.classList.add("hidden");
    startPose(
      videoEl,
      poseCanvas,
      (msg) => (statusEl.textContent = msg),
      (ergo) => {
        if (!ergo) return;
        if (ergo.level >= 2) statusEl.textContent = "⚠ ท่าเสี่ยง: " + ergo.warnings.join(" · ");
        else if (ergo.level === 1) statusEl.textContent = "🟡 ท่าทางพอใช้ — ปรับให้ตรงขึ้นได้อีก";
        else statusEl.textContent = "🟢 ท่าทางดี (ตามหลักการยศาสตร์)";
      }
    ).catch((err) => (statusEl.textContent = "เปิด Pose ไม่สำเร็จ: " + err.message));
  } else {
    stopPose();
    poseCanvas.classList.add("hidden");
    gridEl.classList.remove("hidden");
    updateStatus();
  }
}


// ---------- คีย์ลัด (keyboard shortcut) ----------
// รายการ action ที่ผูกคีย์ได้ + ตัวจัดการ
const KEY_ACTIONS = [
  { id: "capture", label: "📸 แคปภาพ", hint: "เก็บเฟรมกล้องเข้าคลัง (โหมดกล้อง)", run: () => captureFrame() },
  { id: "toggleMode", label: "สลับ ต้นฉบับ ↔ Ergonomics Pose", hint: "เฉพาะโหมดกล้อง", run: () => { if (state.source === "camera") setCameraMode(state.cameraMode === "pose" ? "original" : "pose"); } },
  { id: "toggleSource", label: "สลับ กล้อง ↔ ภาพนิ่ง", hint: "", run: () => setSource(state.source === "camera" ? "image" : "camera") },
  { id: "prevImage", label: "◀ ภาพก่อนหน้า", hint: "โหมดชุดภาพ (batch)", run: () => { if (state.images.length > 1) setImageIndex(state.imageIndex - 1); } },
  { id: "nextImage", label: "▶ ภาพถัดไป", hint: "โหมดชุดภาพ (batch)", run: () => { if (state.images.length > 1) setImageIndex(state.imageIndex + 1); } },
  { id: "export", label: "⬇ โค้ด YOLO", hint: "เปิดหน้าต่างโค้ดเทรน YOLO", run: () => openExportCode() },
  { id: "saveImages", label: "💾 เซฟภาพ", hint: "บันทึกภาพผลลัพธ์เป็น PNG", run: () => saveImages() },
];
const DEFAULT_KEYMAP = { capture: " ", toggleMode: "m", toggleSource: "c", prevImage: "ArrowLeft", nextImage: "ArrowRight", export: "e" };

function loadKeymap() {
  try {
    const saved = JSON.parse(localStorage.getItem("mv-keymap") || "null");
    return saved && typeof saved === "object" ? { ...DEFAULT_KEYMAP, ...saved } : { ...DEFAULT_KEYMAP };
  } catch {
    return { ...DEFAULT_KEYMAP };
  }
}
state.keymap = loadKeymap();

function saveKeymap(km) {
  state.keymap = km;
  try { localStorage.setItem("mv-keymap", JSON.stringify(km)); } catch { /* ignore */ }
}

// global keydown → เรียก action ตามคีย์ที่ผูกไว้ (ข้ามเมื่อกำลังพิมพ์ในช่อง input)
document.addEventListener("keydown", (e) => {
  // Ctrl+/ (หรือ Cmd+/) → โฟกัสช่องค้นหาเครื่องมือ (§11, UI-only ไม่ยุ่ง keymap) — ทำงานแม้กำลังพิมพ์อยู่ในช่องอื่น
  if ((e.ctrlKey || e.metaKey) && e.key === "/") {
    e.preventDefault();
    palettePanel.focusSearch();
    return;
  }
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
  // โฟกัสอยู่บนปุ่มแล้วกด Space/Enter → ปล่อยให้ปุ่มนั้นทำงานเนทีฟ (กันยิง action ซ้อน)
  if (t && t.tagName === "BUTTON" && (e.key === " " || e.key === "Enter")) return;
  // มี dialog เปิดอยู่ → ไม่จับคีย์ลัด แต่รองรับ ESC ปิด modal (Design §22)
  const overlays = document.querySelectorAll(".modal-overlay");
  if (overlays.length) {
    if (e.key === "Escape") {
      const top = overlays[overlays.length - 1];
      // กดเฉพาะปุ่มปิด/ยกเลิกที่ติดป้าย data-close ชัดเจน (ห้ามเดา — กันไปโดนปุ่มยืนยันที่มี side-effect)
      const closeBtn = top.querySelector("[data-close]");
      if (closeBtn) { e.preventDefault(); closeBtn.click(); }
    }
    return;
  }
  // Undo / Redo (§6)
  if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) {
    e.preventDefault();
    if (e.shiftKey) editor.redo(); else editor.undo();
    return;
  }
  if ((e.ctrlKey || e.metaKey) && (e.key === "y" || e.key === "Y")) {
    e.preventDefault();
    editor.redo();
    return;
  }
  if (e.ctrlKey || e.altKey || e.metaKey) return; // ปล่อยคีย์ลัดระบบผ่าน
  const key = e.key;
  for (const a of KEY_ACTIONS) {
    const bound = state.keymap[a.id];
    if (bound && bound.toLowerCase() === key.toLowerCase()) {
      e.preventDefault();
      a.run();
      return;
    }
  }
});

function openSettingsDialog() {
  openSettings(
    KEY_ACTIONS.map((a) => ({ id: a.id, label: a.label, hint: a.hint })),
    { ...state.keymap },
    (km) => saveKeymap(km)
  );
}

// empty state ของโหมดภาพนิ่ง: โชว์เมื่อ source=image และยังไม่มีภาพ (ไม่เด้ง dialog บังคับ)
function updateImageEmpty() {
  const show = state.source === "image" && state.images.length === 0;
  imageEmptyEl.classList.toggle("hidden", !show);
  gridEl.classList.toggle("hidden", show);
}
document.getElementById("ie-single").addEventListener("click", () => fileInput.click());
document.getElementById("ie-folder").addEventListener("click", () => folderInput.click());

// ---- Live View / Canvas controls (Design §5/6) ----
const stageEl = document.querySelector(".stage");
document.getElementById("lv-capture").addEventListener("click", () => captureFrame());
document.getElementById("lv-original").addEventListener("click", () => setCameraMode("original"));
document.getElementById("lv-pose").addEventListener("click", () => setCameraMode("pose"));
document.getElementById("lv-toggle").addEventListener("click", () => togglePause());
const fullscreenBtn = document.getElementById("lv-fullscreen");
function setStageFullscreen(on) {
  stageEl.classList.toggle("stage-fullscreen", on);
  document.body.classList.toggle("stage-fullscreen-open", on);
  fullscreenBtn.textContent = on ? "✕" : "⛶";
  fullscreenBtn.title = on ? "ออกจากเต็มจอ (Esc)" : "เต็มจอ";
}
fullscreenBtn.addEventListener("click", () => setStageFullscreen(!stageEl.classList.contains("stage-fullscreen")));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && stageEl.classList.contains("stage-fullscreen")) setStageFullscreen(false);
});
document.getElementById("cv-zoom-in").addEventListener("click", () => editor.zoomIn());
document.getElementById("cv-zoom-out").addEventListener("click", () => editor.zoomOut());
document.getElementById("cv-fit").addEventListener("click", () => editor.resetView());
document.getElementById("cv-delete").addEventListener("click", () => editor.deleteSelected());
document.getElementById("cv-undo").addEventListener("click", () => editor.undo());
document.getElementById("cv-redo").addEventListener("click", () => editor.redo());

// อัปเดตป้าย Live View (โหมด) — เบา ๆ ทุก 1 วิ
const lvModeEl = document.getElementById("lv-mode");
const lvMemEl = document.getElementById("lv-mem");
// การ์ด "ข้อมูลภาพ" มุมซ้ายบน (Design node-card §3) — แก้เฉพาะ text node กัน reflow ทุกวิ
const liInfoEl = document.getElementById("live-info");
const liSourceEl = document.getElementById("li-source");
const liResEl = document.getElementById("li-res");
const liFpsRowEl = document.getElementById("li-fps-row");
const liFpsEl = document.getElementById("li-fps");
const liColorEl = document.getElementById("li-color");
setInterval(() => {
  const mode = state.paused ? "หยุด"
    : state.source === "camera"
    ? (state.cameraMode === "pose" ? "Pose" : "สด")
    : state.source === "video" ? "วิดีโอ" : "ภาพนิ่ง";
  lvModeEl.textContent = mode;
  const w = videoEl.videoWidth, h = videoEl.videoHeight;
  // FPS จริง (เฟรมที่ประมวลผลใน 1 วิ) + RAM (JS heap)
  const fps = state.paused ? 0 : procFrames;
  procFrames = 0;
  const mem = performance.memory;
  lvMemEl.textContent = mem ? `${Math.round(mem.usedJSHeapSize / 1048576)} MB` : "";
  // ปุ่มแคปโชว์เฉพาะโหมดกล้อง (captureFrame ทำงานเฉพาะกล้อง)
  document.getElementById("lv-capture").style.display = state.source === "camera" ? "" : "none";
  const cameraControlsVisible = state.source === "camera";
  const originalBtn = document.getElementById("lv-original");
  const poseBtn = document.getElementById("lv-pose");
  const toggleBtn = document.getElementById("lv-toggle");
  originalBtn.style.display = cameraControlsVisible ? "" : "none";
  poseBtn.style.display = cameraControlsVisible ? "" : "none";
  toggleBtn.style.display = cameraControlsVisible ? "" : "none";
  originalBtn.classList.toggle("active", state.cameraMode === "original");
  poseBtn.classList.toggle("active", state.cameraMode === "pose");
  toggleBtn.textContent = state.paused ? "▶" : "⏸";
  toggleBtn.title = state.paused ? "เริ่มประมวลผลต่อ" : "หยุดการประมวลผลชั่วคราว";
  updatePerf(fps);

  // การ์ดข้อมูลภาพ: ไม่มีแหล่งภาพเลย → ซ่อนทั้งใบ
  const isCamera = state.source === "camera";
  const isVideo = state.source === "video";
  const hasStillImage = state.source === "image" && state.images.length > 0;
  const hasSource = (isCamera && cameraStarted) || isVideo || hasStillImage;
  liInfoEl.classList.toggle("hidden", !hasSource);
  if (hasSource) {
    liSourceEl.textContent = isCamera ? "กล้อง" : isVideo ? "วิดีโอ" : "ไฟล์ภาพ";
    if (isCamera && w) liResEl.textContent = `${w}×${h}`;
    else if (hasStillImage) {
      const im = state.images[state.imageIndex];
      liResEl.textContent = im ? `${im.w}×${im.h}` : "—";
    } else liResEl.textContent = "—";
    // FPS มีความหมายเฉพาะภาพสด (กล้อง/วิดีโอ) — ภาพนิ่งไม่มีเฟรมต่อวินาทีให้ดู
    liFpsRowEl.classList.toggle("hidden", !isCamera);
    if (isCamera) liFpsEl.textContent = state.paused ? "0" : String(fps);
    liColorEl.textContent = "สี (RGB)";
  }
  renderCameraControl();
}, 1000);

// ---------- Performance panel (Design §17) ----------
const statusFpsEl = document.getElementById("status-fps");
const statusCpuEl = document.getElementById("status-cpu");
const statusRamEl = document.getElementById("status-ram");
const statusGpuEl = document.getElementById("status-gpu");
function updatePerf(fps) {
  if (statusFpsEl) statusFpsEl.textContent = "FPS " + fps;
  const mem = performance.memory;
  // ถ้าอยู่ในแอป Electron → ดึง CPU/RAM จริงจาก main process; ถ้าเปิดในเบราว์เซอร์ → JS heap
  if (window.mvMetrics && window.mvMetrics.available) {
    window.mvMetrics.get().then((m) => {
      if (!m) return;
      if (statusCpuEl) statusCpuEl.textContent = `CPU ${Math.round(m.cpu)}%`;
      if (statusRamEl) statusRamEl.textContent = `RAM ${(m.memMB / 1024).toFixed(1)} GB`;
    }).catch(() => {});
    if (statusGpuEl) statusGpuEl.textContent = "GPU —"; // GPU usage ไม่พร้อมใช้ข้ามแพลตฟอร์ม — ไม่แสดงค่าปลอม
  } else {
    if (statusCpuEl) statusCpuEl.textContent = "CPU —";
    if (statusRamEl) statusRamEl.textContent = mem ? `RAM ${Math.round(mem.usedJSHeapSize / 1048576)} MB` : "RAM —";
    if (statusGpuEl) statusGpuEl.textContent = "GPU —";
  }
}

// ---------- Logs panel (Design §18) — เก็บสถานะการทำงานอัตโนมัติ ----------
const logsList = document.getElementById("logs-list");
let lastLog = "";
function logEvent(text) {
  const t = (text || "").trim();
  if (!t || t === lastLog) return; // กันซ้ำติดกัน
  lastLog = t;
  const row = document.createElement("div");
  row.className = "log-row";
  const time = new Date().toLocaleTimeString("th-TH", { hour12: false });
  row.innerHTML = `<span class="log-time">${time}</span><span class="log-msg"></span>`;
  row.querySelector(".log-msg").textContent = t;
  logsList.prepend(row);
  while (logsList.children.length > 100) logsList.lastChild.remove();
  // auto-expand เมื่อเจอข้อความผิดพลาด (Design §27) — นักเรียนจะได้เห็นเหตุผลเอง
  if (/ผิดพลาด|ไม่สามารถ|ล้มเหลว|ไม่พร้อม|^✕|^⚠/.test(t)) {
    const sec = document.getElementById("logs-section");
    if (sec) sec.open = true;
  }
}
// ดักการเปลี่ยนข้อความ statusbar อัตโนมัติ → ลง log โดยไม่ต้องแก้ทุกจุดที่ตั้ง status
new MutationObserver(() => logEvent(statusEl.textContent)).observe(statusEl, {
  childList: true, characterData: true, subtree: true,
});
document.getElementById("logs-clear").addEventListener("click", (e) => {
  e.preventDefault(); e.stopPropagation();
  logsList.innerHTML = ""; lastLog = "";
});

// ---------- คลังภาพที่แคปไว้ (capture tray) ----------
const trayEl = document.getElementById("capture-tray");

// แคปเฟรมกล้องปัจจุบัน → เก็บเข้าคลัง (ภาพดิบจากกล้อง เพื่อเอาไปวิเคราะห์/เซฟได้)
function captureFrame() {
  if (state.source !== "camera") {
    statusEl.textContent = "แคปภาพได้เฉพาะโหมดกล้อง";
    return;
  }
  const w = videoEl.videoWidth, h = videoEl.videoHeight;
  if (!w || !h) {
    statusEl.textContent = "กล้องยังไม่พร้อม — รอสักครู่แล้วลองใหม่";
    return;
  }
  const cv = document.createElement("canvas");
  cv.width = w;
  cv.height = h;
  cv.getContext("2d").drawImage(videoEl, 0, 0, w, h);
  const n = state.captures.length + 1;
  const ts = new Date();
  const pad = (x) => String(x).padStart(2, "0");
  const name = `capture_${ts.getFullYear()}${pad(ts.getMonth() + 1)}${pad(ts.getDate())}_${pad(ts.getHours())}${pad(ts.getMinutes())}${pad(ts.getSeconds())}`;
  state.captures.push({ canvas: cv, w, h, name });
  statusEl.textContent = `📸 แคปแล้ว — เก็บไว้ในคลัง (ทั้งหมด ${n} ภาพ)`;
  renderTray();
}

// Run / Pause การประมวลผลสด (ตรงเจตนา "Run Pipeline" แบบเข้ากับสถาปัตยกรรม live)
function togglePause() {
  state.paused = !state.paused;
  if (state.source === "camera" && state.cameraMode === "pose") {
    if (state.paused) pausePose(); else resumePose();
  }
  sourceBar.render();
  statusEl.textContent = state.paused
    ? "⏸ หยุดประมวลผลชั่วคราว — กด ▶ รัน เพื่อดำเนินการต่อ"
    : "▶ กำลังประมวลผลสด";
}

function selectCapture(i) {
  state.selectedCapture = i;
  renderTray();
}
function deleteCapture(i) {
  state.captures.splice(i, 1);
  // ปรับ index ที่เลือกให้ยังชี้ภาพเดิม: ลบตัวที่เลือก → ยกเลิกเลือก, ลบตัวก่อนหน้า → เลื่อนลง
  if (i === state.selectedCapture) state.selectedCapture = -1;
  else if (i < state.selectedCapture) state.selectedCapture--;
  if (state.selectedCapture >= state.captures.length) state.selectedCapture = state.captures.length - 1;
  renderTray();
}

// เซฟภาพแคปทั้งหมด — เลือกโฟลเดอร์ครั้งเดียว (Electron) หรือ fallback ดาวน์โหลด
async function saveCaptures() {
  if (state.captures.length === 0) return;
  let dir = null;
  if (canExportFolder) {
    dir = await window.mvExport.chooseDir();
    if (!dir) {
      statusEl.textContent = "ยกเลิกการบันทึกภาพแคป";
      return;
    }
  }
  let n = 0;
  for (const cap of state.captures) {
    const blob = await canvasToBlob(cap.canvas);
    if (await saveOne(dir, `${cap.name}.png`, blob)) n++;
  }
  statusEl.textContent = `บันทึก ${n} ภาพแคป${dir ? " ลงโฟลเดอร์ที่เลือกแล้ว" : "แล้ว"}`;
}

// ใช้ภาพในคลังแคปเป็น "แหล่งภาพนิ่ง" เข้ากระดาน — ต่อ filter/Export ได้เหมือนภาพนำเข้า
function useCapturesAsSource() {
  if (state.captures.length === 0) return;
  state.images = state.captures.map((c) => ({ canvas: c.canvas, w: c.w, h: c.h, name: c.name + ".png" }));
  decodeCache.clear();
  state.imageIndex = 0;
  setSource("image");
  setImageIndex(0);
  statusEl.textContent = `▶ ใช้ภาพแคป ${state.images.length} ภาพเป็นแหล่งภาพแล้ว — ต่อ filter ได้เลย (เลื่อน Next/Prev ดูทีละภาพ)`;
}

// เปิดหน้าต่างดู Ergonomics จากภาพแคปที่เลือก
async function openErgoFromCapture() {
  const i = state.selectedCapture;
  const cap = state.captures[i];
  if (!cap) {
    statusEl.textContent = "เลือกภาพในคลังก่อน แล้วกด 'ดู Ergonomics'";
    return;
  }
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  const modal = document.createElement("div");
  modal.className = "modal";
  overlay.appendChild(modal);
  const h = document.createElement("h2");
  h.className = "modal-title";
  h.textContent = "🧍 Ergonomics — " + cap.name;
  modal.appendChild(h);
  const status = document.createElement("div");
  status.className = "ergo-status";
  status.textContent = "กำลังวิเคราะห์ท่าทาง …";
  modal.appendChild(status);
  const out = document.createElement("canvas");
  out.className = "ergo-view";
  modal.appendChild(out);
  const actions = document.createElement("div");
  actions.className = "modal-actions";
  const closeBtn = document.createElement("button");
  closeBtn.className = "btn btn-primary";
  closeBtn.textContent = "ปิด";
  closeBtn.setAttribute("data-close", ""); // ESC ปิดได้ (Design §22)
  let closed = false;
  const close = () => { closed = true; overlay.remove(); };
  closeBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  actions.appendChild(closeBtn);
  modal.appendChild(actions);
  document.body.appendChild(overlay);

  try {
    const ergo = await renderStillErgonomics(cap.canvas, out, (m) => { if (!closed) status.textContent = m; });
    if (closed) return; // ปิดหน้าต่างระหว่างวิเคราะห์ → ไม่ต้องเซ็ตผลบน element ที่ถูกลบ
    if (!ergo) {
      status.textContent = "⚠ ตรวจไม่พบร่างกายในภาพนี้ — ลองแคปภาพที่เห็นลำตัวชัด ๆ";
    } else if (ergo.level >= 2) {
      status.textContent = "⚠ ท่าเสี่ยง: " + ergo.warnings.join(" · ");
      status.style.color = "#fca5a5";
    } else if (ergo.level === 1) {
      status.textContent = "🟡 ท่าทางพอใช้ — ปรับให้ตรงขึ้นได้อีก";
      status.style.color = "#fcd34d";
    } else {
      status.textContent = "🟢 ท่าทางดี (ตามหลักการยศาสตร์)";
      status.style.color = "#86efac";
    }
  } catch (err) {
    if (!closed) status.textContent = "วิเคราะห์ไม่สำเร็จ: " + err.message;
  }
}

// วาดถาดภาพแคป
function renderTray() {
  if (state.captures.length === 0) {
    trayEl.classList.add("hidden");
    trayEl.innerHTML = "";
    return;
  }
  trayEl.classList.remove("hidden");
  trayEl.innerHTML = "";

  const head = document.createElement("div");
  head.className = "tray-head";
  const title = document.createElement("div");
  title.className = "tray-title";
  title.textContent = `🖼 คลังภาพแคป (${state.captures.length})`;
  head.appendChild(title);
  const acts = document.createElement("div");
  acts.className = "tray-actions";
  const useBtn = document.createElement("button");
  useBtn.className = "btn btn-primary";
  useBtn.textContent = "▶ ใช้ในกระดาน";
  useBtn.title = "ใช้ภาพในคลังเป็นแหล่งภาพนิ่ง — ต่อ filter / Export ได้เหมือนภาพนำเข้า (เลื่อนดูทีละภาพ)";
  useBtn.addEventListener("click", () => useCapturesAsSource());
  acts.appendChild(useBtn);
  const ergoBtn = document.createElement("button");
  ergoBtn.className = "btn";
  ergoBtn.textContent = "🧍 ดู Ergonomics";
  ergoBtn.title = "วิเคราะห์การยศาสตร์จากภาพที่เลือกในคลัง";
  ergoBtn.disabled = state.selectedCapture < 0;
  ergoBtn.addEventListener("click", () => openErgoFromCapture());
  const saveBtn = document.createElement("button");
  saveBtn.className = "btn";
  saveBtn.textContent = "💾 เซฟทั้งหมด";
  saveBtn.title = "เลือกโฟลเดอร์แล้วบันทึกภาพแคปทั้งหมดเป็น PNG";
  saveBtn.addEventListener("click", () => saveCaptures());
  const clearBtn = document.createElement("button");
  clearBtn.className = "btn";
  clearBtn.textContent = "ล้างคลัง";
  clearBtn.addEventListener("click", () => { state.captures = []; state.selectedCapture = -1; renderTray(); });
  acts.appendChild(ergoBtn);
  acts.appendChild(saveBtn);
  acts.appendChild(clearBtn);
  head.appendChild(acts);
  trayEl.appendChild(head);

  const strip = document.createElement("div");
  strip.className = "tray-strip";
  state.captures.forEach((cap, i) => {
    const item = document.createElement("div");
    item.className = "tray-item" + (i === state.selectedCapture ? " selected" : "");
    item.title = cap.name + " — คลิกเพื่อเลือก";
    const img = document.createElement("img");
    img.src = cap.canvas.toDataURL("image/png");
    item.appendChild(img);
    const idx = document.createElement("span");
    idx.className = "tray-idx";
    idx.textContent = String(i + 1);
    item.appendChild(idx);
    const del = document.createElement("button");
    del.className = "tray-del";
    del.textContent = "✕";
    del.title = "ลบภาพนี้";
    del.addEventListener("click", (e) => { e.stopPropagation(); deleteCapture(i); });
    item.appendChild(del);
    item.addEventListener("click", () => selectCapture(i));
    strip.appendChild(item);
  });
  trayEl.appendChild(strip);
}

// ---------- Deep Learning: YOLO detect ในกล่องผลลัพธ์ที่ต่อสายผ่านกล่อง DL ----------
let detBusy = false;
let detBoxes = [];
function connectedDLDisplays() {
  return state.pipelines.filter((d) => d.connected && d.dl);
}
// วาดภาพ + กล่อง detect ลงจอผลลัพธ์ (grid cell) + thumbnail ใน node
function drawDLCells(dls, src, w, h) {
  const scale = Math.max(1, w / 640);
  for (const d of dls) {
    const oc = displayCanvasFor(d.displayId);
    if (oc.width !== w) { oc.width = w; oc.height = h; }
    const octx = oc.getContext("2d");
    octx.drawImage(src, 0, 0, w, h);
    dlDraw(octx, detBoxes, scale);
    editor.setDisplayImage(d.displayId, oc);
    const cell = cells.get(d.displayId);
    if (cell) {
      if (cell.canvas.width !== w) { cell.canvas.width = w; cell.canvas.height = h; }
      cell.ctx.drawImage(oc, 0, 0);
    }
  }
}
// เรียกทุกเฟรม (กล้อง) หรือครั้งเดียว (ภาพนิ่ง) — src = video/canvas
function renderDLDisplays(src, w, h) {
  const dls = connectedDLDisplays();
  if (dls.length === 0) return;
  if (!dlLoaded()) {
    // ยังไม่โหลดโมเดล → โชว์ภาพเฉย ๆ พร้อมข้อความ
    drawDLCells(dls, src, w, h);
    return;
  }
  drawDLCells(dls, src, w, h);
  if (!detBusy) {
    detBusy = true;
    dlDetect(src, w, h, { labels: state.labels })
      .then((bs) => {
        detBoxes = bs;
        drawDLCells(connectedDLDisplays(), src, w, h); // วาดกล่องใหม่ (สำคัญสำหรับภาพนิ่ง)
      })
      .catch(() => {})
      .finally(() => { detBusy = false; });
  }
}

// เลือกไฟล์โมเดล .onnx → โหลด → เข้าโหมด detect ทันที
modelInput.addEventListener("change", async () => {
  const file = modelInput.files?.[0];
  modelInput.value = "";
  if (!file) return;
  try {
    let bytes;
    if (/\.pt$/i.test(file.name)) {
      // .pt รันตรง ๆ ไม่ได้ — แปลงเป็น .onnx ด้วย ultralytics ในเครื่องก่อน
      if (!window.mvDL?.available) {
        statusEl.textContent = "แปลง .pt ได้เฉพาะในแอป (Electron) เท่านั้น";
        return;
      }
      const ptPath = file.path;
      if (!ptPath) {
        statusEl.textContent = "อ่านที่อยู่ไฟล์ .pt ไม่ได้";
        return;
      }
      statusEl.textContent = `กำลังแปลง ${file.name} → .onnx (ใช้ ultralytics ในเครื่อง) …`;
      const res = await window.mvDL.convertPt(ptPath);
      if (!res.ok) {
        statusEl.textContent =
          "แปลง .pt ไม่สำเร็จ: " + res.error + " — ต้องมี Python + ultralytics ในเครื่อง (pip install ultralytics)";
        return;
      }
      bytes = res.bytes instanceof Uint8Array ? res.bytes : new Uint8Array(res.bytes);
    } else {
      bytes = new Uint8Array(await file.arrayBuffer());
    }
    statusEl.textContent = `กำลังโหลดโมเดล ${file.name} … (ครั้งแรกอาจใช้เวลาสักครู่)`;
    const { inputSize, names } = await dlLoadModel(bytes);
    state.labels = names && names.length ? names : COCO; // ชื่อคลาสจากตัวโมเดล ไม่งั้น COCO
    const hasDL = connectedDLDisplays().length > 0;
    statusEl.textContent = hasDL
      ? `โมเดลพร้อม (${file.name}, ${state.labels.length} คลาส) — ตรวจจับในกล่องผลลัพธ์ที่ต่อสายผ่านกล่อง DL`
      : `โมเดลพร้อม (${file.name}, ${state.labels.length} คลาส) — ต่อสาย แหล่งภาพ → กล่อง YOLO → ผลลัพธ์ เพื่อดู detection`;
    if (state.source === "image") requestProcess();
    if (state.selectedDL) renderDetail();
  } catch (err) {
    statusEl.textContent = "โหลดโมเดลไม่สำเร็จ: " + err.message;
  }
});

// เลือกไฟล์ labels → ตั้งชื่อคลาส
labelsInput.addEventListener("change", async () => {
  const file = labelsInput.files?.[0];
  labelsInput.value = "";
  if (!file) return;
  const text = await file.text();
  const labels = parseLabels(text, file.name);
  if (labels.length) {
    state.labels = labels;
    statusEl.textContent = `ตั้งชื่อคลาสแล้ว ${labels.length} คลาส (จาก ${file.name})`;
  } else {
    statusEl.textContent = "อ่านชื่อคลาสไม่ได้ — ใช้ COCO ต่อไป";
  }
  if (state.selectedDL) renderDetail();
});

// แยกชื่อคลาสจากไฟล์ classes.txt/.names (บรรทัดละคลาส) หรือ data.yaml (names:)
function parseLabels(text, name) {
  if (/\.ya?ml$/i.test(name)) {
    const idx = text.indexOf("names:");
    if (idx < 0) return [];
    const after = text.slice(idx + 6);
    const inline = after.match(/\[([^\]]*)\]/);
    if (inline) {
      return inline[1].split(",").map((s) => s.trim().replace(/^['"]|['"]$/g, "")).filter(Boolean);
    }
    const out = [];
    for (const ln of after.split(/\r?\n/).slice(1)) {
      if (/^\s*$/.test(ln)) continue;
      if (/^\S/.test(ln)) break; // ถึง key ถัดไป (ไม่เยื้อง)
      const m = ln.match(/^\s*(?:-\s*|\d+\s*:\s*)?['"]?(.+?)['"]?\s*$/);
      if (m && m[1]) out.push(m[1].trim());
    }
    return out;
  }
  return text.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
}

// ---------- image import (เดี่ยว/หลายไฟล์/โฟลเดอร์) ----------
// โหลดไฟล์ภาพหนึ่งไฟล์ → { canvas, w, h, name } (ย่อต้นฉบับกันหน่วยความจำ)
function loadImageFile(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const CAP = 1920;
      let w = img.naturalWidth;
      let h = img.naturalHeight;
      const scale = Math.min(1, CAP / Math.max(w, h));
      w = Math.round(w * scale);
      h = Math.round(h * scale);
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      canvas.getContext("2d").drawImage(img, 0, 0, w, h);
      URL.revokeObjectURL(url);
      resolve({ canvas, w, h, name: file.name });
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(null); // ข้ามไฟล์ที่โหลดไม่ได้
    };
    img.src = url;
  });
}

function importFiles(fileList) {
  const files = Array.from(fileList || []).filter((f) => f.type.startsWith("image/"));
  if (files.length === 0) {
    statusEl.textContent = "ไม่พบไฟล์ภาพในสิ่งที่เลือก";
    return;
  }
  files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  // เก็บแค่ไฟล์ (lazy decode) — ไม่ถอดทั้งหมดพร้อมกัน
  state.images = files.map((f) => ({ file: f, name: f.name }));
  decodeCache.clear();
  state.imageIndex = 0;
  state.source = "image";
  gridEl.classList.remove("hidden");
  cameraError.classList.add("hidden");
  updateImageEmpty();
  sourceBar.render();
  statusEl.textContent =
    files.length > 1
      ? `นำเข้า ${files.length} ภาพแล้ว — แสดงภาพ 1/${files.length} (กด ▶ ดูภาพถัดไป · Export = ประมวลผลทุกภาพ)`
      : `นำเข้าภาพแล้ว — ${files[0].name}`;
  requestProcess();
}

// นำเข้าภาพ/วิดีโอ (ปุ่มเดียว): ถ้าเลือกวิดีโอ → เด้ง popup แยกเฟรม, ถ้าเป็นภาพ → นำเข้าปกติ
const isVideoFile = (f) => f.type.startsWith("video/") || /\.(mp4|mov|avi|mkv|webm|m4v|ts)$/i.test(f.name);
fileInput.addEventListener("change", () => {
  const files = Array.from(fileInput.files || []);
  fileInput.value = "";
  const videos = files.filter(isVideoFile);
  const images = files.filter((f) => !isVideoFile(f));
  if (videos.length > 0) {
    // นำเข้าวิดีโอ = เล่นในแหล่งภาพก่อน (แยกเฟรมทีหลังด้วยปุ่มในจอ)
    let msg = `เล่นวิดีโอ: ${videos[0].name} — กด “🎬 แยกเฟรม” เมื่อต้องการ`;
    if (videos.length > 1) msg += ` (เลือกวิดีโอ ${videos.length} ไฟล์ — รองรับทีละไฟล์)`;
    if (images.length > 0) msg += ` · ข้ามภาพ ${images.length} ภาพที่เลือกมาด้วย`;
    statusEl.textContent = msg;
    playVideoSource(videos[0]);
  } else {
    importFiles(files);
  }
});
folderInput.addEventListener("change", () => {
  importFiles(folderInput.files);
  folderInput.value = "";
});

// ---------- โหมดวิดีโอ: เล่นในแหล่งภาพก่อน แล้วค่อยแยกเฟรมเมื่อต้องการ ----------
let videoFile = null;
let videoURL = null;
function playVideoSource(file) {
  videoFile = file;
  if (videoURL) URL.revokeObjectURL(videoURL);
  videoURL = URL.createObjectURL(file);
  state.source = "video";
  stopPose();
  // ซ่อนโหมดอื่น แสดงเครื่องเล่นวิดีโอ
  gridEl.classList.add("hidden");
  poseCanvas.classList.add("hidden");
  detCanvas.classList.add("hidden");
  imageEmptyEl.classList.add("hidden");
  cameraError.classList.add("hidden");
  videoSourceEl.classList.remove("hidden");
  sourceVideoEl.src = videoURL;
  sourceVideoEl.play().catch(() => {});
  sourceBar.render();
}
function closeVideoSource() {
  sourceVideoEl.pause();
  sourceVideoEl.removeAttribute("src");
  sourceVideoEl.load();
  if (videoURL) { URL.revokeObjectURL(videoURL); videoURL = null; }
  videoFile = null;
  videoSourceEl.classList.add("hidden");
}
document.getElementById("vs-split").addEventListener("click", () => {
  if (videoFile) { sourceVideoEl.pause(); openVideoFramesDialog(videoFile); }
});
document.getElementById("vs-close").addEventListener("click", () => {
  closeVideoSource();
  setSource("image"); // กลับไปโหมดภาพนิ่ง (empty state ถ้ายังไม่มีภาพ)
});

// อัปโหลดวิดีโอ → แยกเฟรม → ใช้เป็นภาพนิ่งชุด (ต่อ filter/Export ได้)
let videoDialogOpen = false;
function openVideoFramesDialog(file) {
  if (videoDialogOpen) return; // กันเปิดซ้อน
  videoDialogOpen = true;
  openVideoFrames(file, (frames) => {
    videoDialogOpen = false;
    if (!frames) { statusEl.textContent = "ยกเลิกการแยกเฟรม"; return; }
    if (frames.length === 0) { statusEl.textContent = "ไม่ได้เฟรมจากวิดีโอ (ลองไฟล์อื่น)"; return; }
    closeVideoSource(); // ปิดเครื่องเล่นวิดีโอ → เข้าโหมดภาพนิ่งด้วยเฟรมที่แยกได้
    state.images = frames; // frames มี .canvas อยู่แล้ว (ensureDecoded คืนตรง ๆ)
    decodeCache.clear();
    state.imageIndex = 0;
    state.source = "image";
    gridEl.classList.remove("hidden");
    cameraError.classList.add("hidden");
    updateImageEmpty();
    sourceBar.render();
    setImageIndex(0);
    statusEl.textContent = `🎬 แยกได้ ${frames.length} เฟรม — แสดงเฟรม 1/${frames.length} (กด ▶ ดูถัดไป · Export ประมวลผลทุกเฟรม)`;
  });
}

// ---------- processing ----------
function connectedDisplayIds() {
  return state.pipelines.filter((d) => d.connected).map((d) => d.displayId);
}
// pipeline ที่ worker OpenCV ต้องทำ (ไม่รวมกล่องที่ผ่าน DL — DL รันในฝั่ง renderer)
function workerPipelines() {
  const pipelines = state.pipelines.filter((d) => d.connected && !d.dl);
  const preview = selectedPreviewPipeline();
  if (preview) pipelines.push(preview);
  return pipelines;
}

async function requestProcess() {
  if (state.source !== "image" || state.images.length === 0) return;
  const idx = state.imageIndex;
  const im = await ensureDecoded(idx);
  if (!im || idx !== state.imageIndex) return; // เปลี่ยนภาพระหว่างถอดรหัส → ทิ้ง
  ensureGrid(connectedDisplayIds());
  updateCaptions();
  const { canvas, w, h } = im;
  const d = processDims(w, h); // ย่อตามระดับที่เลือก
  procCanvas.width = d.w;
  procCanvas.height = d.h;
  const ctx = procCanvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(canvas, 0, 0, d.w, d.h);
  sendFrame(ctx.getImageData(0, 0, d.w, d.h));
  renderDLDisplays(canvas, w, h); // กล่องที่ต่อผ่าน DL → รัน YOLO บนภาพนี้
}

function pump() {
  // หยุด (Pause) → ไม่ดึงเฟรมใหม่ จอค้างเฟรมล่าสุด (ประหยัด CPU)
  if (state.paused) { requestAnimationFrame(pump); return; }
  // ประมวลผลผ่าน node-graph เฉพาะโหมดกล้อง Original (โหมด Pose วาดเอง ไม่ต้องผ่าน worker)
  if (state.source === "camera" && state.cameraMode === "original" && cameraStarted) {
    const nw = videoEl.videoWidth;
    const nh = videoEl.videoHeight;
    if (nw > 0 && nh > 0) {
      ensureGrid(connectedDisplayIds());
      updateCaptions();
      // ย่อลงตั้งแต่ตอน capture — ลดทั้งงาน getImageData และงานประมวลผลใน worker
      const { w, h } = processDims(nw, nh);
      scratchCanvas.width = w;
      scratchCanvas.height = h;
      const ctx = scratchCanvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(videoEl, 0, 0, w, h);
      sendFrame(ctx.getImageData(0, 0, w, h));
      renderDLDisplays(videoEl, nw, nh); // กล่องที่ต่อผ่าน DL → YOLO บนเฟรมสด
    }
  }
  requestAnimationFrame(pump);
}

function sendFrame(imageData) {
  if (!workerReady) return;
  if (workerPipelines().length === 0) return; // ไม่มีกล่อง (ที่ไม่ใช่ DL) ให้ประมวลผล
  if (busy) { pendingProcess = true; return; } // ค้างไว้ → ประมวลผลค่าล่าสุดเมื่อ worker ว่าง
  busy = true;
  const copy = new Uint8ClampedArray(imageData.data); // ไม่ทำลายต้นฉบับที่ cache ไว้
  worker.postMessage(
    {
      type: "frame",
      width: imageData.width,
      height: imageData.height,
      buffer: copy.buffer,
      pipelines: workerPipelines(),
    },
    [copy.buffer]
  );
}

// ค่าดิบของ step หนึ่ง ๆ (สำหรับ dropdown): [{k,v}] ตัด mode + prefix opId__ ออก
function stepRawParams(s) {
  const f = FILTERS[s.id];
  if (!f) return [];
  const out = [];
  for (const p of f.params) {
    if (p.showIf && s.params[p.showIf.key] !== p.showIf.value) continue;
    if (p.key === "mode") continue;
    const k = p.key.includes("__") ? p.key.split("__").pop() : p.key;
    out.push({ k, v: s.params[p.key], label: p.label || k });
  }
  return out;
}
function stepLabel(s) {
  const f = FILTERS[s.id];
  if (!f) return s.id;
  return typeof f.subtitle === "function" ? f.subtitle(s.params) : f.name;
}

// สร้างสายทอด (breadcrumb) ในหัวจอ: ภาพ › [step ▾] › [step ▾]
// แต่ละ step คือปุ่ม dropdown — คลิกเปิดกล่องลอยแสดงค่าดิบ + ปุ่มไปแก้ค่าที่ node
function buildChain(cell, dp) {
  // key กันสร้างซ้ำ (จะได้ไม่ปิด dropdown ที่เปิดค้างอยู่โดยไม่จำเป็น)
  const key = dp.connected
    ? "c|" + dp.pipeline.map((s) => s.nodeId + ":" + stepLabel(s) + ":" + JSON.stringify(stepRawParams(s))).join("|")
    : "x";
  if (cell.chainKey === key) return;
  cell.chainKey = key;
  const chain = cell.chain;
  // A menu may be portaled to <body> while open. Move it back before rebuilding
  // so no detached menu remains after a parameter/pipeline update.
  closeAllChainMenus();
  chain.innerHTML = "";

  if (!dp.connected) {
    const e = document.createElement("span");
    e.className = "stage-sub";
    e.textContent = "(ยังไม่ได้ต่อสาย)";
    chain.appendChild(e);
    return;
  }

  // ต้นทาง = ภาพ
  const src = document.createElement("span");
  src.className = "chain-src";
  src.textContent = "ภาพ";
  chain.appendChild(src);

  dp.pipeline.forEach((s) => {
    const sep = document.createElement("span");
    sep.className = "chain-sep";
    sep.textContent = "›";
    chain.appendChild(sep);

    const wrap = document.createElement("span");
    wrap.className = "chain-item";

    const raws = stepRawParams(s);

    // สูตรตายตัว (ไม่มีพารามิเตอร์) → แสดงชื่อเฉย ๆ ไม่มี dropdown (คลิกไปที่ node ได้)
    if (raws.length === 0) {
      const plain = document.createElement("button");
      plain.className = "chain-step chain-step-plain";
      plain.innerHTML = `<span class="chain-name">${stepLabel(s)}</span>`;
      plain.title = "สูตรตายตัว — ไม่มีค่าให้ปรับ (คลิกเพื่อไปที่ node นี้)";
      plain.addEventListener("click", (ev) => {
        ev.stopPropagation();
        closeAllChainMenus();
        editor.selectNodeById(s.nodeId);
      });
      wrap.appendChild(plain);
      chain.appendChild(wrap);
      return;
    }

    // มีพารามิเตอร์ → เป็น dropdown ▾ โชว์ค่าดิบ + ปุ่มไปแก้ค่า
    const btn = document.createElement("button");
    btn.className = "chain-step";
    btn.innerHTML = `<span class="chain-name">${stepLabel(s)}</span><span class="chain-caret">▾</span>`;
    // hover-preview: เห็นค่าดิบทันทีไม่ต้องคลิกเปิด dropdown ก่อน
    btn.title = `${stepLabel(s)} · ${raws.map((r) => `${r.k}=${r.v}`).join(", ")} — คลิกเพื่อปรับ`;

    const menu = document.createElement("div");
    menu.className = "chain-menu hidden";
    for (const r of raws) {
      const row = document.createElement("div");
      row.className = "chain-kv";
      // ป้ายไทยให้อ่านออก + key ดิบ (โยงกับโค้ด) เช่น "ขนาดเคอร์เนล (ksize)"
      row.innerHTML = `<span class="chain-k">${r.label} <em>(${r.k})</em></span><span class="chain-v">${r.v}</span>`;
      menu.appendChild(row);
    }
    const edit = document.createElement("button");
    edit.className = "chain-edit";
    edit.textContent = "✎ ไปแก้ค่า";
    edit.addEventListener("click", (ev) => {
      ev.stopPropagation();
      closeAllChainMenus();
      editor.selectNodeById(s.nodeId);
    });
    menu.appendChild(edit);

    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const open = !menu.classList.contains("hidden");
      closeAllChainMenus();
      // Clicking a breadcrumb Step always selects its graph node, even when a
      // different Display is currently selected in Properties.
      editor.selectNodeById(s.nodeId);
      if (!open) {
        menu.__chainAnchor = wrap;
        document.body.appendChild(menu); // escape stage overflow clipping
        menu.classList.remove("hidden");
        const rect = btn.getBoundingClientRect();
        const gap = 5;
        const menuRect = menu.getBoundingClientRect();
        const left = Math.max(8, Math.min(rect.left, window.innerWidth - menuRect.width - 8));
        const below = rect.bottom + gap;
        const top = below + menuRect.height <= window.innerHeight - 8
          ? below
          : Math.max(8, rect.top - menuRect.height - gap);
        menu.style.left = `${Math.round(left)}px`;
        menu.style.top = `${Math.round(top)}px`;
      }
    });

    wrap.appendChild(btn);
    wrap.appendChild(menu);
    chain.appendChild(wrap);
  });
}

function closeAllChainMenus() {
  document.querySelectorAll(".chain-menu").forEach((m) => {
    m.classList.add("hidden");
    m.style.left = "";
    m.style.top = "";
    const anchor = m.__chainAnchor;
    if (anchor?.isConnected && m.parentElement !== anchor) anchor.appendChild(m);
    else if (m.parentElement === document.body && !anchor?.isConnected) m.remove();
    m.__chainAnchor = null;
  });
}
// คลิกที่อื่น = ปิด dropdown
document.addEventListener("click", closeAllChainMenus);

// อัปเดตหัวจอทุกช่อง: chip (Display N / ชื่อ) + สายทอด step แบบ dropdown
function updateCaptions() {
  for (const dp of state.pipelines) {
    const cell = cells.get(dp.displayId);
    if (!cell) continue;
    const chip = displayChip(dp);
    if (cell.chip.textContent !== chip) cell.chip.textContent = chip;
    buildChain(cell, dp);
  }
}

// วาดผลลัพธ์ทุกกล่อง: แต่ละกล่อง = 1 จอใน grid (พร้อมหัวข้อ) + thumbnail ในตัว node
function drawResults(results) {
  // grid ครอบทุกกล่องที่ต่อสาย (รวมกล่อง DL ที่ renderer วาดเอง)
  ensureGrid(connectedDisplayIds());
  updateCaptions();
  if (results.length === 0) return;

  for (const r of results) {
    const { displayId, width, height, buffer } = r;
    const img = new ImageData(new Uint8ClampedArray(buffer), width, height);

    if (typeof displayId === "string" && displayId.startsWith(SELECTED_PREVIEW_PREFIX)) {
      const currentPreviewId = selectedPreviewPipeline()?.displayId;
      if (displayId !== currentPreviewId) continue; // ผลจากกล่องที่เลิกเลือกแล้ว — ห้ามย้อนมาทับ preview ใหม่
      if (selectedPreviewCanvas.width !== width) selectedPreviewCanvas.width = width;
      if (selectedPreviewCanvas.height !== height) selectedPreviewCanvas.height = height;
      selectedPreviewCanvas.getContext("2d").putImageData(img, 0, 0);
      selectedPreviewCanvas.dataset.ready = "true";
      selectedPreviewLatestImage = { img, width, height };
      if (selectedPreviewView !== "output" && performance.now() - selectedPreviewAnalysisAt >= 250) {
        renderSelectedPreviewAnalysis(img, width, height);
      }
      selectedPreviewEmpty.classList.add("hidden");
      setSelectedPreviewView(selectedPreviewView, false);
      continue;
    }

    // 1) blit ลง offscreen canvas ของกล่องนี้ แล้วส่งให้ editor วาดในตัว node
    const oc = displayCanvasFor(displayId);
    if (oc.width !== width) oc.width = width;
    if (oc.height !== height) oc.height = height;
    oc.getContext("2d").putImageData(img, 0, 0);
    editor.setDisplayImage(displayId, oc);

    // 2) วาดลงจอของกล่องนี้ใน grid ใหญ่
    const cell = cells.get(displayId);
    if (cell) {
      if (cell.canvas.width !== width) cell.canvas.width = width;
      if (cell.canvas.height !== height) cell.canvas.height = height;
      cell.ctx.putImageData(img, 0, 0);
    }
  }
}

// ---------- helpers ----------
function showApp() {
  loadingEl.classList.add("hidden");
  appEl.classList.remove("hidden");
}
function showCameraError(msg) {
  cameraError.innerHTML = "";
  cameraError.classList.add("stage-error");
  const icon = document.createElement("div");
  icon.className = "err-icon";
  icon.textContent = "📷⚠️";
  cameraError.appendChild(icon);
  const title = document.createElement("div");
  title.className = "err-title";
  title.textContent = "เปิดกล้องไม่สำเร็จ";
  cameraError.appendChild(title);
  // msg มาจาก error ระบบ — ใช้ textContent กันเผื่อไว้ (ไม่ใช่ innerHTML)
  const text = document.createElement("div");
  text.className = "err-msg";
  text.textContent = msg;
  cameraError.appendChild(text);
  const steps = document.createElement("span");
  steps.className = "err-steps";
  steps.textContent = "ลองแก้: ตรวจสายกล้อง · ปิดโปรแกรมอื่นที่กำลังใช้กล้อง";
  cameraError.appendChild(steps);

  const actions = document.createElement("div");
  actions.className = "err-actions";
  const retryBtn = document.createElement("button");
  retryBtn.className = "btn btn-primary";
  retryBtn.textContent = "เลือกกล้องใหม่";
  retryBtn.addEventListener("click", async () => {
    await refreshCameras();
    cameraStarted = false;
    await ensureCamera();
  });
  const stillBtn = document.createElement("button");
  stillBtn.className = "btn";
  stillBtn.textContent = "ใช้ภาพนิ่งแทน";
  stillBtn.addEventListener("click", () => {
    setSource("image");
    fileInput.click();
  });
  actions.appendChild(retryBtn);
  actions.appendChild(stillBtn);
  cameraError.appendChild(actions);

  cameraError.classList.remove("hidden");
  gridEl.classList.add("hidden");
  statusEl.textContent = "✕ กล้องไม่พร้อมใช้งาน";
}

main();

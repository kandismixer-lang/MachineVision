// จุดเริ่มของ renderer: node-graph สไตล์ Node-RED + worker + แหล่งภาพ (กล้อง/ภาพนิ่ง)
// การไหล: [แหล่งภาพ] → filter nodes → [ผลลัพธ์] — กราฟเปลี่ยนเมื่อไร ประมวลผลใหม่ทันที
import { createGraphEditor } from "./graph/editor.js";
import { createPalettePanel } from "./ui/palette-panel.js";
import { initUpdatePopup } from "./ui/update-popup.js";
import { createDetailPanel, stepAndNotesOf, infoTextsOf } from "./ui/detail-panel.js";
import { createClassPanel } from "./ui/class-panel.js";
import { createSplitPanel } from "./ui/split-panel.js";
import { createAnnotateCanvas } from "./ui/annotate-canvas.js";
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
import {
  getAnnotations as datasetGetAnnotations,
  getClasses as datasetGetClasses,
  addClass as datasetAddClass,
  setAnnotations as datasetSetAnnotations,
  setSplit as datasetSetSplit,
  getSplit as datasetGetSplit,
  assignSplitsByRatio as datasetAssignSplitsByRatio,
  onChange as datasetOnChange,
} from "./dataset/annotations.js";
import { exportYoloDataset } from "./dataset/export-yolo.js";
import { importYoloDataset } from "./dataset/import-yolo.js";
import { defaultAugmentConfig, normalizeAugmentConfig } from "./dataset/augment.js";
import { createAugmentDialog } from "./ui/augment-dialog.js";
import { initLayoutResize, initViewMenu } from "./ui/layout-resize.js";
// ใบอนุญาตฟอนต์ (SIL OFL-1.1) ต้องไปกับตัวแอปที่ bundle ฟอนต์ IBM Plex — import เพื่อให้ Vite คัดลอกเข้า dist/assets (ไม่ได้ใช้ค่า)
import "../assets/fonts/OFL.txt?url";
import { statusText, sourceDetail, statusLevel, sourceChipLabel } from "./graph/layout-math.js";
import { createTileInspector, resetHoveredView } from "./ui/pixel-inspector.js";
import {
  histogramPlan, resolveThresholdT, markerX, histogramBufferSize, flowActive, routeWorkerError, ERROR_FALLBACK_TEXT, nthStepNodeId, parseNavKey, selectedIdOf,
  SELECTED_PREVIEW_PREFIX, SELECTED_INPUT_PREFIX, buildInputPipeline, pickWorkerPipelines, classifyResultId,
} from "./ui/phase3-math.js";
import { localParamsFor } from "./filters/ui-meta.js";

// framework เป้าหมายของ dataset/โค้ดเทรน: 'yolo' (YOLO txt) | 'rfdetr' (COCO json) — เลือกในหน้าต่าง Export Code (จำค่าล่าสุดไว้)
let exportFramework = "yolo";
export function getExportFramework() { return exportFramework; }
export function setExportFramework(v) { exportFramework = v === "rfdetr" ? "rfdetr" : "yolo"; }

// Offline augmentation config (แบบ Roboflow) — ใช้ตอน Export Dataset สร้างภาพก็อปที่แปลงแล้ว เฉพาะ train
let datasetAugment = defaultAugmentConfig();
export function getDatasetAugment() { return datasetAugment; }
export function setDatasetAugment(v) { datasetAugment = normalizeAugmentConfig(v); }

// UI version switch: เปลี่ยนเป็น "Version1ui" เพื่อย้อนกลับหน้าตาเดิมทันที
// (DOM/worker/processing ใช้ชุดเดียวกันทั้งสองรุ่น)
const UI_VERSION = "Version2ui";
const isUiV2 = UI_VERSION === "Version2ui";
document.body.classList.add(isUiV2 ? "ui-v2" : "ui-v1");
document.body.dataset.uiVersion = UI_VERSION;
initLayoutResize();

// Version1ui ใช้ตำแหน่ง Export เดิมใต้รายการคลาส ส่วน Version2ui ใช้ workflow 3 ขั้น
if (!isUiV2) {
  const legacyClassSection = document.getElementById("class-panel-section");
  const exportButton = document.getElementById("export-yolo-zip-btn");
  const exportSection = document.getElementById("dataset-export-section");
  if (legacyClassSection && exportButton) legacyClassSection.appendChild(exportButton);
  exportSection?.remove();
}

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
// จุดสถานะหน้าข้อความ: เขียว = ปกติ · ส้ม = ยังไม่มี Display ต่อสาย · แดง = ข้อผิดพลาด (ตัดสินจากข้อความ — ดู statusLevel)
// ข้อความถูกเขียนจากหลายที่ จึงใช้ MutationObserver แทนการไล่แก้ทุกจุด (ตั้ง data-level ไม่ทำให้ observer ยิงซ้ำ)
if (statusEl) {
  const syncStatusLevel = () => { statusEl.dataset.level = statusLevel(statusEl.textContent); };
  new MutationObserver(syncStatusLevel).observe(statusEl, { childList: true, characterData: true, subtree: true });
  syncStatusLevel();
}
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
const inspectorTabs = [...document.querySelectorAll("[data-inspector-tab]")];
let inspectorTab = "inspect";

function setInspectorTab(next) {
  if (!isUiV2 || !["inspect", "preview", "dataset"].includes(next)) return;
  inspectorTab = next;
  for (const tab of inspectorTabs) {
    const active = tab.dataset.inspectorTab === next;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
    tab.tabIndex = active ? 0 : -1;
  }

  const groups = {
    inspect: [document.getElementById("detail-panel"), document.getElementById("camera-control-panel")],
    preview: [document.getElementById("selected-preview-panel")],
    dataset: [
      document.querySelector(".dataset-workflow"),
      document.getElementById("class-panel-section"),
      document.getElementById("split-panel-section"),
      document.getElementById("dataset-export-section"),
    ],
  };
  for (const [name, elements] of Object.entries(groups)) {
    for (const element of elements) element?.classList.toggle("v2-tab-hidden", name !== next);
  }
  if (next === "dataset") {
    const classes = document.getElementById("class-panel-section");
    const split = document.getElementById("split-panel-section");
    if (classes) classes.open = true;
    if (split) split.open = true;
    const exportSec = document.getElementById("dataset-export-section");
    if (exportSec) exportSec.open = true;
  }
}

// EXPORT (Guided): เลือก Dataset / Training Code → โชว์คำอธิบาย + ปุ่มของตัวที่เลือกตัวเดียว
// ปุ่ม export จริงยังเป็น id เดิม (export-yolo-zip-btn / export-yolo-code-btn) แค่ซ่อนตัวที่ไม่ได้เลือก
const exportKindBtns = [...document.querySelectorAll(".export-kind-btn")];
function setExportKind(kind) {
  for (const btn of exportKindBtns) {
    const on = btn.dataset.exportKind === kind;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-selected", String(on));
    btn.tabIndex = on ? 0 : -1;
    const pane = document.getElementById(btn.getAttribute("aria-controls"));
    if (pane) pane.hidden = !on;
  }
}
for (const btn of exportKindBtns) {
  btn.addEventListener("click", () => setExportKind(btn.dataset.exportKind));
  btn.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const next = exportKindBtns[(exportKindBtns.indexOf(btn) + 1) % exportKindBtns.length];
    setExportKind(next.dataset.exportKind);
    next.focus();
  });
}

if (isUiV2) {
  for (const tab of inspectorTabs) {
    tab.addEventListener("click", () => setInspectorTab(tab.dataset.inspectorTab));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const index = inspectorTabs.indexOf(tab);
      const direction = event.key === "ArrowRight" ? 1 : -1;
      const next = inspectorTabs[(index + direction + inspectorTabs.length) % inspectorTabs.length];
      setInspectorTab(next.dataset.inspectorTab);
      next.focus();
    });
  }
  // เปิดแอปมาเริ่มที่ "ปรับค่า" เสมอ — ไม่จำ tab สุดท้าย (ผู้ใช้ไม่อยากเด้งไป Dataset)
  try { localStorage.removeItem("mv-ui-v2-inspector-tab"); } catch {}
  setInspectorTab("inspect");
}
let selectedPreviewGeneration = 0;
let selectedPreviewView = "output";
let selectedPreviewLatestImage = null;
let selectedPreviewAnalysisAt = 0;
const selectedPreviewHistogramBins = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
// 3.2 histogram ขาเข้า (ภาพก่อนเข้ากล่องที่เลือก = ผลของสายถึงกล่องก่อนหน้า): ขอจาก worker เป็น pipeline เพิ่มอีกสายเมื่ออยู่แท็บ Histogram
// และวิธีที่เลือกเป็นกลุ่มเกณฑ์/ปรับภาพเท่านั้น · กล้องสดขอไม่ถี่กว่าทุก 300 ms · ภาพนิ่งขอทุกครั้งที่ประมวลผล (ครั้งเดียวต่อค่าที่ปรับ)
const selectedInputBins = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)]; // R,G,B,luma
let selectedInputReady = false;
let selectedInputSamples = 0;
let selectedInputAt = 0;
const HIST_LW = 320; // ขนาด "ตรรกะ" ของ canvas histogram (ขนาด buffer จริง = CSS × DPR — ดู histogramBufferSize)
const HIST_LH = 172;
const HISTOGRAM_DEFAULT_OPS = new Set([
  "grayscale", "colorspace", "channel_split", "rgb_adjust", "color_inrange", "invert",
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
  detConf: 0.25, // ความไวการจับของ YOLO Detect (0.05–0.90) — single source, ใช้ทั้งกล้องสดและภาพนิ่ง
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
  for (const c of cells.values()) c.insp?.destroy();
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

    // viewport ของภาพ (3.1): canvas ผลลัพธ์ + ตารางพิกเซล (overlay) + แถบ readout + ป้ายซูม — ซูม/เลื่อนด้วย CSS transform ของ canvas
    const view = document.createElement("div");
    view.className = "stage-view";
    const cvs = document.createElement("canvas");
    cvs.className = "stage-canvas";
    const gridCv = document.createElement("canvas");
    gridCv.className = "stage-pxgrid";
    gridCv.setAttribute("aria-hidden", "true");
    const readout = document.createElement("div");
    readout.className = "stage-readout mono";
    readout.hidden = true;
    readout.setAttribute("aria-hidden", "true");
    readout.title = "พิกัดและค่าสีของพิกเซลที่เมาส์ชี้ (นับบนภาพที่ประมวลผลแล้ว)";
    readout.innerHTML = '<span class="px-swatch"></span><span class="px-pos"></span><span class="px-val"></span>';
    const badge = document.createElement("button");
    badge.type = "button";
    badge.className = "stage-zoom-badge mono";
    badge.hidden = true;
    badge.title = "กลับมาพอดีจอ (ดับเบิลคลิกที่ภาพ หรือกดปุ่ม 0)";
    badge.setAttribute("aria-label", "กลับมาพอดีจอ");
    badge.innerHTML = '<span class="zb-pct"></span><span class="zb-fit">⤢ พอดีจอ</span>';
    view.append(cvs, gridCv, readout, badge);

    root.appendChild(cap);
    root.appendChild(view);
    gridEl.appendChild(root);
    const cell = { root, canvas: cvs, ctx: cvs.getContext("2d"), chip, chain, lastImg: null, insp: null };
    cell.insp = createTileInspector({ view, canvas: cvs, gridCanvas: gridCv, readout, badge }, { getImage: () => cell.lastImg });
    cells.set(id, cell);
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
    editor.clearNodeErrors(); // เฟรมนี้สำเร็จทั้งหมด → เลิกแสดง error ราย node (3.6)
    state.notes = msg.notes || []; // ตั้งก่อนวาด: histogram อ่านค่าเกณฑ์ Otsu จาก note ของเฟรมเดียวกัน
    drawResults(msg.results || []);
    busy = false;
    // U3: detail-panel อัปเดตตัวเลข/ข้อความในเซลล์เองแบบ in-place — rebuild ทั้งแผงเฉพาะตอน "โครง" เปลี่ยนจริง
    // (เลิก JSON.stringify notes ทุกเฟรม ซึ่งทำให้ regionprops/Motion rebuild แผงทุกเฟรม — P-16)
    if (!detailPanel.applyNotes(state.notes)) renderDetail();
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
    // 3.6 worker บอกว่า node ไหนพัง → กรอบประแดง + "!" + ข้อความบนการ์ดนั้น (ไม่มี nodeId = มีแต่ status bar เหมือนเดิม)
    const failed = routeWorkerError(msg);
    // ข้อยกเว้น OpenCV.js เป็นตัวเลขดิบ → ไม่แสดงเลขเปล่าใน status bar (ใช้ข้อความสำรองแล้วไม่ต้องมีวงเล็บซ้ำ)
    statusEl.textContent = "✕ ประมวลผลไม่สำเร็จ — ลองปรับค่าหรือเปลี่ยนภาพดู" + (failed.text === ERROR_FALLBACK_TEXT ? "" : " (" + failed.text + ")");
    editor.clearNodeErrors();
    if (failed.nodeId != null) editor.setNodeError(failed.nodeId, failed.text);
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
const cvZoomPctEl = document.getElementById("cv-zoom-pct");
const editor = createGraphEditor(graphCanvasEl, {
  // ป้าย % ซูมในหัวกระดาน (ล้อเมาส์ / ปุ่ม − ＋ / Fit) — editor แจ้งเฉพาะตอนข้อความเปลี่ยน
  onZoomChange: (text) => { if (cvZoomPctEl) cvZoomPctEl.textContent = text; },
  onPipelineChange: (pipelines) => {
    state.pipelines = pipelines;
    editor.clearNodeErrors(); // กราฟเปลี่ยน → เลิกแสดงสถานะ error เดิม (ถ้ายังพังอยู่ เฟรมถัดไปจะตั้งใหม่)
    // เก็บกวาด canvas ของกล่องผลลัพธ์ที่ถูกลบไปแล้ว
    const alive = new Set(pipelines.map((d) => d.displayId));
    for (const id of displayCanvases.keys()) {
      if (!alive.has(id)) displayCanvases.delete(id);
    }
    updateStatus();
    updateCaptions(); // ชื่อ/สายเปลี่ยน → อัปเดตหัวจอทันที
    // วิธีของกล่องที่เลือกถูกเปลี่ยนนอกแผง (เมนู Properties ของ LiteGraph) → แผงถือ step เก่า ต้องสร้างใหม่
    if (state.selectedNode && !state.selectedDL && !state.selectedDisplay &&
        detailPanel.isViewStale(stepAndNotesOf(state.selectedNode, []).step)) renderDetail();
    requestProcess();
  },
  onSelect: (sel) => {
    selectedPreviewGeneration++;
    selectedPreviewLatestImage = null;
    selectedPreviewAnalysisAt = 0;
    selectedInputReady = false;
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

// ต้องการ histogram ขาเข้าไหม: อยู่แท็บ Histogram + วิธีเป็นกลุ่มเกณฑ์/ปรับภาพ + มีสาย preview
function wantsInputHistogram() {
  return selectedPreviewView === "histogram" && histogramPlan(selectedPreviewOpId()) !== "plain";
}

// pipeline "ขาเข้า" ของกล่องที่เลือก = สายเดียวกับ preview ตัดขั้นสุดท้ายออก (ขั้นแรก → ภาพต้นฉบับ)
function selectedInputPipeline() {
  if (!wantsInputHistogram()) return null;
  const preview = selectedPreviewPipeline();
  if (!preview) return null;
  return buildInputPipeline(preview, `${SELECTED_INPUT_PREFIX}:${state.selectedNode.id}:${selectedPreviewGeneration}`);
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
  // ภาพนิ่งไม่มี pump — สลับมาแท็บ Histogram แล้วต้องสั่งประมวลผลเพื่อให้ได้ภาพขาเข้ามาด้วย
  if (userInitiated && view === "histogram" && !selectedInputReady && wantsInputHistogram()) requestProcess();
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

// นับ histogram ของภาพขาเข้า (R,G,B + luma) ลง selectedInputBins ด้วยการสุ่มเท่ากับของขาออก → คืนจำนวนตัวอย่าง
function accumulateInputHistogram(data, pixelCount) {
  for (const bins of selectedInputBins) bins.fill(0);
  const step = Math.max(1, Math.ceil(pixelCount / 160000));
  let n = 0;
  for (let p = 0; p < pixelCount; p += step) {
    const i = p * 4;
    const r = data[i], g = data[i + 1], b = data[i + 2];
    selectedInputBins[0][r]++; selectedInputBins[1][g]++; selectedInputBins[2][b]++;
    selectedInputBins[3][Math.round(0.299 * r + 0.587 * g + 0.114 * b)]++;
    n++;
  }
  return n;
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

  // วางแผนกราฟตามวิธีที่เลือก (3.2): threshold = histogram ขาเข้า + เส้นเกณฑ์ T · compare = ขาเข้า (จาง) ซ้อนขาออก · plain = เหมือนเดิม
  const plan = histogramPlan(selectedPreviewOpId());
  const haveInput = plan !== "plain" && selectedInputReady && selectedInputSamples > 0;
  let marker = null;
  let legend = "";
  if (plan === "threshold" && haveInput) {
    const sel = state.selectedNode;
    const { step, notes } = stepAndNotesOf(sel, state.notes);
    // Otsu ที่ยังไม่มี note (กล่องไม่ได้ต่อ Display) → คำนวณจาก histogram ขาเข้าเอง (สูตรเดียวกับ OpenCV)
    marker = resolveThresholdT(selectedPreviewOpId(), localParamsFor(sel.filterId, sel.getParams()), infoTextsOf(notes), selectedInputBins[3]);
    legend = "ขาเข้า (ก่อนตัดเกณฑ์)";
  } else if (plan === "compare" && haveInput) {
    legend = "จาง=ขาเข้า เข้ม=ขาออก";
  }

  // canvas คมที่ DPR > 1: เรขาคณิตคิดบนพื้น 320×172 คงที่ แล้วขยาย transform ตามขนาด buffer จริง (3.8)
  const hb = histogramBufferSize(selectedPreviewHistogram.getBoundingClientRect().width, window.devicePixelRatio, HIST_LW, HIST_LH);
  if (selectedPreviewHistogram.width !== hb.w) selectedPreviewHistogram.width = hb.w;
  if (selectedPreviewHistogram.height !== hb.h) selectedPreviewHistogram.height = hb.h;
  const ctx = selectedPreviewHistogram.getContext("2d");
  ctx.setTransform(hb.scale, 0, 0, hb.scale, 0, 0);
  const w = HIST_LW;
  const h = HIST_LH;
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
  // เส้นหลัก: threshold → luma ของภาพขาเข้า · อื่น ๆ → ขาออก (เทา = เส้นเดียว, สี = R/G/B)
  const mainBins = plan === "threshold" && haveInput ? [[selectedInputBins[3], "#38d9ff"]] : null;
  const channels = mainBins || (grayscale ? [[hist[0], "#38d9ff"]] : [
    [hist[0], "#ff5577"], [hist[1], "#45e59a"], [hist[2], "#4aa3ff"],
  ]);
  // ขาเข้าที่จางอยู่ข้างหลัง (compare) — ขาออกเป็นเทาล้วนก็เทียบด้วย luma ขาเข้าเส้นเดียว; ปรับสัดส่วนตามจำนวนตัวอย่างให้เทียบกันได้
  let faint = null;
  let faintScale = 1;
  if (plan === "compare" && haveInput) {
    faint = grayscale ? [[selectedInputBins[3], "#38d9ff"]] : [
      [selectedInputBins[0], "#ff5577"], [selectedInputBins[1], "#45e59a"], [selectedInputBins[2], "#4aa3ff"],
    ];
    faintScale = samples / selectedInputSamples;
  }
  const mainScale = mainBins ? samples / selectedInputSamples : 1; // luma ขาเข้านับตัวอย่างต่างกับขาออก (ปกติเท่ากัน)
  let peak = 1;
  for (const [bins] of channels) {
    for (let i = 0; i < 256; i++) if (bins[i] * mainScale > peak) peak = bins[i] * mainScale;
  }
  if (faint) for (const [bins] of faint) for (let i = 0; i < 256; i++) if (bins[i] * faintScale > peak) peak = bins[i] * faintScale;
  const plot = (bins, color, lineW, alpha, scale) => {
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineW;
    ctx.beginPath();
    for (let i = 0; i < 256; i++) {
      const x = 10 + (i / 255) * (w - 20);
      const normalized = Math.log1p(bins[i] * scale) / Math.log1p(peak);
      const y = h - 22 - normalized * (h - 34);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  };
  if (faint) for (const [bins, color] of faint) plot(bins, color, 1.5, 0.5, faintScale);
  for (const [bins, color] of channels) plot(bins, color, grayscale || mainBins ? 2 : 1.35, 1, mainScale);
  // เส้นเกณฑ์ T (เส้นประสีส้ม = --warning) + ป้าย "T = 137"
  if (marker != null) {
    const mx = markerX(marker, 10, w - 10);
    ctx.strokeStyle = "#ff9f43";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(mx, 8); ctx.lineTo(mx, h - 22); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#ff9f43";
    ctx.font = "600 11px 'IBM Plex Mono', Consolas, monospace";
    const label = "T = " + marker;
    const right = mx > w - 70;
    ctx.textAlign = right ? "right" : "left";
    ctx.fillText(label, right ? mx - 5 : mx + 5, 21);
    ctx.textAlign = "left";
  }
  ctx.fillStyle = "#71849d";
  ctx.font = "500 10px 'IBM Plex Mono', Consolas, monospace"; // ตัวเลขล้วน → Plex Mono (มี fallback)
  ctx.fillText("0", 10, h - 7);
  ctx.fillText("128", w / 2 - 9, h - 7);
  ctx.fillText("255", w - 30, h - 7);
  if (legend) {
    ctx.font = "500 10px 'IBM Plex Sans Thai', 'Leelawadee UI', 'Segoe UI', sans-serif";
    ctx.fillText(legend, 34, h - 7);
  }
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
  onTemplates: () => openTemplates(), // ปุ่ม "เริ่มจากเทมเพลต" ใน empty state ของ Properties
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

// โหมดตีกรอบ (Annotate) ประกาศ (แบบ let) ไว้ก่อน — createClassPanel ด้านล่าง render() ทันทีตอนสร้าง
// (เรียก getActiveIdx() ทันทีถ้าคลาสมีอยู่แล้ว) จึงต้องมีตัวแปรนี้พร้อมใช้ก่อนถูกเรียก ไม่งั้นชน TDZ ของ const
let annotateCanvas;

// แผงจัดการคลาสของ dataset — คุยกับ dataset/annotations.js ตรง ๆ
// ไม่ผูกกับ state ของภาพที่เลือกอยู่ (จัดการคลาสได้ตลอด ไม่ว่าจะมีภาพหรือกำลังใช้กล้อง)
// หมายเหตุความปลอดภัย: createClassPanel() เรียก render() ทันทีแบบ synchronous ตอนสร้าง — ถ้าผู้ใช้เคยเพิ่มคลาสไว้แล้ว
// (persist ผ่าน localStorage) getActiveIdx() จะถูกเรียกจริงตั้งแต่บรรทัดนี้ ก่อนที่ annotateCanvas จะถูก assign ด้านล่าง
// (ยังเป็น undefined อยู่) — ต้อง optional-chain กันแอปแครชตอนเปิดถ้ามีคลาสเก่าอยู่แล้ว
const classPanel = createClassPanel(document.getElementById("class-panel"), {
  onPick: (idx) => annotateCanvas?.setActiveClassIdx(idx),
  getActiveIdx: () => (annotateCanvas?.isActive() ? annotateCanvas.getActiveClassIdx() : -1),
  getClassCounts: () => datasetGetClasses().map((_, classIdx) =>
    state.images.reduce((sum, im) => sum + (datasetGetAnnotations(im.name).some((a) => a.classIdx === classIdx) ? 1 : 0), 0)),
  getDatasetCount: () => state.images.reduce((sum, im) => sum + (datasetGetAnnotations(im.name).length ? 1 : 0), 0),
});

// แผงแบ่ง dataset train/valid/test (Phase 4) — คุยกับ dataset/annotations.js ตรง ๆ เหมือนแผงคลาส
// "ภาพที่มีกรอบ" เท่านั้นที่นับ/แบ่งได้ (ภาพเปล่า export ไม่ได้อยู่แล้ว — ดู exportYoloZip ด้านล่าง)
let datasetSplitConfirmed = false;
let datasetRevision = 0;
let datasetZipRevision = -1;
const splitPanel = createSplitPanel(document.getElementById("split-panel"), {
  getAnnotatedKeys: () => state.images.map((im) => im.name).filter((name) => datasetGetAnnotations(name).length > 0),
  getCurrentKey: () => currentImageKey(),
  getClassCount: () => datasetGetClasses().length,
  onApply: () => { datasetSplitConfirmed = true; },
});
datasetOnChange(() => { datasetRevision++; });

function refreshDatasetPanels({ resetSplit = false } = {}) {
  if (resetSplit) {
    datasetSplitConfirmed = false;
    datasetRevision++;
  }
  classPanel.render();
  splitPanel.render();
}

// ใช้เฉพาะภาพนิ่ง (image mode); คุยกับ dataset/annotations.js ผ่านโมดูลนี้เท่านั้น
annotateCanvas = createAnnotateCanvas({
  canvas: document.getElementById("ann-canvas"),
  toolbar: document.getElementById("ann-toolbar"),
  emptyClassesEl: document.getElementById("ann-empty-classes"),
  getImageKey: () => currentImageKey(),
  getOriginalImage: async () => {
    if (state.source !== "image" || state.images.length === 0) return null;
    return ensureDecoded(state.imageIndex);
  },
  getFilteredCanvas: () => {
    const dp = state.pipelines.find((d) => d.connected);
    if (!dp) return null;
    const oc = displayCanvases.get(dp.displayId);
    return oc && oc.width ? oc : null;
  },
  focusClassPanel: () => {
    setInspectorTab("dataset");
    const sec = document.getElementById("class-panel-section");
    if (sec) {
      sec.open = true;
      sec.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  },
  // ann-canvas แทนที่ stage-grid ชั่วคราวขณะอยู่ในโหมด (เหมือน pose/det canvas) — คืนค่าเสมอไม่ว่าจะออกทางไหน
  onEnter: () => gridEl.classList.add("hidden"),
  onExit: () => gridEl.classList.remove("hidden"),
  setStatus: (msg) => { statusEl.textContent = msg; },
  getImagePosition: () =>
    state.source === "image" && state.images.length
      ? { index: state.imageIndex, total: state.images.length }
      : null,
  onNavImage: (dir) => { if (state.images.length > 1) setImageIndex(state.imageIndex + dir); },
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
  exportBusy: () => exporting,
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
  onClearImages: () => clearImages(),
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

// คีย์ของภาพปัจจุบัน สำหรับผูก annotation ต่อภาพ (dataset/annotations.js) — key ด้วยชื่อไฟล์ (ดูเหตุผลใน annotations.js)
// โหมดกล้อง/ไม่มีภาพนำเข้า = ไม่มี key (คืน null) — ให้ UI annotation ต่อภาพ (เฟสถัดไป) ปิดใช้งานเมื่อไม่มี key
function currentImageKey() {
  if (state.source !== "image" || state.images.length === 0) return null;
  return state.images[state.imageIndex]?.name || null;
}

// เลื่อนดูภาพในชุด (วนรอบ) แล้ว re-process
function setImageIndex(i) {
  if (state.images.length === 0) return;
  const n = state.images.length;
  state.imageIndex = ((i % n) + n) % n;
  state.source = "image";
  detPendingArgs = null; // กัน detect ค้างจากภาพเดิมมารันทับภาพที่เพิ่งเปลี่ยนไป
  sourceBar.render();
  updateImageEmpty();
  statusEl.textContent = `ภาพ ${state.imageIndex + 1}/${n} — ${state.images[state.imageIndex].name}`;
  requestProcess();
  annotateCanvas.onImageChanged(); // no-op ถ้าไม่ได้อยู่ในโหมดตีกรอบ — โหลดพื้นหลัง+กรอบของภาพใหม่
  splitPanel.render(); // อัปเดตปุ่ม split ต่อภาพให้ตรงกับภาพที่เพิ่งเปลี่ยนไป
}

// export ผลลัพธ์แต่ละกล่อง (ที่ต่อสายครบ) เป็นไฟล์ PNG
// หมายเหตุ: รูปแบบ export ปรับเปลี่ยนได้ภายหลัง (ตอนนี้ default = PNG ต่อกล่อง)
function sanitizeName(s) {
  return (s || "result").replace(/[\\/:*?"<>|]+/g, "_").trim().slice(0, 60) || "result";
}
// ปุ่ม "โค้ด YOLO": เปิดหน้าต่างโค้ดเทรนอย่างเดียว (ไม่เซฟภาพ)
const exportDialog = createExportDialog();
function openExportCode() {
  if (exporting) {
    statusEl.textContent = "กำลัง Export อยู่ — รอให้เสร็จก่อนเปิด YOLO Code";
    return;
  }
  const labeled = state.images.some((im) =>
    datasetGetAnnotations(im.name).some((ann) => ann.type !== "line"));
  exportDialog.open(() => {}, {
    datasetReady: labeled && datasetGetClasses().length > 0,
    zipReady: datasetZipRevision >= 0 && datasetZipRevision === datasetRevision,
  }); // โชว์/ดาวน์โหลดโค้ดเท่านั้น — เซฟภาพแยกเป็นอีกปุ่ม
}
document.getElementById("export-yolo-code-btn")?.addEventListener("click", openExportCode);
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
// ใส่ timeout กันแขวนถาวรถ้า worker ค้าง/ตายแบบเงียบ (ไม่ throw error event แต่ก็ไม่ตอบกลับ) —
// ถ้าครบเวลาแล้วยังไม่มีผล ให้คืน [] (ฝั่งเรียกจะ fallback ไปใช้ภาพต้นฉบับ) และเคลียร์ waiter ที่ค้างทิ้ง กัน leak
const EXPORT_TIMEOUT_MS = 20000;
let exportTimedOutCount = 0; // นับจำนวนครั้งที่ timeout ระหว่าง export ปัจจุบัน (ไว้แจ้งผู้ใช้)
function processExport(imageData, pipelines) {
  return new Promise((resolve) => {
    const id = ++exportSeq;
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      exportWaiters.delete(id);
      exportTimedOutCount++;
      resolve([]);
    }, EXPORT_TIMEOUT_MS);
    exportWaiters.set(id, (msg) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(msg.results || []);
    });
    const copy = new Uint8ClampedArray(imageData.data);
    worker.postMessage(
      { type: "export", reqId: id, width: imageData.width, height: imageData.height, buffer: copy.buffer, pipelines },
      [copy.buffer]
    );
  });
}
// เหมือน resultToPng แต่คืน canvas ตรง ๆ (ใช้ตอน export dataset YOLO ที่ต้องเข้ารหัส JPEG เอง ไม่ใช่ PNG)
function resultToCanvas(result) {
  const c = document.createElement("canvas");
  c.width = result.width;
  c.height = result.height;
  c.getContext("2d").putImageData(
    new ImageData(new Uint8ClampedArray(result.buffer), result.width, result.height),
    0,
    0
  );
  return c;
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
// ระหว่าง export ใด ๆ (batch/YOLO) ทำงานอยู่ ปุ่ม export อื่นต้อง disable กันผู้ใช้กดซ้อน
// (เช่นกด Save Picture ระหว่าง YOLO export ค้าง → เปิด dialog เลือกโฟลเดอร์ซ้อนแล้ว exportBatch return เงียบ สับสน)
function setExportButtonsDisabled(disabled) {
  const ids = ["save-picture-btn", "export-code-btn", "export-yolo-code-btn", "export-yolo-zip-btn"];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) el.disabled = disabled;
  }
}
async function exportBatch(connected, dir) {
  if (exporting) return;
  const total = state.images.length;
  // โหมด fallback (ไม่มีโฟลเดอร์) ถ้าไฟล์เยอะจะเด้ง dialog รัว — เตือนก่อน
  const files = total * connected.length;
  if (!dir && files > 60 && !window.confirm(`จะบันทึกทั้งหมด ${files} ไฟล์ ดำเนินการต่อ?`)) return;
  exporting = true;
  setExportButtonsDisabled(true);
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
    setExportButtonsDisabled(false);
  }
}

// Export dataset ที่ตีกรอบไว้ (dataset/annotations.js) เป็น .zip รูปแบบ YOLO — ดู dataset/export-yolo.js
// ภาพที่ export = ตามโหมดพื้นหลังที่ผู้ใช้เลือกไว้ตอนตีกรอบ (bgMode: ต้นฉบับ/ผ่าน filter) — อ่านจาก annotateCanvas.getBgMode()
// ภาพที่ไม่มีกรอบเลย ถูกข้ามไปอัตโนมัติ (นับแจ้งในสรุปผล) — YOLO เทรนภาพเปล่าไม่ได้ในเฟสนี้
// Augmentation modal (แบบ Roboflow) — บังคับขึ้นตอน Export Dataset
async function getAugSampleCanvas() {
  const hasLabel = (name) => datasetGetAnnotations(name).some((a) => a.type !== "line");
  const rec = state.images.find((im) => hasLabel(im.name)) || state.images[0];
  if (!rec) return null;
  if (rec.canvas) return rec.canvas;
  const im = await loadImageFile(rec.file);
  return im ? im.canvas : null;
}
// นับภาพต่อ split (เฉพาะภาพที่มี label เทรนได้) — ใช้โชว์ multiplier ในหน้า Create
// ถ้ายังไม่ได้แบ่ง split จะจำลอง 70/20/10 (ตรงกับ auto-split ที่จะเกิดหลังปิด modal)
// เพื่อไม่ให้เลขในหน้า Create หลอก (aug เกิดเฉพาะ train)
function getAugCounts() {
  const trainable = state.images.filter((im) =>
    datasetGetAnnotations(im.name).some((a) => a.type !== "line"));
  const alreadySplit = trainable.some((im) => datasetGetSplit(im.name) !== "train");
  if (!datasetSplitConfirmed && !alreadySplit) {
    const total = trainable.length;
    const valid = Math.round(total * 0.2);
    const test = Math.round(total * 0.1);
    return { train: Math.max(0, total - valid - test), valid, test };
  }
  const c = { train: 0, valid: 0, test: 0 };
  for (const im of trainable) {
    const s = datasetGetSplit(im.name);
    if (c[s] != null) c[s]++;
  }
  return c;
}
const augmentDialog = createAugmentDialog({ getSampleCanvas: getAugSampleCanvas, getCounts: getAugCounts });

let datasetExportPending = false;
async function exportYoloZip() {
  if (datasetExportPending || exporting) return;
  datasetExportPending = true;
  try {
    await runDatasetExport();
  } finally {
    datasetExportPending = false;
  }
}
async function runDatasetExport() {
  if (exporting) return;
  if (state.source !== "image" || state.images.length === 0) {
    statusEl.textContent = "โหมดตีกรอบใช้ได้เฉพาะภาพนิ่ง — นำเข้าภาพก่อน export dataset";
    return;
  }
  if (datasetGetClasses().length === 0) {
    statusEl.textContent = "ยังไม่มีคลาส — สร้างคลาสก่อน export dataset";
    return;
  }
  const hasTrainableLabel = (name) => datasetGetAnnotations(name).some((ann) => ann.type !== "line");
  const anyAnnotated = state.images.some((im) => hasTrainableLabel(im.name));
  if (!anyAnnotated) {
    statusEl.textContent = "ยังไม่มี Bounding Box หรือ Polygon สำหรับเทรน — Line ใช้ทำ annotation ได้แต่ Export YOLO ไม่ได้";
    return;
  }

  const annotatedKeys = state.images.map((im) => im.name).filter(hasTrainableLabel);

  // บังคับขั้นตอน Augmentation ก่อน (แบบ Roboflow) — Cancel = ยกเลิก export
  const proceed = await augmentDialog.openForExport();
  if (!proceed) {
    statusEl.textContent = "ยกเลิกการ export dataset";
    return;
  }

  let dir = null;
  if (canExportFolder) {
    dir = await window.mvExport.chooseDir();
    if (!dir) {
      statusEl.textContent = "ยกเลิกการ export dataset";
      return;
    }
  }

  // เลือกปลายทางสำเร็จแล้วค่อย auto-split เพื่อให้การกดยกเลิกไม่เปลี่ยน dataset
  // และเคารพ 100/0/0 ที่ผู้ใช้กด Apply ใน session นี้
  const alreadySplit = annotatedKeys.some((k) => datasetGetSplit(k) !== "train");
  let autoSplitApplied = false;
  if (!datasetSplitConfirmed && !alreadySplit && annotatedKeys.length > 0) {
    datasetAssignSplitsByRatio(annotatedKeys, { train: 70, valid: 20, test: 10 });
    splitPanel.render();
    autoSplitApplied = true;
  }

  // โหมด 'filtered' ต้องมีกล่องผลลัพธ์ต่อสายครบอย่างน้อย 1 กล่อง ไม่งั้น fallback ไปใช้ภาพต้นฉบับแทน (แจ้งผู้ใช้)
  const bgMode = annotateCanvas.getBgMode();
  const filterDp = bgMode === "filtered" ? state.pipelines.find((d) => d.connected) : null;
  if (bgMode === "filtered" && !filterDp) {
    statusEl.textContent = "โหมดพื้นหลัง 'ผ่าน filter' แต่ยังไม่มีกล่องผลลัพธ์ที่ต่อสายครบ — จะใช้ภาพต้นฉบับแทน";
  }

  exporting = true;
  setExportButtonsDisabled(true);
  exportTimedOutCount = 0;
  let fallbackToOriginal = 0; // นับภาพที่ควรเป็น 'filtered' แต่ worker error/timeout → ใช้ภาพต้นฉบับแทน (แจ้งผู้ใช้ตอนจบ)
  const entries = state.images.map((rec) => ({
    key: rec.name,
    getCanvas: async () => {
      // ภาพจากคลังแคป: ถอดไว้แล้ว (canvas อยู่ในหน่วยความจำ); ภาพจากไฟล์: ถอดทีละภาพ (memory คงที่)
      const im = rec.canvas ? { canvas: rec.canvas, w: rec.w, h: rec.h } : await loadImageFile(rec.file);
      if (!im) return null;
      if (!filterDp) return im.canvas;
      const data = imageToProcData(im);
      const results = await processExport(data, state.pipelines);
      const r = results.find((x) => x.displayId === filterDp.displayId);
      if (!r) fallbackToOriginal++;
      return r ? resultToCanvas(r) : im.canvas;
    },
  }));
  const exportDatasetRevision = datasetRevision;

  try {
    const exporter = exportYoloDataset;
    const { blob, stats } = await exporter(entries, {
      augment: datasetAugment,
      onProgress: (done, total) => {
        statusEl.textContent = `กำลังแพ็ค dataset... ${done}/${total} ภาพ`;
      },
    });
    const bytes = new Uint8Array(await blob.arrayBuffer());
    if (dir) {
      await window.mvExport.writeFile(dir, "dataset.zip", bytes);
    } else {
      downloadBlob(blob, "dataset.zip");
    }
    const kind = stats.framework === "rfdetr" ? "RF-DETR (COCO)" : stats.task === "segment" ? "YOLO segmentation" : "YOLO detect";
    statusEl.textContent =
      `Export ${kind} dataset เสร็จ — ` +
      `${stats.exported} ภาพ (ข้าม ${stats.skipped} ภาพไม่มีกรอบ), ` +
      `${stats.boxes} label, ${stats.classes} คลาส · ` +
      (stats.augmented > 0 ? `+${stats.augmented} ภาพ augment · ` : "") +
      `train ${stats.perSplit.train} / valid ${stats.perSplit.valid} / test ${stats.perSplit.test}` +
      (stats.skippedLines > 0 ? ` · ⚠ ข้าม ${stats.skippedLines} เส้น (line เทรนไม่ได้)` : "") +
      (dir ? " (ลงโฟลเดอร์ที่เลือก)" : "") +
      (fallbackToOriginal > 0
        ? ` · ⚠ ${fallbackToOriginal} ภาพใช้ต้นฉบับแทน (filter ไม่พร้อม)`
        : "") +
      (autoSplitApplied
        ? " · แบ่งอัตโนมัติ 70/20/10 ให้แล้ว"
        : stats.perSplit.valid === 0
          ? " · ⚠ ยังไม่ได้แบ่ง valid — ใช้ train validate ไปก่อน (แนะนำกดปุ่ม 🔀 สุ่มแบ่ง)"
          : "");
    if (datasetRevision === exportDatasetRevision) datasetZipRevision = exportDatasetRevision;
  } catch (err) {
    statusEl.textContent = `✕ Export dataset ไม่สำเร็จ: ${err.message || err}`;
  } finally {
    exporting = false;
    setExportButtonsDisabled(false);
  }
}

const exportYoloZipBtn = document.getElementById("export-yolo-zip-btn");
if (exportYoloZipBtn) exportYoloZipBtn.addEventListener("click", () => exportYoloZip());

// popup แจ้งเตือนอัปเดต (เด้งเองตอนมีเวอร์ชันใหม่ + โชว์ release notes)
initUpdatePopup();

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
      getConf: () => state.detConf,
      onConf: (v) => {
        state.detConf = v;
        // เลื่อนสไลเดอร์แล้วตรวจจับใหม่ทันทีบนภาพปัจจุบัน — กล้องสดจะใช้ค่าใหม่ในเฟรมถัดไปเองอยู่แล้ว
        // (renderDLDisplays เรียกทุกเฟรม) ส่วนภาพนิ่งต้องสั่ง requestProcess() เพื่อรันซ้ำทันที
        if (state.source === "image") requestProcess();
      },
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
  // step พก nodeId + วิธีปัจจุบัน แล้วกรอง note ตามกล่องจริง (ไม่ใช่ตามชนิด filter) และทิ้ง note ของวิธีเก่า
  // — ตรรกะอยู่ใน detail-panel.js (stepAndNotesOf) เพื่อให้เทสต์พฤติกรรมได้
  const { step, notes } = stepAndNotesOf(sel, state.notes);

  // ถ้าผู้ใช้กำลังลาก slider อยู่ อย่า rebuild ทั้ง panel (จะหลุดมือ) —
  // อัปเดตเฉพาะข้อความรายงานผล (เช่น จำนวนวัตถุ) แทน
  const active = document.activeElement;
  // เฉพาะ INPUT (สไลเดอร์/ช่องตัวเลข) — SELECT "วิธี" ไม่นับ: หลังเลือกวิธีแผงสร้างใหม่แล้วคืนโฟกัสให้ select ตัวใหม่
  // ถ้านับ SELECT ด้วย โครงใหม่ (เช่น ตารางค่าของวิธีใหม่) จะไม่ถูก rebuild เพราะ select ยังโฟกัสอยู่
  if (active && detailPanelEl.contains(active) && active.tagName === "INPUT") {
    // ข้อความ note-info อาจเป็น readout tile (หลาย span) — เขียนผ่าน detail-panel เพื่อไม่ทำโครง tile พัง
    detailPanel.updateInfo(infoTextsOf(notes));
    return;
  }

  detailPanel.render(step, notes);
}

function updateStatus() {
  // ตรรกะข้อความอยู่ใน graph/layout-math.js (เทสต์ได้) — ใช้ชื่อขั้นตอนที่คนอ่านรู้เรื่อง (stepLabel) ไม่ใช่ id ภายใน
  statusEl.textContent = statusText(state.pipelines, stepLabel);
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
    syncWireFlow();
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
  if (s !== "image" && annotateCanvas.isActive()) annotateCanvas.exit(); // ออกจากภาพนิ่ง → ออกจากโหมดตีกรอบด้วย
  if (s !== "video" && videoURL) closeVideoSource(); // ออกจากโหมดวิดีโอ
  state.source = s;
  state.paused = false; // paused มีอายุเฉพาะ session ปัจจุบัน — สลับ source แล้วกลับมารันสด
  syncWireFlow();
  pendingProcess = false; // กัน pendingProcess ค้างข้ามโหมด (เช่น กล้อง→ภาพนิ่ง) ประมวลผลเกิน
  detPendingArgs = null; // กัน detect ค้างจากภาพ/โหมดเดิมมารันทับภาพใหม่
  invalidateSelectedPreview();
  renderSelectedPreviewState(); // สลับ source → ถ้า preview ใช้ไม่ได้ ให้เคลียร์/โชว์ข้อความ ไม่ค้างภาพเก่า
  sourceBar.render();
  splitPanel.render(); // สลับ source → currentImageKey() เปลี่ยน (เช่น image→camera คืน null) ต้องซ่อน/อัปเดตแถบ per-image ทันที กันปุ่ม Train/Valid/Test ค้าง key เดิม

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
  syncWireFlow();
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
  // ESC ต้องปิด dialog ได้แม้ focus อยู่ใน input/select ของ dialog
  const overlays = document.querySelectorAll(".modal-overlay");
  if (overlays.length) {
    if (e.key === "Escape") {
      const top = overlays[overlays.length - 1];
      const closeBtn = top.querySelector("[data-close]");
      if (closeBtn) { e.preventDefault(); closeBtn.click(); }
    }
    return;
  }
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
  // โฟกัสอยู่บนปุ่มแล้วกด Space/Enter → ปล่อยให้ปุ่มนั้นทำงานเนทีฟ (กันยิง action ซ้อน)
  if (t && t.tagName === "BUTTON" && (e.key === " " || e.key === "Enter")) return;
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
  // UI เฟส 3.7: F = Fit · 1–9 = เลือกกล่องที่ N ตามเลขบนการ์ด · 0 = จอผลลัพธ์ที่ชี้อยู่กลับพอดีจอ
  // อยู่หลังคีย์ที่ผู้ใช้ผูกเอง (ถ้าผูก f/ตัวเลขไว้ ใช้ตามที่ผูก) · โหมดตีกรอบใช้เลข 1–9 เลือกคลาสอยู่แล้ว จึงข้าม
  if (annotateCanvas.isActive()) return;
  const nav = parseNavKey(e);
  if (!nav) return;
  if (nav.type === "fit") {
    e.preventDefault();
    editor.fitView();
  } else if (nav.type === "step") {
    const id = nthStepNodeId(state.pipelines, nav.n, selectedIdOf(state));
    if (id != null) {
      e.preventDefault();
      editor.selectNodeById(id);
    }
  } else if (nav.type === "reset-view") {
    e.preventDefault();
    if (!resetHoveredView()) for (const c of cells.values()) c.insp?.resetView(); // ไม่ได้ชี้จอไหน → ทุกจอ
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
  if (show && annotateCanvas.isActive()) annotateCanvas.exit(); // ภาพหมด (เช่นลบภาพสุดท้ายออก) — ออกจากโหมดตีกรอบ
  imageEmptyEl.classList.toggle("hidden", !show);
  // อยู่ในโหมด annotate ต้องคง grid ซ่อนไว้เสมอ — ไม่งั้น caption "Display N" ของจอผลลัพธ์
  // จะโผล่ทับ toolbar (ann-canvas letterbox ไม่ครอบมุมบนซ้าย) เวลา updateImageEmpty ถูกเรียกซ้ำ
  gridEl.classList.toggle("hidden", show || annotateCanvas.isActive());
}
document.getElementById("ie-single").addEventListener("click", () => fileInput.click());
document.getElementById("ie-folder").addEventListener("click", () => folderInput.click());

// ---- Live View / Canvas controls (Design §5/6) ----
const stageEl = document.querySelector(".stage");
document.getElementById("lv-capture").addEventListener("click", () => captureFrame());
document.getElementById("lv-original").addEventListener("click", () => setCameraMode("original"));
document.getElementById("lv-pose").addEventListener("click", () => setCameraMode("pose"));
const annotateBtn = document.getElementById("lv-annotate");
annotateBtn.addEventListener("click", () => {
  if (annotateBtn.disabled) return;
  if (annotateCanvas.isActive()) annotateCanvas.exit();
  else annotateCanvas.enter();
});
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
document.getElementById("cv-fit").addEventListener("click", () => editor.fitView());
// เมนู "มุมมอง ▾": preset สัดส่วนจอ + ความกว้างแผง (ทางเลือกที่ไม่ต้องลาก) · เปลี่ยนแล้ววัด canvas ใหม่ — resize() จะ fit ให้เองถ้ามุมมองยังไม่ถูกซูม/เลื่อน
initViewMenu(document.getElementById("cv-view-menu"), { onChange: () => editor.resize() });
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
let lastSourceInfoSig = "";
setInterval(() => {
  const mode = state.paused ? "หยุด"
    : state.source === "camera"
    ? (state.cameraMode === "pose" ? "Pose" : "สด")
    : state.source === "video" ? "วิดีโอ" : "ภาพนิ่ง";
  lvModeEl.textContent = mode;
  syncWireFlow(); // กันสถานะหลุด (กล้องดับ/สลับ source ทางอื่น) — ไม่ทำอะไรถ้าค่าไม่เปลี่ยน
  // จุดเขียวเฉพาะภาพสด — ภาพนิ่ง/วิดีโอ/หยุด ใช้จุดเทา (ไม่พึ่งสีอย่างเดียว: ข้อความโหมดอยู่ข้างจุดเสมอ)
  lvModeEl.parentElement.classList.toggle("is-live", mode === "สด" || mode === "Pose");
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
  // ปุ่มตีกรอบ: โผล่เฉพาะโหมดภาพนิ่ง — ปิดใช้งาน (พร้อม tooltip อธิบาย) ถ้ายังไม่มีภาพ
  const isStillImage = state.source === "image";
  annotateBtn.style.display = isStillImage ? "" : "none";
  const noImages = state.images.length === 0;
  annotateBtn.disabled = noImages;
  annotateBtn.title = noImages ? "นำเข้าภาพก่อนถึงจะตีกรอบได้" : "ตีกรอบ (bounding box) สำหรับ dataset";
  annotateBtn.classList.toggle("active", annotateCanvas.isActive());
  updatePerf(fps);

  // การ์ดข้อมูลภาพ: ไม่มีแหล่งภาพเลย → ซ่อนทั้งใบ
  const isCamera = state.source === "camera";
  const isVideo = state.source === "video";
  const hasStillImage = state.source === "image" && state.images.length > 0;
  const hasSource = (isCamera && cameraStarted) || isVideo || hasStillImage;
  liInfoEl.classList.toggle("hidden", !hasSource);
  if (hasSource) {
    // ป้ายโหมดอยู่ที่ pill แล้ว → ช่องนี้บอกชื่อไฟล์ (บรรทัดเดียวในหัว VIEWER)
    liSourceEl.textContent = isCamera ? "กล้อง" : isVideo ? (videoFile ? videoFile.name : "วิดีโอ")
      : (state.images[state.imageIndex] ? state.images[state.imageIndex].name : "ไฟล์ภาพ");
    if (isCamera && w) liResEl.textContent = `${w}×${h}`;
    else if (hasStillImage) {
      const im = state.images[state.imageIndex];
      liResEl.textContent = im && im.w && im.h ? `${im.w}×${im.h}` : "—";
    } else liResEl.textContent = "—";
    // FPS มีความหมายเฉพาะภาพสด (กล้อง/วิดีโอ) — ภาพนิ่งไม่มีเฟรมต่อวินาทีให้ดู
    liFpsRowEl.classList.toggle("hidden", !isCamera);
    if (isCamera) liFpsEl.textContent = state.paused ? "0" : String(fps);
    liColorEl.textContent = "สี (RGB)";
  }
  // Source node บนกระดานชื่อตามโหมดจริง (กล้อง/ภาพนิ่ง/วิดีโอ) — ส่งเฉพาะตอนค่าเปลี่ยน
  {
    let detail = "";
    if (isCamera) {
      const port = state.cameras.findIndex((c) => c.deviceId === state.cameraId) + 1;
      detail = sourceDetail({ w, h, port });
    } else if (isVideo) {
      detail = sourceDetail({ name: videoFile && videoFile.name, w, h });
    } else if (hasStillImage) {
      const im = state.images[state.imageIndex];
      detail = im ? sourceDetail({ name: im.name, w: im.w, h: im.h }) : "";
    }
    const kind = isCamera ? "camera" : isVideo ? "video" : "image";
    const sig = kind + "|" + detail;
    if (sig !== lastSourceInfoSig) {
      lastSourceInfoSig = sig;
      editor.setSourceInfo({ kind, detail });
    }
  }
  renderCameraControl();
}, 1000);

// ---------- Performance panel (Design §17) ----------
const statusFpsEl = document.getElementById("status-fps");
const statusCpuEl = document.getElementById("status-cpu");
const statusRamEl = document.getElementById("status-ram");
const statusGpuEl = document.getElementById("status-gpu");
function updatePerf(fps) {
  if (statusFpsEl) {
    statusFpsEl.textContent = "FPS " + fps;
    statusFpsEl.classList.toggle("hidden", state.source !== "camera"); // FPS มีความหมายเฉพาะภาพสด
  }
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
// 3.5 สายวิ่งบนกระดาน: เฉพาะกล้องสดกำลังเดิน (ไม่ pause, โหมดต้นฉบับ) และผู้ใช้ไม่ได้ตั้งลดแอนิเมชัน — editor ไม่มี loop ต่อเนื่องเมื่อปิด
const reducedMotionMql = typeof matchMedia === "function" ? matchMedia("(prefers-reduced-motion: reduce)") : null;
function syncWireFlow() {
  editor.setFlow(flowActive({
    source: state.source, cameraMode: state.cameraMode, cameraStarted, paused: state.paused,
    reducedMotion: !!(reducedMotionMql && reducedMotionMql.matches),
  }));
}
function togglePause() {
  state.paused = !state.paused;
  syncWireFlow();
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
  refreshDatasetPanels({ resetSplit: true });
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
let detPendingArgs = null; // มีคำขอ detect ค้างระหว่าง detect รอบก่อนยังไม่เสร็จ (เช่น ลากสไลเดอร์ conf เร็ว ๆ) → เก็บ src/w/h ล่าสุดไว้รันซ้ำ
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
      cell.lastImg = null; // จอ DL วาดผ่าน drawImage — inspector อ่านพิกเซลจาก canvas ตรง ๆ
      cell.insp.onFrame();
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
  if (detBusy) {
    // detect รอบก่อนยังไม่เสร็จ → เก็บ args ล่าสุดไว้ รันซ้ำทันทีที่ว่าง (กันค้างที่ conf ค่ากลางทางตอนลากสไลเดอร์เร็ว)
    detPendingArgs = { src, w, h };
    return;
  }
  detBusy = true;
  dlDetect(src, w, h, { labels: state.labels, conf: state.detConf })
    .then((bs) => {
      detBoxes = bs;
      drawDLCells(connectedDLDisplays(), src, w, h); // วาดกล่องใหม่ (สำคัญสำหรับภาพนิ่ง)
    })
    .catch((e) => console.error("[YOLO detect]", e))
    .finally(() => {
      detBusy = false;
      if (detPendingArgs) {
        const { src: ps, w: pw, h: ph } = detPendingArgs;
        detPendingArgs = null;
        renderDLDisplays(ps, pw, ph); // อ่าน state.detConf ล่าสุดตอนรันจริง จึงลงเอยที่ค่าสไลเดอร์ล่าสุดเสมอ
      }
    });
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
    const { inputSize, names, seg, kind: modelKind } = await dlLoadModel(bytes);
    // ชื่อคลาส: จากในโมเดลก่อน (YOLO ฝังให้) · ถ้าไม่มี (เช่น RF-DETR/.onnx เปล่า) ใช้คลาสของ dataset
    // ที่ผู้ใช้สร้างไว้ในแอป (เช่น "Defect") · สุดท้ายค่อย fallback COCO — จะได้ไม่ต้องโหลดไฟล์ชื่อคลาสเอง
    const datasetNames = datasetGetClasses().map((c) => c.name);
    state.labels = names && names.length ? names : datasetNames.length ? datasetNames : COCO;
    const hasDL = connectedDLDisplays().length > 0;
    const kind = modelKind === "rfdetr" ? "RF-DETR" : seg ? "segmentation (วาด mask)" : "detection";
    statusEl.textContent = hasDL
      ? `โมเดล ${kind} พร้อม (${file.name}, ${state.labels.length} คลาส) — ดูผลในกล่องผลลัพธ์ที่ต่อสายผ่านกล่อง DL`
      : `โมเดล ${kind} พร้อม (${file.name}, ${state.labels.length} คลาส) — ต่อสาย แหล่งภาพ → กล่อง YOLO → ผลลัพธ์`;
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
  // COCO json (_annotations.coco.json ของ RF-DETR) — อ่านชื่อจาก categories
  // ตัด supercategory id 0 ("objects"/"none") ทิ้ง แล้วเรียงตาม id → ชื่อคลาสตรง index โมเดล
  if (/\.json$/i.test(name)) {
    try {
      const d = JSON.parse(text);
      const cats = Array.isArray(d.categories) ? d.categories : [];
      const real = cats
        .filter((c) => c && typeof c.name === "string" && !(c.id === 0 && /^(objects|none|background)$/i.test(c.name)))
        .sort((a, b) => (a.id || 0) - (b.id || 0));
      return real.map((c) => c.name);
    } catch {
      return [];
    }
  }
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

// เคลียร์ภาพที่นำเข้าทั้งหมด (เริ่ม dataset ชุดใหม่) — ไม่ลบป้ายกำกับ/คลาส (เก็บใน localStorage ตาม key ชื่อภาพ)
function clearImages() {
  if (state.images.length === 0) return;
  const n = state.images.length;
  if (!window.confirm(`ลบภาพที่นำเข้าทั้งหมด ${n} ภาพ?\n(ป้ายกำกับ/คลาสไม่ถูกลบ — นำเข้าภาพชุดใหม่ได้เลย)`)) return;
  if (annotateCanvas.isActive()) annotateCanvas.exit();
  state.images = [];
  decodeCache.clear();
  state.imageIndex = 0;
  updateImageEmpty();
  sourceBar.render();
  refreshDatasetPanels();
  statusEl.textContent = `เคลียร์ภาพแล้ว (${n} ภาพ) — นำเข้า dataset ชุดใหม่ได้เลย`;
}

function importFiles(fileList) {
  const incoming = Array.from(fileList || []).filter((f) => f.type.startsWith("image/"));
  if (incoming.length === 0) {
    statusEl.textContent = "ไม่พบไฟล์ภาพในสิ่งที่เลือก";
    return;
  }
  // นำเข้าแบบ "เพิ่ม" (ไม่ลบภาพเก่า) — กันชื่อซ้ำด้วยชื่อไฟล์ (เป็น key ของ annotation อยู่แล้ว)
  const hadImages = state.images.length > 0;
  const currentName = hadImages ? state.images[state.imageIndex]?.name : null;
  const existing = new Set(state.images.map((im) => im.name));
  const fresh = incoming.filter((f) => {
    if (existing.has(f.name)) return false;
    existing.add(f.name);
    return true;
  });
  const dup = incoming.length - fresh.length;
  if (fresh.length === 0) {
    statusEl.textContent = `ภาพทั้ง ${incoming.length} มีอยู่แล้ว — ไม่ได้เพิ่ม`;
    return;
  }
  // ต่อท้ายแล้วเรียงใหม่ทั้งชุด (index เปลี่ยน → ล้าง decodeCache ที่ key ด้วย index กันชี้ผิดภาพ)
  state.images = state.images.concat(fresh.map((f) => ({ file: f, name: f.name })));
  state.images.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  decodeCache.clear();
  // คงภาพที่กำลังดูไว้ (หา index ใหม่หลัง sort) — ถ้าเพิ่งเริ่ม ให้เริ่มภาพแรก
  if (hadImages && currentName) {
    const ni = state.images.findIndex((im) => im.name === currentName);
    state.imageIndex = ni >= 0 ? ni : 0;
  } else {
    state.imageIndex = 0;
  }
  refreshDatasetPanels({ resetSplit: !hadImages }); // เพิ่มภาพ = ไม่รีเซ็ต split เดิม
  state.source = "image";
  gridEl.classList.remove("hidden");
  cameraError.classList.add("hidden");
  updateImageEmpty();
  sourceBar.render();
  const total = state.images.length;
  statusEl.textContent = hadImages
    ? `เพิ่ม ${fresh.length} ภาพ${dup ? ` (ข้ามซ้ำ ${dup})` : ""} — รวม ${total} ภาพ`
    : total > 1
      ? `นำเข้า ${total} ภาพแล้ว — แสดงภาพ 1/${total} (กด ▶ ดูภาพถัดไป · Export = ประมวลผลทุกภาพ)`
      : `นำเข้าภาพแล้ว — ${state.images[0].name}`;
  requestProcess();
}

// นำเข้าภาพ/วิดีโอ (ปุ่มเดียว): ถ้าเลือกวิดีโอ → เด้ง popup แยกเฟรม, ถ้าเป็นภาพ → นำเข้าปกติ
const isVideoFile = (f) => f.type.startsWith("video/") || /\.(mp4|mov|avi|mkv|webm|m4v|ts)$/i.test(f.name);
let datasetImportPending = false;
fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files || []);
  fileInput.value = "";
  const archives = files.filter((f) => /\.zip$/i.test(f.name));
  if (archives.length) {
    if (files.length !== 1) {
      statusEl.textContent = "นำเข้า Dataset ZIP ครั้งละหนึ่งไฟล์ แยกจากภาพ/วิดีโอ";
      return;
    }
    if (exporting || datasetExportPending || datasetImportPending) return;
    datasetImportPending = true;
    try {
      statusEl.textContent = "กำลังอ่าน YOLO / Roboflow Dataset…";
      const imported = await importYoloDataset(archives[0]);
      const names = datasetGetClasses().map((c) => c.name);
      const mapping = imported.classNames.map((name) => {
        let idx = names.indexOf(name);
        if (idx < 0) { idx = datasetAddClass(name); names.push(name); }
        return idx;
      });
      const used = new Set(state.images.map((im) => im.name.toLowerCase()));
      const newFiles = imported.images.map((im) => {
        const base = im.file.name.replace(/\.[^.]+$/, "");
        const extension = im.file.name.slice(base.length);
        let name = im.file.name, suffix = 2;
        while (used.has(name.toLowerCase()) || datasetGetAnnotations(name).length) {
          name = `${base}_import${suffix++}${extension}`;
        }
        used.add(name.toLowerCase());
        datasetSetAnnotations(name, im.annotations.map((a) => ({ ...a, classIdx: mapping[a.classIdx] })));
        datasetSetSplit(name, im.split);
        return new File([im.file], name, { type: im.file.type });
      });
      importFiles(newFiles);
      datasetSplitConfirmed = true;
      refreshDatasetPanels();
      statusEl.textContent = `นำเข้า YOLO / Roboflow ${newFiles.length} ภาพ พร้อม Label และ Split` +
        (imported.warnings.length ? ` · ${imported.warnings.length} ภาพไม่มีไฟล์ Label` : "");
    } catch (err) {
      statusEl.textContent = `นำเข้า Dataset ไม่สำเร็จ: ${err.message || err}`;
    } finally { datasetImportPending = false; }
    return;
  }
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
    refreshDatasetPanels({ resetSplit: true });
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
  const preview = selectedPreviewPipeline();
  // ภาพขาเข้าต้องมาก่อน preview ในข้อความเดียวกัน · กล้องสดขอไม่ถี่กว่า INPUT_HIST_INTERVAL_MS (ดู pickWorkerPipelines)
  return pickWorkerPipelines({
    displays: state.pipelines.filter((d) => d.connected && !d.dl),
    preview,
    input: preview ? selectedInputPipeline() : null,
    source: state.source,
    inputReady: selectedInputReady,
    inputAt: selectedInputAt,
    now: performance.now(),
  });
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
  } else if (state.source === "video" && sourceVideoEl.readyState >= 2 && !sourceVideoEl.paused && !sourceVideoEl.ended) {
    // โหมดวิดีโอ: ป้อนเฟรมที่กำลังเล่นเข้า pipeline + DL detect เหมือนกล้องสด
    const nw = sourceVideoEl.videoWidth;
    const nh = sourceVideoEl.videoHeight;
    if (nw > 0 && nh > 0) {
      ensureGrid(connectedDisplayIds());
      updateCaptions();
      const { w, h } = processDims(nw, nh);
      scratchCanvas.width = w;
      scratchCanvas.height = h;
      const ctx = scratchCanvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(sourceVideoEl, 0, 0, w, h);
      sendFrame(ctx.getImageData(0, 0, w, h));
      renderDLDisplays(sourceVideoEl, nw, nh); // กล่องที่ต่อผ่าน DL → detect บนเฟรมวิดีโอ
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
    ? "c|" + sourceChipLabel(state.source) + "|" + dp.pipeline.map((s) => s.nodeId + ":" + stepLabel(s) + ":" + JSON.stringify(stepRawParams(s))).join("|")
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
  src.textContent = sourceChipLabel(state.source); // ภาพนิ่ง / กล้อง / วิดีโอ ตามโหมด (เดิมตายตัวว่า "ภาพ")
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

    // 3.2 ภาพขาเข้าของกล่องที่เลือก → เก็บเฉพาะ histogram (ไม่วาดที่ไหน) · ผลของรุ่นเก่า (generation ไม่ตรง) ทิ้ง
    const kind = classifyResultId(displayId, selectedInputPipeline()?.displayId, selectedPreviewPipeline()?.displayId);
    if (kind === "stale") continue; // ผลของรุ่นเก่า/กล่องที่เลิกเลือกแล้ว — ห้ามย้อนมาทับ
    if (kind === "input") {
      selectedInputSamples = accumulateInputHistogram(img.data, width * height);
      selectedInputReady = true;
      selectedInputAt = performance.now();
      continue;
    }
    if (kind === "preview") {
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
      cell.lastImg = img; // ให้ pixel inspector อ่านค่าตรงจากข้อมูลเฟรมนี้ (ไม่ต้อง getImageData ต่อการขยับเมาส์)
      cell.insp.onFrame();
    }
  }
  // มีผลลัพธ์ filter ใหม่เข้ามา — ถ้าอยู่ในโหมดตีกรอบและกำลังดูพื้นหลังแบบ filtered ให้รีเฟรชพื้นหลังทันที
  // (กันเคส getFilteredCanvas() คืน null ตอนสลับไปแล้วแต่ pipeline เพิ่งประมวลผลเสร็จทีหลัง)
  annotateCanvas?.onFilteredResult?.();
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

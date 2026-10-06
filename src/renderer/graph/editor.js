// Node-graph editor สไตล์ Node-RED (ครอบ LiteGraph.js)
// ผู้เรียนลาก node มาต่อสาย: [แหล่งภาพ] → filter → filter → [ผลลัพธ์]
// รองรับ "ผลลัพธ์" ได้หลายกล่อง — แต่ละกล่องสกัด pipeline ของตัวเอง (ย้อนจากกล่องนั้น
// กลับไปหาแหล่งภาพ) และวาด preview ของสายตัวเองในกล่อง
import { LiteGraph, LGraph, LGraphCanvas } from "litegraph.js";
import "litegraph.js/css/litegraph.css";
import { FILTERS, defaultParams, GROUP_COLORS, migrateLegacyStep } from "../filters/registry.js";
import { computeFit, findFreeSpot, canvasBufferSize, applyFit, isViewUntouched, planInsert, formatZoom } from "./layout-math.js";
import { GROUP_ICON, OP_PRIMARY, currentOpId, localParamsFor, hexA } from "../filters/ui-meta.js";

// ---- node card (Design node-card §1 + UI redesign spec §5) — const/font hoisted ไป module scope กัน alloc ทุกเฟรม ----
// ฟอนต์ canvas ต้องมี fallback เสมอ (เดิมไม่มี → เบราว์เซอร์ตกไป serif) — ฟอนต์จริงโหลดจาก @font-face ใน app.css
const NC_STACK = "'IBM Plex Sans Thai', 'Leelawadee UI', 'Noto Sans Thai', 'Segoe UI', Tahoma, sans-serif";
const NC_FONT_TITLE = `600 13px ${NC_STACK}`;
const NC_FONT_LINE1 = `500 12px ${NC_STACK}`;
const NC_FONT_PORT = `500 11px ${NC_STACK}`;
const NC_FONT_MONO = `500 11px 'IBM Plex Mono', Consolas, ${NC_STACK}`;
const NC_FONT_ICON = `14px 'Segoe UI Emoji', 'Segoe UI Symbol', ${NC_STACK}`;
const NC_FONT_PILL = `500 10.5px ${NC_STACK}`;
const NC_ICON_FALLBACK = "🧩";
const NC_W = 192; // ความกว้างการ์ดทุกชนิด
const NC_RADIUS = 10;
const NC_FOCUS = "#8cc8ff"; // ring ตอนเลือก (ไม่ใช้สีหมวด — Segmentation แดงจะดูเหมือน error)
const NC_BODY = "#0f1b2b";
const NC_TITLE_BASE = "#132235";

// ผสมสี hex: a*t + b*(1-t) → ได้สีทึบ (ไม่โปร่ง ไม่ให้สายเห็นทะลุหัวการ์ด)
function mixHex(a, b, t) {
  const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [x, y] = [p(a), p(b)];
  return "#" + x.map((v, i) => Math.round(v * t + y[i] * (1 - t)).toString(16).padStart(2, "0")).join("");
}

// แถบสีหมวด 3px ด้านบนหัวการ์ด + กรอบ/ring ตามสถานะ (default / hover / selected)
function drawCardChrome(ctx, node) {
  const W = node.size[0];
  const H = node.size[1];
  const TH = LiteGraph.NODE_TITLE_HEIGHT;
  const sel = !!node.is_selected;
  const hov = !!node.mouseOver && !sel;
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(0, -TH, W, H + TH, NC_RADIUS);
  ctx.clip();
  ctx.fillStyle = node.boxcolor || "#2997ff";
  ctx.fillRect(0, -TH, W, 3);
  ctx.restore();
  ctx.save();
  if (sel) {
    ctx.strokeStyle = NC_FOCUS;
    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(140,200,255,0.45)";
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.roundRect(-3, -TH - 3, W + 6, H + TH + 6, NC_RADIUS + 3);
  } else {
    ctx.strokeStyle = `rgba(255,255,255,${hov ? 0.24 : 0.1})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(0.5, -TH + 0.5, W - 1, H + TH - 1, NC_RADIUS);
  }
  ctx.stroke();
  ctx.restore();
}

// จุดสถานะที่หัวการ์ด: อยู่ใน pipeline = จุดเขียวทึบ, ไม่อยู่ = วงกลวง (ไม่พึ่งสีอย่างเดียว), ปิดอยู่ = จุดเทา
// __mvInPipeline ตั้งใน tracePipeline (เฉพาะฝั่ง editor — ไม่แตะ pipeline ที่ส่งให้ worker)
function drawStatusDot(ctx, node, disabled) {
  const cx = node.size[0] - 14;
  const cy = -LiteGraph.NODE_TITLE_HEIGHT / 2;
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, 3, 0, Math.PI * 2);
  if (disabled) {
    ctx.fillStyle = "#64748b";
    ctx.fill();
  } else if (node.__mvInPipeline) {
    ctx.fillStyle = "#27c98a";
    ctx.fill();
  } else {
    ctx.strokeStyle = "#8592a4";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  ctx.restore();
}

// สีชื่อหัวการ์ด: LiteGraph อ่าน constructor.title_text_color ทันทีหลังเรียก onDrawTitleText ของ node นั้น
// → ตั้งค่าตามสถานะ node ก่อนทุกครั้ง (กล่องที่ไม่อยู่ใน pipeline = #a7b2c3 เป๊ะ, ปกติ = null → ใช้สีเริ่มต้น)
function titleColorHook() {
  const dim = !this.__mvInPipeline && this.mode !== LiteGraph.NEVER;
  this.constructor.title_text_color = dim ? "#a7b2c3" : null;
}

// ชื่อหัวการ์ดที่ตัดวงเล็บ + ตัด … ให้พอดีก่อนถึงจุดสถานะ (กว้าง − 28 ซ้าย − 24 ขวา)
let titleMeasureCtx = null;
function fitTitleOf(node) {
  const raw = String(node.title == null ? "" : node.title);
  if (!titleMeasureCtx) titleMeasureCtx = document.createElement("canvas").getContext("2d");
  titleMeasureCtx.font = NC_FONT_TITLE;
  const stripped = raw.replace(/\s*\([^)]*\)\s*/g, " ").trim() || raw;
  return ellipsisText(titleMeasureCtx, node, "title", stripped, node.size[0] - 28 - 24);
}
function installCardHooks(Ctor) {
  Ctor.prototype.onDrawTitleText = titleColorHook;
  Ctor.prototype.getTitle = function () { return fitTitleOf(this); };
}

// ลายเฉียงจาง ๆ บนการ์ดที่ปิดอยู่ + ป้าย "ปิดอยู่"
function drawDisabledMark(ctx, node) {
  const W = node.size[0];
  const H = node.size[1];
  const TH = LiteGraph.NODE_TITLE_HEIGHT;
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(0, -TH, W, H + TH, NC_RADIUS);
  ctx.clip();
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = -H - TH; x < W; x += 6) {
    ctx.moveTo(x, H);
    ctx.lineTo(x + H + TH, -TH);
  }
  ctx.stroke();
  ctx.restore();
  ctx.save();
  ctx.font = NC_FONT_PILL;
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "center";
  const pw = 44;
  ctx.fillStyle = "rgba(5,11,20,0.8)";
  ctx.beginPath();
  ctx.roundRect(W - pw - 30, -TH + 6, pw, 16, 8);
  ctx.fill();
  ctx.fillStyle = "#a7b2c3";
  ctx.fillText("ปิดอยู่", W - pw / 2 - 30, -TH + 17.5);
  ctx.restore();
}

// ข้อมูลแหล่งภาพ (สลับตามโหมด) — app.js เรียก setSourceInfo เมื่อเปลี่ยน
const SOURCE_KINDS = {
  camera: { title: "Camera Source", line1: "กล้อง (สด)", icon: "📷" },
  image: { title: "Image Source", line1: "ภาพนิ่ง", icon: "🖼️" },
  video: { title: "Video Source", line1: "วิดีโอ", icon: "🎬" },
};
let sourceInfo = { kind: "camera", detail: "" };
function sourceTitle() {
  return "1. " + (SOURCE_KINDS[sourceInfo.kind] || SOURCE_KINDS.camera).title;
}

// ตัดข้อความให้พอดีความกว้าง (…) — cache ต่อ node กัน measureText ซ้ำทุกเฟรม
function ellipsisText(ctx, node, cacheKey, text, maxWidth) {
  if (!node.__mvTextCache) node.__mvTextCache = {};
  const cacheK = cacheKey + "|" + text + "|" + maxWidth;
  const cached = node.__mvTextCache[cacheKey];
  if (cached && cached.k === cacheK) return cached.v;
  let out = text;
  if (ctx.measureText(text).width > maxWidth) {
    let lo = 0, hi = text.length;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (ctx.measureText(text.slice(0, mid) + "…").width <= maxWidth) lo = mid;
      else hi = mid - 1;
    }
    out = text.slice(0, lo) + "…";
  }
  node.__mvTextCache[cacheKey] = { k: cacheK, v: out };
  return out;
}

const SOURCE_TYPE = "mv/source";
const DISPLAY_TYPE = "mv/display";
const DL_TYPE = "mv/dl-yolo";

let registered = false;

// เก็บ canvas preview ของแต่ละกล่องผลลัพธ์ (keyed by node.id)
const displayImages = new Map();

function registerNodeTypes(onGraphChange) {
  if (registered) return;
  registered = true;

  // ปิดเมนู node ประเภทอื่นที่ติดมากับ LiteGraph — เหลือเฉพาะของเรา
  LiteGraph.clearRegisteredTypes();

  // chip ไอคอน 20x20 (มุมซ้ายของแถวเนื้อหา)
  function drawIconChip(ctx, accent, x, y, icon) {
    ctx.fillStyle = hexA(accent, 0.16);
    ctx.strokeStyle = hexA(accent, 0.5);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x, y, 20, 20, 5);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#f5f7fa";
    ctx.font = NC_FONT_ICON;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(icon, x + 10, y + 15);
    ctx.textAlign = "left";
  }

  // --- node แหล่งภาพ (ชื่อ/ไอคอน/รายละเอียดสลับตามโหมด — ดู setSourceInfo) ---
  function SourceNode() {
    this.addOutput("ภาพ", "image");
    this.boxcolor = "#2997ff"; // จุดสี Source = Blue (Design §3.1)
    this.color = mixHex(this.boxcolor, NC_TITLE_BASE, 0.18);
    this.bgcolor = NC_BODY;
    this.size = [NC_W, 56];
  }
  SourceNode.title = "📷 แหล่งภาพ";
  SourceNode.prototype.onConnectionsChange = onGraphChange;
  SourceNode.prototype.onDrawForeground = function (ctx) {
    if (this.flags && this.flags.collapsed) return;
    const kind = SOURCE_KINDS[sourceInfo.kind] || SOURCE_KINDS.camera;
    ctx.save();
    drawIconChip(ctx, this.boxcolor, 10, 32, kind.icon);
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    ctx.font = NC_FONT_LINE1;
    ctx.fillStyle = "#f5f7fa";
    ctx.fillText(kind.line1, 38, 41);
    if (sourceInfo.detail) {
      ctx.font = NC_FONT_MONO;
      ctx.fillStyle = "#a7b2c3";
      ctx.fillText(ellipsisText(ctx, this, "src", sourceInfo.detail, this.size[0] - 38 - 10), 38, 53);
    }
    ctx.restore();
    drawStatusDot(ctx, this, false);
    drawCardChrome(ctx, this);
  };
  installCardHooks(SourceNode);
  LiteGraph.registerNodeType(SOURCE_TYPE, SourceNode);

  // --- node แสดงผลลัพธ์ (วาด preview ของสายตัวเองในช่อง well) ---
  function DisplayNode() {
    this.addInput("ภาพ", "image");
    this.size = [NC_W, 96];
    this.boxcolor = "#6d7bff"; // จุดสี Output = Indigo (Design §3.1)
    this.color = mixHex(this.boxcolor, NC_TITLE_BASE, 0.18);
    this.bgcolor = NC_BODY;
  }
  DisplayNode.title = "🖥️ ผลลัพธ์";
  DisplayNode.prototype.onConnectionsChange = onGraphChange;
  DisplayNode.prototype.onRemoved = function () {
    displayImages.delete(this.id);
  };
  // เก็บชื่อที่ผู้ใช้ตั้งเอง (__mvName) ลง serialize → ไม่หายตอน undo/redo (และ save/load ภายหลัง)
  DisplayNode.prototype.onSerialize = function (o) {
    o.mvName = this.__mvName || null;
  };
  DisplayNode.prototype.onConfigure = function (o) {
    if (o && o.mvName !== undefined) this.__mvName = o.mvName;
  };
  // วาดภาพผลลัพธ์ใน well (fit รักษาอัตราส่วน) — แถวพอร์ต "ภาพ" อยู่เหนือ well ไม่ทับกัน
  DisplayNode.prototype.onDrawForeground = function (ctx) {
    if (this.flags && this.flags.collapsed) return;
    const wx = 8, wy = 26, ww = this.size[0] - 16, wh = this.size[1] - wy - 8;
    ctx.save();
    ctx.fillStyle = "#050b14";
    ctx.beginPath();
    ctx.roundRect(wx, wy, ww, wh, 6);
    ctx.fill();
    const img = displayImages.get(this.id);
    if (!img || !img.width) {
      ctx.fillStyle = "#8592a4";
      ctx.font = NC_FONT_PORT;
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      ctx.fillText("ยังไม่มีสายเข้า", wx + ww / 2, wy + wh / 2 + 4);
    } else {
      const ar = img.width / img.height;
      let dw = ww - 2;
      let dh = dw / ar;
      if (dh > wh - 2) {
        dh = wh - 2;
        dw = dh * ar;
      }
      ctx.save();
      ctx.beginPath();
      ctx.roundRect(wx, wy, ww, wh, 6);
      ctx.clip();
      ctx.drawImage(img, wx + (ww - dw) / 2, wy + (wh - dh) / 2, dw, dh);
      ctx.restore();
    }
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(wx + 0.5, wy + 0.5, ww - 1, wh - 1, 6);
    ctx.stroke();
    ctx.restore();
    drawStatusDot(ctx, this, false);
    drawCardChrome(ctx, this);
  };
  installCardHooks(DisplayNode);
  LiteGraph.registerNodeType(DISPLAY_TYPE, DisplayNode);

  // --- node Deep Learning (YOLO Detect) — ต่อสาย แหล่งภาพ → กล่องนี้ → ผลลัพธ์ ---
  // ไม่บังคับ size เอง — ให้ LiteGraph auto-size ตาม slot เพื่อให้จุดต่อ (วาด) ตรงกับพื้นที่คลิก
  function DLNode() {
    this.addInput("เข้า", "image");
    this.addOutput("ออก", "image");
    this.boxcolor = "#ef5bff"; // จุดสี Deep Learning = Pink (Design §3.1)
    this.color = mixHex(this.boxcolor, NC_TITLE_BASE, 0.18);
    this.bgcolor = NC_BODY;
    this.size = [NC_W, 56];
    this.__mvInPipeline = false;
  }
  DLNode.title = "🧠 Model Detect";
  DLNode.prototype.onConnectionsChange = onGraphChange;
  DLNode.prototype.onDrawForeground = function (ctx) {
    if (this.flags && this.flags.collapsed) return;
    const inFlow = !!this.__mvInPipeline;
    ctx.save();
    if (!inFlow) ctx.globalAlpha = 0.7;
    drawIconChip(ctx, this.boxcolor, 10, 32, "🧠");
    ctx.textBaseline = "alphabetic";
    ctx.font = NC_FONT_LINE1;
    ctx.fillStyle = mixHex(this.boxcolor, "#ffffff", 0.75);
    ctx.fillText("Deep Learning", 38, 47);
    ctx.restore();
    drawStatusDot(ctx, this, false);
    drawCardChrome(ctx, this);
  };
  installCardHooks(DLNode);
  LiteGraph.registerNodeType(DL_TYPE, DLNode);

  // --- node ของ filter แต่ละตัวจาก registry (ลงสีตามกลุ่ม) ---
  for (const [id, def] of Object.entries(FILTERS)) {
    const colors = GROUP_COLORS[def.group] || { title: "#475569", body: "#1e293b" };
    const tint = mixHex(colors.title, "#ffffff", 0.75); // สีตัวอักษรหมวดในตัวการ์ด (อ่านง่ายบนพื้นเข้ม)
    function FilterNode() {
      this.addInput("เข้า", "image");
      if (def.inputs === 2) this.addInput("เข้า B", "image"); // Tier 2 §A.1 — เฉพาะ op ที่รับภาพ 2 ใบ
      this.addOutput("ออก", "image");
      this.properties = defaultParams(id);
      this.filterId = id;
      // หัวการ์ด = สีหมวดผสมกับพื้นเข้มแบบทึบ (ไม่โปร่ง สายไม่เห็นทะลุ) + แถบสีหมวด 3px ใน drawCardChrome
      this.color = mixHex(colors.title, NC_TITLE_BASE, 0.18);
      this.bgcolor = NC_BODY;
      this.boxcolor = colors.title; // จุดสีประจำหมวดที่หัว node
      this.size = [NC_W, 56]; // แถวพอร์ต + แถวเนื้อหา (หมวด + ค่าหลัก)
      this.__mvInPipeline = false;
    }
    // ชื่อบนกระดานตัดวงเล็บอังกฤษออก ให้สั้น ไม่ล้น (palette ยังใช้ชื่อเต็ม)
    FilterNode.title = def.name.replace(/\s*\([^)]*\)\s*/g, "").trim();
    FilterNode.prototype.filterId = id;
    FilterNode.prototype.onConnectionsChange = onGraphChange;
    FilterNode.prototype.onPropertyChanged = onGraphChange;
    // การ์ดการทดลอง: icon chip + ชื่อหมวด (ซ้าย) + ค่าหลักที่ตั้งไว้ (ขวา, mono)
    FilterNode.prototype.onDrawForeground = function (ctx) {
      if (this.flags && this.flags.collapsed) return;
      const disabled = this.mode === LiteGraph.NEVER;
      const inFlow = !!this.__mvInPipeline;
      const W = this.size[0];
      const accent = this.boxcolor;

      ctx.save();
      if (disabled) ctx.globalAlpha = 0.45;
      else if (!inFlow) ctx.globalAlpha = 0.7;

      drawIconChip(ctx, accent, 10, 32, GROUP_ICON[this.filterId] || NC_ICON_FALLBACK);

      // op/method ปัจจุบัน → ค่าหลัก (ถ้ามี) ชิดขวาแบบ mono
      const opId = currentOpId(this.filterId, this.properties);
      const primaryFn = OP_PRIMARY[opId];
      let value = "";
      try {
        if (primaryFn) value = primaryFn(localParamsFor(this.filterId, this.properties)) || "";
      } catch { value = ""; }
      const opDef = FILTERS[this.filterId];
      const nameText = opDef ? opDef.name : this.filterId;
      ctx.textBaseline = "alphabetic";
      // แบ่งความกว้างแถวเนื้อหา: ชื่อหมวดได้ที่ก่อน (สูงสุด 92px) ที่เหลือให้ค่าหลัก — ไม่ให้ฝั่งใดถูกตัดเหลือแต่ "…"
      const rowW = W - 38 - 10;
      ctx.font = NC_FONT_LINE1;
      const nameBudget = value ? Math.min(ctx.measureText(nameText).width, 92) : rowW;
      let valueW = 0;
      if (value) {
        ctx.font = NC_FONT_MONO;
        const v = ellipsisText(ctx, this, "v", value, Math.max(30, rowW - nameBudget - 8));
        valueW = ctx.measureText(v).width;
        ctx.textAlign = "right";
        ctx.fillStyle = "#a7b2c3";
        ctx.fillText(v, W - 10, 47);
      }
      ctx.textAlign = "left";
      ctx.font = NC_FONT_LINE1;
      ctx.fillStyle = tint;
      ctx.fillText(ellipsisText(ctx, this, "l1", nameText, W - 38 - 10 - (valueW ? valueW + 8 : 0)), 38, 47);
      ctx.restore();

      if (disabled) drawDisabledMark(ctx, this);
      drawStatusDot(ctx, this, disabled);
      drawCardChrome(ctx, this);
    };
    installCardHooks(FilterNode);
    LiteGraph.registerNodeType("filters/" + id, FilterNode);
  }
}

// cb: onPipelineChange(pipelines), onSelect(nodeInfo|null)
//   pipelines = [{ displayId, pipeline, connected }]  (หนึ่งรายการต่อหนึ่งกล่องผลลัพธ์)
//   nodeInfo  = { filterId, getParams(), setParam(key, value) }
export function createGraphEditor(canvasEl, cb) {
  let notify = () => {};
  registerNodeTypes(() => notify());

  const graph = new LGraph();
  const canvas = new LGraphCanvas(canvasEl, graph);
  canvas.allow_searchbox = false; // กันเมนูซับซ้อนเกินจำเป็นสำหรับมือใหม่
  // ต่อสายง่ายขึ้นสำหรับมือใหม่: จุด/เส้นใหญ่ขึ้น + ระยะจับ (hit-area) กว้างขึ้น
  canvas.connections_width = 2; // เส้นบางลงตาม Design §12 (~2-3px) แต่ยังจับง่าย
  canvas.render_connection_arrows = false;
  canvas.render_canvas_border = false; // กรอบ "ขนาดจอ" ที่ origin กราฟ — เดิมซ่อนอยู่ที่ขอบ canvas แต่พอ Fit เลื่อนมุมมองจะเห็นเป็นสี่เหลี่ยมลอย
  canvas.round_radius = 10; // มุมมนการ์ด node (Design §11.1 radius-md)
  canvas.background_image = null;
  canvas.show_info = false; // ปิด debug overlay (T/I/N/V/FPS ซ้ายล่าง) — เหลือ FPS ประมวลผลจริงอันเดียว
  // ป้ายซูมในหัวกระดาน: ทุกการซูม (ล้อเมาส์/ปุ่ม/Fit) ทำให้ canvas วาดใหม่ → อ่าน scale ตอนวาดพื้นหลังแล้วแจ้งเมื่อ % เปลี่ยน
  let lastZoomText = "";
  canvas.onDrawBackground = () => {
    const t = formatZoom(canvas.ds && canvas.ds.scale);
    if (t !== lastZoomText) { lastZoomText = t; if (cb.onZoomChange) cb.onZoomChange(t); }
  };
  // สีพื้นกระดาน + เส้นกริดให้เข้าโทน dark-industrial (Design §3.1)
  canvas.clear_background_color = "#08111d"; // = --surface-canvas
  LiteGraph.NODE_TITLE_HEIGHT = 28; // เดิม 30 — หัวการ์ดกระชับ (hit-test ของ litegraph อ่านค่านี้ตอนรันจึงตามอัตโนมัติ)
  LiteGraph.NODE_TITLE_TEXT_Y = 19;
  LiteGraph.NODE_DEFAULT_BGCOLOR = NC_BODY;
  LiteGraph.NODE_DEFAULT_COLOR = NC_TITLE_BASE;
  LiteGraph.NODE_TITLE_COLOR = "#f5f7fa";
  LiteGraph.NODE_SELECTED_TITLE_COLOR = "#ffffff";
  LiteGraph.NODE_DEFAULT_BOXCOLOR = "rgba(255,255,255,0.35)";
  LiteGraph.NODE_BOX_OUTLINE_COLOR = "transparent";
  LiteGraph.NODE_TEXT_COLOR = "#8592a4"; // ป้าย "เข้า/ออก/ภาพ" ของพอร์ต
  LiteGraph.DEFAULT_SHADOW_COLOR = "rgba(1,6,14,0.45)";
  LiteGraph.LINK_COLOR = "#4a6285";
  LiteGraph.NODE_SLOT_HEIGHT = 24; // ช่องจุดต่อสายสูงขึ้น = จับง่ายขึ้น
  LiteGraph.link_center_field = 6;
  // ค่าที่ LGraphCanvas คัดลอกจาก LiteGraph ตอน constructor (ตั้งทีหลังไม่มีผล) → ตั้งบน canvas ตรง ๆ
  canvas.default_link_color = "#4a6285";
  canvas.node_title_color = "#f5f7fa";
  canvas.render_connections_border = false;
  canvas.default_connection_color = { input_off: "#6f86a6", input_on: "#58b4ff", output_off: "#6f86a6", output_on: "#58b4ff" };
  canvas.default_connection_color_byType = { image: "#58b4ff" };
  canvas.default_connection_color_byTypeOff = { image: "#6f86a6" };
  // ฟอนต์ของ litegraph เอง (ชื่อหัวการ์ด/ป้ายพอร์ต) — ค่าเดิมคือ Arial
  canvas.title_text_font = NC_FONT_TITLE;
  canvas.inner_text_font = NC_FONT_PORT;
  // ฟอนต์เว็บโหลดแบบ async: เมื่อพร้อมแล้วล้าง cache ตัดข้อความ + วาดใหม่ (ไม่งั้นเฟรมแรกใช้ฟอนต์สำรองค้างอยู่)
  if (document.fonts && document.fonts.ready) {
    const redrawAfterFonts = () => {
      graph._nodes.forEach((n) => { n.__mvTextCache = null; });
      canvas.setDirty(true, true);
    };
    // canvas ไม่ได้ใช้ฟอนต์ผ่าน DOM → สั่งโหลดน้ำหนักที่การ์ดใช้เองก่อน แล้ววาดใหม่
    Promise.all([
      document.fonts.load(NC_FONT_TITLE, "ก"),
      document.fonts.load(NC_FONT_LINE1, "ก"),
      document.fonts.load(NC_FONT_MONO, "0"),
    ]).catch(() => {}).then(() => document.fonts.ready).then(redrawAfterFonts);
  }

  // --- สร้างกราฟตั้งต้น: แหล่งภาพ → ผลลัพธ์ ---
  const src = LiteGraph.createNode(SOURCE_TYPE);
  src.pos = [40, 80];
  graph.add(src);
  const dst = LiteGraph.createNode(DISPLAY_TYPE);
  dst.pos = [260, 80];
  graph.add(dst);
  src.connect(0, dst, 0);

  // เดินย้อนจาก node.inputs[startInputIdx] กลับไปหา SourceNode แบบ recursive — ใช้ร่วมกันทั้ง trace
  // หลัก (input 0 ของ tracePipeline/traceNodePipeline) และ trace ของ input ที่ 2 (สำหรับ op ที่
  // inputs===2, เช่น imabsdiff) รองรับ 2-image op ซ้อน 2-image op ได้ด้วย (src2 ของ src2) เพราะเรียก
  // ตัวเองซ้ำทุกครั้งที่เจอ def.inputs===2 ไม่ว่าจะเจอระหว่าง walk หลักหรือ walk ของ src2 เอง
  //
  // opts.stopAtDL: true = หยุดทันทีที่เจอกล่อง YOLO Detect (พฤติกรรมเดิมของ traceNodePipeline — DL มี
  //   worker คนละตัว ไม่รองรับต่อสายผ่านสำหรับ preview เดี่ยว/สาย src2) false (default) = เดินผ่านต่อไป
  //   ไม่หยุด แค่ไม่เพิ่มลง chain (พฤติกรรมเดิมเป๊ะของ tracePipeline หลัก ซึ่งใช้ hasDL บอกภายหลังว่าต้อง
  //   รัน YOLO detect คั่นกลางด้วย) — ต้องแยก 2 พฤติกรรมนี้ไว้ให้ตรงเดิมทั้งคู่ เพราะเป็น pure refactor
  // opts.collectPathNodes: true = เก็บ pathNodes (ทุก node ที่เดินผ่าน รวม DL ด้วย) ไว้ใช้ตั้งชื่อลำดับขั้น
  //   ใน tracePipeline เท่านั้น — เดินของ src2/traceNodePipeline ไม่ใช้ (pathNodes: [] เฉย ๆ)
  function traceChainFrom(startNode, startInputIdx, opts = {}) {
    const { stopAtDL = false, collectPathNodes = false } = opts;
    const chain = [];
    // NIT (qa, 18a): เดิม `collectPathNodes ? [] : []` เป็น ternary ที่ไม่มีผลต่างจริง (ทั้งสอง branch ได้
    // [] เหมือนกัน) — เปลี่ยนเป็น null เมื่อไม่ต้องเก็บ เพื่อสื่อเจตนาให้ชัดว่า "ไม่ได้ใช้งาน" จริง ๆ (ยืนยัน
    // แล้วว่าปลอดภัย: ทุกจุดที่เรียก traceChainFrom แบบ collectPathNodes:false — src2 recursion และ
    // traceNodePipeline — ไม่เคยอ่าน .pathNodes จากผลลัพธ์เลยสักจุดเดียว ทั้งใน editor.js เอง, cv-worker.js,
    // หรือ codegen.js อ่านแค่ .pipeline/.connected เท่านั้น)
    const pathNodes = collectPathNodes ? [] : null;
    const visited = new Set([startNode.id]); // กัน cycle ย้อนกลับมาที่ startNode เอง (ไม่เคยเกิดในกราฟปกติ)
    let node = startNode;
    let inputIdx = startInputIdx;
    let reachedSource = false;
    let hasDL = false;

    while (node) {
      const inp = node.inputs && node.inputs[inputIdx];
      const linkId = inp && inp.link != null ? inp.link : null;
      if (linkId == null) break;
      const link = graph.links[linkId];
      if (!link) break;
      const prev = graph.getNodeById(link.origin_id);
      if (!prev || visited.has(prev.id)) break;
      visited.add(prev.id);

      if (prev.type === SOURCE_TYPE) {
        reachedSource = true;
        break;
      }
      if (prev.type === DL_TYPE) {
        if (stopAtDL) break; // เหมือนเดิม: DL มี worker คนละตัว ไม่รองรับต่อสายผ่าน (preview/src2)
        hasDL = true; // สายผ่านกล่อง YOLO Detect — พฤติกรรมเดิมของ tracePipeline หลัก เดินต่อได้
        if (collectPathNodes) pathNodes.unshift(prev);
        node = prev;
        inputIdx = 0;
        continue;
      }
      if (collectPathNodes) pathNodes.unshift(prev);
      if (prev.filterId) {
        const def = FILTERS[prev.filterId];
        const step = {
          id: prev.filterId,
          params: { ...prev.properties },
          enabled: prev.mode !== LiteGraph.NEVER,
          nodeId: prev.id,
        };
        if (def && def.inputs === 2) {
          // สาย B ของ prev เอง — เดิน input ที่ 1 (ช่องที่สอง) ของ prev แบบ recursive ผ่านฟังก์ชันเดียวกัน
          // (เดินแบบ conservative เสมอ: หยุดถ้าเจอ DL — ยังไม่มี design รองรับ DL คั่นกลางสาย B รอบนี้)
          step.src2 = traceChainFrom(prev, 1, { stopAtDL: true });
        }
        chain.unshift(step);
      }
      node = prev;
      inputIdx = 0; // หลัง hop แรก เดินตาม input หลัก (0) ของทุก node ที่เจอต่อไปเสมอ
    }

    return {
      pipeline: reachedSource ? chain : [],
      connected: reachedSource,
      hasDL,
      pathNodes,
    };
  }

  // --- สกัด pipeline ของกล่องผลลัพธ์หนึ่งกล่อง (เดินย้อนจากกล่องกลับไปหาแหล่งภาพ) ---
  //   autoNum = ลำดับกล่อง (1-based) ใช้ตั้งชื่ออัตโนมัติ "Display N"
  function tracePipeline(displayNode, autoNum) {
    const { pipeline: chain, connected: reachedSource, hasDL, pathNodes } =
      traceChainFrom(displayNode, 0, { stopAtDL: false, collectPathNodes: true });

    // เลขขั้นตอนอิงเส้นทางจริง: Source = 1, Function = 2..N, Display = N+1
    if (reachedSource) {
      const sourceNode = graph.findNodesByType(SOURCE_TYPE)[0];
      if (sourceNode) {
        sourceNode.title = sourceTitle();
        sourceNode.__mvInPipeline = true;
      }
      displayNode.__mvInPipeline = true;
      pathNodes.forEach((pathNode, i) => {
        pathNode.__mvInPipeline = true; // ใช้วาดจุดสถานะบนการ์ด (ฝั่ง editor เท่านั้น ไม่เข้า pipeline ที่ส่ง worker)
        const label = pathNode.filterId
          ? (typeof FILTERS[pathNode.filterId]?.subtitle === "function"
              ? FILTERS[pathNode.filterId].subtitle(pathNode.properties)
              : FILTERS[pathNode.filterId]?.name)
          : "Model Detect";
        pathNode.title = `${i + 2}. ${label || "Function"}`;
      });
    }
    const displayLabel = displayNode.__mvName || `Display ${autoNum}`;
    const displayTitle = reachedSource ? `${pathNodes.length + 2}. ${displayLabel}` : displayLabel;
    if (displayNode.title !== displayTitle) {
      displayNode.title = displayTitle;
      canvas.setDirty(true, true);
    }

    return {
      displayId: displayNode.id,
      name: displayNode.__mvName || null, // ชื่อที่ผู้ใช้ตั้งเอง (ถ้ามี)
      autoNum, // ลำดับกล่อง (สำหรับชื่อ "Display N")
      pipeline: reachedSource ? chain : [],
      connected: reachedSource,
      dl: reachedSource && hasDL, // ผลลัพธ์นี้ต้องรัน YOLO detect
    };
  }

  // สกัดทุกกล่องผลลัพธ์ (autoNum = ลำดับ 1-based ตามลำดับสร้าง)
  function extractPipelines() {
    // รีเซ็ตสถานะ "อยู่ใน pipeline" ก่อน — tracePipeline ตั้งเป็น true ให้เฉพาะกล่องบนสายที่ต่อถึงผลลัพธ์
    for (const n of graph._nodes) n.__mvInPipeline = false;
    const displays = graph.findNodesByType(DISPLAY_TYPE);
    const out = displays.map((d, i) => tracePipeline(d, i + 1));
    canvas.setDirty(true, true);
    return out;
  }

  // สกัดสายจาก SOURCE มาถึง Function ที่เลือก (รวม Function นั้นด้วย)
  // ใช้สร้าง Output Preview ด้านขวา โดยไม่ต้องเพิ่ม Display node ปลอมลงกราฟ
  // targetNode เองรวมอยู่ใน chain เสมอ (ต่างจาก tracePipeline ที่เริ่มเดินจาก "หน้า" displayNode หนึ่ง
  // hop) — ส่วนที่เหลือ (ย้อนจาก input 0 ของ targetNode กลับไปหาแหล่งภาพ) มอบให้ traceChainFrom เดินแทน
  // ใช้ stopAtDL:true เหมือน traceNodePipeline เดิมเป๊ะ (preview เดี่ยวไม่รองรับสายผ่าน DL)
  function traceNodePipeline(targetNode) {
    const targetStep = targetNode.filterId
      ? {
          id: targetNode.filterId,
          params: { ...targetNode.properties },
          enabled: targetNode.mode !== LiteGraph.NEVER,
          nodeId: targetNode.id,
        }
      : null;
    if (targetStep) {
      const def = FILTERS[targetNode.filterId];
      if (def && def.inputs === 2) targetStep.src2 = traceChainFrom(targetNode, 1, { stopAtDL: true });
    }
    const r = traceChainFrom(targetNode, 0, { stopAtDL: true });
    if (!r.connected) return { connected: false, pipeline: [] };
    return { connected: true, pipeline: targetStep ? [...r.pipeline, targetStep] : r.pipeline };
  }

  // ดึงลำดับ filter ปัจจุบันเป็น "steps" (รูปแบบเดียวกับ applyTemplate) — ใช้เซฟเป็นเทมเพลต
  function exportTemplateSteps() {
    const displays = graph.findNodesByType(DISPLAY_TYPE) || [];
    for (const d of displays) {
      const { pipeline, connected } = tracePipeline(d, 1);
      if (connected && pipeline.length) {
        return pipeline.map((s) => ({ id: s.id, params: { ...s.params } }));
      }
    }
    return [];
  }

  // ---- Undo / Redo (§6): เก็บ snapshot ของกราฟทุกครั้งที่เปลี่ยน ----
  let history = [];
  let histIdx = -1;
  let applyingHistory = false;
  const snapshot = () => JSON.stringify(graph.serialize());
  function recordHistory() {
    if (applyingHistory) return;
    const s = snapshot();
    if (history[histIdx] === s) return; // ไม่มีอะไรเปลี่ยน
    history = history.slice(0, histIdx + 1);
    history.push(s);
    histIdx = history.length - 1;
    if (history.length > 60) { history.shift(); histIdx--; } // จำกัดความลึก
  }
  function applyState(s) {
    applyingHistory = true;
    graph.configure(JSON.parse(s));
    graph.start(); // configure→clear หยุด loop — เปิดใหม่กัน node ที่พึ่ง onExecute เงียบตาย
    applyingHistory = false;
    canvas.setDirty(true, true);
    cb.onSelect(null);
    cb.onPipelineChange(extractPipelines());
  }
  function undo() { if (histIdx > 0) { histIdx--; applyState(history[histIdx]); } }
  function redo() { if (histIdx < history.length - 1) { histIdx++; applyState(history[histIdx]); } }

  let scheduled = false;
  function flushNotify() {
    scheduled = false;
    cb.onPipelineChange(extractPipelines());
    recordHistory();
  }
  notify = () => {
    // debounce เล็กน้อย — การต่อ/ถอดสายหนึ่งครั้งยิง event หลายรอบ
    if (scheduled) return;
    scheduled = true;
    setTimeout(flushNotify, 30);
  };
  // action ที่เป็นก้อนเดียวชัดเจน (เพิ่ม/ลบกล่อง) → บันทึก history ทันที
  // กันการกดเพิ่มรัว ๆ ถูก debounce รวมเป็น undo ก้อนเดียว (แก้ #9)
  function notifyNow() {
    scheduled = false;
    flushNotify();
  }

  // เน้นเส้นเชื่อม (เข้า+ออก) ของ node ที่เลือก → เห็น data flow ง่ายขึ้น (Design §22)
  let highlightedLinkObjs = [];
  function highlightNodeLinks(node) {
    // litegraph เก็บ highlighted_links เป็น object keyed by link id (ไม่ใช่ array)
    // litegraph วาดสาย highlight เป็นสีขาวตายตัว (renderLink) → ไม่ใช้ highlighted_links แต่ตั้ง link.color
    // เป็นสีโฟกัส (#8cc8ff) ให้สายของกล่องที่เลือก แล้วคืนค่าเมื่อเลือกกล่องอื่น
    for (const l of highlightedLinkObjs) l.color = null;
    highlightedLinkObjs = [];
    if (node) {
      const mark = (id) => {
        const l = id != null ? graph.links[id] : null;
        if (l) { l.color = NC_FOCUS; highlightedLinkObjs.push(l); }
      };
      (node.inputs || []).forEach((i) => { if (i) mark(i.link); });
      (node.outputs || []).forEach((o) => { (o && o.links ? o.links : []).forEach(mark); });
    }
    canvas.highlighted_links = {};
    canvas.setDirty(true, true);
  }

  // --- selection → detail panel (filter = ปรับค่า / display = ตั้งชื่อ) ---
  canvas.onNodeSelected = (node) => {
    highlightNodeLinks(node);
    if (node && node.filterId) {
      cb.onSelect({
        kind: "filter",
        id: node.id,
        filterId: node.filterId,
        getParams: () => ({ ...node.properties }),
        getPreviewPipeline: () => traceNodePipeline(node),
        setParam: (key, value) => {
          node.setProperty(key, value);
          notify();
        },
      });
    } else if (node && node.type === DL_TYPE) {
      cb.onSelect({ kind: "dl", id: node.id }); // กล่อง YOLO Detect → แผงขวาโชว์ browse
    } else if (node && node.type === DISPLAY_TYPE) {
      cb.onSelect({
        kind: "display",
        id: node.id,
        getName: () => node.__mvName || "",
        setName: (name) => {
          node.__mvName = name || null;
          node.title = name ? "🖥️ " + name : "🖥️ ผลลัพธ์";
          canvas.setDirty(true, true);
          notify(); // อัปเดต pipeline (ชื่อ) → grid caption ตาม
        },
      });
    } else {
      cb.onSelect(null);
    }
  };
  canvas.onNodeDeselected = () => { highlightNodeLinks(null); cb.onSelect(null); };

  graph.onNodeAdded = () => notify();
  graph.onNodeRemoved = (node) => {
    // ระหว่าง undo/redo (graph.configure ลบ-สร้างใหม่ทั้งกราฟ) → ไม่แตะ selection/auto-recreate
    // มิฉะนั้น source จะถูกสร้างซ้ำชนกับตัวที่ configure กำลัง restore
    if (applyingHistory) return;
    // เคลียร์แผงขวาเสมอ — กันค้างอ้าง node ที่เพิ่งถูกลบ (display/DL/filter ที่กำลังเลือก)
    cb.onSelect(null);
    // แหล่งภาพห้ามหาย (palette ไม่มีปุ่มเพิ่มกลับ) — ถ้าถูกลบจนไม่เหลือ สร้างใหม่ให้อัตโนมัติ
    if (node && node.type === SOURCE_TYPE && graph.findNodesByType(SOURCE_TYPE).length === 0) {
      const newSrc = LiteGraph.createNode(SOURCE_TYPE);
      newSrc.pos = [40, 80];
      graph.add(newSrc);
    }
    notify();
  };

  // --- เพิ่ม filter node จาก palette: วางแบบไล่เฉียง กันวางทับกันเป๊ะเมื่อกดรัว ๆ (แก้ #8) ---
  // จุดกลางของส่วนที่มองเห็นบนกระดาน (พิกัด graph) — กล่องใหม่วางตรงนี้ ไม่หลุดนอกจอเมื่อซูม/เลื่อนไปแล้ว
  function visibleCenter() {
    const ds = canvas.ds;
    const w = canvasEl.width || 600;
    const h = canvasEl.height || 300;
    if (!ds || !ds.convertCanvasToOffset) return [260, 80];
    return ds.convertCanvasToOffset([w / 2, h / 2]);
  }
  // วางกล่องใหม่ใกล้กึ่งกลางส่วนที่มองเห็น แต่ไม่ทับกล่องเดิม (เดิมกล่องแรกทับพอร์ต Source/Display ของกราฟตั้งต้น)
  function nodeRects() {
    const TH = LiteGraph.NODE_TITLE_HEIGHT;
    return graph._nodes.map((n) => ({ x0: n.pos[0], y0: n.pos[1] - TH, x1: n.pos[0] + n.size[0], y1: n.pos[1] + n.size[1] }));
  }
  function placeAtCenter(node) {
    const rects = nodeRects();
    // ขอบเขตที่มองเห็นที่ scale ต่ำสุด (.6) นับจากมุมซ้ายบนของกราฟ — จอกว้างแต่เตี้ย จึงไล่ขวาก่อนแล้วลงแถว
    const minS = 0.6, pad = 40;
    const bounds = rects.length ? {
      x1: Math.min(...rects.map((r) => r.x0)) + (canvasEl.width - 2 * pad) / minS,
      y1: Math.min(...rects.map((r) => r.y0)) + (canvasEl.height - 2 * pad) / minS,
    } : null;
    node.pos = findFreeSpot(rects, visibleCenter(), node.size, LiteGraph.NODE_TITLE_HEIGHT, 16, bounds);
  }
  // กล่องปลายน้ำทั้งหมดของ node (เดินตามสายออกทุกเส้น กันวนด้วย seen) — ใช้เลื่อนไปขวาเพื่อเว้นที่ให้กล่องที่แทรก
  function downstreamOf(node) {
    const seen = new Set([node.id]);
    const out = [];
    const queue = [node];
    while (queue.length) {
      const cur = queue.shift();
      for (const o of cur.outputs || []) {
        for (const lid of (o && o.links) || []) {
          const l = graph.links[lid];
          const t = l ? graph.getNodeById(l.target_id) : null;
          if (t && !seen.has(t.id)) { seen.add(t.id); out.push(t); queue.push(t); }
        }
      }
    }
    return out;
  }
  // "แทรกหลังกล่องที่เลือก": เลือก Source/filter อยู่ 1 กล่อง + คลิกเครื่องมือใน palette → ต่อ selected → ใหม่ → ปลายทางเดิม
  // เงื่อนไขข้าม (คืน null → ผู้เรียกวางที่ว่างแบบเดิม) ตัดสินใน planInsert (layout-math.js)
  function trySpliceAfterSelected(node, def) {
    const ids = Object.keys(canvas.selected_nodes || {});
    if (ids.length !== 1) return null;
    const sel = canvas.selected_nodes[ids[0]];
    if (!sel || !sel.outputs || !sel.outputs[0]) return null;
    const kind = sel.type === SOURCE_TYPE ? "source" : sel.filterId ? "filter" : sel.type === DISPLAY_TYPE ? "display" : "dl";
    const outLinks = (sel.outputs[0].links || []).filter((lid) => graph.links[lid]);
    const down = downstreamOf(sel);
    const directIds = new Set(outLinks.map((lid) => graph.links[lid].target_id));
    const downIds = new Set(down.map((n) => n.id));
    const TH = LiteGraph.NODE_TITLE_HEIGHT;
    const obstacles = graph._nodes
      .filter((n) => n !== sel && !downIds.has(n.id))
      .map((n) => ({ x0: n.pos[0], y0: n.pos[1] - TH, x1: n.pos[0] + n.size[0], y1: n.pos[1] + n.size[1] }));
    const plan = planInsert({
      sel: { kind, x: sel.pos[0], y: sel.pos[1], w: sel.size[0], h: sel.size[1] },
      newNode: { w: node.size[0], h: node.size[1], inputs: def && def.inputs },
      fanOut: outLinks.length,
      downstream: down.map((n) => ({ id: n.id, x: n.pos[0], y: n.pos[1], w: n.size[0], h: n.size[1], direct: directIds.has(n.id) })),
      obstacles, TH,
    });
    if (plan.mode !== "splice") return null;
    return { plan, sel, outLinks, down };
  }
  function addFilterNode(filterId, opId = null) {
    const node = LiteGraph.createNode("filters/" + filterId);
    if (opId && Object.prototype.hasOwnProperty.call(node.properties || {}, "mode")) {
      node.properties.mode = opId;
      const filter = FILTERS[filterId];
      node.title = typeof filter?.subtitle === "function"
        ? filter.subtitle(node.properties)
        : (filter?.name || opId);
    }
    const sp = trySpliceAfterSelected(node, FILTERS[filterId]);
    if (sp) {
      // ทุกอย่างทำแบบ synchronous แล้ว notifyNow() ครั้งเดียวท้ายฟังก์ชัน → history ได้ snapshot เดียว = undo ก้อนเดียว
      const { plan, sel, outLinks } = sp;
      const old = outLinks.map((lid) => graph.links[lid]).map((l) => ({ id: l.target_id, slot: l.target_slot }));
      const originSlot = outLinks.length ? graph.links[outLinks[0]].origin_slot : 0;
      for (const n of sp.down) n.pos[0] += plan.dx;
      node.pos = plan.pos;
      graph.add(node);
      for (const t of old) { // ใหม่ → ปลายทางเดิม (input รับได้เส้นเดียว → แทนสายเดิมของ selected อัตโนมัติ)
        const target = graph.getNodeById(t.id);
        if (target) node.connect(0, target, t.slot);
      }
      sel.connect(originSlot, node, 0);
    } else {
      placeAtCenter(node);
      graph.add(node);
    }
    revealNew(node);
    canvas.selectNodes([node]);
    canvas.onNodeSelected(node);
    notifyNow();
    return node;
  }

  // --- เพิ่มกล่องผลลัพธ์ใหม่ (ยังไม่ต่อสาย ผู้ใช้ลากเอง) ---
  function addDisplayNode() {
    const node = LiteGraph.createNode(DISPLAY_TYPE);
    placeAtCenter(node);
    graph.add(node);
    revealNew(node);
    canvas.selectNodes([node]);
    canvas.onNodeSelected(node); // เปิดแผงขวา (ตั้งชื่อกล่อง) ทันที ให้สอดคล้องกับ filter/DL
    notifyNow();
    return node;
  }

  // --- เพิ่มกล่อง YOLO Detect (Deep Learning) ---
  function addDLNode() {
    const node = LiteGraph.createNode(DL_TYPE);
    placeAtCenter(node);
    graph.add(node);
    revealNew(node);
    canvas.selectNodes([node]);
    canvas.onNodeSelected(node);
    notifyNow();
    return node;
  }

  // --- วางเทมเพลตสำเร็จรูป: แหล่งภาพ → [filter chain] → ผลลัพธ์ (Design §9) ---
  // steps = [{ id, params? }]  (id = filterId, params = ตั้งค่าเริ่ม เช่น {mode:'canny'})
  function applyTemplate(steps) {
    const src = graph.findNodesByType(SOURCE_TYPE)[0];
    if (!src) return;
    // ล้างกล่องเดิม (filter/DL) กันกราฟรกเมื่อวางเทมเพลตซ้ำ — เก็บแหล่งภาพ + ผลลัพธ์ไว้
    for (const n of [...graph._nodes]) {
      if (n.type !== SOURCE_TYPE && n.type !== DISPLAY_TYPE) graph.remove(n);
    }
    let dst = graph.findNodesByType(DISPLAY_TYPE)[0];
    if (!dst) dst = addDisplayNode();
    // เรียงเป็นแถวเดียวซ้าย→ขวา: Source [40,80] → ขั้นตอนต่อกัน → Display อยู่ใต้ขั้นตอนสุดท้าย
    src.pos = [40, 80];
    let prev = src;
    let x = src.pos[0] + NC_W + 28;
    let lastX = null;
    for (const raw of steps) {
      const step = migrateLegacyStep(raw); // เทมเพลตเก่าอ้างกลุ่มเก่า (dropdown ไม่ครบหมวด) → ย้ายไปกลุ่มระดับหมวดที่ครบ
      const n = LiteGraph.createNode("filters/" + step.id);
      if (!n) continue; // กัน id ผิด (createNode คืน null)
      if (step.params) Object.assign(n.properties, step.params);
      n.pos = [x, 80];
      lastX = x;
      x += NC_W + 28;
      graph.add(n);
      prev.connect(0, n, 0); // out → in (source/filter output ต่อได้หลายเส้น)
      prev = n;
    }
    dst.pos = lastX == null ? [src.pos[0] + NC_W + 28, 80] : [lastX, 80 + 56 + 24 + LiteGraph.NODE_TITLE_HEIGHT];
    prev.connect(0, dst, 0); // ตัวสุดท้าย → ผลลัพธ์ (แทนสายเดิมของ display)
    canvas.setDirty(true, true);
    notify();
    requestAnimationFrame(() => fitView()); // จัดมุมมองให้เห็นทุกกล่องหลังวางเทมเพลต
  }

  // --- เลือก node ตาม id (ใช้ตอนคลิกขั้นตอนใน dropdown ของจอผลลัพธ์) ---
  function selectNodeById(id) {
    const n = graph.getNodeById(id);
    if (!n) return;
    canvas.selectNodes([n]);
    canvas.onNodeSelected(n); // เปิด detail-panel ให้แก้ค่าทันที
    canvas.setDirty(true, true);
  }

  // --- อัปเดต preview ของกล่องผลลัพธ์ (app เรียกทุกเฟรม) ---
  function setDisplayImage(displayId, imageCanvas) {
    if (imageCanvas) displayImages.set(displayId, imageCanvas);
    canvas.setDirty(true, true);
  }

  // --- ควบคุมมุมมองกระดาน (ปุ่ม zoom/fit) ---
  function zoomBy(f) {
    const ds = canvas.ds;
    if (!ds) return;
    const target = Math.max(0.25, Math.min(3, ds.scale * f));
    // ซูมยึดกลางจอ (ปรับ offset ให้) — ลื่นกว่าเซ็ต scale ตรง ๆ
    if (ds.changeScale) ds.changeScale(target, [canvasEl.width / 2, canvasEl.height / 2]);
    else ds.scale = target;
    canvas.setDirty(true, true);
  }
  // จัดมุมมองให้เห็นทุกกล่องพอดี (ผูกกับปุ่ม Fit) — scale ไม่ต่ำกว่า 0.6 เพราะต่ำกว่านี้ litegraph ซ่อนข้อความ
  let fitSnap = null; // มุมมองหลัง fit ล่าสุด — ถ้าผู้ใช้ยังไม่ซูม/เลื่อน resize จะ fit ซ้ำให้อัตโนมัติ
  function fitView(opts) {
    const ds = canvas.ds;
    if (!ds) return;
    const fit = computeFit(nodeRects(), canvasEl.width, canvasEl.height, opts);
    if (!fit) return;
    fitSnap = applyFit(ds, fit); // snapshot อ่านกลับจาก ds (Float32Array) ไม่ใช่ค่า double ก่อนปัดเศษ
    canvas.setDirty(true, true);
  }
  function viewUntouched() {
    return isViewUntouched(canvas.ds, fitSnap);
  }
  // กล่องที่เพิ่งวางต้องอยู่ในจอ: ถ้ายังหลุด (เช่น scale ต่ำสุด .6 แล้วหรือ host เตี้ยมาก) → pan ให้กล่องอยู่กลางจอ
  function ensureVisible(node, keepSnap) {
    const ds = canvas.ds;
    const vw = canvasEl.width, vh = canvasEl.height;
    if (!ds || !vw || !vh) return;
    const TH = LiteGraph.NODE_TITLE_HEIGHT;
    const sx0 = (node.pos[0] + ds.offset[0]) * ds.scale, sx1 = (node.pos[0] + node.size[0] + ds.offset[0]) * ds.scale;
    const sy0 = (node.pos[1] - TH + ds.offset[1]) * ds.scale, sy1 = (node.pos[1] + node.size[1] + ds.offset[1]) * ds.scale;
    if (sx0 >= 0 && sy0 >= 0 && sx1 <= vw && sy1 <= vh) return;
    ds.offset[0] = vw / (2 * ds.scale) - (node.pos[0] + node.size[0] / 2);
    ds.offset[1] = vh / (2 * ds.scale) - (node.pos[1] + (node.size[1] - TH) / 2);
    if (keepSnap) fitSnap = { s: ds.scale, ox: ds.offset[0], oy: ds.offset[1] }; // ยังนับเป็น "ยังไม่ถูกแตะ"
    canvas.setDirty(true, true);
  }
  // หลังวางกล่องใหม่: ถ้ามุมมองยังเป็นของ Fit → fit ใหม่ แล้วให้แน่ใจว่าเห็นกล่องนั้น
  function revealNew(node) {
    const untouched = viewUntouched();
    if (untouched) fitView();
    ensureVisible(node, untouched);
  }
  function resetView() {
    const ds = canvas.ds;
    if (!ds) return;
    fitSnap = null;
    ds.scale = 1;
    ds.offset[0] = 0;
    ds.offset[1] = 0;
    canvas.setDirty(true, true);
  }

  // --- ปรับขนาด canvas ให้ตรงกับ element จริง (LiteGraph ต้องการ pixel จริง) ---
  // สำคัญ: buffer ภายในต้อง = ขนาดที่แสดงจริง มิฉะนั้นพิกัดคลิก/จุดต่อสายจะเพี้ยน
  // (ยิ่งลงล่างยิ่งคลาด) เพราะ CSS ยืด canvas เป็น 100% แต่ buffer ค้างขนาดเดิม
  let lastW = 0, lastH = 0;
  let didInitialFit = false;
  function resize() {
    const rect = canvasEl.parentElement.getBoundingClientRect();
    // buffer = ขนาด CSS จริงเป๊ะ ไม่มี floor (host อาจเตี้ยถึง ~116px ตอนลากแบ่งสุดขอบ) · ซ่อนอยู่ → ข้าม
    const size = canvasBufferSize(rect);
    if (!size) return;
    const { w, h } = size;
    if (w === lastW && h === lastH) return;
    lastW = w; lastH = h;
    const wasUntouched = viewUntouched();
    canvas.resize(w, h);
    // ครั้งแรกที่รู้ขนาดจริง → fit · หลังจากนั้น fit ซ้ำตอน resize เฉพาะเมื่อผู้ใช้ยังไม่ซูม/เลื่อน/รีเซ็ตมุมมอง
    if (!didInitialFit || wasUntouched) {
      didInitialFit = true;
      fitView();
    }
  }
  window.addEventListener("resize", resize);
  // ResizeObserver: จับทุกครั้งที่กล่องกราฟถูกบีบ/ขยาย (เช่น grid ผลลัพธ์โตขึ้น)
  // แม้ window ไม่ resize — กันจุดต่อสาย hit-area เพี้ยน
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => resize()).observe(canvasEl.parentElement);
  }
  requestAnimationFrame(resize);

  graph.start();

  // รายงาน pipeline ตั้งต้น (แหล่งภาพ → ผลลัพธ์) — ตอนต่อสายครั้งแรก notify ยังเป็น noop
  // ถ้าไม่เรียกตรงนี้ state.pipelines จะว่างจนกว่าผู้ใช้จะกดอะไรสักอย่าง (กล้องเลยไม่ติด)
  notify();

  // แก้บั๊ค fan-out (#1): ก่อนลบกล่อง ให้ต่อ upstream กลับไปยัง downstream "ทุกเส้น"
  // (litegraph เดิม splice แค่ outputs[0].links[0] → กล่องที่แตกไปหลายผลลัพธ์ ตัวที่เหลือหลุดสายเงียบ ๆ)
  function reconnectAround(node) {
    if (!node.inputs || !node.outputs || !node.inputs[0] || !node.outputs[0]) return;
    const inLinkId = node.inputs[0].link;
    if (inLinkId == null) return;
    const inLink = graph.links[inLinkId];
    if (!inLink) return;
    const origin = graph.getNodeById(inLink.origin_id);
    if (!origin) return;
    const outLinks = node.outputs[0].links ? [...node.outputs[0].links] : [];
    for (const lid of outLinks) {
      const l = graph.links[lid];
      if (!l) continue;
      const target = graph.getNodeById(l.target_id);
      if (target) origin.connect(inLink.origin_slot, target, l.target_slot); // input รับได้เส้นเดียว → แทนสายเดิมอัตโนมัติ
    }
  }
  // ครอบ deleteSelectedNodes ครั้งเดียว → ใช้ได้ทั้งปุ่มถังขยะและปุ่ม Delete/Backspace บนคีย์บอร์ด
  const _origDeleteSelectedNodes = canvas.deleteSelectedNodes.bind(canvas);
  canvas.deleteSelectedNodes = function () {
    const sel = this.selected_nodes || {};
    const refused = [];
    for (const id in sel) {
      const n = sel[id];
      if (!n) continue;
      // แหล่งภาพห้ามลบ (#2): palette ไม่มีปุ่มเพิ่มกลับ + กันสายทั้งกราฟหายรวดเดียว → เอาออกจาก selection
      if (n.type === SOURCE_TYPE) { delete sel[id]; refused.push(n); continue; }
      reconnectAround(n);
    }
    _origDeleteSelectedNodes();
    // เลือกแหล่งภาพกลับ (ring + สายสีโฟกัสให้ตรงกัน) — เดิมสายค้างสีน้ำเงินทั้งที่ node ไม่ถูกเลือกแล้ว
    if (refused.length) {
      canvas.selectNodes(refused);
      canvas.onNodeSelected(refused[0]);
    }
  };

  // ลบ node ที่เลือก (แหล่งภาพถูกป้องกันไว้ใน deleteSelectedNodes ด้านบน)
  function deleteSelected() {
    if (canvas.deleteSelectedNodes) canvas.deleteSelectedNodes();
    canvas.setDirty(true, true);
    notifyNow();
  }

  return { addFilterNode, addDisplayNode, addDLNode, selectNodeById, setDisplayImage, extractPipelines, exportTemplateSteps, resize,
    zoomIn: () => zoomBy(1.2), zoomOut: () => zoomBy(1 / 1.2), resetView, fitView, deleteSelected, applyTemplate,
    // แหล่งภาพเปลี่ยนโหมด/ไฟล์ → ชื่อ/ไอคอน/บรรทัดรายละเอียดของ Source node ตาม (app.js เรียกเฉพาะตอนค่าเปลี่ยน)
    setSourceInfo: (info) => {
      sourceInfo = { kind: info && SOURCE_KINDS[info.kind] ? info.kind : "camera", detail: (info && info.detail) || "" };
      for (const n of graph.findNodesByType(SOURCE_TYPE)) {
        if (typeof n.title === "string" && n.title.startsWith("1. ")) n.title = sourceTitle();
        n.__mvTextCache = null;
      }
      canvas.setDirty(true, true);
    },
    undo, redo, graph, canvas };
}

// Node-graph editor สไตล์ Node-RED (ครอบ LiteGraph.js)
// ผู้เรียนลาก node มาต่อสาย: [แหล่งภาพ] → filter → filter → [ผลลัพธ์]
// รองรับ "ผลลัพธ์" ได้หลายกล่อง — แต่ละกล่องสกัด pipeline ของตัวเอง (ย้อนจากกล่องนั้น
// กลับไปหาแหล่งภาพ) และวาด preview ของสายตัวเองในกล่อง
import { LiteGraph, LGraph, LGraphCanvas } from "litegraph.js";
import "litegraph.js/css/litegraph.css";
import { OPS, FILTERS, defaultParams, GROUP_COLORS, migrateLegacyStep } from "../filters/registry.js";
import { computeFit, findFreeSpot, applyFit, isViewUntouched, planInsert, formatZoom } from "./layout-math.js";
import { canvasBufferSizeDpr, dotGridRange, flowT, flowDotCount, visibleAreaOf, isFlowLink, createFlowLoop, buildStepIds } from "../ui/phase3-math.js";
import { GROUP_ICON, OP_ICON, OP_PRIMARY, currentOpId, localParamsFor, shortToolName, contrastOn } from "../filters/ui-meta.js";
import { graphColors, onThemeChange } from "../ui/theme.js";

// ---- node card สไตล์ Arctic Blue ----
// Filter/Source/Model Detect = การ์ดขาวกะทัดรัด (ขอบบาง มุมมน เงาจาง): ชื่อ (ตัวหนา) อยู่บนซ้ายในการ์ด · กล่องไอคอนสีหมวดทางซ้าย
// · ไฟสถานะสามดวงข้างกล่องไอคอน (แดง error / เหลือง ยังไม่อยู่ใน pipeline / เขียว อยู่ใน pipeline) · บรรทัดล่าง = ค่าหลัก/ข้อมูลแหล่งภาพ/ข้อความ error
// Display = ชื่ออยู่ "เหนือ" การ์ด preview · การ์ด preview รักษาอัตราส่วนภาพ · ใต้การ์ด = ไฟสามดวง + ขนาดภาพ
// พอร์ตเข้า/ออกอยู่ที่ขอบซ้าย/ขวาของการ์ดที่กึ่งกลางความสูงกล่องไอคอน (getConnectionPos ตัวเดียวใช้ทั้งวาด/ลากสาย/hit-test)
// พิกัดในไฟล์นี้เป็นพิกัดภายใน node (0,0 = pos): พื้นที่ชื่ออยู่ y ∈ [-NODE_TITLE_HEIGHT, 0) — LiteGraph นับเป็นส่วนของ node
// (isPointInside/getBounding/nodeRects ใช้ NODE_TITLE_HEIGHT เหมือนเดิม) จึงคลิก/ลากที่ชื่อก็เลือก/ย้าย node ได้
// ฟอนต์ canvas ต้องมี fallback เสมอ (เดิมไม่มี → เบราว์เซอร์ตกไป serif) — ฟอนต์จริงโหลดจาก @font-face ใน app.css
const NC_STACK = "'IBM Plex Sans Thai', 'Leelawadee UI', 'Noto Sans Thai', 'Segoe UI', Tahoma, sans-serif";
const NC_FONT_TITLE = `700 12px ${NC_STACK}`; // ชื่อ node
const NC_FONT_PORT = `500 11px ${NC_STACK}`;
const NC_FONT_MONO = `500 10.5px 'IBM Plex Mono', Consolas, ${NC_STACK}`; // บรรทัดค่าหลัก/ขนาดภาพ
const NC_FONT_GLYPH = `20px 'Segoe UI Emoji', 'Segoe UI Symbol', ${NC_STACK}`;
const NC_FONT_PILL = `500 10.5px ${NC_STACK}`;
const NC_ICON_FALLBACK = "🧩";
const NC_NAME_H = 24; // ความสูงแถวชื่อเหนือ pos (= LiteGraph.NODE_TITLE_HEIGHT) — การ์ดทั่วไปวาดชื่อในแถวนี้
const NC_PAD = 10; // ระยะขอบในการ์ด
const NC_BOX = 38; // กล่องไอคอน
const NC_W = 148; // ความกว้างการ์ดทั่วไป (Source / Filter / DL)
const NC_RADIUS = 8;
const NC_GAP = 44; // ช่องว่างแนวนอนระหว่างการ์ดตอนวางเทมเพลต/กราฟตั้งต้น (เผื่อพอร์ต + หัวลูกศร + ให้เห็นสาย)
const NC_YELLOW = "#e0a100"; // ไฟเหลือง "ยังไม่อยู่ใน pipeline" (ใช้ได้ทั้งสองธีม)
// layout: w/h = ขนาด node (ใต้ pos) · box = กล่องไอคอน/preview · card = สี่เหลี่ยมการ์ดที่วาดพื้น/ขอบ · kind: "std" | "display"
const LAYOUT_STD = {
  kind: "std",
  w: NC_W,
  h: 62,
  box: { x: NC_PAD, y: 2, w: NC_BOX, h: NC_BOX },
  card: { x: 0, y: -NC_NAME_H, w: NC_W, h: 62 + NC_NAME_H },
};
// Display: การ์ด preview 16:9 (ชื่ออยู่เหนือการ์ด) + พื้นที่ใต้การ์ด 32px สำหรับไฟ + ขนาดภาพ
const LAYOUT_DISPLAY = {
  kind: "display",
  w: 160,
  h: 90 + 32,
  box: { x: 0, y: 0, w: 160, h: 90 },
  card: { x: 0, y: 0, w: 160, h: 90 },
};
// Display สูงกว่า node ทั่วไป: ขยับ y ให้พอร์ตเข้า (กลางการ์ด preview) ตรงระดับพอร์ตออกของ node ทั่วไปที่ pos.y เท่ากัน
const DISPLAY_DY = (LAYOUT_STD.box.y + LAYOUT_STD.box.h / 2) - (LAYOUT_DISPLAY.box.y + LAYOUT_DISPLAY.box.h / 2);
// สีของกราฟ (ขอบเลือก, พื้น/ขอบการ์ด, error 3.6, จุดพื้นหลัง 3.4, จุดวิ่งบนสาย 3.5 ฯลฯ) อยู่ใน ui/theme.js ตามธีมสว่าง/มืด
// → gc() คืนพาเลตต์ปัจจุบัน (object เดิมทุกครั้ง ไม่ alloc ต่อเฟรม) · ขอบตอนเลือกไม่ใช้สีหมวด (Segmentation แดงจะดูเหมือน error)
const gc = graphColors;

// ผสมสี hex: a*t + b*(1-t) → ได้สีทึบ (ไม่โปร่ง ไม่ให้สายเห็นทะลุหัวการ์ด)
function mixHex(a, b, t) {
  const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [x, y] = [p(a), p(b)];
  return "#" + x.map((v, i) => Math.round(v * t + y[i] * (1 - t)).toString(16).padStart(2, "0")).join("");
}

// สีของ node แต่ละชนิด เก็บที่ class (Ctor.color/bgcolor — LiteGraph ใช้เป็น fallback ของ node.color)
// ไม่ใส่ที่ instance เพราะ graph.serialize เก็บ node.color ลง snapshot (undo) → สลับธีมแล้วสีเก่าจะค้าง
// พื้น node = โปร่งใส: การ์ด/ชื่อ/ไฟวาดเองใน onDrawForeground
const cardClasses = [];
function paintCardClass(Ctor, accent) {
  const g = gc();
  Ctor.color = mixHex(accent, g.titleBase, g.titleMixAmt);
  Ctor.bgcolor = "transparent";
}
function registerCardColors(Ctor, accent) {
  cardClasses.push([Ctor, accent]);
  paintCardClass(Ctor, accent);
}

// ชื่อบนกระดานตัดวงเล็บอังกฤษออก ให้สั้นเหมือน palette — ตัด … ตอนวาด (ellipsisText)
function titleOf(node) {
  if (node.__mvTitleRaw === node.title && node.__mvTitle != null) return node.__mvTitle;
  const raw = String(node.title == null ? "" : node.title);
  // ตัดอีโมจิ/สัญลักษณ์นำหน้า ("📷 แหล่งภาพ") — ไอคอนอยู่ในกล่องแล้ว · เลขขั้น "1. " ขึ้นต้นด้วยตัวเลขจึงไม่โดนตัด
  const t = shortToolName(raw).replace(/^[^\p{L}\p{N}]+/u, "").trim();
  node.__mvTitleRaw = node.title;
  node.__mvTitle = t || raw;
  return node.__mvTitle;
}

// พอร์ตวางที่ขอบการ์ด: เข้า = ขอบซ้าย (x=0) · ออก = ขอบขวา (x=w) · กลางความสูงกล่องไอคอน/preview
// op 2 อินพุตวางซ้อนกันรอบกึ่งกลาง · override ที่ prototype (ไม่เก็บ pos ลง serialize) → โปรเจกต์/undo เก่าโหลดแล้วตำแหน่งถูกเสมอ
// ทั้งการวาดจุด + ลากสาย + hit-test ของ LiteGraph เรียก getConnectionPos ตัวนี้ที่เดียว จึงตรงกันเป๊ะ
function cardConnectionPos(isInput, slot, out) {
  out = out || new Float32Array(2);
  const L = this.mvLayout;
  const n = isInput && this.inputs ? this.inputs.length : 1;
  const dy = isInput && n === 2 ? (slot === 0 ? -8 : 8) : 0;
  out[0] = this.pos[0] + (isInput ? 0 : L.w);
  out[1] = this.pos[1] + L.box.y + L.box.h / 2 + dy;
  return out;
}
// จุดพอร์ตกลม · ป้ายชื่อ slot ยังอยู่ในข้อมูล (pipeline/โปรเจกต์เก็บชื่อ) แต่ไม่โชว์: LiteGraph.NODE_TEXT_COLOR = โปร่งใส (applyGraphTheme)
function styleSlots(node) {
  for (const sl of node.inputs || []) sl.shape = LiteGraph.CIRCLE_SHAPE;
  for (const sl of node.outputs || []) sl.shape = LiteGraph.CIRCLE_SHAPE;
  const b = node.inputs && node.inputs[1];
  if (b && node.filterId) {
    if (OPS[currentOpId(node.filterId, node.properties)]?.usesSrc2 === false) {
      // เก็บพอร์ต/สายเดิมไว้ให้โปรเจกต์เก่าและการสลับวิธี แต่บอกชัดว่าวิธีนี้ไม่ใช้ B
      b.color_on = b.color_off = gc().statusOff;
      b.label = "ไม่ใช้ในวิธีนี้";
      b.mvUnused = true;
    } else {
      delete b.color_on;
      delete b.color_off;
      delete b.label;
      delete b.mvUnused;
    }
  }
}
function installCardHooks(Ctor, layout) {
  Ctor.title_mode = LiteGraph.NO_TITLE;
  Ctor.collapsable = false; // ย่อการ์ดไม่มีความหมายกับ layout แบบการ์ดคงที่
  Ctor.prototype.resizable = false; // ขนาดคงที่ตาม layout
  Ctor.prototype.mvLayout = layout;
  Ctor.prototype.getTitle = function () { return titleOf(this); };
  Ctor.prototype.getConnectionPos = cardConnectionPos;
  const prevConfigure = Ctor.prototype.onConfigure;
  Ctor.prototype.onConfigure = function (o) {
    if (prevConfigure) prevConfigure.call(this, o);
    // configure ของ LiteGraph ทับ size/slot ด้วยค่าที่เซฟไว้ (โปรเจกต์เก่า = การ์ดขนาดเดิม, slot ไม่มี shape) → คืนค่า layout ปัจจุบัน
    this.size = [layout.w, layout.h];
    if (this.flags) this.flags.collapsed = false;
    styleSlots(this);
  };
}

// ไฟสามดวง: ซ้าย→ขวา = แดง (error) · เหลือง (ยังไม่อยู่ใน pipeline) · เขียว (อยู่ใน pipeline)
// ไม่พึ่งสีอย่างเดียว: ดวงที่ติดเป็นวงกลมทึบ ที่เหลือเป็นวงโปร่ง · ปิดอยู่ = วงโปร่งสีเทาทั้งสามดวง
function drawTrafficLight(ctx, cx, cy, lit, step) { // lit: "err" | "idle" | "run" | null (ปิดอยู่) · cx = กึ่งกลางดวงที่สอง
  const g = gc();
  const idx = lit === "err" ? 0 : lit === "idle" ? 1 : lit === "run" ? 2 : -1;
  ctx.save();
  ctx.lineWidth = 1.2;
  for (let i = 0; i < 3; i++) {
    ctx.beginPath();
    ctx.arc(cx + (i - 1) * step, cy, 3.5, 0, Math.PI * 2);
    if (i === idx) {
      const color = i === 0 ? g.err : i === 1 ? NC_YELLOW : g.statusOn;
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = color;
    } else {
      ctx.strokeStyle = lit == null ? g.statusOff : g.statusIdle;
    }
    ctx.stroke();
  }
  ctx.restore();
}

// วาดการ์ด Arctic Blue ทั้งใบ (เรียกจาก onDrawForeground ของทุกชนิด node — ช่องพอร์ต LiteGraph วาดทับทีหลัง)
//   o.accent = สีหมวด · o.glyph = ไอคอนในกล่อง (ถ้าไม่มี o.drawBox) · o.sub = บรรทัดเล็ก (ค่าหลัก/ข้อมูล)
//   o.drawBox(ctx, box) = วาดเนื้อในการ์ดเอง (preview ของ Display) · o.disabled = ปิดอยู่
function drawCard(ctx, node, o) {
  const L = node.mvLayout;
  const W = L.w, box = L.box, card = L.card;
  const isDisplay = L.kind === "display";
  const sel = !!node.is_selected;
  const hov = !!node.mouseOver && !sel;
  const disabled = !!o.disabled;
  const inFlow = !!node.__mvInPipeline;
  const err = node.__mvError;
  const g = gc();

  // 1) พื้นการ์ด + ขอบ (เลือก = ขอบฟ้า 2px + เรืองจาง · ชี้ = ขอบเข้มขึ้น · ปกติ = ขอบบาง 1px) + เงาจาง
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(card.x, card.y, card.w, card.h, NC_RADIUS);
  ctx.shadowColor = sel ? g.focusGlow : g.shadow;
  ctx.shadowBlur = sel ? 10 : 4;
  ctx.shadowOffsetY = sel ? 0 : 1;
  ctx.fillStyle = g.body;
  ctx.fill();
  ctx.shadowColor = "transparent";
  ctx.beginPath();
  const bw = sel ? 2 : 1;
  ctx.roundRect(card.x + bw / 2, card.y + bw / 2, card.w - bw, card.h - bw, NC_RADIUS);
  ctx.strokeStyle = sel ? g.focus : hov ? g.cardBorderHover : g.cardBorder;
  ctx.lineWidth = bw;
  ctx.stroke();
  ctx.restore();

  // 2) ชื่อ + กล่องไอคอน/preview — จางลงเมื่อไม่อยู่ใน pipeline / ปิดอยู่
  ctx.save();
  // ชื่อ: "ไม่อยู่ใน pipeline" จางด้วยสี titleTextDim อย่างเดียว (ไม่คูณ alpha ซ้ำ — 0.7 ทำ contrast บนขาวเหลือ ~3.1:1 < 4.5:1)
  if (disabled) ctx.globalAlpha = 0.45;
  ctx.font = NC_FONT_TITLE;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = sel ? g.titleTextSel : inFlow && !disabled ? g.titleText : g.titleTextDim;
  const nx = isDisplay ? 2 : NC_PAD;
  ctx.fillText(ellipsisText(ctx, node, "nm", titleOf(node), (isDisplay ? W : W - NC_PAD * 2) - 2), nx, -8);
  if (!disabled && !inFlow) ctx.globalAlpha = 0.7; // กล่องไอคอน/preview ยังจางตามเดิม
  if (o.drawBox) {
    o.drawBox(ctx, box);
  } else {
    ctx.fillStyle = o.accent;
    ctx.beginPath();
    ctx.roundRect(box.x, box.y, box.w, box.h, 8);
    ctx.fill();
    ctx.fillStyle = contrastOn(o.accent);
    ctx.font = NC_FONT_GLYPH;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(o.glyph || NC_ICON_FALLBACK, box.x + box.w / 2, box.y + box.h / 2 + 1);
  }
  if (disabled) { // ลายเฉียงทับกล่อง
    ctx.beginPath();
    ctx.roundRect(box.x, box.y, box.w, box.h, 8);
    ctx.clip();
    ctx.strokeStyle = g.statusOff;
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = box.x - box.h; x < box.x + box.w; x += 6) {
      ctx.moveTo(x, box.y + box.h);
      ctx.lineTo(x + box.h, box.y);
    }
    ctx.stroke();
  }
  ctx.restore();

  // 3) error: กรอบประแดงรอบการ์ด (ไม่พึ่งสีอย่างเดียว: ประ + ไฟแดง + ข้อความ)
  if (err && !disabled) {
    ctx.save();
    ctx.strokeStyle = g.err;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.roundRect(card.x + 1, card.y + 1, card.w - 2, card.h - 2, NC_RADIUS);
    ctx.stroke();
    ctx.restore();
  }

  // 4) ไฟสามดวง + บรรทัดเล็ก (ค่าหลัก · ถ้า error = ข้อความ error สีแดง · ปิดอยู่ = ป้าย "ปิดอยู่")
  //    การ์ดทั่วไป: ไฟอยู่ขวากล่องไอคอน บรรทัดเล็กอยู่ล่างชิดซ้าย · Display: ไฟกลางใต้การ์ด ตามด้วยขนาดภาพ
  const lit = disabled ? null : err ? "err" : inFlow ? "run" : "idle";
  let sx, sy, align, maxW;
  if (isDisplay) {
    const ly = box.y + box.h + 10;
    drawTrafficLight(ctx, W / 2, ly, lit, 11);
    sx = W / 2; sy = ly + 17; align = "center"; maxW = W - 8;
  } else {
    drawTrafficLight(ctx, box.x + box.w + 14 + 11, box.y + box.h / 2, lit, 11);
    sx = NC_PAD; sy = box.y + box.h + 15; align = "left"; maxW = W - NC_PAD * 2;
  }
  ctx.save();
  ctx.textAlign = align;
  ctx.textBaseline = "alphabetic";
  if (disabled) {
    ctx.font = NC_FONT_PILL;
    const pw = 46;
    const px = align === "center" ? sx - pw / 2 : sx;
    ctx.fillStyle = g.pillBg;
    ctx.beginPath();
    ctx.roundRect(px, sy - 11, pw, 15, 7.5);
    ctx.fill();
    ctx.fillStyle = g.pillText;
    ctx.textAlign = "center";
    ctx.fillText("ปิดอยู่", px + pw / 2, sy);
  } else if (err || o.sub) {
    ctx.font = NC_FONT_MONO;
    ctx.fillStyle = err ? g.errText : g.textSub;
    ctx.fillText(ellipsisText(ctx, node, "sub", err ? String(err) : o.sub, maxW), sx, sy);
  }
  if (node.inputs && node.inputs[1]?.mvUnused) {
    const port = cardConnectionPos.call(node, true, 1);
    ctx.font = NC_FONT_PORT;
    ctx.fillStyle = g.textSub;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.globalAlpha = 0.55;
    ctx.fillText(`B · ${node.inputs[1].label}`, port[0] - node.pos[0] - 8, port[1] - node.pos[1]);
  }
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

  // --- node แหล่งภาพ (ชื่อ/ไอคอน/รายละเอียดสลับตามโหมด — ดู setSourceInfo) ---
  function SourceNode() {
    this.addOutput("ภาพ", "image");
    this.boxcolor = "#2997ff"; // สี Source = Blue (Design §3.1)
    this.size = [LAYOUT_STD.w, LAYOUT_STD.h];
    styleSlots(this);
  }
  registerCardColors(SourceNode, "#2997ff");
  SourceNode.title = "📷 แหล่งภาพ";
  SourceNode.prototype.onConnectionsChange = onGraphChange;
  SourceNode.prototype.onDrawForeground = function (ctx) {
    const kind = SOURCE_KINDS[sourceInfo.kind] || SOURCE_KINDS.camera;
    drawCard(ctx, this, { accent: this.boxcolor, glyph: kind.icon, sub: sourceInfo.detail || kind.line1 });
  };
  installCardHooks(SourceNode, LAYOUT_STD);
  LiteGraph.registerNodeType(SOURCE_TYPE, SourceNode);

  // --- node แสดงผลลัพธ์ (วาด preview ของสายตัวเองในช่อง well) ---
  function DisplayNode() {
    this.addInput("ภาพ", "image");
    this.size = [LAYOUT_DISPLAY.w, LAYOUT_DISPLAY.h];
    this.boxcolor = "#6d7bff"; // สี Output = Indigo (Design §3.1)
    styleSlots(this);
  }
  registerCardColors(DisplayNode, "#6d7bff");
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
  // กล่อง = ช่อง preview สด: วาดภาพผลลัพธ์ใน well (fit รักษาอัตราส่วน) · พอร์ต "ภาพ" อยู่กลางขอบซ้ายของ well
  DisplayNode.prototype.onDrawForeground = function (ctx) {
    const img = displayImages.get(this.id);
    const accent = this.boxcolor;
    drawCard(ctx, this, {
      accent,
      sub: img && img.width ? `${img.width}×${img.height}` : "",
      drawBox: (c, box) => {
        // preview อยู่ในการ์ด (inset 4px) รักษาอัตราส่วนภาพ — ขอบการ์ด/ขอบเลือกวาดโดย drawCard
        const wx = box.x + 4, wy = box.y + 4, ww = box.w - 8, wh = box.h - 8;
        c.fillStyle = gc().well;
        c.beginPath();
        c.roundRect(wx, wy, ww, wh, 5);
        c.fill();
        if (!img || !img.width || !img.height) {
          c.fillStyle = gc().wellText;
          c.font = NC_FONT_PORT;
          c.textAlign = "center";
          c.textBaseline = "alphabetic";
          c.fillText("ยังไม่มีสายเข้า", wx + ww / 2, wy + wh / 2 + 4);
        } else {
          const ar = img.width / img.height;
          let dw = ww;
          let dh = dw / ar;
          if (dh > wh) {
            dh = wh;
            dw = dh * ar;
          }
          c.save();
          c.beginPath();
          c.roundRect(wx, wy, ww, wh, 5);
          c.clip();
          c.drawImage(img, wx + (ww - dw) / 2, wy + (wh - dh) / 2, dw, dh);
          c.restore();
        }
      },
    });
  };
  installCardHooks(DisplayNode, LAYOUT_DISPLAY);
  LiteGraph.registerNodeType(DISPLAY_TYPE, DisplayNode);

  // --- node Deep Learning (YOLO Detect) — ต่อสาย แหล่งภาพ → กล่องนี้ → ผลลัพธ์ ---
  function DLNode() {
    this.addInput("เข้า", "image");
    this.addOutput("ออก", "image");
    this.boxcolor = "#ef5bff"; // สี Deep Learning = Pink (Design §3.1)
    this.size = [LAYOUT_STD.w, LAYOUT_STD.h];
    this.__mvInPipeline = false;
    styleSlots(this);
  }
  registerCardColors(DLNode, "#ef5bff");
  DLNode.title = "🧠 Model Detect";
  DLNode.prototype.onConnectionsChange = onGraphChange;
  DLNode.prototype.onDrawForeground = function (ctx) {
    drawCard(ctx, this, { accent: this.boxcolor, glyph: "🧠", sub: "Deep Learning" });
  };
  installCardHooks(DLNode, LAYOUT_STD);
  LiteGraph.registerNodeType(DL_TYPE, DLNode);

  // --- node ของ filter แต่ละตัวจาก registry (ลงสีตามกลุ่ม) ---
  for (const [id, def] of Object.entries(FILTERS)) {
    const colors = GROUP_COLORS[def.group] || { title: "#475569", body: "#1e293b" };
    function FilterNode() {
      this.addInput("เข้า", "image");
      if (def.inputs === 2) this.addInput("เข้า B", "image"); // Tier 2 §A.1 — เฉพาะ op ที่รับภาพ 2 ใบ (พอร์ตซ้อนสองอันรอบกลางกล่อง)
      this.addOutput("ออก", "image");
      this.properties = defaultParams(id);
      this.filterId = id;
      this.boxcolor = colors.title; // สีประจำหมวด = สีกล่องไอคอน
      this.size = [LAYOUT_STD.w, LAYOUT_STD.h];
      this.__mvInPipeline = false;
      styleSlots(this);
    }
    // ชื่อบนกระดานตัดวงเล็บอังกฤษออก ให้สั้น ไม่ล้น (ชื่อเต็มอยู่ใน tooltip ของ palette)
    registerCardColors(FilterNode, colors.title);
    FilterNode.title = shortToolName(def.name);
    FilterNode.prototype.filterId = id;
    FilterNode.prototype.onConnectionsChange = onGraphChange;
    FilterNode.prototype.onPropertyChanged = onGraphChange;
    // การ์ด: ชื่อ (เหนือกล่อง) · กล่องไอคอนสีหมวด (glyph ของ op ปัจจุบัน) · ไฟ · ค่าหลักที่ตั้งไว้ (ใต้ไฟ)
    FilterNode.prototype.onDrawForeground = function (ctx) {
      styleSlots(this);
      const opId = currentOpId(this.filterId, this.properties);
      const primaryFn = OP_PRIMARY[opId];
      let value = "";
      try {
        if (primaryFn) value = primaryFn(localParamsFor(this.filterId, this.properties)) || "";
      } catch { value = ""; }
      drawCard(ctx, this, {
        accent: this.boxcolor,
        glyph: OP_ICON[opId] || GROUP_ICON[this.filterId] || NC_ICON_FALLBACK,
        sub: value,
        disabled: this.mode === LiteGraph.NEVER,
      });
    };
    installCardHooks(FilterNode, LAYOUT_STD);
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
  canvas.connections_width = 2.5; // เส้นชัดบนพื้นขาว (~2-3px) แต่ยังจับง่าย
  canvas.render_connection_arrows = false;
  canvas.render_canvas_border = false; // กรอบ "ขนาดจอ" ที่ origin กราฟ — เดิมซ่อนอยู่ที่ขอบ canvas แต่พอ Fit เลื่อนมุมมองจะเห็นเป็นสี่เหลี่ยมลอย
  canvas.round_radius = 10; // มุมมนการ์ด node (Design §11.1 radius-md)
  canvas.background_image = null;
  canvas.show_info = false; // ปิด debug overlay (T/I/N/V/FPS ซ้ายล่าง) — เหลือ FPS ประมวลผลจริงอันเดียว
  // ป้ายซูมในหัวกระดาน: ทุกการซูม (ล้อเมาส์/ปุ่ม/Fit) ทำให้ canvas วาดใหม่ → อ่าน scale ตอนวาดพื้นหลังแล้วแจ้งเมื่อ % เปลี่ยน
  let lastZoomText = "";
  canvas.onDrawBackground = (ctx, visible) => {
    const t = formatZoom(canvas.ds && canvas.ds.scale);
    if (t !== lastZoomText) { lastZoomText = t; if (cb.onZoomChange) cb.onZoomChange(t); }
    drawDotGrid(ctx, visible);
  };
  // 3.4 พื้นหลังจุด: ระยะห่าง 24 world px · วาดเฉพาะ scale > 0.6 · จุดทั้งหมดอยู่ใน path เดียว (fill ครั้งเดียวต่อเฟรม)
  function drawDotGrid(ctx, visible) {
    const r = dotGridRange(visible, canvas.ds.scale);
    if (!r) return;
    ctx.fillStyle = gc().dot;
    ctx.beginPath();
    for (let c = 0; c < r.cols; c++) {
      const x = r.x0 + c * r.spacing;
      for (let k = 0; k < r.rows; k++) ctx.rect(x - 1, r.y0 + k * r.spacing - 1, 2, 2);
    }
    ctx.fill();
  }

  // 3.8 ตัวหนังสือคมที่ DPR > 1: buffer = ขนาด CSS × DPR แต่ LiteGraph ยังคิดทุกอย่าง (สเกล/ออฟเซ็ต/เมาส์) เป็นหน่วย CSS px
  // → แค่คูณ kx/ky ตอนตั้ง transform ของ ctx และหารตอนคำนวณพื้นที่ที่มองเห็น · adjustMouseEvent ใช้ getBoundingClientRect (CSS px)
  // อยู่แล้วจึงไม่ต้องแก้ → การจับเมาส์/สายตรงเท่าเดิมทุก DPR
  let kx = 1, ky = 1; // buffer px ต่อ CSS px (แกนละตัว: buffer ปัดเศษจำนวนเต็ม จึงใช้อัตราส่วนจริงให้ตรงเป๊ะ)
  let cssW = 0, cssH = 0; // ขนาด CSS จริงของ canvas (หน่วยที่ใช้คำนวณ fit/วางกล่อง/ซูมกลางจอ)
  const ds0 = canvas.ds;
  const baseComputeVisible = ds0.computeVisibleArea;
  ds0.toCanvasContext = function (ctx) {
    ctx.scale(this.scale * kx, this.scale * ky);
    ctx.translate(this.offset[0], this.offset[1]);
  };
  ds0.computeVisibleArea = function (viewport) {
    if (viewport || !this.element) return baseComputeVisible.call(this, viewport);
    visibleAreaOf(this.offset, this.element.width, this.element.height, kx, ky, this.scale, this.visible_area);
  };
  const viewW = () => cssW || canvasEl.width;
  const viewH = () => cssH || canvasEl.height;

  // 3.5 สายวิ่ง: จุดเล็ก ๆ ไหลบนสายที่ทั้งสองปลายอยู่ใน pipeline — เปิดเฉพาะตอน app สั่ง (กล้องสดกำลังเดิน) และผู้ใช้ไม่ได้ปิดแอนิเมชัน
  // LiteGraph ไม่มีแอนิเมชันสายแบบต่อเนื่อง (flow ของมันผูกกับ event trigger 1 วินาทีและขาวล้วน) → ครอบ renderLink เอง
  // ไม่มี loop วาดต่อเนื่องตอนปิด: RAF ถูกยกเลิกทันที (ภาพนิ่ง/pause = ไม่มี timer ใดทำงานเพิ่มจาก LiteGraph เดิม)
  const reducedMql = typeof matchMedia === "function" ? matchMedia("(prefers-reduced-motion: reduce)") : null;
  let flowWanted = false;
  const flowOn = () => flowWanted && !(reducedMql && reducedMql.matches);
  const baseRenderLink = canvas.renderLink;
  canvas.renderLink = function (ctx, a, b, link, skipBorder, flow, color, startDir, endDir, sub) {
    baseRenderLink.call(this, ctx, a, b, link, skipBorder, flow, color, startDir, endDir, sub);
    if (link && !flow && this.ds.scale >= 0.4) drawLinkArrow(this, ctx, a, b, color || link.color || this.default_link_color, startDir, endDir);
    if (!link || flow || !flowOn() || !linkInPipeline(link)) return;
    const n = flowDotCount(Math.hypot(b[0] - a[0], b[1] - a[1]));
    const now = LiteGraph.getTime();
    ctx.fillStyle = gc().flow;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const p = this.computeConnectionPoint(a, b, flowT(now, i, n), startDir, endDir);
      ctx.moveTo(p[0] + 2.4, p[1]);
      ctx.arc(p[0], p[1], 2.4, 0, Math.PI * 2);
    }
    ctx.fill();
  };
  // หัวลูกศรเข้าพอร์ตปลายทาง (ทิศตามเส้นโค้งช่วงท้าย) — ปลายแหลมห่างจุดพอร์ต 5px ไม่ให้ทับจุดพอร์ต
  function drawLinkArrow(cv, ctx, a, b, color, startDir, endDir) {
    const p = cv.computeConnectionPoint(a, b, 0.94, startDir, endDir);
    const ang = Math.atan2(b[1] - p[1], b[0] - p[0]);
    const ux = Math.cos(ang), uy = Math.sin(ang);
    const tx = b[0] - ux * 5, ty = b[1] - uy * 5;
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - ux * 9 - uy * 4.5, ty - uy * 9 + ux * 4.5);
    ctx.lineTo(tx - ux * 9 + uy * 4.5, ty - uy * 9 - ux * 4.5);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
  function linkInPipeline(link) {
    return isFlowLink(graph.getNodeById(link.origin_id), graph.getNodeById(link.target_id));
  }
  function anyFlowLink() {
    for (const id in graph.links) { const l = graph.links[id]; if (l && linkInPipeline(l)) return true; }
    return false;
  }
  // loop ~25 fps เฉพาะตอนเปิด · ปิด → ยกเลิก RAF ทันทีแล้ววาดใหม่ 1 ครั้งเพื่อลบจุดที่ค้าง (ตรรกะใน createFlowLoop เทสต์ได้)
  const flowLoop = createFlowLoop({
    raf: (fn) => requestAnimationFrame(fn),
    caf: (id) => cancelAnimationFrame(id),
    isOn: flowOn,
    hasWork: anyFlowLink,
    redraw: () => canvas.setDirty(true, true),
    onStop: () => canvas.setDirty(true, true),
  });
  const syncFlow = () => flowLoop.sync();
  if (reducedMql && reducedMql.addEventListener) reducedMql.addEventListener("change", syncFlow);
  // สีพื้นกระดาน + เส้นกริดให้เข้าโทน dark-industrial (Design §3.1)
  // การ์ดไม่มีแถบหัวแล้ว (NO_TITLE) → ค่านี้คือ "ความสูงพื้นที่ชื่อเหนือ pos" ที่ LiteGraph นับเป็นส่วนของ node (hit-test/bounding ตามอัตโนมัติ)
  LiteGraph.NODE_TITLE_HEIGHT = NC_NAME_H;
  LiteGraph.NODE_BOX_OUTLINE_COLOR = "transparent";
  LiteGraph.NODE_SLOT_HEIGHT = 24; // ช่องจุดต่อสายสูงขึ้น = จับง่ายขึ้น
  LiteGraph.link_center_field = 6;
  // ค่าที่ LGraphCanvas คัดลอกจาก LiteGraph ตอน constructor (ตั้งทีหลังไม่มีผล) → ตั้งบน canvas ตรง ๆ
  canvas.render_connections_border = false;
  // สีทั้งหมดที่ขึ้นกับธีม (พื้นกระดาน = --surface-canvas, การ์ด, ตัวอักษร, สาย, พอร์ต) ตั้งในฟังก์ชันเดียว
  // เรียกตอนเริ่ม + ทุกครั้งที่สลับธีม (ไม่ต้อง reload) — ค่าที่ LGraphCanvas คัดลอกตอน constructor ต้องตั้งบน canvas ตรง ๆ
  function applyGraphTheme() {
    const g = gc();
    canvas.clear_background_color = g.canvasBg;
    LiteGraph.NODE_DEFAULT_BGCOLOR = g.body;
    LiteGraph.NODE_DEFAULT_COLOR = g.titleBase;
    LiteGraph.NODE_TITLE_COLOR = g.titleText;
    LiteGraph.NODE_SELECTED_TITLE_COLOR = g.titleTextSel;
    LiteGraph.NODE_DEFAULT_BOXCOLOR = g.defaultBox;
    LiteGraph.NODE_TEXT_COLOR = "transparent"; // ซ่อนป้าย "เข้า/ออก/ภาพ" ของพอร์ต (การ์ดโชว์แค่จุดพอร์ต · ชื่อ slot ยังอยู่ในข้อมูล)
    LiteGraph.DEFAULT_SHADOW_COLOR = g.shadow;
    LiteGraph.LINK_COLOR = g.link;
    canvas.default_link_color = g.link;
    canvas.node_title_color = g.titleText;
    canvas.default_connection_color = { input_off: g.portIdle, input_on: g.portOn, output_off: g.portIdle, output_on: g.portOn };
    canvas.default_connection_color_byType = { image: g.portOn };
    canvas.default_connection_color_byTypeOff = { image: g.portIdle };
    for (const [Ctor, accent] of cardClasses) paintCardClass(Ctor, accent);
    graph._nodes.forEach((n) => { n.__mvTextCache = null; });
    canvas.setDirty(true, true);
  }
  applyGraphTheme();
  onThemeChange(() => {
    applyGraphTheme();
    highlightNodeLinks(highlightedNode); // ทาสีสาย highlight ใหม่ตามธีม (ประกาศทีหลัง จึงเรียกเฉพาะตอนสลับธีม)
  });
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
      document.fonts.load(NC_FONT_MONO, "0"),
    ]).catch(() => {}).then(() => document.fonts.ready).then(redrawAfterFonts);
  }

  // --- สร้างกราฟตั้งต้น: แหล่งภาพ → ผลลัพธ์ ---
  const src = LiteGraph.createNode(SOURCE_TYPE);
  src.pos = [40, 80];
  graph.add(src);
  const dst = LiteGraph.createNode(DISPLAY_TYPE);
  dst.pos = [src.pos[0] + NC_W + NC_GAP, 80 + DISPLAY_DY]; // พอร์ตออกของ Source ตรงระดับพอร์ตเข้าของ Display
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
    const sourceNode = reachedSource ? graph.findNodesByType(SOURCE_TYPE)[0] : null;
    if (reachedSource) {
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
      // id ของกล่องบนสายนี้ตามเลขที่เห็นบนการ์ด (1 = แหล่งภาพ … สุดท้าย = Display) — คีย์ 1–9 ใช้ (nthStepNodeId)
      stepIds: reachedSource && sourceNode ? buildStepIds(sourceNode.id, pathNodes.map((n) => n.id), displayNode.id) : [],
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
    // configure สร้าง node/link ชุดใหม่ + ไม่มีการเลือก (onSelect(null)) → ล้าง highlight ที่ค้าง ไม่งั้นสลับธีมแล้วทาสีสายของ node ที่ไม่มีแล้ว
    highlightedNode = null;
    highlightedLinkObjs = [];
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
  let highlightedNode = null; // จำ node ล่าสุดไว้ทาสีสายใหม่เมื่อสลับธีม
  function highlightNodeLinks(node) {
    highlightedNode = node || null;
    // litegraph เก็บ highlighted_links เป็น object keyed by link id (ไม่ใช่ array)
    // litegraph วาดสาย highlight เป็นสีขาวตายตัว (renderLink) → ไม่ใช้ highlighted_links แต่ตั้ง link.color
    // เป็นสีโฟกัส (#8cc8ff) ให้สายของกล่องที่เลือก แล้วคืนค่าเมื่อเลือกกล่องอื่น
    for (const l of highlightedLinkObjs) l.color = null;
    highlightedLinkObjs = [];
    if (node) {
      const mark = (id) => {
        const l = id != null ? graph.links[id] : null;
        if (l) { l.color = gc().focus; highlightedLinkObjs.push(l); }
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
  // Ctrl+A / ลากกรอบเลือก เรียก selectNodes ตรง (ไม่ผ่าน onNodeSelected) → litegraph ใส่ highlighted_links = สาย "#FFF"
  // บนกระดานขาวสายหายเหลือแต่หัวลูกศร → ล้างทิ้งทุกครั้งที่ selection เปลี่ยน (ใช้ link.color ของ highlightNodeLinks แทน)
  canvas.onSelectionChange = () => { canvas.highlighted_links = {}; };

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
    const w = viewW() || 600;
    const h = viewH() || 300;
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
      x1: Math.min(...rects.map((r) => r.x0)) + (viewW() - 2 * pad) / minS,
      y1: Math.min(...rects.map((r) => r.y0)) + (viewH() - 2 * pad) / minS,
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
      obstacles, TH, gap: NC_GAP, // ระยะห่างเท่ากับตอนวางเทมเพลต (เผื่อพอร์ต + หัวลูกศร)
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
    let x = src.pos[0] + NC_W + NC_GAP;
    for (const raw of steps) {
      const step = migrateLegacyStep(raw); // เทมเพลตเก่าอ้างกลุ่มเก่า (dropdown ไม่ครบหมวด) → ย้ายไปกลุ่มระดับหมวดที่ครบ
      const n = LiteGraph.createNode("filters/" + step.id);
      if (!n) continue; // กัน id ผิด (createNode คืน null)
      if (step.params) Object.assign(n.properties, step.params);
      n.pos = [x, 80];
      x += NC_W + NC_GAP;
      graph.add(n);
      prev.connect(0, n, 0); // out → in (source/filter output ต่อได้หลายเส้น)
      prev = n;
    }
    // การ์ดแคบลงมาก → ผลลัพธ์อยู่ต่อท้ายแถวเดียวกัน (พอร์ตตรงระดับ ไม่ต้องหักลงแถวใหม่เหมือนการ์ด 192px เดิม)
    dst.pos = [x, 80 + DISPLAY_DY];
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
    if (ds.changeScale) ds.changeScale(target, [viewW() / 2, viewH() / 2]);
    else ds.scale = target;
    canvas.setDirty(true, true);
  }
  // จัดมุมมองให้เห็นทุกกล่องพอดี (ผูกกับปุ่ม Fit) — scale ไม่ต่ำกว่า 0.6 เพราะต่ำกว่านี้ litegraph ซ่อนข้อความ
  let fitSnap = null; // มุมมองหลัง fit ล่าสุด — ถ้าผู้ใช้ยังไม่ซูม/เลื่อน resize จะ fit ซ้ำให้อัตโนมัติ
  function fitView(opts) {
    const ds = canvas.ds;
    if (!ds) return;
    const fit = computeFit(nodeRects(), viewW(), viewH(), opts);
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
    const vw = viewW(), vh = viewH();
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
    // buffer = ขนาด CSS × DPR (ไม่มี floor — host อาจเตี้ยถึง ~116px ตอนลากแบ่งสุดขอบ) · ซ่อนอยู่ → ข้าม
    // kx/ky = อัตราส่วนจริงของ buffer ที่ปัดเศษแล้วต่อขนาด CSS → ภาพที่วาดกับพิกัดเมาส์ตรงกันเป๊ะที่ทุก DPR
    const size = canvasBufferSizeDpr(rect, window.devicePixelRatio);
    if (!size) return;
    const { w, h } = size;
    if (w === lastW && h === lastH && Math.abs(size.cw - cssW) < 0.01 && Math.abs(size.ch - cssH) < 0.01) return;
    lastW = w; lastH = h;
    kx = size.kx; ky = size.ky; cssW = size.cw; cssH = size.ch;
    const wasUntouched = viewUntouched();
    canvas.resize(w, h);
    canvas.setDirty(true, true); // resize() ของ LiteGraph ข้ามถ้า buffer เท่าเดิม แต่ kx/ky อาจเปลี่ยน
    // ครั้งแรกที่รู้ขนาดจริง → fit · หลังจากนั้น fit ซ้ำตอน resize เฉพาะเมื่อผู้ใช้ยังไม่ซูม/เลื่อน/รีเซ็ตมุมมอง
    if (!didInitialFit || wasUntouched) {
      didInitialFit = true;
      fitView();
    }
  }
  window.addEventListener("resize", resize);
  // DPR เปลี่ยนโดยขนาดหน้าต่างไม่เปลี่ยน (ย้ายหน้าต่างข้ามจอที่สเกลต่างกัน) ไม่ยิง resize → ฟังผ่าน matchMedia แล้วผูกใหม่ทุกครั้ง
  let dprMql = null;
  function onDprChange() { watchDpr(); resize(); }
  function watchDpr() {
    if (typeof matchMedia !== "function") return;
    if (dprMql && dprMql.removeEventListener) dprMql.removeEventListener("change", onDprChange);
    dprMql = matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    if (dprMql.addEventListener) dprMql.addEventListener("change", onDprChange);
  }
  watchDpr();
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
    // 3.6 error ราย node: app.js เรียกเมื่อ worker บอก nodeId ที่พัง / ล้างเมื่อเฟรมสำเร็จหรือกราฟเปลี่ยน (เปลี่ยนค่าจริงเท่านั้นถึงวาดใหม่)
    setNodeError: (id, text) => {
      const n = id == null ? null : graph.getNodeById(id);
      if (!n) return false;
      if (n.__mvError !== text) { n.__mvError = text; canvas.setDirty(true, true); }
      return true;
    },
    clearNodeErrors: () => {
      let any = false;
      for (const n of graph._nodes) if (n.__mvError) { n.__mvError = null; any = true; }
      if (any) canvas.setDirty(true, true);
    },
    // 3.5 เปิด/ปิดสายวิ่ง (app.js สั่งตามสถานะกล้อง) — ปิดเมื่อผู้ใช้ตั้ง prefers-reduced-motion ด้วย
    setFlow: (on) => { flowWanted = !!on; syncFlow(); },
    undo, redo, graph, canvas };
}

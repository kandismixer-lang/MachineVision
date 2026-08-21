// Node-graph editor สไตล์ Node-RED (ครอบ LiteGraph.js)
// ผู้เรียนลาก node มาต่อสาย: [แหล่งภาพ] → filter → filter → [ผลลัพธ์]
// รองรับ "ผลลัพธ์" ได้หลายกล่อง — แต่ละกล่องสกัด pipeline ของตัวเอง (ย้อนจากกล่องนั้น
// กลับไปหาแหล่งภาพ) และวาด preview ของสายตัวเองในกล่อง
import { LiteGraph, LGraph, LGraphCanvas } from "litegraph.js";
import "litegraph.js/css/litegraph.css";
import { FILTERS, defaultParams, GROUP_COLORS } from "../filters/registry.js";
import { GROUP_ICON, OP_PRIMARY, currentOpId, localParamsFor, hexA } from "../filters/ui-meta.js";

// ---- node card (Design node-card §1) — const/font hoisted ไป module scope กัน alloc ทุกเฟรม ----
const NC_FONT_ICON = "16px 'Noto Sans Thai'";
const NC_FONT_LINE1 = "600 12px 'Noto Sans Thai'";
const NC_FONT_LINE2 = "500 11px 'Noto Sans Thai'";
const NC_ICON_FALLBACK = "🧩";

function drawNodeStateFrame(ctx, node) {
  const W = node.size[0];
  const H = node.size[1];
  const accent = node.boxcolor || "#2997ff";
  const selected = !!node.is_selected;
  const hovered = !!node.mouseOver && !selected;
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(1, -29, W - 2, H + 28, 10);
  ctx.strokeStyle = selected ? accent : hovered ? hexA(accent, 0.72) : hexA(accent, 0.38);
  ctx.lineWidth = selected ? 2 : 1;
  if (selected) {
    ctx.shadowColor = hexA(accent, 0.34);
    ctx.shadowBlur = 6;
  }
  ctx.stroke();
  ctx.restore();
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

  // --- node แหล่งภาพ ---
  function SourceNode() {
    this.addOutput("ภาพ", "image");
    this.color = "#16233a"; // แถบหัวมืด
    this.bgcolor = "#101927";
    this.boxcolor = "#2997ff"; // จุดสี Source = Blue (Design §3.1)
    this.size = [220, 116];
  }
  SourceNode.title = "📷 แหล่งภาพ";
  SourceNode.prototype.onConnectionsChange = onGraphChange;
  SourceNode.prototype.onDrawForeground = function (ctx) {
    if (this.flags && this.flags.collapsed) return;
    const accent = this.boxcolor;
    ctx.fillStyle = hexA(accent, 0.12);
    ctx.beginPath();
    ctx.roundRect(12, 34, 30, 30, 8);
    ctx.fill();
    ctx.strokeStyle = hexA(accent, 0.48);
    ctx.stroke();
    ctx.fillStyle = "#f5f7fa";
    ctx.font = NC_FONT_ICON;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("📷", 27, 49);
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.font = NC_FONT_LINE1;
    ctx.fillText("Camera Source", 52, 35);
    ctx.font = NC_FONT_LINE2;
    ctx.fillStyle = "#9fb0c4";
    ctx.fillText("Live image frame", 52, 54);
    ctx.fillStyle = hexA(accent, 0.76);
    ctx.beginPath();
    ctx.roundRect(12, this.size[1] - 4, this.size[0] - 24, 2, 1);
    ctx.fill();
    drawNodeStateFrame(ctx, this);
  };
  LiteGraph.registerNodeType(SOURCE_TYPE, SourceNode);

  // --- node แสดงผลลัพธ์ (วาด preview ของสายตัวเอง) ---
  function DisplayNode() {
    this.addInput("ภาพ", "image");
    this.size = [220, 138];
    this.color = "#16233a"; // แถบหัวมืด
    this.bgcolor = "#101927";
    this.boxcolor = "#6d7bff"; // จุดสี Output = Indigo (Design §3.1)
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
  // วาดภาพผลลัพธ์ในตัว node (fit รักษาอัตราส่วน)
  DisplayNode.prototype.onDrawForeground = function (ctx) {
    if (this.flags && this.flags.collapsed) return;
    drawNodeStateFrame(ctx, this);
    const img = displayImages.get(this.id);
    const m = 7;
    const top = 6;
    const w = this.size[0] - m * 2;
    const h = this.size[1] - top - m;
    if (!img || !img.width) {
      ctx.fillStyle = "#64748b";
      ctx.font = "11px sans-serif";
      ctx.fillText("(ยังไม่มีสายเข้า)", m, top + 16);
      return;
    }
    const ar = img.width / img.height;
    let dw = w;
    let dh = w / ar;
    if (dh > h) {
      dh = h;
      dw = h * ar;
    }
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(m, top, w, h, 7);
    ctx.clip();
    ctx.drawImage(img, m + (w - dw) / 2, top + (h - dh) / 2, dw, dh);
    ctx.restore();
    ctx.strokeStyle = hexA(this.boxcolor, 0.55);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(m, top, w, h, 7);
    ctx.stroke();
  };
  LiteGraph.registerNodeType(DISPLAY_TYPE, DisplayNode);

  // --- node Deep Learning (YOLO Detect) — ต่อสาย แหล่งภาพ → กล่องนี้ → ผลลัพธ์ ---
  // ไม่บังคับ size เอง — ให้ LiteGraph auto-size ตาม slot เพื่อให้จุดต่อ (วาด) ตรงกับพื้นที่คลิก
  function DLNode() {
    this.addInput("เข้า", "image");
    this.addOutput("ออก", "image");
    this.boxcolor = "#ef5bff"; // จุดสี Deep Learning = Pink (Design §3.1)
    this.color = hexA(this.boxcolor, 0.24);
    this.bgcolor = hexA(this.boxcolor, 0.075);
    this.size = [220, 116];
  }
  DLNode.title = "🧠 YOLO Detect";
  DLNode.prototype.onConnectionsChange = onGraphChange;
  DLNode.prototype.onDrawForeground = function (ctx) {
    drawNodeStateFrame(ctx, this);
  };
  LiteGraph.registerNodeType(DL_TYPE, DLNode);

  // --- node ของ filter แต่ละตัวจาก registry (ลงสีตามกลุ่ม) ---
  for (const [id, def] of Object.entries(FILTERS)) {
    const colors = GROUP_COLORS[def.group] || { title: "#475569", body: "#1e293b" };
    function FilterNode() {
      this.addInput("เข้า", "image");
      this.addOutput("ออก", "image");
      this.properties = defaultParams(id);
      this.filterId = id;
      // แถบหัวมืด + จุดสี accent (ตามภาพ mockup) → ตัวหนังสือขาวอ่านชัด ไม่ใช่พื้นสีสดทั้งแถบ
    // Category color is carried by a restrained tint, not a flat neon fill.
    this.color = hexA(colors.title, 0.24);
    this.bgcolor = hexA(colors.title, 0.075);
      this.boxcolor = colors.title; // จุดสีประจำหมวดที่หัว node
      this.size = [220, 116]; // การ์ดอ่านแบบ flow: title + method + ค่าหลัก + status
    }
    // ชื่อบนกระดานตัดวงเล็บอังกฤษออก ให้สั้น ไม่ล้น (palette ยังใช้ชื่อเต็ม)
    FilterNode.title = def.name.replace(/\s*\([^)]*\)\s*/g, "").trim();
    FilterNode.prototype.filterId = id;
    FilterNode.prototype.onConnectionsChange = onGraphChange;
    FilterNode.prototype.onPropertyChanged = onGraphChange;
    // การ์ดการทดลอง: icon chip + ชื่อ op/method (บรรทัด1) + ค่าหลักที่ตั้งไว้ (บรรทัด2)
    FilterNode.prototype.onDrawForeground = function (ctx) {
      if (this.flags && this.flags.collapsed) return;
      const disabled = this.mode === LiteGraph.NEVER;
      if (disabled) ctx.globalAlpha = 0.45;

      const W = this.size[0];
      const H = this.size[1];
      const accent = this.boxcolor;

      // Quiet content surface beneath the port row.
      ctx.fillStyle = "rgba(255,255,255,0.022)";
      ctx.beginPath();
      ctx.roundRect(7, 30, W - 14, H - 37, 8);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.055)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(8, 29.5);
      ctx.lineTo(W - 8, 29.5);
      ctx.stroke();
      ctx.fillStyle = hexA(accent, 0.78);
      ctx.beginPath();
      ctx.roundRect(12, H - 4, W - 24, 2, 1);
      ctx.fill();

      // 1) icon chip — วางใต้แถว slot (เข้า/ออก อยู่บนสุดของ body) กันตัวหนังสือซ้อนกัน
      const cx = 12, cy = 40, csize = 28, cr = 8;
      ctx.fillStyle = hexA(accent, 0.16);
      ctx.strokeStyle = hexA(accent, 0.5);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(cx, cy, csize, csize, cr);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#f5f7fa";
      ctx.font = NC_FONT_ICON;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(GROUP_ICON[this.filterId] || NC_ICON_FALLBACK, cx + csize / 2, cy + csize / 2 + 1);
      ctx.textAlign = "left";

      // 2) บรรทัดข้อความ: op/method ปัจจุบัน + ค่าหลัก (ถ้ามี)
      const opId = currentOpId(this.filterId, this.properties);
      const primaryFn = OP_PRIMARY[opId];
      let line2 = "";
      try {
        if (primaryFn) line2 = primaryFn(localParamsFor(this.filterId, this.properties)) || "";
      } catch { line2 = ""; }
      const opDef = FILTERS[this.filterId];
      // The title bar identifies the selected Function; the body identifies its
      // parent category so learners can scan both levels without duplicated text.
      const line1Text = opDef ? opDef.name : this.filterId;
      const textW = W - 50 - 12;

      ctx.font = NC_FONT_LINE1;
      ctx.fillStyle = "#f5f7fa";
      ctx.textBaseline = "top";
      const l1 = ellipsisText(ctx, this, "l1", line1Text, textW);
      ctx.fillText(l1, 50, line2 ? 43 : 52);

      if (line2) {
        ctx.font = NC_FONT_LINE2;
        const l2 = ellipsisText(ctx, this, "l2", line2, textW);
        ctx.fillStyle = "#aebbd0";
        ctx.fillText(l2, 50, 63);
      }

      // running indicator แบบจุดสถานะ — เห็นว่า node อยู่ใน flow โดยไม่ต้องใส่คำอธิบายยาว
      ctx.fillStyle = disabled ? "#64748b" : "#7bcf35";
      ctx.beginPath();
      ctx.arc(17, H - 15, 4, 0, Math.PI * 2);
      ctx.fill();

      if (disabled) ctx.globalAlpha = 1;

      drawNodeStateFrame(ctx, this);
    };
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
  canvas.round_radius = 10; // มุมมนการ์ด node (Design §11.1 radius-md)
  canvas.background_image = null;
  canvas.show_info = false; // ปิด debug overlay (T/I/N/V/FPS ซ้ายล่าง) — เหลือ FPS ประมวลผลจริงอันเดียว
  // สีพื้นกระดาน + เส้นกริดให้เข้าโทน dark-industrial (Design §3.1)
  canvas.clear_background_color = "#080d16"; // = --bg-app
  LiteGraph.NODE_DEFAULT_BGCOLOR = "#101927";
  LiteGraph.NODE_DEFAULT_COLOR = "#142033";
  LiteGraph.NODE_TITLE_COLOR = "#f5f7fa";
  LiteGraph.NODE_SELECTED_TITLE_COLOR = "#ffffff";
  LiteGraph.NODE_DEFAULT_BOXCOLOR = "rgba(255,255,255,0.35)";
  LiteGraph.NODE_BOX_OUTLINE_COLOR = "transparent";
  LiteGraph.LINK_COLOR = "#3a4a63";
  LiteGraph.NODE_SLOT_HEIGHT = 24; // ช่องจุดต่อสายสูงขึ้น = จับง่ายขึ้น
  LiteGraph.link_center_field = 6;

  // --- สร้างกราฟตั้งต้น: แหล่งภาพ → ผลลัพธ์ ---
  const src = LiteGraph.createNode(SOURCE_TYPE);
  src.pos = [60, 120];
  graph.add(src);
  const dst = LiteGraph.createNode(DISPLAY_TYPE);
  dst.pos = [440, 100];
  graph.add(dst);
  src.connect(0, dst, 0);

  // --- สกัด pipeline ของกล่องผลลัพธ์หนึ่งกล่อง (เดินย้อนจากกล่องกลับไปหาแหล่งภาพ) ---
  //   autoNum = ลำดับกล่อง (1-based) ใช้ตั้งชื่ออัตโนมัติ "Display N"
  function tracePipeline(displayNode, autoNum) {
    const chain = [];
    const pathNodes = [];
    const visited = new Set();
    let node = displayNode;
    let reachedSource = false;
    let hasDL = false;

    while (node) {
      const inp = node.inputs && node.inputs[0];
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
      pathNodes.unshift(prev);
      if (prev.type === DL_TYPE) hasDL = true; // สายผ่านกล่อง YOLO Detect
      else if (prev.filterId) {
        chain.unshift({
          id: prev.filterId,
          params: { ...prev.properties },
          enabled: prev.mode !== LiteGraph.NEVER,
          nodeId: prev.id,
        });
      }
      node = prev;
    }

    // เลขขั้นตอนอิงเส้นทางจริง: Camera = 1, Function = 2..N, Display = N+1
    if (reachedSource) {
      const sourceNode = graph.findNodesByType(SOURCE_TYPE)[0];
      if (sourceNode) sourceNode.title = "1. Camera";
      pathNodes.forEach((pathNode, i) => {
        const label = pathNode.filterId
          ? (typeof FILTERS[pathNode.filterId]?.subtitle === "function"
              ? FILTERS[pathNode.filterId].subtitle(pathNode.properties)
              : FILTERS[pathNode.filterId]?.name)
          : "YOLO Detect";
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
    const displays = graph.findNodesByType(DISPLAY_TYPE);
    return displays.map((d, i) => tracePipeline(d, i + 1));
  }

  // สกัดสายจาก SOURCE มาถึง Function ที่เลือก (รวม Function นั้นด้วย)
  // ใช้สร้าง Output Preview ด้านขวา โดยไม่ต้องเพิ่ม Display node ปลอมลงกราฟ
  function traceNodePipeline(targetNode) {
    const chain = [];
    const visited = new Set();
    let node = targetNode;
    let reachedSource = false;
    while (node && !visited.has(node.id)) {
      visited.add(node.id);
      if (node.filterId) {
        chain.unshift({
          id: node.filterId,
          params: { ...node.properties },
          enabled: node.mode !== LiteGraph.NEVER,
          nodeId: node.id,
        });
      }
      const inp = node.inputs && node.inputs[0];
      const link = inp && inp.link != null ? graph.links[inp.link] : null;
      if (!link) break;
      const prev = graph.getNodeById(link.origin_id);
      if (!prev) break;
      if (prev.type === SOURCE_TYPE) {
        reachedSource = true;
        break;
      }
      // Preview รอบนี้รองรับ OpenCV Function chain เท่านั้น (DL มี worker คนละตัว)
      if (prev.type === DL_TYPE) break;
      node = prev;
    }
    return { connected: reachedSource, pipeline: reachedSource ? chain : [] };
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
  function highlightNodeLinks(node) {
    // litegraph เก็บ highlighted_links เป็น object keyed by link id (ไม่ใช่ array)
    const hl = {};
    if (node) {
      (node.inputs || []).forEach((i) => { if (i && i.link != null) hl[i.link] = true; });
      (node.outputs || []).forEach((o) => { (o && o.links ? o.links : []).forEach((l) => { hl[l] = true; }); });
    }
    canvas.highlighted_links = hl;
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
      newSrc.pos = [60, 120];
      graph.add(newSrc);
    }
    notify();
  };

  // --- เพิ่ม filter node จาก palette: วางแบบไล่เฉียง กันวางทับกันเป๊ะเมื่อกดรัว ๆ (แก้ #8) ---
  let addSeq = 0;
  function addFilterNode(filterId, opId = null) {
    const node = LiteGraph.createNode("filters/" + filterId);
    if (opId && Object.prototype.hasOwnProperty.call(node.properties || {}, "mode")) {
      node.properties.mode = opId;
      const filter = FILTERS[filterId];
      node.title = typeof filter?.subtitle === "function"
        ? filter.subtitle(node.properties)
        : (filter?.name || opId);
    }
    const off = addSeq++ % 10;
    node.pos = [220 + off * 30, 260 + off * 26];
    graph.add(node);
    canvas.selectNodes([node]);
    canvas.onNodeSelected(node);
    notifyNow();
    return node;
  }

  // --- เพิ่มกล่องผลลัพธ์ใหม่ (ยังไม่ต่อสาย ผู้ใช้ลากเอง) ---
  let displayY = 300;
  function addDisplayNode() {
    const node = LiteGraph.createNode(DISPLAY_TYPE);
    displayY += 40;
    node.pos = [700, 120 + (displayY % 300)];
    graph.add(node);
    canvas.selectNodes([node]);
    canvas.onNodeSelected(node); // เปิดแผงขวา (ตั้งชื่อกล่อง) ทันที ให้สอดคล้องกับ filter/DL
    notifyNow();
    return node;
  }

  // --- เพิ่มกล่อง YOLO Detect (Deep Learning) ---
  function addDLNode() {
    const node = LiteGraph.createNode(DL_TYPE);
    node.pos = [700, 300];
    graph.add(node);
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
    let prev = src;
    let x = 240;
    for (const step of steps) {
      const n = LiteGraph.createNode("filters/" + step.id);
      if (!n) continue; // กัน id ผิด (createNode คืน null)
      if (step.params) Object.assign(n.properties, step.params);
      n.pos = [x, 300];
      x += 210;
      graph.add(n);
      prev.connect(0, n, 0); // out → in (source/filter output ต่อได้หลายเส้น)
      prev = n;
    }
    prev.connect(0, dst, 0); // ตัวสุดท้าย → ผลลัพธ์ (แทนสายเดิมของ display)
    canvas.setDirty(true, true);
    notify();
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
  function resetView() {
    const ds = canvas.ds;
    if (!ds) return;
    ds.scale = 1;
    ds.offset[0] = 0;
    ds.offset[1] = 0;
    canvas.setDirty(true, true);
  }

  // --- ปรับขนาด canvas ให้ตรงกับ element จริง (LiteGraph ต้องการ pixel จริง) ---
  // สำคัญ: buffer ภายในต้อง = ขนาดที่แสดงจริง มิฉะนั้นพิกัดคลิก/จุดต่อสายจะเพี้ยน
  // (ยิ่งลงล่างยิ่งคลาด) เพราะ CSS ยืด canvas เป็น 100% แต่ buffer ค้างขนาดเดิม
  let lastW = 0, lastH = 0;
  function resize() {
    const rect = canvasEl.parentElement.getBoundingClientRect();
    const w = Math.max(200, Math.round(rect.width));
    const h = Math.max(150, Math.round(rect.height));
    if (w === lastW && h === lastH) return;
    lastW = w; lastH = h;
    canvas.resize(w, h);
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
    for (const id in sel) {
      const n = sel[id];
      if (!n) continue;
      // แหล่งภาพห้ามลบ (#2): palette ไม่มีปุ่มเพิ่มกลับ + กันสายทั้งกราฟหายรวดเดียว → เอาออกจาก selection
      if (n.type === SOURCE_TYPE) { delete sel[id]; continue; }
      reconnectAround(n);
    }
    _origDeleteSelectedNodes();
  };

  // ลบ node ที่เลือก (แหล่งภาพถูกป้องกันไว้ใน deleteSelectedNodes ด้านบน)
  function deleteSelected() {
    if (canvas.deleteSelectedNodes) canvas.deleteSelectedNodes();
    canvas.setDirty(true, true);
    notifyNow();
  }

  return { addFilterNode, addDisplayNode, addDLNode, selectNodeById, setDisplayImage, extractPipelines, exportTemplateSteps, resize,
    zoomIn: () => zoomBy(1.2), zoomOut: () => zoomBy(1 / 1.2), resetView, deleteSelected, applyTemplate,
    undo, redo, graph, canvas };
}

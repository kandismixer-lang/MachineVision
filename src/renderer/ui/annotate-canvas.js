// โหมดตีกรอบ (Annotate) — วาด/แก้ไข bounding box บนภาพนิ่งปัจจุบัน แล้วบันทึกผ่าน
// src/renderer/dataset/annotations.js (single source of truth) เท่านั้น — ไม่เก็บ state กรอบซ้ำที่อื่น
// (ยกเว้น cache ชั่วคราวระหว่างลาก ที่ commit กลับ registry ตอนปล่อยเมาส์)
//
// พิกัด: #ann-canvas ใช้ backing-store = ขนาดพิกเซลจริงของภาพพื้นหลัง (ต้นฉบับ/ผ่าน filter)
// ส่วน CSS (.pose-canvas: max-width/height:100%, object-fit:contain) จัด letterbox ให้อัตโนมัติ
// → แปลง screen↔normalized ด้วย getBoundingClientRect() ตรง ๆ ไม่ต้องคำนวณ fit-rect เอง
//
// ทุกขนาดที่วาด (เส้น/handle/hit-radius) เป็น "พิกเซลของ canvas" ไม่ใช่พิกเซลจอ — ถ้าภาพความละเอียดสูง
// แสดงผลย่อเล็กลงมาก เส้น/handle ตายตัวจะเล็กจนมองไม่เห็น จึงต้องคูณ scale = canvas.width / rect.width เสมอ
import {
  getClasses,
  getAnnotations,
  setAnnotations,
  addAnnotation,
  removeAnnotation,
  onChange,
} from "../dataset/annotations.js";

const MIN_DRAG_PX = 5; // ลากน้อยกว่านี้ (พิกเซลจอ) ถือว่าไม่ได้ตั้งใจวาดกรอบ — ทิ้ง
const MIN_BOX_NORM = 0.01; // ขนาดกรอบขั้นต่ำ (normalized) กันกรอบยุบเป็น 0 ตอน resize
const HANDLE_SIZE_PX = 8; // ขนาด handle มุม/ขอบ (คูณ scale ตอนวาดจริง)
const HANDLE_HIT_PX = 10; // รัศมีจับ handle (คูณ scale)

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

// ลำดับ handle รอบกรอบ: มุม 4 + กลางขอบ 4 — ชื่อบอกทิศ (n/s/e/w ผสมกันได้)
function handleList(px) {
  const { x, y, w, h } = px;
  return [
    { name: "nw", x, y },
    { name: "n", x: x + w / 2, y },
    { name: "ne", x: x + w, y },
    { name: "w", x, y: y + h / 2 },
    { name: "e", x: x + w, y: y + h / 2 },
    { name: "sw", x, y: y + h },
    { name: "s", x: x + w / 2, y: y + h },
    { name: "se", x: x + w, y: y + h },
  ];
}

function pointInBox(nx, ny, box) {
  return (
    nx >= box.cx - box.w / 2 &&
    nx <= box.cx + box.w / 2 &&
    ny >= box.cy - box.h / 2 &&
    ny <= box.cy + box.h / 2
  );
}

const CLOSE_POLY_PX = 12; // คลิกใกล้จุดแรกเท่านี้ (พิกเซลจอ) ถือว่า "ปิดรูป" polygon
const LINE_HIT_PX = 8; // รัศมีจับเส้น (พิกเซลจอ)

// bbox normalized (cx,cy,w,h) ครอบ annotation ใด ๆ — ใช้วางป้ายชื่อ + คำนวณคร่าว ๆ
function annBounds(a) {
  if (a.type === "polygon" || a.type === "line") {
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (const p of a.points) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, w: maxX - minX, h: maxY - minY, minX, minY };
  }
  return { cx: a.cx, cy: a.cy, w: a.w, h: a.h, minX: a.cx - a.w / 2, minY: a.cy - a.h / 2 };
}

// จุด (nx,ny) อยู่ในรูปปิด polygon ไหม (ray casting)
function pointInPolygon(nx, ny, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i].x, yi = pts[i].y, xj = pts[j].x, yj = pts[j].y;
    const intersect = yi > ny !== yj > ny && nx < ((xj - xi) * (ny - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

// ระยะจากจุดถึงส่วนของเส้น (พิกัด normalized) — คืนค่า normalized
function distToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
  t = Math.min(1, Math.max(0, t));
  const cx = ax + t * dx, cy = ay + t * dy;
  return Math.hypot(px - cx, py - cy);
}

function distToPolyline(nx, ny, pts) {
  let best = Infinity;
  for (let i = 0; i < pts.length - 1; i++) {
    const d = distToSegment(nx, ny, pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y);
    if (d < best) best = d;
  }
  return best;
}

// hit-test ทั่วไปตามชนิด annotation
function pointHitsAnn(nx, ny, a, scale, canvasW) {
  if (a.type === "polygon") return pointInPolygon(nx, ny, a.points);
  if (a.type === "line") {
    const thresh = (LINE_HIT_PX * scale) / (canvasW || 1);
    return distToPolyline(nx, ny, a.points) <= thresh;
  }
  return pointInBox(nx, ny, a);
}

export function createAnnotateCanvas({
  canvas,
  toolbar,
  emptyClassesEl,
  getImageKey,
  getOriginalImage, // async () => { canvas, w, h } | null
  getFilteredCanvas, // () => HTMLCanvasElement | null (ผลลัพธ์ pipeline ล่าสุด ถ้ามี)
  focusClassPanel, // () => void — เลื่อน/เปิดแผงคลาสขวา
  onEnter, // () => void — เรียกตอนเข้าโหมด (เช่น ซ่อน stage-grid)
  onExit, // () => void — เรียกตอนออกจากโหมด ไม่ว่าจะออกทางไหน (ปุ่ม/Esc/สลับ source)
  setStatus, // (msg) => void — แจ้งสั้น ๆ เช่นตอน fallback พื้นหลังจาก filtered ไปต้นฉบับ
  getImagePosition, // () => { index, total } | null — เลขภาพปัจจุบัน (โชว์ในแถบ Label โดยเฉพาะ fullscreen)
  onNavImage, // (dir: -1|1) => void — เลื่อนภาพก่อนหน้า/ถัดไป (ลูกศร ← → ในโหมด Label — ใช้ตอน fullscreen ที่ batch nav หาย)
}) {
  const ctx = canvas.getContext("2d");
  canvas.tabIndex = 0; // ให้รับ keydown ได้ (scope คีย์ลัดไว้ที่ canvas นี้เท่านั้น กันชนกับกระดาน LiteGraph)

  let active = false;
  let bgMode = "original"; // 'original' | 'filtered' — โหมดที่ผู้ใช้ "ขอ" ไว้ (เฟส export จะใช้ต่อ)
  let bgActual = "original"; // 'original' | 'filtered' — สิ่งที่ "กำลังแสดงจริง" (ต่างจาก bgMode ตอน fallback)
  let bgSource = null; // canvas ที่ใช้เป็นพื้นหลังปัจจุบัน (ต้นฉบับ หรือ ผลลัพธ์ filter)
  let bgGeneration = 0;
  let activeClassIdx = -1;
  let selectedIndex = -1;
  let drag = null;
  let rafPending = false;
  let unsubscribe = null;
  let tool = "bbox"; // 'bbox' | 'polygon' | 'line' — เครื่องมือวาดปัจจุบัน
  let polyDraft = null; // { type, points:[{x,y}], curX, curY } — รูป polygon/line ที่กำลังวาด (ยังไม่ commit)

  // ---------- toolbar (สร้างครั้งเดียว) ----------
  toolbar.innerHTML = "";
  const exitBtn = document.createElement("button");
  exitBtn.type = "button";
  exitBtn.className = "annotate-btn annotate-exit";
  exitBtn.textContent = "✕ ออก";
  exitBtn.title = "ออกจากโหมด Label (Esc)";
  exitBtn.addEventListener("click", () => exit());

  const toggleBgBtn = document.createElement("button");
  toggleBgBtn.type = "button";
  toggleBgBtn.className = "annotate-btn";
  toggleBgBtn.title = "สลับพื้นหลัง ต้นฉบับ/ผ่าน filter";
  toggleBgBtn.addEventListener("click", () => {
    bgMode = bgMode === "original" ? "filtered" : "original";
    loadBackground();
  });

  const chip = document.createElement("span");
  chip.className = "annotate-chip";
  const chipDot = document.createElement("span");
  chipDot.className = "annotate-chip-dot";
  const chipName = document.createElement("span");
  chipName.className = "annotate-chip-name";
  chip.append(chipDot, chipName);

  const countEl = document.createElement("span");
  countEl.className = "annotate-count";

  // ตัวนับภาพปัจจุบัน (ภาพ X/N) — โชว์ในแถบ Label ให้รู้ว่ากำลัง label ภาพไหน (สำคัญตอน fullscreen ที่แถบบนหาย)
  const imgPosEl = document.createElement("span");
  imgPosEl.className = "annotate-imgpos";

  const hintEl = document.createElement("span");
  hintEl.className = "annotate-hint";
  hintEl.textContent = "ลากเมาส์เพื่อตีกรอบ · เลือกคลาสก่อน (กด 1-9)";

  // ---------- ปุ่มเลือกเครื่องมือ (bbox / polygon / line) แบบ Roboflow ----------
  const TOOL_DEFS = [
    { id: "bbox", icon: "▭", label: "กรอบ", hint: "ลากเมาส์เพื่อตีกรอบสี่เหลี่ยม · เลือกคลาสก่อน (กด 1-9)" },
    { id: "polygon", icon: "⬡", label: "Polygon", hint: "คลิกวางจุดทีละจุด · คลิกจุดแรกหรือกด Enter เพื่อปิดรูป · Esc ยกเลิก" },
    { id: "line", icon: "╱", label: "Line", hint: "คลิกวางจุดทีละจุด · ดับเบิลคลิกหรือกด Enter เพื่อจบเส้น · Esc ยกเลิก" },
  ];
  const toolGroup = document.createElement("span");
  toolGroup.className = "annotate-tools";
  const toolBtns = [];
  for (const t of TOOL_DEFS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "annotate-btn annotate-tool" + (t.id === tool ? " active" : "");
    btn.textContent = `${t.icon} ${t.label}`;
    btn.title = t.hint;
    btn.addEventListener("click", () => setTool(t.id));
    toolBtns.push(btn);
    toolGroup.appendChild(btn);
  }

  function setTool(id) {
    if (tool === id) return;
    cancelPolyDraft(); // เปลี่ยนเครื่องมือ = ทิ้งรูปที่วาดค้าง
    tool = id;
    selectedIndex = -1;
    toolBtns.forEach((b, i) => b.classList.toggle("active", TOOL_DEFS[i].id === tool));
    const def = TOOL_DEFS.find((d) => d.id === tool);
    if (def) hintEl.textContent = def.hint;
    canvas.style.cursor = tool === "bbox" ? "crosshair" : "copy";
    scheduleDraw();
  }

  // จัดกลุ่ม (Roboflow-style): [เครื่องมือ] ┊ [คลาส·จำนวน] ┊ [คำใบ้]  →  ขวาสุด [พื้นหลัง][ออก]
  const mkSep = () => { const s = document.createElement("span"); s.className = "annotate-sep"; return s; };
  const info = document.createElement("span");
  info.className = "annotate-info";
  info.append(imgPosEl, chip, countEl);
  const right = document.createElement("span");
  right.className = "annotate-right";
  right.append(toggleBgBtn, exitBtn);
  toolbar.append(toolGroup, mkSep(), info, mkSep(), hintEl, right);

  // ---------- แถบเลือกคลาส (classpicker) — แถวที่ 2 ของ toolbar ----------
  const classPickerEl = document.createElement("div");
  classPickerEl.className = "annotate-classpicker hidden";
  classPickerEl.setAttribute("aria-label", "เลือกคลาสก่อนตีกรอบ");
  toolbar.appendChild(classPickerEl);

  let pickerSig = null; // ลายเซ็นรายการคลาสล่าสุดที่ build ไว้ — กัน rebuild DOM ทุกเฟรม (ดู renderClassPicker)
  let pickerBtns = []; // ปุ่มที่สร้างแล้ว เรียงตาม idx

  classPickerEl.addEventListener("mousedown", (e) => e.preventDefault()); // กัน canvas เสีย focus
  classPickerEl.addEventListener("click", (e) => {
    const btn = e.target.closest(".annotate-classpick");
    if (!btn || !classPickerEl.contains(btn)) return;
    const idx = Number(btn.dataset.idx);
    if (!Number.isInteger(idx)) return;
    selectClass(idx);
    canvas.focus();
  });

  const stageBody = canvas.parentNode || null; // ใช้ toggle ".ann-mode-dock" ตอนเข้า/ออกโหมด (dock layout ใน app.css)

  // ป้ายคลาสที่กำลังเลือก — ใหญ่ กลางบน เห็นตลอดขณะ label (เดิม chip ใน toolbar เล็ก/โดนบังง่าย)
  const classBadge = document.createElement("div");
  classBadge.className = "annotate-class-badge hidden";
  const badgeDot = document.createElement("span");
  badgeDot.className = "annotate-badge-dot";
  const badgeName = document.createElement("span");
  badgeName.className = "annotate-badge-name";
  classBadge.append(badgeDot, badgeName);
  if (stageBody) stageBody.appendChild(classBadge);

  if (emptyClassesEl) {
    emptyClassesEl.innerHTML = "";
    const msg = document.createElement("div");
    msg.textContent = "ยังไม่มีคลาส — สร้างคลาสก่อนถึงจะตีกรอบได้";
    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "btn annotate-btn";
    openBtn.style.marginTop = "8px";
    openBtn.textContent = "เปิดแผงคลาส";
    openBtn.addEventListener("click", () => focusClassPanel?.());
    emptyClassesEl.append(msg, openBtn);
  }

  // ---------- พื้นหลัง ----------
  async function loadBackground() {
    const myGen = ++bgGeneration;
    let src = null;
    let usedFallback = false;
    if (bgMode === "filtered") {
      src = getFilteredCanvas?.() || null;
      if (!src) usedFallback = true; // ยังไม่มีผลลัพธ์ filter (เช่นเพิ่งสลับภาพ ยัง process ไม่เสร็จ)
    }
    if (!src) {
      const orig = await getOriginalImage?.();
      if (myGen !== bgGeneration) return; // ภาพเปลี่ยน/ออกจากโหมดระหว่างรอ decode — ทิ้งผลเก่า
      src = orig ? orig.canvas : null;
    }
    if (myGen !== bgGeneration) return;
    bgSource = src;
    bgActual = usedFallback ? "original" : bgMode; // สะท้อนสิ่งที่ "แสดงจริง" — ปุ่ม/ป้ายอิงจากตัวนี้ ไม่ใช่ bgMode ที่ขอไว้
    if (src && src.width && src.height) {
      if (canvas.width !== src.width) canvas.width = src.width;
      if (canvas.height !== src.height) canvas.height = src.height;
    }
    // แจ้งสั้น ๆ เฉพาะตอน fallback จริง (ขอ filtered แต่ยังไม่มีให้ใช้) — โชว์ต้นฉบับไปพลาง ไม่ค้างจอดำเงียบ ๆ
    if (usedFallback) setStatus?.("ยังไม่มีผลลัพธ์ผ่าน filter ให้แสดง — ใช้ภาพต้นฉบับแทนไปก่อน");
    draw();
  }

  // เรียกตอน pipeline ส่งผลลัพธ์ใหม่มา (เช่นแก้ค่า filter/ต่อสายใหม่) — ถ้ากำลังรอ/แสดง filtered อยู่
  // ให้รีเฟรชพื้นหลังให้เอง ไม่ต้องให้ผู้ใช้กดสลับปุ่ม 2 ครั้ง
  function onFilteredResult() {
    if (!active || bgMode !== "filtered") return;
    loadBackground();
  }

  // ---------- วาด ----------
  function boxPx(box) {
    const w = canvas.width, h = canvas.height;
    const bw = box.w * w, bh = box.h * h;
    return { x: box.cx * w - bw / 2, y: box.cy * h - bh / 2, w: bw, h: bh };
  }

  function currentScale() {
    const rect = canvas.getBoundingClientRect();
    return rect.width ? canvas.width / rect.width : 1;
  }

  function drawBox(box, cls, selected, scale, dashed) {
    const p = boxPx(box);
    ctx.save();
    ctx.strokeStyle = cls.color;
    ctx.lineWidth = (selected ? 3 : 2) * scale;
    if (dashed) ctx.setLineDash([6 * scale, 4 * scale]);
    ctx.strokeRect(p.x, p.y, p.w, p.h);
    ctx.setLineDash([]);
    // ป้ายชื่อคลาสเหนือกรอบ
    ctx.font = `${12 * scale}px sans-serif`;
    const padding = 3 * scale;
    const labelH = 15 * scale;
    const textW = ctx.measureText(cls.name).width;
    ctx.fillStyle = cls.color;
    ctx.fillRect(p.x, p.y - labelH, textW + padding * 2, labelH);
    ctx.fillStyle = "#0a0e14";
    ctx.fillText(cls.name, p.x + padding, p.y - labelH * 0.28);
    if (selected) {
      ctx.fillStyle = cls.color;
      const hs = HANDLE_SIZE_PX * scale;
      for (const hpt of handleList(p)) ctx.fillRect(hpt.x - hs / 2, hpt.y - hs / 2, hs, hs);
    }
    ctx.restore();
  }

  // ป้ายชื่อคลาสที่พิกัด canvas (x,y) = มุมซ้ายบนของป้าย
  function drawLabel(x, y, cls, scale) {
    ctx.font = `${12 * scale}px sans-serif`;
    const padding = 3 * scale;
    const labelH = 15 * scale;
    const textW = ctx.measureText(cls.name).width;
    ctx.fillStyle = cls.color;
    ctx.fillRect(x, y - labelH, textW + padding * 2, labelH);
    ctx.fillStyle = "#0a0e14";
    ctx.fillText(cls.name, x + padding, y - labelH * 0.28);
  }

  // วาด polygon (ปิดรูป) หรือ line (เปิด) ของ annotation ที่บันทึกแล้ว
  function drawPoly(a, cls, selected, scale) {
    const w = canvas.width, h = canvas.height;
    const pts = a.points;
    if (!pts.length) return;
    ctx.save();
    ctx.strokeStyle = cls.color;
    ctx.lineWidth = (selected ? 3 : 2) * scale;
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(pts[0].x * w, pts[0].y * h);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * w, pts[i].y * h);
    if (a.type === "polygon") {
      ctx.closePath();
      ctx.fillStyle = cls.color + "26"; // เติมสีจาง (~15% alpha) ให้เห็นพื้นที่รูป
      ctx.fill();
    }
    ctx.stroke();
    // จุดยอด (เห็นชัดตอนถูกเลือก)
    if (selected) {
      ctx.fillStyle = cls.color;
      const r = (HANDLE_SIZE_PX / 2) * scale;
      for (const p of pts) { ctx.beginPath(); ctx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2); ctx.fill(); }
    }
    ctx.restore();
    const b = annBounds(a);
    drawLabel(b.minX * w, b.minY * h, cls, scale);
  }

  // วาดรูปที่กำลังลากวาด (draft) — จุดที่วางแล้ว + เส้นยางไปยังเคอร์เซอร์
  function drawDraft(scale) {
    if (!polyDraft) return;
    const w = canvas.width, h = canvas.height;
    const pts = polyDraft.points;
    const cls = getClasses()[activeClassIdx] || { name: "?", color: "#3b82f6" };
    ctx.save();
    ctx.strokeStyle = cls.color;
    ctx.lineWidth = 2 * scale;
    ctx.setLineDash([6 * scale, 4 * scale]);
    ctx.beginPath();
    ctx.moveTo(pts[0].x * w, pts[0].y * h);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * w, pts[i].y * h);
    if (polyDraft.curX != null) ctx.lineTo(polyDraft.curX * w, polyDraft.curY * h);
    ctx.stroke();
    ctx.setLineDash([]);
    // จุดที่วางแล้ว — จุดแรกใหญ่กว่า (เป้าปิดรูป)
    const r = 4 * scale;
    pts.forEach((p, i) => {
      ctx.fillStyle = i === 0 ? "#fff" : cls.color;
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, i === 0 ? r * 1.4 : r, 0, Math.PI * 2);
      ctx.fill();
      if (i === 0) { ctx.strokeStyle = cls.color; ctx.lineWidth = 2 * scale; ctx.stroke(); }
    });
    ctx.restore();
  }

  function draw() {
    if (!active) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (bgSource) ctx.drawImage(bgSource, 0, 0, canvas.width, canvas.height);
    else { ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height); }

    const key = getImageKey?.();
    const anns = key ? getAnnotations(key) : [];
    const classes = getClasses();
    const scale = currentScale();

    anns.forEach((a, i) => {
      const cls = classes[a.classIdx] || { name: "?", color: "#888" };
      const sel = i === selectedIndex;
      if (a.type === "polygon" || a.type === "line") {
        // ระหว่างลากย้าย ใช้ preview (จุดที่ขยับแล้ว) แทนของจริง
        const shown = drag && drag.type === "movePoly" && drag.idx === i ? drag.previewAnn : a;
        drawPoly(shown, cls, sel, scale);
      } else {
        const box = drag && drag.type !== "new" && drag.idx === i ? drag.previewBox : a;
        drawBox(box, cls, sel, scale, false);
      }
    });

    if (drag && drag.type === "new") {
      const cx = (drag.startNx + drag.curNx) / 2;
      const cy = (drag.startNy + drag.curNy) / 2;
      const w = Math.abs(drag.curNx - drag.startNx);
      const h = Math.abs(drag.curNy - drag.startNy);
      const cls = classes[activeClassIdx] || { name: "?", color: "#3b82f6" };
      drawBox({ cx, cy, w, h }, cls, true, scale, true);
    }

    drawDraft(scale);
    updateToolbar(anns.length);
  }

  function scheduleDraw() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => { rafPending = false; draw(); });
  }

  function updateToolbar(countOverride) {
    const classes = getClasses();
    const key = getImageKey?.();
    const count = countOverride != null ? countOverride : (key ? getAnnotations(key).length : 0);
    const cls = classes[activeClassIdx];
    chipDot.style.background = cls ? cls.color : "#666";
    chipName.textContent = cls ? cls.name : "ยังไม่เลือกคลาส";
    // badge ใหญ่กลางบน — สะท้อนคลาสที่กำลังเลือก
    badgeDot.style.background = cls ? cls.color : "#666";
    badgeName.textContent = cls ? cls.name : "เลือกคลาสก่อน (กด 1-9)";
    classBadge.classList.toggle("hidden", classes.length === 0);
    countEl.textContent = `${count} กรอบ`;
    // เลขภาพปัจจุบัน (ภาพ X/N) — ซ่อนถ้าไม่มีข้อมูล
    const pos = getImagePosition?.();
    if (pos && pos.total > 0) {
      imgPosEl.textContent = `ภาพ ${pos.index + 1}/${pos.total}`;
      imgPosEl.style.display = "";
    } else {
      imgPosEl.style.display = "none";
    }
    // อิงจาก bgActual (สิ่งที่แสดงจริงบนจอ) ไม่ใช่ bgMode (ที่ขอไว้) — กันปุ่มบอกสถานะกลับด้านตอน fallback
    toggleBgBtn.textContent = bgActual === "original" ? "🎨 ดูผ่าน Filter" : "🖼 ดูต้นฉบับ";
    const noClasses = classes.length === 0;
    if (emptyClassesEl) emptyClassesEl.classList.toggle("hidden", !noClasses);
    hintEl.classList.toggle("hidden", noClasses);
    renderClassPicker();
  }

  // เลือกคลาส idx ให้เป็น activeClassIdx — ถ้ามีกรอบ/รูปถูกเลือกอยู่ ให้เปลี่ยนคลาสของมันไปด้วย
  // (ย้ายมาจาก setActiveClassIdx เดิม แบบไม่แก้ตรรกะ — ใช้ร่วมกันทั้ง public API และคลิก classpicker
  // ⚠️ ต่างจาก handler คีย์ 1-9 ตรงที่คีย์ 1-9 ตอนมีกรอบถูกเลือกจะไม่เปลี่ยน activeClassIdx — ห้ามรวมกัน)
  function selectClass(idx) {
    const classes = getClasses();
    if (idx < 0 || idx >= classes.length) return;
    activeClassIdx = idx;
    if (selectedIndex >= 0) {
      const key = getImageKey?.();
      if (key) {
        const arr = getAnnotations(key);
        if (arr[selectedIndex]) { arr[selectedIndex] = { ...arr[selectedIndex], classIdx: idx }; setAnnotations(key, arr); }
      }
    }
    scheduleDraw();
  }

  // สร้าง/อัปเดตปุ่มเลือกคลาสแบบ incremental — ห้าม rebuild DOM ทุกครั้ง เพราะ updateToolbar()
  // ถูกเรียกจาก draw() ทุก requestAnimationFrame ระหว่างลากกรอบ (ดูคอมเมนต์หัวไฟล์เรื่อง draw loop)
  function renderClassPicker() {
    const classes = getClasses();
    classPickerEl.classList.toggle("hidden", classes.length === 0);
    if (classes.length === 0) {
      if (pickerSig !== "") { classPickerEl.innerHTML = ""; pickerBtns = []; pickerSig = ""; }
      return;
    }
    const SEP_FIELD = String.fromCharCode(0);
    const SEP_ITEM = String.fromCharCode(1);
    const sig = classes.map((c) => `${c.name}${SEP_FIELD}${c.color}`).join(SEP_ITEM);
    if (sig !== pickerSig) {
      // rebuild เฉพาะตอนรายการคลาสเปลี่ยนจริง (เพิ่ม/ลบ/เปลี่ยนชื่อ/เปลี่ยนสี)
      classPickerEl.innerHTML = "";
      pickerBtns = classes.map((c, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "annotate-classpick";
        b.dataset.idx = String(i);
        b.title = i < 9 ? `${c.name} (กด ${i + 1})` : c.name;
        const dot = document.createElement("span");
        dot.className = "cp-dot";
        dot.style.background = c.color;
        const name = document.createElement("span");
        name.className = "cp-name";
        name.textContent = c.name;
        b.append(dot, name);
        if (i < 9) {
          const key = document.createElement("span");
          key.className = "cp-key";
          key.textContent = String(i + 1);
          b.appendChild(key);
        }
        return b;
      });
      pickerBtns.forEach((b) => classPickerEl.appendChild(b));
      pickerSig = sig;
    }
    // อัปเดต state ที่เปลี่ยนบ่อย (ราคาถูก) ทุกครั้ง
    pickerBtns.forEach((b, i) => {
      const on = i === activeClassIdx;
      b.classList.toggle("active", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  // ---------- input ----------
  function eventToNorm(e) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return { nx: 0, ny: 0 };
    return {
      nx: clamp((e.clientX - rect.left) / rect.width, 0, 1),
      ny: clamp((e.clientY - rect.top) / rect.height, 0, 1),
    };
  }

  function hitTestHandle(nx, ny, box, scale) {
    const p = boxPx(box);
    const px = nx * canvas.width, py = ny * canvas.height;
    const r = HANDLE_HIT_PX * scale;
    let best = null, bestDist = r;
    for (const hpt of handleList(p)) {
      const d = Math.hypot(px - hpt.x, py - hpt.y);
      if (d <= bestDist) { bestDist = d; best = hpt.name; }
    }
    return best;
  }

  function onPointerDown(e) {
    if (e.button !== undefined && e.button !== 0) return;
    canvas.focus();
    const key = getImageKey?.();
    const classes = getClasses();
    if (!key || classes.length === 0) return;
    const { nx, ny } = eventToNorm(e);
    const anns = getAnnotations(key);
    const scale = currentScale();

    // (1) กำลังวาด polygon/line อยู่ → คลิก = วางจุดเพิ่ม (หรือปิดรูปถ้าคลิกใกล้จุดแรก)
    if (polyDraft) {
      const first = polyDraft.points[0];
      const dPx = Math.hypot((nx - first.x) * canvas.width, (ny - first.y) * canvas.height) / scale;
      const enough = polyDraft.points.length >= (polyDraft.type === "polygon" ? 3 : 2);
      if (enough && dPx <= CLOSE_POLY_PX) { finishPolyDraft(); return; }
      polyDraft.points.push({ x: nx, y: ny });
      scheduleDraw();
      return;
    }

    // (2) แก้ไขรูปที่ถูกเลือกอยู่: bbox → resize handle/move · polygon/line → move ทั้งรูป
    if (selectedIndex >= 0 && anns[selectedIndex]) {
      const sel = anns[selectedIndex];
      if (sel.type === "polygon" || sel.type === "line") {
        if (pointHitsAnn(nx, ny, sel, scale, canvas.width)) {
          drag = { type: "movePoly", idx: selectedIndex, startNx: nx, startNy: ny, orig: sel, previewAnn: sel };
          capture(e);
          return;
        }
      } else {
        const handle = hitTestHandle(nx, ny, sel, scale);
        if (handle) {
          drag = {
            type: "resize", idx: selectedIndex, handle,
            left: sel.cx - sel.w / 2, right: sel.cx + sel.w / 2,
            top: sel.cy - sel.h / 2, bottom: sel.cy + sel.h / 2,
            previewBox: { ...sel },
          };
          capture(e);
          return;
        }
        if (pointInBox(nx, ny, sel)) {
          drag = { type: "move", idx: selectedIndex, startNx: nx, startNy: ny, orig: { ...sel }, previewBox: { ...sel } };
          capture(e);
          return;
        }
      }
    }

    // (3) คลิกโดนรูปที่มีอยู่ → เลือก (แล้วลากย้ายได้เลย)
    for (let i = anns.length - 1; i >= 0; i--) {
      if (pointHitsAnn(nx, ny, anns[i], scale, canvas.width)) {
        selectedIndex = i;
        const a = anns[i];
        if (a.type === "polygon" || a.type === "line") {
          drag = { type: "movePoly", idx: i, startNx: nx, startNy: ny, orig: a, previewAnn: a };
        } else {
          drag = { type: "move", idx: i, startNx: nx, startNy: ny, orig: { ...a }, previewBox: { ...a } };
        }
        capture(e);
        scheduleDraw();
        return;
      }
    }

    // (4) คลิกที่ว่าง → เริ่มวาดรูปใหม่ตามเครื่องมือปัจจุบัน
    selectedIndex = -1;
    if (activeClassIdx < 0 || activeClassIdx >= classes.length) { scheduleDraw(); return; }
    if (tool === "polygon" || tool === "line") {
      polyDraft = { type: tool, points: [{ x: nx, y: ny }], curX: nx, curY: ny };
      scheduleDraw();
      return;
    }
    drag = { type: "new", startNx: nx, startNy: ny, curNx: nx, curNy: ny };
    capture(e);
    scheduleDraw();
  }

  function capture(e) {
    try { if (e.pointerId != null) canvas.setPointerCapture(e.pointerId); } catch { /* ignore */ }
  }

  function onPointerMove(e) {
    // เส้นยางของ polygon/line ที่กำลังวาด (ไม่ต้องมี drag — ขยับเมาส์เฉย ๆ ก็อัปเดต)
    if (polyDraft) {
      const { nx, ny } = eventToNorm(e);
      polyDraft.curX = nx; polyDraft.curY = ny;
      scheduleDraw();
      return;
    }
    if (!drag) return;
    const { nx, ny } = eventToNorm(e);
    if (drag.type === "new") {
      drag.curNx = nx; drag.curNy = ny;
    } else if (drag.type === "movePoly") {
      const dx = nx - drag.startNx, dy = ny - drag.startNy;
      drag.previewAnn = {
        type: drag.orig.type,
        classIdx: drag.orig.classIdx,
        points: drag.orig.points.map((p) => ({ x: clamp(p.x + dx, 0, 1), y: clamp(p.y + dy, 0, 1) })),
      };
    } else if (drag.type === "move") {
      const dx = nx - drag.startNx, dy = ny - drag.startNy;
      const halfW = drag.orig.w / 2, halfH = drag.orig.h / 2;
      const cx = clamp(drag.orig.cx + dx, halfW, 1 - halfW);
      const cy = clamp(drag.orig.cy + dy, halfH, 1 - halfH);
      drag.previewBox = { ...drag.orig, cx, cy };
    } else if (drag.type === "resize") {
      let { left, right, top, bottom, handle } = drag;
      if (handle.includes("w")) left = nx;
      if (handle.includes("e")) right = nx;
      if (handle.includes("n")) top = ny;
      if (handle.includes("s")) bottom = ny;
      if (right - left < MIN_BOX_NORM) { if (handle.includes("w")) left = right - MIN_BOX_NORM; else right = left + MIN_BOX_NORM; }
      if (bottom - top < MIN_BOX_NORM) { if (handle.includes("n")) top = bottom - MIN_BOX_NORM; else bottom = top + MIN_BOX_NORM; }
      left = clamp(left, 0, 1); right = clamp(right, 0, 1);
      top = clamp(top, 0, 1); bottom = clamp(bottom, 0, 1);
      drag.left = left; drag.right = right; drag.top = top; drag.bottom = bottom;
      drag.previewBox = {
        classIdx: drag.previewBox.classIdx,
        cx: (left + right) / 2, cy: (top + bottom) / 2,
        w: right - left, h: bottom - top,
      };
    }
    scheduleDraw();
  }

  function onPointerUp(e) {
    if (!drag) return;
    const key = getImageKey?.();
    if (drag.type === "new") {
      const rect = canvas.getBoundingClientRect();
      const dxPx = Math.abs(drag.curNx - drag.startNx) * rect.width;
      const dyPx = Math.abs(drag.curNy - drag.startNy) * rect.height;
      if (key && dxPx >= MIN_DRAG_PX && dyPx >= MIN_DRAG_PX) {
        const cx = (drag.startNx + drag.curNx) / 2;
        const cy = (drag.startNy + drag.curNy) / 2;
        const w = Math.abs(drag.curNx - drag.startNx);
        const h = Math.abs(drag.curNy - drag.startNy);
        selectedIndex = addAnnotation(key, { classIdx: activeClassIdx, cx, cy, w, h });
      }
    } else if ((drag.type === "move" || drag.type === "resize") && key) {
      const arr = getAnnotations(key);
      if (arr[drag.idx]) {
        arr[drag.idx] = { ...arr[drag.idx], ...drag.previewBox };
        setAnnotations(key, arr);
      }
    } else if (drag.type === "movePoly" && key) {
      const arr = getAnnotations(key);
      if (arr[drag.idx]) {
        arr[drag.idx] = drag.previewAnn;
        setAnnotations(key, arr);
      }
    }
    drag = null;
    try { if (e.pointerId != null) canvas.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    scheduleDraw();
  }

  // ทิ้งรูป polygon/line ที่วาดค้างอยู่ (Esc/เปลี่ยนเครื่องมือ/ออกโหมด)
  function cancelPolyDraft() {
    if (!polyDraft) return;
    polyDraft = null;
    scheduleDraw();
  }

  // จบการวาด polygon/line → บันทึกเป็น annotation (ถ้าจุดพอ) แล้วเลือกให้เลย
  function finishPolyDraft() {
    if (!polyDraft) return;
    const key = getImageKey?.();
    const min = polyDraft.type === "polygon" ? 3 : 2;
    if (key && polyDraft.points.length >= min && activeClassIdx >= 0) {
      selectedIndex = addAnnotation(key, {
        type: polyDraft.type,
        classIdx: activeClassIdx,
        points: polyDraft.points.map((p) => ({ x: p.x, y: p.y })),
      });
    } else if (polyDraft.points.length < min) {
      // จุดไม่พอ (เช่นดับเบิลคลิกเร็วไป) — บอกผู้ใช้แทนที่จะทิ้งเงียบ ๆ
      setStatus?.(polyDraft.type === "polygon" ? "ต้องมีอย่างน้อย 3 จุดถึงจะปิด polygon ได้" : "ต้องมีอย่างน้อย 2 จุดถึงจะเป็นเส้นได้");
    }
    polyDraft = null;
    scheduleDraw();
  }

  function onDblClick(e) {
    if (!polyDraft) return;
    e.preventDefault();
    // ดับเบิลคลิกจะยิง pointerdown มาก่อน 1 จุด (จุดซ้ำติดกัน) — ตัดจุดสุดท้ายที่ซ้อนออก
    const pts = polyDraft.points;
    if (pts.length >= 2) {
      const a = pts[pts.length - 1], b = pts[pts.length - 2];
      if (Math.hypot(a.x - b.x, a.y - b.y) < 0.005) pts.pop();
    }
    finishPolyDraft();
  }

  function onKeyDown(e) {
    if (!active) return;
    if (document.querySelector(".modal-overlay")) return; // ปล่อยให้ modal จัดการ Esc ของมันเองก่อน
    if (e.key === "Enter" && polyDraft) { e.preventDefault(); finishPolyDraft(); return; }
    // ลูกศร ← → เลื่อนภาพก่อนหน้า/ถัดไป (ไม่ทำถ้ากำลังวาด/ลากค้าง) — ใช้ตอน fullscreen ที่ batch nav หาย
    if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && !polyDraft && !drag && onNavImage) {
      e.preventDefault();
      onNavImage(e.key === "ArrowLeft" ? -1 : 1);
      return;
    }
    if (e.key >= "1" && e.key <= "9") {
      const idx = Number(e.key) - 1;
      const classes = getClasses();
      if (idx >= classes.length) return;
      e.preventDefault();
      if (selectedIndex >= 0) {
        const key = getImageKey?.();
        if (key) {
          const arr = getAnnotations(key);
          if (arr[selectedIndex]) { arr[selectedIndex] = { ...arr[selectedIndex], classIdx: idx }; setAnnotations(key, arr); }
        }
      } else {
        activeClassIdx = idx;
      }
      scheduleDraw();
      return;
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      // กำลังวาด polygon/line อยู่ → ลบจุดล่าสุดทีละจุด (ครบแล้วเลิกวาด)
      if (polyDraft) {
        e.preventDefault();
        polyDraft.points.pop();
        if (polyDraft.points.length === 0) polyDraft = null;
        scheduleDraw();
        return;
      }
      // keydown ผูกไว้ที่ #ann-canvas เท่านั้น (ไม่ใช่ document) — LiteGraph canvas ของกระดานคนละตัว
      // และไม่มีทาง focus ได้ระหว่างโหมดนี้ (canvas.focus() ถูกเรียกตอน enter + ทุก pointerdown)
      // จึงไม่มีการชนคีย์กับ LiteGraph จริง แม้จะมี node ถูกเลือกค้างอยู่บนกระดานก็ตาม
      if (selectedIndex >= 0) {
        e.preventDefault();
        const key = getImageKey?.();
        if (key) removeAnnotation(key, selectedIndex);
        selectedIndex = -1;
        scheduleDraw();
      }
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation(); // กันไม่ให้คาสเคดไปโดน global handler (เช่นออก fullscreen) — Esc ต้องถอยทีละขั้น
      if (polyDraft) { cancelPolyDraft(); return; }
      if (drag && drag.type === "new") { drag = null; scheduleDraw(); return; }
      if (selectedIndex >= 0) { selectedIndex = -1; scheduleDraw(); return; }
      exit();
    }
  }

  // ---------- เข้า/ออกโหมด ----------
  function enter() {
    if (active) return;
    active = true;
    selectedIndex = -1;
    drag = null;
    const classes = getClasses();
    if (activeClassIdx < 0 && classes.length > 0) activeClassIdx = 0;
    canvas.classList.remove("hidden");
    toolbar.classList.remove("hidden");
    stageBody?.classList.add("ann-mode-dock");
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    canvas.addEventListener("keydown", onKeyDown);
    canvas.addEventListener("dblclick", onDblClick);
    unsubscribe = onChange(() => {
      // แผงคลาสอาจลบคลาส → remap/ลบกรอบข้ามภาพทั้งหมด → selectedIndex ที่ค้างไว้อาจไม่ตรงกรอบเดิมอีกต่อไป
      const key = getImageKey?.();
      const count = key ? getAnnotations(key).length : 0;
      if (selectedIndex >= count) selectedIndex = -1;
      // คลาสที่กำลังจะใช้วาดกรอบใหม่ (activeClassIdx) อาจเกินช่วงถ้าถูกลบไปแล้วเช่นกัน
      const classes = getClasses();
      if (activeClassIdx >= classes.length) activeClassIdx = classes.length - 1;
      scheduleDraw();
    });
    canvas.focus();
    onEnter?.();
    loadBackground();
  }

  function exit() {
    if (!active) return;
    active = false;
    drag = null;
    polyDraft = null;
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("pointercancel", onPointerUp);
    canvas.removeEventListener("keydown", onKeyDown);
    canvas.removeEventListener("dblclick", onDblClick);
    if (unsubscribe) { unsubscribe(); unsubscribe = null; }
    canvas.classList.add("hidden");
    toolbar.classList.add("hidden");
    stageBody?.classList.remove("ann-mode-dock");
    classBadge.classList.add("hidden");
    if (emptyClassesEl) emptyClassesEl.classList.add("hidden");
    bgSource = null;
    bgGeneration++; // ทิ้งงาน decode ที่ค้างอยู่
    onExit?.();
  }

  // เรียกตอนเปลี่ยนภาพ (Prev/Next) ขณะยังอยู่ในโหมด — โหลดพื้นหลัง+กรอบของภาพใหม่
  function onImageChanged() {
    if (!active) return;
    selectedIndex = -1;
    drag = null;
    polyDraft = null;
    loadBackground();
  }

  return {
    enter,
    exit,
    onImageChanged,
    onFilteredResult,
    isActive: () => active,
    // 'original' | 'filtered' — โหมดพื้นหลังที่ผู้ใช้ "ขอ" ไว้ (ใช้ตอน export dataset เลือกภาพต้นฉบับ/ผ่าน filter)
    getBgMode: () => bgMode,
    getActiveClassIdx: () => activeClassIdx,
    setActiveClassIdx: (idx) => selectClass(idx),
  };
}

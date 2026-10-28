// Panel ซ้าย: palette ของ node — คลิกเพื่อเพิ่มลงผืนกราฟ (สไตล์ Node-RED)
import { FILTERS, OPS, GROUP_COLORS } from "../filters/registry.js";
import { GROUP_ICON, OP_ICON, shortToolName, contrastOn } from "../filters/ui-meta.js";

export const TOOL_SECTIONS = [
  { title: "COLOR", ops: ["grayscale", "colorspace", "colorspace_inverse", "channel_split", "rgb_adjust", "color_inrange", "kmeans_segment", "slic", "white_balance", "invert", "pseudocolor"] },
  { title: "ENHANCE", ops: ["brightness_contrast", "gamma", "hist_equalize", "clahe", "contrast_stretch", "flatfield_correction", "quantize", "unsharp", "laplacian_sharpen", "custom_kernel", "dehaze", "lowlight_enhance"] },
  { title: "GEOMETRY", ops: ["crop", "roi_polygon", "resize", "rotate", "flip", "perspective_warp", "undistort", "translate_shear"] },
  { title: "FILTER & DENOISE", ops: ["salt_pepper_noise", "gaussian_noise", "median_blur", "gaussian_blur", "box_blur", "bilateral", "wiener_filter", "guided_filter", "nlm_denoise", "deconv_wiener", "deconv_lucy"] },
  { title: "MORPHOLOGY", ops: ["morph_erode", "morph_dilate", "morph_open", "morph_close", "morph_gradient", "morph_tophat", "morph_blackhat", "morph_fill_holes", "morph_clear_border", "morph_remove_small", "morph_clean", "morph_majority", "morph_skeleton", "morph_spur", "morph_thin", "morph_shrink", "morph_reconstruct_open", "morph_convex_hull", "morph_endpoints_branchpoints", "morph_hit_or_miss", "morph_regional_max", "morph_regional_min", "morph_h_maxima", "morph_h_minima", "morph_extended_max", "morph_extended_min"] },
  { title: "SEGMENTATION & EDGE", ops: ["binary_threshold", "otsu_threshold", "adaptive_threshold", "multi_otsu_threshold", "canny", "sobel", "laplacian_edge", "scharr_edge", "prewitt_edge", "roberts_edge", "log_edge", "watershed", "watershed_marker", "grabcut", "region_growing"] },
  { title: "ANALYSIS & DETECTION", ops: ["find_contours", "connected_components", "distance_transform", "local_std_filter", "local_entropy_filter", "gabor_filter", "glcm", "focus_score", "regionprops", "euler_number", "feret", "line_profile", "shape_filter", "bounding_boxes", "hough_circles", "hough_lines", "harris_corners", "shi_tomasi_corners", "orb_keypoints", "akaze_keypoints", "fast_keypoints", "pass_fail"] },
  { title: "FREQUENCY DOMAIN", ops: ["fft_magnitude", "fft_filter", "bandpass_notch", "dct_jpeg"] },
  { title: "IMAGE ARITHMETIC", ops: ["imabsdiff", "imfuse", "mask", "imarith", "logical_ops", "inpaint", "hist_match", "roi_apply"] },
  { title: "IMAGE REGISTRATION", ops: ["template_matching", "ecc_register"] },
  { title: "IMAGE QUALITY", ops: ["psnr", "ssim", "delta_e", "seg_score"] },
  { title: "MOTION", ops: ["mog2", "farneback_optical_flow", "pyrlk_optical_flow"] },
];

// 1 หมวดบน palette = 1 node (2026-10-07): 7 หมวดที่ op กระจายหลายกลุ่มเดิมชี้ไปกลุ่มระดับหมวด (dropdown "เลือกวิธี"
// ครบทุก op ของหมวดตามลำดับ TOOL_SECTIONS) · กลุ่มเก่า (color_space, colour, blur, watershed, psnr, mog2 ...) ยังลงทะเบียนใน
// FILTERS เป็น legacy แต่ไม่อยู่ในลิสต์นี้ → ไม่โผล่บน palette · tests/category-merge.test.mjs ล็อกให้ทุกหมวดตรงกัน
const VISIBLE_FILTER_IDS = [
  "color_tools", "enhance_tools", "geometry", "filter_denoise", "morph_tools",
  "segment_edge", "analysis_detect",
  "frequency", "image_arithmetic",
  "image_registration", "image_quality", "motion",
];

export const OP_TO_FILTER = {};
for (const filterId of VISIBLE_FILTER_IDS) {
  const filter = FILTERS[filterId];
  const mode = filter.params.find((p) => p.key === "mode");
  if (mode) mode.options.forEach((o) => { OP_TO_FILTER[o.value] = filterId; });
  else OP_TO_FILTER[filterId] = filterId;
}

// ---------- "ใช้ล่าสุด" (Phase 3.3) ----------
// เก็บ opId ของเครื่องมือที่เพิ่งเพิ่มลงกระดาน (ใหม่สุดก่อน, ไม่ซ้ำ, ≤ 5) ใน localStorage["mv-palette-recent"] เป็น JSON array
export const PALETTE_RECENT_KEY = "mv-palette-recent";
export const PALETTE_RECENT_MAX = 5;

// raw = ข้อความดิบจาก localStorage (หรือ null) · isKnown(opId) = ยังมีเครื่องมือนี้บน palette อยู่ไหม → รายการสะอาด (ค่าพัง/ซ้ำ/ไม่รู้จักถูกทิ้ง)
export function resolveRecent(raw, isKnown, max = PALETTE_RECENT_MAX) {
  let list = [];
  if (typeof raw === "string") {
    try {
      const a = JSON.parse(raw);
      if (Array.isArray(a)) list = a;
    } catch { /* ค่าพัง → ว่าง */ }
  }
  const out = [];
  for (const id of list) {
    if (typeof id === "string" && !out.includes(id) && (!isKnown || isKnown(id))) out.push(id);
    if (out.length >= max) break;
  }
  return out;
}

// เพิ่ม opId ที่เพิ่งใช้ไว้หน้าสุด (ถ้ามีอยู่แล้วย้ายขึ้นมา) → รายการใหม่ ไม่แก้ของเดิม
export function pushRecent(list, opId, max = PALETTE_RECENT_MAX) {
  if (typeof opId !== "string" || !opId) return list.slice(0, max);
  return [opId, ...list.filter((x) => x !== opId)].slice(0, max);
}

// ประวัติ "ใช้ล่าสุด" เก่าที่เก็บ "morphology" → "morph_erode" (ไม่ซ้ำ) · ค่าพัง/ไม่ใช่ JSON array คืนตามเดิมให้ resolveRecent จัดการ
export function migrateRecentIds(raw) {
  if (typeof raw !== "string") return raw;
  try {
    const a = JSON.parse(raw);
    if (!Array.isArray(a)) return raw;
    return JSON.stringify(a.map((id) => (id === "morphology" ? "morph_erode" : id)));
  } catch { return raw; }
}
function loadRecent() {
  let raw = null;
  try { raw = localStorage.getItem(PALETTE_RECENT_KEY); } catch { /* ไม่มีรายการ */ }
  // "morphology" (เครื่องมือเดี่ยวเดิมก่อน 2026-10-18) → morph_erode (ค่าเริ่มต้นเดิม) ไม่ทิ้งประวัติ
  return resolveRecent(migrateRecentIds(raw), (id) => !!OP_TO_FILTER[id] && !!OPS[id]);
}
function saveRecent(list) {
  try { localStorage.setItem(PALETTE_RECENT_KEY, JSON.stringify(list)); } catch { /* ไม่บันทึกก็ไม่เป็นไร */ }
}

// ค้นหาเครื่องมือ: ตัวพิมพ์เล็ก/ใหญ่ไม่สำคัญ · ค้นจากชื่อและรหัสภายใน (เช่น "otsu") · คำค้นว่าง = ตรงทุกตัว
export function toolMatches(name, opId, query) {
  const q = String(query || "").trim().toLowerCase();
  if (!q) return true;
  return String(name || "").toLowerCase().includes(q) || String(opId || "").toLowerCase().includes(q);
}

// cb: onAdd(filterId, opId) — each visible tile is one concrete Function,
// while the graph still creates the compatible category node and presets its mode.
export function createPalettePanel(container, cb) {
  let query = ""; // คำค้นเครื่องมือ (คงไว้ข้ามการ re-render)
  let paletteSearchEl = null; // input ช่องค้นหาปัจจุบัน (re-render ใหม่ทุกครั้ง) — ให้ Ctrl+/ โฟกัสได้
  let sections = [];  // [{ title, color, wrap, head, list, countEl, rows: [{el, name, opId}] }] — rows ใน list = ผลค้นหาแบบ inline
  let aiRow = null;   // { wrap, el, name } แถว Model Detect (ถ้ามี)
  let emptyEl = null; // "ไม่พบเครื่องมือ"
  let recent = loadRecent(); // opId ที่ใช้ล่าสุด (ใหม่สุดก่อน)
  let recentBlock = null; // { wrap, list, rows: [{el, name, opId}] } — ซ่อนทั้งก้อนเมื่อว่าง/ไม่ตรงคำค้น
  // popup ของหมวด (ไม่จำข้ามการเปิดแอป): openSection = หมวดที่เปิดอยู่ (หรือ null)
  let openSection = null;
  let pop = null; // { el, title, count, list, close } — สร้างใหม่ทุก render()

  const isSearching = () => query.trim() !== "";
  // node อยู่ใต้ root (หรือเป็น root เอง) ไหม — เดินตาม parentNode (ไม่พึ่ง Node.contains)
  const within = (root, node) => { for (let n = node; n; n = n.parentNode) if (n === root) return true; return false; };

  // ซ่อน/แสดงตามคำค้น โดยไม่ rebuild (คงโฟกัสช่องค้นหา)
  // ค้นอยู่ → ปิด popup แล้วแสดงเฉพาะเครื่องมือที่ตรงแบบ inline ใต้หัวหมวด · ไม่ค้น → เหลือแต่หัวหมวด (กดเปิด popup)
  function applyFilter() {
    const searching = isSearching();
    if (searching && openSection) closePopup(false);
    let total = 0;
    for (const s of sections) {
      let n = 0;
      for (const r of s.rows) {
        const m = toolMatches(r.name + " " + s.title, r.opId, query); // รวมชื่อหมวด: ค้น "morphology" เจอทั้งหมวด
        r.el.hidden = !m;
        if (m) n++;
      }
      total += n;
      s.wrap.hidden = searching && n === 0;
      s.list.hidden = !(searching && n > 0);
      s.head.setAttribute("aria-expanded", openSection === s ? "true" : "false");
      // ระหว่างค้นหา หัวหมวดไม่เปิด popup (ผลค้นหาแสดงใต้หัวหมวดแล้ว) — บอกผู้ใช้ด้วยสไตล์ + tooltip แทนการคลิกแล้วเงียบ
      s.head.setAttribute("aria-disabled", searching ? "true" : "false");
      s.head.title = searching ? "กำลังค้นหา — ผลลัพธ์แสดงใต้หมวดนี้ · ล้างคำค้นก่อนจึงเปิดหมวดเป็นหน้าต่างได้" : "";
      s.countEl.textContent = String(searching ? n : s.rows.length);
    }
    if (recentBlock) {
      let n = 0;
      for (const r of recentBlock.rows) {
        const m = toolMatches(r.name, r.opId, query);
        r.el.hidden = !m;
        if (m) n++;
      }
      recentBlock.wrap.hidden = n === 0; // ว่าง/ไม่ตรงคำค้น → ไม่เหลือหัวข้อโดดเดี่ยว
    }
    if (aiRow) {
      const m = toolMatches(aiRow.name, "model detect yolo", query);
      aiRow.wrap.hidden = !m;
      if (m) total++;
    }
    if (emptyEl) emptyEl.hidden = !(searching && total === 0);
  }

  // ---------- popup ของหมวด ----------
  // วางชิดขอบขวาของ palette · ระดับเดียวกับหัวหมวดที่กด · ไม่ล้นหน้าต่าง (fixed → ต้องอยู่ใต้ <body> เพราะ .panel-left เป็น
  // overflow + container-type ซึ่งจะตัด/ผูก position:fixed ของลูก)
  function positionPopup() {
    if (!openSection || !pop) return;
    if (typeof container.getBoundingClientRect !== "function" || typeof window === "undefined") return;
    const cr = container.getBoundingClientRect();
    if (!cr.width || !cr.height) { closePopup(false); return; } // palette ถูกซ่อน (collapse) → ไม่มีที่ให้ยึด
    const hr = openSection.head.getBoundingClientRect();
    const M = 8;
    const vw = window.innerWidth, vh = window.innerHeight;
    const left = Math.min(cr.right + 6, Math.max(M, vw - 200 - M));
    const st = pop.el.style;
    st.left = left + "px";
    st.width = Math.max(200, Math.min(360, vw - left - M)) + "px";
    st.maxHeight = Math.max(160, vh - 2 * M) + "px";
    const h = pop.el.offsetHeight || 0;
    st.top = Math.max(M, Math.min(hr.top - 6, vh - M - h)) + "px";
  }

  function openPopup(s, byKeyboard) {
    if (!pop || isSearching()) return;
    openSection = s;
    pop.title.textContent = s.title;
    pop.count.textContent = String(s.rows.length);
    pop.el.setAttribute("aria-label", s.title);
    pop.el.style.setProperty("--cat-color", s.color);
    pop.list.innerHTML = "";
    for (const r of buildRows(s.section)) pop.list.appendChild(r.el);
    pop.list.scrollTop = 0;
    pop.el.hidden = false;
    for (const x of sections) x.head.setAttribute("aria-expanded", x === s ? "true" : "false");
    positionPopup();
    if (byKeyboard) { const first = pop.list.querySelector(".tool-row"); (first || pop.close).focus(); }
  }

  // restoreFocus: Esc → ถ้าโฟกัสอยู่ใน popup/หัวหมวดที่เปิด ให้กลับไปที่หัวหมวด (ไม่แย่งโฟกัสจากที่อื่น)
  function closePopup(restoreFocus) {
    if (!openSection) return;
    const s = openSection;
    openSection = null;
    if (pop) pop.el.hidden = true;
    for (const x of sections) x.head.setAttribute("aria-expanded", "false");
    const active = typeof document !== "undefined" ? document.activeElement : null;
    if (restoreFocus && active && (within(pop && pop.el, active) || active === s.head)) s.head.focus();
  }

  function onHeadClick(s, ev) {
    if (isSearching()) return; // ค้นอยู่ → ผลแสดง inline แล้ว
    if (openSection === s) closePopup(false);
    else openPopup(s, !!ev && ev.detail === 0); // detail 0 = กดด้วยคีย์บอร์ด (Enter/Space)
  }

  function onEscape(ev) {
    if (!openSection || !ev || ev.key !== "Escape") return;
    if (typeof document === "undefined" || document.querySelector(".modal-overlay")) return;
    const active = document.activeElement;
    if (active && active !== document.body && !within(pop && pop.el, active) && !within(container, active)) return;
    ev.preventDefault();
    ev.stopImmediatePropagation();
    closePopup(true);
  }
  function onOutsidePointer(ev) {
    if (!openSection) return;
    const t = ev && ev.target;
    if (within(pop && pop.el, t) || within(container, t)) return;
    closePopup(false);
  }
  // listener ระดับหน้าต่าง: ผูกครั้งเดียวตอนสร้าง panel (render() ถูกเรียกซ้ำได้) · ไม่มี document/window (test) → ข้าม
  if (typeof document !== "undefined" && typeof document.addEventListener === "function") {
    document.addEventListener("keydown", onEscape, true);
    document.addEventListener("pointerdown", onOutsidePointer, true);
  }
  if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
    window.addEventListener("resize", positionPopup);
  }
  if (typeof container.addEventListener === "function") container.addEventListener("scroll", positionPopup); // หัวหมวดเลื่อน → popup ตามระดับ
  // ลากปรับความกว้าง/ซ่อนแถบ → ขอบขวาเปลี่ยน/หาย
  if (typeof ResizeObserver !== "undefined" && typeof Element !== "undefined" && container instanceof Element) new ResizeObserver(() => positionPopup()).observe(container);

  function buildPopup() {
    const el = document.createElement("div");
    el.className = "pal-popover";
    el.id = "pal-popover";
    el.setAttribute("role", "dialog");
    el.hidden = true;
    const head = document.createElement("div");
    head.className = "pal-pop-head";
    const title = document.createElement("span");
    title.className = "pal-pop-title";
    const count = document.createElement("span");
    count.className = "pal-pop-count mono";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "pal-pop-close";
    close.setAttribute("aria-label", "ปิดหน้าต่างหมวด");
    close.title = "ปิด (Esc)";
    close.textContent = "×";
    close.addEventListener("click", () => closePopup(true));
    head.append(title, count, close);
    const list = document.createElement("div");
    list.className = "pal-list pal-pop-list";
    el.append(head, list);
    return { el, title, count, list, close };
  }

  // tile สไตล์ KNIME: ชื่อสั้นอยู่บน · ใต้ชื่อเป็นกล่องไอคอนสีหมวด + สามเหลี่ยมพอร์ต (เข้าซ้าย / ออกขวา)
  // ชื่อเต็มอยู่ใน title (tooltip) · inputs = 2 → สามเหลี่ยมเข้าซ้อนสองอัน · ยังเป็น button.tool-row เหมือนเดิม
  function makeToolRow(label, iconText, color, title, onClick, ports = { inputs: 1, outputs: 1 }) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tool-row";
    b.title = title;
    b.style.setProperty("--tool-color", color);
    const nm = document.createElement("span");
    nm.className = "af-name";
    nm.textContent = shortToolName(label);
    const wrap = document.createElement("span");
    wrap.className = "af-ico-wrap";
    wrap.setAttribute("aria-hidden", "true");
    const mkPort = (cls) => { const p = document.createElement("span"); p.className = "af-port " + cls; return p; };
    if (ports.inputs === 2) wrap.append(mkPort("af-port-in af-port-top"), mkPort("af-port-in af-port-bot"));
    else if (ports.inputs > 0) wrap.appendChild(mkPort("af-port-in"));
    const ic = document.createElement("span");
    ic.className = "af-icon";
    ic.textContent = iconText;
    ic.style.background = color;
    ic.style.color = contrastOn(color);
    wrap.appendChild(ic);
    if (ports.outputs > 0) wrap.appendChild(mkPort("af-port-out"));
    b.append(nm, wrap);
    b.addEventListener("click", onClick);
    return b;
  }

  // tile ทั้งหมวด (ใช้ทั้งใน popup และผลค้นหา inline)
  function buildRows(section) {
    const rows = [];
    for (const opId of section.ops) {
      const filterId = OP_TO_FILTER[opId];
      const filter = FILTERS[filterId];
      const op = OPS[opId];
      if (!filter || !op) continue;
      const colors = GROUP_COLORS[filter.group] || { title: "#475569" };
      const el = makeToolRow(op.name, OP_ICON[opId] || GROUP_ICON[filterId] || "◇", colors.title,
        `${op.name} — Add this Function to Pipeline Canvas`, () => addTool(filterId, opId), { inputs: filter.inputs === 2 ? 2 : 1, outputs: 1 });
      rows.push({ el, name: op.name, opId });
    }
    return rows;
  }

  // ทุกทางที่เพิ่มเครื่องมือ (แถวปกติ + แถวใน "ใช้ล่าสุด") ผ่านที่เดียว: จำ opId แล้วค่อยเรียก onAdd เดิม
  function addTool(filterId, opId) {
    recent = pushRecent(recent, opId);
    saveRecent(recent);
    cb.onAdd(filterId, opId);
    renderRecent();
    positionPopup(); // บล็อก "ใช้ล่าสุด" อาจเพิ่ม/ลดความสูง → หัวหมวดขยับ · popup คงเปิดไว้ให้เพิ่มหลายตัว
  }

  // สร้างบล็อก "ใช้ล่าสุด" ใหม่ในที่เดิม (ไม่ rebuild ทั้งแผง → ไม่เสียตำแหน่งเลื่อน/โฟกัสช่องค้นหา)
  function renderRecent() {
    const fresh = document.createElement("div");
    fresh.className = "palette-recent";
    const t = document.createElement("div");
    t.className = "palette-section-title";
    t.textContent = "ใช้ล่าสุด · RECENT";
    fresh.appendChild(t);
    const rows = [];
    for (const opId of recent) {
      const filterId = OP_TO_FILTER[opId];
      const filter = FILTERS[filterId];
      const op = OPS[opId];
      if (!filter || !op) continue;
      const colors = GROUP_COLORS[filter.group] || { title: "#475569" };
      const el = makeToolRow(op.name, OP_ICON[opId] || GROUP_ICON[filterId] || "◇", colors.title,
        `${op.name} — เพิ่มลงกระดานอีกครั้ง`, () => addTool(filterId, opId), { inputs: filter.inputs === 2 ? 2 : 1, outputs: 1 });
      fresh.appendChild(el);
      rows.push({ el, name: op.name, opId });
    }
    // แถวที่โฟกัสอยู่ (เช่นกดคีย์บอร์ดเพิ่มเครื่องมือ) ถูกแทนที่ทั้งก้อน → คืนโฟกัสให้แถวที่ opId เดียวกัน ไม่ให้หลุดไป body
    const active = typeof document !== "undefined" ? document.activeElement : null;
    const focusedOp = recentBlock && active ? (recentBlock.rows.find((r) => r.el === active) || {}).opId : null;
    if (recentBlock && recentBlock.wrap.isConnected) recentBlock.wrap.replaceWith(fresh);
    recentBlock = { wrap: fresh, rows };
    if (focusedOp) { const again = rows.find((r) => r.opId === focusedOp); if (again) again.el.focus(); }
    if (!rows.length) fresh.hidden = true;
    applyFilter();
    return fresh;
  }

  function render() {
    closePopup(false);
    if (pop && pop.el.remove) pop.el.remove(); // popup อยู่ใต้ <body> → ต้องเอาออกเองก่อนสร้างใหม่
    container.innerHTML = "";
    sections = [];
    aiRow = null;
    recentBlock = null;

    const h = document.createElement("h2");
    const hText = document.createElement("span");
    hText.textContent = "เครื่องมือ · Tools";
    h.appendChild(hText);
    // ปุ่มซ่อนแถบ — layout-resize.js ฟังคลิก [data-collapse] ที่ document (render นี้สร้างปุ่มใหม่ทุกครั้ง)
    const collapseBtn = document.createElement("button");
    collapseBtn.type = "button";
    collapseBtn.className = "panel-collapse-btn";
    collapseBtn.setAttribute("data-collapse", "palette");
    collapseBtn.setAttribute("aria-controls", "palette-panel");
    collapseBtn.setAttribute("aria-label", "ซ่อนแถบเครื่องมือ");
    collapseBtn.title = "ซ่อนแถบเครื่องมือ";
    collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false"><path d="M15 5l-7 7 7 7" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    h.appendChild(collapseBtn);
    container.appendChild(h);

    // ช่องค้นหาเครื่องมือ (§4/§11) — icon ซ้าย + keycap "Ctrl /" ขวา (จางลงเมื่อโฟกัส)
    const searchWrap = document.createElement("div");
    searchWrap.className = "palette-search-wrap";
    const searchIcon = document.createElement("span");
    searchIcon.className = "ps-icon";
    searchIcon.textContent = "🔍";
    const search = document.createElement("input");
    search.type = "text";
    search.className = "palette-search";
    search.placeholder = "ค้นหาเครื่องมือ…";
    search.setAttribute("aria-label", "ค้นหาเครื่องมือ");
    search.value = query;
    search.addEventListener("input", () => { query = search.value; applyFilter(); }); // applyFilter ปิด popup เองเมื่อเริ่มค้น
    const searchHint = document.createElement("span");
    searchHint.className = "keycap ps-hint";
    searchHint.textContent = "Ctrl /";
    searchWrap.appendChild(searchIcon);
    searchWrap.appendChild(search);
    searchWrap.appendChild(searchHint);
    container.appendChild(searchWrap);
    paletteSearchEl = search; // ให้ global shortcut (Ctrl+/) โฟกัสได้

    // แถวโปรเจกต์: Template / + Display / Save as Template (ปุ่มรองสูง 28px)
    const proj = document.createElement("div");
    proj.className = "palette-project";
    const mkProj = (cls, text, title, fn) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "proj-btn " + cls;
      b.textContent = text;
      b.title = title;
      b.addEventListener("click", () => fn());
      proj.appendChild(b);
    };
    if (cb.onTemplates) mkProj("tmpl-btn", "▦ Template", "วาง pipeline ตัวอย่างสำเร็จรูปให้ทันที", cb.onTemplates);
    if (cb.onAddDisplay) mkProj("add-disp", "＋ Display", "เพิ่มกล่องผลลัพธ์อีกอัน แล้วลากสายจากจุดใดก็ได้มาต่อ เพื่อดูภาพขั้นนั้นแยกต่างหาก", cb.onAddDisplay);
    if (cb.onSaveTemplate) mkProj("tmpl-save-btn", "💾 Save as Template", "บันทึก pipeline บนกระดานตอนนี้เป็นเทมเพลต — เปิดใช้ซ้ำได้ที่ปุ่มเทมเพลต", cb.onSaveTemplate);
    if (proj.children.length) container.appendChild(proj);

    // "ใช้ล่าสุด" อยู่บนสุดของรายการเครื่องมือ (ใต้ปุ่มโปรเจกต์) — ว่างอยู่ = ซ่อน
    container.appendChild(renderRecent());

    // หมวด Deep Learning — เพิ่ม "กล่อง YOLO Detect" ลงกราฟ แล้ว browse โมเดลผ่านกล่อง
    if (cb.onAddDL) {
      const wrap = document.createElement("div");
      wrap.className = "palette-ai";
      const t = document.createElement("div");
      t.className = "palette-section-title";
      t.textContent = "AI & MODEL";
      const row = makeToolRow("Model Detect", "🧠", "#f472b6",
        "วางกล่อง YOLO Detect ลงกราฟ แล้วคลิกกล่องเพื่อโหลดไฟล์โมเดล (.onnx) และเริ่มตรวจจับ", () => cb.onAddDL());
      wrap.append(t, row);
      container.appendChild(wrap);
      aiRow = { wrap, el: row, name: "Model Detect" };
    }

    TOOL_SECTIONS.forEach((section) => {
      const wrap = document.createElement("div");
      wrap.className = "pal-section";

      // สีประจำหมวด = สีของกลุ่มของเครื่องมือตัวแรก (ใช้เป็นแถบซ้ายหัวหมวด + แถบบน popup)
      const firstFilter = FILTERS[OP_TO_FILTER[section.ops.find((id) => OP_TO_FILTER[id] && OPS[id])]];
      const color = ((firstFilter && GROUP_COLORS[firstFilter.group]) || { title: "#475569" }).title;

      const head = document.createElement("button");
      head.type = "button";
      head.className = "pal-head";
      head.style.setProperty("--cat-color", color);
      head.setAttribute("aria-haspopup", "dialog");
      head.setAttribute("aria-controls", "pal-popover");
      head.setAttribute("aria-expanded", "false");
      const chev = document.createElement("span");
      chev.className = "pal-chev";
      chev.setAttribute("aria-hidden", "true");
      chev.textContent = "▸"; // บอกว่า "เปิดออกด้านข้าง"
      const ttl = document.createElement("span");
      ttl.className = "pal-title";
      ttl.textContent = section.title;
      const countEl = document.createElement("span");
      countEl.className = "pal-count mono";
      head.append(chev, ttl, countEl);

      // รายการ inline: ซ่อนตามปกติ · โชว์เฉพาะตอนค้นหา
      const list = document.createElement("div");
      list.className = "pal-list";
      const rows = buildRows(section);
      for (const r of rows) list.appendChild(r.el);

      const s = { title: section.title, section, color, wrap, head, list, countEl, rows };
      head.addEventListener("click", (ev) => onHeadClick(s, ev));
      wrap.append(head, list);
      container.appendChild(wrap);
      sections.push(s);
    });

    emptyEl = document.createElement("div");
    emptyEl.className = "palette-empty";
    emptyEl.setAttribute("role", "status");
    emptyEl.textContent = "ไม่พบเครื่องมือ";
    emptyEl.hidden = true;
    container.appendChild(emptyEl);

    pop = buildPopup();
    (document.body || container).appendChild(pop.el);
    applyFilter(); // ใช้คำค้นที่ค้างไว้ (ถ้ามี)
  }

  // โฟกัส + เลือกข้อความในช่องค้นหา (ใช้กับคีย์ลัด Ctrl+/ — UI-only, ไม่ยุ่ง keymap ระบบ)
  function focusSearch() {
    if (paletteSearchEl) { paletteSearchEl.focus(); paletteSearchEl.select(); }
  }

  return { render, focusSearch };
}

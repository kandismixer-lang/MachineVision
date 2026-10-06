// Panel ซ้าย: palette ของ node — คลิกเพื่อเพิ่มลงผืนกราฟ (สไตล์ Node-RED)
import { FILTERS, OPS, GROUP_COLORS } from "../filters/registry.js";
import { GROUP_ICON } from "../filters/ui-meta.js";

export const TOOL_SECTIONS = [
  { title: "COLOR", ops: ["grayscale", "colorspace", "channel_split", "rgb_adjust", "color_inrange", "kmeans_segment", "slic", "white_balance", "invert"] },
  { title: "ENHANCE", ops: ["brightness_contrast", "gamma", "hist_equalize", "clahe", "contrast_stretch", "flatfield_correction", "unsharp", "laplacian_sharpen", "custom_kernel"] },
  { title: "GEOMETRY", ops: ["crop", "roi_polygon", "resize", "rotate", "flip", "perspective_warp", "undistort", "translate_shear"] },
  { title: "FILTER & DENOISE", ops: ["salt_pepper_noise", "gaussian_noise", "median_blur", "gaussian_blur", "box_blur", "bilateral", "wiener_filter"] },
  { title: "MORPHOLOGY", ops: ["morphology"] },
  { title: "SEGMENTATION & EDGE", ops: ["binary_threshold", "otsu_threshold", "adaptive_threshold", "multi_otsu_threshold", "canny", "sobel", "laplacian_edge", "scharr_edge", "prewitt_edge", "roberts_edge", "log_edge", "watershed", "grabcut", "region_growing"] },
  { title: "ANALYSIS & DETECTION", ops: ["find_contours", "connected_components", "distance_transform", "local_std_filter", "local_entropy_filter", "regionprops", "feret", "line_profile", "bounding_boxes", "hough_circles", "hough_lines", "harris_corners", "shi_tomasi_corners", "orb_keypoints", "akaze_keypoints", "fast_keypoints"] },
  { title: "FREQUENCY DOMAIN", ops: ["fft_magnitude", "fft_filter", "bandpass_notch"] },
  { title: "IMAGE ARITHMETIC", ops: ["imabsdiff", "imfuse", "mask", "imarith", "logical_ops", "inpaint", "hist_match"] },
  { title: "TEMPLATE MATCHING", ops: ["template_matching"] },
  { title: "IMAGE QUALITY", ops: ["psnr", "ssim"] },
  { title: "MOTION", ops: ["mog2", "farneback_optical_flow", "pyrlk_optical_flow"] },
];

// 1 หมวดบน palette = 1 node (2026-10-07): 7 หมวดที่ op กระจายหลายกลุ่มเดิมชี้ไปกลุ่มระดับหมวด (dropdown "เลือกวิธี"
// ครบทุก op ของหมวดตามลำดับ TOOL_SECTIONS) · กลุ่มเก่า (color_space, colour, blur, watershed, psnr, mog2 ...) ยังลงทะเบียนใน
// FILTERS เป็น legacy แต่ไม่อยู่ในลิสต์นี้ → ไม่โผล่บน palette · tests/category-merge.test.mjs ล็อกให้ทุกหมวดตรงกัน
const VISIBLE_FILTER_IDS = [
  "color_tools", "enhance_tools", "geometry", "filter_denoise", "morphology",
  "segment_edge", "analysis_detect",
  "frequency", "image_arithmetic",
  "template_matching", "image_quality", "motion",
];

const OP_ICON = {
  grayscale: "◐", colorspace: "🎨", channel_split: "◉", rgb_adjust: "🎚", color_inrange: "🌈", invert: "◑",
  kmeans_segment: "⬡", white_balance: "⚖",
  brightness_contrast: "☀", gamma: "γ", hist_equalize: "▥", clahe: "▦", unsharp: "✦", laplacian_sharpen: "◇",
  custom_kernel: "◫",
  // ใหม่ (2026-10-01, IP batch ENHANCE/MORPHOLOGY) — เช็คแล้วไม่ชนกับ glyph ใดในไฟล์นี้หรือ GROUP_ICON
  contrast_stretch: "↔", // ลูกศรซ้าย-ขวา สื่อ "ยืดช่วง"
  flatfield_correction: "🔆", // ดวงอาทิตย์มีรัศมี สื่อ "แก้แสง"
  crop: "⛝", resize: "⤢", rotate: "↻", flip: "⇋", perspective_warp: "⏢", undistort: "⊙",
  salt_pepper_noise: "⁙", median_blur: "●", gaussian_blur: "✺", box_blur: "▧", bilateral: "◎",
  morphology: "▣", binary_threshold: "◩", otsu_threshold: "◧", adaptive_threshold: "◨", multi_otsu_threshold: "◪",
  canny: "⌁", sobel: "∿", laplacian_edge: "◈", scharr_edge: "∾",
  watershed: "≋", grabcut: "✂", find_contours: "⌬", connected_components: "⠿", distance_transform: "◔",
  // ใหม่ (2026-09-30) — เช็คทุก glyph กับของเดิมทั้งไฟล์และกับ GROUP_ICON ใน ui-meta.js แล้วไม่ชนกันเลย
  prewitt_edge: "⊿", // สามเหลี่ยมมุมฉาก สื่อ "ความชัน" แบบตรงไปตรงมา
  roberts_edge: "⋱", // จุดเรียงแนวทแยง สื่อ kernel 2x2 ตามแนวทแยง
  log_edge: "⊚", // วงกลมซ้อน สื่อ "หมวกเม็กซิกัน" ของ LoG
  region_growing: "🪄", // ไม้กายสิทธิ์ = Magic Wand ตรงตัว
  local_std_filter: "▒", local_entropy_filter: "▓",
  // ใหม่ (Tier 2, 18c) — เช็คแล้วว่าไม่ชนกับ glyph ใดข้างบนทั้งหมด (~59 ตัว), 📏 (ไม้บรรทัด) สื่อ "การวัดค่า"
  regionprops: "📏",
  bounding_boxes: "□", hough_circles: "○", hough_lines: "╱", harris_corners: "⌖", shi_tomasi_corners: "✛",
  orb_keypoints: "✧", akaze_keypoints: "◆", fast_keypoints: "✪",
  fft_magnitude: "▨", fft_filter: "⬒",
  imabsdiff: "Δ", // ใหม่ (Tier 2, 18a) — เช็คแล้วว่าไม่ชนกับ glyph ใดข้างบน, Δ สื่อ "ความต่าง" ตรงตัว
  // ใหม่ (Tier 2, 18b) — เช็คทุกตัวกับ glyph เดิมทั้งหมดด้านบน (~57 ตัว) แล้วไม่ชนกันเลย:
  imfuse: "⊕", // วงกลม+บวก สื่อ "รวม/ผสม" สองภาพเข้าด้วยกัน
  mask: "⬚", // สี่เหลี่ยมประ สื่อ "หน้ากาก/พื้นที่ตัดออก"
  template_matching: "⌕", // แว่นขยาย-สแกน สื่อ "ค้นหา" — จงใจไม่ใช้ 🎯 (ชนกับ GROUP_ICON.object_detection ในไฟล์ ui-meta.js)
  psnr: "📶", // สัญญาณ สื่อ "signal-to-noise"
  ssim: "≈", // ประมาณเท่ากัน สื่อ "ความคล้าย/โครงสร้างใกล้เคียงกัน"
  // ใหม่ (2026-10-02, IP batch 2 ชิ้น B) — เช็คทุกตัวกับ glyph เดิมทั้งไฟล์และ GROUP_ICON ใน ui-meta.js แล้วไม่ชน
  // (± ของ GROUP_ICON.image_arithmetic จงใจไม่ใช้ซ้ำ)
  imarith: "÷", // เครื่องหมายหาร สื่อ "บวก/ลบ/คูณ/หาร"
  logical_ops: "∧", // AND ของตรรกศาสตร์
  inpaint: "🩹", // พลาสเตอร์ สื่อ "ซ่อมรอย"
  hist_match: "📈", // เส้นกราฟขาขึ้น สื่อ "CDF สะสม" — ไม่ใช้ 📊 เพราะ app.js/detail-panel.js ใช้นำหน้าข้อความผลลัพธ์ (debugger 2026-10-03)
  // ใหม่ (2026-10-03, IP batch 2 ชิ้น C) — เช็คทุกตัวกับ glyph เดิมทั้งไฟล์, GROUP_ICON ใน ui-meta.js และ app.js/detail-panel.js แล้วไม่ชน
  gaussian_noise: "░", // แรเงาเม็ดละเอียด สื่อ "เกรนทั่วภาพ" (▒/▓ ของ local_std/entropy เข้มกว่า แยกด้วยตาได้)
  wiener_filter: "◍", // วงกลมมีเส้นแนวตั้งด้านใน สื่อ "เกลี่ยแบบเลือกบริเวณ" (ต่างจาก ◉/◎/● ของกลุ่ม blur)
  translate_shear: "▱", // สี่เหลี่ยมด้านขนาน = รูปทรงของสี่เหลี่ยมที่ถูก shear ตรงตัว
  // ใหม่ (2026-10-03, IP batch 2 ชิ้น D) — เช็คทุกตัวกับ glyph เดิมทั้งไฟล์, GROUP_ICON และ app.js/detail-panel.js แล้วไม่ชน
  feret: "⌀", // สัญลักษณ์เส้นผ่านศูนย์กลาง สื่อ "วัดความกว้างแบบคาลิปเปอร์"
  line_profile: "〽", // เส้นกราฟขึ้นลง สื่อ "กราฟความสว่างตามเส้น" (ไม่ใช่ 〰 ของ GROUP_ICON.frequency)
  roi_polygon: "⬠", // ห้าเหลี่ยม สื่อ "บริเวณหลายเหลี่ยม" (ต่างจาก ⏢ perspective_warp / ⬡ kmeans)
  // ใหม่ (2026-10-04, IP batch 2 ชิ้น E) — เช็คแล้วไม่ชนกับ OP_ICON / GROUP_ICON / app.js / detail-panel.js / editor.js (🧩 ถูกใช้เป็น fallback ใน editor.js จึงไม่เลือก)
  slic: "⊞", // ตารางสี่เหลี่ยมแบ่งช่อง สื่อ "แบ่งภาพเป็นก้อนตามตาราง"
  bandpass_notch: "◙", // วงกลมซ้อนในกรอบ สื่อ "วงแหวนความถี่ที่เลือกเก็บ/ตัด"
  // ใหม่ (Tier 2, 18d) — เช็คทุกตัวกับ glyph เดิมทั้งหมดด้านบน และกับ GROUP_ICON ใน ui-meta.js แล้วไม่ชนกันเลย
  mog2: "🎞", // ฟิล์มหลายเฟรม สื่อ "ต้องดูหลายเฟรมต่อเนื่อง"
  farneback_optical_flow: "⇶", // ลูกศรสามเส้นขนานกัน สื่อ "การไหลทั้งภาพ" (dense flow)
  pyrlk_optical_flow: "⤳", // ลูกศรหางคลื่น สื่อ "เส้นทางของจุดที่ถูกติดตาม" (sparse flow)
};

export const OP_TO_FILTER = {};
for (const filterId of VISIBLE_FILTER_IDS) {
  const filter = FILTERS[filterId];
  const mode = filter.params.find((p) => p.key === "mode");
  if (mode) mode.options.forEach((o) => { OP_TO_FILTER[o.value] = filterId; });
  else OP_TO_FILTER[filterId] = filterId;
}

// ---------- สถานะพับ/กางของหมวด (Phase 2.3) ----------
// เก็บใน localStorage["mv-palette-open"] เป็น JSON { "COLOR": true, "ENHANCE": false, ... }
// ครั้งแรก (ไม่มีค่า/พัง) → เปิดเฉพาะ COLOR · หลังจากนั้นใช้ค่าที่บันทึกไว้ (หมวดที่ไม่เคยบันทึก = พับ)
export const PALETTE_OPEN_KEY = "mv-palette-open";
export const PALETTE_FIRST_OPEN = "COLOR";

// raw = ข้อความดิบจาก localStorage (หรือ null) → { [title]: boolean } ครบทุกหมวดใน titles
export function resolveOpenState(raw, titles) {
  let saved = null;
  if (typeof raw === "string") {
    try {
      const o = JSON.parse(raw);
      if (o && typeof o === "object" && !Array.isArray(o)) saved = o;
    } catch { /* ค่าพัง → ถือว่าครั้งแรก */ }
  }
  const out = {};
  for (const t of titles) out[t] = saved ? saved[t] === true : t === PALETTE_FIRST_OPEN;
  return out;
}

function loadOpenState(titles) {
  let raw = null;
  try { raw = localStorage.getItem(PALETTE_OPEN_KEY); } catch { /* ใช้ค่าเริ่มต้น */ }
  return resolveOpenState(raw, titles);
}
function saveOpenState(state) {
  try { localStorage.setItem(PALETTE_OPEN_KEY, JSON.stringify(state)); } catch { /* ไม่บันทึกก็ไม่เป็นไร */ }
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
  const openState = loadOpenState(TOOL_SECTIONS.map((s) => s.title));
  let sections = [];  // [{ title, head, list, countEl, rows: [{el, name, opId}] }]
  let aiRow = null;   // { wrap, el, name } แถว Model Detect (ถ้ามี)
  let emptyEl = null; // "ไม่พบเครื่องมือ"

  // ซ่อน/แสดงตามคำค้น โดยไม่ rebuild (คงโฟกัสช่องค้นหา)
  // ค้นอยู่ → หมวดที่มีผลตรงกางออกอัตโนมัติ (ไม่แตะค่าที่บันทึก) · ไม่ค้น → กลับไปตามสถานะที่บันทึก
  function applyFilter() {
    const searching = query.trim() !== "";
    let total = 0;
    for (const s of sections) {
      let n = 0;
      for (const r of s.rows) {
        const m = toolMatches(r.name, r.opId, query);
        r.el.hidden = !m;
        if (m) n++;
      }
      total += n;
      const show = !searching || n > 0;
      s.wrap.hidden = !show;
      const open = searching ? n > 0 : openState[s.title];
      s.list.hidden = !open;
      s.head.setAttribute("aria-expanded", open ? "true" : "false");
      // ระหว่างค้นหา การพับ/กางถูกล็อก (กางตามผลค้นหา) — บอกผู้ใช้ด้วยสไตล์ + tooltip แทนการคลิกแล้วเงียบ
      s.head.setAttribute("aria-disabled", searching ? "true" : "false");
      s.head.title = searching ? "กำลังค้นหา — ล้างคำค้นก่อนจึงพับ/กางหมวดเองได้" : "";
      s.countEl.textContent = String(searching ? n : s.rows.length);
    }
    if (aiRow) {
      const m = toolMatches(aiRow.name, "model detect yolo", query);
      aiRow.wrap.hidden = !m;
      if (m) total++;
    }
    if (emptyEl) emptyEl.hidden = !(searching && total === 0);
  }

  function toggleSection(s) {
    if (query.trim() !== "") return; // ระหว่างค้นหา การกางเป็นแบบอัตโนมัติ — ไม่ให้สลับ/บันทึกทับ
    openState[s.title] = !openState[s.title];
    saveOpenState(openState);
    applyFilter();
    // หมวดที่อยู่ล่างสุดของ palette กางออกไปใต้ขอบที่มองเห็น — ผู้ใช้เห็นแค่ลูกศรหมุนแล้วนึกว่ากดไม่ได้
    // จึงเลื่อนให้รายการที่เพิ่งกางเห็นได้ (สูงกว่ากรอบ → ชิดหัวหมวด)
    if (openState[s.title]) revealSection(s);
  }

  function revealSection(s) {
    const sc = container;
    if (typeof s.wrap.getBoundingClientRect !== "function" || typeof sc.getBoundingClientRect !== "function") return;
    // ตำแหน่งในพิกัดเนื้อหาของกรอบ scroll (ไม่พึ่ง offsetParent)
    const top = s.wrap.getBoundingClientRect().top - sc.getBoundingClientRect().top + sc.scrollTop;
    const bottom = top + s.wrap.offsetHeight;
    let target = null;
    if (s.wrap.offsetHeight > sc.clientHeight || top < sc.scrollTop) target = top;
    else if (bottom > sc.scrollTop + sc.clientHeight) target = bottom - sc.clientHeight;
    if (target != null) sc.scrollTop = Math.max(0, Math.round(target));
  }

  function makeToolRow(label, iconText, color, title, onClick) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tool-row";
    b.title = title;
    b.style.setProperty("--tool-color", color);
    const ic = document.createElement("span");
    ic.className = "af-icon";
    ic.textContent = iconText;
    ic.style.background = `color-mix(in srgb, ${color} 22%, transparent)`;
    ic.style.borderColor = `color-mix(in srgb, ${color} 55%, transparent)`;
    const nm = document.createElement("span");
    nm.className = "af-name";
    nm.textContent = label;
    b.append(ic, nm);
    b.addEventListener("click", onClick);
    return b;
  }

  function render() {
    container.innerHTML = "";
    sections = [];
    aiRow = null;

    const h = document.createElement("h2");
    h.textContent = "เครื่องมือ · Tools";
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
    search.addEventListener("input", () => { query = search.value; applyFilter(); });
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

    TOOL_SECTIONS.forEach((section, si) => {
      const wrap = document.createElement("div");
      wrap.className = "pal-section";
      const listId = "pal-list-" + si;

      const head = document.createElement("button");
      head.type = "button";
      head.className = "pal-head";
      head.setAttribute("aria-controls", listId);
      head.setAttribute("aria-expanded", "false");
      const chev = document.createElement("span");
      chev.className = "pal-chev";
      chev.setAttribute("aria-hidden", "true");
      chev.textContent = "▾";
      const ttl = document.createElement("span");
      ttl.className = "pal-title";
      ttl.textContent = section.title;
      const countEl = document.createElement("span");
      countEl.className = "pal-count mono";
      head.append(chev, ttl, countEl);

      const list = document.createElement("div");
      list.className = "pal-list";
      list.id = listId;

      const rows = [];
      for (const opId of section.ops) {
        const filterId = OP_TO_FILTER[opId];
        const filter = FILTERS[filterId];
        const op = OPS[opId];
        if (!filter || !op) continue;
        const colors = GROUP_COLORS[filter.group] || { title: "#475569" };
        const el = makeToolRow(op.name, OP_ICON[opId] || GROUP_ICON[filterId] || "◇", colors.title,
          `${op.name} — Add this Function to Pipeline Canvas`, () => cb.onAdd(filterId, opId));
        list.appendChild(el);
        rows.push({ el, name: op.name, opId });
      }

      const s = { title: section.title, wrap, head, list, countEl, rows };
      head.addEventListener("click", () => toggleSection(s));
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

    applyFilter(); // ใช้คำค้นที่ค้างไว้ (ถ้ามี) + สถานะพับ/กาง
  }

  // โฟกัส + เลือกข้อความในช่องค้นหา (ใช้กับคีย์ลัด Ctrl+/ — UI-only, ไม่ยุ่ง keymap ระบบ)
  function focusSearch() {
    if (paletteSearchEl) { paletteSearchEl.focus(); paletteSearchEl.select(); }
  }

  return { render, focusSearch };
}

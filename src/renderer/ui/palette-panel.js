// Panel ซ้าย: palette ของ node — คลิกเพื่อเพิ่มลงผืนกราฟ (สไตล์ Node-RED)
import { FILTERS, OPS, GROUP_COLORS } from "../filters/registry.js";
import { GROUP_ICON } from "../filters/ui-meta.js";

export const TOOL_SECTIONS = [
  { title: "COLOR", ops: ["grayscale", "colorspace", "channel_split", "color_inrange", "invert"] },
  { title: "ENHANCE", ops: ["brightness_contrast", "gamma", "hist_equalize", "clahe", "unsharp", "laplacian_sharpen"] },
  { title: "FILTER & DENOISE", ops: ["salt_pepper_noise", "median_blur", "gaussian_blur", "box_blur", "bilateral"] },
  { title: "MORPHOLOGY", ops: ["morphology"] },
  { title: "SEGMENTATION & EDGE", ops: ["binary_threshold", "otsu_threshold", "adaptive_threshold", "canny", "sobel", "watershed"] },
  { title: "ANALYSIS & DETECTION", ops: ["find_contours", "connected_components", "bounding_boxes", "hough_circles", "harris_corners", "shi_tomasi_corners"] },
];

const VISIBLE_FILTER_IDS = [
  "color_space", "image_adjust", "sharpen", "noise", "blur", "morphology",
  "threshold", "edge_detection", "shape_analysis", "object_detection",
  "watershed",
];

const OP_ICON = {
  grayscale: "◐", colorspace: "🎨", channel_split: "◉", color_inrange: "🌈", invert: "◑",
  brightness_contrast: "☀", gamma: "γ", hist_equalize: "▥", clahe: "▦", unsharp: "✦", laplacian_sharpen: "◇",
  salt_pepper_noise: "⁙", median_blur: "●", gaussian_blur: "✺", box_blur: "▧", bilateral: "◎",
  morphology: "▣", binary_threshold: "◩", otsu_threshold: "◧", adaptive_threshold: "◨",
  canny: "⌁", sobel: "∿", watershed: "≋", find_contours: "⌬", connected_components: "⠿",
  bounding_boxes: "□", hough_circles: "○", harris_corners: "⌖", shi_tomasi_corners: "✛",
};

export const OP_TO_FILTER = {};
for (const filterId of VISIBLE_FILTER_IDS) {
  const filter = FILTERS[filterId];
  const mode = filter.params.find((p) => p.key === "mode");
  if (mode) mode.options.forEach((o) => { OP_TO_FILTER[o.value] = filterId; });
  else OP_TO_FILTER[filterId] = filterId;
}

// cb: onAdd(filterId, opId) — each visible tile is one concrete Function,
// while the graph still creates the compatible category node and presets its mode.
export function createPalettePanel(container, cb) {
  let query = ""; // คำค้นเครื่องมือ (คงไว้ข้ามการ re-render)
  let paletteSearchEl = null; // input ช่องค้นหาปัจจุบัน (re-render ใหม่ทุกครั้ง) — ให้ Ctrl+/ โฟกัสได้

  // ซ่อน/แสดงปุ่มตามคำค้น โดยไม่ rebuild (คงโฟกัสช่องค้นหา)
  function applyFilter() {
    const q = query.trim().toLowerCase();
    container.querySelectorAll(".palette-group").forEach((g) => {
      let any = false;
      g.querySelectorAll(".add-filter-btn").forEach((b) => {
        const match = !q || b.textContent.toLowerCase().includes(q);
        b.style.display = match ? "" : "none";
        if (match) any = true;
      });
      g.style.display = any ? "" : "none";
      const heading = g.previousElementSibling;
      if (heading?.classList.contains("palette-section-title")) {
        heading.style.display = any ? "" : "none";
      }
    });
  }

  function render() {
    container.innerHTML = "";

    const h = document.createElement("h2");
    h.textContent = "Tools";
    container.appendChild(h);

    const tip = document.createElement("div");
    tip.className = "palette-tip";
    tip.textContent = "คลิกเพื่อเพิ่มลงกระดาน แล้วลากสายต่อ: แหล่งภาพ → ประมวลผล → ผลลัพธ์";
    container.appendChild(tip);

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

    const actionsTitle = document.createElement("div");
    actionsTitle.className = "palette-section-title";
    actionsTitle.textContent = "PROJECT & OUTPUT";
    container.appendChild(actionsTitle);

    // ปุ่มเทมเพลตสำเร็จรูป (Design §9)
    if (cb.onTemplates) {
      const tb = document.createElement("button");
      tb.className = "add-display-btn tmpl-btn";
      tb.textContent = "▦ Template";
      tb.title = "วาง pipeline ตัวอย่างสำเร็จรูปให้ทันที";
      tb.addEventListener("click", () => cb.onTemplates());
      container.appendChild(tb);
    }

    // ปุ่มเซฟ pipeline ปัจจุบันเป็นเทมเพลต (บันทึกไว้ใช้ซ้ำ)
    if (cb.onSaveTemplate) {
      const sb = document.createElement("button");
      sb.className = "add-display-btn tmpl-save-btn";
      sb.textContent = "💾 Save as Template";
      sb.title = "บันทึก pipeline บนกระดานตอนนี้เป็นเทมเพลต — เปิดใช้ซ้ำได้ที่ปุ่มเทมเพลต";
      sb.addEventListener("click", () => cb.onSaveTemplate());
      container.appendChild(sb);
    }

    // ปุ่มเพิ่มกล่องผลลัพธ์ (ดูหลายผลลัพธ์พร้อมกัน)
    if (cb.onAddDisplay) {
      const db = document.createElement("button");
      db.className = "add-display-btn";
      db.textContent = "＋ Add Display";
      db.title = "เพิ่มกล่องผลลัพธ์อีกอัน แล้วลากสายจากจุดใดก็ได้มาต่อ เพื่อดูภาพขั้นนั้นแยกต่างหาก";
      db.addEventListener("click", () => cb.onAddDisplay());
      container.appendChild(db);
    }

    // หมวด Deep Learning — เพิ่ม "กล่อง YOLO Detect" ลงกราฟ แล้ว browse โมเดลผ่านกล่อง
    if (cb.onAddDL) {
      const aiSection = document.createElement("div");
      aiSection.className = "palette-section-title";
      aiSection.textContent = "AI & MODEL";
      container.appendChild(aiSection);
      const g = document.createElement("div");
      g.className = "palette-group";

      const mb = document.createElement("button");
      mb.className = "add-filter-btn";
      mb.style.borderLeft = "4px solid #f472b6";
      mb.textContent = "🧠 Model Detect";
      mb.title = "วางกล่อง YOLO Detect ลงกราฟ แล้วคลิกกล่องเพื่อโหลดไฟล์โมเดล (.onnx) และเริ่มตรวจจับ";
      mb.addEventListener("click", () => cb.onAddDL());
      g.appendChild(mb);
      container.appendChild(g);
    }

    for (const section of TOOL_SECTIONS) {
      const sectionTitle = document.createElement("div");
      sectionTitle.className = "palette-section-title";
      sectionTitle.textContent = section.title;
      container.appendChild(sectionTitle);
      const g = document.createElement("div");
      g.className = "palette-group palette-tools-grid";

      for (const opId of section.ops) {
        const filterId = OP_TO_FILTER[opId];
        const filter = FILTERS[filterId];
        const op = OPS[opId];
        if (!filter || !op) continue;
        const colors = GROUP_COLORS[filter.group] || { title: "#475569" };
        const b = document.createElement("button");
        b.className = "add-filter-btn tool-tile";
        b.title = `${op.name} — Add this Function to Pipeline Canvas`;
        b.style.setProperty("--tool-color", colors.title);

        const ic = document.createElement("span");
        ic.className = "af-icon";
        ic.textContent = OP_ICON[opId] || GROUP_ICON[filterId] || "◇";
        ic.style.background = `color-mix(in srgb, ${colors.title} 22%, transparent)`;
        ic.style.borderColor = `color-mix(in srgb, ${colors.title} 55%, transparent)`;
        b.appendChild(ic);

        const nm = document.createElement("span");
        nm.className = "af-name";
        nm.textContent = op.name;
        b.appendChild(nm);

        b.addEventListener("click", () => cb.onAdd(filterId, opId));
        g.appendChild(b);
      }
      container.appendChild(g);
    }

    applyFilter(); // ใช้คำค้นที่ค้างไว้ (ถ้ามี)
  }

  // โฟกัส + เลือกข้อความในช่องค้นหา (ใช้กับคีย์ลัด Ctrl+/ — UI-only, ไม่ยุ่ง keymap ระบบ)
  function focusSearch() {
    if (paletteSearchEl) { paletteSearchEl.focus(); paletteSearchEl.select(); }
  }

  return { render, focusSearch };
}

// Pixel inspector + ซูม/เลื่อน ต่อ 1 จอผลลัพธ์ (viewer tile) — UI เฟส 3.1
//  · เมาส์ชี้ → พิกัด x, y ของภาพ + ค่า (เทา หรือ R,G,B) ในแถบเล็กล่างจอ
//  · ล้อเมาส์ซูมรอบเคอร์เซอร์ · ลากเพื่อเลื่อน · ซูม ≥ 800% เห็นตารางพิกเซลและปิด smoothing · ดับเบิลคลิก/คีย์ 0 = พอดีจอ
//  · แต่ละจอเป็นอิสระต่อกัน · ไม่แตะ pixel ของ canvas (ซูมด้วย CSS transform) → ลูปวาดเฟรมของ app.js ไม่ช้าลง
//  · mousemove ไม่สร้างอ็อบเจ็กต์ใหม่: ใช้ rect ที่แคชไว้ + อ็อบเจ็กต์พิกัดตัวเดียว + อ่านค่าจาก ImageData ของเฟรมล่าสุดโดยตรง
//    (ไม่เรียก getImageData) และเขียน DOM เมื่อ "พิกเซลที่ชี้ เปลี่ยน" เท่านั้น
import {
  FIT_VIEW, PX_GRID_SCALE, clampView, zoomAt, panBy, wheelFactor, screenToImage, visiblePixelRange,
  displayScale, readRGB, formatPixel, formatPos, formatZoomPct,
} from "./phase3-math.js";

let hovered = null; // inspector ที่เมาส์ชี้อยู่ (ใช้กับคีย์ 0)

// คีย์ 0 → จอที่เมาส์ชี้อยู่กลับพอดีจอ · คืน true ถ้ามีจอที่จัดการให้
export function resetHoveredView() {
  if (!hovered) return false;
  hovered.resetView();
  return true;
}

// parts = { view, canvas, gridCanvas, readout, badge } (DOM) · opts.getImage() → ImageData ล่าสุดของจอนี้ หรือ null (เช่นจอ DL)
export function createTileInspector(parts, opts = {}) {
  const { view: viewEl, canvas, gridCanvas, readout, badge } = parts;
  const posEl = readout.querySelector(".px-pos");
  const valEl = readout.querySelector(".px-val");
  const swatch = readout.querySelector(".px-swatch");
  const gridCtx = gridCanvas.getContext("2d");

  let view = { z: 1, tx: 0, ty: 0 };
  const geom = { vw: 0, vh: 0, iw: 0, ih: 0 };
  let left = 0, top = 0; // มุมซ้ายบนของ viewport บนหน้า (แคช — อัปเดตตอนเมาส์เข้า/ซูม/กดเมาส์/ปรับขนาด ไม่ใช่ทุก mousemove)
  const hov = { x: 0, y: 0, inside: false }; // พิกัดภาพที่ชี้อยู่ (อ็อบเจ็กต์เดียวใช้ซ้ำ)
  let shownX = -1, shownY = -1; // พิกเซลที่แสดงอยู่ในแถบ (กันเขียน DOM ซ้ำ)
  const rgb = [0, 0, 0];
  let pan = null; // { id, x, y } ขณะลากเลื่อน
  let lastTransform = "";
  let ptrX = 0, ptrY = 0; // ตำแหน่งเมาส์ล่าสุด (clientX/Y) — ใช้คำนวณพิกเซลที่ชี้ใหม่เมื่อภาพเปลี่ยนขนาด
  let measuredAt = 0;
  let readAt = 0; // เวลาอ่านค่าครั้งล่าสุด (ใช้ throttle จอที่ไม่มี ImageData)
  let lastCrisp = false;
  let gridDirty = true;

  function measure() {
    const r = viewEl.getBoundingClientRect();
    left = r.left; top = r.top;
    geom.vw = r.width; geom.vh = r.height;
    geom.iw = canvas.width; geom.ih = canvas.height;
  }

  function applyView() {
    const t = view.z === 1 && view.tx === 0 && view.ty === 0 ? "" : `translate(${view.tx}px, ${view.ty}px) scale(${view.z})`;
    if (t !== lastTransform) { lastTransform = t; canvas.style.transform = t; }
    const s = displayScale(view, geom);
    const crisp = s >= PX_GRID_SCALE;
    if (crisp !== lastCrisp) { lastCrisp = crisp; canvas.classList.toggle("px-crisp", crisp); }
    const zoomed = view.z > 1.0001;
    viewEl.classList.toggle("is-zoomed", zoomed);
    badge.hidden = !zoomed;
    if (zoomed) badge.firstElementChild.textContent = formatZoomPct(s);
    gridDirty = true;
    drawGrid();
  }

  // ตารางพิกเซล + กรอบพิกเซลที่ชี้ — วาดเมื่อ ≥ 800% เท่านั้น (ที่สเกลนี้เส้นไม่เกินราวสองร้อย → path เดียว)
  function drawGrid() {
    const s = displayScale(view, geom);
    const on = s >= PX_GRID_SCALE && geom.iw > 0;
    if (!on) {
      if (gridCanvas.dataset.on === "1") { gridCtx.clearRect(0, 0, gridCanvas.width, gridCanvas.height); gridCanvas.dataset.on = "0"; }
      return;
    }
    const dpr = Math.min(3, Math.max(1, (typeof window !== "undefined" && window.devicePixelRatio) || 1));
    const bw = Math.round(geom.vw * dpr), bh = Math.round(geom.vh * dpr);
    if (gridCanvas.width !== bw || gridCanvas.height !== bh) { gridCanvas.width = bw; gridCanvas.height = bh; gridDirty = true; }
    gridCanvas.dataset.on = "1";
    gridCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    gridCtx.clearRect(0, 0, geom.vw, geom.vh);
    const r = visiblePixelRange(view, geom);
    gridCtx.lineWidth = 1;
    gridCtx.strokeStyle = "rgba(255,255,255,.28)";
    gridCtx.beginPath();
    for (let i = r.x0; i <= r.x1; i++) { const x = Math.round(r.ox + i * r.s) + 0.5; gridCtx.moveTo(x, 0); gridCtx.lineTo(x, geom.vh); }
    for (let j = r.y0; j <= r.y1; j++) { const y = Math.round(r.oy + j * r.s) + 0.5; gridCtx.moveTo(0, y); gridCtx.lineTo(geom.vw, y); }
    gridCtx.stroke();
    if (hov.inside) { // กรอบพิกเซลที่ชี้อยู่
      gridCtx.strokeStyle = "#8cc8ff"; // = --focus-color
      gridCtx.lineWidth = 2;
      gridCtx.strokeRect(r.ox + hov.x * r.s, r.oy + hov.y * r.s, r.s, r.s);
    }
    gridDirty = false;
  }

  function readPixel() {
    const img = opts.getImage ? opts.getImage() : null;
    if (img && img.width === geom.iw && img.height === geom.ih) return readRGB(img.data, img.width, img.height, hov.x, hov.y, rgb);
    // จอที่ไม่มี ImageData ของเฟรม (จอ Deep Learning วาดผ่าน drawImage) — อ่านทีละ 1 พิกเซล (ไม่ใช่เส้นทางหลัก)
    try {
      const d = canvas.getContext("2d").getImageData(hov.x, hov.y, 1, 1).data;
      rgb[0] = d[0]; rgb[1] = d[1]; rgb[2] = d[2];
      return rgb;
    } catch { return null; }
  }

  function showReadout(force) {
    if (!hov.inside || geom.iw === 0) { readout.hidden = true; shownX = shownY = -1; return; }
    if (!force && hov.x === shownX && hov.y === shownY) return;
    shownX = hov.x; shownY = hov.y;
    readAt = Date.now();
    const px = readPixel();
    readout.hidden = false;
    posEl.textContent = formatPos(hov.x, hov.y);
    valEl.textContent = formatPixel(px);
    swatch.style.background = px ? `rgb(${px[0]},${px[1]},${px[2]})` : "transparent";
    if (lastCrisp && gridCanvas.dataset.on === "1") drawGrid(); // ย้ายกรอบพิกเซลที่ชี้
  }

  function pointerPos(e) {
    ptrX = e.clientX; ptrY = e.clientY;
    return screenToImage(view, ptrX - left, ptrY - top, geom, hov);
  }

  viewEl.addEventListener("pointerenter", () => { hovered = api; measure(); });
  viewEl.addEventListener("pointerleave", () => {
    if (hovered === api) hovered = null;
    if (!pan) { hov.inside = false; showReadout(true); drawGrid(); }
  });
  viewEl.addEventListener("pointermove", (e) => {
    // ทาย "ตำแหน่ง tile ขยับโดยไม่เปลี่ยนขนาด" (เลย์เอาต์ข้างเคียงเปลี่ยน/หน้าเลื่อน): วัดซ้ำเป็นระยะ ไม่ใช่ทุก mousemove
    if (Date.now() - measuredAt > 250) { measure(); measuredAt = Date.now(); }
    if (pan && pan.id === e.pointerId) {
      view = panBy(view, e.clientX - pan.x, e.clientY - pan.y, geom);
      pan.x = e.clientX; pan.y = e.clientY;
      applyView();
    }
    pointerPos(e);
    showReadout(false);
  });
  viewEl.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || e.target === badge || badge.contains(e.target)) return;
    measure();
    if (view.z <= 1.0001) return; // พอดีจอ = ไม่มีอะไรให้เลื่อน
    pan = { id: e.pointerId, x: e.clientX, y: e.clientY };
    viewEl.classList.add("is-panning");
    try { viewEl.setPointerCapture(e.pointerId); } catch { /* ข้ามได้ */ }
    e.preventDefault();
  });
  const endPan = (e) => {
    if (!pan || pan.id !== e.pointerId) return;
    pan = null;
    viewEl.classList.remove("is-panning");
    try { viewEl.releasePointerCapture(e.pointerId); } catch { /* ข้ามได้ */ }
  };
  viewEl.addEventListener("pointerup", endPan);
  viewEl.addEventListener("pointercancel", endPan);
  viewEl.addEventListener("wheel", (e) => {
    if (geom.iw === 0) return;
    e.preventDefault(); // ล้อเมาส์บนจอผลลัพธ์ = ซูมภาพ (ไม่เลื่อนหน้า)
    measure();
    view = zoomAt(view, wheelFactor(e.deltaY), e.clientX - left, e.clientY - top, geom);
    applyView();
    pointerPos(e);
    showReadout(true);
  }, { passive: false });
  viewEl.addEventListener("dblclick", (e) => {
    if (badge.contains(e.target)) return;
    api.resetView();
  });
  badge.addEventListener("click", (e) => { e.stopPropagation(); api.resetView(); });

  // ขนาดจอเปลี่ยน (ลากแบ่งพื้นที่/เปลี่ยน layout) → วัดใหม่ + บีบมุมมองให้อยู่ในกรอบ
  const ro = typeof ResizeObserver !== "undefined"
    ? new ResizeObserver(() => { measure(); if (geom.vw > 0) { view = clampView(view, geom); applyView(); } })
    : null;
  if (ro) ro.observe(viewEl);

  const api = {
    // เฟรมใหม่เข้าจอนี้ (drawResults เรียกหลัง putImageData) — ขนาดภาพเปลี่ยน → พอดีจอใหม่ · กำลังชี้อยู่ → ค่าในแถบตามเฟรมสด
    onFrame() {
      if (geom.iw !== canvas.width || geom.ih !== canvas.height) {
        measure();
        view = { ...FIT_VIEW };
        applyView();
        // ภาพเปลี่ยนขนาดระหว่างที่ชี้อยู่ → พิกเซลใต้เมาส์เปลี่ยน คำนวณจากตำแหน่งเมาส์ล่าสุดใหม่
        if (hov.inside || shownX >= 0) { screenToImage(view, ptrX - left, ptrY - top, geom, hov); showReadout(true); return; }
      }
      if (!hov.inside) return;
      // จอที่ไม่มี ImageData (DL) ต้องอ่านผ่าน getImageData → อ่านซ้ำไม่ถี่กว่า ~4 ครั้ง/วินาที (พิกเซลที่ชี้เปลี่ยนยังอ่านทันทีใน pointermove)
      const hasImg = opts.getImage && opts.getImage();
      if (hasImg || Date.now() - readAt >= 250) showReadout(true);
    },
    resetView() {
      view = { ...FIT_VIEW };
      measure();
      applyView();
      if (hov.inside) showReadout(true);
    },
    getView: () => view,
    destroy() {
      if (ro) ro.disconnect();
      if (hovered === api) hovered = null;
    },
  };
  measure();
  return api;
}

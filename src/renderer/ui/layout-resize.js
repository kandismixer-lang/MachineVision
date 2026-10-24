// ลากปรับขนาดเลย์เอาต์ (Version2ui เท่านั้น) — spec: specs/resizable_layout_spec.md
// 3 ตัวจับ: A = ความกว้างแถบเครื่องมือ (palette), B = ความกว้างแผงคุณสมบัติ (props),
// C = แบ่งความสูง จอผลลัพธ์ (.stage) กับ Pipeline (.graph-wrap)
// หลักการ: `state` = ค่าที่ผู้ใช้ตั้ง (null = ค่าเริ่มต้นของ CSS) · render() คำนวณค่าจริงหลัง clamp
// แล้วเขียน CSS var แบบ idempotent — ไม่มีค่าที่เก็บไว้ = ไม่มี inline var = หน้าตาเหมือนเดิมทุกพิกเซล
import { LIMITS, RATIO, RAIL_W, clamp, load, save } from "./layout-state.js";
const STEP = { col: 16, colBig: 64, row: 24, rowBig: 96 };
const DRAG_THRESHOLD = 3;
const FIELD = { palette: "paletteW", props: "propsW", stage: "stageRatio" };
const VAR = { palette: "--palette-w", props: "--props-w" };
const NAME = { palette: "แถบเครื่องมือ", props: "แผงคุณสมบัติ" };
const HIDDEN = { palette: "paletteHidden", props: "propsHidden" };

// preset สำหรับเมนู "มุมมอง" (Phase 2.4 — ทางเลือกแทนการลาก, WCAG 2.5.7) · ค่าผ่านการ clamp/persist ทางเดียวกับการลาก
export const STAGE_PRESETS = [
  { id: "stage-big", label: "ภาพใหญ่", hint: "70%", ratio: 0.70 },
  { id: "stage-balanced", label: "สมดุล", hint: "56%", ratio: 0.56 },
  { id: "graph-big", label: "กระดานใหญ่", hint: "35%", ratio: 0.35 },
];
export const WIDTH_PRESETS = {
  palette: { label: "แถบเครื่องมือ (ซ้าย)", narrow: 220, wide: 320 },
  props: { label: "แผงคุณสมบัติ (ขวา)", narrow: 300, wide: 440 },
};

const px = (s) => parseFloat(s) || 0;

let live = null; // instance ปัจจุบันของ initLayoutResize (เมนู "มุมมอง" เรียกผ่าน setStageRatio / setPanelWidth)
const subscribers = new Set(); // แจ้งเมื่อสถานะซ่อน/แสดงแผงเปลี่ยน (เมนู/คีย์ลัด/ปุ่มอัปเดต aria)

// ตั้งสัดส่วนจอผลลัพธ์ : กระดาน (0.1–0.9) แล้วบันทึกเหมือนการลาก · คืน false ถ้ายังไม่พร้อม (ไม่ใช่ ui-v2)
export function setStageRatio(r) { return live ? live.setStageRatio(r) : false; }
// ตั้งความกว้างแถบเครื่องมือ/แผงคุณสมบัติ (px) — clamp ตามขอบเขตเดียวกับการลาก
export function setPanelWidth(which, w) { return live ? live.setPanelWidth(which, w) : false; }
// ซ่อน/แสดงแถบเครื่องมือ (which="palette") หรือแผงขวา ("props") — เหลือ rail 32px ให้กดเปิดกลับ · ความกว้างเดิมที่ผู้ใช้ตั้งไว้ถูกเก็บและคืนตอนเปิด
export function setPanelCollapsed(which, hidden) { return live ? live.setPanelCollapsed(which, hidden) : false; }
export function togglePanel(which) { return live ? live.togglePanel(which) : false; }
export function isPanelCollapsed(which) { return live ? live.isPanelCollapsed(which) : false; }
export function onPanelCollapseChange(fn) { subscribers.add(fn); return () => subscribers.delete(fn); }

export function initLayoutResize() {
  const noop = {
    reset() {}, setStageRatio() { return false; }, setPanelWidth() { return false; }, snapshot() { return null; },
    setPanelCollapsed() { return false; }, togglePanel() { return false; }, isPanelCollapsed() { return false; },
  };
  if (!document.body.classList.contains("ui-v2")) return noop;
  const ws = document.querySelector(".workspace");
  const center = document.querySelector(".center-col");
  const stage = center?.querySelector(":scope > .stage");
  const tray = document.getElementById("capture-tray");
  const palette = document.getElementById("palette-panel");
  const props = document.getElementById("props-panel") || document.querySelector(".panel-right");
  const handles = [...document.querySelectorAll(".lr-handle[data-lr]")];
  if (!ws || !center || !stage || !palette || !props || !handles.length) return noop;
  const panelEl = { palette, props };
  const rails = { palette: document.getElementById("palette-rail"), props: document.getElementById("props-rail") };

  const state = load();
  const isHidden = (w) => !!state[HIDDEN[w]];
  // กว้างจริงของแผงในตอนนี้ (แผงที่ซ่อน = ความกว้าง rail) — ใช้คำนวณพื้นที่ที่เหลือให้ฝั่งตรงข้าม
  const widthOf = (w) => (isHidden(w) ? RAIL_W : panelEl[w].getBoundingClientRect().width);
  const readout = document.createElement("div");
  readout.className = "lr-readout";
  readout.setAttribute("aria-hidden", "true");
  document.body.appendChild(readout);

  let drag = null;          // สถานะการลากปัจจุบัน
  let clampRaf = 0;
  let saveTimer = 0;
  let readoutTimer = 0;
  let animTimer = 0;

  // ---------- วัดขนาด (อ่านอย่างเดียว) ----------
  function measurable() {
    return ws.clientWidth > 0 && center.clientHeight > 0 && !stage.classList.contains("stage-fullscreen");
  }
  function measureCols() {
    const cs = getComputedStyle(ws);
    const inner = ws.clientWidth - px(cs.paddingLeft) - px(cs.paddingRight) - 2 * px(cs.columnGap);
    return {
      inner,
      def: {
        palette: px(cs.getPropertyValue("--palette-w-def")) || palette.getBoundingClientRect().width,
        props: px(cs.getPropertyValue("--props-w-def")) || props.getBoundingClientRect().width,
      },
    };
  }
  function measureRows() {
    const cs = getComputedStyle(center);
    const gap = px(cs.rowGap);
    const trayShown = !!tray && !tray.classList.contains("hidden");
    const trayH = trayShown ? tray.getBoundingClientRect().height : 0;
    const inner = center.clientHeight - px(cs.paddingTop) - px(cs.paddingBottom);
    return { avail: inner - (trayShown ? 2 * gap + trayH : gap) };
  }
  // ความสูงจอผลลัพธ์ที่ใช้ได้จริง (ตาม §5: ถ้าพื้นที่ไม่พอทั้งคู่ แบ่ง 180:150)
  function clampStagePx(v, avail) {
    const { stageMinH: a, graphMinH: b } = LIMITS;
    if (avail < a + b) return (avail * a) / (a + b);
    return clamp(v, a, avail - b);
  }
  // ความกว้างด้านที่ลาก เมื่อด้านตรงข้ามกว้าง otherW
  function colMax(which, inner, otherW) {
    return Math.max(LIMITS[which].min, Math.min(LIMITS[which].max, inner - otherW - LIMITS.centerMinW));
  }

  // ---------- เขียน (idempotent: ไม่ set ซ้ำถ้าค่าเดิม → ไม่ปลุก ResizeObserver วนลูป) ----------
  function setVar(el, name, value) {
    if (value == null) {
      if (el.style.getPropertyValue(name) !== "") el.style.removeProperty(name);
      if (el.style.length === 0 && el.hasAttribute("style")) el.removeAttribute("style"); // ไม่ทิ้ง style="" ค้าง
    } else if (el.style.getPropertyValue(name) !== value) {
      el.style.setProperty(name, value);
    }
  }
  function writeWidth(which, w) { setVar(ws, VAR[which], w == null ? null : `${Math.round(w)}px`); }
  function writeSplit(k) {
    if (k == null) { setVar(center, "--stage-fr", null); setVar(center, "--graph-fr", null); return; }
    k = clamp(Math.round(k), 1, 999);
    setVar(center, "--stage-fr", `${k}fr`);
    setVar(center, "--graph-fr", `${1000 - k}fr`);
  }

  // คำนวณค่าจริงของคอลัมน์จาก state (หดตามสัดส่วนส่วนเกินเหนือ min เมื่อจอแคบ)
  function effectiveCols(m) {
    const eff = {};
    let total = 0;
    for (const w of ["palette", "props"]) {
      const v = state[FIELD[w]];
      eff[w] = v == null ? null : clamp(v, LIMITS[w].min, LIMITS[w].max);
      total += isHidden(w) ? RAIL_W : (eff[w] ?? m.def[w]);
    }
    const overflow = total - (m.inner - LIMITS.centerMinW);
    if (overflow > 0) {
      const ex = {};
      let sumEx = 0;
      for (const w of ["palette", "props"]) {
        ex[w] = eff[w] == null || isHidden(w) ? 0 : eff[w] - LIMITS[w].min; // แผงที่ซ่อนไม่ถูกหด (เก็บค่าเดิมไว้คืนตอนเปิด)
        sumEx += ex[w];
      }
      if (sumEx > 0) {
        const take = Math.min(overflow, sumEx);
        for (const w of ["palette", "props"]) if (eff[w] != null && !isHidden(w)) eff[w] -= (take * ex[w]) / sumEx;
      }
    }
    return eff;
  }

  // render: state → CSS var (ข้ามเมื่อกำลังลาก — ตอนลาก drag loop เป็นเจ้าของ var)
  function render() {
    if (drag) return;
    if (!measurable()) {
      // ยังวัดไม่ได้ (แอปซ่อนอยู่ / เต็มจอ) → ใส่ค่าดิบไว้ก่อน แล้ว clamp ภายหลังใน ResizeObserver
      if (stage.classList.contains("stage-fullscreen")) return;
      writeWidth("palette", state.paletteW);
      writeWidth("props", state.propsW);
      writeSplit(state.stageRatio == null ? null : state.stageRatio * 1000);
      return;
    }
    const mc = measureCols();
    const eff = effectiveCols(mc);
    const { avail } = measureRows();
    let k = null;
    if (state.stageRatio != null && avail > 0) k = (clampStagePx(state.stageRatio * avail, avail) / avail) * 1000;
    writeWidth("palette", eff.palette);
    writeWidth("props", eff.props);
    if (avail > 0) writeSplit(k);
    updateAria(mc, eff, avail);
  }
  function scheduleRender() {
    if (clampRaf) return;
    clampRaf = requestAnimationFrame(() => { clampRaf = 0; render(); });
  }

  // ---------- ARIA ----------
  function handleOf(which) { return handles.find((h) => h.dataset.lr === which); }
  function setAria(h, min, max, now, text) {
    if (!h) return;
    h.setAttribute("aria-valuemin", String(Math.round(min)));
    h.setAttribute("aria-valuemax", String(Math.round(max)));
    h.setAttribute("aria-valuenow", String(Math.round(now)));
    h.setAttribute("aria-valuetext", text);
  }
  function ariaCol(which, w, max) { setAria(handleOf(which), LIMITS[which].min, max, w, `กว้าง ${Math.round(w)} พิกเซล`); }
  function ariaRow(stagePx, avail) {
    if (!(avail > 0)) return;
    const s = splitPct(stagePx, avail);
    const { stageMinH: a, graphMinH: b } = LIMITS;
    // พื้นที่ไม่พอทั้งคู่ → สัดส่วนถูกล็อกที่ 180:150 (min = max = ค่าปัจจุบัน ไม่ให้ช่วงกลับหัว)
    const lo = avail < a + b ? a / (a + b) : a / avail;
    const hi = avail < a + b ? lo : (avail - b) / avail;
    setAria(handleOf("stage"), lo * 100, hi * 100, s, splitText(s));
  }
  function updateAria(mc, eff, avail) {
    const pw = isHidden("palette") ? RAIL_W : (eff.palette ?? mc.def.palette);
    const qw = isHidden("props") ? RAIL_W : (eff.props ?? mc.def.props);
    ariaCol("palette", pw, colMax("palette", mc.inner, qw));
    ariaCol("props", qw, colMax("props", mc.inner, pw));
    if (avail > 0) {
      const sp = state.stageRatio == null ? stage.getBoundingClientRect().height
        : clampStagePx(state.stageRatio * avail, avail);
      ariaRow(sp, avail);
    }
  }
  const splitPct = (stagePx, avail) => clamp(Math.round((stagePx / avail) * 100), 0, 100);
  const splitText = (s) => `จอผลลัพธ์ ${s}% · Pipeline ${100 - s}%`;

  // ---------- readout pill ----------
  function showReadout(text, limit, pos) {
    clearTimeout(readoutTimer);
    readout.textContent = text;
    if (limit) {
      const span = document.createElement("span");
      span.className = "lr-limit";
      span.textContent = "· สุดแล้ว";
      readout.appendChild(span);
    }
    readout.classList.add("show");
    const w = readout.offsetWidth, h = readout.offsetHeight;
    const x = clamp(pos.x, 4, Math.max(4, window.innerWidth - w - 4));
    const y = clamp(pos.y, 4, Math.max(4, window.innerHeight - h - 4));
    readout.style.left = `${x}px`;
    readout.style.top = `${y}px`;
  }
  // timer ใช้ร่วมทุกตัวจับ → ตอนซ่อนล้าง is-limit ทุกตัว (กันเส้นส้มค้างเมื่อสลับไปจับอีกตัวก่อนหมดเวลา)
  function clearLimits(except) {
    for (const x of handles) if (x !== except) x.classList.remove("is-limit");
  }
  function hideReadout(delay) {
    clearTimeout(readoutTimer);
    readoutTimer = setTimeout(() => {
      readout.classList.remove("show");
      clearLimits(null);
    }, delay);
  }
  function nearHandle(h) {
    const r = h.getBoundingClientRect();
    return h.dataset.lr === "stage" ? { x: r.left + r.width / 2 - 60, y: r.bottom + 6 } : { x: r.right + 6, y: r.top + r.height / 2 - 12 };
  }

  // ---------- คำนวณค่าใหม่ของตัวจับ (ใช้ร่วม drag + คีย์บอร์ด) ----------
  // ctx = ค่าที่วัดครั้งเดียวตอนเริ่ม: inner, otherW (คอลัมน์) / avail (แถว)
  function computeValue(which, desired, ctx) {
    if (which === "stage") {
      const v = clampStagePx(desired, ctx.avail);
      return { v, limit: Math.abs(v - desired) > 0.5 };
    }
    const max = colMax(which, ctx.inner, ctx.otherW);
    const v = Math.round(clamp(desired, LIMITS[which].min, max));
    return { v, max, limit: Math.abs(v - desired) > 0.5 };
  }
  // เขียนค่าชั่วคราวระหว่างลาก (write=false สำหรับคีย์บอร์ด) + aria + readout
  function applyLive(which, res, ctx, pos, h, write = true) {
    if (which === "stage") {
      if (write) writeSplit((res.v / ctx.avail) * 1000);
      ariaRow(res.v, ctx.avail);
      showReadout(splitText(splitPct(res.v, ctx.avail)), res.limit, pos);
    } else {
      if (write) writeWidth(which, res.v);
      ariaCol(which, res.v, res.max);
      showReadout(`${NAME[which]} ${Math.round(res.v)} px`, res.limit, pos);
    }
    clearLimits(h);
    h.classList.toggle("is-limit", res.limit);
  }
  // บันทึกค่าลง state (ค่าหลัง clamp ที่ผู้ใช้เห็น)
  function commit(which, res, ctx) {
    if (which === "stage") {
      state.stageRatio = clamp(res.v / ctx.avail, RATIO.min, RATIO.max);
      return;
    }
    state[FIELD[which]] = clamp(res.v, LIMITS[which].min, LIMITS[which].max);
    // ด้านตรงข้ามเป็นค่าที่ผู้ใช้ตั้ง และตอนนี้ถูกหดอัตโนมัติ (จอแคบ) → เก็บความกว้างที่เห็นจริงไว้ด้วย
    // ไม่งั้น render() จะคำนวณการหดตามสัดส่วนใหม่จากค่าเดิม → ด้านที่ไม่ได้ลากกระโดดตอนปล่อยเมาส์
    const other = which === "palette" ? "props" : "palette";
    const otherW = Math.round(ctx.otherW);
    if (!isHidden(other) && state[FIELD[other]] != null && otherW < state[FIELD[other]]) {
      state[FIELD[other]] = clamp(otherW, LIMITS[other].min, LIMITS[other].max);
    }
  }
  function startContext(which) {
    if (which === "stage") {
      return { avail: measureRows().avail, start: stage.getBoundingClientRect().height };
    }
    const other = which === "palette" ? "props" : "palette";
    return {
      inner: measureCols().inner,
      otherW: widthOf(other),
      start: panelEl[which].getBoundingClientRect().width,
    };
  }

  // ---------- reset ----------
  function reset(which = "all") {
    const list = which === "all" ? ["palette", "props", "stage"] : [which];
    const els = new Set();
    if (which === "all") {
      // reset-all ขณะโฟกัสอยู่บนปุ่ม rail (rail ถูกซ่อน → โฟกัสหลุดไป body) → จำไว้ แล้วย้ายไปปุ่มซ่อนของแผงนั้น
      const railFocus = ["palette", "props"].find((w) => rails[w]?.contains(document.activeElement));
      for (const w of ["palette", "props"]) if (isHidden(w)) {
        state[HIDDEN[w]] = false;
        els.add(ws);
        for (const fn of [...subscribers]) { try { fn(w, false); } catch (e) { console.error(e); } }
      }
      applyCollapsedDom();
      if (railFocus) panelEl[railFocus].querySelector("[data-collapse]")?.focus();
    }
    for (const w of list) {
      if (!(w in FIELD)) continue;
      state[FIELD[w]] = null;
      els.add(w === "stage" ? center : ws);
    }
    if (!els.size) return;
    clearTimeout(saveTimer);
    save(state);
    clearTimeout(animTimer);
    ws.classList.remove("lr-animating");
    center.classList.remove("lr-animating");
    for (const el of els) el.classList.add("lr-animating");
    animTimer = setTimeout(() => { ws.classList.remove("lr-animating"); center.classList.remove("lr-animating"); }, 200);
    render();
    if (which !== "all") {
      const h = handleOf(which);
      if (h) {
        h.classList.remove("is-limit");
        showReadout("คืนค่าเดิมแล้ว", false, nearHandle(h));
        hideReadout(900);
      }
    }
  }

  // เริ่มลาก/กดคีย์ → หยุด transition ของการ reset ทันที (ไม่ให้ track ไหลตามช้า ๆ ระหว่างลาก)
  function stopAnim() {
    clearTimeout(animTimer);
    ws.classList.remove("lr-animating");
    center.classList.remove("lr-animating");
  }

  // ---------- drag ----------
  function onKeyDuringDrag(e) {
    if (e.key !== "Escape" || !drag) return;
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();
    endDrag(false);
  }
  function endDrag(apply) {
    const d = drag;
    if (!d) return;
    if (d.raf) cancelAnimationFrame(d.raf);
    let final = null;
    if (apply && d.passed) final = computeValue(d.which, desiredFor(d), d.ctx);
    drag = null;
    window.removeEventListener("keydown", onKeyDuringDrag, true);
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerEnd);
    window.removeEventListener("pointercancel", onPointerEnd);
    window.removeEventListener("blur", onWindowBlur);
    d.h.removeEventListener("lostpointercapture", onPointerEnd);
    try { if (d.h.hasPointerCapture(d.id)) d.h.releasePointerCapture(d.id); } catch { /* ignore */ }
    document.body.classList.remove("lr-dragging", "lr-dragging-col", "lr-dragging-row");
    d.h.classList.remove("is-dragging");
    if (final) {
      commit(d.which, final, d.ctx);
      clearTimeout(saveTimer);
      save(state);
    } else {
      d.h.classList.remove("is-limit");
    }
    render(); // ยกเลิก/คลิกเฉย ๆ → คืนตาม state · ลากจริง → เขียนค่าที่ clamp แล้ว
    if (d.passed) hideReadout(apply ? 600 : 0);
  }
  function desiredFor(d) {
    const delta = d.which === "stage" ? d.y - d.y0 : d.x - d.x0;
    return d.which === "props" ? d.ctx.start - delta : d.ctx.start + delta;
  }
  function frame() {
    const d = drag;
    if (!d) return;
    d.raf = 0;
    const res = computeValue(d.which, desiredFor(d), d.ctx);
    applyLive(d.which, res, d.ctx, { x: d.x + 14, y: d.y + 14 }, d.h);
  }
  function onPointerMove(e) {
    const d = drag;
    if (!d || e.pointerId !== d.id) return;
    d.x = e.clientX;
    d.y = e.clientY;
    if (!d.passed) {
      if (Math.hypot(d.x - d.x0, d.y - d.y0) < DRAG_THRESHOLD) return;
      d.passed = true;
    }
    if (!d.raf) d.raf = requestAnimationFrame(frame);
  }
  function onPointerEnd(e) {
    if (!drag || e.pointerId !== drag.id) return;
    endDrag(true);
  }
  function onWindowBlur() { endDrag(true); } // สลับหน้าต่างกลางการลาก → pointerup อาจไม่มาถึง
  function onPointerDown(e) {
    const h = e.currentTarget;
    if (e.button !== 0 || !e.isPrimary) return;
    if (drag) endDrag(true); // การลากค้าง (พลาด pointerup) → ปิดก่อน ไม่ให้ตัวจับตายถาวร
    if (!measurable()) return;
    const which = h.dataset.lr;
    if (!(which in FIELD)) return;
    const ctx = startContext(which);
    if (which === "stage" && !(ctx.avail > 0)) return; // ไม่มีพื้นที่ให้แบ่ง → กัน NaNfr / ratio เพี้ยน
    e.preventDefault();
    stopAnim();
    // capture อาจไม่สำเร็จแบบเงียบ ๆ (เช่น buttons=0) → ฟัง move/up ที่ window แทน ใช้ได้ทั้งมีและไม่มี capture
    try { h.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    drag = {
      h, which, id: e.pointerId, ctx,
      x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY, passed: false, raf: 0,
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerEnd);
    window.addEventListener("pointercancel", onPointerEnd);
    window.addEventListener("blur", onWindowBlur);
    h.addEventListener("lostpointercapture", onPointerEnd);
    window.addEventListener("keydown", onKeyDuringDrag, true);
    h.classList.add("is-dragging");
    document.body.classList.add("lr-dragging", which === "stage" ? "lr-dragging-row" : "lr-dragging-col");
  }

  // ---------- keyboard ----------
  function onKeyDown(e) {
    const h = e.currentTarget;
    const which = h.dataset.lr;
    if (!(which in FIELD)) return;
    const isRow = which === "stage";
    let dir = 0;
    if (e.key === "Home") {
      e.preventDefault();
      e.stopPropagation();
      if (!drag) reset(which);
      return;
    }
    if (!e.key.startsWith("Arrow")) return;
    // กลืนลูกศรทั้ง 4 บนตัวจับเสมอ (ไม่ให้หลุดไปเลื่อนภาพก่อน/ถัดไปใน app.js) แต่ทำงานเฉพาะทิศของตัวจับ
    e.preventDefault();
    e.stopPropagation();
    if (isRow && (e.key === "ArrowUp" || e.key === "ArrowDown")) dir = e.key === "ArrowDown" ? 1 : -1;
    else if (!isRow && (e.key === "ArrowLeft" || e.key === "ArrowRight")) dir = e.key === "ArrowRight" ? 1 : -1;
    else return;
    if (drag || !measurable()) return;
    stopAnim();
    const step = isRow ? (e.shiftKey ? STEP.rowBig : STEP.row) : (e.shiftKey ? STEP.colBig : STEP.col);
    // ตัวจับเลื่อนตามทิศลูกศร: B (props) เลื่อนซ้าย = กว้างขึ้น
    const delta = which === "props" ? -dir * step : dir * step;
    const ctx = startContext(which);
    if (isRow && !(ctx.avail > 0)) return;
    const res = computeValue(which, ctx.start + delta, ctx);
    commit(which, res, ctx);
    render(); // เขียน var จาก state (แหล่งเดียว) แล้วแสดงผลอย่างเดียว
    applyLive(which, res, ctx, nearHandle(h), h, false);
    hideReadout(900);
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => save(state), 250);
  }

  // ---------- ตั้งค่าแบบไม่ลาก (preset) ----------
  function persistNow() { clearTimeout(saveTimer); save(state); }
  function setStageRatioImpl(r) {
    if (!(typeof r === "number" && Number.isFinite(r)) || drag) return false;
    state.stageRatio = clamp(r, RATIO.min, RATIO.max);
    persistNow();
    stopAnim();
    render();
    return true;
  }
  function setPanelWidthImpl(which, w) {
    if (!(which in panelEl) || !(typeof w === "number" && Number.isFinite(w)) || drag) return false;
    stopAnim();
    if (measurable()) {
      const ctx = startContext(which);
      commit(which, computeValue(which, w, ctx), ctx); // clamp ตามที่ว่างจริง + เก็บความกว้างด้านตรงข้ามถ้าถูกหด (เหมือนปล่อยเมาส์)
    } else {
      state[FIELD[which]] = clamp(w, LIMITS[which].min, LIMITS[which].max);
    }
    persistNow();
    render();
    return true;
  }
  // ---------- ซ่อน/แสดงแผง ----------
  function applyCollapsedDom() {
    for (const w of ["palette", "props"]) {
      const hid = isHidden(w);
      ws.classList.toggle(w === "palette" ? "pal-collapsed" : "props-collapsed", hid);
      if (rails[w]) rails[w].hidden = !hid;
      panelEl[w].inert = hid; // แผงที่ซ่อน (display:none) ห้ามรับโฟกัส/ถูกอ่านโดย screen reader
    }
  }
  function setPanelCollapsedImpl(which, hidden) {
    if (!(which in HIDDEN) || drag) return false;
    hidden = !!hidden;
    if (isHidden(which) === hidden) return true;
    state[HIDDEN[which]] = hidden;
    persistNow();
    stopAnim();
    ws.classList.add("lr-animating");
    animTimer = setTimeout(() => ws.classList.remove("lr-animating"), 200);
    applyCollapsedDom();
    render();
    for (const fn of [...subscribers]) { try { fn(which, hidden); } catch (e) { console.error(e); } }
    // โฟกัส: ซ่อน → ไปที่ปุ่มเปิดบน rail · เปิด → กลับไปที่ปุ่มซ่อนของแผง (ถ้ามี) ไม่ให้โฟกัสหลุด
    if (hidden) rails[which]?.querySelector("button")?.focus();
    else if (rails[which]?.contains(document.activeElement)) panelEl[which].querySelector("[data-collapse]")?.focus();
    return true;
  }
  live = {
    setPanelCollapsed: setPanelCollapsedImpl,
    togglePanel: (w) => setPanelCollapsedImpl(w, !isHidden(w)),
    isPanelCollapsed: isHidden,
    reset,
    setStageRatio: setStageRatioImpl,
    setPanelWidth: setPanelWidthImpl,
    snapshot: () => ({ stageRatio: state.stageRatio, paletteW: state.paletteW, propsW: state.propsW, paletteHidden: isHidden("palette"), propsHidden: isHidden("props") }),
  };

  // ปุ่มซ่อน/แสดง: ปุ่มบน rail (static ใน index.html) + ปุ่มในหัวแผง (palette-panel.js สร้างใหม่ทุกครั้งที่ render จึงฟังที่ document)
  document.addEventListener("click", (e) => {
    const b = e.target.closest?.("[data-collapse],[data-expand]");
    if (!b) return;
    const which = b.dataset.collapse || b.dataset.expand;
    if (which in HIDDEN) setPanelCollapsedImpl(which, !!b.dataset.collapse);
  });
  applyCollapsedDom();
  for (const h of handles) {
    h.addEventListener("pointerdown", onPointerDown);
    h.addEventListener("dblclick", (e) => { e.preventDefault(); reset(h.dataset.lr); });
    h.addEventListener("keydown", onKeyDown);
  }

  render();
  if (typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(scheduleRender);
    ro.observe(ws);
    ro.observe(center);
    ro.observe(stage); // เข้า/ออกเต็มจอ → clamp ใหม่หลังออก
    if (tray) ro.observe(tray);
  }
  return live;
}

// ---------- เมนู "มุมมอง ▾" (ปุ่มในหัวกระดาน) ----------
// เมนู ARIA แบบ portal ไป <body> (ไม่ถูก .graph-wrap ตัดขอบ): ↑/↓ เลื่อน · Home/End · Esc ปิดแล้วคืนโฟกัสให้ปุ่ม · Tab/คลิกนอกเมนูปิด
// opts.onChange() เรียกหลังเลือก preset (app.js ใช้วัด canvas ใหม่ → fit ถ้ามุมมองยังไม่ถูกแตะ)
export function initViewMenu(btn, opts = {}) {
  if (!btn) return { close() {} };
  let menu = null;

  function entries() {
    const snap = live ? live.snapshot() : null;
    const near = (a, b) => a != null && Math.abs(a - b) < 0.006;
    const list = [{ head: "สัดส่วนจอผลลัพธ์ : กระดาน" }];
    for (const p of STAGE_PRESETS) {
      list.push({ label: p.label, hint: p.hint, checked: !!snap && near(snap.stageRatio, p.ratio), run: () => setStageRatio(p.ratio) });
    }
    for (const which of ["palette", "props"]) {
      const wp = WIDTH_PRESETS[which];
      const cur = snap ? snap[FIELD[which]] : null;
      list.push({ head: "ความกว้าง" + wp.label });
      list.push({ label: "แคบ", hint: wp.narrow + " px", checked: cur != null && Math.abs(cur - wp.narrow) < 1, run: () => setPanelWidth(which, wp.narrow) });
      list.push({ label: "กว้าง", hint: wp.wide + " px", checked: cur != null && Math.abs(cur - wp.wide) < 1, run: () => setPanelWidth(which, wp.wide) });
    }
    list.push({ head: "แผง" });
    for (const which of ["palette", "props"]) {
      const hid = !!snap && snap[HIDDEN[which]];
      list.push({ label: (hid ? "แสดง" : "ซ่อน") + WIDTH_PRESETS[which].label, hint: "", plain: true, run: () => togglePanel(which) });
    }
    list.push({ head: "" });
    list.push({ label: "คืนค่าเดิมทั้งหมด", hint: "", plain: true, run: () => { if (live) live.reset("all"); } });
    return list;
  }

  function items() { return menu ? [...menu.querySelectorAll('[role^="menuitem"]')] : []; }
  function focusAt(i) {
    const els = items();
    if (!els.length) return;
    els[(i + els.length) % els.length].focus();
  }

  function close(returnFocus = false) {
    if (!menu) return;
    menu.remove();
    menu = null;
    btn.setAttribute("aria-expanded", "false");
    document.removeEventListener("pointerdown", onOutside, true);
    window.removeEventListener("blur", onWinBlur);
    window.removeEventListener("resize", onWinBlur);
    if (returnFocus) btn.focus();
  }
  function onOutside(e) { if (menu && !menu.contains(e.target) && !btn.contains(e.target)) close(false); }
  function onWinBlur() { close(false); }

  function open(focusLast = false) {
    if (menu) return;
    menu = document.createElement("div");
    menu.className = "view-menu";
    menu.setAttribute("role", "menu");
    menu.setAttribute("aria-label", "มุมมอง");
    for (const e of entries()) {
      if ("head" in e) {
        if (e.head) {
          const h = document.createElement("div");
          h.className = "vm-head";
          h.setAttribute("role", "presentation");
          h.textContent = e.head;
          menu.appendChild(h);
        } else {
          const sep = document.createElement("div");
          sep.className = "vm-sep";
          sep.setAttribute("role", "separator");
          menu.appendChild(sep);
        }
        continue;
      }
      const b = document.createElement("button");
      b.type = "button";
      b.className = "vm-item";
      b.tabIndex = -1;
      if (e.plain) b.setAttribute("role", "menuitem");
      else { b.setAttribute("role", "menuitemradio"); b.setAttribute("aria-checked", e.checked ? "true" : "false"); }
      const mark = document.createElement("span");
      mark.className = "vm-mark";
      mark.setAttribute("aria-hidden", "true");
      mark.textContent = e.checked ? "✓" : "";
      const lab = document.createElement("span");
      lab.className = "vm-label";
      lab.textContent = e.label;
      const hint = document.createElement("span");
      hint.className = "vm-hint mono";
      hint.textContent = e.hint;
      b.append(mark, lab, hint);
      b.addEventListener("click", () => {
        e.run();
        close(true);
        if (opts.onChange) opts.onChange();
      });
      menu.appendChild(b);
    }
    menu.addEventListener("keydown", onMenuKey);
    document.body.appendChild(menu);
    // วางใต้ปุ่ม ชิดขวา (ล้นล่าง → ขึ้นไปเหนือปุ่ม) ภายในกรอบจอ
    const r = btn.getBoundingClientRect();
    const mw = menu.offsetWidth, mh = menu.offsetHeight;
    const left = Math.max(8, Math.min(window.innerWidth - mw - 8, r.right - mw));
    let top = r.bottom + 4;
    if (top + mh > window.innerHeight - 8) top = Math.max(8, r.top - mh - 4);
    menu.style.left = left + "px";
    menu.style.top = top + "px";
    btn.setAttribute("aria-expanded", "true");
    document.addEventListener("pointerdown", onOutside, true);
    window.addEventListener("blur", onWinBlur);
    window.addEventListener("resize", onWinBlur);
    const els = items();
    const checked = els.findIndex((x) => x.getAttribute("aria-checked") === "true");
    focusAt(focusLast ? els.length - 1 : Math.max(0, checked));
  }

  function onMenuKey(e) {
    const els = items();
    const i = els.indexOf(document.activeElement);
    // ทุกปุ่มที่กดขณะเมนูเปิดอยู่ในเมนู → ไม่ให้หลุดไปคีย์ลัดของแอป (c/e/m, ←/→ เลื่อนภาพ, Esc ออกเต็มจอ ฯลฯ)
    e.stopPropagation();
    if (e.key === "Escape") { e.preventDefault(); close(true); }
    else if (e.key === "ArrowDown") { e.preventDefault(); focusAt(i + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); focusAt(i - 1); }
    else if (e.key === "Home") { e.preventDefault(); focusAt(0); }
    else if (e.key === "End") { e.preventDefault(); focusAt(els.length - 1); }
    else if (e.key === "Tab") { e.preventDefault(); close(true); } // ปิดแล้วคืนโฟกัสให้ปุ่ม (ไม่ปล่อยให้โฟกัสหลุดไปที่ BODY)
  }

  btn.addEventListener("click", () => { if (menu) close(true); else open(false); });
  btn.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); open(false); }
    else if (e.key === "ArrowUp") { e.preventDefault(); open(true); }
  });
  return { close };
}

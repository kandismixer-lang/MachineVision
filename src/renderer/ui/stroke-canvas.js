import {
  STROKE_MAX_POINTS, STROKE_MAX_STROKES, STROKE_MAX_TOTAL, STROKE_Q, STROKE_RR_DEFAULT,
  normalizeStrokes, pixelToStrokeQ, rasterizeStrokes, strokeAspectChanged,
  strokePixel, strokeRadiusPx,
} from "../filters/stroke-mask.js";
import { createStrokeModel } from "./stroke-model.js";

function allowedFor(param) {
  const labels = Array.isArray(param?.labels) ? param.labels : [];
  const allowed = new Set(labels.map((label) => label?.value).filter((v) => Number.isInteger(v) && v >= 1 && v <= 7));
  if (param?.eraser === true) allowed.add(0);
  return allowed;
}

function rgb(color, alpha = 1) {
  if (!Array.isArray(color) || color.length < 3) return `rgba(255,255,255,${alpha})`;
  return `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
}

export function normalizeStrokeRadiusPreference(raw) {
  if (raw === null || raw === "") return STROKE_RR_DEFAULT;
  const value = Number(raw);
  if (!Number.isInteger(value)) return STROKE_RR_DEFAULT;
  return Math.min(100, Math.max(1, value));
}

function readRadius() {
  try {
    const saved = localStorage.getItem("mv-stroke-rr");
    return normalizeStrokeRadiusPreference(saved);
  } catch { /* ไม่ให้การตั้งค่าเบราว์เซอร์ขัดการวาด */ }
  return STROKE_RR_DEFAULT;
}

export function createStrokeCanvas({ canvas, toolbar, onEnter, onExit, setStatus, commit, getValue, onRequestInput }) {
  const ctx = canvas.getContext("2d");
  const stageBody = canvas.parentNode || null;
  const inputCanvas = document.createElement("canvas");
  const resultCanvas = document.createElement("canvas");
  const rasterCanvas = document.createElement("canvas");
  const inputCtx = inputCanvas.getContext("2d", { willReadFrequently: true });
  const resultCtx = resultCanvas.getContext("2d", { willReadFrequently: true });
  const rasterCtx = rasterCanvas.getContext("2d");
  const badge = document.createElement("div");
  badge.className = "annotate-class-badge hidden";
  const badgeDot = document.createElement("span");
  badgeDot.className = "annotate-badge-dot";
  const badgeName = document.createElement("span");
  badgeName.className = "annotate-badge-name";
  badge.append(badgeDot, badgeName);
  stageBody?.appendChild(badge);

  let active = false;
  let target = null;
  let model = null;
  let lastValue = null;
  let lastCommittedValue = null;
  let inputWidth = 0, inputHeight = 0;
  let resultWidth = 0, resultHeight = 0;
  let currentLabel = 1;
  let rr = readRadius();
  let rrInput = null, rrText = null, editHint = null, warningChip = null;
  let labelButtons = [];
  let eraserButton = null, undoButton = null, redoButton = null;
  let previewButton = null;
  let mapDirty = true;
  let showResult = false;
  let showIntro = false;
  let hoverPoint = null;
  let pointerId = null;
  let lastPoint = null;
  let rafPending = false;
  let commitTimer = null;
  let pendingCommit = undefined;
  let keyHandler = null;
  let resizeHandler = null;

  function labelInfo(label) {
    return (target?.param?.labels || []).find((item) => item.value === label) || null;
  }

  function activeColor() {
    return currentLabel === 0 ? null : labelInfo(currentLabel)?.color || [255, 255, 255];
  }

  function updateBadge() {
    const info = labelInfo(currentLabel);
    badgeDot.style.backgroundColor = info ? `rgb(${info.color.join(",")})` : "#666";
    badgeName.textContent = currentLabel === 0 ? "ยางลบ" : (info?.name || "เลือกป้าย");
  }

  function scheduleDraw() {
    if (!active || rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => { rafPending = false; draw(); });
  }

  function markMapDirty() {
    mapDirty = true;
    scheduleDraw();
  }

  function refreshRaster() {
    mapDirty = false;
    if (!inputWidth || !inputHeight || !lastValue) {
      rasterCanvas.width = 0;
      rasterCanvas.height = 0;
      return null;
    }
    const result = rasterizeStrokes(inputWidth, inputHeight, lastValue, allowedFor(target.param));
    if (!result) {
      rasterCanvas.width = 0;
      rasterCanvas.height = 0;
      return null;
    }
    if (rasterCanvas.width !== inputWidth) rasterCanvas.width = inputWidth;
    if (rasterCanvas.height !== inputHeight) rasterCanvas.height = inputHeight;
    const pixels = rasterCtx.createImageData(inputWidth, inputHeight);
    for (let i = 0; i < result.labels.length; i++) {
      const label = result.labels[i];
      if (!label) continue;
      const color = labelInfo(label)?.color || [255, 255, 255];
      const at = i * 4;
      pixels.data[at] = color[0];
      pixels.data[at + 1] = color[1];
      pixels.data[at + 2] = color[2];
      pixels.data[at + 3] = 255;
    }
    rasterCtx.putImageData(pixels, 0, 0);
    return result;
  }

  function currentScale() {
    const rect = canvas.getBoundingClientRect();
    return rect.width ? canvas.width / rect.width : 1;
  }

  function drawDraft() {
    const draft = model?.draft();
    if (!draft || !inputWidth || !inputHeight) return;
    ctx.save();
    ctx.strokeStyle = currentLabel === 0 ? "rgba(255,255,255,0.9)" : rgb(activeColor(), 0.55);
    ctx.lineWidth = (2 * strokeRadiusPx(draft[1], inputWidth, inputHeight) + 1);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (currentLabel === 0) ctx.setLineDash([4 * currentScale(), 3 * currentScale()]);
    ctx.beginPath();
    for (let i = 2; i + 1 < draft.length; i += 2) {
      const x = strokePixel(draft[i], inputWidth) + 0.5;
      const y = strokePixel(draft[i + 1], inputHeight) + 0.5;
      if (i === 2) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    if (draft.length === 4) ctx.lineTo(strokePixel(draft[2], inputWidth) + 0.5, strokePixel(draft[3], inputHeight) + 0.5);
    ctx.stroke();
    ctx.restore();
  }

  function drawCursor() {
    if (!hoverPoint || !inputWidth || !inputHeight || model?.draft()) return;
    const radius = strokeRadiusPx(rr, inputWidth, inputHeight);
    const x = strokePixel(hoverPoint.qx, inputWidth) + 0.5;
    const y = strokePixel(hoverPoint.qy, inputHeight) + 0.5;
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius + 0.5, 0, Math.PI * 2);
    ctx.strokeStyle = currentLabel === 0 ? "rgba(255,255,255,0.9)" : rgb(activeColor(), 0.9);
    ctx.lineWidth = Math.max(1, currentScale());
    if (currentLabel === 0) ctx.setLineDash([3 * currentScale(), 2 * currentScale()]);
    ctx.stroke();
    ctx.restore();
  }

  function updateToolbar() {
    if (!target) return;
    labelButtons.forEach(({ button, label }) => {
      button.classList.toggle("active", currentLabel === label.value);
      button.setAttribute("aria-pressed", currentLabel === label.value ? "true" : "false");
      button.title = label.hint || label.name || "";
    });
    if (eraserButton) eraserButton.classList.toggle("active", currentLabel === 0);
    if (undoButton) undoButton.disabled = !model?.canUndo;
    if (redoButton) redoButton.disabled = !model?.canRedo;
    if (rrText) rrText.textContent = `รัศมี ${strokeRadiusPx(rr, inputWidth, inputHeight)} px`;
    if (warningChip) {
      const changed = inputWidth && inputHeight && strokeAspectChanged(lastValue, inputWidth, inputHeight);
      warningChip.classList.toggle("hidden", !changed);
    }
    if (editHint) editHint.textContent = currentLabel === 0
      ? "ลากทับเส้นที่วาดผิดเพื่อลบออก"
      : (labelInfo(currentLabel)?.hint || "ลากเมาส์เพื่อขีดเส้นบนภาพ");
    updateBadge();
    if (previewButton) previewButton.textContent = showResult ? "🖼 ดูภาพเข้า" : "🎯 ดูผล";
  }

  function draw() {
    if (!active) return;
    if (mapDirty) refreshRaster();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const background = showResult ? resultCanvas : inputCanvas;
    const hasBackground = showResult ? resultWidth > 0 && resultHeight > 0 : inputWidth > 0 && inputHeight > 0;
    if (hasBackground && background.width && background.height) {
      ctx.drawImage(background, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = "#101820";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = `${14 * currentScale()}px sans-serif`;
      ctx.fillText(showResult ? "กำลังคำนวณผล…" : "กำลังโหลดภาพที่เข้ากล่องนี้…", canvas.width / 2, canvas.height / 2);
    }
    if (!showResult && rasterCanvas.width && rasterCanvas.height) {
      ctx.save();
      ctx.globalAlpha = 0.55;
      ctx.drawImage(rasterCanvas, 0, 0);
      ctx.restore();
    } else if (showResult && rasterCanvas.width && rasterCanvas.height) {
      ctx.save();
      ctx.globalAlpha = 0.35;
      ctx.drawImage(rasterCanvas, 0, 0, canvas.width, canvas.height);
      ctx.restore();
    }
    drawDraft();
    drawCursor();
    if (showIntro && inputWidth && inputHeight) {
      const labels = target.param.labels;
      const first = labels[0]?.name || `ป้าย ${labels[0]?.value ?? 1}`;
      const second = labels[1]?.name || `ป้าย ${labels[1]?.value ?? 2}`;
      ctx.save();
      ctx.fillStyle = "rgba(0,0,0,0.72)";
      ctx.fillRect(canvas.width * 0.1, canvas.height * 0.42, canvas.width * 0.8, canvas.height * 0.16);
      ctx.fillStyle = "#fff";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = `${Math.max(12, 14 * currentScale())}px sans-serif`;
      ctx.fillText(`เริ่มจากเลือก “${first}” แล้วลากเส้นทับวัตถุ`, canvas.width / 2, canvas.height * 0.47);
      ctx.fillText(`จากนั้นเลือก “${second}” แล้วลากเส้นบนพื้นหลัง`, canvas.width / 2, canvas.height * 0.53);
      ctx.restore();
    }
    updateToolbar();
  }

  function eventPoint(event) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height || !inputWidth || !inputHeight) return null;
    const nx = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const ny = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
    const px = Math.min(inputWidth - 1, Math.floor(nx * inputWidth));
    const py = Math.min(inputHeight - 1, Math.floor(ny * inputHeight));
    return { qx: pixelToStrokeQ(px, inputWidth), qy: pixelToStrokeQ(py, inputHeight) };
  }

  function capture(event) {
    try { if (event.pointerId != null) canvas.setPointerCapture(event.pointerId); } catch { /* การลากยังใช้ได้ถ้าจับ pointer ไม่สำเร็จ */ }
  }

  function releasePointer() {
    try { if (pointerId != null) canvas.releasePointerCapture(pointerId); } catch { /* จับ pointer ไว้แล้วหรือไม่ก็ได้ */ }
    pointerId = null;
  }

  function coalesced(event) {
    const events = event.getCoalescedEvents?.();
    return events?.length ? events : [event];
  }

  function setLocalValue(next) {
    lastValue = normalizeStrokes(next, allowedFor(target.param));
    mapDirty = true;
    scheduleDraw();
  }

  function scheduleCommit(next) {
    pendingCommit = next;
    if (commitTimer) clearTimeout(commitTimer);
    commitTimer = setTimeout(() => flushCommit(), 120);
  }

  function flushCommit() {
    if (commitTimer) { clearTimeout(commitTimer); commitTimer = null; }
    if (pendingCommit === undefined) return;
    const next = pendingCommit;
    pendingCommit = undefined;
    lastCommittedValue = next;
    commit?.(target?.paramKey, next);
  }

  function finishDraft(point) {
    if (!model?.draft()) return;
    if (point) model.extend(point.qx, point.qy, inputWidth, inputHeight, true);
    const next = model.end();
    setLocalValue(next);
    scheduleCommit(next);
  }

  function onPointerDown(event) {
    if (pointerId !== null) return;
    if (event.pointerType === "touch") return;
    const penErase = event.pointerType === "pen" && (((event.buttons || 0) & 32) !== 0 || event.button === 5);
    if (event.button !== 0 && !penErase) return;
    if (penErase && target.param.eraser !== true) {
      setStatus?.("เครื่องมือนี้ยังไม่มียางลบ");
      return;
    }
    if (!inputWidth || !inputHeight) return;
    const point = eventPoint(event);
    if (!point) return;
    event.preventDefault();
    canvas.focus();
    showIntro = false;
    hoverPoint = point;
    const label = penErase && target.param.eraser ? 0 : currentLabel;
    const started = model.begin(label, rr, point.qx, point.qy, inputWidth, inputHeight);
    if (!started.ok) {
      setStatus?.(started.reason === "จำนวนเส้นครบแล้ว" || started.reason === "จำนวนจุดรวมครบแล้ว"
        ? `วาดได้สูงสุด ${STROKE_MAX_STROKES} เส้น / ${STROKE_MAX_TOTAL.toLocaleString("th-TH")} จุด — กด ↶ ย้อน หรือ 🗑 ล้าง ก่อน`
        : started.reason);
      return;
    }
    pointerId = event.pointerId;
    lastPoint = point;
    capture(event);
    scheduleDraw();
  }

  function onPointerMove(event) {
    const point = eventPoint(event);
    if (!point) return;
    hoverPoint = point;
    if (model?.draft()) {
      for (const item of coalesced(event)) {
        const next = eventPoint(item);
        if (!next) continue;
        lastPoint = next;
        const result = model.extend(next.qx, next.qy, inputWidth, inputHeight);
        if (result === "full") {
          const longLine = model.draftPointCount >= STROKE_MAX_POINTS;
          setStatus?.(longLine
            ? "เส้นนี้ยาวมากแล้ว — ปล่อยเมาส์แล้วเริ่มเส้นใหม่"
            : `วาดได้สูงสุด ${STROKE_MAX_STROKES} เส้น / ${STROKE_MAX_TOTAL.toLocaleString("th-TH")} จุด — กด ↶ ย้อน หรือ 🗑 ล้าง ก่อน`);
          break;
        }
      }
    }
    scheduleDraw();
  }

  function onPointerUp(event) {
    if (!model?.draft()) return;
    const point = eventPoint(event) || lastPoint;
    if (point) {
      for (const item of coalesced(event)) {
        const next = eventPoint(item);
        if (next) model.extend(next.qx, next.qy, inputWidth, inputHeight);
      }
      finishDraft(point);
    } else finishDraft(lastPoint);
    releasePointer();
    lastPoint = null;
  }

  function chooseLabel(label) {
    showIntro = false;
    currentLabel = label;
    updateToolbar();
    scheduleDraw();
  }

  function setRadius(next) {
    showIntro = false;
    rr = Math.min(100, Math.max(1, Math.round(next)));
    if (rrInput) rrInput.value = String(rr);
    try { localStorage.setItem("mv-stroke-rr", String(rr)); } catch { /* จำขนาดพู่กันได้เมื่อเบราว์เซอร์อนุญาต */ }
    updateToolbar();
    scheduleDraw();
  }

  function refreshHistoryButtons() {
    if (undoButton) undoButton.disabled = !model?.canUndo;
    if (redoButton) redoButton.disabled = !model?.canRedo;
  }

  function changeHistory(action) {
    if (!model) return;
    showIntro = false;
    const next = model[action]();
    setLocalValue(next);
    scheduleCommit(next);
    refreshHistoryButtons();
  }

  function clearStrokes() {
    showIntro = false;
    const hadStrokes = !!model?.value;
    const next = model?.clear() ?? null;
    if (!hadStrokes) return;
    setLocalValue(next);
    scheduleCommit(next);
    setStatus?.("ล้างเส้นแล้ว — กด ↶ ย้อน เพื่อเอาคืน");
  }

  function makeButton(className, text, title, action) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = text;
    button.title = title || "";
    button.addEventListener("click", action);
    toolbar.appendChild(button);
    return button;
  }

  function addSep() {
    const sep = document.createElement("span");
    sep.className = "annotate-sep";
    toolbar.appendChild(sep);
  }

  function buildToolbar() {
    toolbar.innerHTML = "";
    labelButtons = [];
    eraserButton = undoButton = redoButton = previewButton = null;
    const title = document.createElement("span");
    title.className = "annotate-chip stroke-title";
    title.textContent = `✏️ วาดเส้น · ${target.opName || "ภาพ"}`;
    toolbar.appendChild(title);

    const tools = document.createElement("div");
    tools.className = "annotate-tools";
    toolbar.appendChild(tools);
    const labels = Array.isArray(target.param.labels) ? target.param.labels : [];
    labels.forEach((label, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "annotate-btn annotate-tool";
      const dot = document.createElement("span");
      dot.className = "stroke-label-dot";
      if (Array.isArray(label.color)) dot.style.backgroundColor = `rgb(${label.color.join(",")})`;
      const name = document.createElement("span");
      name.textContent = `${label.name || `ป้าย ${label.value}`} ${index + 1}`;
      button.append(dot, name);
      button.title = label.hint || label.name || "";
      button.setAttribute("aria-label", `${label.name || `ป้าย ${label.value}`}, ปุ่ม ${index + 1}`);
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => chooseLabel(label.value));
      tools.appendChild(button);
      labelButtons.push({ button, label });
    });
    if (target.param.eraser === true) {
      eraserButton = document.createElement("button");
      eraserButton.type = "button";
      eraserButton.className = "annotate-btn annotate-tool";
      eraserButton.textContent = `◌ ยางลบ ${labels.length + 1}`;
      eraserButton.title = "ลากทับเส้นที่วาดผิดเพื่อลบออก";
      eraserButton.addEventListener("click", () => chooseLabel(0));
      tools.appendChild(eraserButton);
    }

    addSep();
    const size = document.createElement("label");
    size.className = "stroke-size";
    const sizeName = document.createElement("span");
    sizeName.textContent = "ขนาด";
    rrInput = document.createElement("input");
    rrInput.type = "range";
    rrInput.min = "1";
    rrInput.max = "100";
    rrInput.step = "1";
    rrInput.value = String(rr);
    rrInput.setAttribute("aria-label", "ขนาดพู่กัน");
    rrText = document.createElement("span");
    rrText.className = "stroke-radius mono";
    rrInput.addEventListener("input", () => setRadius(Number(rrInput.value)));
    size.append(sizeName, rrInput, rrText);
    toolbar.appendChild(size);

    addSep();
    undoButton = makeButton("annotate-btn", "↶ ย้อน", "ย้อนเส้นล่าสุด (Ctrl+Z)", () => changeHistory("undo"));
    redoButton = makeButton("annotate-btn", "↷ ทำซ้ำ", "ทำเส้นที่ย้อนกลับมาอีกครั้ง (Ctrl+Y)", () => changeHistory("redo"));
    makeButton("annotate-btn", "🗑 ล้าง", "ล้างเส้นทั้งหมด", clearStrokes);
    editHint = document.createElement("span");
    editHint.className = "annotate-hint stroke-hint";
    toolbar.appendChild(editHint);
    warningChip = document.createElement("span");
    warningChip.className = "stroke-warn hidden";
    warningChip.textContent = "⚠ ภาพเข้าคนละสัดส่วนกับตอนวาด — เส้นถูกยืดตาม ลองดูว่ายังทับวัตถุถูกที่ไหม";
    toolbar.appendChild(warningChip);

    const right = document.createElement("span");
    right.className = "annotate-right";
    toolbar.appendChild(right);
    previewButton = document.createElement("button");
    previewButton.type = "button";
    previewButton.className = "annotate-btn";
    previewButton.addEventListener("click", () => { showIntro = false; showResult = !showResult; scheduleDraw(); });
    right.appendChild(previewButton);
    const exit = document.createElement("button");
    exit.type = "button";
    exit.className = "annotate-btn annotate-exit";
    exit.textContent = "✓ เสร็จ";
    exit.addEventListener("click", () => exitMode());
    right.appendChild(exit);
    updateToolbar();
  }

  function onKeyDown(event) {
    if (!active || document.querySelector(".modal-overlay")) return;
    const input = event.target;
    if ((input?.tagName === "INPUT" && input !== rrInput) || input?.tagName === "SELECT" ||
        input?.tagName === "TEXTAREA" || input?.isContentEditable) return;
    const code = event.code || "";
    if (event.ctrlKey || event.metaKey) {
      if (code === "KeyZ") {
        event.preventDefault(); event.stopPropagation();
        changeHistory(event.shiftKey ? "redo" : "undo");
      } else if (code === "KeyY") {
        event.preventDefault(); event.stopPropagation();
        changeHistory("redo");
      }
      return;
    }
    if (event.altKey) return;
    const digit = code.match(/^(?:Digit|Numpad)([1-9])$/);
    if (digit) {
      const n = Number(digit[1]);
      const labels = target.param.labels || [];
      if (n <= labels.length) chooseLabel(labels[n - 1].value);
      else if (n === labels.length + 1 && target.param.eraser === true) chooseLabel(0);
      event.preventDefault(); event.stopPropagation();
      return;
    }
    if (code === "BracketLeft" || code === "BracketRight") {
      setRadius(rr + (code === "BracketLeft" ? -1 : 1) * (event.shiftKey ? 5 : 1));
      event.preventDefault(); event.stopPropagation();
      return;
    }
    if (code === "Tab") return;
    if ((code === "Enter" || code === "Space") && (input?.tagName === "BUTTON" || input?.closest?.("button"))) return;
    if (code === "Enter") {
      event.preventDefault(); event.stopPropagation();
      exitMode();
      return;
    }
    if (code === "Escape") {
      event.preventDefault(); event.stopPropagation();
      if (model?.draft()) {
        model.cancel();
        releasePointer();
        lastPoint = null;
        scheduleDraw();
      } else exitMode();
      return;
    }
    if (input === rrInput && (code === "ArrowLeft" || code === "ArrowRight" || code === "ArrowUp" || code === "ArrowDown")) {
      event.stopPropagation();
      return;
    }
    if (code === "ArrowLeft" || code === "ArrowRight") return;
    event.preventDefault();
    event.stopPropagation();
  }

  function detachEvents() {
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("pointercancel", onPointerUp);
    canvas.removeEventListener("pointerleave", onPointerLeave);
    canvas.removeEventListener("contextmenu", onContextMenu);
    document.removeEventListener("keydown", keyHandler, true);
    window.removeEventListener("resize", resizeHandler);
  }

  function onPointerLeave() {
    if (!model?.draft()) { hoverPoint = null; scheduleDraw(); }
  }

  function onContextMenu(event) { event.preventDefault(); }

  function exitMode({ cancelDraft = false } = {}) {
    if (!active) return;
    if (model?.draft()) {
      if (cancelDraft) model.cancel();
      else finishDraft(lastPoint);
    }
    active = false;
    flushCommit();
    detachEvents();
    releasePointer();
    inputCanvas.width = inputCanvas.height = 0;
    resultCanvas.width = resultCanvas.height = 0;
    rasterCanvas.width = rasterCanvas.height = 0;
    canvas.classList.add("hidden");
    toolbar.classList.add("hidden");
    stageBody?.classList.remove("ann-mode-dock");
    badge.classList.add("hidden");
    hoverPoint = null;
    const old = target;
    target = null;
    model = null;
    onExit?.(old);
  }

  function enter(next) {
    if (active) {
      const sameTarget = target?.nodeId === next?.nodeId && target?.paramKey === next?.paramKey;
      exitMode();
      if (sameTarget) return true;
    }
    if (!next || !next.param || !Array.isArray(next.param.labels) || next.param.labels.length === 0) return false;
    target = next;
    inputWidth = inputHeight = resultWidth = resultHeight = 0;
    inputCanvas.width = inputCanvas.height = 0;
    resultCanvas.width = resultCanvas.height = 0;
    rasterCanvas.width = rasterCanvas.height = 0;
    const allowed = allowedFor(next.param);
    model = createStrokeModel({ allowed, initial: getValue?.(next.paramKey, next.nodeId) });
    lastValue = model.value;
    lastCommittedValue = lastValue;
    currentLabel = next.param.labels[0].value;
    showIntro = !lastValue;
    showResult = false;
    active = true;
    mapDirty = true;
    canvas.classList.remove("hidden");
    toolbar.classList.remove("hidden");
    stageBody?.classList.add("ann-mode-dock");
    buildToolbar();
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    canvas.addEventListener("pointerleave", onPointerLeave);
    canvas.addEventListener("contextmenu", onContextMenu);
    keyHandler = onKeyDown;
    resizeHandler = scheduleDraw;
    document.addEventListener("keydown", keyHandler, true);
    window.addEventListener("resize", resizeHandler);
    canvas.focus();
    badge.classList.remove("hidden");
    updateBadge();
    onEnter?.({ nodeId: next.nodeId, paramKey: next.paramKey, param: next.param, opName: next.opName });
    onRequestInput?.();
    scheduleDraw();
    return true;
  }

  function onInputImage(image) {
    if (!active) return;
    if (!image || !(image.width > 0) || !(image.height > 0)) return;
    inputWidth = image.width;
    inputHeight = image.height;
    inputCanvas.width = inputWidth;
    inputCanvas.height = inputHeight;
    inputCtx.putImageData(image, 0, 0);
    canvas.width = inputWidth;
    canvas.height = inputHeight;
    mapDirty = true;
    scheduleDraw();
  }

  function onResultImage(image) {
    if (!active) return;
    if (!image || !(image.width > 0) || !(image.height > 0)) return;
    resultWidth = image.width;
    resultHeight = image.height;
    resultCanvas.width = resultWidth;
    resultCanvas.height = resultHeight;
    resultCtx.putImageData(image, 0, 0);
    if (showResult) scheduleDraw();
  }

  function onExternalValue(value) {
    if (!active || !model) return;
    if (pendingCommit !== undefined) return;
    const normalized = normalizeStrokes(value, allowedFor(target.param));
    if (JSON.stringify(normalized) === JSON.stringify(lastCommittedValue)) return;
    if (commitTimer) { clearTimeout(commitTimer); commitTimer = null; }
    pendingCommit = undefined;
    lastValue = model.reset(normalized);
    lastCommittedValue = lastValue;
    showIntro = !lastValue;
    markMapDirty();
  }

  return {
    enter,
    exit: exitMode,
    isActive: () => active,
    activeNodeId: () => target?.nodeId ?? null,
    onInputImage,
    onResultImage,
    onExternalValue,
  };
}

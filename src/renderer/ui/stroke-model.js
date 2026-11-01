import {
  STROKE_MAX_POINTS, STROKE_MAX_STROKES, STROKE_MAX_TOTAL, STROKE_Q,
  normalizeStrokes, strokePixel, strokeRadiusPx,
} from "../filters/stroke-mask.js";

function pointValue(q) {
  if (typeof q !== "number" || !Number.isFinite(q)) return 0;
  return Math.min(STROKE_Q, Math.max(0, Math.floor(q + 0.5)));
}

function imageSize(n) {
  if (!Number.isFinite(n)) return 0;
  return Math.min(100000, Math.max(0, Math.floor(n + 0.5)));
}

function copyValue(value) {
  if (!value) return null;
  return { v: value.v, w: value.w, h: value.h, s: value.s.map((line) => line.slice()) };
}

function sameValue(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function pointCount(value) {
  return value ? value.s.reduce((n, line) => n + (line.length - 2) / 2, 0) : 0;
}

export function createStrokeModel({ allowed = new Set([0, 1, 2, 3, 4, 5, 6, 7]), initial = null } = {}) {
  const labels = allowed instanceof Set ? allowed : new Set([0, 1, 2, 3, 4, 5, 6, 7]);
  let value = normalizeStrokes(initial, labels);
  let draftLine = null;
  let draftW = 0, draftH = 0;
  const undoStack = [], redoStack = [];

  function remember() {
    undoStack.push(copyValue(value));
    if (undoStack.length > 100) undoStack.shift();
    redoStack.length = 0;
  }

  function begin(label, rr, qx, qy, W, H) {
    if (draftLine) return { ok: false, reason: "กำลังวาดเส้นอยู่" };
    if (!labels.has(label) || !Number.isInteger(label) || label < 0 || label > 7) return { ok: false, reason: "ป้ายนี้ใช้ไม่ได้" };
    const count = value?.s.length || 0;
    const points = pointCount(value);
    if (count >= STROKE_MAX_STROKES) return { ok: false, reason: "จำนวนเส้นครบแล้ว" };
    if (points >= STROKE_MAX_TOTAL) return { ok: false, reason: "จำนวนจุดรวมครบแล้ว" };
    const w = imageSize(W), h = imageSize(H);
    const radius = Math.min(100, Math.max(1, Math.floor((Number.isFinite(rr) ? rr : 1) + 0.5)));
    draftLine = [label, radius, pointValue(qx), pointValue(qy)];
    draftW = w;
    draftH = h;
    return { ok: true };
  }

  function extend(qx, qy, W, H, force = false) {
    if (!draftLine) return "skipped";
    const x = pointValue(qx), y = pointValue(qy);
    const w = imageSize(W), h = imageSize(H);
    const lastX = strokePixel(draftLine[draftLine.length - 2], w);
    const lastY = strokePixel(draftLine[draftLine.length - 1], h);
    const nextX = strokePixel(x, w), nextY = strokePixel(y, h);
    const minDistance = Math.max(1, Math.floor(strokeRadiusPx(draftLine[1], w, h) / 2));
    if (nextX === lastX && nextY === lastY) return "skipped";
    if (!force && Math.max(Math.abs(nextX - lastX), Math.abs(nextY - lastY)) < minDistance) return "skipped";
    if ((draftLine.length - 2) / 2 >= STROKE_MAX_POINTS || pointCount(value) + (draftLine.length - 2) / 2 >= STROKE_MAX_TOTAL) return "full";
    draftLine.push(x, y);
    draftW = w;
    draftH = h;
    return "appended";
  }

  function end() {
    if (!draftLine) return copyValue(value);
    const s = value ? value.s.map((line) => line.slice()) : [];
    s.push(draftLine.slice());
    remember();
    value = { v: 1, w: draftW, h: draftH, s };
    draftLine = null;
    return copyValue(value);
  }

  function cancel() {
    draftLine = null;
    return copyValue(value);
  }

  function undo() {
    draftLine = null;
    if (!undoStack.length) return copyValue(value);
    redoStack.push(copyValue(value));
    value = undoStack.pop();
    return copyValue(value);
  }

  function redo() {
    draftLine = null;
    if (!redoStack.length) return copyValue(value);
    undoStack.push(copyValue(value));
    if (undoStack.length > 100) undoStack.shift();
    value = redoStack.pop();
    return copyValue(value);
  }

  function clear() {
    draftLine = null;
    if (!value) return null;
    remember();
    value = null;
    return null;
  }

  function reset(next) {
    const norm = normalizeStrokes(next, labels);
    if (!sameValue(value, norm)) {
      value = norm;
      draftLine = null;
      redoStack.length = 0;
    }
    return copyValue(value);
  }

  return {
    begin,
    extend,
    end,
    cancel,
    undo,
    redo,
    clear,
    reset,
    draft: () => draftLine ? draftLine.slice() : null,
    get value() { return copyValue(value); },
    get pointCount() { return pointCount(value); },
    get draftPointCount() { return draftLine ? (draftLine.length - 2) / 2 : 0; },
    get totalPointCount() { return pointCount(value) + (draftLine ? (draftLine.length - 2) / 2 : 0); },
    get strokeCount() { return value?.s.length || 0; },
    get draftStrokeCount() { return value?.s.length || 0; },
    get canUndo() { return undoStack.length > 0; },
    get canRedo() { return redoStack.length > 0; },
  };
}

export const STROKE_VERSION = 1;
export const STROKE_Q = 10000;
export const STROKE_RR_MIN = 1;
export const STROKE_RR_MAX = 100;
export const STROKE_RR_DEFAULT = 12;
export const STROKE_MAX_STROKES = 100;
export const STROKE_MAX_POINTS = 1000;
export const STROKE_MAX_TOTAL = 10000;

const ALL_LABELS = new Set([0, 1, 2, 3, 4, 5, 6, 7]);

function isPlainObject(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function integerInRange(value, lo, hi, fallback = lo) {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return Math.min(hi, Math.max(lo, Math.floor(value + 0.5)));
}

function allowedSet(allowed) {
  return allowed instanceof Set ? allowed : ALL_LABELS;
}

export function normalizeStrokes(value, allowed = ALL_LABELS) {
  if (!isPlainObject(value) || !Object.hasOwn(value, "v") || value.v !== STROKE_VERSION ||
      !Object.hasOwn(value, "s") || !Array.isArray(value.s)) return null;

  const labels = allowedSet(allowed);
  const w = integerInRange(Object.hasOwn(value, "w") ? value.w : 0, 0, 100000, 0);
  const h = integerInRange(Object.hasOwn(value, "h") ? value.h : 0, 0, 100000, 0);
  const s = [];
  let total = 0;

  for (const raw of value.s) {
    if (s.length >= STROKE_MAX_STROKES || total >= STROKE_MAX_TOTAL) break;
    if (!Array.isArray(raw) || raw.length < 4) continue;
    let valid = true;
    for (let i = 0; i < raw.length; i++) {
      if (typeof raw[i] !== "number" || !Number.isFinite(raw[i])) { valid = false; break; }
    }
    if (!valid) continue;

    const length = raw.length % 2 ? raw.length - 1 : raw.length;
    if (length < 4) continue;

    const label = Math.floor(raw[0] + 0.5);
    if (label < 0 || label > 7 || !labels.has(label)) continue;

    const rr = integerInRange(raw[1], STROKE_RR_MIN, STROKE_RR_MAX, STROKE_RR_MIN);
    const linePoints = Math.min(STROKE_MAX_POINTS, (length - 2) / 2, STROKE_MAX_TOTAL - total);
    if (linePoints < 1) break;
    const line = [label, rr];
    for (let i = 0; i < linePoints * 2; i++) {
      line.push(integerInRange(raw[i + 2], 0, STROKE_Q, 0));
    }
    s.push(line);
    total += linePoints;
  }

  return s.length ? { v: STROKE_VERSION, w, h, s } : null;
}

export function strokePixel(q, n) {
  const size = Number.isFinite(n) ? Math.floor(n) : 0;
  if (size <= 0) return 0;
  const coord = integerInRange(q, 0, STROKE_Q, 0);
  return Math.min(size - 1, Math.floor((coord * size) / STROKE_Q));
}

export function pixelToStrokeQ(px, n) {
  const size = Number.isFinite(n) ? Math.floor(n) : 0;
  if (size <= 0) return 0;
  const pixel = Math.min(size - 1, Math.max(0, Number.isFinite(px) ? Math.floor(px) : 0));
  return Math.min(STROKE_Q, Math.floor(((2 * pixel + 1) * STROKE_Q) / (2 * size)));
}

export function strokeRadiusPx(rr, W, H) {
  const shortSide = Math.max(0, Math.floor(Math.min(Number.isFinite(W) ? W : 0, Number.isFinite(H) ? H : 0)));
  const radius = integerInRange(rr, STROKE_RR_MIN, STROKE_RR_MAX, STROKE_RR_MIN);
  return Math.floor((2 * radius * shortSide + 1000) / 2000);
}

function integerSqrt(value) {
  let t = Math.floor(Math.sqrt(value));
  while (t * t > value) t--;
  while ((t + 1) * (t + 1) <= value) t++;
  return t;
}

export function diskHalfWidths(radius) {
  const r = Math.max(0, Math.floor(Number.isFinite(radius) ? radius : 0));
  const hw = new Int32Array(r + 1);
  for (let k = 0; k <= r; k++) hw[k] = integerSqrt(r * r + r - k * k);
  return hw;
}

export function forEachCenter(stroke, W, H, visit) {
  const w = Number.isFinite(W) ? Math.floor(W) : 0;
  const h = Number.isFinite(H) ? Math.floor(H) : 0;
  if (w <= 0 || h <= 0 || !Array.isArray(stroke) || stroke.length < 4 || typeof visit !== "function") return;

  let x0 = strokePixel(stroke[2], w), y0 = strokePixel(stroke[3], h);
  visit(x0, y0);
  for (let i = 4; i + 1 < stroke.length; i += 2) {
    const x1 = strokePixel(stroke[i], w), y1 = strokePixel(stroke[i + 1], h);
    const dx = x1 - x0, dy = y1 - y0;
    const n = Math.max(Math.abs(dx), Math.abs(dy));
    if (n > 0) {
      for (let k = 1; k <= n; k++) {
        const cx = x0 + Math.floor((2 * k * dx + n) / (2 * n));
        const cy = y0 + Math.floor((2 * k * dy + n) / (2 * n));
        visit(cx, cy);
      }
    }
    x0 = x1;
    y0 = y1;
  }
}

function subtractRanges(uncovered, covered) {
  const remaining = [];
  let rangeIndex = 0;
  for (const [left, right] of uncovered) {
    let x = left, i = rangeIndex;
    while (i < covered.length && covered[i][1] < x) i++;
    while (i < covered.length && covered[i][0] <= right) {
      const [start, end] = covered[i];
      if (start > x) remaining.push([x, start - 1]);
      if (end >= x) x = end + 1;
      if (x > right) break;
      i++;
    }
    if (x <= right) remaining.push([x, right]);
    rangeIndex = i;
  }
  return remaining;
}

export function strokeAspectChanged(value, W, H) {
  const norm = normalizeStrokes(value);
  if (!norm || norm.w <= 0 || norm.h <= 0 || !(W > 0) || !(H > 0)) return false;
  const width = Math.floor(W), height = Math.floor(H);
  const oldArea = norm.w * height;
  return Math.abs(oldArea - norm.h * width) * 100 > oldArea;
}

export function rasterizeStrokes(W, H, value, allowed = ALL_LABELS) {
  const norm = normalizeStrokes(value, allowed);
  if (!norm) return null;
  const w = Math.max(0, Math.floor(Number.isFinite(W) ? W : 0));
  const h = Math.max(0, Math.floor(Number.isFinite(H) ? H : 0));
  const labels = new Uint8Array(w * h);
  const centerGeneration = new Uint8Array(w * h);
  const widthsByRadius = new Map();
  const uncoveredByRow = w > 0 && h > 0 ? Array.from({ length: h }, () => [[0, w - 1]]) : [];

  if (w > 0 && h > 0) {
    for (let lineIndex = norm.s.length - 1; lineIndex >= 0; lineIndex--) {
      const line = norm.s[lineIndex];
      const label = line[0];
      const radius = strokeRadiusPx(line[1], w, h);
      let hw = widthsByRadius.get(radius);
      if (!hw) { hw = diskHalfWidths(radius); widthsByRadius.set(radius, hw); }
      const generation = lineIndex + 1;
      const centerRows = Array.from({ length: h }, () => []);
      let minX = w, maxX = -1, minY = h, maxY = -1;
      forEachCenter(line, w, h, (cx, cy) => {
        const at = cy * w + cx;
        if (centerGeneration[at] !== generation) {
          centerGeneration[at] = generation;
          centerRows[cy].push(cx);
        }
        minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);
      });
      if (maxX < minX) continue;
      const outX0 = Math.max(0, minX - radius), outX1 = Math.min(w - 1, maxX + radius);
      const outY0 = Math.max(0, minY - radius), outY1 = Math.min(h - 1, maxY + radius);
      const rowRuns = new Array(h);
      for (let sourceY = minY; sourceY <= maxY; sourceY++) {
        const centers = centerRows[sourceY];
        if (!centers.length) continue;
        centers.sort((a, b) => a - b);
        const runs = [];
        let runStart = centers[0], previous = centers[0];
        for (let i = 1; i < centers.length; i++) {
          const x = centers[i];
          if (x > previous + 1) {
            runs.push([runStart, previous]);
            runStart = x;
          }
          previous = x;
        }
        runs.push([runStart, previous]);
        const gapOrder = [];
        for (let i = 1; i < runs.length; i++) gapOrder.push([runs[i][0] - runs[i - 1][1], i - 1]);
        gapOrder.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
        rowRuns[sourceY] = { runs, gapOrder };
      }

      for (let y = outY0; y <= outY1; y++) {
        let uncovered = uncoveredByRow[y];
        if (!uncovered.length) continue;
        for (let dy = 0; dy <= radius && uncovered.length; dy++) {
          for (let side = 0; side < (dy ? 2 : 1) && uncovered.length; side++) {
            const sourceY = y + (side ? dy : -dy);
            if (sourceY < minY || sourceY > maxY) continue;
            const rowData = rowRuns[sourceY];
            if (!rowData) continue;
            const { runs, gapOrder } = rowData;
            const half = hw[dy], threshold = 2 * half + 1;
            let lo = 0, hi = gapOrder.length;
            while (lo < hi) {
              const mid = (lo + hi) >> 1;
              if (gapOrder[mid][0] > threshold) lo = mid + 1;
              else hi = mid;
            }
            const intervals = [];
            if (lo === 0) {
              intervals.push([Math.max(outX0, runs[0][0] - half), Math.min(outX1, runs[runs.length - 1][1] + half)]);
            } else {
              const breaks = gapOrder.slice(0, lo).map((item) => item[1]).sort((a, b) => a - b);
              let first = 0;
              for (const last of breaks) {
                intervals.push([Math.max(outX0, runs[first][0] - half), Math.min(outX1, runs[last][1] + half)]);
                first = last + 1;
              }
              intervals.push([Math.max(outX0, runs[first][0] - half), Math.min(outX1, runs[runs.length - 1][1] + half)]);
            }
            const row = y * w;
            for (const [left, right] of intervals) {
              for (const [remainingLeft, remainingRight] of uncovered) {
                const start = Math.max(left, remainingLeft), end = Math.min(right, remainingRight);
                if (start <= end) labels.fill(label, row + start, row + end + 1);
              }
            }
            uncovered = subtractRanges(uncovered, intervals);
          }
        }
        uncoveredByRow[y] = uncovered;
      }
    }
  }

  const counts = new Int32Array(8);
  for (const label of labels) counts[label]++;
  return {
    labels,
    counts,
    strokes: norm.s.length,
    points: norm.s.reduce((sum, line) => sum + (line.length - 2) / 2, 0),
    aspectChanged: strokeAspectChanged(norm, w, h),
  };
}

export function summarizeStrokes(value, param = {}) {
  const paramLabels = Array.isArray(param?.labels) ? param.labels : [];
  const allowed = new Set(paramLabels.map((label) => label?.value).filter((v) => Number.isInteger(v) && v >= 1 && v <= 7));
  if (param?.eraser === true) allowed.add(0);
  const norm = normalizeStrokes(value, allowed);
  if (!norm) return { text: "ยังไม่มีเส้น", short: "ยังไม่มีเส้น", empty: true };

  const counts = new Map();
  for (const line of norm.s) counts.set(line[0], (counts.get(line[0]) || 0) + 1);
  const textParts = [], shortParts = [];
  for (const label of paramLabels) {
    const count = counts.get(label.value) || 0;
    if (!count) continue;
    const name = String(label.name || `ป้าย ${label.value}`);
    textParts.push(`${name} ${count} เส้น`);
    shortParts.push(`${name} ${count}`);
  }
  const erased = counts.get(0) || 0;
  if (erased) { textParts.push(`ยางลบ ${erased} ครั้ง`); shortParts.push(`ยางลบ ${erased}`); }
  return {
    text: textParts.join(" · ") || "ยังไม่มีเส้น",
    short: shortParts.join(" · ") || "ยังไม่มีเส้น",
    empty: false,
  };
}

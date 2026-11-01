import { normalizeStrokes } from "./filters/stroke-mask.js";

const PYTHON_KEYWORDS = new Set([
  "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
  "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
  "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
]);

function safeVariableName(name) {
  return typeof name === "string" && /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) && !PYTHON_KEYWORDS.has(name) ? name : "strokes5g";
}

function allowedForParam(param) {
  const labels = Array.isArray(param?.labels) ? param.labels : [];
  const allowed = new Set(labels.map((x) => x?.value).filter((v) => Number.isInteger(v) && v >= 1 && v <= 7));
  if (param?.eraser === true) allowed.add(0);
  return allowed;
}

export function strokesPyLiteral(value, varName, param = {}) {
  const name = safeVariableName(varName);
  const norm = normalizeStrokes(value, allowedForParam(param));
  if (!norm) return [`${name} = []`];

  const lines = ["# เส้นขีดจากแอป: ป้ายและพิกัดเป็นจำนวนเต็ม", `${name} = [`];
  for (const stroke of norm.s) {
    const pairs = [];
    for (let i = 2; i + 1 < stroke.length; i += 2) pairs.push(`${stroke[i]}, ${stroke[i + 1]}`);
    for (let start = 0; start < pairs.length; start += 8) {
      const chunk = pairs.slice(start, start + 8).join(", ");
      if (start === 0) {
        const close = start + 8 >= pairs.length ? "])," : ",";
        lines.push(`    (${stroke[0]}, ${stroke[1]}, [${chunk}${close}`);
      } else {
        const close = start + 8 >= pairs.length ? "])," : ",";
        lines.push(`             ${chunk}${close}`);
      }
    }
  }
  lines.push("]");
  return lines;
}

export function strokeRasterPyLines() {
  return [
    "def stroke_labels(w, h, strokes):",
    "    import math",
    "    lab = np.zeros((h, w), np.uint8)",
    "    if w == 0 or h == 0:",
    "        return lab",
    "    s_min = min(w, h)",
    "    for label, rr, pts in strokes:",
    "        r = (2 * rr * s_min + 1000) // 2000",
    "        px = np.array([min(w - 1, (pts[i] * w) // 10000) for i in range(0, len(pts), 2)], np.int64)",
    "        py = np.array([min(h - 1, (pts[i] * h) // 10000) for i in range(1, len(pts), 2)], np.int64)",
    "        if px.size == 0:",
    "            continue",
    "        cx_parts, cy_parts = [px[:1]], [py[:1]]",
    "        for i in range(1, len(px)):",
    "            x0, y0, dx, dy = int(px[i - 1]), int(py[i - 1]), int(px[i] - px[i - 1]), int(py[i] - py[i - 1])",
    "            n = max(abs(dx), abs(dy))",
    "            if n:",
    "                k = np.arange(1, n + 1, dtype=np.int64)",
    "                xs = x0 + (2 * k * dx + n) // (2 * n)",
    "                ys = y0 + (2 * k * dy + n) // (2 * n)",
    "                cx_parts.append(xs)",
    "                cy_parts.append(ys)",
    "        cx, cy = np.concatenate(cx_parts), np.concatenate(cy_parts)",
    "        keep = (cx >= -r - 1) & (cx <= w + r) & (cy >= -r - 1) & (cy <= h + r)",
    "        cx, cy = cx[keep], cy[keep]",
    "        if cx.size == 0:",
    "            continue",
    "        cx_min, cx_max = int(cx.min()), int(cx.max())",
    "        cy_min, cy_max = int(cy.min()), int(cy.max())",
    "        span_x = cx_max - cx_min + 1",
    "        centers = np.unique((cy - cy_min) * span_x + (cx - cx_min))",
    "        cy, cx = centers // span_x + cy_min, centers % span_x + cx_min",
    "        y0, y1 = max(0, int(cy.min()) - r), min(h - 1, int(cy.max()) + r)",
    "        x0, x1 = max(0, int(cx.min()) - r), min(w - 1, int(cx.max()) + r)",
    "        if y1 < y0 or x1 < x0:",
    "            continue",
    "        ox, oy = min(x0 - r - 1, cx_min), min(y0 - r - 1, cy_min)",
    "        ex, ey = max(x1 + r + 1, cx_max), max(y1 + r + 1, cy_max)",
    "        cw, ch = ex - ox + 1, ey - oy + 1",
    "        src = np.full((ch, cw), 255, np.uint8)",
    "        src[cy - oy, cx - ox] = 0",
    "        d = cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(np.float64)",
    "        cov = (d * d) <= (r * r + r + 0.5)",
    "        cov = cov[y0 - oy:y1 - oy + 1, x0 - ox:x1 - ox + 1]",
    "        lab[y0:y1 + 1, x0:x1 + 1][cov] = label",
    "    return lab",
  ];
}

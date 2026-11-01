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
    "        hw = np.array([math.isqrt(r * r + r - k * k) for k in range(r + 1)], np.int64)",
    "        px = np.array([min(w - 1, (pts[i] * w) // 10000) for i in range(0, len(pts), 2)], np.int64)",
    "        py = np.array([min(h - 1, (pts[i] * h) // 10000) for i in range(1, len(pts), 2)], np.int64)",
    "        if px.size == 0:",
    "            continue",
    "        centers = np.zeros((h, w), np.bool_)",
    "        centers[py[0], px[0]] = True",
    "        for i in range(1, len(px)):",
    "            x0, y0, dx, dy = int(px[i - 1]), int(py[i - 1]), int(px[i] - px[i - 1]), int(py[i] - py[i - 1])",
    "            n = max(abs(dx), abs(dy))",
    "            if n:",
    "                k = np.arange(1, n + 1, dtype=np.int64)",
    "                xs = x0 + (2 * k * dx + n) // (2 * n)",
    "                ys = y0 + (2 * k * dy + n) // (2 * n)",
    "                centers[ys, xs] = True",
    "        cy, cx = np.nonzero(centers)",
    "        bx0, bx1 = max(0, int(cx.min()) - r), min(w - 1, int(cx.max()) + r)",
    "        by0, by1 = max(0, int(cy.min()) - r), min(h - 1, int(cy.max()) + r)",
    "        diff = np.zeros((by1 - by0 + 1, bx1 - bx0 + 2), np.int32)",
    "        for source_y in range(int(cy.min()), int(cy.max()) + 1):",
    "            row = centers[source_y, bx0:bx1 + 1]",
    "            edges = np.diff(np.concatenate(([False], row, [False])).astype(np.int8))",
    "            starts = np.flatnonzero(edges == 1) + bx0",
    "            if starts.size == 0:",
    "                continue",
    "            ends = np.flatnonzero(edges == -1) - 1 + bx0",
    "            gaps = starts[1:] - ends[:-1]",
    "            max_gap = int(gaps.max()) if gaps.size else 0",
    "            target_y = np.arange(max(by0, source_y - r), min(by1, source_y + r) + 1, dtype=np.intp)",
    "            half = hw[np.abs(target_y - source_y)]",
    "            solid = 2 * half + 1 >= max_gap",
    "            rows = target_y[solid] - by0",
    "            left = np.maximum(bx0, starts[0] - half[solid]) - bx0",
    "            right = np.minimum(bx1, ends[-1] + half[solid]) - bx0",
    "            diff[rows, left] += 1",
    "            diff[rows, right + 1] -= 1",
    "            rows = target_y[~solid] - by0",
    "            half = half[~solid]",
    "            for start, end in zip(starts, ends):",
    "                left = np.maximum(bx0, start - half) - bx0",
    "                right = np.minimum(bx1, end + half) - bx0",
    "                diff[rows, left] += 1",
    "                diff[rows, right + 1] -= 1",
    "        cover = np.cumsum(diff, axis=1, dtype=np.int32)[:, :-1] > 0",
    "        view = lab[by0:by1 + 1, bx0:bx1 + 1]",
    "        view[cover] = label",
    "    return lab",
  ];
}

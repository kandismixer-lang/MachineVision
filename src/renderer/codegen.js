// สร้างโค้ด Python (OpenCV) ตามลำดับ Box ที่ภาพวิ่งผ่านจริง ๆ ในกล่องผลลัพธ์หนึ่งกล่อง
// (คนละอย่างกับปุ่ม Export ที่เป็นโค้ดเทรน YOLO)
//
// pipeline = [{ id (filterId), params, nodeId }, ...] เรียงจากแหล่งภาพ → ปลายทาง
// แต่ละ step map เป็นโค้ด cv2.* ตามค่าที่ตั้งไว้จริง
// จัดการแปลงเป็นขาวดำอัตโนมัติ (เหมือนในแอป) เมื่อ op ต้องการภาพ gray
//
// หมายเหตุ: โค้ดที่ gen ใช้ BGR (cv2.imread) ตามธรรมเนียม OpenCV — ต่างจากในแอปที่ทำงานบน RGB
// ผลลัพธ์เชิงหลักการเหมือนกัน (ค่าเนื้อสี H, threshold ฯลฯ เท่ากัน)
import { OPS } from "./filters/registry.js";

// ชื่อไทยของ op (โยงกับที่ผู้เรียนเห็นบนจอ) + teaching note สั้น ๆ ต่อ op คลาสสิก
const TEACH = {
  grayscale: "รวม 3 ช่องสีเป็นความสว่างเดียว — พื้นฐานก่อน threshold/edge",
  gaussian_blur: "เฉลี่ยเพื่อนบ้านแบบถ่วงน้ำหนักระฆังคว่ำ — ลด noise แบบสุ่ม",
  median_blur: "ใช้ค่ากลางของเพื่อนบ้าน จึงลบ noise เกลือ-พริกไทยดีกว่า Gaussian",
  binary_threshold: "สว่างเกินเกณฑ์ = ขาว, ต่ำกว่า = ดำ — แบ่งภาพขั้นพื้นฐานสุด",
  otsu_threshold: "หาเกณฑ์ที่ดีที่สุดให้อัตโนมัติจาก histogram",
  canny: "ตรวจขอบหลายขั้น — มาตรฐานทองของการหาขอบ",
  morphology: "จัดรูปทรงด้วย kernel (กัด/ขยาย/อุดรู/ลบจุด)",
  find_contours: "หาเส้นรอบวัตถุขาว แล้วนับ/วาด",
};
function thaiName(opId) {
  return (OPS[opId] && OPS[opId].name) || opId;
}

function odd(n) {
  n = Math.round(Number(n) || 1);
  return n % 2 === 0 ? n + 1 : n;
}
function int(n) {
  return Math.round(Number(n) || 0);
}
function num(n) {
  return Number(n) || 0;
}

// แยก opId จริง + ค่าพารามิเตอร์ (ตัด prefix opId__ ของ merged node ออก)
function resolveOp(step) {
  const p = step.params || {};
  const hasMode = Object.prototype.hasOwnProperty.call(p, "mode");
  const opId = hasMode ? p.mode : step.id;
  const local = {};
  if (hasMode) {
    const prefix = opId + "__";
    for (const k in p) if (k.startsWith(prefix)) local[k.slice(prefix.length)] = p[k];
  } else {
    for (const k in p) local[k] = p[k];
  }
  return { opId, local };
}

// ตารางแปลง op → โค้ด Python
//   needs: 'gray' = ต้องแปลงเป็นขาวดำก่อน (แทรกให้อัตโนมัติ) | 'any'
//   out:   สภาพภาพหลังทำงาน 'gray' | 'color' | 'same'  (ใช้ track ว่าตอนนี้ภาพขาวดำหรือสี)
//   gen(p, ctx): คืน array ของบรรทัดโค้ด (ctx.gray = ตอนนี้ภาพเป็นขาวดำหรือไม่)
const GEN = {
  // ---- ปริภูมิสี ----
  grayscale: { needs: "any", out: "gray", gen: (p, ctx) =>
    ctx.gray ? ["# ภาพเป็นขาวดำอยู่แล้ว"] : ["img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"] },

  colorspace: { needs: "any", out: "color", gen: (p, ctx) => {
    const code = { hsv: "COLOR_BGR2HSV", hls: "COLOR_BGR2HLS", lab: "COLOR_BGR2Lab", ycrcb: "COLOR_BGR2YCrCb" }[p.space] || "COLOR_BGR2HSV";
    const lines = ctx.gray ? ["img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)"] : [];
    lines.push(`img = cv2.cvtColor(img, cv2.${code})  # ดูข้อมูลในปริภูมิสี ${p.space}`);
    return lines;
  } },

  channel_split: { needs: "any", out: "gray", gen: (p, ctx) => {
    const isHSV = ["h", "s", "v"].includes(p.channel);
    const idx = { r: 2, g: 1, b: 0, h: 0, s: 1, v: 2 }[p.channel]; // BGR: b=0,g=1,r=2
    const lines = [];
    if (ctx.gray) lines.push("img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)");
    if (isHSV) lines.push("img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)");
    lines.push(`img = cv2.split(img)[${idx}]  # ดูช่อง ${String(p.channel).toUpperCase()}`);
    return lines;
  } },

  // ---- ปรับปรุงภาพ ----
  brightness_contrast: { needs: "any", out: "same", gen: (p) =>
    [`img = cv2.convertScaleAbs(img, alpha=${num(p.alpha)}, beta=${int(p.beta)})`] },

  gamma: { needs: "any", out: "same", gen: (p) => [
    `inv_gamma = 1.0 / ${num(p.g) || 1}`,
    "lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(\"uint8\")",
    "img = cv2.LUT(img, lut)",
  ] },

  hist_equalize: { needs: "gray", out: "gray", gen: () => ["img = cv2.equalizeHist(img)"] },

  clahe: { needs: "gray", out: "gray", gen: (p) => [
    `clahe = cv2.createCLAHE(clipLimit=${num(p.clip)}, tileGridSize=(${int(p.tile)}, ${int(p.tile)}))`,
    "img = clahe.apply(img)",
  ] },

  unsharp: { needs: "any", out: "same", gen: (p) => {
    const k = odd(p.radius);
    return [
      `blur = cv2.GaussianBlur(img, (${k}, ${k}), 0)`,
      `img = cv2.addWeighted(img, ${1 + num(p.amount)}, blur, ${-num(p.amount)}, 0)`,
    ];
  } },

  laplacian_sharpen: { needs: "any", out: "same", gen: (p) => [
    "lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)",
    "lap = cv2.convertScaleAbs(lap)",
    `img = cv2.addWeighted(img, 1, lap, ${-num(p.strength)}, 0)`,
  ] },

  // ---- ฟื้นฟูภาพ ----
  salt_pepper_noise: { needs: "any", out: "same", gen: (p) => [
    `# ใส่ noise เกลือ-พริกไทย ${int(p.amount)}%`,
    `n = int(img.shape[0] * img.shape[1] * ${num(p.amount) / 100})`,
    "ys = np.random.randint(0, img.shape[0], n)",
    "xs = np.random.randint(0, img.shape[1], n)",
    "vals = np.where(np.random.rand(n) < 0.5, 0, 255)",
    "img[ys, xs] = vals[:, None] if img.ndim == 3 else vals",
  ] },

  median_blur: { needs: "any", out: "same", gen: (p) => [`img = cv2.medianBlur(img, ${odd(p.ksize)})`] },
  gaussian_blur: { needs: "any", out: "same", gen: (p) => { const k = odd(p.ksize); return [`img = cv2.GaussianBlur(img, (${k}, ${k}), 0)`]; } },
  box_blur: { needs: "any", out: "same", gen: (p) => { const k = Math.max(1, int(p.ksize)); return [`img = cv2.blur(img, (${k}, ${k}))`]; } },
  bilateral: { needs: "any", out: "same", gen: (p) =>
    [`img = cv2.bilateralFilter(img, ${int(p.d)}, ${int(p.sigmaColor)}, ${int(p.sigmaSpace)})`] },

  // ---- สัณฐานวิทยา ----
  morphology: { needs: "gray", out: "gray", gen: (p) => {
    const shape = { rect: "MORPH_RECT", ellipse: "MORPH_ELLIPSE", cross: "MORPH_CROSS" }[p.shape] || "MORPH_RECT";
    const k = Math.max(1, int(p.ksize));
    const it = Math.max(1, int(p.iterations));
    const lines = [`kernel = cv2.getStructuringElement(cv2.${shape}, (${k}, ${k}))`];
    if (p.op === "erode") lines.push(`img = cv2.erode(img, kernel, iterations=${it})`);
    else if (p.op === "dilate") lines.push(`img = cv2.dilate(img, kernel, iterations=${it})`);
    else {
      const opc = { open: "MORPH_OPEN", close: "MORPH_CLOSE", gradient: "MORPH_GRADIENT", tophat: "MORPH_TOPHAT", blackhat: "MORPH_BLACKHAT" }[p.op] || "MORPH_OPEN";
      lines.push(`img = cv2.morphologyEx(img, cv2.${opc}, kernel, iterations=${it})`);
    }
    return lines;
  } },

  // ---- แบ่งส่วนภาพ ----
  binary_threshold: { needs: "gray", out: "gray", gen: (p) => {
    const t = p.invert ? "cv2.THRESH_BINARY_INV" : "cv2.THRESH_BINARY";
    return [`_, img = cv2.threshold(img, ${int(p.thresh)}, 255, ${t})`];
  } },

  otsu_threshold: { needs: "gray", out: "gray", gen: (p) => {
    const t = p.invert ? "cv2.THRESH_BINARY_INV" : "cv2.THRESH_BINARY";
    return [`_, img = cv2.threshold(img, 0, 255, ${t} + cv2.THRESH_OTSU)  # หาเกณฑ์อัตโนมัติ`];
  } },

  adaptive_threshold: { needs: "gray", out: "gray", gen: (p) =>
    [`img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ${Math.max(3, odd(p.blockSize))}, ${int(p.C)})`] },

  canny: { needs: "gray", out: "gray", gen: (p) => [`img = cv2.Canny(img, ${int(p.t1)}, ${int(p.t2)})`] },

  sobel: { needs: "gray", out: "gray", gen: (p) => {
    const k = odd(p.ksize);
    if (p.dir === "x") return [`img = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=${k}))`];
    if (p.dir === "y") return [`img = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=${k}))`];
    return [
      `gx = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=${k}))`,
      `gy = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=${k}))`,
      "img = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)",
    ];
  } },

  find_contours: { needs: "gray", out: "color", gen: (p) => [
    "contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)",
    "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
    `for c in contours:`,
    `    if cv2.contourArea(c) < ${int(p.minArea)}:`,
    "        continue",
    "    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)",
  ] },

  connected_components: { needs: "gray", out: "color", gen: () => [
    "n, labels = cv2.connectedComponents(img)",
    "hue = np.uint8(179 * labels / max(1, n - 1))",
    "blank = np.full_like(hue, 255)",
    "img = cv2.merge([hue, blank, blank])",
    "img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)",
    "img[labels == 0] = 0  # พื้นหลังเป็นสีดำ",
  ] },

  watershed: { needs: "any", out: "color", gen: (p) => [
    "gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
    "img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img",
    "_, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)",
    "kernel = np.ones((3, 3), np.uint8)",
    "opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)",
    "sure_bg = cv2.dilate(opening, kernel, iterations=3)",
    "dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)",
    `_, sure_fg = cv2.threshold(dist, ${num(p.ratio)} * dist.max(), 255, 0)`,
    "sure_fg = np.uint8(sure_fg)",
    "unknown = cv2.subtract(sure_bg, sure_fg)",
    "_, markers = cv2.connectedComponents(sure_fg)",
    "markers = markers + 1",
    "markers[unknown == 255] = 0",
    "markers = cv2.watershed(img, markers)",
    "img[markers == -1] = (0, 0, 255)  # เส้นแบ่งวัตถุสีแดง",
  ] },

  // ---- ตรวจจับวัตถุ ----
  bounding_boxes: { needs: "gray", out: "color", gen: (p) => [
    "contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)",
    "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
    "for c in contours:",
    `    if cv2.contourArea(c) < ${int(p.minArea)}:`,
    "        continue",
    "    x, y, w, h = cv2.boundingRect(c)",
    "    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)",
  ] },

  hough_circles: { needs: "gray", out: "color", gen: (p) => {
    let maxR = int(p.maxR), minR = int(p.minR);
    if (maxR > 0 && maxR < minR) maxR = 0;
    return [
      "blur = cv2.medianBlur(img, 5)",
      `circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, ${num(p.dp)}, ${int(p.minDist)},`,
      `                           param1=100, param2=${int(p.param2)}, minRadius=${minR}, maxRadius=${maxR})`,
      "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
      "if circles is not None:",
      "    for x, y, r in np.uint16(np.around(circles))[0]:",
      "        cv2.circle(img, (x, y), r, (0, 255, 0), 2)",
      "        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)",
    ];
  } },

  grabcut: { needs: "any", out: "color", gen: (p) => {
    const lines = ["img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img",
      "mask = np.zeros(img.shape[:2], np.uint8)",
      "bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)",
      `mx, my = int(img.shape[1] * ${num(p.margin) / 100}), int(img.shape[0] * ${num(p.margin) / 100})`,
      "rect = (mx, my, img.shape[1] - 2 * mx, img.shape[0] - 2 * my)",
      `cv2.grabCut(img, mask, rect, bgd, fgd, ${Math.max(1, int(p.iter))}, cv2.GC_INIT_WITH_RECT)`,
      "fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(\"uint8\")"];
    if (p.output === "mask") lines.push("img = fg");
    else lines.push("img = cv2.bitwise_and(img, img, mask=fg)");
    return lines;
  } },

  // ---- ประมวลผลภาพสี ----
  color_inrange: { needs: "any", out: null, gen: (p, ctx) => {
    const lines = [];
    if (ctx.gray) lines.push("img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)");
    lines.push("hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)");
    lines.push(`lower = np.array([${int(p.hMin)}, ${int(p.sMin)}, ${int(p.vMin)}])`);
    lines.push(`upper = np.array([${int(p.hMax)}, 255, 255])`);
    lines.push("mask = cv2.inRange(hsv, lower, upper)");
    if (p.output === "mask") lines.push("img = mask");
    else lines.push("img = cv2.bitwise_and(img, img, mask=mask)");
    return lines;
  }, outOf: (p) => (p.output === "mask" ? "gray" : "color") },

  invert: { needs: "any", out: "same", gen: () => ["img = cv2.bitwise_not(img)"] },

  harris_corners: { needs: "gray", out: "color", gen: (p) => {
    const ksize = p.ksize % 2 === 0 ? p.ksize + 1 : p.ksize;
    return [
      "gray_f = np.float32(img)",
      `resp = cv2.cornerHarris(gray_f, ${int(p.blockSize)}, ${ksize}, ${num(p.k)})`,
      "resp = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX)",
      "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
      `thresh = ${num(p.thresh)} / 100 * resp.max()`,
      "ys, xs = np.where(resp > thresh)",
      "for x, y in zip(xs, ys):",
      "    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 1)",
    ];
  } },

  shi_tomasi_corners: { needs: "gray", out: "color", gen: (p) => [
    `corners = cv2.goodFeaturesToTrack(img, ${int(p.maxCorners)}, ${num(p.quality) / 100}, ${int(p.minDist)},`,
    `                                  blockSize=${int(p.blockSize)}, useHarrisDetector=${p.useHarris ? "True" : "False"})`,
    "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
    "if corners is not None:",
    "    for x, y in corners.reshape(-1, 2):",
    "        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), 2)",
  ] },
};

// สร้างโค้ด Python เต็มจาก pipeline
export function buildPythonCode(pipeline, opts = {}) {
  const name = opts.displayName || "ผลลัพธ์";
  const body = [];
  const ctx = { gray: false }; // เริ่มจากภาพสี (cv2.imread = BGR)

  if (!pipeline || pipeline.length === 0) {
    body.push("# กล่องนี้แสดงภาพต้นฉบับ — ยังไม่ผ่าน filter ใด");
  }

  let stepNo = 0;
  for (const step of pipeline) {
    const { opId, local } = resolveOp(step);
    const g = GEN[opId];
    if (!g) {
      body.push(`# (ข้าม: ยังไม่รองรับการ gen โค้ดของ '${opId}')`);
      continue;
    }
    stepNo++;
    // แปลงเป็นขาวดำอัตโนมัติถ้า op ต้องการ (เหมือนที่แอปทำให้เบื้องหลัง)
    if (g.needs === "gray" && !ctx.gray) {
      body.push("img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (แปลงขาวดำอัตโนมัติ)");
      ctx.gray = true;
    }
    body.push(`# ── ขั้นที่ ${stepNo}: ${thaiName(opId)} (${opId}) ──`);
    if (TEACH[opId]) body.push(`# ${TEACH[opId]}`);
    for (const line of g.gen(local, ctx)) body.push(line);
    // อัปเดตสถานะภาพ (ขาวดำ/สี)
    const out = g.outOf ? g.outOf(local) : g.out;
    if (out === "gray") ctx.gray = true;
    else if (out === "color") ctx.gray = false;
    body.push("");
  }

  const indented = body.map((l) => (l === "" ? "" : "    " + l)).join("\n");

  return `# -*- coding: utf-8 -*-
# โค้ดนี้สร้างอัตโนมัติจากลำดับ Box ในกล่องผลลัพธ์: "${name}"
# ทำ image processing ตามขั้นตอนเดียวกับที่เห็นในโปรแกรม
import cv2
import numpy as np


def process(path):
    img = cv2.imread(path)  # อ่านภาพ (BGR)
    if img is None:
        raise FileNotFoundError(path)

${indented}
    return img


if __name__ == "__main__":
    result = process("input.jpg")   # << ใส่ path ภาพของคุณ
    cv2.imwrite("output.png", result)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
`;
}

// UI metadata — เพิ่มความสวย/สื่อความ โดยไม่แตะ backend algorithm
import { OPS, fixed2, signedText, quantizeBits, SHAPE_FILTER_PROP_TH } from "./registry.js";

// ไอคอนประจำการ์ดเครื่องมือ (Design §7.1 — ภาพ mockup มี icon chip หน้าแต่ละเครื่องมือ)
export const GROUP_ICON = {
  color_space: "🎨",
  image_adjust: "☀",
  sharpen: "✦",
  noise: "⁙",
  blur: "◉",
  enhancement: "✨",
  restoration: "💧",
  morphology: "🔲",
  threshold: "◩",
  edge_detection: "⌁",
  shape_analysis: "⌬",
  watershed: "≋",
  segmentation: "✂️",
  object_detection: "🎯",
  colour: "🌈",
  geometry: "📐",
  frequency: "〰",
  image_arithmetic: "±", // ใหม่ (Tier 2, 18a) — เช็คแล้วว่าไม่ชนกับ glyph ใดข้างบน
  // ใหม่ (Tier 2, 18b) — ใช้ตรงกับ OP_ICON ของ palette-panel.js (filterId เดียวกับ opId เพราะเป็น
  // single-op group ตามแนวเดียวกับ watershed/segmentation) เช็คแล้วไม่ชนกับ glyph เดิมด้านบน
  template_matching: "⌕",
  psnr: "📶",
  ssim: "≈",
  // ใหม่ (Tier 2, 18d) — ตรงกับ OP_ICON ของ palette-panel.js (single-op group: filterId = opId)
  mog2: "🎞",
  farneback_optical_flow: "⇶",
  pyrlk_optical_flow: "⤳",
  // ใหม่ (2026-09-30) — single-op group เหมือน watershed/grabcut (filterId = opId) ตรงกับ OP_ICON
  // ของ palette-panel.js · prewitt/roberts/log_edge อยู่ในกลุ่ม edge_detection เดิม จึงไม่มีบรรทัดที่นี่
  region_growing: "🪄",
  // ใหม่ (2026-10-07) — กลุ่มระดับหมวด palette (1 หมวด = 1 node) · เช็คแล้วไม่ชนกับ glyph ใน GROUP_ICON/OP_ICON/src ใด ๆ
  color_tools: "🖌",
  enhance_tools: "🌟",
  filter_denoise: "🧹",
  segment_edge: "🧱",
  analysis_detect: "🧮",
  image_quality: "🏅",
  motion: "🏃",
};

// ค่าตัวเลขที่การ์ดโชว์ ต้องเป็น "ค่าที่ op ใช้จริง" ไม่ใช่ค่าดิบใน properties — ไฟล์โปรเจกต์ที่แก้มือ
// (หรือ import มาจากเวอร์ชันที่ช่วงค่าต่างไป) ใส่ history: 9999 ได้ แต่ registry clamp เหลือ 2000 ก่อนส่งเข้า
// OpenCV เสมอ ถ้าการ์ดโชว์ 9999 ผู้ใช้จะอ่านค่าที่ไม่ตรงกับสิ่งที่โปรแกรมทำ · min/max/default อ่านจาก OPS
// (แหล่งความจริงเดียวกับที่ registry ใช้ clamp และที่ UI ใช้สร้าง slider) จึงเลื่อนออกจากกันไม่ได้
// กติกาแปลงค่าเหมือน numOr/numIn/intIn ใน registry.js เป๊ะ: อะไรที่ Number() แล้วไม่ finite = ใช้ค่า default
function pnum(opId, key, v, { int = false } = {}) {
  const spec = ((OPS[opId] && OPS[opId].params) || []).find((s) => s.key === key);
  if (!spec) return v; // ไม่มี param นี้ใน registry (ไม่ควรเกิด) — คืนค่าดิบ ดีกว่าโชว์ undefined
  const n = Number(v);
  const x = Number.isFinite(n) ? n : Number(spec.default);
  const c = Math.min(spec.max, Math.max(spec.min, x));
  return int ? Math.round(c) : c;
}

// ข้อความการ์ดของ op ที่มีตัวเลือก operation (IP batch 2 ชิ้น B)
const IMARITH_LABEL = { add: "A + B", subtract: "A − B", multiply: "A × B ÷ 255", divide: "A ÷ B × 255" };
const LOGICAL_LABEL = { and: "A AND B", or: "A OR B", xor: "A XOR B", a_and_not_b: "A AND NOT B" };
function ownOr(table, key, fallback) {
  return Object.hasOwn(table, key) ? table[key] : table[fallback];
}

// "บรรทัด 2" ของการ์ด node บนกระดาน (Design node-card §1) — สรุปค่าหลักที่กำลังตั้งไว้
// รับ local params (sub-params ของ op ที่เลือก ตัด prefix opId__ ออกแล้ว)
// key ตรวจสอบกับ src/renderer/filters/registry.js จริงแล้ว (ไม่ใช่ชื่อที่เดาไว้)
// IP batch 4A — ต่อท้ายหน่วย mm เมื่อเปิดใช้ (mmPerPx > 0) · 0 = ข้อความเดิมทุกตัวอักษร
function mmTail(opId, p) {
  const f = pnum(opId, "mmPerPx", p.mmPerPx);
  return f > 0 ? ` · ${f} มม./px` : "";
}

export const OP_PRIMARY = {
  grayscale: () => "BGR → เทา (Gray)",
  colorspace: (p) => "BGR → " + (p.space || "HSV").toUpperCase(),
  channel_split: (p) => "ช่อง: " + (p.channel || "R").toUpperCase(),
  rgb_adjust: (p) => `R ${p.rGain ?? 1} · G ${p.gGain ?? 1} · B ${p.bGain ?? 1}`,
  brightness_contrast: (p) => `คอนทราสต์ ${p.alpha ?? 1.2} · สว่าง ${p.beta ?? 0}`,
  gamma: (p) => (p.curve === "log" ? "log: c·log(1+r)" : "gamma: " + (p.g ?? 1.5)),
  hist_equalize: () => "ปรับฮิสโตแกรมทั้งภาพ",
  clahe: (p) => "clip: " + (p.clip ?? 2),
  // ใหม่ (2026-10-01, IP batch ENHANCE) — ตัวเลขอ่านผ่าน pnum เพื่อให้การ์ดโชว์ค่าหลัง clamp
  contrast_stretch: (p) => `ตัดมืด ${pnum("contrast_stretch", "lowPct", p.lowPct)}% · ตัดสว่าง ${pnum("contrast_stretch", "highPct", p.highPct)}%`,
  flatfield_correction: (p) => "เบลอพื้นหลัง: " + pnum("flatfield_correction", "blurSize", p.blurSize, { int: true }) + "px",
  unsharp: (p) => "ความเข้ม: " + (p.amount ?? 1),
  laplacian_sharpen: (p) => "ความเข้ม: " + (p.strength ?? 1),
  // IP batch 4A — Motion/Disk แสดงขนาด/มุม PSF · preset อื่นข้อความเดิม
  custom_kernel: (p) => {
    if (p.preset === "motion") return `motion ${pnum("custom_kernel", "psfSize", p.psfSize, { int: true })}px ${pnum("custom_kernel", "psfAngle", p.psfAngle, { int: true })}° · ${p.amount ?? 1}x`;
    if (p.preset === "disk") return `disk r${pnum("custom_kernel", "psfSize", p.psfSize, { int: true })} · ${p.amount ?? 1}x`;
    return `${p.preset ?? "sharpen"} · ${p.amount ?? 1}x`;
  },
  salt_pepper_noise: (p) => "noise: " + (p.amount ?? 5) + "%",
  // ใหม่ (2026-10-03, IP batch 2 ชิ้น C) — ตัวเลขทุกตัวผ่าน pnum (ค่าหลัง clamp ที่ op ใช้จริง) · int เฉพาะ param ที่ registry ใช้ intIn
  gaussian_noise: (p) => `${p.kind === "speckle" ? "speckle " : ""}σ ${pnum("gaussian_noise", "sigma", p.sigma)} · seed ${pnum("gaussian_noise", "seed", p.seed, { int: true })}`,
  wiener_filter: (p) => {
    const k0 = pnum("wiener_filter", "ksize", p.ksize, { int: true });
    const k = k0 % 2 === 0 ? k0 + 1 : k0; // เหมือน ensureOdd ใน registry (14 → 15)
    const nv = pnum("wiener_filter", "noiseVar", p.noiseVar);
    return `Wiener ${k}×${k} · noise ${nv > 0 ? nv : "auto"}`;
  },
  median_blur: (p) => "ksize: " + (p.ksize ?? 3),
  gaussian_blur: (p) => "ksize: " + (p.ksize ?? 5),
  box_blur: (p) => "ksize: " + (p.ksize ?? 5),
  bilateral: (p) => "d: " + (p.d ?? 7),
  morphology: (p) => (
    p.op === "fill_holes" ? "เติมรูในวัตถุ"
    : p.op === "clear_border" ? "ลบวัตถุที่แตะขอบ"
    : p.op === "remove_small" ? `ลบวัตถุ < ${p.minArea ?? 150}px`
    : p.op === "skeleton" ? "Skeleton / เส้นแกนกลาง"
    : p.op === "reconstruct_open" ? "Opening by Reconstruction"
    // ใหม่ (2026-10-01, IP batch MORPHOLOGY)
    : p.op === "convex_hull" ? "Convex Hull (รูปนูน)"
    : p.op === "endpoints_branchpoints" ? "จุดปลาย/จุดแยก (ต่อจาก Skeleton)"
    : (p.op || "erode") + " · k=" + (p.ksize ?? 5)
  ),
  binary_threshold: (p) => "thresh: " + (p.thresh ?? 127),
  otsu_threshold: (p) => (p.method === "triangle" ? "วิธี: Triangle (อัตโนมัติ)" : "วิธี: Otsu (อัตโนมัติ)"),
  adaptive_threshold: (p) => "block: " + (p.blockSize ?? 11),
  multi_otsu_threshold: () => "วิธี: Multi-Otsu (3 กลุ่ม, อัตโนมัติ)",
  canny: (p) => `ขอบ ${p.t1 ?? 50}–${p.t2 ?? 150}`,
  sobel: (p) => "ทิศ: " + String(p.dir || "x").toUpperCase(),
  laplacian_edge: (p) => "ksize: " + (p.ksize ?? 3),
  scharr_edge: (p) => "ทิศ: " + String(p.dir || "both").toUpperCase(),
  // ใหม่ (2026-09-30) — ตัวเลขทุกตัวอ่านผ่าน pnum เพื่อให้การ์ดโชว์ "ค่าหลัง clamp" ตรงกับที่ op ใช้จริง
  prewitt_edge: (p) => "ทิศ: " + String(p.dir || "both").toUpperCase(),
  roberts_edge: (p) => "ทิศ: " + String(p.dir || "both").toUpperCase(),
  // zeroThresh ไม่ใส่ { int: true }: registry ใช้ numIn (ไม่ปัด) กับ param นี้ ถ้าการ์ดปัดให้ ค่าที่โชว์จะ
  // ไม่ตรงกับค่าที่ op ใช้จริงเมื่อไฟล์โปรเจกต์ถูกแก้มือเป็น 8.4 — int จะใส่เฉพาะ param ที่ registry ใช้ intIn
  log_edge: (p) => `sigma ${pnum("log_edge", "sigma", p.sigma)} · เกณฑ์ ${pnum("log_edge", "zeroThresh", p.zeroThresh)}`,
  find_contours: () => "หาเส้นรอบวัตถุ",
  connected_components: () => "นับ/ระบายวัตถุ",
  distance_transform: (p) => "mask: " + (p.maskSize ?? "5") + "x" + (p.maskSize ?? "5"),
  local_std_filter: (p) => "หน้าต่าง: " + (p.windowSize ?? 9) + "px",
  local_entropy_filter: (p) => (p.windowSize ?? "5") + "x" + (p.windowSize ?? "5") + " · " + (p.bins ?? "16") + " ระดับ",
  regionprops: (p) => `วัดค่าต่อวัตถุ · พื้นที่ขั้นต่ำ ${p.minArea ?? 50}px · แสดงสูงสุด ${Number(p.maxObjects ?? 500) === 0 ? "ไม่จำกัด" : (p.maxObjects ?? 500)}` + mmTail("regionprops", p),
  // ใหม่ (2026-10-03, IP batch 2 ชิ้น D) — ตัวเลขผ่าน pnum (ค่าที่ op ใช้จริงหลัง clamp) · view เลือกด้วย own-property เหมือน registry
  feret: (p) => `Feret · พื้นที่ขั้นต่ำ ${pnum("feret", "minArea", p.minArea)}px · สูงสุด ${pnum("feret", "maxObjects", p.maxObjects, { int: true })} ชิ้น` + mmTail("feret", p),
  // ใหม่ (2026-10-14, IP batch 3) — ตัวเลขผ่าน pnum (ค่าหลัง clamp ที่ op ใช้จริง) · ช่วง lo–hi / พื้นที่แสดงจากน้อยไปมากเหมือนที่ op สลับให้
  quantize: (p) => {
    const n = pnum("quantize", "levels", p.levels, { int: true });
    return `${n} ระดับต่อช่อง (${quantizeBits(n)} bit)`;
  },
  shape_filter: (p) => {
    const op = "shape_filter";
    const prop = ownOr(SHAPE_FILTER_PROP_TH, p.prop, "circularity");
    let lo = pnum(op, "lo", p.lo), hi = pnum(op, "hi", p.hi);
    if (lo > hi) { const t = lo; lo = hi; hi = t; }
    let a = pnum(op, "minArea", p.minArea), b = pnum(op, "maxArea", p.maxArea);
    if (b > 0 && b < a) { const t = a; a = b; b = t; }
    const area = b > 0 ? `${a}–${b} px` : `≥ ${a} px`;
    const base = prop === "" ? `พื้นที่ ${area}` : `${prop} ${fixed2(lo)}–${fixed2(hi)} · ${area}`;
    return base + ownOr({ overlay: "", mask: " · Mask" }, p.view, "overlay");
  },
  line_profile: (p) => {
    const op = "line_profile";
    const base = `เส้น (${pnum(op, "x1", p.x1)}%,${pnum(op, "y1", p.y1)}%) → (${pnum(op, "x2", p.x2)}%,${pnum(op, "y2", p.y2)}%)`;
    return ownOr({ overlay: base, plot: `${base} · กราฟเต็มภาพ` }, p.view, "overlay") + mmTail(op, p);
  },
  watershed: () => "แยกวัตถุติดกัน",
  bounding_boxes: () => "ล้อมกรอบวัตถุ",
  hough_circles: (p) => `รัศมี ${p.minR ?? 0}–${p.maxR ?? 0}`,
  hough_lines: (p) => `โหวต ≥${p.threshold ?? 80} · ยาว≥${p.minLineLength ?? 40}px`,
  orb_keypoints: (p) => `สูงสุด ${p.maxFeatures ?? 300} จุด`,
  akaze_keypoints: (p) => `แสดงสูงสุด ${p.maxPoints ?? 300} จุด`,
  fast_keypoints: (p) => `threshold ${p.threshold ?? 10} · สูงสุด ${p.maxPoints ?? 300} จุด`,
  grabcut: () => "ตัดพื้นหลัง",
  region_growing: (p) => `จุดเริ่ม (${pnum("region_growing", "seedX", p.seedX)}%, ${pnum("region_growing", "seedY", p.seedY)}%) · ต่างได้ ${pnum("region_growing", "tolerance", p.tolerance, { int: true })}`,
  color_inrange: (p) => (p.colorSpace === "lab"
    ? `Lab a*${p.labAMin ?? 150}-${p.labAMax ?? 220} b*${p.labBMin ?? 130}-${p.labBMax ?? 200}`
    : "Hue: " + (p.hMin ?? 35) + "–" + (p.hMax ?? 85)),
  kmeans_segment: (p) => `K=${p.k ?? 4} กลุ่มสี`,
  white_balance: (p) => (
    p.method === "white_patch" ? "ปรับสมดุลสี (White Patch)"
    : p.method === "ref_point" ? `ปรับสมดุลสี (จุดอ้างอิง ${pnum("white_balance", "refX", p.refX)}%, ${pnum("white_balance", "refY", p.refY)}%)`
    : "ปรับสมดุลสีอัตโนมัติ (Gray World)"
  ),
  invert: () => "กลับสี",
  crop: (p) => `${p.w ?? 80}%×${p.h ?? 80}% @ (${p.x ?? 10}%,${p.y ?? 10}%)`,
  // interp: auto = ข้อความเดิมทุกตัวอักษร · เลือกเอง = ต่อท้ายชื่อวิธี (ไม่รู้จัก → ว่าง)
  resize: (p) => "สเกล: " + (p.scale ?? 50) + "%" + ownOr({ auto: "", nearest: " · Nearest", bilinear: " · Bilinear", bicubic: " · Bicubic" }, p.interp, "auto"),
  rotate: (p) => `มุม ${p.angle ?? 15}° · ${p.expand ? "ขยายผืนภาพ" : "คงขนาดเดิม"}` + ownOr({ auto: "", nearest: " · Nearest", bilinear: " · Bilinear", bicubic: " · Bicubic" }, p.interp, "auto"),
  flip: (p) => "แนว: " + (p.mode === "v" ? "แนวตั้ง" : p.mode === "both" ? "ทั้งสองแนว" : "แนวนอน"),
  perspective_warp: () => "บิด perspective ตาม 4 มุม",
  undistort: (p) => `k1=${p.k1 ?? -0.3} k2=${p.k2 ?? 0}`,
  translate_shear: (p) => {
    const op = "translate_shear";
    // shear ปัด 2 ตำแหน่งด้วย fixed2 (กติกาเดียวกับ info ของ op และสคริปต์ Python) แล้วตัดศูนย์ท้าย — ค่าที่สะสมจาก slider
    // step 0.05 (เช่น 0.15000000000000002) หรือ 1e-7 จากไฟล์ที่แก้มือไม่โชว์เศษ float · ปัดแล้วเป็น 0 ทั้งคู่ = ไม่เอียง (info ก็แสดง 0.00)
    const kx = Number(fixed2(pnum(op, "shearX", p.shearX))), ky = Number(fixed2(pnum(op, "shearY", p.shearY)));
    const base = `เลื่อน ${pnum(op, "tx", p.tx)}%, ${pnum(op, "ty", p.ty)}%`;
    return kx === 0 && ky === 0 ? base : `${base} · shear ${kx}, ${ky}`;
  },
  roi_polygon: (p) => `ROI ${pnum("roi_polygon", "points", p.points, { int: true })} จุด · ${ownOr({ apply: "ตัดนอกกรอบ", mask: "Mask", overlay: "ดูตำแหน่ง" }, p.view, "apply")}`,
  // ใหม่ (2026-10-04, IP batch 2 ชิ้น E) — ตัวเลขผ่าน pnum (ค่าหลัง clamp ที่ op ใช้จริง) · ตัวเลือกที่ไม่รู้จัก ("constructor" ฯลฯ) แสดงเป็นค่าเริ่มต้น
  slic: (p) => `~${pnum("slic", "count", p.count, { int: true })} ก้อน · compactness ${pnum("slic", "compactness", p.compactness)}${ownOr({ boundary: "", mean: " · สีเฉลี่ย" }, p.view, "boundary")}`,
  bandpass_notch: (p) => {
    const op = "bandpass_notch";
    const kind = ownOr({ bandpass: "bandpass", bandreject: "bandreject", notch: "notch" }, p.kind, "bandpass");
    const shape = ownOr({ gaussian: "Gaussian", ideal: "Ideal" }, p.shape, "gaussian");
    const tail = ownOr({ result: "", spectrum: " · สเปกตรัม" }, p.view, "result");
    if (kind === "notch") return `Notch (${signedText(pnum(op, "nx", p.nx))}%, ${signedText(pnum(op, "ny", p.ny))}%) r ${pnum(op, "nr", p.nr)}%${tail}`;
    const a = pnum(op, "rIn", p.rIn), b = pnum(op, "rOut", p.rOut);
    return `${kind === "bandreject" ? "Band-reject" : "Band-pass"} ${Math.min(a, b)}–${Math.max(a, b)}% · ${shape}${tail}`;
  },
  fft_magnitude: () => "แสดง spectrum ความถี่",
  fft_filter: (p) => `${p.type === "high" ? "High" : "Low"}-pass · r=${p.cutoff ?? 30} หน่วยความถี่ (${p.shape === "butterworth" ? `butterworth n=${pnum("fft_filter", "order", p.order, { int: true })}` : (p.shape ?? "gaussian")})`,
  imabsdiff: () => "|A − B|",
  imfuse: (p) => (p.mode === "checkerboard" ? `Checkerboard · tile=${p.tileSize ?? 32}px` : `Blend · A ${p.alpha ?? 50}%`),
  mask: (p) => (p.invert ? "Mask B (กลับด้าน)" : "Mask ผ่าน B"),
  // ใหม่ (2026-10-02, IP batch 2 ชิ้น B) — ค่า operation ที่ไม่รู้จัก (รวม "constructor" ฯลฯ) แสดงเป็นค่าเริ่มต้น
  // ตรงกับที่ op ใช้จริง (registry เลือกด้วย Object.hasOwn เหมือนกัน)
  imarith: (p) => ownOr(IMARITH_LABEL, p.operation, "add"),
  logical_ops: (p) => ownOr(LOGICAL_LABEL, p.operation, "and"),
  inpaint: (p) => `${p.method === "ns" ? "Navier-Stokes" : "Telea"} · r=${pnum("inpaint", "radius", p.radius, { int: true })}`,
  hist_match: () => "ปรับ histogram ของ A ตาม B",
  template_matching: (p) => `${p.method === "ccorr" ? "Cross-Corr" : "Corr Coeff"} · เกณฑ์ ${p.threshold ?? 70}%`,
  psnr: () => "เทียบคุณภาพ A vs B (dB)",
  ssim: () => "เทียบโครงสร้าง A vs B",
  // ใหม่ (2026-10-15, IP batch 4A) — ตัวเลขผ่าน pnum (ค่าหลัง clamp ที่ op ใช้จริง) · ตัวเลือกที่ไม่รู้จักแสดงเป็นค่าเริ่มต้นเหมือน registry
  guided_filter: (p) => `r=${pnum("guided_filter", "radius", p.radius, { int: true })} · ε ${pnum("guided_filter", "eps", p.eps)}`,
  focus_score: (p) => `${p.method === "tenengrad" ? "Tenengrad" : "Laplacian var"} · กลาง ${pnum("focus_score", "roiPct", p.roiPct)}%`,
  pseudocolor: (p) => `${ownOr({ jet: "jet", hot: "hot", cool: "cool" }, p.map, "jet")} · ${p.range === "minmax" ? "ยืดช่วง" : "0–255"}`,
  delta_e: (p) => `${p.formula === "ciede2000" ? "CIEDE2000" : "CIE76"} · ${p.compare === "mean_b" ? "เทียบสีเฉลี่ย B" : "ทีละพิกเซล"} · เกณฑ์ ${pnum("delta_e", "tol", p.tol)}`,
  roi_apply: (p) => {
    const op = "roi_apply";
    return `${p.shape === "ellipse" ? "วงรี" : "สี่เหลี่ยม"} ${pnum(op, "x", p.x)}%,${pnum(op, "y", p.y)}% ${pnum(op, "w", p.w)}×${pnum(op, "h", p.h)}%`
      + (p.invert === true ? " · กลับด้าน" : "")
      + (pnum(op, "feather", p.feather, { int: true }) > 0 ? ` · ขอบนุ่ม ${pnum(op, "feather", p.feather, { int: true })}px` : "");
  },
  // Tier 2, 18d — Motion (ค่าเริ่มต้น/ช่วง อ่านจาก OPS ผ่าน pnum: history 10-2000 def 500, varThreshold 4-100
  // def 16, maxMotion 1-50 def 10, maxPoints 10-300 def 100, trailLength 1-50 def 10 · int = ตัวที่ registry
  // ใช้ intIn (ปัดเป็นจำนวนเต็ม) ตรงตามตารางของเทสต์ tier2d-MN-1)
  mog2: (p) => `${p.view === "overlay" ? "ซ้อนสีบนภาพ" : "Mask ขาว-ดำ"} · จำ ${pnum("mog2", "history", p.history, { int: true })} เฟรม · ไว ${pnum("mog2", "varThreshold", p.varThreshold)}`,
  farneback_optical_flow: (p) => `${p.view === "arrows" ? "ลูกศร" : "วงล้อสี"} · สว่างสุด ${pnum("farneback_optical_flow", "maxMotion", p.maxMotion)} px/เฟรม`,
  pyrlk_optical_flow: (p) => `${pnum("pyrlk_optical_flow", "maxPoints", p.maxPoints, { int: true })} จุด · เส้นทาง ${pnum("pyrlk_optical_flow", "trailLength", p.trailLength, { int: true })} เฟรม`,
};

// ดึง op id ปัจจุบันของ node จาก properties (merged group มี "mode", กลุ่มเดี่ยว เช่น morphology ไม่มี)
export function currentOpId(filterId, properties) {
  return (properties && properties.mode) || filterId;
}

// ตัด prefix "opId__" ออกจาก sub-params ของ merged group → คืน local params (key เดิมของ op)
// กลุ่มที่ไม่มี "mode" (เช่น morphology) → properties เป็น local params อยู่แล้ว ส่งตรง ๆ
export function localParamsFor(filterId, properties) {
  if (!properties) return {};
  if (properties.mode == null) return properties;
  const prefix = properties.mode + "__";
  const local = {};
  for (const k in properties) {
    if (k.startsWith(prefix)) local[k.slice(prefix.length)] = properties[k];
  }
  return local;
}

// สีคอลัมน์ hex + ความโปร่งใส (ใช้วาด icon chip/border บน node card)
export function hexA(hex, a) {
  if (!hex || hex[0] !== "#") return hex;
  const h = hex.slice(1);
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

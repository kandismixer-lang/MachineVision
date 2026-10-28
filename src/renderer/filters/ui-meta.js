// UI metadata — เพิ่มความสวย/สื่อความ โดยไม่แตะ backend algorithm
import { OPS, MORPH_OP_IDS, passFailParams, pfNum, boolOr, fixed2, signedText, quantizeBits, SHAPE_FILTER_PROP_TH, HITMISS_PATTERNS } from "./registry.js";

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
  image_registration: "🧭", // 2026-10-24 แทนหมวด Template Matching · เช็คแล้วไม่ชนกับ glyph ใน GROUP_ICON/OP_ICON
  // ใหม่ (2026-10-18) — Morphology แยกเป็นเครื่องมือรายตัว · เช็คแล้วไม่ชนกับ glyph ใน GROUP_ICON/OP_ICON/src ใด ๆ (morphology 🔲 เดิมคงไว้เป็น legacy)
  morph_tools: "⬓",
};

export const OP_ICON = {
  grayscale: "◐", colorspace: "🎨", channel_split: "◉", rgb_adjust: "🎚", color_inrange: "🌈", invert: "◑",
  kmeans_segment: "⬡", white_balance: "⚖",
  brightness_contrast: "☀", gamma: "γ", hist_equalize: "▥", clahe: "▦", unsharp: "✦", laplacian_sharpen: "◇",
  custom_kernel: "◫",
  // ใหม่ (2026-10-01, IP batch ENHANCE/MORPHOLOGY) — เช็คแล้วไม่ชนกับ glyph ใดในไฟล์นี้หรือ GROUP_ICON
  contrast_stretch: "↔", // ลูกศรซ้าย-ขวา สื่อ "ยืดช่วง"
  flatfield_correction: "🔆", // ดวงอาทิตย์มีรัศมี สื่อ "แก้แสง"
  // ใหม่ (2026-10-14, IP batch 3) — เช็คแล้วไม่ชนกับ OP_ICON / GROUP_ICON / NC_ICON_FALLBACK
  quantize: "≣", // แถบซ้อนกัน สื่อ "ระดับสี"
  shape_filter: "⛛", // กรวยกรอง สื่อ "คัดวัตถุ"
  crop: "⛝", resize: "⤢", rotate: "↻", flip: "⇋", perspective_warp: "⏢", undistort: "⊙",
  salt_pepper_noise: "⁙", median_blur: "●", gaussian_blur: "✺", box_blur: "▧", bilateral: "◎",
  binary_threshold: "◩", otsu_threshold: "◧", adaptive_threshold: "◨", multi_otsu_threshold: "◪",
  canny: "⌁", sobel: "∿", laplacian_edge: "◈", scharr_edge: "∾",
  watershed: "≋", grabcut: "✂", find_contours: "⌬", connected_components: "⠿", distance_transform: "◔",
  // ใหม่ (2026-09-30) — เช็คทุก glyph กับของเดิมทั้งไฟล์และกับ GROUP_ICON ใน ui-meta.js แล้วไม่ชนกันเลย
  prewitt_edge: "⊿", // สามเหลี่ยมมุมฉาก สื่อ "ความชัน" แบบตรงไปตรงมา
  roberts_edge: "⋱", // จุดเรียงแนวทแยง สื่อ kernel 2x2 ตามแนวทแยง
  log_edge: "⊚", // วงกลมซ้อน สื่อ "หมวกเม็กซิกัน" ของ LoG
  region_growing: "🪄", // ไม้กายสิทธิ์ = Magic Wand ตรงตัว
  local_std_filter: "▒", local_entropy_filter: "▓",
  // ใหม่ (Tier 2, 18c) — เช็คแล้วว่าไม่ชนกับ glyph ใดข้างบนทั้งหมด (~59 ตัว), 📏 (ไม้บรรทัด) สื่อ "การวัดค่า"
  regionprops: "📏",
  bounding_boxes: "□", hough_circles: "○", hough_lines: "╱", harris_corners: "⌖", shi_tomasi_corners: "✛",
  orb_keypoints: "✧", akaze_keypoints: "◆", fast_keypoints: "✪",
  fft_magnitude: "▨", fft_filter: "⬒",
  imabsdiff: "Δ", // ใหม่ (Tier 2, 18a) — เช็คแล้วว่าไม่ชนกับ glyph ใดข้างบน, Δ สื่อ "ความต่าง" ตรงตัว
  // ใหม่ (Tier 2, 18b) — เช็คทุกตัวกับ glyph เดิมทั้งหมดด้านบน (~57 ตัว) แล้วไม่ชนกันเลย:
  imfuse: "⊕", // วงกลม+บวก สื่อ "รวม/ผสม" สองภาพเข้าด้วยกัน
  mask: "⬚", // สี่เหลี่ยมประ สื่อ "หน้ากาก/พื้นที่ตัดออก"
  template_matching: "⌕", // แว่นขยาย-สแกน สื่อ "ค้นหา" — จงใจไม่ใช้ 🎯 (ชนกับ GROUP_ICON.object_detection ในไฟล์ ui-meta.js)
  psnr: "📶", // สัญญาณ สื่อ "signal-to-noise"
  ssim: "≈", // ประมาณเท่ากัน สื่อ "ความคล้าย/โครงสร้างใกล้เคียงกัน"
  // ใหม่ (2026-10-02, IP batch 2 ชิ้น B) — เช็คทุกตัวกับ glyph เดิมทั้งไฟล์และ GROUP_ICON ใน ui-meta.js แล้วไม่ชน
  // (± ของ GROUP_ICON.image_arithmetic จงใจไม่ใช้ซ้ำ)
  imarith: "÷", // เครื่องหมายหาร สื่อ "บวก/ลบ/คูณ/หาร"
  logical_ops: "∧", // AND ของตรรกศาสตร์
  inpaint: "🩹", // พลาสเตอร์ สื่อ "ซ่อมรอย"
  hist_match: "📈", // เส้นกราฟขาขึ้น สื่อ "CDF สะสม" — ไม่ใช้ 📊 เพราะ app.js/detail-panel.js ใช้นำหน้าข้อความผลลัพธ์ (debugger 2026-10-03)
  // ใหม่ (2026-10-03, IP batch 2 ชิ้น C) — เช็คทุกตัวกับ glyph เดิมทั้งไฟล์, GROUP_ICON ใน ui-meta.js และ app.js/detail-panel.js แล้วไม่ชน
  gaussian_noise: "░", // แรเงาเม็ดละเอียด สื่อ "เกรนทั่วภาพ" (▒/▓ ของ local_std/entropy เข้มกว่า แยกด้วยตาได้)
  wiener_filter: "◍", // วงกลมมีเส้นแนวตั้งด้านใน สื่อ "เกลี่ยแบบเลือกบริเวณ" (ต่างจาก ◉/◎/● ของกลุ่ม blur)
  translate_shear: "▱", // สี่เหลี่ยมด้านขนาน = รูปทรงของสี่เหลี่ยมที่ถูก shear ตรงตัว
  // ใหม่ (2026-10-03, IP batch 2 ชิ้น D) — เช็คทุกตัวกับ glyph เดิมทั้งไฟล์, GROUP_ICON และ app.js/detail-panel.js แล้วไม่ชน
  feret: "⌀", // สัญลักษณ์เส้นผ่านศูนย์กลาง สื่อ "วัดความกว้างแบบคาลิปเปอร์"
  line_profile: "〽", // เส้นกราฟขึ้นลง สื่อ "กราฟความสว่างตามเส้น" (ไม่ใช่ 〰 ของ GROUP_ICON.frequency)
  roi_polygon: "⬠", // ห้าเหลี่ยม สื่อ "บริเวณหลายเหลี่ยม" (ต่างจาก ⏢ perspective_warp / ⬡ kmeans)
  // ใหม่ (2026-10-04, IP batch 2 ชิ้น E) — เช็คแล้วไม่ชนกับ OP_ICON / GROUP_ICON / app.js / detail-panel.js / editor.js (🧩 ถูกใช้เป็น fallback ใน editor.js จึงไม่เลือก)
  slic: "⊞", // ตารางสี่เหลี่ยมแบ่งช่อง สื่อ "แบ่งภาพเป็นก้อนตามตาราง"
  bandpass_notch: "◙", // วงกลมซ้อนในกรอบ สื่อ "วงแหวนความถี่ที่เลือกเก็บ/ตัด"
  // ใหม่ (2026-10-15, IP batch 4A) — เช็คแล้วไม่ชนกับ OP_ICON / GROUP_ICON / NC_ICON_FALLBACK (🧩) / 📊 / 📌
  guided_filter: "≀", pseudocolor: "🌡", focus_score: "⦿", delta_e: "◬", roi_apply: "⊡",
  // ใหม่ (2026-10-15, IP batch 4B) — เช็คแล้วไม่ชนกับ OP_ICON / GROUP_ICON / NC_ICON_FALLBACK (🧩) / 📊 / 📌
  deconv_wiener: "⟲", deconv_lucy: "⌾", ecc_register: "⧉", gabor_filter: "⋈", glcm: "⧈", nlm_denoise: "✣", dehaze: "🌫", dct_jpeg: "⬗",
  // ใหม่ (2026-10-17, P-8) — ✔ สื่อ "ผ่าน" · เช็คแล้วไม่ชนกับ OP_ICON / GROUP_ICON / NC_ICON_FALLBACK (🧩)
  pass_fail: "✔",
  // ใหม่ (2026-10-18, Morphology แยกเครื่องมือ) — เช็คแล้วไม่ชนกับ OP_ICON / GROUP_ICON / NC_ICON_FALLBACK (🧩) / 📊 / 📌 และไม่มีใน src ที่อื่น
  // (morphology "▣" เดิมถูกเอาออก: ตัว legacy ไม่อยู่บน palette แล้ว)
  morph_erode: "⊖", morph_dilate: "◼", morph_open: "◌", morph_close: "▣", morph_gradient: "◭", morph_tophat: "⌃", morph_blackhat: "⌄",
  morph_fill_holes: "⬤", morph_clear_border: "⊠", morph_remove_small: "∴",
  morph_skeleton: "🦴", morph_reconstruct_open: "⥀", morph_convex_hull: "⬣", morph_endpoints_branchpoints: "⋔", morph_hit_or_miss: "⊘",
  morph_regional_max: "▲", morph_regional_min: "▽", morph_h_maxima: "△", morph_extended_max: "⌂",
  // ใหม่ (Tier 2, 18d) — เช็คทุกตัวกับ glyph เดิมทั้งหมดด้านบน และกับ GROUP_ICON ใน ui-meta.js แล้วไม่ชนกันเลย
  mog2: "🎞", // ฟิล์มหลายเฟรม สื่อ "ต้องดูหลายเฟรมต่อเนื่อง"
  farneback_optical_flow: "⇶", // ลูกศรสามเส้นขนานกัน สื่อ "การไหลทั้งภาพ" (dense flow)
  pyrlk_optical_flow: "⤳", // ลูกศรหางคลื่น สื่อ "เส้นทางของจุดที่ถูกติดตาม" (sparse flow)
};

// ชื่อสั้นบน tile/การ์ด: ตัดส่วนในวงเล็บออก (กติกาเดียวกับป้ายชื่อบนกระดาน) — ชื่อเต็มอยู่ใน tooltip
export function shortToolName(name) {
  const raw = String(name == null ? "" : name);
  return raw.replace(/\s*\([^)]*\)\s*/g, " ").trim() || raw;
}

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

// PSF บนการ์ดของ Wiener / Lucy (IP batch 4B): "motion 9px 0°" หรือ "disk r9" — ชนิดเลือกด้วย own-property เหมือน registry
function psfCardText(p) {
  const size = pnum("deconv_wiener", "psfSize", p.psfSize, { int: true });
  return p.psf === "disk" ? `disk r${size}` : `motion ${size}px ${pnum("deconv_wiener", "psfAngle", p.psfAngle, { int: true })}°`;
}

// ข้อความการ์ดของ op ที่มีตัวเลือก operation (IP batch 2 ชิ้น B)
const IMARITH_LABEL = { add: "A + B", subtract: "A − B", multiply: "A × B ÷ 255", divide: "A ÷ B × 255" };
const LOGICAL_LABEL = { and: "A AND B", or: "A OR B", xor: "A XOR B", a_and_not_b: "A AND NOT B" };
// ชื่อสั้นของ metric ของ pass_fail บนการ์ด (key ตรงกับ PASS_FAIL_METRICS ใน registry.js)
const PASS_FAIL_SHORT = { count: "จำนวนวัตถุ", area_sum: "พื้นที่รวม", area_max: "พื้นที่ใหญ่สุด", roi_mean: "สว่างใน ROI", focus: "ความคม", de_mean: "ΔE เฉลี่ย" };
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
    // ใหม่ (2026-10-15, IP batch 4B) — แม่แบบเลือกด้วย own-property เหมือน registry (ค่าแปลก → จุดเดี่ยว)
    : p.op === "hit_or_miss" ? `Hit-or-Miss · ${ownOr(HITMISS_PATTERNS, p.hmPattern, "isolated").th}`
    : p.op === "regional_max" ? "Regional Maxima"
    : p.op === "regional_min" ? "Regional Minima"
    : p.op === "h_maxima" ? `H-maxima h=${pnum("morphology", "hHeight", p.hHeight, { int: true })}`
    : p.op === "extended_max" ? `Extended Maxima h=${pnum("morphology", "hHeight", p.hHeight, { int: true })}`
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
  rotate: (p) => `มุม ${p.angle ?? 15}° · ${boolOr(p.expand, false) ? "ขยายผืนภาพ" : "คงขนาดเดิม"}` + ownOr({ auto: "", nearest: " · Nearest", bilinear: " · Bilinear", bicubic: " · Bicubic" }, p.interp, "auto"),
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
  mask: (p) => (boolOr(p.invert, false) ? "Mask B (กลับด้าน)" : "Mask ผ่าน B"),
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
      + (boolOr(p.invert, false) ? " · กลับด้าน" : "")
      + (pnum(op, "feather", p.feather, { int: true }) > 0 ? ` · ขอบนุ่ม ${pnum(op, "feather", p.feather, { int: true })}px` : "");
  },
  // ใหม่ (2026-10-15, IP batch 4B) — ตัวเลขผ่าน pnum (ค่าหลัง clamp ที่ op ใช้จริง) · ตัวเลือกที่ไม่รู้จักแสดงเป็นค่าเริ่มต้นเหมือน registry
  deconv_wiener: (p) => `${psfCardText(p)} · NSR ${pnum("deconv_wiener", "nsr", p.nsr)}`,
  deconv_lucy: (p) => `${pnum("deconv_lucy", "iterations", p.iterations, { int: true })} รอบ · ${psfCardText(p)}`,
  ecc_register: (p) => `${ownOr({ translation: "เลื่อนอย่างเดียว", euclidean: "เลื่อน + หมุน", affine: "เลื่อน + หมุน + ย่อขยาย + เฉือน", homography: "มุมมองเปลี่ยน" }, p.motion, "euclidean")} · ≤ ${pnum("ecc_register", "iterations", p.iterations, { int: true })} รอบ · ย่อ ${pnum("ecc_register", "maxSide", p.maxSide, { int: true })}px`,
  gabor_filter: (p) => `λ ${pnum("gabor_filter", "wavelength", p.wavelength, { int: true })}px · θ ${pnum("gabor_filter", "orientation", p.orientation, { int: true })}°`,
  glcm: (p) => `${ownOr({ 2: "2", 4: "4", 8: "8", 16: "16", 32: "32", 64: "64" }, p.levels, "8")} ระดับ · d=${pnum("glcm", "distance", p.distance, { int: true })} · ${ownOr({ 0: "0°", 45: "45°", 90: "90°", 135: "135°", avg: "เฉลี่ย 4 ทิศ" }, p.angle, "0")}`,
  nlm_denoise: (p) => `h ${pnum("nlm_denoise", "h", p.h, { int: true })} · แผ่น ${ownOr({ 3: "3", 5: "5", 7: "7" }, p.patch, "5")} · ค้นหา ${ownOr({ 7: "7", 11: "11", 15: "15" }, p.search, "11")}`,
  dehaze: (p) => {
    const ps0 = pnum("dehaze", "patch", p.patch, { int: true });
    return `ω ${pnum("dehaze", "amount", p.amount)} · หน้าต่าง ${ps0 % 2 === 0 ? ps0 + 1 : ps0}` + (!boolOr(p.refine, true) ? " · ไม่เกลี่ย" : "");
  },
  dct_jpeg: (p) => (p.method === "zonal" ? `ความถี่ต่ำ ${pnum("dct_jpeg", "keep", p.keep, { int: true })}×${pnum("dct_jpeg", "keep", p.keep, { int: true })}` : `JPEG q${pnum("dct_jpeg", "quality", p.quality, { int: true })}`),
  // ใหม่ (2026-10-17, P-8) — ค่าหลัง clamp/own-property ผ่าน passFailParams เหมือน op
  pass_fail: (p) => {
    const q = passFailParams(p);
    const crit = q.cond === "range" ? `${pfNum(q.lo)}–${pfNum(q.hi)}` : q.cond === "ge" ? `≥ ${pfNum(q.lo)}` : q.cond === "le" ? `≤ ${pfNum(q.hi)}` : `= ${pfNum(q.lo)}`;
    return `${PASS_FAIL_SHORT[q.metric]} ผ่านเมื่อ ${crit}`;
  },
  // Tier 2, 18d — Motion (ค่าเริ่มต้น/ช่วง อ่านจาก OPS ผ่าน pnum: history 10-2000 def 500, varThreshold 4-100
  // def 16, maxMotion 1-50 def 10, maxPoints 10-300 def 100, trailLength 1-50 def 10 · int = ตัวที่ registry
  // ใช้ intIn (ปัดเป็นจำนวนเต็ม) ตรงตามตารางของเทสต์ tier2d-MN-1)
  mog2: (p) => `${p.view === "overlay" ? "ซ้อนสีบนภาพ" : "Mask ขาว-ดำ"} · จำ ${pnum("mog2", "history", p.history, { int: true })} เฟรม · ไว ${pnum("mog2", "varThreshold", p.varThreshold)}`,
  farneback_optical_flow: (p) => `${p.view === "arrows" ? "ลูกศร" : "วงล้อสี"} · สว่างสุด ${pnum("farneback_optical_flow", "maxMotion", p.maxMotion)} px/เฟรม`,
  pyrlk_optical_flow: (p) => `${pnum("pyrlk_optical_flow", "maxPoints", p.maxPoints, { int: true })} จุด · เส้นทาง ${pnum("pyrlk_optical_flow", "trailLength", p.trailLength, { int: true })} เฟรม`,
};

// Morphology แยกเครื่องมือ (2026-10-18): การ์ดของ morph_<op> ใช้ข้อความเดียวกับ morphology เดิมที่เลือก op นั้น (เรียกตัวเดิมด้วย op ตายตัว)
for (const id of MORPH_OP_IDS) {
  const op = id.slice("morph_".length);
  OP_PRIMARY[id] = (p) => OP_PRIMARY.morphology({ ...p, op });
}

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

// สีตัวอักษร/ไอคอนที่อ่านออกบนพื้น hex นั้น (ขาวบนสีเข้ม · เกือบดำบนสีสว่าง เช่น เหลือง/เขียวอ่อน) — ใช้กับกล่องไอคอนแบบ KNIME
// (palette tile + การ์ดบนกระดาน) · อิงความสว่างสัมพัทธ์ WCAG; hex ผิดรูปแบบ → ขาว
const contrastCache = new Map(); // สีหมวดมีไม่กี่ค่า → คำนวณครั้งเดียว ไม่ทำซ้ำทุกเฟรม
export function contrastOn(hex) {
  const cached = contrastCache.get(hex);
  if (cached) return cached;
  if (!hex || hex[0] !== "#") { contrastCache.set(hex, "#ffffff"); return "#ffffff"; }
  const h = hex.slice(1);
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const ch = [0, 2, 4].map((i) => {
    const v = parseInt(full.slice(i, i + 2), 16) / 255;
    return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  });
  const L = 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2];
  const color = L > 0.4 ? "#1a1a1a" : "#ffffff";
  contrastCache.set(hex, color);
  return color;
}

// ---------------------------------------------------------------------------
// OPTION_GROUPS — จัด dropdown ยาว ๆ เป็นหัวข้อ <optgroup> ตาม "เป้าหมายงาน" (ภาษาไทย) ช่วยมือใหม่เลือกวิธีถูก
// key: "<filterId>.mode" (select "เลือกวิธี" ของ node หมวด) | "<opId>.<localKey>" (select ภายใน op เช่น morphology.op)
// value: [[หัวข้อ, [value, ...]], ...] — ลำดับ value ในกลุ่ม = ลำดับที่แสดง · ตารางนี้ "ไม่แก้" options ใน registry (ค่า/ลำดับเดิมทุกตัว)
// value ที่ไม่อยู่ในตาราง (เช่น op ใหม่ที่ลืมใส่) → ไปกลุ่ม "อื่น ๆ" ท้ายสุด (และเทสต์ param-visibility ล้ม ให้รู้ว่าต้องเพิ่ม)
// ---------------------------------------------------------------------------
export const OPTION_GROUPS_OTHER = "อื่น ๆ";
export const OPTION_GROUPS = {
  // Morphology แยกเครื่องมือ (2026-10-18) — 4 กลุ่ม ลำดับกลุ่ม/ตัวเลือก = ลำดับใน dropdown จริง (morph_tools)
  "morph_tools.mode": [
    ["พื้นฐาน", ["morph_erode", "morph_dilate", "morph_open", "morph_close", "morph_gradient", "morph_tophat", "morph_blackhat"]],
    ["ทำความสะอาด", ["morph_fill_holes", "morph_clear_border", "morph_remove_small"]],
    ["รูปร่าง", ["morph_skeleton", "morph_reconstruct_open", "morph_convex_hull", "morph_endpoints_branchpoints", "morph_hit_or_miss"]],
    ["ยอด/หุบ", ["morph_regional_max", "morph_regional_min", "morph_h_maxima", "morph_extended_max"]],
  ],
  "morphology.op": [
    ["พื้นฐาน — กัด / พอก", ["erode", "dilate", "open", "close"]],
    ["ดึงขอบ / ส่วนสว่าง-มืดเฉพาะที่", ["gradient", "tophat", "blackhat"]],
    ["เก็บกวาดวัตถุขาว-ดำ", ["remove_small", "fill_holes", "clear_border", "reconstruct_open", "convex_hull"]],
    ["เส้นแกนกลางและลายเฉพาะ", ["skeleton", "endpoints_branchpoints", "hit_or_miss"]],
    ["หายอด / หุบ (ทำ marker ให้ Watershed)", ["regional_max", "regional_min", "h_maxima", "extended_max"]],
  ],
  "analysis_detect.mode": [
    ["วัตถุ — หา / นับ / วัด", ["find_contours", "connected_components", "bounding_boxes", "regionprops", "shape_filter", "feret", "distance_transform", "line_profile"]],
    ["หาเส้นตรง / วงกลม", ["hough_lines", "hough_circles"]],
    ["หาจุดมุม / จุดเด่น", ["harris_corners", "shi_tomasi_corners", "fast_keypoints", "orb_keypoints", "akaze_keypoints"]],
    ["ลายพื้นผิวและความคม", ["local_std_filter", "local_entropy_filter", "gabor_filter", "glcm", "focus_score"]],
    ["ตัดสินผ่าน / ไม่ผ่าน", ["pass_fail"]],
  ],
  "segment_edge.mode": [
    ["แยกขาว-ดำ (Threshold)", ["binary_threshold", "otsu_threshold", "adaptive_threshold", "multi_otsu_threshold"]],
    ["หาขอบ (Edge)", ["canny", "sobel", "scharr_edge", "prewitt_edge", "roberts_edge", "laplacian_edge", "log_edge"]],
    ["แยกวัตถุเป็นชิ้น", ["watershed", "grabcut", "region_growing"]],
  ],
  "enhance_tools.mode": [
    ["ความสว่าง / คอนทราสต์", ["brightness_contrast", "gamma", "contrast_stretch", "hist_equalize", "clahe", "flatfield_correction"]],
    ["ความคม", ["unsharp", "laplacian_sharpen", "custom_kernel"]],
    ["งานเฉพาะทาง", ["dehaze", "quantize"]],
  ],
  "filter_denoise.mode": [
    ["ทำภาพเสียเพื่อทดลอง (ใส่ noise)", ["salt_pepper_noise", "gaussian_noise"]],
    ["ลด noise / เบลอ", ["median_blur", "gaussian_blur", "box_blur", "bilateral", "guided_filter", "nlm_denoise", "wiener_filter"]],
    ["แก้ภาพเบลอ (รู้ PSF)", ["deconv_wiener", "deconv_lucy"]],
  ],
  "color_tools.mode": [
    ["แปลง / ดูช่องสี", ["grayscale", "invert", "colorspace", "channel_split", "pseudocolor"]],
    ["ปรับสี", ["rgb_adjust", "white_balance"]],
    ["เลือก / แบ่งตามสี", ["color_inrange", "kmeans_segment", "slic"]],
  ],
  "image_arithmetic.mode": [
    ["เทียบ / รวมสองภาพ", ["imabsdiff", "imfuse", "imarith", "logical_ops"]],
    ["ใช้ B เป็นหน้ากาก / บริเวณ", ["mask", "roi_apply", "inpaint"]],
    ["ปรับ A ให้ตรงกับ B", ["hist_match"]],
  ],
  "custom_kernel.preset": [
    ["ไม่เปลี่ยนภาพ (เทียบ)", ["identity"]],
    ["คมขึ้น / หาขอบ", ["sharpen", "edge", "outline", "emboss"]],
    ["เบลอเฉลี่ย", ["box3", "box5"]],
    ["จำลองภาพเบลอ (PSF)", ["motion", "disk"]],
  ],
  "colorspace.space": [
    ["ใช้บ่อยในงานตรวจจับ", ["hsv", "lab", "ycrcb", "hls"]],
    ["มาตรฐาน / ทฤษฎี (ค่า 8 bit ถูกบีบ)", ["xyz", "luv", "yuv"]],
  ],
  "channel_split.channel": [
    ["RGB", ["r", "g", "b"]],
    ["HSV", ["h", "s", "v"]],
  ],
};

// คืน [[หัวข้อ, [option object, ...]], ...] สำหรับ select ตัวนี้ (option object = ตัวเดียวกับใน param.options)
// ไม่มีตารางของ select นี้ → null (วาดแบบเรียงเรียบเหมือนเดิม) · value ที่ตารางไม่รู้จัก → กลุ่ม "อื่น ๆ" ท้ายสุด
// filterId = id ของ node (FILTERS key) · param.key = "mode" | "<opId>__<localKey>" (merged) | "<localKey>" (node เดี่ยว เช่น morphology)
export function optionGroupsFor(filterId, param) {
  let tableKey;
  if (param.key === "mode") tableKey = filterId + ".mode";
  else if (param.key.includes("__")) tableKey = param.key.replace("__", ".");
  else tableKey = filterId + "." + param.key;
  const table = Object.hasOwn(OPTION_GROUPS, tableKey) ? OPTION_GROUPS[tableKey] : null;
  if (!table) return null;
  const byValue = new Map(param.options.map((o) => [o.value, o]));
  const used = new Set();
  const out = [];
  for (const [title, values] of table) {
    const opts = [];
    for (const v of values) {
      if (byValue.has(v) && !used.has(v)) { opts.push(byValue.get(v)); used.add(v); }
    }
    if (opts.length) out.push([title, opts]);
  }
  const rest = param.options.filter((o) => !used.has(o.value));
  if (rest.length) out.push([OPTION_GROUPS_OTHER, rest]);
  return out;
}

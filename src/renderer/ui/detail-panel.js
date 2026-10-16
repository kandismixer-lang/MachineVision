// Panel ขวา: รายละเอียดของ filter ที่เลือก + คำอธิบายการทำงาน + สไลเดอร์ปรับค่า
import { FILTERS, defaultParams } from "../filters/registry.js";

// ---------- helper แสดงผลล้วน (Phase 2.2) — ไม่ยุ่งกับ state ของบล็อกค่า ----------

// แยกข้อความ note-info เป็น readout tile: "ค่าเกณฑ์ที่ Otsu คำนวณได้ = 137" | "พบวัตถุ 5 ชิ้น"
// → { label, value, unit } · ไม่เข้ารูปแบบ/ไม่มีตัวเลข/ไม่มีป้าย → null (ผู้เรียกแสดงเป็นแถวข้อความธรรมดาเหมือนเดิม)
export function splitReadout(t) {
  const m = String(t == null ? "" : t).trim().match(/^(.*?)\s*[=:]\s*(-?[\d.,]+)\s*(.*)$/) ||
    String(t == null ? "" : t).trim().match(/^(.*?)\s(-?[\d.,]+)\s*(\S*)$/);
  // value ต้องมีตัวเลข · ป้ายห้ามว่าง
  // ข้อความที่มีเลขหลายตัว (regex ข้อ 2 จะจับเลขตัวสุดท้าย → tile โชว์เลขผิด) → null เสมอ:
  //   · ป้ายมีตัวเลข (เช่น "ลบวัตถุเล็ก 3 ชิ้น (เหลือ 5 ชิ้น)") · "หน่วย" มีตัวเลข หรืออักขระ ( ) / · = — – (เช่น "ชิ้น)", "px²", "= 12, 200")
  if (!m || !m[1].trim() || !/\d/.test(m[2])) return null;
  if (/\d/.test(m[1]) || /[\d()/·=—–]/.test(m[3])) return null;
  return { label: m[1].trim(), value: m[2], unit: m[3].trim() };
}

// select ที่ <= 3 ตัวเลือกและชื่อสั้นทุกตัว → segmented control · "mode" และที่เหลือ → <select> (spec §6.2)
export const SEG_MAX_OPTIONS = 3;
export const SEG_MAX_LABEL = 12; // ตัวอักษรต่อป้าย — 3 ปุ่มในแผง ~280px ต้องพอดี (ตอนนี้เข้าเงื่อนไข: ทิศ Sobel/Scharr/Prewitt)
export function selectStyle(param) {
  const opts = (param && param.options) || [];
  if (param && param.key === "mode") return "select";
  if (opts.length === 0 || opts.length > SEG_MAX_OPTIONS) return "select";
  return opts.every((o) => String(o.label).length <= SEG_MAX_LABEL) ? "seg" : "select";
}

// ค่าต่างจาก default หรือไม่ (ใช้โชว์ปุ่ม ↺) — ทน float (step 0.1) และ default ที่ไม่มี
export function differsFromDefault(value, def) {
  if (def === undefined || def === null) return false;
  const a = Number(value), b = Number(def);
  if (Number.isNaN(a) || Number.isNaN(b)) return value !== def;
  return Math.abs(a - b) > 1e-9;
}

// ===================================================================
// U3 — บล็อก "ค่าที่วัดได้" (อ่าน ctx.values ที่ cv-worker ส่งมาเป็น note)
// รูปทรงข้อมูลมี 2 แบบ (Tier 2 §A.2):
//   { kind: "scalars", fields: [{ key, label, value }] }   ← label เป็นไทยมาจาก worker แล้ว
//   { kind: "table", columns: ["area",…], rows: [[…]], total? } ← columns เป็น key ดิบ ต้องแปลชื่อที่นี่
// ===================================================================
const PAINT_MS = 200;    // ทาสีค่าไม่เกิน 5 ครั้ง/วินาที (อ่านทัน + งานน้อย)
const ROWS_DEFAULT = 20; // แถวที่เห็นตอนแรก
const ROWS_STEP = 20;    // กด "แสดงเพิ่ม" ทีละกี่แถว
const ROWS_MAX = 200;    // เพดานแถวบนหน้าจอ (กัน DOM โตไม่หยุด)
const WAIT_MS = 1500;    // ไม่มีค่ามานานเท่านี้ → ขึ้นสถานะ "กำลังรอค่า"

// key -> { th: หัวคอลัมน์ไทย (ใช้กับ table), d: ทศนิยม, w: ความกว้างคอลัมน์ px,
//          pri: true = คอลัมน์หลัก (เห็นตั้งแต่แรก), tip: คำอธิบายสำหรับมือใหม่ }
// เป็น registry เดียวของไฟล์นี้ — ห้ามกระจายชื่อ/หน่วย/ทศนิยมไปฟังก์ชันอื่น
// key ที่ไม่มีในตารางนี้ยังต้องแสดงได้เสมอ (หัวคอลัมน์ = key ดิบ, ทศนิยม = heuristic ใน fmtVal)
const VAL_META = {
  index:          { th: "#",              d: 0, w: 38, pri: true,  tip: "ลำดับวัตถุ — ตรงกับเลขที่วาดบนภาพ" },
  area:           { th: "พื้นที่ (px²)",    d: 0, w: 82, pri: true,  tip: "พื้นที่ภายในเส้นขอบ (ลากผ่านกลางพิกเซลขอบ นับรวมรู) — น้อยกว่าจำนวนพิกเซลเล็กน้อย ดู 'พื้นที่นับพิกเซล'" },
  cx:             { th: "กลาง X (px)",    d: 1, w: 78, pri: true,  tip: "จุดศูนย์กลางแนวนอน นับจากขอบซ้ายของภาพ" },
  cy:             { th: "กลาง Y (px)",    d: 1, w: 78, pri: true,  tip: "จุดศูนย์กลางแนวตั้ง นับจากขอบบนของภาพ" },
  perimeter:      { th: "เส้นรอบรูป (px)", d: 1, w: 92, pri: false, tip: "ความยาวเส้นขอบรอบวัตถุ" },
  circularity:    { th: "ความกลม",        d: 3, w: 76, pri: true,  tip: "1 = กลมสมบูรณ์ · ยิ่งน้อยยิ่งเรียวยาวหรือขอบขรุขระ" },
  bbox_x:         { th: "กรอบ X",         d: 0, w: 62, pri: false, tip: "มุมซ้ายบนของกรอบสี่เหลี่ยมที่ล้อมวัตถุ" },
  bbox_y:         { th: "กรอบ Y",         d: 0, w: 62, pri: false, tip: "มุมซ้ายบนของกรอบสี่เหลี่ยมที่ล้อมวัตถุ" },
  bbox_w:         { th: "กรอบ กว้าง",     d: 0, w: 70, pri: false, tip: "ความกว้างของกรอบที่ล้อมวัตถุ (px)" },
  bbox_h:         { th: "กรอบ สูง",       d: 0, w: 66, pri: false, tip: "ความสูงของกรอบที่ล้อมวัตถุ (px)" },
  orientation_deg:{ th: "มุมเอียง (°)",   d: 1, w: 82, pri: false, tip: "มุมของแกนยาวของวัตถุ — เครื่องหมาย — แปลว่าวัตถุเล็กเกินกว่าจะคำนวณได้" },
  // IP batch 3 — คอลัมน์ใหม่ของ Region Properties (นับพิกเซล แบบ MATLAB) + ตัวนับของ Filter by Shape
  area_px:        { th: "พื้นที่นับพิกเซล", d: 0, w: 92, pri: false, tip: "จำนวนพิกเซลขาวของวัตถุ (ไม่นับรู) — ใช้คิดความตัน/เต็มกรอบ/เส้นผ่านศูนย์กลางเทียบเท่า และเป็นพื้นที่ที่ Filter by Shape ใช้" },
  equiv_diameter: { th: "เส้นผ่านศูนย์กลางเทียบเท่า (px)", d: 1, w: 110, pri: false, tip: "เส้นผ่านศูนย์กลางของวงกลมที่มีพื้นที่เท่าวัตถุนี้" },
  extent:         { th: "เต็มกรอบ", d: 3, w: 70, pri: false, tip: "พื้นที่ ÷ กรอบสี่เหลี่ยมตรง · สี่เหลี่ยมตรง = 1 · วงกลม ≈ 0.77" },
  solidity:       { th: "ความตัน", d: 3, w: 70, pri: true, tip: "พื้นที่ ÷ รูปนูนที่ห่อวัตถุ · 1 = ไม่มีรอยเว้า/รู · ยิ่งน้อยยิ่งเว้าแหว่ง" },
  major_axis:     { th: "แกนยาว (px)", d: 1, w: 82, pri: false, tip: "แกนยาวของวงรีที่กระจายตัวเท่ากับวัตถุ" },
  minor_axis:     { th: "แกนสั้น (px)", d: 1, w: 82, pri: false, tip: "แกนสั้นของวงรีที่กระจายตัวเท่ากับวัตถุ" },
  eccentricity:   { th: "ความรี", d: 3, w: 70, pri: true, tip: "0 = กลม · 0.87 ≈ ยาว 2 เท่าของกว้าง · 0.97 ≈ 4 เท่า · ใกล้ 1 = เป็นเส้น" },
  mean_intensity: { th: "ความสว่างเฉลี่ย", d: 1, w: 96, pri: false, tip: "ค่าเฉลี่ยของภาพที่เข้ากล่องนี้บนพิกเซลของวัตถุ — ภาพขาว-ดำจะได้ 255 เสมอ" },
  sf_total:       { d: 0 },
  sf_kept:        { d: 0 },
  sf_removed_area: { d: 0 },
  sf_removed_shape: { d: 0, tip: "ผ่านเกณฑ์พื้นที่ แต่ค่ารูปร่างอยู่นอกช่วงที่ตั้ง" },
  // Feret (IP batch 2 ชิ้น D) — วัดบนขอบพิกเซล
  feret_max:      { th: "Feret ยาวสุด (px)", d: 2, w: 96, pri: true, tip: "ระยะที่คาลิปเปอร์กางกว้างสุดเมื่อหนีบจากทุกทิศ (เส้นแดงบนภาพ)" },
  feret_min:      { th: "Feret แคบสุด (px)", d: 2, w: 96, pri: true, tip: "ระยะที่คาลิปเปอร์หนีบได้แคบสุด (ความกว้างของกรอบเขียว)" },
  feret_len:      { th: "ยาวตามกรอบแคบสุด (px)", d: 2, w: 110, pri: true, tip: "ความยาวของกรอบเขียว — ใช้แทน 'ด้านยาว' ของวัตถุ" },
  feret_aspect:   { th: "ยาว÷แคบ",        d: 2, w: 70, pri: true,  tip: "1 = กว้างเท่ากันทุกทิศ · ยิ่งมากยิ่งยาวรี" },
  feret_max_deg:  { th: "มุมแนวยาว (°)",  d: 1, w: 82, pri: false, tip: "มุมของเส้นแดง: 0° = แนวนอน เพิ่มตามเข็มนาฬิกา" },
  feret_min_deg:  { th: "มุมแนวแคบ (°)",  d: 1, w: 82, pri: false, tip: "มุมของทิศที่แคบสุด: 0° = แนวนอน เพิ่มตามเข็มนาฬิกา" },
  // scalars: ชื่อใช้ label จาก worker เป็นหลัก ที่นี่ใช้แค่ d/tip
  psnr_db:        { d: 2, tip: "ยิ่งสูงยิ่งเหมือนกัน · >30 dB ถือว่าใกล้เคียงมาก", infTip: "ค่าสูงไม่จำกัด — สองภาพเหมือนกันทุกพิกเซล" },
  mse:            { d: 2, tip: "ความต่างเฉลี่ยยกกำลังสอง · 0 = เหมือนกันเป๊ะ" },
  ssim:           { d: 4, tip: "1 = เหมือนกันทุกประการ · ใกล้ 0 = ต่างกันมาก" },
  foreground_pct: { d: 1, tip: "สัดส่วนพื้นที่ภาพที่ถือว่าเคลื่อนไหว" },
  foreground_px:  { d: 0 },
  frame:          { d: 0, tip: "นับจากเฟรมแรกที่กล่องนี้เริ่มเรียนรู้" },
  mean_motion_px: { d: 2, tip: "ระยะที่ภาพขยับโดยเฉลี่ยต่อ 1 เฟรม" },
  max_motion_px:  { d: 1 },
  moving_pct:     { d: 1 },
  mean_dx_px:     { d: 2, tip: "บวก = ขยับไปทางขวา · ลบ = ไปทางซ้าย" },
  mean_dy_px:     { d: 2, tip: "บวก = ขยับลง · ลบ = ขยับขึ้น" },
  points:         { d: 0 },
  tracked:        { d: 0 },
  lost:           { d: 0 },
  reseeded:       { d: 0, tip: "เฟรมนี้เพิ่งเลือกจุดเด่นชุดใหม่หรือไม่" },
  match_score:    { d: 1, tip: "100% = เหมือนแม่แบบทุกพิกเซล" },
  x:              { d: 0 },
  y:              { d: 0 },
  // Line Profile / ROI Polygon (IP batch 2 ชิ้น D) — ชื่อมาจาก label ของ worker · ที่นี่แค่ทศนิยม/คำอธิบาย
  line_length_px: { d: 1 },
  samples:        { d: 0 },
  profile_min:    { d: 0 },
  profile_max:    { d: 0 },
  profile_mean:   { d: 1 },
  profile_contrast: { d: 0, tip: "ยิ่งมาก ขอบ/ลวดลายบนเส้นยิ่งชัด" },
  edge_jump:      { d: 0, tip: "บวก = มืดไปสว่าง · ลบ = สว่างไปมืด (มองจาก A ไป B)" },
  edge_pos_px:    { d: 1, tip: "วงกลมม่วงบนภาพ" },
  edge_pair_px:   { d: 1, tip: "ความกว้างของแถบที่เส้นตัดผ่าน — ตรงเมื่อเส้นตั้งฉากกับแถบ" },
  roi_vertices:   { d: 0 },
  roi_area_px:    { d: 0 },
  roi_area_pct:   { d: 1 },
  roi_perimeter_px: { d: 1 },
  roi_mean:       { d: 1 },
  roi_std:        { d: 1, tip: "ยิ่งมาก ความสว่างใน ROI ยิ่งไม่สม่ำเสมอ" },
  roi_crossed:    { tip: "ใช่ = ลำดับจุดไขว้กัน ควรเรียงจุดตามเข็มหรือทวนเข็มนาฬิกา" },
  // IP batch 4A — หน่วยมิลลิเมตร (เมื่อตั้ง "ขนาดจริงต่อพิกเซล" > 0) · ΔE · ความคม · Apply in ROI (ชื่อของ scalar มาจาก label ของ worker)
  area_mm2:       { th: "พื้นที่ (มม.²)", d: 2, w: 92, pri: true, tip: "พื้นที่นับพิกเซล × (มม./px)²" },
  perimeter_mm:   { th: "เส้นรอบรูป (มม.)", d: 2, w: 96, pri: false },
  equiv_diameter_mm: { th: "เส้นผ่านศูนย์กลางเทียบเท่า (มม.)", d: 2, w: 118, pri: false },
  major_axis_mm:  { th: "แกนยาว (มม.)", d: 2, w: 86, pri: false },
  minor_axis_mm:  { th: "แกนสั้น (มม.)", d: 2, w: 86, pri: false },
  feret_max_mm:   { th: "Feret ยาวสุด (มม.)", d: 2, w: 100, pri: true },
  feret_min_mm:   { th: "Feret แคบสุด (มม.)", d: 2, w: 100, pri: true },
  feret_len_mm:   { th: "ยาวตามกรอบแคบสุด (มม.)", d: 2, w: 118, pri: true },
  line_length_mm: { d: 2 },
  edge_pos_mm:    { d: 2 },
  edge_pair_mm:   { d: 2 },
  focus_score:    { d: 1, tip: "ใช้เทียบภาพฉากเดียวกันเท่านั้น — ยิ่งสูงยิ่งคม (ภาพสว่างหรือ noise มากก็ทำให้สูงขึ้นได้)" },
  focus_mean:     { d: 1 },
  de_mean:        { d: 2, tip: "ต่ำกว่า 1 แทบมองไม่ออก · ราว 2–3 เริ่มเห็น · เกิน 5 เห็นชัด (ค่าประมาณ)" },
  de_max:         { d: 2 },
  de_over_pct:    { d: 1 },
  de_a_L:         { d: 2 },
  de_a_a:         { d: 2 },
  de_a_b:         { d: 2 },
  de_b_L:         { d: 2 },
  de_b_a:         { d: 2 },
  de_b_b:         { d: 2 },
  ra_px:          { d: 0 },
  ra_pct:         { d: 1 },
  // IP batch 4B — จัดภาพ ECC · Gabor · GLCM · Dehaze · DCT/JPEG (ชื่อของ scalar มาจาก label ของ registry)
  ecc_ok:         { d: 0, tip: "1 = ลู่เข้าและจัดภาพได้ · 0 = ไม่สำเร็จ (ภาพเรียบ/เล็ก/ไม่ลู่เข้า) — ค่าอื่นจะว่าง" },
  ecc_rho:        { d: 3, tip: "ยิ่งใกล้ 1 ยิ่งทับกันดี" },
  ecc_dx:         { d: 2 },
  ecc_dy:         { d: 2 },
  ecc_angle_deg:  { d: 2 },
  ecc_scale:      { d: 3 },
  gabor_mean:     { d: 1 },
  gabor_max:      { d: 1 },
  glcm_contrast:    { d: 4, tip: "สูง = เพื่อนบ้านต่างกันมาก (ลายหยาบ/ขอบเยอะ)" },
  glcm_correlation: { d: 4, tip: "ใกล้ 1 = เพื่อนบ้านไปทางเดียวกัน · ว่างเมื่อภาพมีระดับเดียว" },
  glcm_energy:      { d: 4, tip: "สูง = ลายซ้ำ ๆ เป็นระเบียบ" },
  glcm_homogeneity: { d: 4, tip: "สูง = ภาพเรียบ" },
  glcm_pairs:       { d: 0 },
  dh_a_r:         { d: 1 },
  dh_a_g:         { d: 1 },
  dh_a_b:         { d: 1 },
  dh_t_mean:      { d: 3 },
  dct_psnr:       { d: 2, tip: "ยิ่งสูงยิ่งเหมือนภาพเดิม", infTip: "ค่าสูงไม่จำกัด — เหมือนเดิมทุกพิกเซล" },
  dct_zero_pct:   { d: 1 },
  dct_blocks:     { d: 0 },
};
const META_FALLBACK = { w: 88, pri: true }; // key ที่ไม่รู้จัก → d undefined = ใช้ heuristic ใน fmtVal
function metaOf(key) {
  const m = VAL_META[key];
  return m || META_FALLBACK;
}

// cache Intl.NumberFormat ตามจำนวนทศนิยม — new ทุกเซลล์ทุกเฟรมคือกับดัก perf ตัวจริง
const NF = new Map();
function nf(d) {
  let f = NF.get(d);
  if (!f) {
    f = new Intl.NumberFormat("th-TH", { minimumFractionDigits: d, maximumFractionDigits: d });
    NF.set(d, f);
  }
  return f;
}
const fmtInt = (n) => nf(0).format(n);

// จัดรูปค่า 1 ตัว → { t: ข้อความ, c: คลาสเสริม, ti: tooltip (ค่าดิบ) }
function fmtVal(v, meta) {
  const m = meta || META_FALLBACK;
  if (v === null || v === undefined) return { t: "—", c: "dv-na", ti: "ไม่มีค่า" };
  if (typeof v === "boolean") return { t: v ? "ใช่" : "ไม่ใช่", c: "", ti: "" };
  if (typeof v !== "number") return { t: String(v), c: "", ti: "" };
  if (Number.isNaN(v)) return { t: "—", c: "dv-na", ti: "คำนวณค่านี้ไม่ได้" };
  if (!Number.isFinite(v)) return { t: v > 0 ? "∞" : "−∞", c: "dv-inf", ti: v > 0 ? (m.infTip || "ค่าเป็นอนันต์") : "ค่าเป็นลบอนันต์" };
  const av = Math.abs(v);
  if (av >= 1e9 || (av > 0 && av < 1e-4)) return { t: v.toExponential(2), c: "", ti: String(v) };
  const d = m.d !== undefined ? m.d : (Number.isInteger(v) ? 0 : 2);
  return { t: nf(d).format(v), c: "", ti: String(v) };
}

// signature ของ "โครง" note — ห้ามมีตัวเลข/เนื้อข้อความ ไม่งั้นกลับไป rebuild ทุกเฟรม (P-16)
function structSig(mine) {
  let s = "";
  let nv = 0;
  for (const n of mine) {
    if (n.autoGray) s += "g|";
    else if (n.info) s += "i|"; // ไม่เอาเนื้อข้อความ — Motion เปลี่ยนข้อความทุกเฟรม
    else if (n.values) {
      const v = n.values;
      nv++;
      s += "v:" + v.kind + ":" +
        (v.kind === "table"
          ? (v.columns || []).join(",")                 // ไม่เอา rows.length — จำนวนวัตถุกระพริบตลอด
          : (v.fields || []).map((f) => (f ? f.key : "")).join(",")) + "|";
    }
  }
  return s + "#" + nv;
}

// เอา values ของ note "ตัวแรก" เสมอ — กล่องผลลัพธ์กล่องแรกคือกล่องที่รัน op จริง
// ตัวถัดมาได้ค่าจาก memo (cv-worker.js:184-188) ซึ่งถ้าวันหนึ่ง regress จะไม่มีค่า → อ่านตัวแรกจึงเป็นเกราะ
function firstValues(mine) {
  return (mine.find((n) => n.values) || {}).values || null;
}

// รหัส "มุมมอง" ของแผง = filter + วิธีที่เลือกอยู่
// กล่องรวม (mergeGroup) ใช้ step.id เดียวกันทุกวิธี เช่น shape_analysis มี 6 วิธี แต่มีแค่ Region
// Properties ที่ส่ง ctx.values — ถ้าคีย์สถานะด้วย step.id อย่างเดียว ค่าของวิธีเก่าจะค้างข้ามไปวิธีใหม่
// ใช้สัญญาเดิมของ registry คือ f.subtitle(params) (= ชื่อวิธีที่เลือก) จึงไม่ต้องรู้ key ภายใน (__mode)
// node ปกติไม่มี subtitle → ได้ "<id>|" คงที่ พฤติกรรมเท่าเดิมทุกประการ
function viewIdOf(step) {
  if (!step) return "";
  const f = FILTERS[step.id];
  return step.id + "|" + (f && typeof f.subtitle === "function" ? f.subtitle(step.params) : "");
}

// note ของ "กล่องนี้" ในมุมมองปัจจุบัน — ที่เดียวที่ตัดสินว่า note ไหนเป็นของแผง (app.js ก็ใช้ตัวเดียวกัน)
//  · nodeId ต้องตรงกัน (worker ใส่ step.nodeId ให้ทุก note · กล่องชนิดเดียวกัน 2 กล่องจึงไม่ปนกัน · note สาย B
//    ก็ใช้ nodeId ของ node ต้นทางในสาย B) — ไม่พึ่ง n.id (id ของ filter) อีก
//  · note ที่พก view (= f.subtitle ตอนคำนวณ) ต้องตรงกับ subtitle ของมุมมองตอนนี้ ไม่งั้นเป็นของ "วิธีเก่า" ที่ตามมา
//    กับเฟรมค้างในท่อ → ทิ้ง (แน่นอน ไม่ต้องจับเวลา · ถ้าเป็นเฟรมเดียวที่ส่งหลังสลับ ก็เป็นเฟรมของวิธีใหม่อยู่แล้ว)
export function notesOfStep(step, notes) {
  if (!step || !notes) return [];
  const f = FILTERS[step.id];
  const hasView = f && typeof f.subtitle === "function";
  const view = hasView ? f.subtitle(step.params || {}) : undefined;
  return notes.filter((n) => n.nodeId === step.nodeId && (view === undefined || n.view === undefined || n.view === view));
}

// สร้าง step + note ของกล่องที่เลือก จาก "sel" ของ editor ({ filterId, id, getParams() }) — ใช้ทั้งทาง render ใหม่
// และทางลาก slider (updateInfo) ใน app.js จึงเป็นฟังก์ชัน pure ที่ทดสอบพฤติกรรมได้
//  · params ต้องมาจาก sel.getParams() (วิธีปัจจุบันของกล่องรวม) ไม่งั้น view ตกเป็นค่า default แล้วไม่เห็นค่าเลย
//  · nodeId = sel.id (id ของ node ใน LiteGraph = step.nodeId ที่ worker ใส่ใน note)
export function stepAndNotesOf(sel, allNotes) {
  const step = { id: sel.filterId, params: sel.getParams(), enabled: true, nodeId: sel.id };
  return { step, notes: notesOfStep(step, allNotes) };
}

// ข้อความ info ของ note ที่กรองแล้ว (ใช้กับ updateInfo ตอนลาก slider)
export function infoTextsOf(notes) {
  return (notes || []).filter((n) => n.info).map((n) => n.info);
}

// cb: onParamChange(key, value)
export function createDetailPanel(container, cb) {
  // step ล่าสุด (ใช้ re-render ตัวเองเมื่อเปลี่ยน "วิธี" ใน merged node)
  let curStep = null;
  let curNotes = null;
  // true ระหว่างที่ "แผงนี้เอง" เป็นคนสั่งเปลี่ยน param — ใช้แยกจากการเปลี่ยนวิธีนอกแผง (isViewStale)
  let ownChange = false;
  function notifyParam(key, value) {
    ownChange = true;
    try { cb.onParamChange(key, value); } finally { ownChange = false; }
  }
  // ตัวควบคุมของพารามิเตอร์ที่รับโฟกัสคืนหลัง re-render: "key|kind" -> element (kind: main = select/ปุ่ม segmented ที่เลือกอยู่/toggle · slider · box)
  let paramFocus = new Map();
  const pkeyOf = (key, kind) => key + "|" + kind;
  // ติดป้ายให้ตัวควบคุม (data-pkey/data-pkind) + ลงทะเบียนไว้ให้ render() หาเจอหลังสร้างใหม่
  function tagParam(el, key, kind, reg = true) {
    el.setAttribute("data-pkey", key);
    el.setAttribute("data-pkind", kind);
    if (reg) paramFocus.set(pkeyOf(key, kind), el);
  }
  const infoState = new WeakMap(); // .detail-note-info -> { info, tile: {l,v,u}|null } (ข้อความล่าสุดที่วาดไว้ — เขียนเฉพาะเมื่อเปลี่ยน)

  // --- U3: สถานะของบล็อก "ค่าที่วัดได้" ---
  let valuesView = null;   // บล็อก .dv ที่อยู่ในแผงตอนนี้ (null = ไม่มี)
  let infoEls = [];        // .detail-note-info ที่สร้างไว้ (เรียงตาม note ที่มี info)
  let builtSig = "";       // signature ของโครงที่แผงตอนนี้สร้างไว้
  let panelInert = true;   // true = แผงนี้ไม่มีบล็อกค่าและจะไม่มีวันมี → applyNotes ไม่ต้องให้ rebuild
  let lastPaint = 0;
  let pendingTimer = null;
  let pendingMine = null;
  // สถานะที่ผู้ใช้ตั้งเอง — คงไว้ตราบที่ยังเป็น "มุมมองเดิม" (filter + วิธีที่เลือก) · ดู viewIdOf()
  let lastValuesViewId = null;
  let frozen = false;
  let frozenSnap = null;
  let shownLimit = ROWS_DEFAULT;
  let showAllCols = false;
  let lastValues = null;   // ค่าชุดล่าสุดที่เคยได้ของ filter นี้ (ใช้ตอน "รอค่า" / ปลดตรึง / เพิ่มแถว)
  let lastSeenAt = 0;

  // บันทึกค่าชุดใหม่ที่เพิ่งได้ — ที่เดียวที่เขียน lastValues/lastSeenAt
  function captureValues(v) {
    lastValues = v;
    lastSeenAt = Date.now();
  }

  // ทิ้งอ้างอิง DOM เก่าทั้งหมด + ล้าง timer ที่ค้าง — ต้องเรียกทุกครั้งที่ container.innerHTML = ""
  // ไม่งั้น trailing timeout จะไปเขียน node ที่หลุดออกจากต้นไม้แล้ว
  function resetValuesView() {
    if (pendingTimer) { clearTimeout(pendingTimer); pendingTimer = null; }
    pendingMine = null;
    valuesView = null;
    infoEls = [];
    builtSig = "";
    lastPaint = 0;
  }

  // re-render แผงหลังผู้ใช้กดปุ่มที่เปลี่ยนหน้าตาแผง (select "เลือกวิธี" / toggle)
  // ถ้าเปลี่ยน "วิธี" จริง note ที่ค้างอยู่ใน curNotes เป็นของวิธีเก่าทั้งชุด (app.js อัปเดต state.notes เฉพาะตอน
  // worker ส่งผลเฟรมถัดไป) — render() กรองทิ้งด้วย view tag ให้เอง จึงไม่ได้ตารางผีของวิธีเก่า รอ note เฟรมถัดไป
  function rerenderAfterParamChange() {
    if (!curStep) return;
    // note ของวิธีเก่าถูกกรองทิ้งเองใน render() ด้วย view tag (notesOfStep) — ไม่ต้องแยกกรณี/ไม่ต้องเฝ้าเวลา
    render(curStep, curNotes);
  }

  // เปลี่ยน "วิธี" (select/segmented) แล้วแผงถูกสร้างใหม่ — คืนโฟกัสให้ตัวควบคุมตัวใหม่ (คีย์บอร์ดไม่หลุดจากที่เดิม)
  function changeChoice(param, nextValue) {
    notifyParam(param.key, nextValue);
    // re-render เพื่ออัปเดตตัวเลือกที่เลือก + สลับ sub-params (กรณี "เลือกวิธี")
    if (curStep) {
      curStep.params[param.key] = nextValue;
      rerenderAfterParamChange(); // render() คืนโฟกัสให้ตัวควบคุมตัวเดิมเอง (ดู restoreFocus)
    }
  }

  // โฟกัสที่อยู่ในแผงก่อน render() ล้าง DOM — จำ key/ชนิด/ตำแหน่ง caret ไว้ แล้วคืนให้ element ที่มี key เดียวกันหลังสร้างใหม่
  // (แผงถูกสร้างใหม่ได้อีกครั้งเมื่อ signature ของ note เปลี่ยนหลังเปลี่ยนวิธี → activeElement ไม่ควรกลายเป็น BODY)
  function captureFocus() {
    const a = typeof document !== "undefined" ? document.activeElement : null;
    if (!a || !a.getAttribute) return null;
    const key = a.getAttribute("data-pkey");
    if (key == null) return null;
    const sel = typeof a.selectionStart === "number" ? [a.selectionStart, a.selectionEnd] : null;
    return { id: pkeyOf(key, a.getAttribute("data-pkind") || "main"), sel };
  }
  function restoreFocus(rec) {
    if (!rec) return;
    const el = paramFocus.get(rec.id);
    if (!el || !el.focus) return;
    el.focus();
    if (rec.sel && el.setSelectionRange) { try { el.setSelectionRange(rec.sel[0], rec.sel[1]); } catch { /* type=number ไม่รองรับ */ } }
  }

  // เขียน note-info ลงตัวองค์ประกอบ: เข้ารูปแบบ → readout tile (ป้าย/ตัวเลข/หน่วย) · ไม่เข้า → แถวข้อความ "📊 …" เดิม
  // เขียน DOM เฉพาะส่วนที่เปลี่ยน (Motion เปลี่ยนข้อความทุกเฟรม)
  function setInfo(el, info) {
    let st = infoState.get(el);
    if (!st) { st = { info: null, tile: null }; infoState.set(el, st); }
    if (st.info === info) return;
    st.info = info;
    const parts = splitReadout(info);
    if (!parts) {
      if (st.tile) { el.textContent = ""; st.tile = null; el.classList.remove("readout"); }
      const txt = "📊 " + info;
      if (el.textContent !== txt) el.textContent = txt;
      return;
    }
    if (!st.tile) {
      el.textContent = "";
      const l = document.createElement("span");
      l.className = "rt-label";
      const v = document.createElement("span");
      v.className = "rt-value mono";
      const u = document.createElement("span");
      u.className = "rt-unit";
      el.append(l, v, u);
      el.classList.add("readout");
      st.tile = { l, v, u };
    }
    if (st.tile.l.textContent !== parts.label) st.tile.l.textContent = parts.label;
    if (st.tile.v.textContent !== parts.value) st.tile.v.textContent = parts.value;
    if (st.tile.u.textContent !== parts.unit) st.tile.u.textContent = parts.unit;
  }

  // render(step, notes) — step = { id, params, enabled } หรือ null ถ้าไม่ได้เลือก
  // notes = รายการ note ของ filter นี้ เช่น [{autoGray:true},{info:"พบวัตถุ 3 ชิ้น"}]
  function render(step, notes) {
    notes = step ? notesOfStep(step, notes) : null; // กรอง nodeId + view อีกชั้น (idempotent) — ผู้เรียกจะส่งดิบหรือกรองแล้วก็ได้
    curStep = step;
    curNotes = notes;
    resetValuesView();
    panelInert = true; // ปลดเป็น false ใน buildValuesBlock เมื่อรู้แน่ว่าแผงนี้คุมโครงได้
    const focusRec = captureFocus();
    paramFocus = new Map();
    container.innerHTML = "";

    if (!step) {
      const empty = document.createElement("div");
      empty.className = "detail-empty";
      empty.innerHTML =
        `<div class="de-icon">🎛️</div>` +
        `<div class="de-title">เลือกกล่องบนกระดานเพื่อปรับค่า</div>` +
        `<div class="de-desc">เลือกกล่องบน Pipeline Canvas เพื่อดูและปรับพารามิเตอร์</div>`;
      // ทางลัดสำหรับคนเริ่มใหม่: วางเทมเพลตสำเร็จรูป (callback มาจาก app.js — ไม่ import app.js ที่นี่)
      if (cb.onTemplates) {
        const tb = document.createElement("button");
        tb.type = "button";
        tb.className = "btn de-template";
        tb.textContent = "▦ เริ่มจากเทมเพลต";
        tb.title = "วาง pipeline ตัวอย่างสำเร็จรูปให้ทันที";
        tb.addEventListener("click", () => cb.onTemplates());
        empty.appendChild(tb);
      }
      container.appendChild(empty);
      return;
    }

    const f = FILTERS[step.id];
    if (!f) return;

    // หัวขั้นตอนครั้งเดียว: ชื่อ (merged node = ชื่อ "วิธี" ที่เลือกอยู่) + ชิปหมวด
    // เดิมซ้ำ 3 ครั้ง (ชื่อกลุ่ม h2 / ชิป / "▶ ชื่อวิธี")
    const head = document.createElement("div");
    head.className = "detail-head";
    const title = document.createElement("h2");
    title.className = "detail-title";
    title.textContent = typeof f.subtitle === "function" ? f.subtitle(step.params) : f.name;
    head.appendChild(title);
    if (f.group && f.group !== title.textContent) {
      const group = document.createElement("span");
      group.className = "detail-group";
      group.textContent = f.group;
      head.appendChild(group);
    }
    container.appendChild(head);

    // note ข้อความ: autoGray อยู่บนสุด (ข้อมูลประกอบ) · note-info (📊) เก็บเป็นกลุ่ม readout tile วางใต้พารามิเตอร์ (ดู finishRender)
    // note ที่มีแต่ values ไม่มีข้อความให้แสดง — ข้ามไป ไม่งั้นได้ div เปล่าที่กิน margin 14px
    const readouts = document.createElement("div");
    readouts.className = "readouts";
    const readoutsHead = document.createElement("div");
    readoutsHead.className = "params-head readouts-head";
    readoutsHead.textContent = "ค่าที่วัดได้";
    readouts.appendChild(readoutsHead);
    for (const note of notes || []) {
      if (!note.autoGray && !note.info) continue;
      const n = document.createElement("div");
      n.className = "detail-note";
      if (note.autoGray) {
        n.textContent = "ℹ️ filter นี้ทำงานกับภาพขาวดำ — ระบบแปลงภาพให้อัตโนมัติก่อนประมวลผล";
        container.appendChild(n);
      } else if (note.info) {
        n.classList.add("detail-note-info");
        setInfo(n, note.info);
        infoEls.push(n);
        readouts.appendChild(n);
      }
    }

    // ส่วนท้ายของแผง (ค่าที่วัดได้) — เรียกจาก "ทั้งสองทาง" (มี/ไม่มีพารามิเตอร์) ไม่งั้น SSIM (ไม่มี param) จะไม่มีตารางเลย
    // ตำแหน่ง DOM = ใต้พารามิเตอร์ (spec §6.2) · ตัวบล็อกค่า (buildValuesBlock) ไม่ถูกแตะ — เปลี่ยนแค่จุดที่เรียก
    const finishRender = () => {
      if (infoEls.length) container.appendChild(readouts);
      buildValuesBlock(step, notes);
      readoutsHead.hidden = !!valuesView; // มีบล็อกค่า → หัว "ค่าที่วัดได้" ของบล็อกนั้นพอแล้ว
      restoreFocus(focusRec);
    };

    // กรอง param ที่มี showIf ให้โชว์เฉพาะเมื่อ "วิธี" ตรงกับที่เลือก
    const visible = f.params.filter(
      (p) => !p.showIf || step.params[p.showIf.key] === p.showIf.value
    );

    let defaults = {};
    try { defaults = defaultParams(step.id); } catch { /* ไม่มี default → ไม่โชว์ปุ่มรีเซ็ต */ }
    // ลำดับ: หัว → วิธี (mode) → "พารามิเตอร์" + ตัวที่เหลือ (หรือ "วิธีนี้ไม่มีค่าให้ปรับ") → ค่าที่วัดได้
    const modeParam = visible.find((q) => q.key === "mode");
    if (modeParam) container.appendChild(buildParam(modeParam, step.params[modeParam.key], defaults[modeParam.key]));
    const rest = visible.filter((q) => q !== modeParam);
    if (rest.length === 0) {
      const none = document.createElement("div");
      none.className = "param-hint";
      none.textContent = "วิธีนี้ไม่มีค่าให้ปรับ";
      container.appendChild(none);
    } else {
      // หัวข้อคั่น "พารามิเตอร์" (Design §19) — แยกส่วน "อ่าน" (learning) ออกจาก "ปรับ" (controls)
      const phead = document.createElement("div");
      phead.className = "params-head";
      phead.textContent = "พารามิเตอร์";
      container.appendChild(phead);
      for (const param of rest) {
        container.appendChild(buildParam(param, step.params[param.key], defaults[param.key]));
      }
    }
    finishRender();
  }

  // แสดงแผงตั้งชื่อกล่องผลลัพธ์ — info = { id, getName, setName }
  function renderDisplay(info) {
    curStep = null;
    curNotes = null;
    resetValuesView();
    panelInert = true;
    container.innerHTML = "";

    const title = document.createElement("h2");
    title.className = "detail-title";
    title.textContent = "🖥️ กล่องผลลัพธ์";
    container.appendChild(title);

    const desc = document.createElement("div");
    desc.className = "detail-desc";
    desc.textContent =
      "ตั้งชื่อกล่องนี้ให้สื่อว่าแสดงภาพอะไร — ชื่อจะไปขึ้นที่หัวจอผลลัพธ์ด้านบน ถ้าเว้นว่างจะใช้คำอธิบายอัตโนมัติจากสายที่ต่อ";
    container.appendChild(desc);

    const wrap = document.createElement("div");
    wrap.className = "param";
    const lab = document.createElement("div");
    lab.className = "param-label";
    lab.innerHTML = "<span>ชื่อกล่อง</span>";
    wrap.appendChild(lab);
    const input = document.createElement("input");
    input.type = "text";
    input.className = "name-input";
    input.placeholder = "เว้นว่าง = ใช้ชื่ออัตโนมัติ Display N";
    input.value = info.getName();
    input.addEventListener("input", () => info.setName(input.value.trim()));
    wrap.appendChild(input);
    container.appendChild(wrap);

    // รายการขั้นตอนที่ภาพผ่าน (F2) — ค่าดิบ + คลิกไปแก้ที่ node
    if (info.getSteps) renderSteps(container, info);

    // ปุ่มสร้างโค้ด Python (OpenCV) ตามลำดับ Box ที่ภาพวิ่งผ่าน
    if (info.onGenCode) {
      const gen = document.createElement("button");
      gen.className = "btn btn-primary";
      gen.style.width = "100%";
      gen.style.marginTop = "12px";
      gen.textContent = "🐍 สร้างโค้ด Python (OpenCV)";
      gen.title = "สร้างโค้ด image processing ตามลำดับ Box ที่ต่อเข้ากล่องนี้ (คนละอันกับปุ่ม Export เทรน YOLO)";
      gen.addEventListener("click", () => info.onGenCode());
      container.appendChild(gen);

      const genHint = document.createElement("div");
      genHint.className = "param-hint";
      genHint.style.marginTop = "6px";
      genHint.textContent = "ได้โค้ด cv2.* ตามขั้นตอนจริง เอาไปรัน/ศึกษาต่อได้";
      container.appendChild(genHint);
    }
  }

  // สร้างรายการขั้นตอน (ค่าดิบ key=value) ในแผงขวา
  function renderSteps(container, info) {
    const dp = info.getSteps();
    const head = document.createElement("div");
    head.className = "steps-head";
    head.textContent = "ขั้นตอนที่ภาพผ่าน (คลิกเพื่อไปแก้ค่า)";
    container.appendChild(head);

    if (!dp || !dp.connected) {
      const e = document.createElement("div");
      e.className = "steps-empty";
      e.textContent = "กล่องนี้ยังไม่ได้ต่อสายถึงแหล่งภาพ";
      container.appendChild(e);
      return;
    }
    const src = document.createElement("div");
    src.className = "steps-src";
    src.textContent = "● แหล่งภาพ";
    container.appendChild(src);

    if (dp.pipeline.length === 0) {
      const e = document.createElement("div");
      e.className = "steps-empty";
      e.textContent = "ภาพต้นฉบับ — ยังไม่ผ่าน filter ใด";
      container.appendChild(e);
      return;
    }
    dp.pipeline.forEach((s, i) => {
      const f = FILTERS[s.id];
      const row = document.createElement("button");
      row.className = "step-row";
      row.type = "button";
      const idx = document.createElement("span");
      idx.className = "pipe-index";
      idx.textContent = String(i + 1);
      const name = document.createElement("span");
      name.className = "step-name";
      name.textContent = f ? (typeof f.subtitle === "function" ? f.subtitle(s.params) : f.name) : s.id;
      row.appendChild(idx);
      row.appendChild(name);
      if (f) {
        for (const p of f.params) {
          if (p.showIf && s.params[p.showIf.key] !== p.showIf.value) continue;
          if (p.key === "mode") continue;
          const chip = document.createElement("span");
          chip.className = "step-param";
          const k = p.key.includes("__") ? p.key.split("__").pop() : p.key;
          chip.textContent = `${p.label || k}=${s.params[p.key]}`;
          chip.title = k; // key ดิบ (โยงกับโค้ด)
          row.appendChild(chip);
        }
      }
      const arr = document.createElement("span");
      arr.className = "step-arrow";
      arr.textContent = "›";
      row.appendChild(arr);
      row.addEventListener("click", () => info.onJump(s.nodeId));
      container.appendChild(row);
    });
  }

  // แผงกล่อง YOLO Detect — browse โมเดล/labels + เริ่มตรวจจับ
  // info = { getStatus(), detecting, onBrowseModel(), onBrowseLabels(), onToggle() }
  function renderDL(info) {
    curStep = null;
    curNotes = null;
    resetValuesView();
    panelInert = true;
    container.innerHTML = "";

    const title = document.createElement("h2");
    title.className = "detail-title";
    title.textContent = "🧠 Model Detect";
    container.appendChild(title);

    const desc = document.createElement("div");
    desc.className = "detail-desc";
    desc.textContent =
      "เลือกไฟล์โมเดล YOLO (best.pt ที่เทรนมา หรือ yolo สำเร็จรูป / .onnx) แล้วตรวจจับวัตถุบนกล้องในโปรแกรมได้เลย — ชื่อคลาสดึงจากตัวโมเดลอัตโนมัติ";
    container.appendChild(desc);

    const status = document.createElement("div");
    status.className = "detail-note detail-note-info";
    status.textContent = "📊 " + info.getStatus();
    container.appendChild(status);

    const mk = (label, onClick, primary) => {
      const b = document.createElement("button");
      b.className = primary ? "btn btn-primary" : "btn";
      b.style.width = "100%";
      b.style.marginTop = "8px";
      b.textContent = label;
      b.addEventListener("click", onClick);
      container.appendChild(b);
    };
    mk("📂 เลือกโมเดล (best.pt / yolo / .onnx)", () => info.onBrowseModel(), true);
    mk("🏷️ โหลดชื่อคลาส (data.yaml / classes.txt / .json)", () => info.onBrowseLabels(), false);

    const hint = document.createElement("div");
    hint.className = "param-hint";
    hint.style.marginTop = "10px";
    hint.textContent = "ต่อสาย: แหล่งภาพ → กล่องนี้ → ผลลัพธ์ แล้วกล่องผลลัพธ์จะโชว์การตรวจจับ";
    container.appendChild(hint);
  }

  function buildParam(param, value, defVal) {
    const wrap = document.createElement("div");
    wrap.className = "param";

    if (param.type === "boolean") {
      // Toggle ON/OFF อ่านง่าย (Design §22) — คลิกทั้งแถวได้ ค่ายังส่ง boolean เท่าเดิม
      const row = document.createElement("button");
      row.type = "button";
      row.className = "param-toggle" + (value ? " on" : "");
      row.setAttribute("role", "switch");
      tagParam(row, param.key, "main");
      row.setAttribute("aria-checked", value ? "true" : "false");
      const lab = document.createElement("span");
      lab.className = "pt-label";
      lab.textContent = param.label;
      const sw = document.createElement("span");
      sw.className = "pt-switch";
      sw.innerHTML = `<span class="pt-knob"></span>`;
      const st = document.createElement("span");
      st.className = "pt-state";
      st.textContent = value ? "ON" : "OFF";
      row.appendChild(lab);
      row.appendChild(st);
      row.appendChild(sw);
      row.addEventListener("click", () => {
        const nv = !row.classList.contains("on");
        notifyParam(param.key, nv);
        if (curStep) { curStep.params[param.key] = nv; rerenderAfterParamChange(); }
      });
      wrap.appendChild(row);
      return wrap;
    }

    // select: "mode" และที่มี > 3 ตัวเลือก → <select> (กินที่น้อย เลื่อนด้วยคีย์บอร์ดได้) · <= 3 ตัวเลือกสั้น ๆ → segmented control
    if (param.type === "select") {
      const lab = document.createElement("div");
      lab.className = "param-label";
      lab.innerHTML = `<span>${param.label}</span>`;
      wrap.appendChild(lab);

      const foundIdx = param.options.findIndex((o) => o.value === value);
      if (selectStyle(param) === "seg") {
        const seg = document.createElement("div");
        seg.className = "seg-control";
        seg.setAttribute("role", "group");
        seg.setAttribute("aria-label", param.label);
        param.options.forEach((opt, i) => {
          const b = document.createElement("button");
          b.type = "button";
          b.className = "seg-btn" + (opt.value === value ? " active" : "");
          b.setAttribute("aria-pressed", opt.value === value ? "true" : "false");
          b.textContent = opt.label;
          if (opt.hint) b.title = opt.hint;
          b.setAttribute("data-pkey", param.key);
          b.setAttribute("data-pkind", "main");
          if (opt.value === value) paramFocus.set(pkeyOf(param.key, "main"), b);
          b.addEventListener("click", () => {
            if (opt.value === value) return;
            changeChoice(param, opt.value);
          });
          seg.appendChild(b);
        });
        wrap.appendChild(seg);
        return wrap;
      }

      const sel = document.createElement("select");
      sel.className = "field-select";
      sel.setAttribute("aria-label", param.label);
      tagParam(sel, param.key, "main");
      // ค่าที่เก็บไว้ไม่อยู่ในตัวเลือก → ใส่ตัวเลือกว่าง (ซ่อน) ไว้หน้าสุดและเลือกมันไว้ ไม่งั้นจะโชว์ตัวเลือกแรกทั้งที่ค่าจริงเป็นอย่างอื่น
      // แล้วเลือกตัวแรกไม่ commit (change ไม่ยิงเมื่อเลือกตัวเดิม)
      const offset = foundIdx < 0 ? 1 : 0;
      if (offset) {
        const ph = document.createElement("option");
        ph.textContent = "— เลือก —";
        ph.hidden = true;
        ph.disabled = true;
        sel.appendChild(ph);
      }
      param.options.forEach((opt) => {
        const o = document.createElement("option");
        o.textContent = opt.label;
        if (opt.hint) o.title = opt.hint;
        sel.appendChild(o);
      });
      sel.selectedIndex = foundIdx + offset;
      // อ่านตัวเลือกจาก index ไม่ใช่ value ของ <option> — value ใน registry เป็นชนิดอะไรก็ได้ (ไม่ถูกแปลงเป็นสตริง)
      sel.addEventListener("change", () => {
        const opt = param.options[sel.selectedIndex - offset];
        if (!opt || opt.value === value) return;
        changeChoice(param, opt.value);
      });
      wrap.appendChild(sel);
      return wrap;
    }

    // number → สองบรรทัด: [ป้าย | ช่องค่า | ↺] / [สไลเดอร์เต็มกว้าง] / [min … max] (spec §6.2)
    wrap.className = "param param-num";
    const lab = document.createElement("div");
    lab.className = "pn-label";
    lab.textContent = param.label;
    const input = document.createElement("input");
    input.type = "range";
    input.className = "pn-slider";
    input.min = param.min;
    input.max = param.max;
    input.step = param.step || 1;
    input.value = value;
    tagParam(input, param.key, "slider");
    // ช่องค่า = input ตัวเลข → คลิกพิมพ์ค่าเองได้ + sync กับสไลเดอร์สองทาง
    const box = document.createElement("input");
    box.type = "number";
    box.className = "pn-value mono";
    box.min = param.min;
    box.max = param.max;
    box.step = param.step || 1;
    box.value = value;
    tagParam(box, param.key, "box");
    // ↺ คืนค่าเริ่มต้น — โชว์เมื่อค่าต่างจาก default (ซ่อนด้วย visibility เพื่อไม่ให้ช่องค่าขยับ)
    const reset = document.createElement("button");
    reset.type = "button";
    reset.className = "pn-reset";
    reset.textContent = "↺";
    reset.setAttribute("aria-label", `คืนค่าเริ่มต้นของ ${param.label} (${defVal})`);
    reset.title = `คืนค่าเริ่มต้น (${defVal})`;
    const syncReset = () => {
      const off = !differsFromDefault(input.value, defVal);
      reset.disabled = off;
      reset.classList.toggle("is-off", off);
      reset.setAttribute("aria-hidden", off ? "true" : "false");
    };
    reset.addEventListener("click", () => {
      const d = Number(defVal);
      input.value = d;
      box.value = d;
      notifyParam(param.key, d);
      box.focus(); // ปุ่ม ↺ ปิดตัวเองหลังคืนค่า — ย้ายโฟกัสไปช่องค่า ไม่ให้หลุดไป BODY
      syncReset();
    });
    // เลื่อนสไลเดอร์ → อัปเดตช่องเลข
    input.addEventListener("input", () => {
      const v = Number(input.value);
      box.value = v;
      notifyParam(param.key, v);
      syncReset();
    });
    // พิมพ์เลขในช่อง → อัปเดตสไลเดอร์+ภาพ (clamp ค่าที่ส่งออก แต่ "ไม่แตะช่อง"
    // ระหว่างพิมพ์ ไม่งั้นค่าที่ต่ำกว่า min จะโดนเด้งจนพิมพ์เลขหลายหลักไม่ได้ — snap ช่องตอน commit)
    const applyTyped = () => {
      if (box.value === "" || box.value === "-") return; // กำลังพิมพ์ ยังไม่ครบ
      const v = Number(box.value);
      if (Number.isNaN(v)) return;
      const c = Math.min(param.max, Math.max(param.min, v));
      input.value = c;
      notifyParam(param.key, c);
      syncReset();
    };
    box.addEventListener("input", applyTyped);
    // ออกจากช่อง/กด Enter → snap ค่าให้อยู่ในช่วงและแสดงผลจริง
    const commit = () => {
      if (box.value === "" || box.value === "-" || Number.isNaN(Number(box.value))) {
        box.value = input.value; // เว้นว่าง/ค่าพัง → คืนค่าเดิม ไม่เด้งเป็น min
        return;
      }
      const v = Math.min(param.max, Math.max(param.min, Number(box.value)));
      box.value = v;
      input.value = v;
      notifyParam(param.key, v);
      syncReset();
    };
    box.addEventListener("change", commit);
    box.addEventListener("keydown", (e) => { if (e.key === "Enter") { commit(); box.blur(); } });
    // min/max ที่ปลายสไลเดอร์ (mono, สีจาง)
    const range = document.createElement("div");
    range.className = "pn-range mono";
    const rmin = document.createElement("span");
    rmin.textContent = String(param.min);
    const rmax = document.createElement("span");
    rmax.textContent = String(param.max);
    range.append(rmin, rmax);
    syncReset();
    wrap.append(lab, box, reset, input, range);
    return wrap;
  }

  // =================================================================
  // U3 — บล็อก "ค่าที่วัดได้"
  // =================================================================

  // สร้างบล็อกและตั้ง builtSig ให้ตรงกับ note ชุดปัจจุบัน (เรียกจาก render() เท่านั้น)
  function buildValuesBlock(step, notes) {
    // เปลี่ยน filter หรือเปลี่ยน "วิธี" ในกล่องรวม → ล้างสถานะที่ผู้ใช้ตั้งไว้
    // มุมมองเดิม (เช่น re-render จากการปรับ param ธรรมดา) → คงไว้ตามสเปก §5
    const viewId = viewIdOf(step);
    if (viewId !== lastValuesViewId) {
      lastValuesViewId = viewId;
      frozen = false;
      frozenSnap = null;
      shownLimit = ROWS_DEFAULT;
      showAllCols = false;
      lastValues = null;
      lastSeenAt = 0;
    }
    const mine = notesOfStep(step, notes);
    builtSig = structSig(mine);
    panelInert = false; // ตั้งแต่นี้ applyNotes เทียบ signature ได้จริง

    // ค่าที่ค้างจาก "วิธีเก่า" ไม่มีทางเข้ามาที่นี่ — notesOfStep ทิ้งด้วย view tag แล้ว (เดิมต้องเฝ้าเฟรมค้างด้วยเวลา)
    // ไม่มี values เฟรมนี้ → คง lastValues ไว้ (op ที่มีสาขา early-return เช่น psnr/ssim ตอนยังไม่ต่อ "เข้า B")
    const v = firstValues(mine);
    if (v) captureValues(v);
    // ไม่มีค่าเฟรมนี้แต่เคยมี → คงบล็อกไว้ (กันแผงกระโดด) แล้วเข้าสถานะ "รอค่า"
    const shape = v || lastValues;
    if (!shape) return;

    const dup = mine.reduce((a, n) => a + (n.values ? 1 : 0), 0);
    const view = buildValues(shape, dup);
    if (!view) return; // kind ที่ไม่รู้จัก → ไม่สร้างบล็อก ไม่ throw
    container.appendChild(view.root);
    valuesView = view;
    paintNow(frozen && frozenSnap ? frozenSnap : lastValues);
  }

  // ทาสีทันที ข้าม throttle (ใช้ตอนสร้างบล็อก / ปลดตรึง / เพิ่มแถว / สลับคอลัมน์)
  function paintNow(v) {
    if (pendingTimer) { clearTimeout(pendingTimer); pendingTimer = null; }
    pendingMine = null;
    lastPaint = Date.now();
    if (valuesView && v) valuesView.paint(v);
    updateWaiting();
  }

  function updateWaiting() {
    if (!valuesView) return;
    valuesView.setWaiting(!frozen && lastSeenAt > 0 && Date.now() - lastSeenAt > WAIT_MS);
  }

  // สร้าง DOM ของบล็อกครั้งเดียว — คืน { root, paint(values), setWaiting(on) }
  function buildValues(v, dupCount) {
    const kind = v && v.kind;
    if (kind !== "table" && kind !== "scalars") return null;

    const root = document.createElement("section");
    root.className = frozen ? "dv is-frozen" : "dv";
    root.setAttribute("data-dv-kind", kind);

    const head = document.createElement("div");
    head.className = "dv-head";
    const title = document.createElement("span");
    title.className = "dv-title";
    title.textContent = "ค่าที่วัดได้";
    head.appendChild(title);

    const hint = document.createElement("div");
    hint.className = "dv-hint";

    const view = { root, kind, cols: [], metas: [], trs: [], prev: [], waiting: false };

    view.syncHint = function () {
      const t = frozen
        ? "ตรึงค่าไว้อยู่ ตัวเลขจะไม่อัปเดตจนกว่าจะกดอีกครั้ง"
        : view.waiting
          ? "กำลังรอค่าจากเฟรมถัดไป… (ถ้าค้างนาน ให้ตรวจว่ากล่องนี้ต่อสายถึงกล่องผลลัพธ์แล้ว)"
          : "ตัวเลขที่เครื่องมือนี้วัดได้จากภาพเฟรมล่าสุด";
      if (hint.textContent !== t) hint.textContent = t;
    };
    view.setWaiting = function (on) {
      if (view.waiting === on) return;
      view.waiting = on;
      root.classList.toggle("is-waiting", on);
      view.syncHint();
    };

    const freezeBtn = document.createElement("button");
    freezeBtn.type = "button";
    freezeBtn.className = "dv-btn dv-freeze";
    freezeBtn.title = "หยุดตัวเลขไว้ชั่วคราวเพื่ออ่านให้ทัน — ภาพและการประมวลผลยังทำงานตามปกติ";
    const syncFreezeBtn = () => {
      freezeBtn.setAttribute("aria-pressed", frozen ? "true" : "false");
      freezeBtn.textContent = frozen ? "📌 ตรึงอยู่ — กดเพื่อเล่นต่อ" : "📌 ตรึงค่า";
    };
    syncFreezeBtn();
    freezeBtn.addEventListener("click", () => {
      frozen = !frozen;
      frozenSnap = frozen ? lastValues : null;
      root.classList.toggle("is-frozen", frozen);
      syncFreezeBtn();
      view.syncHint();
      if (!frozen) paintNow(lastValues); // ปลดตรึง → ทาค่าล่าสุดทันที ไม่รอ throttle
      else updateWaiting();              // ตรึงอยู่ → ไม่ต้องขึ้น "กำลังรอค่า"
    });

    root.appendChild(head);
    root.appendChild(hint);
    if (dupCount > 1) {
      const d = document.createElement("div");
      d.className = "dv-dup";
      d.textContent = "⚠️ มีกล่องชนิดเดียวกันมากกว่า 1 กล่องที่ต่อจอผลลัพธ์ — ตารางนี้แสดงค่าจากกล่องแรก";
      root.appendChild(d);
    }

    if (kind === "table") buildTableBody(view, v, head);
    else buildScalarBody(view, v);

    head.appendChild(freezeBtn);
    view.syncHint();
    return view;
  }

  // ---- kind: "table" ----
  function buildTableBody(view, v, head) {
    const cols = Array.isArray(v.columns) ? v.columns.slice() : [];
    view.cols = cols;
    view.metas = cols.map(metaOf);
    // จำนวนแถว — มีเฉพาะ kind "table" จึงสร้างที่นี่ (ฝั่ง scalars ไม่ต้องมี)
    const countEl = document.createElement("span");
    countEl.className = "dv-count mono";
    // ลำดับใน .dv-head: ชื่อ → จำนวนแถว → ปุ่มสลับคอลัมน์ → ปุ่มตรึง (freezeBtn ถูก append ทีหลังใน buildValues)
    head.appendChild(countEl);

    const scroll = document.createElement("div");
    scroll.className = "dv-scroll";
    const table = document.createElement("table");
    table.className = "dv-table mono" + (showAllCols ? " show-all" : "");
    const thead = document.createElement("thead");
    const htr = document.createElement("tr");
    cols.forEach((key, i) => {
      const m = view.metas[i];
      const th = document.createElement("th");
      th.scope = "col";
      th.className = "dv-th";
      th.textContent = m.th || String(key);
      // key ที่ไม่รู้จักไม่มี tip → เอา key ดิบเป็น tooltip แทน หัวคอลัมน์ที่ยาวเกินช่องจะถูกตัดด้วย
      // text-overflow (ดู .dv-th ใน app.css) ผู้ใช้จึงต้องอ่านชื่อเต็มได้จากการเอาเมาส์ชี้เสมอ
      th.title = m.tip || String(key);
      // ความกว้างตายตัวคู่กับ table-layout:fixed → คอลัมน์ไม่ขยับแม้จำนวนหลักเปลี่ยน
      th.style.width = (m.w || 88) + "px";
      if (!m.pri) th.setAttribute("data-dv-secondary", "");
      htr.appendChild(th);
    });
    thead.appendChild(htr);
    table.appendChild(thead);
    const tbody = document.createElement("tbody");
    table.appendChild(tbody);
    scroll.appendChild(table);
    view.table = table;
    view.tbody = tbody;

    // ปุ่มสลับคอลัมน์ — มีเฉพาะตอนที่มีคอลัมน์รองให้สลับจริง ๆ
    if (view.metas.some((m) => !m.pri)) {
      const colsBtn = document.createElement("button");
      colsBtn.type = "button";
      colsBtn.className = "dv-btn dv-cols";
      colsBtn.title = "ตารางนี้มีคอลัมน์มากกว่าที่เห็น กดเพื่อดูครบทุกคอลัมน์";
      const sync = () => {
        colsBtn.setAttribute("aria-pressed", showAllCols ? "true" : "false");
        colsBtn.textContent = showAllCols ? "⇄ เฉพาะที่สำคัญ" : "⇄ ทุกคอลัมน์";
      };
      sync();
      colsBtn.addEventListener("click", () => {
        showAllCols = !showAllCols;
        table.classList.toggle("show-all", showAllCols); // สลับคลาสอย่างเดียว ไม่สร้าง DOM ใหม่
        sync();
        paintNow(frozen && frozenSnap ? frozenSnap : lastValues);
      });
      head.appendChild(colsBtn);
    }

    const empty = document.createElement("div");
    empty.className = "dv-empty";
    empty.hidden = true;
    empty.textContent = "ยังไม่พบวัตถุในภาพนี้ — ลองลดค่า พื้นที่ขั้นต่ำ ด้านล่าง หรือทำ Threshold ให้ภาพเป็นขาว-ดำก่อน";
    const foot = document.createElement("div");
    foot.className = "dv-foot";
    const footTxt = document.createElement("span");
    footTxt.className = "dv-foot-txt";
    const moreBtn = document.createElement("button");
    moreBtn.type = "button";
    moreBtn.className = "dv-more";
    moreBtn.hidden = true;
    moreBtn.textContent = "แสดงเพิ่มอีก " + ROWS_STEP + " แถว";
    moreBtn.addEventListener("click", () => {
      shownLimit = Math.min(ROWS_MAX, shownLimit + ROWS_STEP);
      growRows(view, shownLimit); // ที่เดียวที่สร้าง DOM นอก render()
      paintNow(frozen && frozenSnap ? frozenSnap : lastValues);
    });
    foot.appendChild(footTxt);
    foot.appendChild(moreBtn);

    view.root.appendChild(scroll);
    view.root.appendChild(empty);
    view.root.appendChild(foot);
    growRows(view, Math.min(shownLimit, ROWS_MAX));

    view.paint = function (val) {
      const rows = Array.isArray(val.rows) ? val.rows : [];
      const nc = view.cols.length;
      const isEmpty = rows.length === 0;
      if (scroll.hidden !== isEmpty) scroll.hidden = isEmpty;
      if (empty.hidden === isEmpty) empty.hidden = !isEmpty;
      if (foot.hidden !== isEmpty) foot.hidden = isEmpty;

      const shown = Math.min(rows.length, view.trs.length);
      for (let r = 0; r < view.trs.length; r++) {
        const slot = view.trs[r];
        if (r >= shown) {
          if (!slot.tr.hidden) slot.tr.hidden = true;
          continue;
        }
        if (slot.tr.hidden) slot.tr.hidden = false;
        const data = Array.isArray(rows[r]) ? rows[r] : [];
        const base = r * nc;
        for (let c = 0; c < nc; c++) {
          const td = slot.tds[c];
          const cell = fmtVal(data[c], view.metas[c]);
          if (view.prev[base + c] !== cell.t) {
            td.textContent = cell.t;
            view.prev[base + c] = cell.t;
          }
          const cls = cell.c ? "dv-td " + cell.c : "dv-td";
          if (td.className !== cls) td.className = cls;
          if (td.title !== cell.ti) td.title = cell.ti;
        }
      }

      const cnt = fmtInt(rows.length) + " แถว";
      if (countEl.textContent !== cnt) countEl.textContent = cnt;

      // ข้อความท้ายตารางต้องไม่โกหก — แยก "ที่แสดงบนจอ" / "ที่ worker ส่งมา" / "ที่พบจริงทั้งหมด"
      const total = typeof val.total === "number" ? val.total : rows.length;
      const capped = total > rows.length; // param "แสดงวัตถุสูงสุด" ตัดมาตั้งแต่ฝั่ง worker
      let txt;
      if (shown < rows.length) {
        txt = shownLimit >= ROWS_MAX
          ? `แสดงได้สูงสุด ${ROWS_MAX} แถวบนหน้าจอ (ค่าทั้งหมดยังถูกคำนวณครบ) — จากที่ส่งมา ${fmtInt(rows.length)} แถว`
          : `แสดง ${fmtInt(shown)} จาก ${fmtInt(rows.length)} แถว`;
      } else {
        txt = `แสดงครบ ${fmtInt(rows.length)} แถว`;
      }
      if (capped) txt += ` · พบวัตถุจริงทั้งหมด ${fmtInt(total)} ชิ้น (ค่า "แสดงวัตถุสูงสุด" จำกัดให้วัดละเอียดแค่ ${fmtInt(rows.length)} ชิ้นแรก)`;
      if (footTxt.textContent !== txt) footTxt.textContent = txt;

      const showMore = shown < rows.length && shownLimit < ROWS_MAX;
      if (moreBtn.hidden === showMore) moreBtn.hidden = !showMore;
    };
  }

  // สร้างแถวใน pool เพิ่มจนครบ n แถว (แถวส่วนเกินถูกซ่อนไว้ ไม่ต้องสร้างใหม่ตอนจำนวนวัตถุกระพริบ)
  function growRows(view, n) {
    const cap = Math.min(n, ROWS_MAX);
    while (view.trs.length < cap) {
      const tr = document.createElement("tr");
      tr.className = "dv-tr";
      tr.hidden = true;
      const tds = [];
      for (let c = 0; c < view.cols.length; c++) {
        const td = document.createElement("td");
        td.className = "dv-td";
        if (!view.metas[c].pri) td.setAttribute("data-dv-secondary", "");
        tr.appendChild(td);
        tds.push(td);
      }
      view.tbody.appendChild(tr);
      view.trs.push({ tr, tds });
    }
  }

  // ---- kind: "scalars" ----
  function buildScalarBody(view, v) {
    const fields = Array.isArray(v.fields) ? v.fields : [];
    const list = document.createElement("div");
    list.className = "dv-list";
    view.items = fields.map((f) => {
      const key = f ? f.key : undefined;
      const m = metaOf(key);
      const row = document.createElement("div");
      row.className = "dv-row";
      const k = document.createElement("span");
      k.className = "dv-k";
      // ชื่อ: label จาก worker ก่อนเสมอ (ไทย + มีหน่วยแล้ว) → VAL_META → key ดิบ
      k.textContent = (f && f.label) || m.th || (key !== undefined ? String(key) : "—");
      if (m.tip) k.title = m.tip;
      const val = document.createElement("span");
      val.className = "dv-v mono";
      row.appendChild(k);
      row.appendChild(val);
      list.appendChild(row);
      return { el: val, meta: m };
    });
    view.root.appendChild(list);

    view.paint = function (data) {
      const fs = Array.isArray(data.fields) ? data.fields : [];
      for (let i = 0; i < view.items.length; i++) {
        const it = view.items[i];
        const f = fs[i];
        const cell = fmtVal(f ? f.value : undefined, it.meta);
        if (view.prev[i] !== cell.t) {
          it.el.textContent = cell.t;
          view.prev[i] = cell.t;
        }
        const cls = cell.c ? "dv-v mono " + cell.c : "dv-v mono";
        if (it.el.className !== cls) it.el.className = cls;
        if (it.el.title !== cell.ti) it.el.title = cell.ti;
      }
    };
  }

  // ---- อัปเดตในที่ (ถูกเรียกทุกเฟรมจาก app.js) ----

  // คืน true = จัดการเองแล้ว / false = โครงเปลี่ยน caller ต้องเรียก renderDetail()
  function applyNotes(notes) {
    if (!curStep || panelInert) return true;
    const mine = notesOfStep(curStep, notes);
    // เก็บ "ตัวที่กรองแล้ว" — rerenderAfterParamChange() ภายในไฟล์นี้ส่ง curNotes ต่อเข้า render()
    // ซึ่งคาดหวัง note ของ filter นี้เท่านั้น
    curNotes = mine;
    if (structSig(mine) !== builtSig) return false;
    schedulePaint(mine);
    return true;
  }

  function schedulePaint(mine) {
    const t = Date.now();
    const wait = PAINT_MS - (t - lastPaint);
    if (wait <= 0) {
      if (pendingTimer) { clearTimeout(pendingTimer); pendingTimer = null; }
      pendingMine = null;
      lastPaint = t;
      doPaint(mine);
      return;
    }
    // trailing: โหมดภาพนิ่ง/กล้องหยุด พึ่ง timeout ตัวนี้ล้วน ๆ ให้ค่าสุดท้ายถูกทาเสมอ
    pendingMine = mine;
    if (!pendingTimer) {
      pendingTimer = setTimeout(() => {
        pendingTimer = null;
        const m = pendingMine;
        pendingMine = null;
        if (!m) return;
        lastPaint = Date.now();
        doPaint(m);
      }, wait);
    }
  }

  function doPaint(mine) {
    paintInfo(mine);
    paintValues(firstValues(mine));
  }

  // ข้อความ 📊 — เขียนทับเฉพาะเมื่อเปลี่ยนจริง (จำนวน/ลำดับคงที่เพราะ signature ตรงกันแล้ว)
  function paintInfo(mine) {
    if (infoEls.length === 0) return;
    let i = 0;
    for (const n of mine) {
      if (!n.info) continue;
      const el = infoEls[i++];
      if (!el) break;
      setInfo(el, n.info);
    }
  }

  function paintValues(v) {
    if (v) captureValues(v);
    if (!valuesView) return;
    if (v && !frozen) valuesView.paint(v);
    updateWaiting();
  }

  // อัปเดตเฉพาะข้อความ note-info ในที่ (app.js เรียกตอนผู้ใช้กำลังลากสไลเดอร์ ซึ่งห้าม rebuild แผง)
  function updateInfo(infos) {
    infoEls.forEach((el, i) => { if (infos[i] != null) setInfo(el, infos[i]); });
  }

  // วิธี (view) ของ step ที่แผงถือไม่ตรงกับ step ปัจจุบัน และไม่ใช่แผงนี้เองที่เปลี่ยน → ถูกเปลี่ยนจากนอกแผง
  // (เช่นเมนู Properties ของ LiteGraph) — caller ต้อง renderDetail() ไม่งั้น note ใหม่ถูกกรองทิ้งหมดแล้วแผงไม่ rebuild
  function isViewStale(step) {
    return !ownChange && !!curStep && !!step && viewIdOf(curStep) !== viewIdOf(step);
  }

  return { render, renderDisplay, renderDL, applyNotes, updateInfo, isViewStale };
}

// UI metadata — เพิ่มความสวย/สื่อความ โดยไม่แตะ backend algorithm

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
};

// "บรรทัด 2" ของการ์ด node บนกระดาน (Design node-card §1) — สรุปค่าหลักที่กำลังตั้งไว้
// รับ local params (sub-params ของ op ที่เลือก ตัด prefix opId__ ออกแล้ว)
// key ตรวจสอบกับ src/renderer/filters/registry.js จริงแล้ว (ไม่ใช่ชื่อที่เดาไว้)
export const OP_PRIMARY = {
  grayscale: () => "BGR → เทา (Gray)",
  colorspace: (p) => "BGR → " + (p.space || "HSV").toUpperCase(),
  channel_split: (p) => "ช่อง: " + (p.channel || "R").toUpperCase(),
  brightness_contrast: (p) => `คอนทราสต์ ${p.alpha ?? 1.2} · สว่าง ${p.beta ?? 0}`,
  gamma: (p) => "gamma: " + (p.g ?? 1.5),
  hist_equalize: () => "ปรับฮิสโตแกรมทั้งภาพ",
  clahe: (p) => "clip: " + (p.clip ?? 2),
  unsharp: (p) => "ความเข้ม: " + (p.amount ?? 1),
  laplacian_sharpen: (p) => "ความเข้ม: " + (p.strength ?? 1),
  salt_pepper_noise: (p) => "noise: " + (p.amount ?? 5) + "%",
  median_blur: (p) => "ksize: " + (p.ksize ?? 3),
  gaussian_blur: (p) => "ksize: " + (p.ksize ?? 5),
  box_blur: (p) => "ksize: " + (p.ksize ?? 5),
  bilateral: (p) => "d: " + (p.d ?? 7),
  morphology: (p) => (p.op || "erode") + " · k=" + (p.ksize ?? 5),
  binary_threshold: (p) => "thresh: " + (p.thresh ?? 127),
  otsu_threshold: () => "วิธี: Otsu (อัตโนมัติ)",
  adaptive_threshold: (p) => "block: " + (p.blockSize ?? 11),
  canny: (p) => `ขอบ ${p.t1 ?? 50}–${p.t2 ?? 150}`,
  sobel: (p) => "ทิศ: " + String(p.dir || "x").toUpperCase(),
  find_contours: () => "หาเส้นรอบวัตถุ",
  connected_components: () => "นับ/ระบายวัตถุ",
  watershed: () => "แยกวัตถุติดกัน",
  bounding_boxes: () => "ล้อมกรอบวัตถุ",
  hough_circles: (p) => `รัศมี ${p.minR ?? 0}–${p.maxR ?? 0}`,
  grabcut: () => "ตัดพื้นหลัง",
  color_inrange: (p) => "Hue: " + (p.hMin ?? 35) + "–" + (p.hMax ?? 85),
  invert: () => "กลับสี",
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

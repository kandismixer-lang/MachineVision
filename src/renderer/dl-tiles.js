// ตรวจแบบแบ่ง tile (sliced inference / SAHI) — ฟังก์ชันล้วน ไม่แตะ DOM (เทสต์ด้วย node ได้)
// ใช้กับ defect เล็ก ๆ บนภาพใหญ่: ถ้าบีบภาพเต็มลงขนาด input ของโมเดล (เช่น 384) วัตถุจะเหลือไม่กี่พิกเซล
// → หั่นภาพเป็นชิ้น ๆ ให้เหลื่อมกัน ตรวจทีละชิ้น แล้วรวมกล่องกลับเป็นพิกัดภาพเต็ม

// ตำแหน่งเริ่มของ tile ตามแกนเดียว: ก้าวทีละ tile×(1−overlap) และชิ้นสุดท้ายชิดขอบพอดี (ไม่ล้นภาพ)
// length ≤ tile → ชิ้นเดียวที่ 0
export function tilePositions(length, tile, overlap) {
  const L = Math.max(1, Math.round(length));
  const T = Math.max(1, Math.round(tile));
  if (L <= T) return [0];
  const ov = Math.min(0.9, Math.max(0, Number(overlap) || 0));
  const step = Math.max(1, Math.round(T * (1 - ov)));
  const out = [];
  for (let p = 0; p + T < L; p += step) out.push(p);
  if (out[out.length - 1] !== L - T) out.push(L - T);
  return out;
}

// กรอบ tile ทั้งหมดของภาพ W×H — ด้านที่สั้นกว่า tile ใช้ความยาวเท่าภาพ (ไม่เติมขอบ)
export function tileGrid(W, H, tile, overlap) {
  const tw = Math.min(Math.round(tile), Math.round(W));
  const th = Math.min(Math.round(tile), Math.round(H));
  const rects = [];
  for (const y of tilePositions(H, tile, overlap)) {
    for (const x of tilePositions(W, tile, overlap)) rects.push({ x, y, w: tw, h: th });
  }
  return rects;
}

// intersection / พื้นที่กล่องที่เล็กกว่า — กล่องครึ่งเดียวที่โดนขอบ tile ตัด (อยู่ในกล่องเต็ม) ได้ ~1
// ส่วน IoU ธรรมดาจะต่ำ (กล่องเล็กกว่ามาก) จึงไม่ถูกตัดทิ้ง
function intersectOverSmaller(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const small = Math.min(a.w * a.h, b.w * b.h);
  return small > 0 ? (ix * iy) / small : 0;
}

// รวมกล่องจากทุก tile: เรียงคะแนนมาก→น้อย เก็บกล่องที่ไม่ทับกล่องที่เก็บแล้ว (คลาสเดียวกัน) เกิน thr
export function mergeTileBoxes(boxes, thr = 0.5) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of sorted) {
    if (keep.every((k) => k.cls !== b.cls || intersectOverSmaller(k, b) < thr)) keep.push(b);
  }
  return keep;
}

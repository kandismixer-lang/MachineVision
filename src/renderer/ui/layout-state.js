// สถานะเลย์เอาต์ที่เก็บถาวร (ตรรกะล้วน ไม่แตะ DOM — เทสต์ได้ใน node) — ใช้โดย layout-resize.js
// key "mv-layout-v1": { v:1, paletteW, propsW, stageRatio, paletteHidden?, propsHidden? }
// ฟิลด์ซ่อนแผงเป็นของเสริม: ค่าเก่าที่ไม่มีฟิลด์นี้ยังโหลดได้ (= แสดงแผง) · ชนิดผิดของฟิลด์ซ่อน = ถือว่าไม่ซ่อน (ไม่ทิ้งทั้งก้อน)
export const KEY = "mv-layout-v1";
export const LIMITS = {
  palette: { min: 200, max: 440 },
  props: { min: 280, max: 560 },
  centerMinW: 480,
  stageMinH: 180,
  graphMinH: 150,
};
export const RATIO = { min: 0.1, max: 0.9 };
// ความกว้างแถบพับ (rail) ที่เหลือเมื่อซ่อนแผง — ต้องตรงกับ --rail-w ใน app.css
export const RAIL_W = 32;

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export function emptyState() {
  return { paletteW: null, propsW: null, stageRatio: null, paletteHidden: false, propsHidden: false };
}

// อ่านจาก localStorage — ทุกทางพังได้ → เงียบ ใช้ค่าเริ่มต้น (ค่าเสียทั้งก้อน = ลบทิ้ง)
export function load(storage = globalThis.localStorage) {
  let raw = null;
  try { raw = storage.getItem(KEY); } catch { return emptyState(); }
  if (raw == null) return emptyState();
  try {
    const o = JSON.parse(raw);
    if (!o || typeof o !== "object" || o.v !== 1) throw new Error("bad");
    const num = (v, lo, hi) => {
      if (v == null) return null;
      if (typeof v !== "number" || !Number.isFinite(v)) throw new Error("bad");
      return clamp(v, lo, hi);
    };
    return {
      paletteW: num(o.paletteW, LIMITS.palette.min, LIMITS.palette.max),
      propsW: num(o.propsW, LIMITS.props.min, LIMITS.props.max),
      stageRatio: num(o.stageRatio, RATIO.min, RATIO.max),
      paletteHidden: o.paletteHidden === true,
      propsHidden: o.propsHidden === true,
    };
  } catch {
    try { storage.removeItem(KEY); } catch { /* ignore */ }
    return emptyState();
  }
}

// ไม่มีอะไรต่างจากค่าเริ่มต้น → ลบ key (ไม่ทิ้งขยะ) · ไม่งั้นเขียนฟิลด์ซ่อนเฉพาะเมื่อ true (ไฟล์เก่า/ใหม่ปะปนกันได้)
export function save(s, storage = globalThis.localStorage) {
  try {
    const hidden = !!(s.paletteHidden || s.propsHidden);
    if (s.paletteW == null && s.propsW == null && s.stageRatio == null && !hidden) storage.removeItem(KEY);
    else {
      const o = { v: 1, paletteW: s.paletteW, propsW: s.propsW, stageRatio: s.stageRatio };
      if (s.paletteHidden) o.paletteHidden = true;
      if (s.propsHidden) o.propsHidden = true;
      storage.setItem(KEY, JSON.stringify(o));
    }
  } catch { /* ignore */ }
}

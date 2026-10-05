// Single source of truth ของ dataset สำหรับระบบ annotation/labeling (สไตล์ Roboflow)
// Phase 1: เก็บแค่ "โครงข้อมูล" — คลาส + annotation ต่อภาพ + train/valid/test split
// ยังไม่มี UI ตีกรอบ/export YOLO/auto-label (ดูสเปก Phase ถัดไป) — ไฟล์นี้เป็น pure module ไม่แตะ DOM เลย
//
// โครงสร้างข้อมูล:
//   datasetClasses: [{ name, color }]
//     - index ในอาเรย์นี้ "คือ" classIdx ที่ YOLO ใช้ตรง ๆ (ไม่มี id แยกต่างหาก)
//   images: { [imageKey]: { annotations: [{classIdx, cx, cy, w, h}], split } }
//     - คีย์ด้วย "ชื่อไฟล์ภาพ" (เช่น state.images[i].name) เพราะ derive ได้จากไฟล์เสมอ
//       ต่างจาก imageIndex ที่ไม่เสถียรข้าม session (ภาพไม่ persist canvas ต้อง import ใหม่ทุกครั้ง)
//       และชื่อภาพตรงกับชื่อไฟล์ label ตอน export YOLO (basename เดียวกัน) พอดี
//     - ⚠ ข้อจำกัด Phase 1: ถ้ามีภาพชื่อซ้ำกัน (แคปซ้ำชื่อ/import จากคนละโฟลเดอร์) จะถูกมองเป็นภาพเดียวกัน
//       ยังไม่ทำ de-dupe ในเฟสนี้ (ตั้งใจเลื่อนไปแก้ในเฟสถัดไปตามสเปก)
//     - cx, cy, w, h = normalized 0..1 แบบ center-based (มาตรฐาน YOLO) เก็บ normalized ตั้งแต่จุดสร้างเลย
//       เพื่อให้ export ในเฟสถัดไปใช้ค่าตรง ๆ ได้โดยไม่ต้องแปลงซ้ำ

const STORAGE_KEY = "mv-dataset";
const SCHEMA_VERSION = 1;
const VALID_SPLITS = new Set(["train", "valid", "test"]);
const DEFAULT_SPLIT = "train";
const SAVE_DEBOUNCE_MS = 200;

// สีเริ่มต้นไล่ตามลำดับตอน addClass ไม่ระบุสีมาเอง (วนซ้ำเมื่อคลาสเกินจำนวนนี้)
const DEFAULT_PALETTE = [
  "#3b82f6", "#ef4444", "#22c55e", "#eab308", "#a855f7",
  "#06b6d4", "#f97316", "#ec4899", "#84cc16", "#14b8a6",
];

let datasetClasses = []; // [{ name, color }]
let images = Object.create(null); // imageKey -> { annotations, split }
const listeners = new Set();
let saveTimer = null;

// ---------- helper เล็ก ๆ ----------
function clamp01(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.min(1, Math.max(0, n));
}

function normalizeSplit(s) {
  return VALID_SPLITS.has(s) ? s : DEFAULT_SPLIT;
}

function emptyImageRecord() {
  return { annotations: [], split: DEFAULT_SPLIT };
}

// คืน record ของภาพ (สร้างใหม่ถ้ายังไม่มี) — ต้องมี imageKey เสมอ (ดู note เรื่อง key ด้านบน)
function ensureImageRecord(imageKey) {
  if (!imageKey || typeof imageKey !== "string") {
    throw new Error("dataset/annotations: imageKey ต้องเป็นชื่อไฟล์ภาพ (string ไม่ว่าง)");
  }
  let rec = images[imageKey];
  if (!rec) {
    rec = emptyImageRecord();
    images[imageKey] = rec;
  }
  return rec;
}

function notifyChange() {
  for (const cb of listeners) {
    try {
      cb();
    } catch (err) {
      console.error("mv-dataset: onChange callback error", err);
    }
  }
}

// เขียนลง localStorage จริง ๆ (เรียกผ่าน scheduleSave หรือ save() เท่านั้น)
function persist() {
  try {
    const payload = { version: SCHEMA_VERSION, classes: datasetClasses, images };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch (err) {
    console.error("mv-dataset: บันทึกลง localStorage ไม่สำเร็จ", err);
  }
}

// debounce การเขียนดิสก์ (เรียกถี่ตอนลาก/พิมพ์ได้โดยไม่กระตุก) แต่แจ้ง UI ทันทีทุกครั้ง
function scheduleSave() {
  notifyChange();
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    saveTimer = null;
    persist();
  }, SAVE_DEBOUNCE_MS);
}

// บันทึกทันที (flush การ debounce ที่ค้างอยู่) — ให้ผู้เรียกภายนอกเรียกเองได้ถ้าต้องการความชัวร์
export function save() {
  if (saveTimer) {
    clearTimeout(saveTimer);
    saveTimer = null;
  }
  persist();
}

function load() {
  let raw;
  try {
    raw = localStorage.getItem(STORAGE_KEY);
  } catch {
    raw = null;
  }
  if (!raw) return;
  try {
    const data = JSON.parse(raw);
    if (!data || typeof data !== "object") return;

    datasetClasses = Array.isArray(data.classes)
      ? data.classes
          .filter((c) => c && typeof c.name === "string")
          .map((c, i) => ({
            name: c.name,
            color: typeof c.color === "string" && c.color ? c.color : DEFAULT_PALETTE[i % DEFAULT_PALETTE.length],
          }))
      : [];

    const rawImages = data.images && typeof data.images === "object" ? data.images : {};
    const next = Object.create(null);
    for (const key of Object.keys(rawImages)) {
      const im = rawImages[key];
      if (!im || typeof im !== "object") continue;
      const anns = Array.isArray(im.annotations) ? im.annotations : [];
      next[key] = {
        split: normalizeSplit(im.split),
        annotations: anns
          .filter((a) => a && Number.isInteger(a.classIdx) && a.classIdx >= 0 && a.classIdx < datasetClasses.length)
          .map((a) => {
            // ข้ามตัวที่ points เพี้ยน (จุดน้อยเกิน) แทนที่จะทำ load ทั้งก้อนล้ม — คืน null แล้วกรองออก
            try {
              return validateAnnotationInput(a);
            } catch {
              return null;
            }
          })
          .filter(Boolean),
      };
    }
    images = next;
  } catch (err) {
    console.error("mv-dataset: อ่านข้อมูลเดิมจาก localStorage ไม่สำเร็จ", err);
  }
}

load(); // โหลดอัตโนมัติตอน import module (ตามสเปก)

// ---------- subscribe การเปลี่ยนแปลง (ให้ UI เช่นแผงคลาส re-render) ----------
// คืนฟังก์ชัน unsubscribe — ผู้เรียกต้องเก็บไว้เรียกตอนเลิกใช้ panel กันลิสเนอร์ค้าง
export function onChange(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

// ---------- คลาส ----------
export function getClasses() {
  return datasetClasses.map((c) => ({ name: c.name, color: c.color }));
}

export function addClass(name, color) {
  const trimmed = typeof name === "string" ? name.trim() : "";
  if (!trimmed) throw new Error("addClass: ต้องมีชื่อคลาส");
  const idx = datasetClasses.length;
  const c = typeof color === "string" && color ? color : DEFAULT_PALETTE[idx % DEFAULT_PALETTE.length];
  datasetClasses.push({ name: trimmed, color: c });
  scheduleSave();
  return idx;
}

export function updateClass(idx, patch = {}) {
  const cls = datasetClasses[idx];
  if (!cls) throw new Error(`updateClass: ไม่พบคลาสลำดับ ${idx}`);
  if (typeof patch.name === "string" && patch.name.trim()) cls.name = patch.name.trim();
  if (typeof patch.color === "string" && patch.color) cls.color = patch.color;
  scheduleSave();
}

// ลบคลาสที่ idx
// เหตุผลที่ต้อง "remap" ไม่ใช่แค่ลบ: classIdx = ตำแหน่งในอาเรย์ datasetClasses ตรง ๆ
// พอ splice(idx,1) ออก คลาสที่อยู่หลัง idx ทุกตัวจะเลื่อนตำแหน่งลดลง 1 ทันที
// ถ้าไม่ลด classIdx ของ annotation เดิมตาม จะเกิด "off-by-one" ชี้ผิดคลาสแบบเงียบ ๆ (บั๊กที่ตรวจจับยากมาก)
// ส่วน annotation ที่ชี้ "คลาสที่กำลังถูกลบพอดี" ไม่มีคลาสให้ remap ไปหา จึงต้องลบทิ้งเท่านั้น
export function removeClass(idx) {
  if (!Number.isInteger(idx) || idx < 0 || idx >= datasetClasses.length) {
    throw new Error(`removeClass: index ${idx} ไม่ถูกต้อง`);
  }
  datasetClasses.splice(idx, 1);
  for (const key of Object.keys(images)) {
    const rec = images[key];
    rec.annotations = rec.annotations
      .filter((a) => a.classIdx !== idx)
      .map((a) => (a.classIdx > idx ? { ...a, classIdx: a.classIdx - 1 } : a));
  }
  scheduleSave();
}

// ---------- annotation ต่อภาพ ----------
// รองรับ 3 ชนิด (Roboflow-style):
//   bbox    : { type:'bbox', classIdx, cx, cy, w, h }         — กรอบสี่เหลี่ยม (ชนิดเดิม)
//   polygon : { type:'polygon', classIdx, points:[{x,y}...] } — รูปปิด (ใช้ export YOLO-seg)
//   line    : { type:'line',    classIdx, points:[{x,y}...] } — เส้นเปิด (annotate อย่างเดียว)
// ข้อมูลเก่าที่ไม่มี field `type` = ถือเป็น 'bbox' (back-compat กับ dataset ที่บันทึกไว้ก่อนเฟสนี้)
const POLY_TYPES = new Set(["polygon", "line"]);

function normType(t) {
  if (t === "polygon" || t === "line") return t;
  return "bbox"; // undefined/อื่น ๆ = bbox (รวมข้อมูลเก่า)
}

function clampPoints(pts, minLen) {
  if (!Array.isArray(pts)) throw new Error("annotation: points ต้องเป็นอาเรย์");
  const out = pts
    .filter((p) => p && Number.isFinite(Number(p.x)) && Number.isFinite(Number(p.y)))
    .map((p) => ({ x: clamp01(p.x), y: clamp01(p.y) }));
  if (out.length < minLen) throw new Error(`annotation: ต้องมีจุดอย่างน้อย ${minLen} จุด`);
  return out;
}

function validateAnnotationInput(ann) {
  if (!ann || !Number.isInteger(ann.classIdx)) {
    throw new Error("annotation: classIdx ต้องเป็นจำนวนเต็ม");
  }
  if (ann.classIdx < 0 || ann.classIdx >= datasetClasses.length) {
    throw new Error(`annotation: classIdx ${ann.classIdx} ไม่มีอยู่ในรายการคลาส`);
  }
  const type = normType(ann.type);
  if (type === "bbox") {
    return {
      type: "bbox",
      classIdx: ann.classIdx,
      cx: clamp01(ann.cx),
      cy: clamp01(ann.cy),
      w: clamp01(ann.w),
      h: clamp01(ann.h),
    };
  }
  // polygon ต้องปิดรูปได้ (>=3 จุด), line เป็นเส้น (>=2 จุด)
  return {
    type,
    classIdx: ann.classIdx,
    points: clampPoints(ann.points, type === "polygon" ? 3 : 2),
  };
}

// clone annotation ให้ลึกพอ (points เป็นอาเรย์ ต้องก็อปกันผู้เรียกแก้ทับข้อมูลจริง)
function cloneAnnotation(a) {
  if (POLY_TYPES.has(a.type)) {
    return { type: a.type, classIdx: a.classIdx, points: a.points.map((p) => ({ x: p.x, y: p.y })) };
  }
  return { ...a };
}

export function getAnnotations(imageKey) {
  const rec = imageKey ? images[imageKey] : null;
  return rec ? rec.annotations.map(cloneAnnotation) : [];
}

// มี polygon สักอันไหมทั้ง dataset → งานเป็น segmentation (ใช้เลือกโมเดล -seg ตอน gen โค้ดเทรน)
export function hasPolygonAnnotations() {
  for (const key of Object.keys(images)) {
    if (images[key].annotations.some((a) => a.type === "polygon")) return true;
  }
  return false;
}

export function setAnnotations(imageKey, arr) {
  const rec = ensureImageRecord(imageKey);
  rec.annotations = Array.isArray(arr) ? arr.map(validateAnnotationInput) : [];
  scheduleSave();
}

export function addAnnotation(imageKey, ann) {
  const rec = ensureImageRecord(imageKey);
  const clean = validateAnnotationInput(ann);
  rec.annotations.push(clean);
  scheduleSave();
  return rec.annotations.length - 1;
}

export function removeAnnotation(imageKey, annIndex) {
  const rec = imageKey ? images[imageKey] : null;
  if (!rec || annIndex < 0 || annIndex >= rec.annotations.length) return;
  rec.annotations.splice(annIndex, 1);
  scheduleSave();
}

export function clearAnnotations(imageKey) {
  const rec = imageKey ? images[imageKey] : null;
  if (!rec) return;
  rec.annotations = [];
  scheduleSave();
}

// ---------- split (train/valid/test) ----------
export function getSplit(imageKey) {
  const rec = imageKey ? images[imageKey] : null;
  return rec ? rec.split : DEFAULT_SPLIT;
}

export function setSplit(imageKey, split) {
  const rec = ensureImageRecord(imageKey);
  rec.split = normalizeSplit(split);
  scheduleSave();
}

// สุ่ม shuffle แบบ Fisher–Yates (ไม่แก้ array เดิม)
function shuffled(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// แบ่ง keys ตามอัตราส่วน train/valid/test (ปัดเศษให้ผลรวม = จำนวนภาพทั้งหมดเป๊ะ)
// ratios = { train, valid, test } (ตัวเลขสัดส่วน เช่น 70/20/10) — single source ใช้ทั้ง split-panel และ export auto-apply
export function assignSplitsByRatio(keys, ratios) {
  const total = keys.length;
  const sum = ratios.train + ratios.valid + ratios.test || 1;
  const nTrain = Math.round((ratios.train / sum) * total);
  const nValid = Math.round((ratios.valid / sum) * total);
  // test = ที่เหลือ กันปัดเศษพลาดจนผลรวมไม่เท่า total
  const nValidClamped = Math.min(nValid, total - nTrain > 0 ? total - nTrain : 0);
  const order = shuffled(keys);
  let i = 0;
  // TODO(scale): setSplit เรียกทีละภาพ → re-render ทุกครั้งผ่าน onChange listeners (N ครั้งต่อการสุ่มแบ่ง)
  // เมื่อ dataset ใหญ่ (1000+ ภาพ) ควรมี batch-notify API ใน annotations.js (เช่น setSplitMany) เพื่อ notify ครั้งเดียว
  for (; i < Math.min(nTrain, total); i++) setSplit(order[i], "train");
  for (; i < Math.min(nTrain + nValidClamped, total); i++) setSplit(order[i], "valid");
  for (; i < total; i++) setSplit(order[i], "test");
}

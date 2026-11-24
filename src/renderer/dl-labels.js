// อ่านชื่อคลาสจากไฟล์ที่ผู้ใช้เลือก (ปุ่มเลือกไฟล์ชื่อคลาสของ Model Detect)
//   - COCO json (_annotations.coco.json ที่ใช้เทรน RF-DETR) → categories
//   - data.yaml (names:) ของ YOLO
//   - classes.txt / .names (บรรทัดละคลาส)

// ค่า supercategory ที่ถือว่า "ไม่มีพ่อ" — ชุดเดียวกับ rfdetr (datasets/coco.py: _SUPERCATEGORY_PLACEHOLDERS)
const SUPER_PLACEHOLDERS = new Set(["", "none", "null"]);

const catId = (c) => {
  const n = Number(c && c.id);
  return c && c.id !== null && c.id !== "" && Number.isInteger(n) ? n : null;
};

// ชื่อคลาสตามลำดับ index ของโมเดล RF-DETR = กติกาเดียวกับ rfdetr ตอนเทรน (filter_parent_categories + cat2label):
// เรียงตาม id → ตัดหมวด "แม่" ที่ไม่มี annotation ทิ้ง (ชื่อของมันเป็น supercategory ของหมวดอื่น เช่นรากที่ Roboflow ใส่เป็น id 0
// ชื่อ "objects" หรือชื่อโปรเจกต์) → ที่เหลือคือคลาส 0..N-1 · COCO ต้นฉบับ (person/vehicle…) ไม่มีหมวดไหนถูกตัด
export function cocoClassNames(data) {
  const cats = (Array.isArray(data && data.categories) ? data.categories : []).filter((c) => c && typeof c.name === "string");
  const ordered = cats
    .map((c, i) => ({ c, i, id: catId(c) }))
    .sort((a, b) => (a.id === null) - (b.id === null) || (a.id ?? 0) - (b.id ?? 0) || a.i - b.i)
    .map((x) => x.c);
  const parents = new Set();
  for (const c of ordered) {
    const sup = c.supercategory === undefined ? "none" : c.supercategory;
    if (sup !== null && !SUPER_PLACEHOLDERS.has(sup) && sup !== c.name) parents.add(sup);
  }
  if (!parents.size) return ordered.map((c) => c.name);
  const annotated = new Set(
    (Array.isArray(data.annotations) ? data.annotations : []).map((a) => Number(a && a.category_id)).filter(Number.isInteger)
  );
  const kept = ordered.filter((c) => !(parents.has(c.name) && !annotated.has(catId(c))));
  return (kept.length ? kept : ordered).map((c) => c.name);
}

export function parseLabels(text, name) {
  if (/\.json$/i.test(name)) {
    try {
      return cocoClassNames(JSON.parse(text));
    } catch {
      return [];
    }
  }
  if (/\.ya?ml$/i.test(name)) {
    const idx = text.indexOf("names:");
    if (idx < 0) return [];
    const after = text.slice(idx + 6);
    const inline = after.match(/\[([^\]]*)\]/);
    if (inline) {
      return inline[1].split(",").map((s) => s.trim().replace(/^['"]|['"]$/g, "")).filter(Boolean);
    }
    const out = [];
    for (const ln of after.split(/\r?\n/).slice(1)) {
      if (/^\s*$/.test(ln)) continue;
      if (/^\S/.test(ln)) break; // ถึง key ถัดไป (ไม่เยื้อง)
      const m = ln.match(/^\s*(?:-\s*|\d+\s*:\s*)?['"]?(.+?)['"]?\s*$/);
      if (m && m[1]) out.push(m[1].trim());
    }
    return out;
  }
  return text.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
}

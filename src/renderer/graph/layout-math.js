// คณิตศาสตร์ล้วนของกระดาน (ไม่ import LiteGraph/DOM) — แยกออกจาก editor.js เพื่อเทสต์ได้
// rect = { x0, y0, x1, y1 } พิกัด graph (รวมแถบหัวการ์ดแล้ว)

// คำนวณมุมมอง "พอดีทุกกล่อง": คืน { scale, ox, oy } หรือ null ถ้าไม่มีกล่อง/viewport ว่าง
// scale ถูกล็อก [min,max] — ถ้าเนื้อหากว้างเกินแม้ที่ scale ต่ำสุด → ชิดซ้าย (ไม่ตัดต้นทางทิ้งทั้งสองข้าง)
export function computeFit(rects, vw, vh, { pad = 40, min = 0.6, max = 1 } = {}) {
  if (!rects || !rects.length || !(vw > 0) || !(vh > 0)) return null;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const r of rects) {
    x0 = Math.min(x0, r.x0); y0 = Math.min(y0, r.y0);
    x1 = Math.max(x1, r.x1); y1 = Math.max(y1, r.y1);
  }
  const scale = Math.max(min, Math.min(max, (vw - 2 * pad) / (x1 - x0), (vh - 2 * pad) / (y1 - y0)));
  let ox = vw / (2 * scale) - (x0 + x1) / 2;
  let oy = vh / (2 * scale) - (y0 + y1) / 2;
  if ((x1 - x0) * scale > vw - 2 * pad) ox = pad / scale - x0; // กว้างเกิน → ชิดซ้าย
  if ((y1 - y0) * scale > vh - 2 * pad) oy = pad / scale - y0; // สูงเกิน → ชิดบน
  return { scale, ox, oy };
}

const hit = (a, b, m) => a.x0 < b.x1 + m && a.x1 > b.x0 - m && a.y0 < b.y1 + m && a.y1 > b.y0 - m;

// หาตำแหน่งวางกล่องใหม่ (pos = มุมซ้ายบนของตัวการ์ด ไม่รวมหัว) ใกล้ centre โดยไม่ทับกล่องเดิม
// ลำดับลอง: (1) ตรงกลาง (2) ขวาสุดของกราฟ +28 ในแถวกลางเดียวกัน (จอกว้างแต่เตี้ย → ไล่ขวาก่อน)
// (3) แถวถัดลงไปทีละแถว สแกนจากซ้ายสุดของกราฟไปขวา · bounds = { x1, y1 } (ถ้ามี) จำกัดไม่ให้เลยขอบที่มองเห็นที่ scale ต่ำสุด
// ไม่เจอที่ในขอบเขต → ที่แรกที่ไม่ทับ (ผู้เรียก pan ตามเอง) → ท้ายสุดใต้ bbox
export function findFreeSpot(rects, centre, size, TH, margin = 16, bounds = null) {
  const [w, h] = size;
  const free = (px, py) => {
    const bx = { x0: px, y0: py - TH, x1: px + w, y1: py + h };
    return !rects.some((r) => hit(bx, r, margin));
  };
  const inside = (px, py) => !bounds || (px + w <= bounds.x1 && py + h <= bounds.y1);
  const cy = centre[1] - h / 2;
  const left = rects.length ? Math.min(...rects.map((r) => r.x0)) : centre[0] - w / 2;
  const right = rects.length ? Math.max(...rects.map((r) => r.x1)) : centre[0] + w / 2;
  const rowStep = TH + h + 24;
  const cand = [[centre[0] - w / 2, cy], [right + 28, cy]];
  for (let k = 1; k <= 20; k++) {
    for (let x = left; x <= right + 28 + w * 8; x += w + 28) cand.push([x, cy + k * rowStep]);
  }
  let firstFree = null;
  for (const [px, py] of cand) {
    if (!free(px, py)) continue;
    if (inside(px, py)) return [px, py];
    if (!firstFree) firstFree = [px, py];
  }
  if (firstFree) return firstFree;
  const bottom = rects.reduce((m, r) => Math.max(m, r.y1), -Infinity);
  return [centre[0] - w / 2, bottom + margin + TH];
}

// ใส่ผล fit ลง ds แล้ว "อ่านกลับจาก ds" เป็น snapshot — ds.offset ของ LiteGraph เป็น Float32Array
// (330.8 → 330.79998779) ถ้าเก็บค่า double ก่อนปัดเศษ การเทียบทีหลังจะไม่เท่ากันเสมอ
export function applyFit(ds, fit) {
  ds.scale = fit.scale;
  ds.offset[0] = fit.ox;
  ds.offset[1] = fit.oy;
  return { s: ds.scale, ox: ds.offset[0], oy: ds.offset[1] };
}

// ผู้ใช้ยังไม่ซูม/เลื่อนนับจาก snapshot? (ใช้ตัดสินใจ fit ซ้ำอัตโนมัติ)
export function isViewUntouched(ds, snap) {
  return !!(snap && ds && Math.abs(ds.scale - snap.s) < 1e-6 &&
    Math.abs(ds.offset[0] - snap.ox) < 1e-6 && Math.abs(ds.offset[1] - snap.oy) < 1e-6);
}

// ขนาด buffer ของ canvas = ขนาด CSS จริงเป๊ะ (ห้ามมี floor — ไม่งั้นพิกัดคลิก/พอร์ตเพี้ยน) · ซ่อนอยู่ → null
export function canvasBufferSize(rect) {
  if (!rect || !(rect.width >= 1) || !(rect.height >= 1)) return null;
  return { w: Math.round(rect.width), h: Math.round(rect.height) };
}

// ข้อความ status bar: "Display 1: A → B" ต่อกล่อง
export function statusText(dps, stepLabel) {
  const connected = dps.filter((d) => d.connected);
  if (dps.length === 0) return "ยังไม่มีกล่องผลลัพธ์";
  if (connected.length === 0) {
    return "⚠ ยังไม่มีกล่องผลลัพธ์ใดต่อสายถึงแหล่งภาพ — ลากสายจากแหล่งภาพ ผ่าน filter ไปยังผลลัพธ์ให้ครบ";
  }
  const parts = connected.map((d) => {
    const chain = d.pipeline.length ? d.pipeline.map(stepLabel).join(" → ") : "ภาพต้นฉบับ";
    return `${d.name || `Display ${d.autoNum}`}: ${chain}`;
  });
  const extra = dps.length - connected.length;
  return parts.join("   |   ") + (extra > 0 ? `   (อีก ${extra} กล่องยังไม่ต่อสาย)` : "");
}

// บรรทัดรายละเอียดของ Source node: [ชื่อไฟล์] · [กว้าง×สูง] · [Port N] — ข้ามส่วนที่ยังไม่รู้ (กัน "undefined×undefined")
export function sourceDetail({ name, w, h, port } = {}) {
  return [name, w && h ? `${w}×${h}` : "", port ? `Port ${port}` : ""].filter(Boolean).join(" · ");
}

// ป้ายซูมในหัวกระดาน: scale 0.86 → "86%"
export function formatZoom(scale) {
  return Number.isFinite(scale) ? `${Math.round(scale * 100)}%` : "—";
}

// ระดับของจุดสถานะบน status bar จากข้อความ: ✕ = error (แดง) · ⚠ / ยังไม่มีกล่องผลลัพธ์ = ยังไม่มี Display ต่อสาย (ส้ม) · อื่น ๆ = ปกติ (เขียว)
export function statusLevel(text) {
  const t = String(text || "").trimStart();
  if (t.startsWith("✕")) return "error";
  if (t.startsWith("⚠") || t.startsWith("ยังไม่มีกล่องผลลัพธ์")) return "warn";
  // ข้อความล้มเหลวที่ไม่มี ✕ นำหน้า (โหลดโมเดลไม่สำเร็จ / เปิด Pose ไม่สำเร็จ / ไม่พบไฟล์ภาพ …) — ตรวจแล้วทุกข้อความใน app.js ที่มี "ไม่พบ" เป็นความล้มเหลว
  if (/ไม่สำเร็จ|ขัดข้อง|ไม่พบ/.test(t)) return "error";
  return "ok";
}

// ชิปต้นทางใน breadcrumb ตามโหมดของแหล่งภาพ (เดิมตายตัวว่า "ภาพ")
export function sourceChipLabel(source) {
  return source === "camera" ? "กล้อง" : source === "video" ? "วิดีโอ" : "ภาพนิ่ง";
}

// ตัดสินใจ "แทรกหลังกล่องที่เลือก" (insert after selected) — pure ไม่แตะ LiteGraph
//   sel = { kind: "source"|"filter"|"display"|"dl"|..., x, y, w, h }  (x,y = pos ของการ์ด ไม่รวมแถบหัว)
//   newNode = { w, h, inputs }  ·  fanOut = จำนวนสายออกของกล่องที่เลือก
//   downstream = [{ id, x, y, w, h, direct? }] กล่องปลายน้ำทั้งหมด (direct = เป้าหมายที่สายออกของ selected ต่ออยู่ตรง ๆ) · obstacles = กล่องอื่นที่ไม่เกี่ยวข้อง (rect รวมแถบหัว)
// คืน { mode: "free", reason } (ให้ใช้วางที่ว่างแบบเดิม) หรือ { mode: "splice", pos, shiftIds, dx }
export function planInsert({ sel, newNode, fanOut = 0, downstream = [], obstacles = [], TH = 28, gap = 28, margin = 8 }) {
  if (!sel || (sel.kind !== "source" && sel.kind !== "filter")) return { mode: "free", reason: "selected-kind" };
  if (newNode.inputs === 2) return { mode: "free", reason: "two-inputs" };
  if (fanOut > 1) return { mode: "free", reason: "fan-out" };
  const pos = [sel.x + sel.w + gap, sel.y];
  const box = { x0: pos[0], y0: pos[1] - TH, x1: pos[0] + newNode.w, y1: pos[1] + newNode.h };
  // เป้าหมายตรง ๆ ต้องอยู่ขวาของกล่องใหม่ + gap — เลื่อนปลายน้ำทั้งก้อนด้วยระยะเดียวกัน (คงรูปทรงเดิม)
  // วัดจากเป้าหมายตรง ๆ ไม่ใช่กล่องซ้ายสุดของก้อน (กล่องปลายน้ำที่อยู่ซ้ายไกลจะทำให้เลื่อนเกินจำเป็น)
  let dx = 0;
  if (downstream.length) {
    const direct = downstream.filter((d) => d.direct);
    const minX = Math.min(...(direct.length ? direct : downstream).map((d) => d.x));
    dx = Math.max(0, box.x1 + gap - minX);
  }
  const clash = (r) => obstacles.some((o) => hit(r, o, margin));
  if (clash(box)) return { mode: "free", reason: "blocked" };
  for (const d of downstream) {
    const r = { x0: d.x + dx, y0: d.y - TH, x1: d.x + dx + d.w, y1: d.y + d.h };
    if (clash(r) || hit(box, r, margin)) return { mode: "free", reason: "blocked-downstream" }; // ปลายน้ำที่เลื่อนแล้วต้องไม่ชนทั้งกล่องอื่นและกล่องใหม่
  }
  return { mode: "splice", pos, shiftIds: downstream.map((d) => d.id), dx };
}

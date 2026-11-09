// จัด layout จอผลลัพธ์ใน VIEWER ให้ภาพใหญ่ที่สุด (2026-11-09)
// เดิมใช้ cols = ceil(sqrt(n)) ตายตัว → 3 จอได้ 2×2 เหลือช่องว่าง ภาพเล็ก
// ตอนนี้ลองทุกจำนวนคอลัมน์ แล้วเลือกอันที่ภาพ (ตามสัดส่วนจริง, object-fit: contain) แสดงได้สูงที่สุด

// n = จำนวนจอ · W,H = พื้นที่ grid (ไม่รวม padding) · aspect = กว้าง/สูงของภาพ
// capH = ความสูงหัวจอ (ชื่อ + สายทอด) · gap = ช่องไฟระหว่างจอ
export function bestGridCols(n, W, H, aspect, capH = 0, gap = 0) {
  if (!(n > 1)) return 1;
  if (!(W > 0 && H > 0)) return Math.ceil(Math.sqrt(n));
  const a = aspect > 0 && Number.isFinite(aspect) ? aspect : 4 / 3;
  let best = 1;
  let bestH = -1;
  let bestEmpty = Infinity;
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols);
    const cw = (W - gap * (cols - 1)) / cols;
    const ch = (H - gap * (rows - 1)) / rows - capH;
    if (cw <= 0 || ch <= 0) continue;
    const imgH = Math.min(cw / a, ch); // ความสูงภาพที่แสดงได้จริงในช่อง
    const empty = rows * cols - n;
    // ภาพใหญ่กว่าชนะ · ถ้าเท่ากัน (ต่างไม่ถึง 0.5px) เลือกแบบช่องว่างน้อยกว่า
    if (imgH > bestH + 0.5 || (Math.abs(imgH - bestH) <= 0.5 && empty < bestEmpty)) {
      best = cols;
      bestH = imgH;
      bestEmpty = empty;
    }
  }
  return best;
}

// ตำแหน่งของแต่ละจอบน grid ที่มี cols*2 track (แต่ละจอกว้าง 2 track)
// แถวสุดท้ายที่ไม่เต็มจะถูกเลื่อนมาอยู่ตรงกลาง — คืน [{ start, span }] (start นับจาก 1 แบบ CSS grid)
export function gridPlacement(n, cols) {
  const out = [];
  const c = Math.max(1, cols | 0);
  const lastRowStart = n - (n % c || c);
  const rem = n - lastRowStart;
  const offset = c - rem; // track ที่ต้องเลื่อน (ครึ่งหนึ่งของช่องว่าง × 2 track)
  for (let i = 0; i < n; i++) {
    const col = i % c;
    const start = i >= lastRowStart && rem < c ? col * 2 + 1 + offset : col * 2 + 1;
    out.push({ start, span: 2 });
  }
  return out;
}

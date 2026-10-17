// P-8 C-A — แปลง ctx.values (§A.2: kind "table" | "scalars") เป็นข้อความ CSV · ฟังก์ชันล้วน (ไม่แตะ DOM) ทดสอบใน node ได้
// รูปแบบ: คั่นจุลภาค · ขึ้นบรรทัดใหม่ CRLF · escape " เป็น "" · ตัวเลขดิบ (ไม่ปัดทศนิยม) · null/undefined/NaN → ว่าง
// Infinity → inf / -inf · boolean → true/false · หัวคอลัมน์ = key อังกฤษของ worker (ไม่ใช่ป้ายไทย)
// opts.bom (ค่าเริ่มต้น true) = ขึ้นต้นด้วย U+FEFF ให้ Excel อ่านภาษาไทยถูก (ไฟล์ .csv) — ตอนคัดลอกลงคลิปบอร์ดส่ง bom:false
// table → หัว + 1 แถวต่อวัตถุ (ถ้า total > จำนวนแถว = ถูกจำกัด → ต่อท้าย `# total,<จำนวนจริง>`)
// scalars → หัว (key ของทุกช่อง) + 1 แถวค่า · kind อื่น/ไม่มีข้อมูล → null

const BOM = "\uFEFF"; // U+FEFF — UTF-8 BOM ให้ Excel อ่านไทยถูก

export function cell(v) {
  if (v === null || v === undefined) return "";
  let s;
  if (typeof v === "number") {
    if (Number.isNaN(v)) return "";
    s = Number.isFinite(v) ? String(v) : (v > 0 ? "inf" : "-inf");
  } else if (typeof v === "boolean") {
    s = v ? "true" : "false";
  } else {
    s = String(v);
  }
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

// rows = อาร์เรย์ของแถว (แต่ละแถวเป็นอาร์เรย์ของค่า) → ข้อความ CSV กฎเดียวกับ valuesToCsv (ใช้ใน batch CSV)
export function rowsToCsv(rows, opts = {}) {
  const bom = opts.bom === false ? "" : BOM;
  return bom + rows.map((r) => r.map(cell).join(",")).join("\r\n") + "\r\n";
}

export function valuesToCsv(values, opts = {}) {
  if (!values || typeof values !== "object") return null;
  let header, rows;
  if (values.kind === "table") {
    header = Array.isArray(values.columns) ? values.columns : [];
    rows = Array.isArray(values.rows) ? values.rows : [];
  } else if (values.kind === "scalars") {
    const fields = Array.isArray(values.fields) ? values.fields : [];
    header = fields.map((f) => (f ? f.key : ""));
    rows = [fields.map((f) => (f ? f.value : null))];
  } else {
    return null;
  }
  const n = header.length;
  const lines = [header.map(cell).join(",")];
  for (const r of rows) {
    const a = Array.isArray(r) ? r.slice(0, n) : []; // แถวยาวกว่าหัว → ตัดทิ้ง
    while (a.length < n) a.push(null); // แถวสั้นกว่าหัว → เติมช่องว่างให้ครบ
    lines.push(a.map(cell).join(","));
  }
  // table ที่มี total ตัวเลขและ total > จำนวนแถว (ถูก cap ด้วย maxObjects) → ต่อท้ายบรรทัด "# total,<n>" ให้ไม่เสียจำนวนวัตถุจริง
  if (values.kind === "table" && typeof values.total === "number" && Number.isFinite(values.total) && values.total > rows.length) {
    lines.push("# total," + values.total);
  }
  const bom = opts.bom === false ? "" : BOM;
  return bom + lines.join("\r\n") + "\r\n";
}

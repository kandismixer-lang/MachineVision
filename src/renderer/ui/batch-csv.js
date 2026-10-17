// P-8 C-B — รวมค่าที่วัดได้ (ctx.values) ของ "ทุกภาพในโฟลเดอร์" เป็น CSV 2 ไฟล์ · ฟังก์ชันล้วน (ไม่แตะ DOM/IPC) ทดสอบใน node ได้
// - <stem>-summary.csv  : 1 แถวต่อภาพ = file, [verdict, reason เมื่อมีกล่อง pass_fail], ค่า scalars ของทุกกล่อง, [error เมื่อมีภาพพัง]
//     คอลัมน์ของกล่อง = "<op>.<key>" (op ซ้ำกัน 2 กล่อง → "<op>#2.<key>" ตามลำดับที่เจอก่อน) · table → "<op>.total" (จำนวนวัตถุจริง ไม่ถูก cap)
// - <stem>-objects.csv : เฉพาะเมื่อมีกล่อง table · 1 แถวต่อวัตถุ = file, box, row, <คอลัมน์ของตาราง (union ข้ามกล่อง/ภาพ)>
// รูปแบบไฟล์ (BOM, คั่นจุลภาค, CRLF, escape, null/inf) ใช้ rowsToCsv ของ values-csv.js ตัวเดียวกับปุ่ม "💾 CSV" ของแผงค่า
// หน่วยความจำ: เก็บเฉพาะค่า (ไม่เก็บภาพ) · แถววัตถุสะสมได้ไม่เกิน maxObjectRows (ค่าเริ่มต้น 50,000 แถวทั้งชุด) ที่เกินถูกทิ้งและนับไว้ใน stats

import { rowsToCsv } from "./values-csv.js";

export const MAX_OBJECT_ROWS = 50000;

export function createBatchCsv(opts = {}) {
  const maxObjectRows = Number.isFinite(opts.maxObjectRows) ? Math.max(0, opts.maxObjectRows) : MAX_OBJECT_ROWS;
  const boxes = new Map(); // boxKey -> { op } (ลำดับ = ลำดับที่เจอครั้งแรก)
  const images = []; // { file, error? | boxes: Map(boxKey -> compact values) }
  let objectRowsKept = 0;
  let objectRowsDropped = 0;
  let errors = 0;

  // entries = result.values จาก worker: [{ op, id, nodeId, values }] (กล่องที่มี ctx.values ตามลำดับ pipeline)
  function add(file, entries) {
    const img = { file: String(file ?? ""), boxes: new Map() };
    (Array.isArray(entries) ? entries : []).forEach((e, idx) => {
      const v = e && e.values;
      if (!v || typeof v !== "object") return;
      // กล่องเดียวกันอาจโผล่ซ้ำจากหลายสายผลลัพธ์ → ใช้ nodeId กันซ้ำ (ไม่มี nodeId = id + ลำดับ)
      const key = e.nodeId != null ? "n" + e.nodeId : String(e.id) + "@" + idx;
      if (img.boxes.has(key)) return;
      let compact = null;
      if (v.kind === "scalars" && Array.isArray(v.fields)) {
        compact = { kind: "scalars", fields: v.fields.filter(Boolean).map((f) => ({ key: String(f.key), value: f.value })) };
      } else if (v.kind === "table") {
        const columns = Array.isArray(v.columns) ? v.columns.map(String) : [];
        const rows = Array.isArray(v.rows) ? v.rows : [];
        const keep = rows.slice(0, Math.max(0, maxObjectRows - objectRowsKept));
        objectRowsKept += keep.length;
        objectRowsDropped += rows.length - keep.length;
        const total = typeof v.total === "number" && Number.isFinite(v.total) ? v.total : rows.length;
        compact = { kind: "table", columns, rows: keep.map((r) => (Array.isArray(r) ? r.slice(0, columns.length) : [])), total };
      }
      if (!compact) return;
      if (!boxes.has(key)) boxes.set(key, { op: String(e.op || e.id || "box") });
      img.boxes.set(key, compact);
    });
    images.push(img);
  }

  function addError(file, message) {
    errors++;
    images.push({ file: String(file ?? ""), error: String(message ?? "error"), boxes: new Map() });
  }

  // ชื่อเรียกของแต่ละกล่อง: op แรก = "<op>", ตัวที่ 2 ขึ้นไป = "<op>#2" ...
  function labels() {
    const count = new Map();
    const out = new Map();
    for (const [key, b] of boxes) {
      const n = (count.get(b.op) || 0) + 1;
      count.set(b.op, n);
      out.set(key, n === 1 ? b.op : b.op + "#" + n);
    }
    return out;
  }

  function build(stem) {
    const stats = { images: images.length, errors, objectRowsKept, objectRowsDropped, boxes: boxes.size };
    if (images.length === 0 || boxes.size === 0) return { files: [], stats }; // ไม่มีกล่องที่ให้ค่า = ไม่มีไฟล์ (แม้มีภาพพัง)
    const label = labels();
    const pfCount = [...boxes.values()].filter((b) => b.op === "pass_fail").length;
    // verdict/reason รวมของทุก pass_fail · ถ้ามี pass_fail กล่องเดียว result/reason ไม่ต้องซ้ำในคอลัมน์ของกล่อง
    const skipKeys = (b) => (b.op === "pass_fail" && pfCount === 1 ? ["result", "reason"] : []);

    // ---- summary ----
    const boxCols = new Map(); // boxKey -> คอลัมน์ของกล่อง (ลำดับที่เจอครั้งแรก) — แล้วเรียงตามลำดับกล่อง
    const rowsCells = images.map((img) => {
      const cells = new Map();
      for (const [key, v] of img.boxes) {
        const b = boxes.get(key);
        const lb = label.get(key);
        if (!boxCols.has(key)) boxCols.set(key, []);
        const mine = boxCols.get(key);
        const put = (c, val) => { cells.set(c, val); if (!mine.includes(c)) mine.push(c); };
        if (v.kind === "scalars") {
          const skip = skipKeys(b);
          for (const f of v.fields) if (!skip.includes(f.key)) put(lb + "." + f.key, f.value);
        } else {
          put(lb + ".total", v.total);
        }
      }
      return cells;
    });
    const cols = [];
    for (const key of boxes.keys()) if (boxCols.has(key)) cols.push(...boxCols.get(key));
    const verdictOf = (img) => {
      const rs = [];
      for (const [key, v] of img.boxes) {
        if (boxes.get(key).op !== "pass_fail" || v.kind !== "scalars") continue;
        const get = (k) => (v.fields.find((f) => f.key === k) || {}).value;
        rs.push({ result: get("result"), reason: get("reason") });
      }
      if (rs.length === 0) return ["", ""];
      const bad = rs.filter((r) => r.result !== "OK");
      return bad.length === 0
        ? ["OK", ""]
        : ["NG", bad.map((r) => r.reason).filter((r) => r != null && r !== "").join("; ")];
    };
    const hasVerdict = pfCount > 0;
    const header = ["file"];
    if (hasVerdict) header.push("verdict", "reason");
    header.push(...cols);
    if (errors > 0) header.push("error");
    const summary = [header];
    images.forEach((img, i) => {
      const row = [img.file];
      if (hasVerdict) row.push(...verdictOf(img));
      for (const c of cols) row.push(rowsCells[i].has(c) ? rowsCells[i].get(c) : null);
      if (errors > 0) row.push(img.error ?? "");
      summary.push(row);
    });
    const files = [{ name: stem + "-summary.csv", text: rowsToCsv(summary) }];

    // ---- objects (เฉพาะเมื่อมีกล่อง table) ----
    let anyTable = false;
    const objCols = [];
    const objSeen = new Set();
    for (const img of images) {
      for (const v of img.boxes.values()) {
        if (v.kind !== "table") continue;
        anyTable = true;
        for (const c of v.columns) if (!objSeen.has(c)) { objSeen.add(c); objCols.push(c); }
      }
    }
    if (anyTable) {
      const out = [["file", "box", "row", ...objCols]];
      let kept = 0;
      let dropped = 0;
      for (const img of images) {
        for (const [key, v] of img.boxes) {
          if (v.kind !== "table") continue;
          kept += v.rows.length;
          dropped += Math.max(0, v.total - v.rows.length); // ทั้งที่ op ตัดเอง (maxObjects) และที่ตัดเพราะเพดานของชุด
          const at = objCols.map((c) => v.columns.indexOf(c));
          v.rows.forEach((r, i) => out.push([img.file, label.get(key), i + 1, ...at.map((j) => (j < 0 ? null : r[j] ?? null))]));
        }
      }
      // แถววัตถุถูกตัด → ต่อท้าย "# truncated,<kept>,<dropped>" (แบบเดียวกับ "# total,<n>" ของ valuesToCsv)
      let text = rowsToCsv(out);
      if (dropped > 0) text += "# truncated," + kept + "," + dropped + "\r\n";
      files.push({ name: stem + "-objects.csv", text });
    }
    return { files, stats };
  }

  return { add, addError, build };
}

// เขียนไฟล์ CSV ที่ build ได้: มี dir (Electron) → writeFile(dir, name, bytes) · ไม่มี → download(blob, name) (fallback เบราว์เซอร์ เหมือนภาพ)
// คืนจำนวนไฟล์ที่ส่งไป
export async function saveBatchCsv(files, { dir, writeFile, download }) {
  let n = 0;
  for (const f of files) {
    if (dir && writeFile) {
      await writeFile(dir, f.name, new TextEncoder().encode(f.text)); // BOM (U+FEFF) → EF BB BF
    } else if (download) {
      download(new Blob([f.text], { type: "text/csv;charset=utf-8" }), f.name);
    } else {
      continue;
    }
    n++;
  }
  return n;
}

// จบ batch: build + เขียนไฟล์ + คืนข้อความต่อท้ายสถานะ (csv = null → ตัวเลือกปิด → "" ไม่เขียนอะไร) · เขียนพลาดไม่ throw (ภาพบันทึกไปแล้ว)
export async function finishBatchCsv(csv, { dir, writeFile, download, stem }) {
  if (!csv) return "";
  const { files, stats } = csv.build(stem);
  if (files.length === 0) return ""; // ไม่มีกล่องที่ให้ค่า → เงียบ
  try {
    const n = await saveBatchCsv(files, { dir, writeFile, download });
    let note = ` · CSV ${n} ไฟล์`;
    if (stats.errors > 0) note += ` (ภาพพลาด ${stats.errors})`;
    if (stats.objectRowsDropped > 0) note += ` · แถววัตถุเกิน ${stats.objectRowsKept} แถวถูกตัดทิ้ง ${stats.objectRowsDropped}`;
    return note;
  } catch {
    return " · บันทึก CSV ไม่สำเร็จ";
  }
}

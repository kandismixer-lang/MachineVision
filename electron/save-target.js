// ตัวช่วยเลือกปลายทางของ Save Picture (แยกจาก main.js เพื่อเทสได้โดยไม่ต้องเปิด Electron)
// - ภาพเดียว  → ไฟล์ .png ตามชื่อที่ผู้ใช้ตั้งในหน้าต่าง Save As
// - หลายภาพ  → สร้างโฟลเดอร์ใหม่ตามชื่อที่ตั้ง แล้วเซฟทุกไฟล์ลงในนั้น
import path from "node:path";

// ชื่อไฟล์/โฟลเดอร์ที่ Windows รับได้ (ตัดอักขระต้องห้าม + จุด/ช่องว่างท้ายชื่อ)
export function safeBaseName(name, fallback = "FrameLab") {
  const s = String(name || "")
    .replace(/[\\/:*?"<>|\x00-\x1f]+/g, "_")
    .replace(/[. ]+$/g, "")
    .trim();
  return s || fallback;
}

// บังคับนามสกุล .png (ผู้ใช้พิมพ์ชื่อไม่ใส่นามสกุล หรือใส่ .jpg มา → ยังเป็น PNG จริง)
export function ensurePngPath(p) {
  return /\.png$/i.test(p) ? p : p.replace(/\.(jpe?g|bmp|tiff?|webp)$/i, "") + ".png";
}

// ถ้ามีโฟลเดอร์/ไฟล์ชื่อนี้อยู่แล้ว → ต่อท้าย " (2)", " (3)" … แบบ Windows (ไม่เขียนทับของเดิม)
// exists(p) => Promise<boolean>
export async function uniqueFolderPath(base, exists) {
  if (!(await exists(base))) return base;
  for (let i = 2; i < 1000; i++) {
    const p = `${base} (${i})`;
    if (!(await exists(p))) return p;
  }
  throw new Error("หาชื่อโฟลเดอร์ว่างไม่ได้");
}

// แปลงผลจาก dialog.showSaveDialog เป็นปลายทางจริง
// multi = false → { kind: "file", path } · multi = true → { kind: "folder", path } (path ของโฟลเดอร์ที่จะสร้าง)
export async function resolveSaveTarget(filePath, multi, exists) {
  const dir = path.dirname(filePath);
  if (!multi) {
    const name = safeBaseName(path.basename(filePath).replace(/\.png$/i, ""), "result");
    return { kind: "file", path: ensurePngPath(path.join(dir, name)) };
  }
  // หลายภาพ: ชื่อที่ตั้ง = ชื่อโฟลเดอร์ (ตัด .png ที่ติดมาจากตัวกรองทิ้ง)
  const name = safeBaseName(path.basename(filePath).replace(/\.png$/i, ""));
  return { kind: "folder", path: await uniqueFolderPath(path.join(dir, name), exists) };
}

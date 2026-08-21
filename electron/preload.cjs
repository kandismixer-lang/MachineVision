// Preload ทำงานใน context แยก (contextIsolation) — เปิดสะพานเฉพาะที่จำเป็น
// Phase 1 ยังไม่ต้องใช้ IPC พิเศษ แต่วางโครงไว้ให้ Phase 3 (save/load pipeline)
const { contextBridge, ipcRenderer } = require("electron");

// เวอร์ชันจริงจาก package.json — main.js ส่งมาทาง additionalArguments (--app-version=x.y.z)
// preload รันแบบ sandboxed ใน Electron รุ่นนี้ ทำให้ require("../package.json") ใช้ไม่ได้โดยตรง
const verArg = process.argv.find((a) => a.startsWith("--app-version="));
const appVersion = verArg ? verArg.slice("--app-version=".length) : "";

contextBridge.exposeInMainWorld("appInfo", {
  platform: process.platform,
  isDev: process.env.NODE_ENV === "development",
  version: appVersion,
});

// สะพานสำหรับบันทึกภาพ dataset ลงโฟลเดอร์เดียว (เลือกที่ครั้งเดียว ไม่เด้ง dialog รัว)
contextBridge.exposeInMainWorld("mvExport", {
  available: true,
  chooseDir: () => ipcRenderer.invoke("export:chooseDir"),
  writeFile: (dir, name, bytes) => ipcRenderer.invoke("export:writeFile", { dir, name, bytes }),
});

// สะพานอ่านค่าประสิทธิภาพระบบ (CPU/RAM จาก main process) — Design §17
contextBridge.exposeInMainWorld("mvMetrics", {
  available: true,
  get: () => ipcRenderer.invoke("sys:metrics"),
});

// สะพานแปลงโมเดล .pt → .onnx (ผ่าน ultralytics ในเครื่อง)
contextBridge.exposeInMainWorld("mvDL", {
  available: true,
  convertPt: (ptPath) => ipcRenderer.invoke("dl:convertPt", ptPath),
});

// สะพานปุ่ม "เช็คอัปเดตตอนนี้" — check() เริ่มเช็ค, onStatus(cb) รับผลจริง (checking/available/
// not-available/downloaded/error) จาก main.js กันเงียบเหมือน auto-check ตอนเปิดแอป
contextBridge.exposeInMainWorld("mvUpdate", {
  available: true,
  check: () => ipcRenderer.invoke("update:check"),
  onStatus: (cb) => {
    const handler = (_e, data) => cb(data);
    ipcRenderer.on("update:status", handler);
    return () => ipcRenderer.removeListener("update:status", handler);
  },
});

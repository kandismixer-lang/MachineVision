import { app, BrowserWindow, session, ipcMain, dialog, nativeImage, shell } from "electron";
import path from "node:path";
import fs from "node:fs/promises";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import electronUpdater from "electron-updater"; // electron-updater เป็น CommonJS — ต้อง import แบบ default แล้วดึงออกมา
const { autoUpdater } = electronUpdater;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isDev = process.env.NODE_ENV === "development";
// บางเครื่อง (GPU/driver บางรุ่น) พอกล้องเปิดพร้อม hardware-accelerated rendering
// จะทำให้ display driver ค้าง/รีสตาร์ทตัวเอง จนทั้งเครื่องต้อง restart — ปิด GPU accel
// กันไว้ก่อนเป็นค่า default (ตั้ง MV_GPU=1 เพื่อเปิดกลับถ้าเครื่องไม่มีปัญหาและอยากได้ perf เพิ่ม)
if (process.env.MV_GPU !== "1") {
  app.disableHardwareAcceleration();
}
// เปิด DevTools อัตโนมัติเฉพาะเมื่อสั่ง (ตั้ง OPEN_DEVTOOLS=1) — ปกติไม่เปิด
const openDevtools = process.env.OPEN_DEVTOOLS === "1";
// ไอคอนแอป (taskbar / หัวหน้าต่าง) — ใช้ .ico (พื้นหลังโปร่งใส หลายขนาด) แทน .jpg เดิม
const appIcon = nativeImage.createFromPath(path.join(__dirname, "icon.ico"));

let mainWindow = null; // อ้างอิงหน้าต่างล่าสุด — ใช้ส่งสถานะ auto-update กลับไป renderer
// เก็บสถานะ "โหลดอัปเดตเสร็จแล้ว" ระดับ process — event update-downloaded ยิงครั้งเดียว และมัก
// ยิงตั้งแต่เปิดแอป (auto-check เบื้องหลัง) ก่อนผู้ใช้เปิดหน้าตั้งค่าด้วยซ้ำ ถ้าเก็บสถานะไว้แค่ใน
// modal จะพลาด event นี้ ปุ่มไม่มีทางเปลี่ยนเป็น "ติดตั้ง" — จึงเก็บที่นี่ให้ renderer query ได้ทุกเมื่อ
let updateDownloadedInfo = null; // { version, file } เมื่อโหลดเสร็จ (file = path ตัวติดตั้ง), null เมื่อยังไม่มี
// portable build อัปเดตอัตโนมัติไม่ได้ (ไม่มี installer) — electron-builder ตั้ง env นี้ให้เฉพาะ portable
const isPortable = !!process.env.PORTABLE_EXECUTABLE_DIR;
// update feed (latest.yml + ตัวติดตั้ง NSIS) มีเฉพาะ Windows — build Mac ตั้ง publish: null จึงไม่มี app-update.yml
const isUpdatablePlatform = process.platform === "win32";
const UNSUPPORTED_PLATFORM_MSG = "เวอร์ชันนี้อัปเดตอัตโนมัติไม่ได้ — โหลดไฟล์ใหม่จากหน้า GitHub Releases";

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    title: "Machine Vision — เรียนรู้การประมวลผลภาพ",
    backgroundColor: "#1e1e1e",
    icon: appIcon,
    show: false, // กันหน้าต่างกะพริบดำตอนเปิด — โชว์ทีหลังตอนวาดเสร็จแล้วเท่านั้น (ดูจุด ready-to-show ด้านล่าง)
    webPreferences: {
      // ความปลอดภัย: ไม่เปิด nodeIntegration ใน renderer, แยก context
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      // ส่งเลขเวอร์ชันจริงให้ preload อ่าน — preload sandboxed require("../package.json") ไม่ได้
      additionalArguments: [`--app-version=${app.getVersion()}`],
    },
  });

  // แสดงหน้าต่างเมื่อ renderer วาดเฟรมแรกเสร็จจริงเท่านั้น — บางเครื่อง (GPU/driver ช้า
  // หรือช่วงโหลด OpenCV.js ที่หนัก) ถ้าโชว์หน้าต่างทันทีตั้งแต่สร้างจะเห็นเป็นจอว่าง/ดำกะพริบก่อนเนื้อหาขึ้น
  win.once("ready-to-show", () => win.show());
  mainWindow = win;
  win.on("closed", () => {
    if (mainWindow === win) mainWindow = null;
  });

  if (isDev) {
    win.loadURL("http://localhost:5173");
    if (openDevtools) win.webContents.openDevTools({ mode: "detach" });
  } else {
    win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
  return win;
}

// ส่งสถานะ auto-update กลับไปให้ renderer โชว์ผลจริง (แทนที่จะเงียบเวลาเช็คไม่เจอ/error)
function sendUpdateStatus(status, extra = {}) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("update:status", { status, ...extra });
  }
}

// ---------- IPC: บันทึกภาพ dataset ลงโฟลเดอร์เดียว (เลือกที่ครั้งเดียว) ----------
ipcMain.handle("export:chooseDir", async () => {
  const res = await dialog.showOpenDialog({
    title: "เลือกโฟลเดอร์สำหรับบันทึกภาพ dataset",
    properties: ["openDirectory", "createDirectory"],
  });
  if (res.canceled || res.filePaths.length === 0) return null;
  return res.filePaths[0];
});

// ---------- IPC: อ่านค่าประสิทธิภาพระบบ (Design §17) ----------
// รวม CPU% ของทุก process ในแอป + RAM ทั้งแอป จาก app.getAppMetrics()
// สีพื้นหน้าต่างตามธีมของ renderer (เห็นตอน resize/รีโหลดก่อน renderer วาดทัน) — dark = สีเดิม
ipcMain.on("theme:set", (e, theme) => {
  const win = BrowserWindow.fromWebContents(e.sender);
  if (win && !win.isDestroyed()) win.setBackgroundColor(theme === "dark" ? "#1e1e1e" : "#e8eae6");
});

ipcMain.handle("sys:metrics", async () => {
  try {
    const metrics = app.getAppMetrics();
    let cpu = 0;
    let mem = 0; // KB (workingSetSize)
    for (const m of metrics) {
      cpu += (m.cpu && m.cpu.percentCPUUsage) || 0;
      mem += (m.memory && m.memory.workingSetSize) || 0;
    }
    return { cpu, memMB: mem / 1024 };
  } catch {
    return null;
  }
});

ipcMain.handle("export:writeFile", async (_e, { dir, name, bytes }) => {
  // กันเขียนออกนอกโฟลเดอร์ที่เลือก: ใช้เฉพาะชื่อไฟล์ (ตัด path)
  const safe = path.basename(String(name || "file")).replace(/[\\/:*?"<>|]+/g, "_");
  await fs.writeFile(path.join(dir, safe), Buffer.from(bytes));
  return true;
});

// ---------- IPC: แปลงโมเดล .pt → .onnx ด้วย ultralytics ในเครื่อง ----------
// shell:false → args ไม่ถูก re-parse (กัน command injection จากชื่อไฟล์ที่มี ' ; & `)
// + timeout กัน python/yolo ค้าง (network/prompt/import แขวน) แล้ว IPC invoke แขวนตาม
function runCmd(cmd, args, cwd, timeoutMs = 180000) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { cwd, shell: false });
    let err = "";
    const timer = setTimeout(() => {
      try { p.kill(); } catch { /* ignore */ }
      reject(new Error("หมดเวลา (timeout) ขณะแปลงโมเดล"));
    }, timeoutMs);
    p.stderr.on("data", (d) => (err += d.toString()));
    p.on("error", (e) => { clearTimeout(timer); reject(e); });
    p.on("close", (code) => {
      clearTimeout(timer);
      code === 0 ? resolve() : reject(new Error(err || "exit " + code));
    });
  });
}

ipcMain.handle("dl:convertPt", async (_e, ptPath) => {
  const dir = path.dirname(ptPath);
  const onnxPath = path.join(dir, path.basename(ptPath, path.extname(ptPath)) + ".onnx");
  // ส่ง path เป็น sys.argv[1] แทนการฝังลงโค้ด → ไม่มีทางแตกออกจาก string literal
  const pyExport = "import sys; from ultralytics import YOLO; YOLO(sys.argv[1]).export(format='onnx')";
  // ลองหลายวิธี: yolo CLI → python → python3
  const attempts = [
    ["yolo", ["export", `model=${ptPath}`, "format=onnx"]],
    ["python", ["-c", pyExport, ptPath]],
    ["python3", ["-c", pyExport, ptPath]],
  ];
  let lastErr = "";
  for (const [cmd, args] of attempts) {
    try {
      await runCmd(cmd, args, dir);
      const bytes = await fs.readFile(onnxPath);
      return { ok: true, bytes: new Uint8Array(bytes), onnxPath };
    } catch (e) {
      lastErr = e.message || String(e);
    }
  }
  return { ok: false, error: lastErr || "แปลงไม่สำเร็จ" };
});

// ---------- IPC: ปุ่ม "เช็คอัปเดตตอนนี้" ในแผงตั้งค่า ----------
// autoUpdater.checkForUpdates() ยิง event เดียวกับตอนเช็คอัตโนมัติตอนเปิดแอป —
// sendUpdateStatus ส่งผลจริงกลับไปโชว์ ไม่ปล่อยให้เงียบเหมือน checkForUpdatesAndNotify() เดิม
ipcMain.handle("update:check", async () => {
  if (isDev) return { ok: false, error: "โหมด dev ไม่มี update feed ให้เช็ค" };
  if (isPortable) return { ok: false, error: "เวอร์ชัน portable อัปเดตอัตโนมัติไม่ได้ — โหลดตัวติดตั้ง (Setup) มาใช้แทน" };
  if (!isUpdatablePlatform) return { ok: false, error: UNSUPPORTED_PLATFORM_MSG };
  try {
    await autoUpdater.checkForUpdates();
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e?.message || String(e) };
  }
});

// ---------- IPC: อ่านสถานะอัปเดตปัจจุบัน (เรียกตอนเปิดหน้าตั้งค่า) ----------
// event update-downloaded ยิงครั้งเดียวและมักยิงก่อนผู้ใช้เปิดหน้าตั้งค่า — หน้าตั้งค่าจึงต้อง
// query สถานะตอนเปิด เพื่อรู้ว่าควรโชว์ปุ่ม "ติดตั้ง" เลยไหม (ไม่งั้นพลาด event ปุ่มไม่เปลี่ยน)
ipcMain.handle("update:getState", () => ({
  downloaded: !!updateDownloadedInfo,
  version: updateDownloadedInfo?.version,
  releaseNotes: updateDownloadedInfo?.releaseNotes,
  portable: isPortable,
}));

// ---------- IPC: ปุ่ม "ติดตั้งอัปเดต" (หลังโหลดอัปเดตเสร็จ) ----------
// เปิดตัวติดตั้งด้วย shell.openPath (= ShellExecute เหมือนผู้ใช้ดับเบิลคลิกไฟล์เอง) แทน
// autoUpdater.quitAndInstall() ที่ spawn แบบ silent (/S) ซึ่ง Windows Defender มักบล็อกเพราะ
// ดูเหมือน installer เถื่อน — วิธีนี้ตัวติดตั้งเด้ง UI ปกติ (ผู้ใช้กดต่อจนจบ) แล้วมันปิด-เปิดแอป
// ใหม่ให้เอง จบในโปรแกรม ไม่ต้องออกไปหาไฟล์เอง
ipcMain.handle("update:install", async () => {
  if (isDev) return { ok: false, error: "โหมด dev ติดตั้งอัปเดตไม่ได้" };
  if (isPortable) return { ok: false, error: "เวอร์ชัน portable อัปเดตอัตโนมัติไม่ได้" };
  if (!isUpdatablePlatform) return { ok: false, error: UNSUPPORTED_PLATFORM_MSG };
  if (!updateDownloadedInfo?.file) return { ok: false, error: "ยังไม่มีไฟล์อัปเดตที่โหลดเสร็จ" };
  const err = await shell.openPath(updateDownloadedInfo.file); // คืน "" เมื่อสำเร็จ, ข้อความ error เมื่อล้มเหลว
  if (err) return { ok: false, error: err };
  return { ok: true };
});

app.whenReady().then(() => {
  // Windows: ตั้ง AppUserModelID ให้ taskbar จัดกลุ่ม/แสดงไอคอนของแอปเราถูกต้อง
  if (process.platform === "win32") app.setAppUserModelId("com.irish.machinevision");

  // อนุญาตเฉพาะสิทธิ์กล้อง (media) — ปฏิเสธสิทธิ์อื่นทั้งหมด
  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === "media");
  });

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });

  // เช็ค+โหลด+ติดตั้งอัปเดตจาก GitHub Releases อัตโนมัติ (เฉพาะ build จริง — dev ไม่มี feed ให้เช็ค)
  // ใช้ได้เฉพาะกับตัว NSIS installer (perMachine:false) เท่านั้น — portable/unpacked ไม่รองรับ
  // ทุก event ส่งสถานะกลับไป renderer ด้วย (sendUpdateStatus) ให้ปุ่ม "เช็คอัปเดต" ในแผงตั้งค่าโชว์ผลจริงได้
  if (!isDev && !isPortable && isUpdatablePlatform) {
    autoUpdater.on("checking-for-update", () => sendUpdateStatus("checking"));
    autoUpdater.on("update-available", (info) => sendUpdateStatus("available", {
      version: info?.version, releaseNotes: info?.releaseNotes, releaseName: info?.releaseName,
    }));
    autoUpdater.on("update-not-available", () => sendUpdateStatus("not-available"));
    // ความคืบหน้าการโหลดไฟล์อัปเดต — ส่ง %/ไบต์ ให้ renderer โชว์แถบโหลด
    autoUpdater.on("download-progress", (p) => sendUpdateStatus("progress", {
      percent: p?.percent, transferred: p?.transferred, total: p?.total, bytesPerSecond: p?.bytesPerSecond,
    }));
    autoUpdater.on("update-downloaded", (info) => {
      // เก็บ path ไฟล์ติดตั้งที่โหลดเสร็จด้วย (info.downloadedFile) — ใช้เปิดแบบดับเบิลคลิกตอนกดติดตั้ง
      updateDownloadedInfo = { version: info?.version, file: info?.downloadedFile, releaseNotes: info?.releaseNotes };
      sendUpdateStatus("downloaded", { version: info?.version, releaseNotes: info?.releaseNotes });
    });
    autoUpdater.on("error", (err) => {
      const message = err?.message || String(err);
      console.error("[autoUpdater] error:", message);
      sendUpdateStatus("error", { message });
    });
    autoUpdater.checkForUpdatesAndNotify().catch((err) =>
      console.error("[autoUpdater] check failed:", err?.message || err)
    );
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

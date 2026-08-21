// โหมด Pose (MediaPipe) — ตรวจจับโครงร่างกายจากกล้องสด แล้ววาด skeleton ทับ
// โหลดไลบรารี + โมเดลแบบ lazy (หนัก ~6MB) เฉพาะเมื่อเปิดโหมดนี้ครั้งแรก
import { computeErgonomics, drawErgonomics } from "./ergonomics.js";

// เส้นที่วาด: เฉพาะลำตัว + แขน (ถึงข้อมือ) + ขา (ถึงข้อเท้า) — ไม่เอา หน้า/คอ/มือ/เท้า
const BODY_CONNECTIONS = [
  { start: 11, end: 12 }, // คานไหล่
  { start: 23, end: 24 }, // คานสะโพก
  { start: 11, end: 13 }, { start: 13, end: 15 }, // แขนซ้าย: ไหล่-ศอก-ข้อมือ
  { start: 12, end: 14 }, { start: 14, end: 16 }, // แขนขวา
  { start: 23, end: 25 }, { start: 25, end: 27 }, // ขาซ้าย: สะโพก-เข่า-ข้อเท้า
  { start: 24, end: 26 }, { start: 26, end: 28 }, // ขาขวา
];
// ลำตัว = เส้นกระดูกสันหลังเส้นเดียว (กลางไหล่ → กลางสะโพก)
function drawSpine(ctx, canvas, lm) {
  const midX = (a, b) => ((lm[a].x + lm[b].x) / 2) * canvas.width;
  const midY = (a, b) => ((lm[a].y + lm[b].y) / 2) * canvas.height;
  ctx.save();
  ctx.strokeStyle = "#00ff88";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(midX(11, 12), midY(11, 12)); // กลางไหล่
  ctx.lineTo(midX(23, 24), midY(23, 24)); // กลางสะโพก
  ctx.stroke();
  ctx.restore();
}
// จุดที่วาด (ไม่รวมจุดหน้า/มือ/เท้า)
const BODY_POINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

let landmarker = null;
let drawing = null;
let PoseCls = null;
let raf = 0;
let running = false;
let paused = false; // freeze ภาพนิ่ง (F3)

// โหลด/สร้าง PoseLandmarker (ครั้งเดียว) — ไฟล์ wasm + โมเดลอยู่ในเครื่อง (public/mediapipe)
async function ensureLandmarker(onStatus) {
  if (landmarker) return;
  onStatus?.("กำลังโหลดโมเดล Pose … (ครั้งแรกอาจใช้เวลาสักครู่)");
  const vision = await import("@mediapipe/tasks-vision");
  const { PoseLandmarker, FilesetResolver, DrawingUtils } = vision;
  PoseCls = PoseLandmarker;
  const fileset = await FilesetResolver.forVisionTasks("./mediapipe/wasm");
  let created;
  try {
    created = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "./mediapipe/pose_landmarker_lite.task", delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  } catch {
    // การ์ดจอบางเครื่องใช้ GPU ไม่ได้ → ถอยไป CPU
    created = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "./mediapipe/pose_landmarker_lite.task", delegate: "CPU" },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  }
  landmarker = created;
  drawing = null; // สร้างตอนมี ctx จริง
}

// เริ่มโหมด: วน detect ทุกเฟรมจาก video แล้ววาดลง canvas overlay
export async function startPose(videoEl, canvas, onStatus, onErgo) {
  await ensureLandmarker(onStatus);
  const ctx = canvas.getContext("2d");
  const { DrawingUtils } = await import("@mediapipe/tasks-vision");
  drawing = new DrawingUtils(ctx);
  running = true;
  onStatus?.("Pose พร้อม — ขยับตัวหน้ากล้องได้เลย");

  let lastTs = -1;
  const loop = () => {
    if (!running) return;
    // freeze: ไม่ดึงเฟรมใหม่/ไม่ detect — canvas คงภาพ+skeleton+ผลของเฟรมล่าสุดไว้
    if (paused) {
      raf = requestAnimationFrame(loop);
      return;
    }
    if (videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
      if (canvas.width !== videoEl.videoWidth) {
        canvas.width = videoEl.videoWidth;
        canvas.height = videoEl.videoHeight;
      }
      let ts = performance.now();
      if (ts <= lastTs) ts = lastTs + 1; // detectForVideo ต้องการเวลาเพิ่มขึ้นเสมอ
      lastTs = ts;
      let res = null;
      try {
        res = landmarker.detectForVideo(videoEl, ts);
      } catch {
        // ข้ามเฟรมที่ error (เช่น video ยังไม่พร้อม)
      }
      // วาดภาพกล้องเต็มเฟรม (แทน clearRect — canvas นี้แสดงผลเอง ไม่พึ่ง element video)
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
      if (res && res.landmarks) {
        for (const lm of res.landmarks) {
          drawing.drawConnectors(lm, BODY_CONNECTIONS, { color: "#00ff88", lineWidth: 3 });
          drawSpine(ctx, canvas, lm); // ลำตัวเป็นเส้นเดียว
          drawing.drawLandmarks(BODY_POINTS.map((i) => lm[i]), { color: "#00e0ff", radius: 4 });
        }
        // วิเคราะห์การยศาสตร์ (RULA) จากคนแรก แล้ววาดผล + เตือน
        const ergo = computeErgonomics(res.landmarks[0]);
        drawErgonomics(ctx, canvas, ergo);
        if (onErgo) onErgo(ergo);
      }
    }
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
}

// ---- วิเคราะห์ Ergonomics จาก "ภาพนิ่ง" (ภาพที่แคปไว้) ----
// ใช้ landmarker คนละตัว (โหมด IMAGE) จะได้ไม่กวนกับสตรีมกล้องสด (โหมด VIDEO)
let imgLandmarker = null;
async function ensureImageLandmarker(onStatus) {
  if (imgLandmarker) return;
  onStatus?.("กำลังเตรียมโมเดลวิเคราะห์ภาพนิ่ง …");
  const vision = await import("@mediapipe/tasks-vision");
  const { PoseLandmarker, FilesetResolver } = vision;
  const fileset = await FilesetResolver.forVisionTasks("./mediapipe/wasm");
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: "./mediapipe/pose_landmarker_lite.task", delegate },
    runningMode: "IMAGE",
    numPoses: 1,
  });
  try {
    imgLandmarker = await PoseLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch {
    imgLandmarker = await PoseLandmarker.createFromOptions(fileset, opts("CPU"));
  }
}

// วาดภาพ + skeleton + ผล Ergonomics ลง outCanvas จากภาพต้นทาง (canvas/image)
// คืน ergo (หรือ null ถ้าตรวจไม่พบคน)
export async function renderStillErgonomics(srcCanvas, outCanvas, onStatus) {
  await ensureImageLandmarker(onStatus);
  outCanvas.width = srcCanvas.width;
  outCanvas.height = srcCanvas.height;
  const ctx = outCanvas.getContext("2d");
  ctx.drawImage(srcCanvas, 0, 0);
  const { DrawingUtils } = await import("@mediapipe/tasks-vision");
  const du = new DrawingUtils(ctx);
  const res = imgLandmarker.detect(srcCanvas);
  if (!res || !res.landmarks || res.landmarks.length === 0) return null;
  const lm = res.landmarks[0];
  du.drawConnectors(lm, BODY_CONNECTIONS, { color: "#00ff88", lineWidth: 3 });
  drawSpine(ctx, outCanvas, lm);
  du.drawLandmarks(BODY_POINTS.map((i) => lm[i]), { color: "#00e0ff", radius: 4 });
  const ergo = computeErgonomics(lm);
  drawErgonomics(ctx, outCanvas, ergo);
  return ergo;
}

export function stopPose() {
  running = false;
  paused = false; // เคลียร์ freeze ทุกครั้งที่หยุด กัน state ค้าง
  if (raf) cancelAnimationFrame(raf);
  raf = 0;
}

// แคปภาพ (freeze) / กลับ live — ไม่ต้อง re-detect หรือโหลดโมเดลซ้ำ กลับ live ทันที
export function pausePose() {
  paused = true;
}
export function resumePose() {
  paused = false;
}

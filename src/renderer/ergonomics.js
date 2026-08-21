// วิเคราะห์การยศาสตร์ (Ergonomics) แบบ RULA อย่างง่าย จาก pose landmarks (MediaPipe)
// คำนวณมุมข้อต่อหลัก แล้วให้ระดับความเสี่ยง + คำเตือนเมื่อท่าทางผิดหลัก
//
// landmark index ของ MediaPipe Pose:
//   7 หูซ้าย, 8 หูขวา, 11 ไหล่ซ้าย, 12 ไหล่ขวา, 13 ศอกซ้าย, 14 ศอกขวา,
//   15 ข้อมือซ้าย, 16 ข้อมือขวา, 23 สะโพกซ้าย, 24 สะโพกขวา,
//   25 เข่าซ้าย, 26 เข่าขวา, 27 ข้อเท้าซ้าย, 28 ข้อเท้าขวา

const DEG = 180 / Math.PI;

function sub(a, b) {
  return { x: a.x - b.x, y: a.y - b.y };
}
// มุมของเวกเตอร์เทียบแนวดิ่ง (0° = ตั้งตรง)
function angleFromVertical(v) {
  return Math.atan2(Math.abs(v.x), Math.abs(v.y)) * DEG;
}
// มุมที่จุด b ระหว่างแขน a-b-c
function jointAngle(a, b, c) {
  const ba = sub(a, b);
  const bc = sub(c, b);
  const dot = ba.x * bc.x + ba.y * bc.y;
  const m = Math.hypot(ba.x, ba.y) * Math.hypot(bc.x, bc.y) || 1e-6;
  return Math.acos(Math.max(-1, Math.min(1, dot / m))) * DEG;
}

// ระดับความเสี่ยง: 0 = ดี, 1 = ระวัง, 2 = เสี่ยง
function riskTrunk(a) {
  return a < 10 ? 0 : a < 20 ? 1 : 2;
}
function riskUpperArm(a) {
  return a < 20 ? 0 : a < 45 ? 1 : 2;
}
function riskElbow(flex) {
  // ช่วงเหมาะ ~60–100° ของการงอศอก
  return flex >= 60 && flex <= 100 ? 0 : flex >= 45 && flex <= 120 ? 1 : 2;
}

// คำนวณ Ergonomics จาก landmarks ของคนเดียว (array 33 จุด)
export function computeErgonomics(lm) {
  if (!lm || lm.length < 29) return null;
  const vis = (i) => (lm[i] && lm[i].visibility != null ? lm[i].visibility : 1);
  // เลือกฝั่งที่กล้องเห็นชัดกว่า (เหมาะกับการถ่ายด้านข้าง)
  const leftScore = vis(11) + vis(13) + vis(15) + vis(23) + vis(25) + vis(27);
  const rightScore = vis(12) + vis(14) + vis(16) + vis(24) + vis(26) + vis(28);
  const useLeft = leftScore >= rightScore;
  const I = useLeft
    ? { sh: 11, el: 13, wr: 15, hip: 23 }
    : { sh: 12, el: 14, wr: 16, hip: 24 };
  const P = (i) => ({ x: lm[i].x, y: lm[i].y });

  const sh = P(I.sh), el = P(I.el), wr = P(I.wr), hip = P(I.hip);

  const trunk = angleFromVertical(sub(sh, hip)); // ก้มหลัง (0 = ตั้งตรง)
  const upperArm = jointAngle(el, sh, hip); // ยกต้นแขนเทียบลำตัว
  const elbowFlex = 180 - jointAngle(sh, el, wr); // การงอข้อศอก (0 = เหยียดตรง)

  const parts = [
    { key: "trunk", label: "หลัง", value: Math.round(trunk), risk: riskTrunk(trunk),
      warn: "หลังก้ม/บิดมาก — เสี่ยงปวดหลังส่วนล่าง" },
    { key: "upperArm", label: "ต้นแขน", value: Math.round(upperArm), risk: riskUpperArm(upperArm),
      warn: "ยกต้นแขนสูง/กางออกมาก — ล้าหัวไหล่" },
    { key: "elbow", label: "ข้อศอก", value: Math.round(elbowFlex), risk: riskElbow(elbowFlex),
      warn: "มุมข้อศอกไม่เหมาะ (ควร ~60–100°)" },
  ];

  const level = Math.max(...parts.map((p) => p.risk));
  const warnings = parts.filter((p) => p.risk >= 2).map((p) => p.warn);

  return { side: useLeft ? "ซ้าย" : "ขวา", parts, level, warnings };
}

const RISK_COLOR = ["#22c55e", "#f59e0b", "#ef4444"]; // ดี / ระวัง / เสี่ยง
const RISK_TEXT = ["ดี", "ระวัง", "เสี่ยง"];

// วาดแผงผลวิเคราะห์ + คำเตือนลงบน overlay canvas
export function drawErgonomics(ctx, canvas, ergo) {
  if (!ergo) return;
  const W = canvas.width;
  const s = W / 640; // สเกลตามความกว้างจริงของภาพ
  const pad = 12 * s;
  const lineH = 26 * s;
  const boxW = 250 * s;
  const boxH = lineH * (ergo.parts.length + 1) + pad;

  ctx.save();
  // พื้นแผง
  ctx.fillStyle = "rgba(15,23,42,0.78)";
  ctx.fillRect(pad, pad, boxW, boxH);
  ctx.font = `${15 * s}px sans-serif`;
  ctx.textBaseline = "middle";

  // หัวข้อ
  ctx.fillStyle = "#e2e8f0";
  ctx.fillText(`Ergonomics (ฝั่ง${ergo.side})`, pad * 1.6, pad + lineH * 0.6);

  ergo.parts.forEach((p, i) => {
    const y = pad + lineH * (i + 1.6);
    ctx.fillStyle = RISK_COLOR[p.risk];
    // จุดสถานะ
    ctx.beginPath();
    ctx.arc(pad * 1.8, y, 5 * s, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#e2e8f0";
    ctx.fillText(`${p.label}: ${p.value}°  (${RISK_TEXT[p.risk]})`, pad * 3.2, y);
  });
  ctx.restore();

  // แถบเตือนใหญ่เมื่อมีท่าเสี่ยง
  if (ergo.level >= 2) {
    ctx.save();
    const bh = 46 * s;
    ctx.fillStyle = "rgba(239,68,68,0.9)";
    ctx.fillRect(0, 0, W, bh);
    ctx.fillStyle = "#fff";
    ctx.font = `bold ${20 * s}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("⚠ ท่าทางเสี่ยง: " + ergo.warnings.join(" · "), W / 2, bh / 2);
    // กรอบแดงเตือน
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 6 * s;
    ctx.strokeRect(0, 0, W, canvas.height);
    ctx.restore();
  }
}

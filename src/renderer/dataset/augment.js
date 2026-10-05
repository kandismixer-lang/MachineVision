// Offline augmentation (แบบ Roboflow) — สร้างภาพก็อปที่แปลงแล้วตอน export
// รองรับ image-level augmentation ครบชุด:
//   geometry (แปลงพิกัด label ตาม): flipH, flipV, rotation, crop(zoom), shear
//   color/pixel (ไม่ขยับกรอบ): grayscale, hue, saturation, brightness, exposure, blur, noise
//
// ใช้เฉพาะภาพ split=train (valid/test ต้องเป็นภาพจริง)
// annotation เป็นพิกัด normalized 0..1 — geometry แปลงผ่าน affine matrix (pixel space)
//
// ไม่พึ่ง state ของแอป — สร้าง canvas ชั่วคราวเองเพื่อวาดภาพที่แปลงแล้ว

// ---------- config เริ่มต้น ----------
export function defaultAugmentConfig() {
  return {
    copies: 3,
    flipH: { on: false },
    flipV: { on: false },
    rotation: { on: false, deg: 15 },
    crop: { on: false, pct: 20 },
    shear: { on: false, deg: 10 },
    grayscale: { on: false },
    hue: { on: false, deg: 25 },
    saturation: { on: false, pct: 30 },
    brightness: { on: false, pct: 25 },
    exposure: { on: false, pct: 25 },
    blur: { on: false, px: 2 },
    noise: { on: false, pct: 5 },
  };
}

// รวม config ที่บันทึกไว้กับ default (กันฟิลด์ขาดเวลาเพิ่ม aug ใหม่)
export function normalizeAugmentConfig(v) {
  const d = defaultAugmentConfig();
  if (!v || typeof v !== "object") return d;
  const out = { copies: clampInt(v.copies, 0, 10, d.copies) };
  for (const k of Object.keys(d)) {
    if (k === "copies") continue;
    const src = v[k] || {};
    out[k] = { ...d[k], ...src, on: !!src.on };
  }
  return out;
}

function clampInt(v, lo, hi, dflt) {
  let n = Math.round(Number(v));
  if (!Number.isFinite(n)) n = dflt;
  return Math.max(lo, Math.min(hi, n));
}

const GEOM_KEYS = ["flipH", "flipV", "rotation", "crop", "shear"];
const HAS_AUG_KEYS = [...GEOM_KEYS, "grayscale", "hue", "saturation", "brightness", "exposure", "blur", "noise"];

// เปิดใช้ไหม: copies > 0 และเลือก aug อย่างน้อย 1 อย่าง
export function augmentEnabled(cfg) {
  if (!cfg || cfg.copies <= 0) return false;
  return HAS_AUG_KEYS.some((k) => cfg[k] && cfg[k].on);
}

// ป้ายสรุปสั้น ๆ ให้ UI (เช่น "×3 · flip, rotation, blur")
const AUG_LABELS = {
  flipH: "flip", flipV: "flip-v", rotation: "rotation", crop: "crop", shear: "shear",
  grayscale: "grayscale", hue: "hue", saturation: "saturation",
  brightness: "brightness", exposure: "exposure", blur: "blur", noise: "noise",
};
export function augmentSummaryText(cfg) {
  if (!augmentEnabled(cfg)) return "ปิด — เพิ่มภาพเทรนแบบ Roboflow";
  const on = HAS_AUG_KEYS.filter((k) => cfg[k] && cfg[k].on).map((k) => AUG_LABELS[k]);
  return `×${cfg.copies} · ${on.join(", ")}`;
}

// ---------- affine matrix (pixel space) ----------
// x' = a*x + c*y + e ; y' = b*x + d*y + f  (ตรงกับ canvas setTransform)
const I = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
function mul(B, A) {
  // ผลลัพธ์ = ทำ A ก่อน แล้ว B
  return {
    a: B.a * A.a + B.c * A.b,
    b: B.b * A.a + B.d * A.b,
    c: B.a * A.c + B.c * A.d,
    d: B.b * A.c + B.d * A.d,
    e: B.a * A.e + B.c * A.f + B.e,
    f: B.b * A.e + B.d * A.f + B.f,
  };
}
const translate = (tx, ty) => ({ a: 1, b: 0, c: 0, d: 1, e: tx, f: ty });
const scale = (sx, sy) => ({ a: sx, b: 0, c: 0, d: sy, e: 0, f: 0 });
const rotate = (rad) => ({ a: Math.cos(rad), b: Math.sin(rad), c: -Math.sin(rad), d: Math.cos(rad), e: 0, f: 0 });
const shearM = (shx, shy) => ({ a: 1, b: shy, c: shx, d: 1, e: 0, f: 0 });
// ทำ transform M รอบจุดศูนย์ (cx,cy)
const around = (M, cx, cy) => mul(translate(cx, cy), mul(M, translate(-cx, -cy)));

function applyMat(M, x, y) {
  return { x: M.a * x + M.c * y + M.e, y: M.b * x + M.d * y + M.f };
}

const rand = (lo, hi) => lo + Math.random() * (hi - lo);

// ---------- สร้าง plan 1 ชุดสำหรับ 1 ก็อป ----------
// preview=true → ใช้ค่าสูงสุดแบบคงที่ (โชว์ตัวอย่าง before/after ให้เห็นชัด)
export function buildPlan(cfg, W, H, preview = false) {
  const cx = W / 2, cy = H / 2;
  let M = I;
  let hasGeom = false;

  if (cfg.flipH?.on && (preview || Math.random() < 0.5)) {
    M = mul(around(scale(-1, 1), cx, cy), M);
    hasGeom = true;
  }
  if (cfg.flipV?.on && (preview || Math.random() < 0.5)) {
    M = mul(around(scale(1, -1), cx, cy), M);
    hasGeom = true;
  }
  if (cfg.rotation?.on) {
    const deg = preview ? cfg.rotation.deg : rand(-cfg.rotation.deg, cfg.rotation.deg);
    M = mul(around(rotate((deg * Math.PI) / 180), cx, cy), M);
    hasGeom = true;
  }
  if (cfg.shear?.on) {
    const s = preview ? cfg.shear.deg : rand(-cfg.shear.deg, cfg.shear.deg);
    M = mul(around(shearM(Math.tan((s * Math.PI) / 180), 0), cx, cy), M);
    hasGeom = true;
  }
  if (cfg.crop?.on) {
    const z = 1 + (preview ? cfg.crop.pct : rand(0, cfg.crop.pct)) / 100; // ซูมเข้า
    // เลือกจุดซูมสุ่มในภาพ (preview = กลาง)
    const px = preview ? cx : rand(0, W);
    const py = preview ? cy : rand(0, H);
    M = mul(around(scale(z, z), px, py), M);
    hasGeom = true;
  }

  // ---- color/pixel filter ----
  const parts = [];
  if (cfg.grayscale?.on) parts.push("grayscale(1)");
  if (cfg.hue?.on) parts.push(`hue-rotate(${(preview ? cfg.hue.deg : rand(-cfg.hue.deg, cfg.hue.deg)).toFixed(1)}deg)`);
  if (cfg.saturation?.on) parts.push(`saturate(${(1 + (preview ? cfg.saturation.pct : rand(-cfg.saturation.pct, cfg.saturation.pct)) / 100).toFixed(3)})`);
  if (cfg.brightness?.on) parts.push(`brightness(${(1 + (preview ? cfg.brightness.pct : rand(-cfg.brightness.pct, cfg.brightness.pct)) / 100).toFixed(3)})`);
  if (cfg.exposure?.on) parts.push(`contrast(${(1 + (preview ? cfg.exposure.pct : rand(-cfg.exposure.pct, cfg.exposure.pct)) / 100).toFixed(3)})`);
  if (cfg.blur?.on) parts.push(`blur(${(preview ? cfg.blur.px : rand(0, cfg.blur.px)).toFixed(2)}px)`);

  const noiseStd = cfg.noise?.on ? (preview ? cfg.noise.pct : rand(0, cfg.noise.pct)) * 2.55 : 0;

  const filter = parts.join(" ");
  // identity = ก็อปนี้ไม่ต่างจากต้นฉบับเลย (เช่น เปิด Flip อย่างเดียวแล้วสุ่มไม่พลิก) → ผู้เรียกข้ามได้ กันภาพซ้ำ
  const identity = !hasGeom && !filter && noiseStd === 0;
  return { matrix: M, hasGeom, filter, noiseStd, identity };
}

// ---------- วาดภาพจาก plan ----------
export function applyPlan(src, plan, outW, outH) {
  const w = outW || src.width, h = outH || src.height;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (plan.filter) ctx.filter = plan.filter;
  const M = plan.matrix;
  // ถ้า output ย่อขนาด ต้อง scale matrix ตามสัดส่วน (preview thumbnail)
  const sx = w / src.width, sy = h / src.height;
  if (sx !== 1 || sy !== 1) {
    // S*M : a=sx·a, b=sy·b, c=sx·c, d=sy·d, e=sx·e, f=sy·f
    ctx.setTransform(M.a * sx, M.b * sy, M.c * sx, M.d * sy, M.e * sx, M.f * sy);
  } else {
    ctx.setTransform(M.a, M.b, M.c, M.d, M.e, M.f);
  }
  ctx.drawImage(src, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.filter = "none";
  if (plan.noiseStd > 0) addNoise(ctx, w, h, plan.noiseStd);
  return c;
}

function addNoise(ctx, w, h, std) {
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = clamp8(d[i] + gauss() * std);
    d[i + 1] = clamp8(d[i + 1] + gauss() * std);
    d[i + 2] = clamp8(d[i + 2] + gauss() * std);
  }
  ctx.putImageData(img, 0, 0);
}
function clamp8(v) { return v < 0 ? 0 : v > 255 ? 255 : v; }
function gauss() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- แปลง annotation ตาม plan (geometry เท่านั้น) ----------
// คืน array ใหม่ — ตัด annotation ที่หลุดออกนอกภาพหมดทิ้ง
export function transformAnnotations(anns, plan, W, H) {
  if (!plan.hasGeom) return anns;
  const M = plan.matrix;
  const out = [];
  for (const a of anns) {
    if (a.type === "polygon" || a.type === "line") {
      const pts = a.points.map((p) => {
        const q = applyMat(M, p.x * W, p.y * H);
        return { x: clamp01(q.x / W), y: clamp01(q.y / H) };
      });
      // drop เมื่อ degenerate — AABB ของจุดที่ clamp แล้วบางไปในแกนใดแกนหนึ่ง
      // (polygon: ต้องมีทั้งกว้างและสูง · line: ต้องไม่ยุบเป็นจุดเดียว)
      let pminX = 1, pminY = 1, pmaxX = 0, pmaxY = 0;
      for (const p of pts) {
        if (p.x < pminX) pminX = p.x;
        if (p.y < pminY) pminY = p.y;
        if (p.x > pmaxX) pmaxX = p.x;
        if (p.y > pmaxY) pmaxY = p.y;
      }
      const pw = pmaxX - pminX, ph = pmaxY - pminY;
      if (a.type === "polygon") {
        if (pw <= 0.001 || ph <= 0.001) continue; // ยุบเป็นเส้น/จุด = พื้นที่ ~0
      } else {
        if (pw <= 0.001 && ph <= 0.001) continue; // line ยุบเป็นจุด
      }
      out.push({ ...a, points: pts });
    } else {
      // bbox → 4 มุม → transform → AABB → clip
      const l = (a.cx - a.w / 2) * W, r = (a.cx + a.w / 2) * W;
      const t = (a.cy - a.h / 2) * H, b = (a.cy + a.h / 2) * H;
      const corners = [applyMat(M, l, t), applyMat(M, r, t), applyMat(M, r, b), applyMat(M, l, b)];
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const q of corners) {
        if (q.x < minX) minX = q.x;
        if (q.y < minY) minY = q.y;
        if (q.x > maxX) maxX = q.x;
        if (q.y > maxY) maxY = q.y;
      }
      minX = Math.max(0, Math.min(W, minX));
      maxX = Math.max(0, Math.min(W, maxX));
      minY = Math.max(0, Math.min(H, minY));
      maxY = Math.max(0, Math.min(H, maxY));
      const nw = (maxX - minX) / W, nh = (maxY - minY) / H;
      if (nw <= 0.001 || nh <= 0.001) continue; // กรอบเล็กเกิน/หลุดภาพ
      out.push({ ...a, cx: (minX + maxX) / 2 / W, cy: (minY + maxY) / 2 / H, w: nw, h: nh });
    }
  }
  return out;
}
function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

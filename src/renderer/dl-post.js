// กล่องหลัง Model Detect (2026-11-26): 📍 Centroid + 🔢 Object Count (เส้นนับ / โซนนับ)
// ต่อสาย: แหล่งภาพ → Model Detect → [Centroid] → [Object Count …] → ผลลัพธ์
// ไฟล์นี้เป็นตรรกะล้วน (ไม่แตะ DOM นอกจาก ctx ที่ส่งมาวาด) — โค้ด Python ใน codegen-dl.js ทำตามทีละขั้นแบบเดียวกัน
//   centroid:  "box"   = กึ่งกลางกรอบ
//              "shape" = จุดศูนย์ถ่วงของรูปทรงชิ้นงานในกรอบ (mask ของโมเดล seg ถ้ามี ไม่งั้น Otsu ในกรอบ → moments)
//   tracker:   จับคู่จุดของเฟรมนี้กับ "ตำแหน่งที่ทำนาย" ของชิ้นเดิม (ตำแหน่งล่าสุด + ความเร็ว × เวลาที่ผ่านไป) แบบใกล้สุดก่อน
//              ภายในระยะ maxDist (% ของเส้นทแยงภาพ ต่อ 1/30 วินาที — ตรวจห่างกว่านั้นขยายระยะตามเวลา สูงสุด 4 เท่า)
//              → ตรวจช้า (เช่น CPU/WASM ตรวจได้ ~5 ครั้ง/วินาที ขณะวิดีโอเล่น 30 fps) ก็ยังตามชิ้นเดิมได้
//   line:      นับเมื่อชิ้นงาน (track) ข้ามเส้นตรง A→B · ชิ้นละครั้งต่อหนึ่งเส้น · แยกทิศ IN / OUT
//   zone:      NOW = จำนวนชิ้นในกรอบตอนนี้ · TOTAL = ชิ้น (track) ที่เคยเข้ากรอบ นับชิ้นละครั้ง

export const CENTROID_DEFAULTS = Object.freeze({ method: "shape", showCoords: true });
export const COUNT_DEFAULTS = Object.freeze({
  mode: "line", // "line" | "zone"
  // เส้นนับ (เปอร์เซ็นต์ของกว้าง/สูงภาพ) — ค่าเริ่ม = เส้นตั้งกลางภาพ
  x1: 50, y1: 5, x2: 50, y2: 95,
  dir: "both", // "both" | "in" | "out"
  // โซนนับ (เปอร์เซ็นต์) — มุมซ้ายบน + กว้าง/สูง
  zx: 30, zy: 20, zw: 40, zh: 60,
  cls: "", // "" = ทุกคลาส · ไม่งั้น = ชื่อคลาสที่จะนับ
  maxDist: 12, // ระยะจับคู่ชิ้นเดิมข้ามเฟรม (% ของเส้นทแยงภาพ)
  maxMissed: 10, // ชิ้นหายจากการตรวจได้กี่ครั้งก่อนลืม
});

export const FRAME_DT = 1 / 30; // ช่วงเวลาอ้างอิงของ maxDist (1 เฟรมที่ 30 fps)
export const MAX_GATE_SCALE = 4; // ตรวจห่างมาก → ขยายระยะจับคู่ได้ไม่เกินกี่เท่า
const num = (v, d) => (Number.isFinite(Number(v)) ? Number(v) : d);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export function normCentroid(p = {}) {
  return {
    method: p.method === "box" ? "box" : "shape",
    showCoords: p.showCoords !== false,
  };
}

export function normCount(p = {}) {
  const d = COUNT_DEFAULTS;
  const pct = (k) => clamp(num(p[k], d[k]), 0, 100);
  return {
    mode: p.mode === "zone" ? "zone" : "line",
    x1: pct("x1"), y1: pct("y1"), x2: pct("x2"), y2: pct("y2"),
    dir: ["both", "in", "out"].includes(p.dir) ? p.dir : "both",
    zx: pct("zx"), zy: pct("zy"), zw: pct("zw"), zh: pct("zh"),
    cls: typeof p.cls === "string" ? p.cls : "",
    maxDist: clamp(num(p.maxDist, d.maxDist), 1, 100),
    maxMissed: Math.round(clamp(num(p.maxMissed, d.maxMissed), 0, 300)),
  };
}

// ---------- centroid ----------

const jsRound = (x) => Math.floor(x + 0.5);

// ขอบเขตพิกเซลของกรอบบนภาพ (ตัดให้อยู่ในภาพ) — ใช้ร่วมกับ Python: floor ซ้าย/บน · ceil ขวา/ล่าง
export function boxPixelRect(b, W, H) {
  const x0 = clamp(Math.floor(b.x), 0, W), y0 = clamp(Math.floor(b.y), 0, H);
  const x1 = clamp(Math.ceil(b.x + b.w), 0, W), y1 = clamp(Math.ceil(b.y + b.h), 0, H);
  return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 };
}

// เกณฑ์ Otsu จาก histogram 256 ช่อง (ค่าที่ > เกณฑ์ = กลุ่มสว่าง) · เท่ากันเอาเกณฑ์ตัวแรก
export function otsuThreshold(hist, total) {
  let sumAll = 0;
  for (let i = 0; i < 256; i++) sumAll += i * hist[i];
  let wB = 0, sumB = 0, best = -1, thr = 0;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sumAll - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) { best = between; thr = t; }
  }
  return thr;
}

// จุดศูนย์ถ่วงของชิ้นงานในกรอบจาก RGBA (พิกัดในกรอบ) — คืน null ถ้าแยกชิ้นงานจากพื้นไม่ได้
//   gray = (299 R + 587 G + 114 B + 500) // 1000 · Otsu · กลุ่มที่แตะขอบกรอบมากกว่า = พื้นหลัง
export function shapeCentroidRGBA(data, w, h) {
  if (w < 2 || h < 2) return null;
  const n = w * h;
  const gray = new Uint8Array(n);
  const hist = new Float64Array(256);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const g = Math.floor((299 * data[p] + 587 * data[p + 1] + 114 * data[p + 2] + 500) / 1000);
    gray[i] = g;
    hist[g]++;
  }
  const t = otsuThreshold(hist, n);
  let borderFg = 0, border = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (y !== 0 && y !== h - 1 && x !== 0 && x !== w - 1) continue;
      border++;
      if (gray[y * w + x] > t) borderFg++;
    }
  }
  const fgBright = borderFg * 2 <= border; // สว่างแตะขอบไม่เกินครึ่ง → ชิ้นงานคือส่วนสว่าง
  let m00 = 0, m10 = 0, m01 = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if ((gray[y * w + x] > t) !== fgBright) continue;
      m00++; m10 += x; m01 += y;
    }
  }
  if (m00 === 0 || m00 === n) return null;
  return { x: m10 / m00 + 0.5, y: m01 / m00 + 0.5, area: m00 };
}

// mask ของ YOLO-seg (ครอปในกรอบ ความละเอียด proto) → จุดศูนย์ถ่วงเป็นสัดส่วน 0..1 ของกรอบ
export function maskCentroidFrac(mask) {
  if (!mask || !mask.mw || !mask.mh || !mask.data) return null;
  const { mw: w, mh: h, data } = mask;
  let m00 = 0, m10 = 0, m01 = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (data[y * w + x] < 128) continue;
      m00++; m10 += x + 0.5; m01 += y + 0.5;
    }
  }
  return m00 ? { fx: m10 / m00 / w, fy: m01 / m00 / h } : null;
}

// คำนวณจุดของทุกกล่อง · readRGBA(rect) → Uint8ClampedArray ของพื้นที่นั้น (ใช้เมื่อ method = shape)
// คืน [{ x, y, cls, label, score, box, method }] — method = "mask" | "shape" | "box" (ที่ใช้จริง)
export function computeCentroids(boxes, params, W, H, readRGBA) {
  const p = normCentroid(params);
  return (boxes || []).map((b) => {
    const base = { cls: b.cls, label: b.label, score: b.score, box: b };
    if (p.method === "shape") {
      const m = b.mask && b.mask.data ? maskCentroidFrac(b.mask) : null;
      if (m) return { ...base, x: b.x + m.fx * b.w, y: b.y + m.fy * b.h, method: "mask" };
      const r = boxPixelRect(b, W, H);
      if (readRGBA && r.w >= 2 && r.h >= 2) {
        let c = null;
        try { c = shapeCentroidRGBA(readRGBA(r), r.w, r.h); } catch { c = null; }
        if (c) return { ...base, x: r.x0 + c.x, y: r.y0 + c.y, method: "shape" };
      }
    }
    return { ...base, x: b.x + b.w / 2, y: b.y + b.h / 2, method: "box" };
  });
}

// ---------- tracker ----------

export function createTracker() {
  let tracks = [];
  let nextId = 1;
  return {
    reset() { tracks = []; nextId = 1; },
    get tracks() { return tracks; },
    // pts = [{x, y, cls, label}] · maxDistPx = ระยะจับคู่ต่อ FRAME_DT · dt = วินาทีตั้งแต่ update ครั้งก่อน
    // คืน tracks ที่ยังจำอยู่ (track.missed === 0 = เจอเฟรมนี้)
    update(pts, maxDistPx, maxMissed, dt = FRAME_DT) {
      const step = Number.isFinite(dt) && dt > 0 ? dt : 0;
      const pairs = [];
      tracks.forEach((t, ti) => {
        const since = t.since + step; // เวลาตั้งแต่เห็นชิ้นนี้ครั้งล่าสุด
        const ex = t.x + t.vx * since, ey = t.y + t.vy * since; // ตำแหน่งที่ทำนาย
        const gate = maxDistPx * Math.min(MAX_GATE_SCALE, Math.max(1, since / FRAME_DT));
        pts.forEach((p, pi) => {
          const d = Math.hypot(p.x - ex, p.y - ey);
          if (d <= gate) pairs.push([d, ti, pi]);
        });
      });
      pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
      const usedT = new Set(), usedP = new Set();
      for (const [, ti, pi] of pairs) {
        if (usedT.has(ti) || usedP.has(pi)) continue;
        usedT.add(ti); usedP.add(pi);
        const t = tracks[ti], p = pts[pi];
        const since = t.since + step;
        if (since > 0) { // ความเร็ว (px/วินาที) เฉลี่ยกับค่าเดิมครึ่งต่อครึ่ง กันกระตุก
          const vx = (p.x - t.x) / since, vy = (p.y - t.y) / since;
          if (t.hits > 1) { t.vx = (t.vx + vx) / 2; t.vy = (t.vy + vy) / 2; } else { t.vx = vx; t.vy = vy; }
        }
        t.x = p.x; t.y = p.y; t.cls = p.cls; t.label = p.label;
        t.missed = 0; t.since = 0; t.hits++;
      }
      const kept = [];
      tracks.forEach((t, ti) => {
        if (!usedT.has(ti)) { t.missed++; t.since += step; }
        if (t.missed <= maxMissed) kept.push(t);
      });
      pts.forEach((p, pi) => {
        if (usedP.has(pi)) return;
        kept.push({ id: nextId++, x: p.x, y: p.y, vx: 0, vy: 0, since: 0, cls: p.cls, label: p.label, missed: 0, hits: 1 });
      });
      tracks = kept;
      return tracks;
    },
  };
}

// ---------- counter ----------

const cross = (ax, ay, bx, by) => ax * by - ay * bx;
const sign = (v) => (v > 0 ? 1 : v < 0 ? -1 : 0);

// ด้านของจุดเทียบเส้น A→B: +1 / -1 / 0 (อยู่บนเส้น) · ฝั่ง -1 = ฝั่งที่ลูกศร IN ชี้ไป
export function sideOf(ax, ay, bx, by, x, y) {
  return sign(cross(bx - ax, by - ay, x - ax, y - ay));
}

// ส่วนของเส้น P0→P1 ตัดเส้น A→B (รวมปลายเส้น A/B) ไหม — ใช้เมื่อรู้แล้วว่า P0, P1 อยู่คนละฝั่งของเส้นตรง AB
export function crossesSegment(ax, ay, bx, by, x0, y0, x1, y1) {
  const s0 = sign(cross(x1 - x0, y1 - y0, ax - x0, ay - y0));
  const s1 = sign(cross(x1 - x0, y1 - y0, bx - x0, by - y0));
  return s0 === 0 || s1 === 0 || s0 !== s1;
}

const NAMES_DEF = { line: "Line", zone: "Zone" };

// ตัวนับหนึ่งกล่อง — มี tracker ของตัวเอง (กล่องนับหลายกล่องบนสายเดียวกันไม่รบกวนกัน)
export function createCounter() {
  const tracker = createTracker();
  let st = null;
  const fresh = () => ({ in: 0, out: 0, total: 0, now: 0, byClass: {}, counted: new Set(), ref: new Map(), events: [] });
  st = fresh();
  return {
    reset() { tracker.reset(); st = fresh(); },
    get tracks() { return tracker.tracks; },
    get state() { return st; },
    // pts = จุดของเฟรมนี้ · still = ภาพนิ่ง (ไม่ติดตามข้ามภาพ: เริ่มนับใหม่ทุกภาพ)
    update(pts, params, W, H, still = false, dt = FRAME_DT) {
      const p = normCount(params);
      if (still) this.reset();
      st.events = [];
      const diag = Math.hypot(W, H);
      const tracks = tracker.update(pts, (p.maxDist / 100) * diag, p.maxMissed, dt);
      const want = (t) => !p.cls || t.label === p.cls;
      const bump = (t, key) => {
        st[key]++;
        st.total++;
        const k = t.label ?? String(t.cls);
        st.byClass[k] = (st.byClass[k] || 0) + 1;
        st.events.push({ id: t.id, label: k, kind: key });
      };
      if (p.mode === "zone") {
        const zx = (p.zx / 100) * W, zy = (p.zy / 100) * H, zx2 = zx + (p.zw / 100) * W, zy2 = zy + (p.zh / 100) * H;
        let now = 0;
        for (const t of tracks) {
          if (t.missed !== 0 || !want(t)) continue;
          if (t.x < zx || t.x > zx2 || t.y < zy || t.y > zy2) continue;
          now++;
          if (!st.counted.has(t.id)) { st.counted.add(t.id); bump(t, "in"); }
        }
        st.now = now;
      } else {
        const ax = (p.x1 / 100) * W, ay = (p.y1 / 100) * H, bx = (p.x2 / 100) * W, by = (p.y2 / 100) * H;
        const alive = new Set();
        for (const t of tracks) {
          alive.add(t.id);
          if (t.missed !== 0) continue;
          const s = sideOf(ax, ay, bx, by, t.x, t.y);
          if (s === 0) continue;
          const r = st.ref.get(t.id);
          st.ref.set(t.id, { s, x: t.x, y: t.y });
          if (!r || r.s === s || st.counted.has(t.id) || !want(t)) continue;
          if (!crossesSegment(ax, ay, bx, by, r.x, r.y, t.x, t.y)) continue;
          const key = r.s > 0 ? "in" : "out"; // จากฝั่ง + ไปฝั่ง − = IN (ทิศลูกศร)
          if (p.dir !== "both" && p.dir !== key) continue;
          st.counted.add(t.id);
          bump(t, key);
        }
        for (const id of st.ref.keys()) if (!alive.has(id)) st.ref.delete(id);
        st.now = 0;
      }
      return st;
    },
  };
}

// ข้อความสรุปบนภาพ (อังกฤษ ให้ตรงกับ cv2.putText ในโค้ด Python ที่เขียนไทยไม่ได้)
export function counterText(name, params, st) {
  const p = normCount(params);
  if (p.mode === "zone") return `${name}: NOW ${st.now}  TOTAL ${st.total}`;
  if (p.dir === "in") return `${name}: IN ${st.in}`;
  if (p.dir === "out") return `${name}: OUT ${st.out}`;
  return `${name}: ${st.total}  (IN ${st.in} / OUT ${st.out})`;
}

export function counterName(params, index) {
  return `${NAMES_DEF[normCount(params).mode]} ${index}`;
}

// ---------- วาด ----------

export const POST_COLORS = Object.freeze({
  centroid: "#ff3b5c",
  line: "#ffd60a",
  zone: "#34d399",
  panel: "rgba(0,0,0,0.6)",
  text: "#ffffff",
});

export function drawCentroids(ctx, pts, params, scale = 1) {
  const p = normCentroid(params);
  ctx.save();
  ctx.font = `${12 * scale}px sans-serif`;
  ctx.textBaseline = "alphabetic";
  for (const c of pts) {
    ctx.beginPath();
    ctx.arc(c.x, c.y, 4 * scale, 0, Math.PI * 2);
    ctx.fillStyle = POST_COLORS.centroid;
    ctx.fill();
    ctx.lineWidth = 1.5 * scale;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
    if (p.showCoords) {
      const text = `(${jsRound(c.x)}, ${jsRound(c.y)})`;
      // ชิดขอบขวา → เขียนทางซ้ายของจุดแทน (ไม่ให้ตัวเลขล้นภาพ)
      const tw = ctx.measureText(text).width;
      const cw = ctx.canvas?.width || Infinity;
      const tx = c.x + 6 * scale + tw > cw ? c.x - 6 * scale - tw : c.x + 6 * scale;
      ctx.lineWidth = 3 * scale;
      ctx.strokeStyle = "rgba(0,0,0,0.75)";
      ctx.strokeText(text, tx, c.y - 6 * scale);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, tx, c.y - 6 * scale);
    }
  }
  ctx.restore();
}

// วาดเส้น/โซน (ไม่มีป้ายสรุป) — ใช้ทั้งตอนวาดผลและตอนลากวาดเส้นใหม่
export function drawCounterShape(ctx, params, W, H, scale = 1, flash = false) {
  const p = normCount(params);
  ctx.save();
  if (p.mode === "zone") {
    const x = (p.zx / 100) * W, y = (p.zy / 100) * H, w = (p.zw / 100) * W, h = (p.zh / 100) * H;
    ctx.fillStyle = flash ? "rgba(52,211,153,0.35)" : "rgba(52,211,153,0.15)";
    ctx.fillRect(x, y, w, h);
    ctx.lineWidth = 2 * scale;
    ctx.strokeStyle = POST_COLORS.zone;
    ctx.strokeRect(x, y, w, h);
  } else {
    const ax = (p.x1 / 100) * W, ay = (p.y1 / 100) * H, bx = (p.x2 / 100) * W, by = (p.y2 / 100) * H;
    ctx.lineWidth = (flash ? 4 : 2.5) * scale;
    ctx.strokeStyle = flash ? "#22ff88" : POST_COLORS.line;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.fillStyle = POST_COLORS.line;
    for (const [x, y] of [[ax, ay], [bx, by]]) { ctx.beginPath(); ctx.arc(x, y, 3.5 * scale, 0, Math.PI * 2); ctx.fill(); }
    // ลูกศรทิศ IN: ตั้งฉากกับเส้น ชี้ไปฝั่ง − (ขวามือเมื่อมองจาก A ไป B บนภาพ)
    const len = Math.hypot(bx - ax, by - ay);
    if (len > 1) {
      const nx = (by - ay) / len, ny = -(bx - ax) / len; // ตั้งฉาก ชี้ฝั่ง −
      const mx = (ax + bx) / 2, my = (ay + by) / 2, L = 16 * scale;
      const tx = mx - nx * L * 0.5, ty = my - ny * L * 0.5, hx = mx + nx * L, hy = my + ny * L;
      ctx.lineWidth = 2 * scale;
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(hx, hy);
      ctx.moveTo(hx, hy);
      ctx.lineTo(hx - nx * 6 * scale + ny * 5 * scale, hy - ny * 6 * scale - nx * 5 * scale);
      ctx.moveTo(hx, hy);
      ctx.lineTo(hx - nx * 6 * scale - ny * 5 * scale, hy - ny * 6 * scale + nx * 5 * scale);
      ctx.stroke();
      ctx.font = `bold ${11 * scale}px sans-serif`;
      ctx.fillText("IN", hx + nx * 4 * scale, hy + ny * 4 * scale + 4 * scale);
    }
  }
  ctx.restore();
}

// counters = [{ name, params, state, tracks, flash }] · วาดเส้น/โซน + เลข id ของชิ้นที่ติดตาม + ป้ายสรุปมุมซ้ายบน
export function drawCounters(ctx, counters, W, H, scale = 1) {
  ctx.save();
  for (const c of counters) drawCounterShape(ctx, c.params, W, H, scale, c.flash);
  const first = counters[0];
  if (first && first.tracks) {
    ctx.font = `bold ${11 * scale}px sans-serif`;
    for (const t of first.tracks) {
      if (t.missed !== 0) continue;
      ctx.lineWidth = 3 * scale;
      ctx.strokeStyle = "rgba(0,0,0,0.75)";
      ctx.strokeText(`#${t.id}`, t.x + 6 * scale, t.y + 14 * scale);
      ctx.fillStyle = "#ffd60a";
      ctx.fillText(`#${t.id}`, t.x + 6 * scale, t.y + 14 * scale);
    }
  }
  ctx.font = `bold ${13 * scale}px sans-serif`;
  ctx.textBaseline = "alphabetic";
  let y = 6 * scale;
  const rowH = 20 * scale;
  for (const c of counters) {
    const text = counterText(c.name, c.params, c.state);
    const tw = ctx.measureText(text).width + 12 * scale;
    ctx.fillStyle = POST_COLORS.panel;
    ctx.fillRect(6 * scale, y, tw, rowH);
    ctx.fillStyle = normCount(c.params).mode === "zone" ? POST_COLORS.zone : POST_COLORS.line;
    ctx.fillText(text, 12 * scale, y + rowH - 6 * scale);
    y += rowH + 4 * scale;
  }
  ctx.restore();
}

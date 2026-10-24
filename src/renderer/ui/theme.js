// ธีมของแอป (สว่าง/มืด) — แหล่งเดียวของสถานะธีม + สีที่ canvas ของกราฟต้องใช้
// - สถานะอยู่ที่ <html data-theme="light|dark"> (CSS ใช้ตัวนี้) · เก็บการเลือกใน localStorage "mv-theme"
// - ค่าเริ่มต้น = light (ไม่มีค่าที่เก็บไว้) · สคริปต์เล็ก ๆ ใน index.html <head> ตั้ง attribute ก่อนวาดเฟรมแรกกันจอมืดวาบ
// - canvas (LiteGraph) อ่านสีจาก CSS variable ไม่ได้ → graphColors() เก็บพาเลตต์ของกราฟทั้งสองธีมไว้ที่นี่
//   (ค่า dark = ค่าที่เคย hardcode ใน editor.js เป๊ะ)

const KEY = "mv-theme";
const THEMES = ["light", "dark"];
export const DEFAULT_THEME = "light";

const GRAPH_COLORS = {
  dark: {
    canvasBg: "#08111d",          // = --surface-canvas
    body: "#0f1b2b",              // พื้นการ์ด node
    titleBase: "#132235",         // พื้นหัวการ์ด (ผสมกับสีหมวด 18%)
    titleMixAmt: 0.18,
    titleText: "#f5f7fa",
    titleTextSel: "#ffffff",
    titleTextDim: "#a7b2c3",      // การ์ดที่ไม่อยู่ใน pipeline
    focus: "#8cc8ff",
    focusGlow: "rgba(140,200,255,0.45)",
    cardBorder: "rgba(255,255,255,0.1)",
    cardBorderHover: "rgba(255,255,255,0.24)",
    err: "#ff5577",
    errText: "#ffb3c1",
    errBar: "rgba(5,11,20,0.88)",
    onErr: "#ffffff",             // "!" บนวงกลมแดง
    dot: "rgba(255,255,255,.04)", // จุดพื้นหลังกระดาน
    flow: "#cfe9ff",              // จุดวิ่งบนสาย
    statusOff: "#64748b",
    statusOn: "#27c98a",
    statusIdle: "#8592a4",
    hatch: "rgba(255,255,255,0.04)",
    pillBg: "rgba(5,11,20,0.8)",
    pillText: "#a7b2c3",
    textMain: "#f5f7fa",          // ชื่อบรรทัดแรกในการ์ด / ไอคอน
    textSub: "#a7b2c3",           // รายละเอียด / ค่าหลัก
    well: "#050b14",              // ช่องแสดงภาพในกล่องผลลัพธ์
    wellText: "#8592a4",
    wellBorder: "rgba(255,255,255,0.08)",
    tintTo: "#ffffff",            // ตัวอักษรชื่อหมวด = สีหมวดผสมไปทางขาว
    tintAmt: 0.75,
    defaultBox: "rgba(255,255,255,0.35)",
    portText: "#8592a4",
    shadow: "rgba(1,6,14,0.45)",
    link: "#4a6285",
    portIdle: "#6f86a6",
    portOn: "#58b4ff",
  },
  light: {
    // Cloud White (2026-10-24, Vision_boi/design.md) — ตรงกับ token ใน app.css :root[data-theme="light"]
    canvasBg: "#F3F4F1",          // = --surface-canvas
    body: "#FAFAF7",              // = --node-body
    titleBase: "#F0F1EE",         // = --node-title
    titleMixAmt: 0.10,
    titleText: "#283746",
    titleTextSel: "#283746",
    titleTextDim: "#566572",
    focus: "#3569AA",             // โหนดที่เลือก = ขอบน้ำเงิน
    focusGlow: "rgba(53,105,170,0.26)",
    cardBorder: "#CBD1D4",
    cardBorderHover: "#9AA4AB",
    err: "#B91C1C",
    errText: "#B91C1C",
    errBar: "rgba(252,240,238,0.96)",
    onErr: "#ffffff",
    dot: "rgba(86,101,114,.16)",
    flow: "#2B578E",
    statusOff: "#B9C0C4",
    statusOn: "#166534",
    statusIdle: "#8A949C",
    hatch: "rgba(40,55,70,0.05)",
    pillBg: "#ECEEE9",
    pillText: "#566572",
    textMain: "#283746",
    textSub: "#566572",
    well: "#E1E3DF",              // = --viewer-bg (พื้นช่อง preview ในการ์ด Display)
    wellText: "#566572",
    wellBorder: "#CBD1D4",
    tintTo: "#283746",            // โหมดสว่าง: ผสมสีหมวดไปทางตัวอักษรหลักให้เข้มพอบนพื้นขาวขุ่น
    tintAmt: 0.5,
    defaultBox: "rgba(40,55,70,0.35)",
    portText: "#566572",
    shadow: "rgba(40,55,70,0.09)",
    link: "#7C9CC6",
    portIdle: "#7E8A93",
    portOn: "#3569AA",
  },

};

// สีของ Histogram ในแผงพรีวิว (canvas อ่าน CSS variable ไม่ได้เหมือนกราฟ) · dark = ค่าเดิมใน app.js ก่อนแยกธีม
const HIST_COLORS = {
  dark: {
    bg: "#050b14", grid: "rgba(126, 160, 198, .13)", axis: "#71849d",
    luma: "#38d9ff", r: "#ff5577", g: "#45e59a", b: "#4aa3ff", marker: "#ff9f43",
  },
  light: {
    // Cloud White: พื้นขาวขุ่นเดียวกับช่องกรอก · เส้น/ตัวอักษรเข้มพอบนพื้นสว่าง (แกน #566572 = 5.7:1)
    bg: "#FAFAF7", grid: "rgba(40, 55, 70, .10)", axis: "#566572",
    luma: "#3569AA", r: "#C0392B", g: "#1E7A43", b: "#2F5FD0", marker: "#B45309",
  },
};

function readStored() {
  try {
    const v = localStorage.getItem(KEY);
    return THEMES.includes(v) ? v : null;
  } catch {
    return null;
  }
}

let current = readStored() || DEFAULT_THEME;
const listeners = new Set();

function applyAttr(t) {
  const el = typeof document !== "undefined" ? document.documentElement : null;
  if (el) el.setAttribute("data-theme", t);
}
// แจ้ง Electron main ให้ตั้งสีพื้นหน้าต่าง (มีเฉพาะใน Electron — preload เปิด window.mvTheme)
function syncNative(t) {
  try { if (typeof window !== "undefined" && window.mvTheme) window.mvTheme.set(t); } catch { /* ไม่มีสะพาน — ข้าม */ }
}
applyAttr(current);
syncNative(current);

export function getTheme() {
  return current;
}

// พาเลตต์สีของกราฟตามธีมปัจจุบัน — คืน object เดิมทุกครั้ง (ไม่ alloc ต่อเฟรม)
export function graphColors() {
  return GRAPH_COLORS[current];
}

// พาเลตต์สีของ Histogram ในแผงพรีวิวตามธีมปัจจุบัน
export function histColors() {
  return HIST_COLORS[current];
}

export function setTheme(t, { persist = true } = {}) {
  if (!THEMES.includes(t)) return;
  const changed = t !== current;
  current = t;
  applyAttr(t);
  syncNative(t);
  if (persist) {
    try { localStorage.setItem(KEY, t); } catch { /* โหมด private/ไม่มี storage — ใช้ได้แค่ session นี้ */ }
  }
  if (changed) for (const fn of [...listeners]) { try { fn(t); } catch (e) { console.error(e); } }
}

export function toggleTheme() {
  setTheme(current === "light" ? "dark" : "light");
  return current;
}

// สมัครรับแจ้งเมื่อธีมเปลี่ยน — คืนฟังก์ชันยกเลิก
export function onThemeChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

// ผูกปุ่มสลับธีม (sun/moon) — อัปเดต aria-label/tooltip ให้บอก "การกระทำ" ที่จะเกิด (ไม่ใช้ aria-pressed — ป้ายกับสถานะไม่ขัดกัน)
export function bindThemeToggle(btn) {
  if (!btn) return;
  const sync = () => {
    const dark = current === "dark";
    btn.dataset.theme = current;
    const label = dark ? "สลับเป็นธีมสว่าง" : "สลับเป็นธีมมืด";
    btn.title = `${label} (ตอนนี้: ${dark ? "มืด" : "สว่าง"})`;
    btn.setAttribute("aria-label", label);
  };
  btn.addEventListener("click", () => toggleTheme());
  onThemeChange(sync);
  sync();
}

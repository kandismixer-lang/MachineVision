// Panel ขวา: แบ่ง dataset เป็น train/valid/test (Phase 4 ของระบบ annotation)
// คุยกับ dataset/annotations.js (single source of truth) ผ่าน assignSplitsByRatio
// UX: แถบสัดส่วน (bar) ลากปรับได้ + พิมพ์เลขได้ · ปุ่ม "ยืนยันการแบ่ง" โผล่เมื่อค่าถูกปรับ
//     กดยืนยัน = สุ่มแบ่งจริง แล้วปุ่มหาย (dirty=false) · ปรับใหม่เมื่อไรปุ่มโผล่กลับ (dirty=true)
import { getSplit, assignSplitsByRatio, onChange } from "../dataset/annotations.js";

const SPLITS = ["train", "valid", "test"];
const SPLIT_LABEL = { train: "Train", valid: "Valid", test: "Test" };

export function createSplitPanel(container, opts = {}) {
  const { getAnnotatedKeys, onApply } = opts;

  // state คงข้ามการ re-render (assignSplitsByRatio → onChange → render ใหม่)
  const ratios = { train: 70, valid: 20, test: 10 }; // สัดส่วน (น้ำหนัก) — backend normalize ด้วยผลรวมอยู่แล้ว
  let dirty = true; // true = ค่าถูกปรับแต่ยังไม่กดยืนยัน → โชว์ปุ่มยืนยัน

  const sum = () => Math.max(1, ratios.train + ratios.valid + ratios.test);
  const pct = (s) => (ratios[s] / sum()) * 100;

  function render() {
    container.innerHTML = "";
    const keys = getAnnotatedKeys ? getAnnotatedKeys() : [];

    // ---------- สรุปจำนวนต่อ split ----------
    const counts = { train: 0, valid: 0, test: 0 };
    for (const k of keys) counts[getSplit(k)]++;

    const summary = document.createElement("div");
    summary.className = "split-panel-summary";
    summary.textContent = keys.length
      ? `${keys.length} annotated images · Current: Train ${counts.train} · Valid ${counts.valid} · Test ${counts.test}`
      : "No annotated images yet — label images before applying a split";
    container.appendChild(summary);

    // ---------- แถบสัดส่วน (bar) — ลากปรับได้ ----------
    const bar = document.createElement("div");
    bar.className = "split-bar";
    const segs = {};
    const segLabels = {};
    const handles = {}; // handle ที่ขอบ train|valid และ valid|test
    for (const s of SPLITS) {
      const seg = document.createElement("div");
      seg.className = "split-seg split-seg-" + s;
      seg.style.width = pct(s) + "%";
      const lab = document.createElement("span");
      lab.className = "split-seg-label";
      seg.appendChild(lab);
      segLabels[s] = lab;
      segs[s] = seg;
      bar.appendChild(seg);
    }
    // handle 2 อัน (ขอบระหว่าง train|valid และ valid|test)
    for (const h of ["a", "b"]) {
      const hd = document.createElement("div");
      hd.className = "split-handle";
      handles[h] = hd;
      bar.appendChild(hd);
    }
    container.appendChild(bar);

    // ---------- ช่องเลข (พิมพ์ได้) ----------
    const ratioRow = document.createElement("div");
    ratioRow.className = "split-panel-ratio-row";
    const inputs = {};
    for (const s of SPLITS) {
      const field = document.createElement("label");
      field.className = "split-panel-ratio-field";
      const dot = document.createElement("span");
      dot.className = "split-dot split-dot-" + s;
      const span = document.createElement("span");
      span.textContent = SPLIT_LABEL[s];
      const input = document.createElement("input");
      input.type = "number";
      input.min = "0";
      input.step = "1";
      input.value = String(Math.round(ratios[s]));
      input.className = "split-panel-ratio-input";
      const count = document.createElement("span");
      count.className = "split-ratio-count mono";
      inputs[s] = input;
      field.append(dot, span, input, count);
      field._countEl = count;
      ratioRow.appendChild(field);
    }
    container.appendChild(ratioRow);

    // ---------- คำอธิบาย: % นี้ = กี่รูปของทั้งหมด ----------
    const desc = document.createElement("div");
    desc.className = "split-desc";
    container.appendChild(desc);

    // จำนวนภาพต่อ split ที่ "จะได้" ตามสัดส่วนปัจจุบัน (mirror logic ของ assignSplitsByRatio)
    function previewCounts() {
      const total = keys.length, s = sum();
      const nTrain = Math.round((ratios.train / s) * total);
      const nValid = Math.round((ratios.valid / s) * total);
      const nValidC = Math.min(nValid, Math.max(0, total - nTrain));
      const nTest = total - nTrain - nValidC;
      return { train: nTrain, valid: nValidC, test: nTest };
    }

    // ---------- ปุ่มยืนยัน (โผล่เฉพาะตอน dirty) ----------
    const confirmBtn = document.createElement("button");
    confirmBtn.type = "button";
    confirmBtn.className = "btn btn-primary split-confirm";
    confirmBtn.textContent = "✓ Apply Split";
    confirmBtn.disabled = keys.length === 0;
    confirmBtn.style.display = dirty ? "" : "none";
    container.appendChild(confirmBtn);

    const doneHint = document.createElement("div");
    doneHint.className = "split-done-hint";
    doneHint.textContent = "✓ Split applied — adjust a value to split again";
    doneHint.style.display = dirty ? "none" : "";
    container.appendChild(doneHint);

    // อัปเดต bar + label% + ช่องเลข + คำอธิบาย จาก ratios (ไม่ re-render — กันเสีย focus ตอนพิมพ์/ลาก)
    function paint() {
      const cnt = previewCounts();
      const totalRatio = ratios.train + ratios.valid + ratios.test;
      const totalValid = Math.abs(totalRatio - 100) < 0.001;
      for (const s of SPLITS) {
        const p = Math.round(pct(s));
        segs[s].style.width = pct(s) + "%";
        // โชว์ label เฉพาะแท่งที่กว้างพอ (กันตัวหนังสือล้น) — แคบ ๆ โชว์แค่ %
        segLabels[s].textContent = p >= 16 ? `${SPLIT_LABEL[s]} ${p}%` : p >= 7 ? `${p}%` : "";
        if (document.activeElement !== inputs[s]) inputs[s].value = String(Math.round(ratios[s]));
        inputs[s].closest(".split-panel-ratio-field")._countEl.textContent = `${cnt[s]} images`;
      }
      handles.a.style.left = pct("train") + "%";
      handles.b.style.left = pct("train") + pct("valid") + "%";
      desc.textContent = `${keys.length} images · ${getClassCountText()} · Total ${totalRatio}%`;
      desc.classList.toggle("invalid", !totalValid);
      confirmBtn.disabled = keys.length === 0 || !totalValid;
    }

    function getClassCountText() {
      return opts.getClassCount ? `${opts.getClassCount()} classes` : "Dataset ready";
    }
    paint();

    // ค่าเปลี่ยน → dirty + โชว์ปุ่มยืนยันกลับมา
    function markDirty() {
      if (!dirty) { dirty = true; confirmBtn.style.display = keys.length ? "" : "none"; doneHint.style.display = "none"; }
    }

    // พิมพ์เลข
    for (const s of SPLITS) {
      inputs[s].addEventListener("input", () => {
        ratios[s] = Math.max(0, Number(inputs[s].value) || 0);
        markDirty();
        paint(); // อัปเดต bar/label/คำอธิบาย (paint ไม่แตะ input ที่กำลัง focus อยู่)
      });
    }

    // ลาก handle (a = ขอบ train|valid, b = ขอบ valid|test) — ตั้งเป็นเปอร์เซ็นต์รวม 100
    function dragHandle(which, e) {
      e.preventDefault();
      const rect = bar.getBoundingClientRect();
      if (!rect.width) return; // กัน bar ยุบ 0 (จอแคบ/ยังไม่ layout) → หารศูนย์เป็น NaN
      const move = (ev) => {
        let p = ((ev.clientX - rect.left) / rect.width) * 100;
        p = Math.min(100, Math.max(0, p));
        let a = pct("train"), b = pct("train") + pct("valid");
        if (which === "a") a = Math.min(p, b); else b = Math.max(p, a);
        // ปัด "ตำแหน่งขอบ" ก่อน แล้วค่อยหาผลต่าง เพื่อให้ผลรวมเป็น 100 เสมอ
        const edgeA = Math.min(100, Math.max(0, Math.round(a)));
        const edgeB = Math.min(100, Math.max(edgeA, Math.round(b)));
        ratios.train = edgeA;
        ratios.valid = edgeB - edgeA;
        ratios.test = 100 - edgeB;
        markDirty();
        paint();
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    }
    handles.a.addEventListener("pointerdown", (e) => dragHandle("a", e));
    handles.b.addEventListener("pointerdown", (e) => dragHandle("b", e));

    // ยืนยัน → สุ่มแบ่งจริง แล้วปุ่มหาย (assignSplitsByRatio จะ trigger onChange → render, dirty=false)
    confirmBtn.addEventListener("click", () => {
      if (ratios.train + ratios.valid + ratios.test <= 0) return;
      if (keys.length === 0) return;
      dirty = false;
      assignSplitsByRatio(keys, { ...ratios });
      onApply?.({ ...ratios });
    });
  }

  render();
  // subscribe ครั้งเดียวตอนสร้าง panel (เหมือน class-panel.js) — กัน listener ค้างเวลา re-render บ่อย
  const unsubscribe = onChange(render);

  return {
    render,
    destroy: () => unsubscribe(),
  };
}

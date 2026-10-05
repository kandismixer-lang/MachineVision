// ขั้นตอน Augmentation (แบบ Roboflow) — บังคับขึ้นตอนกด Export Dataset
// 3 มุมมองในหน้าต่างเดียว:
//   list   : รายการ aug ที่เพิ่มไว้ (แถว + Edit + ลบ) · ปุ่ม Add Augmentation Step · Continue
//   picker : เลือก aug ที่จะเพิ่ม (การ์ด thumbnail)
//   edit   : ตั้งค่า aug ตัวเดียว — preview before/after ใหญ่ + slider
//
// openForExport() คืน Promise<boolean> : true = ไปต่อ (export), false = ยกเลิก
// ค่าถูกบันทึกที่ app.js (setDatasetAugment) ทันทีเมื่อกด Apply/ลบ
import { getDatasetAugment, setDatasetAugment } from "../app.js";
import {
  normalizeAugmentConfig,
  defaultAugmentConfig,
  buildPlan,
  applyPlan,
} from "../dataset/augment.js";

const AUG_DEFS = [
  { key: "flipH", label: "Flip", desc: () => "แนวนอน (ซ้าย-ขวา)" },
  { key: "flipV", label: "Flip V", desc: () => "แนวตั้ง (บน-ล่าง)" },
  { key: "rotation", label: "Rotation", param: { field: "deg", min: 1, max: 45, step: 1, unit: "°" },
    desc: (c) => `สูงสุด ±${c.deg}°`, blurb: "หมุนภาพสุ่ม ช่วยให้โมเดลทนต่อมุมกล้องที่ต่างกัน" },
  { key: "crop", label: "Crop", param: { field: "pct", min: 1, max: 50, step: 1, unit: "%" },
    desc: (c) => `ซูมสูงสุด ${c.pct}%`, blurb: "สุ่มซูม/ครอบตัด ช่วยให้โมเดลเห็นวัตถุขนาดต่างกัน" },
  { key: "shear", label: "Shear", param: { field: "deg", min: 1, max: 30, step: 1, unit: "°" },
    desc: (c) => `สูงสุด ±${c.deg}°`, blurb: "เฉือนภาพ จำลองการมองวัตถุจากมุมเอียง" },
  { key: "grayscale", label: "Grayscale", desc: () => "แปลงเป็นขาว-ดำ",
    blurb: "ทำภาพขาว-ดำ ช่วยให้โมเดลไม่ยึดติดกับสี" },
  { key: "hue", label: "Hue", param: { field: "deg", min: 1, max: 180, step: 1, unit: "°" },
    desc: (c) => `±${c.deg}°`, blurb: "เปลี่ยนโทนสี จำลองแสง/กล้องที่ต่างกัน" },
  { key: "saturation", label: "Saturation", param: { field: "pct", min: 1, max: 100, step: 1, unit: "%" },
    desc: (c) => `±${c.pct}%`, blurb: "ปรับความอิ่มสี จำลองสภาพแสงต่างกัน" },
  { key: "brightness", label: "Brightness", param: { field: "pct", min: 1, max: 60, step: 1, unit: "%" },
    desc: (c) => `±${c.pct}%`, blurb: "ปรับความสว่าง จำลองสภาพแสงต่างกัน" },
  { key: "exposure", label: "Exposure", param: { field: "pct", min: 1, max: 60, step: 1, unit: "%" },
    desc: (c) => `±${c.pct}%`, blurb: "ปรับโทนสว่าง-มืด (contrast) จำลองการรับแสงต่างกัน" },
  { key: "blur", label: "Blur", param: { field: "px", min: 0.5, max: 10, step: 0.5, unit: "px" },
    desc: (c) => `สูงสุด ${c.px}px`, blurb: "เบลอแบบ Gaussian สุ่ม ช่วยให้โมเดลทนต่อภาพหลุดโฟกัส" },
  { key: "noise", label: "Noise", param: { field: "pct", min: 0.5, max: 15, step: 0.5, unit: "%" },
    desc: (c) => `สูงสุด ${c.pct}%`, blurb: "ใส่สัญญาณรบกวน จำลองภาพจากกล้องคุณภาพต่ำ" },
];
const DEF_BY_KEY = Object.fromEntries(AUG_DEFS.map((d) => [d.key, d]));
const AUG_ICON = {
  flipH: "↔", flipV: "↕", rotation: "↻", crop: "⌗", shear: "◇",
  grayscale: "◐", hue: "◒", saturation: "◉", brightness: "☀",
  exposure: "◑", blur: "●", noise: "⠿",
};

// canvas ตัวอย่างย่อขนาด (สี่เหลี่ยมจัตุรัส) จาก sample จริง หรือ placeholder
function makeWork(sample, size) {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");
  if (sample && sample.width) {
    const s = Math.min(sample.width, sample.height);
    ctx.drawImage(sample, (sample.width - s) / 2, (sample.height - s) / 2, s, s, 0, 0, size, size);
  } else {
    const g = ctx.createLinearGradient(0, 0, size, size);
    g.addColorStop(0, "#3a4a63");
    g.addColorStop(1, "#20304a");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("no image", size / 2, size / 2);
  }
  return c;
}

// config เฉพาะ aug ตัวเดียว (สำหรับ preview)
function solo(def, local) {
  const cfg = defaultAugmentConfig();
  for (const k of Object.keys(cfg)) if (cfg[k] && typeof cfg[k] === "object") cfg[k].on = false;
  cfg[def.key] = { ...local[def.key], on: true };
  return cfg;
}

export function createAugmentDialog({ getSampleCanvas, getCounts } = {}) {
  async function openForExport() {
    const local = normalizeAugmentConfig(getDatasetAugment());
    const sample = getSampleCanvas ? await getSampleCanvas() : null;

    return new Promise((resolve) => {
      const overlay = document.createElement("div");
      overlay.className = "modal-overlay";
      const modal = document.createElement("div");
      modal.className = "modal augment-modal";
      overlay.appendChild(modal);

      let done = false;
      function finish(ok) {
        if (done) return;
        done = true;
        if (ok) setDatasetAugment(local); // บันทึกเฉพาะเมื่อ Export จริง — ยกเลิก/Escape ไม่แตะ config เดิม
        overlay.remove();
        document.removeEventListener("keydown", onKey);
        resolve(ok);
      }
      function onKey(e) { if (e.key === "Escape") finish(false); }
      document.addEventListener("keydown", onKey);
      overlay.addEventListener("click", (e) => { if (e.target === overlay) finish(false); });

      // ---------- helpers ----------
      function header(title, onClose) {
        const head = document.createElement("div");
        head.className = "augment-head";
        const h = document.createElement("div");
        h.className = "modal-title";
        h.textContent = title;
        const x = document.createElement("button");
        x.className = "modal-x";
        x.type = "button";
        x.textContent = "✕";
        x.addEventListener("click", onClose);
        head.append(h, x);
        return head;
      }

      // ---------- VIEW: list ----------
      function renderList(selectedKey = null, draft = null) {
        modal.innerHTML = "";
        modal.className = "modal augment-modal augment-modal--workspace";
        modal.appendChild(header("Augmentation", () => finish(false)));

        const sub = document.createElement("div");
        sub.className = "modal-sub";
        sub.textContent = "สร้างภาพเทรนเพิ่มจากภาพเดิม (เฉพาะชุด train) — เพิ่มขั้นตอนหรือกด Continue เพื่อ export";
        modal.appendChild(sub);

        const workspace = document.createElement("div");
        workspace.className = "aug-workspace";
        const stepPane = document.createElement("section");
        stepPane.className = "aug-step-pane";
        const stepTitle = document.createElement("div");
        stepTitle.className = "aug-pane-title";
        stepTitle.textContent = "ขั้นตอนการ Augmentation";
        const list = document.createElement("div");
        list.className = "aug-steps";
        const active = AUG_DEFS.filter((d) => local[d.key].on);
        if (!selectedKey) {
          const firstEditable = active.find((d) => d.param);
          if (firstEditable) {
            selectedKey = firstEditable.key;
            draft = { ...local[firstEditable.key] };
          }
        }
        if (active.length === 0) {
          const empty = document.createElement("div");
          empty.className = "aug-empty";
          empty.textContent = "ยังไม่มีขั้นตอน — กด “+ Add Augmentation Step” (ไม่บังคับ)";
          list.appendChild(empty);
        }
        for (const [index, def] of active.entries()) {
          const row = document.createElement("div");
          row.className = "aug-step-row" + (selectedKey === def.key ? " selected" : "");
          row.dataset.aug = def.key;
          if (def.param) {
            row.classList.add("clickable");
            row.tabIndex = 0;
            row.setAttribute("role", "button");
            row.setAttribute("aria-label", `เปิด Preview และตั้งค่า ${def.label}`);
            const openEditor = () => renderList(def.key, { ...local[def.key] });
            row.addEventListener("click", (e) => {
              if (!e.target.closest("button")) openEditor();
            });
            row.addEventListener("keydown", (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                openEditor();
              }
            });
          }
          const order = document.createElement("span");
          order.className = "aug-step-order";
          order.textContent = String(index + 1);
          const icon = document.createElement("span");
          icon.className = "aug-step-icon";
          icon.textContent = AUG_ICON[def.key] || "✦";
          const info = document.createElement("div");
          info.className = "aug-step-info";
          info.innerHTML = `<strong>${def.label}</strong><span>${def.desc(local[def.key])}</span>`;
          const editBtn = document.createElement("button");
          editBtn.type = "button";
          editBtn.className = "aug-step-edit";
          editBtn.textContent = def.param ? "Edit" : "คงที่";
          editBtn.disabled = !def.param;
          editBtn.addEventListener("click", () => renderList(def.key, { ...local[def.key] }));
          const del = document.createElement("button");
          del.type = "button";
          del.className = "aug-step-del";
          del.textContent = "✕";
          del.title = "ลบขั้นตอนนี้";
          del.addEventListener("click", () => { local[def.key].on = false; renderList(); });
          row.append(order, icon, info, editBtn, del);
          list.appendChild(row);
        }
        const addBtn = document.createElement("button");
        addBtn.type = "button";
        addBtn.className = "aug-add-step";
        addBtn.innerHTML = "<span>＋</span> Add Augmentation Step";
        addBtn.addEventListener("click", renderPicker);
        list.appendChild(addBtn);
        stepPane.append(stepTitle, list);

        const inspector = document.createElement("section");
        inspector.className = "aug-inspector";
        const selectedDef = selectedKey ? DEF_BY_KEY[selectedKey] : null;
        if (selectedDef && selectedDef.param && draft) {
          renderInlineEditor(inspector, selectedDef, draft);
        } else {
          const empty = document.createElement("div");
          empty.className = "aug-inspector-empty";
          empty.innerHTML = `<span class="aug-inspector-mark">✦</span><strong>เลือกขั้นตอนเพื่อดู Preview</strong><span>กด Edit ที่ขั้นตอนซึ่งมีพารามิเตอร์ เพื่อเปรียบเทียบภาพก่อนและหลัง</span>`;
          inspector.appendChild(empty);
        }

        workspace.append(stepPane, inspector);
        modal.appendChild(workspace);

        modal.appendChild(footer());
      }

      function renderInlineEditor(host, def, draft) {
        const p = def.param;
        const head = document.createElement("div");
        head.className = "aug-inspector-head";
        head.innerHTML = `<div><span class="aug-inspector-icon">${AUG_ICON[def.key] || "✦"}</span><strong>${def.label}</strong></div><span>Preview & Settings</span>`;

        const previewTabs = document.createElement("div");
        previewTabs.className = "aug-preview-tabs";
        previewTabs.innerHTML = `<span>Before</span><span>After</span><span class="active">Compare</span>`;

        const compare = document.createElement("div");
        compare.className = "aug-compare";
        const work = makeWork(sample, 300);
        const beforeWrap = mkPreview("ต้นฉบับ", 300);
        beforeWrap.canvas.getContext("2d").drawImage(work, 0, 0);
        const afterWrap = mkPreview("", 300);
        compare.append(beforeWrap.box, afterWrap.box);

        const controls = document.createElement("div");
        controls.className = "aug-inline-controls";
        const blurb = document.createElement("p");
        blurb.textContent = def.blurb || "";
        const label = document.createElement("label");
        label.textContent = `ค่าสูงสุด (${p.unit === "°" ? "องศา" : p.unit})`;
        const sliderRow = document.createElement("div");
        sliderRow.className = "aug-slider-row aug-slider-row--inline";
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = String(p.min);
        slider.max = String(p.max);
        slider.step = String(p.step);
        slider.value = String(draft[p.field]);
        slider.className = "aug-slider aug-slider-big";
        const val = document.createElement("output");
        val.className = "aug-val aug-val-box";
        sliderRow.append(slider, val);

        const editorActions = document.createElement("div");
        editorActions.className = "aug-inline-actions";
        const cancel = document.createElement("button");
        cancel.className = "btn";
        cancel.textContent = "ยกเลิก";
        cancel.addEventListener("click", () => renderList());
        const apply = document.createElement("button");
        apply.className = "btn btn-primary";
        apply.textContent = "Apply";
        apply.addEventListener("click", () => {
          local[def.key] = { ...draft, on: true };
          renderList(def.key, { ...local[def.key] });
        });
        editorActions.append(cancel, apply);

        const redraw = () => {
          const v = Number(slider.value);
          draft[p.field] = v;
          val.textContent = `${v}${p.unit}`;
          afterWrap.label.textContent = `${def.label} ${v}${p.unit}`;
          const cfg = normalizeAugmentConfig(local);
          cfg[def.key] = { ...draft, on: true };
          const plan = buildPlan(solo(def, cfg), 300, 300, true);
          const out = applyPlan(work, plan, 300, 300);
          const ctx = afterWrap.canvas.getContext("2d");
          ctx.clearRect(0, 0, 300, 300);
          ctx.drawImage(out, 0, 0);
        };
        slider.addEventListener("input", redraw);
        redraw();

        controls.append(blurb, label, sliderRow, editorActions);
        host.append(head, previewTabs, compare, controls);
      }

      function footer() {
        const foot = document.createElement("div");
        foot.className = "augment-foot";

        const actions = document.createElement("div");
        actions.className = "modal-actions";
        const cancel = document.createElement("button");
        cancel.className = "btn";
        cancel.textContent = "ยกเลิก";
        cancel.addEventListener("click", () => finish(false));
        const next = document.createElement("button");
        next.className = "btn btn-primary";
        next.textContent = "Next →";
        next.addEventListener("click", renderCreate);
        actions.append(cancel, next);

        foot.append(actions);
        return foot;
      }

      // ---------- VIEW: create (เลือก Augmentation Multiplier ก่อน export) ----------
      async function renderCreate() {
        const counts = getCounts ? await getCounts() : { train: 0, valid: 0, test: 0 };
        const base = counts.train || 0;
        const others = (counts.valid || 0) + (counts.test || 0);
        modal.innerHTML = "";
        modal.className = "modal augment-modal";
        modal.appendChild(header("Create", renderList));

        const sub = document.createElement("div");
        sub.className = "modal-sub";
        sub.textContent = "เลือกขนาดเวอร์ชัน — ยิ่งภาพเยอะยิ่งเทรนนาน แต่มักได้โมเดลแม่นขึ้น";
        modal.appendChild(sub);

        const field = document.createElement("div");
        field.className = "aug-create-field";
        const flabel = document.createElement("label");
        flabel.className = "aug-create-label";
        flabel.textContent = "Augmentation Multiplier";
        const select = document.createElement("select");
        select.className = "aug-create-select";
        const hasAug = AUG_DEFS.some((d) => local[d.key].on);
        // 1x = ไม่ augment · 2x-6x = augment (copies = mult - 1) แต่ต้องมีขั้นตอนก่อน
        const maxMult = hasAug ? 6 : 1;
        for (let m = 1; m <= maxMult; m++) {
          const opt = document.createElement("option");
          const total = base * m + others;
          opt.value = String(m);
          opt.textContent = base ? `${total} ภาพ (${m}x)` : `${m}x`;
          if (m === Math.min(maxMult, (local.copies || 0) + 1)) opt.selected = true;
          select.appendChild(opt);
        }
        field.append(flabel, select);
        modal.appendChild(field);

        if (!hasAug) {
          const note = document.createElement("div");
          note.className = "aug-empty";
          note.textContent = "ยังไม่ได้เพิ่มขั้นตอน Augmentation — จะ export ภาพเดิม (1x). กด ← กลับ เพื่อเพิ่ม";
          modal.appendChild(note);
        }

        const foot = document.createElement("div");
        foot.className = "augment-foot";
        const actions = document.createElement("div");
        actions.className = "modal-actions";
        const back = document.createElement("button");
        back.className = "btn";
        back.textContent = "← กลับ";
        back.addEventListener("click", () => renderList());
        const exportBtn = document.createElement("button");
        exportBtn.className = "btn btn-primary";
        exportBtn.textContent = "⇩ Export";
        exportBtn.addEventListener("click", () => {
          local.copies = Math.max(0, Number(select.value) - 1);
          finish(true);
        });
        actions.append(back, exportBtn);
        foot.appendChild(actions);
        modal.appendChild(foot);
      }

      // ---------- VIEW: picker ----------
      function renderPicker() {
        modal.innerHTML = "";
        modal.className = "modal augment-modal";
        modal.appendChild(header("เลือก Augmentation", renderList));
        const sub = document.createElement("div");
        sub.className = "modal-sub";
        sub.textContent = "แตะการ์ดเพื่อเพิ่ม (ตัวที่ปรับค่าได้จะเปิดหน้าตั้งค่า)";
        modal.appendChild(sub);

        const grid = document.createElement("div");
        grid.className = "augment-grid";
        const avail = AUG_DEFS.filter((d) => !local[d.key].on);
        const work = makeWork(sample, 120);
        if (avail.length === 0) {
          const e = document.createElement("div");
          e.className = "aug-empty";
          e.textContent = "เพิ่มครบทุกตัวแล้ว";
          grid.appendChild(e);
        }
        for (const def of avail) {
          const card = document.createElement("div");
          card.className = "aug-card";
          const thumb = document.createElement("button");
          thumb.type = "button";
          thumb.className = "aug-thumb";
          const cv = document.createElement("canvas");
          cv.width = 120;
          cv.height = 120;
          const plan = buildPlan(solo(def, local), 120, 120, true);
          const out = applyPlan(work, plan, 120, 120);
          cv.getContext("2d").drawImage(out, 0, 0);
          thumb.appendChild(cv);
          const name = document.createElement("div");
          name.className = "aug-name";
          name.textContent = def.label;
          card.append(thumb, name);
          const add = () => {
            if (def.param) renderEdit(def, "picker");
            else { local[def.key].on = true; renderList(); }
          };
          thumb.addEventListener("click", add);
          name.addEventListener("click", add);
          grid.appendChild(card);
        }
        modal.appendChild(grid);

        const foot = document.createElement("div");
        foot.className = "augment-foot";
        const back = document.createElement("button");
        back.className = "btn";
        back.textContent = "← กลับ";
        back.addEventListener("click", renderList);
        foot.appendChild(back);
        modal.appendChild(foot);
      }

      // ---------- VIEW: edit (preview before/after + slider) ----------
      function renderEdit(def, backTo) {
        const wasOn = local[def.key].on;
        const prev = { ...local[def.key] };
        modal.innerHTML = "";
        modal.className = "modal augment-modal";
        modal.appendChild(header(def.label, () => finish(false)));

        const body = document.createElement("div");
        body.className = "aug-edit-body";

        // ซ้าย: before/after
        const preCol = document.createElement("div");
        preCol.className = "aug-edit-previews";
        const work = makeWork(sample, 240);
        const beforeWrap = mkPreview("ต้นฉบับ");
        beforeWrap.canvas.getContext("2d").drawImage(work, 0, 0);
        const afterWrap = mkPreview("");
        preCol.append(beforeWrap.box, afterWrap.box);

        // ขวา: คำอธิบาย + slider
        const ctrl = document.createElement("div");
        ctrl.className = "aug-edit-ctrl";
        const blurb = document.createElement("p");
        blurb.className = "aug-edit-blurb";
        blurb.textContent = def.blurb || "";
        ctrl.appendChild(blurb);

        const p = def.param;
        const sliderRow = document.createElement("div");
        sliderRow.className = "aug-slider-row";
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = String(p.min);
        slider.max = String(p.max);
        slider.step = String(p.step);
        slider.value = String(local[def.key][p.field]);
        slider.className = "aug-slider aug-slider-big";
        const val = document.createElement("span");
        val.className = "aug-val";
        sliderRow.append(slider, val);
        ctrl.appendChild(sliderRow);
        const maxHint = document.createElement("div");
        maxHint.className = "aug-max-hint";
        maxHint.textContent = `${p.max}${p.unit}`;
        ctrl.appendChild(maxHint);

        body.append(preCol, ctrl);
        modal.appendChild(body);

        const redraw = () => {
          const v = Number(slider.value);
          local[def.key][p.field] = v;
          val.textContent = `${v}${p.unit}`;
          afterWrap.label.textContent = `${v}${p.unit}`;
          const plan = buildPlan(solo(def, local), 240, 240, true);
          const out = applyPlan(work, plan, 240, 240);
          const c = afterWrap.canvas.getContext("2d");
          c.clearRect(0, 0, 240, 240);
          c.drawImage(out, 0, 0);
        };
        redraw();
        slider.addEventListener("input", redraw);

        // footer
        const foot = document.createElement("div");
        foot.className = "augment-foot";
        const actions = document.createElement("div");
        actions.className = "modal-actions";
        const cancel = document.createElement("button");
        cancel.className = "btn";
        cancel.textContent = "ยกเลิก";
        cancel.addEventListener("click", () => {
          local[def.key] = { ...prev, on: wasOn }; // คืนค่าเดิม
          backTo === "picker" ? renderPicker() : renderList();
        });
        const apply = document.createElement("button");
        apply.className = "btn btn-primary";
        apply.textContent = "Apply";
        apply.addEventListener("click", () => {
          local[def.key].on = true;
          renderList();
        });
        actions.append(cancel, apply);
        foot.appendChild(actions);
        modal.appendChild(foot);
      }

      function mkPreview(labelText, size = 240) {
        const box = document.createElement("div");
        box.className = "aug-preview-box";
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const label = document.createElement("div");
        label.className = "aug-preview-label";
        label.textContent = labelText;
        box.append(canvas, label);
        return { box, canvas, label };
      }

      renderList();
      document.body.appendChild(overlay);
    });
  }

  return { openForExport };
}

// Panel ขวา: รายละเอียดของ filter ที่เลือก + คำอธิบายการทำงาน + สไลเดอร์ปรับค่า
import { FILTERS } from "../filters/registry.js";

// cb: onParamChange(key, value)
export function createDetailPanel(container, cb) {
  // step ล่าสุด (ใช้ re-render ตัวเองเมื่อเปลี่ยน "วิธี" ใน merged node)
  let curStep = null;
  let curNotes = null;

  // render(step, notes) — step = { id, params, enabled } หรือ null ถ้าไม่ได้เลือก
  // notes = รายการ note ของ filter นี้ เช่น [{autoGray:true},{info:"พบวัตถุ 3 ชิ้น"}]
  function render(step, notes) {
    curStep = step;
    curNotes = notes;
    container.innerHTML = "";

    if (!step) {
      const empty = document.createElement("div");
      empty.className = "detail-empty";
      empty.innerHTML =
        `<div class="de-icon">🎛️</div>` +
        `<div class="de-title">ยังไม่ได้เลือกเครื่องมือ</div>` +
        `<div class="de-desc">เลือกกล่องบน Pipeline Canvas เพื่อดูและปรับพารามิเตอร์</div>`;
      container.appendChild(empty);
      return;
    }

    const f = FILTERS[step.id];
    if (!f) return;

    const title = document.createElement("h2");
    title.className = "detail-title";
    title.textContent = f.name;
    container.appendChild(title);

    const group = document.createElement("span");
    group.className = "detail-group";
    group.textContent = f.group;
    container.appendChild(group);

    // merged node: โชว์ชื่อ "วิธี" ที่เลือกอยู่ใต้หัวข้อ
    if (typeof f.subtitle === "function") {
      const sub = document.createElement("div");
      sub.className = "detail-subtitle";
      sub.textContent = "▶ " + f.subtitle(step.params);
      container.appendChild(sub);
    }

    for (const note of notes || []) {
      const n = document.createElement("div");
      n.className = "detail-note";
      if (note.autoGray) {
        n.textContent = "ℹ️ filter นี้ทำงานกับภาพขาวดำ — ระบบแปลงภาพให้อัตโนมัติก่อนประมวลผล";
      } else if (note.info) {
        n.textContent = "📊 " + note.info;
        n.classList.add("detail-note-info");
      }
      container.appendChild(n);
    }

    // กรอง param ที่มี showIf ให้โชว์เฉพาะเมื่อ "วิธี" ตรงกับที่เลือก
    const visible = f.params.filter(
      (p) => !p.showIf || step.params[p.showIf.key] === p.showIf.value
    );

    if (visible.length === 0) {
      const none = document.createElement("div");
      none.className = "param-hint";
      none.textContent = "วิธีนี้ไม่มีค่าให้ปรับ";
      container.appendChild(none);
      return;
    }

    // หัวข้อคั่น "พารามิเตอร์" (Design §19) — แยกส่วน "อ่าน" (learning) ออกจาก "ปรับ" (controls)
    const phead = document.createElement("div");
    phead.className = "params-head";
    phead.textContent = "พารามิเตอร์";
    container.appendChild(phead);

    for (const param of visible) {
      container.appendChild(buildParam(param, step.params[param.key]));
    }
  }

  // แสดงแผงตั้งชื่อกล่องผลลัพธ์ — info = { id, getName, setName }
  function renderDisplay(info) {
    curStep = null;
    curNotes = null;
    container.innerHTML = "";

    const title = document.createElement("h2");
    title.className = "detail-title";
    title.textContent = "🖥️ กล่องผลลัพธ์";
    container.appendChild(title);

    const desc = document.createElement("div");
    desc.className = "detail-desc";
    desc.textContent =
      "ตั้งชื่อกล่องนี้ให้สื่อว่าแสดงภาพอะไร — ชื่อจะไปขึ้นที่หัวจอผลลัพธ์ด้านบน ถ้าเว้นว่างจะใช้คำอธิบายอัตโนมัติจากสายที่ต่อ";
    container.appendChild(desc);

    const wrap = document.createElement("div");
    wrap.className = "param";
    const lab = document.createElement("div");
    lab.className = "param-label";
    lab.innerHTML = "<span>ชื่อกล่อง</span>";
    wrap.appendChild(lab);
    const input = document.createElement("input");
    input.type = "text";
    input.className = "name-input";
    input.placeholder = "เว้นว่าง = ใช้ชื่ออัตโนมัติ Display N";
    input.value = info.getName();
    input.addEventListener("input", () => info.setName(input.value.trim()));
    wrap.appendChild(input);
    container.appendChild(wrap);

    // รายการขั้นตอนที่ภาพผ่าน (F2) — ค่าดิบ + คลิกไปแก้ที่ node
    if (info.getSteps) renderSteps(container, info);

    // ปุ่มสร้างโค้ด Python (OpenCV) ตามลำดับ Box ที่ภาพวิ่งผ่าน
    if (info.onGenCode) {
      const gen = document.createElement("button");
      gen.className = "btn btn-primary";
      gen.style.width = "100%";
      gen.style.marginTop = "12px";
      gen.textContent = "🐍 สร้างโค้ด Python (OpenCV)";
      gen.title = "สร้างโค้ด image processing ตามลำดับ Box ที่ต่อเข้ากล่องนี้ (คนละอันกับปุ่ม Export เทรน YOLO)";
      gen.addEventListener("click", () => info.onGenCode());
      container.appendChild(gen);

      const genHint = document.createElement("div");
      genHint.className = "param-hint";
      genHint.style.marginTop = "6px";
      genHint.textContent = "ได้โค้ด cv2.* ตามขั้นตอนจริง เอาไปรัน/ศึกษาต่อได้";
      container.appendChild(genHint);
    }
  }

  // สร้างรายการขั้นตอน (ค่าดิบ key=value) ในแผงขวา
  function renderSteps(container, info) {
    const dp = info.getSteps();
    const head = document.createElement("div");
    head.className = "steps-head";
    head.textContent = "ขั้นตอนที่ภาพผ่าน (คลิกเพื่อไปแก้ค่า)";
    container.appendChild(head);

    if (!dp || !dp.connected) {
      const e = document.createElement("div");
      e.className = "steps-empty";
      e.textContent = "กล่องนี้ยังไม่ได้ต่อสายถึงแหล่งภาพ";
      container.appendChild(e);
      return;
    }
    const src = document.createElement("div");
    src.className = "steps-src";
    src.textContent = "● แหล่งภาพ";
    container.appendChild(src);

    if (dp.pipeline.length === 0) {
      const e = document.createElement("div");
      e.className = "steps-empty";
      e.textContent = "ภาพต้นฉบับ — ยังไม่ผ่าน filter ใด";
      container.appendChild(e);
      return;
    }
    dp.pipeline.forEach((s, i) => {
      const f = FILTERS[s.id];
      const row = document.createElement("button");
      row.className = "step-row";
      row.type = "button";
      const idx = document.createElement("span");
      idx.className = "pipe-index";
      idx.textContent = String(i + 1);
      const name = document.createElement("span");
      name.className = "step-name";
      name.textContent = f ? (typeof f.subtitle === "function" ? f.subtitle(s.params) : f.name) : s.id;
      row.appendChild(idx);
      row.appendChild(name);
      if (f) {
        for (const p of f.params) {
          if (p.showIf && s.params[p.showIf.key] !== p.showIf.value) continue;
          if (p.key === "mode") continue;
          const chip = document.createElement("span");
          chip.className = "step-param";
          const k = p.key.includes("__") ? p.key.split("__").pop() : p.key;
          chip.textContent = `${p.label || k}=${s.params[p.key]}`;
          chip.title = k; // key ดิบ (โยงกับโค้ด)
          row.appendChild(chip);
        }
      }
      const arr = document.createElement("span");
      arr.className = "step-arrow";
      arr.textContent = "›";
      row.appendChild(arr);
      row.addEventListener("click", () => info.onJump(s.nodeId));
      container.appendChild(row);
    });
  }

  // แผงกล่อง YOLO Detect — browse โมเดล/labels + เริ่มตรวจจับ
  // info = { getStatus(), detecting, onBrowseModel(), onBrowseLabels(), onToggle() }
  function renderDL(info) {
    curStep = null;
    curNotes = null;
    container.innerHTML = "";

    const title = document.createElement("h2");
    title.className = "detail-title";
    title.textContent = "🧠 YOLO Detect";
    container.appendChild(title);

    const desc = document.createElement("div");
    desc.className = "detail-desc";
    desc.textContent =
      "เลือกไฟล์โมเดล YOLO (best.pt ที่เทรนมา หรือ yolo สำเร็จรูป / .onnx) แล้วตรวจจับวัตถุบนกล้องในโปรแกรมได้เลย — ชื่อคลาสดึงจากตัวโมเดลอัตโนมัติ";
    container.appendChild(desc);

    const status = document.createElement("div");
    status.className = "detail-note detail-note-info";
    status.textContent = "📊 " + info.getStatus();
    container.appendChild(status);

    const mk = (label, onClick, primary) => {
      const b = document.createElement("button");
      b.className = primary ? "btn btn-primary" : "btn";
      b.style.width = "100%";
      b.style.marginTop = "8px";
      b.textContent = label;
      b.addEventListener("click", onClick);
      container.appendChild(b);
    };
    mk("📂 เลือกโมเดล (best.pt / yolo / .onnx)", () => info.onBrowseModel(), true);
    // เลือกไฟล์ชื่อคลาสเอง (classes.txt / data.yaml / .names) — สำหรับ .onnx ที่ไม่มีชื่อคลาสในตัว
    if (info.onBrowseLabels) mk("🏷️ เลือกไฟล์ชื่อคลาส (classes.txt / data.yaml)", () => info.onBrowseLabels(), false);

    const hint = document.createElement("div");
    hint.className = "param-hint";
    hint.style.marginTop = "10px";
    hint.textContent = "ต่อสาย: แหล่งภาพ → กล่องนี้ → ผลลัพธ์ แล้วกล่องผลลัพธ์จะโชว์การตรวจจับ";
    container.appendChild(hint);
  }

  function buildParam(param, value) {
    const wrap = document.createElement("div");
    wrap.className = "param";

    if (param.type === "boolean") {
      // Toggle ON/OFF อ่านง่าย (Design §22) — คลิกทั้งแถวได้ ค่ายังส่ง boolean เท่าเดิม
      const row = document.createElement("button");
      row.type = "button";
      row.className = "param-toggle" + (value ? " on" : "");
      row.setAttribute("role", "switch");
      row.setAttribute("aria-checked", value ? "true" : "false");
      const lab = document.createElement("span");
      lab.className = "pt-label";
      lab.textContent = param.label;
      const sw = document.createElement("span");
      sw.className = "pt-switch";
      sw.innerHTML = `<span class="pt-knob"></span>`;
      const st = document.createElement("span");
      st.className = "pt-state";
      st.textContent = value ? "ON" : "OFF";
      row.appendChild(lab);
      row.appendChild(st);
      row.appendChild(sw);
      row.addEventListener("click", () => {
        const nv = !row.classList.contains("on");
        cb.onParamChange(param.key, nv);
        if (curStep) { curStep.params[param.key] = nv; render(curStep, curNotes); }
      });
      wrap.appendChild(row);
      return wrap;
    }

    // ทุก select → แสดงเป็นรายการปุ่มติ๊กเลือก (แทน dropdown)
    if (param.type === "select") {
      const lab = document.createElement("div");
      lab.className = "param-label";
      lab.innerHTML = `<span>${param.label}</span>`;
      wrap.appendChild(lab);

      const list = document.createElement("div");
      list.className = "choice-list";
      for (const opt of param.options) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "choice" + (opt.value === value ? " active" : "");
        const check = document.createElement("span");
        check.className = "choice-check";
        check.textContent = opt.value === value ? "✓" : "";
        const textCol = document.createElement("span");
        textCol.className = "choice-text";
        const text = document.createElement("span");
        text.className = "choice-label";
        text.textContent = opt.label;
        textCol.appendChild(text);
        b.appendChild(check);
        b.appendChild(textCol);
        b.addEventListener("click", () => {
          if (opt.value === value) return;
          cb.onParamChange(param.key, opt.value);
          // re-render เพื่ออัปเดตติ๊ก + สลับ sub-params (กรณี "เลือกวิธี")
          if (curStep) {
            curStep.params[param.key] = opt.value;
            render(curStep, curNotes);
          }
        });
        list.appendChild(b);
      }
      wrap.appendChild(list);
      return wrap;
    }

    // number → grid: label / slider(1fr) / value box(ชิดขวา ตรงแนวทุกแถว)
    wrap.className = "param param-num";
    const lab = document.createElement("div");
    lab.className = "pn-label";
    lab.textContent = param.label;
    const input = document.createElement("input");
    input.type = "range";
    input.className = "pn-slider";
    input.min = param.min;
    input.max = param.max;
    input.step = param.step || 1;
    input.value = value;
    // ช่องค่า = input ตัวเลข → คลิกพิมพ์ค่าเองได้ + sync กับสไลเดอร์สองทาง
    const box = document.createElement("input");
    box.type = "number";
    box.className = "pn-value mono";
    box.min = param.min;
    box.max = param.max;
    box.step = param.step || 1;
    box.value = value;
    // เลื่อนสไลเดอร์ → อัปเดตช่องเลข
    input.addEventListener("input", () => {
      const v = Number(input.value);
      box.value = v;
      cb.onParamChange(param.key, v);
    });
    // พิมพ์เลขในช่อง → อัปเดตสไลเดอร์+ภาพ (clamp ค่าที่ส่งออก แต่ "ไม่แตะช่อง"
    // ระหว่างพิมพ์ ไม่งั้นค่าที่ต่ำกว่า min จะโดนเด้งจนพิมพ์เลขหลายหลักไม่ได้ — snap ช่องตอน commit)
    const applyTyped = () => {
      if (box.value === "" || box.value === "-") return; // กำลังพิมพ์ ยังไม่ครบ
      const v = Number(box.value);
      if (Number.isNaN(v)) return;
      const c = Math.min(param.max, Math.max(param.min, v));
      input.value = c;
      cb.onParamChange(param.key, c);
    };
    box.addEventListener("input", applyTyped);
    // ออกจากช่อง/กด Enter → snap ค่าให้อยู่ในช่วงและแสดงผลจริง
    const commit = () => {
      if (box.value === "" || box.value === "-" || Number.isNaN(Number(box.value))) {
        box.value = input.value; // เว้นว่าง/ค่าพัง → คืนค่าเดิม ไม่เด้งเป็น min
        return;
      }
      const v = Math.min(param.max, Math.max(param.min, Number(box.value)));
      box.value = v;
      input.value = v;
      cb.onParamChange(param.key, v);
    };
    box.addEventListener("change", commit);
    box.addEventListener("keydown", (e) => { if (e.key === "Enter") { commit(); box.blur(); } });
    wrap.append(lab, input, box);
    return wrap;
  }

  return { render, renderDisplay, renderDL };
}

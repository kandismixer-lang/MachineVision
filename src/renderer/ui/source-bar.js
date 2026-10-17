// แถบด้านบน: สลับแหล่งภาพ (กล้อง / นำเข้าภาพ) + ปุ่มเลือกไฟล์ภาพ + เลือกระดับความละเอียด
// cb: getSource(), onSource('camera'|'image'), onImportClick(),
//     levels[], getLevel(), onLevel(value)
import { keyLabel } from "./settings-dialog.js";

export function createSourceBar(container, cb) {
  function render() {
    const source = cb.getSource();
    container.innerHTML = "";

    // 3 คลัสเตอร์: [แหล่งภาพ] | [นำเข้า] ...spacer... [ส่งออก/ตั้งค่า]
    // แต่ละคลัสเตอร์เป็น 2 แถว: caption (ป้ายกำกับเล็ก ๆ) + body (ปุ่ม/ควบคุมเดิม)
    // แถบบนแถวเดียว: caption ถูกซ่อนด้วย CSS (ยังอ่านได้ด้วย screen reader) + role/aria-label ระบุกลุ่ม
    const CLUSTER_LABELS = {
      SOURCE: "แหล่งภาพ", MODE: "โหมดกล้อง", IMPORT: "นำเข้าภาพ", CONTROL: "ควบคุม", "EXPORT & SETTINGS": "บันทึกและตั้งค่า",
    };
    function makeCluster(extraClass, caption) {
      const cluster = document.createElement("div");
      cluster.className = "tb-cluster " + extraClass;
      cluster.setAttribute("role", "group");
      cluster.setAttribute("aria-label", CLUSTER_LABELS[caption] || caption);
      const cap = document.createElement("div");
      cap.className = "tb-caption";
      cap.textContent = caption;
      const body = document.createElement("div");
      body.className = "tb-cluster-body";
      cluster.appendChild(cap);
      cluster.appendChild(body);
      return { cluster, body };
    }
    const srcC = makeCluster("tb-source", "SOURCE");
    const modeC = makeCluster("tb-mode", "MODE");
    const importC = makeCluster("tb-import", "IMPORT");
    const importCluster = importC.body; // เนื้อหาเดิม append เข้า body — ครอบด้วย caption ตอนประกอบแถบด้านล่าง
    const controlC = makeCluster("tb-control", "CONTROL");
    const outputC = makeCluster("tb-output", "EXPORT & SETTINGS");
    const outputCluster = outputC.body;

    const seg = document.createElement("div");
    seg.className = "seg";

    const camBtn = document.createElement("button");
    camBtn.textContent = "📷 กล้อง";
    camBtn.title = "ใช้ภาพสดจากกล้อง webcam";
    if (source === "camera") camBtn.classList.add("active");
    camBtn.addEventListener("click", () => cb.onSource("camera"));
    seg.appendChild(camBtn);

    const imgBtn = document.createElement("button");
    imgBtn.textContent = "🖼️ ภาพนิ่ง";
    imgBtn.title = "นำเข้าภาพ/วิดีโอจากไฟล์ — ทดลองซ้ำได้ผลเดิม เหมาะฝึกทีละหลักการ";
    if (source === "image") imgBtn.classList.add("active");
    imgBtn.addEventListener("click", () => cb.onSource("image"));
    seg.appendChild(imgBtn);

    srcC.body.appendChild(seg); // SOURCE cluster = กล้อง/ภาพนิ่ง เท่านั้น

    // โหมดย่อยของกล้อง → MODE cluster แยกต่างหาก: [📸 แคปภาพ][ต้นฉบับ][🧍 Pose] ● สด (กล้อง:▼)
    if (source === "camera" && cb.onCameraMode) {
      const mode = cb.getCameraMode ? cb.getCameraMode() : "original";
      const modeRow = document.createElement("div");
      modeRow.className = "mode-row";

      // ปุ่มแคปภาพ — เก็บเฟรมกล้องเข้าคลัง (action ไม่ใช่ toggle จึงแยกจาก seg)
      if (cb.onCapture) {
        const capBtn = document.createElement("button");
        capBtn.className = "btn btn-primary";
        capBtn.textContent = "📸 ถ่ายภาพ";
        capBtn.title = "เก็บเฟรมกล้องปัจจุบันเข้าคลัง (ดูและเซฟได้ในถาดล่าง)";
        // โชว์ keycap ของคีย์ลัดที่ผูกไว้ (ตั้งได้ที่ปุ่มฟันเฟือง ⚙)
        const sc = cb.getShortcut ? cb.getShortcut("capture") : "";
        if (sc) {
          const cap = document.createElement("span");
          cap.className = "keycap";
          cap.textContent = keyLabel(sc);
          capBtn.appendChild(document.createTextNode(" "));
          capBtn.appendChild(cap);
        }
        capBtn.addEventListener("click", () => cb.onCapture());
        modeRow.appendChild(capBtn);
      }

      // เลือกกล้อง (Port) เมื่อมีหลายตัว — ต่อท้ายในแถวเดียวกัน
      const cams = cb.getCameras ? cb.getCameras() : [];
      if (cams.length > 1) {
        const camWrap = document.createElement("label");
        camWrap.className = "level-select cam-select";
        camWrap.title = "เลือกกล้องที่ต่อกับเครื่อง";
        camWrap.appendChild(document.createTextNode("Camera:"));
        const csel = document.createElement("select");
        cams.forEach((c, i) => {
          const o = document.createElement("option");
          o.value = c.deviceId;
          const cleanName = c.label.replace(/\s*\([^)]*\)\s*$/, "").trim();
          o.textContent = `Port ${i + 1} — ${cleanName}`;
          if (c.deviceId === cb.getCameraId()) o.selected = true;
          csel.appendChild(o);
        });
        csel.addEventListener("change", () => cb.onSelectCamera(csel.value));
        camWrap.appendChild(csel);
        modeRow.appendChild(camWrap);
      }

      // สถานะ ● สด (เขียว) เมื่อกำลังประมวลผลสด ไม่ได้ pause
      if (!(cb.isPaused && cb.isPaused())) {
        const live = document.createElement("span");
        live.className = "tb-live";
        live.textContent = "● Live";
        modeRow.appendChild(live);
      }

      modeC.body.appendChild(modeRow);
    }

    // primary สลับตามบริบท: โหมดภาพนิ่ง → "นำเข้าภาพ" เด่น | โหมดกล้อง → "แคปภาพ" เด่น (นำเข้าเป็นเทา)
    const importBtn = document.createElement("button");
    importBtn.className = source === "image" ? "btn btn-primary" : "btn";
    importBtn.textContent = "＋ นำเข้าภาพ / วิดีโอ…";
    importBtn.title = "เลือกภาพ วิดีโอ หรือ YOLO / Roboflow Dataset (.zip) พร้อม Label และ Split";
    importBtn.addEventListener("click", () => cb.onImportClick());
    importCluster.appendChild(importBtn);

    // ปุ่มนำเข้าทั้งโฟลเดอร์ (batch)
    if (cb.onImportFolder) {
      const folderBtn = document.createElement("button");
      folderBtn.className = "btn";
      folderBtn.className = "btn import-folder-btn";
      folderBtn.textContent = "📁";
      folderBtn.title = "เลือกทั้งโฟลเดอร์ — นำเข้าหลายภาพพร้อมกัน แล้วกด Next ดูทีละภาพ (ทุกภาพผ่าน filter เดียวกัน)";
      folderBtn.addEventListener("click", () => cb.onImportFolder());
      importCluster.appendChild(folderBtn);
    }

    // ตัวนำทางชุดภาพ (แสดงเมื่อมีภาพหลายรูป): ◀  ภาพ X/N  ▶
    const batch = cb.getBatch && cb.getBatch();
    if (batch && batch.total > 1) {
      const nav = document.createElement("div");
      nav.className = "batch-nav";

      const prev = document.createElement("button");
      prev.className = "batch-btn";
      prev.textContent = "◀";
      prev.title = "ภาพก่อนหน้า";
      prev.addEventListener("click", () => cb.onPrev());

      const info = document.createElement("span");
      info.className = "batch-info";
      const shortName = batch.name.length > 22 ? batch.name.slice(0, 21) + "…" : batch.name;
      info.textContent = `ภาพ ${batch.index + 1}/${batch.total} · ${shortName}`;
      info.title = batch.name;

      const next = document.createElement("button");
      next.className = "batch-btn";
      next.textContent = "▶";
      next.title = "ภาพถัดไป";
      next.addEventListener("click", () => cb.onNext());

      nav.appendChild(prev);
      nav.appendChild(info);
      nav.appendChild(next);
      importCluster.appendChild(nav);
    }

    // ปุ่มเคลียร์ภาพทั้งหมด — โผล่เมื่อมีภาพนำเข้าอยู่ (โหมดภาพนิ่ง) เพื่อเริ่ม dataset ใหม่
    if (cb.onClearImages && source === "image" && batch && batch.total >= 1) {
      const clearBtn = document.createElement("button");
      clearBtn.className = "btn import-clear-btn";
      clearBtn.textContent = "🗑 เคลียร์ภาพ";
      clearBtn.title = "ลบภาพที่นำเข้าทั้งหมด (เริ่มนำเข้า dataset ชุดใหม่) — ป้ายกำกับ/คลาสไม่ถูกลบ";
      clearBtn.addEventListener("click", () => cb.onClearImages());
      importCluster.appendChild(clearBtn);
    }

    // ปุ่ม Run / Pause การประมวลผลสด — เฉพาะโหมดกล้อง (มี live loop ให้หยุดจริง)
    if (cb.onTogglePause && source === "camera") {
      const paused = cb.isPaused ? cb.isPaused() : false;
      const runBtn = document.createElement("button");
      runBtn.className = paused ? "btn btn-run" : "btn btn-run running";
      runBtn.textContent = paused ? "▶ Run" : "⏸ Pause";
      runBtn.title = paused ? "เริ่มประมวลผลสดต่อ" : "หยุดประมวลผลชั่วคราว (ประหยัด CPU)";
      runBtn.addEventListener("click", () => cb.onTogglePause());
      controlC.body.appendChild(runBtn);
    }

    // ปุ่มเซฟภาพ (PNG) และปุ่มโค้ด YOLO — แยกกันชัดเจน (ghost — ปลายทาง)
    if (cb.onSaveImages) {
      const saveBtn = document.createElement("button");
      saveBtn.id = "save-picture-btn"; // ให้ app.js หาเจอเพื่อ disable ระหว่าง export อื่นทำงานอยู่
      saveBtn.className = "btn btn-save"; // ปุ่มรอง (เดิมเขียวทึบ + ตัวขาว contrast ~2.1:1 ไม่ผ่าน)
      saveBtn.textContent = "💾 Save Picture";
      saveBtn.title = "บันทึกภาพผลลัพธ์แต่ละกล่องเป็นไฟล์ PNG (เลือกโฟลเดอร์)";
      if (cb.exportBusy && cb.exportBusy()) {
        saveBtn.disabled = true;
        saveBtn.title = "กำลัง export อยู่ — รอให้เสร็จก่อน";
      }
      saveBtn.addEventListener("click", () => cb.onSaveImages());
      outputCluster.appendChild(saveBtn);
      // P-8 C-B: ภาพนิ่งหลายภาพ → เลือกได้ว่าจะเขียน CSV ค่าที่วัดได้ของทุกภาพลงโฟลเดอร์เดียวกับภาพหรือไม่
      if (cb.onBatchCsv && source === "image" && batch && batch.total > 1) {
        const wrap = document.createElement("label");
        wrap.className = "batch-csv-opt";
        wrap.title = "ตอนเซฟทั้งโฟลเดอร์ จะเขียนไฟล์ CSV ค่าที่วัดได้ของทุกภาพ (สรุปต่อภาพ + รายวัตถุ) ไว้ข้างภาพผลลัพธ์ — ถ้าไม่มีกล่องที่วัดค่าจะไม่สร้างไฟล์";
        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = cb.getBatchCsv ? !!cb.getBatchCsv() : true;
        box.addEventListener("change", () => cb.onBatchCsv(box.checked));
        const txt = document.createElement("span");
        txt.textContent = "📊 CSV ค่าที่วัด";
        wrap.append(box, txt);
        outputCluster.appendChild(wrap);
      }
    }
    // Version2ui ย้าย Export YOLO Code ไปอยู่ใน Dataset workflow เพื่อไม่ปนกับการเซฟภาพ
    if (cb.onExportCode && !document.body.classList.contains("ui-v2")) {
      const codeBtn = document.createElement("button");
      codeBtn.id = "export-code-btn";
      codeBtn.className = "btn btn-yolo-ghost"; // รอง (ghost ม่วง) — เด่นน้อยกว่า Save
      codeBtn.textContent = "⬇ Export Yolo";
      codeBtn.title = "ดาวน์โหลดโค้ดเทรน YOLO (.ipynb) ";
      if (cb.exportBusy && cb.exportBusy()) {
        codeBtn.disabled = true;
        codeBtn.title = "กำลัง export อยู่ — รอให้เสร็จก่อน";
      }
      codeBtn.addEventListener("click", () => cb.onExportCode());
      outputCluster.appendChild(codeBtn);
    }

    // เลือกระดับความละเอียดประมวลผล (ยิ่งเล็กยิ่งลื่น ลด delay)
    if (cb.levels) {
      const wrap = document.createElement("label");
      wrap.className = "level-select";
      wrap.title = "ย่อภาพก่อนประมวลผลเพื่อลดหน่วง — เลือกให้เหมาะกับเครื่อง";
      const lvLabel = document.createElement("span");
      lvLabel.className = "level-label";
      lvLabel.textContent = "ความละเอียด:";
      wrap.appendChild(lvLabel);
      const sel = document.createElement("select");
      for (const lv of cb.levels) {
        const o = document.createElement("option");
        o.value = lv.value;
        o.textContent = lv.label;
        if (lv.value === cb.getLevel()) o.selected = true;
        sel.appendChild(o);
      }
      sel.addEventListener("change", () => cb.onLevel(sel.value));
      wrap.appendChild(sel);
      outputCluster.appendChild(wrap);
    }

    // ปุ่มฟันเฟือง ⚙ — ตั้งค่าคีย์ลัด (ขวาบนสุด)
    if (cb.onOpenSettings) {
      const gear = document.createElement("button");
      gear.className = "btn btn-icon";
      gear.textContent = "⚙";
      gear.title = "ตั้งค่าคีย์ลัด (keyboard shortcut)";
      gear.addEventListener("click", () => cb.onOpenSettings());
      outputCluster.appendChild(gear);
    }

    // ประกอบแถบ: SOURCE | [MODE |] IMPORT ...spacer... [CONTROL |] EXPORT
    container.appendChild(srcC.cluster);
    const div1 = document.createElement("div");
    div1.className = "tb-divider";
    container.appendChild(div1);
    // MODE โผล่เฉพาะโหมดกล้อง (มีปุ่มโหมดย่อย)
    if (modeC.body.children.length) {
      container.appendChild(modeC.cluster);
      const divM = document.createElement("div");
      divM.className = "tb-divider";
      container.appendChild(divM);
    }
    container.appendChild(importC.cluster);
    const spacer = document.createElement("div");
    spacer.className = "tb-spacer";
    container.appendChild(spacer);
    // "ควบคุม" โผล่เฉพาะโหมดกล้อง (มีปุ่ม Run/Pause) — ไม่โผล่ cluster ว่างเปล่า
    if (controlC.body.children.length) {
      container.appendChild(controlC.cluster);
      const div2 = document.createElement("div");
      div2.className = "tb-divider";
      container.appendChild(div2);
    }
    container.appendChild(outputC.cluster);
  }

  return { render };
}

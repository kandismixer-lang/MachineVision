// หน้าต่างตั้งค่าก่อน Export — ถามค่าเทรน (YOLO หรือ RF-DETR) แล้วส่งค่ากลับ
// เปิดด้วย open(onConfirm) : onConfirm({ epochs, batch, dataPath, framework, ... })
import { getExportFramework, setExportFramework } from "../app.js";
import { getClasses } from "../dataset/annotations.js";

export function createExportDialog() {
  const BATCHES = [16, 32, 64];
  const IMG_SIZES = [
    { v: 640, label: "640", hint: "เร็ว" },
    { v: 960, label: "960", hint: "กลาง" },
    { v: 1280, label: "1280", hint: "แม่น" },
  ];
  // เวอร์ชัน YOLO ให้เลือกตอน export — ตั้งแต่ v11 ขึ้นไป ultralytics ตัด "v" ออก (yolo11n)
  const YOLO_VERSIONS = [
    { v: "yolov8n", label: "YOLOv8", hint: "เสถียร" },
    { v: "yolo11n", label: "YOLO11", hint: "แนะนำ" },
    { v: "yolo12n", label: "YOLO12", hint: "ใหม่" },
    { v: "yolo26n", label: "YOLO26", hint: "ล่าสุด" },
  ];
  // อุปกรณ์ที่ใช้เทรน — เป็นค่าเริ่มของ dropdown DEVICE ใน notebook (บน Colab ยังเปลี่ยนได้)
  const DEVICES = [
    { v: "auto", label: "Auto", hint: "GPU ถ้ามี", summary: "Auto (GPU ถ้ามี)" },
    { v: "gpu", label: "GPU", hint: "บังคับ", summary: "GPU" },
    { v: "cpu", label: "CPU", hint: "ช้ามาก", summary: "CPU" },
  ];

  function open(onConfirm, readiness = {}) {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";

    const modal = document.createElement("div");
    modal.className = "modal export-yolo-modal";
    overlay.appendChild(modal);

    const header = document.createElement("div");
    header.className = "yolo-export-header";
    const codeIcon = document.createElement("div");
    codeIcon.className = "yolo-code-icon";
    codeIcon.textContent = "</>";
    const heading = document.createElement("div");
    heading.className = "yolo-export-heading";

    const h = document.createElement("h2");
    h.className = "modal-title";
    h.textContent = "Export YOLO Code";
    heading.appendChild(h);

    const sub = document.createElement("div");
    sub.className = "modal-sub";
    sub.textContent = "ตั้งค่าการเทรน แล้วสร้าง Jupyter Notebook (.ipynb) สำหรับ Google Colab";
    heading.appendChild(sub);

    const headerClose = document.createElement("button");
    headerClose.type = "button";
    headerClose.className = "yolo-modal-close";
    headerClose.textContent = "×";
    headerClose.title = "Close";
    headerClose.setAttribute("data-close", "");
    header.append(codeIcon, heading, headerClose);
    modal.appendChild(header);

    // เตือนเรื่อง labeling — ภาพเปล่าเทรนไม่ได้ ต้องตีกรอบ+ระบุคลาสก่อน
    const note = document.createElement("div");
    note.className = "yolo-readiness";
    const readyItem = (ready, label) =>
      `<span class="yolo-ready-item ${ready ? "ready" : "pending"}"><b>${ready ? "✓" : "!"}</b>${label}</span>`;
    note.innerHTML =
      readyItem(!!readiness.datasetReady, readiness.datasetReady ? "Dataset labeled" : "Dataset needs labels") +
      readyItem(!!readiness.zipReady, readiness.zipReady ? "dataset.zip ready" : "Export dataset.zip") +
      `<span class="yolo-how">Upload ZIP to Colab before running</span>`;
    modal.appendChild(note);

    const body = document.createElement("div");
    body.className = "yolo-export-body";
    const setupPane = document.createElement("section");
    setupPane.className = "yolo-setup-pane";
    const setupTitle = document.createElement("h3");
    setupTitle.textContent = "Training setup";
    setupPane.appendChild(setupTitle);

    // framework เลือกในหน้าต่างนี้ (เริ่มจากค่าที่เลือกครั้งก่อน) — dataset .zip ใช้แบบ YOLO ร่วมกันอยู่แล้ว
    let framework = getExportFramework();
    // ซ่อน field เฉพาะ YOLO (เวอร์ชัน + imgsz) + ปรับ label ปุ่มตาม framework
    function applyFramework() {
      const isYolo = framework === "yolo";
      fwButtons.forEach((b) => b.classList.toggle("active", b.dataset.fw === framework));
      fwHint.textContent = isYolo
        ? "YOLO — เร็ว เบา · ultralytics"
        : "RF-DETR — แม่นกว่า กิน VRAM มากกว่า · package rfdetr";
      if (vField) vField.style.display = isYolo ? "" : "none";
      if (iField) iField.style.display = isYolo ? "" : "none";
      // aField (augmentation dataset) ใช้ทั้ง YOLO และ RF-DETR — ไม่ซ่อน
      if (rfField) rfField.style.display = isYolo ? "none" : ""; // ขนาดโมเดล RF-DETR โชว์เฉพาะ rfdetr
      ok.textContent = isYolo ? "สร้าง YOLO Notebook" : "สร้าง RF-DETR Notebook";
      if (h) h.textContent = isYolo ? "Export YOLO Code" : "Export RF-DETR Code";
      if (bTitle) bTitle.textContent = "Batch size";
      if (bHint) bHint.textContent = isYolo ? "จำนวนภาพต่อรอบ" : "ภาพต่อ step (เล็กกัน VRAM บน T4)";
    }
    const summaryPane = document.createElement("aside");
    summaryPane.className = "yolo-summary-pane";
    body.append(setupPane, summaryPane);
    modal.appendChild(body);

    // ===== Framework (YOLO / RF-DETR) — เปลี่ยนโมเดล + batch ที่ให้เลือก + โค้ดที่สร้าง =====
    const fwField = document.createElement("div");
    fwField.className = "setup-field";
    const fwLabel = document.createElement("div");
    fwLabel.className = "setup-label";
    fwLabel.innerHTML = "<span>Framework</span><span class='setup-hint'>โมเดลที่จะเทรนบน Colab</span>";
    const fwToggle = document.createElement("div");
    fwToggle.className = "framework-toggle";
    fwToggle.setAttribute("role", "tablist");
    fwToggle.setAttribute("aria-label", "Framework");
    const fwButtons = [["yolo", "YOLO"], ["rfdetr", "RF-DETR"]].map(([v, label]) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "fw-btn";
      b.dataset.fw = v;
      b.textContent = label;
      b.addEventListener("click", () => {
        if (framework === v) return;
        framework = v;
        setExportFramework(v); // จำไว้ เปิดครั้งหน้าเริ่มที่ตัวเดิม
        renderBatchCards();
        applyFramework();
        updateSummary();
      });
      fwToggle.appendChild(b);
      return b;
    });
    const fwHint = document.createElement("div");
    fwHint.className = "framework-hint";
    fwField.append(fwLabel, fwToggle, fwHint);
    setupPane.appendChild(fwField);

    // ===== Epochs (พิมพ์เอา + ปุ่ม −/+) =====
    const epField = document.createElement("div");
    epField.className = "setup-field";
    const epLabel = document.createElement("div");
    epLabel.className = "setup-label";
    epLabel.innerHTML = "<span>จำนวน Epochs</span><span class='setup-hint'>จำนวนรอบการเทรน</span>";
    epField.appendChild(epLabel);

    const stepper = document.createElement("div");
    stepper.className = "stepper";
    const minus = document.createElement("button");
    minus.type = "button";
    minus.className = "stepper-btn";
    minus.textContent = "−";
    const epInput = document.createElement("input");
    epInput.type = "number";
    epInput.min = "1";
    epInput.step = "1";
    epInput.value = "50";
    epInput.className = "stepper-input";
    const plus = document.createElement("button");
    plus.type = "button";
    plus.className = "stepper-btn";
    plus.textContent = "+";
    const clampEp = () => {
      let v = Math.round(Number(epInput.value) || 1);
      if (v < 1) v = 1;
      epInput.value = String(v);
    };
    minus.addEventListener("click", () => { epInput.value = String(Math.max(1, (Math.round(Number(epInput.value) || 1)) - 10)); updateSummary(); });
    plus.addEventListener("click", () => { epInput.value = String((Math.round(Number(epInput.value) || 0)) + 10); updateSummary(); });
    epInput.addEventListener("change", () => { clampEp(); updateSummary(); });
    epInput.addEventListener("input", updateSummary);
    stepper.appendChild(minus);
    stepper.appendChild(epInput);
    stepper.appendChild(plus);
    epField.appendChild(stepper);
    setupPane.appendChild(epField);

    // ===== Batch (การ์ดใหญ่ให้กดเลือก) =====
    const bField = document.createElement("div");
    bField.className = "setup-field";
    const bLabel = document.createElement("div");
    bLabel.className = "setup-label";
    const bTitle = document.createElement("span");
    const bHint = document.createElement("span");
    bHint.className = "setup-hint";
    bLabel.append(bTitle, bHint);
    bField.appendChild(bLabel);

    const grid = document.createElement("div");
    grid.className = "batch-grid batch-select-grid";
    // RF-DETR ใช้ batch จริงเล็ก (2/4/8 กัน OOM บน T4) · YOLO ใช้ batch ปกติ (16/32/64)
    // สลับ framework → สร้างการ์ดใหม่ + รีเซ็ตเป็นค่าเริ่มของ framework นั้น
    let batch;
    function renderBatchCards() {
      const batchOptions = framework === "rfdetr" ? [2, 4, 8] : BATCHES;
      batch = framework === "rfdetr" ? 4 : 32;
      const cards = [];
      grid.replaceChildren();
      for (const b of batchOptions) {
        const card = document.createElement("button");
        card.type = "button";
        card.className = "batch-card" + (b === batch ? " active" : "");
        card.textContent = String(b);
        card.addEventListener("click", () => {
          batch = b;
          cards.forEach((c) => c.classList.toggle("active", Number(c.textContent) === batch));
          updateSummary();
        });
        cards.push(card);
        grid.appendChild(card);
      }
    }
    renderBatchCards();
    bField.appendChild(grid);
    setupPane.appendChild(bField);

    // ===== เวอร์ชัน YOLO (Dropdown — มีตัวเลือกเดียวที่ชัดเจน ไม่กินพื้นที่แบบการ์ด) =====
    const vField = document.createElement("div");
    vField.className = "setup-field";
    const vLabel = document.createElement("div");
    vLabel.className = "setup-label";
    vLabel.innerHTML = "<span>เวอร์ชัน YOLO</span><span class='setup-hint'>เลือกโมเดลพื้นฐานที่จะเทรน</span>";
    vField.appendChild(vLabel);

    let model = "yolo11n";
    const versionSelect = document.createElement("select");
    versionSelect.className = "yolo-version-select";
    versionSelect.setAttribute("aria-label", "YOLO version");
    for (const m of YOLO_VERSIONS) {
      const option = document.createElement("option");
      option.value = m.v;
      option.textContent = `${m.label} — ${m.hint}`;
      option.selected = m.v === model;
      versionSelect.appendChild(option);
    }
    versionSelect.addEventListener("change", () => { model = versionSelect.value; updateSummary(); });
    vField.appendChild(versionSelect);
    setupPane.appendChild(vField);

    // ===== ขนาดโมเดล RF-DETR (โชว์เฉพาะตอนเลือก RF-DETR) — เล็ก = เบา เร็ว เหมาะ T4 ฟรี =====
    const RFDETR_MODELS = [
      { v: "RFDETRNano", label: "Nano", hint: "เล็กสุด เร็วสุด (แนะนำ T4 ฟรี)" },
      { v: "RFDETRSmall", label: "Small", hint: "กลาง" },
      { v: "RFDETRBase", label: "Base", hint: "ใหญ่ แม่นสุด (กิน VRAM สูง)" },
    ];
    let rfModel = "RFDETRNano";
    const rfField = document.createElement("div");
    rfField.className = "setup-field";
    const rfLabel = document.createElement("div");
    rfLabel.className = "setup-label";
    rfLabel.innerHTML = "<span>ขนาดโมเดล RF-DETR</span><span class='setup-hint'>เล็ก = เบา เร็ว · ใหญ่ = แม่นแต่กิน VRAM</span>";
    rfField.appendChild(rfLabel);
    const rfSelect = document.createElement("select");
    rfSelect.className = "yolo-version-select";
    rfSelect.setAttribute("aria-label", "RF-DETR model size");
    for (const m of RFDETR_MODELS) {
      const option = document.createElement("option");
      option.value = m.v;
      option.textContent = `${m.label} — ${m.hint}`;
      option.selected = m.v === rfModel;
      rfSelect.appendChild(option);
    }
    rfSelect.addEventListener("change", () => { rfModel = rfSelect.value; updateSummary(); });
    rfField.appendChild(rfSelect);
    setupPane.appendChild(rfField);

    // ===== ขนาดภาพ (imgsz) — เร็ว↔แม่น เหมือน Roboflow =====
    const iField = document.createElement("div");
    iField.className = "setup-field";
    const iLabel = document.createElement("div");
    iLabel.className = "setup-label";
    iLabel.innerHTML =
      "<span>ขนาดภาพ (imgsz)</span><span class='setup-hint'>เล็ก = เร็ว (จับวัตถุจิ๋วยากขึ้น) · ใหญ่ = แม่น (ช้าลง)</span>";
    iField.appendChild(iLabel);

    const imgGrid = document.createElement("div");
    imgGrid.className = "batch-grid image-size-grid";
    let imgsz = 640;
    const imgCards = [];
    for (const s of IMG_SIZES) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "batch-card" + (s.v === imgsz ? " active" : "");
      card.innerHTML = `${s.label} <span class="setup-hint">(${s.hint})</span>`;
      card.addEventListener("click", () => {
        imgsz = s.v;
        imgCards.forEach((c, idx) => c.classList.toggle("active", IMG_SIZES[idx].v === imgsz));
        updateSummary();
      });
      imgCards.push(card);
      imgGrid.appendChild(card);
    }
    iField.appendChild(imgGrid);
    setupPane.appendChild(iField);

    // ===== อุปกรณ์ที่ใช้เทรน (Auto / GPU / CPU) — ใช้ทั้ง YOLO และ RF-DETR =====
    const dField = document.createElement("div");
    dField.className = "setup-field";
    const dLabel = document.createElement("div");
    dLabel.className = "setup-label";
    dLabel.innerHTML = "<span>อุปกรณ์ที่ใช้เทรน</span><span class='setup-hint'>Auto = ใช้ GPU ถ้ามี ไม่มีใช้ CPU · เปลี่ยนทีหลังใน Colab ได้</span>";
    dField.appendChild(dLabel);
    const devGrid = document.createElement("div");
    devGrid.className = "batch-grid device-grid";
    let device = "auto";
    const devCards = [];
    for (const d of DEVICES) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "batch-card" + (d.v === device ? " active" : "");
      card.dataset.device = d.v;
      card.innerHTML = `${d.label} <span class="setup-hint">(${d.hint})</span>`;
      card.addEventListener("click", () => {
        device = d.v;
        devCards.forEach((c) => c.classList.toggle("active", c.dataset.device === device));
        updateSummary();
      });
      devCards.push(card);
      devGrid.appendChild(card);
    }
    dField.appendChild(devGrid);
    setupPane.appendChild(dField);


    // ===== สรุปค่าที่จะใช้ใน Notebook =====
    const summaryTitle = document.createElement("h3");
    summaryTitle.textContent = "Notebook summary";
    summaryPane.appendChild(summaryTitle);
    const summaryValues = {};
    const summaryRows = [
      ["model", "◇", "Model"],
      ["epochs", "↻", "Epochs"],
      ["batch", "⌘", "Batch"],
      ["image", "▧", "Image"],
      ["device", "⚙", "Device"],
      ["output", "▤", "Output"],
    ];
    for (const [key, icon, label] of summaryRows) {
      const row = document.createElement("div");
      row.className = `yolo-summary-row yolo-summary-${key}`;
      const glyph = document.createElement("span");
      glyph.className = "yolo-summary-icon";
      glyph.textContent = icon;
      const name = document.createElement("span");
      name.textContent = label;
      const value = document.createElement("strong");
      summaryValues[key] = value;
      row.append(glyph, name, value);
      summaryPane.appendChild(row);
    }
    const summaryNote = document.createElement("div");
    summaryNote.className = "yolo-summary-note";
    summaryNote.textContent = "ⓘ You can change these values later in Colab.";
    summaryPane.appendChild(summaryNote);

    function updateSummary() {
      if (!summaryValues.model) return;
      const isYolo = framework === "yolo";
      const selected = YOLO_VERSIONS.find((m) => m.v === model);
      const rfSel = RFDETR_MODELS.find((m) => m.v === rfModel);
      summaryValues.model.textContent = isYolo ? (selected?.label || model) : `RF-DETR ${rfSel?.label || ""}`.trim();
      summaryValues.epochs.textContent = String(Math.max(1, Math.round(Number(epInput.value) || 1)));
      summaryValues.batch.textContent = String(batch);
      // resolution ของ RF-DETR ต่างตามรุ่น (Nano 384 / Small 512 / Base 560)
      const rfRes = rfModel === "RFDETRNano" ? 384 : rfModel === "RFDETRSmall" ? 512 : 560;
      summaryValues.image.textContent = isYolo ? `${imgsz} px` : `${rfRes} px`;
      summaryValues.device.textContent = DEVICES.find((d) => d.v === device)?.summary || device;
      summaryValues.output.textContent = isYolo ? "train_yolo.ipynb" : "train_rfdetr.ipynb";
    }
    updateSummary();

    // ===== ปุ่ม =====
    const actions = document.createElement("div");
    actions.className = "modal-actions";
    const cancel = document.createElement("button");
    cancel.className = "btn";
    cancel.textContent = "ยกเลิก";
    cancel.setAttribute("data-close", ""); // ESC ปิด (Design §22)
    const ok = document.createElement("button");
    ok.className = "btn btn-primary";
    ok.textContent = "สร้าง YOLO Notebook";
    actions.appendChild(cancel);
    actions.appendChild(ok);
    modal.appendChild(actions);

    function close() {
      overlay.remove();
    }
    headerClose.addEventListener("click", close);
    cancel.addEventListener("click", close);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) close();
    });
    ok.addEventListener("click", () => {
      clampEp();
      const epochs = Math.max(1, Math.round(Number(epInput.value) || 50));
      // seg = มี polygon ใน dataset ไหม → เลือกโมเดล -seg ให้ตรงตั้งแต่ตอน gen โค้ด (ไม่ต้อง auto-detect บน Colab)
      const opts = { epochs, batch, imgsz, model, rfModel, framework, device, dataPath: "", numClasses: getClasses().length };
      showCodeView(modal, opts, close);
      onConfirm(opts); // ให้ app จัดการ export ภาพ dataset ต่อ
    });

    applyFramework(); // ตั้งสถานะเริ่มต้นตาม framework ที่จำไว้
    document.body.appendChild(overlay);
    epInput.focus();
    epInput.select();
  }

  // มุมมองโค้ด (สไตล์ Roboflow): โชว์โค้ดให้คัดลอก + ปุ่มดาวน์โหลด
  function showCodeView(modal, opts, close) {
    const code = buildTrainingCode(opts);
    modal.innerHTML = "";
    modal.classList.remove("export-yolo-modal");
    modal.classList.add("modal-wide");

    const h = document.createElement("h2");
    h.className = "modal-title";
    h.textContent = opts.framework === "rfdetr" ? "โค้ดเทรน RF-DETR" : "โค้ดเทรน YOLO";
    modal.appendChild(h);

    const sub = document.createElement("div");
    sub.className = "modal-sub";
    sub.textContent = "คัดลอกโค้ดไปวางใน Colab/Notebook ได้เลย หรือกดดาวน์โหลดเป็นไฟล์ .py";
    modal.appendChild(sub);

    const pre = document.createElement("pre");
    pre.className = "code-block";
    pre.textContent = code;
    modal.appendChild(pre);

    const actions = document.createElement("div");
    actions.className = "modal-actions";

    const copyBtn = document.createElement("button");
    copyBtn.className = "btn btn-primary";
    copyBtn.textContent = "📋 คัดลอกโค้ด";
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code);
      } catch {
        // fallback ถ้า clipboard API ใช้ไม่ได้
        const ta = document.createElement("textarea");
        ta.value = code;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      copyBtn.textContent = "✓ คัดลอกแล้ว";
      setTimeout(() => (copyBtn.textContent = "📋 คัดลอกโค้ด"), 1500);
    });

    const dlBtn = document.createElement("button");
    dlBtn.className = "btn";
    dlBtn.textContent = "⬇ ดาวน์โหลด .ipynb";
    dlBtn.title = "ไฟล์ Jupyter Notebook — เปิดใน Google Colab แล้วกด Run ได้เลย";
    dlBtn.addEventListener("click", () => {
      const nb = buildNotebook(opts);
      const url = URL.createObjectURL(new Blob([nb], { type: "application/x-ipynb+json" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = opts.framework === "rfdetr" ? "train_rfdetr.ipynb" : "train_yolo.ipynb";
      a.click();
      URL.revokeObjectURL(url);
    });

    const closeBtn = document.createElement("button");
    closeBtn.className = "btn";
    closeBtn.textContent = "ปิด";
    closeBtn.setAttribute("data-close", ""); // ESC ปิด (Design §22)
    closeBtn.addEventListener("click", close);

    actions.appendChild(copyBtn);
    actions.appendChild(dlBtn);
    actions.appendChild(closeBtn);
    modal.appendChild(actions);
  }

  return { open };
}

// แปลงข้อความโค้ดเป็น source ของ notebook cell (แต่ละบรรทัดมี \n ต่อท้าย ยกเว้นบรรทัดสุดท้าย)
function toCellSource(text) {
  const lines = text.split("\n");
  return lines.map((l, i) => (i < lines.length - 1 ? l + "\n" : l));
}

// helper สร้าง code cell
function codeCell(text) {
  return { cell_type: "code", execution_count: null, metadata: {}, outputs: [], source: toCellSource(text) };
}

// บล็อกโค้ดแยกส่วน (ใช้ทั้งใน notebook และรวมเป็นข้อความคัดลอก)
const INSTALL_BLOCK = "!pip install ultralytics";
const IMPORTS_BLOCK = `import numpy as np
import ultralytics
import tensorflow as tf
import seaborn
import matplotlib.pyplot as plt
from ultralytics import YOLO
from IPython.display import Image`;

// ขั้นตอนก่อนเทรน (Phase 4): ผู้ใช้ export "dataset.zip" จากแอป (ปุ่ม 📦 Export Dataset ใน DATASET CLASSES)
// แล้วอัปโหลดไฟล์นั้นเข้า Colab (แถบไฟล์ด้านซ้าย หรือ Files > Upload) ก่อนกด Run ทั้งหมด
// เซลล์นี้แตกไฟล์ zip นั้นให้เป็นโฟลเดอร์ dataset/ ตรงตามโครงที่ export-yolo.js สร้าง (images/, labels/, data.yaml)
const UNZIP_BLOCK = `ZIP = ""
!unzip -qo "{ZIP}" -d /content/dataset`;

// ต้องใช้ imgsz เดียวกันทั้งตอนเทรนและตอน export — ไม่งั้น onnx จะถูกล็อกที่ความละเอียดอื่น
// ทำให้จับวัตถุ/defect เล็กไม่เจอตอนใช้งานจริงในแอป
const DEFAULT_IMG_SIZE = 640;

// เลือกอุปกรณ์ตอนรัน (2026-11-20): เดิมบังคับ device = 0 → ลืมเปิด GPU บน Colab แล้วพัง และเลือก CPU ไม่ได้
// "#@param" ทำให้ Colab แสดงเป็น dropdown ให้เลือก (นอก Colab เป็นแค่คอมเมนต์) · auto = GPU ถ้ามี ไม่มีใช้ CPU
// ค่าเริ่มของ dropdown = อุปกรณ์ที่เลือกในหน้าต่าง Export Code (ค่าแปลก/ไม่ส่งมา = "auto")
const devParam = (device) => {
  const d = ["auto", "gpu", "cpu"].includes(device) ? device : "auto";
  return `# ── เลือกอุปกรณ์ที่ใช้เทรน ── "auto" = GPU ถ้ามี (ไม่มีใช้ CPU) · "gpu" = บังคับ GPU · "cpu" = บังคับ CPU (ช้ามาก)
DEVICE = "${d}"  #@param ["auto", "gpu", "cpu"]`;
};
const NO_GPU_MSG = "ไม่เจอ GPU — บน Colab ไปที่ Runtime > Change runtime type > T4 GPU แล้วรันใหม่";

function trainBlock({ epochs, batch, dataPath, imgsz, model, device }) {
  const sz = imgsz || DEFAULT_IMG_SIZE;
  const base = model || "yolo11n";
  const modelFile = base + ".pt";
  const data = JSON.stringify(dataPath || "/content/dataset/data.yaml");
  return `${devParam(device)}
import torch
if DEVICE == "gpu" and not torch.cuda.is_available():
    raise RuntimeError("${NO_GPU_MSG}")
device = "cpu" if DEVICE == "cpu" or not torch.cuda.is_available() else 0
print("เทรนบน:", "GPU " + torch.cuda.get_device_name(0) if device == 0 else "CPU")

model = YOLO("${modelFile}")
results = model.train(
    data= ${data},
    epochs= ${epochs},
    batch = ${batch},
    imgsz= ${sz},
    device = device
)`;
}

// แปลงโมเดลที่เทรนเสร็จเป็น .onnx (เอาไปใช้ในแอป FrameLab ได้เลย)
// imgsz ต้องตรงกับตอนเทรนเสมอ — ไม่งั้นแอปโหลด onnx แล้วจับวัตถุไม่เจอ
function exportBlock(imgsz) {
  const sz = imgsz || DEFAULT_IMG_SIZE;
  return `model.export(format="onnx", imgsz=${sz})`;
}

// ===== RF-DETR — ใช้ dataset.zip ได้ทั้งแบบ YOLO และ COCO (rfdetr>=1.8 ดูเอง: มี train/_annotations.coco.json = COCO) =====
// rfdetr ตรวจรูปแบบ YOLO จาก data.yaml + <split>/images + <split>/labels และอ่านชื่อคลาสจาก names: ใน data.yaml
// คลาสเป็น 0-indexed ตรงกับแอป (ไม่มี supercategory แบบ COCO ของ Roboflow มาเลื่อน index)
// ต้อง [train,loggers] ไม่งั้นขาด pytorch_lightning ตอน model.train()
const RFDETR_INSTALL_BLOCK = `!pip install "rfdetr[train,loggers]==1.8.2"`;
// rfdetr บังคับต้องมีโฟลเดอร์ valid/ (ไม่อ่าน path จาก data.yaml) — ถ้าผู้ใช้ไม่ได้แบ่ง valid
// zip จะมีแค่ train/ → rfdetr มองไม่ออกว่าเป็น YOLO แล้วไปหา COCO แทนจน error · เลยก็อป train มาเป็น valid
const RFDETR_UNZIP_BLOCK = `${UNZIP_BLOCK}
import os, shutil
if not os.path.isdir("/content/dataset/valid"):
    print("ไม่มี valid split — ใช้ภาพ train เป็น valid แทน (ควรแบ่ง valid ในแอปเพื่อวัดผลจริง)")
    shutil.copytree("/content/dataset/train", "/content/dataset/valid")`;

function rfdetrTrainBlock({ epochs, batch, dataPath, rfModel, numClasses, device }) {
  const dir = JSON.stringify(dataPath || "/content/dataset");
  const cls = rfModel || "RFDETRNano";
  const bs = batch || 4; // batch จริง (เล็กกัน VRAM บน T4) — ไม่ใช้ grad accum แล้ว
  // num_classes ต้องตรงกับจำนวนคลาสจริงของ dataset — ไม่งั้น onnx ที่ export ออกมาจะมีหัวคลาส COCO เต็ม (91)
  // แทนคลาสของผู้ใช้ ทำให้จับผิด/index เพี้ยน. ถ้าไม่รู้จำนวนคลาส (dialog เปิดได้อิสระ) ให้ปล่อย default ของ rfdetr ไป
  const nc = Number.isInteger(numClasses) && numClasses >= 1 ? numClasses : null;
  const ctor = nc ? `${cls}(num_classes=${nc})` : `${cls}()`;
  // RF-DETR เลือก cuda เองถ้ามี — บังคับ CPU ด้วยการซ่อน GPU ก่อน import torch (ไม่พึ่ง argument device ที่ต่างกันตามเวอร์ชัน)
  return `${devParam(device)}
# เปลี่ยนค่าแล้วต้อง Runtime > Restart session ก่อนรันใหม่ (torch จำ GPU ไว้ตั้งแต่ import ครั้งแรก)
import os
if DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
if DEVICE == "gpu" and not torch.cuda.is_available():
    raise RuntimeError("${NO_GPU_MSG}")
print("เทรนบน:", "GPU " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
from rfdetr import ${cls}

model = ${ctor}
model.train(
    dataset_dir= ${dir},
    epochs= ${epochs},
    batch_size= ${bs},
    lr= 1e-4,
)`;
}

const RFDETR_EXPORT_BLOCK = `!pip install onnx onnxruntime onnxsim
model.export()`;

// RF-DETR ไม่ฝังชื่อคลาสใน .onnx (YOLO ฝังให้) → เซลล์นี้ฝังเอง เป็น metadata "names" = {0: 'ชื่อ', …} รูปแบบเดียวกับ
// ultralytics — แอปอ่านได้ทันทีตอนโหลดโมเดล ไม่ต้องโหลดไฟล์ชื่อคลาสแยก · ชื่อมาจาก dataset ที่ใช้เทรน:
// COCO (train/_annotations.coco.json จากแอปหรือ Roboflow) ใช้กติกาเดียวกับ rfdetr (เรียงตาม id, ตัดหมวดแม่ที่ไม่มี annotation)
// YOLO ใช้ names ใน data.yaml · เทียบจำนวนกับช่องคลาสของโมเดลจริงอีกชั้น (ช่องสุดท้าย = "ไม่มีวัตถุ") กันชื่อเลื่อน
export const RFDETR_NAMES_BLOCK = `# ── ฝังชื่อคลาสลงไฟล์ .onnx (แอปอ่านชื่อเองตอนโหลดโมเดล) ──
import glob, json, os, onnx
DATASET = "/content/dataset"

def coco_class_names(path):
    d = json.load(open(path, encoding="utf-8"))
    cats = sorted(d.get("categories", []), key=lambda c: int(c["id"]))
    used = {int(a["category_id"]) for a in d.get("annotations", [])}
    sup = lambda c: c.get("supercategory", "none")
    parents = {sup(c) for c in cats if sup(c) not in ("", "none", "null", None) and sup(c) != c["name"]}
    kept = [c for c in cats if not (c["name"] in parents and int(c["id"]) not in used)] or cats
    return [c["name"] for c in kept], [c["name"] for c in cats]

def yolo_class_names(path):
    import yaml
    n = yaml.safe_load(open(path, encoding="utf-8")).get("names", [])
    n = [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
    return n, n

coco_file = os.path.join(DATASET, "train", "_annotations.coco.json")
names, all_names = coco_class_names(coco_file) if os.path.exists(coco_file) else yolo_class_names(os.path.join(DATASET, "data.yaml"))

onnx_path = sorted(glob.glob("output/**/*.onnx", recursive=True) + glob.glob("*.onnx"), key=os.path.getmtime)[-1]
m = onnx.load(onnx_path)
dims = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim] for o in m.graph.output}
logit = dims.get("labels") or next((v for v in dims.values() if len(v) == 3 and v[-1] not in (0, 4)), None)
slots = logit[-1] - 1 if logit and logit[-1] > 1 else None
if slots is not None and slots != len(names) and slots == len(all_names):
    names = all_names
if slots is not None and slots != len(names):
    print(f"⚠️ โมเดลมี {slots} คลาส แต่ dataset มี {len(names)} ชื่อ — ตรวจ num_classes ตอนเทรน")
    names = (names + [f"class_{i}" for i in range(len(names), slots)])[:slots]
for p in [p for p in m.metadata_props if p.key == "names"]:
    m.metadata_props.remove(p)
m.metadata_props.add(key="names", value=repr({i: n for i, n in enumerate(names)}))
onnx.save(m, onnx_path)
print("ฝังชื่อคลาสแล้ว:", onnx_path, names)
try:
    from google.colab import files
    files.download(onnx_path)
except ImportError:
    pass`;

// สร้างไฟล์ Jupyter Notebook (.ipynb) — เปิดใน Colab แล้ว Run ได้เลย
export function buildNotebook(opts) {
  const rf = opts.framework === "rfdetr";
  const cells = rf
    ? [
        codeCell(RFDETR_INSTALL_BLOCK),
        codeCell(RFDETR_UNZIP_BLOCK),
        codeCell(rfdetrTrainBlock(opts)),
        codeCell(RFDETR_EXPORT_BLOCK),
        codeCell(RFDETR_NAMES_BLOCK),
      ]
    : [
        codeCell(INSTALL_BLOCK),
        codeCell(IMPORTS_BLOCK),
        codeCell(UNZIP_BLOCK),
        codeCell(trainBlock(opts)),
        codeCell(exportBlock(opts.imgsz)),
      ];
  const nb = {
    cells,
    metadata: {
      kernelspec: { display_name: "Python 3", name: "python3" },
      language_info: { name: "python" },
    },
    nbformat: 4,
    nbformat_minor: 5,
  };
  return JSON.stringify(nb, null, 1);
}

// โค้ดรวม (สำหรับปุ่มคัดลอก / แสดงในหน้าต่าง)
export function buildTrainingCode(opts) {
  if (opts.framework === "rfdetr") {
    return `${RFDETR_INSTALL_BLOCK}\n\n${RFDETR_UNZIP_BLOCK}\n\n${rfdetrTrainBlock(opts)}\n${RFDETR_EXPORT_BLOCK}\n\n${RFDETR_NAMES_BLOCK}\n`;
  }
  // Colab ไม่มี ultralytics ติดมา — วางโค้ดนี้เซลล์เดียวต้องติดตั้งก่อน import (เหมือน notebook)
  return `${INSTALL_BLOCK}\n\n${IMPORTS_BLOCK}\n\n${UNZIP_BLOCK}\n\n${trainBlock(opts)}\n\n${exportBlock(opts.imgsz)}\n`;
}

// หน้าต่างตั้งค่าก่อน Export — ถามค่าเทรน YOLO แล้วส่งค่ากลับ
// เปิดด้วย open(onConfirm) : onConfirm({ epochs, batch, dataPath })
export function createExportDialog() {
  const BATCHES = [16, 32, 64];

  function open(onConfirm) {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";

    const modal = document.createElement("div");
    modal.className = "modal";
    overlay.appendChild(modal);

    const h = document.createElement("h2");
    h.className = "modal-title";
    h.textContent = "⚙️ ตั้งค่าเทรน YOLO";
    modal.appendChild(h);

    const sub = document.createElement("div");
    sub.className = "modal-sub";
    sub.textContent = "ตั้งค่าแล้วกดสร้างโค้ด — ได้โค้ดเทรน (.ipynb) สำหรับ Colab (เซฟภาพใช้ปุ่ม 💾 เซฟภาพ แยกต่างหาก)";
    modal.appendChild(sub);

    // เตือนเรื่อง labeling — ภาพเปล่าเทรนไม่ได้ ต้องตีกรอบ+ระบุคลาสก่อน
    const note = document.createElement("div");
    note.className = "modal-note";
    note.innerHTML =
      "⚠ ก่อนเทรน: ภาพ dataset ต้อง <b>ติดป้าย (label)</b> คือตีกรอบวัตถุ + ระบุชื่อคลาสก่อน " +
      "(ภาพเปล่ายังเทรนไม่ได้) — ใช้เครื่องมือฟรีเช่น Roboflow (roboflow.com) หรือ CVAT";
    modal.appendChild(note);

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
    minus.addEventListener("click", () => { epInput.value = String(Math.max(1, (Math.round(Number(epInput.value) || 1)) - 10)); });
    plus.addEventListener("click", () => { epInput.value = String((Math.round(Number(epInput.value) || 0)) + 10); });
    epInput.addEventListener("change", clampEp);
    stepper.appendChild(minus);
    stepper.appendChild(epInput);
    stepper.appendChild(plus);
    epField.appendChild(stepper);
    modal.appendChild(epField);

    // ===== Batch (การ์ดใหญ่ให้กดเลือก) =====
    const bField = document.createElement("div");
    bField.className = "setup-field";
    const bLabel = document.createElement("div");
    bLabel.className = "setup-label";
    bLabel.innerHTML = "<span>Batch size</span><span class='setup-hint'>จำนวนภาพต่อรอบ</span>";
    bField.appendChild(bLabel);

    const grid = document.createElement("div");
    grid.className = "batch-grid";
    let batch = 32;
    const cards = [];
    for (const b of BATCHES) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "batch-card" + (b === batch ? " active" : "");
      card.textContent = String(b);
      card.addEventListener("click", () => {
        batch = b;
        cards.forEach((c) => c.classList.toggle("active", Number(c.textContent) === batch));
      });
      cards.push(card);
      grid.appendChild(card);
    }
    bField.appendChild(grid);
    modal.appendChild(bField);

    // ===== ปุ่ม =====
    const actions = document.createElement("div");
    actions.className = "modal-actions";
    const cancel = document.createElement("button");
    cancel.className = "btn";
    cancel.textContent = "ยกเลิก";
    cancel.setAttribute("data-close", ""); // ESC ปิด (Design §22)
    const ok = document.createElement("button");
    ok.className = "btn btn-primary";
    ok.textContent = "⬇ Export";
    actions.appendChild(cancel);
    actions.appendChild(ok);
    modal.appendChild(actions);

    function close() {
      overlay.remove();
    }
    cancel.addEventListener("click", close);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) close();
    });
    ok.addEventListener("click", () => {
      clampEp();
      const epochs = Math.max(1, Math.round(Number(epInput.value) || 50));
      const opts = { epochs, batch, dataPath: "" }; // data path เว้นว่างเสมอ
      showCodeView(modal, opts, close);
      onConfirm(opts); // ให้ app จัดการ export ภาพ dataset ต่อ
    });

    document.body.appendChild(overlay);
    epInput.focus();
    epInput.select();
  }

  // มุมมองโค้ด (สไตล์ Roboflow): โชว์โค้ดให้คัดลอก + ปุ่มดาวน์โหลด
  function showCodeView(modal, opts, close) {
    const code = buildTrainingCode(opts);
    modal.innerHTML = "";

    const h = document.createElement("h2");
    h.className = "modal-title";
    h.textContent = "โค้ดเทรน YOLO";
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
      a.download = "train_yolo.ipynb";
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

function trainBlock({ epochs, batch, dataPath }) {
  return `model=YOLO("yolo12n.pt")
results = model.train(
    data= "${dataPath || ""}",
    epochs= ${epochs},
    batch = ${batch},
    imgsz= 640,
    optimizer =  "Adam",


    device = "0" #or "CPU"
)`;
}

// แปลงโมเดลที่เทรนเสร็จเป็น .onnx (เอาไปใช้ในแอป Machine Vision ได้เลย)
const EXPORT_BLOCK = `
model.export(format="onnx")`;

// สร้างไฟล์ Jupyter Notebook (.ipynb) — เปิดใน Colab แล้ว Run ได้เลย
// 3 cell แยก: ติดตั้ง / import / เทรน
export function buildNotebook(opts) {
  const nb = {
    cells: [
      codeCell(INSTALL_BLOCK),
      codeCell(IMPORTS_BLOCK),
      codeCell(trainBlock(opts)),
      codeCell(EXPORT_BLOCK),
    ],
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
  return `${IMPORTS_BLOCK}\n\n${trainBlock(opts)}\n\n${EXPORT_BLOCK}\n`;
}

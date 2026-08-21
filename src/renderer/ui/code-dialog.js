// หน้าต่างแสดงโค้ด Python ที่ gen จาก pipeline — คัดลอก / ดาวน์โหลด .py ได้
export function showCodeDialog(code, opts = {}) {
  const title = opts.title || "โค้ด Python (OpenCV)";
  const filename = opts.filename || "image_processing.py";

  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  const modal = document.createElement("div");
  modal.className = "modal modal-wide";
  overlay.appendChild(modal);

  const h = document.createElement("h2");
  h.className = "modal-title";
  h.textContent = title;
  modal.appendChild(h);

  const sub = document.createElement("div");
  sub.className = "modal-sub";
  sub.textContent = "แต่ละขั้นในโค้ด = แต่ละ Box ที่คุณต่อในกระดาน (node → โค้ดตัวเดียวกัน) — ลองเทียบดู · คัดลอกไปรัน หรือดาวน์โหลด .py ได้";
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
  dlBtn.textContent = "⬇ ดาวน์โหลด .py";
  dlBtn.addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([code], { type: "text/x-python" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  });

  const closeBtn = document.createElement("button");
  closeBtn.className = "btn";
  closeBtn.textContent = "ปิด";
  closeBtn.setAttribute("data-close", ""); // ESC ปิด (Design §22)
  const close = () => overlay.remove();
  closeBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });

  actions.appendChild(copyBtn);
  actions.appendChild(dlBtn);
  actions.appendChild(closeBtn);
  modal.appendChild(actions);

  document.body.appendChild(overlay);
}

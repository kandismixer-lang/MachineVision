// Template Gallery (§9)

// แกลเลอรีเทมเพลต — เลือกแล้ววาง pipeline สำเร็จรูป
// templates = [{ name, level, desc, steps, custom? }]  onPick(steps)  opts.onDelete(idx)->bool
export function showTemplates(templates, onPick, opts) {
  const overlay = el("div", "modal-overlay");
  const modal = el("div", "modal modal-wide");
  overlay.appendChild(modal);
  modal.appendChild(el("h2", "modal-title", "📋 เลือกเทมเพลต"));
  modal.appendChild(el("div", "modal-sub", "วาง pipeline สำเร็จรูปให้ทันที — หรือเลือกเทมเพลตที่คุณเซฟไว้"));

  const grid = el("div", "tmpl-grid");
  templates.forEach((t, idx) => {
    const card = el("div", "tmpl-card");
    card.innerHTML =
      `<div class="tmpl-name"></div><div class="tmpl-desc"></div>` +
      `<div class="tmpl-obj"><span class="tmpl-obj-ic">🎯</span><span class="tmpl-obj-t"></span></div>` +
      `<div class="tmpl-meta"><span class="tmpl-level"></span><span class="tmpl-time"></span></div>` +
      `<button class="btn btn-primary tmpl-go" type="button">เริ่มทดลอง →</button>`;
    card.querySelector(".tmpl-name").textContent = t.name;
    card.querySelector(".tmpl-desc").textContent = t.desc || "";
    const objT = card.querySelector(".tmpl-obj-t");
    if (t.objective) objT.textContent = t.objective; else card.querySelector(".tmpl-obj").style.display = "none";
    card.querySelector(".tmpl-level").textContent = "ระดับ: " + (t.level || "Beginner");
    card.querySelector(".tmpl-time").textContent = t.minutes ? `⏱ ~${t.minutes} นาที` : "";
    card.querySelector(".tmpl-go").addEventListener("click", () => { overlay.remove(); onPick(t.steps); });
    // เทมเพลตที่เซฟเอง → มีปุ่มลบ ✕
    if (t.custom && opts && opts.onDelete) {
      const del = el("button", "tmpl-del", "✕");
      del.title = "ลบเทมเพลตนี้";
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        if (opts.onDelete(t)) card.remove();
      });
      card.appendChild(del);
    }
    grid.appendChild(card);
  });
  modal.appendChild(grid);

  const actions = el("div", "modal-actions");
  const cancel = el("button", "btn", "ปิด");
  cancel.setAttribute("data-close", ""); // ESC ปิด (Design §22)
  cancel.addEventListener("click", () => overlay.remove());
  actions.appendChild(cancel);
  modal.appendChild(actions);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

  document.body.appendChild(overlay);
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

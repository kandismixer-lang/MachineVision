// Popup แจ้งเตือนเมื่อมีอัปเดต — เด้งอัตโนมัติตอน available/downloaded พร้อมโชว์ "มีอะไรใหม่"
// ใช้ bridge window.mvUpdate (จาก preload) — dev/browser ไม่มี → ไม่ทำอะไร
// - available  : เด้งบอกว่ามีเวอร์ชันใหม่ + กำลังดาวน์โหลด + release notes
// - downloaded : เปลี่ยนเป็นปุ่ม "ติดตั้งเลย" (เรียก mvUpdate.install)
// เด้งครั้งเดียวต่อเวอร์ชันต่อ session (กดภายหลังแล้วไม่รบกวนซ้ำ จนกว่าจะโหลดเสร็จ)

function notesToText(notes) {
  if (!notes) return "";
  let raw;
  if (Array.isArray(notes)) raw = notes.map((n) => (typeof n === "string" ? n : n?.note || "")).join("\n\n");
  else raw = String(notes);
  // release body จาก GitHub อาจเป็น HTML — ถอดแท็กเป็นข้อความล้วนด้วย DOMParser
  // (เอกสาร inert: ไม่รันสคริปต์ ไม่โหลด <img>/onerror — ต่างจาก innerHTML บน div)
  try {
    const doc = new DOMParser().parseFromString(raw, "text/html");
    return (doc.body.textContent || "").trim();
  } catch {
    return raw.replace(/<[^>]*>/g, "").trim();
  }
}

export function initUpdatePopup() {
  const u = window.mvUpdate;
  if (!u || typeof u.onStatus !== "function") return; // dev/เบราว์เซอร์

  let overlay = null;
  let shownVersion = null; // เวอร์ชันที่เด้งไปแล้ว (กันเด้งซ้ำตอน available ยิงหลายรอบ)
  let installed = false;

  function close() {
    overlay?.remove();
    overlay = null;
  }

  function render(version, notes, downloaded) {
    close();
    overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    const modal = document.createElement("div");
    modal.className = "modal update-modal";
    overlay.appendChild(modal);

    const head = document.createElement("div");
    head.className = "update-head";
    const h = document.createElement("div");
    h.className = "modal-title";
    h.textContent = downloaded ? "✅ อัปเดตพร้อมติดตั้ง" : "🎉 มีเวอร์ชันใหม่";
    const x = document.createElement("button");
    x.className = "modal-x";
    x.type = "button";
    x.textContent = "✕";
    x.addEventListener("click", close);
    head.append(h, x);
    modal.appendChild(head);

    const ver = document.createElement("div");
    ver.className = "update-version";
    ver.textContent = version ? `เวอร์ชัน ${version}` : "";
    modal.appendChild(ver);

    const sub = document.createElement("div");
    sub.className = "modal-sub";
    sub.textContent = downloaded
      ? "ดาวน์โหลดเสร็จแล้ว — กดติดตั้งเพื่ออัปเดต (แอปจะปิด-ติดตั้ง-เปิดใหม่ให้เอง)"
      : "กำลังดาวน์โหลดอัตโนมัติ… ติดตั้งได้เมื่อโหลดเสร็จ";
    modal.appendChild(sub);

    const notesText = notesToText(notes);
    const notesTitle = document.createElement("div");
    notesTitle.className = "update-notes-title";
    notesTitle.textContent = "มีอะไรใหม่";
    modal.appendChild(notesTitle);
    const box = document.createElement("div");
    box.className = "update-notes";
    box.textContent = notesText || "— ไม่มีรายละเอียด (ดูเพิ่มเติมได้ที่หน้า Releases บน GitHub) —";
    modal.appendChild(box);

    const foot = document.createElement("div");
    foot.className = "update-foot";
    const later = document.createElement("button");
    later.className = "btn";
    later.textContent = "ภายหลัง";
    later.addEventListener("click", close);
    const action = document.createElement("button");
    action.className = "btn btn-primary";
    if (downloaded) {
      action.textContent = "⟳ ติดตั้งเลย";
      action.addEventListener("click", async () => {
        if (installed) return;
        installed = true;
        action.disabled = true;
        action.textContent = "กำลังเปิดตัวติดตั้ง…";
        let res;
        try {
          res = await u.install?.();
        } catch (e) {
          res = { ok: false, error: e?.message || String(e) };
        }
        // reject/undefined ก็ถือว่าไม่สำเร็จ — reset ปุ่มให้กดใหม่ได้ ไม่ค้าง
        if (!res || res.ok === false) {
          installed = false;
          action.disabled = false;
          action.textContent = "⟳ ติดตั้งเลย";
          sub.textContent = `✕ ติดตั้งไม่สำเร็จ: ${res?.error || "unknown"}`;
        }
      });
    } else {
      action.textContent = "กำลังดาวน์โหลด…";
      action.disabled = true;
    }
    foot.append(later, action);
    modal.appendChild(foot);

    document.body.appendChild(overlay);
  }

  function handle(data) {
    if (!data) return;
    if (data.status === "available") {
      if (shownVersion === data.version) return; // เด้งไปแล้วสำหรับเวอร์ชันนี้
      shownVersion = data.version;
      render(data.version, data.releaseNotes, false);
    } else if (data.status === "downloaded") {
      // โหลดเสร็จ → เด้ง/อัปเดตเป็นปุ่มติดตั้ง (แม้ผู้ใช้กดภายหลังไปแล้วก็เด้งอีกครั้งตอนพร้อมติดตั้ง)
      shownVersion = data.version;
      render(data.version, data.releaseNotes, true);
    }
  }

  u.onStatus(handle);
  // เผื่อ event downloaded ยิงก่อน popup ผูก listener (ยิงครั้งเดียว) — ดึงสถานะปัจจุบันมาเช็ก
  u.getState?.().then((s) => {
    if (s?.downloaded) handle({ status: "downloaded", version: s.version, releaseNotes: s.releaseNotes });
  }).catch(() => {});
}

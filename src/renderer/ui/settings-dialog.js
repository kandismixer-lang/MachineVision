// หน้าตั้งค่าคีย์ลัด (keyboard shortcut) — กดปุ่ม "อัดปุ่ม" แล้วกดคีย์ที่ต้องการ
// actions = [{ id, label, hint }]  keymap = { id: keyString }
// onChange(newKeymap) เรียกทุกครั้งที่แก้ (บันทึกทันที)

// แปลง key ดิบ (e.key) → ข้อความอ่านง่าย
export function keyLabel(k) {
  if (k == null || k === "") return "—";
  const map = {
    " ": "Space",
    ArrowLeft: "←",
    ArrowRight: "→",
    ArrowUp: "↑",
    ArrowDown: "↓",
    Escape: "Esc",
    Enter: "Enter",
    Backspace: "Backspace",
  };
  if (map[k]) return map[k];
  return k.length === 1 ? k.toUpperCase() : k;
}

export function openSettings(actions, keymap, onChange) {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  const modal = document.createElement("div");
  modal.className = "modal";
  overlay.appendChild(modal);

  const h = document.createElement("h2");
  h.className = "modal-title";
  h.textContent = "⚙️ ตั้งค่าคีย์ลัด";
  modal.appendChild(h);

  const sub = document.createElement("div");
  sub.className = "modal-sub";
  sub.textContent = "คลิกที่ช่องคีย์ทางขวาของแต่ละรายการ แล้วกดปุ่มบนคีย์บอร์ดที่อยากใช้ · คีย์ลัดทำงานเมื่อไม่ได้กำลังพิมพ์ในช่องข้อความ";
  modal.appendChild(sub);

  let recording = null; // id ที่กำลังอัดอยู่

  const list = document.createElement("div");
  list.className = "keymap-list";
  modal.appendChild(list);

  function render() {
    list.innerHTML = "";
    for (const a of actions) {
      const row = document.createElement("div");
      row.className = "keymap-row";

      const info = document.createElement("div");
      info.className = "keymap-info";
      const lab = document.createElement("div");
      lab.className = "keymap-label";
      lab.textContent = a.label;
      info.appendChild(lab);
      if (a.hint) {
        const hint = document.createElement("div");
        hint.className = "keymap-hint";
        hint.textContent = a.hint;
        info.appendChild(hint);
      }
      row.appendChild(info);

      const isRec = recording === a.id;
      const keyBtn = document.createElement("button");
      keyBtn.className = "keycap-btn" + (isRec ? " recording" : "");
      keyBtn.textContent = isRec ? "กดปุ่ม…" : keyLabel(keymap[a.id]);
      keyBtn.title = "คลิกเพื่ออัดปุ่มใหม่";
      keyBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        recording = isRec ? null : a.id;
        render();
      });
      row.appendChild(keyBtn);

      const clr = document.createElement("button");
      clr.className = "keycap-clear";
      clr.textContent = "✕";
      clr.title = "ล้างคีย์ลัดนี้";
      clr.addEventListener("click", (e) => {
        e.stopPropagation();
        keymap[a.id] = "";
        recording = null;
        onChange({ ...keymap });
        render();
      });
      row.appendChild(clr);

      list.appendChild(row);
    }
  }
  render();

  // ระหว่างเปิดหน้านี้ ดักคีย์เพื่ออัด (capture phase กันไปชนคีย์ลัดจริง)
  function onKey(e) {
    if (!recording) return;
    if (e.key === "Escape") {
      recording = null;
      render();
      e.preventDefault();
      e.stopPropagation();
      return;
    }
    // กันคีย์ modifier ล้วน ๆ
    if (["Shift", "Control", "Alt", "Meta"].includes(e.key)) return;
    // กันคีย์ซ้ำกับ action อื่น → เคลียร์ของเดิมก่อน
    for (const id in keymap) {
      if (id !== recording && keymap[id] && keymap[id].toLowerCase() === e.key.toLowerCase()) {
        keymap[id] = "";
      }
    }
    keymap[recording] = e.key;
    recording = null;
    onChange({ ...keymap });
    render();
    e.preventDefault();
    e.stopPropagation();
  }
  document.addEventListener("keydown", onKey, true);

  // ปุ่มเช็คอัปเดตตอนนี้ — โชว์ผลจริงเสมอ (ไม่เงียบเหมือน auto-check ตอนเปิดแอป)
  // ใช้ได้เฉพาะแอปที่ build จริงผ่าน Electron (mvUpdate bridge จาก preload) ไม่ใช่ dev/browser
  let unsubUpdate = null;
  let clearUpdateTimeout = null;
  if (window.mvUpdate?.available) {
    const updateBox = document.createElement("div");
    updateBox.className = "update-check-box";
    const updateRow = document.createElement("div");
    updateRow.className = "update-check-row";
    const checkBtn = document.createElement("button");
    checkBtn.className = "btn";
    checkBtn.textContent = "🔄 เช็คอัปเดต";
    const statusText = document.createElement("span");
    statusText.className = "update-check-status";
    statusText.textContent = "";
    updateRow.append(checkBtn, statusText);
    updateBox.appendChild(updateRow);
    modal.appendChild(updateBox);

    // กัน "กำลังเช็ค…" ค้างตลอดไป ถ้า autoUpdater ไม่ยิง event กลับมาเลย (เช่นเช็คไม่ได้เงียบ ๆ
    // ตอนรันแบบ unpacked/dev หรือ edge case อื่น) — ตั้งเวลาสำรอง ถ้าไม่มี event ใน 15 วิ ถือว่า timeout
    let timeoutId = null;
    function clearPendingTimeout() {
      if (timeoutId) { clearTimeout(timeoutId); timeoutId = null; }
    }
    function armTimeout() {
      clearPendingTimeout();
      timeoutId = setTimeout(() => {
        statusText.textContent = "เช็คไม่สำเร็จ: หมดเวลา (ไม่ได้รับผลจากระบบอัปเดต)";
        checkBtn.disabled = false;
      }, 15000);
    }
    clearUpdateTimeout = clearPendingTimeout;

    unsubUpdate = window.mvUpdate.onStatus((data) => {
      clearPendingTimeout();
      if (data.status === "checking") { statusText.textContent = "กำลังเช็ค…"; armTimeout(); }
      else if (data.status === "available") statusText.textContent = `พบเวอร์ชันใหม่ (v${data.version}) — กำลังโหลด…`;
      else if (data.status === "downloaded") statusText.textContent = `โหลดเสร็จแล้ว (v${data.version}) — ปิด-เปิดแอปใหม่เพื่อติดตั้ง`;
      else if (data.status === "not-available") statusText.textContent = "เป็นเวอร์ชันล่าสุดอยู่แล้ว ✓";
      else if (data.status === "error") statusText.textContent = "เช็คไม่สำเร็จ: " + data.message;
      // ระหว่าง "available" (กำลังโหลดไฟล์อัปเดต) ก็ต้อง disable ด้วย ไม่ใช่แค่ "checking" —
      // ไม่งั้นกดซ้ำระหว่างโหลดจะไปตั้ง timeout ใหม่ทับข้อความ "กำลังโหลด…" กลายเป็น "หมดเวลา" หลอก
      checkBtn.disabled = data.status === "checking" || data.status === "available";
    });

    checkBtn.addEventListener("click", async () => {
      statusText.textContent = "กำลังเช็ค…";
      checkBtn.disabled = true;
      armTimeout();
      const res = await window.mvUpdate.check();
      if (!res.ok) {
        clearPendingTimeout();
        statusText.textContent = "เช็คไม่สำเร็จ: " + res.error;
        checkBtn.disabled = false;
      }
      // ถ้า ok=true ผลจริงจะตามมาทาง onStatus event (checking/available/not-available/error)
      // หรือถ้าไม่มา event เลย armTimeout() ที่ตั้งไว้ข้างบนจะ fallback ให้เอง
    });
  }

  const actionsBar = document.createElement("div");
  actionsBar.className = "modal-actions";
  const closeBtn = document.createElement("button");
  closeBtn.className = "btn btn-primary";
  closeBtn.textContent = "เสร็จ";
  closeBtn.setAttribute("data-close", ""); // ESC ปิดได้เหมือน modal อื่น
  function close() {
    document.removeEventListener("keydown", onKey, true);
    if (unsubUpdate) unsubUpdate();
    if (clearUpdateTimeout) clearUpdateTimeout();
    overlay.remove();
  }
  closeBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });
  actionsBar.appendChild(closeBtn);
  modal.appendChild(actionsBar);

  document.body.appendChild(overlay);
}

// อัปโหลดวิดีโอ → แยกเป็นเฟรมภาพนิ่ง (แบบ Roboflow)
// 2 สเต็ป: (1) เลือกความถี่/จำนวน  (2) แยกแล้วโชว์ grid ตัวอย่าง เลื่อนดู → ยืนยันใช้
// open(file, onDone) : onDone(frames|null)  frames = [{ canvas, w, h, name }]

const DEFAULT_INTERVAL = 1;
const MIN_INTERVAL = 1 / 60; // ถี่สุด ~60 fps (0.0167 วิ)
const HARD_MAX = 1000; // เพดานเฟรม (safety) — ย่อภาพช่วยกัน OOM; เตือนเมื่อเยอะมาก
const WARN_FRAMES = 500; // เกินนี้เตือนเรื่องหน่วยความจำ
const MAX_SIDE = 1280; // ย่อด้านยาวกัน OOM

export function openVideoFrames(file, onDone) {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  const modal = document.createElement("div");
  modal.className = "modal modal-wide";
  overlay.appendChild(modal);

  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.preload = "metadata";
  video.muted = true;
  video.src = url;

  let interval = DEFAULT_INTERVAL;
  let maxFrames = HARD_MAX;
  let cancelled = false;
  let extracting = false;
  let finished = false;

  function done(frames) {
    if (finished) return; // กันเรียกซ้ำ (ดับเบิลคลิก / cancel+overlay)
    finished = true;
    URL.revokeObjectURL(url);
    overlay.remove();
    onDone(frames);
  }

  // ================= สเต็ป 1: ตั้งค่า =================
  function renderSettings() {
    modal.innerHTML = "";
    cancelled = false;
    extracting = false;

    const durTxt = (video.duration && isFinite(video.duration)) ? video.duration.toFixed(1) : "…";
    modal.append(
      el("h2", "modal-title", `🎬 แยกเฟรมจาก ${file.name} (${durTxt} วิ)`),
      el("div", "modal-sub", "เลื่อนเลือกความถี่ (หรือพิมพ์ตัวเลข) แล้วกด “แสดงตัวอย่าง” เพื่อดูเฟรมที่จะได้")
    );

    // ===== การ์ดตั้งค่า sample (สไตล์ Roboflow) =====
    const card = el("div", "vf-sample-card");
    const rateRow = el("div", "vf-rate-row", "", `<span>ดึงเฟรม</span><span class="vf-rate"></span>`);
    const rateEl = rateRow.querySelector(".vf-rate");
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = "1000"; slider.step = "1"; slider.className = "vf-slider";
    const ends = el("div", "vf-ends", "", `<span>ถี่สุด ~60 เฟรม/วิ</span><span>1 เฟรม/คลิป</span>`);
    const divider = el("div", "vf-divider");
    const output = el("div", "vf-output");
    const ivInput = document.createElement("input");
    ivInput.type = "number"; ivInput.step = "0.001"; ivInput.className = "vf-num-input";
    const countInput = document.createElement("input");
    countInput.type = "number"; countInput.min = "1"; countInput.className = "vf-num-input";
    output.append(document.createTextNode("ดึง 1 เฟรมทุก ๆ "), ivInput, document.createTextNode(" วิ  ≈ "), countInput, document.createTextNode(" ภาพ"));
    const warn = el("div", "vf-warn");
    card.append(rateRow, slider, ends, divider, output, warn);
    modal.append(card);

    // progress + ปุ่ม
    const prog = el("div", "vf-progress hidden");
    const bar = el("div", "vf-bar");
    prog.append(bar);
    const progText = el("div", "vf-progress-text hidden");
    const actions = el("div", "modal-actions");
    const cancel = el("button", "btn", "ยกเลิก");
    cancel.setAttribute("data-close", ""); // ESC ยกเลิก/ปิด (Design §22)
    const ok = el("button", "btn btn-primary", "🔍 แสดงตัวอย่าง");
    actions.append(cancel, ok);
    modal.append(prog, progText, actions);
    cancel.addEventListener("click", () => { cancelled = true; done(null); });

    // ---- log mapping ระหว่าง slider(0..1000) กับ interval ----
    const maxIv = () => Math.max(MIN_INTERVAL * 2, (isFinite(video.duration) && video.duration) || 1);
    const posOf = (iv) => Math.round(1000 * Math.log(iv / MIN_INTERVAL) / Math.log(maxIv() / MIN_INTERVAL));
    const ivOf = (pos) => MIN_INTERVAL * Math.pow(maxIv() / MIN_INTERVAL, pos / 1000);
    const estN = (iv) => {
      const dur = video.duration || 0;
      return Math.min(Math.floor(dur / iv) + 1, HARD_MAX, Math.floor(dur * 60) + 1);
    };

    let ready = false;
    function sync() {
      if (!ready) return;
      interval = Math.max(MIN_INTERVAL, Math.min(interval, maxIv()));
      const n = estN(interval);
      slider.value = String(posOf(interval));
      if (document.activeElement !== ivInput) ivInput.value = interval.toFixed(3);
      if (document.activeElement !== countInput) countInput.value = String(n);
      rateEl.textContent = `ทุก ${interval.toFixed(3)} วิ`;
      ok.textContent = `🔍 แสดงตัวอย่าง ${n} เฟรม`;
      warn.textContent = n > WARN_FRAMES ? "⚠ เฟรมเยอะ อาจใช้หน่วยความจำสูง/ช้า" : "";
    }
    slider.addEventListener("input", () => { interval = ivOf(Number(slider.value)); sync(); });
    ivInput.addEventListener("input", () => { const v = Number(ivInput.value); if (v > 0) { interval = v; sync(); } });
    countInput.addEventListener("input", () => {
      const n = Math.max(1, Math.min(HARD_MAX, Math.round(Number(countInput.value) || 1)));
      if (video.duration) { interval = video.duration / n; sync(); }
    });

    function onReady() {
      ready = true;
      // ตั้งค่าเริ่ม: DEFAULT_INTERVAL แต่ไม่เกินความยาวคลิป
      interval = Math.max(MIN_INTERVAL, Math.min(DEFAULT_INTERVAL, maxIv()));
      // อัปเดตหัวข้อให้มี duration จริง
      modal.querySelector(".modal-title").textContent = `🎬 แยกเฟรมจาก ${file.name} (${video.duration.toFixed(1)} วิ)`;
      sync();
    }
    const durOk = () => video.duration && isFinite(video.duration);
    if (video.readyState >= 1 && durOk()) onReady();
    else {
      video.onloadedmetadata = () => { if (durOk()) onReady(); else { warn.textContent = "อ่านความยาววิดีโอไม่ได้ — ไฟล์อาจไม่รองรับ"; ok.disabled = true; } };
      video.onerror = () => { warn.textContent = "เปิดวิดีโอไม่สำเร็จ — ไฟล์อาจไม่รองรับ"; ok.disabled = true; };
    }

    ok.addEventListener("click", async () => {
      if (!ready || !video.videoWidth || !video.videoHeight) { warn.textContent = "อ่านข้อมูลวิดีโอไม่ได้ — ไฟล์อาจไม่รองรับ"; return; }
      maxFrames = HARD_MAX;
      ok.disabled = true; cancel.textContent = "หยุด"; slider.disabled = true; ivInput.disabled = true; countInput.disabled = true;
      prog.classList.remove("hidden"); progText.classList.remove("hidden");
      extracting = true;
      const frames = await extract((i, total) => {
        bar.style.width = Math.round((i / total) * 100) + "%";
        progText.textContent = `กำลังแยกเฟรม ${i}/${total} …`;
      });
      extracting = false;
      if (cancelled) return;
      if (!frames || frames.length === 0) { warn.textContent = "แยกเฟรมไม่ได้ — ลองไฟล์อื่นหรือลดจำนวน"; ok.disabled = false; cancel.textContent = "ยกเลิก"; slider.disabled = false; ivInput.disabled = false; countInput.disabled = false; return; }
      renderPreview(frames);
    });
  }

  // ================= สเต็ป 2: preview grid =================
  function renderPreview(frames) {
    modal.innerHTML = "";
    modal.append(el("h2", "modal-title", `🖼 ตัวอย่างเฟรม — ได้ ${frames.length} เฟรม`));
    modal.append(el("div", "modal-sub", "เลื่อนดูเฟรมที่จะได้ · พอใจแล้วกด “ใช้เฟรมเหล่านี้” หรือกลับไปปรับความถี่"));

    const grid = el("div", "vf-grid");
    frames.forEach((f, i) => {
      const cell = el("div", "vf-cell");
      // thumbnail แคนวาสเล็ก (เลี่ยง toDataURL ที่ encode PNG แบบ blocking ต่อทุกเฟรม)
      const thumb = document.createElement("canvas");
      const tw = 160, th = Math.max(1, Math.round((f.h / f.w) * tw));
      thumb.width = tw; thumb.height = th;
      const tctx = thumb.getContext("2d");
      if (tctx) tctx.drawImage(f.canvas, 0, 0, tw, th); // เฟรมเยอะมากอาจชน canvas limit → ข้าม thumbnail นี้
      cell.append(thumb, el("span", "vf-idx", String(i + 1)));
      grid.append(cell);
    });
    modal.append(grid);

    const actions = el("div", "modal-actions");
    const back = el("button", "btn", "← ปรับความถี่ใหม่");
    const use = el("button", "btn btn-primary", `✓ ใช้ ${frames.length} เฟรมนี้`);
    actions.append(back, use);
    modal.append(actions);

    back.addEventListener("click", () => renderSettings());
    use.addEventListener("click", () => done(frames));
  }

  // แยกเฟรมจริง — คืน array (หรือ null ถ้าล้มเหลว) ; onProgress(i,total)
  async function extract(onProgress) {
    const scale = Math.min(1, MAX_SIDE / Math.max(video.videoWidth, video.videoHeight));
    const fw = Math.max(1, Math.round(video.videoWidth * scale));
    const fh = Math.max(1, Math.round(video.videoHeight * scale));
    const base = (file.name.replace(/\.[^.]+$/, "") || "video").replace(/[\\/:*?"<>|]+/g, "_");
    const times = [];
    for (let t = 0; t < video.duration && times.length < maxFrames; t += interval) times.push(t);
    const width = String(times.length).length;
    const frames = [];
    try {
      for (let i = 0; i < times.length; i++) {
        if (cancelled) return null;
        await seekTo(video, times[i]);
        if (cancelled) return null;
        const c = document.createElement("canvas");
        c.width = fw; c.height = fh;
        const ctx = c.getContext("2d");
        if (!ctx) throw new Error("canvas เต็ม");
        ctx.drawImage(video, 0, 0, fw, fh);
        frames.push({ canvas: c, w: fw, h: fh, name: `${base}_f${String(i + 1).padStart(width, "0")}.png` });
        onProgress(i + 1, times.length);
      }
    } catch {
      return frames.length ? frames : null;
    }
    return frames;
  }

  overlay.addEventListener("click", (e) => { if (e.target === overlay && !extracting) { cancelled = true; done(null); } });
  document.body.appendChild(overlay);
  renderSettings();
}

// helper สร้าง element สั้น ๆ
function el(tag, cls, text, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  else if (text != null) e.textContent = text;
  return e;
}

// seek ไปเวลาที่ต้องการแล้วรอเฟรมพร้อม — กันแขวน (นัดจค่าเท่าเดิม + timeout + error)
function seekTo(video, t) {
  return new Promise((resolve, reject) => {
    const dur = video.duration;
    let target = Math.min(t, Math.max(0, dur - 0.001));
    if (Math.abs(video.currentTime - target) < 1e-3) target = Math.min(target + 0.033, Math.max(0, dur - 0.001));
    let d = false;
    const cleanup = () => { clearTimeout(timer); video.removeEventListener("seeked", onSeeked); video.removeEventListener("error", onErr); };
    const onSeeked = () => { if (d) return; d = true; cleanup(); resolve(); };
    const onErr = () => { if (d) return; d = true; cleanup(); reject(new Error("seek error")); };
    const timer = setTimeout(() => { if (d) return; d = true; cleanup(); resolve(); }, 4000);
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onErr);
    video.currentTime = target;
  });
}

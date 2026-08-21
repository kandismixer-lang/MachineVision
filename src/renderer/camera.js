// จัดการกล้อง webcam ด้วย getUserMedia — รองรับหลายกล้อง (เลือก device ได้)

// หยุด stream เดิมของ video (ปล่อยกล้องตัวก่อนก่อนสลับ)
function stopStream(videoEl) {
  const s = videoEl.srcObject;
  if (s && s.getTracks) s.getTracks().forEach((t) => t.stop());
  videoEl.srcObject = null;
}

// นับรุ่นการเปิดกล้อง — กันเรียกซ้อน (สลับ Port รัว ๆ) แล้ว track ของรุ่นเก่าค้าง/ภาพผิดตัว
let camGen = 0;

// เริ่มกล้องและผูกกับ <video>; ระบุ deviceId เพื่อเลือกกล้องตัวใดตัวหนึ่ง
// คืน stream เมื่อพร้อม / โยน error ภาษาไทยถ้าเปิดไม่ได้
export async function startCamera(videoEl, deviceId) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("เบราว์เซอร์/แอปนี้ไม่รองรับการเข้าถึงกล้อง");
  }

  const video = deviceId
    ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
    : { width: { ideal: 1280 }, height: { ideal: 720 } };

  const gen = ++camGen; // รุ่นล่าสุด
  stopStream(videoEl); // ปล่อยกล้องเดิมก่อน

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  } catch (err) {
    if (gen !== camGen) return null; // รุ่นนี้ถูกแซงแล้ว → กลืน error ไม่ให้เด้งทับรุ่นล่าสุด
    if (err.name === "NotAllowedError") {
      throw new Error("ไม่ได้รับอนุญาตให้ใช้กล้อง — โปรดอนุญาตสิทธิ์กล้องแล้วเปิดแอปใหม่");
    }
    if (err.name === "NotFoundError") {
      throw new Error("ไม่พบกล้อง — โปรดต่อกล้อง webcam แล้วลองใหม่");
    }
    throw new Error("เปิดกล้องไม่สำเร็จ: " + err.message);
  }

  // มีการเรียก startCamera ใหม่แซงระหว่างรอ getUserMedia → ทิ้งสตรีมนี้ (กัน track รั่ว/ภาพผิดตัว)
  if (gen !== camGen) {
    stream.getTracks().forEach((t) => t.stop());
    return null;
  }

  videoEl.srcObject = stream;
  await videoEl.play();

  if (videoEl.videoWidth === 0) {
    await new Promise((resolve) => {
      videoEl.addEventListener("loadedmetadata", resolve, { once: true });
    });
  }

  return stream;
}

// เดาว่าเป็นกล้องในตัวโน้ตบุ๊ก/เครื่อง (ให้เป็น Port 1 เสมอ)
function isBuiltIn(label) {
  return /integrated|built.?in|internal|facetime|hd\s?camera|hd\s?webcam/i.test(label || "");
}

// รายชื่อกล้องที่ต่ออยู่ (label จะมีค่าหลังได้รับสิทธิ์กล้องแล้ว)
// คืน [{ deviceId, label, builtIn }] — เรียงกล้องในตัวเครื่องขึ้นก่อนเสมอ (Port 1)
export async function listCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices
    .filter((d) => d.kind === "videoinput")
    .map((d) => ({ deviceId: d.deviceId, label: d.label || "กล้อง", builtIn: isBuiltIn(d.label) }));
  // กล้องในตัวเครื่องขึ้นก่อน (stable sort คงลำดับเดิมในกลุ่มเดียวกัน)
  cams.sort((a, b) => (b.builtIn ? 1 : 0) - (a.builtIn ? 1 : 0));
  return cams;
}

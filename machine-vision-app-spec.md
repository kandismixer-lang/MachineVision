# สเปกโปรเจกต์: Machine Vision Desktop App สำหรับการเรียนการสอน

## 1. วัตถุประสงค์

พัฒนา Desktop application สำหรับสอนแนวคิด Machine Vision เบื้องต้น โดยให้ผู้เรียนสามารถ:
- ดูภาพจากกล้อง webcam แบบ real-time
- สลับโหมดการแสดงผลภาพ (ขาวดำ, RGB, HSV, HSL)
- ต่อ filter หลายตัวเป็นลำดับ (pipeline) แบบ block-based เพื่อเห็นผลลัพธ์แต่ละขั้นตอน

## 2. เทคโนโลยีหลัก

| องค์ประกอบ | เทคโนโลยีที่ใช้ | หมายเหตุ |
|---|---|---|
| Desktop shell | Electron | ครอบ Chromium + Node.js runtime |
| ประมวลผลภาพ | OpenCV.js (WebAssembly build) | ฟังก์ชันหน้าตาเหมือน Python OpenCV แต่รันเป็น JavaScript ล้วน |
| รับภาพจากกล้อง | Web API `navigator.mediaDevices.getUserMedia` | ทำงานได้ตรงในตัว Electron renderer ไม่ต้องพึ่ง library เพิ่ม |
| Block-based filter editor | LiteGraph.js ✅ (ตัดสินใจแล้ว 2026-07-16) | Visual node graph สไตล์ Node-RED: ลาก node ต่อสาย [แหล่งภาพ] → filter → [ผลลัพธ์] — ผู้ใช้เลือกแนวทางนี้เพราะเหมาะกับการเรียนการสอน เห็นการไหลของภาพชัดเจน |

**ไม่ใช้ Python** ในเวอร์ชันนี้ — ทุกอย่างรันด้วย JavaScript ทั้ง main process และ renderer process ของ Electron เพื่อให้แจกจ่ายแอปให้ผู้เรียนได้ง่าย ไม่ต้องติดตั้ง Python หรือ dependency เพิ่มเติมบนเครื่องผู้เรียน

## 3. โครงสร้างระบบ (Architecture)

```
Electron App
├── Main process (JS/Node)
│   └── จัดการหน้าต่าง, เมนู, IPC
└── Renderer process (Chromium)
    ├── Webcam (getUserMedia) → capture frame
    ├── OpenCV.js (wasm) → แปลงสี / ใช้ filter
    ├── Node-graph filter editor → กำหนดลำดับ pipeline
    └── Canvas → แสดงผลลัพธ์ / preview
```

การทำงาน: กล้อง → capture frame → ส่งเข้า pipeline ที่ผู้เรียนต่อ block ไว้ (เช่น Grayscale → Gaussian Blur → Threshold) → OpenCV.js ประมวลผลตามลำดับ node → วาดผลลัพธ์ลง Canvas ทุกเฟรม

## 4. ขอบเขตการพัฒนา แบ่งเป็น Phase

### Phase 1 — Core video pipeline
- [ ] Scaffold Electron app พื้นฐาน
- [ ] โหลด OpenCV.js เข้ามาใช้งานในเครื่อง (bundle ไว้ ไม่พึ่ง CDN ตอน production)
- [ ] เปิดกล้อง webcam แสดงผลสด
- [ ] Mode selector พื้นฐาน: Grayscale / RGB / HSV / HSL (ยังไม่มี block chain)

### Phase 2 — Filter chain แบบ block
- [ ] เพิ่ม node-graph editor (LiteGraph.js)
- [ ] สร้าง filter block ชุดแรก แบ่งเป็นกลุ่มดังนี้ (ดูรายละเอียดในหัวข้อ 4.1)
- [ ] Executor ไล่ประมวลผลตามลำดับการต่อ node ทุกเฟรม
- [ ] รองรับการผสมผสาน block ข้ามกลุ่มกันได้อิสระ (เช่น Blur → Sharpen → Morphological → Threshold)

#### 4.1 รายการ Filter Block แบ่งตามกลุ่ม

> **อัปเดต 2026-07-16**: ปรับการจัดกลุ่ม filter ให้ตรงกับหัวข้อการเรียนการสอน 6 เรื่อง (ตามคำขอ):
> 1. **การปรับปริภูมิสี** — Grayscale, แปลง HSV/HLS/Lab/YCrCb, แยกดูช่องสี
> 2. **Image Enhancement** — Brightness/Contrast, Gamma, Histogram Equalization, CLAHE, Unsharp, Laplacian Sharpen
> 3. **Image Restoration** — ใส่ noise เกลือ-พริกไทยเป็นโจทย์ แล้วฟื้นฟูด้วย Median/Gaussian/Box/Bilateral
> 4. **Morphological Processing** — Erode/Dilate/Open/Close/Gradient/TopHat/BlackHat (เลือก kernel ได้)
> 5. **Image Segmentation** — Threshold (Binary/Otsu/Adaptive), Canny, Sobel, Find Contours + นับวัตถุ
> 6. **Colour Image Processing** — เลือกช่วงสี HSV (inRange), Invert
>
> ทั้งหมด implement แล้วเป็น node ในกระดาน (23 nodes) — รายการกลุ่มด้านล่างนี้คือแผนเดิม เก็บไว้อ้างอิง

**กลุ่ม Color / Color space**
- Grayscale
- Color convert: RGB, HSV, HSL
- Channel split (แยกดู R, G, B หรือ H, S, V แต่ละ channel)

**กลุ่ม Blur (Smoothing)**
- Gaussian Blur
- Average / Box Blur
- Median Blur
- Bilateral Filter (blur แบบรักษาขอบภาพ)

**กลุ่ม Sharpening**
- Sharpen (unsharp mask / kernel convolution)
- Laplacian sharpening

**กลุ่ม Morphological Operations**
- Erosion
- Dilation
- Opening (Erosion ตามด้วย Dilation)
- Closing (Dilation ตามด้วย Erosion)
- Morphological Gradient
- Top Hat / Black Hat
- (แต่ละตัวต้องเลือก kernel shape/size ได้ เช่น rect, ellipse, cross)

**กลุ่ม Edge Detection**
- Sobel
- Canny
- Laplacian

**กลุ่ม Thresholding**
- Binary Threshold
- Adaptive Threshold

### Phase 3 — Polish สำหรับใช้สอนจริง
- [ ] บันทึก/โหลด pipeline เป็นไฟล์ (เก็บโครงสร้างกราฟเป็น JSON)
- [ ] แสดงค่า FPS เพื่อให้เห็นผลกระทบของแต่ละ filter ต่อความเร็ว
- [ ] (ถ้าจำเป็น) ย้ายการประมวลผลไป Web Worker เพื่อไม่ให้ UI กระตุกเมื่อ pipeline ซับซ้อน

## 5. สิ่งที่ยังไม่รวมในเวอร์ชันแรก (Non-goals)

- ไม่รวมโมเดล deep learning (เช่น YOLO) ในเวอร์ชันแรก — จะพิจารณาเพิ่มเป็น advanced block ในอนาคตหากต้องการ
- ไม่ผูกกับ Python backend ใดๆ ในสถาปัตยกรรมหลัก

## 6. คำถามที่ต้องตกลงร่วมกันก่อนเริ่มเขียนโค้ด

1. จากรายการ filter ในหัวข้อ 4.1 มีตัวไหนที่อยากให้ทำก่อนใน Phase 2 รอบแรก และตัวไหนเลื่อนไปรอบหลังได้?
2. ต้องการบันทึกภาพ/วิดีโอผลลัพธ์ออกเป็นไฟล์ไหม (export)?
3. เป้าหมายผู้ใช้งานคือผู้เรียนระดับไหน (ต้องออกแบบ UI ให้ซับซ้อนแค่ไหน)?

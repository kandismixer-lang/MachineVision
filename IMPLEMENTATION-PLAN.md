# แผนพัฒนา (Implementation Plan) — Machine Vision Desktop App

> อ้างอิงจาก `machine-vision-app-spec.md`
> ข้อตกลงรอบนี้: วางโครงสร้าง + แผนก่อน (ยังไม่ลงโค้ด) · ทำ filter ครบทุกกลุ่ม · กลุ่มเป้าหมาย = ผู้เรียนมือใหม่มาก (UI ต้องเรียบง่าย)

---

## 1. หลักการออกแบบสำหรับ "มือใหม่มาก"

- เปิดแอปแล้วเห็นภาพจากกล้องทันที ไม่ต้องตั้งค่าอะไรก่อน
- ปุ่ม/โหมดพื้นฐาน (Grayscale / RGB / HSV / HSL) เข้าถึงได้ในคลิกเดียว โดยไม่ต้องยุ่งกับ node-graph
- node-graph (Phase 2) เป็นโหมดขั้นสูงที่ "เลือกเปิด" ได้ ไม่บังคับ
- ทุก filter block มีค่า default ที่ใช้ได้ทันที ผู้เรียนแค่ลากมาต่อก็เห็นผล
- มี label ภาษาไทยกำกับ + tooltip อธิบายสั้น ๆ ว่า filter แต่ละตัวทำอะไร
- **เมื่อต่อ block ผิดชนิด input** (เช่น ต่อภาพสีเข้า Canny) ระบบ auto-convert ให้ + แสดง badge เตือนอ่อน ๆ ว่า "แปลงเป็น grayscale ให้อัตโนมัติ" แทนที่จะ error

---

## 2. โครงสร้างโฟลเดอร์ที่เสนอ

```
Machinvision app/
├── package.json
├── vite.config.js           # bundler (renderer) — ดูหัวข้อ 6 การตัดสินใจ
├── electron/
│   ├── main.js              # Main process: สร้างหน้าต่าง, เมนู, IPC, permission กล้อง
│   └── preload.js           # bridge ปลอดภัย (contextIsolation=true, nodeIntegration=false)
├── src/
│   ├── index.html           # โครง UI หลัก
│   ├── styles/
│   │   └── app.css
│   ├── renderer/
│   │   ├── app.js           # จุดเริ่มของ renderer, ต่อกล้อง + วน render loop
│   │   ├── camera.js        # จัดการ getUserMedia + capture frame + เลือกกล้อง
│   │   ├── cv/
│   │   │   ├── opencv-loader.js   # โหลด OpenCV.js (bundle local)
│   │   │   └── mat-utils.js       # helper แปลง canvas ↔ cv.Mat + auto-convert channel
│   │   ├── modes/
│   │   │   └── color-modes.js     # Grayscale/RGB/HSV/HSL (Phase 1)
│   │   ├── filters/               # filter block ทุกกลุ่ม (Phase 2)
│   │   │   ├── index.js           # registry: ชนิด input/output + default params ของแต่ละ block
│   │   │   ├── color.js
│   │   │   ├── blur.js
│   │   │   ├── sharpen.js
│   │   │   ├── morphology.js
│   │   │   ├── edge.js
│   │   │   └── threshold.js
│   │   ├── graph/                 # node-graph editor (Phase 2)
│   │   │   ├── editor.js          # ครอบ LiteGraph.js
│   │   │   └── executor.js        # topological sort + ไล่ประมวลผล DAG ทุกเฟรม
│   │   └── ui/
│   │       ├── mode-bar.js        # แถบเลือกโหมดพื้นฐาน
│   │       └── fps-meter.js       # แสดง FPS (Phase 3)
├── vendor/
│   ├── opencv/                    # OpenCV.js (wasm) เก็บ local ไม่พึ่ง CDN
│   └── litegraph/                 # LiteGraph.js เก็บ local
├── build/                         # ผลลัพธ์ packaging (installer) — electron-builder
└── pipelines/                     # ตัวอย่าง pipeline บันทึกเป็น JSON (Phase 3)
```

---

## 3. แผนแบ่งเป็น Phase (งานย่อยที่ตรวจได้)

### Phase 1 — Core video pipeline
1. `npm init` + ติดตั้ง Electron + Vite, ตั้ง security config (`contextIsolation=true`, `nodeIntegration=false`) + จัดการ permission กล้องใน `main.js` → เปิดหน้าต่างเปล่าได้
2. วาง OpenCV.js (wasm) ลง `vendor/opencv/` แล้วโหลดผ่าน `opencv-loader.js` (bundle local + หน้า loading รอ wasm พร้อม)
3. `camera.js` เปิด webcam ด้วย `getUserMedia` แสดงสดลง `<canvas>` (+ เลือกกล้องถ้ามีหลายตัว)
4. `color-modes.js` + `mode-bar.js` สลับ Grayscale/RGB/HSV/HSL ได้จริง
   - **หมายเหตุ HSL**: OpenCV.js ไม่มี `COLOR_RGB2HSL` — ใช้ `COLOR_RGB2HLS` (ลำดับ channel = H-L-S ต้องสลับตอนแสดง/อธิบาย)
5. เกณฑ์ผ่าน Phase 1: เปิดแอป → เห็นภาพสด → กดสลับ 4 โหมดได้ลื่น

### Phase 2 — Filter chain แบบ block (ทำ filter ครบทุกกลุ่ม)
6. วาง LiteGraph.js ลง `vendor/litegraph/` + `editor.js` แสดง canvas node-graph
7. สร้าง `registry` (index.js) กำหนด **ชนิด input/output ของแต่ละ block** (1-channel / 3-channel) + default params — เพื่อให้ executor ตรวจและ auto-convert ได้
8. สร้าง filter block ตาม registry — ทุกกลุ่ม:
   - **Color**: Grayscale, Color convert (RGB/HSV/HLS), Channel split *(หลาย output)*
   - **Blur**: Gaussian, Average/Box, Median, Bilateral
   - **Sharpening**: Unsharp mask, Laplacian sharpening
   - **Morphology**: Erosion, Dilation, Opening, Closing, Gradient, Top Hat, Black Hat (เลือก kernel: rect/ellipse/cross + size) *(รับ 1-channel)*
   - **Edge**: Sobel, Canny, Laplacian *(รับ 1-channel)*
   - **Threshold**: Binary, Adaptive *(รับ 1-channel)*
9. `executor.js`: ทำ **topological sort** ของ DAG แล้วไล่ประมวลผลทุกเฟรม — ตรวจชนิด input ทุกเส้นเชื่อม ถ้าไม่ตรงให้ **auto-convert** (เช่น สี→grayscale) + ส่ง flag ให้ UI ขึ้น badge เตือน
10. รองรับต่อ block ข้ามกลุ่มได้อิสระ (เช่น Blur → Sharpen → Morphology → Threshold) และ branch หลายเส้นจาก Channel split
11. เกณฑ์ผ่าน Phase 2: ต่อ block ≥3 ตัวข้ามกลุ่ม + ต่อ Channel split แตกสองเส้น แล้วเห็นผลลัพธ์ต่อเนื่องทุกเฟรม โดยไม่ error เมื่อชนิด input ไม่ตรง

### Phase 3 — Polish + แจกจ่าย
12. บันทึก/โหลด pipeline เป็น JSON (`pipelines/`)
13. FPS meter (`fps-meter.js`)
14. (ถ้าจำเป็น) ย้ายประมวลผลไป Web Worker + OffscreenCanvas กัน UI กระตุก
15. **Packaging**: ตั้ง electron-builder สร้าง installer (Windows `.exe` ก่อน) → ผู้เรียนติดตั้งได้โดยไม่ต้องลง Node/Python เลย

---

## 4. ความเสี่ยง / จุดที่ต้องระวัง

- **OpenCV.js wasm ขนาดใหญ่ (~8-10MB)** — ต้องรอโหลดตอนเปิดแอป ควรมีหน้า loading
- **Performance ต่อเฟรม** — filter หนัก (Bilateral, Median kernel ใหญ่) อาจทำ FPS ตก → เตรียม Web Worker ไว้เผื่อ
- **Memory ของ cv.Mat** — ต้อง `.delete()` ทุก Mat ที่สร้าง ไม่งั้น memory leak ในลูป render (แนะนำทำ helper บริหาร Mat pool ต่อเฟรม)
- **สิทธิ์กล้อง** — Electron ต้องจัดการ permission ของ getUserMedia (`setPermissionRequestHandler`)
- **ชนิด input ไม่ตรงระหว่าง block** — จัดการด้วย auto-convert + badge เตือน (ดู Phase 2 ข้อ 9) กันมือใหม่งง
- **OpenCV.js ผูกกับ global (`cv`)** — ไม่ใช่ ES module สะอาด ต้องโหลดผ่าน loader + รอ `onRuntimeInitialized` ก่อนใช้งาน

---

## สถานะการพัฒนา (อัปเดต 2026-07-16)

**เสร็จแล้ว:**
- ✅ Phase 1: Electron + Vite + OpenCV.js (ผ่าน Web Worker) + กล้องสด
- ✅ **Web Worker ถูกดึงมาทำใน Phase 1** (จำเป็น) — OpenCV คอมไพล์ wasm แบบ synchronous ทำให้ main thread ค้าง จึงย้ายไป worker แยก thread
- ✅ Phase 2 — **node-graph สไตล์ Node-RED** (LiteGraph.js) ตามการตัดสินใจของผู้ใช้ 2026-07-16:
  - กระดานต่อ node: ลาก [📷 แหล่งภาพ] → filter nodes → [🖥️ ผลลัพธ์] ต่อสายได้อิสระ
  - เปิดแอปมามีกราฟตั้งต้น "แหล่งภาพ → ผลลัพธ์" ต่อสายให้แล้ว (มือใหม่เห็นภาพทันที)
  - ถอด/ต่อสายเมื่อไร ประมวลผลใหม่ทันที; สายไม่ครบ → แจ้งเตือนใน status bar
  - Palette ซ้าย: คลิกเพิ่ม filter node ลงกระดาน (จัดกลุ่ม)
  - Detail panel ขวา: คลิก node → เห็นคำอธิบายการทำงาน + สไลเดอร์ปรับค่า มีผลสด
  - นำเข้าภาพนิ่ง + สลับกล้อง/ภาพนิ่งได้
  - Filter ชุดหลัก: Grayscale, Gaussian/Median Blur, Canny/Sobel, Binary/Adaptive Threshold
  - auto-convert เป็นขาวดำเมื่อ filter ต้องการ + แจ้งใน detail panel

> **หมายเหตุการตัดสินใจ**: เดิมทำเป็น panel ต่อชั้น (ชั่วคราว) แต่ผู้ใช้เลือกเปลี่ยนเป็น node-graph สไตล์ Node-RED เพราะเหมาะกับการเรียนการสอนมากกว่า — เห็น "การไหลของภาพ" เป็นเส้นสายชัดเจน ตรงกับสเปกหัวข้อ 2 ที่วาง LiteGraph.js ไว้แต่แรก

- ✅ **ชุด filter ตามหัวข้อการสอน 6 เรื่อง** (เพิ่มตามคำขอผู้ใช้ 2026-07-16) — รวม 23 nodes ทดสอบผ่านครบทุกตัว:
  | หัวข้อวิชา | Nodes |
  |---|---|
  | 1. ปริภูมิสี (Color Space) | Grayscale, แปลงปริภูมิสี (HSV/HLS/Lab/YCrCb), แยกดูช่องสี (R/G/B/H/S/V) |
  | 2. ปรับปรุงภาพ (Enhancement) | ความสว่าง/คอนทราสต์, Gamma, Histogram Equalization, CLAHE, Unsharp Mask, Laplacian Sharpen |
  | 3. ฟื้นฟูภาพ (Restoration) | ใส่ noise เกลือ-พริกไทย (โจทย์ทดลอง), Median, Gaussian, Box, Bilateral |
  | 4. สัณฐานวิทยา (Morphology) | Erode/Dilate/Open/Close/Gradient/TopHat/BlackHat + เลือก kernel shape/size/iterations |
  | 5. แบ่งส่วนภาพ (Segmentation) | Binary/Otsu/Adaptive Threshold, Canny, Sobel, หาเส้นรอบวัตถุ+นับจำนวน (Contours) |
  | 6. ประมวลผลภาพสี (Colour) | เลือกช่วงสี HSV (inRange) mask/extract, Invert |
- ✅ ระบบรายงานผลใน detail panel: Otsu แสดงค่าเกณฑ์ที่คำนวณได้, Contours แสดงจำนวนวัตถุที่พบ

**ถัดไป:** branch หลายเส้น (แสดงผลหลายจุดพร้อมกัน), บันทึก/โหลดกราฟเป็น JSON, FPS meter, packaging (.exe)

---

## 6. การตัดสินใจ / คำถามที่ยังค้าง (ตอบเมื่อพร้อมลงโค้ด)

| # | หัวข้อ | ข้อเสนอ | สถานะ |
|---|---|---|---|
| 1 | node-graph library | **LiteGraph.js** (เบา, เหมาะ real-time) | ✅ ยืนยันแล้ว — UI แบบ Node-RED ตามที่ผู้ใช้เลือก (2026-07-16) |
| 2 | Bundler | **Vite** (จัดการ dependency ยุคใหม่ง่าย) | ✅ ใช้งานจริงแล้ว |
| 3 | Export ภาพ/วิดีโอผลลัพธ์ | ยังไม่กำหนด (สเปกหัวข้อ 6 ข้อ 2) | รอตอบ |
| 4 | เป้าหมาย platform packaging | Windows `.exe` ก่อน (เครื่องผู้ใช้เป็น Windows 10) | รอยืนยัน |
| 5 | เริ่มลงมือ Phase 1 เมื่อไร | ให้ review แผนก่อน แล้วค่อยเริ่ม | รอสัญญาณ |
```


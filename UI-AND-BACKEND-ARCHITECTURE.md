# Machine Vision by IRiSH — UI & Backend Architecture (ปัจจุบัน)

> อธิบายทั้ง **หน้าตา/UI** และ **สถาปัตยกรรม backend (มุม coder)** ของโปรแกรมเวอร์ชันปัจจุบัน
> v0.1.0 · อัปเดต 2026-08-17
> คู่กับ [UI-LAYOUT-CURRENT.md](UI-LAYOUT-CURRENT.md) (รายละเอียดปุ่ม/เลย์เอาต์)

---

# ส่วนที่ 1 — ภาพรวมสถาปัตยกรรม

## 1.1 เป็นแอปแบบไหน
Electron desktop app · UI เป็น web (HTML/CSS/JS ล้วน ไม่มี framework) · งานหนัก (OpenCV, YOLO) รันใน **Web Worker** แยกจาก UI thread

```
┌─ Electron Main process (electron/main.js) ─────────────────┐
│  • สร้างหน้าต่าง · โหลด UI (dev: localhost:5173 / prod: dist)│
│  • IPC: เลือกโฟลเดอร์ · เขียนไฟล์ · อ่าน CPU/RAM · แปลง .pt │
└───────────────▲────────────────────────────────────────────┘
                │ contextBridge (preload.cjs)
┌───────────────┴─ Renderer (UI thread, app.js) ─────────────┐
│  • state · โหมด · วาดผลลัพธ์ลง canvas · LiteGraph editor    │
│  • ส่งเฟรม → worker, รับผลกลับ → วาด                        │
└───┬──────────────────┬───────────────────┬─────────────────┘
    │ postMessage       │                   │
┌───▼─── cv-worker ──┐ ┌▼── dl-worker ──┐  ┌▼─ MediaPipe (main thread) ┐
│ OpenCV.js (wasm)   │ │ onnxruntime-web │  │ PoseLandmarker            │
│ รัน filter pipeline│ │ YOLO inference  │  │ Ergonomics (ท่าทาง)       │
└────────────────────┘ └─────────────────┘  └───────────────────────────┘
```

## 1.2 เทคโนโลยีหลัก
| ชั้น | ใช้อะไร |
|---|---|
| Shell | Electron (main + preload + renderer) |
| Build | Vite (root=src, base="./", outDir=../dist) + electron-builder (nsis + portable) |
| Image processing | `@techstark/opencv-js` (wasm) ใน Web Worker |
| Node graph | `litegraph.js` (วาดบน canvas) |
| Deep Learning | `onnxruntime-web` (YOLO .onnx) ใน Web Worker แยก |
| Pose | `@mediapipe/tasks-vision` PoseLandmarker |

---

# ส่วนที่ 2 — Backend / Processing (มุม coder)

## 2.1 แผนผังไฟล์ (renderer)
```
src/renderer/
├── app.js              (1644) coordinator หลัก — state, โหมด, loop, วาดผล, IPC, keymap
├── camera.js           (75)   เปิด/ปิดกล้อง (getUserMedia) + camGen guard กัน track leak
├── codegen.js          (318)  สร้างโค้ด Python/OpenCV จาก pipeline
├── ergonomics.js       (126)  คำนวณมุม/เกณฑ์การยศาสตร์จาก pose landmarks
├── mediapipe-pose.js   (172)  PoseLandmarker (VIDEO สด + IMAGE สำหรับภาพแคป)
├── dl-yolo.js          (154)  ฝั่ง renderer ของ YOLO — คุย dl-worker, จัดคิวโหลดโมเดล
├── dl-worker.js        (88)   Web Worker: onnxruntime-web inference
├── cv/cv-worker.js     (157)  Web Worker: OpenCV.js รัน filter pipeline
├── filters/
│   ├── registry.js     OPS (algorithm จริง) + FILTERS (กลุ่ม merge) + GROUP_COLORS/ORDER
│   └── ui-meta.js      metadata สำหรับ UI (ไอคอน, ค่าหลักบน node) — ไม่มี algorithm
├── graph/editor.js     (530)  LiteGraph: node type, ต่อสาย, สกัด pipeline, undo/redo
└── ui/                 component ของแผง/dialog ต่าง ๆ
```

## 2.2 หัวใจ: Filter Registry (`filters/registry.js`)
โครงสร้าง 2 ชั้น:
- **`OPS`** = op จริง 28 ตัว แต่ละตัวมี `{ name, needs, params[], apply(cv, src, params, ctx) }`
  - `apply()` = ฟังก์ชันเรียก `cv.*` จริง (นี่คือ algorithm — **ห้ามแตะเวลาแก้ UI**)
  - `needs` = "gray" | "any" (บอกว่าต้องแปลงขาวดำก่อนไหม)
  - `params[]` = schema: `{ key, type, label, min, max, step, default, options?, showIf? }`
- **`FILTERS`** = 7 กล่องที่โชว์บน palette (รวม op หลายตัวเป็น 1 กล่อง + ตัวเลือก "วิธี")
  - `mergeGroup()` รวม op → เพิ่ม select `mode` + prefix sub-params `opId__key` กันชนกัน
  - op ที่ทำงานจริง = `properties.mode`; ตัด prefix ด้วย `localParamsFor()`

**สำคัญ:** ชื่อ (name/group) เป็น **display อย่างเดียว** — ตอนนี้เป็นภาษาอังกฤษ ไม่กระทบ `apply`/schema

## 2.3 Pipeline execution (loop)
```
กล้อง/ภาพ → app.js pump() → ดึง ImageData → worker.postMessage({type:"frame", pipelines})
   → cv-worker: สำหรับแต่ละ Display ที่ต่อสาย → runPipeline(input, steps)
       ทีละ step: ถ้า needs==="gray" && input สี → toGray ก่อน → f.apply(cv, mat, params)
   → ได้ผล RGBA + notes (เช่น "พบวัตถุ 3 ชิ้น") → postMessage({type:"result", results})
   → app.js drawResults() → putImageData ลง canvas ของแต่ละ Display
```
- **cv-worker protocol:** `init` → `ready` · `frame` → `result`/`error` · `export` → `export-result`
- ผลส่งกลับแบบ **Transferable** (ArrayBuffer) เพื่อไม่ copy
- **memory:** ทุก `cv.Mat` ต้อง `.delete()` (จัดการใน worker)

## 2.4 Frame coalescing (กัน lag)
`pendingProcess` flag: ถ้า worker busy อยู่ตอนสไลเดอร์ขยับ → เก็บคำขอไว้ แล้วประมวลผล **ค่าล่าสุด** เมื่อ worker ว่าง (โหมดภาพนิ่ง) → เลื่อนค่าแล้วผลสุดท้ายเรนเดอร์เสมอ ไม่ค้างค่ากลางทาง

## 2.5 Node graph (`graph/editor.js`)
- 3 node type: `mv/source` (แหล่งภาพ) · `mv/display` (ผลลัพธ์) · `mv/dl-yolo` (YOLO)
- filter node ลงทะเบียนเป็น `filters/<id>`
- **`tracePipeline(displayNode)`** = เดินย้อนจาก Display กลับไป source → ได้ลำดับ step ที่ภาพวิ่งผ่าน
- **`extractPipelines()`** = ทำทุก Display → ส่งให้ app.js (`onPipelineChange`) → app ส่งต่อ worker
- undo/redo = `graph.serialize()`/`configure()` เก็บ history 60 ชั้น (JSON snapshot)
- ชื่อกล่องที่ผู้ใช้ตั้ง (`__mvName`) เก็บผ่าน `onSerialize`/`onConfigure` → ไม่หายตอน undo
- node card วาดเองด้วย `FilterNode.prototype.onDrawForeground(ctx)` (icon + ชื่อ + ค่าหลัก)

## 2.6 Deep Learning (YOLO)
- `dl-yolo.js` (renderer) จัดคิวโหลดโมเดลแบบ serialize (loadChain) กัน race + กู้ worker เมื่อ error
- `dl-worker.js`: `onnxruntime-web` executionProvider "wasm", numThreads=1 (เลี่ยง SharedArrayBuffer)
- โหลด `load` → `loaded` · ส่ง tensor `[1,3,size,size]` → `result` (boxes)
- .pt → .onnx: ผ่าน IPC `dl:convertPt` (เรียก ultralytics ในเครื่อง — ต้องมี Python/ultralytics)

## 2.7 Pose / Ergonomics
- `mediapipe-pose.js`: PoseLandmarker 2 ตัว — VIDEO (กล้องสด) + IMAGE (ภาพแคป)
- `ergonomics.js`: คำนวณมุมข้อต่อ → ระดับความเสี่ยง 4 ระดับ (ดี/พอใช้/เสี่ยง/ไม่พบร่างกาย)

## 2.8 Codegen & Export
- `codegen.js`: แปลง pipeline → โค้ด Python `cv2.*` ตามลำดับจริง (ติดตาม gray-state, BGR)
- Export YOLO (`ui/export-dialog.js`): สร้าง dataset + Jupyter `.ipynb` เทรน — **โค้ดที่ generate ล็อก ห้ามแก้**

## 2.9 Electron main (`electron/main.js`) + IPC
| IPC channel | ทำอะไร |
|---|---|
| `export:chooseDir` | เปิด dialog เลือกโฟลเดอร์ (เลือกครั้งเดียว) |
| `export:writeFile` | เขียนไฟล์ลงโฟลเดอร์ (กันเขียนนอก path — ใช้ basename) |
| `sys:metrics` | อ่าน CPU%/RAM จาก `app.getAppMetrics()` |
| `dl:convertPt` | แปลงโมเดล .pt → .onnx (spawn ultralytics) |

preload (`preload.cjs`) เปิดสะพานเฉพาะที่จำเป็น: `appInfo`, `mvExport`, `mvMetrics`, `mvDL`

## 2.10 Build
| คำสั่ง | ผล |
|---|---|
| `npm run dev` | vite dev (5173) + electron |
| `npm run build` | vite build → `dist/` |
| `npm run dist` | build + electron-builder --win (**nsis installer + portable .exe**) |
| `npm run dist:dir` | build + unpacked (ไม่แพ็ก installer) |

config: appId `com.irish.machinevision` · productName "Machine Vision By IRiSH" · **asar:false** · nsis (oneClick:false เลือกโฟลเดอร์ได้)

---

# ส่วนที่ 3 — UI / Frontend (ปัจจุบัน)

## 3.1 โครงหน้าจอ
3 คอลัมน์: **ซ้าย 260px · กลาง ยืด · ขวา 360px** · Live View ได้พื้นที่มากกว่า Pipeline (~58/42, Live View มี floor สูงกว่า = พระเอกเสมอ)

```
┌ TOPBAR: [โลโก้] SOURCE│MODE│IMPORT ··· CONTROL│EXPORT ┐
├ TOOLS │ LIVE VIEW (แถบหัว + ภาพ) │ PROPERTIES        ┤
│       │ CAPTURE TRAY             │ ─────             │
│       │ PIPELINE CANVAS          │ PERFORMANCE·LOGS  │
├ STATUSBAR: สถานะ ···· FPS N ···· เวอร์ชัน ───────────┤
```

## 3.2 Topbar — 5 cluster (bilingual caption)
`SOURCE / แหล่งภาพ` · `MODE / โหมด` · `IMPORT / นำเข้า` · `CONTROL / ควบคุม` · `EXPORT / ส่งออก & ตั้งค่า`
- **SOURCE:** 📷 กล้อง · 🖼️ ภาพนิ่ง
- **MODE** (เฉพาะกล้อง): 📸 แคปภาพ · ต้นฉบับ · 🧍 Ergonomics Pose · **● สด** (เขียว) · เลือกกล้อง
- **IMPORT:** ＋ นำเข้าภาพ/วิดีโอ · 📁 โฟลเดอร์ · ◀ ภาพ X/N ▶
- **CONTROL** (เฉพาะกล้อง): ⏸ หยุด / ▶ รัน
- **EXPORT:** 💾 Save Picture (เขียว) · ⬇ Export Yolo (ม่วง) · ความละเอียด · ⚙

## 3.3 Panel ซ้าย — "เครื่องมือ (Tools)"
🔍 ค้นหา (Ctrl+/) · ▦ เทมเพลต · ＋ เพิ่มผลลัพธ์ · 🧠 YOLO Detect · **7 หมวด (ชื่ออังกฤษ)**:
Color Space · Enhancement · Restoration · Morphology · Segmentation · Object Detection · Colour
การ์ด: `[icon chip สีหมวด] ชื่อ` คลิก = วาง node

## 3.4 Live View
แถบหัวจริง `LIVE VIEW ● สด  แหล่ง··· ความละเอียด··· FPS··· สี···  MEM` (ไม่ลอยทับภาพ)
พื้นที่ภาพ (`.stage-body`) สลับ overlay: grid ผลลัพธ์ / pose / detect / วิดีโอ / empty / error
ปุ่มลอย: 📸 แคป · ⛶ เต็มจอ

## 3.5 Pipeline Canvas
node card 220×92 (แถบหัวมืด + จุดสีหมวด + ชื่ออังกฤษ + ค่าหลัก "ksize: 5")
เส้นเชื่อม 2px · ปุ่มควบคุม: ↶↷ ＋− ⤢ 🗑
breadcrumb บนหัวจอผลลัพธ์: `ภาพ › Gaussian Blur ▾ › Canny ▾`

## 3.6 Panel ขวา — "คุณสมบัติ (PROPERTIES)"
- เลือก filter: ชื่อ + หมวด + ▶ วิธี + **พารามิเตอร์** (section):
  - number → grid `label / slider / [ค่า]` (ค่าชิดขวาตรงแนวทุกแถว)
  - select → ปุ่มติ๊ก ✓ (ชื่อวิธีอังกฤษ)
  - boolean → toggle ON/OFF
- เลือก Display: ตั้งชื่อ + ขั้นตอน + 🐍 สร้างโค้ด Python
- เลือก YOLO: สถานะ + 📂 เลือกโมเดล
- ท้ายแผง (พับได้): **PERFORMANCE** (FPS/CPU/RAM/GPU"—") · **LOGS** (auto-expand ตอน error)

## 3.7 Design system
| อย่าง | ค่า |
|---|---|
| ธีม | Dark industrial (--bg-app #07101b, --bg-panel #0d1827, --bg-card #111e2f) |
| ตัวอักษร | Inter + Noto Sans Thai + tabular-nums |
| Radius | 6/10/14 · Spacing 4/8/12/16/24/32 |
| สีหมวด | accent ที่ icon/เส้น/จุด/ขอบตอนเลือก — ไม่ลงพื้นเต็มการ์ด |
| Scrollbar | slim 8px · muted |

## 3.8 State / Interaction
- **Empty** (ภาพนิ่งไม่มีภาพ): 🖼️ + ปุ่ม ＋เลือกภาพ / 📁โฟลเดอร์
- **Error** (กล้อง): ⚠ + ขั้นตอน + [เลือกกล้องใหม่][ใช้ภาพนิ่งแทน]
- **Loading**: spinner + ข้อความตามบริบท
- **คีย์ลัด**: Space แคป · M/C สลับโหมด · ←→ ภาพ · E โค้ด · Ctrl+Z/Y · Delete · Ctrl+/ · Esc

---

# ส่วนที่ 4 — สัญญา (contract) ที่ห้ามแตะเวลาแก้ UI

งาน UI ทั้งหมดที่ผ่านมาเป็น **frontend/presentation ล้วน** — ไม่แตะสิ่งเหล่านี้:
```
OpenCV apply() ทุก filter · cv-worker protocol · pipeline execution order
parameter schema (min/max/step/default/key) · needs
YOLO inference/export · Python generator · MediaPipe/Ergonomics
node type/id (mv/source, mv/display, mv/dl-yolo, filters/<id>)
keymap action logic · undo/redo semantics · camera/capture logic
```
กติกา: ปรับได้ทุกอย่างถ้า **input เดิม → processing เดิม → output เดิม** และ contract ที่ module อื่นอ่านยังครบ

---

# ส่วนที่ 5 — สิ่งที่ตัดออก / dead code

**ตัดออกถาวร (requirement ล่าสุด):** Learning Mode · learning card (💡🧪📌) · subtitle เครื่องมือ · param desc/hint · Onboarding เด้งครั้งแรก · แถบ Tips

**dead code เหลือค้าง (ไม่มีผล — เก็บกวาดภายหลังได้):**
- CSS: `.learn-*`, `.param-desc`, `.choice-hint`, `.af-sub`, `.props-tips`, `.detail-panel-sub`
- `ui-meta.js`: `OP_EXPERIMENT`, `OP_USECASE`, `GROUP_SUBTITLE` (ไม่ถูก import แล้ว)
- `registry.js`: field `desc`/`hint`/`opt.hint` ของ param (ไม่ถูก render แล้ว — แต่ไม่กระทบ apply)
- `onboarding.js`: `showOnboarding()` (ยังมีแต่ไม่ถูกเรียก) · `showTemplates()` ยังใช้อยู่

**ยังใช้อยู่จาก ui-meta:** `GROUP_ICON` (ไอคอนการ์ด+node) · `OP_PRIMARY` (ค่าหลักบน node) · `currentOpId`/`localParamsFor`/`hexA`

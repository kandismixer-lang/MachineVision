// Phase 3: ประกอบ dataset ที่ตีกรอบไว้ (dataset/annotations.js) ให้เป็นไฟล์ .zip รูปแบบ YOLO
// อ่านคลาส/annotation/split จาก annotations.js (single source of truth) ตรง ๆ — ไม่รับ/เก็บ state ซ้ำที่นี่
// ไม่แตะ DOM เลย (เหมือน annotations.js) — ฝั่งเรียก (app.js) มีหน้าที่หา canvas ของแต่ละภาพมาป้อนให้
//
// โครง .zip ที่ได้:
//   data.yaml
//   images/<split>/<name>.jpg
//   labels/<split>/<name>.txt   (1 บรรทัดต่อกรอบ: "classIdx cx cy w h" normalized 0..1 — ตรงตาม YOLO เป๊ะ)
//
// เผื่อ Phase 4 (UI แบ่ง train/valid/test): โมดูลนี้อ่าน getSplit(key) ต่อภาพอยู่แล้ว รองรับ 3 split ครบ
// ไม่ต้องแก้อะไรเพิ่มตอน Phase 4 มาต่อ (แค่ผู้ใช้ตั้ง split ต่างกันในแผงคลาส/ที่ไหนสักที่)
import JSZip from "jszip";
import { getClasses, getAnnotations, getSplit } from "./annotations.js";
import { augmentEnabled, buildPlan, applyPlan, transformAnnotations } from "./augment.js";

const f6 = (v) => Number(v).toFixed(6);

// ---------- แปลง annotation เป็นบรรทัด label ----------
// detect: "cls cx cy w h" (bbox center-based) — polygon ถูกลดรูปเป็น bbox ครอบ, line ข้าม
// segment: "cls x1 y1 x2 y2 ..." (พิกัด polygon) — bbox ถูกกางเป็นสี่เหลี่ยม 4 มุม, line ข้าม
function bboxOfPoints(pts) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, w: maxX - minX, h: maxY - minY };
}

// รูปแบบ detect — คืน null ถ้าแปลงไม่ได้ (เช่น line)
function toDetectLine(a) {
  if (a.type === "line") return null;
  if (a.type === "polygon") {
    const b = bboxOfPoints(a.points);
    return `${a.classIdx} ${f6(b.cx)} ${f6(b.cy)} ${f6(b.w)} ${f6(b.h)}`;
  }
  return `${a.classIdx} ${f6(a.cx)} ${f6(a.cy)} ${f6(a.w)} ${f6(a.h)}`;
}

// รูปแบบ segment — คืน null ถ้าแปลงไม่ได้ (line ไม่ใช่พื้นที่ปิด จึงข้าม)
function toSegLine(a) {
  if (a.type === "line") return null;
  let pts;
  if (a.type === "polygon") {
    pts = a.points;
  } else {
    // bbox → สี่เหลี่ยม 4 มุม (ตามเข็มนาฬิกา)
    const l = a.cx - a.w / 2, r = a.cx + a.w / 2, t = a.cy - a.h / 2, b = a.cy + a.h / 2;
    pts = [{ x: l, y: t }, { x: r, y: t }, { x: r, y: b }, { x: l, y: b }];
  }
  return `${a.classIdx} ${pts.map((p) => `${f6(p.x)} ${f6(p.y)}`).join(" ")}`;
}

// เข้ารหัสภาพทีละใบเป็น JPEG (ไม่ค้าง memory หลายใบพร้อมกัน — รองรับ dataset ใหญ่)
function canvasToJpegBlob(canvas, quality = 0.92) {
  return new Promise((resolve, reject) => {
    if (!canvas || typeof canvas.toBlob !== "function") {
      reject(new Error("export-yolo: canvas ไม่ถูกต้อง (ไม่มี toBlob)"));
      return;
    }
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("export-yolo: แปลง canvas เป็น JPEG ไม่สำเร็จ"));
    }, "image/jpeg", quality);
  });
}

function sanitizeBase(name, fallbackIdx) {
  const base = (name || "").replace(/\.[^./\\]+$/, "");
  const clean = base.replace(/[\\/:*?"<>|]+/g, "_").trim();
  return clean || `image_${fallbackIdx + 1}`;
}

// กันชื่อ base ชนกันภายใน split เดียวกัน (เช่น "photo.jpg" กับ "photo.png" → base "photo" ทั้งคู่)
// ไม่ de-dupe → zip.file() เขียนทับของเก่าเงียบ ๆ (ภาพ+label หายทั้งใบ แต่ stats นับไปแล้วก่อนเขียน)
// ต่อ suffix _2, _3, ... จนกว่าจะไม่ชนภายใน split นั้น — คำนวณครั้งเดียวแล้วใช้ base เดียวกันทั้ง .jpg และ .txt
function dedupeBase(usedBySplit, split, base) {
  let used = usedBySplit.get(split);
  if (!used) {
    used = new Set();
    usedBySplit.set(split, used);
  }
  if (!used.has(base)) {
    used.add(base);
    return base;
  }
  let n = 2;
  let candidate = `${base}_${n}`;
  while (used.has(candidate)) {
    n++;
    candidate = `${base}_${n}`;
  }
  used.add(candidate);
  return candidate;
}

/**
 * ประกอบ YOLO dataset เป็น Blob ของไฟล์ .zip เดียว
 * @param {Array<{key: string, getCanvas: () => Promise<HTMLCanvasElement|null>}>} images
 *   ลำดับตรงกับ state.images ของแอป — key = ชื่อไฟล์ภาพ (ตรงกับที่ใช้ผูก annotation ใน annotations.js)
 *   getCanvas เรียกเฉพาะภาพที่มีกรอบเท่านั้น (lazy) — คืน canvas พื้นหลังตามโหมดที่ตีกรอบ (ต้นฉบับ/ผ่าน filter)
 * @param {{onProgress?: (done:number, total:number) => void}} opts
 * @returns {Promise<{blob: Blob, stats: {totalImages:number, exported:number, skipped:number, boxes:number, classes:number, perSplit:{train:number,valid:number,test:number}}}>}
 */
export async function exportYoloDataset(images, { onProgress, augment } = {}) {
  const augOn = augmentEnabled(augment); // สร้างภาพ aug เพิ่มเฉพาะ split=train
  const classes = getClasses();
  if (classes.length === 0) {
    throw new Error("ยังไม่มีคลาส — สร้างคลาสอย่างน้อย 1 คลาสก่อน export");
  }

  // Drawing a polygon does not change the training task. Preserve source points
  // in annotations.js and export their enclosing box for detection.
  const task = "detect";
  const toLine = toDetectLine;

  const zip = new JSZip();
  const perSplit = { train: 0, valid: 0, test: 0 };
  let exported = 0;
  let skipped = 0;
  let boxes = 0;
  let skippedLines = 0; // annotation ที่แปลงไม่ได้ (เช่น line) — ข้ามรายเส้น ไม่ทำทั้งภาพหาย
  let augmented = 0; // ภาพ augment ที่สร้างเพิ่ม (offline)
  const total = images.length;
  const usedBySplit = new Map(); // split -> Set<base> ที่ใช้แล้ว (กัน overwrite ใน zip เมื่อชื่อ base ซ้ำ)

  for (let i = 0; i < total; i++) {
    const { key, getCanvas } = images[i];
    const anns = key ? getAnnotations(key) : [];
    // ภาพไม่มีกรอบเลย → ข้าม (YOLO เทรนภาพเปล่าไม่ได้ในเฟสนี้ — ดู note Phase 3)
    if (anns.length === 0) {
      skipped++;
      onProgress?.(i + 1, total);
      continue;
    }

    // แปลงเป็นบรรทัด label ตามงาน (detect/segment) — ตัดตัวที่แปลงไม่ได้ (line) ออก
    const lines = [];
    for (const a of anns) {
      const line = toLine(a);
      if (line == null) { skippedLines++; continue; }
      lines.push(line);
    }
    // ภาพนี้เหลือแต่ line (แปลงไม่ได้) → ไม่มี label เทรนได้ ข้ามทั้งภาพ
    if (lines.length === 0) {
      skipped++;
      onProgress?.(i + 1, total);
      continue;
    }

    const canvas = await getCanvas();
    if (!canvas || !canvas.width || !canvas.height) {
      skipped++;
      onProgress?.(i + 1, total);
      continue;
    }

    const split = getSplit(key);
    const base = dedupeBase(usedBySplit, split, sanitizeBase(key, i));

    const blob = await canvasToJpegBlob(canvas);
    const bytes = new Uint8Array(await blob.arrayBuffer());
    // TODO(scale): dataset ใหญ่มาก — เก็บทุกไฟล์ใน memory ก่อน generateAsync (peak ~3x) อาจต้อง stream ลงดิสก์ทีละไฟล์แทน
    zip.file(`${split}/images/${base}.jpg`, bytes);
    zip.file(`${split}/labels/${base}.txt`, lines.join("\n") + "\n");

    perSplit[split]++;
    boxes += lines.length;
    exported++;

    // ---- Offline augmentation: สร้างภาพก็อปที่แปลงแล้ว เฉพาะ train ----
    if (augOn && split === "train") {
      const W = canvas.width, H = canvas.height;
      for (let a = 0; a < augment.copies; a++) {
        const plan = buildPlan(augment, W, H);
        if (plan.identity) continue; // ก็อปนี้เหมือนต้นฉบับ (flip สุ่มไม่ติด) — ข้าม กันภาพซ้ำ
        const augAnns = transformAnnotations(anns, plan, W, H);
        const augLines = [];
        for (const an of augAnns) {
          const line = toLine(an);
          if (line != null) augLines.push(line);
        }
        if (augLines.length === 0) continue;
        const augCanvas = applyPlan(canvas, plan);
        const augBase = dedupeBase(usedBySplit, split, `${base}_aug${a + 1}`);
        const augBlob = await canvasToJpegBlob(augCanvas);
        const augBytes = new Uint8Array(await augBlob.arrayBuffer());
        zip.file(`${split}/images/${augBase}.jpg`, augBytes);
        zip.file(`${split}/labels/${augBase}.txt`, augLines.join("\n") + "\n");
        perSplit[split]++;
        boxes += augLines.length;
        augmented++;
      }
    }

    onProgress?.(i + 1, total);
  }

  if (exported === 0) {
    throw new Error("ไม่มีภาพที่มีกรอบเลย — ตีกรอบอย่างน้อย 1 ภาพก่อน export");
  }

  // สร้าง data.yaml จาก perSplit จริง — ทุก path ที่อ้างต้องมีภาพจริงในนั้น
  // (กัน FileNotFoundError ตอนเทรนบน Colab เมื่อผู้ใช้ไม่ได้แบ่ง valid/test)
  const hasSplit = (s) => perSplit[s] > 0;
  const trainDir = hasSplit("train") ? "train" : hasSplit("valid") ? "valid" : "test";
  const valDir = hasSplit("valid") ? "valid" : trainDir;

  // path = /content/dataset (ที่ที่โน้ตบุ๊กสั่ง unzip ไป) — ตั้งไว้เลยเพื่อให้ !unzip แล้วเทรนได้ทันที
  // ไม่ต้องแก้ data.yaml เพิ่มบน Colab (Ultralytics resolve path relative ไม่ตรงถ้าใช้ "." )
  const yamlLines = [
    `train: ${trainDir}/images`,
    `val: ${valDir}/images`,
  ];
  if (hasSplit("test")) yamlLines.push("test: test/images");
  yamlLines.push(
    `nc: ${classes.length}`,
    `names: [${classes.map((c) => JSON.stringify(c.name)).join(", ")}]`,
    ""
  );
  const yaml = yamlLines.join("\n");
  zip.file("data.yaml", yaml);

  // TODO(scale): generateAsync ประกอบ zip ทั้งก้อนใน memory (peak ~3x ของ dataset) — dataset ใหญ่มากควร stream เขียนลงดิสก์ทีละไฟล์แทน
  const zipBlob = await zip.generateAsync({ type: "blob" });
  return {
    blob: zipBlob,
    stats: {
      totalImages: total,
      exported,
      skipped,
      boxes,
      skippedLines,
      augmented, // ภาพ augment ที่สร้างเพิ่ม (offline)
      task, // 'detect' | 'segment' — ฝั่งเรียกใช้บอกผู้ใช้ + เลือกโมเดล -seg ตอนเทรน
      classes: classes.length,
      perSplit,
    },
  };
}

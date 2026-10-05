// Export dataset เป็นรูปแบบ COCO (สำหรับ RF-DETR) — ต่างจาก YOLO txt
// โครง .zip (แบบ Roboflow COCO ที่ rfdetr ใช้):
//   <split>/_annotations.coco.json   (1 ไฟล์ต่อ split)
//   <split>/<name>.jpg
// โดย split = train/valid/test — rfdetr อ่านโฟลเดอร์นี้ตรง ๆ
//
// COCO json:
//   images:      [{ id, file_name, width, height }]
//   annotations: [{ id, image_id, category_id, bbox:[x,y,w,h] (px ซ้ายบน), area, iscrowd:0, segmentation }]
//   categories:  [{ id:0, name:"objects", supercategory:"none" }, { id:1.., name, supercategory:"objects" }]
//                (รูปแบบ Roboflow — category_id ของ annotation เริ่มที่ 1 = classIdx+1)
import JSZip from "jszip";
import { getClasses, getAnnotations, getSplit } from "./annotations.js";
import { augmentEnabled, buildPlan, applyPlan, transformAnnotations } from "./augment.js";

function canvasToJpegBlob(canvas, quality = 0.92) {
  return new Promise((resolve, reject) => {
    if (!canvas || typeof canvas.toBlob !== "function") {
      reject(new Error("export-coco: canvas ไม่ถูกต้อง"));
      return;
    }
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("export-coco: แปลง JPEG ไม่สำเร็จ"))), "image/jpeg", quality);
  });
}

function sanitizeBase(name, i) {
  const base = (name || "").replace(/\.[^./\\]+$/, "").replace(/[\\/:*?"<>|]+/g, "_").trim();
  return base || `image_${i + 1}`;
}

function dedupeBase(usedBySplit, split, base) {
  let used = usedBySplit.get(split);
  if (!used) { used = new Set(); usedBySplit.set(split, used); }
  if (!used.has(base)) { used.add(base); return base; }
  let n = 2;
  while (used.has(`${base}_${n}`)) n++;
  used.add(`${base}_${n}`);
  return `${base}_${n}`;
}

// annotation → bbox [x,y,w,h] px (ซ้ายบน) + segmentation (ถ้ามี polygon) + area
function annToCoco(a, W, H) {
  if (a.type === "polygon") {
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    const seg = [];
    for (const p of a.points) {
      seg.push(p.x * W, p.y * H);
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    const x = minX * W, y = minY * H, w = (maxX - minX) * W, h = (maxY - minY) * H;
    // area จริงของ polygon (shoelace) — ไม่ใช่ bbox area (pycocotools ใช้จัด small/medium/large AP)
    let acc = 0;
    for (let i = 0, n = seg.length / 2; i < n; i++) {
      const j = (i + 1) % n;
      acc += seg[i * 2] * seg[j * 2 + 1] - seg[j * 2] * seg[i * 2 + 1];
    }
    return { bbox: [x, y, w, h], area: Math.abs(acc) / 2, segmentation: [seg] };
  }
  if (a.type === "line") return null; // เส้นเปิดไม่ใช่ object — ข้าม
  const w = a.w * W, h = a.h * H;
  const x = (a.cx - a.w / 2) * W, y = (a.cy - a.h / 2) * H;
  return { bbox: [x, y, w, h], area: w * h, segmentation: [] };
}

function round2(v) { return Math.round(v * 100) / 100; }

/**
 * ประกอบ COCO dataset เป็น .zip
 * @param {Array<{key:string, getCanvas:()=>Promise<HTMLCanvasElement|null>}>} images
 * @param {{onProgress?:(done:number,total:number)=>void}} opts
 */
export async function exportCocoDataset(images, { onProgress, augment } = {}) {
  const augOn = augmentEnabled(augment); // สร้างภาพ aug เพิ่มเฉพาะ split=train
  const classes = getClasses();
  if (classes.length === 0) throw new Error("ยังไม่มีคลาส — สร้างคลาสอย่างน้อย 1 คลาสก่อน export");

  const zip = new JSZip();
  // categories แบบ Roboflow: id 0 = supercategory, จริง 1..N
  const categories = [{ id: 0, name: "objects", supercategory: "none" }];
  classes.forEach((c, i) => categories.push({ id: i + 1, name: c.name, supercategory: "objects" }));

  // สะสมต่อ split
  const coco = {
    train: { images: [], annotations: [], categories },
    valid: { images: [], annotations: [], categories },
    test: { images: [], annotations: [], categories },
  };
  const perSplit = { train: 0, valid: 0, test: 0 };
  const usedBySplit = new Map();
  const imgId = { train: 0, valid: 0, test: 0 };
  let annId = 1;
  let exported = 0, skipped = 0, boxes = 0, skippedLines = 0, augmented = 0;
  const total = images.length;

  for (let i = 0; i < total; i++) {
    const { key, getCanvas } = images[i];
    const anns = key ? getAnnotations(key) : [];
    if (anns.length === 0) { skipped++; onProgress?.(i + 1, total); continue; }

    const canvas = await getCanvas();
    if (!canvas || !canvas.width || !canvas.height) { skipped++; onProgress?.(i + 1, total); continue; }
    const W = canvas.width, H = canvas.height;

    // แปลง annotation ทั้งหมดของภาพ (ตัด line ทิ้ง)
    const cocoAnns = [];
    for (const a of anns) {
      const c = annToCoco(a, W, H);
      if (!c) { skippedLines++; continue; }
      cocoAnns.push({ classIdx: a.classIdx, ...c });
    }
    if (cocoAnns.length === 0) { skipped++; onProgress?.(i + 1, total); continue; }

    const split = getSplit(key);
    const base = dedupeBase(usedBySplit, split, sanitizeBase(key, i));
    const fileName = `${base}.jpg`;

    const blob = await canvasToJpegBlob(canvas);
    zip.file(`${split}/${fileName}`, new Uint8Array(await blob.arrayBuffer()));

    const id = imgId[split]++;
    coco[split].images.push({ id, file_name: fileName, width: W, height: H });
    for (const c of cocoAnns) {
      coco[split].annotations.push({
        id: annId++,
        image_id: id,
        category_id: c.classIdx + 1, // 1-indexed (0 = supercategory)
        bbox: c.bbox.map(round2),
        area: round2(c.area),
        segmentation: c.segmentation.map((s) => s.map(round2)),
        iscrowd: 0,
      });
    }
    perSplit[split]++;
    boxes += cocoAnns.length;
    exported++;

    // ---- Offline augmentation: สร้างภาพก็อปที่แปลงแล้ว เฉพาะ train ----
    if (augOn && split === "train") {
      for (let a = 0; a < augment.copies; a++) {
        const plan = buildPlan(augment, W, H);
        if (plan.identity) continue; // ก็อปนี้เหมือนต้นฉบับ (flip สุ่มไม่ติด) — ข้าม กันภาพซ้ำ
        const augAnns = transformAnnotations(anns, plan, W, H);
        const augCoco = [];
        for (const an of augAnns) {
          const c = annToCoco(an, W, H);
          if (c) augCoco.push({ classIdx: an.classIdx, ...c });
        }
        if (augCoco.length === 0) continue;
        const augCanvas = applyPlan(canvas, plan);
        const augBase = dedupeBase(usedBySplit, split, `${base}_aug${a + 1}`);
        const augFile = `${augBase}.jpg`;
        const augBlob = await canvasToJpegBlob(augCanvas);
        zip.file(`${split}/${augFile}`, new Uint8Array(await augBlob.arrayBuffer()));
        const aid = imgId[split]++;
        coco[split].images.push({ id: aid, file_name: augFile, width: W, height: H });
        for (const c of augCoco) {
          coco[split].annotations.push({
            id: annId++,
            image_id: aid,
            category_id: c.classIdx + 1,
            bbox: c.bbox.map(round2),
            area: round2(c.area),
            segmentation: c.segmentation.map((s) => s.map(round2)),
            iscrowd: 0,
          });
        }
        perSplit[split]++;
        boxes += augCoco.length;
        augmented++;
      }
    }

    onProgress?.(i + 1, total);
  }

  if (exported === 0) throw new Error("ไม่มีภาพที่มีกรอบเลย — ตีกรอบอย่างน้อย 1 ภาพก่อน export");

  // เขียน _annotations.coco.json เฉพาะ split ที่มีภาพ (กัน rfdetr อ่านโฟลเดอร์ว่าง error)
  for (const s of ["train", "valid", "test"]) {
    if (perSplit[s] > 0) zip.file(`${s}/_annotations.coco.json`, JSON.stringify(coco[s]));
  }

  // RF-DETR export onnx ไม่ฝังชื่อคลาสมาด้วย — แนบไฟล์ชื่อคลาสไว้ที่ root ของ zip
  // ให้ผู้ใช้โหลดกลับเข้าแอป (ปุ่ม "โหลดชื่อคลาส" ในแผง Model Detect) หลังเทรนเสร็จ
  const yaml = `nc: ${classes.length}\nnames: [${classes.map((c) => JSON.stringify(c.name)).join(", ")}]\n`;
  zip.file("data.yaml", yaml);
  zip.file("classes.txt", classes.map((c) => c.name).join("\n") + "\n");

  const zipBlob = await zip.generateAsync({ type: "blob" });
  return {
    blob: zipBlob,
    stats: { totalImages: total, exported, skipped, boxes, skippedLines, augmented, classes: classes.length, perSplit, framework: "rfdetr" },
  };
}

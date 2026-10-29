// Registry ของ filter — แหล่งข้อมูลเดียว (single source of truth)
// จัดกลุ่มตามหัวข้อการสอน Image Processing:
//   1. Color Space      2. Enhancement
//   3. Restoration      4. Morphology
//   5. Segmentation   6. ประมวลผลภาพสี (Colour Processing)
//
// โครงสร้าง 2 ชั้น:
//   OPS      = แต่ละ operation ย่อย (grayscale, gaussian_blur, canny, ...)
//   FILTERS  = 1 กลุ่ม = 1 node เดียว มี select "เลือกวิธี" แล้วโชว์ param ของวิธีนั้น
//              (รวมหลาย op ในกลุ่มเดียวกันให้เป็น node เดียว)
//
// แต่ละ OP:
//   name    ชื่อแสดงผล (ไทย)
//   needs   'gray' = ต้องการภาพขาวดำ (worker แปลงให้อัตโนมัติ) | 'any'
//   desc    อธิบายหลักการทำงาน (เพื่อการเรียนรู้)
//   params  พารามิเตอร์ปรับได้ (สร้าง UI อัตโนมัติ) — key เป็น "local" ของ op
//   apply(cv, src, p, ctx)  คืน cv.Mat ใหม่ — ctx.info = ข้อความรายงานผล (ถ้ามี)
//
// แต่ละ FILTER (merged node):
//   name    ชื่อกลุ่มที่แสดงบน palette/node
//   group   หัวข้อวิชา
//   needs(p)  ฟังก์ชัน คืน 'gray'/'any' ตามวิธีที่เลือก (worker เรียกด้วย params ปัจจุบัน)
//   desc(p)   ฟังก์ชัน คืนคำอธิบายของวิธีที่เลือก
//   subtitle(p) คืนชื่อวิธีที่เลือก (โชว์ใต้หัวข้อ)
//   params  = [ mode select, ...sub-params ของทุก op (prefix ด้วย opId__, มี showIf) ]
//   apply   dispatch ไปยัง op.apply ตาม p.mode (ดึง sub-params ที่ prefix ตรงกัน)

function ensureOdd(n) {
  n = Math.round(n);
  return n % 2 === 0 ? n + 1 : n;
}

// strel เพิ่มเติมที่ cv.getStructuringElement ไม่มี (ไม่มีค่าคงที่ MORPH_DIAMOND/OCTAGON ใน OpenCV เลย
// แม้แต่ฝั่ง C++ ต้นฉบับ) — สร้าง binary mask เองแล้วส่งเป็น kernel ตรง ๆ (cv.erode/dilate/morphologyEx
// รับ kernel เป็น Mat ธรรมดา ไม่จำเป็นต้องมาจาก getStructuringElement เท่านั้น — ยืนยันแล้วว่าใช้ได้จริง)
function buildCustomStrel(cv, shape, k, lineAngleDeg) {
  // ksize ที่ไม่ใช่เลขจำกัด (NaN/Infinity) เป็นไปได้แค่จาก saved-project ที่เสียหาย (UI ปกติ clamp ค่าไว้แล้ว)
  // — กัน RangeError: Invalid array length จาก `new Array(NaN * NaN)` โดยไม่กระทบ rect/ellipse/cross เลย
  // (เส้นทางนั้นไม่เรียกฟังก์ชันนี้)
  if (!Number.isFinite(k)) k = 5;
  const r = Math.floor(k / 2);
  const size = 2 * r + 1;
  const data = new Array(size * size).fill(0);
  if (shape === "diamond") {
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      if (Math.abs(x - r) + Math.abs(y - r) <= r) data[y * size + x] = 1;
    }
  } else if (shape === "octagon") {
    // octagon = สี่เหลี่ยมตัดมุม — ตัดมุมออกประมาณ r/2.5 (สัดส่วนใกล้เคียง strel('octagon') ของ MATLAB)
    const cut = Math.round(r / 2.5);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const dx = Math.abs(x - r), dy = Math.abs(y - r);
      if (dx <= r && dy <= r && dx + dy <= 2 * r - cut) data[y * size + x] = 1;
    }
  } else if (shape === "line") {
    // เส้น 1 มิติผ่านจุดกลาง ตามมุมที่กำหนด (องศา, 0=แนวนอน) ความยาว = k
    const angle = ((Number.isFinite(lineAngleDeg) ? lineAngleDeg : 0) * Math.PI) / 180;
    const dx = Math.cos(angle), dy = Math.sin(angle);
    for (let t = -r; t <= r; t++) {
      const x = Math.round(r + t * dx), y = Math.round(r + t * dy);
      if (x >= 0 && x < size && y >= 0 && y < size) data[y * size + x] = 1;
    }
  }
  return cv.matFromArray(size, size, cv.CV_8UC1, data);
}

// fftshift/ifftshift (quadrant swap) — ใช้ทั้งก่อนสร้าง mask (ให้ความถี่ต่ำอยู่กลางภาพ) และก่อน inverse
// (ต้อง unshift กลับก่อน inverse เสมอ เพราะ inverse DFT คาดหวังความถี่ต่ำอยู่มุมเหมือนตอน forward)
// ใช้ได้ถูกต้องเฉพาะ mat มิติคู่ทั้งกว้าง/สูง — ผู้เรียกต้อง pad ให้เป็นเลขคู่ก่อนเรียก (ดู fft_magnitude/fft_filter)
function fftQuadrantSwap(cv, mat) {
  const cx = Math.floor(mat.cols / 2);
  const cy = Math.floor(mat.rows / 2);
  const q0 = mat.roi(new cv.Rect(0, 0, cx, cy));
  const q1 = mat.roi(new cv.Rect(cx, 0, mat.cols - cx, cy));
  const q2 = mat.roi(new cv.Rect(0, cy, cx, mat.rows - cy));
  const q3 = mat.roi(new cv.Rect(cx, cy, mat.cols - cx, mat.rows - cy));
  const tmp = new cv.Mat();
  q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
  q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
  tmp.delete(); q0.delete(); q1.delete(); q2.delete(); q3.delete();
}

// ใช้ร่วมกันโดย orb_keypoints/akaze_keypoints/fast_keypoints ด้านล่าง — อ่าน KeyPoint แต่ละตัวจาก
// kpv ครั้งเดียว (kpv.get(i) เป็น plain value object ไม่มี .delete()) เป็น plain object ธรรมดา แล้วจัด
// อันดับด้วย "total order" ที่ตายตัวไม่ขึ้นกับลำดับที่ detector คืนมา จากนั้นตัดเอาแค่ cap ตัวแรก
//
// เหตุผลที่ต้องมี tie-break ต่อจาก response (สำคัญ อย่าลบ): ยืนยันเชิงประจักษ์แล้วว่า opencv.js (libc++)
// กับ cv2 5.0.0 (MSVC STL) ให้ "ชุด" keypoint เดียวกันเป๊ะที่ความละเอียดเดียวกัน (0 set differences) แต่
// detector คืนมาคนละ "ลำดับ" กัน (nth_element ไม่ใช่ full sort ข้างใน ORB) — response มี tie เยอะมาก (นับ
// ร้อยจุดใน top-300 มีค่าซ้ำกัน) ถ้า sort ตาม response เพียงอย่างเดียวแล้วตัด top-N ตอนที่จุดตัดตกกลางกลุ่ม
// tie พอดี ฝั่ง JS กับฝั่ง Python (โค้ด export) จะเก็บคนละจุดกัน แม้จะรันบนภาพเดียวกันที่ความละเอียดเดียวกัน
// — ต้องมี total order ที่ deterministic เหมือนกันทั้งสองฝั่งเพื่อกันปัญหานี้ (ดู KEYPOINT_SORT_KEY ที่
// codegen.js ต้องตรงกับ comparator นี้เป๊ะทุก key/ทุกทิศทาง)
function rankKeypoints(kpv, cap) {
  const total = kpv.size();
  const all = new Array(total);
  for (let i = 0; i < total; i++) {
    const kp = kpv.get(i); // plain value object — ไม่มี .delete() ต้องไม่เรียก
    all[i] = { x: kp.pt.x, y: kp.pt.y, size: kp.size, angle: kp.angle, response: kp.response, octave: kp.octave };
  }
  all.sort((a, b) => {
    if (b.response !== a.response) return b.response - a.response; // response สูงสุดก่อน
    if (a.y !== b.y) return a.y - b.y; // แล้ว y น้อยสุดก่อน (บนก่อนล่าง)
    if (a.x !== b.x) return a.x - b.x; // แล้ว x น้อยสุดก่อน (ซ้ายก่อนขวา)
    if (b.size !== a.size) return b.size - a.size; // แล้ว size ใหญ่สุดก่อน
    if (a.angle !== b.angle) return a.angle - b.angle; // แล้ว angle น้อยสุดก่อน
    return a.octave - b.octave; // สุดท้าย octave น้อยสุดก่อน
  });
  return { total, kept: all.slice(0, cap) };
}

// ===================================================================
// regionprops (Tier 2, 18c) helpers
// ===================================================================
// สแกน point data ของ contour (CV_32SC2: x,y สลับกัน) ครั้งเดียว ได้ area + bbox_x/bbox_y + จุด "บนสุด-แล้ว-
// ซ้ายสุด" (ty,tx) โดยไม่ต้องเรียก cv.contourArea/cv.boundingRect ผ่าน embind ต่อชิ้น (วัดจริงที่ 1080p noise
// 10%/137,578 contours: 96ms เทียบ 116ms) — area ใช้ shoelace สูตรเดียวกับ cv::contourArea (ผลรวม
// x_i*y_{i+1} - y_i*x_{i+1} แล้ว |·|/2) บนพิกัดจำนวนเต็ม ทุกผลคูณ/ผลรวมเป็นจำนวนเต็มที่ double แทนได้เป๊ะ
// (พิกัดภาพ < 2^26) จึงเท่ากับ cv.contourArea ทุกบิต — ยืนยันแล้ว 0 mismatch จาก 224,181 contours และมีเทสต์
// regionprops-18 คุมไว้ (Python ใช้ cv2.contourArea ตรง ๆ ได้ค่าเดียวกัน)
// bbox_x/bbox_y = min x / min y ของจุดทั้งหมด = cv.boundingRect().x/.y เป๊ะ (boundingRect ของจุด integer)
// ty = min y = by เสมอ; tx = x น้อยสุดในแถวบนสุด → (by, tx) คือจุดแรกตาม row-major ของ contour ซึ่ง
// "ไม่ซ้ำกันโดยโครงสร้าง": จุดบน contour จาก findContours เป็นพิกเซล foreground ของวัตถุนั้นเอง และวัตถุ
// (connected component) ต่างชิ้นไม่มีพิกเซลร่วมกัน
export function regionScan(data) {
  const L = data.length;
  if (L === 0) return { area: 0, bx: 0, by: 0, tx: 0 };
  let s = 0, mx = Infinity, my = Infinity, tx = Infinity;
  for (let k = 0; k < L; k += 2) {
    const x = data[k], y = data[k + 1];
    const k2 = k + 2 === L ? 0 : k + 2;
    s += x * data[k2 + 1] - y * data[k2];
    if (x < mx) mx = x;
    if (y < my) { my = y; tx = x; } else if (y === my && x < tx) tx = x;
  }
  return { area: Math.abs(s) / 2, bx: mx, by: my, tx };
}

// total order ของแถว regionprops (qa 18c รอบ 1: 3 key เดิมไม่ total เพราะวัตถุสองชิ้นมี area, bbox_y, bbox_x
// เท่ากันได้จริง) — area DESC, bbox_y ASC, bbox_x ASC, cx ASC, cy ASC, แล้วจุดบนสุด-ซ้ายสุด (ty,tx) ASC ซึ่ง
// ไม่ซ้ำกันโดยโครงสร้าง (ดู regionScan) จึงไม่มีทางคืน 0 ระหว่างวัตถุต่างชิ้น — ty เท่ากับ by เสมอ (เทียบไป
// แล้วที่ key ที่ 2) จึงเหลือ tx เป็น key สุดท้าย · centroidOf(rec) เติม rec.cx/rec.cy แบบ lazy (moments แพง)
// เรียกเฉพาะเมื่อ 3 key แรกเสมอกัน — ลำดับผลลัพธ์เหมือนคำนวณล่วงหน้าทุกชิ้นเป๊ะ · codegen.js ต้องใช้ key
// ชุดเดียวกันทุกตัว/ทุกทิศทาง: (-area, by, bx, cx, cy, tx)
export function compareRegionCandidates(a, b, centroidOf) {
  if (b.area !== a.area) return b.area - a.area;
  if (a.by !== b.by) return a.by - b.by;
  if (a.bx !== b.bx) return a.bx - b.bx;
  centroidOf(a);
  centroidOf(b);
  if (a.cx !== b.cx) return a.cx - b.cx;
  if (a.cy !== b.cy) return a.cy - b.cy;
  return a.tx - b.tx;
}

// ภาพเล็กเกินไป (เช่น Crop ซ้อน Crop จนเหลือ 1-2px ด้านใดด้านหนึ่ง) — ยืนยันแล้วว่า ORB/AKAZE's .detect()
// throw C++ exception ตรง ๆ เมื่อด้านใดด้านหนึ่งของภาพวิเคราะห์ <3px (2x2 ปลอดภัย แต่ 1xN/Nx1 ไม่) และการ
// throw แต่ละครั้งรั่วหน่วยความจำเล็กน้อย "ภายใน" opencv.js เอง (นอกเหนือการควบคุมของ try/finally ฝั่งเรา)
// — ต้องเช็คมิติแล้วข้ามการเรียก detector ไปเลยก่อนที่จะมีโอกาส throw ไม่ใช่ปล่อยให้ throw แล้วมาจับทีหลัง
function tooSmallForKeypoints(analysis) {
  return analysis.rows < 3 || analysis.cols < 3;
}
const KEYPOINTS_TOO_SMALL_INFO = "ภาพเล็กเกินไปสำหรับหาจุดเด่น (ต้องกว้างและสูงอย่างน้อย 3 px)";

// ภาพสองใบต้องมีขนาดเท่ากันก่อนทำ arithmetic ระหว่างภาพ (imabsdiff ฯลฯ) ถ้าขนาดไม่ตรงกัน resize
// secondary ให้เท่ากับ primary ก่อนเสมอ (แจ้งผู้ใช้ผ่าน ctx.info) แทนการปฏิเสธ/throw — ใช้งานง่ายกว่ามาก
// สำหรับผู้เรียนที่ต่อสาย 2 เส้นจากจุดต่างขนาดกันในกราฟโดยไม่ตั้งใจ (เช่น เส้นหนึ่งผ่าน Crop อีกเส้นไม่ผ่าน)
// คืน { mat, resized } — mat เป็น secondary เดิม (resized:false, ห้าม delete ซ้ำ) หรือ Mat ใหม่ที่ต้อง
// delete เอง (resized:true) — caller ต้องเช็ค resized ก่อน delete เสมอ (ดู imabsdiff.apply ด้านล่าง)
function matchSize(cv, primary, secondary) {
  if (primary.cols === secondary.cols && primary.rows === secondary.rows) return { mat: secondary, resized: false };
  const out = new cv.Mat();
  try {
    cv.resize(secondary, out, new cv.Size(primary.cols, primary.rows), 0, 0, cv.INTER_LINEAR);
  } catch (err) {
    out.delete(); // cv.resize throw กลางทาง (ขนาดผิดปกติ/heap เต็ม) ต้องไม่ปล่อย out ลอยค้าง
    throw err;
  }
  return { mat: out, resized: true };
}

// ===================================================================
// ตัวช่วยร่วมของ op สองภาพรุ่นใหม่ (IP batch 2 ชิ้น B: imarith/logical_ops/inpaint/hist_match ขึ้นไป —
// batch ถัดไปที่เพิ่ม op สองภาพให้เรียกตัวนี้ซ้ำ ไม่ต้องคัดลอก) · imabsdiff/imfuse/mask รุ่นเก่าคงโค้ดเดิมไว้
// ===================================================================
const NO_SRC2_INFO = "ยังไม่ได้ต่อสาย 'เข้า B' — ลากเส้นจากอีกจุดหนึ่งมาต่อที่ช่องอินพุตที่สองของกล่องนี้";
const EMPTY_SRC2_INFO = "ภาพที่ต่อเข้า 'เข้า B' ว่าง (0×0 px) — ตรวจขั้นตอนในสาย B ว่าไม่ได้ตัดภาพจนหมด";
const RESIZED_B_INFO = " (ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)";
// op ที่ตัดภาพที่ 127 (logical_ops/inpaint) ต่อท้ายข้อความนี้เมื่อได้ภาพสี — ผลของ Color InRange ค่าเริ่มต้น ("เฉพาะส่วนที่เป็นสีนั้น")
// เป็นภาพสี สีเข้มอย่างแดง/น้ำเงินมีค่าเทาต่ำกว่า 127 จึงกลายเป็นดำทั้งที่ InRange คัดไว้แล้ว (debugger 2026-10-03)
const COLOUR_CUT_INFO = " · ภาพสีถูกแปลงเป็นขาวดำแล้วตัดที่ 127 — ถ้ามาจาก Color InRange ให้ตั้ง 'แสดงผลเป็น' = Mask ขาวดำ";

// ยังไม่มีภาพ B ที่ใช้ได้ (ไม่ได้ต่อสาย หรือ B เป็น 0×0 ซึ่ง cv.resize/คำนวณต่อจะ throw C++ exception) →
// คืนสำเนา A (Mat ใหม่ที่ผู้เรียกเป็นเจ้าของ) + ข้อความอธิบาย · มี B ใช้ได้ → คืน null ให้ op ทำงานต่อ
// ใช้: `const pass = passthroughNoSrc2(src, ctx); if (pass) return pass;` — หลังบรรทัดนี้รับประกันว่า ctx
// และ ctx.src2 มีอยู่จริง · ctx.src2 เป็นของยืมจาก runPipeline ห้าม delete
function passthroughNoSrc2(src, ctx) {
  const hasB = Boolean(ctx && ctx.src2);
  if (hasB && !isEmptyMat(ctx.src2)) return null;
  if (ctx) ctx.info = hasB ? EMPTY_SRC2_INFO : NO_SRC2_INFO;
  return src.clone();
}

// คู่ของ grayIfNeeded ด้านล่าง: คืน Mat RGB 3 ช่องที่ "ผู้เรียกเป็นเจ้าของ" หรือ null ถ้า m เป็น 3 ช่องอยู่แล้ว
// (ใช้ m ตรง ๆ ไม่คัดลอกเปล่า ๆ) · สัญญาเดียวกับ grayIfNeeded: รองรับเฉพาะ 1 / 3 / 4 ช่อง
function rgbIfNeeded(cv, m) {
  if (m.channels() === 3) return null;
  const out = new cv.Mat();
  try {
    cv.cvtColor(m, out, m.channels() === 4 ? cv.COLOR_RGBA2RGB : cv.COLOR_GRAY2RGB);
  } catch (err) {
    out.delete();
    throw err;
  }
  return out;
}

// imarith: ทุกตัว saturate ใน 8 บิต (เกิน 255 → 255, ติดลบ → 0) ไม่วนกลับแบบ uint8 ของ numpy ·
// cv.add(Mat, Scalar) throw ในบิลด์นี้ จึงใช้แบบ Mat กับ Mat เสมอ · หาร 0 ได้ 0 (ตรวจแล้วตรงกับ cv2)
const IMARITH_OPS = {
  add: { info: "A + B (เกิน 255 ตัดที่ 255)", run: (cv, a, b, dst) => cv.add(a, b, dst) },
  subtract: { info: "A − B (ติดลบตัดเป็น 0 — ถ้าต้องการความต่างทั้งสองทางให้ใช้ Absolute Difference)", run: (cv, a, b, dst) => cv.subtract(a, b, dst) },
  multiply: { info: "A × B ÷ 255 (B ขาว = คงค่า A, B ดำ = 0)", run: (cv, a, b, dst) => cv.multiply(a, b, dst, 1 / 255) },
  divide: { info: "A ÷ B × 255 (ตรงที่ B = 0 ได้ 0)", run: (cv, a, b, dst) => cv.divide(a, b, dst, 255) },
};

// logical_ops: รับภาพขาวดำ 0/255 สองใบ (threshold 127 แล้ว) · notB = ต้องกลับด้าน B ก่อน (A AND NOT B)
const LOGICAL_OPS = {
  and: { info: "A AND B — ขาวเฉพาะที่ขาวทั้งคู่", run: (cv, a, b, dst) => cv.bitwise_and(a, b, dst) },
  or: { info: "A OR B — ขาวถ้าขาวอย่างน้อยหนึ่งภาพ", run: (cv, a, b, dst) => cv.bitwise_or(a, b, dst) },
  xor: { info: "A XOR B — ขาวเฉพาะที่ขาวภาพเดียว", run: (cv, a, b, dst) => cv.bitwise_xor(a, b, dst) },
  a_and_not_b: { info: "A AND NOT B — หัก B ออกจาก A (ขาวเฉพาะที่ A ขาวแต่ B ดำ)", notB: true, run: (cv, a, b, dst) => cv.bitwise_and(a, b, dst) },
};

// inpaint: เวลาประมวลผลโตตามพื้นที่ mask × รัศมี² — mask ที่ครอบเกินนี้ (%) ไม่ซ่อม กันกล้องสดค้างเป็นวินาที
const INPAINT_MAX_MASK_PCT = 25;

// hist_match: histogram 256 ระดับ (Float32Array จาก calcHist — นับเป็นจำนวนเต็ม) → CDF สะสมแบบ float64
// (number ของ JS) หารด้วยจำนวนพิกเซลทั้งหมด · ตรงกับ np.cumsum(h)/h.sum() หลัง .astype(np.float64) ในโค้ดที่ export
function cdf256(h) {
  let total = 0;
  for (let i = 0; i < 256; i++) total += h[i];
  const cdf = new Float64Array(256);
  let cum = 0;
  for (let i = 0; i < 256; i++) {
    cum += h[i];
    cdf[i] = cum / total;
  }
  return cdf;
}

// ===================================================================
// ตัวช่วยของ IP batch 2 ชิ้น C (gaussian_noise / wiener_filter / translate_shear)
// ===================================================================
// PRNG แบบมี seed ตัวเดียวของทั้งไฟล์ (mulberry32: state 32 บิต เพิ่มทีละ 0x6D2B79F5 แล้ว hash) — noise แบบอื่นในอนาคต
// (speckle/poisson) ต้องใช้ตัวนี้/seededGaussian ซ้ำ ห้ามเขียน PRNG ใหม่ · ห้ามใช้ cv.randn (ส่ง Scalar/ตัวเลขแล้ว abort
// ทั้ง process และ RNG ของ cv2 กับ opencv.js ให้ค่าไม่ตรงกัน) · ตัวที่ i (นับจาก 1) = hash((seed + i·0x6D2B79F5) mod 2^32)
// codegen.js (seededGaussianPy) คำนวณแบบ vectorized จากสูตรนี้ตรง ๆ จึงได้ค่าเท่ากันทุก bit — ห้ามเปลี่ยนลำดับ/ค่าคงที่
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
}

// ค่าสุ่มแบบเกาส์มาตรฐาน (mean 0, sd 1) n ค่าจาก seed — Box-Muller ทีละคู่: u1 = (r+1)/2^32 อยู่ในช่วง (0, 1] กัน log(0),
// u2 = r/2^32 · z[2j] = rad·cos(2π·u2), z[2j+1] = rad·sin(2π·u2) · n คี่ = ทิ้งค่า sin ของคู่สุดท้าย
export function seededGaussian(seed, n) {
  const next = mulberry32(seed);
  const z = new Float64Array(n);
  for (let j = 0; j < n; j += 2) {
    const u1 = (next() + 1) / 4294967296;
    const u2 = next() / 4294967296;
    const rad = Math.sqrt(-2 * Math.log(u1));
    const ang = 2 * Math.PI * u2;
    z[j] = rad * Math.cos(ang);
    if (j + 1 < n) z[j + 1] = rad * Math.sin(ang);
  }
  return z;
}

// ภาพ 4 ช่อง (RGBA) → Mat RGB 3 ช่องที่ "ผู้เรียกเป็นเจ้าของ" · ภาพ 1/3 ช่องคืน null (ใช้ m ตรง ๆ ไม่คัดลอกเปล่า ๆ)
// ต่างจาก rgbIfNeeded ตรงที่ภาพขาวดำคงเป็น 1 ช่อง — op ชิ้น C ทำงานได้ทั้งขาวดำและสี ไม่ต้องขยายเป็นสี
function dropAlpha(cv, m) {
  if (m.channels() !== 4) return null;
  const out = new cv.Mat();
  try {
    cv.cvtColor(m, out, cv.COLOR_RGBA2RGB);
  } catch (err) {
    out.delete();
    throw err;
  }
  return out;
}

// wiener_filter: ksize สูงสุด 15 — ผลรวมกำลังสองในหน้าต่างเก็บใน float32 ซึ่งแทนจำนวนเต็มได้เป๊ะถึง 2^24 เท่านั้น
// (15²·255² = 14,630,625 < 16,777,216 แต่ 17²·255² = 18,792,225 เกิน) ห้ามเพิ่ม max โดยไม่เปลี่ยนเป็น float64
// translate_shear: |1 − kx·ky| ต่ำกว่านี้ = matrix เกือบ singular ภาพยุบเป็นเส้น → ไม่ warp
const SHEAR_SINGULAR_LIMIT = 0.05;
// ข้อความ % ทศนิยม 1 ตำแหน่งที่ปัดครึ่งขึ้นเหมือน Python ที่ export (np.floor(x*10+0.5)/10 แล้ว :.1f) — toFixed กับ :.1f
// ปัดค่าที่ลงครึ่งพอดีต่างกัน
function fixed1(x) { return (Math.floor(x * 10 + 0.5) / 10).toFixed(1); }
// ทศนิยม 2 ตำแหน่งแบบปัดครึ่งขึ้น (ตัวเลข shear ของ translate_shear) — Python ที่ export ใช้ np.floor(x*100+0.5)/100 แล้ว :.2f
// ให้ข้อความตรงกัน (toFixed(2) ของ −0.125 ได้ "-0.13" แต่ :.2f ของ 0.125 ได้ "0.12" — ปัดค่าที่ลงครึ่งพอดีไม่เหมือนกัน) ·
// ui-meta.js ใช้ตัวนี้ปัดตัวเลขบนการ์ดด้วย จึงเป็นกติกาเดียวทั้งแอป · ไม่มีวันได้ "-0.00" (floor ของค่า ≥ 0 ไม่ใช่ −0)
export function fixed2(x) { return (Math.floor(x * 100 + 0.5) / 100).toFixed(2); }

// B7 (backlog 2026-10-17, P-13 + P-17) — ตัวอ่าน param กลาง (แอปและ codegen ใช้ตัวเดียวกัน จึงให้ผลตรงกันแม้ param เป็น string จากไฟล์ที่แก้มือ)
// finiteOr: เลขจำกัด → ตัวนั้น · สตริงตัวเลข ("0", " 2.5 ") → เป็นเลข · อื่น ๆ (undefined/null/""/NaN/Infinity/boolean/object) → ค่า default d
export function finiteOr(v, d) {
  if (typeof v === "number") return Number.isFinite(v) ? v : d;
  if (typeof v === "string" && v.trim() !== "") { const x = Number(v); return Number.isFinite(x) ? x : d; }
  return d;
}
// boolOr: true/false · "true"/"false" · 1/0 → boolean · อื่น ๆ → d (เดิมใช้ truthiness ทำให้ "false" กลายเป็น true เงียบ ๆ)
export function boolOr(v, d) {
  if (v === true || v === 1 || v === "true") return true;
  if (v === false || v === 0 || v === "false") return false;
  return d;
}

// ใช้ร่วมกันโดย op ที่ต้องย่อภาพวิเคราะห์ก่อนประมวลผลหนัก (Tier 2, 18b) — เหมือน harris_corners/
// fft_magnitude ที่มีอยู่เดิมแต่เขียน inline ซ้ำ ๆ ทุกจุด — คืน {scale, invScale, w, h} ของภาพย่อ ไม่ resize
// ให้เองตรงนี้ (ผู้เรียกยังต้องเรียก cv.resize เอง เพราะบาง op เช่น template_matching ต้องย่อภาพ 2 ใบ
// พร้อมกันด้วย scale เดียวกัน)
function analysisScale(cols, rows, maxSide) {
  const scale = Math.min(1, maxSide / Math.max(cols, rows));
  return {
    scale,
    invScale: 1 / scale,
    w: Math.max(1, Math.round(cols * scale)),
    h: Math.max(1, Math.round(rows * scale)),
  };
}

// สร้าง mask ตารางหมากรุก (CV_8UC1, 0/255) สำหรับ imfuse โหมด checkerboard — ไม่มี cv call เดียวที่ทำสิ่งนี้
// ได้ตรง ๆ (ยืนยันแล้ว) จึงสร้างเอง เหมือนแนวทางเดียวกับ buildCustomStrel ด้านบน (เขียนลง TypedArray ตรง ๆ
// แล้วส่งเข้า cv.matFromArray) — ใช้ .fill() เป็นช่วง ๆ ต่อ tile แทนการวน per-pixel เพื่อความเร็ว (วัดจริงใน
// Node ก่อนเขียนโค้ดนี้: 1920x1080 tile=32 ≈0.95ms, tile=8 คือเล็กสุดที่ UI อนุญาต ≈3ms ต่อครั้ง — เร็วพอ
// ไม่ต้อง cache/ย่อภาพวิเคราะห์เลย)
function buildCheckerboardMask(cv, cols, rows, tile) {
  const data = new Uint8Array(cols * rows);
  for (let y = 0; y < rows; y++) {
    const tileY = (y / tile) | 0;
    const rowOffset = y * cols;
    let x = 0;
    while (x < cols) {
      const tileX = (x / tile) | 0;
      const on = ((tileX + tileY) & 1) === 0;
      const endX = Math.min(cols, (tileX + 1) * tile);
      if (on) data.fill(255, rowOffset + x, rowOffset + endX);
      x = endX;
    }
  }
  return cv.matFromArray(rows, cols, cv.CV_8UC1, data);
}

// ===================================================================
// ตัวช่วยของกลุ่ม Segmentation & Edge (2026-09-30 — prewitt/roberts/log_edge/region_growing)
// ===================================================================
// ภาพว่าง (0x0) ทำให้ filter2D/GaussianBlur/floodFill throw C++ exception ซึ่ง "รั่ว ~760 B ในตัว opencv.js เอง
// ทุกครั้งที่ throw" แก้ในแอปไม่ได้ (WORK-TRACKER §1) → op ต้องกันไม่ให้ถึงจุด throw ไม่ใช่จับทีหลัง
function isEmptyMat(m) { return m.rows < 1 || m.cols < 1; }

// คืน Mat ขาวดำ 1 ช่องที่ "ผู้เรียกเป็นเจ้าของและต้อง delete เอง" — ถ้า src เป็น 1 ช่องอยู่แล้วคืน null เพื่อให้
// ผู้เรียกใช้ src ตรง ๆ (ไม่จอง/คัดลอกเปล่า ๆ ทุกเฟรม) แนวเดียวกับ matchSize ที่คืน { mat, resized } ให้เช็คก่อน
// delete · จำเป็นกับ op ที่อ่านบัฟเฟอร์เองทีละพิกเซล (log_edge) หรือที่ OpenCV throw กับ 4 ช่อง (region_growing)
//
// *** สัญญาของฟังก์ชัน (ห้ามเรียกนอกเงื่อนไขนี้) ***: รองรับ src ที่มี 1 / 3 / 4 ช่องเท่านั้น
// Mat 2 ช่อง (เช่นผลของ cv.merge สองชั้น หรือ CV_8UC2 ที่ op สร้างเอง) จะตกมาที่ COLOR_RGB2GRAY แล้ว
// throw C++ exception ซึ่งรั่ว ~760 B ทุกครั้งที่ throw (WORK-TRACKER §1 — แก้ในแอปไม่ได้)
// ตอนนี้เข้าไม่ถึงจาก runPipeline เพราะ cv-worker ส่งมาแค่ 1/3/4 ช่อง แต่ helper นี้เป็นของกลางระดับไฟล์
// → op ใหม่ที่จะหยิบไปใช้กับ Mat ที่สร้างเอง ต้องเช็ค src.channels() ให้อยู่ใน {1, 3, 4} ก่อนเรียกเสมอ
function grayIfNeeded(cv, src) {
  if (src.channels() === 1) return null;
  const g = new cv.Mat();
  try {
    cv.cvtColor(src, g, src.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY);
  } catch (err) {
    g.delete(); // cvtColor throw กลางทาง ต้องไม่ปล่อย g ลอยค้าง
    throw err;
  }
  return g;
}

// ===================================================================
// ตัวช่วยของกลุ่ม Motion (Tier 2, 18d — mog2 / farneback_optical_flow / pyrlk_optical_flow)
// ===================================================================
// ตัวเลขจาก param (อาจเป็น string จาก saved project — บทเรียน P-13/18c) — 0 เป็นค่าที่ใช้ได้
function numOr(v, d) { return finiteOr(v, d); }
// clamp ให้อยู่ใน "ช่วงเดียวกับ min/max ของ param นั้นใน UI" เสมอ (ไฟล์โปรเจกต์ที่แก้มือแล้วใส่ค่าเกินช่วง
// ต้องไม่หลุดเข้า OpenCV) — ตัวเลข lo/hi ทุกจุดถูกล็อกไว้กับ params ด้วยเทสต์ tier2d-MN-1 (static read)
function numIn(v, d, lo, hi) { return Math.min(hi, Math.max(lo, numOr(v, d))); }
function intIn(v, d, lo, hi) { return Math.round(numIn(v, d, lo, hi)); }
// ภาพวิเคราะห์ขาวดำ 1 ช่อง ขนาดตาม analysisScale (a) — caller delete เอง รองรับ input 1/3/4 ช่อง
// (optical flow ทั้งสองแบบ throw C++ ถ้าได้ภาพ 3 ช่อง — V4/V5 — จึงต้องผ่านตัวนี้เสมอ)
function motionGray(cv, src, a) {
  let g = new cv.Mat();
  try {
    if (src.channels() === 1) src.copyTo(g);
    else cv.cvtColor(src, g, src.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY);
    if (a.scale < 1) {
      const r = new cv.Mat();
      try { cv.resize(g, r, new cv.Size(a.w, a.h), 0, 0, cv.INTER_AREA); } catch (e) { r.delete(); throw e; }
      g.delete(); g = null;
      return r;
    }
    const out = g; g = null;
    return out;
  } finally { if (g) g.delete(); }
}
// สำเนา RGB 3 ช่องของ src (ใช้วาดทับ) — caller delete เอง
function motionRGB(cv, src) {
  const o = new cv.Mat();
  try {
    if (src.channels() === 1) cv.cvtColor(src, o, cv.COLOR_GRAY2RGB);
    else if (src.channels() === 4) cv.cvtColor(src, o, cv.COLOR_RGBA2RGB);
    else src.copyTo(o);
    return o;
  } catch (e) { o.delete(); throw e; }
}

// ===================================================================
// ตัวช่วยของ IP batch 2 ชิ้น D (2026-10-03 — feret / line_profile / roi_polygon) · ใช้ร่วมกับเครื่องมือวัดตัวต่อไปได้
// ===================================================================
// % → พิกเซล: 0% = พิกเซลแรก, 100% = พิกเซลสุดท้าย (ใช้ n − 1 แบบเดียวกับ region_growing) · ปัดด้วย floor(x + 0.5) ซึ่งเป็นนิพจน์เดียวกับ
// Python ที่ export `int(pct / 100.0 * (n - 1) + 0.5)` (x ≥ 0 เสมอ int() จึงเท่ากับ floor) — ไม่แตะ region_growing (ยังใช้ Math.round เดิม)
function pctToPx(pct, d, n) {
  return Math.floor((numIn(pct, d, 0, 100) / 100) * (n - 1) + 0.5);
}

// จุดศูนย์กลางของ contour สูตรเดียวกับ regionprops (setCentroid) เป๊ะ: moments · m00 = 0 (วัตถุ 1 px / เส้นบาง) → กลางกรอบ boundingRect
// — ตัวเลข cx/cy ของ feret จึงเท่ากับตาราง Region Properties ทุกแถว (ไม่แตะ regionprops)
function contourCentroid(cv, c) {
  const m = cv.moments(c); // plain object ไม่มี .delete()
  if (m.m00 === 0) {
    const r = cv.boundingRect(c);
    return { cx: r.x + r.width / 2, cy: r.y + r.height / 2 };
  }
  return { cx: m.m10 / m.m00, cy: m.m01 / m.m00 };
}

// ===================================================================
// ตัวช่วยของ IP batch 3 (2026-10-14 — regionprops +8 คอลัมน์ / shape_filter / quantize / interpolation)
// ===================================================================
// IP batch 3 — คอลัมน์ใหม่ของ Region Properties (ลำดับนี้ = ลำดับในตาราง) · shape_filter ใช้ key เดียวกัน
const REGION_BASE_COLUMNS = ["index", "area", "cx", "cy", "perimeter", "circularity", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "orientation_deg"];
export const REGION_SHAPE_COLUMNS = ["area_px", "equiv_diameter", "extent", "solidity", "major_axis", "minor_axis", "eccentricity", "mean_intensity"];
const REGION_COLUMNS_ALL = [...REGION_BASE_COLUMNS, ...REGION_SHAPE_COLUMNS];
const REGION_MEAN_COL = REGION_COLUMNS_ALL.indexOf("mean_intensity");

// กรอบล้อมของจุด contour (จำนวนเต็ม) = cv.boundingRect เป๊ะ โดยไม่เรียก embind
export function regionBox(d) {
  let mx = Infinity, my = Infinity, Mx = -Infinity, My = -Infinity;
  for (let k = 0; k < d.length; k += 2) {
    const x = d[k], y = d[k + 1];
    if (x < mx) mx = x;
    if (x > Mx) Mx = x;
    if (y < my) my = y;
    if (y > My) My = y;
  }
  return { x: mx, y: my, w: Mx - mx + 1, h: My - my + 1 };
}

// ช่วง x (รวมปลาย) ของ "จุดกลางพิกเซล" ที่อยู่ในหรือบนรูปนูน ต่อแถว — เลขจำนวนเต็มล้วน ไม่ขึ้นกับการระบายของ fillPoly
// hd = hull.data32S (x0,y0,x1,y1,...) · box = กรอบของ contour (รูปนูนอยู่ในกรอบเสมอ)
export function hullRowSpans(hd, box) {
  const m = hd.length / 2;
  const lo = new Int32Array(box.h).fill(2147483647), hi = new Int32Array(box.h).fill(-2147483648);
  for (let i = 0; i < m; i++) {
    const j = i + 1 === m ? 0 : i + 1;
    const ax = hd[2 * i], ay = hd[2 * i + 1], bx = hd[2 * j], by = hd[2 * j + 1];
    if (ay === by) { // ขอบแนวนอน (รวมรูปนูนจุดเดียว)
      const r = ay - box.y, a = Math.min(ax, bx), b = Math.max(ax, bx);
      if (a < lo[r]) lo[r] = a;
      if (b > hi[r]) hi[r] = b;
      continue;
    }
    const y0 = Math.min(ay, by), y1 = Math.max(ay, by), dx = bx - ax;
    for (let y = y0; y <= y1; y++) {
      let num = ax * (by - ay) + (y - ay) * dx, den = by - ay;
      if (den < 0) { num = -num; den = -den; }
      const fl = Math.floor(num / den), ce = -Math.floor(-num / den); // |num| เล็ก, den ≤ จำนวนแถว → floor ของ double = floor จริง
      const r = y - box.y;
      if (ce < lo[r]) lo[r] = ce;
      if (fl > hi[r]) hi[r] = fl;
    }
  }
  return { lo, hi };
}

// ผลรวมดิบต่อวัตถุ: พิกเซลของวัตถุ = อยู่ในเส้นขอบนอก (fillPoly) AND เป็นสีขาวใน bin (รูไม่นับ) · พิกัดนับจากมุมกรอบ
// needHull = true → hullN = จำนวนพิกเซลของ "รูปนูน ∪ วัตถุ" (นับด้วยเลขจำนวนเต็ม) · false → null
export function regionPixelStats(cv, c, box, binData, grayData, W, needHull) {
  let obj = null, vec = null, hull = null;
  try {
    obj = cv.Mat.zeros(box.h, box.w, cv.CV_8UC1);
    vec = new cv.MatVector();
    vec.push_back(c);
    cv.fillPoly(obj, vec, new cv.Scalar(255), cv.LINE_8, 0, new cv.Point(-box.x, -box.y));
    let spans = null;
    if (needHull) {
      hull = new cv.Mat();
      cv.convexHull(c, hull, false, true);
      spans = hullRowSpans(hull.data32S, box);
    }
    const od = obj.data;
    let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sum = 0, outside = 0;
    for (let y = 0; y < box.h; y++) {
      const row = (box.y + y) * W + box.x, orow = y * box.w;
      for (let x = 0; x < box.w; x++) {
        if (od[orow + x] === 0 || binData[row + x] === 0) continue;
        n++; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; sum += grayData[row + x];
        if (spans !== null && (box.x + x < spans.lo[y] || box.x + x > spans.hi[y])) outside++;
      }
    }
    let hullN = null;
    if (spans !== null) {
      hullN = outside;
      for (let r = 0; r < box.h; r++) if (spans.hi[r] >= spans.lo[r]) hullN += spans.hi[r] - spans.lo[r] + 1;
    }
    return { n, sx, sy, sxx, syy, sxy, sum, hullN };
  } finally {
    if (obj) obj.delete();
    if (vec) vec.delete();
    if (hull) hull.delete();
  }
}

// ค่าแบบ MATLAB regionprops (วงรีที่มี second moment เท่ากัน + 1/12 ต่อพิกเซล) · ตัวเศษเป็น BigInt (เกิน 2^53 ได้ที่ 1080p)
// Number(BigInt) ของ JS กับ float(int) ของ Python ปัดแบบเดียวกัน (nearest-even) → เท่ากันทุก bit
export function regionShapeValues(s, box) {
  const n = s.n, nn = n * n, B = BigInt;
  const cxx = Number(B(n) * B(s.sxx) - B(s.sx) * B(s.sx)) / nn + 1 / 12;
  const cyy = Number(B(n) * B(s.syy) - B(s.sy) * B(s.sy)) / nn + 1 / 12;
  const cxy = Number(B(n) * B(s.sxy) - B(s.sx) * B(s.sy)) / nn;
  const common = Math.sqrt((cxx - cyy) * (cxx - cyy) + 4 * cxy * cxy);
  const major = 2 * Math.SQRT2 * Math.sqrt(cxx + cyy + common);
  const minor = 2 * Math.SQRT2 * Math.sqrt(cxx + cyy - common);
  const ecc = 2 * Math.sqrt(Math.max(0, (major / 2) * (major / 2) - (minor / 2) * (minor / 2))) / major;
  return {
    area_px: n, equiv_diameter: Math.sqrt(4 * n / Math.PI), extent: n / (box.w * box.h),
    solidity: s.hullN === null ? null : n / s.hullN,
    major_axis: major, minor_axis: minor, eccentricity: ecc, mean_intensity: s.sum / n,
  };
}

// ช่วง 0–255 แบ่ง N ช่วงเท่ากัน · แต่ละช่วงแทนด้วยค่าที่กระจายเต็ม 0–255 (ปัดครึ่งขึ้น, เลขจำนวนเต็มล้วน)
export function quantizeLut(N) {
  const t = new Uint8Array(256);
  for (let v = 0; v < 256; v++) t[v] = Math.floor((Math.floor((v * N) / 256) * 510 + (N - 1)) / (2 * (N - 1)));
  return t;
}
export function quantizeBits(N) { let b = 0; while ((1 << b) < N) b++; return b; }

// IP batch 3 — วิธี interpolation ที่เลือกได้ · "auto" ไม่อยู่ในตาราง = ใช้วิธีเดิมของแต่ละ op (ไฟล์โปรเจกต์/โค้ด export เดิมไม่เปลี่ยน)
const INTERP_FLAG = { nearest: "INTER_NEAREST", bilinear: "INTER_LINEAR", bicubic: "INTER_CUBIC" };
const INTERP_NAME = { nearest: "Nearest neighbour", bilinear: "Bilinear", bicubic: "Bicubic" };
function interpParam(autoLabel) {
  return { key: "interp", type: "select", label: "วิธีคำนวณพิกเซลใหม่ (interpolation)", options: [
    { value: "auto", label: autoLabel },
    { value: "nearest", label: "Nearest neighbour (ใช้พิกเซลเดิมที่อยู่ใกล้ — เห็นเป็นบล็อก/ขั้นบันได)" },
    { value: "bilinear", label: "Bilinear (เฉลี่ย 2×2 พิกเซลรอบจุด — นุ่ม)" },
    { value: "bicubic", label: "Bicubic (ใช้ 4×4 พิกเซล — คมกว่า Bilinear ขอบอาจเกินเป็นเส้นจาง ๆ)" },
  ], default: "auto" };
}
function interpKey(v) { return Object.hasOwn(INTERP_FLAG, v) ? v : "auto"; }

// shape_filter: ชื่อไทยของ prop (ใช้ในข้อความ info) · view · ฟิลด์ scalars 4 ตัว (ลำดับตายตัว)
export const SHAPE_FILTER_PROP_TH = { circularity: "ความกลม", solidity: "ความตัน", eccentricity: "ความรี", extent: "เต็มกรอบ", none: "" };
const SHAPE_FILTER_VIEWS = { overlay: true, mask: true };
const SHAPE_FILTER_FIELDS = [
  ["sf_total", "วัตถุทั้งหมด (ชิ้น)"],
  ["sf_kept", "เก็บไว้ (ชิ้น)"],
  ["sf_removed_area", "คัดทิ้งเพราะพื้นที่ (ชิ้น)"],
  ["sf_removed_shape", "คัดทิ้งเพราะรูปร่าง (ชิ้น)"],
];

// มุมของทิศทาง (dx, dy) ที่ normalise แล้ว (dy > 0 หรือ dy = 0 กับ dx > 0) → 0–180° · 0° = แนวนอน เพิ่มตามเข็มนาฬิกา (แกน y ของภาพชี้ลง)
function feretAngle(dx, dy) {
  let a = (Math.atan2(dy, dx) * 180) / Math.PI;
  if (a < 0) a += 180;
  if (a >= 180) a = 0;
  return a;
}

// Feret ของ contour หนึ่งชิ้น "บนขอบพิกเซล": พิกเซล (x, y) = สี่เหลี่ยม [x−½, x+½] × [y−½, y+½] → มุมทั้ง 4 คูณ 2 ให้เป็นจำนวนเต็ม
// (2x ± 1, 2y ± 1) → convex hull (≥ 4 จุดเสมอ) → brute force O(h²) ด้วยเลขจำนวนเต็มล้วน (ระยะเท่ากับ Python ทุก bit)
// กลับทิศให้อยู่ในครึ่งระนาบเดียว: dy < 0 หรือ (dy = 0 และ dx < 0) → กลับ · กลับด้วย `0 - v` (ไม่ใช้ -v) กัน −0 หลุดไปเป็นมุม "-0"
// เสมอกัน → เลือกมุมเล็กกว่า: แทนเมื่อ cross(เดิม, ใหม่) < 0 · คืนพิกัดวาดเป็นพิกเซลแล้ว (Python: `(v + 1) // 2`, `np.floor(X / 2 + 0.5)`)
function feretOfContour(cv, c) {
  const d = c.data32S;
  const n = d.length >> 1;
  const corners = new Int32Array(n * 8); // ลำดับเดียวกับ Python: มุม 4 จุดของพิกเซลที่ 1, ของพิกเซลที่ 2, …
  for (let k = 0; k < n; k++) {
    const x = 2 * d[2 * k], y = 2 * d[2 * k + 1], o = k * 8;
    corners[o] = x - 1; corners[o + 1] = y - 1;
    corners[o + 2] = x + 1; corners[o + 3] = y - 1;
    corners[o + 4] = x + 1; corners[o + 5] = y + 1;
    corners[o + 6] = x - 1; corners[o + 7] = y + 1;
  }
  let pts = null, hull = null, h;
  try {
    pts = cv.matFromArray(4 * n, 1, cv.CV_32SC2, corners);
    hull = new cv.Mat();
    cv.convexHull(pts, hull, false, true);
    h = hull.data32S.slice(); // สำเนา JS — view ของ heap ใช้ต่อหลัง delete ไม่ได้
  } finally {
    if (pts) pts.delete();
    if (hull) hull.delete();
  }
  const m = h.length >> 1;
  // ยาวสุด (Max Feret) = คู่จุดบน hull ที่ห่างกันมากสุด
  let best = -1, bdx = 0, bdy = 0, bi = 0, bj = 0;
  for (let i = 0; i < m; i++) {
    for (let j = i + 1; j < m; j++) {
      let dx = h[2 * j] - h[2 * i], dy = h[2 * j + 1] - h[2 * i + 1];
      const d2 = dx * dx + dy * dy;
      if (dy < 0 || (dy === 0 && dx < 0)) { dx = 0 - dx; dy = 0 - dy; }
      if (d2 > best || (d2 === best && bdx * dy - bdy * dx < 0)) { best = d2; bdx = dx; bdy = dy; bi = i; bj = j; }
    }
  }
  // แคบสุด (Min Feret) = ด้านของ hull ที่ระยะไปจุดไกลสุดน้อยสุด (rotating calipers แบบ brute force) · len = ช่วงที่จุดยื่นไปตามด้านนั้น
  let wMin = Infinity, ndx = 0, ndy = 0, eI = 0, eX = 0, eY = 0, eLo = 0, eHi = 0, eS = 0;
  for (let i = 0; i < m; i++) {
    const j = i + 1 === m ? 0 : i + 1;
    const hx = h[2 * i], hy = h[2 * i + 1];
    const ex = h[2 * j] - hx, ey = h[2 * j + 1] - hy;
    const e2 = ex * ex + ey * ey;
    if (e2 === 0) continue;
    let mc = 0, cS = 0, lo = 0, hi = 0;
    for (let k = 0; k < m; k++) {
      const px = h[2 * k] - hx, py = h[2 * k + 1] - hy;
      const cr = ex * py - ey * px; // ระยะตั้งฉากจากด้าน × |e|
      const dt = ex * px + ey * py; // ตำแหน่งตามแนวด้าน × |e|
      const ac = cr < 0 ? 0 - cr : cr;
      if (ac > mc) { mc = ac; cS = cr; }
      if (dt < lo) lo = dt; else if (dt > hi) hi = dt;
    }
    const w = mc / Math.sqrt(e2);
    let nx = 0 - ey, ny = ex; // ทิศของความกว้าง = ตั้งฉากกับด้าน
    if (ny < 0 || (ny === 0 && nx < 0)) { nx = 0 - nx; ny = 0 - ny; }
    if (w < wMin || (w === wMin && ndx * ny - ndy * nx < 0)) {
      wMin = w; ndx = nx; ndy = ny; eI = i; eX = ex; eY = ey; eLo = lo; eHi = hi; eS = cS;
    }
  }
  const feretMax = Math.sqrt(best) / 2;
  const feretMin = wMin / 2;
  const feretLen = (eHi - eLo) / Math.sqrt(eX * eX + eY * eY) / 2;
  // มุมของกรอบที่แคบสุด: จุดบนด้าน + (t ตามแนวด้าน, s ตั้งฉาก) / e2 → พิกัด ×2 → พิกเซล
  const hx = h[2 * eI], hy = h[2 * eI + 1], e2 = eX * eX + eY * eY;
  const corner = (t, s) => [
    Math.floor((hx + (eX * t - eY * s) / e2) / 2 + 0.5),
    Math.floor((hy + (eY * t + eX * s) / e2) / 2 + 0.5),
  ];
  return {
    max: feretMax, min: feretMin, len: feretLen, aspect: feretLen / feretMin,
    maxDeg: feretAngle(bdx, bdy), minDeg: feretAngle(ndx, ndy),
    maxA: [(h[2 * bi] + 1) >> 1, (h[2 * bi + 1] + 1) >> 1],
    maxB: [(h[2 * bj] + 1) >> 1, (h[2 * bj + 1] + 1) >> 1],
    box: [corner(eLo, 0), corner(eHi, 0), corner(eHi, eS), corner(eLo, eS)],
  };
}

// ctx.values ของ 3 op — ทุกเส้นทาง (รวม 0×0 / ไม่พบวัตถุ / A = B / ภาพเล็ก) ส่ง key/คอลัมน์ชุดเดียวกันเสมอ (detail-panel staleWatch พึ่งข้อนี้)
// สร้าง object ใหม่ทุกครั้ง (ห้ามแชร์ข้ามเฟรม) · ค่าที่คำนวณไม่ได้ = null (แผงขวาแสดง "—")
export const FERET_COLUMNS = ["index", "area", "cx", "cy", "feret_max", "feret_min", "feret_len", "feret_aspect", "feret_max_deg", "feret_min_deg"];
const LINE_PROFILE_FIELDS = [
  ["line_length_px", "ความยาวเส้น (px)"],
  ["samples", "จำนวนจุดที่อ่าน"],
  ["profile_min", "ค่าต่ำสุดบนเส้น"],
  ["profile_max", "ค่าสูงสุดบนเส้น"],
  ["profile_mean", "ค่าเฉลี่ยบนเส้น"],
  ["profile_contrast", "ความต่าง สูงสุด−ต่ำสุด"],
  ["edge_jump", "ค่าที่กระโดดแรงสุด (+ = มืด→สว่าง)"],
  ["edge_pos_px", "ตำแหน่งขอบแรงสุด นับจาก A (px)"],
  ["edge_pair_px", "ระยะระหว่างขอบขึ้นกับขอบลง (px)"],
];
const ROI_FIELDS = [
  ["roi_vertices", "จำนวนจุด"],
  ["roi_area_px", "พื้นที่ ROI (px)"],
  ["roi_area_pct", "พื้นที่ ROI (% ของภาพ)"],
  ["roi_perimeter_px", "เส้นรอบรูป (px)"],
  ["roi_mean", "ความสว่างเฉลี่ยใน ROI"],
  ["roi_std", "ส่วนเบี่ยงเบนความสว่างใน ROI"],
  ["roi_crossed", "เส้นขอบไขว้กัน"],
];
function scalarValues(spec, vals) {
  return { kind: "scalars", fields: spec.map(([key, label], i) => ({ key, label, value: vals[i] === undefined ? null : vals[i] })) };
}
// select param: เลือกด้วย own-property เท่านั้น ("constructor" จากไฟล์โปรเจกต์ที่แก้มือ → ค่าเริ่มต้น) แบบเดียวกับชิ้น B
const LINE_PROFILE_VIEWS = { overlay: true, plot: true };
const ROI_VIEWS = { apply: true, mask: true, overlay: true };
function ownKey(table, v, d) { return Object.hasOwn(table, v) ? v : d; }
// ค่าเริ่มต้นของจุด P1..P6 (%) — เป็นรูปหลายเหลี่ยมแบบไม่ไขว้ทุก n = 3..6 (เทสต์ปักไว้) · params ของ roi_polygon สร้างจากตารางนี้
const ROI_DEFAULTS = [[30, 20], [70, 20], [90, 85], [10, 85], [5, 55], [15, 30]];
const LINE_PROFILE_MIN_GRAPH = 64; // ภาพเล็กกว่านี้ (ด้านใดด้านหนึ่ง) ไม่วาดกราฟ — ตัวเลขยังคำนวณครบ
// ส่วนของ ROI ที่คำนวณจากจุดล้วน ๆ (เลขจำนวนเต็ม — Python ทำลูปเดียวกัน): เส้นรอบรูปรวมด้านปิด · ด้านที่ไม่ติดกันแตะ/ตัดกัน ·
// ทุกจุดอยู่บนเส้นตรงเดียว (รวมทุกจุดซ้อนกัน) — ใช้ "เรียงเป็นเส้นตรง" แทน shoelace = 0 เพราะโบว์ไทสมมาตรก็มี shoelace = 0 ทั้งที่มีพื้นที่จริง
function roiOrient(ax, ay, bx, by, cx, cy) { return Math.sign((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)); }
function roiOnSeg(ax, ay, bx, by, cx, cy) {
  return Math.min(ax, bx) <= cx && cx <= Math.max(ax, bx) && Math.min(ay, by) <= cy && cy <= Math.max(ay, by);
}
function roiSegTouch(q, a, b, c, e) { // ด้าน q[a]→q[b] กับ q[c]→q[e] ตัดกันจริงหรือแตะกัน (รวมซ้อนทับแนวเดียวกัน)
  const [x1, y1, x2, y2, x3, y3, x4, y4] = [q[2 * a], q[2 * a + 1], q[2 * b], q[2 * b + 1], q[2 * c], q[2 * c + 1], q[2 * e], q[2 * e + 1]];
  const d1 = roiOrient(x3, y3, x4, y4, x1, y1), d2 = roiOrient(x3, y3, x4, y4, x2, y2);
  const d3 = roiOrient(x1, y1, x2, y2, x3, y3), d4 = roiOrient(x1, y1, x2, y2, x4, y4);
  if (d1 * d2 < 0 && d3 * d4 < 0) return true;
  return (d1 === 0 && roiOnSeg(x3, y3, x4, y4, x1, y1)) || (d2 === 0 && roiOnSeg(x3, y3, x4, y4, x2, y2))
    || (d3 === 0 && roiOnSeg(x1, y1, x2, y2, x3, y3)) || (d4 === 0 && roiOnSeg(x1, y1, x2, y2, x4, y4));
}
function roiShape(q, n) {
  let perimeter = 0;
  for (let i = 0; i < n; i++) {
    const j = i + 1 === n ? 0 : i + 1;
    const dx = q[2 * j] - q[2 * i], dy = q[2 * j + 1] - q[2 * i + 1];
    perimeter += Math.sqrt(dx * dx + dy * dy);
  }
  // ตัดจุดที่ซ้ำกับจุดก่อนหน้า (รวมจุดสุดท้ายซ้ำจุดแรก) ก่อนเช็กไขว้ — ด้านยาว 0 ทำให้สองด้านที่ใช้จุดร่วมกันดูเหมือน "ไม่ติดกันแต่แตะกัน" (ไขว้เท็จ)
  const u = [];
  for (let i = 0; i < n; i++) {
    const x = q[2 * i], y = q[2 * i + 1], L = u.length;
    if (L && u[L - 2] === x && u[L - 1] === y) continue;
    u.push(x, y);
  }
  while (u.length > 2 && u[u.length - 2] === u[0] && u[u.length - 1] === u[1]) u.length -= 2;
  const m = u.length >> 1;
  let crossed = false;
  for (let i = 0; i < m && !crossed; i++) {
    for (let j = i + 2; j < m; j++) {
      if (i === 0 && j === m - 1) continue; // ด้านแรกกับด้านปิดติดกันที่จุด 1
      if (roiSegTouch(u, i, i + 1, j, j + 1 === m ? 0 : j + 1)) { crossed = true; break; }
    }
  }
  let k = 1;
  while (k < n && q[2 * k] === q[0] && q[2 * k + 1] === q[1]) k++;
  let degenerate = true;
  if (k < n) {
    const ux = q[2 * k] - q[0], uy = q[2 * k + 1] - q[1];
    for (let i = k + 1; i < n; i++) {
      if (ux * (q[2 * i + 1] - q[1]) - uy * (q[2 * i] - q[0]) !== 0) { degenerate = false; break; }
    }
  }
  return { perimeter, crossed, degenerate };
}

// ===================================================================
// ตัวช่วยของ IP batch 2 ชิ้น E (2026-10-04 — slic / bandpass_notch) · ใช้ซ้ำกับ superpixel / ตัวกรองความถี่ตัวต่อไปได้
// ===================================================================
const SLIC_MAX_SIDE = 320; // วิเคราะห์ superpixel บนภาพย่อไม่เกินนี้ (เหมือน K-means) แล้วขยายป้ายกำกับกลับด้วย nearest neighbour
const BANDPASS_MAX_SIDE = 1920; // ต่างจาก FFT Low/High-pass (640): ย่อเฉพาะภาพที่ใหญ่กว่า 1920 px — ที่ 640 ลาย notch ผิดตำแหน่งและผลเบลอ
const SLIC_FIELDS = [
  ["sp_count", "จำนวน superpixel ที่ได้"],
  ["sp_requested", "จำนวนที่ตั้งไว้"],
  ["sp_grid", "จุดเริ่มบนตาราง"],
  ["sp_step_px", "ระยะห่างจุดเริ่ม (px ของภาพจริง)"],
  ["sp_mean_area_px", "พื้นที่เฉลี่ยต่อก้อน (px)"],
];
const SLIC_VIEWS = { boundary: true, mean: true };
const BANDPASS_KINDS = { bandpass: true, bandreject: true, notch: true };
const BANDPASS_SHAPES = { gaussian: true, ideal: true };
const BANDPASS_VIEWS = { result: true, spectrum: true };
// เลขมีเครื่องหมายสำหรับข้อความ (+25 / −25 ด้วย U+2212 / 0) — ใช้ทั้ง info และการ์ด · −0 แสดงเป็น "0"
export function signedText(v) { return v > 0 ? `+${v}` : v < 0 ? `−${-v}` : "0"; }

// SLIC (Achanta 2012 แบบย่อ) บน JS ล้วน — F = ข้อมูลฟีเจอร์ interleave nch ช่อง (Uint8: Lab 8-bit 3 ช่อง หรือความสว่าง 1 ช่อง) ขนาด w×h
// ลำดับคำนวณคงที่ทุกขั้น (เลขจำนวนเต็มล้วนในผลรวม, float64 ในระยะ, ไม่มีการเช็คลู่เข้า) → Python ที่ export ได้ป้ายกำกับเท่ากันทุกพิกเซล
// คืน { labels: Int32Array (0..count−1 เรียงตามพิกเซลแรกที่พบแบบ raster), count, nx, ny, S, merged }
export function slicLabels(F, nch, w, h, K, m, iters) {
  const N = w * h;
  const S = Math.sqrt(N / K);
  const nx = Math.max(1, Math.min(w, Math.floor(w / S + 0.5)));
  const ny = Math.max(1, Math.min(h, Math.floor(h / S + 0.5)));
  const nc = nx * ny;
  const KL = 100 / 255; // Lab 8-bit: L ถูกยืดเป็น 0..255 → คืนเป็นสเกล 0..100 (a, b มีแค่เลื่อน +128 จึงไม่ยืด)
  const wS = (m / S) * (m / S);
  const cx = new Float64Array(nc), cy = new Float64Array(nc);
  const c0 = new Float64Array(nc), c1 = new Float64Array(nc), c2 = new Float64Array(nc);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const x = Math.floor(((2 * i + 1) * w) / (2 * nx)), y = Math.floor(((2 * j + 1) * h) / (2 * ny));
      cx[k] = x; cy[k] = y;
      const o = (y * w + x) * nch;
      c0[k] = F[o];
      if (nch === 3) { c1[k] = F[o + 1]; c2[k] = F[o + 2]; }
    }
  }
  const labels = new Int32Array(N);
  for (let y = 0; y < h; y++) {
    const row = Math.floor((y * ny) / h) * nx;
    for (let x = 0; x < w; x++) labels[y * w + x] = row + Math.floor((x * nx) / w);
  }
  const dist = new Float64Array(N);
  const cnt = new Float64Array(nc), sx = new Float64Array(nc), sy = new Float64Array(nc);
  const s0 = new Float64Array(nc), s1 = new Float64Array(nc), s2 = new Float64Array(nc);
  for (let it = 0; it < iters; it++) {
    dist.fill(Infinity);
    for (let k = 0; k < nc; k++) {
      const x0 = Math.max(0, Math.ceil(cx[k] - S)), x1 = Math.min(w - 1, Math.floor(cx[k] + S));
      const y0 = Math.max(0, Math.ceil(cy[k] - S)), y1 = Math.min(h - 1, Math.floor(cy[k] + S));
      const ck0 = c0[k], ck1 = c1[k], ck2 = c2[k], ckx = cx[k], cky = cy[k];
      for (let y = y0; y <= y1; y++) {
        const dy = y - cky;
        for (let x = x0; x <= x1; x++) {
          const i = y * w + x;
          const dx = x - ckx;
          const dL = (F[i * nch] - ck0) * KL;
          let D;
          if (nch === 3) {
            const da = F[i * 3 + 1] - ck1, db = F[i * 3 + 2] - ck2;
            D = dL * dL + da * da + db * db + (dx * dx + dy * dy) * wS;
          } else {
            D = dL * dL + (dx * dx + dy * dy) * wS;
          }
          if (D < dist[i]) { dist[i] = D; labels[i] = k; }
        }
      }
    }
    cnt.fill(0); sx.fill(0); sy.fill(0); s0.fill(0); s1.fill(0); s2.fill(0);
    for (let y = 0, i = 0; y < h; y++) {
      for (let x = 0; x < w; x++, i++) {
        const l = labels[i];
        cnt[l] += 1; sx[l] += x; sy[l] += y;
        s0[l] += F[i * nch];
        if (nch === 3) { s1[l] += F[i * 3 + 1]; s2[l] += F[i * 3 + 2]; }
      }
    }
    for (let k = 0; k < nc; k++) {
      if (cnt[k] > 0) {
        cx[k] = sx[k] / cnt[k]; cy[k] = sy[k] / cnt[k];
        c0[k] = s0[k] / cnt[k]; c1[k] = s1[k] / cnt[k]; c2[k] = s2[k] / cnt[k];
      }
    }
  }
  // รวมเศษชิ้นเล็ก: เดินพิกเซลตาม raster · ก้อนที่เชื่อมกัน 4 ทิศ (ป้ายเดิมเดียวกัน) ที่ขนาด ≤ limit และมีเพื่อนบ้านที่ติดป้ายใหม่แล้ว → ใช้ป้ายของเพื่อนบ้านนั้น
  const limit = Math.floor(N / nc) >> 2;
  const out = new Int32Array(N).fill(-1);
  const queue = new Int32Array(N);
  let cur = 0, merged = 0;
  for (let i0 = 0; i0 < N; i0++) {
    if (out[i0] >= 0) continue;
    const old = labels[i0];
    const sxp = i0 % w, syp = (i0 - sxp) / w;
    let adj = -1; // ป้ายใหม่ของเพื่อนบ้านที่ติดป้ายแล้วตัวแรก ลำดับ ซ้าย บน ขวา ล่าง
    if (sxp > 0 && out[i0 - 1] >= 0) adj = out[i0 - 1];
    else if (syp > 0 && out[i0 - w] >= 0) adj = out[i0 - w];
    else if (sxp < w - 1 && out[i0 + 1] >= 0) adj = out[i0 + 1];
    else if (syp < h - 1 && out[i0 + w] >= 0) adj = out[i0 + w];
    out[i0] = cur;
    queue[0] = i0;
    let head = 0, tail = 1;
    while (head < tail) {
      const i = queue[head++];
      const px = i % w, py = (i - px) / w;
      if (px > 0 && out[i - 1] < 0 && labels[i - 1] === old) { out[i - 1] = cur; queue[tail++] = i - 1; }
      if (py > 0 && out[i - w] < 0 && labels[i - w] === old) { out[i - w] = cur; queue[tail++] = i - w; }
      if (px < w - 1 && out[i + 1] < 0 && labels[i + 1] === old) { out[i + 1] = cur; queue[tail++] = i + 1; }
      if (py < h - 1 && out[i + w] < 0 && labels[i + w] === old) { out[i + w] = cur; queue[tail++] = i + w; }
    }
    if (tail <= limit && adj >= 0) {
      for (let q = 0; q < tail; q++) out[queue[q]] = adj;
      merged++;
    } else {
      cur++;
    }
  }
  return { labels: out, count: cur, nx, ny, S, merged };
}

// ค่าเกน (0..1) ของตัวกรองความถี่หนึ่งจุด · fx, fy = ความถี่ปรับเป็นสัดส่วนของ Nyquist (−1..1) พิกัดแบบไม่ shift (DC = 0,0)
// prm = { lo, hi, nx, ny, nr } (% ของ Nyquist) · notch คู่สมมาตรที่ (+nx,+ny) และ (−nx,−ny) · Python ที่ export ใช้นิพจน์ข้อความเดียวกัน
function bandGain(kind, shape, prm, fx, fy) {
  if (kind === "notch") {
    const a = fx * 100 - prm.nx, b = fy * 100 - prm.ny, c = fx * 100 + prm.nx, d = fy * 100 + prm.ny;
    // หมายเหตุ: แถว/คอลัมน์ Nyquist (fx = −1) ไม่มีคู่ที่ +1 — เมื่อ |nx|+nr หรือ |ny|+nr ≈ 100% เกนตรงนั้นไม่สมมาตรแบบ Hermitian ·
    // DFT_REAL_OUTPUT ใช้ครึ่งสเปกตรัม · parity กับ Python ยังตรง เพราะสองฝั่งเรียกแบบเดียวกัน (ไม่เปลี่ยนพฤติกรรม)
    const d1 = a * a + b * b, d2 = c * c + d * d, r2 = prm.nr * prm.nr;
    if (shape === "ideal") return (d1 <= r2 || d2 <= r2) ? 0 : 1;
    return (1 - Math.exp(-d1 / (2 * r2))) * (1 - Math.exp(-d2 / (2 * r2)));
  }
  const lo = prm.lo, hi = prm.hi;
  const r = Math.sqrt(fx * fx + fy * fy) * 100;
  const bp = shape === "ideal"
    ? ((r >= lo && r <= hi) ? 1 : 0)
    : (lo > 0 ? 1 - Math.exp(-(r * r) / (2 * lo * lo)) : 1) * Math.exp(-(r * r) / (2 * hi * hi));
  return kind === "bandreject" ? 1 - bp : bp;
}

// ข้อความ info ของ bandpass_notch — ลำดับต่อท้ายคงที่ (สลับขอบ · ย่อภาพ · เตือนความสว่าง · ภาพเรียบ · มุมมองสเปกตรัม) ตามสเปก E §5
function bandpassInfo(kind, shape, prm, P, Q, swapped, scale, aw, ah, view, dcGain, flat) {
  const gl = shape === "ideal" ? "Ideal" : "Gaussian";
  let info;
  if (kind === "notch") {
    info = `Notch (${gl}) ตัดจุดคู่ที่ (${signedText(prm.nx)}%, ${signedText(prm.ny)}%) และ (${signedText(-prm.nx)}%, ${signedText(-prm.ny)}%) รัศมี ${prm.nr}% · สเปกตรัม ${P}×${Q} · เปิดมุมมอง "สเปกตรัมหลังกรอง" เพื่อเล็งวงมืดให้ทับจุดสว่างของลายซ้ำ`;
  } else if (kind === "bandreject") {
    info = `Band-reject (${gl}) ตัดวงแหวนความถี่ ${prm.lo}–${prm.hi}% ของ Nyquist · สเปกตรัม ${P}×${Q}`;
  } else {
    info = `Band-pass (${gl}) เก็บวงแหวนความถี่ ${prm.lo}–${prm.hi}% ของ Nyquist · สเปกตรัม ${P}×${Q} · ยืดค่าผลลัพธ์ให้เต็มช่วง 0–255 เพื่อให้มองเห็น`;
  }
  if (swapped) info += " · สลับขอบใน/ขอบนอกให้แล้ว";
  if (scale < 1) info += ` · ภาพใหญ่กว่า ${BANDPASS_MAX_SIDE} px จึงย่อเหลือ ${aw}×${ah} ก่อนกรอง แล้วขยายกลับ`;
  if (view === "result" && kind !== "bandpass" && dcGain < 0.5) info += " · ⚠️ ตัวกรองนี้ตัดค่าเฉลี่ยความสว่าง (จุดกลางสเปกตรัม) ทิ้งด้วย ภาพจึงมืดลง";
  if (flat) info += " · ไม่มีรายละเอียดในช่วงความถี่นี้ (ภาพเรียบในช่วงนี้) จึงได้ภาพดำ";
  if (view === "spectrum") info += " · มุมมองสเปกตรัม: จุดกลาง = ความถี่ต่ำ ส่วนที่มืดคือส่วนที่ถูกตัด";
  return info;
}

// ===================================================================
// ตัวช่วยของ IP batch 4A (2026-10-15) — psfKernel / guidedFilterF32 / PSEUDOCOLOR_MAPS / MM_COLUMNS / ΔE
// batch 4B ใช้ psfKernel (Wiener/Lucy-Richardson) และ guidedFilterF32 (Dehaze) ตัวเดียวกัน — ห้ามเขียนซ้ำ
// ===================================================================
// PSF สำหรับ Custom Kernel (motion/disk) — นับ "จุดย่อย" 16×16 ต่อพิกเซลที่อยู่ในรูป แล้วหารด้วยจำนวนจุดย่อยรวม → ผลรวม = 1 เสมอ
export const PSF_SS = 16;
export function psfKernel(kind, size, angleDeg) {
  const SS = PSF_SS;
  let n, cnt;
  if (kind === "disk") {
    const R = size, h = Math.ceil(R - 0.5);
    n = 2 * h + 1; cnt = new Int32Array(n * n);
    const lim = 4 * SS * SS * R * R; // (2·SS·R)² — พิกัดจุดย่อยคูณ 2·SS ให้เป็นจำนวนเต็ม → เทียบแบบไม่มีเศษ
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      let c = 0;
      for (let sy = 0; sy < SS; sy++) for (let sx = 0; sx < SS; sx++) {
        const px = (x - h) * 2 * SS + (2 * sx + 1 - SS), py = (y - h) * 2 * SS + (2 * sy + 1 - SS);
        if (px * px + py * py <= lim) c++;
      }
      cnt[y * n + x] = c;
    }
  } else { // motion: เส้นยาว L ผ่านจุดกลาง หนา 1 px (ปลายมน) · มุมทวนเข็มแบบ MATLAB (แกน y ของภาพชี้ลง จึงใช้ −sin)
    const L = size, th = (angleDeg * Math.PI) / 180, c = Math.cos(th), s = -Math.sin(th), hl = (L - 1) / 2;
    const h = Math.ceil(hl * Math.max(Math.abs(c), Math.abs(s)) + 0.5);
    n = 2 * h + 1; cnt = new Int32Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      let k = 0;
      for (let sy = 0; sy < SS; sy++) for (let sx = 0; sx < SS; sx++) {
        const px = (x - h) + (2 * sx + 1 - SS) / (2 * SS), py = (y - h) + (2 * sy + 1 - SS) / (2 * SS);
        const t = Math.max(-hl, Math.min(hl, px * c + py * s));
        const dx = px - t * c, dy = py - t * s;
        if (dx * dx + dy * dy <= 0.25) k++;
      }
      cnt[y * n + x] = k;
    }
  }
  // ตัดวงนอกที่เป็น 0 ทั้งวง (ทั้ง 4 ด้านพร้อมกัน — ขนาดยังเป็นเลขคี่ จุดกลางไม่ขยับ)
  let t = 0;
  const ringZero = (r) => { for (let i = r; i < n - r; i++) if (cnt[r * n + i] || cnt[(n - 1 - r) * n + i] || cnt[i * n + r] || cnt[i * n + n - 1 - r]) return false; return true; };
  while (n - 2 * t > 1 && ringZero(t)) t++;
  const m = n - 2 * t, k = new Float64Array(m * m);
  let tot = 0;
  for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) tot += cnt[(y + t) * n + x + t];
  for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) k[y * m + x] = cnt[(y + t) * n + x + t] / tot;
  return { n: m, k };
}

// Guided Filter (He et al. 2010) บน float32 สเกล 0–255 — I = ภาพนำทาง, P = ภาพที่กรอง (Mat CV_32FC1 ขนาดเท่ากัน)
// I, P ต้องเป็น Mat ต่อเนื่อง (continuous) เพราะอ่าน data32F ตรง ๆ — ห้ามส่ง roi() ที่ยังไม่ clone · คืน Mat CV_32FC1 ใหม่ที่ "ผู้เรียกเป็นเจ้าของ" · eps255 = ε × 65025 · ทุกขั้น float32 ด้วย Math.fround ตรงกับ numpy float32 ใน Python ที่ export
// (opencv.js build นี้ไม่มี ximgproc.guidedFilter จึงสร้างจาก boxFilter)
export function guidedFilterF32(cv, I, P, r, eps255) {
  const f = Math.fround;
  const e32 = f(eps255);
  const temps = [];
  const mk = () => { const m = new cv.Mat(); temps.push(m); return m; };
  const size = new cv.Size(2 * r + 1, 2 * r + 1), anchor = new cv.Point(-1, -1);
  const box = (m) => { const o = mk(); cv.boxFilter(m, o, -1, size, anchor, true, cv.BORDER_REFLECT_101); return o; };
  let q = null;
  try {
    const mI = box(I), mP = box(P);
    const II = mk(); cv.multiply(I, I, II);
    const IP = mk(); cv.multiply(I, P, IP);
    const cI = box(II), cIP = box(IP);
    const rows = I.rows, cols = I.cols, n = rows * cols;
    const A = mk(); A.create(rows, cols, cv.CV_32FC1);
    const B = mk(); B.create(rows, cols, cv.CV_32FC1);
    const aI = mI.data32F, aP = mP.data32F, aCI = cI.data32F, aCIP = cIP.data32F, aA = A.data32F, aB = B.data32F;
    for (let i = 0; i < n; i++) {
      const v = f(aCI[i] - f(aI[i] * aI[i]));
      const cov = f(aCIP[i] - f(aI[i] * aP[i]));
      const a = f(cov / f(v + e32));
      aA[i] = a;
      aB[i] = f(aP[i] - f(a * aI[i]));
    }
    const mA = box(A), mB = box(B);
    q = new cv.Mat(); q.create(rows, cols, cv.CV_32FC1);
    const aq = q.data32F, dI = I.data32F, dA = mA.data32F, dB = mB.data32F;
    for (let i = 0; i < n; i++) aq[i] = f(f(dA[i] * dI[i]) + dB[i]);
    const out = q; q = null;
    return out;
  } finally {
    if (q) q.delete();
    for (const m of temps) m.delete();
  }
}

// แผนที่สี 256 ระดับ → [R,G,B] · ใช้แค่เลขจำนวนเต็ม/ครึ่ง (JS = Python ทุกค่า) · เพิ่มแผนที่ใหม่ = เพิ่ม 1 แถว (+ 1 แถวใน PSEUDOCOLOR_PY ของ codegen.js)
const clamp255 = (v) => Math.floor(Math.min(255, Math.max(0, v)) + 0.5);
export const PSEUDOCOLOR_MAPS = {
  jet: (i) => [clamp255(382.5 - Math.abs(4 * i - 765)), clamp255(382.5 - Math.abs(4 * i - 510)), clamp255(382.5 - Math.abs(4 * i - 255))],
  hot: (i) => [clamp255(3 * i), clamp255(3 * i - 255), clamp255(3 * i - 510)],
  cool: (i) => [i, 255 - i, 255],
};
const PSEUDOCOLOR_TH = { jet: "Jet", hot: "Hot", cool: "Cool" };
const PSEUDOCOLOR_ENDS = { jet: ["น้ำเงินเข้ม", "แดงเข้ม"], hot: ["ดำ", "ขาว"], cool: ["ฟ้า", "ชมพู"] };

// px → mm ด้วยตัวคูณเดียว · power 1 = ความยาว, 2 = พื้นที่ · [key ใหม่, key ต้นทาง, power]
export const MM_COLUMNS = {
  regionprops: [["area_mm2", "area_px", 2], ["perimeter_mm", "perimeter", 1], ["equiv_diameter_mm", "equiv_diameter", 1], ["major_axis_mm", "major_axis", 1], ["minor_axis_mm", "minor_axis", 1]],
  feret: [["feret_max_mm", "feret_max", 1], ["feret_min_mm", "feret_min", 1], ["feret_len_mm", "feret_len", 1]],
  line_profile: [["line_length_mm", "line_length_px", 1], ["edge_pos_mm", "edge_pos_px", 1], ["edge_pair_mm", "edge_pair_px", 1]],
};
export function mmValue(v, f, power) { return v === null || v === undefined ? null : power === 2 ? v * (f * f) : v * f; }
function mmScaleParam() {
  return { key: "mmPerPx", type: "number", label: "ขนาดจริงต่อพิกเซล (มม./px)", min: 0, max: 100, step: 0.001, default: 0,
    hint: "0 = แสดงเป็นพิกเซลอย่างเดียว",
    desc: "วัดของที่รู้ขนาดจริงในภาพก่อน เช่น เหรียญ 26 มม. วัดได้ 130 px → 26 ÷ 130 = 0.2 มม./px · ใช้ได้เมื่อกล้องตั้งฉากกับชิ้นงานและระยะเท่ากันทั้งภาพ" };
}

// ตารางที่ต่อคอลัมน์ mm ท้ายตาราง (f > 0 เท่านั้น — f = 0 คืนคอลัมน์/แถวเดิมทุก byte) · แถวเดิมไม่ถูกแก้ (สร้างแถวใหม่)
function mmTable(opId, f, columns, rows) {
  if (!(f > 0)) return { columns: [...columns], rows };
  const spec = MM_COLUMNS[opId];
  const idx = spec.map((sp) => columns.indexOf(sp[1]));
  return {
    columns: [...columns, ...spec.map((sp) => sp[0])],
    rows: rows.map((r) => [...r, ...spec.map((sp, j) => mmValue(r[idx[j]], f, sp[2]))]),
  };
}
const MM_SUFFIX = (f) => ` · 1 px = ${f} มม.`;
const LINE_PROFILE_MM_LABELS = ["ความยาวเส้น (มม.)", "ตำแหน่งขอบที่กระโดดแรง นับจาก A (มม.)", "ระยะระหว่างขอบขึ้นกับขอบลง (มม.)"];
function lineProfileValues(vals, f) {
  const v = scalarValues(LINE_PROFILE_FIELDS, vals);
  if (f > 0) {
    MM_COLUMNS.line_profile.forEach((sp, j) => {
      const src = LINE_PROFILE_FIELDS.findIndex((fl) => fl[0] === sp[1]);
      v.fields.push({ key: sp[0], label: LINE_PROFILE_MM_LABELS[j], value: mmValue(vals[src] === undefined ? null : vals[src], f, sp[2]) });
    });
  }
  return v;
}

// ฟิลด์/ตัวเลือกของ op ใหม่ batch 4A (select อ่านด้วย ownKey — ค่าแปลก/"constructor" ตกค่าเริ่มต้น)
const FOCUS_FIELDS = [["focus_score", "คะแนนความคม"], ["focus_mean", "ความสว่างเฉลี่ยในบริเวณที่วัด"]];
const FOCUS_METHODS = { lap_var: true, tenengrad: true };
const FOCUS_VIEWS = { lap: true, pass: true };
const DELTA_E_FIELDS = [
  ["de_mean", "ΔE เฉลี่ย"], ["de_max", "ΔE สูงสุด"], ["de_over_pct", "พื้นที่ที่ ΔE เกินเกณฑ์ (%)"],
  ["de_a_L", "A: L* เฉลี่ย"], ["de_a_a", "A: a* เฉลี่ย"], ["de_a_b", "A: b* เฉลี่ย"],
  ["de_b_L", "B: L* เฉลี่ย"], ["de_b_a", "B: a* เฉลี่ย"], ["de_b_b", "B: b* เฉลี่ย"],
];
const DELTA_E_FORMULAS = { cie76: true, ciede2000: true };
const DELTA_E_COMPARES = { pixel: true, mean_b: true };
const DELTA_E_VIEWS = { heatmap: true, over: true };
const ROI_APPLY_FIELDS = [["ra_px", "พิกเซลที่ใช้ผล B"], ["ra_pct", "สัดส่วนที่ใช้ผล B (%)"]];
const ROI_APPLY_SHAPES = { rect: true, ellipse: true };
const ROI_APPLY_VIEWS = { result: true, outline: true };
// RGB 8 bit → Lab float32 (L 0–100, a/b ประมาณ ±128) — convertTo(1/255) ก่อน cvtColor ตรงกับ img.astype(np.float32) / 255 ใน Python · คืน Mat ใหม่ที่ผู้เรียกเป็นเจ้าของ
function toLabF32(cv, rgb) {
  const F = new cv.Mat();
  const lab = new cv.Mat();
  try {
    rgb.convertTo(F, cv.CV_32F, 1 / 255);
    cv.cvtColor(F, lab, cv.COLOR_RGB2Lab);
    return lab;
  } catch (err) {
    lab.delete();
    throw err;
  } finally {
    F.delete();
  }
}

// ΔE (ความต่างสี) ระหว่างสอง Lab — CIE76 = ระยะตรง · CIEDE2000 = Sharma, Wu, Dalal (2005) kL = kC = kH = 1 · double ล้วน
export function deltaE76(L1, a1, b1, L2, a2, b2) {
  const dL = L1 - L2, da = a1 - a2, db = b1 - b2;
  return Math.sqrt(dL * dL + da * da + db * db);
}
const POW25_7 = 6103515625; // 25^7
const RAD = Math.PI / 180;
function hueDeg(b, ap) { // 0–360 · a′ = b = 0 → 0
  if (ap === 0 && b === 0) return 0;
  const h = Math.atan2(b, ap) * (180 / Math.PI);
  return h < 0 ? h + 360 : h;
}
export function deltaE2000(L1, a1, b1, L2, a2, b2) {
  const C1 = Math.sqrt(a1 * a1 + b1 * b1), C2 = Math.sqrt(a2 * a2 + b2 * b2);
  const Cb = (C1 + C2) / 2, Cb7 = Math.pow(Cb, 7);
  const G = 0.5 * (1 - Math.sqrt(Cb7 / (Cb7 + POW25_7)));
  const ap1 = (1 + G) * a1, ap2 = (1 + G) * a2;
  const Cp1 = Math.sqrt(ap1 * ap1 + b1 * b1), Cp2 = Math.sqrt(ap2 * ap2 + b2 * b2);
  const h1 = hueDeg(b1, ap1), h2 = hueDeg(b2, ap2);
  const dLp = L2 - L1, dCp = Cp2 - Cp1;
  const cc = Cp1 * Cp2;
  let dhp = 0;
  if (cc !== 0) {
    dhp = h2 - h1;
    if (dhp > 180) dhp -= 360; else if (dhp < -180) dhp += 360;
  }
  const dHp = 2 * Math.sqrt(cc) * Math.sin((dhp * RAD) / 2);
  const Lb = (L1 + L2) / 2, Cbp = (Cp1 + Cp2) / 2;
  let hb;
  if (cc === 0) hb = h1 + h2;
  else if (Math.abs(h1 - h2) <= 180) hb = (h1 + h2) / 2;
  else if (h1 + h2 < 360) hb = (h1 + h2 + 360) / 2;
  else hb = (h1 + h2 - 360) / 2;
  const T = 1 - 0.17 * Math.cos((hb - 30) * RAD) + 0.24 * Math.cos(2 * hb * RAD) + 0.32 * Math.cos((3 * hb + 6) * RAD) - 0.20 * Math.cos((4 * hb - 63) * RAD);
  const dth = 30 * Math.exp(-(((hb - 275) / 25) * ((hb - 275) / 25)));
  const Cbp7 = Math.pow(Cbp, 7);
  const RC = 2 * Math.sqrt(Cbp7 / (Cbp7 + POW25_7));
  const SL = 1 + (0.015 * (Lb - 50) * (Lb - 50)) / Math.sqrt(20 + (Lb - 50) * (Lb - 50));
  const SC = 1 + 0.045 * Cbp, SH = 1 + 0.015 * Cbp * T;
  const RT = -Math.sin(2 * dth * RAD) * RC;
  const tL = dLp / SL, tC = dCp / SC, tH = dHp / SH;
  return Math.sqrt(tL * tL + tC * tC + tH * tH + RT * tC * tH);
}

// White Balance แบบ White Patch / จุดอ้างอิง (IP batch 4A) — ทำบนข้อมูลตรง ๆ ไม่ใช้ convertTo (ปัดครึ่งขึ้นด้วย floor(x+0.5) เหมือน Python ที่ export)
// src ต้องเป็นภาพสี 3/4 ช่อง ไม่ว่าง · คืน Mat RGB ใหม่ที่ผู้เรียกเป็นเจ้าของ
function whiteBalanceByGain(cv, src, p, ctx, method) {
  const rgb = dropAlpha(cv, src); // null เมื่อ 3 ช่องอยู่แล้ว
  let dst = null;
  try {
    const x = rgb || src;
    const W = x.cols, H = x.rows, N = W * H, d = x.data;
    const gains = [1, 1, 1];
    let info;
    if (method === "white_patch") {
      const hist = [new Int32Array(256), new Int32Array(256), new Int32Array(256)];
      for (let i = 0; i < N; i++) { const o = i * 3; hist[0][d[o]]++; hist[1][d[o + 1]]++; hist[2][d[o + 2]]++; }
      const k = Math.floor((N * 99 + 99) / 100); // ตำแหน่งที่ 99% (นับจากพิกเซลมืดสุด) ปัดขึ้น
      const v99 = [0, 0, 0];
      for (let c = 0; c < 3; c++) {
        let cum = 0, v = 0;
        for (; v < 255; v++) { cum += hist[c][v]; if (cum >= k) break; }
        v99[c] = v;
        gains[c] = v > 0 ? 255 / v : 1;
      }
      info = `White Patch: ค่าที่ 99% R/G/B = ${v99[0]}/${v99[1]}/${v99[2]} → เกน R×${fixed2(gains[0])} G×${fixed2(gains[1])} B×${fixed2(gains[2])}`;
    } else {
      const cx = pctToPx(p.refX, 50, W), cy = pctToPx(p.refY, 50, H), r = intIn(p.refSize, 5, 1, 50);
      const x0 = Math.max(0, cx - r), x1 = Math.min(W - 1, cx + r), y0 = Math.max(0, cy - r), y1 = Math.min(H - 1, cy + r);
      const sum = [0, 0, 0];
      for (let y = y0; y <= y1; y++) for (let xx = x0; xx <= x1; xx++) { const o = (y * W + xx) * 3; sum[0] += d[o]; sum[1] += d[o + 1]; sum[2] += d[o + 2]; }
      const n = (x1 - x0 + 1) * (y1 - y0 + 1);
      const m = [sum[0] / n, sum[1] / n, sum[2] / n];
      const t = (m[0] + m[1] + m[2]) / 3;
      for (let c = 0; c < 3; c++) gains[c] = m[c] > 1e-3 ? t / m[c] : 1;
      info = `จุดอ้างอิง (${cx}, ${cy}) px รัศมี ${r} · สีเฉลี่ย R/G/B = ${fixed1(m[0])}/${fixed1(m[1])}/${fixed1(m[2])} → เกน R×${fixed2(gains[0])} G×${fixed2(gains[1])} B×${fixed2(gains[2])}`;
      if (t < 20) info += " · จุดอ้างอิงมืดมาก ผลอาจเพี้ยน — เลือกบริเวณเทาหรือขาว";
    }
    const luts = gains.map((g) => { const l = new Uint8Array(256); for (let v = 0; v < 256; v++) l[v] = Math.min(255, Math.floor(v * g + 0.5)); return l; });
    dst = new cv.Mat(H, W, cv.CV_8UC3);
    const od = dst.data, len = N * 3;
    for (let k = 0; k < len; k++) od[k] = luts[k % 3][d[k]];
    if (ctx) ctx.info = info;
    const out = dst; dst = null;
    return out;
  } finally {
    if (dst) dst.delete();
    if (rgb) rgb.delete();
  }
}

// ===================================================================
// ตัวช่วยของ IP batch 4B (2026-10-15) — Wiener/Lucy (PSF) · ECC · Hit-or-Miss · Regional Max · Gabor · GLCM · NLM · Dehaze · DCT/JPEG
// ตัวช่วยที่ไม่ใช่ cv (typed array ล้วน) export ให้เทสต์หน่วยเรียกโดยไม่ต้องมี opencv · ตัวช่วยของ 4A (psfKernel / guidedFilterF32) เรียกใช้เท่านั้น ห้ามแก้
// ===================================================================
// นโยบาย op ช้า (แหล่งความจริงเดียว — codegen import ค่าเหล่านี้ ไม่ hardcode ซ้ำ): ภาพใหญ่กว่า NLM_MAX_SIDE ถูกย่อก่อน · Lucy จำกัดรอบ · ECC ย่อภาพก่อนจับคู่
export const NLM_MAX_SIDE = 640;
export const LUCY_MAX_ITERS = 50;
// B6 (backlog 2026-10-17): งบงาน Lucy = พิกเซล × ช่อง × รอบ — ภาพ 640×480 สี 20 รอบ (ค่าที่ใช้ได้สบายใน UI) เป็นเพดาน · เกินงบ → ลดรอบ (ไม่ย่อภาพ เพราะ PSF จะผิดสเกล)
export const LUCY_WORK_BUDGET = 640 * 480 * 3 * 20;
// รอบที่ใช้จริง: เท่าที่ขอ ถ้าไม่เกินงบ · เกินงบ → max(1, floor(งบ ÷ (กว้าง×สูง×ช่อง))) (ไม่เกินที่ขอ)
export function lucyBudgetIters(w, h, ch, requested) {
  return Math.min(requested, Math.max(1, Math.floor(LUCY_WORK_BUDGET / Math.max(1, w * h * ch))));
}
export const ECC_DEFAULT_SIDE = 320;
const SLOW_MODE_HINT = "ช้า — เหมาะกับภาพนิ่ง (กล้องสดจะกระตุก)";
// ทศนิยม 3 ตำแหน่งแบบปัดครึ่งขึ้น (คู่กับ fixed2) — Python ใช้ np.floor(x*1000+0.5)/1000 แล้ว :.3f
export function fixed3(x) { return (Math.floor(x * 1000 + 0.5) / 1000).toFixed(3); }

// ---- PSF ของ Wiener / Lucy-Richardson (ช่วง/ค่าเริ่มต้นเท่ากับ param ของ custom_kernel) ----
const PSF_KINDS = { motion: true, disk: true };
export function psfParamDefs() {
  return [
    { key: "psf", type: "select", label: "ชนิด PSF (รูปแบบการเบลอ)", options: [
      { value: "motion", label: "Motion (เบลอจากการขยับ)" },
      { value: "disk", label: "Disk (เบลอจากโฟกัสหลุด)" },
    ], default: "motion" },
    { key: "psfSize", type: "number", label: "ขนาด PSF (px) — ต้องตรงกับที่ภาพเบลอจริง", min: 1, max: 50, step: 1, default: 9, desc: "Motion = ความยาวของรอยขยับ · Disk = รัศมีวงกลม — ตั้งให้เท่ากับตอนที่ทำให้ภาพเบลอ (เช่นค่าใน Custom Kernel)" },
    { key: "psfAngle", visibleWhen: [{key:"psf",in:["motion"]}], type: "number", label: "มุม PSF (องศา) — ใช้กับ Motion เท่านั้น", min: 0, max: 179, step: 1, default: 0, desc: "0° = ขยับแนวนอน · 90° = แนวตั้ง · มุมเพิ่มทวนเข็มนาฬิกา" },
  ];
}
export function psfInfo(kind, size, angle, n) {
  return kind === "disk" ? `disk รัศมี ${size} px (${n}x${n})` : `motion ยาว ${size} px มุม ${angle}° (${n}x${n})`;
}
// อ่านชุด param PSF (select ด้วย ownKey, ตัวเลข clamp) แล้วสร้าง kernel — ใช้ร่วมกันใน Wiener และ Lucy
function psfFromParams(p) {
  const kind = ownKey(PSF_KINDS, p.psf, "motion");
  const size = intIn(p.psfSize, 9, 1, 50), angle = intIn(p.psfAngle, 0, 0, 179);
  return { kind, size, angle, psf: psfKernel(kind, size, angle) };
}

// ---- Hit-or-Miss: 4 แม่แบบ 3×3 (1 = ต้องขาว, −1 = ต้องดำ, 0 = ไม่สน) แถวบน → ล่าง · เพิ่มแม่แบบ = เพิ่ม 1 แถว (และ option ใน hmPattern) ----
export const HITMISS_PATTERNS = {
  isolated: { th: "จุดเดี่ยว", label: "จุดเดี่ยว (ขาวล้อมด้วยดำ)", k: [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]] },
  hole: { th: "รูจุดเดียว", label: "รูจุดเดียว (ดำล้อมด้วยขาว)", k: [[1, 1, 1], [1, -1, 1], [1, 1, 1]] },
  corner: { th: "มุมนูน", label: "มุมนูน (มุมขวาบน; รวม 4 ทิศ = ทุกมุม)", k: [[0, -1, -1], [1, 1, -1], [0, 1, 0]] },
  line_end: { th: "ปลายเส้น", label: "ปลายเส้นตั้ง/นอน (ปลายบน; รวม 4 ทิศ)", k: [[-1, -1, -1], [-1, 1, -1], [0, 1, 0]] },
};
// หมุนตามเข็มนาฬิกา 90° (array 3×3) — new[i][j] = old[2−j][i]
export function rotate90cw(k) {
  const n = k.length;
  return k.map((_, i) => k[i].map((__, j) => k[n - 1 - j][i]));
}
// kernel ทั้งหมดของแม่แบบ (rotate = 4 ทิศ ตัดตัวซ้ำออก — ผลรวม OR ไม่เปลี่ยน) → array ของ array ยาว 9
export function hitMissKernels(pattern, rotate) {
  const out = [], seen = new Set();
  let k = HITMISS_PATTERNS[pattern].k;
  for (let r = 0; r < (rotate ? 4 : 1); r++) {
    const flat = k.flat(), key = flat.join(",");
    if (!seen.has(key)) { seen.add(key); out.push(flat); }
    k = rotate90cw(k);
  }
  return out;
}

// ---- Morphological reconstruction by dilation 8-connected แบบ hybrid ของ Vincent (1993) ----
// J, I = Int32Array ขนาด W·H (J ≤ I ทุกจุด) · แก้ J ในที่แล้วคืน J · O(N) แม้บนที่ราบยาว (ห้ามใช้วิธีวน dilate ในแอป)
export function reconstructDilate(J, I, W, H) {
  const N = W * H;
  if (N === 0) return J;
  // (1) raster scan: J[p] = min(max(J[p], J[q] ทุก q ใน N+(p)), I[p]) · N+ = ซ้าย, ซ้ายบน, บน, ขวาบน
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = y * W + x;
      let m = J[p];
      if (x > 0 && J[p - 1] > m) m = J[p - 1];
      if (y > 0) {
        if (x > 0 && J[p - W - 1] > m) m = J[p - W - 1];
        if (J[p - W] > m) m = J[p - W];
        if (x < W - 1 && J[p - W + 1] > m) m = J[p - W + 1];
      }
      J[p] = m < I[p] ? m : I[p];
    }
  }
  // (2) anti-raster scan ด้วย N− = ขวา, ขวาล่าง, ล่าง, ซ้ายล่าง + ใส่คิวถ้ายังมีเพื่อนบ้านที่ต้องรับค่าต่อ
  const queue = new Int32Array(N), inq = new Uint8Array(N);
  let qh = 0, qt = 0, qn = 0;
  const push = (p) => { if (!inq[p]) { inq[p] = 1; queue[qt] = p; qt = qt + 1 === N ? 0 : qt + 1; qn++; } };
  for (let y = H - 1; y >= 0; y--) {
    for (let x = W - 1; x >= 0; x--) {
      const p = y * W + x;
      let m = J[p];
      const hasR = x < W - 1, hasD = y < H - 1, hasL = x > 0;
      if (hasR && J[p + 1] > m) m = J[p + 1];
      if (hasD) {
        if (hasR && J[p + W + 1] > m) m = J[p + W + 1];
        if (J[p + W] > m) m = J[p + W];
        if (hasL && J[p + W - 1] > m) m = J[p + W - 1];
      }
      const v = m < I[p] ? m : I[p];
      J[p] = v;
      let need = false;
      if (hasR && J[p + 1] < v && J[p + 1] < I[p + 1]) need = true;
      else if (hasD) {
        if (hasR && J[p + W + 1] < v && J[p + W + 1] < I[p + W + 1]) need = true;
        else if (J[p + W] < v && J[p + W] < I[p + W]) need = true;
        else if (hasL && J[p + W - 1] < v && J[p + W - 1] < I[p + W - 1]) need = true;
      }
      if (need) push(p);
    }
  }
  // (3) ส่งค่าต่อทางคิว (FIFO) ทั้ง 8 ทิศ
  while (qn > 0) {
    const p = queue[qh]; qh = qh + 1 === N ? 0 : qh + 1; qn--; inq[p] = 0;
    const px = p % W, py = (p - px) / W, vp = J[p];
    for (let dy = -1; dy <= 1; dy++) {
      const yy = py + dy;
      if (yy < 0 || yy >= H) continue;
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const xx = px + dx;
        if (xx < 0 || xx >= W) continue;
        const q = yy * W + xx;
        if (J[q] < vp && I[q] !== J[q]) { J[q] = vp < I[q] ? vp : I[q]; push(q); }
      }
    }
  }
  return J;
}
// Regional Maxima: บริเวณที่สว่างกว่าทุกพิกเซลรอบตัว (8-connected) → Uint8Array 0/255 · I = Int32Array
export function regionalMaxMask(I, W, H) {
  const n = I.length, J = new Int32Array(n);
  for (let i = 0; i < n; i++) J[i] = I[i] - 1;
  reconstructDilate(J, I, W, H);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = I[i] - J[i] >= 1 ? 255 : 0;
  return out;
}
// H-maxima: กดยอดทุกยอดลง h ระดับ (ตัดที่ 0 เหมือน imhmax บน uint8) → Int32Array ใหม่
export function hMaximaImage(I, W, H, h) {
  const n = I.length, J = new Int32Array(n);
  for (let i = 0; i < n; i++) J[i] = Math.max(I[i] - h, 0);
  return reconstructDilate(J, I, W, H);
}
// แหล่งเดียวสำหรับ Extended Max ทั้ง Morphology และ Marker Watershed (5A)
export function extendedMaxMask(I, W, H, h) { return regionalMaxMask(hMaximaImage(I, W, H, h), W, H); }

// ---- Gabor (สูตร getGaborKernel ของ OpenCV — build นี้ไม่มี จึงคำนวณเอง · ได้ kernel float32 เท่ากับ cv2.getGaborKernel ทุกค่า) ----
export function gaborKernel(k, sigma, thetaDeg, lambd, gamma, psi) {
  const th = (thetaDeg * Math.PI) / 180, c = Math.cos(th), s = Math.sin(th), half = (k - 1) / 2;
  const ex = -0.5 / (sigma * sigma), ey = -0.5 / ((sigma / gamma) * (sigma / gamma)), cs = (2 * Math.PI) / lambd;
  const out = new Float32Array(k * k);
  for (let y = -half; y <= half; y++) {
    for (let x = -half; x <= half; x++) {
      const xr = x * c + y * s, yr = -x * s + y * c;
      out[(half - y) * k + (half - x)] = Math.exp(ex * xr * xr + ey * yr * yr) * Math.cos(cs * xr + psi);
    }
  }
  return out;
}
// σ จาก bandwidth (octave) ตามนิยามของ MATLAB imgaborfilt · ลำดับการคำนวณตายตัว (Python ใช้ลำดับเดียวกัน)
export function gaborSigma(lambd, bandwidth) {
  return lambd / Math.PI * Math.sqrt(Math.LN2 / 2) * (Math.pow(2, bandwidth) + 1) / (Math.pow(2, bandwidth) - 1);
}
export function gaborHalf(sigma, gamma) { return Math.min(49, Math.ceil(3 * sigma / gamma)); }

// ---- GLCM ----
// q = ระดับเทาที่ลดแล้ว (Uint8Array, 0..L−1) · offset (dx, dy) · คืน Float64Array L·L ของจำนวนคู่ [ระดับพิกเซล][ระดับเพื่อนบ้าน]
export function glcmCounts(q, W, H, L, dx, dy) {
  const C = new Float64Array(L * L);
  const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy), x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) C[q[y * W + x] * L + q[(y + dy) * W + x + dx]]++;
  }
  return C;
}
// 4 ค่าของ graycoprops (index เริ่ม 0 — ผลเท่า MATLAB ที่เริ่ม 1) · ไม่มีคู่เลย → null · correlation null เมื่อทุกคู่อยู่ในระดับเดียว (σ = 0)
export function glcmProps(C, L) {
  let N = 0;
  const rowNZ = new Uint8Array(L), colNZ = new Uint8Array(L);
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) { const c = C[i * L + j]; N += c; if (c > 0) { rowNZ[i] = 1; colNZ[j] = 1; } }
  if (N === 0) return null;
  let contrast = 0, energy = 0, homogeneity = 0, mi = 0, mj = 0;
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      const p = C[i * L + j] / N;
      contrast += (i - j) * (i - j) * p;
      energy += p * p;
      homogeneity += p / (1 + Math.abs(i - j));
      mi += i * p;
      mj += j * p;
    }
  }
  let vi = 0, vj = 0, cov = 0;
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      const p = C[i * L + j] / N;
      vi += (i - mi) * (i - mi) * p;
      vj += (j - mj) * (j - mj) * p;
      cov += (i - mi) * (j - mj) * p;
    }
  }
  let nzr = 0, nzc = 0;
  for (let i = 0; i < L; i++) { nzr += rowNZ[i]; nzc += colNZ[i]; }
  // σ = 0 เมื่อแถวหรือคอลัมน์ที่มีคู่เหลือระดับเดียว (ตัดสินจากจำนวนเต็ม ไม่พึ่งเศษปัดของ float)
  const correlation = nzr <= 1 || nzc <= 1 ? null : cov / (Math.sqrt(vi) * Math.sqrt(vj));
  return { contrast, correlation, energy, homogeneity, pairs: N };
}

// ---- Non-local Means (ภาพ 1 หรือ 3 ช่อง interleaved Uint8Array) ----
// ลูป offset (dy นอก dx ใน) → d2 ต่อพิกเซล → integral image → หน้าต่าง patch ตัดที่ขอบ → น้ำหนัก exp(−(S/n)/h²) · ห้ามเปลี่ยนลำดับการบวก (Python บวกลำดับเดียวกัน)
// d2 / integral เป็น Int32Array: ผลรวมจริงของหน้าต่างไม่เกิน 49·3·255² ≈ 9.6 ล้าน (< 2³¹) และผลต่างของ integral ที่ wrap รอบ 2³² ให้ค่าถูกเป๊ะเสมอ
// (S จึงเป็นจำนวนเต็มแม่นเท่ากับแบบ Float64) · แยกฟังก์ชันตามจำนวนช่อง (1 / 3 / อื่น) เพื่อให้แต่ละลูป JIT แบบเดียว: รวมลูป 1 ช่องกับ 3 ช่องไว้ฟังก์ชันเดียว
// วัดแล้วช้าลง 2 เท่าเมื่อสลับเรียกภาพเทา/สีในโปรเซสเดียวกัน (ผลเท่ากันทุก bit)
function nlmD2(data, W, H, ch, dx, dy, xb, d2) {
  for (let x = 0; x < W; x++) { const t = x + dx; xb[x] = t < 0 ? 0 : t >= W ? W - 1 : t; }
  for (let y = 0; y < H; y++) {
    const ty = y + dy, rb = (ty < 0 ? 0 : ty >= H ? H - 1 : ty) * W, ra = y * W;
    if (ch === 1) {
      for (let x = 0; x < W; x++) { const t = data[ra + x] - data[rb + xb[x]]; d2[ra + x] = t * t; }
    } else if (ch === 3) {
      for (let x = 0; x < W; x++) {
        const p = (ra + x) * 3, b = (rb + xb[x]) * 3, t0 = data[p] - data[b], t1 = data[p + 1] - data[b + 1], t2 = data[p + 2] - data[b + 2];
        d2[ra + x] = t0 * t0 + t1 * t1 + t2 * t2;
      }
    } else {
      for (let x = 0; x < W; x++) { const p = ra + x, b = rb + xb[x]; let sq = 0; for (let c = 0; c < ch; c++) { const t = data[p * ch + c] - data[b * ch + c]; sq += t * t; } d2[p] = sq; }
    }
  }
}
// integral image ของ d2 แล้วคำนวณน้ำหนัก w ต่อพิกเซลของ offset นี้ (หน้าต่างตัดที่ขอบภาพ) → wbuf · บวกเข้า wsum ตามลำดับพิกเซล
function nlmWeights(d2, W, H, ch, hh, dy, xb, xl, xh, yl, yh, I, wbuf, wsum) {
  const W1 = W + 1;
  for (let y = 0; y < H; y++) {
    let rs = 0;
    for (let x = 0; x < W; x++) { rs += d2[y * W + x]; I[(y + 1) * W1 + x + 1] = I[y * W1 + x + 1] + rs; }
  }
  for (let y = 0; y < H; y++) {
    const y0 = yl[y], y1 = yh[y], wy = y1 - y0 + 1, i1 = (y1 + 1) * W1, i0 = y0 * W1;
    for (let x = 0; x < W; x++) {
      const x0 = xl[x], x1 = xh[x];
      const S = (I[i1 + x1 + 1] - I[i0 + x1 + 1] - I[i1 + x0] + I[i0 + x0]) | 0;
      const w = Math.exp(-(S / (wy * (x1 - x0 + 1) * ch)) / hh);
      wbuf[y * W + x] = w;
      wsum[y * W + x] += w;
    }
  }
}
function nlmAccum1(data, W, H, dy, xb, wbuf, out) {
  for (let y = 0; y < H; y++) {
    const ty = y + dy, rb = (ty < 0 ? 0 : ty >= H ? H - 1 : ty) * W, ra = y * W;
    for (let x = 0; x < W; x++) out[ra + x] += wbuf[ra + x] * data[rb + xb[x]];
  }
}
function nlmAccum3(data, W, H, dy, xb, wbuf, out) {
  for (let y = 0; y < H; y++) {
    const ty = y + dy, rb = (ty < 0 ? 0 : ty >= H ? H - 1 : ty) * W, ra = y * W;
    for (let x = 0; x < W; x++) {
      const w = wbuf[ra + x], p3 = (ra + x) * 3, b3 = (rb + xb[x]) * 3;
      out[p3] += w * data[b3]; out[p3 + 1] += w * data[b3 + 1]; out[p3 + 2] += w * data[b3 + 2];
    }
  }
}
function nlmAccumN(data, W, H, ch, dy, xb, wbuf, out) {
  for (let y = 0; y < H; y++) {
    const ty = y + dy, rb = (ty < 0 ? 0 : ty >= H ? H - 1 : ty) * W, ra = y * W;
    for (let x = 0; x < W; x++) { const w = wbuf[ra + x], p = ra + x, b = rb + xb[x]; for (let c = 0; c < ch; c++) out[p * ch + c] += w * data[b * ch + c]; }
  }
}
export function nlmCore(data, W, H, ch, h, pr, sr) {
  const N = W * H, hh = h * h;
  const out = new Float64Array(N * ch), wsum = new Float64Array(N), wbuf = new Float64Array(N), d2 = new Int32Array(N), I = new Int32Array((W + 1) * (H + 1));
  const xb = new Int32Array(W), xl = new Int32Array(W), xh = new Int32Array(W), yl = new Int32Array(H), yh = new Int32Array(H);
  for (let x = 0; x < W; x++) { xl[x] = Math.max(0, x - pr); xh[x] = Math.min(W - 1, x + pr); }
  for (let y = 0; y < H; y++) { yl[y] = Math.max(0, y - pr); yh[y] = Math.min(H - 1, y + pr); }
  for (let dy = -sr; dy <= sr; dy++) {
    for (let dx = -sr; dx <= sr; dx++) {
      nlmD2(data, W, H, ch, dx, dy, xb, d2);
      nlmWeights(d2, W, H, ch, hh, dy, xb, xl, xh, yl, yh, I, wbuf, wsum);
      if (ch === 1) nlmAccum1(data, W, H, dy, xb, wbuf, out);
      else if (ch === 3) nlmAccum3(data, W, H, dy, xb, wbuf, out);
      else nlmAccumN(data, W, H, ch, dy, xb, wbuf, out);
    }
  }
  const res = new Uint8Array(N * ch);
  for (let p = 0; p < N; p++) {
    for (let c = 0; c < ch; c++) {
      const v = Math.floor(out[p * ch + c] / wsum[p] + 0.5);
      res[p * ch + c] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return res;
}

// ---- DCT 8×8 / JPEG ----
let DCT8_TABLE = null;
// C[u][x] = α(u)·cos((2x+1)uπ/16) · α(0) = √(1/8), α(u>0) = √(2/8) · Float64Array 64 (แถว u, คอลัมน์ x) · สร้างครั้งเดียว
export function dct8Table() {
  if (!DCT8_TABLE) {
    DCT8_TABLE = new Float64Array(64);
    for (let u = 0; u < 8; u++) {
      const a = u === 0 ? Math.sqrt(1 / 8) : Math.sqrt(2 / 8);
      for (let x = 0; x < 8; x++) DCT8_TABLE[u * 8 + x] = a * Math.cos(((2 * x + 1) * u * Math.PI) / 16);
    }
  }
  return DCT8_TABLE;
}
// F = C·B·Cᵀ (B, F = Float64Array 64 แถว y/u คอลัมน์ x/v) · แยกเป็นสองรอบ (T = C·B แล้ว F = T·Cᵀ)
export function dct8Forward(B, F) {
  const C = dct8Table(), T = new Float64Array(64);
  for (let u = 0; u < 8; u++) for (let x = 0; x < 8; x++) { let s = 0; for (let y = 0; y < 8; y++) s += C[u * 8 + y] * B[y * 8 + x]; T[u * 8 + x] = s; }
  for (let u = 0; u < 8; u++) for (let v = 0; v < 8; v++) { let s = 0; for (let x = 0; x < 8; x++) s += T[u * 8 + x] * C[v * 8 + x]; F[u * 8 + v] = s; }
  return F;
}
// B = Cᵀ·F·C
export function dct8Inverse(F, B) {
  const C = dct8Table(), T = new Float64Array(64);
  for (let y = 0; y < 8; y++) for (let v = 0; v < 8; v++) { let s = 0; for (let u = 0; u < 8; u++) s += C[u * 8 + y] * F[u * 8 + v]; T[y * 8 + v] = s; }
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) { let s = 0; for (let v = 0; v < 8; v++) s += T[y * 8 + v] * C[v * 8 + x]; B[y * 8 + x] = s; }
  return B;
}
// ตารางความสว่างมาตรฐาน JPEG Annex K.1 (แถวบน → ล่าง)
const JPEG_Q50 = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99];
// สเกลแบบ IJG: S = q < 50 ? floor(5000/q) : 200 − 2q · Qs = max(1, floor((Q·S + 50)/100)) · q ต้องเป็นจำนวนเต็ม 1–100
export function jpegQTable(quality) {
  const S = quality < 50 ? Math.floor(5000 / quality) : 200 - 2 * quality;
  return Int32Array.from(JPEG_Q50, (q) => Math.max(1, Math.floor((q * S + 50) / 100)));
}

// ===================================================================
// ตัวช่วยของ P-8 (2026-10-17 — pass_fail: ตัดสิน OK / NG จากค่าที่กล่องวัดก่อนหน้าส่งมา)
// ===================================================================
// worker ส่ง ctx.upstream = { op, values } ของกล่อง "ล่าสุดที่มี ctx.values" บนสายหลัก (op = id จริงของ op เช่น "regionprops")
// pass_fail เลือกตัวเลขจาก values ตาม metric · ต้องมาจาก op ที่ถูกต้อง (op) ไม่งั้น = ไม่มีค่า → NG (fail-safe)
// คอลัมน์พื้นที่: ใช้ area_mm2 ถ้ากล่อง Region Properties ตั้ง มม./px > 0 (มีคอลัมน์นี้) ไม่งั้น area_px — แถวที่ส่งมาถูกจำกัดด้วย maxObjects
function pfAreaCells(v) {
  if (!v || v.kind !== "table" || !Array.isArray(v.columns) || !Array.isArray(v.rows) || v.rows.length === 0) return null;
  let ci = v.columns.indexOf("area_mm2");
  if (ci < 0) ci = v.columns.indexOf("area_px");
  if (ci < 0) return null;
  return v.rows.map((r) => (Array.isArray(r) ? r[ci] : null));
}
function pfScalar(v, key) {
  if (!v || v.kind !== "scalars" || !Array.isArray(v.fields)) return null;
  const f = v.fields.find((x) => x && x.key === key);
  return f ? f.value : null;
}
// คืนตัวเลข (หรือ null/NaN = ไม่มีค่า) · ผลรวมบวกตามลำดับแถวจากบนลงล่าง (Python ที่ export บวกลำดับเดียวกัน)
function pfSum(cells) {
  if (!cells) return null;
  let s = 0;
  for (const x of cells) { if (typeof x !== "number") return NaN; s += x; }
  return s;
}
function pfMax(cells) {
  if (!cells) return null;
  let m = -Infinity;
  for (const x of cells) { if (typeof x !== "number") return NaN; if (x > m) m = x; }
  return m;
}
export const PASS_FAIL_METRICS = {
  count: { op: "regionprops", th: "Region Properties", pick: (v) => (v && v.kind === "table" ? (typeof v.total === "number" ? v.total : (Array.isArray(v.rows) ? v.rows.length : null)) : null) },
  area_sum: { op: "regionprops", th: "Region Properties", pick: (v) => pfSum(pfAreaCells(v)) },
  area_max: { op: "regionprops", th: "Region Properties", pick: (v) => pfMax(pfAreaCells(v)) },
  roi_mean: { op: "roi_polygon", th: "ROI Polygon", pick: (v) => pfScalar(v, "roi_mean") },
  focus: { op: "focus_score", th: "Focus / Sharpness Score", pick: (v) => pfScalar(v, "focus_score") },
  de_mean: { op: "delta_e", th: "Color Difference ΔE", pick: (v) => pfScalar(v, "de_mean") },
  iou: { op: "seg_score", th: "Segmentation Score", pick: (v) => pfScalar(v, "iou") },
  dice: { op: "seg_score", th: "Segmentation Score", pick: (v) => pfScalar(v, "dice") },
  piqe: { op: "piqe", th: "PIQE", pick: (v) => pfScalar(v, "piqe") },
};
// เกณฑ์แสดงบนข้อความ: ปัด 2 ตำแหน่งแล้วตัดศูนย์ท้าย (0.30000000000000004 → 0.3 · 1 → 1)
export const pfNum = (x) => String(Math.round(x * 100) / 100);
const PASS_FAIL_CONDS = { range: true, ge: true, le: true, eq: true };
const PASS_FAIL_FIELDS = [
  ["result", "ผลตัดสิน (OK / NG)"], ["measured", "ค่าที่วัดได้"], ["lo", "เกณฑ์ต่ำสุด (หรือค่าเป้าหมาย)"], ["hi", "เกณฑ์สูงสุด"],
  ["reason", "เหตุผล (NG เพราะไม่มีค่า หรือเกณฑ์ต่ำสุด > สูงสุด)"],
];
const PASS_FAIL_LIMIT = 10000000; // ช่วง lo/hi ของ param (ค่าที่วัดทุกตัวไม่ติดลบ)
// param → ค่าที่ใช้จริง (own-property สำหรับ select · clamp ตัวเลข) — แอปและ codegen ใช้ตัวเดียวกัน
export function passFailParams(p) {
  p = p || {};
  return {
    metric: ownKey(PASS_FAIL_METRICS, p.metric, "count"),
    cond: ownKey(PASS_FAIL_CONDS, p.cond, "range"),
    lo: Math.min(PASS_FAIL_LIMIT, Math.max(0, finiteOr(p.lo, 1))),
    hi: Math.min(PASS_FAIL_LIMIT, Math.max(0, finiteOr(p.hi, 100))),
  };
}
// ตัดสินล้วน ๆ (ไม่แตะ cv) → { result: "OK"|"NG", measured, lo, hi, reason } · lo/hi = null เมื่อเงื่อนไขไม่ใช้ · ไม่มีค่า = NG + reason เสมอ
export function passFailVerdict(p, upstream) {
  const { metric, cond, lo, hi } = passFailParams(p);
  const m = PASS_FAIL_METRICS[metric];
  const useLo = cond !== "le", useHi = cond === "range" || cond === "le";
  const out = { result: "NG", measured: null, lo: useLo ? lo : null, hi: useHi ? hi : null, reason: null };
  const v = upstream && upstream.values;
  if (!v) { out.reason = "ไม่มีค่าจากกล่องก่อนหน้า (ต้องต่อหลังกล่องที่วัดค่า)"; return out; }
  if (upstream.op !== m.op) { out.reason = `ค่าล่าสุดมาจากกล่องอื่น — ต้องต่อหลังกล่อง ${m.th}`; return out; }
  const x = m.pick(v);
  if (typeof x !== "number" || !Number.isFinite(x)) {
    out.reason = v.kind === "table" && Array.isArray(v.rows) && v.rows.length === 0 && metric !== "count"
      ? "ไม่พบวัตถุ (ตารางว่าง) จึงวัดค่านี้ไม่ได้" : "ค่าที่ต้องใช้ว่างหรือไม่ใช่ตัวเลข";
    return out;
  }
  out.measured = x;
  if (cond === "range" && lo > hi) { out.reason = "เกณฑ์ต่ำสุดมากกว่าเกณฑ์สูงสุด"; return out; }
  const ok = cond === "range" ? x >= lo && x <= hi
    : cond === "ge" ? x >= lo
    : cond === "le" ? x <= hi
    : Math.abs(x - lo) <= 1e-9 * Math.max(1, Math.abs(lo)); // "=" เทียบทศนิยมแบบยอมคลาด 1e-9 (จำนวนเต็มเท่ากันเป๊ะ)
  out.result = ok ? "OK" : "NG";
  return out;
}

// ===================================================================
// OPS — operation ย่อยแต่ละตัว (logic เดิม ไม่เปลี่ยน)
// ===================================================================
export const OPS = {
  // ---------------- 1. Color Space ----------------
  grayscale: {
    name: "Grayscale",
    needs: "any",
    desc: "ทิ้งสีทิ้งไป เหลือแต่ความสว่างขาว-ดำ ตาคนเราไวกับสีเขียวมากสุด เลยให้เขียวมีน้ำหนักเยอะกว่าแดงกับน้ำเงิน งานส่วนใหญ่อย่างหาขอบหรือแยกวัตถุ เริ่มจากตรงนี้ก่อนเสมอ เพราะเหลือข้อมูลชั้นเดียว คิดง่ายกว่ากันเยอะ",
    params: [],
    apply(cv, src) {
      const dst = new cv.Mat();
      if (src.channels() === 1) src.copyTo(dst);
      else cv.cvtColor(src, dst, cv.COLOR_RGB2GRAY);
      return dst;
    },
  },

  colorspace: {
    name: "Color Space Convert",
    needs: "any",
    desc: "แปลงภาพจาก RGB ไปยังปริภูมิสีอื่น แล้วแสดงค่าแต่ละ channel เป็นสีตรง ๆ (false-color) เพื่อให้เห็นว่าข้อมูลถูกจัดเก็บต่างกันอย่างไร — HSV/HLS แยกเนื้อสีออกจากความสว่าง, Lab ใกล้การรับรู้ของมนุษย์, YCrCb ใช้ในระบบวิดีโอ/JPEG · XYZ/Luv/YUV: ภาพ 8 bit ทุกช่องถูกบีบให้อยู่ใน 0–255 ตัวเลขจึงไม่ใช่ค่าจริงตามตำรา — ถ้าจะวัดความต่างสีเป็นตัวเลขจริงให้ใช้ Color Difference (ΔE)",
    params: [
      { key: "space", type: "select", label: "ปริภูมิสี", options: [
        { value: "hsv", label: "HSV (เนื้อสี-ความอิ่ม-ความสว่าง)", hint: "แยกเนื้อสีออกจากความสว่าง เหมาะกับตรวจจับวัตถุด้วยสี" },
        { value: "hls", label: "HLS", hint: "คล้าย HSV แต่สลับลำดับ Lightness/Saturation" },
        { value: "lab", label: "Lab", hint: "ใกล้เคียงการรับรู้สีของมนุษย์ นิยมใช้เทียบความต่างสี" },
        { value: "ycrcb", label: "YCrCb", hint: "แยกความสว่าง (Y) ออกจากสี ใช้ในวิดีโอ/JPEG" },
        { value: "xyz", label: "XYZ (CIE 1931)", hint: "ปริภูมิมาตรฐานที่ใช้เป็นตัวกลางแปลงสี · Y ≈ ความสว่าง" },
        { value: "luv", label: "Luv (CIE L*u*v*)", hint: "คล้าย Lab ใกล้การรับรู้ของคน · ภาพ 8 bit เก็บ L×255/100 และเลื่อน u, v ให้เป็นบวก" },
        { value: "yuv", label: "YUV", hint: "Y = ความสว่าง U/V = ความต่างสี ใช้ในวิดีโอ (คล้าย YCrCb แต่สูตรต่างกันเล็กน้อย)" },
      ], default: "hsv" },
    ],
    apply(cv, src, p) {
      const rgb = new cv.Mat();
      if (src.channels() === 1) cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
      else src.copyTo(rgb);
      const dst = new cv.Mat();
      const code = { hsv: cv.COLOR_RGB2HSV, hls: cv.COLOR_RGB2HLS, lab: cv.COLOR_RGB2Lab, ycrcb: cv.COLOR_RGB2YCrCb, xyz: cv.COLOR_RGB2XYZ, luv: cv.COLOR_RGB2Luv, yuv: cv.COLOR_RGB2YUV }[p.space];
      cv.cvtColor(rgb, dst, code);
      rgb.delete();
      return dst;
    },
  },

  channel_split: {
    name: "Channel Split",
    needs: "any",
    desc: "ดึงข้อมูลออกมาดูทีละ channel เป็นภาพขาวดำ — เช่น ช่อง H (เนื้อสี) จะสว่างตามชนิดของสี ไม่ใช่ความสว่างจริง ช่วยให้เข้าใจว่าแต่ละปริภูมิสีเก็บอะไรไว้ตรงไหน",
    params: [
      { key: "channel", type: "select", label: "ช่องที่ดู", options: [
        { value: "r", label: "R (แดง)" },
        { value: "g", label: "G (เขียว)" },
        { value: "b", label: "B (น้ำเงิน)" },
        { value: "h", label: "H (เนื้อสี)" },
        { value: "s", label: "S (ความอิ่มสี)" },
        { value: "v", label: "V (ความสว่าง)" },
      ], default: "h" },
    ],
    apply(cv, src, p) {
      const rgb = new cv.Mat();
      if (src.channels() === 1) cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
      else src.copyTo(rgb);
      const spaceMat = new cv.Mat();
      const isHSV = ["h", "s", "v"].includes(p.channel);
      if (isHSV) cv.cvtColor(rgb, spaceMat, cv.COLOR_RGB2HSV);
      else rgb.copyTo(spaceMat);
      const idx = { r: 0, g: 1, b: 2, h: 0, s: 1, v: 2 }[p.channel];
      const vec = new cv.MatVector();
      cv.split(spaceMat, vec);
      const dst = new cv.Mat();
      const ch = vec.get(idx);
      ch.copyTo(dst);
      ch.delete(); // MatVector.get() คืน Mat wrapper ใหม่ทุกครั้ง — ต้อง delete เอง ไม่งั้นรั่วทุกเฟรม
      vec.delete();
      spaceMat.delete();
      rgb.delete();
      return dst;
    },
  },

  rgb_adjust: {
    name: "RGB Channel Adjust",
    needs: "any",
    desc: "ปรับความสว่างของแต่ละช่องสี R/G/B แยกจากกัน (ใช้ผลคูณ (Gain) และการบวก (Offset) แบบเดียวกับ Brightness/Contrast แต่ทำทีละช่อง) — เพิ่ม R อย่างเดียวจะทำให้ภาพออกโทนแดง โดยไม่กระทบ G กับ B",
    params: [
      { key: "rGain", type: "number", label: "แดง (R) — ผลคูณ", min: 0, max: 3, step: 0.05, default: 1, desc: "คูณความสว่างเฉพาะช่องแดง" },
      { key: "rOffset", type: "number", label: "แดง (R) — บวก/ลบ", min: -100, max: 100, step: 1, default: 0, desc: "บวก/ลบความสว่างเฉพาะช่องแดง — ค่าลบทำให้มืดลงจริง (ไม่ใช่กลับค่า)" },
      { key: "gGain", type: "number", label: "เขียว (G) — ผลคูณ", min: 0, max: 3, step: 0.05, default: 1, desc: "คูณความสว่างเฉพาะช่องเขียว" },
      { key: "gOffset", type: "number", label: "เขียว (G) — บวก/ลบ", min: -100, max: 100, step: 1, default: 0, desc: "บวก/ลบความสว่างเฉพาะช่องเขียว — ค่าลบทำให้มืดลงจริง (ไม่ใช่กลับค่า)" },
      { key: "bGain", type: "number", label: "น้ำเงิน (B) — ผลคูณ", min: 0, max: 3, step: 0.05, default: 1, desc: "คูณความสว่างเฉพาะช่องน้ำเงิน" },
      { key: "bOffset", type: "number", label: "น้ำเงิน (B) — บวก/ลบ", min: -100, max: 100, step: 1, default: 0, desc: "บวก/ลบความสว่างเฉพาะช่องน้ำเงิน — ค่าลบทำให้มืดลงจริง (ไม่ใช่กลับค่า)" },
    ],
    apply(cv, src, p) {
      const rgb = new cv.Mat();
      if (src.channels() === 1) cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
      else src.copyTo(rgb);
      const vec = new cv.MatVector();
      cv.split(rgb, vec);
      const rCh = vec.get(0);
      const gCh = vec.get(1);
      const bCh = vec.get(2);
      const rOut = new cv.Mat();
      const gOut = new cv.Mat();
      const bOut = new cv.Mat();
      // convertTo (ไม่ใช่ convertScaleAbs) — convertScaleAbs หาค่าสัมบูรณ์ก่อนตัด ทำให้ offset ติดลบ
      // "สะท้อนค่ากลับ" แทนที่จะทำให้มืดลงจริง ส่วน convertTo คำนวณ α·x+β แล้วตัดขอบ 0–255 ตรง ๆ
      rCh.convertTo(rOut, -1, p.rGain, p.rOffset);
      gCh.convertTo(gOut, -1, p.gGain, p.gOffset);
      bCh.convertTo(bOut, -1, p.bGain, p.bOffset);
      const merged = new cv.MatVector();
      merged.push_back(rOut);
      merged.push_back(gOut);
      merged.push_back(bOut);
      const dst = new cv.Mat();
      cv.merge(merged, dst);
      rCh.delete();
      gCh.delete();
      bCh.delete();
      rOut.delete();
      gOut.delete();
      bOut.delete();
      vec.delete();
      merged.delete();
      rgb.delete();
      return dst;
    },
  },

  // ---------------- 2. Enhancement ----------------
  brightness_contrast: {
    name: "Brightness / Contrast",
    needs: "any",
    desc: "ปรับภาพแบบเชิงเส้น: ใหม่ = α·เดิม + β โดย α (contrast) ยืด/บีบช่วงความสว่าง และ β (brightness) เลื่อนความสว่างทั้งภาพขึ้นลง — เป็นการปรับปรุงภาพขั้นพื้นฐานที่สุด",
    params: [
      { key: "alpha", type: "number", label: "คอนทราสต์ (α)", min: 0.1, max: 3, step: 0.1, default: 1.2, desc: "คูณความสว่างทุกจุด — มาก = ต่างสว่าง-มืดชัดขึ้น, น้อยกว่า 1 = ภาพแบนลง" },
      { key: "beta", type: "number", label: "ความสว่าง (β)", min: -100, max: 100, step: 1, default: 0, desc: "บวก/ลบความสว่างทั้งภาพเท่า ๆ กันทุกจุด" },
    ],
    apply(cv, src, p) {
      // α·x+β แล้วปัดครึ่งขึ้นและตัด 0–255 (เหมือน imlincomb ของ MATLAB) — เดิมใช้ convertScaleAbs ซึ่งหาค่าสัมบูรณ์ก่อนตัด
      // β ติดลบจึง "สะท้อนค่ากลับ" (จุดมืดสุดกลับสว่างขึ้น) · ทำเป็น LUT 256 ค่าด้วย double ให้ตรงกับ Python ที่ export ทุกค่า
      const a = Number(p.alpha) || 0, b = Number(p.beta) || 0; // = num() ของ codegen (โปรเจกต์ที่แก้มือเก็บเป็นสตริงได้)
      const lutArr = new Array(256);
      for (let i = 0; i < 256; i++) lutArr[i] = Math.min(255, Math.max(0, Math.floor(i * a + b + 0.5)));
      const lut = cv.matFromArray(1, 256, cv.CV_8UC1, lutArr);
      const dst = new cv.Mat();
      try {
        cv.LUT(src, lut, dst);
        return dst;
      } catch (err) {
        dst.delete();
        throw err;
      } finally {
        lut.delete();
      }
    },
  },

  gamma: {
    name: "Gamma Correction",
    needs: "any",
    desc: "ปรับความสว่างแบบไม่เชิงเส้นด้วยเส้นโค้ง power-law: ใหม่ = (เดิม/255)^(1/γ)·255 — γ>1 ดึงส่วนมืดให้สว่างขึ้นโดยไม่ทำให้ส่วนสว่างแตก (ต่างจากการเพิ่ม brightness ตรง ๆ) ใช้แก้ภาพย้อนแสง/มืดเกิน · Log: ใหม่ = 255·ln(1+เดิม)/ln(256) เป็นการแปลงความเข้มแบบที่ 3 ในตำรา (negative, log, power-law) ใช้ดูภาพที่ช่วงค่ากว้างมาก เช่น สเปกตรัม FFT · ช่วงมืดถูกดึงขึ้นแรงกว่า γ ทั่วไป",
    params: [
      { key: "g", visibleWhen: [{key:"curve",in:["power"]}], type: "number", label: "ค่า Gamma (γ)", min: 0.2, max: 3, step: 0.1, default: 1.5, desc: "โค้งความสว่าง — γ>1 ดึงส่วนมืดให้สว่างขึ้น, γ<1 กดส่วนสว่างให้มืดลง" },
      { key: "curve", type: "select", label: "รูปโค้ง", options: [
        { value: "power", label: "Power-law / Gamma (ใช้ค่า γ)" },
        { value: "log", label: "Log: c·log(1+r) (ไม่ใช้ค่า γ)", hint: "ขยายช่วงมืดให้เห็นรายละเอียด บีบช่วงสว่าง · c = 255/log(256) ให้ 255 → 255" },
      ], default: "power" },
    ],
    apply(cv, src, p, ctx) {
      if (p.curve === "log") {
        if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
        const logArr = new Array(256);
        for (let i = 0; i < 256; i++) logArr[i] = Math.floor(Math.log1p(i) * (255 / Math.log(256)) + 0.5);
        const logLut = cv.matFromArray(1, 256, cv.CV_8UC1, logArr);
        const out = new cv.Mat();
        try {
          cv.LUT(src, logLut, out);
          if (ctx) ctx.info = `Log transform: ใหม่ = 255·ln(1+เดิม)/ln(256) — เดิม 10 → ${logArr[10]}, 128 → ${logArr[128]} (ค่า γ ไม่ถูกใช้)`;
          return out;
        } catch (err) {
          out.delete();
          throw err;
        } finally {
          logLut.delete();
        }
      }
      const lutArr = new Array(256);
      for (let i = 0; i < 256; i++) lutArr[i] = Math.min(255, Math.round(Math.pow(i / 255, 1 / p.g) * 255));
      const lut = cv.matFromArray(1, 256, cv.CV_8UC1, lutArr);
      const dst = new cv.Mat();
      cv.LUT(src, lut, dst);
      lut.delete();
      return dst;
    },
  },

  hist_equalize: {
    name: "Histogram Equalization",
    needs: "gray",
    desc: "กระจายความสว่างให้ทั่วช่วง 0–255 อย่างสม่ำเสมอโดยดูจาก histogram ทั้งภาพ — ภาพที่ซีด/ทึบ (ความสว่างกระจุกตัว) จะมีรายละเอียดชัดขึ้นทันที จุดอ่อนคือถ้าภาพมีบริเวณสว่างจัดจะถูกดึงไปทั้งภาพ",
    params: [],
    apply(cv, src) {
      const dst = new cv.Mat();
      cv.equalizeHist(src, dst);
      return dst;
    },
  },

  clahe: {
    name: "CLAHE",
    needs: "gray",
    desc: "Histogram Equalization แบบแบ่งภาพเป็นตารางเล็ก ๆ แล้วปรับแยกกัน พร้อมจำกัดคอนทราสต์ (clip limit) กัน noise ถูกขยาย — แก้จุดอ่อนของ Equalization ธรรมดาเมื่อแสงในภาพไม่สม่ำเสมอ",
    params: [
      { key: "clip", type: "number", label: "Clip limit", min: 1, max: 10, step: 0.5, default: 2, desc: "เพดานคอนทราสต์ต่อตาราง — สูง = ดันคอนทราสต์แรงขึ้นแต่ noise เด่นขึ้นด้วย" },
      { key: "tile", type: "number", label: "ขนาดตาราง", min: 2, max: 16, step: 1, default: 8, desc: "จำนวนช่องที่แบ่งภาพต่อด้าน — มาก = ปรับละเอียดขึ้นแต่อาจเห็นรอยต่อตาราง" },
    ],
    apply(cv, src, p) {
      const dst = new cv.Mat();
      const clahe = new cv.CLAHE(p.clip, new cv.Size(p.tile, p.tile));
      clahe.apply(src, dst);
      clahe.delete();
      return dst;
    },
  },

  // Contrast Stretch / imadjust (IP batch 2026-10-01, ENHANCE) — ยืดช่วงความสว่างโดยตัดหัว-ท้ายของ
  // histogram ทิ้งก่อนยืด (ต่างจาก hist_equalize ที่กระจายทั้ง histogram ใหม่หมด) อยู่ใน node เดียวกับ
  // hist_equalize/clahe ให้เทียบกันได้ตรง ๆ (ทั้งคู่ needs:"gray" เหมือนกัน)
  contrast_stretch: {
    name: "Contrast Stretch (Imadjust)",
    needs: "gray",
    desc: "ยืดช่วงความสว่างที่ใช้จริงในภาพให้เต็ม 0–255 โดยตัดส่วนมืดสุด/สว่างสุดทิ้งก่อนตามเปอร์เซ็นต์ที่ตั้งไว้ (กัน noise สุดขั้วไม่กี่จุดดึงช่วงทั้งหมด) — ต่างจาก Brightness/Contrast ที่ปรับแบบตายตัวไม่สนใจเนื้อภาพ และต่างจาก Histogram Equalization ที่กระจาย histogram ใหม่ทั้งหมด (ซึ่งอาจทำให้ภาพดูไม่เป็นธรรมชาติ) เป็นเครื่องมือปรับคอนทราสต์ที่ใช้บ่อยมากในงาน MATLAB (imadjust/stretchlim)",
    params: [
      { key: "lowPct", type: "number", label: "ตัดขอบมืด (%)", min: 0, max: 10, step: 0.5, default: 1, desc: "ตัดพิกเซลมืดสุดกี่ % ทิ้งก่อนยืด — กัน noise จุดดำไม่กี่จุดทำให้ยืดช่วงผิด" },
      { key: "highPct", type: "number", label: "ตัดขอบสว่าง (%)", min: 0, max: 10, step: 0.5, default: 1, desc: "ตัดพิกเซลสว่างสุดกี่ % ทิ้งก่อนยืด — กัน noise จุดขาวไม่กี่จุดทำให้ยืดช่วงผิด" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const lowPct = numIn(p.lowPct, 1, 0, 10);
      const highPct = numIn(p.highPct, 1, 0, 10);
      let hist = null, srcVec = null, mask = null, dst = null, lut = null;
      try {
        hist = new cv.Mat();
        srcVec = new cv.MatVector();
        srcVec.push_back(src);
        mask = new cv.Mat(); // mask ว่าง = ไม่กรอง ใช้ทุกพิกเซล (ธรรมเนียมเดิมของ cv.calcHist)
        // channel [0] เท่านั้น: สัญญาของ op นี้คือ needs:"gray" — cv-worker.js แปลงเป็น 1 ช่องให้เสมอก่อน
        // เรียก apply() จริง จึงรับประกันว่า channel 0 คือช่องเดียวที่มี (เหมือน hist_equalize/clahe ที่ก็
        // พึ่งสัญญาเดียวกันและจะพังถ้าได้ภาพหลายช่องตรง ๆ) — ถ้าเรียก apply() ตรง ๆ นอก runPipeline ด้วยภาพ
        // สี (เช่นในเทสต์ที่จงใจทดสอบ rgba) จะได้ histogram จากช่อง R เท่านั้นแล้วใช้ LUT เดียวกันทาทุกช่อง
        // (ไม่ throw ไม่ garbage แต่ไม่ใช่ "contrast stretch ต่อช่อง") — ยอมรับได้ตามสัญญาเดิมของไฟล์
        cv.calcHist(srcVec, [0], mask, hist, [256], [0, 256]);
        const h = hist.data32F; // cache getter ก่อนเข้าลูปเสมอ (กฎเหล็กของไฟล์นี้)
        const total = src.rows * src.cols;
        const lowCount = (total * lowPct) / 100;
        const highCount = (total * (100 - highPct)) / 100;
        // หา low = ค่าความสว่างแรกที่ cumulative > lowPct% ของพิกเซลทั้งหมด (strict ">" — เหมือน stretchlim
        //   ของ MATLAB: find(cdf > tol_low, 1)) ถ้าใช้ ">=" ตอน lowPct=0 → lowCount=0 และ cum>=0 จริงตั้งแต่
        //   bin 0 ทั้งที่ bin 0 ว่าง ทำให้ low=0 ผิดเสมอ (ภาพที่มืดสุดคือ 50 จะไม่ถูกยืดฝั่งมืดเลย)
        // หา high = ค่าความสว่างแรกที่ cumulative >= (100-highPct)% ของพิกเซลทั้งหมด (">=" — find(cdf >= tol_high, 1)
        //   ตอน highPct=0 → highCount=total ได้ค่าสว่างสุดจริงของภาพพอดี)
        let cum = 0;
        let low = 0;
        let high = 255;
        let foundLow = false;
        let foundHigh = false;
        for (let i = 0; i < 256; i++) {
          cum += h[i];
          if (!foundLow && cum > lowCount) { low = i; foundLow = true; }
          if (!foundHigh && cum >= highCount) { high = i; foundHigh = true; break; }
        }
        // ภาพความสว่างสม่ำเสมอเกือบทั้งหมด (เช่น flat field) → high <= low → ห้ามหาร 0 คืนภาพเดิม
        if (high <= low) {
          if (ctx) ctx.info = "ภาพมีความสว่างสม่ำเสมอ ไม่มีช่วงให้ยืด";
          return src.clone();
        }
        const lutArr = new Array(256);
        for (let i = 0; i < 256; i++) {
          let v;
          if (i <= low) v = 0;
          else if (i >= high) v = 255;
          else v = Math.round(((i - low) / (high - low)) * 255);
          lutArr[i] = Math.max(0, Math.min(255, v));
        }
        lut = cv.matFromArray(1, 256, cv.CV_8UC1, lutArr);
        dst = new cv.Mat();
        cv.LUT(src, lut, dst);
        if (ctx) ctx.info = `ยืดช่วง [${low}, ${high}] → [0, 255]`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (hist) hist.delete();
        if (srcVec) srcVec.delete();
        if (mask) mask.delete();
        if (lut) lut.delete();
      }
    },
  },

  // Flat-field Correction / imflatfield (IP batch 2026-10-01, ENHANCE) — ประมาณพื้นหลัง (แสงไม่สม่ำเสมอ)
  // ด้วย Gaussian Blur รัศมีใหญ่มาก แล้วหารภาพต้นทางด้วยพื้นหลังนั้น คูณกลับด้วยค่าเฉลี่ยของพื้นหลังเพื่อให้
  // ความสว่างโดยรวมใกล้เคียงเดิม — ตัดสินใจรวมไว้ใน node เดียวกับ hist_equalize/clahe/contrast_stretch
  // (ไม่แยก node ใหม่) เพราะ param set มีแค่ตัวเดียว (ขนาดเบลอ) ไม่ต่างจากกลุ่มเดิมมากพอจะทำให้ node รก —
  // บันทึกเหตุผลนี้ไว้ตาม WORKLOG ด้วย (qa สั่งให้ตัดสินใจเองได้)
  flatfield_correction: {
    name: "Flat-field Correction",
    needs: "any",
    desc: "แก้ปัญหาแสงไม่สม่ำเสมอทั่วภาพ (เช่น ขอบภาพมืดกว่ากลางภาพ) โดยประมาณ 'พื้นหลังของแสง' ด้วยการเบลอภาพแรงมาก (เบลอจนไม่เหลือรายละเอียดวัตถุ เหลือแต่แนวโน้มความสว่างกว้าง ๆ) แล้วหารภาพต้นทางด้วยพื้นหลังนั้น — บริเวณที่สว่างเกินจริงเพราะแสงจะถูกลดลง บริเวณที่มืดเกินจริงจะถูกดึงขึ้น ทำให้ Threshold ขั้นตอนถัดไปทำงานแม่นยำขึ้นมาก (ต่างจาก Top-hat ที่ลบพื้นหลังทิ้งทั้งหมดเหลือวัตถุสว่างบนพื้นดำ เหมาะก่อน threshold และต้องให้วัตถุเล็กกว่าขนาด kernel — ตัวนี้แค่หารแสงที่ไม่สม่ำเสมอออก แต่ยังคงความสว่างโดยรวมของพื้นหลังไว้ ภาพจึงยังดูเป็นภาพปกติ) ขนาดเบลอต้องใหญ่กว่าวัตถุที่สนใจมาก ๆ ไม่งั้นจะ 'กิน' รายละเอียดของวัตถุไปด้วยไม่ใช่แค่แสง — ถ้าวัตถุเริ่มจางหายไป ให้เพิ่มขนาดเบลอขึ้น",
    params: [
      { key: "blurSize", type: "number", label: "ขนาดเบลอพื้นหลัง (px)", min: 15, max: 401, step: 2, default: 121, hint: "เลขคี่ — ต้องใหญ่กว่าวัตถุมาก ๆ ไม่งั้นจะกินรายละเอียดวัตถุไปด้วย · ประมาณพื้นหลังบนภาพย่อ ≤240px เสมอก่อนขยายกลับ จึงเร็วแม้ที่ความละเอียดสูง (1080p ที่ค่าเริ่มต้น ~0.05 วินาที/เฟรม)", desc: "ขนาดตัวกรอง Gaussian Blur ที่ใช้ประมาณพื้นหลังของแสง — ยิ่งใหญ่ยิ่งได้เฉพาะแนวโน้มแสงกว้าง ๆ" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const k = ensureOdd(intIn(p.blurSize, 121, 15, 401));
      // พื้นหลังของแสงที่ได้จาก GaussianBlur รัศมีใหญ่เป็นสัญญาณความถี่ต่ำมากอยู่แล้วโดยนิยาม (นั่นคือเหตุผล
      // ที่ใช้มันเป็น "พื้นหลัง") จึงย่อภาพลงก่อนเบลอ แล้วขยายพื้นหลังที่ได้กลับเป็นขนาดเต็มด้วย cv.resize
      // ก่อนนำไปหารกับภาพต้นฉบับความละเอียดเต็ม — ขั้นตอนที่ทำงานกับความละเอียดเต็มทุกพิกเซล (หาร/คูณกลับ
      // ด้วยค่าเฉลี่ย/convertTo ด้านล่างทั้งหมด) ยังคงอยู่กับ src เต็มความละเอียดเดิมเสมอ ไม่ใช่การย่อผลลัพธ์
      // สุดท้าย (ต่างจาก skeleton/local_entropy_filter ที่ย่อทั้งการวิเคราะห์และผลลัพธ์) — qa วัดจริงแล้วเร็ว
      // ขึ้น ~37 เท่าที่ 1920x1080 (2016ms → 54ms) แทบไม่เสียคุณภาพ (maxdiff ≤7/255 เทียบ Gaussian เต็ม
      // ความละเอียดตรง ๆ) เพราะพื้นหลังที่ถูกต้องคือแค่ค่าประมาณจาก Gaussian blur อยู่แล้ว ไม่ใช่ค่าจริงทาง
      // ฟิสิกส์ — ดู WORKLOG
      const maxAnalysisSide = 240;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      // ขนาดเบลอต้องย่อตามสัดส่วนเดียวกับภาพ ไม่งั้นพื้นหลังที่ได้จะ "แคบ" กว่าที่ผู้ใช้ตั้งไว้มาก — ต้องตรงกับ
      // สูตรฝั่ง codegen.js เป๊ะ (round แบบเดียวกัน) เพื่อรักษา parity
      const bgK = ensureOdd(Math.max(3, Math.round(k * scale)));
      let analysis = null, bgSmall = null, bg = null, srcF = null, bgF = null, epsMat = null, bgEps = null, ratio = null, meanMat = null, scaled = null, dst = null;
      try {
        if (scale < 1) {
          analysis = new cv.Mat();
          const aw = Math.max(1, Math.round(src.cols * scale));
          const ah = Math.max(1, Math.round(src.rows * scale));
          cv.resize(src, analysis, new cv.Size(aw, ah), 0, 0, cv.INTER_AREA);
          bgSmall = new cv.Mat();
          cv.GaussianBlur(analysis, bgSmall, new cv.Size(bgK, bgK), 0, 0, cv.BORDER_DEFAULT);
          bg = new cv.Mat();
          cv.resize(bgSmall, bg, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_LINEAR);
        } else {
          bg = new cv.Mat();
          cv.GaussianBlur(src, bg, new cv.Size(k, k), 0, 0, cv.BORDER_DEFAULT);
        }
        // cv.mean คืน plain array [c0,c1,c2,c3] ในบิลด์นี้ (ยืนยันแล้ว) ไม่ใช่ wrapper — ไม่ต้อง .delete()
        const meanVal = cv.mean(bg);
        srcF = new cv.Mat();
        src.convertTo(srcF, cv.CV_32F);
        bgF = new cv.Mat();
        bg.convertTo(bgF, cv.CV_32F);
        // บวก epsilon กันหารศูนย์ (พื้นหลังมืดสนิทบางจุด) — ต้องเป็น Mat เพราะ cv.add ในบิลด์นี้ไม่รับ
        // Scalar ตรง ๆ (ยืนยันแล้ว: throw "Cannot pass ... as a Mat")
        epsMat = new cv.Mat(bgF.rows, bgF.cols, bgF.type(), new cv.Scalar(1, 1, 1, 1));
        bgEps = new cv.Mat();
        cv.add(bgF, epsMat, bgEps);
        ratio = new cv.Mat();
        cv.divide(srcF, bgEps, ratio);
        // คูณกลับด้วยค่าเฉลี่ยของพื้นหลัง "ต่อช่อง" (ไม่ใช่ค่าเดียวทั้งภาพ) — สร้าง Mat คงที่ต่อช่องแล้ว
        // multiply ตรง ๆ (ด้วยเหตุผลเดียวกับ epsMat ด้านบน: cv.multiply ก็ไม่รับ Scalar ตรง ๆ เช่นกัน)
        // ภาพ 4 ช่อง (RGBA): meanVal[3] คือค่าเฉลี่ย alpha ด้วย → alpha channel ถูกปรับตามสูตรเดียวกับช่องสี
        // (ratio*mean) แทนที่จะคงค่าเดิมหรือบังคับ opaque — เป็นไปตามธรรมเนียมเดิมของไฟล์นี้ที่ op อื่น เช่น
        // brightness_contrast (LUT) ก็ประมวลผล alpha เหมือนทุกช่องเช่นกัน ไม่ใช่บั๊ก
        meanMat = new cv.Mat(ratio.rows, ratio.cols, ratio.type(), new cv.Scalar(meanVal[0], meanVal[1], meanVal[2], meanVal[3]));
        scaled = new cv.Mat();
        cv.multiply(ratio, meanMat, scaled);
        dst = new cv.Mat();
        // scaled ไม่มีทางติดลบ (src/bgEps >= 0 เสมอ) convertTo จึงแค่ saturate+ปัดเศษ ไม่ต้องใช้ convertScaleAbs
        scaled.convertTo(dst, src.type());
        if (ctx) {
          ctx.info = scale < 1
            ? `ปรับแสงด้วยพื้นหลังเบลอขนาด ${k}px (ประมาณพื้นหลังที่ภาพย่อ ${analysis.cols}x${analysis.rows}px แล้วขยายกลับ)`
            : `ปรับแสงด้วยพื้นหลังเบลอขนาด ${k}px`;
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (analysis) analysis.delete();
        if (bgSmall) bgSmall.delete();
        if (bg) bg.delete();
        if (srcF) srcF.delete();
        if (bgF) bgF.delete();
        if (epsMat) epsMat.delete();
        if (bgEps) bgEps.delete();
        if (ratio) ratio.delete();
        if (meanMat) meanMat.delete();
        if (scaled) scaled.delete();
      }
    },
  },

  quantize: {
    name: "Quantize / Posterize (ลดจำนวนระดับสี)",
    needs: "any",
    desc: "พิกเซลแต่ละจุดถูกเก็บเป็นตัวเลขที่มีจำนวนระดับจำกัด — 8 bit = 256 ระดับ ยิ่งเหลือระดับน้อยก็ยิ่งใช้ bit น้อยลง · เหลือ N ระดับ ภาพที่ไล่สีเนียน ๆ จะกลายเป็นแถบสีเป็นขั้น ๆ (เส้นขอบปลอม — เรียกว่า posterize) ภาพสีทำทีละช่องสี จึงได้ N×N×N สีที่เป็นไปได้ · ช่วง 0–255 ถูกแบ่งเป็น N ช่วงเท่า ๆ กัน เหมือน imquantize(A, thresholds) ของ MATLAB ที่ตั้งเส้นแบ่งห่างเท่ากัน (ถ้าอยากได้เส้นแบ่งที่เลือกจาก histogram ให้ใช้กล่อง Multi-Otsu) · แอปกระจายค่าผลลัพธ์กลับเต็ม 0–255 เพื่อให้เห็นชัด (imquantize ของ MATLAB คืนเลขลำดับ 1..N) · ไม่มี dithering (ไม่กระจาย error ของสี) · 2 ระดับ = ผลเดียวกับ Binary Threshold ที่ 127 · ดู histogram ก่อน/หลัง: หลัง Quantize จะเหลือเป็นแท่งแหลม N แท่ง",
    params: [
      { key: "levels", type: "number", label: "จำนวนระดับต่อช่อง", min: 2, max: 256, step: 1, default: 4, desc: "256 = ภาพเดิม (8 bit) · 2 = ขาว/ดำ" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const N = intIn(p.levels, 4, 2, 256);
      let x = null, lut = null, dst = null;
      try {
        x = dropAlpha(cv, src);
        lut = new cv.Mat(1, 256, cv.CV_8UC1);
        lut.data.set(quantizeLut(N));
        dst = new cv.Mat();
        cv.LUT(x || src, lut, dst);
        if (ctx) {
          const bits = quantizeBits(N);
          let info;
          if (N === 256) info = "256 ระดับ = 8 bit เท่าภาพเดิม — ภาพไม่เปลี่ยน";
          else if (src.channels() === 1) info = `เหลือ ${N} ระดับเทา (ใช้ ${bits} bit ต่อพิกเซล · ภาพเดิม 256 ระดับ = 8 bit) · ค่า 0–255 ถูกแบ่งเป็น ${N} ช่วงเท่ากัน แต่ละช่วงเหลือค่าเดียว`;
          else info = `เหลือ ${N} ระดับต่อช่องสี × 3 ช่อง = ${N * N * N} สีที่เป็นไปได้ (${bits} bit ต่อช่อง = ${3 * bits} bit ต่อพิกเซล · ภาพเดิม 24 bit)`;
          if (N !== 256 && (N & (N - 1)) !== 0) info += ` · ${N} ไม่ใช่กำลังของ 2: ${bits} bit เก็บได้ ${2 ** bits} ระดับ จึงใช้ไม่เต็ม`;
          if (N === 2) info += " · 2 ระดับ = ผลเดียวกับ Binary Threshold ที่ 127";
          ctx.info = info;
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (x) x.delete();
        if (lut) lut.delete();
      }
    },
  },

  unsharp: {
    name: "Sharpen (Unsharp Mask)",
    needs: "any",
    desc: "เพิ่มความคมด้วยหลักการ 'ลบภาพเบลอออกจากภาพจริง': ส่วนต่างคือรายละเอียด/ขอบ แล้วบวกกลับเข้าไปตามน้ำหนัก — ใหม่ = (1+a)·เดิม − a·เบลอ ยิ่ง a มาก ยิ่งคม (มากไปจะเกิดขอบซ้อน)",
    params: [
      { key: "amount", type: "number", label: "ความแรง", min: 0, max: 3, step: 0.1, default: 1, desc: "น้ำหนักที่บวกรายละเอียด/ขอบกลับเข้าไป — มาก = คมขึ้น (มากไปเกิดขอบซ้อน)" },
      { key: "radius", type: "number", label: "รัศมีเบลอ", min: 1, max: 15, step: 2, default: 5, hint: "เลขคี่", desc: "ขนาดเบลอที่ใช้แยกรายละเอียดออกจากภาพ — ใหญ่ = ดึงรายละเอียดหยาบขึ้นมาเน้น" },
    ],
    apply(cv, src, p) {
      const blur = new cv.Mat();
      const k = ensureOdd(p.radius);
      cv.GaussianBlur(src, blur, new cv.Size(k, k), 0);
      const dst = new cv.Mat();
      cv.addWeighted(src, 1 + p.amount, blur, -p.amount, 0, dst);
      blur.delete();
      return dst;
    },
  },

  laplacian_sharpen: {
    name: "Laplacian Sharpening",
    needs: "any",
    desc: "ใช้ Laplacian (อนุพันธ์อันดับสอง) หา 'จุดที่ความสว่างเปลี่ยนฉับพลัน' แล้วนำไปหักออกจากภาพเดิม ทำให้ขอบถูกเน้นขึ้น — เป็นวิธี sharpen แบบคลาสสิกในตำรา",
    params: [
      { key: "strength", type: "number", label: "ความแรง", min: 0, max: 3, step: 0.1, default: 1, desc: "น้ำหนักที่หักเส้นขอบ (Laplacian) ออกจากภาพเดิม — มาก = ขอบเด่นขึ้น" },
    ],
    apply(cv, src, p) {
      // g = f − strength·∇²f ตามตำรา — ต้องลบแบบมีเครื่องหมาย (เดิมใส่ค่าสัมบูรณ์ก่อนลบ ขอบเลยกลายเป็นร่องดำทั้งสองฝั่ง)
      // ksize 1 = kernel 4 ทิศ [0 1 0;1 −4 1;0 1 0] (= fspecial('laplacian',0) ของ MATLAB) · ksize 3 ของ OpenCV เป็นแบบแนวทแยง
      // ∇² ของภาพ 8 bit ด้วย kernel จำนวนเต็มเป็นจำนวนเต็มเสมอ (float32 เก็บได้เป๊ะ) แล้วคิด f − s·∇²f ด้วย double
      // ปัดครึ่งขึ้นเอง — ไม่ใช้ addWeighted/convertTo เพราะการปัดของ float32 ต่างกันได้ตาม build ของ OpenCV (แอปกับ Python ไม่ตรงกัน)
      const f = new cv.Mat();
      const lap = new cv.Mat();
      let dst = null;
      try {
        src.convertTo(f, cv.CV_32F);
        cv.Laplacian(f, lap, cv.CV_32F, 1);
        dst = new cv.Mat(src.rows, src.cols, src.type());
        const S = src.data, L = lap.data32F, D = dst.data, s = p.strength;
        for (let i = 0; i < D.length; i++) {
          const v = Math.floor(S[i] - s * L[i] + 0.5);
          D[i] = v < 0 ? 0 : v > 255 ? 255 : v;
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        f.delete();
        lap.delete();
      }
    },
  },

  custom_kernel: {
    name: "Custom Kernel (Convolution)",
    needs: "any",
    desc: "ทดลอง 'ตัวกรองภาพ' (kernel/mask) แบบต่าง ๆ ที่ใช้กันจริงในตำรา — แต่ละแบบคือตัวเลข 3x3 หรือ 5x5 ที่คูณ-บวกกับพิกเซลรอบข้างแล้วได้ผลต่างกันโดยสิ้นเชิง (คมขึ้น/หาขอบ/นูนสามมิติ/เบลอ) รอบนี้ยังพิมพ์ matrix เองไม่ได้ ใช้ preset สำเร็จรูปก่อน · Motion/Disk คือ PSF (point spread function = หน้าตาของจุดเดียวหลังเบลอ) ใช้สร้างภาพเบลอที่รู้สาเหตุแน่นอน เพื่อทดลองแก้เบลอ (deblurring) · ค่าใน kernel ได้จากการนับว่าแต่ละพิกเซลถูกเส้น/วงกลมทับกี่ส่วน จึงใกล้เคียง fspecial ของ MATLAB แต่ไม่เท่าทุกทศนิยม",
    params: [
      { key: "preset", type: "select", label: "แบบ Kernel", options: [
        { value: "identity", label: "Identity (ไม่เปลี่ยนภาพ)", hint: "kernel ที่ให้ผลเหมือนภาพเดิม — ใช้เทียบว่า 'ไม่ทำอะไร' หน้าตาเป็นอย่างไร" },
        { value: "sharpen", label: "Sharpen 3x3 (คมขึ้น)", hint: "เพิ่มความคมแบบคลาสสิก: กลางบวกแรง รอบข้างลบ" },
        { value: "edge", label: "Edge Detect 3x3 (หาขอบ)", hint: "กลางบวกเท่าผลรวมรอบข้าง (ลบทั้งหมด) — เหลือแต่บริเวณที่ความสว่างเปลี่ยน" },
        { value: "emboss", label: "Emboss (นูนสามมิติ)", hint: "เน้นขอบทางเฉียง ให้ความรู้สึกนูน-เว้าคล้ายแกะสลัก" },
        { value: "outline", label: "Outline (เส้นขอบหนา)", hint: "คล้าย Sharpen แต่ใช้เพื่อนบ้านครบ 8 ทิศ (แรงกว่า) — ผลรวม kernel = 1 จึงยังคงความสว่างเฉลี่ยไว้ ไม่ใช่ตัวหาขอบแท้ ๆ ที่ผลรวมเป็น 0 แบบ Edge Detect" },
        { value: "box3", label: "Box Blur 3x3 (เบลอเฉลี่ย)", hint: "เฉลี่ยพิกเซลรอบข้างเท่า ๆ กัน 9 จุด — เทียบกับ Blur node ที่มีอยู่แล้ว" },
        { value: "box5", label: "Box Blur 5x5 (เบลอเฉลี่ยกว้างขึ้น)", hint: "เฉลี่ยพิกเซลรอบข้าง 25 จุด — เบลอแรงกว่า box3" },
        { value: "motion", label: "Motion Blur (เบลอจากการขยับ)", hint: "จำลองภาพเบลอเพราะกล้องหรือวัตถุขยับเป็นเส้นตรง ยาว 'ขนาด PSF' px ที่ 'มุม PSF' — เทียบ fspecial('motion') ของ MATLAB" },
        { value: "disk", label: "Disk (เบลอจากโฟกัสหลุด)", hint: "เฉลี่ยในวงกลมรัศมี 'ขนาด PSF' px — เทียบ fspecial('disk')" },
      ], default: "sharpen" },
      { key: "amount", type: "number", label: "ความแรง (ผสมกับภาพเดิม)", min: 0, max: 2, step: 0.1, default: 1, desc: "0 = ภาพเดิมล้วน, 1 = ผลจาก kernel เต็มที่, มากกว่า 1 = เน้นผลต่างจากภาพเดิมให้แรงขึ้นอีก (เหมือน unsharp mask)" },
      { key: "psfSize", visibleWhen: [{key:"preset",in:["motion","disk"]}], type: "number", label: "ขนาด PSF (px) — ใช้กับ Motion/Disk เท่านั้น", min: 1, max: 50, step: 1, default: 9, desc: "Motion = ความยาวของรอยขยับ · Disk = รัศมีวงกลม" },
      { key: "psfAngle", visibleWhen: [{key:"preset",in:["motion"]}], type: "number", label: "มุม PSF (องศา) — ใช้กับ Motion เท่านั้น", min: 0, max: 179, step: 1, default: 0, desc: "0° = ขยับแนวนอน · 90° = แนวตั้ง · มุมเพิ่มทวนเข็มนาฬิกาแบบ MATLAB" },
    ],
    apply(cv, src, p, ctx) {
      // fspecial-style presets: ระบุ kernel + ค่า scale (ผลรวม, ใช้ normalize เพื่อไม่ให้ภาพสว่าง/มืดขึ้นเวลาเบลอ)
      const PRESETS = {
        identity: { k: [0, 0, 0, 0, 1, 0, 0, 0, 0], size: 3 },
        sharpen: { k: [0, -1, 0, -1, 5, -1, 0, -1, 0], size: 3 },
        edge: { k: [-1, -1, -1, -1, 8, -1, -1, -1, -1], size: 3 },
        emboss: { k: [-2, -1, 0, -1, 1, 1, 0, 1, 2], size: 3 },
        outline: { k: [-1, -1, -1, -1, 9, -1, -1, -1, -1], size: 3 },
        box3: { k: new Array(9).fill(1 / 9), size: 3 },
        box5: { k: new Array(25).fill(1 / 25), size: 5 },
      };
      // Motion/Disk = PSF จาก psfKernel (ตัวเดียวกับที่ batch 4B ใช้) · preset เดิมเลือกด้วย own-property (ค่าแปลกเช่น "constructor" → sharpen)
      const psfKind = p.preset === "motion" || p.preset === "disk" ? p.preset : null;
      if (psfKind && isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      let preset = null;
      let psfInfo = null;
      let kernel;
      if (psfKind) {
        const L = intIn(p.psfSize, 9, 1, 50);
        const ang = intIn(p.psfAngle, 0, 0, 179);
        const psf = psfKernel(psfKind, L, ang);
        kernel = cv.matFromArray(psf.n, psf.n, cv.CV_32FC1, Array.from(psf.k));
        psfInfo = psfKind === "motion" ? `Kernel: motion (${psf.n}x${psf.n}) · ยาว ${L} px มุม ${ang}°` : `Kernel: disk (${psf.n}x${psf.n}) · รัศมี ${L} px`;
      } else {
        preset = Object.hasOwn(PRESETS, p.preset) ? PRESETS[p.preset] : PRESETS.sharpen;
        kernel = cv.matFromArray(preset.size, preset.size, cv.CV_32FC1, preset.k);
      }
      let filtered = null;
      let dst = null;
      try {
        filtered = new cv.Mat();
        cv.filter2D(src, filtered, -1, kernel, new cv.Point(-1, -1), 0, cv.BORDER_DEFAULT);
        const amount = finiteOr(p.amount, 1);
        dst = new cv.Mat();
        if (amount === 1) {
          filtered.copyTo(dst);
        } else {
          // ผสมระหว่างภาพเดิมกับผลลัพธ์ตามน้ำหนัก amount (0=เดิมล้วน, 1=ผลเต็มที่, >1=เน้นผลต่างเพิ่ม)
          cv.addWeighted(src, 1 - amount, filtered, amount, 0, dst);
        }
        if (ctx) ctx.info = psfInfo || `Kernel: ${p.preset} (${preset.size}x${preset.size})`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        kernel.delete();
        if (filtered) filtered.delete();
      }
    },
  },

  // ---------------- 3. Restoration ----------------
  salt_pepper_noise: {
    name: "Salt & Pepper Noise",
    needs: "any",
    desc: "จำลองภาพเสียโดยสุ่มพิกเซลให้เป็นขาวล้วน (เกลือ) หรือดำล้วน (พริกไทย) — ใช้เป็นโจทย์ทดลอง: ใส่ noise แล้วต่อด้วย filter ฟื้นฟู (เช่น Median) เพื่อดูว่าตัวไหนกู้ภาพได้ดีที่สุด",
    params: [
      { key: "amount", type: "number", label: "ปริมาณ (%)", min: 0, max: 25, step: 1, default: 5, desc: "สัดส่วนพิกเซลที่ถูกสุ่มให้เป็นขาว/ดำสนิท — มาก = ภาพเสียหนักขึ้น" },
    ],
    apply(cv, src, p) {
      const dst = new cv.Mat();
      src.copyTo(dst);
      const total = dst.rows * dst.cols;
      const n = Math.floor(total * (p.amount / 100));
      const ch = dst.channels();
      for (let i = 0; i < n; i++) {
        const idx = (Math.random() * total) | 0;
        const v = Math.random() < 0.5 ? 0 : 255;
        for (let c = 0; c < ch; c++) dst.data[idx * ch + c] = v;
      }
      return dst;
    },
  },

  median_blur: {
    name: "Median Blur",
    needs: "any",
    desc: "แทนแต่ละจุดด้วยค่ากลาง ๆ ของเพื่อนบ้าน ตัวเก่งเรื่องลบจุดขาว-ดำที่กระเด็นเป็นเม็ด (noise เกลือ-พริกไทย) เพราะจุดขาวโพลนหรือดำสนิทเป็นค่าสุดโต่ง พอเรียงหาค่ากลางมันเลยโดนเขี่ยทิ้ง แถมขอบภาพยังอยู่ครบกว่าการเฉลี่ยธรรมดา",
    params: [
      { key: "ksize", type: "number", label: "ขนาด kernel", min: 3, max: 9, step: 2, default: 3, hint: "เลขคี่", desc: "ขนาดบริเวณที่ใช้หาค่ากลาง — ยิ่งมาก ยิ่งลบ noise ก้อนใหญ่ได้แต่รายละเอียดเล็กหายตาม" },
    ],
    apply(cv, src, p) {
      const dst = new cv.Mat();
      cv.medianBlur(src, dst, ensureOdd(p.ksize));
      return dst;
    },
  },

  gaussian_blur: {
    name: "Gaussian Blur",
    needs: "any",
    desc: "เกลี่ยภาพให้เนียนด้วยการเฉลี่ยกับจุดรอบ ๆ จุดที่อยู่ใกล้กลางนับน้ำหนักมากกว่าจุดไกล ลบเม็ด noise จาง ๆ ได้ดี แต่ก็แลกมากับขอบที่มัวลงหน่อย ยิ่งเพิ่มขนาด kernel ภาพก็ยิ่งฟุ้ง",
    params: [
      { key: "ksize", type: "number", label: "ขนาด kernel", min: 1, max: 31, step: 2, default: 5, hint: "เลขคี่", desc: "ขนาดบริเวณที่เกลี่ยเฉลี่ย — ยิ่งมาก ยิ่งเบลอ/เนียนขึ้น" },
    ],
    apply(cv, src, p) {
      const k = ensureOdd(p.ksize);
      const dst = new cv.Mat();
      cv.GaussianBlur(src, dst, new cv.Size(k, k), 0);
      return dst;
    },
  },

  box_blur: {
    name: "Average (Box) Blur",
    needs: "any",
    desc: "เฉลี่ยพิกเซลรอบข้างแบบน้ำหนักเท่ากันทุกตัว — ง่ายและเร็วที่สุด แต่ให้ผลหยาบกว่า Gaussian (เกิดลายบล็อก) เหมาะไว้เปรียบเทียบให้เห็นความต่างของ kernel",
    params: [
      { key: "ksize", type: "number", label: "ขนาด kernel", min: 1, max: 31, step: 2, default: 5, desc: "ขนาดบริเวณที่เฉลี่ย — ยิ่งมาก ยิ่งเบลอ/หยาบขึ้น" },
    ],
    apply(cv, src, p) {
      const k = Math.max(1, Math.round(p.ksize));
      const dst = new cv.Mat();
      cv.blur(src, dst, new cv.Size(k, k));
      return dst;
    },
  },

  bilateral: {
    name: "Bilateral Blur",
    needs: "any",
    desc: "เบลอแบบฉลาด: เฉลี่ยเฉพาะพิกเซลที่ 'อยู่ใกล้และสีคล้ายกัน' จึงลด noise ได้โดยขอบยังคม (ที่ขอบ สีต่างกันมาก จึงไม่ถูกเฉลี่ยข้าม) — ใช้ในโหมดถ่ายภาพผิวเนียน ข้อเสียคือช้ากว่าตัวอื่นมาก",
    params: [
      { key: "d", type: "number", label: "เส้นผ่านศูนย์กลาง", min: 3, max: 15, step: 2, default: 7, desc: "ขนาดบริเวณรอบแต่ละจุดที่พิจารณา — ใหญ่ = เบลอกว้างขึ้นแต่ช้าลงมาก" },
      { key: "sigmaColor", type: "number", label: "ความต่างสีที่ยอมรับ", min: 10, max: 200, step: 5, default: 75, desc: "สีต่างกันได้แค่ไหนถึงยังนับว่า 'คล้ายกัน' — มาก = เบลอข้ามขอบสีได้ง่ายขึ้น" },
      { key: "sigmaSpace", type: "number", label: "ระยะที่ยอมรับ", min: 10, max: 200, step: 5, default: 75, desc: "ระยะห่างที่ยอมให้มีผลต่อกัน — มาก = พิจารณาพิกเซลที่อยู่ไกลออกไปด้วย" },
    ],
    apply(cv, src, p) {
      const dst = new cv.Mat();
      cv.bilateralFilter(src, dst, p.d, p.sigmaColor, p.sigmaSpace, cv.BORDER_DEFAULT);
      return dst;
    },
  },

  // ---------------- 4. Morphology ----------------
  morphology: {
    name: "Morphological Operation",
    needs: "gray",
    desc: "จัดการ 'รูปทรง' ของวัตถุขาวในภาพขาว-ดำ ด้วยการเอาตัวแปรง (kernel) ไล่ทาบทั่วภาพ: Erode กัดวัตถุให้ผอมลง/ลบจุดจิ๋วรบกวน, Dilate พอกให้อ้วนขึ้น/อุดรูโหว่, Opening ลบจุดเล็กแต่คงขนาดเดิม, Closing อุดรูข้างในวัตถุ, Gradient ดึงเฉพาะเส้นขอบ ส่วน TopHat/BlackHat ดึงส่วนที่สว่าง/มืดกว่ารอบข้าง มักใช้ต่อหลัง threshold · Hit-or-Miss หาลายแม่แบบ 3×3 (บางช่องต้องขาว บางช่องต้องดำ) โดย erode ภาพขาวด้วยช่อง 'ต้องขาว' แล้ว AND กับ erode ภาพกลับด้านด้วยช่อง 'ต้องดำ' — นอกขอบภาพนับเป็น 'ไม่สน' (ตามเอกสาร bwhitmiss ของ MATLAB สร้างจาก imerode ซึ่งถือว่านอกภาพเป็น +∞ จึงน่าจะนับแบบเดียวกัน แต่ยังไม่ได้ทดสอบกับ MATLAB จริง)  · กรณีหาปลายเส้นทั่วไป Endpoints & Branchpoints สะดวกกว่า · Regional Maxima / H-maxima / Extended Maxima ใช้หา 'ยอด' ในภาพเทาหรือภาพ Distance Transform เพื่อทำ marker แยกวัตถุที่ติดกัน — noise ทำให้เกิดยอดเล็กจำนวนมาก เพิ่ม h หรือ blur ก่อน · นับแบบ 8-connected เหมือนค่าเริ่มต้นของ MATLAB",
    params: [
      { key: "op", type: "select", label: "Operation", options: [
        { value: "erode", label: "Erosion (กัดให้เล็กลง)", hint: "กัดขอบวัตถุขาวให้ผอมลง ลบจุดจิ๋วรบกวน" },
        { value: "dilate", label: "Dilation (ขยายให้ใหญ่ขึ้น)", hint: "พอกขอบวัตถุขาวให้อ้วนขึ้น อุดรูโหว่เล็ก ๆ" },
        { value: "open", label: "Opening (ลบจุดเล็ก)", hint: "Erode แล้ว Dilate — ลบจุดรบกวนเล็กโดยไม่เปลี่ยนขนาดวัตถุใหญ่" },
        { value: "close", label: "Closing (อุดรู)", hint: "Dilate แล้ว Erode — อุดรูเล็ก ๆ ในตัววัตถุ" },
        { value: "gradient", label: "Gradient (ขอบวัตถุ)", hint: "Dilate ลบ Erode — ดึงเฉพาะเส้นขอบของวัตถุ" },
        { value: "tophat", label: "Top Hat", hint: "ภาพเดิม ลบ Opening — ดึงส่วนที่สว่างกว่ารอบข้าง" },
        { value: "blackhat", label: "Black Hat", hint: "Closing ลบภาพเดิม — ดึงส่วนที่มืดกว่ารอบข้าง" },
        { value: "fill_holes", label: "Fill Holes (อุดรูในวัตถุ)", hint: "เติมรูปิดสนิทภายในวัตถุขาว โดยไม่กระทบขนาดรอบนอก — ภาพที่ไม่ใช่ขาว-ดำล้วนจะถูก binarize ที่ค่ากึ่งกลาง 127 ก่อนเสมอ" },
        { value: "clear_border", label: "Clear Border (ลบวัตถุที่แตะขอบภาพ)", hint: "ลบวัตถุขาวที่มีพิกเซลแม้แต่จุดเดียวติดขอบภาพทั้ง 4 ด้าน — เหลือแต่วัตถุที่อยู่เต็มตัวในเฟรม" },
        { value: "remove_small", label: "Remove Small Objects (ลบวัตถุเล็ก)", hint: "ลบวัตถุขาวที่มีพื้นที่น้อยกว่าเกณฑ์ทิ้ง — ใช้ค่า 'พื้นที่ขั้นต่ำ' ด้านล่างเป็นเกณฑ์" },
        { value: "skeleton", label: "Skeleton / Thinning (เส้นแกนกลาง)", hint: "ลดวัตถุขาวให้เหลือเส้นแกนกลางหนา 1 พิกเซล (Zhang-Suen) ที่ความละเอียดเต็ม — ช้ากว่า operation อื่นในกลุ่มนี้ (วัตถุยิ่งหนายิ่งช้า ภาพ 1080p ที่มีก้อนใหญ่ราว 0.3 วินาที เหมาะกับภาพนิ่งมากกว่ากล้องสด) ไม่ใช้ค่า 'รูปร่าง kernel'/'ขนาด kernel'/'จำนวนรอบ' ด้านล่างเลย — ถ้าเส้นแกนกลางไปวิ่งอยู่ในพื้นหลังแทนที่จะอยู่ในวัตถุ แปลว่าพื้นหลังเป็นสีขาว ให้สลับขาว-ดำ (Invert) มาก่อน เพราะ op นี้คิดว่าพิกเซลขาวคือ 'วัตถุ' เสมอ (นอกขอบภาพนับเป็นพื้นหลัง)" },
        { value: "reconstruct_open", label: "Opening by Reconstruction (คงรูปร่างเดิม)", hint: "เหมือน Opening แต่รักษาขอบวัตถุที่รอดไว้แม่นกว่า (ไม่มุมมน) — ใช้ ksize/จำนวนรอบ ด้านล่างเป็นขนาดกัดตอนสร้าง marker แต่ 'รูปร่าง kernel' ด้านล่างไม่มีผล (ใช้ Rect เสมอตอนกัด marker)" },
        { value: "convex_hull", label: "Convex Hull (ห่อด้วยรูปนูน)", hint: "ห่อแต่ละวัตถุด้วยรูปนูนที่เล็กที่สุดที่ครอบวัตถุไว้ได้พอดี (เหมือนรัดหนังยางรอบวัตถุ) แล้วเทียบพื้นที่เดิมกับพื้นที่รูปนูนเป็นค่า solidity (รวมทุกชิ้น ชิ้นใหญ่มีน้ำหนักมากกว่า) เส้นบาง/จุดเดี่ยวก็ถูกห่อและนับเป็นหนึ่งชิ้นด้วย — ใช้หารอยบิ่น/รอยเว้าของวัตถุ (solidity ยิ่งต่ำ ยิ่งเว้าแหว่งมาก) ไม่ใช้ค่า 'รูปร่าง kernel'/'ขนาด kernel'/'จำนวนรอบ' ด้านล่างเลย" },
        { value: "endpoints_branchpoints", label: "Endpoints & Branchpoints (จุดปลาย/จุดแยก)", hint: "⚠️ ใช้ต่อจาก Skeleton เสมอ — ถ้าวัตถุยังหนาอยู่ (ไม่ใช่เส้นบาง 1 พิกเซล) การนับจะไม่มีความหมาย นับเพื่อนบ้านสีขาว 8 ทิศรอบแต่ละพิกเซลเส้น: มีเพื่อนบ้าน 1 จุด = จุดปลาย (วาดวงเขียว), ส่วนจุดแยก (วาดวงแดง) ใช้ Crossing Number (นับจำนวนช่วงที่ขาว-ดำสลับกันรอบวงแหวนเพื่อนบ้าน หาร 2) ≥3 — ไม่ใช่แค่นับจำนวนเพื่อนบ้านตรง ๆ เพราะพิกเซลข้างจุดตัดจริงก็มีเพื่อนบ้าน ≥3 ได้เหมือนกันแต่ไม่ใช่จุดแยกจริง เป็นค่าประมาณ ไม่ใช่ lookup table แบบที่ MATLAB ใช้จริง ใช้นับจำนวนกิ่ง/ปลายของรอยร้าวหรือเส้นลายเส้นได้ ไม่ใช้ค่า 'รูปร่าง kernel'/'ขนาด kernel'/'จำนวนรอบ' ด้านล่างเลย" },
        { value: "hit_or_miss", label: "Hit-or-Miss (หาลายแม่แบบ 3×3)", hint: "หาตำแหน่งที่ลายรอบพิกเซลตรงแม่แบบ: บางช่องต้องขาว บางช่องต้องดำ บางช่องไม่สน — ภาพที่ไม่ใช่ขาว-ดำล้วนจะถูกตัดที่ 127 ก่อน · ไม่ใช้ค่า รูปร่าง/ขนาด kernel/จำนวนรอบ ด้านล่าง" },
        { value: "regional_max", label: "Regional Maxima (ยอดเฉพาะที่)", hint: "บริเวณที่สว่างกว่าทุกพิกเซลรอบตัว (ใช้ภาพเทา ไม่ตัด 127) — ผลเป็นภาพขาว-ดำ · ไม่ใช้ค่า รูปร่าง/ขนาด kernel/จำนวนรอบ ด้านล่าง" },
        { value: "regional_min", label: "Regional Minima (หุบเฉพาะที่)", hint: "บริเวณที่มืดกว่าทุกพิกเซลรอบตัว (ใช้ภาพเทา ไม่ตัด 127) — ผลเป็นภาพขาว-ดำ" },
        { value: "h_maxima", label: "H-maxima transform (ตัดยอดที่เตี้ยกว่า h)", hint: "ผลเป็นภาพเทา: ยอดทุกยอดถูกกดลง h ระดับ ยอดที่สูงไม่ถึง h หายไป — ใช้ค่า 'ความสูง h' ด้านล่าง" },
        { value: "extended_max", label: "Extended Maxima (ยอดที่สูงกว่ารอบข้างอย่างน้อย h)", hint: "= Regional Maxima ของ H-maxima · ใช้ทำ marker ให้ Watershed — ใช้ค่า 'ความสูง h' ด้านล่าง" },
        { value: "clean", label: "Clean (ลบจุดเดี่ยว)", hint: "ลบจุดขาวที่ไม่มีเพื่อนบ้านขาว 8 ทิศ ทำหนึ่งรอบ" },
        { value: "majority", label: "Majority (เสียงข้างมาก)", hint: "ขาวเมื่อในหน้าต่าง 3×3 มีขาวอย่างน้อย 5 จุด" },
        { value: "spur", label: "Spur (ตัดกิ่งเส้น)", hint: "ลบจุดปลายที่มีเพื่อนบ้านขาวพอดีหนึ่งจุด" },
        { value: "thin", label: "Thin (ทำให้บาง)", hint: "ทำให้บางด้วย Lam–Lee–Suen · 0 = จนนิ่ง" },
        { value: "shrink", label: "Shrink (หดเป็นจุด)", hint: "หดวัตถุโดยรักษาจำนวนวัตถุและรู · 0 = จนนิ่ง" },
        { value: "h_minima", label: "H-minima (ถมหุบตื้นกว่า h)", hint: "ถมหุบตื้นในภาพเทา ไม่ตัดที่ 127" },
        { value: "extended_min", label: "Extended Min (หุบลึก ≥ h)", hint: "หาตำแหน่งหุบหลังถมหุบตื้นด้วย h" },
      ], default: "erode" },
      { key: "shape", visibleWhen: [{key:"op",in:["erode","dilate","open","close","gradient","tophat","blackhat"]}], type: "select", label: "รูปร่าง kernel (ไม่มีผลกับ Skeleton และ Opening by Reconstruction)", options: [
        { value: "rect", label: "สี่เหลี่ยม (Rect)" },
        { value: "ellipse", label: "วงรี (Ellipse — คล้าย disk แต่อ้วนกว่า strel('disk') ของ MATLAB เล็กน้อย)" },
        { value: "cross", label: "กากบาท (Cross)" },
        { value: "diamond", label: "ข้าวหลามตัด (Diamond)", hint: "ระยะ Manhattan (|Δx|+|Δy|) ≤ รัศมี — มุมแหลมกว่า Ellipse" },
        { value: "octagon", label: "แปดเหลี่ยม (Octagon)", hint: "สี่เหลี่ยมตัดมุม — ใกล้เคียงวงกลมกว่า Diamond แต่ไม่กลมเท่า Ellipse" },
        { value: "line", label: "เส้นตรง (Line, ปรับมุมได้)", hint: "kernel มิติเดียว ใช้ปรับ Operation ตามแนวเส้นที่ต้องการ (เช่นลบ/เน้นเส้นแนวนอน-แนวตั้ง) — ที่มุมเฉียง (เช่น 45°) จำนวนพิกเซลจริงในเส้นอาจน้อยกว่าขนาดที่ตั้งไว้ เพราะต้องปัดพิกัดให้ลงตารางพิกเซล (เกิดกับ strel('line') ของ MATLAB เช่นกัน ไม่ใช่บั๊ก)" },
      ], default: "rect" },
      { key: "ksize", visibleWhen: [{key:"op",in:["erode","dilate","open","close","gradient","tophat","blackhat","reconstruct_open"]}], type: "number", label: "ขนาด kernel", min: 1, max: 21, step: 2, default: 5, desc: "ขนาดตัวแปรงที่ไล่ทาบภาพ — ใหญ่ = ผลกัด/พอกแรงขึ้น" },
      { key: "iterations", visibleWhen: [{key:"op",in:["erode","dilate","open","close","gradient","tophat","blackhat","reconstruct_open","majority","spur","thin","shrink"]}], type: "number", label: "จำนวนรอบ", min: 1, max: 5, step: 1, default: 1, desc: "ทำ operation ซ้ำกี่รอบ — มาก = ผลยิ่งเข้มขึ้น (สำหรับ Opening by Reconstruction คือจำนวนรอบกัดตอนสร้าง marker)" },
      { key: "minArea", visibleWhen: [{key:"op",in:["remove_small"]}], type: "number", label: "พื้นที่ขั้นต่ำ (px) — ใช้กับ Remove Small Objects เท่านั้น", min: 10, max: 5000, step: 10, default: 150, desc: "วัตถุที่มีพื้นที่น้อยกว่านี้ (พิกเซล) จะถูกลบทิ้ง — พารามิเตอร์นี้ไม่มีผลกับ Operation อื่นนอกจาก Remove Small Objects" },
      { key: "lineAngle", visibleWhen: [{key:"op",in:["erode","dilate","open","close","gradient","tophat","blackhat"]},{key:"shape",in:["line"]}], type: "number", label: "มุมของเส้น (องศา) — ใช้กับรูปร่าง Line เท่านั้น", min: 0, max: 179, step: 1, default: 0, desc: "0° = แนวนอน, 90° = แนวตั้ง — ไม่มีผลถ้าเลือกรูปร่างอื่น" },
      { key: "hmPattern", visibleWhen: [{key:"op",in:["hit_or_miss"]}], type: "select", label: "แม่แบบ Hit-or-Miss — ใช้กับ Hit-or-Miss เท่านั้น", options: Object.entries(HITMISS_PATTERNS).map(([value, v]) => ({ value, label: v.label })), default: "isolated" },
      { key: "hmRotate", visibleWhen: [{key:"op",in:["hit_or_miss"]}], type: "boolean", label: "รวมทั้ง 4 ทิศ (หมุนทีละ 90°) — ใช้กับ Hit-or-Miss เท่านั้น", default: true },
      { key: "hHeight", visibleWhen: [{key:"op",in:["h_maxima","extended_max","h_minima","extended_min"]}], type: "number", label: "ความสูง/ความลึก h — ใช้กับ H-maxima / Extended Maxima / H-minima / Extended Minima เท่านั้น", min: 1, max: 255, step: 1, default: 10, desc: "ยอดที่สูงกว่ารอบข้างไม่ถึง h ระดับ (0–255) จะถูกตัดทิ้ง — เพิ่ม h เมื่อภาพมี noise มาก" },
    ],
    apply(cv, src, p, ctx) {
      // IP batch 4B — ตัวเลือกใหม่ (ไม่ใช้ shape/ksize/iterations) · path เดิมทุกตัวด้านล่างไม่เปลี่ยน
      if (p.op === "hit_or_miss") return hitMissOp(cv, src, p, ctx);
      if (p.op === "regional_max" || p.op === "regional_min" || p.op === "h_maxima" || p.op === "extended_max") return regionalOp(cv, src, p, ctx);
      if (Object.hasOwn(MORPH_5A, p.op)) return morphology5A(cv, src, p, ctx);
      if (p.op === "fill_holes") {
        // imfill semantics: เติมเฉพาะ "รู" (บริเวณ 0) ที่ปิดสนิท คือไม่ต่อกับขอบภาพเลยแม้แต่จุดเดียว
        // แนวทาง: (1) binarize ที่ 127 กันภาพเกรย์ที่ไม่ใช่ 0/255 ล้วน (2) เติมขอบ 0 หนา 1px รอบภาพ
        // เพื่อให้พื้นหลังทุกจุดที่ชนขอบภาพเดิม (ต่อให้วัตถุชนขอบ) กลายเป็นก้อนเดียวที่ต่อกับ (0,0)
        // (3) flood-fill จาก (0,0) ด้วย 255 บนสำเนา padded → พื้นหลังที่ต่อขอบกลายเป็น 255 ทั้งหมด
        // เหลือเฉพาะรูที่ปิดสนิทเป็น 0 (4) holes = NOT(filled) แล้ว crop กลับขนาดเดิม (5) OR กลับเข้า bin
        let bin = null;
        let padded = null;
        let filled = null;
        let ffMask = null;
        let filledRoi = null;
        let holes = null;
        let dst = null;
        try {
          bin = new cv.Mat();
          cv.threshold(src, bin, 127, 255, cv.THRESH_BINARY);
          padded = new cv.Mat();
          cv.copyMakeBorder(bin, padded, 1, 1, 1, 1, cv.BORDER_CONSTANT, new cv.Scalar(0));
          filled = new cv.Mat();
          padded.copyTo(filled);
          // floodFill mask ต้องมีขนาด H+2 x W+2 ของภาพที่ flood (คือ padded) จึงเป็น padded+2, single-channel 8U
          ffMask = new cv.Mat(padded.rows + 2, padded.cols + 2, cv.CV_8UC1, new cv.Scalar(0));
          cv.floodFill(filled, ffMask, new cv.Point(0, 0), new cv.Scalar(255));
          // crop กลับขนาดเดิมก่อน invert — เลี่ยงพึ่งพา mask ของ floodFill (ค่า 1 vs 255) ไปเลย
          filledRoi = filled.roi(new cv.Rect(1, 1, src.cols, src.rows));
          holes = new cv.Mat();
          cv.bitwise_not(filledRoi, holes); // 255 เฉพาะรูที่ปิดสนิท (พื้นหลังที่ต่อขอบ/วัตถุเดิม = 0)
          dst = new cv.Mat();
          cv.bitwise_or(bin, holes, dst);
          if (ctx) ctx.info = `เติมรูที่ปิดสนิทในวัตถุ (${cv.countNonZero(holes)} px ที่ถูกเติม)`;
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          if (bin) bin.delete();
          if (padded) padded.delete();
          if (filled) filled.delete();
          if (ffMask) ffMask.delete();
          if (filledRoi) filledRoi.delete();
          if (holes) holes.delete();
        }
      }
      if (p.op === "clear_border") {
        // imclearborder: ลบวัตถุ (label) ที่ bounding box แตะขอบภาพด้านใดด้านหนึ่ง — bbox แตะขอบ
        // ก็ต่อเมื่อมีพิกเซลจริงของวัตถุนั้นอยู่ที่แถว/คอลัมน์ขอบภาพ (เพราะ bbox คือกรอบที่แนบสนิทที่สุด)
        const bin = new cv.Mat();
        const labels = new cv.Mat();
        const stats = new cv.Mat();
        const centroids = new cv.Mat();
        let dst = null;
        try {
          cv.threshold(src, bin, 127, 255, cv.THRESH_BINARY);
          const n = cv.connectedComponentsWithStats(bin, labels, stats, centroids, 8, cv.CV_32S);
          const cols = bin.cols, rows = bin.rows;
          const removeLabel = new Uint8Array(n); // removeLabel[0] (background) stays 0 — never "removed"
          let removedCount = 0;
          for (let i = 1; i < n; i++) {
            const left = stats.intAt(i, cv.CC_STAT_LEFT);
            const top = stats.intAt(i, cv.CC_STAT_TOP);
            const w = stats.intAt(i, cv.CC_STAT_WIDTH);
            const h = stats.intAt(i, cv.CC_STAT_HEIGHT);
            if (left === 0 || top === 0 || left + w === cols || top + h === rows) {
              removeLabel[i] = 1;
              removedCount++;
            }
          }
          dst = new cv.Mat(rows, cols, cv.CV_8UC1);
          const labelData = labels.data32S;
          const outData = dst.data;
          for (let i = 0; i < labelData.length; i++) {
            const lbl = labelData[i];
            outData[i] = (lbl !== 0 && !removeLabel[lbl]) ? 255 : 0;
          }
          if (ctx) ctx.info = `ลบวัตถุที่แตะขอบภาพ ${removedCount} ชิ้น (เหลือ ${n - 1 - removedCount} ชิ้น)`;
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          bin.delete();
          labels.delete();
          stats.delete();
          centroids.delete();
        }
      }
      if (p.op === "remove_small") {
        const bin = new cv.Mat();
        const labels = new cv.Mat();
        const stats = new cv.Mat();
        const centroids = new cv.Mat();
        let dst = null;
        try {
          cv.threshold(src, bin, 127, 255, cv.THRESH_BINARY);
          const n = cv.connectedComponentsWithStats(bin, labels, stats, centroids, 8, cv.CV_32S);
          const minArea = finiteOr(p.minArea, 150);
          const removeLabel = new Uint8Array(n);
          let removedCount = 0;
          for (let i = 1; i < n; i++) {
            const area = stats.intAt(i, cv.CC_STAT_AREA);
            if (area < minArea) {
              removeLabel[i] = 1;
              removedCount++;
            }
          }
          dst = new cv.Mat(bin.rows, bin.cols, cv.CV_8UC1);
          const labelData = labels.data32S;
          const outData = dst.data;
          for (let i = 0; i < labelData.length; i++) {
            const lbl = labelData[i];
            outData[i] = (lbl !== 0 && !removeLabel[lbl]) ? 255 : 0;
          }
          if (ctx) ctx.info = `ลบวัตถุเล็ก ${removedCount} ชิ้น (เหลือ ${n - 1 - removedCount} ชิ้น)`;
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          bin.delete();
          labels.delete();
          stats.delete();
          centroids.delete();
        }
      }
      if (p.op === "skeleton") {
        // Zhang-Suen thinning — ไม่มี cv.ximgproc.thinning ใน build นี้ (ยืนยันแล้ว) จึงเขียนเอง
        // ทำที่ความละเอียดเต็มเสมอ (เดิมย่อ ≤240px แล้วขยายกลับ → เส้นหนาหลาย px ไม่ใช่ 1 px) และเติมขอบดำ 1 px รอบภาพ
        // ให้พิกเซลแถวขอบภาพถูกบางลงได้ด้วย (นอกภาพ = พื้นหลัง เหมือน bwmorph ของ MATLAB)
        // เร็วพอเพราะแต่ละรอบตรวจเฉพาะ "พิกเซลที่ยังอาจถูกลบได้": จุดที่ตรวจแล้วไม่ผ่านทั้ง step 1 และ 2
        // โดยเพื่อนบ้านไม่เปลี่ยน จะไม่มีวันผ่าน จึงตัดออกจนกว่าเพื่อนบ้านจะถูกลบ — ผลเท่ากับไล่ตรวจทั้งภาพทุกรอบ
        let bin = null, dst = null;
        try {
          bin = new cv.Mat();
          cv.threshold(src, bin, 127, 255, cv.THRESH_BINARY);
          const w = bin.cols, h = bin.rows;
          const W = w + 2;
          const img = new Uint8Array(W * (h + 2));
          const fail = new Uint8Array(img.length); // bit 1 = ตรวจที่ step 1 แล้วไม่ผ่าน, bit 2 = step 2
          const inList = new Uint8Array(img.length);
          let active = [];
          const b8 = bin.data;
          for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
              if (!b8[y * w + x]) continue;
              const i = (y + 1) * W + x + 1;
              img[i] = 1;
              inList[i] = 1;
              active.push(i);
            }
          }
          // ลำดับเพื่อนบ้าน p2..p9 ตามเข็มนาฬิกาเริ่มจากบน
          const OFF = [-W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1];
          let iterations = 0;
          for (;;) {
            iterations++;
            let changed = false;
            for (const step of [1, 2]) {
              const toDelete = [];
              for (const i of active) {
                if (!img[i]) continue;
                const p2 = img[i - W], p3 = img[i - W + 1], p4 = img[i + 1], p5 = img[i + W + 1];
                const p6 = img[i + W], p7 = img[i + W - 1], p8 = img[i - 1], p9 = img[i - W - 1];
                const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                const A = (!p2 && p3) + (!p3 && p4) + (!p4 && p5) + (!p5 && p6)
                  + (!p6 && p7) + (!p7 && p8) + (!p8 && p9) + (!p9 && p2);
                let ok = B >= 2 && B <= 6 && A === 1;
                if (ok) ok = step === 1 ? (p2 * p4 * p6 === 0 && p4 * p6 * p8 === 0) : (p2 * p4 * p8 === 0 && p2 * p6 * p8 === 0);
                if (ok) toDelete.push(i);
                else fail[i] |= step;
              }
              if (toDelete.length === 0) continue;
              changed = true;
              for (const i of toDelete) img[i] = 0;
              for (const i of toDelete) {
                for (const o of OFF) {
                  const j = i + o;
                  if (!img[j]) continue;
                  fail[j] = 0; // เพื่อนบ้านเปลี่ยน → ต้องตรวจใหม่ทั้งสอง step
                  if (!inList[j]) { inList[j] = 1; active.push(j); }
                }
              }
              const next = [];
              for (const i of active) {
                if (img[i] && fail[i] !== 3) next.push(i);
                else inList[i] = 0;
              }
              active = next;
            }
            if (!changed) break;
          }
          dst = new cv.Mat(h, w, cv.CV_8UC1);
          const out = dst.data;
          for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) out[y * w + x] = img[(y + 1) * W + x + 1] ? 255 : 0;
          }
          if (ctx) ctx.info = `Skeleton (thinning) — ${iterations} รอบ ที่ ${w}x${h}px`;
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          if (bin) bin.delete();
        }
      }
      if (p.op === "reconstruct_open") {
        // Opening by Reconstruction: marker = erosion ของภาพเดียวกัน N รอบ, mask = ภาพเดิม แล้วทำ
        // geodesic dilation (dilate แล้ว clip ไม่ให้เกิน mask) ซ้ำจนลู่เข้า — คงขอบวัตถุที่รอดไว้แม่นกว่า
        // Opening ธรรมดา (ไม่ทำให้มุมมน) เพราะ mask บังคับรูปร่างเดิมไว้ตลอด ไม่ใช่แค่ erode+dilate ตรง ๆ
        //
        // คำนวณผลลัพธ์ที่ "ลู่เข้าจริง" (converged) แบบ O(N) ตรง ๆ แทนการวน dilate+min ทีละรอบจนกว่าจะ
        // นิ่ง (ของเดิมจำกัดไว้ 300 รอบ ซึ่งไม่พอสำหรับโครงสร้างเรียวยาว เช่น รอยร้าว ทำให้ผลไม่ลู่เข้าจริง
        // ทั้งที่ ctx.info เคยบอกว่า "ลู่เข้า") — พิสูจน์ทางคณิตศาสตร์: marker เป็น subset ของ bin เสมอ
        // (erode ของ bin ด้วย kernel ที่จุด anchor อยู่ในตัวมันเอง ย่อมได้ผลอยู่ใน bin) และ geodesic dilation
        // ด้วย kernel 3x3 rect ขยาย marker ออกไปทีละ 1px รอบ ๆ วนไปเรื่อย ๆ โดยถูก clip ไว้ไม่ให้ล้น mask
        // เสมอ — จึงไม่มีทางกระโดดข้าม 8-connected component ไปยังอีกก้อนได้ ผลลัพธ์ที่ลู่เข้าจึงเท่ากับ
        // "เก็บทุก component ของ bin (นับแบบ 8-connected) ที่มี marker pixel อยู่ในนั้นอย่างน้อย 1 จุด"
        // พอดี ไม่มากไม่น้อยกว่านั้น — พิสูจน์เทียบกับ loop แบบเดิมที่ไม่จำกัดรอบ (unbounded cap) แล้วว่า
        // ตรงกันทุกพิกเซลบนภาพทดสอบหลายแบบ รวมภาพรอยร้าวจริงที่ยาวเกิน 300px (ดู work log ของ coder)
        const kSize = Math.max(1, Math.round(p.ksize) || 5);
        const erosionIters = Math.max(1, Math.round(p.iterations) || 1);
        const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(kSize, kSize));
        let bin = null, marker = null, labels = null, dst = null;
        try {
          bin = new cv.Mat();
          cv.threshold(src, bin, 127, 255, cv.THRESH_BINARY);
          marker = new cv.Mat();
          cv.erode(bin, marker, kernel, new cv.Point(-1, -1), erosionIters);
          labels = new cv.Mat();
          const n = cv.connectedComponents(bin, labels, 8, cv.CV_32S);
          const labelData = labels.data32S;
          const markerData = marker.data;
          const keepLabel = new Uint8Array(n); // keepLabel[0] (พื้นหลัง) เป็น 0 เสมอ ไม่ถูกเก็บ
          for (let i = 0; i < labelData.length; i++) {
            const lbl = labelData[i];
            if (lbl !== 0 && markerData[i] !== 0) keepLabel[lbl] = 1;
          }
          let keptCount = 0;
          for (let i = 1; i < n; i++) if (keepLabel[i]) keptCount++;
          dst = new cv.Mat(bin.rows, bin.cols, cv.CV_8UC1);
          const outData = dst.data;
          for (let i = 0; i < labelData.length; i++) {
            const lbl = labelData[i];
            outData[i] = (lbl !== 0 && keepLabel[lbl]) ? 255 : 0;
          }
          if (ctx) ctx.info = `Opening by Reconstruction — เก็บ ${keptCount}/${Math.max(0, n - 1)} วัตถุ (คำนวณผลลู่เข้าจริงแบบ O(N) ผ่าน connected components)`;
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          kernel.delete();
          if (bin) bin.delete();
          if (marker) marker.delete();
          if (labels) labels.delete();
        }
      }
      if (p.op === "convex_hull") {
        // bwconvhull: binarize เหมือน fill_holes/clear_border เสมอ → หาแต่ละวัตถุด้วย findContours
        // (RETR_EXTERNAL — สนใจแค่รูปทรงภายนอก ไม่ใช่รูในวัตถุ) → ห่อแต่ละ contour ด้วย convexHull →
        // วาดรูปนูนที่ได้ลงบน mask ผลลัพธ์ที่เริ่มจาก 0 ทั้งภาพ แล้วรายงาน solidity รวม = Σพื้นที่เดิม/Σพื้นที่รูปนูน (ถ่วงตามพื้นที่ ไม่ใช่ค่าเฉลี่ยของ solidity รายชิ้น)
        const bin = new cv.Mat();
        const contours = new cv.MatVector();
        const hier = new cv.Mat();
        let dst = null;
        try {
          cv.threshold(src, bin, 127, 255, cv.THRESH_BINARY);
          cv.findContours(bin, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
          dst = new cv.Mat.zeros(bin.rows, bin.cols, cv.CV_8UC1);
          let totalForegroundArea = 0;
          let totalHullArea = 0;
          let shapeCount = 0;
          for (let i = 0; i < contours.size(); i++) {
            const contour = contours.get(i);
            // ห่อ/วาด/นับทุก contour ไม่ข้ามชิ้นไหนเลย — ห้ามข้ามตามจำนวน "จุด" ของ contour เพราะ
            // CHAIN_APPROX_SIMPLE บีบเส้นตรงบาง 1 พิกเซลทุกความยาวให้เหลือแค่ 2 จุด (ปลายสองข้าง) ถ้าข้าม <3 จุด
            // เส้นบาง/จุดเดี่ยวจะหายจากผลลัพธ์และจากจำนวนที่นับ (bwconvhull('objects') เก็บไว้) — ยืนยันแล้วทั้ง
            // opencv.js และ cv2: convexHull กับ contour 1-2 จุดไม่ throw และ drawContours(-1) ของ hull 1-2 จุด
            // วาดพิกเซลเดิมกลับมาเป๊ะ; contourArea ของมันเป็น 0 ทั้งคู่ (solidity รวมจึงไม่เพี้ยน)
            const hull = new cv.Mat();
            try {
              cv.convexHull(contour, hull, false, true);
              const hullVec = new cv.MatVector();
              // push_back ต้องอยู่ใน try นี้ด้วย (ไม่ใช่ก่อนเปิด try) — ถ้า push_back throw (หายากแต่เป็น
              // ไปได้ เช่น out of memory) hullVec ที่เพิ่งสร้างต้องยังถูก delete ผ่าน finally ด้านล่าง
              // (debugger จับได้: รอบก่อนวาง push_back ไว้ก่อนเปิด try → hullVec หลุดการ cleanup ถ้า throw)
              try {
                hullVec.push_back(hull);
                cv.drawContours(dst, hullVec, 0, new cv.Scalar(255), -1);
              } finally {
                hullVec.delete();
              }
              totalForegroundArea += cv.contourArea(contour);
              totalHullArea += cv.contourArea(hull);
              shapeCount++;
            } finally {
              hull.delete();
              contour.delete();
            }
          }
          if (shapeCount === 0) {
            if (ctx) ctx.info = "ไม่พบวัตถุ (ลองทำ Threshold ให้เป็นขาว-ดำมาก่อน)";
          } else {
            const solidity = totalHullArea > 0 ? totalForegroundArea / totalHullArea : 1;
            if (ctx) ctx.info = `ห่อวัตถุ ${shapeCount} ชิ้นด้วยรูปนูน · solidity รวม (ถ่วงตามพื้นที่) ${solidity.toFixed(2)} (1.0 = นูนเต็มที่อยู่แล้ว)`;
          }
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          bin.delete();
          contours.delete();
          hier.delete();
        }
      }
      if (p.op === "endpoints_branchpoints") {
        // bwmorph('endpoints')/bwmorph('branchpoints') แบบประมาณ: binarize แล้วนับเพื่อนบ้านขาว 8 ทิศ
        // รอบแต่ละพิกเซลขาว (แนวทางเดียวกับ helper P(x,y,dx,dy) ของ skeleton ด้านบน — คืน 0 นอกขอบภาพ)
        // เพื่อนบ้าน 1 จุด = จุดปลาย (endpoint, วาดวงเขียว) · จุดแยก (branchpoint, วาดวงแดง) = Crossing Number
        // >= 3 (ดูคอมเมนต์ที่จุดคำนวณ CN ด้านล่าง) — *** นี่คือค่าประมาณ ไม่ใช่ lookup table ที่ MATLAB bwmorph
        // ใช้จริง *** ต้องใช้ต่อจาก Skeleton เสมอ (ดู hint ของตัวเลือกนี้) ไม่งั้นวัตถุหนาจะทำให้พิกเซลภายในหลาย
        // จุดนับเป็นจุดแยกจนไม่มีความหมาย — เวลา (qa วัดหลังเปลี่ยนเป็น CN): เส้นบางปกติ ~7-10 ms ที่ทุก
        // ความละเอียด แต่ภาพขาวล้วน 1920x1080 ~156 ms และกระดานหมากรุก 1 พิกเซลที่ 1920x1080 ~1064 ms
        // (กรณีเลวร้ายที่ผู้ใช้ไม่ควรเจอถ้าทำ Skeleton ก่อน) จึงไม่ย่อภาพก่อนวิเคราะห์เหมือน skeleton/entropy filter
        //
        // guard ภาพว่างบังคับ (ต่างจาก convex_hull/contrast_stretch ที่ guard แค่กันไว้เฉย ๆ): ยืนยันแล้วว่า
        // cv.cvtColor(..., COLOR_GRAY2RGB) ที่ใช้ด้านล่าง throw C++ exception จริงบนภาพ 0x0 (คนละเคสกับ
        // cv.threshold/cv.findContours ที่ op อื่นในกลุ่มนี้ใช้ ซึ่งไม่ throw กับภาพว่าง)
        if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
        const bin = new cv.Mat();
        let dst = null;
        try {
          cv.threshold(src, bin, 127, 255, cv.THRESH_BINARY);
          const w = bin.cols;
          const h = bin.rows;
          const data = bin.data; // cache getter ก่อนเข้าลูปเสมอ (กฎเหล็กของไฟล์นี้)
          dst = new cv.Mat();
          cv.cvtColor(bin, dst, cv.COLOR_GRAY2RGB);
          if (cv.countNonZero(bin) === 0) {
            if (ctx) ctx.info = "ไม่พบเส้น (ลองทำ Skeleton ก่อน)";
            return dst;
          }
          const green = new cv.Scalar(0, 255, 0);
          const red = new cv.Scalar(255, 0, 0);
          let endpoints = 0;
          let branchpoints = 0;
          // ลำดับเพื่อนบ้าน 8 ทิศตามเข็มนาฬิกา (N,NE,E,SE,S,SW,W,NW) คงที่เสมอ — ใช้คำนวณ Crossing Number (CN)
          // ของจุดแยก (ดูคอมเมนต์อธิบายเต็ม ๆ ที่จุดคำนวณ CN ด้านล่าง)
          const RING_OFFSETS = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
          const ring = new Uint8Array(8);
          for (let y = 0; y < h; y++) {
            const row = y * w;
            for (let x = 0; x < w; x++) {
              if (data[row + x] === 0) continue;
              let n = 0;
              for (let ri = 0; ri < 8; ri++) {
                const xx = x + RING_OFFSETS[ri][0];
                const yy = y + RING_OFFSETS[ri][1];
                const v = (xx < 0 || xx >= w || yy < 0 || yy >= h) ? 0 : (data[yy * w + xx] !== 0 ? 1 : 0);
                ring[ri] = v;
                n += v;
              }
              if (n === 1) {
                endpoints++; cv.circle(dst, new cv.Point(x, y), 3, green, -1);
              } else {
                // Crossing Number (CN): นับจำนวนช่วงที่ค่าสลับขาว<->ดำรอบวงแหวนเพื่อนบ้าน 8 ทิศ (รวมช่วงจาก
                // จุดสุดท้ายวนกลับไปจุดแรกด้วย) แล้วหาร 2 — มาตรฐานที่ถูกต้องกว่าการนับจำนวนเพื่อนบ้านตรง ๆ
                // (ของเดิม n>=3): พิกเซลที่อยู่ "ติดกับ" จุดตัดจริง (ไม่ใช่ตัวจุดตัดเอง) ก็มี n>=3 ได้เหมือนกัน
                // เพราะมีเพื่อนบ้านทแยงไปติดกับอีกเส้นที่ตัดผ่าน ทำให้นับจุดแยกเกินจริงรอบจุดตัดเดียว (qa
                // พิสูจน์ด้วยภาพกากบาท 21x21px: นับเพื่อนบ้านตรง ๆ ได้ 5 จุดแยกผิด ทั้งที่มีจุดตัดจริงแค่ 1
                // จุด) — พิกเซลข้างจุดตัดมีเพื่อนบ้านขาวเรียงเป็น "ก้อนต่อเนื่อง" รอบวงแหวน (CN ต่ำ) ต่างจาก
                // ตัวจุดตัดจริงที่เพื่อนบ้านขาว-ดำสลับกันหลายช่วง (CN สูง) — endpoint (n===1) ไม่ต้องคำนวณ CN
                // แยกเพราะเพื่อนบ้านเดียวมีแค่ 1 ก้อนเสมอ (CN=1 พอดี ตรงกับเงื่อนไข n===1 ด้านบนอยู่แล้ว)
                let transitions = 0;
                for (let ri = 0; ri < 8; ri++) if (ring[ri] !== ring[(ri + 1) % 8]) transitions++;
                if (transitions / 2 >= 3) { branchpoints++; cv.circle(dst, new cv.Point(x, y), 3, red, -1); }
              }
            }
          }
          if (ctx) ctx.info = `พบจุดปลาย ${endpoints} จุด · จุดแยก ${branchpoints} จุด`;
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          bin.delete();
        }
      }
      let kernel;
      if (p.shape === "diamond" || p.shape === "octagon" || p.shape === "line") {
        const k0 = Math.max(1, Math.round(p.ksize));
        kernel = buildCustomStrel(cv, p.shape, k0, p.lineAngle);
      } else {
        const shape = { rect: cv.MORPH_RECT, ellipse: cv.MORPH_ELLIPSE, cross: cv.MORPH_CROSS }[p.shape];
        const k0 = Math.max(1, Math.round(p.ksize));
        kernel = cv.getStructuringElement(shape, new cv.Size(k0, k0));
      }
      let dst = null;
      try {
        dst = new cv.Mat();
        const anchor = new cv.Point(-1, -1);
        if (p.op === "erode") cv.erode(src, dst, kernel, anchor, p.iterations);
        else if (p.op === "dilate") cv.dilate(src, dst, kernel, anchor, p.iterations);
        else {
          const opCode = { open: cv.MORPH_OPEN, close: cv.MORPH_CLOSE, gradient: cv.MORPH_GRADIENT, tophat: cv.MORPH_TOPHAT, blackhat: cv.MORPH_BLACKHAT }[p.op];
          cv.morphologyEx(src, dst, opCode, kernel, anchor, p.iterations);
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        kernel.delete();
      }
    },
  },

  // ---------------- 5. Segmentation ----------------
  binary_threshold: {
    name: "Binary Threshold",
    modeHint: "ตั้งเกณฑ์เดียวเองด้วยมือ ใช้ได้ดีถ้าแสงในภาพสม่ำเสมอ",
    needs: "gray",
    desc: "ขีดเส้นความสว่างไว้เส้นเดียว จุดไหนสว่างกว่าเกณฑ์ก็เป็นขาว ต่ำกว่าก็เป็นดำ ไม่มีสีเทาตรงกลางเลย เป็นวิธีแยกวัตถุออกจากพื้นหลังที่ตรงไปตรงมาที่สุด ใช้ได้ดีถ้าแสงในภาพเสมอกันทั้งรูป",
    params: [
      { key: "thresh", type: "number", label: "ค่าเกณฑ์", min: 0, max: 255, step: 1, default: 127, desc: "เส้นแบ่งความสว่าง — จุดที่สว่างกว่านี้เป็นขาว ต่ำกว่าเป็นดำ" },
      { key: "invert", type: "boolean", label: "สลับขาว-ดำ", default: false },
    ],
    apply(cv, src, p) {
      const dst = new cv.Mat();
      cv.threshold(src, dst, p.thresh, 255, boolOr(p.invert, false) ? cv.THRESH_BINARY_INV : cv.THRESH_BINARY);
      return dst;
    },
  },

  otsu_threshold: {
    name: "Otsu Threshold",
    modeHint: "ให้ระบบหาเกณฑ์ที่ดีที่สุดให้เอง ไม่ต้องเดาค่า",
    needs: "gray",
    desc: "ขี้เกียจหาเกณฑ์เองก็ให้ Otsu หาให้ มันดูการกระจายความสว่างทั้งภาพ แล้วเลือกจุดตัดที่แยกของมืดกับของสว่างออกจากกันได้เด็ดขาดที่สุด ไม่ต้องนั่งเดาค่าเอง ค่าที่มันคิดได้จะโผล่ที่แถบรายละเอียด · Triangle (Zack 1977) ไม่มีใน MATLAB · ภาพค่าเดียวทั้งภาพ OpenCV ให้ค่าเกณฑ์ ≈ ค่านั้น − 1 (ค่า 0 ได้ 1, ค่า 1 ได้ −1 ซึ่งแสดงเป็น = -1)",
    params: [
      { key: "invert", type: "boolean", label: "สลับขาว-ดำ", default: false },
      { key: "method", type: "select", label: "วิธีหาเกณฑ์", options: [
        { value: "otsu", label: "Otsu (แยกสองกลุ่มให้ห่างกัน)" },
        { value: "triangle", label: "Triangle (เหมาะกับ histogram ยอดเดียว)", hint: "ลากเส้นจากยอด histogram ไปปลายหาง แล้วเลือกจุดที่ห่างจากเส้นมาก — เหมาะกับวัตถุเล็กบนพื้นเรียบ" },
      ], default: "otsu" },
    ],
    apply(cv, src, p, ctx) {
      const triangle = p.method === "triangle";
      if (triangle && isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const dst = new cv.Mat();
      const type = (boolOr(p.invert, false) ? cv.THRESH_BINARY_INV : cv.THRESH_BINARY) + (triangle ? cv.THRESH_TRIANGLE : cv.THRESH_OTSU);
      try {
        const val = cv.threshold(src, dst, 0, 255, type);
        if (ctx) ctx.info = `ค่าเกณฑ์ที่ ${triangle ? "Triangle" : "Otsu"} คำนวณได้ = ${Math.round(val)}`;
        return dst;
      } catch (err) {
        dst.delete();
        throw err;
      }
    },
  },

  multi_otsu_threshold: {
    name: "Multi-level Otsu (3 กลุ่ม)",
    modeHint: "แบ่งภาพเป็น 3 ระดับความสว่างอัตโนมัติ (มืด/กลาง/สว่าง) แทนที่จะแบ่งแค่ขาว-ดำ",
    needs: "gray",
    desc: "ขยายแนวคิด Otsu จาก 'แบ่ง 2 กลุ่ม' เป็น 'แบ่ง 3 กลุ่ม' โดยหาเกณฑ์ 2 จุดที่ทำให้ความแปรปรวนระหว่างกลุ่มมากที่สุด (หลักการเดียวกับ Otsu ปกติ ขยายจาก 1 เกณฑ์เป็น 2 เกณฑ์) — แต่ละกลุ่มจะถูกแทนด้วยค่าความสว่างตัวแทน (0/128/255) ให้เห็นการแบ่งชัดเจน",
    params: [],
    apply(cv, src, p, ctx) {
      const hist = new cv.Mat();
      let srcVec = null;
      let histMask = null;
      let dst = null;
      try {
        srcVec = new cv.MatVector();
        srcVec.push_back(src);
        const channels = [0];
        const histSize = [256];
        const ranges = [0, 256];
        histMask = new cv.Mat();
        cv.calcHist(srcVec, channels, histMask, hist, histSize, ranges);

        // prefix sums ของ histogram เพื่อคำนวณ mean/weight ของแต่ละช่วงแบบ O(1)
        const P = new Float64Array(257);
        const S = new Float64Array(257);
        for (let i = 0; i < 256; i++) {
          const cnt = hist.floatAt(i, 0);
          P[i + 1] = P[i] + cnt;
          S[i + 1] = S[i] + i * cnt;
        }
        const total = P[256];
        const muT = total > 0 ? S[256] / total : 0;
        const classStats = (lo, hi) => {
          const w = P[hi] - P[lo];
          const s = S[hi] - S[lo];
          return { w, mean: w > 0 ? s / w : 0 };
        };

        // ค่าตั้งต้น (85,170) ใช้เมื่อไม่มีเกณฑ์ที่ valid เลย (เช่น ภาพมีค่าความสว่างน้อยกว่า 3 ระดับ)
        let bestT1 = 85, bestT2 = 170, bestScore = -1;
        for (let t1 = 1; t1 < 254; t1++) {
          for (let t2 = t1 + 1; t2 < 255; t2++) {
            const c0 = classStats(0, t1);
            const c1 = classStats(t1, t2);
            const c2 = classStats(t2, 256);
            if (c0.w === 0 || c1.w === 0 || c2.w === 0) continue; // ข้ามเกณฑ์ที่ทำให้กลุ่มใดกลุ่มหนึ่งว่างเปล่า (กันภาพ degenerate เลือกเกณฑ์แรกที่เจอผิด ๆ)
            const score =
              c0.w * (c0.mean - muT) * (c0.mean - muT) +
              c1.w * (c1.mean - muT) * (c1.mean - muT) +
              c2.w * (c2.mean - muT) * (c2.mean - muT);
            if (score > bestScore) {
              bestScore = score;
              bestT1 = t1;
              bestT2 = t2;
            }
          }
        }

        // สร้าง LUT: [0,t1) -> 0, [t1,t2) -> 128, [t2,256) -> 255
        const lutArr = new Array(256);
        for (let i = 0; i < 256; i++) lutArr[i] = i < bestT1 ? 0 : i < bestT2 ? 128 : 255;
        const lut = cv.matFromArray(1, 256, cv.CV_8UC1, lutArr);
        dst = new cv.Mat();
        cv.LUT(src, lut, dst);
        lut.delete();

        if (ctx) ctx.info = `เกณฑ์ที่คำนวณได้ = ${bestT1}, ${bestT2}`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        hist.delete();
        if (srcVec) srcVec.delete();
        if (histMask) histMask.delete();
      }
    },
  },

  adaptive_threshold: {
    name: "Adaptive Threshold",
    modeHint: "คำนวณเกณฑ์แยกทีละพื้นที่ ทนแสง/เงาไม่สม่ำเสมอได้ดี",
    needs: "gray",
    desc: "คำนวณค่าเกณฑ์แยกกันในแต่ละบริเวณเล็ก ๆ (ไม่ใช้ค่าเดียวทั้งภาพ) จึงทนต่อแสงไม่สม่ำเสมอ/เงาได้ดี — เหมาะกับภาพเอกสาร ลายมือ ที่แสงตกไม่เท่ากัน",
    params: [
      { key: "blockSize", type: "number", label: "ขนาดบริเวณ", min: 3, max: 51, step: 2, default: 11, hint: "เลขคี่ ≥ 3", desc: "ขนาดพื้นที่ที่ใช้คำนวณค่าเกณฑ์เฉพาะที่ — ใหญ่ = ปรับตามแสงแบบหยาบขึ้น" },
      { key: "C", type: "number", label: "ค่าชดเชย C", min: -20, max: 20, step: 1, default: 2, desc: "ค่าที่หักออกจากค่าเฉลี่ยก่อนเทียบเกณฑ์ — ปรับความไวในการตัดขาว/ดำ" },
    ],
    apply(cv, src, p) {
      const dst = new cv.Mat();
      cv.adaptiveThreshold(src, dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, Math.max(3, ensureOdd(p.blockSize)), p.C);
      return dst;
    },
  },

  canny: {
    name: "Canny Edge",
    needs: "gray",
    desc: "ตัวหาขอบที่แม่นสุดในตำรา ทำงานเป็นสเต็ป: ดูก่อนว่าความสว่างเปลี่ยนแรงตรงไหน จับเส้นให้บางเฉียบ แล้วใช้สองเกณฑ์คัด — ขอบเข้มชัดเก็บแน่นอน ส่วนขอบจาง ๆ จะเก็บเฉพาะที่ต่อกับขอบเข้มเท่านั้น เลยได้เส้นสะอาดไม่รก",
    params: [
      { key: "t1", type: "number", label: "เกณฑ์ต่ำ", min: 0, max: 255, step: 1, default: 50, desc: "ขอบอ่อนที่จะถูกเก็บไว้ ถ้าต่อเนื่องกับขอบเข้ม" },
      { key: "t2", type: "number", label: "เกณฑ์สูง", min: 0, max: 255, step: 1, default: 150, desc: "ขอบเข้มที่ถือว่าเป็นขอบแน่นอน" },
    ],
    apply(cv, src, p) {
      const dst = new cv.Mat();
      cv.Canny(src, dst, p.t1, p.t2);
      return dst;
    },
  },

  sobel: {
    name: "Sobel Edge",
    needs: "gray",
    desc: "หาความชัน (gradient) ของความสว่างในแนว X/Y — ค่าชันสูง = ขอบ ให้ผลนุ่มกว่า Canny และเลือกดูทิศทางของขอบได้ (แนวตั้ง/แนวนอน)",
    params: [
      { key: "dir", type: "select", label: "ทิศทาง", options: [
        { value: "x", label: "แนวตั้ง (X)", hint: "จับความชันในแนวนอน → เห็นเส้นขอบแนวตั้งชัด" },
        { value: "y", label: "แนวนอน (Y)", hint: "จับความชันในแนวตั้ง → เห็นเส้นขอบแนวนอนชัด" },
        { value: "both", label: "รวมสองแนว", hint: "รวมขอบทั้งแนวตั้งและแนวนอนเข้าด้วยกัน" },
      ], default: "both" },
      { key: "ksize", type: "number", label: "ขนาด kernel", min: 1, max: 7, step: 2, default: 3, hint: "เลขคี่", desc: "ขนาดตัวกรองหาความชัน — ใหญ่ = จับขอบหยาบ/กว้างกว่า" },
    ],
    apply(cv, src, p) {
      const k = ensureOdd(p.ksize);
      const dst = new cv.Mat();
      const abs = (dx, dy) => {
        const g = new cv.Mat();
        const a = new cv.Mat();
        cv.Sobel(src, g, cv.CV_16S, dx, dy, k);
        cv.convertScaleAbs(g, a);
        g.delete();
        return a;
      };
      if (p.dir === "x") {
        const a = abs(1, 0);
        a.copyTo(dst);
        a.delete();
      } else if (p.dir === "y") {
        const a = abs(0, 1);
        a.copyTo(dst);
        a.delete();
      } else {
        const ax = abs(1, 0);
        const ay = abs(0, 1);
        cv.addWeighted(ax, 0.5, ay, 0.5, 0, dst);
        ax.delete();
        ay.delete();
      }
      return dst;
    },
  },

  laplacian_edge: {
    name: "Laplacian Edge",
    needs: "gray",
    desc: "หาขอบด้วยอนุพันธ์อันดับสอง (Laplacian) โดยตรง — ต่างจาก Sobel ที่ดูความชัน (อันดับหนึ่ง) ตัวนี้ไวต่อจุดที่ความสว่าง 'โค้งเปลี่ยนทิศ' จึงจับขอบบางละเอียดได้ดี แต่ไวต่อ noise มากกว่า Sobel/Canny — มักต้อง Blur ก่อนเสมอ",
    params: [
      { key: "ksize", type: "number", label: "ขนาด kernel", min: 1, max: 7, step: 2, default: 3, hint: "เลขคี่", desc: "ขนาดตัวกรองหาอนุพันธ์อันดับสอง — ใหญ่ = จับขอบหยาบ/กว้างกว่า" },
    ],
    apply(cv, src, p, ctx) {
      const k = ensureOdd(finiteOr(p.ksize, 3));
      let lap = null;
      let dst = null;
      try {
        lap = new cv.Mat();
        cv.Laplacian(src, lap, cv.CV_16S, k);
        dst = new cv.Mat();
        cv.convertScaleAbs(lap, dst);
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (lap) lap.delete();
      }
    },
  },

  scharr_edge: {
    name: "Scharr Edge",
    needs: "gray",
    desc: "หาความชัน (gradient) แบบเดียวกับ Sobel แต่ใช้ kernel ที่ออกแบบให้แม่นเชิงทิศทางกว่า (rotationally symmetric) โดยเฉพาะที่ ksize=3 — เห็นผลชัดกว่า Sobel ในภาพที่มีขอบเฉียง",
    params: [
      { key: "dir", type: "select", label: "ทิศทาง", options: [
        { value: "x", label: "แนวตั้ง (X)", hint: "จับความชันในแนวนอน → เห็นเส้นขอบแนวตั้งชัด" },
        { value: "y", label: "แนวนอน (Y)", hint: "จับความชันในแนวตั้ง → เห็นเส้นขอบแนวนอนชัด" },
        { value: "both", label: "รวมสองแนว", hint: "รวมขอบทั้งแนวตั้งและแนวนอนเข้าด้วยกัน" },
      ], default: "both" },
    ],
    apply(cv, src, p, ctx) {
      const abs = (dx, dy) => {
        const g = new cv.Mat();
        const a = new cv.Mat();
        try {
          cv.Scharr(src, g, cv.CV_16S, dx, dy);
          cv.convertScaleAbs(g, a);
          return a;
        } finally {
          g.delete();
        }
      };
      let dst = null;
      let ax = null;
      let ay = null;
      try {
        dst = new cv.Mat();
        if (p.dir === "x") {
          ax = abs(1, 0);
          ax.copyTo(dst);
        } else if (p.dir === "y") {
          ay = abs(0, 1);
          ay.copyTo(dst);
        } else {
          ax = abs(1, 0);
          ay = abs(0, 1);
          cv.addWeighted(ax, 0.5, ay, 0.5, 0, dst);
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (ax) ax.delete();
        if (ay) ay.delete();
      }
    },
  },

  // ===================================================================
  // Prewitt / Roberts / LoG (2026-09-30) — เข้ากล่อง "Edge Detection" เดิม เพื่อให้สลับ dropdown เทียบ
  // Canny → Sobel → Scharr → Laplacian → Prewitt → Roberts → LoG บนภาพเดียวกันได้ (แบบฝึกหัดบทแรกของตำรา)
  // ทั้ง prewitt/roberts ใช้โครง CV_16S + convertScaleAbs + addWeighted(0.5, 0.5) เหมือน sobel/scharr_edge
  // เป๊ะ — ห้ามเปลี่ยนไปใช้ cv.magnitude (มีในบิลด์และใช้ได้ แต่รวมสองแนวคนละสูตรกับ Sobel จึงเทียบกันไม่ได้
  // และหลุด parity กับโค้ด Python ที่ codegen.js export)
  // ===================================================================
  prewitt_edge: {
    name: "Prewitt Edge",
    needs: "gray",
    desc: "หาขอบด้วยการเทียบความสว่างซ้าย-ขวา (หรือบน-ล่าง) โดยเฉลี่ยเพื่อนบ้านทั้ง 3 แถวด้วยน้ำหนักเท่ากันหมด (เฉลี่ยแบบ 'กล่อง') — ต่างจาก Sobel ที่ถ่วงน้ำหนักแถวกลางเป็น 2 เท่า บนภาพถ่ายทั่วไปภาพของ Prewitt จะจางกว่า Sobel ประมาณ 3/4 เท่า เพราะผลรวมน้ำหนักในตัวกรองน้อยกว่า · แต่ถ้าขอบคอนทราสต์สูงมาก (ภาพขาว-ดำล้วน หรือภาพที่ผ่าน Threshold มาแล้ว) ค่าจะตันที่ค่าสูงสุดทั้งคู่จนได้ภาพเหมือนกันเป๊ะ ความต่างจะเห็นชัดบนภาพถ่ายที่มีขอบนุ่ม ๆ · ส่วนความทนต่อจุดรบกวน (noise) วัดแล้วใกล้เคียงกันมาก · เหมาะใช้เทียบกับ Sobel/Scharr บนภาพเดียวกันเพื่อดูว่า 'น้ำหนักใน kernel' เปลี่ยนผลลัพธ์อย่างไร · ใช้ในงานจริงตอนอยากได้เส้นขอบคร่าว ๆ เร็ว ๆ ก่อนส่งต่อ Threshold",
    params: [
      { key: "dir", type: "select", label: "ทิศทาง", options: [
        { value: "x", label: "แนวตั้ง (X)", hint: "จับความชันในแนวนอน → เห็นเส้นขอบแนวตั้งชัด" },
        { value: "y", label: "แนวนอน (Y)", hint: "จับความชันในแนวตั้ง → เห็นเส้นขอบแนวนอนชัด" },
        { value: "both", label: "รวมสองแนว", hint: "รวมขอบทั้งแนวตั้งและแนวนอนเข้าด้วยกัน" },
      ], default: "both" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      // kernel 3x3 ของ Prewitt: น้ำหนักทุกแถว/ทุกคอลัมน์เท่ากัน (Sobel ใช้ 1,2,1 — จุดต่างเดียวของสองตัวนี้)
      const abs = (kdata) => {
        const k = cv.matFromArray(3, 3, cv.CV_32FC1, kdata);
        const g = new cv.Mat();
        const a = new cv.Mat();
        try {
          cv.filter2D(src, g, cv.CV_16S, k, new cv.Point(-1, -1), 0, cv.BORDER_DEFAULT);
          cv.convertScaleAbs(g, a);
          return a;
        } catch (err) {
          a.delete();
          throw err;
        } finally {
          g.delete();
          k.delete();
        }
      };
      const KX = [-1, 0, 1, -1, 0, 1, -1, 0, 1]; // ความชันแนวนอน → ขอบแนวตั้ง
      const KY = [-1, -1, -1, 0, 0, 0, 1, 1, 1]; // ความชันแนวตั้ง → ขอบแนวนอน
      let dst = null;
      let ax = null;
      let ay = null;
      try {
        dst = new cv.Mat();
        // ภาพ 3/4 ช่อง (เรียก op ตรง ๆ นอก runPipeline) ไม่ต้อง guard: filter2D ทำงานทีละช่องอย่างถูกต้อง
        // ได้ edge map ของแต่ละช่อง เหมือน sobel/scharr_edge/laplacian_edge ที่มีอยู่เดิมทุกตัว
        if (p.dir === "x") {
          ax = abs(KX);
          ax.copyTo(dst);
        } else if (p.dir === "y") {
          ay = abs(KY);
          ay.copyTo(dst);
        } else { // ค่าประหลาด/undefined ตกมาที่ "both" เสมอ ห้าม throw
          ax = abs(KX);
          ay = abs(KY);
          cv.addWeighted(ax, 0.5, ay, 0.5, 0, dst);
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (ax) ax.delete();
        if (ay) ay.delete();
      }
    },
  },

  roberts_edge: {
    name: "Roberts Cross Edge",
    needs: "gray",
    desc: "ตัวหาขอบที่เก่าและเล็กที่สุดในตำรา ใช้ตัวกรองแค่ 2×2 สองตัว เทียบพิกเซลตามแนวทแยงคนละแนวกัน (ตัวหนึ่งแนว ↘ อีกตัวแนว ↗) จึงเร็วมากและจับขอบได้บางที่สุด แต่ไวต่อ noise สูง · ในโหมด 'รวมสองแนว' ขอบแนวตั้งและแนวนอนจะสว่างเต็มที่ (255) เพราะตัวกรองทั้งสองตอบสนองพร้อมกัน ส่วนขอบที่เอียง 45° ได้ความสว่างแค่ครึ่งเดียว (128) เพราะมีตัวกรองเดียวที่ตอบสนอง — ถ้าอยากเน้นขอบเอียงให้เลือกดูทีละแนว · ใช้เป็นตัวเปรียบเทียบให้เห็นว่า 'kernel ยิ่งเล็ก ยิ่งคม แต่ยิ่งไวต่อจุดรบกวน' ในงานจริงมักใช้กับภาพที่คมและสะอาดอยู่แล้ว เช่น ภาพสแกนลายเส้น",
    params: [
      { key: "dir", type: "select", label: "ทิศทาง", options: [
        { value: "d1", label: "แนวทแยง ↘ (Gx)", hint: "เทียบพิกเซลตามแนวทแยงลงขวา → เห็นขอบที่เอียงไปทาง ↗ ชัด" },
        { value: "d2", label: "แนวทแยง ↗ (Gy)", hint: "เทียบพิกเซลตามแนวทแยงขึ้นขวา → เห็นขอบที่เอียงไปทาง ↘ ชัด" },
        { value: "both", label: "รวมสองแนว", hint: "รวมขอบทั้งสองแนวทแยงเข้าด้วยกัน — เห็นขอบครบทุกทิศ" },
      ], default: "both" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const abs = (kdata) => {
        const k = cv.matFromArray(2, 2, cv.CV_32FC1, kdata);
        const g = new cv.Mat();
        const a = new cv.Mat();
        try {
          // anchor ต้องเป็น (-1,-1) เสมอ: เป็นค่า default ที่ cv2.filter2D ของ Python ใช้ ถ้าเปลี่ยนเป็น (0,0)
          // "เพื่อแก้การเลื่อนครึ่งพิกเซล" ภาพฝั่ง JS จะหลุด parity กับโค้ดที่ export ทันที (จำนวนพิกเซลขอบเท่ากัน
          // แต่ตำแหน่งเลื่อนไป 1 px) — ยืนยันแล้วในใบสั่งงาน §3.2
          cv.filter2D(src, g, cv.CV_16S, k, new cv.Point(-1, -1), 0, cv.BORDER_DEFAULT);
          cv.convertScaleAbs(g, a);
          return a;
        } catch (err) {
          a.delete();
          throw err;
        } finally {
          g.delete();
          k.delete();
        }
      };
      const K1 = [1, 0, 0, -1]; // ↘
      const K2 = [0, 1, -1, 0]; // ↗
      let dst = null;
      let a1 = null;
      let a2 = null;
      try {
        dst = new cv.Mat();
        if (p.dir === "d1") {
          a1 = abs(K1);
          a1.copyTo(dst);
        } else if (p.dir === "d2") {
          a2 = abs(K2);
          a2.copyTo(dst);
        } else {
          a1 = abs(K1);
          a2 = abs(K2);
          cv.addWeighted(a1, 0.5, a2, 0.5, 0, dst);
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (a1) a1.delete();
        if (a2) a2.delete();
      }
    },
  },

  log_edge: {
    name: "LoG Edge (จุดตัดศูนย์)",
    needs: "gray",
    desc: "วิธีหาขอบแบบคลาสสิกของ Marr-Hildreth: เบลอภาพด้วย Gaussian ก่อน (ลด noise) แล้วหาอนุพันธ์อันดับสองด้วย Laplacian — ตำแหน่งที่ค่าของ Laplacian 'เปลี่ยนจากบวกเป็นลบ' (จุดตัดศูนย์) คือกึ่งกลางของขอบจริง ผลที่ได้จึงเป็นเส้นขอบบางหนา 1 พิกเซล ต่างจาก Sobel/Prewitt ที่ได้แถบขอบหนาหลายพิกเซล · ถ้าตั้ง 'เกณฑ์ความชันที่จุดตัด' เป็น 0 เส้นจะปิดเป็นวงครบทุกวง พอเพิ่มเกณฑ์ขึ้นเพื่อตัดขอบจาง ๆ ที่เกิดจาก noise ทิ้ง เส้นก็จะเริ่มขาดเป็นท่อน (ที่ค่าตั้งต้นก็ขาดแล้ว) — เป็นการแลกกันระหว่าง 'เส้นต่อเนื่อง' กับ 'เส้นสะอาด' · เปลี่ยนมุมมองเป็น 'ค่าของ LoG' เพื่อดูรูปหมวกเม็กซิกัน (สว่าง = บวก, เทากลาง = ศูนย์, มืด = ลบ) แล้วจะเข้าใจทันทีว่าจุดตัดศูนย์มาจากไหน · ในงานจริงใช้ตอนต้องการเส้นขอบต่อเนื่องเพื่อไปวัดขนาดหรือหา contour ต่อ",
    params: [
      { key: "sigma", type: "number", label: "ความแรงการเบลอ (sigma)", min: 0.5, max: 5, step: 0.1, default: 2, hint: "มาก = ขอบน้อยลง ต้องลดเกณฑ์ตาม", desc: "ความกว้างของ Gaussian ที่ใช้ลด noise ก่อนหาอนุพันธ์ — ใหญ่ขึ้น = กรอง noise ได้มากแต่กลืนรายละเอียดเล็ก ๆ ไปด้วย · ถ้าเพิ่มค่านี้แล้วเส้นขอบหายไปเกือบหมด ไม่ใช่โปรแกรมพัง: ยิ่งเบลอมาก ค่าของ LoG ยิ่งเล็กลง (ประมาณตาม sigma ยกกำลังสอง) แต่ 'เกณฑ์ความชันที่จุดตัด' ไม่ได้ลดตามให้ ต้องลดเกณฑ์เองด้วย · ค่าสูงยังทำให้ช้าลงมาก — ที่ 1080p sigma 5 ใช้เวลาราว 360 ms ต่อเฟรม (ประมาณ 3 ภาพ/วินาที)" },
      { key: "zeroThresh", type: "number", label: "เกณฑ์ความชันที่จุดตัด", min: 0, max: 50, step: 2, default: 8, hint: "0 = เอาทุกจุดตัด", desc: "จุดตัดศูนย์ที่ค่าสองข้างต่างกันน้อยกว่านี้จะไม่นับเป็นขอบ — ตั้ง 0 จะเห็นเส้นเต็มภาพ (บทเรียนว่าทำไม LoG ดิบ ๆ ใช้ไม่ได้) · เลื่อนทีละ 2 เพราะค่าที่ Laplacian คำนวณได้เป็นจำนวนเต็มคู่เสมอ เลขคี่จึงให้ผลเท่ากับเลขคู่ที่อยู่ก่อนหน้า" },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "zerocross", label: "เส้นขอบ (จุดตัดศูนย์)", hint: "ผลลัพธ์ที่ใช้งานจริง — เส้นขอบบาง 1 พิกเซล" },
        { value: "response", label: "ค่าของ LoG (หมวกเม็กซิกัน)", hint: "โหมดอธิบายหลักการ: 128 = ศูนย์, สว่าง = บวก, มืด = ลบ (ปรับความสว่างใหม่ทุกเฟรม บนกล้องสดจึงกะพริบ)" },
      ], default: "zerocross" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const sigma = numIn(p.sigma, 2, 0.5, 5); // clamp ก่อนเสมอ: sigma <= 0 คู่กับ ksize (0,0) ทำให้ GaussianBlur throw
      const thr = numIn(p.zeroThresh, 8, 0, 50); // 0 ต้องใช้ได้จริง — ห้ามใช้ `|| d` (P-13)
      let gray = null; // null = src เป็น 1 ช่องอยู่แล้ว (ห้าม delete src)
      let blur = null;
      let lap = null;
      let out = null;
      try {
        // *** บังคับ *** ลูป zero-crossing ด้านล่างอ่าน data32F แบบ 1 ช่อง ถ้า Laplacian ออกมา 3 ช่องจะอ่านค่า
        // ผิดทั้งภาพโดยไม่ throw (บั๊กเงียบ) — ต่างจาก prewitt/roberts ที่ filter2D จัดการหลายช่องได้เอง
        gray = grayIfNeeded(cv, src);
        const g = gray || src;
        blur = new cv.Mat();
        // ksize (0,0) = ให้ OpenCV คำนวณขนาด kernel จาก sigma เอง (สูตรเดียวกันทั้ง opencv.js และ cv2 → parity ตรง)
        cv.GaussianBlur(g, blur, new cv.Size(0, 0), sigma, sigma, cv.BORDER_DEFAULT);
        lap = new cv.Mat();
        cv.Laplacian(blur, lap, cv.CV_32F, 3); // ksize คงที่ 3 โดยตั้งใจ ไม่เปิดให้ปรับ (พารามิเตอร์น้อย = สอนง่าย)
        if (p.view === "response") {
          const mm = cv.minMaxLoc(lap);
          const maxAbs = Math.max(Math.abs(mm.minVal), Math.abs(mm.maxVal));
          const alpha = maxAbs > 1e-6 ? 127 / maxAbs : 1; // ภาพเรียบสนิท → alpha 1 กันหารศูนย์
          out = new cv.Mat();
          // beta 128 = "ค่าศูนย์อยู่กลางสเกลเทา" — ห้ามใช้ normalize(NORM_MINMAX) แทน เพราะศูนย์จะไม่อยู่ที่ 128
          // แล้วภาพจะไม่สอนอะไรเลย และทำ parity กับ Python ยากขึ้น
          cv.convertScaleAbs(lap, out, alpha, 128);
          if (ctx) ctx.info = `ค่า LoG: ต่ำสุด ${mm.minVal.toFixed(1)} · สูงสุด ${mm.maxVal.toFixed(1)} (128 = ศูนย์)`;
        } else {
          const w = lap.cols;
          const h = lap.rows;
          const f = lap.data32F; // cache getter! (data/data32F สร้าง TypedArray view ใหม่ทุกครั้งที่อ่าน)
          out = new cv.Mat.zeros(h, w, cv.CV_8UC1);
          const o = out.data; // cache getter! เช่นกัน
          let n = 0;
          // แถวสุดท้ายและคอลัมน์สุดท้ายเป็น 0 เสมอ "โดยการออกแบบ" (ไม่มีเพื่อนบ้านขวา/ล่างให้เทียบ) —
          // ห้ามแก้ให้ "ครบ" ภายหลัง จะทำ parity กับโค้ด Python ที่ export ไปพัง
          for (let y = 0; y < h - 1; y++) {
            const row = y * w;
            for (let x = 0; x < w - 1; x++) {
              const c = f[row + x];
              const r = f[row + x + 1];
              const d = f[row + w + x];
              if (((c > 0) !== (r > 0) && Math.abs(c - r) > thr)
                || ((c > 0) !== (d > 0) && Math.abs(c - d) > thr)) { o[row + x] = 255; n++; }
            }
          }
          if (ctx) ctx.info = `พบขอบ (จุดตัดศูนย์) ${n} พิกเซล · sigma ${sigma} · เกณฑ์ ${thr}`;
        }
        return out;
      } catch (err) {
        if (out) out.delete();
        throw err;
      } finally {
        if (lap) lap.delete();
        if (blur) blur.delete();
        if (gray) gray.delete();
      }
    },
  },

  find_contours: {
    name: "Find Contours",
    needs: "gray",
    desc: "หาเส้นรอบรูปของวัตถุ (บริเวณสีขาว) ในภาพ binary แล้ววาดทับด้วยสีต่าง ๆ พร้อมนับจำนวน — ต่อหลัง Threshold เพื่อดูว่าการแบ่งส่วนภาพนำไปสู่การ 'นับวัตถุ' ได้อย่างไร จำนวนที่พบแสดงที่แถบรายละเอียด",
    params: [
      { key: "minArea", type: "number", label: "พื้นที่ขั้นต่ำ (px)", min: 0, max: 5000, step: 50, default: 100, hint: "กรองจุดเล็ก ๆ ทิ้ง", desc: "วัตถุที่มีพื้นที่น้อยกว่านี้ (พิกเซล) จะไม่ถูกนับว่าเป็นวัตถุ" },
    ],
    apply(cv, src, p, ctx) {
      const contours = new cv.MatVector();
      const hier = new cv.Mat();
      cv.findContours(src, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      const dst = new cv.Mat();
      cv.cvtColor(src, dst, cv.COLOR_GRAY2RGB);
      let count = 0;
      for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour);
        contour.delete();
        if (area < p.minArea) continue;
        count++;
        const color = new cv.Scalar((i * 97 + 60) % 256, (i * 151 + 120) % 256, (i * 67 + 200) % 256);
        cv.drawContours(dst, contours, i, color, 2);
      }
      if (ctx) ctx.info = `พบวัตถุ ${count} ชิ้น`;
      contours.delete();
      hier.delete();
      return dst;
    },
  },

  connected_components: {
    name: "Connected Components",
    needs: "gray",
    desc: "จัดกลุ่มพิกเซลสีขาวที่ติดกันเป็น 'วัตถุ' แต่ละก้อน แล้วระบายสีต่างกันทีละก้อน (ต่างจาก Contours ที่วาดแค่เส้นรอบ — อันนี้ระบายทั้งพื้นที่) ต่อหลัง Threshold เพื่อดูการแบ่งวัตถุแบบพื้นที่ พร้อมนับจำนวน",
    params: [],
    apply(cv, src, p, ctx) {
      // connectedComponents ต้องรับภาพ binary 8-bit เท่านั้น ภาพ gray จากกล้อง
      // ยังมีค่าระหว่าง 0..255 จึงแปลงด้วย Otsu ก่อนเสมอ (ภาพ binary เดิมยังได้ผลเดิม)
      const binary = new cv.Mat();
      const labels = new cv.Mat();
      let dst = null;
      try {
        cv.threshold(src, binary, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
        const n = cv.connectedComponents(binary, labels, 8, cv.CV_32S);
        dst = new cv.Mat(labels.rows, labels.cols, cv.CV_8UC3);
        const labelData = labels.data32S;
        const rgb = dst.data;
        for (let i = 0, j = 0; i < labelData.length; i++, j += 3) {
          const label = labelData[i];
          if (label === 0) {
            rgb[j] = 0; rgb[j + 1] = 0; rgb[j + 2] = 0;
          } else {
            rgb[j] = 55 + ((label * 67) % 200);
            rgb[j + 1] = 55 + ((label * 131) % 200);
            rgb[j + 2] = 55 + ((label * 197) % 200);
          }
        }
        if (ctx) ctx.info = `พบวัตถุ ${Math.max(0, n - 1)} ชิ้น`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        binary.delete();
        labels.delete();
      }
    },
  },

  distance_transform: {
    name: "Distance Transform",
    needs: "gray",
    desc: "คำนวณ 'ระยะห่างจากพื้นหลังที่ใกล้ที่สุด' ของทุกพิกเซลสีขาว แล้วแสดงเป็นความสว่าง (ยิ่งอยู่ลึกกลางวัตถุ ยิ่งสว่าง) — เป็นขั้นตอนภายในของ Watershed ที่ใช้หา 'ใจกลาง' ของแต่ละวัตถุ ดูเดี่ยว ๆ ตรงนี้ช่วยให้เข้าใจว่า Watershed ตัดสินใจแยกวัตถุติดกันอย่างไร",
    params: [
      { key: "maskSize", type: "select", label: "ความละเอียดของระยะ", options: [
        { value: "3", label: "3x3 (เร็ว, หยาบกว่า)" },
        { value: "5", label: "5x5 (แม่นกว่า, ช้ากว่านิดหน่อย)" },
      ], default: "5" },
    ],
    apply(cv, src, p, ctx) {
      const bin = new cv.Mat();
      const dist = new cv.Mat();
      let dst = null;
      try {
        cv.threshold(src, bin, 127, 255, cv.THRESH_BINARY);
        const noBackground = cv.countNonZero(bin) === bin.rows * bin.cols; // ไม่มีพิกเซลดำ (พื้นหลัง) เลย -> ไม่มีจุดอ้างอิงระยะ
        const maskSize = p.maskSize === "3" ? 3 : 5;
        cv.distanceTransform(bin, dist, cv.DIST_L2, maskSize);
        dst = new cv.Mat();
        cv.normalize(dist, dst, 0, 255, cv.NORM_MINMAX);
        dst.convertTo(dst, cv.CV_8U);
        if (ctx) {
          if (noBackground) {
            ctx.info = "ไม่มีพื้นหลัง (สีดำ) ในภาพ จึงวัดระยะไม่ได้ — ลองใส่ Threshold ก่อน";
          } else {
            const mm = cv.minMaxLoc(dist);
            ctx.info = `ระยะไกลสุดจากขอบ = ${mm.maxVal.toFixed(1)} px`;
          }
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        bin.delete();
        dist.delete();
      }
    },
  },

  local_std_filter: {
    name: "Local Std Filter (Texture)",
    needs: "gray",
    desc: "หา 'ความหยาบของพื้นผิว' (texture) ในแต่ละบริเวณเล็ก ๆ โดยคำนวณส่วนเบี่ยงเบนมาตรฐาน (standard deviation) ของความสว่างรอบแต่ละพิกเซล — บริเวณเรียบ (พื้นหลัง, กระดาษเปล่า) จะได้ค่าต่ำ/มืด บริเวณมีลายละเอียด/noise/ขอบวัตถุ จะได้ค่าสูง/สว่าง ใช้แยกพื้นผิวเรียบจากพื้นผิวมีลาย โดยไม่สนใจว่าสว่างหรือมืดเฉลี่ยเท่าไหร่",
    params: [
      { key: "windowSize", type: "number", label: "ขนาดบริเวณ (px)", min: 3, max: 21, step: 2, default: 9, hint: "เลขคี่", desc: "ขนาดหน้าต่างที่ใช้คำนวณค่าเบี่ยงเบนมาตรฐาน — ใหญ่ = จับ texture หยาบ/กว้างขึ้น" },
    ],
    apply(cv, src, p, ctx) {
      const k = ensureOdd(finiteOr(p.windowSize, 9));
      let f32 = null, mean = null, sq = null, meanSq = null, meanSquared = null, variance = null, stdF = null, norm = null, dst = null;
      try {
        f32 = new cv.Mat();
        src.convertTo(f32, cv.CV_32F);
        mean = new cv.Mat();
        cv.boxFilter(f32, mean, cv.CV_32F, new cv.Size(k, k));
        sq = new cv.Mat();
        cv.multiply(f32, f32, sq);
        meanSq = new cv.Mat();
        cv.boxFilter(sq, meanSq, cv.CV_32F, new cv.Size(k, k));
        meanSquared = new cv.Mat();
        cv.multiply(mean, mean, meanSquared);
        variance = new cv.Mat();
        cv.subtract(meanSq, meanSquared, variance);
        // float rounding อาจเหลือ variance ติดลบนิดหน่อย (เช่น -1e-6) ในบริเวณเรียบสนิท — ยืนยันแล้วว่า
        // cv.sqrt กับค่าลบให้ NaN จึงต้อง clip ที่ 0 ก่อนเสมอ ใช้ cv.threshold(...THRESH_TOZERO) (ยืนยันว่า
        // ปลอดภัยกว่าเดา signature ของ cv.max — ดู work log)
        cv.threshold(variance, variance, 0, 0, cv.THRESH_TOZERO);
        stdF = new cv.Mat();
        cv.sqrt(variance, stdF);
        norm = new cv.Mat();
        cv.normalize(stdF, norm, 0, 255, cv.NORM_MINMAX);
        dst = new cv.Mat();
        norm.convertTo(dst, cv.CV_8U);
        const mm = cv.minMaxLoc(variance);
        if (ctx) ctx.info = `ความหยาบพื้นผิวสูงสุด (ก่อน normalize) = ${Math.sqrt(mm.maxVal).toFixed(1)}`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (f32) f32.delete();
        if (mean) mean.delete();
        if (sq) sq.delete();
        if (meanSq) meanSq.delete();
        if (meanSquared) meanSquared.delete();
        if (variance) variance.delete();
        if (stdF) stdF.delete();
        if (norm) norm.delete();
      }
    },
  },

  local_entropy_filter: {
    name: "Local Entropy Filter (Texture)",
    needs: "gray",
    desc: "วัด 'ความซับซ้อน/สุ่ม' ของพื้นผิวในแต่ละบริเวณเล็ก ๆ ด้วยค่า entropy ทางสถิติ — บริเวณที่มีความสว่างหลากหลายมาก (ลาย, noise, พื้นผิวขรุขระ) จะได้ entropy สูง บริเวณเรียบ/สีเดียวจะได้ entropy ต่ำ ต่างจาก Std Filter เพราะ entropy ไวต่อ 'ความหลากหลายของระดับสี' มากกว่าความกว้างของการกระจาย — รอบนี้ลดความละเอียด histogram ลง (ไม่ใช่ 256 ระดับเต็มแบบตำรา) เพื่อให้ยังทำงานได้ทันกับกล้องสด และวิเคราะห์บนภาพย่อ ≤320px เสมอ",
    params: [
      { key: "windowSize", type: "select", label: "ขนาดหน้าต่าง", options: [
        { value: "3", label: "3x3 (เร็วสุด)" },
        { value: "5", label: "5x5 (แนะนำ)" },
        { value: "7", label: "7x7 (ช้ากว่า, เห็น texture หยาบขึ้น)" },
      ], default: "5" },
      { key: "bins", type: "select", label: "จำนวนระดับสีที่ใช้นับ (หยาบกว่าของจริงเพื่อความเร็ว)", options: [
        { value: "8", label: "8 ระดับ (เร็วสุด, หยาบสุด)" },
        { value: "16", label: "16 ระดับ (แนะนำ)" },
        { value: "32", label: "32 ระดับ (ช้ากว่า, ละเอียดขึ้น)" },
      ], default: "16" },
    ],
    apply(cv, src, p, ctx) {
      const maxAnalysisSide = 320;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      const win = { 3: 3, 5: 5, 7: 7 }[p.windowSize] || 5;
      const bins = { 8: 8, 16: 16, 32: 32 }[p.bins] || 16;
      const shift = 8 - Math.log2(bins);
      const half = Math.floor(win / 2);
      let analysis = null, small = null, dst = null;
      try {
        analysis = new cv.Mat();
        const aw = Math.max(1, Math.round(src.cols * scale));
        const ah = Math.max(1, Math.round(src.rows * scale));
        if (scale < 1) cv.resize(src, analysis, new cv.Size(aw, ah), 0, 0, cv.INTER_AREA);
        else src.copyTo(analysis);

        const data = analysis.data; // Uint8Array, 1 channel (needs:"gray")
        const w = analysis.cols, h = analysis.rows;
        const out = new Uint8Array(w * h);
        const hist = new Int32Array(bins);
        const maxEntropy = Math.log2(bins); // ใช้ normalize output ให้เป็น 0-255
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            hist.fill(0);
            let n = 0;
            for (let dy = -half; dy <= half; dy++) {
              const yy = y + dy;
              if (yy < 0 || yy >= h) continue;
              const rowBase = yy * w;
              for (let dx = -half; dx <= half; dx++) {
                const xx = x + dx;
                if (xx < 0 || xx >= w) continue;
                hist[data[rowBase + xx] >> shift]++;
                n++;
              }
            }
            let e = 0;
            for (let i = 0; i < bins; i++) {
              if (hist[i] > 0) { const pr = hist[i] / n; e -= pr * Math.log2(pr); }
            }
            out[y * w + x] = Math.round((e / maxEntropy) * 255);
          }
        }
        small = new cv.Mat(h, w, cv.CV_8UC1);
        small.data.set(out);
        dst = new cv.Mat();
        if (scale < 1) cv.resize(small, dst, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_LINEAR);
        else small.copyTo(dst);
        if (ctx) ctx.info = `Entropy filter ${win}x${win}, ${bins} ระดับ (ประมาณ ไม่ใช่ 256 ระดับเต็ม)`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (analysis) analysis.delete();
        if (small) small.delete();
      }
    },
  },

  // ===================================================================
  // Region Properties (Tier 2, 18c) — sibling ของ find_contours/connected_components ในกลุ่มเดียวกัน
  // ตัวแรกที่ใช้ ctx.values แบบ "table" (แถวต่อวัตถุ) แทน "scalars" (ค่าตัวเดียว) ที่ 18b ใช้ — ดู
  // worklog 18b CV-eng recheck: ยืนยันแล้วไม่ต้องมีการย่อภาพวิเคราะห์ (178 วัตถุ @ 1080p ≈39ms วัดจริง)
  // จึงคำนวณจาก contour ที่ full-resolution เสมอ ไม่มี downscale/opt-in param ใด ๆ ในรอบนี้
  // ===================================================================
  regionprops: {
    name: "Region Properties (วัดค่าต่อวัตถุ)",
    needs: "gray",
    desc: "วัดคุณสมบัติของแต่ละวัตถุ (blob) แยกเป็นรายชิ้น ไม่ใช่แค่นับจำนวนรวมแบบ Find Contours — ต่อชิ้น: พื้นที่ (area), จุดศูนย์กลาง (centroid), กรอบล้อม (bounding box), เส้นรอบรูป (perimeter), ความกลม (circularity, 1=กลมสมบูรณ์), และมุมเอียงของแกนหลัก (orientation) เป็นเครื่องมือวัดค่าจริงจังที่สุดในแอปนี้ ใกล้เคียง regionprops ของ MATLAB — ภาพที่ไม่ใช่ขาว-ดำล้วนจะถูก binarize ที่ค่ากึ่งกลาง 127 ก่อนเสมอ (เหมือนธรรมเนียม fill_holes ในไฟล์นี้) วัตถุที่แตะขอบภาพจะยังถูกวัด/นับตามปกติ (ค่าที่ได้อาจไม่สมบูรณ์เพราะถูกตัดด้วยขอบเฟรม — เหมือนพฤติกรรมเริ่มต้นของ regionprops ใน MATLAB) · มุมเอียง (orientation_deg) มีหน่วยเป็นองศา ช่วง 0–180°: 0° = แนวนอน และมุมเพิ่มขึ้นตามเข็มนาฬิกา (เพราะแกน y ของภาพชี้ลงล่าง ไม่ใช่ขึ้นบนแบบกราฟคณิตศาสตร์) เช่น 90° = แนวตั้ง, 45° = เอียงลงไปทางขวา — 0° กับ 180° คือแนวเดียวกัน (แกนไม่มีหัวท้าย) · วัตถุที่มีจุดบนเส้นรอบรูปน้อยกว่า 5 จุด (เช่น สี่เหลี่ยมตรง) จะไม่มีค่ามุม — ในตารางแสดงเป็นเครื่องหมาย — (ไม่ใช่ 0°) และโค้ด Python ที่ export แสดงเป็น - · จุดน้อยกว่า 5 เกิดกับสี่เหลี่ยมที่ขอบเป็นเส้นตรงล้วน (ตั้งตรง หรือเอียง 45° พอดี จะเหลือแค่ 4 จุดมุม) และวัตถุที่เล็กมาก · วัตถุที่กลมหรือเกือบกลม (ความรีใกล้ 0) ก็ไม่มีมุมที่มีความหมาย เพราะไม่มีทิศ 'ยาว' ให้วัด ค่ามุมจะเปลี่ยนไปมาตามขอบพิกเซลเพียงเล็กน้อย (วงกลมอาจได้ 90° ทั้งที่ไม่มีทิศไหนพิเศษ) — ให้ดูความรีก่อนเชื่อมุม" +
      " · คอลัมน์เพิ่มเติม (ชื่อตรงกับ regionprops ของ MATLAB): ความตัน (solidity) = พื้นที่ ÷ พื้นที่รูปนูนที่ห่อวัตถุ (1 = ไม่เว้า/ไม่มีรู) · ความรี (eccentricity) 0 = กลม → ใกล้ 1 = เป็นเส้น (0.87 ≈ ยาว 2 เท่าของกว้าง, 0.97 ≈ 4 เท่า) · แกนยาว/แกนสั้น (major/minor axis) ของวงรีที่กระจายตัวเท่าวัตถุ · เต็มกรอบ (extent) = พื้นที่ ÷ กรอบสี่เหลี่ยมตรง · เส้นผ่านศูนย์กลางเทียบเท่า (equivalent diameter) · ความสว่างเฉลี่ย (mean intensity)" +
      " · คอลัมน์ใหม่เหล่านี้นับจากพิกเซลจริง (area_px = พื้นที่นับพิกเซล รูไม่ถูกนับ เหมือน MATLAB) ส่วน area เดิมคือพื้นที่ของเส้นหลายเหลี่ยมที่ลากผ่านกลางพิกเซลขอบ และนับรวมรู (สี่เหลี่ยม 10×10: area 81, area_px 100)" +
      " · แกนยาว/แกนสั้น/ความรี มีค่าให้ทุกวัตถุ แม้มุมเอียง (orientation) จะแสดง —" +
      " · ความสว่างเฉลี่ย = ความสว่างของภาพที่เข้ากล่องนี้ (MATLAB ใช้ภาพที่สองใน regionprops(BW, I) แต่กล่องนี้มีทางเข้าเดียว) — หลัง Threshold จึงเป็น 255 เสมอ; ถ้าอยากวัดความสว่างจริงให้ต่อภาพเทาเข้ากล่องนี้ตรง ๆ (กล่องแยกวัตถุที่ > 127 เอง)" +
      " · เส้นรูปนูนของความตัน ผ่านกลางพิกเซล วัตถุเล็กมากจึงอาจต่างจาก MATLAB (ซึ่งห่อที่มุมพิกเซล)" +
      " · วัตถุเกาะที่อยู่ในรูของวัตถุอื่นจะนับรวมกับวัตถุชั้นนอก (MATLAB นับเป็นอีกชิ้นหนึ่ง)" +
      " · แปลงเป็นหน่วยจริงต้องมีของอ้างอิงในภาพ · ตัวคูณเดียวใช้ได้เมื่อเลนส์ไม่บิดเบี้ยวมาก (ดู Undistort) และชิ้นงานอยู่ระนาบเดียว · พื้นที่คูณ (มม./px)²",
    params: [
      { key: "minArea", type: "number", label: "พื้นที่ขั้นต่ำ (px)", min: 0, max: 5000, step: 10, default: 50, desc: "วัตถุที่เล็กกว่านี้จะไม่ถูกนับ/วัด (กันจุด noise เล็ก ๆ ปนเข้ามา)" },
      { key: "maxObjects", type: "number", label: "แสดงวัตถุสูงสุด", min: 0, max: 5000, step: 50, default: 500, hint: "0 = ไม่จำกัด", desc: "วัด/วาดเฉพาะวัตถุที่ใหญ่ที่สุดตามจำนวนนี้ (จำนวนวัตถุทั้งหมดที่พบยังนับครบและแสดงเสมอ) — กันภาพที่มีจุด noise เป็นหมื่นจุดทำให้โปรแกรมค้าง ค่าที่แสดงทุกแถวยังเป็นค่าจริงเต็มความละเอียด ไม่ได้ประมาณ; ตั้ง 0 = วัดทุกชิ้นไม่จำกัด (อาจช้ามากถ้าวัตถุเยอะ)" },
      mmScaleParam(),
    ],
    apply(cv, src, p, ctx) {
      const mmF = numIn(p.mmPerPx, 0, 0, 100);
      if (isEmptyMat(src)) {
        if (ctx) {
          ctx.info = "ภาพว่าง";
          ctx.values = { kind: "table", ...mmTable("regionprops", mmF, REGION_COLUMNS_ALL, []), total: 0 };
        }
        return src.clone();
      }
      // (1) การ์ด binary input: gray จากกล้อง/แหล่งอื่นยังมีค่าระหว่าง 0..255 ไม่ใช่ binary ล้วน —
      // binarize ที่ 127 ก่อนเสมอ (ธรรมเนียมเดียวกับ fill_holes/connected_components ในไฟล์นี้)
      const bin = new cv.Mat();
      const contours = new cv.MatVector();
      const hier = new cv.Mat();
      let dst = null;
      try {
        cv.threshold(src, bin, 127, 255, cv.THRESH_BINARY);
        // (5) วัตถุที่แตะขอบภาพ: ใช้ RETR_EXTERNAL ตามปกติ (เหมือน find_contours/bounding_boxes) —
        // ไม่กรอง/ไม่แยกกรณีวัตถุที่แตะขอบทิ้ง ยังคงวัด/นับตามปกติ ตามพฤติกรรมเริ่มต้นของ regionprops
        // ใน MATLAB (นับรวมวัตถุที่ถูกตัดโดยขอบเฟรม แต่ผู้ใช้ต้องเข้าใจว่าค่าของวัตถุเหล่านั้นอาจไม่สมบูรณ์)
        cv.findContours(bin, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        dst = new cv.Mat();
        cv.cvtColor(src, dst, cv.COLOR_GRAY2RGB);

        // coerce ด้วย Number() ก่อนเช็ค finite — ตรงกับ finiteOr() ของ codegen.js เป๊ะ (debugger 18c fix-round พบว่า
        // ถ้า param มาเป็น string เช่น "0" จาก saved project เดิม Number.isFinite("0") เป็น false ทำให้ JS ตกไปใช้
        // ค่าเริ่มต้น แต่ Python แปลงเป็น 0 — ผลตรงข้ามกันระหว่างแอปกับโค้ด export)
        const minAreaVal = finiteOr(p.minArea, 50); // minArea:0 เป็นค่าที่ใช้งานได้จริง (ไม่กรองอะไรเลย) ต้องไม่ตกเป็น falsy fallback
        // maxObjects: 0 = ไม่จำกัด (ค่าที่ใช้งานได้จริง ต้องไม่ตกเป็น falsy fallback) — โปรเจกต์ที่ไม่มี key นี้
        // (undefined) ได้ค่าเริ่มต้น 500
        const cap = Math.max(0, Math.round(finiteOr(p.maxObjects, 500)));
        const n = contours.size();

        // ===== รอบที่ 1 (ถูก): คัดเฉพาะ contour ที่ผ่าน minArea + เก็บ sort key ที่คำนวณถูก ๆ =====
        // qa วัดจริง (1080p salt-and-pepper 10%, minArea=0): ~138k วัตถุ ดีไซน์เดิมที่คำนวณทุกอย่าง (moments,
        // arcLength, fitEllipse, putText) ให้ทุกชิ้น ใช้ ~1.6 วินาทีต่อเฟรม — ต้นทุนอยู่ที่งานต่อวัตถุ ไม่ใช่
        // findContours จึงแยกเป็น: (1) รอบถูกเก็บแค่ area/bbox_x/bbox_y/จุดบนสุดซ้ายสุด ด้วยการสแกน point
        // data ครั้งเดียว (regionScan) (2) sort ด้วย total order เต็ม (3) ตัดเหลือ cap ชิ้น (4) คำนวณแพง+วาด
        // เฉพาะชิ้นที่รอด — cx/cy (ต้องใช้ cv.moments ซึ่งแพง ~2.5µs/ชิ้น วัดแล้ว) คำนวณแบบ lazy เฉพาะตอนที่
        // สองชิ้นเสมอกันใน 3 key แรกเท่านั้น ได้ลำดับเดียวกันเป๊ะกับการคำนวณล่วงหน้าทุกชิ้น (comparator ไม่เคย
        // ดู cx/cy ถ้า 3 key แรกตัดสินได้แล้ว)
        const cands = [];
        let areaSumAll = 0;
        for (let i = 0; i < n; i++) {
          const c = contours.get(i); // MatVector.get() คืน Mat ต้อง .delete() เสมอ
          try {
            // (2) area = polygon area ของเส้นรอบรูป (เท่ากับ cv.contourArea เป๊ะ — ดู regionScan) ไม่ใช่นับพิกเซล
            // ตรงตามธรรมเนียม find_contours/bounding_boxes (ถ้าวัตถุมีรู พื้นที่จะนับรวมรูด้วยเพราะ RETR_EXTERNAL)
            const s = regionScan(c.data32S);
            if (s.area < minAreaVal) continue;
            cands.push({ i, area: s.area, bx: s.bx, by: s.by, tx: s.tx, cx: undefined, cy: undefined });
            areaSumAll += s.area;
          } finally {
            c.delete();
          }
        }

        // cx/cy แบบ lazy (ดูเหตุผลด้านบน) — m00=0 (วัตถุ 1px/พื้นที่ 0) fallback เป็นจุดกึ่งกลาง bounding box
        // กัน NaN จาก 0/0 (ยังวาดได้จริง ไม่ใช่ null)
        const setCentroid = (rec, c) => {
          const m = cv.moments(c); // plain object ไม่มี .delete()
          if (m.m00 === 0) {
            const r = cv.boundingRect(c);
            rec.cx = r.x + r.width / 2;
            rec.cy = r.y + r.height / 2;
          } else {
            rec.cx = m.m10 / m.m00;
            rec.cy = m.m01 / m.m00;
          }
        };
        const centroidOf = (rec) => {
          if (rec.cx === undefined) {
            const c = contours.get(rec.i);
            try { setCentroid(rec, c); } finally { c.delete(); }
          }
          return rec;
        };
        // (9) total order ตายตัวเหมือนกันทั้ง JS/Python — ดู compareRegionCandidates
        cands.sort((a, b) => compareRegionCandidates(a, b, centroidOf));

        const total = cands.length;
        const shown = cap === 0 ? cands : cands.slice(0, cap);

        // ===== รอบที่ 2 (แพง): วัดเต็ม + วาด เฉพาะชิ้นที่รอด cap =====
        const rows = [];
        for (let idx = 0; idx < shown.length; idx++) {
          const rec = shown[idx];
          let perimeter, r, orientation = null, shape;
          const c = contours.get(rec.i);
          try {
            // (6) perimeter: cv.arcLength(c, true) — เส้นรอบรูปแบบปิด ตามสเปก
            perimeter = cv.arcLength(c, true);
            r = cv.boundingRect(c); // plain {x,y,width,height} ไม่มี .delete()
            if (rec.cx === undefined) setCentroid(rec, c);
            // IP batch 3: คอลัมน์รูปร่างแบบนับพิกเซล (ตัวช่วยชุดเดียวกับ shape_filter) — ไม่เรียก boundingRect/moments/contourArea ซ้ำ
            const sv = regionShapeValues(regionPixelStats(cv, c, { x: r.x, y: r.y, w: r.width, h: r.height }, bin.data, src.data, src.cols, true), { w: r.width, h: r.height });
            shape = REGION_SHAPE_COLUMNS.map((k) => sv[k]);
            // (3) fitEllipse ต้องมีอย่างน้อย 5 จุดบนเส้นรอบรูป ไม่งั้น throw C++ exception ตรง ๆ → null
            if (c.rows >= 5) {
              const rot = cv.fitEllipse(c);
              // (8) ยืนยันเชิงประจักษ์ (วาดวงรีด้วย cv.ellipse ที่มุมรู้ค่าแล้ว fit กลับ): rot.angle เป็นมุมแกน
              // "width" ของ RotatedRect ซึ่งตั้งฉากกับแกนหลักจริง — แปลงเป็นมุมแกนหลักด้วย (rot.angle+90)%180
              // Convention ที่ได้ (ตรงกับที่ desc บอกผู้เรียน): 0° = แนวนอน, มุมเพิ่มตามเข็มนาฬิกาเพราะแกน y
              // ของภาพชี้ลง (90° = แนวตั้ง), ช่วง 0-180° (แกนไม่มีหัวท้าย 0°≡180°) — Python ต้องใช้สูตรเดียวกัน
              orientation = (rot.angle + 90) % 180;
            }
          } finally {
            c.delete();
          }
          // circularity = 4πA/P² — perimeter=0 กันหารศูนย์ แล้ว clamp [0,1] ตามสเปก (หมายเหตุ: area/perimeter
          // มาจาก polygon เดียวกัน isoperimetric inequality จึงรับประกัน ≤1 อยู่แล้ว clamp เป็น defensive ล้วน
          // — ตรวจแล้ว ดู worklog 18c)
          let circularity = perimeter === 0 ? 0 : (4 * Math.PI * rec.area) / (perimeter * perimeter);
          circularity = Math.max(0, Math.min(1, circularity));
          const index = idx + 1; // ตรงกับตัวเลขที่วาดบนภาพ ให้ผู้เรียนจับคู่แถวในตาราง <-> วัตถุที่เห็นได้
          rows.push([index, rec.area, rec.cx, rec.cy, perimeter, circularity, r.x, r.y, r.width, r.height, orientation, ...shape]);
          // วาด: กรอบล้อม (เขียว) + จุดศูนย์กลาง (แดง) + เลขลำดับ (เหลือง) — ctx.values ยังไม่มีที่แสดงบนจอ
          // (detail-panel.js ปิดไว้ตามสเปก §F) เลขที่วาดจึงเป็นทางเดียวที่ผู้เรียนจับคู่แถวกับวัตถุได้
          cv.rectangle(dst, new cv.Point(r.x, r.y), new cv.Point(r.x + r.width, r.y + r.height), new cv.Scalar(0, 255, 0), 2);
          cv.circle(dst, new cv.Point(Math.round(rec.cx), Math.round(rec.cy)), 3, new cv.Scalar(255, 0, 0), -1);
          cv.putText(dst, String(index), new cv.Point(r.x, Math.max(10, r.y - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, new cv.Scalar(255, 255, 0), 1);
        }

        if (ctx) {
          if (total === 0) {
            ctx.info = "ไม่พบวัตถุ (ลองลดค่าพื้นที่ขั้นต่ำ หรือตรวจสอบว่าภาพผ่าน Threshold ให้เป็นขาว-ดำมาก่อน)";
          } else {
            // พื้นที่เฉลี่ยคิดจาก "ทุกชิ้นที่พบ" (area รู้ครบจากรอบถูกแล้ว) ไม่ใช่เฉพาะที่แสดง — ตัวเลขนี้จึงถูกต้องเสมอ
            const avg = (areaSumAll / total).toFixed(0);
            ctx.info = shown.length < total
              ? `พบวัตถุ ${total} ชิ้น แสดง ${shown.length} ชิ้นที่ใหญ่ที่สุด (ปรับ 'แสดงวัตถุสูงสุด' ได้) — พื้นที่เฉลี่ย ${avg}px²`
              : `พบวัตถุ ${total} ชิ้น — พื้นที่เฉลี่ย ${avg}px²`;
            if (rows.length > 0 && rows.every((row) => row[REGION_MEAN_COL] === 255)) ctx.info += " · ภาพที่เข้ามาเป็นขาว-ดำอยู่แล้ว ความสว่างเฉลี่ยจึงเท่ากับค่าขาวเต็มทุกชิ้น";
          }
          // ctx.values: table แถวต่อวัตถุ (ทุกแถวยาวเท่า columns.length เป๊ะ) + total = จำนวนที่พบจริงทั้งหมด
          // (field เสริม additive — rows อาจน้อยกว่า total เมื่อ cap ทำงาน)
          if (mmF > 0) ctx.info += MM_SUFFIX(mmF);
          ctx.values = { kind: "table", ...mmTable("regionprops", mmF, REGION_COLUMNS_ALL, rows), total };
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        bin.delete();
        contours.delete();
        hier.delete();
      }
    },
  },

  watershed: {
    name: "Watershed",
    needs: "any",
    desc: "แยกวัตถุที่ 'ติดกัน/ซ้อนกัน' ออกจากกัน โดยมองความสว่างเป็นภูมิประเทศแล้ว 'เติมน้ำ' จากใจกลางแต่ละวัตถุ เส้นที่น้ำสองก้อนชนกันคือขอบแบ่ง — แก้ปัญหาที่ threshold ธรรมดาแล้ววัตถุติดกันเป็นก้อนเดียว (วาดเส้นแบ่งสีแดง)",
    params: [
      { key: "ratio", type: "number", label: "ความเข้มแยกวัตถุ", min: 0.2, max: 0.9, step: 0.05, default: 0.5, hint: "สูง = แยกมากขึ้น", desc: "สัดส่วนระยะจากใจกลางวัตถุที่ถือว่า 'แน่ใจว่าเป็นวัตถุ' ก่อนเริ่มเติมน้ำแยก" },
    ],
    apply(cv, src, p, ctx) {
      const rgb = new cv.Mat();
      if (src.channels() === 1) cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
      else src.copyTo(rgb);
      const gray = new cv.Mat();
      cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
      const thr = new cv.Mat();
      cv.threshold(gray, thr, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
      const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
      const anchor = new cv.Point(-1, -1);
      const opening = new cv.Mat();
      cv.morphologyEx(thr, opening, cv.MORPH_OPEN, kernel, anchor, 2);
      const sureBg = new cv.Mat();
      cv.dilate(opening, sureBg, kernel, anchor, 3);
      const dist = new cv.Mat();
      cv.distanceTransform(opening, dist, cv.DIST_L2, 5);
      const mm = cv.minMaxLoc(dist);
      const sureFg = new cv.Mat();
      cv.threshold(dist, sureFg, p.ratio * mm.maxVal, 255, cv.THRESH_BINARY);
      sureFg.convertTo(sureFg, cv.CV_8U);
      const unknown = new cv.Mat();
      cv.subtract(sureBg, sureFg, unknown);
      const markers = new cv.Mat();
      const nn = cv.connectedComponents(sureFg, markers, 8, cv.CV_32S);
      const one = new cv.Mat(markers.rows, markers.cols, markers.type(), new cv.Scalar(1));
      cv.add(markers, one, markers); // เลื่อน label ให้พื้นหลัง = 1 (ไม่ใช่ 0)
      markers.setTo(new cv.Scalar(0), unknown); // บริเวณไม่แน่ใจ = 0
      cv.watershed(rgb, markers);
      const dst = new cv.Mat();
      rgb.copyTo(dst);
      const neg = new cv.Mat(markers.rows, markers.cols, markers.type(), new cv.Scalar(-1));
      const border = new cv.Mat();
      cv.compare(markers, neg, border, cv.CMP_EQ); // เส้นแบ่ง markers == -1
      dst.setTo(new cv.Scalar(255, 0, 0), border);
      if (ctx) ctx.info = `แยกได้ ${Math.max(0, nn - 1)} วัตถุ`;
      rgb.delete();
      gray.delete();
      thr.delete();
      kernel.delete();
      opening.delete();
      sureBg.delete();
      dist.delete();
      sureFg.delete();
      unknown.delete();
      markers.delete();
      one.delete();
      neg.delete();
      border.delete();
      return dst;
    },
  },

  // ================= 7. Object Detection =================
  bounding_boxes: {
    name: "Bounding Boxes",
    needs: "gray",
    desc: "หาวัตถุ (บริเวณสีขาว) แล้ววาด 'กรอบสี่เหลี่ยม' ล้อมรอบแต่ละชิ้น พร้อมนับจำนวน — เป็นรูปแบบพื้นฐานของการตรวจจับวัตถุ (object detection) ต่อหลัง Threshold/Morphology",
    params: [
      { key: "minArea", type: "number", label: "พื้นที่ขั้นต่ำ (px)", min: 0, max: 5000, step: 50, default: 200, hint: "กรองจุดเล็ก ๆ ทิ้ง", desc: "วัตถุที่มีพื้นที่น้อยกว่านี้ (พิกเซล) จะไม่ถูกวาดกรอบล้อม" },
    ],
    apply(cv, src, p, ctx) {
      const contours = new cv.MatVector();
      const hier = new cv.Mat();
      cv.findContours(src, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      const dst = new cv.Mat();
      cv.cvtColor(src, dst, cv.COLOR_GRAY2RGB);
      let count = 0;
      for (let i = 0; i < contours.size(); i++) {
        const c = contours.get(i);
        const area = cv.contourArea(c);
        if (area < p.minArea) {
          c.delete();
          continue;
        }
        count++;
        const r = cv.boundingRect(c);
        c.delete();
        cv.rectangle(dst, new cv.Point(r.x, r.y), new cv.Point(r.x + r.width, r.y + r.height), new cv.Scalar(0, 255, 0), 2);
      }
      if (ctx) ctx.info = `ตรวจพบวัตถุ ${count} ชิ้น`;
      contours.delete();
      hier.delete();
      return dst;
    },
  },

  hough_circles: {
    name: "Hough Circles",
    needs: "gray",
    desc: "ค้นหา 'วงกลม' ในภาพด้วยการโหวตจากขอบภาพ (Hough transform) แล้ววาดวงที่พบทับ พร้อมนับจำนวน — ใช้ตรวจจับวัตถุทรงกลม เช่น เหรียญ ลูกบอล รู เจาะ",
    params: [
      { key: "param2", type: "number", label: "ความเข้มการตรวจ", min: 10, max: 150, step: 5, default: 40, hint: "① ปรับตัวนี้ก่อน — ต่ำ=เจอเยอะ (มีของปลอม), สูง=เฉพาะวงคมชัด" },
      { key: "minDist", type: "number", label: "ระยะห่างขั้นต่ำระหว่างวง", min: 5, max: 200, step: 5, default: 30, hint: "② กันนับวงเดียวซ้ำหลายครั้ง — ตั้งราว ๆ ขนาดวงจริงในภาพ" },
      { key: "minR", type: "number", label: "รัศมีเล็กสุด", min: 0, max: 200, step: 5, default: 0, hint: "③ ตัดวงเล็กเกินทิ้ง (0 = ไม่จำกัด)", desc: "วงกลมที่รัศมีเล็กกว่านี้ (พิกเซล) จะไม่ถูกนับ" },
      { key: "maxR", type: "number", label: "รัศมีใหญ่สุด", min: 0, max: 300, step: 5, default: 0, hint: "③ ตัดวงใหญ่เกินทิ้ง (0 = ไม่จำกัด)", desc: "วงกลมที่รัศมีใหญ่กว่านี้ (พิกเซล) จะไม่ถูกนับ" },
      { key: "dp", type: "number", label: "ความละเอียด (dp)", min: 1, max: 3, step: 0.5, default: 1, hint: "ปกติ 1 กำลังดี ไม่ต้องแตะบ่อย" },
    ],
    apply(cv, src, p, ctx) {
      // Hough ที่ Full HD และ maxR=0 สร้าง accumulator ใหญ่มากจน worker ดูเหมือนค้าง
      // ตรวจบนภาพไม่เกิน 640px แล้วแปลงพิกัดกลับไปยังภาพจริง ค่าที่ผู้ใช้ตั้งยังเป็น px ของภาพจริง
      const maxAnalysisSide = 640;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      const resized = new cv.Mat();
      const blur = new cv.Mat();
      const circles = new cv.Mat();
      let dst = null;
      try {
        const analysis = scale < 1 ? resized : src;
        if (scale < 1) {
          // B4/review: สัดส่วนสุดโต่งทำให้ Math.round ได้ 0 → ใช้ analysisScale (อย่างน้อย 1 px)
          const sz = analysisScale(src.cols, src.rows, maxAnalysisSide);
          cv.resize(src, resized, new cv.Size(sz.w, sz.h), 0, 0, cv.INTER_AREA);
        }
        cv.medianBlur(analysis, blur, 5); // ลด noise ก่อน กันตรวจเจอวงปลอม

        // ภาพ texture/noise สูงทำให้ Hough สร้าง candidate จำนวนมหาศาล
        // และอาจกินเวลาหลายวินาทีแม้ภาพเล็ก จึงหยุดเร็วพร้อมคำแนะนำให้ Blur ก่อน
        const edges = new cv.Mat();
        cv.Canny(blur, edges, 100, 200);
        const edgeRatio = cv.countNonZero(edges) / Math.max(1, edges.rows * edges.cols);
        const edgeContours = new cv.MatVector();
        const edgeHierarchy = new cv.Mat();
        cv.findContours(edges, edgeContours, edgeHierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
        const edgeGroups = edgeContours.size();
        edgeContours.delete();
        edgeHierarchy.delete();
        edges.delete();
        if (edgeRatio > 0.18 || edgeGroups > 180) {
          dst = new cv.Mat();
          cv.cvtColor(src, dst, cv.COLOR_GRAY2RGB);
          if (ctx) ctx.info = "ภาพมีขอบรบกวนมาก — เพิ่ม Blur ก่อน Hough Circles";
          return dst;
        }

        const minR = Math.round(p.minR * scale);
        let maxR = Math.round(p.maxR * scale);
        if (maxR > 0 && maxR < minR) maxR = 0;
        cv.HoughCircles(
          blur, circles, cv.HOUGH_GRADIENT, p.dp,
          Math.max(1, p.minDist * scale), 100, p.param2, minR, maxR,
        );

        dst = new cv.Mat();
        cv.cvtColor(src, dst, cv.COLOR_GRAY2RGB);
        const count = circles.cols;
        const invScale = 1 / scale;
        for (let i = 0; i < count; i++) {
          const x = circles.data32F[i * 3] * invScale;
          const y = circles.data32F[i * 3 + 1] * invScale;
          const r = circles.data32F[i * 3 + 2] * invScale;
          const center = new cv.Point(x, y);
          cv.circle(dst, center, r, new cv.Scalar(0, 255, 0), 2);
          cv.circle(dst, center, 2, new cv.Scalar(255, 0, 0), 3);
        }
        if (ctx) ctx.info = `พบวงกลม ${count} วง`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        resized.delete();
        blur.delete();
        circles.delete();
      }
    },
  },

  hough_lines: {
    name: "Hough Lines",
    needs: "gray",
    desc: "ค้นหา 'เส้นตรง' ในภาพด้วยการโหวตจากขอบภาพ (Hough transform แบบ Probabilistic) — ใช้ตรวจแนวขอบ/แนวเส้นที่แข็ง เช่น ขอบชิ้นงาน แนวราง เส้นบนกระดาน วาดเส้นที่พบทับภาพเดิมพร้อมนับจำนวน",
    params: [
      { key: "t1", type: "number", label: "เกณฑ์ Canny ต่ำ", min: 0, max: 255, step: 1, default: 50, hint: "① ใช้หาขอบก่อนหาเส้น (เหมือน Canny)", desc: "ขอบอ่อนที่จะถูกเก็บไว้ถ้าต่อเนื่องกับขอบเข้ม — ใช้เตรียมภาพก่อนหาเส้นตรง" },
      { key: "t2", type: "number", label: "เกณฑ์ Canny สูง", min: 0, max: 255, step: 1, default: 150, hint: "② ขอบเข้มที่ถือว่าเป็นขอบแน่นอน" },
      { key: "threshold", type: "number", label: "คะแนนโหวตขั้นต่ำ", min: 10, max: 300, step: 5, default: 80, hint: "③ ปรับตัวนี้ก่อน — ต่ำ=เจอเส้นเยอะ(มีของปลอม), สูง=เฉพาะเส้นชัด" },
      { key: "minLineLength", type: "number", label: "ความยาวเส้นขั้นต่ำ (px)", min: 5, max: 500, step: 5, default: 40, desc: "เส้นที่สั้นกว่านี้จะถูกทิ้ง — กันจับเส้นปลอมสั้น ๆ" },
      { key: "maxLineGap", type: "number", label: "ช่องว่างที่ยอมรวมเป็นเส้นเดียว (px)", min: 0, max: 100, step: 1, default: 10, desc: "จุดขอบที่ขาดกันไม่เกินนี้ จะถูกนับเป็นเส้นเดียวกัน" },
    ],
    apply(cv, src, p, ctx) {
      // Downscale for analysis on very large frames, same pattern as hough_circles
      // เพื่อไม่ให้ worker ดูเหมือนค้างเวลาภาพ Full HD
      const maxAnalysisSide = 640;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      const analysis = new cv.Mat();
      const edges = new cv.Mat();
      const lines = new cv.Mat();
      let dst = null;
      try {
        if (scale < 1) {
          // B4/review: สัดส่วนสุดโต่งทำให้ Math.round ได้ 0 → ใช้ analysisScale (อย่างน้อย 1 px)
          const sz = analysisScale(src.cols, src.rows, maxAnalysisSide);
          cv.resize(src, analysis, new cv.Size(sz.w, sz.h), 0, 0, cv.INTER_AREA);
        } else {
          src.copyTo(analysis);
        }
        cv.Canny(analysis, edges, p.t1, p.t2);
        // คะแนนโหวตขั้นต่ำ (threshold) เป็นจำนวนพิกเซลขอบที่รองรับเส้น ซึ่งแปรผันตามความยาวเส้นเป็นพิกเซล
        // เช่นเดียวกับ minLineLength/maxLineGap — ถ้าไม่คูณ scale ตอนวิเคราะห์ภาพย่อ (>640px) ผลที่ได้
        // ในแอปจะต่างจากโค้ด Python ที่ generate ไว้ (รันเต็มความละเอียดเสมอ ไม่ย่อภาพ) จึงต้องคูณ scale ด้วย
        const scaledThreshold = Math.max(1, Math.round(p.threshold * scale));
        cv.HoughLinesP(edges, lines, 1, Math.PI / 180, scaledThreshold, p.minLineLength * scale, p.maxLineGap * scale);

        dst = new cv.Mat();
        cv.cvtColor(src, dst, cv.COLOR_GRAY2RGB);
        // HoughLinesP บน build นี้คืน Mat รูปร่าง Nx1 แบบ 4-channel (CV_32SC4) — ยืนยันด้วย
        // Node smoke test แล้ว (lines.rows=N, cols=1, channels=4) จึงอ่าน data32S แบบ flat index i*4 ได้ตรง
        const n = lines.rows;
        const invScale = 1 / scale;
        for (let i = 0; i < n; i++) {
          const x1 = lines.data32S[i * 4] * invScale;
          const y1 = lines.data32S[i * 4 + 1] * invScale;
          const x2 = lines.data32S[i * 4 + 2] * invScale;
          const y2 = lines.data32S[i * 4 + 3] * invScale;
          cv.line(dst, new cv.Point(x1, y1), new cv.Point(x2, y2), new cv.Scalar(0, 255, 0), 2);
        }
        if (ctx) ctx.info = `พบเส้นตรง ${n} เส้น`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        analysis.delete();
        edges.delete();
        lines.delete();
      }
    },
  },

  harris_corners: {
    name: "Harris Corner Detection",
    needs: "gray",
    desc: "หา 'มุม' (corner) ในภาพด้วยวิธี Harris — ดูว่าความสว่างรอบแต่ละจุดเปลี่ยนมากแค่ไหนในทุกทิศทาง จุดที่เปลี่ยนมากทุกทิศทาง = มุม ใช้หาจุดเด่นสำหรับ tracking/matching ภาพ",
    params: [
      { key: "blockSize", type: "number", label: "ขนาดพื้นที่รอบจุด", min: 2, max: 15, step: 1, default: 2, hint: "① พื้นที่ (px) ที่ใช้ดูรอบแต่ละพิกเซล" },
      { key: "ksize", type: "number", label: "ขนาด Sobel kernel", min: 1, max: 7, step: 2, default: 3, hint: "② ต้องเป็นเลขคี่เสมอ — ปรับความคมของขอบที่ใช้วิเคราะห์" },
      { key: "k", type: "number", label: "ค่า k (Harris)", min: 0.01, max: 0.3, step: 0.01, default: 0.04, hint: "③ ปกติ 0.04–0.06" },
      { key: "thresh", type: "number", label: "ความไวจับมุม (%)", min: 1, max: 100, step: 1, default: 1, hint: "④ เกณฑ์ = กี่ % ของค่าตอบสนองสูงสุดในภาพ (1% = MinQuality 0.01 ค่าเริ่มต้นของ MATLAB) · ต่ำ=เจอมุมเยอะ(มีของปลอมปน), สูง=เฉพาะมุมชัด" },
    ],
    apply(cv, src, p, ctx) {
      // ภาพใหญ่วิ่ง cornerHarris เต็มความละเอียดจะช้า — วิเคราะห์บนภาพย่อไม่เกิน 480px แล้วคูณพิกัดกลับ
      const maxAnalysisSide = 480;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      const analysis = new cv.Mat();
      const resp = new cv.Mat();
      const dil = new cv.Mat();
      const kern = cv.Mat.ones(3, 3, cv.CV_8U);
      let out = null;
      try {
        if (scale < 1) {
          // B4: สัดส่วนสุดโต่ง (~1000:1) ทำให้ Math.round ได้ 0 → ต้องมีอย่างน้อย 1 px (เหมือน orb/Python)
          const sz = analysisScale(src.cols, src.rows, maxAnalysisSide);
          cv.resize(src, analysis, new cv.Size(sz.w, sz.h), 0, 0, cv.INTER_AREA);
        } else {
          src.copyTo(analysis);
        }
        const ksize = p.ksize % 2 === 0 ? p.ksize + 1 : p.ksize; // Sobel kernel ต้องเป็นเลขคี่
        cv.cornerHarris(analysis, resp, p.blockSize, ksize, p.k, cv.BORDER_DEFAULT);
        // เกณฑ์ = thresh% ของค่า R สูงสุด (แบบ MinQuality ของ MATLAB corner/detectHarrisFeatures) — เดิม normalize
        // min-max ก่อน ซึ่ง R ของขอบติดลบ ทำให้พื้นเรียบ (R≈0) กลายเป็น ~70–85/255 และผ่านเกณฑ์ต่ำ ๆ ทั้งภาพ
        // แล้วเก็บเฉพาะจุดที่เป็นค่าสูงสุดในหน้าต่าง 3x3 (non-max suppression) → 1 จุดต่อ 1 มุม ไม่ใช่นับพิกเซล
        cv.dilate(resp, dil, kern);
        const maxR = cv.minMaxLoc(resp).maxVal;
        const R = resp.data32F;
        const D = dil.data32F;
        const cols = resp.cols;
        const taken = new Uint8Array(R.length); // ค่าเท่ากันติดกัน (plateau) เก็บแค่ตัวแรกตามลำดับแถว
        const points = [];
        if (maxR > 0) {
          const thr = (p.thresh / 100) * maxR;
          for (let idx = 0; idx < R.length; idx++) {
            if (R[idx] < thr || R[idx] < D[idx] || taken[idx]) continue; // >= เกณฑ์ → 100% ยังได้จุดสูงสุด
            const x = idx % cols, y = (idx / cols) | 0;
            points.push(x, y);
            if (points.length / 2 > 3000) break; // threshold ต่ำเกินไป → เจอเป็นพัน หยุดก่อนค้าง
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                const xx = x + dx, yy = y + dy;
                if (xx >= 0 && xx < cols && yy >= 0 && yy < resp.rows) taken[yy * cols + xx] = 1;
              }
            }
          }
        }

        out = new cv.Mat();
        cv.cvtColor(src, out, cv.COLOR_GRAY2RGB);
        const n = points.length / 2;
        if (n > 3000) {
          if (ctx) ctx.info = `จุดมุมเยอะเกิน (>3000) — เพิ่มค่า "ความไวจับมุม" ขึ้น`;
          return out;
        }
        const invScale = 1 / scale;
        for (let i = 0; i < n; i++) {
          const x = points[i * 2] * invScale;
          const y = points[i * 2 + 1] * invScale;
          cv.circle(out, new cv.Point(x, y), 3, new cv.Scalar(0, 255, 0), 1);
        }
        if (ctx) ctx.info = `พบมุม ${n} จุด`;
        return out;
      } catch (err) {
        if (out) out.delete();
        throw err;
      } finally {
        analysis.delete();
        resp.delete();
        dil.delete();
        kern.delete();
      }
    },
  },

  shi_tomasi_corners: {
    name: "Shi-Tomasi Corner Detection",
    needs: "gray",
    desc: "หา 'มุมเด่น' ที่ดีที่สุดในภาพด้วยวิธี Shi-Tomasi (goodFeaturesToTrack) — เลือกเฉพาะจุดมุมคุณภาพดีตามจำนวนที่ตั้งไว้ นิยมใช้เป็นจุดเริ่มต้นสำหรับ optical flow/tracking",
    params: [
      { key: "maxCorners", type: "number", label: "จำนวนมุมสูงสุด", min: 1, max: 500, step: 1, default: 100, hint: "① จำกัดจำนวนจุดมุมที่ดีที่สุดที่จะแสดง" },
      { key: "quality", type: "number", label: "คุณภาพขั้นต่ำ (%)", min: 1, max: 100, step: 1, default: 1, hint: "② % เทียบกับมุมที่ดีที่สุดในภาพ — ต่ำ=เจอเยอะ" },
      { key: "minDist", type: "number", label: "ระยะห่างขั้นต่ำ", min: 1, max: 100, step: 1, default: 10, hint: "③ กันจุดมุมซ้อนกันใกล้เกินไป (px)" },
      { key: "blockSize", type: "number", label: "ขนาดพื้นที่รอบจุด", min: 2, max: 15, step: 1, default: 3 },
      { key: "useHarris", type: "boolean", label: "ใช้สูตร Harris", default: false, hint: "ปิด=Shi-Tomasi ปกติ, เปิด=ใช้สูตร Harris แทน" },
    ],
    apply(cv, src, p, ctx) {
      // goodFeaturesToTrack คำนวณ eigenvalue ทั้งภาพทุกเฟรม ไม่ขึ้นกับ maxCorners —
      // ภาพ Full HD จะหนักกว่า op อื่น จึงย่อวิเคราะห์ไม่เกิน 480px แล้วคูณพิกัดกลับ เหมือน harris_corners
      const maxAnalysisSide = 480;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      const analysis = new cv.Mat();
      const mask = new cv.Mat();
      const corners = new cv.Mat();
      let out = null;
      try {
        if (scale < 1) {
          // B4: สัดส่วนสุดโต่ง (~1000:1) ทำให้ Math.round ได้ 0 → ต้องมีอย่างน้อย 1 px (เหมือน orb/Python)
          const sz = analysisScale(src.cols, src.rows, maxAnalysisSide);
          cv.resize(src, analysis, new cv.Size(sz.w, sz.h), 0, 0, cv.INTER_AREA);
        } else {
          src.copyTo(analysis);
        }
        cv.goodFeaturesToTrack(
          analysis, corners, p.maxCorners, Math.max(0.001, p.quality / 100), p.minDist * scale,
          mask, p.blockSize, boolOr(p.useHarris, false), 0.04
        );
        out = new cv.Mat();
        cv.cvtColor(src, out, cv.COLOR_GRAY2RGB);
        const n = corners.rows;
        const invScale = 1 / scale;
        for (let i = 0; i < n; i++) {
          const x = corners.data32F[i * 2] * invScale;
          const y = corners.data32F[i * 2 + 1] * invScale;
          cv.circle(out, new cv.Point(x, y), 4, new cv.Scalar(0, 255, 0), 2);
        }
        if (ctx) ctx.info = `พบมุมเด่น ${n} จุด`;
        return out;
      } catch (err) {
        if (out) out.delete();
        throw err;
      } finally {
        analysis.delete();
        mask.delete();
        corners.delete();
      }
    },
  },

  orb_keypoints: {
    name: "ORB Keypoints",
    needs: "gray",
    desc: "หาจุดเด่น (keypoint) ด้วยวิธี ORB — รวมสองไอเดีย: หามุมแบบเร็ว (คล้าย FAST) แล้ววัดทิศทางของแต่ละจุดเพิ่ม (ต่างจาก Harris/Shi-Tomasi ที่หาแค่ตำแหน่ง ไม่มีทิศทาง) ทำให้จับคู่จุดเดียวกันได้แม้ภาพหมุน — เป็นจุดเริ่มต้นของงาน image matching/panorama (การจับคู่ 2 ภาพยังไม่ทำในรอบนี้ ต้องรอ Tier 2 ที่รับภาพได้ 2 สาย)",
    params: [
      { key: "maxFeatures", type: "number", label: "จำนวนจุดสูงสุด", min: 10, max: 1000, step: 10, default: 300, desc: "จำกัดจำนวนจุดเด่นที่แสดงสูงสุด — ORB เลือกจุดที่ 'เด่นที่สุด' (คะแนน response สูงสุด) ไว้ก่อนเสมอ ไม่ใช่สุ่มตัด" },
    ],
    apply(cv, src, p, ctx) {
      // ย่อภาพวิเคราะห์ไม่เกิน 480px แล้วคูณพิกัดกลับ ตรงกับ harris_corners/shi_tomasi_corners
      const maxAnalysisSide = 480;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      // clamp ทั้งขอบล่าง (0/ค่าติดลบไม่มีความหมาย) และขอบบน (native cv.ORB(nfeatures) ที่ใหญ่ผิดปกติ เช่น
      // >=1e9 จาก project ที่แก้ไฟล์ตรง ๆ ทำให้ constructor throw allocation error — UI ปกติ max แค่ 1000 อยู่แล้ว)
      const maxFeatures = Math.min(1000, Math.max(1, Math.round(finiteOr(p.maxFeatures, 300))));
      let analysis = null, orb = null, kpv = null, mask = null, out = null;
      try {
        analysis = new cv.Mat();
        if (scale < 1) {
          const w = Math.max(1, Math.round(src.cols * scale));
          const h = Math.max(1, Math.round(src.rows * scale));
          cv.resize(src, analysis, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
        } else {
          src.copyTo(analysis);
        }

        out = new cv.Mat();
        cv.cvtColor(src, out, cv.COLOR_GRAY2RGB);
        if (tooSmallForKeypoints(analysis)) {
          if (ctx) ctx.info = KEYPOINTS_TOO_SMALL_INFO;
          return out;
        }

        orb = new cv.ORB(maxFeatures);
        kpv = new cv.KeyPointVector();
        mask = new cv.Mat();
        orb.detect(analysis, kpv, mask);

        // ยืนยันแล้วเชิงประจักษ์ (Node, ตรงกับ cv2 5.0.0 เป๊ะทั้งสองฝั่ง) ว่า nfeatures ของ ORB ไม่ใช่ hard
        // cap จริง — บนภาพที่มีลวดลาย/ขอบชัดเจน (ต่างจากภาพ noise สุ่มที่ทดสอบไว้ในสเปกตอนแรก) จำนวนจุดที่
        // ได้อาจเกิน maxFeatures ไปมากก็ได้ (เช่น maxFeatures=20 แต่ได้ 74 จุด) — จึงต้องตัดด้วย response
        // ranking ซ้ำอีกชั้นแบบเดียวกับ AKAZE/FAST (ผ่าน rankKeypoints ที่มี tie-break ตายตัว) เพื่อให้คำว่า
        // "จำนวนจุดสูงสุด" ในหน้า UI เป็นจริงเสมอ ไม่ใช่แค่ "ค่าที่ส่งเข้า constructor" ที่ไม่ได้บังคับผลจริง
        const { total, kept } = rankKeypoints(kpv, maxFeatures);
        if (total === 0) {
          if (ctx) ctx.info = "ไม่พบจุดเด่นในภาพนี้ — ลองภาพที่มีลวดลาย/ขอบชัดเจนกว่านี้";
          return out;
        }

        const invScale = 1 / scale;
        for (const kp of kept) {
          const x = kp.x * invScale, y = kp.y * invScale;
          const r = Math.max(2, (kp.size / 2) * invScale);
          cv.circle(out, new cv.Point(x, y), Math.round(r), new cv.Scalar(0, 255, 0), 1);
          if (kp.angle !== -1) {
            const rad = (kp.angle * Math.PI) / 180;
            const ex = x + r * Math.cos(rad), ey = y + r * Math.sin(rad);
            cv.line(out, new cv.Point(x, y), new cv.Point(ex, ey), new cv.Scalar(0, 255, 0), 1);
          }
        }
        if (ctx) ctx.info = total > maxFeatures
          ? `พบจุดเด่น ${total} จุด — แสดง ${maxFeatures} จุดที่คะแนนดีที่สุด (ORB)`
          : `พบจุดเด่น ${total} จุด (ORB)`;
        return out;
      } catch (err) {
        if (out) out.delete();
        throw err;
      } finally {
        if (analysis) analysis.delete();
        if (orb) orb.delete();
        if (kpv) kpv.delete();
        if (mask) mask.delete();
      }
    },
  },

  akaze_keypoints: {
    name: "AKAZE Keypoints",
    needs: "gray",
    desc: "หาจุดเด่นด้วยวิธี AKAZE — คล้าย ORB (มีทิศทางด้วย) แต่วิเคราะห์ภาพในหลายระดับความละเอียดแบบไม่เชิงเส้น (nonlinear scale space) ทำให้จับรายละเอียดได้แม่นกว่าในบางกรณี แต่ช้ากว่า ORB — build ของโปรแกรมนี้ไม่มีช่องปรับความไวของ AKAZE โดยตรง จึงปรับได้แค่ 'จำนวนจุดสูงสุดที่แสดง' ด้านล่าง (เลือกจุดที่คะแนนดีที่สุดไว้ก่อน)",
    params: [
      { key: "maxPoints", type: "number", label: "จำนวนจุดสูงสุดที่แสดง", min: 10, max: 1000, step: 10, default: 300, desc: "AKAZE ไม่มีช่องปรับความไวในตัว (ข้อจำกัดของ build นี้) — ใช้ค่านี้ตัดเอาเฉพาะจุดที่คะแนนดีที่สุด (response สูงสุด) ไว้แสดง" },
    ],
    apply(cv, src, p, ctx) {
      const maxAnalysisSide = 480;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      const maxPoints = Math.max(1, Math.round(finiteOr(p.maxPoints, 300)));
      let analysis = null, akaze = null, kpv = null, mask = null, out = null;
      try {
        analysis = new cv.Mat();
        if (scale < 1) {
          const w = Math.max(1, Math.round(src.cols * scale));
          const h = Math.max(1, Math.round(src.rows * scale));
          cv.resize(src, analysis, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
        } else {
          src.copyTo(analysis);
        }

        out = new cv.Mat();
        cv.cvtColor(src, out, cv.COLOR_GRAY2RGB);
        if (tooSmallForKeypoints(analysis)) {
          if (ctx) ctx.info = KEYPOINTS_TOO_SMALL_INFO;
          return out;
        }

        akaze = new cv.AKAZE(); // build นี้รับ arg ตัวสร้างไม่ได้เลย (unbound descriptor-type enum) — ต้องเรียกแบบไม่มี arg เท่านั้น
        kpv = new cv.KeyPointVector();
        mask = new cv.Mat();
        akaze.detect(analysis, kpv, mask);

        // จัดอันดับตาม response + tie-break ตายตัว (ดู rankKeypoints ด้านบน) แล้วตัดเอาแค่ maxPoints ตัวที่ดีที่สุด
        const { total, kept } = rankKeypoints(kpv, maxPoints);
        if (total === 0) {
          if (ctx) ctx.info = "ไม่พบจุดเด่นในภาพนี้ — ลองภาพที่มีลวดลาย/ขอบชัดเจนกว่านี้";
          return out;
        }

        const invScale = 1 / scale;
        for (const kp of kept) {
          const x = kp.x * invScale, y = kp.y * invScale;
          const r = Math.max(2, (kp.size / 2) * invScale);
          cv.circle(out, new cv.Point(x, y), Math.round(r), new cv.Scalar(0, 200, 255), 1);
          if (kp.angle !== -1) {
            const rad = (kp.angle * Math.PI) / 180;
            const ex = x + r * Math.cos(rad), ey = y + r * Math.sin(rad);
            cv.line(out, new cv.Point(x, y), new cv.Point(ex, ey), new cv.Scalar(0, 200, 255), 1);
          }
        }
        if (ctx) ctx.info = total > maxPoints
          ? `พบจุดเด่น ${total} จุด — แสดง ${maxPoints} จุดที่คะแนนดีที่สุด (AKAZE)`
          : `พบจุดเด่น ${total} จุด (AKAZE)`;
        return out;
      } catch (err) {
        if (out) out.delete();
        throw err;
      } finally {
        if (analysis) analysis.delete();
        if (akaze) akaze.delete();
        if (kpv) kpv.delete();
        if (mask) mask.delete();
      }
    },
  },

  fast_keypoints: {
    name: "FAST Keypoints",
    needs: "gray",
    desc: "หาจุดเด่นแบบเร็วที่สุดในกลุ่มนี้ด้วยวิธี FAST — ดูวงแหวนพิกเซลรอบจุดว่ามีช่วงที่สว่าง/มืดกว่ากลางต่อเนื่องกันยาวพอไหม ไม่คำนวณทิศทางหรือขนาด (จุดจะเป็นวงกลมขนาดเท่ากันหมด ไม่มีเส้นบอกทิศทาง) เหมาะเป็นขั้นตอนแรกที่ต้องเร็วมาก ๆ ก่อนนำไปคำนวณอย่างอื่นต่อ (เช่น ORB ก็ใช้ FAST เป็นแกนหาจุดอยู่ข้างในเหมือนกัน)",
    params: [
      { key: "threshold", type: "number", label: "ความไว (threshold)", min: 1, max: 100, step: 1, default: 10, desc: "ต่ำ = เจอจุดเยอะ (ไวมาก, มีของปลอมปน), สูง = เฉพาะจุดที่ต่างจากรอบข้างชัดเจนมาก" },
      { key: "maxPoints", type: "number", label: "จำนวนจุดสูงสุดที่แสดง", min: 10, max: 2000, step: 10, default: 300, desc: "FAST ไม่จำกัดจำนวนจุดในตัวเอง (อาจเจอเป็นหมื่นจุดถ้า threshold ต่ำ) ใช้ค่านี้ตัดเอาเฉพาะจุดที่คะแนนดีที่สุดไว้แสดง" },
    ],
    apply(cv, src, p, ctx) {
      const maxAnalysisSide = 480;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      const threshold = Math.max(1, Math.round(finiteOr(p.threshold, 10)));
      const maxPoints = Math.max(1, Math.round(finiteOr(p.maxPoints, 300)));
      let analysis = null, fast = null, kpv = null, mask = null, out = null;
      try {
        analysis = new cv.Mat();
        if (scale < 1) {
          const w = Math.max(1, Math.round(src.cols * scale));
          const h = Math.max(1, Math.round(src.rows * scale));
          cv.resize(src, analysis, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
        } else {
          src.copyTo(analysis);
        }

        out = new cv.Mat();
        cv.cvtColor(src, out, cv.COLOR_GRAY2RGB);
        if (tooSmallForKeypoints(analysis)) {
          if (ctx) ctx.info = KEYPOINTS_TOO_SMALL_INFO;
          return out;
        }

        fast = new cv.FastFeatureDetector(threshold);
        kpv = new cv.KeyPointVector();
        mask = new cv.Mat();
        fast.detect(analysis, kpv, mask);

        const { total, kept } = rankKeypoints(kpv, maxPoints);
        if (total === 0) {
          if (ctx) ctx.info = "ไม่พบจุดเด่นในภาพนี้ — ลองลดค่า 'ความไว' ลง หรือใช้ภาพที่มีลวดลายชัดเจนกว่านี้";
          return out;
        }

        const invScale = 1 / scale;
        for (const kp of kept) {
          const x = kp.x * invScale, y = kp.y * invScale;
          // FAST คืน size คงที่เล็ก ๆ เสมอ แต่ angle=-1 เสมอ (ไม่มีทิศทาง) — วาดวงกลมเปล่า ไม่มีเส้นบอกทิศทาง
          const r = Math.max(2, (kp.size / 2) * invScale);
          cv.circle(out, new cv.Point(x, y), Math.round(r), new cv.Scalar(255, 128, 0), 1);
        }
        if (ctx) ctx.info = total > maxPoints
          ? `พบจุดเด่น ${total} จุด — แสดง ${maxPoints} จุดที่คะแนนดีที่สุด (FAST)`
          : `พบจุดเด่น ${total} จุด (FAST)`;
        return out;
      } catch (err) {
        if (out) out.delete();
        throw err;
      } finally {
        if (analysis) analysis.delete();
        if (fast) fast.delete();
        if (kpv) kpv.delete();
        if (mask) mask.delete();
      }
    },
  },

  grabcut: {
    name: "GrabCut",
    needs: "any",
    desc: "แยก 'วัตถุหลัก' ออกจากพื้นหลังอัตโนมัติ โดยเดาว่าวัตถุอยู่กลางภาพ แล้วใช้สถิติสีของในกรอบ vs นอกกรอบตัดพื้นหลังออก — ตัวอย่างการทำ object segmentation ที่ตัดฉากหลังทิ้ง (ช้ากว่าตัวอื่น เหมาะใช้กับภาพนิ่ง)",
    params: [
      { key: "margin", type: "number", label: "ขอบที่ถือเป็นพื้นหลัง (%)", min: 5, max: 40, step: 5, default: 12, desc: "ระยะจากขอบภาพเข้ามาที่ระบบเดาว่าเป็นพื้นหลังแน่นอน ใช้เริ่มวิเคราะห์" },
      { key: "iter", type: "number", label: "จำนวนรอบวิเคราะห์", min: 1, max: 5, step: 1, default: 2, hint: "มาก = แม่นขึ้น/ช้าขึ้น", desc: "จำนวนรอบที่ปรับปรุงการแยกวัตถุ-พื้นหลังซ้ำ" },
      { key: "output", type: "select", label: "แสดงผลเป็น", options: [
        { value: "extract", label: "เฉพาะวัตถุ (ตัดพื้นหลัง)" },
        { value: "mask", label: "Mask ขาวดำ" },
      ], default: "extract" },
    ],
    apply(cv, src, p) {
      const rgb = new cv.Mat();
      if (src.channels() === 1) cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
      else src.copyTo(rgb);
      const mask = new cv.Mat();
      const bgd = new cv.Mat();
      const fgd = new cv.Mat();
      const mx = Math.round((rgb.cols * p.margin) / 100);
      const my = Math.round((rgb.rows * p.margin) / 100);
      const rect = new cv.Rect(mx, my, Math.max(1, rgb.cols - 2 * mx), Math.max(1, rgb.rows - 2 * my));
      cv.grabCut(rgb, mask, rect, bgd, fgd, Math.round(p.iter), cv.GC_INIT_WITH_RECT);
      // foreground = ค่า 1 (GC_FGD) หรือ 3 (GC_PR_FGD) → เลขคี่ → (mask & 1) > 0
      const one = new cv.Mat(mask.rows, mask.cols, mask.type(), new cv.Scalar(1));
      const odd = new cv.Mat();
      cv.bitwise_and(mask, one, odd);
      const zero = new cv.Mat(mask.rows, mask.cols, mask.type(), new cv.Scalar(0));
      const fgMask = new cv.Mat();
      cv.compare(odd, zero, fgMask, cv.CMP_GT); // > 0 → 255 (foreground)
      let dst;
      if (p.output === "mask") {
        dst = new cv.Mat();
        fgMask.copyTo(dst);
      } else {
        dst = new cv.Mat();
        cv.bitwise_and(rgb, rgb, dst, fgMask); // เก็บเฉพาะวัตถุ พื้นหลังดำ
      }
      rgb.delete();
      mask.delete();
      bgd.delete();
      fgd.delete();
      one.delete();
      odd.delete();
      zero.delete();
      fgMask.delete();
      return dst;
    },
  },

  // Region Growing / Magic Wand (2026-09-30) — segmentation ตัวที่ 3 ของกลุ่ม ต่อจาก watershed/grabcut
  region_growing: {
    name: "Region Growing (ขยายพื้นที่จากจุดเริ่ม)",
    needs: "gray",
    desc: "เลือก 'จุดเริ่ม' หนึ่งจุดบนภาพ แล้วโปรแกรมจะลามออกไปยังพิกเซลข้างเคียงที่ความสว่างใกล้เคียงกับจุดเริ่ม (ไม่เกินค่าความต่างที่ตั้งไว้) หยุดเองเมื่อเจอขอบที่ความสว่างเปลี่ยนแรง — เป็นวิธีแบ่งส่วนภาพที่เข้าใจง่ายที่สุด และเป็นตัวเดียวกับเครื่องมือ 'ไม้กายสิทธิ์ (Magic Wand)' ในโปรแกรมแต่งภาพ · ในงานตรวจสอบใช้เลือกเฉพาะชิ้นงานหรือเฉพาะรอยเปื้อนที่รู้ตำแหน่งคร่าว ๆ อยู่แล้ว โดยไม่ต้องตั้ง threshold ทั้งภาพ · ค่าความต่างยิ่งมาก พื้นที่ยิ่งลามกว้าง ถ้าลามทั้งภาพแปลว่าตั้งไว้สูงเกินไป ส่วนถ้าได้แค่จุดเดียวหรือภาพดำสนิท แปลว่าตั้งไว้ต่ำเกินไป หรือภาพมีจุดรบกวน (noise) มากจนพิกเซลที่ติดกันกระโดดเกินเกณฑ์ — กรณีหลังให้ต่อ Gaussian Blur ไว้ก่อนหน้าแล้วค่อยเข้าตัวนี้ จะกลับมาใช้ค่าเดิมได้",
    params: [
      { key: "seedX", type: "number", label: "จุดเริ่ม X (%)", min: 0, max: 100, step: 1, default: 50, desc: "ตำแหน่งแนวนอนของจุดเริ่ม คิดเป็น % ของความกว้างภาพ (ใช้ % เหมือน Crop จึงไม่ต้องตั้งใหม่เมื่อความละเอียดกล้องเปลี่ยน)" },
      { key: "seedY", type: "number", label: "จุดเริ่ม Y (%)", min: 0, max: 100, step: 1, default: 50, desc: "ตำแหน่งแนวตั้งของจุดเริ่ม คิดเป็น % ของความสูงภาพ" },
      { key: "tolerance", type: "number", label: "ความต่างที่ยอมรับ", min: 0, max: 100, step: 1, default: 12, hint: "มาก = ลามกว้าง", desc: "ระดับเทาที่ต่างจากจุดเริ่มได้มากที่สุดแล้วยังถือว่าเป็นพื้นที่เดียวกัน — 0 = รับเฉพาะค่าที่เท่ากันเป๊ะ" },
      { key: "rangeMode", type: "select", label: "วิธีเทียบความต่าง", options: [
        { value: "fixed", label: "เทียบกับจุดเริ่ม (แนะนำ)", hint: "ทุกพิกเซลต้องต่างจากจุดเริ่มไม่เกินค่าที่ตั้ง → ผลลัพธ์คาดเดาได้" },
        { value: "floating", label: "เทียบกับเพื่อนบ้าน", hint: "เทียบทีละก้าวกับพิกเซลข้างเคียง จึงไต่ระดับความสว่างไปได้ไกลมาก" },
      ], default: "fixed" },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "mask", label: "หน้ากากขาว-ดำ", hint: "ต่อ Morphology / Bounding Boxes ได้ทันที" },
        { value: "overlay", label: "ระบายทับภาพเดิม", hint: "ใช้ตอนกำลังหาตำแหน่งจุดเริ่มที่ถูกต้อง" },
      ], default: "mask" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const tol = intIn(p.tolerance, 12, 0, 100); // 0 ต้องใช้ได้จริง — ห้ามใช้ `||` (P-13)
      let gray = null; // null = src เป็น 1 ช่องอยู่แล้ว (ห้าม delete src)
      let mask = null;
      let roi = null;
      let dst = null;
      try {
        // *** บังคับ *** ภาพ 4 ช่อง (RGBA) ทำให้ floodFill throw C++ exception และโหมด overlay ที่ใช้
        // COLOR_GRAY2RGB ก็ throw ถ้า input ไม่ใช่ 1 ช่อง → ต้องแปลงก่อนเสมอ ไม่ใช่จับ exception ทีหลัง
        gray = grayIfNeeded(cv, src);
        const g = gray || src;
        const cols = g.cols;
        const rows = g.rows;
        // *** บังคับ *** เมล็ดนอกกรอบ (x = cols, x = -1, y = rows) ทำให้ floodFill throw ทุกครั้ง (รั่ว ~760 B
        // ต่อครั้ง แก้ไม่ได้) — ใช้ (cols - 1) ไม่ใช่ cols เพื่อให้ 100% ตกที่พิกเซลสุดท้ายพอดี ไม่ใช่เลยขอบไป 1
        const sx = Math.min(cols - 1, Math.max(0, Math.round(numIn(p.seedX, 50, 0, 100) / 100 * (cols - 1))));
        const sy = Math.min(rows - 1, Math.max(0, Math.round(numIn(p.seedY, 50, 0, 100) / 100 * (rows - 1))));
        mask = new cv.Mat.zeros(rows + 2, cols + 2, cv.CV_8UC1); // floodFill บังคับขนาด +2 (ขนาดอื่น = throw)
        // 4 = 4-connectivity (ตั้งใจไม่เปิดเป็นพารามิเตอร์ เพื่อคุมจำนวน param ให้เท่ากับ node อื่นในกลุ่ม — เป็น
        //     การตัดสินใจ ไม่ใช่การลืม)
        // (255 << 8) = ค่าที่เขียนลง mask ถ้าไม่ตั้งบิต 8–15 จะได้ 1 แทน 255 แล้วภาพผลลัพธ์จะดำสนิททั้งภาพ
        // FLOODFILL_MASK_ONLY = ห้ามเขียนทับภาพต้นทาง! ถ้าไม่ใส่ floodFill จะแก้พิกเซลของ src จริง ๆ ซึ่ง
        //     runPipeline แชร์ Mat ตัวนั้นให้กล่องอื่นในกราฟอยู่ (ทดสอบแล้ว: ไม่ใส่ → src เปลี่ยน 3072 จุด)
        const flags = 4 | (255 << 8) | cv.FLOODFILL_MASK_ONLY
          | (p.rangeMode === "floating" ? 0 : cv.FLOODFILL_FIXED_RANGE);
        const filled = cv.floodFill(g, mask, new cv.Point(sx, sy), new cv.Scalar(0), new cv.Rect(),
          new cv.Scalar(tol, tol, tol, tol), new cv.Scalar(tol, tol, tol, tol), flags);
        roi = mask.roi(new cv.Rect(1, 1, cols, rows)); // roi() คืน wrapper ใหม่ที่ต้อง delete
        dst = new cv.Mat();
        if (p.view === "overlay") {
          cv.cvtColor(g, dst, cv.COLOR_GRAY2RGB); // RGB ไม่ใช่ BGR — แอปทำงานบน RGB
          dst.setTo(new cv.Scalar(0, 255, 0), roi); // ระบายเขียวเฉพาะพื้นที่ที่เลือก
          cv.circle(dst, new cv.Point(sx, sy), 4, new cv.Scalar(255, 0, 0), 2); // จุดเริ่มสีแดง (รัศมีคงที่ 4 —
          // ภาพที่เล็กกว่านั้น cv.circle ตัดขอบให้เอง ไม่ throw)
        } else {
          roi.copyTo(dst); // CV_8UC1 ค่า 0/255
        }
        if (ctx) {
          ctx.info = `เลือกได้ ${filled} พิกเซล (${(100 * filled / (cols * rows)).toFixed(1)}% ของภาพ) · จุดเริ่ม (${sx}, ${sy}) · ความต่างไม่เกิน ${tol}`;
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (roi) roi.delete();
        if (mask) mask.delete();
        if (gray) gray.delete();
      }
    },
  },

  // ================= 6. ประมวลผลภาพสี (Colour Processing) ================= (คงเลขหัวข้อเดิมไว้)
  color_inrange: {
    name: "HSV inRange",
    needs: "any",
    desc: "คัดเฉพาะพิกเซลที่มี 'เนื้อสี (H)' อยู่ในช่วงที่กำหนด — เป็นหัวใจของการตรวจจับวัตถุด้วยสี (เช่น หาลูกบอลสีแดง) ทำใน HSV เพราะเนื้อสีแยกจากความสว่าง ทำให้ทนต่อแสงเปลี่ยน H ของ OpenCV มีช่วง 0–179 (แดง≈0/179, เขียว≈60, น้ำเงิน≈120)",
    params: [
      { key: "colorSpace", type: "select", label: "ปริภูมิสี", options: [
        { value: "hsv", label: "HSV (เนื้อสี H)", hint: "แยกเนื้อสีออกจากความสว่าง — ใช้บ่อยสุด" },
        { value: "lab", label: "Lab (a*/b*)", hint: "ทนแสงเปลี่ยนได้ดีกว่า HSV ในบางกรณี ใกล้เคียงการรับรู้สีของมนุษย์" },
      ], default: "hsv" },
      { key: "hMin", visibleWhen: [{key:"colorSpace",in:["hsv"]}], type: "number", label: "H ต่ำสุด", min: 0, max: 179, step: 1, default: 35, desc: "ขอบล่างของช่วงเนื้อสีที่จะคัดเลือก (0–179) — ใช้เมื่อเลือกปริภูมิสี HSV" },
      { key: "hMax", visibleWhen: [{key:"colorSpace",in:["hsv"]}], type: "number", label: "H สูงสุด", min: 0, max: 179, step: 1, default: 85, desc: "ขอบบนของช่วงเนื้อสีที่จะคัดเลือก (0–179) — ใช้เมื่อเลือกปริภูมิสี HSV" },
      { key: "sMin", visibleWhen: [{key:"colorSpace",in:["hsv"]}], type: "number", label: "S ขั้นต่ำ (กันสีซีด)", min: 0, max: 255, step: 1, default: 60, desc: "ตัดสีที่จางเกินไป (ใกล้เทา) ทิ้ง — ยิ่งสูง ยิ่งต้องเป็นสีสดจึงจะติด (HSV เท่านั้น)" },
      { key: "vMin", visibleWhen: [{key:"colorSpace",in:["hsv"]}], type: "number", label: "V ขั้นต่ำ (กันส่วนมืด)", min: 0, max: 255, step: 1, default: 60, desc: "ตัดส่วนที่มืดเกินไปทิ้ง — ยิ่งสูง ยิ่งต้องสว่างจึงจะติด (HSV เท่านั้น)" },
      // พารามิเตอร์ Lab — แสดงเฉพาะเมื่อ colorSpace = "lab" (visibleWhen) · H/S/V ด้านบนแสดงเฉพาะ "hsv"
      { key: "labLMin", visibleWhen: [{key:"colorSpace",in:["lab"]}], type: "number", label: "Lab L ต่ำสุด (ความสว่าง)", min: 0, max: 255, step: 1, default: 0, desc: "ขอบล่างของความสว่าง (L) ที่จะคัดเลือก — ใช้เมื่อเลือกปริภูมิสี Lab" },
      { key: "labLMax", visibleWhen: [{key:"colorSpace",in:["lab"]}], type: "number", label: "Lab L สูงสุด (ความสว่าง)", min: 0, max: 255, step: 1, default: 255, desc: "ขอบบนของความสว่าง (L) — ใช้เมื่อเลือกปริภูมิสี Lab" },
      { key: "labAMin", visibleWhen: [{key:"colorSpace",in:["lab"]}], type: "number", label: "Lab a* ต่ำสุด (เขียว↔แดง)", min: 0, max: 255, step: 1, default: 150, hint: "128 = กลาง, <128 ออกเขียว, >128 ออกแดง (8-bit Lab: a*,b* ถูกบวก offset 128)", desc: "ขอบล่างของแกนสีเขียว-แดง — ใช้เมื่อเลือกปริภูมิสี Lab" },
      { key: "labAMax", visibleWhen: [{key:"colorSpace",in:["lab"]}], type: "number", label: "Lab a* สูงสุด (เขียว↔แดง)", min: 0, max: 255, step: 1, default: 220, desc: "ขอบบนของแกนสีเขียว-แดง — ใช้เมื่อเลือกปริภูมิสี Lab" },
      { key: "labBMin", visibleWhen: [{key:"colorSpace",in:["lab"]}], type: "number", label: "Lab b* ต่ำสุด (น้ำเงิน↔เหลือง)", min: 0, max: 255, step: 1, default: 130, hint: "128 = กลาง, <128 ออกน้ำเงิน, >128 ออกเหลือง", desc: "ขอบล่างของแกนสีน้ำเงิน-เหลือง — ใช้เมื่อเลือกปริภูมิสี Lab" },
      { key: "labBMax", visibleWhen: [{key:"colorSpace",in:["lab"]}], type: "number", label: "Lab b* สูงสุด (น้ำเงิน↔เหลือง)", min: 0, max: 255, step: 1, default: 200, desc: "ขอบบนของแกนสีน้ำเงิน-เหลือง — ใช้เมื่อเลือกปริภูมิสี Lab" },
      { key: "output", type: "select", label: "แสดงผลเป็น", options: [
        { value: "mask", label: "Mask ขาวดำ" },
        { value: "extract", label: "เฉพาะส่วนที่เป็นสีนั้น" },
      ], default: "extract" },
    ],
    apply(cv, src, p) {
      const rgb = new cv.Mat();
      if (src.channels() === 1) cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
      else src.copyTo(rgb);

      const colorSpace = p.colorSpace || "hsv"; // undefined -> "hsv" กันโปรเจกต์เก่าที่ไม่มี key นี้พังพฤติกรรม

      let mask;
      if (colorSpace === "lab") {
        const lab = new cv.Mat();
        // ยืนยันด้วย Node smoke test (ภาพ RGB บริสุทธิ์แดง/เขียว/น้ำเงิน/ขาว/ดำ): 8-bit Lab ของ build นี้
        // L สเกล 0–255, a*/b* ถูกบวก offset 128 (กลาง=128) ตรงตามคอมเมนต์ — แดงจริง a*≈208 เขียว a*≈42
        cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
        mask = new cv.Mat();
        const low = new cv.Mat(lab.rows, lab.cols, lab.type(), [p.labLMin, p.labAMin, p.labBMin, 0]);
        const high = new cv.Mat(lab.rows, lab.cols, lab.type(), [p.labLMax, p.labAMax, p.labBMax, 255]);
        cv.inRange(lab, low, high, mask);
        low.delete();
        high.delete();
        lab.delete();
      } else {
        // ---- HSV เดิม ไม่แตะต้อง (backward compatible กับโปรเจกต์เก่าทุกอัน) ----
        const hsv = new cv.Mat();
        cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
        mask = new cv.Mat();
        if (p.hMin > p.hMax) {
          // H วนรอบ (เช่น สีแดง คาบ 0/179) → รวมสองช่วง: [0..hMax] และ [hMin..179]
          const lo1 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, p.sMin, p.vMin, 0]);
          const hi1 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [p.hMax, 255, 255, 255]);
          const lo2 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [p.hMin, p.sMin, p.vMin, 0]);
          const hi2 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [179, 255, 255, 255]);
          const m1 = new cv.Mat();
          const m2 = new cv.Mat();
          cv.inRange(hsv, lo1, hi1, m1);
          cv.inRange(hsv, lo2, hi2, m2);
          cv.bitwise_or(m1, m2, mask);
          lo1.delete(); hi1.delete(); lo2.delete(); hi2.delete(); m1.delete(); m2.delete();
        } else {
          const low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [p.hMin, p.sMin, p.vMin, 0]);
          const high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [p.hMax, 255, 255, 255]);
          cv.inRange(hsv, low, high, mask);
          low.delete();
          high.delete();
        }
        hsv.delete();
      }

      let dst;
      if (p.output === "mask") {
        dst = new cv.Mat();
        mask.copyTo(dst);
      } else {
        dst = new cv.Mat();
        cv.bitwise_and(rgb, rgb, dst, mask);
      }
      rgb.delete();
      mask.delete();
      return dst;
    },
  },

  invert: {
    name: "Invert",
    needs: "any",
    desc: "กลับค่าทุกพิกเซล: ใหม่ = 255 − เดิม (เหมือนฟิล์ม negative) — ตัวอย่างการทำ point operation กับภาพสีที่ง่ายที่สุด และมีประโยชน์จริงเวลาต้องสลับวัตถุดำบนพื้นขาวให้เป็นขาวบนดำก่อนเข้า morphology/contours",
    params: [],
    apply(cv, src) {
      const dst = new cv.Mat();
      cv.bitwise_not(src, dst);
      return dst;
    },
  },

  white_balance: {
    name: "White Balance (Gray World)",
    needs: "any",
    desc: "แก้สีที่เพี้ยนจากแสงไม่ปกติ (เช่น ภาพออกเหลือง/ฟ้าทั้งภาพ) ด้วยสมมติฐาน 'Gray World': เฉลี่ยสีทั้งภาพควรออกเป็นสีเทากลาง ๆ ถ้าไม่เป็นแบบนั้น แสดงว่าแสงเพี้ยน จึงปรับแต่ละช่องสี (R/G/B) ให้ค่าเฉลี่ยเท่ากันหมด — ใช้ได้ผลดีกับภาพที่มีสีหลากหลายทั่วเฟรม · มี 3 สมมติฐานให้เลือก: เฉลี่ยเป็นเทา (Gray World) / ส่วนสว่างเป็นขาว (White Patch) / จุดที่บอกว่าเป็นเทา (จุดอ้างอิง) · Gray World พังเมื่อสีเดียวครอบภาพ (เช่น ใบไม้เต็มเฟรม) · MATLAB chromadapt แปลงในปริภูมิ XYZ แบบ Bradford แต่แอปคูณเกนตรงในช่อง R/G/B (หลักการเดียวกัน ตัวเลขต่างกัน)",
    params: [
      { key: "method", type: "select", label: "วิธี", options: [
        { value: "gray_world", label: "Gray World (เฉลี่ยทั้งภาพควรเป็นเทา)" },
        { value: "white_patch", label: "White Patch (ส่วนสว่างควรเป็นขาว)", hint: "ใช้ค่าที่ตำแหน่ง 99% ของแต่ละช่องเป็น 'ขาว' — เหมาะกับภาพที่มีของขาวจริงในฉาก" },
        { value: "ref_point", label: "จุดอ้างอิงสีเทา/ขาว", hint: "บอกตำแหน่งของที่รู้ว่าเป็นเทาหรือขาว (กระดาษ/การ์ดเทา) — ใช้ได้แม้สีเดียวครอบภาพ" },
      ], default: "gray_world" },
      { key: "refX", visibleWhen: [{key:"method",in:["ref_point"]}], type: "number", label: "จุดอ้างอิง แนวนอน (%) — ใช้กับจุดอ้างอิงเท่านั้น", min: 0, max: 100, step: 1, default: 50 },
      { key: "refY", visibleWhen: [{key:"method",in:["ref_point"]}], type: "number", label: "จุดอ้างอิง แนวตั้ง (%) — ใช้กับจุดอ้างอิงเท่านั้น", min: 0, max: 100, step: 1, default: 50 },
      { key: "refSize", visibleWhen: [{key:"method",in:["ref_point"]}], type: "number", label: "รัศมีบริเวณอ้างอิง (px) — ใช้กับจุดอ้างอิงเท่านั้น", min: 1, max: 50, step: 1, default: 5 },
    ],
    apply(cv, src, p, ctx) {
      const wbMethod = p.method === "white_patch" || p.method === "ref_point" ? p.method : null;
      if (wbMethod) {
        if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
        if (src.channels() !== 1) return whiteBalanceByGain(cv, src, p, ctx, wbMethod);
      }
      let rgb = null;
      let vec = null;
      let rCh = null;
      let gCh = null;
      let bCh = null;
      let rOut = null;
      let gOut = null;
      let bOut = null;
      let merged = null;
      let dst = null;
      try {
        rgb = new cv.Mat();
        if (src.channels() === 1) {
          // ภาพขาวดำไม่มีสีให้ปรับ — คืนภาพเดิม (RGB) โดยไม่ต้องคำนวณอะไร
          cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
          dst = new cv.Mat();
          rgb.copyTo(dst);
          if (ctx) ctx.info = "ภาพขาวดำ ไม่มีสีให้ปรับสมดุล";
          return dst;
        }
        src.copyTo(rgb);
        const meanVal = cv.mean(rgb); // [meanR, meanG, meanB, 0]
        const meanR = meanVal[0], meanG = meanVal[1], meanB = meanVal[2];
        const grayMean = (meanR + meanG + meanB) / 3;
        // ป้องกันหารด้วยศูนย์ (ภาพดำสนิททั้งใบ)
        const gainR = meanR > 1e-3 ? grayMean / meanR : 1;
        const gainG = meanG > 1e-3 ? grayMean / meanG : 1;
        const gainB = meanB > 1e-3 ? grayMean / meanB : 1;

        vec = new cv.MatVector();
        cv.split(rgb, vec);
        rCh = vec.get(0);
        gCh = vec.get(1);
        bCh = vec.get(2);
        rOut = new cv.Mat();
        gOut = new cv.Mat();
        bOut = new cv.Mat();
        rCh.convertTo(rOut, -1, gainR, 0);
        gCh.convertTo(gOut, -1, gainG, 0);
        bCh.convertTo(bOut, -1, gainB, 0);
        merged = new cv.MatVector();
        merged.push_back(rOut);
        merged.push_back(gOut);
        merged.push_back(bOut);
        dst = new cv.Mat();
        cv.merge(merged, dst);
        if (ctx) ctx.info = `เกน R×${gainR.toFixed(2)} G×${gainG.toFixed(2)} B×${gainB.toFixed(2)}`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (rgb) rgb.delete();
        if (vec) vec.delete();
        if (rCh) rCh.delete();
        if (gCh) gCh.delete();
        if (bCh) bCh.delete();
        if (rOut) rOut.delete();
        if (gOut) gOut.delete();
        if (bOut) bOut.delete();
        if (merged) merged.delete();
      }
    },
  },

  kmeans_segment: {
    name: "K-means Color Segmentation",
    needs: "any",
    desc: "แบ่งภาพเป็นกลุ่มสี K กลุ่มโดยอัตโนมัติ (unsupervised) — จับกลุ่มพิกเซลที่มีสีใกล้เคียงกันเข้าด้วยกัน แล้วแทนทุกพิกเซลในกลุ่มด้วยสีเฉลี่ยของกลุ่มนั้น ต่างจาก HSV inRange ที่ต้องกำหนดช่วงสีเอง เพราะตัวนี้ให้ระบบหาขอบเขตสีที่แยกกลุ่มได้ดีที่สุดเอง",
    params: [
      { key: "k", type: "number", label: "จำนวนกลุ่มสี (K)", min: 2, max: 12, step: 1, default: 4, desc: "จำนวนกลุ่มสีที่ต้องการแบ่ง — มาก = รายละเอียดสีมากขึ้นแต่ตีความยากขึ้น" },
      { key: "iterations", type: "number", label: "จำนวนรอบคำนวณสูงสุด", min: 5, max: 50, step: 5, default: 15, desc: "จำนวนรอบสูงสุดที่ปรับปรุงจุดศูนย์กลางกลุ่มสี — มาก = แม่นขึ้นแต่ช้าขึ้น" },
    ],
    apply(cv, src, p, ctx) {
      // K-means บนภาพเต็มความละเอียด (โดยเฉพาะกล้องสด) หนักมาก — วิเคราะห์บนภาพย่อไม่เกิน 320px
      // แล้วขยายผลลัพธ์ (label→สี) กลับด้วย INTER_NEAREST (ห้าม interpolate label/สี แบบต่อเนื่อง)
      const maxAnalysisSide = 320;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      const rgb = new cv.Mat();
      const analysis = new cv.Mat();
      const samplesC3 = new cv.Mat();
      const labels = new cv.Mat();
      const centers = new cv.Mat();
      let samples = null;
      let small = null;
      let dst = null;
      try {
        if (src.channels() === 1) cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
        else src.copyTo(rgb);

        if (scale < 1) {
          cv.resize(rgb, analysis, new cv.Size(Math.max(1, Math.round(rgb.cols * scale)), Math.max(1, Math.round(rgb.rows * scale))), 0, 0, cv.INTER_AREA);
        } else {
          rgb.copyTo(analysis);
        }

        const n = analysis.rows * analysis.cols;
        // cv.kmeans ต้องการ sample เป็น CV_32F แถวละ 1 sample, คอลัมน์ = จำนวน channel (3)
        // build นี้ไม่มี Mat.prototype.reshape (ยืนยันด้วย Node smoke test) จึงคัดลอกข้อมูล 32F
        // แบบ flat ออกมาสร้าง Mat รูปร่าง n x 3 x 1channel ใหม่ตรง ๆ แทนการ "reshape" ของแม็ทเดิม
        analysis.convertTo(samplesC3, cv.CV_32F);
        samples = cv.matFromArray(n, 3, cv.CV_32FC1, new Float32Array(samplesC3.data32F));

        // n ตัวอย่างอาจน้อยกว่า K ที่ตั้งไว้ (เช่น crop เหลือภาพจิ๋ว) — cv.kmeans throw ถ้า n < K
        // จึง clamp K ไม่ให้เกินจำนวนตัวอย่าง (ยังคง >= 2 ถ้ามีตัวอย่างพอ)
        const rawK = Math.round(p.k) || 2;
        const K = Math.max(n >= 2 ? 2 : 1, Math.min(rawK, n));
        // cv.TERM_CRITERIA_EPS / cv.TERM_CRITERIA_MAX_ITER ไม่ถูก bind มาใน build นี้ (ยืนยันด้วย
        // Node smoke test — cv.TERM_CRITERIA_EPS === undefined) จึงต้องใช้เลขคงที่ของ OpenCV เอง:
        // COUNT/MAX_ITER = 1, EPS = 2 → รวมกัน = 3 (ทดสอบแล้วว่า cv.kmeans ทำงานถูกต้องด้วยค่านี้)
        const TERM_CRITERIA_EPS_MAX_ITER = 3;
        const criteria = new cv.TermCriteria(TERM_CRITERIA_EPS_MAX_ITER, Math.max(1, Math.round(p.iterations)), 1.0);
        cv.kmeans(samples, K, labels, criteria, 3, cv.KMEANS_PP_CENTERS, centers);

        // สร้างภาพย่อ (label → สีเฉลี่ยของกลุ่ม) จาก labels/centers
        small = new cv.Mat(analysis.rows, analysis.cols, cv.CV_8UC3);
        const centersF = centers.data32F; // K x 3, float32
        const labelData = labels.data32S; // n x 1 int32
        const outData = small.data;
        for (let i = 0, px = 0; i < n; i++, px += 3) {
          const lbl = labelData[i];
          outData[px] = Math.round(centersF[lbl * 3]);
          outData[px + 1] = Math.round(centersF[lbl * 3 + 1]);
          outData[px + 2] = Math.round(centersF[lbl * 3 + 2]);
        }

        dst = new cv.Mat();
        if (scale < 1) {
          cv.resize(small, dst, new cv.Size(rgb.cols, rgb.rows), 0, 0, cv.INTER_NEAREST);
        } else {
          small.copyTo(dst);
        }
        if (ctx) ctx.info = `แบ่งภาพเป็น ${K} กลุ่มสี`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        rgb.delete();
        analysis.delete();
        samplesC3.delete();
        labels.delete();
        centers.delete();
        if (samples) samples.delete();
        if (small) small.delete();
      }
    },
  },

  // ---------------- 8. Geometry ----------------
  crop: {
    name: "Crop",
    needs: "any",
    desc: "ตัดภาพให้เหลือเฉพาะบริเวณสี่เหลี่ยมที่กำหนด — กำหนดเป็น % ของภาพเดิม (ไม่ใช่พิกเซลตรง ๆ) เพื่อให้ใช้ได้กับภาพทุกขนาด รอบนี้ยังตั้งค่าด้วยตัวเลข ยังลากกรอบบนภาพไม่ได้",
    params: [
      { key: "x", type: "number", label: "ขอบซ้าย (%)", min: 0, max: 90, step: 1, default: 10, desc: "ระยะจากขอบซ้ายของภาพเดิม เป็น % ของความกว้าง" },
      { key: "y", type: "number", label: "ขอบบน (%)", min: 0, max: 90, step: 1, default: 10, desc: "ระยะจากขอบบนของภาพเดิม เป็น % ของความสูง" },
      { key: "w", type: "number", label: "ความกว้าง (%)", min: 5, max: 100, step: 1, default: 80, desc: "ความกว้างของพื้นที่ที่ตัด เป็น % ของความกว้างภาพเดิม" },
      { key: "h", type: "number", label: "ความสูง (%)", min: 5, max: 100, step: 1, default: 80, desc: "ความสูงของพื้นที่ที่ตัด เป็น % ของความสูงภาพเดิม" },
    ],
    apply(cv, src, p, ctx) {
      const cols = src.cols, rows = src.rows;
      // กันพารามิเตอร์ที่ไม่ใช่ตัวเลข/NaN (เช่น params ที่ยังไม่ถูกตั้งค่า) ไม่ให้ได้ Mat ขนาด 0x0
      const fallback = (v, d) => (Number.isFinite(v) ? v : d);
      let x = Math.round((fallback(p.x, 10) / 100) * cols);
      let y = Math.round((fallback(p.y, 10) / 100) * rows);
      let w = Math.round((fallback(p.w, 80) / 100) * cols);
      let h = Math.round((fallback(p.h, 80) / 100) * rows);
      // Clamp กัน ROI หลุดขอบภาพ (ค่า % รวมกันเกิน 100 ได้ เช่น x=90,w=100)
      x = Math.max(0, Math.min(x, cols - 1));
      y = Math.max(0, Math.min(y, rows - 1));
      w = Math.max(1, Math.min(w, cols - x));
      h = Math.max(1, Math.min(h, rows - y));
      const rect = new cv.Rect(x, y, w, h);
      let roi = null;
      let dst = null;
      try {
        roi = src.roi(rect); // roi() คืน view ไม่ใช่ copy — clone ก่อนคืนค่า กัน caller delete() แล้วพัง
        dst = new cv.Mat();
        roi.copyTo(dst);
        if (ctx) ctx.info = `ตัดภาพเหลือ ${w}×${h} px`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (roi) roi.delete();
      }
    },
  },

  resize: {
    name: "Resize",
    needs: "any",
    desc: "ปรับขนาดภาพทั้งใบด้วยการ resample พิกเซลใหม่ — ย่อ (Area) ให้ภาพเรียบ ไม่มีจุดฟันปลา ส่วนขยาย (Cubic) ให้ภาพเนียนกว่าการเพิ่มพิกเซลแบบซ้ำตรง ๆ" +
      " · วิธีคำนวณพิกเซลใหม่ (interpolation): ตำแหน่งพิกเซลใหม่มักตกอยู่ระหว่างพิกเซลเดิม จึงต้องคำนวณค่า — Nearest neighbour ใช้พิกเซลเดิมที่อยู่ใกล้จุดนั้น (ขยายแล้วเห็นเป็นบล็อก), Bilinear เฉลี่ย 2×2 พิกเซลรอบจุด (นุ่ม), Bicubic ใช้ 4×4 พิกเซล (คมกว่า ขอบอาจมีเส้นจาง ๆ เกินออกมา) · ย่อภาพโดยไม่เฉลี่ยพิกเซลที่ถูกทิ้งจะเกิดลายคลื่น (aliasing) — Resize แบบอัตโนมัติใช้ Area ซึ่งเฉลี่ยให้ · ค่าเริ่มต้นของ MATLAB ต่างกัน (imresize = bicubic พร้อม antialiasing, imrotate = nearest) ส่วน \"อัตโนมัติ\" ของแอปนี้ยังใช้วิธีเดิมของตัวเอง",
    params: [
      { key: "scale", type: "number", label: "อัตราขยาย/ย่อ (%)", min: 10, max: 400, step: 5, default: 50, desc: "100% = ขนาดเดิม, ต่ำกว่า = ย่อ, สูงกว่า = ขยาย" },
      interpParam("อัตโนมัติ (ย่อ = Area · ขยาย = Bicubic — แบบเดิม)"),
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const rawScale = Number(p.scale);
      const s = Math.max(0.05, (Number.isFinite(rawScale) ? rawScale : 50) / 100);
      let w = Math.max(1, Math.round(src.cols * s));
      let h = Math.max(1, Math.round(src.rows * s));
      // เพดานความปลอดภัยด้านหน่วยความจำ: ขยาย 400% บนภาพ Full HD ได้ภาพ ~33 ล้านพิกเซล (~132MB/เฟรม
      // แบบ RGBA) ถ้าต่อ Resize หลายตัว/ใช้กับกล้องสด WASM heap อาจไม่พอ — จำกัดที่ 20 ล้านพิกเซลรวม
      // (เพียงพอสำหรับงานสอน เทียบเท่าภาพราว 5000×4000) โดยลดสัดส่วนกว้าง/สูงเท่ากันเพื่อคงอัตราส่วนเดิม
      const MAX_OUTPUT_PIXELS = 20_000_000;
      let clamped = false;
      if (w * h > MAX_OUTPUT_PIXELS) {
        const shrink = Math.sqrt(MAX_OUTPUT_PIXELS / (w * h));
        w = Math.max(1, Math.round(w * shrink));
        h = Math.max(1, Math.round(h * shrink));
        clamped = true;
      }
      let dst = null;
      try {
        dst = new cv.Mat();
        const k = interpKey(p.interp);
        const interp = k === "auto" ? (s < 1 ? cv.INTER_AREA : cv.INTER_CUBIC) : cv[INTERP_FLAG[k]];
        cv.resize(src, dst, new cv.Size(w, h), 0, 0, interp);
        if (ctx) {
          ctx.info = clamped
            ? `ปรับขนาดเป็น ${w}×${h} px (ลดขนาดลงจากที่ตั้งไว้ เพื่อกันหน่วยความจำเกิน ${(MAX_OUTPUT_PIXELS / 1e6).toFixed(0)} ล้านพิกเซล)`
            : `ปรับขนาดเป็น ${w}×${h} px`;
          if (k !== "auto") {
            ctx.info += ` · ${INTERP_NAME[k]}`;
            if (s < 1) ctx.info += " · ย่อด้วยวิธีนี้ไม่เฉลี่ยพิกเซลที่ถูกทิ้ง ลายละเอียดถี่ ๆ อาจกลายเป็นลายคลื่น (aliasing) — แบบอัตโนมัติใช้ Area ซึ่งเฉลี่ยให้";
          }
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      }
    },
  },

  rotate: {
    name: "Rotate",
    needs: "any",
    desc: "หมุนภาพรอบจุดกึ่งกลางตามมุมที่กำหนด — เลือกได้ว่าจะ 'คงขนาดเดิม' (มุมภาพอาจถูกตัดออกไปถ้าหมุนเฉียง) หรือ 'ขยายผืนภาพให้พอดี' (เห็นภาพเต็มไม่ถูกตัด แต่ขนาดภาพใหญ่ขึ้น มีพื้นหลังเติมเข้ามา)" +
      " · วิธีคำนวณพิกเซลใหม่ (interpolation): ตำแหน่งพิกเซลใหม่มักตกอยู่ระหว่างพิกเซลเดิม จึงต้องคำนวณค่า — Nearest neighbour ใช้พิกเซลเดิมที่อยู่ใกล้จุดนั้น (ขยายแล้วเห็นเป็นบล็อก), Bilinear เฉลี่ย 2×2 พิกเซลรอบจุด (นุ่ม), Bicubic ใช้ 4×4 พิกเซล (คมกว่า ขอบอาจมีเส้นจาง ๆ เกินออกมา) · ย่อภาพโดยไม่เฉลี่ยพิกเซลที่ถูกทิ้งจะเกิดลายคลื่น (aliasing) — Resize แบบอัตโนมัติใช้ Area ซึ่งเฉลี่ยให้ · ค่าเริ่มต้นของ MATLAB ต่างกัน (imresize = bicubic พร้อม antialiasing, imrotate = nearest) ส่วน \"อัตโนมัติ\" ของแอปนี้ยังใช้วิธีเดิมของตัวเอง",
    params: [
      { key: "angle", type: "number", label: "มุมหมุน (องศา)", min: -180, max: 180, step: 1, default: 15, desc: "ค่าบวก = หมุนทวนเข็มนาฬิกา, ค่าลบ = หมุนตามเข็มนาฬิกา (ตามธรรมเนียม OpenCV)" },
      { key: "expand", type: "boolean", label: "ขยายผืนภาพให้พอดี (ไม่ตัดมุม)", default: true },
      interpParam("อัตโนมัติ (Bilinear — แบบเดิม)"),
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const k = interpKey(p.interp);
      const angle = Number(p.angle) || 0; // กัน undefined/NaN (params ยังไม่ถูกตั้งค่า)
      const w = src.cols, h = src.rows;
      const center = new cv.Point(w / 2, h / 2);
      let M = null;
      let dst = null;
      try {
        M = cv.getRotationMatrix2D(center, angle, 1.0);
        let outW = w, outH = h;
        if (boolOr(p.expand, false)) {
          const rad = (Math.abs(angle) * Math.PI) / 180;
          const cosA = Math.abs(Math.cos(rad));
          const sinA = Math.abs(Math.sin(rad));
          outW = Math.round(w * cosA + h * sinA);
          outH = Math.round(w * sinA + h * cosA);
          // เลื่อนส่วน translation ของ M ให้ผืนภาพใหญ่ขึ้น (ซึ่งตอนนี้ใหญ่ขึ้น) อยู่กึ่งกลางกับเนื้อหาที่หมุนแล้ว
          // M เป็น 2x3 float64: [ [a,b,tx], [c,d,ty] ] — ปรับ tx,ty ตามขนาดผืนภาพที่โตขึ้น
          M.doublePtr(0, 2)[0] += (outW - w) / 2;
          M.doublePtr(1, 2)[0] += (outH - h) / 2;
        }
        dst = new cv.Mat();
        const borderValue = src.channels() === 1 ? new cv.Scalar(0) : new cv.Scalar(0, 0, 0);
        cv.warpAffine(src, dst, M, new cv.Size(outW, outH), k === "auto" ? cv.INTER_LINEAR : cv[INTERP_FLAG[k]], cv.BORDER_CONSTANT, borderValue);
        if (ctx) ctx.info = `หมุน ${angle}° → ${outW}×${outH} px` + (k === "auto" ? "" : ` · ${INTERP_NAME[k]}`);
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (M) M.delete();
      }
    },
  },

  flip: {
    name: "Flip",
    needs: "any",
    desc: "กลับภาพในแนวนอนหรือแนวตั้ง (เหมือนกระจกเงา) — ใช้ตรวจสอบว่า pipeline ทำงานสมมาตรถูกต้องหรือจำลองมุมกล้องที่ติดกลับด้าน",
    params: [
      { key: "mode", type: "select", label: "แนวกลับภาพ", options: [
        { value: "h", label: "แนวนอน (Horizontal)", hint: "กลับซ้าย-ขวา" },
        { value: "v", label: "แนวตั้ง (Vertical)", hint: "กลับบน-ล่าง" },
        { value: "both", label: "ทั้งสองแนว", hint: "กลับซ้าย-ขวา และ บน-ล่าง" },
      ], default: "h" },
    ],
    apply(cv, src, p) {
      const dst = new cv.Mat();
      const code = { h: 1, v: 0, both: -1 }[p.mode] ?? 1; // undefined mode -> default แนวนอน
      cv.flip(src, dst, code);
      return dst;
    },
  },

  perspective_warp: {
    name: "Perspective Warp",
    needs: "any",
    desc: "บิดภาพแบบ perspective (มุมมอง 3 มิติ) โดยกำหนดว่า 4 มุมของภาพเดิมจะ 'ย้ายไปอยู่ตรงไหน' บนภาพใหม่ — ต่างจาก Rotate/Flip ที่เป็น affine (เส้นตรงยังขนานกันเสมอ) เพราะ perspective ทำให้เส้นที่เคยขนานกันบรรจบกันได้ (เหมือนแก้ภาพถ่ายเอียง/สแกนเอกสารมุมมองเบี้ยว)",
    params: [
      { key: "tlX", type: "number", label: "มุมซ้ายบน — X (%)", min: -50, max: 50, step: 1, default: 10, desc: "ระยะที่มุมซ้ายบนของภาพเดิมเลื่อนในแนวนอน (% ของความกว้าง, ค่าบวก = เลื่อนขวา)" },
      { key: "tlY", type: "number", label: "มุมซ้ายบน — Y (%)", min: -50, max: 50, step: 1, default: 0, desc: "ระยะที่มุมซ้ายบนเลื่อนในแนวตั้ง (ค่าบวก = เลื่อนลง)" },
      { key: "trX", type: "number", label: "มุมขวาบน — X (%)", min: -50, max: 50, step: 1, default: -10, desc: "ระยะที่มุมขวาบนของภาพเดิมเลื่อนในแนวนอน" },
      { key: "trY", type: "number", label: "มุมขวาบน — Y (%)", min: -50, max: 50, step: 1, default: 0, desc: "ระยะที่มุมขวาบนเลื่อนในแนวตั้ง" },
      { key: "brX", type: "number", label: "มุมขวาล่าง — X (%)", min: -50, max: 50, step: 1, default: 0, desc: "ระยะที่มุมขวาล่างของภาพเดิมเลื่อนในแนวนอน" },
      { key: "brY", type: "number", label: "มุมขวาล่าง — Y (%)", min: -50, max: 50, step: 1, default: 0, desc: "ระยะที่มุมขวาล่างเลื่อนในแนวตั้ง" },
      { key: "blX", type: "number", label: "มุมซ้ายล่าง — X (%)", min: -50, max: 50, step: 1, default: 0, desc: "ระยะที่มุมซ้ายล่างของภาพเดิมเลื่อนในแนวนอน" },
      { key: "blY", type: "number", label: "มุมซ้ายล่าง — Y (%)", min: -50, max: 50, step: 1, default: 0, desc: "ระยะที่มุมซ้ายล่างเลื่อนในแนวตั้ง" },
    ],
    apply(cv, src, p, ctx) {
      const w = src.cols, h = src.rows;
      const pct = (v, d) => (Number.isFinite(v) ? v : d) / 100;
      const srcQuad = [0, 0, w - 1, 0, w - 1, h - 1, 0, h - 1]; // TL, TR, BR, BL
      const dstQuad = [
        pct(p.tlX, 10) * w, pct(p.tlY, 0) * h,
        w - 1 + pct(p.trX, -10) * w, pct(p.trY, 0) * h,
        w - 1 + pct(p.brX, 0) * w, h - 1 + pct(p.brY, 0) * h,
        pct(p.blX, 0) * w, h - 1 + pct(p.blY, 0) * h,
      ];
      let srcPts = null;
      let dstPts = null;
      let M = null;
      let dst = null;
      try {
        srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, srcQuad);
        dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, dstQuad);
        M = cv.getPerspectiveTransform(srcPts, dstPts);
        dst = new cv.Mat();
        const borderValue = src.channels() === 1 ? new cv.Scalar(0) : new cv.Scalar(0, 0, 0);
        cv.warpPerspective(src, dst, M, new cv.Size(w, h), cv.INTER_LINEAR, cv.BORDER_CONSTANT, borderValue);
        if (ctx) ctx.info = "บิด perspective ตามตำแหน่ง 4 มุมที่กำหนด";
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (srcPts) srcPts.delete();
        if (dstPts) dstPts.delete();
        if (M) M.delete();
      }
    },
  },

  undistort: {
    name: "Undistort (Lens Correction)",
    needs: "any",
    desc: "แก้ภาพที่บิดเบี้ยวจากเลนส์กล้อง (บวมออกตรงกลางแบบ 'พุง' หรือบุ๋มเข้าแบบ 'หมอน') ด้วยค่าความบิดเบี้ยวที่ตั้งเอง (k1, k2) — ไม่ใช่การ Calibrate กล้องแบบเต็มรูปแบบ (ไม่ต้องถ่ายกระดานตาราง) เหมาะสำหรับทดลองความเข้าใจว่าค่า k1/k2 มีผลต่อภาพอย่างไร ไม่ใช่ค่าที่แม่นยำสำหรับกล้องจริง",
    params: [
      { key: "k1", type: "number", label: "ความบิดแนวรัศมี k1", min: -1, max: 1, step: 0.05, default: -0.3, desc: "ค่าลบ = แก้ภาพพุง (บาร์เรล, พบบ่อยในเลนส์มุมกว้าง), ค่าบวก = แก้ภาพบุ๋ม (พินคูชั่น)" },
      { key: "k2", type: "number", label: "ความบิดแนวรัศมี k2", min: -1, max: 1, step: 0.05, default: 0.0, desc: "ปรับละเอียดเพิ่มจาก k1 — มีผลมากตรงมุมภาพ" },
    ],
    apply(cv, src, p, ctx) {
      const w = src.cols, h = src.rows;
      const f = Math.max(w, h);
      const k1 = finiteOr(p.k1, -0.3);
      const k2 = finiteOr(p.k2, 0);
      let cameraMatrix = null, distCoeffs = null, dst = null;
      try {
        cameraMatrix = cv.matFromArray(3, 3, cv.CV_64FC1, [f, 0, w / 2, 0, f, h / 2, 0, 0, 1]);
        distCoeffs = cv.matFromArray(1, 5, cv.CV_64FC1, [k1, k2, 0, 0, 0]);
        dst = new cv.Mat();
        cv.undistort(src, dst, cameraMatrix, distCoeffs);
        if (ctx) ctx.info = `k1=${k1}, k2=${k2} (fx=fy=${f}, กึ่งกลางภาพเป็นจุดหลัก)`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (cameraMatrix) cameraMatrix.delete();
        if (distCoeffs) distCoeffs.delete();
      }
    },
  },

  // ---------------- 9. Frequency Domain ----------------
  fft_magnitude: {
    name: "FFT Magnitude Spectrum",
    needs: "gray",
    desc: "แปลงภาพจากโดเมนพิกเซล (spatial) ไปเป็นโดเมนความถี่ (frequency) ด้วย Fourier Transform แล้ววาดขนาด (magnitude) ของแต่ละความถี่เป็นความสว่าง — จุดกลางภาพ = ความถี่ต่ำ (ความสว่างเปลี่ยนช้า/ภาพรวม), ยิ่งห่างจากกลาง = ความถี่สูง (รายละเอียด/ขอบ/noise) เป็นพื้นฐานก่อนทำ low-pass/high-pass filter",
    params: [],
    apply(cv, src, p, ctx) {
      const maxAnalysisSide = 640;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      const analysis = new cv.Mat();
      let padded = null;
      let floatMat = null;
      let planes = null;
      let zeros = null;
      let complexI = null;
      let dftOut = null;
      let splitPlanes = null;
      let re = null;
      let im = null;
      let mag = null;
      let one = null;
      let norm = null;
      let mag8u = null;
      let dst = null;
      try {
        const analysisW = Math.max(1, Math.round(src.cols * scale));
        const analysisH = Math.max(1, Math.round(src.rows * scale));
        if (scale < 1) {
          cv.resize(src, analysis, new cv.Size(analysisW, analysisH), 0, 0, cv.INTER_AREA);
        } else {
          src.copyTo(analysis);
        }
        let optRows = cv.getOptimalDFTSize(analysis.rows);
        let optCols = cv.getOptimalDFTSize(analysis.cols);
        if (optRows % 2 !== 0) optRows += 1; // fftQuadrantSwap ต้องการมิติคู่
        if (optCols % 2 !== 0) optCols += 1;
        padded = new cv.Mat();
        cv.copyMakeBorder(analysis, padded, 0, optRows - analysis.rows, 0, optCols - analysis.cols, cv.BORDER_CONSTANT, new cv.Scalar(0));
        floatMat = new cv.Mat();
        padded.convertTo(floatMat, cv.CV_32F);
        planes = new cv.MatVector();
        zeros = new cv.Mat.zeros(padded.rows, padded.cols, cv.CV_32F);
        planes.push_back(floatMat);
        planes.push_back(zeros);
        complexI = new cv.Mat();
        cv.merge(planes, complexI);

        dftOut = new cv.Mat();
        cv.dft(complexI, dftOut, cv.DFT_COMPLEX_OUTPUT);
        fftQuadrantSwap(cv, dftOut);

        splitPlanes = new cv.MatVector();
        cv.split(dftOut, splitPlanes);
        re = splitPlanes.get(0);
        im = splitPlanes.get(1);
        mag = new cv.Mat();
        cv.magnitude(re, im, mag);

        one = new cv.Mat(mag.rows, mag.cols, mag.type(), new cv.Scalar(1));
        cv.add(mag, one, mag);
        cv.log(mag, mag);
        norm = new cv.Mat();
        cv.normalize(mag, norm, 0, 255, cv.NORM_MINMAX);
        mag8u = new cv.Mat();
        norm.convertTo(mag8u, cv.CV_8U);

        dst = new cv.Mat();
        cv.resize(mag8u, dst, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_LINEAR);
        if (ctx) ctx.info = "จุดสว่างกลางภาพ = ความถี่ต่ำ, ขอบภาพ = ความถี่สูง";
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        analysis.delete();
        if (padded) padded.delete();
        if (floatMat) floatMat.delete();
        if (planes) planes.delete();
        if (zeros) zeros.delete();
        if (complexI) complexI.delete();
        if (dftOut) dftOut.delete();
        if (splitPlanes) splitPlanes.delete();
        if (re) re.delete();
        if (im) im.delete();
        if (mag) mag.delete();
        if (one) one.delete();
        if (norm) norm.delete();
        if (mag8u) mag8u.delete();
      }
    },
  },

  fft_filter: {
    name: "FFT Low/High-pass Filter",
    needs: "gray",
    desc: "กรองภาพในโดเมนความถี่: Low-pass เก็บเฉพาะความถี่ต่ำ (ภาพรวม/เบลอ, ตัดรายละเอียด/noise ทิ้ง) ส่วน High-pass เก็บเฉพาะความถี่สูง (ขอบ/รายละเอียด, ตัดภาพรวมทิ้ง) — แบบ Ideal ตัดเฉียบขาดที่รัศมีเดียว (อาจเห็นรอยคลื่นแทรกซ้อน 'ringing' ทั่วภาพ), แบบ Gaussian ตัดแบบค่อยเป็นค่อยไป (นุ่มกว่า ไม่มี ringing แบบนั้น) — หมายเหตุ: FFT มองภาพเป็น 'ภาพวนซ้ำ' (periodic) เสมอ ถ้าขอบซ้าย-ขวา หรือบน-ล่างของภาพมีความสว่างต่างกันมาก อาจเห็นแถบ ringing บาง ๆ ที่ริมภาพได้แม้ใช้ Gaussian — เป็นข้อจำกัดของ FFT พื้นฐาน ไม่ใช่บั๊ก · Butterworth: H = 1/(1+(D/D0)^2n) — n = 1–2 แทบไม่มี ringing, n สูงคล้าย Ideal (ตำราสอนคู่กับ Ideal/Gaussian)",
    params: [
      { key: "type", type: "select", label: "ชนิดตัวกรอง", options: [
        { value: "low", label: "Low-pass (เก็บภาพรวม/เบลอ)", hint: "ตัดรายละเอียด/ขอบ/noise ความถี่สูงทิ้ง" },
        { value: "high", label: "High-pass (เก็บขอบ/รายละเอียด)", hint: "ตัดภาพรวม/ความสว่างเฉลี่ยทิ้ง เหลือแต่ขอบ" },
      ], default: "low" },
      { key: "shape", type: "select", label: "รูปแบบขอบตัด", options: [
        { value: "gaussian", label: "Gaussian (นุ่ม, ไม่มี ringing)" },
        { value: "ideal", label: "Ideal (ตัดเฉียบขาด, อาจมี ringing)" },
        { value: "butterworth", label: "Butterworth (นุ่ม ปรับความชันด้วยลำดับ n)" },
      ], default: "gaussian" },
      { key: "cutoff", type: "number", label: "รัศมีตัด (ความถี่ นับจากกลางสเปกตรัม)", min: 2, max: 200, step: 2, default: 30, desc: "รัศมีวงกลมความถี่ที่เป็นเส้นแบ่งเก็บ/ตัด — Gaussian ใช้ค่านี้เป็น σ โดยประมาณ" },
      { key: "order", visibleWhen: [{key:"shape",in:["butterworth"]}], type: "number", label: "ลำดับ n — ใช้กับ Butterworth เท่านั้น", min: 1, max: 10, step: 1, default: 2, desc: "n น้อย = ลาดนุ่มคล้าย Gaussian · n มาก = ชันเข้าใกล้ Ideal (เริ่มเห็น ringing)" },
    ],
    apply(cv, src, p, ctx) {
      // IP batch 4A: Butterworth เท่านั้นที่มี guard ภาพว่าง (รูปแบบเดิมคงพฤติกรรมเดิม — throw บน 0×0)
      if (p.shape === "butterworth" && isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const maxAnalysisSide = 640;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      const analysis = new cv.Mat();
      let padded = null;
      let floatMat = null;
      let planes = null;
      let zeros = null;
      let complexI = null;
      let dftOut = null;
      let mask1ch = null;
      let maskVec = null;
      let mask2ch = null;
      let filtered = null;
      let inv = null;
      let invCropped = null;
      let normed = null;
      let u8 = null;
      let dst = null;
      try {
        const analysisW = Math.max(1, Math.round(src.cols * scale));
        const analysisH = Math.max(1, Math.round(src.rows * scale));
        if (scale < 1) {
          cv.resize(src, analysis, new cv.Size(analysisW, analysisH), 0, 0, cv.INTER_AREA);
        } else {
          src.copyTo(analysis);
        }
        let optRows = cv.getOptimalDFTSize(analysis.rows);
        let optCols = cv.getOptimalDFTSize(analysis.cols);
        if (optRows % 2 !== 0) optRows += 1;
        if (optCols % 2 !== 0) optCols += 1;
        padded = new cv.Mat();
        cv.copyMakeBorder(analysis, padded, 0, optRows - analysis.rows, 0, optCols - analysis.cols, cv.BORDER_CONSTANT, new cv.Scalar(0));
        floatMat = new cv.Mat();
        padded.convertTo(floatMat, cv.CV_32F);
        planes = new cv.MatVector();
        zeros = new cv.Mat.zeros(padded.rows, padded.cols, cv.CV_32F);
        planes.push_back(floatMat);
        planes.push_back(zeros);
        complexI = new cv.Mat();
        cv.merge(planes, complexI);

        dftOut = new cv.Mat();
        cv.dft(complexI, dftOut, cv.DFT_COMPLEX_OUTPUT);
        fftQuadrantSwap(cv, dftOut);

        // สร้าง mask 1 channel เป็น float ด้วยลูป JS ตรง ๆ (เร็วพอ — ทดสอบแล้ว ~640x480 ใช้ <10ms)
        const cutoff = finiteOr(p.cutoff, 30);
        const bwN = intIn(p.order, 2, 1, 10); // ลำดับของ Butterworth (ไม่ใช้กับรูปแบบอื่น)
        // รัศมีความถี่ (รอบต่อภาพ) ไม่เปลี่ยนตามความละเอียด ห้ามคูณ scale ซ้ำ — ไม่เช่นนั้นภาพ >640px จะถูกกรองแรงเกินจริง
        const cx = dftOut.cols / 2, cy = dftOut.rows / 2;
        const maskData = new Float32Array(dftOut.rows * dftOut.cols);
        for (let y = 0; y < dftOut.rows; y++) {
          for (let x = 0; x < dftOut.cols; x++) {
            const dx = x - cx, dy = y - cy;
            const d = Math.sqrt(dx * dx + dy * dy);
            let lowVal;
            if (p.shape === "ideal") {
              lowVal = d <= cutoff ? 1 : 0;
            } else if (p.shape === "butterworth") {
              lowVal = 1 / (1 + Math.pow(d / cutoff, 2 * bwN));
            } else {
              lowVal = Math.exp(-(d * d) / (2 * cutoff * cutoff));
            }
            maskData[y * dftOut.cols + x] = p.type === "high" ? 1 - lowVal : lowVal;
          }
        }
        mask1ch = cv.matFromArray(dftOut.rows, dftOut.cols, cv.CV_32FC1, maskData);
        maskVec = new cv.MatVector();
        maskVec.push_back(mask1ch);
        maskVec.push_back(mask1ch);
        mask2ch = new cv.Mat();
        cv.merge(maskVec, mask2ch);

        filtered = new cv.Mat();
        cv.multiply(dftOut, mask2ch, filtered);

        fftQuadrantSwap(cv, filtered); // unshift ก่อน inverse เสมอ

        inv = new cv.Mat();
        cv.dft(filtered, inv, cv.DFT_INVERSE + cv.DFT_SCALE + cv.DFT_REAL_OUTPUT);

        invCropped = inv.roi(new cv.Rect(0, 0, analysis.cols, analysis.rows));
        normed = new cv.Mat();
        cv.normalize(invCropped, normed, 0, 255, cv.NORM_MINMAX);
        u8 = new cv.Mat();
        normed.convertTo(u8, cv.CV_8U);

        dst = new cv.Mat();
        cv.resize(u8, dst, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_LINEAR);
        if (ctx) ctx.info = `${p.type === "high" ? "High" : "Low"}-pass (${p.shape}${p.shape === "butterworth" ? ` n=${bwN}` : ""}), รัศมี ${cutoff} (หน่วยความถี่ นับจากกลางสเปกตรัม — ไม่ใช่พิกเซลของภาพจริง)`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        analysis.delete();
        if (padded) padded.delete();
        if (floatMat) floatMat.delete();
        if (planes) planes.delete();
        if (zeros) zeros.delete();
        if (complexI) complexI.delete();
        if (dftOut) dftOut.delete();
        if (mask1ch) mask1ch.delete();
        if (maskVec) maskVec.delete();
        if (mask2ch) mask2ch.delete();
        if (filtered) filtered.delete();
        if (inv) inv.delete();
        if (normed) normed.delete();
        if (u8) u8.delete();
        if (invCropped) invCropped.delete();
      }
    },
  },

  // ===================================================================
  // Image Arithmetic — op สองภาพนำเข้าตัวแรก (Tier 2, 18a) — ยืนยัน ctx.src2 เป็น cv.Mat ดิบที่
  // runPipeline ยืมให้ใช้ชั่วคราว ห้าม op นี้ .delete() มัน (เหมือนที่ src เองก็ไม่เคย delete ทุก op)
  // ===================================================================
  imabsdiff: {
    name: "Absolute Difference (|A-B|)",
    needs: "any",
    desc: "หาความต่างสัมบูรณ์ระหว่างภาพสองใบทีละพิกเซล (|A−B|) — บริเวณที่เหมือนกันจะได้ค่าใกล้ 0 (ดำ) บริเวณที่ต่างกันจะสว่างขึ้นตามระดับความต่าง ใช้ตรวจจับความเคลื่อนไหว/ความเปลี่ยนแปลงระหว่างสองภาพ (เช่น ภาพก่อน-หลัง หรือเฟรมปัจจุบันเทียบกับพื้นหลัง) ต่อสาย 'เข้า B' ด้านล่างของกล่องนี้เพื่อกำหนดภาพที่สอง",
    params: [],
    apply(cv, src, p, ctx) {
      if (!ctx || !ctx.src2) {
        // ยังไม่ได้ต่อสาย B — คืนภาพเดิมพร้อมข้อความอธิบาย แทนการ throw (เหมือนกรณี "ไม่พบจุดเด่น" ของรอบก่อน)
        // — มี Mat 2 ตัว (dst/rgb) ต้อง try/catch/finally ตาม Iron Rule ข้อ 1 เหมือนทุก branch อื่นในไฟล์นี้
        let dst = null, rgb = null;
        try {
          dst = new cv.Mat();
          rgb = new cv.Mat();
          if (src.channels() === 1) cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB); else src.copyTo(rgb);
          rgb.copyTo(dst);
          if (ctx) ctx.info = "ยังไม่ได้ต่อสาย 'เข้า B' — ลากเส้นจากอีกจุดหนึ่งมาต่อที่ช่องอินพุตที่สองของกล่องนี้";
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          if (rgb) rgb.delete();
        }
      }
      // CV-eng recheck (18a): ถ้าภาพทั้งสองเป็นขาวดำอยู่แล้ว (เช่น Grayscale→Gaussian เทียบ Grayscale→Median
      // ตามตัวอย่างการเรียนการสอนทั่วไป) ให้ absdiff ตรง ๆ แบบ 1-channel เลย ไม่ต้อง upconvert เป็น RGB
      // ก่อน — เดิมโค้ด upconvert ทั้งคู่เป็น RGB เสมอแม้ทั้งคู่จะเป็นขาวดำอยู่แล้ว (ได้ผลถูกต้องเชิงตัวเลข
      // เพราะ R=G=B เท่ากันหมด แต่ผลลัพธ์กลายเป็นภาพสี 3 ช่องทั้งที่ไม่มีข้อมูลสีจริง ผู้เรียนที่ export โค้ด
      // Python ไปดู img.shape จะงงว่าทำไมเป็น (h,w,3) ทั้งที่ไล่ Grayscale มาทั้งสองสาย) — เก็บ branch เดิม
      // ไว้สำหรับกรณีที่มีอย่างน้อยหนึ่งภาพเป็นสี (ยังต้อง upconvert ให้ channel ตรงกันก่อนเทียบเหมือนเดิม)
      const bothGray = src.channels() === 1 && ctx.src2.channels() === 1;
      if (bothGray) {
        let matched = null, dst = null;
        try {
          const { mat: b, resized } = matchSize(cv, src, ctx.src2);
          matched = resized ? b : null;
          dst = new cv.Mat();
          cv.absdiff(src, matched || ctx.src2, dst);
          if (ctx) ctx.info = resized
            ? "ความต่างสัมบูรณ์ |A-B| (ขาวดำ, ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)"
            : "ความต่างสัมบูรณ์ |A-B| (ขาวดำ)";
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          if (matched) matched.delete();
        }
      }
      let rgbA = null, rgbB = null, matched = null, dst = null;
      try {
        rgbA = new cv.Mat();
        if (src.channels() === 1) cv.cvtColor(src, rgbA, cv.COLOR_GRAY2RGB); else src.copyTo(rgbA);
        rgbB = new cv.Mat();
        if (ctx.src2.channels() === 1) cv.cvtColor(ctx.src2, rgbB, cv.COLOR_GRAY2RGB); else ctx.src2.copyTo(rgbB);
        const { mat: b, resized } = matchSize(cv, rgbA, rgbB);
        matched = resized ? b : null; // เฉพาะกรณี resize แล้วต้อง delete เอง (ไม่ใช่กรณีใช้ rgbB ตรง ๆ)
        dst = new cv.Mat();
        cv.absdiff(rgbA, matched || rgbB, dst);
        if (ctx) ctx.info = resized
          ? "ความต่างสัมบูรณ์ |A-B| (ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)"
          : "ความต่างสัมบูรณ์ |A-B|";
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (rgbA) rgbA.delete();
        if (rgbB) rgbB.delete();
        if (matched) matched.delete();
      }
    },
  },

  // Tier 2, 18b — สองภาพเข้าตัวที่ 2 ของกลุ่ม Image Arithmetic (image_arithmetic FILTERS group เดิมจาก
  // 18a มี inputs:2 อยู่แล้ว แค่เพิ่ม opId เข้า opIds list)
  imfuse: {
    name: "Fuse / Blend Two Images",
    needs: "any",
    desc: "ซ้อนภาพสองใบเข้าด้วยกันเพื่อเทียบความต่าง — เลือกได้ว่าจะผสมแบบโปร่งแสง (blend, เห็นทั้งคู่จางๆ ซ้อนกัน) หรือแบบตารางหมากรุก (checkerboard, สลับภาพ A/B เป็นตาราง ช่วยเทียบขอบต่อขอบ) ต่อสาย 'เข้า B' เพื่อกำหนดภาพที่สอง",
    params: [
      { key: "mode", type: "select", label: "รูปแบบการซ้อน", options: [
        { value: "blend", label: "โปร่งแสง (Blend)" },
        { value: "checkerboard", label: "ตารางหมากรุก (Checkerboard)" },
      ], default: "blend" },
      { key: "alpha", visibleWhen: [{key:"mode",in:["blend"]}], type: "number", label: "น้ำหนักภาพ A (%) — ใช้กับโหมดโปร่งแสงเท่านั้น", min: 0, max: 100, step: 5, default: 50, desc: "100%=เห็นแค่ A, 0%=เห็นแค่ B, 50%=ผสมเท่ากัน" },
      { key: "tileSize", visibleWhen: [{key:"mode",in:["checkerboard"]}], type: "number", label: "ขนาดตาราง (px) — ใช้กับโหมด Checkerboard เท่านั้น", min: 8, max: 128, step: 8, default: 32 },
    ],
    apply(cv, src, p, ctx) {
      if (!ctx || !ctx.src2) {
        // ยังไม่ได้ต่อสาย B — คืนภาพเดิมพร้อมข้อความอธิบาย (pattern เดียวกับ imabsdiff ทุกจุด)
        let dst = null, rgb = null;
        try {
          dst = new cv.Mat();
          rgb = new cv.Mat();
          if (src.channels() === 1) cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB); else src.copyTo(rgb);
          rgb.copyTo(dst);
          if (ctx) ctx.info = "ยังไม่ได้ต่อสาย 'เข้า B' — ลากเส้นจากอีกจุดหนึ่งมาต่อที่ช่องอินพุตที่สองของกล่องนี้";
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          if (rgb) rgb.delete();
        }
      }
      // bothGray fast path เดียวกับ imabsdiff (18a recheck) — ไม่ upconvert เป็น RGB โดยไม่จำเป็นถ้าทั้งคู่
      // ขาวดำอยู่แล้ว (เช่น เทียบ Grayscale→Threshold สองสาย) ผลลัพธ์จะได้ภาพขาวดำต่อ ไม่ใช่สี 3 ช่องปลอม ๆ
      const bothGray = src.channels() === 1 && ctx.src2.channels() === 1;
      let a = null, bRaw = null, matched = null, mask = null, maskInv = null, tmpA = null, tmpB = null, dst = null;
      try {
        a = new cv.Mat();
        if (bothGray || src.channels() > 1) src.copyTo(a); else cv.cvtColor(src, a, cv.COLOR_GRAY2RGB);
        bRaw = new cv.Mat();
        if (bothGray || ctx.src2.channels() > 1) ctx.src2.copyTo(bRaw); else cv.cvtColor(ctx.src2, bRaw, cv.COLOR_GRAY2RGB);
        const { mat: bm, resized } = matchSize(cv, a, bRaw);
        matched = resized ? bm : null;
        const b = matched || bRaw;
        const infoSuffix = resized ? " (ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)" : "";

        if (p.mode === "checkerboard") {
          const tile = Math.max(1, Math.round(finiteOr(p.tileSize, 32)));
          // ไม่มี cv call เดียวที่ทำ "checkerboard compose" ตรง ๆ — สร้าง mask เอง แล้ว bitwise_and 2 ครั้ง
          // + add (ตามที่ยืนยันไว้แล้วว่าเป็นวิธีมาตรฐานสำหรับ opencv.js — ดู buildCheckerboardMask ด้านบน)
          mask = buildCheckerboardMask(cv, a.cols, a.rows, tile);
          maskInv = new cv.Mat();
          cv.bitwise_not(mask, maskInv);
          tmpA = new cv.Mat();
          tmpB = new cv.Mat();
          cv.bitwise_and(a, a, tmpA, mask);
          cv.bitwise_and(b, b, tmpB, maskInv);
          dst = new cv.Mat();
          cv.add(tmpA, tmpB, dst);
          if (ctx) ctx.info = `ซ้อนแบบตารางหมากรุก (ขนาดตาราง ${tile}px)` + infoSuffix;
        } else {
          const alphaPct = Math.max(0, Math.min(100, finiteOr(p.alpha, 50)));
          const alpha = alphaPct / 100;
          dst = new cv.Mat();
          cv.addWeighted(a, alpha, b, 1 - alpha, 0, dst);
          if (ctx) ctx.info = `ผสมโปร่งแสง A ${Math.round(alphaPct)}% / B ${Math.round(100 - alphaPct)}%` + infoSuffix;
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (a) a.delete();
        if (bRaw) bRaw.delete();
        if (matched) matched.delete();
        if (mask) mask.delete();
        if (maskInv) maskInv.delete();
        if (tmpA) tmpA.delete();
        if (tmpB) tmpB.delete();
      }
    },
  },

  // Tier 2, 18b — สามภาพเข้าตัวที่ 3 ของกลุ่ม Image Arithmetic — B ถูกใช้เป็น "หน้ากาก" (mask) ไม่ใช่ภาพเต็ม
  // (threshold B ให้เป็นขาวดำล้วนภายในก่อนเสมอ — เหมือนธรรมเนียม fill_holes ในไฟล์นี้ที่ binarize ที่ 127)
  mask: {
    name: "Mask (กรองผ่านภาพที่สอง)",
    needs: "any",
    desc: "ใช้ภาพที่สอง (B) เป็น 'หน้ากาก' (mask) กรองภาพหลัก (A) — พิกเซลของ B ที่สว่างกว่าเกณฑ์กึ่งกลาง (127) ถือเป็น 'เปิด' (คงพิกเซล A ไว้ตามเดิม) ส่วนที่มืดกว่าจะถูกทำให้ดำ เหมือนการตัดสเตนซิลภาพ A ด้วยรูปทรงของ B — เหมาะกับการใช้ผลลัพธ์จาก Threshold/Find Contours อีกสายหนึ่งมาเป็น mask ต่อสาย 'เข้า B' เพื่อกำหนดภาพ mask",
    params: [
      { key: "invert", type: "boolean", label: "กลับด้าน mask (invert)", default: false, desc: "เปิด = สลับ 'เปิด'/'ปิด' ของ mask (กรองเอาส่วนที่เดิมจะถูกซ่อนไว้แทน)" },
    ],
    apply(cv, src, p, ctx) {
      if (!ctx || !ctx.src2) {
        let dst = null, rgb = null;
        try {
          dst = new cv.Mat();
          rgb = new cv.Mat();
          if (src.channels() === 1) cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB); else src.copyTo(rgb);
          rgb.copyTo(dst);
          if (ctx) ctx.info = "ยังไม่ได้ต่อสาย 'เข้า B' — ลากเส้นจากอีกจุดหนึ่งมาต่อที่ช่องอินพุตที่สองของกล่องนี้";
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          if (rgb) rgb.delete();
        }
      }
      let gray2 = null, matched = null, bin = null, inv = null, dst = null;
      try {
        gray2 = new cv.Mat();
        if (ctx.src2.channels() === 1) ctx.src2.copyTo(gray2); else cv.cvtColor(ctx.src2, gray2, cv.COLOR_RGB2GRAY);
        const { mat: bm, resized } = matchSize(cv, src, gray2);
        matched = resized ? bm : null;
        const bSrc = matched || gray2;
        bin = new cv.Mat();
        cv.threshold(bSrc, bin, 127, 255, cv.THRESH_BINARY);
        let finalMask = bin;
        if (boolOr(p.invert, false)) {
          inv = new cv.Mat();
          cv.bitwise_not(bin, inv);
          finalMask = inv;
        }
        dst = new cv.Mat();
        cv.bitwise_and(src, src, dst, finalMask);
        const kept = cv.countNonZero(finalMask);
        if (ctx) ctx.info = `กรอง A ผ่านหน้ากาก B${boolOr(p.invert, false) ? " (กลับด้าน)" : ""} — เปิด ${kept} px` +
          (resized ? " (ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)" : "");
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (gray2) gray2.delete();
        if (matched) matched.delete();
        if (bin) bin.delete();
        if (inv) inv.delete();
      }
    },
  },

  // ===================================================================
  // IP batch 2 ชิ้น B (2026-10-02) — op สองภาพเพิ่มเติมในกลุ่ม Image Arithmetic (ต่อท้าย opIds เดิม)
  // ทุกตัวใช้ passthroughNoSrc2 (ด้านบนใกล้ matchSize) จัดการกรณียังไม่มีภาพ B · ctx.src2 เป็นของยืม ห้าม delete
  // ===================================================================
  imarith: {
    name: "Add / Subtract / Multiply / Divide (A ∘ B)",
    needs: "any",
    desc: "คำนวณเลขคณิตระหว่างภาพสองใบทีละพิกเซล — ผลเป็นภาพ 8 บิต ค่าที่เกิน 255 ถูกตัดที่ 255 และค่าที่ติดลบถูกตัดเป็น 0 (ไม่วนกลับ) · บวก: รวมความสว่างสองภาพ · ลบ: A − B ตรงที่ B สว่างกว่า A จะได้ 0 (ถ้าต้องการความต่างทั้งสองทางให้ใช้ Absolute Difference) · คูณ: A × B ÷ 255 ตรงที่ B ขาว (255) ได้ค่า A เดิม ตรงที่ B ดำได้ 0 จึงใช้ B เป็นหน้ากากแบบนุ่มได้ · หาร: A ÷ B × 255 ตรงที่ B = 0 ได้ 0 (พฤติกรรมของ OpenCV) · บวก/ลบให้ผลเท่า imadd/imsubtract ของ MATLAB แต่คูณ/หารต่างกัน: immultiply คิด A × B แล้วตัดที่ 255, imdivide คิด A ÷ B (A > 0 กับ B = 0 ได้ 255) — ของแอปสเกลด้วย 255 ให้ใช้ B เป็นหน้ากากได้ · ถ้าภาพหนึ่งเป็นสีอีกภาพเป็นขาวดำ จะแปลงทั้งคู่เป็นภาพสีก่อน ต่อสาย 'เข้า B' เพื่อกำหนดภาพที่สอง",
    params: [
      { key: "operation", type: "select", label: "การคำนวณ", options: [
        { value: "add", label: "บวก (A + B)", hint: "เกิน 255 ตัดที่ 255" },
        { value: "subtract", label: "ลบ (A − B)", hint: "ติดลบตัดเป็น 0 — ต้องการความต่างทั้งสองทางใช้ Absolute Difference" },
        { value: "multiply", label: "คูณ (A × B ÷ 255)", hint: "B ขาว = คงค่า A · B ดำ = 0" },
        { value: "divide", label: "หาร (A ÷ B × 255)", hint: "ตรงที่ B = 0 ได้ 0" },
      ], default: "add" },
    ],
    apply(cv, src, p, ctx) {
      const pass = passthroughNoSrc2(src, ctx);
      if (pass) return pass;
      if (isEmptyMat(src)) { ctx.info = "ภาพว่าง"; return src.clone(); }
      const op = Object.hasOwn(IMARITH_OPS, p.operation) ? IMARITH_OPS[p.operation] : IMARITH_OPS.add;
      // ทั้งคู่ขาวดำ → คำนวณ 1 ช่องเลย (fast path แบบ imabsdiff) · นอกนั้นแปลงทั้งคู่เป็น RGB ก่อน
      const bothGray = src.channels() === 1 && ctx.src2.channels() === 1;
      let rgbA = null, rgbB = null, matched = null, dst = null;
      try {
        if (!bothGray) {
          rgbA = rgbIfNeeded(cv, src);
          rgbB = rgbIfNeeded(cv, ctx.src2);
        }
        const a = rgbA || src;
        const bRaw = rgbB || ctx.src2;
        const { mat: bm, resized } = matchSize(cv, a, bRaw);
        matched = resized ? bm : null;
        dst = new cv.Mat();
        op.run(cv, a, matched || bRaw, dst);
        ctx.info = op.info + (bothGray ? " (ขาวดำ)" : "") + (resized ? RESIZED_B_INFO : "");
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (rgbA) rgbA.delete();
        if (rgbB) rgbB.delete();
        if (matched) matched.delete();
      }
    },
  },

  logical_ops: {
    name: "Logical AND / OR / XOR",
    needs: "any",
    desc: "รวมภาพขาวดำ (binary) สองใบด้วยตรรกะทีละพิกเซล — ก่อนคำนวณ ทั้ง A และ B จะถูกแปลงเป็นขาวดำแล้วตัดที่ 127 เสมอ (สว่างกว่า 127 = ขาว, 127 หรือมืดกว่า = ดำ) ภาพที่ยังไม่ใช่ขาวดำล้วนจึงถูกตัดที่ 127 ก่อน ถ้าต้องการคุมเกณฑ์เองให้ต่อ Threshold มาก่อนทั้งสองสาย และถ้าต่อมาจาก Color InRange ให้ตั้ง 'แสดงผลเป็น' = Mask ขาวดำ ก่อน (ค่าเริ่มต้น 'เฉพาะส่วนที่เป็นสีนั้น' เป็นภาพสี สีเข้มเช่นแดงหรือน้ำเงินมีค่าเทาต่ำกว่า 127 จึงกลายเป็นดำ) · AND = ขาวเฉพาะที่ขาวทั้งคู่ · OR = ขาวถ้าขาวอย่างน้อยหนึ่งภาพ · XOR = ขาวเฉพาะที่ขาวภาพเดียว · A AND NOT B = หัก B ออกจาก A · ตัวอย่างคลาสสิก: สีแดงใน HSV มีค่า H อยู่สองฝั่งของวงล้อ (ราว 0–10 และ 170–179) โค้ด OpenCV ทั่วไปจึงทำ inRange 2 ช่วงแล้ว OR กัน (Color InRange ของแอปนี้รวมสองช่วงให้เองเมื่อตั้ง H ต่ำสุดมากกว่า H สูงสุด — แต่ถ้าจะรวมสีที่ต่างกันจริง เช่น แดง OR เหลือง ให้ต่อ InRange สองสาย ตั้งแสดงผลเป็น Mask ขาวดำทั้งคู่ แล้วรวมด้วย OR ที่นี่) ต่อสาย 'เข้า B' เพื่อกำหนดภาพที่สอง",
    params: [
      { key: "operation", type: "select", label: "ตรรกะ", options: [
        { value: "and", label: "AND (ขาวทั้งคู่)", hint: "ขาวเฉพาะที่ A และ B ขาวทั้งคู่" },
        { value: "or", label: "OR (ขาวภาพใดก็ได้)", hint: "ขาวถ้า A หรือ B ขาวอย่างน้อยหนึ่งภาพ" },
        { value: "xor", label: "XOR (ขาวภาพเดียว)", hint: "ขาวเฉพาะที่ A กับ B ต่างกัน" },
        { value: "a_and_not_b", label: "A AND NOT B (หัก B ออกจาก A)", hint: "ขาวเฉพาะที่ A ขาวแต่ B ดำ" },
      ], default: "and" },
    ],
    apply(cv, src, p, ctx) {
      const pass = passthroughNoSrc2(src, ctx);
      if (pass) return pass;
      if (isEmptyMat(src)) { ctx.info = "ภาพว่าง"; return src.clone(); }
      const op = Object.hasOwn(LOGICAL_OPS, p.operation) ? LOGICAL_OPS[p.operation] : LOGICAL_OPS.and;
      let grayA = null, grayB = null, matched = null, binA = null, binB = null, notB = null, dst = null;
      try {
        grayA = grayIfNeeded(cv, src);
        grayB = grayIfNeeded(cv, ctx.src2);
        // ลำดับเดียวกับ op mask: B แปลงเป็นขาวดำ → ปรับขนาด → threshold 127
        const { mat: bm, resized } = matchSize(cv, src, grayB || ctx.src2);
        matched = resized ? bm : null;
        binA = new cv.Mat();
        cv.threshold(grayA || src, binA, 127, 255, cv.THRESH_BINARY);
        binB = new cv.Mat();
        cv.threshold(matched || grayB || ctx.src2, binB, 127, 255, cv.THRESH_BINARY);
        dst = new cv.Mat();
        if (op.notB) {
          notB = new cv.Mat();
          cv.bitwise_not(binB, notB);
        }
        op.run(cv, binA, notB || binB, dst);
        const colourIn = src.channels() !== 1 || ctx.src2.channels() !== 1;
        ctx.info = `${op.info} · ขาว ${cv.countNonZero(dst)} px` + (resized ? RESIZED_B_INFO : "") + (colourIn ? COLOUR_CUT_INFO : "");
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (grayA) grayA.delete();
        if (grayB) grayB.delete();
        if (matched) matched.delete();
        if (binA) binA.delete();
        if (binB) binB.delete();
        if (notB) notB.delete();
      }
    },
  },

  inpaint: {
    name: "Inpaint (ซ่อมบริเวณตาม mask B)",
    needs: "any",
    desc: "ซ่อมรอยตำหนิเล็ก ๆ (รอยขีด จุดฝุ่น ตัวหนังสือทับภาพ) โดยเติมบริเวณนั้นด้วยสีจากพิกเซลรอบ ๆ (เหมือน inpaint/regionfill ของ MATLAB) — ภาพ B คือ mask: พิกเซลของ B ที่สว่างกว่า 127 คือบริเวณที่ต้องซ่อม (B ที่เป็นภาพสีจะถูกแปลงเป็นขาวดำก่อน — ถ้า B มาจาก Color InRange ให้ตั้ง 'แสดงผลเป็น' = Mask ขาวดำ) ปกติได้ B มาจากสายที่ Threshold รอยที่ต้องการไว้ · Telea เติมจากขอบเข้าไปด้านในโดยถ่วงน้ำหนักตามระยะ · Navier-Stokes ต่อแนวเส้นของภาพเข้าไปในบริเวณที่ซ่อม · ใช้กับรอยเล็กเท่านั้น: ถ้า mask ครอบเกิน 25% ของภาพ จะไม่ซ่อมและคืนภาพเดิม (เวลาที่ใช้โตตามพื้นที่ mask × รัศมี² — บริเวณกว้างบนภาพความละเอียดสูงอาจใช้เป็นวินาทีต่อเฟรมจนกล้องสดค้าง และบริเวณกว้างมักถูกเติมเป็นสีเรียบ ๆ ไม่เหลือรายละเอียด) ต่อสาย 'เข้า B' เพื่อกำหนด mask",
    params: [
      { key: "method", type: "select", label: "วิธีซ่อม", options: [
        { value: "telea", label: "Telea (Fast Marching)", hint: "เติมจากขอบเข้าด้านใน ถ่วงน้ำหนักตามระยะ" },
        { value: "ns", label: "Navier-Stokes", hint: "ต่อแนวเส้น/ขอบของภาพเข้าไปในบริเวณที่ซ่อม" },
      ], default: "telea" },
      { key: "radius", type: "number", label: "รัศมีอ้างอิง (px)", min: 1, max: 7, step: 1, default: 3, hint: "ระยะรอบจุดที่ใช้เดาสี — ใหญ่ = ช้าลงมาก · วัดจริงที่ r=7 mask ครอบ 25% ภาพสี (Node, OpenCV.js 4.10): 640×480 ≈ 0.12–0.29 วินาที · 1280×720 ≈ 0.37–0.86 วินาทีต่อเฟรม ขึ้นกับรูปทรงของ mask (จุดกระจายช้ากว่าก้อนเดียว)", desc: "รัศมีของบริเวณรอบ ๆ ที่ใช้คำนวณสีที่จะเติม — รอยเล็กใช้ 3 ก็พอ" },
    ],
    apply(cv, src, p, ctx) {
      const pass = passthroughNoSrc2(src, ctx);
      if (pass) return pass;
      if (isEmptyMat(src)) { ctx.info = "ภาพว่าง"; return src.clone(); } // cv.inpaint throw จริงบน 0×0
      const radius = intIn(p.radius, 3, 1, 7);
      const useNS = p.method === "ns";
      let grayB = null, matched = null, maskBin = null, rgb = null, dst = null;
      try {
        grayB = grayIfNeeded(cv, ctx.src2);
        const { mat: bm, resized } = matchSize(cv, src, grayB || ctx.src2);
        matched = resized ? bm : null;
        maskBin = new cv.Mat();
        cv.threshold(matched || grayB || ctx.src2, maskBin, 127, 255, cv.THRESH_BINARY);
        const suffix = (resized ? RESIZED_B_INFO : "") + (ctx.src2.channels() !== 1 ? COLOUR_CUT_INFO : "");
        const nz = cv.countNonZero(maskBin);
        const total = src.rows * src.cols;
        if (nz === 0) {
          ctx.info = "mask ว่าง ไม่มีบริเวณให้ซ่อม (ภาพ B ไม่มีพิกเซลที่สว่างกว่า 127)" + suffix;
          return src.clone();
        }
        // เทียบด้วยจำนวนเต็มล้วน (ไม่หารเป็นทศนิยม) ให้โค้ด Python ที่ export ตัดสินเหมือนกันเป๊ะ · % ในข้อความปัดขึ้น
        // (ไม่โชว์ "25.0% เกิน 25%") — Python ใช้ -(-a // b) แบบเดียวกัน
        if (nz * 100 > INPAINT_MAX_MASK_PCT * total) {
          ctx.info = `ไม่ซ่อม: mask ครอบ ${(Math.ceil((nz * 1000) / total) / 10).toFixed(1)}% ของภาพ (เกิน ${INPAINT_MAX_MASK_PCT}%) — Inpaint ใช้ซ่อมรอยเล็ก ๆ ให้ใส่ Threshold ในสาย B จนเหลือเฉพาะรอยที่ต้องการซ่อมก่อน` + suffix;
          return src.clone();
        }
        // cv.inpaint รับเฉพาะ 1 หรือ 3 ช่อง (throw กับ RGBA) → ภาพ 4 ช่องแปลงเป็น RGB ก่อน (ผลจึงเป็น 3 ช่อง)
        if (src.channels() === 4) {
          rgb = new cv.Mat();
          cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
        }
        dst = new cv.Mat();
        cv.inpaint(rgb || src, maskBin, dst, radius, useNS ? cv.INPAINT_NS : cv.INPAINT_TELEA);
        ctx.info = `ซ่อม ${nz} px (${(Math.floor((nz * 1000) / total + 0.5) / 10).toFixed(1)}% ของภาพ) ด้วย ${useNS ? "Navier-Stokes" : "Telea"} r=${radius}` + suffix;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (grayB) grayB.delete();
        if (matched) matched.delete();
        if (maskBin) maskBin.delete();
        if (rgb) rgb.delete();
      }
    },
  },

  hist_match: {
    name: "Histogram Matching (ปรับโทนให้เหมือนภาพ B)",
    needs: "any",
    desc: "ปรับการกระจายความสว่างของภาพ A ให้มี histogram ใกล้เคียงภาพ B (เช่น ทำให้ภาพที่ถ่ายต่างแสงกันมีโทนใกล้กันก่อนนำไปเทียบ) — ใช้การจับคู่ CDF แบบคลาสสิก 256 ระดับ (Gonzalez & Woods): ค่า i ของ A ถูกแทนด้วยค่า j แรกที่สัดส่วนสะสมของ B ถึงสัดส่วนสะสมของ A ที่ i · ไม่ใช่สูตรเดียวกับ imhistmatch ของ MATLAB (ค่าเริ่มต้นของ MATLAB ใช้ 64 ระดับ) ผลจึงอาจต่างจาก MATLAB ได้ · ภาพสีทั้งคู่ = จับคู่ทีละช่อง R/G/B · A สี B ขาวดำ = ใช้ histogram ขาวดำของ B กับทุกช่อง · A ขาวดำ = แปลง B เป็นขาวดำก่อน · ขนาดของ B ไม่จำเป็นต้องเท่า A เพราะใช้แค่สัดส่วนของ histogram ต่อสาย 'เข้า B' เพื่อกำหนดภาพอ้างอิง",
    params: [],
    apply(cv, src, p, ctx) {
      const pass = passthroughNoSrc2(src, ctx);
      if (pass) return pass;
      if (isEmptyMat(src)) { ctx.info = "ภาพว่าง"; return src.clone(); }
      let rgbA = null, refConv = null, aVec = null, bVec = null, mask = null, hist = null, lut = null, dst = null;
      try {
        if (src.channels() === 4) { // A RGBA → RGB (alpha ไม่ใช่ข้อมูลโทนสี)
          rgbA = new cv.Mat();
          cv.cvtColor(src, rgbA, cv.COLOR_RGBA2RGB);
        }
        const a = rgbA || src;
        const nCh = a.channels(); // 1 หรือ 3
        // การจับคู่ช่อง: A ขาวดำ → B แปลงเป็นขาวดำ · A สี → B สีใช้ช่องต่อช่อง (RGBA→RGB ก่อน) / B ขาวดำใช้ช่องเดียวกับทุกช่อง
        if (nCh === 1) refConv = grayIfNeeded(cv, ctx.src2);
        else if (ctx.src2.channels() === 4) refConv = rgbIfNeeded(cv, ctx.src2);
        const ref = refConv || ctx.src2;
        const refGray = ref.channels() === 1;
        aVec = new cv.MatVector();
        aVec.push_back(a);
        bVec = new cv.MatVector();
        bVec.push_back(ref);
        mask = new cv.Mat(); // mask ว่าง = ใช้ทุกพิกเซล
        hist = new cv.Mat();
        const lutArr = new Uint8Array(256 * nCh); // LUT 1×256 แบบ interleave (ช่อง c ของระดับ i อยู่ที่ i*nCh+c)
        for (let c = 0; c < nCh; c++) {
          cv.calcHist(aVec, [c], mask, hist, [256], [0, 256]);
          const cdfA = cdf256(hist.data32F);
          cv.calcHist(bVec, [refGray ? 0 : c], mask, hist, [256], [0, 256]);
          const cdfB = cdf256(hist.data32F);
          // LUT[i] = j แรกที่ cdfB[j] >= cdfA[i] (ไม่เกิน 255) — cdfA ไม่ลดลงเมื่อ i เพิ่ม จึงเดิน j ต่อจากเดิมได้
          // (ผลเท่ากับ np.searchsorted(cdf_b, cdf_a, side='left') ที่ clip 255 ในโค้ด Python ที่ export)
          let j = 0;
          for (let i = 0; i < 256; i++) {
            while (j < 255 && cdfB[j] < cdfA[i]) j++;
            lutArr[i * nCh + c] = j;
          }
        }
        lut = cv.matFromArray(1, 256, nCh === 1 ? cv.CV_8UC1 : cv.CV_8UC3, lutArr);
        dst = new cv.Mat();
        cv.LUT(a, lut, dst);
        ctx.info = "ปรับ histogram ของ A ให้เหมือน B " + (nCh === 1 ? "(ขาวดำ)" : refGray ? "(3 ช่อง ใช้ histogram ขาวดำของ B กับทุกช่อง)" : "(3 ช่อง แยกช่อง)");
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (rgbA) rgbA.delete();
        if (refConv) refConv.delete();
        if (aVec) aVec.delete();
        if (bVec) bVec.delete();
        if (mask) mask.delete();
        if (hist) hist.delete();
        if (lut) lut.delete();
      }
    },
  },

  // ===================================================================
  // Template Matching (Tier 2, 18b) — กลุ่มเดี่ยวใหม่ inputs:2 (primary = ภาพค้นหา, src2 = template)
  // ===================================================================
  template_matching: {
    name: "Normalized Cross-Correlation",
    needs: "gray",
    desc: "หาตำแหน่งของ 'ภาพตัวอย่างเล็ก' (template) ในภาพใหญ่ (search image) โดยเลื่อน template ไปทาบทั่วภาพแล้ววัดความคล้ายที่แต่ละตำแหน่ง — จุดที่คล้ายที่สุดคือคำตอบ ใช้หาตำแหน่งชิ้นงาน/โลโก้/สัญลักษณ์ที่รู้หน้าตาแน่นอนอยู่แล้วในภาพกล้อง ต่อสาย 'เข้า B' ด้วยภาพ template (ต้องเล็กกว่าภาพหลักทั้งกว้างและสูง)",
    params: [
      { key: "method", type: "select", label: "วิธีวัดความคล้าย", options: [
        { value: "ccoeff", label: "Correlation Coefficient (แนะนำ, ทนแสงเปลี่ยนได้ดี)" },
        { value: "ccorr", label: "Cross Correlation (เร็วกว่า แต่ไวต่อความสว่าง)" },
      ], default: "ccoeff" },
      { key: "threshold", type: "number", label: "เกณฑ์ความมั่นใจขั้นต่ำ (%)", min: 0, max: 100, step: 1, default: 70, desc: "ถ้าค่าที่ดีที่สุดต่ำกว่านี้ จะถือว่า 'ไม่พบ' แทนที่จะวาดกรอบผิดตำแหน่ง" },
    ],
    apply(cv, src, p, ctx) {
      // src มาเป็นขาวดำแล้วเสมอ (needs:"gray" — worker แปลงอัตโนมัติ) แต่ ctx.src2 (template) มาดิบเสมอ
      // (worker ไม่รู้ว่า op นี้อยากได้ B เป็นขาวดำด้วย — ต้องแปลงเอง ตาม Iron Rule ข้อ 2)
      if (!ctx || !ctx.src2) {
        let dst = null, rgb = null;
        try {
          dst = new cv.Mat();
          rgb = new cv.Mat();
          cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
          rgb.copyTo(dst);
          if (ctx) ctx.info = "ยังไม่ได้ต่อสาย 'เข้า B' (ภาพ template) — ลากเส้นจากอีกจุดหนึ่งมาต่อที่ช่องอินพุตที่สองของกล่องนี้";
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          if (rgb) rgb.delete();
        }
      }
      let template = null, rgbOut = null, analysisSearch = null, analysisTemplate = null, result = null,
        roi = null, resultFull = null, dst = null;
      try {
        template = new cv.Mat();
        if (ctx.src2.channels() === 1) ctx.src2.copyTo(template); else cv.cvtColor(ctx.src2, template, cv.COLOR_RGB2GRAY);

        rgbOut = new cv.Mat();
        cv.cvtColor(src, rgbOut, cv.COLOR_GRAY2RGB);

        // การ์ดสำคัญ (ยืนยันเชิงประจักษ์ก่อนเขียนโค้ดนี้): cv.matchTemplate ไม่ throw เมื่อ template ใหญ่กว่า
        // search image — คืนผลลัพธ์ขนาดเพี้ยนแทน (ทดสอบจริง: search 20x20 + template 50x50 ได้ result
        // 31x31 ที่ไม่มีความหมายเลย ไม่ error อะไรทั้งสิ้น) ต้องเช็คมิติเองก่อนเรียกเสมอ ไม่ใช่ทางเลือกเสริม
        if (template.cols > src.cols || template.rows > src.rows) {
          dst = new cv.Mat();
          rgbOut.copyTo(dst);
          if (ctx) ctx.info = "ภาพ template (เข้า B) ต้องเล็กกว่าภาพหลักทั้งกว้างและสูง — ลองใช้ template เล็กลง";
          return dst;
        }

        // ย่อภาพค้นหา + template ไปพร้อมกันด้วย scale เดียวกัน (วัดจริง: matchTemplate เต็ม 1080p กับ
        // template 100x100 ~70-200ms ช้าเกินสำหรับกล้องสด, ย่อเหลือ ≤640px ด้านยาว ~6ms) แล้วขยาย matchLoc
        // กลับด้วย invScale ก่อนวาดกรอบ
        const maxAnalysisSide = 640;
        const { scale, invScale, w: aw, h: ah } = analysisScale(src.cols, src.rows, maxAnalysisSide);
        analysisSearch = new cv.Mat();
        analysisTemplate = new cv.Mat();
        if (scale < 1) {
          const tw = Math.max(1, Math.round(template.cols * scale));
          const th = Math.max(1, Math.round(template.rows * scale));
          cv.resize(src, analysisSearch, new cv.Size(aw, ah), 0, 0, cv.INTER_AREA);
          cv.resize(template, analysisTemplate, new cv.Size(tw, th), 0, 0, cv.INTER_AREA);
        } else {
          src.copyTo(analysisSearch);
          template.copyTo(analysisTemplate);
        }
        // template ย่อแล้วอาจเกินภาพค้นหาย่อแล้วไปได้เล็กน้อยจากการปัดเศษที่ไม่ตรงกันเป๊ะ (เคสขอบ ๆ) — กันซ้ำ
        if (analysisTemplate.cols > analysisSearch.cols || analysisTemplate.rows > analysisSearch.rows) {
          dst = new cv.Mat();
          rgbOut.copyTo(dst);
          if (ctx) ctx.info = "ภาพ template (เข้า B) ต้องเล็กกว่าภาพหลักทั้งกว้างและสูง — ลองใช้ template เล็กลง";
          return dst;
        }

        const methodConst = p.method === "ccorr" ? cv.TM_CCORR_NORMED : cv.TM_CCOEFF_NORMED;
        result = new cv.Mat();
        cv.matchTemplate(analysisSearch, analysisTemplate, result, methodConst);
        // cv.minMaxLoc: tie-break ระหว่างตำแหน่งคะแนนเท่ากันเป็นคุณสมบัติของอัลกอริทึม C++ ต้นฉบับเอง (สแกน
        // แบบ row-major แล้วอัปเดตเฉพาะตอนพบค่า "มากกว่าเดิมอย่างเคร่งครัด" — จุดแรกที่สแกนเจอค่าสูงสุดจึง
        // ชนะเสมอ แบบ deterministic) ทั้ง opencv.js และ cv2 ผูกกับ core C++ เดียวกัน จึงได้ตำแหน่งเดียวกันเป๊ะ
        // โดยไม่ต้องเขียน tie-break เพิ่มเติมระดับ application เอง — ต่างจาก rankKeypoints ด้านบนที่ต้องมี
        // tie-break เอง เพราะ ORB ใช้ nth_element (ลำดับไม่ deterministic ข้าม build) ไม่ใช่ full scan แบบนี้
        const mm = cv.minMaxLoc(result);
        let bestVal = mm.maxVal;
        // rescale ด้วย invScale กลับตำแหน่งเต็มความละเอียด (ค่าประมาณจาก downscaled search — refine ต่อ
        // ด้วย ROI re-score ด้านล่างถ้า scale<1)
        let fullX = Math.round(mm.maxLoc.x * invScale);
        let fullY = Math.round(mm.maxLoc.y * invScale);

        // qa รอบแก้ 18b พบว่า downscaled matchTemplate ให้ตำแหน่งแม่นพอ (คลาดเคลื่อนไม่กี่ px) แต่คะแนน
        // (match_score) คลาดเคลื่อนได้มาก (วัดจริง: exact match เต็มความละเอียด 100% → downscaled 87.7%
        // ที่ 1920x1080) มากพอจะพลิกผล ผ่าน/ไม่ผ่าน threshold เริ่มต้นในกรณีขอบ ๆ — coordinator ตัดสินใจ
        // เลือก "Option B": ใช้ downscaled search หาตำแหน่งคร่าว ๆ (เร็ว) แล้ว re-score ที่ความละเอียดเต็ม
        // เฉพาะในกรอบเล็ก ๆ (ROI) รอบตำแหน่งนั้นแทนที่จะรัน matchTemplate เต็มภาพที่ full-res ทั้งใบ (ซึ่ง
        // วัดจริงแล้วช้ากว่ามาก ~70-90ms ทั้งเฟรม vs ~11-23ms แบบ ROI ที่ 1920x1080 — ดู worklog/bench) —
        // ผลคือค่า match_score ที่แสดงและใช้ตัดสิน threshold เป็นค่าจริงที่ full-res เสมอ ไม่ใช่แค่ "แจ้งเตือน"
        // margin ของ ROI ต้องกันทั้ง 2 แหล่งคลาดเคลื่อน: (1) quantization จากการปัดเศษพิกัดใน downscaled
        // space คูณกลับด้วย invScale (สูงสุด ~invScale px) และ (2) การเลื่อนของตำแหน่งพีคจาก resize
        // interpolation เอง (วัดจริงพบสูงสุด ~8px ที่ 1920x1080/template 100px) — ใช้ margin = ceil(invScale)+8
        if (scale < 1) {
          const margin = Math.ceil(invScale) + 8;
          const roiX0 = Math.max(0, fullX - margin);
          const roiY0 = Math.max(0, fullY - margin);
          const roiX1 = Math.min(src.cols, fullX + template.cols + margin);
          const roiY1 = Math.min(src.rows, fullY + template.rows + margin);
          const roiW = roiX1 - roiX0;
          const roiH = roiY1 - roiY0;
          // ROI เล็กกว่า template เกิดได้ยากมาก (ต้องมี template เกือบเท่าภาพทั้งใบ + จุดที่เจออยู่ติดขอบ
          // พอดี จน clamp กินพื้นที่จนเหลือไม่พอ) — ถ้าเกิดขึ้นจริง ข้ามการ re-score ทิ้งไว้เป็นค่าประมาณจาก
          // downscaled search เดิม (ยังดีกว่าไม่มีผลลัพธ์เลย หรือ throw)
          if (roiW >= template.cols && roiH >= template.rows) {
            roi = src.roi(new cv.Rect(roiX0, roiY0, roiW, roiH)); // roi() คืน view ไม่ใช่ copy — ต้อง delete() เอง
            resultFull = new cv.Mat();
            cv.matchTemplate(roi, template, resultFull, methodConst);
            const mmFull = cv.minMaxLoc(resultFull);
            bestVal = mmFull.maxVal;
            fullX = roiX0 + mmFull.maxLoc.x;
            fullY = roiY0 + mmFull.maxLoc.y;
          }
        }

        const thresholdPct = finiteOr(p.threshold, 70);
        const scorePct = bestVal * 100;

        dst = new cv.Mat();
        rgbOut.copyTo(dst);
        if (scorePct < thresholdPct) {
          if (ctx) {
            ctx.info = `ไม่พบ — คะแนนดีที่สุด ${scorePct.toFixed(1)}% ต่ำกว่าเกณฑ์ ${thresholdPct}%`;
            ctx.values = { kind: "scalars", fields: [
              { key: "match_score", label: "คะแนนความคล้าย (%)", value: scorePct },
              { key: "x", label: "ตำแหน่ง X", value: null },
              { key: "y", label: "ตำแหน่ง Y", value: null },
            ] };
          }
          return dst;
        }
        cv.rectangle(dst, new cv.Point(fullX, fullY), new cv.Point(fullX + template.cols, fullY + template.rows), new cv.Scalar(0, 255, 0), 2);
        if (ctx) {
          ctx.info = `พบตำแหน่งที่ (${fullX}, ${fullY}) — คะแนนความคล้าย ${scorePct.toFixed(1)}%`;
          ctx.values = { kind: "scalars", fields: [
            { key: "match_score", label: "คะแนนความคล้าย (%)", value: scorePct },
            { key: "x", label: "ตำแหน่ง X", value: fullX },
            { key: "y", label: "ตำแหน่ง Y", value: fullY },
          ] };
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (template) template.delete();
        if (rgbOut) rgbOut.delete();
        if (analysisSearch) analysisSearch.delete();
        if (analysisTemplate) analysisTemplate.delete();
        if (result) result.delete();
        if (roi) roi.delete();
        if (resultFull) resultFull.delete();
      }
    },
  },

  // ===================================================================
  // Image Quality (Tier 2, 18b) — PSNR/SSIM กลุ่ม "Image Quality" ใหม่ inputs:2 (เทียบภาพเต็ม A vs B
  // เหมือน imabsdiff — ใช้ matchSize เดียวกัน ไม่ใช่กรณี template ที่ตั้งใจให้ขนาดต่างกัน)
  // ===================================================================
  psnr: {
    name: "PSNR (Peak Signal-to-Noise Ratio)",
    needs: "gray",
    desc: "วัดความเหมือนของภาพสองใบเป็นตัวเลขเดียว (หน่วย dB) — คำนวณจากค่าเฉลี่ยความต่างกำลังสองของพิกเซล (MSE) แล้วแปลงเป็นสเกล dB ยิ่งค่าสูง (>30 dB มักถือว่าดี) ยิ่งสองภาพเหมือนกันมาก ใช้วัดคุณภาพการบีบอัด/ประมวลผลภาพเทียบกับภาพต้นฉบับ — build ของโปรแกรมนี้ไม่มี cv.PSNR ในตัว (ยืนยันแล้ว) จึงคำนวณเองจาก MSE ต่อสาย 'เข้า B' เพื่อกำหนดภาพที่สอง",
    params: [],
    apply(cv, src, p, ctx) {
      if (!ctx || !ctx.src2) {
        let dst = null, rgb = null;
        try {
          dst = new cv.Mat();
          rgb = new cv.Mat();
          cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
          rgb.copyTo(dst);
          if (ctx) ctx.info = "ยังไม่ได้ต่อสาย 'เข้า B' — ลากเส้นจากอีกจุดหนึ่งมาต่อที่ช่องอินพุตที่สองของกล่องนี้";
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          if (rgb) rgb.delete();
        }
      }
      let gray2 = null, matched = null, diff = null, diff32 = null, sq = null, dst = null;
      try {
        gray2 = new cv.Mat();
        if (ctx.src2.channels() === 1) ctx.src2.copyTo(gray2); else cv.cvtColor(ctx.src2, gray2, cv.COLOR_RGB2GRAY);
        const { mat: bm, resized } = matchSize(cv, src, gray2);
        matched = resized ? bm : null;
        const b = matched || gray2;

        diff = new cv.Mat();
        cv.absdiff(src, b, diff);
        diff32 = new cv.Mat();
        diff.convertTo(diff32, cv.CV_32F);
        sq = new cv.Mat();
        cv.multiply(diff32, diff32, sq);
        const mse = cv.mean(sq)[0];
        // mse=0 (สองภาพเหมือนกันทุกพิกเซลเป๊ะ) ต้องได้ psnr=Infinity เสมอ — ยืนยันเชิงประจักษ์ใน Node ก่อน
        // เขียนโค้ดนี้ว่า 10*Math.log10(255*255/0) ได้ Infinity ตามธรรมชาติของ JS อยู่แล้ว (ไม่ throw/ไม่ NaN)
        // แต่ยังใส่ branch ชัดเจนไว้กัน edge case ของ -0 (255*255/-0 === -Infinity, log10(-Infinity)===NaN
        // ซึ่งผิด — ในทางปฏิบัติ cv.mean() ของภาพเหมือนกันเป๊ะไม่ควรได้ -0 แต่ปลอดภัยไว้ก่อนดีกว่า และอ่านง่ายกว่า)
        const psnr = mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);

        dst = new cv.Mat();
        diff.copyTo(dst); // ภาพผลลัพธ์ = แผนที่ความต่าง |A-B| (เหมือน imabsdiff) ให้มีอะไรดูเสมอ ไม่ใช่แค่ตัวเลข
        if (ctx) {
          ctx.info = `PSNR = ${psnr === Infinity ? "∞ (เหมือนกันทุกพิกเซล)" : psnr.toFixed(2) + " dB"} (MSE = ${mse.toFixed(2)})` +
            (resized ? " (ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)" : "");
          ctx.values = { kind: "scalars", fields: [
            { key: "psnr_db", label: "PSNR (dB)", value: psnr },
            { key: "mse", label: "MSE", value: mse },
          ] };
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (gray2) gray2.delete();
        if (matched) matched.delete();
        if (diff) diff.delete();
        if (diff32) diff32.delete();
        if (sq) sq.delete();
      }
    },
  },

  ssim: {
    name: "SSIM (Structural Similarity)",
    needs: "gray",
    desc: "วัดความเหมือนของภาพสองใบโดยเทียบ 'โครงสร้าง' (ความสว่างเฉลี่ย, ความคมชัด, รูปแบบ) ในบริเวณเล็ก ๆ ทั่วภาพ แทนที่จะเทียบพิกเซลต่อพิกเซลตรง ๆ แบบ PSNR — ค่าอยู่ระหว่าง -1 ถึง 1 (1 = เหมือนกันทุกประการ) มักสอดคล้องกับการรับรู้ของสายตามนุษย์ดีกว่า PSNR — build ของโปรแกรมนี้ไม่มี SSIM สำเร็จรูป คำนวณเองด้วยสูตรมาตรฐาน (Wang et al. 2004) ผ่าน GaussianBlur แทน windowed mean/variance — ค่าดีฟอลต์คำนวณที่ความละเอียดเต็ม (ถูกต้องที่สุด แต่ที่ 1080p วัดจริงได้ ~82-127ms/ครั้ง ช้าสำหรับกล้องสด) เลือกโหมด 'เร็ว' ได้ถ้าต้องการความไวแทนความแม่นยำ (ย่อภาพเหลือ ≤640px ก่อนคำนวณ ~10ms แต่ค่าที่ได้อาจคลาดเคลื่อนจากค่าจริงได้มาก) ต่อสาย 'เข้า B' เพื่อกำหนดภาพที่สอง",
    params: [
      { key: "speed", type: "select", label: "โหมดคำนวณ", options: [
        { value: "full", label: "ละเอียดเต็ม (ค่าถูกต้อง)" },
        { value: "fast", label: "เร็ว (ย่อภาพ ≤640px)" },
      ], default: "full", desc: "'ละเอียดเต็ม' คำนวณจากภาพจริงเสมอ ให้ค่า SSIM ที่ถูกต้อง แต่ภาพใหญ่จะช้า — 'เร็ว' ย่อภาพก่อนคำนวณเพื่อความไว (เช่นดูค่าแบบสด ๆ ขณะปรับพารามิเตอร์) แลกกับความแม่นยำที่ลดลง โดยเฉพาะภาพที่มีรายละเอียดเล็ก ๆ หรือสัญญาณรบกวนมาก" },
    ],
    apply(cv, src, p, ctx) {
      if (!ctx || !ctx.src2) {
        let dst = null, rgb = null;
        try {
          dst = new cv.Mat();
          rgb = new cv.Mat();
          cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
          rgb.copyTo(dst);
          if (ctx) ctx.info = "ยังไม่ได้ต่อสาย 'เข้า B' — ลากเส้นจากอีกจุดหนึ่งมาต่อที่ช่องอินพุตที่สองของกล่องนี้";
          return dst;
        } catch (err) {
          if (dst) dst.delete();
          throw err;
        } finally {
          if (rgb) rgb.delete();
        }
      }
      let gray2 = null, matched = null, analysisA = null, analysisB = null,
        af = null, bf = null, mu_a = null, mu_b = null, mu_a2 = null, mu_b2 = null, mu_ab = null,
        a2 = null, b2 = null, ab = null, sigma_a2 = null, sigma_b2 = null, sigma_ab = null,
        c1Mat = null, c2Mat = null, t1 = null, t2 = null, t3 = null, t4 = null,
        numer = null, denom = null, ssimMap = null, normed = null, ssim8u = null, dst = null;
      try {
        gray2 = new cv.Mat();
        if (ctx.src2.channels() === 1) ctx.src2.copyTo(gray2); else cv.cvtColor(ctx.src2, gray2, cv.COLOR_RGB2GRAY);
        const { mat: bm, resized } = matchSize(cv, src, gray2);
        matched = resized ? bm : null;
        const b = matched || gray2;

        // qa รอบแก้ 18b พบว่า downscale ที่เคยบังคับใช้เสมอ (ไม่มีทางเลือก) เปลี่ยนค่า SSIM ได้มาก
        // (1920x1080: full-res=0.3446 vs downscaled=0.9471 — ~175% relative) โดยไม่แจ้งผู้ใช้เลย —
        // เปลี่ยนเป็น speed param แทน "full" (ดีฟอลต์ กัน backward-compat กับโปรเจกต์เก่าที่ไม่มี key นี้
        // ด้วย — p.speed undefined ต้องได้ "full" เสมอ ไม่ใช่ "fast") คำนวณที่ความละเอียดเต็มเสมอ
        // ("fast" เท่านั้นที่ย่อ ≤640px แบบเดิม, มี ctx.info แจ้งเมื่อ scale<1 จริง — ดูด้านล่าง)
        const speed = p && p.speed === "fast" ? "fast" : "full";
        const maxAnalysisSide = 640;
        let scale = 1, aw = src.cols, ah = src.rows;
        if (speed === "fast") {
          ({ scale, w: aw, h: ah } = analysisScale(src.cols, src.rows, maxAnalysisSide));
        }
        analysisA = new cv.Mat();
        analysisB = new cv.Mat();
        if (scale < 1) {
          cv.resize(src, analysisA, new cv.Size(aw, ah), 0, 0, cv.INTER_AREA);
          cv.resize(b, analysisB, new cv.Size(aw, ah), 0, 0, cv.INTER_AREA);
        } else {
          src.copyTo(analysisA);
          b.copyTo(analysisB);
        }

        // สูตร Wang et al. 2004 ผ่าน windowed mean/variance/covariance ด้วย GaussianBlur (11x11, sigma=1.5)
        // — ยืนยันแล้วว่า build นี้ไม่มี cv.SSIM ในตัว
        const C1 = (0.01 * 255) ** 2;
        const C2 = (0.03 * 255) ** 2;
        const win = new cv.Size(11, 11);
        af = new cv.Mat(); analysisA.convertTo(af, cv.CV_32F);
        bf = new cv.Mat(); analysisB.convertTo(bf, cv.CV_32F);
        mu_a = new cv.Mat(); cv.GaussianBlur(af, mu_a, win, 1.5);
        mu_b = new cv.Mat(); cv.GaussianBlur(bf, mu_b, win, 1.5);
        mu_a2 = new cv.Mat(); cv.multiply(mu_a, mu_a, mu_a2);
        mu_b2 = new cv.Mat(); cv.multiply(mu_b, mu_b, mu_b2);
        mu_ab = new cv.Mat(); cv.multiply(mu_a, mu_b, mu_ab);
        a2 = new cv.Mat(); cv.multiply(af, af, a2);
        b2 = new cv.Mat(); cv.multiply(bf, bf, b2);
        ab = new cv.Mat(); cv.multiply(af, bf, ab);
        sigma_a2 = new cv.Mat(); cv.GaussianBlur(a2, sigma_a2, win, 1.5); cv.subtract(sigma_a2, mu_a2, sigma_a2);
        sigma_b2 = new cv.Mat(); cv.GaussianBlur(b2, sigma_b2, win, 1.5); cv.subtract(sigma_b2, mu_b2, sigma_b2);
        sigma_ab = new cv.Mat(); cv.GaussianBlur(ab, sigma_ab, win, 1.5); cv.subtract(sigma_ab, mu_ab, sigma_ab);

        c1Mat = new cv.Mat(mu_ab.rows, mu_ab.cols, mu_ab.type(), new cv.Scalar(C1));
        c2Mat = new cv.Mat(sigma_ab.rows, sigma_ab.cols, sigma_ab.type(), new cv.Scalar(C2));

        // t1 = 2*mu_a*mu_b + C1, t2 = 2*sigma_ab + C2 — ใช้ addWeighted (alpha*src1+beta*src2+gamma) แทน
        // multiply+add แยก 2 ขั้น ประหยัด Mat ชั่วคราว (ไม่ต้องมี "2" Mat แยกเหมือน "one"/"C1" Mat ด้านบน)
        t1 = new cv.Mat(); cv.addWeighted(mu_ab, 2, c1Mat, 1, 0, t1);
        t2 = new cv.Mat(); cv.addWeighted(sigma_ab, 2, c2Mat, 1, 0, t2);
        numer = new cv.Mat(); cv.multiply(t1, t2, numer);

        t3 = new cv.Mat(); cv.add(mu_a2, mu_b2, t3); cv.add(t3, c1Mat, t3);
        t4 = new cv.Mat(); cv.add(sigma_a2, sigma_b2, t4); cv.add(t4, c2Mat, t4);
        denom = new cv.Mat(); cv.multiply(t3, t4, denom);

        ssimMap = new cv.Mat();
        cv.divide(numer, denom, ssimMap);
        const ssimMean = cv.mean(ssimMap)[0];

        // ภาพผลลัพธ์ = แผนที่ SSIM ต่อพิกเซล normalize 0-255 (มืด = คล้ายน้อยเฉพาะจุด) แล้วขยายกลับขนาดเต็ม
        normed = new cv.Mat();
        cv.normalize(ssimMap, normed, 0, 255, cv.NORM_MINMAX);
        ssim8u = new cv.Mat();
        normed.convertTo(ssim8u, cv.CV_8U);
        dst = new cv.Mat();
        cv.resize(ssim8u, dst, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_LINEAR);

        if (ctx) {
          // qa วัดจริงพบว่า downscale วิเคราะห์ (≤640px) เปลี่ยนค่า SSIM ได้มาก ไม่ใช่แค่ error เล็กน้อย
          // (เคสทดสอบ 1920x1080: full-res SSIM=0.3446 แต่ downscaled SSIM=0.9471 — ต่างกัน ~175% relative
          // เพราะการย่อภาพ (INTER_AREA) เป็นการกรองความถี่สูงออกไปเอง ขัดกับสิ่งที่ SSIM ควรวัดพอดี) —
          // รอบแก้นี้เปลี่ยนดีฟอลต์เป็น "full" (ไม่ย่อ ค่าถูกต้องเสมอ) แต่ยังเก็บ "fast" ไว้เป็นทางเลือก —
          // ต้องแจ้งผู้ใช้เสมอเมื่อ downscale ทำงานจริง (scale<1, เกิดได้เฉพาะโหมด fast) ตามธรรมเนียมเดิม
          // ของไฟล์นี้ (ดู matchSize resize ของ imabsdiff/psnr ด้านบน) + แจ้งเตือนถ้าเลือก full กับภาพใหญ่
          // (ช้ากว่าปกติ วัดจริงที่ 1080p ~82-127ms) ให้ผู้ใช้รู้ว่ามีโหมด fast ให้สลับได้ถ้าต้องการความไว
          const fastNote = scale < 1
            ? " (คำนวณจากภาพที่ย่อขนาดลงก่อนเพื่อความเร็ว ค่าอาจต่างจากภาพความละเอียดเต็มจริงได้มาก โดยเฉพาะถ้าภาพมีรายละเอียดเล็ก ๆ หรือสัญญาณรบกวนมาก)"
            : "";
          const bigImageHint = (speed === "full" && Math.max(src.cols, src.rows) > maxAnalysisSide)
            ? " (ภาพใหญ่ อาจคำนวณช้ากว่าปกติ — เลือกโหมด 'เร็ว' ได้ถ้าต้องการความไวแทนความแม่นยำ)"
            : "";
          ctx.info = `SSIM = ${ssimMean.toFixed(4)} (1 = เหมือนกันทุกประการ)` +
            (resized ? " (ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)" : "") +
            fastNote + bigImageHint;
          ctx.values = { kind: "scalars", fields: [{ key: "ssim", label: "SSIM", value: ssimMean }] };
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (gray2) gray2.delete();
        if (matched) matched.delete();
        if (analysisA) analysisA.delete();
        if (analysisB) analysisB.delete();
        if (af) af.delete();
        if (bf) bf.delete();
        if (mu_a) mu_a.delete();
        if (mu_b) mu_b.delete();
        if (mu_a2) mu_a2.delete();
        if (mu_b2) mu_b2.delete();
        if (mu_ab) mu_ab.delete();
        if (a2) a2.delete();
        if (b2) b2.delete();
        if (ab) ab.delete();
        if (sigma_a2) sigma_a2.delete();
        if (sigma_b2) sigma_b2.delete();
        if (sigma_ab) sigma_ab.delete();
        if (c1Mat) c1Mat.delete();
        if (c2Mat) c2Mat.delete();
        if (t1) t1.delete();
        if (t2) t2.delete();
        if (t3) t3.delete();
        if (t4) t4.delete();
        if (numer) numer.delete();
        if (denom) denom.delete();
        if (ssimMap) ssimMap.delete();
        if (normed) normed.delete();
        if (ssim8u) ssim8u.delete();
      }
    },
  },

  // ===================================================================
  // Motion (Tier 2, 18d) — op ที่ "จำ" ข้อมูลข้ามเฟรมผ่าน ctx.state (§A.3 + §C.4-final)
  // สัญญาของ state (ทั้ง 3 op): (1) ไม่มี ctx.state (smoke/unit call) = ทำตัวเป็น "เฟรมแรก" แล้วทิ้ง state
  // ชั่วคราวใน finally (2) ตั้ง __dispose ก่อนจองของถาวรชิ้นแรก และเรียกซ้ำได้ (3) แทนที่ Mat ถาวร "หลัง" ทุกอย่าง
  // ที่ throw ได้สำเร็จแล้วเท่านั้น (4) ไม่ delete src/ctx.src2 · output เป็น Mat ใหม่ขนาด src.cols×src.rows เสมอ
  // (5) พิกัดภาพย่อ → ภาพจริง ด้วย × invScale (6) ห้ามปล่อยให้ C++ throw ในการใช้งานปกติ (รั่ว ~760B/ครั้ง) — guard ก่อน
  // ===================================================================
  mog2: {
    name: "Background Subtraction (MOG2)",
    needs: "any",
    desc: "แยก 'สิ่งที่กำลังเคลื่อนไหว' ออกจากพื้นหลังที่นิ่ง — กล่องนี้ค่อย ๆ เรียนรู้ว่าพื้นหลังหน้าตาเป็นอย่างไรจากหลายเฟรมที่ผ่านมา แล้วทำเครื่องหมายพิกเซลที่ต่างจากพื้นหลังเป็นสีขาว ใช้กับกล้องที่ตั้งนิ่ง ๆ หรือวิดีโอ (ภาพนิ่งภาพเดียวใช้ไม่ได้ เพราะต้องมีหลายเฟรม) — ช่วง 30 เฟรมแรกยังเรียนรู้อยู่ ผลจะยังไม่นิ่ง, วัตถุที่หยุดนิ่งนาน ๆ จะค่อย ๆ กลายเป็นพื้นหลัง ต่อด้วย Morphology (Opening) เพื่อลบจุดเล็ก ๆ แล้วใช้ Bounding Boxes ตีกรอบวัตถุที่เคลื่อนไหวได้ทันที — เปลี่ยนค่าใด ๆ หรือกล่องก่อนหน้า จะเริ่มเรียนรู้ใหม่ · เวลา export ภาพหลายรูป แต่ละภาพจะถือเป็นเฟรมแรกเสมอ ผลจึงเป็นภาพดำล้วนทุกใบ (ยังไม่มีพื้นหลังให้เทียบ) — กล่องนี้ดูผลได้จากภาพสดในแอปเท่านั้น",
    params: [
      { key: "history", type: "number", label: "จำนวนเฟรมที่ใช้จำพื้นหลัง", min: 10, max: 2000, step: 10, default: 500, desc: "ยิ่งมาก พื้นหลังยิ่งมั่นคง แต่ปรับตัวตามแสงที่เปลี่ยนช้าลง" },
      { key: "varThreshold", type: "number", label: "ความไว (ยิ่งต่ำยิ่งไว)", min: 4, max: 100, step: 1, default: 16, desc: "ต่ำ = จับความเปลี่ยนแปลงเล็กน้อยได้ (มีจุด noise ปน), สูง = เฉพาะที่ต่างชัด ๆ" },
      { key: "detectShadows", type: "boolean", label: "แยกเงาออก (ไม่นับเงาเป็นวัตถุ)", default: true, desc: "เงาของวัตถุจะไม่ถูกนับเป็นส่วนที่เคลื่อนไหว" },
      { key: "view", type: "select", label: "แสดงผลแบบ", options: [
        { value: "mask", label: "หน้ากากขาว-ดำ (ต่อกล่องอื่นได้)" },
        { value: "overlay", label: "ซ้อนสีเขียวบนภาพจริง" },
      ], default: "mask" },
    ],
    apply(cv, src, p, ctx) {
      const temp = !(ctx && ctx.state);
      const st = temp ? {} : ctx.state;
      let analysis = null, fgA = null, fg = null, dst = null, green = null, bl = null;
      try {
        if (!st.__dispose) st.__dispose = () => { if (st.mog) { st.mog.delete(); st.mog = null; } };
        if (!st.mog) {
          // เปลี่ยน param ไม่ต้องเช็คเอง — getNodeState คืน state เปล่าให้ (fingerprint ต้นน้ำรวม param ของกล่องนี้)
          st.mog = new cv.BackgroundSubtractorMOG2(intIn(p.history, 500, 10, 2000), numIn(p.varThreshold, 16, 4, 100), boolOr(p.detectShadows, true));
          st.frameCount = 0;
        }
        // ย่อภาพวิเคราะห์ ≤480px (MOG2 ที่ 1080p ภาพ noise ใช้ ~139ms) แล้วขยาย mask กลับแบบ NEAREST (ภาพ label ห้าม linear)
        const a = analysisScale(src.cols, src.rows, 480);
        analysis = new cv.Mat();
        if (a.scale < 1) cv.resize(src, analysis, new cv.Size(a.w, a.h), 0, 0, cv.INTER_AREA); else src.copyTo(analysis);
        fgA = new cv.Mat();
        st.mog.apply(analysis, fgA, -1);
        // เงา = 127 (เมื่อ detectShadows) — ตัดทิ้งให้เหลือ 0/255 ไม่งั้น Find Contours/Bounding Boxes ปลายน้ำนับเงาเป็นวัตถุ
        cv.threshold(fgA, fgA, 200, 255, cv.THRESH_BINARY);
        st.frameCount++; // นับ "หลัง" ทุกอย่างที่ throw ได้ผ่านไปแล้ว — เฟรมที่ล้มกลางทางต้องไม่ทำให้ตัวนับ/ข้อความ "กำลังเรียนรู้ (เฟรม n/30)" เพี้ยน
        fg = new cv.Mat();
        if (a.scale < 1) cv.resize(fgA, fg, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_NEAREST); else fgA.copyTo(fg);
        // เฟรมแรก OpenCV ให้ foreground 100% ทุกพิกเซล (V2) — ห้ามหลุดออกไปเป็นผลลัพธ์
        if (st.frameCount === 1) fg.setTo(new cv.Scalar(0));
        const px = cv.countNonZero(fg);
        const total = Math.max(1, src.cols * src.rows);
        const pct = (100 * px) / total;
        if (p.view === "overlay") {
          dst = motionRGB(cv, src);
          green = new cv.Mat(dst.rows, dst.cols, dst.type(), new cv.Scalar(0, 255, 0));
          bl = new cv.Mat();
          cv.addWeighted(dst, 0.5, green, 0.5, 0, bl);
          bl.copyTo(dst, fg); // นอก mask = พิกเซลเดิมเป๊ะ
        } else { dst = fg; fg = null; }
        if (ctx) {
          const n = st.frameCount;
          ctx.info = n === 1 ? "เฟรมแรก — เริ่มเรียนรู้พื้นหลัง (ยังแยกวัตถุไม่ได้)"
            : n < 30 ? `กำลังเรียนรู้พื้นหลัง (เฟรม ${n}/30) ผลยังไม่นิ่ง — ตั้งกล้องให้นิ่ง`
            : `พบส่วนที่เคลื่อนไหว ${pct.toFixed(1)}% ของภาพ (${px} px)`;
          ctx.values = { kind: "scalars", fields: [
            { key: "foreground_pct", label: "พื้นที่เคลื่อนไหว (%)", value: pct },
            { key: "foreground_px", label: "จำนวนพิกเซลที่เคลื่อนไหว", value: px },
            { key: "frame", label: "เฟรมที่เรียนรู้แล้ว", value: n },
          ] };
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        for (const m of [analysis, fgA, fg, green, bl]) if (m) m.delete();
        if (temp && st.__dispose) st.__dispose();
      }
    },
  },

  farneback_optical_flow: {
    name: "Optical Flow — ทั้งภาพ (Farneback)",
    needs: "any",
    desc: "วัดว่า 'ทุกจุดในภาพ' ขยับไปทางไหนและเร็วแค่ไหนระหว่างเฟรมก่อนหน้ากับเฟรมปัจจุบัน (dense optical flow) — โหมดวงล้อสี: สีบอกทิศทาง (แดง=ไปขวา, เขียวอมเหลือง=ลง, ฟ้า=ไปซ้าย, ม่วง=ขึ้น) ความสว่างบอกความเร็ว จุดที่นิ่งจะเป็นสีดำ — โหมดลูกศร: วาดเส้นทิศการเคลื่อนที่ทับภาพจริง ต้องใช้กล้องหรือวิดีโอ (เฟรมแรกยังไม่มีอะไรให้เทียบ) — ย่อภาพวิเคราะห์เหลือไม่เกิน 320px เพื่อความเร็ว · เวลา export ภาพหลายรูป แต่ละภาพจะถือเป็นเฟรมแรกเสมอ จึงไม่มีการเคลื่อนไหวให้แสดง (โหมดวงล้อสีได้ภาพดำล้วน โหมดลูกศรได้ภาพเดิม) — กล่องนี้ดูผลได้จากภาพสดในแอปเท่านั้น",
    params: [
      { key: "view", type: "select", label: "แสดงผลแบบ", options: [
        { value: "color", label: "วงล้อสี (ทิศ=สี, เร็ว=สว่าง)" },
        { value: "arrows", label: "ลูกศรบนภาพจริง" },
      ], default: "color" },
      { key: "maxMotion", type: "number", label: "ความเร็วที่แสดงสว่างสุด (px/เฟรม)", min: 1, max: 50, step: 1, default: 10, desc: "การเคลื่อนที่ตั้งแต่ค่านี้ขึ้นไปจะสว่างเต็มที่ (หน่วยเป็นพิกเซลของภาพจริง ไม่ใช่ภาพย่อ) — ใช้กับโหมดวงล้อสี" },
    ],
    apply(cv, src, p, ctx) {
      const temp = !(ctx && ctx.state);
      const st = temp ? {} : ctx.state;
      const maxMotion = numIn(p.maxMotion, 10, 1, 50);
      let cur = null, flow = null, dst = null, ch = null, fx = null, fy = null, mag = null, ang = null;
      let hh = null, vv = null, ss = null, mv = null, hsv = null, rgbA = null;
      try {
        if (!st.__dispose) st.__dispose = () => { if (st.prevGray) { st.prevGray.delete(); st.prevGray = null; } };
        const a = analysisScale(src.cols, src.rows, 320);
        const inv = a.invScale;
        cur = motionGray(cv, src, a); // Farneback throw C++ ถ้าเป็นภาพ 3 ช่อง (V4) — ใช้ขาวดำเสมอ
        // เฟรมแรก / หลัง reset / ขนาดไม่ตรง (guard V4: สองเฟรมขนาดต่างกัน = C++ throw) → คืนภาพเดิม เก็บเป็นเฟรมก่อนหน้า
        if (!st.prevGray || st.prevGray.cols !== cur.cols || st.prevGray.rows !== cur.rows) {
          dst = motionRGB(cv, src); // สร้าง output ก่อน แล้วค่อยแตะ state (contract ข้อ 3)
          if (st.prevGray) st.prevGray.delete();
          st.prevGray = cur; cur = null; st.frameCount = 1;
          if (ctx) {
            ctx.info = "เฟรมแรก — กำลังรอเฟรมถัดไปเพื่อเทียบการเคลื่อนไหว (ต้องใช้กล้องหรือวิดีโอ)";
            ctx.values = farnebackValues(0, 0, 0, 0, 0, 1);
          }
          return dst;
        }
        flow = new cv.Mat();
        // ค่าคงที่ตาม OpenCV tutorial (ไม่ให้ผู้ใช้ปรับ): pyr_scale 0.5, levels 3, winsize 15, iterations 3, poly_n 5, poly_sigma 1.2, flags 0
        cv.calcOpticalFlowFarneback(st.prevGray, cur, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        ch = new cv.MatVector(); cv.split(flow, ch); fx = ch.get(0); fy = ch.get(1); // get() คืน Mat ต้อง delete
        mag = new cv.Mat(); ang = new cv.Mat(); cv.cartToPolar(fx, fy, mag, ang, true); // มุมเป็นองศา 0..360
        // ตัวเลขทั้งหมดแปลงเป็น "px ของภาพจริง" (× inv) — คิดก่อน swap state
        const meanMotion = cv.mean(mag)[0] * inv;
        const maxM = cv.minMaxLoc(mag).maxVal * inv;
        const md = mag.data32F;
        const movingThr = 1 / inv; // ≥ 1px ของภาพจริง
        let moving = 0;
        for (let i = 0; i < md.length; i++) if (md[i] >= movingThr) moving++;
        const movingPct = (100 * moving) / Math.max(1, md.length);
        const meanDx = cv.mean(fx)[0] * inv;
        const meanDy = cv.mean(fy)[0] * inv;

        if (p.view === "arrows") {
          dst = motionRGB(cv, src);
          // ตารางจุดในภาพย่อ: ≤ ~1200 ลูกศรทุกความละเอียด · ความยาวลูกศร ×3 (คงที่) ให้เห็นการขยับเล็ก ๆ ได้
          const step = Math.max(8, Math.ceil(Math.sqrt((cur.cols * cur.rows) / 1200)));
          const th = Math.max(1, Math.round(Math.min(src.cols, src.rows) / 400));
          const col = new cv.Scalar(0, 255, 0);
          const fd = flow.data32F;
          for (let y = step >> 1; y < cur.rows; y += step) {
            for (let x = step >> 1; x < cur.cols; x += step) {
              const k = (y * cur.cols + x) * 2;
              const dx = fd[k], dy = fd[k + 1];
              if (Math.hypot(dx, dy) * inv < 1) continue; // นิ่ง (<1px ภาพจริง) ไม่วาด → ภาพนิ่ง = ภาพเดิมเป๊ะ
              const p0 = new cv.Point(Math.round(x * inv), Math.round(y * inv));
              const p1 = new cv.Point(Math.round((x + 3 * dx) * inv), Math.round((y + 3 * dy) * inv));
              cv.line(dst, p0, p1, col, th); // build นี้ไม่มี cv.arrowedLine (V7) — ใช้เส้น + จุดที่โคนแทน
              cv.circle(dst, p0, th, col, -1);
            }
          }
        } else {
          // H = มุม/2 (OpenCV H 0..180), S = 255, V = ความเร็ว สเกลคงที่ (ห้าม normalize ต่อเฟรม — noise ตอนนิ่งจะกลายเป็นสว่างเต็ม)
          hh = new cv.Mat(); ang.convertTo(hh, cv.CV_8U, 0.5);
          vv = new cv.Mat(); mag.convertTo(vv, cv.CV_8U, (255 * inv) / maxMotion);
          ss = new cv.Mat(cur.rows, cur.cols, cv.CV_8UC1, new cv.Scalar(255));
          mv = new cv.MatVector(); mv.push_back(hh); mv.push_back(ss); mv.push_back(vv);
          hsv = new cv.Mat(); cv.merge(mv, hsv);
          rgbA = new cv.Mat(); cv.cvtColor(hsv, rgbA, cv.COLOR_HSV2RGB);
          dst = new cv.Mat();
          if (a.scale < 1) cv.resize(rgbA, dst, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_LINEAR); else rgbA.copyTo(dst);
        }
        st.prevGray.delete(); st.prevGray = cur; cur = null; st.frameCount++; // swap เป็นขั้นสุดท้าย (contract ข้อ 3)
        if (ctx) {
          ctx.info = `ขยับเฉลี่ย ${meanMotion.toFixed(2)} px/เฟรม · สูงสุด ${maxM.toFixed(1)} · พื้นที่ที่ขยับ ${movingPct.toFixed(1)}%`;
          ctx.values = farnebackValues(meanMotion, maxM, movingPct, meanDx, meanDy, st.frameCount);
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        for (const m of [cur, flow, ch, fx, fy, mag, ang, hh, vv, ss, mv, hsv, rgbA]) if (m) m.delete();
        if (temp && st.__dispose) st.__dispose();
      }
    },
  },

  pyrlk_optical_flow: {
    name: "Optical Flow — ติดตามจุด (Lucas-Kanade)",
    needs: "any",
    desc: "เลือก 'จุดเด่น' (มุมของวัตถุ) ในภาพ แล้วติดตามว่าแต่ละจุดเคลื่อนไปไหนในเฟรมถัด ๆ ไป วาดเป็นเส้นทางสีเขียวตามหลังจุดสีแดง (sparse optical flow) — เบากว่าแบบทั้งภาพมาก เหมาะดูทิศทาง/ความเร็วของวัตถุที่มีลวดลายหรือมุมชัด ถ้าจุดหลุดหายไปเกินครึ่ง จะหาจุดใหม่ให้อัตโนมัติ พื้นผิวเรียบ ๆ (ผนังสีเดียว, ภาพมืด) จะหาจุดไม่เจอ ต้องใช้กล้องหรือวิดีโอ · เวลา export ภาพหลายรูป แต่ละภาพจะถือเป็นเฟรมแรกเสมอ จึงเห็นแค่จุดสีแดงที่เพิ่งเลือก ไม่มีเส้นทางการเคลื่อนที่ — กล่องนี้ดูผลได้จากภาพสดในแอปเท่านั้น",
    params: [
      { key: "maxPoints", type: "number", label: "จำนวนจุดสูงสุดที่ติดตาม", min: 10, max: 300, step: 10, default: 100, desc: "ยิ่งมากยิ่งละเอียด แต่ภาพจะรก" },
      { key: "trailLength", type: "number", label: "ความยาวเส้นทาง (เฟรม)", min: 1, max: 50, step: 1, default: 10, desc: "จำนวนเฟรมย้อนหลังที่วาดเป็นเส้นตามหลังแต่ละจุด" },
    ],
    apply(cv, src, p, ctx) {
      const temp = !(ctx && ctx.state);
      const st = temp ? {} : ctx.state;
      const maxPoints = intIn(p.maxPoints, 100, 10, 300);
      const trailLength = intIn(p.trailLength, 10, 1, 50);
      let cur = null, prevPts = null, nextPts = null, status = null, errM = null, corners = null, noMask = null, dst = null;
      try {
        if (!st.__dispose) st.__dispose = () => { if (st.prevGray) { st.prevGray.delete(); st.prevGray = null; } st.trails = null; };
        // ย่อ ≤480px (gftt ที่ 1080p ใช้ ~40ms — V8) · state เก็บพิกัดในหน่วย "px ภาพย่อ" เป็น JS array ล้วน (ไม่มี Mat จุดถาวร)
        const a = analysisScale(src.cols, src.rows, 480);
        const inv = a.invScale;
        cur = motionGray(cv, src, a);
        // guard V5: เฟรมก่อนหน้าขนาดต่างกัน = C++ throw → ถือเป็นเฟรมแรก (ของเก่าถูกแทนที่ตอน swap ท้ายสุด)
        const samePrev = !!st.prevGray && st.prevGray.cols === cur.cols && st.prevGray.rows === cur.rows;
        const oldTrails = samePrev && st.trails ? st.trails : null;
        let seedCount = samePrev ? st.seedCount || 0 : 0;
        let trails = null;
        let tracked = 0, lost = 0, sumMotion = 0, reseeded = 0;
        // guard V5 (บังคับ): 0 จุด = C++ throw ทุกเฟรม (ภาพดำ/เรียบ) → ติดตามเฉพาะเมื่อมีจุดจริง
        if (oldTrails && oldTrails.length > 0) {
          const n = oldTrails.length;
          const flat = new Float32Array(n * 2);
          for (let i = 0; i < n; i++) {
            const last = oldTrails[i][oldTrails[i].length - 1];
            flat[2 * i] = last[0];
            flat[2 * i + 1] = last[1];
          }
          prevPts = cv.matFromArray(n, 1, cv.CV_32FC2, flat);
          nextPts = new cv.Mat(); status = new cv.Mat(); errM = new cv.Mat();
          cv.calcOpticalFlowPyrLK(st.prevGray, cur, prevPts, nextPts, status, errM, new cv.Size(21, 21), 3,
            new cv.TermCriteria(cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 10, 0.03)); // build นี้ไม่มี cv.TERM_CRITERIA_*
          const np = nextPts.data32F;
          const sd = status.data;
          trails = [];
          for (let i = 0; i < n; i++) {
            const x = np[2 * i], y = np[2 * i + 1];
            if (sd[i] === 1 && x >= 0 && x <= cur.cols - 1 && y >= 0 && y <= cur.rows - 1) {
              const t = oldTrails[i].slice(); // สำเนา — ห้ามแก้ state เดิมก่อนทุกอย่างสำเร็จ (contract ข้อ 3)
              const last = t[t.length - 1];
              sumMotion += Math.hypot(x - last[0], y - last[1]);
              t.push([x, y]);
              while (t.length > trailLength + 1) t.shift();
              trails.push(t);
              tracked++;
            } else lost++;
          }
        }
        // หาจุดใหม่เมื่อยังไม่มีจุด หรือเหลือไม่ถึงครึ่งของตอนหาจุดครั้งล่าสุด (เทียบกับ seedCount ไม่ใช่ maxPoints —
        // ฉากที่มีมุมน้อยกว่า maxPoints มาก ๆ จะได้ไม่หาจุดใหม่ทุกเฟรมจนเส้นทางไม่เคยยาวขึ้น)
        if (!trails || trails.length === 0 || trails.length < Math.ceil(0.5 * seedCount)) {
          corners = new cv.Mat(); noMask = new cv.Mat();
          cv.goodFeaturesToTrack(cur, corners, maxPoints, 0.01, 7, noMask, 7, false, 0.04); // ภาพเรียบ/เล็ก = 0 จุด ไม่ throw (V6)
          const cd = corners.data32F;
          trails = [];
          for (let i = 0; i < corners.rows; i++) trails.push([[cd[2 * i], cd[2 * i + 1]]]);
          seedCount = corners.rows;
          reseeded = 1;
        }
        dst = motionRGB(cv, src);
        const green = new cv.Scalar(0, 255, 0);
        const red = new cv.Scalar(255, 0, 0);
        const toPt = (q) => new cv.Point(Math.round(q[0] * inv), Math.round(q[1] * inv));
        for (const t of trails) {
          for (let j = 1; j < t.length; j++) cv.line(dst, toPt(t[j - 1]), toPt(t[j]), green, 2);
          cv.circle(dst, toPt(t[t.length - 1]), 3, red, -1);
        }
        const firstFrame = !samePrev;
        // swap เป็นขั้นสุดท้าย (contract ข้อ 3)
        if (st.prevGray) st.prevGray.delete();
        st.prevGray = cur; cur = null;
        st.trails = trails;
        st.seedCount = seedCount;
        st.frameCount = firstFrame ? 1 : (st.frameCount || 0) + 1;
        if (ctx) {
          const points = trails.length;
          const meanMotion = tracked > 0 ? (sumMotion / tracked) * inv : 0;
          ctx.info = points === 0 ? "ไม่พบจุดเด่นให้ติดตาม (ภาพเรียบหรือมืดเกินไป) — ลองหันกล้องไปที่พื้นผิวที่มีลวดลายหรือมุม"
            : firstFrame ? `เฟรมแรก — เลือกจุดเด่นได้ ${points} จุด กำลังรอเฟรมถัดไปเพื่อติดตาม`
            : `ติดตาม ${points} จุด · ขยับเฉลี่ย ${meanMotion.toFixed(2)} px/เฟรม${reseeded ? " · หาจุดใหม่" : ""}`;
          ctx.values = { kind: "scalars", fields: [
            { key: "points", label: "จำนวนจุด", value: points },
            { key: "tracked", label: "ติดตามได้", value: tracked },
            { key: "lost", label: "จุดที่หลุด", value: lost },
            { key: "mean_motion_px", label: "ขยับเฉลี่ย (px/เฟรม)", value: meanMotion },
            { key: "reseeded", label: "หาจุดใหม่", value: reseeded },
            { key: "frame", label: "เฟรม", value: st.frameCount },
          ] };
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        for (const m of [cur, prevPts, nextPts, status, errM, corners, noMask]) if (m) m.delete();
        if (temp && st.__dispose) st.__dispose();
      }
    },
  },

  // ===================================================================
  // IP batch 2 ชิ้น C (2026-10-03) — gaussian_noise (FILTERS.noise) · wiener_filter (FILTERS.blur) · translate_shear
  // (FILTERS.geometry) — รวมไว้ท้าย OPS เป็นก้อนเดียว (ลำดับ key ของ OPS ไม่มีผลต่อการทำงาน node ใช้ลำดับจาก opIds)
  // ⚠️ gaussian_noise ต้องอยู่ "หลัง" log_edge: เทสต์ ipe-27 อ่าน numIn(p.sigma, …) ตัวแรกของไฟล์เป็นของ log_edge
  // ===================================================================
  // IP batch 2 ชิ้น C (2026-10-03) — imnoise(..., 'gaussian') แบบมี seed: ลายเดิมทุกเฟรม และสคริปต์ Python ได้ผลเท่ากันทุก byte
  gaussian_noise: {
    name: "Gaussian Noise (เกรนแบบเกาส์)",
    needs: "any",
    desc: "จำลองภาพที่มีเกรน (เช่นภาพถ่ายในที่แสงน้อย) ด้วย noise แบบบวก: ทุกพิกเซลและทุกช่องสีถูกบวกด้วยค่าสุ่มของตัวเองอย่างอิสระ ค่าสุ่มมาจากการแจกแจงแบบเกาส์ (ระฆังคว่ำ) ค่าเฉลี่ย 0 — ราว 68% ของค่าสุ่มอยู่ในช่วง ±σ · ค่าที่เกิน 0–255 ถูกตัดที่ขอบ บริเวณที่มืดหรือสว่างจัดจึงได้ noise ไม่สมมาตร · seed เดิม = ลาย noise เดิมทุกเฟรม (เทียบก่อน/หลังกรองได้นิ่ง และสคริปต์ Python ที่ export ได้ผลเท่ากันทุก byte) ต่างจาก imnoise ของ MATLAB ที่สุ่มใหม่ทุกครั้งที่เรียก · ชวนลอง: ต่อ Gaussian Blur หรือ Wiener Filter แล้วเทียบกับ Median Blur ขนาดเดียวกัน — กับ noise แบบนี้การเฉลี่ยมักลดเกรนได้มากกว่า ตรงข้ามกับกรณี Salt & Pepper ที่ Median ได้เปรียบ · แบบ Speckle (คูณ): ใหม่ = เดิม + เดิม × (σ/255) × z — พิกเซลดำ (0) ไม่เปลี่ยนเลย · ต่างจาก MATLAB imnoise('speckle') ที่ใช้ค่าสุ่มแบบ uniform — แอปใช้แบบเกาส์ (ตัวสุ่มเดียวกับแบบบวก seed เดิมได้ลายเดิม)",
    params: [
      { key: "sigma", type: "number", label: "ความแรง σ (ระดับสี 0–255)", min: 0, max: 100, step: 1, default: 20, hint: "MATLAB imnoise ค่าเริ่มต้น var 0.01 ≈ σ 25.5", desc: "ส่วนเบี่ยงเบนมาตรฐานของ noise ในหน่วยระดับสี — มาก = เกรนแรงขึ้น · 0 = ไม่เติม noise" },
      { key: "seed", type: "number", label: "seed (ลายสุ่ม)", min: 1, max: 99999, step: 1, default: 1, desc: "เลขตั้งต้นของการสุ่ม — seed เดิมได้ลาย noise เดิมทุกครั้ง เปลี่ยนเลขเพื่อดูลายใหม่" },
      { key: "kind", type: "select", label: "ชนิด noise", options: [
        { value: "additive", label: "แบบบวก (เกรนเท่ากันทุกความสว่าง)" },
        { value: "speckle", label: "Speckle (แบบคูณ — เกรนแรงตามความสว่าง)", hint: "พบในภาพอัลตราซาวด์/เรดาร์ · บริเวณมืดแทบไม่มีเกรน" },
      ], default: "additive" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const sigma = numIn(p.sigma, 20, 0, 100);
      const seed = intIn(p.seed, 1, 1, 99999);
      let dst = null;
      try {
        dst = dropAlpha(cv, src) || src.clone(); // RGBA → RGB ใหม่ (ใช้เป็น dst ได้เลย) · 1/3 ช่อง → สำเนา
        if (sigma === 0) {
          if (ctx) ctx.info = "σ = 0 — ไม่เติม noise (ภาพเดิม)";
          return dst;
        }
        // ตัวอย่างที่ k ใช้กับ dst.data[k] ตามลำดับหน่วยความจำ (แถว → คอลัมน์ → ช่อง R,G,B) — Python กลับลำดับช่องให้ตรง
        const n = dst.rows * dst.cols * dst.channels();
        const z = seededGaussian(seed, n);
        const d = dst.data;
        let clipped = 0;
        if (p.kind === "speckle") {
          const sc = sigma / 255;
          for (let k = 0; k < n; k++) {
            const v = Math.floor(d[k] + d[k] * sc * z[k] + 0.5);
            if (v < 0) { d[k] = 0; clipped++; } else if (v > 255) { d[k] = 255; clipped++; } else d[k] = v;
          }
          if (ctx) ctx.info = `เติม speckle noise (แรงตามความสว่าง) σ=${sigma} ที่ความสว่าง 255 (seed ${seed}) · ชนขอบ 0/255 ${fixed1((clipped * 100) / n)}% ของค่า`;
          return dst;
        }
        for (let k = 0; k < n; k++) {
          const v = Math.floor(d[k] + sigma * z[k] + 0.5);
          if (v < 0) { d[k] = 0; clipped++; } else if (v > 255) { d[k] = 255; clipped++; } else d[k] = v;
        }
        if (ctx) ctx.info = `เติม noise แบบเกาส์ σ=${sigma} (seed ${seed}) · ชนขอบ 0/255 ${fixed1((clipped * 100) / n)}% ของค่า`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      }
    },
  },

  // IP batch 2 ชิ้น C (2026-10-03) — wiener2 ของ MATLAB (ทำแยกทีละช่องสี · ขอบภาพสะท้อนแทนการเติม 0)
  wiener_filter: {
    name: "Wiener Filter (ลด noise แบบปรับตามบริเวณ)",
    needs: "any",
    desc: "ลด noise แบบปรับตามบริเวณ (wiener2): ในหน้าต่าง k×k รอบแต่ละจุด หาค่าเฉลี่ยและ variance (ความแกว่งของค่า) แล้วเทียบกับ variance ของ noise — บริเวณเรียบ (variance ใกล้ noise) ถูกเกลี่ยเข้าหาค่าเฉลี่ยแรง ส่วนขอบ/รายละเอียด (variance สูงกว่า noise มาก) ถูกแตะน้อย ขอบจึงมักคมกว่าการใช้ Gaussian Blur ขนาดเดียวกัน · noise variance = 0 คือให้ประมาณจากภาพเองแบบ MATLAB (ค่าเฉลี่ยของ variance ท้องถิ่นทั้งภาพ — ภาพที่มีขอบ/ลวดลายมากจะได้ค่าประมาณสูงกว่า noise จริง) · ภาพสีทำแยกทีละช่อง R/G/B โดยแต่ละช่องประมาณ noise ของตัวเอง (MATLAB wiener2 รับเฉพาะภาพ 2 มิติ) · ขอบภาพใช้การสะท้อนค่า ต่างจาก MATLAB ที่เติม 0 → ผลริมภาพไม่เท่ากับ MATLAB · เหมาะกับ noise แบบเกาส์ (ลองต่อหลัง Gaussian Noise)",
    params: [
      { key: "ksize", type: "number", label: "ขนาดหน้าต่าง", min: 3, max: 15, step: 2, default: 5, hint: "เลขคี่ · สูงสุด 15", desc: "ขนาดบริเวณที่ใช้หาค่าเฉลี่ย/variance — ใหญ่ = เกลี่ยบริเวณเรียบได้เนียนขึ้นแต่รายละเอียดเล็กหายมากขึ้น" },
      { key: "noiseVar", type: "number", label: "noise variance", min: 0, max: 10000, step: 10, default: 0, hint: "0 = ประมาณจากภาพเองแบบ MATLAB", desc: "variance ของ noise (หน่วยระดับสียกกำลังสอง เช่น noise σ 20 → 400) — บริเวณที่ variance ท้องถิ่นไม่เกินค่านี้จะถูกแทนด้วยค่าเฉลี่ยทั้งหมด" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); } // boxFilter throw จริงบน 0×0
      const k = ensureOdd(intIn(p.ksize, 5, 3, 15));
      const noiseVar = numIn(p.noiseVar, 0, 0, 10000);
      let rgb = null, f = null, sq = null, s1 = null, s2 = null, dst = null;
      try {
        rgb = dropAlpha(cv, src);
        const x = rgb || src;
        const size = new cv.Size(k, k);
        const anchor = new cv.Point(-1, -1);
        // ผลรวม (ไม่หาร) ในหน้าต่างเป็น float32 — ค่าเป็นจำนวนเต็มแม่นเป๊ะเมื่อ k ≤ 15 (ดูคอมเมนต์ในส่วนตัวช่วยชิ้น C) · ลบ f/sq ทันทีที่ใช้เสร็จ
        // ให้มี float32 ขนาดเท่าภาพพร้อมกันไม่เกิน 3 ตัว
        s1 = new cv.Mat();
        cv.boxFilter(x, s1, cv.CV_32F, size, anchor, false, cv.BORDER_REFLECT_101);
        f = new cv.Mat();
        x.convertTo(f, cv.CV_32F);
        sq = new cv.Mat();
        cv.multiply(f, f, sq);
        f.delete(); f = null;
        s2 = new cv.Mat();
        cv.boxFilter(sq, s2, -1, size, anchor, false, cv.BORDER_REFLECT_101);
        sq.delete(); sq = null;

        const n = k * k;
        const ch = x.channels();
        const total = x.rows * x.cols;
        const len = total * ch;
        const a1 = s1.data32F, a2 = s2.data32F, xd = x.data;
        // noise ต่อช่อง: กำหนดเอง หรือค่าเฉลี่ยของ variance ท้องถิ่นของช่องนั้น (บวกตามลำดับ i — Python ใช้ cumsum ลำดับเดียวกัน)
        const noise = new Float64Array(ch);
        if (noiseVar > 0) noise.fill(noiseVar);
        else {
          for (let i = 0; i < len; i++) {
            const m = a1[i] / n;
            noise[i % ch] += a2[i] / n - m * m;
          }
          for (let c = 0; c < ch; c++) noise[c] /= total;
        }
        dst = new cv.Mat(x.rows, x.cols, x.type());
        const od = dst.data;
        // คำนวณ m/v ซ้ำในลูปนี้แทนการเก็บเป็น Float64Array ขนาดเท่าภาพ (ถูกกว่าเรื่องหน่วยความจำ)
        for (let i = 0; i < len; i++) {
          const m = a1[i] / n;
          const v = a2[i] / n - m * m;
          const nz = noise[i % ch];
          const den = Math.max(v, nz);
          const g = Math.max(v - nz, 0);
          const out = den > 0 ? m + ((xd[i] - m) / den) * g : m;
          const q = Math.floor(out + 0.5);
          od[i] = q < 0 ? 0 : q > 255 ? 255 : q;
        }
        if (ctx) {
          ctx.info = noiseVar > 0
            ? `Wiener ${k}×${k} · noise variance = ${noiseVar} (กำหนดเอง)`
            : `Wiener ${k}×${k} · noise variance ${ch === 1 ? "" : "R/G/B "}= ${Array.from(noise, fixed1).join("/")} (ประมาณจากภาพ)`;
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        for (const m of [rgb, f, sq, s1, s2]) if (m) m.delete();
      }
    },
  },

  // IP batch 2 ชิ้น C (2026-10-03) — imtranslate + shear ของ affine2d (หน่วย % · shear รอบกึ่งกลางภาพ)
  translate_shear: {
    name: "Translate / Shear (เลื่อน/เอียงภาพ)",
    needs: "any",
    desc: "เลื่อนภาพ (translate) และเอียงภาพ (shear) ด้วย affine transform — ขนาดภาพคงเดิม ส่วนที่ว่างเติมสีดำ (เหมือน imtranslate ที่ FillValues = 0) · ระยะเลื่อนเป็น % ของความกว้าง/ความสูง (ภาพทุกความละเอียดได้ผลเท่ากัน — MATLAB imtranslate ใช้หน่วยพิกเซล) แล้วปัดเป็นจำนวนเต็มพิกเซล ถ้าเลื่อนอย่างเดียว ภาพจึงไม่เบลอ · shear x = k คือแต่ละแถวเลื่อนไปทางขวา k × (ระยะที่แถวนั้นอยู่ต่ำกว่ากึ่งกลางภาพ) ส่วน shear y ทำแบบเดียวกันในแนวตั้ง — เอียงรอบกึ่งกลางภาพ (affine2d ของ MATLAB เอียงรอบมุมซ้ายบน) และต้องเฉลี่ยค่าจากพิกเซลข้างเคียง (bilinear) ภาพที่เอียงจึงนุ่มลงเล็กน้อย · ถ้าตั้ง shear x และ y พร้อมกันจนผลคูณใกล้ 1 ภาพจะยุบเป็นเส้น จึงไม่ทำและแจ้งให้ลดค่า",
    params: [
      { key: "tx", type: "number", label: "เลื่อนแนวนอน (% ของความกว้าง)", min: -100, max: 100, step: 1, default: 10, desc: "ค่าบวก = เลื่อนไปทางขวา, ค่าลบ = ไปทางซ้าย · ±100 = เลื่อนออกนอกกรอบทั้งภาพ (เมื่อ shear แนวนอน = 0 — ถ้าเอียงแนวนอนด้วย บางส่วนอาจถูกเอียงกลับเข้ากรอบ)" },
      { key: "ty", type: "number", label: "เลื่อนแนวตั้ง (% ของความสูง)", min: -100, max: 100, step: 1, default: 5, desc: "ค่าบวก = เลื่อนลง, ค่าลบ = เลื่อนขึ้น" },
      { key: "shearX", type: "number", label: "shear แนวนอน", min: -1, max: 1, step: 0.05, default: 0, desc: "เอียงแนวนอน: 0.2 = แถวที่อยู่ต่ำกว่ากึ่งกลาง 100 px เลื่อนไปขวา 20 px (แถวบนเลื่อนไปซ้าย) · 0 = ไม่เอียง" },
      { key: "shearY", type: "number", label: "shear แนวตั้ง", min: -1, max: 1, step: 0.05, default: 0, desc: "เอียงแนวตั้ง: 0.2 = คอลัมน์ที่อยู่ขวาของกึ่งกลาง 100 px เลื่อนลง 20 px (คอลัมน์ซ้ายเลื่อนขึ้น) · 0 = ไม่เอียง" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); } // warpAffine throw จริงบน 0×0
      const tx = numIn(p.tx, 10, -100, 100);
      const ty = numIn(p.ty, 5, -100, 100);
      const kx = numIn(p.shearX, 0, -1, 1);
      const ky = numIn(p.shearY, 0, -1, 1);
      const w = src.cols, h = src.rows;
      let rgb = null, M = null, dst = null;
      try {
        rgb = dropAlpha(cv, src);
        const x = rgb || src;
        const det = 1 - kx * ky;
        if (Math.abs(det) < SHEAR_SINGULAR_LIMIT) {
          // ไม่พิมพ์ค่า 1 − kx·ky: ตัวเลขที่ปัดแล้วอาจดูเหมือนไม่ถึงเกณฑ์ (เช่น kx 0.9500000000000001 จาก slider → "0.95×1 = 0.05")
          if (ctx) ctx.info = `ไม่เลื่อน/เอียง: shear x × shear y ใกล้ 1 เกินไป (shear x=${fixed2(kx)} y=${fixed2(ky)}) ภาพจะยุบเป็นเส้น — ลด shear x หรือ shear y ลง`;
          dst = x.clone();
          return dst;
        }
        // ปัดแบบ floor(x + 0.5) ให้ตรงกับ int(np.floor(x + 0.5)) ในสคริปต์ Python ทุกค่า (x ติดลบได้ จึงไม่ใช้ int(x + 0.5))
        const txp = Math.floor((tx / 100) * w + 0.5);
        const typ = Math.floor((ty / 100) * h + 0.5);
        const cx = (w - 1) / 2, cy = (h - 1) / 2;
        // จุด (x0, y0) ไปอยู่ที่ x' = x0 + kx·(y0 − cy) + txp, y' = y0 + ky·(x0 − cx) + typ
        M = cv.matFromArray(2, 3, cv.CV_64F, [1, kx, -kx * cy + txp, ky, 1, -ky * cx + typ]);
        dst = new cv.Mat();
        cv.warpAffine(x, dst, M, new cv.Size(w, h), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
        const sgn = (v) => (v >= 0 ? "+" : "") + v;
        // "ออกนอกกรอบหมด" จริงแน่ ๆ เมื่อเลื่อนแนวนอน ≥ w โดย shear x = 0 (x' = x0 + txp ไม่ขึ้นกับ shear y) หรือแนวตั้ง ≥ h โดย shear y = 0
        // — ถ้าเอียงแกนเดียวกับที่เลื่อน บางแถว/คอลัมน์ถูกเอียงกลับเข้ากรอบได้ (tx 100 + shear x −1 ยังเห็น ~9%) จึงไม่อ้าง
        const gone = (kx === 0 && Math.abs(txp) >= w) || (ky === 0 && Math.abs(typ) >= h) ? " (ภาพเลื่อนออกนอกกรอบหมด)" : "";
        if (ctx) ctx.info = `เลื่อน (${sgn(txp)}, ${sgn(typ)}) px · shear x=${fixed2(kx)} y=${fixed2(ky)}${gone}`;
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (rgb) rgb.delete();
        if (M) M.delete();
      }
    },
  },

  // ===================================================================
  // IP batch 2 ชิ้น D (2026-10-03) — เครื่องมือวัด: feret + line_profile (FILTERS.shape_analysis) · roi_polygon (FILTERS.geometry)
  // รวมไว้ท้าย OPS เป็นก้อนเดียวเหมือนชิ้น C (เทสต์ static เดิมบางตัวอ่าน numIn(p.…) "ตัวแรกของไฟล์")
  // ===================================================================
  // bwferet ของ MATLAB — วัดบนขอบพิกเซล (สเปก D §4.1) · ลำดับ/เลขวัตถุตรงกับ Region Properties (helper คัด/เรียงตัวเดียวกัน)
  feret: {
    name: "Feret Diameter (วัดความกว้างแบบคาลิปเปอร์)",
    needs: "gray",
    desc: "วัดขนาดวัตถุแบบเอาคาลิปเปอร์ (เวอร์เนียร์) หนีบจากทุกทิศทาง: ระยะที่ปากคาลิปเปอร์กางกว้างสุดคือ Max Feret (เส้นแดง) ระยะที่หนีบได้แคบสุดคือ Min Feret (กรอบเขียว = กรอบแคบสุดที่ล้อมวัตถุได้) — ค่าไม่ขึ้นกับการหมุนของวัตถุ ต่างจาก Bounding Box ที่เป็นกรอบตั้งตรงเสมอ วัตถุเอียงจึงได้กรอบใหญ่กว่าตัวจริง · วัดบนขอบพิกเซล (ไม่ใช่จุดกึ่งกลางพิกเซล): วัตถุยาว 10 px วัดได้ 10 พอดี พิกเซลเดี่ยว 1 จุดได้ Min = 1 และ Max = 1.41 (เส้นทแยงของสี่เหลี่ยม 1×1) · Max Feret ของสี่เหลี่ยมผืนผ้าคือ 'เส้นทแยงมุม' ไม่ใช่ด้านยาว — ความยาวด้านยาวดูที่คอลัมน์ 'ยาวตามกรอบแคบสุด' · คาลิปเปอร์แตะเฉพาะเปลือกนอก (convex hull) วัตถุที่มีรูจึงได้ค่าเท่ากับวัตถุตันขนาดเดียวกัน และส่วนที่เว้าเข้าไปไม่มีผล · ภาพที่ไม่ใช่ขาว-ดำล้วนจะถูกตัดที่ 127 ก่อน (เหมือน Region Properties) · เลขและลำดับวัตถุตรงกับ Region Properties เมื่อตั้งพื้นที่ขั้นต่ำเท่ากันและจำนวนวัตถุไม่เกินจำนวนสูงสุดของทั้งสองกล่อง · มุม: 0° = แนวนอน เพิ่มตามเข็มนาฬิกา (แกน y ของภาพชี้ลง) ช่วง 0–180° · ใกล้เคียง bwferet ของ MATLAB แต่ไม่รับประกันว่าตรงทุกทศนิยม (วิธีนับขอบพิกเซลอาจต่างกันเล็กน้อย) · แปลงเป็นหน่วยจริงต้องมีของอ้างอิงในภาพ · ตัวคูณเดียวใช้ได้เมื่อเลนส์ไม่บิดเบี้ยวมาก (ดู Undistort) และชิ้นงานอยู่ระนาบเดียว · พื้นที่คูณ (มม./px)²",
    params: [
      { key: "minArea", type: "number", label: "พื้นที่ขั้นต่ำ (px)", min: 0, max: 5000, step: 10, default: 50, desc: "วัตถุที่เล็กกว่านี้ไม่ถูกวัด (กันจุด noise เล็ก ๆ) — ความหมายเดียวกับ Region Properties: พื้นที่ภายในเส้นขอบของวัตถุ" },
      { key: "maxObjects", type: "number", label: "วัดวัตถุสูงสุด", min: 1, max: 500, step: 1, default: 50, desc: "วัดและวาดเฉพาะวัตถุตามจำนวนนี้ เรียงจากใหญ่ไปเล็ก (จำนวนวัตถุทั้งหมดที่พบยังรายงานครบ) — ไม่มีตัวเลือก 'ไม่จำกัด' เพราะกรอบหลายร้อยกรอบซ้อนกันจะอ่านไม่ออก" },
      mmScaleParam(),
    ],
    apply(cv, src, p, ctx) {
      const mmF = numIn(p.mmPerPx, 0, 0, 100);
      if (isEmptyMat(src)) { // cvtColor(GRAY2RGB) throw จริงบน 0×0
        if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = { kind: "table", ...mmTable("feret", mmF, FERET_COLUMNS, []), total: 0 }; }
        return src.clone();
      }
      const minArea = numIn(p.minArea, 50, 0, 5000);
      const cap = intIn(p.maxObjects, 50, 1, 500);
      let gray = null, bin = null, contours = null, hier = null, dst = null;
      try {
        gray = grayIfNeeded(cv, src); // ผู้เรียกตรง (เทสต์/โค้ดอื่น) อาจส่งภาพสีมา — worker ส่งขาวดำให้อยู่แล้ว
        const g = gray || src;
        bin = new cv.Mat();
        contours = new cv.MatVector();
        hier = new cv.Mat();
        cv.threshold(g, bin, 127, 255, cv.THRESH_BINARY);
        cv.findContours(bin, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        dst = new cv.Mat();
        cv.cvtColor(g, dst, cv.COLOR_GRAY2RGB);
        // คัด + เรียงด้วยตัวช่วยชุดเดียวกับ regionprops (area DESC, y, x, cx, cy, tx) → หมายเลขวัตถุตรงกัน
        const cands = [];
        for (let i = 0; i < contours.size(); i++) {
          const c = contours.get(i);
          try {
            const s = regionScan(c.data32S);
            if (s.area < minArea) continue;
            cands.push({ i, area: s.area, bx: s.bx, by: s.by, tx: s.tx, cx: undefined, cy: undefined });
          } finally {
            c.delete();
          }
        }
        const centroidOf = (rec) => {
          if (rec.cx === undefined) {
            const c = contours.get(rec.i);
            try { const q = contourCentroid(cv, c); rec.cx = q.cx; rec.cy = q.cy; } finally { c.delete(); }
          }
          return rec;
        };
        cands.sort((a, b) => compareRegionCandidates(a, b, centroidOf));
        const total = cands.length;
        const shown = cands.slice(0, cap);
        const rows = [];
        const GREEN = new cv.Scalar(0, 255, 0), RED = new cv.Scalar(255, 0, 0), YELLOW = new cv.Scalar(255, 255, 0);
        for (let idx = 0; idx < shown.length; idx++) {
          const rec = shown[idx];
          let f;
          const c = contours.get(rec.i);
          try {
            if (rec.cx === undefined) { const q = contourCentroid(cv, c); rec.cx = q.cx; rec.cy = q.cy; }
            f = feretOfContour(cv, c);
          } finally {
            c.delete();
          }
          const index = idx + 1;
          rows.push([index, rec.area, rec.cx, rec.cy, f.max, f.min, f.len, f.aspect, f.maxDeg, f.minDeg]);
          // วาด: กรอบแคบสุด (เขียว 4 เส้น) → แนวยาวสุด (แดง) → เลขวัตถุ (เหลือง ตำแหน่งเดียวกับ regionprops) — Python วาดลำดับเดียวกัน
          for (let q = 0; q < 4; q++) {
            const a = f.box[q], b = f.box[(q + 1) % 4];
            cv.line(dst, new cv.Point(a[0], a[1]), new cv.Point(b[0], b[1]), GREEN, 1);
          }
          cv.line(dst, new cv.Point(f.maxA[0], f.maxA[1]), new cv.Point(f.maxB[0], f.maxB[1]), RED, 1);
          cv.putText(dst, String(index), new cv.Point(rec.bx, Math.max(10, rec.by - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1);
        }
        if (ctx) {
          if (total === 0) {
            ctx.info = "ไม่พบวัตถุ (ลองลดค่าพื้นที่ขั้นต่ำ หรือทำ Threshold ให้เป็นขาว-ดำก่อน)";
          } else {
            const head = shown.length < total
              ? `พบวัตถุ ${total} ชิ้น วัด ${shown.length} ชิ้นแรกเรียงจากใหญ่ไปเล็ก (ปรับ 'วัดวัตถุสูงสุด' ได้)`
              : `วัด Feret ของวัตถุ ${total} ชิ้น`;
            ctx.info = `${head} · ชิ้น #1 ยาวสุด ${fixed1(rows[0][4])} px · แคบสุด ${fixed1(rows[0][5])} px (กรอบเขียว = กรอบที่แคบสุด, เส้นแดง = แนวที่ยาวสุด)`;
          }
          if (mmF > 0) ctx.info += MM_SUFFIX(mmF);
          ctx.values = { kind: "table", ...mmTable("feret", mmF, FERET_COLUMNS, rows), total };
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (gray) gray.delete();
        if (bin) bin.delete();
        if (contours) contours.delete();
        if (hier) hier.delete();
      }
    },
  },

  // improfile ของ MATLAB (nearest neighbour) — กราฟความสว่างตามเส้น A→B (สเปก D §4.2)
  line_profile: {
    name: "Line Profile (กราฟความสว่างตามเส้น)",
    needs: "any",
    desc: "อ่านความสว่างของพิกเซลทีละจุดตามเส้นตรงจาก A (จุดเขียว) ไป B (จุดแดง) แล้ววาดเป็นกราฟ: แกนนอน = ระยะจาก A, แกนตั้ง = ความสว่าง 0–255 (เส้นเทาล่าง/บน = 0/255) — บริเวณเรียบกราฟจะแบน ขอบวัตถุกราฟจะกระโดดขึ้นหรือลง ส่วน noise ทำให้กราฟหยัก · ภาพสีวัดจากความสว่างแบบขาวดำ (แปลงเป็นเทาก่อน) · อ่านจากพิกเซลที่อยู่ใกล้จุดบนเส้นมากสุด (nearest neighbour) ตามค่าเริ่มต้นของ improfile ใน MATLAB · วงกลมม่วงบนภาพและเส้นม่วงบนกราฟ = ตำแหน่งที่ค่ากระโดดแรงสุด (ถ้าค่าบนเส้นเท่ากันทุกจุดจะไม่มีขอบ จึงไม่วาดวงกลม/เส้นม่วงและช่องตัวเลขขอบเป็น —) · ระยะระหว่างขอบขึ้นกับขอบลงคือความกว้างของแถบสว่าง/มืดที่เส้นตัดผ่าน ใช้วัดความกว้างชิ้นงานได้ แต่ตรงเฉพาะเมื่อเส้นตัดแถบในแนวตั้งฉาก ถ้าเส้นเฉียงค่าจะยาวกว่าความกว้างจริง · ตำแหน่ง A/B ตั้งเป็น % ของขนาดภาพ (ยังลากเส้นบนภาพโดยตรงไม่ได้) · ภาพที่เล็กกว่า 64×64 px ไม่วาดกราฟ แต่ตัวเลขยังคำนวณครบ · แปลงเป็นหน่วยจริงต้องมีของอ้างอิงในภาพ · ตัวคูณเดียวใช้ได้เมื่อเลนส์ไม่บิดเบี้ยวมาก (ดู Undistort) และชิ้นงานอยู่ระนาบเดียว · พื้นที่คูณ (มม./px)²",
    params: [
      { key: "x1", type: "number", label: "จุด A แนวนอน (%)", min: 0, max: 100, step: 1, default: 10, desc: "ตำแหน่งจุดเริ่มเส้น (เขียว) เป็น % ของความกว้างภาพ — 100% = พิกเซลขวาสุด" },
      { key: "y1", type: "number", label: "จุด A แนวตั้ง (%)", min: 0, max: 100, step: 1, default: 50, desc: "ตำแหน่งจุดเริ่มเส้น (เขียว) เป็น % ของความสูงภาพ — 100% = แถวล่างสุด" },
      { key: "x2", type: "number", label: "จุด B แนวนอน (%)", min: 0, max: 100, step: 1, default: 90, desc: "ตำแหน่งจุดปลายเส้น (แดง) เป็น % ของความกว้างภาพ" },
      { key: "y2", type: "number", label: "จุด B แนวตั้ง (%)", min: 0, max: 100, step: 1, default: 50, desc: "ตำแหน่งจุดปลายเส้น (แดง) เป็น % ของความสูงภาพ" },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "overlay", label: "เส้น + กราฟบนภาพ", hint: "เห็นทั้งตำแหน่งเส้นและกราฟ (แถบกราฟย้ายขึ้นบนเองถ้าเส้นอยู่ด้านล่าง)" },
        { value: "plot", label: "กราฟเต็มภาพ", hint: "ใช้ทั้งภาพวาดกราฟ อ่านรายละเอียดง่ายขึ้น" },
      ], default: "overlay" },
      mmScaleParam(),
    ],
    apply(cv, src, p, ctx) {
      const mmF = numIn(p.mmPerPx, 0, 0, 100);
      if (isEmptyMat(src)) {
        if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = lineProfileValues([], mmF); }
        return src.clone();
      }
      const W = src.cols, H = src.rows;
      const ax = pctToPx(p.x1, 10, W), ay = pctToPx(p.y1, 50, H);
      const bx = pctToPx(p.x2, 90, W), by = pctToPx(p.y2, 50, H);
      const view = ownKey(LINE_PROFILE_VIEWS, p.view, "overlay");
      const small = W < LINE_PROFILE_MIN_GRAPH || H < LINE_PROFILE_MIN_GRAPH;
      let gray = null, dst = null, band = null, poly = null, polyVec = null;
      try {
        gray = grayIfNeeded(cv, src);
        const gd = (gray || src).data; // Mat ต่อเนื่องเสมอ (cvtColor / ภาพจาก worker) → index = y·W + x
        const dx = bx - ax, dy = by - ay;
        const N = Math.max(Math.abs(dx), Math.abs(dy)) + 1;
        const xs = new Int32Array(N), ys = new Int32Array(N), prof = new Int32Array(N);
        let sum = 0, vMin = 255, vMax = 0;
        for (let k = 0; k < N; k++) {
          // nearest neighbour: ลำดับการคำนวณเดียวกับ Python (x1 + (x2 − x1)·k / (N − 1) + 0.5 แล้ว floor)
          xs[k] = N === 1 ? ax : Math.floor(ax + (dx * k) / (N - 1) + 0.5);
          ys[k] = N === 1 ? ay : Math.floor(ay + (dy * k) / (N - 1) + 0.5);
          const v = gd[ys[k] * W + xs[k]];
          prof[k] = v;
          sum += v;
          if (v < vMin) vMin = v;
          if (v > vMax) vMax = v;
        }
        const L = Math.sqrt(dx * dx + dy * dy);
        let edgeJump = null, edgePos = null, edgePair = null, iStar = -1;
        if (N >= 2) {
          const step = L / (N - 1);
          let bestAbs = -1, rIdx = -1, rVal = 0, fIdx = -1, fVal = 0;
          for (let i = 0; i < N - 1; i++) {
            const d = prof[i + 1] - prof[i];
            const ad = d < 0 ? 0 - d : d;
            if (ad > bestAbs) { bestAbs = ad; iStar = i; } // ตัวแรกถ้าเท่ากัน
            if (d > rVal) { rVal = d; rIdx = i; } // ขอบขึ้นแรงสุด
            if (d < fVal) { fVal = d; fIdx = i; } // ขอบลงแรงสุด
          }
          if (bestAbs <= 0) {
            iStar = -1; // เส้นแบน (ทุกจุดเท่ากัน) ไม่มีขอบ → ไม่มีตัวเลขขอบ ไม่วงกลม/เส้นม่วง (ตัดสินใจ 2026-10-03)
          } else {
            edgeJump = prof[iStar + 1] - prof[iStar];
            edgePos = (iStar + 0.5) * step;
            if (rIdx >= 0 && fIdx >= 0) edgePair = Math.abs(rIdx - fIdx) * step;
          }
        }
        const YELLOW = new cv.Scalar(255, 255, 0), GREEN = new cv.Scalar(0, 255, 0), RED = new cv.Scalar(255, 0, 0);
        const MAGENTA = new cv.Scalar(255, 0, 255), GREY = new cv.Scalar(128, 128, 128);
        const plotView = view === "plot" && !small; // ภาพเล็ก: โหมดกราฟเต็มภาพถอยเป็นภาพเส้นไม่มีกราฟ
        if (plotView) {
          dst = new cv.Mat(H, W, cv.CV_8UC3, new cv.Scalar(24, 24, 24));
        } else {
          dst = motionRGB(cv, src);
          cv.line(dst, new cv.Point(ax, ay), new cv.Point(bx, by), YELLOW, 1);
          cv.circle(dst, new cv.Point(ax, ay), 3, GREEN, -1);
          cv.circle(dst, new cv.Point(bx, by), 3, RED, -1);
          if (iStar >= 0) cv.circle(dst, new cv.Point(xs[iStar + 1], ys[iStar + 1]), 4, MAGENTA, 1);
        }
        if (!small) {
          let gy0 = 0, gh = H;
          const gx0 = 0, gw = W;
          if (!plotView) {
            gh = Math.floor(0.35 * H);
            gy0 = ay >= H - gh && by >= H - gh ? 0 : H - gh; // A และ B อยู่ในแถบล่างทั้งคู่ → ย้ายกราฟขึ้นบน ไม่ให้ทับเส้น
            band = dst.roi(new cv.Rect(0, gy0, W, gh));
            cv.convertScaleAbs(band, band, 0.3, 0); // ทำแถบกราฟให้มืดลง
          }
          const gyOf = (v) => gy0 + (gh - 1) - Math.floor((2 * v * (gh - 1) + 255) / 510);
          cv.line(dst, new cv.Point(gx0, gyOf(0)), new cv.Point(gx0 + gw - 1, gyOf(0)), GREY, 1);
          cv.line(dst, new cv.Point(gx0, gyOf(255)), new cv.Point(gx0 + gw - 1, gyOf(255)), GREY, 1);
          if (N === 1) {
            cv.circle(dst, new cv.Point(gx0 + Math.floor((gw - 1) / 2), gyOf(prof[0])), 2, YELLOW, -1);
          } else {
            const gp = new Int32Array(2 * N);
            for (let k = 0; k < N; k++) {
              gp[2 * k] = gx0 + Math.floor((k * (gw - 1)) / (N - 1));
              gp[2 * k + 1] = gyOf(prof[k]);
            }
            poly = cv.matFromArray(N, 1, cv.CV_32SC2, gp);
            polyVec = new cv.MatVector();
            polyVec.push_back(poly);
            cv.polylines(dst, polyVec, false, YELLOW, 1);
            if (iStar >= 0) {
              const xe = gx0 + Math.floor(((iStar + 1) * (gw - 1)) / (N - 1));
              cv.line(dst, new cv.Point(xe, gy0), new cv.Point(xe, gy0 + gh - 1), MAGENTA, 1);
            }
          }
        }
        if (ctx) {
          let info = N === 1
            ? "จุด A กับ B อยู่ตำแหน่งเดียวกัน — ปรับให้ห่างกันก่อน"
            : `เส้น A(${ax},${ay})→B(${bx},${by}) ยาว ${fixed1(L)} px · อ่าน ${N} จุด · ค่า ${vMin}–${vMax} · ${iStar < 0 ? "ค่าบนเส้นเท่ากันทุกจุด ไม่มีขอบ" : `ขอบแรงสุดห่างจาก A ${fixed1(edgePos)} px (กระโดด ${edgeJump >= 0 ? "+" : ""}${edgeJump})`}`;
          if (small) info += ` · ภาพเล็กกว่า ${LINE_PROFILE_MIN_GRAPH}×${LINE_PROFILE_MIN_GRAPH} px จึงไม่วาดกราฟ (ตัวเลขยังคำนวณครบ)`;
          if (mmF > 0) info += MM_SUFFIX(mmF);
          ctx.info = info;
          ctx.values = lineProfileValues([L, N, vMin, vMax, sum / N, vMax - vMin, edgeJump, edgePos, edgePair], mmF);
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (gray) gray.delete();
        if (band) band.delete();
        if (poly) poly.delete();
        if (polyVec) polyVec.delete();
      }
    },
  },

  // IP batch 3 — bwpropfilt + bwareafilt ในกล่องเดียว: เก็บ/ทิ้งทั้งชิ้นตามค่าที่วัดได้ (ตัวช่วยชุดเดียวกับคอลัมน์ใหม่ของ Region Properties)
  shape_filter: {
    name: "Filter by Shape (คัดวัตถุตามรูปร่าง)",
    needs: "gray",
    desc: "เก็บหรือทิ้งวัตถุ 'ทั้งชิ้น' ตามตัวเลขที่วัดได้ — เป็นขั้นต่อจาก Remove Small (ที่คัดตามพื้นที่อย่างเดียว) เทียบได้กับ bwpropfilt + bwareafilt ของ MATLAB รวมในกล่องเดียว · ตัวเลขที่ใช้คัดคือค่าเดียวกับคอลัมน์ใน Region Properties ของวัตถุชิ้นเดียวกัน (ความกลม ความตัน ความรี เต็มกรอบ พื้นที่นับพิกเซล) — ดูตารางก่อนแล้วค่อยตั้งช่วงที่ต้องการ · ความกลม = 4π×พื้นที่ ÷ เส้นรอบรูป² (วงกลมในภาพพิกเซลไม่มีทางได้ 1 — ราว 0.85–0.89, สี่เหลี่ยม ≈ 0.79) · ความตัน = พื้นที่ ÷ รูปนูนที่ห่อวัตถุ (1 = ไม่เว้า/ไม่มีรู) · ความรี 0 = กลม, 0.87 ≈ ยาว 2 เท่าของกว้าง, 0.97 ≈ 4 เท่า, เส้น ≈ 1 · เต็มกรอบ = พื้นที่ ÷ กรอบสี่เหลี่ยมตรง (สี่เหลี่ยมตรง = 1, วงกลม ≈ 0.77) · ช่วงพื้นที่ใช้จำนวนพิกเซลที่นับจริง ไม่ใช่ 'area' แบบเส้นหลายเหลี่ยมที่ 'พื้นที่ขั้นต่ำ' ของ Region Properties ใช้ · วัตถุ = บริเวณสีขาวหลังตัดที่ > 127 รูในวัตถุยังเป็นรู วัตถุเกาะที่อยู่ในรูจะตามวัตถุชั้นนอก · แสดงผลแบบ overlay: ขาว = เก็บ, แดง = คัดทิ้ง ทั้ง overlay และ Mask ต่อเข้า Region Properties ได้ (สีแดงถูกตัดทิ้งที่ 127) · ต่างจาก MATLAB: bwpropfilt ใช้ 8-connected components และนับวัตถุเกาะเป็นอีกชิ้นหนึ่ง ส่วนเส้นรูปนูนของความตัน ผ่านกลางพิกเซล",
    params: [
      { key: "prop", type: "select", label: "คัดตามค่า", options: [
        { value: "circularity", label: "ความกลม (circularity)", hint: "4π×พื้นที่ ÷ เส้นรอบรูป² (ค่าเดียวกับคอลัมน์ 'ความกลม') · วงกลมในภาพพิกเซลได้ราว 0.85–0.89 · สี่เหลี่ยม ≈ 0.79 · วงกลมรัศมีน้อยกว่า 8 px ได้ต่ำกว่า 0.8" },
        { value: "solidity", label: "ความตัน (solidity)", hint: "1 = ไม่มีรอยเว้า/รู · รูปตัว L ≈ 0.61 · วงแหวน ≈ 0.61" },
        { value: "eccentricity", label: "ความรี (eccentricity)", hint: "0 = กลม · 0.87 ≈ ยาว 2 เท่าของกว้าง · 0.97 ≈ 4 เท่า · เส้น ≈ 1" },
        { value: "extent", label: "เต็มกรอบ (extent)", hint: "สี่เหลี่ยมตรง = 1 · วงกลม ≈ 0.77 · วัตถุเอียงหรือบางยาว = ต่ำ" },
        { value: "none", label: "ไม่ใช้ (คัดตามพื้นที่อย่างเดียว)" },
      ], default: "circularity" },
      { key: "lo", visibleWhen: [{key:"prop",notIn:["none"]}], type: "number", label: "ค่าต่ำสุดที่เก็บ", min: 0, max: 1, step: 0.01, default: 0.8, desc: "เก็บวัตถุที่ค่าที่เลือกไม่ต่ำกว่านี้ (รวมค่านี้)" },
      { key: "hi", visibleWhen: [{key:"prop",notIn:["none"]}], type: "number", label: "ค่าสูงสุดที่เก็บ", min: 0, max: 1, step: 0.01, default: 1, desc: "เก็บวัตถุที่ค่าที่เลือกไม่สูงกว่านี้ (รวมค่านี้)" },
      { key: "minArea", type: "number", label: "พื้นที่ขั้นต่ำ (นับพิกเซล)", min: 0, max: 100000, step: 10, default: 50, desc: "วัตถุที่มีพิกเซลขาวน้อยกว่านี้ถูกทิ้ง (กันจุด noise)" },
      { key: "maxArea", type: "number", label: "พื้นที่สูงสุด (นับพิกเซล)", min: 0, max: 5000000, step: 100, default: 0, hint: "0 = ไม่จำกัด", desc: "วัตถุที่มีพิกเซลขาวมากกว่านี้ถูกทิ้ง" },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "overlay", label: "ขาว = เก็บ · แดง = คัดทิ้ง" },
        { value: "mask", label: "Mask ขาว-ดำ (เฉพาะที่เก็บ)" },
      ], default: "overlay" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) {
        if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(SHAPE_FILTER_FIELDS, []); }
        return src.clone();
      }
      const prop = ownKey(SHAPE_FILTER_PROP_TH, p.prop, "circularity");
      const view = ownKey(SHAPE_FILTER_VIEWS, p.view, "overlay");
      let lo = numIn(p.lo, 0.8, 0, 1), hi = numIn(p.hi, 1, 0, 1);
      let minA = numIn(p.minArea, 50, 0, 100000), maxA = numIn(p.maxArea, 0, 0, 5000000);
      let rangeSwapped = false, areaSwapped = false;
      if (lo > hi) { const t = lo; lo = hi; hi = t; rangeSwapped = true; }
      if (maxA > 0 && maxA < minA) { const t = minA; minA = maxA; maxA = t; areaSwapped = true; }
      const W = src.cols, H = src.rows;
      let gray = null, bin = null, contours = null, hier = null, keptMask = null, result = null;
      try {
        gray = grayIfNeeded(cv, src);
        const g = gray || src;
        bin = new cv.Mat();
        contours = new cv.MatVector();
        hier = new cv.Mat();
        cv.threshold(g, bin, 127, 255, cv.THRESH_BINARY);
        cv.findContours(bin, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        keptMask = cv.Mat.zeros(H, W, cv.CV_8UC1);
        const total = contours.size();
        let kept = 0, removedArea = 0, removedShape = 0;
        for (let i = 0; i < total; i++) {
          const c = contours.get(i);
          try {
            const box = regionBox(c.data32S);
            if (box.w * box.h < minA) { removedArea++; continue; } // เร็วขึ้นอย่างเดียว: area_px ≤ w·h
            const s = regionPixelStats(cv, c, box, bin.data, g.data, W, prop === "solidity");
            if (s.n < minA || (maxA > 0 && s.n > maxA)) { removedArea++; continue; }
            if (prop !== "none") {
              let val;
              if (prop === "circularity") {
                const per = cv.arcLength(c, true);
                const A = regionScan(c.data32S).area;
                val = per === 0 ? 0 : (4 * Math.PI * A) / (per * per);
                val = Math.max(0, Math.min(1, val));
              } else {
                val = regionShapeValues(s, box)[prop];
              }
              if (!(val >= lo && val <= hi)) { removedShape++; continue; }
            }
            kept++;
            cv.drawContours(keptMask, contours, i, new cv.Scalar(255), cv.FILLED, cv.LINE_8, hier, 0);
          } finally {
            c.delete();
          }
        }
        cv.bitwise_and(keptMask, bin, keptMask); // รูของวัตถุที่เก็บยังดำ · วัตถุเกาะตามวัตถุชั้นนอก
        if (view === "mask") {
          result = keptMask;
          keptMask = null;
        } else {
          result = new cv.Mat(H, W, cv.CV_8UC3, new cv.Scalar(0, 0, 0));
          result.setTo(new cv.Scalar(255, 0, 0, 0), bin);
          result.setTo(new cv.Scalar(255, 255, 255, 0), keptMask);
        }
        if (ctx) {
          let info;
          if (total === 0) {
            info = "ไม่พบวัตถุ (ภาพดำล้วน — ต่อจาก Threshold ให้เป็นขาว-ดำก่อน)";
          } else {
            info = `เก็บ ${kept} จาก ${total} ชิ้น · คัดทิ้งเพราะพื้นที่ ${removedArea} ชิ้น`
              + (prop !== "none" ? ` · คัดทิ้งเพราะ${SHAPE_FILTER_PROP_TH[prop]} ${removedShape} ชิ้น (เก็บช่วง ${fixed2(lo)}–${fixed2(hi)})` : "")
              + ` · พื้นที่ที่เก็บ ${minA}–${maxA > 0 ? maxA : "ไม่จำกัด"} px`;
          }
          if (rangeSwapped) info += " · สลับค่าต่ำสุด/สูงสุดให้แล้ว";
          if (areaSwapped) info += " · สลับพื้นที่ขั้นต่ำ/สูงสุดให้แล้ว";
          if (view === "overlay") info += " · ขาว = เก็บ แดง = คัดทิ้ง (ต่อเข้า Region Properties ได้เลย สีแดงจะถูกตัดทิ้งที่ 127)";
          ctx.info = info;
          ctx.values = scalarValues(SHAPE_FILTER_FIELDS, [total, kept, removedArea, removedShape]);
        }
        return result;
      } catch (err) {
        if (result) result.delete();
        throw err;
      } finally {
        if (gray) gray.delete();
        if (bin) bin.delete();
        if (contours) contours.delete();
        if (hier) hier.delete();
        if (keptMask) keptMask.delete();
      }
    },
  },

  // roipoly ของ MATLAB — ROI หลายเหลี่ยม 3–6 จุด หน่วย % (สเปก D §4.3)
  roi_polygon: {
    name: "ROI Polygon (บริเวณที่สนใจแบบหลายเหลี่ยม)",
    needs: "any",
    desc: "กำหนดบริเวณที่สนใจ (ROI) เป็นรูปหลายเหลี่ยม 3–6 จุด แล้วตัดส่วนนอกกรอบทิ้ง — งานตรวจสอบด้วยกล้องแทบทุกงาน 'ดูเฉพาะบริเวณนี้' (เช่น ดูแค่ฉลากบนขวด) เพื่อไม่ให้พื้นหลังรบกวนผลวัดของขั้นถัดไป · แสดงผลได้ 3 แบบ: ตัดนอกกรอบเป็นดำ / Mask ขาว-ดำ (ต่อเข้าช่อง 'เข้า B' ของ Image Arithmetic → Mask ได้) / ดูตำแหน่ง (เส้นขอบเหลือง + เลขจุด นอกกรอบมืดลง) · ตัวเลขที่รายงาน: พื้นที่ ROI, เส้นรอบรูป, ความสว่างเฉลี่ยและส่วนเบี่ยงเบนภายใน ROI (ภาพสีคิดจากความสว่างแบบขาวดำ) · ต่อจุดตามลำดับ 1→2→…→กลับจุด 1 ให้เรียงตามเข็มหรือทวนเข็มนาฬิกา ถ้าลำดับไขว้กันรูปจะกลายเป็นโบว์ไท · คล้าย roipoly ของ MATLAB แต่ตั้งตำแหน่งจุดเป็น % ของขนาดภาพ (100% = พิกเซลสุดท้าย) ยังคลิกวาดบนภาพไม่ได้",
    params: [
      { key: "points", type: "number", label: "จำนวนจุด", min: 3, max: 6, step: 1, default: 4, desc: "ใช้จุดที่ 1 ถึงจุดนี้ ต่อกันตามลำดับแล้วปิดกลับไปจุดที่ 1 (จุดที่เกินจำนวนนี้ไม่ถูกใช้)" },
      ...ROI_DEFAULTS.flatMap(([dx, dy], i) => {
        const k = i + 1;
        const when = k >= 4 ? ` · ใช้เมื่อจำนวนจุด ≥ ${k}` : "";
        return [
          { key: `p${k}x`, type: "number", label: `จุด ${k} X (%)${when}`, min: 0, max: 100, step: 1, default: dx, desc: `ตำแหน่งแนวนอนของจุดที่ ${k} เป็น % ของความกว้างภาพ` },
          { key: `p${k}y`, type: "number", label: `จุด ${k} Y (%)${when}`, min: 0, max: 100, step: 1, default: dy, desc: `ตำแหน่งแนวตั้งของจุดที่ ${k} เป็น % ของความสูงภาพ` },
        ];
      }),
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "apply", label: "ตัดนอกกรอบเป็นดำ", hint: "ขั้นถัดไปเห็นเฉพาะบริเวณใน ROI" },
        { value: "mask", label: "Mask ขาว-ดำ (ต่อเข้า B ของ Mask ได้)", hint: "ขาว = ใน ROI · ดำ = นอก ROI" },
        { value: "overlay", label: "ดูตำแหน่ง (เส้นขอบ + เลขจุด)", hint: "ใช้ตอนกำลังจัดตำแหน่งจุด" },
      ], default: "apply" },
    ],
    apply(cv, src, p, ctx) {
      const n = intIn(p.points, 4, 3, 6);
      const view = ownKey(ROI_VIEWS, p.view, "apply");
      if (isEmptyMat(src)) {
        if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(ROI_FIELDS, [n]); }
        return src.clone();
      }
      const W = src.cols, H = src.rows;
      const q = new Int32Array(2 * n);
      for (let i = 0; i < n; i++) {
        q[2 * i] = pctToPx(p[`p${i + 1}x`], ROI_DEFAULTS[i][0], W);
        q[2 * i + 1] = pctToPx(p[`p${i + 1}y`], ROI_DEFAULTS[i][1], H);
      }
      let mask = null, poly = null, polyVec = null, gray = null, rgb = null, base = null, dst = null;
      try {
        mask = cv.Mat.zeros(H, W, cv.CV_8UC1);
        poly = cv.matFromArray(n, 1, cv.CV_32SC2, q);
        polyVec = new cv.MatVector();
        polyVec.push_back(poly);
        cv.fillPoly(mask, polyVec, new cv.Scalar(255)); // LINE_8 · โบว์ไทเติมทั้งสองข้าง · จุดซ้อนกัน = 1 px
        // วัดบนภาพขาวดำ: ผลรวมจำนวนเต็ม (Python ใช้ int64 ได้ค่าเดียวกันทุก bit)
        gray = grayIfNeeded(cv, src);
        const gd = (gray || src).data, md = mask.data;
        let a = 0, s = 0, s2 = 0;
        for (let i = 0; i < md.length; i++) {
          if (md[i] !== 0) { const v = gd[i]; a++; s += v; s2 += v * v; }
        }
        const mean = a > 0 ? s / a : null;
        const std = a > 0 ? Math.sqrt(Math.max(0, s2 / a - mean * mean)) : null; // ส่วนเบี่ยงเบนแบบประชากร
        const shape = roiShape(q, n);
        if (view === "mask") {
          dst = mask; // ส่งต่อความเป็นเจ้าของ (finally ไม่ลบ)
          mask = null;
        } else if (view === "overlay") {
          base = motionRGB(cv, src);
          dst = new cv.Mat();
          cv.convertScaleAbs(base, dst, 0.4, 0); // นอกกรอบมืดลง
          base.copyTo(dst, mask); // ในกรอบ = ภาพเดิม
          const YELLOW = new cv.Scalar(255, 255, 0), WHITE = new cv.Scalar(255, 255, 255);
          cv.polylines(dst, polyVec, true, YELLOW, 2);
          for (let i = 0; i < n; i++) {
            cv.circle(dst, new cv.Point(q[2 * i], q[2 * i + 1]), 3, YELLOW, -1);
            cv.putText(dst, String(i + 1), new cv.Point(q[2 * i] + 5, Math.max(10, q[2 * i + 1] - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1);
          }
        } else {
          rgb = dropAlpha(cv, src); // RGBA → RGB · ขาวดำคง 1 ช่อง
          const x = rgb || src;
          dst = cv.Mat.zeros(H, W, x.type());
          x.copyTo(dst, mask);
        }
        if (ctx) {
          let info = `ROI ${n} จุด · ${a} px (${fixed1((100 * a) / (W * H))}% ของภาพ) · ความสว่างเฉลี่ย ${a > 0 ? fixed1(mean) : "—"}`;
          if (view !== "overlay") info += " (เลือก 'แสดงผลเป็น: ดูตำแหน่ง' เพื่อเห็นเส้นขอบและเลขจุด)";
          // เรียงเป็นเส้นตรงแล้ว ด้านที่ซ้อนกันย่อมแตะกัน — เตือนเรื่องเส้นตรงอย่างเดียว (ข้อความไขว้จะชวนเข้าใจผิด)
          if (shape.degenerate) info += " · ⚠️ ทุกจุดเรียงเป็นเส้นตรง/ซ้อนกัน ROI จึงเหลือแค่เส้นบาง ๆ";
          else if (shape.crossed) info += " · ⚠️ ลำดับจุดไขว้กัน รูปจะไม่ใช่รูปที่ตั้งใจ — เรียงจุดตามเข็มหรือทวนเข็มนาฬิกา";
          ctx.info = info;
          ctx.values = scalarValues(ROI_FIELDS, [n, a, (100 * a) / (W * H), shape.perimeter, mean, std, shape.crossed]);
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        for (const m of [mask, poly, polyVec, gray, rgb, base]) if (m) m.delete();
      }
    },
  },

  // ===================================================================
  // IP batch 2 ชิ้น E (2026-10-04) — slic (FILTERS.colour) · bandpass_notch (FILTERS.frequency) · รวมไว้ท้าย OPS เหมือนชิ้น C/D
  // ===================================================================
  // superpixels ของ MATLAB — SLIC บนภาพย่อ ≤ 320 px แล้วขยายป้ายกำกับกลับ (สเปก E §4) · cv.ximgproc ไม่มีใน build นี้ จึงเขียน slicLabels เอง
  slic: {
    name: "SLIC Superpixels (แบ่งภาพเป็นก้อนเล็กตามสีและตำแหน่ง)",
    needs: "any",
    desc: "แบ่งภาพเป็นก้อนเล็ก ๆ ที่เชื่อมต่อกัน (superpixel) โดยดูทั้งสีและตำแหน่งพร้อมกัน — พิกเซลที่สีใกล้กันและอยู่ใกล้กัน (ในระยะราวหนึ่งช่องตาราง) มักอยู่ก้อนเดียวกัน — พื้นที่เรียบสีเดียวก็ยังถูกตัดเป็นชิ้นขนาดเท่าช่องตาราง ต่างจาก K-means ในกล่อง Colour ที่ดูแค่สี จึงรวมพิกเซลที่อยู่ไกลกันแต่สีคล้ายกันเข้ากลุ่มเดียว กลุ่มเลยกระจัดกระจาย · ค่า compactness มาก = ก้อนเป็นสี่เหลี่ยมสม่ำเสมอ น้อย = ก้อนเกาะตามขอบสีมากขึ้นแต่รูปร่างไม่เรียบ · superpixel ไม่ได้บอกว่าอะไรคือวัตถุ — เป็นขั้นเตรียมข้อมูล ลดพิกเซลหลายแสนจุดให้เหลือหน่วยไม่กี่ร้อยก้อน ให้ขั้นถัดไปทำงานง่ายขึ้น · จำนวนก้อนที่ได้จริงไม่เท่ากับที่ตั้งไว้เป๊ะ (ตารางจุดเริ่มต้นปัดเศษ และเศษชิ้นเล็กถูกรวมเข้าก้อนข้างเคียง) เหมือน superpixels ของ MATLAB ที่คืนจำนวนจริงมาให้ · ภาพที่ใหญ่กว่า 320 px คำนวณบนสำเนาที่ย่อแล้วขยายผลกลับ เส้นขอบจึงเป็นขั้นบันได · ต่างจากบทความต้นฉบับ (Achanta 2012) ตรงที่ไม่ขยับจุดเริ่มไปยังจุดที่ความชันต่ำสุดในละแวก 3×3 (กล่องนี้ไม่ได้ทำขั้นนี้) และวนรอบจำนวนคงที่ — ใกล้เคียง superpixels ของ MATLAB แต่ไม่ตรงกันทุกพิกเซล · superpixels ของ MATLAB ใช้ Method 'slic0' (ปรับ compactness อัตโนมัติ) เป็นค่าเริ่มต้น ส่วนกล่องนี้เทียบได้กับ Method 'slic' · ภาพขาวดำใช้ความสว่างอย่างเดียว",
    params: [
      { key: "count", type: "number", label: "จำนวน superpixel ที่ต้องการ (โดยประมาณ)", min: 10, max: 1000, step: 10, default: 200, desc: "จำนวนก้อนที่อยากได้ — จำนวนจริงที่ได้อาจต่างเล็กน้อย (ดูในช่องข้อความใต้ภาพ)" },
      { key: "compactness", type: "number", label: "ความสม่ำเสมอของก้อน (compactness)", min: 1, max: 40, step: 1, default: 10, desc: "มาก = ก้อนเป็นสี่เหลี่ยมสม่ำเสมอ · น้อย = ก้อนเกาะตามขอบสี" },
      { key: "iterations", type: "number", label: "จำนวนรอบปรับก้อน", min: 1, max: 20, step: 1, default: 10, desc: "จำนวนรอบที่ขยับศูนย์กลางก้อนให้เข้ากับสีและตำแหน่ง — มาก = ขอบแม่นขึ้นแต่ช้าขึ้น" },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "boundary", label: "เส้นขอบ superpixel บนภาพ", hint: "เส้นเหลืองคั่นระหว่างก้อน ภาพเดิมอยู่ด้านล่าง" },
        { value: "mean", label: "ระบายสีเฉลี่ยของแต่ละก้อน", hint: "ทุกพิกเซลในก้อนเดียวกันได้สีเฉลี่ยของก้อนนั้น" },
      ], default: "boundary" },
    ],
    apply(cv, src, p, ctx) {
      const K = intIn(p.count, 200, 10, 1000);
      const m = numIn(p.compactness, 10, 1, 40);
      const iters = intIn(p.iterations, 10, 1, 20);
      const view = ownKey(SLIC_VIEWS, p.view, "boundary");
      if (isEmptyMat(src)) { // cvtColor(RGB2Lab) / resize throw จริงบน 0×0
        if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(SLIC_FIELDS, [undefined, K]); }
        return src.clone();
      }
      const W = src.cols, H = src.rows;
      let rgbA = null, small = null, lab = null, dst = null;
      try {
        rgbA = dropAlpha(cv, src); // RGBA → RGB · ขาวดำ/สีคงเดิม
        const x = rgbA || src;
        const nch = x.channels();
        const scale = Math.min(1, SLIC_MAX_SIDE / Math.max(W, H));
        const aw = Math.max(1, Math.round(W * scale)), ah = Math.max(1, Math.round(H * scale));
        let analysis = x;
        if (scale < 1) {
          small = new cv.Mat();
          cv.resize(x, small, new cv.Size(aw, ah), 0, 0, cv.INTER_AREA);
          analysis = small;
        }
        let F;
        if (nch === 3) {
          lab = new cv.Mat();
          cv.cvtColor(analysis, lab, cv.COLOR_RGB2Lab);
          F = lab.data;
        } else {
          F = analysis.data;
        }
        const r = slicLabels(F, nch, aw, ah, K, m, iters);
        const labels = r.labels, count = r.count;
        // ขยายป้ายกำกับกลับขนาดจริงแบบ nearest neighbour ตำแหน่งกึ่งกลางพิกเซล (เลขจำนวนเต็มล้วน)
        const xa = new Int32Array(W), ya = new Int32Array(H);
        for (let i = 0; i < W; i++) xa[i] = Math.floor(((2 * i + 1) * aw) / (2 * W));
        for (let j = 0; j < H; j++) ya[j] = Math.floor(((2 * j + 1) * ah) / (2 * H));
        dst = view === "mean" ? new cv.Mat(H, W, x.type()) : new cv.Mat();
        if (view === "boundary") {
          if (nch === 1) cv.cvtColor(x, dst, cv.COLOR_GRAY2RGB);
          else x.copyTo(dst);
        }
        const xd = x.data, dd = dst.data; // อ่าน .data หลังจองทุก Mat แล้ว (heap โตแล้ว view เก่าใช้ไม่ได้)
        const fillRow = (row, y) => { const b = ya[y] * aw; for (let i = 0; i < W; i++) row[i] = labels[b + xa[i]]; };
        const rowA = new Int32Array(W), rowB = new Int32Array(W);
        if (view === "mean") {
          const sums = new Float64Array(count * nch), cnts = new Float64Array(count);
          for (let y = 0; y < H; y++) {
            fillRow(rowA, y);
            for (let i = 0; i < W; i++) {
              const l = rowA[i], o = (y * W + i) * nch;
              cnts[l] += 1;
              for (let c = 0; c < nch; c++) sums[l * nch + c] += xd[o + c];
            }
          }
          for (let y = 0; y < H; y++) {
            fillRow(rowA, y);
            for (let i = 0; i < W; i++) {
              const l = rowA[i], o = (y * W + i) * nch;
              for (let c = 0; c < nch; c++) dd[o + c] = Math.floor(sums[l * nch + c] / cnts[l] + 0.5);
            }
          }
        } else {
          let cur = rowA, nxt = rowB;
          fillRow(cur, 0);
          for (let y = 0; y < H; y++) {
            const hasNext = y + 1 < H;
            if (hasNext) fillRow(nxt, y + 1);
            for (let i = 0; i < W; i++) {
              if ((i + 1 < W && cur[i + 1] !== cur[i]) || (hasNext && nxt[i] !== cur[i])) {
                const o = (y * W + i) * 3;
                dd[o] = 255; dd[o + 1] = 255; dd[o + 2] = 0; // เหลือง (ลำดับช่องของแอป = RGB)
              }
            }
            const t = cur; cur = nxt; nxt = t;
          }
        }
        if (ctx) {
          let info = `แบ่งภาพเป็น ${count} superpixel (ตั้งไว้ ${K} · ตารางเริ่ม ${r.nx}×${r.ny} · รวมเศษชิ้นเล็ก ${r.merged} ชิ้นเข้ากับก้อนข้างเคียง)`;
          info += nch === 3 ? " · วัดความต่างสีในปริภูมิ Lab" : " · ภาพขาวดำ: ใช้ความสว่างอย่างเดียว";
          if (scale < 1) info += ` · คำนวณบนภาพย่อ ${aw}×${ah} แล้วขยายผลกลับ`;
          ctx.info = info;
          ctx.values = scalarValues(SLIC_FIELDS, [count, K, r.nx * r.ny, r.S / scale, (W * H) / count]);
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        if (rgbA) rgbA.delete();
        if (small) small.delete();
        if (lab) lab.delete();
      }
    },
  },

  // Band-pass / Band-reject / Notch (Gonzalez) — DFT 64 บิต ความละเอียดเต็มถึง 1920 px (สเปก E §5) · ไม่เคยตั้ง ctx.values
  bandpass_notch: {
    name: "Band-pass / Notch Filter (กรองช่วงความถี่ / ลบลายซ้ำ)",
    needs: "gray",
    desc: "กรองภาพในโดเมนความถี่แบบเจาะเป็นช่วง: Band-pass = High-pass ที่ขอบใน คูณ Low-pass ที่ขอบนอก (ต่อยอดจาก FFT Low/High-pass) เก็บเฉพาะ 'วงแหวน' ความถี่ตรงกลาง · Band-reject ตรงข้าม (ตัดวงแหวนทิ้ง) · Notch ตัดจุดความถี่เป็นคู่สมมาตร (สเปกตรัมของภาพที่เป็นค่าจริง (ภาพทั่วไป) สมมาตรรอบจุดกลางเสมอ) ใช้กับ noise ที่เป็นลายซ้ำ ๆ เช่น แถบรบกวนจากสัญญาณไฟ ลายผ้า ลายพิมพ์ ซึ่งฟิลเตอร์ในโดเมนพิกเซลลบไม่ได้ถ้าไม่ทำให้ภาพเบลอ · หน่วย = % ของ Nyquist (100% = ลายสลับขาว-ดำทีละ 1 px) ลายที่ซ้ำทุก T px อยู่ที่ประมาณ 200/T % (ซ้ำทุก 8 px → 25%) ต่างจากหน่วยของ FFT Low/High-pass · แกน y ของสเปกตรัมชี้ลง: ค่า ny เป็นบวก = อยู่ต่ำกว่าจุดกลาง · หาตำแหน่งจุดสว่างจากมุมมอง 'สเปกตรัมหลังกรอง' ของกล่องนี้เอง เพราะ FFT Magnitude ย่อภาพไม่เกิน 640 px ก่อนคำนวณ ตำแหน่งจุดจึงไม่ตรงกันเมื่อภาพใหญ่กว่า 640 px · กล่องนี้คำนวณที่ความละเอียดเต็มถึง 1920 px (ต่างจาก FFT Low/High-pass) จึงไม่เสียรายละเอียดจากการย่อภาพ ภาพที่ใหญ่กว่านั้นย่อก่อนแล้วขยายกลับ · แบบ Ideal อาจเห็น ringing ส่วน Gaussian นุ่มกว่า และ FFT มองภาพเป็นภาพวนซ้ำ จึงอาจเห็นแถบบาง ๆ ที่ริมภาพ · Band-pass ยืดค่าให้เต็ม 0–255 (เมื่อขอบในมากกว่า 0 ความสว่างเฉลี่ยถูกตัดออกไปด้วย) ส่วน Band-reject/Notch คงความสว่างเดิมถ้าไม่ได้ตัดจุดกลางสเปกตรัม (ถ้าตัด ภาพจะมืดลง และถ้าตัดมากจะมีคำเตือน) ค่าที่เกินถูกตัดที่ 0/255 · ภาพสีแปลงเป็นขาวดำก่อน · ลายซ้ำหลายคู่ให้ต่อกล่องนี้ซ้อนกัน (หนึ่งกล่อง = หนึ่งคู่จุด)",
    params: [
      { key: "kind", type: "select", label: "ชนิดตัวกรอง", options: [
        { value: "bandpass", label: "Band-pass (เก็บวงแหวน)", hint: "เก็บเฉพาะความถี่ในวงแหวน ตัดข้างในและข้างนอกทิ้ง" },
        { value: "bandreject", label: "Band-reject (ตัดวงแหวน)", hint: "ตัดความถี่ในวงแหวน เก็บส่วนที่เหลือ" },
        { value: "notch", label: "Notch (ตัดจุดคู่)", hint: "ตัดจุดความถี่เป็นคู่สมมาตร ใช้ลบลายซ้ำ" },
      ], default: "bandpass" },
      { key: "shape", type: "select", label: "รูปแบบขอบตัด", options: [
        { value: "gaussian", label: "Gaussian (นุ่ม)" },
        { value: "ideal", label: "Ideal (ตัดเฉียบขาด อาจมี ringing)" },
      ], default: "gaussian" },
      { key: "rIn", visibleWhen: [{key:"kind",in:["bandpass","bandreject"]}], type: "number", label: "ขอบในของวงแหวน (% ของ Nyquist)", min: 0, max: 100, step: 1, default: 5, desc: "ใช้กับ Band-pass / Band-reject — 0 = เริ่มจากจุดกลางสเปกตรัม · ถ้ามากกว่าขอบนอก ระบบสลับให้เอง" },
      { key: "rOut", visibleWhen: [{key:"kind",in:["bandpass","bandreject"]}], type: "number", label: "ขอบนอกของวงแหวน (% ของ Nyquist)", min: 1, max: 100, step: 1, default: 30, desc: "ใช้กับ Band-pass / Band-reject — ลายที่ซ้ำทุก T px อยู่ที่ประมาณ 200/T %" },
      { key: "nx", visibleWhen: [{key:"kind",in:["notch"]}], type: "number", label: "Notch ห่างจากกลางแนวนอน (%)", min: -100, max: 100, step: 0.5, default: 25, desc: "ใช้กับ Notch — ตำแหน่งจุดแรกของคู่ เทียบกับจุดกลางสเปกตรัม (จุดที่สองอยู่ฝั่งตรงข้ามเอง)" },
      { key: "ny", visibleWhen: [{key:"kind",in:["notch"]}], type: "number", label: "Notch ห่างจากกลางแนวตั้ง (%)", min: -100, max: 100, step: 0.5, default: 0, desc: "ใช้กับ Notch — แกนชี้ลงเหมือนภาพ: ค่าบวก = อยู่ต่ำกว่าจุดกลาง" },
      { key: "nr", visibleWhen: [{key:"kind",in:["notch"]}], type: "number", label: "รัศมีวง Notch (%)", min: 1, max: 30, step: 0.5, default: 5, desc: "ใช้กับ Notch — ขนาดของวงที่ตัด เป็น % ของ Nyquist" },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "result", label: "ภาพหลังกรอง" },
        { value: "spectrum", label: "สเปกตรัมหลังกรอง (log)", hint: "ส่วนที่มืดคือส่วนที่ถูกตัด ใช้เล็งตำแหน่ง Notch" },
      ], default: "result" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); } // cvtColor/resize throw จริงบน 0×0 · ไม่ตั้ง ctx.values เลย
      const kind = ownKey(BANDPASS_KINDS, p.kind, "bandpass");
      const shape = ownKey(BANDPASS_SHAPES, p.shape, "gaussian");
      const view = ownKey(BANDPASS_VIEWS, p.view, "result");
      const rIn = numIn(p.rIn, 5, 0, 100), rOut = numIn(p.rOut, 30, 1, 100);
      const prm = { lo: Math.min(rIn, rOut), hi: Math.max(rIn, rOut), nx: numIn(p.nx, 25, -100, 100), ny: numIn(p.ny, 0, -100, 100), nr: numIn(p.nr, 5, 1, 30) };
      const swapped = kind !== "notch" && rIn > rOut;
      const W = src.cols, H = src.rows;
      const scale = Math.min(1, BANDPASS_MAX_SIDE / Math.max(W, H));
      const aw = Math.max(1, Math.round(W * scale)), ah = Math.max(1, Math.round(H * scale));
      let gray = null, small = null, padded = null, f = null, spec = null, inv = null, u8 = null, dst = null;
      try {
        gray = grayIfNeeded(cv, src); // worker ส่งขาวดำให้อยู่แล้ว — ผู้เรียกตรงอาจส่งภาพสี
        let g = gray || src;
        if (scale < 1) {
          small = new cv.Mat();
          cv.resize(g, small, new cv.Size(aw, ah), 0, 0, cv.INTER_AREA);
          g = small;
        }
        let P = cv.getOptimalDFTSize(aw), Q = cv.getOptimalDFTSize(ah);
        if (P % 2 !== 0) P += 1; // กลางสเปกตรัม = P/2 ต้องเป็นจำนวนเต็ม
        if (Q % 2 !== 0) Q += 1;
        padded = new cv.Mat();
        cv.copyMakeBorder(g, padded, 0, Q - ah, 0, P - aw, cv.BORDER_CONSTANT, new cv.Scalar(0));
        f = new cv.Mat();
        padded.convertTo(f, cv.CV_64F);
        padded.delete(); padded = null;
        spec = new cv.Mat();
        cv.dft(f, spec, cv.DFT_COMPLEX_OUTPUT); // 64 บิต: ผลตรงกับ Python ทุกพิกเซล (32 บิตต่างได้ ±1)
        f.delete(); f = null;
        const hp = P / 2, hq = Q / 2;
        const fxs = new Float64Array(P);
        for (let x = 0; x < P; x++) fxs[x] = (x < hp ? x : x - P) / hp; // พิกัดแบบไม่ shift: DC อยู่ที่ (0,0)
        const dcGain = bandGain(kind, shape, prm, 0, 0);
        if (view === "result") {
          const sd = spec.data64F;
          for (let y = 0; y < Q; y++) {
            const fy = (y < hq ? y : y - Q) / hq;
            for (let x = 0; x < P; x++) {
              const h = bandGain(kind, shape, prm, fxs[x], fy);
              const i = 2 * (y * P + x);
              sd[i] *= h; sd[i + 1] *= h;
            }
          }
          inv = new cv.Mat();
          cv.dft(spec, inv, cv.DFT_INVERSE + cv.DFT_SCALE + cv.DFT_REAL_OUTPUT);
          spec.delete(); spec = null;
          u8 = new cv.Mat(ah, aw, cv.CV_8UC1);
          const iv = inv.data64F, od = u8.data;
          let flat = false;
          if (kind === "bandpass") {
            let mn = Infinity, mx = -Infinity;
            for (let y = 0; y < ah; y++) for (let x = 0; x < aw; x++) { const v = iv[y * P + x]; if (v < mn) mn = v; if (v > mx) mx = v; }
            const range = mx - mn;
            if (range < 1) {
              flat = true; od.fill(0);
            } else {
              const k = 255 / range;
              for (let y = 0; y < ah; y++) for (let x = 0; x < aw; x++) od[y * aw + x] = Math.floor((iv[y * P + x] - mn) * k + 0.5);
            }
          } else {
            for (let y = 0; y < ah; y++) for (let x = 0; x < aw; x++) od[y * aw + x] = Math.min(255, Math.max(0, Math.floor(iv[y * P + x] + 0.5)));
          }
          if (scale < 1) {
            dst = new cv.Mat();
            cv.resize(u8, dst, new cv.Size(W, H), 0, 0, cv.INTER_LINEAR);
          } else {
            dst = u8; u8 = null; // ส่งต่อความเป็นเจ้าของ (finally ไม่ลบ)
          }
          if (ctx) ctx.info = bandpassInfo(kind, shape, prm, P, Q, swapped, scale, aw, ah, view, dcGain, flat);
        } else {
          const sd = spec.data64F;
          const lm = new Float64Array(P * Q);
          let mn = Infinity, mx = -Infinity;
          for (let y = 0; y < Q; y++) {
            const fy = (y < hq ? y : y - Q) / hq;
            const rowOut = ((y + hq) % Q) * P;
            for (let x = 0; x < P; x++) {
              const h = bandGain(kind, shape, prm, fxs[x], fy);
              const i = 2 * (y * P + x);
              sd[i] *= h; sd[i + 1] *= h;
              const v = Math.log(1 + Math.sqrt(sd[i] * sd[i] + sd[i + 1] * sd[i + 1]));
              lm[rowOut + (x + hp) % P] = v;
              if (v < mn) mn = v;
              if (v > mx) mx = v;
            }
          }
          u8 = new cv.Mat(Q, P, cv.CV_8UC1);
          const od = u8.data;
          const range = mx - mn;
          if (range < 1e-3) {
            od.fill(0);
          } else {
            const k = 255 / range;
            for (let i = 0; i < lm.length; i++) od[i] = Math.floor((lm[i] - mn) * k + 0.5);
          }
          dst = new cv.Mat();
          cv.resize(u8, dst, new cv.Size(W, H), 0, 0, cv.INTER_LINEAR);
          if (ctx) ctx.info = bandpassInfo(kind, shape, prm, P, Q, swapped, scale, aw, ah, view, dcGain, false);
        }
        return dst;
      } catch (err) {
        if (dst) dst.delete();
        throw err;
      } finally {
        for (const m of [gray, small, padded, f, spec, inv, u8]) if (m) m.delete();
      }
    },
  },

  // ===================================================================
  // IP batch 4A (2026-10-15) — 5 op ใหม่: guided_filter · focus_score · pseudocolor · delta_e · roi_apply
  // ===================================================================
  // imguidedfilter ของ MATLAB (แบบ self-guided) — He et al. 2010 ผ่าน guidedFilterF32 (ตัวช่วยกลาง batch 4B ใช้ต่อ)
  guided_filter: {
    name: "Guided Filter (เกลี่ยแบบรักษาขอบ)",
    needs: "any",
    desc: "เกลี่ยภาพให้เรียบแต่คงขอบไว้: ใช้ 'ภาพนำทาง' (guide) บอกว่าขอบอยู่ตรงไหน แล้วเกลี่ยเฉพาะบริเวณเรียบ — กล่องนี้ใช้ภาพตัวเองเป็นภาพนำทาง (self-guided แบบค่าเริ่มต้นของ imguidedfilter ใน MATLAB) · ผลคล้าย Bilateral แต่คำนวณด้วย box filter ล้วน ความเร็วจึงไม่ขึ้นกับขนาดหน้าต่าง · ภาพสีทำทีละช่อง R/G/B (เหมือน imguidedfilter ของ MATLAB เมื่อภาพและภาพนำทางเป็นภาพสีทั้งคู่) · ขอบภาพใช้การสะท้อนค่า · เป็นตัวช่วยของ Dehaze ในชุดถัดไป (ปรับแผนที่ความโปร่งใสให้ตามขอบวัตถุ)",
    params: [
      { key: "radius", type: "number", label: "รัศมีหน้าต่าง r", min: 1, max: 20, step: 1, default: 2, desc: "หน้าต่าง (2r+1)×(2r+1) — MATLAB ค่าเริ่มต้น 5×5 (r = 2)" },
      { key: "eps", type: "number", label: "ความเรียบ ε", min: 0.001, max: 1, step: 0.001, default: 0.01, hint: "MATLAB ค่าเริ่มต้น 0.01", desc: "ขอบที่ต่างกันน้อยกว่าราว √ε × 255 ระดับจะถูกเกลี่ย ขอบที่ต่างกว่านั้นถูกรักษาไว้" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const r = intIn(p.radius, 2, 1, 20);
      const eps = numIn(p.eps, 0.01, 0.001, 1);
      let rgb = null, F = null, q = null, vec = null, qv = null, dst = null;
      try {
        rgb = dropAlpha(cv, src);
        const x = rgb || src;
        F = new cv.Mat();
        x.convertTo(F, cv.CV_32F); // ไม่ scale — ค่า 0–255 ตรง ๆ แม่นเป๊ะ
        const eps255 = eps * 65025;
        if (x.channels() === 1) {
          q = guidedFilterF32(cv, F, F, r, eps255);
        } else {
          vec = new cv.MatVector();
          cv.split(F, vec);
          qv = new cv.MatVector();
          for (let c = 0; c < 3; c++) {
            const ch = vec.get(c);
            let qc = null;
            try {
              qc = guidedFilterF32(cv, ch, ch, r, eps255);
              qv.push_back(qc);
            } finally {
              ch.delete();
              if (qc) qc.delete();
            }
          }
          q = new cv.Mat();
          cv.merge(qv, q);
        }
        dst = new cv.Mat(x.rows, x.cols, x.type());
        const od = dst.data, qd = q.data32F, len = qd.length;
        for (let i = 0; i < len; i++) {
          const v = Math.floor(qd[i] + 0.5);
          od[i] = v < 0 ? 0 : v > 255 ? 255 : v;
        }
        if (ctx) ctx.info = `Guided filter r=${r} (หน้าต่าง ${2 * r + 1}×${2 * r + 1}) · ε=${eps} — ขอบที่ต่างกันเกินราว ${fixed1(Math.sqrt(eps) * 255)} ระดับสีถูกรักษาไว้`;
        const out = dst; dst = null;
        return out;
      } finally {
        if (dst) dst.delete();
        if (q) q.delete();
        if (F) F.delete();
        if (vec) vec.delete();
        if (qv) qv.delete();
        if (rgb) rgb.delete();
      }
    },
  },

  // วัดความคม (variance of Laplacian / Tenengrad) บนกล่องกลางภาพ — ไม่อยู่ใน MATLAB IPT แต่ใช้ทั่วไปในงาน Machine Vision
  focus_score: {
    name: "Focus / Sharpness Score (วัดความคมชัด)",
    needs: "gray",
    desc: "วัดว่าภาพคมชัดแค่ไหนเป็นตัวเลขเดียว: ภาพที่โฟกัสดีมีขอบคม → Laplacian แกว่งแรง → ความแปรปรวนสูง ภาพเบลอได้ค่าต่ำ · ตัวเลขไม่มีหน่วยและไม่มีค่าตายตัวที่แปลว่า \"คม\" — ขึ้นกับฉาก แสง ขนาดภาพ จึงใช้เทียบภาพของฉากเดียวกัน เช่น หมุนโฟกัสหาค่าสูงสุด (autofocus) หรือคัดภาพเบลอทิ้ง · ภาพสว่าง/คอนทราสต์สูงขึ้น คะแนนก็สูงขึ้นแม้คมเท่าเดิม · noise ทำให้คะแนนสูงเกินจริง — ต่อ Gaussian Blur 3×3 ก่อนถ้า noise เยอะ · ไม่อยู่ใน MATLAB IPT แต่ใช้ทั่วไปในงาน Machine Vision",
    params: [
      { key: "method", type: "select", label: "วิธีวัด", options: [
        { value: "lap_var", label: "ความแปรปรวนของ Laplacian (นิยมใช้)" },
        { value: "tenengrad", label: "Tenengrad (ค่าเฉลี่ยของ gx² + gy² จาก Sobel)" },
      ], default: "lap_var" },
      { key: "roiPct", type: "number", label: "ขนาดบริเวณกลางภาพที่วัด (%)", min: 10, max: 100, step: 5, default: 100, desc: "100 = ทั้งภาพ · 50 = กล่องกลางภาพกว้าง/สูงครึ่งหนึ่ง — ใช้เมื่อสนใจเฉพาะวัตถุกลางเฟรม" },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "lap", label: "ภาพขอบ |Laplacian| (ตรงไหนคมจะสว่าง)" },
        { value: "pass", label: "ภาพเดิม (ใช้แค่ตัวเลข)" },
      ], default: "lap" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) {
        if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(FOCUS_FIELDS, []); }
        return src.clone();
      }
      const method = ownKey(FOCUS_METHODS, p.method, "lap_var");
      const view = ownKey(FOCUS_VIEWS, p.view, "lap");
      const pct = numIn(p.roiPct, 100, 10, 100);
      let gray = null, crop = null, L = null, gx = null, gy = null, L2 = null, dst = null;
      try {
        gray = grayIfNeeded(cv, src);
        const g = gray || src;
        const W = g.cols, H = g.rows;
        const rw = Math.max(1, Math.floor(W * pct / 100 + 0.5)), rh = Math.max(1, Math.floor(H * pct / 100 + 0.5));
        const x0 = Math.floor((W - rw) / 2), y0 = Math.floor((H - rh) / 2);
        const n = rw * rh;
        // ต้อง clone — Laplacian/Sobel บน roi() อ่านพิกเซลนอกกรอบเป็นขอบ แต่ Python ตัดภาพแล้วใช้ขอบสะท้อนในตัว
        const roi = g.roi(new cv.Rect(x0, y0, rw, rh));
        try { crop = roi.clone(); } finally { roi.delete(); }
        let score;
        if (method === "tenengrad") {
          gx = new cv.Mat(); gy = new cv.Mat();
          cv.Sobel(crop, gx, cv.CV_16S, 1, 0, 3);
          cv.Sobel(crop, gy, cv.CV_16S, 0, 1, 3);
          const ax = gx.data16S, ay = gy.data16S;
          let s = 0;
          for (let i = 0; i < n; i++) s += ax[i] * ax[i] + ay[i] * ay[i];
          score = s / n;
        } else {
          L = new cv.Mat();
          cv.Laplacian(crop, L, cv.CV_16S, 1);
          const a = L.data16S;
          let s1 = 0, s2 = 0;
          for (let i = 0; i < n; i++) { const v = a[i]; s1 += v; s2 += v * v; }
          score = s2 / n - (s1 / n) * (s1 / n);
        }
        const cd = crop.data;
        let sm = 0;
        for (let i = 0; i < n; i++) sm += cd[i];
        const mean = sm / n;
        dst = new cv.Mat();
        if (view === "pass") {
          g.copyTo(dst);
        } else {
          L2 = new cv.Mat();
          cv.Laplacian(g, L2, cv.CV_16S, 1);
          cv.convertScaleAbs(L2, dst);
        }
        if (ctx) {
          ctx.info = `คะแนนความคม (${method === "tenengrad" ? "Tenengrad" : "variance of Laplacian"}) = ${fixed1(score)} · วัดในกล่อง ${rw}×${rh} px กลางภาพ · ใช้เทียบภาพขนาดเดียวกัน ฉากเดียวกันเท่านั้น` + (score === 0 ? " · ภาพเรียบสนิท (ไม่มีขอบเลย)" : "");
          ctx.values = scalarValues(FOCUS_FIELDS, [score, mean]);
        }
        const out = dst; dst = null;
        return out;
      } finally {
        if (dst) dst.delete();
        for (const m of [gray, crop, L, gx, gy, L2]) if (m) m.delete();
      }
    },
  },

  // ind2rgb / colormap ของ MATLAB — ตารางสร้างเองจากสูตรเส้นตรง (opencv.js build นี้ไม่มี applyColorMap)
  pseudocolor: {
    name: "Pseudocolor (ระบายสีตามค่าเทา)",
    needs: "gray",
    desc: "ตาคนแยกระดับเทาที่ใกล้กันได้ยาก แต่แยกสีง่าย การระบายสีตามค่าจึงช่วยอ่านภาพอย่าง Distance Transform, ผลต่างภาพ, LoG · สีไม่ได้เพิ่มข้อมูล — ถ้าต่อขั้นที่ต้องการภาพเทา จะได้ความสว่างของสีแทนค่าเดิม · \"ยืดช่วง\" ทำให้เห็นความต่างเล็ก ๆ แต่สีเดียวกันในสองภาพอาจหมายถึงค่าต่างกัน · ใกล้เคียง ind2rgb(I, jet(256)) ของ MATLAB (jet ของแอปเป็นสูตรเส้นตรง ไม่ใช่ตารางของ MATLAB ทุกค่า) · OpenCV.js ชุดนี้ไม่มี applyColorMap จึงสร้างตารางเอง",
    params: [
      { key: "map", type: "select", label: "ชุดสี", options: [
        { value: "jet", label: "Jet (น้ำเงิน → ฟ้า → เหลือง → แดง)" },
        { value: "hot", label: "Hot (ดำ → แดง → เหลือง → ขาว)" },
        { value: "cool", label: "Cool (ฟ้า → ชมพู)" },
      ], default: "jet" },
      { key: "range", type: "select", label: "ช่วงค่า", options: [
        { value: "full", label: "ตามค่าจริง 0–255" },
        { value: "minmax", label: "ยืดค่าต่ำ–สูงของภาพให้เต็มแถบสี" },
      ], default: "full" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const mapId = ownKey(PSEUDOCOLOR_MAPS, p.map, "jet");
      const minmax = p.range === "minmax";
      let gray = null, dst = null;
      try {
        gray = grayIfNeeded(cv, src);
        const g = gray || src;
        const gd = g.data, n = g.rows * g.cols;
        let lo = 0, hi = 255;
        if (minmax) {
          lo = 255; hi = 0;
          for (let i = 0; i < n; i++) { const v = gd[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
        }
        const tab = new Uint8Array(768); // R,G,B ต่อค่า 0–255 (หลังยืดช่วง)
        for (let i = 0; i < 256; i++) tab.set(PSEUDOCOLOR_MAPS[mapId](i), i * 3);
        const idxOf = (v) => (minmax ? (hi > lo ? Math.floor(((v - lo) * 510 + (hi - lo)) / (2 * (hi - lo))) : 0) : v);
        dst = new cv.Mat(g.rows, g.cols, cv.CV_8UC3);
        const od = dst.data;
        for (let i = 0; i < n; i++) {
          const t = idxOf(gd[i]) * 3, o = i * 3;
          od[o] = tab[t]; od[o + 1] = tab[t + 1]; od[o + 2] = tab[t + 2];
        }
        if (ctx) {
          const ends = PSEUDOCOLOR_ENDS[mapId];
          let info = `${PSEUDOCOLOR_TH[mapId]}: ค่า 0 = ${ends[0]} → 255 = ${ends[1]} · สีเป็นแค่การแสดงผล ข้อมูลยังเป็นค่าเทาเดิม`;
          if (minmax) {
            info += ` · ยืดช่วง ${lo}–${hi} ให้เต็มแถบสี`;
            if (hi === lo) info += ` · ภาพมีค่าเดียว (${lo}) จึงได้สีต้นแถบทั้งภาพ`;
          }
          ctx.info = info;
        }
        const out = dst; dst = null;
        return out;
      } finally {
        if (dst) dst.delete();
        if (gray) gray.delete();
      }
    },
  },

  // imcolordiff ของ MATLAB — ΔE ระหว่างสองภาพ (CIE76 / CIEDE2000) · A = ภาพหลัก, B = ctx.src2 (ของยืม ห้าม delete)
  delta_e: {
    name: "Color Difference ΔE (ความต่างสี)",
    needs: "any",
    desc: "วัดความต่างของสีสองภาพเป็นตัวเลขเดียวต่อพิกเซล (ΔE) · แปลงเป็น Lab ก่อน เพราะระยะใน Lab ใกล้การรับรู้ของคนกว่า RGB · ตัวเลขช่วยอ่าน: ต่ำกว่า 1 แทบมองไม่ออก · ราว 2–3 เริ่มเห็นเมื่อวางคู่กัน · เกิน 5 เห็นชัด · เกิน 10 คนละสี (ค่าประมาณ ขึ้นกับสีและแสง) · CIEDE2000 แก้ข้อที่ CIE76 ให้ค่าสูงเกินในสีอิ่มจัด ใช้ในงานตรวจสีสินค้าปัจจุบัน · \"สีเฉลี่ยของ B\" ใช้ตรวจว่าชิ้นงานสีตรงกับชิ้นตัวอย่างไหม: ตัดชิ้นตัวอย่างด้วย Crop แล้วต่อเข้า B · กล้องที่ปรับแสง/White Balance อัตโนมัติทำให้ ΔE เปลี่ยนเองแม้ของจริงไม่เปลี่ยน — ล็อกการตั้งค่ากล้องก่อนวัดจริง · เทียบ deltaE (CIE76) ของ MATLAB · imcolordiff ของ MATLAB ค่าเริ่มต้นเป็น CIE94 ซึ่งแอปไม่มี ต้องตั้ง \"ciede2000\" ถึงจะเทียบกับโหมด CIEDE2000 ได้ · ภาพ Lab 8 bit ใน Color Space Convert ใช้วัดแบบนี้ไม่ได้เพราะถูกบีบสเกล",
    params: [
      { key: "formula", type: "select", label: "สูตร", options: [
        { value: "cie76", label: "CIE76 (ระยะตรงใน Lab)" },
        { value: "ciede2000", label: "CIEDE2000 (ปรับตามการรับรู้ — มาตรฐานปัจจุบัน)", hint: "ช้ากว่า CIE76 ราว 3–4 เท่า — กล้องสดใช้ CIE76 หรือโหมดเทียบสีเฉลี่ย B" },
      ], default: "cie76" },
      { key: "compare", type: "select", label: "เทียบแบบ", options: [
        { value: "pixel", label: "ทีละพิกเซล (A กับ B ตำแหน่งเดียวกัน)" },
        { value: "mean_b", label: "ทุกพิกเซลของ A กับสีเฉลี่ยของ B (B เป็นชิ้นตัวอย่างขนาดใดก็ได้)" },
      ], default: "pixel" },
      { key: "tol", type: "number", label: "เกณฑ์ ΔE ที่ถือว่าต่าง", min: 0, max: 50, step: 0.1, default: 2.3, hint: "≈ 2.3 = เริ่มเห็นความต่างด้วยตา (CIE76)" },
      { key: "scaleMax", type: "number", label: "ΔE ที่แสดงเป็นสีขาว", min: 1, max: 100, step: 1, default: 20 },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "heatmap", label: "แผนที่ความต่าง (ยิ่งขาวยิ่งต่าง)" },
        { value: "over", label: "Mask จุดที่ต่างเกินเกณฑ์" },
      ], default: "heatmap" },
    ],
    apply(cv, src, p, ctx) {
      const pass = passthroughNoSrc2(src, ctx);
      if (pass) { if (ctx) ctx.values = scalarValues(DELTA_E_FIELDS, []); return pass; }
      if (isEmptyMat(src)) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(DELTA_E_FIELDS, []); return src.clone(); }
      const formula = ownKey(DELTA_E_FORMULAS, p.formula, "cie76");
      const compare = ownKey(DELTA_E_COMPARES, p.compare, "pixel");
      const view = ownKey(DELTA_E_VIEWS, p.view, "heatmap");
      const tol = numIn(p.tol, 2.3, 0, 50), scaleMax = numIn(p.scaleMax, 20, 1, 100);
      const fn = formula === "ciede2000" ? deltaE2000 : deltaE76;
      let rgbA = null, rgbB = null, matched = null, labA = null, labB = null, dst = null;
      try {
        rgbA = rgbIfNeeded(cv, src);
        rgbB = rgbIfNeeded(cv, ctx.src2);
        const a = rgbA || src;
        const bRaw = rgbB || ctx.src2;
        let b = bRaw, resized = false;
        if (compare === "pixel") {
          const m = matchSize(cv, a, bRaw);
          b = m.mat; resized = m.resized;
          matched = resized ? m.mat : null;
        }
        labA = toLabF32(cv, a);
        labB = toLabF32(cv, b);
        const N = a.rows * a.cols, nb = b.rows * b.cols;
        const la = labA.data32F, lb = labB.data32F;
        let sAL = 0, sAa = 0, sAb = 0, sBL = 0, sBa = 0, sBb = 0;
        for (let i = 0; i < N; i++) { sAL += la[i * 3]; sAa += la[i * 3 + 1]; sAb += la[i * 3 + 2]; }
        for (let i = 0; i < nb; i++) { sBL += lb[i * 3]; sBa += lb[i * 3 + 1]; sBb += lb[i * 3 + 2]; }
        const mA = [sAL / N, sAa / N, sAb / N], mB = [sBL / nb, sBa / nb, sBb / nb];
        dst = new cv.Mat(a.rows, a.cols, cv.CV_8UC1);
        const od = dst.data;
        const heat = view === "heatmap";
        let sum = 0, max = 0, over = 0;
        for (let i = 0; i < N; i++) {
          const j = compare === "pixel" ? i * 3 : -1;
          const dE = j >= 0 ? fn(la[i * 3], la[i * 3 + 1], la[i * 3 + 2], lb[j], lb[j + 1], lb[j + 2]) : fn(la[i * 3], la[i * 3 + 1], la[i * 3 + 2], mB[0], mB[1], mB[2]);
          sum += dE;
          if (dE > max) max = dE;
          if (dE > tol) over++;
          od[i] = heat ? Math.min(255, Math.floor(dE * 255 / scaleMax + 0.5)) : (dE > tol ? 255 : 0);
        }
        const mean = sum / N, pct = (over * 100) / N;
        if (ctx) {
          let info = `ΔE (${formula === "ciede2000" ? "CIEDE2000" : "CIE76"}) เฉลี่ย ${fixed2(mean)} · สูงสุด ${fixed2(max)} · เกินเกณฑ์ ${tol} อยู่ ${fixed1(pct)}% ของภาพ`;
          if (compare === "mean_b") info += ` · เทียบกับสีเฉลี่ยของ B (L* ${fixed1(mB[0])} a* ${fixed1(mB[1])} b* ${fixed1(mB[2])})`;
          if (resized) info += RESIZED_B_INFO;
          ctx.info = info;
          ctx.values = scalarValues(DELTA_E_FIELDS, [mean, max, pct, mA[0], mA[1], mA[2], mB[0], mB[1], mB[2]]);
        }
        const out = dst; dst = null;
        return out;
      } finally {
        if (dst) dst.delete();
        for (const m of [labA, labB, matched, rgbA, rgbB]) if (m) m.delete();
      }
    },
  },

  // roifilt2 ของ MATLAB — ผล B (ภาพที่กรองแล้ว) ในกรอบ, ภาพ A นอกกรอบ · A = ภาพเดิม, B = ภาพที่ผ่านตัวกรอง
  roi_apply: {
    name: "Apply in ROI (ใช้ผลเฉพาะในกรอบ)",
    needs: "any",
    desc: "ใช้ผลของตัวกรองเฉพาะในกรอบที่กำหนด ส่วนนอกกรอบคงภาพเดิม · วิธีต่อสาย: ภาพเดียวกันแตกสองทาง ทางหนึ่งเข้า A ตรง ๆ อีกทางผ่านตัวกรองที่อยากใช้ (เช่น Gaussian Blur เบลอป้ายทะเบียน) แล้วเข้า B · ต่างจาก Mask ที่ทำให้นอกกรอบดำ — กล่องนี้คงภาพเดิมไว้นอกกรอบ · เทียบ roifilt2 ของ MATLAB (MATLAB รับตัวกรองเป็นพารามิเตอร์ แอปให้เลือกตัวกรองอะไรก็ได้ในสาย B) · กรอบกำหนดเป็น % ของภาพ ยังลากบนภาพไม่ได้ · ถ้าต้องการบริเวณรูปทรงอิสระ ใช้ ROI Polygon หรือ Threshold สร้าง mask แล้วต่อ Mask · ภาพ B ต่างขนาดถูกย่อ/ขยายให้เท่า A",
    params: [
      { key: "shape", type: "select", label: "รูปกรอบ", options: [
        { value: "rect", label: "สี่เหลี่ยม" },
        { value: "ellipse", label: "วงรี (พอดีกรอบ)" },
      ], default: "rect" },
      { key: "x", type: "number", label: "มุมซ้ายบนของกรอบ แนวนอน (%)", min: 0, max: 100, step: 1, default: 25 },
      { key: "y", type: "number", label: "มุมซ้ายบนของกรอบ แนวตั้ง (%)", min: 0, max: 100, step: 1, default: 25 },
      { key: "w", type: "number", label: "ความกว้างของกรอบ (%)", min: 1, max: 100, step: 1, default: 50 },
      { key: "h", type: "number", label: "ความสูงของกรอบ (%)", min: 1, max: 100, step: 1, default: 50 },
      { key: "invert", type: "boolean", label: "ใช้ผล B นอกกรอบแทน", default: false },
      { key: "feather", type: "number", label: "ขอบนุ่ม (px)", min: 0, max: 50, step: 1, default: 0, desc: "0 = ขอบคม · มากขึ้น = ค่อย ๆ ผสมตรงรอยต่อ" },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "result", label: "ผลลัพธ์" },
        { value: "outline", label: "ผลลัพธ์ + เส้นกรอบสีเหลือง" },
      ], default: "result" },
    ],
    apply(cv, src, p, ctx) {
      const pass = passthroughNoSrc2(src, ctx);
      if (pass) { if (ctx) ctx.values = scalarValues(ROI_APPLY_FIELDS, []); return pass; }
      if (isEmptyMat(src)) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(ROI_APPLY_FIELDS, []); return src.clone(); }
      const shape = ownKey(ROI_APPLY_SHAPES, p.shape, "rect");
      const view = ownKey(ROI_APPLY_VIEWS, p.view, "result");
      const invert = boolOr(p.invert, false);
      const feather = intIn(p.feather, 0, 0, 50);
      const W = src.cols, H = src.rows;
      // กรอบเป็นพิกเซล (จำนวนเต็ม) — x0/y0 ไม่เกิน W−1/H−1 เพื่อให้กรอบมีอย่างน้อย 1 px เสมอ
      const x0 = Math.min(W - 1, Math.floor(W * numIn(p.x, 25, 0, 100) / 100));
      const y0 = Math.min(H - 1, Math.floor(H * numIn(p.y, 25, 0, 100) / 100));
      const x1 = Math.min(W, x0 + Math.max(1, Math.floor(W * numIn(p.w, 50, 1, 100) / 100 + 0.5)));
      const y1 = Math.min(H, y0 + Math.max(1, Math.floor(H * numIn(p.h, 50, 1, 100) / 100 + 0.5)));
      const ww = x1 - x0, hh = y1 - y0;
      let rgbA = null, rgbB = null, matched = null, maskMat = null, mf = null, mb = null, dst = null, rgbOut = null;
      try {
        const colour = src.channels() !== 1 || ctx.src2.channels() !== 1;
        let a = src, bRaw = ctx.src2;
        if (colour) {
          rgbA = rgbIfNeeded(cv, src);
          rgbB = rgbIfNeeded(cv, ctx.src2);
          a = rgbA || src; bRaw = rgbB || ctx.src2;
        }
        const m = matchSize(cv, a, bRaw);
        matched = m.resized ? m.mat : null;
        const b = m.mat;
        // inside[i] = อยู่ในกรอบ (ก่อน invert) · use[i] = ใช้ผล B (หลัง invert)
        const N = W * H;
        const inside = new Uint8Array(N);
        for (let j = y0; j < y1; j++) {
          for (let i = x0; i < x1; i++) {
            let ok = true;
            if (shape === "ellipse") {
              const dx = 2 * i + 1 - (x0 + x1), dy = 2 * j + 1 - (y0 + y1);
              ok = dx * dx * hh * hh + dy * dy * ww * ww <= ww * ww * hh * hh;
            }
            if (ok) inside[j * W + i] = 1;
          }
        }
        const maskData = new Uint8Array(N);
        let raCount = 0;
        for (let i = 0; i < N; i++) { const u = invert ? !inside[i] : inside[i]; if (u) { maskData[i] = 255; raCount++; } }
        if (feather === 0) {
          dst = a.clone();
          maskMat = cv.matFromArray(H, W, cv.CV_8UC1, maskData);
          b.copyTo(dst, maskMat);
        } else {
          const mfData = new Float32Array(N);
          for (let i = 0; i < N; i++) mfData[i] = maskData[i] / 255;
          mf = cv.matFromArray(H, W, cv.CV_32FC1, mfData);
          mb = new cv.Mat();
          cv.GaussianBlur(mf, mb, new cv.Size(0, 0), feather / 2, feather / 2, cv.BORDER_REPLICATE);
          dst = a.clone();
          const ch = a.channels(), ad = a.data, bd = b.data, od = dst.data, wd = mb.data32F;
          for (let i = 0; i < N; i++) {
            const w = wd[i];
            for (let c = 0; c < ch; c++) {
              const k = i * ch + c;
              od[k] = Math.floor(ad[k] + (bd[k] - ad[k]) * w + 0.5);
            }
          }
        }
        let out = dst;
        if (view === "outline") {
          if (dst.channels() === 1) {
            rgbOut = new cv.Mat();
            cv.cvtColor(dst, rgbOut, cv.COLOR_GRAY2RGB);
            out = rgbOut;
          }
          const od = out.data;
          for (let j = y0; j < y1; j++) {
            for (let i = x0; i < x1; i++) {
              const k = j * W + i;
              if (!inside[k]) continue;
              const edge = i === 0 || j === 0 || i === W - 1 || j === H - 1 || !inside[k - 1] || !inside[k + 1] || !inside[k - W] || !inside[k + W];
              if (edge) { od[k * 3] = 255; od[k * 3 + 1] = 255; od[k * 3 + 2] = 0; }
            }
          }
        }
        if (ctx) {
          const pct = (raCount * 100) / N;
          let info = `ใช้ภาพ B ${invert ? "นอก" : "ใน"}กรอบ${shape === "rect" ? "สี่เหลี่ยม" : "วงรี"} (${x0}, ${y0})–(${x1 - 1}, ${y1 - 1}) px · ${raCount} px (${fixed1(pct)}%)`;
          if (feather > 0) info += ` · ขอบนุ่ม ${feather} px`;
          if (m.resized) info += RESIZED_B_INFO;
          ctx.info = info;
          ctx.values = scalarValues(ROI_APPLY_FIELDS, [raCount, pct]);
        }
        const result = out;
        if (out === dst) dst = null; else rgbOut = null;
        return result;
      } finally {
        if (dst) dst.delete();
        for (const m of [rgbOut, maskMat, mf, mb, matched, rgbA, rgbB]) if (m) m.delete();
      }
    },
  },

  // ===================================================================
  // IP batch 4B (2026-10-15) — 8 op ใหม่ (สเปก specs/ip_batch4B_2026-10-15.md) · ตัวช่วยอยู่ด้านบน (ก่อน OPS) และท้ายตาราง OPS
  // ===================================================================
  // Wiener Deconvolution — F = G·H*/(|H|² + NSR) ในโดเมนความถี่ · ภาพเทา 1 ช่อง / สี 3 ช่อง (ใช้ OTF ชุดเดียว) · ขอบภาพใช้การสะท้อนค่า
  deconv_wiener: {
    name: "Wiener Deconvolution (แก้ภาพเบลอเมื่อรู้ PSF)",
    needs: "any",
    desc: "แก้ภาพเบลอ 'ย้อนกลับ' เมื่อรู้หน้าตาของการเบลอ: PSF (point spread function) คือหน้าตาของจุดเดียวหลังเบลอ — ถ้ารู้ PSF ก็ 'หารกลับ' ในโดเมนความถี่ได้ · สูตร F = G·H*/(|H|² + NSR) — G คือภาพเบลอ, H คือ PSF, NSR กันการหารด้วยค่าเกือบ 0 ที่ทำให้ noise ระเบิด · ลองแบบนี้: ใช้ Custom Kernel preset Motion (ค่าเริ่มต้น 9 px 0°) ทำภาพเบลอก่อน แล้วต่อกล่องนี้ด้วยค่าเริ่มต้น → ภาพกลับมาคมขึ้น · ถ้าขนาด/มุม PSF ไม่ตรงกับที่เบลอจริง ผลจะแย่ลง · รอยคลื่น (ringing) รอบขอบคมเป็นเรื่องปกติของ deconvolution และขอบภาพใช้การสะท้อนค่าแทน edgetaper ของ MATLAB · ภาพ 8 bit มีรอยปัดเศษเสมอ NSR = 0 จึงแทบไม่เคยใช้ได้ · เทียบ deconvwnr(I, PSF, NSR) ของ MATLAB · ไม่รองรับการเดา PSF เอง (blind deconvolution)",
    params: [
      ...psfParamDefs(),
      { key: "nsr", type: "number", label: "NSR (noise ÷ สัญญาณ)", min: 0, max: 0.5, step: 0.001, default: 0.01, desc: "0 = หารตรง (inverse filter) ขยาย noise รุนแรง · มากขึ้น = เนียนขึ้นแต่คมน้อยลง · MATLAB deconvwnr ใช้ความหมายเดียวกัน" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const { kind, size, angle, psf } = psfFromParams(p);
      const nsr = numIn(p.nsr, 0.01, 0, 0.5);
      let rgb = null, dst = null;
      try {
        rgb = dropAlpha(cv, src);
        dst = wienerDeconvolve(cv, rgb || src, psf, nsr);
        if (ctx) ctx.info = `Wiener deconvolution · PSF ${psfInfo(kind, size, angle, psf.n)} · NSR ${nsr}` + (nsr === 0 ? " · NSR = 0 คือหารตรง (inverse filter) — noise และรอยปัดเศษ 8 bit ถูกขยายมาก" : "") + bigImageNote(src, 0);
        const out = dst; dst = null;
        return out;
      } finally {
        if (dst) dst.delete();
        if (rgb) rgb.delete();
      }
    },
  },

  // Lucy-Richardson — สูตรพื้นฐาน (ไม่มีตัวเร่ง/damping แบบ MATLAB) · filter2D เป็น correlation → ใช้ PSF พลิก (Kf) เป็น convolution แล้ว PSF ตรงเป็น adjoint
  deconv_lucy: {
    name: "Lucy-Richardson Deconvolution (แก้เบลอแบบวนรอบ)",
    modeHint: SLOW_MODE_HINT,
    needs: "any",
    desc: "เดาภาพคมทีละรอบ: เบลอภาพที่เดาด้วย PSF → เทียบกับภาพจริง → แก้ภาพที่เดาตามอัตราส่วน (จริง ÷ ที่เบลอ) แล้ววนซ้ำ · ต่างจาก Wiener ตรงที่ค่าไม่ติดลบเสมอและค่อย ๆ คมขึ้นตามจำนวนรอบ · รอบมากเกิน noise ถูกขยาย — ดูผลทีละ 5–10 รอบ · ช้า: ภาพ 640×480 สี 10 รอบ ราวครึ่งวินาที เหมาะกับภาพนิ่ง · เทียบ deconvlucy(I, PSF, N) ของ MATLAB — MATLAB มีตัวเร่ง (acceleration) และ damping ส่วนแอปเป็นสูตรพื้นฐาน ผลจึงต่างกันเมื่อรอบเท่ากัน",
    params: [
      ...psfParamDefs(),
      { key: "iterations", type: "number", label: "จำนวนรอบ", min: 1, max: LUCY_MAX_ITERS, step: 1, default: 10, desc: "มากขึ้น = คมขึ้นแต่ noise และ ringing เพิ่ม · MATLAB ค่าเริ่มต้น 10" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const { kind, size, angle, psf } = psfFromParams(p);
      const asked = intIn(p.iterations, 10, 1, LUCY_MAX_ITERS);
      let rgb = null, dst = null;
      try {
        rgb = dropAlpha(cv, src);
        const x = rgb || src;
        const iters = lucyBudgetIters(x.cols, x.rows, x.channels(), asked); // B6: ภาพใหญ่ลดรอบ ไม่ย่อภาพ
        dst = lucyDeconvolve(cv, x, psf, iters);
        if (ctx) ctx.info = `Lucy-Richardson ${iters} รอบ · PSF ${psfInfo(kind, size, angle, psf.n)}` + (x.channels() === 3 ? " · ภาพสีใช้เวลาราว 3 เท่าของภาพเทา" : "") + (iters < asked ? ` · ภาพใหญ่ จึงลดเหลือ ${iters} รอบ (ขอ ${asked} รอบ)` : "") + bigImageNote(src, iters);
        const out = dst; dst = null;
        return out;
      } finally {
        if (dst) dst.delete();
        if (rgb) rgb.delete();
      }
    },
  },

  // ECC (findTransformECC ของ OpenCV) — A = ภาพที่ขยับ, B = ภาพอ้างอิง · ย่อภาพก่อนจับคู่แล้วขยาย warp กลับ · ไม่ลู่เข้า/ภาพเรียบ/เล็ก = คืน A เดิม (ไม่ throw)
  ecc_register: {
    name: "Image Registration ECC (จัดภาพให้ตรงกัน)",
    needs: "any",
    desc: "หาการขยับ (เลื่อน / หมุน / ย่อขยาย / มุมมอง) ที่ทำให้ภาพ A ทับภาพ B ได้ตรง โดยเพิ่มค่าความสัมพันธ์ (ECC) ทีละรอบ · ใช้คู่กับ Absolute Difference เพื่อหาสิ่งที่เปลี่ยนไประหว่างสองภาพ (เช่นชิ้นงานกับภาพต้นแบบที่วางเอียงต่างกัน) · ต้องเหลื่อมกันไม่มาก (ราวไม่เกิน 10% ของขนาดภาพ) และมีลวดลาย — ภาพเรียบหรือเลื่อนไกลจะไม่ลู่เข้า · เลือกแบบการขยับให้น้อยเท่าที่พอ แบบที่อิสระมากลู่เข้ายากกว่าและช้ากว่า · ค่า dx, dy = ตำแหน่งใน A ของมุมซ้ายบนของ B · เทียบ imregister ของ MATLAB (MATLAB ใช้ optimizer อีกแบบ) แอปใช้ findTransformECC ของ OpenCV — สคริปต์ Python (cv2 5.0) ได้ค่าต่างจากแอปได้ราว 0.05 px",
    params: [
      { key: "motion", type: "select", label: "แบบการขยับ", options: [
        { value: "translation", label: "เลื่อนอย่างเดียว" },
        { value: "euclidean", label: "เลื่อน + หมุน" },
        { value: "affine", label: "เลื่อน + หมุน + ย่อขยาย + เฉือน" },
        { value: "homography", label: "มุมมองเปลี่ยน (perspective)" },
      ], default: "euclidean" },
      { key: "iterations", type: "number", label: "จำนวนรอบสูงสุด", min: 10, max: 300, step: 10, default: 50 },
      { key: "maxSide", type: "number", label: "ย่อภาพก่อนจับคู่ (ด้านยาว px)", min: 64, max: 1024, step: 32, default: ECC_DEFAULT_SIDE, desc: "เล็ก = เร็ว · ใหญ่ = ละเอียดขึ้น (640 ≈ 180 ms)" },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "aligned", label: "A ที่จัดแล้ว" },
        { value: "overlay", label: "ซ้อนสี (ม่วง = A, เขียว = B, เทา = ตรงกัน)" },
        { value: "diff", label: "ความต่าง |A ที่จัดแล้ว − B|" },
      ], default: "aligned" },
    ],
    apply(cv, src, p, ctx) {
      const pass = passthroughNoSrc2(src, ctx);
      if (pass) { if (ctx) ctx.values = scalarValues(ECC_FIELDS, [0]); return pass; }
      if (isEmptyMat(src)) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(ECC_FIELDS, [0]); return src.clone(); }
      const motion = ownKey(ECC_MOTIONS, p.motion, "euclidean");
      const iters = intIn(p.iterations, 50, 10, 300), maxSide = intIn(p.maxSide, ECC_DEFAULT_SIDE, 64, 1024);
      const view = ownKey(ECC_VIEWS, p.view, "aligned");
      const W = src.cols, H = src.rows, homog = motion === "homography";
      let matched = null, gA0 = null, gB0 = null, rA = null, rB = null, mean = null, sd = null, warp = null, mask = null, out = null, gO0 = null, extra = null, dst = null;
      // ล้มเหลว (ภาพเล็ก/เรียบ/ไม่ลู่เข้า) → คืน A เดิม + ecc_ok = 0 · ห้ามแตะค่าที่ catch ได้ (ถ้าเป็นเลข = C++ exception ที่รั่วอยู่แล้ว แก้ในแอปไม่ได้)
      const fail = (msg) => { ctx.info = msg; ctx.values = scalarValues(ECC_FIELDS, [0]); return src.clone(); };
      try {
        const m = matchSize(cv, src, ctx.src2);
        matched = m.resized ? m.mat : null;
        const Bm = m.mat;
        gA0 = grayIfNeeded(cv, src);
        gB0 = grayIfNeeded(cv, Bm);
        const gA = gA0 || src, gB = gB0 || Bm;
        const sc = analysisScale(W, H, maxSide), s = sc.scale;
        let sA = gA, sB = gB;
        if (s < 1) {
          rA = new cv.Mat(); cv.resize(gA, rA, new cv.Size(sc.w, sc.h), 0, 0, cv.INTER_AREA); sA = rA;
          rB = new cv.Mat(); cv.resize(gB, rB, new cv.Size(sc.w, sc.h), 0, 0, cv.INTER_AREA); sB = rB;
        }
        if (Math.min(sA.cols, sA.rows) < 16) return fail("ภาพเล็กเกินไปสำหรับจัดภาพ (ต้องกว้าง/สูงอย่างน้อย 16 px หลังย่อ)");
        mean = new cv.Mat(); sd = new cv.Mat();
        cv.meanStdDev(sA, mean, sd);
        const stdA = sd.doubleAt(0, 0);
        cv.meanStdDev(sB, mean, sd);
        const stdB = sd.doubleAt(0, 0);
        if (stdA < 1 || stdB < 1) return fail("ภาพเรียบเกินไป ไม่มีลวดลายให้จับคู่");
        warp = cv.matFromArray(homog ? 3 : 2, 3, cv.CV_32FC1, homog ? [1, 0, 0, 0, 1, 0, 0, 0, 1] : [1, 0, 0, 0, 1, 0]);
        mask = new cv.Mat();
        const crit = new cv.TermCriteria(cv.TermCriteria_COUNT + cv.TermCriteria_EPS, iters, 1e-5);
        let rho = 0, ok = false;
        // template = B, input = A — warp แปลงพิกัดของ B ไปหา A · ต้องส่ง 7 อาร์กิวเมนต์
        try { rho = cv.findTransformECC(sB, sA, warp, cv[ECC_MOTIONS[motion].code], crit, mask, 5); ok = true; } catch (e) { ok = false; }
        if (!ok) return fail("จัดภาพไม่สำเร็จ (ไม่ลู่เข้า) — ลองเพิ่มจำนวนรอบ ใช้แบบเลื่อนอย่างเดียว หรือให้สองภาพเหลื่อมกันน้อยลง · ถ้าล้มเหลวซ้ำ ๆ อย่าปล่อยให้รันต่อเนื่องบนกล้องสด (หน่วยความจำจะรั่วทีละน้อยทุกเฟรม) — ใช้กับภาพนิ่งแทน");
        // ขยาย warp กลับขนาดจริง (ค่าเลื่อนเป็น px ของภาพย่อ) · ใช้ s ตัวเดียวทั้ง x/y ทั้งที่ภาพย่อปัดขนาดแยกแกน (คลาดน้อยกว่า 0.03 px) — ตามสเปก Python ทำเหมือนกัน
        const w = warp.data32F;
        w[2] = w[2] / s; w[5] = w[5] / s;
        if (homog) { w[6] = w[6] * s; w[7] = w[7] * s; }
        const dx = w[2], dy = w[5], w00 = w[0], w01 = w[1], w10 = w[3], w11 = w[4];
        const angle = Math.atan2(w10, w00) * 180 / Math.PI, scale = Math.sqrt(Math.abs(w00 * w11 - w01 * w10));
        out = new cv.Mat();
        const flags = cv.INTER_LINEAR + cv.WARP_INVERSE_MAP;
        if (homog) cv.warpPerspective(src, out, warp, new cv.Size(W, H), flags, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
        else cv.warpAffine(src, out, warp, new cv.Size(W, H), flags, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
        if (view === "aligned") { dst = out; out = null; }
        else {
          gO0 = grayIfNeeded(cv, out);
          const gO = gO0 || out;
          if (view === "diff") {
            extra = new cv.Mat();
            cv.absdiff(gO, gB, extra);
            dst = extra; extra = null;
          } else {
            const N = W * H, go = gO.data, gb = gB.data, res = new Uint8Array(N * 3);
            for (let i = 0; i < N; i++) { res[3 * i] = go[i]; res[3 * i + 1] = gb[i]; res[3 * i + 2] = go[i]; }
            dst = cv.matFromArray(H, W, cv.CV_8UC3, res);
          }
        }
        ctx.values = scalarValues(ECC_FIELDS, [1, rho, dx, dy, angle, scale]);
        ctx.info = `จัดภาพแบบ${ECC_MOTIONS[motion].th} · A เลื่อน (${fixed2(dx)}, ${fixed2(dy)}) px · หมุน ${fixed2(angle)}° · ขยาย ×${fixed3(scale)} · ρ = ${fixed3(rho)}`
          + (s < 1 ? ` · จับคู่บนภาพย่อ ${sc.w}×${sc.h}` : "") + (m.resized ? RESIZED_B_INFO : "");
        const result = dst; dst = null;
        return result;
      } finally {
        for (const mm of [dst, extra, gO0, out, warp, mask, sd, mean, rB, rA, gB0, gA0, matched]) if (mm) mm.delete();
      }
    },
  },

  // Gabor — kernel จาก gaborKernel (สูตร getGaborKernel) · ส่วนจริงลบค่าเฉลี่ยของ kernel (พื้นเรียบ → 0) · ปรับให้ค่าสูงสุดเป็น 255 ทุกภาพ
  gabor_filter: {
    name: "Gabor Filter (หาลายตามความถี่และทิศ)",
    needs: "gray",
    desc: "Gabor = คลื่น cos คูณระฆังเกาส์ — ตอบสนองแรงตรงที่มีลายซ้ำระยะ λ ในทิศ θ · ใช้ทำ texture: ต่อหลายกล่องที่ θ ต่างกัน (เช่น 0/45/90/135) แล้วเทียบกัน · magnitude ไม่สนว่าลายเลื่อนไปครึ่งช่วงคลื่น ส่วน real เห็นลายบวก/ลบ · ค่าเริ่มต้น bandwidth/aspect ตาม imgaborfilt ของ MATLAB แต่แอปลบค่าเฉลี่ยของ kernel ส่วนจริงออก ผลจึงไม่เท่า MATLAB ทุกค่า · OpenCV.js ชุดนี้ไม่มี getGaborKernel แอปจึงคำนวณสูตรเดียวกันเอง · ภาพปรับความสว่างอัตโนมัติ — สองภาพเทียบกันได้จากตัวเลขในแผงเท่านั้น",
    params: [
      { key: "wavelength", type: "number", label: "ความยาวคลื่น λ (px)", min: 2, max: 64, step: 1, default: 8, desc: "ระยะห่างของลายที่ต้องการหา" },
      { key: "orientation", type: "number", label: "มุม θ (องศา)", min: 0, max: 179, step: 1, default: 0, desc: "0° = ตอบสนองลายเส้นแนวตั้ง (คลื่นวิ่งตามแนวนอน) · 90° = ลายแนวนอน" },
      { key: "bandwidth", type: "number", label: "แบนด์วิดท์ (octave)", min: 0.5, max: 3, step: 0.1, default: 1, hint: "MATLAB ค่าเริ่มต้น 1" },
      { key: "aspect", type: "number", label: "อัตราส่วนรูปไข่ γ", min: 0.2, max: 1, step: 0.05, default: 0.5, hint: "MATLAB ค่าเริ่มต้น 0.5" },
      { key: "output", type: "select", label: "แสดงผลเป็น", options: [
        { value: "magnitude", label: "ขนาดผลตอบสนอง (ไม่สนเฟส)" },
        { value: "real", label: "ส่วนจริง (เห็นลายบวก/ลบ)" },
      ], default: "magnitude" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(GABOR_FIELDS, []); } return src.clone(); }
      const lambd = intIn(p.wavelength, 8, 2, 64), theta = intIn(p.orientation, 0, 0, 179);
      const bw = numIn(p.bandwidth, 1, 0.5, 3), gamma = numIn(p.aspect, 0.5, 0.2, 1);
      const output = ownKey(GABOR_OUTPUTS, p.output, "magnitude");
      const sigma = gaborSigma(lambd, bw), half = gaborHalf(sigma, gamma), k = 2 * half + 1;
      const even = gaborKernel(k, sigma, theta, lambd, gamma, 0);
      let sum = 0;
      for (let i = 0; i < even.length; i++) sum += even[i];
      const mu = sum / (k * k);
      for (let i = 0; i < even.length; i++) even[i] = Math.fround(even[i] - mu);
      const odd = gaborKernel(k, sigma, theta, lambd, gamma, Math.PI / 2);
      let g = null, F = null, ke = null, ko = null, re = null, im = null, mg = null, dst = null;
      try {
        g = grayIfNeeded(cv, src);
        const s = g || src;
        const W = s.cols, H = s.rows, N = W * H;
        F = new cv.Mat();
        s.convertTo(F, cv.CV_32F);
        ke = cv.matFromArray(k, k, cv.CV_32FC1, even);
        ko = cv.matFromArray(k, k, cv.CV_32FC1, odd);
        re = new cv.Mat(); im = new cv.Mat();
        const anchor = new cv.Point(-1, -1);
        cv.filter2D(F, re, cv.CV_32F, ke, anchor, 0, cv.BORDER_REFLECT_101);
        cv.filter2D(F, im, cv.CV_32F, ko, anchor, 0, cv.BORDER_REFLECT_101);
        let resp = re;
        if (output === "magnitude") { mg = new cv.Mat(); cv.magnitude(re, im, mg); resp = mg; }
        const rd = resp.data32F, o = new Uint8Array(N);
        let mx = 0, tot = 0;
        for (let i = 0; i < N; i++) { const a = Math.abs(rd[i]); tot += a; if (a > mx) mx = a; }
        // ภาพเรียบสนิท (ทุกพิกเซลเท่ากัน) → ไม่มีลาย: kernel ส่วนจริงลบค่าเฉลี่ยแล้วแต่ float32 ยังเหลือเศษ (วัดได้ถึง ~0.044 ที่ภาพขาว 255 กับ kernel ใหญ่)
        // ซึ่งการปรับให้ค่าสูงสุดเป็น 255 จะขยายเป็นภาพขาวทั้งภาพ จึงตรวจว่าภาพเรียบจากข้อมูลต้นทางโดยตรง ไม่ใช้เกณฑ์ขนาดของเศษ
        let none = !(mx > 0);
        if (!none) { const sd0 = s.data; let lo = 255, hi = 0; for (let i = 0; i < N; i++) { const v = sd0[i]; if (v < lo) lo = v; if (v > hi) hi = v; } none = lo === hi; }
        if (none) { mx = 0; tot = 0; }
        if (output === "magnitude") for (let i = 0; i < N; i++) o[i] = none ? 0 : Math.floor(rd[i] * 255 / mx + 0.5);
        else for (let i = 0; i < N; i++) o[i] = none ? 128 : Math.floor(128 + rd[i] * 127 / mx + 0.5);
        dst = cv.matFromArray(H, W, cv.CV_8UC1, o);
        if (ctx) {
          ctx.values = scalarValues(GABOR_FIELDS, [tot / N, mx]);
          ctx.info = `Gabor λ=${lambd} px θ=${theta}° (σ=${fixed2(sigma)} px, kernel ${k}×${k}) · `
            + (none ? "ไม่พบลายที่ความถี่และทิศนี้ (ภาพเรียบ ทุกพิกเซลเท่ากัน)"
              : output === "magnitude"
                ? "สว่าง = มีลายที่ความถี่และทิศนี้ · ปรับให้ค่าสูงสุดเป็น 255 ทุกภาพ (ค่าจริงดูที่แผงตัวเลข)"
                : "เทากลาง 128 = ไม่มีลาย · สว่างกว่า = ผลบวก มืดกว่า = ผลลบ · ปรับให้ค่าสูงสุดของ |ผล| อยู่ที่ขอบช่วง (ค่าจริงดูที่แผงตัวเลข)");
        }
        const out = dst; dst = null;
        return out;
      } finally {
        for (const mm of [dst, mg, im, re, ko, ke, F, g]) if (mm) mm.delete();
      }
    },
  },

  // GLCM — นับคู่พิกเซลทั้งภาพ (ไม่ต้องใช้ cv) · 256×256 = ตาราง GLCM ขยายเป็นบล็อก · ค่าเริ่มต้นตรงกับ MATLAB graycomatrix (8 ระดับ, [0 1], ไม่ symmetric)
  glcm: {
    name: "GLCM Texture (ตารางคู่ระดับเทา)",
    needs: "gray",
    desc: "นับว่าคู่พิกเซลที่ห่างกัน d ในทิศที่เลือก มีระดับเทา (i, j) บ่อยแค่ไหน แล้วสรุปเป็น 4 ค่า: Contrast สูง = เพื่อนบ้านต่างกันมาก (ลายหยาบ/ขอบเยอะ) · Energy สูง = ลายซ้ำ ๆ เป็นระเบียบ · Homogeneity สูง = เรียบ · Correlation ใกล้ 1 = เพื่อนบ้านไปทางเดียวกัน · เปรียบเทียบได้เฉพาะภาพที่ตั้งค่าเดียวกัน (ระดับ/ระยะ/ทิศ) · ใช้ทั้งภาพ (MATLAB graycomatrix ก็เป็นทั้งภาพ) — ถ้าสนใจบริเวณเดียวให้ Crop ก่อน · เทียบ graycomatrix + graycoprops ของ MATLAB ค่าเริ่มต้นตรงกับ MATLAB (8 ระดับ, [0 1], ไม่ symmetric)",
    params: [
      { key: "levels", type: "select", label: "จำนวนระดับเทา", options: [
        { value: "2", label: "2" }, { value: "4", label: "4" }, { value: "8", label: "8" }, { value: "16", label: "16" }, { value: "32", label: "32" }, { value: "64", label: "64" },
      ], default: "8", hint: "MATLAB ค่าเริ่มต้น 8" },
      { key: "distance", type: "number", label: "ระยะคู่พิกเซล (px)", min: 1, max: 10, step: 1, default: 1 },
      { key: "angle", type: "select", label: "ทิศของคู่", options: [
        { value: "0", label: "0° (ขวา)" }, { value: "45", label: "45° (ขวาบน)" }, { value: "90", label: "90° (บน)" }, { value: "135", label: "135° (ซ้ายบน)" }, { value: "avg", label: "เฉลี่ย 4 ทิศ" },
      ], default: "0", hint: "MATLAB ค่าเริ่มต้น Offset [0 1] = 0°" },
      { key: "symmetric", type: "boolean", label: "นับทั้งสองทาง (symmetric)", default: false },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "matrix", label: "ตาราง GLCM (ขยายเป็นภาพ 256×256)" },
        { value: "quantized", label: "ภาพหลังลดระดับเทา" },
      ], default: "matrix" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(GLCM_FIELDS, []); } return src.clone(); }
      const L = Number(ownKey(GLCM_LEVELS, p.levels, "8"));
      const d = intIn(p.distance, 1, 1, 10);
      const angle = ownKey(GLCM_ANGLES, p.angle, "0");
      const sym = boolOr(p.symmetric, false);
      const view = ownKey(GLCM_VIEWS, p.view, "matrix");
      let g = null, dst = null;
      try {
        g = grayIfNeeded(cv, src);
        const s = g || src;
        const W = s.cols, H = s.rows, N = W * H;
        const sd = s.data, q = new Uint8Array(N);
        for (let i = 0; i < N; i++) q[i] = Math.min(L - 1, Math.floor((sd[i] * L) / 255));
        const dirs = angle === "avg" ? [0, 45, 90, 135] : [Number(angle)];
        const OFF = { 0: [d, 0], 45: [d, -d], 90: [0, -d], 135: [-d, -d] };
        const total = new Float64Array(L * L);
        const props = [];
        let pairs = 0;
        for (const a of dirs) {
          let C = glcmCounts(q, W, H, L, OFF[a][0], OFF[a][1]);
          if (sym) { const T = new Float64Array(L * L); for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) T[i * L + j] = C[i * L + j] + C[j * L + i]; C = T; }
          const pr = glcmProps(C, L);
          props.push(pr);
          if (pr) pairs += pr.pairs;
          for (let i = 0; i < L * L; i++) total[i] += C[i];
        }
        const avgOf = (key) => {
          let sum = 0;
          for (const pr of props) { if (!pr || pr[key] === null) return null; sum += pr[key]; }
          return sum / props.length;
        };
        const vals = pairs === 0 ? [null, null, null, null, 0]
          : [avgOf("contrast"), avgOf("correlation"), avgOf("energy"), avgOf("homogeneity"), pairs];
        if (view === "matrix") {
          const o = new Uint8Array(256 * 256), bs = 256 / L;
          let cmax = 0;
          for (let i = 0; i < L * L; i++) if (total[i] > cmax) cmax = total[i];
          for (let y = 0; y < 256; y++) {
            const i = Math.floor(y / bs);
            for (let x = 0; x < 256; x++) {
              const c = total[i * L + Math.floor(x / bs)];
              o[y * 256 + x] = c === 0 ? 0 : Math.floor(255 * Math.log1p(c) / Math.log1p(cmax) + 0.5);
            }
          }
          dst = cv.matFromArray(256, 256, cv.CV_8UC1, o);
        } else {
          const o = new Uint8Array(N);
          for (let i = 0; i < N; i++) o[i] = Math.floor(q[i] * 255 / (L - 1) + 0.5);
          dst = cv.matFromArray(H, W, cv.CV_8UC1, o);
        }
        if (ctx) {
          ctx.values = scalarValues(GLCM_FIELDS, vals);
          const angTxt = angle === "avg" ? "เฉลี่ย 4 ทิศ" : `${angle}°`;
          if (pairs === 0) ctx.info = `ภาพเล็กกว่าระยะ ${d} px — ไม่มีคู่พิกเซลให้นับ`;
          else {
            const t3 = (v) => (v === null ? "—" : fixed3(v));
            ctx.info = `GLCM ${L} ระดับ · ระยะ ${d} px ทิศ ${angTxt} · contrast ${vals[0] === null ? "—" : fixed2(vals[0])} · energy ${t3(vals[2])} · homogeneity ${t3(vals[3])} · correlation ${t3(vals[1])}`
              + (view === "matrix" ? " · แถว = ระดับของพิกเซล คอลัมน์ = ระดับของเพื่อนบ้าน · เส้นทแยงสว่าง = ภาพเรียบ" : "")
              + (vals[1] === null ? (props.some((pr) => !pr) ? " · บางทิศไม่มีคู่พิกเซล (ภาพเล็กกว่าระยะ) จึงหาค่าเฉลี่ย 4 ทิศไม่ได้" : " · ภาพมีระดับเดียว correlation หาไม่ได้") : "");
          }
        }
        const out = dst; dst = null;
        return out;
      } finally {
        if (dst) dst.delete();
        if (g) g.delete();
      }
    },
  },

  // NLM — เขียนเองใน nlmCore (ไม่มี fastNlMeansDenoising ใน build นี้) · ภาพด้านยาว > NLM_MAX_SIDE ย่อก่อนแล้วขยายผลกลับ
  nlm_denoise: {
    name: "Non-local Means (ลด noise แบบเทียบแผ่น)",
    modeHint: SLOW_MODE_HINT,
    needs: "any",
    desc: "แทนที่จะเฉลี่ยเฉพาะเพื่อนบ้านติดกัน NLM เฉลี่ยพิกเซลที่ 'ลายรอบตัวคล้ายกัน' จากทั้งหน้าต่างค้นหา — ลายผ้า/ตัวอักษรที่ซ้ำ ๆ จึงไม่เบลอ · h คุมว่าคล้ายแค่ไหนถึงนับ · ช้ามาก: 640×480 ราว 1 วินาที (ค้นหา 11) ใช้กับภาพนิ่ง หรือย่อภาพก่อน · เทียบ imnlmfilt ของ MATLAB (MATLAB เดาความแรงจาก noise เองและค้นหา 21×21) · fastNlMeansDenoising ของ OpenCV ไม่มีใน OpenCV.js ชุดนี้ และสูตรต่างจากของแอป — สคริปต์ Python จึงใช้สูตรของแอป",
    params: [
      { key: "h", type: "number", label: "ความแรง h", min: 1, max: 50, step: 1, default: 10, desc: "มาก = เรียบขึ้นแต่รายละเอียดหาย · ค่าใกล้ σ ของ noise มักได้ผลดี" },
      { key: "patch", type: "select", label: "ขนาดแผ่นที่ใช้เทียบ (px)", options: [{ value: "3", label: "3" }, { value: "5", label: "5" }, { value: "7", label: "7" }], default: "5", hint: "MATLAB ค่าเริ่มต้น 5" },
      { key: "search", type: "select", label: "ขนาดหน้าต่างค้นหา (px)", options: [{ value: "7", label: "7" }, { value: "11", label: "11" }, { value: "15", label: "15" }], default: "11", hint: "MATLAB ใช้ 21 — แอปใช้ 11 เพราะ 21 ช้ากว่าราว 10 เท่า" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const h = intIn(p.h, 10, 1, 50);
      const patch = Number(ownKey(NLM_PATCHES, p.patch, "5")), search = Number(ownKey(NLM_SEARCHES, p.search, "11"));
      let rgb = null, small = null, proc = null, dst = null;
      try {
        rgb = dropAlpha(cv, src);
        const x = rgb || src;
        const W = x.cols, H = x.rows, ch = x.channels();
        const sc = analysisScale(W, H, NLM_MAX_SIDE);
        let work = x;
        if (sc.scale < 1) { small = new cv.Mat(); cv.resize(x, small, new cv.Size(sc.w, sc.h), 0, 0, cv.INTER_AREA); work = small; }
        const res = nlmCore(work.data, work.cols, work.rows, ch, h, (patch - 1) / 2, (search - 1) / 2);
        proc = cv.matFromArray(work.rows, work.cols, work.type(), res);
        if (sc.scale < 1) { dst = new cv.Mat(); cv.resize(proc, dst, new cv.Size(W, H), 0, 0, cv.INTER_LINEAR); } else { dst = proc; proc = null; }
        if (ctx) {
          ctx.info = `Non-local Means h=${h} · แผ่น ${patch}×${patch} · ค้นหา ${search}×${search}`
            + (sc.scale < 1 ? ` · ประมวลผลบนภาพย่อ ${sc.w}×${sc.h} แล้วขยายกลับ (ภาพใหญ่ช้ามาก)` : "") + " · ช้า: เหมาะกับภาพนิ่ง";
        }
        const out = dst; dst = null;
        return out;
      } finally {
        for (const mm of [dst, proc, small, rgb]) if (mm) mm.delete();
      }
    },
  },

  // Dehaze (He et al. 2009, dark channel prior) — guided filter จาก 4A ปรับแผนที่ t · ทุกขั้นกำหนดลำดับ/ความละเอียด (double/float32) ตายตัวเพื่อให้ Python ที่ export ได้ผลเท่ากัน
  dehaze: {
    name: "Dehaze (ลดหมอก)",
    needs: "any",
    desc: "สมมติฐาน dark channel: ภาพกลางแจ้งที่ไม่มีหมอก ในหน้าต่างเล็ก ๆ จะมีช่องสีใดช่องหนึ่งที่มืดเกือบ 0 — หมอกทำให้ค่านี้สูงขึ้น จึงใช้ประมาณความหนาของหมอก · A = สีของหมอก (จากจุดที่หมอกหนา) · t = สัดส่วนแสงจากวัตถุที่ทะลุหมอกมาถึงกล้อง · ท้องฟ้าและวัตถุสีขาวขัดกับสมมติฐาน → อาจสีเพี้ยน/คล้ำเกินจริง · guided filter ทำให้แผนที่หมอกตามขอบวัตถุ ไม่เป็นบล็อก · เทียบ imreducehaze ของ MATLAB (หลักการเดียวกัน รายละเอียดต่าง) · พบไม่บ่อยในงาน Machine Vision ในโรงงาน แต่เห็นผลชัดกับภาพกลางแจ้ง",
    params: [
      { key: "amount", type: "number", label: "ปริมาณการลดหมอก ω", min: 0.1, max: 1, step: 0.05, default: 0.95, hint: "ค่าเริ่มต้นของแอป 0.95 (ตามบทความ dark channel ของ He et al.) · imreducehaze ของ MATLAB ใช้ Amount เริ่มต้น 1 และวิธีต่างกัน ผลจึงไม่เท่ากัน" },
      { key: "patch", type: "number", label: "ขนาดหน้าต่าง dark channel (px)", min: 3, max: 31, step: 2, default: 15 },
      { key: "refine", type: "boolean", label: "เกลี่ยแผนที่หมอกตามขอบวัตถุ (guided filter)", default: true },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "result", label: "ภาพที่ลดหมอกแล้ว" },
        { value: "transmission", label: "แผนที่ความโปร่งใส t (ขาว = ใส)" },
        { value: "dark", label: "dark channel (ขาว = หมอกหนา)" },
      ], default: "result" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(DEHAZE_FIELDS, []); } return src.clone(); }
      const omega = numIn(p.amount, 0.95, 0.1, 1), ps = ensureOdd(intIn(p.patch, 15, 3, 31));
      const refine = boolOr(p.refine, true), view = ownKey(DEHAZE_VIEWS, p.view, "result");
      let rgb = null, ker = null, d8m = null, dark = null, mnM = null, dnM = null, gm = null, G = null, Pm = null, qM = null, dst = null;
      try {
        rgb = dropAlpha(cv, src);
        const x = rgb || src;
        const W = x.cols, H = x.rows, ch = x.channels(), N = W * H;
        ker = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(ps, ps));
        // 1) dark channel = min ข้ามช่อง แล้ว erode (ทุกครั้งที่อ่าน .data อ่านใหม่ก่อนใช้ — หลังจองแมตใหม่ heap อาจขยายจน view เก่าใช้ไม่ได้)
        const d8 = new Uint8Array(N);
        { const xd = x.data; for (let i = 0; i < N; i++) { let m = xd[i * ch]; for (let c = 1; c < ch; c++) if (xd[i * ch + c] < m) m = xd[i * ch + c]; d8[i] = m; } }
        d8m = cv.matFromArray(H, W, cv.CV_8UC1, d8);
        dark = new cv.Mat();
        cv.erode(d8m, dark, ker);
        // 2) แสงหมอก A = ค่าเฉลี่ยของจุดที่ dark สูงสุด 0.1% (อย่างน้อย 1 จุด)
        const A = [0, 0, 0];
        {
          const dd = dark.data, xd = x.data, hist = new Int32Array(256);
          for (let i = 0; i < N; i++) hist[dd[i]]++;
          const kTop = Math.max(1, Math.floor(N / 1000));
          let T = 0, cum = 0;
          for (let v = 255; v >= 0; v--) { cum += hist[v]; if (cum >= kTop) { T = v; break; } }
          const sums = [0, 0, 0];
          let cnt = 0;
          for (let i = 0; i < N; i++) if (dd[i] >= T) { cnt++; for (let c = 0; c < ch; c++) sums[c] += xd[i * ch + c]; }
          for (let c = 0; c < ch; c++) A[c] = Math.max(sums[c] / cnt, 1);
        }
        // 3) แผนที่หมอกหยาบ: 1 − ω · erode(min_c(x_c / A_c)) (เก็บเป็น float32 ตามสเปก)
        const mn = new Float32Array(N);
        { const xd = x.data; for (let i = 0; i < N; i++) { let m = xd[i * ch] / A[0]; for (let c = 1; c < ch; c++) { const v = xd[i * ch + c] / A[c]; if (v < m) m = v; } mn[i] = m; } }
        mnM = cv.matFromArray(H, W, cv.CV_32FC1, mn);
        dnM = new cv.Mat();
        cv.erode(mnM, dnM, ker);
        const t = new Float32Array(N);
        { const dn = dnM.data32F; for (let i = 0; i < N; i++) t[i] = 1 - omega * dn[i]; }
        // 4) เกลี่ยตามขอบวัตถุด้วย guided filter (guide = ภาพเทา)
        if (refine) {
          if (ch === 3) { gm = new cv.Mat(); cv.cvtColor(x, gm, cv.COLOR_RGB2GRAY); G = new cv.Mat(); gm.convertTo(G, cv.CV_32F); }
          else { G = new cv.Mat(); x.convertTo(G, cv.CV_32F); }
          const P = new Float32Array(N);
          for (let i = 0; i < N; i++) P[i] = t[i] * 255;
          Pm = cv.matFromArray(H, W, cv.CV_32FC1, P);
          qM = guidedFilterF32(cv, G, Pm, Math.max(4, 2 * ps), 65.025);
          const qd = qM.data32F;
          for (let i = 0; i < N; i++) t[i] = qd[i] / 255;
        }
        // 5) กัน t ใกล้ 0 (double) แล้วคืนค่าภาพ J = (I − A)/t + A
        const tt = new Float64Array(N);
        let tsum = 0;
        for (let i = 0; i < N; i++) { tt[i] = Math.max(t[i], 0.1); tsum += tt[i]; }
        const tMean = tsum / N;
        if (view === "result") {
          dst = new cv.Mat(H, W, x.type());
          const od = dst.data, xd = x.data;
          for (let i = 0; i < N; i++) for (let c = 0; c < ch; c++) {
            const v = Math.floor((xd[i * ch + c] - A[c]) / tt[i] + A[c] + 0.5);
            od[i * ch + c] = v < 0 ? 0 : v > 255 ? 255 : v;
          }
        } else if (view === "transmission") {
          const o = new Uint8Array(N);
          for (let i = 0; i < N; i++) o[i] = Math.floor(tt[i] * 255 + 0.5);
          dst = cv.matFromArray(H, W, cv.CV_8UC1, o);
        } else { dst = dark; dark = null; }
        if (ctx) {
          const a = ch === 1 ? [A[0], A[0], A[0]] : A;
          ctx.values = scalarValues(DEHAZE_FIELDS, [a[0], a[1], a[2], tMean]);
          ctx.info = `Dehaze ω=${omega} หน้าต่าง ${ps}×${ps} · แสงหมอก A (R/G/B) = ${fixed1(a[0])}/${fixed1(a[1])}/${fixed1(a[2])} · ความโปร่งใสเฉลี่ย ${fixed2(tMean)}`
            + (refine ? "" : " · ไม่เกลี่ยแผนที่หมอก (เห็นเป็นบล็อกรอบขอบวัตถุ)");
        }
        const out = dst; dst = null;
        return out;
      } finally {
        for (const mm of [dst, qM, Pm, G, gm, dnM, mnM, dark, d8m, ker, rgb]) if (mm) mm.delete();
      }
    },
  },

  // DCT 8×8 / JPEG-style (เฉพาะภาพเทา, ไม่เข้ารหัสไฟล์) — ไม่มี cv.dct ใน build นี้ จึงคูณ matrix เอง (dct8Forward/dct8Inverse)
  dct_jpeg: {
    name: "DCT / JPEG-style (บีบอัดแบบ JPEG)",
    needs: "gray",
    desc: "JPEG แบ่งภาพเป็นบล็อก 8×8 แล้วใช้ DCT แปลงพิกเซลเป็น 'ความถี่' — พลังงานส่วนใหญ่รวมอยู่มุมซ้ายบน (ความถี่ต่ำ) · quantization หารด้วยตาราง Q แล้วปัดเศษ ความถี่สูงส่วนใหญ่กลายเป็น 0 → ไฟล์เล็กลง (ขั้นเข้ารหัสไฟล์จริง เช่น Huffman ไม่ได้ทำในกล่องนี้ จึงไม่มีขนาดไฟล์ให้ดู) · quality ต่ำ → เห็นขอบบล็อก 8×8 (blocking) และคลื่นรอบขอบ · แบบ 'เก็บความถี่ต่ำ' แสดงว่าภาพยังจำได้แม้เก็บ 16/64 ค่า · JPEG สีจริงแยก Y/Cb/Cr และลดความละเอียดสีด้วย — กล่องนี้ทำเฉพาะภาพเทา · เทียบ dct2 + blockproc ของ MATLAB · OpenCV.js ชุดนี้ไม่มี cv.dct แอปจึงคูณ matrix เอง · เป็นเรื่องหลักการบีบอัด ไม่ใช่งานตรวจสอบโดยตรง",
    params: [
      { key: "method", type: "select", label: "วิธี", options: [
        { value: "jpeg", label: "บีบอัดแบบ JPEG (quality)" },
        { value: "zonal", label: "เก็บเฉพาะความถี่ต่ำ K×K ต่อบล็อก" },
      ], default: "jpeg" },
      { key: "quality", visibleWhen: [{key:"method",in:["jpeg"]}], type: "number", label: "คุณภาพ — ใช้กับแบบ JPEG เท่านั้น", min: 1, max: 100, step: 1, default: 50, desc: "100 = เกือบเหมือนเดิม · 10 = เห็นเป็นบล็อก 8×8" },
      { key: "keep", visibleWhen: [{key:"method",in:["zonal"]}], type: "number", label: "จำนวนความถี่ที่เก็บต่อด้าน K — ใช้กับแบบเก็บความถี่ต่ำเท่านั้น", min: 1, max: 8, step: 1, default: 4 },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [
        { value: "result", label: "ภาพหลังบีบอัด" },
        { value: "coeffs", label: "สัมประสิทธิ์ DCT ต่อบล็อก (log)" },
        { value: "error", label: "ความต่างจากภาพเดิม ×8" },
      ], default: "result" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(DCT_FIELDS, []); } return src.clone(); }
      const method = ownKey(DCT_METHODS, p.method, "jpeg"), quality = intIn(p.quality, 50, 1, 100), K = intIn(p.keep, 4, 1, 8);
      const view = ownKey(DCT_VIEWS, p.view, "result");
      let g = null, pad = null, dst = null;
      try {
        g = grayIfNeeded(cv, src);
        const s = g || src;
        const W = s.cols, H = s.rows;
        const Wp = Math.ceil(W / 8) * 8, Hp = Math.ceil(H / 8) * 8;
        pad = new cv.Mat();
        cv.copyMakeBorder(s, pad, 0, Hp - H, 0, Wp - W, cv.BORDER_REPLICATE);
        const pd = pad.data, rec = new Uint8Array(Wp * Hp), coef = view === "coeffs" ? new Uint8Array(Wp * Hp) : null;
        const Q = jpegQTable(quality), blk = new Float64Array(64), F = new Float64Array(64), Fq = new Float64Array(64), R = new Float64Array(64);
        const L1024 = Math.log1p(1024);
        let zeros = 0;
        const bw = Wp / 8, bh = Hp / 8;
        for (let by = 0; by < bh; by++) {
          for (let bx = 0; bx < bw; bx++) {
            for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) blk[y * 8 + x] = pd[(by * 8 + y) * Wp + bx * 8 + x] - 128;
            dct8Forward(blk, F);
            for (let u = 0; u < 8; u++) {
              for (let v = 0; v < 8; v++) {
                const i = u * 8 + v;
                if (method === "jpeg") {
                  const z = F[i] / Q[i], qf = Math.sign(z) * Math.floor(Math.abs(z) + 0.5);
                  Fq[i] = qf * Q[i];
                  if (qf === 0) zeros++;
                } else if (u < K && v < K) Fq[i] = F[i];
                else { Fq[i] = 0; zeros++; }
              }
            }
            dct8Inverse(Fq, R);
            for (let y = 0; y < 8; y++) {
              for (let x = 0; x < 8; x++) {
                const o = (by * 8 + y) * Wp + bx * 8 + x, v = Math.floor(R[y * 8 + x] + 128 + 0.5);
                rec[o] = v < 0 ? 0 : v > 255 ? 255 : v;
                if (coef) coef[o] = Math.min(255, Math.floor(255 * Math.log1p(Math.abs(Fq[y * 8 + x])) / L1024 + 0.5));
              }
            }
          }
        }
        const o = new Uint8Array(W * H);
        let sse = 0;
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const a = pd[y * Wp + x], b = rec[y * Wp + x], e = a - b;
            sse += e * e;
            o[y * W + x] = view === "result" ? b : view === "coeffs" ? coef[y * Wp + x] : Math.min(255, Math.abs(e) * 8);
          }
        }
        dst = cv.matFromArray(H, W, cv.CV_8UC1, o);
        const mse = sse / (W * H), psnr = mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);
        const blocks = bw * bh, zeroPct = zeros * 100 / (blocks * 64);
        if (ctx) {
          ctx.values = scalarValues(DCT_FIELDS, [psnr, zeroPct, blocks]);
          const psnrTxt = psnr === Infinity ? "∞ (เหมือนเดิมทุกพิกเซล)" : `${fixed2(psnr)} dB`;
          ctx.info = (method === "jpeg"
            ? `DCT 8×8 แบบ JPEG quality ${quality} · สัมประสิทธิ์เป็น 0 ${fixed1(zeroPct)}% · PSNR ${psnrTxt}`
            : `เก็บความถี่ต่ำ ${K}×${K} จาก 8×8 ต่อบล็อก (${K * K}/64) · PSNR ${psnrTxt}`)
            + (view === "coeffs" ? " · มุมซ้ายบนของทุกบล็อก = ความถี่ต่ำ (สว่าง)" : "");
        }
        const out = dst; dst = null;
        return out;
      } finally {
        for (const mm of [dst, pad, g]) if (mm) mm.delete();
      }
    },
  },

  // P-8 (2026-10-17) — ตัดสิน OK / NG: อ่านค่าล่าสุดของกล่องวัดบนสายหลัก (ctx.upstream จาก worker) เทียบเกณฑ์ · ภาพ = ภาพที่เข้ามา + กรอบเขียว/แดง + ป้าย
  // IP batch 5A — ใช้แกน Extended Max เดียวกับ Morphology; watershed เดิมคงเดิม
  watershed_marker: {
    name: "Marker Watershed (แยกด้วย marker)",
    needs: "any",
    modeHint: "ใช้ยอดของระยะห่างจากขอบเป็นจุดเริ่มแยกวัตถุที่ติดกัน",
    desc: "แยกวัตถุที่ติดกันด้วย marker จาก Extended Max บน Distance Transform · เลือกวัตถุมืดหรือสว่างให้ตรงภาพ · เพิ่ม h เพื่อลดยอดเล็กและลดจำนวนชิ้นที่แยก · ภาพรายละเอียดมากอาจช้า เหมาะกับภาพนิ่ง",
    params: [
      { key: "polarity", type: "select", label: "สีของวัตถุ", options: [{ value: "dark", label: "วัตถุมืดบนพื้นสว่าง" }, { value: "bright", label: "วัตถุสว่างบนพื้นมืด" }], default: "dark", desc: "เลือกด้านของ Otsu ที่เป็นวัตถุ" },
      { key: "hHeight", type: "number", label: "ความสูง h ของ marker (px)", min: 1, max: 50, step: 1, default: 2, hint: "มาก = marker น้อยลง แยกน้อยลง" },
      { key: "view", type: "select", label: "แสดงผลเป็น", options: [{ value: "boundaries", label: "เส้นแบ่ง (แดง) บนภาพเดิม" }, { value: "labels", label: "ระบายสีแต่ละวัตถุ" }], default: "boundaries", desc: "สีประจำวัตถุคงเดิมเมื่อ label เดิม" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(OBJECT_FIELDS, []); } return src.clone(); }
      const h = intIn(p.hHeight, 2, 1, 50);
      const polarity = ownKey({ dark: true, bright: true }, p.polarity, "dark");
      const view = ownKey({ boundaries: true, labels: true }, p.view, "boundaries");
      let rgb = null, gray = null, fg = null, ker = null, dist = null, mask = null, bg = null, markers = null, dst = null;
      try {
        rgb = rgbIfNeeded(cv, src);
        const x = rgb || src, W = x.cols, H = x.rows, N = W * H;
        gray = new cv.Mat(); cv.cvtColor(x, gray, cv.COLOR_RGB2GRAY);
        fg = new cv.Mat(); cv.threshold(gray, fg, 0, 255, (polarity === "dark" ? cv.THRESH_BINARY_INV : cv.THRESH_BINARY) + cv.THRESH_OTSU);
        ker = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
        cv.morphologyEx(fg, fg, cv.MORPH_OPEN, ker, new cv.Point(-1, -1), 2);
        if (cv.countNonZero(fg) === 0) {
          dst = x.clone();
          if (ctx) { ctx.info = "ไม่พบวัตถุ · ลองเปลี่ยนสีของวัตถุ"; ctx.values = scalarValues(OBJECT_FIELDS, [0]); }
        } else {
          dist = new cv.Mat(); cv.distanceTransform(fg, dist, cv.DIST_L2, 5);
          const d8 = new Int32Array(N);
          { const dd = dist.data32F; for (let i = 0; i < N; i++) d8[i] = Math.min(255, Math.floor(dd[i] + 0.5)); }
          const mm = extendedMaxMask(d8, W, H, h);
          { const fd = fg.data; for (let i = 0; i < N; i++) if (!fd[i]) mm[i] = 0; }
          mask = cv.matFromArray(H, W, cv.CV_8UC1, mm);
          markers = new cv.Mat(); const objects = cv.connectedComponents(mask, markers, 8, cv.CV_32S) - 1;
          bg = new cv.Mat(); cv.dilate(fg, bg, ker, new cv.Point(-1, -1), 3);
          { const md = markers.data32S, bd = bg.data; for (let i = 0; i < N; i++) md[i] = !bd[i] ? 1 : mm[i] ? md[i] + 1 : 0; }
          cv.watershed(x, markers);
          dst = view === "boundaries" ? x.clone() : new cv.Mat(H, W, cv.CV_8UC3, new cv.Scalar(0));
          const md = markers.data32S, od = dst.data, colors = new Map();
          for (let i = 0; i < N; i++) {
            const label = md[i];
            if (view === "boundaries") { if (label === -1) od.set([255, 0, 0], i * 3); }
            else if (label === -1) od.set([255, 255, 255], i * 3);
            else if (label >= 2) {
              if (!colors.has(label)) colors.set(label, markerLabelRGB(label));
              od.set(colors.get(label), i * 3);
            }
          }
          if (ctx) { ctx.info = `แยกได้ ${objects} วัตถุ · marker จาก Extended Max h = ${h}`; ctx.values = scalarValues(OBJECT_FIELDS, [objects]); }
        }
        const out = dst; dst = null; return out;
      } finally { for (const m of [dst, markers, bg, mask, dist, ker, fg, gray, rgb]) if (m) m.delete(); }
    },
  },

  seg_score: {
    name: "Segmentation Score (IoU / Dice)",
    needs: "any",
    modeHint: "เทียบ mask ที่แบ่งได้ใน A กับ mask คำตอบใน B",
    desc: "วัดความตรงกันของ mask สองภาพ · ทั้งคู่แปลงเป็นเทาแล้วใช้ค่า > 127 เป็นวัตถุ · B ต่างขนาดถูกปรับให้เท่า A ด้วย nearest-neighbor · ขาว = ตรงกัน แดง = เกิน ฟ้า = ขาด · IoU / Dice ยิ่งใกล้ 1 ยิ่งตรงกัน · ทุกอัตราส่วนที่ตัวหารเป็น 0 → ถือว่า 1 · ต่อ mask คำตอบที่ช่อง เข้า B",
    params: [],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(SEG_FIELDS, []); } return src.clone(); }
      let ga = null, gb = null, matched = null, dst = null;
      try {
        if (!ctx || !ctx.src2 || isEmptyMat(ctx.src2)) {
          dst = rgbIfNeeded(cv, src) || src.clone();
          if (ctx) { ctx.info = ctx.src2 ? EMPTY_SRC2_INFO : NO_SRC2_INFO; delete ctx.values; }
        } else {
          ga = grayIfNeeded(cv, src); gb = grayIfNeeded(cv, ctx.src2);
          const a = ga || src, b0 = gb || ctx.src2;
          const resized = a.rows !== b0.rows || a.cols !== b0.cols;
          if (resized) { matched = new cv.Mat(); cv.resize(b0, matched, new cv.Size(a.cols, a.rows), 0, 0, cv.INTER_NEAREST); }
          const b = matched || b0, N = a.rows * a.cols;
          dst = new cv.Mat(a.rows, a.cols, cv.CV_8UC3, new cv.Scalar(0));
          const ad = a.data, bd = b.data, od = dst.data;
          let tp = 0, fp = 0, fn = 0;
          for (let i = 0; i < N; i++) {
            const aa = ad[i] > 127, bb = bd[i] > 127;
            if (aa && bb) { tp++; od.set([255, 255, 255], i * 3); }
            else if (aa) { fp++; od.set([230, 60, 60], i * 3); }
            else if (bb) { fn++; od.set([60, 140, 255], i * 3); }
          }
          const ratio = (n, d) => d === 0 ? 1 : n / d;
          const iou = ratio(tp, tp + fp + fn), dice = ratio(2 * tp, 2 * tp + fp + fn), precision = ratio(tp, tp + fp), recall = ratio(tp, tp + fn);
          if (ctx) {
            ctx.info = `IoU ${fixed4(iou)} · Dice ${fixed4(dice)} · precision ${fixed4(precision)} · recall ${fixed4(recall)}` + (resized ? RESIZED_B_INFO : "");
            ctx.values = scalarValues(SEG_FIELDS, [iou, dice, precision, recall, tp, fp, fn]);
          }
        }
        const out = dst; dst = null; return out;
      } finally { for (const m of [dst, matched, gb, ga]) if (m) m.delete(); }
    },
  },

  lowlight_enhance: {
    name: "Low-light Enhance (เพิ่มแสงภาพมืด)",
    needs: "any",
    modeHint: "กลับสี ลดหมอก แล้วกลับสีอีกครั้งเพื่อเพิ่มความสว่าง",
    desc: "เพิ่มความสว่างในภาพมืดด้วย 255 − Dehaze(255 − ภาพ) · ใช้ dark channel และ guided filter ตัวเดียวกับ Dehaze · แนวเดียวกับ imlocalbrighten ของ MATLAB แต่รายละเอียดต่างกัน · เปิดคงบริเวณสว่างเพื่อผสมภาพเดิมตามความสว่างแต่ละพิกเซล · ภาพใหญ่หรือหน้าต่างกว้างอาจช้า",
    params: [
      { key: "amount", type: "number", label: "ปริมาณเพิ่มความสว่าง", min: 0.1, max: 1, step: 0.05, default: 1, hint: "มาก = เพิ่มความสว่างมากขึ้น" },
      { key: "patch", type: "number", label: "ขนาดหน้าต่าง dark channel (px)", min: 3, max: 31, step: 2, default: 15, desc: "หน้าต่างใหญ่ช่วยประมาณบริเวณมืดที่กว้างขึ้น" },
      { key: "refine", type: "boolean", label: "เกลี่ยแผนที่ตามขอบวัตถุ (guided filter)", default: true, desc: "ลดรอยบล็อกตามขอบวัตถุ" },
      { key: "alphaBlend", type: "boolean", label: "คงบริเวณที่สว่างอยู่แล้วไว้", default: false, desc: "ส่วนสว่างใช้ภาพเดิมมาก ส่วนมืดใช้ภาพที่เพิ่มแสงมาก" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const amount = numIn(p.amount, 1, 0.1, 1), patch = ensureOdd(intIn(p.patch, 15, 3, 31));
      const refine = boolOr(p.refine, true), blend = boolOr(p.alphaBlend, false);
      let rgb = null, inv = null, dh = null, g = null, dst = null;
      try {
        rgb = dropAlpha(cv, src); const x = rgb || src;
        inv = new cv.Mat(); cv.bitwise_not(x, inv);
        dh = OPS.dehaze.apply(cv, inv, { amount, patch, refine, view: "result" });
        dst = new cv.Mat(); cv.bitwise_not(dh, dst);
        if (blend) {
          g = grayIfNeeded(cv, x);
          const gd = (g || x).data, xd = x.data, od = dst.data, ch = x.channels();
          for (let i = 0; i < gd.length; i++) {
            const a = gd[i] / 255;
            for (let c = 0; c < ch; c++) { const j = i * ch + c; od[j] = Math.floor(a * xd[j] + (1 - a) * od[j] + 0.5); }
          }
        }
        if (ctx) ctx.info = `เพิ่มแสง ${amount} · หน้าต่าง ${patch}×${patch}` + (refine ? " · เกลี่ยตามขอบ" : " · ไม่เกลี่ยตามขอบ") + (blend ? " · คงบริเวณสว่าง" : "");
        const out = dst; dst = null; return out;
      } finally { for (const m of [dst, g, dh, inv, rgb]) if (m) m.delete(); }
    },
  },

  colorspace_inverse: {
    name: "Convert Back to RGB (แปลงกลับสี)",
    needs: "any",
    modeHint: "ใช้ต่อจาก Color Space Convert โดยเลือกปริภูมิสีเดียวกัน",
    desc: "แปลงค่า 3 ช่องแบบ 8 bit กลับเป็น RGB · วิธีใช้: Color Space Convert → ปรับช่องสี → Convert Back to RGB · เลือกปริภูมิเดียวกับตอนแปลงไป · การบีบค่า 8 bit และสีที่เกินขอบเขตทำให้กลับมาไม่เท่าภาพเดิมทุกค่า โดยเฉพาะ Lab / Luv / YUV / XYZ",
    params: [], // ตั้งจาก colorspace.params หลังสร้าง OPS เพื่อใช้ตัวเลือกแหล่งเดียวกัน
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const space = ownKey(INVERSE_CODES, p.space, "hsv");
      let rgb = null, dst = null;
      try {
        dst = new cv.Mat();
        if (src.channels() === 1) {
          cv.cvtColor(src, dst, cv.COLOR_GRAY2RGB);
          if (ctx) ctx.info = "ภาพเข้าเป็นภาพเทา ไม่มีข้อมูลสี 3 ช่องให้แปลงกลับ — ต่อจาก Color Space Convert";
        } else {
          rgb = dropAlpha(cv, src);
          if (space === "hsv") {
            // H uses OpenCV's 0..180 scale. Float64 + floor(x + .5) in both runtimes
            // avoids the 8-bit HSV conversion rounding difference in OpenCV 4.x/5.x.
            const x = rgb || src, data = x.data;
            dst.create(x.rows, x.cols, cv.CV_8UC3);
            const out = dst.data;
            const order = [[0, 3, 1], [2, 0, 1], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]];
            for (let i = 0; i < data.length; i += 3) {
              const hue = data[i] / 30, sector = Math.floor(hue), f = hue - sector;
              const s = data[i + 1] / 255, v = data[i + 2];
              const tab = [v, v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))];
              const indices = order[sector % 6];
              for (let c = 0; c < 3; c++) out[i + c] = Math.floor(tab[indices[c]] + 0.5);
            }
          } else cv.cvtColor(rgb || src, dst, cv[INVERSE_CODES[space]]);
          if (ctx) ctx.info = `${space.toUpperCase()} → RGB · ค่า 8 bit อาจสูญเสียรายละเอียดสี`;
        }
        const out = dst; dst = null; return out;
      } finally { if (dst) dst.delete(); if (rgb) rgb.delete(); }
    },
  },

  euler_number: {
    name: "Hole Count / Euler (นับรูในวัตถุ)",
    needs: "any",
    modeHint: "นับวัตถุและรูปิดสนิท แล้วคำนวณ Euler = วัตถุ − รู",
    desc: "แปลงเป็นเทาและใช้ค่า > 127 เป็นวัตถุ เหมือน Region Properties · รูที่ต่อกับขอบภาพไม่นับ · วัตถุต่อกันแบบ 8 ทิศใช้พื้นหลัง 4 ทิศ และสลับกันเมื่อเลือก 4 ทิศ · เทาอ่อน = วัตถุ แดง = รู · เทียบ bweuler ของ MATLAB",
    params: [{ key: "conn", type: "select", label: "การเชื่อมต่อของวัตถุ", options: [{ value: "8", label: "8 ทิศ (รวมมุมทแยง)" }, { value: "4", label: "4 ทิศ (บน ล่าง ซ้าย ขวา)" }], default: "8", hint: "ค่าเริ่มต้นเดียวกับ bweuler ของ MATLAB" }],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(EULER_FIELDS, []); } return src.clone(); }
      const conn = Number(ownKey({ 8: true, 4: true }, p.conn, "8"));
      let g = null, bin = null, inv = null, fgLabels = null, bgLabels = null, dst = null;
      try {
        g = grayIfNeeded(cv, src); const x = g || src;
        bin = new cv.Mat(); cv.threshold(x, bin, 127, 255, cv.THRESH_BINARY);
        fgLabels = new cv.Mat(); const objects = cv.connectedComponents(bin, fgLabels, conn, cv.CV_32S) - 1;
        inv = new cv.Mat(); cv.bitwise_not(bin, inv);
        bgLabels = new cv.Mat(); const nb = cv.connectedComponents(inv, bgLabels, conn === 8 ? 4 : 8, cv.CV_32S);
        const W = x.cols, H = x.rows, touches = new Uint8Array(nb);
        { const ld = bgLabels.data32S; for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) if (!y || !xx || y === H - 1 || xx === W - 1) touches[ld[y * W + xx]] = 1; }
        let holes = 0; for (let i = 1; i < nb; i++) if (!touches[i]) holes++;
        dst = new cv.Mat(H, W, cv.CV_8UC3, new cv.Scalar(0));
        const bd = bin.data, ld = bgLabels.data32S, od = dst.data;
        for (let i = 0; i < W * H; i++) if (bd[i]) od.set([200, 200, 200], i * 3); else if (ld[i] && !touches[ld[i]]) od.set([230, 60, 60], i * 3);
        if (ctx) { ctx.info = `วัตถุ ${objects} ชิ้น · รู ${holes} รู · Euler = ${objects - holes}`; ctx.values = scalarValues(EULER_FIELDS, [objects, holes, objects - holes]); }
        const out = dst; dst = null; return out;
      } finally { for (const m of [dst, bgLabels, fgLabels, inv, bin, g]) if (m) m.delete(); }
    },
  },

  // IP batch 5B — additive operations; existing operation bodies stay unchanged.
  image_pyramid: {
    name: "Image Pyramid (พีระมิดภาพ ย่อ/ขยายทีละขั้น)",
    needs: "any",
    modeHint: "ย่อหรือขยายด้วย Gaussian แล้วดูทีละขั้นหรือเรียงทุกขั้น",
    desc: "สร้างภาพหลายขนาดด้วย Gaussian 5×5 ของ OpenCV เหมือน impyramid · Laplacian แสดงรายละเอียดที่หายไปด้วยค่าสีเทากลาง 128 · ภาพเทายังคงเป็นเทา · หยุดย่อเมื่อด้านใดเล็กกว่า 2 px และหยุดขยายก่อนเกิน 4096 px · ภาพเรียงวางชิดบน เว้น 4 px บนพื้นดำ",
    params: [
      { key: "kind", type: "select", label: "ชนิดพีระมิด", options: [{ value: "gaussian", label: "Gaussian (ภาพเบลอย่อลง)" }, { value: "laplacian", label: "Laplacian (รายละเอียดที่หายไปในแต่ละขั้น)" }], default: "gaussian", desc: "Gaussian ดูภาพหลายขนาด · Laplacian ดูรายละเอียดของแต่ละขั้น" },
      { key: "direction", visibleWhen: [{ key: "kind", in: ["gaussian"] }], type: "select", label: "ทิศทาง", options: [{ value: "reduce", label: "ย่อ (Reduce)" }, { value: "expand", label: "ขยาย (Expand)" }], default: "reduce", hint: "แต่ละขั้นย่อครึ่งหนึ่งหรือขยายสองเท่า" },
      { key: "levels", type: "number", label: "จำนวนขั้น", min: 1, max: 5, step: 1, default: 1, hint: "หยุดก่อนเมื่อภาพเล็กเกินไปหรือขยายเกินขอบเขต" },
      { key: "view", type: "select", label: "ภาพที่แสดง", options: [{ value: "level", label: "เฉพาะขั้นสุดท้าย" }, { value: "mosaic", label: "ทุกขั้นเรียงกัน" }], default: "level", desc: "ภาพเรียงอาจแสดงไม่ครบถ้าความกว้างรวมเกิน 4096 px" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(PYRAMID_FIELDS, []); } return src.clone(); }
      const kind = ownKey({ gaussian: true, laplacian: true }, p.kind, "gaussian");
      const direction = kind === "laplacian" ? "reduce" : ownKey({ reduce: true, expand: true }, p.direction, "reduce");
      const levels = intIn(p.levels, 1, 1, 5), view = ownKey({ level: true, mosaic: true }, p.view, "level");
      const owned = [];
      const mat = (...args) => { const m = new cv.Mat(...args); owned.push(m); return m; };
      let dst = null;
      try {
        const alpha = dropAlpha(cv, src);
        if (alpha) owned.push(alpha);
        const input = alpha || src, gs = [input], ls = [];
        let stop = "";
        for (let k = 0; k < levels; k++) {
          const prev = gs[gs.length - 1];
          if (direction === "reduce" && (prev.cols < 2 || prev.rows < 2)) { stop = ` · หยุดที่ขั้น ${k}: ภาพเล็กเกินย่อต่อ`; break; }
          if (direction === "expand" && Math.max(prev.cols * 2, prev.rows * 2) > 4096) { stop = ` · หยุดที่ขั้น ${k}: ขยายต่อจะเกิน 4096 px`; break; }
          const next = mat();
          if (direction === "reduce") cv.pyrDown(prev, next);
          else cv.pyrUp(prev, next);
          gs.push(next);
          if (kind === "laplacian") {
            const up = mat(), a = mat(), b = mat(), delta = mat(), tile = mat();
            cv.pyrUp(next, up, new cv.Size(prev.cols, prev.rows));
            prev.convertTo(a, cv.CV_16S);
            up.convertTo(b, cv.CV_16S);
            cv.subtract(a, b, delta);
            delta.convertTo(tile, cv.CV_8U, 1, 128);
            ls.push(tile);
          }
        }
        if (kind === "laplacian" && !ls.length) ls.push(mat(input.rows, input.cols, input.type(), new cv.Scalar(128, 128, 128, 128)));
        const tiles = kind === "laplacian" ? ls : gs, last = tiles[tiles.length - 1];
        if (view === "level") dst = last.clone();
        else {
          const shown = [];
          let width = 0, height = 0;
          for (const tile of tiles) {
            const nextWidth = width + (shown.length ? 4 : 0) + tile.cols;
            if (shown.length && nextWidth > 4096) { stop += ` · แสดง ${shown.length}/${tiles.length} ภาพ: ความกว้างรวมเกิน 4096 px`; break; }
            shown.push(tile);
            width = nextWidth;
            height = Math.max(height, tile.rows);
          }
          dst = new cv.Mat(height, width, input.type(), new cv.Scalar(0));
          const od = dst.data, ch = input.channels();
          let offset = 0;
          for (const tile of shown) {
            const td = tile.data;
            for (let y = 0; y < tile.rows; y++) od.set(td.subarray(y * tile.cols * ch, (y + 1) * tile.cols * ch), (y * width + offset) * ch);
            offset += tile.cols + 4;
          }
        }
        if (ctx) {
          ctx.info = `${kind === "laplacian" ? "Laplacian" : "Gaussian"} ${direction === "reduce" ? "ย่อ" : "ขยาย"} ${gs.length - 1} ขั้น: ${input.cols}×${input.rows} → ${last.cols}×${last.rows}` + stop;
          ctx.values = scalarValues(PYRAMID_FIELDS, [gs.length - 1, last.cols, last.rows]);
        }
        const out = dst; dst = null; return out;
      } finally { if (dst) dst.delete(); for (const m of owned) m.delete(); }
    },
  },

  frangi_vesselness: {
    name: "Ridge / Vesselness (Frangi · หาเส้น/รอยแตก/เส้นเลือด)",
    needs: "gray",
    modeHint: SLOW_MODE_HINT + " · Frangi หลายความหนา ภาพใหญ่หรือ σ กว้างใช้เวลามาก",
    desc: "เน้นเส้นหรือรอยแตกจากความโค้งของภาพหลายขนาด คล้าย fibermetric · สว่าง = เป็นเส้นมาก · เลือกเส้นมืดเพื่อหารอยแตกบนผิว · ต่อ Otsu หรือ Binary Threshold เพื่อได้ mask ของเส้น · beta ต่ำช่วยลดก้อนกลม · σ ใหญ่หรือภาพใหญ่ใช้เวลามาก",
    params: [
      { key: "polarity", type: "select", label: "สีของเส้น", options: [{ value: "bright", label: "เส้นสว่างบนพื้นมืด" }, { value: "dark", label: "เส้นมืดบนพื้นสว่าง (รอยแตก)" }], default: "bright", hint: "เลือกตามความสว่างของเส้นเทียบกับพื้นหลัง" },
      { key: "sigmaMin", type: "number", label: "ความหนาเส้นเล็กสุด (σ, px)", min: 1, max: 8, step: 0.5, default: 1, hint: "เริ่มหาเส้นที่ขนาดนี้ แล้วเพิ่มทีละ 0.5" },
      { key: "sigmaMax", type: "number", label: "ความหนาเส้นใหญ่สุด (σ, px)", min: 1, max: 10, step: 0.5, default: 3, desc: "ถ้าน้อยกว่าค่าต้นจะสลับสองค่าให้เอง" },
      { key: "beta", type: "number", label: "ความไวต่อก้อนกลม (beta)", min: 0.1, max: 2, step: 0.05, default: 0.5, hint: "น้อย = ตัดก้อนกลมออกมากขึ้น" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(RIDGE_FIELDS, []); } return src.clone(); }
      let lo = numIn(p.sigmaMin, 1, 1, 8), hi = numIn(p.sigmaMax, 3, 1, 10);
      if (lo > hi) [lo, hi] = [hi, lo];
      const beta = numIn(p.beta, 0.5, 0.1, 2), polarity = ownKey({ bright: true, dark: true }, p.polarity, "bright");
      const nSteps = Math.floor((hi - lo) / 0.5 + 1e-9);
      let gray = null, input = null, blur = null, xx = null, yy = null, xy = null, dst = null;
      try {
        gray = grayIfNeeded(cv, src);
        const x = gray || src, n = x.rows * x.cols, data = x.data;
        const best = new Float64Array(n), l1 = new Float64Array(n), l2 = new Float64Array(n);
        // A constant has exactly zero Hessian; avoid normalizing floating cancellation noise.
        const constant = data.every((v) => v === data[0]);
        if (!constant) {
          input = cv.matFromArray(x.rows, x.cols, cv.CV_64FC1, Float64Array.from(data, (v) => v / 255));
          blur = new cv.Mat(); xx = new cv.Mat(); yy = new cv.Mat(); xy = new cv.Mat();
          for (let k = 0; k <= nSteps; k++) {
            const sigma = lo + 0.5 * k, size = 2 * Math.ceil(3 * sigma) + 1, scale = sigma * sigma;
            cv.GaussianBlur(input, blur, new cv.Size(size, size), sigma, sigma, cv.BORDER_REPLICATE);
            cv.Sobel(blur, xx, cv.CV_64F, 2, 0, 3, 1, 0, cv.BORDER_REPLICATE);
            cv.Sobel(blur, yy, cv.CV_64F, 0, 2, 3, 1, 0, cv.BORDER_REPLICATE);
            cv.Sobel(blur, xy, cv.CV_64F, 1, 1, 3, 1, 0, cv.BORDER_REPLICATE);
            const xd = xx.data64F, yd = yy.data64F, zd = xy.data64F;
            let maxS = 0;
            for (let i = 0; i < n; i++) {
              const a = xd[i] * scale, b = yd[i] * scale, d = zd[i] * scale;
              const diff = a - b, tmp = Math.sqrt(diff * diff + 4 * d * d);
              const u = (a + b + tmp) / 2, v = (a + b - tmp) / 2;
              const ordered = Math.abs(u) <= Math.abs(v);
              l1[i] = ordered ? u : v; l2[i] = ordered ? v : u;
              const s = Math.sqrt(l1[i] * l1[i] + l2[i] * l2[i]);
              if (s > maxS) maxS = s;
            }
            const c = 0.5 * maxS;
            if (!c) continue;
            for (let i = 0; i < n; i++) {
              const a = l1[i], b = l2[i];
              if (!b || (polarity === "bright" ? b > 0 : b < 0)) continue;
              const ratio = a / b, rb2 = ratio * ratio, s2 = a * a + b * b;
              const v = Math.exp(-rb2 / (2 * beta * beta)) * (1 - Math.exp(-s2 / (2 * c * c)));
              if (v > best[i]) best[i] = v;
            }
          }
        }
        let maxV = 0, count = 0;
        for (const v of best) if (v > maxV) maxV = v;
        dst = new cv.Mat(x.rows, x.cols, cv.CV_8UC1, new cv.Scalar(0));
        const outData = dst.data;
        for (let i = 0; i < n; i++) {
          outData[i] = maxV ? Math.floor(best[i] / maxV * 255 + 0.5) : 0;
          if (outData[i] >= 128) count++;
        }
        if (ctx) {
          ctx.info = `Frangi σ ${lo}–${hi} (${nSteps + 1} ขนาด) · ${polarity === "bright" ? "เส้นสว่าง" : "เส้นมืด"}` + (!maxV ? " · ไม่พบโครงสร้างเส้น" : "");
          ctx.values = scalarValues(RIDGE_FIELDS, [Math.floor(maxV * 10000 + 0.5) / 10000, count / n * 100]);
        }
        const out = dst; dst = null; return out;
      } finally { for (const m of [dst, xy, yy, xx, blur, input, gray]) if (m) m.delete(); }
    },
  },

  anisotropic_diffusion: {
    name: "Anisotropic Diffusion (ลดนอยส์แต่คงขอบ · Perona–Malik)",
    needs: "any",
    modeHint: SLOW_MODE_HINT + " · เกลี่ยนอยส์ 4 ทิศทางโดยลดการเกลี่ยข้ามขอบวัตถุ",
    desc: "ลดนอยส์แต่คงขอบด้วย Perona–Malik คล้าย imdiffusefilt · ทำแยกแต่ละช่องสี · แต่ละรอบใช้ค่าจากรอบก่อนพร้อมกันทั้งภาพ · kappa ต่ำเก็บขอบความต่างน้อยไว้ด้วย ค่าสูงเกลี่ยได้มากขึ้น · เพิ่มรอบเพื่อให้พื้นเรียบขึ้น · ภาพใหญ่และจำนวนรอบมากใช้เวลามาก",
    params: [
      { key: "iterations", type: "number", label: "จำนวนรอบ", min: 1, max: 50, step: 1, default: 10, hint: "มาก = เรียบมากขึ้นและใช้เวลามากขึ้น" },
      { key: "kappa", type: "number", label: "ความต่างระดับสีที่ถือว่าเป็นขอบ (κ)", min: 1, max: 100, step: 1, default: 20, desc: "ความต่างต่ำกว่า κ ถูกเกลี่ยมาก ความต่างสูงถูกเก็บไว้มาก" },
      { key: "conduction", type: "select", label: "วิธีลดการเกลี่ยที่ขอบ", options: [{ value: "exponential", label: "Exponential (เก็บขอบคม)" }, { value: "quadratic", label: "Quadratic (เกลี่ยพื้นที่กว้าง)" }], default: "exponential", hint: "เริ่มด้วย Exponential แล้วเปรียบเทียบขอบที่ได้" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const iterations = intIn(p.iterations, 10, 1, 50), kappa = numIn(p.kappa, 20, 1, 100);
      const conduction = ownKey({ exponential: true, quadratic: true }, p.conduction, "exponential");
      let alpha = null, dst = null;
      try {
        alpha = dropAlpha(cv, src);
        const x = alpha || src, w = x.cols, h = x.rows, ch = x.channels(), stride = w * ch;
        let current = Float64Array.from(x.data), next = new Float64Array(current.length);
        const flux = conduction === "exponential" ? (d) => { const r = d / kappa; return Math.exp(-(r * r)) * d; } : (d) => { const r = d / kappa; return (1 / (1 + r * r)) * d; };
        for (let k = 0; k < iterations; k++) {
          for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) for (let c = 0; c < ch; c++) {
            const i = y * stride + xx * ch + c, v = current[i];
            const dn = y ? current[i - stride] - v : 0, ds = y + 1 < h ? current[i + stride] - v : 0;
            const de = xx + 1 < w ? current[i + ch] - v : 0, dw = xx ? current[i - ch] - v : 0;
            next[i] = v + 0.25 * (flux(dn) + flux(ds) + flux(de) + flux(dw));
          }
          [current, next] = [next, current];
        }
        dst = new cv.Mat(h, w, x.type());
        const data = dst.data;
        for (let i = 0; i < data.length; i++) data[i] = Math.max(0, Math.min(255, Math.floor(current[i] + 0.5)));
        if (ctx) ctx.info = `Perona–Malik ${iterations} รอบ · κ = ${kappa} · ${conduction === "exponential" ? "เอ็กซ์โพเนนเชียล" : "กำลังสอง"}`;
        const out = dst; dst = null; return out;
      } finally { if (dst) dst.delete(); if (alpha) alpha.delete(); }
    },
  },

  debayer: {
    name: "Debayer / Demosaic (แปลงภาพดิบจากเซนเซอร์เป็นภาพสี)",
    needs: "any",
    modeHint: "แปลงภาพดิบ Bayer8 เป็นสี หรือจำลองภาพดิบเพื่อทดลอง",
    desc: "เซนเซอร์กล้องสีส่วนใหญ่ใช้ฟิลเตอร์ Bayer แต่ละพิกเซลเห็นแค่ 1 สี · ภาพ machine vision แบบ raw (Bayer8) ต้อง debayer ก่อน · ถ้าเพี้ยนเขียวหรือม่วงให้ลองเปลี่ยน pattern · ทดลองด้วย mosaic → demosaic · ภาพสีที่เข้า demosaic จะแปลงเป็นเทาก่อน",
    params: [
      { key: "mode", type: "select", label: "การแปลง", options: [{ value: "demosaic", label: "Debayer: ภาพดิบ → ภาพสี" }, { value: "mosaic", label: "จำลองภาพดิบจากภาพสี (เพื่อทดลอง)" }], default: "demosaic", hint: "เลือกจำลองเพื่อทดสอบ Bayer pattern ด้วยภาพสีทั่วไป" },
      { key: "pattern", type: "select", label: "ลำดับฟิลเตอร์สี Bayer", options: [{ value: "rggb", label: "RGGB (R มุมซ้ายบน)" }, { value: "bggr", label: "BGGR (B มุมซ้ายบน)" }, { value: "grbg", label: "GRBG (G มุมซ้ายบน R อยู่ขวา)" }, { value: "gbrg", label: "GBRG (G มุมซ้ายบน B อยู่ขวา)" }], default: "rggb", desc: "ต้องตรงกับเซนเซอร์กล้องหรือขั้นที่จำลองภาพดิบ" },
      { key: "method", visibleWhen: [{ key: "mode", in: ["demosaic"] }], type: "select", label: "วิธีเติมสี", options: [{ value: "bilinear", label: "Bilinear (เร็ว)" }, { value: "ea", label: "Edge-aware (ลดสีเพี้ยนที่ขอบ)" }], default: "bilinear", hint: "Bilinear เร็วสุด · Edge-aware ช้ากว่าเล็กน้อยแต่สีที่ขอบเพี้ยนน้อยกว่า" },
    ],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
      const mode = ownKey({ demosaic: true, mosaic: true }, p.mode, "demosaic");
      const pattern = ownKey(BAYER_TILES, p.pattern, "rggb"), method = ownKey({ bilinear: true, ea: true }, p.method, "bilinear");
      let alpha = null, gray = null, dst = null;
      try {
        alpha = dropAlpha(cv, src);
        const x = alpha || src;
        dst = new cv.Mat();
        if (mode === "mosaic") {
          dst.create(x.rows, x.cols, cv.CV_8UC1);
          const ch = x.channels(), data = x.data, out = dst.data, tile = BAYER_TILES[pattern];
          for (let y = 0; y < x.rows; y++) for (let xx = 0; xx < x.cols; xx++) {
            const i = y * x.cols + xx;
            out[i] = data[i * ch + (ch === 1 ? 0 : tile[(y % 2) * 2 + xx % 2])];
          }
          if (ctx) ctx.info = `จำลองภาพดิบ ${pattern.toUpperCase()} (1 ช่อง)`;
        } else {
          gray = grayIfNeeded(cv, x);
          const raw = gray || x, prefix = BAYER_CODES[pattern];
          // opencv.js 4.10 COLOR_Bayer*2RGB_VNG ให้ผลเลื่อน 2 แถว + สีเพี้ยน จึงไม่รองรับ VNG (Bilinear/EA ตรงกับ cv2)
          cv.cvtColor(raw, dst, cv[`COLOR_${prefix}2RGB${method === "ea" ? "_EA" : ""}`]);
          if (ctx) ctx.info = `Debayer ${pattern.toUpperCase()} · ${method === "ea" ? "Edge-aware" : "Bilinear"}`;
        }
        const out = dst; dst = null; return out;
      } finally { for (const m of [dst, gray, alpha]) if (m) m.delete(); }
    },
  },

  piqe: {
    name: "PIQE (คะแนนคุณภาพภาพ ไม่ต้องมีภาพอ้างอิง)",
    needs: "any",
    usesSrc2: false, // วัดภาพ A ใบเดียว แม้ host จะเก็บพอร์ต B ไว้สำหรับวิธีอื่น
    modeHint: "วัดภาพเดียว 0–100 ค่าต่ำ = คุณภาพดี พร้อมระบายบริเวณที่มีปัญหา",
    desc: "ประเมินคุณภาพภาพโดยไม่ต้องมีภาพอ้างอิงตามแนว Venkatanath (2015) · ไม่ใช้ภาพ B · คะแนนอาจไม่ตรงกับ MATLAB ทุกหลัก · ค่าต่ำ = คุณภาพดี · ใช้บล็อก 16×16 ตัดเศษขอบขวา/ล่างทิ้ง · แดง = รอยบล็อก ฟ้า = นอยส์ ม่วง = ทั้งสอง · ถ้าไม่มีบริเวณที่มีรายละเอียดพอจะประเมินจะได้คะแนน 100 จากสูตร ไม่ได้แปลว่าภาพแย่",
    params: [],
    apply(cv, src, p, ctx) {
      if (isEmptyMat(src)) { if (ctx) { ctx.info = "ภาพว่าง"; ctx.values = scalarValues(PIQE_FIELDS, []); } return src.clone(); }
      let gray = null, input = null, square = null, mean = null, meanSquare = null, dst = null;
      try {
        dst = rgbIfNeeded(cv, src) || src.clone();
        if (src.cols < 16 || src.rows < 16) {
          if (ctx) { ctx.info = "ภาพเล็กกว่า 16×16 ประเมินไม่ได้"; ctx.values = scalarValues(PIQE_FIELDS, []); }
        } else {
          gray = grayIfNeeded(cv, src);
          const x = gray || src, w = x.cols, h = x.rows;
          input = new cv.Mat(); x.convertTo(input, cv.CV_64F);
          square = new cv.Mat(h, w, cv.CV_64FC1);
          const id = input.data64F, sd = square.data64F;
          for (let i = 0; i < id.length; i++) sd[i] = id[i] * id[i];
          mean = new cv.Mat(); meanSquare = new cv.Mat();
          cv.GaussianBlur(input, mean, new cv.Size(7, 7), 7 / 6, 7 / 6, cv.BORDER_REPLICATE);
          cv.GaussianBlur(square, meanSquare, new cv.Size(7, 7), 7 / 6, 7 / 6, cv.BORDER_REPLICATE);
          const md = mean.data64F, qd = meanSquare.data64F, mscn = new Float64Array(id.length);
          for (let i = 0; i < id.length; i++) mscn[i] = (id[i] - md[i]) / (Math.sqrt(Math.abs(qd[i] - md[i] * md[i])) + 1);
          let active = 0, artifactCount = 0, noiseCount = 0, distortion = 0;
          const od = dst.data;
          for (let by = 0; by + 16 <= h; by += 16) for (let bx = 0; bx + 16 <= w; bx += 16) {
            const block = [], center = [], surround = [];
            for (let y = 0; y < 16; y++) for (let xx = 0; xx < 16; xx++) {
              const v = mscn[(by + y) * w + bx + xx];
              block.push(v);
              (xx === 7 || xx === 8 ? center : surround).push(v);
            }
            const variance = variance5B(block);
            if (variance <= 0.1) continue;
            active++;
            const edges = [block.slice(0, 16), block.slice(240), Array.from({ length: 16 }, (_, i) => block[i * 16]), Array.from({ length: 16 }, (_, i) => block[i * 16 + 15])];
            let artifact = false;
            for (const edge of edges) for (let k = 0; k <= 10; k++) if (Math.sqrt(variance5B(edge.slice(k, k + 6))) < 0.1) artifact = true;
            const sc = Math.sqrt(variance5B(center)), ss = Math.sqrt(variance5B(surround)), denom = Math.max(sc, ss);
            const beta = denom ? Math.abs(sc - ss) / denom : 0;
            const noise = Math.sqrt(variance) > 2 * beta;
            if (artifact) { artifactCount++; distortion += 1 - variance; }
            if (noise) { noiseCount++; distortion += variance; }
            if (artifact || noise) {
              const color = artifact && noise ? [170, 80, 220] : artifact ? [230, 60, 60] : [60, 140, 255];
              for (let y = by; y < by + 16; y++) for (let xx = bx; xx < bx + 16; xx++) for (let c = 0; c < 3; c++) {
                const i = (y * w + xx) * 3 + c;
                od[i] = Math.floor((3 * od[i] + 2 * color[c] + 2) / 5);
              }
            }
          }
          const score = Math.max(0, Math.min(100, (distortion + 1) / (active + 1) * 100));
          const grade = score <= 20 ? "ดีมาก" : score <= 35 ? "ดี" : score <= 50 ? "พอใช้" : score <= 80 ? "แย่" : "แย่มาก";
          if (ctx) {
            ctx.info = `PIQE = ${fixed2(score)} (${grade}) · active ${active} block · artifact ${artifactCount} · noise ${noiseCount}` + (!active ? " · ไม่มีบริเวณที่มีรายละเอียดพอจะประเมิน (คะแนน 100 จากสูตร ไม่ได้แปลว่าภาพแย่)" : "");
            ctx.values = scalarValues(PIQE_FIELDS, [score, active, artifactCount, noiseCount]);
          }
        }
        const out = dst; dst = null; return out;
      } finally { for (const m of [dst, meanSquare, mean, square, input, gray]) if (m) m.delete(); }
    },
  },

  pass_fail: {
    name: "Pass / Fail (ตัดสิน OK / NG)",
    needs: "any",
    desc: "ตัดสินว่างานผ่าน (OK) หรือไม่ผ่าน (NG) จากตัวเลขที่กล่องวัดค่าก่อนหน้าวัดได้ — ต่อกล่องนี้ 'ต่อจาก' กล่องวัด เช่น Region Properties → Pass / Fail · ภาพที่ออกคือภาพเดิม + กรอบสีเขียว (OK) หรือแดง (NG) + ป้าย OK/NG · ถ้าไม่มีค่าให้ตัดสิน (ยังไม่ได้ต่อกล่องวัด, ไม่พบวัตถุ, ค่าว่าง) จะตัดสินเป็น NG เสมอ (กันของเสียหลุด) · ใช้ค่าของกล่องวัดล่าสุดบนสายหลักเท่านั้น (ไม่ข้ามไปสาย B) กล่องที่ปิด (bypass) ถูกข้าม · ⚠️ Region Properties วัดและส่งค่าเฉพาะวัตถุที่ใหญ่ที่สุดตามค่า 'แสดงวัตถุสูงสุด' — พื้นที่รวม/ใหญ่สุดจึงคิดจากชิ้นที่แสดงเท่านั้น (จำนวนวัตถุใช้ยอดที่พบจริงทั้งหมด)",
    params: [
      { key: "metric", type: "select", label: "ค่าที่ใช้ตัดสิน", options: [
        { value: "count", label: "จำนวนวัตถุ (ต่อหลัง Region Properties)" },
        { value: "area_sum", label: "พื้นที่รวมทุกชิ้น (ต่อหลัง Region Properties)", hint: "px หรือ มม.² ถ้าตั้ง มม./px ที่กล่อง Region Properties" },
        { value: "area_max", label: "พื้นที่ชิ้นที่ใหญ่สุด (ต่อหลัง Region Properties)", hint: "px หรือ มม.² ถ้าตั้ง มม./px ที่กล่อง Region Properties" },
        { value: "roi_mean", label: "ความสว่างเฉลี่ยใน ROI (ต่อหลัง ROI Polygon)" },
        { value: "focus", label: "คะแนนความคม (ต่อหลัง Focus / Sharpness Score)" },
        { value: "de_mean", label: "ΔE เฉลี่ย (ต่อหลัง Color Difference ΔE)" },
        { value: "iou", label: "IoU (ต่อหลัง Segmentation Score)" },
        { value: "dice", label: "Dice (ต่อหลัง Segmentation Score)" },
        { value: "piqe", label: "PIQE (ต่อหลัง PIQE · ค่าต่ำ = คุณภาพดี)" },
      ], default: "count", desc: "เลือกค่าที่จะเอามาตัดสิน — ต้องต่อกล่องวัดชนิดที่ระบุในวงเล็บไว้ก่อนหน้ากล่องนี้" },
      { key: "cond", type: "select", label: "เงื่อนไขที่ถือว่าผ่าน", options: [
        { value: "range", label: "อยู่ในช่วง ต่ำสุด–สูงสุด (รวมขอบ)" },
        { value: "ge", label: "มากกว่าหรือเท่ากับ (≥)" },
        { value: "le", label: "น้อยกว่าหรือเท่ากับ (≤)" },
        { value: "eq", label: "เท่ากับค่าเป้าหมาย (=)" },
      ], default: "range" },
      { key: "lo", type: "number", label: "ค่าต่ำสุด / ค่าเป้าหมาย", min: 0, max: PASS_FAIL_LIMIT, step: 0.1, default: 1, visibleWhen: [{ key: "cond", notIn: ["le"] }], hint: "พิมพ์ตัวเลขลงในช่องได้", desc: "ค่าที่วัดได้ต้อง ≥ ค่านี้ (โหมด 'เท่ากับ' ใช้เป็นค่าเป้าหมาย)" },
      { key: "hi", type: "number", label: "ค่าสูงสุด", min: 0, max: PASS_FAIL_LIMIT, step: 0.1, default: 100, visibleWhen: [{ key: "cond", in: ["range", "le"] }], hint: "พิมพ์ตัวเลขลงในช่องได้", desc: "ค่าที่วัดได้ต้อง ≤ ค่านี้" },
    ],
    apply(cv, src, p, ctx) {
      const v = passFailVerdict(p, ctx && ctx.upstream);
      if (ctx) {
        ctx.values = scalarValues(PASS_FAIL_FIELDS, [v.result, v.measured, v.lo, v.hi, v.reason]);
        const { cond } = passFailParams(p);
        const crit = cond === "range" ? `${pfNum(v.lo)}–${pfNum(v.hi)}` : cond === "ge" ? `≥ ${pfNum(v.lo)}` : cond === "le" ? `≤ ${pfNum(v.hi)}` : `= ${pfNum(v.lo)}`;
        ctx.info = v.reason ? `${v.result}: ${v.reason}` : `${v.result}: ค่าที่วัดได้ ${fixed2(v.measured)} · เกณฑ์ผ่าน ${crit}`;
      }
      if (isEmptyMat(src)) return src.clone();
      const dst = rgbIfNeeded(cv, src) || src.clone(); // ต้องเป็นภาพสี เพื่อให้กรอบสีเห็นได้แม้ input เป็นขาวดำ
      try {
        const W = dst.cols, H = dst.rows;
        const t = Math.max(2, Math.round(Math.min(W, H) / 50)), half = Math.floor(t / 2);
        const color = v.result === "OK" ? new cv.Scalar(0, 200, 0, 255) : new cv.Scalar(255, 0, 0, 255);
        cv.rectangle(dst, new cv.Point(half, half), new cv.Point(W - 1 - half, H - 1 - half), color, t);
        const scale = Math.max(0.6, Math.min(W, H) / 300);
        cv.putText(dst, v.result, new cv.Point(t + 8, t + 8 + Math.round(20 * scale)), cv.FONT_HERSHEY_SIMPLEX, scale, color, Math.max(1, Math.round(scale * 2)));
      } catch (err) {
        dst.delete();
        throw err;
      }
      return dst;
    },
  },
};

// ---- ตัวช่วย/ตารางของ IP batch 4B ที่ต้องใช้ cv หรืออ้างจาก apply (function declaration ถูก hoist · const ใช้ตอนรันหลังโหลดโมดูลเสร็จ) ----
// IP batch 5A — ค่า scalars และตารางเพื่อนบ้านไม่มี Mat และไม่มี state ข้ามเฟรม
const OBJECT_FIELDS = [["objects", "จำนวนวัตถุ"]];
const SEG_FIELDS = [["iou", "IoU"], ["dice", "Dice"], ["precision", "Precision"], ["recall", "Recall"], ["tp", "พิกเซลตรงกัน"], ["fp", "พิกเซลเกิน"], ["fn", "พิกเซลขาด"]];
const PYRAMID_FIELDS = [["pyr_levels", "จำนวนขั้นที่ทำได้"], ["pyr_width", "ความกว้างขั้นสุดท้าย"], ["pyr_height", "ความสูงขั้นสุดท้าย"]];
const RIDGE_FIELDS = [["ridge_max", "การตอบสนองเส้นก่อนปรับสเกล"], ["ridge_pct", "พิกเซลเส้นสว่างอย่างน้อย 128 (%)"]];
const PIQE_FIELDS = [["piqe", "คะแนน PIQE"], ["piqe_active", "บล็อกที่มีรายละเอียด"], ["piqe_artifact", "บล็อกที่มีรอย"], ["piqe_noise", "บล็อกที่มีนอยส์"]];
const BAYER_TILES = { rggb: [0, 1, 1, 2], bggr: [2, 1, 1, 0], grbg: [1, 0, 2, 1], gbrg: [1, 2, 0, 1] };
const BAYER_CODES = { rggb: "BayerBG", bggr: "BayerRG", grbg: "BayerGB", gbrg: "BayerGR" };
// Two passes, row-major addition, N-1 divisor — emitted Python uses this exact order.
function variance5B(values) {
  let sum = 0;
  for (const v of values) sum += v;
  const mean = sum / values.length;
  let ss = 0;
  for (const v of values) { const d = v - mean; ss += d * d; }
  return ss / (values.length - 1);
}
const EULER_FIELDS = [...OBJECT_FIELDS, ["holes", "จำนวนรู"], ["euler", "Euler number"]];
const INVERSE_CODES = { hsv: "COLOR_HSV2RGB", hls: "COLOR_HLS2RGB", lab: "COLOR_Lab2RGB", ycrcb: "COLOR_YCrCb2RGB", xyz: "COLOR_XYZ2RGB", luv: "COLOR_Luv2RGB", yuv: "COLOR_YUV2RGB" };
OPS.colorspace_inverse.params = [{ ...OPS.colorspace.params[0], desc: "เลือกปริภูมิเดียวกับ Color Space Convert ก่อนหน้า", hint: "ภาพเข้าต้องเก็บค่า 3 ช่องของปริภูมินี้" }];
const fixed4 = (v) => (Math.floor(v * 10000 + 0.5) / 10000).toFixed(4);
export function markerLabelRGB(label) {
  const h = ((label - 2) * 137.508) % 360 / 60, c = 200 / 255, x = c * (1 - Math.abs(h % 2 - 1)), m = 1 - c;
  const rgb = h < 1 ? [c, x, 0] : h < 2 ? [x, c, 0] : h < 3 ? [0, c, x] : h < 4 ? [0, x, c] : h < 5 ? [x, 0, c] : [c, 0, x];
  return rgb.map((v) => Math.floor((v + m) * 255 + 0.5));
}
const MORPH_5A = { clean: true, majority: true, spur: true, thin: true, shrink: true, h_minima: true, extended_min: true };
// E, NE, N, NW, W, SW, S, SE — Lam–Lee–Suen และ simple point 8/4 ใช้วงแหวนเดียวกัน
const MORPH_RING = [[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]];
function morph5ALuts() {
  const thin = [new Uint8Array(256), new Uint8Array(256)], simple = new Uint8Array(256), count = new Uint8Array(256);
  for (let mask = 0; mask < 256; mask++) {
    const x = Array.from({ length: 9 }, (_, i) => (mask >> (i % 8)) & 1);
    let X = 0, n1 = 0, n2 = 0;
    for (let i = 0; i < 8; i += 2) { X += !x[i] && (x[i + 1] || x[i + 2]) ? 1 : 0; n1 += x[i] || x[i + 1]; n2 += x[i + 1] || x[i + 2]; }
    count[mask] = x.slice(0, 8).reduce((a, b) => a + b, 0);
    const g = X === 1 && Math.min(n1, n2) >= 2 && Math.min(n1, n2) <= 3;
    thin[0][mask] = g && !((x[1] || x[2] || !x[7]) && x[0]) ? 1 : 0;
    thin[1][mask] = g && !((x[5] || x[6] || !x[3]) && x[4]) ? 1 : 0;
    const components = (value, conn) => {
      const seen = new Uint8Array(8); let n = 0;
      for (let i = 0; i < 8; i++) {
        if (seen[i] || x[i] !== value) continue;
        seen[i] = 1; const queue = [i]; let cardinal = i % 2 === 0;
        for (let q = 0; q < queue.length; q++) for (let j = 0; j < 8; j++) {
          if (seen[j] || x[j] !== value) continue;
          const a = MORPH_RING[queue[q]], b = MORPH_RING[j], dx = Math.abs(a[0] - b[0]), dy = Math.abs(a[1] - b[1]);
          if (conn === 8 ? Math.max(dx, dy) === 1 : dx + dy === 1) { seen[j] = 1; queue.push(j); cardinal ||= j % 2 === 0; }
        }
        if (value || cardinal) n++;
      }
      return n;
    };
    simple[mask] = components(1, 8) === 1 && components(0, 4) === 1 ? 1 : 0;
  }
  return { thin, simple, count };
}
export const MORPH_5A_LUT = morph5ALuts();
// Binary input 0/255; padding 0. Shrink snapshots each directional border, then
// checks/deletes in four parity subfields using the padded array coordinates.
// Each subfield deletes in parallel; its pixels are never 8-adjacent.
export function binaryMorph5A(data, w, h, op, iterations) {
  const W = w + 2, a = new Uint8Array(W * (h + 2));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) a[(y + 1) * W + x + 1] = data[y * w + x] > 127 ? 1 : 0;
  const off = [1, 1 - W, -W, -1 - W, -1, W - 1, W, W + 1];
  const maskAt = (i) => a[i + 1] | a[i + 1 - W] << 1 | a[i - W] << 2 | a[i - 1 - W] << 3 | a[i - 1] << 4 | a[i - 1 + W] << 5 | a[i + W] << 6 | a[i + 1 + W] << 7;
  const cap = op === "clean" ? 1 : iterations || (op === "spur" ? Math.max(w, h) : w * h);
  let rounds = 0, removed = 0, added = 0;
  for (; rounds < cap;) {
    rounds++; let changed = false;
    const steps = op === "thin" ? [0, 1] : op === "shrink" ? [2, 6, 0, 4] : [0];
    for (const step of steps) {
      if (op === "shrink") {
        const candidates = [[], [], [], []];
        for (let y = 1; y <= h; y++) for (let x = 1; x <= w; x++) {
          const i = y * W + x;
          if (a[i] && !a[i + off[step]]) candidates[(y & 1) * 2 + (x & 1)].push(i);
        }
        for (let k = 0; k < 4; k++) {
          const changes = [];
          for (const i of candidates[k]) {
            if (!a[i]) continue;
            const mask = maskAt(i);
            if (MORPH_5A_LUT.count[mask] >= 1 && MORPH_5A_LUT.simple[mask]) changes.push(i);
          }
          for (const i of changes) a[i] = 0;
          removed += changes.length;
          if (changes.length) changed = true;
        }
        continue;
      }
      const changes = [];
      for (let y = 1; y <= h; y++) for (let x = 1; x <= w; x++) {
        const i = y * W + x;
        if (!a[i] && op !== "majority") continue;
        const mask = maskAt(i), n = MORPH_5A_LUT.count[mask];
        const del = op === "clean" ? n === 0 : op === "spur" ? n === 1 : MORPH_5A_LUT.thin[step][mask];
        if (op === "majority" ? a[i] !== (n + a[i] >= 5 ? 1 : 0) : del) {
          changes.push(i);
        }
      }
      for (const i of changes) { const old = a[i]; a[i] = op === "majority" ? 1 - a[i] : 0; if (old) removed++; else added++; }
      if (changes.length) changed = true;
    }
    if (!changed) break;
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = a[(y + 1) * W + x + 1] * 255;
  return { out, rounds, removed, added };
}
function morphology5A(cv, src, p, ctx) {
  if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
  const op = p.op, h = intIn(p.hHeight, 10, 1, 255);
  const iterations = op === "majority" ? intIn(p.iterations, 1, 1, 5) : intIn(p.iterations, 1, 0, 100);
  let g = null, dst = null, mask = null, labels = null;
  try {
    g = grayIfNeeded(cv, src); const x = g || src, W = x.cols, H = x.rows;
    let out, info;
    if (op === "h_minima" || op === "extended_min") {
      const inv = Int32Array.from(x.data, (v) => 255 - v);
      out = op === "h_minima" ? Uint8Array.from(hMaximaImage(inv, W, H, h), (v) => 255 - v) : extendedMaxMask(inv, W, H, h);
      if (op === "h_minima") info = `H-minima · ถมหุบขึ้น ${h} ระดับ`;
      else {
        mask = cv.matFromArray(H, W, cv.CV_8UC1, out); labels = new cv.Mat();
        const n = cv.connectedComponents(mask, labels, 8, cv.CV_32S) - 1;
        info = `Extended Min (h=${h}) · พบ ${n} บริเวณ (${cv.countNonZero(mask)} พิกเซล)`;
      }
    } else {
      const r = binaryMorph5A(x.data, W, H, op, iterations); out = r.out;
      const name = { clean: "ลบจุดเดี่ยว", majority: "เสียงข้างมาก", spur: "ตัดกิ่งเส้น", thin: "ทำให้บาง", shrink: "หดเป็นจุด" }[op];
      info = `${name} · ลบ ${r.removed} พิกเซล` + (op === "majority" ? ` · เพิ่ม ${r.added} พิกเซล` : "") + ` · ${r.rounds} รอบ` + (iterations === 0 ? " (จนนิ่ง)" : "");
    }
    dst = cv.matFromArray(H, W, cv.CV_8UC1, out);
    if (ctx) ctx.info = info;
    const result = dst; dst = null; return result;
  } finally { for (const m of [dst, labels, mask, g]) if (m) m.delete(); }
}

const ECC_FIELDS = [["ecc_ok", "จัดภาพสำเร็จ (1 = ใช่)"], ["ecc_rho", "ความเข้ากัน ρ (0–1)"], ["ecc_dx", "เลื่อนแนวนอน (px)"], ["ecc_dy", "เลื่อนแนวตั้ง (px)"], ["ecc_angle_deg", "หมุน (°)"], ["ecc_scale", "ขยาย (เท่า)"]];
const ECC_MOTIONS = {
  translation: { th: "เลื่อนอย่างเดียว", code: "MOTION_TRANSLATION" },
  euclidean: { th: "เลื่อน + หมุน", code: "MOTION_EUCLIDEAN" },
  affine: { th: "เลื่อน + หมุน + ย่อขยาย + เฉือน", code: "MOTION_AFFINE" },
  homography: { th: "มุมมองเปลี่ยน (perspective)", code: "MOTION_HOMOGRAPHY" },
};
const ECC_VIEWS = { aligned: true, overlay: true, diff: true };
const GABOR_FIELDS = [["gabor_mean", "ผลตอบสนองเฉลี่ย"], ["gabor_max", "ผลตอบสนองสูงสุด"]];
const GABOR_OUTPUTS = { magnitude: true, real: true };
const GLCM_FIELDS = [["glcm_contrast", "Contrast (ความต่างของคู่)"], ["glcm_correlation", "Correlation (ความสัมพันธ์)"], ["glcm_energy", "Energy (ความสม่ำเสมอ)"], ["glcm_homogeneity", "Homogeneity (ความเรียบ)"], ["glcm_pairs", "จำนวนคู่ที่นับ"]];
const GLCM_LEVELS = { 2: true, 4: true, 8: true, 16: true, 32: true, 64: true };
const GLCM_ANGLES = { 0: true, 45: true, 90: true, 135: true, avg: true };
const GLCM_VIEWS = { matrix: true, quantized: true };
const NLM_PATCHES = { 3: true, 5: true, 7: true };
const NLM_SEARCHES = { 7: true, 11: true, 15: true };
const DEHAZE_FIELDS = [["dh_a_r", "แสงหมอก A ช่อง R"], ["dh_a_g", "แสงหมอก A ช่อง G"], ["dh_a_b", "แสงหมอก A ช่อง B"], ["dh_t_mean", "ความโปร่งใสเฉลี่ย t"]];
const DEHAZE_VIEWS = { result: true, transmission: true, dark: true };
const DCT_FIELDS = [["dct_psnr", "PSNR เทียบภาพเดิม (dB)"], ["dct_zero_pct", "สัมประสิทธิ์ที่เป็น 0 (%)"], ["dct_blocks", "จำนวนบล็อก 8×8"]];
const DCT_METHODS = { jpeg: true, zonal: true };
const DCT_VIEWS = { result: true, coeffs: true, error: true };
const MORPH_NEW_LABEL = { regional_max: "Regional Maxima", regional_min: "Regional Minima", h_maxima: "H-maxima", extended_max: "Extended Maxima" };

// แกน Wiener / Lucy (export ให้เทสต์ใช้ PSF ที่ไม่สมมาตรได้ — psfKernel สร้างแต่ PSF ที่สมมาตรจุดกลาง ซึ่ง OTF เป็นจำนวนจริง เทสต์ conj / การพลิก kernel จึงต้องใช้ PSF อื่น)
// x = Mat 8U 1 หรือ 3 ช่อง (ไม่มี alpha) · psf = { n (คี่), k: array n·n } · คืน Mat 8U ใหม่ขนาดเท่า x (ผู้เรียก delete)
export function wienerDeconvolve(cv, x, psf, nsr) {
  const n = psf.n, hc = (n - 1) / 2;
  let vec = null, phMat = null, Hm = null;
  try {
    const W = x.cols, H = x.rows, ch = x.channels();
    const Wp = cv.getOptimalDFTSize(W + 2 * n), Hp = cv.getOptimalDFTSize(H + 2 * n);
    // psf2otf: วาง PSF แล้วหมุนวนให้จุดกลางอยู่ที่ (0,0)
    const ph = new Float32Array(Hp * Wp);
    for (let y = 0; y < n; y++) for (let xx = 0; xx < n; xx++) ph[((y - hc + Hp) % Hp) * Wp + ((xx - hc + Wp) % Wp)] = psf.k[y * n + xx];
    phMat = cv.matFromArray(Hp, Wp, cv.CV_32FC1, ph);
    Hm = new cv.Mat();
    cv.dft(phMat, Hm, cv.DFT_COMPLEX_OUTPUT);
    const res = new Uint8Array(W * H * ch);
    if (ch === 1) {
      wienerChannel(cv, x, W, H, n, Wp, Hp, Hm, nsr, res, 0, 1);
    } else {
      vec = new cv.MatVector();
      cv.split(x, vec);
      for (let c = 0; c < ch; c++) {
        const m = vec.get(c);
        try { wienerChannel(cv, m, W, H, n, Wp, Hp, Hm, nsr, res, c, ch); } finally { m.delete(); }
      }
    }
    return cv.matFromArray(H, W, x.type(), res); // คืน Mat ใหม่ที่ผู้เรียกเป็นเจ้าของ (matFromArray เป็นขั้นสุดท้าย — ถ้า throw ก็ยังไม่มี Mat ของผลลัพธ์ให้ต้องลบ)
  } finally {
    if (Hm) Hm.delete();
    if (phMat) phMat.delete();
    if (vec) vec.delete();
  }
}
export function lucyDeconvolve(cv, x, psf, iters) {
  const n = psf.n;
  let vec = null, K = null, Kf = null;
  try {
    const W = x.cols, H = x.rows, ch = x.channels();
    K = cv.matFromArray(n, n, cv.CV_32FC1, Array.from(psf.k));
    Kf = new cv.Mat();
    cv.flip(K, Kf, -1);
    const res = new Uint8Array(W * H * ch);
    if (ch === 1) {
      lucyChannel(cv, x, K, Kf, iters, res, 0, 1);
    } else {
      vec = new cv.MatVector();
      cv.split(x, vec);
      for (let c = 0; c < ch; c++) {
        const m = vec.get(c);
        try { lucyChannel(cv, m, K, Kf, iters, res, c, ch); } finally { m.delete(); }
      }
    }
    return cv.matFromArray(H, W, x.type(), res);
  } finally {
    if (Kf) Kf.delete();
    if (K) K.delete();
    if (vec) vec.delete();
  }
}

// ภาพใหญ่กว่า 640×480 พิกเซล: Wiener / Lucy ไม่ย่อภาพ (ผลต้องเท่ากับภาพจริง) จึงแจ้งเตือนในข้อความแทน — Lucy 1080p สี 10 รอบ ≈ 7 วินาที, 50 รอบ ≈ 35 วินาที
function bigImageNote(src, iters) {
  if (src.cols * src.rows <= 640 * 480) return "";
  return ` · ภาพใหญ่ (${src.cols}×${src.rows} px) ช้า${iters > 0 ? ` — Lucy ${iters} รอบบนภาพขนาดนี้อาจใช้เวลาหลายวินาที` : ""} ควรย่อภาพ (Resize) ก่อน หรือใช้กับภาพนิ่ง`;
}

// Wiener ต่อ 1 ช่อง: ขอบสะท้อน → dft → G·H*/(|H|²+NSR) (double) → dft ย้อน → ตัดขอบ → ปัดครึ่งขึ้น · เขียนลง res (interleaved) ช่อง c
function wienerChannel(cv, m, W, H, n, Wp, Hp, Hm, nsr, res, c, ch) {
  let big = null, f = null, G = null, inv = null;
  try {
    big = new cv.Mat();
    cv.copyMakeBorder(m, big, n, Hp - H - n, n, Wp - W - n, cv.BORDER_REFLECT);
    f = new cv.Mat();
    big.convertTo(f, cv.CV_32F);
    G = new cv.Mat();
    cv.dft(f, G, cv.DFT_COMPLEX_OUTPUT);
    { // ไม่จองแมตใหม่ระหว่างถือ view (heap อาจขยายแล้ว view เก่าใช้ไม่ได้)
      const g = G.data32F, hd = Hm.data32F, len = Hp * Wp;
      for (let i = 0; i < len; i++) {
        const gr = g[2 * i], gi = g[2 * i + 1], hr = hd[2 * i], hi = hd[2 * i + 1];
        const den = hr * hr + hi * hi + nsr;
        if (den > 0) { g[2 * i] = (gr * hr + gi * hi) / den; g[2 * i + 1] = (gi * hr - gr * hi) / den; }
        else { g[2 * i] = 0; g[2 * i + 1] = 0; }
      }
    }
    inv = new cv.Mat();
    cv.dft(G, inv, cv.DFT_INVERSE + cv.DFT_SCALE + cv.DFT_REAL_OUTPUT);
    const d = inv.data32F;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = Math.floor(d[(y + n) * Wp + x + n] + 0.5);
        res[(y * W + x) * ch + c] = Number.isFinite(v) ? (v < 0 ? 0 : v > 255 ? 255 : v) : 0;
      }
    }
  } finally {
    for (const mm of [inv, G, f, big]) if (mm) mm.delete();
  }
}

// Lucy-Richardson ต่อ 1 ช่อง (float32 ทุกขั้น) · Mat c / r / corr จองนอกลูป ไม่ new ในลูป
function lucyChannel(cv, m, K, Kf, iters, res, c, ch) {
  let g = null, est = null, cc = null, r = null, corr = null;
  try {
    g = new cv.Mat();
    m.convertTo(g, cv.CV_32F);
    est = g.clone();
    cc = new cv.Mat(); corr = new cv.Mat();
    r = new cv.Mat(g.rows, g.cols, cv.CV_32FC1);
    const n = g.rows * g.cols, anchor = new cv.Point(-1, -1);
    for (let it = 0; it < iters; it++) {
      cv.filter2D(est, cc, -1, Kf, anchor, 0, cv.BORDER_REFLECT_101);
      { const gd = g.data32F, cd = cc.data32F, rd = r.data32F; for (let i = 0; i < n; i++) rd[i] = cd[i] > 1e-6 ? gd[i] / cd[i] : 0; }
      cv.filter2D(r, corr, -1, K, anchor, 0, cv.BORDER_REFLECT_101);
      { const ed = est.data32F, cr = corr.data32F; for (let i = 0; i < n; i++) ed[i] = ed[i] * cr[i]; }
    }
    const ed = est.data32F;
    for (let i = 0; i < n; i++) {
      const v = Math.floor(ed[i] + 0.5);
      res[i * ch + c] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  } finally {
    for (const mm of [corr, r, cc, est, g]) if (mm) mm.delete();
  }
}

// Morphology ตัวเลือกใหม่ (IP batch 4B) — Hit-or-Miss: bin ที่ 127 → MORPH_HITMISS ต่อ kernel (หมุน 4 ทิศถ้าเลือก) → OR สะสม
function hitMissOp(cv, src, p, ctx) {
  if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
  const pat = ownKey(HITMISS_PATTERNS, p.hmPattern, "isolated");
  const rotate = boolOr(p.hmRotate, true);
  let g = null, bin = null, acc = null;
  try {
    g = grayIfNeeded(cv, src); // worker ส่งภาพเทามาให้อยู่แล้ว (needs: gray) — กัน MORPH_HITMISS throw C++ ถ้าถูกเรียกตรงด้วยภาพสี
    bin = new cv.Mat();
    cv.threshold(g || src, bin, 127, 255, cv.THRESH_BINARY);
    acc = cv.Mat.zeros(bin.rows, bin.cols, cv.CV_8UC1);
    for (const flat of hitMissKernels(pat, rotate)) {
      let k = null, tmp = null;
      try {
        k = cv.matFromArray(3, 3, cv.CV_32SC1, flat);
        tmp = new cv.Mat();
        cv.morphologyEx(bin, tmp, cv.MORPH_HITMISS, k);
        cv.bitwise_or(acc, tmp, acc);
      } finally {
        if (tmp) tmp.delete();
        if (k) k.delete();
      }
    }
    if (ctx) ctx.info = `Hit-or-Miss (${HITMISS_PATTERNS[pat].th}${rotate ? " · 4 ทิศ" : ""}) พบ ${cv.countNonZero(acc)} ตำแหน่ง`;
    const out = acc; acc = null;
    return out;
  } finally {
    if (acc) acc.delete();
    if (bin) bin.delete();
    if (g) g.delete();
  }
}

// Morphology ตัวเลือกใหม่ (IP batch 4B) — Regional Max/Min, H-maxima, Extended Maxima บนภาพเทา (ไม่ตัด 127) ด้วย reconstructDilate แบบ O(N)
function regionalOp(cv, src, p, ctx) {
  if (isEmptyMat(src)) { if (ctx) ctx.info = "ภาพว่าง"; return src.clone(); }
  const op = p.op, h = intIn(p.hHeight, 10, 1, 255);
  let g = null, maskM = null, labels = null, dst = null;
  try {
    g = grayIfNeeded(cv, src);
    const s = g || src;
    const W = s.cols, H = s.rows, N = W * H;
    const I = new Int32Array(N);
    let flat = false;
    { const sd = s.data; let mn = 255, mx = 0; for (let i = 0; i < N; i++) { const v = sd[i]; I[i] = v; if (v < mn) mn = v; if (v > mx) mx = v; } flat = mn === mx; }
    let mask, grey = null;
    if (op === "regional_max") mask = regionalMaxMask(I, W, H);
    else if (op === "regional_min") { for (let i = 0; i < N; i++) I[i] = 255 - I[i]; mask = regionalMaxMask(I, W, H); }
    else if (op === "extended_max") mask = extendedMaxMask(I, W, H, h);
    else { grey = hMaximaImage(I, W, H, h); mask = null; }
    const count = (a) => { let c = 0; for (let i = 0; i < N; i++) if (a[i]) c++; return c; };
    const comps = (a) => {
      maskM = cv.matFromArray(H, W, cv.CV_8UC1, a);
      labels = new cv.Mat();
      const nl = cv.connectedComponents(maskM, labels, 8, cv.CV_32S);
      labels.delete(); labels = null; maskM.delete(); maskM = null;
      return nl - 1;
    };
    if (op === "h_maxima") {
      const o = new Uint8Array(N);
      for (let i = 0; i < N; i++) o[i] = grey[i];
      dst = cv.matFromArray(H, W, cv.CV_8UC1, o);
      if (ctx) ctx.info = `H-maxima · กดยอดลง ${h} ระดับ · ยอดที่เหลือ ${comps(regionalMaxMask(grey, W, H))} บริเวณ`;
    } else {
      dst = cv.matFromArray(H, W, cv.CV_8UC1, mask);
      if (ctx) ctx.info = `${MORPH_NEW_LABEL[op]}${op === "extended_max" ? ` (h=${h})` : ""} · พบ ${comps(mask)} บริเวณ (${count(mask)} พิกเซล)` + (flat ? " · ภาพมีค่าเดียว ทั้งภาพจึงเป็นยอดเดียว" : "");
    }
    const out = dst; dst = null;
    return out;
  } finally {
    for (const mm of [dst, labels, maskM, g]) if (mm) mm.delete();
  }
}

// ctx.values ของ farneback_optical_flow (ใช้ทั้งเฟรมแรกและเฟรมปกติ — ค่าเป็น px ของภาพจริง)
function farnebackValues(mean, max, pct, dx, dy, frame) {
  return { kind: "scalars", fields: [
    { key: "mean_motion_px", label: "ความเร็วเฉลี่ย (px/เฟรม)", value: mean },
    { key: "max_motion_px", label: "ความเร็วสูงสุด (px/เฟรม)", value: max },
    { key: "moving_pct", label: "พื้นที่ที่ขยับ (%)", value: pct },
    { key: "mean_dx_px", label: "ขยับแนวนอนเฉลี่ย (px)", value: dx },
    { key: "mean_dy_px", label: "ขยับแนวตั้งเฉลี่ย (px)", value: dy },
    { key: "frame", label: "เฟรม", value: frame },
  ] };
}

// ---- Morphology แยกเป็นเครื่องมือรายตัว (2026-10-18) — 1 operation = 1 เครื่องมือ (เหมือนหมวดอื่น) ----
// OPS.morphology (ด้านบน) ยังเป็นที่เก็บโค้ดประมวลผลเพียงที่เดียว: แต่ละตัวใหม่เรียก OPS.morphology.apply ด้วย op ตายตัว
// → พิกเซล + ctx.info เท่าเดิมทุกบิต · params ของแต่ละตัวคัดจาก OPS.morphology.params ตาม visibleWhen เดิม (ไม่เขียนซ้ำ)
// ตัวเก่า (FILTERS.morphology) ยังลงทะเบียนเป็น legacy ให้เทมเพลต/undo เก่า + migrateLegacyStep ย้ายมาตัวใหม่
const MORPH_SPLIT = [
  { op: "erode", name: "Erosion (กัดให้เล็กลง)",
    modeHint: "กัดขอบวัตถุขาวให้ผอมลง ลบจุดจิ๋วรบกวน",
    desc: "กัดขอบวัตถุขาวให้ผอมลง: ตัวแปรง (kernel) ไล่ทาบทั่วภาพ จุดไหนที่แปรงทาบแล้วมีพิกเซลดำปนอยู่ก็กลายเป็นดำ — ลบจุดขาวจิ๋วที่เป็น noise ได้ แต่วัตถุจริงก็ผอมลงด้วย ใช้กับภาพขาว-ดำหลัง threshold" },
  { op: "dilate", name: "Dilation (ขยายให้ใหญ่ขึ้น)",
    modeHint: "พอกขอบวัตถุขาวให้อ้วนขึ้น อุดรูโหว่เล็ก ๆ",
    desc: "พอกขอบวัตถุขาวให้อ้วนขึ้น: จุดไหนที่แปรงทาบแล้วเจอพิกเซลขาวแม้แต่จุดเดียวก็กลายเป็นขาว — อุดรูโหว่/รอยขาดเล็ก ๆ ได้ แต่วัตถุก็ใหญ่ขึ้นด้วย (ตรงข้ามกับ Erosion) ใช้กับภาพขาว-ดำหลัง threshold" },
  { op: "open", name: "Opening (ลบจุดเล็ก)",
    modeHint: "Erode แล้ว Dilate — ลบจุดรบกวนเล็กโดยไม่เปลี่ยนขนาดวัตถุใหญ่",
    desc: "Erode แล้ว Dilate ต่อกัน: จุดรบกวนเล็ก ๆ หายไปตั้งแต่ขั้นกัด ส่วนวัตถุใหญ่ที่รอดมาถูกพอกกลับเป็นขนาดเดิม — ลบ noise โดยไม่ทำให้วัตถุหลักผอมลง" },
  { op: "close", name: "Closing (อุดรู)",
    modeHint: "Dilate แล้ว Erode — อุดรูเล็ก ๆ ในตัววัตถุ",
    desc: "Dilate แล้ว Erode ต่อกัน: รูเล็ก ๆ/รอยขาดในตัววัตถุถูกอุดตั้งแต่ขั้นพอก แล้วกัดกลับให้ขนาดนอกเท่าเดิม — ใช้ซ่อมวัตถุที่แหว่งเป็นรูพรุน" },
  { op: "gradient", name: "Gradient (ขอบวัตถุ)",
    modeHint: "Dilate ลบ Erode — ดึงเฉพาะเส้นขอบของวัตถุ",
    desc: "ภาพที่ Dilate แล้ว ลบด้วยภาพที่ Erode แล้ว — เหลือเฉพาะแถบขอบของวัตถุ (ความหนาแถบขึ้นกับขนาด kernel) ใช้ดึงเส้นรอบวัตถุจากภาพขาว-ดำ" },
  { op: "tophat", name: "Top Hat",
    modeHint: "ภาพเดิม ลบ Opening — ดึงส่วนที่สว่างกว่ารอบข้าง",
    desc: "ภาพเดิม ลบ Opening — เหลือเฉพาะส่วนที่สว่างกว่ารอบข้างและเล็กกว่า kernel (เช่น จุดสว่างบนพื้นไม่เรียบ) มักใช้ปรับแสงพื้นหลังก่อน threshold" },
  { op: "blackhat", name: "Black Hat",
    modeHint: "Closing ลบภาพเดิม — ดึงส่วนที่มืดกว่ารอบข้าง",
    desc: "Closing ลบภาพเดิม — เหลือเฉพาะส่วนที่มืดกว่ารอบข้างและเล็กกว่า kernel (เช่น รอยขีดข่วนสีเข้ม ตัวหนังสือบนพื้นสว่าง)" },
  { op: "fill_holes", name: "Fill Holes (อุดรูในวัตถุ)",
    modeHint: "เติมรูปิดสนิทภายในวัตถุขาว โดยไม่กระทบขนาดรอบนอก",
    desc: "เติมรูปิดสนิทภายในวัตถุขาว โดยไม่กระทบขนาดรอบนอก (รูที่ต่อกับขอบภาพไม่ถูกเติม) — ภาพที่ไม่ใช่ขาว-ดำล้วนจะถูก binarize ที่ค่ากึ่งกลาง 127 ก่อนเสมอ ไม่มีพารามิเตอร์ให้ปรับ" },
  { op: "clear_border", name: "Clear Border (ลบวัตถุติดขอบ)",
    modeHint: "ลบวัตถุขาวที่มีพิกเซลติดขอบภาพ เหลือแต่วัตถุที่อยู่เต็มตัวในเฟรม",
    desc: "ลบวัตถุขาวที่มีพิกเซลแม้แต่จุดเดียวติดขอบภาพทั้ง 4 ด้าน — เหลือแต่วัตถุที่อยู่เต็มตัวในเฟรม (วัตถุที่ถูกตัดครึ่งโดยขอบภาพวัดขนาดจริงไม่ได้ จึงมักตัดทิ้งก่อนนับ/วัด) ภาพที่ไม่ใช่ขาว-ดำล้วนจะถูก binarize ที่ 127 ก่อน ไม่มีพารามิเตอร์ให้ปรับ" },
  { op: "remove_small", name: "Remove Small (ลบวัตถุเล็ก)",
    modeHint: "ลบวัตถุขาวที่มีพื้นที่น้อยกว่าเกณฑ์ทิ้ง",
    desc: "ลบวัตถุขาวที่มีพื้นที่น้อยกว่าเกณฑ์ 'พื้นที่ขั้นต่ำ' ทิ้ง — ต่างจาก Opening ตรงที่ไม่ทำให้วัตถุที่เหลือเปลี่ยนรูป ใช้ล้างจุด noise ก่อนนับวัตถุ" },
  { op: "clean", name: "Clean (ลบจุดเดี่ยว)",
    modeHint: "ลบจุดขาวที่ไม่มีเพื่อนบ้านขาว 8 ทิศ ทำหนึ่งรอบ",
    desc: "ลบเฉพาะจุดเดี่ยว โดยไม่ลบปลายเส้นที่มีเพื่อนบ้าน · ใช้ค่า > 127 เป็นวัตถุ นอกภาพเป็นดำ" },
  { op: "majority", name: "Majority (เสียงข้างมาก)",
    modeHint: "ขาวเมื่อในหน้าต่าง 3×3 มีขาวอย่างน้อย 5 จุด",
    desc: "ลดจุดรบกวนด้วยเสียงข้างมากใน 3×3 รวมตัวกลาง · ทำซ้ำตามจำนวนรอบ หยุดก่อนเมื่อนิ่ง นอกภาพเป็นดำ" },
  { op: "skeleton", name: "Skeleton (เส้นแกนกลาง)",
    modeHint: "ลดวัตถุขาวให้เหลือเส้นแกนกลางหนา 1 พิกเซล",
    desc: "ลดวัตถุขาวให้เหลือเส้นแกนกลางหนา 1 พิกเซล (Zhang-Suen) ที่ความละเอียดเต็ม — ช้ากว่า operation อื่นในหมวดนี้ (วัตถุยิ่งหนายิ่งช้า ภาพ 1080p ที่มีก้อนใหญ่ราว 0.3 วินาที เหมาะกับภาพนิ่งมากกว่ากล้องสด) ไม่มีพารามิเตอร์ให้ปรับ · ถ้าเส้นแกนกลางไปวิ่งอยู่ในพื้นหลังแทนที่จะอยู่ในวัตถุ แปลว่าพื้นหลังเป็นสีขาว ให้สลับขาว-ดำ (Invert) มาก่อน เพราะ op นี้คิดว่าพิกเซลขาวคือ 'วัตถุ' เสมอ (นอกขอบภาพนับเป็นพื้นหลัง)" },
  { op: "spur", name: "Spur (ตัดกิ่งเส้น)",
    modeHint: "ลบจุดปลายที่มีเพื่อนบ้านขาวพอดีหนึ่งจุด",
    desc: "ตัดปลายเส้นพร้อมกันทั้งภาพต่อรอบ · จุดเดี่ยวไม่ถูกลบ · 0 = จนนิ่ง ไม่เกินด้านยาวของภาพ ใช้ค่า > 127" },
  { op: "thin", name: "Thin (ทำให้บาง)",
    modeHint: SLOW_MODE_HINT + " · Lam–Lee–Suen · 0 = จนนิ่ง",
    desc: "ทำวัตถุขาวให้บางด้วย Lam–Lee–Suen สองขั้นย่อย ต่างจาก Skeleton ที่ใช้ Zhang-Suen · รักษาการเชื่อมต่อและปลายเส้นบาง · ภาพก้อนหนาอาจช้า ควรใช้ภาพนิ่ง" },
  { op: "shrink", name: "Shrink (หดเป็นจุด)",
    modeHint: SLOW_MODE_HINT + " · รักษาจำนวนวัตถุและรู · 0 = จนนิ่ง",
    desc: "หดขอบ N/S/E/W ด้วย simple point แบบ 8/4 · ลบพร้อมกันในแต่ละกลุ่มพิกเซลสลับ 2×2 เพื่อรักษา Euler · วัตถุไม่มีรูเหลือจุด วัตถุมีรูเหลือวงรอบรู · ภาพก้อนหนาอาจช้า ควรใช้ภาพนิ่ง" },
  { op: "reconstruct_open", name: "Recon. Opening (คงรูปร่าง)",
    modeHint: "เหมือน Opening แต่รักษาขอบวัตถุที่รอดไว้แม่นกว่า (ไม่มุมมน)",
    desc: "เหมือน Opening แต่รักษาขอบวัตถุที่รอดไว้แม่นกว่า (ไม่มุมมน): กัดภาพเป็น marker แล้วคืนรูปร่างเดิมให้ทุกวัตถุที่ยังเหลือ marker อยู่ — ขนาด kernel/จำนวนรอบ คือขนาดกัดตอนสร้าง marker (ใช้ kernel สี่เหลี่ยมเสมอ จึงไม่มีตัวเลือกรูปร่าง)" },
  { op: "convex_hull", name: "Convex Hull (ห่อด้วยรูปนูน)",
    modeHint: "ห่อแต่ละวัตถุด้วยรูปนูนที่เล็กที่สุดที่ครอบได้พอดี + ค่า solidity",
    desc: "ห่อแต่ละวัตถุด้วยรูปนูนที่เล็กที่สุดที่ครอบวัตถุไว้ได้พอดี (เหมือนรัดหนังยางรอบวัตถุ) แล้วเทียบพื้นที่เดิมกับพื้นที่รูปนูนเป็นค่า solidity (รวมทุกชิ้น ชิ้นใหญ่มีน้ำหนักมากกว่า) เส้นบาง/จุดเดี่ยวก็ถูกห่อและนับเป็นหนึ่งชิ้นด้วย — ใช้หารอยบิ่น/รอยเว้าของวัตถุ (solidity ยิ่งต่ำ ยิ่งเว้าแหว่งมาก) ไม่มีพารามิเตอร์ให้ปรับ" },
  { op: "endpoints_branchpoints", name: "Endpoints (จุดปลาย/จุดแยก)",
    modeHint: "ใช้ต่อจาก Skeleton เสมอ — วงเขียว = จุดปลาย, วงแดง = จุดแยก",
    desc: "⚠️ ใช้ต่อจาก Skeleton เสมอ — ถ้าวัตถุยังหนาอยู่ (ไม่ใช่เส้นบาง 1 พิกเซล) การนับจะไม่มีความหมาย นับเพื่อนบ้านสีขาว 8 ทิศรอบแต่ละพิกเซลเส้น: มีเพื่อนบ้าน 1 จุด = จุดปลาย (วาดวงเขียว), ส่วนจุดแยก (วาดวงแดง) ใช้ Crossing Number (นับจำนวนช่วงที่ขาว-ดำสลับกันรอบวงแหวนเพื่อนบ้าน หาร 2) ≥3 — ไม่ใช่แค่นับจำนวนเพื่อนบ้านตรง ๆ เพราะพิกเซลข้างจุดตัดจริงก็มีเพื่อนบ้าน ≥3 ได้เหมือนกันแต่ไม่ใช่จุดแยกจริง เป็นค่าประมาณ ไม่ใช่ lookup table แบบที่ MATLAB ใช้จริง ใช้นับจำนวนกิ่ง/ปลายของรอยร้าวหรือเส้นลายเส้นได้ ไม่มีพารามิเตอร์ให้ปรับ" },
  { op: "hit_or_miss", name: "Hit-or-Miss (หาลายแม่แบบ)",
    modeHint: "หาตำแหน่งที่ลายรอบพิกเซลตรงแม่แบบ: บางช่องต้องขาว บางช่องต้องดำ",
    desc: "หาตำแหน่งที่ลายรอบพิกเซลตรงแม่แบบ 3×3: บางช่องต้องขาว บางช่องต้องดำ บางช่องไม่สน — ทำโดย erode ภาพขาวด้วยช่อง 'ต้องขาว' แล้ว AND กับ erode ภาพกลับด้านด้วยช่อง 'ต้องดำ' · ภาพที่ไม่ใช่ขาว-ดำล้วนจะถูกตัดที่ 127 ก่อน · นอกขอบภาพนับเป็น 'ไม่สน' (ตามเอกสาร bwhitmiss ของ MATLAB สร้างจาก imerode ซึ่งถือว่านอกภาพเป็น +∞ จึงน่าจะนับแบบเดียวกัน แต่ยังไม่ได้ทดสอบกับ MATLAB จริง)  · กรณีหาปลายเส้นทั่วไป Endpoints & Branchpoints สะดวกกว่า" },
  { op: "regional_max", name: "Regional Max (ยอดเฉพาะที่)",
    modeHint: "บริเวณที่สว่างกว่าทุกพิกเซลรอบตัว (ใช้ภาพเทา)",
    desc: "บริเวณที่สว่างกว่าทุกพิกเซลรอบตัว (ใช้ภาพเทา ไม่ตัด 127) — ผลเป็นภาพขาว-ดำ ใช้หา 'ยอด' ในภาพเทาหรือภาพ Distance Transform เพื่อทำ marker แยกวัตถุที่ติดกัน · noise ทำให้เกิดยอดเล็กจำนวนมาก ลอง blur ก่อน หรือใช้ H-maxima / Extended Maxima · นับแบบ 8-connected เหมือนค่าเริ่มต้นของ MATLAB ไม่มีพารามิเตอร์ให้ปรับ" },
  { op: "regional_min", name: "Regional Min (หุบเฉพาะที่)",
    modeHint: "บริเวณที่มืดกว่าทุกพิกเซลรอบตัว (ใช้ภาพเทา)",
    desc: "บริเวณที่มืดกว่าทุกพิกเซลรอบตัว (ใช้ภาพเทา ไม่ตัด 127) — ผลเป็นภาพขาว-ดำ ใช้หา 'หุบ' ในภาพเทา · noise ทำให้เกิดหุบเล็กจำนวนมาก ลอง blur ก่อน · นับแบบ 8-connected ไม่มีพารามิเตอร์ให้ปรับ" },
  { op: "h_maxima", name: "H-maxima (ตัดยอดต่ำกว่า h)",
    modeHint: "ผลเป็นภาพเทา: ยอดทุกยอดถูกกดลง h ระดับ ยอดที่สูงไม่ถึง h หายไป",
    desc: "ผลเป็นภาพเทา: ยอดทุกยอดถูกกดลง h ระดับ ยอดที่สูงไม่ถึง h หายไป — ใช้ค่า 'ความสูง h' กรอง noise เล็ก ๆ ออกจากยอด · นับแบบ 8-connected" },
  { op: "h_minima", name: "H-minima (ถมหุบตื้นกว่า h)",
    modeHint: "ถมหุบตื้นในภาพเทา ไม่ตัดที่ 127",
    desc: "กลับสี → H-maxima → กลับสี · ถมหุบตื้นกว่า h ด้วย reconstruction แบบ 8 ทิศ ผลเป็นภาพเทา" },
  { op: "extended_max", name: "Extended Max (ยอดสูง ≥ h)",
    modeHint: "= Regional Maxima ของ H-maxima · ใช้ทำ marker ให้ Watershed",
    desc: "= Regional Maxima ของ H-maxima: เก็บเฉพาะยอดที่สูงกว่ารอบข้างอย่างน้อย h ระดับ ผลเป็นภาพขาว-ดำ ใช้ทำ marker ให้ Watershed แยกวัตถุที่ติดกัน — เพิ่ม h เมื่อภาพมี noise มาก · นับแบบ 8-connected" },
  { op: "extended_min", name: "Extended Min (หุบลึก ≥ h)",
    modeHint: "หาตำแหน่งหุบหลังถมหุบตื้นด้วย h",
    desc: "Regional Max ของ H-maxima บนภาพกลับสี · ใช้ภาพเทา ไม่ตัดที่ 127 ผลเป็น 0/255 ตรงหุบ แบบ 8 ทิศ" },
];
export const MORPH_OP_IDS = MORPH_SPLIT.map((m) => "morph_" + m.op);
// ข้อความ param ของตัวใหม่ (ตัดวลี "ใช้กับ X เท่านั้น/ไม่มีผลกับ Y" ที่ใช้แยกตัวในแผงเดิมออก เพราะตอนนี้ param แสดงเฉพาะตัวที่ใช้จริง)
const MORPH_PARAM_TWEAK = {
  shape: { label: "รูปร่าง kernel" },
  minArea: { label: "พื้นที่ขั้นต่ำ (px)", desc: "วัตถุขาวที่มีพื้นที่น้อยกว่านี้ (พิกเซล) จะถูกลบทิ้ง" },
  lineAngle: { label: "มุมของเส้น (องศา)", desc: "0° = แนวนอน, 90° = แนวตั้ง" },
  hmPattern: { label: "แม่แบบ Hit-or-Miss" },
  hmRotate: { label: "รวมทั้ง 4 ทิศ (หมุนทีละ 90°)" },
  iterations: { desc: "ทำ operation ซ้ำกี่รอบ — มาก = ผลยิ่งเข้มขึ้น" },
  hHeight: { label: "ความสูง h", desc: "ยอดที่สูงกว่ารอบข้างไม่ถึง h ระดับ (0–255) จะถูกตัดทิ้ง — เพิ่ม h เมื่อภาพมี noise มาก" },
};
// reconstruct_open ใช้ ksize/iterations เหมือนกัน แต่ความหมายคือขนาด/จำนวนรอบกัดตอนสร้าง marker → label ต่างจากตัวอื่น
const MORPH_PARAM_TWEAK_BY_OP = {
  spur: { iterations: { min: 0, max: 100, default: 1, hint: "0 = จนนิ่ง", desc: "ทำซ้ำตามจำนวนรอบ · 0 = ทำจนภาพไม่เปลี่ยน" } },
  thin: { iterations: { min: 0, max: 100, default: 0, hint: "0 = จนนิ่ง", desc: "ทำซ้ำตามจำนวนรอบ · 0 = ทำจนภาพไม่เปลี่ยน" } },
  shrink: { iterations: { min: 0, max: 100, default: 0, hint: "0 = จนนิ่ง", desc: "ทำซ้ำตามจำนวนรอบ · 0 = ทำจนภาพไม่เปลี่ยน" } },
  h_minima: { hHeight: { label: "ความลึก h", desc: "ถมหุบตื้นกว่า h ระดับ (0–255)" } },
  extended_min: { hHeight: { label: "ความลึก h", desc: "หาตำแหน่งหุบหลังถมหุบตื้นกว่า h ระดับ" } },
  reconstruct_open: {
    ksize: { label: "ขนาด kernel (กัด marker)", desc: "ขนาดตัวแปรงที่ใช้กัดตอนสร้าง marker — ใหญ่ = วัตถุเล็กถูกตัดทิ้งมากขึ้น" },
    iterations: { label: "จำนวนรอบกัด marker", desc: "จำนวนรอบกัดตอนสร้าง marker — มาก = วัตถุบาง/เล็กยิ่งถูกตัดทิ้ง" },
  },
};
for (const m of MORPH_SPLIT) {
  const id = "morph_" + m.op;
  const params = [];
  for (const q of OPS.morphology.params) {
    if (q.key === "op") continue;
    const own = (q.visibleWhen || []).find((c) => c.key === "op");
    if (!own || !own.in.includes(m.op)) continue; // param นี้ไม่ใช่ของ op นี้
    const rest = q.visibleWhen.filter((c) => c !== own); // เหลือเงื่อนไขภายใน (เช่น lineAngle ต้อง shape=line)
    const { visibleWhen, ...base } = q;
    params.push({ ...base, ...MORPH_PARAM_TWEAK[q.key], ...((MORPH_PARAM_TWEAK_BY_OP[m.op] || {})[q.key]), ...(rest.length ? { visibleWhen: rest } : {}) });
  }
  OPS[id] = {
    name: m.name,
    modeHint: m.modeHint,
    needs: "gray",
    desc: m.desc,
    params,
    // เรียกตัวเดิมด้วย op ตายตัว (ไม่ copy โค้ดประมวลผล) — ค่า param อื่นผ่านตรง ๆ
    apply(cv, src, p, ctx) {
      return OPS.morphology.apply(cv, src, { ...p, op: m.op }, ctx);
    },
  };
}

// ===================================================================
// รวม op หลายตัวในกลุ่มเดียว → node เดียว (มี select "เลือกวิธี")
// ===================================================================
const MODE_KEY = "mode"; // key ของ select ที่ใช้เลือก op

// param นี้ "แสดงอยู่ไหม" ตามค่าปัจจุบัน — ที่เดียวที่ detail-panel (render/renderSteps) และ app.js (stepRawParams) เรียกใช้
// 1) showIf = เงื่อนไข "วิธี" ชั้นนอกของ mergeGroup (เดิม ไม่แก้)  2) visibleWhen = array (AND) เงื่อนไขจาก select/boolean ภายใน op
//    รูปแบบ { key, in:[...] } | { key, notIn:[...] } · ตัวคุมต้องเป็น select/boolean (re-render เมื่อเปลี่ยน) ห้ามเป็นสไลเดอร์ตัวเลข
// ค่าตัวคุมที่แปลก/ไม่มี → ใช้ default ของตัวคุม (ให้ตรงกับที่ apply() ตกไปใช้) · ซ่อนแค่การแสดงผล ไม่แตะค่าที่เก็บไว้
export function isParamVisible(param, params, allParams) {
  if (param.showIf && params[param.showIf.key] !== param.showIf.value) return false;
  for (const c of param.visibleWhen || []) {
    const ctrl = allParams.find((q) => q.key === c.key);
    let v = params[c.key];
    if (ctrl && ctrl.type === "select" && !ctrl.options.some((o) => o.value === v)) v = ctrl.default;
    if (ctrl && ctrl.type === "boolean" && typeof v !== "boolean") v = ctrl.default;
    if (c.in && !c.in.includes(v)) return false;
    if (c.notIn && c.notIn.includes(v)) return false;
  }
  return true;
}

// จัดลำดับ param ที่แสดง: ตัวคุม (ถูกอ้างเป็น visibleWhen.key) ต้องมาก่อน param ที่มันคุม — ไม่งั้นพอ param ที่คุมหายไป
// select จะเลื่อนขึ้นมาอยู่ใต้เมาส์ (เช่น gamma: g มาก่อน curve) · ลำดับอื่นคงเดิม (ดึงเฉพาะตัวคุมที่ตามหลังขึ้นมา)
export function orderParamsControllersFirst(list) {
  const out = [];
  const placed = new Set();
  const place = (q) => {
    if (placed.has(q)) return;
    placed.add(q); // เพิ่มก่อน recurse — กันวนถ้ามีอ้างกันเป็นวง
    for (const c of q.visibleWhen || []) {
      const ctrl = list.find((x) => x.key === c.key);
      if (ctrl) place(ctrl);
    }
    out.push(q);
  };
  for (const q of list) place(q);
  return out;
}

// สร้าง merged filter จากรายชื่อ op
// opts (ใหม่, Tier 2 §A.1) — เช่น { inputs: 2 } — spread ทับผลลัพธ์สุดท้าย เพิ่ม field แบบ additive
// ล้วน ๆ (ไม่แตะ field เดิมใด ๆ ถ้าไม่ได้ส่ง opts มา) — ทุก FILTERS entry เดิมไม่ได้รับผลกระทบ
function mergeGroup(name, group, opIds, defaultMode, legacyOpIds = [], opts = {}) {
  // param แรก = select เลือกวิธี (controls = สั่งให้ UI re-render เมื่อเปลี่ยน)
  const params = [
    {
      key: MODE_KEY,
      type: "select",
      label: "เลือกวิธี",
      controls: true,
      options: opIds.map((id) => ({ value: id, label: OPS[id].name, hint: OPS[id].modeHint })),
      default: defaultMode,
    },
  ];
  // sub-params ของแต่ละ op — prefix ด้วย opId__ กัน key ชนกัน + showIf โชว์เฉพาะวิธีที่เลือก
  // legacyOpIds เก็บพารามิเตอร์ของโปรเจกต์เก่าไว้ แต่ไม่แสดงเป็นตัวเลือกใหม่ใน UI
  for (const id of [...opIds, ...legacyOpIds]) {
    for (const p of OPS[id].params) {
      // visibleWhen (ซ่อน param ตาม select ภายใน op) — เติม prefix ให้ key ของเงื่อนไขด้วย ไม่งั้นอ้างตัวคุมผิด key
      // (ทำกับทุก op เหมือนกันหมด: imfuse/flip มี param ภายในชื่อ "mode" ซึ่งไม่ชนกับ MODE_KEY ตัวนอกเพราะมี prefix)
      params.push({
        ...p,
        key: id + "__" + p.key,
        showIf: { key: MODE_KEY, value: id },
        ...(p.visibleWhen ? { visibleWhen: p.visibleWhen.map((c) => ({ ...c, key: id + "__" + c.key })) } : {}),
      });
    }
  }

  return {
    name,
    group,
    params,
    // needs ขึ้นกับวิธีที่เลือก ณ รันไทม์ (worker เรียกด้วย params ปัจจุบัน)
    needs: (p) => (OPS[p && p[MODE_KEY]] || OPS[defaultMode]).needs,
    // desc/subtitle ตามวิธีที่เลือก
    desc: (p) => (OPS[p && p[MODE_KEY]] || OPS[defaultMode]).desc,
    subtitle: (p) => (OPS[p && p[MODE_KEY]] || OPS[defaultMode]).name,
    apply(cv, src, p, ctx) {
      const modeId = (p && p[MODE_KEY]) || defaultMode;
      const op = OPS[modeId] || OPS[defaultMode];
      // ดึง sub-params ของ op ที่เลือก แล้วตัด prefix ออกกลับเป็น key เดิม
      const prefix = modeId + "__";
      const local = {};
      for (const k in p) {
        if (k.startsWith(prefix)) local[k.slice(prefix.length)] = p[k];
      }
      return op.apply(cv, src, local, ctx);
    },
    ...opts, // spread ท้ายสุด — เพิ่ม inputs:2 (หรือไม่มีอะไรเลย) โดยไม่ทับ field ด้านบนที่ตั้งใจไว้แล้ว
  };
}

// FILTERS = node ที่แสดงบน palette จริง (1 กลุ่ม = 1 node)
export const FILTERS = {
  color_space: mergeGroup("Color Space", "Color Space",
    ["grayscale", "colorspace", "channel_split", "rgb_adjust", "color_inrange", "invert"], "grayscale"),

  image_adjust: mergeGroup("Image Adjust", "Image Adjust",
    // contrast_stretch/flatfield_correction เพิ่ม 2026-10-01 (IP batch, หมวด ENHANCE) — ต่อท้ายลิสต์เดิม
    // เพื่อไม่ให้ลำดับ/mode ของ 4 ตัวเดิมขยับ (ไฟล์โปรเจกต์เก่าที่บันทึก mode ไว้ยังใช้ได้เหมือนเดิม)
    ["brightness_contrast", "gamma", "hist_equalize", "clahe", "contrast_stretch", "flatfield_correction"], "brightness_contrast"),

  sharpen: mergeGroup("Sharpen", "Sharpen",
    ["unsharp", "laplacian_sharpen", "custom_kernel"], "unsharp"),

  // IP batch 2 ชิ้น C (2026-10-03) ต่อท้าย gaussian_noise / wiener_filter / translate_shear ในสามกลุ่มนี้ — ห้ามสลับลำดับ/default เดิม
  // (ไฟล์โปรเจกต์เก่าที่บันทึก mode ไว้ต้องเปิดได้เหมือนเดิม)
  noise: mergeGroup("Noise", "Noise",
    ["salt_pepper_noise", "gaussian_noise"], "salt_pepper_noise"),

  blur: mergeGroup("Blur", "Blur",
    ["median_blur", "gaussian_blur", "box_blur", "bilateral", "wiener_filter"], "gaussian_blur"),

  // legacy (2026-10-18): morphology แยกเป็น 19 เครื่องมือแล้ว (morph_tools ด้านล่าง) — ตัวเดิมที่มี select "Operation" คงไว้ให้เทมเพลต/undo เก่า
  // + migrateLegacyStep ย้ายมา morph_tools · ไม่อยู่บน palette
  morphology: { ...OPS.morphology, group: "Morphology" },

  threshold: mergeGroup("Threshold", "Threshold",
    ["binary_threshold", "otsu_threshold", "adaptive_threshold", "multi_otsu_threshold"], "binary_threshold"),

  edge_detection: mergeGroup("Edge Detection", "Edge Detection",
    ["canny", "sobel", "laplacian_edge", "scharr_edge", "prewitt_edge", "roberts_edge", "log_edge"], "canny"),

  // IP batch 2 ชิ้น D (2026-10-03) ต่อท้าย feret / line_profile — ห้ามสลับลำดับ/default เดิม (ไฟล์โปรเจกต์เก่าที่บันทึก mode ไว้ต้องเปิดได้เหมือนเดิม)
  shape_analysis: mergeGroup("Shape Analysis", "Shape Analysis",
    ["find_contours", "connected_components", "distance_transform", "local_std_filter", "local_entropy_filter", "regionprops", "feret", "line_profile"], "find_contours", ["watershed"]),

  // Watershed เป็น segmentation algorithm แยกวัตถุที่แตะกัน ไม่ใช่การนับ component
  watershed: { ...OPS.watershed, group: "Segmentation" },

  // GrabCut ก็เป็น segmentation algorithm เช่นกัน (ไม่ใช่ shape analysis) — ตามแนวเดียวกับ Watershed
  grabcut: { ...OPS.grabcut, group: "Segmentation" },

  // Region Growing (2026-09-30) — เป็น segmentation algorithm ตัวที่ 3 ของกลุ่ม ตามแนวเดียวกับ
  // watershed/grabcut (op เดียว ไม่มี mode select) group เดิม "Segmentation" ไม่ต้องเพิ่มสี/ลำดับใหม่
  region_growing: { ...OPS.region_growing, group: "Segmentation" },

  object_detection: mergeGroup("Object Detection", "Object Detection",
    ["bounding_boxes", "hough_circles", "hough_lines", "harris_corners", "shi_tomasi_corners", "orb_keypoints", "akaze_keypoints", "fast_keypoints"], "bounding_boxes"),

  // IP batch 2 ชิ้น E (2026-10-04) ต่อท้าย slic (colour) / bandpass_notch (frequency) — ห้ามสลับลำดับ/default เดิม
  colour: mergeGroup("Colour", "Colour",
    ["color_inrange", "kmeans_segment", "white_balance", "invert", "slic"], "color_inrange"),

  // IP batch 2 ชิ้น D เพิ่ม roi_polygon · 2026-10-07: เรียง dropdown ให้ตรงกับ palette (roi_polygon ต่อจาก crop) —
  // mode เก็บเป็นชื่อ op (ไม่ใช่ index) โปรเจกต์/เทมเพลตเก่าจึงไม่กระทบ · default "resize" เหมือนเดิม
  geometry: mergeGroup("Geometry", "Geometry",
    ["crop", "roi_polygon", "resize", "image_pyramid", "rotate", "flip", "perspective_warp", "undistort", "translate_shear"], "resize"),

  frequency: mergeGroup("Frequency Domain", "Frequency Domain",
    ["fft_magnitude", "fft_filter", "bandpass_notch", "dct_jpeg"], "fft_magnitude"),

  // Image Arithmetic — กลุ่มใหม่ (Tier 2, 18a) ป้อนภาพ 2 ใบ (inputs:2 → FilterNode เพิ่มช่องอินพุต
  // "เข้า B" ให้อัตโนมัติ) กลุ่มนี้แยกจาก group อื่นเสมอเพราะ input contract ต่างกันจริง (ตามแนวเดียวกับ
  // "Geometry"/"Frequency Domain" ที่แยกกลุ่มใหม่ตอน contract ต่างจากกลุ่มเดิม)
  // 18b เพิ่ม imfuse/mask เข้ากลุ่มเดิม (opIds list) — group/inputs:2 เดิมจาก 18a ใช้ต่อได้เลย ไม่ต้องแก้
  // IP batch 2 ชิ้น B (2026-10-02) ต่อท้าย imarith/logical_ops/inpaint/hist_match — ห้ามสลับลำดับ 3 ตัวเดิม
  // (ไฟล์โปรเจกต์เก่าที่บันทึก mode ไว้ต้องเปิดได้เหมือนเดิม)
  // 2026-10-24 ecc_register ย้ายไปหมวด Image Registration — คงไว้เป็น legacyOpIds ให้ undo/โปรเจกต์เก่ายังรันได้ (migrateLegacyStep ย้ายเทมเพลตไป image_registration)
  image_arithmetic: mergeGroup("Image Arithmetic", "Image Arithmetic",
    ["imabsdiff", "imfuse", "mask", "imarith", "logical_ops", "inpaint", "hist_match", "roi_apply"], "imabsdiff", ["ecc_register"], { inputs: 2 }),

  // Template Matching (Tier 2, 18b) — กลุ่มเดี่ยวใหม่ inputs:2 (ตามแนวเดียวกับ morphology/watershed/
  // grabcut ที่มี op เดียวไม่ต้องมี mode select wrapper) primary = ภาพค้นหา, src2 = template
  template_matching: { ...OPS.template_matching, group: "Template Matching", inputs: 2 },

  // Image Quality (Tier 2, 18b) — กลุ่มใหม่ inputs:2 มี 2 op แยกเป็นคนละ node กันเลย (ตามแนวเดียวกับ
  // Segmentation ที่มี watershed/grabcut เป็นคนละ FILTERS entry แต่ใช้ group display name เดียวกัน — ไม่ใช้
  // mode-select node เดียว เพราะ PSNR/SSIM เป็นเมตริกที่ต่างกันจริง ไม่ใช่ "วิธีเดียวกันหลายโหมด")
  psnr: { ...OPS.psnr, group: "Image Quality", inputs: 2 },
  ssim: { ...OPS.ssim, group: "Image Quality", inputs: 2 },

  // Motion (Tier 2, 18d) — op ที่ "จำ" ข้อมูลข้ามเฟรม (ctx.state) — stateful:true ให้ runPipeline กันการรันซ้ำ
  // ในเฟรมเดียวกัน (กล่องผลลัพธ์ + พรีวิวของ node ที่เลือก) ดู §C.4-final 0-C · inputs คง undefined (ภาพเดียว)
  // หนึ่ง node ต่อหนึ่งอัลกอริทึม (แนวเดียวกับ psnr/ssim) ไม่มี mode select
  mog2: { ...OPS.mog2, group: "Motion", stateful: true },
  farneback_optical_flow: { ...OPS.farneback_optical_flow, group: "Motion", stateful: true },
  pyrlk_optical_flow: { ...OPS.pyrlk_optical_flow, group: "Motion", stateful: true },

  // ---- กลุ่มระดับ "หมวด palette" (2026-10-07) — 1 หมวดบน palette = 1 node ----
  // op list ต้องเท่ากับ TOOL_SECTIONS ของหมวดนั้น "ตามลำดับ" (palette-panel.js) · default = op แรกของหมวด
  // tests/category-merge.test.mjs ล็อกให้ตรงกัน — เพิ่ม op ใหม่ในหมวดแล้วต้องมาเพิ่มที่นี่ด้วย ไม่งั้นเทสต์ล้ม
  // กลุ่มเก่าด้านบน (color_space, colour, image_adjust, ...) คงไว้เป็น legacy ให้เทมเพลต/โปรเจกต์เก่า แต่ palette ไม่ชี้มาแล้ว
  color_tools: mergeGroup("Color", "Color",
    ["grayscale", "colorspace", "colorspace_inverse", "debayer", "channel_split", "rgb_adjust", "color_inrange", "kmeans_segment", "slic", "white_balance", "invert", "pseudocolor"], "grayscale"),

  enhance_tools: mergeGroup("Enhance", "Enhance",
    ["brightness_contrast", "gamma", "hist_equalize", "clahe", "contrast_stretch", "flatfield_correction", "quantize", "unsharp", "laplacian_sharpen", "custom_kernel", "dehaze", "lowlight_enhance"], "brightness_contrast"),

  filter_denoise: mergeGroup("Filter & Denoise", "Filter & Denoise",
    ["salt_pepper_noise", "gaussian_noise", "median_blur", "gaussian_blur", "box_blur", "bilateral", "wiener_filter", "guided_filter", "anisotropic_diffusion", "nlm_denoise", "deconv_wiener", "deconv_lucy"], "salt_pepper_noise"),

  segment_edge: mergeGroup("Segment & Edge", "Segmentation & Edge",
    ["binary_threshold", "otsu_threshold", "adaptive_threshold", "multi_otsu_threshold", "canny", "sobel", "laplacian_edge", "scharr_edge", "prewitt_edge", "roberts_edge", "log_edge", "frangi_vesselness", "watershed", "watershed_marker", "grabcut", "region_growing"], "binary_threshold"),

  analysis_detect: mergeGroup("Analyze / Detect", "Analysis & Detection",
    ["find_contours", "connected_components", "distance_transform", "local_std_filter", "local_entropy_filter", "gabor_filter", "glcm", "focus_score", "regionprops", "euler_number", "feret", "line_profile", "shape_filter", "bounding_boxes", "hough_circles", "hough_lines", "harris_corners", "shi_tomasi_corners", "orb_keypoints", "akaze_keypoints", "fast_keypoints", "pass_fail"], "find_contours"),

  // name (ชื่อบนการ์ด node, ถูกตัดที่ ~92px) ของ 2 กลุ่มข้างบนย่อให้พอดี · group (ชิปหมวดใน detail-panel/สี) ยังเป็นชื่อเต็มเหมือนหมวด palette

  // PSNR/SSIM รับภาพ 2 ใบ → inputs:2 (ช่อง "เข้า B" ใน editor.js)
  image_quality: mergeGroup("Image Quality", "Image Quality",
    ["psnr", "ssim", "delta_e", "seg_score", "piqe"], "psnr", [], { inputs: 2 }),

  // Motion ทั้งสามจำ state ข้ามเฟรม → stateful:true · เปลี่ยนวิธี = params เปลี่ยน → getNodeState (fingerprint) reset+dispose state เก่าเอง
  motion: mergeGroup("Motion", "Motion",
    ["mog2", "farneback_optical_flow", "pyrlk_optical_flow"], "mog2", [], { stateful: true }),

  // Image Registration (2026-10-24) — แทนหมวด TEMPLATE MATCHING ให้ตรง MATLAB ที่รวม normxcorr2 กับ imregister ไว้หมวดเดียว
  // template_matching (เดี่ยว) / image_arithmetic ที่ mode = ecc_register คงเป็น legacy · migrateLegacyStep ย้ายมาที่นี่
  image_registration: mergeGroup("Image Registration", "Image Registration",
    ["template_matching", "ecc_register"], "template_matching", [], { inputs: 2 }),

  // Morphology (2026-10-18) — 1 operation = 1 เครื่องมือ · ลำดับ dropdown = ลำดับ TOOL_SECTIONS.MORPHOLOGY · default = Erosion (เหมือนค่าเริ่มต้นเดิม)
  // ไม่อยู่ใน CATEGORY_FILTER_IDS (ที่มี OLD_HOST ให้เทียบ) — ย้ายจาก morphology ด้วย migrateMorphologyStep ด้านล่างแทน
  morph_tools: mergeGroup("Morphology", "Morphology", MORPH_OP_IDS, "morph_erode"),

  // Legacy groups are kept for saved templates/projects created by earlier versions.
  // They are registered and executable, but the palette no longer exposes them.

  enhancement: mergeGroup("Enhancement", "Enhancement",
    ["brightness_contrast", "gamma", "hist_equalize", "clahe", "unsharp", "laplacian_sharpen"], "brightness_contrast"),

  restoration: mergeGroup("Restoration", "Restoration",
    ["salt_pepper_noise", "median_blur", "gaussian_blur", "box_blur", "bilateral"], "gaussian_blur"),

  segmentation: mergeGroup("Segmentation", "Segmentation",
    ["binary_threshold", "otsu_threshold", "adaptive_threshold", "canny", "sobel", "find_contours", "connected_components", "watershed"], "binary_threshold"),
};

// สีประจำแต่ละกลุ่ม (ใช้ทั้งกับ node บนกราฟ และปุ่มบน palette)
// title = สีแถบหัว node, body = สีพื้น node
// สีหมวด (Design(1).md §3.1) — accent เป็นตัวช่วยจำที่ title/border, body เป็นดำ panel เดียวกันทั้งหมด
const NODE_BODY = "#101927"; // = --bg-panel
export const GROUP_COLORS = {
  "Color Space":      { title: "#22d3ee", body: NODE_BODY }, // Cyan
  "Image Adjust":     { title: "#27c98a", body: NODE_BODY }, // Green
  "Sharpen":          { title: "#8bd450", body: NODE_BODY }, // Lime
  "Geometry":         { title: "#facc15", body: NODE_BODY }, // Yellow
  "Noise":            { title: "#f59e0b", body: NODE_BODY }, // Amber
  "Blur":             { title: "#38bdf8", body: NODE_BODY }, // Sky
  "Enhancement":    { title: "#27c98a", body: NODE_BODY }, // Green
  "Restoration":      { title: "#ff9f43", body: NODE_BODY }, // Orange
  "Morphology":     { title: "#9b6cff", body: NODE_BODY }, // Purple
  "Threshold":        { title: "#f97316", body: NODE_BODY }, // Orange
  "Edge Detection":   { title: "#ec4899", body: NODE_BODY }, // Pink
  "Shape Analysis":   { title: "#a78bfa", body: NODE_BODY }, // Lavender
  "Segmentation":     { title: "#ff5577", body: NODE_BODY }, // Red/Pink
  "Segmentation":   { title: "#ff5577", body: NODE_BODY }, // Red/Pink
  "Object Detection": { title: "#8b7bff", body: NODE_BODY }, // Violet
  "Colour":       { title: "#2997ff", body: NODE_BODY }, // Blue
  "Frequency Domain": { title: "#c084fc", body: NODE_BODY }, // Violet-purple, distinct from existing swatches
  "Image Arithmetic": { title: "#14b8a6", body: NODE_BODY }, // Teal — distinct from cyan(#22d3ee)/sky(#38bdf8)
  // Tier 2, 18b — เช็คแล้วว่า hue ทั้งสองไม่ชนกับสวอตช์เดิมด้านบน (คำนวณ hue ของทุกสีเดิมก่อนเลือก):
  // แดง (hue≈0°) ไม่มีสีไหนอยู่ในช่วงนี้เลย (สีที่ใกล้สุดคือ Segmentation ที่ hue≈348° ห่าง ~12° และ
  // "Frequency Domain"'s orange2 ที่ hue≈25° ห่าง ~25°) — เขียวหญ้า (hue≈127°) อยู่ตรงกลางระหว่าง Lime
  // (hue≈93°) กับ Image Adjust/Enhancement (hue≈157°) ห่างจากทั้งคู่ ~30°
  "Template Matching": { title: "#ef4444", body: NODE_BODY }, // Red — สื่อ "หากรอบเป้าหมาย"
  "Image Quality":     { title: "#2ecc40", body: NODE_BODY }, // Grass green — สื่อ "คะแนนคุณภาพ"
  // Tier 2, 18d — fuchsia (hue≈292°) เช็คกับทุกสีด้านบนแล้ว: ใกล้สุดคือ Frequency Domain #c084fc (hue≈270°, ห่าง ~22°)
  // และ Edge Detection #ec4899 (hue≈330°, ห่าง ~38°) — อิ่มตัวกว่า #c084fc มากจึงแยกด้วยตาได้
  "Motion":            { title: "#d946ef", body: NODE_BODY }, // Fuchsia — สื่อ "การเคลื่อนไหว/ข้ามเฟรม"
  // กลุ่มระดับหมวด palette (2026-10-07) — ใช้สีเดียวกับกลุ่มเก่าตัวแรกของหมวด ผู้ใช้เห็นสีหมวดเดิมไม่เปลี่ยน
  // (Image Quality / Motion ใช้ชื่อกลุ่มเดิม จึงใช้สีเดิมทันที)
  "Color":                { title: "#22d3ee", body: NODE_BODY }, // Cyan = Color Space
  "Enhance":              { title: "#27c98a", body: NODE_BODY }, // Green = Image Adjust
  "Filter & Denoise":     { title: "#38bdf8", body: NODE_BODY }, // Sky = Blur
  "Segmentation & Edge":  { title: "#ff5577", body: NODE_BODY }, // Red/Pink = Segmentation
  "Analysis & Detection": { title: "#a78bfa", body: NODE_BODY }, // Lavender = Shape Analysis
  "Image Registration":   { title: "#ef4444", body: NODE_BODY }, // Red = Template Matching
};

// ลำดับกลุ่มตามหัวข้อการสอน
export const GROUP_ORDER = [
  "Color Space",
  "Image Adjust",
  "Geometry",
  "Sharpen",
  "Noise",
  "Blur",
  "Morphology",
  "Threshold",
  "Edge Detection",
  "Shape Analysis",
  "Segmentation",
  "Object Detection",
  "Colour",
  "Frequency Domain",
  "Image Arithmetic",
  "Template Matching",
  "Image Quality",
  "Motion",
  "Color",
  "Enhance",
  "Filter & Denoise",
  "Segmentation & Edge",
  "Analysis & Detection",
  "Image Registration",
];

// resolve needs รองรับทั้ง string และ function (merged filter)
export function needsOf(filter, params) {
  return typeof filter.needs === "function" ? filter.needs(params || {}) : filter.needs;
}

// id จริงของ op ที่ step นี้รัน: กล่องรวม (mergeGroup) = ค่า mode ที่เลือก (ไม่มี = วิธีเริ่มต้น เหมือน apply ของกลุ่ม) · กล่องเดี่ยว = step.id
// worker ใช้ติดป้ายให้ ctx.upstream ของ pass_fail
export function opIdOfStep(step) {
  const f = FILTERS[step && step.id];
  const first = f && f.params && f.params[0];
  if (first && first.key === MODE_KEY && first.controls === true) {
    const m = step.params && step.params[MODE_KEY];
    return typeof m === "string" && Object.hasOwn(OPS, m) ? m : first.default; // เหมือน apply ของกลุ่ม: mode ที่ไม่รู้จัก → วิธีเริ่มต้น
  }
  return step && step.id;
}

// คืนค่า default params ของ filter (ใช้ตอนเพิ่ม node ใหม่)
export function defaultParams(filterId) {
  const f = FILTERS[filterId];
  const p = {};
  for (const param of f.params) p[param.key] = param.default;
  return p;
}

// ---------- migration: step ของกลุ่มเก่า → กลุ่มระดับหมวด palette (2026-10-07) ----------
// ใช้ตอนวางเทมเพลต (editor.applyTemplate) — เทมเพลต/โปรเจกต์เก่าอ้าง filter id ของกลุ่มเก่า (color_space,
// segmentation, restoration, watershed, psnr ...) ซึ่ง dropdown ไม่ครบหมวด · ย้ายไปกลุ่มใหม่ที่ครบ
// key ของ param เป็น "opId__key" เหมือนกันทุก mergeGroup จึงคัดลอกตรง ๆ ได้ · op เดี่ยวเก่า (ไม่มี mode) ต้องเติม prefix + mode
export const CATEGORY_FILTER_IDS = ["color_tools", "enhance_tools", "filter_denoise", "segment_edge", "analysis_detect", "image_quality", "motion", "image_registration"];
const OP_TO_CATEGORY_FILTER = {};
for (const gid of CATEGORY_FILTER_IDS) {
  for (const o of FILTERS[gid].params[0].options) OP_TO_CATEGORY_FILTER[o.value] = gid;
}
// คืน step ใหม่ (ไม่แก้ของเดิม) · step ที่ไม่ต้องย้าย (กลุ่มใหม่อยู่แล้ว / id ไม่รู้จัก / mode ไม่รู้จัก) → คืนตัวเดิม · morphology เดิม → morph_tools (migrateMorphologyStep ท้ายไฟล์)
export function migrateLegacyStep(step) {
  if (!step || !Object.hasOwn(FILTERS, step.id) || CATEGORY_FILTER_IDS.includes(step.id)) return step;
  if (step.id === "morphology") return migrateMorphologyStep(step);
  const params = step.params || {};
  const merged = FILTERS[step.id].params.some((q) => q.key === MODE_KEY);
  const op = merged ? (params[MODE_KEY] != null ? params[MODE_KEY] : defaultParams(step.id)[MODE_KEY]) : step.id;
  if (!Object.hasOwn(OP_TO_CATEGORY_FILTER, op)) return step;
  const out = { [MODE_KEY]: op };
  for (const k of Object.keys(params)) {
    if (k === MODE_KEY) continue;
    out[merged ? k : op + "__" + k] = params[k];
  }
  return { ...step, id: OP_TO_CATEGORY_FILTER[op], params: out };
}

// morphology เดิม (select "op" + params รวม) → morph_tools (mode = morph_<op> + params ของ op นั้นพร้อม prefix "morph_<op>__")
// op ไม่มี → "erode" (default เดิม) · op ไม่รู้จัก → คืน step เดิม (legacy morphology ยังรันได้) · คัดเฉพาะ key ที่ตัวใหม่ใช้จริง ไม่ mutate input
function migrateMorphologyStep(step) {
  const params = step.params || {};
  const op = params.op != null ? params.op : "erode";
  const newId = "morph_" + op;
  if (typeof op !== "string" || !MORPH_OP_IDS.includes(newId)) return step;
  const out = { [MODE_KEY]: newId };
  for (const q of OPS[newId].params) {
    if (Object.hasOwn(params, q.key)) out[newId + "__" + q.key] = params[q.key];
  }
  return { ...step, id: "morph_tools", params: out };
}

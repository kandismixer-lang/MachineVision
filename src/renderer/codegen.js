// สร้างโค้ด Python (OpenCV) ตามลำดับ Box ที่ภาพวิ่งผ่านจริง ๆ ในกล่องผลลัพธ์หนึ่งกล่อง
// (คนละอย่างกับปุ่ม Export ที่เป็นโค้ดเทรน YOLO)
//
// pipeline = [{ id (filterId), params, nodeId }, ...] เรียงจากแหล่งภาพ → ปลายทาง
// แต่ละ step map เป็นโค้ด cv2.* ตามค่าที่ตั้งไว้จริง
// จัดการแปลงเป็นขาวดำอัตโนมัติ (เหมือนในแอป) เมื่อ op ต้องการภาพ gray
//
// หมายเหตุ: โค้ดที่ gen ใช้ BGR (cv2.imread) ตามธรรมเนียม OpenCV — ต่างจากในแอปที่ทำงานบน RGB
// ผลลัพธ์เชิงหลักการเหมือนกัน (ค่าเนื้อสี H, threshold ฯลฯ เท่ากัน)
import { OPS, FILTERS, passFailParams, PASS_FAIL_METRICS, NLM_MAX_SIDE, LUCY_MAX_ITERS, LUCY_WORK_BUDGET, ECC_DEFAULT_SIDE, finiteOr, boolOr, HITMISS_PATTERNS } from "./filters/registry.js";

// ชื่อไทยของ op (โยงกับที่ผู้เรียนเห็นบนจอ) + teaching note สั้น ๆ ต่อ op คลาสสิก
const TEACH = {
  grayscale: "รวม 3 ช่องสีเป็นความสว่างเดียว — พื้นฐานก่อน threshold/edge",
  gaussian_blur: "เฉลี่ยเพื่อนบ้านแบบถ่วงน้ำหนักระฆังคว่ำ — ลด noise แบบสุ่ม",
  median_blur: "ใช้ค่ากลางของเพื่อนบ้าน จึงลบ noise เกลือ-พริกไทยดีกว่า Gaussian",
  binary_threshold: "สว่างเกินเกณฑ์ = ขาว, ต่ำกว่า = ดำ — แบ่งภาพขั้นพื้นฐานสุด",
  otsu_threshold: "หาเกณฑ์ที่ดีที่สุดให้อัตโนมัติจาก histogram",
  canny: "ตรวจขอบหลายขั้น — มาตรฐานทองของการหาขอบ",
  morphology: "จัดรูปทรงด้วย kernel (กัด/ขยาย/อุดรู/ลบจุด)",
  find_contours: "หาเส้นรอบวัตถุขาว แล้วนับ/วาด",
  pass_fail: "เทียบค่าที่วัดได้กับเกณฑ์ → OK (ผ่าน) / NG (ไม่ผ่าน) — ไม่มีค่าให้ตัดสิน = NG",
};
function thaiName(opId) {
  return (OPS[opId] && OPS[opId].name) || opId;
}

function odd(n) {
  n = Math.round(Number(n) || 1);
  return n % 2 === 0 ? n + 1 : n;
}
function int(n) {
  return Math.round(Number(n) || 0);
}
function num(n) {
  return Number(n) || 0;
}
// ต่างจาก num()/int()/odd() ด้านบนที่ตั้งใจให้ 0/falsy กลายเป็น fallback (ใช้กับพารามิเตอร์ที่ 0 ไม่มีความหมายจริง เช่น ksize) —
// finiteOr() (ตัวกลางจาก registry.js ที่ import มาด้านล่าง) ใช้กับพารามิเตอร์ที่ 0 เป็นค่าที่ใช้งานได้จริง (เช่น tlX/trX ของ
// perspective_warp, k1/k2 ของ undistort) · ผลลัพธ์: เลขจำกัด → ตัวนั้น, สตริงตัวเลข ("0", "5") → เลข, อย่างอื่น
// (undefined/null/""/NaN/Infinity/boolean/สตริงที่ไม่ใช่เลข เช่น "5px") → ค่า default · แอปใช้ฟังก์ชันตัวเดียวกัน จึงตรงกันเสมอ
// B7: finiteOr / boolOr อยู่ที่ registry.js (แหล่งเดียว — ตัวเดียวกับที่แอปใช้ จึงตรงกันแม้ param เป็น string)
// clamp ให้ตรงกับ numIn() ของ registry.js เป๊ะ (finiteOr แล้ว clamp เข้าช่วง min/max ของ param นั้น) — ใช้กับ
// op ที่ registry clamp ค่าไว้ ถ้า codegen ไม่ clamp ด้วย ไฟล์โปรเจกต์ที่แก้มือจะทำให้แอปกับสคริปต์ให้ผลต่างกัน
function clampNum(v, d, lo, hi) {
  return Math.min(hi, Math.max(lo, finiteOr(v, d)));
}

// ใช้ร่วมกันโดย GEN.orb_keypoints/akaze_keypoints/fast_keypoints ด้านล่าง — ต้องตรงกับ comparator ของ
// rankKeypoints() ใน registry.js เป๊ะทุก key/ทุกทิศทาง (response ลด, แล้ว y เพิ่ม, x เพิ่ม, size ลด, angle
// เพิ่ม, octave เพิ่ม — ไม่ใช้ reverse=True เพราะกลับด้านทุก key พร้อมกัน ไม่ตรงกับทิศทางที่ต้องการต่อ key)
// เหตุผล (ดู comment เต็มที่ rankKeypoints ใน registry.js): opencv.js กับ cv2 ให้ "ชุด" keypoint เดียวกัน
// เป๊ะที่ความละเอียดเดียวกัน แต่ detector คืนมาคนละลำดับกัน (nth_element ไม่ใช่ full sort) — response มี tie
// เยอะมาก ถ้าไม่มี tie-break ตายตัวที่เหมือนกันทั้งสองฝั่ง จุดตัด top-N จะสุ่มไปตามลำดับภายในของแต่ละ build
// IP batch 2 ชิ้น B — ตารางบรรทัด Python ต่อ operation ของ imarith/logical_ops (key ตรงกับ IMARITH_OPS/
// LOGICAL_OPS ใน registry.js) · pickOwn เลือกด้วย own-property เท่านั้น ("constructor" จากไฟล์โปรเจกต์ที่แก้มือ
// ต้องตกไปที่ค่าเริ่มต้นเหมือนแอป ไม่ใช่ได้ฟังก์ชันของ Object.prototype)
const IMARITH_PY = {
  add: "img = cv2.add(img, arith_b)  # A + B เกิน 255 ตัดที่ 255",
  subtract: "img = cv2.subtract(img, arith_b)  # A - B ติดลบตัดเป็น 0",
  multiply: "img = cv2.multiply(img, arith_b, scale=1/255)  # A x B / 255",
  divide: "img = cv2.divide(img, arith_b, scale=255)  # A / B x 255 ตรงที่ B = 0 ได้ 0",
};
const LOGICAL_PY = {
  and: "img = cv2.bitwise_and(a_bin, b_bin)  # ขาวเฉพาะที่ขาวทั้งคู่",
  or: "img = cv2.bitwise_or(a_bin, b_bin)  # ขาวถ้าขาวอย่างน้อยหนึ่งภาพ",
  xor: "img = cv2.bitwise_xor(a_bin, b_bin)  # ขาวเฉพาะที่ขาวภาพเดียว",
  a_and_not_b: "img = cv2.bitwise_and(a_bin, cv2.bitwise_not(b_bin))  # หัก B ออกจาก A",
};
function pickOwn(table, key, fallback) {
  return Object.hasOwn(table, key) ? table[key] : table[fallback];
}
// IP batch 3 — interpolation ของ Resize/Rotate (auto/ไม่รู้จัก = ไม่อยู่ในตาราง → บรรทัดเดิมทุกตัวอักษร) · prop/view ของ shape_filter
const INTERP_PY = { nearest: "cv2.INTER_NEAREST", bilinear: "cv2.INTER_LINEAR", bicubic: "cv2.INTER_CUBIC" };
const SHAPE_FILTER_PROP_PY = { circularity: "circularity", solidity: "solidity", eccentricity: "eccentricity", extent: "extent", none: "none" };
const SHAPE_FILTER_VIEW_PY = { overlay: "overlay", mask: "mask" };
// IP batch 3 — คู่แฝดของ regionPixelStats + regionShapeValues + hullRowSpans ใน registry.js (สูตร/ลำดับเดียวกัน → เท่ากันทุก bit)
// ไม่มีตัวแปร img ข้างใน (กันการเปลี่ยนชื่อ img ของสาย 2 ภาพ) · ys เป็นเลขแถวสัมพันธ์กับกรอบ จึงใช้ lo[ys] ได้ตรง ๆ
function regionStatsPyLines() {
  return [
    "def region_stats(c, x, y, w, h, gray, bin_img, need_hull):",
    "    # ค่ารูปร่างแบบนับพิกเซล — สูตร/ลำดับเดียวกับ regionPixelStats + regionShapeValues ของแอป",
    "    obj = np.zeros((h, w), np.uint8)",
    "    cv2.fillPoly(obj, [c], 255, cv2.LINE_8, 0, (-x, -y))",
    "    sel = (obj > 0) & (bin_img[y:y + h, x:x + w] > 0)  # อยู่ในเส้นขอบนอก และเป็นสีขาว (รูไม่นับ)",
    "    ys, xs = np.nonzero(sel)",
    "    xs = xs.astype(np.int64)",
    "    ys = ys.astype(np.int64)",
    "    n = int(xs.size)",
    "    sx, sy = int(xs.sum()), int(ys.sum())",
    "    sxx, syy, sxy = int((xs * xs).sum()), int((ys * ys).sum()), int((xs * ys).sum())",
    "    total = int(gray[y:y + h, x:x + w][sel].sum(dtype=np.int64))",
    "    solidity = None",
    "    if need_hull:",
    "        hp = cv2.convexHull(c).reshape(-1, 2).astype(np.int64)",
    "        lo = np.full(h, 2147483647, np.int64)",
    "        hi = np.full(h, -2147483648, np.int64)",
    "        m = len(hp)",
    "        for i in range(m):",
    "            ax, ay = int(hp[i, 0]), int(hp[i, 1])",
    "            bx, by = int(hp[(i + 1) % m, 0]), int(hp[(i + 1) % m, 1])",
    "            if ay == by:",
    "                lo[ay - y] = min(int(lo[ay - y]), ax, bx)",
    "                hi[ay - y] = max(int(hi[ay - y]), ax, bx)",
    "                continue",
    "            rows = np.arange(min(ay, by), max(ay, by) + 1, dtype=np.int64)",
    "            den = by - ay",
    "            num = ax * den + (rows - ay) * (bx - ax)",
    "            if den < 0:",
    "                num, den = -num, -den",
    "            lo[rows - y] = np.minimum(lo[rows - y], -((-num) // den))  # ปัดขึ้น",
    "            hi[rows - y] = np.maximum(hi[rows - y], num // den)  # ปัดลง",
    "        outside = int(((xs + x < lo[ys]) | (xs + x > hi[ys])).sum())",
    "        solidity = n / (int(np.maximum(hi - lo + 1, 0).sum()) + outside)",
    "    cxx = float(n * sxx - sx * sx) / (n * n) + 1 / 12",
    "    cyy = float(n * syy - sy * sy) / (n * n) + 1 / 12",
    "    cxy = float(n * sxy - sx * sy) / (n * n)",
    "    common = np.sqrt((cxx - cyy) * (cxx - cyy) + 4 * cxy * cxy)",
    "    major = 2 * np.sqrt(2) * np.sqrt(cxx + cyy + common)",
    "    minor = 2 * np.sqrt(2) * np.sqrt(cxx + cyy - common)",
    "    ecc = 2 * np.sqrt(max(0.0, (major / 2) * (major / 2) - (minor / 2) * (minor / 2))) / major",
    "    return {'area_px': n, 'equiv_diameter': np.sqrt(4 * n / np.pi), 'extent': n / (w * h), 'solidity': solidity,",
    "            'major_axis': major, 'minor_axis': minor, 'eccentricity': ecc, 'mean_intensity': total / n}",
  ];
}
// ภาพ B → ขาวดำ → ปรับขนาดเท่า A → threshold 127 ได้ b_bin (ลำดับเดียวกับ GEN.mask และ registry.js)
function bBinaryLines() {
  return [
    "b_gray = img_b if img_b.ndim == 2 else cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)",
    "if img.shape[:2] != b_gray.shape[:2]:",
    "    b_gray = cv2.resize(b_gray, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)",
    "_, b_bin = cv2.threshold(b_gray, 127, 255, cv2.THRESH_BINARY)",
  ];
}

// IP batch 2 ชิ้น C — บรรทัด Python ที่สร้าง `z` = ค่าสุ่มเกาส์ n ค่า (nExpr = นิพจน์ Python) ตรงกับ seededGaussian(seed, n)
// ของ registry.js ทุก bit: mulberry32 ตัวที่ i (นับจาก 1) คำนวณตรงจาก (seed + i·0x6D2B79F5) mod 2^32 แบบ vectorized
// (ไม่วนทีละค่า) แล้ว hash ด้วย uint32 ที่ล้นแบบวนรอบเหมือน Math.imul · Box-Muller ทีละคู่ (n คี่ = ทิ้งค่าสุดท้าย)
// noise แบบอื่นในอนาคต (speckle/poisson) ใช้ตัวนี้ซ้ำ ห้ามเขียน PRNG ใหม่
function seededGaussianPy(nExpr, seed) {
  return [
    `n_vals = ${nExpr}`,
    "n_pairs = (n_vals + 1) // 2",
    `rnd = ((np.uint64(${seed}) + np.arange(1, 2 * n_pairs + 1, dtype=np.uint64) * np.uint64(0x6D2B79F5)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)`,
    'with np.errstate(over="ignore"):  # mulberry32: คูณ/บวก uint32 ที่ล้นแล้ววนรอบ (ตั้งใจ)',
    "    rnd = (rnd ^ (rnd >> np.uint32(15))) * (rnd | np.uint32(1))",
    "    rnd = rnd ^ (rnd + (rnd ^ (rnd >> np.uint32(7))) * (rnd | np.uint32(61)))",
    "    rnd = (rnd ^ (rnd >> np.uint32(14))).astype(np.float64)",
    "rad = np.sqrt(-2 * np.log((rnd[0::2] + 1) / 4294967296))  # Box-Muller: u1 อยู่ในช่วง (0, 1] กัน log(0)",
    "ang = 2 * np.pi * (rnd[1::2] / 4294967296)",
    "z = np.empty(2 * n_pairs)",
    "z[0::2] = rad * np.cos(ang)",
    "z[1::2] = rad * np.sin(ang)",
    "z = z[:n_vals]",
  ];
}

// ---------------------------------------------------------------------------------------------------------------
// IP batch 4A (2026-10-15) — ตัวช่วยของ Custom Kernel (PSF) / Guided Filter / Pseudocolor / px→mm
// ---------------------------------------------------------------------------------------------------------------
// ฝาแฝดของ psfKernel (registry.js) — ลำดับนิพจน์เดียวกับ JS: px * c + py * s, np.clip(..., -hl, hl), dx * dx + dy * dy <= 0.25, disk เป็นเลขจำนวนเต็ม
// vectorize จุดย่อย 16×16 ด้วย numpy (แกน 4 มิติ: y, x, sy, sx) · ห้ามมี token `img` ในบรรทัดเหล่านี้ (สาย B แทนที่ \bimg\b)
function psfKernelPyLines() {
  return [
    "def psf_kernel(kind, size, angle_deg):",
    "    # PSF (point spread function) นับจุดย่อย 16x16 ต่อพิกเซลที่อยู่ในรูป แล้วหารด้วยจำนวนรวม → ผลรวม = 1",
    "    ss = 16",
    "    sub = 2 * np.arange(ss, dtype=np.int64) + 1 - ss",
    "    if kind == \"disk\":",
    "        h = int(np.ceil(size - 0.5))",
    "        n = 2 * h + 1",
    "        lim = 4 * ss * ss * size * size  # (2*ss*R)^2 — พิกัดจุดย่อยคูณ 2*ss ให้เป็นจำนวนเต็ม",
    "        base = (np.arange(n, dtype=np.int64) - h) * 2 * ss",
    "        gx = base[None, :, None, None] + sub[None, None, None, :]",
    "        gy = base[:, None, None, None] + sub[None, None, :, None]",
    "        cnt = ((gx * gx + gy * gy) <= lim).sum(axis=(2, 3))",
    "    else:",
    "        th = angle_deg * np.pi / 180",
    "        c, s = np.cos(th), -np.sin(th)  # มุมทวนเข็มแบบ MATLAB (แกน y ของภาพชี้ลง)",
    "        hl = (size - 1) / 2",
    "        h = int(np.ceil(hl * max(abs(c), abs(s)) + 0.5))",
    "        n = 2 * h + 1",
    "        frac = sub / (2 * ss)",
    "        pos = np.arange(n, dtype=np.int64) - h",
    "        gx = pos[None, :, None, None] + frac[None, None, None, :]",
    "        gy = pos[:, None, None, None] + frac[None, None, :, None]",
    "        t = np.clip(gx * c + gy * s, -hl, hl)",
    "        dx = gx - t * c",
    "        dy = gy - t * s",
    "        cnt = ((dx * dx + dy * dy) <= 0.25).sum(axis=(2, 3))",
    "    k0 = 0  # ตัดวงนอกที่เป็น 0 ทั้งวง (ทั้ง 4 ด้านพร้อมกัน)",
    "    while n - 2 * k0 > 1:",
    "        r = k0",
    "        if cnt[r, r:n - r].any() or cnt[n - 1 - r, r:n - r].any() or cnt[r:n - r, r].any() or cnt[r:n - r, n - 1 - r].any():",
    "            break",
    "        k0 += 1",
    "    m = n - 2 * k0",
    "    cnt = cnt[k0:k0 + m, k0:k0 + m]",
    "    return cnt / cnt.sum()",
  ];
}
// ฝาแฝดของ guidedFilterF32 — float32 ทุกขั้นตามลำดับ op ของ JS (a = cov / (v + eps) · b = mP − a·mI · q = box(a)·I + box(b))
function guidedPyLines() {
  return [
    "def guided_f32(I, P, r, eps255):",
    "    # Guided Filter (He et al. 2010) บน float32 สเกล 0-255 — ขอบภาพสะท้อนค่า (BORDER_REFLECT_101)",
    "    k = (2 * r + 1, 2 * r + 1)",
    "    box = lambda m: cv2.boxFilter(m, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT_101)",
    "    mI, mP = box(I), box(P)",
    "    cI, cIP = box(I * I), box(I * P)",
    "    v = cI - mI * mI",
    "    cov = cIP - mI * mP",
    "    a = cov / (v + eps255)",
    "    b = mP - a * mI",
    "    return box(a) * I + box(b)",
  ];
}
// แผนที่สีของ Pseudocolor — สูตรเดียวกับ PSEUDOCOLOR_MAPS ใน registry.js (เพิ่มแผนที่ใหม่ = เพิ่ม 1 แถวทั้งสองที่)
const PSEUDOCOLOR_PY = {
  jet: [
    "lut_r = np.floor(np.clip(382.5 - np.abs(4 * i - 765), 0, 255) + 0.5).astype(np.uint8)",
    "lut_g = np.floor(np.clip(382.5 - np.abs(4 * i - 510), 0, 255) + 0.5).astype(np.uint8)",
    "lut_b = np.floor(np.clip(382.5 - np.abs(4 * i - 255), 0, 255) + 0.5).astype(np.uint8)",
  ],
  hot: [
    "lut_r = np.floor(np.clip(3 * i, 0, 255) + 0.5).astype(np.uint8)",
    "lut_g = np.floor(np.clip(3 * i - 255, 0, 255) + 0.5).astype(np.uint8)",
    "lut_b = np.floor(np.clip(3 * i - 510, 0, 255) + 0.5).astype(np.uint8)",
  ],
  cool: [
    "lut_r = i.astype(np.uint8)",
    "lut_g = (255 - i).astype(np.uint8)",
    "lut_b = np.full(256, 255, np.uint8)",
  ],
};
// px → mm: คอลัมน์ mm ต่อท้ายหลังบรรทัด print เดิมทั้งหมด (เฉพาะ f > 0) — ทศนิยม 2 ตำแหน่งแบบ np.floor(v * 100 + 0.5) / 100
// หัวตารางขึ้นต้น `object` (ห้ามขึ้นต้น `index  area` — เทสต์ regionprops-28b ตัดตารางเดิมด้วยหัวนี้)
const mm2 = (expr) => `{np.floor((${expr}) * 100 + 0.5) / 100:.2f}`;
function mmPyLines(opId, f) {
  if (!(f > 0)) return [];
  const head = [`mm_per_px = ${String(f)}  # มิลลิเมตรต่อ 1 พิกเซล`];
  if (opId === "regionprops") {
    return [
      ...head,
      "if regions:",
      "    print('object  area_mm2  perimeter_mm  equiv_diam_mm  major_axis_mm  minor_axis_mm')",
      "for idx, r in enumerate(regions[:20], start=1):",
      `    print(f"{idx}  ${mm2("r['area_px'] * (mm_per_px * mm_per_px)")}  ${mm2("r['perimeter'] * mm_per_px")}  ${mm2("r['equiv_diameter'] * mm_per_px")}  ${mm2("r['major_axis'] * mm_per_px")}  ${mm2("r['minor_axis'] * mm_per_px")}")`,
    ];
  }
  if (opId === "feret") {
    return [
      ...head,
      "if feret_rows:",
      "    print('object  feret_max_mm  feret_min_mm  feret_len_mm')",
      "for r in feret_rows[:20]:",
      `    print(f"{r[0]}  ${mm2("r[4] * mm_per_px")}  ${mm2("r[5] * mm_per_px")}  ${mm2("r[6] * mm_per_px")}")`,
    ];
  }
  return [ // line_profile
    ...head,
    "line_length_mm = line_length_px * mm_per_px",
    "edge_pos_mm = None if edge_pos_px is None else edge_pos_px * mm_per_px",
    "edge_pair_mm = None if edge_pair_px is None else edge_pair_px * mm_per_px",
    `print(f"line length: ${mm2("line_length_mm")} mm")`,
    `print("edge position from A: " + ("-" if edge_pos_mm is None else f"${mm2("edge_pos_mm")} mm"))`,
    `print("edge pair distance: " + ("-" if edge_pair_mm is None else f"${mm2("edge_pair_mm")} mm"))`,
  ];
}

const KEYPOINT_SORT_KEY = "key=lambda k: (-k.response, k.pt[1], k.pt[0], -k.size, k.angle, k.octave)";

// แยก opId จริง + ค่าพารามิเตอร์ (ตัด prefix opId__ ของ merged node ออก)
function resolveOp(step) {
  const p = step.params || {};
  const hasMode = Object.prototype.hasOwnProperty.call(p, "mode");
  const opId = hasMode ? p.mode : step.id;
  const local = {};
  if (hasMode) {
    const prefix = opId + "__";
    for (const k in p) if (k.startsWith(prefix)) local[k.slice(prefix.length)] = p[k];
  } else {
    for (const k in p) local[k] = p[k];
  }
  return { opId, local };
}

// ตารางแปลง op → โค้ด Python
//   needs: 'gray' = ต้องแปลงเป็นขาวดำก่อน (แทรกให้อัตโนมัติ) | 'any'
//   out:   สภาพภาพหลังทำงาน 'gray' | 'color' | 'same'  (ใช้ track ว่าตอนนี้ภาพขาวดำหรือสี)
//   gen(p, ctx): คืน array ของบรรทัดโค้ด (ctx.gray = ตอนนี้ภาพเป็นขาวดำหรือไม่ · ctx.mirrored = ลำดับช่องสี ดูด้านล่าง)
//   appChannelOrder: true = หลัง step นี้ช่องของ cv2 เรียงเหมือนแอปแล้ว (ไม่ใช่ BGR ที่กลับด้านของ RGB) — ดู ctx.mirrored ด้านล่าง
//   grayOutOf(p): true = ภาพ Python หลัง step นี้เป็น 2 มิติจริงแม้ out จะบอก "color" — ใช้กับ ctx.mirrored เท่านั้น (ไม่แตะ ctx.gray
//                 จึงไม่เปลี่ยนบรรทัด Python ของ step ใด) · ตอนนี้มีแค่ grabcut แบบ mask (ดูคอมเมนต์ที่ GEN.grabcut)
//
// ctx.mirrored (IP batch 2 ชิ้น C รอบแก้หลังรีวิว, MAJOR-1) — "ช่องของภาพใน Python = ช่องของแอปเรียงกลับด้าน" แบบ compile-time:
// แอปเก็บภาพสีเป็น R,G,B แต่ cv2.imread ให้ B,G,R จึงเริ่มที่ true · หลัง Color Space ทั้งสองฝั่งเก็บ H,S,V (หรือ L,a,b ฯลฯ)
// เรียงเหมือนกันในหน่วยความจำ (แอป RGB2HSV / Python BGR2HSV) → false · ขั้นที่ได้ภาพขาวดำ (รวม BGR2GRAY อัตโนมัติของ
// needs:"gray" และ grayOutOf) → true อีกครั้ง · ขั้นสี→สีอื่นไม่เปลี่ยน · ข้อคงที่: ctx.gray === true ⇒ ctx.mirrored === true เสมอ
// (ภาพขาวดำเกิดได้จาก 2 ทางข้างบนเท่านั้น) ขั้นที่แปลงขาวดำกลับเป็นสี (GRAY2BGR แล้ววาดสีแบบ BGR) จึงได้ true ต่อไปเองโดยไม่ต้องมีกฎแยก
// GEN ใดที่ผลขึ้นกับลำดับช่องสี (เช่น ค่าสุ่มต่อช่องของ gaussian_noise, ลำดับที่พิมพ์ของ wiener_filter) ต้องอ่าน flag นี้ผ่าน
// isMirrored(ctx) เท่านั้น — แหล่งความจริงเดียว ห้ามเดาลำดับช่องเองใน GEN
// ⚠️ buildPythonCode2Image (P-21 นอกขอบเขต) ไม่ได้ track flag นี้ → isMirrored คืน true เมื่อไม่มีค่า (= พฤติกรรมเดิมก่อนแก้)
function isMirrored(ctx) {
  return !ctx || ctx.mirrored !== false;
}

// IP batch 2 ชิ้น D — ตัวช่วยของเครื่องมือวัด (feret / line_profile / roi_polygon) อ่านลำดับช่องผ่าน isMirrored อย่างเดียว
// สีที่วาด: แอปให้ค่าเป็นลำดับช่องของแอป (R,G,B) → tuple ของ Python กลับด้าน (B,G,R) เมื่อช่องของ cv2 กลับด้านอยู่ · หลัง Color Space ไม่กลับ
// (ภาพที่ได้จึงตรงกับแอปทุก byte ในหน่วยความจำ — ขั้นถัดไปเห็นค่าเดียวกัน)
function pyColor(ctx, rgb) {
  return `(${(isMirrored(ctx) ? [...rgb].reverse() : rgb).join(", ")})`;
}
// สี → ขาวดำภายใน op (แอปใช้ RGB2GRAY กับช่องตามลำดับของแอปเสมอ) — ช่องกลับด้าน = BGR2GRAY · หลัง Color Space = RGB2GRAY
function pyGrayCode(ctx) {
  return isMirrored(ctx) ? "cv2.COLOR_BGR2GRAY" : "cv2.COLOR_RGB2GRAY";
}
// คัด + เรียงวัตถุชุดเดียวกับ GEN.regionprops ทุกตัวอักษรของ sort key (เทสต์ static ปักไว้ — ไม่ refactor GEN.regionprops รอบนี้)
// ต่างกันแค่ cap ไม่มี "0 = ไม่จำกัด" (feret จำกัด 1–500)
function regionCandidatesPy(minArea, cap) {
  return [
    "# คัดตาม minArea + เรียง (area DESC, y ASC, x ASC, cx ASC, cy ASC, tx ASC) — เลขวัตถุตรงกับ Region Properties",
    "cands = []",
    "for c in contours:",
    "    area = cv2.contourArea(c)",
    `    if area < ${minArea}:`,
    "        continue",
    "    x, y, w, h = cv2.boundingRect(c)",
    "    m = cv2.moments(c)",
    "    cx = (x + w / 2) if m['m00'] == 0 else m['m10'] / m['m00']  # m00=0 (วัตถุ 1px) -> จุดกลางกรอบ",
    "    cy = (y + h / 2) if m['m00'] == 0 else m['m01'] / m['m00']",
    "    pts = c.reshape(-1, 2)",
    "    tx = int(pts[pts[:, 1] == y][:, 0].min())  # จุดบนสุด-ซ้ายสุด: ไม่ซ้ำกันระหว่างวัตถุต่างชิ้นโดยโครงสร้าง",
    "    cands.append(((-area, y, x, cx, cy, tx), c, area, cx, cy))",
    "cands.sort(key=lambda t: t[0])",
    "total_found = len(cands)",
    `max_objects = ${cap}`,
    "shown = cands[:max_objects]",
  ];
}
// ทศนิยม 1 ตำแหน่งปัดครึ่งขึ้นแบบ fixed1 ของ registry.js (นิพจน์ Python ใน f-string)
function pyFixed1(expr) {
  return `{np.floor(${expr} * 10 + 0.5) / 10:.1f}`;
}
// % → พิกเซล แบบ pctToPx ของ registry.js: 100% = พิกเซลสุดท้าย · ไม่ติดลบเสมอ int() จึงเท่ากับ floor (ห้าม round())
function pyPctToPx(pct, sizeExpr) {
  return `int(${pct} / 100.0 * (${sizeExpr} - 1) + 0.5)`;
}
const LINE_PROFILE_VIEW_PY = { overlay: "overlay", plot: "plot" };
const ROI_VIEW_PY = { apply: "apply", mask: "mask", overlay: "overlay" };
// ค่าเริ่มต้นของ param อ่านจาก OPS (แหล่งความจริงเดียวกับ UI/registry) — ใช้กับจุด P1..P6 ของ roi_polygon
function paramDefault(opId, key) {
  return OPS[opId].params.find((s) => s.key === key).default;
}

// IP batch 2 ชิ้น E — ตัวช่วยของ slic / bandpass_notch (สเปก E §4–§5) · ตารางตัวเลือก (pickOwn = own-property เท่านั้น)
const SLIC_VIEW_PY = { boundary: "boundary", mean: "mean" };
const BANDPASS_KIND_PY = { bandpass: "bandpass", bandreject: "bandreject", notch: "notch" };
const BANDPASS_SHAPE_PY = { gaussian: "gaussian", ideal: "ideal" };
const BANDPASS_VIEW_PY = { result: "result", spectrum: "spectrum" };
// def slic_labels(feat, w, h, K, m, iters) ซ้อนในสคริปต์ — ลำดับคำนวณ/เงื่อนไขเดียวกับ slicLabels ของ registry.js ทุกขั้น (ผลรวมเป็นเลขจำนวนเต็มใน float64,
// ระยะ float64 ตามลำดับ ((dL² + da²) + db²) + ระยะตำแหน่ง · window ของแต่ละจุดศูนย์กลางตามลำดับ k แล้วเทียบ D < dist แบบ strict · BFS เศษชิ้นเล็กลำดับ ซ้าย บน ขวา ล่าง)
function slicPyLines() {
  return [
    "def slic_labels(feat, w, h, K, m, iters):",
    "    f = feat.astype(np.float64)",
    "    if feat.ndim == 2:",
    "        f = f[:, :, None]",
    "    nch = f.shape[2]",
    "    n_px = w * h",
    "    S = np.sqrt(n_px / K)",
    "    nx = max(1, min(w, int(np.floor(w / S + 0.5))))",
    "    ny = max(1, min(h, int(np.floor(h / S + 0.5))))",
    "    nc = nx * ny",
    "    kl = 100 / 255  # Lab 8-bit: L ถูกยืดเป็น 0..255 จึงคืนเป็นสเกล 0..100 (a, b ไม่ยืด)",
    "    w_s = (m / S) * (m / S)",
    "    jj, ii = np.divmod(np.arange(nc), nx)",
    "    ix = ((2 * ii + 1) * w) // (2 * nx)",
    "    iy = ((2 * jj + 1) * h) // (2 * ny)",
    "    cf = f[iy, ix].astype(np.float64)  # สีของจุดเริ่ม (nc, nch)",
    "    cx, cy = ix.astype(np.float64), iy.astype(np.float64)",
    "    ys, xs = np.mgrid[0:h, 0:w]",
    "    lab = ((ys * ny) // h) * nx + (xs * nx) // w",
    "    xf, yf = xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)",
    "    chans = [f[:, :, c].ravel() for c in range(nch)]",
    "    for _ in range(iters):",
    "        dist = np.full((h, w), np.inf)",
    "        for k in range(nc):",
    "            x0, x1 = max(0, int(np.ceil(cx[k] - S))), min(w - 1, int(np.floor(cx[k] + S)))",
    "            y0, y1 = max(0, int(np.ceil(cy[k] - S))), min(h - 1, int(np.floor(cy[k] + S)))",
    "            if x0 > x1 or y0 > y1:",
    "                continue",
    "            win = f[y0:y1 + 1, x0:x1 + 1]",
    "            dx = (np.arange(x0, x1 + 1) - cx[k])[None, :]",
    "            dy = (np.arange(y0, y1 + 1) - cy[k])[:, None]",
    "            dl = (win[:, :, 0] - cf[k, 0]) * kl",
    "            if nch == 3:",
    "                da = win[:, :, 1] - cf[k, 1]",
    "                db = win[:, :, 2] - cf[k, 2]",
    "                d = dl * dl + da * da + db * db + (dx * dx + dy * dy) * w_s",
    "            else:",
    "                d = dl * dl + (dx * dx + dy * dy) * w_s",
    "            sub = dist[y0:y1 + 1, x0:x1 + 1]",
    "            sel = d < sub",
    "            sub[sel] = d[sel]",
    "            lab[y0:y1 + 1, x0:x1 + 1][sel] = k",
    "        lf = lab.ravel()",
    "        cnt = np.bincount(lf, minlength=nc).astype(np.float64)",
    "        ok = cnt > 0",
    "        cx[ok] = np.bincount(lf, weights=xf, minlength=nc)[ok] / cnt[ok]",
    "        cy[ok] = np.bincount(lf, weights=yf, minlength=nc)[ok] / cnt[ok]",
    "        for c in range(nch):",
    "            cf[ok, c] = np.bincount(lf, weights=chans[c], minlength=nc)[ok] / cnt[ok]",
    "    # รวมเศษชิ้นเล็ก: ลำดับ raster, เพื่อนบ้านลำดับ ซ้าย บน ขวา ล่าง",
    "    old_l = lab.ravel().tolist()",
    "    out = [-1] * n_px",
    "    limit = (n_px // nc) >> 2",
    "    cur, merged = 0, 0",
    "    for i0 in range(n_px):",
    "        if out[i0] >= 0:",
    "            continue",
    "        old = old_l[i0]",
    "        px0, py0 = i0 % w, i0 // w",
    "        adj = -1",
    "        if px0 > 0 and out[i0 - 1] >= 0:",
    "            adj = out[i0 - 1]",
    "        elif py0 > 0 and out[i0 - w] >= 0:",
    "            adj = out[i0 - w]",
    "        elif px0 < w - 1 and out[i0 + 1] >= 0:",
    "            adj = out[i0 + 1]",
    "        elif py0 < h - 1 and out[i0 + w] >= 0:",
    "            adj = out[i0 + w]",
    "        out[i0] = cur",
    "        queue = [i0]",
    "        head = 0",
    "        while head < len(queue):",
    "            i = queue[head]",
    "            head += 1",
    "            px, py = i % w, i // w",
    "            if px > 0 and out[i - 1] < 0 and old_l[i - 1] == old:",
    "                out[i - 1] = cur",
    "                queue.append(i - 1)",
    "            if py > 0 and out[i - w] < 0 and old_l[i - w] == old:",
    "                out[i - w] = cur",
    "                queue.append(i - w)",
    "            if px < w - 1 and out[i + 1] < 0 and old_l[i + 1] == old:",
    "                out[i + 1] = cur",
    "                queue.append(i + 1)",
    "            if py < h - 1 and out[i + w] < 0 and old_l[i + w] == old:",
    "                out[i + w] = cur",
    "                queue.append(i + w)",
    "        if len(queue) <= limit and adj >= 0:",
    "            for i in queue:",
    "                out[i] = adj",
    "            merged += 1",
    "        else:",
    "            cur += 1",
    "    return np.array(out, dtype=np.int64).reshape(h, w), cur, nx, ny, merged",
  ];
}
// เกน H ของ bandpass_notch (นิพจน์ข้อความเดียวกับ bandGain ของ registry.js) — ต้องมี fx (1, P) และ fy (Q, 1) แล้ว · ออกโค้ดเฉพาะชนิด/รูปแบบที่เลือก
function bandGainPyLines(kind, shape, prm) {
  if (kind === "notch") {
    return [
      `nx, ny, nr = ${prm.nx}, ${prm.ny}, ${prm.nr}  # % ของ Nyquist (ny บวก = ต่ำกว่าจุดกลาง)`,
      "a, b = fx * 100 - nx, fy * 100 - ny",
      "c, d = fx * 100 + nx, fy * 100 + ny",
      "d1, d2, r2 = a * a + b * b, c * c + d * d, nr * nr",
      shape === "ideal" ? "H = np.where((d1 <= r2) | (d2 <= r2), 0.0, 1.0)" : "H = (1 - np.exp(-d1 / (2 * r2))) * (1 - np.exp(-d2 / (2 * r2)))",
    ];
  }
  return [
    `lo, hi = ${prm.lo}, ${prm.hi}  # % ของ Nyquist`,
    "r = np.sqrt(fx * fx + fy * fy) * 100",
    shape === "ideal"
      ? "bp = ((r >= lo) & (r <= hi)).astype(np.float64)"
      : "bp = (1 - np.exp(-(r * r) / (2 * lo * lo)) if lo > 0 else 1) * np.exp(-(r * r) / (2 * hi * hi))",
    kind === "bandreject" ? "H = 1 - bp" : "H = bp",
  ];
}

const GEN = {
  // ---- ปริภูมิสี ----
  grayscale: { needs: "any", out: "gray", gen: (p, ctx) =>
    ctx.gray ? ["# ภาพเป็นขาวดำอยู่แล้ว"] : ["img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"] },

  colorspace: { needs: "any", out: "color", appChannelOrder: true, gen: (p, ctx) => {
    const code = { hsv: "COLOR_BGR2HSV", hls: "COLOR_BGR2HLS", lab: "COLOR_BGR2Lab", ycrcb: "COLOR_BGR2YCrCb", xyz: "COLOR_BGR2XYZ", luv: "COLOR_BGR2Luv", yuv: "COLOR_BGR2YUV" }[p.space] || "COLOR_BGR2HSV";
    const lines = ctx.gray ? ["img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)"] : [];
    lines.push(`img = cv2.cvtColor(img, cv2.${code})  # ดูข้อมูลในปริภูมิสี ${p.space}`);
    return lines;
  } },

  channel_split: { needs: "any", out: "gray", gen: (p, ctx) => {
    const isHSV = ["h", "s", "v"].includes(p.channel);
    const idx = { r: 2, g: 1, b: 0, h: 0, s: 1, v: 2 }[p.channel]; // BGR: b=0,g=1,r=2
    const lines = [];
    if (ctx.gray) lines.push("img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)");
    if (isHSV) lines.push("img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)");
    lines.push(`img = cv2.split(img)[${idx}]  # ดูช่อง ${String(p.channel).toUpperCase()}`);
    return lines;
  } },

  rgb_adjust: { needs: "any", out: "color", gen: (p, ctx) => {
    const lines = [];
    if (ctx.gray) lines.push("img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)");
    // OpenCV เก็บภาพเป็น BGR ไม่ใช่ RGB — b=0, g=1, r=2
    lines.push("b, g, r = cv2.split(img)");
    // ใช้ np.clip แทน cv2.convertScaleAbs — convertScaleAbs หาค่าสัมบูรณ์ก่อนตัด ทำให้ offset ติดลบ
    // "สะท้อนค่ากลับ" แทนที่จะทำให้มืดลงจริง (ต้องตรงกับ Mat.convertTo ฝั่งแอป)
    lines.push(`r = np.clip(r.astype(np.float32) * ${num(p.rGain)} + ${int(p.rOffset)}, 0, 255).astype(np.uint8)`);
    lines.push(`g = np.clip(g.astype(np.float32) * ${num(p.gGain)} + ${int(p.gOffset)}, 0, 255).astype(np.uint8)`);
    lines.push(`b = np.clip(b.astype(np.float32) * ${num(p.bGain)} + ${int(p.bOffset)}, 0, 255).astype(np.uint8)`);
    lines.push("img = cv2.merge([b, g, r])");
    return lines;
  } },

  // ---- ปรับปรุงภาพ ----
  brightness_contrast: { needs: "any", out: "same", gen: (p) =>
    [`img = cv2.convertScaleAbs(img, alpha=${num(p.alpha)}, beta=${int(p.beta)})`] },

  gamma: { needs: "any", out: "same", gen: (p) => p.curve === "log" ? [
    // IP batch 4A — Log transform: ปัดครึ่งขึ้นด้วย np.floor(x + 0.5) เหมือน Math.floor(... + 0.5) ของแอป
    "lut = np.floor(np.log1p(np.arange(256)) * (255 / np.log(256)) + 0.5).astype(np.uint8)",
    "img = cv2.LUT(img, lut)",
    "print(\"log transform: s = 255*ln(1+r)/ln(256)\")",
  ] : [
    `inv_gamma = 1.0 / ${num(p.g) || 1}`,
    // B2 (backlog 2026-10-17): ปัดครึ่งขึ้นเหมือน Math.round ของแอป (เดิม astype ตัดทิ้ง → ต่าง 1 ระดับ)
    "lut = np.minimum(255, np.floor(np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]) + 0.5)).astype(np.uint8)",
    "img = cv2.LUT(img, lut)",
  ] },

  hist_equalize: { needs: "gray", out: "gray", gen: () => ["img = cv2.equalizeHist(img)"] },

  clahe: { needs: "gray", out: "gray", gen: (p) => [
    `clahe = cv2.createCLAHE(clipLimit=${num(p.clip)}, tileGridSize=(${int(p.tile)}, ${int(p.tile)}))`,
    "img = clahe.apply(img)",
  ] },

  // ---- Contrast Stretch / Flat-field Correction (IP batch 2026-10-01, ENHANCE) ----
  // ทั้ง 2 ตัว export ได้จริง — ค่าที่ฝังต้อง clamp ด้วยช่วงเดียวกับ registry.js เป๊ะ (clampNum) เหมือน
  // prewitt/roberts/log_edge/region_growing ของรอบก่อน
  contrast_stretch: { needs: "gray", out: "gray", gen: (p) => {
    const lowPct = clampNum(p.lowPct, 1, 0, 10);
    const highPct = clampNum(p.highPct, 1, 0, 10);
    return [
      "hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().astype(np.float64)  # float64 เหมือน JS — NumPy 2 ให้ float32 + float ยังเป็น float32 ทำให้เทียบ cum กับ count เพี้ยน",
      "total = img.shape[0] * img.shape[1]",
      `low_count = total * ${lowPct} / 100`,
      `high_count = total * (100 - ${highPct}) / 100`,
      "cum = 0.0",
      "low, high = 0, 255",
      "found_low = False",
      "found_high = False",
      "for i in range(256):",
      "    cum += hist[i]",
      "    if not found_low and cum > low_count:  # strict > เหมือน stretchlim ของ MATLAB และ registry.js",
      "        low, found_low = i, True",
      "    if not found_high and cum >= high_count:",
      "        high, found_high = i, True",
      "        break",
      "if high <= low:",
      '    print("flat: no range to stretch")',
      "else:",
      "    idx = np.arange(256, dtype=np.float64)",
      "    raw = np.floor((idx - low) / (high - low) * 255.0 + 0.5)  # int(x+0.5) ไม่ใช่ round() — ให้ตรง Math.round ของ JS",
      "    lut = np.where(idx <= low, 0, np.where(idx >= high, 255, raw))",
      "    lut = np.clip(lut, 0, 255).astype(np.uint8)",
      "    img = cv2.LUT(img, lut)",
      '    print(f"stretched [{low}, {high}] -> [0, 255]")',
    ];
  } },

  // IP batch 3 — LUT จำนวนเต็มล้วน สูตรเดียวกับ quantizeLut ของ registry.js (ไม่ใช้ round/** 0.5) → ผลเท่าแอปทุก byte
  quantize: { needs: "any", out: "same", gen: (p) => {
    const N = Math.round(clampNum(p.levels, 4, 2, 256));
    let bits = 0;
    while ((1 << bits) < N) bits++;
    return [
      `levels = ${N}`,
      "q_lut = ((((np.arange(256) * levels) // 256) * 510 + (levels - 1)) // (2 * (levels - 1))).astype(np.uint8)",
      "img = cv2.LUT(img, q_lut)",
      `print(f"quantize: {levels} levels per channel (${bits} bit)")`,
    ];
  } },

  flatfield_correction: { needs: "any", out: "same", gen: (p) => {
    const k = odd(clampNum(p.blurSize, 121, 15, 401));
    return [
      `k = ${k}`,
      // ย่อภาพก่อนประมาณพื้นหลัง แล้วขยายกลับ (ตรงกับ registry.js เป๊ะ — ดูคอมเมนต์เต็ม ๆ ที่นั่นว่าทำไม
      // ย่อได้โดยแทบไม่เสียความแม่นยำ) — scale ต้องคำนวณตอนรันไทม์ (ไม่รู้ขนาดภาพตอน gen โค้ด) เหมือน
      // skeleton ด้านบน int(x + 0.5) แทน round()/Math.round ปกติ เพื่อให้ปัดเศษตรงกับฝั่ง JS เป๊ะ
      "rows, cols = img.shape[:2]",
      "scale = min(1.0, 240 / max(rows, cols))",
      "if scale < 1:",
      "    aw, ah = max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))",
      "    analysis = cv2.resize(img, (aw, ah), interpolation=cv2.INTER_AREA)",
      "    bg_k = max(3, int(k * scale + 0.5))",
      "    if bg_k % 2 == 0:",
      "        bg_k += 1",
      "    bg_small = cv2.GaussianBlur(analysis, (bg_k, bg_k), 0)",
      "    bg = cv2.resize(bg_small, (cols, rows), interpolation=cv2.INTER_LINEAR)",
      "else:",
      "    bg = cv2.GaussianBlur(img, (k, k), 0)",
      "mean_val = cv2.mean(bg)",
      "src_f = img.astype(np.float32)",
      "bg_f = bg.astype(np.float32)",
      "bg_eps = bg_f + 1.0  # กันหารศูนย์ (พื้นหลังมืดสนิทบางจุด) — ต้องตรงกับ epsilon ฝั่งแอปเป๊ะ",
      "ratio = src_f / bg_eps",
      "if img.ndim == 3:",
      "    scale_arr = np.array(mean_val[:img.shape[2]], dtype=np.float32)",
      "    scaled = ratio * scale_arr",
      "else:",
      "    scaled = ratio * mean_val[0]",
      // scaled ไม่มีทางติดลบ (src/bg_eps >= 0 เสมอ) convertScaleAbs จึงแค่ saturate+ปัดเศษ ตรงกับ
      // Mat.convertTo ฝั่งแอป (ทั้งคู่ใช้ saturate_cast แบบ ties-to-even เดียวกัน — ยืนยันแล้วว่าตรงกันเป๊ะ)
      "img = cv2.convertScaleAbs(scaled)",
    ];
  } },

  unsharp: { needs: "any", out: "same", gen: (p) => {
    const k = odd(p.radius);
    return [
      `blur = cv2.GaussianBlur(img, (${k}, ${k}), 0)`,
      `img = cv2.addWeighted(img, ${1 + num(p.amount)}, blur, ${-num(p.amount)}, 0)`,
    ];
  } },

  laplacian_sharpen: { needs: "any", out: "same", gen: (p) => [
    "lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)",
    "lap = cv2.convertScaleAbs(lap)",
    `img = cv2.addWeighted(img, 1, lap, ${-num(p.strength)}, 0)`,
  ] },

  custom_kernel: { needs: "any", out: "same", gen: (p) => {
    const PRESETS = {
      identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
      sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
      edge: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
      emboss: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
      outline: [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
      box3: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
      box5: Array.from({ length: 5 }, () => new Array(5).fill(1 / 25)),
    };
    const amount = Number.isFinite(Number(p.amount)) ? Number(p.amount) : 1;
    const psfKind = p.preset === "motion" || p.preset === "disk" ? p.preset : null;
    const lines = [];
    if (psfKind) {
      // IP batch 4A — Motion/Disk: PSF จาก psf_kernel (ฝาแฝดของ psfKernel ในแอป)
      const L = Math.round(clampNum(p.psfSize, 9, 1, 50)), ang = Math.round(clampNum(p.psfAngle, 0, 0, 179));
      lines.push(
        ...psfKernelPyLines(),
        `kernel = psf_kernel("${psfKind}", ${L}, ${ang}).astype(np.float32)`,
        "filtered = cv2.filter2D(img, -1, kernel)",
      );
    } else {
      const k = pickOwn(PRESETS, p.preset, "sharpen");
      lines.push(
        `kernel = np.array(${JSON.stringify(k)}, dtype=np.float32)`,
        "filtered = cv2.filter2D(img, -1, kernel)",
      );
    }
    if (amount === 1) lines.push("img = filtered");
    else lines.push(`img = cv2.addWeighted(img, ${1 - amount}, filtered, ${amount}, 0)`);
    return lines;
  } },

  // ---- ฟื้นฟูภาพ ----
  salt_pepper_noise: { needs: "any", out: "same", gen: (p) => [
    `# ใส่ noise เกลือ-พริกไทย ${int(p.amount)}%`,
    `n = int(img.shape[0] * img.shape[1] * ${num(p.amount) / 100})`,
    "ys = np.random.randint(0, img.shape[0], n)",
    "xs = np.random.randint(0, img.shape[1], n)",
    "vals = np.where(np.random.rand(n) < 0.5, 0, 255)",
    "img[ys, xs] = vals[:, None] if img.ndim == 3 else vals",
  ] },

  // IP batch 2 ชิ้น C — ค่าที่ฝัง clamp ช่วงเดียวกับ registry.js (numIn/intIn) · ได้ผลเท่าแอปทุก byte (PRNG ตัวเดียวกัน)
  gaussian_noise: { needs: "any", out: "same", gen: (p, ctx) => {
    const sigma = clampNum(p.sigma, 20, 0, 100);
    const seed = Math.round(clampNum(p.seed, 1, 1, 99999));
    if (sigma === 0) return ["# σ = 0 → ไม่เติม noise (ภาพเดิม)", 'print("gaussian noise: sigma = 0, image unchanged")'];
    const speckle = p.kind === "speckle";
    return [
      `# noise แบบเกาส์ σ=${sigma} seed=${seed} — ลำดับการสุ่มเดียวกับในแอป จึงได้ลายเดิมทุกครั้งที่รัน`,
      ...seededGaussianPy("img.size", seed),
      "if img.ndim == 3:",
      isMirrored(ctx)
        ? "    z = z.reshape(img.shape[0], img.shape[1], 3)[:, :, ::-1]  # แอปเรียงค่าสุ่มตามช่อง R,G,B แต่ cv2 เก็บภาพเป็น B,G,R"
        : "    z = z.reshape(img.shape[0], img.shape[1], 3)  # หลัง Color Space ช่องของ cv2 เรียงเหมือนแอปแล้ว จึงไม่กลับลำดับ",
      "else:",
      "    z = z.reshape(img.shape[0], img.shape[1])",
      speckle
        ? `noisy = np.floor(img.astype(np.float64) + img.astype(np.float64) * ${String(sigma / 255)} * z + 0.5)  # speckle: เดิม + เดิม x (sigma/255) x z`
        : `noisy = np.floor(img.astype(np.float64) + ${sigma} * z + 0.5)`,
      "clipped = int(np.count_nonzero((noisy < 0) | (noisy > 255)))  # ค่าที่เกิน 0..255 แล้วถูกตัดที่ขอบ",
      "img = np.clip(noisy, 0, 255).astype(np.uint8)",
      speckle
        ? `print(f"speckle noise sigma=${sigma} (seed ${seed}): clipped at 0/255 {np.floor(clipped * 100 / noisy.size * 10 + 0.5) / 10:.1f}% of values")`
        : `print(f"gaussian noise sigma=${sigma} (seed ${seed}): clipped at 0/255 {np.floor(clipped * 100 / noisy.size * 10 + 0.5) / 10:.1f}% of values")`,
    ];
  } },

  median_blur: { needs: "any", out: "same", gen: (p) => [`img = cv2.medianBlur(img, ${odd(p.ksize)})`] },
  gaussian_blur: { needs: "any", out: "same", gen: (p) => { const k = odd(p.ksize); return [`img = cv2.GaussianBlur(img, (${k}, ${k}), 0)`]; } },
  box_blur: { needs: "any", out: "same", gen: (p) => { const k = Math.max(1, int(p.ksize)); return [`img = cv2.blur(img, (${k}, ${k}))`]; } },
  bilateral: { needs: "any", out: "same", gen: (p) =>
    [`img = cv2.bilateralFilter(img, ${int(p.d)}, ${int(p.sigmaColor)}, ${int(p.sigmaSpace)})`] },

  // IP batch 2 ชิ้น C — wiener2 แยกทีละช่อง · ลำดับการคำนวณเดียวกับ registry.js ทุกขั้น (ผลรวม float32 → float64, cumsum
  // = บวกตามลำดับเหมือน JS) จึงได้ผลเท่าแอปทุก byte · ksize clamp 3..15 แล้วบังคับคี่เหมือน ensureOdd(intIn(...))
  wiener_filter: { needs: "any", out: "same", gen: (p, ctx) => {
    const k = odd(clampNum(p.ksize, 5, 3, 15));
    const noiseVar = clampNum(p.noiseVar, 0, 0, 10000);
    const box = "(k, k), normalize=False, borderType=cv2.BORDER_REFLECT_101";
    return [
      `k = ${k}`,
      "n = k * k",
      `s1 = cv2.boxFilter(img, cv2.CV_32F, ${box}).astype(np.float64)  # ผลรวมในหน้าต่าง (float32 เป็นจำนวนเต็มแม่นเป๊ะเมื่อ k <= 15)`,
      "f32 = img.astype(np.float32)",
      `s2 = cv2.boxFilter(f32 * f32, -1, ${box}).astype(np.float64)  # ผลรวมกำลังสองในหน้าต่าง`,
      "mu = s1 / n  # ค่าเฉลี่ยท้องถิ่น",
      "var = s2 / n - mu * mu  # variance ท้องถิ่น",
      ...(noiseVar > 0
        ? [`noise = ${noiseVar}  # กำหนดเอง`]
        : [
          "# ประมาณ noise = ค่าเฉลี่ยของ variance ท้องถิ่นทั้งภาพ แยกทีละช่อง (cumsum = บวกตามลำดับเหมือนแอป)",
          "if img.ndim == 3:",
          "    noise = np.array([np.cumsum(var[:, :, c].ravel())[-1] / (img.shape[0] * img.shape[1]) for c in range(img.shape[2])])",
          "else:",
          "    noise = np.cumsum(var.ravel())[-1] / (img.shape[0] * img.shape[1])",
        ]),
      "den = np.maximum(var, noise)",
      "gain = np.maximum(var - noise, 0)",
      'with np.errstate(invalid="ignore", divide="ignore"):  # den = 0 (บริเวณเรียบสนิท) ใช้ค่าเฉลี่ยแทน',
      "    out = np.where(den > 0, mu + (img.astype(np.float64) - mu) / den * gain, mu)",
      "img = np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)",
      ...(noiseVar > 0
        ? [`print("wiener ${k}x${k}: noise variance = ${noiseVar} (given)")`]
        : isMirrored(ctx)
          ? [
            'noise_txt = "/".join(f"{np.floor(v * 10 + 0.5) / 10:.1f}" for v in np.atleast_1d(noise)[::-1])  # cv2 เก็บ B,G,R → แสดงเป็น R/G/B เหมือนแอป',
            `print(f"wiener ${k}x${k}: noise variance = {noise_txt} (estimated)")`,
          ]
          : [
            // หลัง Color Space ช่องไม่ใช่ R/G/B แล้ว — พิมพ์ตามลำดับช่องของภาพ (ตรงกับลำดับในข้อความของแอป) และไม่อ้างว่าเป็น R/G/B
            'noise_txt = "/".join(f"{np.floor(v * 10 + 0.5) / 10:.1f}" for v in np.atleast_1d(noise))  # ลำดับช่องเดียวกับแอป (หลัง Color Space)',
            `print(f"wiener ${k}x${k}: noise variance per channel (image order) = {noise_txt} (estimated)")`,
          ]),
    ];
  } },

  // ---- สัณฐานวิทยา ----
  // outOf แทน out คงที่ เพราะ endpoints_branchpoints (ใหม่ 2026-10-01) คืนภาพสี (BGR) เสมอ ส่วน op อื่น
  // ทั้งหมดในกลุ่มนี้คืนภาพขาวดำ — ตามแนวเดียวกับ region_growing ที่ out ขึ้นกับ view
  morphology: { needs: "gray", outOf: (p) => (p.op === "endpoints_branchpoints" ? "color" : "gray"), gen: (p) => {
    // IP batch 4B — ตัวเลือกใหม่ (ไม่ใช้ shape/ksize/iterations) · path เดิมทุกตัวด้านล่างไม่เปลี่ยน
    if (p.op === "hit_or_miss") {
      const patKey = Object.hasOwn(HITMISS_PATTERNS, p.hmPattern) ? p.hmPattern : "isolated";
      const rotate = boolOr(p.hmRotate, true);
      return [
        `# Hit-or-Miss แม่แบบ ${patKey} (1 = ต้องขาว, -1 = ต้องดำ, 0 = ไม่สน) · นอกขอบภาพ OpenCV นับเป็น "ไม่สน"`,
        "_, b = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)",
        `k = np.array(${JSON.stringify(HITMISS_PATTERNS[patKey].k)}, np.int32)`,
        "ks = [k]",
        ...(rotate ? ["for _ in range(3):", "    ks.append(np.rot90(ks[-1], -1))  # หมุนตามเข็มนาฬิกาทีละ 90 องศา"] : []),
        "out = np.zeros_like(b)",
        "for kk in ks:",
        "    out = np.bitwise_or(out, cv2.morphologyEx(b, cv2.MORPH_HITMISS, np.ascontiguousarray(kk)))",
        `print(f"hit-or-miss ${patKey}: {int(np.count_nonzero(out))} matches")`,
        "img = out",
      ];
    }
    if (p.op === "regional_max" || p.op === "regional_min" || p.op === "h_maxima" || p.op === "extended_max") {
      const h = Math.round(clampNum(p.hHeight, 10, 1, 255));
      return [
        "# reconstruction by dilation ตามนิยาม (ช้ากว่าของแอปที่เป็น O(N) แต่ได้ผลเท่ากันเป๊ะ เพราะคำตอบมีหนึ่งเดียว) · ใช้ภาพเทา ไม่ตัด 127",
        "def recon(marker, mask):",
        "    m = marker.astype(np.int16)",
        "    k3 = np.ones((3, 3), np.uint8)",
        "    while True:",
        "        nxt = np.minimum(cv2.dilate(m, k3), mask)",
        "        if np.array_equal(nxt, m):",
        "            return m",
        "        m = nxt",
        "def regional_max(a):",
        "    return np.where(a - recon(a - 1, a) >= 1, 255, 0).astype(np.uint8)",
        "i16 = img.astype(np.int16)",
        ...(p.op === "regional_max"
          ? ["out = regional_max(i16)", "n_cc, _ = cv2.connectedComponents(out, connectivity=8)", 'print(f"regional maxima: {n_cc - 1} regions ({int(np.count_nonzero(out))} px)")']
          : p.op === "regional_min"
            ? ["out = regional_max(255 - i16)", "n_cc, _ = cv2.connectedComponents(out, connectivity=8)", 'print(f"regional minima: {n_cc - 1} regions ({int(np.count_nonzero(out))} px)")']
            : p.op === "h_maxima"
              ? [`hm = recon(np.maximum(i16 - ${h}, 0), i16)`, "out = hm.astype(np.uint8)", "n_cc, _ = cv2.connectedComponents(regional_max(hm), connectivity=8)", `print(f"h-maxima h=${h}: {n_cc - 1} regions left")`]
              : [`hm = recon(np.maximum(i16 - ${h}, 0), i16)`, "out = regional_max(hm)", "n_cc, _ = cv2.connectedComponents(out, connectivity=8)", `print(f"extended maxima h=${h}: {n_cc - 1} regions ({int(np.count_nonzero(out))} px)")`]),
        "img = out",
      ];
    }
    if (p.op === "fill_holes") {
      // imfill: เติมเฉพาะรูที่ปิดสนิท (ไม่ต่อกับขอบภาพ) — เติมขอบ 0 หนา 1px รอบภาพก่อน flood จาก (0,0)
      // เพื่อให้พื้นหลังทุกจุดที่ชนขอบเดิม (แม้วัตถุจะชนขอบภาพ) ต่อกันเป็นก้อนเดียวกับจุดเริ่ม flood เสมอ
      return [
        "_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # binarize ก่อนเสมอ (ถ้าเป็นขาว-ดำอยู่แล้วค่าไม่เปลี่ยน)",
        "pad = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)",
        "filled = pad.copy()",
        "ff_mask = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)",
        "cv2.floodFill(filled, ff_mask, (0, 0), 255)",
        "holes = cv2.bitwise_not(filled)[1:-1, 1:-1]  # crop กลับขนาดเดิม",
        "img = cv2.bitwise_or(img, holes)  # เติมรูที่ปิดสนิทในวัตถุ",
      ];
    }
    if (p.op === "clear_border") {
      return [
        // binarize ก่อนเสมอ (เหมือน fill_holes ด้านบน) — connectedComponentsWithStats นับพิกเซล
        // nonzero ใด ๆ เป็น foreground เดียวกัน ถ้าไม่ binarize ก่อน ภาพ gray ทั่วไปจะกลายเป็น
        // "วัตถุ" เดียวทั้งเฟรม (แทบทุกจุด nonzero) แล้วโดนลบทิ้งหมดเพราะแตะขอบแน่นอน
        "_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)",
        "n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)",
        "h, w = img.shape[:2]",
        // เวกเตอไรซ์ด้วย stats ตรง ๆ (ไม่วน out[labels==i] ทีละชิ้น) — เร็วขึ้นมากบนภาพที่มีหลายหมื่น component
        "left, top, ww, hh = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]",
        "touches_border = (left == 0) | (top == 0) | (left + ww == w) | (top + hh == h)",
        "keep = np.zeros(n, dtype=bool)",
        "keep[1:] = ~touches_border[1:]  # ข้ามป้าย 0 (พื้นหลัง)",
        "img = np.where(keep[labels], 255, 0).astype(np.uint8)",
      ];
    }
    if (p.op === "remove_small") {
      const minArea = Number.isFinite(Number(p.minArea)) ? int(p.minArea) : 150;
      return [
        // binarize ก่อนเสมอ ด้วยเหตุผลเดียวกับ clear_border ด้านบน
        "_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)",
        "n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)",
        `min_area = ${minArea}`,
        // เวกเตอไรซ์เหมือน clear_border ด้านบน
        "keep = np.zeros(n, dtype=bool)",
        "keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area",
        "img = np.where(keep[labels], 255, 0).astype(np.uint8)",
      ];
    }
    if (p.op === "skeleton") {
      // Zhang-Suen thinning — พอร์ตจาก JS ตรง ๆ (ไม่ vectorize ต่าง จากเหตุผลเดียวกับ multi_otsu_threshold)
      // ย่อภาพวิเคราะห์ ≤240px เหมือนแอปตัวจริงเสมอ (int(x+0.5) ไม่ใช่ round() — กติกาการปัดเลขรันไทม์)
      //
      // สำคัญ: ทดสอบเปรียบเทียบกับ JS จริงแล้วพบว่า np.roll-based vectorization (ตามที่เคยลองร่างไว้)
      // มี 2 บั๊กที่ทำให้ผลต่างจาก JS มาก ไม่ใช่แค่ผลกระทบเล็ก ๆ ที่ขอบภาพ: (1) np.roll wrap-around ทำให้
      // พิกเซลที่ขอบภาพ "เห็น" พิกเซลจากขอบตรงข้ามเป็นเพื่อนบ้าน (ต้อง zero ค่าที่ wrap เข้ามาในทุกทิศ)
      // (2) JS วน y=1..h-2, x=1..w-2 เท่านั้น (ไม่แตะพิกเซลที่ขอบภาพนอกสุด 1px เลย) แต่เวอร์ชัน vectorized
      // เดิมคำนวณทั้งภาพรวมขอบ ทำให้ลบพิกเซลขอบภาพที่ JS ไม่ลบ — ยืนยันด้วย diff เป็น 0 พิกเซลบนภาพทดสอบ
      // 3 แบบ (สี่เหลี่ยมแตะขอบ, สี่เหลี่ยมไม่แตะขอบ, กากบาทแตะขอบทั้ง 4 ด้าน) หลังแก้ 2 จุดนี้แล้วเท่านั้น
      // จึงพอร์ตแบบ direct loop (ไม่ vectorize) แทน เพื่อความถูกต้อง ไม่ใช่ความเร็ว (เหมือน multi-Otsu)
      return [
        "_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)",
        "rows, cols = bin_img.shape",
        "scale = min(1.0, 240 / max(rows, cols))",
        "if scale < 1:",
        "    analysis = cv2.resize(bin_img, (max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))), interpolation=cv2.INTER_NEAREST)",
        "else:",
        "    analysis = bin_img",
        "sk = (analysis > 0).astype(np.uint8)",
        "a_h, a_w = sk.shape",
        "def _p(x, y, dx, dy):",
        "    xx, yy = x + dx, y + dy",
        "    if xx < 0 or xx >= a_w or yy < 0 or yy >= a_h:",
        "        return 0",
        "    return int(sk[yy, xx])",
        "changed = True",
        "iterations = 0",
        "while changed and iterations < 100:",
        "    changed = False",
        "    iterations += 1",
        "    for step in (1, 2):",
        "        to_delete = []",
        "        for y in range(1, a_h - 1):",
        "            for x in range(1, a_w - 1):",
        "                if sk[y, x] != 1:",
        "                    continue",
        "                p2, p3, p4, p5 = _p(x, y, 0, -1), _p(x, y, 1, -1), _p(x, y, 1, 0), _p(x, y, 1, 1)",
        "                p6, p7, p8, p9 = _p(x, y, 0, 1), _p(x, y, -1, 1), _p(x, y, -1, 0), _p(x, y, -1, -1)",
        "                neigh = [p2, p3, p4, p5, p6, p7, p8, p9]",
        "                b = sum(neigh)",
        "                if b < 2 or b > 6:",
        "                    continue",
        "                a = sum(1 for i in range(8) if neigh[i] == 0 and neigh[(i + 1) % 8] == 1)",
        "                if a != 1:",
        "                    continue",
        "                if step == 1:",
        "                    if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:",
        "                        continue",
        "                else:",
        "                    if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:",
        "                        continue",
        "                to_delete.append((y, x))",
        "        if to_delete:",
        "            changed = True",
        "            for (y, x) in to_delete:",
        "                sk[y, x] = 0",
        "sk = (sk * 255).astype(np.uint8)",
        "img = cv2.resize(sk, (cols, rows), interpolation=cv2.INTER_NEAREST) if scale < 1 else sk",
      ];
    }
    if (p.op === "reconstruct_open") {
      // ตรงกับ registry.js: คำนวณผลลู่เข้าจริงแบบ O(N) ด้วย connected components แทนการวน dilate+min
      // จนกว่าจะนิ่ง (ของเดิมจำกัด 300 รอบ ไม่พอสำหรับรอยร้าวยาว) — ดูคำอธิบายพิสูจน์เต็ม ๆ ใน registry.js
      const k = Math.max(1, int(p.ksize) || 5);
      const erosionIters = Math.max(1, int(p.iterations) || 1);
      return [
        "_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)",
        `kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (${k}, ${k}))`,
        `marker = cv2.erode(bin_img, kernel, iterations=${erosionIters})`,
        "n, labels = cv2.connectedComponents(bin_img, connectivity=8)",
        "keep = np.zeros(n, dtype=bool)",
        "keep[np.unique(labels[marker > 0])] = True",
        "keep[0] = False  # พื้นหลัง ไม่เก็บ",
        "img = np.where(keep[labels], 255, 0).astype(np.uint8)",
      ];
    }
    if (p.op === "convex_hull") {
      // bwconvhull: ตรงกับ registry.js เป๊ะ — binarize → findContours (RETR_EXTERNAL) → ห่อแต่ละชิ้น
      // ด้วย convexHull → วาดลง mask ว่าง → รายงาน solidity รวม (Σพื้นที่เดิม/Σพื้นที่รูปนูน)
      return [
        "_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)",
        "contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)",
        "out = np.zeros_like(bin_img)",
        "total_fg_area = 0.0",
        "total_hull_area = 0.0",
        "shape_count = 0",
        // ห่อทุก contour ไม่ข้ามชิ้นไหน (เส้นบาง 1 พิกเซลมีแค่ 2 จุดหลัง CHAIN_APPROX_SIMPLE) — เหมือน registry.js
        "for c in contours:",
        "    hull = cv2.convexHull(c)",
        "    cv2.drawContours(out, [hull], 0, 255, -1)",
        "    total_fg_area += cv2.contourArea(c)",
        "    total_hull_area += cv2.contourArea(hull)",
        "    shape_count += 1",
        "if shape_count == 0:",
        '    print("no objects found")',
        "else:",
        "    solidity = total_fg_area / total_hull_area if total_hull_area > 0 else 1",
        '    print(f"wrapped {shape_count} object(s), area-weighted solidity {solidity:.2f}")',
        "img = out",
      ];
    }
    if (p.op === "endpoints_branchpoints") {
      // bwmorph('endpoints')/('branchpoints') แบบประมาณ — ตรงกับ registry.js เป๊ะ (ดูคอมเมนต์เต็ม ๆ ที่นั่น):
      // endpoint = เพื่อนบ้านขาว 8 ทิศเท่ากับ 1 · branchpoint = Crossing Number (นับช่วงขาว<->ดำสลับกันรอบ
      // วงแหวนเพื่อนบ้าน 8 ทิศ หาร 2) >= 3 -- ไม่ใช่แค่นับจำนวนเพื่อนบ้านตรง ๆ (ของเดิม n>=3 นับจุดแยกเกินจริง
      // รอบจุดตัดจริง -- qa พิสูจน์ด้วยภาพกากบาท) *** ใช้ต่อจาก Skeleton เสมอ ***
      return [
        "_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)",
        "out = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)",
        "if cv2.countNonZero(bin_img) == 0:",
        '    print("no line found (try Skeleton first)")',
        "else:",
        "    hh, ww = bin_img.shape",
        "    endpoints = 0",
        "    branchpoints = 0",
        // ลำดับเพื่อนบ้าน 8 ทิศตามเข็มนาฬิกา (N,NE,E,SE,S,SW,W,NW) คงที่เสมอ — ต้องตรงกับ RING_OFFSETS ของ
        // registry.js เป๊ะ เพื่อให้ Crossing Number คำนวณได้ค่าเดียวกัน
        "    ring_offsets = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]",
        "    for y in range(hh):",
        "        for x in range(ww):",
        "            if bin_img[y, x] == 0:",
        "                continue",
        "            ring = [0] * 8",
        "            n = 0",
        "            for ri, (dx, dy) in enumerate(ring_offsets):",
        "                xx, yy = x + dx, y + dy",
        "                v = 1 if (0 <= xx < ww and 0 <= yy < hh and bin_img[yy, xx] != 0) else 0",
        "                ring[ri] = v",
        "                n += v",
        "            if n == 1:",
        "                endpoints += 1",
        "                cv2.circle(out, (x, y), 3, (0, 255, 0), -1)  # เขียว (BGR) = จุดปลาย",
        "            else:",
        "                transitions = sum(1 for ri in range(8) if ring[ri] != ring[(ri + 1) % 8])",
        "                if transitions // 2 >= 3:",
        "                    branchpoints += 1",
        "                    cv2.circle(out, (x, y), 3, (0, 0, 255), -1)  # แดง (BGR) = จุดแยก",
        '    print(f"endpoints={endpoints} branchpoints={branchpoints}")',
        "img = out",
      ];
    }
    // รูปทรง kernel ที่ cv2.getStructuringElement ไม่มี (diamond/octagon/line) — สร้าง binary mask เอง
    // ด้วย loop ตรง ๆ ให้ตรงกับ buildCustomStrel() ฝั่ง JS ทุกประการ (ดู registry.js)
    const k = Math.max(1, int(p.ksize));
    const it = Math.max(1, int(p.iterations));
    const lines = [];
    if (p.shape === "diamond" || p.shape === "octagon" || p.shape === "line") {
      const r = Math.floor(k / 2);
      const size = 2 * r + 1;
      lines.push(
        `_r = ${r}`,
        `_size = ${size}`,
        "_kernel = np.zeros((_size, _size), np.uint8)",
      );
      if (p.shape === "diamond") {
        lines.push(
          "for _y in range(_size):",
          "    for _x in range(_size):",
          "        if abs(_x - _r) + abs(_y - _r) <= _r:",
          "            _kernel[_y, _x] = 1",
        );
      } else if (p.shape === "octagon") {
        const cut = Math.round(r / 2.5);
        lines.push(
          `_cut = ${cut}`,
          "for _y in range(_size):",
          "    for _x in range(_size):",
          "        _dx, _dy = abs(_x - _r), abs(_y - _r)",
          "        if _dx <= _r and _dy <= _r and _dx + _dy <= 2 * _r - _cut:",
          "            _kernel[_y, _x] = 1",
        );
      } else {
        const angle = ((Number.isFinite(p.lineAngle) ? p.lineAngle : 0) * Math.PI) / 180;
        lines.push(
          `_angle = ${angle}`,
          "for _t in range(-_r, _r + 1):",
          // RULE 7: int(x + 0.5) ไม่ใช่ round() — ให้ตรงกับ Math.round() ฝั่ง JS แม้พิกัดเป็นเลขจำนวนเต็ม
          // ตอนนี้เกือบทั้งหมด (ป้องกันไว้เพื่อความเข้ากันได้ในอนาคตถ้าค่ามุมเปลี่ยนเป็นอื่น)
          "    _x = int(_r + _t * np.cos(_angle) + 0.5)",
          "    _y = int(_r + _t * np.sin(_angle) + 0.5)",
          "    if 0 <= _x < _size and 0 <= _y < _size:",
          "        _kernel[_y, _x] = 1",
        );
      }
      lines.push("kernel = _kernel");
    } else {
      const shape = { rect: "MORPH_RECT", ellipse: "MORPH_ELLIPSE", cross: "MORPH_CROSS" }[p.shape] || "MORPH_RECT";
      lines.push(`kernel = cv2.getStructuringElement(cv2.${shape}, (${k}, ${k}))`);
    }
    if (p.op === "erode") lines.push(`img = cv2.erode(img, kernel, iterations=${it})`);
    else if (p.op === "dilate") lines.push(`img = cv2.dilate(img, kernel, iterations=${it})`);
    else {
      const opc = { open: "MORPH_OPEN", close: "MORPH_CLOSE", gradient: "MORPH_GRADIENT", tophat: "MORPH_TOPHAT", blackhat: "MORPH_BLACKHAT" }[p.op] || "MORPH_OPEN";
      lines.push(`img = cv2.morphologyEx(img, cv2.${opc}, kernel, iterations=${it})`);
    }
    return lines;
  } },

  // ---- แบ่งส่วนภาพ ----
  binary_threshold: { needs: "gray", out: "gray", gen: (p) => {
    const t = boolOr(p.invert, false) ? "cv2.THRESH_BINARY_INV" : "cv2.THRESH_BINARY";
    return [`_, img = cv2.threshold(img, ${int(p.thresh)}, 255, ${t})`];
  } },

  otsu_threshold: { needs: "gray", out: "gray", gen: (p) => {
    const t = boolOr(p.invert, false) ? "cv2.THRESH_BINARY_INV" : "cv2.THRESH_BINARY";
    if (p.method === "triangle") return [`_, img = cv2.threshold(img, 0, 255, ${t} + cv2.THRESH_TRIANGLE)  # หาเกณฑ์อัตโนมัติแบบ Triangle`];
    return [`_, img = cv2.threshold(img, 0, 255, ${t} + cv2.THRESH_OTSU)  # หาเกณฑ์อัตโนมัติ`];
  } },

  adaptive_threshold: { needs: "gray", out: "gray", gen: (p) =>
    [`img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ${Math.max(3, odd(p.blockSize))}, ${int(p.C)})`] },

  multi_otsu_threshold: { needs: "gray", out: "gray", gen: () => [
    "hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().astype(np.float64)  # float64 เหมือน JS — float32 แม่นแค่ 16.7 ล้านพิกเซล (NumPy 2)",
    "P = np.concatenate(([0], np.cumsum(hist)))",
    "S = np.concatenate(([0], np.cumsum(np.arange(256) * hist)))",
    "total = P[256]",
    "mu_t = S[256] / total if total > 0 else 0",
    "best_score, best_t1, best_t2 = -1, 85, 170",
    "for t1 in range(1, 254):",
    "    for t2 in range(t1 + 1, 255):",
    "        w0, w1, w2 = P[t1] - P[0], P[t2] - P[t1], P[256] - P[t2]",
    "        if w0 == 0 or w1 == 0 or w2 == 0:",
    "            continue  # ข้ามเกณฑ์ที่ทำให้กลุ่มใดกลุ่มหนึ่งว่างเปล่า (กันภาพ degenerate)",
    "        m0 = (S[t1] - S[0]) / w0 if w0 > 0 else 0",
    "        m1 = (S[t2] - S[t1]) / w1 if w1 > 0 else 0",
    "        m2 = (S[256] - S[t2]) / w2 if w2 > 0 else 0",
    "        score = w0 * (m0 - mu_t) ** 2 + w1 * (m1 - mu_t) ** 2 + w2 * (m2 - mu_t) ** 2",
    "        if score > best_score:",
    "            best_score, best_t1, best_t2 = score, t1, t2",
    "lut = np.where(np.arange(256) < best_t1, 0, np.where(np.arange(256) < best_t2, 128, 255)).astype(np.uint8)",
    "img = cv2.LUT(img, lut)",
  ] },

  canny: { needs: "gray", out: "gray", gen: (p) => [`img = cv2.Canny(img, ${int(p.t1)}, ${int(p.t2)})`] },

  sobel: { needs: "gray", out: "gray", gen: (p) => {
    const k = odd(p.ksize);
    if (p.dir === "x") return [`img = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=${k}))`];
    if (p.dir === "y") return [`img = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=${k}))`];
    return [
      `gx = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=${k}))`,
      `gy = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=${k}))`,
      "img = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)",
    ];
  } },

  laplacian_edge: { needs: "gray", out: "gray", gen: (p) => {
    const k = odd(p.ksize);
    return [
      `lap = cv2.Laplacian(img, cv2.CV_16S, ksize=${k})`,
      "img = cv2.convertScaleAbs(lap)",
    ];
  } },

  scharr_edge: { needs: "gray", out: "gray", gen: (p) => {
    if (p.dir === "x") return ["img = cv2.convertScaleAbs(cv2.Scharr(img, cv2.CV_16S, 1, 0))"];
    if (p.dir === "y") return ["img = cv2.convertScaleAbs(cv2.Scharr(img, cv2.CV_16S, 0, 1))"];
    return [
      "gx = cv2.convertScaleAbs(cv2.Scharr(img, cv2.CV_16S, 1, 0))",
      "gy = cv2.convertScaleAbs(cv2.Scharr(img, cv2.CV_16S, 0, 1))",
      "img = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)",
    ];
  } },

  // ---- Prewitt / Roberts / LoG / Region Growing (2026-09-30) ----
  // ทั้ง 4 ตัว export ได้จริง (ไม่ใช่ app-only) — ค่าที่ฝังต้อง clamp ด้วยช่วงเดียวกับ registry.js เป๊ะ
  // ผ่าน clampNum() ด้านล่าง ไม่งั้นไฟล์โปรเจกต์ที่แก้มือจะทำให้แอปกับสคริปต์ให้ผลต่างกัน
  prewitt_edge: { needs: "gray", out: "gray", gen: (p) => {
    const KX = "kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)";
    const KY = "ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], np.float32)";
    if (p.dir === "x") return [KX, "img = cv2.convertScaleAbs(cv2.filter2D(img, cv2.CV_16S, kx))"];
    if (p.dir === "y") return [KY, "img = cv2.convertScaleAbs(cv2.filter2D(img, cv2.CV_16S, ky))"];
    return [
      KX, KY,
      "gx = cv2.convertScaleAbs(cv2.filter2D(img, cv2.CV_16S, kx))",
      "gy = cv2.convertScaleAbs(cv2.filter2D(img, cv2.CV_16S, ky))",
      "img = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)",
    ];
  } },

  roberts_edge: { needs: "gray", out: "gray", gen: (p) => {
    const K1 = "k1 = np.array([[1,0],[0,-1]], np.float32)";
    const K2 = "k2 = np.array([[0,1],[-1,0]], np.float32)";
    // cv2.filter2D ใช้ anchor (-1,-1) เป็นค่า default อยู่แล้ว = ค่าเดียวกับฝั่ง JS (ห้ามใส่ (0,0))
    if (p.dir === "d1") return [K1, "img = cv2.convertScaleAbs(cv2.filter2D(img, cv2.CV_16S, k1))"];
    if (p.dir === "d2") return [K2, "img = cv2.convertScaleAbs(cv2.filter2D(img, cv2.CV_16S, k2))"];
    return [
      K1, K2,
      "g1 = cv2.convertScaleAbs(cv2.filter2D(img, cv2.CV_16S, k1))",
      "g2 = cv2.convertScaleAbs(cv2.filter2D(img, cv2.CV_16S, k2))",
      "img = cv2.addWeighted(g1, 0.5, g2, 0.5, 0)",
    ];
  } },

  log_edge: { needs: "gray", out: "gray", gen: (p) => {
    const sigma = clampNum(p.sigma, 2, 0.5, 5);
    const thr = clampNum(p.zeroThresh, 8, 0, 50);
    const head = [
      `blur = cv2.GaussianBlur(img, (0, 0), ${sigma})`,
      "lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)",
    ];
    if (p.view === "response") {
      return [...head,
        "max_abs = max(abs(float(lap.min())), abs(float(lap.max())))",
        "alpha = 127.0 / max_abs if max_abs > 1e-6 else 1.0",
        // ห้ามใช้ .astype(np.uint8) แทน convertScaleAbs — numpy ตัดเศษทิ้ง แต่ OpenCV ปัดแบบ saturate_cast
        "img = cv2.convertScaleAbs(lap, alpha=alpha, beta=128)  # 128 = ศูนย์, สว่าง = บวก, มืด = ลบ",
      ];
    }
    return [...head,
      "c = lap[:-1, :-1]; r = lap[:-1, 1:]; d = lap[1:, :-1]",
      `m = (((c > 0) != (r > 0)) & (np.abs(c - r) > ${thr})) | (((c > 0) != (d > 0)) & (np.abs(c - d) > ${thr}))`,
      "zc = np.zeros(lap.shape, np.uint8)",
      "zc[:-1, :-1] = np.where(m, 255, 0)",
      "img = zc  # เส้นขอบจากจุดตัดศูนย์ (แถว/คอลัมน์สุดท้ายเป็น 0 เสมอ เหมือนในแอป)",
    ];
  } },

  // out ขึ้นกับโหมด: mask = ภาพขาวดำ 1 ช่อง, overlay = ภาพสี 3 ช่อง → ใช้ outOf() แบบเดียวกับ GEN.grabcut
  region_growing: { needs: "gray", outOf: (p) => (p.view === "overlay" ? "color" : "gray"), gen: (p) => {
    const sx = clampNum(p.seedX, 50, 0, 100);
    const sy = clampNum(p.seedY, 50, 0, 100);
    const tol = Math.round(clampNum(p.tolerance, 12, 0, 100));
    const fixed = p.rangeMode === "floating" ? "" : " | cv2.FLOODFILL_FIXED_RANGE";
    const lines = [
      "h, w = img.shape[:2]",
      // *** ห้ามใช้ round() ตรงนี้ *** — Python round() ปัดครึ่งเข้าหา "เลขคู่" (round-half-to-even) แต่ฝั่งแอป
      // (registry.js, region_growing) ใช้ Math.round ของ JS ซึ่งปัดครึ่ง "ขึ้น" เสมอ → เมล็ดจะตกคนละพิกเซล
      // เช่น ภาพกว้าง 22 ที่ 50% แอปได้ x=11 แต่ Python round() ได้ 10 แล้ว flood fill ก็ได้คนละพื้นที่
      // `int(v + 0.5)` = ปัดครึ่งขึ้น เท่ากับ Math.round เป๊ะ เพราะ v = <%> / 100 * (w - 1) ไม่มีทางติดลบ
      // (clampNum บังคับ % ให้อยู่ใน 0–100 และ w >= 1) และ int() ของ Python ตัดเศษเข้าหาศูนย์ = floor() เมื่อค่า
      // ไม่ติดลบ — เลือกวิธีนี้แทน math.floor() เพื่อไม่ต้องเติม `import math` เข้า header ของสคริปต์ทุกไฟล์
      `sx = min(w - 1, max(0, int(${sx} / 100.0 * (w - 1) + 0.5)))`,
      `sy = min(h - 1, max(0, int(${sy} / 100.0 * (h - 1) + 0.5)))`,
      "mask = np.zeros((h + 2, w + 2), np.uint8)",
      // (255 << 8) = ค่าที่เขียนลง mask · FLOODFILL_MASK_ONLY = ห้ามเขียนทับภาพต้นทาง
      `flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY${fixed}`,
      // cv2.floodFill คืน tuple 4 ตัว ต้อง unpack ให้ครบ ไม่งั้นได้ tuple มาเป็นภาพ
      `filled, _, mask, _ = cv2.floodFill(img, mask, (sx, sy), 0, (${tol},) * 3, (${tol},) * 3, flags)`,
      "region = mask[1:h + 1, 1:w + 1]",
      'print(f"เลือกได้ {filled} พิกเซล")',
    ];
    if (p.view === "overlay") {
      lines.push(
        "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
        "img[region > 0] = (0, 255, 0)  # ฝั่ง Python เป็น BGR — (0,255,0) ก็ยังเป็นสีเขียว",
        "cv2.circle(img, (sx, sy), 4, (0, 0, 255), 2)  # จุดเริ่มสีแดง (BGR)",
      );
    } else {
      lines.push("img = region  # หน้ากากขาว-ดำ (0/255)");
    }
    return lines;
  } },

  find_contours: { needs: "gray", out: "color", gen: (p) => [
    "contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)",
    "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
    `for c in contours:`,
    `    if cv2.contourArea(c) < ${int(p.minArea)}:`,
    "        continue",
    "    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)",
  ] },

  connected_components: { needs: "gray", out: "color", gen: () => [
    "n, labels = cv2.connectedComponents(img)",
    "hue = np.uint8(179 * labels / max(1, n - 1))",
    "blank = np.full_like(hue, 255)",
    "img = cv2.merge([hue, blank, blank])",
    "img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)",
    "img[labels == 0] = 0  # พื้นหลังเป็นสีดำ",
  ] },

  distance_transform: { needs: "gray", out: "gray", gen: (p) => [
    `_, bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)`,
    `dist = cv2.distanceTransform(bin, cv2.DIST_L2, ${p.maskSize === "3" ? 3 : 5})`,
    "img = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)",
  ] },

  local_std_filter: { needs: "gray", out: "gray", gen: (p) => {
    // odd(p.windowSize) เดิม fallback เป็น 1 เมื่อ params ไม่ครบ (odd() ใช้ `Number(n) || 1`) ต่างจาก
    // registry.js ที่ fallback เป็น 9 (`ensureOdd(Number.isFinite(p.windowSize) ? p.windowSize : 9)`)
    // — ใช้ finiteOr() ให้ fallback ตรงกันก่อน แล้วค่อยปัดเป็นเลขคี่แบบเดียวกับ ensureOdd (ไม่มี `|| 1` ซ้อน)
    const wsRaw = finiteOr(p.windowSize, 9);
    let k = Math.round(wsRaw);
    if (k % 2 === 0) k += 1;
    return [
      "f32 = img.astype(np.float32)",
      `mean = cv2.boxFilter(f32, cv2.CV_32F, (${k}, ${k}))`,
      `mean_sq = cv2.boxFilter(f32 * f32, cv2.CV_32F, (${k}, ${k}))`,
      "variance = np.maximum(mean_sq - mean * mean, 0)  # clip tiny negative float noise to 0 before sqrt",
      "std = np.sqrt(variance)",
      "normed = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX)",
      // JS: norm.convertTo(dst, cv.CV_8U) ใช้ cvRound (ปัดเศษ) ไม่ใช่ truncate — np.rint ปัดแบบ
      // round-half-to-even ตรงกับ cvRound ของ OpenCV เป๊ะ (ต่างจาก .astype(np.uint8) ที่ตัดทศนิยมทิ้งเฉย ๆ)
      "img = np.rint(normed).astype(np.uint8)",
    ];
  } },

  local_entropy_filter: { needs: "gray", out: "gray", gen: (p) => {
    const win = { "3": 3, "5": 5, "7": 7 }[p.windowSize] || 5;
    const bins = { "8": 8, "16": 16, "32": 32 }[p.bins] || 16;
    const shift = 8 - Math.log2(bins);
    const half = Math.floor(win / 2);
    return [
      "rows, cols = img.shape",
      "scale = min(1.0, 320 / max(rows, cols))",
      // RULE 7: int(x + 0.5) ไม่ใช่ round() — ขนาดนี้ขึ้นกับ rows/cols จริงตอน runtime เหมือนบั๊ก FFT เดิม
      "if scale < 1:",
      "    analysis = cv2.resize(img, (max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))), interpolation=cv2.INTER_AREA)",
      "else:",
      "    analysis = img",
      "a_h, a_w = analysis.shape",
      `win, bins, shift, half = ${win}, ${bins}, ${shift}, ${half}`,
      "max_entropy = np.log2(bins)",
      "out = np.zeros((a_h, a_w), dtype=np.uint8)",
      // JS ไม่ pad ขอบภาพ แต่ตัดพิกเซลนอกภาพออก (skip out-of-bounds) แล้วหารด้วยจำนวนพิกเซล "ในภาพจริง"
      // เท่านั้น (ดู registry.js) — Python เดิมใช้ BORDER_REPLICATE (ยืดพิกเซลขอบ) แล้วหารด้วยขนาด
      // หน้าต่างเต็มเสมอ ต่างจาก JS ที่แถบขอบภาพ (แคบ ๆ กว้าง half px) ทำให้ผลต่างกัน — แก้ให้ slice
      // ตรง ๆ จาก analysis (Python จะ clip ขอบที่เกินขอบภาพให้อัตโนมัติเมื่อ slice เกินขนาด array พอดี
      // กับที่ JS ทำด้วย if (xx<0||xx>=w) continue) แล้วหารด้วยขนาด slice จริง (window.size) ไม่ใช่ win*win คงที่
      "for y in range(a_h):",
      "    for x in range(a_w):",
      "        window = analysis[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1] >> shift",
      "        hist = np.bincount(window.flatten(), minlength=bins).astype(np.float64)",
      "        hist = hist[hist > 0] / window.size",
      "        e = -np.sum(hist * np.log2(hist))",
      "        out[y, x] = int((e / max_entropy) * 255 + 0.5)  # int(x+0.5), not round() — rule 7",
      "img = cv2.resize(out, (cols, rows)) if scale < 1 else out",
    ];
  } },

  // Region Properties (Tier 2, 18c) — ต้อง mirror registry.js's OPS.regionprops.apply() เป๊ะทุกจุดที่มีผลต่อ
  // ตัวเลข: cv2.contourArea (ไม่ใช่นับพิกเซล) + RETR_EXTERNAL, cv2.arcLength(c, True) ปิด, m00==0 guard
  // (fallback เป็นจุดกึ่งกลาง bounding box), perimeter==0 guard + clamp circularity เข้า [0,1], fitEllipse
  // ต้อง >=5 จุดไม่งั้น None, สูตรแปลงมุม (angle+90)%180 (ดู comment เต็มที่ registry.js), และลำดับ sort
  // เดียวกันเป๊ะ (total order: area DESC, by ASC, bx ASC, cx ASC, cy ASC, tx ASC — ดู compareRegionCandidates
  // ใน registry.js) + cap maxObjects หลัง sort (0 = ไม่จำกัด) แล้วค่อยวัดแพง/วาดเฉพาะชิ้นที่รอด
  // หมายเหตุ: JS คำนวณ cx/cy แบบ lazy (เฉพาะตอน 3 key แรกเสมอกัน) เพื่อความเร็ว ส่วน Python คำนวณทุกชิ้นใน
  // รอบแรก — ลำดับผลลัพธ์เท่ากันเป๊ะ เพราะ key ที่ 4-5 ถูกใช้เฉพาะเมื่อ 3 key แรกเสมอกันในทั้งสองฝั่ง
  regionprops: { needs: "gray", out: "color", gen: (p) => {
    const minArea = finiteOr(p.minArea, 50); // minArea:0 เป็นค่าที่ใช้งานได้จริง — ต้องใช้ finiteOr ไม่ใช่ int()/`|| d`
    // ตรงกับ registry.js: Math.max(0, Math.round(finite ? v : 500)) — 0 = ไม่จำกัด (ค่าที่ใช้งานได้จริง)
    const cap = Math.max(0, Math.round(finiteOr(p.maxObjects, 500)));
    return [
      "_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)",
      "contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)",
      "gray_src = img  # ภาพเทาก่อนวาด (ใช้วัดความสว่างเฉลี่ย)",
      "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
      ...regionStatsPyLines(),
      "# รอบที่ 1: คัดตาม minArea + เก็บ sort key (area DESC, y ASC, x ASC, cx ASC, cy ASC, tx ASC)",
      "cands = []",
      "for c in contours:",
      "    area = cv2.contourArea(c)",
      `    if area < ${minArea}:`,
      "        continue",
      "    x, y, w, h = cv2.boundingRect(c)",
      "    m = cv2.moments(c)",
      "    cx = (x + w / 2) if m['m00'] == 0 else m['m10'] / m['m00']  # m00=0 (วัตถุ 1px) -> จุดกลางกรอบ",
      "    cy = (y + h / 2) if m['m00'] == 0 else m['m01'] / m['m00']",
      "    pts = c.reshape(-1, 2)",
      "    tx = int(pts[pts[:, 1] == y][:, 0].min())  # จุดบนสุด-ซ้ายสุด: ไม่ซ้ำกันระหว่างวัตถุต่างชิ้นโดยโครงสร้าง",
      "    cands.append(((-area, y, x, cx, cy, tx), c, area, cx, cy))",
      "cands.sort(key=lambda t: t[0])",
      "total_found = len(cands)",
      `max_objects = ${cap}  # 0 = ไม่จำกัด`,
      "shown = cands if max_objects == 0 else cands[:max_objects]",
      "# รอบที่ 2: วัดเต็ม + วาด เฉพาะชิ้นที่รอด cap (ค่าทุกแถวที่แสดงเป็นค่าจริง ไม่ได้ประมาณ)",
      "regions = []",
      "for _key, c, area, cx, cy in shown:",
      "    perimeter = cv2.arcLength(c, True)",
      "    x, y, w, h = cv2.boundingRect(c)",
      "    circularity = 0 if perimeter == 0 else (4 * np.pi * area) / (perimeter * perimeter)",
      "    circularity = max(0.0, min(1.0, circularity))",
      "    orientation = None",
      "    if len(c) >= 5:",
      "        (_, _), (_, _), angle = cv2.fitEllipse(c)",
      "        # มุมแกนหลัก (องศา 0-180): 0 = แนวนอน, เพิ่มตามเข็มนาฬิกาเพราะแกน y ของภาพชี้ลง (90 = แนวตั้ง)",
      "        # fitEllipse คืนมุมของแกน width ที่ตั้งฉากกับแกนหลัก จึงต้อง +90 แล้ว mod 180",
      "        orientation = (angle + 90) % 180",
      "    regions.append({'area': area, 'cx': cx, 'cy': cy, 'perimeter': perimeter, 'circularity': circularity,",
      "                     'bx': x, 'by': y, 'bw': w, 'bh': h, 'orientation': orientation})",
      "    regions[-1].update(region_stats(c, x, y, w, h, gray_src, bin_img, True))",
      "if total_found == 0:",
      "    print(\"ไม่พบวัตถุ\")",
      "else:",
      "    avg_area = sum(t[2] for t in cands) / total_found",
      "    shown_note = f\" แสดง {len(regions)} ชิ้นที่ใหญ่ที่สุด\" if len(regions) < total_found else \"\"",
      "    print(f\"พบวัตถุ {total_found} ชิ้น{shown_note} — พื้นที่เฉลี่ย {avg_area:.0f}px²\")",
      "# พิมพ์ตารางค่าที่วัดได้ (20 แถวแรก) — แอปยังไม่มีที่แสดงตัวเลขต่อวัตถุบนจอ โค้ดที่ export จึงเป็นที่ที่ผู้เรียนเห็นตัวเลขจริง",
      "if regions:",
      "    print('index  area  cx  cy  perimeter  circularity  bbox(x,y,w,h)  orientation_deg')",
      "for idx, r in enumerate(regions[:20], start=1):",
      "    ori = '-' if r['orientation'] is None else f\"{r['orientation']:.1f}\"",
      "    print(f\"{idx}  {r['area']:.1f}  {r['cx']:.1f}  {r['cy']:.1f}  {r['perimeter']:.1f}  {r['circularity']:.2f}  ({r['bx']},{r['by']},{r['bw']},{r['bh']})  {ori}\")",
      "if regions:",
      "    print('object  area_px  equiv_diam  extent  solidity  major_axis  minor_axis  eccentricity  mean_intensity')",
      "for idx, r in enumerate(regions[:20], start=1):",
      "    print(f\"{idx}  {r['area_px']}  {r['equiv_diameter']:.1f}  {r['extent']:.3f}  {r['solidity']:.3f}  {r['major_axis']:.1f}  {r['minor_axis']:.1f}  {r['eccentricity']:.3f}  {r['mean_intensity']:.1f}\")",
      "if regions and all(r['mean_intensity'] == 255 for r in regions):",
      "    print(\"note: the input is already black/white, so mean_intensity is 255 for every object\")",
      ...mmPyLines("regionprops", clampNum(p.mmPerPx, 0, 0, 100)),
      "for idx, r in enumerate(regions, start=1):",
      "    cv2.rectangle(img, (r['bx'], r['by']), (r['bx'] + r['bw'], r['by'] + r['bh']), (0, 255, 0), 2)",
      "    cv2.circle(img, (int(r['cx'] + 0.5), int(r['cy'] + 0.5)), 3, (0, 0, 255), -1)",
      "    cv2.putText(img, str(idx), (r['bx'], max(10, r['by'] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)",
    ];
  } },

  // IP batch 2 ชิ้น D — bwferet บนขอบพิกเซล: ลำดับคำนวณ/กติกาเสมอกันเหมือน feretOfContour ของ registry.js ทุกขั้น (เลข Python int ล้วน
  // ระยะจึงเท่าแอปทุก bit · มุมต่าง ≤ 1 ulp ของ atan2) · ⚠️ ภาพสีหลัง Color Space ถูกแปลงขาวดำด้วย BGR2GRAY อัตโนมัติ (backlog P-23 ของเดิม)
  feret: { needs: "gray", out: "color", gen: (p, ctx) => {
    const minArea = clampNum(p.minArea, 50, 0, 5000);
    const cap = Math.round(clampNum(p.maxObjects, 50, 1, 500));
    const green = pyColor(ctx, [0, 255, 0]), red = pyColor(ctx, [255, 0, 0]), yellow = pyColor(ctx, [255, 255, 0]);
    return [
      "_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)",
      "contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)",
      "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
      ...regionCandidatesPy(minArea, cap),
      "feret_rows = []",
      "for idx, (_key, c, area, cx, cy) in enumerate(shown, start=1):",
      "    # พิกเซล (x, y) = สี่เหลี่ยมกว้าง 1 px → มุมทั้ง 4 คูณ 2 ให้เป็นจำนวนเต็ม (2x ± 1, 2y ± 1) แล้วหา convex hull",
      "    p2 = c.reshape(-1, 2).astype(np.int64) * 2",
      "    corners = np.stack([p2 + (-1, -1), p2 + (1, -1), p2 + (1, 1), p2 + (-1, 1)], axis=1).reshape(-1, 2)",
      "    hull = cv2.convexHull(corners.astype(np.int32)).reshape(-1, 2).tolist()",
      "    m_h = len(hull)",
      "    # ยาวสุด (Max Feret): คู่จุดบน hull ที่ห่างกันมากสุด · เสมอกัน → มุมเล็กกว่า",
      "    best, bdx, bdy, bi, bj = -1, 0, 0, 0, 0",
      "    for i in range(m_h):",
      "        for j in range(i + 1, m_h):",
      "            dx, dy = hull[j][0] - hull[i][0], hull[j][1] - hull[i][1]",
      "            d2 = dx * dx + dy * dy",
      "            if dy < 0 or (dy == 0 and dx < 0):",
      "                dx, dy = -dx, -dy",
      "            if d2 > best or (d2 == best and bdx * dy - bdy * dx < 0):",
      "                best, bdx, bdy, bi, bj = d2, dx, dy, i, j",
      "    # แคบสุด (Min Feret): ด้านของ hull ที่ระยะไปจุดไกลสุดน้อยสุด · len = ช่วงที่วัตถุยื่นไปตามด้านนั้น",
      "    w_min, ndx, ndy, e_i, e_x, e_y, e_lo, e_hi, e_s = float('inf'), 0, 0, 0, 0, 0, 0, 0, 0",
      "    for i in range(m_h):",
      "        j = 0 if i + 1 == m_h else i + 1",
      "        hx, hy = hull[i]",
      "        ex, ey = hull[j][0] - hx, hull[j][1] - hy",
      "        e2 = ex * ex + ey * ey",
      "        if e2 == 0:",
      "            continue",
      "        mc, cs, lo, hi = 0, 0, 0, 0",
      "        for kx, ky in hull:",
      "            px, py = kx - hx, ky - hy",
      "            cr = ex * py - ey * px  # ระยะตั้งฉากจากด้าน x |e|",
      "            dt = ex * px + ey * py  # ตำแหน่งตามแนวด้าน x |e|",
      "            if abs(cr) > mc:",
      "                mc, cs = abs(cr), cr",
      "            if dt < lo:",
      "                lo = dt",
      "            elif dt > hi:",
      "                hi = dt",
      "        wd = mc / np.sqrt(e2)",
      "        nx, ny = -ey, ex",
      "        if ny < 0 or (ny == 0 and nx < 0):",
      "            nx, ny = -nx, -ny",
      "        if wd < w_min or (wd == w_min and ndx * ny - ndy * nx < 0):",
      "            w_min, ndx, ndy, e_i, e_x, e_y, e_lo, e_hi, e_s = wd, nx, ny, i, ex, ey, lo, hi, cs",
      "    f_max = float(np.sqrt(best) / 2)",
      "    f_min = float(w_min / 2)",
      "    f_len = float((e_hi - e_lo) / np.sqrt(e_x * e_x + e_y * e_y) / 2)",
      "    angles = []",
      "    for adx, ady in ((bdx, bdy), (ndx, ndy)):  # 0 = แนวนอน เพิ่มตามเข็มนาฬิกา (แกน y ชี้ลง) ช่วง 0-180",
      "        a = float(np.arctan2(ady, adx) * 180 / np.pi)",
      "        if a < 0:",
      "            a += 180",
      "        if a >= 180:",
      "            a = 0.0",
      "        angles.append(a)",
      "    feret_rows.append((idx, area, cx, cy, f_max, f_min, f_len, f_len / f_min, angles[0], angles[1]))",
      "    # วาด: กรอบแคบสุด (เขียว) → แนวยาวสุด (แดง) → เลขวัตถุ (เหลือง ตำแหน่งเดียวกับ Region Properties)",
      "    hx, hy = hull[e_i]",
      "    e2 = e_x * e_x + e_y * e_y",
      "    box = [(int(np.floor((hx + (e_x * t - e_y * s) / e2) / 2 + 0.5)), int(np.floor((hy + (e_y * t + e_x * s) / e2) / 2 + 0.5)))",
      "           for t, s in ((e_lo, 0), (e_hi, 0), (e_hi, e_s), (e_lo, e_s))]",
      "    for q in range(4):",
      `        cv2.line(img, box[q], box[(q + 1) % 4], ${green}, 1)`,
      `    cv2.line(img, ((hull[bi][0] + 1) // 2, (hull[bi][1] + 1) // 2), ((hull[bj][0] + 1) // 2, (hull[bj][1] + 1) // 2), ${red}, 1)`,
      "    bx, by, _, _ = cv2.boundingRect(c)",
      `    cv2.putText(img, str(idx), (bx, max(10, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ${yellow}, 1)`,
      "if total_found == 0:",
      "    print(\"ไม่พบวัตถุ (ลองลดค่าพื้นที่ขั้นต่ำ หรือทำ Threshold ให้เป็นขาว-ดำก่อน)\")",
      "else:",
      "    if len(feret_rows) < total_found:",
      "        head = f\"พบวัตถุ {total_found} ชิ้น วัด {len(feret_rows)} ชิ้นแรกเรียงจากใหญ่ไปเล็ก (ปรับ 'วัดวัตถุสูงสุด' ได้)\"",
      "    else:",
      "        head = f\"วัด Feret ของวัตถุ {total_found} ชิ้น\"",
      `    print(f"{head} · ชิ้น #1 ยาวสุด ${pyFixed1("feret_rows[0][4]")} px · แคบสุด ${pyFixed1("feret_rows[0][5]")} px (กรอบเขียว = กรอบที่แคบสุด, เส้นแดง = แนวที่ยาวสุด)")`,
      "# ตารางค่า (20 แถวแรก) · หน่วย px · มุมเป็นองศา",
      "if feret_rows:",
      "    print('index  area  cx  cy  feret_max  feret_min  feret_len  feret_aspect  feret_max_deg  feret_min_deg')",
      "for r in feret_rows[:20]:",
      "    print(f\"{r[0]}  {r[1]:.1f}  {r[2]:.1f}  {r[3]:.1f}  {r[4]:.2f}  {r[5]:.2f}  {r[6]:.2f}  {r[7]:.2f}  {r[8]:.1f}  {r[9]:.1f}\")",
      ...mmPyLines("feret", clampNum(p.mmPerPx, 0, 0, 100)),
    ];
  } },

  // IP batch 3 — Filter by Shape: เลือกเฉพาะแขนงของ prop ที่ใช้ตอน gen (ไม่มี if ใน Python) · ตัวเลข/prop/view clamp+สลับตอน gen เหมือน registry.js
  shape_filter: { needs: "gray", outOf: (p) => (pickOwn(SHAPE_FILTER_VIEW_PY, p.view, "overlay") === "mask" ? "gray" : "color"), gen: (p, ctx) => {
    const prop = pickOwn(SHAPE_FILTER_PROP_PY, p.prop, "circularity");
    const mask = pickOwn(SHAPE_FILTER_VIEW_PY, p.view, "overlay") === "mask";
    let lo = clampNum(p.lo, 0.8, 0, 1), hi = clampNum(p.hi, 1, 0, 1);
    let minA = clampNum(p.minArea, 50, 0, 100000), maxA = clampNum(p.maxArea, 0, 0, 5000000);
    let rangeSwapped = false, areaSwapped = false;
    if (lo > hi) { const t = lo; lo = hi; hi = t; rangeSwapped = true; }
    if (maxA > 0 && maxA < minA) { const t = minA; minA = maxA; maxA = t; areaSwapped = true; }
    const lines = [
      "_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)",
      "contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)",
      ...regionStatsPyLines(),
      `lo, hi = ${String(lo)}, ${String(hi)}  # ช่วงค่าที่เก็บ`,
      `min_area, max_area = ${String(minA)}, ${String(maxA)}  # max_area 0 = ไม่จำกัด`,
      "kept_mask = np.zeros(bin_img.shape, np.uint8)",
      "kept = removed_area = removed_shape = 0",
      "for i, c in enumerate(contours):",
      "    x, y, w, h = cv2.boundingRect(c)",
      "    if w * h < min_area:",
      "        removed_area += 1",
      "        continue",
      `    st = region_stats(c, x, y, w, h, img, bin_img, ${prop === "solidity" ? "True" : "False"})`,
      "    if st['area_px'] < min_area or (max_area > 0 and st['area_px'] > max_area):",
      "        removed_area += 1",
      "        continue",
    ];
    if (prop === "circularity") {
      lines.push(
        "    perimeter = cv2.arcLength(c, True)",
        "    area = cv2.contourArea(c)",
        "    v = 0 if perimeter == 0 else (4 * np.pi * area) / (perimeter * perimeter)",
        "    v = max(0.0, min(1.0, v))",
      );
    } else if (prop !== "none") {
      lines.push(`    v = st['${prop}']`);
    }
    if (prop !== "none") {
      lines.push(
        "    if not (lo <= v <= hi):",
        "        removed_shape += 1",
        "        continue",
      );
    }
    lines.push(
      "    kept += 1",
      "    cv2.drawContours(kept_mask, contours, i, 255, -1)",
      "kept_mask = cv2.bitwise_and(kept_mask, bin_img)",
      'print(f"shape filter: kept {kept} of {len(contours)} objects (removed by area {removed_area}, by shape {removed_shape})")',
    );
    if (rangeSwapped) lines.push('print("note: lo/hi were swapped")');
    if (areaSwapped) lines.push('print("note: min/max area were swapped")');
    if (mask) {
      lines.push("img = kept_mask");
    } else {
      lines.push(
        "out = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)",
        `out[bin_img > 0] = ${pyColor(ctx, [255, 0, 0])}`,
        "out[kept_mask > 0] = (255, 255, 255)",
        "img = out",
      );
    }
    return lines;
  } },

  // IP batch 2 ชิ้น D — improfile (nearest neighbour) · นิพจน์/ลำดับคำนวณเดียวกับ registry.js จึงได้ภาพ + ตัวเลขเท่าแอปทุก byte
  line_profile: { needs: "any", out: "color", gen: (p, ctx) => {
    const x1 = clampNum(p.x1, 10, 0, 100), y1 = clampNum(p.y1, 50, 0, 100);
    const x2 = clampNum(p.x2, 90, 0, 100), y2 = clampNum(p.y2, 50, 0, 100);
    const plot = pickOwn(LINE_PROFILE_VIEW_PY, p.view, "overlay") === "plot";
    const yellow = pyColor(ctx, [255, 255, 0]), green = pyColor(ctx, [0, 255, 0]), red = pyColor(ctx, [255, 0, 0]);
    const magenta = pyColor(ctx, [255, 0, 255]), grey = pyColor(ctx, [128, 128, 128]);
    const lineDraw = [ // ภาพเดิม + เส้น A→B + จุด A/B + วงขอบแรงสุด
      "out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()",
      `cv2.line(out, (x1, y1), (x2, y2), ${yellow}, 1)`,
      `cv2.circle(out, (x1, y1), 3, ${green}, -1)  # A`,
      `cv2.circle(out, (x2, y2), 3, ${red}, -1)  # B`,
      "if i_star >= 0:",
      `    cv2.circle(out, (int(xs[i_star + 1]), int(ys[i_star + 1])), 4, ${magenta}, 1)  # ตำแหน่งขอบแรงสุด`,
    ];
    return [
      "h, w = img.shape[:2]",
      `gray = img if img.ndim == 2 else cv2.cvtColor(img, ${pyGrayCode(ctx)})  # วัดจากความสว่างแบบขาวดำ`,
      `x1, y1 = ${pyPctToPx(x1, "w")}, ${pyPctToPx(y1, "h")}  # จุด A (% → px, 100% = พิกเซลสุดท้าย)`,
      `x2, y2 = ${pyPctToPx(x2, "w")}, ${pyPctToPx(y2, "h")}  # จุด B`,
      "dx, dy = x2 - x1, y2 - y1",
      "n = max(abs(dx), abs(dy)) + 1  # จำนวนจุดที่อ่าน",
      "if n == 1:",
      "    xs, ys = np.array([x1]), np.array([y1])",
      "else:",
      "    idx = np.arange(n, dtype=np.float64)",
      "    xs = np.floor(x1 + (x2 - x1) * idx / (n - 1) + 0.5).astype(np.int64)  # nearest neighbour (ค่าเริ่มต้นของ improfile)",
      "    ys = np.floor(y1 + (y2 - y1) * idx / (n - 1) + 0.5).astype(np.int64)",
      "profile = gray[ys, xs].astype(np.int64)",
      "line_length_px = float(np.sqrt(dx * dx + dy * dy))",
      "p_min, p_max = int(profile.min()), int(profile.max())",
      "profile_mean = int(profile.sum()) / n",
      "edge_jump = edge_pos_px = edge_pair_px = None",
      "i_star = -1",
      "if n >= 2:",
      "    step = line_length_px / (n - 1)",
      "    diff = np.diff(profile)",
      "    if np.any(diff != 0):  # เส้นแบน (ทุกจุดเท่ากัน) ไม่มีขอบ → ค่าขอบเป็น None",
      "        i_star = int(np.argmax(np.abs(diff)))  # ขอบแรงสุด (ตัวแรกถ้าเท่ากัน)",
      "        edge_jump = int(diff[i_star])",
      "        edge_pos_px = (i_star + 0.5) * step",
      "        r_i, f_i = int(np.argmax(diff)), int(np.argmin(diff))  # ขอบขึ้น / ขอบลงแรงสุด",
      "        if diff[r_i] > 0 and diff[f_i] < 0:",
      "            edge_pair_px = abs(r_i - f_i) * step",
      "small = w < 64 or h < 64  # ภาพเล็กไม่วาดกราฟ (ตัวเลขยังครบ)",
      ...(plot
        ? [
          "plot_view = not small",
          "if plot_view:",
          "    out = np.full((h, w, 3), 24, np.uint8)  # กราฟเต็มภาพบนพื้นเทาเข้ม",
          "else:  # ภาพเล็ก: ถอยเป็นภาพเส้นไม่มีกราฟ",
          ...lineDraw.map((l) => "    " + l),
        ]
        : ["plot_view = False", ...lineDraw]),
      "if not small:",
      "    if plot_view:",
      "        gx0, gy0, gw, gh = 0, 0, w, h",
      "    else:",
      "        gh = int(0.35 * h)",
      "        gx0, gw = 0, w",
      "        gy0 = 0 if (y1 >= h - gh and y2 >= h - gh) else h - gh  # เส้นอยู่ล่างทั้งคู่ → ย้ายกราฟขึ้นบน",
      "        out[gy0:gy0 + gh] = cv2.convertScaleAbs(out[gy0:gy0 + gh], alpha=0.3, beta=0)  # แถบกราฟมืดลง",
      "    y_0 = gy0 + (gh - 1) - (255 // 510)",
      "    y_255 = gy0 + (gh - 1) - ((2 * 255 * (gh - 1) + 255) // 510)",
      `    cv2.line(out, (gx0, y_0), (gx0 + gw - 1, y_0), ${grey}, 1)`,
      `    cv2.line(out, (gx0, y_255), (gx0 + gw - 1, y_255), ${grey}, 1)`,
      "    gys = gy0 + (gh - 1) - (2 * profile * (gh - 1) + 255) // 510",
      "    if n == 1:",
      `        cv2.circle(out, (gx0 + (gw - 1) // 2, int(gys[0])), 2, ${yellow}, -1)`,
      "    else:",
      "        gxs = gx0 + (np.arange(n, dtype=np.int64) * (gw - 1)) // (n - 1)",
      `        cv2.polylines(out, [np.stack([gxs, gys], axis=1).astype(np.int32).reshape(-1, 1, 2)], False, ${yellow}, 1)`,
      "        if i_star >= 0:",
      "            xe = gx0 + ((i_star + 1) * (gw - 1)) // (n - 1)",
      `            cv2.line(out, (xe, gy0), (xe, gy0 + gh - 1), ${magenta}, 1)`,
      "img = out",
      "if n == 1:",
      "    info = \"จุด A กับ B อยู่ตำแหน่งเดียวกัน — ปรับให้ห่างกันก่อน\"",
      "else:",
      `    info = f"เส้น A({x1},{y1})→B({x2},{y2}) ยาว ${pyFixed1("line_length_px")} px · อ่าน {n} จุด · ค่า {p_min}–{p_max} · "`,
      "    if i_star < 0:",
      "        info += \"ค่าบนเส้นเท่ากันทุกจุด ไม่มีขอบ\"",
      "    else:",
      `        info += f"ขอบแรงสุดห่างจาก A ${pyFixed1("edge_pos_px")} px (กระโดด {edge_jump:+d})"`,
      "if small:",
      "    info += \" · ภาพเล็กกว่า 64×64 px จึงไม่วาดกราฟ (ตัวเลขยังคำนวณครบ)\"",
      "print(info)",
      "for key, val in ((\"line_length_px\", line_length_px), (\"samples\", n), (\"profile_min\", p_min), (\"profile_max\", p_max),",
      "                 (\"profile_mean\", profile_mean), (\"profile_contrast\", p_max - p_min), (\"edge_jump\", edge_jump),",
      "                 (\"edge_pos_px\", edge_pos_px), (\"edge_pair_px\", edge_pair_px)):",
      "    print(f\"{key} = {val!r}\")",
      "print(\"profile =\", profile.tolist())",
      ...mmPyLines("line_profile", clampNum(p.mmPerPx, 0, 0, 100)),
      "# ดูกราฟแบบละเอียด (ต้องติดตั้ง matplotlib ก่อน: pip install matplotlib) — ลบ # หน้าบรรทัดล่างออก",
      "# import matplotlib.pyplot as plt",
      "# plt.plot(np.arange(n) * (line_length_px / max(n - 1, 1)), profile); plt.xlabel(\"distance from A (px)\"); plt.ylabel(\"brightness\"); plt.show()",
    ];
  } },

  watershed: { needs: "any", out: "color", gen: (p) => [
    "gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
    "img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img",
    "_, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)",
    "kernel = np.ones((3, 3), np.uint8)",
    "opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)",
    "sure_bg = cv2.dilate(opening, kernel, iterations=3)",
    "dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)",
    `_, sure_fg = cv2.threshold(dist, ${num(p.ratio)} * dist.max(), 255, 0)`,
    "sure_fg = np.uint8(sure_fg)",
    "unknown = cv2.subtract(sure_bg, sure_fg)",
    "_, markers = cv2.connectedComponents(sure_fg)",
    "markers = markers + 1",
    "markers[unknown == 255] = 0",
    "markers = cv2.watershed(img, markers)",
    "img[markers == -1] = (0, 0, 255)  # เส้นแบ่งวัตถุสีแดง",
  ] },

  // ---- ตรวจจับวัตถุ ----
  bounding_boxes: { needs: "gray", out: "color", gen: (p) => [
    "contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)",
    "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
    "for c in contours:",
    `    if cv2.contourArea(c) < ${int(p.minArea)}:`,
    "        continue",
    "    x, y, w, h = cv2.boundingRect(c)",
    "    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)",
  ] },

  hough_circles: { needs: "gray", out: "color", gen: (p) => {
    let maxR = int(p.maxR), minR = int(p.minR);
    if (maxR > 0 && maxR < minR) maxR = 0;
    return [
      "blur = cv2.medianBlur(img, 5)",
      `circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, ${num(p.dp)}, ${int(p.minDist)},`,
      `                           param1=100, param2=${int(p.param2)}, minRadius=${minR}, maxRadius=${maxR})`,
      "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
      "if circles is not None:",
      "    for x, y, r in np.uint16(np.around(circles)).reshape(-1, 3):",  // ใช้ได้ทั้ง shape (1,N,3) และ (N,3)
      "        cv2.circle(img, (x, y), r, (0, 255, 0), 2)",
      "        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)",
    ];
  } },

  hough_lines: { needs: "gray", out: "color", gen: (p) => [
    `edges = cv2.Canny(img, ${int(p.t1)}, ${int(p.t2)})`,
    `lines = cv2.HoughLinesP(edges, 1, np.pi / 180, ${int(p.threshold)}, minLineLength=${int(p.minLineLength)}, maxLineGap=${int(p.maxLineGap)})`,
    "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
    "if lines is not None:",
    // OpenCV-Python 5.x คืน lines รูปร่าง (N, 4) ไม่ใช่ (N, 1, 4) เหมือน 4.x — reshape(-1, 4) รองรับทั้งสอง
    "    for x1, y1, x2, y2 in lines.reshape(-1, 4):",
    "        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)",
  ] },

  // ⚠️ output "mask" ได้ภาพ 2 มิติ (img = fg) แต่ out ยังเป็น "color" (ของเดิม — ขั้นถัดไปที่ needs:"gray" หรืออ่าน b,g,r จึงพังใน
  // Python ที่ export · แก้เป็น outOf แบบ region_growing ได้ แต่จะเปลี่ยน Python ของขั้นถัดไปของ op เดิม → แยกตั๋ว) · grayOutOf บอกเฉพาะ
  // ctx.mirrored ว่าภาพเป็นขาวดำแล้ว (ขั้นที่แปลงกลับเป็นสีด้วย img.ndim == 2 เช่น watershed จะได้ BGR อีกครั้ง)
  grabcut: { needs: "any", out: "color", grayOutOf: (p) => p.output === "mask", gen: (p) => {
    const lines = ["img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img",
      "mask = np.zeros(img.shape[:2], np.uint8)",
      "bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)",
      `mx, my = int(img.shape[1] * ${num(p.margin) / 100}), int(img.shape[0] * ${num(p.margin) / 100})`,
      "rect = (mx, my, img.shape[1] - 2 * mx, img.shape[0] - 2 * my)",
      `cv2.grabCut(img, mask, rect, bgd, fgd, ${Math.max(1, int(p.iter))}, cv2.GC_INIT_WITH_RECT)`,
      "fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(\"uint8\")"];
    if (p.output === "mask") lines.push("img = fg");
    else lines.push("img = cv2.bitwise_and(img, img, mask=fg)");
    return lines;
  } },

  // ---- ประมวลผลภาพสี ----
  color_inrange: { needs: "any", out: null, gen: (p, ctx) => {
    const lines = [];
    if (ctx.gray) lines.push("img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)");
    if (p.colorSpace === "lab") {
      lines.push("lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)");
      lines.push(`lower = np.array([${int(p.labLMin)}, ${int(p.labAMin)}, ${int(p.labBMin)}])`);
      lines.push(`upper = np.array([${int(p.labLMax)}, ${int(p.labAMax)}, ${int(p.labBMax)}])`);
      lines.push("mask = cv2.inRange(lab, lower, upper)");
    } else {
      lines.push("hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)");
      lines.push(`lower = np.array([${int(p.hMin)}, ${int(p.sMin)}, ${int(p.vMin)}])`);
      lines.push(`upper = np.array([${int(p.hMax)}, 255, 255])`);
      lines.push("mask = cv2.inRange(hsv, lower, upper)");
    }
    if (p.output === "mask") lines.push("img = mask");
    else lines.push("img = cv2.bitwise_and(img, img, mask=mask)");
    return lines;
  }, outOf: (p) => (p.output === "mask" ? "gray" : "color") },

  kmeans_segment: { needs: "any", out: "color", gen: (p, ctx) => {
    const lines = [];
    if (ctx.gray) lines.push("img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)");
    lines.push(
      "Z = img.reshape((-1, 3)).astype(np.float32)",
      // n ตัวอย่างอาจน้อยกว่า K (เช่นภาพจิ๋วหลัง crop) — cv2.kmeans throw ถ้า n < K จึง clamp ไว้
      `K = min(${Math.max(2, int(p.k))}, Z.shape[0])`,
      `criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, ${int(p.iterations)}, 1.0)`,
      "_, labels, centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)",
      "centers = np.uint8(centers)",
      "img = centers[labels.flatten()].reshape(img.shape)",
    );
    return lines;
  } },

  white_balance: { needs: "any", out: "color", gen: (p, ctx) => {
    const lines = [];
    // IP batch 4A — White Patch / จุดอ้างอิง (ลำดับช่อง: แอป R,G,B ↔ cv2 B,G,R เมื่อ isMirrored) · Gray World / ค่าแปลก = บรรทัดเดิมทุกตัวอักษร
    if (p.method === "white_patch" || p.method === "ref_point") {
      if (ctx.gray) return ["img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)", 'print("white balance: gray image, nothing to adjust")'];
      const ci = isMirrored(ctx) ? "(2, 1, 0)" : "(0, 1, 2)"; // ตำแหน่งช่อง R, G, B ในภาพของ cv2
      const fx2 = (e) => `{np.floor(${e} * 100 + 0.5) / 100:.2f}`;
      const gainText = `gain R x${fx2("gains[0]")} G x${fx2("gains[1]")} B x${fx2("gains[2]")}`;
      const lutLines = [
        "out = np.empty_like(img)",
        `for ch, g in zip(${ci}, gains):`,
        "    lut = np.minimum(255, np.floor(np.arange(256) * g + 0.5)).astype(np.uint8)",
        "    out[:, :, ch] = lut[img[:, :, ch]]",
        "img = out",
      ];
      if (p.method === "white_patch") {
        return [
          "n_px = img.shape[0] * img.shape[1]",
          "k99 = (n_px * 99 + 99) // 100  # ตำแหน่งที่ 99% (นับจากพิกเซลมืดสุด) ปัดขึ้น",
          "v99 = []",
          `for ch in ${ci}:`,
          "    cum = np.cumsum(np.bincount(img[:, :, ch].ravel(), minlength=256))",
          "    v99.append(int(np.searchsorted(cum, k99)))  # ค่าแรกที่ cumsum >= k99",
          "gains = [255 / v if v > 0 else 1 for v in v99]",
          ...lutLines,
          `print(f"white patch: p99 R/G/B = {v99[0]}/{v99[1]}/{v99[2]} ${gainText}")`,
        ];
      }
      const rx = clampNum(p.refX, 50, 0, 100), ry = clampNum(p.refY, 50, 0, 100), rr = Math.round(clampNum(p.refSize, 5, 1, 50));
      const fm = (e) => `{np.floor(${e} * 10 + 0.5) / 10:.1f}`;
      return [
        "h_i, w_i = img.shape[:2]",
        `ref_x, ref_y, ref_r = ${pyPctToPx(rx, "w_i")}, ${pyPctToPx(ry, "h_i")}, ${rr}  # จุดอ้างอิง (% -> px) และรัศมี`,
        "patch = img[max(0, ref_y - ref_r):min(h_i - 1, ref_y + ref_r) + 1, max(0, ref_x - ref_r):min(w_i - 1, ref_x + ref_r) + 1]",
        "sums = patch.reshape(-1, 3).sum(axis=0, dtype=np.int64)",
        "n_ref = patch.shape[0] * patch.shape[1]",
        `m = [sums[ch] / n_ref for ch in ${ci}]  # สีเฉลี่ย R, G, B ของบริเวณอ้างอิง`,
        "t = (m[0] + m[1] + m[2]) / 3",
        "gains = [t / v if v > 1e-3 else 1 for v in m]",
        ...lutLines,
        `print(f"reference point ({ref_x}, {ref_y}) px radius {ref_r}: mean R/G/B ${fm("m[0]")}/${fm("m[1]")}/${fm("m[2]")} ${gainText}")`,
      ];
    }
    if (ctx.gray) lines.push("img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)");
    lines.push(
      // B3 (backlog 2026-10-17): ค่าเฉลี่ยคำนวณจาก uint8 ตรง ๆ (float64 เหมือน cv.mean) + convertScaleAbs ปัดเศษเหมือน convertTo ของแอป (เดิม astype ตัดทิ้ง)
      "b, g, r = cv2.split(img)",
      "gray_mean = (r.mean() + g.mean() + b.mean()) / 3",
      "gain_r = gray_mean / r.mean() if r.mean() > 1e-3 else 1",
      "gain_g = gray_mean / g.mean() if g.mean() > 1e-3 else 1",
      "gain_b = gray_mean / b.mean() if b.mean() > 1e-3 else 1",
      "r = cv2.convertScaleAbs(r, alpha=gain_r)",
      "g = cv2.convertScaleAbs(g, alpha=gain_g)",
      "b = cv2.convertScaleAbs(b, alpha=gain_b)",
      "img = cv2.merge([b, g, r])",
    );
    return lines;
  } },

  invert: { needs: "any", out: "same", gen: () => ["img = cv2.bitwise_not(img)"] },

  crop: { needs: "any", out: "same", gen: (p) => [
    "_h, _w = img.shape[:2]",
    // int(round(...)) ให้ตรงกับ Math.round ฝั่งแอป — int() เฉย ๆ ปัดทิ้งเศษ ทำให้ขนาดต่างกัน ±1px
    `_x = max(0, min(int(round(_w * ${num(p.x) / 100})), _w - 1))`,
    `_y = max(0, min(int(round(_h * ${num(p.y) / 100})), _h - 1))`,
    `_cw = max(1, min(int(round(_w * ${num(p.w) / 100})), _w - _x))`,
    `_ch = max(1, min(int(round(_h * ${num(p.h) / 100})), _h - _y))`,
    "img = img[_y:_y + _ch, _x:_x + _cw].copy()",
  ] },

  resize: { needs: "any", out: "same", gen: (p) => {
    if (Object.hasOwn(INTERP_PY, p.interp)) {
      // เลือกวิธีเอง: ขนาดคำนวณแบบแอป (Math.round + เพดาน 20 ล้านพิกเซล) แล้วส่ง dsize ตรง ๆ — fx/fy ให้ขนาดผิด 1 px ได้ (ไม่ใช้)
      const sc = Math.max(0.05, finiteOr(p.scale, 50) / 100);
      return [
        "h, w = img.shape[:2]",
        `s = ${String(sc)}`,
        "nw, nh = max(1, int(np.floor(w * s + 0.5))), max(1, int(np.floor(h * s + 0.5)))",
        "if nw * nh > 20000000:  # เพดานเดียวกับแอป (กันหน่วยความจำเกิน)",
        "    k = np.sqrt(20000000 / (nw * nh))",
        "    nw, nh = max(1, int(np.floor(nw * k + 0.5))), max(1, int(np.floor(nh * k + 0.5)))",
        `img = cv2.resize(img, (nw, nh), interpolation=${INTERP_PY[p.interp]})`,
      ];
    }
    const s = num(p.scale) / 100 || 0.5;
    return [
      `img = cv2.resize(img, None, fx=${s}, fy=${s}, interpolation=cv2.${s < 1 ? "INTER_AREA" : "INTER_CUBIC"})`,
    ];
  } },

  rotate: { needs: "any", out: "same", gen: (p) => {
    const flagArg = Object.hasOwn(INTERP_PY, p.interp) ? `, flags=${INTERP_PY[p.interp]}` : ""; // auto = บรรทัดเดิม
    const lines = [
      "(h, w) = img.shape[:2]",
      "center = (w / 2, h / 2)",
      `M = cv2.getRotationMatrix2D(center, ${num(p.angle)}, 1.0)`,
    ];
    if (boolOr(p.expand, false)) {
      lines.push(
        `angle_rad = np.deg2rad(abs(${num(p.angle)}))`,
        "cos_a, sin_a = abs(np.cos(angle_rad)), abs(np.sin(angle_rad))",
        // int(round(...)) ให้ตรงกับ Math.round ฝั่งแอป (int() เฉย ๆ ปัดทิ้งเศษ ทำให้ขนาดต่างกัน ±1px)
        "out_w = int(round(w * cos_a + h * sin_a))",
        "out_h = int(round(w * sin_a + h * cos_a))",
        "M[0, 2] += (out_w - w) / 2",
        "M[1, 2] += (out_h - h) / 2",
        `img = cv2.warpAffine(img, M, (out_w, out_h)${flagArg})`,
      );
    } else {
      lines.push(`img = cv2.warpAffine(img, M, (w, h)${flagArg})`);
    }
    return lines;
  } },

  flip: { needs: "any", out: "same", gen: (p) => {
    const code = { h: 1, v: 0, both: -1 }[p.mode] ?? 1;
    return [`img = cv2.flip(img, ${code})`];
  } },

  perspective_warp: { needs: "any", out: "same", gen: (p) => {
    // หมายเหตุ: fallback ที่ 2 (d) ของ pct() ต้องเป็น "ค่า % แบบเดียวกับหน่วยของ v" (เช่น 10 ไม่ใช่ 0.1)
    // เพราะ /100 จะถูกหารทีเดียวตอนท้ายทั้งคู่ — พบว่าฉบับร่างแรกใส่ d เป็น 0.1/-0.1 ผิดหน่วย ทำให้ตอนไม่มี
    // ค่าจริงส่งมา (params ไม่ครบ) ได้ fallback 0.1% แทนที่จะเป็น 10% ตามค่า default ของ registry.js
    //
    // บั๊กที่ 2 (qa เจอในรอบ S5): `(num(v) || d)` ใช้ falsy-check ธรรมดา ทำให้ค่า 0 ที่ผู้ใช้ตั้งจริง ๆ
    // (เช่น tlX=0 ซึ่งเป็นค่าปกติในช่วง -50..50) ถูกตีความว่า "ไม่มีค่า" แล้วแทนด้วย default (10/-10) ผิด
    // ต้องเช็ก finite แทน (ตรงกับ `Number.isFinite(v) ? v : d` ฝั่ง registry.js apply() เป๊ะ) — ใช้ finiteOr()
    const pct = (v, d) => finiteOr(v, d) / 100;
    return [
      "h, w = img.shape[:2]",
      "src_quad = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])",
      "dst_quad = np.float32([",
      `    [${pct(p.tlX, 10)} * w, ${pct(p.tlY, 0)} * h],`,
      `    [w - 1 + ${pct(p.trX, -10)} * w, ${pct(p.trY, 0)} * h],`,
      `    [w - 1 + ${pct(p.brX, 0)} * w, h - 1 + ${pct(p.brY, 0)} * h],`,
      `    [${pct(p.blX, 0)} * w, h - 1 + ${pct(p.blY, 0)} * h],`,
      "])",
      "M = cv2.getPerspectiveTransform(src_quad, dst_quad)",
      "img = cv2.warpPerspective(img, M, (w, h))",
    ];
  } },

  undistort: { needs: "any", out: "same", gen: (p) => {
    // เหมือน perspective_warp: `num(v) || d` ตีความ k1=0 (ค่าปกติ ไม่ใช่ "ไม่มีค่า") ว่าไม่มีค่า แล้วแทน
    // ด้วย -0.3 ผิด — ใช้ finiteOr() เช็ก finite แทน ตรงกับ `Number.isFinite(p.k1) ? p.k1 : -0.3` ของ
    // registry.js (บรรทัด ~2102-2103) เป๊ะ
    const k1 = finiteOr(p.k1, -0.3);
    const k2 = finiteOr(p.k2, 0);
    return [
      "h, w = img.shape[:2]",
      "f = max(w, h)",
      "camera_matrix = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)",
      `dist_coeffs = np.array([${k1}, ${k2}, 0, 0, 0], dtype=np.float64)`,
      "img = cv2.undistort(img, camera_matrix, dist_coeffs)",
    ];
  } },

  // IP batch 2 ชิ้น C — ค่าที่ฝัง clamp ช่วงเดียวกับ registry.js · พิกเซลที่เลื่อนคำนวณตอนรัน (ไม่รู้ขนาดภาพตอน gen) ด้วย
  // int(np.floor(x + 0.5)) เหมือน Math.floor(x + 0.5) ของแอป (x ติดลบได้ — int(x + 0.5) จะปัดเข้าหา 0 ผิด) · guard ใช้นิพจน์เดียวกับแอป
  // เลื่อนอย่างเดียวได้ผลเท่าแอปทุก byte · มี shear: cv2 5.x คำนวณ warpAffine ต่างจาก opencv.js 4.10 เล็กน้อย (ต่างไม่เกินไม่กี่ระดับสี)
  translate_shear: { needs: "any", out: "same", gen: (p) => {
    const tx = clampNum(p.tx, 10, -100, 100);
    const ty = clampNum(p.ty, 5, -100, 100);
    const kx = clampNum(p.shearX, 0, -1, 1);
    const ky = clampNum(p.shearY, 0, -1, 1);
    return [
      "h, w = img.shape[:2]",
      `kx, ky = ${kx}, ${ky}  # shear แนวนอน / แนวตั้ง`,
      "if abs(1 - kx * ky) < 0.05:  # matrix เกือบ singular ภาพจะยุบเป็นเส้น (เหมือนแอป: ไม่ทำ)",
      '    print("translate/shear skipped: shear x * shear y is too close to 1 (the image would collapse to a line)")',
      "else:",
      `    tx = int(np.floor(${tx} / 100 * w + 0.5))  # % ของความกว้าง → พิกเซล`,
      `    ty = int(np.floor(${ty} / 100 * h + 0.5))  # % ของความสูง → พิกเซล`,
      "    cx, cy = (w - 1) / 2, (h - 1) / 2  # shear รอบกึ่งกลางภาพ",
      "    M = np.array([[1, kx, -kx * cy + tx], [ky, 1, -ky * cx + ty]], dtype=np.float64)",
      "    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)",
      // ปัดครึ่งขึ้นแบบ fixed2 ของแอป (:.2f เพียงอย่างเดียวปัด 0.125 เป็น 0.12 แต่แอปได้ 0.13)
      '    print(f"shifted ({tx:+d}, {ty:+d}) px, shear x={np.floor(kx * 100 + 0.5) / 100:.2f} y={np.floor(ky * 100 + 0.5) / 100:.2f}")',
    ];
  } },

  // IP batch 2 ชิ้น D — roipoly: ROI หลายเหลี่ยม 3–6 จุด หน่วย % · ผลรวมเลขจำนวนเต็ม + ลูปเดียวกับ registry.js จึงได้ตัวเลขเท่าแอปทุก bit
  roi_polygon: {
    needs: "any",
    outOf: (p) => ({ apply: "same", mask: "gray", overlay: "color" })[pickOwn(ROI_VIEW_PY, p.view, "apply")],
    gen: (p, ctx) => {
      const n = Math.round(clampNum(p.points, 4, 3, 6));
      const view = pickOwn(ROI_VIEW_PY, p.view, "apply");
      const pcts = [];
      for (let i = 1; i <= n; i++) {
        pcts.push(`(${clampNum(p[`p${i}x`], paramDefault("roi_polygon", `p${i}x`), 0, 100)}, ${clampNum(p[`p${i}y`], paramDefault("roi_polygon", `p${i}y`), 0, 100)})`);
      }
      const yellow = pyColor(ctx, [255, 255, 0]), white = pyColor(ctx, [255, 255, 255]);
      const output = {
        apply: ["img = cv2.bitwise_and(img, img, mask=roi_mask)  # นอกกรอบเป็นดำ"],
        mask: ["img = roi_mask  # ขาว = ใน ROI · ดำ = นอก ROI (ต่อเข้า B ของ Mask ได้)"],
        overlay: [
          "base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img",
          "dim = cv2.convertScaleAbs(base, alpha=0.4)  # นอกกรอบมืดลง",
          "dim[roi_mask > 0] = base[roi_mask > 0]  # ในกรอบ = ภาพเดิม",
          `cv2.polylines(dim, [roi_pts.reshape(-1, 1, 2)], True, ${yellow}, 2)`,
          "for i, (vx, vy) in enumerate(q, start=1):",
          `    cv2.circle(dim, (vx, vy), 3, ${yellow}, -1)`,
          `    cv2.putText(dim, str(i), (vx + 5, max(10, vy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ${white}, 1)`,
          "img = dim",
        ],
      }[view];
      return [
        "h, w = img.shape[:2]",
        `pcts = [${pcts.join(", ")}]  # จุด 1..${n} (x %, y %) ต่อกันตามลำดับแล้วปิดกลับจุด 1`,
        `roi_pts = np.array([[${pyPctToPx("px", "w")}, ${pyPctToPx("py", "h")}] for px, py in pcts], dtype=np.int32)`,
        "q = roi_pts.tolist()",
        "n = len(q)",
        "roi_mask = np.zeros((h, w), np.uint8)",
        "cv2.fillPoly(roi_mask, [roi_pts.reshape(-1, 1, 2)], 255)",
        "# วัดบนภาพขาวดำ: ผลรวมจำนวนเต็ม (int64) เท่ากับแอปทุก bit",
        `gray = img if img.ndim == 2 else cv2.cvtColor(img, ${pyGrayCode(ctx)})`,
        "inside = gray[roi_mask > 0].astype(np.int64)",
        "roi_area_px = int(inside.size)",
        "s1, s2 = int(inside.sum()), int((inside * inside).sum())",
        "roi_mean = s1 / roi_area_px",
        "roi_std = float(np.sqrt(max(0.0, s2 / roi_area_px - roi_mean * roi_mean)))  # ส่วนเบี่ยงเบนแบบประชากร",
        "roi_area_pct = 100 * roi_area_px / (w * h)",
        "roi_perimeter_px = 0.0",
        "for i in range(n):",
        "    j = 0 if i + 1 == n else i + 1",
        "    ex, ey = q[j][0] - q[i][0], q[j][1] - q[i][1]",
        "    roi_perimeter_px += float(np.sqrt(ex * ex + ey * ey))",
        "def roi_orient(ax, ay, bx, by, cx, cy):",
        "    v = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)",
        "    return (v > 0) - (v < 0)",
        "def roi_on_seg(ax, ay, bx, by, cx, cy):",
        "    return min(ax, bx) <= cx <= max(ax, bx) and min(ay, by) <= cy <= max(ay, by)",
        "def roi_touch(a, b, c, e):  # ด้าน a→b กับ c→e ตัดกันจริงหรือแตะกัน (รวมซ้อนทับแนวเดียวกัน)",
        "    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = u[a], u[b], u[c], u[e]",
        "    d1, d2 = roi_orient(x3, y3, x4, y4, x1, y1), roi_orient(x3, y3, x4, y4, x2, y2)",
        "    d3, d4 = roi_orient(x1, y1, x2, y2, x3, y3), roi_orient(x1, y1, x2, y2, x4, y4)",
        "    if d1 * d2 < 0 and d3 * d4 < 0:",
        "        return True",
        "    return ((d1 == 0 and roi_on_seg(x3, y3, x4, y4, x1, y1)) or (d2 == 0 and roi_on_seg(x3, y3, x4, y4, x2, y2))",
        "            or (d3 == 0 and roi_on_seg(x1, y1, x2, y2, x3, y3)) or (d4 == 0 and roi_on_seg(x1, y1, x2, y2, x4, y4)))",
        "# ด้านที่ไม่ติดกันแตะ/ตัดกัน = ลำดับจุดไขว้ (ด้านแรกกับด้านปิดติดกันที่จุด 1) · ตัดจุดที่ซ้ำจุดก่อนหน้า (รวมจุดสุดท้ายซ้ำจุดแรก) ก่อน — ด้านยาว 0 ทำให้ไขว้เท็จ",
        "u = []",
        "for pt in q:",
        "    if not u or u[-1] != pt:",
        "        u.append(pt)",
        "while len(u) > 1 and u[-1] == u[0]:",
        "    u.pop()",
        "m_u = len(u)",
        "roi_crossed = any(roi_touch(i, i + 1, j, 0 if j + 1 == m_u else j + 1)",
        "                  for i in range(m_u) for j in range(i + 2, m_u) if not (i == 0 and j == m_u - 1))",
        "# ทุกจุดอยู่บนเส้นตรงเดียว (รวมทุกจุดซ้อนกัน)",
        "k = 1",
        "while k < n and q[k] == q[0]:",
        "    k += 1",
        "roi_degenerate = k >= n or all((q[k][0] - q[0][0]) * (q[i][1] - q[0][1]) - (q[k][1] - q[0][1]) * (q[i][0] - q[0][0]) == 0",
        "                                for i in range(k + 1, n))",
        ...output,
        `info = f"ROI {n} จุด · {roi_area_px} px (${pyFixed1("roi_area_pct")}% ของภาพ) · ความสว่างเฉลี่ย ${pyFixed1("roi_mean")}"`,
        ...(view === "overlay" ? [] : ["info += \" (เลือก 'แสดงผลเป็น: ดูตำแหน่ง' เพื่อเห็นเส้นขอบและเลขจุด)\""]),
        "if roi_degenerate:",
        "    info += \" · ⚠️ ทุกจุดเรียงเป็นเส้นตรง/ซ้อนกัน ROI จึงเหลือแค่เส้นบาง ๆ\"",
        "elif roi_crossed:",
        "    info += \" · ⚠️ ลำดับจุดไขว้กัน รูปจะไม่ใช่รูปที่ตั้งใจ — เรียงจุดตามเข็มหรือทวนเข็มนาฬิกา\"",
        "print(info)",
        "for key, val in ((\"roi_vertices\", n), (\"roi_area_px\", roi_area_px), (\"roi_area_pct\", roi_area_pct), (\"roi_perimeter_px\", roi_perimeter_px),",
        "                 (\"roi_mean\", roi_mean), (\"roi_std\", roi_std), (\"roi_crossed\", roi_crossed)):",
        "    print(f\"{key} = {val!r}\")",
      ];
    },
  },

  // IP batch 2 ชิ้น E — superpixels (SLIC) · Lab ตอน gen ตาม isMirrored · ขาวดำ/สีตัดสินตอนรันด้วย ndim (ทนกรณี grabcut แบบ mask ที่ ctx.gray ยังเป็น false)
  slic: {
    needs: "any",
    outOf: (p) => (pickOwn(SLIC_VIEW_PY, p.view, "boundary") === "mean" ? "same" : "color"),
    gen: (p, ctx) => {
      const K = Math.round(clampNum(p.count, 200, 10, 1000));
      const m = clampNum(p.compactness, 10, 1, 40);
      const iters = Math.round(clampNum(p.iterations, 10, 1, 20));
      const view = pickOwn(SLIC_VIEW_PY, p.view, "boundary");
      const mirrored = isMirrored(ctx);
      const labCode = mirrored ? "cv2.COLOR_BGR2Lab" : "cv2.COLOR_RGB2Lab";
      const output = view === "mean" ? [
        "flat = full.ravel()",
        "cnt = np.bincount(flat, minlength=sp_count).astype(np.float64)",
        "src_img = img if img.ndim == 3 else img[:, :, None]",
        "means = np.stack([np.floor(np.bincount(flat, weights=src_img[:, :, c].ravel().astype(np.float64), minlength=sp_count) / cnt + 0.5) for c in range(src_img.shape[2])], axis=1)",
        "mean_img = means[full].astype(np.uint8)",
        "img = mean_img[:, :, 0] if img.ndim == 2 else mean_img  # ทุกพิกเซลในก้อนเดียวกันได้สีเฉลี่ยของก้อน",
      ] : [
        mirrored ? "edge_color = (0, 255, 255)  # เหลือง (ช่องของ cv2 กลับด้านจากแอป)" : "edge_color = (0, 255, 255) if img.ndim == 2 else (255, 255, 0)  # เหลือง",
        "if img.ndim == 2:",
        "    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
        "edge = np.zeros((rows, cols), bool)",
        "edge[:, :-1] |= full[:, 1:] != full[:, :-1]",
        "edge[:-1, :] |= full[1:, :] != full[:-1, :]",
        "img = img.copy()",
        "img[edge] = edge_color",
      ];
      return [
        ...slicPyLines(),
        "rows, cols = img.shape[:2]",
        "scale = min(1.0, 320 / max(rows, cols))  # วิเคราะห์บนภาพย่อไม่เกิน 320 px เหมือนแอป แล้วขยายป้ายกำกับกลับ",
        "a_cols, a_rows = max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))",
        "analysis = cv2.resize(img, (a_cols, a_rows), interpolation=cv2.INTER_AREA) if scale < 1 else img",
        `feat = analysis if analysis.ndim == 2 else cv2.cvtColor(analysis, ${labCode})`,
        `sp_labels, sp_count, sp_nx, sp_ny, sp_merged = slic_labels(feat, a_cols, a_rows, ${K}, ${m}, ${iters})`,
        "ya = ((2 * np.arange(rows) + 1) * a_rows) // (2 * rows)",
        "xa = ((2 * np.arange(cols) + 1) * a_cols) // (2 * cols)",
        "full = sp_labels[ya][:, xa]",
        ...output,
        `print(f"SLIC: {sp_count} superpixels (requested ${K}, grid {sp_nx}x{sp_ny}, merged {sp_merged} small fragments), analysis {a_cols}x{a_rows}")`,
      ];
    },
  },

  // IP batch 2 ชิ้น E — band-pass / band-reject / notch · DFT 64 บิตเต็มความละเอียดถึง 1920 px (ไม่ย่อที่ 640 แบบ fft_filter) · ผลตรงแอปทุกพิกเซล
  bandpass_notch: {
    needs: "gray",
    out: "gray",
    gen: (p) => {
      const kind = pickOwn(BANDPASS_KIND_PY, p.kind, "bandpass");
      const shape = pickOwn(BANDPASS_SHAPE_PY, p.shape, "gaussian");
      const view = pickOwn(BANDPASS_VIEW_PY, p.view, "result");
      const rIn = clampNum(p.rIn, 5, 0, 100), rOut = clampNum(p.rOut, 30, 1, 100);
      const prm = { lo: Math.min(rIn, rOut), hi: Math.max(rIn, rOut), nx: clampNum(p.nx, 25, -100, 100), ny: clampNum(p.ny, 0, -100, 100), nr: clampNum(p.nr, 5, 1, 30) };
      const swapped = kind !== "notch" && rIn > rOut;
      const output = view === "spectrum" ? [
        "spec = spec * H[:, :, None]",
        "lm = np.fft.fftshift(np.log(1 + np.sqrt(spec[:, :, 0] * spec[:, :, 0] + spec[:, :, 1] * spec[:, :, 1])))",
        "mn, mx = lm.min(), lm.max()",
        "out = np.zeros((Q, P), np.uint8) if mx - mn < 1e-3 else np.floor((lm - mn) * (255 / (mx - mn)) + 0.5).astype(np.uint8)",
        "img = cv2.resize(out, (cols, rows), interpolation=cv2.INTER_LINEAR)  # สเปกตรัมขยายเป็นขนาดภาพเสมอ",
        "flat = False",
      ] : [
        "spec = spec * H[:, :, None]",
        "inv = cv2.dft(spec, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)",
        "inv = inv[:a_rows, :a_cols]",
        ...(kind === "bandpass" ? [
          "mn, mx = inv.min(), inv.max()",
          "flat = bool(mx - mn < 1)  # ไม่มีรายละเอียดในช่วงนี้ → ภาพดำ",
          "out = np.zeros(inv.shape, np.uint8) if flat else np.floor((inv - mn) * (255 / (mx - mn)) + 0.5).astype(np.uint8)  # ยืดให้เต็ม 0-255",
        ] : [
          "flat = False",
          "out = np.clip(np.floor(inv + 0.5), 0, 255).astype(np.uint8)  # คงความสว่าง ค่าเกินตัดที่ 0/255",
        ]),
        "img = cv2.resize(out, (cols, rows), interpolation=cv2.INTER_LINEAR) if scale < 1 else out",
      ];
      return [
        "rows, cols = img.shape",
        "scale = min(1.0, 1920 / max(rows, cols))  # ย่อเฉพาะภาพที่ใหญ่กว่า 1920 px (ต่างจาก fft_filter ที่ย่อที่ 640)",
        "a_cols, a_rows = max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))",
        "analysis = cv2.resize(img, (a_cols, a_rows), interpolation=cv2.INTER_AREA) if scale < 1 else img",
        "P, Q = cv2.getOptimalDFTSize(a_cols), cv2.getOptimalDFTSize(a_rows)",
        "P += P % 2  # กลางสเปกตรัม = P/2 ต้องเป็นจำนวนเต็ม",
        "Q += Q % 2",
        "padded = cv2.copyMakeBorder(analysis, 0, Q - a_rows, 0, P - a_cols, cv2.BORDER_CONSTANT, value=0)",
        "spec = cv2.dft(padded.astype(np.float64), flags=cv2.DFT_COMPLEX_OUTPUT)  # 64 บิต",
        "hp, hq = P // 2, Q // 2",
        "xi, yi = np.arange(P), np.arange(Q)",
        "fx = (np.where(xi < hp, xi, xi - P) / hp)[None, :]  # พิกัดแบบไม่ shift: DC อยู่ที่ (0, 0)",
        "fy = (np.where(yi < hq, yi, yi - Q) / hq)[:, None]",
        ...bandGainPyLines(kind, shape, prm),
        "H = np.broadcast_to(H, (Q, P))",
        "dc_gain = float(H[0, 0])",
        ...output,
        `print(f"BANDPASS: ${kind}/${shape} spectrum {P}x{Q}")`,
        `print("swapped = ${swapped ? "True" : "False"}")`,
        'print(f"scaled = {scale < 1} analysis {a_cols}x{a_rows}")',
        view === "result" && kind !== "bandpass" ? 'print(f"dc_warning = {dc_gain < 0.5}")' : 'print("dc_warning = False")',
        'print(f"flat = {flat}")',
        `print("view = ${view}")`,
      ];
    },
  },

  // qa-Q1 (keypoints round S4 recheck): แอปจริงย่อภาพวิเคราะห์ไม่เกิน 480px เสมอสำหรับ harris_corners/
  // shi_tomasi_corners/orb_keypoints/akaze_keypoints/fast_keypoints (กันภาพ Full HD ช้าตอนเล่นกล้องสด)
  // แต่โค้ด Python ที่ export ไว้เดิมรันเต็มความละเอียดเสมอ — บนภาพกล้องทั่วไป (เช่น 640x480 ก็เกิน 480px
  // ด้านยาวแล้ว) ทำให้แอปสดกับสคริปต์ export เห็นจุดเด่น/มุมคนละชุดกันแทบทุกครั้ง ไม่ใช่แค่กรณีขอบ ๆ —
  // ตัดสินใจ mirror การย่อภาพให้ตรงกับแอปเป๊ะ (เหมือนที่ fft_magnitude/fft_filter ทำไว้แล้วใน Tier 1.5)
  // เพื่อให้ "โค้ดที่ export ไปตรงกับขั้นตอนจริงที่เห็นในแอป" สมชื่อปุ่ม "สร้างโค้ด Python ตามขั้นตอนจริง"
  // ไม่ใช่ "ปิดกั้น" ผู้เรียนไว้ให้เห็นแค่เวอร์ชัน full-res ที่ต่างจากที่เห็นสด — ใช้ int(x+0.5) ไม่ใช่ round()
  // ทุกจุดที่ปัดเลขจาก runtime dimensions (บทเรียนจาก FFT downscale bug ที่ qa เจอรอบ Tier 1.5)
  harris_corners: { needs: "gray", out: "color", gen: (p) => {
    const ksize = p.ksize % 2 === 0 ? p.ksize + 1 : p.ksize;
    return [
      "rows, cols = img.shape",
      "scale = min(1.0, 480 / max(rows, cols))  # ย่อก่อนวิเคราะห์เหมือนแอปตัวจริง",
      "if scale < 1:",
      "    analysis = cv2.resize(img, (max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))), interpolation=cv2.INTER_AREA)",
      "else:",
      "    analysis = img",
      "gray_f = np.float32(analysis)",
      `resp = cv2.cornerHarris(gray_f, ${int(p.blockSize)}, ${ksize}, ${num(p.k)})`,
      "resp = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX)",
      "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
      // B5 (backlog 2026-10-17): ทำเหมือนแอปเป๊ะ — ปัดเป็น uint8 → threshold → dilate วงรี 3x3 (กากบาท 5 พิกเซล: ขยายจุด ไม่ได้รวมมุมเป็นจุดเดียว) แล้วนับพิกเซลของมาสก์ที่ขยายแล้ว (ตัวเลขอาจมากกว่าจำนวนมุมจริงราว 5 เท่า)
      "mask8 = cv2.convertScaleAbs(resp)  # float -> uint8 ปัดเศษเหมือน convertTo ของแอป",
      `_, mask8 = cv2.threshold(mask8, ${num(p.thresh)} / 100 * 255, 255, cv2.THRESH_BINARY)`,
      "mask8 = cv2.dilate(mask8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # ขยายจุด (กากบาท 5 px) แล้วนับพิกเซล เหมือนแอป",
      "ys, xs = np.where(mask8 > 0)",
      "inv_scale = 1 / scale",
      "if len(xs) > 3000:  # เหมือนแอป: มุมเยอะเกินไม่วาดอะไร",
      "    print('จุดมุมเยอะเกิน (>3000) — เพิ่มค่า \"ความไวจับมุม\" ขึ้น')",
      "else:",
      "    print(f'พบมุม {len(xs)} จุด')",
      "    for x, y in zip(xs, ys):",
      "        cv2.circle(img, (int(x * inv_scale), int(y * inv_scale)), 3, (0, 255, 0), 1)",
    ];
  } },

  shi_tomasi_corners: { needs: "gray", out: "color", gen: (p) => [
    "rows, cols = img.shape",
    "scale = min(1.0, 480 / max(rows, cols))  # ย่อก่อนวิเคราะห์เหมือนแอปตัวจริง",
    "if scale < 1:",
    "    analysis = cv2.resize(img, (max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))), interpolation=cv2.INTER_AREA)",
    "else:",
    "    analysis = img",
    // minDist วัดเป็น px ของภาพที่วิเคราะห์ (analysis) เหมือนฝั่ง JS ที่คูณด้วย scale (registry.js's
    // shi_tomasi_corners: `p.minDist * scale`) — ไม่ใช่ px ของภาพเต็มความละเอียด ต้องคูณ scale ให้ตรงกัน
    `corners = cv2.goodFeaturesToTrack(analysis, ${int(p.maxCorners)}, ${num(p.quality) / 100}, ${int(p.minDist)} * scale,`,
    `                                  blockSize=${int(p.blockSize)}, useHarrisDetector=${boolOr(p.useHarris, false) ? "True" : "False"})`,
    "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
    "inv_scale = 1 / scale",
    "if corners is not None:",
    "    for x, y in corners.reshape(-1, 2):",
    // qa delta #17 MAJOR-B(3): `corners` is float32 (goodFeaturesToTrack's dtype), and under NumPy 2's
    // weak-scalar-promotion rules `float32 * python_float` stays float32 (loses precision vs JS, which
    // multiplies in float64) — cast to Python float first so the multiply matches the JS math exactly
    "        cv2.circle(img, (int(float(x) * inv_scale), int(float(y) * inv_scale)), 4, (0, 255, 0), 2)",
  ] },

  // แก้ไขระหว่าง S4 recheck (qa-Q1): เดิมคอมเมนต์ตรงนี้บอกว่า "ตั้งใจไม่ย่อภาพ ให้ผลแม่นกว่า" แต่พบว่า
  // ขัดกับ fft_magnitude/fft_filter ที่ mirror การย่อภาพไว้แล้ว และภาพกล้องทั่วไป (เช่น 640x480) ก็เกิน
  // 480px ด้านยาวอยู่แล้ว ทำให้แอปสดกับสคริปต์ export เห็นจุดเด่นคนละชุดกันเกือบทุกครั้ง ไม่ใช่กรณีขอบ ๆ
  // (วัดจริง: top-10-by-response overlap เหลือแค่ 3/10 ที่ 640x480) — เปลี่ยนมา mirror การย่อภาพ ≤480px
  // ให้ตรงกับ registry.js เป๊ะ (เหมือน harris_corners/shi_tomasi_corners ด้านบนที่แก้พร้อมกันรอบนี้)
  orb_keypoints: { needs: "gray", out: "color", gen: (p) => {
    // ห้ามใช้ `int(p.maxFeatures) || 300` — ค่า 0 ที่ตั้งใจส่งมาจริง ๆ (ไม่ใช่ค่าที่ขาด) จะโดน `||` เขวี้ยง
    // กลับไปเป็น default 300 ทั้งที่ JS ฝั่ง registry.js ใช้ Number.isFinite() เช็คแค่ "เป็นเลขจริงไหม" เท่านั้น
    // (0 ถือเป็นค่าที่ใช้ได้) ต้องใช้ finiteOr() ให้ตรงกับพฤติกรรม JS เป๊ะ (ดู comment ของ finiteOr ด้านบน)
    // clamp ขอบบนไว้ที่ 1000 เหมือนฝั่ง JS ด้วย (native cv2.ORB_create(nfeatures=...) ค่าใหญ่ผิดปกติ เช่น
    // >=1e9 จาก project ที่แก้ไฟล์ตรง ๆ ทำให้ allocate พังได้ — UI ปกติ max แค่ 1000 อยู่แล้ว)
    const maxFeatures = Math.min(1000, Math.max(1, int(finiteOr(p.maxFeatures, 300))));
    return [
      "rows, cols = img.shape",
      "scale = min(1.0, 480 / max(rows, cols))  # ย่อก่อนวิเคราะห์เหมือนแอปตัวจริง",
      "if scale < 1:",
      "    analysis = cv2.resize(img, (max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))), interpolation=cv2.INTER_AREA)",
      "else:",
      "    analysis = img",
      `orb = cv2.ORB_create(nfeatures=${maxFeatures})`,
      // ภาพเล็กเกินไป (ด้านใดด้านหนึ่ง <3px ของภาพที่วิเคราะห์ analysis) ทำให้ .detect() ของ ORB/AKAZE
      // throw ใน opencv.js จริง (ดู MAJOR-4 ใน WORKLOG) — mirror guard เดียวกันไว้ฝั่ง Python
      "kps = orb.detect(analysis, None) if min(analysis.shape[:2]) >= 3 else []",
      // ยืนยันเชิงประจักษ์แล้ว (JS/Python ตรงกันเป๊ะ) ว่า nfeatures ไม่ใช่ hard cap จริงในทุกภาพ (บางภาพ
      // ได้จุดเกิน nfeatures ไปมาก) — ต้องตัดด้วย response ranking ซ้ำ ให้ตรงกับพฤติกรรมแอปจริงเป๊ะ (ดู
      // comment เดียวกันใน registry.js's orb_keypoints.apply) — ใช้ KEYPOINT_SORT_KEY (มี tie-break ตายตัว
      // ต่อจาก response) ไม่ใช่ reverse=True เฉย ๆ เพื่อให้จุดตัด top-N ตรงกับฝั่ง JS เป๊ะแม้ response จะซ้ำกัน
      `kps = sorted(kps, ${KEYPOINT_SORT_KEY})[:${maxFeatures}]`,
      "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
      "inv_scale = 1 / scale",
      "for kp in kps:",
      "    x, y = kp.pt[0] * inv_scale, kp.pt[1] * inv_scale",
      // qa delta #17 MAJOR-B(1): must mirror registry.js's orb_keypoints exactly — JS computes r as an
      // UNROUNDED float (`Math.max(2, (kp.size/2)*invScale)`) and uses that same unrounded r for BOTH the
      // circle radius (rounded only at the cv.circle call via Math.round -> int(r+0.5), since r is always
      // >=2.0/positive here) and the orientation-line endpoint. The previous code truncated via
      // `int(kp.size/2*inv_scale)` BEFORE taking max(2,...), which under-sized every circle (e.g. FAST
      // size 7 -> r=3 in the export vs the app's r=4) — `(int(x), int(y))` centers stay as-is: confirmed
      // opencv.js/embind truncates float Point coords the same way Python's int() does for positive values.
      "    r = max(2.0, kp.size / 2 * inv_scale)",
      "    cv2.circle(img, (int(x), int(y)), int(r + 0.5), (0, 255, 0), 1)",
      "    if kp.angle != -1:",
      "        rad = np.deg2rad(kp.angle)",
      "        ex, ey = int(x + r * np.cos(rad)), int(y + r * np.sin(rad))",
      "        cv2.line(img, (int(x), int(y)), (ex, ey), (0, 255, 0), 1)",
    ];
  } },

  akaze_keypoints: { needs: "gray", out: "color", gen: (p) => {
    const maxPoints = Math.max(1, int(finiteOr(p.maxPoints, 300)));
    return [
      "# หมายเหตุ: cv2.AKAZE_create ต้องมี opencv-contrib-python ติดตั้งไว้ (บาง pip install opencv-python",
      "# แบบมาตรฐานไม่มี AKAZE/BRISK ในตัว) ถ้าเจอ AttributeError ให้ลอง: pip install opencv-contrib-python",
      "rows, cols = img.shape",
      "scale = min(1.0, 480 / max(rows, cols))  # ย่อก่อนวิเคราะห์เหมือนแอปตัวจริง",
      "if scale < 1:",
      "    analysis = cv2.resize(img, (max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))), interpolation=cv2.INTER_AREA)",
      "else:",
      "    analysis = img",
      "akaze = cv2.AKAZE_create()",
      "kps = akaze.detect(analysis, None) if min(analysis.shape[:2]) >= 3 else []",
      `kps = sorted(kps, ${KEYPOINT_SORT_KEY})[:${maxPoints}]`,
      "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
      "inv_scale = 1 / scale",
      "for kp in kps:",
      "    x, y = kp.pt[0] * inv_scale, kp.pt[1] * inv_scale",
      // qa delta #17 MAJOR-B(1) — same radius fix as orb_keypoints above, see comment there
      "    r = max(2.0, kp.size / 2 * inv_scale)",
      // qa delta #17 MAJOR-B(2): colors must be converted RGB->BGR like every other codegen op with a
      // fixed drawing color (watershed/hough precedent) — registry.js draws Scalar(0,200,255) on an RGB
      // Mat (sky blue); cv2 draws on BGR, so the swapped tuple is (255,200,0), not the unswapped RGB tuple
      // (which would render yellow/orange in the exported image instead of matching the app's blue)
      "    cv2.circle(img, (int(x), int(y)), int(r + 0.5), (255, 200, 0), 1)",
      "    if kp.angle != -1:",
      "        rad = np.deg2rad(kp.angle)",
      "        ex, ey = int(x + r * np.cos(rad)), int(y + r * np.sin(rad))",
      "        cv2.line(img, (int(x), int(y)), (ex, ey), (255, 200, 0), 1)",
    ];
  } },

  fast_keypoints: { needs: "gray", out: "color", gen: (p) => {
    const threshold = Math.max(1, int(finiteOr(p.threshold, 10)));
    const maxPoints = Math.max(1, int(finiteOr(p.maxPoints, 300)));
    return [
      "rows, cols = img.shape",
      "scale = min(1.0, 480 / max(rows, cols))  # ย่อก่อนวิเคราะห์เหมือนแอปตัวจริง",
      "if scale < 1:",
      "    analysis = cv2.resize(img, (max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))), interpolation=cv2.INTER_AREA)",
      "else:",
      "    analysis = img",
      `fast = cv2.FastFeatureDetector_create(threshold=${threshold})`,
      "kps = fast.detect(analysis, None) if min(analysis.shape[:2]) >= 3 else []",
      `kps = sorted(kps, ${KEYPOINT_SORT_KEY})[:${maxPoints}]`,
      "img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
      "inv_scale = 1 / scale",
      "for kp in kps:",
      "    x, y = kp.pt[0] * inv_scale, kp.pt[1] * inv_scale",
      // qa delta #17 MAJOR-B(1) — same radius fix as orb_keypoints above, see comment there
      "    r = max(2.0, kp.size / 2 * inv_scale)",
      // qa delta #17 MAJOR-B(2): registry.js draws Scalar(255,128,0) on RGB (orange) — swapped to BGR
      // (0,128,255) to match; unswapped would render blue in the exported image instead of orange
      "    cv2.circle(img, (int(x), int(y)), int(r + 0.5), (0, 128, 255), 1)",
    ];
  } },

  // ---- Frequency Domain ----
  // หมายเหตุสำคัญ: แอปตัวจริงย่อภาพลงไม่เกิน 640px ก่อนคำนวณ FFT เสมอ (กันกล้องสดค้างที่ 1080p, ~64ms/เฟรม
  // เต็มความละเอียด vs ~16ms ที่ 640px) แล้วขยายผลลัพธ์กลับ — โค้ด Python นี้ต้องทำแบบเดียวกัน ไม่เช่นนั้น
  // ภาพที่ใหญ่กว่า 640px จะได้ผลต่างจากที่เห็นในแอปจริง (spectrum และ high-pass ไวต่อขนาดภาพที่วิเคราะห์)
  fft_magnitude: { needs: "gray", out: "gray", gen: () => [
    "rows, cols = img.shape",
    "scale = min(1.0, 640 / max(rows, cols))  # ย่อก่อนวิเคราะห์เหมือนแอปตัวจริง (กันภาพใหญ่ช้า/ผลต่างจากแอป)",
    "if scale < 1:",
    "    analysis = cv2.resize(img, (max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))), interpolation=cv2.INTER_AREA)",
    "else:",
    "    analysis = img",
    "a_rows, a_cols = analysis.shape",
    "opt_rows, opt_cols = cv2.getOptimalDFTSize(a_rows), cv2.getOptimalDFTSize(a_cols)",
    "opt_rows += opt_rows % 2  # pad ให้เป็นเลขคู่เหมือน JS (fftshift/ifftshift ต้องการมิติคู่)",
    "opt_cols += opt_cols % 2",
    "padded = cv2.copyMakeBorder(analysis, 0, opt_rows - a_rows, 0, opt_cols - a_cols, cv2.BORDER_CONSTANT, value=0)",
    "complex_i = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)",
    "complex_i = np.fft.fftshift(complex_i, axes=(0, 1))",
    "mag = cv2.magnitude(complex_i[:, :, 0], complex_i[:, :, 1])",
    "mag = np.log(mag + 1)",
    "mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)",
    "img = cv2.resize(mag.astype(np.uint8), (cols, rows))",
  ] },

  fft_filter: { needs: "gray", out: "gray", gen: (p) => {
    const cutoff = num(p.cutoff) || 30;
    const isHigh = p.type === "high";
    const isIdeal = p.shape === "ideal";
    // IP batch 4A: Butterworth เปลี่ยนเฉพาะบรรทัด low = ... บรรทัดเดียว (ฝาแฝดของ 1 / (1 + (D/D0)^2n) ในแอป) · รูปแบบอื่น/ค่าแปลกได้บรรทัดเดิมทุกตัวอักษร
    const isButter = p.shape === "butterworth";
    const bwN = Math.round(clampNum(p.order, 2, 1, 10));
    return [
      "rows, cols = img.shape",
      "scale = min(1.0, 640 / max(rows, cols))  # ย่อก่อนวิเคราะห์เหมือนแอปตัวจริง — cutoff ด้านล่างอ้างอิงภาพที่ย่อแล้วนี้เสมอ",
      "if scale < 1:",
      "    analysis = cv2.resize(img, (max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))), interpolation=cv2.INTER_AREA)",
      "else:",
      "    analysis = img",
      "a_rows, a_cols = analysis.shape",
      "opt_rows, opt_cols = cv2.getOptimalDFTSize(a_rows), cv2.getOptimalDFTSize(a_cols)",
      "opt_rows += opt_rows % 2  # pad ให้เป็นเลขคู่เหมือน JS (fftshift/ifftshift ต้องการมิติคู่)",
      "opt_cols += opt_cols % 2",
      "padded = cv2.copyMakeBorder(analysis, 0, opt_rows - a_rows, 0, opt_cols - a_cols, cv2.BORDER_CONSTANT, value=0)",
      "complex_i = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)",
      "complex_i = np.fft.fftshift(complex_i, axes=(0, 1))",
      "cy, cx = opt_rows / 2, opt_cols / 2",
      "yy, xx = np.mgrid[0:opt_rows, 0:opt_cols]",
      "d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)",
      isIdeal ? `low = (d <= ${cutoff}).astype(np.float32)`
        : isButter ? `low = (1 / (1 + (d / ${cutoff}) ** ${2 * bwN})).astype(np.float32)  # Butterworth ลำดับ n=${bwN}`
        : `low = np.exp(-(d ** 2) / (2 * ${cutoff} ** 2)).astype(np.float32)`,
      isHigh ? "mask = 1 - low" : "mask = low",
      "complex_i[:, :, 0] *= mask",
      "complex_i[:, :, 1] *= mask",
      "complex_i = np.fft.ifftshift(complex_i, axes=(0, 1))",
      "inv = cv2.idft(complex_i, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)",
      "inv = inv[:a_rows, :a_cols]",
      "inv = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)",
      "img = cv2.resize(inv, (cols, rows))  # ขยายกลับขนาดภาพจริง (เหมือนแอปตัวจริง)",
    ];
  } },

  // ---- Image Arithmetic (Tier 2, 18a) ----
  // img_b มาจากการเรนเดอร์ src2 sub-pipeline ไว้ก่อนแล้ว (ดู buildPythonCode2Image) — ที่นี่แค่รวมสองภาพ
  //
  // CV-eng recheck (18a): registry.js's imabsdiff.apply() ตอนนี้มี fast-path พิเศษที่คง output เป็นขาวดำ
  // (1-channel) เมื่อภาพทั้งคู่เป็นขาวดำอยู่แล้ว (ไม่ upconvert เป็น RGB โดยไม่จำเป็น) — โค้ด Python ด้านล่าง
  // นี้ **ตั้งใจไม่ mirror พฤติกรรมนั้น** (ยัง upconvert เป็น BGR เสมอเหมือนเดิม) เพราะพิสูจน์แล้วว่าการคง
  // ขาวดำไว้จะทำให้ downstream step ที่ต้องการภาพขาวดำ (needs:"gray") พังจริง — buildPythonCode2Image
  // track "ภาพตอนนี้เป็นขาวดำหรือไม่" ด้วยตัวแปร ctx.gray แบบ compile-time (รู้ตอน gen โค้ด ไม่ใช่ runtime)
  // และ GEN.imabsdiff ประกาศ out:"color" ไว้ตายตัว (เพราะไม่รู้ ณ ตอน gen ว่า B จะเป็นขาวดำด้วยหรือเปล่า —
  // ctxB ของ src2 sub-pipeline เป็นตัวแปร local ที่ไม่ได้ถูกส่งมาที่ gen() ของ step หลัก) ถ้าให้ output จริง
  // เป็นขาวดำ (1-channel) แต่ codegen ยังคง track ว่าเป็น "color" (ctx.gray=false) ต่อไป step ถัดไปที่ขอ
  // ภาพขาวดำจะแทรก `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` ซึ่ง**throw จริง**เมื่อรับภาพ 2D เข้ามา (ยืนยัน
  // แล้วด้วย cv2 5.0.0 จริง: "Bad number of channels") — ทางแก้ที่ถูกต้องสมบูรณ์คือเปลี่ยน downstream gray-
  // check ทั้งระบบให้เช็ค .ndim ที่ runtime แทน compile-time flag แต่นั่นเป็นการเปลี่ยนกลไกหลักของ codegen
  // ทั้งไฟล์ ใหญ่เกินขอบเขตของรอบนี้ — เลือกทางปลอดภัยกว่า (คง Python ไว้แบบเดิม, ยอมรับว่า channel count
  // ของภาพต่างจากแอปสด) แทน เพราะค่าตัวเลขที่ได้ยังตรงกันเป๊ะทุกจุด (ภาพขาวดำที่ห่อเป็น 3 ช่อง R=G=B ก็คือ
  // ค่าเดียวกัน มองด้วยตาไม่ต่างกันเลย) ต่างกันแค่รูปแบบการเก็บข้อมูล ไม่ใช่ความหมาย/ผลลัพธ์ที่ผิดเพี้ยน
  imabsdiff: { needs: "any", out: "color", gen: (p, ctx) => {
    const lines = [];
    if (ctx.gray) { lines.push("img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)"); ctx.gray = false; }
    lines.push(
      "if img_b.ndim == 2:",
      "    img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)",
      "if img.shape[:2] != img_b.shape[:2]:",
      "    img_b = cv2.resize(img_b, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)",
      "img = cv2.absdiff(img, img_b)  # |A-B|",
    );
    return lines;
  } },

  // ---- Image Arithmetic (Tier 2, 18b) ----
  // ใช้ precedent เดียวกับ GEN.imabsdiff ด้านบนเป๊ะ (ไม่ mirror bothGray fast-path ของฝั่ง JS — ยัง
  // upconvert เป็น BGR เสมอ กันไม่ให้ downstream step ที่ต้องการภาพขาวดำพังตอน export — ดูคอมเมนต์เต็มที่
  // GEN.imabsdiff ด้านบนสำหรับเหตุผลละเอียด ค่าตัวเลขยังตรงกันเป๊ะ ต่างแค่รูปแบบ channel)
  imfuse: { needs: "any", out: "color", gen: (p, ctx) => {
    const lines = [];
    if (ctx.gray) { lines.push("img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)"); ctx.gray = false; }
    lines.push(
      "if img_b.ndim == 2:",
      "    img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)",
      "if img.shape[:2] != img_b.shape[:2]:",
      "    img_b = cv2.resize(img_b, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)",
    );
    if (p.mode === "checkerboard") {
      const tile = Math.max(1, int(finiteOr(p.tileSize, 32)));
      // np.where แบบ boolean เท่ากับ bitwise_and x2 + add ของฝั่ง JS เป๊ะเชิงตัวเลข (mask เป็น 0/255 ไบนารี
      // ไม่มีการ blend เลย — เลือก A หรือ B ล้วน ๆ ต่อพิกเซล) แค่เขียนสั้นกว่าด้วย numpy โดยตรง
      lines.push(
        "h, w = img.shape[:2]",
        "yy, xx = np.mgrid[0:h, 0:w]",
        `tile = ${tile}`,
        "checker = ((xx // tile) + (yy // tile)) % 2 == 0",
        "img = np.where(checker[:, :, None], img, img_b)",
      );
    } else {
      const alphaPct = Math.max(0, Math.min(100, finiteOr(p.alpha, 50)));
      const alpha = alphaPct / 100;
      lines.push(`img = cv2.addWeighted(img, ${alpha}, img_b, ${1 - alpha}, 0)`);
    }
    return lines;
  } },

  // mask ไม่บังคับ channel ของ A เลย (ต่างจาก imfuse/imabsdiff) — out:"same" เพราะ bitwise_and ไม่เปลี่ยน
  // จำนวนช่องสีของ A แต่อย่างใด (เหมือน JS's cv.bitwise_and(src,src,dst,finalMask))
  mask: { needs: "any", out: "same", gen: (p) => {
    const lines = [
      "if img_b.ndim == 3:",
      "    img_b_gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)",
      "else:",
      "    img_b_gray = img_b",
      "if img.shape[:2] != img_b_gray.shape[:2]:",
      "    img_b_gray = cv2.resize(img_b_gray, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)",
      "_, mask_bin = cv2.threshold(img_b_gray, 127, 255, cv2.THRESH_BINARY)",
    ];
    if (boolOr(p.invert, false)) lines.push("mask_bin = cv2.bitwise_not(mask_bin)");
    lines.push("img = cv2.bitwise_and(img, img, mask=mask_bin)");
    return lines;
  } },

  // ---- Image Arithmetic เพิ่มเติม (IP batch 2 ชิ้น B, 2026-10-02) ----
  // ใช้ฟังก์ชันของ cv2 เสมอ ไม่ใช้ตัวดำเนินการของ numpy (img + img_b ของ uint8 วนกลับ ไม่ตัดที่ 255) ·
  // เช็คจำนวนช่องด้วย .ndim ตอนรัน (ไม่พึ่ง ctx.gray) · ใช้ตัวแปรของตัวเอง ไม่เขียนทับ img_b ·
  // ภาพ B ที่ใช้เป็นภาพขาวดำ (logical_ops/inpaint) ใช้ bBinaryLines() ร่วมกัน
  //
  // imarith: out:"color" ตาม precedent ของ GEN.imabsdiff (ห้าม mirror fast path ขาวดำของแอป — เหตุผลอยู่ที่
  // คอมเมนต์ของ GEN.imabsdiff) ค่าตัวเลขตรงกันทุกพิกเซล ต่างแค่จำนวนช่อง
  imarith: { needs: "any", out: "color", gen: (p) => [
    "if img.ndim == 2:",
    "    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
    "arith_b = img_b if img_b.ndim == 3 else cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)",
    "if img.shape[:2] != arith_b.shape[:2]:",
    "    arith_b = cv2.resize(arith_b, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)",
    pickOwn(IMARITH_PY, p.operation, "add"),
  ] },

  // logical_ops: out:"gray" — A และ B ถูก threshold 127 ทั้งคู่ (เหมือนแอป) ผลเป็น 0/255 ช่องเดียว
  logical_ops: { needs: "any", out: "gray", gen: (p) => [
    "if img.ndim == 3:",
    "    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
    ...bBinaryLines(),
    "_, a_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)",
    pickOwn(LOGICAL_PY, p.operation, "and"),
    'print(f"white pixels: {cv2.countNonZero(img)}")',
  ] },

  // inpaint: out:"same" · guard ทั้งสองข้อ mirror แอปด้วยเลขจำนวนเต็มแบบเดียวกัน (nz * 100 > 25 * total) ·
  // radius clamp ช่วงเดียวกับ registry (intIn(p.radius, 3, 1, 7))
  inpaint: { needs: "any", out: "same", gen: (p) => {
    const radius = Math.round(clampNum(p.radius, 3, 1, 7));
    const ns = p.method === "ns";
    const flag = ns ? "cv2.INPAINT_NS" : "cv2.INPAINT_TELEA";
    return [
      ...bBinaryLines(),
      "nz = cv2.countNonZero(b_bin)",
      "total = img.shape[0] * img.shape[1]",
      "if nz == 0:",
      '    print("inpaint skipped: mask is empty (no B pixel above 127)")',
      "elif nz * 100 > 25 * total:  # ครอบเกิน 25% ไม่ซ่อม (เหมือนแอป) — Inpaint ใช้กับรอยเล็ก ๆ",
      '    print(f"inpaint skipped: mask covers {-(-nz * 1000 // total) / 10:.1f}% of the image (> 25%)")  # ปัดขึ้นเหมือนแอป',
      "else:",
      `    img = cv2.inpaint(img, b_bin, ${radius}, ${flag})`,
      `    print(f"inpainted {nz} px ({int(nz * 1000 / total + 0.5) / 10:.1f}%) with ${ns ? "Navier-Stokes" : "Telea"} r=${radius}")`,
    ];
  } },

  // hist_match: out:"same" · CDF แบบ float64 (.astype ก่อน cumsum — NumPy 2 สะสม float32 แล้วเพี้ยน ดู
  // contrast_stretch) · searchsorted side='left' = j แรกที่ cdf_b[j] >= cdf_a[i] เหมือนลูปของแอป · ไม่ resize B
  hist_match: { needs: "any", out: "same", gen: () => [
    "if img.ndim == 2:",
    "    hm_ref = img_b if img_b.ndim == 2 else cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)",
    "    planes_a, planes_b = [img], [hm_ref]",
    "else:",
    "    planes_a = cv2.split(img)",
    "    planes_b = cv2.split(img_b) if img_b.ndim == 3 else [img_b] * 3  # B ขาวดำ: ใช้ histogram เดียวกับทุกช่อง",
    "luts = []",
    "for pa, pb in zip(planes_a, planes_b):",
    "    h_a = cv2.calcHist([pa], [0], None, [256], [0, 256]).flatten().astype(np.float64)",
    "    h_b = cv2.calcHist([pb], [0], None, [256], [0, 256]).flatten().astype(np.float64)",
    "    cdf_a = np.cumsum(h_a) / h_a.sum()",
    "    cdf_b = np.cumsum(h_b) / h_b.sum()",
    "    luts.append(np.clip(np.searchsorted(cdf_b, cdf_a, side='left'), 0, 255).astype(np.uint8))",
    "if img.ndim == 2:",
    "    img = cv2.LUT(img, luts[0])",
    "else:",
    "    img = cv2.merge([cv2.LUT(pa, lut) for pa, lut in zip(planes_a, luts)])",
    'print("histogram of A matched to B")',
  ] },

  // ---- Template Matching (Tier 2, 18b, แก้รอบ scope-change ที่ 2) ----
  // ย่อภาพก่อนหาตำแหน่งคร่าว ๆ เสมอ (ต้อง mirror JS — matchTemplate เต็ม 1080p ทั้งเฟรมวัดจริง ~70-90ms
  // ช้าเกินสำหรับกล้องสด, ย่อเหลือ ≤640px ด้านยาว ~6ms) แล้วขยาย matchLoc กลับด้วย int(x+0.5) (ไม่ใช่
  // round() เปล่า ๆ — Iron Rule) — จากนั้น "Option B" (ตัดสินใจร่วมกับ coordinator หลัง qa พบว่า downscaled
  // match_score คลาดเคลื่อนได้มาก 100%→87.7% ที่ 1920x1080): re-score ที่ full-res เฉพาะใน ROI เล็ก ๆ รอบ
  // ตำแหน่งที่เจอ (วัดจริง ~11-23ms ที่ 1920x1080 — เร็วพอ ๆ กับตอนย่อภาพอย่างเดียว) แทนที่จะรัน matchTemplate
  // เต็มเฟรมที่ full-res — ผลคือ score/ตำแหน่งที่แสดงและใช้ตัดสิน threshold เป็นค่าจริงที่ full-res เสมอ
  // ไม่ใช่แค่ "แจ้งเตือน" — guard template>search ก่อนเรียก matchTemplate เสมอ (ยืนยันแล้วว่า cv2 ไม่ throw
  // คืนผลลัพธ์เพี้ยนแทน)
  template_matching: { needs: "gray", out: "color", gen: (p) => {
    const method = p.method === "ccorr" ? "cv2.TM_CCORR_NORMED" : "cv2.TM_CCOEFF_NORMED";
    const thresholdPct = finiteOr(p.threshold, 70);
    return [
      "if img_b.ndim == 3:",
      "    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)",
      "rows, cols = img.shape",
      "t_rows, t_cols = img_b.shape",
      "img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
      "if t_cols > cols or t_rows > rows:",
      "    print(\"ภาพ template (B) ต้องเล็กกว่าภาพหลักทั้งกว้างและสูง\")",
      "    img = img_bgr",
      "else:",
      "    scale = min(1.0, 640 / max(rows, cols))  # ย่อภาพค้นหา+template พร้อมกันเหมือนแอปตัวจริง (หาตำแหน่งคร่าว ๆ)",
      "    if scale < 1:",
      "        a_search = cv2.resize(img, (max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))), interpolation=cv2.INTER_AREA)",
      "        a_tmpl = cv2.resize(img_b, (max(1, int(t_cols * scale + 0.5)), max(1, int(t_rows * scale + 0.5))), interpolation=cv2.INTER_AREA)",
      "    else:",
      "        a_search, a_tmpl = img, img_b",
      "    if a_tmpl.shape[0] > a_search.shape[0] or a_tmpl.shape[1] > a_search.shape[1]:",
      "        print(\"ภาพ template (B) ต้องเล็กกว่าภาพหลักทั้งกว้างและสูง\")",
      "        img = img_bgr",
      "    else:",
      `        result = cv2.matchTemplate(a_search, a_tmpl, ${method})`,
      "        _, max_val, _, max_loc = cv2.minMaxLoc(result)",
      "        inv_scale = 1 / scale",
      "        best_val = max_val",
      "        x = int(max_loc[0] * inv_scale + 0.5)",
      "        y = int(max_loc[1] * inv_scale + 0.5)",
      "        if scale < 1:",
      "            # margin กัน 2 แหล่งคลาดเคลื่อน: (1) quantization จากปัดเศษพิกัดใน downscaled space",
      "            # คูณกลับด้วย inv_scale (สูงสุด ~inv_scale px) (2) การเลื่อนตำแหน่งพีคจาก resize",
      "            # interpolation เอง (วัดจริงสูงสุด ~8px ที่ 1920x1080/template 100px) — mirror JS เป๊ะ",
      "            margin = int(np.ceil(inv_scale)) + 8",
      "            roi_x0 = max(0, x - margin)",
      "            roi_y0 = max(0, y - margin)",
      "            roi_x1 = min(cols, x + t_cols + margin)",
      "            roi_y1 = min(rows, y + t_rows + margin)",
      "            roi_w = roi_x1 - roi_x0",
      "            roi_h = roi_y1 - roi_y0",
      "            if roi_w >= t_cols and roi_h >= t_rows:",
      "                roi = img[roi_y0:roi_y1, roi_x0:roi_x1]",
      `                result_full = cv2.matchTemplate(roi, img_b, ${method})`,
      "                _, max_val_full, _, max_loc_full = cv2.minMaxLoc(result_full)",
      "                best_val = max_val_full",
      "                x = roi_x0 + max_loc_full[0]",
      "                y = roi_y0 + max_loc_full[1]",
      "        score_pct = best_val * 100",
      `        threshold_pct = ${thresholdPct}`,
      "        img = img_bgr",
      "        if score_pct >= threshold_pct:",
      "            cv2.rectangle(img, (x, y), (x + t_cols, y + t_rows), (0, 255, 0), 2)",
      "            print(f\"พบตำแหน่งที่ ({x}, {y}) — คะแนนความคล้าย {score_pct:.1f}%\")",
      "        else:",
      "            print(f\"ไม่พบ — คะแนนดีที่สุด {score_pct:.1f}% ต่ำกว่าเกณฑ์ {threshold_pct}%\")",
    ];
  } },

  // ---- Image Quality (Tier 2, 18b) ----
  // ทั้งคู่ needs:"gray" บังคับ A เป็นขาวดำอยู่แล้วเสมอ (ไม่มีความกำกวมแบบ imfuse/imabsdiff ที่เป็น
  // needs:"any") ผลลัพธ์ภาพจึงเป็นขาวดำเสมอ (out:"gray") ไม่ต้องมี fast-path/divergence-comment ใด ๆ
  psnr: { needs: "gray", out: "gray", gen: () => [
    "if img_b.ndim == 3:",
    "    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)",
    "if img.shape[:2] != img_b.shape[:2]:",
    "    img_b = cv2.resize(img_b, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)",
    "diff = cv2.absdiff(img, img_b)",
    "mse = float(np.mean(diff.astype(np.float64) ** 2))",
    // mse=0 -> Infinity เสมอ (ยืนยันเชิงประจักษ์ฝั่ง JS แล้วว่า log10(x/0) ได้ Infinity ตามธรรมชาติ แต่ฝั่ง
    // Python ต้อง branch เองชัดเจน เพราะ NumPy/Python raise หรือ warn RuntimeWarning: divide by zero แทนที่จะ
    // คืน inf เงียบ ๆ แบบ JS)
    "if mse == 0:",
    "    psnr = float('inf')",
    "    psnr_label = \"∞ (เหมือนกันทุกพิกเซล)\"",
    "else:",
    "    psnr = 10 * np.log10((255 * 255) / mse)",
    "    psnr_label = f\"{psnr:.2f} dB\"",
    "print(f\"PSNR = {psnr_label} (MSE = {mse:.2f})\")",
    "img = diff  # ภาพผลลัพธ์ = แผนที่ความต่าง |A-B| เหมือน imabsdiff ให้มีอะไรดูเสมอ",
  ] },

  // สูตร Wang et al. 2004 ผ่าน GaussianBlur (11x11, sigma=1.5) มือเปล่า — ยืนยันแล้วว่า cv2 5.0.0 ไม่มี
  // SSIM สำเร็จรูป (ไม่ใช่ skimage.metrics.structural_similarity ด้วย เพราะสคริปต์ที่ generate ในแอปนี้
  // import แค่ cv2/numpy เท่านั้น ไม่เพิ่ม dependency ใหม่) — แก้รอบ scope-change ที่ 2: เพิ่ม speed param
  // mirror JS (registry.js) — "full" (ดีฟอลต์ กัน backward-compat กับ pipeline เก่าที่ไม่มี key นี้ด้วย
  // p.speed undefined ต้องได้ "full" เสมอ) ไม่ย่อภาพ ค่าถูกต้องที่สุด (วัดจริงที่ 1080p ~82-127ms) — "fast"
  // ย่อ ≤640px ก่อนคำนวณเหมือนเดิม (~10ms) พร้อม downscale_note แจ้งเมื่อ scale<1 ทำงานจริง (qa วัดจริงพบว่า
  // downscale เปลี่ยนค่า SSIM ได้มาก: 1920x1080 full-res=0.3446 vs downscaled=0.9471 ~175% relative)
  ssim: { needs: "gray", out: "gray", gen: (p) => {
    const speed = p && p.speed === "fast" ? "fast" : "full";
    return [
    "if img_b.ndim == 3:",
    "    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)",
    "if img.shape[:2] != img_b.shape[:2]:",
    "    img_b = cv2.resize(img_b, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)",
    "rows, cols = img.shape",
    speed === "fast"
      ? "scale = min(1.0, 640 / max(rows, cols))  # โหมดเร็ว: ย่อก่อนคำนวณเหมือนแอปตัวจริง (GaussianBlur ซ้ำหลายครั้งช้าที่ full-res)"
      : "scale = 1.0  # โหมดละเอียดเต็ม (ดีฟอลต์) — ไม่ย่อภาพ คำนวณจากภาพจริงเสมอ ค่าที่ได้ถูกต้องที่สุด",
    "downscale_note = \" (คำนวณจากภาพที่ย่อขนาดลงก่อนเพื่อความเร็ว ค่าอาจต่างจากภาพความละเอียดเต็มจริงได้มาก โดยเฉพาะถ้าภาพมีรายละเอียดเล็ก ๆ หรือสัญญาณรบกวนมาก)\" if scale < 1 else \"\"",
    "if scale < 1:",
    "    a_a = cv2.resize(img, (max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))), interpolation=cv2.INTER_AREA)",
    "    a_b = cv2.resize(img_b, (max(1, int(cols * scale + 0.5)), max(1, int(rows * scale + 0.5))), interpolation=cv2.INTER_AREA)",
    "else:",
    "    a_a, a_b = img, img_b",
    "C1 = (0.01 * 255) ** 2",
    "C2 = (0.03 * 255) ** 2",
    "af = a_a.astype(np.float32)",
    "bf = a_b.astype(np.float32)",
    "mu_a = cv2.GaussianBlur(af, (11, 11), 1.5)",
    "mu_b = cv2.GaussianBlur(bf, (11, 11), 1.5)",
    "mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b",
    "sigma_a2 = cv2.GaussianBlur(af * af, (11, 11), 1.5) - mu_a2",
    "sigma_b2 = cv2.GaussianBlur(bf * bf, (11, 11), 1.5) - mu_b2",
    "sigma_ab = cv2.GaussianBlur(af * bf, (11, 11), 1.5) - mu_ab",
    "numer = (2 * mu_ab + C1) * (2 * sigma_ab + C2)",
    "denom = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)",
    "ssim_map = numer / denom",
    "ssim_mean = float(np.mean(ssim_map))",
    "print(f\"SSIM = {ssim_mean:.4f} (1 = เหมือนกันทุกประการ){downscale_note}\")",
    "normed = cv2.normalize(ssim_map, None, 0, 255, cv2.NORM_MINMAX)",
    "img = cv2.resize(normed.astype(np.uint8), (cols, rows))",
    ];
  } },

  // ---- IP batch 4A (2026-10-15) — 5 op ใหม่ ----
  // guided_filter: float32 ทุกขั้นเหมือนแอป (guided_f32) · สีทำทีละช่องด้วย cv2.split/merge · ปัดครึ่งขึ้นด้วย np.floor(x + 0.5)
  guided_filter: { needs: "any", out: "same", gen: (p) => {
    const r = Math.round(clampNum(p.radius, 2, 1, 20));
    const eps = clampNum(p.eps, 0.01, 0.001, 1);
    return [
      ...guidedPyLines(),
      `r = ${r}`,
      `eps255 = np.float32(${String(eps * 65025)})  # ε x 65025 (ทำงานบนสเกล 0-255)`,
      "f32 = img.astype(np.float32)",
      "if img.ndim == 2:",
      "    q = guided_f32(f32, f32, r, eps255)",
      "else:",
      "    q = cv2.merge([guided_f32(c, c, r, eps255) for c in cv2.split(f32)])",
      "img = np.clip(np.floor(q.astype(np.float64) + 0.5), 0, 255).astype(np.uint8)",
      `print(f"guided filter r=${r} eps=${eps}")`,
    ];
  } },

  // focus_score: กล่องกลางภาพ (ตัดด้วย slice) · ผลรวมเป็นจำนวนเต็ม int64 → score เท่าแอปเป๊ะ
  focus_score: { needs: "gray", out: "gray", gen: (p) => {
    const tenengrad = p.method === "tenengrad";
    const pct = clampNum(p.roiPct, 100, 10, 100);
    const pass = p.view === "pass";
    return [
      "rows, cols = img.shape",
      `rw = max(1, int(np.floor(cols * ${String(pct)} / 100 + 0.5)))  # กล่องกลางภาพกว้าง/สูง ${pct}%`,
      `rh = max(1, int(np.floor(rows * ${String(pct)} / 100 + 0.5)))`,
      "x0, y0 = (cols - rw) // 2, (rows - rh) // 2",
      "crop = img[y0:y0 + rh, x0:x0 + rw]  # วัดเฉพาะในกล่อง (Laplacian/Sobel ใช้ขอบสะท้อนของกล่องเอง)",
      "n = rw * rh",
      ...(tenengrad
        ? [
          "gx = cv2.Sobel(crop, cv2.CV_16S, 1, 0, ksize=3).astype(np.int64)",
          "gy = cv2.Sobel(crop, cv2.CV_16S, 0, 1, ksize=3).astype(np.int64)",
          "score = int((gx * gx + gy * gy).sum()) / n",
        ]
        : [
          "v = cv2.Laplacian(crop, cv2.CV_16S, ksize=1).astype(np.int64)",
          "s1, s2 = int(v.sum()), int((v * v).sum())",
          "score = s2 / n - (s1 / n) * (s1 / n)",
        ]),
      "mean_gray = int(crop.astype(np.int64).sum()) / n",
      `print(f"focus score (${tenengrad ? "Tenengrad" : "variance of Laplacian"}) = {np.floor(score * 10 + 0.5) / 10:.1f} in {rw}x{rh} px")`,
      'print(f"focus mean = {np.floor(mean_gray * 10 + 0.5) / 10:.1f}")',
      ...(pass ? [] : ["img = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_16S, ksize=1))  # ภาพ |Laplacian| ทั้งภาพ"]),
    ];
  } },

  // P-8 pass_fail — ตัดสินจาก snapshot `pf_up` ที่ขั้นวัดก่อนหน้าทิ้งไว้ (pfSnapshotLines) · ไม่มีค่า/มาจากกล่องอื่น = NG เหมือนแอป
  // รองรับเฉพาะ metric ที่มาจาก regionprops / roi_polygon / focus_score — delta_e ข้ามพร้อมคำเตือน · print เป็น ASCII ล้วน (V15)
  pass_fail: { needs: "any", out: "color", gen: (p, ctx) => {
    const q = passFailParams(p);
    const m = PASS_FAIL_METRICS[q.metric];
    const srcOp = ctx.pfSource; // op ล่าสุดบนสายหลักที่มี ctx.values (undefined = ยังไม่มี)
    const toColor = ["if img.ndim == 2:", "    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # แปลงเป็นภาพสีเพื่อให้กรอบสีเห็นได้"];
    if (srcOp === m.op && !PF_PY_SOURCES[m.op]) {
      return [
        `# metric '${q.metric}' มาจาก ${thaiName(m.op)} ซึ่งยังไม่รองรับการตัดสินในสคริปต์ Python — ข้ามขั้นนี้ (ภาพผ่านไปเฉย ๆ)`,
        ...toColor,
        `print("WARNING: skipped 'pass_fail' (metric from '${m.op}' is app-only in the Python export) - no RESULT, differs from the app")`,
      ];
    }
    const measure = srcOp === undefined
      ? ["pf_measured = None", "pf_reason = 'no upstream values'"]
      : srcOp !== m.op
        ? ["pf_measured = None", "pf_reason = 'values come from another box'"]
        : [
          ...({
            count: ["pf_measured = pf_up['count']"],
            // ผลรวมเป็นลูปเอง ไม่ใช้ sum() — Python 3.12+ รวมทศนิยมแบบชดเชยต่างจากแอปที่บวกทีละตัว
            area_sum: ["pf_measured = None", "for pf_a in pf_up['areas']:", "    pf_measured = pf_a if pf_measured is None else pf_measured + pf_a"],
            area_max: ["pf_measured = max(pf_up['areas']) if pf_up['areas'] else None"],
          }[q.metric] || ["pf_measured = pf_up['value']"]),
          "pf_reason = None",
          "if pf_measured is None or not np.isfinite(pf_measured):",
          "    pf_measured, pf_reason = None, 'value missing or not a number'",
        ];
    const cond = { range: "pf_lo <= pf_measured <= pf_hi", ge: "pf_measured >= pf_lo", le: "pf_measured <= pf_hi",
      eq: "abs(pf_measured - pf_lo) <= 1e-9 * max(1.0, abs(pf_lo))" }[q.cond];
    const green = pyColor(ctx, [0, 200, 0]), red = pyColor(ctx, [255, 0, 0]);
    return [
      `pf_lo, pf_hi = ${String(q.lo)}, ${String(q.hi)}  # เกณฑ์ (${q.cond}) — ใช้เฉพาะค่าที่เงื่อนไขนี้ต้องการ`,
      ...measure,
      "pf_ok = pf_measured is not None and (" + cond + ")",
      ...(q.cond === "range" ? ["if pf_measured is not None and pf_lo > pf_hi:", "    pf_reason = 'low limit is greater than high limit'"] : []),
      "img = np.ascontiguousarray(img)",
      ...toColor,
      "pf_h, pf_w = img.shape[:2]",
      "pf_t = max(2, int(np.floor(min(pf_w, pf_h) / 50 + 0.5)))  # ความหนากรอบ",
      `pf_color = ${green} if pf_ok else ${red}`,
      "cv2.rectangle(img, (pf_t // 2, pf_t // 2), (pf_w - 1 - pf_t // 2, pf_h - 1 - pf_t // 2), pf_color, pf_t)",
      "pf_scale = max(0.6, min(pf_w, pf_h) / 300)",
      "cv2.putText(img, 'OK' if pf_ok else 'NG', (pf_t + 8, pf_t + 8 + int(np.floor(20 * pf_scale + 0.5))), cv2.FONT_HERSHEY_SIMPLEX, pf_scale, pf_color, max(1, int(np.floor(pf_scale * 2 + 0.5))))",
      "print('RESULT: OK' if pf_ok else 'RESULT: NG')",
      "print(f'measured: {pf_measured!r}')",
      "if pf_reason:",
      "    print('reason: ' + pf_reason)",
    ];
  } },

  // pseudocolor: ตาราง 3 ช่องจากสูตรเดียวกับ PSEUDOCOLOR_MAPS · ภาพหลัง needs:"gray" เป็น 2 มิติ ผลเป็น BGR (mirrored ตามกฎ codegen)
  pseudocolor: { needs: "gray", out: "color", gen: (p) => {
    const mapId = Object.hasOwn(PSEUDOCOLOR_PY, p.map) ? p.map : "jet"; // select อ่านด้วย own-property เท่านั้น
    return [
      "i = np.arange(256)",
      ...PSEUDOCOLOR_PY[mapId],
      ...(p.range === "minmax"
        ? [
          "lo, hi = int(img.min()), int(img.max())  # ยืดค่าต่ำ-สูงของภาพให้เต็มแถบสี",
          "t = ((img.astype(np.int64) - lo) * 510 + (hi - lo)) // (2 * (hi - lo)) if hi > lo else np.zeros(img.shape, np.int64)",
        ]
        : ["t = img  # ตามค่าจริง 0-255"]),
      "img = cv2.merge([lut_b[t], lut_g[t], lut_r[t]])  # เรียงเป็น B,G,R",
      `print("pseudocolor: ${mapId}")`,
    ];
  } },

  // delta_e: 2 ภาพ (img_b มาจากสาย B) · เช็ค .ndim ตอนรัน · Lab float32 ผ่าน astype(float32) / 255 (ตรงกับ convertTo(1/255) ของแอป) · ผลรวมด้วย cumsum (ลำดับเดียวกับแอป)
  delta_e: { needs: "any", out: "gray", gen: (p) => {
    const ciede = p.formula === "ciede2000";
    const meanB = p.compare === "mean_b";
    const tol = clampNum(p.tol, 2.3, 0, 50), scaleMax = clampNum(p.scaleMax, 20, 1, 100);
    const over = p.view === "over";
    return [
      ...(ciede
        ? [
          "def delta_e2000(L1, a1, b1, L2, a2, b2):",
          "    # CIEDE2000 (Sharma, Wu, Dalal 2005; kL = kC = kH = 1) — ลำดับนิพจน์เดียวกับ deltaE2000 ของแอป",
          "    C1, C2 = np.sqrt(a1 * a1 + b1 * b1), np.sqrt(a2 * a2 + b2 * b2)",
          "    Cb7 = ((C1 + C2) / 2) ** 7",
          "    G = 0.5 * (1 - np.sqrt(Cb7 / (Cb7 + 6103515625.0)))",
          "    ap1, ap2 = (1 + G) * a1, (1 + G) * a2",
          "    Cp1, Cp2 = np.sqrt(ap1 * ap1 + b1 * b1), np.sqrt(ap2 * ap2 + b2 * b2)",
          "    h1, h2 = np.degrees(np.arctan2(b1, ap1)) % 360, np.degrees(np.arctan2(b2, ap2)) % 360",
          "    cc = Cp1 * Cp2",
          "    dh = h2 - h1",
          "    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))",
          "    dh = np.where(cc == 0, 0.0, dh)",
          "    dHp = 2 * np.sqrt(cc) * np.sin(np.radians(dh) / 2)",
          "    Lb, Cbp = (L1 + L2) / 2, (Cp1 + Cp2) / 2",
          "    hs = h1 + h2",
          "    hb = np.where(cc == 0, hs, np.where(np.abs(h1 - h2) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))",
          "    T = 1 - 0.17 * np.cos(np.radians(hb - 30)) + 0.24 * np.cos(np.radians(2 * hb)) + 0.32 * np.cos(np.radians(3 * hb + 6)) - 0.20 * np.cos(np.radians(4 * hb - 63))",
          "    dth = 30 * np.exp(-(((hb - 275) / 25) * ((hb - 275) / 25)))",
          "    Cbp7 = Cbp ** 7",
          "    RC = 2 * np.sqrt(Cbp7 / (Cbp7 + 6103515625.0))",
          "    SL = 1 + 0.015 * (Lb - 50) * (Lb - 50) / np.sqrt(20 + (Lb - 50) * (Lb - 50))",
          "    SC = 1 + 0.045 * Cbp",
          "    SH = 1 + 0.015 * Cbp * T",
          "    RT = -np.sin(np.radians(2 * dth)) * RC",
          "    tL, tC, tH = (L2 - L1) / SL, (Cp2 - Cp1) / SC, dHp / SH",
          "    return np.sqrt(tL * tL + tC * tC + tH * tH + RT * tC * tH)",
        ]
        : []),
      "a_bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
      "b_bgr = img_b if img_b.ndim == 3 else cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)",
      ...(meanB
        ? ["# เทียบกับสีเฉลี่ยของ B — ไม่ปรับขนาด B"]
        : [
          "if a_bgr.shape[:2] != b_bgr.shape[:2]:",
          "    b_bgr = cv2.resize(b_bgr, (a_bgr.shape[1], a_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)",
        ]),
      "lab_a = cv2.cvtColor(a_bgr.astype(np.float32) / 255, cv2.COLOR_BGR2Lab).astype(np.float64)  # L 0-100, a/b ประมาณ +-128",
      "lab_b = cv2.cvtColor(b_bgr.astype(np.float32) / 255, cv2.COLOR_BGR2Lab).astype(np.float64)",
      "n_a, n_b = lab_a.shape[0] * lab_a.shape[1], lab_b.shape[0] * lab_b.shape[1]",
      "mean_a = [np.cumsum(lab_a[:, :, c].ravel())[-1] / n_a for c in range(3)]  # บวกตามลำดับเหมือนแอป",
      "mean_b = [np.cumsum(lab_b[:, :, c].ravel())[-1] / n_b for c in range(3)]",
      ...(meanB ? ["ref = [np.float64(v) for v in mean_b]  # สีเฉลี่ยของ B"] : ["ref = [lab_b[:, :, 0], lab_b[:, :, 1], lab_b[:, :, 2]]"]),
      ...(ciede
        ? ["de = delta_e2000(lab_a[:, :, 0], lab_a[:, :, 1], lab_a[:, :, 2], ref[0], ref[1], ref[2])"]
        : [
          "dl, da, db = lab_a[:, :, 0] - ref[0], lab_a[:, :, 1] - ref[1], lab_a[:, :, 2] - ref[2]",
          "de = np.sqrt(dl * dl + da * da + db * db)",
        ]),
      `tol = ${String(tol)}`,
      "de_mean = np.cumsum(de.ravel())[-1] / n_a",
      "over_pct = np.count_nonzero(de > tol) * 100 / n_a",
      over
        ? "img = np.where(de > tol, 255, 0).astype(np.uint8)  # Mask จุดที่ต่างเกินเกณฑ์"
        : `img = np.minimum(255, np.floor(de * 255 / ${String(scaleMax)} + 0.5)).astype(np.uint8)  # ยิ่งขาวยิ่งต่าง (ขาว = ΔE ${scaleMax})`,
      `print(f"delta E (${ciede ? "CIEDE2000" : "CIE76"}): mean {np.floor(de_mean * 100 + 0.5) / 100:.2f} max {np.floor(de.max() * 100 + 0.5) / 100:.2f} over ${String(tol)}: {np.floor(over_pct * 10 + 0.5) / 10:.1f}%")`,
    ];
  } },

  // roi_apply: 2 ภาพ · out:"color" ตาม precedent imarith (ภาพเทาทั้งคู่ Python ได้ 3 ช่องค่าเท่ากัน) · กรอบ/วงรีด้วยเลขจำนวนเต็ม (ไม่ใช้ cv2.ellipse)
  roi_apply: { needs: "any", out: "color", gen: (p, ctx) => {
    const ellipse = p.shape === "ellipse";
    const rx = clampNum(p.x, 25, 0, 100), ry = clampNum(p.y, 25, 0, 100), rw = clampNum(p.w, 50, 1, 100), rh = clampNum(p.h, 50, 1, 100);
    const feather = Math.round(clampNum(p.feather, 0, 0, 50));
    const invert = boolOr(p.invert, false);
    const outline = p.view === "outline";
    return [
      "a_c = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
      "b_c = img_b if img_b.ndim == 3 else cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)",
      "if a_c.shape[:2] != b_c.shape[:2]:",
      "    b_c = cv2.resize(b_c, (a_c.shape[1], a_c.shape[0]), interpolation=cv2.INTER_LINEAR)",
      "h_i, w_i = a_c.shape[:2]",
      `x0 = min(w_i - 1, int(np.floor(w_i * ${String(rx)} / 100)))  # กรอบเป็นพิกเซล (จำนวนเต็ม) อย่างน้อย 1 px`,
      `y0 = min(h_i - 1, int(np.floor(h_i * ${String(ry)} / 100)))`,
      `x1 = min(w_i, x0 + max(1, int(np.floor(w_i * ${String(rw)} / 100 + 0.5))))`,
      `y1 = min(h_i, y0 + max(1, int(np.floor(h_i * ${String(rh)} / 100 + 0.5))))`,
      "jj = np.arange(h_i, dtype=np.int64)[:, None]",
      "ii = np.arange(w_i, dtype=np.int64)[None, :]",
      "inside = (ii >= x0) & (ii < x1) & (jj >= y0) & (jj < y1)",
      ...(ellipse
        ? [
          "ww, hh = x1 - x0, y1 - y0",
          "dx, dy = 2 * ii + 1 - (x0 + x1), 2 * jj + 1 - (y0 + y1)",
          "inside &= (dx * dx * hh * hh + dy * dy * ww * ww) <= ww * ww * hh * hh  # วงรีพอดีกรอบ (เลขจำนวนเต็มล้วน)",
        ]
        : []),
      `use_b = ${invert ? "~inside" : "inside"}  # ตำแหน่งที่ใช้ผลจาก B`,
      ...(feather === 0
        ? ["img = np.where(use_b[:, :, None], b_c, a_c)"]
        : [
          "mf = use_b.astype(np.float32)",
          `mb = cv2.GaussianBlur(mf, (0, 0), ${String(feather / 2)}, borderType=cv2.BORDER_REPLICATE)  # ขอบนุ่ม`,
          "w3 = mb[:, :, None].astype(np.float64)",
          "a64, b64 = a_c.astype(np.float64), b_c.astype(np.float64)",
          "img = np.floor(a64 + (b64 - a64) * w3 + 0.5).astype(np.uint8)",
        ]),
      ...(outline
        ? [
          "pad = np.pad(inside, 1, constant_values=False)  # นอกภาพนับเป็นนอกกรอบ",
          "edge = inside & ~(pad[1:-1, :-2] & pad[1:-1, 2:] & pad[:-2, 1:-1] & pad[2:, 1:-1])",
          `img[edge] = ${pyColor(ctx, [255, 255, 0])}  # เส้นกรอบสีเหลือง`,
        ]
        : []),
      `print(f"roi apply: {int(use_b.sum())} px use B ({np.floor(use_b.sum() * 100 / (w_i * h_i) * 10 + 0.5) / 10:.1f}%)")`,
    ];
  } },
  // ---- IP batch 4B (2026-10-15) — 8 op ใหม่ · ใช้อัลกอริทึมเดียวกับแอป (ไม่ใช่ตัวสำเร็จรูปของ cv2 ที่ให้ผลต่าง) · ค่าคงที่ความเร็วมาจาก registry (ไม่ hardcode ซ้ำ) ----
  // Wiener: psf_kernel (ฝาแฝด 4A) → psf2otf ด้วย np.roll → G·H*/(|H|²+NSR) ด้วย float64 → dft ย้อน · parity เป็น tolerance (dft ของ cv2 5.0 กับ 4.10 ต่างระดับ ulp แล้วถูกขยาย)
  deconv_wiener: { needs: "any", out: "same", gen: (p) => {
    const kind = pickOwn({ motion: "motion", disk: "disk" }, p.psf, "motion");
    const size = Math.round(clampNum(p.psfSize, 9, 1, 50)), ang = Math.round(clampNum(p.psfAngle, 0, 0, 179));
    const nsr = clampNum(p.nsr, 0.01, 0, 0.5);
    return [
      ...psfKernelPyLines(),
      `psf = psf_kernel("${kind}", ${size}, ${ang})`,
      "n = psf.shape[0]",
      "hc = (n - 1) // 2",
      "hh, ww = img.shape[:2]",
      "wp = cv2.getOptimalDFTSize(ww + 2 * n)",
      "hp = cv2.getOptimalDFTSize(hh + 2 * n)",
      "ph = np.zeros((hp, wp), np.float32)",
      "ph[:n, :n] = psf",
      "ph = np.roll(ph, (-hc, -hc), axis=(0, 1))  # psf2otf: จุดกลางของ PSF ไปอยู่ที่ (0, 0) แบบวนรอบ",
      "otf = cv2.dft(ph, flags=cv2.DFT_COMPLEX_OUTPUT)",
      `nsr = ${String(nsr)}  # noise / สัญญาณ — 0 = หารตรง (inverse filter) noise ระเบิด`,
      "def wiener_channel(ch):",
      "    big = cv2.copyMakeBorder(ch, n, hp - hh - n, n, wp - ww - n, cv2.BORDER_REFLECT)  # ขอบสะท้อนค่า (ไม่ใช้ edgetaper)",
      "    g = cv2.dft(big.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)",
      "    gr, gi = g[:, :, 0].astype(np.float64), g[:, :, 1].astype(np.float64)",
      "    hr, hi = otf[:, :, 0].astype(np.float64), otf[:, :, 1].astype(np.float64)",
      "    den = hr * hr + hi * hi + nsr",
      "    ok = den > 0",
      "    safe = np.where(ok, den, 1.0)",
      "    re = np.where(ok, (gr * hr + gi * hi) / safe, 0.0)  # F = G x conj(H) / (|H|^2 + NSR)",
      "    im = np.where(ok, (gi * hr - gr * hi) / safe, 0.0)",
      "    f = np.dstack([re, im]).astype(np.float32)",
      "    inv = cv2.dft(f, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)",
      "    v = np.floor(inv[n:n + hh, n:n + ww].astype(np.float64) + 0.5)",
      "    return np.where(np.isfinite(v), np.clip(v, 0, 255), 0).astype(np.uint8)",
      "if img.ndim == 2:",
      "    img = wiener_channel(img)",
      "else:",
      "    img = cv2.merge([wiener_channel(c) for c in cv2.split(img)])  # ทีละช่อง ใช้ OTF ชุดเดียว",
      `print("wiener deconvolution: psf ${kind === "disk" ? `disk ${size} px` : `motion ${size} px ${ang} deg`}, NSR ${String(nsr)}")`,
    ];
  } },

  // Lucy-Richardson: float32 ทุกขั้นเหมือนแอป · filter2D เป็น correlation → kf (PSF พลิก) เป็น convolution, kk (PSF ตรง) เป็น adjoint
  deconv_lucy: { needs: "any", out: "same", gen: (p) => {
    const kind = pickOwn({ motion: "motion", disk: "disk" }, p.psf, "motion");
    const size = Math.round(clampNum(p.psfSize, 9, 1, 50)), ang = Math.round(clampNum(p.psfAngle, 0, 0, 179));
    const iters = Math.round(clampNum(p.iterations, 10, 1, LUCY_MAX_ITERS));
    return [
      ...psfKernelPyLines(),
      `psf = psf_kernel("${kind}", ${size}, ${ang})`,
      "kk = psf.astype(np.float32)",
      "kf = np.ascontiguousarray(np.flip(psf, (0, 1)).astype(np.float32))",
      `iters = ${iters}`,
      // B6: งบงานเดียวกับแอป (LUCY_WORK_BUDGET) — ภาพใหญ่ลดรอบ ไม่ย่อภาพ
      "lucy_ch = 1 if img.ndim == 2 else 3",
      `iters_used = min(iters, max(1, ${LUCY_WORK_BUDGET} // (img.shape[0] * img.shape[1] * lucy_ch)))`,
      "if iters_used < iters:",
      "    print(f\"ภาพใหญ่ จึงลดเหลือ {iters_used} รอบ (ขอ {iters} รอบ)\")",
      "    iters = iters_used",
      "def lucy_channel(ch):",
      "    g = ch.astype(np.float32)",
      "    est = g.copy()",
      "    for _ in range(iters):",
      "        c = cv2.filter2D(est, -1, kf, borderType=cv2.BORDER_REFLECT_101)  # เบลอภาพที่เดา",
      "        ok = c > 1e-6",
      "        r = np.where(ok, g / np.where(ok, c, np.float32(1)), np.float32(0)).astype(np.float32)  # จริง / ที่เบลอ",
      "        corr = cv2.filter2D(r, -1, kk, borderType=cv2.BORDER_REFLECT_101)",
      "        est = (est * corr).astype(np.float32)",
      "    return np.clip(np.floor(est.astype(np.float64) + 0.5), 0, 255).astype(np.uint8)",
      "if img.ndim == 2:",
      "    img = lucy_channel(img)",
      "else:",
      "    img = cv2.merge([lucy_channel(c) for c in cv2.split(img)])",
      `print(f"lucy-richardson: {iters} iterations, psf ${kind === "disk" ? `disk ${size} px` : `motion ${size} px ${ang} deg`}")`,
    ];
  } },

  // ECC: A = img (ภาพที่ขยับ), B = img_b (อ้างอิง) · guard ภาพเล็ก/เรียบเหมือนแอป · ผล warp ต่างจากแอปได้ราว 0.05 px (cv2 5.0 กับ 4.10) · out:"color" เสมอ (aligned/overlay/diff เป็น 3 ช่อง)
  ecc_register: { needs: "any", out: "color", gen: (p, ctx) => {
    const motion = pickOwn({ translation: "translation", euclidean: "euclidean", affine: "affine", homography: "homography" }, p.motion, "euclidean");
    const iters = Math.round(clampNum(p.iterations, 50, 10, 300)), maxSide = Math.round(clampNum(p.maxSide, ECC_DEFAULT_SIDE, 64, 1024));
    const view = pickOwn({ aligned: "aligned", overlay: "overlay", diff: "diff" }, p.view, "aligned");
    const homog = motion === "homography";
    const gray = pyGrayCode(ctx);
    const fx2 = (e) => `{np.floor(${e} * 100 + 0.5) / 100:.2f}`;
    const fx3 = (e) => `{np.floor(${e} * 1000 + 0.5) / 1000:.3f}`;
    return [
      `a_g = img if img.ndim == 2 else cv2.cvtColor(img, ${gray})`,
      "b_src = img_b if img_b.shape[:2] == img.shape[:2] else cv2.resize(img_b, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)",
      `b_g = b_src if b_src.ndim == 2 else cv2.cvtColor(b_src, ${gray})`,
      "hh, ww = a_g.shape",
      `scale = min(1.0, ${maxSide} / max(ww, hh))  # ย่อภาพก่อนจับคู่`,
      "aw, ah = max(1, int(np.floor(ww * scale + 0.5))), max(1, int(np.floor(hh * scale + 0.5)))",
      "if scale < 1:",
      "    sa = cv2.resize(a_g, (aw, ah), interpolation=cv2.INTER_AREA)",
      "    sb = cv2.resize(b_g, (aw, ah), interpolation=cv2.INTER_AREA)",
      "else:",
      "    sa, sb = a_g, b_g",
      "ecc_ok = False",
      "msg = ''",
      "if min(sa.shape) < 16:",
      "    msg = 'image too small (< 16 px after shrinking)'",
      "elif sa.std() < 1 or sb.std() < 1:",
      "    msg = 'image too flat (no texture to match)'",
      "else:",
      `    warp = np.eye(${homog ? 3 : 2}, 3, dtype=np.float32)` + (homog ? "" : "  # 2x3"),
      "    try:",
      `        rho, warp = cv2.findTransformECC(sb, sa, warp, cv2.MOTION_${motion.toUpperCase()}, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, ${iters}, 1e-5), None, 5)`,
      "        ecc_ok = True",
      "    except cv2.error:",
      "        msg = 'did not converge'",
      "if ecc_ok:",
      "    warp[0, 2] = np.float32(float(warp[0, 2]) / scale)  # ขยายค่าเลื่อนกลับเป็นพิกเซลของภาพจริง",
      "    warp[1, 2] = np.float32(float(warp[1, 2]) / scale)",
      ...(homog ? ["    warp[2, 0] = np.float32(float(warp[2, 0]) * scale)", "    warp[2, 1] = np.float32(float(warp[2, 1]) * scale)"] : []),
      "    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP",
      `    warped = cv2.${homog ? "warpPerspective" : "warpAffine"}(img, warp, (ww, hh), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)`,
      "    w00, w01, w10, w11 = float(warp[0, 0]), float(warp[0, 1]), float(warp[1, 0]), float(warp[1, 1])",
      `    go = warped if warped.ndim == 2 else cv2.cvtColor(warped, ${gray})`,
      ...(view === "aligned"
        ? ["    img = warped"]
        : view === "diff"
          ? ["    img = cv2.absdiff(go, b_g)"]
          : ["    img = np.dstack([go, b_g, go])  # R = A, G = B, B = A (ม่วง = A, เขียว = B)"]),
      `    print(f"ecc ${motion}: dx ${fx2("float(warp[0, 2])")} dy ${fx2("float(warp[1, 2])")} angle ${fx2("np.arctan2(w10, w00) * 180 / np.pi")} scale ${fx3("np.sqrt(abs(w00 * w11 - w01 * w10))")} rho ${fx3("rho")}")`,
      "else:",
      `    print('ecc ${motion}: failed (' + msg + ')')`,
      "if img.ndim == 2:",
      "    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)",
    ];
  } },

  // Gabor: kernel จาก cv2.getGaborKernel (float32 เท่ากับสูตรของแอปทุกค่า) · ส่วนจริงลบค่าเฉลี่ย (บวกตามลำดับเหมือน JS) · ปรับให้ค่าสูงสุดเป็น 255
  gabor_filter: { needs: "gray", out: "gray", gen: (p) => {
    const lambd = Math.round(clampNum(p.wavelength, 8, 2, 64)), theta = Math.round(clampNum(p.orientation, 0, 0, 179));
    const bw = clampNum(p.bandwidth, 1, 0.5, 3), gamma = clampNum(p.aspect, 0.5, 0.2, 1);
    const real = p.output === "real";
    return [
      `lambd = ${lambd}`,
      `theta = ${theta} * np.pi / 180`,
      `bw = ${String(bw)}`,
      `gamma = ${String(gamma)}`,
      "sigma = lambd / np.pi * np.sqrt(np.log(2) / 2) * (2 ** bw + 1) / (2 ** bw - 1)  # MATLAB imgaborfilt: sigma จาก bandwidth (octave)",
      "half = min(49, int(np.ceil(3 * sigma / gamma)))",
      "k = 2 * half + 1",
      "ke = cv2.getGaborKernel((k, k), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)",
      "ke = (ke.astype(np.float64) - np.cumsum(ke.astype(np.float64).ravel())[-1] / ke.size).astype(np.float32)  # ลบค่าเฉลี่ย: พื้นเรียบได้ 0",
      "ko = cv2.getGaborKernel((k, k), sigma, theta, lambd, gamma, np.pi / 2, ktype=cv2.CV_32F)",
      "f32 = img.astype(np.float32)",
      "re = cv2.filter2D(f32, cv2.CV_32F, ke, borderType=cv2.BORDER_REFLECT_101)",
      "im = cv2.filter2D(f32, cv2.CV_32F, ko, borderType=cv2.BORDER_REFLECT_101)",
      real ? "resp = re" : "resp = cv2.magnitude(re, im)",
      "resp64 = resp.astype(np.float64)",
      "mx = float(np.abs(resp).max())",
      "mean_resp = np.cumsum(np.abs(resp64).ravel())[-1] / resp64.size",
      "none = (not (mx > 0)) or int(img.min()) == int(img.max())  # ภาพเรียบสนิท: เศษ float32 ของ kernel ต้องไม่ถูกขยายเป็นภาพขาว",
      "if none:",
      "    mx, mean_resp = 0.0, 0.0",
      real
        ? "img = (np.full(resp.shape, 128.0) if none else np.floor(128 + resp64 * 127 / mx + 0.5)).astype(np.uint8)"
        : "img = (np.zeros(resp.shape) if none else np.floor(resp64 * 255 / mx + 0.5)).astype(np.uint8)",
      `print(f"gabor: lambda ${lambd} theta ${theta} sigma {np.floor(sigma * 100 + 0.5) / 100:.2f} kernel {k}x{k} mean {np.floor(mean_resp * 10 + 0.5) / 10:.1f} max {np.floor(mx * 10 + 0.5) / 10:.1f}")`,
    ];
  } },

  // GLCM: คู่พิกเซลด้วย slicing + bincount (นับจำนวนเต็มเท่าแอปเป๊ะ) · props ด้วย numpy (ลำดับการบวกต่างจากแอป → ค่าต่างได้ระดับ 1e-12)
  glcm: { needs: "gray", out: "gray", gen: (p) => {
    const L = Number(pickOwn({ 2: 2, 4: 4, 8: 8, 16: 16, 32: 32, 64: 64 }, p.levels, 8));
    const d = Math.round(clampNum(p.distance, 1, 1, 10));
    const angle = Object.hasOwn({ 0: 1, 45: 1, 90: 1, 135: 1, avg: 1 }, p.angle) ? String(p.angle) : "0";
    const sym = boolOr(p.symmetric, false);
    const matrix = p.view !== "quantized";
    const OFF = { 0: [d, 0], 45: [d, -d], 90: [0, -d], 135: [-d, -d] };
    const offs = (angle === "avg" ? [0, 45, 90, 135] : [Number(angle)]).map((a) => `(${OFF[a][0]}, ${OFF[a][1]})`).join(", ");
    return [
      `levels = ${L}`,
      "q = np.minimum(levels - 1, (img.astype(np.int64) * levels) // 255)  # ลดระดับเทา",
      "def glcm_counts(dx, dy):",
      "    h, w = q.shape",
      "    if h - abs(dy) <= 0 or w - abs(dx) <= 0:",
      "        return np.zeros((levels, levels))",
      "    a = q[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]",
      "    b = q[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]",
      "    return np.bincount((a * levels + b).ravel(), minlength=levels * levels).reshape(levels, levels).astype(np.float64)",
      "def glcm_props(c):",
      "    n = c.sum()",
      "    if n == 0:",
      "        return None",
      "    pr = c / n",
      "    i, j = np.indices(c.shape)",
      "    contrast = ((i - j) ** 2 * pr).sum()",
      "    energy = (pr * pr).sum()",
      "    homog = (pr / (1 + np.abs(i - j))).sum()",
      "    mi, mj = (i * pr).sum(), (j * pr).sum()",
      "    if (c.sum(axis=1) > 0).sum() <= 1 or (c.sum(axis=0) > 0).sum() <= 1:",
      "        corr = None  # ทุกคู่อยู่ในระดับเดียว (sigma = 0)",
      "    else:",
      "        si = np.sqrt(((i - mi) ** 2 * pr).sum())",
      "        sj = np.sqrt(((j - mj) ** 2 * pr).sum())",
      "        corr = ((i - mi) * (j - mj) * pr).sum() / (si * sj)",
      "    return [contrast, corr, energy, homog, n]",
      `offsets = [${offs}]`,
      "mats = []",
      "props = []",
      "for dx, dy in offsets:",
      "    c = glcm_counts(dx, dy)",
      ...(sym ? ["    c = c + c.T  # นับทั้งสองทาง"] : []),
      "    mats.append(c)",
      "    props.append(glcm_props(c))",
      "pairs = sum(float(m.sum()) for m in mats)",
      "def avg_prop(k):",
      "    vals = [pr[k] if pr is not None else None for pr in props]",
      "    return None if any(v is None for v in vals) else sum(vals) / len(vals)",
      "contrast, corr, energy, homog = avg_prop(0), avg_prop(1), avg_prop(2), avg_prop(3)",
      "total = sum(mats)  # ผลรวม GLCM ของทุกทิศ",
      ...(matrix
        ? [
          "cmax = total.max()",
          "m8 = np.where(total == 0, 0, np.floor(255 * np.log1p(total) / np.log1p(max(cmax, 1)) + 0.5))",
          "img = np.repeat(np.repeat(m8, 256 // levels, axis=0), 256 // levels, axis=1).astype(np.uint8)  # ตาราง GLCM ขยายเป็น 256x256",
        ]
        : ["img = np.floor(q * 255 / (levels - 1) + 0.5).astype(np.uint8)"]),
      "fx = lambda v, d: '-' if v is None else f'{np.floor(v * 10 ** d + 0.5) / 10 ** d:.{d}f}'",
      "if pairs == 0:",
      "    print('glcm: no pixel pairs (image smaller than the distance)')",
      "else:",
      `    print(f"glcm levels {levels} dist ${d} angle ${angle}: contrast {fx(contrast, 2)} energy {fx(energy, 3)} homogeneity {fx(homog, 3)} correlation {fx(corr, 3)} pairs {int(pairs)}")`,
    ];
  } },

  // NLM: ลูป offset ลำดับเดียวกับ nlmCore (dy นอก dx ใน) · S จำนวนเต็มแม่นทั้งสองฝั่ง · cv2.fastNlMeansDenoising ให้ผลต่างจากแอป จึงไม่ใช้
  nlm_denoise: { needs: "any", out: "same", gen: (p) => {
    const h = Math.round(clampNum(p.h, 10, 1, 50));
    const patch = Number(pickOwn({ 3: 3, 5: 5, 7: 7 }, p.patch, 5)), search = Number(pickOwn({ 7: 7, 11: 11, 15: 15 }, p.search, 11));
    return [
      `h = ${h}`,
      `pr = ${(patch - 1) / 2}  # แผ่น ${patch}x${patch}`,
      `sr = ${(search - 1) / 2}  # หน้าต่างค้นหา ${search}x${search}`,
      "hh0, ww0 = img.shape[:2]",
      `scale = min(1.0, ${NLM_MAX_SIDE} / max(ww0, hh0))  # ภาพใหญ่ย่อก่อน (ช้ามาก)`,
      "if scale < 1:",
      "    work = cv2.resize(img, (max(1, int(np.floor(ww0 * scale + 0.5))), max(1, int(np.floor(hh0 * scale + 0.5)))), interpolation=cv2.INTER_AREA)",
      "else:",
      "    work = img",
      "x = work.astype(np.int64) if work.ndim == 3 else work.astype(np.int64)[:, :, None]",
      "hh, ww, nc = x.shape",
      "acc = np.zeros((hh, ww, nc))",
      "wsum = np.zeros((hh, ww))",
      "yy0, xx0 = np.arange(hh), np.arange(ww)",
      "y_lo, y_hi = np.maximum(0, yy0 - pr), np.minimum(hh - 1, yy0 + pr)",
      "x_lo, x_hi = np.maximum(0, xx0 - pr), np.minimum(ww - 1, xx0 + pr)",
      "area = ((y_hi - y_lo + 1)[:, None] * (x_hi - x_lo + 1)[None, :] * nc).astype(np.float64)",
      "for dy in range(-sr, sr + 1):",
      "    for dx in range(-sr, sr + 1):",
      "        nb = x[np.clip(yy0 + dy, 0, hh - 1)][:, np.clip(xx0 + dx, 0, ww - 1)]  # เพื่อนบ้านที่ติดขอบ = ค่าขอบ (replicate)",
      "        d2 = ((x - nb) ** 2).sum(axis=2).astype(np.float64)",
      "        integ = np.zeros((hh + 1, ww + 1))",
      "        integ[1:, 1:] = np.cumsum(np.cumsum(d2, axis=1), axis=0)",
      "        s = integ[y_hi + 1][:, x_hi + 1] - integ[y_lo][:, x_hi + 1] - integ[y_hi + 1][:, x_lo] + integ[y_lo][:, x_lo]",
      "        w = np.exp(-(s / area) / (h * h))",
      "        wsum += w",
      "        acc += w[:, :, None] * nb",
      "res = np.clip(np.floor(acc / wsum[:, :, None] + 0.5), 0, 255).astype(np.uint8)",
      "res = res if nc > 1 else res[:, :, 0]",
      "if scale < 1:",
      "    res = cv2.resize(res, (ww0, hh0), interpolation=cv2.INTER_LINEAR)",
      "img = res",
      `print("non-local means: h ${h}, patch ${patch}, search ${search}")`,
    ];
  } },

  // Dehaze: dark channel prior · float32/float64 ตามจุดเดียวกับแอป (refine = false → ผลเท่าแอปทุก byte) · guided_f32 ฝาแฝดจาก 4A
  dehaze: { needs: "any", out: "same", outOf: (p) => (p.view === "transmission" || p.view === "dark" ? "gray" : "same"), gen: (p, ctx) => {
    const omega = clampNum(p.amount, 0.95, 0.1, 1), ps = odd(clampNum(p.patch, 15, 3, 31));
    const refine = boolOr(p.refine, true);
    const view = pickOwn({ result: "result", transmission: "transmission", dark: "dark" }, p.view, "result");
    const mir = isMirrored(ctx);
    return [
      ...(refine ? guidedPyLines() : []),
      `omega = ${String(omega)}`,
      "x = img if img.ndim == 3 else img[:, :, None]",
      "hh, ww, nc = x.shape",
      "n_px = hh * ww",
      `ker = cv2.getStructuringElement(cv2.MORPH_RECT, (${ps}, ${ps}))`,
      "dark = cv2.erode(np.ascontiguousarray(x.min(axis=2)), ker)  # dark channel",
      "hist = np.bincount(dark.ravel(), minlength=256)",
      "top = max(1, n_px // 1000)  # จุดที่ dark สูงสุด 0.1% (อย่างน้อย 1 จุด)",
      "t_dark = 255 - int(np.argmax(np.cumsum(hist[::-1]) >= top))",
      "sel = dark >= t_dark",
      `atm = np.maximum(x[sel].sum(axis=0, dtype=np.int64) / int(sel.sum()), 1)  # แสงหมอก A ตามลำดับช่องของภาพ${mir ? " (B, G, R)" : ""}`,
      "mn = (x.astype(np.float64) / atm).min(axis=2).astype(np.float32)",
      "dn = cv2.erode(mn, ker)",
      "t = (1 - omega * dn.astype(np.float64)).astype(np.float32)  # แผนที่ความโปร่งใสหยาบ",
      ...(refine
        ? [
          `gray = cv2.cvtColor(img, ${pyGrayCode(ctx)}) if img.ndim == 3 else img`,
          "pp = (t.astype(np.float64) * 255).astype(np.float32)",
          `q = guided_f32(gray.astype(np.float32), pp, ${Math.max(4, 2 * ps)}, np.float32(65.025))  # เกลี่ยตามขอบวัตถุ`,
          "t = (q.astype(np.float64) / 255).astype(np.float32)",
        ]
        : []),
      "t = np.maximum(t.astype(np.float64), 0.1)",
      "t_mean = np.cumsum(t.ravel())[-1] / n_px",
      "res = np.floor((x.astype(np.float64) - atm) / t[:, :, None] + atm + 0.5)",
      "res = np.clip(res, 0, 255).astype(np.uint8)",
      ...(view === "result" ? ["img = res if nc > 1 else res[:, :, 0]"] : view === "transmission" ? ["img = np.floor(t * 255 + 0.5).astype(np.uint8)"] : ["img = dark"]),
      `a3 = [float(atm[0])] * 3 if nc == 1 else [float(v) for v in ${mir ? "atm[::-1]" : "atm"}]  # R, G, B`,
      `print(f"dehaze: A (R/G/B) ${pyFixed1("a3[0]")}/${pyFixed1("a3[1]")}/${pyFixed1("a3[2]")} t_mean {np.floor(t_mean * 100 + 0.5) / 100:.2f}")`,
    ];
  } },

  // DCT 8x8: ตาราง C ด้วย numpy · ทุกบล็อกพร้อมกันด้วย einsum · ปัดแบบ sign x floor(|x| + 0.5) · cv2.dct(block) ให้ค่าเดียวกัน (float32) ใช้ตรวจได้
  dct_jpeg: { needs: "gray", out: "gray", gen: (p) => {
    const zonal = p.method === "zonal";
    const quality = Math.round(clampNum(p.quality, 50, 1, 100)), K = Math.round(clampNum(p.keep, 4, 1, 8));
    const view = pickOwn({ result: "result", coeffs: "coeffs", error: "error" }, p.view, "result");
    return [
      "hh, ww = img.shape",
      "hp, wp = (hh + 7) // 8 * 8, (ww + 7) // 8 * 8",
      "pad = cv2.copyMakeBorder(img, 0, hp - hh, 0, wp - ww, cv2.BORDER_REPLICATE)",
      "c = np.zeros((8, 8))",
      "for u in range(8):",
      "    for xx in range(8):",
      "        c[u, xx] = (np.sqrt(1 / 8) if u == 0 else np.sqrt(2 / 8)) * np.cos((2 * xx + 1) * u * np.pi / 16)",
      "blocks = (pad.astype(np.float64) - 128).reshape(hp // 8, 8, wp // 8, 8).transpose(0, 2, 1, 3)",
      "f = np.einsum('ui,abij,vj->abuv', c, blocks, c)  # F = C B C^T ทุกบล็อก",
      ...(zonal
        ? [
          `keep = ${K}`,
          "zone = (np.arange(8)[:, None] < keep) & (np.arange(8)[None, :] < keep)",
          "fq = np.where(zone, f, 0.0)",
          "n_zero = (64 - keep * keep) * (hp // 8) * (wp // 8)",
        ]
        : [
          "q50 = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,",
          "                18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]).reshape(8, 8)",
          `quality = ${quality}`,
          "s = 5000 // quality if quality < 50 else 200 - 2 * quality  # สเกลแบบ IJG",
          "qs = np.maximum(1, (q50 * s + 50) // 100)",
          "z = f / qs",
          "qf = np.sign(z) * np.floor(np.abs(z) + 0.5)  # ปัดครึ่งออกจาก 0 เหมือนแอป",
          "fq = qf * qs",
          "n_zero = int(np.count_nonzero(qf == 0))",
        ]),
      "rec = np.einsum('ui,abuv,vj->abij', c, fq, c) + 128",
      "rec = np.clip(np.floor(rec + 0.5), 0, 255).transpose(0, 2, 1, 3).reshape(hp, wp).astype(np.uint8)[:hh, :ww]",
      "diff = img.astype(np.float64) - rec",
      "mse = (diff ** 2).sum() / (hh * ww)",
      "psnr = float('inf') if mse == 0 else 10 * np.log10(255 * 255 / mse)",
      ...(view === "result"
        ? ["img = rec"]
        : view === "coeffs"
          ? ["img = np.minimum(255, np.floor(255 * np.log1p(np.abs(fq)) / np.log1p(1024) + 0.5)).transpose(0, 2, 1, 3).reshape(hp, wp)[:hh, :ww].astype(np.uint8)"]
          : ["img = np.minimum(255, np.abs(diff) * 8).astype(np.uint8)"]),
      `print(f"dct ${zonal ? `zonal ${K}x${K}` : `jpeg q${quality}`}: zero {np.floor(n_zero * 100 / ((hp // 8) * (wp // 8) * 64) * 10 + 0.5) / 10:.1f}% psnr " + ("inf" if psnr == float("inf") else f"{np.floor(psnr * 100 + 0.5) / 100:.2f}"))`,
    ];
  } },
};

// Tier 2, 18d — ผู้ใช้เลือก "ใช้ในแอปอย่างเดียว" (D-8 ข้อ 5): op ที่ต้องใช้หลายเฟรมต่อเนื่อง export เป็น
// process(path) ภาพเดี่ยวไม่ได้ — ห้ามเพิ่ม GEN ของ 3 ตัวนี้ (ถ้าเพิ่ม ต้องออกแบบ process_video ใหม่ทั้งหมด)
const APP_ONLY_OPS = {
  mog2: "ต้องเรียนรู้พื้นหลังจากหลายเฟรมต่อเนื่อง (กล้อง/วิดีโอ)",
  farneback_optical_flow: "ต้องเทียบเฟรมปัจจุบันกับเฟรมก่อนหน้า",
  pyrlk_optical_flow: "ต้องติดตามจุดต่อเนื่องข้ามหลายเฟรม",
};

// บล็อกคอมเมนต์ไทย + คำเตือนตอนรัน — ขั้นนี้เป็น identity (ภาพผ่านไปเฉย ๆ ขั้นถัดไปได้ภาพเดิม, ctx.gray ไม่เปลี่ยน)
// บรรทัด print ต้องเป็น **ASCII ล้วน** (V15): บน Windows ภาษาไทย stdout ที่ถูก redirect เป็น cp874 — ไทย/⚠/— ใน
// print() จะ crash ด้วย UnicodeEncodeError (คอมเมนต์ไม่ crash เพราะ Python อ่านไฟล์เป็น utf-8 ตาม coding header)
function appOnlyLines(opId, label) {
  return [
    `# ── ${label}: ${thaiName(opId)} (${opId}) — ใช้ได้ในแอปเท่านั้น ──`,
    `# ${APP_ONLY_OPS[opId]} แต่สคริปต์นี้ประมวลผลภาพนิ่งทีละภาพ จึงยังไม่รองรับการ export เป็น Python`,
    "# ขั้นนี้ถูกข้าม: ขั้นถัดไปจะได้ภาพเดียวกับที่เข้าขั้นนี้ ผลลัพธ์จึงไม่ตรงกับที่เห็นในแอป",
    `print("WARNING: skipped '${opId}' (app-only: needs consecutive video frames) - result differs from the app")`,
  ];
}

// บรรทัดหมายเหตุใต้หัวไฟล์ (ขึ้นเมื่อมี op app-only อย่างน้อย 1 ตัวเท่านั้น — ไม่มี = ไม่มีบรรทัดนี้เลย)
function appOnlyHeader(appOnly) {
  if (!appOnly.length) return "";
  return `\n# หมายเหตุ: มีขั้นตอนที่ใช้ได้ในแอปเท่านั้น (${appOnly.join(", ")}) ซึ่งถูกข้ามในสคริปต์นี้ — ดูคอมเมนต์ที่ขั้นนั้น`;
}

// P-8 pass_fail — ops ที่ใส่ ctx.values ใน registry (ตัวที่ "ล่าสุด" บนสายหลักคือแหล่งค่าของ pass_fail · เทสต์ static เทียบกับ registry.js)
// PF_PY_SOURCES = ตัวที่ gen โค้ด snapshot ให้ได้ (ที่เหลือ: metric ตรงกับ op นั้นแต่ Python ยังไม่รองรับ = คำเตือน)
// ข้อจำกัดที่รู้ (ยอมรับ): template_matching (ไม่มี template / ใหญ่เกินภาพ) และ psnr/ssim (ไม่ได้ต่อ B) แอปคืนภาพโดยไม่มี values แต่ที่นี่ยังนับเป็น
// "กล่องวัดล่าสุด" เสมอ — ผลคือ Python ตัดสิน NG (กล่องผิดชนิด) ในกรณีที่แอปจะใช้ค่าของกล่องวัดก่อนหน้า · เอนเอียงไปทาง NG (fail-safe)
export const PF_VALUE_OPS = new Set(["regionprops", "template_matching", "psnr", "ssim", "mog2", "farneback_optical_flow", "pyrlk_optical_flow", "feret",
  "line_profile", "shape_filter", "roi_polygon", "slic", "focus_score", "delta_e", "roi_apply", "ecc_register", "gabor_filter", "glcm", "dehaze", "dct_jpeg"]);
const PF_PY_SOURCES = { regionprops: true, roi_polygon: true, focus_score: true };
// บรรทัดท้ายขั้นวัด: เก็บค่าที่ pass_fail ต้องใช้ลง `pf_up` ทันที (ตัวแปรของขั้นวัดอาจถูกทับโดยขั้นถัดไป) · เรียกเมื่อ pipeline มี pass_fail เท่านั้น
function pfSnapshotLines(opId, local, ctx) {
  if (!PF_VALUE_OPS.has(opId)) return [];
  ctx.pfSource = opId;
  if (opId === "regionprops") {
    // พื้นที่ = area_mm2 เมื่อ มม./px > 0 (สูตรเดียวกับ mmValue: area_px × (f×f)) ไม่งั้น area_px · แถวที่ถูกจำกัดด้วย maxObjects เท่านั้น
    const area = clampNum(local.mmPerPx, 0, 0, 100) > 0 ? "r['area_px'] * (mm_per_px * mm_per_px)" : "r['area_px']";
    return [`pf_up = {'count': total_found, 'areas': [${area} for r in regions]}  # ค่าให้ขั้น Pass / Fail`];
  }
  if (opId === "roi_polygon") return ["pf_up = {'value': roi_mean}  # ค่าให้ขั้น Pass / Fail"];
  if (opId === "focus_score") return ["pf_up = {'value': score}  # ค่าให้ขั้น Pass / Fail"];
  return [];
}
const hasPassFail = (pipeline) => pipeline.some((st) => resolveOp(st).opId === "pass_fail");

// step ที่ปิดไว้ (bypass: enabled === false) แอปข้ามไปเลย (cv-worker.js runPipeline `step.enabled === false → continue`, ทั้งสายหลักและสาย B) —
// Python ที่ export ต้องข้ามเหมือนกัน ไม่งั้นได้ผลต่างจากที่เห็นในแอป
function activeSteps(pipeline) {
  return (pipeline || []).filter((st) => st && st.enabled !== false);
}

// สร้างโค้ด Python เต็มจาก pipeline
// Tier 2 §C.0/§C.1 — pipeline ที่มี step ใดใช้ FILTERS group ที่ inputs===2 (เช่น image_arithmetic)
// ต้องมีโค้ดสองภาพนำเข้า (process(path_a, path_b)) — แยกไปทำที่ buildPythonCode2Image เพราะรูปแบบ
// ฟังก์ชันต่างจาก process(path) เดี่ยวโดยพื้นฐาน (resolveOp ทำงานกับ step.id ตรง ๆ ซึ่งเป็น FILTERS key
// เดียวกับที่ FILTERS[step.id] ใช้เช็ค — ยืนยันแล้วจาก tracePipeline ของ editor.js)
export function buildPythonCode(pipeline, opts = {}) {
  pipeline = activeSteps(pipeline);
  const has2Image = (pipeline || []).some((step) => step && FILTERS[step.id] && FILTERS[step.id].inputs === 2);
  if (has2Image) return buildPythonCode2Image(pipeline, opts);

  const name = opts.displayName || "ผลลัพธ์";
  const body = [];
  const ctx = { gray: false, mirrored: true }; // เริ่มจากภาพสี (cv2.imread = BGR = ช่องของแอปเรียงกลับด้าน)

  if (!pipeline || pipeline.length === 0) {
    body.push("# กล่องนี้แสดงภาพต้นฉบับ — ยังไม่ผ่าน filter ใด");
  }

  let stepNo = 0;
  const appOnly = [];
  const hasPf = hasPassFail(pipeline);
  for (const step of pipeline) {
    const { opId, local } = resolveOp(step);
    // 18d: op ที่ใช้ได้ในแอปเท่านั้น — ต้องเช็คก่อน GEN[opId] (ทั้ง 3 ตัวไม่มี GEN โดยเจตนา ไม่งั้นจะตกไปที่
    // บรรทัด "(ข้าม: ยังไม่รองรับ…)" ทั่วไปซึ่งไม่ได้บอกเหตุผลและไม่เตือนตอนรัน)
    if (APP_ONLY_OPS[opId]) {
      stepNo++;
      body.push(...appOnlyLines(opId, `ขั้นที่ ${stepNo}`), "");
      appOnly.push(opId);
      if (hasPf && PF_VALUE_OPS.has(opId)) ctx.pfSource = opId; // แอปรันกล่องนี้จริงและมี values → เป็นแหล่งค่าล่าสุดของ pass_fail
      continue;
    }
    const g = GEN[opId];
    if (!g) {
      body.push(`# (ข้าม: ยังไม่รองรับการ gen โค้ดของ '${opId}')`);
      if (hasPf && PF_VALUE_OPS.has(opId)) ctx.pfSource = opId;
      continue;
    }
    stepNo++;
    // แปลงเป็นขาวดำอัตโนมัติถ้า op ต้องการ (เหมือนที่แอปทำให้เบื้องหลัง)
    if (g.needs === "gray" && !ctx.gray) {
      body.push("img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (แปลงขาวดำอัตโนมัติ)");
      ctx.gray = true;
      ctx.mirrored = true;
    }
    body.push(`# ── ขั้นที่ ${stepNo}: ${thaiName(opId)} (${opId}) ──`);
    if (TEACH[opId]) body.push(`# ${TEACH[opId]}`);
    for (const line of g.gen(local, ctx)) body.push(line);
    if (hasPf) body.push(...pfSnapshotLines(opId, local, ctx));
    // อัปเดตสถานะภาพ (ขาวดำ/สี)
    const out = g.outOf ? g.outOf(local) : g.out;
    if (out === "gray") ctx.gray = true;
    else if (out === "color") ctx.gray = false;
    // อัปเดตลำดับช่องสี (ดูคอมเมนต์ ctx.mirrored เหนือ GEN): Color Space → เรียงเหมือนแอป · ได้ภาพขาวดำ → BGR อีกครั้ง
    if (g.appChannelOrder) ctx.mirrored = false;
    else if (ctx.gray || (g.grayOutOf && g.grayOutOf(local))) ctx.mirrored = true;
    body.push("");
  }

  const indented = body.map((l) => (l === "" ? "" : "    " + l)).join("\n");

  return `# -*- coding: utf-8 -*-
# โค้ดนี้สร้างอัตโนมัติจากลำดับ Box ในกล่องผลลัพธ์: "${name}"
# ทำ image processing ตามขั้นตอนเดียวกับที่เห็นในโปรแกรม${appOnlyHeader(appOnly)}
import cv2
import numpy as np
import sys

# บังคับให้ข้อความที่ print() ออกมาเป็น UTF-8 เสมอ — บนวินโดวส์ เวลาส่งผลไปเก็บในไฟล์
# (เช่น python code.py > log.txt) ระบบจะใช้รหัสอักขระเดิมของเครื่อง ซึ่งเขียนภาษาไทยหรือ
# สัญลักษณ์อย่าง px² / ∞ ไม่ได้ แล้วโปรแกรมจะหยุดกลางคันด้วย UnicodeEncodeError
# หมายเหตุ: ถ้านำไฟล์นี้ไป import ใช้ในโปรแกรมอื่น สองบรรทัดล่างนี้จะทำงานตั้งแต่ตอน import และเปลี่ยน
# sys.stdout ของโปรแกรมนั้นทั้งโปรเซสเป็น utf-8 ด้วย (ปกติเป็นผลดี) — ถ้าไม่ต้องการ ให้ย้ายไปไว้ใต้ if __name__
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def process(path):
    img = cv2.imread(path)  # อ่านภาพ (BGR)
    if img is None:
        raise FileNotFoundError(path)

${indented}
    return img


if __name__ == "__main__":
    result = process("input.jpg")   # << ใส่ path ภาพของคุณ
    cv2.imwrite("output.png", result)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
`;
}

// Tier 2 §C.0/§C.1 — เวอร์ชัน 2 ภาพนำเข้า: เรนเดอร์ src2 sub-pipeline (ภาพ B) ลง bodyB ก่อน (ใช้ GEN
// ตัวเดิมทุกตัว แต่แทนที่ "img" ด้วย "img_b" ทุกบรรทัดด้วย word-boundary regex) แล้วเรนเดอร์ pipeline
// หลักตามปกติลง bodyA — จุดที่ step มี src2 จะได้ img_b ที่ประมวลผลแล้วให้ GEN ของมันใช้ (เช่น
// GEN.imabsdiff อ่าน img_b ตรง ๆ)
//
// ตรวจสอบแล้ว (ดู worklog) ว่า trick แทนที่ \bimg\b ปลอดภัยกับทุก GEN ที่มีอยู่จริงในไฟล์นี้ ณ ตอนนี้
// รวมถึง local_std_filter/local_entropy_filter (ตัวแปร local หลายตัวที่ซับซ้อนสุด) และ
// clear_border/remove_small/skeleton (มีตัวแปร "bin_img" ซึ่ง \bimg\b ไม่จับเพราะติดกับ "_" ด้านหน้า
// เป็น word character — ไม่มีการแทนที่ผิดจุดเกิดขึ้นจริงในทุกกรณีที่ทดสอบ) — ดูเทสต์
// "tier2-11" ใน tests/tier2-infra.test.mjs ที่ยืนยันเชิงประจักษ์เรื่องนี้โดยเฉพาะ
// B8 (backlog 2026-10-17, P-21): ภาพ B ของแต่ละ step เริ่มจากสำเนาของภาพ B ดิบ (`img_b_raw`) เสมอ แล้วผ่านสาย B "ของ step นั้นเอง" —
// เดิม B ของทุก step ถูกต่อกันบนตัวแปร img_b ตัวเดียว (step ที่ 2 ได้ B ที่ผ่านสายของ step ที่ 1 มาแล้ว) และ op สองภาพที่อยู่ในสาย B
// ใช้ img_b ตัวเดียวกับสายนั้นเป็นภาพที่สอง (ไม่ได้ใช้ src2 ของตัวเอง) · ตอนนี้ op สองภาพซ้อนกันได้ลึกเท่าที่กราฟมี: ภาพ B ของระดับ d
// ชื่อ img_b (d=1) / img_b2 / img_b3 … (ดู bImageVar) · step สองภาพที่ไม่มี src2 เลย = แอปคืนภาพ A เดิม (passthroughNoSrc2) Python จึงข้ามขั้นนั้น
const bImageVar = (depth) => (depth <= 1 ? "img_b" : `img_b${depth}`);
// สร้างบรรทัดของ "สาย B ของ step" ลง body — คืนชื่อตัวแปรที่เก็บภาพ B ที่ประมวลผลเสร็จแล้ว · depth = ระดับของ step (0 = สายหลัก)
function emitBChain(step, depth, body, appOnly) {
  const w = bImageVar(depth + 1);
  body.push(`${w} = img_b_raw.copy()  # ภาพ B ของขั้นนี้ เริ่มจากภาพ B ดิบ`);
  const ctxB = { gray: false };
  for (const s2 of activeSteps(step.src2.pipeline)) {
    const r2 = resolveOp(s2);
    // 18d: บรรทัดคำเตือนของ op app-only ต้อง "ไม่" ผ่าน replace (ไม่มี token img อยู่แล้ว แต่ push ตรง ๆ ชัดเจนกว่า)
    if (APP_ONLY_OPS[r2.opId]) {
      body.push(...appOnlyLines(r2.opId, "สาย B"), "");
      appOnly.push(r2.opId);
      continue;
    }
    if (r2.opId === "pass_fail") { // ค่าไม่ข้ามสาย: สาย B ไม่มี snapshot ของตัวเอง — ข้ามพร้อมคำเตือน
      body.push("# (Pass / Fail ในสาย B ยังไม่รองรับการ export เป็น Python — ข้ามขั้นนี้)", `print("WARNING: skipped 'pass_fail' in line B (not supported in the Python export)")`);
      continue;
    }
    const g2 = GEN[r2.opId];
    if (!g2) { body.push(`# (ข้าม: ยังไม่รองรับการ gen โค้ดของ '${r2.opId}')`); continue; }
    let second = bImageVar(depth + 2); // ภาพที่สองของ op สองภาพในสาย B (เฉพาะเมื่อ s2 เป็น op สองภาพ)
    if (FILTERS[s2.id] && FILTERS[s2.id].inputs === 2) {
      if (!s2.src2 || !s2.src2.pipeline) { // ไม่มีสาย B เลย: แอปคืนภาพเดิม
        body.push(`# (${thaiName(r2.opId)}: ยังไม่ได้ต่อสาย B — แอปคืนภาพเดิม จึงข้ามขั้นนี้)`);
        continue;
      }
      emitBChain(s2, depth + 1, body, appOnly);
    }
    if (g2.needs === "gray" && !ctxB.gray) {
      body.push(`${w} = cv2.cvtColor(${w}, cv2.COLOR_BGR2GRAY)`);
      ctxB.gray = true;
    }
    for (const line of g2.gen(r2.local, ctxB)) body.push(line.replace(/\bimg_b\b|\bimg\b/g, (t) => (t === "img" ? w : second)));
    const out2 = g2.outOf ? g2.outOf(r2.local) : g2.out;
    if (out2 === "gray") ctxB.gray = true;
    else if (out2 === "color") ctxB.gray = false;
  }
  return w;
}
function buildPythonCode2Image(pipeline, opts = {}) {
  const name = opts.displayName || "ผลลัพธ์";
  const bodyA = [];
  const ctxA = { gray: false };
  let stepNo = 0;
  const appOnly = []; // 18d — op app-only จากทั้งสายหลักและสาย B (ใช้ทำหมายเหตุใต้หัวไฟล์)
  const hasPf = hasPassFail(activeSteps(pipeline));
  for (const step of activeSteps(pipeline)) {
    const { opId, local } = resolveOp(step);
    if (APP_ONLY_OPS[opId]) { // 18d: เช็คก่อน GEN[opId] เหมือนสายเดี่ยว
      stepNo++;
      bodyA.push(...appOnlyLines(opId, `ขั้นที่ ${stepNo}`), "");
      appOnly.push(opId);
      if (hasPf && PF_VALUE_OPS.has(opId)) ctxA.pfSource = opId;
      continue;
    }
    const g = GEN[opId];
    if (!g) {
      bodyA.push(`# (ข้าม: ยังไม่รองรับการ gen โค้ดของ '${opId}')`);
      if (hasPf && PF_VALUE_OPS.has(opId)) ctxA.pfSource = opId;
      continue;
    }
    stepNo++;
    if (FILTERS[step.id] && FILTERS[step.id].inputs === 2) {
      if (!step.src2 || !step.src2.pipeline) {
        // B8: ไม่มีสาย B เลย → แอปคืนภาพ A เดิม (ดู passthroughNoSrc2 ใน registry.js) — ไม่คำนวณกับ path_b
        bodyA.push(`# ── ขั้นที่ ${stepNo}: ${thaiName(opId)} (${opId}) ── ยังไม่ได้ต่อสาย B — แอปคืนภาพ A เดิม จึงข้ามขั้นนี้`, "");
        continue;
      }
      // เรนเดอร์สาย B ของ step นี้ก่อนบรรทัดของ step หลักเสมอ (step หลักต้องใช้ img_b ที่ประมวลผลแล้ว เช่น GEN.imabsdiff อ่าน img_b ตรง ๆ)
      emitBChain(step, 0, bodyA, appOnly);
    }
    if (g.needs === "gray" && !ctxA.gray) {
      bodyA.push("img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (แปลงขาวดำอัตโนมัติ)");
      ctxA.gray = true;
    }
    bodyA.push(`# ── ขั้นที่ ${stepNo}: ${thaiName(opId)} (${opId}) ──`);
    if (TEACH[opId]) bodyA.push(`# ${TEACH[opId]}`);
    for (const line of g.gen(local, ctxA)) bodyA.push(line);
    if (hasPf) bodyA.push(...pfSnapshotLines(opId, local, ctxA));
    const out = g.outOf ? g.outOf(local) : g.out;
    if (out === "gray") ctxA.gray = true;
    else if (out === "color") ctxA.gray = false;
    bodyA.push("");
  }
  const indentedA = bodyA.map((l) => (l === "" ? "" : "    " + l)).join("\n");
  return `# -*- coding: utf-8 -*-
# โค้ดนี้สร้างอัตโนมัติจากลำดับ Box ในกล่องผลลัพธ์: "${name}" (มี 2 ภาพนำเข้า)
# ทำ image processing ตามขั้นตอนเดียวกับที่เห็นในโปรแกรม${appOnlyHeader(appOnly)}
import cv2
import numpy as np
import sys

# บังคับให้ข้อความที่ print() ออกมาเป็น UTF-8 เสมอ — บนวินโดวส์ เวลาส่งผลไปเก็บในไฟล์
# (เช่น python code.py > log.txt) ระบบจะใช้รหัสอักขระเดิมของเครื่อง ซึ่งเขียนภาษาไทยหรือ
# สัญลักษณ์อย่าง px² / ∞ ไม่ได้ แล้วโปรแกรมจะหยุดกลางคันด้วย UnicodeEncodeError
# หมายเหตุ: ถ้านำไฟล์นี้ไป import ใช้ในโปรแกรมอื่น สองบรรทัดล่างนี้จะทำงานตั้งแต่ตอน import และเปลี่ยน
# sys.stdout ของโปรแกรมนั้นทั้งโปรเซสเป็น utf-8 ด้วย (ปกติเป็นผลดี) — ถ้าไม่ต้องการ ให้ย้ายไปไว้ใต้ if __name__
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def process(path_a, path_b):
    img = cv2.imread(path_a)    # ภาพหลัก (Src A)
    img_b = cv2.imread(path_b)  # ภาพที่สอง (Src B)
    if img is None or img_b is None:
        raise FileNotFoundError(path_a if img is None else path_b)
    img_b_raw = img_b  # ภาพ B ดิบ — แต่ละขั้นที่ใช้ภาพ B เริ่มจากสำเนาของภาพนี้ (ไม่ต่อสาย B ของขั้นก่อนหน้า)

${indentedA}
    return img


if __name__ == "__main__":
    result = process("input_a.jpg", "input_b.jpg")   # << ใส่ path ภาพทั้งสองของคุณ
    cv2.imwrite("output.png", result)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
`;
}

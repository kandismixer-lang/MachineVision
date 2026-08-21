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
    desc: "แปลงภาพจาก RGB ไปยังปริภูมิสีอื่น แล้วแสดงค่าแต่ละ channel เป็นสีตรง ๆ (false-color) เพื่อให้เห็นว่าข้อมูลถูกจัดเก็บต่างกันอย่างไร — HSV/HLS แยกเนื้อสีออกจากความสว่าง, Lab ใกล้การรับรู้ของมนุษย์, YCrCb ใช้ในระบบวิดีโอ/JPEG",
    params: [
      { key: "space", type: "select", label: "ปริภูมิสี", options: [
        { value: "hsv", label: "HSV (เนื้อสี-ความอิ่ม-ความสว่าง)", hint: "แยกเนื้อสีออกจากความสว่าง เหมาะกับตรวจจับวัตถุด้วยสี" },
        { value: "hls", label: "HLS", hint: "คล้าย HSV แต่สลับลำดับ Lightness/Saturation" },
        { value: "lab", label: "Lab", hint: "ใกล้เคียงการรับรู้สีของมนุษย์ นิยมใช้เทียบความต่างสี" },
        { value: "ycrcb", label: "YCrCb", hint: "แยกความสว่าง (Y) ออกจากสี ใช้ในวิดีโอ/JPEG" },
      ], default: "hsv" },
    ],
    apply(cv, src, p) {
      const rgb = new cv.Mat();
      if (src.channels() === 1) cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
      else src.copyTo(rgb);
      const dst = new cv.Mat();
      const code = { hsv: cv.COLOR_RGB2HSV, hls: cv.COLOR_RGB2HLS, lab: cv.COLOR_RGB2Lab, ycrcb: cv.COLOR_RGB2YCrCb }[p.space];
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
      vec.get(idx).copyTo(dst);
      vec.delete();
      spaceMat.delete();
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
      const dst = new cv.Mat();
      cv.convertScaleAbs(src, dst, p.alpha, p.beta);
      return dst;
    },
  },

  gamma: {
    name: "Gamma Correction",
    needs: "any",
    desc: "ปรับความสว่างแบบไม่เชิงเส้นด้วยเส้นโค้ง power-law: ใหม่ = (เดิม/255)^(1/γ)·255 — γ>1 ดึงส่วนมืดให้สว่างขึ้นโดยไม่ทำให้ส่วนสว่างแตก (ต่างจากการเพิ่ม brightness ตรง ๆ) ใช้แก้ภาพย้อนแสง/มืดเกิน",
    params: [
      { key: "g", type: "number", label: "ค่า Gamma (γ)", min: 0.2, max: 3, step: 0.1, default: 1.5, desc: "โค้งความสว่าง — γ>1 ดึงส่วนมืดให้สว่างขึ้น, γ<1 กดส่วนสว่างให้มืดลง" },
    ],
    apply(cv, src, p) {
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
      const lap = new cv.Mat();
      const lapAbs = new cv.Mat();
      cv.Laplacian(src, lap, cv.CV_16S, 3);
      cv.convertScaleAbs(lap, lapAbs);
      const dst = new cv.Mat();
      cv.addWeighted(src, 1, lapAbs, -p.strength, 0, dst);
      lap.delete();
      lapAbs.delete();
      return dst;
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
    desc: "จัดการ 'รูปทรง' ของวัตถุขาวในภาพขาว-ดำ ด้วยการเอาตัวแปรง (kernel) ไล่ทาบทั่วภาพ: Erode กัดวัตถุให้ผอมลง/ลบจุดจิ๋วรบกวน, Dilate พอกให้อ้วนขึ้น/อุดรูโหว่, Opening ลบจุดเล็กแต่คงขนาดเดิม, Closing อุดรูข้างในวัตถุ, Gradient ดึงเฉพาะเส้นขอบ ส่วน TopHat/BlackHat ดึงส่วนที่สว่าง/มืดกว่ารอบข้าง มักใช้ต่อหลัง threshold",
    params: [
      { key: "op", type: "select", label: "Operation", options: [
        { value: "erode", label: "Erosion (กัดให้เล็กลง)", hint: "กัดขอบวัตถุขาวให้ผอมลง ลบจุดจิ๋วรบกวน" },
        { value: "dilate", label: "Dilation (ขยายให้ใหญ่ขึ้น)", hint: "พอกขอบวัตถุขาวให้อ้วนขึ้น อุดรูโหว่เล็ก ๆ" },
        { value: "open", label: "Opening (ลบจุดเล็ก)", hint: "Erode แล้ว Dilate — ลบจุดรบกวนเล็กโดยไม่เปลี่ยนขนาดวัตถุใหญ่" },
        { value: "close", label: "Closing (อุดรู)", hint: "Dilate แล้ว Erode — อุดรูเล็ก ๆ ในตัววัตถุ" },
        { value: "gradient", label: "Gradient (ขอบวัตถุ)", hint: "Dilate ลบ Erode — ดึงเฉพาะเส้นขอบของวัตถุ" },
        { value: "tophat", label: "Top Hat", hint: "ภาพเดิม ลบ Opening — ดึงส่วนที่สว่างกว่ารอบข้าง" },
        { value: "blackhat", label: "Black Hat", hint: "Closing ลบภาพเดิม — ดึงส่วนที่มืดกว่ารอบข้าง" },
      ], default: "erode" },
      { key: "shape", type: "select", label: "รูปร่าง kernel", options: [
        { value: "rect", label: "สี่เหลี่ยม (Rect)" },
        { value: "ellipse", label: "วงรี (Ellipse)" },
        { value: "cross", label: "กากบาท (Cross)" },
      ], default: "rect" },
      { key: "ksize", type: "number", label: "ขนาด kernel", min: 1, max: 21, step: 2, default: 5, desc: "ขนาดตัวแปรงที่ไล่ทาบภาพ — ใหญ่ = ผลกัด/พอกแรงขึ้น" },
      { key: "iterations", type: "number", label: "จำนวนรอบ", min: 1, max: 5, step: 1, default: 1, desc: "ทำ operation ซ้ำกี่รอบ — มาก = ผลยิ่งเข้มขึ้น" },
    ],
    apply(cv, src, p) {
      const shape = { rect: cv.MORPH_RECT, ellipse: cv.MORPH_ELLIPSE, cross: cv.MORPH_CROSS }[p.shape];
      const k = Math.max(1, Math.round(p.ksize));
      const kernel = cv.getStructuringElement(shape, new cv.Size(k, k));
      const dst = new cv.Mat();
      const anchor = new cv.Point(-1, -1);
      if (p.op === "erode") cv.erode(src, dst, kernel, anchor, p.iterations);
      else if (p.op === "dilate") cv.dilate(src, dst, kernel, anchor, p.iterations);
      else {
        const opCode = { open: cv.MORPH_OPEN, close: cv.MORPH_CLOSE, gradient: cv.MORPH_GRADIENT, tophat: cv.MORPH_TOPHAT, blackhat: cv.MORPH_BLACKHAT }[p.op];
        cv.morphologyEx(src, dst, opCode, kernel, anchor, p.iterations);
      }
      kernel.delete();
      return dst;
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
      cv.threshold(src, dst, p.thresh, 255, p.invert ? cv.THRESH_BINARY_INV : cv.THRESH_BINARY);
      return dst;
    },
  },

  otsu_threshold: {
    name: "Otsu Threshold",
    modeHint: "ให้ระบบหาเกณฑ์ที่ดีที่สุดให้เอง ไม่ต้องเดาค่า",
    needs: "gray",
    desc: "ขี้เกียจหาเกณฑ์เองก็ให้ Otsu หาให้ มันดูการกระจายความสว่างทั้งภาพ แล้วเลือกจุดตัดที่แยกของมืดกับของสว่างออกจากกันได้เด็ดขาดที่สุด ไม่ต้องนั่งเดาค่าเอง ค่าที่มันคิดได้จะโผล่ที่แถบรายละเอียด",
    params: [
      { key: "invert", type: "boolean", label: "สลับขาว-ดำ", default: false },
    ],
    apply(cv, src, p, ctx) {
      const dst = new cv.Mat();
      const type = (p.invert ? cv.THRESH_BINARY_INV : cv.THRESH_BINARY) + cv.THRESH_OTSU;
      const val = cv.threshold(src, dst, 0, 255, type);
      if (ctx) ctx.info = `ค่าเกณฑ์ที่ Otsu คำนวณได้ = ${Math.round(val)}`;
      return dst;
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
          cv.resize(src, resized, new cv.Size(Math.round(src.cols * scale), Math.round(src.rows * scale)), 0, 0, cv.INTER_AREA);
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

  harris_corners: {
    name: "Harris Corner Detection",
    needs: "gray",
    desc: "หา 'มุม' (corner) ในภาพด้วยวิธี Harris — ดูว่าความสว่างรอบแต่ละจุดเปลี่ยนมากแค่ไหนในทุกทิศทาง จุดที่เปลี่ยนมากทุกทิศทาง = มุม ใช้หาจุดเด่นสำหรับ tracking/matching ภาพ",
    params: [
      { key: "blockSize", type: "number", label: "ขนาดพื้นที่รอบจุด", min: 2, max: 15, step: 1, default: 2, hint: "① พื้นที่ (px) ที่ใช้ดูรอบแต่ละพิกเซล" },
      { key: "ksize", type: "number", label: "ขนาด Sobel kernel", min: 1, max: 7, step: 2, default: 3, hint: "② ต้องเป็นเลขคี่เสมอ — ปรับความคมของขอบที่ใช้วิเคราะห์" },
      { key: "k", type: "number", label: "ค่า k (Harris)", min: 0.01, max: 0.3, step: 0.01, default: 0.04, hint: "③ ปกติ 0.04–0.06" },
      { key: "thresh", type: "number", label: "ความไวจับมุม (%)", min: 1, max: 100, step: 1, default: 1, hint: "④ ต่ำ=เจอมุมเยอะ(มีของปลอมปน), สูง=เฉพาะมุมชัด" },
    ],
    apply(cv, src, p, ctx) {
      // ภาพใหญ่วิ่ง cornerHarris เต็มความละเอียดจะช้า — วิเคราะห์บนภาพย่อไม่เกิน 480px แล้วคูณพิกัดกลับ
      const maxAnalysisSide = 480;
      const scale = Math.min(1, maxAnalysisSide / Math.max(src.cols, src.rows));
      const analysis = new cv.Mat();
      const resp = new cv.Mat();
      const norm = new cv.Mat();
      const mask8 = new cv.Mat();
      const kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
      let out = null;
      try {
        if (scale < 1) {
          cv.resize(src, analysis, new cv.Size(Math.round(src.cols * scale), Math.round(src.rows * scale)), 0, 0, cv.INTER_AREA);
        } else {
          src.copyTo(analysis);
        }
        const ksize = p.ksize % 2 === 0 ? p.ksize + 1 : p.ksize; // Sobel kernel ต้องเป็นเลขคี่
        cv.cornerHarris(analysis, resp, p.blockSize, ksize, p.k, cv.BORDER_DEFAULT);
        cv.normalize(resp, norm, 0, 255, cv.NORM_MINMAX, cv.CV_32FC1);
        norm.convertTo(mask8, cv.CV_8UC1);
        cv.threshold(mask8, mask8, (p.thresh / 100) * 255, 255, cv.THRESH_BINARY);
        // รวมจุดมุมที่อยู่ติดกันให้เป็นก้อนเดียว ก่อนนับ — กันนับจุดเดียวกันซ้ำหลายพิกเซล
        cv.dilate(mask8, mask8, kern);

        // build นี้ไม่ผูก cv.findNonZero มาให้ (มีแค่ใน type declaration) — อ่าน mask8.data (Uint8Array)
        // ตรงๆ แทน ได้ทั้งเร็วกว่า (ไม่เรียก WASM ทีละพิกเซล) และไม่พึ่งฟังก์ชันที่ไม่มีจริง
        const data = mask8.data;
        const cols = mask8.cols;
        const points = [];
        for (let idx = 0; idx < data.length; idx++) {
          if (data[idx]) {
            points.push(idx % cols, (idx / cols) | 0);
            if (points.length / 2 > 3000) break; // threshold ต่ำเกินไป → เจอเป็นพัน หยุดก่อนค้าง
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
        norm.delete();
        mask8.delete();
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
          cv.resize(src, analysis, new cv.Size(Math.round(src.cols * scale), Math.round(src.rows * scale)), 0, 0, cv.INTER_AREA);
        } else {
          src.copyTo(analysis);
        }
        cv.goodFeaturesToTrack(
          analysis, corners, p.maxCorners, Math.max(0.001, p.quality / 100), p.minDist * scale,
          mask, p.blockSize, !!p.useHarris, 0.04
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

  // ================= 6. ประมวลผลภาพสี (Colour Processing) ================= (คงเลขหัวข้อเดิมไว้)
  color_inrange: {
    name: "HSV inRange",
    needs: "any",
    desc: "คัดเฉพาะพิกเซลที่มี 'เนื้อสี (H)' อยู่ในช่วงที่กำหนด — เป็นหัวใจของการตรวจจับวัตถุด้วยสี (เช่น หาลูกบอลสีแดง) ทำใน HSV เพราะเนื้อสีแยกจากความสว่าง ทำให้ทนต่อแสงเปลี่ยน H ของ OpenCV มีช่วง 0–179 (แดง≈0/179, เขียว≈60, น้ำเงิน≈120)",
    params: [
      { key: "hMin", type: "number", label: "H ต่ำสุด", min: 0, max: 179, step: 1, default: 35, desc: "ขอบล่างของช่วงเนื้อสีที่จะคัดเลือก (0–179)" },
      { key: "hMax", type: "number", label: "H สูงสุด", min: 0, max: 179, step: 1, default: 85, desc: "ขอบบนของช่วงเนื้อสีที่จะคัดเลือก (0–179)" },
      { key: "sMin", type: "number", label: "S ขั้นต่ำ (กันสีซีด)", min: 0, max: 255, step: 1, default: 60, desc: "ตัดสีที่จางเกินไป (ใกล้เทา) ทิ้ง — ยิ่งสูง ยิ่งต้องเป็นสีสดจึงจะติด" },
      { key: "vMin", type: "number", label: "V ขั้นต่ำ (กันส่วนมืด)", min: 0, max: 255, step: 1, default: 60, desc: "ตัดส่วนที่มืดเกินไปทิ้ง — ยิ่งสูง ยิ่งต้องสว่างจึงจะติด" },
      { key: "output", type: "select", label: "แสดงผลเป็น", options: [
        { value: "mask", label: "Mask ขาวดำ" },
        { value: "extract", label: "เฉพาะส่วนที่เป็นสีนั้น" },
      ], default: "extract" },
    ],
    apply(cv, src, p) {
      const rgb = new cv.Mat();
      if (src.channels() === 1) cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
      else src.copyTo(rgb);
      const hsv = new cv.Mat();
      cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
      const mask = new cv.Mat();
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
      let dst;
      if (p.output === "mask") {
        dst = new cv.Mat();
        mask.copyTo(dst);
      } else {
        dst = new cv.Mat();
        cv.bitwise_and(rgb, rgb, dst, mask);
      }
      rgb.delete();
      hsv.delete();
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
};

// ===================================================================
// รวม op หลายตัวในกลุ่มเดียว → node เดียว (มี select "เลือกวิธี")
// ===================================================================
const MODE_KEY = "mode"; // key ของ select ที่ใช้เลือก op

// สร้าง merged filter จากรายชื่อ op
function mergeGroup(name, group, opIds, defaultMode, legacyOpIds = []) {
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
      params.push({ ...p, key: id + "__" + p.key, showIf: { key: MODE_KEY, value: id } });
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
  };
}

// FILTERS = node ที่แสดงบน palette จริง (1 กลุ่ม = 1 node)
export const FILTERS = {
  color_space: mergeGroup("Color Space", "Color Space",
    ["grayscale", "colorspace", "channel_split", "color_inrange", "invert"], "grayscale"),

  image_adjust: mergeGroup("Image Adjust", "Image Adjust",
    ["brightness_contrast", "gamma", "hist_equalize", "clahe"], "brightness_contrast"),

  sharpen: mergeGroup("Sharpen", "Sharpen",
    ["unsharp", "laplacian_sharpen"], "unsharp"),

  noise: mergeGroup("Noise", "Noise",
    ["salt_pepper_noise"], "salt_pepper_noise"),

  blur: mergeGroup("Blur", "Blur",
    ["median_blur", "gaussian_blur", "box_blur", "bilateral"], "gaussian_blur"),

  // กลุ่มนี้มี op เดียว — ใช้ตรง ๆ (ไม่ต้องมี select เลือกวิธี)
  morphology: { ...OPS.morphology, group: "Morphology" },

  threshold: mergeGroup("Threshold", "Threshold",
    ["binary_threshold", "otsu_threshold", "adaptive_threshold"], "binary_threshold"),

  edge_detection: mergeGroup("Edge Detection", "Edge Detection",
    ["canny", "sobel"], "canny"),

  shape_analysis: mergeGroup("Shape Analysis", "Shape Analysis",
    ["find_contours", "connected_components"], "find_contours", ["watershed"]),

  // Watershed เป็น segmentation algorithm แยกวัตถุที่แตะกัน ไม่ใช่การนับ component
  watershed: { ...OPS.watershed, group: "Segmentation" },

  object_detection: mergeGroup("Object Detection", "Object Detection",
    ["bounding_boxes", "hough_circles", "harris_corners", "shi_tomasi_corners"], "bounding_boxes"),

  colour: mergeGroup("Colour", "Colour",
    ["color_inrange", "invert"], "color_inrange"),

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
};

// ลำดับกลุ่มตามหัวข้อการสอน
export const GROUP_ORDER = [
  "Color Space",
  "Image Adjust",
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
];

// resolve needs รองรับทั้ง string และ function (merged filter)
export function needsOf(filter, params) {
  return typeof filter.needs === "function" ? filter.needs(params || {}) : filter.needs;
}

// คืนค่า default params ของ filter (ใช้ตอนเพิ่ม node ใหม่)
export function defaultParams(filterId) {
  const f = FILTERS[filterId];
  const p = {};
  for (const param of f.params) p[param.key] = param.default;
  return p;
}

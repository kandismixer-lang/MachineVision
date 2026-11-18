// ตัวรันโมเดล Model Detect แบบ native (onnxruntime-node) ใน main process — เร็วกว่า WASM ใน renderer มาก
// (RF-DETR nano 384, RTX 3050: WASM 8 thread ~0.4 วิ/รอบ · native CPU ~0.09 · DirectML GPU ~0.02)
// ลำดับตัวรัน: DirectML (GPU) → CPU · ถ้าโหลด onnxruntime-node ไม่ได้เลย renderer จะถอยไปใช้ WASM worker เดิม
// บังคับตัวรันได้ด้วย env MV_DL_EP=cpu | dml | wasm (wasm = ปิด native ทั้งหมด)
let ort = null;
let ortError = null;
let session = null;
let inputName = "";
let outputNames = [];
let ep = "";
let gen = 0; // โหลดโมเดลใหม่ = รุ่นใหม่ → คำขอ run ของโมเดลเก่าที่ค้างคิวตอบ skipped ทันที
let queue = Promise.resolve(); // run/โหลดทีละงาน — tile 8 ชิ้นที่ยิงมาพร้อมกันจะเข้าคิวตามลำดับ

const forced = (process.env.MV_DL_EP || "").toLowerCase();

async function loadOrt() {
  if (ort || ortError) return ort;
  if (forced === "wasm") {
    ortError = "ปิด native ด้วย MV_DL_EP=wasm";
    return null;
  }
  try {
    const mod = await import("onnxruntime-node");
    ort = mod.default || mod;
  } catch (e) {
    ortError = e?.message || String(e);
  }
  return ort;
}

export async function nativeAvailable() {
  return !!(await loadOrt());
}

// ขนาด input จาก metadata [1,3,H,W] — ถ้าเป็นมิติแบบสัญลักษณ์ (dynamic) ใช้ imgsz ที่ renderer แกะจากไฟล์ หรือ 640
function inputSizeOf(meta, imgsz) {
  const shape = meta?.shape || [];
  const h = shape[2];
  if (typeof h === "number" && h > 0) return h;
  return typeof imgsz === "number" && imgsz > 0 ? imgsz : 640;
}

function enqueue(job) {
  const p = queue.then(job, job);
  queue = p.catch(() => {});
  return p;
}

export function nativeLoad(bytes, imgsz) {
  gen += 1;
  return enqueue(async () => {
    const o = await loadOrt();
    if (!o) return { ok: false, error: ortError || "onnxruntime-node ใช้ไม่ได้" };
    if (session) {
      try { await session.release(); } catch { /* ignore */ }
      session = null;
    }
    const buf = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const order = forced === "cpu" ? ["cpu"] : forced === "dml" ? ["dml"] : ["dml", "cpu"];
    const errors = [];
    for (const name of order) {
      try {
        session = await o.InferenceSession.create(buf, { executionProviders: [name], graphOptimizationLevel: "all" });
        ep = name;
        break;
      } catch (e) {
        errors.push(`${name}: ${e?.message || e}`);
      }
    }
    if (!session) return { ok: false, error: errors.join(" | ") };
    inputName = session.inputNames[0];
    outputNames = session.outputNames.slice();
    const outMeta = session.outputMetadata || [];
    const ranks = outputNames.map((n) => (outMeta.find((x) => x.name === n)?.shape || []).length);
    return {
      ok: true,
      ep,
      inputSize: inputSizeOf((session.inputMetadata || [])[0], imgsz),
      outputNames,
      outputRanks: ranks,
    };
  });
}

// รัน 1 ครั้ง: data = Float32Array CHW [1,3,size,size] · คืน { outputs: { ชื่อ: { dims, data } } }
export function nativeRun(data, size) {
  const myGen = gen;
  return enqueue(async () => {
    if (!session || myGen !== gen) return { skipped: true };
    const f32 = data instanceof Float32Array ? data : new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
    const t = new ort.Tensor("float32", f32, [1, 3, size, size]);
    const res = await session.run({ [inputName]: t });
    const outputs = {};
    for (const n of outputNames) outputs[n] = { dims: Array.from(res[n].dims), data: res[n].data };
    return { outputs };
  });
}

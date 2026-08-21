import { FILTERS, defaultParams } from "./filters/registry.js";
import { OP_TO_FILTER, TOOL_SECTIONS } from "./ui/palette-panel.js";

// ดึงรายการจาก manifest เดียวกับ Palette เพื่อให้เพิ่ม/ลบ Function แล้ว test เปลี่ยนตามทันที
const CASES = TOOL_SECTIONS.flatMap((section) => section.ops.map((opId) => [opId, OP_TO_FILTER[opId]]));
if (CASES.some(([, filterId]) => !filterId || !FILTERS[filterId])) {
  throw new Error("Palette contains an operation without a registered filter mapping");
}

const report = document.getElementById("report");
const query = new URLSearchParams(location.search);
const WIDTH = Math.max(64, Math.min(1920, Number(query.get("width")) || 320));
const HEIGHT = Math.max(64, Math.min(1080, Number(query.get("height")) || 240));
const REPEATS = Math.max(1, Math.min(10, Number(query.get("repeats")) || 1));
const canvas = document.createElement("canvas");
canvas.width = WIDTH;
canvas.height = HEIGHT;
const g = canvas.getContext("2d", { willReadFrequently: true });

// Mixed fixture: gradient, touching shapes, separate components, a clear circle and noise.
const grad = g.createLinearGradient(0, 0, WIDTH, HEIGHT);
grad.addColorStop(0, "#172554"); grad.addColorStop(1, "#fde68a");
g.fillStyle = grad; g.fillRect(0, 0, WIDTH, HEIGHT);
g.fillStyle = "#ef4444"; g.fillRect(WIDTH * .08, HEIGHT * .15, WIDTH * .27, HEIGHT * .29);
g.fillStyle = "#22c55e"; g.fillRect(WIDTH * .28, HEIGHT * .23, WIDTH * .27, HEIGHT * .29);
g.fillStyle = "#f8fafc"; g.beginPath();
g.arc(WIDTH * .76, HEIGHT * .34, Math.min(WIDTH, HEIGHT) * .10, 0, Math.PI * 2); g.fill();
g.fillStyle = "#111827";
g.fillRect(WIDTH * .09, HEIGHT * .69, WIDTH * .14, HEIGHT * .14);
g.fillRect(WIDTH * .38, HEIGHT * .71, WIDTH * .17, HEIGHT * .13);
for (let i = 0; i < 650; i++) {
  const v = i % 2 ? 255 : 0;
  g.fillStyle = `rgb(${v},${v},${v})`;
  g.fillRect((i * 47) % WIDTH, (i * 83) % HEIGHT, 1, 1);
}
const fixture = g.getImageData(0, 0, WIDTH, HEIGHT);

const POSITIVE_NOTE = {
  otsu_threshold: /Otsu.*=\s*\d+/,
  watershed: /แยกได้\s+[1-9]\d*\s+วัตถุ/,
  find_contours: /พบวัตถุ\s+[1-9]\d*\s+ชิ้น/,
  connected_components: /พบวัตถุ\s+[1-9]\d*\s+ชิ้น/,
  bounding_boxes: /ตรวจพบวัตถุ\s+[1-9]\d*\s+ชิ้น/,
  hough_circles: /พบวงกลม\s+[1-9]\d*\s+วง/,
};

function paramsFor(opId, filterId) {
  const params = defaultParams(filterId);
  if (FILTERS[filterId].params.some((p) => p.key === "mode")) params.mode = opId;
  return params;
}

function runCase(worker, opId, filterId) {
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const timer = setTimeout(() => {
      worker.onmessage = null;
      worker.terminate(); // ห้ามปล่อย response เก่าหลุดไปจับกับ case ถัดไป
      reject(new Error(`${opId}: TIMEOUT > 5 s`));
    }, 5000);
    worker.onmessage = (event) => {
      if (event.data.type !== "result" && event.data.type !== "error") return;
      const ms = Math.round((performance.now() - started) * 10) / 10;
      if (event.data.type === "error") {
        clearTimeout(timer);
        resolve({ opId, ok: false, ms, error: event.data.message });
      }
      else {
        const r = event.data.results?.[0];
        if (!r || r.displayId !== opId) return; // response เก่าหรือผิด case — รอของ case ปัจจุบัน
        clearTimeout(timer);
        const pixels = r.buffer instanceof ArrayBuffer ? new Uint8ClampedArray(r.buffer) : null;
        const expectedBytes = WIDTH * HEIGHT * 4;
        let min = 255, max = 0, hash = 2166136261;
        if (pixels) {
          // sample ทุก ~4K จุด: พอจับภาพว่าง/สีเดียวโดยไม่ทำให้ test ช้า
          const stride = Math.max(4, Math.floor(pixels.length / 4096 / 4) * 4);
          for (let i = 0; i < pixels.length; i += stride) {
            const v = pixels[i]; min = Math.min(min, v); max = Math.max(max, v);
            hash = Math.imul(hash ^ v, 16777619) >>> 0;
          }
        }
        const info = (event.data.notes || []).map((n) => n.info || "").filter(Boolean).join(" | ");
        const checks = {
          dimensions: r.width === WIDTH && r.height === HEIGHT,
          buffer: Boolean(pixels && pixels.byteLength === expectedBytes),
          nonConstant: Boolean(pixels && max > min),
          semanticNote: POSITIVE_NOTE[opId] ? POSITIVE_NOTE[opId].test(info) : true,
        };
        const ok = Object.values(checks).every(Boolean);
        resolve({ opId, ok, ms, checks, signature: pixels ? `${min}-${max}-${hash}` : "none",
          error: ok ? "" : `Output assertion failed (${info || "no note"})`, notes: event.data.notes || [] });
      }
    };
    const copy = new Uint8ClampedArray(fixture.data);
    worker.postMessage({ type: "frame", width: WIDTH, height: HEIGHT, buffer: copy.buffer,
      pipelines: [{ connected: true, displayId: opId,
        pipeline: [{ id: filterId, params: paramsFor(opId, filterId), enabled: true }] }] }, [copy.buffer]);
  });
}

async function main() {
  const worker = new Worker(new URL("./cv/cv-worker.js", import.meta.url), { type: "module" });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("OpenCV init timeout")), 15000);
    worker.onmessage = (event) => {
      if (event.data.type === "ready") { clearTimeout(timer); resolve(); }
    };
    worker.postMessage({ type: "init" });
  });
  const rows = [];
  for (let pass = 1; pass <= REPEATS; pass++) {
    for (const [opId, filterId] of CASES) {
      const row = await runCase(worker, opId, filterId);
      row.pass = pass;
      rows.push(row);
      report.textContent = JSON.stringify(rows, null, 2);
    }
  }
  worker.terminate();
  window.__FILTER_SMOKE_RESULT__ = rows;
  document.title = rows.every((r) => r.ok) ? "PASS - Filter smoke test" : "FAIL - Filter smoke test";
}

main().catch((error) => {
  report.textContent = error.stack || String(error);
  document.title = "FAIL - Filter smoke test";
});

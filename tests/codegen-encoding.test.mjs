// U2 — โค้ด Python ที่ export ต้องพิมพ์ข้อความออกมาได้ทุกเครื่อง ไม่ว่าระบบจะตั้งรหัสอักขระอะไรไว้
//
// บั๊กเดิม (CV eng พบ): บน Windows ภาษาไทย เมื่อ stdout ถูก redirect (เช่น `python code.py > log.txt`
// หรือถูกโปรแกรมอื่นเรียกแล้วอ่านผลผ่าน pipe) Python จะเลือก encoding ของ stdout ตาม locale = cp874
// ซึ่งเขียน "²" (regionprops: px²) และ "∞" (PSNR: ค่าอนันต์) ไม่ได้ → print() โยน UnicodeEncodeError
// สคริปต์ตายกลางคันทั้งที่ประมวลผลภาพเสร็จแล้ว · บนวินโดวส์ที่ไม่ใช่ภาษาไทย (cp1252) หนักกว่านั้นอีก:
// ข้อความภาษาไทย "ทุกบรรทัด" ที่ print ออกมาก็ crash เหมือนกัน
// เทสต์ชุดเดิมมองไม่เห็นบั๊กคลาสนี้เพราะตั้ง PYTHONIOENCODING=utf-8 ให้ทุกครั้ง (ช่วยเหลือ encoding ไว้)
//
// วิธีแก้ที่เลือก (codegen.js): บังคับ stdout ของสคริปต์เป็น UTF-8 ที่หัวไฟล์ แทนการไล่เปลี่ยนอักขระเป็น
// ASCII — เพราะข้อความที่ผู้เรียนต้องอ่านเป็นภาษาไทยเกือบทั้งหมด การ ASCII-ify ทั้งหมดคือการทำลายผลลัพธ์
// ที่ผู้เรียนอ่าน และ "²/∞" เป็นแค่ 2 ตัวจากบั๊กคลาสเดียวกันที่กว้างกว่ามาก
//
// เทสต์ทุกตัวในไฟล์นี้ **ห้ามตั้ง PYTHONIOENCODING** (ENC-2/3/5) หรือจงใจตั้งเป็น codepage แคบ ๆ
// (ENC-4) เพื่อให้เห็นบั๊กคลาสนี้ทันทีถ้ามีใครถอดการแก้ออกในอนาคต
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { buildPythonCode } from '../src/renderer/codegen.js';
import { OPS } from '../src/renderer/filters/registry.js';

// ไฟล์นี้ไม่ต้องใช้ OpenCV.js เลย (สร้างภาพทดสอบเป็น raw bytes ตรง ๆ) — รันเร็วและไม่กวน allocator ของ
// เทสต์ leak ไฟล์อื่น (P-14)
const GUARD_LINES = [
  'import sys',
  'if hasattr(sys.stdout, "reconfigure"):',
  '    sys.stdout.reconfigure(encoding="utf-8", errors="replace")',
];

const TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'codegen-enc-'));
process.on('exit', () => {
  try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch { /* best-effort */ }
});
let __pyAvailable = null;
function pythonAvailable() {
  if (__pyAvailable === null) {
    try { execFileSync('python', ['-c', 'import cv2, numpy'], { encoding: 'utf8' }); __pyAvailable = true; }
    catch { __pyAvailable = false; }
  }
  return __pyAvailable;
}
const HAS_PYTHON = pythonAvailable();

// ภาพ BGR แบบ raw: พื้นดำ + สี่เหลี่ยมขาว (ให้ regionprops เจอวัตถุจริง จะได้พิมพ์บรรทัดที่มี px²)
function bgrWithSquare(w, h, rect) {
  const data = Buffer.alloc(w * h * 3, 0);
  for (let y = rect.y; y < rect.y + rect.h; y++) {
    for (let x = rect.x; x < rect.x + rect.w; x++) data.fill(255, (y * w + x) * 3, (y * w + x) * 3 + 3);
  }
  return { data, width: w, height: h, channels: 3 };
}

// รันสคริปต์ที่ gen มา โดยอ่าน/เขียนภาพเป็น raw bytes · `env` ใช้ปรับ PYTHONIOENCODING ของแต่ละเทสต์
// stdout ถูกอ่านผ่าน pipe เสมอ (execFileSync) = การ redirect จริงที่ทำให้บั๊กนี้โผล่
function runPython(code, raws, env) {
  let patched = code;
  if (raws.length === 1) {
    const r = raws[0];
    assert.ok(code.includes('img = cv2.imread(path)'), 'expected the single-image loader line');
    patched = patched.replace('img = cv2.imread(path)', `img = np.fromfile(path, dtype=np.uint8).reshape((${r.height}, ${r.width}, ${r.channels}))`);
  } else {
    const [a, b] = raws;
    assert.ok(code.includes('img = cv2.imread(path_a)'), 'expected the two-image loader line for A');
    patched = patched
      .replace('img = cv2.imread(path_a)', `img = np.fromfile(path_a, dtype=np.uint8).reshape((${a.height}, ${a.width}, ${a.channels}))`)
      .replace('img_b = cv2.imread(path_b)', `img_b = np.fromfile(path_b, dtype=np.uint8).reshape((${b.height}, ${b.width}, ${b.channels}))`);
  }
  const mainIdx = patched.indexOf('if __name__ == "__main__":');
  assert.ok(mainIdx > 0, 'generated code must contain a __main__ guard to strip');
  patched = patched.slice(0, mainIdx);
  const args = raws.map((_, i) => `sys.argv[${i + 1}]`).join(', ');
  patched += `\nimport sys\nresult = process(${args})\nresult.tofile(sys.argv[${raws.length + 1}])\n`;

  const id = Math.random().toString(36).slice(2);
  const scriptPath = path.join(TMP_DIR, `s_${id}.py`);
  fs.writeFileSync(scriptPath, patched, 'utf8');
  const inPaths = raws.map((r, i) => {
    const p = path.join(TMP_DIR, `in${i}_${id}.bin`);
    fs.writeFileSync(p, r.data);
    return p;
  });
  const outPath = path.join(TMP_DIR, `out_${id}.bin`);
  const childEnv = { ...process.env };
  delete childEnv.PYTHONIOENCODING; // ค่าเริ่มต้นของไฟล์นี้: ไม่ช่วยเหลือ encoding ใด ๆ
  if (env && env.PYTHONIOENCODING) childEnv.PYTHONIOENCODING = env.PYTHONIOENCODING;
  // อ่าน stdout เป็น buffer แล้ว decode เป็น utf-8 เอง — สคริปต์ต้องเขียน UTF-8 ออกมาเสมอหลังการแก้
  const stdout = execFileSync('python', [scriptPath, ...inPaths, outPath], { env: childEnv });
  return { stdout: stdout.toString('utf8'), bytes: new Uint8Array(fs.readFileSync(outPath)) };
}

const rpPipeline = [{ id: 'regionprops', params: { minArea: 10, maxObjects: 500 } }];
const psnrPipeline = [{ id: 'psnr', params: {}, src2: { pipeline: [] } }];
const squareRaw = bgrWithSquare(40, 30, { x: 8, y: 6, w: 12, h: 10 });

// ===================================================================
// ENC-1/ENC-5 — โครงของโค้ดที่ gen ออกมา (static)
// ===================================================================
test('enc-1. both generators (1 image and 2 images) emit the stdout-UTF-8 guard exactly once, after the imports and before any print() -- and the "# -*- coding: utf-8 -*-" header is still the very first line', () => {
  for (const [label, pipeline] of [['1-image', rpPipeline], ['2-image', psnrPipeline]]) {
    const code = buildPythonCode(pipeline, { displayName: 'X' });
    const lines = code.split('\n');
    assert.equal(lines[0], '# -*- coding: utf-8 -*-', `${label}: coding header must stay on line 1`);
    for (const g of GUARD_LINES) {
      const n = lines.filter((l) => l === g).length;
      assert.equal(n, 1, `${label}: guard line must appear exactly once: ${g}`);
    }
    const guardIdx = lines.indexOf(GUARD_LINES[2]);
    assert.ok(guardIdx > lines.indexOf('import numpy as np'), `${label}: the guard must come after the imports`);
    assert.ok(guardIdx < lines.findIndex((l) => l.startsWith('def process')), `${label}: the guard must run before process() is even defined`);
    const firstPrint = lines.findIndex((l) => l.trim().startsWith('print('));
    assert.ok(firstPrint > guardIdx, `${label}: every print() must come after the guard (first print at line ${firstPrint})`);
  }
});

test('enc-5. every op in OPS generates a script that carries the guard (no code path in buildPythonCode can emit a script whose print() depends on the machine locale)', () => {
  let printers = 0;
  for (const id of Object.keys(OPS)) {
    const code = buildPythonCode([{ id, params: {} }]);
    for (const g of GUARD_LINES) assert.ok(code.includes(g), `${id}: missing guard line ${g}`);
    if (/^\s*print\(/m.test(code)) printers++;
  }
  assert.ok(printers >= 4, `expected several ops to print something (found ${printers}) -- otherwise this sweep proves nothing`);
});

// ===================================================================
// ENC-2…ENC-4 — รันจริงด้วย cv2 โดยไม่ช่วยเหลือ encoding
// ===================================================================
test('enc-2. regionprops script runs with stdout redirected and NO PYTHONIOENCODING: exit 0 and the "px²" line really reaches stdout (this is the exact line that crashed with UnicodeEncodeError on Windows cp874)', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  const { stdout, bytes } = runPython(buildPythonCode(rpPipeline), [squareRaw]);
  assert.match(stdout, /พบวัตถุ 1 ชิ้น/, `Thai output must survive the redirect, got: ${JSON.stringify(stdout)}`);
  assert.match(stdout, /px²/, 'the superscript-2 character must reach stdout unharmed');
  assert.match(stdout, /orientation_deg/);
  assert.equal(bytes.length, 40 * 30 * 3, 'the image result must still be produced');
});

test('enc-3. PSNR script on two identical images (the mse == 0 branch) runs with NO PYTHONIOENCODING: exit 0 and the "∞" label reaches stdout', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  const { stdout } = runPython(buildPythonCode(psnrPipeline), [squareRaw, squareRaw]);
  assert.match(stdout, /PSNR = ∞ \(เหมือนกันทุกพิกเซล\) \(MSE = 0\.00\)/, `got: ${JSON.stringify(stdout)}`);
});

test('enc-4. the fix is locale-independent, not "it happens to work on a Thai Windows": forcing stdout to cp874 AND to cp1252 (where even plain Thai is unencodable) still gives exit 0 and byte-identical stdout', { skip: !HAS_PYTHON && 'python/cv2 not available in this environment' }, () => {
  const rpCode = buildPythonCode(rpPipeline);
  const psnrCode = buildPythonCode(psnrPipeline);
  const baseRp = runPython(rpCode, [squareRaw]).stdout;
  const basePsnr = runPython(psnrCode, [squareRaw, squareRaw]).stdout;
  for (const cp of ['cp874', 'cp1252', 'ascii']) {
    assert.equal(runPython(rpCode, [squareRaw], { PYTHONIOENCODING: cp }).stdout, baseRp, `regionprops output changed under PYTHONIOENCODING=${cp}`);
    assert.equal(runPython(psnrCode, [squareRaw, squareRaw], { PYTHONIOENCODING: cp }).stdout, basePsnr, `PSNR output changed under PYTHONIOENCODING=${cp}`);
  }
});

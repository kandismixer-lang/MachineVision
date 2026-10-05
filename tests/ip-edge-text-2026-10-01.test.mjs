// 2026-10-01 — เทสต์ตรึง "ข้อความที่ใช้สอน" ของชุด Edge (prewitt_edge / roberts_edge / log_edge /
// region_growing) หลังรอบ recheck ของ computer-vision-engineer (ดู WORKLOG-2026-09-30-ip-edge-batch.md
// หัวข้อ "รอบ CV eng recheck (2026-10-01)" ข้อ 5 MINOR-A…F)
//
// ทำไมต้องมีไฟล์นี้: แอปนี้สอน machine vision ให้คนที่ยังไม่รู้หลักการ ข้อความที่สอนผิดจึงเป็นบั๊กจริง
// ข้อความ 3 จุดที่เคยผิดถูกวัดด้วยภาพทดสอบจริงแล้วว่าตรงข้ามกับสิ่งที่ผู้เรียนเห็นบนจอ:
//   - roberts "จับขอบทแยงได้ดีกว่าขอบแนวตั้ง/แนวนอน" → วัดจริงที่ dir:"both" ได้ ตั้ง/นอน 255 แต่ทแยง 128
//   - prewitt "ไวต่อ noise มากกว่า Sobel"            → วัดจริง SNR prewitt 9.70 > sobel 8.91 (ตรงข้าม)
//   - log    "เส้นขอบ…ปิดเป็นวงเสมอ"                 → ที่ค่า default (thr 8) ขาดเป็น 1,143 ชิ้น
//
// นโยบายของไฟล์นี้: **ห้าม assert ข้อความเต็มคำต่อคำ** เพราะคนขัดเกลาสำนวนได้ตลอด เทสต์จึงจับเฉพาะ
// "สาระที่ห้ามกลับมา" (regex ของวลีที่สอนผิด) กับ "สาระที่ต้องมี" (คำสำคัญ 1–2 คำ) เท่านั้น
//
// ไฟล์นี้ตั้งใจ **ไม่โหลด OpenCV** — เป็นเทสต์ของข้อมูลใน registry ล้วน ๆ จึง import OPS ตรง ๆ แล้วจบใน
// หลักมิลลิวินาที (registry.js ไม่มี side-effect ตอน import — เทสต์ของ U3 ก็ import แบบนี้)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { OPS } from '../src/renderer/filters/registry.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REGISTRY_PATH = path.join(HERE, '..', 'src', 'renderer', 'filters', 'registry.js');

// ช่องว่างในภาษาไทยมีได้ทั้งแบบมีและไม่มี — normalize ก่อนเทียบ regex เสมอ
function norm(s) { return String(s).replace(/\s+/g, ' '); }
function paramOf(opId, key) {
  const p = (OPS[opId].params || []).find((x) => x.key === key);
  assert.ok(p, `${opId}.params ต้องมีคีย์ ${key}`);
  return p;
}
function optionOf(opId, key, value) {
  const o = (paramOf(opId, key).options || []).find((x) => x.value === value);
  assert.ok(o, `${opId}.${key} ต้องมี option "${value}"`);
  return o;
}

// ===================================================================
// 1. ข้อความที่สอนผิด — ห้ามกลับมา
// ===================================================================

// MINOR-A: ที่ dir:"both" (ค่า default) ขอบแนวตั้ง/แนวนอนได้ 255 ส่วนขอบทแยง 45° ได้ 128
// → ห้ามมีประโยคที่สื่อว่า "ทแยง … ดีกว่า/ชัดกว่า/เด่นกว่า … แนวตั้ง/แนวนอน" อีก
test('ipet-1 roberts_edge.desc ไม่มีคำกล่าวว่า "ทแยงดีกว่าแนวตั้ง/แนวนอน"', () => {
  const d = norm(OPS.roberts_edge.desc);
  assert.doesNotMatch(d, /ทแยง[^·]{0,80}(ดีกว่า|ชัดกว่า|เด่นกว่า|แรงกว่า|สว่างกว่า)/);
  // กันการเขียนสลับข้างให้ความหมายเดิม ("แนวตั้ง/แนวนอน … แย่กว่า … ทแยง")
  assert.doesNotMatch(d, /(แนวตั้ง|แนวนอน)[^·]{0,80}(แย่กว่า|สู้.{0,10}ทแยงไม่ได้)/);
});

// MINOR-B: SNR ที่วัดจริง prewitt 9.70 vs sobel 8.91 (noise sd 8) และ 18.89 vs 17.76 (sd 4)
// → ประโยคเดิมกลับหัว ห้ามกลับมา (และห้ามกลับหัวอีกทางเป็น "Sobel ไวกว่า Prewitt" ซึ่งเกินข้อมูล ~9%)
test('ipet-2 prewitt_edge.desc ไม่มีคำกล่าวว่า "Prewitt ไวต่อ noise มากกว่า Sobel"', () => {
  const d = norm(OPS.prewitt_edge.desc);
  assert.doesNotMatch(d, /Prewitt[^·]{0,60}ไวต่อ\s*noise\s*มากกว่า\s*Sobel/i);
  assert.doesNotMatch(d, /ไวต่อ\s*noise\s*มากกว่า\s*Sobel/i);
});

// MINOR-F: thr 0 → 194 ชิ้น (ปิดเป็นวง) แต่ thr 8 (default) → 1,143 ชิ้น (ขาดเป็นท่อน)
test('ipet-3 log_edge.desc ไม่มีคำว่า "ปิดเป็นวงเสมอ"', () => {
  const d = norm(OPS.log_edge.desc);
  assert.doesNotMatch(d, /ปิดเป็นวง\s*เสมอ/);
  assert.doesNotMatch(d, /เป็นวงปิด\s*เสมอ/);
});

// ===================================================================
// 2. สาระที่ต้องมี (จับคำสำคัญ ไม่ใช่ประโยคเต็ม — ทนต่อการเรียบเรียงใหม่)
// ===================================================================

// MINOR-A: ต้องบอกว่าในโหมดรวมสองแนว ขอบทแยงได้ความสว่างแค่ครึ่งเดียว (128 เทียบ 255)
test('ipet-4 roberts_edge.desc อธิบายว่าโหมดรวมสองแนวให้ขอบทแยงครึ่งเดียว', () => {
  const d = norm(OPS.roberts_edge.desc);
  assert.match(d, /255/);
  assert.match(d, /128|ครึ่ง/);
});

// MINOR-B: ต้องบอกว่าภาพ Prewitt จางกว่า Sobel (184.7 vs 245.4 ≈ 3/4) และ noise ใกล้เคียงกัน
test('ipet-5 prewitt_edge.desc อธิบายว่าภาพจางกว่า Sobel และ noise ใกล้เคียงกัน', () => {
  const d = norm(OPS.prewitt_edge.desc);
  assert.match(d, /จางกว่า|อ่อนกว่า|เข้มน้อยกว่า/);
  assert.match(d, /ใกล้เคียงกัน|พอ ?ๆ ?กัน/);
  // qa recheck 2026-10-01: อัตราส่วน 3/4 จริงเฉพาะขอบคอนทราสต์ต่ำ-กลาง — convertScaleAbs ตันที่ 255
  // (Prewitt ตอบ 3h, Sobel 4h) → h ≥ 85 ทั้งคู่ตันจนภาพเหมือนกันเป๊ะ (วัดจริง: h 10–60 = 0.750,
  // h 70 = 0.820, h 80 = 0.938, h ≥ 85 = 1.000) ข้อความจึงต้องมีเงื่อนไขกำกับเสมอ ห้ามหายไปอีก
  assert.match(d, /ตัน|คอนทราสต์สูง|ขาว-ดำ/);
});

// MINOR-F: "ปิดเป็นวง" ต้องผูกกับการตั้งเกณฑ์เป็น 0 ไม่ใช่พูดลอย ๆ
test('ipet-6 log_edge.desc ผูกเรื่องเส้นปิดเป็นวงเข้ากับการตั้งเกณฑ์ 0', () => {
  const d = norm(OPS.log_edge.desc);
  assert.match(d, /ปิดเป็นวง/);
  assert.match(d, /ขาด/); // ต้องบอกด้วยว่าเพิ่มเกณฑ์แล้วเส้นขาด
});

// MINOR-D: max|LoG| ลดตาม ~1/sigma² แต่ zeroThresh คงที่ → sigma 5 ที่ thr 8 เหลือขอบ 0.13%
test('ipet-7 log_edge.sigma.desc เตือนว่าเพิ่ม sigma แล้วต้องลดเกณฑ์ตาม', () => {
  const p = paramOf('log_edge', 'sigma');
  const d = norm(p.desc);
  assert.match(d, /เกณฑ์/);
  assert.match(d, /ลด/);
  assert.ok(typeof p.hint === 'string' && p.hint.length > 0, 'hint ต้องไม่ว่าง');
});

// MINOR-E: ภาพ noise จัด (sd ≥ 12) ทำให้ค่า default ให้ภาพเกือบดำ — ต่อ Gaussian Blur แล้วกลับมา 29%
test('ipet-8 region_growing.desc แนะนำทางแก้กรณีได้ภาพดำ (ต่อ Blur ก่อน)', () => {
  const d = norm(OPS.region_growing.desc);
  assert.match(d, /Blur|เบลอ/i);
  assert.match(d, /noise|รบกวน/i);
  assert.match(d, /ดำ|จุดเดียว/);
});

// NIT-2: โหมด response ใช้ alpha = 127/max|LoG| ต่อเฟรม → กล้องสดกะพริบ
test('ipet-9 log_edge option "response" เตือนเรื่องความสว่างกะพริบบนกล้องสด', () => {
  const o = optionOf('log_edge', 'view', 'response');
  assert.match(norm(o.hint), /กะพริบ|กระพริบ/);
});

// ===================================================================
// 3. สเกลของสไลเดอร์ (MINOR-C) — step ต้องเป็น 2 และ "ห้ามมีใครเผลอขยับ default ตอนแก้ step"
// ===================================================================
// cv.Laplacian(ksize=3) ใช้ kernel [[2,0,2],[0,-8,0],[2,0,2]] → ค่าทุกค่าเป็นจำนวนเต็มคู่
// (ยืนยัน 307,200 ค่า × 2 ภาพ: non-integer 0, odd 0) → |c−r| คู่เสมอ → thr คี่ให้ผลเท่าคู่ที่อยู่ก่อนหน้า
// (วัดจริง sigma 2: 0:43.45 1:43.45 | 2:30.17 3:30.17 | … | 8:0.89 9:0.89)
test('ipet-10 log_edge.zeroThresh step = 2 และ min/max/default ต้องไม่ถูกขยับ', () => {
  const p = paramOf('log_edge', 'zeroThresh');
  assert.equal(p.step, 2);
  assert.equal(p.min, 0);
  assert.equal(p.max, 50);
  assert.equal(p.default, 8);
});

// ตัวเลข default ของ sigma ถูก CV eng ทดสอบบนภาพจริง 6 แบบแล้วว่าดีที่สุด — ตรึงไว้กันการขยับตามอารมณ์
test('ipet-11 log_edge.sigma ยังเป็นสเกลเดิม (0.5–5 step 0.1 default 2)', () => {
  const p = paramOf('log_edge', 'sigma');
  assert.equal(p.min, 0.5);
  assert.equal(p.max, 5);
  assert.equal(p.step, 0.1);
  assert.equal(p.default, 2);
});

// ค่า default ของ region_growing: tol 20/30 จะทะลุขอบบนภาพสะอาด (33% → 54.7% → 80.2%) ห้ามเปลี่ยน
test('ipet-12 region_growing ยังใช้ค่า default เดิม (tol 12 / fixed / seed 50%)', () => {
  assert.equal(paramOf('region_growing', 'tolerance').default, 12);
  assert.equal(paramOf('region_growing', 'rangeMode').default, 'fixed');
  assert.equal(paramOf('region_growing', 'seedX').default, 50);
  assert.equal(paramOf('region_growing', 'seedY').default, 50);
});

// ทิศทางเริ่มต้นของ prewitt/roberts ต้องเป็น "both" เหมือน sobel/scharr (สลับ dropdown เทียบภาพได้ทันที)
test('ipet-13 prewitt/roberts ยังตั้งต้นที่ "รวมสองแนว"', () => {
  assert.equal(paramOf('prewitt_edge', 'dir').default, 'both');
  assert.equal(paramOf('roberts_edge', 'dir').default, 'both');
});

// hint ของ d1/d2 ถูก CV eng ยืนยันว่าตรงกับตารางที่วัด (d1 เห็นขอบเอียง ↗, d2 เห็นขอบเอียง ↘) ห้ามสลับ
test('ipet-14 roberts_edge hint ของ d1/d2 ยังชี้ทิศตรงกับที่วัดได้', () => {
  assert.match(norm(optionOf('roberts_edge', 'dir', 'd1').hint), /↗/);
  assert.match(norm(optionOf('roberts_edge', 'dir', 'd2').hint), /↘/);
});

// ===================================================================
// 4. กันการลบข้อความทิ้ง
// ===================================================================
test('ipet-15 desc ของทั้ง 4 op ต้องเป็น string ที่ไม่ว่าง', () => {
  for (const id of ['prewitt_edge', 'roberts_edge', 'log_edge', 'region_growing']) {
    assert.equal(typeof OPS[id].desc, 'string', `${id}.desc ต้องเป็น string`);
    assert.ok(OPS[id].desc.trim().length >= 40, `${id}.desc สั้นผิดปกติ (ถูกลบทิ้ง?)`);
  }
});

// NIT-4: สัญญาของ grayIfNeeded() (รองรับ 1/3/4 ช่องเท่านั้น — Mat 2 ช่อง throw C++) ต้องเขียนไว้เหนือ
// ตัวฟังก์ชัน เพราะเป็น helper ของกลางระดับไฟล์ที่ op ใหม่จะหยิบไปใช้
test('ipet-16 registry.js มีคอมเมนต์สัญญาจำนวนช่องเหนือ grayIfNeeded()', () => {
  const src = fs.readFileSync(REGISTRY_PATH, 'utf8');
  const at = src.indexOf('function grayIfNeeded');
  assert.ok(at > 0, 'ต้องมีฟังก์ชัน grayIfNeeded ใน registry.js');
  const above = norm(src.slice(Math.max(0, at - 1500), at));
  assert.match(above, /2 ช่อง/);
  assert.match(above, /1 \/ 3 \/ 4 ช่อง|1\/3\/4 ช่อง/);
});

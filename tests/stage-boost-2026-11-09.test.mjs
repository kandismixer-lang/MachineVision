// ขยายจอผลลัพธ์อัตโนมัติเมื่อมี ≥ 3 จอ (2026-11-09) — ตาราง 2×2 แบบเดิม แต่ VIEWER สูงขึ้นเป็น 70% ให้ภาพไม่เตี้ย
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { effectiveStageRatio, STAGE_BOOST_RATIO } from '../src/renderer/ui/layout-resize.js';

const read = (rel) => fs.readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');

test('sb2-1. effectiveStageRatio: boost raises small/default ratios to 70%, never shrinks a bigger user ratio, off = user value', () => {
  assert.equal(STAGE_BOOST_RATIO, 0.7);
  assert.equal(effectiveStageRatio(null, false, false), null); // ไม่ขยาย = ค่า CSS เดิม
  assert.equal(effectiveStageRatio(0.35, false, false), 0.35);
  assert.equal(effectiveStageRatio(null, true, false), 0.7); // ค่าเริ่มต้น 50% → 70%
  assert.equal(effectiveStageRatio(0.56, true, false), 0.7);
  assert.equal(effectiveStageRatio(0.85, true, false), 0.85); // ผู้ใช้ตั้งใหญ่กว่าอยู่แล้ว → ไม่ลด
  assert.equal(effectiveStageRatio(0.4, true, true), 0.4); // ผู้ใช้ปรับเองระหว่างขยาย → ค่าผู้ใช้ชนะ
});

test('sb2-2. layout-resize: render/aria use the effective ratio; boost is not persisted; user drag/preset yields; reset re-arms', () => {
  const src = read('src/renderer/ui/layout-resize.js');
  assert.match(src, /const stageRatioNow = \(\) => effectiveStageRatio\(state\.stageRatio, boostWanted, boostYield\);/);
  assert.match(src, /const ratio = stageRatioNow\(\);\s*if \(ratio != null && avail > 0\) k =/);
  assert.match(src, /state\.stageRatio = clamp\(res\.v \/ ctx\.avail, RATIO\.min, RATIO\.max\);\s*if \(boostWanted\) boostYield = true;/);
  assert.match(src, /state\.stageRatio = clamp\(r, RATIO\.min, RATIO\.max\);\s*if \(boostWanted\) boostYield = true;/);
  assert.match(src, /if \(w === "stage"\) boostYield = false;/);
  assert.ok(!/state\.stageRatio = STAGE_BOOST_RATIO|state\.stageRatio = effectiveStageRatio/.test(src), 'boost must not be written into the saved state');
  assert.match(src, /function applyBoost\(\) \{\s*if \(!boostWanted\) boostYield = false;\s*if \(drag\) return;/);
});

test('sb2-3. app.js: grid keeps the original sqrt layout and switches the boost on at >= 3 shown displays', () => {
  const app = read('src/renderer/app.js');
  assert.match(app, /import \{ initLayoutResize, initViewMenu, togglePanel, setPanelCollapsed, setStageBoost \} from "\.\/ui\/layout-resize\.js";/);
  assert.match(app, /if \(key === gridKey\) return;\s*\/\/[^\n]*\n\s*setStageBoost\(shown\.length >= 3\);/);
  assert.match(app, /Math\.min\(shown\.length, Math\.ceil\(Math\.sqrt\(shown\.length\)\) \|\| 1\)/);
});

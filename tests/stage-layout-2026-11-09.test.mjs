// จัด layout จอผลลัพธ์ให้ภาพใหญ่ที่สุด (2026-11-09): 3 จอขึ้นไปต้องไม่เหลือที่ว่างโหว่ ๆ แบบ 2×2 ตายตัว
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { bestGridCols, gridPlacement } from '../src/renderer/ui/stage-layout.js';

const A = 800 / 520; // ภาพตัวอย่าง (กว้าง)

test('sl-1. wide short viewer: 3 displays go in one row (bigger than 2×2), 2 stay side by side', () => {
  assert.equal(bestGridCols(3, 850, 370, A, 31, 12), 3);
  assert.equal(bestGridCols(2, 850, 370, A, 31, 12), 2);
  assert.equal(bestGridCols(1, 850, 370, A, 31, 12), 1);
});

test('sl-2. tall viewer stacks; square-ish viewer with 4 → 2×2; portrait images prefer more columns', () => {
  assert.equal(bestGridCols(3, 400, 1200, A, 31, 12), 1);
  assert.equal(bestGridCols(4, 1000, 700, A, 31, 12), 2);
  assert.equal(bestGridCols(3, 900, 600, 0.6, 31, 12), 3);
});

test('sl-3. chosen cols really maximise the shown image height (brute-force check over many sizes)', () => {
  const imgH = (n, W, H, a, cap, gap, cols) => {
    const rows = Math.ceil(n / cols);
    const cw = (W - gap * (cols - 1)) / cols, ch = (H - gap * (rows - 1)) / rows - cap;
    return cw > 0 && ch > 0 ? Math.min(cw / a, ch) : -1;
  };
  for (const n of [2, 3, 4, 5, 6, 7]) for (const W of [500, 900, 1400]) for (const H of [300, 600, 900]) for (const a of [0.75, 1, A, 16 / 9]) {
    const c = bestGridCols(n, W, H, a, 31, 12);
    let best = -1;
    for (let k = 1; k <= n; k++) best = Math.max(best, imgH(n, W, H, a, 31, 12, k));
    assert.ok(imgH(n, W, H, a, 31, 12, c) >= best - 0.5, `n=${n} W=${W} H=${H} a=${a} → ${c}`);
  }
});

test('sl-4. bad input falls back safely (no size yet → sqrt rule; bad aspect → 4:3)', () => {
  assert.equal(bestGridCols(3, 0, 0, A), 2);
  assert.equal(bestGridCols(5, 0, 400, A), 3);
  assert.equal(bestGridCols(3, 850, 370, NaN, 31, 12), bestGridCols(3, 850, 370, 4 / 3, 31, 12));
  assert.equal(bestGridCols(0, 850, 370, A), 1);
});

test('sl-5. gridPlacement: every cell spans 2 tracks; an incomplete last row is centred', () => {
  assert.deepEqual(gridPlacement(3, 2), [{ start: 1, span: 2 }, { start: 3, span: 2 }, { start: 2, span: 2 }]);
  assert.deepEqual(gridPlacement(3, 3).map((p) => p.start), [1, 3, 5]);
  assert.deepEqual(gridPlacement(4, 2).map((p) => p.start), [1, 3, 1, 3]);
  assert.deepEqual(gridPlacement(5, 3).map((p) => p.start), [1, 3, 5, 2, 4]);
  assert.deepEqual(gridPlacement(1, 1), [{ start: 1, span: 2 }]);
});

test('sl-6. app.js wiring: --cols = cols*2 tracks, cells get grid-column, relayout on resize and on image aspect change', () => {
  const app = fs.readFileSync(new URL('../src/renderer/app.js', import.meta.url), 'utf8');
  assert.match(app, /import \{ bestGridCols, gridPlacement \} from "\.\/ui\/stage-layout\.js";/);
  assert.match(app, /gridEl\.style\.setProperty\("--cols", cols \* 2\);/);
  assert.match(app, /cell\.root\.style\.gridColumn = `\$\{p\.start\} \/ span \$\{p\.span\}`;/);
  assert.match(app, /new ResizeObserver\(\(\) => layoutGrid\(\)\)\.observe\(gridEl\);/);
  assert.equal((app.match(/noteGridAspect\((width, height|w, h)\);/g) || []).length, 2); // ผล worker + จอ DL
  assert.ok(!/Math\.ceil\(Math\.sqrt\(shown\.length\)\)/.test(app), 'old fixed sqrt rule is gone');
});

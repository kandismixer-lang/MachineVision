// ซ่อน Display ได้ (2026-11-09): ปุ่ม "ซ่อน" ในหัวจอ → จอที่เหลือขยายเต็ม grid · เอากลับจากแถบ "ซ่อนอยู่" ในหัว VIEWER
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const read = (rel) => fs.readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');
const app = read('src/renderer/app.js');
const grid = app.slice(app.indexOf('function ensureGrid(order)'), app.indexOf('let workerReady = false;'));

test('hd-1. ensureGrid lays out only the shown displays; key covers shown + hidden; cols from shown', () => {
  assert.match(grid, /const shown = order\.filter\(\(id\) => !hiddenDisplays\.has\(id\)\);/);
  assert.match(grid, /const key = shown\.join\(","\) \+ "\|" \+ hidden\.join\(","\);/);
  assert.match(grid, /Math\.min\(shown\.length, Math\.ceil\(Math\.sqrt\(shown\.length\)\) \|\| 1\)/);
  assert.match(grid, /for \(const id of shown\) \{/);
  assert.ok(!/for \(const id of order\) \{\s*const root/.test(grid), 'hidden displays must not get a cell');
});

test('hd-2. never all hidden: when only hidden displays are left they all come back; hide button only when > 1 shown', () => {
  assert.match(grid, /order\.every\(\(id\) => hiddenDisplays\.has\(id\)\)\) \{\s*for \(const id of order\) hiddenDisplays\.delete\(id\);/);
  assert.match(grid, /if \(shown\.length > 1\) \{\s*const hideBtn/);
  assert.match(grid, /e\.stopPropagation\(\);\s*setDisplayHidden\(id, true\);/);
  assert.ok(grid.indexOf('order.every(') < grid.indexOf('const shown ='));
});

test('hd-3. restore bar + immediate repaint from the offscreen canvas; deleted displays leave the hidden set', () => {
  assert.match(grid, /btn\.addEventListener\("click", \(\) => setDisplayHidden\(id, false\)\)/);
  assert.match(grid, /hiddenBarEl\.classList\.toggle\("hidden", hidden\.length === 0\)/);
  assert.match(grid, /function refreshGridNow\(\) \{\s*ensureGrid\(connectedDisplayIds\(\)\);\s*updateCaptions\(\);[\s\S]*?cell\.ctx\.drawImage\(oc, 0, 0\);/);
  assert.match(app, /for \(const id of hiddenDisplays\) \{\s*if \(!alive\.has\(id\)\) hiddenDisplays\.delete\(id\);/);
  assert.match(app, /for \(const \[id, btn\] of hiddenBarBtns\)/); // ชื่อจอที่ซ่อนตามการเปลี่ยนชื่อ
  const html = read('src/index.html');
  assert.match(html, /<span id="lv-hidden-displays" class="lv-hidden hidden"/);
  const css = read('src/styles/app.css');
  assert.match(css, /\.stage-hide-btn \{/);
  assert.match(css, /\.lv-hidden\.hidden \{ display: none; \}/);
});

test('hd-4. hiding is view-only: processing and Save Picture still use every connected display', () => {
  assert.match(app, /function connectedDisplayIds\(\) \{\s*return state\.pipelines\.filter\(\(d\) => d\.connected\)\.map\(\(d\) => d\.displayId\);/);
  assert.match(app, /displays: state\.pipelines\.filter\(\(d\) => d\.connected && !d\.dl\)/);
  const save = app.slice(app.indexOf('async function exportDataset()'), app.indexOf('// canvas แยกสำหรับ export'));
  assert.ok(!/hiddenDisplays/.test(save));
});

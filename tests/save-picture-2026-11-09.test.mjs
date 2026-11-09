// Save Picture แบบ Save As (2026-11-09): Display เดียว = ไฟล์ตั้งชื่อเอง · หลายไฟล์ = สร้างโฟลเดอร์ตามชื่อที่ตั้งอัตโนมัติ
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { safeBaseName, ensurePngPath, uniqueFolderPath, resolveSaveTarget } from '../electron/save-target.js';

const read = (rel) => fs.readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');
const existsIn = (set) => async (p) => set.has(p);
const D = path.join('C:', 'pics');

test('sp-1. safeBaseName strips forbidden chars and trailing dots/spaces, falls back when empty', () => {
  assert.equal(safeBaseName('a:b*c?'), 'a_b_c_');
  assert.equal(safeBaseName('name. . '), 'name');
  assert.equal(safeBaseName('   '), 'FrameLab');
  assert.equal(safeBaseName('', 'result'), 'result');
  assert.equal(safeBaseName('ผลลัพธ์ 1'), 'ผลลัพธ์ 1');
});

test('sp-2. ensurePngPath always ends in .png (keeps .png, swaps other image ext, appends otherwise)', () => {
  assert.equal(ensurePngPath('x.png'), 'x.png');
  assert.equal(ensurePngPath('x.PNG'), 'x.PNG');
  assert.equal(ensurePngPath('x.jpg'), 'x.png');
  assert.equal(ensurePngPath('x'), 'x.png');
  assert.equal(ensurePngPath('v1.2'), 'v1.2.png');
});

test('sp-3. uniqueFolderPath adds " (2)", " (3)" like Windows and never reuses an existing name', async () => {
  const base = path.join(D, 'shots');
  assert.equal(await uniqueFolderPath(base, existsIn(new Set())), base);
  assert.equal(await uniqueFolderPath(base, existsIn(new Set([base]))), `${base} (2)`);
  assert.equal(await uniqueFolderPath(base, existsIn(new Set([base, `${base} (2)`]))), `${base} (3)`);
});

test('sp-4. resolveSaveTarget single → file .png in the chosen dir; multi → new folder (drops .png the dialog may add)', async () => {
  const none = existsIn(new Set());
  assert.deepEqual(await resolveSaveTarget(path.join(D, 'my shot.png'), false, none), { kind: 'file', path: path.join(D, 'my shot.png') });
  assert.deepEqual(await resolveSaveTarget(path.join(D, 'my shot'), false, none), { kind: 'file', path: path.join(D, 'my shot.png') });
  assert.deepEqual(await resolveSaveTarget(path.join(D, 'run1'), true, none), { kind: 'folder', path: path.join(D, 'run1') });
  assert.deepEqual(await resolveSaveTarget(path.join(D, 'run1.png'), true, none), { kind: 'folder', path: path.join(D, 'run1') });
  const taken = existsIn(new Set([path.join(D, 'run1')]));
  assert.deepEqual(await resolveSaveTarget(path.join(D, 'run1'), true, taken), { kind: 'folder', path: path.join(D, 'run1 (2)') });
});

test('sp-5. main.js: save dialog + folder mkdir + single-file writes only to dialog-approved paths', () => {
  const main = read('electron/main.js');
  assert.match(main, /import \{ safeBaseName, resolveSaveTarget \} from "\.\/save-target\.js";/);
  assert.match(main, /ipcMain\.handle\("export:chooseSaveTarget"[\s\S]*?dialog\.showSaveDialog\(/);
  assert.match(main, /filters: multi \? \[\] : \[\{ name: "PNG Image", extensions: \["png"\] \}\]/);
  assert.match(main, /target\.kind === "folder"\) \{\s*await fs\.mkdir\(target\.path, \{ recursive: true \}\);\s*\} else \{\s*approvedSaveFiles\.add\(target\.path\);/);
  assert.match(main, /export:writeSavedFile[\s\S]*?if \(!approvedSaveFiles\.has\(filePath\)\) throw/);
  assert.ok(!/csv/i.test(main), 'electron/main.js must not know about CSV');
});

test('sp-6. preload exposes chooseSaveTarget/writeSavedFile', () => {
  const pre = read('electron/preload.cjs');
  assert.match(pre, /chooseSaveTarget: \(count, defaultName\) => ipcRenderer\.invoke\("export:chooseSaveTarget", \{ count, defaultName \}\)/);
  assert.match(pre, /writeSavedFile: \(filePath, bytes\) => ipcRenderer\.invoke\("export:writeSavedFile", \{ filePath, bytes \}\)/);
});

test('sp-7. app.js: count decides file vs folder; batch counts images × displays; cancel message; save button no longer uses chooseDir', () => {
  const app = read('src/renderer/app.js');
  const fn = app.slice(app.indexOf('async function exportDataset()'), app.indexOf('// canvas แยกสำหรับ export'));
  assert.match(fn, /const count = isBatch \? state\.images\.length \* connected\.length : ready\.length;/);
  assert.match(fn, /window\.mvExport\.chooseSaveTarget\(count, defaultName\)/);
  assert.match(fn, /"ยกเลิกการบันทึกภาพ"/);
  assert.match(fn, /target\.kind === "file"[\s\S]*?window\.mvExport\.writeSavedFile\(target\.path,/);
  assert.match(fn, /if \(target\.kind === "folder"\) dir = target\.path;/);
  assert.match(fn, /await exportBatch\(connected, dir, target && target\.name\);/);
  assert.ok(!/chooseDir/.test(fn), 'Save Picture must not open the folder picker any more');
  // ready ต้องนับก่อนเปิด dialog (ไม่งั้นรู้ไม่ได้ว่าไฟล์เดียวหรือหลายไฟล์)
  assert.ok(fn.indexOf('const ready =') < fn.indexOf('chooseSaveTarget('));
  assert.ok(fn.indexOf('if (exporting)') < fn.indexOf('chooseSaveTarget('), 'busy guard before the dialog');
  assert.match(app, /ลงโฟลเดอร์ "\$\{folderName \|\| dir\}"/);
});

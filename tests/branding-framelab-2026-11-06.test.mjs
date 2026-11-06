import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const read = (relative) => readFileSync(new URL(relative, import.meta.url), 'utf8');
const pkg = JSON.parse(read('../package.json'));
const mac = JSON.parse(read('../electron-builder-mac.json'));
const main = read('../electron/main.js');
const html = read('../src/index.html');
const renderer = read('../src/renderer/app.js');
const sha256 = (relative) => createHash('sha256')
  .update(readFileSync(new URL(relative, import.meta.url))).digest('hex');

test('FrameLab product and shortcut names replace the old branding', () => {
  assert.equal(pkg.build.productName, 'FrameLab');
  assert.equal(pkg.build.nsis.shortcutName, 'FrameLab');
  assert.match(pkg.description, /^FrameLab\b/);
  assert.equal(mac.productName, 'FrameLab');
  assert.equal(mac.dmg.artifactName, 'FrameLab-${version}-mac-${arch}.${ext}');
  assert.match(mac.mac.extendInfo.NSCameraUsageDescription, /^FrameLab ใช้กล้อง/);
});

test('the package identity and update feed remain compatible with existing installs', () => {
  assert.equal(pkg.name, 'machine-vision-app');
  assert.equal(pkg.build.appId, 'com.irish.machinevision');
  assert.equal(mac.appId, 'com.irish.machinevision');
  assert.deepEqual(pkg.build.publish, {
    provider: 'github', owner: 'kandismixer-lang', repo: 'MachineVision', releaseType: 'release',
  });
  assert.equal(mac.publish, null);
});

test('window and document titles use FrameLab by IRiSH', () => {
  assert.match(main, /title:\s*["']FrameLab by IRiSH["']/);
  assert.match(html, /<title>FrameLab by IRiSH<\/title>/);
});

test('the header uses the FrameLab name, IRiSH subtitle and new brand mark', () => {
  assert.match(html, /<h1>FrameLab<\/h1>/);
  assert.match(html, /<span class="subtitle">by IRiSH<\/span>/);
  assert.match(html, /<img src="\.\/assets\/framelab-logo\.png" alt="FrameLab" class="brand-logo"\s*\/>/);
});

test('the loading screen uses the FrameLab mark and its existing text slot', () => {
  assert.match(html, /<img src="\.\/assets\/framelab-logo\.png" alt="FrameLab by IRiSH" class="loading-logo"\s*\/>/);
  assert.match(html, /<p id="loading-text">FrameLab by IRiSH[^<]*<\/p>/);
  // Progress messages keep the name; error messages stay plain so the error text is easy to read.
  for (const assignment of renderer.matchAll(/loadingText\.textContent\s*=\s*([^;]+);/g)) {
    if (/เกิดข้อผิดพลาด/.test(assignment[1])) continue;
    assert.match(assignment[1], /FrameLab by IRiSH/);
  }
});

test('the status bar retains the FrameLab name when the version arrives', () => {
  assert.match(html, /<span class="status-ver" id="status-ver">FrameLab by IRiSH<\/span>/);
  assert.ok(renderer.includes('`FrameLab by IRiSH · v${v}`'));
});

test('userData stays in the exact legacy folder before Electron is ready', () => {
  const pin = /^app\.setPath\(['"]userData['"],\s*path\.join\(app\.getPath\(['"]appData['"]\),\s*['"]Machine Vision By IRiSH['"]\)\);/m.exec(main);
  assert.ok(pin, 'pin userData to the exact old productName at module scope');
  const ready = main.search(/app\.(?:whenReady\s*\(|(?:on|once)\s*\(\s*['"]ready['"])/);
  assert.ok(ready > pin.index, 'set userData before registering app readiness');
});

test('the Windows icon is an exact copy of the generated FrameLab ICO', () => {
  // SHA256 of specs/assets/framelab-icon.ico generated from Pic/Framlab.png on 2026-11-06.
  assert.equal(sha256('../electron/icon.ico'), '8cb280e1405251eac8fae85d75809677145a35063cc94d30ac6d1e09a065a0fc');
});

test('the macOS icon is an exact copy of the generated FrameLab PNG', () => {
  // SHA256 of specs/assets/framelab-icon-1024.png generated from Pic/Framlab.png on 2026-11-06.
  assert.equal(sha256('../build-mac/icon.png'), '343382b96c5aba2d036ce232a425654e83a66a11348bf04d23a0da1e1c62c5f7');
});

test('the old IRiSH images are retained without being referenced by index.html', () => {
  assert.doesNotMatch(html, /irish-[^"'\s]*\.jpg/i);
  assert.ok(readFileSync(new URL('../src/assets/irish-logo.jpg', import.meta.url)).length);
  assert.ok(readFileSync(new URL('../src/assets/irish-mark.jpg', import.meta.url)).length);
});

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

test('Argus product and shortcut names replace the old branding', () => {
  assert.equal(pkg.build.productName, 'Argus');
  assert.equal(pkg.build.nsis.shortcutName, 'Argus');
  assert.match(pkg.description, /^Argus\b/);
  assert.equal(mac.productName, 'Argus');
  assert.equal(mac.dmg.artifactName, 'Argus-${version}-mac-${arch}.${ext}');
  assert.match(mac.mac.extendInfo.NSCameraUsageDescription, /^Argus ใช้กล้อง/);
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

test('window and document titles use Argus by IRiSH', () => {
  assert.match(main, /title:\s*["']Argus by IRiSH["']/);
  assert.match(html, /<title>Argus by IRiSH<\/title>/);
});

test('the header uses the Argus name, IRiSH subtitle and new brand mark', () => {
  assert.match(html, /<h1>Argus<\/h1>/);
  assert.match(html, /<span class="subtitle">by IRiSH<\/span>/);
  assert.match(html, /<img src="\.\/assets\/argus-logo\.png" alt="Argus" class="brand-logo"\s*\/>/);
});

test('the loading screen uses the Argus mark and its existing text slot', () => {
  assert.match(html, /<img src="\.\/assets\/argus-logo\.png" alt="Argus by IRiSH" class="loading-logo"\s*\/>/);
  assert.match(html, /<p id="loading-text">Argus by IRiSH[^<]*<\/p>/);
  // Progress messages keep the name; error messages stay plain so the error text is easy to read.
  for (const assignment of renderer.matchAll(/loadingText\.textContent\s*=\s*([^;]+);/g)) {
    if (/เกิดข้อผิดพลาด/.test(assignment[1])) continue;
    assert.match(assignment[1], /Argus by IRiSH/);
  }
});

test('the status bar retains the Argus name when the version arrives', () => {
  assert.match(html, /<span class="status-ver" id="status-ver">Argus by IRiSH<\/span>/);
  assert.ok(renderer.includes('`Argus by IRiSH · v${v}`'));
});

test('userData stays in the exact legacy folder before Electron is ready', () => {
  const pin = /^app\.setPath\(['"]userData['"],\s*path\.join\(app\.getPath\(['"]appData['"]\),\s*['"]Machine Vision By IRiSH['"]\)\);/m.exec(main);
  assert.ok(pin, 'pin userData to the exact old productName at module scope');
  const ready = main.search(/app\.(?:whenReady\s*\(|(?:on|once)\s*\(\s*['"]ready['"])/);
  assert.ok(ready > pin.index, 'set userData before registering app readiness');
});

test('the Windows icon is an exact copy of the supplied Argus ICO', () => {
  // SHA256 of specs/assets/argus-icon.ico supplied on 2026-10-30.
  assert.equal(sha256('../electron/icon.ico'), '7ae40a7cdfd6ee2ad94d52c731339a44dd408131e9512b86d2ddda608f4670f0');
});

test('the macOS icon is an exact copy of the supplied Argus PNG', () => {
  // SHA256 of specs/assets/argus-icon-1024.png supplied on 2026-10-30.
  assert.equal(sha256('../build-mac/icon.png'), '9e41fdcb6641a439993fe14045f61086cf2fe002b8a62703b5953817e8b0284d');
});

test('the old IRiSH images are retained without being referenced by index.html', () => {
  assert.doesNotMatch(html, /irish-[^"'\s]*\.jpg/i);
  assert.ok(readFileSync(new URL('../src/assets/irish-logo.jpg', import.meta.url)).length);
  assert.ok(readFileSync(new URL('../src/assets/irish-mark.jpg', import.meta.url)).length);
});

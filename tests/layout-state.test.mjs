// สถานะเลย์เอาต์ที่เก็บถาวร (ui/layout-state.js) — รวมฟิลด์ซ่อน/แสดงแผง: persistence, ความเข้ากันได้ย้อนหลัง, clamp
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { KEY, LIMITS, RATIO, RAIL_W, emptyState, load, save } from "../src/renderer/ui/layout-state.js";

function fakeStorage(init) {
  const m = new Map(init === undefined ? [] : [[KEY, init]]);
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => m.set(k, String(v)),
    removeItem: (k) => m.delete(k),
    has: (k) => m.has(k),
    raw: () => m.get(KEY),
  };
}

test("ls-01 ไม่มีค่าที่เก็บ → ค่าเริ่มต้น (ไม่ซ่อนแผง)", () => {
  assert.deepEqual(load(fakeStorage()), emptyState());
  assert.equal(emptyState().paletteHidden, false);
  assert.equal(emptyState().propsHidden, false);
});

test("ls-02 ค่าเก่า (ก่อนมีฟิลด์ซ่อน) ยังโหลดได้ และแสดงแผงทั้งสอง", () => {
  const st = fakeStorage(JSON.stringify({ v: 1, paletteW: 300, propsW: null, stageRatio: 0.5 }));
  assert.deepEqual(load(st), { paletteW: 300, propsW: null, stageRatio: 0.5, paletteHidden: false, propsHidden: false });
});

test("ls-03 ฟิลด์ซ่อน: roundtrip + ความกว้างที่ผู้ใช้ตั้งไว้ยังอยู่ (คืนตอนเปิดแผง)", () => {
  const st = fakeStorage();
  save({ paletteW: 320, propsW: 440, stageRatio: 0.56, paletteHidden: true, propsHidden: false }, st);
  assert.deepEqual(load(st), { paletteW: 320, propsW: 440, stageRatio: 0.56, paletteHidden: true, propsHidden: false });
});

test("ls-04 ซ่อนแผงอย่างเดียว (ไม่เคยลากขนาด) → ยังบันทึก · ทุกอย่างเป็นค่าเริ่มต้น → ลบ key", () => {
  const st = fakeStorage();
  save({ paletteW: null, propsW: null, stageRatio: null, paletteHidden: false, propsHidden: true }, st);
  assert.ok(st.has(KEY));
  assert.equal(load(st).propsHidden, true);
  save(emptyState(), st);
  assert.equal(st.has(KEY), false);
});

test("ls-05 ฟิลด์ซ่อนชนิดผิด = ถือว่าไม่ซ่อน แต่ไม่ทิ้งค่าอื่น · ค่ากว้างผิดชนิด = ทิ้งทั้งก้อน", () => {
  const a = load(fakeStorage(JSON.stringify({ v: 1, paletteW: 250, paletteHidden: "yes", propsHidden: 1 })));
  assert.equal(a.paletteW, 250);
  assert.equal(a.paletteHidden, false);
  assert.equal(a.propsHidden, false);
  const bad = fakeStorage(JSON.stringify({ v: 1, paletteW: "wide", paletteHidden: true }));
  assert.deepEqual(load(bad), emptyState());
  assert.equal(bad.has(KEY), false);
});

test("ls-06 clamp ขอบเขต (ไม่เปลี่ยนจากเดิม) · JSON เสีย/เวอร์ชันอื่น → ค่าเริ่มต้น", () => {
  const s = load(fakeStorage(JSON.stringify({ v: 1, paletteW: 9999, propsW: 1, stageRatio: 5 })));
  assert.equal(s.paletteW, LIMITS.palette.max);
  assert.equal(s.propsW, LIMITS.props.min);
  assert.equal(s.stageRatio, RATIO.max);
  assert.deepEqual(load(fakeStorage("{oops")), emptyState());
  assert.deepEqual(load(fakeStorage(JSON.stringify({ v: 2, paletteHidden: true }))), emptyState());
});

test("ls-07 ไม่มี storage / storage โยน error → เงียบ ใช้ค่าเริ่มต้น", () => {
  const boom = { getItem() { throw new Error("x"); }, setItem() { throw new Error("x"); }, removeItem() { throw new Error("x"); } };
  assert.deepEqual(load(boom), emptyState());
  assert.doesNotThrow(() => save({ ...emptyState(), paletteHidden: true }, boom));
});

test("ls-08 ค่าคงที่ rail ตรงกับ CSS (--rail-w) และมี UI ครบ: rail ×2, ปุ่มซ่อน, รายการเมนู มุมมอง, คีย์ลัดไม่มีค่าเริ่มต้น", () => {
  const css = fs.readFileSync(new URL("../src/styles/app.css", import.meta.url), "utf8");
  assert.match(css, new RegExp(`--rail-w: ${RAIL_W}px`));
  const html = fs.readFileSync(new URL("../src/index.html", import.meta.url), "utf8");
  assert.match(html, /id="palette-rail"[^>]*hidden/);
  assert.match(html, /id="props-rail"[^>]*hidden/);
  assert.match(html, /data-collapse="props"[^>]*aria-label="ซ่อนแผงด้านขวา"/);
  const pal = fs.readFileSync(new URL("../src/renderer/ui/palette-panel.js", import.meta.url), "utf8");
  assert.match(pal, /aria-label", "ซ่อนแถบเครื่องมือ"/);
  const app = fs.readFileSync(new URL("../src/renderer/app.js", import.meta.url), "utf8");
  const km = app.match(/const DEFAULT_KEYMAP = \{([^}]*)\}/)[1];
  assert.ok(!/togglePalette|toggleProps/.test(km), "คีย์ลัดซ่อนแผงต้องไม่มีค่าเริ่มต้น");
});

test("ls-09 เส้นทางที่ผู้ใช้ตั้งใจไปที่แผง (เปิดแผงคลาส / Ctrl+/) เปิดแผงที่ซ่อนก่อน · ปุ่มซ่อนอยู่นอก tablist · แถบแท็บยัง sticky", () => {
  const app = fs.readFileSync(new URL("../src/renderer/app.js", import.meta.url), "utf8");
  assert.match(app, /focusClassPanel: \(\) => \{\s*setPanelCollapsed\("props", false\)/);
  assert.match(app, /palettePanel\.focusSearch/);
  assert.match(app, /setPanelCollapsed\("palette", false\);[^\n]*\n\s*palettePanel\.focusSearch\(\)/);
  const html = fs.readFileSync(new URL("../src/index.html", import.meta.url), "utf8");
  const iAnchor = html.indexOf('class="panel-collapse-anchor"');
  const iTabs = html.indexOf('role="tablist" aria-label="Right panel workspace"');
  assert.ok(iAnchor > 0 && iAnchor < iTabs, "ปุ่มซ่อนต้องเป็นพี่น้องก่อน tablist ไม่ใช่ลูก");
  const tablist = html.slice(iTabs, html.indexOf("</div>", iTabs));
  assert.ok(!tablist.includes("data-collapse"), "tablist ต้องมีแต่ tab");
  const css = fs.readFileSync(new URL("../src/styles/app.css", import.meta.url), "utf8");
  assert.ok(!/\.ui-v2 \.inspector-tabs \{ position: relative; \}/.test(css), "ห้ามทับ position: sticky ของแถบแท็บ");
  assert.match(css, /@container props-panel \(max-width: 330px\)/);
  assert.match(css, /body:not\(\.ui-v2\) \.panel-collapse-btn/);
});

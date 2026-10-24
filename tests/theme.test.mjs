// ธีมสว่าง/มืด (ui/theme.js) — ค่าเริ่มต้น, การเก็บค่า, พาเลตต์กราฟ, ปุ่มสลับ
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

// stub เบราว์เซอร์ขั้นต่ำก่อน import (theme.js อ่าน localStorage/document ตอนโหลดโมดูล)
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
};
const attrs = {};
globalThis.document = { documentElement: { setAttribute: (k, v) => { attrs[k] = v; } } };

const theme = await import("../src/renderer/ui/theme.js");

test("th-01 ค่าเริ่มต้น = light และตั้ง data-theme ตั้งแต่โหลดโมดูล", () => {
  assert.equal(theme.getTheme(), "light");
  assert.equal(attrs["data-theme"], "light");
});

test("th-02 setTheme เก็บลง mv-theme + แจ้ง subscriber เฉพาะตอนเปลี่ยนจริง + ค่าผิดถูกเมิน", () => {
  const seen = [];
  const off = theme.onThemeChange((t) => seen.push(t));
  theme.setTheme("dark");
  theme.setTheme("dark"); // ซ้ำ = ไม่แจ้ง
  theme.setTheme("purple"); // ไม่รู้จัก = เมิน
  assert.deepEqual(seen, ["dark"]);
  assert.equal(store.get("mv-theme"), "dark");
  assert.equal(attrs["data-theme"], "dark");
  off();
  assert.equal(theme.toggleTheme(), "light");
  assert.deepEqual(seen, ["dark"]); // ยกเลิก subscribe แล้ว
});

test("th-03 พาเลตต์กราฟ: dark = ค่าเดิมของ editor.js · สองธีมมีคีย์ครบเท่ากัน · object เดิมทุกครั้ง", () => {
  theme.setTheme("dark", { persist: false });
  const d = theme.graphColors();
  assert.equal(d, theme.graphColors());
  assert.equal(d.canvasBg, "#08111d");
  assert.equal(d.body, "#0f1b2b");
  assert.equal(d.titleBase, "#132235");
  assert.equal(d.focus, "#8cc8ff");
  assert.equal(d.link, "#4a6285");
  assert.equal(d.portOn, "#58b4ff");
  theme.setTheme("light", { persist: false });
  const l = theme.graphColors();
  assert.notEqual(l, d);
  assert.deepEqual(Object.keys(l).sort(), Object.keys(d).sort());
});

test("th-04 index.html ตั้ง data-theme ก่อนวาด (key เดียวกับ theme.js) + มีปุ่มสลับที่ผูกด้วย aria", () => {
  const html = fs.readFileSync(new URL("../src/index.html", import.meta.url), "utf8");
  assert.match(html, /localStorage\.getItem\("mv-theme"\)/);
  assert.match(html, /id="theme-toggle"[^>]*aria-label="สลับธีมสว่าง\/มืด"/);
  assert.ok(html.indexOf("mv-theme") < html.indexOf("app.css"), "สคริปต์ธีมต้องมาก่อน stylesheet");
});

test("th-05 CSS: มีชุด token light และทุก --th-* ที่ใช้มีค่าทั้ง dark/light", () => {
  const css = fs.readFileSync(new URL("../src/styles/app.css", import.meta.url), "utf8");
  assert.match(css, /:root\[data-theme="light"\]\s*\{/);
  const used = new Set([...css.matchAll(/var\((--th-[a-z0-9-]+)\)/g)].map((m) => m[1]));
  assert.ok(used.size > 50);
  const cssNoLight = css.replace(/:root\[data-theme="light"\]\s*\{[^}]*\}/g, "");
  const dark = new Set([...cssNoLight.matchAll(/^  (--th-[a-z0-9-]+):/gm)].map((m) => m[1]));
  for (const u of used) assert.ok(dark.has(u), "ไม่มีค่า dark: " + u);
  const lightBlocks = [...css.matchAll(/:root\[data-theme="light"\]\s*\{([^}]*)\}/g)].map((m) => m[1]).join("\n");
  for (const u of used) assert.ok(lightBlocks.includes(u + ":"), "ไม่มีค่า light: " + u);
});

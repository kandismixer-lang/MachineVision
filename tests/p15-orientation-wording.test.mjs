// P-15 — คำอธิบายมุมเอียง (orientation_deg) ของ regionprops ต้องตรงกับที่แอปส่งออกจริง (ไม่แตะค่าที่คำนวณ)
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { test } from 'node:test';
import cv from '@techstark/opencv-js';
import { OPS } from '../src/renderer/filters/registry.js';

await new Promise((resolve) => {
  const poll = setInterval(() => { if (cv && typeof cv.Mat === 'function' && typeof cv.cvtColor === 'function') { clearInterval(poll); resolve(); } }, 50);
});

const rows = (draw) => {
  const m = cv.Mat.zeros(120, 120, cv.CV_8UC1);
  draw(m);
  const ctx = {};
  const out = OPS.regionprops.apply(cv, m, { minArea: 0, maxObjects: 500, mmPerPx: 0 }, ctx);
  out.delete(); m.delete();
  return ctx.values.rows;
};

test('p15-1. สี่เหลี่ยมเอียง 45° พอดี (4 จุด) ได้ orientation = null (ไม่ใช่ 0) · วงกลมได้ตัวเลข แต่ความรี = 0', () => {
  const diamond = rows((m) => {
    const p = cv.matFromArray(4, 1, cv.CV_32SC2, [60, 20, 100, 60, 60, 100, 20, 60]);
    const v = new cv.MatVector(); v.push_back(p);
    cv.fillPoly(m, v, new cv.Scalar(255)); v.delete(); p.delete();
  });
  assert.equal(diamond.length, 1);
  assert.equal(diamond[0][10], null);
  const circle = rows((m) => cv.circle(m, new cv.Point(60, 60), 30, new cv.Scalar(255), -1));
  assert.equal(typeof circle[0][10], 'number');
  assert.equal(circle[0][17], 0, 'eccentricity of a circle');
});

test('p15-2. desc ของ regionprops และ tip ของ orientation_deg บอกครบ: เครื่องหมาย — / Python "-" / 4 จุด / 45° / กลม', () => {
  const d = OPS.regionprops.desc;
  assert.ok(/จะไม่มีค่ามุม — ในตารางแสดงเป็นเครื่องหมาย — \(ไม่ใช่ 0°\)/.test(d) && /Python ที่ export แสดงเป็น -/.test(d));
  assert.ok(/เอียง 45° พอดี/.test(d) && /4 จุดมุม/.test(d) && /กลมหรือเกือบกลม/.test(d) && /ดูความรีก่อนเชื่อมุม/.test(d));
  const src = fs.readFileSync(new URL('../src/renderer/ui/detail-panel.js', import.meta.url), 'utf8');
  const line = src.split('\n').find((l) => l.includes('orientation_deg:{'));
  assert.ok(/ไม่มีค่ามุม/.test(line) && /น้อยกว่า 5 จุด/.test(line) && /เกือบกลม/.test(line));
});

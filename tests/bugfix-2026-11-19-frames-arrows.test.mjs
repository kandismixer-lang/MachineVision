// 2026-11-19 bugs found while labeling a conveyor video in FrameLab:
// (1) video split: typing "120 frames" showed/extracted 121 (floor(dur/iv)+1 and float accumulation)
// (2) Label mode: ←/→ moved two images (canvas handler + global shortcut both fired)
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { frameCount, frameTimes } from "../src/renderer/ui/video-frames-dialog.js";

test("typing N frames gives exactly N (exact division)", () => {
  for (const dur of [44.8, 44.801, 10, 85.98543333333333, 3.3333333]) {
    for (const n of [1, 7, 30, 99, 120, 500]) {
      assert.equal(frameCount(dur, dur / n), n, `dur=${dur} n=${n}`);
      assert.equal(frameTimes(dur, dur / n).length, n);
    }
  }
});

test("non-divisible interval counts times strictly before the end", () => {
  assert.equal(frameCount(44.8, 0.374), 120); // 0 .. 119*0.374 = 44.506 < 44.8
  assert.equal(frameCount(10, 3), 4); // 0,3,6,9
  assert.equal(frameCount(10, 10), 1); // only t=0
  assert.equal(frameCount(10, 20), 1); // interval longer than the clip
  assert.equal(frameCount(0, 1), 0);
  assert.equal(frameCount(10, 0), 0);
});

test("frame times are i*interval, all < duration, capped by max", () => {
  const t = frameTimes(44.8, 0.374);
  assert.equal(t[0], 0);
  assert.equal(t[119], 119 * 0.374);
  assert.ok(t.every((x) => x < 44.8));
  assert.equal(frameTimes(44.8, 1 / 60, 1000).length, 1000);
  assert.equal(frameTimes(44.8, 0.374, 50).length, 50);
});

test("Label mode arrow keys stop propagation to the global shortcut", () => {
  const src = fs.readFileSync(new URL("../src/renderer/ui/annotate-canvas.js", import.meta.url), "utf8");
  const i = src.indexOf('(e.key === "ArrowLeft" || e.key === "ArrowRight") && !polyDraft');
  assert.ok(i > 0);
  const branch = src.slice(i, src.indexOf("return;", i));
  assert.match(branch, /e\.stopPropagation\(\)/);
  assert.ok(branch.indexOf("stopPropagation") < branch.indexOf("onNavImage("), "stop before navigating");
});

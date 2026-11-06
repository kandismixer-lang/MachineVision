// ส่วนรันท้ายไฟล์ของโค้ด Python ตามแหล่งภาพในแอป (2026-11-06)
// บั๊กเดิม: __main__ บังคับอ่าน "input.jpg" เสมอ ใช้กับกล้องไม่ได้ · ตอนนี้แอปส่ง opts.source
// (ภาพ = path จริง, กล้อง = loop กล้อง, วิดีโอ = loop ไฟล์) · ไม่ส่ง = แม่แบบเดิมทุกไบต์
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { FILTERS, defaultParams } from '../src/renderer/filters/registry.js';
import { buildPythonCode, normalizeSource5S } from '../src/renderer/codegen.js';

let python = false;
try { execFileSync('python', ['-c', 'import cv2,numpy'], { stdio: 'pipe' }); python = true; } catch {}

const step = (id, params = {}) => ({ id, params: { ...defaultParams(id), ...params }, enabled: true, nodeId: 1 });
const twoId = Object.keys(FILTERS).find((id) => FILTERS[id].inputs === 2);
const singles = [[], [step('blur')], [step('color_space'), step('blur')]];
const pairs = [[{ ...step(twoId), src2: { pipeline: [step('blur')] } }]];

// รันโค้ดที่สร้างใน Python จริง: exec ด้วย __name__ อื่น (ไม่เข้า main เอง) แล้วรัน script ทดสอบต่อท้าย
function runPy(code, script) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cg-src-'));
  try {
    fs.writeFileSync(path.join(dir, 'gen.py'), code, 'utf8');
    fs.writeFileSync(path.join(dir, 'drv.py'), `import sys\nns = {'__name__': 'gen'}\nexec(compile(open(sys.argv[1], encoding='utf8').read(), sys.argv[1], 'exec'), ns)\n${script}\n`, 'utf8');
    return execFileSync('python', [path.join(dir, 'drv.py'), path.join(dir, 'gen.py')], { cwd: dir, encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf8' } });
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}
const bodyOf = (code, head) => {
  const start = code.indexOf(head);
  assert.ok(start >= 0, `missing ${head}`);
  return code.slice(start + head.length, code.indexOf('\n    return img\n', start));
};

test('cs-1 ไม่ส่ง source → แม่แบบเดิมทุกไบต์ (ภาพเดียวและสองภาพ)', () => {
  for (const p of [...singles, ...pairs]) {
    const legacy = buildPythonCode(p);
    assert.equal(buildPythonCode(p, {}), legacy);
    assert.equal(buildPythonCode(p, { source: null }), legacy);
    assert.match(legacy, /if __name__ == "__main__":\n    result = process\("input/);
    assert.doesNotMatch(legacy, /import os|process_image|INPUT_PATHS/);
  }
});

test('cs-2 ส่ง source → ขั้นตอนประมวลผลเหมือนเดิมทุกบรรทัด และ process(path) ยังอยู่', () => {
  const kinds = [{ kind: 'image', paths: ['a.png'] }, { kind: 'camera' }, { kind: 'video', path: 'v.mp4' }];
  for (const p of singles) {
    const legacy = bodyOf(buildPythonCode(p), 'raise FileNotFoundError(path)\n');
    for (const source of kinds) {
      const code = buildPythonCode(p, { source });
      assert.equal(bodyOf(code, 'ตามขั้นตอนในโปรแกรม\n'), legacy);
      assert.match(code, /def process\(path\):\n    img = cv2\.imread\(path\)  # อ่านภาพ \(BGR\)\n    if img is None:\n        raise FileNotFoundError\(path\)\n    return process_image\(img\)\n/);
      assert.match(code, /^import os$/m);
      assert.match(code, /if __name__ == "__main__":\n    main\(\)\n$/);
    }
  }
  for (const p of pairs) {
    const legacy = bodyOf(buildPythonCode(p), '(ไม่ต่อสาย B ของขั้นก่อนหน้า)\n');
    const code = buildPythonCode(p, { source: { kind: 'camera' } });
    assert.equal(bodyOf(code, '(ไม่ต่อสาย B ของขั้นก่อนหน้า)\n'), legacy);
    assert.match(code, /def process\(path_a, path_b\):[\s\S]*return process_images\(img, img_b\)/);
    assert.match(code, /result = process_images\(img, img\.copy\(\)\)/);
  }
});

test('cs-3 ภาพนิ่ง: ใส่ path จริงทุกภาพ (ข้ามภาพที่ไม่มีไฟล์) · ไม่มี path เลย → ตัวอย่าง input.jpg', () => {
  const winPath = 'C:\\Users\\me\\Desktop\\งาน ทดสอบ\\ชิ้น "A".png';
  const code = buildPythonCode(singles[1], { source: { kind: 'image', maxSide: 720, paths: [winPath, null, 'D:\\b.jpg'] } });
  assert.ok(code.includes(`    ${JSON.stringify(winPath)},`));
  assert.ok(code.includes('    "D:\\\\b.jpg",'));
  assert.match(code, /# ภาพที่เปิดอยู่ในโปรแกรม \(2 ภาพ\)\n# ไม่ได้ใส่อีก 1 ภาพ/);
  assert.match(code, /^MAX_SIDE = 720 /m);
  const none = buildPythonCode(singles[1], { source: { kind: 'image', paths: [null] } });
  assert.match(none, /"input\.jpg",   # << ใส่ path ภาพของคุณ/);
  assert.match(none, /^MAX_SIDE = 0 /m);
});

test('cs-4 กล้อง: เลขกล้อง/ขนาด/ชื่อ · ชื่อกล้องขึ้นบรรทัดใหม่ไม่หลุดออกจากคอมเมนต์', () => {
  const code = buildPythonCode(singles[1], { source: { kind: 'camera', index: 1, total: 2, label: 'USB Cam\nimport os; os.system("x")', width: 640, height: 480, maxSide: 480 } });
  assert.match(code, /^CAMERA_INDEX = 1   # กล้อง"USB Cam import os; os\.system\("x"\)"ในโปรแกรม \(ตัวที่ 2 จาก 2 ตัว\)/m);
  assert.match(code, /^CAMERA_WIDTH = 640 /m);
  assert.match(code, /^CAMERA_HEIGHT = 480$/m);
  assert.doesNotMatch(code, /^import os; os\.system/m);
  const s = normalizeSource5S({ kind: 'camera', index: -3, width: 'x', height: 1e9 });
  assert.deepEqual([s.index, s.width, s.height], [0, 1280, 16384]);
});

test('cs-5 วิดีโอ: path มี \\ และ " เป็นสตริง Python ถูกต้อง · ไม่มี path → input.mp4', () => {
  const p = 'C:\\v\\คลิป "1".mp4';
  const code = buildPythonCode(singles[1], { source: { kind: 'video', path: p, name: 'x' } });
  assert.ok(code.includes(`VIDEO_PATH = ${JSON.stringify(p)}   # วิดีโอที่เปิดในโปรแกรม`));
  const none = buildPythonCode(singles[1], { source: { kind: 'video', path: null, name: 'clip.mp4' } });
  assert.match(none, /VIDEO_PATH = "input\.mp4"   # << ใส่ path วิดีโอของคุณ \(โปรแกรมอ่าน path ของ "clip\.mp4" ไม่ได้\)/);
});

test('cs-6 Python จริง: ภาพนิ่ง path ภาษาไทย+เว้นวรรค → main() อ่าน ย่อ ประมวลผล และบันทึกผล', { skip: !python }, () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ภาพ ทดสอบ '));
  try {
    const img = path.join(dir, 'ชิ้นงาน 1.png');
    execFileSync('python', ['-c', 'import sys,cv2,numpy as np\nim=np.zeros((60,100,3),np.uint8); im[:, 50:]=200\nok,b=cv2.imencode(".png",im); b.tofile(sys.argv[1])', img], { stdio: 'pipe' });
    const code = buildPythonCode([step('color_space')], { source: { kind: 'image', maxSide: 50, paths: [img] } });
    const out = runPy(code, [
      "import cv2, os",
      "cv2.imshow = lambda *a: None",
      "cv2.waitKey = lambda *a: -1",
      "cv2.destroyAllWindows = lambda: None",
      "ns['main']()",
      "r = ns['read_image'](os.path.join(ns['OUTPUT_DIR'], 'ชิ้นงาน 1_result.png'))",
      "print('OUT', r.shape[1], r.shape[0])",
    ].join('\n'));
    assert.match(out, /OUT 50 30/);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('cs-7 Python จริง: กล้อง/วิดีโอ loop เรียก process ทุกเฟรมจนอ่านไม่ได้ แล้วปล่อยกล้อง', { skip: !python }, () => {
  const fake = [
    "import cv2, numpy as np",
    "class Cap:",
    "    def __init__(self, *a): self.n = 0; self.released = False",
    "    def isOpened(self): return True",
    "    def set(self, *a): return True",
    "    def get(self, *a): return 25.0",
    "    def read(self):",
    "        self.n += 1",
    "        return (self.n <= 3, np.full((40, 80, 3), self.n * 50, np.uint8))",
    "    def release(self): self.released = True",
    "caps = []",
    "cv2.VideoCapture = lambda *a: caps.append(Cap()) or caps[-1]",
    "shown = []",
    "cv2.imshow = lambda name, im: shown.append((name, im.shape))",
    "cv2.waitKey = lambda *a: -1",
    "cv2.destroyAllWindows = lambda: None",
    "ns['main']()",
    "print('FRAMES', len([s for s in shown if s[0] == 'result']), 'RELEASED', caps[-1].released, 'SHAPE', [s[1] for s in shown if s[0] == 'result'][0])",
  ].join('\n');
  for (const source of [{ kind: 'camera', maxSide: 0 }, { kind: 'video', path: 'x.mp4' }]) {
    for (const p of [[step('color_space')], ...pairs]) {
      const out = runPy(buildPythonCode(p, { source }), fake);
      assert.match(out, /FRAMES 3 RELEASED True/, `${source.kind} ${p.length}`);
    }
  }
});

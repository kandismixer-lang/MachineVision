import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
// Isolate the code generator from app startup/DOM; generator functions run unchanged.
const source = (await readFile(new URL('../src/renderer/ui/export-dialog.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\r?$/gm, '');
const { buildNotebook, buildTrainingCode } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const opts = { epochs: 50, batch: 4, imgsz: 640, model: 'yolo11n', rfModel: 'RFDETRNano' };
for (const framework of ['yolo', 'rfdetr']) {
  test(`${framework}: preserve manual ZIP cell and defined training path`, () => {
    const notebook = JSON.parse(buildNotebook({ ...opts, framework, seg: true }));
    const code = notebook.cells.map(cell => cell.source.join('')).join('\n');
    assert.ok(code.includes('ZIP = ""\n!unzip -qo "{ZIP}" -d /content/dataset'));
    assert.doesNotMatch(code, /files.upload|DATA_YAML|DATASET_DIR/);
    if (framework === 'yolo') {
      assert.ok(code.includes('YOLO("yolo11n.pt")'));
      assert.ok(code.includes('data= "/content/dataset/data.yaml"'));
      assert.doesNotMatch(code, /-seg\.pt/);
    } else {
      assert.ok(code.includes('dataset_dir= "/content/dataset"'));
      assert.ok(code.includes('rfdetr[train,loggers]==1.8.2'));
    }
  });
}

// เลือกอุปกรณ์ตอนรัน (2026-11-20): เดิมบังคับ device = 0 → ลืมเปิด GPU บน Colab แล้วพัง และเลือก CPU ไม่ได้
let python = false;
try { (await import('node:child_process')).execFileSync('python', ['-c', 'print(1)'], { stdio: 'pipe' }); python = true; } catch {}
const cellsOf = (framework) => JSON.parse(buildNotebook({ ...opts, framework })).cells.map((c) => c.source.join(''));

test('yolo: DEVICE dropdown (auto/gpu/cpu) feeds model.train, no hard-coded device = 0', () => {
  const train = cellsOf('yolo').find((c) => c.includes('model.train('));
  assert.ok(train.includes('DEVICE = "auto"  #@param ["auto", "gpu", "cpu"]'));
  assert.match(train, /device = device\n\)/);
  assert.doesNotMatch(train, /device = 0\b/);
  assert.ok(train.indexOf('DEVICE = "auto"') < train.indexOf('model = YOLO('));
});

test('rfdetr: DEVICE dropdown hides the GPU before torch/rfdetr import when "cpu"', () => {
  const train = cellsOf('rfdetr').find((c) => c.includes('model.train('));
  assert.ok(train.includes('DEVICE = "auto"  #@param ["auto", "gpu", "cpu"]'));
  const hide = train.indexOf('os.environ["CUDA_VISIBLE_DEVICES"] = ""');
  assert.ok(hide > 0 && hide < train.indexOf('import torch') && hide < train.indexOf('from rfdetr import'));
});

test('yolo: the device-selection lines pick 0 / "cpu" / raise exactly as documented (real Python, stub torch)', { skip: !python }, async () => {
  const { execFileSync } = await import('node:child_process');
  const train = cellsOf('yolo').find((c) => c.includes('model.train('));
  const sel = train.slice(0, train.indexOf('model = YOLO('));
  const run = (choice, cuda) => {
    const script = [
      'import sys, types',
      `t = types.ModuleType("torch"); t.cuda = types.SimpleNamespace(is_available=lambda: ${cuda ? 'True' : 'False'}, get_device_name=lambda i: "Tesla T4")`,
      'sys.modules["torch"] = t',
      'sys.stdout.reconfigure(encoding="utf-8")',
      'try:',
      ...sel.replace(/^DEVICE = "auto"/m, `DEVICE = "${choice}"`).split('\n').map((l) => '    ' + l),
      '    print("RESULT", repr(device))',
      'except RuntimeError as e:',
      '    print("RAISED")',
    ].join('\n');
    return execFileSync('python', ['-c', script], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf8' } }).trim().split('\n').pop();
  };
  assert.equal(run('auto', true), 'RESULT 0');
  assert.equal(run('auto', false), "RESULT 'cpu'");
  assert.equal(run('gpu', true), 'RESULT 0');
  assert.equal(run('gpu', false), 'RAISED');
  assert.equal(run('cpu', true), "RESULT 'cpu'");
});

test('device picked in the Export Code dialog becomes the DEVICE default (both frameworks, notebook + copy code)', () => {
  for (const framework of ['yolo', 'rfdetr']) {
    for (const [given, want] of [['cpu', 'cpu'], ['gpu', 'gpu'], ['auto', 'auto'], [undefined, 'auto'], ['0', 'auto'], ['GPU"; import os', 'auto']]) {
      const nb = JSON.parse(buildNotebook({ ...opts, framework, device: given })).cells.map((c) => c.source.join('')).join('\n');
      const copy = buildTrainingCode({ ...opts, framework, device: given });
      for (const code of [nb, copy]) {
        const m = code.match(/^DEVICE = "([^"]*)"  #@param \["auto", "gpu", "cpu"\]$/m);
        assert.ok(m, `${framework} ${given}: no DEVICE line`);
        assert.equal(m[1], want, `${framework} ${given}`);
      }
    }
  }
});

test('Export Code dialog: Auto / GPU / CPU cards, Device summary row, device passed to the notebook options', async () => {
  const src = await readFile(new URL('../src/renderer/ui/export-dialog.js', import.meta.url), 'utf8');
  assert.match(src, /\{ v: "auto", label: "Auto"[^}]*\},\s*\{ v: "gpu", label: "GPU"[^}]*\},\s*\{ v: "cpu", label: "CPU"/);
  assert.match(src, /\["device", "⚙", "Device"\]/);
  assert.match(src, /const opts = \{[^}]*\bdevice\b[^}]*\};/);
});

test('copy-paste code installs its framework before importing it', () => {
  for (const framework of ['yolo', 'rfdetr']) {
    const code = buildTrainingCode({ ...opts, framework });
    const install = code.indexOf('!pip install');
    assert.ok(install >= 0, `${framework}: missing pip install`);
    const firstImport = code.search(/^(import|from) (ultralytics|rfdetr)/m);
    assert.ok(firstImport > install, `${framework}: imports before installing`);
  }
});

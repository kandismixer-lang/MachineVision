import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
// Isolate the code generator from app startup/DOM; generator functions run unchanged.
const source = (await readFile(new URL('../src/renderer/ui/export-dialog.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\r?$/gm, '');
const { buildNotebook } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
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

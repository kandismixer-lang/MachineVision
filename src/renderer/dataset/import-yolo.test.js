import test from "node:test";
import assert from "node:assert/strict";
import JSZip from "jszip";
import { importYoloDataset, parseYoloNames } from "./import-yolo.js";

async function archive(files) {
  const zip = new JSZip();
  for (const [path, content] of Object.entries(files)) zip.file(path, content);
  return zip.generateAsync({ type: "uint8array" });
}

const fixture = (label = "0 0.5 0.5 0.2 0.4") => ({
  "data.yaml": "names: [cat, dog]\nnc: 2\ntrain: ../train/images\nval: ../valid/images\nroboflow:\n  workspace: example",
  "train/images/a.jpg": new Uint8Array([1, 2, 3]),
  "train/labels/a.txt": label,
});

test("Roboflow boxes: File input and exact integration contract", async () => {
  const zip = new File([await archive(fixture())], "dataset.zip");
  const result = await importYoloDataset(zip);
  assert.deepEqual(Object.keys(result).sort(), ["classNames", "files", "images", "warnings"]);
  assert.deepEqual(result.classNames, ["cat", "dog"]);
  assert.deepEqual(result.warnings, []);
  const im = result.images[0];
  assert.equal(im.file, result.files[0]);
  assert.equal(im.key, im.file.name);
  assert.equal(im.sourcePath, "train/images/a.jpg");
  assert.equal(im.split, "train");
  assert.equal(im.file.type, "image/jpeg");
  assert.deepEqual([...new Uint8Array(await im.file.arrayBuffer())], [1, 2, 3]);
  assert.deepEqual(im.annotations, [{ type: "bbox", classIdx: 0, cx: 0.5, cy: 0.5, w: 0.2, h: 0.4 }]);
});

test("wrapper, images-first layout, nested paths, val alias, polygons and boxes", async () => {
  const result = await importYoloDataset(await archive({
    "export/data.yaml": "names:\n  1: dog\n  0: cat\nnc: 2",
    "export/images/val/nested/a.PNG": "pixels",
    "export/labels/val/nested/a.txt": "\uFEFF1 0 0 1 0 1 1\r\n\r\n0 5e-1 .5 .2 .2\n",
    "export/images/test/b.webp": "pixels",
    "export/labels/test/b.txt": "",
  }));
  const im = result.images.find((i) => i.split === "valid");
  assert.deepEqual(result.classNames, ["cat", "dog"]);
  assert.equal(im.file.type, "image/png");
  assert.deepEqual(im.annotations[0], { type: "polygon", classIdx: 1, points: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }] });
  assert.equal(im.annotations[1].type, "bbox");
  assert.equal(result.images.find((i) => i.split === "test").annotations.length, 0);
  assert.equal(result.warnings.length, 0);
});

test("missing labels are backgrounds; names remain unique across splits and case", async () => {
  const result = await importYoloDataset(await archive({
    "data.yaml": "names: [cat]",
    "train/images/a.jpg": "one",
    "valid/images/A.jpg": "two",
    "test/images/a_2.jpg": "three",
    "train/labels/a.txt": "",
  }));
  assert.equal(new Set(result.files.map((f) => f.name.toLowerCase())).size, 3);
  assert.equal(result.warnings.length, 2);
  assert.ok(result.images.every((i) => i.key === i.file.name && i.annotations.length === 0));
});

for (const [yaml, expected] of [
  ["names: ['a,b', 'Bob''s cat', \"hash # and colon: cat\", plain#hash] # comment", ["a,b", "Bob's cat", "hash # and colon: cat", "plain#hash"]],
  ['names: {1: "dog", 0: \'cat\',}', ["cat", "dog"]],
  ['names: ["line\\tbreak"]', null],
  ["names:\n- cat\n- dog's toy\ntrain: ../train/images", ["cat", "dog's toy"]],
  ["\uFEFFnames:\r\n  - cat # a comment\r\n  - 'dog: big'\r\nnc: 2", ["cat", "dog: big"]],
  ['names: ["say \\"hi\\"", "\\u732b"]', ['say "hi"', "猫"]],
]) {
  test(`restricted YAML: ${JSON.stringify(yaml)}`, () => {
    if (expected) assert.deepEqual(parseYoloNames(yaml), expected);
    else assert.throws(() => parseYoloNames(yaml), /data.yaml/);
  });
}

for (const yaml of [
  "train: train/images", "names: []", "names: {}", "names: [cat]\nnames: [dog]",
  "names: {1: cat}", "names: {0: cat, 0: dog}", "names: [cat, cat]",
  "names: [cat]\nnc: 2", "names: [cat]\nnc: 1\nnc: 1",
  "names: [cat,,dog]", "names: ['unclosed]", "names: [*alias]",
  "names: &anchor [cat]", "names:\n  - !tag cat", "names:\n  - |\n    cat",
  "names:\n  0: cat\n    1: dog", "names:\n  - cat\n  1: dog",
  "names: [cat]\n  - dog", "names: [cat,\n dog]", "names:\n\t- cat",
]) {
  test(`reject unsupported/invalid YAML: ${JSON.stringify(yaml)}`, () => assert.throws(() => parseYoloNames(yaml), /import-yolo: data.yaml:/));
}

for (const [label, error] of [
  ["-1 .5 .5 .2 .2", /class ID/], ["0.5 .5 .5 .2 .2", /class ID/],
  ["2 .5 .5 .2 .2", /outside data.yaml/], ["0 .5 .5 .2", /expected class/],
  ["0 .5 .5 .2 .2 0", /expected class/], ["0 NaN .5 .2 .2", /invalid numeric/],
  ["0 Infinity .5 .2 .2", /invalid numeric/], ["0 0x1 .5 .2 .2", /invalid numeric/],
  ["0 1e999 .5 .2 .2", /normalized/], ["0 -.1 .5 .2 .2", /normalized/],
  ["0 .5 .5 0 .2", /positive/], ["0 .9 .5 .4 .2", /outside the image/],
  ["0 0 0 .5 .5 1 1", /nonzero area/], ["0 0 0 1 0 1 2", /normalized/],
]) {
  test(`reject invalid label: ${label}`, async () => {
    await assert.rejects(importYoloDataset(await archive(fixture(`\n${label}`))), (err) => {
      assert.match(err.message, /train\/labels\/a.txt:2:/);
      assert.match(err.message, error);
      return true;
    });
  });
}

for (const path of ["../escape.txt", "/absolute.txt", "C:/drive.txt", "train\\images\\a.jpg", "nested/../escape.txt", "./relative.txt"]) {
  test(`reject unsafe ZIP path even after JSZip sanitization: ${path}`, async () => {
    await assert.rejects(importYoloDataset(await archive({ ...fixture(), [path]: "bad" })), /unsafe ZIP path/);
  });
}

test("reject ZIP symlinks", async () => {
  const zip = new JSZip();
  for (const [path, data] of Object.entries(fixture())) zip.file(path, data);
  zip.file("link", "target", { unixPermissions: 0o120777 });
  await assert.rejects(importYoloDataset(await zip.generateAsync({ type: "uint8array", platform: "UNIX" })), /symbolic links/);
});

test("reject malformed archives, missing/multiple manifests, no images and orphan labels", async () => {
  await assert.rejects(importYoloDataset(new Uint8Array([1, 2, 3])), /cannot read ZIP/);
  for (const [files, error] of [
    [{ "a.jpg": "image" }, /exactly one data.yaml/],
    [{ ...fixture(), "other/data.yaml": "names: [a]" }, /exactly one data.yaml/],
    [{ "data.yaml": "names: [a]" }, /no supported images/],
    [{ ...fixture(), "valid/labels/missing.txt": "" }, /orphan label/],
    [{ ...fixture(), "train/images/a.png": "image" }, /ambiguous images/],
    [{ ...fixture(), "labels/train/a.txt": "" }, /orphan label/],
  ]) await assert.rejects(importYoloDataset(await archive(files)), error);
});

test("a later invalid label rejects the whole import without touching app globals", async () => {
  const files = { ...fixture(), "valid/images/b.jpg": "image", "valid/labels/b.txt": "bad label" };
  await assert.rejects(importYoloDataset(await archive(files)), /valid\/labels\/b.txt:1/);
  const result = await importYoloDataset(await archive(fixture()));
  assert.equal(result.images.length, 1);
});

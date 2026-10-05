import JSZip from "jszip";

const fail = (message) => { throw new Error(`import-yolo: ${message}`); };
const MIME = { jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", webp: "image/webp", bmp: "image/bmp", tif: "image/tiff", tiff: "image/tiff" };

// A deliberately restricted YAML reader, not a general YAML implementation.
// Supports one-line flow lists/maps and block lists/numeric maps, quoted or
// plain names, comments, doubled single quotes and JSON double-quote escapes.
// Anchors, tags, aliases, multiline scalars/flow collections are rejected.
function uncomment(line) {
  let quote = null;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quote) {
      if (quote === '"' && c === "\\") { i++; continue; }
      if (c === quote) {
        if (quote === "'" && line[i + 1] === "'") { i++; continue; }
        quote = null;
      }
    } else if ((c === '"' || c === "'") && (i === 0 || /[\s[:,{]/.test(line[i - 1]))) quote = c;
    else if (c === "#" && (i === 0 || /\s/.test(line[i - 1]))) return line.slice(0, i).trimEnd();
  }
  if (quote) fail("data.yaml: unterminated quote or unsupported multiline string");
  return line.trimEnd();
}

function scalar(raw) {
  const s = raw.trim();
  let value;
  if (s.startsWith('"')) {
    try { value = JSON.parse(s); } catch { fail(`data.yaml: invalid quoted name ${s}`); }
  } else if (s.startsWith("'")) {
    if (!/^'(?:[^']|'')*'$/.test(s)) fail(`data.yaml: invalid quoted name ${s}`);
    value = s.slice(1, -1).replace(/''/g, "'");
  } else {
    if (!s || /^[!&*|>@`%\[\]{}?,\-]/.test(s) || /:\s|[\[\]{},]/.test(s)) {
      fail(`data.yaml: unsupported name syntax ${s || "(empty)"}; quote the name`);
    }
    value = s;
  }
  if (typeof value !== "string" || !value.trim() || /[\x00-\x1f\x7f]/.test(value)) fail("data.yaml: names must be nonempty single-line strings");
  return value;
}

function flowItems(text) {
  const items = [];
  let start = 0, quote = null;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quote) {
      if (quote === '"' && c === "\\") { i++; continue; }
      if (c === quote) {
        if (quote === "'" && text[i + 1] === "'") { i++; continue; }
        quote = null;
      }
    } else if ((c === '"' || c === "'") && (i === start || /^[\s\d:]*$/.test(text.slice(start, i)))) quote = c;
    else if (c === ",") { items.push(text.slice(start, i).trim()); start = i + 1; }
  }
  if (quote) fail("data.yaml: unterminated quoted name");
  const tail = text.slice(start).trim();
  if (tail) items.push(tail);
  return items;
}

export function parseYoloNames(yaml) {
  const lines = yaml.replace(/^\uFEFF/, "").split(/\r?\n/).map(uncomment);
  if (lines.some((line) => /^\s*(?:---|\.\.\.)\s*$/.test(line))) fail("data.yaml: YAML document markers are unsupported");
  const headers = lines.map((line, i) => /^names\s*:/.test(line) ? i : -1).filter((i) => i >= 0);
  if (headers.length !== 1) fail("data.yaml: expected exactly one top-level names field");
  const index = headers[0];
  const inline = lines[index].replace(/^names\s*:/, "").trim();
  let items, mapping;
  if (inline) {
    mapping = inline.startsWith("{");
    if (!(mapping ? inline.endsWith("}") : inline.startsWith("[") && inline.endsWith("]"))) fail("data.yaml: names must be a list or numeric map on one line, or a block collection");
    items = flowItems(inline.slice(1, -1));
    const next = lines.slice(index + 1).find((line) => line.trim());
    if (next && /^\s/.test(next)) fail("data.yaml: unexpected continuation after names");
  } else {
    const block = [];
    let indent;
    for (const line of lines.slice(index + 1)) {
      if (!line.trim()) continue;
      if (!/^\s/.test(line) && !/^-\s/.test(line)) break;
      if (/^\s*\t/.test(line)) fail("data.yaml: tabs in names indentation are unsupported");
      const n = line.length - line.trimStart().length;
      if (indent === undefined) indent = n;
      if (n !== indent) fail("data.yaml: inconsistent names indentation or nested names");
      block.push(line.trim());
    }
    mapping = !block[0]?.startsWith("-");
    items = block;
  }
  let names;
  if (mapping) {
    const entries = new Map();
    for (const item of items) {
      const match = /^(0|[1-9]\d*)\s*:\s*(.+)$/.exec(item);
      if (!match || !Number.isSafeInteger(Number(match[1]))) fail(`data.yaml: invalid numeric class map entry ${item}`);
      const id = Number(match[1]);
      if (entries.has(id)) fail(`data.yaml: duplicate class ID ${id}`);
      entries.set(id, scalar(match[2]));
    }
    names = Array.from({ length: entries.size }, (_, id) => {
      if (!entries.has(id)) fail("data.yaml: class IDs must be contiguous starting at 0");
      return entries.get(id);
    });
  } else {
    names = items.map((item) => {
      if (!inline && !/^-\s+/.test(item)) fail("data.yaml: mixed list/map names");
      return scalar(inline ? item : item.replace(/^-\s+/, ""));
    });
  }
  if (!names.length) fail("data.yaml: names cannot be empty");
  if (new Set(names).size !== names.length) fail("data.yaml: duplicate class names");
  const counts = lines.filter((line) => /^nc\s*:/.test(line));
  if (counts.length > 1 || (counts.length && !new RegExp(`^nc\\s*:\\s*${names.length}\\s*$`).test(counts[0]))) fail("data.yaml: nc must match the number of names");
  return names;
}

function annotations(text, path, classCount) {
  const result = [];
  for (const [i, raw] of text.replace(/^\uFEFF/, "").split(/\r?\n/).entries()) {
    if (!raw.trim()) continue;
    const error = (reason) => fail(`${path}:${i + 1}: ${reason}`);
    const tokens = raw.trim().split(/\s+/);
    if (!/^\d+$/.test(tokens[0])) error("class ID must be a nonnegative integer");
    const classIdx = Number(tokens[0]);
    if (!Number.isSafeInteger(classIdx) || classIdx >= classCount) error("class ID is outside data.yaml names");
    if (tokens.length !== 5 && (tokens.length < 7 || tokens.length % 2 !== 1)) error("expected class + 4 box coordinates or at least 3 polygon points");
    const coords = tokens.slice(1).map((token) => {
      if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(token)) error(`invalid numeric coordinate ${token}`);
      const value = Number(token);
      if (!Number.isFinite(value) || value < 0 || value > 1) error("coordinates must be finite and normalized to [0, 1]");
      return value;
    });
    if (coords.length === 4) {
      const [cx, cy, w, h] = coords;
      if (!w || !h) error("box width and height must be positive");
      // Allow rounding at six decimal places used by YOLO exporters.
      if (cx - w / 2 < -1e-6 || cy - h / 2 < -1e-6 || cx + w / 2 > 1 + 1e-6 || cy + h / 2 > 1 + 1e-6) error("box extends outside the image");
      result.push({ type: "bbox", classIdx, cx, cy, w, h });
    } else {
      const points = [];
      for (let j = 0; j < coords.length; j += 2) points.push({ x: coords[j], y: coords[j + 1] });
      const area = points.reduce((sum, p, j) => { const q = points[(j + 1) % points.length]; return sum + p.x * q.y - q.x * p.y; }, 0);
      if (Math.abs(area) <= Number.EPSILON) error("polygon must have nonzero area");
      result.push({ type: "polygon", classIdx, points });
    }
  }
  return result;
}

function safePath(path) {
  const parts = path.replace(/\/$/, "").split("/");
  if (!path || /[\\:\x00-\x1f\x7f]/.test(path) || parts.some((p) => !p || p === "." || p === "..")) fail(`unsafe ZIP path: ${path}`);
}

/**
 * Offline, side-effect-free import. Accepts File/Blob, ArrayBuffer or Uint8Array.
 * Returns {files: File[], classNames: string[], images: Array<{
 *   key: string, file: File, sourcePath: string, split: 'train'|'valid'|'test',
 *   annotations: Array<{type:'bbox',classIdx,cx,cy,w,h}|{type:'polygon',classIdx,points}>
 * }>, warnings: string[]}.
 * key === file.name; names are deduplicated across this import (not existing app
 * images). Pair annotations with these keys, never the original basename.
 * Missing labels mean background images and produce warnings; empty labels are
 * valid backgrounds. Orphan labels and ambiguous image/label pairs are errors.
 * Finds one data.yaml (optionally under a wrapper folder). Infers splits from
 * split/images or images/split, with val -> valid. YAML path/train/val/test and
 * other metadata are ignored, including Roboflow's ../ paths; never fetched.
 * Does not decode image pixels, mutate app state, or create object URLs.
 */
export async function importYoloDataset(input) {
  let zip;
  try {
    const bytes = typeof input?.arrayBuffer === "function" ? await input.arrayBuffer() : input;
    zip = await JSZip.loadAsync(bytes);
  } catch (err) { fail(`cannot read ZIP: ${err.message}`); }
  const entries = Object.values(zip.files);
  for (const entry of entries) {
    safePath(entry.name);
    if (entry.unsafeOriginalName !== undefined) safePath(entry.unsafeOriginalName);
    if (((entry.unixPermissions || 0) & 0xf000) === 0xa000) fail(`symbolic links are unsupported: ${entry.name}`);
  }
  const manifests = entries.filter((e) => !e.dir && /(^|\/)data\.yaml$/i.test(e.name));
  if (manifests.length !== 1) fail("expected exactly one data.yaml in ZIP");
  const manifest = manifests[0];
  const classNames = parseYoloNames(await manifest.async("string"));
  const root = manifest.name.slice(0, -"data.yaml".length);
  const imageEntries = [], labels = new Map(), pairKeys = new Set();
  for (const entry of entries) {
    if (entry.dir || !entry.name.startsWith(root)) continue;
    const path = entry.name.slice(root.length);
    const match = /^(train|valid|val|test)\/(images|labels)\/(.+)$/.exec(path);
    const reverse = /^(images|labels)\/(train|valid|val|test)\/(.+)$/.exec(path);
    if (!match && !reverse) continue;
    const splitDir = match ? match[1] : reverse[2];
    const kind = match ? match[2] : reverse[1];
    const relative = (match || reverse)[3];
    const split = splitDir === "val" ? "valid" : splitDir;
    const extension = relative.split(".").pop().toLowerCase();
    const stem = relative.replace(/\.[^/.]+$/, "");
    // Layout is part of the pair key: never attach a label from another tree.
    const key = `${match ? "split-first" : "images-first"}/${splitDir}/${stem}`;
    if (kind === "labels") {
      if (extension !== "txt") continue;
      if (labels.has(key)) fail(`ambiguous labels for ${entry.name}`);
      labels.set(key, entry);
    } else if (MIME[extension]) {
      if (pairKeys.has(key)) fail(`ambiguous images sharing one label: ${entry.name}`);
      pairKeys.add(key);
      imageEntries.push({ entry, key, split, extension });
    }
  }
  for (const [key, entry] of labels) if (!pairKeys.has(key)) fail(`orphan label has no matching supported image: ${entry.name}`);
  if (!imageEntries.length) fail("no supported images in train/valid/test images folders");
  imageEntries.sort((a, b) => a.entry.name < b.entry.name ? -1 : a.entry.name > b.entry.name ? 1 : 0);
  const images = [], warnings = [], used = new Set();
  for (const { entry, key, split, extension } of imageEntries) {
    const label = labels.get(key);
    const anns = label ? annotations(await label.async("string"), label.name, classNames.length) : [];
    if (!label) warnings.push(`${entry.name}: no label file; imported as background`);
    const base = entry.name.split("/").pop();
    let name = base, suffix = 2;
    while (used.has(name.toLowerCase())) name = `${base.replace(/\.[^.]+$/, "")}_${suffix++}.${extension}`;
    used.add(name.toLowerCase());
    const file = new File([await entry.async("uint8array")], name, { type: MIME[extension], lastModified: 0 });
    images.push({ key: name, file, sourcePath: entry.name, split, annotations: anns });
  }
  return { files: images.map((image) => image.file), classNames, images, warnings };
}

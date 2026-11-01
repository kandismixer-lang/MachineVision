export function testPattern5G(width, height, pattern, square, matlabGray) {
  const channels = pattern === "color_bars" ? 3 : 1;
  const data = new Uint8Array(width * height * channels);
  const colors = [[255,255,255],[255,255,0],[0,255,255],[0,255,0],[255,0,255],[255,0,0],[0,0,255],[0,0,0]];
  const radius = Math.max(1, Math.floor(square / 4));
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = y * width + x;
    if (channels === 3) {
      data.set(colors[Math.floor(8 * x / width)], i * 3);
    } else if (pattern === "checkerboard") {
      const bright = ((Math.floor(x / square) + Math.floor(y / square)) & 1) === 1;
      data[i] = bright ? (matlabGray && x >= Math.floor(width / 2) ? 179 : 255) : 0;
    } else if (pattern === "grid") {
      data[i] = x % square === 0 || y % square === 0 || x === width - 1 || y === height - 1 ? 0 : 255;
    } else if (pattern === "dots") {
      const dx = x - Math.floor(x / square) * square - Math.floor(square / 2);
      const dy = y - Math.floor(y / square) * square - Math.floor(square / 2);
      data[i] = dx * dx + dy * dy <= radius * radius ? 255 : 0;
    } else if (pattern === "ramp_h") {
      data[i] = width === 1 ? 0 : Math.floor((510 * x + width - 1) / (2 * (width - 1)));
    } else if (pattern === "ramp_v") {
      data[i] = height === 1 ? 0 : Math.floor((510 * y + height - 1) / (2 * (height - 1)));
    }
  }
  return { data, channels };
}

export function trig12_5G(degrees) {
  const radians = degrees * Math.PI / 180;
  const cosine = Math.floor(Math.cos(radians) * 1e12 + 0.5) / 1e12;
  const sine = Math.floor(Math.sin(radians) * 1e12 + 0.5) / 1e12;
  return [cosine, sine];
}

export function sampleImage5G(data, width, height, channels, sx, sy, interpolation, channel = 0) {
  if (interpolation === "nearest") {
    const x = Math.floor(sx + 0.5), y = Math.floor(sy + 0.5);
    if (x < 0 || x >= width || y < 0 || y >= height) return 0;
    return data[(y * width + x) * channels + channel];
  }
  if (sx < 0 || sx > width - 1 || sy < 0 || sy > height - 1) return 0;
  const x0 = Math.floor(sx), y0 = Math.floor(sy);
  const x1 = Math.min(x0 + 1, width - 1), y1 = Math.min(y0 + 1, height - 1);
  const fx = sx - x0, fy = sy - y0;
  const i00 = (y0 * width + x0) * channels + channel;
  const i10 = (y0 * width + x1) * channels + channel;
  const i01 = (y1 * width + x0) * channels + channel;
  const i11 = (y1 * width + x1) * channels + channel;
  const top = data[i00] + (data[i10] - data[i00]) * fx;
  const bottom = data[i01] + (data[i11] - data[i01]) * fx;
  const value = top + (bottom - top) * fy;
  return Math.min(255, Math.max(0, Math.floor(value + 0.5)));
}

export function similarityTransform5G(data, width, height, channels, angle, scale, tx, ty, interpolation, output) {
  const [cosine, sine] = trig12_5G(angle);
  const centerX = (width - 1) / 2, centerY = (height - 1) / 2;
  const txp = tx / 100 * width, typ = ty / 100 * height;
  let outWidth = width, outHeight = height, ox = 0, oy = 0, oversized = false;
  if (output === "fit") {
    const corners = [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]];
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of corners) {
      const px = scale * (cosine * (x - centerX) + sine * (y - centerY)) + centerX + txp;
      const py = scale * (-sine * (x - centerX) + cosine * (y - centerY)) + centerY + typ;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }
    ox = Math.floor(minX); oy = Math.floor(minY);
    outWidth = Math.ceil(maxX) - ox + 1;
    outHeight = Math.ceil(maxY) - oy + 1;
    if (outWidth * outHeight > 16000000) { outWidth = width; outHeight = height; ox = 0; oy = 0; oversized = true; }
  }
  const out = new Uint8Array(outWidth * outHeight * channels);
  for (let v = 0; v < outHeight; v++) for (let u = 0; u < outWidth; u++) {
    const X = u + ox, Y = v + oy;
    const dx = (X - centerX) - txp, dy = (Y - centerY) - typ;
    const sx = (cosine * dx - sine * dy) / scale + centerX;
    const sy = (sine * dx + cosine * dy) / scale + centerY;
    const at = (v * outWidth + u) * channels;
    for (let channel = 0; channel < channels; channel++) out[at + channel] = sampleImage5G(data, width, height, channels, sx, sy, interpolation, channel);
  }
  const a = scale * cosine, b = scale * sine;
  const matrixX = centerX + txp - a * centerX - b * centerY;
  const matrixY = centerY + typ + b * centerX - a * centerY;
  return {
    data: out, width: outWidth, height: outHeight, channels,
    txp, typ, a, b, matrixX, matrixY,
    usedFit: output === "fit" && !oversized, oversized,
  };
}

export function ballMorph5G(data, width, height, op, ksize, iterations, ballHeight) {
  const radius = Math.floor(ksize / 2);
  const offsets = [];
  if (radius === 0) offsets.push([0, 0, ballHeight]);
  else for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
    const dist2 = dx * dx + dy * dy;
    if (dist2 <= radius * radius) {
      const height = Math.floor(ballHeight * Math.sqrt(1 - dist2 / (radius * radius)) + 0.5);
      offsets.push([dx, dy, height]);
    }
  }
  const offsetX = Int16Array.from(offsets, (offset) => offset[0]);
  const offsetY = Int16Array.from(offsets, (offset) => offset[1]);
  const offsetHeight = Int16Array.from(offsets, (offset) => offset[2]);
  const centerHeight = offsets.find((offset) => offset[0] === 0 && offset[1] === 0)[2];
  const pass = (source, dilate) => {
    const values = new Int16Array(width * height);
    for (let i = 0; i < values.length; i++) values[i] = source[i] + (dilate ? centerHeight : -centerHeight);
    if (dilate) {
      for (let k = 0; k < offsetX.length; k++) {
        const dx = offsetX[k], dy = offsetY[k];
        if (dx === 0 && dy === 0) continue;
        const y0 = Math.max(0, -dy), y1 = Math.min(height, height - dy);
        const x0 = Math.max(0, -dx), x1 = Math.min(width, width - dx);
        if (y1 <= y0 || x1 <= x0) continue;
        const shift = dy * width + dx, hb = offsetHeight[k];
        for (let y = y0; y < y1; y++) {
          let i = y * width + x0, j = i + shift;
          const end = y * width + x1;
          for (; i < end; i++, j++) { const value = source[j] + hb; if (value > values[i]) values[i] = value; }
        }
      }
    } else {
      for (let k = 0; k < offsetX.length; k++) {
        const dx = offsetX[k], dy = offsetY[k];
        if (dx === 0 && dy === 0) continue;
        const y0 = Math.max(0, -dy), y1 = Math.min(height, height - dy);
        const x0 = Math.max(0, -dx), x1 = Math.min(width, width - dx);
        if (y1 <= y0 || x1 <= x0) continue;
        const shift = dy * width + dx, hb = offsetHeight[k];
        for (let y = y0; y < y1; y++) {
          let i = y * width + x0, j = i + shift;
          const end = y * width + x1;
          for (; i < end; i++, j++) { const value = source[j] - hb; if (value < values[i]) values[i] = value; }
        }
      }
    }
    const result = new Uint8Array(width * height);
    for (let i = 0; i < result.length; i++) result[i] = Math.min(255, Math.max(0, values[i]));
    return result;
  };
  const repeat = (source, dilate) => {
    let result = source;
    for (let i = 0; i < iterations; i++) result = pass(result, dilate);
    return result;
  };
  if (op === "erode" || op === "dilate") return repeat(data, op === "dilate");
  if (op === "open") return repeat(repeat(data, false), true);
  if (op === "close") return repeat(repeat(data, true), false);
  if (op === "gradient") {
    const dilated = repeat(data, true), eroded = repeat(data, false), result = new Uint8Array(data.length);
    for (let i = 0; i < result.length; i++) result[i] = Math.max(0, dilated[i] - eroded[i]);
    return result;
  }
  if (op === "tophat") {
    const opened = repeat(repeat(data, false), true), result = new Uint8Array(data.length);
    for (let i = 0; i < result.length; i++) result[i] = Math.max(0, data[i] - opened[i]);
    return result;
  }
  const closed = repeat(repeat(data, true), false), result = new Uint8Array(data.length);
  for (let i = 0; i < result.length; i++) result[i] = Math.max(0, closed[i] - data[i]);
  return result;
}

export function ballKernelSize5G(width, height, op, ksize, iterations, workBudget) {
  const passes = iterations * (op === "erode" || op === "dilate" ? 1 : 2);
  const work = (kernelSize) => {
    const radius = Math.floor(kernelSize / 2);
    let total = 0;
    for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy <= radius * radius) total += Math.max(0, width - Math.abs(dx)) * Math.max(0, height - Math.abs(dy));
    }
    return total * passes;
  };
  let effective = ksize;
  while (effective > 1 && work(effective) > workBudget) effective -= 2;
  return effective;
}

export function mtbHalf5G(data, width, height) {
  const outWidth = Math.floor(width / 2), outHeight = Math.floor(height / 2), result = new Uint8Array(outWidth * outHeight);
  for (let y = 0; y < outHeight; y++) for (let x = 0; x < outWidth; x++) {
    const at = 2 * y * width + 2 * x;
    result[y * outWidth + x] = Math.floor((data[at] + data[at + 1] + data[at + width] + data[at + width + 1] + 2) / 4);
  }
  return result;
}

export function quadtree5G(data, width, height, threshold, maxSize, minSize, view) {
  const outputChannels = view === "blocks" ? 3 : 1;
  if (width === 0 || height === 0) return { data: new Uint8Array(0), channels: outputChannels, blocks: 0, counts: new Map() };
  const leaves = [], counts = new Map();
  const scan = (x0, y0, size) => {
    let minimum = 255, maximum = 0, sum = 0, count = 0;
    const x1 = Math.min(x0 + size, width), y1 = Math.min(y0 + size, height);
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const value = data[y * width + x];
      if (value < minimum) minimum = value;
      if (value > maximum) maximum = value;
      sum += value;
      count++;
    }
    return { x1, y1, minimum, maximum, sum, count };
  };
  const split = (x0, y0, size) => {
    const q = scan(x0, y0, size);
    if (size > minSize && q.maximum - q.minimum > threshold) {
      const half = size / 2;
      for (const dy of [0, half]) for (const dx of [0, half]) {
        const x = x0 + dx, y = y0 + dy;
        if (x < width && y < height) split(x, y, half);
      }
    } else {
      leaves.push({ x0, y0, size, ...q });
      counts.set(size, (counts.get(size) || 0) + 1);
    }
  };
  for (let y = 0; y < height; y += maxSize) for (let x = 0; x < width; x += maxSize) split(x, y, maxSize);
  const output = new Uint8Array(width * height * outputChannels);
  if (view === "blocks") {
    for (let i = 0; i < data.length; i++) output[i * 3] = output[i * 3 + 1] = output[i * 3 + 2] = data[i];
    for (const q of leaves) {
      for (let x = q.x0; x < q.x1; x++) { const i = (q.y0 * width + x) * 3; output[i] = 0; output[i + 1] = 255; output[i + 2] = 0; }
      for (let y = q.y0; y < q.y1; y++) { const i = (y * width + q.x0) * 3; output[i] = 0; output[i + 1] = 255; output[i + 2] = 0; }
    }
  } else if (view === "mean") {
    for (const q of leaves) {
      const mean = Math.floor((2 * q.sum + q.count) / (2 * q.count));
      for (let y = q.y0; y < q.y1; y++) output.fill(mean, y * width + q.x0, y * width + q.x1);
    }
  } else {
    const maxExponent = Math.log2(maxSize), minExponent = Math.log2(minSize), levels = maxExponent - minExponent;
    for (const q of leaves) {
      const k = Math.log2(q.size) - minExponent;
      const value = levels === 0 ? 255 : Math.floor((510 * k + levels) / (2 * levels));
      for (let y = q.y0; y < q.y1; y++) output.fill(value, y * width + q.x0, y * width + q.x1);
    }
  }
  return { data: output, channels: outputChannels, blocks: leaves.length, counts };
}

export function contourLines5G(data, width, height, levelsCount, view) {
  const output = new Uint8Array(width * height * 3);
  let minimum = 255, maximum = 0;
  for (const value of data) { if (value < minimum) minimum = value; if (value > maximum) maximum = value; }
  if (view === "overlay") for (let i = 0; i < data.length; i++) output[i * 3] = output[i * 3 + 1] = output[i * 3 + 2] = data[i];
  else output.fill(255);
  const levels = [];
  if (data.length === 0 || minimum === maximum) return { data: output, levels, flat: true };
  for (let k = 1; k <= levelsCount; k++) levels.push(minimum + (maximum - minimum) * k / (levelsCount + 1));
  const colors = levels.map((_, index) => {
    const t = levelsCount === 1 ? 0.5 : index / (levelsCount - 1);
    const c = (v) => Math.min(1, Math.max(0, v));
    return [
      Math.floor(c(1.5 - Math.abs(4 * t - 3)) * 255 + 0.5),
      Math.floor(c(1.5 - Math.abs(4 * t - 2)) * 255 + 0.5),
      Math.floor(c(1.5 - Math.abs(4 * t - 1)) * 255 + 0.5),
    ];
  });
  for (let k = 0; k < levelsCount; k++) {
    const level = levels[k], color = colors[k];
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const i = y * width + x, value = data[i];
      if (x + 1 < width) {
        const right = data[i + 1];
        if ((value >= level) !== (right >= level)) {
          const at = (value >= level ? i : i + 1) * 3;
          output[at] = color[0]; output[at + 1] = color[1]; output[at + 2] = color[2];
        }
      }
      if (y + 1 < height) {
        const below = data[i + width];
        if ((value >= level) !== (below >= level)) {
          const at = (value >= level ? i : i + width) * 3;
          output[at] = color[0]; output[at + 1] = color[1]; output[at + 2] = color[2];
        }
      }
    }
  }
  return { data: output, levels, flat: false };
}

export function mtbRegister5G(rawA, grayA, grayB, width, height, channels, maxShift, exclusion, view) {
  let levels = Math.log2(maxShift + 1);
  while (levels > 1 && (Math.min(width, height) >> (levels - 1)) < 8) levels--;
  const pyramid = (base) => {
    const result = [base];
    let previousWidth = width, previousHeight = height;
    for (let i = 1; i < levels; i++) {
      const previous = result[i - 1], w1 = Math.floor(previousWidth / 2), h1 = Math.floor(previousHeight / 2);
      const next = mtbHalf5G(previous, previousWidth, previousHeight);
      result.push(next);
      previousWidth = w1; previousHeight = h1;
    }
    return result;
  };
  const pyrA = pyramid(grayA), pyrB = pyramid(grayB), masks = (data) => {
    const histogram = new Uint32Array(256);
    for (const value of data) histogram[value]++;
    const n = data.length;
    let cumulative = 0, median = 0;
    for (; median < 255; median++) { cumulative += histogram[median]; if (2 * cumulative >= n) break; }
    const above = new Uint8Array(n), excluded = new Uint8Array(n);
    for (let i = 0; i < n; i++) { above[i] = data[i] > median ? 1 : 0; excluded[i] = Math.abs(data[i] - median) > exclusion ? 1 : 0; }
    return { median, above, excluded };
  };
  let sx = 0, sy = 0, finalError = 0, comparable = 0, medianA = 0, aboveA = null, excludedA = null;
  const deltas = [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
  for (let level = levels - 1; level >= 0; level--) {
    if (level < levels - 1) { sx *= 2; sy *= 2; }
    const w = Math.floor(width / (2 ** level)), h = Math.floor(height / (2 ** level));
    const a = masks(pyrA[level]), b = masks(pyrB[level]);
    if (level === 0) { medianA = a.median; aboveA = a.above; excludedA = a.excluded; }
    const errorAt = (tx, ty) => {
      let error = 0, used = 0;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const ax = x - tx, ay = y - ty;
        if (ax < 0 || ax >= w || ay < 0 || ay >= h) continue;
        const ai = ay * w + ax, bi = y * w + x;
        if (a.excluded[ai] && b.excluded[bi]) {
          used++;
          if (a.above[ai] !== b.above[bi]) error++;
        }
      }
      return [error, used];
    };
    [finalError, comparable] = errorAt(sx, sy);
    const baseX = sx, baseY = sy;
    let bestX = sx, bestY = sy;
    for (const [dx, dy] of deltas) {
      const tx = baseX + dx, ty = baseY + dy, [candidate, candidateUsed] = errorAt(tx, ty);
      if (candidateUsed > 0 && (comparable === 0 || candidate * comparable < finalError * candidateUsed)) {
        finalError = candidate; comparable = candidateUsed; bestX = tx; bestY = ty;
      }
    }
    sx = bestX; sy = bestY;
    [finalError, comparable] = errorAt(sx, sy);
  }
  const outChannels = channels === 1 ? 1 : 3;
  let output;
  if (view === "aligned") {
    output = new Uint8Array(width * height * outChannels);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const ax = x - sx, ay = y - sy;
      if (ax < 0 || ax >= width || ay < 0 || ay >= height) continue;
      const si = (ay * width + ax) * channels, di = (y * width + x) * outChannels;
      for (let c = 0; c < outChannels; c++) output[di + c] = rawA[si + c];
    }
  } else if (view === "overlay") {
    output = new Uint8Array(width * height * 3);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const ax = x - sx, ay = y - sy, i = y * width + x, j = i * 3;
      const av = ax < 0 || ax >= width || ay < 0 || ay >= height ? 0 : grayA[ay * width + ax];
      output[j] = av; output[j + 1] = grayB[i]; output[j + 2] = av;
    }
  } else if (view === "diff") {
    output = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const ax = x - sx, ay = y - sy, i = y * width + x;
      output[i] = ax < 0 || ax >= width || ay < 0 || ay >= height ? grayB[i] : Math.abs(grayA[ay * width + ax] - grayB[i]);
    }
  } else {
    output = new Uint8Array(width * height);
    for (let i = 0; i < output.length; i++) output[i] = excludedA[i] ? aboveA[i] ? 255 : 0 : 128;
  }
  return { data: output, channels: view === "overlay" ? 3 : view === "aligned" ? outChannels : 1, sx, sy, error: finalError, comparable, levels, median: medianA };
}

export function geodesicSweep5G(width, height, cost, seeds, conn, maxSets) {
  const distance = new Float64Array(width * height);
  distance.fill(Infinity);
  for (const seed of seeds) if (seed >= 0 && seed < distance.length) distance[seed] = 0;
  const diagonal = conn === 8 || conn === true || conn === "quasi" || conn === "chessboard";
  const relax = (px, py, qx, qy, isDiagonal) => {
    if (px < 0 || px >= width || py < 0 || py >= height || qx < 0 || qx >= width || qy < 0 || qy >= height) return false;
    const from = py * width + px, to = qy * width + qx;
    const edge = cost(from, to, isDiagonal);
    if (!Number.isFinite(edge)) return false;
    const candidate = distance[from] + edge;
    if (candidate < distance[to]) { distance[to] = candidate; return true; }
    return false;
  };
  const limit = Math.max(1, Math.floor(maxSets));
  let sweeps = 0, converged = false;
  for (; sweeps < limit; sweeps++) {
    let changed = false;
    for (let x = 1; x < width; x++) for (let y = 0; y < height; y++) {
      if (relax(x - 1, y, x, y, false)) changed = true;
      if (diagonal && y > 0 && relax(x - 1, y - 1, x, y, true)) changed = true;
      if (diagonal && y + 1 < height && relax(x - 1, y + 1, x, y, true)) changed = true;
    }
    for (let x = width - 2; x >= 0; x--) for (let y = 0; y < height; y++) {
      if (relax(x + 1, y, x, y, false)) changed = true;
      if (diagonal && y > 0 && relax(x + 1, y - 1, x, y, true)) changed = true;
      if (diagonal && y + 1 < height && relax(x + 1, y + 1, x, y, true)) changed = true;
    }
    for (let y = 1; y < height; y++) for (let x = 0; x < width; x++) {
      if (relax(x, y - 1, x, y, false)) changed = true;
      if (diagonal && x > 0 && relax(x - 1, y - 1, x, y, true)) changed = true;
      if (diagonal && x + 1 < width && relax(x + 1, y - 1, x, y, true)) changed = true;
    }
    for (let y = height - 2; y >= 0; y--) for (let x = 0; x < width; x++) {
      if (relax(x, y + 1, x, y, false)) changed = true;
      if (diagonal && x > 0 && relax(x - 1, y + 1, x, y, true)) changed = true;
      if (diagonal && x + 1 < width && relax(x + 1, y + 1, x, y, true)) changed = true;
    }
    if (!changed) { sweeps++; converged = true; break; }
  }
  return { distance, sweeps, converged };
}

export function geodesicDistance5G(gray, width, height, method, metric, weight, seedX, seedY, view, threshold, maxSets) {
  const seed = seedY * width + seedX, seedValue = gray[seed];
  const conn = metric === "cityblock" ? 4 : 8;
  const step = (isDiagonal) => isDiagonal && metric === "quasi" ? Math.SQRT2 : 1;
  const cost = (from, to, isDiagonal) => {
    const scale = step(isDiagonal), a = gray[from], b = gray[to];
    if (method === "binary") return b <= 127 ? Infinity : scale;
    if (method === "gray") return (a + b) * 0.5 * scale;
    const da = Math.abs(a - seedValue), db = Math.abs(b - seedValue);
    return (1 + weight * ((da + db) * 0.5)) * scale;
  };
  if (method === "binary" && seedValue <= 127) {
    const channels = view === "distance" ? 3 : 1;
    return { data: new Uint8Array(width * height * channels), channels, distance: new Float64Array(width * height).fill(Infinity), maxDist: 0, reached: 0, area: 0, sweeps: 0, converged: true, seedX, seedY, seedValue, seedBlocked: true };
  }
  const result = geodesicSweep5G(width, height, cost, [seed], conn, maxSets);
  let maxDist = 0, reached = 0;
  for (const value of result.distance) if (Number.isFinite(value)) { reached++; if (value > maxDist) maxDist = value; }
  const channels = view === "distance" ? 3 : 1, data = new Uint8Array(width * height * channels);
  let area = 0;
  for (let i = 0; i < result.distance.length; i++) {
    const value = result.distance[i], reachedPixel = Number.isFinite(value);
    if (view === "segment") {
      if (reachedPixel && value <= threshold * maxDist) { data[i] = 255; area++; }
    } else {
      const at = i * 3;
      if (reachedPixel) { const grayValue = maxDist > 0 ? Math.floor(value / maxDist * 255 + 0.5) : 0; data[at] = grayValue; data[at + 1] = grayValue; data[at + 2] = grayValue; }
      else data[at + 2] = 96;
    }
  }
  if (view === "distance") {
    for (let delta = -3; delta <= 3; delta++) {
      if (seedX + delta >= 0 && seedX + delta < width) { const i = (seedY * width + seedX + delta) * 3; data[i] = 255; data[i + 1] = 0; data[i + 2] = 0; }
      if (seedY + delta >= 0 && seedY + delta < height) { const i = ((seedY + delta) * width + seedX) * 3; data[i] = 255; data[i + 1] = 0; data[i + 2] = 0; }
    }
  }
  return { data, channels, distance: result.distance, maxDist, reached, area, sweeps: result.sweeps, converged: result.converged, seedX, seedY, seedValue, seedBlocked: false };
}

export function radonPeakBand5G(x, y, centerX, centerY, cosine, sine, rho) {
  return Math.abs((x - centerX) * cosine + (centerY - y) * sine - rho) <= 0.5;
}

export function radon5G(gray, width, height, thetaStep, view, workBudget) {
  const original = gray, originalWidth = width, originalHeight = height;
  const factor = Math.max(1, Math.ceil(Math.max(width, height) / 512));
  if (factor > 1) {
    const reducedWidth = Math.ceil(width / factor), reducedHeight = Math.ceil(height / factor), reduced = new Uint8Array(reducedWidth * reducedHeight);
    for (let by = 0; by < reducedHeight; by++) for (let bx = 0; bx < reducedWidth; bx++) {
      const x0 = bx * factor, y0 = by * factor, x1 = Math.min(width, x0 + factor), y1 = Math.min(height, y0 + factor);
      let sum = 0, count = 0;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { sum += gray[y * width + x]; count++; }
      reduced[by * reducedWidth + bx] = Math.floor((2 * sum + count) / (2 * count));
    }
    gray = reduced;
    width = reducedWidth; height = reducedHeight;
  }
  const steps = [1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90];
  let nonzero = 0;
  for (const value of gray) if (value !== 0) nonzero++;
  let degrees = 90;
  for (const candidate of steps) if (candidate >= thetaStep && nonzero * 8 * (180 / candidate) <= 4 * workBudget) { degrees = candidate; break; }
  const centerX = Math.floor((width - 1) / 2), centerY = Math.floor((height - 1) / 2);
  const rx = width - 1 - centerX, ry = height - 1 - centerY;
  const last = Math.ceil(Math.sqrt(rx * rx + ry * ry)) + 1, first = -last, binsCount = 2 * last + 1;
  const thetaCount = Math.ceil(180 / degrees), values = new Float64Array(binsCount * thetaCount);
  const offsets = [[-0.25,-0.25],[0.25,-0.25],[-0.25,0.25],[0.25,0.25]];
  for (let thetaIndex = 0; thetaIndex < thetaCount; thetaIndex++) {
    const theta = thetaIndex * degrees, [cosine, sine] = trig12_5G(theta);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const value = gray[y * width + x];
      if (value === 0) continue;
      const quarter = value * 0.25, px = x - centerX, py = centerY - y;
      for (const [ox, oy] of offsets) {
        const t = ((px + ox) * cosine + (py + oy) * sine) - first;
        const lo = Math.floor(t), left = quarter * ((1 - t) + lo), at = lo * thetaCount + thetaIndex;
        values[at] += left;
        values[at + thetaCount] += quarter - left;
      }
    }
  }
  let maxValue = 0, peakAt = -1;
  for (let i = 0; i < values.length; i++) if (values[i] > maxValue) { maxValue = values[i]; peakAt = i; }
  const peakTheta = peakAt < 0 ? 0 : (peakAt % thetaCount) * degrees;
  const peakRow = peakAt < 0 ? 0 : Math.floor(peakAt / thetaCount), peakRho = peakAt < 0 ? 0 : peakRow + first, peakValue = peakAt < 0 ? 0 : values[peakAt];
  let output;
  if (view === "peak") {
    const [peakCosine, peakSine] = trig12_5G(peakTheta);
    output = new Uint8Array(originalWidth * originalHeight * 3);
    for (let y = 0; y < originalHeight; y++) for (let x = 0; x < originalWidth; x++) {
      const i = y * originalWidth + x, at = i * 3, value = original[i];
      output[at] = output[at + 1] = output[at + 2] = value;
      if (peakAt >= 0 && radonPeakBand5G(Math.floor(x / factor), Math.floor(y / factor), centerX, centerY, peakCosine, peakSine, peakRho)) {
        output[at] = 255; output[at + 1] = 0; output[at + 2] = 0;
      }
    }
  } else {
    output = new Uint8Array(values.length);
    if (maxValue > 0) for (let i = 0; i < values.length; i++) output[i] = Math.floor(values[i] / maxValue * 255 + 0.5);
  }
  return { data: output, width: view === "peak" ? originalWidth : thetaCount, height: view === "peak" ? originalHeight : binsCount, channels: view === "peak" ? 3 : 1, values: nonzero === 0 ? new Float64Array(0) : values, maxValue, peakTheta, peakRho, peakValue, first, last, degrees, thetaCount, nonzero, peakAt, factor, analysisWidth: width, analysisHeight: height };
}

// ปัด sin/cos ที่ 1e-12 เพื่อให้ค่าจากตารางตรงกันข้ามเครื่อง
export function cos12Rad5G(a) { return Math.floor(Math.cos(a) * 1e12 + 0.5) / 1e12; }
export function sin12Rad5G(a) { return Math.floor(Math.sin(a) * 1e12 + 0.5) / 1e12; }

export function reflect101_5G(i, n) {
  if (n === 1) return 0;
  const p = 2 * n - 2;
  i = ((i % p) + p) % p;
  return i >= n ? p - i : i;
}

export function phaseWindow5G(n) {
  const out = new Float64Array(n);
  if (n === 1) { out[0] = 1; return out; }
  for (let i = 0; i < n; i++) out[i] = 0.5 - 0.5 * cos12Rad5G(((2 * Math.PI) * i) / (n - 1));
  return out;
}

export function phasePad5G(gray, width, height, wp, hp, useWindow) {
  const out = new Float64Array(wp * hp), wx = phaseWindow5G(width), wy = phaseWindow5G(height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const value = gray[y * width + x];
    out[y * wp + x] = useWindow ? value * (wy[y] * wx[x]) : value;
  }
  return out;
}

export function phaseCrossPower5G(specA, specB, out) {
  let maxMagnitude = 0;
  for (let i = 0; i < specA.length; i += 2) {
    const ar = specA[i], ai = specA[i + 1], br = specB[i], bi = specB[i + 1];
    const re = ar * br + ai * bi, im = ai * br - ar * bi, magnitude = Math.sqrt(re * re + im * im);
    if (magnitude > maxMagnitude) maxMagnitude = magnitude;
  }
  const minMagnitude = 1e-9 * maxMagnitude;
  for (let i = 0; i < specA.length; i += 2) {
    const ar = specA[i], ai = specA[i + 1], br = specB[i], bi = specB[i + 1];
    const re = ar * br + ai * bi, im = ai * br - ar * bi, m = Math.sqrt(re * re + im * im);
    if (m > minMagnitude) { out[i] = re / m; out[i + 1] = im / m; }
    else { out[i] = 0; out[i + 1] = 0; }
  }
}

export function phasePeak5G(inv, wp, hp, subpixel) {
  let peak = -Infinity, at = 0;
  for (let i = 0, n = wp * hp; i < n; i++) { const value = inv[2 * i]; if (value > peak) { peak = value; at = i; } }
  const py = Math.floor(at / wp), px = at % wp;
  let sx = 0, sy = 0;
  if (subpixel) {
    const left = inv[2 * (py * wp + ((px - 1 + wp) % wp))], right = inv[2 * (py * wp + ((px + 1) % wp))];
    const denX = (left - 2 * peak) + right;
    sx = Math.min(0.5, Math.max(-0.5, denX < 0 ? (left - right) / (2 * denX) : 0));
    const top = inv[2 * (((py - 1 + hp) % hp) * wp + px)], bottom = inv[2 * (((py + 1) % hp) * wp + px)];
    const denY = (top - 2 * peak) + bottom;
    sy = Math.min(0.5, Math.max(-0.5, denY < 0 ? (top - bottom) / (2 * denY) : 0));
  }
  const dx = (px > wp / 2 ? px - wp : px) + sx, dy = (py > hp / 2 ? py - hp : py) + sy;
  let cx = 0 - dx, cy = 0 - dy;
  if (Math.abs(cx - Math.round(cx)) < 1e-9) cx = Math.round(cx);
  if (Math.abs(cy - Math.round(cy)) < 1e-9) cy = Math.round(cy);
  return { px, py, peak, dx, dy, cx, cy };
}

export function phaseViews5G(rawA, grayA, grayB, width, height, channels, cx, cy, view) {
  const grayAligned = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    grayAligned[y * width + x] = sampleImage5G(grayA, width, height, 1, x - cx, y - cy, "bilinear", 0);
  }
  if (view === "aligned") {
    const out = new Uint8Array(width * height * channels);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let c = 0; c < channels; c++) {
      out[(y * width + x) * channels + c] = sampleImage5G(rawA, width, height, channels, x - cx, y - cy, "bilinear", c);
    }
    return { data: out, channels };
  }
  if (view === "overlay") {
    const out = new Uint8Array(width * height * 3);
    for (let i = 0; i < width * height; i++) { out[3 * i] = grayAligned[i]; out[3 * i + 1] = grayB[i]; out[3 * i + 2] = grayAligned[i]; }
    return { data: out, channels: 3 };
  }
  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; i++) out[i] = Math.abs(grayAligned[i] - grayB[i]);
  return { data: out, channels: 1 };
}

export function phaseSurface5G(inv, wp, hp, peak) {
  const out = new Uint8Array(wp * hp), sx = Math.floor(wp / 2), sy = Math.floor(hp / 2);
  if (!(peak > 0)) return out;
  for (let y = 0; y < hp; y++) for (let x = 0; x < wp; x++) {
    const value = inv[2 * (y * wp + x)];
    const target = ((y + sy) % hp) * wp + ((x + sx) % wp);
    out[target] = Math.min(255, Math.floor((Math.max(value, 0) / peak) * 255 + 0.5));
  }
  return out;
}

export function phaseFailure5G(rawA, grayA, width, height, channels, view) {
  if (view === "aligned") return { data: Uint8Array.from(rawA), channels };
  if (view === "mask" || view === "diff" || view === "surface") return { data: Uint8Array.from(grayA), channels: 1 };
  const out = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    if (channels === 1) out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = rawA[i];
    else { out[3 * i] = rawA[i * channels]; out[3 * i + 1] = rawA[i * channels + 1]; out[3 * i + 2] = rawA[i * channels + 2]; }
  }
  return { data: out, channels: 3 };
}

export const ISODATA_PALETTE_5G = [[230,25,75],[60,180,75],[0,130,200],[255,225,25],[245,130,48],[145,30,180],[70,240,240],[240,50,230],[210,245,60],[250,190,212],[0,128,128],[170,110,40]];

export function scribbleSegment5G(pix, width, height, ch, labels, edge, metric, view, maxSets) {
  const n = width * height;
  const cost = (from, to, diagonal) => {
    let d = 0;
    const a = from * ch, b = to * ch;
    for (let c = 0; c < ch; c++) { const delta = Math.abs(pix[a + c] - pix[b + c]); if (delta > d) d = delta; }
    return (1 + edge * d) * (diagonal && metric === "quasi" ? Math.SQRT2 : 1);
  };
  const red = [], blue = [];
  for (let i = 0; i < n; i++) { if (labels[i] === 1) red.push(i); else if (labels[i] === 2) blue.push(i); }
  const dr = geodesicSweep5G(width, height, cost, red, 8, maxSets), db = geodesicSweep5G(width, height, cost, blue, 8, maxSets);
  const areaMask = new Uint8Array(n);
  let area = 0, maxDistance = 0;
  for (let i = 0; i < n; i++) if (dr.distance[i] < db.distance[i]) { areaMask[i] = 1; area++; }
  for (const d of dr.distance) if (Number.isFinite(d) && d > maxDistance) maxDistance = d;
  if (view === "mask") {
    const data = new Uint8Array(n);
    for (let i = 0; i < n; i++) if (areaMask[i]) data[i] = 255;
    return { data, channels: 1, area, red: red.length, blue: blue.length, sweeps: dr.sweeps + db.sweeps, converged: dr.converged && db.converged };
  }
  const data = new Uint8Array(n * 3);
  if (view === "distance") {
    for (let i = 0; i < n; i++) {
      const d = dr.distance[i], at = i * 3;
      if (Number.isFinite(d)) { const v = maxDistance > 0 ? Math.floor(d / maxDistance * 255 + 0.5) : 0; data[at] = data[at + 1] = data[at + 2] = v; }
      else data[at + 2] = 96;
    }
  } else {
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
      const value = pix[i * ch + (ch === 1 ? 0 : c)];
      data[i * 3 + c] = areaMask[i] ? (value + (c === 1 ? 255 : 0) + 1) >> 1 : value;
    }
    for (let i = 0; i < n; i++) if (labels[i] === 1 || labels[i] === 2) {
      const color = labels[i] === 1 ? [230, 30, 30] : [30, 80, 240];
      data.set(color, i * 3);
    }
  }
  return { data, channels: 3, area, red: red.length, blue: blue.length, sweeps: dr.sweeps + db.sweeps, converged: dr.converged && db.converged };
}

export function isodata5G(pix, width, height, ch, k, iterations, minSizePct, splitStd, mergeDist, view, iterBudget) {
  const N = width * height, nf = ch === 1 ? 1 : 3;
  const keys = new Uint32Array(N);
  for (let i = 0; i < N; i++) keys[i] = nf === 1 ? pix[i] : ((pix[3 * i] << 16) | (pix[3 * i + 1] << 8) | pix[3 * i + 2]);
  keys.sort();
  let U = 0;
  for (let i = 0; i < N; i++) if (i === 0 || keys[i] !== keys[i - 1]) U++;
  const ukey = new Uint32Array(U), cnt = new Float64Array(U);
  for (let i = 0, u = -1; i < N; i++) { if (i === 0 || keys[i] !== keys[i - 1]) { u++; ukey[u] = keys[i]; } cnt[u]++; }
  const feats = nf === 1 ? [Float64Array.from(ukey)] : [Float64Array.from(ukey, (v) => v >>> 16), Float64Array.from(ukey, (v) => (v >>> 8) & 255), Float64Array.from(ukey, (v) => v & 255)];
  const lum = new Float64Array(U);
  for (let u = 0; u < U; u++) lum[u] = nf === 1 ? feats[0][u] : 299 * feats[0][u] + 587 * feats[1][u] + 114 * feats[2][u];
  const order = Array.from({ length: U }, (_, u) => u).sort((a, b) => lum[a] - lum[b] || a - b), cum = new Float64Array(U);
  for (let i = 0, run = 0; i < U; i++) { run += cnt[order[i]]; cum[i] = run; }
  let centers = [];
  for (let j = 0; j < k; j++) {
    const rank = Math.floor(((2 * j + 1) * N) / (2 * k));
    let lo = 0, hi = U - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] > rank) hi = mid; else lo = mid + 1; }
    centers.push(feats.map((f) => f[order[lo]]));
  }
  const assign = (cs) => {
    const lab = new Int32Array(U), best = new Float64Array(U).fill(Infinity);
    for (let j = 0; j < cs.length; j++) for (let u = 0; u < U; u++) {
      const t0 = feats[0][u] - cs[j][0]; let d = t0 * t0;
      if (nf === 3) { const t1 = feats[1][u] - cs[j][1]; d = d + t1 * t1; const t2 = feats[2][u] - cs[j][2]; d = d + t2 * t2; }
      if (d < best[u]) { best[u] = d; lab[u] = j; }
    }
    return lab;
  };
  const stats = (lab, kk) => {
    const n = new Float64Array(kk), s1 = feats.map(() => new Float64Array(kk)), s2 = feats.map(() => new Float64Array(kk));
    for (let u = 0; u < U; u++) { const j = lab[u]; n[j] += cnt[u]; for (let c = 0; c < nf; c++) { s1[c][j] += cnt[u] * feats[c][u]; s2[c][j] += cnt[u] * feats[c][u] * feats[c][u]; } }
    return { n, s1, s2 };
  };
  const dropEmpty = (lab, cs, st) => {
    const keep = Array.from(st.n, (v) => v > 0);
    if (keep.every(Boolean)) return [lab, cs, st, 0];
    const remap = new Int32Array(keep.length); let next = 0;
    keep.forEach((keepIt, j) => { remap[j] = keepIt ? next++ : -1; });
    const nextLab = Int32Array.from(lab, (v) => remap[v]), nextCenters = cs.filter((_, j) => keep[j]);
    return [nextLab, nextCenters, stats(nextLab, nextCenters.length), keep.length - nextCenters.length];
  };
  const minSize = Math.max(1, Math.floor((minSizePct * N) / 100)), maxK = 2 * k, floorK = Math.max(1, Math.floor(k / 2));
  let splits = 0, merges = 0, cuts = 0, used = 0, converged = false, prev = null, prevK = -1;
  for (let it = 1; it <= iterBudget; it++) {
    used = it; let lab = assign(centers), st = stats(lab, centers.length), changed = false, didSplit = false;
    const keep = Array.from(st.n, (v) => v >= minSize);
    if (!keep.some(Boolean)) { let best = 0; for (let j = 1; j < st.n.length; j++) if (st.n[j] > st.n[best]) best = j; keep[best] = true; }
    if (!keep.every(Boolean)) {
      cuts += keep.filter((v) => !v).length; changed = true; centers = centers.filter((_, j) => keep[j]);
      lab = assign(centers); st = stats(lab, centers.length);
      let removed; [lab, centers, st, removed] = dropEmpty(lab, centers, st); cuts += removed;
    }
    const kk = centers.length, means = [], sigma = [];
    for (let j = 0; j < kk; j++) {
      const mean = [], sig = [];
      for (let c = 0; c < nf; c++) { const m = st.s1[c][j] / st.n[j], v = st.s2[c][j] / st.n[j] - m * m; mean.push(m); sig.push(Math.sqrt(Math.max(0, v))); }
      means.push(mean); sigma.push(sig);
    }
    centers = means.map((m) => m.slice());
    if (centers.length < maxK) {
      const k0 = centers.length;
      for (let j = 0; j < k0 && centers.length < maxK; j++) {
        let smax = sigma[j][0], cs = 0; for (let c = 1; c < nf; c++) if (sigma[j][c] > smax) { smax = sigma[j][c]; cs = c; }
        if (smax > splitStd && st.n[j] >= 2 * minSize) {
          const half = 0.5 * smax, high = means[j].slice(); high[cs] = means[j][cs] + half; centers[j][cs] = means[j][cs] - half; centers.push(high);
          splits++; didSplit = true; changed = true;
        }
      }
    }
    if (!didSplit && centers.length > floorK) {
      const pairs = [];
      for (let i = 0; i < kk; i++) for (let j = i + 1; j < kk; j++) {
        const t0 = means[i][0] - means[j][0]; let d = t0 * t0;
        if (nf === 3) { const t1 = means[i][1] - means[j][1]; d = d + t1 * t1; const t2 = means[i][2] - means[j][2]; d = d + t2 * t2; }
        const dist = Math.sqrt(d); if (dist < mergeDist) pairs.push([dist, i, j]);
      }
      pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
      const taken = new Set(), gone = new Set(); let done = 0;
      for (const [, i, j] of pairs) {
        if (done >= 2 || centers.length - gone.size <= floorK) break;
        if (taken.has(i) || taken.has(j)) continue;
        const ni = st.n[i], nj = st.n[j]; centers[i] = means[i].map((v, c) => (ni * v + nj * means[j][c]) / (ni + nj));
        gone.add(j); taken.add(i); taken.add(j); done++; merges++; changed = true;
      }
      centers = centers.filter((_, j) => !gone.has(j));
    }
    const same = prev !== null && prevK === kk && lab.every((v, u) => v === prev[u]);
    if (!changed && same) { converged = true; break; }
    prev = lab; prevK = kk;
  }
  let lab = assign(centers), st = stats(lab, centers.length), removed;
  [lab, centers, st, removed] = dropEmpty(lab, centers, st); cuts += removed;
  const means = centers.map((_, j) => feats.map((_, c) => st.s1[c][j] / st.n[j]));
  const outCh = view === "labels" ? 3 : ch, data = new Uint8Array(N * outCh);
  for (let i = 0; i < N; i++) {
    const key = nf === 1 ? pix[i] : ((pix[3 * i] << 16) | (pix[3 * i + 1] << 8) | pix[3 * i + 2]);
    let lo = 0, hi = U - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (ukey[mid] < key) lo = mid + 1; else hi = mid; }
    const j = lab[lo];
    if (view === "labels") data.set(ISODATA_PALETTE_5G[j % 12], i * 3);
    else for (let c = 0; c < nf; c++) data[i * outCh + c] = Math.floor(means[j][c] + 0.5);
  }
  return { data, channels: outCh, clusters: centers.length, used, splits, merges, cuts, converged, iterations: iterBudget };
}

export function mcclellanSynth5G(a, N) {
  const M = (N - 1) / 2, t = [0.125,0.25,0.125,0.25,-0.5,0.25,0.125,0.25,0.125];
  const total = new Float64Array(N * N), prev0 = new Float64Array(N * N);
  let prev2 = null, prev = prev0;
  prev[M * N + M] = 1;
  for (let i = 0; i < total.length; i++) total[i] += a[0] * prev[i];
  for (let n = 1; n <= M; n++) {
    const cur = new Float64Array(N * N);
    if (n === 1) {
      for (let ty = 0; ty < 3; ty++) for (let tx = 0; tx < 3; tx++) cur[(M + ty - 1) * N + M + tx - 1] = t[ty * 3 + tx];
    } else {
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        let acc = 0;
        for (let ty = 0; ty < 3; ty++) for (let tx = 0; tx < 3; tx++) {
          const sy = y - (ty - 1), sx = x - (tx - 1);
          acc += t[ty * 3 + tx] * (sy >= 0 && sy < N && sx >= 0 && sx < N ? prev[sy * N + sx] : 0);
        }
        cur[y * N + x] = 2 * acc - prev2[y * N + x];
      }
    }
    for (let i = 0; i < total.length; i++) total[i] += a[n] * cur[i];
    prev2 = prev; prev = cur;
  }
  return total;
}

export function firDesign5G(N, method, response, c1, c2, win) {
  const M = (N - 1) / 2, flo = Math.min(c1, c2), fhi = Math.max(c1, c2), Rw = M + 1;
  const round = (v) => Math.floor(v * 1e12 + 0.5) / 1e12;
  const winAt = (rho) => rho >= Rw ? 0 : win === "hamming" ? 0.54 + 0.46 * cos12Rad5G(Math.PI * rho / Rw) : win === "hann" ? 0.5 + 0.5 * cos12Rad5G(Math.PI * rho / Rw) : 1;
  const h = new Float64Array(N * N);
  if (method === "mcclellan") {
    const sinc = (x) => x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    const low = (fc, n) => round((fc * sinc(fc * n)) * winAt(Math.abs(n)));
    const a = new Float64Array(M + 1);
    for (let n = 0; n <= M; n++) {
      const delta = n === 0 ? 1 : 0;
      const h1 = response === "lowpass" ? low(c1, n) : response === "highpass" ? delta - low(c1, n) : response === "bandpass" ? low(fhi, n) - low(flo, n) : delta - (low(fhi, n) - low(flo, n));
      a[n] = n === 0 ? h1 : 2 * h1;
    }
    const raw = mcclellanSynth5G(a, N);
    for (let i = 0; i < h.length; i++) h[i] = round(raw[i]);
  } else {
    const hd = new Float64Array(N * N), cosTab = new Float64Array(N);
    for (let a = 0; a < N; a++) cosTab[a] = cos12Rad5G((2 * Math.PI * a) / N);
    for (let k2 = 0; k2 < N; k2++) for (let k1 = 0; k1 < N; k1++) {
      const f1 = ((k1 - M) * 2) / N, f2 = ((k2 - M) * 2) / N, r = Math.sqrt(f1 * f1 + f2 * f2);
      const low = r <= c1 ? 1 : 0, band = r >= flo && r <= fhi ? 1 : 0;
      hd[k2 * N + k1] = response === "lowpass" ? low : response === "highpass" ? 1 - low : response === "bandpass" ? band : 1 - band;
    }
    for (let m2 = -M; m2 <= M; m2++) for (let m1 = -M; m1 <= M; m1++) {
      let acc = 0;
      for (let k2 = 0; k2 < N; k2++) for (let k1 = 0; k1 < N; k1++) acc += hd[k2 * N + k1] * cosTab[(((k1 - M) * m1 + (k2 - M) * m2) % N + N) % N];
      const raw = acc / (N * N), at = (m2 + M) * N + m1 + M;
      h[at] = round(method === "window" ? raw * winAt(Math.sqrt(m1 * m1 + m2 * m2)) : raw);
    }
  }
  let sum = 0; for (const value of h) sum += value;
  return { N, M, h, sum };
}

export function firResponse5G(h, N) {
  const M = (N - 1) / 2, G = 64, c = new Float64Array(G), s = new Float64Array(G);
  for (let a = 0; a < G; a++) { c[a] = cos12Rad5G((2 * Math.PI * a) / G); s[a] = sin12Rad5G((2 * Math.PI * a) / G); }
  const stageR = new Float64Array(G * N), stageI = new Float64Array(G * N);
  for (let g1 = 0; g1 < G; g1++) for (let row = 0; row < N; row++) {
    const m2 = row - M; let re = 0, im = 0;
    for (let col = 0; col < N; col++) {
      const m1 = col - M, a = (((g1 - 32) * m1) % G + G) % G, hv = h[row * N + col];
      re += hv * c[a]; im -= hv * s[a];
    }
    stageR[g1 * N + row] = re; stageI[g1 * N + row] = im;
  }
  const mag = new Float64Array(G * G);
  for (let g2 = 0; g2 < G; g2++) for (let g1 = 0; g1 < G; g1++) {
    let re = 0, im = 0;
    for (let row = 0; row < N; row++) {
      const m2 = row - M, a = (((g2 - 32) * m2) % G + G) % G, ar = stageR[g1 * N + row], ai = stageI[g1 * N + row];
      re += ar * c[a] + ai * s[a]; im += ai * c[a] - ar * s[a];
    }
    mag[g2 * G + g1] = Math.sqrt(re * re + im * im);
  }
  return mag;
}

export function firFilter5G(plane, width, height, h, N, add128) {
  const M = (N - 1) / 2, rx = new Int32Array(width + 2 * M), ry = new Int32Array(height + 2 * M);
  for (let x = 0; x < rx.length; x++) rx[x] = reflect101_5G(x - M, width);
  for (let y = 0; y < ry.length; y++) ry[y] = reflect101_5G(y - M, height);
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let acc = 0;
    for (let dy = -M; dy <= M; dy++) for (let dx = -M; dx <= M; dx++) {
      const value = plane[ry[y + M + dy] * width + rx[x + M + dx]];
      acc += h[(dy + M) * N + dx + M] * value;
    }
    const value = Math.floor((add128 ? acc + 128 : acc) + 0.5);
    out[y * width + x] = Math.min(255, Math.max(0, value));
  }
  return out;
}

export function firResponseImage5G(mag) {
  let scale = 1; for (const value of mag) if (value > scale) scale = value;
  const out = new Uint8Array(256 * 256);
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) {
    const g2 = 63 - Math.floor(y / 4), g1 = Math.floor(x / 4);
    out[y * 256 + x] = Math.floor(mag[g2 * 64 + g1] / scale * 255 + 0.5);
  }
  return out;
}

export function firKernelImage5G(h, N) {
  const q = Math.max(1, Math.floor(256 / N)), side = N * q;
  let maxAbs = 0; for (const value of h) if (Math.abs(value) > maxAbs) maxAbs = Math.abs(value);
  const out = new Uint8Array(side * side);
  for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
    const value = maxAbs > 0 ? Math.floor(128 + (127 * h[Math.floor(y / q) * N + Math.floor(x / q)]) / maxAbs + 0.5) : 128;
    out[y * side + x] = Math.min(255, Math.max(0, value));
  }
  return { data: out, width: side, height: side, scale: q };
}

export function firKernelSize5G(width, height, ch, N, workBudget) {
  let result = N;
  while (result > 3 && result * result * width * height * ch > 25 * workBudget) result -= 2;
  return result;
}

export function exemplar5G(data, width, height, ch, maskData, patch, search, workCap) {
  const P = patch, r = (P - 1) / 2, n = width * height, current = Uint8Array.from(data), omega0 = new Uint8Array(n), omega = new Uint8Array(n);
  let nz = 0, bx0 = width, bx1 = -1, by0 = height, by1 = -1;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (maskData[y * width + x]) {
    const i = y * width + x; omega0[i] = omega[i] = 1; nz++;
    if (x < bx0) bx0 = x; if (x > bx1) bx1 = x; if (y < by0) by0 = y; if (y > by1) by1 = y;
  }
  const stride = width + 1, sat = new Int32Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) sat[(y + 1) * stride + x + 1] = omega0[y * width + x] + sat[y * stride + x + 1] + sat[(y + 1) * stride + x] - sat[y * stride + x];
  const candidates = (s) => {
    const out = [];
    const cy0 = Math.max(r, by0 - s), cy1 = Math.min(height - 1 - r, by1 + s);
    const cx0 = Math.max(r, bx0 - s), cx1 = Math.min(width - 1 - r, bx1 + s);
    if (cy1 < cy0 || cx1 < cx0) return out;
    for (let cy = cy0; cy <= cy1; cy++) for (let cx = cx0; cx <= cx1; cx++) {
      const y0 = Math.max(0, cy - r), y1 = Math.min(height, cy + r + 1), x0 = Math.max(0, cx - r), x1 = Math.min(width, cx + r + 1);
      if (y1 <= y0 || x1 <= x0) continue;
      const a = y1 * stride + x1, b = y0 * stride + x1, c = y1 * stride + x0, d = y0 * stride + x0;
      if (sat[a] - sat[b] - sat[c] + sat[d] === 0) out.push(cy * width + cx);
    }
    return out;
  };
  let s = search, Q = candidates(s);
  if (Q.length === 0) { s = Math.max(width, height); Q = candidates(s); }
  if (Q.length === 0) return { status: "fallback" };
  const pp = P * P, estimate = (qCount) => Math.ceil(nz / Math.max(1, Math.floor(pp / 4))) * qCount * pp * ch;
  let reduced = false;
  while (estimate(Q.length) > workCap) {
    s = Math.floor(s / 2);
    if (s < r + 1) return { status: "refuse" };
    const next = candidates(s);
    if (next.length === 0) return { status: "refuse" };
    Q = next; reduced = true;
  }
  const offsets = [];
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) offsets.push([dy, dx]);
  const confidence = new Float64Array(n); for (let i = 0; i < n; i++) confidence[i] = omega[i] ? 0 : 1;
  const igr = new Int32Array(n);
  for (let i = 0; i < n; i++) { let sum = 0; for (let c = 0; c < ch; c++) sum += current[i * ch + c]; igr[i] = sum; }
  let left = nz, iterations = 0;
  const inside = (x, y) => x >= 0 && x < width && y >= 0 && y < height;
  while (left > 0) {
    const front = [];
    const fy0 = Math.max(0, by0 - 1), fy1 = Math.min(height - 1, by1 + 1), fx0 = Math.max(0, bx0 - 1), fx1 = Math.min(width - 1, bx1 + 1);
    for (let y = fy0; y <= fy1; y++) for (let x = fx0; x <= fx1; x++) {
      const i = y * width + x;
      if (!omega[i]) continue;
      if ((y > 0 && !omega[i - width]) || (y + 1 < height && !omega[i + width]) || (x > 0 && !omega[i - 1]) || (x + 1 < width && !omega[i + 1])) front.push(i);
    }
    let bestPriority = -Infinity, bestIndex = -1, bestConfidence = 0;
    for (const i of front) {
      const x = i % width, y = Math.floor(i / width); let acc = 0, count = 0;
      for (const [dy, dx] of offsets) { const qx = x + dx, qy = y + dy; if (inside(qx, qy)) { acc += confidence[qy * width + qx]; count++; } }
      const cp = acc / count;
      const omL = x > 0 && omega[i - 1] ? 1 : 0, omR = x + 1 < width && omega[i + 1] ? 1 : 0;
      const omT = y > 0 && omega[i - width] ? 1 : 0, omB = y + 1 < height && omega[i + width] ? 1 : 0;
      const nx = omR - omL, ny = omB - omT, nlen = Math.sqrt(nx * nx + ny * ny);
      let bestG2 = -1, gx = 0, gy = 0;
      for (const [dy, dx] of offsets) {
        const qx = x + dx, qy = y + dy;
        if (!inside(qx, qy) || !inside(qx - 1, qy) || !inside(qx + 1, qy) || !inside(qx, qy - 1) || !inside(qx, qy + 1)) continue;
        const qi = qy * width + qx;
        if (omega[qi] || omega[qi - 1] || omega[qi + 1] || omega[qi - width] || omega[qi + width]) continue;
        const gxx = igr[qi + 1] - igr[qi - 1], gyy = igr[qi + width] - igr[qi - width], g2 = gxx * gxx + gyy * gyy;
        if (g2 > bestG2) { bestG2 = g2; gx = gxx; gy = gyy; }
      }
      const D = nlen > 0 ? Math.abs((0 - gy) * nx + gx * ny) / (nlen * 2 * 255 * ch) : 0;
      const priority = cp * (D + 0.001);
      if (priority > bestPriority) { bestPriority = priority; bestIndex = i; bestConfidence = cp; }
    }
    if (bestIndex < 0) return { status: "fallback" };
    const px = bestIndex % width, py = Math.floor(bestIndex / width);
    let bestSSD = Infinity, bestQ = Q[0];
    for (const candidate of Q) {
      const qx = candidate % width, qy = Math.floor(candidate / width); let ssd = 0;
      for (const [dy, dx] of offsets) {
        const x = px + dx, y = py + dy;
        if (!inside(x, y) || omega[y * width + x]) continue;
        const sx = qx + dx, sy = qy + dy, a = (y * width + x) * ch, b = (sy * width + sx) * ch;
        for (let c = 0; c < ch; c++) { const diff = current[a + c] - current[b + c]; ssd += diff * diff; }
      }
      if (ssd < bestSSD) { bestSSD = ssd; bestQ = candidate; }
    }
    const qx = bestQ % width, qy = Math.floor(bestQ / width);
    for (const [dy, dx] of offsets) {
      const x = px + dx, y = py + dy, sx = qx + dx, sy = qy + dy;
      if (!inside(x, y)) continue;
      const i = y * width + x;
      if (!omega[i]) continue;
      const src = (sy * width + sx) * ch, dst = i * ch;
      for (let c = 0; c < ch; c++) current[dst + c] = current[src + c];
      let sum = 0; for (let c = 0; c < ch; c++) sum += current[dst + c];
      igr[i] = sum; confidence[i] = bestConfidence; omega[i] = 0; left--;
    }
    iterations++;
  }
  return { status: "ok", data: current, iterations, search: s, requestedSearch: search, reduced, nz };
}

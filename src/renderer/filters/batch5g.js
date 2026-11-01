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

// Renders a photographic die micrograph, pixel by pixel, off the main thread. Each block of
// the floorplan gets the texture its circuitry has under a microscope: standard-cell rows for
// logic, fine periodic arrays for SRAM, tiled MAC arrays, analog PHYs, metal fill and the pad
// ring. A thin oxide film, varying slowly in thickness, tints everything with the
// interference colours a real die shows, and a raking light adds the sheen.

const hash = (x, y) => {
  let h = (x * 374761393 + y * 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
};
const smooth = (t) => t * t * (3 - 2 * t);
function noise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1);
  const u = smooth(xf), v = smooth(yf);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
const fbm = (x, y) => noise(x, y) * 0.55 + noise(x * 2.1, y * 2.1) * 0.28 + noise(x * 4.3, y * 4.3) * 0.17;

// Colours a die shows under a microscope as the oxide over it thickens: violet, magenta,
// copper, gold, olive, teal and blue, in that order, repeating.
const PAL = [
  [58, 40, 96], [112, 58, 112], [150, 92, 60], [190, 150, 70], [118, 128, 62], [52, 118, 112], [56, 82, 148], [58, 40, 96],
].map((c) => c.map((v) => v / 255));
function film(t, out) {
  t = ((t % 1) + 1) % 1;
  const x = t * (PAL.length - 1), i = Math.floor(x), f = smooth(x - i);
  for (let k = 0; k < 3; k++) out[k] = PAL[i][k] + (PAL[i + 1][k] - PAL[i][k]) * f;
}

// Splits a block into the macros a placed design has: logic partitions of different density
// and small SRAM arrays such as register files and buffers, with routing gutters between.
function macros(seed, w, h) {
  let st = seed >>> 0;
  const rnd = () => { st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296; };
  const out = [];
  const split = (x, y, w, h, d) => {
    if (d === 0 || (w < 24 && h < 24)) {
      out.push({ x, y, w, h, sram: rnd() < 0.24, fo: (rnd() - 0.5) * 0.08, k: 0.82 + rnd() * 0.36 });
      return;
    }
    const vert = w > h ? rnd() < 0.8 : rnd() < 0.2;
    const r = 0.3 + rnd() * 0.4;
    if (vert) { const a = Math.round(w * r); split(x, y, a, h, d - 1); split(x + a, y, w - a, h, d - 1); }
    else { const a = Math.round(h * r); split(x, y, w, a, d - 1); split(x, y + a, w, h - a, d - 1); }
  };
  split(0, 0, w, h, w * h > 20000 ? 4 : 3);
  return out;
}

self.onmessage = ({ data }) => {
  const { size, blocks } = data;
  // Cores of one kind are copies of one design, so they share a layout; other blocks differ.
  const seedOf = (k, i) => (k.m === "pcore" || k.m === "ecore" || k.m === "gpu") ? k.m.length * 104729 : i * 7919 + 13;
  const parts = blocks.map((k, i) => (k.m === "sram" || k.m === "npu" || k.m === "phy") ? null : macros(seedOf(k, i), Math.round(k.w * size), Math.round(k.h * size)));
  const px = new Uint8ClampedArray(size * size * 4);
  const fc = [0, 0, 0];
  const S = size;
  // Feature scale: about one standard-cell row every 3 pixels at this size.
  const row = Math.max(2, Math.round(S / 560));
  const grid = Math.round(S / 38);

  for (let y = 0; y < S; y++) {
    const v = y / S;
    for (let x = 0; x < S; x++) {
      const u = x / S;
      let b = 0, fo = 0, sat = 0.75;
      const edge = Math.min(u, v, 1 - u, 1 - v);

      if (edge < 0.012) {
        // Scribe line and seal ring: bare silicon with a bright metal ring.
        b = edge > 0.006 && edge < 0.009 ? 0.85 : 0.18 + 0.06 * hash(x, y);
        fo = 0.15; sat = 0.3;
      } else if (edge < 0.04) {
        // Pad ring: aluminium bond pads with probe scratches, ESD cells between them.
        const along = (u < 0.04 || u > 0.96) ? v : u;
        const cell = (along * 34) % 1;
        const depth = (edge - 0.012) / 0.028;
        if (cell > 0.16 && cell < 0.84 && depth > 0.14 && depth < 0.8) {
          // Aluminium pad, brushed, with a probe mark near its middle.
          b = 0.72 + 0.12 * noise(x * 0.9, y * 0.05) + 0.06 * (1 - depth);
          const pm = Math.hypot(cell - 0.5, depth - 0.5);
          if (pm < 0.07) b *= 0.78;
          fo = 0.0; sat = 0.06;
        } else if (cell > 0.1 && cell < 0.9) {
          b = 0.2 + 0.18 * hash(Math.floor(x / 2), Math.floor(y / 2));
          fo = 0.15; sat = 0.6;
        } else {
          b = 0.12; fo = 0.2; sat = 0.4;
        }
      } else {
        let blk = null;
        let bi = -1;
        for (let i = 0; i < blocks.length; i++) {
          const k = blocks[i];
          if (u >= k.x && u < k.x + k.w && v >= k.y && v < k.y + k.h) { blk = k; bi = i; break; }
        }
        if (!blk) {
          // Routing channels and metal fill between blocks.
          const fill = (x % 6 < 3) && (y % 6 < 3);
          const bus = (y % (row * 9)) < 1 || (x % (row * 13)) < 1;
          b = bus ? 0.62 : fill ? 0.36 : 0.22;
          fo = 0.55 + 0.1 * noise(u * 9, v * 9);
        } else {
          const lx = (u - blk.x) * S, ly = (v - blk.y) * S, w = blk.w * S, h = blk.h * S;
          const ex = Math.min(lx, w - lx, ly, h - ly);
          let m = blk.m, mk = 1, mfo = 0;
          // Cores carry their own L2 strip along one side.
          if ((m === "pcore" || m === "ecore") && lx < w * 0.3) m = "sram";
          else if (parts[bi]) {
            for (const q of parts[bi]) {
              if (lx >= q.x && lx < q.x + q.w && ly >= q.y && ly < q.y + q.h) {
                if (lx - q.x < 1.2 || ly - q.y < 1.2) mk = 0.45;
                else { mk = q.k; mfo = q.fo; if (q.sram) m = "sram"; }
                break;
              }
            }
          }
          if (m === "sram") {
            // Bit-cell arrays: a fine lattice, split into sub-arrays by wordline drivers.
            const sx = x % 2 === 0 ? 0.86 : 1, sy = y % 3 === 0 ? 0.84 : 1;
            const sub = (lx % (row * 46)) < row * 2 || (ly % (row * 30)) < row;
            b = sub ? 0.32 + 0.2 * hash(Math.floor(lx / 2), Math.floor(ly / 3)) : 0.66 * sx * sy;
            fo = blk.f + 0.04 * fbm(lx / 90, ly / 90); sat = 0.95;
          } else if (m === "npu") {
            // MAC arrays: identical tiles in a strict grid.
            const tx = lx % (row * 6), ty = ly % (row * 6);
            b = tx < row || ty < row ? 0.3 : 0.55 + 0.25 * ((Math.floor(tx / 2) + Math.floor(ty / 2)) % 2);
            fo = blk.f;
          } else if (m === "phy") {
            // Analog PHY: large repeated drivers and capacitor banks.
            const lane = lx % (row * 16);
            const cap = (ly % (row * 10)) < row * 6 && lane > row * 3 && lane < row * 13;
            b = lane < row * 2 ? 0.85 : cap ? 0.5 + 0.12 * ((Math.floor(lx) + Math.floor(ly)) % 2) : 0.3;
            fo = blk.f + 0.25; sat = 0.6;
          } else {
            // Standard-cell logic: rows of cells of random widths, power rails between rows,
            // vertical straps, and placement density that varies across the block.
            const r = Math.floor(ly / row);
            const rail = ly % row < 1 && r % 2 === 0;
            const cw = 1 + Math.floor(hash(r, 7) * 3);
            const c = Math.floor((lx + r * 3) / cw);
            const dens = fbm(lx / (S * 0.04) + blk.x * 40, ly / (S * 0.04));
            const strap = lx % (row * 18) < 1;
            b = rail || strap ? 0.62 : 0.3 + 0.32 * hash(c, r) * (0.4 + dens * 0.9);
            fo = blk.f + 0.06 * (hash(c, r + 99) > 0.93 ? 1 : 0);
          }
          b *= mk; fo += mfo;
          if (ex < 1.5) b *= 0.35;          // the cut between blocks
          else if (ex < 3) b = Math.min(1, b * 1.25); // the block's own ring
        }
      }

      // The top-metal power grid runs across the whole core area, over every block.
      if (edge >= 0.04 && ((x % grid) < 1.5 || (y % grid) < 1.5)) { b = b * 0.75 + 0.14; sat *= 0.85; }

      // Oxide thickness drifts slowly across the die; each block's stack shifts it.
      const t = 0.08 + 0.16 * fbm(u * 1.6 + 3.1, v * 1.6 + 1.7) + fo;
      film(t, fc);
      const lum = b;
      // Raking light from the top left, falling off toward the far corner.
      // Raking light: a broad glossy band across the die and a gentle fall-off to the corners.
      const band = Math.exp(-Math.pow((u * 0.75 + v * 0.65 - 0.62) / 0.2, 2));
      const vig = 1 - 0.35 * Math.pow(Math.hypot(u - 0.45, v - 0.42) / 0.75, 2);
      const light = (0.72 + 0.55 * band) * vig;
      const i = (y * S + x) * 4;
      for (let k = 0; k < 3; k++) {
        const col = lum * ((1 - sat) * 0.8 + sat * fc[k] * 1.55);
        px[i + k] = 255 * Math.min(1, col * light);
      }
      px[i + 3] = 255;
    }
  }
  self.postMessage({ size, px }, [px.buffer]);
};

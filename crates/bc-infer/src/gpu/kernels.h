// The engine's GPU kernels, shared by Metal and CUDA. Each backend prepends a few macros
// (address spaces, warp sums, barriers) and its own entry points; everything here is
// plain C both compilers accept.
//
// Thread groups are 32 threads, one SIMD group (Apple) or warp (NVIDIA), so reductions
// need no shared memory. Weights stay in their ggml block format; every format is read
// 32 weights at a time ("groups"), which is how the matrix product spreads a row over the
// 32 lanes: lane l takes groups l, l + 32, ...
//
// Accumulation order is fixed (no atomics), so a kernel gives the same bits every run,
// and a row's result never depends on how many other rows share the batch: speculative
// checks of several tokens score each one exactly as a single-token pass would.

#define T_F32 0u
#define T_F16 1u
#define T_Q4_0 2u
#define T_Q4_1 3u
#define T_Q5_0 6u
#define T_Q5_1 7u
#define T_Q8_0 8u
#define T_Q4K 12u
#define T_Q5K 13u
#define T_Q6K 14u
#define T_BF16 30u

// Input rows a matrix-product thread group computes at once.
#define NB 8
// Positions of attention scores held at a time, per thread group.
#define TILE 1024
// Largest head dimension (8 values per lane).
#define MAX_HEAD 256

DEVFN uint u16at(GLOBAL const uchar* b, uint i) { return (uint)b[i] | ((uint)b[i + 1] << 8); }
DEVFN uint u32at(GLOBAL const uchar* b, uint i) { return u16at(b, i) | (u16at(b, i + 2) << 16); }
DEVFN float f16at(GLOBAL const uchar* b, uint i) { return H2F(u16at(b, i)); }
DEVFN int i8of(uint b) { return (int)(b ^ 128u) - 128; }

// The 6-bit scale and min of group j in a K-quant block (ggml's get_scale_min_k4).
DEVFN void scale_min_k4(uint j, GLOBAL const uchar* q, PRIV float* sc, PRIV float* mn) {
    if (j < 4u) {
        *sc = (float)(q[j] & 63u);
        *mn = (float)(q[j + 4] & 63u);
    } else {
        *sc = (float)((q[j + 4] & 15u) | ((q[j - 4] >> 6) << 4));
        *mn = (float)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Weight `lane` (0..32) of group g of a stored row, one at a time: for the formats
// group4 has no wide path for. Each follows quant.rs's dequant_block.
DEVFN float weight(GLOBAL const uchar* row, uint ty, uint g, uint lane) {
    uint j = lane & 15u, upper = lane >> 4;
    if (ty == T_F32) return BITS2F(u32at(row, (g * 32u + lane) * 4u));
    if (ty == T_F16) return f16at(row, (g * 32u + lane) * 2u);
    if (ty == T_BF16) return BITS2F(u16at(row, (g * 32u + lane) * 2u) << 16);
    if (ty == T_Q4_0 || ty == T_Q4_1) {
        GLOBAL const uchar* b = row + g * (ty == T_Q4_0 ? 18u : 20u);
        uint q = (b[(ty == T_Q4_0 ? 2u : 4u) + j] >> (4u * upper)) & 15u;
        return ty == T_Q4_0 ? ((float)q - 8.0f) * f16at(b, 0) : (float)q * f16at(b, 0) + f16at(b, 2);
    }
    if (ty == T_Q5_0 || ty == T_Q5_1) {
        GLOBAL const uchar* b = row + g * (ty == T_Q5_0 ? 22u : 24u);
        uint at = ty == T_Q5_0 ? 2u : 4u;
        uint qh = u32at(b, at);
        uint h = upper == 0u ? ((qh >> j) << 4) & 16u : (qh >> (j + 12u)) & 16u;
        uint q = ((b[at + 4 + j] >> (4u * upper)) & 15u) | h;
        return ty == T_Q5_0 ? ((float)q - 16.0f) * f16at(b, 0) : (float)q * f16at(b, 0) + f16at(b, 2);
    }
    if (ty == T_Q8_0) {
        GLOBAL const uchar* b = row + g * 34u;
        return (float)i8of(b[2 + lane]) * f16at(b, 0);
    }
    if (ty == T_Q4K || ty == T_Q5K) {
        // 256 weights a block in 4 chunks of 64: low nibbles are group 2c, high 2c + 1.
        GLOBAL const uchar* b = row + (g / 8u) * (ty == T_Q4K ? 144u : 176u);
        uint sub = g % 8u, c = sub / 2u, hi = sub % 2u;
        float sc, mn;
        scale_min_k4(sub, b + 4, &sc, &mn);
        uint q = (b[(ty == T_Q4K ? 16u : 48u) + 32u * c + lane] >> (4u * hi)) & 15u;
        if (ty == T_Q5K && (b[16 + lane] & ((1u + hi) << (2u * c))) != 0u) q += 16u;
        return f16at(b, 0) * sc * (float)q - f16at(b, 2) * mn;
    }
    if (ty == T_Q6K) {
        GLOBAL const uchar* b = row + (g / 8u) * 210u;
        uint sub = g % 8u, hh = sub / 4u, qr = sub % 4u;
        uint lo = b[64u * hh + ((qr & 1u) != 0u ? 32u : 0u) + lane];
        uint q4 = (lo >> (qr < 2u ? 0u : 4u)) & 15u;
        uint q2 = (b[128u + 32u * hh + lane] >> (2u * qr)) & 3u;
        float sc = (float)i8of(b[192u + 8u * hh + 2u * qr + upper]);
        return f16at(b, 208) * sc * (float)((int)(q4 | (q2 << 4)) - 32);
    }
    return 0.0f;
}

DEVFN float4 x4(GLOBAL const float* xg, uint k) { return ((GLOBAL const float4*)xg)[k]; }
DEVFN float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
// The four weights in the bytes of v (each already masked to its value), minus off.
DEVFN float4 bytes4(uint v, float off) {
    return F4((float)(v & 255u) - off, (float)((v >> 8) & 255u) - off, (float)((v >> 16) & 255u) - off, (float)(v >> 24) - off);
}
DEVFN float4 scale4(float4 v, float s) { return F4(v.x * s, v.y * s, v.z * s, v.w * s); }

// Weights 4k .. 4k+3 of group g of a stored row (k in 0..8). The formats real models use
// most read two or four weight bytes per load; the rest go through weight(). Each
// follows quant.rs's dequant_block.
DEVFN float4 four(GLOBAL const uchar* row, uint ty, uint g, uint k) {
    if (ty == T_Q4_0 || ty == T_Q5_0) {
        // Bytes 4j .. 4j+3 hold weights 4j.. in their low nibbles and 16 + 4j.. in high.
        uint j = k & 3u, hi = k >> 2;
        uint at = ty == T_Q4_0 ? 2u : 6u;
        GLOBAL const uchar* b = row + g * (ty == T_Q4_0 ? 18u : 22u);
        GLOBAL const ushort* q = (GLOBAL const ushort*)(b + at);
        uint v = ((q[2 * j] | ((uint)q[2 * j + 1] << 16)) >> (4u * hi)) & 0x0f0f0f0fu;
        if (ty == T_Q4_0) return scale4(bytes4(v, 8.0f), f16at(b, 0));
        // Bit t of these four bits goes to bit 4 of byte t.
        uint h = (u32at(b, 2) >> (4u * j + 16u * hi)) & 15u;
        v |= ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
        return scale4(bytes4(v, 16.0f), f16at(b, 0));
    }
    if (ty == T_Q8_0) {
        GLOBAL const uchar* b = row + g * 34u;
        GLOBAL const ushort* q = (GLOBAL const ushort*)(b + 2);
        uint a = q[2 * k], c = q[2 * k + 1];
        float d = f16at(b, 0);
        return F4((float)i8of(a & 255u) * d, (float)i8of(a >> 8) * d, (float)i8of(c & 255u) * d, (float)i8of(c >> 8) * d);
    }
    if (ty == T_Q4K || ty == T_Q5K) {
        // 256 weights a block in 4 chunks of 64: low nibbles are group 2c, high 2c + 1.
        GLOBAL const uchar* b = row + (g / 8u) * (ty == T_Q4K ? 144u : 176u);
        uint sub = g % 8u, c = sub / 2u, hi = sub % 2u;
        float sc, mn;
        scale_min_k4(sub, b + 4, &sc, &mn);
        float ds = f16at(b, 0) * sc, dm = f16at(b, 2) * mn;
        uint v = (((GLOBAL const uint*)(b + (ty == T_Q4K ? 16u : 48u) + 32u * c))[k] >> (4u * hi)) & 0x0f0f0f0fu;
        if (ty == T_Q5K) v |= ((((GLOBAL const uint*)(b + 16))[k] >> (hi + 2u * c)) & 0x01010101u) << 4;
        float4 f = bytes4(v, 0.0f);
        return F4(ds * f.x - dm, ds * f.y - dm, ds * f.z - dm, ds * f.w - dm);
    }
    if (ty == T_Q6K) {
        GLOBAL const uchar* b = row + (g / 8u) * 210u;
        uint sub = g % 8u, hh = sub / 4u, qr = sub % 4u;
        GLOBAL const ushort* ql = (GLOBAL const ushort*)(b + 64u * hh + ((qr & 1u) != 0u ? 32u : 0u));
        GLOBAL const ushort* qh = (GLOBAL const ushort*)(b + 128u + 32u * hh);
        uint lo = ql[2 * k] | ((uint)ql[2 * k + 1] << 16), h = qh[2 * k] | ((uint)qh[2 * k + 1] << 16);
        uint v = ((lo >> (qr < 2u ? 0u : 4u)) & 0x0f0f0f0fu) | (((h >> (2u * qr)) & 0x03030303u) << 4);
        float d = f16at(b, 208) * (float)i8of(b[192u + 8u * hh + 2u * qr + (k < 4u ? 0u : 1u)]);
        return scale4(bytes4(v, 32.0f), d);
    }
    return F4(weight(row, ty, g, 4 * k), weight(row, ty, g, 4 * k + 1), weight(row, ty, g, 4 * k + 2), weight(row, ty, g, 4 * k + 3));
}

// One group of 32 weights (g), decoded once into w[k] = weights 4k .. 4k+3: each block's
// scales are read once, its weight bytes two or four at a time.
DEVFN void group4(GLOBAL const uchar* row, uint ty, uint g, PRIV float4* w) {
    if (ty == T_Q4_0) {
        GLOBAL const uchar* b = row + g * 18u;
        GLOBAL const ushort* q = (GLOBAL const ushort*)(b + 2);
        float d = f16at(b, 0);
        for (uint k = 0; k < 4u; k++) {
            uint v = q[2 * k] | ((uint)q[2 * k + 1] << 16);
            w[k] = scale4(bytes4(v & 0x0f0f0f0fu, 8.0f), d);
            w[k + 4] = scale4(bytes4((v >> 4) & 0x0f0f0f0fu, 8.0f), d);
        }
    } else if (ty == T_Q4K || ty == T_Q5K) {
        GLOBAL const uchar* b = row + (g / 8u) * (ty == T_Q4K ? 144u : 176u);
        uint sub = g % 8u, c = sub / 2u, hi = sub % 2u;
        float sc, mn;
        scale_min_k4(sub, b + 4, &sc, &mn);
        float ds = f16at(b, 0) * sc, dm = f16at(b, 2) * mn;
        GLOBAL const uint* q = (GLOBAL const uint*)(b + (ty == T_Q4K ? 16u : 48u) + 32u * c);
        GLOBAL const uint* qh = (GLOBAL const uint*)(b + 16);
        // The fifth bits' mask is zero for Q4K: a branch here, per load, halves the speed.
        uint fifth = ty == T_Q5K ? 0x01010101u : 0u;
        for (uint k = 0; k < 8u; k++) {
            uint v = ((q[k] >> (4u * hi)) & 0x0f0f0f0fu) | (((qh[k] >> (hi + 2u * c)) & fifth) << 4);
            float4 f = bytes4(v, 0.0f);
            w[k] = F4(ds * f.x - dm, ds * f.y - dm, ds * f.z - dm, ds * f.w - dm);
        }
    } else if (ty == T_Q6K) {
        GLOBAL const uchar* b = row + (g / 8u) * 210u;
        uint sub = g % 8u, hh = sub / 4u, qr = sub % 4u;
        GLOBAL const ushort* ql = (GLOBAL const ushort*)(b + 64u * hh + ((qr & 1u) != 0u ? 32u : 0u));
        GLOBAL const ushort* qh = (GLOBAL const ushort*)(b + 128u + 32u * hh);
        GLOBAL const uchar* sc = b + 192u + 8u * hh + 2u * qr;
        float d = f16at(b, 208);
        float d0 = d * (float)i8of(sc[0]), d1 = d * (float)i8of(sc[1]);
        uint shift = qr < 2u ? 0u : 4u;
        for (uint k = 0; k < 8u; k++) {
            uint lo = ql[2 * k] | ((uint)ql[2 * k + 1] << 16), h = qh[2 * k] | ((uint)qh[2 * k + 1] << 16);
            uint v = ((lo >> shift) & 0x0f0f0f0fu) | (((h >> (2u * qr)) & 0x03030303u) << 4);
            w[k] = scale4(bytes4(v, 32.0f), k < 4u ? d0 : d1);
        }
    } else {
        for (uint k = 0; k < 8u; k++) w[k] = four(row, ty, g, k);
    }
}

// Weight rows per matrix-product thread group: one per SIMD group of 32.
#define ROWS_PER_GROUP 4

// out[i][r] = Σ_c W[r][c] · x[i][c], for input rows i0 .. i0 + NB of n. Lane l takes
// groups l, l + 32, ... of row r; each group is decoded once and used for every input
// row, so checking a batch of guesses reads the weights no more than one token does.
DEVFN void matmul(GLOBAL const uchar* wt, GLOBAL const float* x, GLOBAL float* out, uint ty, uint rows, uint cols, uint n, uint row_bytes, uint r, uint i0, uint lane) {
    if (r >= rows) return;
    GLOBAL const uchar* row = wt + (ulong)r * row_bytes;
    uint groups = cols / 32u;
    uint cnt = n - i0 < NB ? n - i0 : NB;
    // Eight named accumulators, not an array: an array indexed in a loop the compiler
    // does not unroll lives in slow per-thread memory.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        float4 w[8];
        group4(row, ty, g, w);
        GLOBAL const float* xg = x + (ulong)i0 * cols + g * 32u;
#define ROW(i, a) if (i < cnt) { GLOBAL const float* xi = xg + (ulong)i * cols; float s = 0.0f; for (uint k = 0; k < 8u; k++) s += dot4(w[k], x4(xi, k)); a += s; }
        ROW(0u, a0) ROW(1u, a1) ROW(2u, a2) ROW(3u, a3) ROW(4u, a4) ROW(5u, a5) ROW(6u, a6) ROW(7u, a7)
#undef ROW
    }
#define OUT(i, a) { float s = WSUM(a); if (lane == 0u && i < cnt) out[(ulong)(i0 + i) * rows + r] = s; }
    OUT(0u, a0) OUT(1u, a1) OUT(2u, a2) OUT(3u, a3) OUT(4u, a4) OUT(5u, a5) OUT(6u, a6) OUT(7u, a7)
#undef OUT
}

// One row of RMSNorm: x / √(mean(x²) + ε) ⊙ w.
DEVFN void rms_norm(GLOBAL const float* x, GLOBAL const float* w, GLOBAL float* out, uint cols, float eps, uint r, uint lane) {
    GLOBAL const float* xr = x + (ulong)r * cols;
    float s = 0.0f;
    for (uint c = lane; c < cols; c += 32u) s += xr[c] * xr[c];
    s = WSUM(s);
    float scale = 1.0f / SQRT(s / (float)cols + eps);
    for (uint c = lane; c < cols; c += 32u) out[(ulong)r * cols + c] = xr[c] * scale * w[c];
}

// Rotates pair i of every head of row r by (pos + r)·θᵢ.
// positions[r] is row r's position: pos + r in a line, pos + depth in a tree.
DEVFN void rope(GLOBAL float* x, GLOBAL const float* inv_freq, GLOBAL const uint* positions, uint heads, uint head_dim, uint rope_dim, uint neox, uint idx) {
    uint hh = rope_dim / 2u;
    uint i = idx % hh, h = (idx / hh) % heads, r = idx / (hh * heads);
    GLOBAL float* hp = x + ((ulong)r * heads + h) * head_dim;
    float theta = (float)positions[r] * inv_freq[i];
    float sn = SIN(theta), cs = COS(theta);
    uint a = neox != 0u ? i : 2u * i;
    uint b = neox != 0u ? i + hh : 2u * i + 1u;
    float x0 = hp[a], x1 = hp[b];
    hp[a] = x0 * cs - x1 * sn;
    hp[b] = x0 * sn + x1 * cs;
}

// Causal grouped-query attention for one head of one new row, against the cache, with a
// running ("online") softmax over tiles of TILE positions: m is the largest score so far,
// l the sum of e^(s − m), and acc the weighted sum of values, all rescaled by e^(m − m')
// when a tile raises the maximum (Milakov and Gimelshein 2018; FlashAttention).
// The k-th of the batch rows a tree row sees: the k-th set bit of its mask.
DEVFN uint nth_bit(uint lo, uint hi, uint k) {
    for (uint b = 0; b < 64u; b++) {
        uint set = b < 32u ? (lo >> b) & 1u : (hi >> (b - 32u)) & 1u;
        if (set != 0u) {
            if (k == 0u) return b;
            k -= 1u;
        }
    }
    return 0u;
}

// For a tree (masks != 0), pos is the tree's first slot: row r sees the cache before it
// and the tree slots its mask (two words per row, bit j is slot pos + j) names, walked in
// the same order as a line would walk them: list index k is cache slot k below pos, then
// the (k - pos)-th visible slot. So a tree row is
// summed in exactly the order a one-path pass would sum it, bit for bit.
DEVFN void attention(GLOBAL const float* q, GLOBAL const float* kc, GLOBAL const float* vc, GLOBAL float* out, LOCAL float* sh,
                     GLOBAL const uint* masks, uint tree,
                     uint n_head, uint n_head_kv, uint head_dim, uint pos, float scale, uint h, uint r, uint lane) {
    uint kvd = n_head_kv * head_dim;
    uint kvh = h / (n_head / n_head_kv);
    uint lo = tree != 0u ? masks[2u * r] : 0u, hi = tree != 0u ? masks[2u * r + 1u] : 0u;
    uint upto = tree != 0u ? pos + POPC(lo) + POPC(hi) : pos + r + 1u;
#define SLOT(k) ((tree == 0u || (k) < pos) ? (k) : pos + nth_bit(lo, hi, (k) - pos))
    GLOBAL const float* qh = q + ((ulong)r * n_head + h) * head_dim;
    float acc[MAX_HEAD / 32];
    for (uint j = 0; j < MAX_HEAD / 32u; j++) acc[j] = 0.0f;
    float m = NEG_INF, l = 0.0f;
    for (uint start = 0; start < upto; start += TILE) {
        uint end = start + TILE < upto ? start + TILE : upto;
        float tmax = NEG_INF;
        for (uint t = start + lane; t < end; t += 32u) {
            GLOBAL const float* kt = kc + (ulong)SLOT(t) * kvd + kvh * head_dim;
            float s = 0.0f;
            for (uint d = 0; d < head_dim; d++) s += qh[d] * kt[d];
            s *= scale;
            sh[t - start] = s;
            tmax = s > tmax ? s : tmax;
        }
        tmax = WMAX(tmax);
        float mn = m > tmax ? m : tmax;
        // A tile with nothing visible yet (a tree row's first tile is never one) keeps m.
        float corr = mn == NEG_INF ? 1.0f : EXP(m - mn);
        BARRIER();
        float part = 0.0f;
        for (uint t = start + lane; t < end; t += 32u) {
            float p = EXP(sh[t - start] - mn);
            sh[t - start] = p;
            part += p;
        }
        l = l * corr + WSUM(part);
        for (uint j = 0; j < MAX_HEAD / 32u; j++) acc[j] *= corr;
        BARRIER();
        for (uint t = start; t < end; t++) {
            float p = sh[t - start];
            GLOBAL const float* vt = vc + (ulong)SLOT(t) * kvd + kvh * head_dim;
            for (uint j = 0; j < MAX_HEAD / 32u; j++) {
                uint d = lane + 32u * j;
                if (d < head_dim) acc[j] += p * vt[d];
            }
        }
        m = mn;
        BARRIER();
    }
    GLOBAL float* o = out + ((ulong)r * n_head + h) * head_dim;
    for (uint j = 0; j < MAX_HEAD / 32u; j++) {
        uint d = lane + 32u * j;
        if (d < head_dim) o[d] = acc[j] / l;
    }
}

#undef SLOT

// Elementwise kernels: one thread per value, i < n.
DEVFN void bias_add(GLOBAL float* x, GLOBAL const float* b, uint n, uint dim, uint i) { if (i < n) x[i] += b[i % dim]; }
DEVFN void silu_mul(GLOBAL float* g, GLOBAL const float* u, uint n, uint i) { if (i < n) { float v = g[i]; g[i] = v / (1.0f + EXP(-v)) * u[i]; } }
DEVFN void add(GLOBAL float* x, GLOBAL const float* y, uint n, uint i) { if (i < n) x[i] += y[i]; }
DEVFN void copy(GLOBAL const float* src, GLOBAL float* dst, uint offset, uint n, uint i) { if (i < n) dst[offset + i] = src[i]; }
// Within one buffer, `to` never after `from`: keeps an accepted tree row in the cache.
DEVFN void move(GLOBAL float* b, uint from, uint to, uint n, uint i) { if (i < n) b[to + i] = b[from + i]; }

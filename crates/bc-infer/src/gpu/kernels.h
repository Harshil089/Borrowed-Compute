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
#define T_Q3K 11u
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

// Q3_K's 6-bit scale i of 16 (quant.rs's q3_scale), and weight t of group sub (0..8) of a
// block: two low bits from qs, the third from hmask (set means no -4).
DEVFN int q3_scale(GLOBAL const uchar* sc, uint i) {
    uint lo, hi;
    if (i < 4u) { lo = sc[i] & 15u; hi = sc[8u + i] & 3u; }
    else if (i < 8u) { lo = sc[i] & 15u; hi = (sc[i + 4u] >> 2) & 3u; }
    else if (i < 12u) { lo = sc[i - 8u] >> 4; hi = (sc[i] >> 4) & 3u; }
    else { lo = sc[i - 8u] >> 4; hi = (sc[i - 4u] >> 6) & 3u; }
    return (int)(lo | (hi << 4)) - 32;
}
DEVFN float q3_value(GLOBAL const uchar* b, uint sub, uint t) {
    // Not `half`: that names a type in Metal.
    uint hb = sub / 4u, j = sub % 4u;
    int q = (int)((b[32u + 32u * hb + t] >> (2u * j)) & 3u) - (((b[t] >> sub) & 1u) != 0u ? 0 : 4);
    return (float)q;
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
    if (ty == T_Q3K) {
        GLOBAL const uchar* b = row + (g / 8u) * 110u;
        uint sub = g % 8u;
        return f16at(b, 108) * (float)q3_scale(b + 96, 2u * sub + (lane < 16u ? 0u : 1u)) * q3_value(b, sub, lane);
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
    } else if (ty == T_Q3K) {
        GLOBAL const uchar* b = row + (g / 8u) * 110u;
        uint sub = g % 8u;
        float d = f16at(b, 108);
        float d0 = d * (float)q3_scale(b + 96, 2u * sub), d1 = d * (float)q3_scale(b + 96, 2u * sub + 1u);
        for (uint k = 0; k < 8u; k++) {
            uint t = 4u * k;
            w[k] = scale4(F4(q3_value(b, sub, t), q3_value(b, sub, t + 1u), q3_value(b, sub, t + 2u), q3_value(b, sub, t + 3u)), k < 4u ? d0 : d1);
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

// Qwen3.5's gated DeltaNet and gated attention.

// Causal convolution of one channel over a batch of rows, then SiLU: tap j < taps - 1
// takes the input j + 1 rows back (from `win`, the last taps - 1 inputs before this batch,
// oldest first, `cd` apart), the last tap the row itself. `win` then holds the batch's
// last inputs. taps <= 9.
DEVFN void conv_silu(GLOBAL const float* in, uint iw, GLOBAL const float* w, uint taps, GLOBAL float* win, GLOBAL float* out, uint cd, uint rows, uint ch) {
    if (ch >= cd) return;
    float h[8];
    for (uint j = 0; j + 1u < taps; j++) h[j] = win[j * cd + ch];
    for (uint t = 0; t < rows; t++) {
        float cur = in[(ulong)t * iw + ch];
        float s = 0.0f;
        for (uint j = 0; j + 1u < taps; j++) s += h[j] * w[ch * taps + j];
        s += cur * w[ch * taps + taps - 1u];
        out[(ulong)t * cd + ch] = s / (1.0f + EXP(-s));
        for (uint j = 0; j + 2u < taps; j++) h[j] = h[j + 1u];
        h[taps - 2u] = cur;
    }
    for (uint j = 0; j + 1u < taps; j++) win[j * cd + ch] = h[j];
}

// Scales head r % heads of row r / heads (rows `stride` apart, heads `hd` long) to unit
// length: x / max(|x|, eps).
DEVFN void l2_heads(GLOBAL float* x, uint stride, uint heads, uint hd, float eps, uint r, uint lane) {
    GLOBAL float* p = x + (ulong)(r / heads) * stride + (r % heads) * hd;
    float s = 0.0f;
    for (uint c = lane; c < hd; c += 32u) s += p[c] * p[c];
    s = WSUM(s);
    float n = SQRT(s);
    float inv = 1.0f / (n > eps ? n : eps);
    for (uint c = lane; c < hd; c += 32u) p[c] *= inv;
}

// The delta rule for value head h, a token at a time, thread j taking column j of the
// head's state S (S[i][j] at i·hd + j): S ← e^g S + β k (v − e^g Sᵀk)ᵀ, out = Sᵀq/√hd.
// Value heads take key heads in turn (h mod k_heads), as the file stores them.
DEVFN void delta_rule(GLOBAL const float* x, GLOBAL const float* ba, uint ba_w, GLOBAL const float* al, uint a_w, uint a_off,
                      GLOBAL const float* dt, GLOBAL const float* a, GLOBAL float* state, uint soff, GLOBAL float* out,
                      uint rows, uint cd, uint key_dim, uint k_heads, uint hd, uint vd, float scale, uint h, uint j) {
    if (j >= hd) return;
    uint kh = h % k_heads;
    GLOBAL float* s = state + soff + (ulong)h * hd * hd;
    for (uint t = 0; t < rows; t++) {
        GLOBAL const float* xt = x + (ulong)t * cd;
        GLOBAL const float* q = xt + kh * hd;
        GLOBAL const float* k = xt + key_dim + kh * hd;
        float v = xt[2u * key_dim + h * hd + j];
        float beta = 1.0f / (1.0f + EXP(-ba[(ulong)t * ba_w + h]));
        float pre = al[(ulong)t * a_w + a_off + h] + dt[h];
        float sp = pre > 20.0f ? pre : LOG(1.0f + EXP(pre));
        float decay = EXP(sp * a[h]);
        float sk = 0.0f;
        for (uint i = 0; i < hd; i++) sk += s[i * hd + j] * k[i];
        float d = (v - sk * decay) * beta;
        float o = 0.0f;
        for (uint i = 0; i < hd; i++) {
            float e = s[i * hd + j] * decay + k[i] * d;
            s[i * hd + j] = e;
            o += e * q[i];
        }
        out[(ulong)t * vd + h * hd + j] = o * scale;
    }
}

// x[i] *= SiLU(z) for the gate of element i (rows `zw` apart, starting at `zoff`).
DEVFN void silu_gate(GLOBAL float* x, GLOBAL const float* z, uint zw, uint zoff, uint vd, uint n, uint i) {
    if (i >= n) return;
    float g = z[(ulong)(i / vd) * zw + zoff + i % vd];
    x[i] *= g / (1.0f + EXP(-g));
}

// Each head of W_q's output holds its query, then its output gate: split them.
DEVFN void split_qgate(GLOBAL const float* full, GLOBAL float* q, GLOBAL float* g, uint hd, uint n, uint i) {
    if (i >= n) return;
    ulong r = i / hd, e = i % hd;
    q[i] = full[r * 2u * hd + e];
    g[i] = full[r * 2u * hd + hd + e];
}

DEVFN void sigmoid_mul(GLOBAL float* x, GLOBAL const float* g, uint n, uint i) { if (i < n) x[i] /= 1.0f + EXP(-g[i]); }

// Mixture of experts.

// Row r's experts: softmax over the E router scores, the k likeliest (the lower index on a
// tie), weights divided by their sum (at least 2^-14) and times `scale` unless 0 or 1.
DEVFN void route(GLOBAL const float* logits, GLOBAL uint* ids, GLOBAL float* wts, uint E, uint k, float scale, uint rows, uint r) {
    if (r >= rows) return;
    GLOBAL const float* l = logits + (ulong)r * E;
    float mx = NEG_INF;
    for (uint e = 0; e < E; e++) mx = l[e] > mx ? l[e] : mx;
    float total = 0.0f;
    for (uint e = 0; e < E; e++) total += EXP(l[e] - mx);
    float picked = 0.0f;
    for (uint s = 0; s < k; s++) {
        uint best = 0;
        float bv = NEG_INF;
        for (uint e = 0; e < E; e++) {
            bool taken = false;
            for (uint t = 0; t < s; t++) taken = taken || ids[r * k + t] == e;
            float p = EXP(l[e] - mx);
            if (!taken && p > bv) { bv = p; best = e; }
        }
        ids[r * k + s] = best;
        wts[r * k + s] = bv / total;
        picked += bv / total;
    }
    picked = picked > 6.103515625e-5f ? picked : 6.103515625e-5f;
    float f = (scale != 0.0f && scale != 1.0f) ? scale : 1.0f;
    for (uint s = 0; s < k; s++) wts[r * k + s] = wts[r * k + s] / picked * f;
}

// One output row of a product with the expert chosen for slot y (ids[y]): weight rows
// ids[y]·estride + roff .. + rows, input row y / xdiv, output row y.
DEVFN void matmul_id(GLOBAL const uchar* w, GLOBAL const float* x, GLOBAL float* out, GLOBAL const uint* ids, uint ty, uint rows, uint cols, uint rb,
                     uint xdiv, uint estride, uint roff, uint r, uint y, uint lane) {
    GLOBAL const uchar* we = w + (ulong)(ids[y] * estride + roff) * rb;
    matmul(we, x + (ulong)(y / xdiv) * cols, out + (ulong)y * rows, ty, rows, cols, 1u, rb, r, 0u, lane);
}

// x[t][c] += Σ_s wts[t·k + s] · d[t·k + s][c], for n = rows · e values.
DEVFN void moe_sum(GLOBAL float* x, GLOBAL const float* d, GLOBAL const float* wts, uint k, uint e, uint n, uint i) {
    if (i >= n) return;
    uint t = i / e, c = i % e;
    float s = 0.0f;
    for (uint j = 0; j < k; j++) s += wts[t * k + j] * d[(ulong)(t * k + j) * e + c];
    x[i] += s;
}

// x[t][c] += σ(g[t]) · y[t][c]: the shared expert, scaled by its gate.
DEVFN void gated_add(GLOBAL float* x, GLOBAL const float* y, GLOBAL const float* g, uint e, uint n, uint i) {
    if (i >= n) return;
    x[i] += y[i] / (1.0f + EXP(-g[i / e]));
}

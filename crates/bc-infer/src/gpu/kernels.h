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

// Weight g*32 + lane of a stored row: the 32 lanes of a SIMD group read one group of 32
// weights together, so their loads of the weights and of the input are side by side in
// memory. Each format follows quant.rs's dequant_block.
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

// Σ w·x over one group of 32 weights (g) against its 32 inputs, for the formats real
// models use most, with wide loads: four inputs at a time, two weight bytes at a time.
// Other formats go through weight() below (has_fast says which).
DEVFN bool has_fast(uint ty) { return ty == T_Q4_0 || ty == T_Q8_0 || ty == T_Q5_0 || ty == T_Q4K || ty == T_Q5K || ty == T_Q6K; }
// Σ of four weights (bytes of v, each masked to its value) against four inputs.
DEVFN float dot4(float a, float b, float c, float d, float4 x) { return a * x.x + b * x.y + c * x.z + d * x.w; }
DEVFN float4 x4(GLOBAL const float* xg, uint k) { return ((GLOBAL const float4*)xg)[k]; }
DEVFN float fast32(GLOBAL const uchar* row, uint ty, uint g, GLOBAL const float* xg) {
    if (ty == T_Q4_0) {
        GLOBAL const uchar* b = row + g * 18u;
        GLOBAL const ushort* q = (GLOBAL const ushort*)(b + 2);
        float lo = 0.0f, hi = 0.0f;
        for (uint k = 0; k < 4u; k++) {
            uint a = q[2 * k], c = q[2 * k + 1];
            float4 xl = x4(xg, k), xh = x4(xg, k + 4);
            lo += ((float)(a & 15u) - 8.0f) * xl.x + ((float)((a >> 8) & 15u) - 8.0f) * xl.y + ((float)(c & 15u) - 8.0f) * xl.z + ((float)((c >> 8) & 15u) - 8.0f) * xl.w;
            hi += ((float)((a >> 4) & 15u) - 8.0f) * xh.x + ((float)((a >> 12) & 15u) - 8.0f) * xh.y + ((float)((c >> 4) & 15u) - 8.0f) * xh.z + ((float)((c >> 12) & 15u) - 8.0f) * xh.w;
        }
        return f16at(b, 0) * (lo + hi);
    }
    if (ty == T_Q8_0) {
        GLOBAL const uchar* b = row + g * 34u;
        GLOBAL const ushort* q = (GLOBAL const ushort*)(b + 2);
        float s = 0.0f;
        for (uint k = 0; k < 8u; k++) {
            uint a = q[2 * k], c = q[2 * k + 1];
            s += dot4((float)i8of(a & 255u), (float)i8of(a >> 8), (float)i8of(c & 255u), (float)i8of(c >> 8), x4(xg, k));
        }
        return f16at(b, 0) * s;
    }
    if (ty == T_Q5_0) {
        GLOBAL const uchar* b = row + g * 22u;
        GLOBAL const ushort* q = (GLOBAL const ushort*)(b + 6);
        uint qh = u32at(b, 2);
        float lo = 0.0f, hi = 0.0f;
        for (uint k = 0; k < 4u; k++) {
            uint a = q[2 * k], c = q[2 * k + 1], j = 4u * k;
            uint h = qh >> j, H = qh >> (j + 16u);
            lo += dot4((float)((a & 15u) | ((h & 1u) << 4)) - 16.0f, (float)(((a >> 8) & 15u) | ((h & 2u) << 3)) - 16.0f,
                       (float)((c & 15u) | ((h & 4u) << 2)) - 16.0f, (float)(((c >> 8) & 15u) | ((h & 8u) << 1)) - 16.0f, x4(xg, k));
            hi += dot4((float)(((a >> 4) & 15u) | ((H & 1u) << 4)) - 16.0f, (float)(((a >> 12) & 15u) | ((H & 2u) << 3)) - 16.0f,
                       (float)(((c >> 4) & 15u) | ((H & 4u) << 2)) - 16.0f, (float)(((c >> 12) & 15u) | ((H & 8u) << 1)) - 16.0f, x4(xg, k + 4));
        }
        return f16at(b, 0) * (lo + hi);
    }
    if (ty == T_Q4K || ty == T_Q5K) {
        GLOBAL const uchar* b = row + (g / 8u) * (ty == T_Q4K ? 144u : 176u);
        uint sub = g % 8u, c = sub / 2u, hi = sub % 2u;
        float sc, mn;
        scale_min_k4(sub, b + 4, &sc, &mn);
        GLOBAL const uint* q = (GLOBAL const uint*)(b + (ty == T_Q4K ? 16u : 48u) + 32u * c);
        GLOBAL const uint* qh = (GLOBAL const uint*)(b + 16);
        uint bit = hi + 2u * c;
        float s = 0.0f, xs = 0.0f;
        for (uint k = 0; k < 8u; k++) {
            uint v = (q[k] >> (4u * hi)) & 0x0f0f0f0fu;
            if (ty == T_Q5K) v |= ((qh[k] >> bit) & 0x01010101u) << 4;
            float4 x = x4(xg, k);
            s += dot4((float)(v & 255u), (float)((v >> 8) & 255u), (float)((v >> 16) & 255u), (float)(v >> 24), x);
            xs += x.x + x.y + x.z + x.w;
        }
        return f16at(b, 0) * sc * s - f16at(b, 2) * mn * xs;
    }
    if (ty == T_Q6K) {
        GLOBAL const uchar* b = row + (g / 8u) * 210u;
        uint sub = g % 8u, hh = sub / 4u, qr = sub % 4u;
        GLOBAL const ushort* ql = (GLOBAL const ushort*)(b + 64u * hh + ((qr & 1u) != 0u ? 32u : 0u));
        GLOBAL const ushort* qh = (GLOBAL const ushort*)(b + 128u + 32u * hh);
        uint shift = qr < 2u ? 0u : 4u;
        float s0 = 0.0f, s1 = 0.0f;
        for (uint k = 0; k < 8u; k++) {
            uint lo = ql[2 * k] | (ql[2 * k + 1] << 16), h = qh[2 * k] | (qh[2 * k + 1] << 16);
            uint v = ((lo >> shift) & 0x0f0f0f0fu) | (((h >> (2u * qr)) & 0x03030303u) << 4);
            float d = dot4((float)(v & 255u) - 32.0f, (float)((v >> 8) & 255u) - 32.0f, (float)((v >> 16) & 255u) - 32.0f, (float)(v >> 24) - 32.0f, x4(xg, k));
            if (k < 4u) s0 += d; else s1 += d;
        }
        GLOBAL const uchar* sc = b + 192u + 8u * hh + 2u * qr;
        return f16at(b, 208) * ((float)i8of(sc[0]) * s0 + (float)i8of(sc[1]) * s1);
    }
    return 0.0f;
}

// Weight rows per matrix-product thread group: one per SIMD group of 32.
#define ROWS_PER_GROUP 4

// out[i][r] = Σ_c W[r][c] · x[i][c], for input rows i0 .. i0 + NB of n.
DEVFN void matmul(GLOBAL const uchar* wt, GLOBAL const float* x, GLOBAL float* out, uint ty, uint rows, uint cols, uint n, uint row_bytes, uint r, uint i0, uint lane) {
    if (r >= rows) return;
    GLOBAL const uchar* row = wt + (ulong)r * row_bytes;
    uint groups = cols / 32u;
    uint end = i0 + NB < n ? i0 + NB : n;
    for (uint i = i0; i < end; i++) {
        GLOBAL const float* xi = x + (ulong)i * cols;
        float acc = 0.0f;
        if (has_fast(ty)) {
            for (uint g = lane; g < groups; g += 32u) acc += fast32(row, ty, g, xi + g * 32u);
        } else {
            for (uint g = 0; g < groups; g++) acc += weight(row, ty, g, lane) * xi[g * 32u + lane];
        }
        acc = WSUM(acc);
        if (lane == 0u) out[(ulong)i * rows + r] = acc;
    }
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
DEVFN void rope(GLOBAL float* x, GLOBAL const float* inv_freq, uint heads, uint head_dim, uint rope_dim, uint pos, uint neox, uint idx) {
    uint hh = rope_dim / 2u;
    uint i = idx % hh, h = (idx / hh) % heads, r = idx / (hh * heads);
    GLOBAL float* hp = x + ((ulong)r * heads + h) * head_dim;
    float theta = (float)(pos + r) * inv_freq[i];
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
DEVFN void attention(GLOBAL const float* q, GLOBAL const float* kc, GLOBAL const float* vc, GLOBAL float* out, LOCAL float* sh,
                     uint n_head, uint n_head_kv, uint head_dim, uint pos, float scale, uint h, uint r, uint lane) {
    uint kvd = n_head_kv * head_dim;
    uint kvh = h / (n_head / n_head_kv);
    uint upto = pos + r + 1u;
    GLOBAL const float* qh = q + ((ulong)r * n_head + h) * head_dim;
    float acc[MAX_HEAD / 32];
    for (uint j = 0; j < MAX_HEAD / 32u; j++) acc[j] = 0.0f;
    float m = NEG_INF, l = 0.0f;
    for (uint start = 0; start < upto; start += TILE) {
        uint end = start + TILE < upto ? start + TILE : upto;
        float tmax = NEG_INF;
        for (uint t = start + lane; t < end; t += 32u) {
            GLOBAL const float* kt = kc + (ulong)t * kvd + kvh * head_dim;
            float s = 0.0f;
            for (uint d = 0; d < head_dim; d++) s += qh[d] * kt[d];
            s *= scale;
            sh[t - start] = s;
            tmax = s > tmax ? s : tmax;
        }
        tmax = WMAX(tmax);
        float mn = m > tmax ? m : tmax;
        float corr = EXP(m - mn);
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
            GLOBAL const float* vt = vc + (ulong)t * kvd + kvh * head_dim;
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

// Elementwise kernels: one thread per value, i < n.
DEVFN void bias_add(GLOBAL float* x, GLOBAL const float* b, uint n, uint dim, uint i) { if (i < n) x[i] += b[i % dim]; }
DEVFN void silu_mul(GLOBAL float* g, GLOBAL const float* u, uint n, uint i) { if (i < n) { float v = g[i]; g[i] = v / (1.0f + EXP(-v)) * u[i]; } }
DEVFN void add(GLOBAL float* x, GLOBAL const float* y, uint n, uint i) { if (i < n) x[i] += y[i]; }
DEVFN void copy(GLOBAL const float* src, GLOBAL float* dst, uint offset, uint n, uint i) { if (i < n) dst[offset + i] = src[i]; }

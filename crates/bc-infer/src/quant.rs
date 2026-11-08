//! Weight formats as GGUF stores them, and the matrix products the model needs.
//!
//! Block formats follow ggml exactly (llama.cpp's `ggml-quants.c`): a block of 32 or 256
//! weights shares one or two f16 scales, and each weight is a small integer.
//! - `Q4_0`: x = d·(q − 8), q in 0..16. `Q4_1`: x = d·q + m.
//! - `Q5_0`: x = d·(q − 16), q in 0..32 (fifth bit packed in `qh`). `Q5_1`: x = d·q + m.
//! - `Q8_0`: x = d·q, q an i8.
//! - `Q4_K`/`Q5_K`: 256 weights in 8 groups of 32; each group has a 6-bit scale and min,
//!   themselves scaled by the block's `d` and `dmin`: x = d·sc·q − dmin·m.
//! - `Q6_K`: 256 weights in 16 groups of 16 with i8 scales: x = d·sc·(q − 32).
//!
//! Decoding a token reads every weight once, so the product works a row at a time:
//! dequantize one row into a small buffer, take its dot product with every input row, move
//! on. The weights stay compressed in memory; only one row per thread is ever expanded.

use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlType {
    /// ggml's number for the type, as GGUF stores it (and the GPU kernels switch on).
    pub fn id(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q4K => 12,
            GgmlType::Q5K => 13,
            GgmlType::Q6K => 14,
            GgmlType::BF16 => 30,
        }
    }

    pub fn from_u32(t: u32) -> Result<Self> {
        Ok(match t {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            30 => GgmlType::BF16,
            other => bail!("weights in ggml type {other} are not supported yet"),
        })
    }

    /// Weights per block and bytes per block.
    pub fn block(self) -> (usize, usize) {
        match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 | GgmlType::BF16 => (1, 2),
            GgmlType::Q4_0 => (32, 18),
            GgmlType::Q4_1 => (32, 20),
            GgmlType::Q5_0 => (32, 22),
            GgmlType::Q5_1 => (32, 24),
            GgmlType::Q8_0 => (32, 34),
            GgmlType::Q4K => (256, 144),
            GgmlType::Q5K => (256, 176),
            GgmlType::Q6K => (256, 210),
        }
    }

    pub fn bytes_for(self, elements: usize) -> Result<usize> {
        let (n, b) = self.block();
        if !elements.is_multiple_of(n) {
            bail!("{elements} weights do not fill whole {self:?} blocks");
        }
        Ok(elements / n * b)
    }
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        // Subnormal: normalize the mantissa.
        (0, mut m) => {
            let mut e = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (31, m) => sign | (0xff << 23) | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        return sign; // too small for this crate's uses (test fixtures); flush to zero
    }
    // Round to nearest.
    let mut h = sign as u32 | ((exp as u32) << 10) | (man >> 13);
    if man & 0x1000 != 0 {
        h += 1;
    }
    h as u16
}

fn f16_at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// The 6-bit scale and min of group `j` in a K-quant block (`get_scale_min_k4`).
fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((q[j] & 63) as f32, (q[j + 4] & 63) as f32)
    } else {
        (
            ((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4)) as f32,
            ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f32,
        )
    }
}

/// Expands one block of `ty` from `src` into `dst` (block-sized).
pub fn dequant_block(ty: GgmlType, src: &[u8], dst: &mut [f32]) {
    match ty {
        GgmlType::F32 => dst[0] = f32::from_le_bytes(src[..4].try_into().unwrap()),
        GgmlType::F16 => dst[0] = f16_at(src, 0),
        GgmlType::BF16 => dst[0] = f32::from_bits((u16::from_le_bytes([src[0], src[1]]) as u32) << 16),
        GgmlType::Q4_0 => {
            let d = f16_at(src, 0);
            for j in 0..16 {
                dst[j] = ((src[2 + j] & 0xf) as f32 - 8.0) * d;
                dst[j + 16] = ((src[2 + j] >> 4) as f32 - 8.0) * d;
            }
        }
        GgmlType::Q4_1 => {
            let (d, m) = (f16_at(src, 0), f16_at(src, 2));
            for j in 0..16 {
                dst[j] = (src[4 + j] & 0xf) as f32 * d + m;
                dst[j + 16] = (src[4 + j] >> 4) as f32 * d + m;
            }
        }
        GgmlType::Q5_0 | GgmlType::Q5_1 => {
            let (d, m, at) = if ty == GgmlType::Q5_0 { (f16_at(src, 0), 0.0, 2) } else { (f16_at(src, 0), f16_at(src, 2), 4) };
            let qh = u32::from_le_bytes(src[at..at + 4].try_into().unwrap());
            let qs = &src[at + 4..at + 20];
            let offset = if ty == GgmlType::Q5_0 { 16.0 } else { 0.0 };
            for j in 0..16 {
                let h0 = ((qh >> j) << 4) & 0x10;
                let h1 = (qh >> (j + 12)) & 0x10;
                dst[j] = (((qs[j] & 0xf) as u32 | h0) as f32 - offset) * d + m;
                dst[j + 16] = (((qs[j] >> 4) as u32 | h1) as f32 - offset) * d + m;
            }
        }
        GgmlType::Q8_0 => {
            let d = f16_at(src, 0);
            for j in 0..32 {
                dst[j] = src[2 + j] as i8 as f32 * d;
            }
        }
        GgmlType::Q4K => {
            let (d, dmin) = (f16_at(src, 0), f16_at(src, 2));
            let scales = &src[4..16];
            let qs = &src[16..144];
            for g in 0..4 {
                let (sc1, m1) = scale_min_k4(2 * g, scales);
                let (sc2, m2) = scale_min_k4(2 * g + 1, scales);
                let q = &qs[32 * g..32 * g + 32];
                let out = &mut dst[64 * g..64 * g + 64];
                for l in 0..32 {
                    out[l] = d * sc1 * (q[l] & 0xf) as f32 - dmin * m1;
                    out[l + 32] = d * sc2 * (q[l] >> 4) as f32 - dmin * m2;
                }
            }
        }
        GgmlType::Q5K => {
            let (d, dmin) = (f16_at(src, 0), f16_at(src, 2));
            let scales = &src[4..16];
            let qh = &src[16..48];
            let qs = &src[48..176];
            for g in 0..4 {
                let (sc1, m1) = scale_min_k4(2 * g, scales);
                let (sc2, m2) = scale_min_k4(2 * g + 1, scales);
                let (u1, u2) = (1u8 << (2 * g), 2u8 << (2 * g));
                let q = &qs[32 * g..32 * g + 32];
                let out = &mut dst[64 * g..64 * g + 64];
                for l in 0..32 {
                    let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                    let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                    out[l] = d * sc1 * ((q[l] & 0xf) + hi1) as f32 - dmin * m1;
                    out[l + 32] = d * sc2 * ((q[l] >> 4) + hi2) as f32 - dmin * m2;
                }
            }
        }
        GgmlType::Q6K => {
            let ql = &src[0..128];
            let qh = &src[128..192];
            let sc = &src[192..208];
            let d = f16_at(src, 208);
            for half in 0..2 {
                let (ql, qh, sc) = (&ql[64 * half..], &qh[32 * half..], &sc[8 * half..]);
                let out = &mut dst[128 * half..128 * half + 128];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    out[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                    out[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                    out[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                    out[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                }
            }
        }
    }
}

/// A weight matrix as stored: `rows` rows of `cols` weights, each row whole blocks.
#[derive(Debug, Clone)]
pub struct QTensor {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub data: Bytes,
}

/// Page alignment of weight storage: Apple's GPU can then use the very same memory
/// (`newBufferWithBytesNoCopy` wants whole 16 KB pages), so weights on a Mac are never
/// held twice.
const PAGE: usize = 16384;

/// Bytes in a page-aligned allocation, padded to whole pages.
pub struct Bytes {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
    /// Kept in RAM (`pin`), never swapped to disk.
    locked: bool,
}

// SAFETY: Bytes owns its allocation and only hands out shared or exclusive borrows of it.
unsafe impl Send for Bytes {}
unsafe impl Sync for Bytes {}

impl Bytes {
    fn layout(len: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(len.div_ceil(PAGE).max(1) * PAGE, PAGE).expect("weight size")
    }

    pub fn from_slice(b: &[u8]) -> Self {
        let layout = Self::layout(b.len());
        // SAFETY: the layout has a non-zero size; the allocation is zeroed, so the padding
        // past `len` is initialized too.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = std::ptr::NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        // SAFETY: both regions are valid for b.len() bytes and do not overlap.
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), ptr.as_ptr(), b.len()) };
        let locked = crate::pin::lock(ptr.as_ptr(), Self::layout(b.len()).size());
        Bytes { ptr, len: b.len(), locked }
    }

    /// Whether these bytes are locked in RAM.
    pub fn locked(&self) -> bool {
        self.locked
    }

    /// The allocation's length, whole pages.
    pub fn padded_len(&self) -> usize {
        Self::layout(self.len).size()
    }
}

impl std::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: ptr is valid for len initialized bytes for as long as self lives.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Bytes {
    fn drop(&mut self) {
        if self.locked {
            crate::pin::unlock(self.ptr.as_ptr(), Self::layout(self.len).size());
        }
        // SAFETY: allocated in from_slice with this same layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), Self::layout(self.len)) }
    }
}

impl Clone for Bytes {
    fn clone(&self) -> Self {
        Bytes::from_slice(self)
    }
}

impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} bytes", self.len)
    }
}

/// Weight-uses per piece of a parallel product: smaller pieces cost more to hand out
/// than to compute.
const PARALLEL_MIN: usize = 1 << 18;

impl QTensor {
    pub fn new(ty: GgmlType, rows: usize, cols: usize, data: Vec<u8>) -> Result<Self> {
        let want = ty.bytes_for(rows * cols)?;
        if data.len() != want {
            bail!("{rows}x{cols} {ty:?} needs {want} bytes, got {}", data.len());
        }
        if !cols.is_multiple_of(ty.block().0) {
            bail!("rows of {cols} do not fill whole {ty:?} blocks");
        }
        Ok(QTensor { ty, rows, cols, data: Bytes::from_slice(&data) })
    }

    pub fn row_bytes(&self) -> usize {
        self.cols / self.ty.block().0 * self.ty.block().1
    }

    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        let (n, b) = self.ty.block();
        let row = &self.data[r * self.row_bytes()..(r + 1) * self.row_bytes()];
        for (blk, dst) in row.chunks_exact(b).zip(out[..self.cols].chunks_exact_mut(n)) {
            dequant_block(self.ty, blk, dst);
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.rows * self.cols];
        for (r, chunk) in out.chunks_exact_mut(self.cols).enumerate() {
            self.dequant_row(r, chunk);
        }
        out
    }

    /// `out[i][r] = Σ_c W[r][c] · x[i][c]` for `n` input rows: `x` is n×cols, `out` n×rows.
    ///
    /// Inputs are quantized to int8 per block of 32 first (scale = max|x|/127), so each
    /// weight block meets its input block as integers: Σ wᵢxᵢ ≈ d_w·d_x·Σ qᵢpᵢ, with the
    /// sum in i32. That is llama.cpp's method; it turns the inner loop into integer
    /// multiply-adds the CPU does 16 at a time.
    pub fn matmul(&self, x: &[f32], n: usize) -> Vec<f32> {
        self.matmul_rows(x, n, self.rows)
    }

    /// [`QTensor::matmul`] against the first `upto` weight rows only: `out` is n×upto.
    pub fn matmul_rows(&self, x: &[f32], n: usize, upto: usize) -> Vec<f32> {
        let total = upto.min(self.rows);
        debug_assert_eq!(x.len(), n * self.cols);
        let acts: Vec<Q8Row> = (0..n).map(|i| Q8Row::new(&x[i * self.cols..(i + 1) * self.cols])).collect();
        // Pieces of at least PARALLEL_MIN weight-uses, several per thread for big products,
        // so a paused thread delays only a small piece and small products stay on one.
        let work = total * self.cols * n.max(1);
        let chunks = (work / PARALLEL_MIN).clamp(1, crate::pool::threads() * 4).min(total.div_ceil(16).max(1));
        let per = total.div_ceil(chunks);
        let slots: Vec<std::sync::Mutex<Vec<f32>>> = (0..chunks).map(|_| Default::default()).collect();
        let integer = unpackable(self.ty);
        crate::pool::run(chunks, &|t| {
            let rows = (t * per).min(total)..((t + 1) * per).min(total);
            let mut part = vec![0.0f32; n * rows.len()];
            let mut buf = Vec::new();
            let (bn, bb) = self.ty.block();
            let mut blk_q = Unpacked::new();
            let mut sums = vec![0f32; n];
            for (k, r) in rows.clone().enumerate() {
                let row = &self.data[r * self.row_bytes()..(r + 1) * self.row_bytes()];
                if integer {
                    // Each block is unpacked once and used for every input row while it
                    // is still in cache: a prompt or a speculative check costs little
                    // more than one token.
                    sums.fill(0.0);
                    for (j, blk) in row.chunks_exact(bb).enumerate() {
                        blk_q.unpack_block(self.ty, blk);
                        for (sum, a) in sums.iter_mut().zip(&acts) {
                            *sum += blk_q.dot(a, j * bn, bn);
                        }
                    }
                    for (i, sum) in sums.iter().enumerate() {
                        part[i * rows.len() + k] = *sum;
                    }
                } else {
                    for i in 0..n {
                        part[i * rows.len() + k] = dot_dequant(self.ty, row, &x[i * self.cols..(i + 1) * self.cols], &mut buf);
                    }
                }
            }
            *slots[t].lock().unwrap() = part;
        });
        let mut out = vec![0.0f32; n * total];
        for (t, slot) in slots.into_iter().enumerate() {
            let part = slot.into_inner().unwrap();
            let start = (t * per).min(total);
            let len = part.len() / n.max(1);
            for i in 0..n {
                out[i * total + start..i * total + start + len].copy_from_slice(&part[i * len..(i + 1) * len]);
            }
        }
        out
    }
}

/// One input row quantized to int8 in blocks of 32, with each half-block's sum.
pub struct Q8Row {
    qs: Vec<i8>,
    d: Vec<f32>,
    sums16: Vec<i32>,
    sums32: Vec<i32>,
}

impl Q8Row {
    pub fn new(x: &[f32]) -> Self {
        let blocks = x.len().div_ceil(32);
        let mut qs = Vec::with_capacity(x.len());
        let mut d = Vec::with_capacity(blocks);
        for b in x.chunks(32) {
            let max = b.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = max / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            qs.extend(b.iter().map(|v| (v * inv).round() as i8));
            d.push(scale);
        }
        let sums16: Vec<i32> = qs.chunks(16).map(|c| c.iter().map(|&q| q as i32).sum()).collect();
        let sums32 = sums16.chunks(2).map(|c| c.iter().sum()).collect();
        Q8Row { qs, d, sums16, sums32 }
    }
}

/// A weight block in one shape for every integer format: groups of small integers, each
/// group with a scale and an offset, w = s·q + o. Groups are 32 long, matching the input
/// blocks, except Q6_K's 16.
struct Unpacked {
    q: [i8; 256],
    s: [f32; 16],
    o: [f32; 16],
    group: usize,
}

fn unpackable(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K)
}

impl Unpacked {
    fn new() -> Self {
        Unpacked { q: [0; 256], s: [0.0; 16], o: [0.0; 16], group: 32 }
    }

    /// Unpacks one block into the start of the buffers.
    fn unpack_block(&mut self, ty: GgmlType, blk: &[u8]) {
        let n = ty.block().0;
        self.group = if ty == GgmlType::Q6K { 16 } else { 32 };
        {
            let q = &mut self.q[..n];
            let (s, o) = (&mut self.s[..n / self.group], &mut self.o[..n / self.group]);
            match ty {
                GgmlType::Q8_0 => {
                    for k in 0..32 {
                        q[k] = blk[2 + k] as i8;
                    }
                    s.fill(f16_at(blk, 0));
                    o.fill(0.0);
                }
                GgmlType::Q4_0 | GgmlType::Q4_1 => {
                    let at = if ty == GgmlType::Q4_0 { 2 } else { 4 };
                    let bias = if ty == GgmlType::Q4_0 { 8 } else { 0 };
                    for k in 0..16 {
                        q[k] = (blk[at + k] & 0xf) as i8 - bias;
                        q[k + 16] = (blk[at + k] >> 4) as i8 - bias;
                    }
                    s.fill(f16_at(blk, 0));
                    o.fill(if ty == GgmlType::Q4_1 { f16_at(blk, 2) } else { 0.0 });
                }
                GgmlType::Q5_0 | GgmlType::Q5_1 => {
                    let at = if ty == GgmlType::Q5_0 { 2 } else { 4 };
                    let bias = if ty == GgmlType::Q5_0 { 16 } else { 0 };
                    let qh = u32::from_le_bytes(blk[at..at + 4].try_into().unwrap());
                    for k in 0..16 {
                        let h0 = (((qh >> k) << 4) & 0x10) as u8;
                        let h1 = ((qh >> (k + 12)) & 0x10) as u8;
                        q[k] = ((blk[at + 4 + k] & 0xf) | h0) as i8 - bias;
                        q[k + 16] = ((blk[at + 4 + k] >> 4) | h1) as i8 - bias;
                    }
                    s.fill(f16_at(blk, 0));
                    o.fill(if ty == GgmlType::Q5_1 { f16_at(blk, 2) } else { 0.0 });
                }
                GgmlType::Q4K | GgmlType::Q5K => {
                    let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
                    let scales = &blk[4..16];
                    let (qh, qs) = if ty == GgmlType::Q4K { (&blk[0..0], &blk[16..144]) } else { (&blk[16..48], &blk[48..176]) };
                    for g in 0..4 {
                        let (sc1, m1) = scale_min_k4(2 * g, scales);
                        let (sc2, m2) = scale_min_k4(2 * g + 1, scales);
                        let qq = &qs[32 * g..32 * g + 32];
                        for l in 0..32 {
                            let (mut lo, mut hi) = ((qq[l] & 0xf) as i8, (qq[l] >> 4) as i8);
                            if ty == GgmlType::Q5K {
                                lo += if qh[l] & (1 << (2 * g)) != 0 { 16 } else { 0 };
                                hi += if qh[l] & (2 << (2 * g)) != 0 { 16 } else { 0 };
                            }
                            q[64 * g + l] = lo;
                            q[64 * g + 32 + l] = hi;
                        }
                        s[2 * g] = d * sc1;
                        o[2 * g] = -dmin * m1;
                        s[2 * g + 1] = d * sc2;
                        o[2 * g + 1] = -dmin * m2;
                    }
                }
                GgmlType::Q6K => {
                    let d = f16_at(blk, 208);
                    for half in 0..2 {
                        let (ql, qh, sc) = (&blk[64 * half..], &blk[128 + 32 * half..], &blk[192 + 8 * half..]);
                        let base = 128 * half;
                        for l in 0..32 {
                            q[base + l] = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i8 - 32;
                            q[base + l + 32] = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                            q[base + l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                            q[base + l + 96] = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                        }
                        // Group k of 16 within this half uses scale k (dequant_block's sc[is + 2·quarter]).
                        for k in 0..8 {
                            s[8 * half + k] = d * (sc[k] as i8) as f32;
                            o[8 * half + k] = 0.0;
                        }
                    }
                }
                _ => unreachable!("checked by unpackable"),
            }
        }
    }

    /// Σ w·x over the unpacked block of `len` weights, against the input starting at
    /// weight `at`: Σ_groups d_x·(s·Σqp + o·Σp).
    #[inline]
    fn dot(&self, a: &Q8Row, at: usize, len: usize) -> f32 {
        let mut sum = 0f32;
        let qs = &a.qs[at..at + len];
        if self.group == 32 {
            let (d, sums) = (&a.d[at / 32..(at + len) / 32], &a.sums32[at / 32..(at + len) / 32]);
            for gi in 0..len / 32 {
                let w: &[i8; 32] = self.q[32 * gi..32 * gi + 32].try_into().unwrap();
                let p: &[i8; 32] = qs[32 * gi..32 * gi + 32].try_into().unwrap();
                sum += d[gi] * (self.s[gi] * idot32(w, p) as f32 + self.o[gi] * sums[gi] as f32);
            }
        } else {
            let sums = &a.sums16[at / 16..(at + len) / 16];
            for gi in 0..len / 16 {
                let w: &[i8; 16] = self.q[16 * gi..16 * gi + 16].try_into().unwrap();
                let p: &[i8; 16] = qs[16 * gi..16 * gi + 16].try_into().unwrap();
                sum += a.d[(at / 16 + gi) / 2] * (self.s[gi] * idot16(w, p) as f32 + self.o[gi] * sums[gi] as f32);
            }
        }
        sum
    }
}

/// Integer dot product; the compiler turns it into the CPU's multiply-add instructions.
#[inline(always)]
fn idot32(a: &[i8; 32], b: &[i8; 32]) -> i32 {
    let mut acc = [0i32; 8];
    for k in 0..32 {
        acc[k % 8] += a[k] as i32 * b[k] as i32;
    }
    acc.iter().sum()
}

#[inline(always)]
fn idot16(a: &[i8; 16], b: &[i8; 16]) -> i32 {
    let mut acc = [0i32; 8];
    for k in 0..16 {
        acc[k % 8] += a[k] as i32 * b[k] as i32;
    }
    acc.iter().sum()
}

/// Dot product of a stored row with an f32 input, block by block (F32, F16, BF16).
fn dot_dequant(ty: GgmlType, row: &[u8], x: &[f32], buf: &mut Vec<f32>) -> f32 {
    let (n, b) = ty.block();
    buf.resize(n, 0.0);
    let mut sum = 0f32;
    for (j, blk) in row.chunks_exact(b).enumerate() {
        dequant_block(ty, blk, buf);
        sum += dot(buf, &x[j * n..(j + 1) * n]);
    }
    sum
}

/// Dot product in 8 lanes, which the compiler turns into SIMD.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

/// Quantizes f32 weights to Q8_0, for test fixtures and the draft model's tests.
pub fn quantize_q8_0(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() / 32 * 34);
    for block in values.chunks(32) {
        let max = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = max / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend(block.iter().map(|v| (v * inv).round().clamp(-127.0, 127.0) as i8 as u8));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversions() {
        for (h, f) in [(0x3c00u16, 1.0f32), (0xc000, -2.0), (0x0000, 0.0), (0x7bff, 65504.0), (0x0001, 5.960_464_5e-8), (0x3555, 0.333_251_95)] {
            assert_eq!(f16_to_f32(h), f, "{h:#x}");
        }
        for v in [1.0f32, -0.5, 1234.0, 0.1] {
            assert!((f16_to_f32(f32_to_f16(v)) - v).abs() <= v.abs() * 1e-3);
        }
    }

    fn f16b(v: f32) -> [u8; 2] {
        f32_to_f16(v).to_le_bytes()
    }

    #[test]
    fn q4_0_and_q8_0_blocks() {
        let mut b = f16b(0.5).to_vec();
        b.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
        let mut out = [0f32; 32];
        dequant_block(GgmlType::Q4_0, &b, &mut out);
        assert_eq!(out[0], -4.0); // (0 − 8)·0.5
        assert_eq!(out[15], 3.5); // (15 − 8)·0.5
        assert_eq!(out[16], 3.5); // high nibble of byte 0 is 15
        let q8 = quantize_q8_0(&(0..32).map(|i| i as f32 - 16.0).collect::<Vec<_>>());
        dequant_block(GgmlType::Q8_0, &q8, &mut out);
        assert!((out[0] + 16.0).abs() < 0.1 && (out[31] - 15.0).abs() < 0.1, "{out:?}");
    }

    #[test]
    fn q5_0_uses_the_fifth_bit() {
        let mut b = f16b(1.0).to_vec();
        b.extend(0x0001_0001u32.to_le_bytes()); // bit 0 → element 0, bit 16 → element 16
        b.extend([0u8; 16]);
        let mut out = [0f32; 32];
        dequant_block(GgmlType::Q5_0, &b, &mut out);
        assert_eq!(out[0], 0.0); // 16 − 16
        assert_eq!(out[1], -16.0);
        assert_eq!(out[16], 0.0); // high half: element j + 16 takes bit j + 16
        assert_eq!(out[20], -16.0);
    }

    #[test]
    fn k_quant_scales_unpack() {
        // Groups 0..4 take the low 6 bits; 4..8 combine nibbles with the top bits.
        let mut s = [0u8; 12];
        s[0] = 0b11_000101; // scale 5 for group 0, top bits 3 for group 4's scale
        s[4] = 7; // min 7 for group 0
        s[8] = 0x2a; // group 4: scale low nibble 0xa, min low nibble 2
        assert_eq!(scale_min_k4(0, &s), (5.0, 7.0));
        assert_eq!(scale_min_k4(4, &s), ((0xa | (3 << 4)) as f32, 2.0));
    }

    #[test]
    fn q4k_block_is_d_sc_q_minus_dmin_m() {
        let mut b = Vec::new();
        b.extend(f16b(0.5));
        b.extend(f16b(0.25));
        let mut scales = [0u8; 12];
        scales[0] = 2; // group 0 scale
        scales[4] = 4; // group 0 min
        b.extend(scales);
        b.extend([0x31u8; 128]); // low nibble 1, high nibble 3
        let mut out = [0f32; 256];
        dequant_block(GgmlType::Q4K, &b, &mut out);
        assert_eq!(out[0], 0.5 * 2.0 * 1.0 - 0.25 * 4.0); // group 0
        assert_eq!(out[32], 0.0); // group 1: scale 0, min 0
    }

    #[test]
    fn q6k_block() {
        let mut b = vec![0u8; 210];
        b[0] = 0x05; // ql[0] low = 5
        b[128] = 0x02; // qh[0] bits 0-1 = 2 → q1 = 5 | 32 = 37
        b[192] = 3; // scale 0
        b[208..210].copy_from_slice(&f16b(0.5));
        let mut out = [0f32; 256];
        dequant_block(GgmlType::Q6K, &b, &mut out);
        assert_eq!(out[0], 0.5 * 3.0 * (37.0 - 32.0));
        assert_eq!(out[1], 0.5 * 3.0 * -32.0);
    }

    #[test]
    fn matmul_matches_naive_on_many_threads() {
        let (rows, cols, n) = (300, 256, 3);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919) % 255) as f32 / 255.0 - 0.5).collect();
        let t = QTensor::new(GgmlType::Q8_0, rows, cols, quantize_q8_0(&w)).unwrap();
        let x: Vec<f32> = (0..n * cols).map(|i| ((i * 31) % 17) as f32 / 17.0 - 0.5).collect();
        let got = t.matmul(&x, n);
        let wd = t.to_f32();
        for i in 0..n {
            for r in 0..rows {
                let want: f32 = (0..cols).map(|c| wd[r * cols + c] * x[i * cols + c]).sum();
                // Inputs go through int8: allow 1% of the sum of magnitudes.
                let mag: f32 = (0..cols).map(|c| (wd[r * cols + c] * x[i * cols + c]).abs()).sum();
                assert!((got[i * rows + r] - want).abs() < mag * 0.01, "{i},{r}: {} vs {want}", got[i * rows + r]);
            }
        }
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::*;

    /// Every integer kernel against dequantize-then-dot, on random blocks.
    #[test]
    fn integer_kernels_match_dequantized_products() {
        let mut seed = 12345u64;
        let mut byte = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 24) as u8
        };
        for ty in [GgmlType::Q4_0, GgmlType::Q5_0, GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K] {
            let (n, b) = ty.block();
            let cols = 512;
            let mut row: Vec<u8> = (0..cols / n * b).map(|_| byte()).collect();
            // Keep scales finite and modest: rewrite each block's f16 fields.
            for blk in row.chunks_exact_mut(b) {
                let at = if ty == GgmlType::Q6K { vec![208] } else if matches!(ty, GgmlType::Q4K | GgmlType::Q5K) { vec![0, 2] } else { vec![0] };
                for i in at {
                    blk[i..i + 2].copy_from_slice(&f32_to_f16(0.01 + (byte() as f32) / 2000.0).to_le_bytes());
                }
            }
            let x: Vec<f32> = (0..cols).map(|_| byte() as f32 / 128.0 - 1.0).collect();
            let t = QTensor::new(ty, 1, cols, row).unwrap();
            let w = t.to_f32();
            let want: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
            let mag: f32 = w.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
            let got = t.matmul(&x, 1)[0];
            assert!((got - want).abs() < mag * 0.01, "{ty:?}: {got} vs {want}");
        }
    }
}

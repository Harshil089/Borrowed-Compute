//! Times one matrix-vector product per format, to tune the kernels.
use std::time::Instant;

use bc_infer::quant::{GgmlType, QTensor};

fn main() {
    let (rows, cols) = (4864, 896 * 2);
    for ty in [GgmlType::Q8_0, GgmlType::Q4_0, GgmlType::Q4K, GgmlType::Q6K] {
        let bytes = ty.bytes_for(rows * cols).unwrap();
        let mut data: Vec<u8> = (0..bytes).map(|i| ((i * 2654435761usize) >> 13) as u8).collect();
        let (_, b) = ty.block();
        for blk in data.chunks_exact_mut(b) {
            let at = match ty { GgmlType::Q6K => 208, _ => 0 };
            blk[at..at + 2].copy_from_slice(&0x2000u16.to_le_bytes());
        }
        let t = QTensor::new(ty, rows, cols, data).unwrap();
        let x: Vec<f32> = (0..cols).map(|i| (i % 7) as f32 - 3.0).collect();
        for n in [1usize, 5] {
            let xs: Vec<f32> = x.iter().cycle().take(cols * n).cloned().collect();
            let _ = t.matmul(&xs, n);
            let start = Instant::now();
            let reps = 20;
            for _ in 0..reps {
                std::hint::black_box(t.matmul(&xs, n));
            }
            let s = start.elapsed().as_secs_f64() / reps as f64;
            println!("{ty:?} n={n}: {:.2} ms, {:.1} GMAC/s, {:.1} GB/s of weights", s * 1e3, (rows * cols * n) as f64 / s / 1e9, bytes as f64 / s / 1e9);
        }
    }
}

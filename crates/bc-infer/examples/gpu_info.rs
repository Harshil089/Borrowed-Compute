//! Which GPU the engine found and how fast it reads weights: `BC_INFER_GPU=debug` says
//! why when there is none.
fn main() {
    match bc_infer::gpu::get() {
        Some(g) => {
            println!("copy {:.1} GB/s (read + write)", g.copy_bandwidth().unwrap() / 1e9);
            use bc_infer::quant::GgmlType::*;
            for ty in [Q4_0, Q4K, Q6K] {
                for n in [1, 2, 8, 16, 41, 64] {
                    println!("{ty:?} product of {n} rows: {:.1} GB/s of weights", g.product_rate(ty, n).unwrap() / 1e9);
                }
            }
            println!("{}: {:.1} GB/s, {:.1} GB for weights, unified {}", g.name, g.bandwidth().unwrap() / 1e9, g.memory() as f64 / 1e9, g.unified());
        }
        None => println!("no usable GPU"),
    }
}

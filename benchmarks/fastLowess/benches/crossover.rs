//! CPU/GPU Crossover Benchmarks.
//!
//! Benchmarks comparing CPU and GPU performance across a range of dataset sizes
//! to identify the crossover point where GPU parallelism outweighs overhead.
//!
//! Run with: `cargo bench --features gpu --bench crossover`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fastLowess::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;
use std::hint::black_box;

// ============================================================================
// Data Generation with Reproducible RNG
// ============================================================================

/// Generate smooth sinusoidal data with Gaussian noise.
fn generate_sine_data(size: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise_dist = Normal::new(0.0, 0.2).unwrap();

    let x: Vec<f64> = (0..size).map(|i| i as f64 * 10.0 / size as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + noise_dist.sample(&mut rng))
        .collect();
    (x, y)
}

// ============================================================================
// Benchmark Functions
// ============================================================================

fn bench_crossover_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover");
    group.sample_size(10); // Reduced sample size for large datasets

    let sizes = [100_000, 250_000, 500_000, 1_000_000, 2_000_000];

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));
        let (x, y) = generate_sine_data(size, 42);

        // CPU Benchmark
        group.bench_with_input(BenchmarkId::new("cpu", size), &size, |b, _| {
            b.iter(|| {
                Lowess::new()
                    .fraction(0.1)
                    .iterations(3)
                    .adapter(Batch)
                    .backend(CPU)
                    .build()
                    .unwrap()
                    .fit(black_box(&x), black_box(&y))
                    .unwrap()
            })
        });

        // GPU Benchmark
        group.bench_with_input(BenchmarkId::new("gpu", size), &size, |b, _| {
            b.iter(|| {
                Lowess::new()
                    .fraction(0.1)
                    .iterations(3)
                    .adapter(Batch)
                    .backend(GPU)
                    .build()
                    .unwrap()
                    .fit(black_box(&x), black_box(&y))
                    .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_crossover_gpu,);

criterion_main!(benches);

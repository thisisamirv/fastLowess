//! GPU-accelerated LOWESS benchmarks using Criterion.
//!
//! Benchmarks cover scenarios compatible with GPU backend limitations:
//! - Scalability (1K to 100K points)
//! - Algorithm parameters (fraction, iterations)
//! - Real-world scenarios (financial, scientific)
//! - Pathological cases (outliers, clustered data, high noise)
//!
//! GPU Backend Limitations:
//! - Only Tricube kernel function
//! - Only Bisquare robustness method
//! - No cross-validation
//! - No intervals
//! - No delta optimization
//! - No diagnostics
//!
//! Run with: `cargo bench --features gpu --bench benchmark_gpu`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fastLowess::prelude::*;
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use std::f64::consts::PI;
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

/// Generate data with outliers (5% of points are extreme).
fn generate_outlier_data(size: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise_dist = Normal::new(0.0, 0.2).unwrap();
    let outlier_dist = Uniform::new(-5.0, 5.0).unwrap();

    let x: Vec<f64> = (0..size).map(|i| i as f64 * 10.0 / size as f64).collect();
    let mut y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + noise_dist.sample(&mut rng))
        .collect();

    // Add 5% outliers
    let n_outliers = size / 20;
    for _ in 0..n_outliers {
        let idx = rng.random_range(0..size);
        y[idx] += outlier_dist.sample(&mut rng);
    }
    (x, y)
}

/// Generate financial time series (trending with volatility).
fn generate_financial_data(size: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let returns_dist = Normal::new(0.0005, 0.02).unwrap(); // Daily returns

    let x: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let mut y = vec![100.0]; // Starting price

    for _ in 1..size {
        let ret = returns_dist.sample(&mut rng);
        let new_price = y.last().unwrap() * (1.0 + ret);
        y.push(new_price);
    }
    (x, y)
}

/// Generate scientific measurement data (exponential decay with oscillations).
fn generate_scientific_data(size: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise_dist = Normal::new(0.0, 0.05).unwrap();

    let x: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let signal = (-xi * 0.3).exp() * (xi * 2.0 * PI).cos();
            signal + noise_dist.sample(&mut rng)
        })
        .collect();
    (x, y)
}

/// Generate clustered x-values (groups with tiny spacing).
fn generate_clustered_data(size: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise_dist = Normal::new(0.0, 0.1).unwrap();

    let x: Vec<f64> = (0..size)
        .map(|i| (i / 100) as f64 + (i % 100) as f64 * 1e-6)
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + noise_dist.sample(&mut rng))
        .collect();
    (x, y)
}

/// Generate high-noise data (SNR < 1).
fn generate_high_noise_data(size: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise_dist = Normal::new(0.0, 2.0).unwrap(); // High noise

    let x: Vec<f64> = (0..size).map(|i| i as f64 * 10.0 / size as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let signal = xi.sin() * 0.5;
            signal + noise_dist.sample(&mut rng)
        })
        .collect();
    (x, y)
}

// ============================================================================
// Benchmark Functions
// ============================================================================

fn bench_scalability_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_scalability");
    group.sample_size(50);

    for size in [1_000, 5_000, 10_000, 50_000, 100_000] {
        group.throughput(Throughput::Elements(size as u64));

        let (x, y) = generate_sine_data(size, 42);

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

fn bench_fraction_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_fraction");
    group.sample_size(100);

    let size = 5000;
    let (x, y) = generate_sine_data(size, 42);

    for frac in [0.05, 0.1, 0.2, 0.3, 0.5, 0.67] {
        group.bench_with_input(BenchmarkId::new("gpu", frac), &frac, |b, &frac| {
            b.iter(|| {
                Lowess::new()
                    .fraction(frac)
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

fn bench_iterations_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_iterations");
    group.sample_size(100);

    let size = 5000;
    let (x, y) = generate_outlier_data(size, 42);

    for iter in [0, 1, 2, 3, 5, 10] {
        group.bench_with_input(BenchmarkId::new("gpu", iter), &iter, |b, &iter| {
            b.iter(|| {
                Lowess::new()
                    .fraction(0.2)
                    .iterations(iter)
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

fn bench_financial_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_financial");
    group.sample_size(100);

    for size in [500, 1000, 5000, 10000] {
        let (x, y) = generate_financial_data(size, 42);

        group.bench_with_input(BenchmarkId::new("price_smoothing", size), &size, |b, _| {
            b.iter(|| {
                Lowess::new()
                    .fraction(0.1)
                    .iterations(2)
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

fn bench_scientific_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_scientific");
    group.sample_size(100);

    for size in [500, 1000, 5000, 10000] {
        let (x, y) = generate_scientific_data(size, 42);

        group.bench_with_input(BenchmarkId::new("spectroscopy", size), &size, |b, _| {
            b.iter(|| {
                Lowess::new()
                    .fraction(0.15)
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

fn bench_pathological_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_pathological");
    group.sample_size(50);

    let size = 5000;

    // Clustered data
    let (x_clustered, y_clustered) = generate_clustered_data(size, 42);
    group.bench_function("clustered", |b| {
        b.iter(|| {
            Lowess::new()
                .fraction(0.3)
                .iterations(2)
                .adapter(Batch)
                .backend(GPU)
                .build()
                .unwrap()
                .fit(black_box(&x_clustered), black_box(&y_clustered))
                .unwrap()
        })
    });

    // High noise
    let (x_noisy, y_noisy) = generate_high_noise_data(size, 42);
    group.bench_function("high_noise", |b| {
        b.iter(|| {
            Lowess::new()
                .fraction(0.5)
                .iterations(5)
                .adapter(Batch)
                .backend(GPU)
                .build()
                .unwrap()
                .fit(black_box(&x_noisy), black_box(&y_noisy))
                .unwrap()
        })
    });

    // Extreme outliers
    let (x_outlier, y_outlier) = generate_outlier_data(size, 42);
    group.bench_function("extreme_outliers", |b| {
        b.iter(|| {
            Lowess::new()
                .fraction(0.2)
                .iterations(10)
                .adapter(Batch)
                .backend(GPU)
                .build()
                .unwrap()
                .fit(black_box(&x_outlier), black_box(&y_outlier))
                .unwrap()
        })
    });

    // Constant y
    let x_const: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let y_const = vec![5.0; size];
    group.bench_function("constant_y", |b| {
        b.iter(|| {
            Lowess::new()
                .fraction(0.2)
                .iterations(2)
                .adapter(Batch)
                .backend(GPU)
                .build()
                .unwrap()
                .fit(black_box(&x_const), black_box(&y_const))
                .unwrap()
        })
    });

    group.finish();
}

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

criterion_group!(
    benches,
    bench_scalability_gpu,
    bench_fraction_gpu,
    bench_iterations_gpu,
    bench_financial_gpu,
    bench_scientific_gpu,
    bench_pathological_gpu,
    bench_crossover_gpu,
);

criterion_main!(benches);

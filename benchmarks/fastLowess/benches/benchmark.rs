//! Industry-level LOWESS benchmarks using Criterion.
//!
//! Benchmarks cover:
//! - Scalability (1K to 100K points)
//! - Algorithm parameters (fraction, iterations, delta)
//! - Real-world scenarios (financial, scientific, genomic)
//! - Pathological cases (outliers, clustered data, high noise)
//!
//! Run with: `cargo bench`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fastLowess::prelude::*;
use ndarray::Array1;
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

/// Generate genomic methylation data (beta values between 0 and 1).
fn generate_genomic_data(size: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise_dist = Normal::new(0.0, 0.1).unwrap();

    let x: Vec<f64> = (0..size).map(|i| (i * 1000) as f64).collect(); // CpG positions
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let base = 0.5 + (xi / 50000.0).sin() * 0.3;
            (base + noise_dist.sample(&mut rng)).clamp(0.0, 1.0)
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

fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group.sample_size(50);

    for size in [1_000, 5_000, 10_000, 50_000, 100_000] {
        group.throughput(Throughput::Elements(size as u64));

        let (x, y) = generate_sine_data(size, 42);

        // Parallel execution (fastLowess enhancement)
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |b, _| {
            b.iter(|| {
                Lowess::new()
                    .fraction(0.1)
                    .iterations(3)
                    .adapter(Batch)
                    .parallel(size >= 1000)
                    .build()
                    .unwrap()
                    .fit(black_box(&x), black_box(&y))
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_fraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("fraction");
    group.sample_size(100);

    let size = 5000;
    let (x, y) = generate_sine_data(size, 42);

    for frac in [0.05, 0.1, 0.2, 0.3, 0.5, 0.67] {
        group.bench_with_input(BenchmarkId::new("batch", frac), &frac, |b, &frac| {
            b.iter(|| {
                Lowess::new()
                    .fraction(frac)
                    .iterations(3)
                    .adapter(Batch)
                    .parallel(true) // size is 5000
                    .build()
                    .unwrap()
                    .fit(black_box(&x), black_box(&y))
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_iterations(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterations");
    group.sample_size(100);

    let size = 5000;
    let (x, y) = generate_outlier_data(size, 42);

    for iter in [0, 1, 2, 3, 5, 10] {
        group.bench_with_input(BenchmarkId::new("batch", iter), &iter, |b, &iter| {
            b.iter(|| {
                Lowess::new()
                    .fraction(0.2)
                    .iterations(iter)
                    .adapter(Batch)
                    .parallel(true) // size is 5000
                    .build()
                    .unwrap()
                    .fit(black_box(&x), black_box(&y))
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta");
    group.sample_size(50);

    let size = 10000;
    let (x, y) = generate_sine_data(size, 42);

    let delta_configs = [
        ("none", 0.0),
        ("small", 0.5),
        ("medium", 2.0),
        ("large", 10.0),
    ];

    for (name, delta) in delta_configs {
        group.bench_with_input(BenchmarkId::new("batch", name), &delta, |b, &delta| {
            b.iter(|| {
                Lowess::new()
                    .fraction(0.2)
                    .iterations(2)
                    .delta(delta)
                    .adapter(Batch)
                    .parallel(true) // size is 10000
                    .build()
                    .unwrap()
                    .fit(black_box(&x), black_box(&y))
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_financial(c: &mut Criterion) {
    let mut group = c.benchmark_group("financial");
    group.sample_size(100);

    for size in [500, 1000, 5000, 10000] {
        let (x, y) = generate_financial_data(size, 42);

        group.bench_with_input(BenchmarkId::new("price_smoothing", size), &size, |b, _| {
            b.iter(|| {
                Lowess::new()
                    .fraction(0.1)
                    .iterations(2)
                    .adapter(Batch)
                    .parallel(size >= 1000)
                    .build()
                    .unwrap()
                    .fit(black_box(&x), black_box(&y))
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_scientific(c: &mut Criterion) {
    let mut group = c.benchmark_group("scientific");
    group.sample_size(100);

    for size in [500, 1000, 5000, 10000] {
        let (x, y) = generate_scientific_data(size, 42);

        group.bench_with_input(BenchmarkId::new("spectroscopy", size), &size, |b, _| {
            b.iter(|| {
                Lowess::new()
                    .fraction(0.15)
                    .iterations(3)
                    .adapter(Batch)
                    .parallel(size >= 1000)
                    .build()
                    .unwrap()
                    .fit(black_box(&x), black_box(&y))
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_genomic(c: &mut Criterion) {
    let mut group = c.benchmark_group("genomic");
    group.sample_size(50);

    for size in [1000, 5000, 10000, 50000] {
        let (x, y) = generate_genomic_data(size, 42);

        group.bench_with_input(BenchmarkId::new("methylation", size), &size, |b, _| {
            b.iter(|| {
                Lowess::new()
                    .fraction(0.1)
                    .iterations(3)
                    .delta(100.0)
                    .adapter(Batch)
                    .parallel(size >= 1000)
                    .build()
                    .unwrap()
                    .fit(black_box(&x), black_box(&y))
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_pathological(c: &mut Criterion) {
    let mut group = c.benchmark_group("pathological");
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
                .parallel(true) // size is 5000
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
                .parallel(true) // size is 5000
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
                .parallel(true) // size is 5000
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
                .parallel(true) // size is 5000
                .build()
                .unwrap()
                .fit(black_box(&x_const), black_box(&y_const))
                .unwrap()
        })
    });

    group.finish();
}

fn bench_weight_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_functions");
    group.sample_size(100);

    let size = 5000;
    let (x, y) = generate_sine_data(size, 42);

    let weight_fns = [
        ("tricube", WeightFunction::Tricube),
        ("gaussian", WeightFunction::Gaussian),
        ("epanechnikov", WeightFunction::Epanechnikov),
    ];

    for (name, wf) in weight_fns {
        group.bench_with_input(BenchmarkId::new("kernel", name), &wf, |b, &wf| {
            b.iter(|| {
                Lowess::new()
                    .fraction(0.2)
                    .iterations(3)
                    .weight_function(wf)
                    .adapter(Batch)
                    .parallel(true)
                    .build()
                    .unwrap()
                    .fit(black_box(&x), black_box(&y))
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_ndarray_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray");
    group.sample_size(50);

    let size = 10_000;
    let (x_vec, y_vec) = generate_sine_data(size, 42);
    let x_arr = Array1::from_vec(x_vec.clone());
    let y_arr = Array1::from_vec(y_vec.clone());

    group.throughput(Throughput::Elements(size as u64));

    // Bench with Array1<f64> (ndarray)
    group.bench_function("ndarray_input", |b| {
        b.iter(|| {
            Lowess::new()
                .adapter(Batch)
                .parallel(true)
                .build()
                .unwrap()
                .fit(black_box(&x_arr), black_box(&y_arr))
                .unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_scalability,
    bench_fraction,
    bench_iterations,
    bench_delta,
    bench_financial,
    bench_scientific,
    bench_genomic,
    bench_pathological,
    bench_weight_functions,
    bench_ndarray_integration,
);

criterion_main!(benches);

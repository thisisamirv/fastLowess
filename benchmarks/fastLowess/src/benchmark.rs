// Rust fastLowess benchmark runner with JSON output for Python comparison.
//
// This is a standalone benchmark program (not using criterion) that outputs
// results in JSON format compatible with the Python statsmodels benchmark.
// To run, use `cargo run --release` for optimized builds.

use fastLowess::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

// ============================================================================
// Constants
// ============================================================================

const WARMUP_ITERATIONS: usize = 3;

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    name: String,
    size: usize,
    iterations: usize,
    mean_time_ms: f64,
    std_time_ms: f64,
    median_time_ms: f64,
    min_time_ms: f64,
    max_time_ms: f64,
}

impl BenchmarkResult {
    fn new(name: String, size: usize, iterations: usize) -> Self {
        Self {
            name,
            size,
            iterations,
            mean_time_ms: 0.0,
            std_time_ms: 0.0,
            median_time_ms: 0.0,
            min_time_ms: 0.0,
            max_time_ms: 0.0,
        }
    }

    fn compute_stats(&mut self, times: &[f64]) {
        if times.is_empty() {
            return;
        }

        // Convert to milliseconds
        let times_ms: Vec<f64> = times.iter().map(|&t| t * 1000.0).collect();

        self.mean_time_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;

        let variance = times_ms
            .iter()
            .map(|&t| {
                let diff = t - self.mean_time_ms;
                diff * diff
            })
            .sum::<f64>()
            / times_ms.len() as f64;
        self.std_time_ms = variance.sqrt();

        let mut sorted = times_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        self.median_time_ms = if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };

        self.min_time_ms = sorted[0];
        self.max_time_ms = sorted[sorted.len() - 1];
    }
}

// ============================================================================
// Data Generation
// ============================================================================

fn generate_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..size).map(|i| i as f64 * 10.0 / size as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let signal = xi.sin();
            let noise = ((i as f64 * 7.3).sin() * 0.5).sin() * 0.2;
            signal + noise
        })
        .collect();
    (x, y)
}

fn generate_data_with_outliers(size: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..size).map(|i| i as f64 * 10.0 / size as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

    // Add outliers (5% of points)
    let n_outliers = (size / 20).max(1);
    for i in 0..n_outliers {
        let idx = (i * size) / n_outliers;
        y[idx] += if i % 2 == 0 { 3.0 } else { -3.0 };
    }

    (x, y)
}

// ============================================================================
// Benchmark Functions
// ============================================================================

fn benchmark_basic_smoothing(sizes: &[usize], iterations: usize) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    for &size in sizes {
        println!("Benchmarking basic smoothing with size={}...", size);
        let mut result =
            BenchmarkResult::new(format!("basic_smoothing_{}", size), size, iterations);

        let (x, y) = generate_data(size);
        let mut times = Vec::new();

        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let _ = Lowess::new()
                .fraction(0.3)
                .iterations(3)
                .adapter(Batch)
                .parallel(true)
                .build()
                .unwrap()
                .fit(&x, &y)
                .unwrap();
        }

        // Benchmark
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = Lowess::new()
                .fraction(0.3)
                .iterations(3)
                .adapter(Batch)
                .parallel(true)
                .build()
                .unwrap()
                .fit(&x, &y)
                .unwrap();
            times.push(start.elapsed().as_secs_f64());
        }

        result.compute_stats(&times);
        println!(
            "  Mean: {:.2} ms ± {:.2} ms",
            result.mean_time_ms, result.std_time_ms
        );
        results.push(result);
    }

    results
}

fn benchmark_fraction_variations(size: usize, iterations: usize) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    let fractions = [0.1, 0.2, 0.3, 0.5, 0.67, 0.8];

    let (x, y) = generate_data(size);

    for &frac in &fractions {
        println!("Benchmarking fraction={}...", frac);
        let mut result = BenchmarkResult::new(format!("fraction_{}", frac), size, iterations);
        let mut times = Vec::new();

        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let _ = Lowess::new()
                .fraction(frac)
                .iterations(3)
                .adapter(Batch)
                .parallel(true)
                .build()
                .unwrap()
                .fit(&x, &y)
                .unwrap();
        }

        // Benchmark
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = Lowess::new()
                .fraction(frac)
                .iterations(3)
                .adapter(Batch)
                .parallel(true)
                .build()
                .unwrap()
                .fit(&x, &y)
                .unwrap();
            times.push(start.elapsed().as_secs_f64());
        }

        result.compute_stats(&times);
        println!(
            "  Mean: {:.2} ms ± {:.2} ms",
            result.mean_time_ms, result.std_time_ms
        );
        results.push(result);
    }

    results
}

fn benchmark_robustness_iterations(size: usize, iterations: usize) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    let niter_values = [0, 1, 2, 3, 5, 10];

    let (x, y) = generate_data_with_outliers(size);

    for &niter in &niter_values {
        println!("Benchmarking robustness iterations={}...", niter);
        let mut result = BenchmarkResult::new(format!("iterations_{}", niter), size, iterations);
        let mut times = Vec::new();

        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let _ = Lowess::new()
                .fraction(0.3)
                .iterations(niter)
                .adapter(Batch)
                .parallel(true)
                .build()
                .unwrap()
                .fit(&x, &y)
                .unwrap();
        }

        // Benchmark
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = Lowess::new()
                .fraction(0.3)
                .iterations(niter)
                .adapter(Batch)
                .parallel(true)
                .build()
                .unwrap()
                .fit(&x, &y)
                .unwrap();
            times.push(start.elapsed().as_secs_f64());
        }

        result.compute_stats(&times);
        println!(
            "  Mean: {:.2} ms ± {:.2} ms",
            result.mean_time_ms, result.std_time_ms
        );
        results.push(result);
    }

    results
}

fn benchmark_delta_parameter(size: usize, iterations: usize) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    let x: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

    let delta_configs = [
        ("delta_none", 0.0),
        ("delta_auto", -1.0), // Use None (auto) - handled specially below
        ("delta_small", 1.0),
        ("delta_large", 10.0),
    ];

    for (name, delta_val) in &delta_configs {
        println!("Benchmarking {} (delta={:.2})...", name, delta_val);
        let mut result = BenchmarkResult::new(name.to_string(), size, iterations);
        let mut times = Vec::new();

        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let builder = Lowess::new().fraction(0.3).iterations(2);
            let builder = if *delta_val < 0.0 {
                builder
            } else {
                builder.delta(*delta_val)
            };
            let _ = builder
                .adapter(Batch)
                .parallel(true)
                .build()
                .unwrap()
                .fit(&x, &y)
                .unwrap();
        }

        // Benchmark
        for _ in 0..iterations {
            let builder = Lowess::new().fraction(0.3).iterations(2);
            let builder = if *delta_val < 0.0 {
                builder
            } else {
                builder.delta(*delta_val)
            };

            let start = Instant::now();
            let _ = builder
                .adapter(Batch)
                .parallel(true)
                .build()
                .unwrap()
                .fit(&x, &y)
                .unwrap();
            times.push(start.elapsed().as_secs_f64());
        }

        result.compute_stats(&times);
        println!(
            "  Mean: {:.2} ms ± {:.2} ms",
            result.mean_time_ms, result.std_time_ms
        );
        results.push(result);
    }

    results
}

fn benchmark_pathological_cases(size: usize, iterations: usize) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // Clustered x values
    println!("Benchmarking clustered_x...");
    let x_clustered: Vec<f64> = (0..size)
        .map(|i| (i / 100) as f64 + (i % 100) as f64 * 1e-6)
        .collect();
    let y_clustered: Vec<f64> = x_clustered.iter().map(|&xi| xi.sin()).collect();

    let mut result = BenchmarkResult::new("clustered_x".to_string(), size, iterations);
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = Lowess::new()
            .fraction(0.5)
            .iterations(2)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x_clustered, &y_clustered)
            .unwrap();
    }

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = Lowess::new()
            .fraction(0.5)
            .iterations(2)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x_clustered, &y_clustered)
            .unwrap();
        times.push(start.elapsed().as_secs_f64());
    }

    result.compute_stats(&times);
    println!(
        "  Mean: {:.2} ms ± {:.2} ms",
        result.mean_time_ms, result.std_time_ms
    );
    results.push(result);

    // Extreme outliers
    println!("Benchmarking extreme_outliers...");
    let x_normal: Vec<f64> = (0..size).map(|i| i as f64 * 10.0 / size as f64).collect();
    let mut y_outliers: Vec<f64> = x_normal.iter().map(|&xi| xi.sin()).collect();
    for i in (0..size).step_by(50) {
        y_outliers[i] += if i % 100 == 0 { 100.0 } else { -100.0 };
    }

    let mut result = BenchmarkResult::new("extreme_outliers".to_string(), size, iterations);
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = Lowess::new()
            .fraction(0.3)
            .iterations(5)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x_normal, &y_outliers)
            .unwrap();
    }

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = Lowess::new()
            .fraction(0.3)
            .iterations(5)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x_normal, &y_outliers)
            .unwrap();
        times.push(start.elapsed().as_secs_f64());
    }

    result.compute_stats(&times);
    println!(
        "  Mean: {:.2} ms ± {:.2} ms",
        result.mean_time_ms, result.std_time_ms
    );
    results.push(result);

    // Constant y values
    println!("Benchmarking constant_y...");
    let y_constant = vec![5.0; size];

    let mut result = BenchmarkResult::new("constant_y".to_string(), size, iterations);
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = Lowess::new()
            .fraction(0.3)
            .iterations(2)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x_normal, &y_constant)
            .unwrap();
    }

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = Lowess::new()
            .fraction(0.3)
            .iterations(2)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x_normal, &y_constant)
            .unwrap();
        times.push(start.elapsed().as_secs_f64());
    }

    result.compute_stats(&times);
    println!(
        "  Mean: {:.2} ms ± {:.2} ms",
        result.mean_time_ms, result.std_time_ms
    );
    results.push(result);

    // High noise
    println!("Benchmarking high_noise...");
    let y_noisy: Vec<f64> = x_normal
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let signal = (xi / 10.0).sin() * 0.1;
            let noise = ((i as f64 * 7.3).sin() * 0.5).sin() * 2.0;
            signal + noise
        })
        .collect();

    let mut result = BenchmarkResult::new("high_noise".to_string(), size, iterations);
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = Lowess::new()
            .fraction(0.6)
            .iterations(3)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x_normal, &y_noisy)
            .unwrap();
    }

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = Lowess::new()
            .fraction(0.6)
            .iterations(3)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x_normal, &y_noisy)
            .unwrap();
        times.push(start.elapsed().as_secs_f64());
    }

    result.compute_stats(&times);
    println!(
        "  Mean: {:.2} ms ± {:.2} ms",
        result.mean_time_ms, result.std_time_ms
    );
    results.push(result);

    results
}

fn benchmark_realistic_scenarios(iterations: usize) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    let size = 1000;

    // Financial time series
    println!("Benchmarking financial_timeseries...");
    let x: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let trend = xi * 0.01;
            let volatility = (xi / 50.0).sin() * 0.5;
            let random_walk = ((i as f64 * 7.3).sin() * 0.5).sin() * 0.3;
            trend + volatility + random_walk
        })
        .collect();

    let mut result = BenchmarkResult::new("financial_timeseries".to_string(), size, iterations);
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = Lowess::new()
            .fraction(0.1)
            .iterations(2)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
    }

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = Lowess::new()
            .fraction(0.1)
            .iterations(2)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        times.push(start.elapsed().as_secs_f64());
    }

    result.compute_stats(&times);
    println!(
        "  Mean: {:.2} ms ± {:.2} ms",
        result.mean_time_ms, result.std_time_ms
    );
    results.push(result);

    // Scientific data
    println!("Benchmarking scientific_data...");
    let x_sci: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();
    let y_sci: Vec<f64> = x_sci
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let signal = (xi * 2.0 * PI).exp() * (xi * 10.0).cos();
            let noise = ((i as f64 * 13.7).sin() * 0.3).sin() * 0.1;
            signal + noise
        })
        .collect();

    let mut result = BenchmarkResult::new("scientific_data".to_string(), size, iterations);
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = Lowess::new()
            .fraction(0.2)
            .iterations(3)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x_sci, &y_sci)
            .unwrap();
    }

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = Lowess::new()
            .fraction(0.2)
            .iterations(3)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x_sci, &y_sci)
            .unwrap();
        times.push(start.elapsed().as_secs_f64());
    }

    result.compute_stats(&times);
    println!(
        "  Mean: {:.2} ms ± {:.2} ms",
        result.mean_time_ms, result.std_time_ms
    );
    results.push(result);

    // Genomic methylation
    println!("Benchmarking genomic_methylation...");
    let x_genomic: Vec<f64> = (0..size).map(|i| (i * 1000) as f64).collect();
    let y_genomic: Vec<f64> = x_genomic
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let local_mean = 0.5 + (xi / 5000.0).sin() * 0.2;
            let noise = ((i as f64 * 17.3).sin() * 0.3).sin() * 0.15;
            (local_mean + noise).clamp(0.0, 1.0)
        })
        .collect();

    let mut result = BenchmarkResult::new("genomic_methylation".to_string(), size, iterations);
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = Lowess::new()
            .fraction(0.2)
            .iterations(3)
            .delta(100.0)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x_genomic, &y_genomic)
            .unwrap();
    }

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = Lowess::new()
            .fraction(0.2)
            .iterations(3)
            .delta(100.0)
            .adapter(Batch)
            .parallel(true)
            .build()
            .unwrap()
            .fit(&x_genomic, &y_genomic)
            .unwrap();
        times.push(start.elapsed().as_secs_f64());
    }

    result.compute_stats(&times);
    println!(
        "  Mean: {:.2} ms ± {:.2} ms",
        result.mean_time_ms, result.std_time_ms
    );
    results.push(result);

    results
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("================================================================================");
    println!("RUST LOWESS BENCHMARK SUITE");
    println!("================================================================================\n");

    let mut all_results = std::collections::HashMap::new();

    // Core benchmarks
    println!("\n================================================================================");
    println!("CORE BENCHMARKS");
    println!("================================================================================\n");

    all_results.insert(
        "basic_smoothing",
        benchmark_basic_smoothing(&[100, 500, 1000, 5000, 10000], 10),
    );

    all_results.insert(
        "fraction_variations",
        benchmark_fraction_variations(1000, 10),
    );

    all_results.insert(
        "robustness_iterations",
        benchmark_robustness_iterations(1000, 10),
    );

    all_results.insert("delta_parameter", benchmark_delta_parameter(5000, 10));

    // Stress tests
    println!("\n================================================================================");
    println!("STRESS TESTS");
    println!("================================================================================\n");

    all_results.insert("pathological_cases", benchmark_pathological_cases(1000, 10));

    // Application scenarios
    println!("\n================================================================================");
    println!("APPLICATION SCENARIOS");
    println!("================================================================================\n");

    all_results.insert("realistic_scenarios", benchmark_realistic_scenarios(10));

    // Save results
    let json = serde_json::to_string_pretty(&all_results).unwrap();

    // Determine workspace root (parent of the crate directory) at compile time,
    // then create/output into workspace_root/output/
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // .../repo/.../rust
    let workspace_root = manifest_dir.parent().unwrap_or(&manifest_dir).to_path_buf(); // fall back to crate dir if no parent
    let out_dir = workspace_root.join("output");
    std::fs::create_dir_all(&out_dir).expect("failed to create output directory");

    let out_path = out_dir.join("rust_benchmark.json");
    let mut file = File::create(&out_path).expect("failed to create output file");
    file.write_all(json.as_bytes())
        .expect("failed to write results");

    println!("\n================================================================================");
    println!("Results saved to {}", out_path.display());
    println!("================================================================================");
}

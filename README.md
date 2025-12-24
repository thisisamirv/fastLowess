# fastLowess

[![Crates.io](https://img.shields.io/crates/v/fastLowess.svg)](https://crates.io/crates/fastLowess)
[![Documentation](https://docs.rs/fastLowess/badge.svg)](https://docs.rs/fastLowess)
[![License](https://img.shields.io/badge/License-AGPL--3.0%20OR%20Commercial-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

**High-performance parallel and GPU-accelerated LOWESS (Locally Weighted Scatterplot Smoothing) for Rust** — A high-level wrapper around the [`lowess`](https://github.com/thisisamirv/lowess) crate that adds rayon-based parallelism, GPU acceleration, and seamless ndarray integration.

> [!IMPORTANT]
> For a minimal, single-threaded, and `no_std` version, use base [`lowess`](https://github.com/thisisamirv/lowess).

## Features

- **Parallel by Default**: Multi-core regression fits via [rayon](https://crates.io/crates/rayon), achieving multiple orders of magnitude speedups on large datasets.
- **ndarray Integration**: Native support for `Array1<T>` and `ArrayView1<T>`.
- **Robust Statistics**: MAD-based scale estimation and IRLS with Bisquare, Huber, or Talwar weighting.
- **Uncertainty Quantification**: Point-wise standard errors, confidence intervals, and prediction intervals.
- **Optimized Performance**: Delta optimization for skipping dense regions and streaming/online modes.
- **Parameter Selection**: Built-in cross-validation for automatic smoothing fraction selection.

## Robustness Advantages

Built on the same core as `lowess`, this implementation is **more robust than statsmodels** due to:

### MAD-Based Scale Estimation

We use **Median Absolute Deviation (MAD)** for scale estimation, which is breakdown-point-optimal:

```text
s = median(|r_i - median(r)|)
```

### Boundary Padding

We apply **boundary policies** (Extend, Reflect, Zero) at dataset edges to maintain symmetric local neighborhoods, preventing the edge bias common in other implementations.

### Gaussian Consistency Factor

For precision in intervals, residual scale is computed using:

```text
sigma = 1.4826 * MAD
```

## Performance Advantages

Benchmarked against Python's `statsmodels`. Achieves **91-3914× faster performance** across all tested scenarios. The parallel implementation ensures that even at extreme scales (100k points), processing remains sub-12ms.

### Summary

The `fastLowess` crate demonstrates massive performance gains over Python's `statsmodels`, ranging from **136x to over 4300x** speedup. The addition of **parallel execution** (via Rayon) and optimized algorithm defaults makes it exceptionally well-suited for high-throughput data processing and large-scale datasets.

### Category Comparison

| Category         | Matched | Median Speedup | Mean Speedup |
|------------------|---------|----------------|--------------|
| **Scalability**  | 5       | **954×**       | 1637×        |
| **Fraction**     | 6       | **571×**       | 552×         |
| **Iterations**   | 6       | **564×**       | 567×         |
| **Pathological** | 4       | **551×**       | 538×         |
| **Financial**    | 4       | **385×**       | 448×         |
| **Scientific**   | 4       | **381×**       | 450×         |
| **Genomic**      | 4       | **23×**        | 27×          |
| **Delta**        | 4       | **5.7×**       | 7.8×         |

### Top 10 Rust Wins

| Benchmark        | statsmodels | fastLowess | Speedup   |
|------------------|-------------|------------|-----------|
| scale_100000     | 43.73s      | 10.1ms     | **4339×** |
| scale_50000      | 11.16s      | 5.26ms     | **2122×** |
| scale_10000      | 663.1ms     | 0.70ms     | **954×**  |
| scientific_10000 | 777.2ms     | 0.83ms     | **941×**  |
| financial_10000  | 497.1ms     | 0.56ms     | **885×**  |
| iterations_0     | 74.2ms      | 0.12ms     | **599×**  |
| financial_5000   | 170.9ms     | 0.29ms     | **595×**  |
| scientific_5000  | 268.5ms     | 0.45ms     | **593×**  |
| fraction_0.2     | 297.0ms     | 0.50ms     | **591×**  |
| scale_5000       | 229.9ms     | 0.39ms     | **590×**  |

## Installation

### CPU Backend (Default)

The default installation includes rayon-based parallelism and ndarray support:

```toml
[dependencies]
fastLowess = "0.3"
```

Or explicitly enable the `cpu` feature:

```toml
[dependencies]
fastLowess = { version = "0.3", features = ["cpu"] }
```

### GPU Backend

For GPU acceleration using `wgpu`, enable the `gpu` feature:

```toml
[dependencies]
fastLowess = { version = "0.3", features = ["gpu"] }
```

> [!NOTE]
> The GPU backend requires compatible GPU hardware and drivers. See the [Backend Comparison](#backend-comparison) section below for feature limitations.

## Quick Start

```rust
use fastLowess::prelude::*;
use ndarray::Array1;

fn main() -> Result<(), LowessError> {
    // Data as ndarray Array1
    let x = Array1::linspace(0.0, 10.0, 100);
    let y = x.mapv(|v| v.sin() + 0.1 * v);

    // Build the model (parallel by default)
    let result = Lowess::new()
        .fraction(0.5)
        .adapter(Batch)
        .parallel(true)
        .build()?
        .fit(&x, &y)?;

    println!("Smoothed values: {:?}", result.y);
    Ok(())
}
```

## Builder Methods

```rust
use fastLowess::prelude::*;

Lowess::new()
    // Smoothing span (0, 1]
    .fraction(0.5)

    // Robustness iterations
    .iterations(3)

    // Interpolation threshold
    .delta(0.01)

    // Kernel selection
    .weight_function(Tricube)

    // Robustness method
    .robustness_method(Bisquare)

    // Zero-weight fallback behavior
    .zero_weight_fallback(UseLocalMean)

    // Boundary handling (for edge effects)
    .boundary_policy(Extend)

    // Confidence intervals
    .confidence_intervals(0.95)

    // Prediction intervals
    .prediction_intervals(0.95)

    // Diagnostics
    .return_diagnostics()
    .return_residuals()
    .return_robustness_weights()

    // Cross-validation (for parameter selection)
    .cross_validate(KFold(5).with_fractions(&[0.3, 0.5, 0.7]).seed(123))

    // Convergence
    .auto_converge(1e-4)

    // Execution mode
    .adapter(Batch)

    // Backend (CPU or GPU)
    .backend(CPU)

    // Parallelism
    .parallel(true)

    // Build the model
    .build()?;
```

### Backend Comparison

| Backend    | Use Case         | Features              | Limitations         |
|------------|------------------|-----------------------|---------------------|
| CPU        | General          | All features          | None                |
| GPU (beta) | High-performance | Special circumstances | Only vanilla LOWESS |

> [!WARNING]
> **GPU Backend Limitations**: The GPU backend is currently in **Beta** and is limited to vanilla LOWESS and does not support all features of the CPU backend:
>
> - Only Tricube kernel function
> - Only Bisquare robustness method
> - Only Batch adapter
> - No cross-validation
> - No intervals
> - No edge handling (bias at edges, original LOWESS behavior)
> - No zero-weight fallback
> - No diagnostics
> - No streaming or online mode

1. **CPU Backend (`Backend::CPU`)**: The default and recommended choice. It is faster for all standard dense computations, supports all features (cross-validation, intervals, etc.), and has zero setup overhead.

2. **GPU Backend (`Backend::GPU`)**: Use **only** if you have a massive dataset (> 250,000 points) **AND** you are using the `delta` optimization (e.g., `delta(0.01)`). In this specific "sparse" scenario, the GPU scales better than the CPU. for dense computation, the CPU is still faster.

> [!NOTE]
> **GPU vs CPU Precision**: Results from the GPU backend are not guaranteed to be identical to the CPU backend due to:
>
> - Different floating-point precision
> - No padding at the edges in the GPU backend
> - Different scale estimation methods (MAD in CPU, MAR in GPU)

## Result Structure

```rust
pub struct LowessResult<T> {
    /// Sorted x values (independent variable)
    pub x: Vec<T>,

    /// Smoothed y values (dependent variable)
    pub y: Vec<T>,

    /// Point-wise standard errors of the fit
    pub standard_errors: Option<Vec<T>>,

    /// Confidence interval bounds (if computed)
    pub confidence_lower: Option<Vec<T>>,
    pub confidence_upper: Option<Vec<T>>,

    /// Prediction interval bounds (if computed)
    pub prediction_lower: Option<Vec<T>>,
    pub prediction_upper: Option<Vec<T>>,

    /// Residuals (y - fit)
    pub residuals: Option<Vec<T>>,

    /// Final robustness weights from outlier downweighting
    pub robustness_weights: Option<Vec<T>>,

    /// Detailed fit diagnostics (RMSE, R^2, Effective DF, etc.)
    pub diagnostics: Option<Diagnostics<T>>,

    /// Number of robustness iterations actually performed
    pub iterations_used: Option<usize>,

    /// Smoothing fraction used (optimal if selected via CV)
    pub fraction_used: T,

    /// RMSE scores for each fraction tested during CV
    pub cv_scores: Option<Vec<T>>,
}
```

> [!TIP]
> **Using with ndarray:** While the result struct uses `Vec<T>` for maximum compatibility, you can effortlessly convert any field to an `Array1` using `Array1::from_vec(result.y)`.

## Streaming Processing

For datasets that don't fit in memory:

```rust
use fastLowess::prelude::*;

let mut processor = Lowess::new()
    .fraction(0.3)
    .iterations(2)
    .adapter(Streaming)
    .parallel(true)   // Enable parallel chunk processing
    .chunk_size(1000)
    .overlap(100)
    .build()?;

// Process data in chunks
for chunk in data_chunks {
    let result = processor.process_chunk(&chunk.x, &chunk.y)?;
}

// Finalize processing
let final_result = processor.finalize()?;
```

## Online Processing

For real-time data streams:

```rust
use fastLowess::prelude::*;

let mut processor = Lowess::new()
    .fraction(0.2)
    .iterations(1)
    .adapter(Online)
    .parallel(false)  // Sequential for lowest per-point latency
    .window_capacity(100)
    .build()?;

// Process points as they arrive
for (x, y) in data_stream {
    if let Some(output) = processor.add_point(x, y)? {
        println!("Smoothed: {}", output.smoothed);
    }
}
```

## Parameter Selection Guide

### Fraction (Smoothing Span)

- **0.1-0.3**: Local, captures rapid changes (wiggly)
- **0.4-0.6**: Balanced, general-purpose
- **0.7-1.0**: Global, smooth trends only
- **Default: 0.67** (2/3, Cleveland's choice)
- **Use CV** when uncertain

### Robustness Iterations

- **0**: Clean data, speed critical
- **1-2**: Light contamination
- **3**: Default, good balance (recommended)
- **4-5**: Heavy outliers
- **>5**: Diminishing returns

### Kernel Function

- **Tricube** (default): Best all-around, smooth, efficient
- **Epanechnikov**: Theoretically optimal MSE
- **Gaussian**: Very smooth, no compact support
- **Uniform**: Fastest, least smooth (moving average)

### Delta Optimization

- **None**: Small datasets (n < 1000)
- **0.01 × range(x)**: Good starting point for dense data
- **Manual tuning**: Adjust based on data density

## Examples

Check the `examples` directory for advanced usage:

```bash
cargo run --example batch_smoothing
cargo run --example online_smoothing
cargo run --example streaming_smoothing
```

## MSRV

Rust **1.85.0** or later (2024 Edition).

## Validation

Validated against:

- **Python (statsmodels)**: Passed on 44 distinct test scenarios.
- **Original Paper**: Reproduces Cleveland (1979) results.

Check [Validation](https://github.com/thisisamirv/fastLowess/tree/bench/validation) for more information. Small variations in results are expected due to differences in scale estimation and padding.

## Related Work

- [lowess (Rust core)](https://github.com/thisisamirv/lowess)
- [fastLowess (Python wrapper)](https://github.com/thisisamirv/fastlowess-py)
- [fastLowess (R wrapper)](https://github.com/thisisamirv/fastlowess-R)

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

## License

Dual-licensed under **AGPL-3.0** (Open Source) or **Commercial License**.
Contact `<thisisamirv@gmail.com>` for commercial inquiries.

## References

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". *JASA*.
- Cleveland, W.S. (1981). "LOWESS: A Program for Smoothing Scatterplots". *The American Statistician*.

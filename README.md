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

| Category         | Matched | Median Speedup | Mean Speedup |
| :--------------- | :------ | :------------- | :----------- |
| **Scalability**  | 5       | **819×**       | 1482×        |
| **Pathological** | 4       | **503×**       | 476×         |
| **Iterations**   | 6       | **491×**       | 496×         |
| **Fraction**     | 6       | **464×**       | 447×         |
| **Financial**    | 4       | **351×**       | 418×         |
| **Scientific**   | 4       | **345×**       | 404×         |
| **Genomic**      | 4       | **22×**        | 26×          |
| **Delta**        | 4       | **5×**         | 6.8×         |

### Top 10 Performance Wins

| Benchmark          | statsmodels | fastLowess | Speedup   |
| :----------------- | :---------- | :--------- | :-------- |
| scale_100000       | 43.727s     | 11.2ms     | **3914×** |
| scale_50000        | 11.160s     | 5.74ms     | **1946×** |
| financial_10000    | 497.1ms     | 0.59ms     | **839×**  |
| scientific_10000   | 777.2ms     | 0.93ms     | **835×**  |
| scale_10000        | 663.1ms     | 0.81ms     | **819×**  |
| clustered          | 267.8ms     | 0.48ms     | **554×**  |
| scale_5000         | 229.9ms     | 0.42ms     | **554×**  |
| fraction_0.1       | 227.9ms     | 0.42ms     | **542×**  |
| fraction_0.05      | 197.2ms     | 0.37ms     | **536×**  |
| financial_5000     | 170.9ms     | 0.32ms     | **536×**  |

Check [Benchmarks](https://github.com/thisisamirv/fastLowess/tree/bench/benchmarks) for detailed comparisons.

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

| Backend | Use Case         | Features              | Limitations         |
|---------|------------------|-----------------------|---------------------|
| CPU     | General          | All features          | None                |
| GPU     | High-performance | Very fast             | Only vanilla LOWESS |

> [!WARNING]
> **GPU Backend Limitations**: The GPU backend is currently limited to vanilla LOWESS and does not support all features of the CPU backend:
>
> - Only Tricube kernel function
> - Only Bisquare robustness method
> - Only Batch adapter
> - No cross-validation
> - No intervals
> - No delta optimization
> - No edge handling (bias at edges, original LOWESS behavior)
> - No zero-weight fallback
> - No diagnostics
> - No streaming or online mode
>
> The GPU backend eliminates CPU-GPU data transfers during robustness iterations, removing synchronization overhead. **Recommended only for very large datasets** where performance is the main priority and edge bias or other features are not a concern.

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

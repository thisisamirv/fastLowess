# fastLowess

[![Crates.io](https://img.shields.io/crates/v/fastLowess.svg)](https://crates.io/crates/fastLowess)
[![Documentation](https://docs.rs/fastLowess/badge.svg)](https://docs.rs/fastLowess)
[![License](https://img.shields.io/badge/License-AGPL--3.0%20OR%20Commercial-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

**High-performance parallel LOWESS (Locally Weighted Scatterplot Smoothing) for Rust** — Built on top of the `lowess` crate with rayon-based parallelism and seamless ndarray integration.

## Why This Crate?

- ⚡ **Blazingly Fast**: 13-41× faster performance than Python's statsmodels with parallel execution
- 🚀 **Parallel by Default**: Automatic multi-core utilization via rayon
- 📊 **ndarray Integration**: First-class support for `Array1<T>` data types
- 🎯 **Production-Ready**: Comprehensive error handling, numerical stability, extensive testing
- 📈 **Feature-Rich**: Confidence/prediction intervals, multiple kernels, cross-validation
- 🔬 **Scientific**: Validated against R and Python implementations
- 🛠️ **Flexible**: Multiple robustness methods, streaming/online modes

## Relationship with `lowess` Crate

`fastLowess` is a high-level wrapper around the core [`lowess`](https://crates.io/crates/lowess) crate that adds:

- **Parallel execution** via [rayon](https://crates.io/crates/rayon) for multi-core systems
- **ndarray support** for seamless integration with scientific computing workflows
- **Extended API** with parallel-specific configuration options

If you need a dependency-free, `no_std`-compatible core implementation, use the base `lowess` crate directly.

## Quick Start

```rust
use fastLowess::prelude::*;

let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

// Basic smoothing (parallel by default)
let result = Lowess::new()
    .fraction(0.5)
    .adapter(Batch)
    .build()?
    .fit(&x, &y)?;

println!("Smoothed: {:?}", result.y);
# Ok::<(), LowessError>(())
```

## Installation

```toml
[dependencies]
fastLowess = "0.1"
```

## Features at a Glance

| Feature                  | Description                             | Use Case                      |
| ------------------------ | --------------------------------------- | ----------------------------- |
| **Parallel Execution**   | Multi-core processing via rayon         | Large datasets, speed         |
| **ndarray Support**      | Native `Array1<T>` compatibility        | Scientific computing          |
| **Robust Smoothing**     | IRLS with Bisquare/Huber/Talwar weights | Outlier-contaminated data     |
| **Confidence Intervals** | Point-wise standard errors & bounds     | Uncertainty quantification    |
| **Cross-Validation**     | Auto-select optimal fraction            | Unknown smoothing parameter   |
| **Multiple Kernels**     | Tricube, Epanechnikov, Gaussian, etc.   | Different smoothness profiles |
| **Streaming Mode**       | Constant memory usage                   | Very large datasets           |
| **Delta Optimization**   | Skip dense regions                      | 10× speedup on dense data     |

## Common Use Cases

### 1. Parallel Processing (Default)

```rust
use fastLowess::prelude::*;

# let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
# let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
// Parallel execution is enabled by default
let result = Lowess::new()
    .fraction(0.5)
    .iterations(3)
    .adapter(Batch)
    .build()?
    .fit(&x, &y)?;

println!("Smoothed values: {:?}", result.y);
# Ok::<(), LowessError>(())
```

### 2. ndarray Integration

```rust
use fastLowess::prelude::*;
use ndarray::Array1;

let x: Array1<f64> = Array1::linspace(0.0, 10.0, 100);
let y: Array1<f64> = x.mapv(|v| v.sin() + 0.1 * v);

// Works directly with ndarray types
let result = Lowess::new()
    .fraction(0.3)
    .adapter(Batch)
    .build()?
    .fit(&x, &y)?;
# Ok::<(), LowessError>(())
```

### 3. Explicit Parallel Control

```rust
use fastLowess::prelude::*;

# let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
# let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
// Disable parallelism for small datasets or debugging
let result = Lowess::new()
    .fraction(0.5)
    .adapter(Batch)
    .parallel(false)  // Force sequential execution
    .build()?
    .fit(&x, &y)?;
# Ok::<(), LowessError>(())
```

### 4. Robust Smoothing (Handle Outliers)

```rust
use fastLowess::prelude::*;

# let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
# let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
let result = Lowess::new()
    .fraction(0.3)
    .iterations(5)                // Robust iterations
    .return_robustness_weights()  // Return outlier weights
    .adapter(Batch)
    .build()?
    .fit(&x, &y)?;

// Check which points were downweighted
if let Some(weights) = &result.robustness_weights {
    for (i, &w) in weights.iter().enumerate() {
        if w < 0.1 {
            println!("Point {} is likely an outlier", i);
        }
    }
}
# Ok::<(), LowessError>(())
```

### 5. Uncertainty Quantification

```rust
use fastLowess::prelude::*;

# let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
# let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
let result = Lowess::new()
    .fraction(0.5)
    .weight_function(WeightFunction::Tricube)
    .confidence_intervals(0.95)
    .prediction_intervals(0.95)
    .adapter(Batch)
    .build()?
    .fit(&x, &y)?;

// Plot confidence bands
for i in 0..x.len() {
    println!("x={:.1}: y={:.2} CI=[{:.2}, {:.2}]",
        result.x[i],
        result.y[i],
        result.confidence_lower.as_ref().unwrap()[i],
        result.confidence_upper.as_ref().unwrap()[i]
    );
}
# Ok::<(), LowessError>(())
```

### 6. Automatic Parameter Selection

```rust
use fastLowess::prelude::*;

# let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
# let y = vec![2.0, 4.1, 5.9, 8.2, 9.8, 12.0, 14.1, 16.0];
// Let cross-validation find the optimal smoothing fraction
let result = Lowess::new()
    .cross_validate(&[0.2, 0.3, 0.5, 0.7], CrossValidationStrategy::KFold, Some(5))
    .adapter(Batch)
    .build()?
    .fit(&x, &y)?;

println!("Optimal fraction: {}", result.fraction_used);
println!("CV RMSE scores: {:?}", result.cv_scores);
# Ok::<(), LowessError>(())
```

### 7. Large Dataset Optimization

```rust
use fastLowess::prelude::*;

# let large_x: Vec<f64> = (0..5000).map(|i| i as f64).collect();
# let large_y: Vec<f64> = large_x.iter().map(|&x| x.sin()).collect();
// Enable all performance optimizations
let result = Lowess::new()
    .fraction(0.3)
    .delta(0.01)        // Skip dense regions
    .adapter(Batch)     // Parallel by default
    .build()?
    .fit(&large_x, &large_y)?;
# Ok::<(), LowessError>(())
```

### 8. Production Monitoring

```rust
use fastLowess::prelude::*;

# let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
# let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
let result = Lowess::new()
    .fraction(0.5)
    .iterations(3)
    .return_diagnostics()
    .adapter(Batch)
    .build()?
    .fit(&x, &y)?;

if let Some(diag) = &result.diagnostics {
    println!("RMSE: {:.4}", diag.rmse);
    println!("R²: {:.4}", diag.r_squared);
    println!("Effective DF: {:.2}", diag.effective_df);

    // Quality checks
    if diag.effective_df < 2.0 {
        eprintln!("Warning: Very low degrees of freedom");
    }
}
# Ok::<(), LowessError>(())
```

### 9. Convenience Constructors

Pre-configured builders for common scenarios:

```rust
use fastLowess::prelude::*;

# let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
# let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
// For noisy data with outliers
let result = Lowess::<f64>::robust().adapter(Batch).build()?.fit(&x, &y)?;

// For speed on clean data
let result = Lowess::<f64>::quick().adapter(Batch).build()?.fit(&x, &y)?;
# Ok::<(), LowessError>(())
```

## API Overview

### Builder Methods

```rust
use fastLowess::prelude::*;

Lowess::new()
    // Core parameters
    .fraction(0.5)                  // Smoothing span (0, 1], default: 0.67
    .iterations(3)                  // Robustness iterations, default: 3
    .delta(0.01)                    // Interpolation threshold
    
    // Parallel execution
    .parallel(true)                 // Enable/disable parallelism (default: true)
    
    // Kernel selection
    .weight_function(WeightFunction::Tricube)  // Default
    
    // Robustness method
    .robustness_method(RobustnessMethod::Bisquare)  // Default
    
    // Intervals & diagnostics
    .confidence_intervals(0.95)
    .prediction_intervals(0.95)
    .return_diagnostics()
    .return_residuals()
    .return_robustness_weights()
    
    // Parameter selection
    .cross_validate(&[0.3, 0.5, 0.7], CrossValidationStrategy::KFold, Some(5))
    
    // Convergence
    .auto_converge(1e-4)
    .max_iterations(20)
    
    // Execution mode
    .adapter(Batch)  // or Streaming, Online
    .build()?;       // Build the model
```

### Result Structure

```rust
pub struct LowessResult<T> {
    pub x: Vec<T>,                          // Sorted x values
    pub y: Vec<T>,                          // Smoothed y values
    pub standard_errors: Option<Vec<T>>,    // Point-wise SE
    pub confidence_lower: Option<Vec<T>>,   // CI lower bound
    pub confidence_upper: Option<Vec<T>>,   // CI upper bound
    pub prediction_lower: Option<Vec<T>>,   // PI lower bound
    pub prediction_upper: Option<Vec<T>>,   // PI upper bound
    pub residuals: Option<Vec<T>>,          // y - fitted
    pub robustness_weights: Option<Vec<T>>, // Final IRLS weights
    pub diagnostics: Option<Diagnostics<T>>,
    pub iterations_used: Option<usize>,     // Actual iterations
    pub fraction_used: T,                   // Selected fraction
    pub cv_scores: Option<Vec<T>>,          // CV RMSE per fraction
}
```

## Execution Modes

Choose the right execution mode based on your use case:

### Batch Processing (Standard)

For complete datasets in memory with full feature support:

```rust
use fastLowess::prelude::*;

# let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
# let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
let model = Lowess::new()
    .fraction(0.5)
    .confidence_intervals(0.95)
    .return_diagnostics()
    .adapter(Batch)  // Parallel by default
    .build()?;

let result = model.fit(&x, &y)?;
# Ok::<(), LowessError>(())
```

### Streaming Processing

For large datasets (>100K points) that don't fit in memory:

```rust
use fastLowess::prelude::*;

let mut processor = Lowess::new()
    .fraction(0.3)
    .iterations(2)
    .adapter(Streaming)
    .chunk_size(1000)   // Process 1000 points at a time
    .overlap(100)       // 100 points overlap between chunks
    .build()?;

// Process data in chunks
for chunk in data_chunks {
    let result = processor.process_chunk(&chunk.x, &chunk.y)?;
    // Handle results incrementally
}

let final_result = processor.finalize()?;
# Ok::<(), LowessError>(())
```

### Online/Incremental Processing

For real-time data streams with sliding window:

```rust
use fastLowess::prelude::*;

let mut processor = Lowess::new()
    .fraction(0.2)
    .iterations(1)
    .adapter(Online)
    .window_capacity(100)  // Keep last 100 points
    .build()?;

// Process points as they arrive
for (x, y) in data_stream {
    if let Some(output) = processor.add_point(x, y)? {
        println!("Smoothed: {}", output.smoothed);
    }
}
# Ok::<(), LowessError>(())
```

## Parallel Execution

### How It Works

`fastLowess` uses rayon for automatic parallelization across multiple CPU cores:

1. **Data partitioning**: Input data is divided across available threads
2. **Independent fits**: Each thread computes local polynomial fits for its partition
3. **Thread-safe collection**: Results are collected without locks via rayon's parallel iterators

### Performance Characteristics

| Dataset Size | Cores | Typical Speedup |
|--------------|-------|-----------------|
| 1,000        | 4     | 2-3×            |
| 10,000       | 4     | 3-4×            |
| 100,000      | 8     | 5-7×            |
| 1,000,000    | 8     | 6-8×            |

### When to Disable Parallelism

```rust
# use fastLowess::prelude::*;
# let x = vec![1.0, 2.0, 3.0];
# let y = vec![1.0, 2.0, 3.0];
// Use .parallel(false) for:
let result = Lowess::new()
    .fraction(0.5)
    .parallel(false)  // Disable parallelism
    .adapter(Batch)
    .build()?
    .fit(&x, &y)?;
# Ok::<(), LowessError>(())
```

Consider disabling parallelism when:

- **Small datasets** (<500 points): Thread overhead exceeds benefits
- **Debugging**: Sequential execution is easier to trace
- **Resource constraints**: Limiting CPU usage on shared systems
- **Single-core systems**: No benefit from parallelism

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

## Error Handling

```rust
use fastLowess::prelude::*;

# let x = vec![1.0, 2.0, 3.0];
# let y = vec![2.0, 4.0, 6.0];
match Lowess::new().adapter(Batch).build()?.fit(&x, &y) {
    Ok(result) => {
        println!("Success: {:?}", result.y);
    },
    Err(LowessError::EmptyInput) => {
        eprintln!("Empty input arrays");
    },
    Err(LowessError::MismatchedInputs { x_len, y_len }) => {
        eprintln!("Length mismatch: x={}, y={}", x_len, y_len);
    },
    Err(LowessError::InvalidFraction(f)) => {
        eprintln!("Invalid fraction: {} (must be in (0, 1])", f);
    },
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
# Ok::<(), LowessError>(())
```

## Examples

Comprehensive examples are available in the `examples/` directory:

- **`batch_smoothing.rs`** - Batch processing scenarios
  - Basic smoothing, robust outlier handling, uncertainty quantification
  - Cross-validation, diagnostics, kernel comparisons
  - Parallel vs sequential execution comparison

- **`online_smoothing.rs`** - Real-time processing
  - Basic streaming, sensor data simulation, outlier handling
  - Window size effects, memory-bounded processing, sliding window behavior

- **`streaming_smoothing.rs`** - Large dataset processing
  - Basic chunking, chunk size comparison, overlap strategies
  - Large dataset processing, outlier handling, file-based simulation

Run examples with:

```bash
cargo run --example batch_smoothing
cargo run --example online_smoothing
cargo run --example streaming_smoothing
```

## Feature Flags

- **`default`**: Standard configuration with parallel support
- **`dev`**: Exposes internal modules for testing

### Standard configuration

```toml
[dependencies]
fastLowess = "0.1"
```

## Validation

This implementation has been extensively validated against:

1. **R's stats::lowess**: Numerical agreement to machine precision
2. **Python's statsmodels**: Validated on 44 test scenarios
3. **Cleveland's original paper**: Reproduces published examples

## MSRV (Minimum Supported Rust Version)

Rust **1.85.0** or later (requires Rust Edition 2024).

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Bug reports and feature requests
- Pull request guidelines
- Development workflow
- Testing requirements

## License

This software is **dual-licensed** under:

1. **AGPL-3.0** — Free for open-source use with source disclosure requirements
2. **Commercial License** — For proprietary/closed-source applications

For commercial licensing inquiries, contact: <thisisamirv@gmail.com>

See [LICENSE](LICENSE) for full details.

## References

**Original papers:**

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". _Journal of the American Statistical Association_, 74(368): 829-836. [DOI:10.2307/2286407](https://doi.org/10.2307/2286407)

- Cleveland, W.S. (1981). "LOWESS: A Program for Smoothing Scatterplots by Robust Locally Weighted Regression". _The American Statistician_, 35(1): 54.

**Related implementations:**

- [R stats::lowess](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lowess.html)
- [Python statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html)
- [lowess crate](https://crates.io/crates/lowess) - Core Rust implementation

## Citation

```bibtex
@software{fastlowess_rust_2025,
  author = {Valizadeh, Amir},
  title = {fastLowess: High-performance parallel LOWESS for Rust},
  year = {2025},
  url = {https://github.com/thisisamirv/fastLowess},
  version = {0.1.0}
}
```

## Author

**Amir Valizadeh**  
📧 <thisisamirv@gmail.com>  
🔗 [GitHub](https://github.com/thisisamirv/fastLowess)

---

**Keywords**: LOWESS, LOESS, local regression, nonparametric regression, smoothing, robust statistics, time series, bioinformatics, genomics, signal processing, parallel, ndarray, rayon

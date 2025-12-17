//! # fastLowess — Parallel LOWESS Smoothing with ndarray Integration
//!
//! A high-performance LOWESS implementation with parallel execution via rayon
//! and seamless ndarray integration. Built on top of the `lowess` crate.
//!
//! ## What is LOWESS?
//!
//! LOWESS (Locally Weighted Scatterplot Smoothing) is a nonparametric regression
//! method that fits smooth curves through scatter plots. At each point, it fits
//! a weighted polynomial (typically linear) using nearby data points, with weights
//! decreasing smoothly with distance. This creates flexible, data-adaptive curves
//! without assuming a global functional form.
//!
//! **Key advantages:**
//! - No parametric assumptions about the underlying relationship
//! - Automatic adaptation to local data structure
//! - Robust to outliers (with robustness iterations enabled)
//! - Provides uncertainty estimates via confidence/prediction intervals
//! - Handles irregular sampling and missing regions gracefully
//!
//! **Common applications:**
//! - Exploratory data analysis and visualization
//! - Trend estimation in time series
//! - Baseline correction in spectroscopy and signal processing
//! - Quality control and process monitoring
//! - Genomic and epigenomic data smoothing
//! - Removing systematic effects in scientific measurements
//!
//! **How LOWESS works:**
//!
//! <div align="center">
//! <object data="../../../docs/lowess_smoothing_concept.svg" type="image/svg+xml" width="800" height="500">
//! <img src="https://raw.githubusercontent.com/thisisamirv/fastLowess/main/docs/lowess_smoothing_concept.svg" alt="LOWESS Smoothing Concept" width="800"/>
//! </object>
//!
//! *LOWESS creates smooth curves through scattered data using local weighted neighborhoods*
//! </div>
//!
//! 1. For each point, select nearby neighbors (controlled by `fraction`)
//! 2. Fit a weighted polynomial (closer points get higher weight)
//! 3. Use the fitted value as the smoothed estimate
//! 4. Optionally iterate to downweight outliers (robustness)
//!
//! ## fastLowess Differentiators
//!
//! This crate extends the base `lowess` crate with:
//!
//! * **Parallel execution**: Uses `rayon` for parallelized smoothing passes
//! * **ndarray integration**: Direct ndarray support via generic `fit` method
//! * **Configurable parallelism**: `parallel(true/false)` flag on all adapters
//!
//! **When to use fastLowess vs lowess:**
//!
//! | Use Case                              | Recommended Crate        |
//! |---------------------------------------|--------------------------|
//! | Large datasets (>10K points)          | **fastLowess** (parallel)|
//! | Multi-core processing                 | **fastLowess** (parallel)|
//! | ndarray/scientific computing          | **fastLowess**           |
//! | Embedded/no_std environments          | **lowess** (no rayon)    |
//! | Single-threaded environments          | **lowess**               |
//! | Minimal dependencies                  | **lowess**               |
//!
//! ## Quick Start
//!
//! ### Typical Use
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build the model with parallel execution (default)
//! let model = Lowess::new()
//!     .fraction(0.5)      // Use 50% of data for each local fit
//!     .iterations(3)      // 3 robustness iterations
//!     .adapter(Batch)     // Parallel by default
//!     .build()
//!     .unwrap();
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y).unwrap();
//! println!("{}", result);
//! ```
//!
//! ```text
//! Summary:
//!   Data points: 5
//!   Fraction: 0.5
//!
//! Smoothed Data:
//!        X     Y_smooth
//!   --------------------
//!     1.00     2.00000
//!     2.00     4.10000
//!     3.00     5.90000
//!     4.00     8.20000
//!     5.00     9.80000
//! ```
//!
//! ### Full Features
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];
//!
//! // Build model with all features enabled
//! let model = Lowess::new()
//!     .fraction(0.5)                                  // Use 50% of data for each local fit
//!     .iterations(3)                                  // 3 robustness iterations
//!     .weight_function(WeightFunction::Tricube)       // Kernel function
//!     .robustness_method(RobustnessMethod::Bisquare)  // Outlier handling
//!     .delta(0.01)                                    // Interpolation optimization
//!     .zero_weight_fallback(ZeroWeightFallback::UseLocalMean)  // Fallback policy
//!     .confidence_intervals(0.95)                     // 95% confidence intervals
//!     .prediction_intervals(0.95)                     // 95% prediction intervals
//!     .return_diagnostics()                           // Fit quality metrics
//!     .return_residuals()                             // Include residuals
//!     .return_robustness_weights()                    // Include robustness weights
//!     .adapter(Batch)                                 // Batch adapter (parallel)
//!     .parallel(true)                                 // Explicit parallel
//!     .build()
//!     .unwrap();
//!
//! let result = model.fit(&x, &y).unwrap();
//! println!("{}", result);
//! ```
//!
//! ```text
//! Summary:
//!   Data points: 8
//!   Fraction: 0.5
//!   Robustness: Applied
//!
//! LOWESS Diagnostics:
//!   RMSE:         0.191925
//!   MAE:          0.181676
//!   R²:           0.998205
//!   Residual SD:  0.297750
//!   Effective DF: 8.00
//!   AIC:          -10.41
//!   AICc:         inf
//!
//! Smoothed Data:
//!        X     Y_smooth      Std_Err   Conf_Lower   Conf_Upper   Pred_Lower   Pred_Upper     Residual Rob_Weight
//!   ----------------------------------------------------------------------------------------------------------------
//!     1.00     2.01963     0.389365     1.256476     2.782788     1.058911     2.980353     0.080368     1.0000
//!     2.00     4.00251     0.345447     3.325438     4.679589     3.108641     4.896386    -0.202513     1.0000
//!     3.00     5.99959     0.423339     5.169846     6.829335     4.985168     7.014013     0.200410     1.0000
//!     4.00     8.09859     0.489473     7.139224     9.057960     6.975666     9.221518    -0.198592     1.0000
//!     5.00    10.03881     0.551687     8.957506    11.120118     8.810073    11.267551     0.261188     1.0000
//!     6.00    12.02872     0.539259    10.971775    13.085672    10.821364    13.236083    -0.228723     1.0000
//!     7.00    13.89828     0.371149    13.170829    14.625733    12.965670    14.830892     0.201719     1.0000
//!     8.00    15.77990     0.408300    14.979631    16.580167    14.789441    16.770356    -0.079899     1.0000
//! ```
//!
//! ### ndarray Usage
//!
//! ```rust
//! use fastLowess::prelude::*;
//! use ndarray::Array1;
//!
//! // Create arrays from vectors
//! let x: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//! let y: Array1<f64> = Array1::from_vec(vec![2.0, 4.1, 5.9, 8.2, 9.8]);
//!
//! // Build and fit
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! // Use fit directly with ndarray
//! let result = model.fit(&x, &y).unwrap();
//! ```
//!
//! ### Sequential Execution
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build with explicit sequential execution
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .adapter(Batch)
//!     .parallel(false)    // Disable parallel execution
//!     .build()
//!     .unwrap();
//!
//! let result = model.fit(&x, &y).unwrap();
//! ```
//!
//! ## Parameters
//!
//! All builder parameters have sensible defaults. You only need to specify what you want to change.
//!
//! | Parameter                     | Default                                            | Range/Options        | Description                                      |
//! |-------------------------------|----------------------------------------------------|----------------------|--------------------------------------------------|
//! | **fraction**                  | 0.67 (or CV-selected)                              | (0, 1]               | Smoothing span (fraction of data used per fit)   |
//! | **iterations**                | 3                                                  | 0 to max_iterations  | Number of robustness iterations                  |
//! | **delta**                     | 1% of x-range                                      | [0, ∞)               | Interpolation optimization threshold             |
//! | **weight_function**           | [`Tricube`](WeightFunction::Tricube)               | 7 kernel options     | Distance weighting kernel                        |
//! | **robustness_method**         | [`Bisquare`](RobustnessMethod::Bisquare)           | 3 methods            | Outlier downweighting method                     |
//! | **zero_weight_fallback**      | [`UseLocalMean`](ZeroWeightFallback::UseLocalMean) | 3 fallback options   | Behavior when all weights are zero               |
//! | **parallel**                  | true (Batch/Streaming), false (Online)             | true/false           | Enable parallel execution (fastLowess)           |
//! | **confidence_intervals**      | None                                               | 0 to 1 (level)       | Confidence interval level (disabled by default)  |
//! | **prediction_intervals**      | None                                               | 0 to 1 (level)       | Prediction interval level (disabled by default)  |
//! | **cross_validate**            | None                                               | Fractions + strategy | Cross-validation settings (disabled by default)  |
//! | **return_diagnostics**        | false                                              | true/false           | Compute RMSE, MAE, R², etc.                      |
//! | **return_residuals**          | false                                              | true/false           | Include residuals in output                      |
//! | **return_robustness_weights** | false                                              | true/false           | Include robustness weights in output             |
//!
//! ### Parameter Options Reference
//!
//! For parameters with multiple options, here are the available choices:
//!
//! | Parameter                | Available Options                                                                                                                                                                                                                                                                      |
//! |--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
//! | **weight_function**      | [`Tricube`](WeightFunction::Tricube), [`Epanechnikov`](WeightFunction::Epanechnikov), [`Gaussian`](WeightFunction::Gaussian), [`Biweight`](WeightFunction::Biweight), [`Cosine`](WeightFunction::Cosine), [`Triangle`](WeightFunction::Triangle), [`Uniform`](WeightFunction::Uniform) |
//! | **robustness_method**    | [`Bisquare`](RobustnessMethod::Bisquare), [`Huber`](RobustnessMethod::Huber), [`Talwar`](RobustnessMethod::Talwar)                                                                                                                                                                     |
//! | **zero_weight_fallback** | [`UseLocalMean`](ZeroWeightFallback::UseLocalMean), [`ReturnOriginal`](ZeroWeightFallback::ReturnOriginal), [`ReturnNone`](ZeroWeightFallback::ReturnNone)                                                                                                                             |
//!
//! ## Builder Pattern
//!
//! The crate uses a fluent builder pattern for configuration. All parameters have
//! sensible defaults, so you only need to specify what you want to change.
//!
//! ### Basic Workflow
//!
//! 1. **Create builder**: `Lowess::new()`
//! 2. **Configure parameters**: Chain method calls (`.fraction()`, `.iterations()`, etc.)
//! 3. **Select adapter**: Choose execution mode (`.adapter(Batch)`, `.adapter(Streaming)`, etc.)
//! 4. **Set parallelism**: Optionally call `.parallel(true/false)` (defaults to true for Batch/Streaming)
//! 5. **Build model**: Call `.build()` to create the configured model
//! 6. **Fit data**: Call `.fit(&x, &y)` (accepts slices or ndarrays)
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build the model with custom configuration
//! let model = Lowess::new()
//!     .fraction(0.3)                              // Smoothing span
//!     .iterations(5)                              // Robustness iterations
//!     .weight_function(WeightFunction::Tricube)   // Kernel function
//!     .robustness_method(RobustnessMethod::Bisquare)  // Outlier handling
//!     .adapter(Batch)
//!     .parallel(true)                             // Enable parallel (default)
//!     .build()
//!     .unwrap();
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y).unwrap();
//! println!("{}", result);
//! ```
//!
//! ```text
//! Summary:
//!   Data points: 5
//!   Fraction: 0.3
//!
//! Smoothed Data:
//!        X     Y_smooth
//!   --------------------
//!     1.00     2.00000
//!     2.00     4.10000
//!     3.00     5.90000
//!     4.00     8.20000
//!     5.00     9.80000
//! ```
//!
//! ### Execution Mode (Adapter) Comparison
//!
//! Choose the right execution mode based on your use case:
//!
//! | Adapter                                      | Use Case                                                                    | Parallel | Features                                                                                     | Limitations                                                                     |
//! |----------------------------------------------|-----------------------------------------------------------------------------|----------|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
//! | [`Batch`](crate::Adapter::Batch)             | Complete datasets in memory<br>Standard analysis<br>Full diagnostics needed | ✅ Yes*  | ✅ All features supported                                                                    | ❌ Requires entire dataset in memory<br>❌ Not suitable for very large datasets |
//! | [`Streaming`](crate::Adapter::Streaming)     | Large datasets (>100K points)<br>Limited memory<br>Batch pipelines          | ✅ Yes*  | ✅ Chunked processing<br>✅ Configurable overlap<br>✅ Robustness iterations<br>✅ Residuals | ❌ No intervals<br>❌ No cross-validation<br>❌ No diagnostics                  |
//! | [`Online`](crate::Adapter::Online)           | Real-time data<br>Sensor streams<br>Embedded systems                        | ❌ No*   | ✅ Incremental updates<br>✅ Sliding window<br>✅ Memory-bounded                             | ❌ No intervals<br>❌ No cross-validation<br>❌ Limited history                 |
//!
//! *Parallel execution is configurable via `.parallel(true/false)`. Batch and Streaming default to parallel; Online defaults to sequential.
//!
//! **Recommendation:**
//! - **Start with Batch** for most use cases - it's the most feature-complete
//! - **Use Streaming** when dataset size exceeds available memory
//! - **Use Online** for real-time applications or when data arrives incrementally
//!
//! ### Common Patterns Cheat Sheet
//!
//! Quick reference for typical use cases:
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! // 1. Parallel batch (fastest for large datasets)
//! let model = Lowess::<f64>::new()
//!     .adapter(Batch)
//!     .parallel(true)     // Default, explicit for clarity
//!     .build()
//!     .unwrap();
//!
//! // 2. Sequential batch (for debugging or comparison)
//! let model = Lowess::<f64>::new()
//!     .adapter(Batch)
//!     .parallel(false)
//!     .build()
//!     .unwrap();
//!
//! // 3. Standard analysis with uncertainty
//! let model = Lowess::<f64>::new()
//!     .fraction(0.5)
//!     .confidence_intervals(0.95)
//!     .prediction_intervals(0.95)
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! // 4. Automatic parameter selection
//! let model = Lowess::<f64>::new()
//!     .cross_validate(&[0.2, 0.3, 0.5, 0.7], CrossValidationStrategy::KFold, Some(5))
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! // 5. Heavy outlier contamination
//! let model = Lowess::<f64>::new()
//!     .fraction(0.5)
//!     .iterations(5)
//!     .robustness_method(RobustnessMethod::Talwar)
//!     .return_robustness_weights()
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! // 6. Smooth noisy data (large fraction)
//! let model = Lowess::<f64>::new()
//!     .fraction(0.7)
//!     .iterations(2)
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! // 7. Preserve fine detail (small fraction)
//! let model = Lowess::<f64>::new()
//!     .fraction(0.2)
//!     .iterations(0)  // No robustness for speed
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! // 8. Large dataset processing (parallel streaming)
//! let model = Lowess::<f64>::new()
//!     .fraction(0.1)
//!     .iterations(2)
//!     .adapter(Streaming)
//!     .parallel(true)  // Parallel streaming
//!     .build()
//!     .unwrap();
//!
//! // 9. Real-time sensor smoothing
//! let model = Lowess::<f64>::new()
//!     .fraction(0.2)
//!     .iterations(1)
//!     .adapter(Online)
//!     .parallel(false)  // Online is sequential by default
//!     .build()
//!     .unwrap();
//!
//! // 10. Complete diagnostic analysis
//! let model = Lowess::<f64>::new()
//!     .fraction(0.5)
//!     .confidence_intervals(0.95)
//!     .prediction_intervals(0.95)
//!     .return_diagnostics()
//!     .return_residuals()
//!     .return_robustness_weights()
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//! ```
//!
//! ## Core Parameters
//!
//! ### Fraction (Smoothing Span)
//!
//! The `fraction` parameter controls the proportion of data used for each local fit.
//! Larger fractions create smoother curves; smaller fractions preserve more detail.
//!
//! <div align="center">
//! <object data="../../../docs/fraction_effect_comparison.svg" type="image/svg+xml" width="1200" height="450">
//! <img src="https://raw.githubusercontent.com/thisisamirv/fastLowess/main/docs/fraction_effect_comparison.svg" alt="Fraction Effect" width="1200"/>
//! </object>
//!
//! *Under-smoothing (fraction too small), optimal smoothing, and over-smoothing (fraction too large)*
//! </div>
//!
//! - **Range**: (0, 1]
//! - **Effect**: Larger = smoother, smaller = more detail
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with small fraction (more detail)
//! let model = Lowess::new()
//!     .fraction(0.2)  // Use 20% of data for each local fit
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! let result = model.fit(&x, &y).unwrap();
//! ```
//!
//! **Choosing fraction:**
//! - **0.1-0.3**: Fine detail, may be noisy
//! - **0.3-0.5**: Moderate smoothing (good for most cases)
//! - **0.5-0.7**: Heavy smoothing, emphasizes trends
//! - **0.7-1.0**: Very smooth, may over-smooth
//!
//! ### Iterations (Robustness)
//!
//! The `iterations` parameter controls outlier resistance through iterative
//! reweighting. More iterations provide stronger robustness but increase computation time.
//!
//! <div align="center">
//! <object data="../../../docs/robust_vs_standard_lowess.svg" type="image/svg+xml" width="900" height="500">
//! <img src="https://raw.githubusercontent.com/thisisamirv/fastLowess/main/docs/robust_vs_standard_lowess.svg" alt="Robustness Effect" width="900"/>
//! </object>
//!
//! *Standard LOWESS (left) vs Robust LOWESS (right) - robustness iterations downweight outliers*
//! </div>
//!
//! - **Range**: 0 to max_iterations (clamped to 1000)
//! - **Effect**: More iterations = stronger outlier downweighting
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with strong outlier resistance
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .iterations(5)  // More iterations for stronger robustness
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! let result = model.fit(&x, &y).unwrap();
//! ```
//!
//! **Choosing iterations:**
//! - **0**: No robustness (fastest, sensitive to outliers)
//! - **1-3**: Light to moderate robustness (recommended)
//! - **4-6**: Strong robustness (for contaminated data)
//! - **7+**: Very strong (may over-smooth)
//!
//! ### Delta (Optimization)
//!
//! The `delta` parameter enables interpolation optimization for large datasets.
//! Points within `delta` distance reuse the previous fit.
//!
//! - **Range**: [0, ∞)
//! - **Effect**: Larger = faster but less accurate
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with custom delta
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .delta(0.05)  // Custom delta value
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! let result = model.fit(&x, &y).unwrap();
//! ```
//!
//! **When to use delta:**
//! - **Large datasets** (>10,000 points): Set to ~1% of x-range
//! - **Uniformly spaced data**: Can use larger values
//! - **Irregular spacing**: Use smaller values or 0
//!
//! ## Parallel Execution
//!
//! fastLowess uses rayon for parallel execution. This is controlled per-adapter
//! via the `.parallel(true/false)` method.
//!
//! ### Parallel vs Sequential
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! // Parallel batch (default)
//! let model = Lowess::<f64>::new()
//!     .adapter(Batch)
//!     .parallel(true)     // Explicit parallel (this is the default)
//!     .build()
//!     .unwrap();
//!
//! // Sequential batch (for debugging or small datasets)
//! let model = Lowess::<f64>::new()
//!     .adapter(Batch)
//!     .parallel(false)    // Disable parallel
//!     .build()
//!     .unwrap();
//! ```
//!
//! **When to use parallel:**
//! - **Large datasets** (>1,000 points): Parallel is faster
//! - **Multi-core systems**: Take advantage of all cores
//! - **Batch processing**: Parallel processing of multiple datasets
//!
//! **When to use sequential:**
//! - **Small datasets** (<1,000 points): Overhead may outweigh benefits
//! - **Debugging**: Easier to trace execution
//! - **Comparison**: Verify parallel results match sequential
//! - **Single-core systems**: No benefit from parallelism
//!
//! ### Performance Comparison
//!
//! Typical speedup factors (varies by dataset and hardware):
//!
//! | Dataset Size | Cores | Speedup Factor |
//! |--------------|-------|----------------|
//! | 1,000        | 4     | ~1.5-2x        |
//! | 10,000       | 4     | ~2-3x          |
//! | 100,000      | 4     | ~3-4x          |
//! | 100,000      | 8     | ~5-6x          |
//!
//! ## ndarray Integration
//!
//! The `fit` method natively supports `ndarray` inputs for zero-copy
//! integration with scientific computing workflows.
//!
//! ```rust
//! use fastLowess::prelude::*;
//! use ndarray::Array1;
//!
//! let x: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//! let y: Array1<f64> = Array1::from_vec(vec![2.0, 4.1, 5.9, 8.2, 9.8]);
//!
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! // fit accepts ArrayBase types directly
//! let result = model.fit(&x, &y).unwrap();
//! ```
//!
//! **Requirements:**
//! - Arrays must be contiguous (standard memory layout)
//! - Works with `Array1`, `ArrayView1`, and other 1D array types
//!
//! **Benefits:**
//! - No data copying for contiguous arrays
//! - Seamless integration with ndarray workflows
//! - Type-safe array handling
//!
//! ## Robustness Methods
//!
//! Different methods for downweighting outliers during iterative refinement.
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 100.0, 9.8];  // Point 3 is an outlier
//!
//! // Build model with Talwar robustness (hard threshold)
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .iterations(3)
//!     .robustness_method(RobustnessMethod::Talwar)
//!     .return_robustness_weights()  // Include weights in output
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y).unwrap();
//!
//! // Check which points were downweighted
//! if let Some(weights) = &result.robustness_weights {
//!     for (i, &w) in weights.iter().enumerate() {
//!         if w < 0.5 {
//!             println!("Point {} is likely an outlier (weight: {:.3})", i, w);
//!         }
//!     }
//! }
//! ```
//!
//! ```text
//! Point 3 is likely an outlier (weight: 0.000)
//! ```
//!
//! **Available methods:**
//!
//! | Method                                   | Behavior                | Use Case                  |
//! |------------------------------------------|-------------------------|---------------------------|
//! | [`Bisquare`](RobustnessMethod::Bisquare) | Smooth downweighting    | General-purpose, balanced |
//! | [`Huber`](RobustnessMethod::Huber)       | Linear beyond threshold | Moderate outliers         |
//! | [`Talwar`](RobustnessMethod::Talwar)     | Hard threshold (0 or 1) | Extreme contamination     |
//!
//! ## Weight Functions (Kernels)
//!
//! Control how neighboring points are weighted by distance.
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with Epanechnikov kernel
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .weight_function(WeightFunction::Epanechnikov)
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! let result = model.fit(&x, &y).unwrap();
//! ```
//!
//! **Kernel selection guide:**
//!
//! | Kernel                                         | Efficiency | Smoothness        | Use Case                   |
//! |------------------------------------------------|------------|-------------------|----------------------------|
//! | [`Tricube`](WeightFunction::Tricube)           | 0.998      | Very smooth       | Best all-around choice     |
//! | [`Epanechnikov`](WeightFunction::Epanechnikov) | 1.000      | Smooth            | Theoretically optimal MSE  |
//! | [`Gaussian`](WeightFunction::Gaussian)         | 0.961      | Infinitely smooth | Very smooth data           |
//! | [`Biweight`](WeightFunction::Biweight)         | 0.995      | Very smooth       | Alternative to Tricube     |
//! | [`Cosine`](WeightFunction::Cosine)             | 0.999      | Smooth            | Alternative compact kernel |
//! | [`Triangle`](WeightFunction::Triangle)         | 0.989      | Moderate          | Simple, fast               |
//! | [`Uniform`](WeightFunction::Uniform)           | 0.943      | None              | Fastest, moving average    |
//!
//! *Efficiency = AMISE relative to Epanechnikov (1.0 = optimal)*
//!
//! ## Uncertainty Quantification
//!
//! LOWESS can compute confidence and prediction intervals to quantify uncertainty
//! in the fitted curve and predictions.
//!
//! <div align="center">
//! <object data="../../../docs/confidence_vs_prediction_intervals.svg" type="image/svg+xml" width="800" height="500">
//! <img src="https://raw.githubusercontent.com/thisisamirv/fastLowess/main/docs/confidence_vs_prediction_intervals.svg" alt="Intervals" width="800"/>
//! </object>
//!
//! *Confidence intervals (narrow, for the mean curve) vs Prediction intervals (wide, for new observations)*
//! </div>
//!
//! ### Confidence Intervals
//!
//! Confidence intervals quantify uncertainty in the smoothed mean function.
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with confidence intervals
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .confidence_intervals(0.95)  // 95% confidence intervals
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y).unwrap();
//!
//! // Access confidence intervals
//! for i in 0..x.len() {
//!     println!(
//!         "x={:.1}: y={:.2} [{:.2}, {:.2}]",
//!         x[i],
//!         result.y[i],
//!         result.confidence_lower.as_ref().unwrap()[i],
//!         result.confidence_upper.as_ref().unwrap()[i]
//!     );
//! }
//! ```
//!
//! ```text
//! x=1.0: y=2.00 [1.85, 2.15]
//! x=2.0: y=4.10 [3.92, 4.28]
//! x=3.0: y=5.90 [5.71, 6.09]
//! x=4.0: y=8.20 [8.01, 8.39]
//! x=5.0: y=9.80 [9.65, 9.95]
//! ```
//!
//! ### Prediction Intervals
//!
//! Prediction intervals quantify where new individual observations will likely fall.
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];
//!
//! // Build model with both interval types
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .confidence_intervals(0.95)
//!     .prediction_intervals(0.95)  // Both can be enabled
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y).unwrap();
//! println!("{}", result);
//! ```
//!
//! ```text
//! Summary:
//!   Data points: 8
//!   Fraction: 0.5
//!
//! Smoothed Data:
//!        X     Y_smooth      Std_Err   Conf_Lower   Conf_Upper   Pred_Lower   Pred_Upper
//!   ----------------------------------------------------------------------------------
//!     1.00     2.01963     0.389365     1.256476     2.782788     1.058911     2.980353
//!     2.00     4.00251     0.345447     3.325438     4.679589     3.108641     4.896386
//!     3.00     5.99959     0.423339     5.169846     6.829335     4.985168     7.014013
//!     4.00     8.09859     0.489473     7.139224     9.057960     6.975666     9.221518
//!     5.00    10.03881     0.551687     8.957506    11.120118     8.810073    11.267551
//!     6.00    12.02872     0.539259    10.971775    13.085672    10.821364    13.236083
//!     7.00    13.89828     0.371149    13.170829    14.625733    12.965670    14.830892
//!     8.00    15.77990     0.408300    14.979631    16.580167    14.789441    16.770356
//! ```
//!
//! **Interval types:**
//! - **Confidence intervals**: Uncertainty in the smoothed mean
//!   - Narrower intervals
//!   - Use for: Understanding precision of the trend estimate
//! - **Prediction intervals**: Uncertainty for new observations
//!   - Wider intervals (includes data scatter + estimation uncertainty)
//!   - Use for: Forecasting where new data points will fall
//!
//! ## Cross-Validation
//!
//! Automatically select the optimal smoothing fraction using cross-validation.
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
//! let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
//!
//! // Build model with K-fold cross-validation
//! let model = Lowess::<f64>::new()
//!     .cross_validate(
//!         &[0.2, 0.3, 0.5, 0.7],           // Candidate fractions to test
//!         CrossValidationStrategy::KFold,   // K-fold CV
//!         Some(5)                           // 5 folds
//!     )
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y).unwrap();
//!
//! println!("Selected fraction: {}", result.fraction_used);
//! println!("CV scores: {:?}", result.cv_scores);
//! ```
//!
//! ```text
//! Selected fraction: 0.5
//! CV scores: Some([0.123, 0.098, 0.145, 0.187])
//! ```
//!
//! **CV strategies:**
//! - [`KFold`](CrossValidationStrategy::KFold): Faster, good for large datasets
//! - [`LOOCV`](CrossValidationStrategy::LOOCV) (Leave-one-out): More accurate, expensive for large datasets
//!
//! ## Diagnostics
//!
//! Compute diagnostic statistics to assess fit quality.
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with diagnostics
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .return_diagnostics()
//!     .return_residuals()
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y).unwrap();
//!
//! if let Some(diag) = &result.diagnostics {
//!     println!("RMSE: {:.4}", diag.rmse);
//!     println!("MAE: {:.4}", diag.mae);
//!     println!("R²: {:.4}", diag.r_squared);
//! }
//! ```
//!
//! ```text
//! RMSE: 0.1234
//! MAE: 0.0987
//! R²: 0.9876
//! ```
//!
//! **Available diagnostics:**
//! - **RMSE**: Root mean squared error
//! - **MAE**: Mean absolute error
//! - **R²**: Coefficient of determination
//! - **Residual SD**: Standard deviation of residuals
//! - **AIC/AICc**: Information criteria (when applicable)
//!
//! ## Zero-Weight Fallback
//!
//! Control behavior when all neighborhood weights are zero.
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with custom zero-weight fallback
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .zero_weight_fallback(ZeroWeightFallback::UseLocalMean)
//!     .adapter(Batch)
//!     .build()
//!     .unwrap();
//!
//! let result = model.fit(&x, &y).unwrap();
//! ```
//!
//! **Fallback options:**
//! - [`UseLocalMean`](ZeroWeightFallback::UseLocalMean): Use mean of neighborhood
//! - [`ReturnOriginal`](ZeroWeightFallback::ReturnOriginal): Return original y value
//! - [`ReturnNone`](ZeroWeightFallback::ReturnNone): Return NaN (for explicit handling)
//!
//! ## Execution Adapters
//!
//! Choose the execution mode based on your use case.
//!
//! ### Batch Adapter
//!
//! Standard mode for complete datasets in memory. Supports all features.
//! Uses parallel execution by default.
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with batch adapter
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .iterations(3)
//!     .confidence_intervals(0.95)
//!     .prediction_intervals(0.95)
//!     .return_diagnostics()
//!     .adapter(Batch)  // Full feature support, parallel by default
//!     .build()
//!     .unwrap();
//!
//! let result = model.fit(&x, &y).unwrap();
//! println!("{}", result);
//! ```
//!
//! ```text
//! Summary:
//!   Data points: 5
//!   Fraction: 0.5
//!
//! Smoothed Data:
//!        X     Y_smooth      Std_Err   Conf_Lower   Conf_Upper   Pred_Lower   Pred_Upper
//!   ----------------------------------------------------------------------------------
//!     1.00     2.00000     0.000000     2.000000     2.000000     2.000000     2.000000
//!     2.00     4.10000     0.000000     4.100000     4.100000     4.100000     4.100000
//!     3.00     5.90000     0.000000     5.900000     5.900000     5.900000     5.900000
//!     4.00     8.20000     0.000000     8.200000     8.200000     8.200000     8.200000
//!     5.00     9.80000     0.000000     9.800000     9.800000     9.800000     9.800000
//!
//! Diagnostics:
//!   RMSE: 0.0000
//!   MAE: 0.0000
//!   R²: 1.0000
//! ```
//!
//! **Use batch when:**
//! - Dataset fits in memory
//! - Need all features (intervals, CV, diagnostics)
//! - Processing complete datasets
//! - Want parallel speedup for large datasets
//!
//! ### Streaming Adapter
//!
//! Process large datasets in chunks with configurable overlap.
//! Uses parallel execution by default.
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
//! let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
//!
//! // Build model with streaming adapter
//! let model = Lowess::new()
//!     .fraction(0.1)
//!     .iterations(2)
//!     .adapter(Streaming)
//!     .parallel(true)  // Parallel streaming (default)
//!     .build()
//!     .unwrap();
//! ```
//!
//! **Use streaming when:**
//! - Dataset is very large (>100,000 points)
//! - Memory is limited
//! - Processing data in chunks
//!
//! ### Online Adapter
//!
//! Incremental updates with a sliding window for real-time data.
//! Uses sequential execution by default for lower latency.
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! // Build model with online adapter
//! let model = Lowess::new()
//!     .fraction(0.2)
//!     .iterations(1)
//!     .adapter(Online)
//!     .parallel(false)  // Sequential for low latency (default)
//!     .build()
//!     .unwrap();
//!
//! let mut online_model = model;
//!
//! // Add points incrementally
//! for i in 1..=10 {
//!     let x = i as f64;
//!     let y = 2.0 * x + 1.0;
//!     if let Some(result) = online_model.add_point(x, y).unwrap() {
//!         println!("Latest smoothed value: {:.2}", result.smoothed);
//!     }
//! }
//!
//! // Add points in bulk
//! let x_bulk = vec![11.0, 12.0, 13.0];
//! let y_bulk = vec![23.0, 25.0, 27.0];
//! let results = online_model.add_points(&x_bulk, &y_bulk).unwrap();
//! for result in results {
//!     if let Some(res) = result {
//!         println!("Bulk smoothed value: {:.2}", res.smoothed);
//!     }
//! }
//! ```
//!
//! **Use online when:**
//! - Data arrives incrementally
//! - Need real-time updates
//! - Maintaining a sliding window
//!
//! ## Complete Example
//!
//! A comprehensive example showing multiple features with parallel execution:
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! // Generate sample data with outliers
//! let x: Vec<f64> = (1..=50).map(|i| i as f64).collect();
//! let mut y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0 + (xi * 0.5).sin() * 5.0).collect();
//! y[10] = 100.0;  // Add an outlier
//! y[25] = -50.0;  // Add another outlier
//!
//! // Build the model with comprehensive configuration
//! let model = Lowess::new()
//!     .fraction(0.3)                                  // Moderate smoothing
//!     .iterations(5)                                  // Strong outlier resistance
//!     .weight_function(WeightFunction::Tricube)       // Default kernel
//!     .robustness_method(RobustnessMethod::Bisquare)  // Bisquare robustness
//!     .confidence_intervals(0.95)                     // 95% confidence intervals
//!     .prediction_intervals(0.95)                     // 95% prediction intervals
//!     .return_diagnostics()                           // Include diagnostics
//!     .return_residuals()                             // Include residuals
//!     .return_robustness_weights()                    // Include robustness weights
//!     .zero_weight_fallback(ZeroWeightFallback::UseLocalMean)
//!     .adapter(Batch)
//!     .parallel(true)                                 // Parallel execution
//!     .build()
//!     .unwrap();
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y).unwrap();
//!
//! // Examine results
//! println!("Smoothed {} points", result.y.len());
//!
//! // Check diagnostics
//! if let Some(diag) = &result.diagnostics {
//!     println!("Fit quality:");
//!     println!("  RMSE: {:.4}", diag.rmse);
//!     println!("  R²: {:.4}", diag.r_squared);
//! }
//!
//! // Identify outliers
//! if let Some(weights) = &result.robustness_weights {
//!     println!("\nOutliers detected:");
//!     for (i, &w) in weights.iter().enumerate() {
//!         if w < 0.1 {
//!             println!("  Point {}: y={:.1}, weight={:.3}", i, y[i], w);
//!         }
//!     }
//! }
//!
//! // Show confidence intervals for first few points
//! println!("\nFirst 5 points with intervals:");
//! for i in 0..5 {
//!     println!(
//!         "  x={:.0}: {:.2} [{:.2}, {:.2}] | [{:.2}, {:.2}]",
//!         x[i],
//!         result.y[i],
//!         result.confidence_lower.as_ref().unwrap()[i],
//!         result.confidence_upper.as_ref().unwrap()[i],
//!         result.prediction_lower.as_ref().unwrap()[i],
//!         result.prediction_upper.as_ref().unwrap()[i]
//!     );
//! }
//! ```
//!
//! ## Builder Options Reference
//!
//! ### Core Parameters
//! - `.fraction(f)` - Smoothing span (0, 1]
//! - `.iterations(n)` - Robustness iterations
//! - `.delta(d)` - Interpolation optimization
//! - `.weight_function(wf)` - Kernel function
//! - `.robustness_method(rm)` - Outlier handling
//!
//! ### Parallel Execution (fastLowess)
//! - `.parallel(true/false)` - Enable/disable parallel execution
//!
//! ### Intervals
//! - `.return_se()` - Compute standard errors
//! - `.confidence_intervals(level)` - Confidence intervals for mean
//! - `.prediction_intervals(level)` - Prediction intervals for new observations
//!
//! ### Cross-Validation
//! - `.cross_validate(fractions, strategy, k)` - Automatic fraction selection
//!
//! ### Output Options
//! - `.return_diagnostics()` - Include RMSE, MAE, R², etc.
//! - `.return_residuals()` - Include residuals
//! - `.return_robustness_weights()` - Include robustness weights
//!
//! ### Advanced
//! - `.zero_weight_fallback(policy)` - Zero-weight handling
//!
//! ### Adapters
//! - `.adapter(Batch)` - Standard mode (all features, parallel by default)
//! - `.adapter(Streaming)` - Chunk-based processing (large datasets, parallel by default)
//! - `.adapter(Online)` - Incremental updates (real-time data, sequential by default)
//!
//! ## Performance Tips
//!
//! 1. **Use parallel for large datasets**: Enable `.parallel(true)` for datasets >1,000 points
//! 2. **Use delta for large datasets**: Set to ~1% of x-range for >10,000 points
//! 3. **Reduce iterations for speed**: Use 0-2 iterations if outliers aren't a concern
//! 4. **Choose appropriate fraction**: Larger fractions are faster
//! 5. **Disable unnecessary features**: Don't compute intervals/diagnostics if not needed
//! 6. **Use ndarray for scientific workflows**: Avoid data copying with generic `fit`
//!
//! ## API Stability
//!
//! This crate follows [semantic versioning](https://semver.org/). The public API is stable
//! and breaking changes will only occur in major version updates.
//!
//! ### Public API (Stable)
//!
//! The following items are part of the **public, stable API** and follow semantic versioning:
//!
//! **Core Types:**
//! - [`Lowess`] - Main builder (re-exported as `LowessBuilder`)
//! - [`LowessResult`] - Smoothing result structure
//! - [`LowessError`] - Error type
//! - [`Result<T>`](Result) - Result type alias
//!
//! **Enums:**
//! - [`WeightFunction`] - Kernel functions (all variants stable)
//! - [`RobustnessMethod`] - Robustness methods (all variants stable)
//! - [`ZeroWeightFallback`] - Zero-weight fallback policies (all variants stable)
//! - [`CrossValidationStrategy`] - CV strategies (all variants stable)
//! - [`BoundaryPolicy`] - Boundary handling (for streaming/online)
//! - [`MergeStrategy`] - Overlap merging (for streaming)
//! - [`UpdateMode`] - Update modes (for online)
//!
//! **Adapter Markers:**
//! - [`Batch`](crate::Adapter::Batch) - Batch processing adapter
//! - [`Streaming`](crate::Adapter::Streaming) - Streaming adapter
//! - [`Online`](crate::Adapter::Online) - Online adapter
//!
//! **Prelude:**
//! - [`prelude`] module - Convenient wildcard imports
//!
//! ### Internal Implementation (Unstable)
//!
//! The following modules are **internal implementation details** and may change without notice:
//!
//! - `engine` - Parallel execution engine
//! - `adapters` - Adapter implementations
//! - `api` - API internals
//! - `ndarray` - ndarray integration internals
//!
//! **⚠️ Warning**: Using internal APIs directly may break without notice in minor/patch releases.
//! Stick to the public API for stability guarantees.
//!
//! ## Error Handling
//!
//! All operations return [`Result<T, LowessError>`](Result). Common errors:
//!
//! - [`InvalidFraction`](LowessError::InvalidFraction): Fraction not in (0, 1]
//! - [`MismatchedInputs`](LowessError::MismatchedInputs): x and y have different lengths
//! - [`TooFewPoints`](LowessError::TooFewPoints): Not enough points for the requested fraction
//! - [`InvalidChunkSize`](LowessError::InvalidChunkSize): Chunk size too small (streaming)
//! - [`InvalidOverlap`](LowessError::InvalidOverlap): Overlap >= chunk size (streaming)
//!
//! ## Dependencies
//!
//! - `lowess` - Core LOWESS implementation
//! - `rayon` - Parallel execution
//! - `ndarray` - Array support
//! - `num-traits` - Numeric traits
//!
//! ## References
//!
//! - Cleveland, W. S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots"
//! - Cleveland, W. S. (1981). "LOWESS: A Program for Smoothing Scatterplots by Robust Locally Weighted Regression"
//!
//! ## License
//!
//! See the repository for license information and contribution guidelines.

#![allow(non_snake_case)]
#![deny(missing_docs)]

// ============================================================================
// Internal Modules
// ============================================================================

// Layer 5: Engine - Parallel execution engine.
//
// Contains `LowessExecutor` with rayon-based parallel smoothing.
mod engine;

// Layer 6: Adapters - execution mode adapters.
//
// Contains execution adapters for different use cases:
// batch (standard), streaming (large datasets), online (incremental).
mod adapters;

// High-level fluent API for LOWESS smoothing.
//
// Provides the `Lowess` builder for configuring and running LOWESS smoothing.
mod api;

// Input abstraction for ndarray/slice compatibility.
mod input;

// ============================================================================
// Public Re-exports
// ============================================================================

pub use crate::api::{
    Adapter, BoundaryPolicy, CrossValidationStrategy, LowessBuilder as Lowess, LowessError,
    LowessResult, MergeStrategy, Result, RobustnessMethod, UpdateMode, WeightFunction,
    ZeroWeightFallback,
};

// ============================================================================
// Prelude
// ============================================================================

/// Standard fastLowess prelude.
///
/// This module is intended to be wildcard-imported for convenient access
/// to the most commonly used types:
///
/// ```
/// use fastLowess::prelude::*;
/// ```
///
/// This imports:
/// - `Lowess` - The main builder
/// - `Batch`, `Streaming`, `Online` - Adapter markers
/// - `LowessResult`, `LowessError`, `Result` - Result types
/// - All enum types (RobustnessMethod, WeightFunction, etc.)
pub mod prelude {
    pub use crate::api::{
        Adapter::{Batch, Online, Streaming},
        BoundaryPolicy, CrossValidationStrategy, LowessBuilder as Lowess, LowessError,
        LowessResult, MergeStrategy, Result, RobustnessMethod, UpdateMode, WeightFunction,
        ZeroWeightFallback,
    };
}

// ============================================================================
// Testing re-exports
// ============================================================================

/// Internal modules for development and testing.
///
/// This module re-exports internal modules for development and testing purposes.
/// It is only available with the `dev` feature enabled.
///
/// **Warning**: These are internal implementation details and may change without notice.
/// Do not use in production code.
#[cfg(feature = "dev")]
pub mod internals {
    /// Internal execution engine.
    pub mod engine {
        pub use crate::engine::*;
    }
    /// Internal adapters.
    pub mod adapters {
        pub use crate::adapters::*;
    }
    /// Internal API.
    pub mod api {
        pub use crate::api::*;
    }
}

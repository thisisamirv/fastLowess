//! Batch adapter for standard LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the batch (standard) execution adapter for LOWESS
//! smoothing. It handles complete datasets in memory with optional parallel
//! processing, making it suitable for small to medium-sized datasets where
//! all data is available upfront. The batch adapter is the simplest and most
//! straightforward way to use LOWESS.
//!
//! ## Design notes
//!
//! * Processes entire dataset in a single pass.
//! * Automatically sorts data by x-values and unsorts results.
//! * Delegates computation to the `lowess` crate's execution engine.
//! * Supports all LOWESS features: robustness, CV, intervals, diagnostics.
//! * Adds parallel execution via `rayon` (fastLowess extension).
//! * Generic over `Float` types to support f32 and f64.
//!
//! ## Key concepts
//!
//! ### Batch Processing
//! The batch adapter:
//! 1. Validates input data
//! 2. Sorts data by x-values (required for LOWESS)
//! 3. Executes LOWESS smoothing via the engine
//! 4. Computes optional outputs (diagnostics, intervals, residuals)
//! 5. Unsorts results to match original input order
//! 6. Packages everything into a `LowessResult`
//!
//! ### Builder Pattern
//! Configuration is done through `ExtendedBatchLowessBuilder`:
//! * Fluent API for setting parameters
//! * Sensible defaults for all parameters
//! * Validation deferred until `fit()` is called
//!
//! ### Automatic Sorting
//! LOWESS requires sorted x-values. The batch adapter:
//! * Automatically sorts input data by x
//! * Tracks original indices
//! * Unsorts all outputs to match original order
//!
//! ## Supported features
//!
//! * **Robustness iterations**: Downweight outliers iteratively
//! * **Cross-validation**: Automatic fraction selection
//! * **Auto-convergence**: Adaptive iteration count
//! * **Confidence intervals**: Uncertainty in fitted curve
//! * **Prediction intervals**: Uncertainty for new observations
//! * **Diagnostics**: RMSE, MAE, R², AIC, AICc
//! * **Residuals**: Differences between original and smoothed values
//! * **Robustness weights**: Final weights from iterative refinement
//! * **Parallel execution**: Rayon-based parallelism (fastLowess extension)
//!
//! ## Invariants
//!
//! * Input arrays x and y must have the same length.
//! * All values must be finite (no NaN or infinity).
//! * At least 2 data points are required.
//! * Fraction must be in (0, 1].
//! * Output order matches input order (automatic unsorting).
//!
//! ## Non-goals
//!
//! * This adapter does not handle streaming data (use streaming adapter).
//! * This adapter does not handle incremental updates (use online adapter).
//! * This adapter does not handle missing values (NaN).

use crate::engine::executor::smooth_pass_parallel;
use crate::input::LowessInput;

use lowess::internals::adapters::batch::BatchLowessBuilder;
use lowess::internals::engine::output::LowessResult;
use lowess::internals::primitives::errors::LowessError;

use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// ============================================================================
// Extended Batch LOWESS Builder
// ============================================================================

/// Builder for batch LOWESS processor with parallel support.
///
/// Provides a fluent API for configuring batch LOWESS smoothing with
/// optional parallel execution. Wraps the core `BatchLowessBuilder` from
/// the `lowess` crate and adds fastLowess-specific extensions.
///
/// # Fields
///
/// * `base` - Core builder from the lowess crate (fraction, iterations, etc.)
/// * `parallel` - Whether to use parallel execution (fastLowess extension)
///
/// # Usage
///
/// Access base builder fields and methods via `.base` or use the provided
/// builder methods which forward to the base builder.
#[derive(Debug, Clone)]
pub struct ExtendedBatchLowessBuilder<T: Float> {
    /// Base builder from the lowess crate
    pub base: BatchLowessBuilder<T>,

    /// Whether to use parallel execution (fastLowess extension)
    pub parallel: bool,
}

impl<T: Float> Default for ExtendedBatchLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> ExtendedBatchLowessBuilder<T> {
    /// Create a new batch LOWESS builder with default parameters.
    ///
    /// # Defaults
    ///
    /// * All base parameters from lowess BatchLowessBuilder
    /// * parallel: true (fastLowess extension)
    fn new() -> Self {
        Self {
            base: BatchLowessBuilder::default(),
            parallel: true,
        }
    }

    /// Set parallel execution mode.
    ///
    /// Parallel execution can significantly speed up processing for large
    /// datasets by distributing the local regression fits across CPU cores.
    ///
    /// # Parameters
    ///
    /// * `parallel` - Whether to enable parallel execution (default: true)
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the batch processor.
    ///
    /// Validates all configuration and creates a ready-to-use processor.
    ///
    /// # Returns
    ///
    /// A configured `ExtendedBatchLowess` processor ready to fit data.
    ///
    /// # Errors
    ///
    /// Returns `LowessError` if configuration is invalid.
    pub fn build(self) -> Result<ExtendedBatchLowess<T>, LowessError> {
        // Check for deferred errors from adapter conversion
        if let Some(ref err) = self.base.deferred_error {
            return Err(err.clone());
        }

        // Validate by attempting to build the base processor
        // This reuses the validation logic centralized in the lowess crate
        let _ = self.base.clone().build()?;

        Ok(ExtendedBatchLowess { config: self })
    }
}

// ============================================================================
// Extended Batch LOWESS Processor
// ============================================================================

/// Batch LOWESS processor with parallel support.
///
/// Performs standard LOWESS smoothing on complete datasets by delegating
/// to the base `lowess` implementation with optional parallel execution.
pub struct ExtendedBatchLowess<T: Float> {
    config: ExtendedBatchLowessBuilder<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ExtendedBatchLowess<T> {
    /// Perform LOWESS smoothing on the provided data.
    ///
    /// This method validates inputs, executes LOWESS smoothing (with parallel
    /// execution if enabled), and returns comprehensive results including
    /// smoothed values and optional outputs (diagnostics, intervals, etc.).
    ///
    /// # Parameters
    ///
    /// * `x` - Independent variable values
    /// * `y` - Dependent variable values (must have same length as x)
    ///
    /// # Returns
    ///
    /// `LowessResult` containing:
    /// * Smoothed y-values
    /// * Standard errors (if intervals requested)
    /// * Confidence/prediction intervals (if requested)
    /// * Residuals (if requested)
    /// * Robustness weights (if requested)
    /// * Diagnostics (if requested)
    ///
    /// # Errors
    ///
    /// Returns `LowessError` if:
    /// * Input arrays have different lengths
    /// * Inputs contain NaN or infinity
    /// * Fewer than 2 data points
    /// * Invalid fraction value
    ///
    /// # Algorithm
    ///
    /// 1. Validate inputs
    /// 2. Sort data by x-values
    /// 3. Execute LOWESS smoothing (parallel if enabled)
    /// 4. Compute residuals
    /// 5. Compute diagnostics (if requested)
    /// 6. Compute intervals (if requested)
    /// 7. Unsort all results to match original order
    /// 8. Package into `LowessResult`
    pub fn fit<I1, I2>(self, x: &I1, y: &I2) -> Result<LowessResult<T>, LowessError>
    where
        I1: LowessInput<T> + ?Sized,
        I2: LowessInput<T> + ?Sized,
    {
        let x_slice = x.as_lowess_slice()?;
        let y_slice = y.as_lowess_slice()?;

        // Configure the base builder with parallel callback if enabled
        let mut builder = self.config.base;

        if self.config.parallel {
            builder.custom_smooth_pass = Some(smooth_pass_parallel);
        } else {
            builder.custom_smooth_pass = None;
        }

        // Delegate execution to the base implementation
        let processor = builder.build()?;
        processor.fit(x_slice, y_slice)
    }
}

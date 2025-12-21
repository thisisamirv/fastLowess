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
//! * Delegates computation to the execution engine.
//! * Supports all LOWESS features: robustness, CV, intervals, diagnostics.
//! * Adds parallel execution via `rayon` (fastLowess extension).
//! * Uses builder pattern for configuration.
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
//! * **Diagnostics**: RMSE, MAE, R^2, AIC, AICc
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
//!
//! ## Visibility
//!
//! The batch adapter is part of the public API through the high-level
//! `Lowess` builder. Direct usage of `BatchLowess` is possible but not
//! the primary interface.

use crate::engine::executor::smooth_pass_parallel;
use crate::input::LowessInput;

use lowess::internals::adapters::batch::BatchLowessBuilder;
use lowess::internals::algorithms::regression::ZeroWeightFallback;
use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::engine::output::LowessResult;
use lowess::internals::evaluation::cv::CVMethod;
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::errors::LowessError;
use lowess::internals::primitives::partition::BoundaryPolicy;

use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// ============================================================================
// Extended Batch LOWESS Builder
// ============================================================================

/// Builder for batch LOWESS processor with parallel support.
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
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    // ========================================================================
    // Shared Setters
    // ========================================================================

    /// Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.base = self.base.fraction(fraction);
        self
    }

    /// Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base = self.base.iterations(iterations);
        self
    }

    /// Set the delta parameter for interpolation optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.base = self.base.delta(delta);
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.base = self.base.weight_function(wf);
        self
    }

    /// Set the robustness method for outlier handling.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.base = self.base.robustness_method(method);
        self
    }

    /// Set the zero-weight fallback policy.
    pub fn zero_weight_fallback(mut self, fallback: ZeroWeightFallback) -> Self {
        self.base = self.base.zero_weight_fallback(fallback);
        self
    }

    /// Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.base = self.base.boundary_policy(policy);
        self
    }

    /// Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base = self.base.auto_converge(tolerance);
        self
    }

    /// Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base = self.base.compute_residuals(enabled);
        self
    }

    /// Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base = self.base.return_robustness_weights(enabled);
        self
    }

    // ========================================================================
    // Batch-Specific Setters
    // ========================================================================

    /// Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base = self.base.return_diagnostics(enabled);
        self
    }

    /// Enable confidence intervals at the specified level.
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.base = self.base.confidence_intervals(level);
        self
    }

    /// Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.base = self.base.prediction_intervals(level);
        self
    }

    /// Enable cross-validation with the specified fractions.
    pub fn cross_validate(mut self, fractions: Vec<T>) -> Self {
        self.base = self.base.cross_validate(fractions);
        self
    }

    /// Set the cross-validation method.
    pub fn cv_method(mut self, method: CVMethod) -> Self {
        self.base = self.base.cv_method(method);
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the batch processor.
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
pub struct ExtendedBatchLowess<T: Float> {
    config: ExtendedBatchLowessBuilder<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ExtendedBatchLowess<T> {
    /// Perform LOWESS smoothing on the provided data.
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

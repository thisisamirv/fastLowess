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
//! * Uses builder pattern for configuration.
//! * Generic over `Float` types to support f32 and f64.
//! * **fastLowess addition**: Supports optional parallel execution via rayon.
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
//! Configuration is done through `BatchLowessBuilder`:
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
//! * **Parallel execution**: Optional parallel smoothing via rayon (fastLowess)
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

use crate::engine::executor::{LowessConfig, LowessExecutor};
use crate::input::LowessInput;

use lowess::testing::algorithms::interpolation::calculate_delta;
use lowess::testing::algorithms::regression::ZeroWeightFallback;
use lowess::testing::algorithms::robustness::RobustnessMethod;
use lowess::testing::engine::output::LowessResult;
use lowess::testing::engine::validator::Validator;
use lowess::testing::evaluation::cv::CVMethod;
use lowess::testing::evaluation::diagnostics::Diagnostics;
use lowess::testing::evaluation::intervals::IntervalMethod;
use lowess::testing::math::kernel::WeightFunction;
use lowess::testing::primitives::errors::LowessError;
use lowess::testing::primitives::sorting::{sort_by_x, unsort};

use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;
use std::vec::Vec;

// ============================================================================
// Batch LOWESS Builder
// ============================================================================

/// Builder for batch LOWESS processor.
///
/// Provides a fluent API for configuring batch LOWESS smoothing. All
/// parameters have sensible defaults and can be set independently.
///
/// # Fields
///
/// ## Core Parameters
/// * `fraction` - Smoothing fraction (0, 1]. If None, defaults to 0.67 or uses CV
/// * `iterations` - Number of robustness iterations (default: 3)
/// * `delta` - Optimization delta for point skipping (default: auto-computed)
///
/// ## Algorithm Configuration
/// * `weight_function` - Kernel weight function (default: Tricube)
/// * `robustness_method` - Robustness weighting method (default: Bisquare)
/// * `zero_weight_fallback` - Zero weight fallback policy (default: UseLocalMean)
///
/// ## Cross-Validation
/// * `cv_fractions` - Fractions to test for CV (default: None)
/// * `cv_method` - CV method (default: None, uses KFold(5) if cv_fractions provided)
///
/// ## Auto-Convergence
/// * `auto_convergence` - Convergence tolerance (default: None)
///
/// ## Uncertainty Quantification
/// * `interval_type` - Confidence/prediction interval configuration (default: None)
///
/// ## Output Options
/// * `compute_diagnostics` - Whether to compute diagnostic statistics (default: false)
/// * `compute_residuals` - Whether to return residuals (default: false)
/// * `compute_robustness_weights` - Whether to return robustness weights (default: false)
///
/// ## Execution Options (fastLowess)
/// * `parallel` - Whether to use parallel execution (default: true)
#[derive(Debug, Clone)]
pub struct BatchLowessBuilder<T: Float> {
    /// Smoothing fraction (span)
    pub fraction: T,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Optimization delta
    pub delta: Option<T>,

    /// Kernel weight function
    pub weight_function: WeightFunction,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Confidence/Prediction interval configuration
    pub interval_type: Option<IntervalMethod<T>>,

    /// Fractions for cross-validation
    pub cv_fractions: Option<Vec<T>>,

    /// Cross-validation method
    pub cv_method: Option<CVMethod>,

    /// Deferred error from adapter conversion
    pub deferred_error: Option<LowessError>,

    /// Tolerance for auto-convergence
    pub auto_convergence: Option<T>,

    /// Whether to compute diagnostic statistics
    pub compute_diagnostics: bool,

    /// Whether to return residuals
    pub compute_residuals: bool,

    /// Whether to return robustness weights
    pub compute_robustness_weights: bool,

    /// Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Whether to use parallel execution (fastLowess addition)
    pub parallel: bool,
}

impl<T: Float> Default for BatchLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> BatchLowessBuilder<T> {
    /// Create a new batch LOWESS builder with default parameters.
    ///
    /// # Defaults
    ///
    /// * fraction: 0.67
    /// * iterations: 3
    /// * delta: None (auto-computed as 1% of x-range)
    /// * weight_function: Tricube
    /// * robustness_method: Bisquare
    /// * interval_type: None
    /// * cv_fractions: None
    /// * cv_method: None
    /// * auto_convergence: None
    /// * compute_diagnostics: false
    /// * compute_residuals: false
    /// * compute_robustness_weights: false
    /// * zero_weight_fallback: UseLocalMean
    /// * parallel: true (fastLowess)
    fn new() -> Self {
        Self {
            fraction: T::from(0.67).unwrap(),
            iterations: 3,
            delta: None,
            weight_function: WeightFunction::Tricube,
            robustness_method: RobustnessMethod::Bisquare,
            interval_type: None,
            cv_fractions: None,
            cv_method: None,
            deferred_error: None,
            auto_convergence: None,
            compute_diagnostics: false,
            compute_residuals: false,
            compute_robustness_weights: false,
            zero_weight_fallback: ZeroWeightFallback::UseLocalMean,
            parallel: true,
        }
    }

    /// Set parallel execution mode.
    ///
    /// # Note
    ///
    /// Parallel execution can significantly speed up processing for large datasets.
    /// Default is true for batch adapter.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the batch processor.
    ///
    /// # Returns
    ///
    /// A configured `BatchLowess` processor ready to fit data.
    pub fn build(self) -> Result<BatchLowess<T>, LowessError> {
        if let Some(err) = self.deferred_error {
            return Err(err);
        }

        // Validate configuration
        Validator::validate_fraction(self.fraction)?;

        if let Some(delta) = self.delta {
            Validator::validate_delta(delta)?;
        }

        if let Some(ref method) = self.interval_type {
            Validator::validate_interval_level(method.level)?;
        }

        if let Some(ref fracs) = self.cv_fractions {
            Validator::validate_cv_fractions(fracs)?;
        }

        if let Some(tol) = self.auto_convergence {
            Validator::validate_tolerance(tol)?;
        }

        Ok(BatchLowess { config: self })
    }
}

// ============================================================================
// Batch LOWESS Processor
// ============================================================================

/// Batch LOWESS processor.
///
/// Performs standard LOWESS smoothing on complete datasets. Handles sorting,
/// execution, and result packaging.
///
/// # Usage
///
/// Create via `BatchLowess::builder()`, configure, build, and call `fit()`.
pub struct BatchLowess<T: Float> {
    config: BatchLowessBuilder<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> BatchLowess<T> {
    /// Perform LOWESS smoothing on the provided data.
    ///
    /// # Parameters
    ///
    /// * `x` - Independent variable values
    /// * `y` - Dependent variable values
    ///
    /// # Returns
    ///
    /// `LowessResult` containing smoothed values and optional outputs.
    ///
    /// # Errors
    ///
    /// Returns `LowessError` if:
    /// * Input validation fails
    /// * Arrays have mismatched lengths
    /// * Values are not finite
    /// * Fraction is invalid
    /// * Interval computation fails
    ///
    /// # Algorithm
    ///
    /// 1. Validate inputs
    /// 2. Sort data by x-values
    /// 3. Execute LOWESS smoothing
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

        Validator::validate_inputs(x_slice, y_slice)?;

        // Sort data by x using sorting module
        let sorted = sort_by_x(x_slice, y_slice);
        let delta = calculate_delta(self.config.delta, &sorted.x)?;

        let zw_flag: u8 = self.config.zero_weight_fallback.to_u8();

        // Configure batch execution
        let config = LowessConfig {
            fraction: Some(self.config.fraction),
            iterations: self.config.iterations,
            delta,
            weight_function: self.config.weight_function,
            zero_weight_fallback: zw_flag,
            robustness_method: self.config.robustness_method,
            cv_fractions: self.config.cv_fractions,
            cv_method: self.config.cv_method,
            auto_convergence: self.config.auto_convergence,
            return_variance: self.config.interval_type,
            parallel: self.config.parallel,
        };

        // Execute unified LOWESS
        let result = LowessExecutor::run_with_config(&sorted.x, &sorted.y, config);

        let y_smooth = result.smoothed;
        let std_errors = result.std_errors;
        let iterations_used = result.iterations;
        let fraction_used = result.used_fraction;
        let cv_scores = result.cv_scores;

        // Calculate residuals
        let n = x_slice.len();
        let mut residuals = Vec::with_capacity(n);
        for (i, &smoothed_val) in y_smooth.iter().enumerate().take(n) {
            residuals.push(sorted.y[i] - smoothed_val);
        }

        // Placeholder robustness weights (actual weights not currently returned by executor)
        let rob_weights = if self.config.compute_robustness_weights {
            vec![T::one(); n]
        } else {
            Vec::new()
        };

        // Compute diagnostic statistics if requested
        let diagnostics = if self.config.compute_diagnostics {
            Some(Diagnostics::compute(
                &sorted.y,
                &y_smooth,
                &residuals,
                std_errors.as_deref(),
            ))
        } else {
            None
        };

        // Compute intervals
        let (conf_lower, conf_upper, pred_lower, pred_upper) =
            if let Some(method) = &self.config.interval_type {
                if let Some(se) = &std_errors {
                    method.compute_intervals(&y_smooth, se, &residuals)?
                } else {
                    (None, None, None, None)
                }
            } else {
                (None, None, None, None)
            };

        // Unsort results using sorting module
        let indices = &sorted.indices;
        let y_smooth_out = unsort(&y_smooth, indices);
        let std_errors_out = std_errors.as_ref().map(|se| unsort(se, indices));
        let residuals_out = if self.config.compute_residuals {
            Some(unsort(&residuals, indices))
        } else {
            None
        };
        let rob_weights_out = if self.config.compute_robustness_weights {
            Some(unsort(&rob_weights, indices))
        } else {
            None
        };
        let cl_out = conf_lower.as_ref().map(|v| unsort(v, indices));
        let cu_out = conf_upper.as_ref().map(|v| unsort(v, indices));
        let pl_out = pred_lower.as_ref().map(|v| unsort(v, indices));
        let pu_out = pred_upper.as_ref().map(|v| unsort(v, indices));

        Ok(LowessResult {
            x: x_slice.to_vec(),
            y: y_smooth_out,
            standard_errors: std_errors_out,
            confidence_lower: cl_out,
            confidence_upper: cu_out,
            prediction_lower: pl_out,
            prediction_upper: pu_out,
            residuals: residuals_out,
            robustness_weights: rob_weights_out,
            fraction_used,
            iterations_used,
            cv_scores,
            diagnostics,
        })
    }
}

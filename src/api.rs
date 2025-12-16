//! High-level API for LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the primary user-facing API for LOWESS (Locally
//! Weighted Scatterplot Smoothing). It offers a fluent builder pattern for
//! configuring smoothing parameters and selecting execution adapters (batch,
//! streaming, or online) based on use case requirements.
//!
//! ## Design notes
//!
//! * Uses builder pattern for ergonomic configuration.
//! * Supports multiple execution adapters via trait-based dispatch.
//! * Provides sensible defaults for all parameters.
//! * Validates configuration at build time, not construction time.
//! * Generic over `Float` types to support f32 and f64.
//! * Re-exports commonly used types for convenience.
//! * **fastLowess addition**: Supports optional parallel execution via rayon.
//!
//! ## Key concepts
//!
//! ### Builder Pattern
//! The `LowessBuilder` provides a fluent API:
//! ```text
//! LowessBuilder::new()
//!     .fraction(0.5)
//!     .iterations(2)
//!     .confidence_intervals(0.95)
//!     .adapter(Batch)
//!     .fit(x, y)
//! ```
//!
//! ### Adapter Selection
//! Three execution adapters are available:
//! * **Batch**: Complete datasets in memory (most features, parallel by default)
//! * **Streaming**: Large datasets processed in chunks (parallel by default)
//! * **Online**: Incremental updates with sliding window (sequential by default)
//!
//! ### Configuration Categories
//! Parameters are organized into logical groups:
//! * Core: fraction, iterations, delta, weight function, robustness
//! * Intervals: confidence/prediction intervals, standard errors
//! * Cross-validation: automatic fraction selection
//! * Auto-convergence: adaptive iteration count
//! * Output options: diagnostics, residuals, weights
//!
//! ## Invariants
//!
//! * Builder can be cloned and reused.
//! * Configuration is immutable after adapter selection.
//! * All parameters have valid defaults.
//! * Validation occurs at adapter build time.
//!
//! ## Non-goals
//!
//! * This module does not perform smoothing directly.
//! * This module does not validate input data.
//! * This module does not handle data loading or preprocessing.
//! * This module does not provide plotting or visualization.
//!
//! ## Visibility
//!
//! This module is the primary public API for the crate. Types and traits
//! defined here are stable and follow semantic versioning.

use std::result;
use std::vec::Vec;

use num_traits::Float;

// Internal adapters
use crate::adapters::batch::BatchLowessBuilder;
use crate::adapters::online::OnlineLowessBuilder;
use crate::adapters::streaming::StreamingLowessBuilder;

// Publicly re-exported types
pub use lowess::testing::algorithms::regression::ZeroWeightFallback;
pub use lowess::testing::algorithms::robustness::RobustnessMethod;
pub use lowess::testing::engine::output::LowessResult;
pub use lowess::testing::evaluation::cv::CVMethod;
pub use lowess::testing::evaluation::intervals::IntervalMethod;
pub use lowess::testing::math::kernel::WeightFunction;
pub use lowess::testing::primitives::errors::LowessError;
pub use lowess::testing::primitives::partition::{BoundaryPolicy, MergeStrategy, UpdateMode};

// ============================================================================
// Type Aliases
// ============================================================================

/// Result type alias for LOWESS operations.
///
/// Convenience alias for `Result<T, LowessError>`.
pub type Result<T> = result::Result<T, LowessError>;

// ============================================================================
// Adapter Module
// ============================================================================

/// Adapter selection namespace.
///
/// Contains marker types for selecting execution adapters.
/// Use `Adapter::Batch`, `Adapter::Streaming`, or `Adapter::Online`.
#[allow(non_snake_case)]
pub mod Adapter {
    pub use super::{Batch, Online, Streaming};
}

// ============================================================================
// Builder
// ============================================================================

/// LOWESS builder with comprehensive configuration options.
///
/// Provides a fluent API for configuring and running LOWESS smoothing.
/// Use `LowessBuilder::new()` to create a builder with sensible defaults,
/// then chain configuration methods and select an adapter to execute.
///
/// # Fields
///
/// ## Core Parameters
/// * `fraction` - Smoothing fraction (0, 1] (default: None, uses 0.67 or CV)
/// * `iterations` - Number of robustness iterations (default: 3)
/// * `delta` - Delta for interpolation optimization (default: None, auto 1% of range)
/// * `weight_function` - Kernel weight function (default: Tricube)
/// * `robustness_method` - Robustness weighting method (default: Bisquare)
/// * `zero_weight_fallback` - Zero-weight fallback policy (default: UseLocalMean)
///
/// ## Interval Computation
/// * `interval_type` - Confidence/prediction interval configuration (default: None)
///
/// ## Cross-Validation
/// * `cv_fractions` - Fractions to test for CV (default: None)
/// * `cv_method` - CV method (default: None)
///
/// ## Auto-Convergence
/// * `auto_convergence` - Convergence tolerance (default: None)
/// * `max_iterations` - Maximum iterations for auto-convergence (default: 20)
///
/// ## Output Options
/// * `compute_diagnostics` - Whether to compute diagnostic statistics (default: false)
/// * `compute_residuals` - Whether to return residuals (default: false)
/// * `compute_robustness_weights` - Whether to return robustness weights (default: false)
#[derive(Debug, Clone)]
pub struct LowessBuilder<T> {
    /// Smoothing fraction (span)
    pub fraction: Option<T>,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Delta for interpolation optimization
    pub delta: Option<T>,

    /// Weight/kernel function
    pub weight_function: WeightFunction,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Type of interval to compute (includes level)
    pub interval_type: Option<IntervalMethod<T>>,

    /// Cross-validation fractions
    pub cv_fractions: Option<Vec<T>>,

    /// Cross-validation method
    pub cv_method: Option<CVMethod>,

    /// Auto-convergence tolerance
    pub auto_convergence: Option<T>,

    /// Max iterations for auto-convergence
    pub max_iterations: usize,

    /// Whether to compute diagnostic statistics
    pub compute_diagnostics: bool,

    /// Whether to return residuals
    pub compute_residuals: bool,

    /// Whether to return robustness weights
    pub compute_robustness_weights: bool,

    /// Zero-weight fallback policy
    pub zero_weight_fallback: ZeroWeightFallback,
}

impl<T: Float> Default for LowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> LowessBuilder<T> {
    // ========================================================================
    // Adapter Selection
    // ========================================================================

    /// Select the execution adapter (Batch, Streaming, or Online).
    ///
    /// This method converts the builder into an adapter-specific builder
    /// that can execute smoothing operations.
    ///
    /// # Type Parameters
    ///
    /// * `A` - The adapter type (Batch, Streaming, or Online)
    ///
    /// # Returns
    ///
    /// An adapter-specific builder ready to execute smoothing.
    pub fn adapter<A>(self, _adapter: A) -> A::Output
    where
        A: LowessAdapter<T>,
    {
        A::convert(self)
    }

    // ========================================================================
    // Constructor
    // ========================================================================

    /// Create a new builder with default settings.
    ///
    /// # Defaults
    ///
    /// * fraction: None (will default to 0.67 or use CV)
    /// * iterations: 3
    /// * delta: None (will default to 1% of x-range)
    /// * weight_function: Tricube
    /// * robustness_method: Bisquare
    /// * interval_type: None
    /// * cv_fractions: None
    /// * cv_method: None
    /// * auto_convergence: None
    /// * max_iterations: 20
    /// * compute_diagnostics: false
    /// * compute_residuals: false
    /// * compute_robustness_weights: false
    /// * zero_weight_fallback: UseLocalMean
    pub fn new() -> Self {
        Self {
            fraction: None,
            iterations: 3,
            delta: None,
            weight_function: WeightFunction::Tricube,
            robustness_method: RobustnessMethod::Bisquare,
            interval_type: None,
            cv_fractions: None,
            cv_method: None,
            auto_convergence: None,
            max_iterations: 20,
            compute_diagnostics: false,
            compute_residuals: false,
            compute_robustness_weights: false,
            zero_weight_fallback: ZeroWeightFallback::UseLocalMean,
        }
    }

    /// Configure the zero-weight fallback behavior.
    ///
    /// Determines what to do when all neighborhood weights are zero.
    pub fn zero_weight_fallback(mut self, policy: ZeroWeightFallback) -> Self {
        self.zero_weight_fallback = policy;
        self
    }

    // ========================================================================
    // Core Parameters
    // ========================================================================

    /// Set smoothing fraction (span).
    ///
    /// # Range
    ///
    /// * 0 < fraction ≤ 1
    pub fn fraction(mut self, fraction: T) -> Self {
        self.fraction = Some(fraction);
        self
    }

    /// Set number of robustness iterations.
    ///
    /// # Typical values
    ///
    /// * 0: No robustness (fastest)
    /// * 1-3: Light to moderate robustness (recommended)
    /// * 4+: Heavy robustness (slower)
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set Delta (distance within which to skip re-regression).
    ///
    /// # Default
    ///
    /// If None, defaults to 1% of x-range (0.01 * (max(x) - min(x)))
    pub fn delta(mut self, delta: T) -> Self {
        self.delta = Some(delta);
        self
    }

    /// Set weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.weight_function = wf;
        self
    }

    /// Set robustness method.
    pub fn robustness_method(mut self, rm: RobustnessMethod) -> Self {
        self.robustness_method = rm;
        self
    }

    // ========================================================================
    // Interval Computation
    // ========================================================================

    /// Compute standard errors.
    ///
    /// Enables computation of standard errors for the fitted values.
    /// Required for confidence and prediction intervals.
    pub fn return_se(mut self) -> Self {
        if self.interval_type.is_none() {
            self.interval_type = Some(IntervalMethod::se());
        }
        self
    }

    /// Compute confidence intervals for the mean response.
    ///
    /// # Parameters
    ///
    /// * `level` - Confidence level (e.g., 0.95 for 95% CI)
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.interval_type = Some(match self.interval_type {
            Some(existing) if existing.prediction => {
                // Already have prediction intervals, enable both
                IntervalMethod {
                    level,
                    confidence: true,
                    prediction: true,
                    se: true,
                }
            }
            _ => IntervalMethod::confidence(level),
        });
        self
    }

    /// Compute prediction intervals for new observations.
    ///
    /// # Parameters
    ///
    /// * `level` - Confidence level (e.g., 0.95 for 95% PI)
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.interval_type = Some(match self.interval_type {
            Some(existing) if existing.confidence => {
                // Already have confidence intervals, enable both
                IntervalMethod {
                    level,
                    confidence: true,
                    prediction: true,
                    se: true,
                }
            }
            _ => IntervalMethod::prediction(level),
        });
        self
    }

    // ========================================================================
    // Cross-Validation
    // ========================================================================

    /// Enable cross-validation to select optimal fraction.
    ///
    /// # Parameters
    ///
    /// * `fractions` - Candidate fractions to test
    /// * `method` - CV strategy (KFold or LeaveOneOut)
    /// * `k` - Number of folds for KFold (ignored for LeaveOneOut)
    pub fn cross_validate(
        mut self,
        fractions: &[T],
        method: CrossValidationStrategy,
        k: Option<usize>,
    ) -> Self {
        self.cv_fractions = Some(fractions.to_vec());
        self.cv_method = Some(match method {
            CrossValidationStrategy::KFold => CVMethod::KFold(k.unwrap_or(5)),
            CrossValidationStrategy::LeaveOneOut | CrossValidationStrategy::LOOCV => {
                CVMethod::LOOCV
            }
        });
        self
    }

    // ========================================================================
    // Auto-Convergence
    // ========================================================================

    /// Enable automatic convergence detection.
    ///
    /// Stops iterations early when smoothed values change by less than
    /// the specified tolerance.
    ///
    /// # Parameters
    ///
    /// * `tolerance` - Maximum relative change to consider converged
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.auto_convergence = Some(tolerance);
        self
    }

    /// Set maximum iterations for auto-convergence.
    ///
    /// Only used when `auto_converge` is enabled. Prevents infinite loops
    /// if convergence is not reached.
    ///
    /// # Clamping
    ///
    /// * Values of 0 are clamped to 1
    /// * Values > 1000 are clamped to 1000
    ///
    /// # Default
    ///
    /// 20 iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        // Clamp to [1, 1000]
        self.max_iterations = max_iter.clamp(1, 1000);
        self
    }

    // ========================================================================
    // Diagnostics and Output Options
    // ========================================================================

    /// Include diagnostic statistics in the result.
    ///
    /// Computes RMSE, MAE, R², residual SD, and optionally AIC/AICc.
    pub fn return_diagnostics(mut self) -> Self {
        self.compute_diagnostics = true;
        self
    }

    /// Include residuals in the result.
    ///
    /// Returns the differences between original and smoothed values.
    pub fn return_residuals(mut self) -> Self {
        self.compute_residuals = true;
        self
    }

    /// Include robustness weights in the result.
    ///
    /// Returns the final weights from iterative robustness refinement.
    pub fn return_robustness_weights(mut self) -> Self {
        self.compute_robustness_weights = true;
        self
    }

    // ========================================================================
    // Diagnostic Accessors
    // ========================================================================

    /// Create a quick scanner with minimal robustness.
    ///
    /// Preset configuration for fast smoothing:
    /// * iterations: 0 (no robustness)
    /// * fraction: 0.5
    pub fn quick() -> Self {
        Self::new().iterations(0).fraction(T::from(0.5).unwrap())
    }

    /// Create a robust scanner with heavy outlier downweighting.
    ///
    /// Preset configuration for robust smoothing:
    /// * iterations: 5
    /// * robustness_method: Bisquare
    pub fn robust() -> Self {
        Self::new()
            .iterations(5)
            .robustness_method(RobustnessMethod::Bisquare)
    }
}

// ============================================================================
// Adapter Selection
// ============================================================================

/// Trait for converting LowessBuilder into a specific adapter builder.
///
/// This trait enables the adapter selection pattern where users can
/// choose between Batch, Streaming, or Online execution modes.
pub trait LowessAdapter<T: Float> {
    /// The output type after conversion (adapter-specific builder).
    type Output;

    /// Convert a LowessBuilder into an adapter-specific builder.
    ///
    /// # Parameters
    ///
    /// * `builder` - The configured LowessBuilder
    ///
    /// # Returns
    ///
    /// An adapter-specific builder ready to execute smoothing.
    fn convert(builder: LowessBuilder<T>) -> Self::Output;
}

// ============================================================================
// Adapter Marker Types
// ============================================================================

/// Marker type for batch adapter selection.
///
/// Use with `builder.adapter(Batch)` to select batch execution mode.
/// Batch mode processes complete datasets in memory with full feature support.
/// Uses parallel execution by default (fastLowess).
#[derive(Debug, Clone, Copy)]
pub struct Batch;

impl<T: Float> LowessAdapter<T> for Batch {
    type Output = BatchLowessBuilder<T>;

    fn convert(builder: LowessBuilder<T>) -> Self::Output {
        BatchLowessBuilder {
            fraction: builder.fraction.unwrap_or_else(|| T::from(0.67).unwrap()),
            iterations: builder.iterations,
            delta: builder.delta,
            weight_function: builder.weight_function,
            robustness_method: builder.robustness_method,
            interval_type: builder.interval_type,
            cv_fractions: builder.cv_fractions,
            cv_method: builder.cv_method,
            auto_convergence: builder.auto_convergence,
            compute_diagnostics: builder.compute_diagnostics,
            compute_residuals: builder.compute_residuals,
            compute_robustness_weights: builder.compute_robustness_weights,
            zero_weight_fallback: builder.zero_weight_fallback,
            deferred_error: None,
            parallel: true, // Default to parallel
        }
    }
}

/// Marker type for streaming adapter selection.
///
/// Use with `builder.adapter(Streaming)` to select streaming execution mode.
/// Streaming mode processes large datasets in chunks with configurable overlap.
/// Uses parallel execution by default.
///
/// # Fields
///
/// * `chunk_size` - Size of each processing chunk (default: 5000)
/// * `overlap` - Overlap between consecutive chunks (default: 500)
/// * `boundary_policy` - Boundary handling policy (default: Extend)
/// * `merge_strategy` - Overlap merging strategy (default: WeightedAverage)
#[derive(Debug, Clone, Copy)]
pub struct Streaming;

impl<T: Float> LowessAdapter<T> for Streaming {
    type Output = StreamingLowessBuilder<T>;

    fn convert(builder: LowessBuilder<T>) -> Self::Output {
        StreamingLowessBuilder {
            chunk_size: 5000,
            overlap: 500,
            fraction: builder.fraction.unwrap_or_else(|| T::from(0.1).unwrap()),
            iterations: builder.iterations,
            delta: builder.delta.unwrap_or_else(T::zero),
            weight_function: builder.weight_function,
            boundary_policy: BoundaryPolicy::Extend,
            robustness_method: builder.robustness_method,
            zero_weight_fallback: builder.zero_weight_fallback,
            merge_strategy: MergeStrategy::WeightedAverage,
            compute_residuals: builder.compute_residuals,
            deferred_error: None,
            parallel: true, // Default to parallel (fastLowess)
        }
    }
}

/// Marker type for online adapter selection.
///
/// Use with `builder.adapter(Online)` to select online execution mode.
/// Online mode maintains a sliding window for incremental updates.
/// Uses sequential execution by default for lower latency.
///
/// # Fields
///
/// * `window_capacity` - Maximum number of points to retain (default: 1000)
/// * `min_points` - Minimum points before smoothing starts (default: 3)
/// * `update_mode` - Update mode for incremental processing (default: Incremental)
#[derive(Debug, Clone, Copy)]
pub struct Online;

impl<T: Float> LowessAdapter<T> for Online {
    type Output = OnlineLowessBuilder<T>;

    fn convert(builder: LowessBuilder<T>) -> Self::Output {
        OnlineLowessBuilder {
            window_capacity: 1000,
            min_points: 3,
            fraction: builder.fraction.unwrap_or_else(|| T::from(0.2).unwrap()),
            iterations: builder.iterations,
            weight_function: builder.weight_function,
            update_mode: UpdateMode::Incremental,
            robustness_method: builder.robustness_method,
            zero_weight_fallback: builder.zero_weight_fallback,
            compute_residuals: builder.compute_residuals,
            deferred_error: None,
            parallel: false, // Sequential for lower latency
        }
    }
}

// ============================================================================
// Cross-Validation Strategy
// ============================================================================

/// Cross-validation strategy for automatic fraction selection.
///
/// Determines how the dataset is split for cross-validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossValidationStrategy {
    /// K-fold cross-validation.
    ///
    /// Splits data into k folds, using each fold as validation set once.
    KFold,

    /// Leave-one-out cross-validation.
    ///
    /// Uses each point as a validation set once (equivalent to n-fold CV).
    LeaveOneOut,

    /// Leave-one-out cross-validation (alias).
    ///
    /// Alias for `LeaveOneOut` to match `CVMethod::LOOCV`.
    LOOCV,
}

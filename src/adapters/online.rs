//! Online adapter for incremental LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the online (incremental) execution adapter for LOWESS
//! smoothing. It maintains a sliding window of recent observations and produces
//! smoothed values for new points as they arrive, making it suitable for
//! real-time data streams and incremental updates where data arrives
//! sequentially.
//!
//! ## Design notes
//!
//! * Uses a fixed-size circular buffer (VecDeque) for the sliding window.
//! * Automatically evicts oldest points when capacity is reached.
//! * Performs full LOWESS smoothing on the current window for each new point.
//! * Supports basic smoothing and residuals only (no intervals or diagnostics).
//! * Requires minimum number of points before smoothing begins.
//! * Generic over `Float` types to support f32 and f64.
//! * **fastLowess addition**: Supports optional parallel execution via rayon.
//!
//! ## Key concepts
//!
//! ### Sliding Window
//! The online adapter maintains a fixed-size window:
//! ```text
//! Initial state (capacity=5):
//! Buffer: [_, _, _, _, _]
//!
//! After 3 points:
//! Buffer: [x1, x2, x3, _, _]
//!
//! After 7 points (buffer full, oldest dropped):
//! Buffer: [x3, x4, x5, x6, x7]
//!         ↑ oldest    newest ↑
//! ```
//!
//! ### Incremental Processing
//! For each new point:
//! 1. Validate the point (finite values)
//! 2. Add to window (evict oldest if at capacity)
//! 3. Check if minimum points threshold is met
//! 4. Perform LOWESS smoothing on current window
//! 5. Return smoothed value for the newest point
//!
//! ### Update Modes
//! * **Incremental**: Fast updates, good accuracy (O(window))
//! * **Full**: Slower updates, excellent accuracy (O(window²))
//!
//! ### Initialization Phase
//! Before `min_points` are accumulated, `add_point()` returns `None`.
//! Once enough points are available, smoothing begins.
//!
//! ## Supported features
//!
//! * **Robustness iterations**: Downweight outliers iteratively
//! * **Residuals**: Differences between original and smoothed values
//! * **Window snapshots**: Get full `LowessResult` for current window
//! * **Reset capability**: Clear window for handling data gaps
//! * **Parallel execution**: Optional parallel smoothing via rayon (fastLowess)
//!
//! ## Invariants
//!
//! * Window size never exceeds capacity.
//! * All values in window are finite (no NaN or infinity).
//! * At least `min_points` are required before smoothing.
//! * Window maintains insertion order (oldest to newest).
//! * Smoothing is performed on sorted window data.
//!
//! ## Non-goals
//!
//! * This adapter does not support confidence/prediction intervals.
//! * This adapter does not compute diagnostic statistics.
//! * This adapter does not support cross-validation.
//! * This adapter does not handle batch processing (use batch adapter).
//! * This adapter does not handle out-of-order points.
//!
//! ## Visibility
//!
//! The online adapter is part of the public API through the high-level
//! `Lowess` builder. Direct usage of `OnlineLowess` is possible but not
//! the primary interface.

use crate::engine::executor::{ExecutorOutput, LowessConfig, LowessExecutor};

use lowess::testing::algorithms::regression::ZeroWeightFallback;
use lowess::testing::algorithms::robustness::RobustnessMethod;
use lowess::testing::engine::validator::Validator;
use lowess::testing::math::kernel::WeightFunction;
use lowess::testing::primitives::errors::LowessError;
use lowess::testing::primitives::partition::UpdateMode;

use crate::input::LowessInput;
use num_traits::Float;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::result::Result;
use std::vec::Vec;

// ============================================================================
// Online LOWESS Builder
// ============================================================================

/// Builder for online LOWESS processor.
///
/// Configures parameters for real-time streaming data processing with a
/// sliding window. Provides a fluent API for setting all parameters.
///
/// # Fields
///
/// ## Window Configuration
/// * `window_capacity` - Maximum number of points to retain (default: 1000)
/// * `min_points` - Minimum points before smoothing starts (default: 3)
///
/// ## Core Parameters
/// * `fraction` - Smoothing fraction (default: 0.2)
/// * `iterations` - Number of robustness iterations (default: 1)
///
/// ## Algorithm Configuration
/// * `weight_fn` - Kernel weight function (default: Tricube)
/// * `robustness_method` - Robustness weighting method (default: Bisquare)
/// * `zero_weight_fallback` - Zero weight fallback policy (default: UseLocalMean)
/// * `update_mode` - Update mode for incremental processing (default: Incremental)
///
/// ## Output Options
/// * `compute_residuals` - Whether to return residuals (default: false)
///
/// ## Execution Options (fastLowess)
/// * `parallel` - Whether to use parallel execution (default: false)
///
/// ## Internal
/// * `deferred_error` - Deferred error from adapter conversion (default: None)
///
/// # Note
///
/// Online adapter supports basic smoothing and residuals only.
/// For advanced features (confidence intervals, diagnostics), use the Batch adapter.
#[derive(Debug, Clone)]
pub struct OnlineLowessBuilder<T: Float> {
    /// Window capacity (maximum number of points to retain)
    pub window_capacity: usize,

    /// Minimum points before smoothing starts
    pub min_points: usize,

    /// Smoothing fraction (span)
    pub fraction: T,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Kernel weight function
    pub weight_function: WeightFunction,

    /// Update mode for incremental processing
    pub update_mode: UpdateMode,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Whether to return residuals
    pub compute_residuals: bool,

    /// Deferred error from adapter conversion
    pub deferred_error: Option<LowessError>,

    /// Whether to use parallel execution (fastLowess addition)
    pub parallel: bool,
}

impl<T: Float> Default for OnlineLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> OnlineLowessBuilder<T> {
    /// Create a new online LOWESS builder with default parameters.
    ///
    /// # Defaults
    ///
    /// * window_capacity: 1000
    /// * min_points: 3
    /// * fraction: 0.2
    /// * iterations: 1
    /// * weight_fn: Tricube
    /// * update_mode: Incremental
    /// * robustness_method: Bisquare
    /// * zero_weight_fallback: UseLocalMean
    /// * compute_residuals: false
    /// * parallel: false (fastLowess)
    /// * deferred_error: None
    fn new() -> Self {
        Self {
            window_capacity: 1000,
            min_points: 3,
            fraction: T::from(0.2).unwrap(),
            iterations: 1,
            weight_function: WeightFunction::default(),
            update_mode: UpdateMode::default(),
            robustness_method: RobustnessMethod::Bisquare,
            zero_weight_fallback: ZeroWeightFallback::default(),
            compute_residuals: false,
            deferred_error: None,
            parallel: false, // Online mode typically sequential for latency
        }
    }

    // ========================================================================
    // Window Configuration Setters
    // ========================================================================

    /// Set window capacity (maximum number of points to retain).
    ///
    /// # Typical values
    ///
    /// * Small windows: 100-500 (fast, less smooth)
    /// * Medium windows: 500-2000 (balanced)
    /// * Large windows: 2000-10000 (slow, very smooth)
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        self.window_capacity = capacity;
        self
    }

    /// Set minimum points before smoothing starts.
    ///
    /// # Constraints
    ///
    /// * Minimum: 2 (required for linear regression)
    /// * Maximum: window_capacity
    pub fn min_points(mut self, min: usize) -> Self {
        self.min_points = min;
        self
    }

    /// Set parallel execution mode.
    ///
    /// # Note
    ///
    /// For online processing, sequential execution (parallel=false) is typically
    /// preferred for lower latency. Parallel execution may be beneficial for
    /// very large windows.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the online processor.
    ///
    /// # Returns
    ///
    /// A configured `OnlineLowess` processor ready to accept points.
    ///
    /// # Errors
    ///
    /// Returns `LowessError` if:
    /// * A deferred error was set during configuration
    /// * Window capacity is too small (< 3)
    /// * min_points is invalid (< 2 or > window_capacity)
    pub fn build(self) -> Result<OnlineLowess<T>, LowessError> {
        if let Some(err) = self.deferred_error {
            return Err(err);
        }

        // Validate fraction
        Validator::validate_fraction(self.fraction)?;

        // Validate configuration early
        Validator::validate_window_capacity(self.window_capacity, 3)?;
        Validator::validate_min_points(self.min_points, self.window_capacity)?;

        let capacity = self.window_capacity;
        Ok(OnlineLowess {
            config: self,
            window_x: VecDeque::with_capacity(capacity),
            window_y: VecDeque::with_capacity(capacity),
        })
    }
}

// ============================================================================
// Online LOWESS Output
// ============================================================================

/// Result of a single online update.
///
/// Contains the smoothed value and optional outputs for a single point
/// added to the online processor.
///
/// # Fields
///
/// * `smoothed` - Smoothed value for the latest point
/// * `std_error` - Standard error (currently not computed in online mode)
/// * `residual` - Residual (y - smoothed) if requested
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineOutput<T> {
    /// Smoothed value for the latest point
    pub smoothed: T,

    /// Standard error (if computed)
    pub std_error: Option<T>,

    /// Residual (y - smoothed)
    pub residual: Option<T>,
}

// ============================================================================
// Online LOWESS Processor
// ============================================================================

/// Online LOWESS processor for streaming data.
///
/// Maintains a sliding window of recent observations and produces smoothed
/// values for new points as they arrive. Uses a circular buffer to
/// efficiently manage the window.
///
/// # Fields
///
/// * `config` - Configuration from builder
/// * `window_x` - Circular buffer of x-values
/// * `window_y` - Circular buffer of y-values
pub struct OnlineLowess<T: Float> {
    config: OnlineLowessBuilder<T>,
    window_x: VecDeque<T>,
    window_y: VecDeque<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> OnlineLowess<T> {
    /// Add a new point and get its smoothed value.
    ///
    /// # Parameters
    ///
    /// * `x` - X-coordinate of new point
    /// * `y` - Y-coordinate of new point
    ///
    /// # Returns
    ///
    /// * `Ok(Some(OnlineOutput))` - If enough points accumulated
    /// * `Ok(None)` - If still in initialization phase (< min_points)
    /// * `Err(LowessError)` - If point validation fails
    ///
    /// # Errors
    ///
    /// Returns `LowessError::InvalidNumericValue` if x or y is not finite
    /// (NaN or infinity).
    ///
    /// # Algorithm
    ///
    /// 1. Validate new point (finite values)
    /// 2. Add to window (evict oldest if at capacity)
    /// 3. Check if minimum points threshold is met
    /// 4. Special case: exactly 2 points uses direct linear fit
    /// 5. Otherwise: perform full LOWESS smoothing on window
    /// 6. Return smoothed value for newest point
    pub fn add_point(&mut self, x: T, y: T) -> Result<Option<OnlineOutput<T>>, LowessError> {
        // Validate new point
        if !x.is_finite() {
            return Err(LowessError::InvalidNumericValue(format!(
                "x={}",
                x.to_f64().unwrap_or(f64::NAN)
            )));
        }
        if !y.is_finite() {
            return Err(LowessError::InvalidNumericValue(format!(
                "y={}",
                y.to_f64().unwrap_or(f64::NAN)
            )));
        }

        // Add to window
        self.window_x.push_back(x);
        self.window_y.push_back(y);

        // Evict oldest if over capacity
        if self.window_x.len() > self.config.window_capacity {
            self.window_x.pop_front();
            self.window_y.pop_front();
        }

        // Check if we have enough points
        if self.window_x.len() < self.config.min_points {
            return Ok(None);
        }

        // Convert window to vectors for smoothing
        let x_vec: Vec<T> = self.window_x.iter().copied().collect();
        let y_vec: Vec<T> = self.window_y.iter().copied().collect();

        // Special case: exactly two points, use exact linear fit
        if x_vec.len() == 2 {
            let x0 = x_vec[0];
            let x1 = x_vec[1];
            let y0 = y_vec[0];
            let y1 = y_vec[1];

            let smoothed = if x1 != x0 {
                let slope = (y1 - y0) / (x1 - x0);
                y0 + slope * (x1 - x0)
            } else {
                // Identical x: use mean for stability
                (y0 + y1) / T::from(2.0).unwrap()
            };

            let residual = y - smoothed;

            return Ok(Some(OnlineOutput {
                smoothed,
                std_error: None,
                residual: Some(residual),
            }));
        }

        // Smooth using LOWESS for windows of size >= 3
        let config = LowessConfig {
            fraction: Some(self.config.fraction),
            iterations: self.config.iterations,
            delta: T::zero(), // No delta optimization in online mode
            weight_function: self.config.weight_function,
            zero_weight_fallback: self.config.zero_weight_fallback.to_u8(),
            robustness_method: self.config.robustness_method,
            cv_fractions: None,
            cv_method: None,
            auto_convergence: None,
            return_variance: None,
            parallel: self.config.parallel,
        };

        let result: ExecutorOutput<T> = LowessExecutor::run_with_config(&x_vec, &y_vec, config);
        let smoothed_vec = result.smoothed;
        let se_vec = result.std_errors;

        // Get the smoothed value for the newest point (last)
        let smoothed = smoothed_vec.last().copied().ok_or_else(|| {
            LowessError::InvalidNumericValue("No smoothed output produced".into())
        })?;
        let std_error = se_vec.as_ref().and_then(|v| v.last().copied());
        let residual = y - smoothed;

        Ok(Some(OnlineOutput {
            smoothed,
            std_error,
            residual: Some(residual),
        }))
    }

    /// Add multiple points in bulk.
    ///
    /// # Parameters
    ///
    /// * `x` - X-coordinates for the points
    /// * `y` - Y-coordinates for the points
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<Option<OnlineOutput>>)` - Vector of results for each point
    /// * `Err(LowessError)` - If point validation fails or input lengths mismatch
    ///
    /// # Errors
    ///
    /// Returns `LowessError::InvalidNumericValue` if any x or y is not finite.
    /// Returns `LowessError::InvalidInput` if x and y have different lengths.
    ///
    /// # Algorithm
    ///
    /// Iterates through all provided points and calls `add_point` for each one.
    /// Results are collected and returned in the same order as inputs.
    pub fn add_points<I1, I2>(
        &mut self,
        x: &I1,
        y: &I2,
    ) -> Result<Vec<Option<OnlineOutput<T>>>, LowessError>
    where
        I1: LowessInput<T> + ?Sized,
        I2: LowessInput<T> + ?Sized,
    {
        let x_slice = x.as_lowess_slice()?;
        let y_slice = y.as_lowess_slice()?;

        if x_slice.len() != y_slice.len() {
            return Err(LowessError::InvalidInput(format!(
                "Input length mismatch: x={}, y={}",
                x_slice.len(),
                y_slice.len()
            )));
        }

        let mut results = Vec::with_capacity(x_slice.len());
        for (xi, yi) in x_slice.iter().zip(y_slice.iter()) {
            results.push(self.add_point(*xi, *yi)?);
        }
        Ok(results)
    }

    /// Get the current window size.
    ///
    /// # Returns
    ///
    /// Number of points currently in the window.
    pub fn window_size(&self) -> usize {
        self.window_x.len()
    }

    /// Clear the window.
    ///
    /// Resets the processor to initial state by removing all points from
    /// the window. Useful for handling gaps in time series data or
    /// restarting the smoothing process.
    pub fn reset(&mut self) {
        self.window_x.clear();
        self.window_y.clear();
    }
}

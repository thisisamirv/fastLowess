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
//! * Uses a fixed-size circular buffer for the sliding window.
//! * Automatically evicts oldest points when capacity is reached.
//! * Performs full LOWESS smoothing on the current window for each new point.
//! * Supports basic smoothing and residuals only (no intervals or diagnostics).
//! * Requires minimum number of points before smoothing begins.
//! * Delegates computation to the `lowess` crate's online adapter.
//! * Adds parallel execution via `rayon` (fastLowess extension).
//! * Generic over `Float` types to support f32 and f64.
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
//! * **Reset capability**: Clear window for handling data gaps
//! * **Parallel execution**: Rayon-based parallelism (fastLowess extension)
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

use crate::engine::executor::smooth_pass_parallel;
use crate::input::LowessInput;

pub use lowess::internals::adapters::online::OnlineOutput;
use lowess::internals::adapters::online::{OnlineLowess, OnlineLowessBuilder};
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::errors::LowessError;
use lowess::internals::primitives::partition::UpdateMode;

use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// ============================================================================
// Extended Online LOWESS Builder
// ============================================================================

/// Builder for online LOWESS processor with parallel support.
///
/// Configures parameters for real-time streaming data processing with a
/// sliding window. Wraps the core `OnlineLowessBuilder` from the `lowess`
/// crate and adds fastLowess-specific extensions (parallel execution).
///
/// # Fields
///
/// * `base` - Core builder from the lowess crate (window, fraction, etc.)
/// * `parallel` - Whether to use parallel execution (fastLowess extension)
///
/// # Note
///
/// Online adapter supports basic smoothing and residuals only.
/// For advanced features (confidence intervals, diagnostics), use the
/// Batch adapter.
#[derive(Debug, Clone)]
pub struct ExtendedOnlineLowessBuilder<T: Float> {
    /// Base builder from the lowess crate
    pub base: OnlineLowessBuilder<T>,

    /// Whether to use parallel execution (fastLowess extension)
    pub parallel: bool,
}

impl<T: Float> Default for ExtendedOnlineLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> ExtendedOnlineLowessBuilder<T> {
    /// Create a new online LOWESS builder with default parameters.
    ///
    /// # Defaults
    ///
    /// * All base parameters from lowess OnlineLowessBuilder
    /// * parallel: true (fastLowess extension)
    fn new() -> Self {
        Self {
            base: OnlineLowessBuilder::default(),
            parallel: true,
        }
    }

    /// Set parallel execution mode.
    ///
    /// Parallel execution can speed up processing for windows with many
    /// points by distributing the local regression fits across CPU cores.
    ///
    /// # Parameters
    ///
    /// * `parallel` - Whether to enable parallel execution (default: true)
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set the maximum window capacity.
    ///
    /// # Parameters
    ///
    /// * `capacity` - Maximum number of points to retain in the window
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        self.base = self.base.window_capacity(capacity);
        self
    }

    /// Set the minimum points required before smoothing starts.
    ///
    /// # Parameters
    ///
    /// * `min_points` - Minimum points before `add_point()` returns smoothed values
    pub fn min_points(mut self, min_points: usize) -> Self {
        self.base = self.base.min_points(min_points);
        self
    }

    /// Set the update mode for incremental processing.
    ///
    /// # Parameters
    ///
    /// * `mode` - Update mode (Incremental or Full)
    pub fn update_mode(mut self, mode: UpdateMode) -> Self {
        self.base.update_mode = mode;
        self
    }

    /// Set the kernel weight function.
    ///
    /// # Parameters
    ///
    /// * `wf` - Weight function (e.g., Tricube, Epanechnikov)
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.base.weight_function = wf;
        self
    }
}

// ============================================================================
// Extended Online LOWESS Processor
// ============================================================================

/// Online LOWESS processor with parallel support.
///
/// Performs incremental LOWESS smoothing on streaming data by maintaining
/// a sliding window and delegating to the base `lowess` implementation
/// with optional parallel execution.
pub struct ExtendedOnlineLowess<T: Float> {
    processor: OnlineLowess<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ExtendedOnlineLowess<T> {
    /// Add a new point and return the smoothed value.
    ///
    /// Adds the point to the sliding window, evicting the oldest point if
    /// at capacity, then performs LOWESS smoothing on the current window.
    ///
    /// # Parameters
    ///
    /// * `x` - Independent variable value
    /// * `y` - Dependent variable value
    ///
    /// # Returns
    ///
    /// * `Ok(Some(output))` - Smoothed value and metadata (if enough points)
    /// * `Ok(None)` - Not enough points yet (initialization phase)
    ///
    /// # Errors
    ///
    /// Returns `LowessError` if input values are not finite.
    pub fn add_point(&mut self, x: T, y: T) -> Result<Option<OnlineOutput<T>>, LowessError> {
        self.processor.add_point(x, y)
    }

    /// Add multiple points and return their smoothed values.
    ///
    /// Convenience method for processing multiple points sequentially.
    ///
    /// # Parameters
    ///
    /// * `x` - Independent variable values
    /// * `y` - Dependent variable values (must have same length as x)
    ///
    /// # Returns
    ///
    /// Vector of optional smoothed values corresponding to each input point.
    ///
    /// # Errors
    ///
    /// Returns `LowessError` if input arrays have different lengths or
    /// contain non-finite values.
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
            return Err(LowessError::InvalidInput("x and y lengths differ".into()));
        }

        let mut results = Vec::with_capacity(x_slice.len());
        for (xi, yi) in x_slice.iter().zip(y_slice.iter()) {
            results.push(self.add_point(*xi, *yi)?);
        }
        Ok(results)
    }

    /// Reset the processor, clearing all window data.
    ///
    /// Useful for handling data gaps or starting a new sequence.
    pub fn reset(&mut self) {
        self.processor.reset();
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ExtendedOnlineLowessBuilder<T> {
    /// Build the online processor.
    ///
    /// Validates all configuration and creates a ready-to-use processor.
    ///
    /// # Returns
    ///
    /// A configured `ExtendedOnlineLowess` processor ready to receive points.
    ///
    /// # Errors
    ///
    /// Returns `LowessError` if configuration is invalid.
    pub fn build(self) -> Result<ExtendedOnlineLowess<T>, LowessError> {
        // Check for deferred errors from adapter conversion
        if let Some(ref err) = self.base.deferred_error {
            return Err(err.clone());
        }

        // Configure the base builder with parallel callback if enabled
        let mut builder = self.base.clone();

        if self.parallel {
            builder.custom_smooth_pass = Some(smooth_pass_parallel);
        } else {
            builder.custom_smooth_pass = None;
        }

        // Delegate execution to the base implementation
        let processor = builder.build()?;

        Ok(ExtendedOnlineLowess { processor })
    }
}

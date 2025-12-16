//! Streaming adapter for large-scale LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the streaming execution adapter for LOWESS smoothing
//! on datasets too large to fit in memory. It divides the data into overlapping
//! chunks, processes each chunk independently, and merges the results while
//! handling boundary effects. This enables LOWESS smoothing on arbitrarily
//! large datasets with controlled memory usage.
//!
//! ## Design notes
//!
//! * Processes data in fixed-size chunks with configurable overlap.
//! * Automatically sorts data within each chunk.
//! * Merges overlapping regions using configurable strategies.
//! * Handles boundary effects at chunk edges.
//! * Supports basic smoothing and residuals only (no intervals or diagnostics).
//! * Generic over `Float` types to support f32 and f64.
//! * Stateful: maintains overlap buffer between chunks.
//! * **fastLowess addition**: Supports optional parallel execution via rayon.
//!
//! ## Key concepts
//!
//! ### Chunked Processing
//! The streaming adapter divides data into overlapping chunks:
//! ```text
//! Chunk 1: [==========]
//! Chunk 2:       [==========]
//! Chunk 3:             [==========]
//!          ↑overlap↑
//! ```
//!
//! ### Overlap Strategy
//! Overlap ensures smooth transitions between chunks:
//! * **Rule of thumb**: overlap = 2 × window_size
//! * Larger overlap = better boundary handling, more computation
//! * Smaller overlap = faster processing, potential edge artifacts
//!
//! ### Merge Strategies
//! When chunks overlap, values are merged using:
//! * **Average**: Simple average of overlapping values
//! * **Weighted**: Distance-weighted average
//! * **KeepFirst**: Use value from first chunk
//! * **KeepLast**: Use value from last chunk
//!
//! ### Boundary Policies
//! At dataset boundaries:
//! * **Extend**: Extend window to dataset edge
//! * **Truncate**: Use smaller window near edges
//! * **Mirror**: Mirror data at boundaries
//!
//! ### Processing Flow
//! For each chunk:
//! 1. Validate chunk data
//! 2. Sort chunk by x-values
//! 3. Perform LOWESS smoothing
//! 4. Extract non-overlapping portion
//! 5. Merge overlap with previous chunk
//! 6. Buffer overlap for next chunk
//!
//! ## Supported features
//!
//! * **Robustness iterations**: Downweight outliers iteratively
//! * **Residuals**: Differences between original and smoothed values
//! * **Delta optimization**: Point skipping for dense data
//! * **Configurable chunking**: Chunk size and overlap
//! * **Merge strategies**: Multiple overlap merging options
//! * **Parallel execution**: Optional parallel smoothing via rayon (fastLowess)
//!
//! ## Invariants
//!
//! * Chunk size must be larger than overlap.
//! * Overlap must be large enough for local smoothing.
//! * All values must be finite (no NaN or infinity).
//! * At least 2 points required per chunk.
//! * Output order matches input order within each chunk.
//!
//! ## Non-goals
//!
//! * This adapter does not support confidence/prediction intervals.
//! * This adapter does not compute diagnostic statistics.
//! * This adapter does not support cross-validation.
//! * This adapter does not handle batch processing (use batch adapter).
//! * This adapter does not handle incremental updates (use online adapter).
//! * This adapter requires chunks to be provided in stream order.
//!
//! ## Visibility
//!
//! The streaming adapter is part of the public API through the high-level
//! `Lowess` builder. Direct usage of `StreamingLowess` is possible but not
//! the primary interface.

use crate::engine::executor::{ExecutorOutput, LowessConfig, LowessExecutor};
use crate::input::LowessInput;

use lowess::testing::algorithms::regression::ZeroWeightFallback;
use lowess::testing::algorithms::robustness::RobustnessMethod;
use lowess::testing::engine::output::LowessResult;
use lowess::testing::engine::validator::Validator;
use lowess::testing::math::kernel::WeightFunction;
use lowess::testing::primitives::errors::LowessError;
use lowess::testing::primitives::partition::{BoundaryPolicy, MergeStrategy};
use lowess::testing::primitives::sorting::sort_by_x;

use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;
use std::vec::Vec;

// ============================================================================
// Streaming LOWESS Builder
// ============================================================================

/// Builder for streaming LOWESS processor.
///
/// Configures parameters for chunked processing of large datasets that
/// don't fit in memory. Provides a fluent API for setting all parameters.
///
/// # Fields
///
/// ## Chunking Configuration
/// * `chunk_size` - Size of each processing chunk (default: 5000)
/// * `overlap` - Overlap between consecutive chunks (default: 500)
///
/// ## Core Parameters
/// * `fraction` - Smoothing fraction (default: 0.1)
/// * `iterations` - Number of robustness iterations (default: 2)
/// * `delta` - Delta parameter for interpolation (default: 0.0)
///
/// ## Algorithm Configuration
/// * `weight_fn` - Kernel weight function (default: Tricube)
/// * `robustness_method` - Robustness weighting method (default: Bisquare)
/// * `zero_weight_fallback` - Zero weight fallback policy (default: UseLocalMean)
/// * `boundary_policy` - Boundary handling policy (default: Extend)
/// * `merge_strategy` - Overlap merging strategy (default: Average)
///
/// ## Output Options
/// * `compute_residuals` - Whether to return residuals (default: false)
///
/// ## Execution Options (fastLowess)
/// * `parallel` - Whether to use parallel execution (default: true)
///
/// ## Internal
/// * `deferred_error` - Deferred error from adapter conversion (default: None)
///
/// # Note
///
/// Streaming adapter supports basic smoothing and residuals only.
/// For advanced features (confidence intervals, diagnostics, robustness weights),
/// use the Batch adapter.
#[derive(Debug, Clone)]
pub struct StreamingLowessBuilder<T: Float> {
    /// Chunk size for processing
    pub chunk_size: usize,

    /// Overlap between chunks
    pub overlap: usize,

    /// Smoothing fraction (span)
    pub fraction: T,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Delta parameter for interpolation
    pub delta: T,

    /// Kernel weight function
    pub weight_function: WeightFunction,

    /// Boundary handling policy
    pub boundary_policy: BoundaryPolicy,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Merging strategy for overlapping chunks
    pub merge_strategy: MergeStrategy,

    /// Whether to return residuals
    pub compute_residuals: bool,

    /// Deferred error from adapter conversion
    pub deferred_error: Option<LowessError>,

    /// Whether to use parallel execution (fastLowess addition)
    pub parallel: bool,
}

impl<T: Float> Default for StreamingLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> StreamingLowessBuilder<T> {
    /// Create a new streaming LOWESS builder with default parameters.
    ///
    /// # Defaults
    ///
    /// * chunk_size: 5000
    /// * overlap: 500
    /// * fraction: 0.1
    /// * iterations: 2
    /// * delta: 0.0
    /// * weight_fn: Tricube
    /// * boundary_policy: Extend
    /// * robustness_method: Bisquare
    /// * zero_weight_fallback: UseLocalMean
    /// * merge_strategy: Average
    /// * compute_residuals: false
    /// * parallel: true (fastLowess)
    /// * deferred_error: None
    fn new() -> Self {
        Self {
            chunk_size: 5000,
            overlap: 500,
            fraction: T::from(0.1).unwrap(),
            iterations: 2,
            delta: T::zero(),
            weight_function: WeightFunction::default(),
            boundary_policy: BoundaryPolicy::default(),
            robustness_method: RobustnessMethod::Bisquare,
            zero_weight_fallback: ZeroWeightFallback::default(),
            merge_strategy: MergeStrategy::default(),
            compute_residuals: false,
            deferred_error: None,
            parallel: true,
        }
    }

    // ========================================================================
    // Chunking Configuration Setters
    // ========================================================================

    /// Set chunk size for processing.
    ///
    /// # Typical values
    ///
    /// * Small chunks: 1,000-5,000 (low memory, more overhead)
    /// * Medium chunks: 5,000-20,000 (balanced)
    /// * Large chunks: 20,000-100,000 (high memory, less overhead)
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set overlap between chunks.
    ///
    /// # Rule of thumb
    ///
    /// overlap = 2 × window_size, where window_size = fraction × chunk_size
    ///
    /// # Constraints
    ///
    /// overlap < chunk_size
    pub fn overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set kernel weight function.
    pub fn weight_function(mut self, weight_function: WeightFunction) -> Self {
        self.weight_function = weight_function;
        self
    }

    /// Set parallel execution mode.
    ///
    /// # Note
    ///
    /// Parallel execution can significantly speed up processing for large chunks.
    /// Default is true for streaming adapter.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the streaming processor.
    ///
    /// # Returns
    ///
    /// A configured `StreamingLowess` processor ready to process chunks.
    ///
    /// # Errors
    ///
    /// Returns `LowessError` if:
    /// * A deferred error was set during configuration
    /// * Fraction is not in (0, 1]
    /// * Chunk size is too small (< 10)
    /// * Overlap is too large (>= chunk_size)
    pub fn build(self) -> Result<StreamingLowess<T>, LowessError> {
        if let Some(err) = self.deferred_error {
            return Err(err);
        }

        // Validate fraction
        Validator::validate_fraction(self.fraction)?;

        // Validate delta
        Validator::validate_delta(self.delta)?;

        // Validate chunk size
        Validator::validate_chunk_size(self.chunk_size, 10)?;

        // Validate overlap
        Validator::validate_overlap(self.overlap, self.chunk_size)?;

        Ok(StreamingLowess {
            config: self,
            overlap_buffer_x: Vec::new(),
            overlap_buffer_y: Vec::new(),
            overlap_buffer_smoothed: Vec::new(),
        })
    }
}

// ============================================================================
// Streaming LOWESS Processor
// ============================================================================

/// Streaming LOWESS processor for large datasets.
///
/// Processes data in chunks with configurable overlap to handle boundary
/// effects. Maintains state between chunks to merge overlapping regions.
///
/// # Fields
///
/// * `config` - Configuration from builder
/// * `overlap_buffer_x` - Buffered x-values from previous chunk's overlap
/// * `overlap_buffer_y` - Buffered y-values from previous chunk's overlap
/// * `overlap_buffer_smoothed` - Buffered smoothed values from previous chunk's overlap
pub struct StreamingLowess<T: Float> {
    config: StreamingLowessBuilder<T>,
    overlap_buffer_x: Vec<T>,
    overlap_buffer_y: Vec<T>,
    overlap_buffer_smoothed: Vec<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> StreamingLowess<T> {
    /// Process a chunk of data.
    ///
    /// Returns the smoothed values for the non-overlapping portion of this chunk.
    /// The overlap region is buffered for merging with the next chunk.
    ///
    /// # Parameters
    ///
    /// * `x` - X-coordinates for this chunk
    /// * `y` - Y-coordinates for this chunk
    ///
    /// # Returns
    ///
    /// `LowessResult` containing smoothed values for the non-overlapping portion.
    ///
    /// # Errors
    ///
    /// Returns `LowessError` if:
    /// * `MismatchedInputs`: x and y have different lengths
    /// * `InvalidNumericValue`: Inf or NaN in input
    /// * `TooFewPoints`: Chunk too small (min 2 points)
    ///
    /// # Algorithm
    ///
    /// 1. Validate chunk data
    /// 2. Sort chunk by x-values
    /// 3. Perform LOWESS smoothing on entire chunk
    /// 4. Merge overlap region with previous chunk (if any)
    /// 5. Extract non-overlapping portion for output
    /// 6. Buffer overlap region for next chunk
    pub fn process_chunk<I1, I2>(&mut self, x: &I1, y: &I2) -> Result<LowessResult<T>, LowessError>
    where
        I1: LowessInput<T> + ?Sized,
        I2: LowessInput<T> + ?Sized,
    {
        // Convert input to slices
        let x_slice = x.as_lowess_slice()?;
        let y_slice = y.as_lowess_slice()?;

        // Validate inputs using standard validator
        Validator::validate_inputs(x_slice, y_slice)?;

        // Sort chunk by x
        let sorted = sort_by_x(x_slice, y_slice);

        // Configure LOWESS for this chunk
        // Combine with overlap from previous chunk
        let prev_overlap_len: usize = self.overlap_buffer_smoothed.len();
        let (combined_x, combined_y) = if self.overlap_buffer_x.is_empty() {
            (sorted.x.clone(), sorted.y.clone())
        } else {
            let mut cx = core::mem::take(&mut self.overlap_buffer_x);
            cx.extend_from_slice(&sorted.x);
            let mut cy = core::mem::take(&mut self.overlap_buffer_y);
            cy.extend_from_slice(&sorted.y);
            (cx, cy)
        };

        let config = LowessConfig {
            fraction: Some(self.config.fraction),
            iterations: self.config.iterations,
            delta: self.config.delta,
            weight_function: self.config.weight_function,
            zero_weight_fallback: self.config.zero_weight_fallback.to_u8(),
            robustness_method: self.config.robustness_method,
            cv_fractions: None,
            cv_method: None,
            auto_convergence: None,
            return_variance: None,
            parallel: self.config.parallel,
        };

        // Execute LOWESS on combined data
        let result: ExecutorOutput<T> =
            LowessExecutor::run_with_config(&combined_x, &combined_y, config);
        let smoothed = result.smoothed;

        // Determine how much to return vs buffer
        let combined_len: usize = combined_x.len();
        let overlap_start: usize = combined_len.saturating_sub(self.config.overlap);
        let return_start: usize = prev_overlap_len;

        // Build output: merged overlap (if any) + new data
        let mut y_smooth_out = Vec::new();
        if prev_overlap_len > 0 {
            // Merge the overlap region
            let prev_smooth: Vec<T> = core::mem::take(&mut self.overlap_buffer_smoothed);
            for (i, (&prev_val, &curr_val)) in prev_smooth
                .iter()
                .zip(smoothed.iter())
                .take(prev_overlap_len)
                .enumerate()
            {
                let merged = match self.config.merge_strategy {
                    MergeStrategy::Average => (prev_val + curr_val) / T::from(2.0).unwrap(),
                    MergeStrategy::WeightedAverage => {
                        let weight = T::from(i as f64 / prev_overlap_len as f64).unwrap();
                        prev_val * (T::one() - weight) + curr_val * weight
                    }
                    MergeStrategy::TakeFirst => prev_val,
                    MergeStrategy::TakeLast => curr_val,
                };
                y_smooth_out.push(merged);
            }
        }

        // Add non-overlap portion
        if return_start < overlap_start {
            y_smooth_out.extend_from_slice(&smoothed[return_start..overlap_start]);
        }

        // Calculate residuals for output
        let end_idx: usize = return_start + y_smooth_out.len();
        let residuals_out = if self.config.compute_residuals {
            let y_slice = &combined_y[return_start..end_idx];
            Some(
                y_slice
                    .iter()
                    .zip(y_smooth_out.iter())
                    .map(|(y_val, s)| *y_val - *s)
                    .collect(),
            )
        } else {
            None
        };

        // Buffer overlap for next chunk
        if overlap_start < combined_len {
            self.overlap_buffer_x = combined_x[overlap_start..].to_vec();
            self.overlap_buffer_y = combined_y[overlap_start..].to_vec();
            self.overlap_buffer_smoothed = smoothed[overlap_start..].to_vec();
        } else {
            self.overlap_buffer_x.clear();
            self.overlap_buffer_y.clear();
            self.overlap_buffer_smoothed.clear();
        }

        // Note: We return results in sorted order (by x) for streaming chunks.
        // Unsorting partial results is ambiguous since we only return a subset of the chunk.
        // The full batch adapter handles global unsorting when processing complete datasets.
        let x_out = combined_x[return_start..end_idx].to_vec();

        Ok(LowessResult {
            x: x_out,
            y: y_smooth_out,
            standard_errors: None,
            confidence_lower: None,
            confidence_upper: None,
            prediction_lower: None,
            prediction_upper: None,
            residuals: residuals_out,
            robustness_weights: None,
            diagnostics: None,
            iterations_used: result.iterations,
            fraction_used: self.config.fraction,
            cv_scores: None,
        })
    }

    /// Finalize processing and get any remaining buffered data.
    ///
    /// Call this after processing all chunks to get smoothed values for
    /// the final overlap region.
    ///
    /// # Returns
    ///
    /// `LowessResult` containing smoothed values for the buffered overlap,
    /// or an empty result if no data is buffered.
    pub fn finalize(&mut self) -> Result<LowessResult<T>, LowessError> {
        if self.overlap_buffer_x.is_empty() {
            return Ok(LowessResult {
                x: Vec::new(),
                y: Vec::new(),
                standard_errors: None,
                confidence_lower: None,
                confidence_upper: None,
                prediction_lower: None,
                prediction_upper: None,
                residuals: None,
                robustness_weights: None,
                diagnostics: None,
                iterations_used: None,
                fraction_used: self.config.fraction,
                cv_scores: None,
            });
        }

        // Return buffered overlap data
        let residuals = if self.config.compute_residuals {
            let mut res = Vec::with_capacity(self.overlap_buffer_x.len());
            for (i, &smoothed) in self.overlap_buffer_smoothed.iter().enumerate() {
                res.push(self.overlap_buffer_y[i] - smoothed);
            }
            Some(res)
        } else {
            None
        };

        let result = LowessResult {
            x: self.overlap_buffer_x.clone(),
            y: self.overlap_buffer_smoothed.clone(),
            standard_errors: None,
            confidence_lower: None,
            confidence_upper: None,
            prediction_lower: None,
            prediction_upper: None,
            residuals,
            robustness_weights: None,
            diagnostics: None,
            iterations_used: None,
            fraction_used: self.config.fraction,
            cv_scores: None,
        };

        // Clear buffers
        self.overlap_buffer_x.clear();
        self.overlap_buffer_y.clear();
        self.overlap_buffer_smoothed.clear();

        Ok(result)
    }

    /// Reset the processor state.
    ///
    /// Clears all buffered overlap data. Useful when starting a new stream
    /// or handling gaps in the data.
    pub fn reset(&mut self) {
        self.overlap_buffer_x.clear();
        self.overlap_buffer_y.clear();
        self.overlap_buffer_smoothed.clear();
    }
}

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
//! * Supports smoothing, residuals, robustness weights, and diagnostics.
//! * Delegates computation to the `lowess` crate's streaming adapter.
//! * Adds parallel execution via `rayon` (fastLowess extension).
//! * Generic over `Float` types to support f32 and f64.
//! * Stateful: maintains overlap buffer between chunks.
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
//! * **Extend**: Extend the first/last values to fill neighborhoods
//! * **Reflect**: Reflect values around the boundary
//! * **Zero**: Use zero padding at boundaries
//!
//! ### Processing Flow
//! For each chunk:
//! 1. Validate chunk data
//! 2. Sort chunk by x-values
//! 3. Perform LOWESS smoothing (parallel if enabled)
//! 4. Extract non-overlapping portion
//! 5. Merge overlap with previous chunk
//! 6. Buffer overlap for next chunk
//!
//! ## Supported features
//!
//! * **Robustness iterations**: Downweight outliers iteratively
//! * **Residuals**: Differences between original and smoothed values
//! * **Robustness weights**: Return weights used for each point
//! * **Diagnostics**: RMSE, MAE, R^2, Residual SD (cumulative)
//! * **Delta optimization**: Point skipping for dense data
//! * **Configurable chunking**: Chunk size and overlap
//! * **Merge strategies**: Multiple overlap merging options
//! * **Parallel execution**: Rayon-based parallelism (fastLowess extension)
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

use crate::engine::executor::smooth_pass_parallel;

use lowess::internals::adapters::streaming::{StreamingLowess, StreamingLowessBuilder};
use lowess::internals::algorithms::regression::ZeroWeightFallback;
use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::engine::output::LowessResult;
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::errors::LowessError;
use lowess::internals::primitives::partition::{BoundaryPolicy, MergeStrategy};

use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// ============================================================================
// Extended Streaming LOWESS Builder
// ============================================================================

/// Builder for streaming LOWESS processor with parallel support.
#[derive(Debug, Clone)]
pub struct ExtendedStreamingLowessBuilder<T: Float> {
    /// Base builder from the lowess crate
    pub base: StreamingLowessBuilder<T>,

    /// Whether to use parallel execution (fastLowess extension)
    pub parallel: bool,
}

impl<T: Float> Default for ExtendedStreamingLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> ExtendedStreamingLowessBuilder<T> {
    /// Create a new streaming LOWESS builder with default parameters.
    fn new() -> Self {
        Self {
            base: StreamingLowessBuilder::default(),
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
    // Streaming-Specific Setters
    // ========================================================================

    /// Set the chunk size for processing.
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.base = self.base.chunk_size(size);
        self
    }

    /// Set the overlap between consecutive chunks.
    pub fn overlap(mut self, size: usize) -> Self {
        self.base = self.base.overlap(size);
        self
    }

    /// Set the merge strategy for overlapping values.
    pub fn merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.base = self.base.merge_strategy(strategy);
        self
    }

    /// Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base = self.base.return_diagnostics(enabled);
        self
    }
}

// ============================================================================
// Extended Streaming LOWESS Processor
// ============================================================================

/// Streaming LOWESS processor with parallel support.
pub struct ExtendedStreamingLowess<T: Float> {
    processor: StreamingLowess<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ExtendedStreamingLowess<T> {
    /// Process a chunk of data.
    pub fn process_chunk(&mut self, x: &[T], y: &[T]) -> Result<LowessResult<T>, LowessError> {
        self.processor.process_chunk(x, y)
    }

    /// Finalize processing and get remaining buffered data.
    pub fn finalize(&mut self) -> Result<LowessResult<T>, LowessError> {
        self.processor.finalize()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ExtendedStreamingLowessBuilder<T> {
    /// Build the streaming processor.
    pub fn build(self) -> Result<ExtendedStreamingLowess<T>, LowessError> {
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

        let processor = builder.build()?;

        Ok(ExtendedStreamingLowess { processor })
    }
}

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
//! * **Extend**: Extend window to dataset edge
//! * **Truncate**: Use smaller window near edges
//! * **Mirror**: Mirror data at boundaries
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
//! * This adapter does not compute diagnostic statistics.
//! * This adapter does not support cross-validation.
//! * This adapter does not handle batch processing (use batch adapter).
//! * This adapter does not handle incremental updates (use online adapter).
//! * This adapter requires chunks to be provided in stream order.

use crate::engine::executor::smooth_pass_parallel;

use lowess::internals::adapters::streaming::{StreamingLowess, StreamingLowessBuilder};
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
///
/// Configures parameters for chunked processing of large datasets that
/// don't fit in memory. Wraps the core `StreamingLowessBuilder` from the
/// `lowess` crate and adds fastLowess-specific extensions (parallel execution).
///
/// # Fields
///
/// * `base` - Core builder from the lowess crate (chunk size, overlap, etc.)
/// * `parallel` - Whether to use parallel execution (fastLowess extension)
///
/// # Note
///
/// Streaming adapter supports basic smoothing and residuals only.
/// For advanced features (confidence intervals, diagnostics), use the
/// Batch adapter.
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
    ///
    /// # Defaults
    ///
    /// * All base parameters from lowess StreamingLowessBuilder
    /// * parallel: true (fastLowess extension)
    fn new() -> Self {
        Self {
            base: StreamingLowessBuilder::default(),
            parallel: true,
        }
    }

    /// Set parallel execution mode.
    ///
    /// Parallel execution can significantly speed up processing for large
    /// chunks by distributing the local regression fits across CPU cores.
    ///
    /// # Parameters
    ///
    /// * `parallel` - Whether to enable parallel execution (default: true)
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set the chunk size for processing.
    ///
    /// Larger chunks use more memory but may produce smoother results.
    ///
    /// # Parameters
    ///
    /// * `size` - Number of points per chunk (must be > overlap)
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.base = self.base.chunk_size(size);
        self
    }

    /// Set the overlap between consecutive chunks.
    ///
    /// Overlap ensures smooth transitions between chunks. Rule of thumb:
    /// overlap should be at least 2× the LOWESS window size.
    ///
    /// # Parameters
    ///
    /// * `size` - Number of overlapping points between chunks
    pub fn overlap(mut self, size: usize) -> Self {
        self.base = self.base.overlap(size);
        self
    }

    /// Set the boundary handling policy.
    ///
    /// # Parameters
    ///
    /// * `policy` - How to handle boundaries (Extend, Truncate, Mirror)
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.base.boundary_policy = policy;
        self
    }

    /// Set the merge strategy for overlapping values.
    ///
    /// # Parameters
    ///
    /// * `strategy` - How to merge overlapping values (Average, Weighted, etc.)
    pub fn merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.base.merge_strategy = strategy;
        self
    }

    /// Set the kernel weight function.
    ///
    /// # Parameters
    ///
    /// * `wf` - Weight function (e.g., Tricube, Epanechnikov)
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.base = self.base.weight_function(wf);
        self
    }
}

// ============================================================================
// Extended Streaming LOWESS Processor
// ============================================================================

/// Streaming LOWESS processor with parallel support.
///
/// Performs chunked LOWESS smoothing on large datasets by delegating to
/// the base `lowess` implementation with optional parallel execution.
/// Maintains state between chunks for proper overlap handling.
pub struct ExtendedStreamingLowess<T: Float> {
    processor: StreamingLowess<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ExtendedStreamingLowess<T> {
    /// Process a chunk of data.
    ///
    /// Call this method repeatedly with sequential chunks of data for true
    /// streaming processing. The processor maintains overlap buffers between
    /// calls to ensure smooth transitions.
    ///
    /// # Parameters
    ///
    /// * `x` - Independent variable values for this chunk
    /// * `y` - Dependent variable values for this chunk
    ///
    /// # Returns
    ///
    /// `LowessResult` containing smoothed values for the non-overlapping
    /// portion of this chunk (merged with previous chunk's overlap).
    ///
    /// # Errors
    ///
    /// Returns `LowessError` if chunk validation fails.
    pub fn process_chunk(&mut self, x: &[T], y: &[T]) -> Result<LowessResult<T>, LowessError> {
        self.processor.process_chunk(x, y)
    }

    /// Finalize processing and get remaining buffered data.
    ///
    /// Call this after processing all chunks to retrieve any remaining
    /// data in the overlap buffer. This ensures no data is lost.
    ///
    /// # Returns
    ///
    /// `LowessResult` containing smoothed values for the final overlap region.
    ///
    /// # Errors
    ///
    /// Returns `LowessError` if finalization fails.
    pub fn finalize(&mut self) -> Result<LowessResult<T>, LowessError> {
        self.processor.finalize()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ExtendedStreamingLowessBuilder<T> {
    /// Build the streaming processor.
    ///
    /// Validates all configuration and creates a ready-to-use processor.
    ///
    /// # Returns
    ///
    /// A configured `ExtendedStreamingLowess` processor ready to receive chunks.
    ///
    /// # Errors
    ///
    /// Returns `LowessError` if configuration is invalid (e.g., chunk_size <= overlap).
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

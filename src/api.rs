//! High-level API for LOWESS smoothing with parallel execution support.
//!
//! ## Purpose
//!
//! This module provides the primary user-facing API for LOWESS (Locally
//! Weighted Scatterplot Smoothing) with parallel execution capabilities.
//! It extends the `lowess` crate's API by providing Extended* adapters
//! that support parallel execution.
//!
//! ## Design notes
//!
//! * Re-exports `LowessBuilder` and `LowessAdapter` from the `lowess` crate.
//! * Provides custom `Batch`, `Streaming`, and `Online` marker types that
//!   return Extended* adapter builders with parallel execution support.
//! * All Extended* adapters wrap the base `lowess` adapter structs.
//!
//! ## Key concepts
//!
//! ### Builder Pattern
//! The `LowessBuilder` provides a fluent API (inherited from lowess):
//! ```text
//! Lowess::<f64>::new()
//!     .fraction(0.5)
//!     .iterations(2)
//!     .confidence_intervals(0.95)
//!     .adapter(Batch)  // Uses ExtendedBatchLowessBuilder with parallel=true
//!     .fit(x, y)
//! ```
//!
//! ### Adapter Selection
//! Three execution adapters are available:
//! * **Batch**: Complete datasets in memory (parallel by default)
//! * **Streaming**: Large datasets processed in chunks (parallel by default)
//! * **Online**: Incremental updates with sliding window (sequential by default)
//!
//! ## Visibility
//!
//! This module is the primary public API for the crate. Types and traits
//! defined here are stable and follow semantic versioning.

use std::result;

use num_traits::Float;

// Internal extended adapters
use crate::adapters::batch::ExtendedBatchLowessBuilder;
use crate::adapters::online::ExtendedOnlineLowessBuilder;
use crate::adapters::streaming::ExtendedStreamingLowessBuilder;

// ============================================================================
// Re-exports from lowess crate
// ============================================================================

// Import base marker types for delegation
use lowess::internals::api::Batch as BaseBatch;
use lowess::internals::api::Online as BaseOnline;
use lowess::internals::api::Streaming as BaseStreaming;

// Unused imports for user convenience
#[allow(unused_imports)]
pub use lowess::internals::api::CrossValidationStrategy;
#[allow(unused_imports)]
pub use lowess::internals::evaluation::cv::CVMethod;
#[allow(unused_imports)]
pub use lowess::internals::evaluation::intervals::IntervalMethod;

// Re-export LowessBuilder and related types from lowess
pub use lowess::internals::api::{LowessAdapter, LowessBuilder};

// Publicly re-exported types
pub use lowess::internals::algorithms::regression::ZeroWeightFallback;
pub use lowess::internals::algorithms::robustness::RobustnessMethod;
pub use lowess::internals::engine::output::LowessResult;
pub use lowess::internals::math::kernel::WeightFunction;
pub use lowess::internals::primitives::errors::LowessError;
pub use lowess::internals::primitives::partition::{BoundaryPolicy, MergeStrategy, UpdateMode};

// ============================================================================
// Type Aliases
// ============================================================================

/// Result type alias for LOWESS operations.
pub type Result<T> = result::Result<T, LowessError>;

// ============================================================================
// Adapter Module
// ============================================================================

/// Adapter selection namespace.
#[allow(non_snake_case)]
pub mod Adapter {
    pub use super::{Batch, Online, Streaming};
}

// ============================================================================
// Adapter Marker Types
// ============================================================================

/// Marker type for batch adapter selection.
#[derive(Debug, Clone, Copy)]
pub struct Batch;

impl<T: Float> LowessAdapter<T> for Batch {
    type Output = ExtendedBatchLowessBuilder<T>;

    fn convert(builder: LowessBuilder<T>) -> Self::Output {
        // Delegate to base implementation to create base builder
        let base = <BaseBatch as LowessAdapter<T>>::convert(builder);

        // Wrap with extension fields
        ExtendedBatchLowessBuilder {
            base,
            parallel: true, // Default to parallel
        }
    }
}

/// Marker type for streaming adapter selection.
///
/// Use with `builder.adapter(Streaming)` to select streaming execution mode.
/// Streaming mode processes large datasets in chunks with configurable overlap.
/// Uses parallel execution by default (fastLowess).
#[derive(Debug, Clone, Copy)]
pub struct Streaming;

impl<T: Float> LowessAdapter<T> for Streaming {
    type Output = ExtendedStreamingLowessBuilder<T>;

    fn convert(builder: LowessBuilder<T>) -> Self::Output {
        // Delegate to base implementation to create base builder
        let base = <BaseStreaming as LowessAdapter<T>>::convert(builder);

        // Wrap with extension fields
        ExtendedStreamingLowessBuilder {
            base,
            parallel: true, // Default to parallel (fastLowess)
        }
    }
}

/// Marker type for online adapter selection.
///
/// Use with `builder.adapter(Online)` to select online execution mode.
/// Online mode maintains a sliding window for incremental updates.
/// Uses sequential execution by default for lower latency.
#[derive(Debug, Clone, Copy)]
pub struct Online;

impl<T: Float> LowessAdapter<T> for Online {
    type Output = ExtendedOnlineLowessBuilder<T>;

    fn convert(builder: LowessBuilder<T>) -> Self::Output {
        // Delegate to base implementation to create base builder
        let base = <BaseOnline as LowessAdapter<T>>::convert(builder);

        // Wrap with extension fields
        ExtendedOnlineLowessBuilder {
            base,
            parallel: false, // Sequential for lower latency
        }
    }
}

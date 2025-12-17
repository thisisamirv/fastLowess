//! Parallel execution engine for LOWESS smoothing operations.
//!
//! ## Purpose
//!
//! This module provides the parallel smoothing function that is injected into
//! the `lowess` crate's execution engine. It enables multi-threaded execution
//! of the local regression fits, significantly speeding up LOWESS smoothing
//! for large datasets.
//!
//! ## Design notes
//!
//! * Provides a drop-in replacement for the sequential smoothing pass.
//! * Uses `rayon` for data-parallel execution across CPU cores.
//! * Reuses weight buffers per thread via `map_init` to minimize allocations.
//! * Compatible with the `SmoothPassFn` signature expected by `lowess`.
//! * Generic over `Float` types to support f32 and f64.
//!
//! ## Key concepts
//!
//! ### Parallel Fitting
//! Instead of fitting points sequentially, the parallel executor:
//! 1. Distributes points across available CPU cores
//! 2. Each thread fits its assigned points independently
//! 3. Results are collected and written to the output buffer
//!
//! ### Buffer Reuse
//! To minimize allocation overhead in tight loops:
//! * Each thread maintains its own weight buffer via `map_init`
//! * Buffers are zeroed and reused for each point (O(N) memset vs O(N) alloc)
//! * This provides significant speedup for large datasets
//!
//! ### Integration with lowess
//! The `smooth_pass_parallel` function matches the `SmoothPassFn` type:
//! ```text
//! fn(x, y, window_size, use_robustness, robustness_weights, y_smooth, weight_fn, zero_weight_flag)
//! ```
//! This allows it to be injected into the `lowess` executor's iteration loop.
//!
//! ## Invariants
//!
//! * Input x-values are assumed to be sorted.
//! * All arrays (x, y, y_smooth, robustness_weights) have the same length.
//! * Robustness weights are in [0, 1].
//! * Window size is at least 1 and at most n.
//!
//! ## Visibility
//!
//! This module is an internal implementation detail. The parallel function
//! is exported for use by the adapters but may change without notice.

use lowess::internals::algorithms::regression::{
    LinearRegression, Regression, RegressionContext, ZeroWeightFallback,
};
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::window::Window;
use num_traits::Float;
use rayon::prelude::*;

// ============================================================================
// Parallel Smoothing Function
// ============================================================================

/// Perform a single smoothing pass over all points in parallel.
///
/// This function distributes the local regression fits across CPU cores
/// using `rayon`, providing significant speedup for large datasets.
///
/// # Parameters
///
/// * `x` - Independent variable values (sorted)
/// * `y` - Dependent variable values
/// * `window_size` - Size of the local window for each fit
/// * `use_robustness` - Whether to apply robustness weights
/// * `robustness_weights` - Array of robustness weights (used if `use_robustness` is true)
/// * `y_smooth` - Output buffer for smoothed values (same length as x)
/// * `weight_function` - Kernel weight function
/// * `zero_weight_flag` - Zero weight fallback policy (0=UseLocalMean, 1=ReturnOriginal, 2=ReturnNone)
///
/// # Algorithm
///
/// 1. Create parallel iterator over all point indices
/// 2. For each point (in parallel):
///    - Initialize/reuse weight buffer
///    - Compute window bounds centered on current point
///    - Perform weighted local regression
///    - Return fitted value
/// 3. Collect results and copy to output buffer
///
/// # Performance
///
/// * O(N × k) total work distributed across P cores
/// * Where N = number of points, k = window size, P = CPU cores
/// * Memory: O(N) per thread for weight buffers (reused)
#[allow(clippy::too_many_arguments)]
pub fn smooth_pass_parallel<T>(
    x: &[T],
    y: &[T],
    window_size: usize,
    use_robustness: bool,
    robustness_weights: &[T],
    y_smooth: &mut [T],
    weight_function: WeightFunction,
    zero_weight_flag: u8,
) where
    T: Float + Send + Sync,
{
    let n = x.len();
    if n == 0 {
        return;
    }

    let zero_weight_fallback = ZeroWeightFallback::from_u8(zero_weight_flag);
    let fitter = LinearRegression;

    // Parallel fitting of all points
    // Use map_init to reuse the weights buffer per thread, reducing allocation overhead
    let results: Vec<T> = (0..n)
        .into_par_iter()
        .map_init(
            || vec![T::zero(); n],
            |weights, i| {
                // Reset weights buffer to zero for the current iteration
                // This is O(N) but just memset, much faster than allocation
                weights.fill(T::zero());

                let mut window = Window::initialize(i, window_size, n);
                window.recenter(x, i, n);

                let ctx = RegressionContext {
                    x,
                    y,
                    idx: i,
                    window,
                    use_robustness,
                    robustness_weights: if use_robustness {
                        robustness_weights
                    } else {
                        &[]
                    },
                    weights, // Reused buffer
                    weight_function,
                    zero_weight_fallback,
                };

                fitter.fit(ctx).unwrap_or(y[i])
            },
        )
        .collect();

    y_smooth.copy_from_slice(&results);
}

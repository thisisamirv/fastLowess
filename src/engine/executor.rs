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
//! * Supports delta optimization for sparse fitting with interpolation.
//!
//! ## Key concepts
//!
//! ### Parallel Fitting
//! Instead of fitting points sequentially, the parallel executor:
//! 1. Distributes points across available CPU cores
//! 2. Each thread fits its assigned points independently
//! 3. Results are collected and written to the output buffer
//!
//! ### Delta Optimization
//! When delta > 0, instead of fitting every point:
//! 1. Pre-compute "anchor" points spaced at least delta apart
//! 2. Parallel fit only those anchor points
//! 3. Linearly interpolate between anchors for intermediate points
//!
//! This provides significant speedup on dense data with minimal accuracy loss.
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
//! fn(x, y, window_size, delta, use_robustness, robustness_weights, y_smooth, weight_fn, zero_weight_flag)
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
/// When delta > 0, it uses delta optimization to fit only anchor points
/// and interpolate between them.
///
/// # Parameters
///
/// * `x` - Independent variable values (sorted)
/// * `y` - Dependent variable values
/// * `window_size` - Size of the local window for each fit
/// * `delta` - Interpolation optimization threshold (0 = fit all points)
/// * `use_robustness` - Whether to apply robustness weights
/// * `robustness_weights` - Array of robustness weights (used if `use_robustness` is true)
/// * `y_smooth` - Output buffer for smoothed values (same length as x)
/// * `weight_function` - Kernel weight function
/// * `zero_weight_flag` - Zero weight fallback policy (0=UseLocalMean, 1=ReturnOriginal, 2=ReturnNone)
///
/// # Algorithm
///
/// When delta = 0:
/// 1. Create parallel iterator over all point indices
/// 2. For each point (in parallel): perform weighted local regression
/// 3. Collect results and copy to output buffer
///
/// When delta > 0:
/// 1. Pre-compute anchor points spaced at least delta apart
/// 2. Parallel fit only anchor points
/// 3. Linearly interpolate between anchors
///
/// # Performance
///
/// * O(N × k) total work distributed across P cores (when delta = 0)
/// * O(A × k) + O(N) when delta > 0, where A = number of anchors
/// * Where N = number of points, k = window size, P = CPU cores
/// * Memory: O(N) per thread for weight buffers (reused)
#[allow(clippy::too_many_arguments)]
pub fn smooth_pass_parallel<T>(
    x: &[T],
    y: &[T],
    window_size: usize,
    delta: T,
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

    // If delta > 0, use delta optimization with anchor points
    if delta > T::zero() && n > 2 {
        // Step 1: Pre-compute anchor points (points to fit explicitly)
        let anchors = compute_anchor_points(x, delta);

        if anchors.is_empty() {
            // Fallback: fit all points if no anchors computed
            fit_all_points_parallel(
                x,
                y,
                window_size,
                use_robustness,
                robustness_weights,
                y_smooth,
                weight_function,
                zero_weight_fallback,
                &fitter,
            );
            return;
        }

        // Step 2: Parallel fit anchor points
        let anchor_values: Vec<(usize, T)> = anchors
            .par_iter()
            .map_init(
                || vec![T::zero(); n],
                |weights, &i| {
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
                        weights,
                        weight_function,
                        zero_weight_fallback,
                    };

                    (i, fitter.fit(ctx).unwrap_or(y[i]))
                },
            )
            .collect();

        // Step 3: Write anchor values and interpolate between them
        for &(idx, value) in &anchor_values {
            y_smooth[idx] = value;
        }

        // Interpolate between consecutive anchors
        for window in anchors.windows(2) {
            let start = window[0];
            let end = window[1];
            interpolate_gap(x, y_smooth, start, end);
        }

        // Handle any remaining points after the last anchor
        if let Some(&last_anchor) = anchors.last() {
            if last_anchor < n - 1 {
                // Fit the last point and interpolate
                let mut weights = vec![T::zero(); n];
                let mut window = Window::initialize(n - 1, window_size, n);
                window.recenter(x, n - 1, n);

                let ctx = RegressionContext {
                    x,
                    y,
                    idx: n - 1,
                    window,
                    use_robustness,
                    robustness_weights: if use_robustness {
                        robustness_weights
                    } else {
                        &[]
                    },
                    weights: &mut weights,
                    weight_function,
                    zero_weight_fallback,
                };

                y_smooth[n - 1] = fitter.fit(ctx).unwrap_or(y[n - 1]);
                interpolate_gap(x, y_smooth, last_anchor, n - 1);
            }
        }
    } else {
        // No delta optimization: fit all points in parallel
        fit_all_points_parallel(
            x,
            y,
            window_size,
            use_robustness,
            robustness_weights,
            y_smooth,
            weight_function,
            zero_weight_fallback,
            &fitter,
        );
    }
}

/// Compute anchor points for delta optimization.
///
/// Returns indices of points that should be explicitly fitted.
/// Points are spaced at least delta apart in x-values.
fn compute_anchor_points<T: Float>(x: &[T], delta: T) -> Vec<usize> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    let mut anchors = vec![0]; // Always include the first point
    let mut last_x = x[0];

    for (i, &xi) in x.iter().enumerate().skip(1) {
        if xi - last_x >= delta {
            anchors.push(i);
            last_x = xi;
        }
    }

    // Ensure the last point is included if not already
    if *anchors.last().unwrap_or(&0) != n - 1 {
        anchors.push(n - 1);
    }

    anchors
}

/// Linearly interpolate between two fitted anchor points.
fn interpolate_gap<T: Float>(x: &[T], y_smooth: &mut [T], start: usize, end: usize) {
    if end <= start + 1 {
        return;
    }

    let x0 = x[start];
    let x1 = x[end];
    let y0 = y_smooth[start];
    let y1 = y_smooth[end];

    let denom = x1 - x0;
    if denom <= T::zero() {
        let avg = (y0 + y1) / T::from(2.0).unwrap();
        for ys in y_smooth.iter_mut().take(end).skip(start + 1) {
            *ys = avg;
        }
        return;
    }

    for k in (start + 1)..end {
        let alpha = (x[k] - x0) / denom;
        y_smooth[k] = y0 + alpha * (y1 - y0);
    }
}

/// Fit all points in parallel (no delta optimization).
#[allow(clippy::too_many_arguments)]
fn fit_all_points_parallel<T>(
    x: &[T],
    y: &[T],
    window_size: usize,
    use_robustness: bool,
    robustness_weights: &[T],
    y_smooth: &mut [T],
    weight_function: WeightFunction,
    zero_weight_fallback: ZeroWeightFallback,
    fitter: &LinearRegression,
) where
    T: Float + Send + Sync,
{
    let n = x.len();

    let results: Vec<T> = (0..n)
        .into_par_iter()
        .map_init(
            || vec![T::zero(); n],
            |weights, i| {
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
                    weights,
                    weight_function,
                    zero_weight_fallback,
                };

                fitter.fit(ctx).unwrap_or(y[i])
            },
        )
        .collect();

    y_smooth.copy_from_slice(&results);
}

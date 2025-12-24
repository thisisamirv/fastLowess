//! Parallel execution engine for LOWESS smoothing operations.
//!
//! ## Purpose
//!
//! This module provides the parallel smoothing function that is injected into
//! the `lowess` crate's execution engine. It enables multi-threaded execution
//! of the local regression fits, significantly speeding up LOWESS smoothing
//! for large datasets by utilizing all available CPU cores.
//!
//! ## Design notes
//!
//! * **Implementation**: Provides a drop-in replacement for the sequential smoothing pass.
//! * **Parallelism**: Uses `rayon` for data-parallel execution across CPU cores.
//! * **Optimization**: Reuses weight buffers per thread to minimize allocations.
//! * **Interpolation**: Supports delta optimization for sparse fitting.
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Parallel Fitting**: Distributes points across CPU cores independently.
//! * **Delta Optimization**: Fits only "anchor" points and interpolates between them.
//! * **Buffer Reuse**: Thread-local scratch buffers to avoid allocation overhead.
//! * **Integration**: Plugs into the `lowess` executor via the `SmoothPassFn` hook.
//!
//! ## Invariants
//!
//! * Input x-values are assumed to be monotonically increasing (sorted).
//! * All buffers have the same length as the input data.
//! * Robustness weights are expected to be in [0, 1].
//! * Window size is at least 1 and at most n.
//!
//! ## Non-goals
//!
//! * This module does not handle the iteration loop (handled by `lowess::executor`).
//! * This module does not validate input data (handled by `validator`).
//! * This module does not sort input data (caller's responsibility).

// Feature-gated imports
#[cfg(feature = "cpu")]
use rayon::prelude::*;

// External dependencies
use num_traits::Float;

// Export dependencies from lowess crate
use lowess::internals::algorithms::regression::{
    LinearRegression, Regression, RegressionContext, ZeroWeightFallback,
};
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::window::Window;

// ============================================================================
// Parallel Smoothing Function
// ============================================================================

/// Perform a single smoothing pass over all points in parallel.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cpu")]
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

/// Compute anchor points for delta optimization using O(log n) binary search.
#[cfg(feature = "cpu")]
fn compute_anchor_points<T: Float>(x: &[T], delta: T) -> Vec<usize> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    let mut anchors = vec![0];
    let mut last_fitted = 0usize;

    while last_fitted < n - 1 {
        let cutpoint = x[last_fitted] + delta;
        let next_idx = x[last_fitted + 1..].partition_point(|&xi| xi <= cutpoint) + last_fitted + 1;

        let x_last = x[last_fitted];
        let mut tie_end = last_fitted;
        for (i, &xi) in x
            .iter()
            .enumerate()
            .take(next_idx.min(n))
            .skip(last_fitted + 1)
        {
            if xi == x_last {
                anchors.push(i);
                tie_end = i;
            } else {
                break;
            }
        }
        if tie_end > last_fitted {
            last_fitted = tie_end;
        }

        let current = usize::max(next_idx.saturating_sub(1), last_fitted + 1).min(n - 1);
        if current <= last_fitted {
            break;
        }

        anchors.push(current);
        last_fitted = current;
    }

    if *anchors.last().unwrap_or(&0) != n - 1 {
        anchors.push(n - 1);
    }

    anchors
}

/// Linearly interpolate between two fitted anchor points.
#[cfg(feature = "cpu")]
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
        y_smooth[(start + 1)..end].fill(avg);
        return;
    }

    let slope = (y1 - y0) / denom;
    for k in (start + 1)..end {
        y_smooth[k] = y0 + (x[k] - x0) * slope;
    }
}

/// Fit all points in parallel (no delta optimization).
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cpu")]
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

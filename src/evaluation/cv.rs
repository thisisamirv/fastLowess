//! Parallel cross-validation for LOWESS bandwidth selection.

// External dependencies
use num_traits::Float;
use rayon::prelude::*;
use std::cmp::Ordering::Equal;
use std::fmt::Debug;

// Export dependencies from lowess crate
use lowess::internals::engine::executor::{LowessConfig, LowessExecutor};
use lowess::internals::evaluation::cv::CVKind;

/// Perform cross-validation to select the best fraction in parallel.
pub fn cv_pass_parallel<T>(
    x: &[T],
    y: &[T],
    fractions: &[T],
    method: CVKind,
    config: &LowessConfig<T>,
) -> (T, Vec<T>)
where
    T: Float + Send + Sync + Debug + 'static,
{
    if fractions.is_empty() {
        return (T::zero(), Vec::new());
    }

    // Parallelize over candidate fractions
    let scores: Vec<T> = fractions
        .par_iter()
        .map(|&frac| {
            // Use the base CV logic for a single fraction
            // This ensures exact consistency with the sequential implementation in 'lowess'
            let (_, s) = method.run(x, y, &[frac], config.seed, |tx, ty, f| {
                let mut fold_config = config.clone();
                fold_config.fraction = Some(f);
                fold_config.cv_fractions = None;

                LowessExecutor::run_with_config(tx, ty, fold_config).smoothed
            });
            s[0]
        })
        .collect();

    let best_idx = scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    (fractions[best_idx], scores)
}

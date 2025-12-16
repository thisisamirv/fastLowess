//! Execution engine for LOWESS smoothing operations.
//!
//! ## Purpose
//!
//! This module provides the core execution engine that orchestrates LOWESS
//! smoothing operations. It handles the iteration loop, robustness weight
//! updates, convergence checking, cross-validation, and variance estimation.
//! The executor is the central component that coordinates all lower-level
//! algorithms to produce smoothed results.
//!
//! ## Design notes
//!
//! * Provides both configuration-based and parameter-based entry points.
//! * Handles cross-validation for automatic fraction selection.
//! * Supports auto-convergence for adaptive iteration counts.
//! * Manages working buffers efficiently to minimize allocations.
//! * Uses delta optimization for performance on dense data.
//! * Separates concerns: fitting, interpolation, robustness, convergence.
//! * Generic over `Float` types to support f32 and f64.
//! * **Adds parallel execution via rayon** (the key difference from lowess crate).
//!
//! ## Key concepts
//!
//! ### Execution Flow
//! 1. Validate and prepare parameters (window size, delta, etc.)
//! 2. Allocate working buffers (y_smooth, weights, residuals)
//! 3. Perform initial smoothing pass (iteration 0)
//! 4. For each robustness iteration:
//!    - Compute residuals
//!    - Update robustness weights
//!    - Re-smooth with combined weights
//!    - Check convergence (if enabled)
//! 5. Optionally compute standard errors
//! 6. Return results
//!
//! ## Visibility
//!
//! This module is an internal implementation detail used by the LOWESS
//! adapters. The `LowessExecutor` struct and `LowessConfig` are used
//! internally and may change without notice.

use lowess::testing::algorithms::interpolation::{interpolate_gap, skip_close_points};
use lowess::testing::algorithms::regression::{
    GLSModel, LinearRegression, Regression, RegressionContext, ZeroWeightFallback,
};
use lowess::testing::algorithms::robustness::RobustnessMethod;
use lowess::testing::evaluation::cv::CVMethod;
use lowess::testing::evaluation::intervals::IntervalMethod;
use lowess::testing::math::kernel::WeightFunction;
use lowess::testing::primitives::window::Window;

use num_traits::Float;
use rayon::prelude::*;
use std::fmt::Debug;
use std::mem;
use std::vec::Vec;

// ============================================================================
// Output Types
// ============================================================================

/// Output from LOWESS execution.
///
/// This is the unified result type returned by [`LowessExecutor::run`] and
/// [`LowessExecutor::run_with_config`]. It contains the smoothed values and
/// optional metadata about the execution.
///
/// # Fields
///
/// * `smoothed` - Smoothed y-values
/// * `std_errors` - Standard errors (if variance estimation was requested)
/// * `iterations` - Number of iterations performed (if auto-convergence was used)
/// * `used_fraction` - Fraction used for smoothing (selected by CV or configured)
/// * `cv_scores` - RMSE scores for each tested fraction (if CV was performed)
#[derive(Debug, Clone)]
pub struct ExecutorOutput<T> {
    /// Smoothed y-values
    pub smoothed: Vec<T>,

    /// Standard errors (if confidence_method was provided)
    pub std_errors: Option<Vec<T>>,

    /// Number of iterations used (if auto-convergence was active)
    pub iterations: Option<usize>,

    /// Fraction used for smoothing (selected by CV or configured)
    pub used_fraction: T,

    /// CV scores for each tested fraction (if CV was performed)
    pub cv_scores: Option<Vec<T>>,
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for LOWESS execution.
///
/// This struct provides a declarative way to configure LOWESS smoothing.
/// It is used by adapters to pass configuration to the executor.
///
/// # Fields
///
/// * `fraction` - Smoothing fraction (0, 1]. If None and cv_fractions is provided, CV selects it
/// * `iterations` - Number of robustness iterations
/// * `delta` - Delta parameter for interpolation optimization
/// * `weight_function` - Kernel weight function
/// * `zero_weight_fallback` - Zero weight fallback policy (0=UseLocalMean, 1=ReturnOriginal, 2=ReturnNone)
/// * `robustness_method` - Robustness weighting method
/// * `cv_fractions` - Fractions to test for cross-validation
/// * `cv_method` - Cross-validation method (k-fold or LOOCV)
/// * `auto_convergence` - Auto-convergence tolerance
/// * `return_variance` - Variance estimation settings
/// * `parallel` - Whether to use parallel execution (fastLowess addition)
#[derive(Debug, Clone)]
pub struct LowessConfig<T> {
    /// Smoothing fraction (0.0 to 1.0).
    /// If None and cv_fractions is provided, selection is performed.
    /// If None and cv_fractions is None, defaults to 0.67.
    pub fraction: Option<T>,

    /// Number of iterations (robustness)
    pub iterations: usize,

    /// Delta parameter for optimization
    pub delta: T,

    /// Weight function (kernel)
    pub weight_function: WeightFunction,

    /// Zero weight fallback policy (u8 flag for now)
    pub zero_weight_fallback: u8,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Fractions to test for cross-validation
    pub cv_fractions: Option<Vec<T>>,

    /// Cross-validation method
    pub cv_method: Option<CVMethod>,

    /// Auto-convergence tolerance
    pub auto_convergence: Option<T>,

    /// Variance estimation settings
    pub return_variance: Option<IntervalMethod<T>>,

    /// Whether to use parallel execution (fastLowess addition)
    pub parallel: bool,
}

impl<T: Float> Default for LowessConfig<T> {
    fn default() -> Self {
        Self {
            fraction: None,
            iterations: 3,
            delta: T::zero(),
            weight_function: WeightFunction::default(),
            zero_weight_fallback: 0,
            robustness_method: RobustnessMethod::default(),
            cv_fractions: None,
            cv_method: None,
            auto_convergence: None,
            return_variance: None,
            parallel: true,
        }
    }
}

// ============================================================================
// Internal Types
// ============================================================================

/// Working buffers for LOWESS iteration.
///
/// Encapsulates all temporary storage needed during the iteration loop.
/// Buffers are allocated once and reused across iterations for efficiency.
struct IterationBuffers<T> {
    /// Current smoothed values
    y_smooth: Vec<T>,

    /// Previous iteration values (for convergence check)
    y_prev: Vec<T>,

    /// Robustness weights
    robustness_weights: Vec<T>,

    /// Residuals buffer
    residuals: Vec<T>,

    /// Kernel weights scratch buffer
    weights: Vec<T>,
}

impl<T: Float> IterationBuffers<T> {
    /// Allocate all working buffers for LOWESS iteration.
    ///
    /// # Parameters
    ///
    /// * `n` - Number of data points
    /// * `use_convergence` - Whether to allocate y_prev buffer for convergence checking
    ///
    /// # Returns
    ///
    /// Initialized buffers with appropriate sizes.
    fn allocate(n: usize, use_convergence: bool) -> Self {
        Self {
            y_smooth: vec![T::zero(); n],
            y_prev: if use_convergence {
                vec![T::zero(); n]
            } else {
                Vec::new()
            },
            robustness_weights: vec![T::one(); n],
            residuals: vec![T::zero(); n],
            weights: vec![T::zero(); n],
        }
    }
}

// ============================================================================
// LowessExecutor
// ============================================================================

/// Unified executor for LOWESS smoothing operations.
///
/// Encapsulates all parameters needed to run LOWESS smoothing and provides
/// convenient methods for different execution modes (basic, with variance,
/// auto-convergence, cross-validation).
///
/// # Fields
///
/// * `fraction` - Smoothing fraction (0, 1]
/// * `iterations` - Number of robustness iterations
/// * `delta` - Delta for interpolation optimization
/// * `weight_function` - Kernel weight function
/// * `zero_weight_fallback` - Zero weight fallback flag
/// * `robustness_method` - Robustness method
/// * `parallel` - Whether to use parallel execution (fastLowess addition)
#[derive(Debug, Clone)]
pub struct LowessExecutor<T: Float> {
    /// Smoothing fraction (0.0 - 1.0)
    pub fraction: T,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Delta for interpolation optimization
    pub delta: T,

    /// Kernel weight function
    pub weight_function: WeightFunction,

    /// Zero weight fallback flag (0=UseLocalMean, 1=ReturnOriginal, 2=ReturnNone)
    pub zero_weight_fallback: u8,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Whether to use parallel execution (fastLowess addition)
    pub parallel: bool,
}

impl<T: Float> Default for LowessExecutor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> LowessExecutor<T> {
    // ========================================================================
    // Constructor and Builder Methods
    // ========================================================================

    /// Create a new executor with default parameters.
    ///
    /// # Defaults
    ///
    /// * fraction: 0.67
    /// * iterations: 3
    /// * delta: 0.0
    /// * weight_function: Tricube
    /// * zero_weight_fallback: 0 (UseLocalMean)
    /// * robustness_method: Bisquare
    /// * parallel: true
    pub fn new() -> Self {
        Self {
            fraction: T::from(0.67).unwrap_or_else(|| T::from(0.5).unwrap()),
            iterations: 3,
            delta: T::zero(),
            weight_function: WeightFunction::Tricube,
            zero_weight_fallback: 0,
            robustness_method: RobustnessMethod::Bisquare,
            parallel: true,
        }
    }

    /// Set the smoothing fraction.
    pub fn fraction(mut self, frac: T) -> Self {
        self.fraction = frac;
        self
    }

    /// Set the number of robustness iterations.
    pub fn iterations(mut self, niter: usize) -> Self {
        self.iterations = niter;
        self
    }

    /// Set the delta parameter for interpolation optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.delta = delta;
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.weight_function = wf;
        self
    }

    /// Set the zero weight fallback policy.
    pub fn zero_weight_fallback(mut self, flag: u8) -> Self {
        self.zero_weight_fallback = flag;
        self
    }

    /// Set the robustness method.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.robustness_method = method;
        self
    }

    /// Set parallel execution mode.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    // ========================================================================
    // Specialized Fitting Functions
    // ========================================================================

    /// Fit the first point and initialize the smoothing window.
    ///
    /// # Parameters
    ///
    /// * `x` - Independent variable values (sorted)
    /// * `y` - Dependent variable values
    /// * `window_size` - Size of the local window
    /// * `use_robustness` - Whether to use robustness weights
    /// * `robustness_weights` - Robustness weights array
    /// * `weights` - Kernel weights scratch buffer
    /// * `weight_function` - Kernel function
    /// * `zero_weight_fallback` - Zero weight fallback policy
    /// * `fitter` - Regression fitter
    /// * `y_smooth` - Output buffer for smoothed values
    ///
    /// # Returns
    ///
    /// Initial window positioned at index 0.
    #[allow(clippy::too_many_arguments)]
    fn fit_first_point<Fitter>(
        x: &[T],
        y: &[T],
        window_size: usize,
        use_robustness: bool,
        robustness_weights: &[T],
        weights: &mut [T],
        weight_function: WeightFunction,
        zero_weight_fallback: ZeroWeightFallback,
        fitter: &Fitter,
        y_smooth: &mut [T],
    ) -> Window
    where
        Fitter: Regression<T> + ?Sized,
    {
        let n = x.len();
        let mut window = Window::initialize(0, window_size, n);
        window.recenter(x, 0, n);

        let ctx = RegressionContext {
            x,
            y,
            idx: 0,
            window,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
        };

        y_smooth[0] = fitter.fit(ctx).unwrap_or_else(|| y[0]);
        window
    }

    /// Main fitting loop: iterate through remaining points with delta-skipping and interpolation.
    ///
    /// # Parameters
    ///
    /// * `x` - Independent variable values (sorted)
    /// * `y` - Dependent variable values
    /// * `delta` - Delta threshold for point skipping
    /// * `use_robustness` - Whether to use robustness weights
    /// * `robustness_weights` - Robustness weights array
    /// * `weights` - Kernel weights scratch buffer
    /// * `weight_function` - Kernel function
    /// * `zero_weight_fallback` - Zero weight fallback policy
    /// * `fitter` - Regression fitter
    /// * `y_smooth` - Output buffer for smoothed values
    /// * `window` - Initial window from first point
    ///
    /// # Algorithm
    ///
    /// 1. Determine next point to fit using delta optimization
    /// 2. Update window to center around current point
    /// 3. Fit current point
    /// 4. Linearly interpolate between last fitted and current
    /// 5. Repeat until all points processed
    #[allow(clippy::too_many_arguments)]
    fn fit_and_interpolate_remaining<Fitter>(
        x: &[T],
        y: &[T],
        delta: T,
        use_robustness: bool,
        robustness_weights: &[T],
        weights: &mut [T],
        weight_function: WeightFunction,
        zero_weight_fallback: ZeroWeightFallback,
        fitter: &Fitter,
        y_smooth: &mut [T],
        mut window: Window,
    ) where
        Fitter: Regression<T> + ?Sized,
    {
        let n = x.len();
        let mut last_fitted = 0usize;

        loop {
            // Determine next point to fit based on delta optimization
            let current = skip_close_points(x, y_smooth, delta, &mut last_fitted, n, None);

            // If skip_close_points didn't advance, we are done
            if current <= last_fitted {
                break;
            }

            // Update window to be centered around current point
            window.recenter(x, current, n);

            // Fit current point
            let ctx = RegressionContext {
                x,
                y,
                idx: current,
                window,
                use_robustness,
                robustness_weights,
                weights,
                weight_function,
                zero_weight_fallback,
            };

            y_smooth[current] = fitter.fit(ctx).unwrap_or_else(|| y[current]);

            // Linearly interpolate between last fitted and current
            interpolate_gap(x, y_smooth, last_fitted, current);
            last_fitted = current;
        }

        // Final interpolation to the end if necessary
        if last_fitted < n.saturating_sub(1) {
            interpolate_gap(x, y_smooth, last_fitted, n - 1);
        }
    }

    /// Perform a single smoothing pass over all points.
    ///
    /// Combines first-point fitting with the main interpolation loop.
    ///
    /// # Parameters
    ///
    /// * `x` - Independent variable values (sorted)
    /// * `y` - Dependent variable values
    /// * `window_size` - Size of the local window
    /// * `delta` - Delta threshold for point skipping
    /// * `use_robustness` - Whether to use robustness weights
    /// * `robustness_weights` - Robustness weights array
    /// * `y_smooth` - Output buffer for smoothed values
    /// * `weight_function` - Kernel function
    /// * `weights` - Kernel weights scratch buffer
    /// * `zero_weight_flag` - Zero weight fallback flag
    /// * `fitter` - Regression fitter
    #[allow(clippy::too_many_arguments)]
    fn smooth_pass<Fitter>(
        x: &[T],
        y: &[T],
        window_size: usize,
        delta: T,
        use_robustness: bool,
        robustness_weights: &[T],
        y_smooth: &mut [T],
        weight_function: WeightFunction,
        weights: &mut [T],
        zero_weight_flag: u8,
        fitter: &Fitter,
    ) where
        Fitter: Regression<T> + ?Sized,
    {
        let n = x.len();
        if n == 0 {
            return;
        }

        let zero_weight_fallback = ZeroWeightFallback::from_u8(zero_weight_flag);

        // Fit first point
        let window = Self::fit_first_point(
            x,
            y,
            window_size,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
            fitter,
            y_smooth,
        );

        // Fit remaining points with interpolation
        Self::fit_and_interpolate_remaining(
            x,
            y,
            delta,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
            fitter,
            y_smooth,
            window,
        );
    }

    /// Perform a single smoothing pass over all points (parallel).
    #[allow(clippy::too_many_arguments)]
    fn smooth_pass_parallel(
        x: &[T],
        y: &[T],
        window_size: usize,
        use_robustness: bool,
        robustness_weights: &[T],
        y_smooth: &mut [T],
        weight_function: WeightFunction,
        zero_weight_flag: u8,
    ) where
        T: Send + Sync,
    {
        let n = x.len();
        if n == 0 {
            return;
        }

        let zero_weight_fallback = ZeroWeightFallback::from_u8(zero_weight_flag);
        let fitter = LinearRegression;

        // Parallel fitting of all points
        let results: Vec<T> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut weights = vec![T::zero(); n];
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
                    weights: &mut weights,
                    weight_function,
                    zero_weight_fallback,
                };

                fitter.fit(ctx).unwrap_or(y[i])
            })
            .collect();

        y_smooth.copy_from_slice(&results);
    }

    // ========================================================================
    // Iteration Control Functions
    // ========================================================================

    /// Check convergence between current and previous smoothed values.
    ///
    /// # Parameters
    ///
    /// * `y_smooth` - Current smoothed values
    /// * `y_prev` - Previous iteration smoothed values
    /// * `tolerance` - Convergence tolerance
    ///
    /// # Returns
    ///
    /// `true` if max absolute change is within tolerance.
    fn check_convergence(y_smooth: &[T], y_prev: &[T], tolerance: T) -> bool {
        let max_change = y_smooth
            .iter()
            .zip(y_prev.iter())
            .fold(T::zero(), |maxv, (&current, &previous)| {
                T::max(maxv, (current - previous).abs())
            });

        max_change <= tolerance
    }

    /// Update robustness weights based on residuals.
    ///
    /// # Parameters
    ///
    /// * `y` - Original y-values
    /// * `y_smooth` - Smoothed y-values
    /// * `residuals` - Residuals buffer (will be updated)
    /// * `robustness_weights` - Robustness weights buffer (will be updated)
    /// * `robustness_updater` - Robustness method
    ///
    /// # Algorithm
    ///
    /// 1. Compute residuals: r_i = y_i - ŷ_i
    /// 2. Apply robustness method to compute weights from residuals
    fn update_robustness_weights(
        y: &[T],
        y_smooth: &[T],
        residuals: &mut [T],
        robustness_weights: &mut [T],
        robustness_updater: &RobustnessMethod,
    ) {
        for i in 0..y.len() {
            residuals[i] = y[i] - y_smooth[i];
        }
        robustness_updater.apply_robustness_weights(residuals, robustness_weights);
    }

    /// Compute standard errors using the provided interval method.
    ///
    /// # Parameters
    ///
    /// * `x` - Independent variable values (sorted)
    /// * `y` - Dependent variable values
    /// * `y_smooth` - Smoothed y-values
    /// * `window_size` - Size of the local window
    /// * `robustness_weights` - Robustness weights array
    /// * `weight_function` - Kernel function
    /// * `interval_method` - Interval method for SE computation
    ///
    /// # Returns
    ///
    /// Vector of standard errors.
    fn compute_std_errors(
        x: &[T],
        y: &[T],
        y_smooth: &[T],
        window_size: usize,
        robustness_weights: &[T],
        weight_function: WeightFunction,
        interval_method: &IntervalMethod<T>,
    ) -> Vec<T> {
        let n = x.len();
        let mut se = vec![T::zero(); n];
        interval_method.compute_window_se(
            x,
            y,
            y_smooth,
            window_size,
            robustness_weights,
            &mut se,
            &|t| weight_function.compute_weight(t),
        );
        se
    }

    /// Perform the full LOWESS iteration loop.
    ///
    /// Handles robustness iterations, convergence checking, and optional SE computation.
    ///
    /// # Parameters
    ///
    /// * `x` - Independent variable values (sorted)
    /// * `y` - Dependent variable values
    /// * `window_size` - Size of the local window
    /// * `niter` - Maximum number of robustness iterations
    /// * `delta` - Delta threshold for point skipping
    /// * `weight_function` - Kernel function
    /// * `zero_weight_flag` - Zero weight fallback flag
    /// * `fitter` - Regression fitter
    /// * `robustness_updater` - Robustness method
    /// * `interval_method` - Optional interval method for SE computation
    /// * `convergence_tolerance` - Optional convergence tolerance
    /// * `parallel` - Whether to use parallel processing
    ///
    /// # Returns
    ///
    /// Tuple of (smoothed values, standard errors, iterations performed).
    #[allow(clippy::too_many_arguments)]
    fn iteration_loop<Fitter>(
        &self,
        x: &[T],
        y: &[T],
        window_size: usize,
        niter: usize,
        delta: T,
        weight_function: WeightFunction,
        zero_weight_flag: u8,
        fitter: &Fitter,
        robustness_updater: &RobustnessMethod,
        interval_method: Option<&IntervalMethod<T>>,
        convergence_tolerance: Option<T>,
        parallel: bool,
    ) -> (Vec<T>, Option<Vec<T>>, usize)
    where
        Fitter: Regression<T> + ?Sized,
        T: Send + Sync,
    {
        let n = x.len();
        let mut buffers = IterationBuffers::allocate(n, convergence_tolerance.is_some());
        let mut iterations_performed = 0;

        // Copy initial y values to y_smooth
        buffers.y_smooth.copy_from_slice(y);

        // Smoothing iterations with robustness updates
        for iter in 0..=niter {
            iterations_performed = iter;

            // Swap buffers if checking convergence (save previous state)
            if convergence_tolerance.is_some() && iter > 0 {
                mem::swap(&mut buffers.y_smooth, &mut buffers.y_prev);
            }

            if parallel {
                // Perform parallel smoothing pass
                Self::smooth_pass_parallel(
                    x,
                    y,
                    window_size,
                    iter > 0, // use_robustness only after first iteration
                    &buffers.robustness_weights,
                    &mut buffers.y_smooth,
                    weight_function,
                    zero_weight_flag,
                );
            } else {
                // Perform sequential smoothing pass
                Self::smooth_pass(
                    x,
                    y,
                    window_size,
                    delta,
                    iter > 0, // use_robustness only after first iteration
                    &buffers.robustness_weights,
                    &mut buffers.y_smooth,
                    weight_function,
                    &mut buffers.weights,
                    zero_weight_flag,
                    fitter,
                );
            }

            // Check convergence if tolerance is provided (skip on first iteration)
            if let Some(tol) = convergence_tolerance {
                if iter > 0 && Self::check_convergence(&buffers.y_smooth, &buffers.y_prev, tol) {
                    break;
                }
            }

            // Update robustness weights for next iteration (skip last)
            if iter < niter {
                Self::update_robustness_weights(
                    y,
                    &buffers.y_smooth,
                    &mut buffers.residuals,
                    &mut buffers.robustness_weights,
                    robustness_updater,
                );
            }
        }

        // Compute standard errors if requested
        let std_errors = interval_method.map(|im| {
            Self::compute_std_errors(
                x,
                y,
                &buffers.y_smooth,
                window_size,
                &buffers.robustness_weights,
                weight_function,
                im,
            )
        });

        (buffers.y_smooth, std_errors, iterations_performed)
    }

    // ========================================================================
    // Main Entry Points
    // ========================================================================

    /// Run LOWESS smoothing with a configuration struct.
    ///
    /// This is the primary entry point for adapters. It handles:
    /// * Cross-validation (if configured)
    /// * Auto-convergence (if configured)
    /// * Variance estimation (if configured)
    /// * Parallel vs sequential execution
    ///
    /// # Parameters
    ///
    /// * `x` - Independent variable values (must be sorted)
    /// * `y` - Dependent variable values
    /// * `config` - Configuration struct
    ///
    /// # Returns
    ///
    /// [`ExecutorOutput`] containing smoothed values and metadata.
    pub fn run_with_config(x: &[T], y: &[T], config: LowessConfig<T>) -> ExecutorOutput<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let default_frac = T::from(0.67).unwrap_or(T::from(0.5).unwrap());
        let fraction = config.fraction.unwrap_or(default_frac);

        let executor = LowessExecutor::new()
            .fraction(fraction)
            .iterations(config.iterations)
            .delta(config.delta)
            .weight_function(config.weight_function)
            .zero_weight_fallback(config.zero_weight_fallback)
            .robustness_method(config.robustness_method)
            .parallel(config.parallel);

        // Handle cross-validation if configured
        if let Some(cv_fracs) = config.cv_fractions {
            if cv_fracs.is_empty() {
                // Fallback to standard run with default fraction
                return executor.run(x, y, Some(fraction), None, None, None);
            }

            let cv_method = config.cv_method.unwrap_or(CVMethod::KFold(5));

            // Run CV to find best fraction
            let (best_frac, scores) = cv_method.run(x, y, &cv_fracs, |tx, ty, f| {
                executor.run(tx, ty, Some(f), None, None, None).smoothed
            });

            // Run final pass with best fraction
            let mut output = executor.run(
                x,
                y,
                Some(best_frac),
                Some(config.iterations),
                config.auto_convergence,
                config.return_variance.as_ref(),
            );
            output.cv_scores = Some(scores);
            output.used_fraction = best_frac;
            output
        } else {
            // Direct run (no CV)
            executor.run(
                x,
                y,
                config.fraction,
                Some(config.iterations),
                config.auto_convergence,
                config.return_variance.as_ref(),
            )
        }
    }

    /// Run LOWESS smoothing with explicit parameters.
    ///
    /// This is a lower-level entry point with direct parameter control.
    ///
    /// # Parameters
    ///
    /// * `x` - Independent variable values (must be sorted)
    /// * `y` - Dependent variable values
    /// * `fraction` - Optional fraction override (defaults to configured)
    /// * `max_iter` - Optional maximum iterations override (defaults to configured)
    /// * `tolerance` - Optional auto-convergence tolerance
    /// * `confidence_method` - Optional confidence interval method
    ///
    /// # Returns
    ///
    /// [`ExecutorOutput`] containing smoothed values and metadata.
    ///
    /// # Special cases
    ///
    /// * **Too few points** (n < 2): Returns original y-values
    /// * **Global regression** (fraction ≥ 1.0): Uses OLS on entire dataset
    fn run(
        &self,
        x: &[T],
        y: &[T],
        fraction: Option<T>,
        max_iter: Option<usize>,
        tolerance: Option<T>,
        confidence_method: Option<&IntervalMethod<T>>,
    ) -> ExecutorOutput<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let n = x.len();
        let eff_fraction = fraction.unwrap_or(self.fraction);

        // Edge case: too few points
        if n < 2 {
            return ExecutorOutput {
                smoothed: y.to_vec(),
                std_errors: if confidence_method.is_some() {
                    Some(vec![T::zero(); n])
                } else {
                    None
                },
                iterations: None,
                used_fraction: eff_fraction,
                cv_scores: None,
            };
        }

        // Handle global regression (fraction >= 1.0)
        if eff_fraction >= T::one() {
            let smoothed = GLSModel::global_ols(x, y);
            return ExecutorOutput {
                smoothed,
                std_errors: if confidence_method.is_some() {
                    Some(vec![T::zero(); n])
                } else {
                    None
                },
                iterations: None,
                used_fraction: eff_fraction,
                cv_scores: None,
            };
        }

        // Calculate window size and prepare fitter
        let window_size = Window::calculate_span(n, eff_fraction);
        let fitter = LinearRegression;
        let target_iterations = max_iter.unwrap_or(self.iterations);

        // Run the iteration loop
        let (y_smooth, se, iters_used) = self.iteration_loop(
            x,
            y,
            window_size,
            target_iterations,
            self.delta,
            self.weight_function,
            self.zero_weight_fallback,
            &fitter,
            &self.robustness_method,
            confidence_method,
            tolerance,
            self.parallel,
        );

        ExecutorOutput {
            smoothed: y_smooth,
            std_errors: se,
            iterations: if tolerance.is_some() {
                Some(iters_used)
            } else {
                None
            },
            used_fraction: eff_fraction,
            cv_scores: None,
        }
    }
}

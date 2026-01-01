#!/usr/bin/env Rscript
# R LOWESS Validation Script
# Generates reference outputs using stats::lowess
# Strictly limited to scenarios supported by the original lowess algorithm (local linear, degree=1)

library(jsonlite)

OUTPUT_DIR <- "output/r/"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

run_scenario <- function(name, x, y, frac, iter, delta = NULL, notes = "", ...) {
  cat(sprintf("Running scenario: %s\n", name))

  # Config
  # stats::lowess parameters:
  # x, y: data
  # f: smoother span (fraction)
  # iter: number of robustness iterations
  # delta: point skipping threshold
  
  args <- list(x = x, y = y, f = frac, iter = iter)
  
  # Handle delta logic
  # If delta is explicitly provided, use it.
  # Otherwise stats::lowess uses default 0.01 * range
  if (!is.null(delta)) {
     args$delta <- delta
  }
  
  fit <- do.call(stats::lowess, args)
  
  # lowess output is a list with $x (sorted) and $y (fitted)
  fitted <- fit$y

  # Create output structure
  data <- list(
    name = name,
    notes = notes,
    input = list(
      x = x,
      y = y
    ),
    params = list(
      fraction = frac,
      iterations = iter,
      delta = args$delta,
      extra = list(...)
    ),
    result = list(
      fitted = fitted
    )
  )

  # Save to JSON
  path <- file.path(OUTPUT_DIR, paste0(name, ".json"))
  write_json(data, path, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null")
}

generate_data <- function(n = 100, kind = "linear", noise = 0.0,
                          range_min = 0.0, range_max = 1.0, outlier_ratio = 0.0) {
  set.seed(42)

  x <- seq(range_min, range_max, length.out = n)

  if (kind == "linear") {
    y <- 2 * x + 1
  } else if (kind == "sine") {
    y <- sin(4 * x)
  } else if (kind == "step") {
    y <- ifelse(x < (range_min + range_max) / 2, 0.0, 1.0)
  } else if (kind == "constant") {
    y <- rep(5.0, n)
  } else {
    y <- x
  }

  if (noise > 0) {
    y <- y + rnorm(n, 0, noise)
  }

  if (outlier_ratio > 0) {
    n_out <- as.integer(n * outlier_ratio)
    indices <- sample(n, n_out, replace = FALSE)
    y[indices] <- y[indices] + 10.0
  }

  list(x = x, y = y)
}

main <- function() {
  # 01. Tiny Linear
  data <- generate_data(n = 10, kind = "linear")
  run_scenario("01_tiny_linear", data$x, data$y, frac = 0.8, iter = 0)

  # 02. Sine Standard
  data <- generate_data(n = 100, kind = "sine", noise = 0.1)
  run_scenario("02_sine_standard", data$x, data$y, frac = 0.3, iter = 0)

  # 03. Sine Robust
  # iter=3 is default, using 4 to match previous testing
  data <- generate_data(n = 100, kind = "sine", outlier_ratio = 0.05)
  run_scenario("03_sine_robust", data$x, data$y, frac = 0.3, iter = 4)

  # 04. Large scale
  data <- generate_data(n = 500, kind = "sine")
  run_scenario("04_large_scale", data$x, data$y, frac = 0.1, iter = 0)

  # 05. High Smoothness
  data <- generate_data(n = 100, kind = "linear", noise = 0.5)
  run_scenario("05_high_smoothness", data$x, data$y, frac = 0.9, iter = 0)

  # 06. Low Smoothness (Direct surface via delta=0)
  data <- generate_data(n = 100, kind = "sine")
  run_scenario("06_low_smoothness", data$x, data$y,
    frac = 0.05, iter = 0,
    delta = 0.0 # equivalent to surface="direct"
  )

  # 07. Constant Function
  data <- generate_data(n = 50, kind = "constant")
  run_scenario("07_constant", data$x, data$y, frac = 0.5, iter = 0)

  # 08. Step Function
  data <- generate_data(n = 100, kind = "step")
  run_scenario("08_step_func", data$x, data$y, frac = 0.4, iter = 0)

  # 09. End-effects Left
  data <- generate_data(n = 50, kind = "linear", noise = 0.1)
  run_scenario("09_end_effects_left", data$x, data$y,
    frac = 0.3, iter = 0,
    notes = "Check left boundary"
  )

  # 10. End-effects Right
  run_scenario("10_end_effects_right", data$x, data$y,
    frac = 0.3, iter = 0,
    notes = "Check right boundary"
  )

  # 11. Sparse Data
  data <- generate_data(n = 20, range_max = 100.0, kind = "linear", noise = 1.0)
  run_scenario("11_sparse_data", data$x, data$y, frac = 0.6, iter = 0)

  # 12. Dense Data
  data <- generate_data(n = 1000, kind = "sine", noise = 0.1)
  run_scenario("12_dense_data", data$x, data$y,
    frac = 0.01, iter = 0,
    delta = 0.0 # surface="direct"
  )

  # 13. Iter 2 Check
  data <- generate_data(n = 100, kind = "sine", outlier_ratio = 0.05)
  run_scenario("13_iter_2", data$x, data$y, frac = 0.4, iter = 2)

  # 14. Interpolate Exact
  data <- generate_data(n = 50, kind = "linear")
  run_scenario("14_interpolate_exact", data$x, data$y, frac = 0.5, iter = 0)

  # 15. Zero Variance
  data <- generate_data(n = 10, kind = "constant")
  run_scenario("15_zero_variance", data$x, data$y, frac = 0.5, iter = 0)

  cat("\nAll R stats::lowess validation scenarios completed!\n")
}

main()

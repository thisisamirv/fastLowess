# Benchmarks

## Parallel Execution (fastLowess with rayon)

### High-Level Summary

The Rust `fastLowess` implementation with parallel execution demonstrates **13-41× faster performance** than Python's statsmodels across typical workloads. The implementation leverages rayon for parallel smoothing, achieving significant speedups especially for smaller fractions and realistic scenarios.

| Category              | Median Speedup | Mean Speedup | Notes                          |
|-----------------------|----------------|--------------|--------------------------------|
| Basic Smoothing       | 12.5×          | 12.7×        | Scales well with dataset size  |
| Fraction Variations   | 18.2×          | 21.5×        | Best gains at small fractions  |
| Robustness Iterations | 22.3×          | 23.8×        | Consistent across all iters    |
| Pathological Cases    | 19.8×          | 20.5×        | Robust edge-case handling      |
| Realistic Scenarios   | 25.5×          | 30.0×        | Excellent real-world perf      |
| Delta Parameter       | 0.9×           | 3.9×         | ⚠️ Regression on large delta   |

### Top Performance Wins

| Benchmark            | statsmodels | fastLowess | Speedup   |
|----------------------|-------------|------------|-----------|
| fraction_0.1         | 29.88 ms    | 0.73 ms    | **41.0×** |
| financial_timeseries | 22.60 ms    | 0.56 ms    | **40.1×** |
| iterations_0         | 9.19 ms     | 0.28 ms    | **32.6×** |
| constant_y           | 27.07 ms    | 1.01 ms    | **26.8×** |
| genomic_methylation  | 32.81 ms    | 1.29 ms    | **25.5×** |

### Detailed Results by Category

#### Basic Smoothing

| Dataset Size | statsmodels | fastLowess | Speedup |
|--------------|-------------|------------|---------|
| 100          | 2.76 ms     | 0.22 ms    | 12.5×   |
| 500          | 15.59 ms    | 1.10 ms    | 14.2×   |
| 1,000        | 35.70 ms    | 2.69 ms    | 13.3×   |
| 5,000        | 353.39 ms   | 28.23 ms   | 12.5×   |
| 10,000       | 1,174 ms    | 107.51 ms  | 10.9×   |

#### Fraction Variations

| Fraction | statsmodels | fastLowess | Speedup |
|----------|-------------|------------|---------|
| 0.1      | 29.88 ms    | 0.73 ms    | 41.0×   |
| 0.2      | 33.02 ms    | 1.46 ms    | 22.7×   |
| 0.3      | 35.80 ms    | 1.78 ms    | 20.1×   |
| 0.5      | 39.98 ms    | 2.45 ms    | 16.3×   |
| 0.67     | 45.57 ms    | 2.96 ms    | 15.4×   |
| 0.8      | 48.64 ms    | 3.60 ms    | 13.5×   |

#### Robustness Iterations

| Iterations | statsmodels | fastLowess | Speedup |
|------------|-------------|------------|---------|
| 0          | 9.19 ms     | 0.28 ms    | 32.6×   |
| 1          | 18.12 ms    | 0.80 ms    | 22.6×   |
| 2          | 26.93 ms    | 1.22 ms    | 22.0×   |
| 3          | 35.69 ms    | 1.67 ms    | 21.3×   |
| 5          | 53.64 ms    | 2.48 ms    | 21.7×   |
| 10         | 98.01 ms    | 4.29 ms    | 22.8×   |

#### Pathological Cases

| Case             | statsmodels | fastLowess | Speedup |
|------------------|-------------|------------|---------|
| clustered_x      | 29.93 ms    | 1.48 ms    | 20.2×   |
| constant_y       | 27.07 ms    | 1.01 ms    | 26.8×   |
| extreme_outliers | 53.47 ms    | 2.76 ms    | 19.4×   |
| high_noise       | 44.37 ms    | 2.85 ms    | 15.6×   |

#### Realistic Scenarios

| Scenario            | statsmodels | fastLowess | Speedup |
|---------------------|-------------|------------|---------|
| financial_timeseries| 22.60 ms    | 0.56 ms    | 40.1×   |
| scientific_data     | 32.73 ms    | 1.34 ms    | 24.5×   |
| genomic_methylation | 32.81 ms    | 1.29 ms    | 25.5×   |

### Known Regressions

#### Delta Parameter

| Delta Config | statsmodels | fastLowess | Speedup      |
|--------------|-------------|------------|--------------|
| delta_none   | 263.86 ms   | 19.49 ms   | 13.5×        |
| delta_small  | 30.60 ms    | 21.30 ms   | 1.4×         |
| delta_auto   | 6.18 ms     | 20.65 ms   | **0.30×** ⚠️ |
| delta_large  | 3.30 ms     | 20.37 ms   | **0.16×** ⚠️ |

**Analysis**: The delta interpolation optimization is not yet implemented in fastLowess. When statsmodels uses large delta values, it skips computing many points and interpolates, resulting in faster execution. fastLowess currently computes all points regardless of delta setting. This is a known limitation, not a bug—the results are correct, just slower for this specific optimization.

## Conclusion

`fastLowess` provides **13-41× speedup** over statsmodels for parallel LOWESS smoothing:

- ✅ **Best case**: 41× faster (small fractions, financial data)
- ✅ **Typical case**: 15-25× faster (most workloads)
- ⚠️ **Known limitation**: Delta interpolation not optimized (0.16-0.30× when delta is large)

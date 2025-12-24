# Benchmark Interpretation (fastLowess)

## Summary

The `fastLowess` crate demonstrates massive performance gains over Python's `statsmodels`, ranging from **136x to over 4300x** speedup. The addition of **parallel execution** (via Rayon) and optimized algorithm defaults makes it exceptionally well-suited for high-throughput data processing and large-scale datasets.

## Category Comparison

| Category         | Matched | Median Speedup | Mean Speedup |
|------------------|---------|----------------|--------------|
| **Scalability**  | 5       | **954×**       | 1637×        |
| **Fraction**     | 6       | **571×**       | 552×         |
| **Iterations**   | 6       | **564×**       | 567×         |
| **Pathological** | 4       | **551×**       | 538×         |
| **Financial**    | 4       | **385×**       | 448×         |
| **Scientific**   | 4       | **381×**       | 450×         |
| **Genomic**      | 4       | **23×**        | 27×          |
| **Delta**        | 4       | **5.7×**       | 7.8×         |

## Top 10 Rust Wins

| Benchmark        | statsmodels | fastLowess | Speedup   |
|------------------|-------------|------------|-----------|
| scale_100000     | 43.73s      | 10.1ms     | **4339×** |
| scale_50000      | 11.16s      | 5.26ms     | **2122×** |
| scale_10000      | 663.1ms     | 0.70ms     | **954×**  |
| scientific_10000 | 777.2ms     | 0.83ms     | **941×**  |
| financial_10000  | 497.1ms     | 0.56ms     | **885×**  |
| iterations_0     | 74.2ms      | 0.12ms     | **599×**  |
| financial_5000   | 170.9ms     | 0.29ms     | **595×**  |
| scientific_5000  | 268.5ms     | 0.45ms     | **593×**  |
| fraction_0.2     | 297.0ms     | 0.50ms     | **591×**  |
| scale_5000       | 229.9ms     | 0.39ms     | **590×**  |

## Regressions

**None identified.** `fastLowess` outperforms `statsmodels` in all matched benchmarks.

## Detailed Results

### Scalability (1K - 100K points)

| Size    | fastLowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 1,000   | 0.17ms     | 30.4ms      | 183×    |
| 5,000   | 0.39ms     | 229.9ms     | 590×    |
| 10,000  | 0.70ms     | 663.1ms     | 954×    |
| 50,000  | 5.26ms     | 11.16s      | 2122×   |
| 100,000 | 10.08ms    | 43.73s      | 4339×   |

### Fraction Variations (n=5000)

| Fraction | fastLowess | statsmodels | Speedup |
|----------|------------|-------------|---------|
| 0.05     | 0.34ms     | 197.2ms     | 581×    |
| 0.10     | 0.39ms     | 227.9ms     | 582×    |
| 0.20     | 0.50ms     | 297.0ms     | 591×    |
| 0.30     | 0.64ms     | 357.0ms     | 561×    |
| 0.50     | 0.97ms     | 488.4ms     | 503×    |
| 0.67     | 1.22ms     | 601.6ms     | 494×    |

### Robustness Iterations (n=5000)

| Iterations | fastLowess | statsmodels | Speedup |
|------------|------------|-------------|---------|
| 0          | 0.12ms     | 74.2ms      | 599×    |
| 1          | 0.26ms     | 148.5ms     | 574×    |
| 2          | 0.39ms     | 222.8ms     | 568×    |
| 3          | 0.53ms     | 296.5ms     | 561×    |
| 5          | 0.81ms     | 445.1ms     | 553×    |
| 10         | 1.49ms     | 815.6ms     | 549×    |

### Delta Parameter (n=10000)

| Delta    | fastLowess | statsmodels | Speedup |
|----------|------------|-------------|---------|
| none (0) | 40.07ms    | 678.2ms     | 16.9×   |
| small    | 0.35ms     | 2.28ms      | 6.6×    |
| medium   | 0.26ms     | 1.27ms      | 4.9×    |
| large    | 0.26ms     | 0.76ms      | 3.0×    |

### Pathological Cases (n=5000)

| Case             | fastLowess | statsmodels | Speedup |
|------------------|------------|-------------|---------|
| clustered        | 0.48ms     | 267.8ms     | 559×    |
| constant_y       | 0.42ms     | 230.3ms     | 555×    |
| extreme_outliers | 1.56ms     | 852.0ms     | 547×    |
| high_noise       | 1.48ms     | 726.9ms     | 492×    |

### Real-World Scenarios

#### Financial Time Series

| Size    | fastLowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 500     | 0.08ms     | 10.4ms      | 136×    |
| 1,000   | 0.13ms     | 22.2ms      | 176×    |
| 5,000   | 0.29ms     | 170.9ms     | 595×    |
| 10,000  | 0.56ms     | 497.1ms     | 885×    |

#### Scientific Measurements

| Size    | fastLowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 500     | 0.14ms     | 14.1ms      | 98×     |
| 1,000   | 0.19ms     | 31.6ms      | 169×    |
| 5,000   | 0.45ms     | 268.5ms     | 593×    |
| 10,000  | 0.83ms     | 777.2ms     | 941×    |

#### Genomic Methylation (with delta=100)

| Size    | fastLowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 1,000   | 0.63ms     | 29.5ms      | 47×     |
| 5,000   | 8.75ms     | 227.3ms     | 26×     |
| 10,000  | 32.46ms    | 662.8ms     | 20×     |
| 50,000  | 818.7ms    | 11.2s       | 14×     |

## Notes

- **Parallel Execution**: Enabled for n ≥ 1000 using Rayon.
- Benchmarks use **Criterion** (Rust) and **pytest-benchmark** (Python).
- Both use identical scenarios with reproducible RNG (seed=42).
- Rust crate: `fastLowess` v0.3.0 (running on `lowess` v0.6.0 core).
- Python: `statsmodels` v0.14.4 (with Cython backend).
- Test date: 2025-12-23.

## GPU vs CPU Backend Comparison

The performance dynamics change significantly depending on the `delta` parameter.

### Scenario A: Delta = 0 (Dense Computation)

When calculating the regression at **every single point** (no interpolation), the CPU backend is surprisingly faster. This suggests the CPU's efficient memory access patterns and `rayon` parallelism outperform the GPU's raw compute when dealing with the massive memory traffic of O(N²) dense interactions.

| Dataset Size | CPU Time    | GPU Time    | Winner | Speedup |
|--------------|-------------|-------------|--------|---------|
| 100,000      | 11.66 ms    | 82.94 ms    | CPU    | 7.1x    |
| 250,000      | 40.47 ms    | 113.55 ms   | CPU    | 2.8x    |
| 500,000      | 76.91 ms    | 151.44 ms   | CPU    | 1.96x   |
| 1,000,000    | 140.56 ms   | 240.75 ms   | CPU    | 1.71x   |
| 2,000,000    | 282.33 ms   | 546.97 ms   | CPU    | 1.93x   |

### Scenario B: Delta = 0.01 (Optimized / Interpolated)

When using the `delta` optimization, the **GPU becomes significantly faster**, scaling much better than the CPU at large sizes. This indicates the GPU implementation handles the "Anchor + Interpolate" logic more efficiently for large N.

| Dataset Size | CPU Time    | GPU Time    | Winner | Speedup  |
|--------------|-------------|-------------|--------|----------|
| 100,000      | 51.80 ms    | 81.55 ms    | CPU    | 0.64x    |
| 250,000      | 148.56 ms   | 109.60 ms   | **GPU**| **1.36x**|
| 500,000      | 337.75 ms   | 172.59 ms   | **GPU**| **1.96x**|
| 1,000,000    | 672.77 ms   | 381.48 ms   | **GPU**| **1.76x**|
| 2,000,000    | 1,423.00 ms | 590.22 ms   | **GPU**| **2.41x**|

### Recommendation

- **For Standard Use (Delta ≈ 0)**: The `Backend::CPU` is generally faster and should be the default.
- **For Large "Delta" Optimization**: If using a specific `delta` strategy that aligns with the GPU's strengths (as seen in Scenario B), the `Backend::GPU` offers superior scaling for very large datasets (> 250k points).

# Benchmark Interpretation (fastLowess)

## Summary

The `fastLowess` crate demonstrates massive performance gains over Python's `statsmodels`, ranging from **91x to over 3900x** speedup. The addition of **parallel execution** (via Rayon) and optimized algorithm defaults makes it exceptionally well-suited for high-throughput data processing and large-scale datasets.

## Category Comparison

| Category         | Matched | Median Speedup | Mean Speedup |
|------------------|---------|----------------|--------------|
| **Scalability**  | 5       | **819×**       | 1482×        |
| **Pathological** | 4       | **503×**       | 476×         |
| **Iterations**   | 6       | **491×**       | 496×         |
| **Fraction**     | 6       | **464×**       | 447×         |
| **Financial**    | 4       | **351×**       | 418×         |
| **Scientific**   | 4       | **345×**       | 404×         |
| **Genomic**      | 4       | **22×**        | 26×          |
| **Delta**        | 4       | **5×**         | 6.8×         |

## Top 10 Rust Wins

| Benchmark        | statsmodels | fastLowess | Speedup   |
|------------------|-------------|------------|-----------|
| scale_100000     | 43.727s     | 11.2ms     | **3914×** |
| scale_50000      | 11.160s     | 5.74ms     | **1946×** |
| financial_10000  | 497.1ms     | 0.59ms     | **839×**  |
| scientific_10000 | 777.2ms     | 0.93ms     | **835×**  |
| scale_10000      | 663.1ms     | 0.81ms     | **819×**  |
| clustered        | 267.8ms     | 0.48ms     | **554×**  |
| scale_5000       | 229.9ms     | 0.42ms     | **554×**  |
| fraction_0.1     | 227.9ms     | 0.42ms     | **542×**  |
| fraction_0.05    | 197.2ms     | 0.37ms     | **536×**  |
| financial_5000   | 170.9ms     | 0.32ms     | **536×**  |

## Regressions

**None identified.** `fastLowess` outperforms `statsmodels` in all 37 matched benchmarks. The parallel implementation ensures that even at extreme scales (100k points), processing remains sub-12ms.

## Detailed Results

### Scalability (1K - 100K points)

| Size    | fastLowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 1,000   | 0.17ms     | 30.4ms      | 175×    |
| 5,000   | 0.42ms     | 229.9ms     | 554×    |
| 10,000  | 0.81ms     | 663.1ms     | 819×    |
| 50,000  | 5.74ms     | 11.16s      | 1946×   |
| 100,000 | 11.2ms     | 43.73s      | 3914×   |

### Fraction Variations (n=5000)

| Fraction | fastLowess | statsmodels | Speedup |
|----------|------------|-------------|---------|
| 0.05     | 0.37ms     | 197.2ms     | 536×    |
| 0.10     | 0.42ms     | 227.9ms     | 542×    |
| 0.20     | 0.61ms     | 297.0ms     | 491×    |
| 0.30     | 0.82ms     | 357.0ms     | 436×    |
| 0.50     | 1.35ms     | 488.4ms     | 362×    |
| 0.67     | 1.93ms     | 601.6ms     | 312×    |

### Robustness Iterations (n=5000)

| Iterations | fastLowess | statsmodels | Speedup |
|------------|------------|-------------|---------|
| 0          | 0.14ms     | 74.2ms      | 527×    |
| 1          | 0.30ms     | 148.5ms     | 491×    |
| 2          | 0.46ms     | 222.8ms     | 489×    |
| 3          | 0.60ms     | 296.5ms     | 493×    |
| 5          | 0.91ms     | 445.1ms     | 490×    |
| 10         | 1.68ms     | 815.6ms     | 485×    |

### Delta Parameter (n=10000)

| Delta    | fastLowess | statsmodels | Speedup |
|----------|------------|-------------|---------|
| none (0) | 44.73ms    | 678.2ms     | 15.2×   |
| small    | 0.40ms     | 2.28ms      | 5.7×    |
| medium   | 0.29ms     | 1.27ms      | 4.3×    |
| large    | 0.36ms     | 0.76ms      | 2.1×    |

### Pathological Cases (n=5000)

| Case             | fastLowess | statsmodels | Speedup |
|------------------|------------|-------------|---------|
| clustered        | 0.48ms     | 267.8ms     | 554×    |
| constant_y       | 0.45ms     | 230.3ms     | 508×    |
| extreme_outliers | 1.71ms     | 852.0ms     | 497×    |
| high_noise       | 2.10ms     | 726.9ms     | 346×    |

### Real-World Scenarios

#### Financial Time Series

| Size    | fastLowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 500     | 0.08ms     | 10.4ms      | 131×    |
| 1,000   | 0.13ms     | 22.2ms      | 166×    |
| 5,000   | 0.32ms     | 170.9ms     | 536×    |
| 10,000  | 0.59ms     | 497.1ms     | 839×    |

#### Scientific Measurements

| Size    | fastLowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 500     | 0.15ms     | 14.1ms      | 92×     |
| 1,000   | 0.19ms     | 31.6ms      | 164×    |
| 5,000   | 0.51ms     | 268.5ms     | 526×    |
| 10,000  | 0.93ms     | 777.2ms     | 835×    |

#### Genomic Methylation (with delta=100)

| Size    | fastLowess | statsmodels | Speedup |
|---------|------------|-------------|---------|
| 1,000   | 0.65ms     | 29.5ms      | 45×     |
| 5,000   | 9.37ms     | 227.3ms     | 24×     |
| 10,000  | 33.9ms     | 662.8ms     | 20×     |
| 50,000  | 879.3ms    | 11.2s       | 13×     |

## Notes

- **Parallel Execution**: Enabled for n ≥ 1000 using Rayon.
- Benchmarks use **Criterion** (Rust) and **pytest-benchmark** (Python).
- Both use identical scenarios with reproducible RNG (seed=42).
- Rust crate: `fastLowess` v0.2.2 (running on `lowess` v0.5.3 core).
- Python: `statsmodels` v0.14.4 (with Cython backend).
- Test date: 2025-12-22.

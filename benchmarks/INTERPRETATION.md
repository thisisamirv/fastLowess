# Benchmark Interpretation (fastLowess)

## Summary

The `fastLowess` crate demonstrates massive performance gains over Python's `statsmodels`. The introduction of a dedicated "Serial vs. Parallel" benchmark for the Rust CPU backend reveals that **Parallel Execution (Rayon)** is the decisive winner across almost all standard benchmarks, often achieving **multi-hundred-fold speedups**.

## Consolidated Comparison

The table below shows speedups relative to the **baseline**.

- **Standard Benchmarks**: Baseline is `statsmodels` (Python).
- **Large Scale Benchmarks**: Baseline is `Rust (Serial)` (1x), as `statsmodels` times out.

| Name                  | statsmodels |      R      |  Rust (CPU)*  | Rust (GPU)|
|-----------------------|-------------|-------------|---------------|-----------|
|                       |             |             |               |           |
| clustered             |  162.77ms   |  [82.8x]²   |  [203-433x]¹  |   32.4x   |
| constant_y            |  133.63ms   |  [92.3x]²   |  [212-410x]¹  |   17.5x   |
| delta_large           |   0.51ms    |   [0.8x]²   |  [3.8-2.2x]¹  |   0.1x    |
| delta_medium          |   0.79ms    |   [1.3x]²   |  [4.4-3.4x]¹  |   0.1x    |
| delta_none            |  414.86ms   |    2.5x     |  [3.8-13x]²   | [63.5x]¹  |
| delta_small           |   1.45ms    |   [1.7x]²   |  [4.3-4.5x]¹  |   0.2x    |
| extreme_outliers      |  488.96ms   |  [106.4x]²  |  [201-388x]¹  |   28.9x   |
| financial_1000        |   13.55ms   |  [76.6x]²   |  [145-108x]¹  |   4.7x    |
| financial_10000       |  302.20ms   |  [168.3x]²  |  [453-611x]¹  |   26.3x   |
| financial_500         |   6.49ms    |  [58.0x]¹   |  [113-58x]²   |   2.7x    |
| financial_5000        |  103.94ms   |  [117.3x]²  |  [296-395x]¹  |   14.1x   |
| fraction_0.05         |  122.00ms   |  [177.6x]²  |  [421-350x]¹  |   14.5x   |
| fraction_0.1          |  140.59ms   |  [112.8x]²  |  [291-366x]¹  |   15.9x   |
| fraction_0.2          |  181.57ms   |  [85.3x]²   |  [210-419x]¹  |   19.3x   |
| fraction_0.3          |  220.98ms   |  [84.8x]²   |  [168-380x]¹  |   22.4x   |
| fraction_0.5          |  296.47ms   |  [80.9x]²   |  [146-415x]¹  |   27.3x   |
| fraction_0.67         |  362.59ms   |  [83.1x]²   |  [129-413x]¹  |   32.0x   |
| genomic_1000          |   17.82ms   |  [15.9x]²   |   [19-33x]¹   |   6.5x    |
| genomic_10000         |  399.90ms   |    3.6x     |  [5.3-16x]²   | [70.3x]¹  |
| genomic_5000          |  138.49ms   |    5.0x     |  [7.0-19x]²   | [34.8x]¹  |
| genomic_50000         |  6776.57ms  |    2.4x     |  [3.5-11x]²   | [269.2x]¹ |
| high_noise            |  435.85ms   |  [132.6x]²  |  [134-375x]¹  |   32.3x   |
| iterations_0          |   45.18ms   |  [128.4x]²  |  [266-405x]¹  |   10.6x   |
| iterations_1          |   94.10ms   |  [114.3x]²  |  [236-384x]¹  |   14.4x   |
| iterations_10         |  495.65ms   |  [116.0x]²  |  [204-369x]¹  |   27.0x   |
| iterations_2          |  135.48ms   |  [109.0x]²  |  [219-432x]¹  |   16.6x   |
| iterations_3          |  181.56ms   |  [108.8x]²  |  [213-382x]¹  |   18.7x   |
| iterations_5          |  270.58ms   |  [110.4x]²  |  [208-345x]¹  |   22.7x   |
| scale_1000            |   17.95ms   |  [82.6x]²   |  [150-107x]¹  |   8.1x    |
| scale_10000           |  408.13ms   |  [178.1x]²  |  [433-552x]¹  |   76.3x   |
| scale_5000            |  139.81ms   |  [133.6x]²  |  [289-401x]¹  |   28.8x   |
| scale_50000           |  6798.58ms  |  [661.0x]²  | [1077-1264x]¹ |  277.2x   |
| scientific_1000       |   19.04ms   |  [70.1x]²   |  [113-115x]¹  |   5.4x    |
| scientific_10000      |  479.57ms   |  [190.7x]²  |  [370-663x]¹  |   35.2x   |
| scientific_500        |   8.59ms    |  [49.6x]²   |   [91-52x]¹   |   3.2x    |
| scientific_5000       |  161.42ms   |  [124.9x]²  |  [244-427x]¹  |   17.9x   |
| scale_100000**        |      -      |      -      |    1-1.3x     |   0.3x    |
| scale_1000000**       |      -      |      -      |    1-1.3x     |   0.3x    |
| scale_2000000**       |      -      |      -      |    1-1.5x     |   0.3x    |
| scale_250000**        |      -      |      -      |    1-1.4x     |   0.3x    |
| scale_500000**        |      -      |      -      |    1-1.3x     |   0.3x    |

\* **Rust (CPU)**: Shows range `Seq - Par`. E.g., `12-48x` means 12x speedup (Sequential) and 48x speedup (Parallel). Rank determined by Parallel speedup.
\*\* **Large Scale**: `Rust (Serial)` is the baseline (1x).

¹ Winner (Fastest implementation)
² Runner-up (Second fastest implementation)

## Key Takeaways

1. **Rust (Parallel CPU)** is the dominant performer for general-purpose workloads, consistently achieving the highest speedups (often 300x-500x over statsmodels).
2. **R (stats::lowess)** is a very strong runner-up, frequently outperforming statsmodels by ~80-150x, but generally trailing Rust Parallel.
3. **Rust (GPU)** excels in specific high-compute scenarios (e.g., `genomic` with large datasets or `delta_none` where interpolation is skipped), but carries overhead that makes it slower than the highly optimized CPU backend for smaller datasets.
4. **Large Scale Scaling**: At very large scales (100k - 2M points), the parallel CPU backend maintains a modest lead (1.3x - 1.5x) over the sequential CPU backend, likely bottlenecked by memory bandwidth rather than compute.
5. **Small vs Large Delta**: Setting `delta=0` (no interpolation, `delta_none`) allows the GPU to shine (63.5x speedup), outperforming both CPU variants due to the massive O(N²) interaction workload being parallelized across thousands of GPU cores.

## Recommendation

- **Default**: Use **Rust CPU Parallel** for best all-around performance.
- **High Throughput / No-Delta**: Consider **Rust GPU** if you specifically require exact `delta=0` calculations on large datasets, where it significantly outperforms the CPU.

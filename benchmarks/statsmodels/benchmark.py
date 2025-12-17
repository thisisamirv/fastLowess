"""
Benchmark statsmodels LOWESS implementation.

This script benchmarks Python's statsmodels.nonparametric.lowess
to compare against the Rust implementation.

Requirements:
    pip install statsmodels numpy pandas matplotlib seaborn

Run with:
    python benchmark_statsmodels.py
"""

import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import json
from pathlib import Path

# Robust import for statsmodels.lowess (location changed across versions)
try:
    # modern location used by many releases
    from statsmodels.nonparametric.smoothers_lowess import lowess
except Exception:
    try:
        # older import path used in some examples
        from statsmodels.nonparametric.lowess import lowess  # type: ignore
    except Exception:
        raise RuntimeError(
            "statsmodels.lowess not found. Install statsmodels and ensure you're running "
            "the script with the same Python interpreter where the package is installed:\n\n"
            "    python -m pip install --upgrade statsmodels numpy pandas\n\n"
            "Then run the script with that python (e.g. `python benchmark_statsmodels.py`)."
        )


class BenchmarkResult:
    """Store benchmark timing results."""
    
    def __init__(self, name: str, size: int, iterations: int = 10):
        self.name = name
        self.size = size
        self.iterations = iterations
        self.times: List[float] = []
        self.mean_time: float = 0.0
        self.std_time: float = 0.0
        self.median_time: float = 0.0
        self.min_time: float = 0.0
        self.max_time: float = 0.0
    
    def add_time(self, elapsed: float):
        """Add a timing measurement."""
        self.times.append(elapsed)
    
    def finalize(self):
        """Compute statistics from collected times."""
        if self.times:
            self.mean_time = np.mean(self.times)
            self.std_time = np.std(self.times)
            self.median_time = np.median(self.times)
            self.min_time = np.min(self.times)
            self.max_time = np.max(self.times)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            'name': self.name,
            'size': self.size,
            'iterations': self.iterations,
            'mean_time_ms': self.mean_time * 1000,
            'std_time_ms': self.std_time * 1000,
            'median_time_ms': self.median_time * 1000,
            'min_time_ms': self.min_time * 1000,
            'max_time_ms': self.max_time * 1000,
        }


def generate_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for benchmarking.
    
    Args:
        size: Number of data points
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (x, y) arrays
    """
    np.random.seed(seed)
    x = np.linspace(0, 10, size)
    y = np.sin(x) + np.random.normal(0, 0.2, size)
    return x, y


def generate_data_with_outliers(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with outliers.
    
    Args:
        size: Number of data points
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (x, y) arrays with outliers
    """
    np.random.seed(seed)
    x = np.linspace(0, 10, size)
    y = np.sin(x) + np.random.normal(0, 0.2, size)
    
    # Add outliers (5% of points)
    n_outliers = max(1, size // 20)
    outlier_indices = np.random.choice(size, n_outliers, replace=False)
    y[outlier_indices] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(2, 5, n_outliers)
    
    return x, y


def benchmark_basic_smoothing(sizes: List[int], iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark basic LOWESS smoothing with different dataset sizes.
    
    Args:
        sizes: List of dataset sizes to test
        iterations: Number of iterations per size
    
    Returns:
        List of BenchmarkResult objects
    """
    results = []
    
    for size in sizes:
        print(f"Benchmarking basic smoothing with size={size}...")
        result = BenchmarkResult(f"basic_smoothing_{size}", size, iterations)
        
        x, y = generate_data(size)
        
        # Warmup
        _ = lowess(y, x, frac=0.3, it=3, return_sorted=False)
        
        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            _ = lowess(y, x, frac=0.3, it=3, return_sorted=False)
            elapsed = time.perf_counter() - start
            result.add_time(elapsed)
        
        result.finalize()
        results.append(result)
        print(f"  Mean: {result.mean_time*1000:.2f} ms ± {result.std_time*1000:.2f} ms")
    
    return results


def benchmark_fraction_variations(size: int = 1000, iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark different smoothing fractions.
    
    Args:
        size: Dataset size
        iterations: Number of iterations per fraction
    
    Returns:
        List of BenchmarkResult objects
    """
    results = []
    fractions = [0.1, 0.2, 0.3, 0.5, 0.67, 0.8]
    
    x, y = generate_data(size)
    
    for frac in fractions:
        print(f"Benchmarking fraction={frac}...")
        result = BenchmarkResult(f"fraction_{frac}", size, iterations)
        
        # Warmup
        _ = lowess(y, x, frac=frac, it=3, return_sorted=False)
        
        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            _ = lowess(y, x, frac=frac, it=3, return_sorted=False)
            elapsed = time.perf_counter() - start
            result.add_time(elapsed)
        
        result.finalize()
        results.append(result)
        print(f"  Mean: {result.mean_time*1000:.2f} ms ± {result.std_time*1000:.2f} ms")
    
    return results


def benchmark_robustness_iterations(size: int = 1000, iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark different numbers of robustness iterations.
    
    Args:
        size: Dataset size
        iterations: Number of iterations per robustness setting
    
    Returns:
        List of BenchmarkResult objects
    """
    results = []
    niter_values = [0, 1, 2, 3, 5, 10]
    
    x, y = generate_data_with_outliers(size)
    
    for niter in niter_values:
        print(f"Benchmarking robustness iterations={niter}...")
        result = BenchmarkResult(f"iterations_{niter}", size, iterations)
        
        # Warmup
        _ = lowess(y, x, frac=0.3, it=niter, return_sorted=False)
        
        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            _ = lowess(y, x, frac=0.3, it=niter, return_sorted=False)
            elapsed = time.perf_counter() - start
            result.add_time(elapsed)
        
        result.finalize()
        results.append(result)
        print(f"  Mean: {result.mean_time*1000:.2f} ms ± {result.std_time*1000:.2f} ms")
    
    return results


def benchmark_delta_parameter(size: int = 5000, iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark delta parameter effects.
    
    Args:
        size: Dataset size
        iterations: Number of iterations per delta value
    
    Returns:
        List of BenchmarkResult objects
    """
    results = []
    
    x = np.linspace(0, 50, size)
    y = np.sin(x)
    
    # Delta values as fraction of data range
    x_range = x[-1] - x[0]
    delta_configs = [
        ("delta_none", 0.0),
        ("delta_auto", 0.01 * x_range),  # Statsmodels default
        ("delta_small", 0.1),
        ("delta_large", 1.0),
    ]
    
    for name, delta in delta_configs:
        print(f"Benchmarking {name} (delta={delta:.2f})...")
        result = BenchmarkResult(name, size, iterations)
        
        # Warmup
        _ = lowess(y, x, frac=0.3, it=2, delta=delta, return_sorted=False)
        
        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            _ = lowess(y, x, frac=0.3, it=2, delta=delta, return_sorted=False)
            elapsed = time.perf_counter() - start
            result.add_time(elapsed)
        
        result.finalize()
        results.append(result)
        print(f"  Mean: {result.mean_time*1000:.2f} ms ± {result.std_time*1000:.2f} ms")
    
    return results


def benchmark_pathological_cases(size: int = 1000, iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark edge cases and pathological inputs.
    
    Args:
        size: Dataset size
        iterations: Number of iterations per case
    
    Returns:
        List of BenchmarkResult objects
    """
    results = []
    
    # Clustered x values
    print("Benchmarking clustered_x...")
    x_clustered = np.array([i // 100 + (i % 100) * 1e-6 for i in range(size)])
    y_clustered = np.sin(x_clustered)
    result = BenchmarkResult("clustered_x", size, iterations)
    
    # Warmup
    _ = lowess(y_clustered, x_clustered, frac=0.5, it=2, return_sorted=False)
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = lowess(y_clustered, x_clustered, frac=0.5, it=2, return_sorted=False)
        elapsed = time.perf_counter() - start
        result.add_time(elapsed)
    
    result.finalize()
    results.append(result)
    print(f"  Mean: {result.mean_time*1000:.2f} ms ± {result.std_time*1000:.2f} ms")
    
    # Extreme outliers
    print("Benchmarking extreme_outliers...")
    x_normal = np.linspace(0, 10, size)
    y_outliers = np.sin(x_normal)
    for i in range(0, size, 50):
        y_outliers[i] += 100.0 if i % 100 == 0 else -100.0
    
    result = BenchmarkResult("extreme_outliers", size, iterations)
    
    # Warmup
    _ = lowess(y_outliers, x_normal, frac=0.3, it=5, return_sorted=False)
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = lowess(y_outliers, x_normal, frac=0.3, it=5, return_sorted=False)
        elapsed = time.perf_counter() - start
        result.add_time(elapsed)
    
    result.finalize()
    results.append(result)
    print(f"  Mean: {result.mean_time*1000:.2f} ms ± {result.std_time*1000:.2f} ms")
    
    # Constant y values
    print("Benchmarking constant_y...")
    y_constant = np.full(size, 5.0)
    result = BenchmarkResult("constant_y", size, iterations)
    
    # Warmup
    _ = lowess(y_constant, x_normal, frac=0.3, it=2, return_sorted=False)
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = lowess(y_constant, x_normal, frac=0.3, it=2, return_sorted=False)
        elapsed = time.perf_counter() - start
        result.add_time(elapsed)
    
    result.finalize()
    results.append(result)
    print(f"  Mean: {result.mean_time*1000:.2f} ms ± {result.std_time*1000:.2f} ms")
    
    # High noise
    print("Benchmarking high_noise...")
    signal = np.sin(x_normal / 10.0) * 0.1
    noise = np.sin(np.arange(size) * 7.3) * 2.0
    y_noisy = signal + noise
    result = BenchmarkResult("high_noise", size, iterations)
    
    # Warmup
    _ = lowess(y_noisy, x_normal, frac=0.6, it=3, return_sorted=False)
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = lowess(y_noisy, x_normal, frac=0.6, it=3, return_sorted=False)
        elapsed = time.perf_counter() - start
        result.add_time(elapsed)
    
    result.finalize()
    results.append(result)
    print(f"  Mean: {result.mean_time*1000:.2f} ms ± {result.std_time*1000:.2f} ms")
    
    return results


def benchmark_realistic_scenarios(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark realistic application scenarios.
    
    Args:
        iterations: Number of iterations per scenario
    
    Returns:
        List of BenchmarkResult objects
    """
    results = []
    
    # Financial time series
    print("Benchmarking financial_timeseries...")
    size = 1000
    x = np.arange(size, dtype=float)
    trend = x * 0.01
    volatility = np.sin(x / 50.0) * 0.5
    random_walk = np.cumsum(np.random.normal(0, 0.01, size))
    y = trend + volatility + random_walk
    
    result = BenchmarkResult("financial_timeseries", size, iterations)
    
    # Warmup
    _ = lowess(y, x, frac=0.1, it=2, return_sorted=False)
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = lowess(y, x, frac=0.1, it=2, return_sorted=False)
        elapsed = time.perf_counter() - start
        result.add_time(elapsed)
    
    result.finalize()
    results.append(result)
    print(f"  Mean: {result.mean_time*1000:.2f} ms ± {result.std_time*1000:.2f} ms")
    
    # Scientific measurement data
    print("Benchmarking scientific_data...")
    x_sci = np.linspace(0, 10, size)
    signal = np.exp(x_sci * 0.2) * np.cos(x_sci * 10)
    noise = np.random.normal(0, 0.1, size)
    y_sci = signal + noise
    
    result = BenchmarkResult("scientific_data", size, iterations)
    
    # Warmup
    _ = lowess(y_sci, x_sci, frac=0.2, it=3, return_sorted=False)
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = lowess(y_sci, x_sci, frac=0.2, it=3, return_sorted=False)
        elapsed = time.perf_counter() - start
        result.add_time(elapsed)
    
    result.finalize()
    results.append(result)
    print(f"  Mean: {result.mean_time*1000:.2f} ms ± {result.std_time*1000:.2f} ms")
    
    # Genomic methylation data
    print("Benchmarking genomic_methylation...")
    x_genomic = np.arange(0, size * 1000, 1000, dtype=float)
    local_mean = 0.5 + np.sin(x_genomic / 5000.0) * 0.2
    noise = np.random.normal(0, 0.15, size)
    y_genomic = np.clip(local_mean + noise, 0.0, 1.0)
    
    result = BenchmarkResult("genomic_methylation", size, iterations)
    
    # Warmup
    _ = lowess(y_genomic, x_genomic, frac=0.2, it=3, delta=100.0, return_sorted=False)
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = lowess(y_genomic, x_genomic, frac=0.2, it=3, delta=100.0, return_sorted=False)
        elapsed = time.perf_counter() - start
        result.add_time(elapsed)
    
    result.finalize()
    results.append(result)
    print(f"  Mean: {result.mean_time*1000:.2f} ms ± {result.std_time*1000:.2f} ms")
    
    return results


def save_results(all_results: Dict[str, List[BenchmarkResult]], filename: str = "statsmodels_benchmark.json"):
    """Save benchmark results to JSON file in the workspace-level `output/` directory."""
    output = {}
    for category, results in all_results.items():
        output[category] = [r.to_dict() for r in results]

    # Determine workspace root (benchmarks directory)
    script_dir = Path(__file__).resolve().parent
    benchmarks_dir = script_dir.parent
    
    out_dir = benchmarks_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")


def print_summary(all_results: Dict[str, List[BenchmarkResult]]):
    """Print summary statistics.
    
    Args:
        all_results: Dictionary of benchmark category -> results
    """
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    for category, results in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}")
        print("-" * 80)
        for result in results:
            print(f"{result.name:30s} | Mean: {result.mean_time*1000:8.2f} ms | "
                  f"Std: {result.std_time*1000:6.2f} ms | "
                  f"Min: {result.min_time*1000:8.2f} ms | "
                  f"Max: {result.max_time*1000:8.2f} ms")


def main():
    """Run all benchmarks."""
    print("="*80)
    print("STATSMODELS LOWESS BENCHMARK SUITE")
    print("="*80)
    print()
    
    all_results = {}
    
    # Core benchmarks
    print("\n" + "="*80)
    print("CORE BENCHMARKS")
    print("="*80 + "\n")
    
    all_results['basic_smoothing'] = benchmark_basic_smoothing(
        sizes=[100, 500, 1000, 5000, 10000],
        iterations=10
    )
    
    all_results['fraction_variations'] = benchmark_fraction_variations(
        size=1000,
        iterations=10
    )
    
    all_results['robustness_iterations'] = benchmark_robustness_iterations(
        size=1000,
        iterations=10
    )
    
    all_results['delta_parameter'] = benchmark_delta_parameter(
        size=5000,
        iterations=10
    )
    
    # Stress tests
    print("\n" + "="*80)
    print("STRESS TESTS")
    print("="*80 + "\n")
    
    all_results['pathological_cases'] = benchmark_pathological_cases(
        size=1000,
        iterations=10
    )
    
    # Application scenarios
    print("\n" + "="*80)
    print("APPLICATION SCENARIOS")
    print("="*80 + "\n")
    
    all_results['realistic_scenarios'] = benchmark_realistic_scenarios(
        iterations=10
    )
    
    # Print summary and save
    print_summary(all_results)
    save_results(all_results)


if __name__ == '__main__':
    main()

"""
Industry-level LOWESS benchmarks with JSON output for comparison with Rust.

Benchmarks are aligned with the Rust criterion benchmarks to enable direct comparison.
Results are written to benchmarks/output/statsmodels_benchmark.json.

Run with: python3 benchmark_pytest.py
Or with pytest: pytest benchmark_pytest.py -v --benchmark-json=output.json
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from dataclasses import dataclass, asdict

# Robust import for statsmodels.lowess
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    from statsmodels.nonparametric.lowess import lowess


# ============================================================================
# Benchmark Result Storage
# ============================================================================


@dataclass
class BenchmarkResult:
    """Store benchmark timing results."""
    name: str
    size: int
    iterations: int
    mean_time_ms: float = 0.0
    std_time_ms: float = 0.0
    median_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0


def run_benchmark(name: str, size: int, func, iterations: int = 10, warmup: int = 2) -> BenchmarkResult:
    """Run a benchmark with warmup and timing."""
    result = BenchmarkResult(name=name, size=size, iterations=iterations)
    
    # Warmup runs
    for _ in range(warmup):
        func()
    
    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    result.mean_time_ms = np.mean(times)
    result.std_time_ms = np.std(times)
    result.median_time_ms = np.median(times)
    result.min_time_ms = np.min(times)
    result.max_time_ms = np.max(times)
    
    return result


# ============================================================================
# Data Generation (Aligned with Rust)
# ============================================================================


def generate_sine_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate smooth sinusoidal data with Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, size)
    y = np.sin(x) + rng.normal(0, 0.2, size)
    return x, y


def generate_outlier_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data with 5% outliers."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, size)
    y = np.sin(x) + rng.normal(0, 0.2, size)
    
    # Add 5% outliers
    n_outliers = size // 20
    for _ in range(n_outliers):
        idx = rng.integers(0, size)
        y[idx] += rng.uniform(-5, 5)
    return x, y


def generate_financial_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate financial time series (geometric Brownian motion)."""
    rng = np.random.default_rng(seed)
    x = np.arange(size, dtype=float)
    
    # Simulate stock prices with realistic volatility
    y = [100.0]  # Starting price
    for _ in range(1, size):
        ret = rng.normal(0.0005, 0.02)
        y.append(y[-1] * (1 + ret))
    return x, np.array(y)


def generate_scientific_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate scientific measurement data (exponential decay with oscillations)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, size)
    signal = np.exp(-x * 0.3) * np.cos(x * 2 * np.pi)
    noise = rng.normal(0, 0.05, size)
    return x, signal + noise


def generate_genomic_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate genomic methylation data (beta values 0-1)."""
    rng = np.random.default_rng(seed)
    x = np.arange(size) * 1000.0  # CpG positions
    base = 0.5 + np.sin(x / 50000.0) * 0.3
    noise = rng.normal(0, 0.1, size)
    y = np.clip(base + noise, 0.0, 1.0)
    return x, y


def generate_clustered_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate clustered x-values (groups with tiny spacing)."""
    rng = np.random.default_rng(seed)
    x = np.array([(i // 100) + (i % 100) * 1e-6 for i in range(size)])
    y = np.sin(x) + rng.normal(0, 0.1, size)
    return x, y


def generate_high_noise_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate high-noise data (SNR < 1)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, size)
    signal = np.sin(x) * 0.5
    noise = rng.normal(0, 2.0, size)  # High noise
    return x, signal + noise


# ============================================================================
# Benchmark Categories (Aligned with Rust criterion benchmarks)
# ============================================================================


def benchmark_scalability(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark performance scaling with dataset size."""
    print("\n" + "=" * 80)
    print("SCALABILITY")
    print("=" * 80)
    
    results = []
    sizes = [1000, 5000, 10000, 50000, 100000]
    
    for size in sizes:
        x, y = generate_sine_data(size, seed=42)
        
        def run():
            lowess(y, x, frac=0.1, it=3, return_sorted=False)
        
        result = run_benchmark(f"scale_{size}", size, run, iterations)
        results.append(result)
        print(f"  scale_{size}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_fraction(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark different smoothing fractions."""
    print("\n" + "=" * 80)
    print("FRACTION")
    print("=" * 80)
    
    results = []
    size = 5000
    fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.67]
    x, y = generate_sine_data(size, seed=42)
    
    for frac in fractions:
        def run(f=frac):
            lowess(y, x, frac=f, it=3, return_sorted=False)
        
        result = run_benchmark(f"fraction_{frac}", size, run, iterations)
        results.append(result)
        print(f"  fraction_{frac}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_iterations(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark different robustness iterations."""
    print("\n" + "=" * 80)
    print("ITERATIONS")
    print("=" * 80)
    
    results = []
    size = 5000
    iter_values = [0, 1, 2, 3, 5, 10]
    x, y = generate_outlier_data(size, seed=42)
    
    for it in iter_values:
        def run(n=it):
            lowess(y, x, frac=0.2, it=n, return_sorted=False)
        
        result = run_benchmark(f"iterations_{it}", size, run, iterations)
        results.append(result)
        print(f"  iterations_{it}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_delta(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark delta parameter effects."""
    print("\n" + "=" * 80)
    print("DELTA")
    print("=" * 80)
    
    results = []
    size = 10000
    x, y = generate_sine_data(size, seed=42)
    
    delta_configs = [
        ("delta_none", 0.0),
        ("delta_small", 0.5),
        ("delta_medium", 2.0),
        ("delta_large", 10.0),
    ]
    
    for name, delta in delta_configs:
        def run(d=delta):
            lowess(y, x, frac=0.2, it=2, delta=d, return_sorted=False)
        
        result = run_benchmark(name, size, run, iterations)
        results.append(result)
        print(f"  {name}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_financial(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark with financial time series data."""
    print("\n" + "=" * 80)
    print("FINANCIAL")
    print("=" * 80)
    
    results = []
    sizes = [500, 1000, 5000, 10000]
    
    for size in sizes:
        x, y = generate_financial_data(size, seed=42)
        
        def run():
            lowess(y, x, frac=0.1, it=2, return_sorted=False)
        
        result = run_benchmark(f"financial_{size}", size, run, iterations)
        results.append(result)
        print(f"  financial_{size}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_scientific(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark with scientific measurement data."""
    print("\n" + "=" * 80)
    print("SCIENTIFIC")
    print("=" * 80)
    
    results = []
    sizes = [500, 1000, 5000, 10000]
    
    for size in sizes:
        x, y = generate_scientific_data(size, seed=42)
        
        def run():
            lowess(y, x, frac=0.15, it=3, return_sorted=False)
        
        result = run_benchmark(f"scientific_{size}", size, run, iterations)
        results.append(result)
        print(f"  scientific_{size}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_genomic(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark with genomic methylation data."""
    print("\n" + "=" * 80)
    print("GENOMIC")
    print("=" * 80)
    
    results = []
    sizes = [1000, 5000, 10000, 50000]
    
    for size in sizes:
        x, y = generate_genomic_data(size, seed=42)
        
        def run():
            lowess(y, x, frac=0.1, it=3, delta=100.0, return_sorted=False)
        
        result = run_benchmark(f"genomic_{size}", size, run, iterations)
        results.append(result)
        print(f"  genomic_{size}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_pathological(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark with pathological/edge case data."""
    print("\n" + "=" * 80)
    print("PATHOLOGICAL")
    print("=" * 80)
    
    results = []
    size = 5000
    
    # Clustered
    x_clustered, y_clustered = generate_clustered_data(size, seed=42)
    result = run_benchmark(
        "clustered", size,
        lambda: lowess(y_clustered, x_clustered, frac=0.3, it=2, return_sorted=False),
        iterations
    )
    results.append(result)
    print(f"  clustered: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    # High noise
    x_noisy, y_noisy = generate_high_noise_data(size, seed=42)
    result = run_benchmark(
        "high_noise", size,
        lambda: lowess(y_noisy, x_noisy, frac=0.5, it=5, return_sorted=False),
        iterations
    )
    results.append(result)
    print(f"  high_noise: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    # Extreme outliers
    x_outlier, y_outlier = generate_outlier_data(size, seed=42)
    result = run_benchmark(
        "extreme_outliers", size,
        lambda: lowess(y_outlier, x_outlier, frac=0.2, it=10, return_sorted=False),
        iterations
    )
    results.append(result)
    print(f"  extreme_outliers: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    # Constant y
    x_const = np.arange(size, dtype=float)
    y_const = np.full(size, 5.0)
    result = run_benchmark(
        "constant_y", size,
        lambda: lowess(y_const, x_const, frac=0.2, it=2, return_sorted=False),
        iterations
    )
    results.append(result)
    print(f"  constant_y: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results





# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Run all benchmarks and save results."""
    print("=" * 80)
    print("STATSMODELS LOWESS BENCHMARK SUITE (Aligned with Rust)")
    print("=" * 80)
    
    iterations = 10
    all_results: Dict[str, List[BenchmarkResult]] = {}
    
    # Run all benchmark categories
    all_results["scalability"] = benchmark_scalability(iterations)
    all_results["fraction"] = benchmark_fraction(iterations)
    all_results["iterations"] = benchmark_iterations(iterations)
    all_results["delta"] = benchmark_delta(iterations)
    all_results["financial"] = benchmark_financial(iterations)
    all_results["scientific"] = benchmark_scientific(iterations)
    all_results["genomic"] = benchmark_genomic(iterations)
    all_results["pathological"] = benchmark_pathological(iterations)

    
    # Convert to JSON-serializable format
    output = {}
    for category, results in all_results.items():
        output[category] = [asdict(r) for r in results]
    
    # Save to output directory
    script_dir = Path(__file__).resolve().parent
    benchmarks_dir = script_dir.parent
    out_dir = benchmarks_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "statsmodels_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Results saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

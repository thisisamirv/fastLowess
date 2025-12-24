import json
from pathlib import Path
from statistics import mean, median, stdev
import csv
import math

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def pick_time_value(entry: dict):
    """Robustly pick a numeric timing from an entry.
    Prefer mean_time_ms, then median_time_ms, then max_time_ms, then any numeric field.
    Returns (value_ms: float or None, size: int or None).
    """
    for key in ("mean_time_ms", "median_time_ms", "max_time_ms"):
        if key in entry:
            try:
                return float(entry[key]), entry.get("size")
            except Exception:
                pass
    # fallback: search for first numeric value
    for k, v in entry.items():
        if isinstance(v, (int, float)):
            # ignore small integer metadata like iteration counts if name-like keys present
            if k in ("iterations", "size", "runs"):
                continue
            try:
                return float(v), entry.get("size")
            except Exception:
                pass
    return None, entry.get("size")

def build_map(entries):
    # allow entries that might already be a dict of results
    out = {}
    for e in entries:
        name = e.get("name") or e.get("id") or e.get("test") or None
        if not name:
            # generate fallback unique name if missing
            name = json.dumps(e, sort_keys=True)
        out[name] = e
    return out

def compare_category(cpu_entries, gpu_entries):
    cpu_map = build_map(cpu_entries)
    gpu_map = build_map(gpu_entries)
    common = sorted(set(cpu_map.keys()) & set(gpu_map.keys()))
    rows = []
    speedups = []
    
    for name in common:
        c_entry = cpu_map[name]
        g_entry = gpu_map[name]
        c_val, c_size = pick_time_value(c_entry)
        g_val, g_size = pick_time_value(g_entry)

        row = {
            "name": name,
            "cpu_value_ms": c_val,
            "gpu_value_ms": g_val,
            "cpu_size": c_size,
            "gpu_size": g_size,
            "notes": []
        }

        if c_val is None or g_val is None:
            row["notes"].append("missing_metric")
            rows.append(row)
            continue

        # core comparisons
        if g_val == 0 or c_val == 0:
            speedup = None
        else:
            speedup = c_val / g_val  # >1 => GPU faster by this factor
        row["speedup_gpu_over_cpu"] = speedup
        if speedup is not None:
            row["log2_speedup"] = math.log2(speedup) if speedup > 0 else None
            # If GPU is faster (speedup > 1), time decreased.
            # % change usually implies (new - old) / old
            # (gpu - cpu) / cpu * 100
            row["percent_change_gpu_vs_cpu"] = ((g_val - c_val) / c_val) * 100.0
            speedups.append(speedup)

        # absolute diffs
        row["absolute_diff_ms"] = None if c_val is None or g_val is None else (g_val - c_val)
        row["abs_percent_vs_cpu"] = None if c_val == 0 else abs(row["absolute_diff_ms"]) / c_val * 100.0

        # per-point normalization if size available and >0
        size = c_size or g_size
        if size:
            try:
                size_i = int(size)
                row["cpu_ms_per_point"] = c_val / size_i
                row["gpu_ms_per_point"] = g_val / size_i
                row["speedup_per_point"] = None if row["gpu_ms_per_point"] == 0 else row["cpu_ms_per_point"] / row["gpu_ms_per_point"]
            except Exception:
                row["notes"].append("bad_size")

        rows.append(row)
    
    summary = {
        "compared": len(common),
        "mean_speedup": mean(speedups) if speedups else None,
        "median_speedup": median(speedups) if speedups else None,
        "count_with_metrics": len(speedups),
    }
    return rows, summary

def main():
    repo_root = Path(__file__).resolve().parent
    # walk up to workspace root (same heuristic as other scripts)
    workspace = repo_root
    for _ in range(6):
        if (workspace / "output").exists():
            break
        if workspace.parent == workspace:
            break
        workspace = workspace.parent
    out_dir = workspace / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_path = out_dir / "rust_benchmark.json"
    gpu_path = out_dir / "rust_benchmark_gpu.json"

    if not cpu_path.exists() or not gpu_path.exists():
        missing = []
        if not cpu_path.exists():
            missing.append(str(cpu_path))
        if not gpu_path.exists():
            missing.append(str(gpu_path))
        print("Missing files:", ", ".join(missing))
        return

    cpu_data = load_json(cpu_path)
    gpu_data = load_json(gpu_path)

    all_keys = sorted(set(cpu_data.keys()) | set(gpu_data.keys()))
    comparison = {}
    overall_speedups = []

    # detailed rows for CSV
    csv_rows = []
    
    for key in all_keys:
        c_entries = cpu_data.get(key, [])
        g_entries = gpu_data.get(key, [])
        
        # CPU/GPU matching relies on names. 
        # Note: GPU benches might be subsets of CPU benches.
        rows, summary = compare_category(c_entries, g_entries)
        
        comparison[key] = {"rows": rows, "summary": summary}
        if summary["median_speedup"] is not None:
            overall_speedups.append(summary["median_speedup"])
        for row in rows:
            csv_rows.append({
                "category": key,
                **row
            })

    print("\nBenchmark comparison (cpu_ms / gpu_ms) -> Speedup > 1.0 means GPU is faster:")
    for key, data in comparison.items():
        s = data["summary"]
        if s["compared"] > 0:
            print(f"- {key}: compared={s['compared']}, median_speedup={s['median_speedup']:.4f}, mean_speedup={s['mean_speedup']:.4f}")

    # Top wins and regressions across all categories
    all_rows = [r for cat in comparison.values() for r in cat["rows"] if r.get("speedup_gpu_over_cpu") is not None]
    if all_rows:
        sorted_by_speed = sorted(all_rows, key=lambda r: r["speedup_gpu_over_cpu"] or 0, reverse=True)
        sorted_by_regression = sorted(all_rows, key=lambda r: r["speedup_gpu_over_cpu"] or 0)

        print("\nTop 10 GPU wins (largest cpu_ms / gpu_ms):")
        for r in sorted_by_speed[:10]:
            print(f"  {r['name']}: cpu={r['cpu_value_ms']:.4f}ms, gpu={r['gpu_value_ms']:.4f}ms, speedup={r['speedup_gpu_over_cpu']:.4f}x")

        print("\nTop 10 regressions (CPU faster than GPU):")
        for r in sorted_by_regression[:10]:
            if r["speedup_gpu_over_cpu"] < 1.0:
                print(f"  {r['name']}: cpu={r['cpu_value_ms']:.4f}ms, gpu={r['gpu_value_ms']:.4f}ms, speedup={r['speedup_gpu_over_cpu']:.4f}x")

    # Print detailed per-category rows to console
    print("\nDetailed per-category results:")
    for cat, data in comparison.items():
        rows = data["rows"]
        if not rows:
            continue
        print(f"\nCategory: {cat} (compared={data['summary']['compared']})")
        # header
        print(f"{'name':60} {'cpu_ms':>10} {'gpu_ms':>10} {'speedup':>8} {'%chg':>8} {'notes'}")
        for r in rows:
            name = (r.get("name") or "")[:60].ljust(60)
            cpu_v = r.get("cpu_value_ms")
            gpu_v = r.get("gpu_value_ms")
            sp = r.get("speedup_gpu_over_cpu")
            pct = r.get("percent_change_gpu_vs_cpu")
            notes = ";".join(r.get("notes", []))
            
            cpu_s = f"{cpu_v:.4f}" if isinstance(cpu_v, (int, float)) else "N/A"
            gpu_s = f"{gpu_v:.4f}" if isinstance(gpu_v, (int, float)) else "N/A"
            sp_s = f"{sp:.2f}x" if isinstance(sp, (int, float)) else "N/A"
            pct_s = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "N/A"
            print(f"{name} {cpu_s:>10} {gpu_s:>10} {sp_s:>8} {pct_s:>8} {notes}")

if __name__ == "__main__":
    main()

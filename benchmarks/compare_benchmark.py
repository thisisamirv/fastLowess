import json
from pathlib import Path
from statistics import mean, median
import math

def load_json(p: Path):
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def pick_time_value(entry: dict):
    """Robustly pick a numeric timing from an entry."""
    for key in ("mean_time_ms", "median_time_ms", "max_time_ms"):
        if key in entry:
            try:
                return float(entry[key]), entry.get("size")
            except Exception:
                pass
    # fallback
    for k, v in entry.items():
        if isinstance(v, (int, float)):
            if k in ("iterations", "size", "runs"):
                continue
            try:
                return float(v), entry.get("size")
            except Exception:
                pass
    return None, entry.get("size")

def build_map(entries):
    out = {}
    for e in entries:
        name = e.get("name") or e.get("id") or e.get("test") or None
        if not name:
            name = json.dumps(e, sort_keys=True)
        out[name] = e
    return out

def compare_category(candidate_entries, baseline_entries):
    cand_map = build_map(candidate_entries)
    base_map = build_map(baseline_entries)
    
    common = sorted(set(cand_map.keys()) & set(base_map.keys()))
    rows = []
    speedups = []
    
    for name in common:
        c_entry = cand_map[name]
        b_entry = base_map[name]
        
        c_val, c_size = pick_time_value(c_entry)
        b_val, b_size = pick_time_value(b_entry)
        
        # c_val = candidate time (e.g. Rust), b_val = baseline time (e.g. Statsmodels)
        
        row = {
            "name": name,
            "candidate_ms": c_val,
            "baseline_ms": b_val,
            "size": c_size or b_size,
            "notes": []
        }
        
        if c_val is None or b_val is None:
            row["notes"].append("missing_metric")
            rows.append(row)
            continue
            
        if c_val == 0 or b_val == 0:
            speedup = None
        else:
            # Speedup = Baseline / Candidate
            # Example: Statsmodels=100ms, Rust=10ms -> Speedup = 10x
            speedup = b_val / c_val
            
        row["speedup"] = speedup
        if speedup is not None:
             speedups.append(speedup)
             
        rows.append(row)
        
    summary = {
        "compared": len(common),
        "mean_speedup": mean(speedups) if speedups else None,
        "median_speedup": median(speedups) if speedups else None,
    }
    return rows, summary

def load_all_data(output_dir: Path):
    files = {
        "Rust (CPU)": output_dir / "rust_benchmark_cpu.json",
        "Rust (Serial)": output_dir / "rust_benchmark_cpu_serial.json",
        "Rust (GPU)": output_dir / "rust_benchmark_gpu.json",
        "R": output_dir / "r_benchmark.json",
        "statsmodels": output_dir / "statsmodels_benchmark.json"
    }
    
    data = {}
    for label, path in files.items():
        loaded = load_json(path)
        if loaded:
            # Flatten category structure: {category: [entries]} -> {name: entry}
            flat = {}
            for cat, entries in loaded.items():
                for entry in entries:
                    name = entry.get("name")
                    if name:
                        flat[name] = entry
            data[label] = flat
    return data

def main():
    repo_root = Path(__file__).resolve().parent
    workspace = repo_root
    for _ in range(6):
        if (workspace / "output").exists():
            break
        if workspace.parent == workspace:
            break
        workspace = workspace.parent
    out_dir = workspace / "output"
    
    data = load_all_data(out_dir)
    stats_data = data.get("statsmodels")
    
    if not stats_data:
        print("Statsmodels baseline data not found or empty.")
        return

    # Collect all benchmark names
    all_names = set(stats_data.keys())
    for label, d in data.items():
        if label != "statsmodels":
            all_names.update(d.keys())
            
    large_scale_benchmarks = {
        "scale_100000", "scale_1000000", "scale_1e+05",
        "scale_250000", "scale_500000", 
        "scale_2000000"
    }
            
    regular_names = sorted([n for n in all_names if n not in large_scale_benchmarks])
    large_scale_names = sorted([n for n in all_names if n in large_scale_benchmarks])
    sorted_names = regular_names + large_scale_names
    
    # Columns
    candidates = ["R", "Rust (CPU)*", "Rust (GPU)"]
    
    # Print Table Header
    # Format: 
    # Name | statsmodels | R | Rust (CPU)* | Rust (GPU) |
    print(f"{'Name':<21} | {'statsmodels':^11} | {'R':^11} | {'Rust (CPU)*':^13} | {'Rust (GPU)':^9} |")
    print("-" * 80)

    for name in sorted_names:
        is_large_scale = name in large_scale_benchmarks
        display_name = f"{name}**" if is_large_scale else name

        # Baseline logic
        base_col_str = "-"
        base_val = None
        
        if is_large_scale:
            # Baseline is Rust (Serial)
            serial_data = data.get("Rust (Serial)", {})
            base_entry = serial_data.get(name)
        else:
             # Baseline is statsmodels
             base_entry = stats_data.get(name)
             if base_entry:
                  base_val, _ = pick_time_value(base_entry)
                  if base_val and base_val > 0:
                      base_col_str = f"{base_val:.2f}ms"
                  
        if base_entry and (base_val is None): # Need to parse for large scale if not parsed above
             base_val, _ = pick_time_value(base_entry)

        row_str = f"{display_name:<21} | {base_col_str:^11} |"

        if base_val is None or base_val == 0:
             # Missing baseline
             for _ in candidates:
                 row_str += f" {'-':^13} |" if "Rust (CPU)" in _ else f" {'-':^9} |" if "Rust (GPU)" in _ else f" {'-':^11} |"
        else:
            # Collect speedups for ranking
            # Structure: list of (candidate_label, speedup_val, display_str, raw_speedup_for_rank)
            results = []
            
            for cand in candidates:
                if cand == "R" and is_large_scale:
                    results.append((cand, None, "-", -1))
                    continue
                
                if cand == "Rust (CPU)*":
                     serial_data = data.get("Rust (Serial)", {})
                     par_data = data.get("Rust (CPU)", {})
                     s_entry = serial_data.get(name)
                     p_entry = par_data.get(name)
                     
                     s_val = pick_time_value(s_entry)[0] if s_entry else None
                     p_val = pick_time_value(p_entry)[0] if p_entry else None
                     
                     s_speedup_str = "?"
                     p_speedup_str = "?"
                     rank_val = -1
                     
                     if is_large_scale:
                         # Serial is baseline (1x)
                         s_speedup_str = "1"
                         if p_val and p_val > 0:
                             p_speedup = base_val / p_val
                             rank_val = p_speedup
                             p_speedup_str = f"{p_speedup:.1f}" if p_speedup < 10 else f"{p_speedup:.0f}"
                     else:
                         if s_val and s_val > 0:
                             s_speedup = base_val / s_val
                             s_speedup_str = f"{s_speedup:.1f}" if s_speedup < 10 else f"{s_speedup:.0f}"
                         
                         if p_val and p_val > 0:
                             p_speedup = base_val / p_val
                             rank_val = p_speedup
                             p_speedup_str = f"{p_speedup:.1f}" if p_speedup < 10 else f"{p_speedup:.0f}"
                             
                     disp = "-"
                     if s_speedup_str != "?" or p_speedup_str != "?":
                         disp = f"{s_speedup_str}-{p_speedup_str}x"
                         
                     results.append((cand, rank_val, disp, rank_val))
                     
                else: # R or Rust (GPU)
                    cand_data = data.get(cand, {})
                    cand_entry = cand_data.get(name)
                    rank_val = -1
                    disp = "-"
                    
                    if cand_entry:
                        c_val, _ = pick_time_value(cand_entry)
                        if c_val and c_val > 0:
                             speedup = base_val / c_val
                             rank_val = speedup
                             disp = f"{speedup:.1f}x"
                    
                    results.append((cand, rank_val, disp, rank_val))

            # Rank
            # Filter valid speedups > 0
            valid_ranks = sorted([r[3] for r in results if r[3] > 0], reverse=True)
            
            final_cells = []
            for cand, _, disp, r_val in results:
                cell_text = disp
                # Only apply highlighting for non-large-scale benchmarks (where we have all candidates)
                if not is_large_scale and r_val > 0 and valid_ranks:
                    if r_val == valid_ranks[0]:
                        cell_text = f"[{disp}]\u00b9"
                    elif len(valid_ranks) > 1 and r_val == valid_ranks[1]:
                         cell_text = f"[{disp}]\u00b2"
                
                final_cells.append(cell_text)

            # Columns correspond to R, Rust (CPU)*, Rust (GPU)
            # Adjust widths: R=11, CPU=13, GPU=9
            row_str += f" {final_cells[0]:^11} | {final_cells[1]:^13} | {final_cells[2]:^9} |"
            
        print(row_str)

    print("-" * 80)
    print("* Rust (CPU) column shows speedup range: Serial-Parallel (e.g., 12-48x means 12x speedup sequential, 48x parallel)")
    print("** For large scale benchmarks (scale_100000+), Rust (Serial) is the baseline (1x).")
    print("\u00b9 Winner (Fastest implementation)")
    print("\u00b2 Runner-up (Second fastest implementation)")

if __name__ == "__main__":
    main()

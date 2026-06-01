"""
scripts/exp3_main_benchmark.py
Experiment 3: Main Benchmark -- SPART vs JIT vanilla across all 4 CSVs.

This replaces Table III in the paper with corrected, reproducible numbers.

Two methods compared with IDENTICAL ego trajectory on each CSV:
  - JIT vanilla (B): build_segments(all) + scan_jit_vanilla each frame, no state
  - SPART full  (E): run_frame with eligibility scheduling + angular pruning,
                     serial JIT kernel (parallel=False), track_memory=False

Metrics reported per CSV and aggregated:
  scan_hz            -- 1000 / mean_scan_ms  (scan kernel throughput)
  full_hz            -- 1000 / mean_full_ms  (full pipeline throughput)
  mean_candidates    -- mean intersection tests per frame
  sei                -- Scan Efficiency Index = K * M_all / candidates  (SPART only)
  muting_pct         -- mean muting ratio %  (SPART only)
  speedup_scan       -- SPART scan_hz / JIT vanilla scan_hz
  speedup_full       -- SPART full_hz / JIT vanilla full_hz

Gate:
  SPART mean full-pipeline Hz >= 1000 across all CSVs    (real-time capable; 100x 10Hz LiDAR)
  SPART mean scan Hz > JIT vanilla mean scan Hz           (core claim)

Outputs (results/benchmark/):
  benchmark_summary.json
  per_frame_benchmark.csv          -- frame-level timing for all CSVs + methods
  fig1_full_hz_bar.png             -- full-pipeline throughput per CSV
  fig2_scan_hz_bar.png             -- scan-kernel throughput per CSV
  fig3_timing_evolution.png        -- per-frame timing trace (CSV_000)
  fig4_speedup_summary.png         -- speedup SPART/JIT across CSVs
"""

import argparse
import csv as _csv
import glob
import json
import math
import os
import sys
import time

import numpy as np
import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("[WARN] matplotlib not installed -- figures will be skipped")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spart.core import SPART
from spart.utils import (
    load_interaction_csv,
    make_ego_trajectory,
    get_frame_list,
    build_segments_from_frame,
)
from spart.scan_kernel import warmup_numba_kernel
from spart.vanilla_kernel import scan_jit_vanilla, warmup_jit_vanilla

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_COL_SPART = "#9467bd"   # purple
_COL_JIT   = "#ff7f0e"   # orange


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config():
    path = os.path.join(_ROOT, "configs", "benchmark_config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Per-CSV runner
# ---------------------------------------------------------------------------

def run_one_csv(csv_path, cfg, verbose=True):
    """
    Run JIT vanilla and SPART on one CSV.
    Returns (jit_rows, spart_rows, csv_summary_dict).
    """
    csv_name  = os.path.basename(csv_path)
    n_frames  = int(cfg["n_frames"])
    fov_deg   = float(cfg["fov_deg"])
    delta_deg = float(cfg["delta_deg"])
    r_max     = float(cfg["r_max_m"])
    ego_cfg   = {
        "radius": float(cfg["ego_circle_radius_m"]),
        "speed":  float(cfg["ego_circle_speed_mps"]),
        "center": tuple(cfg["ego_circle_center"]),
    }

    if verbose:
        print(f"\n  Loading {csv_name} ...")
    frames_dict, timestamps = load_interaction_csv(csv_path)
    frame_list = get_frame_list(frames_dict, n_frames=n_frames)
    ego_traj   = make_ego_trajectory(
        frame_ids=frame_list, timestamps=timestamps, mode="circular", **ego_cfg
    )

    fov   = math.radians(fov_deg)
    delta = math.radians(delta_deg)
    K     = int(math.floor(fov / delta + 0.5)) + 1

    mean_n_veh = float(np.mean([len(frames_dict[f]) for f in frame_list]))
    if verbose:
        print(f"  Frames: {len(frame_list)}, K beams: {K}, "
              f"mean N_veh: {mean_n_veh:.2f}")

    # ---- JIT vanilla (B) ----
    if verbose:
        print(f"  [JIT vanilla] running {len(frame_list):,} frames ...")
    jit_scan_ms  = []
    jit_full_ms  = []
    jit_cand     = []
    jit_rows     = []

    t0_jit = time.perf_counter()
    for fid in frame_list:
        fv    = frames_dict[fid]
        ego_s = ego_traj[fid]

        t0_pipe = time.perf_counter()
        segs, tids = build_segments_from_frame(fv)
        _, _, _, tc, scan_t = scan_jit_vanilla(
            ego_s, segs, fov_deg=fov_deg, delta_deg=delta_deg,
            r_max=r_max, seg_track_ids=tids)
        t_full = time.perf_counter() - t0_pipe

        jit_scan_ms.append(float(scan_t) * 1000.0)
        jit_full_ms.append(float(t_full) * 1000.0)
        jit_cand.append(int(tc))
        jit_rows.append({
            "csv": csv_name, "frame_id": fid, "method": "jit_vanilla",
            "n_vehicles": len(fv),
            "scan_ms": float(scan_t) * 1000.0,
            "full_ms": float(t_full) * 1000.0,
            "candidates": int(tc),
            "muted_ratio": float("nan"),
        })
    if verbose:
        print(f"  [JIT vanilla] done {time.perf_counter()-t0_jit:.1f}s | "
              f"mean scan={np.mean(jit_scan_ms):.4f}ms "
              f"full={np.mean(jit_full_ms):.4f}ms")

    # ---- SPART (E, serial, no memory tracking) ----
    if verbose:
        print(f"  [SPART]       running {len(frame_list):,} frames ...")
    spart_scan_ms  = []
    spart_full_ms  = []
    spart_cand     = []
    spart_muted    = []
    spart_rows     = []

    spart = SPART(
        {"fov_deg": fov_deg, "delta_deg": delta_deg,
         "r_max_m": r_max, "enable_grid": False,
         "parallel_kernel": False},
        track_memory=False,
    )

    t0_sp = time.perf_counter()
    for fid in frame_list:
        fv    = frames_dict[fid]
        ego_s = ego_traj[fid]
        tcurr = timestamps[fid]

        t0_pipe = time.perf_counter()
        _, _, _, met = spart.run_frame(fv, ego_s, tcurr)
        t_full = time.perf_counter() - t0_pipe

        scan_ms = float(met["time_scan_sec"]) * 1000.0
        full_ms = float(t_full) * 1000.0
        tc      = int(met["total_candidates"])
        muted   = float(met["muted_ratio"])

        spart_scan_ms.append(scan_ms)
        spart_full_ms.append(full_ms)
        spart_cand.append(tc)
        spart_muted.append(muted)
        spart_rows.append({
            "csv": csv_name, "frame_id": fid, "method": "spart",
            "n_vehicles": len(fv),
            "scan_ms": scan_ms,
            "full_ms": full_ms,
            "candidates": tc,
            "muted_ratio": muted,
        })
    if verbose:
        print(f"  [SPART]       done {time.perf_counter()-t0_sp:.1f}s | "
              f"mean scan={np.mean(spart_scan_ms):.4f}ms "
              f"full={np.mean(spart_full_ms):.4f}ms  "
              f"muted={100*np.mean(spart_muted):.1f}%")

    # ---- Per-CSV summary ----
    jit_mean_scan   = float(np.mean(jit_scan_ms))
    jit_mean_full   = float(np.mean(jit_full_ms))
    sp_mean_scan    = float(np.mean(spart_scan_ms))
    sp_mean_full    = float(np.mean(spart_full_ms))
    sp_mean_cand    = float(np.mean(spart_cand))
    sp_mean_muted   = float(np.mean(spart_muted))
    jit_mean_cand   = float(np.mean(jit_cand))
    # SEI computed from aggregate means (avoids per-frame inflation when candidates=0)
    sp_mean_sei     = float(K * mean_n_veh * 4) / max(1.0, sp_mean_cand)

    csv_summary = {
        "csv":            csv_name,
        "n_frames":       len(frame_list),
        "K_beams":        K,
        "mean_n_veh":     mean_n_veh,
        "jit_vanilla": {
            "mean_scan_ms":  jit_mean_scan,
            "mean_full_ms":  jit_mean_full,
            "scan_hz":       1000.0 / max(jit_mean_scan, 1e-9),
            "full_hz":       1000.0 / max(jit_mean_full, 1e-9),
            "mean_candidates": jit_mean_cand,
        },
        "spart": {
            "mean_scan_ms":  sp_mean_scan,
            "mean_full_ms":  sp_mean_full,
            "scan_hz":       1000.0 / max(sp_mean_scan, 1e-9),
            "full_hz":       1000.0 / max(sp_mean_full, 1e-9),
            "mean_candidates": sp_mean_cand,
            "mean_sei":      sp_mean_sei,
            "muting_pct":    100.0 * sp_mean_muted,
        },
        "speedup_scan":  (1000.0 / max(sp_mean_scan, 1e-9)) / max(1000.0 / max(jit_mean_scan, 1e-9), 1e-9),
        "speedup_full":  (1000.0 / max(sp_mean_full, 1e-9)) / max(1000.0 / max(jit_mean_full, 1e-9), 1e-9),
    }
    return (jit_rows, spart_rows,
            jit_scan_ms, jit_full_ms,
            spart_scan_ms, spart_full_ms,
            csv_summary)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _csv_labels(summaries):
    return [s["csv"].replace("vehicle_tracks_", "CSV_").replace(".csv", "")
            for s in summaries]


def fig_full_hz(summaries, out_dir):
    if not _HAS_MPL:
        return
    labels  = _csv_labels(summaries) + ["Mean"]
    jit_hz  = [s["jit_vanilla"]["full_hz"] for s in summaries]
    sp_hz   = [s["spart"]["full_hz"] for s in summaries]
    jit_hz.append(float(np.mean(jit_hz)))
    sp_hz.append(float(np.mean(sp_hz)))

    x     = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, jit_hz, width, color=_COL_JIT,  alpha=0.85,
           label="JIT vanilla", edgecolor="k", lw=0.5)
    ax.bar(x + width/2, sp_hz,  width, color=_COL_SPART, alpha=0.85,
           label="SPART full",  edgecolor="k", lw=0.5)

    for i, (j, s) in enumerate(zip(jit_hz, sp_hz)):
        sp_val = s / max(j, 1e-9)
        ax.text(x[i] + width/2, s + 100,
                f"{sp_val:.1f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Full-pipeline throughput (Hz)")
    ax.set_title("Main Benchmark: Full-Pipeline Throughput\n"
                 "Annotations: speedup SPART / JIT vanilla")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, lw=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig1_full_hz_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig_scan_hz(summaries, out_dir):
    if not _HAS_MPL:
        return
    labels  = _csv_labels(summaries) + ["Mean"]
    jit_hz  = [s["jit_vanilla"]["scan_hz"] for s in summaries]
    sp_hz   = [s["spart"]["scan_hz"] for s in summaries]
    jit_hz.append(float(np.mean(jit_hz)))
    sp_hz.append(float(np.mean(sp_hz)))

    x     = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, jit_hz, width, color=_COL_JIT,  alpha=0.85,
           label="JIT vanilla", edgecolor="k", lw=0.5)
    ax.bar(x + width/2, sp_hz,  width, color=_COL_SPART, alpha=0.85,
           label="SPART full",  edgecolor="k", lw=0.5)

    for i, (j, s) in enumerate(zip(jit_hz, sp_hz)):
        sp_val = s / max(j, 1e-9)
        ax.text(x[i] + width/2, s + 500,
                f"{sp_val:.1f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Scan-kernel throughput (Hz)")
    ax.set_title("Main Benchmark: Scan-Kernel Throughput\n"
                 "Annotations: speedup SPART / JIT vanilla")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, lw=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig2_scan_hz_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig_timing_evolution(jit_full_ms, spart_full_ms, csv_name, out_dir):
    """Per-frame timing trace for one CSV (rolling mean)."""
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    window = 50

    for vals, label, color in [
        (jit_full_ms,   "JIT vanilla", _COL_JIT),
        (spart_full_ms, "SPART full",  _COL_SPART),
    ]:
        roll = np.convolve(vals, np.ones(window) / window, mode="valid")
        xs   = np.arange(window // 2, window // 2 + len(roll))
        ax.plot(xs, roll, lw=1.0, color=color, alpha=0.9, label=label)

    ax.set_ylabel("Full-pipeline time (ms)\n50-frame rolling mean")
    ax.set_xlabel("Frame index")
    ax.set_title(f"Per-Frame Timing Trace — {csv_name.replace('.csv','')}")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, lw=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig3_timing_evolution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig_speedup_summary(summaries, out_dir):
    if not _HAS_MPL:
        return
    labels   = _csv_labels(summaries) + ["Mean"]
    sp_scan  = [s["speedup_scan"] for s in summaries]
    sp_full  = [s["speedup_full"] for s in summaries]
    sp_scan.append(float(np.mean(sp_scan)))
    sp_full.append(float(np.mean(sp_full)))

    x     = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, sp_scan, width, color=_COL_SPART, alpha=0.85,
           label="Scan kernel speedup", edgecolor="k", lw=0.5)
    ax.bar(x + width/2, sp_full, width, color=_COL_SPART, alpha=0.45,
           label="Full pipeline speedup", edgecolor="k", lw=0.5, hatch="//")

    for i, (sc, sf) in enumerate(zip(sp_scan, sp_full)):
        ax.text(x[i] - width/2, sc + 0.05, f"{sc:.1f}x",
                ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width/2, sf + 0.05, f"{sf:.1f}x",
                ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup (SPART / JIT vanilla)")
    ax.set_title("Main Benchmark: SPART Speedup over JIT Vanilla")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, lw=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig4_speedup_summary.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary(all_summaries, gate_passed):
    print("\n" + "=" * 88)
    print("EXP 3  MAIN BENCHMARK SUMMARY")
    print("=" * 88)
    hdr = (f"  {'CSV':<22} {'N_veh':>6} {'Method':<14} "
           f"{'Scan Hz':>10} {'Full Hz':>10} {'Candidates':>12} "
           f"{'SEI':>8} {'Muted%':>8} {'Spdup':>7}")
    print(hdr)
    print("  " + "-" * 84)
    for s in all_summaries:
        csv_short = s["csv"].replace("vehicle_tracks_","").replace(".csv","")
        # JIT vanilla row
        j = s["jit_vanilla"]
        print(f"  {csv_short:<22} {s['mean_n_veh']:>6.2f} {'JIT vanilla':<14} "
              f"{j['scan_hz']:>10.0f} {j['full_hz']:>10.0f} "
              f"{j['mean_candidates']:>12.0f} {'--':>8} {'--':>8} {'--':>7}")
        # SPART row
        sp = s["spart"]
        print(f"  {'':<22} {'':<6} {'SPART':<14} "
              f"{sp['scan_hz']:>10.0f} {sp['full_hz']:>10.0f} "
              f"{sp['mean_candidates']:>12.0f} "
              f"{sp['mean_sei']:>8.2f} {sp['muting_pct']:>8.1f} "
              f"{s['speedup_full']:>6.1f}x")
        print()

    # Aggregate
    agg_sp_scan = float(np.mean([s["spart"]["scan_hz"]  for s in all_summaries]))
    agg_sp_full = float(np.mean([s["spart"]["full_hz"]  for s in all_summaries]))
    agg_jt_scan = float(np.mean([s["jit_vanilla"]["scan_hz"] for s in all_summaries]))
    agg_jt_full = float(np.mean([s["jit_vanilla"]["full_hz"] for s in all_summaries]))
    agg_sei     = float(np.mean([s["spart"]["mean_sei"]  for s in all_summaries]))
    agg_muted   = float(np.mean([s["spart"]["muting_pct"] for s in all_summaries]))
    agg_sp_scan_speedup = agg_sp_scan / max(agg_jt_scan, 1)
    agg_sp_full_speedup = agg_sp_full / max(agg_jt_full, 1)

    print("  " + "-" * 84)
    print(f"  {'AGGREGATE (4 CSVs)':<22} {'':<6} {'JIT vanilla':<14} "
          f"{agg_jt_scan:>10.0f} {agg_jt_full:>10.0f} {'':>12} {'':>8} {'':>8} {'':>7}")
    print(f"  {'':<22} {'':<6} {'SPART':<14} "
          f"{agg_sp_scan:>10.0f} {agg_sp_full:>10.0f} "
          f"{'':>12} {agg_sei:>8.2f} {agg_muted:>8.1f} "
          f"{agg_sp_full_speedup:>6.1f}x")
    print()
    gate_str = "PASS" if gate_passed else "FAIL"
    print(f"  Gate 1: SPART mean full-pipeline Hz >= 1000  "
          f"({agg_sp_full:.0f} Hz)  --> {gate_str}")
    core_ok = agg_sp_scan > agg_jt_scan
    print(f"  Gate 2: SPART scan Hz > JIT vanilla scan Hz  "
          f"({agg_sp_scan:.0f} > {agg_jt_scan:.0f})  --> "
          f"{'PASS' if core_ok else 'FAIL'}")
    print("=" * 88)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 3: Main Benchmark")
    parser.add_argument("--csv-dir", default=None,
                        help="Directory containing vehicle_tracks_*.csv files "
                             "(default: project root)")
    args = parser.parse_args()

    csv_dir = args.csv_dir if args.csv_dir else _ROOT
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "vehicle_tracks_0*.csv")))
    if not csv_paths:
        print(f"[ERROR] No vehicle_tracks_*.csv found in {csv_dir}")
        sys.exit(1)
    print(f"[INFO] Found {len(csv_paths)} CSV(s): "
          f"{[os.path.basename(p) for p in csv_paths]}")

    out_dir = os.path.join(_ROOT, "results", "benchmark")
    os.makedirs(out_dir, exist_ok=True)

    cfg = _load_config()

    print("\n[INFO] Warming up JIT kernels ...")
    t0 = time.perf_counter()
    warmup_numba_kernel(cfg["fov_deg"], cfg["delta_deg"], cfg["r_max_m"])
    warmup_jit_vanilla(cfg["fov_deg"], cfg["delta_deg"], cfg["r_max_m"])
    print(f"  Compiled in {time.perf_counter()-t0:.2f}s")

    print(f"\n[INFO] Running benchmark on {len(csv_paths)} CSVs "
          f"({cfg['n_frames']} frames each) ...")
    t_total = time.perf_counter()

    all_rows     = []
    all_summaries = []
    first_jit_full_ms   = None
    first_spart_full_ms = None
    first_csv_name      = None

    for csv_path in csv_paths:
        (jit_rows, spart_rows,
         jit_full_scan_ms, jit_full_full_ms,
         spart_full_scan_ms, spart_full_full_ms,
         csv_summary) = run_one_csv(csv_path, cfg)

        all_rows.extend(jit_rows)
        all_rows.extend(spart_rows)
        all_summaries.append(csv_summary)

        if first_jit_full_ms is None:
            first_jit_full_ms   = jit_full_full_ms
            first_spart_full_ms = spart_full_full_ms
            first_csv_name      = os.path.basename(csv_path)

    print(f"\n[INFO] All CSVs done in {time.perf_counter()-t_total:.1f}s")

    # Gate check
    agg_spart_full_hz = float(np.mean([s["spart"]["full_hz"] for s in all_summaries]))
    agg_spart_scan_hz = float(np.mean([s["spart"]["scan_hz"] for s in all_summaries]))
    agg_jit_scan_hz   = float(np.mean([s["jit_vanilla"]["scan_hz"] for s in all_summaries]))
    gate_passed = (agg_spart_full_hz >= 1000.0) and (agg_spart_scan_hz > agg_jit_scan_hz)

    # Aggregate summary entry
    agg_entry = {
        "n_csvs": len(all_summaries),
        "per_csv": all_summaries,
        "aggregate": {
            "spart": {
                "mean_scan_hz":  agg_spart_scan_hz,
                "mean_full_hz":  agg_spart_full_hz,
                "mean_sei":      float(np.mean([s["spart"]["mean_sei"] for s in all_summaries])),
                "mean_muting_pct": float(np.mean([s["spart"]["muting_pct"] for s in all_summaries])),
            },
            "jit_vanilla": {
                "mean_scan_hz": agg_jit_scan_hz,
                "mean_full_hz": float(np.mean([s["jit_vanilla"]["full_hz"] for s in all_summaries])),
            },
            "speedup_scan": agg_spart_scan_hz / max(agg_jit_scan_hz, 1),
            "speedup_full": agg_spart_full_hz / max(float(np.mean([s["jit_vanilla"]["full_hz"] for s in all_summaries])), 1),
        },
        "gate_passed": gate_passed,
    }

    print("\n[INFO] Saving outputs ...")
    json_out = os.path.join(out_dir, "benchmark_summary.json")
    with open(json_out, "w") as f:
        json.dump(agg_entry, f, indent=2, default=float)
    print(f"  Saved {json_out}")

    keys = list(all_rows[0].keys())
    csv_out = os.path.join(out_dir, "per_frame_benchmark.csv")
    with open(csv_out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"  Saved {csv_out}")

    print("\n[INFO] Generating figures ...")
    fig_full_hz(all_summaries, out_dir)
    fig_scan_hz(all_summaries, out_dir)
    if first_jit_full_ms is not None:
        fig_timing_evolution(first_jit_full_ms, first_spart_full_ms,
                             first_csv_name, out_dir)
    fig_speedup_summary(all_summaries, out_dir)

    print_summary(all_summaries, gate_passed)
    sys.exit(0 if gate_passed else 1)


if __name__ == "__main__":
    main()

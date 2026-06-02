"""
scripts/exp2_ablation.py
Experiment 2: Ablation Study -- isolating contribution of each SPART component.

Five conditions on vehicle_tracks_000.csv (primary CSV), n_frames=8750, identical
circular ego trajectory.

Label | Condition           | JIT | Temporal Muting | Angular Pruning
------+---------------------+-----+-----------------+----------------
  A   | Python vanilla      | No  | No              | No
  B   | JIT vanilla         | Yes | No              | No
  C   | JIT + temporal only | Yes | Yes             | No
  D   | JIT + angular only  | Yes | No              | Yes
  E   | SPART full          | Yes | Yes             | Yes

Metrics per condition:
  mean_scan_ms         -- mean scan-kernel time per frame (ms)
  mean_fullpipeline_ms -- mean full-pipeline time per frame (ms)
  mean_candidates      -- mean intersection tests per frame
  mean_sei             -- mean Scan Efficiency Index per frame
  speedup_scan_vs_A    -- mean_scan_A / mean_scan_X
  speedup_full_vs_A    -- mean_fullpipeline_A / mean_fullpipeline_X
  muted_ratio_mean     -- fraction of vehicles muted (conditions C and E only)

Gate: SEI_E >= 50x (deterministic, machine-independent; see per_condition["E"]["mean_sei"]).

Outputs (results/ablation/):
  ablation_summary.json
  per_frame_ablation.csv
  fig1_timing_bar.png          -- mean time per condition (log scale)
  fig2_candidates_bar.png      -- mean intersection tests per condition
  fig3_timing_evolution.png    -- per-frame timing for B-E over time
  fig4_sei_bar.png             -- mean SEI per condition
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
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("[WARN] matplotlib not installed -- figures will be skipped")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spart.core import SPART, _estimate_next_revisit
from spart.utils import (
    load_interaction_csv,
    make_ego_trajectory,
    get_frame_list,
    build_segments_from_frame,
)
from spart.scan_kernel import warmup_numba_kernel
from spart.vanilla_kernel import (
    scan_python_vanilla,
    scan_jit_vanilla,
    scan_jit_angular_only,
    warmup_jit_vanilla,
    warmup_jit_angular,
)

_COLORS = {
    "A": "#d62728",   # red
    "B": "#ff7f0e",   # orange
    "C": "#2ca02c",   # green
    "D": "#1f77b4",   # blue
    "E": "#9467bd",   # purple
}
_LABELS = {
    "A": "A: Python vanilla",
    "B": "B: JIT vanilla",
    "C": "C: JIT + muting",
    "D": "D: JIT + angular",
    "E": "E: SPART full",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "benchmark_config.yaml",
    )
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Condition C helper (JIT + temporal muting, no angular pruning)
# ---------------------------------------------------------------------------

def _run_condition_c_frame(fv, ego_s, tcurr, eligibility,
                           fov_deg, delta_deg, r_max):
    """
    One frame of Condition C: SPART eligibility scheduling + vanilla-JIT kernel.
    Returns (scan_time_s, fullpipeline_time_s, total_candidates, muted_ratio).
    """
    t_pipe = time.perf_counter()
    ego_pos = np.array([ego_s["x"], ego_s["y"]], dtype=np.float64)
    ego_vel = np.array([ego_s["vx"], ego_s["vy"]], dtype=np.float64)

    process_veh, skip_veh = [], []
    for v in fv:
        tid = int(v["track_id"])
        (process_veh if tcurr >= eligibility.get(tid, 0.0) else skip_veh).append(v)

    if process_veh:
        segs, tids = build_segments_from_frame(process_veh)
    else:
        segs = np.zeros((0, 2, 2), dtype=np.float64)
        tids = np.array([], dtype=np.int64)

    _, _, _, tc, scan_t = scan_jit_vanilla(
        ego_s, segs, fov_deg=fov_deg, delta_deg=delta_deg,
        r_max=r_max, seg_track_ids=tids,
    )

    for v in process_veh:
        tid = int(v["track_id"])
        eligibility[tid] = _estimate_next_revisit(
            ego_pos, ego_vel,
            np.asarray(v["pos"]), np.asarray(v["vel"]),
            r_max, tcurr,
        )

    muted = len(skip_veh) / len(fv) if fv else 0.0
    return float(scan_t), time.perf_counter() - t_pipe, int(tc), muted


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(csv_path, cfg, verbose=True):
    """
    Run all 5 conditions on csv_path.  Returns (rows, summary_dict).
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

    print(f"\n  Loading {csv_name} ...")
    frames_dict, timestamps = load_interaction_csv(csv_path)
    frame_list = get_frame_list(frames_dict, n_frames=n_frames)
    ego_traj   = make_ego_trajectory(
        frame_ids=frame_list, timestamps=timestamps, mode="circular", **ego_cfg
    )
    print(f"  Frames: {len(frame_list)}, n_vehicles mean: "
          f"{np.mean([len(frames_dict[f]) for f in frame_list]):.2f}")

    # K_beams is constant (depends only on fov_deg / delta_deg)
    fov   = math.radians(fov_deg)
    delta = math.radians(delta_deg)
    K     = int(math.floor(fov / delta + 0.5)) + 1

    rows         = []   # per-frame rows (all conditions)
    conditions   = ["A", "B", "C", "D", "E"]
    timing_scan  = {c: [] for c in conditions}
    timing_full  = {c: [] for c in conditions}
    candidates   = {c: [] for c in conditions}
    muted_ratios = {c: [] for c in conditions}   # only C and E

    # ---- Condition A: pure Python vanilla ----
    print("\n  [A] Python vanilla (no JIT, no pruning, no muting) ...")
    t_a_start = time.perf_counter()
    for idx, fid in enumerate(frame_list):
        fv    = frames_dict[fid]
        ego_s = ego_traj[fid]

        t0 = time.perf_counter()
        _, _, _, tc, scan_t = scan_python_vanilla(
            ego_s, fv, fov_deg=fov_deg, delta_deg=delta_deg, r_max=r_max)
        t_full = time.perf_counter() - t0

        timing_scan["A"].append(float(scan_t))
        timing_full["A"].append(float(t_full))
        candidates["A"].append(int(tc))

        if verbose and (idx + 1) % 500 == 0:
            pct = 100.0 * (idx + 1) / len(frame_list)
            print(f"    {idx+1}/{len(frame_list)} ({pct:.0f}%) "
                  f"-- {time.perf_counter()-t_a_start:.1f}s")

    print(f"  [A] Done in {time.perf_counter()-t_a_start:.1f}s "
          f"| mean scan={1000*np.mean(timing_scan['A']):.3f}ms")

    # ---- Condition B: JIT vanilla ----
    print("\n  [B] JIT vanilla (JIT, no pruning, no muting) ...")
    t0_b = time.perf_counter()
    for fid in frame_list:
        fv    = frames_dict[fid]
        ego_s = ego_traj[fid]
        segs, tids = build_segments_from_frame(fv)

        t0 = time.perf_counter()
        _, _, _, tc, scan_t = scan_jit_vanilla(
            ego_s, segs, fov_deg=fov_deg, delta_deg=delta_deg,
            r_max=r_max, seg_track_ids=tids)
        t_full = time.perf_counter() - t0 + (time.perf_counter() - t0) * 0
        # re-measure full pipeline (build + scan)
        t0_pipe = time.perf_counter()
        segs2, tids2 = build_segments_from_frame(fv)
        _, _, _, _tc2, _st2 = scan_jit_vanilla(
            ego_s, segs2, fov_deg=fov_deg, delta_deg=delta_deg,
            r_max=r_max, seg_track_ids=tids2)
        t_full = time.perf_counter() - t0_pipe

        timing_scan["B"].append(float(scan_t))
        timing_full["B"].append(float(t_full))
        candidates["B"].append(int(tc))

    print(f"  [B] Done in {time.perf_counter()-t0_b:.1f}s "
          f"| mean scan={1000*np.mean(timing_scan['B']):.3f}ms")

    # ---- Condition C: JIT + temporal muting only ----
    print("\n  [C] JIT + temporal muting (no angular pruning) ...")
    eligibility_c = {}
    t0_c = time.perf_counter()
    for fid in frame_list:
        fv    = frames_dict[fid]
        ego_s = ego_traj[fid]
        tcurr = timestamps[fid]
        scan_t, t_full, tc, muted = _run_condition_c_frame(
            fv, ego_s, tcurr, eligibility_c, fov_deg, delta_deg, r_max)
        timing_scan["C"].append(scan_t)
        timing_full["C"].append(t_full)
        candidates["C"].append(tc)
        muted_ratios["C"].append(muted)

    print(f"  [C] Done in {time.perf_counter()-t0_c:.1f}s "
          f"| mean scan={1000*np.mean(timing_scan['C']):.3f}ms"
          f"  muted={100*np.mean(muted_ratios['C']):.1f}%")

    # ---- Condition D: JIT + angular pruning only ----
    print("\n  [D] JIT + angular pruning (no temporal muting) ...")
    t0_d = time.perf_counter()
    for fid in frame_list:
        fv    = frames_dict[fid]
        ego_s = ego_traj[fid]
        segs, tids = build_segments_from_frame(fv)

        t0_pipe = time.perf_counter()
        segs2, tids2 = build_segments_from_frame(fv)
        _, _, _, tc, _cpb, scan_t = scan_jit_angular_only(
            ego_s, segs2, fov_deg=fov_deg, delta_deg=delta_deg,
            r_max=r_max, seg_track_ids=tids2)
        t_full = time.perf_counter() - t0_pipe

        timing_scan["D"].append(float(scan_t))
        timing_full["D"].append(float(t_full))
        candidates["D"].append(int(tc))

    print(f"  [D] Done in {time.perf_counter()-t0_d:.1f}s "
          f"| mean scan={1000*np.mean(timing_scan['D']):.3f}ms")

    # ---- Condition E: Full SPART ----
    print("\n  [E] SPART full (JIT + muting + angular pruning) ...")
    spart_e = SPART({"fov_deg": fov_deg, "delta_deg": delta_deg,
                     "r_max_m": r_max, "enable_grid": False},
                    track_memory=False)
    t0_e = time.perf_counter()
    for fid in frame_list:
        fv    = frames_dict[fid]
        ego_s = ego_traj[fid]
        tcurr = timestamps[fid]

        t0_pipe = time.perf_counter()
        _, _, _, met = spart_e.run_frame(fv, ego_s, tcurr)
        t_full = time.perf_counter() - t0_pipe

        timing_scan["E"].append(float(met["time_scan_sec"]))
        timing_full["E"].append(t_full)
        candidates["E"].append(int(met["total_candidates"]))
        muted_ratios["E"].append(float(met["muted_ratio"]))

    print(f"  [E] Done in {time.perf_counter()-t0_e:.1f}s "
          f"| mean scan={1000*np.mean(timing_scan['E']):.3f}ms"
          f"  muted={100*np.mean(muted_ratios['E']):.1f}%")

    # ---- Build per-frame rows ----
    for idx, fid in enumerate(frame_list):
        row = {
            "frame_id":       fid,
            "n_vehicles":     len(frames_dict[fid]),
        }
        for c in conditions:
            row[f"{c}_scan_ms"]    = 1000.0 * timing_scan[c][idx]
            row[f"{c}_full_ms"]    = 1000.0 * timing_full[c][idx]
            row[f"{c}_candidates"] = candidates[c][idx]
        for c in ["C", "E"]:
            row[f"{c}_muted_ratio"] = muted_ratios[c][idx]
        rows.append(row)

    # ---- Aggregate summary ----
    mean_scan = {c: float(np.mean(timing_scan[c])) for c in conditions}
    mean_full = {c: float(np.mean(timing_full[c])) for c in conditions}
    mean_cand = {c: float(np.mean(candidates[c])) for c in conditions}
    mean_sei  = {c: float(K * np.mean([len(frames_dict[f]) for f in frame_list]) * 4
                          / max(1, mean_cand[c])) for c in conditions}

    per_condition = {}
    for c in conditions:
        per_condition[c] = {
            "label":               _LABELS[c],
            "mean_scan_ms":        1000.0 * mean_scan[c],
            "mean_fullpipeline_ms": 1000.0 * mean_full[c],
            "mean_candidates":     mean_cand[c],
            "mean_sei":            mean_sei[c],
            "speedup_scan_vs_A":   mean_scan["A"] / max(mean_scan[c], 1e-9),
            "speedup_full_vs_A":   mean_full["A"] / max(mean_full[c], 1e-9),
        }
        if c in ("C", "E"):
            per_condition[c]["muted_ratio_mean"] = float(np.mean(muted_ratios[c]))

    speedup_e_full = per_condition["E"]["speedup_full_vs_A"]
    speedup_e_scan = per_condition["E"]["speedup_scan_vs_A"]
    sei_e          = per_condition["E"]["mean_sei"]

    # Gate: SEI >= 50 (algorithmic test-reduction, deterministic).
    # Full-pipeline speedup is timing-dependent and bounded by Python overhead;
    # it is reported but not used as gate.  SEI captures the core claim.
    gate_passed = sei_e >= 50.0

    summary = {
        "csv":             csv_name,
        "n_frames":        len(frame_list),
        "K_beams":         K,
        "mean_n_vehicles": float(np.mean([len(frames_dict[f]) for f in frame_list])),
        "per_condition":   per_condition,
        "gate_passed":     gate_passed,
        "sei_E":           sei_e,
        "speedup_E_vs_A_scan_kernel":    speedup_e_scan,
        "speedup_E_vs_A_full_pipeline":  speedup_e_full,
    }
    return rows, summary, timing_scan, timing_full, candidates, muted_ratios


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_timing_bar(summary, out_dir):
    if not _HAS_MPL:
        return
    conditions = ["A", "B", "C", "D", "E"]
    pc = summary["per_condition"]

    scan_vals = [pc[c]["mean_scan_ms"] for c in conditions]
    full_vals = [pc[c]["mean_fullpipeline_ms"] for c in conditions]

    x     = np.arange(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_s = ax.bar(x - width/2, scan_vals, width,
                    color=[_COLORS[c] for c in conditions], alpha=0.85,
                    label="Scan kernel", edgecolor="k", lw=0.5)
    bars_f = ax.bar(x + width/2, full_vals, width,
                    color=[_COLORS[c] for c in conditions], alpha=0.45,
                    label="Full pipeline", edgecolor="k", lw=0.5, hatch="//")

    # Annotate speedup vs A
    for i, c in enumerate(conditions):
        sp = pc[c]["speedup_full_vs_A"]
        ax.text(x[i] + width/2, full_vals[i] * 1.15,
                f"{sp:.0f}x" if sp < 1000 else f"{sp:.0f}x",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_LABELS[c] for c in conditions], fontsize=8)
    ax.set_ylabel("Mean time per frame (ms, log scale)")
    ax.set_title(
        f"Ablation: Per-frame Timing by Condition\n"
        f"{summary['csv'].replace('.csv','')} | N={summary['n_frames']:,} frames"
        f"  |  Annotations: speedup vs A (full pipeline)"
    )
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, lw=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig1_timing_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig_candidates_bar(summary, out_dir):
    if not _HAS_MPL:
        return
    conditions = ["A", "B", "C", "D", "E"]
    pc         = summary["per_condition"]
    vals       = [pc[c]["mean_candidates"] for c in conditions]
    seis       = [pc[c]["mean_sei"] for c in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.bar(conditions, vals,
            color=[_COLORS[c] for c in conditions], edgecolor="k", lw=0.5, alpha=0.85)
    for i, (c, v) in enumerate(zip(conditions, vals)):
        ax1.text(i, v * 1.05, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Mean intersection tests per frame")
    ax1.set_title("Candidates (intersection tests) per condition")
    ax1.yaxis.grid(True, alpha=0.3, lw=0.4)

    ax2.bar(conditions, seis,
            color=[_COLORS[c] for c in conditions], edgecolor="k", lw=0.5, alpha=0.85)
    for i, (c, v) in enumerate(zip(conditions, seis)):
        ax2.text(i, v * 1.02, f"{v:.2f}x", ha="center", va="bottom", fontsize=8)
    ax2.axhline(1.0, color="black", lw=1, ls="--", alpha=0.5)
    ax2.set_ylabel("SEI = K x M_all / candidates")
    ax2.set_title("Scan Efficiency Index per condition")
    ax2.yaxis.grid(True, alpha=0.3, lw=0.4)

    plt.suptitle(f"Ablation: Intersection Test Reduction  "
                 f"({summary['csv'].replace('.csv','')})", y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig2_candidates_bar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def fig_timing_evolution(frame_list, timing_full, muted_ratios, summary, out_dir):
    """Per-frame full-pipeline timing for B, C, D, E (skip A -- too slow to show)."""
    if not _HAS_MPL:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax_t = axes[0]
    for c in ["B", "C", "D", "E"]:
        vals_ms = [1000.0 * v for v in timing_full[c]]
        # Rolling mean over 50 frames for readability
        window = 50
        roll = np.convolve(vals_ms, np.ones(window) / window, mode="valid")
        xs   = np.arange(window // 2, window // 2 + len(roll))
        ax_t.plot(xs, roll, lw=1.0, color=_COLORS[c], alpha=0.9, label=_LABELS[c])

    ax_t.set_ylabel("Full pipeline time (ms)\n50-frame rolling mean")
    ax_t.set_title(
        f"Per-Frame Timing Evolution  ({summary['csv'].replace('.csv','')})\n"
        f"(Conditions B-E only -- A excluded for scale; Condition A mean: "
        f"{summary['per_condition']['A']['mean_fullpipeline_ms']:.1f} ms)"
    )
    ax_t.legend(fontsize=8, loc="upper right")
    ax_t.yaxis.grid(True, alpha=0.3, lw=0.4)

    ax_m = axes[1]
    for c in ["C", "E"]:
        vals = [100.0 * v for v in muted_ratios[c]]
        roll = np.convolve(vals, np.ones(window) / window, mode="valid")
        xs   = np.arange(window // 2, window // 2 + len(roll))
        ax_m.plot(xs, roll, lw=1.0, color=_COLORS[c], alpha=0.9,
                  label=_LABELS[c])

    ax_m.set_ylabel("Muting ratio (%)\n50-frame rolling mean")
    ax_m.set_xlabel("Frame index")
    ax_m.legend(fontsize=8)
    ax_m.yaxis.grid(True, alpha=0.3, lw=0.4)

    plt.tight_layout()
    out = os.path.join(out_dir, "fig3_timing_evolution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig_sei_bar(summary, out_dir):
    """SEI and speedup combined summary figure."""
    if not _HAS_MPL:
        return
    conditions  = ["A", "B", "C", "D", "E"]
    pc          = summary["per_condition"]

    speedup_s   = [pc[c]["speedup_scan_vs_A"] for c in conditions]
    speedup_f   = [pc[c]["speedup_full_vs_A"] for c in conditions]

    x     = np.arange(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - width/2, speedup_s, width,
           color=[_COLORS[c] for c in conditions], alpha=0.85,
           label="Scan kernel speedup", edgecolor="k", lw=0.5)
    ax.bar(x + width/2, speedup_f, width,
           color=[_COLORS[c] for c in conditions], alpha=0.45,
           label="Full pipeline speedup", edgecolor="k", lw=0.5, hatch="//")

    for i, c in enumerate(conditions):
        ax.text(i - width/2, speedup_s[i] + 0.5,
                f"{speedup_s[i]:.0f}x", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, speedup_f[i] + 0.5,
                f"{speedup_f[i]:.0f}x", ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([_LABELS[c] for c in conditions], fontsize=8)
    ax.set_ylabel("Speedup vs Condition A")
    ax.set_title(
        f"Ablation: Speedup over Python Vanilla (Condition A)\n"
        f"{summary['csv'].replace('.csv','')} | N={summary['n_frames']:,} frames"
    )
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, lw=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig4_sei_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(summary):
    print("\n" + "=" * 72)
    print("EXP 2  ABLATION SUMMARY")
    print("=" * 72)
    print(f"  CSV:          {summary['csv']}")
    print(f"  N frames:     {summary['n_frames']:,}")
    print(f"  K beams:      {summary['K_beams']}")
    print(f"  Mean N_veh:   {summary['mean_n_vehicles']:.2f}")
    print()
    print(f"  {'Cond':<6} {'Scan (ms)':<12} {'Full (ms)':<12} {'Candidates':<12}"
          f" {'SEI':<8} {'Spdup-scan':<12} {'Spdup-full':<12} {'Muted'}")
    print("  " + "-" * 80)
    for c in ["A", "B", "C", "D", "E"]:
        pc = summary["per_condition"][c]
        muted = f"{pc.get('muted_ratio_mean',0)*100:.1f}%" if c in ("C","E") else "--"
        print(f"  {c:<6} {pc['mean_scan_ms']:<12.4f} "
              f"{pc['mean_fullpipeline_ms']:<12.4f} "
              f"{pc['mean_candidates']:<12.0f} "
              f"{pc['mean_sei']:<8.2f} "
              f"{pc['speedup_scan_vs_A']:<12.1f} "
              f"{pc['speedup_full_vs_A']:<12.1f} "
              f"{muted}")
    print()
    sei_e = summary["sei_E"]
    sp_s  = summary["speedup_E_vs_A_scan_kernel"]
    sp_f  = summary["speedup_E_vs_A_full_pipeline"]
    gate_s = "PASS" if summary["gate_passed"] else "FAIL (expected >= 50)"
    print(f"  Gate: SEI_E = {sei_e:.1f}x  --> {gate_s}")
    print(f"  Info: scan-kernel speedup E/A = {sp_s:.1f}x,  "
          f"full-pipeline speedup E/A = {sp_f:.1f}x")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def save_csv(rows, out_dir):
    if not rows:
        return
    keys = list(rows[0].keys())
    out  = os.path.join(out_dir, "per_frame_ablation.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 2: Ablation Study")
    parser.add_argument("--csv", default=None,
                        help="Primary CSV path (default: auto-discover vehicle_tracks_000.csv)")
    args = parser.parse_args()

    if args.csv:
        csv_path = os.path.abspath(args.csv)
    else:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates_csv = sorted(glob.glob(os.path.join(root, "vehicle_tracks_0*.csv")))
        if not candidates_csv:
            print("[ERROR] No vehicle_tracks_*.csv found. Supply --csv.")
            sys.exit(1)
        csv_path = candidates_csv[0]

    print(f"[INFO] Primary CSV: {os.path.basename(csv_path)}")

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "ablation",
    )
    os.makedirs(out_dir, exist_ok=True)

    cfg = _load_config()

    print("\n[INFO] Warming up JIT kernels ...")
    t0 = time.perf_counter()
    warmup_numba_kernel(cfg["fov_deg"], cfg["delta_deg"], cfg["r_max_m"])
    warmup_jit_vanilla(cfg["fov_deg"], cfg["delta_deg"], cfg["r_max_m"])
    warmup_jit_angular(cfg["fov_deg"], cfg["delta_deg"], cfg["r_max_m"])
    print(f"  JIT compiled in {time.perf_counter()-t0:.2f}s")

    print("\n[INFO] Running ablation (5 conditions x 8750 frames) ...")
    print("       Condition A (pure Python) will take ~40-60 seconds. Please wait.")

    t_total = time.perf_counter()
    rows, summary, timing_scan, timing_full, cand, muted_ratios = run_ablation(
        csv_path, cfg)
    print(f"\n[INFO] All conditions complete in {time.perf_counter()-t_total:.1f}s")

    # Load frame list for evolution plot
    frames_dict, timestamps = {}, {}  # already computed inside run_ablation; reload
    # (just pass timing arrays directly -- frame_list ordering matches rows)
    frame_list = [r["frame_id"] for r in rows]

    print("\n[INFO] Saving outputs ...")
    save_csv(rows, out_dir)
    json_out = os.path.join(out_dir, "ablation_summary.json")
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  Saved {json_out}")

    print("\n[INFO] Generating figures ...")
    fig_timing_bar(summary, out_dir)
    fig_candidates_bar(summary, out_dir)
    fig_timing_evolution(frame_list, timing_full, muted_ratios, summary, out_dir)
    fig_sei_bar(summary, out_dir)

    print_summary(summary)
    sys.exit(0 if summary["gate_passed"] else 1)


if __name__ == "__main__":
    main()

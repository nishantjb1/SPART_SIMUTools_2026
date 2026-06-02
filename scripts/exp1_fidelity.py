"""
scripts/exp1_fidelity.py
Experiment 1: Fidelity Validation -- SPART vs JIT-vanilla (Condition B).

Two claims verified for each CSV:
  (1) false_positive_rate == 0.0
        For processed (non-muted) vehicles SPART never reports a hit that
        vanilla-JIT misses.  By design: same ray-segment math; angular pruning
        only reduces candidates, never adds spurious intersections; processed
        vehicles are a strict subset of all vehicles.
  (2) max_range_error == 0.0
        For beams where both methods detect a vehicle, the returned ranges are
        bit-identical.  Same intersection formula, same ego pose, same segment.

Documents:
  false_negative_rate -- fraction of vanilla-JIT hits that SPART misses due
  to temporal muting (eligibility scheduling).
  All FNs are attributed to muted vehicles and their distances logged.

Gate: gate passes only if (1) AND (2) hold on ALL CSVs.

Usage:
    python scripts/exp1_fidelity.py [--csvs path1.csv path2.csv ...]
    Omit --csvs to auto-discover vehicle_tracks_*.csv in the project root.

Outputs (in results/fidelity/):
    fidelity_summary.json       -- aggregate stats + gate verdict
    per_frame_fidelity.csv      -- per-frame metrics for all sampled frames
    fig1_range_scatter.png      -- SPART vs vanilla range (should be y=x)
    fig2_error_histogram.png    -- |SPART - vanilla| distribution (all zeros)
    fig3_fn_rate_per_frame.png  -- false-negative rate over time
    fig4_muted_ratio_per_frame.png -- muting rate evolution
    fig5_example_frame.png      -- bird's-eye scan visualisation
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

# Non-interactive matplotlib: avoid GUI window popups on Windows
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

from spart.core import SPART
from spart.utils import (
    load_interaction_csv,
    make_ego_trajectory,
    get_frame_list,
    build_segments_from_frame,
)
from spart.scan_kernel import warmup_numba_kernel
from spart.vanilla_kernel import scan_jit_vanilla, warmup_jit_vanilla

try:
    from spart.utils import get_vehicle_corners as _get_corners
    _HAS_CORNERS = True
except (ImportError, AttributeError):
    _HAS_CORNERS = False

# ---------------------------------------------------------------------------
# Colour palette (blue, orange, green, red -- colour-blind safe-ish)
# ---------------------------------------------------------------------------
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


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
# Per-CSV fidelity run
# ---------------------------------------------------------------------------

def run_fidelity_on_csv(csv_path, cfg, verbose=True):
    """
    Run SPART continuously for n_frames; on 500 evenly-sampled frames also
    run vanilla-JIT and compare beam-by-beam.

    Returns (rows, viz_data):
        rows     -- list of per-frame metric dicts
        viz_data -- dict with data for the bird's-eye figure (one frame)
    """
    csv_name   = os.path.basename(csv_path)
    n_frames   = int(cfg["n_frames"])
    warmup_n   = int(cfg["fidelity_warmup_frames"])
    n_sample   = int(cfg["fidelity_n_sample_frames"])
    fov_deg    = float(cfg["fov_deg"])
    delta_deg  = float(cfg["delta_deg"])
    r_max      = float(cfg["r_max_m"])
    ego_cfg    = {
        "radius": float(cfg["ego_circle_radius_m"]),
        "speed":  float(cfg["ego_circle_speed_mps"]),
        "center": tuple(cfg["ego_circle_center"]),
    }

    if verbose:
        print(f"\n  [CSV] {csv_name}")

    frames_dict, timestamps = load_interaction_csv(csv_path)
    frame_list = get_frame_list(frames_dict, n_frames=n_frames)
    ego_traj   = make_ego_trajectory(
        frame_ids=frame_list, timestamps=timestamps, mode="circular", **ego_cfg
    )

    # Sample n_sample frames evenly from [warmup_n, end)
    remaining   = frame_list[warmup_n:]
    stride      = max(1, len(remaining) // n_sample)
    sample_fids = remaining[::stride][:n_sample]
    sample_set  = set(sample_fids)
    viz_idx     = len(sample_fids) // 4   # capture frame ~25% through

    if verbose:
        print(f"  Frames in file: {len(frames_dict)}, using n_frames={n_frames}")
        print(f"  Warmup: {warmup_n}, Sampling: {len(sample_fids)} "
              f"(stride={stride})")

    spart = SPART({
        "fov_deg":    fov_deg,
        "delta_deg":  delta_deg,
        "r_max_m":    r_max,
        "enable_grid": False,
    })

    rows         = []
    viz_data     = None
    sample_count = 0
    t_start      = time.perf_counter()

    for fid in frame_list:
        fv     = frames_dict[fid]
        ego_s  = ego_traj[fid]
        tcurr  = timestamps[fid]

        # Capture which vehicles are muted BEFORE the SPART call
        if fid in sample_set:
            muted_pre = {
                int(v["track_id"]) for v in fv
                if tcurr < spart._eligibility.get(int(v["track_id"]), 0.0)
            }

        # Run SPART (maintains eligibility state continuously)
        ang_s, rng_s, ht_s, met_s = spart.run_frame(fv, ego_s, tcurr)

        if fid not in sample_set:
            continue

        # ----- Run vanilla-JIT on PROCESSED vehicles only (same kernel input as SPART) -----
        # This is the geometric correctness reference: same ego, same segments -> must agree.
        proc_vehs  = [v for v in fv if int(v["track_id"]) not in muted_pre]
        segs_proc, tids_proc = build_segments_from_frame(proc_vehs)
        ang_vp, rng_vp, ht_vp, _tc_p, _tv_p = scan_jit_vanilla(
            ego_s, segs_proc,
            fov_deg=fov_deg, delta_deg=delta_deg, r_max=r_max,
            seg_track_ids=tids_proc,
        )

        # ----- Run vanilla-JIT on ALL vehicles (for FN / muting impact analysis) -----
        segs_all, tids_all = build_segments_from_frame(fv)
        ang_va, rng_va, ht_va, _tc_a, _tv_a = scan_jit_vanilla(
            ego_s, segs_all,
            fov_deg=fov_deg, delta_deg=delta_deg, r_max=r_max,
            seg_track_ids=tids_all,
        )

        K = int(ang_s.shape[0])

        # ----- Geometric correctness: SPART vs vanilla-on-processed -----
        # Both see the exact same segments; results must be bit-identical.
        spart_hit   = ~np.isnan(rng_s)
        vproc_hit   = ~np.isnan(rng_vp)
        both_proc   = spart_hit & vproc_hit
        # FP: SPART reports a hit that vanilla (on same input) misses -> impossible
        fp_mask     = spart_hit & ~vproc_hit
        # Angular-pruning miss: vanilla-on-processed hits but SPART misses -> pruning bug
        ap_miss     = ~spart_hit & vproc_hit

        n_spart   = int(np.sum(spart_hit))
        n_vproc   = int(np.sum(vproc_hit))
        n_both    = int(np.sum(both_proc))
        n_fp      = int(np.sum(fp_mask))
        n_ap_miss = int(np.sum(ap_miss))

        # Range error on both-hit beams: expected exactly 0.0
        if n_both > 0:
            errs           = np.abs(rng_s[both_proc] - rng_vp[both_proc])
            max_range_err  = float(np.max(errs))
            mean_range_err = float(np.mean(errs))
        else:
            max_range_err = mean_range_err = 0.0

        # ----- FN analysis: vanilla-all vs vanilla-processed (muting impact) -----
        vall_hit  = ~np.isnan(rng_va)
        fn_mask   = vall_hit & ~spart_hit  # vanilla-all hit but SPART missed
        n_fn      = int(np.sum(fn_mask))
        n_vanilla = int(np.sum(vall_hit))

        ego_pos    = np.array([ego_s["x"], ego_s["y"]])
        veh_by_tid = {int(v["track_id"]): v for v in fv}
        n_fn_muted = 0
        fn_dists   = []

        for k in range(K):
            if not fn_mask[k]:
                continue
            tid = int(ht_va[k])
            if tid < 0 or tid not in veh_by_tid:
                continue
            v    = veh_by_tid[tid]
            dist = float(np.linalg.norm(np.asarray(v["pos"]) - ego_pos))
            fn_dists.append(dist)
            if tid in muted_pre:
                n_fn_muted += 1

        row = {
            "csv":                   csv_name,
            "frame_id":              fid,
            "sample_idx":            sample_count,
            "K_beams":               K,
            "spart_hits":            n_spart,
            "vanilla_proc_hits":     n_vproc,
            "vanilla_all_hits":      n_vanilla,
            "both_proc_hits":        n_both,
            # Gate metrics
            "fp_beams":              n_fp,
            "angular_pruning_misses": n_ap_miss,
            "false_positive_rate":   n_fp / K if K > 0 else 0.0,
            "angular_miss_rate":     n_ap_miss / n_vproc if n_vproc > 0 else 0.0,
            "max_range_error_m":     max_range_err,
            "mean_range_error_m":    mean_range_err,
            # Muting / FN metrics
            "fn_beams":              n_fn,
            "false_negative_rate":   n_fn / n_vanilla if n_vanilla > 0 else 0.0,
            "n_vehicles":            met_s["n_vehicles"],
            "n_processed":           met_s["n_processed"],
            "n_muted":               met_s["n_skipped"],
            "muted_ratio":           met_s["muted_ratio"],
            "n_fn_due_muting":       n_fn_muted,
            "fn_total_attributed":   len(fn_dists),
            "fn_mean_dist_m":        float(np.mean(fn_dists)) if fn_dists else 0.0,
            "fn_max_dist_m":         float(np.max(fn_dists)) if fn_dists else 0.0,
            # Internal arrays for scatter / histogram plots
            "_rng_s_both":           rng_s[both_proc].tolist(),
            "_rng_vp_both":          rng_vp[both_proc].tolist(),
        }
        rows.append(row)

        # Capture one frame for the bird's-eye visualisation
        if sample_count == viz_idx:
            viz_data = {
                "csv_name":  csv_name,
                "fid":       fid,
                "fv":        fv,
                "ego_s":     ego_s,
                "ang_s":     ang_s.copy(),
                "rng_s":     rng_s.copy(),
                "rng_va":    rng_va.copy(),
                "fp_mask":   fp_mask.copy(),
                "fn_mask":   fn_mask.copy(),
                "both_hit":  both_proc.copy(),
                "muted_pre": muted_pre,
                "r_max":     r_max,
                "fov_deg":   fov_deg,
                "n_fp":      n_fp,
                "n_fn":      n_fn,
                "n_both":    n_both,
            }

        sample_count += 1
        if verbose and sample_count % 100 == 0:
            pct = 100.0 * sample_count / len(sample_fids)
            print(f"    {sample_count}/{len(sample_fids)} ({pct:.0f}%) "
                  f"-- {time.perf_counter()-t_start:.1f}s")

    if verbose:
        print(f"  Done: {len(rows)} frames analyzed in "
              f"{time.perf_counter()-t_start:.1f}s")
    return rows, viz_data


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(all_rows):
    """Return per-CSV and overall summary dicts."""
    if not all_rows:
        return {"per_csv": {}, "overall": {}, "gate_passed": False}

    from collections import defaultdict
    by_csv = defaultdict(list)
    for r in all_rows:
        by_csv[r["csv"]].append(r)

    per_csv       = {}
    g_fp          = 0
    g_fn          = 0
    g_vanilla     = 0
    g_both        = 0
    g_max_err     = 0.0
    g_fn_rates    = []
    g_muted       = []

    for cname, rows in by_csv.items():
        fp_tot   = sum(r["fp_beams"] for r in rows)
        fn_tot   = sum(r["fn_beams"] for r in rows)
        v_tot    = sum(r["vanilla_all_hits"] for r in rows)
        b_tot    = sum(r["both_proc_hits"] for r in rows)
        k_tot    = sum(r["K_beams"] for r in rows)
        ap_tot   = sum(r["angular_pruning_misses"] for r in rows)
        max_err  = max((r["max_range_error_m"] for r in rows), default=0.0)
        fn_m     = sum(r["n_fn_due_muting"] for r in rows)
        fn_rates  = [r["false_negative_rate"] for r in rows]
        muted     = [r["muted_ratio"] for r in rows]

        per_csv[cname] = {
            "n_frames_sampled":          len(rows),
            "total_fp_beams":            fp_tot,
            "total_angular_pruning_misses": ap_tot,
            "total_fn_beams":            fn_tot,
            "total_vanilla_all_hits":    v_tot,
            "total_both_proc_hits":      b_tot,
            "false_positive_rate":       fp_tot / k_tot if k_tot else 0.0,
            "angular_miss_rate":         ap_tot / max(1, sum(r["vanilla_proc_hits"] for r in rows)),
            "false_negative_rate_mean":  float(np.mean(fn_rates)) if fn_rates else 0.0,
            "false_negative_rate_max":   float(np.max(fn_rates)) if fn_rates else 0.0,
            "max_range_error_m":         max_err,
            "mean_muted_ratio":          float(np.mean(muted)) if muted else 0.0,
            "fn_due_muting_pct":         100.0 * fn_m / fn_tot if fn_tot else 0.0,
            "gate_fp_pass":              fp_tot == 0,
            "gate_ap_pass":              ap_tot == 0,
            "gate_err_pass":             max_err == 0.0,
        }
        g_fp      += fp_tot
        g_fn      += fn_tot
        g_vanilla += v_tot
        g_both    += b_tot
        g_max_err  = max(g_max_err, max_err)
        g_fn_rates.extend(fn_rates)
        g_muted.extend(muted)

    k_all       = sum(r["K_beams"] for r in all_rows)
    g_ap        = sum(r["angular_pruning_misses"] for r in all_rows)
    gate_passed = (g_fp == 0 and g_ap == 0 and g_max_err == 0.0)

    overall = {
        "n_csvs":                    len(by_csv),
        "n_frames_total":            len(all_rows),
        "total_fp_beams":            g_fp,
        "total_angular_pruning_misses": g_ap,
        "total_fn_beams":            g_fn,
        "total_vanilla_all_hits":    g_vanilla,
        "total_both_proc_hits":      g_both,
        "false_positive_rate":       g_fp / k_all if k_all else 0.0,
        "false_negative_rate_mean":  float(np.mean(g_fn_rates)) if g_fn_rates else 0.0,
        "false_negative_rate_max":   float(np.max(g_fn_rates)) if g_fn_rates else 0.0,
        "max_range_error_m":         g_max_err,
        "mean_muted_ratio":          float(np.mean(g_muted)) if g_muted else 0.0,
        "gate_passed":               gate_passed,
    }
    return {"per_csv": per_csv, "overall": overall, "gate_passed": gate_passed}


# ---------------------------------------------------------------------------
# Figure 1 -- Range scatter
# ---------------------------------------------------------------------------

def fig_range_scatter(all_rows, out_dir):
    if not _HAS_MPL:
        return
    csv_names = sorted(set(r["csv"] for r in all_rows))
    fig, ax   = plt.subplots(figsize=(6, 6))

    all_s, all_v = [], []  # SPART ranges and vanilla-on-processed ranges
    patches      = []

    for ci, cname in enumerate(csv_names):
        rs_cat = []
        rv_cat = []
        for r in all_rows:
            if r["csv"] == cname:
                rs_cat.extend(r["_rng_s_both"])
                rv_cat.extend(r["_rng_vp_both"])
        if not rs_cat:
            continue
        all_s.extend(rs_cat)
        all_v.extend(rv_cat)
        col = _COLORS[ci % len(_COLORS)]
        ax.scatter(rv_cat, rs_cat, s=0.5, alpha=0.1, color=col,
                   rasterized=True)
        patches.append(mpatches.Patch(color=col,
                                      label=cname.replace(".csv", "")))

    if not all_s:
        plt.close()
        return

    lo = min(min(all_s), min(all_v))
    hi = max(max(all_s), max(all_v))
    ax.plot([lo, hi], [lo, hi], "r-", lw=1.5)
    patches.append(plt.Line2D([0], [0], color="r", lw=1.5, label="y = x"))

    max_err = max(abs(s - v) for s, v in zip(all_s, all_v))
    ax.set_xlabel("Vanilla-JIT range (m)")
    ax.set_ylabel("SPART range (m)")
    ax.set_title(
        f"SPART vs Vanilla-JIT Range Agreement\n"
        f"N = {len(all_s):,} beam-pairs across {len(csv_names)} CSVs"
        f"  |  max_error = {max_err:.2e} m"
    )
    ax.legend(handles=patches, fontsize=8, markerscale=6)
    ax.set_aspect("equal")
    plt.tight_layout()
    out = os.path.join(out_dir, "fig1_range_scatter.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 -- Range error histogram
# ---------------------------------------------------------------------------

def fig_error_histogram(all_rows, out_dir):
    if not _HAS_MPL:
        return
    errors = []
    for r in all_rows:
        for s, v in zip(r["_rng_s_both"], r["_rng_vp_both"]):
            errors.append(abs(s - v))

    fig, ax = plt.subplots(figsize=(7, 4))
    if errors:
        max_e = max(errors)
        if max_e == 0.0:
            ax.bar(0, len(errors), width=0.0005, color="#1f77b4", edgecolor="k")
            ax.set_xlim(-0.002, 0.05)
            ax.text(0.002, len(errors) * 0.85,
                    f"All {len(errors):,} errors = 0.000 m",
                    fontsize=11, fontweight="bold", color="green")
        else:
            ax.hist(errors, bins=60, color="#1f77b4", edgecolor="k", lw=0.4)
            ax.set_yscale("log")
    ax.set_xlabel("|SPART - Vanilla| range (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"Range Error Distribution  (N = {len(errors):,} beam-pairs)")
    plt.tight_layout()
    out = os.path.join(out_dir, "fig2_error_histogram.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 -- FN rate per sampled frame
# ---------------------------------------------------------------------------

def fig_fn_rate(all_rows, out_dir):
    if not _HAS_MPL:
        return
    csv_names = sorted(set(r["csv"] for r in all_rows))
    n = len(csv_names)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows),
                              squeeze=False)

    for ci, cname in enumerate(csv_names):
        ax    = axes[ci // ncols][ci % ncols]
        rows  = [r for r in all_rows if r["csv"] == cname]
        idxs  = [r["sample_idx"] for r in rows]
        fn_r  = [100.0 * r["false_negative_rate"] for r in rows]
        col   = _COLORS[ci % len(_COLORS)]
        ax.plot(idxs, fn_r, lw=0.7, alpha=0.8, color=col)
        if fn_r:
            mean_fn = float(np.mean(fn_r))
            ax.axhline(mean_fn, color=col, lw=1.5, ls="--",
                       label=f"mean = {mean_fn:.2f}%")
        ax.set_ylabel("FN rate (%)")
        ax.set_title(cname.replace(".csv", ""))
        ax.legend(fontsize=8)

    for ci in range(n, nrows * ncols):
        axes[ci // ncols][ci % ncols].set_visible(False)

    for r in range(nrows):
        axes[r][-1].set_xlabel("Sample index") if axes[r][-1].get_visible() else None

    fig.suptitle(
        "False Negative Rate per Sampled Frame\n"
        "(Misses caused by temporal eligibility muting -- expected < 5%)",
        y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "fig3_fn_rate_per_frame.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 -- Muted ratio per sampled frame
# ---------------------------------------------------------------------------

def fig_muted_ratio(all_rows, out_dir):
    if not _HAS_MPL:
        return
    csv_names = sorted(set(r["csv"] for r in all_rows))
    n = len(csv_names)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows),
                              squeeze=False)

    for ci, cname in enumerate(csv_names):
        ax    = axes[ci // ncols][ci % ncols]
        rows  = [r for r in all_rows if r["csv"] == cname]
        idxs  = [r["sample_idx"] for r in rows]
        muted = [100.0 * r["muted_ratio"] for r in rows]
        col   = _COLORS[ci % len(_COLORS)]
        ax.plot(idxs, muted, lw=0.7, alpha=0.8, color=col)
        if muted:
            m = float(np.mean(muted))
            ax.axhline(m, color=col, lw=1.5, ls="--", label=f"mean = {m:.1f}%")
        ax.set_ylabel("Muted ratio (%)")
        ax.set_title(cname.replace(".csv", ""))
        ax.legend(fontsize=8)

    for ci in range(n, nrows * ncols):
        axes[ci // ncols][ci % ncols].set_visible(False)

    for r in range(nrows):
        axes[r][-1].set_xlabel("Sample index") if axes[r][-1].get_visible() else None

    fig.suptitle(
        "Temporal Muting Rate per Sampled Frame\n"
        "(Vehicles deferred by eligibility scheduler)",
        y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "fig4_muted_ratio_per_frame.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5 -- Bird's-eye example frames
# ---------------------------------------------------------------------------

def fig_example_frames(viz_list, out_dir):
    if not _HAS_MPL or not viz_list:
        return
    n     = len(viz_list)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(9 * ncols, 9 * nrows),
                              squeeze=False)

    for ci, vd in enumerate(viz_list):
        _draw_bird_eye(axes[ci // ncols][ci % ncols], vd)

    for ci in range(n, nrows * ncols):
        axes[ci // ncols][ci % ncols].set_visible(False)

    plt.tight_layout()
    out = os.path.join(out_dir, "fig5_example_frame.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def _draw_bird_eye(ax, vd):
    fv        = vd["fv"]
    ego_s     = vd["ego_s"]
    ang_s     = vd["ang_s"]
    rng_s     = vd["rng_s"]
    rng_v     = vd["rng_va"]   # vanilla-all for FN visualisation
    fp_mask   = vd["fp_mask"]
    fn_mask   = vd["fn_mask"]
    both_hit  = vd["both_hit"]
    muted_pre = vd["muted_pre"]
    r_max     = vd["r_max"]
    fov_deg   = vd["fov_deg"]
    n_fp      = vd["n_fp"]
    n_fn      = vd["n_fn"]
    n_both    = vd["n_both"]

    ex = float(ego_s["x"])
    ey = float(ego_s["y"])
    ep = float(ego_s["psi_rad"])

    # FOV wedge
    wedge = mpatches.Wedge(
        (ex, ey), r_max,
        theta1=math.degrees(ep) - fov_deg / 2,
        theta2=math.degrees(ep) + fov_deg / 2,
        alpha=0.06, color="gray", zorder=0,
    )
    ax.add_patch(wedge)

    # r_max circle
    ax.add_patch(plt.Circle((ex, ey), r_max, fill=False,
                             ls="--", lw=0.8, color="black", alpha=0.35, zorder=1))

    # Scan rays (hit beams only)
    for k in range(len(ang_s)):
        th = ang_s[k]
        dx = math.cos(th)
        dy = math.sin(th)
        if both_hit[k]:
            r = float(rng_s[k])
            ax.plot([ex, ex + r * dx], [ey, ey + r * dy],
                    color="#1f77b4", lw=0.5, alpha=0.6, zorder=2)
        elif fn_mask[k] and not math.isnan(rng_v[k]):
            r = float(rng_v[k])
            ax.plot([ex, ex + r * dx], [ey, ey + r * dy],
                    color="#ff7f0e", lw=0.8, alpha=0.85, ls="--", zorder=2)
        elif fp_mask[k] and not math.isnan(rng_s[k]):
            r = float(rng_s[k])
            ax.plot([ex, ex + r * dx], [ey, ey + r * dy],
                    color="red", lw=1.5, alpha=1.0, zorder=3)

    # Vehicles
    for v in fv:
        tid    = int(v["track_id"])
        vx     = float(v["pos"][0])
        vy     = float(v["pos"][1])
        muted  = tid in muted_pre
        color  = "#ff7f0e" if muted else "#2ca02c"
        marker = "^" if muted else "o"

        if _HAS_CORNERS:
            try:
                corners = _get_corners(
                    v["pos"], float(v["yaw"]),
                    float(v["L"]), float(v["W"]),
                )
                px = list(corners[:, 0]) + [corners[0, 0]]
                py = list(corners[:, 1]) + [corners[0, 1]]
                ax.fill(px, py, alpha=0.25, color=color, zorder=4)
                ax.plot(px, py, "-", color=color, lw=0.7, zorder=5)
            except Exception:
                ax.plot(vx, vy, marker, color=color, ms=5, zorder=4)
        else:
            ax.plot(vx, vy, marker, color=color, ms=5, zorder=4)

    # Ego
    ax.plot(ex, ey, "k*", ms=14, zorder=6)

    legend_items = [
        plt.Line2D([0], [0], color="#1f77b4", lw=1.2,
                   label=f"Both agree ({n_both} beams)"),
        plt.Line2D([0], [0], color="#ff7f0e", lw=1.2, ls="--",
                   label=f"FN: vanilla only ({n_fn} beams)"),
        mpatches.Patch(color="#2ca02c", alpha=0.5, label="Processed veh."),
        mpatches.Patch(color="#ff7f0e", alpha=0.5, label="Muted veh."),
    ]
    if n_fp > 0:
        legend_items.insert(
            2, plt.Line2D([0], [0], color="red", lw=1.5,
                          label=f"FP: SPART only ({n_fp} beams)"))
    ax.legend(handles=legend_items, fontsize=7, loc="upper right")
    ax.set_aspect("equal")
    margin = r_max * 1.35
    ax.set_xlim(ex - margin, ex + margin)
    ax.set_ylim(ey - margin, ey + margin)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"{vd['csv_name'].replace('.csv','')} | frame {vd['fid']}\n"
        f"FP = {n_fp}  |  FN = {n_fn}  |  agree = {n_both}"
    )
    ax.grid(True, alpha=0.25, lw=0.4)


# ---------------------------------------------------------------------------
# Save per-frame CSV
# ---------------------------------------------------------------------------

def save_csv(all_rows, out_dir):
    if not all_rows:
        return
    keys = [k for k in all_rows[0].keys() if not k.startswith("_")]
    out  = os.path.join(out_dir, "per_frame_fidelity.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row[k] for k in keys})
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(summary):
    ov = summary["overall"]
    print("\n" + "=" * 72)
    print("EXP 1  FIDELITY SUMMARY")
    print("=" * 72)
    print(f"  CSVs processed         : {ov['n_csvs']}")
    print(f"  Frames sampled (total) : {ov['n_frames_total']:,}")
    print(f"  Both-proc beam-pairs   : {ov['total_both_proc_hits']:,}")
    print(f"  FP beams (impossible)  : {ov['total_fp_beams']}"
          f"  (rate = {ov['false_positive_rate']:.2e})")
    print(f"  Angular-pruning misses : {ov['total_angular_pruning_misses']}"
          f"  (gate: must be 0)")
    print(f"  FN beams (from muting) : {ov['total_fn_beams']:,}")
    print(f"  FN rate  mean / max    : "
          f"{ov['false_negative_rate_mean']*100:.3f}% / "
          f"{ov['false_negative_rate_max']*100:.3f}%")
    print(f"  Max range error        : {ov['max_range_error_m']:.2e} m")
    print(f"  Mean muting rate       : {ov['mean_muted_ratio']*100:.1f}%")
    print()
    print("  Per-CSV gate  (FP=0  and  max_err=0):")
    for cname, cs in summary["per_csv"].items():
        fp_s  = "PASS" if cs["gate_fp_pass"]  else "FAIL"
        ap_s  = "PASS" if cs["gate_ap_pass"]  else "FAIL"
        err_s = "PASS" if cs["gate_err_pass"] else "FAIL"
        print(f"    {cname:30s}  FP={fp_s}  AP={ap_s}  err={err_s}"
              f"  | FN_mean={cs['false_negative_rate_mean']*100:.2f}%"
              f"  muted={cs['mean_muted_ratio']*100:.1f}%"
              f"  FN_from_muting={cs['fn_due_muting_pct']:.1f}%")
    print()
    verdict = "GATE PASSED" if summary["gate_passed"] else "GATE FAILED"
    print(f"  {verdict}  (gate = FP==0 AND angular_pruning_misses==0 AND max_err==0.0)")
    if summary["gate_passed"]:
        print("")
    else:
        ov = summary["overall"]
        if ov["total_fp_beams"] > 0:
            print("  --> FP > 0 (SPART hit but vanilla-on-processed missed)")
        if ov["total_angular_pruning_misses"] > 0:
            print("  --> Angular pruning misses (SPART missed a processed vehicle)")
        if ov["max_range_error_m"] > 0:
            print("  --> Range mismatch on same-segment pairs")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 1: Fidelity Validation")
    parser.add_argument(
        "--csvs", nargs="+",
        help="CSV paths; default: auto-discover vehicle_tracks_*.csv in project root",
    )
    args = parser.parse_args()

    if args.csvs:
        csv_paths = [os.path.abspath(p) for p in args.csvs]
    else:
        root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_paths = sorted(glob.glob(os.path.join(root, "vehicle_tracks_*.csv")))

    if not csv_paths:
        print("[ERROR] No CSVs found. Supply --csvs or place vehicle_tracks_*.csv "
              "in project root.")
        sys.exit(1)

    print(f"[INFO] CSVs to process: {[os.path.basename(p) for p in csv_paths]}")

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "fidelity",
    )
    os.makedirs(out_dir, exist_ok=True)

    cfg = _load_config()

    # JIT warmup (one-time cost, reported separately from per-frame timing)
    print("\n[INFO] Warming up JIT kernels ...")
    t0 = time.perf_counter()
    warmup_numba_kernel(cfg["fov_deg"], cfg["delta_deg"], cfg["r_max_m"])
    warmup_jit_vanilla(cfg["fov_deg"], cfg["delta_deg"], cfg["r_max_m"])
    print(f"  JIT compiled in {time.perf_counter()-t0:.2f}s")

    # Run fidelity on each CSV
    print("\n[INFO] Running fidelity experiment ...")
    all_rows  = []
    viz_list  = []
    total_t   = time.perf_counter()

    for csv_path in csv_paths:
        rows, vd = run_fidelity_on_csv(csv_path, cfg)
        all_rows.extend(rows)
        if vd is not None:
            viz_list.append(vd)

    print(f"\n[INFO] All CSVs done in {time.perf_counter()-total_t:.1f}s total")

    # Aggregate
    summary = aggregate(all_rows)

    # Save outputs
    print("\n[INFO] Saving outputs ...")
    save_csv(all_rows, out_dir)
    json_out = os.path.join(out_dir, "fidelity_summary.json")
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  Saved {json_out}")

    # Figures
    print("\n[INFO] Generating figures ...")
    fig_range_scatter(all_rows, out_dir)
    fig_error_histogram(all_rows, out_dir)
    fig_fn_rate(all_rows, out_dir)
    fig_muted_ratio(all_rows, out_dir)
    fig_example_frames(viz_list, out_dir)

    # Print summary + gate
    print_summary(summary)

    sys.exit(0 if summary["gate_passed"] else 1)


if __name__ == "__main__":
    main()

"""
scripts/exp4_grid_crossover.py
Experiment 4: Grid Crossover Study.

Uses SYNTHETIC scenes (no INTERACTION dataset required).
N vehicles are placed uniformly at random in a 60×60 m area, given constant
velocities, and simulated for 500 frames (dt = 0.1 s, 10 Hz).

Three methods compared per density N:
  JIT vanilla  -- build_segments(all N) + JIT scan, no state
  SPART no-grid -- muting + angular pruning, no spatial grid
  SPART grid    -- muting + angular pruning + spatial grid

Why the grid helps at high N:
  Without grid: build_segments for ALL processed vehicles, then angular pruning
  inside the kernel reduces intersection tests. But build cost is O(N_processed).
  With grid: a cell lookup first identifies which processed vehicles are in the
  FOV sector, then build_segments only for that smaller set. As N grows, the
  fraction excluded by the grid grows (only ~33% of scene is in 120° FOV), so
  the grid overhead amortises and saves build + precomputation time.

Crossover N* = smallest N where grid full_hz > nogrid full_hz.

Gate: SPART_grid full_hz > SPART_nogrid full_hz at N = 200.

Outputs (results/grid_crossover/):
  grid_crossover_summary.json
  per_condition_means.csv
  fig1_full_hz_vs_N.png         -- full-pipeline Hz vs N (main figure)
  fig2_scan_hz_vs_N.png         -- scan-kernel Hz vs N
  fig3_candidates_vs_N.png      -- mean candidates per frame vs N
  fig4_muting_vs_N.png          -- muting rate and grid pre-filter rate vs N
"""

import csv as _csv
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
    print("[WARN] matplotlib not found -- figures skipped")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spart.core import SPART
from spart.utils import (
    generate_ego_circular,
    build_segments_from_frame,
)
from spart.scan_kernel import warmup_numba_kernel
from spart.vanilla_kernel import scan_jit_vanilla, warmup_jit_vanilla

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_C_JIT   = "#ff7f0e"   # orange
_C_NOGRID = "#1f77b4"  # blue
_C_GRID   = "#2ca02c"  # green


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config():
    p = os.path.join(_ROOT, "configs", "benchmark_config.yaml")
    with open(p) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Synthetic scene generator
# ---------------------------------------------------------------------------

AREA_M   = 60.0
EGO_CTR  = (30.0, 30.0)   # centre of synthetic scene
DT       = 0.1             # seconds per frame  (10 Hz)
VEH_L    = 4.5             # vehicle length (m)
VEH_W    = 2.0             # vehicle width (m)


def _make_vehicles(n_veh: int, rng: np.random.RandomState):
    """
    N vehicles uniformly placed in [0, AREA_M]^2, random heading & speed.
    """
    cx, cy = EGO_CTR
    half = AREA_M / 2.0
    vehicles = []
    for i in range(n_veh):
        x = cx + rng.uniform(-half, half)
        y = cy + rng.uniform(-half, half)
        speed = rng.uniform(2.0, 8.0)
        yaw   = rng.uniform(-math.pi, math.pi)
        vx    = speed * math.cos(yaw)
        vy    = speed * math.sin(yaw)
        vehicles.append({
            "track_id": i + 1,
            "pos":  np.array([x, y], dtype=np.float64),
            "vel":  np.array([vx, vy], dtype=np.float64),
            "yaw":  yaw,
            "L":    VEH_L,
            "W":    VEH_W,
            "speed": speed,
        })
    return vehicles


def _step(vehicles, dt):
    """
    Advance all vehicles by dt seconds.
    Vehicles wrap around the [0, AREA_M]^2 boundary.
    """
    cx, cy = EGO_CTR
    half = AREA_M / 2.0
    for v in vehicles:
        v["pos"][0] += v["vel"][0] * dt
        v["pos"][1] += v["vel"][1] * dt
        # wrap-around
        if v["pos"][0] < cx - half:
            v["pos"][0] += AREA_M
        elif v["pos"][0] > cx + half:
            v["pos"][0] -= AREA_M
        if v["pos"][1] < cy - half:
            v["pos"][1] += AREA_M
        elif v["pos"][1] > cy + half:
            v["pos"][1] -= AREA_M


def _build_synthetic_scene(n_veh: int, n_frames: int, seed: int):
    """
    Simulate n_frames of n_veh constant-velocity vehicles.
    Returns frames_dict, timestamps, ego_traj.
    """
    rng = np.random.RandomState(seed)
    vehicles = _make_vehicles(n_veh, rng)

    # Deep-copy initial state so step does not corrupt across seeds
    import copy
    vehicles = copy.deepcopy(vehicles)

    frames_dict = {}
    timestamps  = {}
    ego_traj    = {}

    for fid in range(n_frames):
        t = fid * DT
        timestamps[fid] = t
        # Snapshot vehicle positions (copy pos array so step does not alias)
        frames_dict[fid] = [
            {
                "track_id": v["track_id"],
                "pos":  v["pos"].copy(),
                "vel":  v["vel"].copy(),
                "yaw":  v["yaw"],
                "L":    v["L"],
                "W":    v["W"],
                "speed": v["speed"],
            }
            for v in vehicles
        ]
        ego_traj[fid] = generate_ego_circular(
            t, radius=15.0, speed=5.398298934, center=EGO_CTR)
        _step(vehicles, DT)

    return frames_dict, timestamps, ego_traj


# ---------------------------------------------------------------------------
# Per-(N, seed) runner
# ---------------------------------------------------------------------------

WARMUP_FRAMES = 50   # frames discarded from timing (eligibility warm-up)


def _run_one(n_veh, n_frames, seed, fov_deg, delta_deg, r_max, cell_size):
    frames_dict, timestamps, ego_traj = _build_synthetic_scene(
        n_veh, n_frames, seed)
    frame_list = list(range(n_frames))

    fov   = math.radians(fov_deg)
    delta = math.radians(delta_deg)
    K     = int(math.floor(fov / delta + 0.5)) + 1

    jit_full_ms    = []
    jit_scan_ms    = []
    jit_cand       = []
    ng_full_ms     = []
    ng_scan_ms     = []
    ng_cand        = []
    ng_muted       = []
    g_full_ms      = []
    g_scan_ms      = []
    g_cand         = []
    g_muted        = []
    g_pre_filter   = []   # fraction of processed vehicles kept after grid filter

    # ---- JIT vanilla ----
    for fid in frame_list:
        fv    = frames_dict[fid]
        ego_s = ego_traj[fid]
        t0 = time.perf_counter()
        segs, tids = build_segments_from_frame(fv)
        _, _, _, tc, scan_t = scan_jit_vanilla(
            ego_s, segs, fov_deg=fov_deg, delta_deg=delta_deg, r_max=r_max,
            seg_track_ids=tids)
        t_full = time.perf_counter() - t0
        if fid >= WARMUP_FRAMES:
            jit_full_ms.append(t_full * 1000.0)
            jit_scan_ms.append(float(scan_t) * 1000.0)
            jit_cand.append(int(tc))

    # ---- SPART no-grid ----
    spart_ng = SPART(
        {"fov_deg": fov_deg, "delta_deg": delta_deg,
         "r_max_m": r_max, "enable_grid": False,
         "parallel_kernel": False},
        track_memory=False)

    for fid in frame_list:
        fv    = frames_dict[fid]
        ego_s = ego_traj[fid]
        tcurr = timestamps[fid]
        t0    = time.perf_counter()
        _, _, _, met = spart_ng.run_frame(fv, ego_s, tcurr)
        t_full = time.perf_counter() - t0
        if fid >= WARMUP_FRAMES:
            ng_full_ms.append(t_full * 1000.0)
            ng_scan_ms.append(float(met["time_scan_sec"]) * 1000.0)
            ng_cand.append(int(met["total_candidates"]))
            ng_muted.append(float(met["muted_ratio"]))

    # ---- SPART with grid ----
    spart_g = SPART(
        {"fov_deg": fov_deg, "delta_deg": delta_deg,
         "r_max_m": r_max, "enable_grid": True,
         "cell_size": cell_size, "parallel_kernel": False},
        track_memory=False)
    spart_g.init_grid(frames_dict)

    # Monkey-patch run_frame to also record grid pre-filter ratio
    original_run_frame = spart_g.run_frame

    pre_filter_ratios = []

    def _patched_run_frame(fv, ego_s, tcurr):
        from spart.core import _grid_candidates
        import math as _math
        angles, ranges, hit_tracks, metrics = original_run_frame(fv, ego_s, tcurr)
        # Approximate grid pre-filter: n_processed vs n_selected (not directly exposed)
        # Use n_processed and n_segments as proxy: segments/4 = selected vehicles
        n_proc = metrics["n_processed"]
        n_sel  = metrics["n_segments"] // 4 if metrics["n_segments"] > 0 else 0
        ratio  = (n_proc - n_sel) / max(1, n_proc)  # fraction filtered out by grid
        pre_filter_ratios.append(ratio)
        return angles, ranges, hit_tracks, metrics

    spart_g.run_frame = _patched_run_frame

    for fid in frame_list:
        fv    = frames_dict[fid]
        ego_s = ego_traj[fid]
        tcurr = timestamps[fid]
        t0    = time.perf_counter()
        _, _, _, met = spart_g.run_frame(fv, ego_s, tcurr)
        t_full = time.perf_counter() - t0
        if fid >= WARMUP_FRAMES:
            g_full_ms.append(t_full * 1000.0)
            g_scan_ms.append(float(met["time_scan_sec"]) * 1000.0)
            g_cand.append(int(met["total_candidates"]))
            g_muted.append(float(met["muted_ratio"]))
            if len(pre_filter_ratios) >= fid - WARMUP_FRAMES + 1:
                g_pre_filter.append(pre_filter_ratios[-1])

    def _hz(ms_list):
        m = float(np.mean(ms_list)) if ms_list else 1e9
        return 1000.0 / max(m, 1e-6)

    def _sei(ms_list, cand_list):
        m_cand = float(np.mean(cand_list)) if cand_list else 1
        return float(K * n_veh * 4) / max(1.0, m_cand)

    return {
        "n_veh": n_veh, "seed": seed,
        "jit": {
            "full_hz": _hz(jit_full_ms),
            "scan_hz": _hz(jit_scan_ms),
            "mean_cand": float(np.mean(jit_cand)) if jit_cand else 0,
        },
        "nogrid": {
            "full_hz": _hz(ng_full_ms),
            "scan_hz": _hz(ng_scan_ms),
            "mean_cand": float(np.mean(ng_cand)) if ng_cand else 0,
            "sei": _sei(ng_scan_ms, ng_cand),
            "muting_pct": 100.0 * float(np.mean(ng_muted)) if ng_muted else 0,
        },
        "grid": {
            "full_hz": _hz(g_full_ms),
            "scan_hz": _hz(g_scan_ms),
            "mean_cand": float(np.mean(g_cand)) if g_cand else 0,
            "sei": _sei(g_scan_ms, g_cand),
            "muting_pct": 100.0 * float(np.mean(g_muted)) if g_muted else 0,
            "grid_filter_pct": 100.0 * float(np.mean(g_pre_filter)) if g_pre_filter else 0,
        },
    }


# ---------------------------------------------------------------------------
# Aggregate over seeds
# ---------------------------------------------------------------------------

def _agg(rows, method_key, metric_key):
    vals = [r[method_key][metric_key] for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def aggregate_by_density(all_rows, densities):
    summary = {}
    for N in densities:
        rows = [r for r in all_rows if r["n_veh"] == N]
        entry = {"n_veh": N, "n_seeds": len(rows)}
        for method in ("jit", "nogrid", "grid"):
            keys = list(rows[0][method].keys())
            entry[method] = {}
            for k in keys:
                mean, std = _agg(rows, method, k)
                entry[method][k] = mean
                entry[method][k + "_std"] = std
        summary[N] = entry
    return summary


# ---------------------------------------------------------------------------
# Find crossover N*
# ---------------------------------------------------------------------------

def find_crossover(summary, densities):
    """
    Crossover N* = smallest N where grid full_hz > nogrid full_hz.
    Returns None if grid never beats nogrid in the tested range.
    """
    crossover = None
    for N in densities:
        s = summary[N]
        if s["grid"]["full_hz"] > s["nogrid"]["full_hz"]:
            crossover = N
            break
    return crossover


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_full_hz(summary, densities, out_dir):
    if not _HAS_MPL:
        return
    Ns   = densities
    jit  = [summary[N]["jit"]["full_hz"]    for N in Ns]
    ng   = [summary[N]["nogrid"]["full_hz"]  for N in Ns]
    g    = [summary[N]["grid"]["full_hz"]    for N in Ns]
    ng_s = [summary[N]["nogrid"]["full_hz_std"] for N in Ns]
    g_s  = [summary[N]["grid"]["full_hz_std"]   for N in Ns]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(Ns, jit, "o--", color=_C_JIT,    lw=1.5, ms=6, label="JIT vanilla")
    ax.errorbar(Ns, ng, yerr=ng_s, fmt="s-", color=_C_NOGRID,
                lw=1.5, ms=6, capsize=4, label="SPART no-grid")
    ax.errorbar(Ns, g,  yerr=g_s,  fmt="^-", color=_C_GRID,
                lw=1.5, ms=6, capsize=4, label="SPART + grid")

    # Mark crossover
    for i in range(len(Ns) - 1):
        if g[i + 1] > ng[i + 1] and g[i] <= ng[i]:
            ax.axvline(Ns[i + 1], color="gray", lw=1, ls=":", alpha=0.7)
            ax.text(Ns[i + 1], max(jit) * 0.85,
                    f"N*={Ns[i+1]}", ha="center", fontsize=9, color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("N vehicles in scene")
    ax.set_ylabel("Full-pipeline throughput (Hz)")
    ax.set_title("Grid Crossover: Full-Pipeline Throughput vs Scene Density\n"
                 "Error bars = std over 3 seeds")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, lw=0.4)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(N) for N in Ns])
    plt.tight_layout()
    out = os.path.join(out_dir, "fig1_full_hz_vs_N.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig_scan_hz(summary, densities, out_dir):
    if not _HAS_MPL:
        return
    Ns  = densities
    ng  = [summary[N]["nogrid"]["scan_hz"] for N in Ns]
    g   = [summary[N]["grid"]["scan_hz"]   for N in Ns]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(Ns, ng, "s-", color=_C_NOGRID, lw=1.5, ms=6, label="SPART no-grid")
    ax.plot(Ns, g,  "^-", color=_C_GRID,   lw=1.5, ms=6, label="SPART + grid")
    ax.set_xscale("log")
    ax.set_xlabel("N vehicles in scene")
    ax.set_ylabel("Scan-kernel throughput (Hz)")
    ax.set_title("Grid Crossover: Scan-Kernel Throughput vs N")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, lw=0.4)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(N) for N in Ns])
    plt.tight_layout()
    out = os.path.join(out_dir, "fig2_scan_hz_vs_N.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig_candidates(summary, densities, out_dir):
    if not _HAS_MPL:
        return
    Ns  = densities
    jit = [summary[N]["jit"]["mean_cand"]    for N in Ns]
    ng  = [summary[N]["nogrid"]["mean_cand"] for N in Ns]
    g   = [summary[N]["grid"]["mean_cand"]   for N in Ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(Ns, jit, "o--", color=_C_JIT,    lw=1.5, ms=6, label="JIT vanilla")
    ax1.plot(Ns, ng,  "s-",  color=_C_NOGRID, lw=1.5, ms=6, label="SPART no-grid")
    ax1.plot(Ns, g,   "^-",  color=_C_GRID,   lw=1.5, ms=6, label="SPART + grid")
    ax1.set_xscale("log")
    ax1.set_xlabel("N vehicles")
    ax1.set_ylabel("Mean candidates / frame")
    ax1.set_title("Intersection tests per frame vs N")
    ax1.legend(fontsize=8)
    ax1.yaxis.grid(True, alpha=0.3, lw=0.4)
    ax1.set_xticks(Ns); ax1.set_xticklabels([str(N) for N in Ns])

    sei_ng = [summary[N]["nogrid"]["sei"] for N in Ns]
    sei_g  = [summary[N]["grid"]["sei"]   for N in Ns]
    ax2.plot(Ns, sei_ng, "s-", color=_C_NOGRID, lw=1.5, ms=6, label="SPART no-grid")
    ax2.plot(Ns, sei_g,  "^-", color=_C_GRID,   lw=1.5, ms=6, label="SPART + grid")
    ax2.set_xscale("log")
    ax2.set_xlabel("N vehicles")
    ax2.set_ylabel("SEI = K × M_all / candidates")
    ax2.set_title("Scan Efficiency Index vs N")
    ax2.legend(fontsize=8)
    ax2.yaxis.grid(True, alpha=0.3, lw=0.4)
    ax2.set_xticks(Ns); ax2.set_xticklabels([str(N) for N in Ns])
    plt.tight_layout()
    out = os.path.join(out_dir, "fig3_candidates_vs_N.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def fig_muting(summary, densities, out_dir):
    if not _HAS_MPL:
        return
    Ns  = densities
    mut_ng = [summary[N]["nogrid"]["muting_pct"]       for N in Ns]
    mut_g  = [summary[N]["grid"]["muting_pct"]          for N in Ns]
    filt_g = [summary[N]["grid"]["grid_filter_pct"]     for N in Ns]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(Ns, mut_ng, "s-",  color=_C_NOGRID, lw=1.5, ms=6,
            label="Muting % (no-grid)")
    ax.plot(Ns, mut_g,  "^-",  color=_C_GRID,   lw=1.5, ms=6,
            label="Muting % (grid)")
    ax.plot(Ns, filt_g, "^--", color=_C_GRID,   lw=1.5, ms=6, alpha=0.5,
            label="Grid pre-filter % (of processed)")
    ax.set_xscale("log")
    ax.set_xlabel("N vehicles")
    ax.set_ylabel("Fraction (%)")
    ax.set_title("Muting rate and grid pre-filter rate vs N")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, lw=0.4)
    ax.set_xticks(Ns); ax.set_xticklabels([str(N) for N in Ns])
    plt.tight_layout()
    out = os.path.join(out_dir, "fig4_muting_vs_N.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(summary, densities, crossover):
    print("\n" + "=" * 96)
    print("EXP 4  GRID CROSSOVER SUMMARY  (synthetic scenes, 3 seeds × 500 frames)")
    print("=" * 96)
    hdr = (f"  {'N':>5}  {'Method':<16}"
           f"{'Full Hz':>10} {'Scan Hz':>10} {'Candidates':>12}"
           f"{'SEI':>8} {'Muted%':>8} {'GridFilt%':>10}")
    print(hdr)
    print("  " + "-" * 86)
    for N in densities:
        s = summary[N]
        for method, label in [("jit","JIT vanilla"),("nogrid","SPART no-grid"),("grid","SPART+grid")]:
            m = s[method]
            sei  = f"{m['sei']:.1f}x"   if "sei"             in m else "--"
            mut  = f"{m['muting_pct']:.1f}%" if "muting_pct" in m else "--"
            gf   = (f"{m['grid_filter_pct']:.1f}%"
                    if "grid_filter_pct" in m else "--")
            cand = f"{m['mean_cand']:,.0f}"
            print(f"  {N:>5}  {label:<16}"
                  f"{m['full_hz']:>10.0f} {m['scan_hz']:>10.0f} {cand:>12}"
                  f"{sei:>8} {mut:>8} {gf:>10}")
        print()

    if crossover is not None:
        print(f"  Crossover N* = {crossover}  "
              f"(grid faster than no-grid from N={crossover} onward)")
    else:
        print("  No crossover found in tested range — grid not yet faster than no-grid at N=200")

    gate_ok = summary[densities[-1]]["grid"]["full_hz"] > \
              summary[densities[-1]]["nogrid"]["full_hz"]
    gate_s  = "PASS" if gate_ok else "FAIL"
    g_hz  = summary[densities[-1]]["grid"]["full_hz"]
    ng_hz = summary[densities[-1]]["nogrid"]["full_hz"]
    print(f"\n  Gate: SPART_grid full_hz > SPART_nogrid at N={densities[-1]} "
          f"({g_hz:.0f} > {ng_hz:.0f})  --> {gate_s}")
    print("=" * 96)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(_ROOT, "results", "grid_crossover")
    os.makedirs(out_dir, exist_ok=True)

    cfg        = _load_config()
    densities  = [int(x) for x in cfg["grid_crossover_densities"]]
    seeds      = [int(x) for x in cfg["grid_crossover_seeds"]]
    n_frames   = int(cfg["grid_crossover_frames_per_seed"])
    fov_deg    = float(cfg["fov_deg"])
    delta_deg  = float(cfg["delta_deg"])
    r_max      = float(cfg["r_max_m"])
    cell_size  = float(cfg.get("cell_size", 10.0))

    print(f"[INFO] Grid crossover study")
    print(f"  Densities: {densities}")
    print(f"  Seeds: {seeds}  |  Frames/seed: {n_frames}  |  Warmup: {WARMUP_FRAMES}")
    print(f"  Scene: {AREA_M}×{AREA_M} m, ego circle r=15m, dt={DT}s")
    print(f"  Cell size: {cell_size} m")

    print("\n[INFO] Warming up JIT ...")
    t0 = time.perf_counter()
    warmup_numba_kernel(fov_deg, delta_deg, r_max)
    warmup_jit_vanilla(fov_deg, delta_deg, r_max)
    print(f"  Compiled in {time.perf_counter()-t0:.2f}s")

    all_rows = []
    t_total  = time.perf_counter()

    for N in densities:
        print(f"\n[N={N:>3}] ", end="", flush=True)
        for seed in seeds:
            print(f"seed={seed} ", end="", flush=True)
            row = _run_one(N, n_frames, seed, fov_deg, delta_deg, r_max, cell_size)
            all_rows.append(row)
            print(f"({row['nogrid']['full_hz']:.0f}/{row['grid']['full_hz']:.0f} Hz) ",
                  end="", flush=True)
        print()

    print(f"\n[INFO] All done in {time.perf_counter()-t_total:.1f}s")

    summary   = aggregate_by_density(all_rows, densities)
    crossover = find_crossover(summary, densities)

    # Save JSON
    out_j = os.path.join(out_dir, "grid_crossover_summary.json")
    with open(out_j, "w") as f:
        json.dump({"densities": densities, "seeds": seeds,
                   "n_frames": n_frames, "warmup": WARMUP_FRAMES,
                   "crossover_N": crossover,
                   "per_density": {str(k): v for k, v in summary.items()},
                   "all_rows": all_rows}, f, indent=2, default=float)
    print(f"  Saved {out_j}")

    # Save CSV
    keys = (["n_veh", "seed"] +
            [f"jit_{k}" for k in all_rows[0]["jit"]] +
            [f"nogrid_{k}" for k in all_rows[0]["nogrid"]] +
            [f"grid_{k}"   for k in all_rows[0]["grid"]])
    out_c = os.path.join(out_dir, "per_condition_means.csv")
    with open(out_c, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            flat = {"n_veh": r["n_veh"], "seed": r["seed"]}
            for m in ("jit", "nogrid", "grid"):
                for k, v in r[m].items():
                    flat[f"{m}_{k}"] = v
            w.writerow(flat)
    print(f"  Saved {out_c}")

    print("\n[INFO] Generating figures ...")
    fig_full_hz(summary, densities, out_dir)
    fig_scan_hz(summary, densities, out_dir)
    fig_candidates(summary, densities, out_dir)
    fig_muting(summary, densities, out_dir)

    print_summary(summary, densities, crossover)
    gate_ok = (summary[densities[-1]]["grid"]["full_hz"] >
               summary[densities[-1]]["nogrid"]["full_hz"])
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()

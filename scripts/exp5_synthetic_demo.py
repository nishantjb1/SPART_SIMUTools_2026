"""
scripts/exp5_synthetic_demo.py
Experiment 5: Synthetic Demo (no INTERACTION dataset required).

Runs SPART on a fully synthetic scene (15 vehicles, 100 frames) and produces:
  1. A GIF animation: top-down view of vehicles, ego, and SPART scan
  2. Point cloud CSV: (frame_id, timestamp_s, beam_idx, angle_rad, range_m, hit_track_id)
  3. Lightweight fidelity check: FP=0, AP=0, range_error=0.000m (on processed vehicles)

Gate: total wall-clock time < config[synthetic_demo_max_seconds] (default 60 s).
      GIF and fidelity outputs are produced regardless; gate only reports timing.

Outputs (results/synthetic_demo/):
  demo_animation.gif     (or PNG frames if Pillow unavailable)
  demo_pointcloud.csv
  demo_fidelity.json
  demo_summary.json
"""

import copy
import csv as _csv
import io
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
    from matplotlib.patches import Wedge
    from matplotlib.collections import LineCollection
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("[WARN] matplotlib not found -- GIF skipped")

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    print("[WARN] Pillow not found -- GIF will be saved as PNG frames")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spart.core import SPART
from spart.utils import build_segments_from_frame, generate_ego_circular
from spart.scan_kernel import warmup_numba_kernel
from spart.vanilla_kernel import scan_jit_vanilla, warmup_jit_vanilla

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Scene geometry
# ---------------------------------------------------------------------------

AREA_M  = 60.0
EGO_CTR = (30.0, 30.0)
DT      = 0.1       # 10 Hz
VEH_L   = 4.5
VEH_W   = 2.0

# GIF rendering settings -- low resolution for speed
_GIF_DPI      = 70
_GIF_FIGSIZE  = (6, 6)
_GIF_STEP     = 2       # render every Nth frame (50 GIF frames from 100 scene frames)
_GIF_FPS      = 10      # playback speed


def _load_config():
    p = os.path.join(_ROOT, "configs", "benchmark_config.yaml")
    with open(p) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Synthetic scene generator (mirrors exp4)
# ---------------------------------------------------------------------------

def _make_vehicles(n_veh: int, rng: np.random.RandomState):
    cx, cy = EGO_CTR
    half = AREA_M / 2.0
    vehicles = []
    for i in range(n_veh):
        x     = cx + rng.uniform(-half, half)
        y     = cy + rng.uniform(-half, half)
        speed = rng.uniform(2.0, 8.0)
        yaw   = rng.uniform(-math.pi, math.pi)
        vehicles.append({
            "track_id": i + 1,
            "pos":  np.array([x, y],                         dtype=np.float64),
            "vel":  np.array([speed * math.cos(yaw),
                              speed * math.sin(yaw)],         dtype=np.float64),
            "yaw":  yaw,
            "L":    VEH_L,
            "W":    VEH_W,
            "speed": speed,
        })
    return vehicles


def _step(vehicles, dt):
    cx, cy = EGO_CTR
    half = AREA_M / 2.0
    for v in vehicles:
        v["pos"] += v["vel"] * dt
        for dim, c in enumerate([cx, cy]):
            if v["pos"][dim] < c - half:
                v["pos"][dim] += AREA_M
            elif v["pos"][dim] > c + half:
                v["pos"][dim] -= AREA_M


def _build_scene(n_veh, n_frames, seed):
    rng      = np.random.RandomState(seed)
    vehicles = copy.deepcopy(_make_vehicles(n_veh, rng))
    frames_dict, timestamps, ego_traj = {}, {}, {}
    for fid in range(n_frames):
        t = fid * DT
        timestamps[fid]  = t
        frames_dict[fid] = [
            {"track_id": v["track_id"],
             "pos":  v["pos"].copy(),
             "vel":  v["vel"].copy(),
             "yaw":  v["yaw"],
             "L":    v["L"],
             "W":    v["W"],
             "speed": v["speed"]}
            for v in vehicles
        ]
        ego_traj[fid] = generate_ego_circular(
            t, radius=15.0, speed=5.398298934, center=EGO_CTR)
        _step(vehicles, DT)
    return frames_dict, timestamps, ego_traj


# ---------------------------------------------------------------------------
# GIF frame renderer
# ---------------------------------------------------------------------------

def _box_corners(cx, cy, yaw, L, W):
    """Return (4, 2) corners of an oriented bounding box."""
    hl, hw = L / 2, W / 2
    local  = np.array([[ hl,  hw], [-hl,  hw], [-hl, -hw], [ hl, -hw]])
    c, s   = math.cos(yaw), math.sin(yaw)
    R      = np.array([[c, -s], [s, c]])
    return (R @ local.T).T + np.array([cx, cy])


def _render_frame(fid, frame_veh, ego_s, angles, ranges, processed_ids,
                  r_max, fov_deg, n_veh, metrics):
    """
    Render one animation frame.  Returns a PIL Image (RGB).
    """
    fig, ax = plt.subplots(figsize=_GIF_FIGSIZE)
    ax.set_aspect("equal")
    cx, cy = EGO_CTR
    half   = AREA_M / 2.0
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_facecolor("#f8f8f8")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ego_x   = float(ego_s["x"])
    ego_y   = float(ego_s["y"])
    ego_psi = float(ego_s["psi_rad"])

    # --- FOV sector ---
    psi_deg = math.degrees(ego_psi)
    half_fov = fov_deg / 2.0
    wedge = Wedge((ego_x, ego_y), r_max,
                  psi_deg - half_fov, psi_deg + half_fov,
                  color="#aec6cf", alpha=0.18, zorder=1)
    ax.add_patch(wedge)

    # r_max circle (dashed)
    circ = plt.Circle((ego_x, ego_y), r_max, fill=False,
                       linestyle="--", linewidth=0.7, color="#999", zorder=1)
    ax.add_patch(circ)

    # --- Scan beams (LineCollection for speed) ---
    hit_lines, miss_lines = [], []
    for k, (ang, rng) in enumerate(zip(angles, ranges)):
        ex, ey = ego_x, ego_y
        if not np.isnan(rng):
            hx = ego_x + rng * math.cos(ang)
            hy = ego_y + rng * math.sin(ang)
            hit_lines.append([(ex, ey), (hx, hy)])
        else:
            mx = ego_x + r_max * math.cos(ang)
            my = ego_y + r_max * math.sin(ang)
            miss_lines.append([(ex, ey), (mx, my)])

    if miss_lines:
        ax.add_collection(LineCollection(miss_lines, colors="#cccccc",
                                         linewidths=0.4, zorder=2))
    if hit_lines:
        ax.add_collection(LineCollection(hit_lines, colors="#e05c3a",
                                         linewidths=0.9, zorder=3))
        for seg in hit_lines:
            ax.plot(*seg[1], "o", color="#c0392b", ms=3, zorder=4)

    # --- Vehicles ---
    for v in frame_veh:
        vx = float(v["pos"][0])
        vy = float(v["pos"][1])
        yaw = float(v["yaw"])
        tid = int(v["track_id"])
        corners = _box_corners(vx, vy, yaw, VEH_L, VEH_W)

        if tid not in processed_ids:
            # muted
            poly = plt.Polygon(corners, closed=True,
                               facecolor="#bbbbbb", edgecolor="#888",
                               alpha=0.35, linewidth=0.8, zorder=5)
        else:
            # processed -- colour by hit status
            poly = plt.Polygon(corners, closed=True,
                               facecolor="#2ecc71", edgecolor="#27ae60",
                               alpha=0.80, linewidth=1.0, zorder=5)

        ax.add_patch(poly)
        ax.text(vx, vy, str(tid), ha="center", va="center",
                fontsize=5, color="black", zorder=6)

    # --- Ego (triangle arrow) ---
    tri_len = 2.5
    dx = tri_len * math.cos(ego_psi)
    dy = tri_len * math.sin(ego_psi)
    ax.annotate("", xy=(ego_x + dx, ego_y + dy), xytext=(ego_x, ego_y),
                arrowprops=dict(arrowstyle="-|>", color="#1a6ba0",
                                lw=1.8, mutation_scale=14))
    ax.plot(ego_x, ego_y, "o", color="#1a6ba0", ms=5, zorder=7)

    n_proc  = metrics["n_processed"]
    n_skip  = metrics["n_skipped"]
    muted_p = int(100 * metrics["muted_ratio"] + 0.5)
    ax.set_title(
        f"Frame {fid+1:3d}/100   t={fid*DT:.1f}s   "
        f"processed={n_proc}/{n_veh}  muted={n_skip} ({muted_p}%)\n"
        f"candidates={metrics['total_candidates']}",
        fontsize=8, pad=4)
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("y (m)", fontsize=7)
    ax.tick_params(labelsize=6)

    # capture to PIL
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_GIF_DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    return img


# ---------------------------------------------------------------------------
# GIF assembly
# ---------------------------------------------------------------------------

def _save_gif(pil_frames, out_path, fps=_GIF_FPS):
    if not pil_frames:
        return
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    size_kb = os.path.getsize(out_path) // 1024
    print(f"  Saved {out_path}  ({len(pil_frames)} frames, {size_kb} KB)")


def _save_png_frames(pil_frames, out_dir):
    """Fallback when Pillow GIF support is unavailable."""
    for i, img in enumerate(pil_frames):
        p = os.path.join(out_dir, f"frame_{i:03d}.png")
        img.save(p)
    print(f"  Saved {len(pil_frames)} PNG frames to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_wall_start = time.perf_counter()

    cfg         = _load_config()
    n_veh       = int(cfg.get("synthetic_demo_n_vehicles", 15))
    n_frames    = int(cfg.get("synthetic_demo_n_frames", 100))
    max_seconds = float(cfg.get("synthetic_demo_max_seconds", 60))
    fov_deg     = float(cfg["fov_deg"])
    delta_deg   = float(cfg["delta_deg"])
    r_max       = float(cfg["r_max_m"])
    seed        = 42

    out_dir = os.path.join(_ROOT, "results", "synthetic_demo")
    os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Synthetic demo")
    print(f"  Vehicles: {n_veh}  |  Frames: {n_frames}  |  Seed: {seed}")
    print(f"  Scene: {AREA_M}×{AREA_M} m, ego r=15m, dt={DT}s")
    print(f"  Timing gate: < {max_seconds:.0f} s")

    # ---- JIT warmup ----
    print("\n[INFO] Warming up JIT ...")
    t0 = time.perf_counter()
    warmup_numba_kernel(fov_deg, delta_deg, r_max)
    warmup_jit_vanilla(fov_deg, delta_deg, r_max)
    t_warmup = time.perf_counter() - t0
    print(f"  Compiled in {t_warmup:.2f} s")

    # ---- Build synthetic scene ----
    print("\n[INFO] Building synthetic scene ...")
    frames_dict, timestamps, ego_traj = _build_scene(n_veh, n_frames, seed)

    # ---- Initialise SPART ----
    spart = SPART(
        {"fov_deg": fov_deg, "delta_deg": delta_deg,
         "r_max_m": r_max, "enable_grid": False,
         "parallel_kernel": False},
        track_memory=False,
    )

    # ---- Run SPART + fidelity check per frame ----
    print("[INFO] Running SPART + fidelity check ...")
    fp_total  = 0
    ap_total  = 0
    max_range_err = 0.0

    frame_data   = []   # for GIF rendering
    pc_rows      = []   # for point cloud CSV

    fov = math.radians(fov_deg)
    delta_r = math.radians(delta_deg)
    K = int(math.floor(fov / delta_r + 0.5)) + 1

    for fid in range(n_frames):
        fv    = frames_dict[fid]
        tcurr = timestamps[fid]
        ego_s = ego_traj[fid]

        # Determine processed vehicle IDs BEFORE run_frame updates eligibility
        processed_ids = {
            int(v["track_id"])
            for v in fv
            if tcurr >= spart._eligibility.get(int(v["track_id"]), 0.0)
        }
        processed_veh = [v for v in fv if int(v["track_id"]) in processed_ids]

        # SPART scan
        angles, ranges, hit_tracks, metrics = spart.run_frame(fv, ego_s, tcurr)

        # Fidelity: vanilla-on-processed (same segment set as SPART)
        if processed_veh:
            van_segs, van_tids = build_segments_from_frame(processed_veh)
            van_angles, van_ranges, _, _, _ = scan_jit_vanilla(
                ego_s, van_segs,
                fov_deg=fov_deg, delta_deg=delta_deg, r_max=r_max,
                seg_track_ids=van_tids,
            )
        else:
            half = fov / 2.0
            van_angles = np.array([ego_s["psi_rad"] - half + k * delta_r
                                   for k in range(K)], dtype=np.float64)
            van_ranges = np.full(K, np.nan, dtype=np.float64)

        # Per-beam comparison
        for k in range(K):
            s_hit = not np.isnan(ranges[k])
            v_hit = not np.isnan(van_ranges[k])
            if s_hit and not v_hit:
                fp_total += 1
            if v_hit and not s_hit:
                ap_total += 1
            if s_hit and v_hit:
                err = abs(float(ranges[k]) - float(van_ranges[k]))
                if err > max_range_err:
                    max_range_err = err

        # Point cloud rows
        for k in range(K):
            pc_rows.append({
                "frame_id":    fid,
                "timestamp_s": round(tcurr, 3),
                "beam_idx":    k,
                "angle_rad":   round(float(angles[k]), 6),
                "range_m":     "" if np.isnan(ranges[k]) else round(float(ranges[k]), 4),
                "hit_track_id": int(hit_tracks[k]),
            })

        # Store for GIF
        if _HAS_MPL and (fid % _GIF_STEP == 0):
            frame_data.append({
                "fid":          fid,
                "frame_veh":    fv,
                "ego_s":        ego_s,
                "angles":       angles,
                "ranges":       ranges,
                "processed_ids": processed_ids,
                "metrics":      metrics,
            })

    print(f"  Fidelity: FP={fp_total}  AP={ap_total}  max_range_err={max_range_err:.6f} m")

    # ---- Save point cloud CSV ----
    pc_path = os.path.join(out_dir, "demo_pointcloud.csv")
    with open(pc_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(pc_rows[0].keys()))
        w.writeheader()
        w.writerows(pc_rows)
    size_kb = os.path.getsize(pc_path) // 1024
    print(f"  Saved {pc_path}  ({len(pc_rows):,} rows, {size_kb} KB)")

    # ---- Render GIF ----
    pil_frames = []
    if _HAS_MPL:
        print(f"\n[INFO] Rendering {len(frame_data)} GIF frames "
              f"(every {_GIF_STEP}nd scene frame) ...")
        t_gif = time.perf_counter()
        for i, fd in enumerate(frame_data):
            img = _render_frame(
                fid=fd["fid"],
                frame_veh=fd["frame_veh"],
                ego_s=fd["ego_s"],
                angles=fd["angles"],
                ranges=fd["ranges"],
                processed_ids=fd["processed_ids"],
                r_max=r_max,
                fov_deg=fov_deg,
                n_veh=n_veh,
                metrics=fd["metrics"],
            )
            pil_frames.append(img)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(frame_data)} frames rendered ...", flush=True)
        print(f"  Render complete in {time.perf_counter()-t_gif:.1f} s")

        gif_path = os.path.join(out_dir, "demo_animation.gif")
        if _HAS_PIL and pil_frames:
            _save_gif(pil_frames, gif_path)
        elif _HAS_MPL and pil_frames:
            _save_png_frames(pil_frames, out_dir)
    else:
        print("[WARN] Skipping GIF -- matplotlib not available")

    # ---- Save fidelity JSON ----
    fidelity_ok = (fp_total == 0 and ap_total == 0 and max_range_err < 1e-6)
    fidelity = {
        "fp_total": fp_total,
        "ap_total": ap_total,
        "max_range_err_m": round(max_range_err, 8),
        "gate": "PASS" if fidelity_ok else "FAIL",
        "n_frames_checked": n_frames,
    }
    fid_path = os.path.join(out_dir, "demo_fidelity.json")
    with open(fid_path, "w") as f:
        json.dump(fidelity, f, indent=2)
    print(f"  Saved {fid_path}")

    # ---- Timing gate ----
    t_total = time.perf_counter() - t_wall_start
    timing_ok = t_total < max_seconds

    summary = {
        "n_vehicles":           n_veh,
        "n_frames":             n_frames,
        "seed":                 seed,
        "fov_deg":              fov_deg,
        "delta_deg":            delta_deg,
        "r_max_m":              r_max,
        "gif_frames_rendered":  len(pil_frames),
        "gif_step":             _GIF_STEP,
        "pointcloud_rows":      len(pc_rows),
        "fidelity":             fidelity,
        "warmup_sec":           round(t_warmup, 2),
        "total_sec":            round(t_total, 2),
        "max_seconds":          max_seconds,
        "timing_gate":          "PASS" if timing_ok else "FAIL",
    }
    sum_path = os.path.join(out_dir, "demo_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {sum_path}")

    # ---- Print final summary ----
    print("\n" + "=" * 72)
    print("EXP 5  SYNTHETIC DEMO SUMMARY")
    print("=" * 72)
    print(f"  Scene:       {n_veh} vehicles, {n_frames} frames, seed={seed}")
    print(f"  Fidelity:    FP={fp_total}  AP={ap_total}  "
          f"max_range_err={max_range_err:.6f} m  "
          f"--> {'PASS' if fidelity_ok else 'FAIL'}")
    print(f"  GIF:         {len(pil_frames)} frames rendered  "
          f"({'demo_animation.gif' if _HAS_PIL else 'PNG frames'})")
    print(f"  Point cloud: {len(pc_rows):,} rows  ({n_frames} frames × {K} beams)")
    print(f"  Warmup:      {t_warmup:.2f} s")
    print(f"  Total time:  {t_total:.2f} s / {max_seconds:.0f} s  "
          f"--> {'PASS' if timing_ok else 'FAIL'}")
    print("=" * 72)

    if not fidelity_ok:
        print("[ERROR] Fidelity gate FAILED -- check kernel correctness.")
    if not timing_ok:
        print(f"[WARN] Timing gate FAILED ({t_total:.1f} s > {max_seconds:.0f} s). "
              "Close other apps and re-run for a clean timing measurement.")

    gate_ok = fidelity_ok  # primary gate is fidelity; timing is advisory on loaded machines
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()

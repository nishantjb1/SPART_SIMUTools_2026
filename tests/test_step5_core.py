"""
tests/test_step5_core.py
Validates spart/core.py (SPART class).
Gate:
  - run_frame() produces non-empty ranges and correct metric keys
  - time_fullpipeline_sec >= time_scan_sec (always)
  - run_dataset() processes correct number of frames
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spart.core import SPART
from spart.utils import (
    generate_ego_circular,
    make_ego_trajectory,
    build_segments_from_frame,
)

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0
_ERRORS = []


def check(name, condition, detail=""):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        msg = f"  [FAIL] {name}" + (f" -- {detail}" if detail else "")
        print(msg)
        _ERRORS.append(msg)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def make_vehicles(n=5, ego_x=1011.0, ego_y=999.0):
    rng = np.random.RandomState(7)
    vehicles = []
    for i in range(n):
        ang  = rng.uniform(-math.pi, math.pi)
        dist = rng.uniform(3.0, 12.0)
        cx   = ego_x + dist * math.cos(ang)
        cy   = ego_y + dist * math.sin(ang)
        vehicles.append({
            "track_id": i + 1,
            "pos":   np.array([cx, cy], dtype=np.float64),
            "vel":   np.array([rng.uniform(-2, 2), rng.uniform(-2, 2)], dtype=np.float64),
            "yaw":   rng.uniform(-math.pi, math.pi),
            "L":     4.0, "W": 2.0, "speed": 1.0,
        })
    return vehicles


def make_fake_dataset(n_frames=20):
    """Build a small synthetic frames_dict, timestamps, ego_traj."""
    ego0 = generate_ego_circular(0.0, radius=15.0, speed=5.398298934,
                                 center=(996.0, 999.0))
    vehicles_base = make_vehicles(n=7, ego_x=ego0["x"], ego_y=ego0["y"])

    frames_dict = {}
    timestamps  = {}
    for i in range(n_frames):
        fid = i
        frames_dict[fid] = list(vehicles_base)   # same scene, minimal
        timestamps[fid]  = i * 0.1

    frame_ids = list(frames_dict.keys())
    ego_traj  = make_ego_trajectory(frame_ids, timestamps, mode="circular",
                                    radius=15.0, speed=5.398298934,
                                    center=(996.0, 999.0))
    return frames_dict, timestamps, ego_traj


def main():
    import time
    print("=" * 60)
    print("SPART Step 5 - core.py validation")
    print("=" * 60)

    # Warm up JIT (already compiled from earlier tests, but be safe)
    from spart.scan_kernel import warmup_numba_kernel
    from spart.vanilla_kernel import warmup_jit_vanilla, warmup_jit_angular
    print("\n[INFO] Ensuring JIT kernels are compiled...")
    t0 = time.perf_counter()
    warmup_numba_kernel()
    warmup_jit_vanilla()
    warmup_jit_angular()
    print(f"  JIT ready in {time.perf_counter()-t0:.2f} s")

    cfg = {
        "fov_deg":    120.0,
        "delta_deg":  0.5,
        "r_max_m":    15.0,
        "enable_grid": False,
    }
    spart = SPART(cfg)

    ego = generate_ego_circular(0.0, radius=15.0, speed=5.398298934,
                                center=(996.0, 999.0))
    vehicles = make_vehicles(n=7, ego_x=ego["x"], ego_y=ego["y"])

    print("\n--- run_frame: basic output ---")
    angles, ranges, hit_tracks, metrics = spart.run_frame(vehicles, ego, tcurr=0.0)

    # 1. angles not empty
    check("1. angles non-empty", angles.shape[0] > 0)

    # 2. ranges same length as angles
    check("2. ranges.shape == angles.shape", ranges.shape == angles.shape)

    # 3. hit_tracks same length
    check("3. hit_tracks.shape == angles.shape", hit_tracks.shape == angles.shape)

    # 4. at least one hit
    check("4. at least one beam hit", int(np.sum(~np.isnan(ranges))) > 0,
          "no hits -- vehicles may be out of FOV")

    print("\n--- run_frame: metric keys ---")
    required_keys = [
        "time_scan_sec", "time_fullpipeline_sec",
        "n_vehicles", "n_processed", "n_skipped",
        "n_segments", "K_beams", "total_candidates",
        "candidates_per_beam", "num_hits", "muted_ratio", "memory_rss_mb",
    ]
    for key in required_keys:
        check(f"5.{key} in metrics", key in metrics)

    print("\n--- run_frame: timing constraint ---")
    check("6. fullpipeline_sec >= scan_sec",
          metrics["time_fullpipeline_sec"] >= metrics["time_scan_sec"],
          f"full={metrics['time_fullpipeline_sec']:.6f}, scan={metrics['time_scan_sec']:.6f}")

    print("\n--- run_frame: muting accumulates ---")
    # Muting requires out-of-range vehicles moving away from ego.
    # Place some vehicles OUTSIDE r_max with outward velocity.
    spart.reset()
    ego_far = generate_ego_circular(0.0, radius=15.0, speed=5.398298934,
                                    center=(996.0, 999.0))
    ex2, ey2 = ego_far["x"], ego_far["y"]
    far_vehicles = list(vehicles)  # base vehicles (in range)
    # Add vehicles outside range moving away -- these get deferred eligibility
    for i, ang in enumerate([0.0, math.pi / 2, math.pi]):
        far_vehicles.append({
            "track_id": 100 + i,
            "pos": np.array([ex2 + 20.0 * math.cos(ang),
                             ey2 + 20.0 * math.sin(ang)], dtype=np.float64),
            "vel": np.array([3.0 * math.cos(ang), 3.0 * math.sin(ang)], dtype=np.float64),
            "yaw": ang, "L": 4.0, "W": 2.0, "speed": 3.0,
        })
    # First run: process all (eligibility initialized)
    spart.run_frame(far_vehicles, ego_far, tcurr=0.0)
    # Second run at same timestamp: far vehicles have deferred eligibility
    _, _, _, m2 = spart.run_frame(far_vehicles, ego_far, tcurr=0.0)
    check("7. second run has muted vehicles from far set (muted_ratio > 0)",
          m2["muted_ratio"] > 0.0, f"muted_ratio={m2['muted_ratio']}")

    print("\n--- run_frame: n_vehicles >= n_processed ---")
    check("8. n_vehicles >= n_processed",
          metrics["n_vehicles"] >= metrics["n_processed"])

    print("\n--- run_frame: empty scene ---")
    spart.reset()
    angles_e, ranges_e, ht_e, m_e = spart.run_frame([], ego, tcurr=0.0)
    check("9. empty scene: all ranges NaN", np.all(np.isnan(ranges_e)))
    check("10. empty scene: n_vehicles=0", m_e["n_vehicles"] == 0)

    print("\n--- run_dataset ---")
    frames_dict, timestamps, ego_traj = make_fake_dataset(n_frames=20)
    spart.reset()
    results = spart.run_dataset(frames_dict, timestamps, ego_traj,
                                n_frames=20)
    rows = results.as_records()

    # 11. Correct number of frames processed
    check("11. run_dataset processes 20 frames", len(rows) == 20)

    # 12. All rows have required keys
    row_keys_ok = all(all(k in r for k in ["frame_id", "time_scan_sec",
                                            "time_fullpipeline_sec",
                                            "N_vehicles", "P_processed"])
                      for r in rows)
    check("12. all rows have required keys", row_keys_ok)

    # 13. time_fullpipeline >= time_scan for every frame
    timing_ok = all(r["time_fullpipeline_sec"] >= r["time_scan_sec"] for r in rows)
    check("13. fullpipeline >= scan for all frames", timing_ok)

    # 14. Later frames have some muting (eligibility active)
    later_muted = any(r["muted_ratio"] > 0 for r in rows[5:])
    check("14. muting activates after initial frames", later_muted)

    # 15. Frame IDs are sequential
    fids = [r["frame_id"] for r in rows]
    check("15. frame_ids sorted", fids == sorted(fids))

    # 16. n_frames limit works
    spart.reset()
    results_5 = spart.run_dataset(frames_dict, timestamps, ego_traj, n_frames=5)
    check("16. n_frames=5 returns 5 rows", len(results_5.as_records()) == 5)

    print("\n" + "=" * 60)
    print(f"Results: {_PASS} passed, {_FAIL} failed")
    if _ERRORS:
        print("Failed checks:")
        for e in _ERRORS:
            print(e)
    print("=" * 60)

    if _FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

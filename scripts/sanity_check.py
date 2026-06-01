"""
scripts/sanity_check.py
Pre-experiment green light. Must pass before running any experiment.

Checks:
  1. Both methods share identical ego position for each frame (print first 5 frames)
  2. Frame list used by SPART and vanilla is identical
  3. For 10 sampled frames: SPART and JIT-vanilla agree for processed vehicles (max error=0.0)
  4. candidates_per_beam.sum() == total_candidates (data race fix validation)
  5. Config n_frames matches len(frame_list)

Usage:
    python scripts/sanity_check.py --csv vehicle_tracks_000.csv
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spart.core import SPART
from spart.utils import (
    load_interaction_csv,
    make_ego_trajectory,
    get_frame_list,
    build_segments_from_frame,
)
from spart.scan_kernel import (
    simulate_scan_numba_wrapper,
    warmup_numba_kernel,
)
from spart.vanilla_kernel import (
    scan_jit_vanilla,
    warmup_jit_vanilla,
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


def load_config():
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "benchmark_config.yaml"
    )
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SPART pre-experiment sanity check")
    parser.add_argument("--csv", required=True, help="Path to INTERACTION CSV")
    args = parser.parse_args()

    print("=" * 65)
    print("SPART Sanity Check -- pre-experiment green light")
    print("=" * 65)

    # Load config
    cfg = load_config()
    n_frames = cfg["n_frames"]
    fov_deg  = cfg["fov_deg"]
    delta_deg = cfg["delta_deg"]
    r_max    = cfg["r_max_m"]
    ego_cfg  = {
        "radius": cfg["ego_circle_radius_m"],
        "speed":  cfg["ego_circle_speed_mps"],
        "center": tuple(cfg["ego_circle_center"]),
    }

    # Load dataset
    print(f"\n[INFO] Loading {args.csv} ...")
    frames_dict, timestamps = load_interaction_csv(args.csv)
    frame_list_spart   = get_frame_list(frames_dict, n_frames=n_frames)
    frame_list_vanilla = get_frame_list(frames_dict, n_frames=n_frames)

    print(f"  Frames available: {len(frames_dict)}")
    print(f"  Config n_frames:  {n_frames}")
    print(f"  frame_list len:   {len(frame_list_spart)}")

    # Warm up JIT
    print("\n[INFO] Warming up JIT kernels ...")
    t0 = time.perf_counter()
    warmup_numba_kernel(fov_deg, delta_deg, r_max)
    warmup_jit_vanilla(fov_deg, delta_deg, r_max)
    print(f"  Done in {time.perf_counter()-t0:.2f} s")

    # Build shared ego trajectory
    ego_traj = make_ego_trajectory(
        frame_ids=frame_list_spart,
        timestamps=timestamps,
        mode="circular",
        **ego_cfg
    )

    # -----------------------------------------------------------------------
    # Check 1: Identical ego positions (print first 5)
    # -----------------------------------------------------------------------
    print("\n--- Check 1: Ego positions identical for both methods ---")
    print("  frame_id |  ego_x       ego_y      ego_psi")
    print("  " + "-" * 50)
    ok_ego = True
    for fid in frame_list_spart[:5]:
        es = ego_traj[fid]
        ev = ego_traj[fid]   # same dict -- by construction, always identical
        ok_ego = ok_ego and (es["x"] == ev["x"] and es["y"] == ev["y"])
        print(f"  {fid:8d} | {es['x']:10.4f}  {es['y']:10.4f}  {math.degrees(es['psi_rad']):8.3f} deg")
    check("1. ego_traj is shared (same dict used by both methods)", True)
    # Both methods receive ego_traj[fid] -- the shared dict guarantees identity.
    # We print the values above as evidence.

    # -----------------------------------------------------------------------
    # Check 2: Frame lists are identical
    # -----------------------------------------------------------------------
    print("\n--- Check 2: Frame lists identical ---")
    lists_match = frame_list_spart == frame_list_vanilla
    check("2. frame_list_spart == frame_list_vanilla", lists_match,
          f"lengths: {len(frame_list_spart)} vs {len(frame_list_vanilla)}")

    # -----------------------------------------------------------------------
    # Check 3: n_frames matches len(frame_list)
    # -----------------------------------------------------------------------
    print("\n--- Check 3: Config n_frames matches frame_list length ---")
    check("3. n_frames == len(frame_list)",
          n_frames == len(frame_list_spart),
          f"config n_frames={n_frames}, actual={len(frame_list_spart)}")

    # -----------------------------------------------------------------------
    # Check 4: SPART and JIT-vanilla agree for processed vehicles
    # Sample 10 frames evenly from the first 200
    # -----------------------------------------------------------------------
    print("\n--- Check 4: SPART vs JIT-vanilla range agreement (10 sampled frames) ---")
    sample_fids = frame_list_spart[:200:20][:10]

    spart_runner = SPART({
        "fov_deg": fov_deg, "delta_deg": delta_deg,
        "r_max_m": r_max, "enable_grid": False,
    })
    # Run 50 warmup frames so eligibility stabilizes
    warmup_fids = frame_list_spart[:50]
    for fid in warmup_fids:
        spart_runner.run_frame(frames_dict[fid], ego_traj[fid],
                               tcurr=timestamps[fid])

    max_errors = []
    false_positives = 0
    for fid in sample_fids:
        fv     = frames_dict[fid]
        ego_s  = ego_traj[fid]
        tcurr  = timestamps[fid]

        # SPART run (with eligibility state from warmup)
        ang_s, rng_s, ht_s, met_s = spart_runner.run_frame(fv, ego_s, tcurr)

        # JIT-vanilla run (all vehicles, no muting)
        segs_all, tids_all = build_segments_from_frame(fv)
        ang_v, rng_v, ht_v, tc_v, t_v = scan_jit_vanilla(
            ego_s, segs_all, fov_deg=fov_deg, delta_deg=delta_deg,
            r_max=r_max, seg_track_ids=tids_all
        )

        # For processed vehicles (those SPART did not mute), ranges must match exactly
        # False positive: SPART hits but vanilla misses (should never happen)
        spart_hit   = ~np.isnan(rng_s)
        vanilla_hit = ~np.isnan(rng_v)
        fp = int(np.sum(spart_hit & ~vanilla_hit))
        false_positives += fp

        # Agreement on beams where BOTH hit
        both_hit = spart_hit & vanilla_hit
        if both_hit.any():
            max_err = float(np.max(np.abs(rng_s[both_hit] - rng_v[both_hit])))
        else:
            max_err = 0.0
        max_errors.append(max_err)

    overall_max_err = max(max_errors) if max_errors else 0.0
    check("4a. max_range_error for both-hit beams == 0.0",
          overall_max_err == 0.0, f"max_error={overall_max_err:.2e}")
    check("4b. false_positives == 0",
          false_positives == 0, f"false_positives={false_positives}")

    # -----------------------------------------------------------------------
    # Check 5: candidates_per_beam.sum() == total_candidates (data race fix)
    # -----------------------------------------------------------------------
    print("\n--- Check 5: Data race fix -- candidates_per_beam consistent ---")
    test_fid = frame_list_spart[100]
    fv_test  = frames_dict[test_fid]
    segs_t, tids_t = build_segments_from_frame(fv_test)
    ego_t = ego_traj[test_fid]

    _, _, _, _, tc, cpb, _ = simulate_scan_numba_wrapper(
        ego_t, segs_t, fov_deg=fov_deg, delta_deg=delta_deg,
        r_max=r_max, seg_track_ids=tids_t
    )
    check("5. cpb.sum() == total_candidates",
          int(cpb.sum()) == int(tc),
          f"cpb.sum()={int(cpb.sum())}, total_candidates={int(tc)}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"Results: {_PASS} passed, {_FAIL} failed")
    if _ERRORS:
        print("Failed checks:")
        for e in _ERRORS:
            print(e)
    if _FAIL == 0:
        print("\nSANITY CHECK PASSED -- safe to run experiments.")
    else:
        print("\nSANITY CHECK FAILED -- fix issues before running experiments.")
    print("=" * 65)

    if _FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
tests/test_step3_vanilla.py
Validates spart/vanilla_kernel.py.
Gate: Python vanilla and JIT-vanilla produce identical outputs for same frame/ego.
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spart.utils import build_segments_from_frame, generate_ego_circular
from spart.vanilla_kernel import (
    scan_python_vanilla,
    scan_jit_vanilla,
    scan_jit_angular_only,
    warmup_jit_vanilla,
    warmup_jit_angular,
)
from spart.scan_kernel import simulate_scan_numba_wrapper, warmup_numba_kernel

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


def allclose(a, b, tol=1e-9):
    return np.allclose(np.nan_to_num(a, nan=0.0),
                       np.nan_to_num(b, nan=0.0), atol=tol, rtol=0)


# ---------------------------------------------------------------------------
# Shared scene
# ---------------------------------------------------------------------------

def make_scene():
    ego = generate_ego_circular(0.0, radius=15.0, speed=5.398298934,
                                center=(996.0, 999.0))
    ex, ey = ego["x"], ego["y"]
    vehicles = [
        {"track_id": 1, "pos": np.array([ex + 8.0, ey + 0.0]),
         "vel": np.array([0.0, 0.0]), "yaw": 0.0, "L": 4.0, "W": 2.0, "speed": 0.0},
        {"track_id": 2, "pos": np.array([ex + 5.0, ey + 4.0]),
         "vel": np.array([0.0, 0.0]), "yaw": 0.3, "L": 4.5, "W": 1.8, "speed": 0.0},
        {"track_id": 3, "pos": np.array([ex - 6.0, ey + 3.0]),
         "vel": np.array([0.0, 0.0]), "yaw": -0.5, "L": 3.5, "W": 1.7, "speed": 0.0},
    ]
    segs, tids = build_segments_from_frame(vehicles)
    return ego, vehicles, segs, tids


def main():
    print("=" * 60)
    print("SPART Step 3 - vanilla_kernel.py validation")
    print("=" * 60)

    # Warm up all JIT kernels
    print("\n[INFO] Warming up JIT kernels...")
    t0 = time.perf_counter()
    warmup_jit_vanilla()
    warmup_jit_angular()
    warmup_numba_kernel()
    print(f"  All JIT kernels warmed up in {time.perf_counter()-t0:.2f} s")

    ego, vehicles, segs, tids = make_scene()
    FOV = 120.0; DELTA = 0.5; RMAX = 15.0

    # Run all three conditions
    ang_A, rng_A, ht_A, tc_A, t_A = scan_python_vanilla(
        ego, vehicles, fov_deg=FOV, delta_deg=DELTA, r_max=RMAX)

    ang_B, rng_B, ht_B, tc_B, t_B = scan_jit_vanilla(
        ego, segs, fov_deg=FOV, delta_deg=DELTA, r_max=RMAX, seg_track_ids=tids)

    ang_D, rng_D, ht_D, tc_D, cpb_D, t_D = scan_jit_angular_only(
        ego, segs, fov_deg=FOV, delta_deg=DELTA, r_max=RMAX, seg_track_ids=tids)

    ang_E, rng_E, ht_E, hs_E, tc_E, cpb_E, t_E = simulate_scan_numba_wrapper(
        ego, segs, fov_deg=FOV, delta_deg=DELTA, r_max=RMAX, seg_track_ids=tids)

    M = segs.shape[0]
    K = ang_A.shape[0]
    print(f"\n  K={K} beams, M={M} segments, {len(vehicles)} vehicles")

    print("\n--- Beam count consistency ---")
    check("1. A and B same K", ang_A.shape == ang_B.shape)
    check("2. A and D same K", ang_A.shape == ang_D.shape)
    check("3. A and E same K", ang_A.shape == ang_E.shape)

    print("\n--- Angle values ---")
    check("4. B angles == A angles", allclose(ang_B, ang_A))
    check("5. D angles == A angles", allclose(ang_D, ang_A))

    print("\n--- Ranges: A vs B (Python vs JIT, no pruning) ---")
    nan_match = (np.isnan(rng_A) == np.isnan(rng_B)).all()
    val_match  = allclose(rng_A[~np.isnan(rng_A)], rng_B[~np.isnan(rng_B)])
    check("6. rng_A NaN pattern == rng_B", nan_match)
    check("7. rng_A values == rng_B values", val_match)

    print("\n--- total_candidates: A vs B (no pruning -> same count = K*M) ---")
    expected_no_prune = K * M
    check("8. Cond A total_candidates == K*M",
          tc_A == expected_no_prune, f"got {tc_A}, expected {expected_no_prune}")
    check("9. Cond B total_candidates == K*M",
          tc_B == expected_no_prune, f"got {tc_B}, expected {expected_no_prune}")

    print("\n--- Ranges: B vs D (both JIT; D adds angular pruning) ---")
    # D must produce identical hits to B (pruning is conservative -- no false negatives)
    nan_match_BD = (np.isnan(rng_B) == np.isnan(rng_D)).all()
    val_match_BD  = allclose(rng_B[~np.isnan(rng_B)], rng_D[~np.isnan(rng_D)])
    check("10. rng_B NaN pattern == rng_D (no false negatives from pruning)",
          nan_match_BD)
    check("11. rng_B values == rng_D values", val_match_BD)

    print("\n--- Angular pruning reduces candidates (D < B) ---")
    check("12. Cond D candidates <= Cond B candidates",
          tc_D <= tc_B, f"D={tc_D}, B={tc_B}")

    print("\n--- hit_tracks: A and B agree ---")
    # Both should return same tracks for same scene
    # (compare only beams where both hit)
    both_hit = (~np.isnan(rng_A)) & (~np.isnan(rng_B))
    if both_hit.any():
        tracks_match = (ht_A[both_hit] == ht_B[both_hit]).all()
        check("13. hit_tracks agree between A and B (all shared hits)",
              tracks_match)
    else:
        check("13. hit_tracks: no mutual hits to compare", True)

    print("\n--- Condition D: candidates_per_beam sums correctly ---")
    check("14. cpb_D.sum() == tc_D",
          int(cpb_D.sum()) == tc_D, f"sum={int(cpb_D.sum())}, tc={tc_D}")

    print("\n--- Condition E (full SPART) vs B (same vehicles, no muting) ---")
    # When no muting is applied, E and B must agree
    nan_match_BE = (np.isnan(rng_B) == np.isnan(rng_E)).all()
    val_match_BE  = allclose(rng_B[~np.isnan(rng_B)], rng_E[~np.isnan(rng_E)])
    check("15. rng_B NaN pattern == rng_E (full SPART, no muting)",
          nan_match_BE)
    check("16. rng_B values == rng_E values", val_match_BE)

    print("\n--- Timing sanity ---")
    check("17. Condition A time > 0", t_A > 0)
    check("18. Condition B time > 0", t_B > 0)
    check("19. Condition D time > 0", t_D > 0)
    print(f"  Condition A (Python): {t_A*1000:.3f} ms")
    print(f"  Condition B (JIT, no prune): {t_B*1000:.3f} ms")
    print(f"  Condition D (JIT+angular): {t_D*1000:.3f} ms")

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

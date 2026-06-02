"""
tests/test_step2_kernel.py
Validates spart/scan_kernel.py.
Gate: serial and parallel kernels agree to within float epsilon.
      total_candidates from parallel == serial count.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spart.scan_kernel import (
    simulate_scan_numba_core,
    simulate_scan_numba_serial,
    warmup_numba_kernel,
    simulate_scan_numba_wrapper,
)
from spart.utils import build_segments_from_frame, generate_ego_circular

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
# Reference scene: 3 vehicles placed around ego
# ---------------------------------------------------------------------------

def make_test_scene():
    """Deterministic scene for kernel validation."""
    ego = generate_ego_circular(0.0, radius=15.0, speed=5.398298934,
                                center=(996.0, 999.0))
    # Three vehicles at known positions within r_max=15 and FOV=120 deg
    ex, ey = ego["x"], ego["y"]
    vehicles = [
        {"track_id": 1, "pos": np.array([ex + 8.0,  ey + 0.0]),
         "vel": np.array([0.0, 0.0]), "yaw": 0.0, "L": 4.0, "W": 2.0, "speed": 0.0},
        {"track_id": 2, "pos": np.array([ex + 5.0,  ey + 5.0]),
         "vel": np.array([0.0, 0.0]), "yaw": 0.3, "L": 4.5, "W": 1.8, "speed": 0.0},
        {"track_id": 3, "pos": np.array([ex - 7.0,  ey + 3.0]),
         "vel": np.array([0.0, 0.0]), "yaw": -0.5, "L": 3.5, "W": 1.7, "speed": 0.0},
    ]
    segments, track_ids = build_segments_from_frame(vehicles)
    return ego, vehicles, segments, track_ids


def main():
    print("=" * 60)
    print("SPART Step 2 - scan_kernel.py validation")
    print("=" * 60)

    # Warmup JIT (measure compile time)
    print("\n[INFO] Warming up Numba JIT (first call triggers compilation)...")
    compile_time = warmup_numba_kernel(fov_deg=120.0, delta_deg=0.5, r_max=15.0)
    print(f"  JIT compile time: {compile_time:.3f} s  (one-time startup cost)")

    ego, vehicles, segments, track_ids = make_test_scene()
    fov   = math.radians(120.0)
    delta = math.radians(0.5)
    r_max = 15.0

    p1s = np.ascontiguousarray(segments[:, 0, :], dtype=np.float64)
    p2s = np.ascontiguousarray(segments[:, 1, :], dtype=np.float64)

    ego_x   = float(ego["x"])
    ego_y   = float(ego["y"])
    ego_psi = float(ego["psi_rad"])

    # Run both kernels
    ang_p, rng_p, hs_p, tc_p, cpb_p = simulate_scan_numba_core(
        ego_x, ego_y, ego_psi, fov, delta, p1s, p2s, r_max)
    ang_s, rng_s, hs_s, tc_s, cpb_s = simulate_scan_numba_serial(
        ego_x, ego_y, ego_psi, fov, delta, p1s, p2s, r_max)

    print("\n--- Kernel agreement (parallel vs serial) ---")

    # 1. Same number of beams
    check("1. K beams match", ang_p.shape == ang_s.shape,
          f"parallel={ang_p.shape}, serial={ang_s.shape}")

    # 2. Angles identical
    check("2. angles identical", allclose(ang_p, ang_s))

    # 3. Ranges identical (NaN-safe)
    nan_match = (np.isnan(rng_p) == np.isnan(rng_s)).all()
    val_match  = allclose(rng_p[~np.isnan(rng_p)], rng_s[~np.isnan(rng_s)])
    check("3. ranges identical (NaN-safe)", nan_match and val_match)

    # 4. Hit segment indices identical
    check("4. hit_segs identical", (hs_p == hs_s).all())

    # 5. total_candidates match between parallel and serial
    check("5. total_candidates: parallel == serial",
          tc_p == tc_s, f"parallel={tc_p}, serial={tc_s}")

    # 6. candidates_per_beam sum == total_candidates (parallel)
    check("6. cpb.sum() == total_candidates (parallel)",
          int(cpb_p.sum()) == int(tc_p),
          f"sum={int(cpb_p.sum())}, total={int(tc_p)}")

    # 7. candidates_per_beam sum == total_candidates (serial)
    check("7. cpb.sum() == total_candidates (serial)",
          int(cpb_s.sum()) == int(tc_s),
          f"sum={int(cpb_s.sum())}, total={int(tc_s)}")

    # 8. At least one beam hit (scene has vehicles in FOV)
    check("8. at least one hit", int(np.sum(~np.isnan(rng_p))) > 0,
          "no beams hit -- scene may be outside FOV")

    # 9. All ranges <= r_max
    valid_ranges = rng_p[~np.isnan(rng_p)]
    check("9. all ranges <= r_max", (valid_ranges <= r_max + 1e-9).all(),
          f"max_range={valid_ranges.max():.3f}")

    # 10. All ranges >= 0
    check("10. all ranges >= 0", (valid_ranges >= 0.0).all())

    print("\n--- Wrapper API ---")

    angles, ranges, hit_tracks, hit_segs, tc, cpb, t_sec = \
        simulate_scan_numba_wrapper(ego, segments, fov_deg=120.0, delta_deg=0.5,
                                    r_max=15.0, seg_track_ids=track_ids)

    # 11. Wrapper returns correct shapes
    K_expected = int(math.floor(fov / delta + 0.5)) + 1
    check("11. wrapper angles shape correct", angles.shape[0] == K_expected)

    # 12. Wrapper hit_tracks contains valid track IDs for hits
    hit_mask = ~np.isnan(ranges)
    valid_tids = hit_tracks[hit_mask]
    known_tids = set([v["track_id"] for v in vehicles])
    check("12. hit_tracks are known track IDs",
          all(int(t) in known_tids for t in valid_tids if int(t) >= 0))

    # 13. Wrapper scan time is positive
    check("13. scan_time_sec > 0", t_sec > 0.0, f"got {t_sec}")

    print("\n--- Empty scene ---")

    empty_segs = np.zeros((0, 2, 2), dtype=np.float64)
    angles_e, ranges_e, ht_e, hs_e, tc_e, cpb_e, t_e = \
        simulate_scan_numba_wrapper(ego, empty_segs, fov_deg=120.0, delta_deg=0.5,
                                    r_max=15.0)

    # 14. Empty scene: all NaN
    check("14. empty scene: all ranges NaN", np.all(np.isnan(ranges_e)))

    # 15. Empty scene: tc == 0
    check("15. empty scene: total_candidates == 0", tc_e == 0)

    print("\n--- Data race validation (multi-run) ---")
    # Run parallel kernel 5 times and check consistent total_candidates
    tc_vals = []
    for _ in range(5):
        _, _, _, tc_run, _ = simulate_scan_numba_core(
            ego_x, ego_y, ego_psi, fov, delta, p1s, p2s, r_max)
        tc_vals.append(int(tc_run))

    all_same = len(set(tc_vals)) == 1
    check("16. total_candidates deterministic across 5 runs",
          all_same, f"got values: {tc_vals}")

    # 17. Parallel == serial on a larger synthetic scene
    rng_gen = np.random.RandomState(42)
    big_vehicles = []
    ex_big = 0.0; ey_big = 0.0
    for i in range(15):
        ang = rng_gen.uniform(-math.pi, math.pi)
        dist = rng_gen.uniform(2.0, 12.0)
        cx = ex_big + dist * math.cos(ang)
        cy = ey_big + dist * math.sin(ang)
        big_vehicles.append({
            "track_id": i + 1,
            "pos": np.array([cx, cy]),
            "vel": np.array([0.0, 0.0]),
            "yaw": rng_gen.uniform(-math.pi, math.pi),
            "L": 4.0, "W": 2.0, "speed": 0.0
        })
    big_segs, _ = build_segments_from_frame(big_vehicles)
    bp1s = np.ascontiguousarray(big_segs[:, 0, :], dtype=np.float64)
    bp2s = np.ascontiguousarray(big_segs[:, 1, :], dtype=np.float64)

    _, brng_p, _, btc_p, _ = simulate_scan_numba_core(
        0.0, 0.0, 0.0, fov, delta, bp1s, bp2s, r_max)
    _, brng_s, _, btc_s, _ = simulate_scan_numba_serial(
        0.0, 0.0, 0.0, fov, delta, bp1s, bp2s, r_max)

    big_nan_match = (np.isnan(brng_p) == np.isnan(brng_s)).all()
    big_val_match  = allclose(brng_p[~np.isnan(brng_p)], brng_s[~np.isnan(brng_s)])
    check("17. 15-vehicle scene: parallel ranges == serial",
          big_nan_match and big_val_match)
    check("18. 15-vehicle scene: total_candidates match",
          btc_p == btc_s, f"parallel={btc_p}, serial={btc_s}")

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

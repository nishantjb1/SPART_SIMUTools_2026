"""
tests/test_step1_utils.py
29 validation checks for spart/utils.py.
Gate: all checks pass.
Usage:
    python tests/test_step1_utils.py
    python tests/test_step1_utils.py --csv vehicle_tracks_000.csv [csv2 ...]
"""

import argparse
import math
import os
import sys

import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spart.utils import (
    angle_diff,
    build_segments_from_frame,
    dataset_stats,
    generate_ego_circular,
    generate_ego_static,
    get_frame_list,
    get_segments_from_corners,
    get_segments_from_vehicle,
    get_vehicle_corners,
    load_interaction_csv,
    make_ego_trajectory,
    normalize_angle,
)

# ---------------------------------------------------------------------------
# Minimal test harness
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0
_ERRORS = []


def check(name: str, condition: bool, detail: str = ""):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        msg = f"  [FAIL] {name}" + (f" -- {detail}" if detail else "")
        print(msg)
        _ERRORS.append(msg)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def allclose(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol, rtol=0)


# ---------------------------------------------------------------------------
# Check group 1: normalize_angle (checks 1-6)
# ---------------------------------------------------------------------------

def test_normalize_angle():
    print("\n--- normalize_angle ---")

    # 1. Zero maps to zero
    check("1. normalize_angle(0) == 0", close(normalize_angle(0.0), 0.0))

    # 2. pi maps to pi (not -pi)
    check("2. normalize_angle(pi) == pi", close(normalize_angle(math.pi), math.pi))

    # 3. -pi maps to pi (edge case fix)
    check("3. normalize_angle(-pi) == pi", close(normalize_angle(-math.pi), math.pi))

    # 4. 3*pi/2 maps to -pi/2
    check("4. normalize_angle(3pi/2) == -pi/2",
          close(normalize_angle(3 * math.pi / 2), -math.pi / 2))

    # 5. -3*pi/2 maps to pi/2
    check("5. normalize_angle(-3pi/2) == pi/2",
          close(normalize_angle(-3 * math.pi / 2), math.pi / 2))

    # 6. 2*pi maps to 0 (not 2*pi)
    check("6. normalize_angle(2pi) == 0", close(normalize_angle(2 * math.pi), 0.0))


# ---------------------------------------------------------------------------
# Check group 2: angle_diff  (checks 7-8)
# ---------------------------------------------------------------------------

def test_angle_diff():
    print("\n--- angle_diff ---")

    # 7. angle_diff(pi, -pi) == 0 (they're the same angle)
    check("7. angle_diff(pi, -pi) == 0", close(angle_diff(math.pi, -math.pi), 0.0))

    # 8. angle_diff(0.1, -0.1) ~= 0.2
    check("8. angle_diff(0.1, -0.1) ~= 0.2", close(angle_diff(0.1, -0.1), 0.2))


# ---------------------------------------------------------------------------
# Check group 3: get_vehicle_corners  (checks 9-13)
# ---------------------------------------------------------------------------

def test_get_vehicle_corners():
    print("\n--- get_vehicle_corners ---")

    # 9. Returns (4,2) float64
    corners = get_vehicle_corners((0.0, 0.0), 0.0, 4.0, 2.0)
    check("9. corners shape (4,2)", corners.shape == (4, 2))

    # 10. float64 dtype
    check("10. corners dtype float64", corners.dtype == np.float64)

    # 11. yaw=0: front-right corner = (+hl, -hw) = (2, -1)
    check("11. front-right at (2,-1) for yaw=0",
          allclose(corners[0], [2.0, -1.0]))

    # 12. yaw=0: front-left corner = (+hl, +hw) = (2, 1)
    check("12. front-left at (2,1) for yaw=0",
          allclose(corners[1], [2.0, 1.0]))

    # 13. yaw=pi/2: front-right = rotate(2,-1) by 90 deg = (1, 2)
    c90 = get_vehicle_corners((0.0, 0.0), math.pi / 2, 4.0, 2.0)
    check("13. yaw=pi/2: first corner ~= (1,2)",
          allclose(c90[0], [1.0, 2.0], tol=1e-9))


# ---------------------------------------------------------------------------
# Check group 4: get_segments_from_corners  (checks 14-16)
# ---------------------------------------------------------------------------

def test_get_segments_from_corners():
    print("\n--- get_segments_from_corners ---")

    corners = get_vehicle_corners((0.0, 0.0), 0.0, 4.0, 2.0)
    segs = get_segments_from_corners(corners)

    # 14. Shape (4,2,2)
    check("14. segments shape (4,2,2)", segs.shape == (4, 2, 2))

    # 15. First segment p1 == first corner
    check("15. segs[0,0] == corners[0]", allclose(segs[0, 0], corners[0]))

    # 16. Last segment p2 wraps to corner[0]
    check("16. segs[3,1] == corners[0]", allclose(segs[3, 1], corners[0]))


# ---------------------------------------------------------------------------
# Check group 5: get_segments_from_vehicle  (check 17)
# ---------------------------------------------------------------------------

def test_get_segments_from_vehicle():
    print("\n--- get_segments_from_vehicle ---")

    v = {"track_id": 1, "pos": np.array([0.0, 0.0]),
         "vel": np.array([0.0, 0.0]), "yaw": 0.0, "L": 4.0, "W": 2.0, "speed": 0.0}
    segs = get_segments_from_vehicle(v)
    check("17. get_segments_from_vehicle returns (4,2,2)", segs.shape == (4, 2, 2))


# ---------------------------------------------------------------------------
# Check group 6: build_segments_from_frame  (checks 18-20)
# ---------------------------------------------------------------------------

def test_build_segments_from_frame():
    print("\n--- build_segments_from_frame ---")

    v1 = {"track_id": 1, "pos": np.array([0.0, 0.0]),
          "vel": np.array([0.0, 0.0]), "yaw": 0.0, "L": 4.0, "W": 2.0, "speed": 0.0}
    v2 = {"track_id": 2, "pos": np.array([10.0, 0.0]),
          "vel": np.array([0.0, 0.0]), "yaw": 0.0, "L": 4.0, "W": 2.0, "speed": 0.0}

    segs, tids = build_segments_from_frame([v1, v2])

    # 18. 2 vehicles -> 8 segments
    check("18. 2 vehicles -> 8 segments", segs.shape == (8, 2, 2))

    # 19. track_ids length matches
    check("19. track_ids length 8", len(tids) == 8)

    # 20. empty list -> (0,2,2) and empty track_ids
    segs0, tids0 = build_segments_from_frame([])
    check("20. empty frame -> 0 segments", segs0.shape == (0, 2, 2) and len(tids0) == 0)


# ---------------------------------------------------------------------------
# Check group 7: generate_ego_circular  (checks 21-22)
# ---------------------------------------------------------------------------

def test_generate_ego_circular():
    print("\n--- generate_ego_circular ---")

    state = generate_ego_circular(0.0, radius=15.0, speed=5.0, center=(100.0, 200.0))

    # 21. At t=0, position is center + (radius, 0)
    check("21. t=0 pos = center + (r,0)",
          close(state["x"], 115.0) and close(state["y"], 200.0))

    # 22. Speed magnitude matches requested speed
    spd = math.hypot(state["vx"], state["vy"])
    check("22. speed magnitude matches", close(spd, 5.0, tol=1e-6))


# ---------------------------------------------------------------------------
# Check group 8: make_ego_trajectory  (checks 23-24)
# ---------------------------------------------------------------------------

def test_make_ego_trajectory():
    print("\n--- make_ego_trajectory ---")

    frame_ids = [0, 1, 2]
    timestamps = {0: 0.0, 1: 0.1, 2: 0.2}

    # 23. Circular mode returns correct number of entries
    traj = make_ego_trajectory(frame_ids, timestamps, mode="circular",
                               radius=15.0, speed=5.0, center=(100.0, 200.0))
    check("23. circular trajectory covers all frame_ids",
          all(fid in traj for fid in frame_ids))

    # 24. All frames produce consistent speed
    speeds = [math.hypot(traj[fid]["vx"], traj[fid]["vy"]) for fid in frame_ids]
    check("24. all circular speeds ~= 5.0",
          all(close(s, 5.0, tol=1e-6) for s in speeds))


# ---------------------------------------------------------------------------
# Check group 9: get_frame_list  (check 25)
# ---------------------------------------------------------------------------

def test_get_frame_list():
    print("\n--- get_frame_list ---")

    fake_frames = {0: [], 1: [], 2: [], 5: [], 10: []}

    # 25. n_frames=3 returns first 3 sorted
    fl = get_frame_list(fake_frames, n_frames=3)
    check("25. get_frame_list n_frames=3 returns [0,1,2]", fl == [0, 1, 2])


# ---------------------------------------------------------------------------
# Check group 10: normalize_angle continuity at ±pi  (check 26)
# ---------------------------------------------------------------------------

def test_normalize_angle_continuity():
    print("\n--- normalize_angle continuity ---")

    # 26. Values just below pi map to values just below pi (not negative)
    eps = 1e-12
    val = normalize_angle(math.pi - eps)
    check("26. normalize_angle(pi-eps) is positive",
          val > 0, detail=f"got {val}")


# ---------------------------------------------------------------------------
# Checks 27-29: real CSV loading (optional, run with --csv flag)
# ---------------------------------------------------------------------------

def test_real_csv(csv_paths):
    print("\n--- Real CSV loading ---")

    if not csv_paths:
        print("  [SKIP] No CSV paths provided. Pass --csv path1 [path2 ...] to enable.")
        return

    for path in csv_paths:
        frames, ts = load_interaction_csv(path)
        stats = dataset_stats(frames, ts)
        name = os.path.basename(path)
        print(f"\n  CSV: {name}")
        print(f"    n_frames              = {stats['n_frames']}")
        print(f"    frame_id range        = {stats['frame_ids_range']}")
        print(f"    mean vehicles/frame   = {stats['mean_vehicles_per_frame']:.2f}")
        print(f"    max vehicles/frame    = {stats['max_vehicles_per_frame']}")
        print(f"    x_range               = {stats['x_range']}")
        print(f"    y_range               = {stats['y_range']}")
        print(f"    timestamp_range (s)   = {stats['timestamp_range_s']}")

    # Use the first CSV for gate checks
    first_path = csv_paths[0]
    frames, ts = load_interaction_csv(first_path)

    # 27. frames_dict is non-empty
    check("27. frames_dict non-empty", len(frames) > 0)

    # 28. Each frame has at least one vehicle
    all_nonempty = all(len(v) > 0 for v in frames.values())
    check("28. all frames have >=1 vehicle", all_nonempty)

    # 29. get_frame_list on real data is sorted
    fl = get_frame_list(frames)
    check("29. frame_list is sorted", fl == sorted(fl))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 1 utility tests")
    parser.add_argument("--csv", nargs="*", default=[], help="Paths to INTERACTION CSVs")
    args = parser.parse_args()

    print("=" * 60)
    print("SPART Step 1 - spart/utils.py validation (29 checks)")
    print("=" * 60)

    test_normalize_angle()
    test_angle_diff()
    test_get_vehicle_corners()
    test_get_segments_from_corners()
    test_get_segments_from_vehicle()
    test_build_segments_from_frame()
    test_generate_ego_circular()
    test_make_ego_trajectory()
    test_get_frame_list()
    test_normalize_angle_continuity()
    test_real_csv(args.csv)

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

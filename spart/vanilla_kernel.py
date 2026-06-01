"""
spart/vanilla_kernel.py
Baseline kernels for ablation study conditions A, B, and D.

Condition A: scan_python_vanilla  -- pure Python, no JIT, no pruning
Condition B: scan_jit_vanilla     -- Numba JIT, no pruning
Condition D: scan_jit_angular_only -- JIT + angular pruning, no temporal muting

All use build_segments_from_frame from utils.py for segment construction.
Condition C (JIT + temporal muting only) is handled in core.py by running
scan_jit_vanilla on eligibility-filtered vehicles.
"""

import math
import time
from typing import List, Tuple

import numpy as np
from numba import njit, float64, int64

from spart.utils import build_segments_from_frame
from spart.scan_kernel import segment_angular_interval


# ---------------------------------------------------------------------------
# Condition A: Pure Python vanilla (no JIT, no pruning)
# ---------------------------------------------------------------------------

def scan_python_vanilla(ego_state: dict, vehicles: List[dict],
                        fov_deg: float = 120.0,
                        delta_deg: float = 0.5,
                        r_max: float = 15.0):
    """
    Condition A: pure Python, no JIT, no angular pruning.
    Tests every segment against every beam -- O(K*M).

    Returns
    -------
    angles : (K,) float64
    ranges : (K,) float64 -- NaN for misses
    hit_tracks : (K,) int -- -1 for misses
    total_candidates : int  (= K * M always)
    scan_time_sec : float
    """
    fov   = math.radians(fov_deg)
    delta = math.radians(delta_deg)
    ego_x   = float(ego_state["x"])
    ego_y   = float(ego_state["y"])
    ego_psi = float(ego_state["psi_rad"])

    segments, track_ids = build_segments_from_frame(vehicles)
    M = segments.shape[0]
    half = fov / 2.0
    K = int(math.floor(fov / delta + 0.5)) + 1

    angles    = np.array([ego_psi - half + k * delta for k in range(K)], dtype=np.float64)
    ranges    = np.full(K, np.nan, dtype=np.float64)
    hit_tracks = np.full(K, -1, dtype=np.int64)

    total_candidates = 0

    t0 = time.perf_counter()

    for k in range(K):
        theta = angles[k]
        d0 = math.cos(theta)
        d1 = math.sin(theta)
        best_t = None
        best_tid = -1

        for i in range(M):
            total_candidates += 1  # count every test (no pruning)
            p1x = float(segments[i, 0, 0])
            p1y = float(segments[i, 0, 1])
            p2x = float(segments[i, 1, 0])
            p2y = float(segments[i, 1, 1])
            sx = p2x - p1x; sy = p2y - p1y
            rx = p1x - ego_x; ry = p1y - ego_y
            denom = d0 * sy - d1 * sx

            if abs(denom) > 1e-9:
                t_i = (rx * sy - ry * sx) / denom
                u_i = (rx * d1 - ry * d0) / denom
                if 0.0 <= t_i <= r_max and 0.0 <= u_i <= 1.0:
                    if best_t is None or t_i < best_t:
                        best_t = t_i
                        best_tid = int(track_ids[i])
            else:
                cross = rx * d1 - ry * d0
                if abs(cross) <= 1e-9:
                    t0c = rx * d0 + ry * d1
                    t1c = (p2x - ego_x) * d0 + (p2y - ego_y) * d1
                    tmin = min(t0c, t1c); tmax = max(t0c, t1c)
                    if tmax >= 0.0 and tmin <= r_max:
                        t_hit = max(0.0, tmin)
                        if best_t is None or t_hit < best_t:
                            best_t = t_hit
                            best_tid = int(track_ids[i])

        if best_t is not None:
            ranges[k]     = best_t
            hit_tracks[k] = best_tid

    scan_time_sec = time.perf_counter() - t0
    return angles, ranges, hit_tracks, total_candidates, scan_time_sec


# ---------------------------------------------------------------------------
# Condition B: JIT vanilla -- JIT, no pruning
# ---------------------------------------------------------------------------

@njit
def _jit_vanilla_core(ego_x: float, ego_y: float, ego_psi: float,
                      fov: float, delta: float,
                      p1s: np.ndarray, p2s: np.ndarray,
                      r_max: float):
    """
    JIT kernel with no angular pruning (tests all M segments per beam).
    Single-threaded to avoid conflating JIT gain with parallelism.
    """
    half = fov / 2.0
    K = int(math.floor(fov / delta + 0.5)) + 1
    M = p1s.shape[0]

    angles   = np.empty(K, dtype=float64)
    for k in range(K):
        angles[k] = ego_psi - half + k * delta

    ranges    = np.full(K, np.nan, dtype=float64)
    hit_segs  = np.full(K, -1, dtype=int64)
    total_candidates = int64(0)

    for k in range(K):
        theta = angles[k]
        d0 = math.cos(theta); d1 = math.sin(theta)
        best_t  = 1e99
        best_idx = -1

        for i in range(M):
            total_candidates += int64(1)  # no pruning: test every segment
            p1x = p1s[i, 0]; p1y = p1s[i, 1]
            p2x = p2s[i, 0]; p2y = p2s[i, 1]
            sx = p2x - p1x; sy = p2y - p1y
            rx = p1x - ego_x; ry = p1y - ego_y
            denom = d0 * sy - d1 * sx

            if abs(denom) > 1e-9:
                t_i = (rx * sy - ry * sx) / denom
                u_i = (rx * d1 - ry * d0) / denom
                if 0.0 <= t_i <= r_max and 0.0 <= u_i <= 1.0:
                    if t_i < best_t:
                        best_t = t_i; best_idx = i
            else:
                cross = rx * d1 - ry * d0
                if abs(cross) <= 1e-9:
                    t0c = rx * d0 + ry * d1
                    t1c = (p2x - ego_x) * d0 + (p2y - ego_y) * d1
                    tmin = t0c if t0c < t1c else t1c
                    tmax = t1c if t1c > t0c else t0c
                    if tmax >= 0.0 and tmin <= r_max:
                        t_hit = tmin if tmin >= 0.0 else 0.0
                        if t_hit < best_t:
                            best_t = t_hit; best_idx = i

        if best_idx != -1:
            ranges[k]   = best_t
            hit_segs[k] = best_idx

    return angles, ranges, hit_segs, total_candidates


def warmup_jit_vanilla(fov_deg: float = 120.0, delta_deg: float = 0.5,
                       r_max: float = 15.0) -> float:
    """Warm up _jit_vanilla_core. Returns compile time in seconds."""
    fov   = math.radians(fov_deg)
    delta = math.radians(delta_deg)
    p1s = np.array([[5.0, 0.0]], dtype=np.float64)
    p2s = np.array([[5.0, 1.0]], dtype=np.float64)
    t0 = time.perf_counter()
    _jit_vanilla_core(0.0, 0.0, 0.0, fov, delta, p1s, p2s, r_max)
    return time.perf_counter() - t0


def scan_jit_vanilla(ego_state: dict, segments: np.ndarray,
                     fov_deg: float = 120.0,
                     delta_deg: float = 0.5,
                     r_max: float = 15.0,
                     seg_track_ids: np.ndarray = None):
    """
    Condition B: Numba JIT, no angular pruning (all M tested per beam).

    Returns
    -------
    angles, ranges, hit_tracks, total_candidates, scan_time_sec
    """
    fov   = math.radians(fov_deg)
    delta = math.radians(delta_deg)
    ego_x   = float(ego_state["x"])
    ego_y   = float(ego_state["y"])
    ego_psi = float(ego_state["psi_rad"])

    M = segments.shape[0]
    half = fov / 2.0
    K = int(math.floor(fov / delta + 0.5)) + 1

    if M == 0:
        angles = np.array([ego_psi - half + k * delta for k in range(K)], dtype=np.float64)
        return angles, np.full(K, np.nan), np.full(K, -1, dtype=np.int64), 0, 0.0

    p1s = np.ascontiguousarray(segments[:, 0, :], dtype=np.float64)
    p2s = np.ascontiguousarray(segments[:, 1, :], dtype=np.float64)

    t0 = time.perf_counter()
    angles, ranges, hit_segs_raw, total_candidates = \
        _jit_vanilla_core(ego_x, ego_y, ego_psi, fov, delta, p1s, p2s, r_max)
    scan_time_sec = time.perf_counter() - t0

    hit_tracks = np.full(K, -1, dtype=np.int64)
    if seg_track_ids is not None:
        for k in range(K):
            si = int(hit_segs_raw[k])
            if si >= 0:
                hit_tracks[k] = int(seg_track_ids[si])

    return angles, ranges, hit_tracks, int(total_candidates), scan_time_sec


# ---------------------------------------------------------------------------
# Condition D: JIT + angular pruning only (no temporal muting)
# ---------------------------------------------------------------------------

@njit
def _jit_angular_core(ego_x: float, ego_y: float, ego_psi: float,
                      fov: float, delta: float,
                      p1s: np.ndarray, p2s: np.ndarray,
                      r_max: float):
    """
    JIT kernel WITH angular interval pruning but NO temporal muting.
    All vehicles are passed in (no eligibility filtering).
    Angular pruning skips segments whose angular interval does not
    overlap the beam direction -- reduces intersection tests.
    """
    half = fov / 2.0
    K = int(math.floor(fov / delta + 0.5)) + 1
    M = p1s.shape[0]

    angles = np.empty(K, dtype=float64)
    for k in range(K):
        angles[k] = ego_psi - half + k * delta

    ranges   = np.full(K, np.nan, dtype=float64)
    hit_segs = np.full(K, -1, dtype=int64)
    candidates_per_beam = np.zeros(K, dtype=int64)

    # Precompute angular intervals for all segments
    seg_amin = np.empty(M, dtype=float64)
    seg_amax = np.empty(M, dtype=float64)
    for i in range(M):
        amin, amax = segment_angular_interval(p1s[i, 0], p1s[i, 1],
                                              p2s[i, 0], p2s[i, 1],
                                              ego_x, ego_y)
        seg_amin[i] = amin
        seg_amax[i] = amax

    for k in range(K):
        theta = angles[k]
        d0 = math.cos(theta); d1 = math.sin(theta)
        best_t  = 1e99
        best_idx = -1
        local_count = int64(0)

        tc = theta % (2.0 * math.pi)
        if tc > math.pi:
            tc -= 2.0 * math.pi

        for i in range(M):
            amin = seg_amin[i]; amax = seg_amax[i]
            inside = False
            if amin <= amax:
                if amin <= tc <= amax:
                    inside = True
            else:
                if tc >= amin or tc <= amax:
                    inside = True
            if not inside:
                continue

            local_count += 1

            p1x = p1s[i, 0]; p1y = p1s[i, 1]
            p2x = p2s[i, 0]; p2y = p2s[i, 1]
            sx = p2x - p1x; sy = p2y - p1y
            rx = p1x - ego_x; ry = p1y - ego_y
            denom = d0 * sy - d1 * sx

            if abs(denom) > 1e-9:
                t_i = (rx * sy - ry * sx) / denom
                u_i = (rx * d1 - ry * d0) / denom
                if 0.0 <= t_i <= r_max and 0.0 <= u_i <= 1.0:
                    if t_i < best_t:
                        best_t = t_i; best_idx = i
            else:
                cross = rx * d1 - ry * d0
                if abs(cross) <= 1e-9:
                    t0c = rx * d0 + ry * d1
                    t1c = (p2x - ego_x) * d0 + (p2y - ego_y) * d1
                    tmin = t0c if t0c < t1c else t1c
                    tmax = t1c if t1c > t0c else t0c
                    if tmax >= 0.0 and tmin <= r_max:
                        t_hit = tmin if tmin >= 0.0 else 0.0
                        if t_hit < best_t:
                            best_t = t_hit; best_idx = i

        candidates_per_beam[k] = local_count
        if best_idx != -1:
            ranges[k]   = best_t
            hit_segs[k] = best_idx

    total_candidates = int64(0)
    for k in range(K):
        total_candidates += candidates_per_beam[k]

    return angles, ranges, hit_segs, total_candidates, candidates_per_beam


def warmup_jit_angular(fov_deg: float = 120.0, delta_deg: float = 0.5,
                       r_max: float = 15.0) -> float:
    """Warm up _jit_angular_core. Returns compile time in seconds."""
    fov   = math.radians(fov_deg)
    delta = math.radians(delta_deg)
    p1s = np.array([[5.0, 0.0]], dtype=np.float64)
    p2s = np.array([[5.0, 1.0]], dtype=np.float64)
    t0 = time.perf_counter()
    _jit_angular_core(0.0, 0.0, 0.0, fov, delta, p1s, p2s, r_max)
    return time.perf_counter() - t0


def scan_jit_angular_only(ego_state: dict, segments: np.ndarray,
                           fov_deg: float = 120.0,
                           delta_deg: float = 0.5,
                           r_max: float = 15.0,
                           seg_track_ids: np.ndarray = None):
    """
    Condition D: JIT + angular pruning, no temporal muting.
    All segments passed in; angular interval check applied inside kernel.

    Returns
    -------
    angles, ranges, hit_tracks, total_candidates, candidates_per_beam, scan_time_sec
    """
    fov   = math.radians(fov_deg)
    delta = math.radians(delta_deg)
    ego_x   = float(ego_state["x"])
    ego_y   = float(ego_state["y"])
    ego_psi = float(ego_state["psi_rad"])

    M = segments.shape[0]
    half = fov / 2.0
    K = int(math.floor(fov / delta + 0.5)) + 1

    if M == 0:
        angles = np.array([ego_psi - half + k * delta for k in range(K)], dtype=np.float64)
        cpb = np.zeros(K, dtype=np.int64)
        return angles, np.full(K, np.nan), np.full(K, -1, dtype=np.int64), 0, cpb, 0.0

    p1s = np.ascontiguousarray(segments[:, 0, :], dtype=np.float64)
    p2s = np.ascontiguousarray(segments[:, 1, :], dtype=np.float64)

    t0 = time.perf_counter()
    angles, ranges, hit_segs_raw, total_candidates, candidates_per_beam = \
        _jit_angular_core(ego_x, ego_y, ego_psi, fov, delta, p1s, p2s, r_max)
    scan_time_sec = time.perf_counter() - t0

    hit_tracks = np.full(K, -1, dtype=np.int64)
    if seg_track_ids is not None:
        for k in range(K):
            si = int(hit_segs_raw[k])
            if si >= 0:
                hit_tracks[k] = int(seg_track_ids[si])

    return angles, ranges, hit_tracks, int(total_candidates), candidates_per_beam, scan_time_sec

"""
spart/scan_kernel.py
Numba JIT scan kernel with data race fixed.

Bug fix vs original notebook:
  Original: total_candidates += 1 inside prange(K) -- non-atomic write, race condition.
  Fixed:    candidates_per_beam[k] accumulates a thread-local counter; summed after prange.
"""

import math
import time
from typing import Tuple

import numpy as np
from numba import njit, prange
from numba import float64, int64


# ---------------------------------------------------------------------------
# Angular interval helper (nopython)
# ---------------------------------------------------------------------------

@njit
def segment_angular_interval(p1x: float, p1y: float,
                              p2x: float, p2y: float,
                              ox: float, oy: float) -> Tuple[float, float]:
    """
    Compute angular interval [amin, amax] that a segment (p1, p2) subtends
    as seen from observer at (ox, oy).

    Handles wraparound: if the segment straddles the -pi/pi boundary, amin > amax
    (caller must treat as [amin, pi] union [-pi, amax]).

    Special cases:
      - Endpoint at observer: returns (-pi, pi) (full circle, conservative).
      - Near-collinear bearings (diff < tol): returns interval of width tol.
    """
    EPS = 1e-9
    a1 = math.atan2(p1y - oy, p1x - ox)
    a2 = math.atan2(p2y - oy, p2x - ox)

    # normalize to (-pi, pi]
    a1 = ((a1 + math.pi) % (2 * math.pi)) - math.pi
    a2 = ((a2 + math.pi) % (2 * math.pi)) - math.pi

    # Handle near-identical bearings
    da = a2 - a1
    if da > math.pi:
        da -= 2 * math.pi
    elif da < -math.pi:
        da += 2 * math.pi

    if abs(da) < EPS:
        # Collinear bearings -- return a tiny interval
        amin = a1 - EPS
        amax = a1 + EPS
        return amin, amax

    # Non-wrapping case: choose shorter arc from a1 to a2
    if da >= 0:
        # a2 is CCW from a1; shorter arc is [a1, a2]
        if a1 <= a2:
            return a1, a2
        else:
            # crosses -pi/pi boundary
            return a1, a2   # a1 > a2 signals wraparound to caller
    else:
        # a2 is CW from a1; shorter arc is [a2, a1]
        if a2 <= a1:
            return a2, a1
        else:
            return a2, a1   # a2 > a1 signals wraparound


# ---------------------------------------------------------------------------
# Core parallel scan kernel (fixed data race)
# ---------------------------------------------------------------------------

@njit(parallel=True)
def simulate_scan_numba_core(ego_x: float, ego_y: float, ego_psi: float,
                              fov: float, delta: float,
                              p1s: np.ndarray, p2s: np.ndarray,
                              r_max: float):
    """
    Parallel Numba scan kernel.

    Parameters
    ----------
    ego_x, ego_y, ego_psi : float
        Ego position and heading.
    fov : float
        Field of view in radians.
    delta : float
        Angular step between beams in radians.
    p1s, p2s : ndarray (M, 2) float64
        Segment endpoints.
    r_max : float
        Maximum range.

    Returns
    -------
    angles : (K,) float64
    ranges : (K,) float64   -- NaN for misses
    hit_segs : (K,) int64   -- -1 for misses
    total_candidates : int64
    candidates_per_beam : (K,) int64
    """
    half = fov / 2.0
    K = int(math.floor(fov / delta + 0.5)) + 1
    M = p1s.shape[0]

    angles = np.empty(K, dtype=float64)
    for k in range(K):
        angles[k] = ego_psi - half + k * delta

    ranges   = np.full(K, np.nan, dtype=float64)
    hit_segs = np.full(K, -1, dtype=int64)
    candidates_per_beam = np.zeros(K, dtype=int64)   # FIX: per-beam, summed after

    # Precompute segment angular intervals
    seg_amin = np.empty(M, dtype=float64)
    seg_amax = np.empty(M, dtype=float64)
    for i in range(M):
        amin, amax = segment_angular_interval(p1s[i, 0], p1s[i, 1],
                                              p2s[i, 0], p2s[i, 1],
                                              ego_x, ego_y)
        seg_amin[i] = amin
        seg_amax[i] = amax

    # Per-beam loop -- safe to parallelize: each beam writes only its own index
    for k in prange(K):
        theta = angles[k]
        d0 = math.cos(theta)
        d1 = math.sin(theta)
        best_t = 1e99
        best_idx = -1
        local_count = int64(0)   # FIX: thread-local accumulator

        # Normalise theta to (-pi, pi] so it matches segment angular intervals.
        # Beam angles theta = ego_psi +/- fov/2 can fall outside (-pi,pi] when
        # ego_psi is near +/-pi.  seg_amin/amax are always in (-pi,pi].
        tc = theta % (2.0 * math.pi)   # [0, 2pi)
        if tc > math.pi:
            tc -= 2.0 * math.pi        # (-pi, pi]

        for i in range(M):
            amin = seg_amin[i]
            amax = seg_amax[i]

            # Angular interval check (handles wraparound)
            inside = False
            if amin <= amax:
                if amin <= tc <= amax:
                    inside = True
            else:
                # Wraparound: interval is [amin, pi] union [-pi, amax]
                if tc >= amin or tc <= amax:
                    inside = True

            if not inside:
                continue

            local_count += 1   # FIX: local, no race

            # Ray-segment intersection
            p1x = p1s[i, 0]; p1y = p1s[i, 1]
            p2x = p2s[i, 0]; p2y = p2s[i, 1]
            sx = p2x - p1x;  sy = p2y - p1y
            rx = p1x - ego_x; ry = p1y - ego_y
            denom = d0 * sy - d1 * sx

            if abs(denom) > 1e-9:
                t_i = (rx * sy - ry * sx) / denom
                u_i = (rx * d1 - ry * d0) / denom
                if 0.0 <= t_i <= r_max and 0.0 <= u_i <= 1.0:
                    if t_i < best_t:
                        best_t = t_i
                        best_idx = i
            else:
                # Parallel or collinear
                cross = rx * d1 - ry * d0
                if abs(cross) <= 1e-9:
                    t0 = rx * d0 + ry * d1
                    t1 = (p2x - ego_x) * d0 + (p2y - ego_y) * d1
                    tmin = t0 if t0 < t1 else t1
                    tmax = t1 if t1 > t0 else t0
                    if tmax >= 0.0 and tmin <= r_max:
                        t_hit = tmin if tmin >= 0.0 else 0.0
                        if t_hit < best_t:
                            best_t = t_hit
                            best_idx = i

        candidates_per_beam[k] = local_count   # FIX: write after inner loop

        if best_idx != -1:
            ranges[k]   = best_t
            hit_segs[k] = best_idx

    total_candidates = int64(0)
    for k in range(K):
        total_candidates += candidates_per_beam[k]

    return angles, ranges, hit_segs, total_candidates, candidates_per_beam


# ---------------------------------------------------------------------------
# Serial reference kernel (for validation -- no prange)
# ---------------------------------------------------------------------------

@njit
def simulate_scan_numba_serial(ego_x: float, ego_y: float, ego_psi: float,
                                fov: float, delta: float,
                                p1s: np.ndarray, p2s: np.ndarray,
                                r_max: float):
    """
    Identical logic to simulate_scan_numba_core but single-threaded (no prange).
    Used for serial/parallel agreement validation in tests.
    Returns same tuple as core kernel.
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

    seg_amin = np.empty(M, dtype=float64)
    seg_amax = np.empty(M, dtype=float64)
    for i in range(M):
        amin, amax = segment_angular_interval(p1s[i, 0], p1s[i, 1],
                                              p2s[i, 0], p2s[i, 1],
                                              ego_x, ego_y)
        seg_amin[i] = amin
        seg_amax[i] = amax

    for k in range(K):   # serial loop, no prange
        theta = angles[k]
        d0 = math.cos(theta)
        d1 = math.sin(theta)
        best_t = 1e99
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
            sx = p2x - p1x;  sy = p2y - p1y
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
                    t0 = rx * d0 + ry * d1
                    t1 = (p2x - ego_x) * d0 + (p2y - ego_y) * d1
                    tmin = t0 if t0 < t1 else t1
                    tmax = t1 if t1 > t0 else t0
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


# ---------------------------------------------------------------------------
# Warmup: trigger JIT compilation and measure compile time
# ---------------------------------------------------------------------------

def warmup_numba_kernel(fov_deg: float = 120.0,
                        delta_deg: float = 0.5,
                        r_max: float = 15.0) -> float:
    """
    Run a dummy call on minimal data to trigger JIT compilation.
    Returns compile time in seconds (one-time startup cost).
    Reports this separately from per-frame timing in all experiments.
    """
    fov   = math.radians(fov_deg)
    delta = math.radians(delta_deg)
    # Two-segment dummy scene
    p1s = np.array([[10.0, 0.0], [5.0, 3.0]], dtype=np.float64)
    p2s = np.array([[10.0, 2.0], [5.0, -3.0]], dtype=np.float64)

    t0 = time.perf_counter()
    simulate_scan_numba_core(0.0, 0.0, 0.0, fov, delta, p1s, p2s, r_max)
    simulate_scan_numba_serial(0.0, 0.0, 0.0, fov, delta, p1s, p2s, r_max)
    compile_time = time.perf_counter() - t0
    return compile_time


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

def simulate_scan_numba_wrapper(ego_state: dict,
                                segments: np.ndarray,
                                fov_deg: float = 120.0,
                                delta_deg: float = 0.5,
                                r_max: float = 15.0,
                                seg_track_ids: np.ndarray = None,
                                parallel: bool = False):
    """
    High-level wrapper. Takes ego dict + (M,2,2) segments array.

    Parameters
    ----------
    parallel : bool
        If True, use the prange parallel kernel (faster when M_processed is large,
        e.g. 50+ segments / 12+ vehicles).  If False (default), use the serial
        kernel which has lower per-call overhead and is faster for sparse scenes.

    Returns
    -------
    angles : (K,) rad
    ranges : (K,) m, NaN for misses
    hit_tracks : (K,) int, -1 for misses
    hit_segs : (K,) int (local index), -1 for misses
    total_candidates : int
    candidates_per_beam : (K,) int
    scan_time_sec : float
    """
    fov   = math.radians(fov_deg)
    delta = math.radians(delta_deg)
    ego_x = float(ego_state["x"])
    ego_y = float(ego_state["y"])
    ego_psi = float(ego_state["psi_rad"])

    M = segments.shape[0]
    half = fov / 2.0
    K = int(math.floor(fov / delta + 0.5)) + 1

    if M == 0:
        angles = np.linspace(ego_psi - half, ego_psi - half + (K - 1) * delta, K)
        ranges    = np.full(K, np.nan, dtype=np.float64)
        hit_tracks = np.full(K, -1, dtype=np.int64)
        hit_segs   = np.full(K, -1, dtype=np.int64)
        cpb = np.zeros(K, dtype=np.int64)
        return angles, ranges, hit_tracks, hit_segs, 0, cpb, 0.0

    p1s = np.ascontiguousarray(segments[:, 0, :], dtype=np.float64)
    p2s = np.ascontiguousarray(segments[:, 1, :], dtype=np.float64)

    t0 = time.perf_counter()
    if parallel:
        angles, ranges, hit_segs_raw, total_candidates, candidates_per_beam = \
            simulate_scan_numba_core(ego_x, ego_y, ego_psi, fov, delta, p1s, p2s, r_max)
    else:
        angles, ranges, hit_segs_raw, total_candidates, candidates_per_beam = \
            simulate_scan_numba_serial(ego_x, ego_y, ego_psi, fov, delta, p1s, p2s, r_max)
    scan_time_sec = time.perf_counter() - t0

    # Map segment indices → track IDs (vectorized, avoids K-iteration Python loop)
    hit_tracks = np.full(K, -1, dtype=np.int64)
    if seg_track_ids is not None:
        valid = hit_segs_raw >= 0
        if valid.any():
            hit_tracks[valid] = seg_track_ids[hit_segs_raw[valid]]

    return (angles, ranges, hit_tracks, hit_segs_raw,
            int(total_candidates), candidates_per_beam, scan_time_sec)

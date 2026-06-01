"""
spart/core.py
Main SPART class orchestrating the full pipeline:
  eligibility check -> (optional grid) -> segment build -> scan kernel -> eligibility update -> metrics

Exposes:
  SPART(config)
  SPART.run_frame(frame_vehicles, ego_state, tcurr) -> (angles, ranges, hit_tracks, metrics)
  SPART.run_dataset(frames_dict, timestamps, ego_traj) -> SPARTResults

Timing is recorded at two scopes per frame:
  time_scan_sec         -- Numba kernel only
  time_fullpipeline_sec -- entire run_frame call (eligibility + build + scan + update)
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from spart.utils import (
    build_segments_from_frame,
    get_frame_list,
    get_memory_mb,
)
from spart.scan_kernel import simulate_scan_numba_wrapper


# ---------------------------------------------------------------------------
# Temporal eligibility helpers
# ---------------------------------------------------------------------------

def _estimate_next_revisit(ego_pos: np.ndarray, ego_vel: np.ndarray,
                            veh_pos: np.ndarray, veh_vel: np.ndarray,
                            r_max: float, tcurr: float,
                            eps: float = 1e-6) -> float:
    """
    Estimate the next time a vehicle will enter range r_max of the ego.
    If already within range, return tcurr (immediately eligible).
    If closing velocity is zero or negative, return tcurr + 0.2 (short delay).
    """
    rel = veh_pos - ego_pos
    dist = float(np.linalg.norm(rel))
    if dist <= r_max + eps:
        return tcurr
    dir_unit = rel / (dist + eps)
    v_rel_along = float(np.dot(veh_vel - ego_vel, dir_unit))
    if v_rel_along <= eps:
        return tcurr + 0.2
    t_est = (dist - r_max) / v_rel_along
    return tcurr + max(0.0, t_est)


# ---------------------------------------------------------------------------
# Optional spatial grid (used only when enable_grid=True)
# ---------------------------------------------------------------------------

def _make_grid(xmin, xmax, ymin, ymax, cell_size):
    nx = max(1, int(math.ceil((xmax - xmin) / cell_size)))
    ny = max(1, int(math.ceil((ymax - ymin) / cell_size)))
    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
            "cell_size": cell_size, "nx": nx, "ny": ny}


def _cell_ix(x, y, grid):
    i = int((x - grid["xmin"]) // grid["cell_size"])
    j = int((y - grid["ymin"]) // grid["cell_size"])
    i = max(0, min(i, grid["nx"] - 1))
    j = max(0, min(j, grid["ny"] - 1))
    return i, j


def _sector_bbox(ego_pos, ego_psi, fov, r_max):
    cx, cy = float(ego_pos[0]), float(ego_pos[1])
    ang1 = ego_psi - fov / 2.0
    ang2 = ego_psi + fov / 2.0
    pts = [(cx + r_max * math.cos(a), cy + r_max * math.sin(a))
           for a in [ang1, ang2, (ang1 + ang2) / 2.0]]
    xs = [p[0] for p in pts] + [cx]
    ys = [p[1] for p in pts] + [cy]
    return min(xs), max(xs), min(ys), max(ys)


def _grid_candidates(vehicles, grid, ego_pos, ego_psi, fov, r_max):
    """Return indices of vehicles likely in FOV/range using grid cell lookup."""
    xmn, xmx, ymn, ymx = _sector_bbox(ego_pos, ego_psi, fov, r_max)
    i0, j0 = _cell_ix(xmn, ymn, grid)
    i1, j1 = _cell_ix(xmx, ymx, grid)
    cells: Dict[Tuple[int, int], List[int]] = {}
    for idx, v in enumerate(vehicles):
        ci, cj = _cell_ix(float(v["pos"][0]), float(v["pos"][1]), grid)
        cells.setdefault((ci, cj), []).append(idx)
    cands = set()
    for ci in range(i0, i1 + 1):
        for cj in range(j0, j1 + 1):
            cands.update(cells.get((ci, cj), []))
    # Coarse angular filter
    cx, cy = float(ego_pos[0]), float(ego_pos[1])
    result = []
    slack = math.radians(2.0)
    for idx in cands:
        v = vehicles[idx]
        dx = float(v["pos"][0]) - cx
        dy = float(v["pos"][1]) - cy
        dist = math.hypot(dx, dy)
        if dist > r_max * 1.5:
            continue
        ang = math.atan2(dy, dx)
        rel = ((ang - ego_psi + math.pi) % (2 * math.pi)) - math.pi
        if abs(rel) <= fov / 2.0 + slack:
            result.append(idx)
    if not result:
        result = [i for i, v in enumerate(vehicles)
                  if math.hypot(float(v["pos"][0]) - cx,
                                float(v["pos"][1]) - cy) <= r_max * 1.2]
    return result


# ---------------------------------------------------------------------------
# SPARTResults dataclass-like container
# ---------------------------------------------------------------------------

class SPARTResults:
    """Container for per-frame results from a full dataset run."""

    def __init__(self):
        self.rows: List[dict] = []

    def append(self, row: dict):
        self.rows.append(row)

    def as_records(self) -> List[dict]:
        return list(self.rows)


# ---------------------------------------------------------------------------
# Main SPART class
# ---------------------------------------------------------------------------

class SPART:
    """
    SPART pipeline orchestrator.

    Parameters
    ----------
    config : dict
        Keys: fov_deg, delta_deg, r_max_m, numba_threads (optional),
              enable_grid (bool), cell_size (float, only if enable_grid),
              grid_padding (float, optional).
    """

    def __init__(self, config: dict, track_memory: bool = True):
        self.fov_deg   = float(config.get("fov_deg",   120.0))
        self.delta_deg = float(config.get("delta_deg", 0.5))
        self.r_max     = float(config.get("r_max_m",   15.0))
        self.enable_grid  = bool(config.get("enable_grid", False))
        self.cell_size    = float(config.get("cell_size", 10.0))
        self.grid_padding = float(config.get("grid_padding", 5.0))
        # parallel_kernel: use prange kernel when True.  False (default) is faster
        # for sparse scenes; the serial kernel has lower per-call overhead and is
        # preferred when fewer than ~50 segments are processed.
        self._parallel_kernel = bool(config.get("parallel_kernel", False))
        # When False, skip the psutil syscall in run_frame for clean timing benchmarks.
        self._track_memory = track_memory

        # Eligibility dict: {track_id: next_eligible_time}
        self._eligibility: Dict[int, float] = {}
        self._grid: Optional[dict] = None

    def reset(self):
        """Clear eligibility state (call between independent runs)."""
        self._eligibility = {}

    def init_grid(self, frames_dict: Dict):
        """Build spatial grid from all vehicle positions across frames."""
        if not self.enable_grid:
            self._grid = None
            return
        xs, ys = [], []
        for vlist in frames_dict.values():
            for v in vlist:
                xs.append(float(v["pos"][0]))
                ys.append(float(v["pos"][1]))
        if not xs:
            self._grid = None
            return
        xmin = min(xs) - self.grid_padding
        xmax = max(xs) + self.grid_padding
        ymin = min(ys) - self.grid_padding
        ymax = max(ys) + self.grid_padding
        self._grid = _make_grid(xmin, xmax, ymin, ymax, self.cell_size)

    def run_frame(self, frame_vehicles: List[dict], ego_state: dict,
                  tcurr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Process one frame.

        Parameters
        ----------
        frame_vehicles : list of vehicle dicts
        ego_state      : dict with x, y, vx, vy, psi_rad
        tcurr          : current timestamp in seconds

        Returns
        -------
        angles         : (K,) rad
        ranges         : (K,) m -- NaN for misses
        hit_tracks     : (K,) int -- -1 for misses
        metrics        : dict with timing, counts, and SEI components
        """
        t_pipeline_start = time.perf_counter()

        ego_pos = np.array([ego_state["x"], ego_state["y"]], dtype=np.float64)
        ego_vel = np.array([ego_state["vx"], ego_state["vy"]], dtype=np.float64)
        ego_psi = float(ego_state["psi_rad"])
        fov     = math.radians(self.fov_deg)

        # 1. Eligibility filter
        process_vehicles = []
        skip_vehicles    = []
        for v in frame_vehicles:
            tid = int(v["track_id"])
            if tcurr >= self._eligibility.get(tid, 0.0):
                process_vehicles.append(v)
            else:
                skip_vehicles.append(v)

        # 2. Optional grid-based candidate selection
        if self.enable_grid and self._grid is not None and process_vehicles:
            cand_idxs = _grid_candidates(
                process_vehicles, self._grid, ego_pos, ego_psi, fov, self.r_max)
            selected = [process_vehicles[i] for i in cand_idxs]
        else:
            selected = process_vehicles

        # 3. Build segments
        if selected:
            segments, seg_track_ids = build_segments_from_frame(selected)
        else:
            segments    = np.zeros((0, 2, 2), dtype=np.float64)
            seg_track_ids = np.array([], dtype=np.int64)

        # 4. Scan kernel (time_scan_sec covers this step only)
        angles, ranges, hit_tracks, hit_segs, total_candidates, \
            candidates_per_beam, time_scan_sec = simulate_scan_numba_wrapper(
                ego_state, segments,
                fov_deg=self.fov_deg,
                delta_deg=self.delta_deg,
                r_max=self.r_max,
                seg_track_ids=seg_track_ids,
                parallel=self._parallel_kernel,
            )

        # 5. Update eligibility for processed vehicles
        for v in process_vehicles:
            tid = int(v["track_id"])
            self._eligibility[tid] = _estimate_next_revisit(
                ego_pos, ego_vel, v["pos"], v["vel"], self.r_max, tcurr)

        # Keep skip vehicles' eligibility unchanged (already future-dated)
        # This line is intentionally a no-op comment -- skipped vehicles keep their
        # existing eligibility timestamp from a prior frame's update.

        t_pipeline_end = time.perf_counter()
        time_fullpipeline_sec = t_pipeline_end - t_pipeline_start

        K = int(angles.shape[0])
        metrics = {
            "time_scan_sec":          time_scan_sec,
            "time_fullpipeline_sec":  time_fullpipeline_sec,
            "n_vehicles":             len(frame_vehicles),
            "n_processed":            len(process_vehicles),
            "n_skipped":              len(skip_vehicles),
            "n_segments":             int(segments.shape[0]),
            "K_beams":                K,
            "total_candidates":       total_candidates,
            "candidates_per_beam":    candidates_per_beam,  # (K,) array
            "num_hits":               int(np.sum(~np.isnan(ranges))),
            "muted_ratio":            (len(skip_vehicles) / len(frame_vehicles)
                                       if frame_vehicles else 0.0),
            "memory_rss_mb":          get_memory_mb() if self._track_memory else 0.0,
        }
        return angles, ranges, hit_tracks, metrics

    def run_dataset(self, frames_dict: Dict[int, List[dict]],
                    timestamps: Dict[int, float],
                    ego_traj: Dict[int, dict],
                    n_frames: Optional[int] = None,
                    frame_start: Optional[int] = None) -> SPARTResults:
        """
        Run SPART on a full dataset.

        Parameters
        ----------
        frames_dict : {frame_id: [vehicle_dict, ...]}
        timestamps  : {frame_id: timestamp_s}
        ego_traj    : {frame_id: ego_state_dict}  -- from make_ego_trajectory()
        n_frames    : limit to first n frames (None = all)
        frame_start : start from this frame ID (None = first)

        Returns
        -------
        SPARTResults with per-frame rows.
        """
        self.reset()
        if self.enable_grid:
            self.init_grid(frames_dict)

        frame_list = get_frame_list(frames_dict, n_frames=n_frames,
                                    frame_start=frame_start)
        results = SPARTResults()

        for fid in frame_list:
            fv      = frames_dict[fid]
            tcurr   = timestamps.get(fid, float(fid) * 0.1)
            ego_s   = ego_traj[fid]

            angles, ranges, hit_tracks, metrics = self.run_frame(fv, ego_s, tcurr)

            row = {
                "frame_id":            fid,
                "timestamp_s":         tcurr,
                "ego_x":               ego_s["x"],
                "ego_y":               ego_s["y"],
                "ego_psi":             ego_s["psi_rad"],
                "N_vehicles":          metrics["n_vehicles"],
                "P_processed":         metrics["n_processed"],
                "S_segments":          metrics["n_segments"],
                "K_beams":             metrics["K_beams"],
                "total_candidates":    metrics["total_candidates"],
                "num_hits":            metrics["num_hits"],
                "muted_ratio":         metrics["muted_ratio"],
                "time_scan_sec":       metrics["time_scan_sec"],
                "time_fullpipeline_sec": metrics["time_fullpipeline_sec"],
                "memory_rss_mb":       metrics["memory_rss_mb"],
            }
            results.append(row)

        return results

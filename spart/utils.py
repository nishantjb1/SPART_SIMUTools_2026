"""
spart/utils.py
Shared geometry, I/O, and ego trajectory utilities for SPART.
All functions are pure Python (no Numba) so they can be imported anywhere.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Angle utilities
# ---------------------------------------------------------------------------

def normalize_angle(a: float) -> float:
    """Map angle to (-pi, pi]. Explicit guard for the -pi edge case."""
    result = ((a + math.pi) % (2 * math.pi)) - math.pi
    if result == -math.pi:
        return math.pi
    return result


def angle_diff(a: float, b: float) -> float:
    """Signed angular difference a - b in (-pi, pi]."""
    return normalize_angle(a - b)


# ---------------------------------------------------------------------------
# Vehicle geometry
# ---------------------------------------------------------------------------

def get_vehicle_corners(center: Tuple[float, float], yaw: float,
                        L: float, W: float) -> np.ndarray:
    """
    Return 4 corners in CCW order: front-right, front-left, rear-left, rear-right.
    Local corners: [[hl,-hw],[hl,hw],[-hl,hw],[-hl,-hw]] rotated by R(yaw) then translated.
    Returns shape (4, 2) float64.
    """
    cx, cy = float(center[0]), float(center[1])
    hl = L / 2.0
    hw = W / 2.0
    local = np.array([
        [ hl, -hw],   # front-right
        [ hl,  hw],   # front-left
        [-hl,  hw],   # rear-left
        [-hl, -hw],   # rear-right
    ], dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (R @ local.T).T + np.array([cx, cy], dtype=np.float64)


def get_segments_from_corners(corners: np.ndarray) -> np.ndarray:
    """
    Convert (4,2) corners to (4,2,2) segments.
    Edge i connects corner[i] to corner[(i+1)%4].
    """
    n = corners.shape[0]
    segs = np.empty((n, 2, 2), dtype=np.float64)
    for i in range(n):
        segs[i, 0] = corners[i]
        segs[i, 1] = corners[(i + 1) % n]
    return segs


def get_segments_from_vehicle(v: dict) -> np.ndarray:
    """One vehicle dict → (4,2,2) segments."""
    corners = get_vehicle_corners(v["pos"], v["yaw"], v["L"], v["W"])
    return get_segments_from_corners(corners)


def build_segments_from_frame(vehicles: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    List of vehicle dicts → (M,2,2) segments and (M,) int track_ids.
    M = 4 * len(vehicles).
    """
    if not vehicles:
        return np.zeros((0, 2, 2), dtype=np.float64), np.array([], dtype=np.int64)
    segs_list = []
    tids = []
    for v in vehicles:
        corners = get_vehicle_corners(v["pos"], v["yaw"], v["L"], v["W"])
        rect_segs = get_segments_from_corners(corners)   # (4,2,2)
        segs_list.append(rect_segs)
        tids.extend([int(v["track_id"])] * 4)
    segments = np.concatenate(segs_list, axis=0)         # (M,2,2)
    return segments, np.array(tids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Ego trajectory generators
# ---------------------------------------------------------------------------

def generate_ego_circular(tcurr: float, radius: float, speed: float,
                           center: Tuple[float, float]) -> dict:
    """
    Deterministic circular ego state for a given timestamp.
    Returns dict with keys: x, y, vx, vy, psi_rad.
    """
    cx, cy = float(center[0]), float(center[1])
    omega = speed / radius          # angular velocity (rad/s)
    theta = omega * tcurr           # current angle
    x = cx + radius * math.cos(theta)
    y = cy + radius * math.sin(theta)
    vx = -speed * math.sin(theta)
    vy =  speed * math.cos(theta)
    psi_rad = math.atan2(vy, vx)
    return {"x": x, "y": y, "vx": vx, "vy": vy, "psi_rad": psi_rad}


def generate_ego_static(x: float = 995.0, y: float = 1000.0,
                         vx: float = 0.0, vy: float = 0.0,
                         psi: float = 0.0) -> dict:
    """Static ego state (for debugging only)."""
    return {"x": float(x), "y": float(y),
            "vx": float(vx), "vy": float(vy), "psi_rad": float(psi)}


def make_ego_trajectory(frame_ids: List[int],
                        timestamps: Dict[int, float],
                        mode: str = "circular",
                        **kwargs) -> Dict[int, dict]:
    """
    Build a dict {frame_id: ego_state} shared by ALL methods.
    mode="circular"  → uses generate_ego_circular with kwargs:
                        radius, speed, center
    mode="static"    → uses generate_ego_static with kwargs:
                        x, y, vx, vy, psi
    This is the single source of truth for ego position in all experiments.
    """
    ego_traj: Dict[int, dict] = {}
    if mode == "circular":
        radius = float(kwargs.get("radius", 15.0))
        speed  = float(kwargs.get("speed",  5.398298934))
        center = kwargs.get("center", (996.0, 999.0))
        for fid in frame_ids:
            t = timestamps.get(fid, float(fid) * 0.1)
            ego_traj[fid] = generate_ego_circular(t, radius, speed, center)
    elif mode == "static":
        state = generate_ego_static(
            x=kwargs.get("x", 995.0), y=kwargs.get("y", 1000.0),
            vx=kwargs.get("vx", 0.0), vy=kwargs.get("vy", 0.0),
            psi=kwargs.get("psi", 0.0)
        )
        for fid in frame_ids:
            ego_traj[fid] = dict(state)
    else:
        raise ValueError(f"Unknown ego mode: {mode!r}. Use 'circular' or 'static'.")
    return ego_traj


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_interaction_csv(path: str,
                         max_frames: Optional[int] = None,
                         frame_start: Optional[int] = None
                         ) -> Tuple[Dict[int, List[dict]], Dict[int, float]]:
    """
    Load one INTERACTION CSV.
    Returns:
        frames_dict  : {frame_id: [vehicle_dict, ...]}
        timestamps   : {frame_id: timestamp_s}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df = df.sort_values(["frame_id", "track_id"]).reset_index(drop=True)

    all_frame_ids = sorted(df["frame_id"].unique())
    if frame_start is not None:
        all_frame_ids = [f for f in all_frame_ids if f >= frame_start]
    if max_frames is not None:
        all_frame_ids = all_frame_ids[:max_frames]

    frames_dict: Dict[int, List[dict]] = {}
    timestamps:  Dict[int, float] = {}

    subset = df[df["frame_id"].isin(set(all_frame_ids))]
    for frame_id, group in subset.groupby("frame_id"):
        fid = int(frame_id)
        vehicles = []
        for _, row in group.iterrows():
            vx_ = float(row["vx"]); vy_ = float(row["vy"])
            spd = float(row["speed"]) if "speed" in row.index else math.hypot(vx_, vy_)
            vehicles.append({
                "track_id": int(row["track_id"]),
                "pos":  np.array([row["x"], row["y"]], dtype=np.float64),
                "vel":  np.array([vx_, vy_], dtype=np.float64),
                "yaw":  float(row["psi_rad"]),
                "L":    float(row["length"]),
                "W":    float(row["width"]),
                "speed": spd,
            })
        frames_dict[fid] = vehicles
        timestamps[fid] = float(group["timestamp_ms"].iloc[0]) / 1000.0

    return frames_dict, timestamps


def load_multiple_csvs(paths: List[str],
                       max_frames_per_csv: Optional[int] = None
                       ) -> Dict[str, Tuple[Dict, Dict]]:
    """
    Load multiple CSVs. Returns {scenario_name: (frames_dict, timestamps_dict)}.
    scenario_name is the basename without extension.
    """
    scenarios: Dict[str, Tuple[Dict, Dict]] = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        frames, ts = load_interaction_csv(path, max_frames=max_frames_per_csv)
        scenarios[name] = (frames, ts)
        print(f"  Loaded {name}: {len(frames)} frames")
    return scenarios


def get_frame_list(frames_dict: Dict[int, List[dict]],
                   n_frames: Optional[int] = None,
                   frame_start: Optional[int] = None) -> List[int]:
    """
    Return sorted list of frame IDs from frames_dict.
    Optionally starts at frame_start and limits to n_frames.
    """
    fids = sorted(frames_dict.keys())
    if frame_start is not None:
        fids = [f for f in fids if f >= frame_start]
    if n_frames is not None:
        fids = fids[:n_frames]
    return fids


def dataset_stats(frames_dict: Dict[int, List[dict]],
                  timestamps: Dict[int, float]) -> dict:
    """Compute summary statistics for a loaded dataset."""
    n_frames = len(frames_dict)
    vehicles_per_frame = [len(v) for v in frames_dict.values()]
    all_x = [v["pos"][0] for vlist in frames_dict.values() for v in vlist]
    all_y = [v["pos"][1] for vlist in frames_dict.values() for v in vlist]
    all_ts = list(timestamps.values())
    return {
        "n_frames": n_frames,
        "frame_ids_range": (min(frames_dict.keys()), max(frames_dict.keys())),
        "mean_vehicles_per_frame": float(np.mean(vehicles_per_frame)),
        "max_vehicles_per_frame":  int(np.max(vehicles_per_frame)),
        "min_vehicles_per_frame":  int(np.min(vehicles_per_frame)),
        "x_range": (float(np.min(all_x)), float(np.max(all_x))),
        "y_range": (float(np.min(all_y)), float(np.max(all_y))),
        "timestamp_range_s": (float(np.min(all_ts)), float(np.max(all_ts))),
        "total_vehicles": sum(vehicles_per_frame),
    }


# ---------------------------------------------------------------------------
# Memory utility
# ---------------------------------------------------------------------------

def get_memory_mb() -> float:
    """Return process RSS memory in MB. Uses psutil if available."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        return float("nan")

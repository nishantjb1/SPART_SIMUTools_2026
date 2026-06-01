"""
SPART — Lightweight 2D LiDAR Emulator for Trajectory-Dataset-Driven AV Simulation Benchmarking
"""

from spart.utils import (
    normalize_angle,
    angle_diff,
    get_vehicle_corners,
    get_segments_from_corners,
    get_segments_from_vehicle,
    build_segments_from_frame,
    generate_ego_circular,
    generate_ego_static,
    make_ego_trajectory,
    load_interaction_csv,
    load_multiple_csvs,
    get_frame_list,
    dataset_stats,
    get_memory_mb,
)

__version__ = "0.1.0"
__all__ = [
    "normalize_angle", "angle_diff",
    "get_vehicle_corners", "get_segments_from_corners",
    "get_segments_from_vehicle", "build_segments_from_frame",
    "generate_ego_circular", "generate_ego_static", "make_ego_trajectory",
    "load_interaction_csv", "load_multiple_csvs", "get_frame_list",
    "dataset_stats", "get_memory_mb",
]

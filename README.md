# SPART — Lightweight 2-D LiDAR Emulator for AV Simulation Benchmarking

SPART (**S**can **P**runing with **A**ngular intervals and **R**ange-gated **T**emporal eligibility) is a fast, lightweight 2-D LiDAR scan emulator designed for trajectory-dataset-driven autonomous vehicle (AV) simulation.
It replaces the brute-force O(K × M) ray–segment intersection search with two complementary pruning strategies that together reduce intersection tests by **~60×** on realistic intersection scenes (INTERACTION dataset, mean ~7 vehicles/frame).

---

## Key Results (EAI SIMUTools 2026)

| Metric | Value |
|--------|-------|
| Scan Efficiency Index (SEI) | **60.3×** fewer intersection tests than brute-force JIT |
| Scan-kernel throughput | **42,748 Hz** (SPART) vs 18,866 Hz (JIT vanilla) — 2.27× speedup |
| Full-pipeline throughput | **6,616 Hz** — 662× real-time at 10 Hz LiDAR |
| Full-pipeline speedup over JIT vanilla | **1.54×** aggregate (0.88–2.39× per session; scales with vehicle density) |
| Geometric fidelity | **FP = 0, AP = 0, range error = 0.000 m** (all 35,000 frames) |
| False Negatives | ~7.4% (100% attributable to temporal muting by design) |

SEI is deterministic and machine-independent. Hz numbers are from a clean machine (fresh restart, no background processes); expect lower on a loaded workstation.

---

## Algorithm Overview

Each frame SPART applies two pruning stages:

1. **Temporal eligibility (muting)**: vehicles outside the ego's detection range are deferred using a linear closing-velocity estimate. Muted vehicles are skipped entirely — no segment building, no kernel call. Achieves ~52% muting on the INTERACTION intersection (~7 vehicles/frame, ego radius = 15 m).

2. **Angular interval pruning**: for each segment kept by the eligibility filter, a tight [θ_min, θ_max] angular interval is precomputed. The scan kernel skips any (beam, segment) pair where the beam angle falls outside the interval. On 241 beams and ~13 segments, this reduces intersection tests from ~3,000 to ~108 per frame.

An optional **spatial grid** (disabled by default) provides additional speedup above ~50 vehicles/frame by pre-filtering processed vehicles outside the 120° FOV before segment building.

---

## Installation

```bash
# Clone the repository
git clone <anonymous-repository-url>
cd SPART_SIMUTools_2026

# Install in editable mode (recommended for reproducibility)
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

**Python ≥ 3.9** required. Tested with Python 3.11, Numba 0.61, NumPy 2.0.

> **First run note:** Numba JIT compilation takes 4–25 seconds on the first call per session (4.7 s on a clean machine; up to 25 s under load). All experiment scripts report compile time separately.

---

## Quick Start — No Dataset Required

```bash
python -m spart demo
```

Generates in `results/synthetic_demo/`:
- `demo_animation.gif` — 50-frame animated top-down scan (vehicles, ego, beams, hits)
- `demo_pointcloud.csv` — 24,100 rows (100 frames × 241 beams)
- `demo_fidelity.json` — geometric correctness check (FP=0, AP=0, err=0.000m)

Completes in under 60 seconds on any modern machine (13 s on a clean machine including JIT warm-up).

---

## Running with the INTERACTION Dataset

Download the INTERACTION dataset (intersection scenario `DR_CHN_Merging_ZS`) from [interaction-dataset.com](https://interaction-dataset.com) and place the CSV files in the project root:

```
vehicle_tracks_000.csv
vehicle_tracks_001.csv
vehicle_tracks_002.csv
vehicle_tracks_003.csv
```

Pre-flight sanity check:

```bash
python -m spart check --csv vehicle_tracks_000.csv
```

Expected output: `6/6 checks PASSED — safe to run experiments`.

---

## Reproducing Paper Experiments

Run scripts in order from the project root. Each script is self-contained and reads parameters from `configs/benchmark_config.yaml`.

```bash
# Experiment 1: Fidelity validation (FP=0, AP=0, range_error=0)
python scripts/exp1_fidelity.py

# Experiment 2: Ablation study (5 conditions, SEI gate)
python scripts/exp2_ablation.py

# Experiment 3: Main benchmark — SPART vs JIT vanilla (4 CSVs)
python scripts/exp3_main_benchmark.py

# Experiment 4: Grid crossover study (synthetic scenes, N = 5–200)
python scripts/exp4_grid_crossover.py

# Experiment 5: Synthetic demo (no dataset required, timing gate < 60 s)
python scripts/exp5_synthetic_demo.py
```

All outputs (figures, JSON summaries, CSVs) are written to `results/<experiment>/`.

**Timing note:** Run on a clean machine with no background processes for stable Hz measurements. SEI values (candidate counts) are deterministic and will match regardless of machine load.

---

## Project Structure

```
spart/                        # Python package
├── __init__.py               # Public API
├── __main__.py               # CLI  (python -m spart)
├── core.py                   # SPART pipeline orchestrator
├── scan_kernel.py            # Numba JIT kernels (parallel + serial)
├── vanilla_kernel.py         # Baseline kernels (conditions A, B, D)
└── utils.py                  # Geometry, I/O, ego trajectory utilities

scripts/                      # Standalone experiment scripts
├── sanity_check.py           # Pre-flight validation
├── exp1_fidelity.py          # Geometric correctness + muting analysis
├── exp2_ablation.py          # Component-by-component ablation
├── exp3_main_benchmark.py    # SPART vs JIT vanilla, 4 CSVs
├── exp4_grid_crossover.py    # Grid module crossover study
└── exp5_synthetic_demo.py    # Reproducible demo (no dataset)

configs/
├── benchmark_config.yaml     # Single source of truth for all experiment params
└── default_config.yaml       # Defaults for python -m spart

results/
├── synthetic_demo/           # Committed: reproducible no-dataset demo outputs
├── fidelity/                 # Git-ignored (large; regenerate with exp1_fidelity.py)
├── ablation/                 # Git-ignored (regenerate with exp2_ablation.py)
├── benchmark/                # Git-ignored (regenerate with exp3_main_benchmark.py)
└── grid_crossover/           # Git-ignored (regenerate with exp4_grid_crossover.py)
```

---

## Configuration

All parameters are controlled from `configs/benchmark_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fov_deg` | 120.0 | LiDAR field of view (degrees) |
| `delta_deg` | 0.5 | Angular resolution → K = 241 beams |
| `r_max_m` | 15.0 | Maximum detection range (m) |
| `n_frames` | 8750 | Frames per CSV used in benchmarks |
| `enable_grid` | false | Spatial hash grid (beneficial at N ≥ 50) |
| `ego_circle_radius_m` | 15.0 | Ego circular trajectory radius |
| `ego_circle_speed_mps` | 5.398 | Ego speed on circular path |

---

## Python API

```python
from spart.core import SPART
from spart.utils import load_interaction_csv, make_ego_trajectory, get_frame_list

# Load data
frames_dict, timestamps = load_interaction_csv("vehicle_tracks_000.csv")
frame_list = get_frame_list(frames_dict, n_frames=8750)
ego_traj   = make_ego_trajectory(frame_list, timestamps, mode="circular",
                                  radius=15.0, speed=5.398, center=(996.0, 999.0))

# Initialise SPART
spart = SPART({"fov_deg": 120.0, "delta_deg": 0.5, "r_max_m": 15.0},
              track_memory=False)

# Run one frame
angles, ranges, hit_tracks, metrics = spart.run_frame(
    frames_dict[frame_list[0]], ego_traj[frame_list[0]], timestamps[frame_list[0]])

print(f"Beams: {len(angles)}, Hits: {metrics['num_hits']}, "
      f"Candidates: {metrics['total_candidates']}, "
      f"Muted: {metrics['muted_ratio']:.1%}")
```

---

## Correctness Fixes Applied in This Release

The following bugs were identified and corrected before this release:

| # | Bug | Impact |
|---|-----|--------|
| 1 | Data race in `total_candidates` inside `prange` | Non-deterministic candidate counts |
| 2 | Ego trajectory mismatch between methods | Unfair timing comparison |
| 3 | Frame count 3-way mismatch | Irreproducible results |
| 4 | Grid module disabled but undisclosed | Incorrect performance characterisation |
| 5 | Hardware specification documented incorrectly | Misleading benchmark context |
| 6 | Theta normalisation missing in angular interval check | 2,155 pruning misses over 2,000 sampled frames |

---

## Citation

```bibtex
@inproceedings{spart2026,
  title     = {SPART: A Lightweight 2-D LiDAR Emulation Tool for
               Trajectory-Dataset-Driven Autonomous Vehicle Testing},
  booktitle = {Proceedings of EAI SIMUTools 2026},
  year      = {2026},
}
```
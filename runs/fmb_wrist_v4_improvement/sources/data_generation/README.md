# MolmoMotion-1M Data Generation Pipeline

Automatic annotation pipeline that extracts **object-grounded 3D point trajectories**
from unconstrained RGB video, as described in *MolmoMotion: Forecasting Point
Trajectories in 3D with Language Instruction*. Given a video and a language
description of an action, the pipeline grounds the moving object, tracks dense
points on it, lifts those tracks into a metric 3D world frame, filters/smooths
them, and cuts the video into short motion-coherent clips.

Applying this pipeline to ~1.16M public video clips produces **MolmoMotion-1M**,
the dataset used to pretrain MolmoMotion.

```
 video + action ──▶ 1. grounding ──▶ 2. depth+pose ──▶ 3. 2D track ──▶ 4. 3D lift ──▶ 5. filter+smooth ──▶ 6. clip
   "pour water       MolmoPoint        ViPE              AllTracker      back-proj      trust-gated         motion
    from the          + SAM3            (metric)          (dense)         to world       ray smoothing       segments
    flask"            + K-means
```

All model components are **frozen, publicly released checkpoints** — no weights are
trained here.

---

## Pipeline stages

| # | Stage | Model / method | Output |
|---|-------|----------------|--------|
| 1 | **Semantic grounding** | Qwen3-0.6B (object phrase) → optional Molmo2-8B re-caption → MolmoPoint-Vid-4B (2D point) → SAM 3 (mask) → K-means (N=100 query points) | `grounding/<vid>/query_points/*.npz` |
| 2 | **Metric depth + pose** | ViPE monocular SLAM | `vipe_results/{depth,pose,intrinsics,rgb}/<vid>.*` |
| 3 | **2D point tracking** | AllTracker (dense, sliding window) | `tracks_2d/<vid>/<vid>_merged.npz` |
| 4 | **3D lift** | back-project visible 2D tracks with ViPE depth/intrinsics/pose into a world frame anchored at the query-time camera | `tracks_3d/<corpus>/<vid>_merged_3d_tracks.npz` |
| 5 | **Filter + smooth** | anchor tracks (K=16) → trust weights → mean-shift auto-split → z-score drop → ray-only consensus smoothing | `final_tracks/<vid>_{3d,2d}.npz` + `<vid>_filter_meta.npz` |
| 6 | **Video-level clipping** | per-frame trimmed-mean 3D displacement, threshold τ, merge gaps, drop short runs | `clips/<corpus>_clips.json` |

### What each stage does

**1. Semantic grounding.** The action description is reduced to a short object
phrase with Qwen3-0.6B. If the phrase is vague (`it`, `the object`, …) or absent,
Molmo2-8B re-captions the video visually first. MolmoPoint-Vid-4B then localizes
the object as a 2D point in the anchor frame — prompted *by the action*
(`"point to {obj} gripped and picked up by the {agent}"`), which disambiguates the
moving object from static distractors. SAM 3 turns that point into a mask, and
**N = 100** query points are sampled with K-means over the mask pixels.

**2. Metric depth + pose.** ViPE runs monocular visual SLAM in a single pass to
produce per-frame metric depth, intrinsics `(fx, fy, cx, cy)`, and a `(T,4,4)`
camera-to-world trajectory anchored at the first frame.

**3. 2D tracking.** AllTracker propagates the query points through the video,
yielding temporally persistent 2D tracks and per-frame visibility.

**4. 3D lift.** Each visible 2D track point is back-projected with the estimated
depth + intrinsics and transformed by the camera pose into a metric world frame
anchored at the query-time camera, giving `(N, T, 3)` metric tracks.

**5. Filtering + smoothing.** Lifted tracks are corrupted by tracking drift and
depth noise. Using the object-level rigidity prior, the smoother selects K=16
anchor tracks, computes per-frame **trust weights** `w = exp(-e/s)^p`, splits
physically separate instances with mean-shift, drops outlier tracks by a MAD
z-score, and smooths each surviving track **along its camera ray only**, penalizing
multi-stride acceleration (Δ ∈ {1,3,5}) following Stereo4D.

**6. Video-level clipping.** The object motion score
`s_t = trimmed-mean_n ||p_n(t) - p_n(t-1)||` is thresholded at τ to extract
contiguous high-motion segments, which become the training clips. Output schema
(ranges inclusive on both ends):

```json
{ "file": "<vid>", "fps": 15, "num_frames": 468,
  "clips_by_object": { "obj0": [[32, 48]], "obj1": [[100, 150], [200, 220]] },
  "num_clips_total": 3 }
```

---

## Repository layout

```
data_generation/
├── run_pipeline.py            # end-to-end driver (stages 1–6)
├── pipeline/
│   ├── grounding_worker.py    # stage 1 (loads all grounding models once)
│   └── clip_segments.py       # stage 6 (video-level motion clipping)
├── configs/
│   ├── default.yaml           # all hyperparameters (paper defaults)
│   ├── human_manipulation.yaml  # agent="hand"  (egocentric / third-person manipulation)
│   ├── robot.yaml             #   agent="robot gripper"  (real-robot manipulation)
│   └── in_the_wild.yaml       #   tracking-mode  (in-the-wild internet video)
├── third_party/               # vendored frozen models (see third_party/README.md)
│   ├── sam3/                  #   SAM 3 + MolmoPoint/Qwen3/Molmo2 grounding glue
│   ├── alltracker/            #   AllTracker 2D tracker
│   └── vipe/                  #   ViPE (depth+pose) + 3D-lift + filter/smooth scripts
├── scripts/
│   ├── install.sh             # deps + editable installs + weight prefetch
│   ├── download_models.sh     # pre-fetch checkpoints (else auto-download on use)
│   ├── check_env.py           # environment sanity check
│   └── run_example.sh
└── examples/tasks_example.json
```

---

## Install

Requires Linux, an NVIDIA GPU (A100/H100-class recommended), CUDA 12.x with `nvcc`
on `PATH` (ViPE compiles a CUDA extension), and Python 3.12.

```bash
conda create -n molmomotion python=3.12 -y && conda activate molmomotion
cd data_generation
bash scripts/install.sh          # pip deps + `pip install -e` sam3 & vipe + prefetch weights
python scripts/check_env.py      # verify
```

`install.sh` runs `pip install -r requirements.txt`, then `pip install -e third_party/sam3`
and `pip install -e third_party/vipe`, then `scripts/download_models.sh`. See
`third_party/README.md` for per-model build notes and licenses.

**Model checkpoints** (auto-download on first use, or prefetch with
`scripts/download_models.sh`):
`allenai/MolmoPoint-Vid-4B`, `allenai/Molmo2-8B`, `Qwen/Qwen3-0.6B` (HuggingFace),
and `aharley/alltracker` (torch.hub, cached in `TORCH_HOME`).

> **`facebook/sam3` is a gated HuggingFace model.** Request access at
> <https://huggingface.co/facebook/sam3>, then authenticate before the first run
> (`huggingface-cli login`, or set `HF_TOKEN`). All HuggingFace weights cache in `HF_HOME`.

> `transformers==4.57.1` is required — MolmoPoint/Molmo2 produce garbage output on 5.x.

---

## Usage

Write a tasks JSON — a list of `{video_id, video_path, action}`:

```json
[
  {"video_id": "pour_water_0001",
   "video_path": "/abs/path/pour_water_0001.mp4",
   "action": "pour water from the tan flask into the red can"}
]
```

Run the whole pipeline:

```bash
python run_pipeline.py \
    --tasks   my_tasks.json \
    --config  configs/human_manipulation.yaml \
    --work_dir ./runs/my_run
```

Run a subset of stages (every stage caches its output and is resumable):

```bash
python run_pipeline.py --tasks my_tasks.json --config configs/robot.yaml \
    --work_dir ./runs/my_run --start_stage 4 --end_stage 6
```

Final per-clip ranges land in `./runs/my_run/clips/<corpus>_clips.json`; the
filtered 3D/2D trajectories are in `./runs/my_run/final_tracks/`.

### Choosing a config

| Source video | Config | Grounding prompt |
|---|---|---|
| Ego/third-person human manipulation | `human_manipulation.yaml` | `point to {obj} gripped and picked up by the hand` |
| Real-robot manipulation | `robot.yaml` | `point to {obj} gripped and picked up by the robot gripper` |
| In-the-wild internet video | `in_the_wild.yaml` | `track the {obj}` |

All hyperparameters (K-means N, fps, smoothing α/λ/z-threshold/anchors, clip
threshold τ, …) live in `configs/default.yaml`; presets override only what differs.

### Scaling

The driver processes a list of videos in one process (Stage 1 loads all grounding
models once). For corpus-scale runs, shard your tasks JSON across jobs (each writes
to its own `work_dir`, or a shared one — outputs are keyed by `video_id`), and merge
the per-shard `clips/*.json` at the end. Stage 2 (ViPE) dominates wall-clock
(~80%); stages 4–6 are CPU-only.

---

## Notes & scope

- This directory is the **data-generation pipeline** only. Model training/eval
  live at the [repo root](../README.md); the PointMotionBench benchmark
  construction lives in [`pointmotionbench/`](../pointmotionbench/).
- ViPE assumes a moving monocular camera. For fixed-rig real-robot video, swap in a
  metric video-depth model + identity pose (not included here).
- Third-party models are vendored unmodified except for the thin integration scripts
  we add (`sam3/molmo2_pointing.py`, `sam3/querypoints_from_video.py`,
  `alltracker/run-query-points.py`, `vipe/scripts/vipe_to_colmap_general.py`,
  `vipe/track-filter-smooth.py`). Upstream licenses are retained in each
  `third_party/<model>/` directory.
- **License caveats.** The vendored components are NOT all Apache-2.0:
  `third_party/sam3/` ships under Meta's SAM License (redistribution permitted
  with the license attached; acknowledgment and use restrictions apply — see
  `third_party/sam3/LICENSE`), and ViPE's dependency stack includes UniDepth
  under **CC BY-NC 4.0 (non-commercial)** — see
  `third_party/vipe/THIRD_PARTY_LICENSES.md`. If you use this pipeline
  commercially, review those terms first.

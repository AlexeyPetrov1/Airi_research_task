---
pretty_name: PointMotionBench
task_categories:
- robotics
---

# PointMotionBench

This is the official benchmark for the paper [MolmoMotion: Forecasting Point Trajectories in 3D with Language Instruction](https://huggingface.co/papers/2606.18558).

A benchmark for evaluating 3D point motion in video, covering egocentric and third-person scenes across three source datasets. Each sample pairs an RGB video clip with per-object 3D and 2D tracked surface points and a human-verified natural-language caption.

## Overview

| Dataset | Clips | Video format | Tracks | Scene type |
|---------|-------|--------------|--------|------------|
| DAVIS | 90 | mp4, 24 fps | 2D + 3D | Third-person, diverse outdoor/indoor |
| HOT3D | 2,475 | mp4, 30 fps | 2D + 3D | Egocentric, object manipulation (Aria) |
| WorldTrack | 155 | npz (frames embedded), 30 fps | 3D (+2D) | Egocentric + studio, 4 splits |

---

## Setup

### Step 1 — Download PointMotionBench

Benchmark data provided in this repository include annotations, captions, indices, and scripts created by Ai2 that correspond to the source datasets.

```bash
# pip install huggingface_hub
# If the download stalls near completion: HF_HUB_DISABLE_XET=1 huggingface-cli download ...
huggingface-cli download allenai/PointMotionBench \
    --repo-type dataset --local-dir $POINTMOTIONBENCH_ROOT
```

---

### Step 2 — DAVIS: Download Videos

DAVIS videos should be reconstructed from the source data at the [official source](https://davischallenge.org/davis2017/code.html). Download **Trainval 2017 - Images (480p)** and **Annotations**, then convert frames to mp4 (requires `ffmpeg`):

```bash
python davis/reconstruct_davis.py \
    --davis-root /path/to/DAVIS \
    --output-dir davis/videos/input_480p
```

---

### Step 3 — HOT3D: Download Videos

HOT3D videos should be reconstructed from the source data at [bop-benchmark/hot3d](https://huggingface.co/datasets/bop-benchmark/hot3d) (HuggingFace).

**Requirements:** `huggingface_hub`, `imageio[ffmpeg]`, `imageio-ffmpeg`, `opencv-python-headless`, `numpy`

```bash
python hot3d/reconstruct_hot3d.py \
    --workdir    /path/to/hot3d_work \
    --output-dir hot3d/rgbs
```

This runs all three stages (download TARs → extract RGB → trim to PointMotionBench windows).
For large-scale extraction, run the three scripts individually — `extract_rgbs.py` supports sharding:

```bash
python hot3d/extract_rgbs.py \
    --clips_dir  /path/to/hot3d_work/train_aria \
    --output_dir /path/to/hot3d_work/rgbs \
    --shard_idx  0 \
    --num_shards 8
```

---

### Step 4 — WorldTrack: Reconstruct Clips

Download the WorldTrack source data (WorldTrack benchmark, introduced in St4RTrack, Feng et al., ICCV 2025 — dataset download available at [HavenFeng/St4RTrack](https://github.com/HavenFeng/St4RTrack)). The source data should have this layout:

```
WorldTrack/
├── adt_mini/        # Aria Digital Twin
├── ds_mini/         # Dynamic Scenes
├── po_mini/         # POtential Objects
└── pstudio_mini/    # PStudio
```

Then extract PointMotionBench clips using the index map from Step 1:

```bash
python worldtrack/reconstruct_worldtrack.py \
    --index_map  worldtrack/worldtrack_index_map.json \
    --src_dir    /path/to/WorldTrack \
    --output_dir worldtrack
```

| Split | Clips | Frames per clip | Scene type |
|-------|-------|-----------------|------------|
| `adt_mini` | 39 | 12–300 | Apartment indoor, egocentric (Aria Digital Twin) |
| `ds_mini` | 52 | 39–128 | Dynamic indoor scenes |
| `po_mini` | 16 | 78–128 | Mixed indoor (cab, seminar, egobody) |
| `pstudio_mini` | 48 | 150 | Studio sports (basketball, football, tennis, etc.) |

## Intended Use

PointMotionBench is provided for benchmarking purposes. 
It intended for research and educational use in accordance with Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use).

## Disclaimer 
PointMotionBench data maps to the videos and other source data that are not shared in this repository. 
Such videos and data are provided by the owners of the source datasets above, and remain subject to their respective 
license terms and use restrictions. Users who access videos and data from these sources are responsible for 
reviewing and confirming that their use complies with the terms and conditions.

## Citation

```bibtex
@misc{zhang2026molmomotionforecastingpointtrajectories,
    title={MolmoMotion: Forecasting Point Trajectories in 3D with Language Instruction}, 
    author={Jianing Zhang and Chenhao Zheng and Yajun Yang and Max Argus and Rustin Soraki and Winson Han and Taira Anderson and Chun-Liang Li and Shuo Liu and Jiafei Duan and Zhongzheng Ren and Jieyu Zhang and Ranjay Krishna},
    year={2026},
    eprint={2606.18558},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2606.18558}, 
}
```
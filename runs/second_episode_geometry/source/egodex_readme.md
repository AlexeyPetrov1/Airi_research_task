# egodex (MolmoMotion-1M)

3D + 2D point-trajectory tracks (object + hand), captions, and per-frame camera on Apple's ego-centric EgoDex corpus. Videos are not redistributed — reconstruct them from upstream EgoDex.

## Unpack

The per-clip directories ship as ~10 GB tar shards under `tracks/` / `camera/`. Extract from the dataset dir:

```bash
cd egodex
for t in */*.tar; do tar -xf "$t" && rm "$t"; done   # extract, then delete each shard
```
→ `tracks/object/` · `tracks/hand/` · `camera/pose/` · `camera/intrinsics/`

## Reconstruct

### Step 1/2 Download upstream EgoDex
Clips ship as per-split zips (`part1`–`part5`, `extra`, `test`); unzip in place to get `{part}/{task}/{index}.mp4`.
```bash
for split in part1 part2 part3 part4 part5 extra test; do
    curl -L -O https://ml-site.cdn-apple.com/datasets/egodex/${split}.zip
    unzip -q ${split}.zip
done
```

### Step 2/2 Re-encode our subset
Use the bundled `imageio-ffmpeg` for the exact shipped encode.
```bash
pip install imageio imageio-ffmpeg==0.6.0 opencv-python
python reconstruct_videos.py \
    --upstream     /path/to/EgoDex_raw \
    --release-root /path/to/molmo-motion-1m/egodex \
    --jobs 8
```
Example output: `videos/{video_id}.mp4`, 854×480 15 FPS (h264, re-timed from upstream 1920×1080 @ 30 FPS), frame-aligned with the tracks (`num_frames`).

## Desired file structure
```
egodex/
├── annotations/                                              [shipped, loose]
│   ├── egodex_clips.json          100,844 object clips
│   ├── egodex_split.json           95,798 train / 5,046 test   (object)
│   ├── egodex_hand_split.json     196,576 train / 5,044 test   (hand)
│   ├── egodex_paired_index.json   100,774 paired ids
│   └── egodex_videos_index.json   201,690 entries
├── tracks/   tracks-*.tar shards → object/  201,688 npz  (100,844 × {2d,3d})   [shipped]
│                                    hand/    403,240 npz  (201,620 × {2d,3d})
├── camera/   camera-*.tar shards → pose/        201,690 npz                    [shipped]
│                                    intrinsics/  201,690 npz
└── videos/             201,690 mp4                            [reconstruct]
```

The annotation JSONs are the source of truth — `egodex_clips.json` is canonical for captions, `num_frames` (=T), and inclusive `clips_by_object` motion ranges; don't enumerate `tracks/`. Load:

```python
import numpy as np

vid = "extra_blowdry_hair_97"  # {part}_{task}_{index}; key into tracks/, camera/, videos/, index

# Object tracks: flat single-object
o2d = np.load(f"tracks/object/{vid}_2d.npz")   # tracks (T,K,2) px (x,y), visibility (T,K), dim (2,) [H,W]
o3d = np.load(f"tracks/object/{vid}_3d.npz")    # points_3d (K,T,3) metric world, visibility (K,T,1)
pts = o3d["points_3d"].transpose(1, 0, 2)       # axis swap: 3D (K,T,...) -> unified (T,K,3)

# Hand tracks: pickled dict, keyed left_hand and/or right_hand (same shapes as object)
h2d = np.load(f"tracks/hand/{vid}_2d.npz", allow_pickle=True).item()
left = h2d.get("left_hand")                     # some clips have only one hand -> use .get()

# Camera
pose = np.load(f"camera/pose/{vid}.npz")         # data (T,4,4) f32, inds (T,) i64
intr = np.load(f"camera/intrinsics/{vid}.npz")   # data (T,4) f32 [fx,fy,cx,cy], inds (T,) i64
```

## Source & License
Upstream videos and pose: [apple/ml-egodex](https://github.com/apple/ml-egodex) — check Apple's license before downloading. The annotations Ai2 produced (3D + 2D object/hand tracks, captions, splits, and per-frame camera) are licensed under **CC BY-NC 4.0**, intended for research and educational use in accordance with Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use). The annotations and captions are provided with pixel coordinates that correspond to the videos from EgoDex; use of reconstructed videos remains subject to the upstream license.

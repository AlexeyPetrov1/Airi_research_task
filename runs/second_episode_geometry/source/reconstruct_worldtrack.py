"""
reconstruct_worldtrack.py

Reconstructs the PointMotionBench worldtrack clips from the original
WorldTrack source NPZs, using the index map (worldtrack_index_map.json).

Inputs:
  - worldtrack_index_map.json  (available on HuggingFace at allenai/PointMotionBench)
  - Original WorldTrack source NPZs

Output:
  - One NPZ per clip, matching the PointMotionBench format exactly,
    written to:  <output_dir>/<dataset>/<clip_name>/<clip_name>.npz

Usage:
    python worldtrack/reconstruct_worldtrack.py \
        --index_map  worldtrack/worldtrack_index_map.json \
        --src_dir    /path/to/WorldTrack \
        --output_dir worldtrack

Dependencies: numpy
"""

import argparse
import json
import numpy as np
from pathlib import Path


def camspace_to_world(tracks_XYZ, visibility, cameras_w2c):
    """Lift camera-space tracks to world space using extrinsics_w2c."""
    T, N, _ = tracks_XYZ.shape
    tracks_world = np.full(tracks_XYZ.shape, np.nan, dtype=np.float64)
    for t in range(T):
        vis_t = visibility[t]
        if not vis_t.any():
            continue
        R    = cameras_w2c[t, :3, :3]
        tvec = cameras_w2c[t, :3,  3]
        tracks_world[t, vis_t] = (R.T @ (tracks_XYZ[t, vis_t].T - tvec[:, None])).T
    return tracks_world


def extract_clip(src_npz, entry):
    """Build one PMB-format NPZ dict from source arrays + index map entry."""
    fi = np.array(entry['frame_indices'], dtype=np.int32)  # frame indices
    pi = np.array(entry['point_indices'], dtype=np.int32)  # point indices

    tracks_XYZ = src_npz['tracks_XYZ'][fi][:, pi].astype(np.float32)
    visibility  = src_npz['visibility'][fi][:, pi]
    images      = src_npz['images_jpeg_bytes'][fi]

    out = {
        'tracks_XYZ':        tracks_XYZ,
        'visibility':        visibility,
        'images_jpeg_bytes': images,
        'fx_fy_cx_cy':       src_npz['fx_fy_cx_cy'].astype(np.float64),
        'clip_frame_indices': fi,
        'clip_objects':      np.array(entry['clip_objects'],  dtype=np.int32),
        'n_objects':         np.int32(entry['n_objects']),
        'object_ids':        np.array(entry['object_ids'],    dtype=np.int32),
        'display_mask':      np.array(entry['display_mask'],  dtype=bool),
        'n_points_orig':     np.int32(entry['n_points_orig']),
        'n_points_active':   np.int32(len(pi)),
    }

    # extrinsics (moving-camera datasets only)
    if 'extrinsics_w2c' in src_npz:
        cameras_w2c = src_npz['extrinsics_w2c'][fi].astype(np.float64)
        out['extrinsics_w2c'] = cameras_w2c
        out['tracks_world']   = camspace_to_world(
            tracks_XYZ.astype(np.float64), visibility, cameras_w2c)
    else:
        # Fixed camera: world space == camera space
        out['tracks_world'] = np.where(
            visibility[:, :, np.newaxis],
            tracks_XYZ.astype(np.float64),
            np.nan)

    # queries_xyt (present in adt_mini and pstudio_mini)
    if 'queries_xyt' in src_npz:
        out['queries_xyt'] = src_npz['queries_xyt'][pi].astype(np.float64)

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--index_map',  default=str(Path(__file__).parent / 'worldtrack_index_map.json'),
                        help='Path to worldtrack_index_map.json (default: alongside this script)')
    parser.add_argument('--src_dir',    required=True,
                        help='Root of original WorldTrack source data')
    parser.add_argument('--output_dir', default=str(Path(__file__).parent),
                        help='Output root; clips written to <output_dir>/<dataset>/<clip>/ (default: alongside this script)')
    parser.add_argument('--dataset',    default=None,
                        help='Only process this dataset (default: all)')
    args = parser.parse_args()

    src_root = Path(args.src_dir)
    out_root = Path(args.output_dir)

    with open(args.index_map) as f:
        index_map = json.load(f)

    if args.dataset:
        index_map = {k: v for k, v in index_map.items()
                     if k.startswith(args.dataset + '/')}

    errors = []
    src_cache = {}  # avoid reloading the same source file
    processed = set()

    total = len(index_map)
    for i, (clip_key, entry) in enumerate(index_map.items(), 1):
        dataset, clip_name = clip_key.split('/', 1)
        src_path = src_root / entry['source']

        if not src_path.exists():
            errors.append(f"{clip_key}: source not found: {src_path}")
            print(f"[{i}/{total}] ERROR  {clip_key}: source not found")
            processed.add(clip_key)
            continue

        try:
            if str(src_path) not in src_cache:
                src_cache[str(src_path)] = np.load(src_path, allow_pickle=True)
            src_npz = src_cache[str(src_path)]

            clip_data = extract_clip(src_npz, entry)

            out_dir = out_root / dataset / clip_name
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(out_dir / f'{clip_name}.npz', **clip_data)

            if entry.get('caption') is not None:
                (out_dir / 'caption.json').write_text(
                    json.dumps(entry['caption'], indent=2))

            print(f"[{i}/{total}] OK  {clip_key}")

        except Exception as e:
            errors.append(f"{clip_key}: {e}")
            print(f"[{i}/{total}] ERROR  {clip_key}: {e}")
        finally:
            processed.add(clip_key)
            # Evict source files no longer needed by any unprocessed clip
            needed = {str(src_root / v['source'])
                      for k, v in index_map.items() if k not in processed}
            for k in [k for k in src_cache if k not in needed]:
                del src_cache[k]

    print(f"\nDone: {total - len(errors)}/{total} clips written to {out_root}")
    if errors:
        print(f"{len(errors)} errors:")
        for e in errors:
            print(f"  {e}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Reconstruct DAVIS PointMotionBench videos from DAVIS 2017 Trainval frames.

Download Trainval 2017 - Images (480p) and Annotations from
https://davischallenge.org/davis2017/code.html, then run this script.

The DAVIS root should have this layout after extraction:
    <davis-root>/
    └── JPEGImages/
        └── 480p/
            └── <sequence>/
                ├── 00000.jpg
                ├── 00001.jpg
                └── ...

Example:
    python davis/reconstruct_davis.py \
        --davis-root /path/to/DAVIS \
        --output-dir davis/videos/input_480p \
        --captions   davis/davis_captions.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TARGET_FPS = 24


def ffmpeg_exe():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError(
            "ffmpeg not found; install a system ffmpeg or `pip install imageio-ffmpeg`."
        )


def encode_sequence(ffmpeg, frames_dir, dst, fps):
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"no .jpg frames found in {frames_dir}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%05d.jpg"),
        "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-an",
        str(tmp),
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(res.stderr.decode(errors="replace").strip() or "ffmpeg failed")
    os.replace(tmp, dst)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--davis-root", required=True, type=Path,
        help="DAVIS 2017 root dir containing JPEGImages/480p/"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "videos" / "input_480p",
        help="Output directory for mp4 files (default: davis/videos/input_480p/)",
    )
    parser.add_argument(
        "--captions", type=Path,
        default=Path(__file__).resolve().parent / "davis_captions.json",
        help="davis_captions.json (default: alongside this script)",
    )
    parser.add_argument("--fps", type=int, default=TARGET_FPS)
    parser.add_argument("--overwrite", action="store_true", help="Re-encode even if output exists")
    args = parser.parse_args()

    frames_root = args.davis_root / "JPEGImages" / "480p"
    if not frames_root.exists():
        print(f"ERROR: {frames_root} does not exist — check --davis-root points to the DAVIS 2017 root.")
        sys.exit(1)

    captions = json.load(open(args.captions))
    sequences = sorted(captions.keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_exe()

    print(f"[reconstruct] {len(sequences)} sequences | davis-root={args.davis_root} | out={args.output_dir}")

    done = skipped = failed = 0
    missing = []

    for i, seq in enumerate(sequences, 1):
        dst = args.output_dir / f"{seq}.mp4"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        frames_dir = frames_root / seq
        if not frames_dir.exists():
            missing.append(seq)
            print(f"[{i}/{len(sequences)}] MISSING  {seq}")
            failed += 1
            continue

        try:
            encode_sequence(ffmpeg, frames_dir, dst, args.fps)
            print(f"[{i}/{len(sequences)}] OK  {seq}")
            done += 1
        except Exception as e:
            print(f"[{i}/{len(sequences)}] ERROR  {seq}: {e}")
            failed += 1

    print(f"\n===== summary =====")
    print(f"encoded ok       : {done}")
    print(f"skipped (exists) : {skipped}")
    print(f"missing upstream : {len(missing)}")
    print(f"failed           : {failed}")
    for seq in missing[:20]:
        print(f"  [MISSING] {seq}")
    if missing or failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

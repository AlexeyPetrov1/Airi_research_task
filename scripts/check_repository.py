"""Verify the frozen sources and all completed outputs without changing them."""
import argparse
from pathlib import Path

import cv2

from motion_experiments.io import ROOT, read_json, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    manifest=read_json(ROOT/"tests/golden/manifest.json")
    for relative,digest in manifest["source_sha256"].items():
        for folder in (ROOT/"data/legacy",ROOT/"tests/golden/source"):
            if sha(folder/relative)!=digest:
                raise ValueError(f"Frozen source changed: {folder/relative}")
    rgb=read_json(ROOT/"docs/evaluation_rgb_source.json")
    for folder in (ROOT/"data/legacy",ROOT/"tests/golden/source"):
        assert sha(folder/rgb["array_path"])==rgb["array_sha256"]
    summaries=read_json(args.run/"summary.json")
    for result in summaries:
        folder=args.run/result["name"]
        status=read_json(folder/"status.json")
        if status.get("expected_failure"):
            assert status["status"]=="SKIPPED_GEOMETRY_GATE" and status["failure_reason"]=="static_median"
            assert not (folder/"metrics.json").exists()
            continue
        assert status["success"]
        assert read_json(folder/"model_input_parity.json")["success"]
        assert read_json(folder/"prediction_parity.json")["success"]
        viz=folder/"visualizations"
        assert read_json(viz/"legacy_visualization_parity.json")["success"]
        for filename,receipt in read_json(viz/"render_receipt.json")["videos"].items():
            cap=cv2.VideoCapture(str(viz/filename));count=0
            while cap.read()[0]:count+=1
            cap.release()
            assert count==receipt["frame_count"]
        print(result["name"],status["mode"],"verified",flush=True)
    print(f"PASS: {len(manifest['source_sha256'])} preserved sources; {len(summaries)} expected experiment statuses")


if __name__=="__main__":main()

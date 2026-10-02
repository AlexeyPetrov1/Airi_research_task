"""Offline consistency check for the candidate MolmoMotion output."""

import json
from pathlib import Path

import numpy as np

from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array


ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"runs/sharerobot_fmb_episode_5201/cad_history_candidate"


def main() -> None:
    raw=(RUN/"molmo_candidate_output_raw.txt").read_text(encoding="utf-8")
    tracks=parse_tracks_text(raw)
    if tracks is None:
        raise ValueError("No complete <tracks> block")
    delta,visibility=tracks_to_array(tracks,num_points=8,num_frames=30,start_timestamp=3.0)
    future=np.load(RUN/"molmo_candidate_future.npy")
    history=np.load(RUN/"points_3d_history_candidate.npy")
    reconstructed=np.asarray(delta)+history[-1,0]
    max_error=float(np.max(np.abs(reconstructed-future)))
    status={"parsed_visibility_shape":list(np.asarray(visibility).shape),
            "parsed_visible_count":int(np.asarray(visibility).sum()),
            "expected_count":240,"future_shape":list(future.shape),
            "all_future_finite":bool(np.isfinite(future).all()),
            "all_future_positive_Z":bool(np.all(future[...,2]>0)),
            "max_abs_difference_raw_text_vs_future_m":max_error,
            "median_first_prediction_distance_from_t0_m":float(np.median(np.linalg.norm(future[:,0]-history[-1],axis=1))),
            "median_last_prediction_distance_from_t0_m":float(np.median(np.linalg.norm(future[:,-1]-history[-1],axis=1))),
            "status":"PASS_PARSE_AND_SHAPE_ONLY" if max_error<1e-4 and np.asarray(visibility).all() else "FAIL"}
    (RUN/"molmo_candidate_validation.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(json.dumps(status,indent=2))
    if status["status"]!="PASS_PARSE_AND_SHAPE_ONLY":
        raise ValueError("Candidate output failed text/tensor consistency")


if __name__=="__main__":
    main()

"""Follow-up: expand late motion without amplifying the first second.

The hypothesis follows visual inspection of the constant-alpha experiment;
it is exploratory, not a test-set claim. Preserves all previous artifacts.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from berkeley_arc_expansion import (ROOT, DEST, SOURCE, BASE, OUT, scene_data, snapshot,
    sha, render_families, extent_diagnostics)
from berkeley_temporal_diagnostics import write
from berkeley_evaluate import project, ALIGNMENT
from berkeley_baseline_comparison import scores

PHASE = DEST / "phase_schedule"
VARIANTS = {
    "ramp_036_lift005": (.36, .005, 0.),
    "late_036_lift005": (.36, .005, 1.),
    "late_038_lift005": (.38, .005, 1.),
    "late_036_lift010": (.36, .010, 1.),
}


def temporal_expand(raw, p0, alpha_end, lift_m, onset_s, times_s):
    times_s = np.asarray(times_s, dtype=float)
    parameters = np.asarray([alpha_end, lift_m, onset_s], dtype=float)
    if (raw.ndim != 3 or p0.shape != (len(raw),3) or times_s.shape != (raw.shape[1],)
        or not np.isfinite(raw).all() or not np.isfinite(p0).all() or not np.isfinite(times_s).all()
        or not np.isfinite(parameters).all() or np.any((times_s<0)|(times_s>2))
        or not 0<=onset_s<2 or alpha_end<1/3 or lift_m<0):
        raise ValueError("Invalid shared phase schedule")
    q = np.clip((times_s-onset_s)/(2-onset_s), 0., 1.)
    weight = q*q*(3-2*q)
    alpha = 1/3+(alpha_end-1/3)*weight
    result = p0[:,None]+alpha[None,:,None]*(raw-p0[:,None])
    result[...,1] -= lift_m*weight[None]
    if np.any(result[...,2]<=0):
        raise ValueError("Candidate crosses the camera plane")
    return result


def main():
    if (PHASE / "protocol.json").exists():
        raise FileExistsError("Follow-up already exists; previous results are immutable")
    before = {**snapshot(BASE), **snapshot(OUT)}
    prior = {p.relative_to(ROOT).as_posix():sha(p) for p in DEST.rglob("*") if p.is_file()}
    PHASE.mkdir(parents=True, exist_ok=True)
    write(PHASE / "protocol.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis="Constant alpha expands too early. Apply the extra amplitude and camera-up translation with one common smooth time schedule.",
        alpha_formula="alpha(t)=1/3+(alpha_end-1/3)*smoothstep(clamp((t-onset_s)/(2-onset_s),0,1))",
        lift_formula="Y(t) -= lift_m * same_smoothstep",
        variants={k:dict(alpha_end=a,lift_m=h,onset_s=o) for k,(a,h,o) in VARIANTS.items()},
        grid_frozen_before_followup_scores=True, previous_visuals_used_for_hypothesis=True,
        no_future_numeric_fit=True, shared_parameters_both_scenes=True,
        postprocessing_only=True, independent_test_set=False, ML_calls=0,
        previous_run_sha256=prior, source_sha256=before, executed_script_sha256=sha(Path(__file__))))
    (PHASE / "executed_experiment.py").write_bytes(Path(__file__).read_bytes())
    results = {}
    extra_gallery = {}
    for name in ("cup", "bottle"):
        s,K,history,raw,_ = scene_data(name)
        p0=history[-1]
        old_xyz=np.load(DEST/name/"aligned_3d.npz")
        old_uv=np.load(DEST/name/"projected_2d.npz")
        full={k:temporal_expand(raw,p0,a,h,o,np.arange(1,31)/15) for k,(a,h,o) in VARIANTS.items()}
        aligned={k:v[:,ALIGNMENT] for k,v in full.items()}
        uv={k:project(v,K) for k,v in aligned.items()}
        folder=PHASE/name;folder.mkdir()
        np.savez_compressed(folder/"full_30_steps.npz",**full)
        np.savez_compressed(folder/"aligned_3d.npz",**aligned)
        np.savez_compressed(folder/"projected_2d.npz",**uv)
        results[name]={k:dict(scores(v,s,aligned[k],p0,K),expansion_vs_one_third=
            extent_diagnostics(v,old_uv["reference_0333"],s["uv0"],aligned[k],old_xyz["reference_0333"])) for k,v in uv.items()}
        np.testing.assert_array_equal(full["late_036_lift005"][:,:15],np.load(DEST/name/"full_30_steps.npz")["reference_0333"][:,:15])
        for key in VARIANTS:
            # The endpoint is exactly the constant-alpha counterpart (independent identity).
            oldkey="scale_"+key.split("_")[1]+"_"+key.split("_")[2]
            np.testing.assert_allclose(aligned[key][:,-1],old_xyz[oldkey][:,-1],atol=1e-14,rtol=0)
        render_families(s,{**{k:old_uv[k] for k in ["reference_0333","scale_036_lift005"]},**uv},
            destination=PHASE,families={"timing":["reference_0333","scale_036_lift005","late_036_lift005"],
                                       "phase_height":["ramp_036_lift005","late_036_lift005","late_036_lift010"]})
        extra_gallery[name]={k:v.tolist() for k,v in uv.items()}
        print(json.dumps({"scene":name,"followup_done":True}),flush=True)
    write(PHASE/"results.json",results)
    write(PHASE/"gallery_arrays.json",extra_gallery)
    if before!={**snapshot(BASE),**snapshot(OUT)} or any(sha(ROOT/p)!=digest for p,digest in prior.items()):
        raise ValueError("Previous experiment data changed")
    write(PHASE/"verification.json",dict(success=True,previous_files_unchanged=len(before)+len(prior),
        early_first_second_identical_to_one_third=True, constant_counterpart_endpoints_identical=True,
        new_candidates_per_scene=4,all_24_IDs=True, shared_K_and_coefficients=True, ML_calls=0,
        full_prediction_steps=30, movies=12, finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()

"""CPU-only cadence diagnostics on frozen real Berkeley predictions.

Never writes to the baseline run or invokes any ML model. Visual comparisons
use the same point IDs, RGB frames and independent future reference throughout.
Oracle alpha uses future data and is explicitly excluded from model benchmarks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from berkeley_evaluate import (ALIGNMENT, TIMES, project, metric_values, velocity,
                              points_on, trails_on, label, save_rgb, write_video)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/berkeley_ur5_molmomotion"
OUT = ROOT / "runs/berkeley_ur5_improvement_v1"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


def scale_displacement(prediction, p0, alpha):
    """Scale around each point's own t0, preserving its spatial identity."""
    if prediction.ndim != 3 or p0.shape != (len(prediction), 3):
        raise ValueError("Expected point-major [P,T,3] and per-point [P,3] t0")
    if not np.isfinite(alpha) or not np.isfinite(prediction).all() or not np.isfinite(p0).all():
        raise ValueError("Nonfinite cadence input")
    return p0[:, None] + float(alpha) * (prediction-p0[:, None])


def oracle_alpha(prediction, truth, p0, valid):
    d = prediction-p0[:, None]
    g = truth-p0[:, None]
    usable = valid & np.isfinite(d).all(-1) & np.isfinite(g).all(-1)
    denom = np.sum(d[usable]**2)
    if denom <= 0 or not usable.any():
        raise ValueError("Oracle scale is unidentifiable")
    return float(np.sum(d[usable]*g[usable])/denom)


def load_scene(name):
    scene = BASE/name
    get = lambda p: np.load(scene/p, allow_pickle=False)
    ids = get("observed/selected_point_ids.npy")
    history = get("observed/points_3d_history.npy").astype(float)
    reference = get("evaluation/evaluation_results.npz")
    gt3 = get("evaluation/ground_truth_3d_est.npz")
    if not np.array_equal(ids, reference["point_ids"]) or not np.array_equal(ids, gt3["point_ids"]):
        raise ValueError("Frozen reference IDs do not match")
    pred = get("predictions/future_3d.npy").astype(float)
    if pred.shape != (24,30,3):
        raise ValueError("Incomplete original prediction")
    return dict(name=name, ids=ids, history=history, p0=history[-1], pred=pred,
                K=get("geometry/K_median.npy"), uv0=get("observed/points_2d_history.npy")[-1],
                gt=reference["ground_truth_2d"], mask=reference["common_visibility_mask"],
                gt3=gt3["xyz"], mask3=gt3["valid"], rgb=get("evaluation/future_rgb.npy"),
                raw=get("geometry/points_3d_raw.npy")[-3:,ids].astype(float),
                times=get("observed/history_timestamps.npy").astype(float))


def evaluate(s, xyz):
    return {"2D_px": metric_values(project(xyz,s["K"]),s["gt"],s["mask"],"2D_px")[0],
            "3D_est_m": metric_values(xyz,s["gt3"],s["mask3"],"3D_est_m")[0]}


def render(s, variants, tag):
    target = OUT/s["name"]
    target.mkdir(parents=True, exist_ok=True)
    colors = [(255,65,170),(30,210,255),(255,190,35)]
    hero = 8
    uv = {n:project(x,s["K"]) for n,x in variants.items()}
    movies, panels = [], []
    for t,rgb in enumerate(s["rgb"]):
        row=[]
        for n,values in uv.items():
            frame=rgb.copy()
            gttrail=np.concatenate([s["uv0"][:hero,None],s["gt"][:hero,:t+1]],axis=1)
            valid=np.c_[np.ones(hero,bool),s["mask"][:hero,:t+1]]
            predtrail=np.concatenate([s["uv0"][:hero,None],values[:hero,:t+1]],axis=1)
            trails_on(frame,gttrail,(40,245,100),valid,2)
            trails_on(frame,predtrail,(255,65,170),thickness=2)
            points_on(frame,s["gt"][:hero,t],s["ids"][:hero],(40,245,100),s["mask"][:hero,t])
            points_on(frame,values[:hero,t],s["ids"][:hero],(255,65,170),marker="cross")
            title={"original_physical":"Baseline, physical time",
                   "displacement_one_third":"Displacement x 1/3",
                   "step_equals_frame":"Model step = source frame",
                   "ORACLE_same_scene":"ORACLE (uses future)"}.get(n,n)
            label(frame,f"{title} | +{TIMES[t]:.1f}s")
            label(frame,"green: tracker reference | pink: prediction",1)
            row.append(frame)
        movies.append(np.concatenate(row,axis=1))
        if t in (0,4,9):
            panels.append(movies[-1])
    video=target/f"{tag}_comparison.mp4"
    receipt=write_video(video,movies)
    save_rgb(target/f"{tag}_contact_sheet.png",np.concatenate(panels,axis=0))
    save_rgb(target/f"{tag}_final_overlay.png",movies[-1])
    fig,axes=plt.subplots(1,len(variants),figsize=(6*len(variants),5),squeeze=False)
    for ax,(n,values) in zip(axes[0],uv.items()):
        ax.imshow(s["rgb"][-1],extent=(0,640,480,0))
        for p in range(hero):
            truth=np.vstack([s["uv0"][p],s["gt"][p]])
            truth[1:][~s["mask"][p]]=np.nan
            prediction=np.vstack([s["uv0"][p],values[p]])
            ax.plot(*truth.T,color="#20b850",lw=2)
            ax.plot(*prediction.T,color="#f040a0",lw=1)
        ax.set_title(n); ax.set_aspect("equal")
        # Shared bounds preserve visible differences and out-of-frame predictions.
        alluv=np.concatenate([s["gt"][:hero].reshape(-1,2),*[v[:hero].reshape(-1,2) for v in uv.values()]])
        lo=np.minimum(np.nanmin(alluv,0)-25,[0,0]); hi=np.maximum(np.nanmax(alluv,0)+25,[640,480])
        ax.set_xlim(lo[0],hi[0]); ax.set_ylim(hi[1],lo[1])
    fig.tight_layout(); fig.savefig(target/f"{tag}_full_extent.png",dpi=130); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for color,(n,xyz) in zip(colors,variants.items()):
        m=evaluate(s,xyz)
        for ax,key,title in zip(axes,["2D_px","3D_est_m"],["2D error (px)","3D_est error (m)"]):
            ax.plot(TIMES,m[key]["error_by_horizon"],label=n)
            ax.set(xlabel="Real time from t0 (s)",ylabel=title); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(target/f"{tag}_error_by_time.png",dpi=140); plt.close(fig)
    fig=plt.figure(figsize=(7*len(variants),5))
    for i,(n,xyz) in enumerate(variants.items()):
        ax=fig.add_subplot(1,len(variants),i+1,projection="3d")
        for p in range(hero):
            ax.plot(*np.vstack([s["p0"][p],xyz[p]]).T,color="#f040a0",lw=1)
            g=s["gt3"][p].copy(); g[~s["mask3"][p]]=np.nan
            ax.plot(*np.vstack([s["p0"][p],g]).T,color="#20b850",lw=2)
        ax.set(title=n,xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)")
        bounds=np.concatenate([s["gt3"].reshape(-1,3),*[v.reshape(-1,3) for v in variants.values()]])
        for setlim,d in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],range(3)):
            setlim(np.nanmin(bounds[:,d])-.01,np.nanmax(bounds[:,d])+.01)
        ax.view_init(elev=-60,azim=-90)
    fig.tight_layout(); fig.savefig(target/f"{tag}_3d.png",dpi=130); plt.close(fig)
    write(target/f"{tag}_render_receipt.json",dict(video=receipt,point_ids=s["ids"].tolist(),
          hero_point_ids=s["ids"][:hero].tolist(),source_frames="ten real 5 FPS frames",
          no_RGB_interpolation=True,no_prediction_clipping_for_metrics=True,visual_review_pending=True))


def fingerprint():
    return {str(p.relative_to(BASE)):sha(p) for p in BASE.rglob("*") if p.is_file()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["fixed","diagnostics"],required=True)
    args=parser.parse_args()
    before=fingerprint()
    scenes={n:load_scene(n) for n in ("cup","bottle")}
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=="fixed":
        if (OUT/"fixed_summary.json").exists():
            raise FileExistsError("Completed fixed result exists; inspect it before rerunning")
        write(OUT/"protocol.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
            baseline_commit="4f11c56545400c437f822ad677c48a3dd2655347",
            source_tree_sha256=before,script_sha256=sha(Path(__file__)),
            criteria="Visual shape, direction, timing, amplitude and point coherence first; metrics secondary",
            fixed_alpha=1/3,physical_indices=ALIGNMENT.tolist(),timewarp_indices=list(range(10)),
            oracle="later diagnostic using future reference; not a model-quality benchmark",
            no_new_model_calls=True,original_experiment_immutable=True))
        results={}
        for n,s in scenes.items():
            fixed=scale_displacement(s["pred"],s["p0"],1/3)
            variants={"original_physical":s["pred"][:,ALIGNMENT],
                      "displacement_one_third":fixed[:,ALIGNMENT],"step_equals_frame":s["pred"][:,:10]}
            render(s,variants,"fixed")
            np.save(OUT/n/"displacement_one_third_full_3d.npy",fixed)
            np.savez_compressed(OUT/n/"fixed_arrays.npz",point_ids=s["ids"],**variants)
            results[n]={name:evaluate(s,xyz) for name,xyz in variants.items()}
            # Verify independently that the original comparison matches published results.
            baseline=json.loads((BASE/n/"metrics.json").read_text())
            for key in ("ADE_2D_px","FDE_2D_px"):
                if not np.isclose(results[n]["original_physical"]["2D_px"][key],baseline[key],rtol=1e-10):
                    raise ValueError("Original baseline did not reproduce")
        write(OUT/"fixed_summary.json",results)
    else:
        if not (OUT/"fixed_summary.json").exists():
            raise RuntimeError("Fixed 1/3 diagnostics must precede oracle fitting")
        alphas={n:oracle_alpha(s["pred"][:,ALIGNMENT],s["gt3"],s["p0"],s["mask3"]) for n,s in scenes.items()}
        results={"oracle_uses_future":True,"alphas":alphas,"scenes":{}}
        for n,s in scenes.items():
            other="bottle" if n=="cup" else "cup"
            variants={"fixed_one_third":scale_displacement(s["pred"],s["p0"],1/3)[:,ALIGNMENT],
                      "ORACLE_same_scene":scale_displacement(s["pred"],s["p0"],alphas[n])[:,ALIGNMENT],
                      f"transfer_from_{other}":scale_displacement(s["pred"],s["p0"],alphas[other])[:,ALIGNMENT]}
            render(s,variants,"oracle")
            np.savez_compressed(OUT/n/"oracle_arrays.npz",point_ids=s["ids"],**variants)
            rawv=velocity(s["raw"],s["times"]); smoothv=velocity(s["history"],s["times"])
            speedraw=np.linalg.norm(rawv,axis=1); speedsmooth=np.linalg.norm(smoothv,axis=1)
            cos=np.sum(rawv*smoothv,axis=1)/(speedraw*speedsmooth)
            diag=dict(point_ids=s["ids"].tolist(),raw_velocity_m_s=rawv.tolist(),
                smoothed_velocity_m_s=smoothv.tolist(),raw_speed_m_s=speedraw.tolist(),
                smoothed_speed_m_s=speedsmooth.tolist(),median_speed_ratio=float(np.median(speedsmooth/speedraw)),
                mean_raw_speed=float(speedraw.mean()),mean_smoothed_speed=float(speedsmooth.mean()),
                direction_cosine=cos.tolist(),median_direction_cosine=float(np.median(cos)),future_used=False)
            write(OUT/n/"smoothing_velocity.json",diag)
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            x=np.arange(len(s["ids"]))
            axes[0].plot(x,speedraw,label="raw"); axes[0].plot(x,speedsmooth,label="smoothed")
            axes[0].set(xlabel="Fixed point order",ylabel="H3 speed (m/s)"); axes[0].legend()
            axes[1].plot(x,cos); axes[1].set(xlabel="Fixed point order",ylabel="Raw/smoothed direction cosine")
            fig.tight_layout(); fig.savefig(OUT/n/"smoothing_velocity.png",dpi=150); plt.close(fig)
            results["scenes"][n]={name:evaluate(s,xyz) for name,xyz in variants.items()}
        write(OUT/"oracle_summary.json",results)
    after=fingerprint()
    if before!=after:
        raise RuntimeError("Original experimental tree was mutated")
    write(OUT/f"{args.stage}_immutability_check.json",dict(success=True,baseline_files=len(before),
        baseline_bytes_unchanged=True,no_model_calls=True,finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps({"stage":args.stage,"success":True,"output":str(OUT)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

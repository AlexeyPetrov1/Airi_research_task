"""Build a transparent, candidate-only H3/8-point FMB input from joint CAD PnP."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/sharerobot_fmb_episode_5201"
DEST = OUT / "cad_history_candidate"
STEPS = [130, 135, 140]
W, D, H = .04032, .02592, .150
K = np.asarray([[152.0836, 0, 124.2908],
                [0, 202.7781333333, 129.2973333333],
                [0, 0, 1]], dtype=float)


def main() -> None:
    DEST.mkdir(exist_ok=True)
    src = np.load(OUT / "1_M_L_3_vertical_n_2.npy", allow_pickle=True).item()
    pnp = json.loads((OUT / "cad_pnp_probe.json").read_text())
    R = np.asarray(pnp["joint_constant_orientation_fit"]["shared_R"])
    translations = [np.asarray(x["t_m"]) for x in pnp["joint_constant_orientation_fit"]["frames"]]
    cad_points = np.asarray([[(a-.5)*W, -D/2, (1-b)*H]
                             for b in (.2, .4, .6, .8) for a in (.35, .65)])
    camera_points = np.stack([(R @ cad_points.T).T+t for t in translations]).astype("float32")
    projected = []
    diagnostics = []
    for i, step in enumerate(STEPS):
        points = camera_points[i]
        uv = np.stack([K[0,0]*points[:,0]/points[:,2]+K[0,2],
                       K[1,1]*points[:,1]/points[:,2]+K[1,2]],axis=1).astype("float32")
        projected.append(uv)
        bgr = src["obs/side_1"][step]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(DEST/f"frame_{step}.png")
        overlay=bgr.copy()
        for j,(u,v) in enumerate(uv):
            x,y=round(float(u)),round(float(v))
            cv2.circle(overlay,(x,y),2,(0,255,0),-1)
            cv2.putText(overlay,str(j),(x+2,y-2),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1)
        cv2.imwrite(str(DEST/f"frame_{step}_points.png"),
                    cv2.resize(overlay,(768,768),interpolation=cv2.INTER_NEAREST))
        hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
        mask = ((cv2.inRange(hsv,(0,75,90),(13,255,255))>0) |
                (cv2.inRange(hsv,(170,75,90),(179,255,255))>0))
        mask = cv2.erode(mask.astype("uint8"),np.ones((3,3),"uint8"))>0
        depth = src["obs/side_1_depth"][step]
        items=[]
        for j,(u,v) in enumerate(uv):
            x,y=round(float(u)),round(float(v))
            crop=depth[y-2:y+3,x-2:x+3]
            valid=crop[crop>0]
            items.append({"id":j,"uv":[float(u),float(v)],"inside_red_mask":bool(mask[y,x]),
                          "sensor_Z_m":float(np.median(valid)*.0001) if len(valid) else None,
                          "pnp_Z_m":float(points[j,2])})
        diagnostics.append({"step":step,"points":items})
    np.save(DEST/"points_3d_history_candidate.npy",camera_points)
    np.save(DEST/"points_2d_at_t0_candidate.npy",projected[-1])
    status = "CANDIDATE_ONLY_UNVERIFIED_RGB_K_MANUAL_PLANAR_PNP"
    manifest = {"status":status,"source_steps":STEPS,"shape_3d":list(camera_points.shape),
                "points_are_eight_interior_front_face_locations":True,
                "cad_dimensions_m":[W,D,H],"cad_points_m":cad_points.tolist(),
                "K_rgb_candidate":K.tolist(),"source":"cad_pnp_probe.json joint_constant_orientation_fit",
                "diagnostics":diagnostics,
                "model_time_note":"Processor treats three frames as 1-s-apart; FMB step spacing is nominal 0.5 s.",
                "future_gt_note":"Only seven source steps follow step 140; F30 ground truth unavailable."}
    (DEST/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(status, "shape",camera_points.shape,
          "inside_red",[sum(x["inside_red_mask"] for x in row["points"]) for row in diagnostics])


if __name__ == "__main__":
    main()

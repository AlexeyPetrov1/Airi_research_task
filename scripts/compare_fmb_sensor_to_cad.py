"""Deproject interior sensor depth at the same eight CAD/PnP RGB landmarks."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/sharerobot_fmb_episode_5201"
DEST = OUT / "cad_history_candidate"


def main() -> None:
    source = np.load(OUT / "1_M_L_3_vertical_n_2.npy", allow_pickle=True).item()
    manifest = json.loads((DEST / "manifest.json").read_text(encoding="utf-8"))
    K = np.asarray(manifest["K_rgb_candidate"], dtype=float)
    cad = np.load(DEST / "points_3d_history_candidate.npy")
    uv = np.asarray([[p["uv"] for p in row["points"]]
                     for row in manifest["diagnostics"]])
    rows = []
    all_abs = []
    all_pairwise = []
    all_sensor = []
    for i, step in enumerate(manifest["source_steps"]):
        depth = source["obs/side_1_depth"][step]
        xyz = np.full((8, 3), np.nan)
        point_rows = []
        for j, (u, v) in enumerate(uv[i]):
            x, y = int(round(u)), int(round(v))
            patch = depth[y-2:y+3, x-2:x+3]
            positive = patch[patch > 0]
            z = float(np.median(positive) * .0001) if len(positive) else np.nan
            xyz[j] = [(u-K[0,2]) / K[0,0] * z,
                      (v-K[1,2]) / K[1,1] * z, z]
            diff = float(np.linalg.norm(xyz[j]-cad[i,j]))
            all_abs.append(diff)
            point_rows.append({"id": j, "uv": [float(u),float(v)],
                               "valid_depth_pixels_in_5x5":int(len(positive)),
                               "sensor_xyz_m":xyz[j].tolist(),
                               "cad_pnp_xyz_m":cad[i,j].astype(float).tolist(),
                               "sensor_minus_cad_xyz_norm_m":diff})
        all_sensor.append(xyz)
        pairwise=[]
        for a,b in combinations(range(8),2):
            e=float(abs(np.linalg.norm(xyz[a]-xyz[b])-
                        np.linalg.norm(cad[i,a]-cad[i,b])))
            pairwise.append(e)
            all_pairwise.append(e)
        rows.append({"step":step,"points":point_rows,
                     "median_xyz_error_m":float(np.nanmedian([p["sensor_minus_cad_xyz_norm_m"] for p in point_rows])),
                     "median_pairwise_distance_error_m":float(np.nanmedian(pairwise))})
    all_sensor=np.asarray(all_sensor)
    z_sensor=all_sensor[:,:,2].ravel()
    z_pnp=cad[:,:,2].ravel()
    point_motion=[]
    for i in range(2):
        sensor_dis=np.linalg.norm(all_sensor[i+1]-all_sensor[i],axis=1)
        pnp_dis=np.linalg.norm(cad[i+1]-cad[i],axis=1)
        point_motion.append({"steps":[manifest["source_steps"][i],manifest["source_steps"][i+1]],
                             "sensor_point_motion_median_m":float(np.nanmedian(sensor_dis)),
                             "pnp_point_motion_median_m":float(np.median(pnp_dis)),
                             "median_abs_difference_m":float(np.nanmedian(abs(sensor_dis-pnp_dis)))})
    result={"status":"CANDIDATE_SENSOR_DEPROJECTION_UNVERIFIED_DEPTH_ALIGNMENT_AND_K",
            "K_candidate":K.tolist(),"scale_m_per_raw":.0001,
            "depth_estimator":"median of positive counts in a 5x5 patch; all points inside red mask",
            "median_xyz_error_m_all_24":float(np.nanmedian(all_abs)),
            "median_pairwise_distance_error_m_all_84":float(np.nanmedian(all_pairwise)),
            "median_cad_over_sensor_Z_ratio":float(np.nanmedian(z_pnp/z_sensor)),
            "implied_scale_if_all_difference_assigned_to_sensor_m_per_raw":
            float((z_sensor@z_pnp)/(z_sensor@z_sensor)*.0001),
            "scale_fit_caveat":"Not a direct scale measurement; also absorbs PnP K, RGB-depth registration, and annotation error.",
            "frames":rows,"temporal_point_motion":point_motion}
    (DEST/"sensor_cad_comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.save(DEST/"sensor_history_candidate.npy",all_sensor.astype("float32"))
    print("median XYZ",result["median_xyz_error_m_all_24"],
          "pairwise",result["median_pairwise_distance_error_m_all_84"])
    for row in rows:
        print(row["step"],row["median_xyz_error_m"],row["median_pairwise_distance_error_m"])


if __name__ == "__main__":
    main()

"""Compare monocular predictions with tentative FMB CAD PnP and sensor depth."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/sharerobot_fmb_episode_5201"
W, D, H = .04032, .02592, .150
K_RGB = np.asarray([[152.0836, 0, 124.2908],
                    [0, 202.7781333333, 129.2973333333],
                    [0, 0, 1]], dtype=np.float64)


def peg_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) > 0) |
            (cv2.inRange(hsv, (170, 75, 90), (179, 255, 255)) > 0))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"))
    i = max(range(1, n), key=lambda j: stats[j, cv2.CC_STAT_AREA])
    return cv2.erode((labels == i).astype("uint8"),
                     np.ones((5, 5), "uint8")) > 0


def robust(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "median": float(np.median(a)) if len(a) else None,
            "p90": float(np.percentile(a, 90)) if len(a) else None}


def temporal_comparison(pred: np.lib.npyio.NpzFile, source: dict, out: Path) -> dict:
    """Compare motion of the same CAD locations and drift on a static board."""
    history_dir = out / "cad_history_candidate"
    manifest = json.loads((history_dir / "manifest.json").read_text(encoding="utf8"))
    steps = manifest["source_steps"]
    if steps != [130, 135, 140] or not set(steps).issubset(set(map(int, pred["steps"]))):
        raise ValueError("Temporal comparison requires steps 130, 135, 140")
    uv = np.asarray([[[int(round(c)) for c in point["uv"]]
                     for point in frame["points"]]
                     for frame in manifest["diagnostics"]], dtype=int)
    if not all(point["inside_red_mask"] for frame in manifest["diagnostics"]
               for point in frame["points"]):
        raise ValueError("A tracked CAD location is outside the red mask")
    cad = np.load(history_dir / "points_3d_history_candidate.npy")
    sensor = np.load(history_dir / "sensor_history_candidate.npy")
    if cad.shape != sensor.shape or cad.shape != (3, 8, 3):
        raise ValueError("Expected matching [3,8,3] histories")
    model = np.asarray([pred[f"points_{step}"][uv[i, :, 1], uv[i, :, 0]]
                        for i, step in enumerate(steps)], dtype=float)
    if not np.isfinite(model).all():
        raise ValueError("Model has nonfinite points on tracked CAD locations")

    pairs = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        dm, dc, ds = model[j] - model[i], cad[j] - cad[i], sensor[j] - sensor[i]
        pairs.append({
            "steps": [steps[i], steps[j]],
            "cad_median_point_displacement_m": float(np.median(np.linalg.norm(dc, axis=1))),
            "sensor_median_point_displacement_m": float(np.median(np.linalg.norm(ds, axis=1))),
            "model_median_point_displacement_m": float(np.median(np.linalg.norm(dm, axis=1))),
            "model_vs_cad_median_vector_error_m": float(np.median(np.linalg.norm(dm - dc, axis=1))),
            "sensor_vs_cad_median_vector_error_m": float(np.median(np.linalg.norm(ds - dc, axis=1))),
            "model_vs_sensor_median_vector_error_m": float(np.median(np.linalg.norm(dm - ds, axis=1))),
            "model_vs_cad_median_displacement_magnitude_error_m": float(np.median(
                np.abs(np.linalg.norm(dm, axis=1) - np.linalg.norm(dc, axis=1)))),
        })

    # The camera and blue board are fixed. An eroded, common blue RGB mask
    # selects the same image pixels in all three frames, excluding moving peg
    # pixels and raw sensor zeros. This checks frame-to-frame depth drift
    # independently of the planar CAD pose estimate.
    blue_masks = []
    for step in steps:
        hsv = cv2.cvtColor(source["obs/side_1"][step], cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (85, 40, 40), (140, 255, 255)) > 0
        blue_masks.append(cv2.erode(blue.astype("uint8"),
                                    np.ones((5, 5), dtype="uint8")) > 0)
    common = np.logical_and.reduce(blue_masks)
    raw_depth = [source["obs/side_1_depth"][step] for step in steps]
    common &= np.logical_and.reduce([z > 0 for z in raw_depth])
    if int(common.sum()) < 1000:
        raise ValueError("Insufficient common valid blue-board pixels")
    model_board = [pred[f"depth_{step}"][common].astype(float) for step in steps]
    sensor_board = [z[common].astype(float) * .0001 for z in raw_depth]
    static_board_pairs = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        static_board_pairs.append({
            "steps": [steps[i], steps[j]],
            "model_median_absolute_Z_change_m": float(np.median(np.abs(model_board[j] - model_board[i]))),
            "sensor_median_absolute_Z_change_m": float(np.median(np.abs(sensor_board[j] - sensor_board[i]))),
        })
    return {
        "status": "CANDIDATE_3D_POINT_MOTION_AND_INDEPENDENT_STATIC_BOARD_DRIFT",
        "tracked_points": 8,
        "point_coordinate_source": "cad_history_candidate/manifest.json: projected visible interior CAD locations",
        "cad_reference_limitation": "RGB K and planar PnP candidate, shared orientation fit",
        "sensor_reference_limitation": "assumed 0.0001 m/count and candidate RGB K; interior 5x5 valid-depth median",
        "point_motion_pairs": pairs,
        "static_board": {
            "method": "5x5-eroded common blue RGB mask; same valid raw-depth pixels across all three frames",
            "common_valid_pixels": int(common.sum()),
            "model_median_Z_m_by_step": [float(np.median(z)) for z in model_board],
            "sensor_median_Z_m_by_step": [float(np.median(z)) for z in sensor_board],
            "pairs": static_board_pairs,
        },
    }


def main() -> None:
    source = np.load(OUT / "1_M_L_3_vertical_n_2.npy", allow_pickle=True).item()
    pnp = json.loads((OUT / "cad_pnp_probe.json").read_text(encoding="utf-8"))
    predictions = sorted(OUT.glob("*_metric_predictions.npz"))
    report = {"status": "EXPLORATORY_PNP_K_AND_SENSOR_ALIGNMENT_UNVERIFIED",
              "sensor_scale_m_per_raw_unit": .0001,
              "pnp_source": "cad_pnp_probe.json",
              "models": {}}
    for pred_path in predictions:
        pred = np.load(pred_path)
        model_rows = []
        pnp_rows = []
        for step in map(int, pred["steps"]):
            bgr = source["obs/side_1"][step]
            mask = peg_mask(bgr)
            sensor = source["obs/side_1_depth"][step].astype(float) * .0001
            valid = mask & (sensor > 0)
            nn_depth = pred[f"depth_{step}"]
            model_rows.append({"step": step, "peg_pixels": int(mask.sum()),
                               "valid_sensor_pixels": int(valid.sum()),
                               "sensor_peg_Z_median_m": float(np.median(sensor[valid])) if valid.any() else None,
                               "model_peg_Z_median_m": float(np.median(nn_depth[valid])) if valid.any() else None,
                               "model_minus_sensor_Z_median_m":
                               float(np.median(nn_depth[valid]-sensor[valid])) if valid.any() else None})
            if step not in (130, 135, 140):
                continue
            row = next(r for r in pnp["frames"] if r["step"] == step)
            pose = row["candidates"][0]
            R, t = np.asarray(pose["R"]), np.asarray(pose["t_m"])
            samples = []
            for a in (.2, .5, .8):
                for b in (.25, .5, .75):
                    cad = np.asarray([(a-.5)*W, -D/2, (1-b)*H])
                    xyz = R @ cad + t
                    u = int(round(K_RGB[0, 0] * xyz[0]/xyz[2] + K_RGB[0, 2]))
                    v = int(round(K_RGB[1, 1] * xyz[1]/xyz[2] + K_RGB[1, 2]))
                    if not (2 <= u < 254 and 2 <= v < 254):
                        continue
                    nn_xyz = pred[f"points_{step}"][v, u].astype(float)
                    nn_z = float(nn_depth[v, u])
                    local = sensor[v-2:v+3, u-2:u+3]
                    local = local[local > 0]
                    samples.append({"uv": [u, v], "inside_eroded_peg": bool(mask[v, u]),
                                    "pnp_xyz_m": xyz.tolist(), "model_xyz_m": nn_xyz.tolist(),
                                    "model_minus_pnp_Z_m": nn_z-float(xyz[2]),
                                    "model_minus_pnp_XYZ_m": float(np.linalg.norm(nn_xyz-xyz)),
                                    "sensor_median_Z_m": float(np.median(local)) if len(local) else None})
            valid_samples = [x for x in samples if x["inside_eroded_peg"]]
            pairwise = []
            for i, j in combinations(range(len(valid_samples)), 2):
                a, b = valid_samples[i], valid_samples[j]
                pnp_d = np.linalg.norm(np.asarray(a["pnp_xyz_m"])-b["pnp_xyz_m"])
                nn_d = np.linalg.norm(np.asarray(a["model_xyz_m"])-b["model_xyz_m"])
                pairwise.append(float(nn_d-pnp_d))
            pnp_rows.append({"step": step, "samples": samples,
                             "inside_eroded_count": len(valid_samples),
                             "model_minus_pnp_Z_m": robust([x["model_minus_pnp_Z_m"] for x in valid_samples]),
                             "model_minus_pnp_XYZ_m": robust([x["model_minus_pnp_XYZ_m"] for x in valid_samples]),
                             "pairwise_distance_error_m": robust([abs(x) for x in pairwise])})
        report["models"][pred_path.stem] = {"peg_sensor_comparison": model_rows,
                                              "pnp_comparison": pnp_rows,
                                              "temporal_comparison": temporal_comparison(pred, source, OUT)}
        print(pred_path.stem)
        for r in model_rows:
            print(r["step"], "peg Z", round(r["sensor_peg_Z_median_m"], 3),
                  round(r["model_peg_Z_median_m"], 3))
        for r in pnp_rows:
            print("pnp", r["step"], "inside", r["inside_eroded_count"],
                  "median XYZ error", r["model_minus_pnp_XYZ_m"]["median"])
    (OUT / "metric_geometry_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

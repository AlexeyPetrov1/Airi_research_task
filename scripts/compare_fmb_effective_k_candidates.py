"""Compare camera-K hypotheses using held-out RGB board and sensor geometry."""

from __future__ import annotations

import csv
import json

import numpy as np

from calibrate_fmb_board_k import HOLDOUT, HOLDOUT_FEATURES, TRAIN, cad_features, depth_crosscheck, extract
from calibrate_fmb_multi_boards_k import DATA, EPISODES
from calibrate_fmb_two_boards_k import fit_multi, score_multi
from probe_fmb_tcp_tip_k import OUT


FIRST_RUN = OUT.parent / "sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126"


def history_rigidity(k: np.ndarray, source: dict) -> tuple[float, float]:
    uv = np.load(FIRST_RUN / "history_points_2d.npy")
    assert uv.shape == (3, 8, 2)
    xyz = []
    for step, rows in zip((124, 125, 126), uv):
        points = []
        depth = source["obs/side_1_depth"][step]
        for uf, vf in rows:
            u, v = np.rint((uf, vf)).astype(int)
            patch = depth[v-2:v+3, u-2:u+3]
            valid = patch[patch > 0]
            z = float(np.median(valid)*.0001)
            points.append([(uf-k[0, 2])*z/k[0, 0],
                           (vf-k[1, 2])*z/k[1, 1], z])
        xyz.append(np.array(points))
    def rms(a, b):
        p, q = a-a.mean(0), b-b.mean(0)
        u, _, vt = np.linalg.svd(p.T@q)
        reflect = np.diag([1., 1., np.linalg.det(vt.T@u.T)])
        rot = vt.T@reflect@u.T
        diff = p@rot.T-q
        return float(np.sqrt(np.mean(np.sum(diff*diff, axis=1)))*1000)
    return rms(xyz[0], xyz[2]), rms(xyz[1], xyz[2])


def main() -> None:
    points, _, _ = cad_features(1)
    features = np.array([i for i in range(len(points)) if i not in HOLDOUT_FEATURES])
    train, hold, rgb, depth = [], [], [], []
    raw_first = None
    for number in EPISODES:
        source = np.load(DATA/f"1_M_L_3_vertical_n_{number}.npy", allow_pickle=True).item()
        if number == 2:
            raw_first = source
        images = source["obs/side_1"]
        obs = {s: extract(images[s], 1)[0] for s in TRAIN+HOLDOUT}
        train.append(np.array([obs[s] for s in TRAIN]))
        hold.append(np.array([obs[s] for s in HOLDOUT]))
        rgb.append(images[list(HOLDOUT)].copy())
        depth.append(source["obs/side_1_depth"][list(HOLDOUT)].copy())
        if number != 2:
            del source
    previous = json.loads((OUT / "multi_board_bundle_calibration.json").read_text(encoding="utf8"))
    variants = json.loads((FIRST_RUN / "k_sensitivity_variants.json").read_text(encoding="utf8"))["variants"]
    fit_k = np.array(previous["best_fit"]["K"])
    variants["K_board_RGB_5_poses"] = {"K": fit_k.tolist()}
    peg = json.loads((OUT.parent / "sharerobot_fmb_episode_5201/effective_k_rounded_cad_silhouette.json").read_text(encoding="utf8"))
    variants["K_peg_RGB_silhouette"] = {"K": peg["best_fit"]["K"]}
    cross = json.loads((OUT / "cross_orientation_silhouette_k_probe.json").read_text(encoding="utf8"))
    variants["K_cross_orientation_RGB"] = {"K": cross["best_fit"]["K"]}
    contact = json.loads((OUT / "board_peg_table_contact_K_probe.json").read_text(encoding="utf8"))
    variants["K_board_peg_contact_RGB"] = {"K": contact["best_fit"]["K"]}
    multi_contact = json.loads((OUT / "multi_horizontal_contact_K_probe.json").read_text(encoding="utf8"))
    variants["K_multi_horizontal_contact_RGB"] = {"K": multi_contact["best_fit"]["K"]}
    tcp = json.loads((OUT / "multi_episode_tcp_tip_K_probe.json").read_text(encoding="utf8"))
    variants["K_multi_episode_RGB_TCP_tip"] = {"K": tcp["best_fit"]["K"]}
    shared_tcp = json.loads((OUT / "shared_tcp_offset_K_probe.json").read_text(encoding="utf8"))
    variants["K_shared_RGB_TCP_tip"] = {"K": shared_tcp["best_fit"]["K"]}
    tcp_checks = tcp["fixed_RGB_only_K_candidate_TCP_checks"]
    rows = []
    old_x = np.array(previous["best_fit"]["x"])
    for name, item in variants.items():
        k = np.array(item["K"])
        seed = old_x.copy()
        seed[:4] = [np.log(k[0, 0]), np.log(k[1, 1]), k[0, 2], k[1, 2]]
        fit = fit_multi(points, tuple(train), features, seed, fix_k=True)
        x = np.array(fit["x"])
        held = score_multi(points, tuple(hold), x, np.arange(len(points)))
        dcheck = [depth_crosscheck(np.r_[x[:4], x[4+6*i:10+6*i]],
                                   rgb[i], depth[i], tuple(range(len(HOLDOUT))))
                  for i in range(len(EPISODES))]
        rigid = history_rigidity(k, raw_first)
        if name in tcp_checks:
            tcp_consistency = (f"heldout RGB tip RMSE "
                               f"{tcp_checks[name]['holdout']['rmse_px']:.2f} px "
                               "after fitting camera extrinsic and per-episode TCP offsets")
        elif name in ("K_multi_episode_RGB_TCP_tip", "K_shared_RGB_TCP_tip"):
            tcp_consistency = "not independent: this K was fitted to RGB/TCP tip data"
        else:
            tcp_consistency = "not evaluated on RGB/TCP tip holdout"
        rows.append({"K_variant": name,
                     "fx_px": float(k[0, 0]), "fy_px": float(k[1, 1]),
                     "cx_px": float(k[0, 2]), "cy_px": float(k[1, 2]),
                     "board_train_rgb_rmse_px_mean": float(np.mean(
                         [row["rmse_px"] for row in fit["train"]])),
                     "board_holdout_rgb_rmse_px_mean": float(np.mean(
                         [row["rmse_px"] for row in held])),
                     "board_depth_median_abs_Z_error_mm_mean": float(np.mean(
                         [row["median_abs_Z_error_mm"] for row in dcheck])),
                     "board_depth_signed_Z_bias_mm_mean": float(np.mean(
                         [row["median_signed_Z_error_mm"] for row in dcheck])),
                     "sensor_history_rigidity_124_to_126_RMS_mm": rigid[0],
                     "sensor_history_rigidity_125_to_126_RMS_mm": rigid[1],
                     "TCP_consistency": tcp_consistency})
    with (OUT / "K_comparison.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "K_comparison.json").write_text(json.dumps({
        "purpose": "Diagnostic comparison; do not select K using sensor depth or MolmoMotion ADE",
        "board_episodes": EPISODES,
        "history_episode": 2,
        "history_steps": [124, 125, 126],
        "rows": rows}, indent=2)+"\n", encoding="utf8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

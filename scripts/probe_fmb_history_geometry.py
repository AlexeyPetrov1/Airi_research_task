"""Pre-inference sensor-history consistency for candidate FMB point sets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probe_fmb_cad_pnp import K


RUN = Path(__file__).resolve().parents[1] / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001"
SELECTED = ((35, .35), (35, .55), (45, .25), (45, .45),
            (55, .35), (55, .75), (65, .25), (65, .55))


def kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ac, bc = a.mean(0), b.mean(0)
    u, _, vh = np.linalg.svd((a - ac).T @ (b - bc))
    reflection = np.eye(3)
    reflection[2, 2] = np.linalg.det(u @ vh)
    rotation = vh.T @ reflection @ u.T
    translation = bc - rotation @ ac
    predicted = (rotation @ a.T).T + translation
    return rotation, translation, float(np.sqrt(np.mean(np.sum((predicted - b)**2, axis=1))))


def main() -> None:
    candidates = json.loads((RUN / "history_point_candidates.json").read_text(encoding="utf8"))
    selected = [next(c for c in candidates if c["y_at_t0"] == y and c["alpha"] == alpha)
                for y, alpha in SELECTED]
    uv = np.asarray([[row["frames"][i]["uv"] for row in selected] for i in range(3)], dtype=float)
    z = np.asarray([[row["frames"][i]["median_raw"] * .0001 for row in selected]
                    for i in range(3)], dtype=float)
    xyz = np.stack(((uv[:, :, 0]-K[0, 2])/K[0, 0]*z,
                    (uv[:, :, 1]-K[1, 2])/K[1, 1]*z, z), axis=2)
    distances = np.linalg.norm(xyz[:, :, None] - xyz[:, None, :], axis=3)
    upper = np.triu_indices(8, 1)
    pairwise = distances[:, upper[0], upper[1]]
    for i, j in ((0, 1), (1, 2), (0, 2)):
        _, _, rmse = kabsch(xyz[i], xyz[j])
        print((124+i, 124+j), "Kabsch mm", round(rmse*1000, 2),
              "point shift mm", np.round(np.linalg.norm(xyz[j]-xyz[i], axis=1)*1000, 1))
    print("median pairwise std mm", np.median(np.std(pairwise, axis=0))*1000,
          "p90", np.percentile(np.std(pairwise, axis=0), 90)*1000,
          "max", np.max(np.std(pairwise, axis=0))*1000)
    print("XYZ ranges m", xyz.min(axis=(0, 1)), xyz.max(axis=(0, 1)))
    np.save(RUN / "history_sensor_candidate_3d.npy", xyz.astype("float32"))


if __name__ == "__main__":
    main()

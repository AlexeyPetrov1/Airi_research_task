"""Plot diagnostic K estimates and independent depth residuals."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from probe_fmb_cad_pnp import K as K_NOMINAL
from probe_fmb_tcp_tip_k import OUT, SOURCE


def params(k: np.ndarray) -> np.ndarray:
    return np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])


def main() -> None:
    tip = json.loads((OUT / "tcp_tip_k_probe.json").read_text(encoding="utf8"))
    cad = json.loads((OUT / "tcp_cad_k_probe.json").read_text(encoding="utf8"))
    bottom = json.loads((OUT / "tcp_bottom_k_probe.json").read_text(encoding="utf8"))
    silhouette = json.loads((SOURCE.parent / "effective_k_rounded_cad_silhouette.json").read_text(
        encoding="utf8"))
    names = ("nominal K", "RGB CAD silhouette", "RGB TCP tip",
             "RGB CAD + TCP", "RGB bottom CAD + TCP")
    values = np.array([params(K_NOMINAL),
                       params(np.array(silhouette["best_fit"]["K"])),
                       params(np.array(tip["best_fit"]["K"])),
                       params(np.array(cad["best_fit"]["K"])),
                       params(np.array(bottom["best_fit"]["K"]))])
    mc = [None, np.array(silhouette["monte_carlo_K_fx_fy_cx_cy"]),
          np.array(tip["annotation_perturbation_K"]),
          np.array(cad["monte_carlo_K"]),
          np.array(bottom["monte_carlo_K"])]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for j, (ax, label) in enumerate(zip(axs.ravel(), ("fx", "fy", "cx", "cy"))):
        for i, name in enumerate(names):
            ax.scatter(values[i, j], i, s=45, label=name if j == 0 else None)
            if mc[i] is not None and len(mc[i]):
                lo, hi = np.percentile(mc[i][:, j], (5, 95))
                ax.plot((lo, hi), (i, i), lw=3, alpha=.6)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel(f"{label} (px)")
        ax.grid(alpha=.2)
    fig.suptitle("FMB side_1: candidate effective K (dots) and 5–95% 2 px noise stress ranges")
    fig.savefig(OUT / "K_candidate_stability.png", dpi=160)
    plt.close(fig)

    rows = cad["sensor_depth_crosscheck_holdout"]["per_frame"]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for row in rows:
        step = row["step"]
        for error in row["signed_error_m"]:
            ax.scatter(step, error*1000, s=8, alpha=.45, color="#356ca8")
    ax.axhline(0, color="black", lw=1)
    ax.axvline(100, color="gray", ls="--", lw=1)
    ax.set(xlabel="FMB frame", ylabel="CAD/PnP Z minus sensor Z (mm)",
           title="Independent depth check of rigid TCP + CAD candidate")
    ax.grid(alpha=.2)
    fig.savefig(OUT / "tcp_cad_sensor_depth_residuals.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

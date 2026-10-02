"""Create a fixed-scale visual comparison for the FMB depth experiment."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/sharerobot_fmb_episode_5201"


def main() -> None:
    source = np.load(OUT/"1_M_L_3_vertical_n_2.npy",allow_pickle=True).item()
    moge2 = np.load(OUT/"moge2_aspect_fov_metric_predictions.npz")
    moge3 = np.load(OUT/"moge3_aspect_fov_metric_predictions.npz")
    uni = np.load(OUT/"unidepthv2_metric_predictions.npz")
    steps = (0,130,140)
    fig, axes = plt.subplots(len(steps),5,figsize=(16,10),constrained_layout=True)
    names = ("RGB", "D405 sensor", "MoGe-2 aspect+FOV",
             "MoGe-3 aspect+FOV", "UniDepthV2")
    for i,step in enumerate(steps):
        bgr=source["obs/side_1"][step]
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        sensor=source["obs/side_1_depth"][step].astype(float)*.0001
        sensor[sensor==0]=np.nan
        maps=(rgb,sensor,moge2[f"depth_{step}"],
              moge3[f"depth_{step}"],uni[f"depth_{step}"])
        for j,(name,arr) in enumerate(zip(names,maps)):
            ax=axes[i,j]
            if j==0:
                ax.imshow(arr)
            else:
                im=ax.imshow(arr,cmap="turbo",vmin=.1,vmax=1.2)
            ax.set_title(f"step {step} · {name}")
            ax.axis("off")
    fig.colorbar(im,ax=axes[:,1:].ravel().tolist(),label="Z (m)",shrink=.75)
    fig.savefig(OUT/"metric_depth_comparison.png",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

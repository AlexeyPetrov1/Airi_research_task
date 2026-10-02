"""Compare dense image flow with sparse KLT on the textureless red peg."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from inspect_fmb_2d_window import red_component
from probe_fmb_cad_pnp import SOURCE
from probe_fmb_future_tracking import initial_points


OUT = Path(__file__).resolve().parents[1] / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001"


def bilinear(flow: np.ndarray, uv: np.ndarray) -> np.ndarray:
    x, y = uv.T
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x0 = np.clip(x0, 0, 254)
    y0 = np.clip(y0, 0, 254)
    dx, dy = (x - x0)[:, None], (y - y0)[:, None]
    return ((1-dx)*(1-dy)*flow[y0, x0] + dx*(1-dy)*flow[y0, x0+1] +
            (1-dx)*dy*flow[y0+1, x0] + dx*dy*flow[y0+1, x0+1])


def follow(frames: np.ndarray, start: int, end: int, p0: np.ndarray) -> dict[int, np.ndarray]:
    tracked = {start: p0.copy()}
    direction = 1 if end > start else -1
    for step in range(start+direction, end+direction, direction):
        before = cv2.cvtColor(frames[step-direction], cv2.COLOR_BGR2GRAY)
        after = cv2.cvtColor(frames[step], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(before, after, None, .5, 4, 25, 5, 7, 1.5, 0)
        tracked[step] = tracked[step-direction] + bilinear(flow, tracked[step-direction])
    return tracked


def main() -> None:
    data = np.load(SOURCE, allow_pickle=True).item()
    frames = data["obs/side_1"]
    initial = initial_points(frames[126])
    positions = follow(frames, 126, 124, initial) | follow(frames, 126, 146, initial)
    canvas = np.full((2*512, 4*512, 3), 255, np.uint8)
    show = (124, 125, 126, 130, 135, 140, 145, 146)
    rows = {}
    for step in range(124, 147):
        mask, _ = red_component(frames[step])
        inside = []
        for u, v in positions[step]:
            x, y = np.rint([u, v]).astype(int)
            inside.append(bool(mask[y, x]) if 0 <= x < 256 and 0 <= y < 256 else False)
        rows[str(step)] = {"uv": positions[step].tolist(), "red": inside}
        if step in show:
            i = show.index(step)
            image = frames[step].copy()
            for j, (u, v) in enumerate(positions[step]):
                xy = tuple(np.rint([u, v]).astype(int))
                cv2.circle(image, xy, 2, (0, 255, 0) if inside[j] else (0, 0, 0), -1)
                cv2.putText(image, str(j), (xy[0]+2, xy[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                            .3, (255, 255, 255), 1)
            cv2.putText(image, str(step), (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (255, 255, 255), 2)
            canvas[i//4*512:(i//4+1)*512, i%4*512:(i%4+1)*512] = cv2.resize(
                image, (512, 512), interpolation=cv2.INTER_NEAREST)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "dense_flow_probe.png"), canvas)
    (OUT / "dense_flow_probe.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
    for step in show:
        print(step, sum(rows[str(step)]["red"]), "on red")


if __name__ == "__main__":
    main()

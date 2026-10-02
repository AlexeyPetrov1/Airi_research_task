"""Probe affine registration of object masks as a second future 2D tracker."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from inspect_fmb_2d_window import red_component
from probe_fmb_cad_pnp import SOURCE
from probe_fmb_future_tracking import initial_points


OUT = Path(__file__).resolve().parents[1] / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001"


def segment(frame: np.ndarray) -> np.ndarray:
    mask, _ = red_component(frame)
    return cv2.GaussianBlur(mask.astype("float32"), (0, 0), 2.)


def follow(frames: np.ndarray, start: int, end: int, initial: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
    result = {start: initial.copy()}
    info = {}
    direction = 1 if end > start else -1
    for step in range(start+direction, end+direction, direction):
        prev = segment(frames[step-direction])
        curr = segment(frames[step])
        pc = np.array(np.where(prev > .3)).mean(axis=1)[::-1]
        cc = np.array(np.where(curr > .3)).mean(axis=1)[::-1]
        warp = np.float32([[1, 0, cc[0]-pc[0]], [0, 1, cc[1]-pc[1]]])
        try:
            score, warp = cv2.findTransformECC(prev, curr, warp, cv2.MOTION_AFFINE,
                                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                100, 1e-6))
            info[step] = {"score": float(score), "warp": warp.tolist()}
        except cv2.error as error:
            info[step] = {"error": str(error), "warp": warp.tolist()}
        homogeneous = np.c_[result[step-direction], np.ones(len(initial))]
        result[step] = (warp @ homogeneous.T).T
    return result, info


def main() -> None:
    data = np.load(SOURCE, allow_pickle=True).item()
    frames = data["obs/side_1"]
    initial = initial_points(frames[126])
    previous, info_prev = follow(frames, 126, 124, initial)
    future, info_future = follow(frames, 126, 146, initial)
    positions = previous | future
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
    cv2.imwrite(str(OUT / "mask_ecc_probe.png"), canvas)
    (OUT / "mask_ecc_probe.json").write_text(json.dumps({"tracks": rows,
        "fits": info_prev | info_future}, indent=2), encoding="utf8")
    for step in show:
        print(step, sum(rows[str(step)]["red"]), "on red", (info_prev | info_future).get(step, {}).get("score"))


if __name__ == "__main__":
    main()

"""Shared lightweight typed dicts for the public data interface.

Defined here (instead of in the dataset modules) so that downstream
preprocessing/eval modules can depend on them without pulling in the full
academic-dataset stack — the rest of which has been stripped from the
release.
"""

from __future__ import annotations

from typing import Dict, TypedDict


class Point(TypedDict):
    point: list[float]
    occluded: bool


class PointTrack(TypedDict):
    """One frame of a multi-object point trajectory.

    `points` is a mapping `object_id -> {'point': [x, y], 'occluded': bool}`,
    used as the on-the-wire representation that the model's text decoder
    parses back into a per-track 2D trajectory.
    """

    frame: int
    time: float
    points: Dict[int, Point]

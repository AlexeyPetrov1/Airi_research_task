"""Original pinhole projection and camera transforms, with explicit conventions."""
import numpy as np


def transform(matrix, points):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def project(points, k):
    points = np.asarray(points)
    uv = points[..., :2] / points[..., 2, None]
    return uv * np.array([k[0, 0], k[1, 1]]) + [k[0, 2], k[1, 2]]


def project_future(points, k, poses=None):
    if poses is None:
        return project(points, k)
    return np.stack([project(transform(np.linalg.inv(poses[t+1]) @ poses[0], points[:, t]), k)
                     for t in range(points.shape[1])], axis=1)


def align_future(future, initial, times):
    source = np.r_[0, np.arange(1, future.shape[1]+1)/15]
    values = np.concatenate([initial[:, None], future], axis=1)
    return np.array([[np.interp(times, source, values[n, :, a]) for a in range(3)]
                     for n in range(len(initial))]).transpose(0, 2, 1)


def linear_velocity(history, times):
    centered = times - times.mean()
    return np.sum(centered[:, None, None] * (history-history.mean(axis=0)), axis=0) / np.sum(centered**2)

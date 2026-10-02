"""Numerical checks for alignment and SIMPLE_RADIAL convention conversion."""
import numpy as np
import cv2
import pycolmap
from scipy.spatial.transform import Rotation
from dobbe_colmap_official import CAMERA_TO_LABEL, camera_candidate, sim3

rng = np.random.default_rng(17)
x = rng.normal(size=(12, 3))
rotation = Rotation.from_rotvec([.2, -.3, .5]).as_matrix()
scale, translation = 2.7, np.array([.4, -.7, .2])
y = scale*x@rotation.T+translation
poses = np.repeat(np.eye(4)[None], len(x), axis=0)
poses[:, :3, 3] = y
poses[:, :3, :3] = rotation @ CAMERA_TO_LABEL.T
trajectory = []
for i, center in enumerate(x):
    c2w = np.eye(4)
    c2w[:3, 3] = center
    trajectory.append({"frame": i, "camera_center": center.tolist(), "c2w_colmap": c2w.tolist()})
result = sim3(trajectory, poses)
assert np.allclose(result["rotation"], rotation)
assert np.allclose(result["translation"], translation)
assert np.isclose(result["scale"], scale)
assert result["ate_rmse_m"] < 1e-12
assert result["rotation_error_median_deg"] < 1e-10
camera = pycolmap.Camera(model="SIMPLE_RADIAL", width=256, height=256, params=[237.2, 128, 128, -.013])
candidate = camera_candidate(camera)
p = candidate["params"]
k = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
xyz = np.array([[.2, .3, 1.], [-.3, .4, 2.], [.4, -.2, 1.5]])
cv_pixels = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, np.array(p[4:]))[0].reshape(-1, 2)
colmap_pixels = camera.img_from_cam(xyz)
assert np.allclose(cv_pixels, colmap_pixels-.5, atol=1e-12)
ray = np.asarray(camera.cam_from_img(colmap_pixels[0]))
assert ray.shape == (2,) and np.allclose(ray, xyz[0, :2]/xyz[0, 2])
assert np.allclose(pycolmap.Rigid3d() * xyz[0], xyz[0])
print("PASS: Sim(3) scale, rotation, translation and orientation; SIMPLE_RADIAL/OpenCV projection")

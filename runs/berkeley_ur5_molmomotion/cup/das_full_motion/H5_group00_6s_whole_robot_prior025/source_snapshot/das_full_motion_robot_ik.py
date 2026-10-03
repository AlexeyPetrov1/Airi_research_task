"""Observed nominal UR5 IK diagnostic for the unchanged group_00 cup arc."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from das_robot_geometry import fk
from das_full_motion_diagnose import SCENE, OUT
from das_prepare_control import project, write_json
from das_full_motion_safety import forbid_real_future


def solve(camera, q0, rotations, translations):
    first = camera @ fk(q0)
    qs = [q0.copy()]
    errors = [[0., 0.]]
    lo = np.full(6, -2*np.pi); hi = -lo
    lo[2] = -np.pi; hi[2] = np.pi
    for r, t in zip(rotations[1:], translations[1:]):
        delta = np.eye(4); delta[:3,:3] = r; delta[:3,3] = t
        target = np.linalg.inv(camera) @ delta @ first[6]
        previous = qs[-1]
        lower = np.maximum(lo, previous-2./8)
        upper = np.minimum(hi, previous+2./8)
        def loss(q):
            actual = fk(q)[6]
            return np.r_[(actual[:3,3]-target[:3,3])/.005,
                Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).as_rotvec()/.05,
                (q-previous)*.01]
        fit = least_squares(loss, np.clip(previous, lower+1e-8, upper-1e-8),
            bounds=(lower,upper), max_nfev=200, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        qs.append(fit.x)
        actual = fk(fit.x)[6]
        errors.append([np.linalg.norm(actual[:3,3]-target[:3,3]),
            np.linalg.norm(Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).as_rotvec())])
    return np.array(qs), np.array(errors), camera @ np.array([fk(q) for q in qs])


def main():
    forbid_real_future()
    out=OUT/'whole_robot_ik_diagnostic';out.mkdir(exist_ok=True)
    if (out/'kinematics.npz').exists():raise FileExistsError('Preserve completed diagnostic')
    motion=np.load(OUT/'group00_stretched_6s/motion.npz')
    observed=np.load(SCENE/'das_robot_cup/observed/nominal_ur5_camera.npz')
    q, errors, fc=solve(observed['camera_from_base'],observed['q0'],motion['R'],motion['t'])
    np.savez_compressed(out/'kinematics.npz',joints=q,fk_camera=fc,errors=errors)
    uv=project(fc[:,:,:3,3].reshape(-1,3),observed['K']).reshape(49,7,2)
    write_json(out/'receipt.json',{'future_used':False,'cup_forecast_changed':False,
        'max_flange_position_error_m':float(errors[:,0].max()),
        'max_flange_orientation_error_rad':float(errors[:,1].max()),
        'max_joint_speed_rad_s':np.max(np.abs(np.diff(q,axis=0))*8,axis=0).tolist(),
        'joint_uv':uv.tolist(), 'method':'Observed-only nominal camera registration, bounded UR5 IK; diagnostic does not replace the exact cup trajectory',
        'limitation':'Approximate video geometry, no factory calibration or full collision validation'})
    print('max flange error',errors.max(0),'initial joints UV',uv[0],'final joints UV',uv[-1],flush=True)


if __name__=='__main__':main()

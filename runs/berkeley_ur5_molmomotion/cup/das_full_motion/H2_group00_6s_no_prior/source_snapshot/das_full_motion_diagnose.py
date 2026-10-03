"""Inspect the frozen forecast without opening any future RGB or annotations."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from das_prepare_control import kabsch, project, write_json

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'runs/berkeley_ur5_molmomotion/cup'
OUT = SCENE / 'das_full_motion'


def weighted_kabsch(p0, target, weights):
    weights = weights / weights.sum()
    a, b = weights @ p0, weights @ target
    u, _, vt = np.linalg.svd(((p0-a)*weights[:, None]).T @ (target-b))
    r = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    return r, b-r@a


def fit_motion(p0, pred, method):
    indices = np.arange(8) if method == 'group00' else np.arange(len(p0))
    source = p0[indices]
    rotations, translations, residuals, weights = [], [], [], []
    for target in pred[indices].transpose(1, 0, 2):
        r, t, rms, _ = kabsch(source, target)
        weight = np.ones(len(source))
        if method == 'robust24':
            # Huber IRLS; fixed 20 mm threshold, independent of future data.
            for _ in range(30):
                error = np.linalg.norm(source@r.T+t-target, axis=1)
                weight = np.minimum(1., .02/np.maximum(error, 1e-9))
                rr, tt = weighted_kabsch(source, target, weight)
                if np.max(np.abs(rr-r)) + np.max(np.abs(tt-t)) < 1e-10:
                    r, t = rr, tt
                    break
                r, t = rr, tt
            rms = np.sqrt(np.mean(np.sum((source@r.T+t-target)**2, axis=1)))
        rotations.append(r); translations.append(t); residuals.append(rms); weights.append(weight)
    return np.array(rotations), np.array(translations), np.array(residuals), np.array(weights)


def main():
    OUT.mkdir(exist_ok=True)
    pred = np.load(SCENE/'predictions/future_3d.npy').astype(float)
    p0 = np.load(SCENE/'observed/points_3d_history.npy')[-1].astype(float)
    k = np.load(SCENE/'geometry/K_median.npy')
    summary = {'future_used': False, 'groups': {}, 'fits': {}}
    for group in range(3):
        points = slice(group*8, (group+1)*8)
        uv = project(pred[points].transpose(1, 0, 2).reshape(-1, 3), k).reshape(30, 8, 2)
        summary['groups'][str(group)] = {'initial_uv': project(p0[points], k).mean(0).tolist(),
            'centers_uv_at_steps_1_8_15_23_30': uv.mean(1)[[0, 7, 14, 22, 29]].tolist()}
    for method in ['group00', 'all24', 'robust24']:
        r, t, rms, weights = fit_motion(p0, pred, method)
        np.savez_compressed(OUT/f'fit_{method}.npz', R=r, t=t, residual=rms, weights=weights)
        rigid = np.einsum('tij,nj->tni', r, p0[:8])+t[:, None]
        uv = project(rigid.reshape(-1, 3), k).reshape(30, 8, 2)
        rawuv = project(pred[:8].transpose(1, 0, 2).reshape(-1, 3), k).reshape(30, 8, 2)
        summary['fits'][method] = {'mean_rms_m': float(rms.mean()), 'final_rms_m': float(rms[-1]),
            'centers_uv_at_steps_1_8_15_23_30': uv.mean(1)[[0, 7, 14, 22, 29]].tolist(),
            'group00_projection_ADE_px': float(np.linalg.norm(uv-rawuv, axis=-1).mean()),
            'rotation_degrees_at_steps_1_8_15_23_30': np.rad2deg(Rotation.from_matrix(r).magnitude())[[0, 7, 14, 22, 29]].tolist()}
    write_json(OUT/'forecast_diagnostics.json', summary)
    print(summary)


if __name__ == '__main__':
    main()

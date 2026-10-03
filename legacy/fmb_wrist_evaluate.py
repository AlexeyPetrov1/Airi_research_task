import numpy as np


def scores(pred,gt,mask):
    dist=np.linalg.norm(pred-gt,axis=-1);valid=mask&np.isfinite(dist)
    return {'ADE':float(np.mean(dist[valid])) if valid.any() else None,'FDE':float(np.mean(dist[:,-1][valid[:,-1]])) if valid[:,-1].any() else None,
            'valid_samples':int(valid.sum()),'valid_final_points':int(valid[:,-1].sum()),
            'per_time_mean':[float(np.mean(dist[:,t][valid[:,t]])) if valid[:,t].any() else None for t in range(20)],
            'per_point_mean':[float(np.mean(dist[i][valid[i]])) if valid[i].any() else None for i in range(24)]}

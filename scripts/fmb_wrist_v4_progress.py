"""Report only fully completed neural variants without changing selection."""
import json,numpy as np
from fmb_wrist_prepare import write
from fmb_wrist_v4_diagnose import BASE,OUT
from fmb_wrist_v4_evaluate import align
from fmb_wrist_evaluate import scores

rows=[]
for camera in ['wrist_2','wrist_1']:
    ref=np.load(BASE/camera/'evaluation/future_reference.npz')
    for name in ['H3_rigid_observed','H1_native']:
        folder=OUT/camera/'variants'/name;receipt=folder/'model_run.json'
        if not receipt.exists() or not json.loads(receipt.read_text())['success']:continue
        p=align(np.load(folder/'future_3d.npy'),np.load(folder/'history_xyz.npy')[-1])
        s=scores(p,ref['GT_3D_est'],ref['common_mask3d'])
        rows.append({'camera':camera,'variant':name,'ADE_mm':s['ADE']*1000,'FDE_mm':s['FDE']*1000,'not_used_for_selection':True})
write(OUT/'neural_progress.json',rows)
print(json.dumps(rows),flush=True)

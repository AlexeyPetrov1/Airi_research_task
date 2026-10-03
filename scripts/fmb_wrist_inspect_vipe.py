"""Inspect actual ViPE IDs and pose calibration before downstream model work."""
import json,zipfile
import numpy as np
from fmb_wrist_prepare import RUN,write
from fmb_wrist_geometry import load_vipe
from fmb_wrist_math import tcp_matrices,hand_eye,static_pairs,static_reprojection

for camera in ['wrist_2','wrist_1']:
    scene=RUN/camera
    if not(scene/'vipe_default/pose/observed_10hz.npz').exists():continue
    for kind in ['pose','intrinsics']:
        data=np.load(scene/f'vipe_default/{kind}/observed_10hz.npz')
        print(camera,kind,'keys',data.files,'shape',data['data'].shape,'IDs',data['inds'].tolist(),flush=True)
    pose,K,depth=load_vipe(scene)
    robot=tcp_matrices(np.load(scene/'observed/tcp_pose_xyzw.npy'));x,y,kin,info=hand_eye(robot,pose,np.arange(40),np.arange(40,50))
    print(camera,'K',K[-1].tolist(),'X',x.tolist(),'heldout',info['heldout'],flush=True)
    write(scene/'vipe_early_hand_eye_probe.json',info)

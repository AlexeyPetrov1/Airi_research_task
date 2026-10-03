"""Retain quantitative unmodified-square ViPE control without changing primary inputs."""
import json
import numpy as np
from fmb_wrist_prepare import RUN,write
from fmb_wrist_geometry import load_vipe
from fmb_wrist_math import tcp_matrices,hand_eye,static_reprojection,static_pairs

for camera in ['wrist_2','wrist_1']:
    scene=RUN/camera;poses,K,z=load_vipe(scene,use_selection=False)
    rgb=np.load(scene/'observed/rgb.npy');sensor=np.load(scene/'observed/sensor_z16.npy').astype(float)*1e-4
    robot=tcp_matrices(np.load(scene/'observed/tcp_pose_xyzw.npy'))
    x,y,kin,info=hand_eye(robot,poses,np.arange(40),np.arange(40,50));pairs=static_pairs(rgb)
    prior=np.load(scene/'geometry/official_K.npy')
    result={'hand_eye':info,'K_median':np.median(K,0).tolist(),'source':'Original 256x256 recorded RGB, native default ViPE unchanged',
       'geometry':{'A_vipe_full':static_reprojection(z,K,poses,pairs),'B_sensor_vipe':static_reprojection(sensor,K,poses,pairs),
                   'C_canonical_vipe_X':static_reprojection(sensor,prior,kin,pairs)},'future_used':False}
    write(scene/'geometry/native_square_control.json',result);print(camera,'native square control retained',flush=True)

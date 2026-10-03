"""Plot the observed-only articulated control and camera-fit uncertainty."""
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from das_full_motion_diagnose import SCENE,OUT
from das_full_motion_safety import forbid_real_future
from das_prepare_control import write_json


def main():
    forbid_real_future()
    kin=OUT/'whole_robot_ik_diagnostic';data=np.load(kin/'kinematics.npz')
    report=json.loads((kin/'receipt.json').read_text())
    camera=json.loads((SCENE/'das_robot_cup/observed/geometry_audit.json').read_text())
    times=np.arange(49)/8
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    axes[0].imshow(Image.open(SCENE/'observed/frame_000063.png'),alpha=.4)
    uv=np.array(report['joint_uv'])
    for joint in [1,2,4,6]:axes[0].plot(*uv[:,joint].T,'o-',markersize=2,label=f'Nominal joint {joint}')
    axes[0].set(xlim=(0,900),ylim=(480,-400),xlabel='Native u (px)',ylabel='Native v (px)',title='IK joint paths; cup geometry stays unchanged')
    axes[0].legend()
    for joint in range(6):axes[1].plot(times,data['joints'][:,joint]-data['joints'][0,joint],label=f'Q{joint+1}')
    axes[1].set(xlabel='Video time (s; 3x stretched)',ylabel='Joint change from observed t0 (rad)',title='Nominal UR5 configuration change');axes[1].legend()
    fig.tight_layout();fig.savefig(kin/'articulated_joint_paths.png',dpi=150);plt.close(fig)
    write_json(kin/'registration_scope.json',{'future_used':False,
        'nominal_DH_source':'https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/',
        'DH_parameters_verified_against_official_UR5_table':True,
        'observed_camera_fit_joint_reprojection_error_px':camera['fit_joint_reprojection_error_px'],
        'observed_camera_fit_joint_depth_error_m':camera['fit_joint_depth_error_m'],
        'numerical_IK_vs_calibration':'Tiny nominal IK residual solves the assumed chain only. It does not reduce observed camera registration uncertainty.',
        'rasterization':'Joint endpoint similarity maps retain full t0 RGB link silhouettes; not a CAD renderer or factory-calibrated 3D reconstruction.',
        'exact_preserved_geometry':'Cup and local wrist/gripper dense SE(3) geometry are copied byte-for-byte from H2.'})


if __name__=='__main__':main()

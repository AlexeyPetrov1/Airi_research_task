"""Observed-selected better-excited control; preserve primary 5201 diagnostics."""
import json,os,shutil
from pathlib import Path
import fmb_wrist_prepare as prepare

base=prepare.RUN
control=prepare.ROOT/'runs/fmb_wrist_v3_control_horizontal_n0'
control.mkdir(exist_ok=True)
shutil.copytree(base/'sources',control/'sources',dirs_exist_ok=True)
shutil.copytree(base/'selection',control/'selection',dirs_exist_ok=True)
protocol=(base/'PROTOCOL.md').read_text()
header='# Observed-selected horizontal n0 control\n\nThis separate run uses raw `1_M_L_3_horizontal_n_0.npy`, t0=216, observed 167–216, H3 214–216, future 217–236. It is FMB-only; no ShareRobot match is asserted. Episode 5201 remains intact in the primary run. This control was selected before predictions/future from stronger observed multi-axis rotation (56.1deg, singular values 2.597/0.336/0.132 rad versus 11.1deg, 0.563/0.096/0.061 for 5201), visible target and sensor support. Its official source profiles and all geometry/time/visibility controls remain the same.\n\n'
(control/'PROTOCOL.md').write_text(header+protocol,encoding='utf-8')
shutil.copyfile(base/'REPRODUCE.md',control/'REPRODUCE.md')
prepare.write(control/'control_selection.json',{'source':'1_M_L_3_horizontal_n_0.npy','candidate_t0':216,'future_used':False,
     'selection_reason':'Main 5201 observed ViPE hand-eye fails physical gate; direct hand-eye bootstrap has large parameter and relative-path spread. Horizontal n0 has stronger multi-axis excitation and target depth support.',
     'primary_5201_preserved':True,'source_robot_rotation_max_deg':56.11043299984557,'source_rotation_singular_values_rad':[2.59700018,.33612023,.13193654],
     'wrist2_t0_supported_depth_interior_pixels':3111,'wrist1_t0_supported_depth_interior_pixels':1504,'share_robot_mapping_status':'UNVERIFIED_FMB_ONLY',
     'main_bootstrap_evidence':'../fmb_wrist_v3/wrist_2/geometry/independent_robot_rgbd_bootstrap.json'})
prepare.RUN=control
prepare.export('1_M_L_3_horizontal_n_0.npy',216)
print('Independent control prepared',control,flush=True)

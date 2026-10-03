"""Serialize the assistant's completed built-in image review; never run a HF VLM."""
import json
from datetime import datetime,timezone
import numpy as np
from fmb_wrist_prepare import ROOT,RUN,write,sha

reviews=[]
assert not(RUN/'builtin_visual_review_observed.json').exists(),'Recorded review is immutable; perform a new documented review instead of changing its timestamp'
def reviewed(path,source_ids,judgment,flags=None):
    reviews.append({'image':path,'sha256':sha(RUN/path),'source_indices':source_ids,'assessment':judgment,'point_flags':flags or {},'inspection_method':'Assistant built-in view_image; qualitative visual judgment'})

reviewed('wrist_2/viz/selected_history_tracks.png',[116,121,124,125,126],
    'All 24 displayed IDs remain on visible red peg surface in these observed frames. Board moves while gripped target is nearly fixed in image. Coverage is concentrated on the top face, with only IDs38,70 lower on the blade. Low texture prevents certification of material-point identity.',
    {'38':'Visible on blade below gripper; weak surface texture, correspondence uncertain','70':'Visible on lower blade; weak surface texture, correspondence uncertain','other_selected_ids':'Visible on top red face, mostly clustered; no obvious background leakage'})
reviewed('wrist_1/viz/selected_history_tracks.png',[116,121,124,125,126],
    'Displayed IDs stay on the red target and cover top face plus blade. No obvious board or metal-gripper points in reviewed frames. Camera motion is visible on the blue board. Low texture still makes small correspondence drift hard to judge.',
    {'0,38':'Near gripper contact edge; visible red surface but occlusion risk','24':'Near right blade boundary; remains inside visible red surface in samples','75':'Near lower blade, weak texture','other_selected_ids':'Inside target in sampled frames, visually stable'})
for camera in ['wrist_2','wrist_1']:
    reviewed(camera+'/viz/selected_24_points.png',[126],'The 24 frozen point IDs are on the visible red object. This confirms visible-surface membership, not metric depth or exact physical correspondence.')
reviewed('wrist_2/viz/pose_K_comparison.png',list(range(77,127)),
    'ViPE path has spikes, particularly near the held-out suffix. AX=XB rotation error grows above20 degrees. Independently selected TCP path diverges substantially from ViPE. Restored-aspect ViPE fx is near official prior; UniDepth fx is larger. Geometry cannot be certified from pose agreement.')
reviewed('wrist_1/viz/pose_K_comparison.png',list(range(77,127)),
    'ViPE camera path shows depth-direction spikes and >30 degree suffix rotation residual. ViPE focal lengths are much larger than official/UniDepth. Selected independent TCP path disagrees with ViPE; do not claim matching calibrated poses.')
reviewed('wrist_2/viz/depth_comparison.png',[77,101,124,126],
    'Sensor support has holes on peg and gripper; ViPE gives continuous depth with much larger scale. Difference maps and colorbars support a serious scale discrepancy. Displayed sensor darkness includes near values, so black-looking regions alone are not treated as invalid.')
reviewed('wrist_2/viz/history_3d_clouds.png',[124,125,126],
    'A uses a substantially larger spatial scale than sensor branches. B and C differ in camera-motion compensation. Two blade points are depth-separated from clustered top-face points. Finite clouds/roundtrip integrity do not prove registration or physical geometry.')
reviewed('wrist_crossview_base_clouds.png',[126],
    'The two blue-board clouds form displaced sheets and the red target clouds do not coincide. This fails absolute multiview agreement, consistent with large hand-eye uncertainty; local reprojection gate is insufficient for physical certification.')
reviewed('side_1/viz/crossview_board_matches.png',[126],
    'Only five depth-supported static feature matches; repetitive holes and different surfaces make some matches ambiguous. No side-camera base alignment is accepted.')
reviewed('wrist_1/viz/intrinsics_observed_frames.png',[77,88,100,111,123,124,125,126],
    'ViPE focal estimate is far above official prior and UniDepth across the entire observed prefix. UniDepth varies over time. Official RGB linkage remains a prior, not measured effective recording calibration.')

frozen={}
for camera in ['wrist_2','wrist_1']:
    scene=RUN/camera;selection=json.loads((scene/'geometry/branch_selection.json').read_text())
    selection.update(physical_geometry_certified=False,primary_interpretation='Observed within-view control only; absolute wrist cross-view agreement fails',builtin_visual_review_completed=True)
    write(scene/'geometry/branch_selection.json',selection)
    for path in [scene/'geometry/branch_selection.json',scene/'geometry/selected_hand_eye_X.npy',scene/'geometry/selected_hand_eye.json',scene/'geometry/official_K.npy',scene/'geometry/vipe_source_selection.json',scene/'observed/selected_point_ids.npy']:
        frozen[str(path.relative_to(RUN))]=sha(path)
    for name in selection['branches']:
        dest=scene/'branches'/name
        for relative in ['observed/points_3d_history.npy','observed/history_rgb.npy','geometry/K.npy','geometry/camera_poses.npy','metadata.json']:
            frozen[str((dest/relative).relative_to(RUN))]=sha(dest/relative)
write(RUN/'builtin_visual_review_observed.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'Root assistant via built-in image inspection',
      'external_HF_VLM_used':False,'future_images_used':False,'reviewed_images':reviews,'frozen_geometry_sha256':frozen,
      'primary_wrist_2':'C_sensor_tcp_official','replication_wrist_1':'C_sensor_tcp_official',
      'inference_allowed_as':'Controlled estimated-geometry comparison; A/B rejected as diagnostics, C provisional within-view primary',
      'limitations':'Qualitative review, single previously studied episode, concentrated wrist2 point coverage, depth scale/registration and hand-eye uncertain; no certified 3D GT.'})
print('Built-in observed review recorded; 24 IDs and observed geometry frozen',flush=True)

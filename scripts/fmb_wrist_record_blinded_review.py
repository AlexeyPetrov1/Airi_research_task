"""Record actual first built-in future inspection before revealing method identities."""
from datetime import datetime,timezone
import numpy as np
from fmb_wrist_prepare import RUN,write,sha

assert not(RUN/'builtin_blinded_review.json').exists(),'Keep first blinded assessment immutable'
images=[]
def seen(path,indices,assessment):images.append({'image':path,'sha256':sha(RUN/path),'source_indices':indices,'assessment':assessment})
for camera in ['wrist_2','wrist_1']:
    seen(camera+'/viz/reference_all24_review.png',[127,136,146],
      'All displayed frozen IDs stay on visible red target in the three sampled future frames. Target remains nearly camera-fixed while board/background moves. No obvious board/gripper leakage. Low texture prevents exact material-identity certification; blade becomes less fully visible near insertion.')
seen('wrist_2/viz/blinded_review_01.png',[127],
     'All five forecasts are still in frame. Slots2/3 appear close to reference, with modest point offsets. Slots1/4/5 already shift head points upward. No physical3D ranking inferred.')
seen('wrist_2/viz/blinded_review_10.png',[136],
     'Slots1/4/5 have severe upward projection errors, with6-7 of8 displayed forecasts outside image. Slot2 remains in frame but cap points shift left/down relative to green reference. Slot3 has smaller apparent2D offsets at1s and remains mostly near red surface.')
seen('wrist_2/viz/blinded_review_20.png',[146],
     'Slots1/4/5 have7 of8 displayed forecasts outside, with long upward arrows. Slot2 remains in frame but forecasts are left/below cap reference. Slot3 remains in frame but cap points drift upward onto gripper/background by2s. None matches all reference points; apparent2D closeness does not certify depth.')
seen('wrist_1/viz/blinded_review_20.png',[146],
     'All three slots show severe projection mismatch at2s. Slot1 shifts far downward, including3 of8 off-image; slot2 has5 of8 outside and distorted vertically separated points; slot3 sends all8 displayed forecasts outside upward/right. Reference remains on red target.')
points={}
for camera in ['wrist_2','wrist_1']:
    ids=np.load(RUN/camera/'observed/selected_point_ids.npy');rows=[]
    for point in ids:
        flag='Inside visible red surface in sampled frames; low texture leaves exact correspondence uncertain'
        if camera=='wrist_2' and point in [38,70]:flag='Blade interior remains on red surface; weak texture and increasing insertion occlusion make material identity uncertain'
        if camera=='wrist_1' and point in [0,38,24]:flag='Red surface near gripper/boundary; sampled membership appears correct, but contact/occlusion risk remains'
        if camera=='wrist_1' and point==75:flag='Lower blade close to insertion region; visible in samples, changing apparent extent and low texture limit identity confidence'
        rows.append({'point_id':int(point),'reviewed_source_indices':[127,136,146],'visible_target_membership':'supported in sampled images','material_identity_confidence':'uncertain','assessment':flag})
    points[camera]=rows
write(RUN/'builtin_blinded_review.json',{'recorded_utc':datetime.now(timezone.utc).isoformat(),'method_map_read_before_record':False,
      'quantitative_metrics_read_before_record':False,'reviewed_images':images,'point_reviews':points,
      'reviewer':'Root assistant via built-in view_image','external_HF_VLM_used':False,
      'HF_flag_scope':'Quality assessment only; native MolmoPoint grounding and MolmoMotion remain part of the experiment',
      'limits':'Only sampled RGB/projection images; cannot establish calibrated3D GT or exact identities on a low-texture object. Prior reconstruction hypotheses were known, so blinding hides labels but is not fully independent.'})
print('First qualitative judgments saved before method-map/metric disclosure',flush=True)

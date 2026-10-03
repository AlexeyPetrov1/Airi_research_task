"""Protocol failures that would invalidate the FMB scientific comparison."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fmb_v2_infer import strict_parse
from fmb_v2_evaluate import interpolate_prediction, metric, visualize
from fmb_v2_preprocess import author_filter

def text(h,f):
    return '<tracks coords="'+';'.join(str(t+h)+' '+' '.join(f'{p} {p*100} 0 0' for p in range(1,9)) for t in range(f))+'">peg</tracks>'

@pytest.mark.parametrize('h,f',[(3,30),(1,32)])
def test_full_parser(h,f):
    delta=strict_parse(text(h,f),h,f)
    assert delta.shape==(8,f,3)
    np.testing.assert_allclose(delta[:,0,0],np.arange(1,9)/10)

@pytest.mark.parametrize('mutation',[
    lambda s:s.replace('2 200 0 0','1 200 0 0',1),
    lambda s:s.replace('3 1 100','4 1 100',1),
    lambda s:s.replace(';32 ', ';33 ',1),
    lambda s:s.replace('100 0 0','100 0',1),
])
def test_bad_ids_or_records_fail(mutation):
    with pytest.raises(ValueError):strict_parse(mutation(text(3,30)),3,30)

def test_nominal_time_interpolation_uses_time_not_frame_index():
    times=np.arange(1,31)/15
    prediction=np.zeros((24,30,3));prediction[:,:,0]=times
    aligned=interpolate_prediction(prediction)
    np.testing.assert_allclose(aligned[0,:,0],np.arange(1,21)/10)
    assert aligned[0,0,0]==pytest.approx(.1)
    with pytest.raises(ValueError):interpolate_prediction(prediction,np.array([2.1]))

def test_common_mask_keeps_large_outside_image_predictions():
    gt=np.zeros((2,20,2));prediction=gt.copy();prediction[0,:,0]=1000
    result,_=metric(prediction,gt,np.ones((2,20),bool),'2D_px')
    assert result['ADE_2D_px']==500

def test_author_filter_handles_only_invisible_tracks_dropped():
    w=np.tile(np.linspace(.3,.8,100),(8,1)).astype(np.float32)
    vis=np.ones((8,100),bool);vis[:,0]=False
    drop=author_filter().filter_tracks_by_trust(w,vis,z_thresh=2)
    assert drop.sum()==1 and drop[0]

def test_media_contains_real_time_frame_count_and_full_extents(tmp_path):
    import cv2
    (tmp_path/'observed').mkdir();(tmp_path/'viz').mkdir()
    history=np.zeros((3,256,256,3),np.uint8);np.save(tmp_path/'observed/rgb.npy',history)
    frames=np.zeros((20,256,256,3),np.uint8)
    ids=np.arange(24);gt=np.full((24,20,2),100.,np.float32)
    prediction=gt.copy();prediction[:,:,0]=np.linspace(100,400,20)
    mask=np.ones((24,20),bool)
    methods={'MolmoMotion_H3':prediction,'static':gt,'constant_velocity':gt}
    values={}
    for name,uv in methods.items():
        m2,_=metric(uv,gt,mask,'2D_px')
        m3,_=metric(np.zeros((24,20,3)),np.zeros((24,20,3)),mask,'3D_est_m')
        values[name]={'2D':m2,'3D_est':m3}
    visualize(tmp_path,frames,np.repeat(gt[:,0][None],3,axis=0),ids,gt,mask,methods,{}, {},{'methods':values})
    cap=cv2.VideoCapture(str(tmp_path/'viz/side_by_side.mp4'))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==20
    assert cap.get(cv2.CAP_PROP_FPS)==pytest.approx(10)
    cap.release()
    assert (tmp_path/'viz/full_extent_trajectories.png').stat().st_size>1000

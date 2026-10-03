"""Preservation receipts and image-only observed inspection material."""
from __future__ import annotations
import json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
from fmb_wrist_prepare import ROOT,RUN,CAMERAS,write,sha
import berkeley_preprocess as b

def run():
    provenance=json.loads((ROOT/'runs/sharerobot_fmb_episode_5201/preflight.json').read_text())
    meta=json.loads((RUN/'wrist_2/metadata.json').read_text())
    assert provenance['source_mapping_status']=='PASS_SOURCE_EPISODE_RECOVERED'
    ismatched=provenance['source_npy_sha256']==meta['source_sha256']
    write(RUN/('source_episode_5201_provenance.json' if ismatched else 'source_fmb_control_provenance.json'),{'source_mapping_status':provenance['source_mapping_status'] if ismatched else 'FMB_ONLY_NO_SHARE_ROBOT_MATCH',
          'new_raw_source_sha256':meta['source_sha256'],'matches_independently_recovered_v1_source':ismatched,
          'source_audit_path':'runs/sharerobot_fmb_episode_5201/preflight.json','source_audit_sha256':sha(ROOT/'runs/sharerobot_fmb_episode_5201/preflight.json')})
    old=json.loads((ROOT/'runs/fmb_v2_berkeley_matched/preserved_v1.json').read_text())['sha256']
    differences=[name for name,digest in old.items() if not(ROOT/name).exists() or sha(ROOT/name)!=digest]
    assert not differences,differences
    write(RUN/'preserved_prior_runs.json',{'v1_checked_files':len(old),'v1_unchanged':True,
          'v2_completion_audit_sha256':sha(ROOT/'runs/fmb_v2_berkeley_matched/completion_audit.json'),
          'v2_report_sha256':sha(ROOT/'report/fmb_v2_berkeley_matched.md'),
          'v2_completion_audit_success':json.loads((ROOT/'runs/fmb_v2_berkeley_matched/completion_audit.json').read_text())['success']})
    for camera in CAMERAS:
        scene=RUN/camera;rgb=np.load(scene/'observed/rgb.npy');depth=np.load(scene/'observed/sensor_z16.npy').astype(float)*1e-4
        indices=np.load(scene/'observed/source_indices.npy');panels=[]
        for i in [0,10,20,30,40,47,48,49]:
            im=cv2.resize(rgb[i],(384,384));cv2.putText(im,f'{camera}: source {indices[i]}',(8,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2);panels.append(im)
        b.save_rgb(scene/'viz/observed_prefix_review.png',np.vstack([np.hstack(panels[:4]),np.hstack(panels[4:])]))
        rows=[]
        for i in [0,24,49]:
            z=depth[i];valid=z>0;filled=np.where(valid,z,np.median(z[valid])).astype(np.float32)
            edge=(np.hypot(cv2.Sobel(filled,cv2.CV_32F,1,0),cv2.Sobel(filled,cv2.CV_32F,0,1))>.012)&(cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8))>0)
            re=cv2.Canny(rgb[i],50,120)>0
            color=(matplotlib.colormaps['magma'](np.clip(z/.8,0,1))[...,:3]*255).astype(np.uint8);color[~valid]=0
            over=rgb[i].copy();over[edge]=[0,255,50];d=color.copy();d[re]=[0,255,255]
            row=[]
            for title,im in [('RGB',rgb[i]),('Z16 x1e-4',color),('Depth edges on RGB',over),('RGB edges on depth',d)]:
                tile=cv2.resize(im,(384,384));cv2.putText(tile,f'{indices[i]} {title}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2);row.append(tile)
            rows.append(np.hstack(row))
        b.save_rgb(scene/'viz/registration_review.png',np.vstack(rows))
    print('Prior FMB files unchanged; observed review material ready',len(old),flush=True)

if __name__=='__main__':run()

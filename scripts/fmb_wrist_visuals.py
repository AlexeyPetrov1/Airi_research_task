"""Observed-only readable plots for the frozen 24 IDs and geometry controls."""
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fmb_wrist_prepare import RUN,write
import berkeley_preprocess as b

def observed(camera):
    scene=RUN/camera;viz=scene/'viz';rgb=np.load(scene/'observed/rgb.npy')
    uv=np.load(scene/'observed/observed_tracks_2d.npz')['tracks'];ids=np.load(scene/'observed/selected_point_ids.npy')
    base=cv2.resize(rgb[-1],(768,768),interpolation=cv2.INTER_LINEAR)
    candidates=b.draw_points(base,uv[-1]*3,radius=2)
    selected=b.draw_points(base,uv[-1,ids]*3,ids,radius=3)
    panels=[]
    for im,label in [(candidates,'100 AllTracker candidates'),(selected,'24 common frozen IDs')]:
        cv2.putText(im,label,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);panels.append(im)
    b.save_rgb(viz/'candidates_to_selected.png',np.hstack(panels))
    b.save_rgb(viz/'selected_24_points.png',selected)
    history_panels=[]
    for t in [39,44,47,48,49]:
        tile=b.draw_points(cv2.resize(rgb[t],(512,512)),uv[t,ids]*2,ids,radius=2)
        cv2.putText(tile,f'{camera} source {np.load(scene/"observed/source_indices.npy")[t]}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);history_panels.append(tile)
    b.save_rgb(viz/'selected_history_tracks.png',np.hstack(history_panels))
    names=list(json.loads((scene/'geometry/branch_selection.json').read_text())['branches'])
    source_ids=json.loads((scene/'metadata.json').read_text())['history_source_indices']
    fig,axes=plt.subplots(len(names),3,figsize=(13,11))
    for row,name in enumerate(names):
        hist=np.load(scene/'branches'/name/'observed/points_3d_history.npy')
        for col,(a,c) in enumerate([(0,1),(0,2),(1,2)]):
            ax=axes[row,col]
            for t in range(3):ax.scatter(hist[t,:,a],hist[t,:,c],s=14,label=f'source {source_ids[t]}')
            for j in range(24):ax.plot(hist[:,j,a],hist[:,j,c],color='gray',alpha=.3)
            ax.set(xlabel='XYZ'[a]+' in C_t0 (m)',ylabel='XYZ'[c]+' in C_t0 (m)',title=name);ax.grid(alpha=.2);ax.axis('equal');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(viz/'history_3d_clouds.png',dpi=150);plt.close(fig)
    Ks=np.load(scene/'geometry/unidepth_K.npy');indices=np.load(scene/'observed/source_indices.npy')[[0,11,23,34,46,47,48,49]].tolist()
    official=np.load(scene/'geometry/official_K.npy');vipe=np.load(scene/'geometry/vipe_K.npy')
    fig,axes=plt.subplots(2,2,figsize=(10,7));values={}
    for ax,(r,c,label) in zip(axes.ravel(),[(0,0,'fx'),(1,1,'fy'),(0,2,'cx'),(1,2,'cy')]):
        v=Ks[:,r,c];ax.plot(indices,v,'o-',label='UniDepth observed');ax.axhline(official[r,c],label='Official profile prior',color='gray')
        ax.plot(np.load(scene/'observed/source_indices.npy'),vipe[:,r,c],label='ViPE',color='orange');ax.set(xlabel='source frame',ylabel=label+' (px)');ax.grid(alpha=.2);ax.legend(fontsize=7)
        values[label]={'median':float(np.median(v)),'min':float(v.min()),'max':float(v.max()),'relative_range':float(np.ptp(v)/np.median(v))}
    fig.tight_layout();fig.savefig(viz/'intrinsics_observed_frames.png',dpi=150);plt.close(fig)
    write(scene/'geometry/intrinsics_variation.json',{'unidepth':values,'source_indices':indices,'future_used':False})
    print(camera,'observed media complete',flush=True)

if __name__=='__main__':
    for camera in ['wrist_2','wrist_1']:observed(camera)

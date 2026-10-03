"""Hide method labels for the assistant's first qualitative forecast assessment."""
import json
import cv2,numpy as np
from fmb_wrist_prepare import RUN,write
from fmb_wrist_math import from_anchor,project
from fmb_wrist_evaluate import prediction_gate,align_prediction,TIMES
import berkeley_preprocess as b

for camera in ['wrist_2','wrist_1']:
    prediction_gate();scene=RUN/camera;ref=np.load(scene/'evaluation/future_reference.npz');rgb=np.load(scene/'evaluation/future_rgb.npy')
    poses=ref['camera_c2w'];K=ref['K'];uv=ref['GT_2D_est'];methods={}
    selection=json.loads((scene/'geometry/branch_selection.json').read_text());primary=selection['primary']
    for branch in sorted((scene/'branches').glob('*')):
        if not(branch/'predictions/future_3d.npy').exists():continue
        h=np.load(branch/'observed/points_3d_history.npy');p=np.load(branch/'predictions/future_3d.npy')
        methods[branch.name]=align_prediction(p,h[-1])
        if branch.name==primary:
            methods[branch.name+'/Static']=np.repeat(h[-1,:,None],20,axis=1)
            methods[branch.name+'/CV']=h[-1,:,None]+((h[2]-h[0])/.2)[:,None]*TIMES[None,:,None]
    names=list(methods);np.random.default_rng(314 if camera=='wrist_2' else 271).shuffle(names)
    write(scene/'evaluation/blinding_map.json',{'slots':{f'Forecast {i+1}':name for i,name in enumerate(names)},'labels_hidden_in_images':True,
          'limitation':'Method identities hidden from first image assessment; reviewer already knows reconstruction hypotheses'})
    projections={name:np.stack([project(from_anchor(xyz[:,t],poses[t+1],poses[0]),K) for t in range(20)],axis=1) for name,xyz in methods.items()}
    for t in [0,9,19]:
        panels=[];base=cv2.resize(rgb[t+1],(512,512));display=np.arange(0,24,3)
        truth=base.copy()
        for j in display:cv2.circle(truth,tuple(np.rint(uv[j,t]*2).astype(int)),4,(40,255,40),-1)
        cv2.putText(truth,f'Reference, t={TIMES[t]:.1f}s',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2);panels.append(truth)
        for i,name in enumerate(names):
            im=base.copy();outside=0
            for j in display:
                p=projections[name][j,t]*2;real=uv[j,t]*2
                if np.isfinite(p).all():
                    if (p>=0).all() and (p<512).all():cv2.circle(im,tuple(np.rint(p).astype(int)),5,(255,70,240),-1)
                    else:
                        outside+=1
                        if (np.abs(p)<1e6).all():cv2.arrowedLine(im,tuple(np.rint(real).astype(int)),tuple(np.rint(p).astype(int)),(255,70,240),1,tipLength=.08)
                cv2.circle(im,tuple(np.rint(real).astype(int)),3,(40,255,40),-1)
            cv2.putText(im,f'Forecast {i+1}, outside {outside}/8',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2);panels.append(im)
        while len(panels)%3:panels.append(np.zeros_like(base))
        image=np.vstack([np.hstack(panels[i:i+3]) for i in range(0,len(panels),3)])
        b.save_rgb(scene/'viz'/f'blinded_review_{t+1:02d}.png',image)
    print(camera,'method identities hidden for image review',flush=True)

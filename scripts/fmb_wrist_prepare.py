"""Observed-only FMB wrist selection, source receipts, and causal exports."""
from __future__ import annotations
import argparse, hashlib, json, os, urllib.request
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
RUN=Path(os.environ.get('FMB_WRIST_RUN',str(ROOT/'runs/fmb_wrist_v3')))
DATA=ROOT.parent/'data/fmb/single_object_manipulation_dataset'
CAMERAS=['side_1','side_2','wrist_1','wrist_2']

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n',encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()

def sources():
    folder=RUN/'sources';folder.mkdir(parents=True,exist_ok=True)
    repos={'calibration':'functional-manipulation-benchmark/functional-manipulation-benchmark.github.io','capture':'rail-berkeley/fmb'}
    commits={k:json.load(urllib.request.urlopen(f'https://api.github.com/repos/{repo}/commits/main'))['sha'] for k,repo in repos.items()}
    tree=json.load(urllib.request.urlopen(f'https://api.github.com/repos/{repos["capture"]}/git/trees/{commits["capture"]}?recursive=1'))
    paths=[x['path'] for x in tree['tree'] if x['path'].endswith('.py') and any(s in x['path'] for s in ['camera/','/envs/','franka_server','collect'])]
    receipts=[]
    for role,repo in repos.items():
        for name in [f'static/files/{c}' for c in CAMERAS] if role=='calibration' else paths:
            url=f'https://raw.githubusercontent.com/{repo}/{commits[role]}/{name}'
            dest=folder/role/name;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(urllib.request.urlopen(url).read())
            receipts.append({'role':role,'repository':repo,'revision':commits[role],'path':name,'url':url,'sha256':sha(dest)})
    write(folder/'source_receipts.json',receipts)
    # Official assets contain rectified sensor profiles, not an EE-to-camera hand-eye transform.
    profiles={}
    for c in CAMERAS:
        raw=json.loads((folder/'calibration/static/files'/c).read_text())
        rows=[]
        for i in range(16):
            vals={key:float(raw.get(f'rectified.{i}.{key}',0)) for key in ['fx','fy','ppx','ppy','width','height']}
            if vals['width']>0 and vals['height']>0 and vals['fx']>1 and vals['fy']>1 and 0<=vals['ppx']<vals['width'] and 0<=vals['ppy']<vals['height']:
                sx,sy=256/vals['width'],256/vals['height']
                vals.update(profile_index=i,K_256_resize=[[vals['fx']*sx,0,(vals['ppx']+.5)*sx-.5],[0,vals['fy']*sy,(vals['ppy']+.5)*sy-.5],[0,0,1]],
                            mapping_status='CANDIDATE: requires RGB profile and actual crop/resize validation')
                rows.append(vals)
        profiles[c]=rows
    write(folder/'official_K_candidates.json',profiles)
    print('Pinned official calibration and capture source',commits,flush=True)

def inspect():
    folder=RUN/'selection';folder.mkdir(parents=True,exist_ok=True)
    rows=[];tiles=[]
    for path in sorted(DATA.glob('*.npy')):
        d=np.load(path,allow_pickle=True).item()
        primitives=np.asarray(d['primitive']).astype(str)
        candidates=np.flatnonzero(np.char.find(primitives,'insert')>=0)
        t0=126 if path.name=='1_M_L_3_vertical_n_2.npy' else int(candidates[0]+10) if len(candidates) else min(100,len(primitives)-21)
        t0=min(t0,len(primitives)-21)
        indices=np.arange(max(0,t0-49),t0+1)
        tcp=d['obs/tcp_pose'][indices]
        rotations=Rotation.from_quat(tcp[:,3:])
        rotvec=(rotations[0].inv()*rotations).as_rotvec()
        singular=np.linalg.svd(rotvec-rotvec.mean(0),compute_uv=False)
        views=[];qualities={}
        for camera in CAMERAS:
            rgb=d['obs/'+camera][t0][...,::-1].copy()
            z=d['obs/'+camera+'_depth'][indices]
            qualities[camera]={'valid_depth_fraction':float(np.mean(z>0)),'median_raw_depth':float(np.median(z[z>0])),
                              'laplacian_variance_t0':float(cv2.Laplacian(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())}
            tile=cv2.resize(rgb,(256,256));cv2.putText(tile,camera,(6,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            views.append(tile)
        tile=np.hstack(views)
        cv2.putText(tile,f'{path.stem} t0={t0}',(6,249),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        cv2.imwrite(str(folder/(path.stem+'.png')),tile[...,::-1]);tiles.append(tile)
        row={'source_file':path.name,'source_sha256':sha(path),'frame_count':len(primitives),'candidate_t0':t0,
             'observed_indices':indices.tolist(),'object_info':d['object_info'],'primitive_at_t0':primitives[t0],
             'tcp_translation_range_mm':(1000*np.ptp(tcp[:,:3],axis=0)).tolist(),'rotation_diversity_singular_values_rad':singular.tolist(),
             'max_rotation_deg':float(np.degrees(np.max(np.linalg.norm(rotvec,axis=1)))),'qualities':qualities,
             'future_pixel_accessed':False,'selection_future_poses_accessed':False}
        rows.append(row);print(path.name,'t0',t0,'rotation',row['max_rotation_deg'],'sv',singular,flush=True)
    write(folder/'candidates.json',rows)
    cv2.imwrite(str(folder/'all_candidates.png'),np.vstack(tiles)[...,::-1])

def export(source,t0):
    indices=np.arange(max(0,t0-49),t0+1)
    path=DATA/source;d=np.load(path,allow_pickle=True).item()
    for camera in CAMERAS:
        dest=RUN/camera/'observed';dest.mkdir(parents=True,exist_ok=True)
        for sub in ['geometry','viz','groups','predictions']:(RUN/camera/sub).mkdir(exist_ok=True)
        rgb=d['obs/'+camera][indices][...,::-1].copy()
        z=d['obs/'+camera+'_depth'][indices]
        np.save(dest/'rgb.npy',rgb);np.save(dest/'sensor_z16.npy',z);np.save(dest/'source_indices.npy',indices)
        np.save(dest/'tcp_pose_xyzw.npy',d['obs/tcp_pose'][indices]);np.save(dest/'timestamps.npy',indices/10)
        np.save(dest/'history_rgb.npy',rgb[-3:]);np.save(dest/'history_source_indices.npy',indices[-3:])
        for local in [0,len(indices)//2,len(indices)-3,len(indices)-2,len(indices)-1]:
            cv2.imwrite(str(dest/f'frame_{indices[local]:06d}.png'),rgb[local,...,::-1])
        import imageio.v2 as imageio
        video=dest/'observed_10hz.mp4'
        imageio.mimwrite(video,rgb,fps=10,codec='libx264rgb',pixelformat='rgb24',macro_block_size=None,ffmpeg_params=['-crf','0','-preset','fast'])
        decoded=np.stack(imageio.mimread(video));assert np.array_equal(decoded,rgb)
        write(RUN/camera/'metadata.json',{'source_path':str(path),'source_sha256':sha(path),'camera':camera,'t0':t0,
              'observed_source_indices':indices.tolist(),'history_source_indices':indices[-3:].tolist(),'future_source_indices':list(range(t0+1,t0+21)),
              'source_fps':10,'model_fps':15,'hardware_timestamps_available':False,'future_used_for_model_input':False,
              'color_transform':'official BGR -> RGB','video_pixels_exact':True,'object_info':d['object_info'],
              'instruction':'Insert the red rectangular peg into the matching hole on the blue board.',
              'target_object':'red rectangular peg',
              'depth_scale_m_per_raw_unit':.0001,'depth_scale_status':'SUPPORTED_HYPOTHESIS_NOT_RECORD_CONFIRMED'})
        print('Exported causal',camera,rgb.shape,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['sources','inspect','export']);p.add_argument('--source');p.add_argument('--t0',type=int)
    a=p.parse_args()
    if a.stage=='sources':sources()
    elif a.stage=='inspect':inspect()
    else:export(a.source,a.t0)

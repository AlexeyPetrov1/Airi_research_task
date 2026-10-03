"""Bounded provenance search for control n3 in the pinned ShareRobot trajectory subset."""
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import cv2
from fmb_v2_scene import ROOT,RUN,SPECS
from search_fmb_second_sharerobot import fetch
from berkeley_preprocess import write_json,sha256

def main():
    manifest=ROOT/'runs/fmb_second_scene/sharerobot_trajectory_manifest.json'
    rows=json.loads(manifest.read_text())
    paths=list(dict.fromkeys(row['image_path'] for row in rows if '57_fmb' in row.get('image_path','')))
    src=np.load(SPECS['fmb_control_n3']['source'],allow_pickle=True).item()
    candidates=[];labels=[]
    for camera in ['side_1','side_2','wrist_1','wrist_2']:
        for step,frame in enumerate(src[f'obs/{camera}']):
            small=cv2.resize(frame,(32,32),interpolation=cv2.INTER_AREA)
            candidates.extend([small,small[...,::-1]])
            labels.extend([(camera,step,'as_stored'),(camera,step,'BGR_to_RGB')])
    candidates=np.array(candidates,dtype=np.int16);results=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for item in pool.map(fetch,paths):
            im=item.pop('image',None)
            if im is not None:
                small=cv2.resize(im,(32,32),interpolation=cv2.INTER_AREA).astype(np.int16)
                errs=np.abs(candidates-small).mean((1,2,3));i=int(errs.argmin());camera,step,color=labels[i]
                frame=src[f'obs/{camera}'][step]
                if color=='BGR_to_RGB':frame=frame[...,::-1]
                item.update(best_camera=camera,best_source_step=step,color=color,
                            small_mae=float(errs[i]),full_mae=float(np.abs(im.astype(np.int16)-frame.astype(np.int16)).mean()))
            results.append(item)
    results.sort(key=lambda row:row.get('full_mae',float('inf')))
    write_json(RUN/'second_share_search.json',{'source':SPECS['fmb_control_n3']['source'].name,
        'source_sha256':sha256(SPECS['fmb_control_n3']['source']),'manifest_sha256':sha256(manifest),
        'scope':'Pinned ShareRobot 3266d929 trajectory subset; all referenced FMB images, every n3 frame, four cameras, both channel orders',
        'images_checked':len(results),'confirmed_mapping':False,
        'limitation':'A small MAE alone is not a proven mapping. Planning archive and all other ShareRobot episodes not exhausted.',
        'results':results})
    print(json.dumps({'images_checked':len(results),'best':results[:2]}),flush=True)

if __name__=='__main__':main()

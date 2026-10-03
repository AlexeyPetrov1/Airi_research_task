"""Check target sensor support before choosing a wrist hybrid scene."""
import json
import cv2,numpy as np
from fmb_wrist_prepare import ROOT,RUN,DATA,write
from fmb_wrist_math import sample_z
import berkeley_preprocess as b

def run():
    specs=json.loads((RUN/'selection/candidates.json').read_text());rows=[];panels=[]
    for spec in specs:
        d=np.load(DATA/spec['source_file'],allow_pickle=True).item();t0=spec['candidate_t0'];color=spec['object_info']['color'];tiles=[]
        for camera in ['wrist_1','wrist_2']:
            rgb=d['obs/'+camera][t0][...,::-1].copy();r,g,blue=rgb.transpose(2,0,1).astype(float)
            if color==4:mask=(r>100)&(g>70)&(blue<.65*r)&(g>.5*r)
            elif color==7:mask=(blue>r*1.25)&(blue>g*1.12)&(blue>65)
            else:mask=(r>g*1.35)&(r>blue*1.35)&(r>80)
            # Target is attached to the gripper in the upper/central wrist view; exclude blue board below.
            mask[200:]=False;mask[:,:70]=False
            n,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            if n>1:
                plausible=[i for i in range(1,n) if centers[i,1]<150 and centers[i,0]>80]
                if plausible:mask=labels==max(plausible,key=lambda i:stats[i,cv2.CC_STAT_AREA])
            depth=d['obs/'+camera+'_depth'][t0].astype(float)*1e-4;valid=(depth>0)&(depth<2)
            vv,uu=np.where(cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8))>0);uv=np.stack([uu,vv],axis=-1)
            zz,spread,coverage=sample_z(depth,uv)
            eligible=np.isfinite(zz)&(spread<.03)
            row={'source_file':spec['source_file'],'t0':t0,'camera':camera,'mask_method':'Observed color connected component; approximate selection audit, not model segmentation',
                'approximate_target_pixels':int(mask.sum()),'sensor_valid_target_fraction':float(valid[mask].mean()) if mask.any() else None,
                'interior_5x5_depth_supported_pixels':int(eligible.sum()),'interior_pixel_count':len(uv),
                'supported_patch_fraction':float(eligible.mean()) if len(eligible) else None,'future_used':False}
            rows.append(row)
            im=rgb.copy();im[mask&(~valid)]=(im[mask&(~valid)]*.4+np.array([0,255,0])*.6).astype(np.uint8)
            title=f'{camera} support={row["interior_5x5_depth_supported_pixels"]}/{len(uv)}'
            cv2.putText(im,title,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);tiles.append(im)
        tile=np.hstack(tiles);cv2.putText(tile,spec['source_file'],(4,248),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);panels.append(tile)
        print(spec['source_file'],[(r['camera'],r['sensor_valid_target_fraction'],r['interior_5x5_depth_supported_pixels']) for r in rows[-2:]],flush=True)
    write(RUN/'selection/target_depth_support.json',rows);b.save_rgb(RUN/'selection/target_depth_support.png',np.vstack(panels))

if __name__=='__main__':run()

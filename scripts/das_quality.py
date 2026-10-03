"""Descriptive temporal consistency and cup appearance measurements."""
import cv2
import numpy as np


def temporal_warp_error(frames):
    h,w=frames.shape[1:3]
    xx,yy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    values=[];fractions=[]
    for old,new in zip(frames[:-1],frames[1:]):
        old_gray=cv2.cvtColor(old,cv2.COLOR_RGB2GRAY)
        new_gray=cv2.cvtColor(new,cv2.COLOR_RGB2GRAY)
        backward=cv2.calcOpticalFlowFarneback(new_gray,old_gray,None,.5,3,15,3,5,1.2,0)
        forward=cv2.calcOpticalFlowFarneback(old_gray,new_gray,None,.5,3,15,3,5,1.2,0)
        mx,my=xx+backward[...,0],yy+backward[...,1]
        sampled_forward=cv2.remap(forward,mx,my,cv2.INTER_LINEAR)
        valid=(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)&(np.linalg.norm(backward+sampled_forward,axis=-1)<1.)
        warped=cv2.remap(old,mx,my,cv2.INTER_LINEAR)
        error=np.abs(new.astype(float)-warped.astype(float)).mean(-1)
        values.append(float(error[valid].mean()) if valid.any() else None)
        fractions.append(float(valid.mean()))
    return {'mean_warp_MAE_0_255':float(np.mean([v for v in values if v is not None])),
            'per_interval_warp_MAE_0_255':values,'mean_valid_fraction':float(np.mean(fractions)),
            'method':'Farneback backward RGB warp; forward-backward residual <1 px; in-bounds only',
            'interpretation':'Includes flow/occlusion errors; low error also rewards static video. Compare at matched times.'}


def cup_appearance(frames,tracks,visible,reference,mask):
    h,w=mask.shape
    ref_hsv=cv2.cvtColor(reference,cv2.COLOR_RGB2HSV)
    ref_hist=cv2.calcHist([ref_hsv],[0,1],mask.astype(np.uint8)*255,[30,32],[0,180,0,256])
    cv2.normalize(ref_hist,ref_hist)
    correlations=[];bounds=[];blue_fractions=[]
    y,x=np.nonzero(mask)
    initial=np.array([x.min(),y.min(),x.max()+1,y.max()+1])
    xy0=tracks[0].mean(0)
    for frame,xy,vis in zip(frames,tracks,visible):
        if vis.sum()<4:
            correlations.append(None);blue_fractions.append(None);bounds.append(None);continue
        shift=xy[vis].mean(0)-xy0
        x1,y1,x2,y2=np.rint(initial+np.tile(shift,2)).astype(int)
        x1,x2=np.clip([x1,x2],0,w);y1,y2=np.clip([y1,y2],0,h)
        if x2<=x1 or y2<=y1:
            correlations.append(None);blue_fractions.append(None);bounds.append(None);continue
        hsv=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_RGB2HSV)
        hist=cv2.calcHist([hsv],[0,1],None,[30,32],[0,180,0,256]);cv2.normalize(hist,hist)
        correlations.append(float(cv2.compareHist(ref_hist,hist,cv2.HISTCMP_CORREL)))
        blue_fractions.append(float(((hsv[...,0]>=85)&(hsv[...,0]<=125)&(hsv[...,1]>=40)).mean()))
        bounds.append([int(x1),int(y1),int(x2),int(y2)])
    return {'tracked_crop_HS_histogram_correlation_to_t0':correlations,
            'tracked_crop_blue_pixel_fraction':blue_fractions,'tracked_crop_bounds':bounds,
            'interpretation':'Translated initial mask bounding box; includes background/occlusion. Color correlation cannot certify cup shape, printed motif, or physical identity.'}

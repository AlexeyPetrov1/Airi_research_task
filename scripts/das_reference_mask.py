"""Keep gripper pixels without confusing dark blue cup pixels with the gripper."""
import cv2,numpy as np

def reference_alpha(reference,mask,protection='gray-only'):
    expanded=cv2.dilate(mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))>0
    yy=np.indices(mask.shape)[0]
    protected=(~mask)&(cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY)<85)&(yy<260)
    if protection=='non-blue-dark':
        hsv=cv2.cvtColor(reference,cv2.COLOR_RGB2HSV)
        blue=(hsv[...,0]>=85)&(hsv[...,0]<=125)&(hsv[...,1]>=40)
        protected&=~blue
    elif protection!='gray-only':raise ValueError(protection)
    expanded&=~protected
    distance=cv2.distanceTransform(expanded.astype(np.uint8),cv2.DIST_L2,5)
    alpha=np.clip(distance/4,0,1);alpha[mask]=1
    return alpha,expanded

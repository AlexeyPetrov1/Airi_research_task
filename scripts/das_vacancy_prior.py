"""Flow denoising prior for causally estimated vacated background pixels.

This is an explicit inference extension, not native DaS or a new checkpoint.
Start-frame latents and every predicted moving foreground support are protected.
"""
import cv2
import numpy as np


def make_latent_masks(alpha,moving,context_pixels=0):
    assert alpha.shape==(480,640) and moving.shape==(49,480,640)
    if context_pixels:
        alpha=cv2.dilate(alpha.astype(np.float32),cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,(2*context_pixels+1,)*2))
    masks=np.zeros((13,60,90),np.float32)
    # Wan latent 0 decodes frame 0; each later latent decodes the next four frames.
    for i in range(1,13):
        foreground=np.any(moving[4*i-3:4*i+1],axis=0).astype(np.uint8)
        protected=cv2.dilate(foreground,np.ones((11,11),np.uint8))>0
        allowed=alpha.copy();allowed[protected]=0
        # Max-pooling protection at the latent scale: a foreground-touching block
        # must never be overwritten by the vacancy prior.
        latent_foreground=cv2.resize(protected.astype(np.float32),(90,60),interpolation=cv2.INTER_AREA)>0
        mask=cv2.resize(allowed.astype(np.float32),(90,60),interpolation=cv2.INTER_AREA)
        mask[latent_foreground]=0
        masks[i]=mask
    return masks


class VacancyPrior:
    def __init__(self,masks,strength,seed,image_latents):
        self.masks=masks;self.strength=strength;self.seed=seed;self.image_latents=image_latents
        self.noise=None;self.applied_steps=0

    def __call__(self,pipeline,index,timestep,callback_kwargs):
        import torch
        latents=callback_kwargs['latents']
        # Encoding order is control-video, original start-image, clean ref-image.
        # Keep two 1-frame encode receipts to avoid accidentally anchoring on t0.
        assert len(self.image_latents)==2,'Expected original start and clean persistent reference encodings'
        empty=self.image_latents[-1].to(latents.device,latents.dtype).expand_as(latents)
        if self.noise is None:
            generator=torch.Generator(device=latents.device).manual_seed(self.seed)
            self.noise=torch.randn(latents.shape,generator=generator,device=latents.device,dtype=latents.dtype)
        sigma=pipeline.scheduler.sigmas[index+1].to(latents.device,latents.dtype)
        background=(1-sigma)*empty+sigma*self.noise
        mask=torch.as_tensor(self.masks,device=latents.device,dtype=latents.dtype)[None,None]*self.strength
        callback_kwargs['latents']=latents*(1-mask)+background*mask
        self.applied_steps+=1
        return callback_kwargs

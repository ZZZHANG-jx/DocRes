import cv2 
import numpy as np
import glob 
import os 
from tqdm import tqdm 
import random
import sys
sys.path.append('./data/MBD')
from MBD import mask_base_dewarper

def shadowExtract(cap_im, alb_im):
    im = cap_im
    alb = alb_im

    ## Avoid some bad cases
    skip = False
    im_min = np.min(im,axis=-1)
    kernel = np.ones((3,3))
    _, mask = cv2.threshold(cv2.cvtColor(alb,cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    mask_erode = cv2.dilate(mask,kernel=kernel)
    mask_erode = cv2.erode(mask_erode,kernel=kernel)
    mask_erode = cv2.erode(mask_erode,iterations=4,kernel=kernel)
    metric = np.min(im_min[mask_erode==255])
    metric_num = 0
    if metric==0 or metric==1:
        metric_num = np.sum(im_min[mask_erode==255]==metric)
        if metric_num>=20:
            skip = True
            pass
            # return None
            # it is recommended to skip this sample as it will introduce some artifacts.

    alb_temp = alb.astype(np.float64)
    alb_temp[alb_temp==0] = alb_temp[alb_temp==0]+1e-5
    shadow = np.clip(im.astype(np.float64)/alb_temp,0,1)
    shadow = (shadow*255).astype(np.uint8)
    return shadow,skip


cap_im = cv2.imread('./data/images/2.png')
alb_im = cv2.imread('./data/images/3.png')

## get mask by binarizing alb_im
_, mask = cv2.threshold(cv2.cvtColor(alb,cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
kernel = np.ones((3,3))
mask = cv2.dilate(mask,iterations=2,kernel=kernel)
mask = cv2.erode(mask,iterations=2,kernel=kernel)

## dewarp cap and alb based on the mask by using MBD method
cap_im, _ = mask_base_dewarper(cap_im, mask_im)
alb_im, _ = mask_base_dewarper(alb_im, mask_im)


shadow_im,skip = shadowExtract(cap_im,alb_im) # It is recommended to skip this sample if skip is True. Based on our observations, images that meet this condition often introduce noise.

cv2.imshow('shadow_im',shadow_im)
cv2.imshow('cap_im',cap_im)
cv2.imshow('alb_im',alb_im)
cv2.imshow('mask',mask)
cv2.waitKey(0)
import cv2 
import numpy as np
import glob 
import os 
from tqdm import tqdm 
import random

im_paths = glob.glob('./img/*/*')

random.shuffle(im_paths)

for im_path in tqdm(im_paths):
    # im_path = './img/1/23-180_5-y4_Page_034-wVO0001-L1_3-T_6600-I_5535.png'
    if '-L1_' in im_path:
        alb_path = im_path.split('-L1_')[0].replace('img/','alb/') + '.png'
    else:
        alb_path = im_path.split('-L2_')[0].replace('img/','alb/') + '.png'

    if not os.path.exists(alb_path):
        print(im_path)
        print(alb_path)

    im = cv2.imread(im_path)
    alb = cv2.imread(alb_path)
    _, mask = cv2.threshold(cv2.cvtColor(alb,cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)


    ## Avoid some bad cases
    im_min = np.min(im,axis=-1)
    kernel = np.ones((3,3))
    mask_erode = cv2.dilate(mask,kernel=kernel)
    mask_erode = cv2.erode(mask_erode,kernel=kernel)
    mask_erode = cv2.erode(mask_erode,iterations=4,kernel=kernel)
    metric = np.min(im_min[mask_erode==255])
    metric_num = 0
    if metric==0 or metric==1:
        metric_num = np.sum(im_min[mask_erode==255]==metric)
        if metric_num>=20:
            # alb_temp = alb.astype(np.float64)
            # alb_temp[alb_temp==0] = alb_temp[alb_temp==0]+1e-5
            # shadow = np.clip(im.astype(np.float64)/alb_temp,0,1)
            # shadow = (shadow*255).astype(np.uint8)
            # shadow_path = im_path.replace('img/','temp/')
            # cv2.imwrite(shadow_path,shadow)
            continue


    alb_temp = alb.astype(np.float64)
    alb_temp[alb_temp==0] = alb_temp[alb_temp==0]+1e-5
    shadow = np.clip(im.astype(np.float64)/alb_temp,0,1)
    shadow = (shadow*255).astype(np.uint8)

    shadow_path = im_path.replace('img/','shadow/')
    cv2.imwrite(shadow_path,shadow)

    mask_path = im_path.replace('img/','mask/')
    cv2.imwrite(mask_path,mask)

    # cv2.imshow('im',im)
    # cv2.imshow('alb',alb)
    # cv2.imshow('shadow',shadow)
    # cv2.imshow('mask_erode',mask_erode)
    # print(im_min[mask_erode==255])
    # print(metric,metric_num)
    # cv2.waitKey(0)

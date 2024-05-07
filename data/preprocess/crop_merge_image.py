import os

import cv2
import numpy as np
# SIZE =256
# BATCH_SIZE = 32
# STRIDES = 256

def split_img(img, size_x, size_y, strides):
    max_y, max_x = img.shape[:2]
    border_y = 0
    if max_y % size_y != 0:
        border_y = size_y - (max_y % size_y)
        img = cv2.copyMakeBorder(img,border_y,0,0,0,cv2.BORDER_REPLICATE)
        # img = cv2.copyMakeBorder(img, border_y, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
    border_x = 0
    if max_x % size_x != 0:
        border_x = size_x - (max_x % size_x)
        # img = cv2.copyMakeBorder(img, 0, 0, border_x, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
        img = cv2.copyMakeBorder(img,0,0,border_x,0,cv2.BORDER_REPLICATE)
    # h,w
    max_y, max_x = img.shape[:2]
    parts = []
    curr_y = 0
    x = 0
    y = 0
    # TODO: rewrite with generators.
    while (curr_y + size_y) <= max_y:
        curr_x = 0
        while (curr_x + size_x) <= max_x:
            parts.append(img[curr_y:curr_y + size_y, curr_x:curr_x + size_x])
            curr_x += strides
        y += 1
        curr_y += strides
        # parts is a list 
        # (windows_number_x*windows_number_y,SIZE,SIZE,3)
    # print(max_y,max_x)
    # print(y,x)
    # print(np.array(parts).shape)
    return parts, border_x, border_y, max_x, max_y


def combine_imgs(border_x,border_y,imgs, max_y, max_x,size_x, size_y, strides):
 
    # weighted_img

    index = int(size_x / strides) 
    weight_img = np.ones(shape=(max_y,max_x))
    weight_img[0:strides] = index
    weight_img[-strides:] = index
    weight_img[:,0:strides]=index
    weight_img[:,-strides:]=index

    # 边上
    i = 0
    for j in range(1,index+1):
        # 左上
        weight_img[0:strides,i:i+strides] = np.ones(shape=(strides,strides))*j
        weight_img[i:i+strides,0:strides] = np.ones(shape=(strides,strides))*j
        # 右上
        weight_img[i:i+strides,-strides:] = np.ones(shape=(strides,strides))*j
        if i == 0:
            weight_img[0:strides,-strides:] = np.ones(shape=(strides,strides))*j
        else:
            weight_img[0:strides,-strides-i:-i] = np.ones(shape=(strides,strides))*j
        # 左下
        weight_img[-strides:,i:i+strides] = np.ones(shape=(strides,strides))*j
        if i == 0:
            weight_img[-strides:,0:strides] = np.ones(shape=(strides,strides))*j
        else:
            weight_img[-strides-i:-i:,0:strides] = np.ones(shape=(strides,strides))*j   
        # 右下
        if i == 0:
            weight_img[-strides:,-strides:] = np.ones(shape=(strides,strides))*j
        else:
            weight_img[-strides-i:-i,-strides:] = np.ones(shape=(strides,strides))*j
            weight_img[-strides:,-strides-i:-i] = np.ones(shape=(strides,strides))*j


        i += strides

    for i in range(strides,max_y-strides,strides):
    	for j in range(strides,max_x-strides,strides):
    		weight_img[i:i+strides,j:j+strides] = np.ones(shape=(strides,strides))*weight_img[i][0]*weight_img[0][j]


    if len(imgs[0].shape)==2:
        new_img = np.zeros(shape=(max_y,max_x))
        weight_img = (1 / weight_img)
    else:
        new_img = np.zeros(shape=(max_y,max_x,imgs[0].shape[-1]))
        weight_img = (1 / weight_img).reshape((max_y,max_x,1))
        weight_img = np.tile(weight_img,(1,1,imgs[0].shape[-1]))

    curr_y = 0
    x = 0
    y = 0
    i = 0
        # TODO: rewrite with generators.
    while (curr_y + size_y) <= max_y:
        curr_x = 0
        while (curr_x + size_x) <= max_x:
            new_img[curr_y:curr_y + size_y, curr_x:curr_x + size_x] += weight_img[curr_y:curr_y + size_y, curr_x:curr_x + size_x]*imgs[i]
            i += 1
            curr_x += strides
        y += 1
        curr_y += strides


    new_img = new_img[border_y:, border_x:]
    # print(border_y,border_x)

    return new_img


def stride_integral(img,stride=32):
    h,w = img.shape[:2]

    if (h%stride)!=0:
        padding_h = stride - (h%stride)
        img = cv2.copyMakeBorder(img,padding_h,0,0,0,borderType=cv2.BORDER_REPLICATE)
    else:
        padding_h = 0
    
    if (w%stride)!=0:
        padding_w = stride - (w%stride)
        img = cv2.copyMakeBorder(img,0,0,padding_w,0,borderType=cv2.BORDER_REPLICATE)
    else:
        padding_w = 0
    
    return img,padding_h,padding_w


def mkdir_s(path: str):
    """Create directory in specified path, if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ =='__main__':
    parts, border_x, border_y, max_x, max_y = split_img(im,512,512,strides=512)
    result = combine_imgs(border_x,border_y,parts, max_y, max_x,512, 512, 512)

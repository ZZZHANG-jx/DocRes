import os
from os.path import join as pjoin
import collections
import json
from numpy.lib.histograms import histogram_bin_edges
import torch
import numpy as np
import cv2
import random
import torch.nn.functional as F
from torch.utils import data
import glob

class DocResTrainDataset(data.Dataset):
    def __init__(self, dataset={}, img_size=512,):
        json_paths = dataset['json_paths']
        self.task = dataset['task']
        self.size = img_size
        self.im_path = dataset['im_path']

        self.datas = []
        for json_path in json_paths:
            with open(json_path,'r') as f:
                data = json.load(f)
                self.datas += data

        self.background_paths = glob.glob('/data2/jiaxin/Training_Data/dewarping/doc_3d/background/*/*/*')
        self.shadow_paths = glob.glob('/data2/jiaxin/Training_Data/illumination/doc3dshadow/new_shadow/*/*')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        in_im,gt_im,dtsprompt = self.data_processing(self.task,data)

        return torch.cat((in_im,dtsprompt),0), gt_im

    def data_processing(self,task,data):
        
        if task=='deblurring':
            ## image prepare
            in_im = cv2.imread(os.path.join(self.im_path,data['in_path']))
            gt_im = cv2.imread(os.path.join(self.im_path,data['gt_path']))
            dtsprompt = self.deblur_dtsprompt(in_im)
            ## get prompt
            in_im, gt_im,dtsprompt = self.randomcrop([in_im,gt_im,dtsprompt])
            in_im  = self.rgbim_transform(in_im)
            gt_im  = self.rgbim_transform(gt_im)
            dtsprompt  = self.rgbim_transform(dtsprompt)
        elif task =='dewarping':
            ## image prepare
            in_im = cv2.imread(os.path.join(self.im_path,data['in_path']))
            mask = cv2.imread(os.path.join(self.im_path,data['mask_path']))[:,:,0]
            bm = np.load(os.path.join(self.im_path,data['gt_path'])).astype(np.float)  #-> 0-448
            bm = cv2.resize(bm,(448,448))
            ## add background 
            background = cv2.imread(random.choice(self.background_paths))
            min_length = min(background.shape[:2])
            crop_size = random.randint(int(min_length*0.5),min_length-1)
            shift_y = np.random.randint(0,background.shape[1]-crop_size)
            shift_x = np.random.randint(0,background.shape[0]-crop_size)
            background = background[shift_x:shift_x+crop_size,shift_y:shift_y+crop_size,:]
            background = cv2.resize(background,(448,448))
            if np.mean(in_im[mask==0])<10:
                in_im[mask==0]=background[mask==0]
            ## random crop and get prompt
            in_im,mask,bm = self.random_margin_bm(in_im,mask,bm) # bm-> 0-1
            in_im = cv2.resize(in_im,(self.size,self.size))
            mask = cv2.resize(mask,(self.size,self.size))
            mask_aug = self.mask_augment(mask)
            in_im[mask_aug==0]=0 
            bm = cv2.resize(bm,(self.size,self.size)) # bm-> 0-1
            bm_shift = (bm*self.size - self.getBasecoord(self.size,self.size))/self.size
            base_coord = self.getBasecoord(self.size,self.size)/self.size

            in_im = self.rgbim_transform(in_im)            
            base_coord = base_coord.transpose(2, 0, 1)
            base_coord = torch.from_numpy(base_coord)

            bm_shift = bm_shift.transpose(2, 0, 1)
            bm_shift = torch.from_numpy(bm_shift)

            mask[mask>155] = 255
            mask[mask<=155] = 0
            mask = mask/255
            mask = np.expand_dims(mask,-1)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(mask)    

            mask_aug[mask_aug>155] = 255
            mask_aug[mask_aug<=155] = 0
            mask_aug = mask_aug/255
            mask_aug = np.expand_dims(mask_aug,-1)
            mask_aug = mask_aug.transpose(2, 0, 1)
            mask_aug = torch.from_numpy(mask_aug)

            in_im = in_im
            gt_im = torch.cat((bm_shift,mask),0)
            dtsprompt = torch.cat((base_coord,mask_aug),0)

        elif task == 'binarization':
            ## image prepare
            in_im = cv2.imread(os.path.join(self.im_path,data['in_path']))
            gt_im = cv2.imread(os.path.join(self.im_path,data['gt_path']))
            ## get prompt
            thr = cv2.imread(os.path.join(self.im_path,data['thr_path']))        
            bin_map = cv2.imread(os.path.join(self.im_path,data['bin_path']))
            gradient = cv2.imread(os.path.join(self.im_path,data['gradient_path']))
            bin_map[bin_map>155]=255
            bin_map[bin_map<=155]=0
            in_im, gt_im,thr,bin_map,gradient = self.randomcrop([in_im,gt_im,thr,bin_map,gradient])
            in_im = self.randomAugment_binarization(in_im)
            gt_im[gt_im>155]=255
            gt_im[gt_im<=155]=0
            gt_im = gt_im[:,:,0]
            ## transform
            in_im  = self.rgbim_transform(in_im)
            thr  = self.rgbim_transform(thr)
            gradient  = self.rgbim_transform(gradient)
            bin_map  = self.rgbim_transform(bin_map)
            gt_im = gt_im.astype(np.float)/255. 
            gt_im = torch.from_numpy(gt_im)
            gt_im = gt_im.unsqueeze(0)
            dtsprompt = torch.cat((thr[0].unsqueeze(0),gradient[0].unsqueeze(0),bin_map[0].unsqueeze(0)),0)
        elif task == 'deshadowing':

            in_im = cv2.imread(os.path.join(self.im_path,data['in_path']))
            gt_im = cv2.imread(os.path.join(self.im_path,data['gt_path']))
            shadow_im = self.deshadow_dtsprompt(in_im)   
            if 'fsdsrd' in data['in_path']:
                in_im = cv2.resize(in_im,(512,512))
                gt_im = cv2.resize(gt_im,(512,512))
                shadow_im = cv2.resize(shadow_im,(512,512))
                in_im, gt_im,shadow_im = self.randomcrop([in_im,gt_im,shadow_im])
            else:
                in_im, gt_im,shadow_im = self.randomcrop([in_im,gt_im,shadow_im])
            in_im  = self.rgbim_transform(in_im)
            gt_im  = self.rgbim_transform(gt_im)
            shadow_im = self.rgbim_transform(shadow_im)
            dtsprompt = shadow_im

        elif task == 'appearance':
            if 'in_path' in data.keys():
                cap_im = cv2.imread(os.path.join(self.im_path,data['in_path']))
                gt_im = cv2.imread(os.path.join(self.im_path,data['gt_path']))
                gt_im,cap_im = self.randomcrop_realdae(gt_im,cap_im)
                cap_im = self.appearance_randomAugmentv1(cap_im)
                enhance_result = self.appearance_dtsprompt(cap_im)
            else:
                gt_im = cv2.imread(os.path.join(self.im_path,data['gt_path']))
                bleed_im = cv2.imread(os.path.join(self.im_path,random.choice(self.datas)['gt_path']))
                bleed_im = cv2.resize(bleed_im,gt_im.shape[:2][::-1])
                gt_im = self.randomcrop([gt_im])[0]
                bleed_im = self.randomcrop([bleed_im])[0]
                cap_im = self.bleed_trough(gt_im,bleed_im)

                shadow_path = random.choice(self.shadow_paths)
                shadow_im = cv2.imread(shadow_path)
                cap_im = self.appearance_randomAugmentv2(cap_im,shadow_im)
                enhance_result = self.appearance_dtsprompt(cap_im)


            in_im = self.rgbim_transform(cap_im)
            gt_im = self.rgbim_transform(gt_im)
            dtsprompt = self.rgbim_transform(enhance_result)

        return in_im, gt_im,dtsprompt

    def randomcrop(self,im_list):
        im_num = len(im_list)
        ## random scale rotate
        if random.uniform(0,1) <= 0.8:
            y,x = im_list[0].shape[:2]
            angle = random.uniform(-180,180)
            scale = random.uniform(0.7,1.5)
            M = cv2.getRotationMatrix2D((int(x/2),int(y/2)),angle,scale)
            for i in range(im_num):
                im_list[i] = cv2.warpAffine(im_list[i],M,(x,y),borderValue=(255,255,255))

        ## random crop
        crop_size = self.size
        for i in range(im_num):
            h,w = im_list[i].shape[:2]
            h = max(h,crop_size)
            w = max(w,crop_size)
            im_list[i] = cv2.resize(im_list[i],(w,h))
        
        if h==crop_size:
            shift_y=0
        else:
            shift_y = np.random.randint(0,h-crop_size)
        if w==crop_size:
            shift_x=0
        else:
            shift_x = np.random.randint(0,w-crop_size)
        for i in range(im_num):
            im_list[i] = im_list[i][shift_y:shift_y+crop_size,shift_x:shift_x+crop_size,:]
        return im_list            

    def deblur_dtsprompt(self,img):
        x = cv2.Sobel(img,cv2.CV_16S,1,0)  
        y = cv2.Sobel(img,cv2.CV_16S,0,1)  
        absX = cv2.convertScaleAbs(x)   # 转回uint8  
        absY = cv2.convertScaleAbs(y)  
        high_frequency = cv2.addWeighted(absX,0.5,absY,0.5,0)
        high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_BGR2GRAY)
        high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_GRAY2BGR)
        return high_frequency


    def appearance_dtsprompt(self,img):
        h,w = img.shape[:2]
        img = cv2.resize(img,(1024,1024))
        rgb_planes = cv2.split(img)
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)
        result_norm = cv2.merge(result_norm_planes)
        result_norm = cv2.resize(result_norm,(w,h))
        return result_norm


    def rgbim_transform(self,im):
        im = im.astype(np.float)/255. 
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im)
        return im


    def random_margin_bm(self,in_im,msk,bm):
        size = in_im.shape[:2]
        [y, x] = (msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)

        s = 20
        s = int(20*size[0]/128)
        difference = int(5*size[0]/128)
        cx1 = random.randint(0, s - difference)
        cx2 = random.randint(0, s - difference) + 1
        cy1 = random.randint(0, s - difference)
        cy2 = random.randint(0, s - difference) + 1

        t = miny-s+cy1
        b = size[0]-maxy-s+cy2
        l = minx-s+cx1
        r = size[1]-maxx-s+cx2

        t = max(0,t)
        b = max(0,b)
        l = max(0,l)
        r = max(0,r)

        in_im = in_im[t:size[0]-b,l:size[1]-r]
        msk = msk[t:size[0]-b,l:size[1]-r]
        bm[:,:,1]=bm[:,:,1]-t
        bm[:,:,0]=bm[:,:,0]-l
        bm=bm/np.array([448-l-r, 448-t-b])

        return in_im,msk,bm

    def mask_augment(self,mask):
        if random.uniform(0,1) <= 0.6:
            if random.uniform(0,1) <= 0.5:
                mask = cv2.resize(mask,(64,64))
            else:
                mask = cv2.resize(mask,(128,128))
            mask = cv2.resize(mask,(256,256))
        mask[mask>155] = 255
        mask[mask<=155] = 0
        return mask

    def bleed_trough(self, in_im, bleed_im):
        if random.uniform(0,1) <= 0.5:
            if random.uniform(0,1) <= 0.8:
                ksize = np.random.randint(1,2)*2 + 1
                bleed_im = cv2.blur(bleed_im,(ksize,ksize))
            bleed_im = cv2.flip(bleed_im,1)
            alpha = random.uniform(0.75,1)
            in_im = cv2.addWeighted(in_im,alpha,bleed_im,1-alpha,0)
        return in_im

    def getBasecoord(self,h,w):
        base_coord0 = np.tile(np.arange(h).reshape(h,1),(1,w)).astype(np.float32)
        base_coord1 = np.tile(np.arange(w).reshape(1,w),(h,1)).astype(np.float32)
        base_coord = np.concatenate((np.expand_dims(base_coord1,-1),np.expand_dims(base_coord0,-1)),-1)
        return base_coord


    def randomcrop_realdae(self,gt_im,cap_im):
        if random.uniform(0,1) <= 0.5:
            y,x = gt_im.shape[:2]
            angle = random.uniform(-30,30)
            scale = random.uniform(0.8,1.5)
            M = cv2.getRotationMatrix2D((int(x/2),int(y/2)),angle,scale)
            gt_im = cv2.warpAffine(gt_im,M,(x,y),borderValue=(255,255,255))
            cap_im = cv2.warpAffine(cap_im,M,(x,y),borderValue=(255,255,255))
        crop_size = self.size
        if gt_im.shape[0] <= crop_size:
            gt_im = cv2.copyMakeBorder(gt_im,crop_size-gt_im.shape[0]+1,0,0,0,borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
            cap_im = cv2.copyMakeBorder(cap_im,crop_size-cap_im.shape[0]+1,0,0,0,borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
        if gt_im.shape[1] <= crop_size:
            gt_im = cv2.copyMakeBorder(gt_im,0,0,crop_size-gt_im.shape[1]+1,0,borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
            cap_im = cv2.copyMakeBorder(cap_im,0,0,crop_size-cap_im.shape[1]+1,0,borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
        shift_y = np.random.randint(0,gt_im.shape[1]-crop_size)
        shift_x = np.random.randint(0,gt_im.shape[0]-crop_size)
        gt_im = gt_im[shift_x:shift_x+crop_size,shift_y:shift_y+crop_size,:]
        cap_im = cap_im[shift_x:shift_x+crop_size,shift_y:shift_y+crop_size,:]
        return gt_im,cap_im


    def randomAugment_binarization(self,in_img):
        h,w = in_img.shape[:2]
        ## brightness
        if random.uniform(0,1) <= 0.5:
            high = 1.3
            low = 0.8
            ratio = np.random.uniform(low,high)
            in_img = in_img.astype(np.float64)*ratio
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## contrast
        if random.uniform(0,1) <= 0.5:
            high = 1.3
            low = 0.8
            ratio = np.random.uniform(low,high)
            gray = cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)
            mean_array = np.ones_like(in_img).astype(np.float64)*mean
            in_img = in_img.astype(np.float64)*ratio + mean_array*(1-ratio)
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## color
        if random.uniform(0,1) <= 0.5:
            high = 0.2
            low = 0.1
            ratio = np.random.uniform(0.1,0.3)
            random_color = np.random.randint(50,200,3).reshape(1,1,3)
            random_color = (random_color*ratio).astype(np.uint8)
            random_color = np.tile(random_color,(self.size,self.size,1))
            in_img = in_img.astype(np.float64)*(1-ratio) + random_color
            in_img = np.clip(in_img,0,255).astype(np.uint8)        
        return in_img


    def deshadow_dtsprompt(self,img):
        h,w = img.shape[:2]
        img = cv2.resize(img,(1024,1024))
        rgb_planes = cv2.split(img)
        result_planes = []
        result_norm_planes = []
        bg_imgs = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            bg_imgs.append(bg_img)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)
        result_norm = cv2.merge(result_norm_planes)
        bg_imgs = cv2.merge(bg_imgs)
        bg_imgs = cv2.resize(bg_imgs,(w,h))
        return bg_imgs









    def randomAugment(self,in_img,gt_img,shadow_img):
        h,w = in_img.shape[:2]
        # random crop
        crop_size = random.randint(128,1024)
        if shadow_img.shape[0] <= crop_size:
            shadow_img = cv2.copyMakeBorder(shadow_img,crop_size-shadow_img.shape[0]+1,0,0,0,borderType=cv2.BORDER_CONSTANT,value=(128,128,128))
        if shadow_img.shape[1] <= crop_size:
            shadow_img = cv2.copyMakeBorder(shadow_img,0,0,crop_size-shadow_img.shape[1]+1,0,borderType=cv2.BORDER_CONSTANT,value=(128,128,128))

        shift_y = np.random.randint(0,shadow_img.shape[1]-crop_size)
        shift_x = np.random.randint(0,shadow_img.shape[0]-crop_size)
        shadow_img = shadow_img[shift_x:shift_x+crop_size,shift_y:shift_y+crop_size,:]
        shadow_img = cv2.resize(shadow_img,(w,h))
        in_img = in_img.astype(np.float64)*(shadow_img.astype(np.float64)+1)/255
        in_img = np.clip(in_img,0,255).astype(np.uint8)

        ## brightness
        if random.uniform(0,1) <= 0.5:
            high = 1.3
            low = 0.8
            ratio = np.random.uniform(low,high)
            in_img = in_img.astype(np.float64)*ratio
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## contrast
        if random.uniform(0,1) <= 0.5:
            high = 1.3
            low = 0.8
            ratio = np.random.uniform(low,high)
            gray = cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)
            mean_array = np.ones_like(in_img).astype(np.float64)*mean
            in_img = in_img.astype(np.float64)*ratio + mean_array*(1-ratio)
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## color
        if random.uniform(0,1) <= 0.5:
            high = 0.2
            low = 0.1
            ratio = np.random.uniform(0.1,0.3)
            random_color = np.random.randint(50,200,3).reshape(1,1,3)
            random_color = (random_color*ratio).astype(np.uint8)
            random_color = np.tile(random_color,(self.img_size[0],self.img_size[1],1))
            in_img = in_img.astype(np.float64)*(1-ratio) + random_color
            in_img = np.clip(in_img,0,255).astype(np.uint8)        
        ## scale and rotate
        if random.uniform(0,1) <= 0:
            y,x = self.img_size
            angle = random.uniform(-180,180)
            scale = random.uniform(0.5,1.5)
            M = cv2.getRotationMatrix2D((int(x/2),int(y/2)),angle,scale)
            in_img = cv2.warpAffine(in_img,M,(x,y),borderValue=0)
            gt_img = cv2.warpAffine(gt_img,M,(x,y),borderValue=0)
        # add noise
        ## jpegcompression
        quanlity_high = 95
        quanlity_low = 45
        quanlity = int(np.random.randint(quanlity_low,quanlity_high))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),quanlity]
        result, encimg = cv2.imencode('.jpg',in_img,encode_param)
        in_img = cv2.imdecode(encimg,1).astype(np.uint8)
        ## gaussiannoise
        mean = 0
        sigma = 0.02
        noise_ratio = 0.004
        num_noise = int(np.ceil(noise_ratio*w))
        coords = [np.random.randint(0,i-1,int(num_noise)) for i in [h,w]] 
        gauss = np.random.normal(mean,sigma,num_noise*3)*255
        guass = np.reshape(gauss,(-1,3))
        in_img = in_img.astype(np.float64)
        in_img[tuple(coords)] += guass
        in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## blur
        ksize = np.random.randint(1,2)*2 + 1
        in_img = cv2.blur(in_img,(ksize,ksize))

        ## erase
        if random.uniform(0,1) <= 0.7:
            for i in range(100):
                area = int(np.random.uniform(0.01,0.05)*h*w)
                ration = np.random.uniform(0.3,1/0.3)
                h_shift = int(np.sqrt(area*ration))
                w_shift = int(np.sqrt(area/ration))
                if (h_shift<h) and (w_shift<w):
                    break
            h_start = np.random.randint(0,h-h_shift)
            w_start = np.random.randint(0,w-w_shift)
            randm_area = np.random.randint(low=0,high=255,size=(h_shift,w_shift,3))
            in_img[h_start:h_start+h_shift,w_start:w_start+w_shift,:] = randm_area


        return in_img, gt_img


    def appearance_randomAugmentv1(self,in_img):

        ## brightness
        if random.uniform(0,1) <= 0.8:
            high = 1.3
            low = 0.5
            ratio = np.random.uniform(low,high)
            in_img = in_img.astype(np.float64)*ratio
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## contrast
        if random.uniform(0,1) <= 0.8:
            high = 1.3
            low = 0.5
            ratio = np.random.uniform(low,high)
            gray = cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)
            mean_array = np.ones_like(in_img).astype(np.float64)*mean
            in_img = in_img.astype(np.float64)*ratio + mean_array*(1-ratio)
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## color
        if random.uniform(0,1) <= 0.8:
            high = 0.2
            low = 0.1
            ratio = np.random.uniform(0.1,0.3)
            random_color = np.random.randint(50,200,3).reshape(1,1,3)
            random_color = (random_color*ratio).astype(np.uint8)
            random_color = np.tile(random_color,(self.size,self.size,1))
            in_img = in_img.astype(np.float64)*(1-ratio) + random_color
            in_img = np.clip(in_img,0,255).astype(np.uint8)        

        return in_img


    def appearance_randomAugmentv2(self,in_img,shadow_img):
        h,w = in_img.shape[:2]
        # random crop
        crop_size = random.randint(96,1024)
        if shadow_img.shape[0] <= crop_size:
            shadow_img = cv2.resize(shadow_img,(crop_size+1,crop_size+1))
        if shadow_img.shape[1] <= crop_size:
            shadow_img = cv2.resize(shadow_img,(crop_size+1,crop_size+1))

        shift_y = np.random.randint(0,shadow_img.shape[1]-crop_size)
        shift_x = np.random.randint(0,shadow_img.shape[0]-crop_size)
        shadow_img = shadow_img[shift_x:shift_x+crop_size,shift_y:shift_y+crop_size,:]
        shadow_img = cv2.resize(shadow_img,(w,h))
        in_img = in_img.astype(np.float64)*(shadow_img.astype(np.float64)+1)/255
        in_img = np.clip(in_img,0,255).astype(np.uint8)

        ## brightness
        if random.uniform(0,1) <= 0.8:
            high = 1.3
            low = 0.5
            ratio = np.random.uniform(low,high)
            in_img = in_img.astype(np.float64)*ratio
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## contrast
        if random.uniform(0,1) <= 0.8:
            high = 1.3
            low = 0.5
            ratio = np.random.uniform(low,high)
            gray = cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)
            mean_array = np.ones_like(in_img).astype(np.float64)*mean
            in_img = in_img.astype(np.float64)*ratio + mean_array*(1-ratio)
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## color
        if random.uniform(0,1) <= 0.8:
            high = 0.2
            low = 0.1
            ratio = np.random.uniform(0.1,0.3)
            random_color = np.random.randint(50,200,3).reshape(1,1,3)
            random_color = (random_color*ratio).astype(np.uint8)
            random_color = np.tile(random_color,(h,w,1))
            in_img = in_img.astype(np.float64)*(1-ratio) + random_color
            in_img = np.clip(in_img,0,255).astype(np.uint8)        

        if random.uniform(0,1) <= 0.8:
            quanlity_high = 95
            quanlity_low = 45
            quanlity = int(np.random.randint(quanlity_low,quanlity_high))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),quanlity]
            result, encimg = cv2.imencode('.jpg',in_img,encode_param)
            in_img = cv2.imdecode(encimg,1).astype(np.uint8)

        return in_img
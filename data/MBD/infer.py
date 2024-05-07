import torch
import argparse
import numpy as np
import torch.nn.functional as F
import glob
import cv2
from tqdm import tqdm

import time
import os 
from model.deep_lab_model.deeplab import *
from MBD import mask_base_dewarper
import time

from utils import cvimg2torch,torch2cvimg



def net1_net2_infer(model,img_paths,args):

    ### validate on the real datasets
    seg_model=model
    seg_model.eval()
    for img_path in tqdm(img_paths):
        if os.path.exists(img_path.replace('_origin','_capture')):
            continue
        t1 = time.time()
        ### segmentation mask predict
        img_org = cv2.imread(img_path)
        h_org,w_org = img_org.shape[:2]
        img = cv2.resize(img_org,(448, 448))       
        img = cv2.GaussianBlur(img,(15,15),0,0)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cvimg2torch(img)

        with torch.no_grad():
            pred = seg_model(img.cuda())
            mask_pred = pred[:,0,:,:].unsqueeze(1)
            mask_pred = F.interpolate(mask_pred,(h_org,w_org))
            mask_pred = mask_pred.squeeze(0).squeeze(0).cpu().numpy()
            mask_pred = (mask_pred*255).astype(np.uint8)
            kernel = np.ones((3,3))
            mask_pred = cv2.dilate(mask_pred,kernel,iterations=3)
            mask_pred = cv2.erode(mask_pred,kernel,iterations=3)
            mask_pred[mask_pred>100] = 255
            mask_pred[mask_pred<100] = 0
            ### tps transform base on the mask
            # dewarp, grid = mask_base_dewarper(img_org,mask_pred)
        try:
            dewarp, grid = mask_base_dewarper(img_org,mask_pred)
        except:
            print('fail')
            grid = np.meshgrid(np.arange(w_org),np.arange(h_org))/np.array([w_org,h_org]).reshape(2,1,1)
            grid = torch.from_numpy((grid-0.5)*2).float().unsqueeze(0).permute(0,2,3,1)
            dewarp = torch2cvimg(F.grid_sample(cvimg2torch(img_org),grid))[0]
            grid = grid[0].numpy()
        # cv2.imshow('in',cv2.resize(img_org,(512,512)))
        # cv2.imshow('out',cv2.resize(dewarp,(512,512)))
        # cv2.waitKey(0)
        cv2.imwrite(img_path.replace('_origin','_capture'),dewarp)
        cv2.imwrite(img_path.replace('_origin','_mask_new'),mask_pred)

        grid0 = cv2.resize(grid[:,:,0],(128,128))
        grid1 = cv2.resize(grid[:,:,1],(128,128))
        grid = np.stack((grid0,grid1),axis=-1)
        np.save(img_path.replace('_origin','_grid1'),grid)


def net1_net2_infer_single_im(img,model_path):
    seg_model = DeepLab(num_classes=1,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False)
    seg_model = torch.nn.DataParallel(seg_model, device_ids=range(torch.cuda.device_count()))
    seg_model.cuda()
    checkpoint = torch.load(model_path)
    seg_model.load_state_dict(checkpoint['model_state'])
    ### validate on the real datasets
    seg_model.eval()
    ### segmentation mask predict
    img_org = img
    h_org,w_org = img_org.shape[:2]
    img = cv2.resize(img_org,(448, 448))       
    img = cv2.GaussianBlur(img,(15,15),0,0)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cvimg2torch(img)

    with torch.no_grad():
        # from torchtoolbox.tools import summary
        # print(summary(seg_model,torch.rand((1, 3, 448, 448)).cuda())) 59.4M 135.6G

        pred = seg_model(img.cuda())
        mask_pred = pred[:,0,:,:].unsqueeze(1)
        mask_pred = F.interpolate(mask_pred,(h_org,w_org))
        mask_pred = mask_pred.squeeze(0).squeeze(0).cpu().numpy()
        mask_pred = (mask_pred*255).astype(np.uint8)
        kernel = np.ones((3,3))
        mask_pred = cv2.dilate(mask_pred,kernel,iterations=3)
        mask_pred = cv2.erode(mask_pred,kernel,iterations=3)
        mask_pred[mask_pred>100] = 255
        mask_pred[mask_pred<100] = 0
        ### tps transform base on the mask
        # dewarp, grid = mask_base_dewarper(img_org,mask_pred)
    # try:
    #     dewarp, grid = mask_base_dewarper(img_org,mask_pred)
    # except:
    #     print('fail')
    #     grid = np.meshgrid(np.arange(w_org),np.arange(h_org))/np.array([w_org,h_org]).reshape(2,1,1)
    #     grid = torch.from_numpy((grid-0.5)*2).float().unsqueeze(0).permute(0,2,3,1)
    #     dewarp = torch2cvimg(F.grid_sample(cvimg2torch(img_org),grid))[0]
    #     grid = grid[0].numpy()
    # cv2.imshow('in',cv2.resize(img_org,(512,512)))
    # cv2.imshow('out',cv2.resize(dewarp,(512,512)))
    # cv2.waitKey(0)
    # cv2.imwrite(img_path.replace('_origin','_capture'),dewarp)
    # cv2.imwrite(img_path.replace('_origin','_mask_new'),mask_pred)

    # grid0 = cv2.resize(grid[:,:,0],(128,128))
    # grid1 = cv2.resize(grid[:,:,1],(128,128))
    # grid = np.stack((grid0,grid1),axis=-1)
    # np.save(img_path.replace('_origin','_grid1'),grid)
    return mask_pred



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--img_folder', nargs='?', type=str, default='./all_data',help='Data path to load data')
    parser.add_argument('--img_rows', nargs='?', type=int, default=448, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=448, 
                        help='Width of the input image')
    parser.add_argument('--seg_model_path', nargs='?', type=str, default='checkpoints/mbd.pkl',
                        help='Path to previous saved model to restart from')
    args = parser.parse_args()

    seg_model = DeepLab(num_classes=1,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False)
    seg_model = torch.nn.DataParallel(seg_model, device_ids=range(torch.cuda.device_count()))
    seg_model.cuda()
    checkpoint = torch.load(args.seg_model_path)
    seg_model.load_state_dict(checkpoint['model_state'])

    im_paths = glob.glob(os.path.join(args.img_folder,'*_origin.*'))

    net1_net2_infer(seg_model,im_paths,args)


import cv2
import numpy as np
import MBD_utils
import torch
import torch.nn.functional as F


def mask_base_dewarper(image,mask):
    '''
    input:
        image -> ndarray HxWx3 uint8
        mask -> ndarray HxW uint8
    return
        dewarped -> ndarray HxWx3 uint8
        grid (optional) -> ndarray HxWx2 -1~1
    '''

    ## get contours
    # _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  ## cv2.__version__ == 3.x
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)  ## cv2.__version__ == 4.x

    ## get biggest contours and four corners based on Douglas-Peucker algorithm
    four_corners, maxArea, contour= MBD_utils.DP_algorithm(contours)
    four_corners = MBD_utils.reorder(four_corners)

    ## reserve biggest contours and remove other noisy contours
    new_mask = np.zeros_like(mask)
    new_mask = cv2.drawContours(new_mask,[contour],-1,255,cv2.FILLED)

    ## obtain middle points
    # ratios = [0.25,0.5,0.75]  # ratios = [0.125,0.25,0.375,0.5,0.625,0.75,0.875]
    ratios = [0.25,0.5,0.75]
    # ratios = [0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4475,0.5,0.5625,0.625,0.06875,0.75,0.8125,0.875,0.9375]
    middle = MBD_utils.findMiddle(corners=four_corners,mask=new_mask,points=ratios)
    
    ## all points
    source_points = np.concatenate((four_corners,middle),axis=0) ## all_point = four_corners(topleft,topright,bottom)+top+bottom+left+right

    ## target points
    h,w = image.shape[:2]
    padding = 0
    target_points = [[padding, padding],[w-padding, padding], [padding, h-padding],[w-padding, h-padding]]
    for ratio in ratios:
        target_points.append([int((w-2*padding)*ratio)+padding,padding])
    for ratio in ratios:
        target_points.append([int((w-2*padding)*ratio)+padding,h-padding])
    for ratio in ratios:
        target_points.append([padding,int((h-2*padding)*ratio)+padding])
    for ratio in ratios:
        target_points.append([w-padding,int((h-2*padding)*ratio)+padding])

    ## dewarp base on cv2
    # pts1 = np.float32(source_points)
    # pts2 = np.float32(target_points)
    # tps = cv2.createThinPlateSplineShapeTransformer()
    # matches = []
    # N = pts1.shape[0]
    # for i in range(0,N):
    #     matches.append(cv2.DMatch(i,i,0))
    # pts1 = pts1.reshape(1,-1,2)
    # pts2 = pts2.reshape(1,-1,2)
    # tps.estimateTransformation(pts2,pts1,matches)
    # dewarped = tps.warpImage(image)

    ## dewarp base on generated grid
    source_points = source_points.reshape(-1,2)/np.array([image.shape[:2][::-1]]).reshape(1,2)
    source_points = torch.from_numpy(source_points).float().cuda()
    source_points = source_points.unsqueeze(0)
    source_points = (source_points-0.5)*2
    target_points = np.asarray(target_points).reshape(-1,2)/np.array([image.shape[:2][::-1]]).reshape(1,2)
    target_points = torch.from_numpy(target_points).float()
    target_points = (target_points-0.5)*2

    model = MBD_utils.TPSGridGen(target_height=256,target_width=256,target_control_points=target_points)
    model = model.cuda()
    grid = model(source_points).view(-1,256,256,2).permute(0,3,1,2)
    grid = F.interpolate(grid,(h,w),mode='bilinear').permute(0,2,3,1)
    dewarped = MBD_utils.torch2cvimg(F.grid_sample(MBD_utils.cvimg2torch(image).cuda(),grid))[0]
    return dewarped,grid[0].cpu().numpy()

def mask_base_cropper(image,mask):
    '''
    input:
        image -> ndarray HxWx3 uint8
        mask -> ndarray HxW uint8
    return
        dewarped -> ndarray HxWx3 uint8
        grid (optional) -> ndarray HxWx2 -1~1
    '''

    ## get contours
    _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  ## cv2.__version__ == 3.x
    # contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)  ## cv2.__version__ == 4.x

    ## get biggest contours and four corners based on Douglas-Peucker algorithm
    four_corners, maxArea, contour= MBD_utils.DP_algorithm(contours)
    four_corners = MBD_utils.reorder(four_corners)

    ## reserve biggest contours and remove other noisy contours
    new_mask = np.zeros_like(mask)
    new_mask = cv2.drawContours(new_mask,[contour],-1,255,cv2.FILLED)

    ## 最小外接矩形
    rect = cv2.minAreaRect(contour) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点坐标
    box = np.int0(box)
    box = box.reshape((4,1,2))    




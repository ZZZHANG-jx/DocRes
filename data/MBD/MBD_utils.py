import cv2
import numpy as np
import copy
import torch
import torch
import itertools
import torch.nn as nn
from torch.autograd import Function, Variable

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def findMiddle(corners,mask,points=[0.25,0.5,0.75]):
    num_middle_points = len(points)
    top = [np.array([])]*num_middle_points
    bottom = [np.array([])]*num_middle_points
    left = [np.array([])]*num_middle_points
    right = [np.array([])]*num_middle_points

    center_top = []
    center_bottom = []
    center_left = []
    center_right = []

    center = (int((corners[0][0][1]+corners[3][0][1])/2),int((corners[0][0][0]+corners[3][0][0])/2))
    for ratio in points:

        center_top.append( (center[0],int(corners[0][0][0]*(1-ratio)+corners[1][0][0]*ratio)) )

        center_bottom.append( (center[0],int(corners[2][0][0]*(1-ratio)+corners[3][0][0]*ratio)) )

        center_left.append( (int(corners[0][0][1]*(1-ratio)+corners[2][0][1]*ratio),center[1]) )

        center_right.append( (int(corners[1][0][1]*(1-ratio)+corners[3][0][1]*ratio),center[1]) )

    for i in range(0,center[0],1):
        for j in range(num_middle_points):
            if top[j].size==0:
                if mask[i,center_top[j][1]]==255:
                    top[j] = np.asarray([center_top[j][1],i])
                    top[j] = top[j].reshape(1,2)

    for i in range(mask.shape[0]-1,center[0],-1):
        for j in range(num_middle_points):
            if bottom[j].size==0:
                if mask[i,center_bottom[j][1]]==255:
                    bottom[j] = np.asarray([center_bottom[j][1],i])
                    bottom[j] = bottom[j].reshape(1,2)

    for i in range(mask.shape[1]-1,center[1],-1):
        for j in range(num_middle_points):
            if right[j].size==0:
                if mask[center_right[j][0],i]==255:
                    right[j] = np.asarray([i,center_right[j][0]])
                    right[j] = right[j].reshape(1,2)

    for i in range(0,center[1]):
        for j in range(num_middle_points):
            if left[j].size==0:
                if mask[center_left[j][0],i]==255:
                    left[j] = np.asarray([i,center_left[j][0]])
                    left[j] = left[j].reshape(1,2) 

    return np.asarray(top+bottom+left+right)

def DP_algorithmv1(contours):
    biggest = np.array([])
    max_area = 0
    step = 0.001
    count = 0
    # while biggest.size==0:
    while True:
        for i in contours:
            # print(i.shape)
            area = cv2.contourArea(i)
            # print(area,cv2.arcLength(i, True))
            if area > cv2.arcLength(i, True)*10:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, (0.01+step*count) * peri, True)
                if area > max_area and len(approx) == 4:
                    max_area = area
                    biggest_contours = i
                    biggest = approx
                    break
                    if abs(max_area - cv2.contourArea(biggest))/max_area > 0.3:
                        biggest = np.array([])
        count += 1
        if count > 200:
            break
    temp = biggest[0]
    return biggest,max_area, biggest_contours

def DP_algorithm(contours):
    biggest = np.array([])
    max_area = 0
    step = 0.001
    count = 0

    ### largest contours
    for i in contours:
        area = cv2.contourArea(i)
        if area > max_area:
            max_area = area
            biggest_contours = i
    peri = cv2.arcLength(biggest_contours, True)

    ### find four corners
    while True:
        approx = cv2.approxPolyDP(biggest_contours, (0.01+step*count) * peri, True)
        if len(approx) == 4:
            biggest = approx
            break
            # if abs(max_area - cv2.contourArea(biggest))/max_area > 0.2:
            # if abs(max_area - cv2.contourArea(biggest))/max_area > 0.4:
                # biggest = np.array([])
        count += 1
        if count > 200:
            break
    return biggest,max_area, biggest_contours

def drawRectangle(img,biggest,color,thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), color, thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), color, thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), color, thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), color, thickness)
    return img

def minAreaRect(contours,img):
    # biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > max_area:
            peri = cv2.arcLength(i, True)
            rect = cv2.minAreaRect(i)
            points = cv2.boxPoints(rect)
            max_area = area
    return points

def cropRectangle(img,biggest):
    # print(biggest)
    w = np.abs(biggest[0][0][0] - biggest[1][0][0])
    h = np.abs(biggest[0][0][1] - biggest[2][0][1])
    new_img = np.zeros((w,h,img.shape[-1]),dtype=np.uint8)
    new_img = img[biggest[0][0][1]:biggest[0][0][1]+h,biggest[0][0][0]:biggest[0][0][0]+w]
    return new_img

def cvimg2torch(img,min=0,max=1):
    '''
    input:
        im -> ndarray uint8 HxWxC 
    return
        tensor -> torch.tensor BxCxHxW 
    '''
    if len(img.shape)==2:
        img = np.expand_dims(img,axis=-1)
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    return img

def torch2cvimg(tensor,min=0,max=1):
    '''
    input:
        tensor -> torch.tensor BxCxHxW C can be 1,3
    return
        im -> ndarray uint8 HxWxC 
    '''
    im_list = []
    for i in range(tensor.shape[0]):
        im = tensor.detach().cpu().data.numpy()[i]
        im = im.transpose(1,2,0)
        im = np.clip(im,min,max)
        im = ((im-min)/(max-min)*255).astype(np.uint8)
        im_list.append(im)
    return im_list



class TPSGridGen(nn.Module):
    def __init__(self, target_height, target_width, target_control_points):
        '''
        target_control_points -> torch.tensor  num_pointx2 -1~1
        source_control_points -> torch.tensor  batch_size x num_point x 2 -1~1
        return:
            grid -> batch_size x hw x 2 -1~1
        '''
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = self.compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate) # HW x 2
        Y, X = target_coordinate.split(1, dim = 1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = self.compute_partial_repr(target_coordinate.to(target_control_points.device), target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim = 1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate
    # phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
    def compute_partial_repr(self, input_points, control_points):
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
        # original implementation, very slow
        # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
        # fix numerical error for 0 * log(0), substitute all nan with 0
        mask = repr_matrix != repr_matrix
        repr_matrix.masked_fill_(mask, 0)
        return repr_matrix




 
    ### deside wheather further process
    # point_area = cv2.contourArea(np.concatenate((biggest_angle[0].reshape(1,1,2),middle[0:3],biggest_angle[1].reshape(1,1,2),middle[9:12],biggest_angle[3].reshape(1,1,2),middle[3:6][::-1],biggest_angle[2].reshape(1,1,2),middle[6:9][::-1]),axis=0))
    #### 最小外接矩形
    # rect = cv2.minAreaRect(contour) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    # box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点坐标
    # box = np.int0(box)
    # box = box.reshape((4,1,2))
    # minrect_area = cv2.contourArea(box)
    # print(abs(minrect_area-point_area)/point_area)
    #### 四个角点 IOU
    # biggest_box = np.concatenate((biggest_angle[0,:,:].reshape(1,1,2),biggest_angle[2,:,:].reshape(1,1,2),biggest_angle[3,:,:].reshape(1,1,2),biggest_angle[1,:,:].reshape(1,1,2)),axis=0)
    # biggest_mask = np.zeros_like(mask)
    # # corner_area = cv2.contourArea(biggest_box)
    # cv2.drawContours(biggest_mask,[biggest_box], -1, color=255, thickness=-1)
    
    # smooth = 1e-5
    # biggest_mask_ = biggest_mask > 50
    # mask_ = mask > 50
    # intersection = (biggest_mask_ & mask_).sum()
    # union = (biggest_mask_ | mask_).sum()
    # iou = (intersection + smooth) / (union + smooth)
    # if iou > 0.975:
    #     skip = True
    # else:
    #     skip = False
    # print(iou)
    # cv2.imshow('mask',cv2.resize(mask,(512,512)))
    # cv2.imshow('biggest_mask',cv2.resize(biggest_mask,(512,512)))
    # cv2.waitKey(0)    

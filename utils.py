from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn.functional as F
import os 
from skimage.filters import threshold_sauvola
import cv2 

def second2hours(seconds):
    h = seconds//3600
    seconds %= 3600
    m = seconds//60
    seconds %= 60

    hms = '{:d} H : {:d} Min'.format(int(h),int(m))
    return hms


def dict2string(loss_dict):
    loss_string = ''
    for key, value in loss_dict.items():
        loss_string += key+' {:.4f}, '.format(value)
    return loss_string[:-2]
def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)    

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])


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
def cvimg2torch(img,min=0,max=1):
    '''
    input:
        im -> ndarray uint8 HxWxC 
    return
        tensor -> torch.tensor BxCxHxW 
    '''
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    return img


def setup_seed(seed):
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed) #cpu
    # torch.cuda.manual_seed_all(seed)  #并行gpu
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    # torch.backends.cudnn.benchmark = False   #训练集变化不大时使训练加速

def SauvolaModBinarization(image,n1=51,n2=51,k1=0.3,k2=0.3,default=True):
    '''
	 Binarization using Sauvola's algorithm
		@name : SauvolaModBinarization
	 parameters
		@param image (numpy array of shape (3/1) of type np.uint8): color or gray scale image
	 optional parameters
		@param n1 (int) : window size for running sauvola during the first pass
		@param n2 (int): window size for running sauvola during the second pass
		@param k1 (float): k value corresponding to sauvola during the first pass
		@param k2 (float): k value corresponding to sauvola during the second pass
		@param default (bool) : bollean variable to set the above parameter as default. 
			@param default is set to True : thus default values of the above optional parameters (n1,n2,k1,k2) are set to
				n1 = 5 % of min(image height, image width)
				n2 = 10 % of min(image height, image width)
				k1 = 0.5
				k2 = 0.5
		Returns
			@return A binary image of same size as @param image
		
		@cite https://drive.google.com/file/d/1D3CyI5vtodPJeZaD2UV5wdcaIMtkBbdZ/view?usp=sharing
    '''

    if(default):
        n1 = int(0.05*min(image.shape[0],image.shape[1]))
        if (n1%2==0):
            n1 = n1+1
        n2 = int(0.1*min(image.shape[0],image.shape[1]))
        if (n2%2==0):
            n2 = n2+1
        k1 = 0.5
        k2 = 0.5
    if(image.ndim==3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.copy(image)
    T1 = threshold_sauvola(gray, window_size=n1,k=k1)
    max_val = np.amax(gray)
    min_val = np.amin(gray)
    C = np.copy(T1)
    C = C.astype(np.float32)
    C[gray > T1] = (gray[gray > T1] - T1[gray > T1])/(max_val - T1[gray > T1])
    C[gray <= T1] = 0
    C = C * 255.0
    new_in = np.copy(C.astype(np.uint8))
    T2 = threshold_sauvola(new_in, window_size=n2,k=k2)
    binary = np.copy(gray)
    binary[new_in <= T2] = 0
    binary[new_in > T2] = 255
    return binary,T2


def getBasecoord(h,w):
    base_coord0 = np.tile(np.arange(h).reshape(h,1),(1,w)).astype(np.float32)
    base_coord1 = np.tile(np.arange(w).reshape(1,w),(h,1)).astype(np.float32)
    base_coord = np.concatenate((np.expand_dims(base_coord1,-1),np.expand_dims(base_coord0,-1)),-1)
    return base_coord






import numpy as np
from scipy import ndimage as ndi

# lookup tables for bwmorph_thin

G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 0, 0], dtype=np.bool)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], dtype=np.bool)

def bwmorph(image, n_iter=None):
    """
    Perform morphological thinning of a binary image
    
    Parameters
    ----------
    image : binary (M, N) ndarray
        The image to be thinned.
    
    n_iter : int, number of iterations, optional
        Regardless of the value of this parameter, the thinned image
        is returned immediately if an iteration produces no change.
        If this parameter is specified it thus sets an upper bound on
        the number of iterations performed.
    
    Returns
    -------
    out : ndarray of bools
        Thinned image.
    
    See also
    --------
    skeletonize
    
    Notes
    -----
    This algorithm [1]_ works by making multiple passes over the image,
    removing pixels matching a set of criteria designed to thin
    connected regions while preserving eight-connected components and
    2 x 2 squares [2]_. In each of the two sub-iterations the algorithm
    correlates the intermediate skeleton image with a neighborhood mask,
    then looks up each neighborhood in a lookup table indicating whether
    the central pixel should be deleted in that sub-iteration.
    
    References
    ----------
    .. [1] Z. Guo and R. W. Hall, "Parallel thinning with
           two-subiteration algorithms," Comm. ACM, vol. 32, no. 3,
           pp. 359-373, 1989.
    .. [2] Lam, L., Seong-Whan Lee, and Ching Y. Suen, "Thinning
           Methodologies-A Comprehensive Survey," IEEE Transactions on
           Pattern Analysis and Machine Intelligence, Vol 14, No. 9,
           September 1992, p. 879
    
    Examples
    --------
    >>> square = np.zeros((7, 7), dtype=np.uint8)
    >>> square[1:-1, 2:-2] = 1
    >>> square[0,1] =  1
    >>> square
    array([[0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> skel = bwmorph_thin(square)
    >>> skel.astype(np.uint8)
    array([[0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    # check parameters
    if n_iter is None:
        n = -1
    elif n_iter <= 0:
        raise ValueError('n_iter must be > 0')
    else:
        n = n_iter
    
    # check that we have a 2d binary image, and convert it
    # to uint8
    skel = np.array(image).astype(np.uint8)
    
    if skel.ndim != 2:
        raise ValueError('2D array required')
    if not np.all(np.in1d(image.flat,(0,1))):
        raise ValueError('Image contains values other than 0 and 1')

    # neighborhood mask
    mask = np.array([[ 8,  4,  2],
                     [16,  0,  1],
                     [32, 64,128]],dtype=np.uint8)

    # iterate either 1) indefinitely or 2) up to iteration limit
    while n != 0:
        before = np.sum(skel) # count points before thinning
        
        # for each subiteration
        for lut in [G123_LUT, G123P_LUT]:
            # correlate image with neighborhood mask
            N = ndi.correlate(skel, mask, mode='constant')
            # take deletion decision from this subiteration's LUT
            D = np.take(lut, N)
            # perform deletion
            skel[D] = 0
            
        after = np.sum(skel) # coint points after thinning
        
        if before == after:
            # iteration had no effect: finish
            break
            
        # count down to iteration limit (or endlessly negative)
        n -= 1
    
    return skel.astype(np.bool)
    
"""
# here's how to make the LUTs
def nabe(n):
    return np.array([n>>i&1 for i in range(0,9)]).astype(np.bool)
def hood(n):
    return np.take(nabe(n), np.array([[3, 2, 1],
                                      [4, 8, 0],
                                      [5, 6, 7]]))
def G1(n):
    s = 0
    bits = nabe(n)
    for i in (0,2,4,6):
        if not(bits[i]) and (bits[i+1] or bits[(i+2) % 8]):
            s += 1
    return s==1
            
g1_lut = np.array([G1(n) for n in range(256)])
def G2(n):
    n1, n2 = 0, 0
    bits = nabe(n)
    for k in (1,3,5,7):
        if bits[k] or bits[k-1]:
            n1 += 1
        if bits[k] or bits[(k+1) % 8]:
            n2 += 1
    return min(n1,n2) in [2,3]
g2_lut = np.array([G2(n) for n in range(256)])
g12_lut = g1_lut & g2_lut
def G3(n):
    bits = nabe(n)
    return not((bits[1] or bits[2] or not(bits[7])) and bits[0])
def G3p(n):
    bits = nabe(n)
    return not((bits[5] or bits[6] or not(bits[3])) and bits[4])
g3_lut = np.array([G3(n) for n in range(256)])
g3p_lut = np.array([G3p(n) for n in range(256)])
g123_lut  = g12_lut & g3_lut
g123p_lut = g12_lut & g3p_lut
"""

"""
author : Peb Ruswono Aryan

metric for evaluating binarization algorithms
implemented : 

 * F-Measure
 * pseudo F-Measure (as in H-DIBCO 2010 & 2012)
 * Peak Signal to Noise Ratio (PSNR)
 * Negative Rate Measure (NRM)
 * Misclassification Penaltiy Measure (MPM)
 * Distance Reciprocal Distortion (DRD)

usage:
	python metric.py test-image.png ground-truth-image.png
"""


def drd_fn(im, im_gt):
	height, width = im.shape
	neg = np.zeros(im.shape)
	neg[im_gt!=im] = 1
	y, x = np.unravel_index(np.flatnonzero(neg), im.shape)
	
	n = 2
	m = n*2+1
	W = np.zeros((m,m), dtype=np.uint8)
	W[n,n] = 1.
	W = cv2.distanceTransform(1-W, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
	W[n,n] = 1.
	W = 1./W
	W[n,n] = 0.
	W /= W.sum()
	
	nubn = 0.
	block_size = 8
	for y1 in range(0, height, block_size):
		for x1 in range(0, width, block_size):
			y2 = min(y1+block_size-1,height-1)
			x2 = min(x1+block_size-1,width-1)
			block_dim = (x2-x1+1)*(y1-y1+1)
			block = 1-im_gt[y1:y2, x1:x2]
			block_sum = np.sum(block)
			if block_sum>0 and block_sum<block_dim:
				nubn += 1

	drd_sum= 0.
	tmp = np.zeros(W.shape)
	for i in range(min(1,len(y))):
		tmp[:,:] = 0 

		x1 = max(0, x[i]-n)
		y1 = max(0, y[i]-n)
		x2 = min(width-1, x[i]+n)
		y2 = min(height-1, y[i]+n)

		yy1 = y1-y[i]+n
		yy2 = y2-y[i]+n
		xx1 = x1-x[i]+n
		xx2 = x2-x[i]+n

		tmp[yy1:yy2+1,xx1:xx2+1] = np.abs(im[y[i],x[i]]-im_gt[y1:y2+1,x1:x2+1])
		tmp *= W

		drd_sum += np.sum(tmp)
	return drd_sum/nubn

def bin_metric(im,im_gt):
	height, width = im.shape
	npixel = height*width

	im[im>0] = 1
	gt_mask = im_gt==0
	im_gt[im_gt>0] = 1

	sk = bwmorph(1-im_gt)
	im_sk = np.ones(im_gt.shape)
	im_sk[sk] = 0
	
	kernel = np.ones((3,3), dtype=np.uint8)
	im_dil = cv2.erode(im_gt, kernel)
	im_gtb = im_gt-im_dil
	im_gtbd = cv2.distanceTransform(1-im_gtb, cv2.DIST_L2, 3)
	
	nd = im_gtbd.sum()

	ptp = np.zeros(im_gt.shape)
	ptp[(im==0) & (im_sk==0)] = 1
	numptp = ptp.sum()

	tp = np.zeros(im_gt.shape)
	tp[(im==0) & (im_gt==0)] = 1
	numtp = tp.sum()

	tn = np.zeros(im_gt.shape)
	tn[(im==1) & (im_gt==1)] = 1
	numtn = tn.sum()

	fp = np.zeros(im_gt.shape)
	fp[(im==0) & (im_gt==1)] = 1
	numfp = fp.sum()

	fn = np.zeros(im_gt.shape)
	fn[(im==1) & (im_gt==0)] = 1
	numfn = fn.sum()

	precision = numtp / (numtp + numfp)
	recall = numtp / (numtp + numfn)
	precall = numptp / np.sum(1-im_sk)
	fmeasure = (2*recall*precision)/(recall+precision)
	pfmeasure = (2*precall*precision)/(precall+precision)

	mse = (numfp+numfn)/npixel
	psnr = 10.*np.log10(1./mse)

	nrfn = numfn / (numfn + numtp)
	nrfp = numfp / (numfp + numtn)
	nrm = (nrfn + nrfp)/2

	im_dn = im_gtbd.copy()
	im_dn[fn==0] = 0
	dn = np.sum(im_dn)
	mpfn = dn / nd

	im_dp = im_gtbd.copy()
	im_dp[fp==0] = 0
	dp = np.sum(im_dp)
	mpfp = dp / nd

	mpm = (mpfp + mpfn) / 2
	drd = drd_fn(im, im_gt)

	return fmeasure, pfmeasure,psnr,nrm, mpm,drd
	# print("F-measure\t: {0}\npF-measure\t: {1}\nPSNR\t\t: {2}\nNRM\t\t: {3}\nMPM\t\t: {4}\nDRD\t\t: {5}".format(fmeasure, pfmeasure, psnr, nrm, mpm, drd))
'''
Misc Utility functions
'''
from collections import OrderedDict
import os
import numpy as np
import torch
import random
import torchvision

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9,):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - iter/max_iter)**power


def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images 
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended

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


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])

def visualize(epoch,model,layer):    
    #get conv layers
    conv_layers=[]
    for m in model.modules():
        if isinstance(m,torch.nn.modules.conv.Conv2d):
            conv_layers.append(m)

    # print conv_layers[layer].weight.data.cpu().numpy().shape
    tensor=conv_layers[layer].weight.data.cpu()
    vistensor(tensor, epoch, ch=0, allkernels=False, nrow=8, padding=1)


def vistensor(tensor, epoch, ch=0, allkernels=False, nrow=8, padding=1): 
    '''
        vistensor: visuzlization tensor
        @ch: visualization channel 
        @allkernels: visualization all tensors
        https://github.com/pedrodiamel/pytorchvision/blob/a14672fe4b07995e99f8af755de875daf8aababb/pytvision/visualization.py#L325
    ''' 
    
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min( (tensor.shape[0]//nrow + 1, 64 )  )
    # print rows
    # print tensor.shape
    grid = utils.make_grid(tensor, nrow=8, normalize=True, padding=padding)
    # print grid.shape
    plt.figure( figsize=(10,10), dpi=200 )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.savefig('./generated/filters_layer1_dwuv_'+str(epoch)+'.png')
    plt.close()


def show_uloss(uwpred,uworg,inp_img, samples=7):
    
    n,c,h,w=inp_img.shape
    # print(labels.shape)
    uwpred=uwpred.detach().cpu().numpy()
    uworg=uworg.detach().cpu().numpy()
    inp_img=inp_img.detach().cpu().numpy()

    #NCHW->NHWC
    uwpred=uwpred.transpose((0, 2, 3, 1))
    uworg=uworg.transpose((0, 2, 3, 1))

    choices=random.sample(range(n), min(n,samples))
    f, axarr = plt.subplots(samples, 3)
    for j in range(samples):
        # print(np.min(labels[j]))
        # print imgs[j].shape
        img=inp_img[j].transpose(1,2,0)
        axarr[j][0].imshow(img[:,:,::-1])
        axarr[j][1].imshow(uworg[j])
        axarr[j][2].imshow(uwpred[j])
    
    plt.savefig('./generated/unwarp.png')
    plt.close()


def show_uloss_visdom(vis,uwpred,uworg,labels_win,out_win,labelopts,outopts,args):
    samples=7
    n,c,h,w=uwpred.shape
    uwpred=uwpred.detach().cpu().numpy()
    uworg=uworg.detach().cpu().numpy()
    out_arr=np.full((samples,3,args.img_rows,args.img_cols),0.0)
    label_arr=np.full((samples,3,args.img_rows,args.img_cols),0.0)
    choices=random.sample(range(n), min(n,samples))
    idx=0
    for c in choices:
        out_arr[idx,:,:,:]=uwpred[c]
        label_arr[idx,:,:,:]=uworg[c]
        idx+=1

    vis.images(out_arr,
               win=out_win,
               opts=outopts)
    vis.images(label_arr,
               win=labels_win,
               opts=labelopts)

def show_unwarp_tnsboard(global_step,writer,uwpred,uworg,grid_samples,gt_tag,pred_tag):
    idxs=torch.LongTensor(random.sample(range(images.shape[0]), min(grid_samples,images.shape[0])))
    grid_uworg = torchvision.utils.make_grid(uworg[idxs],normalize=True, scale_each=True)
    writer.add_image(gt_tag, grid_uworg, global_step)
    grid_uwpr = torchvision.utils.make_grid(uwpred[idxs],normalize=True, scale_each=True)
    writer.add_image(pred_tag, grid_uwpr, global_step)

def show_wc_tnsboard(global_step,writer,images,labels, pred, grid_samples,inp_tag, gt_tag, pred_tag):
    idxs=torch.LongTensor(random.sample(range(images.shape[0]), min(grid_samples,images.shape[0])))
    grid_inp = torchvision.utils.make_grid(images[idxs],normalize=True, scale_each=True)
    writer.add_image(inp_tag, grid_inp, global_step)
    grid_lbl = torchvision.utils.make_grid(labels[idxs],normalize=True, scale_each=True)
    writer.add_image(gt_tag, grid_lbl, global_step)
    grid_pred = torchvision.utils.make_grid(pred[idxs],normalize=True, scale_each=True)
    writer.add_image(pred_tag, grid_pred, global_step)
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
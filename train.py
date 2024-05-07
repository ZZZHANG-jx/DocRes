import os
import cv2 
import time
import random 
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from piq import ssim,psnr
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils import data
import torch.distributed as dist 
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP 


from utils import dict2string,mkdir,get_lr,torch2cvimg,second2hours
from loaders import docres_loader
from models import restormer_arch


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)
# seed_torch()


def getBasecoord(h,w):
    base_coord0 = np.tile(np.arange(h).reshape(h,1),(1,w)).astype(np.float32)
    base_coord1 = np.tile(np.arange(w).reshape(1,w),(h,1)).astype(np.float32)
    base_coord = np.concatenate((np.expand_dims(base_coord1,-1),np.expand_dims(base_coord0,-1)),-1)
    return base_coord

def train(args):

    ## DDP init
    dist.init_process_group(backend='nccl',init_method='env://',timeout=datetime.timedelta(seconds=36000))
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda',args.local_rank)
    torch.cuda.manual_seed_all(42)

    ### Log file:
    mkdir(args.logdir)
    mkdir(os.path.join(args.logdir,args.experiment_name))
    log_file_path=os.path.join(args.logdir,args.experiment_name,'log.txt')
    log_file=open(log_file_path,'a')
    log_file.write('\n---------------  '+args.experiment_name+'  ---------------\n')
    log_file.close()

    ### Setup tensorboard for visualization
    if args.tboard:
        writer = SummaryWriter(os.path.join(args.logdir,args.experiment_name,'runs'),args.experiment_name)

    ### Setup Dataloader
    datasets_setting = [
        {'task':'deblurring','ratio':1,'im_path':'/home/jiaxin/Training_Data/DocRes_data/train/deblurring/','json_paths':['/home/jiaxin/Training_Data/DocRes_data/train/deblurring/tdd/train.json']},
        {'task':'dewarping','ratio':1,'im_path':'/home/jiaxin/Training_Data/DocRes_data/train/dewarping/','json_paths':['/home/jiaxin/Training_Data/DocRes_data/train/dewarping/doc3d/train_1_19.json']},
        {'task':'binarization','ratio':1,'im_path':'/home/jiaxin/Training_Data/DocRes_data/train/binarization/','json_paths':['/home/jiaxin/Training_Data/DocRes_data/train/binarization/train.json']},
        {'task':'deshadowing','ratio':1,'im_path':'/home/jiaxin/Training_Data/DocRes_data/train/deshadowing/','json_paths':['/home/jiaxin/Training_Data/DocRes_data/train/deshadowing/train.json']},
        {'task':'appearance','ratio':1,'im_path':'/home/jiaxin/Training_Data/DocRes_data/train/appearance/','json_paths':['/home/jiaxin/Training_Data/DocRes_data/train/appearance/trainv2.json']}
        ]


    ratios = [dataset_setting['ratio'] for dataset_setting in datasets_setting]
    datasets = [docres_loader.DocResTrainDataset(dataset=dataset_setting,img_size=args.im_size) for dataset_setting in datasets_setting]
    trainloaders = [{'task':datasets_setting[i],'loader':data.DataLoader(dataset=datasets[i], sampler=DistributedSampler(datasets[i]), batch_size=args.batch_size, num_workers=2, pin_memory=True,drop_last=True),'iter_loader':iter(data.DataLoader(dataset=datasets[i], sampler=DistributedSampler(datasets[i]), batch_size=args.batch_size, num_workers=2, pin_memory=True,drop_last=True))} for i in range(len(datasets))]


    ### test loader
    # for i in tqdm(range(args.total_iter)):
    #     loader_index = random.choices(list(range(len(trainloaders))),ratios)[0]
    #     in_im,gt_im = next(trainloaders[loader_index]['iter_loader'])


    ### Setup Model
    model = restormer_arch.Restormer( 
        inp_channels=6, 
        out_channels=3, 
        dim = 48,
        num_blocks = [2,3,3,4],     
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   
        dual_pixel_task = True       
    )
    model=DDP(model.cuda(),device_ids=[args.local_rank],output_device=args.local_rank)

    ### Optimizer
    optimizer= torch.optim.AdamW(model.parameters(),lr=args.l_rate,weight_decay=5e-4)

    ### LR Scheduler 
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.total_iter, eta_min=1e-6, last_epoch=-1)

    ### load checkpoint
    iter_start=0
    if args.resume is not None:                                         
        print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
        x = checkpoint['model_state']
        model.load_state_dict(x,strict=False)
        iter_start=checkpoint['iter']
        print("Loaded checkpoint '{}' (iter {})".format(args.resume, iter_start))

    ###-----------------------------------------Training-----------------------------------------
    ##initialize
    scaler = torch.cuda.amp.GradScaler()
    loss_dict = {}
    total_step = 0
    l2 = nn.MSELoss()
    l1 = nn.L1Loss()
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    m = nn.Sigmoid()
    best = 0
    best_ce = 999

    ## total_steps
    for iters in range(iter_start,args.total_iter):
        start_time = time.time()
        loader_index = random.choices(list(range(len(trainloaders))),ratios)[0]

        try:
            in_im,gt_im = next(trainloaders[loader_index]['iter_loader'])
        except StopIteration:
            trainloaders[loader_index]['iter_loader']=iter(trainloaders[loader_index]['loader'])
            in_im,gt_im = next(trainloaders[loader_index]['iter_loader'])
        in_im = in_im.float().cuda()
        gt_im = gt_im.float().cuda()

        binarization_loss,appearance_loss,dewarping_loss,deblurring_loss,deshadowing_loss = 0,0,0,0,0
        with torch.cuda.amp.autocast():
            pred_im = model(in_im,trainloaders[loader_index]['task']['task'])
            if trainloaders[loader_index]['task']['task'] == 'binarization':
                gt_im = gt_im.long()
                binarization_loss = ce(pred_im[:,:2,:,:], gt_im[:,0,:,:])
                loss = binarization_loss
            elif trainloaders[loader_index]['task']['task'] == 'dewarping':
                dewarping_loss = l1(pred_im[:,:2,:,:], gt_im[:,:2,:,:])
                loss = dewarping_loss
            elif trainloaders[loader_index]['task']['task'] == 'appearance':
                appearance_loss = l1(pred_im, gt_im)
                loss = appearance_loss
            elif trainloaders[loader_index]['task']['task'] == 'deblurring':
                deblurring_loss = l1(pred_im, gt_im)
                loss = deblurring_loss
            elif trainloaders[loader_index]['task']['task'] == 'deshadowing':
                deshadowing_loss = l1(pred_im, gt_im)
                loss = deshadowing_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 
    
        loss_dict['dew_loss']=dewarping_loss.item() if isinstance(dewarping_loss,torch.Tensor) else 0
        loss_dict['app_loss']=appearance_loss.item() if isinstance(appearance_loss,torch.Tensor) else 0
        loss_dict['des_loss']=deshadowing_loss.item() if isinstance(deshadowing_loss,torch.Tensor) else 0
        loss_dict['deb_loss']=deblurring_loss.item() if isinstance(deblurring_loss,torch.Tensor) else 0
        loss_dict['bin_loss']=binarization_loss.item() if isinstance(binarization_loss,torch.Tensor) else 0
        end_time = time.time()
        duration = end_time-start_time
        ## log
        if (iters+1) % 10 == 0:
            ## print
            print('iters [{}/{}] -- '.format(iters+1,args.total_iter)+dict2string(loss_dict)+' --lr {:6f}'.format(get_lr(optimizer))+' -- time {}'.format(second2hours(duration*(args.total_iter-iters))))
            ## tbord
            if args.tboard:
                for key,value in loss_dict.items():
                    writer.add_scalar('Train '+key+'/Iterations', value, total_step)
            ## logfile
            with open(log_file_path,'a') as f:
                f.write('iters [{}/{}] -- '.format(iters+1,args.total_iter)+dict2string(loss_dict)+' --lr {:6f}'.format(get_lr(optimizer))+' -- time {}'.format(second2hours(duration*(args.total_iter-iters)))+'\n')


        if (iters+1) % 5000 == 0:
            state = {'iters': iters+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            if not os.path.exists(os.path.join(args.logdir,args.experiment_name)):
                 os.system('mkdir ' + os.path.join(args.logdir,args.experiment_name))
            if torch.distributed.get_rank()==0:
                torch.save(state, os.path.join(args.logdir,args.experiment_name,"{}.pkl".format(iters+1)))

        sched.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--im_size', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--total_iter', nargs='?', type=int, default=100000, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=10, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=2e-4, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir', nargs='?', type=str, default='./checkpoints/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.add_argument('--local_rank',type=int,default=0,metavar='N')    
    parser.add_argument('--experiment_name', nargs='?', type=str,default='experiment_name',
                        help='the name of this experiment')
    parser.set_defaults(tboard=False)
    args = parser.parse_args()

    train(args)
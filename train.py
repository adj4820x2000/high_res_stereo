from __future__ import print_function
import pdb
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import yaml
from models import hsm
from utils import logger
torch.backends.cudnn.benchmark=True


parser = argparse.ArgumentParser(description='HSM-Net')
parser.add_argument('--config', type=str ,default='argument.yml',
                    help='Configure of post processing')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--mode', default='train',
                    help='train or test')
args = parser.parse_args()

## Load config argument
with open(args.config, 'r') as f:
    config = yaml.load(f)
args.maxdisp = config['maxdisp']
args.logname = config['logname']
args.database = config['database']
args.epochs = config['epochs']
args.batchsize = config['batchsize']
args.loadmodel = config['loadmodel']
args.savemodel = config['savemodel']

torch.manual_seed(args.seed)

model = hsm(args.maxdisp,clean=False,level=1)
model = nn.DataParallel(model)
model.cuda()

## Construct Model & Load Model
if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if ('disp' not in k) }
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

def _init_fn(worker_id):
    np.random.seed()
    random.seed()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


from dataloader import KITTIloader2015 as lk15
from dataloader import KITTIloader2012 as lk12
from dataloader import Carlaloader as lC
from dataloader import MiddleburyLoader as DA

batch_size = args.batchsize
## Control Training Resolution
scale_factor = args.maxdisp / 384.

## Data Loader
# KITTI 2012 & 2015
all_left_img, all_right_img, all_left_disp, all_left_semantic, all_right_semantic, all_left_instance, all_right_instance = lk12.dataloader('%s/KITTI/kitti_2012/training/'%args.database)
loader_kitti12 = DA.myImageFloder(all_left_img,all_right_img,all_left_disp,all_left_semantic, all_right_semantic, all_left_instance, all_right_instance, rand_scale=[0.9,2.4*scale_factor], order=0)
'''all_left_img, all_right_img, all_left_disp, all_left_semantic, all_right_semantic, all_left_instance, all_right_instance,_,_,_ = lk15.dataloader('%s/KITTI/kitti_2015/training/'%args.database,typ='train')
loader_kitti15 = DA.myImageFloder(all_left_img,all_right_img,all_left_disp,all_left_semantic, all_right_semantic, all_left_instance, all_right_instance, rand_scale=[0.9,2.4*scale_factor], order=0)
'''
# Carla
all_left_img, all_right_img, all_left_disp, all_left_semantic, all_right_semantic, all_left_instance, all_right_instance = lC.dataloader('%s/carla/training/'%args.database)
loader_carla = DA.myImageFloder(all_left_img,all_right_img,all_left_disp,all_left_semantic, all_right_semantic, all_left_instance, all_right_instance, rand_scale=[0.36,0.96*scale_factor], rand_bright=[0.8,1.2], order=0)

data_inuse = torch.utils.data.ConcatDataset([loader_kitti12]*80 + [loader_carla]*30)
#data_inuse = torch.utils.data.ConcatDataset([loader_kitti15] + [loader_kitti12]*80 + [loader_carla]*30)

TrainImgLoader = torch.utils.data.DataLoader(
         data_inuse, 
         batch_size= batch_size, shuffle= True, num_workers=batch_size, drop_last=True, worker_init_fn=_init_fn)

print('%d batches per epoch'%(len(data_inuse)//batch_size))

labels = np.array(
[[128, 64,128],
 [244, 35,232],
 [ 70, 70, 70],
 [102, 102, 156],
 [190, 153, 153],
 [153,153,153],
 [250, 170,  30],
 [220, 220,   0],
 [107, 142,  35],
 [152, 251, 152],
 [ 70, 130, 180],
 [220,  20,  60],
 [255,   0,   0],
 [  0,   0, 142],
 [  0,   0,  70],
 [  0,  60, 100],
 [  0,  80, 100],
 [  0,   0, 230],
 [119,  11,  32]] ,np.uint8)

def seg_scale_pyramid(img, num_scales):
        scaled_imgs = [img]
        _, _, h, w = img.size()
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = int(h / ratio)
            nw = int(w / ratio)
            scaled_imgs.append(F.interpolate(img, size=[nh, nw], mode='area'))
        return scaled_imgs

def L2_norm(x, axis=1):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset)
    return l2_norm

def disp_grad_Loss(pred, gt, mask):
    grad_pred_x = torch.abs((pred[:,1:,:] - pred[:,:-1,:]))
    grad_pred_y = torch.abs((pred[:,:,1:] - pred[:,:,:-1]))

    grad_gt_x = torch.abs((gt[:,1:,:] - gt[:,:-1,:]))
    grad_gt_y = torch.abs((gt[:,:,1:] - gt[:,:,:-1]))

    mean_diff_x = torch.abs(grad_pred_x - grad_gt_x).mean()
    mean_diff_y = torch.abs(grad_pred_y - grad_gt_y).mean()
    return (mean_diff_x + mean_diff_y).to(torch.float32)

## Train Process
def train(All_traning_data):
        model.train()

        ## Input & Ground truth Process
        All_traning_data = [Variable(torch.FloatTensor(datas)) for datas in All_traning_data]
        imgL, imgR, disp_true, sem_map_L, sem_map_R, sem_L, sem_R, inst_map_L, inst_map_R, inst_edge_L, inst_edge_R = [datas.cuda() for datas in All_traning_data]
        semantic_map_L_pyramid = seg_scale_pyramid(sem_map_L, 4)
        instance_edge_L_pyramid = seg_scale_pyramid(inst_edge_L, 4)

        mask = (disp_true > 0) & (disp_true < args.maxdisp)
        mask.detach_()

        ## Number of Semantic Class
        n_sem = 19

        optimizer.zero_grad()
        ## Run network
        ## Input : Left & Right 1.Image, 2.Semantic One-hot map, 3.Instance class map and edge
        ## Output : 1.Disparity, 2.Semantic One-hot map, 3.Instance map
        stacked,entropy,pred_seg = model(imgL,imgR, sem_L,sem_R, inst_map_L, inst_map_R, inst_edge_L, inst_edge_R)

        ## Loss Function
        #  disp_loss : smooth l1 loss
        disp_loss = (64./85)*F.smooth_l1_loss(stacked[0][mask], disp_true[mask], size_average=True) + \
                    (16./85)*F.smooth_l1_loss(stacked[1][mask], disp_true[mask], size_average=True) + \
                    (4./85)*F.smooth_l1_loss(stacked[2][mask], disp_true[mask], size_average=True) + \
                    (1./85)*F.smooth_l1_loss(stacked[3][mask], disp_true[mask], size_average=True)
        #  sem_seg_loss : cross_entropy
        sem_seg_loss = (1./8) * (1/2 * F.cross_entropy(pred_seg[0][:,:n_sem,:,:], semantic_map_L_pyramid[0].squeeze(1).long()) + \
                                1/4 * F.cross_entropy(pred_seg[1][:,:n_sem,:,:], semantic_map_L_pyramid[1].squeeze(1).long()) + \
                                1/8 * F.cross_entropy(pred_seg[2][:,:n_sem,:,:], semantic_map_L_pyramid[2].squeeze(1).long()) + \
                                1/16 * F.cross_entropy(pred_seg[3][:,:n_sem,:,:], semantic_map_L_pyramid[3].squeeze(1).long()) )
        #  ins_edge_seg_loss : L2_norm
        ins_edge_seg_loss = (1./80) * (1/2 * L2_norm(pred_seg[0][:,n_sem:n_sem+1,:,:]/2 - instance_edge_L_pyramid[0], None) + \
                                        1/4 * L2_norm(pred_seg[1][:,n_sem:n_sem+1,:,:]/2 - instance_edge_L_pyramid[1], None) + \
                                        1/8 * L2_norm(pred_seg[2][:,n_sem:n_sem+1,:,:]/2 - instance_edge_L_pyramid[2], None) + \
                                        1/16 * L2_norm(pred_seg[3][:,n_sem:n_sem+1,:,:]/2 - instance_edge_L_pyramid[3], None) )
        #  disp_reg_loss : disparity smoothness loss
        disp_reg_loss = (19.2/85) * (1/1 * disp_grad_Loss(stacked[0], disp_true, mask) + \
                                    1/4 * disp_grad_Loss(stacked[1], disp_true, mask) + \
                                    1/16 * disp_grad_Loss(stacked[2], disp_true,mask) + \
                                    1/64 * disp_grad_Loss(stacked[3], disp_true, mask) )

        ## Total Loss & Backpropagation
        total_loss = disp_loss + sem_seg_loss + ins_edge_seg_loss + disp_reg_loss
        total_loss.backward()
        optimizer.step()

        ## Output Process
        a = pred_seg[0][:,:n_sem,:,:].detach().cpu().numpy()
        sem_seg_pred = np.zeros((pred_seg[0].shape[2], pred_seg[0].shape[3], 3), dtype=np.uint8)
        for i in range(len(labels)):
            sem_seg_pred[a[0][i]==1] = labels[i]

        vis = {}
        vis['output3'] = (stacked[0]).detach().cpu().numpy()
        vis['output4'] = (stacked[1]).detach().cpu().numpy()
        vis['output5'] = (stacked[2]).detach().cpu().numpy()
        vis['output6'] = (stacked[3]).detach().cpu().numpy()
        vis['entropy'] = (entropy).detach().cpu().numpy()
        vis['sem_seg_out1'] = torch.from_numpy(np.transpose(sem_seg_pred, (2, 0, 1)))
        lossvalue = [total_loss.data, disp_loss, sem_seg_loss, ins_edge_seg_loss, disp_reg_loss]

        del stacked, entropy
        del total_loss
        return lossvalue,vis

def adjust_learning_rate(optimizer, epoch):
    if epoch <= args.epochs - 1:
        lr = 1e-3
    else:
        lr = 1e-4
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    log = logger.Logger(args.savemodel, name=args.logname)
    total_iters = 0

    for epoch in range(1, args.epochs+1):
        total_train_loss = 0

        ## Set Learning Rate
        adjust_learning_rate(optimizer,epoch)
                
        for batch_idx, All_traning_data in enumerate(TrainImgLoader):
            start_time = time.time()

            ## Perform Training Process
            imgL_crop,imgR_crop, disp_crop_L, sem_map_crop_L, sem_map_crop_R, sem_crop_L, sem_crop_R, inst_map_crop_L, inst_map_crop_R, inst_edge_crop_L, inst_edge_crop_R = All_traning_data
            loss,vis = train(All_traning_data)
            print('Iter %d training loss = %.3f , disp loss = %.3f , seg loss = %.3f , disp loss = %.3f , time = %.2f' %(batch_idx, loss[0], loss[1], loss[2]+loss[3], loss[4], time.time() - start_time))
            total_train_loss += loss[0]

            ## Save Log
            if total_iters %10 == 0:
                log.scalar_summary('train/total_loss_batch',loss[0], total_iters)
                log.scalar_summary('train/disp_loss_batch',loss[1], total_iters)
                log.scalar_summary('train/sem_seg_loss_batch',loss[2], total_iters)
                log.scalar_summary('train/inst_edge_seg_loss_batch',loss[3], total_iters)
                log.scalar_summary('train/disp_reg_loss_batch',loss[4], total_iters)
            if total_iters %100 == 0:
                sem_crop_L_rgb = np.zeros((sem_map_crop_L.shape[2], sem_map_crop_L.shape[3], 3), dtype=np.uint8)
                inst_map_crop_L_rgb = np.zeros((inst_map_crop_L.shape[2], inst_map_crop_L.shape[3], 3), dtype=np.uint8)
                for i in range(len(labels)):
                    sem_crop_L_rgb[sem_map_crop_L[0][0] == i] = labels[i]
                    inst_map_crop_L_rgb[inst_map_crop_L[0][0] == i] = labels[i]
                sem_crop_L_rgb = torch.from_numpy(np.transpose(sem_crop_L_rgb, (2, 0, 1)))
                inst_map_crop_L_rgb = torch.from_numpy(np.transpose(inst_map_crop_L_rgb, (2, 0, 1)))

                log.image_summary('train/left',imgL_crop[0:1],total_iters)
                log.image_summary('train/right',imgR_crop[0:1],total_iters)
                log.image_summary('train/gt0',disp_crop_L[0:1],total_iters)

                log.image_summary('train/gt_sem_seg_L', sem_crop_L_rgb.unsqueeze(0),total_iters)
                log.image_summary('train/pred_sem_seg', vis['sem_seg_out1'].unsqueeze(0),total_iters)
                log.image_summary('train/gt_inst_map_L', inst_map_crop_L_rgb.unsqueeze(0),total_iters)

                log.image_summary('train/entropy',vis['entropy'][0:1],total_iters)
                log.histo_summary('train/disparity_hist',vis['output3'], total_iters)
                log.histo_summary('train/gt_hist',np.asarray(disp_crop_L), total_iters)
                log.image_summary('train/output3',vis['output3'][0:1],total_iters)
                log.image_summary('train/output4',vis['output4'][0:1],total_iters)
                #log.image_summary('train/output5',vis['output5'][0:1],total_iters)
                #log.image_summary('train/output6',vis['output6'][0:1],total_iters)
                    
            total_iters += 1

            ## Save Training Weight Every 1000 Iterations
            if (total_iters + 1)%1000==0:
                savefilename = args.savemodel+'/'+args.logname+'/finetune_'+str(total_iters)+'.tar'
                torch.save({
                    'iters': total_iters,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
                }, savefilename)

        ## Save Final Training Weight
        savefilename = args.savemodel+'/'+args.logname+'/finetune_'+str(total_iters)+'.tar'
        torch.save({
            'iters': total_iters,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(TrainImgLoader),
        }, savefilename)
        log.scalar_summary('train/loss',total_train_loss/len(TrainImgLoader), epoch)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

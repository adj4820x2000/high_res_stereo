import argparse
import cv2
import math
from models import hsm
import numpy as np
import os
import pdb
import skimage.io
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
import yaml
from models.submodule import *
from utils.preprocess import get_transform
import matplotlib.pyplot as plt
#cudnn.benchmark = True
cudnn.benchmark = False

parser = argparse.ArgumentParser(description='HSM')
parser.add_argument('--config', type=str ,default='argument.yml',
                    help='Configure of post processing')
parser.add_argument('--level', type=int, default=1,
                    help='output level of output, default is level 1 (stage 3),\
                          can also use level 2 (stage 2) or level 3 (stage 1)')
args = parser.parse_args()

## Load config argument
with open(args.config, 'r') as f:
    config = yaml.load(f)
args.datapath = config['datapath']
args.loadmodel = config['loadmodel']
args.outdir = config['outdir']
args.clean = config['clean']
args.testres = config['testres']
args.max_disp = config['max_disp']

## Data Loader
from dataloader import listfiles as DA
test_left_img, test_right_img, _, _, left_semantic, right_semantic, left_instance, right_instance = DA.dataloader(args.datapath)

## Construct Model & Load Model
model = hsm(128,args.clean,level=args.level)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

## Dry Run
multip = 48
imgL = np.zeros((1,3,24*multip,32*multip))
imgR = np.zeros((1,3,24*multip,32*multip))
imgL = Variable(torch.FloatTensor(imgL).cuda())
imgR = Variable(torch.FloatTensor(imgR).cuda())
semL = np.zeros((1,19,24*multip,32*multip))
semR = np.zeros((1,19,24*multip,32*multip))
semL = Variable(torch.FloatTensor(semL).cuda())
semR = Variable(torch.FloatTensor(semR).cuda())
inst_map_L = np.zeros((1,1,24*multip,32*multip))
inst_map_R = np.zeros((1,1,24*multip,32*multip))
inst_map_L = Variable(torch.FloatTensor(inst_map_L).cuda())
inst_map_R = Variable(torch.FloatTensor(inst_map_R).cuda())
inst_edge_L = np.zeros((1,1,24*multip,32*multip))
inst_edge_R = np.zeros((1,1,24*multip,32*multip))
inst_edge_L = Variable(torch.FloatTensor(inst_edge_L).cuda())
inst_edge_R = Variable(torch.FloatTensor(inst_edge_R).cuda())
with torch.no_grad():
    model.eval()
    pred_disp, entropy = model(imgL,imgR, semL,semR, inst_map_L,inst_map_R, inst_edge_L,inst_edge_R)

def segmantation_process(seg_in):
    seg_o = cv2.resize(seg_in,None,fx=args.testres,fy=args.testres,interpolation=cv2.INTER_NEAREST)
    seg = np.expand_dims(seg_o.astype(float), axis=0)
    seg = np.reshape(seg,[1,1,seg.shape[1],seg.shape[2]])
    return seg

def segmantation_onehot_process(seg_in):
    seg_o = cv2.resize(seg_in,None,fx=args.testres,fy=args.testres,interpolation=cv2.INTER_NEAREST)
    seg = np.transpose(seg_o, (2,0,1))
    seg = np.expand_dims(seg.astype(float), axis=0)
    seg = np.reshape(seg,[1,19,seg.shape[2],seg.shape[3]])
    return seg

def main():
    processed = get_transform()
    model.eval()
    
    ## Number of Semantic Class
    n_sem = 19

    for inx in range(len(test_left_img)):
        print(test_left_img[inx])

        ## Load Test Data
        imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))[:,:,:3]
        imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))[:,:,:3]
        imgsize = imgL_o.shape[:2]
        semL_o = skimage.io.imread(left_semantic[inx])
        semR_o = skimage.io.imread(right_semantic[inx])
        instanceL_o = np.load(left_instance[inx])
        instanceR_o = np.load(right_instance[inx])

        ## Load Max Disp & Calibration
        if args.max_disp>0:
            if args.max_disp % 16 != 0:
                args.max_disp = 16 * math.floor(args.max_disp/16)
            max_disp = int(args.max_disp)
        else:
            with open(test_left_img[inx].replace('im0.png','calib.txt')) as f:
                lines = f.readlines()
                max_disp = int(int(lines[6].split('=')[-1]))

                ## Focal_len, Baseline
                focal_len = float(lines[0].split('[')[1].split(' ')[0])
                baseline = float(lines[3].split('=')[-1])/1000

        ## Change Max Disp
        tmpdisp = int(max_disp*args.testres//64*64)
        if (max_disp*args.testres/64*64) > tmpdisp:
            model.module.maxdisp = tmpdisp + 64
        else:
            model.module.maxdisp = tmpdisp
        if model.module.maxdisp ==64: model.module.maxdisp=128
        model.module.disp_reg8 =  disparityregression(model.module.maxdisp,16).cuda()
        model.module.disp_reg16 = disparityregression(model.module.maxdisp,16).cuda()
        model.module.disp_reg32 = disparityregression(model.module.maxdisp,32).cuda()
        model.module.disp_reg64 = disparityregression(model.module.maxdisp,64).cuda()
        print(model.module.maxdisp)

        ## Prepcess Test Data
        imgL_o = cv2.resize(imgL_o,None,fx=args.testres,fy=args.testres,interpolation=cv2.INTER_CUBIC)
        imgR_o = cv2.resize(imgR_o,None,fx=args.testres,fy=args.testres,interpolation=cv2.INTER_CUBIC)
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()
        imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
        imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

        sem_L = np.eye(n_sem, dtype='int')[semL_o]
        sem_R = np.eye(n_sem, dtype='int')[semR_o]
        sem_L = segmantation_onehot_process(sem_L)
        sem_R = segmantation_onehot_process(sem_R)
        
        inst_map_L = segmantation_process(instanceL_o[:,:,0])
        inst_map_R = segmantation_process(instanceR_o[:,:,0])

        inst_edge_L_o = cv2.Canny(instanceL_o[:,:,1],1,1)
        inst_edge_R_o = cv2.Canny(instanceR_o[:,:,1],1,1)
        inst_edge_L_o[inst_edge_L_o==255]=1
        inst_edge_R_o[inst_edge_R_o==255]=1
        inst_edge_L = segmantation_process(inst_edge_L_o)
        inst_edge_R = segmantation_process(inst_edge_R_o)

        ## Fast Pad
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]: max_h += 64
        if max_w < imgL.shape[3]: max_w += 64
        top_pad = max_h-imgL.shape[2]
        left_pad = max_w-imgL.shape[3]

        imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        sem_L = np.lib.pad(sem_L,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        sem_R = np.lib.pad(sem_R,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        inst_map_L = np.lib.pad(inst_map_L,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        inst_map_R = np.lib.pad(inst_map_R,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        inst_edge_L = np.lib.pad(inst_edge_L,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        inst_edge_R = np.lib.pad(inst_edge_R,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        sem_L = Variable(torch.FloatTensor(sem_L).cuda())
        sem_R = Variable(torch.FloatTensor(sem_R).cuda())
        inst_map_L = Variable(torch.FloatTensor(inst_map_L).cuda())
        inst_map_R = Variable(torch.FloatTensor(inst_map_R).cuda())
        inst_edge_L = Variable(torch.FloatTensor(inst_edge_L).cuda())
        inst_edge_R = Variable(torch.FloatTensor(inst_edge_R).cuda())

        ## Perform Testing Process
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            pred_disp, entropy = model(imgL,imgR, sem_L, sem_R, inst_map_L, inst_map_R, inst_edge_L, inst_edge_R)

            torch.cuda.synchronize()
            ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad   = max_h-imgL_o.shape[0]
        left_pad  = max_w-imgL_o.shape[1]
        entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]        

        ## Resize to High-resolution
        pred_disp = cv2.resize(pred_disp/args.testres,(imgsize[1],imgsize[0]),interpolation=cv2.INTER_NEAREST)

        ## Save Predictions
        idxname = test_left_img[inx].split('/')[-2]
        if not os.path.exists('%s'%(args.outdir)):
            os.makedirs('%s'%(args.outdir))

        # Clip While Keep Infinity
        invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
        pred_disp[invalid] = np.inf
        pred_disp[pred_disp<1] = 1

        cv2.imwrite('%s/%s-disp.png'% (args.outdir, idxname),pred_disp/pred_disp[~invalid].max()*255)
        #np.save('%s/%s-disp.npy'% (args.outdir, idxname),(pred_disp))

        # Let Disparity Convert to Depth(meter)
        depth = pred_disp
        depth[depth != np.inf] = baseline * focal_len / depth[depth != np.inf]
        depth[depth == np.inf] = depth[depth != np.inf].max()
        np.save('%s/%s-depth.npy'% (args.outdir, idxname),(depth))

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


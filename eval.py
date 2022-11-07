import cv2
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='HSM')
parser.add_argument('--evalpath', default='./eval',
                    help='evaluation data path')
parser.add_argument('--GTpath', default='./data-my',
                    help='ground truth path')
args = parser.parse_args()

def calc_3pe_standalone(disp_src, disp_dst):

    assert disp_src.shape == disp_dst.shape, "{}, {}".format(
        disp_src.shape, disp_dst.shape)
    assert len(disp_src.shape) == 2  # (N*M)

    not_empty = (disp_src > 0) & (~np.isnan(disp_src)) & (disp_dst > 0) & (
        ~np.isnan(disp_dst))

    disp_src_flatten = disp_src[not_empty].flatten().astype(np.float32)
    disp_dst_flatten = disp_dst[not_empty].flatten().astype(np.float32)

    disp_diff_l = abs(disp_src_flatten - disp_dst_flatten)

    accept_3p = (disp_diff_l <= 3) | (disp_diff_l <= disp_dst_flatten * 0.05)
    err_3p = 1 - np.count_nonzero(accept_3p) / len(disp_diff_l)

    return err_3p

indir = args.evalpath
gtdir = args.GTpath

files = os.listdir(gtdir)

avge = 0
abse = 0
rmse = 0
thPE = 0

for file in files:
    predname = indir + '/' + str(file).zfill(4) + '-depth.npy'
    predfile = np.load(predname)
    GTfile = cv2.imread(gtdir+'/'+file+'/disp.png',0)
    
    ## Focal_len, Baseline
    with open(gtdir+'/'+file+'/calib.txt') as f:
        lines = f.readlines()        
        focal_len = float(lines[0].split('[')[1].split(' ')[0])
        baseline = float(lines[3].split('=')[-1])/1000

    # if pred is depth，convert depth to disparity
    predfile = baseline*focal_len/predfile

    # Calculate AvgErr, RMSE, ABSE, 3PE 
    ERR = np.abs(predfile-GTfile)
    avge += ERR[GTfile>=1].mean()
    rmse += np.sqrt((ERR[GTfile>=1]**2).mean())
    abse += (ERR[GTfile>=1] / GTfile[GTfile>=1]).mean()        
    
    x = calc_3pe_standalone(GTfile, predfile)
    thPE += x
    
print("AvgErr", avge/len(files))
print("ABSE", abse/len(files))
print("RMSE", rmse/len(files))
print("3PE", thPE/len(files))
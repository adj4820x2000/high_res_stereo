import os, torch, torch.utils.data as data
from PIL import Image
import numpy as np
from utils import preprocess
from utils import readpfm as rp
from . import flow_transforms
import pdb
import torchvision
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
import cv2

IMG_EXTENSIONS = [
 '.jpg', '.JPG', '.jpeg', '.JPEG',
 '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any((filename.endswith(extension) for extension in IMG_EXTENSIONS))


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    if '.png' in path:
        data = Image.open(path)
        data = np.ascontiguousarray(data,dtype=np.float32)/256
        return data
    else:    
        return rp.readPFM(path)[0]

def segmantation_process(seg_in, top_pad, left_pad):
    seg_in = np.expand_dims(np.expand_dims(seg_in, 0), 0)
    seg_in = np.lib.pad(seg_in, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)[0]
    seg_in = np.ascontiguousarray(seg_in, dtype=np.float32)

    return seg_in

def segmantation_onehot_process(seg_in, top_pad, left_pad):
    seg_in = np.expand_dims(np.transpose(np.eye(19, dtype='int')[seg_in], (2,0,1)), 0)
    seg_in = np.lib.pad(seg_in, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)[0]
    seg_in = np.ascontiguousarray(seg_in, dtype=np.float32)

    return seg_in

class myImageFloder(data.Dataset):

    def __init__(self, left, right, left_disparity, left_semantic, right_semantic, left_instance, right_instance, right_disparity=None,
                 loader=default_loader, dploader=disparity_loader, rand_scale=[0.255,0.6], rand_bright=[0.5,2.], order=0):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.disp_R = right_disparity
        self.loader = loader
        self.dploader = dploader
        self.rand_scale = rand_scale
        self.rand_bright = rand_bright
        self.order = order

        self.left_semantic = left_semantic
        self.right_semantic = right_semantic
        self.left_instance = left_instance
        self.right_instance = right_instance

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        left_img = self.loader(left)
        right_img = self.loader(right)
        disp_L = self.disp_L[index]
        dataL = self.dploader(disp_L)
        dataL[dataL == np.inf] = 0

        left_semantic = self.left_semantic[index]
        sem_L = Image.open(left_semantic)
        right_semantic = self.right_semantic[index]
        sem_R = Image.open(right_semantic)
        left_instance = self.left_instance[index]
        inst_L = np.load(left_instance)
        right_instance = self.right_instance[index]
        inst_R = np.load(right_instance)

        if not (self.disp_R is None):
            disp_R = self.disp_R[index]
            dataR = self.dploader(disp_R)
            dataR[dataR == np.inf] = 0

        max_h = 2048//4 #512
        max_w = 3072//4 #768

        # photometric unsymmetric-augmentation
        random_brightness = np.random.uniform(self.rand_bright[0], self.rand_bright[1],2)
        random_gamma = np.random.uniform(0.8, 1.2,2)
        random_contrast = np.random.uniform(0.8, 1.2,2)
        left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
        left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
        left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
        right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
        right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
        right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
        right_img = np.asarray(right_img)
        left_img = np.asarray(left_img)

        sem_L = np.ascontiguousarray(sem_L, dtype=np.uint8)
        sem_R = np.ascontiguousarray(sem_R, dtype=np.uint8)
        inst_L = np.ascontiguousarray(inst_L, dtype=np.uint8)
        inst_R = np.ascontiguousarray(inst_R, dtype=np.uint8)

        inst_map_L = inst_L[:,:,0]
        inst_edge_L = cv2.Canny(inst_L[:,:,1],1,1)
        inst_edge_L[inst_edge_L==255]=1
        inst_map_R = inst_R[:,:,0]
        inst_edge_R = cv2.Canny(inst_R[:,:,1],1,1)
        inst_edge_R[inst_edge_R==255]=1
        
        # horizontal flip
        if not (self.disp_R is None):
            if np.random.binomial(1,0.5):
                tmp = right_img
                right_img = left_img[:,::-1]
                left_img = tmp[:,::-1]
                tmp = dataR
                dataR = dataL[:,::-1]
                dataL = tmp[:,::-1]

        # geometric unsymmetric-augmentation
        angle=0;px=0
        if np.random.binomial(1,0.5):
            angle=0.1;px=2
        co_transform = flow_transforms.Compose([
            flow_transforms.RandomVdisp(angle,px),
            flow_transforms.Scale(np.random.uniform(self.rand_scale[0],self.rand_scale[1]),order=self.order),
            flow_transforms.RandomCrop((max_h,max_w)),
            ])
        augmented = [left_img, right_img, sem_L, sem_R, inst_map_L, inst_map_R, inst_edge_L, inst_edge_R]
        augmented,dataL = co_transform(augmented, dataL)
        left_img, right_img, sem_L, sem_R, inst_map_L, inst_map_R, inst_edge_L, inst_edge_R = augmented

        # randomly occlude a region
        if np.random.binomial(1,0.5):
            sx = int(np.random.uniform(50,150))
            sy = int(np.random.uniform(50,150))
            cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
            cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
            right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

        h, w,_ = left_img.shape
        top_pad = max_h - h
        left_pad = max_w - w

        left_img = np.lib.pad(left_img, ((top_pad, 0), (0, left_pad),(0,0)), mode='constant', constant_values=0)
        right_img = np.lib.pad(right_img, ((top_pad, 0), (0, left_pad),(0,0)), mode='constant', constant_values=0)
        processed = preprocess.get_transform()
        left_img = processed(left_img)
        right_img = processed(right_img)

        sem_map_L = segmantation_process(sem_L, top_pad, left_pad)
        sem_map_R = segmantation_process(sem_R, top_pad, left_pad)
        sem_L = segmantation_onehot_process(sem_L, top_pad, left_pad)
        sem_R = segmantation_onehot_process(sem_R, top_pad, left_pad)
        inst_map_L = segmantation_process(inst_map_L, top_pad, left_pad)
        inst_map_R = segmantation_process(inst_map_R, top_pad, left_pad)
        inst_edge_L = segmantation_process(inst_edge_L, top_pad, left_pad)
        inst_edge_R = segmantation_process(inst_edge_R, top_pad, left_pad)

        dataL = np.expand_dims(np.expand_dims(dataL, 0), 0)
        dataL = np.lib.pad(dataL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)[0,0]
        dataL = np.ascontiguousarray(dataL, dtype=np.float32) 

        return (left_img, right_img, dataL, sem_map_L, sem_map_R, sem_L, sem_R, inst_map_L, inst_map_R, inst_edge_L, inst_edge_R)

    def __len__(self):
        return len(self.left)

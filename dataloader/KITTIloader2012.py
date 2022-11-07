import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  left_fold  = 'colored_0/'
  right_fold = 'colored_1/'
  disp_noc   = 'disp_occ/'
  sem_L = 'NVIDIA_semantic/semantic_0/'
  sem_R = 'NVIDIA_semantic/semantic_1/'  
  inst_L = '/PointRend/instance_0/'
  inst_R = '/PointRend/instance_1/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]

  train = image[:]

  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train = [filepath+disp_noc+img for img in train]

  sem_train_L = [filepath+sem_L+img for img in train]
  sem_train_R = [filepath+sem_R+img for img in train]
  inst_train_L = [filepath+inst_L+img.split('.')[0]+'.npy' for img in train]
  inst_train_R = [filepath+inst_R+img.split('.')[0]+'.npy' for img in train]


  return left_train, right_train, disp_train, sem_train_L, sem_train_R, inst_train_L, inst_train_R
  

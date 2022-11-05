import numpy as np
import cv2 #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        npimage = npimage.transpose((3, 0, 1, 2))
        npmask = npmask.transpose((3, 0, 1, 2))
        npmask = npmask.astype("float32")
        npimage = npimage.astype("float32")

        return npimage,npmask


        #读图片（如jpg、png）的代码
        '''
        image = imread(img_path)
        mask = imread(mask_path)

        image = image.astype('float32') / 255
        mask = mask.astype('float32') / 255

        if self.aug:
            if random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                image = image[::-1, :, :].copy()
                mask = mask[::-1, :].copy()

        image = color.gray2rgb(image)
        #image = image[:,:,np.newaxis]
        image = image.transpose((2, 0, 1))
        mask = mask[:,:,np.newaxis]
        mask = mask.transpose((2, 0, 1))       
        return image, mask
        '''


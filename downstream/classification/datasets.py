import os
import sys
import numpy as np
import torch
import random
import glob

from torch.utils.data import Dataset

import torchvision
# import torchvision.transforms as T

from config import parse_arguments
from PIL import Image
import cv2
import SimpleITK as sitk
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DiseaseDataset(Dataset):
    def __init__(self, input_path, mode, image_size, bits, args, transform=None):
        self.mode = mode # Unused variable. However, it will be used for transform
        self.args = args
        self.image_size = image_size
        self.bits = bits
        with open(input_path, "r") as f:
            self.samples = json.load(f)
        
        """
        augmentation strategy 
        """

        if mode == 'train':
            if args.aug == True:
                self.transform = A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    A.OneOf([
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.MotionBlur(p=0.2),
                        # A.IAASharpen(p=0.2),
                        ], p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.2),
                    A.OneOf([
                        A.OpticalDistortion(p=0.3),
                        ], p=0.2),
                    # A.OneOf([
                    #     A.CLAHE(clip_limit=4.0),
                    #     A.Equalize(),
                    #     ], p=0.2),
                    A.OneOf([
                        A.GaussNoise(p=0.2),
                        A.MultiplicativeNoise(p=0.2),
                        ], p=0.2),
                    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0.1, p=0.3),
                    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    A.Normalize(mean=(0.485), std=(0.229)),
                    ToTensorV2(),
                    ])
            else:
                self.transform = A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    A.Normalize(mean=(0.485), std=(0.229)),
                    ToTensorV2(),
                    ])
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Normalize(mean=(0.485), std=(0.229)),
                ToTensorV2(),
                ])

    def __getitem__(self, idx):
        imgs = self.transform(image=self._preprocessing(self.samples['imgs'][idx] , 
                                self.bits))['image']
        labels = np.array(self.samples['labels'][idx]).astype(np.float32)
        return imgs, labels
            
    def __len__(self):
        return len(self.samples['labels'])
    
    def _preprocessing(self, path , bits):
        if bits == 8:
            img = np.array(Image.open(path))
            img = self._min_max_scaling(img)
        else:
            img = np.load(path)
        img = self._standardization(img)
        return img

    def _min_max_scaling(self, img):
        return (img-np.min(img)) / (np.max(img)-np.min(img))
    
    def _standardization(self, img):
        np_img = img.astype(np.float32)
        mean, std = np.mean(np_img), np.std(np_img)
        norm_img = (np_img-mean)/std
        return norm_img

# For test
if __name__ == '__main__':
    dataset = DiseaseDataset('./json/dummy.json')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)

    for imgs, labels in iter(dataloader):
        print(imgs.shape, labels.shape)
        

import glob
import json
import os
import random
import sys

import albumentations as A
import cv2
import numpy as np
import SimpleITK as sitk
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from config import parse_arguments
from PIL import Image
from torch.utils.data import Dataset


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
                        ], p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.2),
                    A.OneOf([
                        A.OpticalDistortion(p=0.3),
                        ], p=0.2),
                    A.OneOf([
                        A.GaussNoise(p=0.2),
                        A.MultiplicativeNoise(p=0.2),
                        ], p=0.2),
                    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0.1, p=0.3),
                    A.Normalize(mean=(0.485), std=(0.229)),
                    ToTensorV2(),
                    ])
            else:
                self.transform = A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    A.Normalize(mean=(0.485), std=(0.229)),
                    ToTensorV2(),
                    ])
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
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
        img = cv2.imread(path)
        img = self.cutoff(img, bits)
        return img

    # our  preprocessing
    def cutoff(self,img, bits):
        img = cv2.resize(img,(512,512))
        img = np.clip(img, 0, np.percentile(img, 99))
        img -= img.min()
        img /= (img.max() - img.min())
        img *= 255
        if bits == 8:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32)
        return img
    
# For test
if __name__ == '__main__':
    dataset = DiseaseDataset('./json/dummy.json')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)

    for imgs, labels in iter(dataloader):
        print(imgs.shape, labels.shape)
        

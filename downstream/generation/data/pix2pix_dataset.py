"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
import torchvision
import glob
import numpy as np
import cv2
import random

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt
        # random.seed(1001)
        if opt.isTrain:
            # Train
            label_paths = glob.glob(os.path.join(opt.dataroot, 'mask', '*'))
            image_paths = glob.glob(os.path.join(opt.dataroot, 'data', '*'))
        else:
            # Test
            label_paths = glob.glob(os.path.join(opt.dataroot, 'test', 'nodule_seg', 'mask', '*'))
            image_paths = glob.glob(os.path.join(opt.dataroot, 'test', 'nodule_seg', 'data', '*'))

        instance_paths = []

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
#         size = 10000
        self.dataset_size = size

        dataloader = []

        for image in image_paths:
            for label in label_paths:
                dataloader.append({'image' : image, 'label' : label})

        self.dataloader = dataloader

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image

        label_path = self.dataloader[index]['label']
        image_path = self.dataloader[index]['image']
        
        label_tensor = cv2.resize(np.load(label_path), (512,512)).reshape(1, 512, 512)
        image_tensor = self.transform(Image.open(image_path))
        instance_tensor = 0
        
        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

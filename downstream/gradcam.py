import sys
sys.path.append('./MONAI')
from monai.visualize import GradCAM
from monai.visualize import CAM
import os
import random
import math

import numpy as np
from config import parse_arguments
from datasets import DiseaseDataset
from models.resnet import resnet50, resnet152
from models.vgg import vgg16,vgg16_bn
from models.densenet import densenet121
import pandas as pd
from utils_folder.utils import AverageMeter, ProgressMeter
from utils_folder.eval_metric import *

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import time
import pathlib
from datetime import datetime
import cv2
import warnings

fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0,2,3,1)

def evaluate_cam(args, loader, model, device, num_classes , class_list, args.log_dir):

    model.eval()
    save_dir = args.log_dir

    for iter_, (imgs, labels) in enumerate(iter(loader)):

        imgs = imgs.to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(imgs)        
        cam = GradCAM(nn_module = model, target_layers = 'layer4')
        result = cam(x=imgs, layer_idx=-1)
        result = result.squeeze()
        heatmap = np.uint8(255 * result)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = heatmap/255

        gt_imgs = fn_tonumpy(imgs)
        cam_imgs = np.stack((gt_imgs[0].squeeze(),)*3, axis=-1)
    
        superimposed_img = heatmap*0.2 + cam_imgs
        superimposed_img /= 1.2

        full_image = np.zeros((512,1024,3))
        full_image[:512, :512, :] = gt_imgs[0][:,:,0]*255
        full_image[:512, 512:1024, :] = superimposed_img*255

        cv2.imwrite(os.path.join(save_dir,'{}_cam.jpg'.format(iter_)), full_image)

def main(args):
    ##### Initial Settings
    csv_data = pd.read_csv(args.csv_file)
    class_list = csv_data.keys().tolist()[5:] # warning
    print("[*] class list : " , class_list)
    num_classes = args.num_class
    downstream = '{}_{}_class'.format(args.downstream_name, num_classes)

    print('\n[*****] ', downstream)
    print('[*] using {} bit images'.format(args.bit))

    # device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[*] device: ', device)

    # path setting
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    folder_name = '{}_{}_{}_bit'.format(args.message, downstream, args.bit)

    args.log_dir = os.path.join(args.log_dir, folder_name)
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # for log
    f = open(os.path.join(args.log_dir,'arguments.txt'), 'w')
    f.write(str(args))
    f.close()
    print('[*] log directory: {} '.format(args.log_dir))
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed) # os 자체의 seed 고정
        np.random.seed(args.seed) # numpy seed 고정 
        torch.cuda.manual_seed(args.seed) # cudnn seed 고정
        torch.backends.cudnn.deterministic = True # cudnn seed 고정(nn.Conv2d)
        torch.backends.cudnn.benchmark = False # CUDA 내부 연산에서 가장 빠른 알고리즘을 찾아 수행


    # select network
    print('[*] build network... backbone: {}'.format(args.backbone))
    if args.backbone == 'resnet50':
        model = resnet50(num_classes=args.num_class)
    elif args.backbone == 'vgg':
        model = vgg16(num_classes=args.num_class)
    elif args.backbone == 'densenet':
        model = densenet169(num_classes=args.num_class)
    elif args.backbone == 'inception':
        model = Inception3(num_classes=args.num_class)
    else:
        ValueError('Have to set the backbone network in [resnet, vgg, densenet]')

    model = model.to(device)

    if args.resume is True:
        checkpoint = torch.load(args.pretrained)
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        model.load_state_dict(pretrained_dict)
        print("load model completed")
    else:
        ValueError('Have to input a pretrained network path')

    ##### Dataset & Dataloader
    print('[*] prepare datasets & dataloader...')
    test_datasets = DiseaseDataset(args.test_path, 'test', args.img_size, args.bits, args)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1, 
    num_workers=args.w, pin_memory=True, drop_last=True)
    
    ##### Train & Test
    print('[*] start a test')
    evaluate_cam(args, test_loader, model, device, num_classes , class_list , args.log_dir)
        
if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)

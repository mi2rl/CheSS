import os
import sys
import random
import math

import numpy as np
from config import parse_arguments
from datasets import DiseaseDataset
from models.resnet import resnet50, resnet152
from models.vgg import vgg16
from models.inception_v3 import Inception3

import pandas as pd
from utils_folder.utils import AverageMeter, ProgressMeter
from utils_folder.eval_metric import *

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import time
import pathlib
from datetime import datetime
import cv2


def evaluate(args, loader, model, device, num_classes , class_list):

    model.eval()
    correct = 0
    total = 0
    overall_logits = []
    overall_preds = []
    overall_gts = []

    for iter_, (imgs, labels) in enumerate(iter(loader)):

        imgs = imgs.to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(imgs)
        outputs = torch.sigmoid(outputs)
        outputs_preds = outputs.clone()
        overall_logits += outputs.cpu().detach().numpy().tolist()

        outputs_preds[outputs_preds >= 0.5] = 1
        outputs_preds[outputs_preds < 0.5] = 0
        total += labels.size(0) * labels.size(1)
        correct += torch.sum(outputs_preds == labels.data).item()

        ## For evaluation
        overall_preds += outputs_preds.cpu().detach().numpy().tolist()
        overall_gts += labels.cpu().detach().numpy().tolist()

    print('[*] Test Acc: {:5f}'.format(100.*correct/total))
    
    AUROCs = compute_AUCs(overall_gts, overall_logits, num_classes , class_list)
    AUROC_avg = np.array(AUROCs).mean()

    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(num_classes):
        if class_list[i] == 'Fracture':
            continue
        print('The AUROC of {} is {}'.format(class_list[i], AUROCs[i]))

    cnf_matrix = compute_confusion_matrix(overall_gts, overall_preds, num_classes , class_list)
    
    for label, matrix in cnf_matrix.items():
        print("Confusion matrix for label {}:".format(label))
        get_mertrix(matrix, args.log_dir, class_list)

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
    folder_name = '{}_{}'.format(args.message , downstream)
    
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
    evaluate(args, test_loader, model, device, num_classes , class_list)
        
if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)

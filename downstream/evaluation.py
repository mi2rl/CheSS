import os
import sys
import random
import math

import numpy as np
from config import parse_arguments
from datasets import DiseaseDataset
from models.resnet import resnet50, resnet152
from models.vgg import vgg16,vgg16_bn
from models.densenet import densenet121

from utils_folder.utils import AverageMeter, ProgressMeter
from utils_folder.eval_metric import get_metric , get_mertrix

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import time
import pathlib
from datetime import datetime
import cv2
from sklearn.metrics import roc_auc_score

def compute_AUCs(gt, pred , num_classes , class_list):

    """
    https://github.com/arnoweng/CheXNet/blob/master/model.py
    Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def evaluate(args, loader, model, num_classes , class_list):
    model.eval()
    
    correct = 0
    total = 0
    
    overall_logits = []
    overall_preds = []
    overall_gts = []

    for iter_, (imgs, labels) in enumerate(iter(loader)):
        imgs = imgs.cuda()
        labels = labels.cuda()
        
        outputs = model(imgs)
        outputs = torch.sigmoid(outputs)
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0

        total += labels.size(0)
        correct += torch.sum(outputs == labels.data).item()
        
        ## For evaluation
        ## have to modify
        overall_logits += [outputs[i,1].cpu().detach().item() for i in range(labels.shape[0])]
        overall_preds += preds.cpu().detach().numpy().tolist()
        overall_gts += labels.cpu().detach().numpy().tolist()

    print('[*] Test Acc: {:5f}'.format(100.*correct/total))
    get_mertrix(overall_gts, overall_preds, overall_logits, args.log_dir, class_list)
    get_metric(overall_gts, overall_preds, overall_logits, args.log_dir, class_list)

    AUROCs = compute_AUCs(gt, pred, num_classes , class_list)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(num_classes):
        print('The AUROC of {} is {}'.format(class_list[i], AUROCs[i]))

def main(args):
    ##### Initial Settings
    csv_data = pd.read_csv(args.csv_file)
    class_list = csv_data.keys().tolist()[5:] # warning
    num_classes = args.num_class
    downstream = '{}_{}_class'.format(args.downstream_name, num_classes)

    print('\n[*****] ', downstream)
    print('[*] using {} bit images'.format(args.bit))

    # device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[*] device: ', device)

    # path setting
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    today = str(datetime.today()).split(' ')[0] + '_' + str(time.strftime('%H%M%S'))
    folder_name = 'test_{}'.format(args.message)
    
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
    if args.backbone == 'resnet':
        model = resnet50(num_classes=args.num_class)
    elif args.backbone == 'vgg':
        model = vgg16(num_classes=args.num_class)
    elif args.backbone == 'densenet':
        model = densenet169(num_classes=args.num_class)
    elif args.backbone == 'inception':
        model = Inception3(num_classes=args.num_class)
    else:
        ValueError('Have to set the backbone network in [resnet, vgg, densenet]')

    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

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
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=4, 
    num_workers=args.w, pin_memory=True, drop_last=True)
    
    ##### Train & Test
    print('[*] start a test')
    evaluate(args, test_loader, model, num_classes , class_list)
        
if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)

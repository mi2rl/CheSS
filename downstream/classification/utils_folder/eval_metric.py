import sys
import os

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import itertools

import json

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
    gt_np = np.array(gt)
    pred_np = np.array(pred)
    for i in range(num_classes):
        if class_list[i] == 'Fracture':
            continue
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def compute_confusion_matrix(gt, preds , num_classes , class_list):
    
    conf_mat_dict={}
    gt_np = np.array(gt)
    pred_np = np.array(preds)
    for i in range(num_classes):
        y_true_label = gt_np[:, i]
        y_pred_label = pred_np[:, i]
        conf_mat_dict[class_list[i]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)

    return conf_mat_dict

def save_confusion_matrix(cm, target_names, log_dir, title='CFMatrix', cmap=None, normalize=False):
    acc = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - acc

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i,j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\n accuracy={:0.4f}'.format(acc))
    plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'))
    
def get_mertrix(confusion_matrix, log_dir, class_list=['Normal', 'Abnormal']):
    
    cnf_matrix = confusion_matrix
    save_confusion_matrix(cnf_matrix, class_list, log_dir)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    F1_Score = 2*(PPV*TPR) / (PPV+TPR)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)


    print('specificity: ', TNR) 
    print('sensitivity (recall): ', TPR) # true positive rate
    print('positive predictive value (precision): ', PPV)
    print('negative predictive value: ', NPV)
    print('ACC: ', ACC)
    print('F1_score: ', F1_Score)


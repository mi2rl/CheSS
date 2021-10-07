import sys
import os

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import itertools

import json

def save_roc_auc_curve(overall_gt, overall_output, log_dir):
    ### ROC, AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    np_gt = np.array(overall_gt)
    np_output = np.array(overall_output)
    fpr, tpr, _ = roc_curve(np_gt, np_output, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' %roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(log_dir, 'roc_auc.png'))

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

# def save_metric(tn, fp, fn, tp, log_dir):
#     tn, fp, fn, tp = tn.item(), fp.item(), fn.item(), tp.item()
    
#     specificity = tn/(tn+fp)
#     sensitivity = tp/(tp+fn)
#     ppv = tp/(tp+fp)
#     if tn+fn == 0:
#         npv = 0
#     else:
#         npv = tn/(tn+fn)
#     acc = (tp+tn) / (tn+fp+fn+tp) 

#     result_dict = {}
#     result_dict['tn'] = tn
#     result_dict['tp'] = tp
#     result_dict['fn'] = fn
#     result_dict['fp'] = fp
#     result_dict['specificity'] = specificity
#     result_dict['sensitivity'] = sensitivity
#     result_dict['ppv'] = ppv
#     result_dict['npv'] = npv
#     result_dict['acc'] = acc 
    
#     print('tn, fp, fn, tp: ', tn, fp, fn, tp)
#     print('specificity: ', specificity)
#     print('sensitivity: ', sensitivity)
#     print('positive predictive value: ', ppv)
#     print('negative predictive value: ', npv)
#     print('acc: ', acc)
    
#     with open(os.path.join(log_dir, 'result_metric.json'), 'w') as f:
#         json.dump(result_dict, f, indent=4)

# def get_metric(gt, pred, logit, log_dir, class_list=['Normal', 'Abnormal']):
#     cf_matrix = confusion_matrix(gt,pred)
#     save_confusion_matrix(cf_matrix, class_list, log_dir)
#     # tn, fp, fn, tp = cf_matrix.ravel()
#     # save_metric(tn, fp, fn, tp, log_dir)
#     save_roc_auc_curve(gt, logit, log_dir)
    
def save_metric(tn, fp, fn, tp, log_dir):
    tn, fp, fn, tp = tn.item(), fp.item(), fn.item(), tp.item()
    
    specificity = tn/(tn+fp)
    sensitivity = tp/(tp+fn)
    ppv = tp/(tp+fp)
    if tn+fn == 0:
        npv = 0
    else:
        npv = tn/(tn+fn)
    acc = (tp+tn) / (tn+fp+fn+tp) 

    result_dict = {}
    result_dict['tn'] = tn
    result_dict['tp'] = tp
    result_dict['fn'] = fn
    result_dict['fp'] = fp
    result_dict['specificity'] = specificity
    result_dict['sensitivity'] = sensitivity
    result_dict['ppv'] = ppv
    result_dict['npv'] = npv
    result_dict['acc'] = acc 
    
    print('tn, fp, fn, tp: ', tn, fp, fn, tp)
    print('specificity: ', specificity)
    print('sensitivity: ', sensitivity)
    print('positive predictive value: ', ppv)
    print('negative predictive value: ', npv)
    print('acc: ', acc)
    
    with open(os.path.join(log_dir, 'result_metric.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)

def get_metric(gt, pred, logit, log_dir, class_list=['Normal', 'Abnormal']):
    cf_matrix = confusion_matrix(gt,pred)
    save_confusion_matrix(cf_matrix, class_list, log_dir)
    # tn, fp, fn, tp = cf_matrix.ravel()
    # save_metric(tn, fp, fn, tp, log_dir)
    save_roc_auc_curve(gt, logit, log_dir)
    
    
def get_mertrix(gt, pred, logit, log_dir, class_list=['Normal', 'Abnormal']):
    
    cnf_matrix = confusion_matrix(gt,pred)
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
    print('acc: ', ACC)
    print('f1_score: ', F1_Score)


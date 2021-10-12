import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from models.resnet import resnet152
from datasets import DiseaseDataset
import pathlib
import os

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            print(x, self.target_layers)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            elif "acm" in name.lower():
                continue
            else:
                x = module(x)
        
        return target_activations, x

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model  #resnet 152
        self.feature_module = feature_module #resnet152. layer4
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        
        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def save_cam(args, imgs, mask, preds, labels, iter_):
    b, c, h, w = imgs.shape
    pic = np.zeros((h, w*2, 3))
    img = np.stack([imgs[0, ...].cpu().detach().numpy().transpose(1,2,0).squeeze(-1)]*3, axis=-1)
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET).astype(np.float32)/255
    cam = heatmap + img.astype(np.float32)
    cam = cam / np.max(cam)

    pic[:, :w, :] = np.uint8(img*255)
    pic[:, w:, :] = np.uint8(cam*255)
    
    pred_class = preds.cpu().detach().numpy()[0]
    label_class = labels.cpu().detach().numpy()[0]

    img_dir = os.path.join(args.log_dir,'imgs')
    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)
    
    tp_dir = os.path.join(img_dir, 'tp')
    pathlib.Path(tp_dir).mkdir(parents=True, exist_ok=True)
    
    tn_dir = os.path.join(img_dir, 'tn')
    pathlib.Path(tn_dir).mkdir(parents=True, exist_ok=True)
    
    fp_dir = os.path.join(img_dir, 'fp')
    pathlib.Path(fp_dir).mkdir(parents=True, exist_ok=True)
    
    fn_dir = os.path.join(img_dir, 'fn')
    pathlib.Path(fn_dir).mkdir(parents=True, exist_ok=True)
    if (pred_class == 1) & (label_class == 1):
        cv2.imwrite('{}/{}.jpg'.format(tp_dir, iter_), pic)

    elif (pred_class == 0) & (label_class == 0):
        cv2.imwrite('{}/{}.jpg'.format(tn_dir, iter_), pic)

    elif (pred_class == 0) & (label_class == 1):
        cv2.imwrite('{}/{}.jpg'.format(fn_dir, iter_), pic)

    elif (pred_class == 1) & (label_class == 0):
        cv2.imwrite('{}/{}.jpg'.format(fp_dir, iter_), pic)







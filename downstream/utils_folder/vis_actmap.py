import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import pathlib

def register_forward_hook(args, model):
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    layer_names = ['1', '2', '3', '4']
    if args.backbone == 'resnet': # block 1, 2, 3, 4
        model.layer1.register_forward_hook(get_activation(layer_names[0])) # 1/2
        model.layer2.register_forward_hook(get_activation(layer_names[1])) # 1/4
        model.layer3.register_forward_hook(get_activation(layer_names[2])) # 1/8
        model.layer4.register_forward_hook(get_activation(layer_names[3])) # 1/16
    elif args.backbone == 'vgg': # 8, 15, 22, 29
        model.features[8].register_forward_hook(get_activation(layer_names[0]))
        model.features[15].register_forward_hook(get_activation(layer_names[1]))
        model.features[22].register_forward_hook(get_activation(layer_names[2]))
        model.features[29].register_forward_hook(get_activation(layer_names[3]))
    elif args.backbone == 'densenet': # denseblock 1, 2, 3, 4
        model.features.denseblock1.register_forward_hook(get_activation(layer_names[0]))
        model.features.denseblock2.register_forward_hook(get_activation(layer_names[1]))
        model.features.denseblock3.register_forward_hook(get_activation(layer_names[2]))
        model.features.denseblock4.register_forward_hook(get_activation(layer_names[3]))
    else:
        ValueError('have to select backbone network in [resnet, vgg, densenet]')

    return activation, layer_names


def visualize_activation_map(args, activation, layer_names, iter_, preds, labels, img):
    acts = []
    img_dir = os.path.join(args.log_dir, 'imgs')
    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)
    
    visual_num = img.shape[0]
    # print("visual_num" , visual_num)
    for layer in layer_names:
        act = activation[layer].squeeze()
        # print("act" , act.shape)
        # if len(act.shape) > 3:
            # b, c, h, w = act.shape
        # act = torch.mean(act, dim=1)
        # act -= act.min(1, keepdim=True)[0]
        # act /= act.max(1, keepdim=True)[0]
        acts.append(act)

    if len(acts) > 0: 
        for batch in range(visual_num): #batch
            np_img = img[batch,0,:,:].cpu().detach().numpy()
            np_img = cv2.resize(np_img, (512, 512))
            np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
            
            full_image = np.zeros((512, 2560, 3))
            full_image[:, :512, ...] = np_img * 255
            for ly in range(len(layer_names)):
                # print("acts[ly].shape" , acts[ly].shape)
                np_img_act = acts[ly][batch,:].cpu().detach().numpy()

                np_img_act = cv2.resize(np_img_act, (512,512))
                np_img_act -= np.min(np_img_act)
                np_img_act /= np.max(np_img_act)

                heat = cv2.applyColorMap(np.uint8(255*np_img_act), cv2.COLORMAP_JET)
                heat = np.float32(heat) /255

                img_cam = np.float32(np_img) + heat
                img_cam = img_cam / np.max(img_cam)
                
                full_image[:, 512*(ly+1):512*(ly+2), ...] = img_cam * 255
            
            label_name = labels[batch]
            pred_name = preds[batch]

            if (label_name == 1) & (pred_name == 1):
                TP_path = os.path.join(img_dir, 'tp')
                pathlib.Path(TP_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(TP_path,'{}_full_{}.jpg'.format(iter_, batch)), full_image)
            elif (label_name == 1) & (pred_name == 0):
                FN_path = os.path.join(img_dir, 'fn')
                pathlib.Path(FN_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(FN_path,'{}_full_{}.jpg'.format(iter_, batch)), full_image)
            elif (label_name == 0) & (pred_name == 0):
                TN_path = os.path.join(img_dir, 'tn')
                pathlib.Path(TN_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(TN_path,'{}_full_{}.jpg'.format(iter_, batch)), full_image)
            else:
                FP_path = os.path.join(img_dir, 'fp')
                pathlib.Path(FP_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(FP_path,'{}_full_{}.jpg'.format(iter_, batch)), full_image)

import os
import glob
import json
import pandas as pd
import random

def split_data(path, train_ratio=0.8):
    shuffled_list = os.listdir(path)
    random.shuffle(shuffled_list)
    len_shuffled_list = len(shuffled_list)
    train_list = shuffled_list[:int(len_shuffled_list*train_ratio)]
    test_list = shuffled_list[int(len_shuffled_list*train_ratio):]

    return train_list, test_list

def make_dict(normal_path, abnormal_path, normal_list, abnormal_list):
    res_dict = {'imgs':[], 'labels':[]}
    for i in range(len(abnormal_list)):
#         print("normal_path, normal_list[i]" , normal_path, normal_list[i])
        f_path = os.path.join(normal_path, normal_list[i])
        res_dict['imgs'].append(f_path)
        res_dict['labels'].append(0)
        # print("abnormal_path" , abnormal_path)
        # print("abnormal_list" , abnormal_list)
        f_path = os.path.join(abnormal_path, abnormal_list[i])
        res_dict['imgs'].append(f_path)
        res_dict['labels'].append(1)
    
    return res_dict

def make_dict_2(normal_path, abnormal_path, normal_list, abnormal_list):
    res_dict = {'imgs':[], 'labels':[]}

    for i in range(len(abnormal_list)):
        f_path = os.path.join(normal_path, normal_list[i])
        res_dict['imgs'].append(f_path)
        res_dict['labels'].append(0)

        f_path = os.path.join(abnormal_path, abnormal_list[i])
        res_dict['imgs'].append(f_path)
        res_dict['labels'].append(1)
    
    return res_dict

def check_data(_dict, name):
    len_normal = len([i for i in _dict['labels'] if i==0])
    len_abnormal = len([i for i in _dict['labels'] if i==1])
    
    print('{} data \n * normal: {}\n * abnormal: {}'.format(name, len_normal, len_abnormal))


data_train_dict = {'imgs':[], 'labels':[]}
data_test_dict = {'imgs':[], 'labels':[]}

train_normal_path = '/mnt/nas100_vol2/Kyungjin.Cho/_imagetoimage/07_nodule_generation/DAGAN/DAGAN_v1/datasets/png/normal/train'
train_abnormal_path = '/mnt/nas100_vol2/Kyungjin.Cho/_imagetoimage/07_nodule_generation/DAGAN/DAGAN_v1/datasets/png/real_nodule/train'
test_normal_path = '/mnt/nas100_vol2/Kyungjin.Cho/_imagetoimage/07_nodule_generation/DAGAN/DAGAN_v1/datasets/png/normal/test'
test_abnormal_path = '/mnt/nas100_vol2/Kyungjin.Cho/_imagetoimage/07_nodule_generation/DAGAN/DAGAN_v1/datasets/png/real_nodule/test'

# test_normal_path = '/mnt/nas100_vol2/Kyungjin.Cho/_imagetoimage/data/SNU/normal'
# test_abnormal_path = '/mnt/nas100_vol2/Kyungjin.Cho/_imagetoimage/data/SNU/abnormal'

# pneumothorax_path = '/workspace/datasets/16_chest_disease/pneumothorax/DCM'
# pleural_effusion_path = '/workspace/datasets/16_chest_disease/pleural_effusion/DCM'

# train_normal, test_normal = split_data(normal_path)
# train_abnormal, test_abnormal = split_data(abnormal_path)
# train_pneumothorax, test_pneumothorax = split_data(pneumothorax_path)
# train_effusion, test_effusion = split_data(pleural_effusion_path)

train_normal = os.listdir('/mnt/nas100_vol2/Kyungjin.Cho/_imagetoimage/07_nodule_generation/DAGAN/DAGAN_v1/datasets/png/normal/train/')
train_abnormal = os.listdir('/mnt/nas100_vol2/Kyungjin.Cho/_imagetoimage/07_nodule_generation/DAGAN/DAGAN_v1/datasets/png/real_nodule/train/')

test_normal = os.listdir('/mnt/nas100_vol2/Kyungjin.Cho/_imagetoimage/07_nodule_generation/DAGAN/DAGAN_v1/datasets/png/normal/test')
test_abnormal = os.listdir('/mnt/nas100_vol2/Kyungjin.Cho/_imagetoimage/07_nodule_generation/DAGAN/DAGAN_v1/datasets/png/normal/test')

# test_normal = os.listdir('/mnt/nas100_vol2/Kyungjin.Cho/_imagetoimage/data/SNU/normal')
# test_abnormal = os.listdir('/mnt/nas100_vol2/Kyungjin.Cho/_imagetoimage/data/SNU/abnormal')

train_normal_abnormal_dict = make_dict(train_normal_path, train_abnormal_path, train_normal, train_abnormal)
test_normal_abnormal_dict = make_dict(test_normal_path, test_abnormal_path, test_normal, test_abnormal)

# with open('train_normal_abnormal_data_2.json', 'w') as f:
#     json.dump(train_normal_abnormal_dict, f, indent=4)
# with open('test_normal_abnormal_data_2.json', 'w') as f:
#     json.dump(test_normal_abnormal_dict, f, indent=4)
    
# with open('SNU_npy_test_normal_abnormal_data_2.json', 'w') as f:
#     json.dump(test_normal_abnormal_dict, f, indent=4)

# train_normal_pneumo_dict = make_dict(normal_path, pneumothorax_path, train_normal, train_pneumothorax)
# test_normal_pneumo_dict = make_dict(normal_path, pneumothorax_path, test_normal, test_pneumothorax)

# with open('train_normal_pneumothorax_data.json', 'w') as f:
#     json.dump(train_normal_pneumo_dict, f, indent=4)
# with open('test_normal_pneumothorax_data.json', 'w') as f:
#     json.dump(test_normal_pneumo_dict, f, indent=4)

# train_normal_effusion_dict = make_dict(normal_path, pleural_effusion_path, train_normal, train_effusion)
# test_normal_effusion_dict = make_dict(normal_path, pleural_effusion_path, test_normal, test_effusion)

with open('train_normal_nodule.json', 'w') as f:
    json.dump(train_normal_abnormal_dict, f, indent=4)
with open('test_normal_nodule.json', 'w') as f:
    json.dump(test_normal_abnormal_dict, f, indent=4)


# check_data(train_normal_abnormal_dict, 'train_normal_abnormal')
# check_data(test_normal_abnormal_dict, 'test_normal_abnormal')

# check_data(train_normal_pneumo_dict, 'train_normal_pneumothorax')
# check_data(test_normal_pneumo_dict, 'test_normal_pneumothorax')

# check_data(train_normal_effusion_dict, 'train_normal_pleuraleffusion')
# check_data(test_normal_effusion_dict, 'test_normal_pleuraleffusion')

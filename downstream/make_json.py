import os
import sys
import numpy as np
import random
import glob
import natsort
import cv2
import tqdm
import random
import pandas as pd
import json


def make_dict(csv_path , whole_img_list):
    
    data = pd.read_csv(csv_path)
    res_dict = {'imgs':[], 'labels':[]}

    for i in tqdm.tqdm(range(len(data))):
        row = data.iloc[i]
        if len(row) < 14:
            labels = [0] * 14
        else:
            labels = []
            for col in row[5:]:
                if col == 1 or col == -1:
                    labels.append(1)
                else:
                    labels.append(0)
        img_name = data.iloc[i]['Path'].split('/')[-3] + '_' + data.iloc[i]['Path'].split('/')[-2] + '_' + data.iloc[i]['Path'].split('/')[-1].split('.')[0] + '.png'
        matching_img_path = [s for s in whole_img_list if img_name in s]
        
        try:
            res_dict['imgs'].append(matching_img_path[0])
            res_dict['labels'].append((labels))
        except:
            print('[*] Error file ' , matching_img_path , img_name )
    return res_dict


def main():

    open_dataset = 'CheXpert-v1.0'
    phase = 'valid'
    root_dir = '/mnt/nas107/open_dataset'
    data_dtype = 'png'
    img_size = '512'

    csv_path = '{}/{}/{}.csv'.format(root_dir, open_dataset, phase)

    img_folder_path_ = natsort.natsorted(glob.glob('{}/{}/preprocessed/{}/{}/{}/*'.format(root_dir, open_dataset, data_dtype, phase, img_size )))
    
    whole_img_list = []

    for i in tqdm.tqdm(range(len(img_folder_path_))):
        in_dir_img_list = natsort.natsorted(glob.glob(img_folder_path_[i] + '/*.png'))
        for idx in range(len(in_dir_img_list)):
            whole_img_list.append(in_dir_img_list[idx])
    
    train_dict = make_dict(csv_path , whole_img_list)
    json_name = './json/{}_chexPert_16_{}.json'.format(phase, img_size)
    with open(json_name, 'w') as f:
        json.dump(train_dict, f, indent=4)

if __name__ == '__main__':
    main()

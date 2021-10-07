

## 5 subclass volo aug oversampling CE
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --msg=Volo_oversampling_lr_aug_normalize_setting_momentum \
--print_freq=10 --w 6 --batch_size 48 --img_size 512 --num_class 5 --backbone volo --aug True \
--train_path ./json/train_normal_5_sub_class_data_revise_oversampling.json \
--test_path ./json/test_normal_5_sub_class_data_revise.json \
--val_path ./json/val_normal_5_sub_class_data_revise.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation.py --msg=test_volo_d1_5_sub_class_lr_aug_normalize_setting_momentum \
--print_freq=10 --w 6 --batch_size 4 --backbone volo --img_size 512 --aug True --num_class 5 \
--train_path ./json/train_normal_5_sub_class_data_revise_oversampling.json \
--test_path ./json/test_normal_5_sub_class_data_revise.json --resume \
--pretrained ./checkpoints/2021-07-05_003711_Volo_oversampling_lr_aug_normalize_setting_momentum/best.pth.tar
# [*] Test Acc: 47.727273
# specificity:  [0.8957346  0.78199052 0.93838863 0.86997636 0.86052009]
# sensitivity (recall):  [0.82075472 0.61320755 0.19811321 0.26666667 0.48571429]
# positive predictive value (precision):  [0.66412214 0.41401274 0.44680851 0.3373494  0.46363636]
# negative predictive value:  [0.95214106 0.88948787 0.82328482 0.82696629 0.8708134 ]
# acc:  [0.88068182 0.74810606 0.78977273 0.75       0.78598485]
# f1_score:  [0.73417722 0.49429658 0.2745098  0.29787234 0.4744186 ]
# auc 0.78

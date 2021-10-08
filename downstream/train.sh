

CUDA_VISIBLE_DEVICES=3 python train.py --msg=test \
--print_freq=10 --w 6 --batch_size 4 --img_size 512 --num_class 14 \
--backbone resnet50 --aug True \
--downstream_name cheXpert \
--csv_file ./csv/valid.csv \
--train_path ./json/train_chexPert_16_512.json \
--val_path ./json/valid_chexPert_16_512.json \
--test_path ./json/test_chexPert_16_512.json

CUDA_VISIBLE_DEVICES=3 python evaluation.py --msg=test \
--print_freq=10 --w 6 --batch_size 1 --backbone resnet50 --img_size 512 \
--aug True --num_class 14 --downstream_name cheXpert \
--csv_file ./csv/valid.csv \
--test_path ./json/test_chexPert_16_512.json
## model training
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --mlp --moco-t 0.2 --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 /mnt/nas107

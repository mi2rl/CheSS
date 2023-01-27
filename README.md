# CheSS: Chest X-ray pre-trained model via Self-Supervised contrastive learning

This is a PyTorch implementation of the [CheSS paper](https://link.springer.com/article/10.1007/s10278-023-00782-4):

```
@article{Cho2023,
  doi = {10.1007/s10278-023-00782-4},
  url = {https://doi.org/10.1007/s10278-023-00782-4},
  year = {2023},
  month = jan,
  publisher = {Springer Science and Business Media {LLC}},
  author = {Kyungjin Cho and Ki Duk Kim and Yujin Nam and Jiheon Jeong and Jeeyoung Kim and Changyong Choi and Soyoung Lee and Jun Soo Lee and Seoyeon Woo and Gil-Sun Hong and Joon Beom Seo and Namkug Kim},
  title = {{CheSS}: Chest X-Ray Pre-trained Model via Self-supervised Contrastive Learning},
  journal = {Journal of Digital Imaging}
}
```

## Requirements

```
pip install -r requirements.txt
```

<img width="1275" alt="Figure" src="https://user-images.githubusercontent.com/108312461/215047851-77a46c0c-9392-4ad0-b71e-eef84fd6cf6f.png">


## Pretrained model weight
[Google Drive](https://drive.google.com/file/d/1C_Gis2qcZcA9X3l2NEHR1oS4Gn_bTxTe/view?usp=share_link)


```python
model = resnet50(num_classes=1000)

pretrained_model = "CheSS pretrained model path"
if pretrained_model is not None:
    if os.path.isfile(pretrained_model):
        print("=> loading checkpoint '{}'".format(pretrained_model))
        checkpoint = torch.load(pretrained_model, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrained_model))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained_model))

    ##freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
            
    model.fc = nn.Linear(2048, num_class)

```

or you can use gdown in Python

```python
!gdown https://drive.google.com/uc?id=1C_Gis2qcZcA9X3l2NEHR1oS4Gn_bTxTe
```

## Contact

<img width="1275" alt="mi2rl" src="https://user-images.githubusercontent.com/108312461/212851640-3e52332d-5346-4c1a-ab32-e337854afe71.png">

Page: https://mi2rl.co 

Email: kjcho@amc.seoul.kr

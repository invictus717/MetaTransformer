# Meta-Transformer for Image Segmentation

Our segmentation code is developed on top of [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2), and [ViT-Adapter](https://arxiv.org/abs/2205.08534).

If the code is helpful for your research, please kindly cite:

```
@article{zhang2023metatransformer,
  title={Meta-Transformer: A Unified Framework for Multimodal Learning}, 
  author={Zhang, Yiyuan and Gong, Kaixiong and Zhang, Kaipeng and Li, Hongsheng and Qiao, Yu and Ouyang, Wanli and Yue, Xiangyu},
  year={2023},
  journal={arXiv preprint arXiv:2307.10802},
}

@article{chen2022vitadapter,
  title={Vision Transformer Adapter for Dense Predictions},
  author={Chen, Zhe and Duan, Yuchen and Wang, Wenhai and He, Junjun and Lu, Tong and Dai, Jifeng and Qiao, Yu},
  journal={arXiv preprint arXiv:2205.08534},
  year={2022}
}
```

## Usage

Install [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).

*We recommended environment: torch1.9 + cuda11.1 for image segmentation and detection tasks.* 
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmengine
pip install mmsegmentation==0.20.2
cd ops & sh make.sh # compile deformable attention
```

## Data Preparation

Preparing ADE20K/Cityscapes/COCO Stuff/Pascal Context according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.

```
wget -c http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
wget -c http://data.csail.mit.edu/places/ADEchallenge/release_test.zip
```

```none
segmentation
├── configs
├── mmcv_custom
├── mmseg_custom
├── data
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
```
*You can also modify the VARIABLE of `data_root` in configs/_base_/datasets/ade20k.py* 
## Training

To train ViT-Adapter-B + UperNet on ADE20k on a single node with 8 gpus run:

```shell
bash dist_train.sh configs/ade20k/upernet_meta_transformer_base_512_160k_ade20k.py 8
```


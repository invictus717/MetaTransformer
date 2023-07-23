# Meta-Transformer for Object Detection

Our segmentation code is developed on top of [MMDetection v2.22.0](https://github.com/open-mmlab/mmdetection/tree/v2.22.0), and [ViT-Adapter](https://arxiv.org/abs/2205.08534).

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

Install [MMDetection v2.22.0](https://github.com/open-mmlab/mmdetection/tree/v2.22.0).

*We recommended environment: torch1.9 + cuda11.1 for image segmentation and detection tasks.* 

```
# recommended environment: torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmengine
pip install mmdet==2.22.0
pip install instaboostfast # for htc++
cd ops & sh make.sh # compile deformable attention
```

## Data Preparation

Prepare COCO according to the guidelines in [MMDetection v2.22.0](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).


## Training

For example:

```shell
bash dist_train.sh configs/cascade_rcnn/cascade_mask_rcnn_meta_transformer_adapter_base_fpn_3x_coco.py 8
```

## Image Demo & Video Demo

Please refer to [issue#23](https://github.com/czczup/ViT-Adapter/issues/23).

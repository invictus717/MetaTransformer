# Meta-Transformer for Image Understanding

This part of code is used for image understanding, and we provide the pretrained checkpoints.

## Environment Setup 

*We have provided the yaml file for a quick start*
```bash
conda env create -f environment.yaml
conda activate mt-seg
```

## Image Classification
<div>

|      Model      |   ImageNet-1K Acc(%)   | Scale | #Param |                                               Download  | 国内下载源                                              |
| :------------: | :----------: | :----------------------: | :----: | :-------------: | :---------------------------------------------------------------------------------------------------: |
| Image_Meta-Transformer-B16  | 85.4  |         Base          |  85M  |   [ckpt](https://drive.google.com/file/d/1YEJ_r5w6N61Fhau55x_f1YOE-QWhSpmh/view?usp=drive_link)    | [ckpt](https://download.openxlab.org.cn/models/zhangyiyuan/MetaTransformer/weight//Image_Meta-Transformer_base_patch16)
| Image_Meta-Transformer-L14  | 88.1 |         Large          |  302M  |   [ckpt](https://drive.google.com/file/d/1EJf4RYA0vl3lt-H1UFbAUjBrrLf-FwNq/view?usp=drive_link)   | [ckpt](https://download.openxlab.org.cn/models/zhangyiyuan/MetaTransformer/weight//Image_Meta-Transformer_large_patch14)

</div>

## Object Detection

For object detection, we use the checkpoints pretraiend checkpoints for the downstream tasks. More details can be found in [this](https://github.com/invictus717/MetaTransformer/blob/master/Image/detection/README.md)

## Image Segmentation

The document for image segmentation can be found [here](https://github.com/invictus717/MetaTransformer/blob/master/Image/segmentation/README.md)
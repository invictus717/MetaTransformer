# Meta-Transformer for Point Cloud Understanding

This part of code is for point cloud understanidng with Meta-Transfomrer. We aim to provide a powerful baseline to handle various point cloud understanding tasks. Our code is developed based on [PointNeXt](https://github.com/guochengqian/PointNeXt). 

## ✨✨✨ Features

* 🌟Simply utilize your foudation model on point cloud understanding. For example, now you can employ visual models such as ResNet-50 or ViT. And We will illustreate how to employ language models for point cloud understanding at [PointLanguage](https://github.com/invictus717/PointLanguage) .

    ```yaml
    model:
    NAME: BaseCls / BaseSeg  # For classification or Segmentation
    encoder_args:
        NAME: Your-ENCODER     # Your foundation model
    cls_args:
        NAME: ClsHead / SegHead # Task specific Head
        num_classes: 15
        in_channels: 1024
        mlps: [512,256]
        norm_args: 
        norm: 'bn1d'
    ```
* 🌟 Support a series of tasks including shape classification, scene segmentation, and object part segmentation.

    |      Model      |   Dataset   |    
    | :----------------------: | :---------------------------------------------------------------------------------------------------: |
    | Shape Classification  | ModelNet-40  |      
    | Shape Classification  | ScanObjectNN  |      
    | Scene Segmentation  | S3DIS  |  
    | Scene Segmentation  | ScanNet  |  
    | Object Segmentation  | ShapeNetPart  |  

    *Note that we will also release the implementation with Meta-Transformer for outdoor LiDAR scene perception tasks very soon.*
* 🌟 Easy to visualize. This code base provides rendering and visualization codes for basic 3D objects and scenes.
    <p align="center" width="100%">
    <img src="..\assets\Meta-Transformer_vis.png"  width="100%" height="100%">
    </p> 
* 🌟 Multiple Embeddings. We emsemble the two mainstream approaches ( Point Patch and Geometric Projection ) together. We further develop this code base with multi-view prompt learning with visual models.

## Usage

### 1. Environment Setup

```
conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html

pip install -r requirements.txt

cd openpoints/cpp/pointnet2_batch && python setup.py install

cd ../chamfer_dist && python setup.py install --user

cd ../emd && python setup.py install --user
```

### 2. Prepare Data

We following the document of [OpenPoints](https://guochengqian.github.io/PointNeXt/examples/scanobjectnn/) to prepare the dataset. They also provide preprocessed dataset by Google Drive. After that, you can organize the foloder as follows:

```none
PointCloud(Indoor)
├── scripts
├── examples
├── openpoints
├── cfgs
│   ├── modelnet40ply2048
│   ├── s3dis
│   ├── scanobjectnn
│   ├── scannet
│   ├── shapenetpart
├── data
│   ├── ShapeNetPart
│   ├── ├── shapenetcore_partanno_segmentation_benchmark_v0_normal
│   ├── ScanObjectNN
│   ├── ├── h5_files
│   ├── ScanNet
│   ├── ├── train
│   ├── ├── val
│   ├── ├── test
│   ├── ├── processed
│   ├── S3DIS
│   ├── ├── s3disfull
│   ├── ModelNet40Ply2048
│   ├── ├── modelnet40_normal_resampled

```

### 3. Train Model

To make the code easier to use, we provide training scripts:

- For Shape Clasification on the ModelNet-40 dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/metatransformer.yaml
```

- For Shape Clasification on the ScanObjectNN dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/metatransformer.yaml
```
- For Scene Segmentation on the S3DIS dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/s3dis/metatransformer.yaml
```

- For Scene Segmentation on the ScanNet dataset:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/segmentation/main.py --cfg cfgs/scannet/metatransformer.yaml 
```

- For Object Segmentation on the ShapeNetPart dataset:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/metatransformer.yaml
```

*Note that we use the NVIDIA A100 GPU for experiments. Therefore, please according to your device to set hyper-parameters.*


## Citation

If the code and paper are helpful for your research, please kindly cite:

```
@article{zhang2023metatransformer,
    title={Meta-Transformer: A Unified Framework for Multimodal Learning}, 
    author={Zhang, Yiyuan and Gong, Kaixiong and Zhang, Kaipeng and Li, Hongsheng and Qiao, Yu and Ouyang, Wanli and Yue, Xiangyu},
    year={2023},
    journal={arXiv preprint arXiv:2307.10802},
}

@InProceedings{qian2022pointnext,
  title   = {PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies},
  author  = {Qian, Guocheng and Li, Yuchen and Peng, Houwen and Mai, Jinjie and Hammoud, Hasan and Elhoseiny, Mohamed and Ghanem, Bernard},
  booktitle=Advances in Neural Information Processing Systems (NeurIPS),
  year    = {2022},
}

@article{peng2023multi,
  title={Multi-view Vision-Prompt Fusion Network: Can 2D Pre-trained Model Boost 3D Point Cloud Data-scarce Learning?},
  author={Peng, Haoyang and Li, Baopu and Zhang, Bo and Chen, Xin and Chen, Tao and Zhu, Hongyuan},
  journal={arXiv preprint arXiv:2304.10224},
  year={2023}
}
```

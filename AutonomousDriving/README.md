# Meta-Transformer for Autonomous Driving (AD)

This part of code is for Autonomous Driving (AD) with Meta-Transfomrer. We aim to provide a powerful baseline to handle joint-dataset training on autonomous driving community. Our code is developed based on [3DTrans](https://github.com/PJLab-ADG/3DTrans). 

## ✨✨✨ Features

* 🌟Simply utilize your foudation model on autonomous driving task. For example, now you can employ the pre-trained backbone parameters such as VoxelResBackBone8x (ResNet structure on 3D) or ViT to handle the joint-dataset training. And We will illustreate how to employ the pre-trained backbone parameters for autonomous driving task.


## 🔓 Model Zoo

🚀 🚀 🚀 Powerful Backbone for 3D LiDAR-based Object Detection!
  
[VoxelResBackBone8x: Jointly trained on Waymo and nuScenes](https://drive.google.com/file/d/1CveuI5XU4drfEatQxS-_hV744XqbyPDK/view).

[VoxelResBackBone8x: Pre-trained on ONCE dataset (1M point data)](https://drive.google.com/file/d/1MG7rZu19oFHi2fZs4xA_Ts1tMzPV8yEi/view?usp=drive_link).


## :muscle: TODO List :muscle:
- [ ] Add the pretraining checkpoint and usage method of ViT-based autonomous driving models.

## Installation for Meta-Transformer on Autonomous Driving

You may refer to [INSTALL.md](docs/INSTALL.md) for performing the installation process.


## Usage

### Problem Definition

The purpose of joint-dataset training on autonomous is to train a unified model from multiple labeled autonomous driving-related domains $s_i$, to obtain more generalizable representations $F$, which would have minimum prediction error on the multiple different domains $s_i$.


### Joint-training stage: train a unified backbone on multiple benchmarks
* Train with consistent point-cloud range (employing Waymo range) using multiple GPUs
* Note that the multi-dataset 3D detection results will become very poor, if the point-cloud-range for different datasets is not aligned.

* Train with Domain Attention (DT) method

```shell script
cd tools
```

```shell script
sh scripts/dist_train_mdf.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_pvrcnn_feat_3_domain_attention.yaml \
--source_one_name waymo
```

* Train with Domain Attention (DT) using multiple machines
```shell script
sh scripts/slurm_train_multi_db.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_pvrcnn_feat_3_domain_attention.yaml \
--source_one_name waymo
```

* Train other baseline detectors such as PV-RCNN++ using multiple GPUs
```shell script
sh scripts/dist_train_mdf.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_pvplus_feat_3_domain_attention.yaml \
--source_one_name waymo
```

* Train other baseline detectors such as Voxel-RCNN using multiple GPUs
```shell script
sh scripts/dist_train_mdf.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_voxel_rcnn_feat_3_domain_attention.yaml \
--source_one_name waymo
```

### Training using another Backbone:
* Train using multiple GPUs
```shell script
sh scripts/dist_train_mdf.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_voxel_rcnn_resnet_feat_3_domain_attention.yaml \
--source_one_name waymo
```

* Train multiple machines
```shell script
sh scripts/slurm_train_multi_db.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_voxel_rcnn_resnet_feat_3_domain_attention.yaml \
--source_one_name waymo
```


### Evaluation stage: evaluate the unified detection model on different datasets:
* Note that for the KITTI-related evaluation, please try --set DATA_CONFIG.FOV_POINTS_ONLY True to enable front view point cloud only. We report the best results on KITTI for testing all epochs on the validation set.

    - ${FIRST_DB_NAME} denotes that the fisrt dataset name of the merged two dataset, which is used to split the merged dataset into two individual datasets.

    - ${DB_SOURCE} denotes the dataset to be tested.


* Test the models using multiple GPUs
```shell script
sh scripts/dist_test_mdf.sh ${NUM_GPUs} \
--cfg_file ${CFG_FILE} \
--ckpt ${CKPT} \
--source_one_name ${FIRST_DB_NAME} \
--source_1 ${DB_SOURCE} 
```

* Test the models using multiple machines
```shell script
sh scripts/slurm_test_mdb_mgpu.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ${CFG_FILE} \
--ckpt ${CKPT} \
--source_one_name ${FIRST_DB_NAME} \
--source_1 ${DB_SOURCE} 
```

&ensp;
## Visualization Tools for Autonomous Driving Task

- Our repository supports the sequence-level visualization function [Quick Sequence Demo](docs/QUICK_SEQUENCE_DEMO.md) to continuously display the prediction results of ground truth of a selected scene.

- **Visualization Demo**: 
  - [Waymo Sequence-level Visualization Demo1](docs/seq_demo_waymo_bev.gif)
  - [Waymo Sequence-level Visualization Demo2](docs/seq_demo_waymo_fp.gif)
  - [nuScenes Sequence-level Visualization Demo](docs/seq_demo_nusc.gif)
  - [ONCE Sequence-level Visualization Demo](docs/seq_demo_once.gif)


## Citation
If you find this project useful in your research, please consider citing:
```
@article{zhang2023metatransformer,
    title={Meta-Transformer: A Unified Framework for Multimodal Learning}, 
    author={Zhang, Yiyuan and Gong, Kaixiong and Zhang, Kaipeng and Li, Hongsheng and Qiao, Yu and Ouyang, Wanli and Yue, Xiangyu},
    year={2023},
    journal={arXiv preprint arXiv:2307.10802},
}
```

```
@article{yuan2023ad,
  title={AD-PT: Autonomous Driving Pre-Training with Large-scale Point Cloud Dataset},
  author={Yuan, Jiakang and Zhang, Bo and Yan, Xiangchao and Chen, Tao and Shi, Botian and Li, Yikang and Qiao, Yu},
  journal={arXiv preprint arXiv:2306.00612},
  year={2023}
}
```

```
@inproceedings{zhang2023uni3d,
  title={Uni3d: A unified baseline for multi-dataset 3d object detection},
  author={Zhang, Bo and Yuan, Jiakang and Shi, Botian and Chen, Tao and Li, Yikang and Qiao, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9253--9262},
  year={2023}
}
```
# OpenPoints

OpenPoints is a library built for fairly benchmarking and easily reproducing point-based methods for point cloud understanding. It is born in the course of [PointNeXt](https://github.com/guochengqian/PointNeXt) project and is used as an engine therein.

**For any question related to OpenPoints, please open an issue in [PointNeXt](https://github.com/guochengqian/PointNeXt) repo.**

OpenPoints currently supports reproducing the following models:
- PointNet
- DGCNN
- DeepGCN
- PointNet++
- ASSANet
- PointMLP
- PointNeXt
- Pix4Point



## Features

1. **Extensibility**: supports many representative networks for point cloud understanding, such as *PointNet, DGCNN, DeepGCN, PointNet++, ASSANet, PointMLP*, and our ***PointNeXt***. More networks can be built easily based on our framework since **OpenPoints support a wide range of basic operations including graph convolutions, self-attention, farthest point sampling, ball query, *e.t.c***.

2. **Ease of Use**: *Build* model, optimizer, scheduler, loss function,  and data loader *easily from cfg*. Train and validate different models on various tasks by simply changing the `cfg\*\*.yaml` file. 

      ```
   model = build_model_from_cfg(cfg.model)
   criterion = build_criterion_from_cfg(cfg.criterion_args)
   ```



## Usage

OpenPoints only serves as an engine. Please refer to [PointNeXt](https://github.com/guochengqian/PointNeXt) for a specific example of how to use and install



## Citation

If you use this library, please kindly acknowledge our work:
```tex
@Article{qian2022pointnext,
  author  = {Qian, Guocheng and Li, Yuchen and Peng, Houwen and Mai, Jinjie and Hammoud, Hasan and Elhoseiny, Mohamed and Ghanem, Bernard},
  title   = {PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies},
  journal = {arXiv:2206.04670},
  year    = {2022},
}
```


# Meta-Transformer for Video Recognition

This part of code is based on [VideoMAE](https://github.com/OpenGVLab/VideoMAEv2). Thanks for their outstanding project.
## Usage

### 1. Environment Setup.

```
pip install -r requirements.txt
```

###  2. Prepare Data. 
*You can easily prepare data with OpenData Lab by running commands below*
```
pip install openxlab
openxlab dataset get --dataset-repo OpenMMLab/Kinetics-400
```

### 3. Train and evaluate model. 
We provide the experiment scripts for easier use:

* Kinetics-400 Dataset
```
bash run.sh
```
*Please edit `run.sh` before running the code.*

## Citation

If the code and paper are helpful for your research, please kindly cite:

```
@article{zhang2023metatransformer,
    title={Meta-Transformer: A Unified Framework for Multimodal Learning}, 
    author={Zhang, Yiyuan and Gong, Kaixiong and Zhang, Kaipeng and Li, Hongsheng and Qiao, Yu and Ouyang, Wanli and Yue, Xiangyu},
    year={2023},
    journal={arXiv preprint arXiv:2307.10802},
}

@misc{videomaev2,
      title={VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking},
      author={Limin Wang and Bingkun Huang and Zhiyu Zhao and Zhan Tong and Yinan He and Yi Wang and Yali Wang and Yu Qiao},
      year={2023},
      eprint={2303.16727},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
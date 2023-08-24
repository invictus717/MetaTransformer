# Meta-Transformer for Audio Understanding

This part of code is for audio data understanidng with Meta-Transfomrer. We conduct experiments on tabular data understanding based on [AST](https://github.com/YuanGongND/ast). Thanks for their outstanding projects.

## Citation

If the code and paper are helpful for your research, please kindly cite:

```
@article{zhang2023metatransformer,
    title={Meta-Transformer: A Unified Framework for Multimodal Learning}, 
    author={Zhang, Yiyuan and Gong, Kaixiong and Zhang, Kaipeng and Li, Hongsheng and Qiao, Yu and Ouyang, Wanli and Yue, Xiangyu},
    year={2023},
    journal={arXiv preprint arXiv:2307.10802},
}

@inproceedings{gong21b_interspeech,
  author={Yuan Gong and Yu-An Chung and James Glass},
  title={{AST: Audio Spectrogram Transformer}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={571--575},
  doi={10.21437/Interspeech.2021-698}
}
```

## Usage

### 1. Environment Setup

*We found that the experiment is compatible with point cloud understanding*

Please refer to our previous [doc](https://github.com/invictus717/MetaTransformer/tree/master/PointCloud#1-environment-setup)

```
pip install -r requirements.txt
```
### 2. Prepare Data

If you don't have the data, the code will download Speech Commands V2 directly.

After taht, we organize the structure of the code as follows:

```none
Audio
├── src
│   ├── models
│   ├── utilities
├── prep_sc.py
│
├── data

```

### 3. Train Model

To make the code easier to use, we provide training scripts to train models:

- For Speech Commands V2 dataset

```bash
bash run_sc.sh
```


### 4. Performance of Meta-Transformer

*Note that #Param denotes the **Trainable** parameters ihe the whole network.*
|      Model      |   Dataset   | Acc. | #Param | 
| :------------: | :----------: | :----------------------: | :----: |
| Meta-Transformer-B16  | Speech Commands V2  |         78.3          |  1.1M  |  
| Meta-Transformer-B16  | Speech Commands V2 |         97.0          |  86.3M  |  





# Meta-Transformer for Time-Series Forecasting

We conduct experiments on Time-Series Forecasting based on [Time-Series-Library](https://github.com/thuml/Time-Series-Library). Thanks for their oustanding project.
## Usage

### 1. Environment Setup.

```
pip install -r requirements.txt
```

###  2. Prepare Data. 
You can obtained the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/84fbc752d0e94980a610/) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy). Then place the downloaded data under the folder `dataset`. Here is a summary of supported datasets.

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

*We only conduct experiments on the ETT, Traffic, Weather, and Exchange datasets for long-term forecasting tasks*

### 3. Train and evaluate model. 
We provide the experiment scripts of all benchmarks under the folder `scripts`. You can reproduce the experiment results as the following examples:

For Long-term Forecasting tasks:
* ETTh1 Dataset
```
bash scripts/long_term_forecast/ETT_script/MetaTransformer_ETTh1.sh
```
* Traffic Dataset
```
bash scripts/long_term_forecast/Traffic_script/MetaTransformer.sh
```
* Weather Dataset
```
bash scripts/long_term_forecast/Weather_script/MetaTransformer.sh
```
* Exchange Dataset
```
bash scripts/long_term_forecast/Exchange_script/MetaTransformer.sh
```
### 4. Performance of Meta-Transformer.

|      Model      |   Dataset   | MAE | #Param |                                               Logs                                                |
| :------------: | :----------: | :----------------------: | :----: | :---------------------------------------------------------------------------------------------------: |
| Meta-Transformer-B16  | ETTh1  |         0.797          |  19K  |   [log](https://drive.google.com/file/d/16Qad-t_C4s1zqeIPTtskO71bHCl2uyZ-/view?usp=drive_link)    |
| Meta-Transformer-B16  | Traffic |         0.372          |  2M  |   [log](https://drive.google.com/file/d/16Qad-t_C4s1zqeIPTtskO71bHCl2uyZ-/view?usp=drive_link)   |
| Meta-Transformer-B16  | Weather |         0.640          |  51K  |   [log](https://drive.google.com/file/d/16Qad-t_C4s1zqeIPTtskO71bHCl2uyZ-/view?usp=drive_link)   |
| Meta-Transformer-B16  | Exchange |         0.961         |  22K  |   [log](https://drive.google.com/file/d/16Qad-t_C4s1zqeIPTtskO71bHCl2uyZ-/view?usp=drive_link)   |


## Citation

If the code and paper are helpful for your research, please kindly cite:

```
@article{zhang2023metatransformer,
    title={Meta-Transformer: A Unified Framework for Multimodal Learning}, 
    author={Zhang, Yiyuan and Gong, Kaixiong and Zhang, Kaipeng and Li, Hongsheng and Qiao, Yu and Ouyang, Wanli and Yue, Xiangyu},
    year={2023},
    journal={arXiv preprint arXiv:2307.10802},
}

@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```
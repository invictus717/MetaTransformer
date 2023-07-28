# Meta-Transformer for Hyper-Spectrum

This part of code is developed based on [Spectralformer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer), we sincerely appreciate their great contribution.

## Citation

If the code and paper are helpful for your research, please kindly cite:

```
@article{zhang2023metatransformer,
    title={Meta-Transformer: A Unified Framework for Multimodal Learning}, 
    author={Zhang, Yiyuan and Gong, Kaixiong and Zhang, Kaipeng and Li, Hongsheng and Qiao, Yu and Ouyang, Wanli and Yue, Xiangyu},
    year={2023},
    journal={arXiv preprint arXiv:2307.10802},
}

@article{hong2022spectralformer,
  title={Spectralformer: Rethinking hyperspectral image classification with transformers},
  author={Hong, Danfeng and Han, Zhu and Yao, Jing and Gao, Lianru and Zhang, Bing and Plaza, Antonio and Chanussot, Jocelyn},
  journal={IEEE Trans. Geosci. Remote Sens.},
  year={2022},
  volume={60},
  pages={1-15},
  note = {DOI: 10.1109/TGRS.2021.3130716}
}
```

## Usage

### 1. Environment Setup

```
pip install scipy einop gdown
```

### 2. Prepare Data

```bash
mkdir data && cd data
gdown 1dsTmdzmy_UT0ASQvyxflhiiHBdFA8g85
gdown 15PFzVS3lzkF43bbZhpV8hZ7q-Kx69_wg
```
After running the commands, we organize the structure of the code as follows:

```none
Hyper-spectrum
├── data
│   ├── IndianPine.mat
│   ├── AVIRIS_colormap.mat
├── train.py
├── metatransformer.py

```
### 3. Train and Evaluate Model

You can easily run the commands:

- For training:

```bash
bash scripts/train.sh
```

- For evaluation:

```bash
bash test.sh
```

### 4. Performance of Meta-Transformer

*Note that #Param denotes the **Trainable** parameters ihe the whole network.*
|      Model      |   Dataset   | OA/AA (%) | #Param |  Log |
| :------------: | :----------: | :----------------------: | :----: |:----: |
| Meta-Transformer-B16  | Indian Pine  |         67.62/78.09          |  0.17M  |  [log](https://drive.google.com/file/d/15uEXqYQNSaMkydD0JHyhzFIEtep6VF3P/view?usp=sharing)





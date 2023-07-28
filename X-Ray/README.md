
# Meta-Transformer for Time-Series Forecasting

We conduct experiments on chest X-ray images  based on [SEViT](https://github.com/faresmalik/SEViT/issues). We sincerely appreciate these open-source projects.

## Citation

If the code and paper are helpful for your research, please kindly cite:

```
@article{zhang2023metatransformer,
    title={Meta-Transformer: A Unified Framework for Multimodal Learning}, 
    author={Zhang, Yiyuan and Gong, Kaixiong and Zhang, Kaipeng and Li, Hongsheng and Qiao, Yu and Ouyang, Wanli and Yue, Xiangyu},
    year={2023},
    journal={arXiv preprint arXiv:2307.10802},
}

@inproceedings{almalik2022self,
  title={Self-Ensembling Vision Transformer (SEViT) for Robust Medical Image Classification},
  author={Almalik, Faris and Yaqub, Mohammad and Nandakumar, Karthik},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={376--386},
  year={2022},
  organization={Springer}
}
```

## Usage

### 1. Environment Setup

```
pip install -r requirements.txt
```

### 2. Prepare data

We use [Chest X-ray dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset), which includes 7000 images (3500 Normal images and 3500 Tuberculosis images). Download the splitted data from [here](https://drive.google.com/drive/folders/1XmdB37YowEHQTak2rU2iqyzHK8WBF7pO?usp=sharing). You can run the commands below:

```bash
pip install gdown
mkdir data && data
gdown 1XmdB37YowEHQTak2rU2iqyzHK8WBF7pO --folder
```
Then you can organize the code as follows:
```none
X-Ray
├── data
│   ├── training
│   ├── testing
│   ├── validation
├── requirements.txt
├── metatransformer.py
```

### 3. Train and evaluate model. 

```
bash train.sh
```

### 4. Performance of Meta-Transformer

|      Model      |   Dataset   | ACC. | #Param |             
| :------------: | :----------: | :----------------------: | :----: | 
| Meta-Transformer-B16  | Chest X-Ray  |         94.1          |  0.75M  | 



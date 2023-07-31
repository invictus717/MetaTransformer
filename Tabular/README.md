# Meta-Transformer for Tabular Data

This part of code is for tabular data understanidng with Meta-Transfomrer. We conduct experiments on tabular data understanding based on [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep), [tabulardl-benchmark](https://github.com/jrzaurin/tabulardl-benchmark), and [TabTransformer](https://arxiv.org/abs/2012.06678). Thanks for their outstanding projects.

## Citation

If the code and paper are helpful for your research, please kindly cite:

```
@article{zhang2023metatransformer,
    title={Meta-Transformer: A Unified Framework for Multimodal Learning}, 
    author={Zhang, Yiyuan and Gong, Kaixiong and Zhang, Kaipeng and Li, Hongsheng and Qiao, Yu and Ouyang, Wanli and Yue, Xiangyu},
    year={2023},
    journal={arXiv preprint arXiv:2307.10802},
}

@article{huang2020tabtransformer,
  title={Tabtransformer: Tabular data modeling using contextual embeddings},
  author={Huang, Xin and Khetan, Ashish and Cvitkovic, Milan and Karnin, Zohar},
  journal={arXiv preprint arXiv:2012.06678},
  year={2020}
}
```

## Usage

### 1. Environment Setup

```
pip install pytorch-widedeep
pip install timm==0.8.0.dev0
```

### 2. Prepare Data

The experiments are conducted on the [Adult Census](https://archive.ics.uci.edu/ml/datasets/adult) and  [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) datasets for tabular data binary classification. To prepare the data, you can easily run the following commands. The process will download the required data, organize it into the appropriate folders, and then execute the data preparation scripts to generate processed data in the "**processed_data**" folder.

```bash
bash get_data.sh
```
In additionm, you can also download the pre-processed data from google drive.
```bash
pip install gdown 
gdown 1weGsu9kk2q5KRf132EWHwOFT09os_X-7
```
You can obtain the `Dataset.zip`, then please simply unzip the file.

After running the commands, we organize the structure of the code as follows:

```none
Tabular
├── analyze_experiments
│   ├── leaderboards
├── prepare_datasets
│
├── processed_data
│
├── run_experiments
│   ├── adult
│   ├── bank_marketing
│   ├── general_utils
│   ├── parsers
│   ├── results
│   ├── run_adult_meta-transformer_experiments.sh
│   ├── run_bankm_meta-transformer_experiments.sh

```

### 3. Train Model

To make the code easier to use, we provide training scripts under `run_experiments` foloders to train models:

- For Adult Census dataset

```bash
bash run_adult_meta-transformer_experiments.sh
```

- For Bank Marketing dataset

```bash
bash run_bankm_meta-transformer_experiments.sh
```

### 4. Performance of Meta-Transformer

*Note that #Param denotes the **Trainable** parameters ihe the whole network.*
|      Model      |   Dataset   | Acc. | #Param | 
| :------------: | :----------: | :----------------------: | :----: |
| Meta-Transformer-B16  | Adult Census  |         85.9          |  19K  |  
| Meta-Transformer-B16  | Bank Marketing |         90.1          |  2M  |  





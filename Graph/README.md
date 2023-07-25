# Meta-Transformer for Graph Understanding

This part implementation for graph understanding is based on [fairseq](https://github.com/facebookresearch/fairseq) library and [TokenGT](https://github.com/jw9730/tokengt). We sincerely appreciate their great contributions.

If the code is helpful for your research, please kindly cite:

```
@article{zhang2023metatransformer,
  title={Meta-Transformer: A Unified Framework for Multimodal Learning}, 
  author={Zhang, Yiyuan and Gong, Kaixiong and Zhang, Kaipeng and Li, Hongsheng and Qiao, Yu and Ouyang, Wanli and Yue, Xiangyu},
  year={2023},
  journal={arXiv preprint arXiv:2307.10802},
}

@article{kim2022pure,
  author    = {Jinwoo Kim and Tien Dat Nguyen and Seonwoo Min and Sungjun Cho and Moontae Lee and Honglak Lee and Seunghoon Hong},
  title     = {Pure Transformers are Powerful Graph Learners},
  journal   = {arXiv},
  volume    = {abs/2207.02505},
  year      = {2022},
  url       = {https://arxiv.org/abs/2207.02505}
}
```

## Usage

*We recommended environment: torch1.9.1 + cuda11.1 for graph data prediction tasks.* 
```bash
### Basic Python Virtual Environment
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install lmdb
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-geometric==1.7.2
pip install tensorboardX==2.4.1
pip install ogb==1.3.2
pip install rdkit-pypi==2021.9.3
pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html
pip install performer-pytorch
pip install tensorboard
pip install setuptools==59.5.0

### Add FairSeq Library
git submodule update --init --recursive
cd fairseq && pip install .
python setup.py build_ext --inplace
```
*Note that the installation may be a little different from [TokenGT](https://github.com/jw9730/tokengt), but we've tested it.*

There is no need for manually preparing data, it can be direcly downloaded and pre-processed for the first time with the [fairseq](https://github.com/facebookresearch/fairseq) library.

## Train & Evaluate Models
* With a frozen backbone:
```bash
cd scripts && bash pcqv2-metatransformer_fixed.sh
```
* Or fully tuning the network:

*Note that there may be warning regarding gradients overflow.However, it does significantly affect the performance.*
```bash
cd scripts && bash pcqv2-metatransformer_finetune.sh
```

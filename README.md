<p align="center" width="100%">
<img src="assets\Meta-Transformer_banner.png"  width="80%" height="80%">
</p>

<div>
<div align="center">
    <a href='https://scholar.google.com/citations?user=KuYlJCIAAAAJ&hl=en' target='_blank'>Yiyuan Zhang<sup>1,2*</sup></a>&emsp;
    <a href='https://kxgong.github.io/' target='_blank'>Kaixiong Gong<sup>1,2*</sup></a>&emsp;
    <a href='http://kpzhang93.github.io/' target='_blank'>Kaipeng Zhang<sup>2,†</sup></a>&emsp;
    </br>
    <a href='http://www.ee.cuhk.edu.hk/~hsli/' target='_blank'>Hongsheng Li <sup>1,2</sup></a>&emsp;
    <a href='https://mmlab.siat.ac.cn/yuqiao/index.html' target='_blank'>Yu Qiao <sup>2</sup></a>&emsp;
    <a href='https://wlouyang.github.io/' target='_blank'>Wanli Ouyang<sup>2</sup></a>&emsp;
    <a href='http://people.eecs.berkeley.edu/~xyyue/' target='_blank'>Xiangyu Yue<sup>1,†,‡</sup></a>
</div>
<div>

<div align="center">
    <sup>1</sup>
    <a href='http://mmlab.ie.cuhk.edu.hk/' target='_blank'>Multimedia Lab, The Chinese University of Hong Kong</a>&emsp;
    </br>
    <sup>2</sup> <a href='https://github.com/OpenGVLab' target='_blank'>OpenGVLab，Shanghai AI Laboratory 
    </a></br>
    <sup>*</sup> Equal Contribution&emsp;
    <sup>†</sup> Corresponding Author&emsp;
    <sup>‡</sup> Project Lead&emsp;
</div>

-----------------

[![arXiv](https://img.shields.io/badge/arxiv-2307.10802-b31b1b?style=plastic&color=b31b1b&link=https%3A%2F%2Farxiv.org%2Fabs%2F2307.10802)](https://arxiv.org/abs/2307.10802)
[![website](https://img.shields.io/badge/Project-Website-brightgreen)](https://kxgong.github.io/meta_transformer/)
[![blog-cn](https://img.shields.io/badge/%E6%9C%BA%E5%99%A8%E4%B9%8B%E5%BF%83-%E7%AE%80%E4%BB%8B-brightgreen)](https://mp.weixin.qq.com/s/r38bzqdJxDZUvtDI0c9CEw)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/papers/2307.10802)
[![OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/zhangyiyuan/MetaTransformer)
![](https://img.shields.io/github/stars/invictus717/MetaTransformer?style=social)
<a href="https://twitter.com/_akhaliq/status/1682248055637041152"><img src="https://img.icons8.com/color/48/000000/twitter.png" width="25" height="25"></a>
<a href="https://www.youtube.com/watch?v=V8L8xbsTyls&ab_channel=CSBoard"><img src="https://img.icons8.com/color/48/000000/youtube-play.png" width="25" height="25"></a> <a href='https://huggingface.co/kxgong/Meta-Transformer'> <img src="assets\icons\huggingface.png" width="25" height="25"> </a> <a href='https://open.spotify.com/episode/6JJxcy2zMtTwr4jXPQEXjh'> <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg" width="20" height="20"></a>


## Meta-Transformer with Large Language Models ✨✨✨

We're thrilled to present [OneLLM](https://github.com/csuhan/OneLLM), ensembling Meta-Transformer framework with Multimodal Large Language Models, which performs multimodal joint training🚀, supports more modalities including fMRI, Depth and Normal Maps 🚀, and demonstrates very impressive performances on **25** benchmarks🚀🚀🚀. 

🔥🔥 The code, pretrained models, and datasets are publicly available at [OneLLM](https://github.com/csuhan/OneLLM).

🔥🔥 Project Website is at [OneLLM](https://onellm.csuhan.com/).

### 🌟 Single Foundation Model Supports A Wide Range of Applications



As a foundation model, Meta-Transformer can handle data from 12 modalities, which determines that it can support a wide range of applications. As shown in this figure, Meta-Transformer can provide services for downstream tasks including stock analysis 📈, weather forecasting ☀️ ☔ ☁️ ❄️ ⛄ ⚡, remote sensing 📡, autonomous driving 🚗, social network 🌍, speech recognition 🔉, etc.

<p align="center" width="100%">
<img src="assets\Meta-Transformer_application.png"  width="100%" height="100%">
</p>

**Table 1**: Meta-Transformer is capable of handling up to 12 modalities, including natural language <img src="assets\icons\text.jpg" width="15" height="15">, RGB images <img src="assets\icons\img.jpg" width="15" height="15">, point clouds <img src="assets\icons\pcd.jpg" width="15" height="15">, audios <img src="assets\icons\audio.jpg" width="15" height="15">, videos <img src="assets\icons\video.jpg" width="15" height="15">, tabular data <img src="assets\icons\table.jpg" width="15" height="15">, graph <img src="assets\icons\graph.jpg" width="15" height="15">, time series data <img src="assets\icons\time.jpg" width="15" height="15">, hyper-spectral images <img src="assets\icons\hyper.jpg" width="15" height="15">, IMU <img src="assets\icons\imu.jpg" width="15" height="15">, medical images <img src="assets\icons\xray.jpg" width="15" height="15">, and infrared images <img src="assets\icons\infrared.jpg" width="15" height="15">.
<p align="left">
<img src="assets\Meta-Transformer_cmp.png" width=100%>
</p>

## 🚩🚩🚩 Shared-Encoder, Unpaired Data, More Modalities 


<div>
  <img class="image" src="assets\Meta-Transformer_teaser.png" width="52%" height="100%">
  <img class="image" src="assets\Meta-Transformer_exp.png" width="45.2%" height="100%">
</div>


This repository is built to explore the potential and extensibility of transformers for multimodal learning. We utilize the advantages of Transformers to deal with length-variant sequences. Then we propose the *Data-to-Sequence* tokenization following a meta-scheme, then we apply it to 12 modalities including text, image, point cloud, audio, video, infrared, hyper-spectral, X-Ray, tabular, graph, time-series, and Inertial Measurement Unit (IMU) data.

<p align="left">
<img src="assets\Meta-Transformer_data2seq.png" width=100%>
</p>

After obtaining the token sequence, we employ a modality-shared encoder to extract representation across different modalities. With task-specific heads, Meta-Transformer can handle various tasks on the different modalities, such as: classification, detection, and segmentation.

<p align="left">
<img src="assets\Meta-Transformer_framework.png" width=100%>
</p>



# 🌟 News
* **2023.8.17:** Release code to directly get embeddings from multiple modalities. We will further release code on utilizing Meta-Transformer for Human-Centric vision tasks.
* **2023.8.2:** 🎉🎉🎉 The implementation of Meta-Transformer for image, point cloud, graph, tabular, time-series, X-Ray, hyper-spectrum, LiDAR data has been released. We also release a very powerful foundation model for Autonomous Driving 🚀🚀🚀.  
* **2023.7.22:** Pretrained weights and a usage demo for our Meta-Transformer have been released. Comprehensive documentation and implementation of the image modality are underway and will be released soon. Stay tuned for more exciting updates!⌛⌛⌛
* **2023.7.21:** Paper is released at [arxiv](https://arxiv.org/abs/2307.10802), and code will be gradually released.
* **2023.7.8:** Github Repository Initialization.

# 🔓 Model Zoo

<!-- <details> -->
<summary> Open-source Modality-Agnostic Models </summary>
<br>
<div>

|      Model      |   Pretraining   | Scale | #Param |                                               Download | 国内下载源                                               |
| :------------: | :----------: | :----------------------: | :----: | :---------------------------------------------------------------------------------------------------: | :--------: | 
| Meta-Transformer-B16  | LAION-2B |         Base          |  85M  |   [ckpt](https://drive.google.com/file/d/19ahcN2QKknkir_bayhTW5rucuAiX0OXq/view?usp=sharing)    | [ckpt](https://download.openxlab.org.cn/models/zhangyiyuan/MetaTransformer/weight//Meta-Transformer_base_patch16_encoder)
| Meta-Transformer-L14  | LAION-2B |         Large          |  302M  |   [ckpt](https://drive.google.com/file/d/15EtzCBAQSqmelhdLz6k880A19_RpcX9B/view?usp=drive_link)   | [ckpt](https://download.openxlab.org.cn/models/zhangyiyuan/MetaTransformer/weight//Meta-Transformer_large_patch14_encoder)

</div>

<!-- </details> -->

<!-- <details> -->
* Demo of Use for Pretrained Encoder

```python
import torch 
import torch.nn as nn
from timm.models.vision_transformer import Block
from Data2Seq import Data2Seq
video_tokenier = Data2Seq(modality='video',dim=768)
audio_tokenier = Data2Seq(modality='audio',dim=768)
time_series_tokenier = Data2Seq(modality='time-series',dim=768)

features = torch.concat([video_tokenizer(video), audio_tokenizer(audio), time_series_tokenizer(time_data)],dim=1)
# For base-scale encoder:
ckpt = torch.load("Meta-Transformer_base_patch16_encoder.pth")
encoder = nn.Sequential(*[
            Block(
                dim=768,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(12)])
encoder.load_state_dict(ckpt,strict=True)
# For large-scale encoder:
ckpt = torch.load("Meta-Transformer_large_patch14_encoder.pth")
encoder = nn.Sequential(*[
            Block(
                dim=1024,
                num_heads=16,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(24)])
encoder.load_state_dict(ckpt,strict=True)
encoded_features = encoder(features)
```
<!-- </details> -->

# 🕙 ToDo
- [ x ] Meta-Transformer with Large Language Models.
- [ x ] Multimodal Joint Training with Meta-Transformer.
- [ x ] Support More Modalities and More Tasks.

# Contact
🚀🚀🚀 We aspire to shape this repository into **a formidable foundation for mainstream AI perception tasks across diverse modalities**. Your contributions can play a significant role in this endeavor, and we warmly welcome your participation in our project!

To contact us, never hestitate to send an email to `yiyuanzhang.ai@gmail.com` ,`kaixionggong@gmail.com`, `zhangkaipeng@pjlab.org.cn`, or `xyyue@ie.cuhk.edu.hk`!
<br></br>

&ensp;
# Citation
If the code and paper help your research, please kindly cite:
```
@article{zhang2023meta,
  title={Meta-transformer: A unified framework for multimodal learning},
  author={Zhang, Yiyuan and Gong, Kaixiong and Zhang, Kaipeng and Li, Hongsheng and Qiao, Yu and Ouyang, Wanli and Yue, Xiangyu},
  journal={arXiv preprint arXiv:2307.10802},
  year={2023}
}
```
# License
This project is released under the [Apache 2.0 license](LICENSE).
# Acknowledgement
This code is developed based on excellent open-sourced projects including [MMClassification](https://github.com/open-mmlab/mmpretrain/tree/mmcls-1.x), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMsegmentation](https://github.com/open-mmlab/mmsegmentation), [OpenPoints](https://github.com/guochengqian/openpoints), [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [Graphomer](https://github.com/microsoft/Graphormer), [SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer), and [ViT-Adapter](https://github.com/czczup/ViT-Adapter).

<p align="center" width="100%">
<img src="assets\Meta-Transformer_banner.png"  width="80%" height="80%">
</p>

<div>
<div align="center">
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=KuYlJCIAAAAJ/' target='_blank'>Yiyuan Zhang<sup>1,2*</sup></a>&emsp;
    <a href='https://kxgong.github.io/' target='_blank'>Kaixiong Gong<sup>1,2*</sup></a>&emsp;
    <a href='http://kpzhang93.github.io/' target='_blank'>Kaipeng Zhang<sup>2,&#x2709</sup></a>&emsp;
    </br>
    <a href='http://www.ee.cuhk.edu.hk/~hsli/' target='_blank'>Hongsheng Li <sup>1,2</sup></a>&emsp;
    <a href='https://mmlab.siat.ac.cn/yuqiao/index.html' target='_blank'>Yu Qiao <sup>2</sup></a>&emsp;
    <a href='https://wlouyang.github.io/' target='_blank'>Wanli Ouyang<sup>2</sup></a>&emsp;
    <a href='http://people.eecs.berkeley.edu/~xyyue/' target='_blank'>Xiangyu Yue<sup>1,2,&#x2709</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>CUHK MMlab&emsp;
    <sup>2</sup>Shanghai AI Laboratory
    </br>
    <sup>*</sup> Equal Contribution&emsp;
    <sup>&#x2709</sup> Corresponding Author
    
</div>

-----------------

[[arXiv](https://kxgong.github.io/meta_transformer/)]
[[Project Page](https://kxgong.github.io/meta_transformer/)]
[[Blogs](https://kxgong.github.io/meta_transformer/)]
[[中文博客](https://kxgong.github.io/meta_transformer/)]



## 🚩🚩🚩 Shared-Encoder, Unpaired Data, More Modalities 

<p align="left">
<img src="assets\Meta-Transformer_cmp.png" width=100%>
</p>

This repository is built to explore the potential and extensiability of transformers for multimodal learning. We utilize the advantages of Transformers to deal with length-variant sequence. Then we proposes the *Data-to-Sequence* tokenization following a meta-scheme, then we apply it to 12 modalities including text, image, point cloud, audio, video, infrared, hyper-spectral, X-Ray, tabular, graph, time-series, and Inertial Measurement Unit (IMU) data.

<p align="left">
<img src="assets\Meta-Transformer_data2seq.png" width=100%>
</p>

After obtaining the token sequence, we employ a modality-shared encoder to extract representation across different modalities. With task-specific heads, Meta-Transformer can hanle various tasks on the different modalities, such as: classification, detection, and segmentation.

<p align="left">
<img src="assets\Meta-Transformer_framework.png" width=100%>
</p>

# 🌟 News
* **2023.7.8:** Github Repository Initialization.

# 🕙 ToDo
- [ ] Meta-Transformer with Large Language Models.
- [ ] Multimodal Joint Training with Meta-Transformer.
- [ ] Support More Modalities and More Tasks.

# Contact
Welcome to contribute to our project!

To contact us, never hestitate to send an email to `yiyuanzhang.ai@gmail.com` ,`zhangkaipeng@pjlab.org.cn`, or `xyyue@ie.cuhk.edu.hk`!
<br></br>

# License
This project is released under the [Apache 2.0 license](LICENSE).
# Acknowledgement
This code is developed based on excellent open-sourced projects including [MMClassification](https://github.com/open-mmlab/mmpretrain/tree/mmcls-1.x), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMsegmentation](https://github.com/open-mmlab/mmsegmentation), [OpenPoints](https://github.com/guochengqian/openpoints), [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [Graphomer](https://github.com/microsoft/Graphormer), [SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer), and [ViT-Adapter](https://github.com/czczup/ViT-Adapter).
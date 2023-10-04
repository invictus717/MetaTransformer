# VideoMAEv2 Model Zoo

## Model Weight Links
Please fill out [VideoMAE V2 Download Request Form](https://docs.google.com/forms/d/e/1FAIpQLSd1SjKMtD8piL9uxGEUwicerxd46bs12QojQt92rzalnoI3JA/viewform?usp=sf_link), **you will see the download link** for the VideoMAE V2 model weights after submission. The form asks for some information about your organization and how you plan to use the model, so that we can better understand the needs of our users and improve our future works. 

The weights of the distilled models can be downloaded directly at [Distillation](#Distillation) section.

## Pre-train

| Model | Config | Dataset | Encoder Masking | Decoder Masking | Epoch | \#Frame |
| :---: | :----  | :-----: | :-------------: | :-------------: | :---: | :-----: |
| ViT-giant | [vit_g_hybrid_pt_1200e]((/scripts/pretrain/vit_g_hybrid_pt.sh)) | UnlabeledHybrid | tube (90%) | running cell (50%) | 1200 | 16 |

- We set **different sampling intervals** for the videos from different sources in unlabeledhybrid: 2 for SSv2 and 4 for the other datasets.


## Fine-tune
| Model | Config | Dataset | Pre-train | Post-pre-train | \#Frame | Top-1 | Top-5 |
| :---: | :----  | :-----: | :-------: | :------------: | :-----: | :---: | :---: |
| ViT-giant | [vit_g_hybrid_pt_1200e_k710_ft](/scripts/finetune/vit_g_k710_ft.sh) | K710 | UnlabeledHybrid | None | 16x5x3 | 83.8 | 96.4 |
| ViT-giant | [vit_g_hybrid_pt_1200e_k400_ft](/scripts/finetune/vit_g_k400_ft.sh) | K400 | UnlabeledHybrid | None | 16x5x3 | 87.2 | 97.4 |
| ViT-giant | [vit_g_hybrid_pt_1200e_k710_it_k400_ft](/scripts/finetune/vit_g_k710_it_k400_ft.sh) | K400 | UnlabeledHybrid | K710 | 16x5x3 | 88.4 | 98.0 |
| ViT-giant | [vit_g_hybrid_pt_1200e_k710_it_k600_ft](/scripts/finetune/vit_g_k710_it_k600_ft.sh) | K600 | UnlabeledHybrid | K710 | 16x5x3 | 88.8 | 98.2 |
| ViT-giant | [vit_g_hybrid_pt_1200e_ssv2_ft](/scripts/finetune/vit_g_ssv2_ft.sh) | SSv2 | UnlabeledHybrid | None | 16x2x3 | 77.0 | 95.9 |
| ViT-giant | [vit_g_hybrid_pt_1200e_k710_it_ucf101_ft](/scripts/finetune/vit_g_k710_it_ucf101_ft.sh) | UCF101 | UnlabeledHybrid | K710 | 16x5x3 | 99.6 | 100.0 |
| ViT-giant | [vit_g_hybrid_pt_1200e_k710_it_hmdb51_ft](/scripts/finetune/vit_g_k710_it_hmdb51_ft.sh) | HMDB51 | UnlabeledHybrid | K710 | 16x5x3 | 88.1 | 98.5 |

- We report the fine-tuning accuracy for **sparse sampling** on SSv2 and for **dense sampling** on the other datasets.
- \#Frame = #input_frame x #clip x #crop.
- all the input resolution is $224^2$.

## Distillation
|  Model  | Dataset | Teacher Model | \#Frame | K710 Top-1 | K400 Top-1 | K600 Top-1 | Checkpoint |
| :-----: | :-----: | :-----------: | :-----: | :--------: | :--------: | :--------: | :--------  |
| ViT-small | K710 | vit_g_hybrid_pt_1200e_k710_ft | 16x5x3 | 77.6 | 83.7 | 83.1 | [vit_s_k710_dl_from_giant.pth](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/distill/vit_s_k710_dl_from_giant.pth) |
| | | fine-tuning accuracy | 16x7x3 | -- | 84.0 | 84.6 | -- | |
| ViT-base | K710 | vit_g_hybrid_pt_1200e_k710_ft | 16x5x3 | 81.5 | 86.6 | 85.9 | [vit_b_k710_dl_from_giant.pth](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/distill/vit_b_k710_dl_from_giant.pth) |
| | | fine-tuning accuracy | 16x7x3 | -- | 87.1 | 87.4 |  |

- We initialize the parameters of the student model with the model obtained after the post-pre-train stage.
- The fine-tuning accuracy refers to the accuracy achieved by further fine-tuning several epochs in the specified dataset after distillation.

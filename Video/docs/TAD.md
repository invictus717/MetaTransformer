# Temporal Action Detection
We use the [ActionFormer](https://github.com/happyharrycn/actionformer_release) detection pipeline as our baseline method and replace its I3D feature with the feature extracted by VideoMAE V2-g.

| Dataset | Backbone | Head | mAP | Features |
| :-----: | :------: | :--: | :-: | :------: |
| THUMOS14 | VideoMAE V2-g | ActionFormer | 69.6 | [th14_mae_g_16_4.tar.gz](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/features/th14_mae_g_16_4.tar.gz) |
| FineAction | VideoMAE V2-g | ActionFormer | 18.2 | [fineaction_mae_g.tar.gz](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/features/fineaction_mae_g.tar.gz) |

## Extract feature
Use `extract_tad_feature.py` to extract the feature of datasets. For example, to extract the feature of THUMOS14, running the following command:
```bash
python extract_tad_feature.py \
    --data_set THUMOS14 \
    --data_path YOUR_PATH/thumos14_videos \
    --save_path YOUR_PATH/th14_vit_g_16_4 \
    --model vit_giant_patch14_224 \
    --ckpt_path YOUR_PATH/vit_g_hyrbid_pt_1200e_k710_ft.pth
```

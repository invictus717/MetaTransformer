# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance_augreg.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]
pretrained = 'pretrained/Image_Meta-Transformer_L14.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='ViTAdapter',
        img_size=336,
        pretrain_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        drop_path_rate=0.4,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        window_attn=[True, True, True, True, True, False,
                     True, True, True, True, True, False,
                     True, True, True, True, True, False,
                     True, True, True, True, True, False],
        window_size=[14, 14, 14, 14, 14, None,
                     14, 14, 14, 14, 14, None,
                     14, 14, 14, 14, 14, None,
                     14, 14, 14, 14, 14, None],
        pretrained=pretrained),
    neck=dict(
        type='FPN',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        num_outs=5)
)
# optimizer
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='RandomCrop',
         crop_type='absolute_range',
         crop_size=(1024, 1024),
         allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.80))
optimizer_config = dict(grad_clip=None)
# fp16 = dict(loss_scale=dict(init_scale=512))
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=2,
    save_last=True,
)
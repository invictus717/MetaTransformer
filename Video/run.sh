#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='path/to/output/'
# OUTPUT_DIR='../work_dir/vit_b_k400_ft'
DATA_PATH='path/to/data/'
# DATA_PATH='../data/kinetics400'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12319 run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 6 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 7e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 90 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \
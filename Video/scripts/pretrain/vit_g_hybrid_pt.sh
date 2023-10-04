#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='YOUR_PATH/work_dir/vit_g_hybrid_pt_1200e'
DATA_PATH='YOUR_PATH/data/hybrid_train.csv'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        --async \
        ${SRUN_ARGS} \
        python -u run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --model pretrain_videomae_giant_patch14_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --with_checkpoint \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 4 \
        --num_workers 10 \
        --opt adamw \
        --lr 6e-4 \
        --clip_grad 0.02 \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 30 \
        --save_ckpt_freq 5 \
        --epochs 300 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        ${PY_ARGS}

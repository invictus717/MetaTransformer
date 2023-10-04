# Fine-tuning VideoMAEv2

VideoMAEv2 adopts the exactly same fine-tuning method as [VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/FINETUNE.md). Our codebase supports **multi-node slurm training** and **multi-node distributed training**. We provide the **off-the-shelf** slurm training scripts in the [fine-tune scripts folder](/scripts/finetune). Below we give an example of the fine-tuning script.

## Slurm Train

To fine-tune VideoMAEv2 ViT-giant on Kinetics-710 with 32 A100-80G (4 nodes x 8 GPUs), you can use the following script file **script/finetune/vit_g_k710_ft.sh**.

```bash
#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=$((12000 + $RANDOM % 20000))  # Randomly set master_port to avoid port conflicts
export OMP_NUM_THREADS=1  # Control the number of threads

OUTPUT_DIR='YOUR_PATH/work_dir/vit_g_hybrid_pt_1200e_k710_ft'  # Your output folder for deepspeed config file, logs and checkpoints
DATA_PATH='YOUR_PATH/data/k710'  # The data list folder. the folder has three files: train.csv, val.csv, test.csv
# finetune data list file follows the following format
# for the video data line: video_path, label
# for the rawframe data line: frame_folder_path, total_frames, label
MODEL_PATH='YOUR_PATH/model_zoo/vit_g_hybrid_pt_1200e.pth'  # Model for initializing parameters

JOB_NAME=$1  # the job name of the slurm task
PARTITION=${PARTITION:-"video"}  # Name of the partition
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-32}  # Number of GPUs
GPUS_PER_NODE=${GPUS_PER_NODE:-8}  # Number of GPUs in each node
CPUS_PER_TASK=${CPUS_PER_TASK:-14}  # Number of CPU cores allocated, number of tasks equal to the number of GPUs used
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:2}  # Other training args

# Please refer to `run_class_finetuning.py` for the meaning of the following hyperreferences
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        --async \
        ${SRUN_ARGS} \
        python run_class_finetuning.py \
        --model vit_giant_patch14_224 \
        --data_set Kinetics-710 \
        --nb_classes 710 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 3 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 10 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.3 \
        --clip_grad 5.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 5 \
        --epochs 35 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \
        ${PY_ARGS}
  ```

Start training by running
```bash
bash script/finetune/vit_g_k710_ft.sh k710_finetune
```
, where 'k710_finetune' is the job name.

If you just want to **test the performance of the model**, change `MODEL_PATH` to the model to be tested, `OUTPUT_DIR` to the path of the folder where the test results are saved, and run the following command:
```bash
bash script/finetune/vit_g_k710_ft.sh k710_model_test --eval
```

## Dist Train

The above slurm training script can be modified to distributed training script as follows:

```bash
#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes

OUTPUT_DIR='YOUR_PATH/work_dir/vit_g_hybrid_pt_1200e_k710_ft'  # Your output folder for deepspeed config file, logs and checkpoints
DATA_PATH='YOUR_PATH/data/k710'  # The data list folder. the folder has three files: train.csv, val.csv, test.csv
# finetune data list file follows the following format
# for the video data line: video_path, label
# for the rawframe data line: frame_folder_path, total_frames, label
MODEL_PATH='YOUR_PATH/model_zoo/vit_g_hybrid_pt_1200e.pth'  # Model for initializing parameters

N_NODES=${N_NODES:-4}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-8}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3}  # Other training args

# Please refer to `run_class_finetuning.py` for the meaning of the following hyperreferences
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=$1 --master_addr=$2 \
        run_class_finetuning.py \
        --model vit_giant_patch14_224 \
        --data_set Kinetics-710 \
        --nb_classes 710 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 3 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 10 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.3 \
        --clip_grad 5.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 5 \
        --epochs 35 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \
        ${PY_ARGS}
```
Start training by run
```bash
NODE_RANK=0  # 0 for the first node 0, 1 for the second node, and so on.
# MASTER_ADDR should be set as the ip of current node

bash dist_train_vit_g_k710_ft.sh $NODE_RANK $MASTER_ADDR
```
at each node. 

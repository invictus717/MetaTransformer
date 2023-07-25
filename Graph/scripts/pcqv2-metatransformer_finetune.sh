#!/usr/bin/env bash

ulimit -c unlimited

fairseq-train \
--user-dir ../metatransformer \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name pcqm4mv2 \
--dataset-source ogb \
--task graph_prediction \
--criterion l1_loss \
--arch tokengt_base \
--lap-node-id \
--lap-node-id-k 16 \
--lap-node-id-sign-flip \
--performer \
--performer-finetune \
--performer-feature-redraw-interval 100 \
--prenorm \
--num-classes 1 \
--attention-dropout 0.0 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.1 \
--lr-scheduler cosine --warmup-updates 1000 --max-update 200000 \
--lr 2e-4 \
--batch-size 128 \
--data-buffer-size 20 \
--load-pretrained-model-output-layer \
--save-dir ./ckpts/pcqv2-tokengt-metatransformerencoder-performer-finetune \
--tensorboard-logdir ./tb/pcqv2-tokengt-metatransformerencoder-performer-finetune \
--pretrained-model-name pcqv2-tokengt-lap16 \
--no-epoch-checkpoints

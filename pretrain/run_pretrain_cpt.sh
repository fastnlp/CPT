#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH="dataset/"
CHECKPOINT_PATH=checkpoints/cpt-base
VOCAB_FILE=vocab/bert_zh_vocab/

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_cpt.py \
       --num-layers 12 \
       --num-decoder-layers 2 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --micro-batch-size 32 \
       --global-batch-size 512 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --mask-prob 0.15 \
       --train-iters 1000000 \
       --lr-decay-iters 1000000 \
       --lr-warmup-fraction .01 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,30,1 \
       --distributed-backend nccl \
       --lr 1e-4 \
       --lr-encoder 5e-5 \
       --lr-decay-style cosine \
       --min-lr 1e-6 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --initial-loss-scale 65536 \
       --log-interval 10 \
       --save-interval 10000 \
       --eval-interval 500 \
       --eval-iters 10 \
       --num-workers 2 \
       --fp16 \
       # --checkpoint-activations

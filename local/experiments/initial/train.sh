#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
export TRANSFORMERS_OFFLINE=1


export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

export NCCL_DEBUG=INFO



NAME=t5ce_ost_continue
portion=0001

MODEL=/home/dongheng/dongheng_scratch/OST/T5_un_ost_T0/model-004k
DATA=/home/dongheng/data/dialogue/pseudo/t0_ost
CONFIG=/home/dongheng/models/t5-base/config.json
TOKENIZER=/home/dongheng/models/t5-base
SAVE=/home/dongheng/dongheng_scratch/OST/${NAME}

mkdir -p $SAVE
cp $0 $SAVE/

python train.py \
  -d $DATA \
  -cn $CONFIG \
  -tn $TOKENIZER \
  -s src tgt \
  --max-tokens 4096 \
  --num-training-steps 100000 \
  -lr 7e-4 \
  --num-warmup-steps 4000 \
  --iter-per-update 2 \
  --save-dir $SAVE \
  --update-per-save 1000 \
  -mn $MODEL \
  --fp32 \
  --label-smoothing 0.1 \
  | tee -a $SAVE/train.log

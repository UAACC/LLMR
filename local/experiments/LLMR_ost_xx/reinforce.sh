#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,5
export TRANSFORMERS_OFFLINE=1


export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

export NCCL_DEBUG=INFO




NAME=T5_un_ost_T0_xx

MODEL=/home/dongheng/dongheng_scratch/OST/T5_un_ost_T0/model-004k
data=ost
DATA=/home/dongheng/data/dialogue/cleaned_${data}_0.1
CONFIG=/home/dongheng/models/t5-base/config.json
TOKENIZER=/home/dongheng/models/t5-base
SAVE=/home/dongheng/dongheng_scratch/OST/T5_un_ost_T0_xx/T5_un_ost_T0_01
reward_model=/home/dongheng/models/t0-3b



mkdir -p $SAVE
cp $0 $SAVE/

python reinforce.py \
  -d $DATA \
  -cn $CONFIG \
  -tn $TOKENIZER \
  -s src src \
  --max-tokens 512 \
  --num-training-steps 100000 \
  -lr 1e-6 \
  --num-warmup-steps 5000 \
  --iter-per-update 16 \
  --save-dir $SAVE \
  -mn $MODEL \
  --reward-model $reward_model \
  --topk 5\
  --scheduler constant\
  --max-norm 1 \
  --softmax \
  --fp32 \
  --entropy 0.1 \
  --denom 100 \
  --deterministic \
  --reward-clip 1 \
  --template templates/dialogue_T0.txt \
  --update-per-save 1000 \
  --update-per-log 1 \
  | tee -a $SAVE/train.log

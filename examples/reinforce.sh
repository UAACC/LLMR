#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train
#SBATCH --account=rrg-lilimou
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gres=gpu:v100l:4
#SBATCH --output=/project/def-lilimou/ychao/logs/output-%j.log
#SBATCH --error=/project/def-lilimou/ychao/logs/error-%j.log

export TRANSFORMERS_OFFLINE=1

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_GID_INDEX=mlx5_0
export NCCL_DEBUG=WARN

export HF_HOME='/project/def-lilimou/ychao/hf'

NAME=llmr-1e-5-bl-rep1-ent0.5-sync5k-denom100-rc1-topa

MODEL_NAME=t5-base
REWARD_MODEL_NAME=t0-3b
WS=/project/def-lilimou/ychao
TEMP_WS=/scratch/ychao

DATA=$WS/data/dialogue/cleaned_dd/single-turn
CONFIG=$HF_HOME/hub/$MODEL_NAME/config.json
TOKENIZER=$HF_HOME/hub/$MODEL_NAME
SAVE=$TEMP_WS/projects/LLMR/ckpts/$NAME


mkdir -p $SAVE
cp $0 $SAVE/

python reinforce.py \
  -d $DATA \
  -cn $CONFIG \
  -tn $TOKENIZER \
  -s src src \
  --max-tokens 1024 \
  --num-training-steps 100000 \
  -lr 1e-5 \
  --num-warmup-steps 4000 \
  --iter-per-update 8 \
  --save-dir $SAVE \
  --update-per-save 1000 \
  -mn $WS/hf/hub/$MODEL_NAME \
  --reward-model $WS/hf/hub/$MODEL_NAME \
  --fp32 \
  --max-norm 1 \
  --softmax \
  --entropy 0.5 \
  --denom 100 \
  --reward-clip 1 \
  --template templates/dialogue.txt \
  --update-per-save 1000 \
  --update-per-log 1 \
  | tee -a $SAVE/train.log

#!/bin/bash



export CUDA_VISIBLE_DEVICES=4



data=dd
dataset=/home/dongheng/data/dialogue/cleaned_${data}/single-turn/train.src
model=/home/dongheng/models/flan_t5_large
template=/home/dongheng/LLMR/templates/dialogue_T0.txt



op=/home/dongheng/data/dialogue/pseudo/flan_t5_${data}/results_${data}_pseudo.tgt
python llm_decoding.py --template $template --max-length-a 3 --max-sentences 16 --do-sample --top-k 1 -i $dataset -mn $model -o $op

  
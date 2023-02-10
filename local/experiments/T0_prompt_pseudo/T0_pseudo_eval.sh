#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

data=ost
dataset=/home/dongheng/data/dialogue/cleaned_${data}/single-turn
T0_prompt_pseudo=/home/dongheng/LLMR/local/experiments/T0_prompt_pseudo
template=/home/dongheng/LLMR/templates/dialogue_T0.txt
model=/home/dongheng/models/t0-3b
src=src
tgt=tgt


for sp in valid test
do
op=/home/dongheng/LLMR/local/experiments/T0_prompt_pseudo/results_${data}_$sp.tgt
python llm_decoding.py --template $template --max-length-a 3 --max-sentences 16 --do-sample --top-k 1 -i $dataset/$sp.src -mn $model -o $op
python metrics.py --lowercase --gen $op --ref $dataset/$sp.$tgt --src $dataset/$sp.$src >> $T0_prompt_pseudo/$sp.$data.log
done
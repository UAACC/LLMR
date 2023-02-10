#!/bin/bash



export CUDA_VISIBLE_DEVICES=0



data=dd
sp=test
Noex=4

dataset=/home/dongheng/data/dialogue/cleaned_${data}/single-turn
model=/home/dongheng/models/t0-3b
template=/home/dongheng/LLMR/templates/dialogue_T0.txt
save=/home/dongheng/LLMR/local/experiments/T0_prompt_pseudo/dd
src=src
tgt=tgt


op=$save/dd.$Noes.tgt
python llm_decoding_BEAM.py --template $template --max-length-a 3 -i $dataset/$sp.$src -mn $model -o $op
python metrics.py --lowercase --gen $op --ref $dataset/$sp.$tgt --src $dataset/$sp.$src >> $save/$sp.$data.log
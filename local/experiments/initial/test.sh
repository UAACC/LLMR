#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
data=ost
dataset=/home/dongheng/data/dialogue/cleaned_${data}/single-turn
model=/home/dongheng/dongheng_scratch/OST/t5ce_ost_continue
src=src
tgt=tgt

ckpt='model-001k'
op=$model/$ckpt/test_ost.tgt
python decoding.py --max-length-a 3 --max-sentences 64 --do-sample --top-k 1 -i $dataset/test.src -mn $model/$ckpt -o $op
echo $ckpt $dataset >> $model/test.$data.log
python metrics.py --lowercase --gen $op --ref $dataset/test.$tgt --src $dataset/test.$src >> $model/test.$data.log

  
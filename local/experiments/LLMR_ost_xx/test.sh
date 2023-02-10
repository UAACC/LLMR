#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
data=ost
dataset=/home/dongheng/data/dialogue/cleaned_${data}/single-turn
model=/home/dongheng/dongheng_scratch/OST/T5_un_ost_T0
src=src
tgt=tgt

ckpt='model-004k'
op=$model/$ckpt/test.tgt
python decoding.py --max-length-a 3 --max-sentences 64 --do-sample --top-k 1 -i $dataset/test.src -mn $model/$ckpt -o $op
echo $ckpt $dataset >> $model/test.$data.log
python metrics.py --lowercase --gen $op --ref $dataset/test.$tgt --src $dataset/test.$src >> $model/test.$data.log

  
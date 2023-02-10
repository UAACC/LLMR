#!/bin/bash

export CUDA_VISIBLE_DEVICES=5


data=ost
dataset=/home/dongheng/data/dialogue/cleaned_${data}/single-turn
model=/home/dongheng/dongheng_scratch/OST/T5_un_ost_T0_xx/T5_un_ost_T0_001
src=src
tgt=tgt

for ckpt in `ls -r $model`
do
    op=$model/$ckpt/valid.tgt
    python decoding.py --max-length-a 3 --max-sentences 64 --do-sample --top-k 1 -i $dataset/valid.src -mn $model/$ckpt -o $op
    echo $ckpt $dataset >> $model/valid.$data.log
    python metrics.py --lowercase --gen $op --ref $dataset/valid.$tgt --src $dataset/valid.$src >> $model/valid.$data.log
done
  
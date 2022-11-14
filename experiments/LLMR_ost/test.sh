#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=test
#SBATCH --account=rrg-lilimou
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=/project/def-lilimou/mrli/logs/output-%j.log
#SBATCH --error=/project/def-lilimou/mrli/logs/error-%j.log

module load python/3.7

# pip install nltk rouge



data=ost
dataset=/project/def-lilimou/ychao/data/dialogue/cleaned_${data}/single-turn
model=/home/mrli/scratch/projects/LLMR/ckpts/LLMR_ost
src=src
tgt=tgt

ckpt='model-006k'
op=$model/$ckpt/test.tgt
python decoding.py --max-length-a 3 --max-sentences 64 --do-sample --top-k 1 -i $dataset/test.src -mn $model/$ckpt -o $op
echo $ckpt $dataset >> $model/test.$data.log
python metrics.py --lowercase --gen $op --ref $dataset/test.$tgt --src $dataset/test.$src >> $model/test.$data.log

  
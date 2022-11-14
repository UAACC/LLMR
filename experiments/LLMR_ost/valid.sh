#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --job-name=valid
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

for ckpt in `ls $model`
do
op=$model/$ckpt/valid.tgt
python decoding.py --max-length-a 3 --max-sentences 64 --do-sample --top-k 1 -i $dataset/valid.src -mn $model/$ckpt -o $op
echo $ckpt $dataset >> $model/valid.$data.log
python metrics.py --lowercase --gen $op --ref $dataset/valid.$tgt --src $dataset/valid.$src >> $model/valid.$data.log
done

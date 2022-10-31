#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --job-name=eval
#SBATCH --account=rrg-lilimou
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=/project/def-lilimou/mrli/logs/output-%j.log
#SBATCH --error=/project/def-lilimou/mrli/logs/error-%j.log

module load python/3.7

# pip install nltk rouge



data=dd
dataset=/project/def-lilimou/ychao/data/dialogue/cleaned_${data}/single-turn
t0_prompt=/home/mrli/projects/def-lilimou/mrli/projects/LLMR/experiments/t0_prompt_no_quo
title=no_quo_eval
src=src
tgt=tgt


for sp in valid test
do
op=/home/mrli/projects/def-lilimou/mrli/projects/LLMR/experiments/t0_prompt/results_no_quo.tgt
python metrics.py --lowercase --gen $op --ref $dataset/$sp.$tgt --src $dataset/$sp.$src >> $t0_prompt/$sp.$data.log
done
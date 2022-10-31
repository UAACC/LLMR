#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=generate
#SBATCH --account=rrg-lilimou
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48000M
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=/project/def-lilimou/mrli/logs/output-%j.log
#SBATCH --error=/project/def-lilimou/mrli/logs/error-%j.log

module load python/3.7

# pip install nltk rouge




dataset=/home/mrli/projects/def-lilimou/ychao/data/dialogue/cleaned_ost/single-turn/train.src
model=/home/mrli/projects/def-lilimou/ychao/hf/hub/t0-3b
template=/home/mrli/projects/def-lilimou/mrli/projects/LLMR/templates/dialogue_T0.txt



op=/home/mrli/projects/def-lilimou/mrli/data/results.tgt
python llm_decoding.py --template $template --max-length-a 3 --max-sentences 64 --do-sample --top-k 1 -i $dataset -mn $model -o $op

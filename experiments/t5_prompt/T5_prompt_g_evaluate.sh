#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --job-name=promptgen/eval
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
model=/home/mrli/projects/def-lilimou/ychao/hf/hub/t0-3b
t5_base=/project/def-lilimou/ychao/hf/hub/t5-base
template=/home/mrli/projects/def-lilimou/mrli/projects/LLMR/templates/dialogue.txt
t5_prompt=/home/mrli/projects/def-lilimou/mrli/projects/LLMR/experiments/t5_prompt
src=src
tgt=tgt

for sp in valid test
do
op=/home/mrli/projects/def-lilimou/mrli/projects/LLMR/experiments/t5_prompt/results_NO_TEM.tgt
python llm_decoding.py --template $template --max-length-a 3 --max-sentences 64 --do-sample --top-k 1 -i $dataset/$sp.src -mn $t5_base -o $op
echo $dataset >> $t5_prompt/$sp.$data.log 
head -n 1 $template >> $t5_prompt/$sp.$data.log 
python metrics.py --lowercase --gen $op --ref $dataset/$sp.$tgt --src $dataset/$sp.$src >> $t5_prompt/$sp.$data.log
done

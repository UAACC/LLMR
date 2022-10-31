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
template=/home/mrli/projects/def-lilimou/mrli/projects/LLMR/templates/dialogue_t0_2.txt
t0_prompt_task=/home/mrli/projects/def-lilimou/mrli/projects/LLMR/experiments/t0_prompt_task
src=src
tgt=tgt

for sp in valid test
do
op=/home/mrli/projects/def-lilimou/mrli/projects/LLMR/experiments/t0_prompt_task/results.tgt
python llm_decoding.py --template $template --max-length-a 3 --max-sentences 64 --do-sample --top-k 1 -i $dataset/$sp.src -mn $model -o $op
echo $dataset >> $t0_prompt_task/$sp.$data.log 
head -n 1 $template >> $t0_prompt_task/$sp.$data.log 
python metrics.py --lowercase --gen $op --ref $dataset/$sp.$tgt --src $dataset/$sp.$src >> $t0_prompt_task/$sp.$data.log
done

  
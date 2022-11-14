#!/bin/bash
model=/home/mrli/scratch/projects/LLMR/ckpts/t5b-t0p-ost

for i in 0{67..87}k
    ckpt = `model-$i`

do
    
    
    echo `$ckpt`
done
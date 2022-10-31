### WARNING
### DO NOT MODIFY UNLESS YOU ARE SURE
# nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# export MASTER_ADDR=$(ssh $head_node hostname --ip-address)
# export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
# # export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_GID_INDEX=mlx5_0
# module load python/3.7
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index transformers torch
### END WARNING

### CONFIGURE
# NAME=		# the name of the model, used for saving
# DATA=		# the data directory
# CONFIG=		# the .json file of the model
# TOKENIZER=	# the directory contains the tokenizer
# SAVE=		# the directory models will be saved under 
# SRC=		# source suffix, e.g. en
# TGT=		# target suffix, e.g. de
# ### END CONFIGURE

# mkdir -p $SAVE

python train_bleu.py \
  -d $DATA \
  -cn $CONFIG \
  -tn $TOKENIZER \
  -s $SRC $TGT  \
  --max-tokens 8192 \
  --num-training-steps 100000 \
  -lr 7e-4 \
  --num-warmup-steps 4000 \
  --iter-per-update 1 \
  --save-dir $SAVE \
  | tee $SAVE/train.log

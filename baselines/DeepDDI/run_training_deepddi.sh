#!/bin/bash
#SBATCH -J deepddi
#SBATCH -o /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.out
#SBATCH -e /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.err
#SBATCH -c 16
#SBATCH -t 1-00:00
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G

base="/home/yeh803/workspace/DDI/NovelDDI/baselines/DeepDDI"
BATCH_SIZE=65536
source activate primekg
cd $base

python train_TWOSIDES.py \
--feature_dim 1024 \
--num_layers 4 \
--batch_size $BATCH_SIZE \
--num_epoch 300 \
--num_workers 2 \
--split_method "split_by_drugs_random" & 

python train_TWOSIDES.py \
--feature_dim 512 \
--num_layers 6 \
--batch_size $BATCH_SIZE \
--num_epoch 300 \
--num_workers 2 \
--split_method "split_by_drugs_random" & 

python train_TWOSIDES.py \
--feature_dim 256 \
--num_layers 6 \
--batch_size $BATCH_SIZE \
--num_epoch 300 \
--num_workers 2 \
--split_method "split_by_drugs_random" & 

python train_TWOSIDES.py \
--feature_dim 512 \
--num_layers 4 \
--batch_size $BATCH_SIZE \
--num_epoch 300 \
--num_workers 2 \
--split_method "split_by_drugs_random" & 

python train_TWOSIDES.py \
--feature_dim 256 \
--num_layers 4 \
--batch_size $BATCH_SIZE \
--num_epoch 300 \
--num_workers 2 \
--split_method "split_by_drugs_random" & 

python train_TWOSIDES.py \
--feature_dim 512 \
--num_layers 4 \
--batch_size $BATCH_SIZE \
--num_epoch 300 \
--num_workers 2 \
--split_method "split_by_triplets" & 

python train_TWOSIDES.py \
--feature_dim 512 \
--num_layers 4 \
--batch_size $BATCH_SIZE \
--num_epoch 300 \
--num_workers 2 \
--split_method "split_by_pairs" 

wait

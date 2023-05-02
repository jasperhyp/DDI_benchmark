#!/bin/bash
#SBATCH -J ssiddi
#SBATCH -o /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.out
#SBATCH -e /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.err
#SBATCH -c 8
#SBATCH -t 2-00:00
#SBATCH -p gpu_requeue
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G

base="/home/yeh803/workspace/DDI/NovelDDI/baselines/ChemicalX"

source activate primekg
cd $base

python train_chemicalx.py \
--model_name "ssiddi" \
--batch_size 80 \
--num_epoch 20 \
--num_workers 16 \
--mol_feat_type "ecfp" \
--split_method "split_by_drugs_random"

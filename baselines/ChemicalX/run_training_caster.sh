#!/bin/bash
#SBATCH -J caster
#SBATCH -o /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.out
#SBATCH -e /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.err
#SBATCH -c 8
#SBATCH -t 2-00:00
#SBATCH -p gpu_requeue
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G

base="/home/yeh803/workspace/DDI/NovelDDI/baselines/ChemicalX"

source activate primekg
cd $base

python train_chemicalx.py \
--model_name "caster" \
--batch_size 180 \
--num_epoch 3 \
--num_workers 16 \
--mol_feat_type "caster" \
--split_method "split_by_drugs_random"

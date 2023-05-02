#!/bin/bash
#SBATCH -J caster
#SBATCH -o /home/yeh803/workspace/DDI/NovelDDI/out/%x.%j.out
#SBATCH -e /home/yeh803/workspace/DDI/NovelDDI/out/%x.%j.out
#SBATCH -c 8
#SBATCH -t 2-00:00
#SBATCH -p gpu_requeue
#SBATCH --qos=gpu_requeue
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=48G

base="/home/yeh803/workspace/DDI/NovelDDI/baselines/CASTER/DDE/"
cd $base

source activate primekg
python run_caster_benchmark.py \
--train_epoch 1 \
--split_method "split_by_drugs_random" \
--batch_size 520 \
--num_workers 8

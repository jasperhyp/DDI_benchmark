#!/bin/bash
#SBATCH -J caster_ddp
#SBATCH -o /home/yeh803/workspace/DDI/NovelDDI/out/%x.%j.out
#SBATCH -e /home/yeh803/workspace/DDI/NovelDDI/out/%x.%j.out
#SBATCH -c 8
#SBATCH -t 1-00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=80G
base="/home/yeh803/workspace/DDI/NovelDDI/baselines/CASTER/DDE/"
cd $base

source activate primekg
python -m torch.distributed.launch --nproc_per_node 4 run_caster_benchmark_ddp.py --train_epoch 2 --split_method "split_by_drugs" --batch_size 512

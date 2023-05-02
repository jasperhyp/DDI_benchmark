#!/bin/bash
#SBATCH -J torchdrug
#SBATCH -o /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.out
#SBATCH -e /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.err
#SBATCH -c 6
#SBATCH -t 2-00:00
#SBATCH -p gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

base="/home/yeh803/workspace/DDI/NovelDDI/baselines/TorchDrug"
BATCH_SIZE=16384
EPOCHS=300

source activate primekg
cd $base

# python train_torchdrug.py \
# --mol_encoder_name "gat" \
# --mol_encoder_edge_input_dim 18 \
# --batch_size $BATCH_SIZE \
# --num_epoch $EPOCHS \
# --num_workers 2 \
# --split_method "split_by_drugs_random" & 

# python train_torchdrug.py \
# --mol_encoder_name "gat" \
# --batch_size $BATCH_SIZE \
# --num_epoch $EPOCHS \
# --num_workers 2 \
# --split_method "split_by_drugs_random" & 

# python train_torchdrug.py \
# --mol_encoder_name "gat" \
# --mol_encoder_edge_input_dim 18 \
# --batch_size $BATCH_SIZE \
# --num_epoch $EPOCHS \
# --num_workers 2 \
# --split_method "split_by_pairs" & 

# python train_torchdrug.py \
# --mol_encoder_name "gat" \
# --mol_encoder_edge_input_dim 18 \
# --batch_size $BATCH_SIZE \
# --num_epoch $EPOCHS \
# --num_workers 2 \
# --split_method "split_by_triplets" & 

# python train_torchdrug.py \
# --mol_encoder_name "gin" \
# --mol_encoder_edge_input_dim 18 \
# --batch_size $BATCH_SIZE \
# --num_epoch $EPOCHS \
# --num_workers 2 \
# --split_method "split_by_drugs_random" & 

# python train_torchdrug.py \
# --mol_encoder_name "neuralfp" \
# --mol_encoder_edge_input_dim 18 \
# --batch_size $BATCH_SIZE \
# --num_epoch $EPOCHS \
# --num_workers 2 \
# --split_method "split_by_drugs_random" & 

# python train_torchdrug.py \
# --mol_encoder_name "mpnn" \
# --mol_encoder_edge_input_dim 18 \
# --batch_size $BATCH_SIZE \
# --num_epoch $EPOCHS \
# --num_workers 2 \
# --split_method "split_by_drugs_random" &

# python train_torchdrug.py \
# --mol_encoder_name "gin" \
# --mol_encoder_edge_input_dim 18 \
# --batch_size $BATCH_SIZE \
# --num_epoch $EPOCHS \
# --num_workers 2 \
# --split_method "split_by_pairs" & 

# python train_torchdrug.py \
# --mol_encoder_name "neuralfp" \
# --mol_encoder_edge_input_dim 18 \
# --batch_size $BATCH_SIZE \
# --num_epoch $EPOCHS \
# --num_workers 2 \
# --split_method "split_by_pairs" & 

# python train_torchdrug.py \
# --mol_encoder_name "mpnn" \
# --mol_encoder_edge_input_dim 18 \
# --batch_size $BATCH_SIZE \
# --num_epoch $EPOCHS \
# --num_workers 2 \
# --split_method "split_by_pairs" &

python train_torchdrug.py \
--mol_encoder_name "gin" \
--mol_encoder_edge_input_dim 18 \
--batch_size $BATCH_SIZE \
--num_epoch $EPOCHS \
--num_workers 2 \
--split_method "split_by_triplets" & 

python train_torchdrug.py \
--mol_encoder_name "neuralfp" \
--mol_encoder_edge_input_dim 18 \
--batch_size $BATCH_SIZE \
--num_epoch $EPOCHS \
--num_workers 2 \
--split_method "split_by_triplets" & 

python train_torchdrug.py \
--mol_encoder_name "mpnn" \
--mol_encoder_edge_input_dim 18 \
--batch_size $BATCH_SIZE \
--num_epoch $EPOCHS \
--num_workers 2 \
--split_method "split_by_triplets" 

wait

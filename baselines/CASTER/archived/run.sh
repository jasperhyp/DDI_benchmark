#!/bin/bash
#SBATCH -c 4                               # Request one core
#SBATCH -t 0-12:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=24G                         # Memory total in MiB (for all cores)
#SBATCH -o run_job.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e run_job.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:3
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=nichola2@asu.edu # send-to address

# You can change hostname to any command you would like to run
# module load gcc/6.2.0
# module load miniconda3/4.10.3
# source activate /home/nih492/.conda/envs/pyg_nick

#wandb agent noveldrugdrug/strAE-nddi/bzptelgk
# wandb agent noveldrugdrug/strAE-nddi/6f9ehzxe

python run_caster_benchmark.py
 

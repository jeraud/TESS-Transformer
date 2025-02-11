#!/bin/bash
#SBATCH -J classifierTrainer_4GPUs
#SBATCH -o myjob_4GPUs_%j.out
#SBATCH -e myjob_4GPUs_%j.err
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=6:33:33
#SBATCH --exclusive

module load anaconda3/2020.02-2ks5tch
module load cuda/11.8
source /home/paulg9/miniforge3/bin/activate

srun /home/paulg9/miniforge3/bin/python train.py
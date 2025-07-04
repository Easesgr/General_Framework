#!/bin/bash
#SBATCH --job-name=LKDA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=gpu
#SBATCH --nodelist=gnode02
#SBATCH --gres=gpu:2

# run python
source /home/a01sun/asc22016000158/anaconda3/bin/activate cuda
#conda activate pytorch_1.8

# noise-free degradations with isotropic Gaussian blurs
python main.py

#cmd /k

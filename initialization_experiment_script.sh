#!/bin/bash
#SBATCH --account=sur_lab
#SBATCH --job-name=initialization_experiment
#SBATCH --partition=bigmem
#SBATCH --time=3-00:00
#SBATCH --mem=2000G
#SBATCH -c 1
#SBATCH --output=/n/home10/colson/metric-learning/out_files/%j.out
#SBATCH --error=/n/home10/colson/metric-learning/out_files/%j.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=BEGIN

##SBATCH --cpus-per-task=64
##SBATCH --gres=gpu:nvidia_a40:2
##SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
##SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
##SBATCH --gres=gpu:4

# Load modules
module load python/3.10.9-fasrc01
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01
mamba activate metriclearning2

python3 initialization_experiment_numpy.py

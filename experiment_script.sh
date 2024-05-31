#!/bin/bash
#SBATCH --account=sur_lab
#SBATCH --partition=gpu
##SBATCH --gres=gpu:nvidia_a40:2
##SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
##SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=64
#SBATCH --output=/n/home10/colson/metric-learning/out_files/%x.out
#SBATCH --error=/n/home10/colson/metric-learning/out_files/%x.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=BEGIN


module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
source activate metriclearning2
python metric_learning_experiment.py

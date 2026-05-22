#!/bin/bash
#SBATCH --account=sur_lab
#SBATCH --job-name=gd_experiment
#SBATCH --partition=bigmem
#SBATCH --time=3-00:00
#SBATCH --mem=1000G
#SBATCH -c 1
#SBATCH --output=/n/home10/colson/metric-learning/out_files/%j.out
#SBATCH --error=/n/home10/colson/metric-learning/out_files/%j.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=BEGIN


# Load modules
module load python/3.10.12-fasrc01 
mamba activate metriclearning2

python3 experiments-unified.py $1 $2 

#!/bin/bash
#SBATCH --account=sur_lab
#SBATCH --job-name=initialization_experiment
#SBATCH --partition=bigmem
#SBATCH --time=2-00:00
#SBATCH --mem=2000G
#SBATCH -c 1
#SBATCH --output=/n/home10/colson/metric-learning/out_files/%j.out
#SBATCH --error=/n/home10/colson/metric-learning/out_files/%j.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=BEGIN


# Load modules
module load python/3.10.9-fasrc01
mamba activate metriclearning2

python3 initialization_experiment.py

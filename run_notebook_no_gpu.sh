#!/bin/bash
#SBATCH --account=sur_lab
#SBATCH --job-name=jupyter
#SBATCH --partition=sur
#SBATCH --time=0-02:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --output=/n/home10/colson/metric-learning/out_files/%x.out
#SBATCH --error=/n/home10/colson/metric-learning/out_files/%x.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=BEGIN

# Load modules
mamba activate metriclearning2
module load python/3.10.12-fasrc01
# module load cuda/11.8.0-fasrc01
# module load cudnn/8.9.2.26_cuda11-fasrc01


jupyter_token="c63f0bc7000ede187e773fae0bd7b3264879b25025cc40c9"

ngrok_start () {
    cd /n/home10/colson/helpers
    echo "starting ngrok"
    kill -9 "$(pgrep ngrok)"
    ngrok http 8889 > /dev/null &
    sleep 5
    curl http://localhost:4040/api/tunnels > ngrok.json
    echo "dumping ngrok info to tunnels.json"
    ngrok_url=`cat ngrok.json | jq -r '.tunnels[0].public_url'`
    jupyter_url="${ngrok_url}/lab?token=${jupyter_token}"
    echo "jupyter url: ${jupyter_url}"
    echo "${jupyter_url}" > jupyter_url.txt
}

ngrok_start

# Run jupyter notebook
cd /n/home10/colson/
/n/home10/colson/.conda/envs/gpu/bin/jupyter lab --no-browser --port=8889 --ip=0.0.0.0 --NotebookApp.token="${jupyter_token}"

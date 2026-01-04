#!/bin/bash
#SBATCH --account=sur_lab
#SBATCH --job-name=jupyter
##SBATCH --partition=bigmem
#SBATCH --partition=shared
#SBATCH --time=0-04:00
##SBATCH --mem=1000G
#SBATCH --mem=184G
#SBATCH --cpus-per-task=1
#SBATCH --output=/n/home10/colson/metric-learning/out_files/%x.out
#SBATCH --error=/n/home10/colson/metric-learning/out_files/%x.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=BEGIN

# Load modules
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
source activate metriclearning2

jupyter_token="c63f0bc7000ede187e773fae0bd7b3264879b25025cc40c9"

ngrok_start () {
    cd /n/home10/colson/llm_fairness/helpers
    echo "starting ngrok"
    kill -9 "$(pgrep ngrok)"
    ngrok http 8889 > /dev/null &
    sleep 5
    curl http://localhost:4040/api/tunnels > /n/home10/colson/metric-learning/ngrok.json
    curl http://localhost:4040/api/tunnels
    echo "dumping ngrok info to tunnels.json"
    ngrok_url=`cat ngrok.json | jq -r '.tunnels[0].public_url'`
    jupyter_url="${ngrok_url}/lab?token=${jupyter_token}"
    echo "jupyter url: ${jupyter_url}"
    echo "${jupyter_url}" > /n/home10/colson/metric-learning/jupyter_url.txt
}

ngrok_start

# Run jupyter notebook
cd /n/home10/colson/
/n/home10/colson/.conda/envs/gpu/bin/jupyter lab --no-browser --port=8889 --ip=0.0.0.0 --NotebookApp.token="${jupyter_token}"

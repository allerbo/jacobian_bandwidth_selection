#!/usr/bin/env bash
#SBATCH -A xxx
#SBATCH -t 1:00:00
#SBATCH -o out_real.txt
#SBATCH -e err_real.txt
#SBATCH -n 1


ml SciPy-bundle/2022.05-foss-2022a
source ~/my_python/bin/activate
python real.py seed=$SLURM_ARRAY_TASK_ID $*

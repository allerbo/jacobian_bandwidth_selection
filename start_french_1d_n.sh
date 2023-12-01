#!/usr/bin/env bash
#SBATCH -A xxx
#SBATCH -t 01:00:00
#SBATCH -o out_french_1d_n.txt
#SBATCH -e err_french_1d_n.txt
#SBATCH -n 1


ml SciPy-bundle/2022.05-foss-2022a
source ~/my_python/bin/activate
python french_1d_n.py $*

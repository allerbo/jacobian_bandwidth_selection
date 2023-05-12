#!/usr/bin/env bash
#SBATCH -A xxx
#SBATCH -t 01:00:00
#SBATCH -o out_french_2d_n.txt
#SBATCH -e err_french_2d_n.txt
#SBATCH -n 32


ml SciPy-bundle/2022.05-foss-2022a
source ~/my_python/bin/activate
python french_2d_n.py $*

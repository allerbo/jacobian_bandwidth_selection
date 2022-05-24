#!/usr/bin/env bash
#SBATCH -A xxxxxxxx
#SBATCH -t 05:00:00
#SBATCH -o out_synth_n.txt
#SBATCH -e err_synth_n.txt
#SBATCH -n 32


ml GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4
ml SciPy-bundle/2019.10-Python-3.7.4
export PYTHONPATH=$PYTHONPATH:~/python_libs/lib/python3.7/site-packages/
python synth_n.py $*

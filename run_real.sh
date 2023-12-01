for data in wood casp house bs energy
do
  sbatch --array=0-99 start_real.sh data=\"$data\"
done

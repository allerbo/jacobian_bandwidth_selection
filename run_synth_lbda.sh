for log_lbda in {-40..20..5}
do
  sbatch start_synth.sh lbda $log_lbda $*
done

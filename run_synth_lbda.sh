for log_lbda in {-40..30}
do
  sbatch start_synth_lbda.sh $log_lbda
done

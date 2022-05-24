for log_lbda in {-40..30}
do
  sbatch start_french_1d_lbda.sh $log_lbda
done

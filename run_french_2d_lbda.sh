for log_lbda in {-40..20..5}
do
  sbatch start_french_2d.sh lbda $log_lbda $*
done

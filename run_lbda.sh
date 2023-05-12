for seed in {0..999}
do
  sbatch start_synth_lbda.sh $seed
  sbatch start_french_1d_lbda.sh $seed
  sbatch start_french_2d_lbda.sh $seed
done

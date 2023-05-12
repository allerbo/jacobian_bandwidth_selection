for seed in {0..999}
do
  sbatch start_synth_n.sh $seed
  sbatch start_french_1d_n.sh $seed
  sbatch start_french_2d_n.sh $seed
done

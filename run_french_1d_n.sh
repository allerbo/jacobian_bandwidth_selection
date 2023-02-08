for n in {200..10..-10}
do
  sbatch start_french_1d.sh n $n 
done

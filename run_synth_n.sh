for n in {100..10..-5}
do
  sbatch start_synth.sh n $n $*
done

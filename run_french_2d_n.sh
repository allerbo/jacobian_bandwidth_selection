for n in {35..10..-5}
do
  sbatch start_french_2d.sh n $n $*
done

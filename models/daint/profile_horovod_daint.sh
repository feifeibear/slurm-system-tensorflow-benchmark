for num_node in 1 2 4 8 16 32 64 128 
do
  echo "doing " $num_node
  sbatch -N $num_node ./submit_replica_daint.sh
done

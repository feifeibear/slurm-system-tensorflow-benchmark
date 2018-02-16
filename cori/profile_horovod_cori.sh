for NET_NAME in vgg16; do
  for BATCH_SIZE in 256; do
    # for num_node in 2 4 8 16; do
    for num_node in 32 64 128; do
      echo "doing " $num_node
      sbatch -N $num_node ./submit_replica_cori.sh $BATCH_SIZE $NET_NAME 3 66
    done
  done
done

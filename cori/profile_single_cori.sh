mkdir -p ./logs
# for MODEL_NAME in vgg16 inception3 resnet50 resnet152; do
for MODEL_NAME in alexnet; do
  # for BATCH_SIZE in 96 128 256 512; do
  for BATCH_SIZE in 8 16 32 64; do
    echo $MODEL_NAME $BATCH_SIZE
    sbatch ./google-benchmarks_local.sh ${MODEL_NAME} ${BATCH_SIZE} 2 136
  done;
done;


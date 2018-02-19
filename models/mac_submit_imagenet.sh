#!/bin/bash -l
#SBATCH -p debug
#SBATCH -C knl,quad,cache
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -J google_benchmark
#SBATCH --output=benchmark_imagenet.%j.log
# set dataset
#export DATA_DIR=${SCRATCH}/data/cifar-10-batches-py
#export DATASET="cifar10"

. ~/Envs/horovod/bin/activate
export DATA_DIR=${SCRATCH}/data/imagenet
export DATASET="imagenet"

# set model
. ./imagenet/resnet50.env

# set batch size
export NUM_NODE=1
BATCH_SIZE=128

# set log dir
export CUR_DIR=`pwd`
export LOG_DIR=${CUR_DIR}/log/${DATASET}_${NETWORK_NAME}_${NUM_NODE}_${BATCH_SIZE}_batch_${OPT}
mkdir -p ${LOG_DIR}

#we need to divide epochs by number of cores
mpirun -np ${NUM_NODE} python3 ../scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--num_gpus=1 \
--local_parameter_device=cpu \
${PYTHON_FLAGS} \
--data_format=NHWC \
--batch_size=${BATCH_SIZE} \
--data_dir=${DATA_DIR} \
--train_dir=${LOG_DIR}/data_dir \
&> ${LOG_DIR}/performance.log
deactivate

#!/bin/bash -l
#SBATCH -p regular
#SBATCH -C knl,quad,cache
#SBATCH -t 5:00:00
#SBATCH -N 256
#SBATCH -L SCRATCH
#SBATCH -J google_benchmark
#SBATCH --output=benchmark_imagenet.%j.log

module load tensorflow/intel-horovod-mpi-head

# set KNL env
export OMP_NUM_THREADS=66
KMP_AFFINITY="granularity=fine,noverbose,compact,1,0"
KMP_SETTINGS=1
KMP_BLOCKTIME=1
export INTER_TH=3
export INTRA_TH=66

# set dataset
#export DATA_DIR=${SCRATCH}/data/cifar-10-batches-py
#export DATASET="cifar10"
export DATA_DIR=${SCRATCH}/data/imagenet
export DATASET="imagenet"

# set model
source ./imagenet/resnet50_8K.env

# set batch size
export NUM_NODE=${SLURM_JOB_NUM_NODES}
BATCH_SIZE=32

# set log dir
export CUR_DIR=`pwd`
export LOG_DIR=${CUR_DIR}/log/${DATASET}_${NETWORK_NAME}_${NUM_NODE}_${BATCH_SIZE}_batch_${OPT}
mkdir -p ${LOG_DIR}

#we need to divide epochs by number of cores
srun -N ${NUM_NODE} -n ${NUM_NODE} -c 272 python ../scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--num_gpus=1 \
--local_parameter_device=cpu \
--device=cpu \
${PYTHON_FLAGS} \
--data_format=NHWC \
--batch_size=${BATCH_SIZE} \
--data_dir=${DATA_DIR} \
--train_dir=${LOG_DIR}/data_dir \
--num_inter_threads=${INTER_TH} \
--num_intra_threads=${INTRA_TH} \
&> ${LOG_DIR}/performance.log

#!/bin/bash -l
#SBATCH -p debug
#SBATCH -C knl,quad,cache
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -J google_benchmark
#SBATCH --output=cifar_benchmark.%j.log

module load tensorflow/intel-horovod-mpi-head

# set KNL env
export OMP_NUM_THREADS=66
KMP_AFFINITY="granularity=fine,noverbose,compact,1,0"
KMP_SETTINGS=1
KMP_BLOCKTIME=1
export INTER_TH=3
export INTRA_TH=66

# set dataset
export DATA_DIR=${SCRATCH}/data/cifar-10-batches-py
export DATASET="cifar10"
#export DATA_DIR=/data2/fjr
#export DATASET="imagenet"

# set model
source ./cifar/resnet56.env

# set batch size
export NUM_NODE=${SLURM_JOB_NUM_NODES}
BATCH_SIZE=`expr 128 / ${NUM_NODE}`
FIRST_DECAY=`expr 80 / ${NUM_NODE}`
SECOND_DECAY=`expr 122 / ${NUM_NODE}`
echo "FIRST_DECAY : " $FIRST_DECAY " SECOND_DECAY: " $SECOND_DECAY

# set log dir
export CUR_DIR=`pwd`
export LOG_DIR=${CUR_DIR}/log/${DATASET}_${NETWORK_NAME}_${NUM_NODE}_${BATCH_SIZE}_batch_${OPT}
mkdir -p ${LOG_DIR}

#we need to divide epochs by number of cores
srun -N ${NUM_NODE} -n ${NUM_NODE} -c 272 python ../scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--num_gpus=1 \
--local_parameter_device=cpu \
${PYTHON_FLAGS} \
--data_format=NHWC \
--batch_size=${BATCH_SIZE} \
--data_dir=${DATA_DIR} \
--train_dir=${LOG_DIR}/data_dir \
--num_inter_threads=${INTER_TH} \
--num_intra_threads=${INTRA_TH} \
&> ${LOG_DIR}/performance.log

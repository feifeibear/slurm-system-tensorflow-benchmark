#!/bin/bash -l
#SBATCH -p debug 
#SBATCH -C knl,quad,cache
#SBATCH -N 1 
#SBATCH -t 00:20:00
#SBATCH -L SCRATCH
#SBATCH -J google_benchmark
#SBATCH --output=./slurm_log/eval_imagenet.%j.log

# Run the following commands on host_0 (10.0.0.1):
module load tensorflow/intel-horovod-mpi-head
# set KNL env
export OMP_NUM_THREADS=66
KMP_AFFINITY="granularity=fine,noverbose,compact,1,0"
KMP_SETTINGS=1
KMP_BLOCKTIME=1
export INTER_TH=3
export INTRA_TH=66



export DATA_DIR=${SCRATCH}/data/imagenet
export DATASET="imagenet"
#export DATA_DIR=/data2/fjr
#export DATASET="imagenet"
export NETWORK_NAME=resnet50
export BATCH_SIZE=128
export CUR_DIR=`pwd`
export NUM_NODE=8
export OPT=momentum
export LOG_DIR=${CUR_DIR}/log/${DATASET}_${NETWORK_NAME}_${NUM_NODE}_${BATCH_SIZE}_batch_${OPT}
mkdir -p ${LOG_DIR}

srun -N 1 -n 1 -c 272 python ../scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--eval=True \
--num_gpus=1 \
--device=cpu \
--use_datasets=False \
--variable_update=horovod \
--momentum=0.9 \
--batch_size=100 \
--data_name=${DATASET} \
--data_dir=${DATA_DIR} \
--eval_dir=${LOG_DIR}/eval_dir \
--train_dir=${LOG_DIR}/data_dir \
--optimizer=${OPT} \
--model=${NETWORK_NAME} \
--data_format=NHWC \
--num_inter_threads=${INTER_TH} \
--num_intra_threads=${INTRA_TH} \
> ${LOG_DIR}/eval.log

tail -f ${LOG_DIR}/eval.log

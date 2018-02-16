#!/bin/bash -l
#SBATCH -p debug
#SBATCH -N 1 
#SBATCH -C knl,quad,cache
#SBATCH -t 00:12:00
#SBATCH -L SCRATCH
#SBATCH -J google_benchmark 
#SBATCH --output=local_benchmark.%j.log

# module load tensorflow/intel-head
# module load tensorflow/intel-head-MKL-DNN
module load tensorflow/intel-horovod-mpi-head
export OMP_NUM_THREADS=66
KMP_AFFINITY="granularity=fine,noverbose,compact,1,0"
KMP_SETTINGS=1
KMP_BLOCKTIME=1

# load virtualenv
#export WORKON_HOME=~/Envs
#source $WORKON_HOME/tf-local/bin/activate
export MODEL_NAME=$1
export BATCH_SIZE=$2
export INTER_TH=$3
export INTRA_TH=$4


# train inception
#inception3 
SCRIPT=../scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py
srun -n 1 -N 1 -c 272 python ${SCRIPT} \
--num_gpus=1 \
--batch_size=${BATCH_SIZE} \
--num_warmup_batches=2 \
--num_batches=10 \
--data_format=NHWC \
--variable_update=parameter_server \
--local_parameter_device=cpu \
--device=cpu \
--optimizer=sgd \
--model=${MODEL_NAME} \
--data_name=imagenet \
--num_inter_threads=${INTER_TH} \
--num_intra_threads=${INTRA_TH} &> ./logs/${MODEL_NAME}_${BATCH_SIZE}_${INTER_TH}_${INTRA_TH}.log

# echo $MODEL_NAME
#--data_dir=/home/ubuntu/imagenet/

# deactivate virtualenv
#deactivate


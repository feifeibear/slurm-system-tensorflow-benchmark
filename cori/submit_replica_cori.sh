#!/bin/bash
#SBATCH -p regular 
#SBATCH -C knl,quad,cache
#SBATCH -t 01:00:00
#SBATCH -L SCRATCH
#SBATCH -J google_benchmark 
#SBATCH --output=horovod_benchmark.%j.log

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers
#   $3: variable_update: parameter_server/distributed_replicated
#   $4: real_data: true/false
module load tensorflow/intel-horovod-mpi-head
export OMP_NUM_THREADS=66
KMP_AFFINITY="granularity=fine,noverbose,compact,1,0"
KMP_SETTINGS=1
KMP_BLOCKTIME=1

mkdir -p log_horovod
export PYTHON=python
export UPDATE_METHOD=horovod
export BATCH_SIZE=$1
export NETWORK_NAME=$2
export INTER_TH=$3
export INTRA_TH=$4

srun -N ${SLURM_JOB_NUM_NODES} -n ${SLURM_JOB_NUM_NODES} -c 272 ${PYTHON} ../scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --num_gpus=1 --batch_size=${BATCH_SIZE} --model=${NETWORK_NAME} \
--variable_update=${UPDATE_METHOD} \
--num_inter_threads=${INTER_TH} \
--num_intra_threads=${INTRA_TH} \
&> ./log_horovod/${UPDATE_METHOD}_${SLURM_JOB_NUM_NODES}_nodes_${NETWORK_NAME}_${BATCH_SIZE}_${INTRA_TH}_${INTER_TH}.log
#--variable_update=replicated --all_reduce_spec=pscpu

# deactivate virtualenv
deactivate

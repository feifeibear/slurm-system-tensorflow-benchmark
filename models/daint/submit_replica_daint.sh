#!/bin/bash

#SBATCH --job-name=dist_google_benchmark
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu
#SBATCH --output=dist_benchmark_daint.%j.log

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers
#   $3: variable_update: parameter_server/distributed_replicated
#   $4: real_data: true/false

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
# module load TensorFlow/1.2.1-CrayGNU-17.08-cuda-8.0-python3
module load TensorFlow/1.4.1-CrayGNU-17.08-cuda-8.0-python3

# load virtualenv
export WORKON_HOME=$SCRATCH/fjr/Envs
source $WORKON_HOME/tf-uber/bin/activate
UPDATE_METHOD=horovod

mkdir -p srun_log
export PYTHON=python3
srun -N ${SLURM_JOB_NUM_NODES} -n ${SLURM_JOB_NUM_NODES} ${PYTHON} ../scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 \
--variable_update=${UPDATE_METHOD} > ./srun_log/${UPDATE_METHOD}_${SLURM_JOB_NUM_NODES}_nodes.log 
#--variable_update=replicated --all_reduce_spec=pscpu

# deactivate virtualenv
deactivate

#!/bin/bash
# Run the following commands on host_0 (10.0.0.1):
. ~/Envs/horovod/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR=/home/fangjr/dataset/cifar-10-batches-py
export DATASET="cifar10"
#export DATA_DIR=/data2/fjr
#export DATASET="imagenet"
export NUM_NODE=2
. ./cifar/resnet56.env
BATCH_SIZE=`expr 128 / ${NUM_NODE}`
FIRST_DECAY=`expr 80 / ${NUM_NODE}`
SECOND_DECAY=`expr 122 / ${NUM_NODE}`
echo "FIRST_DECAY : " $FIRST_DECAY " SECOND_DECAY: " $SECOND_DECAY
export CUR_DIR=`pwd`
export LOG_DIR=${CUR_DIR}/log/${DATASET}_${NETWORK_NAME}_${NUM_NODE}_${BATCH_SIZE}_batch_${OPT}
mkdir -p $LOG_DIR

python3 ../scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--num_gpus=2 \
--local_parameter_device=gpu \
${PYTHON_FLAGS} \
--batch_size=${BATCH_SIZE} \
--data_name=${DATASET} \
--data_dir=${DATA_DIR} \
--train_dir=${LOG_DIR}/data_dir \
--customized_lr="resnet56_cifar_128" \
> ${LOG_DIR}/performance.log

#parameter setting from
#https://github.com/yihui-he/resnet-cifar10-caffe/tree/master/resnet-56
#we need to divide epochs by number of cores
# mpirun -np ${NUM_NODE} python3 ../scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
# --num_gpus=1 \
# --local_parameter_device=gpu \
# ${PYTHON_FLAGS} \
# --batch_size=${BATCH_SIZE} \
# --data_name=${DATASET} \
# --data_dir=${DATA_DIR} \
# --train_dir=${LOG_DIR}/data_dir \
# --customized_lr="resnet56_cifar_128" \
# > ${LOG_DIR}/performance.log

deactivate

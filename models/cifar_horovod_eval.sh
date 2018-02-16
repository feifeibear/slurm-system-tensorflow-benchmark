# Run the following commands on host_0 (10.0.0.1):
export CUDA_VISIBLE_DEVICES=1
. ~/Envs/horovod/bin/activate
export DATA_DIR=/home/fangjr/dataset/cifar-10-batches-py
export DATASET="cifar10"
#export DATA_DIR=/data2/fjr
#export DATASET="imagenet"
export NETWORK_NAME=resnet56
export BATCH_SIZE=64
export CUR_DIR=`pwd`
export OPT=sgd
export LOG_DIR="${CUR_DIR}/log/${DATASET}_${NETWORK_NAME}_${BATCH_SIZE}_batch_${OPT}"
mkdir -p ${LOG_DIR}

python3 ../scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--eval=True \
--local_parameter_device=gpu \
--num_gpus=1 \
--batch_size=${BATCH_SIZE} \
--data_name=${DATASET} \
--data_dir=${DATA_DIR} \
--eval_dir=${LOG_DIR}/eval_dir \
--train_dir=${LOG_DIR}/data_dir \
--variable_update=horovod \
--model=${NETWORK_NAME} \
> ${LOG_DIR}/eval.log

deactivate

tail -f ${LOG_DIR}/eval.log

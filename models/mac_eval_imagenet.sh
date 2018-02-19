. ~/Envs/horovod/bin/activate
export CUDA_VISIBLE_DEVICES=2,3

export DATA_DIR=/data2/fjr
export DATASET="imagenet"
export NETWORK_NAME=resnet50
export BATCH_SIZE=128
export CUR_DIR=`pwd`
export NUM_NODE=1
export OPT=momentum
export LOG_DIR=${CUR_DIR}/log/${DATASET}_${NETWORK_NAME}_${NUM_NODE}_${BATCH_SIZE}_batch_${OPT}
mkdir -p ${LOG_DIR}

python ../scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--eval=True \
--num_gpus=1 \
--device=gpu \
--variable_update=horovod \
--use_datasets=False \
--batch_size=100 \
--data_name=${DATASET} \
--data_dir=${DATA_DIR} \
--eval_dir=${LOG_DIR}/eval_dir \
--train_dir=${LOG_DIR}/data_dir \
--model=${NETWORK_NAME} \
> ${LOG_DIR}/eval.log

tail -f ${LOG_DIR}/eval.log

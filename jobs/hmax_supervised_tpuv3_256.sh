
mode=train_and_eval
resnet_depth=50

base_learning_rate=0.1
use_tpu=True
train_steps=3558
train_batch_size=32768
eval_batch_size=1024
iterations_per_loop=2558
skip_host_call=True
num_cores=256
enable_lars=True
label_smoothing=0.1

experiment_name=$1  # finetune_BU_{bu_loss}_TD_{td_loss}_R50_lr0.1_T0.1
tpu_name=$2
echo 'tpu experiment_name is'
echo $experiment_name
echo 'tpu name is'
echo $tpu_name

export TPU_NAME=$tpu_name  # 'prj-selfsup-tpu'
export STORAGE_BUCKET='gs://serrelab'
# DATA_DIR=gs://imagenet_data/train/
gsutil mkdir $STORAGE_BUCKET/prj-hmax/results_eval
DATA_DIR=$STORAGE_BUCKET/imagenet_dataset/
MODEL_DIR=$STORAGE_BUCKET/prj-hmax/results/$experiment_name
EXPORT_DIR=$STORAGE_BUCKET/prj-hmax/exported/$experiment_name

python3 main.py \
  --tpu=$TPU_NAME\
  --data_dir=$DATA_DIR\
  --model_dir=$MODEL_DIR\
  --export_dir=$EXPORT_DIR\
  --resnet_depth=$resnet_depth\
  --mode=$mode\
  --base_learning_rate=$base_learning_rate\
  --train_steps=$train_steps\
  --train_batch_size=$train_batch_size\
  --eval_batch_size=$eval_batch_size\
  --iterations_per_loop=$iterations_per_loop\
  --skip_host_call=$skip_host_call\
  --num_cores=$num_cores\
  --enable_lars=$enable_lars\
  --label_smoothing=$label_smoothing

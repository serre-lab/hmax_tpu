mode=train_and_eval
depth=50

enable_lars=False
train_batch_size=256
base_learning_rate=0.1
use_tpu=True
dataset=herbarium

new_experiment_name=$1  # finetune_BU_{bu_loss}_TD_{td_loss}_R50_lr0.1_T0.1
old_experiment_name=$2  # "model.ckpt-93836"
checkpoint_name=$3  # "model.ckpt-93836"
tpu_name=$4


experiment_name=$1  # finetune_BU_{bu_loss}_TD_{td_loss}_R50_lr0.1_T0.1
tpu_name=$2
labels=64500


export TPU_NAME=$tpu_name  # 'prj-selfsup-tpu'
export STORAGE_BUCKET='gs://serrelab'
DATA_DIR=$STORAGE_BUCKET/prj-fossil/data/herbarium/

#gsutil mkdir $STORAGE_BUCKET/prj-fossils/
#gsutil mkdir $STORAGE_BUCKET/prj-fossils/results/
#gsutil mkdir $STORAGE_BUCKET/prj-fossils/exported/
# DATA_DIR=$STORAGE_BUCKET/imagenet_dataset/imagenet2012/5.0.0/
MODEL_DIR=$STORAGE_BUCKET/prj-fossil/results/$experiment_name
EXPORT_DIR=$STORAGE_BUCKET/prj-fossil/exported/$experiment_name


python main_herbarium.py --tpu=$TPU_NAME\
                         --model_dir=$MODEL_DIR\
                         --export_dir=$EXPORT_DIR\
  
 
  
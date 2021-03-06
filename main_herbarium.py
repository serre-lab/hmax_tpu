
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import glob 
from absl import app
from absl import flags
from absl import logging
from tqdm.auto import tqdm
import numpy as np
import csv
import pandas as pd
import re, math
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa
from common import inference_warmup
from common import tpu_profiler_hook
from hyperparameters import common_hparams_flags
from hyperparameters import common_tpu_flags
from hyperparameters import flags_to_params
from hyperparameters import params_dict
from configs import resnet_config
from losses import batch_hard_triplet_loss
from models.resnet_model_triplet import get_triplet_model
import random
from classification_models.keras import Classifiers 

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

#common_tpu_flags.define_common_tpu_flags()
#common_hparams_flags.define_common_hparams_flags()
#FLAGS = flags.FLAGS
#import efficientnet.tfkeras as efn
class CFG:
    N_CLASSES = 64500
    IMAGE_SIZE = [2000, 2000]
    EPOCHS = 1
    if IMAGE_SIZE[0] == 256:
        BATCH_SIZE = 64 * 8#strategy.num_replicas_in_sync
    elif IMAGE_SIZE[0] == 384:
        BATCH_SIZE = 32 * 8#strategy.num_replicas_in_sync
    elif IMAGE_SIZE[0] == 600:
        BATCH_SIZE = 16 * 8#strategy.num_replicas_in_sync
    elif IMAGE_SIZE[0] == 2000:
        BATCH_SIZE = 4 * 1#strategy.num_replicas_in_sync
    else:
        BATCH_SIZE = 16 * 8
print(CFG.IMAGE_SIZE)
flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

AUTO = tf.data.experimental.AUTOTUNE
if CFG.IMAGE_SIZE[0]==256:
    TRAINING_FILENAMES =  tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/train/*.tfrec')
    VALIDATION_FILENAMES =  tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/validation/*.tfrec')
    TESTING_FILENAMES = tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/test/*.tfrec')
elif CFG.IMAGE_SIZE[0]==384:
    TRAINING_FILENAMES =  tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/384/train-384/*.tfrec')
    VALIDATION_FILENAMES =  tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/384/val-384/*.tfrec')
    TESTING_FILENAMES = tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/384/test-384/*.tfrec')
elif CFG.IMAGE_SIZE[0]==600:
    TRAINING_FILENAMES =  tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/600/train_2/*.tfrec')
    random.shuffle(TRAINING_FILENAMES)
    TRAINING_FILENAMES = TRAINING_FILENAMES[:int(len(TRAINING_FILENAMES)*0.9)]
    TRAINING_FILENAMES = [f  for f in TRAINING_FILENAMES]
    VALIDATION_FILENAMES = TRAINING_FILENAMES[int(len(TRAINING_FILENAMES)*0.9):] #tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/600/train/*.tfrec')
    TESTING_FILENAMES = tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/600/test_2/*.tfrec')
elif CFG.IMAGE_SIZE[0]==1600:
    TRAINING_FILENAMES =  tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/1600/train_4/*.tfrec')
    random.shuffle(TRAINING_FILENAMES)
    TRAINING_FILENAMES = [f  for f in TRAINING_FILENAMES]
    TRAINING_FILENAMES = TRAINING_FILENAMES[:int(len(TRAINING_FILENAMES)*0.9)]
    VALIDATION_FILENAMES = TRAINING_FILENAMES[int(len(TRAINING_FILENAMES)*0.9):] #tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/600/train/*.tfrec')
    TESTING_FILENAMES = tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/1600/test_4/*.tfrec')
elif CFG.IMAGE_SIZE[0]==2000:
    tf.config.experimental.set_lms_enabled(True)
    TRAINING_FILENAMES =  glob.glob('/cifs/data/tserre_lrs/projects/prj_fossils/data/raw_data/Herbarium_2021_FGVC8/tfrecords/train_3/*.tfrec')#tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/600/train_2/*.tfrec')
    random.shuffle(TRAINING_FILENAMES)
    TRAINING_FILENAMES = [f  for f in TRAINING_FILENAMES]
    TRAINING_FILENAMES = TRAINING_FILENAMES[:int(len(TRAINING_FILENAMES)*0.9)]
    VALIDATION_FILENAMES = TRAINING_FILENAMES[int(len(TRAINING_FILENAMES)*0.9):] #tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/600/train/*.tfrec')
    TESTING_FILENAMES = glob.glob('/cifs/data/tserre_lrs/projects/prj_fossils/data/raw_data/Herbarium_2021_FGVC8/tfrecords/test_3/*.tfrec')#tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/600/test_2/*.tfrec')
else:
    print('NOT implemented')
    pass 
NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES//4//CFG.BATCH_SIZE
NUM_TESTING_IMAGES = count_data_items(TESTING_FILENAMES)
TEST_STEPS = NUM_TESTING_IMAGES#//CFG.BATCH_SIZE
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
VALIDATION_STEPS = NUM_VALIDATION_IMAGES//CFG.BATCH_SIZE
NUM_TEST_IMAGES = count_data_items(TESTING_FILENAMES)
print('Dataset: {} unlabeled test images'.format(NUM_TEST_IMAGES))
print('Dataset: {} labeled train images'.format(NUM_TRAINING_IMAGES))
print('Dataset: {} labeled validation images'.format(NUM_VALIDATION_IMAGES))

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)  # image format uint8 [0,255]
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*CFG.IMAGE_SIZE, 3])
    return image

def get_idx(image, idnum):
    idnum = tf.strings.split(idnum, sep='/')[6]
    idnum = tf.strings.regex_replace(idnum, ".jpg", "")
    idnum = tf.strings.to_number(idnum, out_type=tf.int64)
    return image, idnum

def onehot(image,label):
    return image,tf.one_hot(label, CFG.N_CLASSES)

def onehot_triplet(input,label2):  
    return input[0],tf.one_hot(input[1], CFG.N_CLASSES),tf.one_hot(label2, CFG.N_CLASSES)
    
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label
def data_augment_triplet(image, label,label2):
    image = tf.image.random_flip_left_right(image)
    return (image, label),label2

def read_unlabeled_tfrecord(example):

    if CFG.IMAGE_SIZE[0]==256:
        UNLABELED_TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_idx': tf.io.FixedLenFeature([], tf.string)
        }
    else: 
        UNLABELED_TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_idx': tf.io.FixedLenFeature([], tf.int64)
        }


    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['image_idx']
    return image, idnum

def read_labeled_tfrecord(example):

    if CFG.IMAGE_SIZE[0]==256:
        LABELED_TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_idx': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        }
    else: 
        LABELED_TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_idx': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = example['label']
    return image, label

def read_labeled_tfrecord_triplet(example):
    LABELED_TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_idx': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = example['label']
    return (image, label),label

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    return dataset
def load_dataset_triplet(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord_triplet if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    return dataset

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES)
    dataset = dataset.map(onehot, num_parallel_calls=AUTO)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    #dataset = dataset.cache()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(CFG.BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_training_dataset_triplet():
    dataset = load_dataset_triplet(TRAINING_FILENAMES)
    dataset = dataset.map(onehot_triplet, num_parallel_calls=AUTO)
    dataset = dataset.map(data_augment_triplet, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(CFG.BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset_triplet(ordered=False):
    dataset = load_dataset_triplet(VALIDATION_FILENAMES,ordered=ordered)
    dataset = dataset.map(onehot_triplet, num_parallel_calls=AUTO)
    dataset = dataset.batch(CFG.BATCH_SIZE)
    #dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES,ordered=ordered)
    dataset = dataset.map(onehot, num_parallel_calls=AUTO)
    dataset = dataset.batch(CFG.BATCH_SIZE)
    #dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=True, augmented=False):
    dataset = load_dataset(TESTING_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.map(get_idx, num_parallel_calls=AUTO)
    dataset = dataset.batch(CFG.BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset



def get_model(base_arch='Nasnet',weights='imagenet',include_top=True):

    if base_arch == 'Nasnet':
        base_model = tf.keras.applications.NASNetLarge(
                                     input_shape=(*CFG.IMAGE_SIZE, 3),
                                    include_top=False,
                                    weights=weights,
                                    pooling='avg',
                                    
                            )
    elif base_arch == 'Resnet50v2':
        base_model = tf.keras.applications.ResNet50V2(weights=weights, 
                                    include_top=False, 
                                    pooling='avg',
                                    input_shape=(*CFG.IMAGE_SIZE, 3))
        

    elif base_arch=='EfficientNet':
        conv_model = tf.keras.applications.efficientnet.EfficientNetB0(
                        include_top=False, weights=weights,
                        input_shape=(*CFG.IMAGE_SIZE, 3), )
        base_model = tf.keras.models.Sequential()
        base_model.add(conv_model)
        base_model.add(L.GlobalMaxPooling2D(name="gap"))
        #avoid overfitting
        base_model.add(L.Dropout(dropout_rate=0.2, name="dropout_out"))
                                
    elif base_arch == 'Resnext101':
        base_model, _ = Classifiers.get('resnext101')
    if include_top:
        model = tf.keras.Sequential([
        base_model,
        L.Dense(CFG.N_CLASSES, activation='softmax')
        ])
    
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=[tfa.metrics.F1Score(CFG.N_CLASSES, average='macro'),
                           'accuracy',
                           tf.keras.metrics.TopKCategoricalAccuracy(k=1,name='top1acc'),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=3,name='top3acc'),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=5,name='top5acc')])
    
    return model

def main_triplet(unused_argv):
    MAIN_CKP_DIR = 'ckpt/'
    os.makedirs(MAIN_CKP_DIR,exist_ok=True)
    params = params_dict.ParamsDict(
      resnet_config.RESNET_CFG, resnet_config.RESNET_RESTRICTIONS)
    params = params_dict.override_params_dict(
      params, FLAGS.config_file, is_strict=True)
    params = params_dict.override_params_dict(
          params, FLAGS.params_override, is_strict=True)
    
    params = flags_to_params.override_params_from_input_flags(params, FLAGS)
    # Save params for transfer to GCS
    np.savez( os.path.join(MAIN_CKP_DIR,'params.npz'), **params.as_dict())
    
    params.validate()
    params.lock()
    print(FLAGS.gcp_project)
    print(FLAGS.tpu_zone)
    print(FLAGS.tpu)
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    input_image_shape = (CFG.IMAGE_SIZE[0],CFG.IMAGE_SIZE[1],3)
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=2, min_delta=0.001,
                                                          monitor='val_logits_f1_score', mode='max')
    es_callback = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, 
                                                       monitor='val_logits_f1_score', mode='max',
                                                       restore_best_weights=True)
    
    
    
    embeddings_unit = 256

    with strategy.scope():
        for arch in ['Resnet50v2']:
            for weights in [None,'imagenet']:
                print('Creating model')
                model  = get_triplet_model(input_shape=input_image_shape,embedding_units=256,nb_classes=CFG.N_CLASSES)
                
                if not weights: 
                    ckpt_file = MAIN_CKP_DIR+'%s_NO_imagenet_%s_best.h5'%(arch,weights)
                else: 
                    ckpt_file = MAIN_CKP_DIR+'%s_imagenet_%s_best.h5'%(arch,weights)
                print('saving ckpts in : ',ckpt_file)
                chk_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_file, monitor='val_logits_f1_score', 
                                                          save_best_only=True,
                                                          save_weights_only=True, 
                                                          mode='max')
                tb_callback = tf.keras.callbacks.TensorBoard(
                        log_dir=MAIN_CKP_DIR+'logs', histogram_freq=0, write_graph=True,
                        write_images=False, write_steps_per_second=False, update_freq='epoch',
                        profile_batch=2, embeddings_freq=0, embeddings_metadata=None
                            )
                
                model.summary()
                model.compile(loss={'embedding':batch_hard_triplet_loss, 
                                    'logits': 'categorical_crossentropy'},
                              loss_weights={'embedding': 0.3,
                            'logits': 1.0},
                            optimizer='adam',
                            metrics={'logits': [tfa.metrics.F1Score(CFG.N_CLASSES, average='macro'),
                           'accuracy',
                           tf.keras.metrics.TopKCategoricalAccuracy(k=1,name='top1acc'),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=3,name='top3acc'),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=5,name='top5acc')]},)
                history = model.fit(
                            get_training_dataset().repeat(), 
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=CFG.EPOCHS,
                            validation_data=get_validation_dataset(),
                            callbacks=[lr_callback, chk_callback, es_callback,tb_callback],
                            verbose=1)
                df= pd.DataFrame(history.history)
                
                
                if not weights: 
                    model.save_weights(MAIN_CKP_DIR+'%s_%s_last_triplet_%d.h5'%(arch,'NO_imagenet',embeddings_unit))
                    df.to_csv(MAIN_CKP_DIR+'%s_%s_last_triplet_logger_%d.csv'%(arch,'NO_imagenet',embeddings_unit))
                else: 
                    model.save_weights(MAIN_CKP_DIR+'%s_%s_last_triplet_%d.h5'%(arch,weights,embeddings_unit))
                    df.to_csv(MAIN_CKP_DIR+'%s_%s_last_triplet_logger_%d.csv'%(arch,'imagenet',embeddings_unit))

                print('Calculating predictions...')
                test_ds = get_test_dataset()
                
                predictions = {}
                
                for imgs, idx in test_ds:
                    idx = np.array(idx,np.int64)
                    preds = np.argmax(model(imgs),-1)
                    for pred_id,pred in zip(idx,preds):
                        predictions[pred_id]=pred
                print(len(predictions.keys()))
                header = ['Id','Predicted']
                with open(f'{MAIN_CKP_DIR}_submission_triplet_{arch}_{weights}_{embeddings_unit}.csv','w',encoding='UTF8',newline='') as f:
                    writer =csv.writer(f)
                    writer.writerow(header)

                    for pred_id in predictions:
                        writer.writerow([pred_id,predictions[pred_id]])





def main():
    MAIN_CKP_DIR = 'ckpt/'
    os.makedirs(MAIN_CKP_DIR,exist_ok=True)
    #params = params_dict.ParamsDict(
    #  resnet_config.RESNET_CFG, resnet_config.RESNET_RESTRICTIONS)
    #params = params_dict.override_params_dict(
    #  params, FLAGS.config_file, is_strict=True)
    #params = params_dict.override_params_dict(
    #      params, FLAGS.params_override, is_strict=True)
    
    #params = flags_to_params.override_params_from_input_flags(params, FLAGS)
    # Save params for transfer to GCS
    #np.savez( os.path.join(MAIN_CKP_DIR,'params.npz'), **params.as_dict())
    #if CFG.SIZE != 2000:
        #params.validate()
        #params.lock()
        #print(FLAGS.gcp_project)
        #print(FLAGS.tpu_zone)
        #print(FLAGS.tpu)
    
 
    try: 
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
    except ValueError: # detect GPUs
        print('training on GPU')
        strategy = tf.distribute.MirroredStrategy()
    
    
    
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    
    with strategy.scope():
        # NasNET
        for arch in ['Resnet50v2']:
            for weights in [None,'imagenet']:
                print('Creating model')
                model = get_model(arch,weights)
                model.summary()
                tb_callback = tf.keras.callbacks.TensorBoard(
                        log_dir=MAIN_CKP_DIR+'logs_%s_%s_best_%d'%(arch,weights,CFG.IMAGE_SIZE[0]), histogram_freq=0, write_graph=True,
                        write_images=False,  update_freq='epoch',
                        profile_batch=2, embeddings_freq=0, embeddings_metadata=None
                            )
                lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=2, min_delta=0.001,
                                                          monitor='val_loss', mode='min')
                es_callback = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, 
                                                       monitor='val_loss', mode='min',
                                                       restore_best_weights=True)
                if not weights: 
                    ckpt_file = MAIN_CKP_DIR+'%s_NO_imagenet_%s_best_%d.h5'%(arch,weights,CFG.IMAGE_SIZE[0])
                else: 
                    ckpt_file = MAIN_CKP_DIR+'%s_imagenet_%s_best_%d.h5'%(arch,weights,CFG.IMAGE_SIZE[0])
                print('saving ckpts in : ',ckpt_file)
                chk_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_file, monitor='val_loss', 
                                                          save_best_only=True,
                                                          save_weights_only=True, 
                                                          mode='min')
                try: 
                    print('loading weights from file %s'%ckpt_file)
                    model.load_weights(ckpt_file)
                except:
                    print('loading failed, starting from scratch')
                    
                history = model.fit(
                            get_training_dataset(), 
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=CFG.EPOCHS,
                            validation_data=get_validation_dataset(),
                            callbacks=[lr_callback, chk_callback, es_callback,tb_callback],
                            verbose=1)
                df= pd.DataFrame(history.history)
                
                if not weights: 
                    model.save_weights(MAIN_CKP_DIR+'%s_%s_last.h5'%(arch,'NO_imagenet'))
                    df.to_csv(MAIN_CKP_DIR+'%s_%s_last_history.csv'%(arch,'NO_imagenet'))
                else: 
                    model.save_weights(MAIN_CKP_DIR+'%s_%s_last.h5'%(arch,weights))
                    df.to_csv(MAIN_CKP_DIR+'%s_%s_last_history.csv'%(arch,weights))

            
                #cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.
                #images_ds = cmdataset.map(lambda image, label: image)
                #labels_ds = cmdataset.map(lambda image, label: label).unbatch()
                #cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch
                #cm_probabilities = model.predict(images_ds, steps=VALIDATION_STEPS)
                #cm_predictions = np.argmax(cm_probabilities, axis=-1)
                #print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)
                #print("Predicted labels: ", cm_predictions.shape, cm_predictions)
                print('Calculating predictions...')
                test_ds = get_test_dataset()
                
                predictions = {}
                
                for imgs, idx in test_ds:
                    idx = np.array(idx,np.int64)
                    preds = np.argmax(model(imgs),-1)
                    for pred_id,pred in zip(idx,preds):
                        predictions[pred_id]=pred
                print(len(predictions.keys()))
                header = ['Id','Predicted']
                with open(f'{MAIN_CKP_DIR}_submission_{arch}_{weights}_{CFG.IMAGE_SIZE[0]}.csv','w',encoding='UTF8',newline='') as f:
                    writer =csv.writer(f)
                    writer.writerow(header)

                    for pred_id in predictions:
                        writer.writerow([pred_id,predictions[pred_id]])

            
if __name__ == '__main__':
  #tf.logging.set_verbosity(tf.logging.INFO)
  main()
  #app.run(main_triplet)

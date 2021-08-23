
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

import numpy as np
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


common_tpu_flags.define_common_tpu_flags()
common_hparams_flags.define_common_hparams_flags()
FLAGS = flags.FLAGS
#import efficientnet.tfkeras as efn



AUTO = tf.data.experimental.AUTOTUNE
TRAINING_FILENAMES =  tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/train/*.tfrec')
VALIDATION_FILENAMES =  tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/validation/*.tfrec')
#tpu_name=os.getenv('TPU_NAME')

#cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
#tpu=tpu_name)


class CFG:
    N_CLASSES = 64500
    IMAGE_SIZE = [256, 256]
    EPOCHS = 25
    BATCH_SIZE = 16 * 4#strategy.num_replicas_in_sync



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
    
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_idx': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum=example['image_idx']
    return image, idnum

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_idx': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = example['label']
    return image, label

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

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES)
    dataset = dataset.map(onehot, num_parallel_calls=AUTO)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(CFG.BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset():
    dataset = load_dataset(VALIDATION_FILENAMES)
    dataset = dataset.map(onehot, num_parallel_calls=AUTO)
    dataset = dataset.batch(CFG.BATCH_SIZE)
    #dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def get_model(base_arch='Nasnet',weights='imagenet'):

    if base_arch == 'Nasnet':
        base_model = tf.keras.applications.NASNetLarge(
                                     input_shape=(*CFG.IMAGE_SIZE, 3),
                                    include_top=False,
                                    weights=weights,
                                    input_tensor=None,
                                    pooling=None,
                                    
                            )
    elif base_arch == 'Resnet50v2':
        base_model = tf.keras.applications.ResNet50V2(weights=weights, 
                                    include_top=False, 
                                    pooling='avg',
                                    input_shape=(*CFG.IMAGE_SIZE, 3))
                                
    model = tf.keras.Sequential([
        base_model,
        L.Dense(CFG.N_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=[tfa.metrics.F1Score(CFG.N_CLASSES, average='macro')])
    
    return model

def main(unused_argv):
    MAIN_CKP_DIR = 'gs://serrelab/prj-fossil/exported/'
    params = params_dict.ParamsDict(
      resnet_config.RESNET_CFG, resnet_config.RESNET_RESTRICTIONS)
    params = params_dict.override_params_dict(
      params, FLAGS.config_file, is_strict=True)
    params = params_dict.override_params_dict(
          params, FLAGS.params_override, is_strict=True)
    
    params = flags_to_params.override_params_from_input_flags(params, FLAGS)
    # Save params for transfer to GCS
    np.savez('params.npz', **params.as_dict())
    
    params.validate()
    params.lock()
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu if (FLAGS.tpu or params.use_tpu) else '',
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)

    strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

    print("Number of accelerators: ", strategy.num_replicas_in_sync)

    with strategy.scope():
        # NasNET
        for arch in ['resnet5-v2','Nasnet']:
            for weights in ['imagenet',None]:
                model = get_model()
                model.summary()
                lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=2, min_delta=0.001,
                                                          monitor='val_loss', mode='min')
                es_callback = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, 
                                                       monitor='val_loss', mode='min',
                                                       restore_best_weights=True)
                chk_callback = tf.keras.callbacks.ModelCheckpoint(MAIN_CKP_DIR+'%s_imagenet_%s_best.h5'%(arch,weights), monitor='val_loss', 
                                                          save_best_only=True,
                                                          save_weights_only=True, 
                                                          mode='min')
                history = model.fit(
                            get_training_dataset(), 
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=CFG.EPOCHS,
                            validation_data=get_validation_dataset(),
                            callbacks=[lr_callback, chk_callback, es_callback],
                            verbose=2)
            
                model.save_weights(MAIN_CKP_DIR+'%s_%s_last.h5'%(arch,weights))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)


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

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

common_tpu_flags.define_common_tpu_flags()
common_hparams_flags.define_common_hparams_flags()
FLAGS = flags.FLAGS
#import efficientnet.tfkeras as efn
class CFG:
    N_CLASSES = 64500
    IMAGE_SIZE = [256, 256]
    EPOCHS = 2
    BATCH_SIZE = 16 * 4#strategy.num_replicas_in_sync
    
flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

AUTO = tf.data.experimental.AUTOTUNE
TRAINING_FILENAMES =  tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/train/*.tfrec')
VALIDATION_FILENAMES =  tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/validation/*.tfrec')
TESTING_FILENAMES = tf.io.gfile.glob('gs://serrelab/prj-fossil/data/herbarium/test/*.tfrec')
NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES//CFG.BATCH_SIZE
NUM_TESTING_IMAGES = count_data_items(TESTING_FILENAMES)
TEST_STEPS = NUM_TESTING_IMAGES//CFG.BATCH_SIZE
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
VALIDATION_STEPS = NUM_VALIDATION_IMAGES//CFG.BATCH_SIZE


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

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES,ordered=ordered)
    dataset = dataset.map(onehot, num_parallel_calls=AUTO)
    dataset = dataset.batch(CFG.BATCH_SIZE)
    #dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_testing_dataset():
    dataset = load_dataset(TESTING_FILENAMES,labeled=False,ordered=True)
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
                  metrics=[tfa.metrics.F1Score(CFG.N_CLASSES, average='macro'),
                           'accuracy',
                           tf.keras.metrics.TopKCategoricalAccuracy(k=1,name='top1acc'),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=3,name='top3acc'),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=5,name='top5acc')])
    
    return model

def main(unused_argv):
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
    
    #cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
    #  FLAGS.tpu if (FLAGS.tpu or params.use_tpu) else '',
    #  zone=FLAGS.tpu_zone,
    #  project=FLAGS.gcp_project)
    #tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    #tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    #tf.config.experimental_connect_to_cluster(tpu)
    #tf.tpu.experimental.initialize_tpu_system(tpu)
    #strategy = tf.distribute.experimental.TPUStrategy(tpu)
    #strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    
    with strategy.scope():
        # NasNET
        for arch in ['Resnet50v2','Nasnet']:
            for weights in [None,'imagenet']:
                model = get_model(arch,weights)
                model.summary()
                lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=2, min_delta=0.001,
                                                          monitor='val_loss', mode='min')
                es_callback = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, 
                                                       monitor='val_loss', mode='min',
                                                       restore_best_weights=True)
                if not weights: 
                    ckpt_file = MAIN_CKP_DIR+'%s_NO_imagenet_%s_best.h5'%(arch,weights)
                else: 
                    ckpt_file = MAIN_CKP_DIR+'%s_imagenet_%s_best.h5'%(arch,weights)
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
                            callbacks=[lr_callback, chk_callback, es_callback],
                            verbose=1)

                if not weights: 
                    model.save_weights(MAIN_CKP_DIR+'%s_%s_last.h5'%(arch,'NO_imagenet'))
                else: 
                    model.save_weights(MAIN_CKP_DIR+'%s_%s_last.h5'%(arch,weights))

            
            cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.
            images_ds = cmdataset.map(lambda image, label: image)
            labels_ds = cmdataset.map(lambda image, label: label).unbatch()
            cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch
            cm_probabilities = model.predict(images_ds, steps=VALIDATION_STEPS)
            cm_predictions = np.argmax(cm_probabilities, axis=-1)
            print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)
            print("Predicted labels: ", cm_predictions.shape, cm_predictions)
                
            test_ds = get_testing_dataset() # since we are splitting the dataset and iterating separately on images and ids, order matters.
            print('Computing predictions...')
            test_images_ds = test_ds.map(lambda image, idnum: image)
            probabilities = model.predict(test_images_ds, steps=TEST_STEPS)
            predictions = np.argmax(probabilities, axis=-1)
            print(predictions)
    
            print('Generating submission.csv file...')
            test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
            test_ids = next(iter(test_ids_ds.batch(NUM_TESTING_IMAGES))).numpy().astype('U') # all in one batch
            np.savetxt('%s_submission.csv'%ckpt_file[:-3], np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
        
if __name__ == '__main__':
  #tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)

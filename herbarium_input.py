# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Efficient Herbarium input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import functools
import os
from absl import logging
import tensorflow.compat.v1 as tf
from models import resnet_preprocessing


def image_serving_input_fn():
  """Serving input fn for raw images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    image = resnet_preprocessing.preprocess_image(
        image_bytes=image_bytes, is_training=False)
    return image

  # TODO: here ? 
  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf.estimator.export.TensorServingInputReceiver(
      features=images, receiver_tensors=image_bytes_list)


class HerbariumTFExampleInput(object):
  """Base class for Herbarium input_fn generator.

  Attributes:
    image_preprocessing_fn: function to preprocess images
    is_training: `bool` for whether the input is for training
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    transpose_input: 'bool' for whether to use the double transpose trick
    image_size: size of images
    num_parallel_calls: `int` for the number of parallel threads.
    include_background_label: `bool` for whether to include the background label
    augment_name: `string` that is the name of the augmentation method
        to apply to the image. `autoaugment` if AutoAugment is to be used or
        `randaugment` if RandAugment is to be used. If the value is `None` no
        no augmentation method will be applied applied. See autoaugment.py
        for more details.
    randaug_num_layers: 'int', if RandAug is used, what should the number of
      layers be. See autoaugment.py for detailed description.
    randaug_magnitude: 'int', if RandAug is used, what should the magnitude
      be. See autoaugment.py for detailed description.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               is_training,
               use_bfloat16,
               image_size=256,
               transpose_input=False,
               num_parallel_calls=8,
               include_background_label=False,
               augment_name=None,
               randaug_num_layers=None,
               randaug_magnitude=None):
    self.image_preprocessing_fn = resnet_preprocessing.preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.transpose_input = transpose_input
    self.image_size = image_size
    self.num_parallel_calls = num_parallel_calls
    self.include_background_label = include_background_label
    self.augment_name = augment_name
    self.randaug_num_layers = randaug_num_layers
    self.randaug_magnitude = randaug_magnitude

  def set_shapes(self, batch_size, images, labels):
    """Statically set the batch_size dimension."""
    if self.transpose_input:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([None, None, None, batch_size])))
      images = tf.reshape(images, [-1])
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))
    else:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))

    return images, labels

  def dataset_parser(self, value):
    """Parses an image and its label from a serialized ResNet-50 TFExample.

    Args:
      value: serialized string containing an Herbarium TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    
    LABELED_TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_idx': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = example['label']
    """
    #keys_to_features = {
    #    'image/encoded': tf.FixedLenFeature((), tf.string, ''),
    #    'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
    #}
    keys_to_features = {
       'image': tf.FixedLenFeature((), tf.string, ''),
       'label': tf.FixedLenFeature([], tf.int64, -1),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image'], shape=[])

    image = self.image_preprocessing_fn(
        image_bytes=image_bytes,
        is_training=self.is_training,
        image_size=self.image_size,
        use_bfloat16=self.use_bfloat16,
        augment_name=self.augment_name,
        randaug_num_layers=self.randaug_num_layers,
        randaug_magnitude=self.randaug_magnitude)

    label = tf.cast(
        tf.reshape(parsed['label'], shape=[]), dtype=tf.int32)

    if not self.include_background_label:
      # 'image/class/label' is encoded as an integer from 1 to num_label_classes
      # In order to generate the correct one-hot label vector from this number,
      # we subtract the number by 1 to make it in [0, num_label_classes).
      label -= 1

    return image, label

  @abc.abstractmethod
  def make_source_dataset(self, index, num_hosts):
    """Makes dataset of serialized TFExamples.

    The returned dataset will contain `tf.string` tensors, but these strings are
    serialized `TFExample` records that will be parsed by `dataset_parser`.

    If self.is_training, the dataset should be infinite.

    Args:
      index: current host index.
      num_hosts: total number of hosts.

    Returns:
      A `tf.data.Dataset` object.
    """
    return

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
        `params['batch_size']` is always provided and should be used as the
        effective batch size.

    Returns:
      A `tf.data.Dataset` object.
    """

    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.estimator.tpu.RunConfig for details.
    batch_size = params['batch_size']

    # TODO(dehao): Replace the following with params['context'].current_host
    if 'context' in params:
      current_host = params['context'].current_input_fn_deployment()[1]
      num_hosts = params['context'].num_hosts
    else:
      current_host = 0
      num_hosts = 1

    dataset = self.make_source_dataset(current_host, num_hosts)

    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            self.dataset_parser,
            batch_size=batch_size,
            num_parallel_batches=self.num_parallel_calls,
            drop_remainder=True))

    # Transpose for performance on TPU
    if self.transpose_input:
      dataset = dataset.map(
          lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
          num_parallel_calls=self.num_parallel_calls)

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class HerbariumInput(HerbariumTFExampleInput):
  """Generates Herbarium input_fn from a series of TFRecord files.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:

      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/Herbarium_to_gcs.py
  """

  def __init__(self,
               is_training,
               use_bfloat16,
               transpose_input,
               data_dir,
               image_size=256,
               num_parallel_calls=8,
               cache=False,
               dataset_split=None,
               shuffle_shards=False,
               include_background_label=False,
               augment_name=None,
               randaug_num_layers=None,
               randaug_magnitude=None):
    """Create an input from TFRecord files.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      data_dir: `str` for the directory of the training and validation data; if
        'null' (the literal string 'null') or implicitly False then construct a
        null pipeline, consisting of empty images and blank labels.
        Otherwise, specify the url of the bucket.
      image_size: `int` image height and width.
      num_parallel_calls: concurrency level to use when reading data from disk.
      cache: if true, fill the dataset by repeating from its cache
      dataset_split: If provided, must be one of 'train' or 'validation' and
        specifies the dataset split to read, overriding the default set by
        is_training. In this case, is_training specifies whether the data is
        augmented.
      shuffle_shards: Whether to shuffle the dataset shards.
      include_background_label: Whether to include the background label. If
        this is True, then num_label_classes should be 1001. If False, then
        num_label_classes should be 1000.
      augment_name: `string` that is the name of the augmentation method
        to apply to the image. `autoaugment` if AutoAugment is to be used or
        `randaugment` if RandAugment is to be used. If the value is `None` no
        no augmentation method will be applied applied. See autoaugment.py
        for more details.
      randaug_num_layers: 'int', if RandAug is used, what should the number of
        layers be. See autoaugment.py for detailed description.
      randaug_magnitude: 'int', if RandAug is used, what should the magnitude
        be. See autoaugment.py for detailed description.
    """
    super(HerbariumInput, self).__init__(
        is_training=is_training,
        image_size=image_size,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input,
        include_background_label=include_background_label,
        augment_name=augment_name,
        randaug_num_layers=randaug_num_layers,
        randaug_magnitude=randaug_magnitude)
    self.data_dir = data_dir
    # TODO(b/112427086):  simplify the choice of input source
    # Thomas: this should be the url of the bucket, e.g 'gs://serrelab/prj-fossil/data/herbarium/'
    if self.data_dir == 'null' or not self.data_dir:
      self.data_dir = None
    self.num_parallel_calls = num_parallel_calls
    self.cache = cache
    self.dataset_split = dataset_split
    self.shuffle_shards = shuffle_shards

  def _get_null_input(self, data):
    """Returns a null image (all black pixels).

    Args:
      data: element of a dataset, ignored in this method, since it produces the
        same null image regardless of the element.

    Returns:
      a tensor representing a null image.
    """
    del data  # Unused since output is constant regardless of input
    return tf.zeros([self.image_size, self.image_size, 3],
                    tf.bfloat16 if self.use_bfloat16 else tf.float32)

  def dataset_parser(self, value):
    """See base class."""
    if not self.data_dir:
      return value, tf.constant(0, tf.int32)
    return super(HerbariumInput, self).dataset_parser(value)

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    if not self.data_dir:
      tf.logging.info('Undefined data_dir implies null input')
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

    # Shuffle the filenames to ensure better randomization.
    # TODO: update with correct paths
    if not self.dataset_split:
      file_pattern = os.path.join(
          self.data_dir, 'train/*' if self.is_training else 'validation/*')
          # self.data_dir, 'Herbarium2012-train*' if self.is_training else 'Herbarium2012-validation*')
    else:
      if self.dataset_split not in ['train', 'validation']:
        raise ValueError(
            "If provided, dataset_split must be 'train' or 'validation', was %s"
            % self.dataset_split)
      file_pattern = os.path.join(self.data_dir, self.dataset_split + '-*')

    print(file_pattern)
    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = tf.data.Dataset.list_files(
        file_pattern, shuffle=self.shuffle_shards)
    dataset = dataset.shard(num_hosts, index)

    if self.is_training and not self.cache:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            fetch_dataset, cycle_length=64, sloppy=True))

    if self.cache:
      dataset = dataset.cache().apply(
          tf.data.experimental.shuffle_and_repeat(1024 * 16))
    else:
      dataset = dataset.shuffle(1024)
    return dataset


# Defines a selection of data from a Cloud Bigtable.
BigtableSelection = collections.namedtuple('BigtableSelection', [
    'project',
    'instance',
    'table',
    'prefix',
    'column_family',
    'column_qualifier',
])


class HerbariumBigtableInput(HerbariumTFExampleInput):
  """Generates Herbarium input_fn from a Bigtable for training or evaluation."""

  def __init__(self, is_training, use_bfloat16, transpose_input, selection,
               augment_name, randaug_num_layers, randaug_magnitude):
    """Constructs an Herbarium input from a BigtableSelection.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      selection: a BigtableSelection specifying a part of a Bigtable.
      augment_name: `string` that is the name of the augmentation method
        to apply to the image. `autoaugment` if AutoAugment is to be used or
        `randaugment` if RandAugment is to be used. If the value is `None` no
        no augmentation method will be applied applied. See autoaugment.py
        for more details.
      randaug_num_layers: 'int', if RandAug is used, what should the number of
        layers be. See autoaugment.py for detailed description.
      randaug_magnitude: 'int', if RandAug is used, what should the magnitude
        be. See autoaugment.py for detailed description.
    """
    super(HerbariumBigtableInput, self).__init__(
        is_training=is_training,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input,
        augment_name=augment_name,
        randaug_num_layers=randaug_num_layers,
        randaug_magnitude=randaug_magnitude)
    self.selection = selection

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    try:
      from tensorflow.contrib.cloud import BigtableClient  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      logging.exception('Bigtable is not supported in TensorFlow 2.x.')
      raise e

    data = self.selection
    client = BigtableClient(data.project, data.instance)
    table = client.table(data.table)
    ds = table.parallel_scan_prefix(
        data.prefix, columns=[(data.column_family, data.column_qualifier)])
    # The Bigtable datasets will have the shape (row_key, data)
    ds_data = ds.map(lambda index, data: data)

    if self.is_training:
      ds_data = ds_data.repeat()

    return ds_data

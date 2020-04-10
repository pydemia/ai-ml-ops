from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import models
from . import model_resnet


from tensorflow.compat.v1 import logging as tf_logging
tf_logging.set_verbosity(tf_logging.INFO)


def input_fn(features, labels, batch_size, mode):
  """Input function.

  Args:
    features: (numpy.array) Training or eval data.
    labels: (numpy.array) Labels for training or eval data.
    batch_size: (int)
    mode: tf.estimator.ModeKeys mode

  Returns:
    A tf.estimator.
  """

  features = np.expand_dims(features, axis=-1)
  # Default settings for training.
  if labels is None:
    inputs = features
  else:
    # Change numpy array shape.
    inputs = (features, labels)
  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices(inputs)
  if mode == tf.estimator.ModeKeys.TRAIN:
    # dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    dataset = dataset.shuffle(1000).repeat()
  if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
    dataset = dataset

  # dataset = dataset.batch(batch_size)
  # def pad_fn(image):
  #     return tf.image.resize_with_crop_or_pad(image, 32, 32)
  #     tf_dataset.pad
  # dataset = dataset.map(
  #     lambda feat, lab: pad_fn(tf), lab
  # )
  dataset = dataset.padded_batch(
    batch_size,
    padded_shapes=([32, 32, 3], [1]),
    padding_values=(
      tf.constant(0., shape=(), dtype=tf.float64),
      tf.constant(0, shape=(), dtype=tf.int64),
    ),
  )
  # return dataset.make_one_shot_iterator().get_next()
  return dataset


def serving_input_fn():
  """Defines the features to be passed to the model during inference.

  Expects already tokenized and padded representation of sentences

  Returns:
    A tf.estimator.export.ServingInputReceiver
  """
  feature_placeholder = tf.placeholder(tf.float32, [None, 32 * 32 * 3])
  features = feature_placeholder
  return tf.estimator.export.TensorServingInputReceiver(features,
                                                        feature_placeholder)


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


def keras_model(model_dir, learning_rate):
  """Creates a Keras Sequential model with layers.

  Args:
    model_dir: (str) file path where training files will be written.
    config: (tf.estimator.RunConfig) Configuration options to save model.
    learning_rate: (int) Learning rate.

  Returns:
    A keras.Model
  """
  # model = models.Sequential()
  # model.add(Flatten(input_shape=(28, 28)))
  # model.add(Dense(128, activation=tf.nn.relu))
  # model.add(Dense(10, activation=tf.nn.softmax))


  model = model_resnet.ResNet50(
    include_top=False,  # True,
    weights=None,  # 'imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,  # {None, 'avg', 'max'}
    classes=10,  # 1000,
    #**kwargs,
  )

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True,
  )

  # Compile model with learning parameters.
  # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  model.compile(
      optimizer=optimizer,
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )

  # estimator = tf.keras.estimator.model_to_estimator(
  #     keras_model=model, model_dir=model_dir, config=config)
  # return estimator

  return model


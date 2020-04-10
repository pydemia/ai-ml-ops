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

import argparse

import os
import numpy as np
from . import data_loader
from . import model
from . import utils

import tensorflow as tf
# from tensorflow.contrib.training.python.training import hparam

from tensorflow.compat.v1 import logging as tf_logging
from tensorboard.plugins.hparams import api as hp


def get_args():
  """Argument parser.

	Returns:
	  Dictionary of arguments.
	"""
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--job-dir',
    type=str,
    required=True,
    help='GCS location to write checkpoints and export models')
  parser.add_argument(
    '--train-file',
    type=str,
    required=True,
    help='Training file local or GCS')
  parser.add_argument(
    '--train-labels-file',
    type=str,
    required=True,
    help='Training labels file local or GCS')
  parser.add_argument(
    '--test-file',
    type=str,
    required=True,
    help='Test file local or GCS')
  parser.add_argument(
    '--test-labels-file',
    type=str,
    required=True,
    help='Test file local or GCS')
  parser.add_argument(
    '--num-epochs',
    type=float,
    default=5,
    help='number of times to go through the data, default=5')
  parser.add_argument(
    '--batch-size',
    default=128,
    type=int,
    help='number of records to read during each training step, default=128')
  parser.add_argument(
    '--learning-rate',
    default=.01,
    type=float,
    help='learning rate for gradient descent, default=.001')
  parser.add_argument(
    '--verbosity',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    default='INFO')
  return parser.parse_args()


def train_and_evaluate(hparams):
  """Helper function: Trains and evaluates model.

  Args:
    hparams: (dict) Command line parameters passed from task.py
  """
  # Loads data.
  (train_images, train_labels), (test_images, test_labels) = \
      utils.prepare_data(train_file=hparams.train_file,
                         train_labels_file=hparams.train_labels_file,
                         test_file=hparams.test_file,
                         test_labels_file=hparams.test_labels_file)

  # Scale values to a range of 0 to 1.
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  # Define training steps.
  # train_steps = hparams.num_epochs * len(train_images) / hparams.batch_size

  ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=hparams.job_dir,  # weights.{epoch:02d}-{val_loss:.2f}.hdf5
    monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto',
    save_freq='epoch',  # 'epoch'
    # When using integer, the callback saves the model at end of a batch
    # at which this many samples have been seen since last saving.
  )
  tb_callback = tf.keras.callbacks.TensorBoard(
      log_dir=os.path.join(hparams.job_dir, 'logs'),
      histogram_freq=0, write_graph=True, write_images=False,
      update_freq='epoch',  # {'batch', 'epoch'}, int=batch,
      profile_batch=0,
      embeddings_freq=0,
      embeddings_metadata=None,
  )
  callbacks = [
      ckpt_callback,
      tb_callback,
  ]

  train_labels = np.asarray(train_labels).astype('int').reshape((-1, 1))
  test_labels = np.asarray(test_labels).astype('int').reshape((-1, 1))
  train_dataset = data_loader.input_fn(
    train_images,
    train_labels,
    hparams.batch_size,
    is_training=True,
  )
  eval_dataset = data_loader.input_fn(
    test_images,
    test_labels,
    hparams.batch_size,
    is_training=False,
  )

  keras_model = model.keras_model(
    model_dir=hparams.job_dir,
    learning_rate=hparams.learning_rate
  )
  keras_model.fit(
    train_dataset, callbacks=callbacks,
    validation_data=eval_dataset, validation_steps=None, validation_freq=1,
    epochs=int(hparams.num_epochs), initial_epoch=int(0), steps_per_epoch=None,
    shuffle=True,
    verbose=2,
    class_weight=None, sample_weight=None,
    max_queue_size=10, workers=1,
    use_multiprocessing=False,
  )

  
  keras_model.save(
    os.path.join(hparams.job_dir, "savedmodel"),
    save_format="tf",  # {"tf", "h5"}
  )
  # config = model.get_config()
  # reinitialized_model = keras.Model.from_config(config)


if __name__ == '__main__':

  # args = get_args()
  # tf.logging.set_verbosity(args.verbosity)
  # hparams = hparam.HParams(**args.__dict__)
  # train_and_evaluate(hparams)

  args = get_args()
  tf_logging.set_verbosity(args.verbosity)
  # hparams = hparam.HParams(**args.__dict__)
  train_and_evaluate(args)


# # %%
# from tensorboard.plugins.hparams import api as hp
# adict = {'num_epochs': 5, 'batch_size': 128,
#          'learning_rate': 0.01, 'verbosity': 'INFO'}
# hp.HParam(**adict)

# hp.hparams(adict)

# ?hp.hparams

#!/usr/bin/python3


# %% Set up


from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # %tensorflow_version only exists in Colab.
  # !pip install - q tf-nightly
  %pip install tensorflow-datasets
except Exception:
  pass
import tensorflow as tf

import numpy as np
import IPython.display as display


# %% Fetch the images

# tf.keras.utils.get_file(
#   imagename,
# )

# %%

def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

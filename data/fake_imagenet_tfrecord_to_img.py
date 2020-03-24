
# %% Setup

from __future__ import absolute_import, division, print_function, unicode_literals

try:
    # %tensorflow_version only exists in Colab.
    %pip install -q tensorflow
except Exception:
    pass
import tensorflow as tf

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt


# %%
print(os.getcwd())
print(glob("data/fake-imagenet-tfrecord/train/*"))
# %% Read TFrecord

train_tfr_list = glob("data/fake-imagenet-tfrecord/train/*")
test_tfr_list = glob("data/fake-imagenet-tfrecord/test/*")
filenames = train_tfr_list
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset


def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

# %%
for raw_record in raw_dataset.take(10):
    print(repr(raw_record))

# %%
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

# %%

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    single_record = raw_record.numpy()
    example.ParseFromString(single_record)
    print(example)
# %%
np.array(single_record)
example.features.feature
example.features.feature["image/filename"]

type(example.features.feature["image"])
plt.imshow(example.features.feature["image"])
# %%
aa = tf.image.encode_jpeg(
    single_record,
    optimize_size=False,
)

# %%

parsed_image_dataset = raw_dataset.map(_parse_image_function)

# %%

#record_iterator = tf.data.TFRecordDataset(train_tfr_list)
# for element in raw_dataset.as_numpy_iterator().next():
#     print(element)
element = raw_dataset.as_numpy_iterator().next()

len(element)
element[:2]

# %%

raw_record.__dict__
raw_dataset.__dict__
example.features.feature["image/height"]
aa = dict(example.features.feature)

aa.keys()
aa.items()

type(aa["image/height"])

aa["image/height"].bytes_list
aa["image/height"].float_list
aa["image/height"].int64_list


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

print(glob("/home/pydemia/hdc1/datasets/images/mscoco-tfrecord/train/*"))
#print(glob("data/fake-imagenet-tfrecord/train/*"))
# %% Read TFrecord

#train_tfr_list = glob("data/fake-imagenet-tfrecord/train/*")
train_tfr_list = glob(
    "/home/pydemia/hdc1/datasets/images/mscoco-tfrecord/train/*")
#test_tfr_list = glob("data/fake-imagenet-tfrecord/test/*")
filenames = train_tfr_list
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset


for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    single_record = raw_record.numpy()
    example.ParseFromString(single_record)
    feature = example.features.feature

list(feature)
# %%
def _get_feature_structure(feature):
    feature_attr = [
        attr for attr in feature.__dir__() if attr.endswith("_list")
    ]
    feature_type = [
        attr.split("_list")[0]  # feature.__getattribute__(attr)
        for attr in feature_attr
        if len(feature.__getattribute__(attr).__str__()) > 0
    ]

    return feature_type[0]


def _create_feature_description(raw_dataset):

    feature_base_dict = {
        'int64': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'bytes': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'float': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }

    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        single_record = raw_record.numpy()
        example.ParseFromString(single_record)
        feature = example.features.feature

        feature_description = {}
        for f_key in list(feature):
            f_typ = _get_feature_structure(feature[f_key])
            feature_description[f_key] = feature_base_dict[f_typ]

    return feature_description


#bb = _get_feature_structure(aa)
feature_desc = _create_feature_description(raw_dataset)
print(feature_desc)

def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_desc)


# %%
parsed_dataset = raw_dataset.map(_parse_function)

for parsed_record in parsed_dataset.take(10):
    print(repr(parsed_record))



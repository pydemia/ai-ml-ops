# Walkthrough: Reading and writing image data
# This is an end-to-end example of how to read and write image data using TFRecords.
# Using an image as input data, you will write the data as a TFRecord file,
# then read the file back and display the image.

# This can be useful if, for example,
# you want to use several models on the same input dataset.
# Instead of storing the image data raw,
# it can be preprocessed into the TFRecords format,
# and that can be used in all further processing and modelling.

# First, let's download this image of a cat in the snow and
# this photo of the Williamsburg Bridge, NYC under construction.


# %% Set up

from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # %tensorflow_version only exists in Colab.
  %pip install - q tf-nightly
except Exception:
  pass
import tensorflow as tf

import numpy as np
import IPython.display as display


# %% Fetch the images
cat_in_snow = tf.keras.utils.get_file(
    '320px-Felis_catus-cat_on_snow.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg',
)
williamsburg_bridge = tf.keras.utils.get_file(
    '194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
)


# %%
display.display(display.Image(filename=cat_in_snow))
display.display(display.HTML(
    'Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'
    )
)

display.display(display.Image(filename=williamsburg_bridge))
display.display(display.HTML(
    '<a "href=https://commons.wikimedia.org/wiki/File:New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg">From Wikimedia</a>'
    )
)

# %% Write the TFRecord file

image_labels = {
    cat_in_snow: 0,
    williamsburg_bridge: 1,
}

# This is an example, just using the cat image.
image_string = open(cat_in_snow, 'rb').read()

label = image_labels[cat_in_snow]

# Create a dictionary with features that may be relevant.


def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))


for line in str(image_example(image_string, label)).split('\n')[:15]:
  print(line)
print('...')


# %%

# Write the raw image files to `images.tfrecords`.
# First, process the two images into `tf.Example` messages.
# Then, write to a `.tfrecords` file.
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for filename, label in image_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        writer.write(tf_example.SerializeToString())


# %% Read the TFRecord file

raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset


# %% Recover the images from the TFRecord file

for image_features in parsed_image_dataset:
    image_raw = image_features['image_raw'].numpy()
    display.display(display.Image(data=image_raw))




# %%
williamsburg_bridge

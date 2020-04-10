import numpy as np
import tensorflow as tf


features = np.ones([100, 28, 28])
labels = np.expand_dims(np.array(list(map(str, np.arange(100) + 1))), axis=-1)

features.shape
labels.shape

features = np.expand_dims(features, axis=-1)
features.shape
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
dataset = dataset.padded_batch(
    64,
    padded_shapes=([32, 32, 3], []),
    # padding_values=(0, ),
)

dataset = tf.data.Dataset.from_generator(
    lambda: iter(elements), (tf.int32, tf.int32))
# Pad the first component of the tuple to length 4, and the second
# component to the smallest size that fits.
dataset = dataset.padded_batch(2,
                               padded_shapes=([4], [None]),
                               padding_values=(-1, 100))
list(dataset.as_numpy_iterator())

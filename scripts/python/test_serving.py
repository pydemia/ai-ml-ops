"""A `Tensorflow Serving` snippet for test.
"""

import os
from glob import glob
import numpy as np
import tensorflow as tf

os.chdir(
    '/home/pydemia/git/network_anomaly_detection_tensorflow'
)

print(tf.__version__)


# %% Create a dump model -----------------------------------------------------

def input_fn(
    input_x,
    input_y,
    input_x_dtype=tf.float32,
    input_y_dtype=tf.float32,
    batch_size=64,
    is_training=True,
    drop_remainder=False,
    name='input_fn',
    ):

    with tf.name_scope(name):

        buffer_size = 1000 if is_training else 1

        X_t = tf.placeholder(input_x_dtype, input_x.shape,
                             name='x_tensor_interface')
        Y_t = tf.placeholder(input_y_dtype, input_y.shape,
                             name='y_tensor_interface')

        dataset = tf.data.Dataset.from_tensor_slices((X_t, Y_t))
        dataset = dataset.shuffle(buffer_size=buffer_size)  # reshuffle_each_iteration=True as default.
        dataset = dataset.batch(
            batch_size,
            drop_remainder=drop_remainder,
        )
        dataset = dataset.flat_map(
            lambda data_x, data_y: tf.data.Dataset.zip(
                (
                    tf.data.Dataset.from_tensors(data_x),
                    tf.data.Dataset.from_tensors(data_y),
                    tf.data.Dataset.from_tensors(
                        tf.map_fn(
                            tf.size,
                            data_x,
                            dtype=tf.int32,
                        ),
                    ),
                    tf.data.Dataset.from_tensors(
                        tf.map_fn(
                            tf.size,
                            data_y,
                            dtype=tf.int32,
                        ),
                    ),
                )
            )#.repeat(repeat_num)
        )

        # data_op = dataset.make_initializable_iterator()
        # data_init_op = data_op.initializer
        # next_batch = X_batch, Y_batch, _, _ = data_op.get_next()
        #
        # print('[dtype] X: %s , Y: %s' % (X_batch.dtype, Y_batch.dtype))
        # print('[shape] X: %s , Y: %s' % (X_batch.get_shape(), Y_batch.get_shape()))

    return dataset


def model_fn(
    input_x,
    input_y,
    name='model_fn',
    ):

    Input_x_tensor = tf.placeholder(
        tf.float32,
        shape=(None, 2),
        name='input_x_tensor',
    )
    Input_y_tensor = tf.placeholder(
        tf.float32,
        shape=(None, 2),
        name='input_y_tensor',
    )

    with tf.variable_scope('test_model', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(
            'weights',
            shape=(2, 3),
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(),
        )
        biases = tf.get_variable(
            'biases',
            shape=(3, ),
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(),
        )
        matmul = tf.matmul(
            Input_x_tensor,
            weights,
            name='matmul',
        )
        biased = tf.nn.bias_add(
            matmul,
            biases,
            name='lineared',
        )

    return biased

"""TF-Record Dataset.
"""

import os
import tensorflow as tf
import numpy as np
import itertools as it


# tf.enable_eager_execution()

# os.system(
#     'gcsfuse --implicit-dirs yjkim-datasets ~/mnt/yjkim-datasets'
# )
os.chdir(
    '/home/pydemia/git/tcl_2018_text_summ'
)
print(os.getcwd())


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value)
    )
    return feature


def _float_feature(value):
    """Returns a float_list from a float / double."""
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=value)
    )
    return feature


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=value)
    )
    return feature



tf.serialize_tensor
tf.parse_tensor


# %% TF ----

# import pandas as pd
# rr = pd.read_table('data/news/newsdata_bpe_200000.id', header=None)[0]
#
# rrs = rr.str.split()
# rrs[0]

with open('data/news/newsdata_bpe_100000_soft_minfreq_42.id', 'r') as f:
    bb = [line.split() for line in f.readlines()]

bbb = bb[:20]
len(bb)
# %%


bbb[0]

len(bbb[0])

# %%

def skip_gram_pairs(
        token_ids,
        window_size=4,
        num_grams=6,
        # negative_samples=1.,
        shuffle=False,
        ):

    res_list = []
    for i, xt in enumerate(token_ids):

        # if i < window_size:
        #     xt_f = token_ids[max(0, i-window_size):i]
        # else:
        #     xt_f = token_ids[i-window_size:i]
        xt_f = token_ids[max(0, i-window_size):i]

        xt_b = token_ids[i+1:i+1+window_size]

        x_source = [xt]
        x_target = xt_f + xt_b

        try:
            x_skipped = np.random.choice(x_target, num_grams)
        except ValueError as err:
            # print(
            #     'x_source: %s' % x_source,
            #     'x_target: %s' % x_target,
            #     'xt: %s' % xt,
            #     'xt_f: %s' % xt_f,
            #     'xt_b: %s' % xt_b,
            #     sep='\n',
            # )
            # break
            """
            Case: `xt` Only.
            x_source: ['7']
            x_target: []
            xt: 7
            xt_f: []
            xt_b: []
            x_source: ['70924']
            x_target: []
            xt: 70924
            xt_f: []
            xt_b: []
            x_source: ['2659']
            x_target: []
            xt: 2659
            xt_f: []
            xt_b: []
            """
            continue

        res_list += list(it.product(x_source, x_skipped))

    if shuffle:
        np.random.shuffle(res_list)

    return res_list


def feature_map(arr):

    feature = {
        # 'target': _int64_feature(paired_arr[:, 0]),
        # 'context': _int64_feature(paired_arr[:, 1]),
        # 'target': _int64_feature(arr_shape[0]),
        # 'context': _int64_feature(arr_shape[1]),
        'target': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[arr[0]])
        ),
        'context': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[arr[1]])
        ),
    }
    res = tf.train.Example(
        features=tf.train.Features(feature=feature),
    )
    return res


def feature_map_arr(arr):

    feature = {
        # 'target': _int64_feature(paired_arr[:, 0]),
        # 'context': _int64_feature(paired_arr[:, 1]),
        # 'target': _int64_feature(arr_shape[0]),
        # 'context': _int64_feature(arr_shape[1]),
        'target': tf.train.Feature(
            int64_list=tf.train.Int64List(value=arr[:, 0])
        ),
        'context': tf.train.Feature(
            int64_list=tf.train.Int64List(value=arr[:, 1])
        ),
    }
    res = tf.train.Example(
        features=tf.train.Features(feature=feature),
    )
    return res


def sg_pair_example(
        line_token_ids,
        window_size=5,
        num_grams=6,
        ):

    paired_arr = np.array(
        skip_gram_pairs(
            line_token_ids,
            window_size=window_size,
            num_grams=num_grams,
            # negative_samples=1.,
            shuffle=False,
        ),
        dtype=np.int64,
    )
    arr_shape = paired_arr.shape
    # print(paired_arr[:, 0])

    return [feature_map(arr) for arr in paired_arr]


def sg_pair_array(
        line_token_ids,
        window_size=5,
        num_grams=6,
        ):

    paired_arr = np.array(
        skip_gram_pairs(
            line_token_ids,
            window_size=window_size,
            num_grams=num_grams,
            # negative_samples=1.,
            shuffle=False,
        ),
        # dtype=np.int64,
        dtype=np.str,
    )
    # arr_shape = paired_arr.shape
    # print(paired_arr[:, 0])

    return paired_arr

# bbb[0]
# cc = skip_gram_pairs(bbb[0])
# cc[0]
# %%

with open('data/news/newsdata_bpe_100000_soft_minfreq_42.id', 'r') as f:
    token_ids = [line.split() for line in f.readlines()]
    aa = sg_pair_example(token_ids[0])
# dd = sum([skip_gram_pairs(item) for item in bb], [])
aa

# %% ----
tfrecord_filename = 'data/news_diskio/newsdata_bpe_100000_soft_minfreq_42_4window_6pick.id.tfrecords'
with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
    with open('data/news_diskio/newsdata_bpe_100000_soft_minfreq_42.id', 'r') as f:
        token_ids = [line.split() for line in f.readlines()]
        # for line_token_ids in token_ids:
        #     tf_example_list = sg_pair_example(line_token_ids)
        #     for example in tf_example_list:
        #         writer.write(example.SerializeToString())
        print('line length: ', len(token_ids))
        example_arr = np.concatenate([
            sg_pair_array(
                line_token_ids,
                window_size=4,
                num_grams=6,
            )
            for line_token_ids in token_ids
        ])
        print('array shape: ', example_arr.shape)
        writer.write(feature_map_arr(example_arr).SerializeToString())
        # for example in tf_example_list_gen:
        #     writer.write(example.SerializeToString())

print('finished.')

# %%
# tfrecord_filename = 'data/news/newsdata_bpe_200000_id.tfrecords'
tfrecord_filename = 'data/news/newsdata_bpe_100000_soft_minfreq_42.id.tfrecords'
with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
    with open('data/news/newsdata_bpe_100000_soft_minfreq_42.id', 'r') as f:
        token_ids = [line.split() for line in f.readlines()]
        print('line length: ', len(token_ids))
        for line_token_ids in token_ids:
            tf_example_list = sg_pair_example(line_token_ids)
            for example in tf_example_list:
                writer.write(example.SerializeToString())
print('finished.')

# %%

import multiprocessing as mpr
from pathos.parallel import stats
from pathos.parallel import ParallelPool as Pool


def mp_worker(line_token_ids):
    tf_example_list = sg_pair_example(
        line_token_ids
    )
    return tf_example_list


def mp_handler(
        n,
        raw_filename,
        tfrecord_filename,
        ):

    p = mpr.Pool(n)

    with open(raw_filename, 'r') as f:
        token_ids = [line.split() for line in f.readlines()]
        print('line length: ', len(token_ids))

    with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:

        for tf_example_list in p.imap(mp_worker, token_ids):
            for example in tf_example_list:
                writer.write(example.SerializeToString())

    print('TFRecording finished.')


mp_handler(
    n=4,
    raw_filename='data/news_diskio/newsdata_bpe_100000_soft_minfreq_42.id',
    tfrecord_filename='data/news_diskio/newsdata_bpe_100000_soft_minfreq_42_4window_6pick.id.tfrecords',
)

# %%

from multiprocessing import Process
coord = tf.train.Coordinator()
processes = []
for thread_index in range(8):
    args = (str(thread_index), range(100000))
    p = Process(target=write_tfrecord, args=args)
    p.start()
    processes.append(p)
coord.join(processes)

# %%

raw_dataset = tf.data.TFRecordDataset(
    ['data/news/newsdata_bpe_200000_id.tfrecords']
)

feature_desc = {
    # 'target': tf.VarLenFeature(tf.int64),
    # 'context': tf.VarLenFeature(tf.int64),
    'target': tf.FixedLenFeature([], tf.int64, default_value=[0]),
    'context': tf.FixedLenFeature([], tf.int64, default_value=[0]),
    # 'context': tf.FixedLenFeature([], tf.int64),

}



def _parse_example(example_proto):
    res = tf.parse_single_example(example_proto, feature_desc)
    return res

parsed_dataset = raw_dataset.map(_parse_example)# .shuffle(buffer_size=1000)
parsed_dataset

data_op = parsed_dataset.make_initializable_iterator()
data_init_op = data_op.initializer
next_batch = data_op.get_next()

raw_dataset

#
# for i, xi in enumerate(parsed_dataset):
#     print(i, xi['context'].numpy(), xi['target'].numpy())
#     # print(i, xi)
#     if i > 10:
#         break

with tf.Session() as sess:
    sess.run(data_init_op)

    xi = sess.run(next_batch)

xi
# %%

import keras

cc = keras.preprocessing.sequence.skipgrams(
    bbb[:2],
    vocabulary_size=200_000,
    window_size=4,
    negative_samples=1.,
    shuffle=False,
    categorical=False,
    sampling_table=None,
    seed=None,
)

cc
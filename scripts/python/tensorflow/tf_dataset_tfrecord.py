

# %%

import tensorflow as tf
import tensorflow_datasets as tfds

import IPython.display as display

# Here we assume Eager mode is enabled (TF2), but tfds also works in Graph mode.

print(tf.__version__)
# %%

# See available datasets
print(tfds.list_builders())

# %%

# Construct a tf.data.Dataset
#ds_train = tfds.load(name="mnist", split="train", shuffle_files=True)

# Build your input pipeline
# %%
ds_train = tfds.load(name="coco/2017", split="train", shuffle_files=True)
# dataset = (
#     ds_train
#     .shuffle(1000)
#     .batch(128)
#     .prefetch(10)
# )
# dataset = dataset.make_one_shot_iterator()

dataset = ds_train.make_one_shot_iterator()
feature_sample = dataset.get_next(1)
print(feature_sample.keys())
feature_sample

feature_sample['objects']
feature_sample['image/id']

%pip install matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(features['image'])
# %%

feature_sample.keys()

#dataset = ds_train.make_one_shot_iterator().get_next(2)
dataset = ds_train.as_numpy_iterator() # This will Preserve the nested structure of dataset elements.

aa = dataset.next()
aa

# %%

feature_structure = {}
for key, value in aa.items():
    feature_structure.setdefault(key, value)

feature_structure


# ------- None-shaped Features -------
#
# tfds.features.FeaturesDict({
#     'input': tf.int32,
#     'target': {
#         'height': tf.int32,
#         'width': tf.int32,
#     },
# })
# Will internally store the data as:
#
# {
#     'input': tf.io.FixedLenFeature(shape=(), dtype=tf.int32),
#     'target/height': tf.io.FixedLenFeature(shape=(), dtype=tf.int32),
#     'target/width': tf.io.FixedLenFeature(shape=(), dtype=tf.int32),
# }

# From `https://www.tensorflow.org/datasets/catalog/coco`
feature_structure = tfds.features.FeaturesDict({
    'image': tfds.features.Image(
        shape=(None, None, 3),
        dtype=tf.uint8,
    ),
    'image/filename': tfds.features.Text(
        #shape=(),
        #dtype=tf.string,
    ),
    'image/id': tf.int64,
    'objects': tfds.features.Sequence({
        'area': tf.int64,
        'bbox': tfds.features.BBoxFeature(
            #shape=(4,),
            #dtype=tf.float32,
        ),
        'id': tf.int64,
        'is_crowd': tf.bool,
        'label': tfds.features.ClassLabel(
            #shape=(),
            #dtype=tf.int64,
            num_classes=80,
        ),
    }),
})

# %%
feature_structure['image']
# %%

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def coco_image_example(feature):
    # image_shape = tf.image.decode_jpeg(image_string).shape
    #
    # feature = {
    #   'height': _int64_feature(image_shape[0]),
    #   'width': _int64_feature(image_shape[1]),
    #   'depth': _int64_feature(image_shape[2]),
    #   'label': _int64_feature(label),
    #   'image_raw': _bytes_feature(image_string),
    # }
    feature = tf.FeaturesDict(
        {
            'image': tfds.features.Image(
                shape=(None, None, 3),
                dtype=tf.uint8,
            ),
            'image/filename': tfds.features.Text(
                shape=(),
                dtype=tf.string,
            ),
            'image/id': tf.int64,
            'objects': tfds.features.Sequence(
                {
                    'area': tf.int64,
                    'bbox': tfds.features.BBoxFeature(
                        shape=(4,),
                        dtype=tf.float32,
                    ),
                    'id': tf.int64,
                    'is_crowd': tf.bool,
                    'label': tfds.features.ClassLabel(
                        shape=(),
                        dtype=tf.int64,
                        num_classes=80,
                    ),
                },
            ),
        }
    )
    #return tf.train.Example(features=tf.train.Features(feature=feature))

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto.SerializeToString()
#
# def coco_image_example(feature):
#     image_shape = tf.image.decode_jpeg(image_string).shape
#
#     feature = {
#       'image': _int64_feature(image_shape[0]),
#       'width': _int64_feature(image_shape[1]),
#       'depth': _int64_feature(image_shape[2]),
#       'label': _int64_feature(label),
#       'image_raw': _bytes_feature(image_string),
#     }
#     example_proto = tf.train.Example(
#         features=tf.train.Features(feature=feature)
#     )
#     return example_proto.SerializeToString()

# def ds2tfrecord(ds, filepath):
#     with tf.python_io.TFRecordWriter(filepath) as writer:
#         feat_dict = ds.make_one_shot_iterator().get_next()
#         serialized_dict = {name: tf.serialize_tensor(fea) for name, fea in feat_dict.items()}
#         with tf.Session() as sess:
#             try:
#                 while True:
#                     features = {}
#                     for name, serialized_tensor in serialized_dict.items():
#                         bytes_string = sess.run(serialized_tensor)
#                         bytes_list = tf.train.BytesList(value=[bytes_string])
#                         features[name] = tf.train.Feature(bytes_list=bytes_list)
#                     # Create a Features message using tf.train.Example.
#                     example_proto = tf.train.Example(features=tf.train.Features(feature=features))
#                     example_string = example_proto.SerializeToString()
#                     # Write to TFRecord
#                     writer.write(example_string)
#             except tf.errors.OutOfRangeError:
#                 pass

dataset = (
    ds_train


# %%
def ds2tfrecord(ds, filepath):
    with tf.io.TFRecordWriter(filepath) as writer:
        feat_dict = ds.make_one_shot_iterator().get_next()
        serialized_dict = {name: tf.io.serialize_tensor(fea) for name, fea in feat_dict.items()}
        try:
            while True:
                features = {}
                for key, value in serialized_dict.items():
                    if isinstance(value, tf.Tensor):
                        bytes_string = value  # serialized_tensor
                        bytes_list = tf.train.BytesList(value=[bytes_string])
                        features[key] = tf.train.Feature(bytes_list=bytes_list)
                        # Create a Features message using tf.train.Example.
                        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                        example_string = example_proto.SerializeToString()
                    elif isinstance(value, dict):
                        features[key] = value
                            for value_innerkey, value_innervalue in value.items():


                # Write to TFRecord
                writer.write(example_string)
        except tf.errors.OutOfRangeError:
            pass

def serialize_tensor_to_dict(feature_dict):
    serialized_dict = {
        key: tf.io.serialize_tensor(value)
        for key, value in feature_dict.items()
    }
    return serialized_dict

dd_dataset = dataset.make_one_shot_iterator()

*foo, bar = dd_dataset.get_next()

for foo, bar in dd_dataset.get_next().items():
    print(foo)


bar

dataset = (
    ds_train
    .flat_map(
        lambda *image_data, objects: tf.data.Dataset.zip(
            # tf.data.Datset.from_tensors(
            #     tf.map_fn(tf.io.serialize_tensor, image_data),
            # ),
            # tf.data.Datset.from_tensors(
            #     tf.map_fn(tf.io.serialize_tensor, objects),
            # )
            tf.map_fn(tf.io.serialize_tensor, image_data),
            tf.map_fn(tf.io.serialize_tensor, objects),
        )
    )
)

ds2tfrecord(ds_train, record_file)

# %%
record_file = 'train2017.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    # for filename, label in image_labels.items():
    #     #image_string = open(filename, 'rb').read()
    #     tf_example = image_example(image_string, label)
    #     writer.write(tf_example.SerializeToString())
    for feature in dataset:
        writer.write(feature)

# %%
raw_image_dataset = tf.data.TFRecordDataset('train2017.tfrecords')

# %%
ds_train = tfds.load(name="coco/2017", split="train", shuffle_files=True)
# dataset = (
#     ds_train
#     .shuffle(1000)
#     .batch(128)
#     .prefetch(10)
# )
# dataset = dataset.make_one_shot_iterator()
dataset = ds_train.as_numpy_iterator()
# dataset = (
#     ds_train
#     .range(3)
#     .map(
#         tf.io.serialize_tensor
#     )
# )


record_file = 'train2017.tfrecords'
tfrecord_writer = tf.io.TFRecordWriter(
    path=record_file,
    options=None,
)

for data_feature in dataset:
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=data_feature)
    )
    tfrecord_writer.write(example_proto.SerializeToString())
#tfrecord_writer.write(dataset)

# %%

dataset = tf.data.TFRecordDataset(record_file)
#dataset = dataset.map(lambda x: tf.io.parse_tensor(x, tf.int64))
aa = dataset.as_numpy_iterator().next()
aa
# %%


aa = raw_image_dataset.as_numpy_iterator()
for tmp in aa:
    print(tmp)
#    plt.imshow(_['image'])

print(tmp)

type(aa)
aa.get_next(1)
type(tmp)



# %%
import tensorflow_datasets as tfds
import tensorflow as tf

# Here we assume Eager mode is enabled (TF2), but tfds also works in Graph mode.

print(tf.__version__)
# %%

# See available datasets
print(tfds.list_builders())

# %%

# Construct a tf.data.Dataset
#ds_train = tfds.load(name="mnist", split="train", shuffle_files=True)


# %%

# Build your input pipeline
#ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
feature = ds_train.take(1)

# %%
tf.data.Dataset.unbatch()
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
features = dataset.get_next(1)
features
# %%
for features in ds_train.take(1):
    items = list(features.as_numpy_iterator())
    image, label = items["image"], items["label"]

# %%

aa = list(feature.take(1).as_numpy_iterator())[0]

# %%
# %%

aa[0]

# %%

def _generate_examples(self, images_dir_path, labels):
  # Read the input data out of the source files
  for image_file in tf.io.gfile.listdir(images_dir_path):
    ...
  with tf.io.gfile.GFile(labels) as f:
    ...

  # And yield examples as feature dictionaries
  for image_id, description, label in data:
    yield image_id, {
        "image_description": description,
        "image": "%s/%s.jpeg" % (images_dir_path, image_id),
        "label": label,
    }

# %%

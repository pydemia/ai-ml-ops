

# %%
import tensorflow_datasets as tfds
import tensorflow as tf

# Here we assume Eager mode is enabled (TF2), but tfds also works in Graph mode.

# See available datasets
print(tfds.list_builders())

# %%

# Construct a tf.data.Dataset
#ds_train = tfds.load(name="mnist", split="train", shuffle_files=True)
ds_train = tfds.load(name="coco/2017", split="train", shuffle_files=True)

# %%

# Build your input pipeline
ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
for features in ds_train.take(1):
  image, label = features["image"], features["label"]

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

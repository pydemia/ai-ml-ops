# %%
import tensorflow_datasets.public_api as tfds
%pip install tensorflow_datasets
import tensorflow_datasets.api as tfds_api
import tensorflow_dataset as tfds_api

# %%

%python -m tensorflow_datasets.scripts.create_new_dataset.py \
    - -dataset my_dataset \
    - -type image  # text, audio, translation,...

# %%


class MyDataset(tfds.core.GeneratorBasedBuilder):
  """Short description of my dataset."""

  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # Specifies the tfds.core.DatasetInfo object
    pass  # TODO

  def _split_generators(self, dl_manager):
    # Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    pass  # TODO

  def _generate_examples(self):
    # Yields examples from the dataset
    yield 'key', {}

# %%
tfds_api.download.add_checksums_dir('~/url_checksums')

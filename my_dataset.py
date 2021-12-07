"""my_dataset dataset."""

import tensorflow_datasets as tfds
import patoolib
import os
import gdown
from sklearn.model_selection import train_test_split
import random

_DESCRIPTION = """
This is my demo dataset (description)
"""

_CITATION = """
@article{my-demo-dataset-2021,
               author = {k3lu},}
"""

_ID_DRIVE = "1XQzuNaFZQvMw0kX7egVpGVyXBShMDY1L"
_PATH = './dataset/'
_TRUE_PATH  = _PATH + 'True'
_FALSE_PATH = _PATH + 'False'  

class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(100, 100, 3)),
            'label': tfds.features.ClassLabel(names=['no', 'yes']),
        }),
        supervised_keys=('image', 'label'),
        citation=_CITATION,
    )
  def _get_drive_url(self, id):
        base_url = 'https://drive.google.com/uc?export=download&id=' + str(id)
        return base_url
  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # path = dl_manager.download_and_extract('https://todo-data-url')

    # download and extract dataset
    if (not os.path.exists("dataset.rar")):
      gdown.download(self._get_drive_url(_ID_DRIVE), "dataset.rar", quiet = True)
      os.mkdir("dataset")
      patoolib.extract_archive('dataset.rar', outdir ="dataset")

    # get filename and split train and test
    names = []
    labels = []
    y = []
    for dirpath, dirnames, filenames in os.walk(_TRUE_PATH):
      names.extend(filenames)
      labels.extend([1]*len(filenames))
      break
    for dirpath, dirnames, filenames in os.walk(_FALSE_PATH):
      names.extend(filenames)
      labels.extend([0]*len(filenames))
      break
    X_train, X_test, y_train, y_test = train_test_split(names, labels, test_size=0.2, random_state=42)
    return {
        'train': self._generate_examples(X_train, y_train),
        'test' : self._generate_examples(X_test, y_test)
    }

  def _generate_examples(self, X, y):
    """Yields examples."""
    for name, label in zip(X,y):
      image_id = random.getrandbits(256)
      yield image_id,{
          'image': os.path.join((_TRUE_PATH if label == 1 else _FALSE_PATH) ,name),
          'label': label
      }

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AI Platform package configuration."""
from setuptools import find_packages
from setuptools import setup
import os

#REQUIRED_PACKAGES = ['requests==2.19.1']

def _get_requirements():
  """Parses requirements.txt file."""
  install_requires_tmp = []
  dependency_links_tmp = []
  with open(
      os.path.join(os.path.dirname(__file__), './requirements.txt'), 'r') as f:
    for line in f:
      package_name = line.strip()
      if package_name.startswith('-e '):
        dependency_links_tmp.append(package_name[3:].strip())
      else:
        install_requires_tmp.append(package_name)
  return install_requires_tmp, dependency_links_tmp

REQUIRED_PACKAGES, dependency_links = _get_requirements()


setup(name='fmnist-resnet',
      version='1.2',
      description='FMNIST ResNet - TensorFlow2',
      author='pydemia',
      author_email='pydemia@gmail.com',
      url='',
      # license='Apache 2.0',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=find_packages(exclude=["tutorials*", "samples*", "test*"]),
      exclude_package_data={
            '': [
                  '*_test.py',
            ],
      },
)

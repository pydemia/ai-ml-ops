#!/bin/bash

CONDA_DIR="/home/pydemia/apps/anaconda3"

curl -fsSL https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh | bash -b -f -p $CONDA_DIR
curl -fsSLo Miniconda3-Linux-x86_64.sh \
https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh \
&& (echo "\n";echo yes; echo $CONDA_DIR; echo no) | bash -f Anaconda3-Linux-x86_64.sh \
&& rm Anaconda3-Linux-x86_64.sh

cat anaconda.bashrc >> ~/.bashrc
cat anaconda.zshrc >> ~/.zshrc


bash -ic "conda install conda-build -y \
&& conda update conda -y \
&& conda config --add channels conda-forge"
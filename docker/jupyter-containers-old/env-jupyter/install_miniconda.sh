#!/bin/bash

CONDA_DIR="/home/pydemia/apps/miniconda3"

#wget https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O Miniconda3-Linux-x86_64.sh
curl -fsSLo Miniconda3-Linux-x86_64.sh \
https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh \
&& (echo "\n";echo yes; echo $CONDA_DIR; echo no) | bash -f Miniconda3-Linux-x86_64.sh \
&& rm Miniconda3-Linux-x86_64.sh

cat miniconda.bashrc >> ~/.bashrc
cat miniconda.zshrc >> ~/.zshrc


bash -ic "conda install conda-build -y \
&& conda update conda -y \
&& conda config --add channels conda-forge"
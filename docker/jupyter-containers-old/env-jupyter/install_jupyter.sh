#!/bin/bash

bash -i -c "pip install ipython jupyter \
&& conda install \
nb_conda \
jupyter_contrib_nbextensions \
ipykernel \
ipywidgets \
ipyparallel -y \
&& conda install -c conda-forge nb_conda_kernels -y \
&& pip install jupyter_tensorboard"

bash -i -c "conda install -c conda-forge jupyter_contrib_nbextensions -y \
jupyter contrib nbextension install --user \
conda install -c conda-forge jupyter_nbextensions_configurator -y \
jupyter nbextensions_configurator enable --user \
jupyter nbextension enable codefolding/main \
jupyter nbextension enable varInspector/main"

# Jupyter Config
cp -r ./env-jupyter/.jupyter ./
mkdir -p ./.jupyter/logs

bash -ic "mkdir -p $(jupyter --data-dir)/nbextensions/jupyter_themes"
bash -ic "wget https://raw.githubusercontent.com/merqurio/jupyter_themes/master/theme_selector.js -O $(jupyter --data-dir)/nbextensions/jupyter_themes/theme_selector.js"
bash -ic "jupyter nbextension enable jupyter_themes/theme_selector"

# Install Default Python Kernel
bash -i \
./env-jupyter/install_kernel_python.sh \
--kernel_name=py36 \
--display_name="Py3.6" \
--python_version=3.6

# Install Binary: Julia, Scala
bash -i ./env-jupyter/install_other_langs.sh

# Install R Kernel
bash -i \
./env-jupyter/install_kernel_r.sh \
--display_name="R (conda)"

# Install Julia Kernel
bash -i \
./env-jupyter/install_kernel_julia.sh \

# Install Scala Kernel
bash -i \
./env-jupyter/install_kernel_scala.sh \
--kernel_name=scala212 \
--display_name="Scala 2.12.8" \
--scala_version=2.12.8 \
--almond_version=0.6.0 

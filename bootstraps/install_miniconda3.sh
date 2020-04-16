#!/bin/bash

# Set Locale
apt-get update -y
apt-get install -y locales
locale-gen --purge "en_US.UTF-8"

bash -c "echo 'LC_ALL=en_US.UTF-8' >> /etc/environment"
bash -c "echo 'en_US.UTF-8 UTF-8' >> /etc/locale.gen"
bash -c "echo 'LANG=en_US.UTF-8' > /etc/locale.conf"


# Install Conda
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -O install_miniconda3_py37.sh

bash -c "echo 'export CONDA_PATH=\"/opt/anaconda3\"' >> /etc/profile"
bash -c "echo 'export PATH=\"\$CONDA_PATH/bin:\$CONDA_PATH/sbin:\$CONDA_PATH/condabin:\$PATH\"' >> /etc/profile"

source /etc/profile
source /etc/bash.bashrc

apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

bash ./install_miniconda3_py37.sh \
  -b \
  -p $CONDA_PATH



groupadd conda
chgrp -R conda $CONDA_PATH
chmod 770 -R $CONDA_PATH
chgrp -R conda /usr/local/share/jupyter
chmod 770 -R /usr/local/share/jupyter
adduser $USER conda

conda init
source ~/.bashrc


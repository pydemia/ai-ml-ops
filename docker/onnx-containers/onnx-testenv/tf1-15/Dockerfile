FROM ubuntu:18.04
LABEL maintainer="Youngju Kim pydemia@gmail.com"
ARG DEBIAN_FRONTEND=noninteractive

# USER root

SHELL ["/bin/bash", "-c"]


RUN apt-get update && \
    apt-get install -y \
    build-essential curl \
    vim htop sudo \
    libcurl4-openssl-dev \
    libssl-dev wget \
    git \
    python3-pip && \
    pip3 install --upgrade pip


# Set Locale
RUN apt-get update -y && \
    apt-get install -y locales && \
    locale-gen --purge "en_US.UTF-8"


RUN echo 'LC_ALL=en_US.UTF-8' >> /etc/environment
RUN echo 'en_US.UTF-8 UTF-8' >> /etc/locale.gen
RUN echo 'LANG=en_US.UTF-8' > /etc/locale.conf


# Install Conda

RUN apt-get update --fix-missing && \
    apt-get install -y \
    wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion acl

RUN apt-get install -y \
    libgl1-mesa-glx libegl1-mesa \
    libxrandr2 libxrandr2 libxss1 \
    libxcursor1 libxcomposite1 libasound2 \
    libxi6 libxtst6


ARG CONDA_USER="pydemia"
ARG CONDA_ADMIN="conda"
ARG CONDA_GROUP="conda"
# ARG CONDA_UID="1000"
# ARG CONDA_GID="100"
ARG CONDA_USER_ROOT="${CONDA_USER}_root"

ENV CONDA_PATH "/opt/conda"
ENV PATH "$CONDA_PATH/bin:$CONDA_PATH/sbin:$CONDA_PATH/condabin:$PATH"

ENV JUPYTER_CONFIG_DIR "$CONDA_PATH/etc/jupyter"
ENV JUPYTER_DATA_DIR "$CONDA_PATH/share/jupyter"
# ENV JUPYTER_CONFIG_DIR "/usr/local/etc/jupyter"
# ENV JUPYTER_DATA_DIR "/usr/local/share/jupyter"


# -------------------------------------------------------------------- #
# `-r` option: For system group(gid<1000), which is prior to user group(gid>=1000)
RUN groupadd -r $CONDA_GROUP && \
    #useradd -m -G conda -p $(openssl passwd -1 password) $CONDA_USER && \
    useradd -m -g $CONDA_GROUP $CONDA_ADMIN && \
    useradd -m -g $CONDA_GROUP -p $(openssl passwd -1 $CONDA_USER) $CONDA_USER && \
    usermod -a -G sudo $CONDA_USER && \
    useradd -m -g $CONDA_GROUP --non-unique --uid 0 $CONDA_USER_ROOT && \
    usermod -a -G $CONDA_GROUP root


RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh \
    -O install_conda3_py37.sh

RUN sg - $CONDA_GROUP "mkdir -p $CONDA_PATH" && \
    chmod g+srwx $CONDA_PATH

# `-u` option: If you want to update an existing installation, use the -u option.
RUN sg - $CONDA_GROUP "bash ./install_conda3_py37.sh \
        -u \
        -b \
        -p $CONDA_PATH" && \
    rm ./install_conda3_py37.sh
RUN echo 'export CONDA_PATH=$CONDA_PATH' >> /etc/profile
RUN echo 'export PATH="$CONDA_PATH/bin:$CONDA_PATH/sbin:$CONDA_PATH/condabin:$PATH"' >> /etc/profile
RUN ln -s $CONDA_PATH/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# RUN ln -s $CONDA_PATH/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". $CONDA_PATH/etc/profile.d/conda.sh" >> ~/.bashrc && \
#     echo "conda activate base" >> ~/.bashrc

# USER $CONDA_USER
# RUN echo ". $CONDA_PATH/etc/profile.d/conda.sh" >> ~/.bashrc && \
#     echo "conda activate base" >> ~/.bashrc

USER root
RUN conda init bash && \
    conda init zsh

USER $CONDA_ADMIN
RUN conda init bash && \
    conda init zsh

USER $CONDA_USER
RUN conda init bash && \
    conda init zsh


USER root


# RUN chmod 770 -R $CONDA_PATH
RUN mkdir -p $JUPYTER_DATA_DIR && \
    # chown -R $CODA_ADMIN:$CONDA_GROUP $JUPYTER_DATA_DIR && \
    chgrp -R $CONDA_GROUP $JUPYTER_DATA_DIR && \
    chmod 770 -R $JUPYTER_DATA_DIR

RUN mkdir -p $JUPYTER_CONFIG_DIR && \
    # chown -R $CODA_ADMIN:$CONDA_GROUP $JUPYTER_DATA_DIR && \
    chgrp -R $CONDA_GROUP $JUPYTER_CONFIG_DIR && \
    chmod 770 -R $JUPYTER_CONFIG_DIR

# COPY --chown=$CODA_ADMIN:$CONDA_GROUP \
#    jupyter_notebook_config.py $JUPYTER_CONFIG_DIR/jupyter_notebook_config.py
COPY jupyter_notebook_config.py $JUPYTER_CONFIG_DIR/jupyter_notebook_config.py
RUN chown $CONDA_ADMIN:$CONDA_GROUP $JUPYTER_CONFIG_DIR/jupyter_notebook_config.py && \
    chmod 770 $JUPYTER_CONFIG_DIR/jupyter_notebook_config.py


# RUN chgrp -R $CONDA_GROUP $CONDA_PATH && \
#     chmod 770 -R $CONDA_PATH
# RUN chown -R $CODA_ADMIN:$CONDA_GROUP $CONDA_PATH && \
#     chmod 770 -R $CONDA_PATH


RUN sg - $CONDA_GROUP "conda install \
    ipython jupyter notebook \
    nb_conda ipykernel ipyparallel -y && \
    conda install -c conda-forge \
    ipywidgets \
    nb_conda_kernels \
    -y && \
    pip install tensorboard jupyter_tensorboard"

RUN sg - $CONDA_GROUP "conda install -c conda-forge \
    jupyterhub notebook -y"

# Prevent duplicated conda env list
RUN sg - $CONDA_GROUP "python -m nb_conda_kernels.install -v --disable"


RUN sg - $CONDA_GROUP "conda install -c conda-forge jupyter_contrib_nbextensions -y && \
    conda install -c conda-forge jupyter_nbextensions_configurator -y && \
    jupyter nbextensions_configurator enable && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable varInspector/main && \
    jupyter nbextension enable --py widgetsnbextension"

RUN sg - $CONDA_GROUP "mkdir -p $JUPYTER_DATA_DIR/nbextensions/jupyter_themes && \
    wget https://raw.githubusercontent.com/merqurio/jupyter_themes/master/theme_selector.js \
    -O $JUPYTER_DATA_DIR/nbextensions/jupyter_themes/theme_selector.js && \
    jupyter nbextension enable jupyter_themes/theme_selector"


# Install Python Kernels: with `conda`
# COPY --chown=root:$CONDA_GROUP install_new_kernel.sh install_new_kernel.sh
COPY install_new_kernel.sh install_new_kernel.sh
RUN chgrp $CONDA_GROUP install_new_kernel.sh && \
    chmod +x install_new_kernel.sh


COPY requirements-tf1-15.txt requirements-tf1-15.txt
RUN sg - $CONDA_GROUP "./install_new_kernel.sh \
    -v=3.7 \
    -n=onnx-tf1-15 \
    -d=onnx-py37-tf1-15 \
    -r=requirements-tf1-15.txt"
RUN rm requirements-tf1-15.txt

COPY requirements-tf2-1.txt requirements-tf2-1.txt
RUN sg - $CONDA_GROUP "./install_new_kernel.sh \
    -v=3.7 \
    -n=onnx-tf2-1 \
    -d=onnx-py37-tf2-1 \
    -r=requirements-tf2-1.txt"
RUN rm requirements-tf2-1.txt


# -------------------------------------------------------------------- #
# Install "software-properties-common" (for the "add-apt-repository")
RUN apt-get update && \
    apt-get install -y \
    software-properties-common

# Install default-jdk
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean


# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/default-java
RUN echo 'export JAVA_HOME="/usr/lib/jvm/default-java"' >> /etc/profile


# Install node.js & npm
RUN apt-get update && \
    # apt-get -y install curl gnupg && \
    curl -sL https://deb.nodesource.com/setup_12.x  | bash - && \
    apt-get -y install nodejs && \
    npm install -g npm

RUN npm install -g @bazel/bazelisk && \
    ln -s /usr/bin/bazelisk /usr/local/bin/bazel
# ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel

# RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.3.0/bazelisk-linux-amd64 \
#         -O /usr/local/bin/bazel && \
#         chmod +x /usr/local/bin/bazel


# -------------------------------------------------------------------- #

# RUN setfacl -Rm u::rwx,g:$CONDA_GROUP:rwx,m::rwx,o::0 $WORKDIR

ARG WORKDIR="/workspace"

RUN sg - $CONDA_GROUP "mkdir $WORKDIR" && \
    chown root:conda $WORKDIR && \
    chmod g+srwx $WORKDIR

RUN mkdir /repo
RUN sg - $CONDA_GROUP "git clone https://github.com/tensorflow/tensorflow /repo/tensorflow && \
    cd /repo/tensorflow && \
    bazel build tensorflow/tools/graph_transforms:summarize_graph"
    #--action_env PYTHON_BIN_PATH=/usr/bin/python3
    #--host_javabase=@local_jdk//:jdk
RUN ln -s "/repo/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph" \
    /usr/local/bin/summarize_graph


USER $CONDA_USER


RUN mkdir -p $WORKDIR
RUN mkdir -p $WORKDIR/converter_scripts && \
    mkdir -p $WORKDIR/inference_demos && \
    mkdir -p $WORKDIR/notebooks
COPY converter_scripts/ $WORKDIR/converter_scripts/
COPY inference_demos/ $WORKDIR/inference_demos/
COPY notebooks/ $WORKDIR/notebooks/

USER root
RUN chgrp -R $CONDA_GROUP $WORKDIR/converter_scripts/ && \
    chgrp -R $CONDA_GROUP $WORKDIR/inference_demos/ && \
    chgrp -R $CONDA_GROUP $WORKDIR/notebooks/ && \
    chmod 775 -R $WORKDIR/converter_scripts && \
    chmod 775 -R $WORKDIR/inference_demos && \
    chmod 775 -R $WORKDIR/notebooks



# RUN chgrp -R $CONDA_GROUP $CONDA_PATH && \
#     chmod 770 -R $CONDA_PATH
# RUN chgrp -R $CONDA_GROUP $WORKDIR && \
#     chmod 770 -R $WORKDIR


WORKDIR $WORKDIR

EXPOSE 8888

#USER $CONDA_USER
USER root

# Launch Jupyter notebook
SHELL ["/bin/bash", "-c"]

# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]


CMD ["jupyter", "notebook", "--allow-root", "--port=8888", "--ip=0.0.0.0", "--no-browser", "--config=$JUPYTER_CONFIG_DIR/jupyter_notebook_config.py"]

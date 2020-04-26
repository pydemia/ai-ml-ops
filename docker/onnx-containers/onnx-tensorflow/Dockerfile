FROM ubuntu:18.04
LABEL maintainer="Youngju Kim pydemia@gmail.com"
ARG DEBIAN_FRONTEND=noninteractive

# USER root

SHELL ["/bin/bash", "-c"]


RUN apt-get update && \
    apt-get install -y \
    build-essential \
    vim htop sudo \
    libcurl4-openssl-dev \
    libssl-dev wget curl unzip \
    git cmake tree \
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


# GCLOUD
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y

# AWS CLI 1 (for `/usr/bin/python`)
# RUN apt update && apt add \
#         ca-certificates \
#         groff \
#         less \
#         python \
#         py-pip && \
#     rm -rf /var/cache/apt/* && \
#     pip3 install pip --upgrade && \
#     pip3 install awscli

# AWS CLI 2
RUN curl https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscliv2.zip && \
    unzip awscliv2.zip && \
    ./aws/install

# AZURE CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash


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
    pip install tensorboard jupyter_tensorboard jupyterthemes"

RUN sg - $CONDA_GROUP "conda install -c conda-forge \
    jupyterhub notebook -y"


# Prevent duplicated conda env list
RUN sg - $CONDA_GROUP "python -m nb_conda_kernels.install -v --disable"


RUN sg - $CONDA_GROUP "conda install -c conda-forge \
    jupyter_contrib_nbextensions -y && \
    conda install -c conda-forge jupyter_nbextensions_configurator -y && \
    jupyter nbextensions_configurator enable && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable varInspector/main && \
    jupyter nbextension enable toc2/main && \
    jupyter nbextension enable table_beautifier/main && \
    jupyter nbextension enable freeze/main && \
    jupyter nbextension enable execute_time/ExecuteTime && \
    jupyter nbextension enable --py widgetsnbextension"

RUN sg - $CONDA_GROUP "pip install ipymagic && \
    jupyter nbextension install --py ipymagic && \
    jupyter nbextension enable ipymagic --py"

# RUN sg - $CONDA_GROUP "mkdir -p $JUPYTER_DATA_DIR/nbextensions/jupyter_themes && \
#     wget https://raw.githubusercontent.com/merqurio/jupyter_themes/master/theme_selector.js \
#     -O $JUPYTER_DATA_DIR/nbextensions/jupyter_themes/theme_selector.js && \
#     jupyter nbextension enable jupyter_themes/theme_selector"
RUN mkdir -p $JUPYTER_DATA_DIR/nbextensions/jupyter_themes
COPY theme_selector.js $JUPYTER_DATA_DIR/nbextensions/jupyter_themes/theme_selector.js
RUN chgrp -R $CONDA_GROUP $JUPYTER_DATA_DIR && \
    chmod 770 -R $JUPYTER_DATA_DIR && \
    sg - $CONDA_GROUP "jupyter nbextension enable jupyter_themes/theme_selector"



# Install Python Kernels: with `conda`
# COPY --chown=root:$CONDA_GROUP install_new_kernel.sh install_new_kernel.sh
COPY install_new_kernel.sh install_new_kernel.sh
RUN chgrp $CONDA_GROUP install_new_kernel.sh && \
    chmod 770 install_new_kernel.sh


# For onnx(INSTALL IN NON-CONDA ENVIRONMENT)
RUN apt install cmake protobuf-compiler libprotoc-dev -y


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

ENV WORKDIR="/workspace"

RUN sg - $CONDA_GROUP "mkdir $WORKDIR" && \
    chown root:conda $WORKDIR && \
    chmod g+srwx $WORKDIR


RUN pip install keras_preprocessing

RUN sg - $CONDA_GROUP "git clone https://github.com/tensorflow/tensorflow $WORKDIR/tensorflow-git && \
    cd $WORKDIR/tensorflow-git && \
    bazel build tensorflow/tools/graph_transforms:summarize_graph && \
    bazel build tensorflow/tools/graph_transforms:transform_graph && \
    bazel build tensorflow/python/tools:freeze_graph && \
    bazel build tensorflow/python/tools:optimize_for_inference && \
    cd /"
    #--action_env PYTHON_BIN_PATH=/usr/bin/python3
    #--host_javabase=@local_jdk//:jdk
RUN ln -s "$WORKDIR/tensorflow-git/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph" \
    /usr/local/bin/summarize_graph
RUN ln -s "$WORKDIR/tensorflow-git/bazel-bin/tensorflow/tools/graph_transforms/transform_graph" \
    /usr/local/bin/transform_graph
RUN ln -s "$WORKDIR/tensorflow-git/bazel-bin/tensorflow/python/tools/freeze_graph" \
    /usr/local/bin/freeze_graph
RUN ln -s "$WORKDIR/tensorflow-git/bazel-bin/tensorflow/python/tools/optimize_for_inference" \
    /usr/local/bin/optimize_for_inference


# ONNXRuntime Server: 
# CMake 3.13 or higher is required.  You are running version 3.10.2
# RUN /usr/bin/pip3 install cmake==3.13.3

# RUN sg - $CONDA_GROUP "git clone \
#         -b rel-1.1.1 \
#         --single-branch \
#         https://github.com/microsoft/onnxruntime $WORKDIR/onnxruntime-git && \
#     cd $WORKDIR/onnxruntime-git && \
#     python ./tools/ci_build/build.py \
#         --build_dir ./build \
#         --config Release \
#         --parallel \
#         --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) && \
#     cd /"
# RUN ln -s "$WORKDIR/onnxruntime-git/"

# # On VM:
# echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
# apt-get install apt-transport-https ca-certificates gnupg
# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
# apt-get update && apt-get install google-cloud-sdk


# # On Dockerfile:
# RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
#     curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
#     apt-get update -y && \
#     apt-get install google-cloud-sdk -y

# apt-get install apt-transport-https ca-certificates gnupg
# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
# apt-get update && apt-get install google-cloud-sdk

# git clone \
#     -b rel-1.1.1 \
#     --single-branch \
#     https://github.com/microsoft/onnxruntime $WORKDIR/onnxruntime-git && \
#     cd $WORKDIR/onnxruntime-git


# python ./tools/ci_build/build.py \
#     --build_dir ./build \
#     --config Release \
#     --parallel \
#     --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER)

# ./build.sh \
#     --use_openmp \
#     --build_dir ./build/Linux \
#     --config Release \
#     --build_wheel \
#     --update \
#     --build \
#     --parallel \
#     --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) && \
#     pip install ./build/Linux/Release/dist/*.whl

# master

# git clone --single-branch \
#     -b master \
#     --recursive \
#     https://github.com/Microsoft/onnxruntime onnxruntime-git-master

# ln -sfn /usr/lib/python3/dist-packages/apt_pkg.cpython-36m-x86_64-linux-gnu.so apt_pkg.so
# conda install -c conda-cluster python-apt



# ./tools/ci_build/github/linux/docker/scripts/install_ubuntu.sh \
#     -p $(python --version|egrep -o '([0-9]{1,}\.[0-9]{1,})') && \
# ./tools/ci_build/github/linux/docker/scripts/install_server_deps.sh

# curl -fsSL https://dl.google.com/go/go1.14.2.linux-amd64.tar.gz | tar -zxf - -C /usr/local/
# export PATH="/usr/local/go/bin:${PATH}"

# mkdir -p /onnxruntime/build && \
# cd /onnxruntime/build && \
# cmake -DCMAKE_BUILD_TYPE=Release /onnxruntime/server && \
# make -j$(getconf _NPROCESSORS_ONLN)






# COPY summarize_graph_2_1_prebuild summarize_graph
# COPY transform_graph_2_1_prebuild transform_graph
# RUN chgrp $CONDA_GROUP summarize_graph && \
#     chgrp $CONDA_GROUP transform_graph && \
#     sg - $CONDA_GROUP "cp transform_graph $CONDA_PATH/envs/onnx-tf1-15/bin/ && \
#     cp summarize_graph $CONDA_PATH/envs/onnx-tf1-15/bin/ && \
#     cp transform_graph $CONDA_PATH/envs/onnx-tf2-1/bin/ && \
#     cp summarize_graph $CONDA_PATH/envs/onnx-tf2-1/bin/ && \
#     rm summarize_graph && \
#     rm transform_graph"


# bazel build tensorflow/python/tools:saved_model_cli && \
# bazel build tensorflow/python/tools:freeze_graph && \
# bazel build tensorflow/python/tools:optimize_for_inference


# RUN mkdir -p $WORKDIR
RUN mkdir -p $WORKDIR/converter_scripts && \
    mkdir -p $WORKDIR/inference_demos && \
    mkdir -p $WORKDIR/notebooks && \
    mkdir -p $WORKDIR/trained_models

COPY converter_scripts/ $WORKDIR/converter_scripts/
COPY inference_demos/ $WORKDIR/inference_demos/
COPY notebooks/ $WORKDIR/notebooks/
COPY trained_models/ $WORKDIR/trained_models/

RUN git clone https://github.com/onnx/tutorials onnx-tutorials && \
    cp -r onnx-tutorials/tutorials/assets $WORKDIR/notebooks/ && \
    cp -r onnx-tutorials/tutorials $WORKDIR/onnx-tutorials && \
    rm -r onnx-tutorials

# RUN git clone --no-checkout https://github.com/onnx/tutorials && \
#     cd tutorials && \
#     git config core.sparseCheckout true && \
#     echo "tutorials/assets/*"> .git/info/sparse-checkout && \
#     git checkout master && \
#     cp -r tutorials/assets $WORKDIR/notebooks/ && \
#     cd ../ && \
#     rm -r tutorials


USER root
RUN chgrp -R $CONDA_GROUP $WORKDIR && \
    chmod 775 -R $WORKDIR



# RUN chgrp -R $CONDA_GROUP $CONDA_PATH && \
#     chmod 770 -R $CONDA_PATH
# RUN chgrp -R $CONDA_GROUP $WORKDIR && \
#     chmod 770 -R $WORKDIR


WORKDIR $WORKDIR

EXPOSE 8888

#USER $CONDA_USER
#USER root

# Launch Jupyter notebook
SHELL ["/bin/bash", "-c"]

# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]


CMD ["jupyter", "notebook", "--allow-root", "--port=8888", "--ip=0.0.0.0", "--no-browser", "--config=$JUPYTER_CONFIG_DIR/jupyter_notebook_config.py"]

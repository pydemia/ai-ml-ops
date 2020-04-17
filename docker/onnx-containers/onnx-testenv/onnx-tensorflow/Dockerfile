FROM ubuntu:18.04
MAINTAINER Youngju Kim "pydemia@gmail.com"
ARG DEBIAN_FRONTEND=noninteractive

# USER root

RUN apt-get update && \
    apt-get install -y sudo \
    build-essential curl \
    vim htop \
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


SHELL ["/bin/bash", "-c"]


# Install Conda

RUN apt-get update --fix-missing && \
    apt-get install -y \
    wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN apt-get install -y \
    libgl1-mesa-glx libegl1-mesa \
    libxrandr2 libxrandr2 libxss1 \
    libxcursor1 libxcomposite1 libasound2 \
    libxi6 libxtst6

ENV CONDA_PATH "/opt/conda"
ENV PATH "$CONDA_PATH/bin:$CONDA_PATH/sbin:$CONDA_PATH/condabin:$PATH"


RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh \
        -O install_conda3_py37.sh
RUN echo 'export CONDA_PATH="/opt/conda"' >> /etc/profile
RUN echo 'export PATH="$CONDA_PATH/bin:$CONDA_PATH/sbin:$CONDA_PATH/condabin:$PATH"' >> /etc/profile

# RUN source /etc/profile
# RUN source /etc/bash.bashrc


# RE-source `profile`
SHELL ["/bin/bash", "-c"]

#ENV PATH '$CONDA_PATH/bin:$CONDA_PATH/sbin:$CONDA_PATH/condabin:$PATH'

ARG CONDA_USER="pydemia"
ARG CONDA_GROUP="conda"
#ARG CONDA_UID="1000"
#ARG CONDA_GID="100"

RUN groupadd $CONDA_GROUP && \
    #useradd -m -G conda -p $(openssl passwd -1 password) pydemia && \
    useradd -m -G $CONDA_GROUP $CONDA_USER && \
    usermod -a -G $CONDA_GROUP $CONDA_USER && \
    usermod -a -G $CONDA_GROUP root

RUN sg - $CONDA_GROUP "bash ./install_conda3_py37.sh \
    -b \
    -p $CONDA_PATH" && \
    rm ./install_conda3_py37.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc


# ENV JUPYTER_CONFIG_DIR "/usr/local/etc/jupyter"
# ENV JUPYTER_DATA_DIR "/usr/local/share/jupyter"
ENV JUPYTER_CONFIG_DIR "/opt/conda/etc/jupyter"
ENV JUPYTER_DATA_DIR "/opt/conda/share/jupyter"

COPY jupyter_notebook_config.py $JUPYTER_CONFIG_DIR/jupyter_notebook_config.py

# RUN mkdir /usr/local/share/jupyter && \
#     chgrp -R $CONDA_GROUP /usr/local/share/jupyter && \
#     chmod 770 -R /usr/local/share/jupyter

RUN chgrp -R $CONDA_GROUP $JUPYTER_CONFIG_DIR && \
    chmod 770 -R $JUPYTER_CONFIG_DIR

# --- Replaced with `SG - $CONDA_GROUP "bash ./install_conda3_py37.sh" ---
# RUN chgrp -R $CONDA_GROUP $CONDA_PATH && \
# RUN chmod 770 -R $CONDA_PATH


# Install tf2

RUN conda init bash
RUN conda init zsh

SHELL ["/bin/bash", "-c"]

RUN conda install \
    nb_conda ipykernel ipyparallel -y && \
    conda install -c conda-forge \
    ipywidgets nb_conda_kernels -y && \
    pip install jupyter_tensorboard

RUN conda install -c conda-forge jupyter_contrib_nbextensions -y && \
    # jupyter contrib nbextension install && \
    conda install -c conda-forge jupyter_nbextensions_configurator -y && \
    jupyter nbextensions_configurator enable && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable varInspector/main && \
    jupyter nbextension enable --py widgetsnbextension

RUN mkdir -p $JUPYTER_DATA_DIR/nbextensions/jupyter_themes && \
    wget https://raw.githubusercontent.com/merqurio/jupyter_themes/master/theme_selector.js \
        -O $JUPYTER_DATA_DIR/nbextensions/jupyter_themes/theme_selector.js && \
    jupyter nbextension enable jupyter_themes/theme_selector


COPY requirements-tf1-15.txt requirements-tf1-15.txt
COPY requirements-tf2-1.txt requirements-tf2-1.txt
COPY install_new_kernel.sh install_new_kernel.sh
RUN chmod +x install_new_kernel.sh

RUN ./install_new_kernel.sh \
        -v=3.7 \
        -n=onnx-tf1-15 \
        -d=onnx-py37-tf1-15 \
        -r=requirements-tf1-15.txt

RUN ./install_new_kernel.sh \
        -v=3.7 \
        -n=onnx-tf2-1 \
        -d=onnx-py37-tf2-1 \
        -r=requirements-tf2-1.txt


# Install Python Requirements
# RUN pip3 install -r requirements.txt


# Install "software-properties-common" (for the "add-apt-repository")
RUN apt-get update && \
    apt-get install -y \
    software-properties-common

# # Add the "JAVA" ppa
# RUN add-apt-repository -y \
#     ppa:webupd8team/java
# 
# # Install OpenJDK-8
# RUN apt-get update && \
#     apt-get install -y openjdk-8-jdk && \
#     apt-get install -y ant && \
#     apt-get clean;
# 
# Fix certificate issues
# RUN apt-get update && \
#     apt-get install ca-certificates-java && \
#     apt-get clean && \
#     update-ca-certificates -f;

# Install default-jdk
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean;


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

ARG WORKDIR="/scripts"

RUN mkdir $WORKDIR

RUN git clone https://github.com/tensorflow/tensorflow $WORKDIR/tensorflow && \
        cd $WORKDIR/tensorflow && \
        bazel build tensorflow/tools/graph_transforms:summarize_graph
            #--action_env PYTHON_BIN_PATH=/usr/bin/python3
            #--host_javabase=@local_jdk//:jdk
RUN ln -s "$WORKDIR/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph" \
    /usr/local/bin/summarize_graph


RUN cd $WORKDIR && \
    mkdir converter_scripts && \
    mkdir inference_demos
COPY converter_scripts/ scripts/converter_scripts/
COPY inference_demos/ scripts/inference_demos/
# COPY requirements-tf1-15.txt requirements.txt


WORKDIR $WORKDIR

EXPOSE 8888

# Launch Jupyter notebook
SHELL ["/bin/bash", "--login", "-c"]

# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

ARG CONDA_PATH="/opt/conda"
CMD ["jupyter", "notebook", "--allow-root", "--port=8888", "--ip=0.0.0.0", "--no-browser", "--config=$JUPYTER_CONFIG_DIR/jupyter_notebook_config.py"]

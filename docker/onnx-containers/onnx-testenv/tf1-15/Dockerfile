FROM ubuntu:18.04
MAINTAINER Youngju Kim "pydemia@gmail.com"
ARG DEBIAN_FRONTEND=noninteractive

# USER root

RUN apt-get update && \
    apt-get install -y sudo \
    build-essential curl \
    libcurl4-openssl-dev \
    libssl-dev wget \
    python3-pip \
    git && \
    pip3 install --upgrade pip

RUN mkdir scripts && \
        cd scripts && \
        mkdir converter_scripts && \
        mkdir inference_demos
COPY converter_scripts/ scripts/converter_scripts/
COPY inference_demos/ scripts/inference_demos/
# COPY requirements-tf1-15.txt requirements.txt

# Set Locale
RUN apt-get update -y && \
    apt-get install -y locales && \
    locale-gen --purge "en_US.UTF-8"

RUN bash -c "echo 'LC_ALL=en_US.UTF-8' >> /etc/environment"
RUN bash -c "echo 'en_US.UTF-8 UTF-8' >> /etc/locale.gen"
RUN bash -c "echo 'LANG=en_US.UTF-8' > /etc/locale.conf"

# Install Conda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh \
        -O install_conda3_py37.sh
RUN bash -c "echo 'export CONDA_PATH=\"/opt/conda\"' >> /etc/profile"
RUN bash -c "echo 'export PATH=\"\$CONDA_PATH/bin:\$CONDA_PATH/sbin:\$CONDA_PATH/condabin:\$PATH\"' >> /etc/profile"
# RUN source /etc/profile
# RUN source /etc/bash.bashrc

ENV CONDA_PATH "/opt/conda"
ENV PATH "$CONDA_PATH/bin:$CONDA_PATH/sbin:$CONDA_PATH/condabin:$PATH"

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN apt-get install -y \
        libgl1-mesa-glx libegl1-mesa \
        libxrandr2 libxrandr2 libxss1 \
        libxcursor1 libxcomposite1 libasound2 \
        libxi6 libxtst6

RUN bash ./install_conda3_py37.sh \
    -b \
    -p $CONDA_PATH && \
    rm ./install_conda3_py37.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN mkdir /usr/local/share/jupyter
RUN groupadd conda && \
    chgrp -R conda $CONDA_PATH && \
    chmod 770 -R $CONDA_PATH && \
    chgrp -R conda /usr/local/share/jupyter && \
    chmod 770 -R /usr/local/share/jupyter && \
    usermod -a -G conda root


# Install tf2
SHELL ["/bin/bash", "--login", "-c"]

RUN conda init bash
RUN conda init zsh


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

SHELL ["/bin/bash", "-c"]

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

# 
# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/default-java
RUN export JAVA_HOME


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


RUN git clone https://github.com/tensorflow/tensorflow && \
        cd tensorflow && \
        bazel build tensorflow/tools/graph_transforms:summarize_graph
            #--action_env PYTHON_BIN_PATH=/usr/bin/python3
            #--host_javabase=@local_jdk//:jdk
RUN ln -s "$(pwd)/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph" \
    /usr/local/bin/summarize_graph



WORKDIR /scripts

EXPOSE 8888

# Launch Jupyter notebook
CMD ["jupyter", "notebook", "--allow-root", "--port=8888", "--ip=0.0.0.0", "--no-browser"]

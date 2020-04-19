FROM ubuntu:18.04
LABEL maintainer="Youngju Kim pydemia@gmail.com"
ARG DEBIAN_FRONTEND=noninteractive

# USER root

SHELL ["/bin/bash", "-c"]


RUN apt-get update && \
  apt-get install -y sudo \
  build-essential curl \
  vim htop wget \
  git \
  acl \
  python3-pip && \
  pip3 install --upgrade pip


# -------------------------------------------------------------------- #

ARG CONDA_USER="pydemia"
ARG CONDA_GROUP="conda"
# ARG CONDA_UID="1000"
# ARG CONDA_GID="100"

ENV CONDA_PATH "/opt/conda"
ENV PATH "$CONDA_PATH/bin:$CONDA_PATH/sbin:$CONDA_PATH/condabin:$PATH"

# `-r` option: For system group(gid<1000), which is prior to user group(gid>=1000)
RUN groupadd -r $CONDA_GROUP && \
  #useradd -m -G conda -p $(openssl passwd -1 password) $CONDA_USER && \
  useradd -m -G $CONDA_GROUP $CONDA_USER && \
  # usermod -a -G $CONDA_GROUP $CONDA_USER && \
  usermod -a -G $CONDA_GROUP root


ARG WORKDIR="/workspace"

RUN sg - $CONDA_GROUP "mkdir $WORKDIR"

RUN setfacl -Rm d:g:conda:rwx,g:conda:rwx $WORKDIR
# RUN bash -c "setfacl -Rm d:g:$CONDA_GROUP:rwx $WORKDIR"
# RUN sg - $CONDA_GROUP "mkdir $WORKDIR" && \
#     chmod -R g+s $WORKDIR && \
#     setfacl -Rmd g:$CONDA_GROUP:rwx $WORKDIR
# RUN setfacl -Rm d:g:$CONDA_GROUP:rwx,u::rwx,g:$CONDA_GROUP:rwx,m::rwx,o::0 $WORKDIR
# RUN sg - $CONDA_GROUP "mkdir $WORKDIR" && \
#     chown root:conda $WORKDIR && \
#     chmod g+srwx $WORKDIR
# RUN sg - $CONDA_GROUP "git clone https://github.com/tensorflow/tensorflow $WORKDIR/.repo/tensorflow && \
#     cd $WORKDIR/.repo/tensorflow && \
#     bazel build tensorflow/tools/graph_transforms:summarize_graph"
#     #--action_env PYTHON_BIN_PATH=/usr/bin/python3
#     #--host_javabase=@local_jdk//:jdk
# RUN ln -s "$WORKDIR/.repo/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph" \
#     /usr/local/bin/summarize_graph


RUN sg - $CONDA_GROUP "mkdir -p $WORKDIR/converter_scripts && \
  mkdir -p $WORKDIR/inference_demos"
COPY converter_scripts/ $WORKDIR/converter_scripts/
COPY inference_demos/ $WORKDIR/inference_demos/

RUN chgrp -R $CONDA_GROUP $WORKDIR/converter_scripts/ && \
  chgrp -R $CONDA_GROUP $WORKDIR/inference_demos/


RUN sg - $CONDA_GROUP "mkdir -p $WORKDIR/notebooks"
# RUN chmod 775 -R $WORKDIR

WORKDIR $WORKDIR

EXPOSE 8888

USER $CONDA_USER

# Launch Jupyter notebook
SHELL ["/bin/bash", "-c"]

# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

CMD ["bash"]
# CMD ["jupyter", "notebook", "--allow-root", "--port=8888", "--ip=0.0.0.0", "--no-browser", "--config=$JUPYTER_CONFIG_DIR/jupyter_notebook_config.py"]

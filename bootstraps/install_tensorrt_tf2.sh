#!/bin/bash

export CONDA_ENV_NM_NEW="tensorrt-tf2"
export PY_VER="3.7"
# export BASE_ENVFILE="conda_envfile_tf2-py37.yaml"
# export PIP_PKG_REQ="requirements_tf2-py37_pip.txt"

if [ -z "$BASE_ENVFILE" ]; then
  conda create \
    -n "$CONDA_ENV_NM_NEW" \
    python="$PY_VER" \
    ipykernel -y \
  && \
  conda activate "$CONDA_ENV_NM_NEW" \
  && \
  python -m ipykernel install \
    --name "$CONDA_ENV_NM_NEW" \
    --display-name 'tensorrt-py37-tf2-onnx1.7' \
  # && \
  # pip install -r "$PIP_PKG_REQ" \
  #   --ignore-installed \
  # && \
else
  envsubst < "$BASE_ENVFILE" > "conda_envfile.yaml" \
  && \
  conda env create \
    -n "$CONDA_ENV_NM_NEW" \
    python="$PY_VER" \
    --file "$BASE_ENVFILE" \
  && \
  conda activate "$CONDA_ENV_NM_NEW" \
  && \
  python -m ipykernel install \
    --name "$CONDA_ENV_NM_NEW" \
    --display-name 'tensorrt-py37-tf2-onnx1.7' \
  # && \
  # pip install -r "$PIP_PKG_REQ" \
  #   --ignore-installed \
  # && \
fi

echo "$CONDA_ENV_NM_NEW has been installed."

pip install tensorflow-gpu==2.1 #tensorflow==2.1
conda install -c conda-forge protobuf -y
#pip install onnx==1.7.0
gsutil -m cp gs://yjkim-repository/python_packages/onnx-1.7.0-cp37-cp37m-linux_x86_64.whl ./

pip install onnx-1.7.0-cp37-cp37m-linux_x86_64.whl
pip install pytorch==1.5
pip install -U tf2onnx==1.5.6

conda deactivate
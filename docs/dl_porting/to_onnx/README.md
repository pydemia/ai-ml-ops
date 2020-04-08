# Porting to ONNX

---
## Table of Contents

* to ONNX:
  * Tensorflow -> ONNX
  * PyTorch(Caffe2) -> ONNX
  * Caffe -> ONNX

--- 


### Prerequisite

```sh
gcloud beta compute ssh --zone "us-central1-a" "yjkim-dl-gpu-template-1" --project "ds-ai-platform"
```

* tensorrt-tf2-1
```sh
conda create -n tensorrt-tf2 python=3.7 ipykernel -y
conda activate tensorrt-tf2
python -m ipykernel install --user --name tensorrt-tf2 --display-name "tensorrt-py37-tf2-onnx1.7"

pip install tensorflow-gpu==2.1 #tensorflow==2.1
pip install onnx==1.7.0
pip install pytorch==1.5
pip install tf2onnx==1.5.6

git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
python setup.py sdist bdist bdist_wheel
# python setup.py install


gsutil -m cp gs://yjkim-repository/python_packages/onnx-1.7.0-cp37-cp37m-linux_x86_64.whl ./
gsutil -m cp gs://yjkim-repository/python_packages/onnx-1.7.0.linux-x86_64.tar.gz ./
gsutil -m cp gs://yjkim-repository/python_packages/onnx-1.7.0.tar.gz ./


conda deactivate
```

```sh
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
  #   --user \
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

pip install tensorflow-gpu==2.1 #tensorflow==2.1
pip install onnx==1.7.0
pip install pytorch==1.5
pip install tf2onnx==1.5.6

conda deactivate

```

* tensorrt-tf1-15
```sh
conda create -n tensorrt-tf1 python=3.7 ipykernel -y
conda activate tensorrt-tf1
python -m ipykernel install --user --name tensorrt-tf1 --display-name "tensorrt-py37-tf1-onnx1.7"

pip install tensorflow-gpu==1.15 #tensorflow==1.15
pip install onnx==1.7.0
pip install pytorch==1.5
pip install -U tf2onnx==1.5.6

# botocore 1.12.162 requires docutils>=0.10, which is not installed.
# ERROR: botocore 1.12.162 requires jmespath<1.0.0,>=0.7.1, which is not installed.
# ERROR: awscli 1.16.172 requires docutils>=0.10, which is not installed.
# ERROR: awscli 1.16.172 requires PyYAML<=3.13,>=3.10, which is not installed.
# ERROR: awscli 1.16.172 requires s3transfer<0.3.0,>=0.2.0, which is not installed.

conda deactivate
```

```sh
export CONDA_ENV_NM_NEW="tensorrt-tf1"
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
  #   --user \
    --name "$CONDA_ENV_NM_NEW" \
    --display-name 'tensorrt-py37-tf1-onnx1.7' \
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
    --display-name 'tensorrt-py37-tf1-onnx1.7' \
  # && \
  # pip install -r "$PIP_PKG_REQ" \
  #   --ignore-installed \
  # && \
fi

pip install tensorflow-gpu==1.15 #tensorflow==1.15
pip install onnx==1.7.0
pip install pytorch==1.5
pip install -U tf2onnx==1.5.6

conda deactivate

```

## Tensorflow to ONNX

[github onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)

* Template

```sh
python -m tf2onnx.convert  \
    [--input SOURCE_GRAPHDEF_PB] \
    [--graphdef SOURCE_GRAPHDEF_PB] \
    [--checkpoint SOURCE_CHECKPOINT] \
    [--saved-model SOURCE_SAVED_MODEL] \
    [--output TARGET_ONNX_MODEL] \
    [--inputs GRAPH_INPUTS] \
    [--outputs GRAPH_OUTPUS] \
    [--inputs-as-nchw inputs_provided_as_nchw] \
    [--opset OPSET] \
    [--target TARGET] \
    [--custom-ops list-of-custom-ops] \
    [--fold_const] \
    [--continue_on_error] \
    [--verbose]

```

* Test Model
  - `gs://yjkim-outputs/aiplatform-jobs/builtin-image-classification/test-000`
  - CKPT: `{GIT_DIR}/models/aiplatform-jobs/builtin-image-classification/test-000`
  - SavedModel: `{GIT_DIR}/models/aiplatform-jobs/builtin-image-classification/test-000/model`

```sh
python -m tf2onnx.convert  \
    [--input SOURCE_GRAPHDEF_PB] \
    [--graphdef SOURCE_GRAPHDEF_PB] \
    [--checkpoint SOURCE_CHECKPOINT] \
    [--saved-model SOURCE_SAVED_MODEL] \
    [--output TARGET_ONNX_MODEL] \
    [--inputs GRAPH_INPUTS] \
    [--outputs GRAPH_OUTPUS] \
    [--inputs-as-nchw inputs_provided_as_nchw] \
    [--opset OPSET] \
    [--target TARGET] \
    [--custom-ops list-of-custom-ops] \
    [--fold_const] \
    [--continue_on_error] \
    [--verbose]
```

```sh
python -m tf2onnx.convert  \
    --saved-model ./models/aiplatform-jobs/builtin-image-classification/test-000/model \
    --verbose
```

### Error Case

* `saved_model` is recommended.
* `tensorflow-2.x`
* `protobuf(ldconfig)`
* `parsing error`

```sh
'Value "{}" is not valid attribute data type.'.format(value))
ValueError: Value "name: "__inference_signature_wrapper_22170"
" is not valid attribute data type.
```
* 

## PyTorch(Caffe2) to ONNX



## Caffe to ONNX



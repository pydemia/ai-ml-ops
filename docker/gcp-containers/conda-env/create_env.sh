#!/bin/zsh

# pip freeze > requirements_tf2-py37_pip.txt
# conda list -e > requirements_tf2-py37_conda.txt
# conda env export > conda_envfile_tf2-py37.yml

export CONDA_ENV_NM_NEW="tf2-py37-dockerdev"
export PY_VER="3.7"
export BASE_ENVFILE="conda_envfile_tf2-py37.yaml"
export PIP_PKG_REQ="requirements_tf2-py37_pip.txt"


#   --file "conda_envfile.yaml" \
envsubst < "$BASE_ENVFILE" > "conda_envfile.yaml" \
&& \
# conda create \
#   -n "$CONDA_ENV_NM_NEW" \
#   python="$PY_VER" \
#   ipykernel -y \
conda env create \
  -n "$CONDA_ENV_NM_NEW" \
  python="$PY_VER" \
  --file "$BASE_ENVFILE" \
&& \
conda activate "$CONDA_ENV_NM_NEW" \
&& \
python -m ipykernel install \
  --user \
  --name "$CONDA_ENV_NM_NEW" \
  --display-name 'Py"{$PY_VER}"-TF2-dockerdev (conda env)' \
&& \
# pip install -r "$PIP_PKG_REQ" \
#   --ignore-installed \
# && \
conda deactivate

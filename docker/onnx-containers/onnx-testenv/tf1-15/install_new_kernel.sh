#!/bin/bash


PY_VER="3.7"
CONDA_ENV_NM="py37-tf2-1"
CONDA_DISP_NM=$CONDA_ENV_NM
BASE_ENVFILE=""
PIP_PKG_REQ=""

# -v=3.7 -n="onnx-tf2-1" -d "onnx-py37-tf2-1" -r="requirements.txt"


for i in "$@"; do
  case $i in
    -v=*|--pyversion=*)
    PY_VER="${i#*=}"
    shift # past argument=value
    ;;
    -n=*|--env_name=*)
    CONDA_ENV_NM="${i#*=}"
    shift # past argument=value
    ;;
    -d=*|--display_name=*)
    CONDA_DISP_NM="${i#*=}"
    shift # past argument=value
    ;;
    -e=*|--conda_yaml=*)
    BASE_ENVFILE="${i#*=}"
    shift # past argument=value
    ;;
    -r=*|--pip_req=*)
    PIP_PKG_REQ="${i#*=}"
    shift # past argument=value
    ;;
  esac
done

USAGE='
-v      PY_VER="3.7"
-n      CONDA_ENV_NM="py37-tf2-1"
-d      CONDA_DISP_NM="Python3.7-tf2.1 (conda env)"
-e      BASE_ENVFILE="conda_envfile.yaml"
-r      PIP_PKG_REQ="requirements.txt"
'

if which getopt > /dev/null 2>&1; then
    OPTS=$(getopt hvn:der "$*" 2>/dev/null)
    if [ ! $? ]; then
        printf "%s\\n" "$USAGE"
        exit 2
    fi

    eval set -- "$OPTS"

    while true; do
        case "$1" in
            -h)
                printf "%s\\n" "$USAGE"
                exit 2
                ;;
            -v)
                PY_VER="$2"
                shift
                ;;
            -n)
                CONDA_ENV_NM="$2"
                shift
                ;;
            -d)
                CONDA_DISP_NM="$2"
                shift
                shift
                ;;
            -e)
                BASE_ENVFILE="$2"
                shift
                ;;
            -r)
                PIP_PKG_REQ="$2"
                shift
                ;;
            --)
                shift
                break
                ;;
            *)
                printf "ERROR: did not recognize option '%s', please try -h\\n" "$1"
                exit 1
                ;;
        esac
    done
else
    while getopts "hvn:der" x; do
        case "$x" in
            h)
                printf "%s\\n" "$USAGE"
                exit 2
            ;;
            v)
                PY_VER="3.7"
                ;;
            n)
                CONDA_ENV_NM="py37-tf2-1"
                ;;
            d)
                CONDA_DISP_NM="Python3.7-tf2.1 (conda env)"
                ;;
            e)
                BASE_ENVFILE=""
                ;;
            r)
                PIP_PKG_REQ=""
                ;;
            ?)
                printf "ERROR: did not recognize option '%s', please try -h\\n" "$x"
                exit 1
                ;;
        esac
    done
fi


if [[ -n $CONDA_DISP_NM ]]; then
  CONDA_DISP_NM=$CONDA_ENV_NM
fi

echo "PY_VER=$PY_VER"
echo "CONDA_ENV_NM=$CONDA_ENV_NM"
echo "CONDA_DISP_NM=$CONDA_DISP_NM"
echo "BASE_ENVFILE=$BASE_ENVFILE"
echo "PIP_PKG_REQ=$PIP_PKG_REQ"



if [[ -z $BASE_ENVFILE ]]; then
  conda create -n "$CONDA_ENV_NM" python="$PY_VER" ipykernel -y
else
  # envsubst < "$BASE_ENVFILE" > "conda_envfile.yaml" && \
  conda env create \
    -n $CONDA_ENV_NM \
    python=$PY_VER \
    --file $BASE_ENVFILE
fi


if [[ -z $PIP_PKG_REQ ]]; then
  conda activate $CONDA_ENV_NM && \
  python -m ipykernel install \
    --name $CONDA_ENV_NM \
    --display-name $CONDA_DISP_NM
else
  conda activate $CONDA_ENV_NM && \
  python -m ipykernel install \
    --name $CONDA_ENV_NM \
    --display-name $CONDA_DISP_NM && \
  pip install -r $PIP_PKG_REQ \
    --ignore-installed
fi

# pip install tensorflow-gpu==2.1 #tensorflow==2.1
conda install -c conda-forge protobuf -y
# pip install onnx==1.7.0
# gsutil -m cp gs://yjkim-repository/python_packages/onnx-1.7.0-cp37-cp37m-linux_x86_64.whl ./

# pip install onnx-1.7.0-cp37-cp37m-linux_x86_64.whl
# pip install pytorch==1.5
# pip install -U tf2onnx==1.5.6

conda deactivate


echo "$CONDA_ENV_NM has been installed."
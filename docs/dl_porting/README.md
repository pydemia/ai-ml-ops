# DL Framework Portability Test

DL Frameworks

* Tensorflow == 1.12-1.15, 2.1 (onnx opset == 7-11)
* PyTorch(Caffe2) 1.4
* Caffe
* ONNX >= 1.7
* TensorRT



* onnx-tensorflow
> ONNX-TF requires ONNX (Open Neural Network Exchange) as an external dependency, for any issues related to ONNX installation, we refer our users to ONNX project repository for documentation and help. Notably, please ensure that protoc is available if you plan to install ONNX via pip.
> 
> The specific ONNX release version that we support in the master branch of ONNX-TF can be found here. This information about ONNX version requirement is automatically encoded in setup.py, therefore users needn't worry about ONNX version requirement when installing ONNX-TF.
> 
> To install the latest version of ONNX-TF via pip, run pip install onnx-tf.
> 
> Because users often have their own preferences for which variant of Tensorflow to install (i.e., a GPU version instead of a CPU version), we do not explicitly require tensorflow in the installation script. It is therefore users' responsibility to ensure that the proper variant of Tensorflow is available to ONNX-TF. Moreoever, we require Tensorflow version == 1.15.0.

* onnx version

ONNX version|File format version|Opset version ai.onnx|Opset version ai.onnx.ml|Opset version ai.onnx.training
------------|-------------------|---------------------|------------------------|------------------------------
1.0|3|1|1|-
1.1|3|5|1|-
1.1.2|3|6|1|-
1.2|3|7|1|-
1.3|3|8|1|-
1.4.1|4|9|1|-
1.5.0|5|10|1|-
1.6.0|6|11|2|-
1.7.0|7|12|2|1


* tensorrt

> It should be installed on `root`. It can cause errors without `root` privilege.

```sh
os=”ubuntu1x04”
tag=”cudax.x-trt7.x.x.x-ga-yyyymmdd”
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${tag}/7fa2af80.pub

sudo apt-get update
sudo apt-get install tensorrt

sudo apt-get install python3-libnvinfer-dev # python-libnvinfer-dev
sudo apt-get install uff-converter-tf



os=”ubuntu1x04”
tag=”cudax.x-trt7.x.x.x-ga-yyyymmdd”
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${tag}/7fa2af80.pub

sudo apt-get update
sudo apt-get install libnvinfer7

```

```sh

pip install 'pycuda>=2019.1.1'

```


Development:
---

## Table of Contents

* to ONNX:
  * Tensorflow -> ONNX
  * PyTorch(Caffe2) -> ONNX
  * Caffe -> ONNX
* to TensorRT
  * TensorRT -> ONNX
  * Tensorflow -> TensorRT
  * PyTorch -> TensorRT


```sh
gcloud beta compute ssh --zone "us-central1-a" "yjkim-dl-tensorrt-tf2-1-template-1" --project "ds-ai-platform"
```
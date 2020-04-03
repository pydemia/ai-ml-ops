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
conda create -n tensorrt python=3.7 ipykernel -y
conda activate tensorrt
python -m ipykernel install --user --name tensorrt --display-name "Py37-tf2.1-onnx1.7 (conda env)"

pip install tensorflow-gpu==2.1 #tensorflow==2.1
pip install onnx==1.7.0
pip install pytorch==1.5
pip install -U tf2onnx

```


## Tensorflow to ONNX

[github onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)




## PyTorch(Caffe2) to ONNX



## Caffe to ONNX



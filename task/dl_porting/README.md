# DL Framework Portability Test


## Overview

### Objective

Use `TensorRT` to accelerate inference of a pre-trained model.


> TensorRT: NVIDIA's SDK for high-performance deep learning inference acceleration. > It includes a deep learning inference optimizer and runtime.
> 
> [NVIDIA Dev. Blog: How to Speed Up Deep Learning Inference Using TensorRT (2018.11)](https://devblogs.nvidia.com/speed-up-inference-tensorrt/)
> [NVIDIA Dev. Blog: Speeding up Deep Learning Inference Using TensorFlow, ONNX and TensorRT (2020.03)](https://devblogs.nvidia.com/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/)
> [NVIDIA Dev. Blog: TensorRT Integration Speeds Up TensorFlow Inference(2018.03)](https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/)


![](TensorRT-inference-accelerator-768x296.png)
![](onnx.png)


### Test

#### Case: Converting a pre-trained model to `tensorrt`

1. with `onnx` Converter

| From | To(1 depth) | To(2 depth) |
| :--- | :--- | :--- |
| `tensorflow` | `onnx` | `tensorrt` |
| ~~`pytorch`~~ | ~~`onnx`~~ | ~~`tensorrt`~~ |
| ~~`caffe`~~ | ~~`onnx`~~ | ~~`tensorrt`~~ |


2. without `onnx` Converter

| From | To(1 depth) | To(2 depth) |
| :--- | :--- | :--- |
| `tensorflow` | `tensorrt` |
| ~~`pytorch`~~ | ~~`tensorrt`~~ |
| ~~`caffe`~~ | ~~`tensorrt`~~ |

#### Task List

  1. `tensorflow` -> `onnx`
  2. `onnx` -> `TensorRT`
  3. `tensorflow` -> `TensorRT`

---

### 1. `tensorflow` -> `onnx`

API Type:
* tf-native
  - low/mid level API (with `tf.Session`, `tf.layer`)
  - high-level API
    * `tf.estimator`
    * tf.keras
      * `tf` only(no deps)
      * dep: `keras.io`(ex. `tf.keras.applications`)

Versions:
* `tf-1.x`
* `tf-2.x`

Device Type:
* GPU
  * Single-GPU
  * Multi-GPU



DL Frameworks

* Tensorflow == 1.12-1.15, 2.1 (onnx opset == 7-11)
* ~~PyTorch(Caffe2) 1.4~~
* ~~Caffe~~
* ONNX >= 1.6
* TensorRT





`gsutil -m cp gs://yjkim-repository/python_packages/onnx-1.7.0-cp37-cp37m-linux_x86_64.whl ./`
`gsutil -m cp gs://yjkim-repository/python_packages/onnx-1.7.0.linux-x86_64.tar.gz ./`
`gsutil -m cp gs://yjkim-repository/python_packages/onnx-1.7.0.tar.gz ./`






```sh
gcloud beta compute ssh --zone "us-central1-a" "yjkim-dl-gpu-template-1" --project "ds-ai-platform"
```
{
  "scaleTier": "CUSTOM",
  "masterType": "large_model",
  "workerType": "complex_model_m_p100",
  "parameterServerType": "standard_p100",
  "workerCount": "4",
  "parameterServerCount": "1",
  "packageUris": [
    "gs://yjkim-models/examples/gcp/custom_code/resnet_fmnist_keras/dist/fmnist_resnet-1.2-py3-none-any.whl"
  ],
  "pythonModule": "resnet.task",
  "args": [
    "--train-file=gs://yjkim-dataset/images/fashion-mnist/train-images-idx3-ubyte.gz",
    "--train-labels=gs://yjkim-dataset/images/fashion-mnist/train-labels-idx1-ubyte.gz",
    "--test-file=gs://yjkim-dataset/images/fashion-mnist/t10k-labels-idx1-ubyte.gz",
    "--test-labels-file=gs://yjkim-dataset/images/fashion-mnist/t10k-images-idx3-ubyte.gz",
    "--num-epochs=1000",
    "--batch-size=256",
    "--learning-rate=0.001",
    "--verbosity=INFO"
  ],
  "region": "us-central1",
  "runtimeVersion": "2.1",
  "jobDir": "gs://yjkim-outputs/aiplatform-jobs/custom-fmnist-resnet/test-01/",
  "pythonVersion": "3.7"
}
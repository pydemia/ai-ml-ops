# TPU Settings

## RUN on `cloud-shell`

```sh
ctpu up -machine-type "n1-standard-8" -name "yjkim-tpu"

```

## MNIST


## COCO

```sh
export STORAGE_BUCKET=gs://yjkim-dataset/images/mscoco-gcp

cd /usr/share/tpu/tools/datasets
sudo bash /usr/share/tpu/tools/datasets/download_and_preprocess_coco.sh ./data/dir/coco

gsutil -m cp ./data/dir/coco/*.tfrecord ${STORAGE_BUCKET}/coco
gsutil cp ./data/dir/coco/raw-data/annotations/*.json ${STORAGE_BUCKET}/coco

```

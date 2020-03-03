# TPU Settings

## RUN on `cloud-shell`

```sh
ctpu up -machine-type "n1-standard-8" -name "yjkim-tpu"

```

| 머신 이름 | vCPUs1 | MEM | 최대PD수 | 최대PD크기(TB) | LocalSSD | Network Egress 대역폭(Gbps) |
| :-------------: | :--: | :---: | :-: | :-: | :-: | :-: |
| n1-standard-1   |   1	|   3.75 | 128 | 257 | 예 |   2 |
| n1-standard-2   |   2	|   7.50 | 128 | 257 | 예 |  10 |
| n1-standard-4   |   4	|   15	 | 128 | 257 | 예 |  10 |
| n1-standard-8   |   8	|   30	 | 128 | 257 | 예 |  16 |
| n1-standard-16  |  16	| 60	 | 128 | 257 | 예 | 324 |
| n1-standard-32  |  32	| 120	 | 128 | 257 | 예 | 324 |
| n1-standard-64  |  64	| 240	 | 128 | 257 | 예 | 324 |
| n1-standard-96  |  96	| 360	 | 128 | 257 | 예 | 324 |


> Splitting a TFRecords file into multiple shards has essentially 3 advantages:
> 
> * Easier to shuffle.  
>   As others have pointed out, it makes it easy to shuffle the data at a coarse level (before using a shuffle buffer).
> * Faster to download.  
>   If the files are spread across multiple servers, downloading several files from different servers in parallel will optimize bandwidth usage (rather than downloading one file from a single server). This can improve performance significantly compared to downloading the data from a single server.
> * Simpler to manipulate.  
>   It's easier to deal with 10,000 files of 100MB each rather than with a single 1TB file. Huge files can be a pain to handle: in particular, transfers are much more likely to fail. It's also harder to manipulate subsets of the data when it's all in a single file.


## MNIST

```sh
python /usr/share/tensorflow/tensorflow/examples/how_tos/reading_data/convert_to_records.py --directory=./data
gunzip ./data/*.gz

export STORAGE_BUCKET=gs://yjkim-dataset/images/mnist
gsutil cp -r ./data ${STORAGE_BUCKET}

```

## COCO

```sh
export STORAGE_BUCKET=gs://yjkim-dataset/images/mscoco-gcp

cd /usr/share/tpu/tools/datasets
sudo bash /usr/share/tpu/tools/datasets/download_and_preprocess_coco.sh ./data/dir/coco

gsutil -m cp ./data/dir/coco/train*.tfrecord ${STORAGE_BUCKET}/train
gsutil -m cp ./data/dir/coco/val*.tfrecord ${STORAGE_BUCKET}/val
gsutil -m cp ./data/dir/coco/test*.tfrecord ${STORAGE_BUCKET}/test
gsutil -m cp ./data/dir/coco/unlabeled*.tfrecord ${STORAGE_BUCKET}/unlabeled

gsutil cp -r ./data/dir/coco/raw-data/annotations ${STORAGE_BUCKET}


```

# TPU Settings

## RUN on `cloud-shell`

```sh
ctpu up -machine-type "n1-standard-8" -name "yjkim-tpu"

```

머신 이름	    vCPUs1	   MEM	 최대PD수	최대PD크기(TB)	LocalSSD   Network Egress 대역폭(Gbps)
n1-standard-1	  1	      3.75	   128	    257	          예	          2
n1-standard-2	  2	      7.50	   128	    257	          예	         10
n1-standard-4	  4	      15	     128	    257	          예	         10
n1-standard-8	  8	      30	     128	    257	          예	         16
n1-standard-16	 16	    60	     128	    257	          예	        324
n1-standard-32	 32	    120	     128	    257	          예	        324
n1-standard-64	 64	    240	     128	    257	          예	        324
n1-standard-96	 96	    360	     128	    257	          예	        324



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

gsutil -m cp ./data/dir/coco/*.tfrecord ${STORAGE_BUCKET}/coco
gsutil cp ./data/dir/coco/raw-data/annotations/*.json ${STORAGE_BUCKET}/coco

```

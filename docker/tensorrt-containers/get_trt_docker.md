# Running TensorRT


Before you can run an NGC deep learning framework container, your Docker environment must support NVIDIA GPUs. To run a container, issue the appropriate command as explained in the [Running A Container](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#runcont) chapter in the NVIDIA Containers And Frameworks User Guide and specify the registry, repository, and tags.

On a system with GPU support for NGC containers, the following occurs when running a container:
The Docker engine loads the image into a container which runs the software.
You define the runtime resources of the container by including additional flags and settings that are used with the command. These flags and settings are described in Running A Container.
The GPUs are explicitly defined for the Docker container (defaults to all GPUs, but can be specified using NVIDIA_VISIBLE_DEVICES environment variable). Starting in Docker 19.03, follow the steps as outlined below. For more information, refer to the nvidia-docker documentation here.
The method implemented in your system depends on the DGX OS version installed (for DGX systems), the specific NGC Cloud Image provided by a Cloud Service Provider, or the software that you have installed in preparation for running NGC containers on TITAN PCs, Quadro PCs, or vGPUs.

`docker pull nvcr.io/nvidia/tensorrt:20.03-py3`

f you have Docker 19.03 or later, a typical command to launch the container is:
`docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/tensorrt:<xx.xx>-py<x>`
If you have Docker 19.02 or earlier, a typical command to launch the container is:
`nvidia-docker run -it --rm -v local_dir:container_dir nvcr.io/nvidia/tensorrt:<xx.xx>-py<x>`


---
https://developer.nvidia.com/tensorrt

To install Python sample dependencies, run `/opt/tensorrt/python/python_setup.sh`

To install open source parsers, plugins, and samples, run `/opt/tensorrt/install_opensource.sh`. See <https://github.com/NVIDIA/TensorRT/tree/20.03> for more information.

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use `nvidia-docker run` to start this container; see
   <https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker>.


# NVIDIA Docker

## Installation

### `docker < 19.03`:
<https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)>
<https://nvidia.github.io/nvidia-docker/>

```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -


sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.before_nvidiadocker
sudo apt-get install nvidia-docker2 -y
sudo pkill -SIGHUP dockerd

```

`sudo vim /etc/docker/daemon.json`

```json
{
    // New Options
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    // Old ones
    "data-root": "/mnt/hdc0/docker_data",
    "storage-driver": "overlay2"
}
```

### Upgrade (Deprecated)
```sh
# On debian based distributions: Ubuntu / Debian
sudo apt-get update
#sudo apt-get --only-upgrade install docker-ce nvidia-docker2
sudo apt-get --only-upgrade install nvidia-docker2
sudo systemctl restart docker

# On RPM based distributions: Centos / RHEL / Amazon Linux
sudo yum upgrade -y nvidia-docker2
sudo systemctl restart docker
```

### Usage

nvidia-docker registers a new container runtime to the Docker daemon.
You must select the nvidia runtime when using docker run:

```sh
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi

# All of the following options will continue working
docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
docker run --runtime nvidia nvidia/cuda:10.0-base nvidia-smi
nvidia-docker run nvidia/cuda:10.0-base nvidia-smi
```

If run docker as usual, you will get this warning:
```sh
docker run --rm -it  nvcr.io/nvidia/tensorrt:20.03-py3 

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use 'nvidia-docker run' to start this container; see
   https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker .
```


## docker >= 19.03:
<https://github.com/NVIDIA/nvidia-docker/blob/master/README.md>





---
# Upgrade CUDA Toolkit: 10.2

```sh
=====================
== NVIDIA TensorRT ==
=====================

NVIDIA Release 20.03 (build 10567415)

NVIDIA TensorRT 7.0.0 (c) 2016-2020, NVIDIA CORPORATION.  All rights reserved.
Container image (c) 2020, NVIDIA CORPORATION.  All rights reserved.

https://developer.nvidia.com/tensorrt

To install Python sample dependencies, run /opt/tensorrt/python/python_setup.sh

To install open source parsers, plugins, and samples, run /opt/tensorrt/install_opensource.sh. See https://github.com/NVIDIA/TensorRT/tree/20.03 for more information.

ERROR: This container was built for NVIDIA Driver Release 440.33 or later, but
       version 418.67 was detected and compatibility mode is UNAVAILABLE.

       [[CUDA Driver UNAVAILABLE (cuInit(0) returned 804)]]
```


```sh
# Before Installing, check any process is using nvidia
sudo lsof -n -w /dev/nvidia*

# If exists, kill the process first.


# Install from a local file.
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run

sudo sh cuda_10.2.89_440.33.01_linux.run --silent \
  --override-driver-check

# Install from Pkg Manager
sudo apt-get install nvidia-440 nvidia-modprobe nvidia-settings
```



# Run Docker

The NVIDIA Driver was not detected.
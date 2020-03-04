# GCP Settings

## Keygen

```sh
ssh-keygen -t rsa -C {username}


```



## Anaconda

```sh
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
mkdir $HOME/apps
bash ./Anaconda3-2019.10-Linux-x86_64.sh -b -p $HOME/apps/anaconda3 

echo 'export APPS=$HOME/apps' >> ~/.bashrc
echo 'export PATH=$APPS/anaconda3/bin:$APPS/anaconda3/sbin:$PATH' >> ~/.bashrc



```

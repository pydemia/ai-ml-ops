
#docker build --rm -t pydemia/centos7-locale-zsh -f env-base/dockerfiles/locale_centos7/Dockerfile . 
#docker build --rm -t pydemia/ubuntu1804-locale-zsh -f env-base/dockerfiles/locale_ubuntu1804/Dockerfile .

docker build --rm -t pydemia/jupyter-centos7 -f env-jupyter/dockerfiles/jupyter_centos7/Dockerfile . 
docker build --rm -t pydemia/jupyter-ubuntu1804 -f env-jupyter/dockerfiles/jupyter_ubuntu1804/Dockerfile . 
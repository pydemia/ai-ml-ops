FROM pydemia/ubuntu1804-locale-zsh
MAINTAINER Youngju Jaden Kim <pydemia@gmail.com>

USER root
RUN apt-get update && apt-get install texlive-xetex -y

USER pydemia
WORKDIR /home/pydemia
RUN mkdir /home/pydemia/apps

COPY --chown=pydemia:pydemia env-jupyter/install_miniconda.sh .
COPY --chown=pydemia:pydemia env-jupyter/miniconda.bashrc .
COPY --chown=pydemia:pydemia env-jupyter/miniconda.zshrc .

COPY --chown=pydemia:pydemia env-jupyter ./env-jupyter

RUN bash ./install_miniconda.sh
RUN bash ./env-jupyter/install_jupyter.sh

USER root
#RUN cp /home/pydemia/env-jupyter/jupyter.service /etc/systemd/system/
#RUN systemctl daemon-reload
RUN cp /home/pydemia/env-jupyter/init_jupyter /usr/local/bin/init_jupyter
RUN chmod 755 /usr/local/bin/init_jupyter
RUN cp /home/pydemia/env-jupyter/autostart /usr/local/bin/autostart
RUN chmod 755 /usr/local/bin/autostart

USER pydemia
RUN rm -f ./install_miniconda.sh ./miniconda.bashrc ./miniconda.zshrc
RUN rm -rf ./env-jupyter 

CMD autostart


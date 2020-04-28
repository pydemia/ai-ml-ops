FROM ubuntu:18.04
MAINTAINER Youngju Jaden Kim <pydemia@gmail.com>

#VOLUME ["/sys/fs/cgroup"]
#ENTRYPOINT ["/usr/sbin/init"]

USER root
COPY env-base/install_base.sh .
RUN ./install_base.sh LOCALE=ko_KR
RUN rm -f ./install_bash.sh

ENV LANG=ko_KR.UTF-8
ENV LANGUAGE=ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8

RUN echo -e "/bin/zsh\n/usr/bin/zsh\n" >> /etc/shells
RUN sed -i 's/^auth       required   pam_shells.so/auth       sufficient   pam_shells.so/' /etc/pam.d/chsh
COPY env-base/install_zsh.sh .
RUN ./install_zsh.sh
RUN rm -f ./install_zsh.sh

COPY env-base/install_vim.sh .
RUN ./install_vim.sh
RUN rm -f ./install_vim.sh

USER pydemia
WORKDIR /home/pydemia
COPY --chown=pydemia:pydemia env-base/install_zsh.sh .
RUN bash ./install_zsh.sh
RUN rm -f ./install_zsh.sh

RUN chown -R pydemia:pydemia /home/pydemia/.oh-my-zsh

COPY --chown=pydemia:pydemia env-base/install_vim.sh /home/pydemia/install_vim.sh
RUN bash ./install_vim.sh
RUN rm -f ./install_vim.sh

ENV PYTHONENCODING=UTF-8
ENV LANG=ko_KR.UTF-8
ENV LANGUAGE=ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8

USER root
RUN sed -i 's/^auth       sufficient   pam_shells.so/auth       required   pam_shells.so/' /etc/pam.d/chsh


USER pydemia
WORKDIR /home/pydemia
ENV ZSH_THEME="cobalt2-pydemia"
CMD ["zsh"]

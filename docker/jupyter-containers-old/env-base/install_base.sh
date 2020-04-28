#!/bin/bash

# Get Argument
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            LOCALE)              LOCALE=${VALUE} ;;
            *)   
    esac    
done


# Get Linux Distribution
DIST_NAME=`awk -F= '/^ID=/{print $2}' /etc/*-release | sed 's/\"//g'`
echo "OS Distro: $DIST_NAME"

case "$DIST_NAME" in
  "ubuntu") pkgmgr="apt-get" ;;
  "centos") pkgmgr="yum" ;;
esac

# Install Dev Tools
if [ "$DIST_NAME" = "ubuntu" ]; then
  sed -i 's/^override*/#&/' /etc/yum.conf
fi

$pkgmgr update -y
$pkgmgr install vim zsh curl wget git man -y
if [ "$DIST_NAME" = "ubuntu" ]; then
  $pkgmgr install build-essential -y
  $pkgmgr install python3 -y
elif [ "$DIST_NAME" = "centos" ]; then
  $pkgmgr group mark install "Development Tools"
  $pkgmgr group update "Development Tools"
  $pkgmgr groupinstall -y 'development tools'
  $pkgmgr install sudo -y
  $pkgmgr install python36u python36u-libs python36u-devel python36u-pip -y
fi

$pkgmgr update -y

# Set Locales
# LOCALE=ko_KR
if [ "$DIST_NAME" = "ubuntu" ]; then
  $pkgmgr install locales -y
  echo "LANG=$LOCALE.UTF-8"
  locale-gen $LOCALE.UTF-8
  update-locale LANG=$LOCALE.UTF-8 LC_ALL=$LOCALE.UTF-8; echo "$(locale)"
  #export LANG=$LOCALE.UTF-8; echo "$(locale)"
  #export LC_ALL=$LOCALE.UTF-8; echo $LC_ALL


  ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
elif [ "$DIST_NAME" = "centos" ]; then
  sed -i 's/^override*/#&/' /etc/yum.conf
  echo "LANG=$LOCALE.utf8" > /etc/locale.conf
  $pkgmgr reinstall glibc-common glibc -y

  localedef -f UTF-8 -i $LOCALE $LOCALE.utf8
  echo "RUN LANG"
  export LANG=$LOCALE.utf8; echo "$(locale)"
  export LC_ALL=$LOCALE.utf8; echo $LC_ALL

  ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

fi

# Install sudo
$pkgmgr update -y
$pkgmgr install sudo -y

# Install JDK
if [ "$DIST_NAME" = "ubuntu" ]; then
  $pkgmgr install openjdk-8-jdk -y --fix-missing
elif [ "$DIST_NAME" = "centos" ]; then
  $pkgmgr install java-1.8.0-openjdk java-1.8.0-openjdk-devel -y
fi

echo "$(which java)"

# # Set Default User: pydemia
USERNAME="pydemia"
echo "Set Default User: $USERNAME"

if [ "$DIST_NAME" = "ubuntu" ]; then
  adduser --quiet --disabled-password $USERNAME \
  && echo "$USERNAME:ubuntu" | chpasswd
  #usermod -aG sudo $USERNAME
elif [ "$DIST_NAME" = "centos" ]; then
  adduser $USERNAME --password "centos"
  #usermod -aG wheel $USERNAME # wheel: sudo privileges.
  #echo "pydemia ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/pydemia && \
  #chmod -R 0440 /etc/sudoers.d
fi

JAVA_HOME=`java -XshowSettings:properties -version 2>&1 > /dev/null | grep -E "java.home = ([^ ]*)$"|awk '{print $3}' |sed -n 's/\/jre$//p'`
if [ "$DIST_NAME" = "ubuntu" ]; then
  echo "export JAVA_HOME=\"$JAVA_HOME\"" >> /etc/bash.bashrc
elif [ "$DIST_NAME" = "centos" ]; then
  echo "export JAVA_HOME=\"$JAVA_HOME\"" >> /etc/bashrc
fi

echo "Setting Finished."

#!/bin/bash

# Install Julia
curl -s \
https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.1-linux-x86_64.tar.gz \
 | tar zxf - -C /home/pydemia/apps/

cd /home/pydemia/apps && ln -s $(ls -t|grep ^julia) julia

echo 'export PATH="$APPS_PATH/julia/bin:$PATH"' >> ~/.bashrc 
echo 'export PATH="$APPS_PATH/julia/bin:$PATH"' >> ~/.zshrc


# Install Scala
curl -s \
https://downloads.lightbend.com/scala/2.12.8/scala-2.12.8.tgz \
 | tar zxf - -C /home/pydemia/apps/

cd /home/pydemia/apps && ln -s $(ls -t|grep ^scala) scala

echo 'export PATH="$APPS_PATH/scala/bin:$PATH"' >> ~/.bashrc 
echo 'export PATH="$APPS_PATH/scala/bin:$PATH"' >> ~/.zshrc

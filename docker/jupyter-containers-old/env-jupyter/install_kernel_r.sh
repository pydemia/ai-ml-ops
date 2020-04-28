#!/bin/bash

# Get Argument
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            #--kernel_name)      KERNEL_NAME=${VALUE} ;;
            --display_name)     DISPLAY_NAME=${VALUE} ;;
            #--python_version)   PYTHON_VERSION=${VALUE} ;;
            --r_version)        R_VERSION=${VALUE} ;;
            *)   
    esac    
done

# libreadline-dev libX11-dev libXt-dev
# readline-devel libX11 libX11-devel libXt libXt-devel

# bash -i -c "conda create -n $KERNEL_NAME ipykernel -y"
# bash -i -c "conda activate $KERNEL_NAME \
#     && python -m ipykernel install --user \
#        --name $KERNEL_NAME \
#        --display-name '$DISPLAY_NAME' \
#     && conda install -c r r -y \
#     && conda deactivate"

conda install -c r r-essentials -y

if [[ ! -z ${DISPLAY_NAME+x} ]]; then
  R_KERNEL_DIR=`echo "$(jupyter kernelspec list)" | grep -E "ir[[:blank:]]*([^ ]*$)" |awk '{print $2}'` \
  && sed -i "s/\"display_name\":\"R\",/\"display_name\":\"$DISPLAY_NAME\",/" $R_KERNEL_DIR/kernel.json;
else
  R_KERNEL_DIR=`echo "$(jupyter kernelspec list)" | grep -E "ir[[:blank:]]*([^ ]*$)" |awk '{print $2}'` \
  && sed -i "s/\"display_name\":\"R\",/\"display_name\":\"R (conda)\",/" $R_KERNEL_DIR/kernel.json;
fi

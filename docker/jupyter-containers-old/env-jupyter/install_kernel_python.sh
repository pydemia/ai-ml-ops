#!/bin/bash

# Get Argument
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            --kernel_name)      KERNEL_NAME=${VALUE} ;;
            --display_name)     DISPLAY_NAME=${VALUE} ;;
            --python_version)   PYTHON_VERSION=${VALUE} ;;
            *)   
    esac    
done

bash --rcfile /home/pydemia/.bashrc \
-i -c "conda create -n $KERNEL_NAME python=$PYTHON_VERSION ipykernel -y"
bash --rcfile /home/pydemia/.bashrc \
-i -c "conda activate $KERNEL_NAME \
    && python -m ipykernel install --user \
       --name $KERNEL_NAME \
       --display-name $DISPLAY_NAME \
    && conda deactivate"

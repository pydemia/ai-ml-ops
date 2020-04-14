#!/bin/bash

# Get Argument
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            --kernel_name)      KERNEL_NAME=${VALUE} ;;
            --display_name)     DISPLAY_NAME=${VALUE} ;;
            #--python_version)   PYTHON_VERSION=${VALUE} ;;
            --scala_version)    SCALA_VERSION=${VALUE} ;;
            --almond_version)   ALMOND_VERSION=${VALUE} ;;
            *)   
    esac    
done
SCALA_VERSION_FOR_ALMOND=$SCALA_VERSION

bash -i -c "curl -Lo coursier https://git.io/coursier-cli \
&& chmod +x coursier \
&& ./coursier bootstrap \
    -r jitpack \
    -i user -I user:sh.almond:scala-kernel-api_$SCALA_VERSION_FOR_ALMOND:$ALMOND_VERSION \
    sh.almond:scala-kernel_$SCALA_VERSION_FOR_ALMOND:$ALMOND_VERSION \
    --sources --default=true \
    -o almond \
&& ./almond --install --id $KERNEL_NAME --display-name \"$DISPLAY_NAME\""

rm -f ./almond ./coursier

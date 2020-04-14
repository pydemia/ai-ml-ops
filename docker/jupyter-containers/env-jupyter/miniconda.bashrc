
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/pydemia/apps/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/pydemia/apps/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/pydemia/apps/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/pydemia/apps/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


export APPS_PATH="/home/pydemia/apps"
export PATH="$APPS_PATH/miniconda3/bin:$PATH"
export CONDA_EXEC_PATH="$APP_PATH/miniconda3/bin/conda"

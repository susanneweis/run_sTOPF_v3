#!/bin/bash

# set environment name
env_name="hormone_env"

# activate virtual environment
source /home/sweis/.venvs/$env_name/bin/activate

# run the Python script
python run_sTOPF_v3.py "$1" "$2" "$3" "$4"

# deactivate environment
deactivate


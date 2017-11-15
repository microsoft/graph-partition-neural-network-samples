#!/bin/bash

# Download DIEL data 
wget -qO- http://cs.cmu.edu/%7Ezhiliny/data/diel_data.tar.gz | tar xvz -C ./data/
mv ./data/diel_data/diel ./data/diel
rm -rf ./data/diel_data

# Preprocess DIEL data 
PYTHONPATH=$PWD python gpnn/utils/preprocess_diel.py -d ./data/diel

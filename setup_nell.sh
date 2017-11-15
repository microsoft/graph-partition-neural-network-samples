#!/bin/bash

# Download NELL data 
wget -qO- http://www.cs.cmu.edu/%7Ezhiliny/data/nell_data.tar.gz | tar xvz -C ./data/
mv ./data/nell_data ./data/nell

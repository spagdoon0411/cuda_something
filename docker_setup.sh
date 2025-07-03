#!/bin/bash

export CUDACXX="/usr/local/cuda/bin/nvcc"
echo "export CUDACXX=${CUDACXX}" >> ~/.bashrc

bash scripts/install_gtest.sh

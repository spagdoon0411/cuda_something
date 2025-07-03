#!/bin/bash

sudo apt-get install -y libgtest-dev cmake && \
sudo mkdir -p $HOME/build/googletest && \
cd $HOME/build/googletest && \
sudo cmake /usr/src/googletest/googletest && \
sudo make && \
sudo cp lib/libgtest* /usr/lib/ && \
cd $HOME && \
sudo rm -rf $HOME/build/googletest && \
sudo mkdir -p /usr/local/lib/googletest && \
sudo ln -s /usr/lib/libgtest.a /usr/local/lib/googletest/libgtest.a && \
sudo ln -s /usr/lib/libgtest_main.a /usr/local/lib/googletest/libgtest_main.a


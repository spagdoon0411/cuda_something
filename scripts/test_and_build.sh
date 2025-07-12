#!/bin/bash

git pull
cmake --build build && cd build && ctest && cd ../ 

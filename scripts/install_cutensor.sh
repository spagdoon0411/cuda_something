cd $HOME/build
mkdir -p cutensor && cd cutensor
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcutensor2 libcutensor-dev libcutensor-doc

# Linux - Initial Setup

### Install Intel® Distribution of OpenVINO™ Toolkit
```sh
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16612/l_openvino_toolkit_p_2020.2.120.tgz
tar -xvf l_openvino_toolkit_p_2020.2.120.tgz
cd l_openvino_toolkit_p_2020.2.120
sed -i 's/decline/accept/g' silent.cfg
sudo ./install.sh -s silent.cfg
```
Refer to [this page](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux) for more information about how to install and setup the Intel® Distribution of OpenVINO™ Toolkit.

### Install Nodejs and its dependencies
```
sudo apt-get update
sudo apt-get install nodejs
sudo apt update
sudo apt-get install python3-pip
pip3 install numpy
pip3 install paho-mqtt
sudo apt install libzmq3-dev libkrb5-dev
sudo apt install ffmpeg
sudo apt-get install cmake
```

If you’re prompted to upgrade pip, do not update.

### Install NPM

Follow the instructions in the main README file under "Install npm"

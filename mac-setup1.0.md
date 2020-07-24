# Mac - Initial Setup

### Install Intel® Distribution of OpenVINO™ Toolkit
Steps install OpenVINO Toolkit 2020 on Mac OS:
	* Update Mac OS for Updates and Security Patches
	* Download .dmg file of OpenVINO Toolkit 2020 and 
	* Follow on-screen instructions to complete installation

Refer to [this page](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html) for more information about how to install and setup the Intel® Distribution of OpenVINO™ Toolkit for MacOS.

### Install Nodejs and its dependencies

	* Navigate to the [Node.js download page](https://nodejs.org/en/download/) and install the MacOS version. This should also install `npm` along with `node`. 
	* Verify installation in a terminal with `node -v` and `npm -v` (it should show the installed version).

### Install the following dependencies
	* Install [Python 3.7 or higher](https://www.python.org/downloads/). 
	* Run the following from the terminal:

```
pip3 install numpy
pip3 install paho-mqtt
brew install cmake
brew install zeromq
```

	* FFmpeg Installation

```
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
git checkout 2ca65fc7b74444edd51d5803a2c1e05a801a6023
./configure
make -j4
```

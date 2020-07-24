# People Counter Application:
A People Counter App is optimized to run with minimal latency on the Edge Devices. The model used is optimized with Intel 
[Open Vino Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) , Intel version of OpenVINO Toolkit.
The codebase given inside the Repository also demonstrates how AI Model Inference takes place with a Video and needed parameters for a single picture
or Video Streams from Cameras.

## Explaining Custom Layers:
In understanding WorkFlow of how OpenVino Toolkit converts AI Models into Intermediate Representations, we'll see working of
each Layer in Input Model.Firstly, a Model Optimizer searches the list of known Layers for 
each layer in Input Model. The Inference Engine, then, loads the Layers from Input Model's IR files into the specified Device Plugins, 
which further searches all known Layer implementations for the device. If the model architecture contains layer or layers that are not in the 
list of known layers, the Inference Engine considers that/them be unsupported and alerts for errors. To know supported layers, refer
[Supported Devices documentation](https://docs.openvinotoolkit.org/2019_R1.1/_docs_IE_DG_supported_plugins_Supported_Devices.html) and 
to explor [Intermediate Representations](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model.html)


### Custom Layers Implementation:
While implementing a custom layer for selected pre-trained AI Model in the
Open Vino toolkit, need to add extensions to both the Model Optimizer and the Inference Engine. The extensions are discussed in this Writeup. 

#### Model Optimizer:
The Model Optimizer,firstly, extracts information from the Input Model which includes the topology of Model Layers along with necessary parameters, 
input/s and output format, sizes, etc., for each layer.Then, the Model is optimized with various known characteristics of the layers, interconnects, 
and Data-FLow which partly comes from the layer operations providing details including the shape of the output for each layer. Finally, 
the optimized model is directed to Model IR files needed by the Inference Engine to run it( the Model).
The figure below,shows basic processing steps for the Model Optimizer enlightening two necessary custom layer components, namely
the Custom Layer Extractor and the Custom Layer Operation.

![](images/MO_extensions_flow.png)
<details>
  <summary>Source</summary>
  https://docs.openvinotoolkit.org/
</details>

Here are further details on two Custom Layer components:

1. Custom Layer Extractor:
  It is responsible for identifying custom layer operations and extracting the parameters for each instance of the custom layer. 
  The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. 
  Typically the input layer parameters are unchanged, which is the case covered by this Writeup.

2. Custom Layer Operation:
   This is responsible for specifying attributes that are supported by the Custom Layer and computing the output shape for each instance of the custom layer from its parameters. The '--mo-op' command-line argument, shown in the examples below generates a custom layer 
   operation for the Model Optimizer.

#### Inference Engine :
The figure below,shows the basic flow for Inference Engine enlightening two custom layer extensions for CPU and its Plugins and 
Custom Layer CPU extension and its GPU Extension.

![](images/IE_extensions_flow.png)

<details>
  <summary>Source</summary>
  https://docs.openvinotoolkit.org/
</details>

Each device Plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute 
a custom layer. Here are further details on two Custom Layer Extensions:

1. Custom Layer CPU Extension
   This is a compiled shared Library ('.so' or '.dll' binary) that is required by the CPU Plugin for executing custom layer/operations on the CPU.

2. Custom Layer Kernel Extension
   This is OpenCL source code ('.cl') for the custom layer kernel that is compiled to execute on the GPU along with a layer description 
   file ('.xml') needed by the GPU Plugin for the custom layer kernel.

#### Using Model Extension Generator:
Model Extension Generator tool generates template source code files for each of the extensions needed by the Model Optimizer and 
Inference Engine. Its script is available here-  '/opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py'
The necessary details are given below:

```
Arguments to configure extension generation in the interactive mode:
optional arguments:
  -h, --help            show this help message and exit
  --mo-caffe-ext        generate a Model Optimizer Caffe* extractor
  --mo-mxnet-ext        generate a Model Optimizer MXNet* extractor
  --mo-tf-ext           generate a Model Optimizer TensorFlow* extractor
  --mo-op               generate a Model Optimizer operation
  --ie-cpu-ext          generate an Inference Engine CPU extension
  --ie-gpu-ext          generate an Inference Engine GPU extension
  --output_dir OUTPUT_DIR
                        set an output directory. If not specified, the current
                        directory is used by default.
```

### Reasons for handling Custom Layers:
The Custom Layers Handling is required to convert custom layers as an organization/individual need to develop something new or 
to research on new innovations and the application requires to work smoothly supporting custom layers.
Another common use case would be use of 'Lambda' layers, which(layers) are where arbitrary piece of codes are added to model 
implementation.

## Comparing Model Performance:
I tested different AI Models which are not pre-optimized and ended up using one of the models from Intel OpenVino Model Zoo 
, due to poor performance of other converted models.In this exercise, I focused on Model Accuracy and Inference time and Matrices included. I have also stated all models I experimented and tested with necessary arguments/parameters. For further details, see the well researched summary below:
	

### Model size
| |SSD MobileNet V2|SSD Inception V2|SSD Coco MobileNet V1|
|-|-|-|-|
|Before Conversion|67 MB|98 MB|28 MB|
|After Conversion |65 MB|96 MB|26 MB|

### Inference Time

| |SSD MobileNet V2|SSD Inception V2|SSD Coco MobileNet V1|
|-|-|-|-|
|Before Conversion|53 ms|150 ms|56 ms|
|After Conversion |59 ms|153 ms|61 ms|

### Assess Model Use Cases :
  Based on features and supports available with the selected AI Model, I consider following areas where People Counter Application can be used - with or without''
  necessary customizations:
  
1. Visitors Counter and People Traffic Counter Solutions
  With its reliable Counter feature, following can deploy for this Application for any Visitors counting requirements
  * Retails and Shopping Malls
  * Schools and Colleges
  * Movie Halls and other Business Establishment 
  And, there are many successful Use Cases with similar AI Model for enterprises, i.e.  SafeCount or Traffic 2.0 
  
2. Passenger Counter Solutions
	People Counter Apps potentially can be used for Passenger counting requirements
  * Railways Terminals and Booking Centers
  * Airports and Bus Terminals
  
3. Crowd Counter Solutions:
    It can also be used in monitoring Crowds gathered in different events including
  * Music Concerts
  * Social Gathering
  * Social Network Lives
  
4. Fashion and Lifestyle
	People COunter App with needed customization can be used for different real-time recommendation such announcement, delivering past-purchase details
   * Fashion Houses
   * Fashion Shows , etc.   
   
5. Physical Security System
   Utilizing duration of person per frame, it can be used in designing/erecting Physical Security at
   * Data Centers and SOC
   * Parking Areas
   * Street Shopping Zones
   * Intersections , etc. And, in combination of another AI Model, it can be used for 2-F Authentication OTP for:
      * Data Centers
	  * Research Facilities
	  
6. Business Decision Analytics
	With reliable features of People Counter App, it can be used to design Business Solutions for-
	* Performance Comparisons
	* Optimizing Manpower Deployment
	* Shopping Time Pattern
	* Customers/Visitor Patterns

7. Queue Management Solutions
	Queue Management Solution based on this People Counter App can be designed for
	* Analysis of Customer Interactions
	* Impacts of Queues - leaving without purchases, revisit ,etc.
	* Optimization of Staffs
	* Optimizing ENTRY, Billing and EXIT Processes 

8. Space Management and Positions Optimization
   Realty and other Management can use its Queue Management features to 
   * know optimal Positions of Shops - in Airtports, Bus Terminals, Movie Halls, Street Shops Zones
   * to know Customer Traffics/Shopping
   * to Optimize Delivery - in Food Zone, etc
   
9. Crimes Alerting Systems
	By customizing time of person on frame, People Counter App can be used to alert and warn different crimes and other suspicious activities in
	* Intersections
	* Street Shops
	* Pubs, etc
	
10. Video Forensic Tools 
	This, People Counter App facilitates features to design Forensic Tool by Vendors or to create in-house solution for 
	* Video Analytics Alerts- Alerts from Video streams
    * Anomaly Detection - during Video Forensics 

	
## Assess Effects on End User Needs:

Lighting, AI Model accuracy, and camera focal length and image size have significant effects on a deployed AI Model at Edge Devices.
Below are the potential effects of each of them:

* In case of poor model's accuracy, it may fail dramatically or even completely drop close to zero. 
However, this can be avoided with good hardware that process images from poorly lit regions before passing it to AI model.

* Natural decrease in model accuracy during conversion or other stages may make the model unusable.
  An effective solution to this would be placing the model into validation mode for required time i.e an hour or so. 
  During this time, various devices perform federated learning to give better performance. This can bring up a better performance of the model.

* Distorted inputs from camera due to change in focal length and/or image size will affect the operations because the model
  may fail to sense out input. Secondly, the distorted input may not be detected properly by the model. 
  A known approach to solve this would be to use of augmented images while training models and specifying the threshold values, selected optimization
  in considerations.

## Model Selection/Research:
   To find the suitable AI Model for People Counter App, I started with pre-trained models having People Counter features and those models are:
   * [Model Source](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
   * [SSD Inception V2](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
   * [Person Detection- Retail-0002](https://docs.openvinotoolkit.org/latest/person-detection-retail-0002.html)
   * [SSD Coco Mobilenet v1]( http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)	
   * [Person Detection Retail-0013]( https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)
   * [Pedestrian Detection Adas 002](https://docs.openvinotoolkit.org/2019_R1/_pedestrian_detection_adas_0002_description_pedestrian_detection_adas_0002.html )
  
   The analytical details of those models are: 
- Model 1: SSD Mobilenet
  - [Model Source](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
  - I converted the model to an Intermediate Representation with the following arguments
  
```
python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json

```

  - The model was insufficient for this Application because it wasn't adequate accurate while doing inference. 
  - To improve, tried using some transfer learning techniques and retrained few of the model layers with some additional data but 
    that did not work too well to meet the requirements.
  
- Model 2: SSD Inception V2]
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments
  
```
python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
```
  
  - The model was insufficient for the app because it has pretty high latency in making predictions ~155 ms.
  - I tried to improve the model for the app by reducing the precision of weights, however this had a very huge impact on the accuracy.

- Model 3: SSD Coco MobileNet V1
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments

```
python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json

```
  - The model was suitable for the app because it has a very low Inference Accuracy. It was unable to identify people with their back 
  facing towards the camera making this model unusable.
  
- Model 4: Pedestrian Detetction Adas 002
  - [Model Source](https://docs.openvinotoolkit.org/2019_R1/_pedestrian_detection_adas_0002_description_pedestrian_detection_adas_0002.html )
  - I converted the model to an Intermediate Representation with the following arguments

```
python mo_tf.py --input_model pedestrian_detection_adas_0002.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json

```

## The Model:
To over come challenges mentioned identified or collected during AI Model Research Analysis, I decided to use an AI Model/s written below for People Counter App. They available at 
- [person-detection-retail-0002](https://docs.openvinotoolkit.org/latest/person-detection-retail-0002.html)  and
- [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)

These models are in fact based on the MobileNet model, the MobileNet model performed addressing issues of latency and size apart of few 
inference errors. The latest version has fixed up those errors.

To get higher overall accuracy and low latency, considered [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) 
for People Counter App.

### Downloading Model:

Download all the pre-requisite libraries and performed source the OpenVINO installation using the following commands:

``` sh
pip install requests pyyaml -t /usr/local/lib/python3.5/dist-packages && clear && 
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

```

Navigate to the directory containing the Model Downloader:

``` sh
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/Downloader
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5 -- initializing env

```

Here, firstly, I displayed the available models using

``` sh
sudo ./downloader.py --print_all

```
And, performed optimized download using other arguments '-h' and '--name' and precisions as shown below

``` sh
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

```


## Performing Inference :
Login to Udacity Classroom, load Workspace for Project 1, open a Terminal( File-> New-Terminal) and then execute the following commands, in the 
given sequence

Installing Npm Package:

``` sh
  cd webservice/server
  npm install
  
```

```
  cd node-server
  npm install mosca
  
```

After installing npm and mosca, execute commands below to start Mosca Server :

``` sh
  cd node-server
  node ./server.js
  
```

On successful startup, you should receive a message-

```sh
Mosca Server Started and Terminal became ready to display - People Count/Frame, Time Spent and Total People Count.

```

Now, open second Terminal and execute following commands 

``` sh
cd webservice/ui
npm install

```
After installing npm for UI, execute the commands below to start UI Server:
``` sh
npm run dev

```
On successful startup, you should receive a message-
``` sh
webpack: Compiled successfully

```
And then, open third terminal and execute FFMeg Server:

``` sh
sudo ffserver -f ./ffmpeg/server.conf
```
Finally, open the fourth Terminal and execute main.py with necessary arguments , as shown below to use testing video from 'resource/' folder and
it uses port number '3004'.

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

```

To access the Web Application from Workspace, click on Open App , if does not opened automatically.

### People Counter App Mechanism:
Successful running of this Apps generates three sets of Data- Statistics to MQTT Server, Images Frames to FFmeg Server and UI Server logs.
The main.py() sends statistics- total people count, frame-wise People count and durations to MQTT Server as JSON Data and frames of given  
Video File, Video Streams or picture file to FFMeg Server and which are displayed through Web- Application with Box indicating People on input
file. On running this App for with sample Video, I/we received following Data Communications:

Mosca/Node Server Logs:
![](images/MoscaServer.png)

UI Server Logs :
![](images/UIServerLog.png)

#####
Prepared by: Amrit Chhetri, Copyright(C): MIT/GNU License , July 23,2020 |||




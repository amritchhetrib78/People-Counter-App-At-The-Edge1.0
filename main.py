"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
 Parameters used to Test this Main.py:
 python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

 
 
"""
import time
import socket
import json
import cv2
import os
import sys
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60



def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser



def connect_mqtt():
    ### TODO: Connect to the MQTT client ### Referred from Foundation class
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client



def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialize the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    MODEL_INPUT = args.model
    
    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(MODEL_INPUT, CPU_EXTENSION, DEVICE)
    network_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # Video Streams Check
	 input_type = args.input
    if input_type == 'CAM':
        inputflag = 0

    # Image Input Check
    elif input_type.endswith('.jpg') or input_type.endswith('.bmp') or input_type.endswith('.gif') :
        single_image_mode = True
        inputflag = input_type

    # Video File Check
    else:
        inputflag = input_type
        assert os.path.isfile(input_type), "Input File doesn't exist or Invalid"

    ### TODO: Handle the input stream ### Opening Input using OpenCV
    cap = cv2.VideoCapture(inputflag)
    cap.open(inputflag)

    widthX = int(cap.get(3))
    heightX = int(cap.get(4))

    in_shape = network_shape['image_tensor']
        
    ### TODO: Loop until stream is over ###
	## Variable initialization
	duration_prev = 0.0
    total_count = 0
    duration_check = 0
    request_id=0
    current_count = 0
    counter = 0
    counter_prev = 0
	
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frmX = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frmX, (in_shape[3], in_shape[2]))
        image_tp = image.transpose((2, 0, 1))
        image_tp = image_tp.reshape(1, *image_tp.shape)
		
		int_start = time.time()
  
        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': image_tp,'image_info': image_tp.shape[1:]}
        duration = 0.0
        infer_network.exec_net(net_input, request_id)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
			flag = 0
			det_time = time.time() - int_start

            ### TODO: Get the results of the inference request ###
            net_output = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
			         
            
            probs = net_output[0, 0, :, 2]
            for ax, p in enumerate(probs):
                if p > prob_threshold:
                    flag += 1
                    box = net_output[0, 0, ax, 3:]
                    p1 = (int(box[0] * widthX), int(box[1] * heightX))
                    p2 = (int(box[2] * widthX), int(box[3] * heightX))
                    frmX = cv2.rectangle(frmX, p1, p2, (0, 255, 0), 3)
                    # Count Time Here
					start_time = time.time()
                    duration = duration_prev+ det_time
			### Second time onwards
            if flag != counter:
                counter_prev = counter
                counter = flag
                if duration_check >= 3:
                    duration_prev = duration_check
                    duration_check = 0
                else:
                    duration_check = duration_prev + duration_check
                    duration_prev = 0  # unknown, not needed in this case
            else:
                duration_check += 1
                if duration_check >= 3:
                    current_count = counter
                    if duration_check == 3 and counter > counter_prev:
                        total_count += counter - counter_prev
                    elif duration_check == 3 and counter < counter_prev:
                        duration = int((duration_prev / 10.0) * 1000)                       

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish('Person',
                           payload=json.dumps({
                               'count': current_count, 'total': total_count}),
                           qos=0, retain=False)
            
            #if duration is not None:
            if duration > 0 :
                client.publish('person/duration',
                               payload=json.dumps({'duration': duration}),
                               qos=0, retain=False)
 

        ### TODO: Send the frame to the FFMPEG server ###
        #  Resize the frame
        frmX = cv2.resize(frmX, (768, 432))
        sys.stdout.buffer.write(frmX)
        sys.stdout.flush()

    cap.release()
    cv2.destroyAllWindows()



def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()

import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import datetime
from google.colab.patches import cv2_imshow



# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help=' model location -> tflite file is located in', required=True)
parser.add_argument('--graph', help=' .tflite file,', default='detect.tflite')
parser.add_argument('--labels', help='labelmap file, ', default='classes.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.6)
parser.add_argument('--resolution', help='Webcam resolution', default='1280x720')
parser.add_argument('--edgetpu', help='EDGE TPU', action='store_true')
parser.add_argument('--output_path', help="processed images save directory.", required=True)
args = parser.parse_args()


args = parser.parse_args()

# TensorFlow and model initialization
pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if args.edgetpu:
        from tflite_runtime.interpreter import load_delegate

from tflite_runtime.interpreter import Interpreter

if args.edgetpu:
    args.graph = 'edgetpu.tflite'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, args.modeldir, args.graph)
PATH_TO_LABELS = os.path.join(CWD_PATH, args.modeldir, args.labels)
path = '/content/Embedded_Systms_Scope/model/model_iter2/labelmap.txt'
# Load labels
with open(path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

# Load model
interpreter = Interpreter(model_path="/content/Embedded_Systms_Scope/model/model_iter2/detect.tflite")
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5


# Load image and process it
image = cv2.imread(args.image_path)
frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (width, height))
input_data = np.expand_dims(frame_resized, axis=0)

if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get results
boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class indices
scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence values


#distance estimation
f_mm = 3.6  # focal length in mm
sensor_width_mm = 7.2  # sensor width in mm
image_width_pixels = 1280  # image width in pixels from the resolution

# Convert focal length from mm to pixels
f_pixels = f_mm * (image_width_pixels / sensor_width_mm)
actual_car_width = 1.8  # in meters

object_ids = 0  # Initialize a counter for car ID


for i in range(len(scores)):
    if (scores[i] > float(args.threshold)) and (scores[i] <= 1.0):
        
        #if labels[int(classes[i])] == 'car':  # Check if detected class is 'vehicle'
            object_id = labels[int(classes[i])] 
            object_ids += 1  # Increment the car ID
            print(f"{object_id} ID {object_ids} detected with confidence {scores[i]:.2f}")

            # Get bounding box coordinates and draw box
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * image.shape[1])
            xmax = int(xmax * image.shape[1])
            ymin = int(ymin * image.shape[0])
            ymax = int(ymax * image.shape[0])
            width_pixels = xmax - xmin
           
            distance_meters = (f_pixels * actual_car_width) / width_pixels
            
            color = (10, 255, 0) #green


            if distance_meters < 4.0:
                print(f"Warning: {object_id} {object_ids} is closer than 4")
                color = (0, 0, 255) #red


            print(f"Estimated distance to {object_id} {object_ids}: {distance_meters:.2f} meters")

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            # Object label
            object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
            
            object_label = f'Object ID {object_ids}'  # Car ID label
            confidence_label = f'{object_name}: {scores[i]:.2f}'  # Confidence label
            # Distance label
            distance_label = f'Dist: {distance_meters:.2f}m'  # Distance label

            # Calculate text size for both labels and draw label backgrounds
            
            object_label_size, _ = cv2.getTextSize(object_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            confidence_label_size, _ = cv2.getTextSize(confidence_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            distance_label_size, _ = cv2.getTextSize(distance_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Ensure label background is sized for the longest label
            max_label_width = max(object_label_size[0], confidence_label_size[0], distance_label_size[0])
            total_label_height = object_label_size[1] + confidence_label_size[1] + distance_label_size[1] + 15  # 15 pixels for spacing

            cv2.rectangle(image, (xmin, ymin - total_label_height - 10), (xmin + max_label_width, ymin), (255, 255, 255), cv2.FILLED)


            cv2.putText(image, object_label, (xmin, ymin - total_label_height + object_label_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, confidence_label, (xmin, ymin - total_label_height + object_label_size[1] + confidence_label_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, distance_label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            cv2.imwrite('/content/Embedded_Systms_Scope/out.jpg',image) #add confidence in image

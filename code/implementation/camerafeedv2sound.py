import os
import argparse
import cv2
import numpy as np
import sys
from threading import Thread
import importlib.util
import datetime
import RPi.GPIO as GPIO
import time
from sys import argv
import pygame
import collections

# Path to the sound files and the sound file list
path2 = "/home/chikere/Documents/Embedded_Systms_Scope/"
sounds = {
    'a': "car.mp3",
    'b': "pedes.mp3",
    'd': "traffic.mp3",
}

# Pygame mixer initialization
pygame.mixer.init()
speaker_volume = 0.5
pygame.mixer.music.set_volume(speaker_volume)



whichled=argv[1]
ledaction = argv[2]
LEDa=17
LEDb=18
LEDc=22
LEDd=23
GPIO.setmode(GPIO.BCM)
GPIO.setup(LEDa, GPIO.OUT)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LEDb, GPIO.OUT)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LEDc, GPIO.OUT)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LEDd, GPIO.OUT)

GPIO.output(LEDc, False)


# Initialize counters for detected objects
detection_counters = collections.defaultdict(int)


# Define VideoStream class to handle streaming of video from webcam in separate processing thread - saves on computing
class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

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

resW,resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
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
path = 'model_iter2/classes.txt'
# Load labels
with open(path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

# Load model
interpreter = Interpreter(model_path="model_iter2/detect.tflite")
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]



floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Initialize video stream
videostream = VideoStream(resolution=(int(args.resolution.split('x')[0]), int(args.resolution.split('x')[1])), framerate=30).start()
time.sleep(1)

#distance estimation
f_mm = 50  # focal length in mm - specific to camera so value msut be changed
sensor_width_mm = 5.37 # sensor width in mm
image_width_pixels = 1280  # image width in pixels from the resolution

# Convert focal length from mm to pixels
f_pixels = f_mm * (image_width_pixels / sensor_width_mm)
actual_car_width = 1.8  # in meters
object_ids = 0  # Initialize a counter for object ID




print("Starting detection loop...")


while True:
    frame1 = videostream.read()
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    GPIO.output(LEDc, True) 
    GPIO.output(LEDa, False)
    GPIO.output(LEDb, False)
    GPIO.output(LEDd, False)



    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    ## Get results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class indices
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence values
    for i in range(len(scores)):
        if (scores[i] > float(args.threshold)) and (scores[i] <= 1.0):
            object_id = labels[int(classes[i])] 
            object_ids += 1  # Increment the global object ID
            
            # Increment detection counter for this object type
            detection_counters[object_id] += 1

            print(f"{object_id} ID {object_ids} detected with confidence {scores[i]:.2f}")

            # Get bounding box coordinates and draw box
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])
            width_pixels = xmax - xmin
            distance_meters = (f_pixels * actual_car_width) / width_pixels
                
            color = (10, 255, 0) #green
                
                
                # Check if detection counter has reached the limit, person class issues 
            if detection_counters[object_id] >= 3:
                # Reset counter after playing sound
                detection_counters[object_id] = 0
                
                # Sound and warning logic based on object type
                if object_id in ['person', 'bike']:
                    GPIO.output(LEDb, True)  # Yellow
                    if distance_meters < 10.0:
                        print(f"Warning: {object_id} {object_ids} is closer than 2 meters")
                        pygame.mixer.music.load(path2 + sounds['b'])
                        pygame.mixer.music.play()

                elif object_id in ['traffic sign', 'traffic light']:
                    GPIO.output(LEDd, True) #Blue
                    pygame.mixer.music.load(path2 + sounds['d'])
                    pygame.mixer.music.play()

                else:  # Cars, trucks, etc.
                    GPIO.output(LEDa, True)
                    if distance_meters < 4.0:
                        print(f"Warning: {object_id} {object_ids} is closer than 4 meters")
                        pygame.mixer.music.load(path2 + sounds['a'])
                        pygame.mixer.music.play()
                        color = (255, 0, 0) #red


            print(f"Estimated distance to {object_id} {object_ids}: {distance_meters:.2f} meters") 

            #this can be commented out if not wishing to save images -> just run inference model with PI
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Object label
            object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
            
            object_label = f'Object ID {object_ids}'  # object ID label
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

            cv2.rectangle(frame, (xmin, ymin - total_label_height - 10), (xmin + max_label_width, ymin), (255, 255, 255), cv2.FILLED)


            cv2.putText(frame, object_label, (xmin, ymin - total_label_height + object_label_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, confidence_label, (xmin, ymin - total_label_height + object_label_size[1] + confidence_label_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, distance_label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            #save images into output folder with annotations
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            cv2.imwrite(os.path.join(args.output_path, f"{timestamp}.jpg"),frame) #add results in image

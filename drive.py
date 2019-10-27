
# coding: utf-8

# In[6]:


# drive.py
'''
This "drive" script takes the pre-trained Keras model to predict the steering angles
based on the recorded driving on the lake test track from the Udacity's vehicle simulator,
and sends the throttle, brake and steering angles to the Udacity simulator.
'''

# Import required libraries

import os

# Command line parsing
import argparse

# High level file operation
import shutil

# 64-bit encoding and decoding
import base64

from datetime import datetime

import numpy as np
# JavaScript library for realtime web applications
import socketio

# Concurrent networking library
import eventlet
import eventlet.wsgi

# Python imaging library
from PIL import Image

# Micro web framework
from flask import Flask

# Frameworks for I/O streams
from io import BytesIO

# Interface to the HDF5 binary data format
import h5py

# Keras model
from keras.models import load_model
from keras import __version__ as keras_version

import cv2


# In[4]:


# Show Keras version using
print('Using Keras version: {}'.format(keras_version))


# In[ ]:


def crop(img):
    '''
    Crop out non-road sections of the image - the top (sky) and the bottom (engine cover)
    '''
    return img[60:-25,:,:]

def resize(img,width=200,height=66):
    '''
    Resize the image to the input shape used by the convolution neural network model
    '''
    return cv2.resize(img,(width,height),cv2.INTER_AREA)


# In[5]:


# Create realtime web application
sio = socketio.Server()

# Create micro web framework for the application
app = Flask(__name__)
model = None
prev_image_array = None

# Simple PI controller used for tracking steering angle command
class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral

controller = SimplePIController(0.1, 0.002)
set_speed = 30 # speed in the unit of mph
controller.set_desired(set_speed)

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]

        # The current throttle of the car
        throttle = data["throttle"]

        # The current speed of the car
        speed = data["speed"]

        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)

        # Preprocess the image
        image_array = crop(image_array)
        image_array = resize(image_array)

        # Predict the steering angle based on the image
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        # Set the vehicle speed obtain from the data
        throttle = controller.update(float(speed))

        # Send steering angle and throttle to the simulator
        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

    # Send zero steering angle and throttle
    # upon connection
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engine io's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

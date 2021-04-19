
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import sklearn
import os
import time
import csv
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from math import ceil
import keras.models as md

import utilities
import model

###############################
## 01_Variables ###############

image_h = utilities.image_h     # input height
image_w = utilities.image_w     # inputwidth
image_d = utilities.image_d     # input depht = 3 channels

###############################
## 02_helpful_functions #######
"""
helpful functions and augmentation moved to utilities.py
"""


###############################
## 03_model #########
#import tensorflow.contrib.keras as keras
"""
model moved to model.py, so that autonomous mode with simulator works
"""
  
###############################
## 04_run training #########    


# load driving log (.csv) and add column names to driving log 
columns_name = {0: 'Center', 1: 'Left', 2: 'Right', 3: 'Steering Angle', 4: 'Throttle', 5:'Break', 6:'Speed'}
record_df = utilities.load_drivinglog('my_data/driving_log.csv', columns_name=columns_name)

# builded pipeline for image augmentation
def pipeline_augment(image, steering_angle):
    # random flip horizontally
    image_aug, steering = utilities.flip_augment(image, steering_angle)
    # random shift in x and y direction. shift distance variable = distance
    image_aug, steering = utilities.shift_augment(image, steering_angle, distance=20)
    # random added shadows (by masking and adjusting saturation)
    image_aug = utilities.shadow_augment(image)
    # random added brightness
    image_aug = utilities.brightness_augment(image)
    # noise reduction by gaussian blur 
    image_aug = utilities.gaussian_blur(image)
    return image_aug, steering_angle

# pipeline for loading training/validation data and adding augmentation and preprocessing
def load_training_data(batch):
    """
    in: data frame batch with {0: 'Center', 1: 'Left', 2: 'Right', 3: 'Steering Angle'}
    out: for this batch a list of all images (training data) and steering angles (labels)
    function: 
        1. load images based on the paths in the data frame batch file
        2. adjust steering angle for right or left shifted/recorded images
        3. do image augmentation and preprocessing for the nn model
        4. output all images and steering anles in 2 lists
    """
    
    left_cf = utilities.left_cf     # correction factor for steering angle for left image 
    right_cf = utilities.right_cf   # correction factor for steering angle for right image 
    list_images = []            # empty list for output
    list_steering_angle = []    # empty list for output
    
    for index, row in batch.iterrows():
        # load path for images
        path_left = utilities.get_rel_path(row['Left'])
        path_right = utilities.get_rel_path(row['Right'])
        path_center = utilities.get_rel_path(row['Center'])
        center_angle = float(row['Steering Angle'])
        # load Images
        left_img = utilities.load_image(path_left)
        center_img = utilities.load_image(path_right)
        right_img = utilities.load_image(path_center)
        # For the shifted richt and left images: adjust the steering angle
        lenft_angle = center_angle + left_cf
        right_angle = center_angle - right_cf
        # Augment the Image
        left_img_aug, lenft_angle = pipeline_augment(left_img, lenft_angle)
        center_img_aug, center_angle = pipeline_augment(center_img, center_angle)
        right_img_aug, right_angle = pipeline_augment(right_img, right_angle)
        # Preprocess (Cropping, resizing, transformation in YUV-Colorspace) the augmented images
        left_img_aug_prepro = utilities.preprocess_image(left_img_aug)
        center_img_aug_prepro = utilities.preprocess_image(center_img_aug)
        right_img_aug_prepro = utilities.preprocess_image(right_img_aug)
        # append Images and steering angles to lists for output
        list_images.append(left_img_aug_prepro)
        list_steering_angle.append(lenft_angle)
        list_images.append(center_img_aug_prepro)
        list_steering_angle.append(center_angle)
        list_images.append(right_img_aug_prepro)
        list_steering_angle.append(right_angle)
        
    return list_images, list_steering_angle

batchsize = 32
# building a generator for hand over training data and labels in small batches --> avoiding "out of memory" errors
def batch_generator(data, batchsize=batchsize):
    """
    in: data frame with {0: 'Center', 1: 'Left', 2: 'Right', 3: 'Steering Angle'}
    out: training data (images = X_train) and training labels (steering angles = y_train)
    function: as a generator it returns/yieds the next augmented and preprocessed batch of training/validation data
        1. shuffle the data
        2. load the augmented and preprocessed batch
        3. return it to the model, then yield
    """
    data_length = len(data) # data type = data frame
    while 1:
        sklearn.utils.shuffle(data)
        for count in range(0, data_length, batchsize):
            batch = data[count:count+batchsize]
            images, steering_angles = load_training_data(batch)
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            sklearn.utils.shuffle(X_train, y_train)
            yield X_train, y_train

# splitting the data in training and validation data, using the train_test_split function from sklearn.model_selection
train_samples, valid_samples = train_test_split(record_df, test_size=0.2)

# create the generators for training and validation  
train_generator = batch_generator(train_samples)
valid_generator = batch_generator(valid_samples)

# choosing optimizer: Adam because of adaptive learning rates 
optimizer=Adam(1e-4, decay=0.0)

# choosing batchsize
batchsize = 32
epochs = 60

# loading model
nvidia_model = model.nvidia_model(optimizer)

# adding callbacks: early stopping and saving of best fitted model per epoch
callback_es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=6)
callback_cp = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# start training of the model
net_history = nvidia_model.fit_generator(train_generator, 
                    steps_per_epoch= ceil(len(train_samples)/batchsize),
                    validation_data=valid_generator, 
                    validation_steps= ceil(len(valid_samples)/batchsize),
                    epochs=epochs, callbacks=[callback_es, callback_cp])


import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

###############################
## 01_Variables ###############
image_h = 66    # input height
image_w = 200  # input width
image_d = 3     # input depht = 3 channels

left_cf = 0.2 # correction factor for adjusting steering angle for left image 
right_cf = 0.2 # correction factor for adjusting steering angle for right image 


###############################
## 02_helpful_functions #######

def load_image(image_path):
    """
    in: path of the image
    out: image in RGB
    """
    path = Path(image_path)
    if path.is_file() and path.exists():
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    else:
        image = None
    return image

def preprocess_image(img):
    """
    in: image
    out: preprocessed image
    function: 
        1: cropping image to reduce not needed information
        2: resizing image to target width/heith
        3: the nvidia article uses YUV bevause it separates the brigthness 
        information (y) which is less important, from the color information 
        (UV) which is more important
      
    """
    #img_shape = img.shape
    image = img[35:-25, :, :] #crop out top and bottom 20px
    image_resized = cv2.resize(image, (image_w, image_h), cv2.INTER_AREA)
    image_yuv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2YUV)
    return image_yuv


def load_drivinglog(csv_path, columns_name=None):
    """
    in: path of the csv file, named columns
    out: df of csv driving_log
    """
    rel_path = Path(csv_path)  
    if rel_path.is_file() and rel_path.exists():
        if columns_name:
            df_driving_log = pd.read_csv(csv_path, header=None)
            df_driving_log = df_driving_log.rename(columns=columns_name)
        else:
            df_driving_log = pd.read_csv(csv_path)
            
    return df_driving_log 


def get_rel_path(image_path):
    """
    in: path from csv 
    out: relative path from working dic
    """
    image_path = image_path.replace("\\","/")
    rel_path = './' + image_path.split('/')[-3] + '/' + image_path.split('/')[-2] + '/' + image_path.split('/')[-1]
    return rel_path





###############################
## 03_augmentation #######
"""
The NN is only as good as the data we feed --> 
Enlarge the Dataset by using 5 augmentation techniques.
Augmentation helps to increase the amount of relevant data
"""
#from keras.preprocessing.image import ImageDataGenerator

def shift_augment(image, steering_angle, distance=0.1):
    """
    in: image, steering_angle, target distance to shift the image
    out: shifted image, new steering angle
    function: to learn how to follow the rod if the car position is not central,
    I add a random shift in x and y direction. Here for the cv2 function warpAffline is used.
    """
    h, w, d = image.shape
    shift_in_x =(np.random.rand()- 0.5) * distance
    shift_in_y =(np.random.rand()- 0.5) * distance
    shift = np.float32([[1, 0, shift_in_x], [0, 1, shift_in_y]])
    shift_image = cv2.warpAffine(image, shift, (w, h))
    new_steering_angle = steering_angle + shift_in_x * 0.002
    return shift_image, new_steering_angle

#def random_shadow(img):
def shadow_augment(image):
    """
    in: image
    out: new image with random added shadow area
    function: 
        1. random 2 points at height = 0 and max height
        2. create mask with two-point equation
        3. adding shadow by adjusting S-Channel of HLS colorspace
    code source: https://github.com/BerkeleyLearnVerify/VerifAI/blob/master/examples/data_augmentation/model/utils.py
    """
    h, w , d = image.shape
    x1, y1 = w * np.random.rand(), 0
    x2, y2 = w * np.random.rand(), h
    xm, ym = np.mgrid[0:h, 0:w]
    
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.5, high=0.8)
     # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

#def adjust_brightness(img):
def brightness_augment(img, factor=0.5):
    """
    adjusting brighness includes converting zu HSV and scaling the V channel
    """    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv_img = np.array(hsv_img, dtype=np.float64)
    hsv_img[:, :, 2] = hsv_img[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
    hsv_img[:, :, 2][hsv_img[:, :, 2] > 255] = 255 #reset out of range values
    rgb_img = cv2.cvtColor(np.array(hsv_img, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    return rgb_img 
    
    

#def flip(img, steering):
def flip_augment(img, steering_angle):
    """
    flipping randomloy with openCV
    flipcode > 0: flip horizontally --> that to use
    flipcode = 0: flip vertically
    flipcode < 0: flip vertically and horizontally
    """
    a = np.random.rand() # (0-1)
    if a < 0.5:
        img = cv2.flip(img, 1) #flip horizontally
        if(steering_angle != 0):
            steering_angle = -steering_angle #flip steering angle
    return img, steering_angle


def gaussian_blur(image):
    """
    The Gaussian filter is a low-pass filter so that the high-frequency components are reduced.
    could help by different resolution of the training data and road + environment structure
    """
    a = np.random.rand() # (0-1)
    if a < 0.5:
        image = cv2.GaussianBlur(image,(5,5),0)
    return image

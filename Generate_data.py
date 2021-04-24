
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from skimage.util import random_noise, img_as_ubyte
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from skimage import io, transform
import dataset_loader as dl
from scipy.interpolate import UnivariateSpline
import cv2



"""
reference:
https://github.com/mbeyeler/opencv-python-blueprints/blob/master/chapter1/filters.py
"""


#path to dataset
root_dir = 'dataset/train/'

dataset = dl.DrivingDataset(root_dir)

def generate_data(root_dir,categorical_value, trans_function):
     data_X = []
     data_Y = []
     dataset = dl.DrivingDataset(root_dir,categorical=categorical_value,transform=trans_function)
     for i in range (len(dataset)):
          data_X.append(dataset[i]['image'])
          data_Y.append( dataset[i]['cmd'])
     return [data_X, data_Y]




def gauss_noise_1(image,mu = 0, variance=0.9):
    gauss_img = img_as_ubyte(random_noise(image, mode='gaussian', mean=mu, var=variance, clip=True))
    return gauss_img

def gauss_noise_2(image,mu = -0.2, variance=0.3):
    gauss_img = img_as_ubyte(random_noise(image, mode='gaussian', mean=mu, var=variance, clip=True))
    return gauss_img

def gauss_noise_3(image,mu = 0.1, variance=0.004):
    gauss_img = random_noise(image, mode='gaussian', mean=mu, var=variance, clip=True)
    return gauss_img

def create_LUT_8UC1(x, y):
        """Creates a look-up table using scipy's spline interpolation"""
        spl = UnivariateSpline(x, y)
        return spl(range(256))
    

incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30,  80, 120, 192])

def color_filter_warm(image):
        c_r, c_g, c_b = cv2.split(image)
        c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
        image = cv2.merge((c_r, c_g, c_b))
        
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)


def color_filter_cool(image):
        c_r, c_g, c_b = cv2.split(image)
        c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # decrease color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, decr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)


def warm_and_noise3(image, gaussian=gauss_noise_3, color_filter=color_filter_warm):
    filtered_image = color_filter(image)
    gauss_image = gaussian(filtered_image)
    return gauss_image


def cool_and_noise2(image, gaussian=gauss_noise_2, color_filter=color_filter_cool):
    filtered_image = color_filter(image)
    gauss_image = gaussian(filtered_image)
    return gauss_image




        
#Generare original data
#task1= generate_data(root_dir,False, None)

#Add gaussian noise
#task2= generate_data(root_dir,False, gauss_noise_1)
task3= generate_data(root_dir,False, gauss_noise_2)

#apply warming color filter   
task4= generate_data(root_dir,False, color_filter_warm)

task5= generate_data(root_dir,False, color_filter_cool)

#apply both gaussia noise and color filter
task5= generate_data(root_dir,False, warm_and_noise3)
task6= generate_data(root_dir,False, cool_and_noise2)

io.imshow(task1[0][1])
io.show()
io.imshow(task2[0][1])
io.show()
io.imshow(task3[0][1])
io.show()
io.imshow(task4[0][1])
io.show()
io.imshow(task5[0][1])
io.show()

io.imshow(task6[0][1])
io.show()


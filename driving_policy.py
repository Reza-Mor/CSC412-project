import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from scipy.interpolate import UnivariateSpline
import cv2
import os
import re
from skimage.util import random_noise, img_as_ubyte
from matplotlib import pyplot as plt



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

            
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    

class DiscreteDrivingPolicy(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        
        self.features = nn.Sequential(
            #24 conv, relu, 36 conv, relu, 48 conv, relu, 64 conv, relu, flatten
            #kernel/window of size 4, stride of size 2, and 1 pixel padding
            nn.Conv2d(3, 24, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(24, 36, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(36, 48, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(48, 64, 4, 2, 1),
            nn.ReLU(),
            Flatten(),
        )
        
        self.classifier = nn.Sequential(
            # fc 128, relu, fc 64, relu, fc n classes, relu
            nn.Linear(4096,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.n_classes),
            nn.ReLU(),
        )  
        
        self.apply(weights_init)

    def forward(self, x):
        f = self.features(x)
        logits = self.classifier(f)
        return logits
    
    def test(self, state, device):
        """Please manually change the corresponding filter/noise to state
        """
        #plt.imshow(state)
        #plt.show()
        state = state.astype(np.float)
        image=gauss_noise_1(state)
        #image= gauss_noise_2(state)
        #image= color_filter_warm(state.astype(np.uint8))
        #image= color_filter_cool(state.astype(np.uint8))
        state = image.astype(np.float32)
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0)
        logits = self.forward(state)
        
        y_pred = logits.view(-1, self.n_classes) 
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()
        
        steering_cmd = (steering_class / (self.n_classes - 1.0))*2.0 - 1.0
        
        return steering_cmd
    
    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict( {k:v for k,v in weights.items()}, strict=True)



    
def create_LUT_8UC1(x, y):
    """Creates a look-up table using scipy's spline interpolation"""
    spl = UnivariateSpline(x, y)
    return spl(range(256))

incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                              [0, 70, 140, 210, 256])
decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                              [0, 30, 80, 120, 192])

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
    image = cv2.merge((c_r, c_g, c_b))

    # decrease color saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    c_s = cv2.LUT(c_s, decr_ch_lut).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

def gauss_noise_1(image, mu=0.4, variance=0.2):
    gauss_img = img_as_ubyte(random_noise(image, mode='gaussian', mean=mu, var=variance, clip=True))
    return gauss_img

def gauss_noise_2(image, mu=-0.4, variance=0.2):
    gauss_img = img_as_ubyte(random_noise(image, mode='gaussian', mean=mu, var=variance, clip=True))
    return gauss_img


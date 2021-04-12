from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from os.path import isfile, join
from scipy.interpolate import UnivariateSpline
import cv2
import os
import re
from skimage.util import random_noise, img_as_ubyte

class DrivingDataset(Dataset):
    
    def __init__(self, root_dir, categorical = False, classes=-1, transform=None):
        """
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [f for f in listdir(self.root_dir) if f.endswith('jpg')]
        self.categorical = categorical
        self.classes = classes
        cmds = []
        for basename in self.filenames:
            m = re.search('expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg', basename)
            steering_command = float(m.group(3))
            cmds.append(steering_command)
        self.max_cmd = max(cmds)
        self.min_cmd = min(cmds)

    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        basename = self.filenames[idx]
        img_name = os.path.join(self.root_dir, basename)
        image = io.imread(img_name)

        m = re.search('expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg', basename)
        steering_command = np.array(float(m.group(3)), dtype=np.float32)

        if self.categorical:
            steering_command = int(((steering_command - self.min_cmd)/(self.max_cmd - self.min_cmd)) * (self.classes - 1))
            
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'cmd': steering_command}


def get_dataset(args):
    def create_LUT_8UC1(x, y):
        """Creates a look-up table using scipy's spline interpolation"""
        spl = UnivariateSpline(x, y)
        return spl(range(256))

    def color_filter_warm(image):
        incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                                      [0, 70, 140, 210, 256])
        decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                                      [0, 30, 80, 120, 192])
        c_r, c_g, c_b = cv2.split(image)
        c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
        image = cv2.merge((c_r, c_g, c_b))

        c_h, c_s, c_v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

    def gauss_noise_1(image, mu=0.4, variance=0.2):
        gauss_img = img_as_ubyte(random_noise(image, mode='gaussian', mean=mu, var=variance, clip=True))
        return gauss_img

    def gauss_noise_2(image, mu=-0.4, variance=0.2):
        gauss_img = img_as_ubyte(random_noise(image, mode='gaussian', mean=mu, var=variance, clip=True))
        return gauss_img

    base_train_transforms = [
        transforms.ToTensor()
    ]
    base_test_transforms = [
        transforms.ToTensor()
    ]
    training_dataset_0 = DrivingDataset(root_dir=args.train_dir,
                                      categorical=True,
                                      classes=args.n_steering_classes,
                                      transform=transforms.Compose(base_train_transforms))

    validation_dataset_0 = DrivingDataset(root_dir=args.validation_dir,
                                        categorical=True,
                                        classes=args.n_steering_classes,
                                        transform=transforms.Compose(base_test_transforms))

    training_dataset_1 = DrivingDataset(root_dir=args.train_dir,
                                        categorical=True,
                                        classes=args.n_steering_classes,
                                        transform=transforms.Compose([gauss_noise_1, *base_train_transforms]))

    validation_dataset_1 = DrivingDataset(root_dir=args.validation_dir,
                                          categorical=True,
                                          classes=args.n_steering_classes,
                                          transform=transforms.Compose([gauss_noise_1, *base_test_transforms]))
    training_dataset_2 = DrivingDataset(root_dir=args.train_dir,
                                        categorical=True,
                                        classes=args.n_steering_classes,
                                        transform=transforms.Compose([gauss_noise_2, *base_train_transforms]))

    validation_dataset_2 = DrivingDataset(root_dir=args.validation_dir,
                                          categorical=True,
                                          classes=args.n_steering_classes,
                                          transform=transforms.Compose([gauss_noise_2, *base_test_transforms]))

    return [training_dataset_0, training_dataset_1, training_dataset_2], [validation_dataset_0, validation_dataset_1, validation_dataset_2]
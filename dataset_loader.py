from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from os.path import isfile, join
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
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        basename = self.filenames[idx]
        img_name = os.path.join(self.root_dir, basename)
        image = io.imread(img_name)

        m = re.search('expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg', basename)
        steering_command = np.array(float(m.group(3)), dtype=np.float32)

        if self.categorical:
            steering_command = int(((steering_command + 1.0)/2.0) * (self.classes - 1)) 
            
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'cmd': steering_command}


def get_dataset(args):
    def gauss_noise(image, mu=0, variance=0.1):
        gauss_img = img_as_ubyte(random_noise(image, mode='gaussian', mean=mu, var=variance, clip=True))
        return gauss_img

    base_train_transforms = [
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(degrees=80),
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
                                        transform=transforms.Compose([gauss_noise, *base_train_transforms]))

    validation_dataset_1 = DrivingDataset(root_dir=args.validation_dir,
                                          categorical=True,
                                          classes=args.n_steering_classes,
                                          transform=transforms.Compose([gauss_noise, *base_test_transforms]))
    return [training_dataset_0, training_dataset_1], [validation_dataset_0, validation_dataset_1]
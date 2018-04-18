import torch
import torch.utils.data as data
import numpy
import random
import cv2
import sys
import torch.nn.functional as F
import gzip
import pickle
import numpy as np
import os

class mnist_dataset(data.Dataset):
    def __init__( self, data_path, train ):
        with gzip.open( data_path, 'rb' ) as f:
            train_set, valid_set, test_set = pickle.load( f, encoding = 'latin1' )
        if train:
            self.data = train_set
        else:
            self.data = test_set

        self.inp = self.data[0]
        self.label = self.data[1]

    def __len__(self):
        return len(self.label)

    def __getitem__( self, index ):
        
        data = self.inp[index]
        label = self.label[index]
        data = data.reshape( 1, 28, 28 )
        data += np.random.uniform(0, 1, size=(1, 28, 28)) * np.random.uniform(0, 1, size=(1, 28, 28)) * (data == 0)
        data = torch.from_numpy( data ) - 0.5
        
        """
        batch = {
            'inp': data,
            'gt' : label,
            }
        """
        return data, label
        
class random_gauss(data.Dataset):
    def __init__( self, l, gauss ):
        self.len = l
        self.gauss = gauss

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.gauss:
            return torch.randn( 3, 224, 224 )
        else:
            return torch.zeros(1)

class random_uniform(data.Dataset):
    def __init__( self, l, binary ):
        self.len = l
        self.bi = binary

    def __len__(self):
        return self.len

    def __getitem__( self, index ):
        if self.bi:
            return torch.rand( 1, 224, 224 )
        else:
            return torch.zeros(1)

from PIL import Image

class chestx_dataset(data.Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

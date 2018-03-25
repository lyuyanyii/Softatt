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
        

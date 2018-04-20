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

class cub200_dataset(data.Dataset):
    def __init__(self, data_dir, mode, transform=None):
        image_dirs = []
        labels = []
        ids = []
        image_list_file = os.path.join( data_dir, 'images.txt' )
        with open(image_list_file, 'r') as f:
            for line in f:
                image_dirs.append( os.path.join(data_dir, 'images', line.split()[1]) )
        label_list_file = os.path.join( data_dir, 'image_class_labels.txt' )
        with open(label_list_file, 'r') as f:
            for line in f:
                labels.append( int(line.split()[1]) - 1 )
        id_list_file = os.path.join( data_dir, 'train_test_split.txt' )
        with open(id_list_file, 'r') as f:
            for line in f:
                ids.append( int(line.split()[1]) )
        bbox_list_file = os.path.join( data_dir, 'bounding_boxes.txt' )
        bboxes = []
        with open(bbox_list_file, 'r') as f:
            for line in f:
                idx, x, y, w, h = line.split()
                bboxes.append( (float(x), float(y), float(w), float(h)) )
        flag = not (mode == 'train')

        L = len(image_dirs)
        image_dirs = [ image_dirs[i] for i in range(L) if ids[i] == flag ]
        labels = [ labels[i] for i in range(L) if ids[i] == flag ]
        bboxes = [ bboxes[i] for i in range(L) if ids[i] == flag ]
        
        self.image_dirs, self.labels, self.bboxes = image_dirs, labels, bboxes
        self.transform = transform
        self.mode = mode
        self.len = len(image_dirs)

    def __len__( self ):
        return self.len

    def __getitem__( self, index ):
        image_dir = self.image_dirs[index]
        img = Image.open( image_dir ).convert('RGB')
        label = self.labels[index]
        bbox = self.bboxes[index]
        if self.mode != 'train':
            x, y, w, h = bbox
            img_w, img_h = img.size
            if img_w < img_h:
                r = 256 / img_w
            else:
                r = 256 / img_h
            x, y, w, h = x*r, y*r, w*r, h*r
            img_w, img_h = img_w*r, img_h*r
            x -= img_w/2 - 224/2
            y -= img_h/2 - 224/2
            bbox = x, y, w, h
        if self.transform is not None:
            img = self.transform(img)

        if self.mode == 'train':
            return img, torch.from_numpy(np.array([label]))
        else:
            return img, torch.from_numpy(np.array([label])), torch.from_numpy(np.array(bbox))


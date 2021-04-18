import numpy as np
import os
import pandas as pd
import cv2
import pickle
import torch
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import ast
from builtins import object

class data_preprocess(object):
    def __init__(self, path, size= 256, transform = False):
        super(data_preprocess, self).__init__()
        self.path = path
        self.size = size
        self.transform = transform
        
    #defining function to load data for general ML
    def load_images(self):
        '''
        Function to load images using CV2 and resized to get them ready for training
        Arguments:
        imgPath-- Takes in path of the image as input
        size-- Takes in the required reduced size of image (optional)
        Return:
        images-- Images dataset with all the data features
        labels-- Target labels for all the images in image dataset extracted from the directory name
        '''
        self.images = list()
        self.labels = list()

        for subdir in sorted(os.listdir(self.path)):
            if subdir.strip().startswith('c'):
                subdir_path = os.path.join(self.path, subdir)

                for file in os.listdir(subdir_path):
                    if file.endswith('jpg'):
                        fpath = os.path.join(subdir_path,file)
                        img = cv2.imread(fpath)
                        img_resize = cv2.resize(img,(self.size,self.size))
                        self.images.append(img_resize)
                        self.labels.append(int(fpath.split('/')[-2].replace('c', '')))

        return np.array(self.images), np.array(self.labels)
    
    
    
    def pytorch_dataloader(self):
        '''
        Function to load data as per pytorch dataloader standards to be used with deep learning models
        '''
        #Defining image transforms
        imgTransform =  transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        data = datasets.ImageFolder(self.path, transform =imgTransform)
        return data
    
    
    
    def split_data_train_test(self,data, test_split=0.33):
        '''
        Function to create train test splits for the given data

        data-- takes input the data to be split
        test_split-- takes input the test split ratio 

        Return train and test split datasets
        '''
        
        num_samples =  len(data)
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        split = int(test_split*num_samples)

        index_train, index_test = indices[split:], indices[:split]
        sampler_train = SubsetRandomSampler(index_train)
        sampler_test = SubsetRandomSampler(index_test)

        trainloader = torch.utils.data.DataLoader(data,
                       sampler=sampler_train, batch_size=64)
        testloader = torch.utils.data.DataLoader(data,
                       sampler=sampler_test, batch_size=64)

        return trainloader, testloader

        
    
    #create pickle files
    def pickle_dump(self,transform = False ):
        '''
        Function to call load_images function to laod data and generate the pickle files
        '''
        if transform==False:
            X, y = self.load_images()
            pickle.dump(X, open(str('featureData'+str(self.size)+'.pkl'), 'wb'))
            pickle.dump(y, open(str('targetData'+str(self.size)+'.pkl'),'wb'))
    
        elif transform==True:
            data = self.pytorch_dataloader()
            pickle.dump(data, open('data.pkl','wb'))
        

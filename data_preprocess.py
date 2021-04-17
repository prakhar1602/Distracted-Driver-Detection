import numpy as np
import os
import pandas as pd
import cv2
import pickle
import ast
from builtins import object

class data_preprocess(object):
    def __init__(self, path, size= 128):
        super(data_preprocess, self).__init__()
        self.path = path
        self.size = size
        
    #defining function to load data
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

        return self.images, self.labels
    
    #create pickle files
    def pickle_dump(self):
        '''
        Function to call load_images function to laod data and generate the pickle files
        '''
        X, y = self.load_images()
        pickle.dump(X, open('featureData.pkl', 'wb'))
        pickle.dump(y, open('targetData.pkl','wb'))
    

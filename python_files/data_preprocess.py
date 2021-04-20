from builtins import object
import tensorflow as tf
import os
import numpy as np
import ast
import cv2
from tensorflow import keras
from keras.preprocessing.text import one_hot
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


class data_preprocess(object):
    def __init__(self, path):
        super(data_preprocess, self).__init__()
        self.path = path
        
    #defining function to load data for general ML
    def load_images(self,  size= 128):
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
                        img_resize = cv2.resize(img,(size,size))
                        self.images.append(img_resize)
                        self.labels.append(int(fpath.split('/')[-2].replace('c', '')))

        return self.images, self.labels
    
    
    def keras_dataloader(self, num_classes=10, test_size = 0.2, dim = 128):
        '''
        Function to load images and prepare them to be used with keras models
        Arguments:
        num_classes-- number of target classes
        test_size-- Takes in the required test split size
        dim-- size of image
        Return:
        X and Y train test splits
        '''
        images, labels = self.load_images(size=dim)
        
        X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=test_size)
        #converting labels to categorical values
        Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=num_classes)
        Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=num_classes)
        
        return np.array(X_train, dtype = np.float32), np.array(X_test, dtype = np.float32), Y_train, Y_test
    
    def pytorch_dataloader(self):
        '''
        Function to load data as per pytorch dataloader standards to be used with deep learning models
        '''
        #Defining image transforms
        imgTransform =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,244)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        data = datasets.ImageFolder(self.path, transform =imgTransform)
        return data
    
    
    
    def split_data_train_test(self,data, test_split=0.33, batch_size = 64):
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
                       sampler=sampler_train, batch_size=batch_size)
        testloader = torch.utils.data.DataLoader(data,
                       sampler=sampler_test, batch_size=batch_size)

        return trainloader, testloader

        
    
    #create pickle files
    def pickle_dump(self, transform = False ):
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
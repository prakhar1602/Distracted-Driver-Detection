from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate


class CNN(Model):
    def __init__(self, classes=10, batchDim = -1):
        super(CNN, self).__init__()
        
        #initializing layers
        
        #First convolution layer
        self.conv1 = Conv2D(32, (5,5), strides = (1,1), padding='same', activation='relu')
        self.bn1 = BatchNormalization(axis = batchDim)
        self.pool1 = MaxPooling2D(pool_size=(2,2))
        
        #second convolution layer
        self.conv2 = Conv2D(64, (5,5), strides = (1,1), padding='same', activation='relu')
        self.bn2 = BatchNormalization(axis = batchDim)
        self.pool2 = MaxPooling2D(pool_size=(2,2))
        
        #third convolution layer
        self.conv3 = Conv2D(128, (5,5), strides = (1,1), padding='same', activation='relu')
        self.bn3 = BatchNormalization(axis = batchDim)
        self.pool3 = MaxPooling2D(pool_size=(2,2))
        
        #fourth convolution layer
        self.conv4 = Conv2D(256, (5,5), strides = (1,1), padding='same', activation='relu')
        self.bn4 = BatchNormalization(axis = batchDim)
        self.pool4 = MaxPooling2D(pool_size=(2,2))
        
        #classifier layers (fully connected layers)
        self.flatten = Flatten()
        self.fc1 = Dense(units=1024, activation='relu')
        self.drop1 = Dropout(rate=0.2)
        self.fc2 = Dense(units=512, activation='relu')
        #final softmax layer
        self.fc3 = Dense(units=classes, activation='softmax')
        
    def call(self, x):
        #building up layers

        #first conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        #second conv layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        #third conv layer
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)

        #fourth conv layer
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool4(x)

        #fc layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras.applications import ResNet50


class resnet50(Model):
    def __init__(self, classes=10, layers_unfreeze = 0):
        super(resnet50, self).__init__()
        
        #loading pretrained model without the models final fully connected layers
        self.res = ResNet50(include_top=False)

        self.unfreeze = layers_unfreeze
        
        self.flatten = Flatten()
        self.fc1 = Dense(units=4096, activation='relu')
        self.fc2 = Dense(units=1072, activation='relu')
        self.drop1 = Dropout(rate=0.2)
        #final softmax layer
        self.fc3 = Dense(units=classes, activation='softmax')
        
    def call(self, inputs):
        #building up layers
        #controlling number of layers to be trainable or non-trainable
        if self.unfreeze >0:
            for layer in self.res.layers[:-self.unfreeze]:
                layer.trainable=False
        else:
            for layer in self.res.layers:
                layer.trainable=False

        x = self.res(inputs)

        #fc layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.drop1(x)
        x = self.fc3(x)

        return x
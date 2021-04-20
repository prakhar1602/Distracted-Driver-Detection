from torch import nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self, num_classes, num_input = 4096):
        super(VGG, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.num_classes = num_classes
        self.num_input = num_input
        
        
    def create_model(self):
        '''
        Function to create a VGG16 model using transfer learning
        '''
        #freezing the pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        
        self.model.classifier[6] = nn.Sequential(
            nn.Linear(self.num_input, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes),
            nn.LogSoftmax(dim=1)
            )
            
        return self.model
from builtins import object
import pandas as pd
import torch
from torchvision import datasets

class train_model(object):
    def __init__(self, model, trainset, testset, criterion, optimizer, num_epochs = 5, device='cpu'):
        super(train_model, self).__init__()
        self.model = model
        self.train = trainset
        self.test = testset
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        
    def batchRun(self):
        '''
        Function to run model on the train and test sets
        
        Inputs:
        model-- model to execute
        train-- train dataset
        test-- test dataset
        criterion-- loss criterion for the model
        optimizer-- optimizer used for training the model
        learning_rate-- learning rate for the model
        num_epochs-- number of epochs to train the model
        
        Outpus:
        y_test-- Original test classes
        y_pred-- Predicted classes
        accuracy_df-- dataframe consisitng accuracy and loss scores
        '''
        train_loss = list()
        train_accuracy = list()
        test_accuracy = list()

        for epoch in range(self.num_epochs):
            print('Epoch: ', epoch)
            correct_train = 0
            correct_test = 0
            epoch_train_loss = 0
            
            for  images, labels in self.train:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            epoch_train_loss = epoch_train_loss/len(self.train)
            train_loss.append(epoch_train_loss)
            accuracy1 = self.calculate_acc(self.model,self.train)
            train_accuracy.append(accuracy1)

            print('Train Loss: ',  epoch_train_loss)
            print('Train Accuracy: ', accuracy1)

            #test
            accuracy2 = self.calculate_acc(self.model,self.test)
            test_accuracy.append(accuracy2)
            print('Test Accuracy: ', accuracy2)
        
    

        accuracy_df = pd.DataFrame({'Train Loss': train_loss,
                                   'Train Accuracy': train_accuracy,
                                   'Test Accuracy': test_accuracy})

        return self.model, accuracy_df
    
    def calculate_acc(self,model,loader):
        '''
        Function to calculate accuracy of a model
        
        Inputs:
        model-- model to evaluate
        loader-- data to evaluate the model on
        '''
        model.eval()
        correct = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).float().sum().item()

        accuracy = 100 * correct / len(loader.dataset)
        return accuracy
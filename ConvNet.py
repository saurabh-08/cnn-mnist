import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        
        self.fc1 = nn.Linear(28*28, 100)                        
        self.conv1 = nn.Conv2d(1, 40, kernel_size=5, stride=1)         
        self.conv2 = nn.Conv2d(40, 40, kernel_size=5, stride=1)                         
        self.fc2 = nn.Linear(4*4*40, 100)                  
        self.fc3 = nn.Linear(100, 1000) 
        
        #dropout
        self.dropout = nn.Dropout(0.5)                    
        self.fc_out = nn.Linear(1000 if mode in [4, 5] else 100, 10)

        
        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing   
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        X = X.view(-1, 28*28)       
        X = torch.sigmoid(self.fc1(X))         
        
        return self.fc_out(X)
    
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
#         return NotImplementedError()

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        X = torch.sigmoid(F.max_pool2d(self.conv1(X), 2))         
        X = torch.sigmoid(F.max_pool2d(self.conv2(X), 2))         
        X = X.view(-1, 4*4*40)         
        X = torch.sigmoid(self.fc2(X))         
        
        return self.fc_out(X)
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
#         return NotImplementedError()

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        X = F.relu(F.max_pool2d(self.conv1(X), 2))         
        X = F.relu(F.max_pool2d(self.conv2(X), 2))         
        X = X.view(-1, 4*4*40)         
        X = F.relu(self.fc2(X))         
        
        return self.fc_out(X)
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
#         return NotImplementedError()

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        X = F.relu(F.max_pool2d(self.conv1(X), 2))         
        X = F.relu(F.max_pool2d(self.conv2(X), 2))         
        X = X.view(-1, 4*4*40)         
        X = F.relu(self.fc2(X))         
        X = F.relu(self.fc3(X))         
        
        return self.fc_out(X)
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
#         return NotImplementedError()

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        X = F.relu(F.max_pool2d(self.conv1(X), 2))         
        X = F.relu(F.max_pool2d(self.conv2(X), 2))         
        X = X.view(-1, 4*4*40)         
        X = F.relu(self.fc2(X))         
        X = F.relu(self.fc3(X))         
        X = self.dropout(X)         
        
        return self.fc_out(X)
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
#         return NotImplementedError()
    
    

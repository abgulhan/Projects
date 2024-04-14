'''
Start code for Project 3
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name:
    PSU Email ID:
    Description: (A short description of what each of the functions you've written does.).
}
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Can the MLP be improved?
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): number of input features
            hidden_dim (int): number of hidden units
            output_dim (int): number of output units
        """
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.log_softmax(self.layers(x), dim=1)
 
# make non linear   
class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): number of input features
            hidden_dim (int): number of hidden units
            output_dim (int): number of output units
        """
        super(MLP2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.log_softmax(self.layers(x), dim=1)
    
# make non linear and add another hidden layer  
class MLP3(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        """
        Args:
            input_dim (int): number of input features
            hidden_dim1 (int): number of hidden units in hidden layer 1
            hidden_dim2 (int): number of hidden units in hidden layer 2
            output_dim (int): number of output units
        """
        super(MLP3, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim),
        )

    def forward(self, x):
        return F.log_softmax(self.layers(x), dim=1)

class CNN(nn.Module):
    def __init__(self, input_channels=1, img_size=32, num_classes=17):
        """
        Args:
            input_channels (int): number of channels in the input image
            img_size (int): size of the input image (img_size x img_size)
            num_classes (int): number of classes in the dataset
        """
        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.conv_layers= nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.fc_1 = nn.Linear(128 * (self.img_size // 8) * (self.img_size // 8), 1024)
        self.fc_2 = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return F.log_softmax(x, dim=1)

# TODO: Can the CNN be improved? You may want to add or remove any arguments to the init.
class CNN2(nn.Module):
    def __init__(self, input_channels=1, img_size=32, num_classes=17):
        """
        Args:
            input_channels (int): number of channels in the input image
            img_size (int): size of the input image (img_size x img_size)
            num_classes (int): number of classes in the dataset
        """
        super(CNN2, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.img_size = img_size
        
        """
        From EscherNet paper:
        The first 2 layers are
        two repetitions of Conv-ReLU-Conv-ReLU-Pool with 32
        7x7 filters for the first two Conv layers (conv1 and conv2)
        and with 64 3x3 filters for the others (conv4 and conv5).
        The max pool layers are 2x2 with a stride of 2
        """
        self.conv_layers= nn.Sequential(
            # Layer 1
            nn.Conv2d(self.input_channels, 32, kernel_size=7, padding=1),
            nn.ReLU(), 
            nn.Conv2d(32, 32, kernel_size=7, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        """
        These
        are followed by two fully connected layers with 512 and
        17 neurons, respectively, with a ReLU between them, and
        a final softmax layer
        """
        self.fc_1 = nn.Sequential(
            nn.Linear(64 * (self.img_size-7+2+1-7+2+1)//4 * (self.img_size-7+2+1-7+2+1)//4, 512),
            nn.ReLU(),
        )
        self.fc_2 = nn.Linear(512, self.num_classes)

        self.dropout = nn.Dropout(0.8)
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        return F.log_softmax(x, dim=1)
    
    
class CNN3(nn.Module):
    def __init__(self, input_channels=1, img_size=32, num_classes=17):
        """
        Args:
            input_channels (int): number of channels in the input image
            img_size (int): size of the input image (img_size x img_size)
            num_classes (int): number of classes in the dataset
        """
        super(CNN3, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.img_size = img_size
        
        """
        From EscherNet paper:
        The first 2 layers are
        two repetitions of Conv-ReLU-Conv-ReLU-Pool with 32
        7x7 filters for the first two Conv layers (conv1 and conv2)
        and with 64 3x3 filters for the others (conv4 and conv5).
        The max pool layers are 2x2 with a stride of 2
        """
        self.conv_layers= nn.Sequential(
            # Layer 1
            nn.Conv2d(self.input_channels, 32, kernel_size=7, padding=1),
            nn.ReLU(), 
            nn.Conv2d(32, 32, kernel_size=7, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        """
        These
        are followed by two fully connected layers with 512 and
        17 neurons, respectively, with a ReLU between them, and
        a final softmax layer
        """
        self.fc_1 = nn.Sequential(
            nn.Linear(64 * (self.img_size-7+2+1-7+2+1)//8 * (self.img_size-7+2+1-7+2+1)//8, 512),
            nn.ReLU(),
        )
        self.fc_2 = nn.Linear(512, self.num_classes)

        self.dropout = nn.Dropout(0.8)
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        return F.log_softmax(x, dim=1)
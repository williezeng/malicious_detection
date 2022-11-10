import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        
        # TODO: Initialize a fully convolutional neural network that works.
        self.nn_layers = nn.ModuleList()
        
        self.nn_layers.append(nn.Conv1d(1,96,3))
        self.nn_layers.append(nn.ReLU())
        self.nn_layers.append(nn.MaxPool1d(2,stride=2))
        
        self.nn_layers.append(nn.Conv1d(96,128,3))
        self.nn_layers.append(nn.ReLU())
        self.nn_layers.append(nn.MaxPool1d(2,stride=2))
        
        self.nn_layers.append(nn.Conv1d(128,192,3))
        self.nn_layers.append(nn.ReLU())
        self.nn_layers.append(nn.MaxPool1d(2,stride=2))
        
        self.nn_layers.append(nn.Conv1d(192,256,3))
        self.nn_layers.append(nn.ReLU())
        self.nn_layers.append(nn.MaxPool1d(2,stride=2))
        
        self.nn_layers.append(nn.Conv1d(256,384,3))
        self.nn_layers.append(nn.ReLU())
        self.nn_layers.append(nn.MaxPool1d(2,stride=2))

        self.nn_layers.append(nn.AdaptiveMaxPool1d(10))
        
        self.nn_layers.append(nn.Flatten())
        self.nn_layers.append(nn.Dropout())
        self.nn_layers.append(nn.Linear(3840, 4096))
        self.nn_layers.append(nn.ReLU())
        self.nn_layers.append(nn.Linear(4096, 2))

    def forward(self, x):
        '''Takes in a raw PowerShell script converted to floats and returns a prediction.
        
        Arguments:
        x - n x m matrix where n is the number of samples and m is an array of
            floats representing the PowerShells cript.
            
        Returns: a n x 2 matrix where the [:,0] is clean and [:,1] is malicious.
        '''
        #print("x.shape:", x.shape)
        outs = None
        current = x
        for module in self.nn_layers:
            #print("Current.shape:", current.shape)
            current = module(current)
        outs = current
        return outs
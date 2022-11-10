import os.path
import random
import math
import progressbar

import numpy as np
import torch
import torch.nn.functional as F

class Scanner():
    def __init__(self, name):
        self.model = self.LoadModule(name)
        
    def LoadModule(self, name):
        # get a path to the model
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "checkpoints", name + '.pth')
        
        # load the model
        model = torch.load(path)
        
        # turn-off dropout and other parameters
        model.eval()
        
        return model

    def LoadSample(self, path):
        # opening for [r]eading as [b]inary
        with open(path, "rb") as file:
            # convert to a float tensor
            sample = torch.tensor([[[float(b) for b in file.read()]]], dtype=torch.float32)
            
            # pad to 192 characters if necessary
            if sample.shape[2] < 192:
                sample = F.pad(sample, (0, 192 - sample.shape[2]), "constant", 0)
                
            if torch.cuda.is_available():
                sample = sample.cuda()
        
        return sample
    
    def Score(self, path):
        # load the file as a tensor
        sample = self.LoadSample(path)
        
        # feed the sample through the model to generate probabilities
        with torch.no_grad():
            output = self.model(sample)
        
        # return a dictionary
        return output.squeeze().cpu()
        
    def IsFileMalicious(self, sample):
        # feed the sample through the model to generate probabilities
        output = self.Score(path)
        
        # if the index of the max value of the tensor is 1, then the sample is malicious
        return torch.argmax(output) == 1
        
    def IsSampleMalicious(self, sample):
        # verify dimensions and pad if necessary
        if sample.shape[2] < 5:
            return False
        elif sample.shape[2] < 192:
            sample = F.pad(sample, (0, 192 - sample.shape[2]), "constant", 0)
                
        if torch.cuda.is_available():
            sample = sample.cuda()
            
        # feed the sample through the model to generate probabilities
        output = self.model(sample)
        
        # if the index of the max value of the tensor is 1, then the sample is malicious
        return torch.argmax(output) == 1

    def SigFindPerChar(self, path):
        # load the file as a tensor
        sample = self.LoadSample(path)
        
        if not self.IsSampleMalicious(sample):
            print(f"[*] File: {path}")
            print("[*] Malicious: False")
            return ""
        
        # run the delta debug algorithm on the sample
        # to identify the one-minimal tensor.
        delta = self.SigFindPerCharHelper(sample)
        
    def SigFindPerCharHelper(self, sample):
        n = 2
        delta = sample
        while True:
            if n >= delta.shape[2] * 2:
                break;
            elif n > delta.shape[2]:
                n = delta.shape[2]
            
            # calculate the step size
            step = int(math.ceil(float(delta.shape[2]) / n))
            
            # print verbose message
            print(f"[*] n: {n} length: {delta.shape[2]}")
            
            # generate the deltas
            deltas = self.GenerateDeltas(delta, step)
            found = False
            for deltap in deltas:
                malicious = self.IsSampleMalicious(deltap)
                if malicious:
                    found = True
                    n = 2
                    delta = deltap
                    break
            
            if found:
                continue
            
            if n == 2:
                # no need to calculate the nablas if n is 2. This is
                # a degenerative case where the nablas are equal to the
                # deltaps
                n *= 2
            else:
                # generate the nablas
                nablas = self.GenerateNablas(delta, step)
                
                for nabla in nablas:
                    malicious = self.IsSampleMalicious(deltap)
                    if malicious:
                        found = True
                        n = n - 1
                        delta = nabla
                        break
                
                if not found:
                    n = n * 2
            
        # exit condition met, analysis complete! print the result.
        
        a = bytes([int(x) for x in delta[0,0].tolist()])
        text = a.decode('utf-8')
        print("[*] Analysis complete!")
        print("[*] Signaturized string:")
        print(f"[*] {text}")
        
        return text
            
    def GenerateDeltas(self, delta, step):
        '''
        Generates a set of all chunks from the parent tensor (delta)
        by splitting the tensor into chunks of size step to be scanned
        for malicious content.
        '''
        result = []
        for i in range(0, delta.shape[2], step):
            stride = min(step, delta.shape[2] - i)
            view = delta[:,:,i:i+stride]
            result.append(view)
        return result
        
    def GenerateNablas(self, delta, step):
        '''
        Generates a set of tensors where chunks of size step are
        removed from original tensor in a sliding window.
        '''
        result = []
        for i in range(0, delta.shape[0], step):
            stride = min(step, delta.shape[0] - i)
            view = delta[delta!=delta[:,:,i:i+stride]]
            result.append(view)
        return result
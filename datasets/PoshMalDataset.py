from torch.utils.data.dataset import Dataset  # For custom datasets
import torch.nn.functional as F
import os
import os.path
import random
import numpy as np
import torch
import progressbar

class PoshMalDataset(Dataset):
    def __init__(self, path, clean = 1000, unmodified = 250, insert = 250, obfuscated = 250, bypass = 250, max_filesize=128*1024):
        """
        Custom dataset loader for the poshmal dataset.

        Args:
            path (string): path to poshmal dataset directory.
            clean (int): number of clean samples to load.
            unmodified (int): number of unmodified malware samples to load.
            insert (int): number of insert malware samples to load.
            obfuscated (int): number of obfuscated samples to load.
            bypass (int): number of obfuscated bypass samples to load.
        """
        
        clean_path = os.path.join(path, "clean")
        unmodified_path = os.path.join(path, "malicious-unmodified")
        insert_path = os.path.join(path, "malicious-insert")
        bypass_path = os.path.join(path, "malicious-bypass-techniques-obfuscated")
        obfuscated_path = os.path.join(path, "malicious-obfuscated")
        
        self.samples = []
        self.samples += self.load_from_folder(clean_path, clean, 0, max_filesize)
        self.samples += self.load_from_folder(unmodified_path, unmodified, 1, max_filesize)
        self.samples += self.load_from_folder(insert_path, insert, 1, max_filesize)
        self.samples += self.load_from_folder(bypass_path, bypass, 1, max_filesize)
        self.samples += self.load_from_folder(obfuscated_path, obfuscated, 1, max_filesize)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
    
    def load_from_folder(self, path, count, malicious, max_filesize):
        # Get a list of full filepaths
        print(f"[*] Get a listing a files in {path}.")
        files = []
        for file in os.listdir(path):
            fullname = os.path.join(path, file)
            if os.path.isfile(fullname):
                size = os.path.getsize(fullname)
                if size <= max_filesize:
                    files.append(fullname)
                    
        if len(files) < count:
            print(f"[!] Required {count} samples but only found {len(files)} samples. Padding subset with duplicates.")
        
        random.shuffle(files)
        output = []
        toload = min(len(files), count)
        bar = progressbar.ProgressBar(maxval=toload, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i in range(0, toload):
            bar.update(i)
            data = None
            #print(f"[*] Loading {files[i]}...")
            sample = None
            with open(files[i], "rb") as file: # opening for [r]eading as [b]inary
                #sample = [float(b) for b in file.read()]
                sample = torch.tensor([[[float(b) for b in file.read()]]], dtype=torch.float32)
                if sample.shape[2] < 192:
                    sample = F.pad(sample, (0, 192 - sample.shape[2]), "constant", 0)
                #sample = torch.tensor((1, temp.shape[0]))
                #sample[0] = temp
            
            label = 1 if malicious else 0
            output.append((sample, label))
        bar.finish()
        
        # Duplicate this part of the dataset to ensure that the desired number of samples can be met
        for i in range(0, count - len(files)):
            selected = output[random.randint(0, toload - 1)]
            output.append(selected)
            
        #print(f"[*] Loaded {len(output)} files.")
        
        return output

def simple_collate_fn(data):
    samples,labels = zip(*data)
    labels = torch.tensor(labels, dtype=torch.int64)
    
    return samples,labels

def poshmal_collate_fn(data):
    samples, labels, lengths = zip(*data)
    max_len = max(lengths)
    print("max_len:", max_len)
    features = torch.zeros((len(data), max_len, 1))
    labels = torch.vstack(labels)

    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        print(f"(j, k) = ({j}, {k})")
        features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

    return features.float(), labels
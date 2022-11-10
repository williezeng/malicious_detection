import yaml
import argparse
import time
import copy
import os
import os.path
import random
import math
import gc

import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix

from models import FCNN
from datasets import PoshMalDataset,poshmal_collate_fn,simple_collate_fn

parser = argparse.ArgumentParser(description='Trains a model using the poshmal dataset.')
parser.add_argument('--datadir', default='./../poshmal/')
parser.add_argument('--config', default='./configs/fcnn.yaml')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc
    
def compute_metrics(output, target):
    '''
    Computes the accuracy, precision, recall, and FP rate of the model
    on the specified output.
    
    Returns: (accuracy, precision, recall, FPR)
    '''
    
    # ensure tensors are on the CPU to pass to scikit-learn
    output = output.cpu()
    target = target.cpu()

    _,predictions = torch.max(output, dim=-1)

    tn,fp,fn,tp = confusion_matrix(target, predictions).ravel()

    accuracy = torch.tensor(accuracy_score(target, predictions))
    precision = torch.tensor(precision_score(target, predictions, average='weighted', zero_division=0))
    recall = torch.tensor(recall_score(target, predictions))
    fpr = fp / (fp + tn)
    
    return (accuracy, precision, recall, fpr)

def train(epoch, data_loader, model, optimizer, criterion):
    if torch.cuda.is_available():
        # try to free up unused memory before starting the epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        # print memory stats so we can see if there is an issue
        memory = torch.cuda.memory_stats()
        allocated = memory['allocated_bytes.all.current'] / (1024.0 * 1024.0)
        reserved = memory['reserved_bytes.all.current'] / (1024.0 * 1024.0)
        ooms = memory['num_ooms']
        print("[*] OOMS: {ooms}\tAllocated: {allocated:.4f}MiB\tReserved: {reserved:.4f}MiB".format(ooms=ooms, allocated=allocated, reserved=reserved))

    iter_time = AverageMeter()
    losses = AverageMeter()

    count = 0
    for idx,(samples,labels) in enumerate(data_loader):
        start = time.time()

        # Create a tensor to hold the output
        output_combined = torch.zeros((labels.shape[0], 2))
        indices = [i for i in range(0, labels.shape[0])]
        
        # Ensure the tensors are deployed to the GPU if available
        if torch.cuda.is_available():
            output_combined.cuda()
            labels = labels.cuda()
        
        # Zero out any gradients stored in the model from the previous
        # iteration so that we don't repeat old errors
        optimizer.zero_grad()
        total_loss = 0.0
        for sample_idx,sample in zip(indices, samples):
            # Ensure the tensors are deployed to the GPU if available
            if torch.cuda.is_available():
                sample = sample.cuda()
                
            # Forward pass
            output = model(sample)
            
            # Increase the error for false positives to make the model bias
            # against that term
            weight = 2.0 if torch.argmax(output) == 1 and labels[sample_idx] == 0 else 1.0
            
            # The loss must be normalized by the length of the batch size and
            # multiplied by the false positive bias weight
            loss = criterion(output, labels[sample_idx:sample_idx+1]) / len(samples) * weight
            
            # Calling loss.backward() accumulates the gradients for the model
            loss.backward()
            
            # .item() disconnects the loss from the graph so that we don't hold
            # a reference to that structure in memory
            losses.update(loss.item())
            
            # Store a coyp of the output so that we can compute statistics
            # later in the program
            output_combined[sample_idx] = output.detach()
        
        # Ensure that the output_combined tensor is on the cuda device
        if torch.cuda.is_available():
            output_combined = output_combined.cuda()
            
        # Update the weights of the model based on the cummulative gradients
        # from the loop above
        optimizer.step()

        # Compute the accuracy, precision, recall, and false positive rate
        ac,prec,rec,fpr = compute_metrics(output_combined, labels)

        iter_time.update(time.time() - start)
        count += 1
        if True: # count % 10 == 0:
            print(('[*] Epoch: [{0}][{1}/{2}]\t'
                   'Time: {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss: {loss:.4f}\t'
                   'Acc: {ac:.4f}\t'
                   'Prec: {prec:.4f})\t'
                   'Rec: {recall:.4f}\t'
                   'FPR: {fpr:.4f}')
                   .format(epoch, count, len(data_loader), iter_time=iter_time, loss=losses.sum, ac=ac, prec=prec, recall=rec,fpr=fpr))

def validate(epoch, val_loader, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    ac_total = AverageMeter()
    prec_total = AverageMeter()
    rec_total = AverageMeter()
    fpr_total = AverageMeter()

    num_class = 2
    cm =torch.zeros(num_class, num_class)
    # evaluation loop
    for idx, (samples, labels) in enumerate(val_loader):
        start = time.time()

        # Create a tensor to hold the output
        output_combined = torch.zeros((labels.shape[0], 2))
        indices = [i for i in range(0, labels.shape[0])]

         # Ensure the tensors are deployed to the GPU if available
        if torch.cuda.is_available():
            output_combined.cuda()
            labels = labels.cuda()
        
        with torch.no_grad():
            for sample_idx,sample in zip(indices, samples):
                # Ensure the tensors are deployed to the GPU if available
                if torch.cuda.is_available():
                    sample = sample.cuda()
                    
                #Forward
                output_combined[sample_idx] = model(sample)
        
        if torch.cuda.is_available():
            output_combined = output_combined.cuda()
            
        loss = criterion(output_combined, labels).detach().item()
        ac,prec,rec,fpr = compute_metrics(output_combined, labels)

        # update confusion matrix
        _, preds = torch.max(output_combined, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, output_combined.shape[0])
        ac_total.update(ac, output_combined.shape[0])
        prec_total.update(prec, output_combined.shape[0])
        rec_total.update(rec, output_combined.shape[0])
        fpr_total.update(fpr, output_combined.shape[0])

        iter_time.update(time.time() - start)
        if idx % 2 == 0:
            print(('[*] Epoch: [{0}][{1}/{2}]\t'
               'Time: {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
               'Acccuracy: {ac:.4f}\t'
               'Precision: {prec:.4f}\t'
               'Recall: {rec:.4f}\t'
               'FPR: {fpr:.4f}\t'
               .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, ac = ac, prec=prec, rec=rec, fpr=fpr)))
    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("[*] Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("[*] Validation:")
    print(f"[*] Accuracy:  {ac_total.avg:.4f}")
    print(f"[*] Precision: {prec_total.avg:.4f}")
    print(f"[*] Recall:    {rec_total.avg:.4f}")
    print(f"[*] FPR:       {fpr_total.avg:.4f}")
    
    return ac_total.avg, cm


def adjust_learning_rate(optimizer, epoch, args):
    epoch += 1
    if epoch <= args.warmup:
        lr = args.learning_rate * epoch / args.warmup
    elif epoch > args.steps[1]:
        lr = args.learning_rate * 0.01
    elif epoch > args.steps[0]:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_dataset(args):
    dataset = PoshMalDataset(args.datadir, clean = args.clean, unmodified = args.unmodified, insert = args.insert, obfuscated = args.obfuscated, bypass = args.bypass, max_filesize =  args.max_filesize)
    
    train_length = math.floor(args.train * len(dataset))
    validation_length = math.floor(args.validation * len(dataset))
    test_length = len(dataset) - validation_length - train_length
    
    dataset_train,dataset_validation,dataset_test = random_split(dataset, (train_length, validation_length, test_length))
    
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=simple_collate_fn, shuffle=True)
    loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=args.batch_size, collate_fn=simple_collate_fn, shuffle=True)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, collate_fn=simple_collate_fn, shuffle=True)
    
    return loader_train,loader_validation,loader_test

def main():
    # Parse the arguments
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
            
    # Extract the class weights
    args.class_weights = torch.FloatTensor(args.class_weights)
    if torch.cuda.is_available():
        args.class_weights = args.class_weights.cuda()
    
    # load the dataset
    loader_train,loader_validation,loader_test = load_dataset(args)
    
    # Model selector. Add any additional modules here.
    if args.model == 'FCNN':
        model = FCNN()
    else:
        raise Exception(f"[!] The model {args.model} is not supported.")
    
    # TODO: Print the configuration of the model
    print(model)
    
    # Transfer the model to the GPU if available
    if torch.cuda.is_available():
        print("[*] Hardware: Cuda")
        model = model.cuda()
    else:
        print("[*] Hardware: CPU")

    if args.loss_type == "CE":
        criterion = nn.CrossEntropyLoss(weight=args.class_weights)
    else:
        raise Exception(f"[!] The {args.loss_type} loss function is not supported.")

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.reg)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the module
    best = 0.0
    best_cm = None
    best_model = None
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train loop
        train(epoch, loader_train, model, optimizer, criterion)

        # validation loop
        acc, cm = validate(epoch, loader_validation, model, criterion)

        if acc > best:
            best = acc
            best_cm = cm
            best_model = copy.deepcopy(model)
            
            if args.save_best:
                path = './checkpoints/' + args.model.lower() + '.pth'
                print(f"[*] Exporting best model to {path}")
                torch.save(best_model, path)
                
    acc, cm = validate(epoch, loader_test, model, criterion)

    print('[*] Best Prec @1 Acccuracy: {:.4f}'.format(best))
    per_cls_acc = best_cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("[*] Accuracy of Class {}: {:.4f}".format(i, acc_i))

if __name__ == '__main__':
    main()
import torch
import numpy as np 
import pandas as pd 
import time
from models import n_params

class TrainOutput:
    """
    Container for training output
    """
    def __init__(self, train_loss, val_loss, train_acc, val_acc, n_params):
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.n_params = n_params
        
        
def to_device(data_pre, len_pre, data_post, len_post, labels, device):
    # Very annoying.... must be better way
    data_pre = data_pre.to(device)
    len_pre = len_pre.to(device)
    data_post = data_post.to(device)
    len_post = len_post.to(device)
    labels = labels.to(device)
    return data_pre, len_pre, data_post, len_post, labels


def acc(loader, model):
    """
    Calculate accuracy of model on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for data_pre, len_pre, data_post, len_post, labels in loader:
            data_pre, len_pre, data_post, len_post, labels = to_device(data_pre, len_pre, data_post, len_post, labels, device)
            outputs = model(data_pre, data_post, len_pre, len_post)
            predicted = outputs.max(1, keepdim=True)[1]
            
            total += labels.size(0)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)


def avg_loss(loader, model, criterion):
    """
    Calculate average loss on a dataset
    @param: loader - data loader for the dataset to test against
    """
    loss = 0
    n = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for data_pre, len_pre, data_post, len_post, labels in loader:
            data_pre, len_pre, data_post, len_post, labels = to_device(data_pre, len_pre, data_post, len_post, labels, device)
            outputs = model(data_pre, data_post, len_pre, len_post)
            loss += criterion(outputs, labels)
            n += labels.size(0)
    return loss / n


def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs=10, save_file='model.pt', device=None):
    """
    Train model and save best model based on validation performance
    @param: train_loader - data loader for training set
    @param: val_loader - data loader for validation set
    @param: optimizer - optimizer
    @param: criterion - loss function
    @param: n_epochs - number of epochs to train for
    @param: save_file - path to save best model
    @param: device - device to train model on, "cpu" or "cuda:0" or None

    Returns: TrainingOutput
    """
    
    start = time.time()
    best_acc = 0
    best_loss = float('inf')

    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    for epoch in range(n_epochs):
        print("Starting epoch {}".format(epoch))
        # Iterate over train set
        for batch, (data_pre, len_pre, data_post, len_post, labels) in enumerate(train_loader):
            
            # Gross
            data_pre, len_pre, data_post, len_post, labels = to_device(data_pre, len_pre, data_post, len_post, labels, device)
            
            model.train()
            optimizer.zero_grad()
            
            y_hat = model(data_pre, data_post, len_pre, len_post)
            
            loss = criterion(y_hat, labels)
                        
            loss.backward()
            optimizer.step()
            
            if (batch+1) % 500 == 0:
                model.eval()
                val_acc = acc(val_loader, model)
                print('Epoch: [{}/{}], Step: [{}/{}],Training Loss: {}, Validation Acc: {}, Time: {} sec'.format(epoch+1, n_epochs, batch+1, len(train_loader), loss, val_acc, time.time()-start))
    
        # Calculate validation performance
        model.eval()
        train_acc = acc(train_loader, model)
        val_acc = acc(val_loader, model)
        print('End of epoch {}, Training Acc: {},Validation Acc: {}, Time: {} sec'.format(
                                                                         epoch+1, train_acc, val_acc, time.time()-start))

        if val_acc > best_acc:
            best_acc = val_acc
            print("New best model found, saving at {}".format(save_file))
            torch.save(model.state_dict(), save_file)
        print()
    
    # return the best model and its validation performance
    model.load_state_dict(torch.load(save_file))

    # Inefficiently calculate metrics separately....
    train_loss = avg_loss(train_loader, model, criterion)
    val_loss = avg_loss(val_loader, model, criterion)
    train_acc = acc(train_loader, model)
    val_acc = best_acc
    n_trainable_params = n_params(model)
    res = TrainOutput(train_loss=train_loss.item(), val_loss=val_loss.item(), train_acc=train_acc, val_acc=val_acc, n_params=n_trainable_params)
    return res

import torch
import numpy as np 
import pandas as pd 
import time

def acc(loader, model):
    """
    Calculate accuracy of model on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data_pre, len_pre, data_post, len_post, labels in loader:
            outputs = model(data_pre, data_post, len_pre, len_post)
            predicted = outputs.max(1, keepdim=True)[1]
            
            total += labels.size(0)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)


def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs=10,save_file='model.pt'):
    """
    Train model and save best model based on validation performance
    @param: train_loader - data loader for training set
    @param: val_loader - data loader for validation set
    @param: optimizer - optimizer
    @param: criterion - loss function
    @param: n_epochs - number of epochs to train for
    @param: save_file - path to save best model

    Returns: (model, accuracy)
    """
    
    start = time.time()
    best_acc = 0

    for epoch in range(n_epochs):
        print("Starting epoch {}".format(epoch))
        #sum_loss_training = 0.0
        # Iterate over train set
        for batch, (data_pre, len_pre, data_post, len_post, label) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            
            y_hat = model(data_pre, data_post, len_pre, len_post)
            
            loss = criterion(y_hat, label)
                        
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
    return model, train_acc, val_acc

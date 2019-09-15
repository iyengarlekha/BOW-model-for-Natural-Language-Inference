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


# TODO add save locations
def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs=10, save_file='model.pt'):
    start = time.time()
    best_acc = 0

    for epoch in range(n_epochs):
        print("Starting epoch {}".format(epoch))

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
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}, Time: {} sec'.format( 
                        epoch+1, n_epochs, batch+1, len(train_loader), val_acc, time.time()-start))


        # Calculate validation performance
        model.eval()
        val_acc = acc(val_loader, model)
        print('End of epoch {}, Validation Acc: {}, Time: {} sec'.format( 
                epoch+1, val_acc, time.time()-start))
        if val_acc > best_acc:
            best_acc = val_acc
            print("New best model found, saving at {}".format(save_file))
            torch.save(model.state_dict(), save_file)
        print()

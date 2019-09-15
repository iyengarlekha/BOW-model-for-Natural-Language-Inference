import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    '''
    Logistic regression classification model
    '''
    def __init__(self, embed_layer, embed_dim, num_class, cat_mode='DIRECT'):
        super(LogisticRegression, self).__init__()

        self.embed = embed_layer
        self.cat_model = cat_mode
        if cat_mode == 'DIRECT':
            self.linear = nn.Linear(2*embed_dim, num_class)
        else:
            self.linear = nn.Linear(embed_dim, num_class)
        self.init_weights()
        
    def forward(self, data_pre, data_post, len_pre, len_post):
        out_pre = self.embed(data_pre)
        out_pre = torch.sum(out_pre, dim=1)
        out_pre /= len_pre.view(len_pre.size()[0],1).expand_as(out_pre).float()

        out_post = self.embed(data_post)
        out_post = torch.sum(out_post, dim=1)
        out_post /= len_post.view(len_post.size()[0],1).expand_as(out_post).float()

        if self.cat_model == 'DIRECT':
            out = torch.cat((out_pre, out_post), 1)  
        else:
            out = out_pre.add(out_post)
            out = torch.div(out, 2.0)

        logit = self.linear(out)
        return F.log_softmax(logit, dim=1)
    
    def init_weights(self):
        # Use some specific initialization schemes
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.uniform_(self.linear.bias)


class NeuralNetwork(nn.Module):
    '''
    Neural Network classification model
    '''
    def __init__(self, embed_layer, embed_dim, num_class, hidden_dim, cat_mode='DIRECT'):
        super(NeuralNetwork, self).__init__()
        
        self.embed = embed_layer
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.cat_model = cat_mode
        if cat_mode == 'DIRECT':
            self.linear1 = nn.Linear(2*embed_dim, hidden_dim)
        else:
            self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_class)
        self.init_weights()
        
    def forward(self, data_pre, data_post, len_pre, len_post):
        out_pre = self.embed(data_pre)
        out_pre = torch.sum(out_pre, dim=1)
        out_pre /= len_pre.view(len_pre.size()[0],1).expand_as(out_pre).float()

        out_post = self.embed(data_post)
        out_post = torch.sum(out_post, dim=1)
        out_post /= len_post.view(len_post.size()[0],1).expand_as(out_post).float()

        if self.cat_model == 'DIRECT':
            out = torch.cat((out_pre, out_post), 1)
        else:
            out = out_pre.add(out_post)
            out = torch.div(out,2)

        z1 = self.linear1(out)
        a1 = torch.relu(z1)
        logit = self.linear2(a1)
        return F.log_softmax(logit, dim=1)
    
    def init_weights(self):
        # Use some specific initialization schemes
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.uniform_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.uniform_(self.linear2.bias)
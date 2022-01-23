from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from torch import optim
from torch.autograd import Variable
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalDiscriminator(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        
        self.l0 = nn.Linear(32, 32)
        self.l1 = nn.Linear(32, 32)

        self.l2 = nn.Linear(512, 1)
    def forward(self, y, M, data):

        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        # h0 = Variable(data['feats'].float()).cuda()
        batch_num_nodes = data['num_nodes'].int().numpy()
        M, _ = self.encoder(M, adj, batch_num_nodes)
        # h = F.relu(self.c0(M))
        # h = self.c1(h)
        # h = h.view(y.shape[0], -1)
        h = torch.cat((y, M), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)

class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class Projhead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        #self.linear_shortcut = nn.Linear(input_dim, input_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x)# + self.linear_shortcut(x)


def mixup_data(x, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
     
    #lam = np.random.beta(alpha, alpha)
    lam = np.random.uniform(alpha, 1.0)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    return mixed_x

def mixup_geo_data(x, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
     
    #lam = np.random.beta(alpha, alpha)
    lam = np.random.uniform(alpha, 1.0)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    #mixed_x = torch.mul(torch.pow(x, lam),torch.pow( x[index,:], (1 - lam))) 
    mixed_x = torch.pow(x, lam)*torch.pow( x[index,:], (1 - lam))
    return mixed_x


def mixup_binary_data(x, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
     
    #lam = np.random.beta(alpha, alpha)
    lam = np.random.uniform(alpha, 1.0)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    
    ones = torch.ones_like(x)
    dropout_mask = torch.nn.functional.dropout(ones, p=1-lam, training=True, inplace=False)
    dropout_mask_complement = ones-dropout_mask
    mixed_x = torch.mul(x, dropout_mask)+ torch.mul(x[index,:],dropout_mask_complement) 
    return mixed_x

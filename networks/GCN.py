"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
#from dgl.nn.pytorch.conv import SAGEConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, bias=False, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden,  bias=False, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes,bias=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,g, features,noise,noise_d):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            if noise_d == i:
                h_noise= h+noise
            if noise_d <i:
                h_noise =  layer(g,h_noise)
        h = torch.cat((h_noise,h),1)
        return h
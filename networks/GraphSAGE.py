import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import SAGEConv



class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, bias=True,activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, bias=True,activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, bias=True, activation=None)) # activation None

        
    def forward(self, g, features,noise,noise_d):
        h = features
        h_noise = h
        for i,layer in enumerate(self.layers):
            h = layer(g, h)            
            if noise_d == i:
                h_noise= h+noise
            if noise_d <i:
                h_noise =  layer(g,h_noise)
        h = torch.cat((h_noise,h),1)
        return h
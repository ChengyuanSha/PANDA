import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn


class GCNStack(nn.Module):
    ''' The GCN Model '''

    def __init__(self, input_dim, hidden_dim1, dropout=0.5):
        super(GCNStack, self).__init__()
        # graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(input_dim, hidden_dim1))
        # layer norm
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim1))
        # # post-message-passing
        # self.post_mp = nn.Sequential(
        #     nn.Linear(hidden_dim2, hidden_dim2),
        #     nn.Linear(hidden_dim2, output_dim))

        self.dropout = dropout
        self.num_layers = 1

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            # add dropout layer
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:  # except last layer
                x = self.lns[i](x)

        # x = self.post_mp(x)
        return emb, F.log_softmax(x, dim=1)
        # return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        ''' Define loss functions in NN, L2 norm regularization is implemented in the SGD optimizer weight decay  '''
        # return  F.nll_loss(pred, label) # nn.CrossEntropyLoss()(pred, label)
        return nn.CrossEntropyLoss()(pred, label)



import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn

class GAT(torch.nn.Module):
    def __init__(self, dataset):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        self.conv1 = pyg_nn.GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = pyg_nn.GATConv(self.hid * self.in_head, dataset.num_classes, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, x, edge_index):

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        emb = x
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return emb, F.log_softmax(x, dim=1)


    def loss(self, pred, label):
        ''' Define loss functions in NN, L2 norm regularization is implemented in the SGD optimizer weight decay  '''
        # return  F.nll_loss(pred, label) # nn.CrossEntropyLoss()(pred, label)
        return nn.CrossEntropyLoss()(pred, label)




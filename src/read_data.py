import torch
from community import community_louvain
from torch_geometric.data import Data, InMemoryDataset
import random
import networkx as nx
import math
import pandas as pd
from torch_geometric.utils import from_networkx
import numpy as np


def read():
    ''' read the input data into the pyG Data class format '''
    with open("../data/HMIN_edgelist.csv", 'r') as data:
        G = nx.parse_edgelist(data, delimiter=',', create_using=nx.Graph(), nodetype=int)
    # read autism df
    autism_df = pd.read_csv('../data/labeled_genes.csv')
    autism_df = autism_df.drop_duplicates(subset='entrez_id', keep="last")

    # get the node features
    # G = G.subgraph(autism_nodes)
    edge_index = get_edge_index(G)

    x = get_node_features(G)

    train_mask, test_mask, y = get_train_test_label(G, autism_df)

    num_classes = len(np.unique(y))

    # train_mask, test_mask, num_classes, y = get_train_test_mask(G)

    # make masks
    # disease_mask = list(zip(disease_position, [1 for _ in disease_position]))
    # healthy_mask = list(zip(disease_position, [0 for _ in disease_position]))
    # total_mask = disease_mask + healthy_mask
    # random.shuffle(total_mask)

    # n = x.shape[0]
    # randomassort = list(range(n))
    # random.shuffle(randomassort)
    # max_train = math.floor(len(randomassort) * .1)
    # train_mask_idx = torch.tensor(randomassort[:max_train])
    # test_mask_idx = torch.tensor(randomassort[max_train:])
    # train_mask = torch.zeros(n);
    # test_mask = torch.zeros(n)
    # train_mask.scatter_(0, train_mask_idx, 1)
    # test_mask.scatter_(0, test_mask_idx, 1)
    # train_mask = train_mask.type(torch.bool)
    # test_mask = test_mask.type(torch.bool)

    # data.x = x
    # data.y = y

    data = Data(edge_index=edge_index, x=x, y=y, train_mask=train_mask, test_mask=test_mask,
                num_classes=num_classes)  # .t().contiguous()

    return data


def get_node_features(G):
    ''' use hand designed node features or not'''
    x = torch.eye(G.number_of_nodes(), dtype=torch.float)
    # # feature: node degree
    # degrees = torch.tensor([val for (node, val) in G.degree()], dtype=torch.float)
    # # closeness
    # closeness = torch.tensor([val for (node, val) in nx.closeness_centrality(G).items()], dtype=torch.float)
    # #  Betweenness
    # betweenness =  torch.tensor([val for (node, val) in nx.betweenness_centrality(G).items()], dtype=torch.float)
    # # feature: eigenvector_centrality
    # ec = torch.tensor([val for (node, val) in nx.eigenvector_centrality(G).items()], dtype=torch.float)
    # # feature: page rank
    # pr = torch.tensor([val for (node, val) in nx.pagerank(G, alpha=0.9).items()], dtype=torch.float)
    #
    # x = torch.stack((degrees, closeness, betweenness, pr, ec)).t()
    return x


def get_edge_index(G):
    ''' Get Coordinate of the edge index '''
    adj = nx.to_scipy_sparse_matrix(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def get_train_test_label(G, autism_df):
    ''' Get the training and testing mask '''
    # get the labeled autism nodes position in the node list
    autism_nodes = autism_df['entrez_id'].to_numpy()
    nodes = np.array(G.nodes())
    labeled_index = np.in1d(nodes, autism_nodes)
    y = torch.tensor(autism_df['label'].to_list())

    # divide into more classes according to the confidence score
    autism_df['label'][autism_df['confidence'] == 0.75] = 2
    autism_df['label'][autism_df['confidence'] == 0.5] = 3

    # initialize
    y_label = np.zeros(nodes.shape[0], dtype=np.compat.long)
    train_mask = np.zeros(nodes.shape[0], dtype=bool)
    test_mask = np.zeros(nodes.shape[0], dtype=bool)
    y_label[:] = 4  # temporarily set class 4 to unlabeled data

    # y_label[labeled_index] = torch.tensor(autism_df['label'].to_list())
    temp_label = autism_df['label'].to_numpy()

    y_label[labeled_index] = temp_label

    y_index = torch.randperm(y.shape[0])
    max_train = int(len(y) * 0.8)

    # Using multiple levels of boolean index mask to assign train and test masks
    train_mask.flat[np.flatnonzero(labeled_index)[y_index[:max_train]]] = True
    test_mask.flat[np.flatnonzero(labeled_index)[y_index[max_train:]]] = True

    return torch.tensor(train_mask, dtype=torch.bool),  \
           torch.tensor(test_mask, dtype=torch.bool), torch.tensor(y_label, dtype=torch.long)


if __name__ == '__main__':
    data = read()

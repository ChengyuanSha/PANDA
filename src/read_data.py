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
    # autism df
    autism_df = pd.read_csv('../data/labeled_genes.csv')
    autism_df = autism_df.drop_duplicates(subset='entrez_id', keep="last")
    autism_nodes = autism_df['entrez_id'].to_list()

    # construct PyG style edge COO list
    # source_nodes, target_nodes = [], []
    # for edge in nx.generate_edgelist(G, data=False):
    #     n1, n2 = edge.split(" ")
    #     source_nodes.append(int(n1))
    #     target_nodes.append(int(n2))
    # edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.int64)
    #    data = from_networkx(G)
    G = G.subgraph(autism_nodes)
    edge_index = get_edge_index(G)

    x = get_node_features(G)

    train_mask, test_mask, y = get_train_test_mask2(autism_df)
    num_classes = len(torch.unique(y))

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
    ''' hand designed node features '''
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


def get_train_test_mask2(autism_df):
    ''' Get the training and testing mask '''
    y = torch.tensor(autism_df['label'].to_list())

    train_mask = torch.zeros(y.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(y.shape[0], dtype=torch.bool)

    y_index = torch.randperm(y.shape[0])

    max_train = int(len(y) * 0.8)

    train_mask[y_index[:max_train]] = True
    test_mask[y_index[max_train:]] = True

    return train_mask, test_mask, y


# def get_train_test_mask(G):
#     # Compute communities.
#     partition = community_louvain.best_partition(G)
#     y = torch.tensor([partition[i] for i in G.nodes])
#     # number of classes, +2 labeled ones
#     unique_labels = torch.unique(y)
#     num_classes = len(unique_labels) + 2
#     max_class = torch.max(unique_labels)
#
#     # init mask
#     train_mask = torch.zeros(y.shape[0], dtype=torch.bool)
#     test_mask = torch.zeros(y.shape[0], dtype=torch.bool)
#
#     autism_df = pd.read_csv('labeled_genes.csv')
#     # autism
#     disease_label = autism_df['entrez_id'][autism_df['label'] == 1].tolist()
#     healthy_label = autism_df['entrez_id'][autism_df['label'] == 0].tolist()
#
#     # update labels
#     disease_position = []
#     for c, i in enumerate(disease_label):
#         idx = list(G.nodes).index(i)
#         disease_position.append(idx)
#         y[idx] = 1
#     # update labeled class
#     y[disease_position] = max_class + 1
#
#     healthy_position = []
#     for c, i in enumerate(healthy_label):
#         idx = list(G.nodes).index(i)
#         healthy_position.append(idx)
#         y[idx] = 0
#     # update labeled class
#     y[disease_position] = max_class + 2
#
#     positions = disease_position + healthy_position
#     random.shuffle(positions)
#     max_train = int(len(positions) * 0.8)
#
#     train_mask[positions[:max_train]] = True
#     test_mask[positions[max_train:]] = True
#
#     return train_mask, test_mask, num_classes, y


if __name__ == '__main__':
    data = read()

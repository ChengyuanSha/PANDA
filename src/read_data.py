import torch
from community import community_louvain
from torch_geometric.data import Data, InMemoryDataset
import random
import networkx as nx
import math
import pandas as pd
from torch_geometric.utils import from_networkx
import numpy as np
import os

script_dir = os.path.dirname(__file__)  # absolute dir the script is in

def read():
    """ read the input data into the pyG Data class format """
    edge_list_path = os.path.join(script_dir, "../data/HMIN_edgelist.csv")
    autism_df_path = os.path.join(script_dir, "../data/labeled_genes.csv")

    with open(edge_list_path, 'r') as data:
        G = nx.parse_edgelist(data, delimiter=',', create_using=nx.Graph(), nodetype=int)

    # read autism df
    autism_df = pd.read_csv(autism_df_path)
    autism_df = autism_df.drop_duplicates(subset='entrez_id', keep="last")
    autism_df = autism_df.reset_index()

    # get the node features
    # G = G.subgraph(autism_nodes)
    edge_index = get_edge_index(G)

    x = get_node_features(G)

    train_mask, test_mask, y = get_train_test_label(G, autism_df)

    num_classes = len(np.unique(y))

    data = Data(edge_index=edge_index, x=x, y=y, train_mask=train_mask, test_mask=test_mask,
                num_classes=num_classes)  # .t().contiguous()

    return data


def get_node_features(G):
    """ use hand designed node features or not """
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
    """ Get Coordinate of the edge index """
    adj = nx.to_scipy_sparse_matrix(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def get_train_test_label(G, autism_df, sample_balanced_class=True, assign_unlabeled_using_community=True):
    """ Get the training, testing mask, and y labels """
    # get the labeled autism nodes position in the node list
    autism_nodes = autism_df['entrez_id'].to_numpy()
    all_nodes = np.array(G.nodes())

    # Find the labeled index in all_nodes, all_nodes are unique
    labeled_index = [np.where(all_nodes == i)[0][0] for i in autism_nodes]
    label_mask = np.zeros(all_nodes.shape[0], dtype=bool)
    label_mask[labeled_index] = True

    # process classes based on confidence score
    # autism_df['label'][autism_df['confidence'] == 0.75] = 0
    autism_df['label'][autism_df['confidence'] == 0.5] = 0

    # initialize y_label, train_mask, test_mask to whole graph size
    y_label = np.zeros(all_nodes.shape[0], dtype=np.compat.long)
    y_label[:] = 4  # temporarily set class 4 to unlabeled data
    train_mask = np.zeros(all_nodes.shape[0], dtype=bool)
    test_mask = np.zeros(all_nodes.shape[0], dtype=bool)

    # set labeled node in graph
    y_label[labeled_index] = autism_df['label'].to_numpy()

    # use communities info to assign a label to unlabelled data
    if assign_unlabeled_using_community:
        partion = community_louvain.best_partition(G)
        for c, i in enumerate(all_nodes):
            if c not in labeled_index:
                y_label[c] = partion[i] + 2 # skip first 2 classes

    # randomly assign train test masks
    y_length = len(autism_df.index)
    y_index = np.arange(y_length)

    if sample_balanced_class:
        # random sample 150 samples for each class
        train_index = np.array([np.random.choice(autism_df.index[autism_df['label']==i].tolist(), 150, replace=False) for i in range(2)]).flatten()
        test_index = np.array([i for i in y_index if i not in train_index])
    else:
        # 75% train, 25% test
        max_train = int(y_length * 0.85)
        permutated_list = np.random.permutation(y_index)
        train_index = permutated_list[:max_train]
        test_index = permutated_list[max_train:]
        # train_index = np.append(train_index, test_index[:200])

    # Using multiple levels of boolean index mask to assign train and test masks
    # 1. index to labelled indexes 2. assign to train/test mask to True
    train_mask.flat[np.flatnonzero(label_mask)[train_index]] = True
    test_mask.flat[np.flatnonzero(label_mask)[test_index]] = True

    return torch.tensor(train_mask, dtype=torch.bool),  \
           torch.tensor(test_mask, dtype=torch.bool), torch.tensor(y_label, dtype=torch.long)


if __name__ == '__main__':
    data = read()

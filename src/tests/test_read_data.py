import unittest
from src.read_data import get_train_test_label
import networkx as nx
import pandas as pd
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_something(self):
        with open("../../data/HMIN_edgelist.csv", 'r') as data:
            G = nx.parse_edgelist(data, delimiter=',', create_using=nx.Graph(), nodetype=int)
        autism_df = pd.read_csv('../../data/labeled_genes.csv')
        autism_df = autism_df.drop_duplicates(subset='entrez_id', keep="last")
        train_mask, test_mask, y = get_train_test_label(G, autism_df)
        autism_nodes = autism_df['entrez_id'].to_numpy()
        nodes = np.array(G.nodes())
        labeled_index = np.in1d(nodes, autism_nodes)

        self.assertTrue(np.all(np.in1d(np.where(train_mask), np.where(labeled_index))))
        self.assertTrue(np.all(np.in1d(np.where(test_mask), np.where(labeled_index))))
        self.assertTrue(np.all(np.in1d(np.where(y!=4), np.where(labeled_index))))


if __name__ == '__main__':
    unittest.main()

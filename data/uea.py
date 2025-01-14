#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================
from torch.utils.data import Dataset
import numpy as np
from data.config import register_dataset
from data.utils.uealabelmapping import label_mapping
from utils import Adjacency, NodeFeat
from data.utils.adjacency import adj_matrix
from data.utils.node import node_feature

import torch
import time

class Miss():
    def info(self, s):
        pass


@register_dataset
class UEADataset(Dataset):
    def __init__(self,
                 name: str,
                 path: str,
                 train: bool = False,
                 logger=None,
                 adjName="complete",
                 nodeName="raw",
                 fs=None,
                 bands=None):
        split = "_TRAIN" if train else "_TEST"
        # path = os.path.expanduser("~/data/UEA") 
        path = os.path.join(current_path, "..", "dataset/Multivariate2018_npz")
        data_path = os.path.join(path, name)
        data_file = os.path.join(data_path, name + split + ".npz")
        train = np.load(data_file)
        self.X, self.y = train["X"], train["y"]
        self.X = self.X.transpose(0,2,1)
        self.dict = label_mapping[name]
        self.logger = logger if logger else Miss()
        self.fs = fs
        self.bands = bands

        adj_path = os.path.join(data_path, adjName + split + ".pt")
        if not os.path.exists(adj_path):
            self.adj = self.cal_adj(adjName)
        else:
            self.adj = torch.load(adj_path)

        self.node = self.cal_node(nodeName)


    def cal_adj(self, ADJ_MATRIX):
        self.logger.info("waiting for calculating adj matrix...")
        start = time.time()
        adjacency = adj_matrix[ADJ_MATRIX]
        ADJ = Adjacency()
        ADJ.reset()
        ADJ.set_adj(adjacency)
        adj = [ADJ.adj(torch.tensor(input, dtype=torch.float)) for input in self.X]
        self.logger.info(f"calculate DONE and using {time.time()-start} seconds")
        return adj
    
    def cal_node(self, NODE_):
        self.logger.info("waiting for calculating node...")
        start = time.time()
        node_feat = node_feature[NODE_]
        NODE = NodeFeat()
        NODE.reset()
        NODE.set_node_feat(node_feat)
        if self.fs != 0 and self.bands is not None:
            node = [NODE.node(torch.tensor(input, dtype=torch.float), self.fs, self.bands) for input in self.X]
        elif self.fs != 0:
            node = [NODE.node(torch.tensor(input, dtype=torch.float), self.fs, None) for input in self.X]
        else:
            node = [NODE.node(torch.tensor(input, dtype=torch.float), None, None) for input in self.X]
        self.logger.info(f"calculate DONE and using {time.time()-start} seconds")
        return node


    def num_nodes(self):
        return self.X.shape[1]

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, index):
        x = self.X[index].astype("float32")
        y = self.dict[self.y[index]]
        adj = self.adj[index]
        node = self.node[index]
        return index, x, y, adj, node

if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    import torch
    name = "DuckDuckGeese"
    path = os.path.expanduser("~/data/UEA")
    traindataset = UEADataset(name, path, True)
    print(traindataset.num_nodes())
    # np.set_printoptions(threshold=np.inf)
    # idx, x, y = traindataset.__getitem__(11)
    # print(x.shape, y)
    # print(traindataset.__getitem__(0)[1].shape)
    # testdataset = UEADataset(name, path, False)
    # trainloader = DataLoader(traindataset, 2)
    # testloader = DataLoader(testdataset, 2)
    # trainX = torch.concat([x for x, _ in trainloader])
    # testX = torch.concat([x for x, _ in testloader])
    # print(trainX.shape)
    # print(testX.shape)
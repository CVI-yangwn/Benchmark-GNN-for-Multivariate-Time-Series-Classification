import torch
import dgl
from typing import Callable, Optional, Tuple, List
import os
from data.utils.node import raw

class Lambda():
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.lambdas: torch.Tensor = None
    
    def set_path(self, path: Optional[str]):
        self.path = path
        return self

    def reset(self) -> None:
        self.lambdas = None
        self.path = None

    def lambda_max(self, idx: Optional[torch.Tensor] = None,
            postfix: str = "_TRAIN.pt") -> torch.Tensor:
        p = f"{self.path}{postfix}"
        if self.lambdas is not None:
            return self.lambdas[idx]
        if os.path.exists(p):
            self.lambdas = torch.load(p, map_location=torch.device("cpu"))
            return self.lambdas[idx]
        return None

lambda_train = Lambda()
lambda_test = Lambda()

class Adjacency():
    def __init__(self, adjacency: Optional[Callable] = None,
                 path: Optional[str] = None) -> None:
        self.path = path
        self.adjacency = adjacency
        self.feat: torch.Tensor = None
    
    def set_path(self, path: str):
        self.path = path
        return self

    def path_not_none(self) -> bool:
        return self.path is not None

    def path_exists(self) -> bool:
        return self.path_not_none() and os.path.exists(self.path)

    def set_adj(self, adjacency: Callable):
        self.adjacency = adjacency
        return self

    def reset(self) -> None:
        self.feat = None
        self.path = None
        self.adjacency = None

    def adj(self, x: Optional[torch.Tensor] = None,
            idx: Optional[torch.Tensor] = None,
            postfix: str = "_TRAIN.pt") -> torch.Tensor:
        p = f"{self.path}{postfix}"

        if "COM" in p:
            return torch.ones(x.shape[0], x.shape[1], x.shape[1])
        if self.feat is not None:
            return self.feat[idx]
        if os.path.exists(p):
            self.feat = torch.load(p, map_location=torch.device("cpu"))
            return self.feat[idx]
        # @JM:adj
        return self.adjacency(x)

# train_adj = Adjacency()
# test_adj = Adjacency()

class NodeFeat():
    def __init__(self, node_feat: Optional[Callable] = None,
                 path: Optional[str] = None) -> None:
        self.path = path
        self.node_feat = node_feat
        self.nodes: torch.Tensor = None
    
    def set_path(self, path: str):
        self.path = path
        return self
    
    def path_not_none(self) -> bool:
        return self.path is not None
    
    def path_exists(self) -> bool:
        return self.path_not_none() and os.path.exists(self.path)

    def set_node_feat(self, node_feat: Callable):
        self.node_feat = node_feat
        return self
    
    def reset(self) -> None:
        self.nodes = None
        self.path = None
        self.node_feat = None
    
    def node(self, x: Optional[torch.Tensor] = None,
             fs: Optional[int] = None,
             bands: Optional[List[Tuple[float, float]]] = None,
             idx: Optional[torch.Tensor] = None,
             postfix: str = "_TRAIN.pt") -> torch.Tensor:
        p = f"{self.path}{postfix}"
        if self.nodes is not None:
            return self.nodes[idx]
        if os.path.exists(p):
            self.nodes = torch.load(p, map_location=torch.device("cpu"))
            return self.nodes[idx]
        if self.node_feat.__name__ == "raw":
            return x
        return torch.stack([self.node_feat(g, fs, bands) for g in x])

# train_node = NodeFeat()
# test_node = NodeFeat()

def dgl_graph(adj_matrix):
    adj_matrix = torch.Tensor(adj_matrix)
    src, dst = torch.nonzero(adj_matrix)
    g = dgl.graph((src, dst))
    return g
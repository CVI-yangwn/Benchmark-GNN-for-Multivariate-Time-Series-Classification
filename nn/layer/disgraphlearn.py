import torch
import torch.nn as nn

class DisGraphLearn(nn.Module):
    def __init__(self, num_nodes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_nodes, num_nodes, False)

    def dismatrix(self, nodes: torch.Tensor) -> torch.Tensor:
        res = torch.zeros(len(nodes), len(nodes))
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    res[i][j] = torch.pairwise_distance(nodes[i], nodes[j])
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adj = torch.stack([torch.relu(self.dismatrix(nodes)) for nodes in x])
        adj = torch.softmax(adj, 1)
        adj = self.linear(adj)
        return adj

if __name__ == "__main__":
    x = torch.randn(16, 64, 3000)
    graphlearn = DisGraphLearn(64)
    adj = graphlearn(x)
    print(adj.shape)
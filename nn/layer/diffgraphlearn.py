import torch
import torch.nn as nn

class DiffGraphLearn(nn.Module):
    def __init__(self, node_feat: int) -> None:
        super().__init__()
        self.linear = nn.Linear(node_feat, 1, False)  # length
        # self.norm = nn.BatchNorm1d(1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, t = x.shape
        diff = x.broadcast_to(v, b, v, t).permute(2, 1, 0, 3) - x
        diff = diff.permute(1, 0, 2, 3).view(b*v*v, t)
        # adj = torch.relu(self.norm(self.linear(diff.abs()))).squeeze(-1)
        adj = torch.relu((self.linear(diff.abs()))).squeeze(-1)
        adj = torch.softmax(adj.view(b, v, v), -1)
        return adj

if __name__ == "__main__":
    x = torch.randn(2, 2, 4)
    print(f"data\n{x}")
    graphlearn = DiffGraphLearn(4)
    adj = graphlearn(x)
    print(f"adj\n{adj}")
import torch
from torch import nn
from torch.nn import init

class ChebConv(nn.Module):
    def __init__(self, in_feats, out_feats, k, bias=True):
        super(ChebConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k = k
        self.W = nn.Parameter(torch.Tensor(k, in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()


    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.bias is not None:
            init.zeros_(self.bias)
        for i in range(self._k):
            init.xavier_normal_(self.W[i], init.calculate_gain("relu"))


    def forward(self, adj, feat, lambda_max=None):
        A = adj.to(feat)
        # assert not torch.any(torch.isnan(A)), f"Inf value"
        # num_nodes = A.shape[0]
        batch_size, num_nodes, _ = A.shape

        # in_degree = 1 / A.sum(dim=1).clamp(min=1).sqrt()
        in_degree = 1 / A.sum(dim=2).clamp(min=1).sqrt()
        # D_invsqrt = torch.diag(in_degree)
        D_invsqrt = torch.diag_embed(in_degree)
        # I = torch.eye(num_nodes).to(A)
        I = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1).to(A)
        L = I - D_invsqrt @ A @ D_invsqrt

        if lambda_max is None:
            # lambda_ = torch.eig(L)[0][:, 0]
            # lambda_max = lambda_.max()
            lambda_ = torch.linalg.eig(L)[0].real.detach()
            lambda_max = lambda_.max(-1)[0].view(batch_size, 1, 1)
        else:
            lambda_max = lambda_max.view(batch_size, 1, 1).to(feat)

        L_hat = 2 * L / lambda_max - I
        # Z = [torch.eye(num_nodes).to(A)]
        Z = [torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1).to(A)]
        for i in range(1, self._k):
            if i == 1:
                Z.append(L_hat)
            else:
                Z.append(2 * L_hat @ Z[-1] - Z[-2])

        # Zs = torch.stack(Z, 0)  # (k, n, n)
        Zs = torch.stack(Z, 1)

        # Zh = Zs @ feat.unsqueeze(0) @ self.W
        Zh = torch.matmul(Zs @ feat.unsqueeze(1), self.W)
        # Zh = Zh.sum(0)
        Zh = Zh.sum(1)

        if self.bias is not None:
            Zh = Zh + self.bias
        return Zh

    def __repr__(self):
        custom = f"ChebConv(in_feats={self._in_feats}, " \
                 f"out_feats={self._out_feats}, " \
                 f"k={self._k}, " \
                 f"bias={hasattr(self, 'bias')})"
        return custom

if __name__ == "__main__":
    import numpy as np
    import torch as th
    feat = th.randn(5, 6, 10)
    adj = th.tensor(np.identity(6)).unsqueeze(0).repeat(5, 1, 1).float()
    conv = ChebConv(10, 2, 2)
    conv.train()
    print(conv)
    res = conv(adj, feat)
    print(res.shape)
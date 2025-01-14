import torch as th
from torch import nn
from torch.nn import init
import numpy as np


class DynamicChebConv(nn.Module):

    def __init__(self, in_feats, out_feats, k, num_nodes, bias=True):
        super(DynamicChebConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k = k
        self.A = nn.Parameter(th.eye(num_nodes))
        self.bias_A = nn.Parameter(th.Tensor(1))
        self.W = nn.Parameter(th.Tensor(k, in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.trunc_normal_(self.A, 0, 0.3)
        init.trunc_normal_(self.bias_A, 0, 0.1)
        if self.bias is not None:
            init.zeros_(self.bias)
        for i in range(self._k):
            init.xavier_normal_(self.W[i], init.calculate_gain("relu"))

    def forward(self, adj, feat, lambda_max=None):
        A = self.A @ adj + self.bias_A
        A = th.relu(A)
        # num_nodes = A.shape[0]
        batch_size, num_nodes, _ = A.shape

        # in_degree = 1 / A.sum(dim=1).clamp(min=1).sqrt()
        in_degree = 1 / A.sum(dim=2).clamp(min=1).sqrt()
        # D_invsqrt = th.diag(in_degree)
        D_invsqrt = th.diag_embed(in_degree)
        # I = th.eye(num_nodes).to(A)
        I = th.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1).to(A)
        L = I - D_invsqrt @ A @ D_invsqrt

        if lambda_max is None:
            lambda_ = th.linalg.eig(L)[0].real
            # lambda_max = lambda_.max()
            lambda_max = lambda_.max(-1)[0].view(batch_size, 1, 1)

        L_hat = 2 * L / lambda_max - I
        # Z = [th.eye(num_nodes).to(A)]
        Z = [th.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1).to(A)]
        for i in range(1, self._k):
            if i == 1:
                Z.append(L_hat)
            else:
                Z.append(2 * L_hat @ Z[-1] - Z[-2])

        # Zs = th.stack(Z, 0)
        Zs = th.stack(Z, 1)

        # Zh = Zs @ feat.unsqueeze(0) @ self.W
        Zh = th.matmul(Zs @ feat.unsqueeze(1), self.W)
        # Zh = Zh.sum(0)
        Zh = Zh.sum(1)

        if self.bias is not None:
            Zh = Zh + self.bias
        return Zh

    def __repr__(self):
        custom = f"DynamicChebConv(in_feats={self._in_feats}, " \
                 f"out_feats={self._out_feats}, " \
                 f"k={self._k}, " \
                 f"num_nodes={self.A.shape[0]}, " \
                 f"bias={hasattr(self, 'bias')})"
        return custom


if __name__ == "__main__":
    import numpy as np
    import torch as th
    feat = th.randn(5, 6, 10)
    adj = th.tensor(np.identity(6)).unsqueeze(0).repeat(5, 1, 1).float()
    conv = DynamicChebConv(10, 2, 2, 6)
    print(conv)
    # res = conv(adj, feat)
    # print(res.shape)
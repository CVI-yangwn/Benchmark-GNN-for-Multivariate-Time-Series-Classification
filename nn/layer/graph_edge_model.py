import torch
import torch.nn as nn
import math

class GlobalProj(nn.Module):
    def __init__(self, num_nodes):
        super(GlobalProj, self).__init__()
        self.proj = nn.Conv1d(num_nodes, 1, 3, 1, 1)
        self.bn = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(num_nodes, 1, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x [B,V,T]
        return [B,T]
        '''
        res = x
        x = self.proj(x)
        x = self.bn(x)
        res = self.downsample(res)
        x = res + x
        x = self.relu(x)
        x = x.squeeze(1)
        return x

"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
        queried_values = queried_values.squeeze(2)

        return queried_values.contiguous()

class CrossAttn(nn.Module):
    """ cross attention Module"""
    def __init__(self, in_channels):
        super(CrossAttn, self).__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.lattn = LinearAttention()

        self.linear_k.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_q.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2. / in_channels))

    def forward(self, y, x):
        '''
        y [B,_,T]
        x [B,_,T]
        '''
        query = self.linear_q(y) # [B,_,T/2]
        key = self.linear_k(x) # [B,_,T/2]
        value = self.linear_v(x) # [B,_,T]
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale # [B,_,_]
        attn = self.attend(dots) # [B,_,_]
        out = torch.matmul(attn, value) # [B,_,T]
        # query = query.unsqueeze(2)
        # key = key.unsqueeze(2)
        # value = value.unsqueeze(2)
        # out = self.lattn(query, key, value)
        return out


# in_channels: 节点特征, num_classes: 节点数
# class GEM(nn.Module):
#     def __init__(self, in_feats, num_nodes):
#         super(GEM, self).__init__()
#         self.in_feats = in_feats
#         self.num_nodes = num_nodes
#         self.FAM = CrossAttn(self.in_feats)
#         self.ARM = CrossAttn(self.in_feats)
#         self.edge_proj = nn.Linear(in_feats, in_feats)
#         self.bn = nn.BatchNorm1d(self.num_nodes * self.num_nodes)

#         self.edge_proj.weight.data.normal_(0, math.sqrt(2. / in_feats))
#         self.bn.weight.data.fill_(1)
#         self.bn.bias.data.zero_()

#     def forward(self, node_feature, global_feature):
#         '''
#         node_feature [B,V,T]
#         global_feature [B,T]
#         '''
#         B, V, T = node_feature.shape
#         global_feature = global_feature.unsqueeze(1).repeat(1, V, 1) # [B,T] -> [B,1,T] -> [B,V,T]
#         feat = self.FAM(node_feature, global_feature) # [B,V,T]
#         feat_end = feat.repeat(1, 1, V).view(B, -1, T) # [B,V*V,T]
#         feat_start = feat.repeat(1, V, 1).view(B, -1, T) # [B,V*V,T]
#         feat = self.ARM(feat_start, feat_end) # [B,V*V,T]
#         edge = self.bn(self.edge_proj(feat)) # [B,V*V,T]
#         return edge

class GEM(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GEM, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.FAM = CrossAttn(self.in_channels)
        self.ARM = CrossAttn(self.in_channels)
        self.edge_proj = nn.Linear(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(self.num_classes * self.num_classes)

        self.edge_proj.weight.data.normal_(0, math.sqrt(2. / in_channels))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, class_feature, global_feature):
        B, N, D, C = class_feature.shape
        global_feature = global_feature.repeat(1, N, 1).view(B, N, D, C)
        feat = self.FAM(class_feature, global_feature)
        feat_end = feat.repeat(1, 1, N, 1).view(B, -1, D, C)
        feat_start = feat.repeat(1, N, 1, 1).view(B, -1, D, C)
        feat = self.ARM(feat_start, feat_end)
        edge = self.bn(self.edge_proj(feat))
        return edge

if __name__ == "__main__":
    node_feature = torch.randn(1, 1345, 270)
    # global_feature = torch.randn(4, 64)
    gproj = GlobalProj(1345)
    global_feature = gproj(node_feature)
    gem = GEM(270, 1345)
    edge: torch.Tensor = gem(node_feature, global_feature)
    ...
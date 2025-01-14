#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================

from nn.layer.dynamicchebconv import DynamicChebConv
from nn.config import register_model

import torch
import torch.nn as nn

@register_model
class DGCNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, k, num_nodes, num_class):
        super(DGCNN, self).__init__()
        self.chebconv = DynamicChebConv(in_feats, hidden_feats, k, num_nodes)
        self.mlp = nn.Sequential(
            nn.Linear(num_nodes * hidden_feats, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_class),
        )
        return

    def forward(self, x):
        batch_size, num_nodes, _ = x.shape
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1).to(x)
        x = self.chebconv(A, x)
        x = torch.relu(x)

        batch_size, num_nodes, node_feats = x.shape
        x = torch.reshape(x, [batch_size, num_nodes * node_feats])
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    from nn.config import models
    print(models)
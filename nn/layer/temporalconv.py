import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.__padding = kernel_size - 1  # 2
        self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=0)
    
    def forward(self, x):
        # if self.__padding != 0:
        x = F.pad(x, (self.__padding, 0, 0, 0))  # left, right, up, down
        return self.conv(x)


class TemporalConv(nn.Module):
        """
        https://github.com/LeronQ/STGCN-Pytorch/blob/main/stgcn.py
        Neural network block that applies a temporal convolution to each node of
        a graph in isolation.
        """

        def __init__(self, in_channels, out_channels, kernel_size=3):
            """
            :param in_channels: Number of input features at each node in each time
            step.
            :param out_channels: Desired number of output channels at each node in
            each time step.
            :param kernel_size: Size of the 1D temporal kernel.
            """
            super().__init__()
            self.conv1 = CausalConv2d(in_channels, out_channels, kernel_size)
            self.conv2 = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.conv3 = CausalConv2d(in_channels, out_channels, kernel_size)

        def forward(self, X):
            """
            :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
            num_features=in_channels)
            :return: Output data of shape (batch_size, num_nodes,
            num_timesteps_out, num_features_out=out_channels)
            """
            # Convert into NCHW format for pytorch to perform convolutions.
            X = X.permute(0, 3, 1, 2)
            temp = self.conv1(X) + self.conv2(X) 
            ac = torch.sigmoid(self.conv3(X))
            out = torch.relu(torch.mul(ac, temp) + temp)
            out = out.permute(0, 2, 3, 1)
            return out


class TemporalMaxPool(nn.Module):
    def __init__(self, output_size: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool1d(output_size)

    def forward(self, x: torch.Tensor):
        batch_size, num_nodes, time_points, feats = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(-1, feats, time_points)
        x = self.pool(x)
        x = x.contiguous().view(batch_size, num_nodes, feats, -1)
        x = x.permute(0, 1, 3, 2)
        return x
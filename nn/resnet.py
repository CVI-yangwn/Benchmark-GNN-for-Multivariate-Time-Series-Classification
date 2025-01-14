import torch.nn as nn
import torch

class ResNet(nn.Module):
    def __init__(self, input_shape, nb_classes):
        super(ResNet, self).__init__()
        n_feature_maps = 64

        self.conv_x = nn.Conv1d(input_shape, n_feature_maps, kernel_size=8, padding='same')
        self.bn_x = nn.BatchNorm1d(n_feature_maps)

        self.conv_y = nn.Conv1d(n_feature_maps, n_feature_maps, kernel_size=5, padding='same')
        self.bn_y = nn.BatchNorm1d(n_feature_maps)

        self.conv_z = nn.Conv1d(n_feature_maps, n_feature_maps, kernel_size=3, padding='same')
        self.bn_z = nn.BatchNorm1d(n_feature_maps)

        self.shortcut_y = nn.Conv1d(input_shape, n_feature_maps, kernel_size=1, padding='same')
        self.shortcut_y_bn = nn.BatchNorm1d(n_feature_maps)

        self.output_block_1_relu = nn.ReLU()

        self.conv_x2 = nn.Conv1d(n_feature_maps, n_feature_maps * 2, kernel_size=8, padding='same')
        self.bn_x2 = nn.BatchNorm1d(n_feature_maps * 2)

        self.conv_y2 = nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=5, padding='same')
        self.bn_y2 = nn.BatchNorm1d(n_feature_maps * 2)

        self.conv_z2 = nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=3, padding='same')
        self.bn_z2 = nn.BatchNorm1d(n_feature_maps * 2)

        self.shortcut_y2 = nn.Conv1d(n_feature_maps, n_feature_maps * 2, kernel_size=1, padding='same')
        self.shortcut_y2_bn = nn.BatchNorm1d(n_feature_maps * 2)

        self.output_block_2_relu = nn.ReLU()

        self.conv_x3 = nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=8, padding='same')
        self.bn_x3 = nn.BatchNorm1d(n_feature_maps * 2)

        self.conv_y3 = nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=5, padding='same')
        self.bn_y3 = nn.BatchNorm1d(n_feature_maps * 2)

        self.conv_z3 = nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=3, padding='same')
        self.bn_z3 = nn.BatchNorm1d(n_feature_maps * 2)

        self.shortcut_y3_bn = nn.BatchNorm1d(n_feature_maps * 2)

        self.output_block_3_relu = nn.ReLU()

        self.gap_layer = nn.AdaptiveAvgPool1d(1)
        
        self.fc_layer = nn.Linear(n_feature_maps*2,nb_classes)
        
    def forward(self,x):
        
        # BLOCK 1
        conv_x_out=self.conv_x(x)
        bn_x_out=self.bn_x(conv_x_out)
        bn_x_out = torch.relu(bn_x_out)
        
        conv_y_out=self.conv_y(bn_x_out)
        bn_y_out=self.bn_y(conv_y_out)
        bn_y_out = torch.relu(bn_y_out)
        
        conv_z_out=self.conv_z(bn_y_out)
        bn_z_out=self.bn_z(conv_z_out)
        
        shortcut_y_out=self.shortcut_y(x)
        shortcut_y_bn_out=self.shortcut_y_bn(shortcut_y_out)
        
        output_block_1_add=torch.add(shortcut_y_bn_out,bn_z_out)
        
        output_block_1_relu=self.output_block_1_relu(output_block_1_add)
        
         # BLOCK 2
         
        conv_x_out_2=self.conv_x2(output_block_1_relu)
        
        bn_x_out_2=self.bn_x2(conv_x_out_2)
        bn_x_out_2 = torch.relu(bn_x_out_2)
        
        conv_y_out_2=self.conv_y2(bn_x_out_2)
        
        bn_y_out_2=self.bn_y2(conv_y_out_2)
        bn_y_out_2 = torch.relu(bn_y_out_2)
        
        conv_z_out_2=self.conv_z2(bn_y_out_2)
        
        bn_z_out_2=self.bn_z2(conv_z_out_2)
        
        shortcut_y_out_2=self.shortcut_y2(output_block_1_relu)
        shortcut_y_bn_out_2=self.shortcut_y2_bn(shortcut_y_out_2)
        
        output_block_2_add=torch.add(shortcut_y_bn_out_2,bn_z_out_2)
        
        output_block_2_relu=self.output_block_3_relu(output_block_2_add)
        
        # BLOCK 3
        
        conv_x_out_3=self.conv_x3(output_block_2_relu)
        
        bn_x_out_3=self.bn_x3(conv_x_out_3)
        bn_x_out_3 = torch.relu(bn_x_out_3)
        
        conv_y_out_3=self.conv_y3(bn_x_out_3)

        bn_y_out_3=self.bn_y2(conv_y_out_3)
        bn_y_out_3 = torch.relu(bn_y_out_3)
        
        conv_z_out_3=self.conv_z2(bn_y_out_3)
        
        bn_z_out_3=self.bn_z2(conv_z_out_3)
        
        shortcut_y_bn_out_3=self.shortcut_y3_bn(output_block_2_relu)
        
        output_block_3_add=torch.add(shortcut_y_bn_out_3,bn_z_out_3)
        
        output_block_3_relu=self.output_block_3_relu(output_block_3_add)

        # FINAL
        gap_out = self.gap_layer(output_block_3_relu)
        output = self.fc_layer(torch.squeeze(gap_out, dim=-1))

        return output
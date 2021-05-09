import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from poolings.GSAPool import GSAPool, NPool


class Config():
    def __init__(self):
        self.hidden = 128
        self.pooling_ratio = 0.8
        self.dropout = 0.5
        self.alpha = 0.6
        self.pooling_layer_type = GCNConv
        self.feature_fusion_type = GATConv


class GSANet(torch.nn.Module):
    def __init__(self, config, args):
        super(GSANet, self).__init__()
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.hidden = config.hidden
        self.pooling_ratio = config.pooling_ratio
        self.dropout = config.dropout
        self.alpha = config.alpha

        self.pooling_layer_type = config.pooling_layer_type
        self.feature_fusion_type = config.feature_fusion_type

        self.conv1 = GCNConv(self.num_features, self.hidden)
        self.conv2 = GCNConv(self.hidden, self.hidden)
        self.conv3 = GCNConv(self.hidden, self.hidden)

        # self.pool1 = GSAPool(self.hidden, self.pooling_ratio, self.alpha, self.pooling_layer_type, self.feature_fusion_type)
        # self.pool2 = GSAPool(self.hidden, self.pooling_ratio, self.alpha, self.pooling_layer_type, self.feature_fusion_type)
        # self.pool3 = GSAPool(self.hidden, self.pooling_ratio, self.alpha, self.pooling_layer_type, self.feature_fusion_type)
        self.pool1 = NPool(self.hidden, self.pooling_ratio)
        self.pool2 = NPool(self.hidden, self.pooling_ratio)
        self.pool3 = NPool(self.hidden, self.pooling_ratio)

        self.linear1 = torch.nn.Linear(self.hidden * 2, self.hidden)
        self.linear2 = torch.nn.Linear(self.hidden, self.hidden // 2)
        self.linear3 = torch.nn.Linear(self.hidden // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.log_softmax(self.linear3(x), dim=1)

        return x



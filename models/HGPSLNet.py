import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv
from poolings.HGPSLPool import GCN, HGPSLPool


class Config():
    def __init__(self):
        self.hidden = 128
        self.pooling_ratio = 0.8
        self.dropout = 0
        self.sample_neighbor = True
        self.sparse_attention = False
        self.structure_learning = True
        self.lamb = 1.0


class HGPSLNet(torch.nn.Module):
    def __init__(self, config, args):
        super(HGPSLNet, self).__init__()
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.hidden = config.hidden
        self.pooling_ratio = config.pooling_ratio
        self.dropout = config.dropout
        self.sample = config.sample_neighbor
        self.sparse = config.sparse_attention
        self.sl = config.structure_learning
        self.lamb = config.lamb

        self.conv1 = GCNConv(self.num_features, self.hidden)
        self.conv2 = GCNConv(self.hidden, self.hidden)
        self.conv3 = GCNConv(self.hidden, self.hidden)

        self.pool1 = HGPSLPool(self.hidden, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.hidden, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.linear1 = torch.nn.Linear(self.hidden * 2, self.hidden)
        self.linear2 = torch.nn.Linear(self.hidden, self.hidden // 2)
        self.linear3 = torch.nn.Linear(self.hidden // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None 

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.linear3(x), dim=-1)

        return x
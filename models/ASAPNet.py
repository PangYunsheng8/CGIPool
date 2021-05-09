import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.nn.pool import ASAPooling
# from poolings.ASAPPool import ASAPooling


class Config():
    def __init__(self):
        self.hidden = 128
        self.pooling_ratio = 0.8
        self.dropout = 0.5


class ASAPNet(torch.nn.Module):
    def __init__(self, config, args):
        super(ASAPNet, self).__init__()
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.hidden = config.hidden
        self.pooling_ratio = config.pooling_ratio
        self.dropout = config.dropout

        self.conv1 = GCNConv(self.num_features, self.hidden)
        self.conv2 = GCNConv(self.hidden, self.hidden)
        self.conv3 = GCNConv(self.hidden, self.hidden)

        self.pool1 = ASAPooling(self.hidden, ratio=self.pooling_ratio, GNN=GCNConv)
        self.pool2 = ASAPooling(self.hidden, ratio=self.pooling_ratio, GNN=GCNConv)
        self.pool3 = ASAPooling(self.hidden, ratio=self.pooling_ratio, GNN=GCNConv)

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
    
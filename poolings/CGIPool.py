from torch_geometric.nn import GCNConv, GATConv, LEConv, SAGEConv, GraphConv
from torch_geometric.data import Data
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops, remove_self_loops
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spspmm, coalesce
from torch_scatter import scatter_add, scatter
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, in_channels)
        self.fc2 = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.sigmoid(self.fc2(x))
        return x


class CGIPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, non_lin=torch.tanh):
        super(CGIPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.non_lin = non_lin
        self.hidden_dim = in_channels
        self.transform = GraphConv(in_channels, self.hidden_dim)
        self.pp_conv = GraphConv(self.hidden_dim, self.hidden_dim)
        self.np_conv = GraphConv(self.hidden_dim, self.hidden_dim)

        self.positive_pooling = GraphConv(self.hidden_dim, 1) 
        self.negative_pooling = GraphConv(self.hidden_dim, 1)
  
        self.discriminator = Discriminator(self.hidden_dim)
        self.loss_fn = nn.BCELoss()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))  

        x_transform = F.leaky_relu(self.transform(x, edge_index), 0.2)
        x_tp = F.leaky_relu(self.pp_conv(x, edge_index), 0.2)
        x_tn = F.leaky_relu(self.np_conv(x, edge_index), 0.2)
        s_pp = self.positive_pooling(x_tp, edge_index).squeeze()
        s_np = self.negative_pooling(x_tn, edge_index).squeeze()
 
        perm_positive = topk(s_pp, 1, batch)
        perm_negative = topk(s_np, 1, batch)
        x_pp = x_transform[perm_positive] * self.non_lin(s_pp[perm_positive]).view(-1, 1) 
        x_np = x_transform[perm_negative] * self.non_lin(s_np[perm_negative]).view(-1, 1)

        x_pp_readout = gap(x_pp, batch[perm_positive])
        x_np_readout = gap(x_np, batch[perm_negative])
        x_readout = gap(x_transform, batch)

        positive_pair = torch.cat([x_pp_readout, x_readout], dim=1) 
        negative_pair = torch.cat([x_np_readout, x_readout], dim=1)

        real = torch.ones(positive_pair.shape[0]).cuda()
        fake = torch.zeros(negative_pair.shape[0]).cuda()
        real_loss = self.loss_fn(self.discriminator(positive_pair), real)
        fake_loss = self.loss_fn(self.discriminator(negative_pair), fake)
        discrimination_loss = (real_loss + fake_loss) / 2

        score = (s_pp - s_np)#.squeeze()

        perm = topk(score, self.ratio, batch)
        x = x_transform[perm] * self.non_lin(score[perm]).view(-1, 1)
        batch = batch[perm]

        filter_edge_index, filter_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, filter_edge_index, filter_edge_attr, batch, perm, discrimination_loss
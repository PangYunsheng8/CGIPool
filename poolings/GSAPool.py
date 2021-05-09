import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import GCNConv, GATConv, LEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter
from torch_geometric.utils import softmax


class GeneralInfoScore(MessagePassing):
    def __init__(self):
        super(GeneralInfoScore, self).__init__(aggr='add')
    
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        # add self loop
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes)
        
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes)

        row, col = edge_index
        expand_deg = torch.zeros((edge_weight.size(0), ), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes, ), dtype=dtype, device=edge_index.device)

        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight):
        edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class UniqueInfoScore(object):
    def __call__(self, x, edge_index, edge_weight):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=edge_index.device)

        # score
        row, col = edge_index
        weights = x[row].mul(x[col]).sum(dim=-1)
        weights_sum = scatter_add(weights, row, dim=0, dim_size=x.size(0))
        weights_norm = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        weights_norm = weights_norm.pow(-1)
        weights_norm[weights_norm == float('inf')] = 0
        score = weights_sum.mul(weights_norm)

        # att
        # edge_index, _ = add_remaining_self_loops(edge_index, None, 0, x.size(0))
        # row, col = edge_index
        # weights = x[row].mul(x[col]).sum(dim=-1)
        # att = softmax(weights, row)

        return score#, att


class Smm(torch.nn.Module):
    def __init__(self, in_channels):
        super(Smm, self).__init__()
        self.in_channels = in_channels
    
    def forward(self, original_x, original_edge_index, perm):
        original_num_nodes = original_x.size(0)

        original_edge_index, _ = add_remaining_self_loops(original_edge_index, None, 0, original_num_nodes)
        topk_neighbor_edge_index = self.filter_topk_neighbor(original_edge_index, perm, original_x.size(0))
        row, col = topk_neighbor_edge_index

        # Max fusion
        x = scatter_add(original_x[col].T, row)
        x = x.T[perm]

        return x

    def filter_topk_neighbor(self, original_edge_index, perm, num_nodes=None):
        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        original_row, original_col = original_edge_index
        index = mask[original_row]
        mask = index >= 0 
        row = original_row[mask]
        col = original_col[mask]

        return torch.stack([row, col], dim=0)

    
class NPool(torch.nn.Module):
    def __init__(self, in_channels, ratio, GNN=None, negative_slope=0.2, lamb=0.7):
        super(NPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.GNN = GNN
        self.lamb = lamb

        self.att = nn.Linear(2 * in_channels, 1)
        self.gnn_score = LEConv(self.in_channels, 1)
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels, **kwargs)

    def reset_parameters(self):
        self.att.reset_parameters()
        self.gnn_score.reset_parameters()
        if self.GNN is not None:
            self.gnn_intra_cluster.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        
        N = x.size(0)

        edge_index, _ = add_remaining_self_loops(edge_index, None, fill_value=1, num_nodes=N)

        x_pool = x
        if self.GNN is not None:
            x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index, edge_weight=edge_attr)
        
        score = self.att(torch.cat([x_pool[edge_index[0]], x_pool[edge_index[1]]], dim=-1)).view(-1)
        score = F.leaky_relu(score, self.negative_slope)
        score_att = softmax(score, edge_index[1])
        v_j = x[edge_index[0]] * score_att.view(-1, 1)
        x = scatter(v_j, edge_index[1], dim=0, reduce='add')

        fitness = self.gnn_score(x, edge_index).sigmoid().view(-1)

        score[-N:] = 0
        score_sum = scatter(score, edge_index[1], dim=0, reduce='add')
        final_score = fitness - self.lamb * score_sum

        perm = topk(final_score, self.ratio, batch)
        x = x[perm] * final_score[perm].view(-1, 1)
        batch = batch[perm]
        
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm


class GSAPool(torch.nn.Module):
    def __init__(self, in_channels, ratio, alpha, pool_conv, fusion_conv,
                min_score=None, multiplier=1, non_lin=torch.tanh):
        super(GSAPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.alpha = alpha

        self.sbtl_layer = pool_conv(self.in_channels, 1)
        self.fbtl_layer = nn.Linear(self.in_channels, 1)
        self.fusion = fusion_conv(self.in_channels, self.in_channels, heads=1, concat=True)

        # self.calc_general_score = GeneralInfoScore()
        self.calc_unique_score = UniqueInfoScore()
        self.ammf = Smm(self.in_channels)

        self.min_score = min_score
        self.multiplier = multiplier
        self.non_lin = non_lin

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        # SBTL
        # score_s = self.sbtl_layer(x, edge_index).squeeze()
        # FBTL
        # score_f = self.fbtl_layer(x).squeeze()

        # score = score_s * self.alpha + score_f * (1 - self.alpha)
        # score = score.unsqueeze(-1) if score.dim() == 0 else score

        # if self.min_score is None:
        #     score = self.non_lin(score)
        # else:
        #     score = softmax(score, batch)

        x_unique_score = self.calc_unique_score(x, edge_index, edge_attr)
        score = torch.exp(-1 * x_unique_score)

        perm = topk(score, self.ratio, batch)

        # x = self.fusion(x, edge_index)
        # x = x[perm] * score[perm].view(-1, 1)
        x = self.ammf(x, edge_index, perm)
        # x = self.multiplier * x if self.multiplier != 1 else x  # ? 

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

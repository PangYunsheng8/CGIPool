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


class TwoHopNeighborhood(object):
    def __call__(self, data, include_self=False):
        edge_index = data.edge_index
        n = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, n, n, n)
        value.fill_(0)

        if include_self:
            edge_index = torch.cat([edge_index, index], dim=1)
        else:
            edge_index = index
        data.edge_index, _ = coalesce(edge_index, None, n, n) 

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


# class CalStructureAtt(object):
#     def __call__(self, original_x, edge_index, original_batch, new_edge_index):
        
#         sorted_topk_trans = self.cal_transition_matrix(original_x, edge_index, original_batch)

#         row, col = new_edge_index
#         weights_structure = self.js_divergence(sorted_topk_trans[row], sorted_topk_trans[col])

#         return weights_structure

#     def compute_rwr(self, adj, c=0.1):
#         d = torch.diag(torch.sum(adj, 1))
#         d_inv = d.pow(-0.5)
#         d_inv[d_inv == float('inf')] = 0
#         w_tilda = torch.matmul(d_inv, adj)
#         w_tilda = torch.matmul(w_tilda, d_inv)
#         q = torch.eye(w_tilda.size(0)) - c * w_tilda
#         q_inv = torch.inverse(q)
#         e = torch.eye(w_tilda.size(0))
#         r = (1 - c) * q_inv
#         r = torch.matmul(r, e)
#         return r

#     def cal_transition_matrix(self, x, edge_index, batch, k=8):
#         """
#         Compute random walk with restart score
#         """
#         num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
#         shift_cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
#         cum_num_nodes = num_nodes.cumsum(dim=0)

#         transition_matrix = torch.zeros((x.size(0), torch.max(num_nodes)), dtype=torch.float, device=x.device)
#         adj_graph = torch.zeros((x.size(0), x.size(0)))

#         row, col = edge_index
#         adj_graph[row, col] = 1
#         rwr_out = self.compute_rwr(adj_graph)

#         for idx_i, idx_j in zip(shift_cum_num_nodes, cum_num_nodes):
#             transition_matrix[idx_i: idx_j, :(idx_j - idx_i)] = rwr_out[idx_i: idx_j, idx_i: idx_j]

#         sorted_trans, _ = torch.sort(transition_matrix, descending=True)
#         sorted_topk_trans = sorted_trans[:, :k]

#         return sorted_topk_trans

#     def js_divergence(self, P, Q):
#         def kl_divergence(P, Q):
#             return (P * torch.log2(P / Q)).sum(dim=1)
#         P[P == 0] = 1e-15
#         Q[Q == 0] = 1e-15
#         P = P / P.sum(dim=1)[:, None]
#         Q = Q / Q.sum(dim=1)[:, None]

#         M = 0.5 * (P + Q)

#         js = 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)
#         js[js < 0] = 0

#         return js
class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.conv2 = SAGEConv(out_channels, in_channels)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, in_channels)
        self.fc2 = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.sigmoid(self.fc2(x))
        return x


class FGPool1(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, non_lin=torch.tanh):
        super(FGPool1, self).__init__()
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

        #f_pp, _ = filter_adj(edge_index, edge_attr, perm_positive, num_nodes=s_pp.size(0))
        #f_np, _ = filter_adj(edge_index, edge_attr, perm_negative, num_nodes=s_np.size(0))
        #x_pp = F.leaky_relu(self.pp_conv(x_pp, f_pp), 0.2)
        #x_np = F.leaky_relu(self.np_conv(x_np, f_np), 0.2)

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


class RedConv(torch.nn.Module):
    def __init__(self, in_channels, ratio, negative_slope=0.2, lamb=0.1):
        super(RedConv, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.lamb = lamb
        self.hidden_channels = in_channels

        self.att = nn.Linear(2 * in_channels, 1)
        self.gnn_score = GraphConv(self.in_channels, 1)

        self.transform_neighbor = GCNConv(self.in_channels, self.hidden_channels)
        self.key_linear = torch.nn.Linear(self.hidden_channels, 1)
        self.query_linear = torch.nn.Linear(self.hidden_channels, 1)

    def reset_parameters(self):
        self.att.reset_parameters()
        self.gnn_score.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        N = x.size(0)
        edge_index, _ = add_remaining_self_loops(edge_index, None, fill_value=1, num_nodes=N)

        x_transform = self.transform_neighbor(x=x, edge_index=edge_index, edge_weight=edge_attr)
        x_neighbor = x_transform[edge_index[1]]
        # build neighbor
        x_key_score = F.leaky_relu(self.key_linear(x_neighbor))
        x_key_score = softmax(x_key_score, edge_index[0])
        x_reweight_key = x_neighbor * x_key_score
        x_reweight_key = scatter(x_reweight_key, edge_index[0], dim=0, reduce='add')

        x_query_score = F.leaky_relu(self.query_linear(x_neighbor))
        x_query_score = softmax(x_query_score, edge_index[0])
        x_reweight_query = x_neighbor * x_query_score
        x_reweight_query = scatter(x_reweight_query, edge_index[0], dim=0, reduce='add')

        ker_error = torch.sum(torch.abs(x_reweight_key[edge_index[0]] - x_transform[edge_index[0]]), dim=1)
        query_error = torch.sum(torch.abs(x_reweight_query[edge_index[0]] - x_transform[edge_index[1]]), dim=1)
        red_score = ker_error - query_error

        # score = self.att(torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)).view(-1)
        # score = F.leaky_relu(score, self.negative_slope)

        fitness = self.gnn_score(x, edge_index).sigmoid().view(-1)

        red_score[-N:] = 0
        score_sum = scatter(red_score, edge_index[0], dim=0, reduce='add')
        final_score = fitness - self.lamb * score_sum

        return final_score

class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, Conv=RedConv, non_lin=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_lin = non_lin

        # refining layer
        # self.att_sl = Parameter(torch.Tensor(1, self.in_channels * 2))
        # nn.init.xavier_uniform_(self.att_sl.data)
        # self.neighbor_augment = TwoHopNeighborhood()
        # self.cal_struc_att = CalStructureAtt()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))  
        original_x, original_edge_index, original_batch = x, edge_index, batch

        score = self.score_layer(x, edge_index, edge_attr, batch).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_lin(score[perm]).view(-1, 1)
        batch = batch[perm]

        filter_edge_index, filter_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, filter_edge_index, filter_edge_attr, batch, perm

    # def cal_nhop_neighbor(self, x, edge_index, edge_attr, perm, n_hop=2):
    #     if edge_attr is None:
    #         edge_attr = torch.ones((edge_index.size(1), ), dtype=torch.float, device=edge_index.device)

    #     hop_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    #     hop_data = self.neighbor_augment(hop_data, include_self=False)
    #     for _ in range(n_hop - 2):
    #         hop_data = self.neighbor_augment(hop_data, include_self=True)
    #     hop_edge_index, hop_edge_attr = hop_data.edge_index, hop_data.edge_attr
    #     new_edge_index, new_edge_attr = self.remove_adj(hop_edge_index, None, perm, num_nodes=x.size(0))
    #     new_edge_index, new_edge_attr = remove_self_loops(new_edge_index, new_edge_attr)

    #     return new_edge_index, new_edge_attr

    # def remove_adj(self, edge_index, edge_attr, perm, num_nodes=None):
    #     mask = perm.new_full((num_nodes, ), -1)
    #     mask[perm] = perm

    #     row, col = edge_index
    #     row, col = mask[row], mask[col]
    #     mask = (row >= 0) & (col >= 0)
    #     row, col = row[mask], col[mask]

    #     if edge_attr is not None:
    #         edge_attr = edge_attr[mask]

    #     return torch.stack([row, col], dim=0), edge_attr

        # else:
        #     # get n-hop neighbor exclude 1-hop
        #     new_edge_index, new_edge_attr = self.cal_nhop_neighbor(original_x, original_edge_index, edge_attr, perm)
        #     new_edge_index, _ = torch.sort(new_edge_index, dim=0)
        #     new_edge_index, _ = coalesce(new_edge_index, None, perm.size(0), perm.size(0)) 
        #     row, col = new_edge_index
        #     new_batch = original_batch[row]
        #     if new_batch.size(0) == 0:
        #         filter_edge_index, filter_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))
        #         return x, filter_edge_index, filter_edge_attr, batch, perm

        #     # attentive edge score
        #     weights_feature = (torch.cat([original_x[row], original_x[col]], dim=1) * self.att_sl).sum(dim=-1)
        #     weights_feature = F.leaky_relu(weights_feature, 0.2) 
        #     weights_structure = 1 - self.cal_struc_att(original_x, edge_index, original_batch, new_edge_index)
        #     weights = weights_feature + 1 * weights_structure
        #     # sign = LBSign.apply
        #     # weights = sign(weights)

        #     # select topk edges
        #     print(weights)
        #     print(new_batch)
        #     perm_edge = topk(weights, 0.7, new_batch)
        #     print(perm_edge)
        #     refine_edge_index = new_edge_index[:, perm_edge]

        #     index = torch.LongTensor([1, 0])
        #     refine_reverse_index = torch.zeros_like(refine_edge_index)
        #     refine_reverse_index[index] = refine_edge_index
        #     refine_edge_index = torch.cat([refine_edge_index, refine_reverse_index], dim=1)

        #     # combine 1-hop edges and topk edges
        #     final_edge_index = torch.cat([original_edge_index, refine_edge_index], dim=1)
        #     final_edge_index, _ = filter_adj(final_edge_index, _, perm, num_nodes=original_x.size(0))
        #     final_edge_index, _ = coalesce(final_edge_index, None, perm.size(0), perm.size(0)) 

        #     return x, final_edge_index, None, batch, perm

# class LBSign(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input):
#         return torch.where(input > 0, 1, 0)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.clamp_(0, 1)

import math
import matplotlib.pyplot as plt
import dhg
import torch
import torch.nn as nn
import torch.nn.functional as F


class Readout(nn.Module):
    def __init__(self,in_dim,method = "mean"):
        super(Readout, self).__init__()
        self.method = method
        self.linear = nn.Linear(2*in_dim,in_dim)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self,node_fea,raw):
        if self.method == "mean":
            return torch.mean(self.linear(torch.hstack([node_fea,raw])), dim=0)

class Predictor(nn.Module):
    def __init__(self,in_dim,n_cats):
        super(Predictor, self).__init__()
        self.predict = nn.Linear(in_dim,n_cats)
        torch.nn.init.xavier_uniform_(self.predict.weight)

    def forward(self,node_emb):
        n_cat = self.predict(node_emb)
        return n_cat

class SpatialHyperedge(nn.Module):
    def __init__(self, in_dim, l2_lamda = 0.001, recons_lamda = 0.2, num_node = 10, lb_th = 0):
        super(SpatialHyperedge, self).__init__()
        self.l2_lamda = l2_lamda
        self.recons_lamda = recons_lamda
        # projection matrix
        self.num_node = num_node
        self.r_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim)))
        # reconstruction linear combination
        self.incidence_m = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(num_node,num_node-1)))
        self.lb_th = lb_th

    def forward(self,X):
        self.incidence_all = torch.zeros(self.num_node, self.num_node)
        self_node = torch.eye(self.num_node)
        self.recon_loss = 0
        X = X.detach()

        # calculate reconstruction loss
        master_node_proj_fea = torch.matmul(X, self.r_proj)
        slave_node_idxs = torch.tensor([[i for i in range(self.num_node) if i != node_idx] for node_idx in range(self.num_node)]).type(torch.long).to(X.device)
        slave_node_features = X[slave_node_idxs]
        neigh_recon_fea = torch.bmm(self.incidence_m.unsqueeze(1), slave_node_features).squeeze(1)
        recon_error = torch.norm(master_node_proj_fea - neigh_recon_fea, 2)
        linear_comb_l1 = torch.norm(self.incidence_m, 1)
        linear_comb_l2 = torch.norm(self.incidence_m, 2)
        self.recon_loss += recon_error * self.recons_lamda + linear_comb_l1 + self.l2_lamda * linear_comb_l2

        # generate incidence matrix filtered by 0
        node_linear_comb = torch.clamp(self.incidence_m, min=self.lb_th)
        node_linear_comb_mask = node_linear_comb > self.lb_th
        node_linear_comb = node_linear_comb.masked_fill(~node_linear_comb_mask, value=torch.tensor(0))
        x_index = torch.arange(0, self.num_node).reshape(-1, 1).repeat(1, self.num_node-1).reshape(-1).to(X.device)
        y_index = slave_node_idxs.reshape(-1)
        self.incidence_all[x_index, y_index] = node_linear_comb.reshape(-1)
        self.incidence_all = self.incidence_all + self_node
        
        return self.recon_loss, self.incidence_all

class TemporalHyperedge(nn.Module):
    def __init__(self,in_dim,l2_lamda = 0.001,recons_lamda = 0.2,num_node = 10,lb_th = 0):
        super(TemporalHyperedge, self).__init__()
        self.l2_lamda = l2_lamda
        self.recons_lamda = recons_lamda
        # projection matrix
        self.num_node = num_node
        self.r_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim)))
        # reconstruction linear combination
        self.incidence_m = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(num_node,num_node)))
        self.lb_th = lb_th

    def forward(self, cur, pre):
        self.recon_loss = 0
        cur = cur.detach()
        pre = pre.detach()

        # calculate reconstruction loss
        master_node_proj_fea = torch.matmul(cur, self.r_proj)
        slave_node_idxs = torch.tensor([[i for i in range(self.num_node)] for node_idx in range(self.num_node)]).type(torch.long).to(cur.device)
        slave_node_features = pre[slave_node_idxs]
        neigh_recon_fea = torch.bmm(self.incidence_m.unsqueeze(1), slave_node_features).squeeze(1)
        recon_error = torch.norm(master_node_proj_fea - neigh_recon_fea, 2)
        linear_comb_l1 = torch.norm(self.incidence_m, 1)
        linear_comb_l2 = torch.norm(self.incidence_m, 2)
        self.recon_loss += recon_error * self.recons_lamda + linear_comb_l1 + self.l2_lamda * linear_comb_l2

        # generate incidence matrix filtered by 0
        return self.recon_loss, torch.clamp(self.incidence_m, min=self.lb_th)
        
# class SpatialHyperedge(nn.Module):
#     def __init__(self,in_dim,l2_lamda = 0.001,recons_lamda = 0.2,num_node = 10,lb_th = 0):
#         super(SpatialHyperedge, self).__init__()
#         self.l2_lamda = l2_lamda
#         self.recons_lamda = recons_lamda
#         # projection matrix
#         self.num_node = num_node
#         self.r_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim)))
#         # reconstruction linear combination
#         self.incidence_m = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(num_node,num_node-1)))
#         self.lb_th = lb_th

#     def forward(self,X):
#         self.incidence_all = torch.zeros(self.num_node, self.num_node)
#         self_node = torch.eye(self.num_node)
#         self.recon_loss = 0
#         X = X.detach()
#         for node_idx in range(self.num_node):
#             master_node_fea = X[node_idx]
#             master_node_proj_fea = torch.matmul(master_node_fea, self.r_proj).reshape(1, -1)
#             slave_node_idx = [i for i in range(self.num_node) if i != node_idx]
#             node_linear_comb = self.incidence_m[node_idx].unsqueeze(0)
#             node_linear_comb = torch.clamp(node_linear_comb, min=self.lb_th)

#             node_linear_comb_mask = node_linear_comb > self.lb_th
#             node_linear_comb = node_linear_comb.masked_fill(~node_linear_comb_mask, value=torch.tensor(0))
#             neigh_recon_fea = torch.matmul(node_linear_comb, X[slave_node_idx])
#             self.incidence_all[node_idx][slave_node_idx] = node_linear_comb
#             linear_comb_l1 = torch.linalg.norm(node_linear_comb, 1)
#             linear_comb_l2 = torch.linalg.norm(node_linear_comb, 2)

#             recon_error = torch.linalg.norm(master_node_proj_fea - neigh_recon_fea, 2)
#                 # recon_error = torch.cdist(master_node_proj_fea,neigh_recon_fea,2)
#                 # linear_comb_l1 = torch.linalg.norm(self.incidence_m[neigh_recon_idx],1)
#                 # linear_comb_l2 = torch.linalg.norm(self.incidence_m[neigh_recon_idx],2)
#             node_recons_loss = recon_error * self.recons_lamda + linear_comb_l1 + self.l2_lamda * linear_comb_l2
#                 # node_recons_loss = recon_error.squeeze()
#             self.recon_loss += node_recons_loss
#         self.incidence_all = self.incidence_all + self_node
#         return self.recon_loss, self.incidence_all

# class TemporalHyperedge(nn.Module):
#     def __init__(self,in_dim,l2_lamda = 0.001,recons_lamda = 0.2,num_node = 10,lb_th = 0):
#         super(TemporalHyperedge, self).__init__()
#         self.l2_lamda = l2_lamda
#         self.recons_lamda = recons_lamda
#         # projection matrix
#         self.num_node = num_node
#         self.r_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim)))
#         # reconstruction linear combination
#         self.incidence_m = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(num_node,num_node)))
#         self.lb_th = lb_th

#     def forward(self,cur,pre):
#         # self.incidence_all = torch.zeros(self.num_node, self.num_node)
#         self_node = torch.eye(self.num_node)
#         self.recon_loss = 0
#         cur = cur.detach()
#         pre = pre.detach()
#         for node_idx in range(self.num_node):
#             master_node_fea = cur[node_idx]
#             master_node_proj_fea = torch.matmul(master_node_fea, self.r_proj).reshape(1, -1)
#             slave_node_idx = [i for i in range(self.num_node)]
#             node_linear_comb = self.incidence_m[node_idx].unsqueeze(0)
#             node_linear_comb = torch.clamp(node_linear_comb, min=self.lb_th)

#             node_linear_comb_mask = node_linear_comb > self.lb_th
#             node_linear_comb = node_linear_comb.masked_fill(~node_linear_comb_mask, value=torch.tensor(0))
#             neigh_recon_fea = torch.matmul(node_linear_comb,pre)
#             # self.incidence_all[node_idx][slave_node_idx] = node_linear_comb
#             linear_comb_l1 = torch.linalg.norm(node_linear_comb, 1)
#             linear_comb_l2 = torch.linalg.norm(node_linear_comb, 2)

#             recon_error = torch.linalg.norm(master_node_proj_fea - neigh_recon_fea, 2)
#                 # recon_error = torch.cdist(master_node_proj_fea,neigh_recon_fea,2)
#                 # linear_comb_l1 = torch.linalg.norm(self.incidence_m[neigh_recon_idx],1)
#                 # linear_comb_l2 = torch.linalg.norm(self.incidence_m[neigh_recon_idx],2)
#             node_recons_loss = recon_error * self.recons_lamda + linear_comb_l1 + self.l2_lamda * linear_comb_l2
#                 # node_recons_loss = recon_error.squeeze()
#             self.recon_loss += node_recons_loss
#         # self.incidence_all = self.incidence_all + self_node
#         return self.recon_loss, self.incidence_m

class SpatialHyperedgeMP(nn.Module):
    def __init__(self):
        super(SpatialHyperedgeMP, self).__init__()

    def forward(self,cur,incidence_m):
        edge_fea = torch.mm(incidence_m, cur)
        edge_degree = torch.sum(incidence_m, dim=1).reshape(-1, 1)
        edge_fea_normed = torch.div(edge_fea,edge_degree)
        return edge_fea_normed

class TemporalHyperedgeMP(nn.Module):
    def __init__(self):
        super(TemporalHyperedgeMP, self).__init__()

    def forward(self,cur,pre,incidence_m):
        edge_fea = torch.mm(incidence_m, pre) + cur
        self_degree = torch.ones(incidence_m.shape[0],1)
        edge_degree = torch.sum(incidence_m, dim=1).reshape(-1, 1) + self_degree
        edge_fea_normed = torch.div(edge_fea,edge_degree)
        return edge_fea_normed


# class HHNodeMP(nn.Module):
#     def __init__(self,in_dim = 256,num_node = 10,drop_rate = 0.3):
#         super(HHNodeMP, self).__init__()
#         self.node_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim)))
#         self.spatial_edge_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim)))
#         self.temporal_edge_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim)))
#         self.num_node = num_node
#         self.act_1 = nn.Softmax(dim = 0)
#         self.in_dim = in_dim
#         self.drop = nn.Dropout(drop_rate)
#         self.act = nn.ReLU(inplace=True)
#         self.theta = nn.Linear(in_dim, in_dim, bias=True)
#         torch.nn.init.xavier_uniform_(self.theta.weight)


#     def forward(self,cur,spatial_hyperedge_emb,temporal_hyperedge_emb):
#         rlt = []
#         for node_idx in range(self.num_node):
#             node_fea = cur[node_idx]
#             node_fea = torch.matmul(node_fea, self.node_proj)
#             spatial_hyperedge_fea = spatial_hyperedge_emb[node_idx]
#             temporal_hyperedge_fea = temporal_hyperedge_emb[node_idx]
#             spatial_hyperedge_fea = torch.matmul(spatial_hyperedge_fea, self.spatial_edge_proj)
#             temporal_hyperedge_fea = torch.matmul(temporal_hyperedge_fea, self.temporal_edge_proj)
#             hyperedge = torch.vstack([spatial_hyperedge_fea,temporal_hyperedge_fea])
#             atten = self.act_1(torch.matmul(hyperedge,node_fea.T)/math.sqrt(self.in_dim)).reshape(-1,1)
#             rlt.append(torch.sum(torch.mul(atten,hyperedge),0).reshape(1,-1))
#         return self.drop(self.act(self.theta(torch.vstack(rlt))))

class HHNodeMP(nn.Module):
    def __init__(self,in_dim = 256,num_node = 10,drop_rate = 0.3):
        super(HHNodeMP, self).__init__()
        self.node_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim)))
        self.spatial_edge_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim)))
        self.temporal_edge_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim)))
        self.num_node = num_node
        self.act_1 = nn.Softmax(dim = 0)
        self.in_dim = in_dim
        self.drop = nn.Dropout(drop_rate)
        self.act = nn.ReLU(inplace=True)
        self.theta = nn.Linear(in_dim, in_dim, bias=True)
        torch.nn.init.xavier_uniform_(self.theta.weight)


    def forward(self,cur,spatial_hyperedge_emb,temporal_hyperedge_emb):
        rlt = []
        node_fea = torch.matmul(cur, self.node_proj)
        spatial_hyperedge_fea = torch.matmul(spatial_hyperedge_emb, self.spatial_edge_proj)
        temporal_hyperedge_fea = torch.matmul(temporal_hyperedge_emb, self.temporal_edge_proj)
        spa_attn = torch.einsum('ij, ij->i', spatial_hyperedge_fea, node_fea).reshape(-1, 1) / math.sqrt(self.in_dim)
        tmp_attn = torch.einsum('ij, ij->i', temporal_hyperedge_fea, node_fea).reshape(-1, 1) / math.sqrt(self.in_dim)
        attn = torch.cat((spa_attn, tmp_attn), dim=1).T
        val = attn[0].reshape(-1, 1) * spatial_hyperedge_fea + attn[1].reshape(-1, 1) * temporal_hyperedge_fea
        return self.drop(self.act(self.theta(val)))



class TimeBlock(nn.Module):
    def __init__(self,in_dim = 256,num_node = 10):
        super(TimeBlock, self).__init__()
        self.spatial_hyperedge = SpatialHyperedge(in_dim)
        self.temporal_hyperedge = TemporalHyperedge(in_dim)
        self.spatial_hyperedge_MP = SpatialHyperedgeMP()
        self.temporal_hyperedge_MP = TemporalHyperedgeMP()
        self.node_mp = HHNodeMP()

    def forward(self,cur,pre):
        """
        :param cur: N * d
        :param pred: N * d
        :return: N * d
        """
        spatial_hyperedge_loss,spatial_hyperedge_incidence = self.spatial_hyperedge(cur)
        temporal_hyperedge_loss, temporal_hyperedge_incidence = self.temporal_hyperedge(cur,pre)
        spatial_hyperedge_emb = self.spatial_hyperedge_MP(cur,spatial_hyperedge_incidence)
        temporal_hyperedge_emb = self.temporal_hyperedge_MP(cur,pre,temporal_hyperedge_incidence)
        node_emb = self.node_mp(cur,spatial_hyperedge_emb,temporal_hyperedge_emb)
        return node_emb,temporal_hyperedge_loss+spatial_hyperedge_loss



class Model(nn.Module):
    """
    multi-timestamp training
    """

    def __init__(self,win_size,h_dim = 256,n_cats = 5,recons_lambda = 0.1):
        super(Model, self).__init__()
        self.win_size = win_size
        self.time_cursor = TimeBlock()
        self.predictor = Predictor(h_dim,n_cats = n_cats)
        self.readout = Readout(h_dim)
        self.recons_lambda = recons_lambda

    def forward(self, node_fea):
        recon_loss = 0
        for i in range(self.win_size-1):
            if i == 0:
                pre_node = node_fea[0]
                cur_node = node_fea[1]
                cur_node_emb,r_loss = self.time_cursor(pre_node, cur_node)
                recon_loss += r_loss
            else:
                cur_node = node_fea[i + 1].contiguous()
                pre_node = cur_node_emb.contiguous()
                cur_node_emb,r_loss = self.time_cursor(pre_node, cur_node)
                recon_loss += r_loss
        graph_emb = self.readout(cur_node_emb,node_fea[-1])
        logits = self.predictor(graph_emb)
        return logits,recon_loss*self.recons_lambda




if __name__ == "__main__":
    pass



import dgl
import torch
from torch import nn
from torch.nn import Module
from . import process
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else nn.ReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)

        if self.bias is not None:
            out += self.bias

        return self.act(out), seq_fts


# Applies mean-pooling on neighbors
class AvgNeighbor(nn.Module):
    def __init__(self):
        super(AvgNeighbor, self).__init__()

    def forward(self, seq, adj_ori):
        adj_ori = process.sparse_mx_to_torch_sparse_tensor(adj_ori)
        if torch.cuda.is_available():
            adj_ori = adj_ori.cuda()
        return torch.unsqueeze(torch.spmm(adj_ori, torch.squeeze(seq, 0)), 0)

class Discriminator(nn.Module):
    def __init__(self, n_h1, n_h2):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h1, n_h2, 1)
        self.act = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_c, h_pl, sample_list, s_bias1=None, s_bias2=None):
        sc_1 = torch.squeeze(self.f_k(h_pl, h_c), 2)
        sc_1 = self.act(sc_1)
        sc_2_list = []
        for i in range(len(sample_list)):
            h_mi = torch.unsqueeze(h_pl[0][sample_list[i]],0)
            sc_2_iter = torch.squeeze(self.f_k(h_mi, h_c), 2)
            sc_2_list.append(sc_2_iter)
        sc_2_stack = torch.squeeze(torch.stack(sc_2_list,1),0)
        sc_2 = self.act(sc_2_stack)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        return sc_1, sc_2


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret



class GMI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(GMI, self).__init__()
        self.gcn1 = GCN(n_in, n_h,
                        activation)  # if on citeseer and pubmed, the encoder is 1-layer GCN, you need to modify it
        self.gcn2 = GCN(n_h, n_h, activation)
        self.disc1 = Discriminator(n_in, n_h)
        self.disc2 = Discriminator(n_h, n_h)
        self.avg_neighbor = AvgNeighbor()
        self.prelu = nn.PReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, seq1, adj_ori, neg_num, adj, samp_bias1, samp_bias2):
        h_1, h_w = self.gcn1(seq1, adj)
        h_2, _ = self.gcn2(h_1, adj)
        h_neighbor = self.prelu(self.avg_neighbor(h_w, adj_ori))
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        res_mi_pos, res_mi_neg = self.disc1(h_2, seq1, process.negative_sampling(adj_ori, neg_num), samp_bias1,
                                            samp_bias2)
        # I(h_i; x_j) node j is a neighbor
        res_local_pos, res_local_neg = self.disc2(h_neighbor, h_2, process.negative_sampling(adj_ori, neg_num),
                                                  samp_bias1, samp_bias2)
        """I(w_ij; a_ij)"""
        adj_rebuilt = self.sigm(torch.mm(torch.squeeze(h_2), torch.t(torch.squeeze(h_2))))

        return res_mi_pos, res_mi_neg, res_local_pos, res_local_neg, adj_rebuilt

    # detach the return variables
    def embed(self, seq, adj):
        h_1, _ = self.gcn1(seq, adj)
        h_2, _ = self.gcn2(h_1, adj)

        return h_2.detach()


class Attention_Gate(nn.Module):
    def __init__(self, config):
        super(Attention_Gate, self).__init__()
        self.config = config



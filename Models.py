from matplotlib.pyplot import cla
from Data_Process import *
import torch

import sys
sys.path.append('D:\OneDrive - mail.nwpu.edu.cn\Optimal\Public\Python\Pre_Process')
from Data_Process import *
import torch as th
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import random
from torch_geometric.nn import TransformerConv,GATConv,HeteroConv
import torch.nn.functional as F

from torch_geometric.nn import Linear, HGTConv
from my_hetero_conv import AttentHeteroConv

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
class GraphConvolution(Module):
    """
    Simple pygGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeatn, adj):
        support = th.spmm(infeatn, self.weight)
        output = th.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class myGAE(torch.nn.Module):
    def __init__(self, data):
        super(myGAE, self).__init__()

        
        self.decoder_1 = torch.nn.Sequential(torch.nn.Linear(300, 300)).to('cuda:0')
        self.decoder_2 = torch.nn.Sequential(torch.nn.Linear(300, 300)).to('cuda:0')

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, 300).to('cuda:0')

        
        self.conv_0 = HeteroConv({
            ('meta', 'is_concept_stock_of', "meta"): TransformerConv(300, 300, heads = 2, concat=False, beta=False),
            ('source_emb', 'is_source_of', "meta"): TransformerConv(300, 300, heads = 2, concat=False, beta=False),
            ("meta", "rev_is_source_of", "source_emb"): TransformerConv(300, 300, heads = 2, concat=False, beta=False),
        }, aggr = "sum").to('cuda:0')

        self.linear_a = torch.nn.Linear(300, 300).to('cuda:0')
        self.linear_b = torch.nn.Linear(300,300).to('cuda:0')
        # self.lin = Linear(300, 300)

        self.self_attention_1 = TransformerConv(300, 300, heads = 1, concat=False, beta=False).to('cuda:1')
        self.self_attention_2 = TransformerConv(300, 300, heads = 1, concat=False, beta=False).to('cuda:1')
        

    def Encoder(self, edge_concept, H_0, H_a, hetro_data, edge_a, edge_b):
        x_dict = hetro_data.x_dict
        edge_index_dict = hetro_data.edge_index_dict

        H_0 = self.self_attention_1(H_0.to('cuda:1'), edge_a.to('cuda:1'))
        H_a = self.self_attention_2(H_a.to('cuda:1'), edge_b.to('cuda:1'))

        # H_0 = self.linear_a(H_0.to('cuda:0'))
        # H_a = self.linear_b(H_a.to('cuda:0'))
        H_ens = H_0+H_a
        H_0 = F.normalize(H_0, dim=-1, p=2)
        H_a = F.normalize(H_a, dim=-1, p=2)
        H_ens = F.normalize(H_ens, dim=-1, p=2)

        x_dict['source_emb'] = torch.cat([H_0.to('cuda:0'), H_a.to('cuda:0')],dim=0).to('cuda:0')

        x_dict['meta'] = H_ens.to('cuda:0')

        # print(x_dict)
        # print(test)

        x_dict = self.conv_0(x_dict, edge_index_dict)
        

        representation = (x_dict['meta'])+H_ens.to('cuda:0')
        representation = F.normalize(representation, dim=-1, p=2)
        return representation



    def loss(self, H_2, batch_data, H_0, H_a):
        H_0 = H_0.to('cuda:0')
        H_a = H_a.to('cuda:0')

        concept_idxs = [x[0] for x in batch_data]
        pos_idxs = [x[1] for x in batch_data]
        neg_idxs = [x[2] for x in batch_data]

        emb_concept = H_2[concept_idxs]
        emb_pos_stock = H_2[pos_idxs]
        emb_neg_stock = H_2[neg_idxs]


        triplet_loss = torch.nn.TripletMarginLoss(margin=0.01, p=2)
        graph_loss = triplet_loss(emb_concept, emb_pos_stock, emb_neg_stock)

        # if self.training == False:
        #     print("debug")
        #     print(graph_loss)
        recon_loss = torch.nn.MSELoss()
        
        recon_H_0 = self.decoder_1(H_2)
        recon_H_a = self.decoder_2(H_2)
        

        recon_H_0 = F.normalize(recon_H_0, dim=-1, p=2)
        recon_H_a = F.normalize(recon_H_a, dim=-1, p=2)

        batch_size = 128
        sample_idx = random.sample(range(recon_H_0.shape[0]), batch_size)

        recon_loss_1 =  recon_loss(recon_H_0[sample_idx], H_0[sample_idx])
        recon_loss_2 = recon_loss(recon_H_a[sample_idx], H_a[sample_idx])

        alpha = 0.5
        loss = (1-alpha)*recon_loss_1+ alpha*recon_loss_2+ 0.001* graph_loss
        # loss = (1-alpha)*recon_loss_1+ alpha*recon_loss_2
        return loss

    def forward(self, concept_edge, H_0, H_a, batch_data, hetro_data, edge_a, edge_b):
        Latent_Representation = self.Encoder(concept_edge, H_0, H_a, hetro_data, edge_a, edge_b)
        loss = self.loss(Latent_Representation, batch_data, H_0, H_a)
        return loss, Latent_Representation


    





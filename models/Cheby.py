from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from models.common_blocks import batch_norm
from regularization import *
from infomax import Infomax

class Cheby(nn.Module):
    def __init__(self, args):
        super(Cheby, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden

        self.dropout = args.dropout
        self.cached = True # only could be used in transductive learning
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.type_norm = args.type_norm
        self.skip_weight = args.skip_weight
        self.num_groups = args.num_groups

        ###
        self.alpha = args.alpha
        self.beta = args.beta
        self.loss_corr = 0

        self.infomax = Infomax(self.num_feats, args.dim_hidden)

        # build up the convolutional layers
        if self.num_layers == 1:
            self.layers_GCN.append(ChebConv(self.num_feats, self.num_classes, K=2, bias=True))
        elif self.num_layers == 2:
            self.layers_GCN.append(ChebConv(self.num_feats, self.dim_hidden, K=2, bias=True))
            self.layers_GCN.append(ChebConv(self.dim_hidden, self.num_classes, K=2, bias=True))
        else:
            self.layers_GCN.append(ChebConv(self.num_feats, self.dim_hidden, K=2, bias=True))
            for _ in range(self.num_layers-2):
                self.layers_GCN.append(ChebConv(self.dim_hidden, self.dim_hidden, K=2, bias=True))
            self.layers_GCN.append(ChebConv(self.dim_hidden, self.num_classes, K=2, bias=True))

        # build up the normalization layers
        for i in range(self.num_layers):
            dim_out = self.layers_GCN[i].out_channels
            if self.type_norm in ['None', 'batch', 'pair']:
                skip_connect = False
            else:
                skip_connect = True
            self.layers_bn.append(batch_norm(dim_out, self.type_norm, skip_connect, self.num_groups, self.skip_weight))

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, ChebConv):
                m.reset_parameters()
        self.apply(weight_reset)

    def forward(self, x, edge_index, report_metrics=False):

        for i in range(self.num_layers):
            if i == 0 or i == self.num_layers-1:
                x_1 = x
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index)
            x = self.layers_bn[i](x)
            if self.training:
                if self.alpha > 0 and i != self.num_layers-1:
                    self.loss_corr += self.alpha * loss_corr(x)

                if self.beta > 0 and i!=self.num_layers-1 and i%5==4:
                    self.loss_corr += self.beta * self.infomax.get_loss(x_1, x, layer=self.layers_GCN[0], edge_index=edge_index)

            if i == self.num_layers-2:
                corr = torch_corr(x.t())
                corr = torch.triu(corr, 1).abs()
                n = corr.shape[0]
                self.corr_2 = corr.sum().item() / (n *(n-1) /2)
                if report_metrics:
                    self.sim_2 = get_pairwise_sim(x)

            if i == self.num_layers-2:
                corr = torch_corr(x.t())
                corr = torch.triu(corr, 1).abs()
                n = corr.shape[0]
                self.corr = corr.sum().item() / (n *(n-1) /2)
                self.sim = get_pairwise_sim(x)
                if report_metrics:
                    self.sim = get_pairwise_sim(x)

            if i != self.num_layers-1:
                x = F.relu(x)

        return x



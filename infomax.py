"""Borrowed from https://github.com/PetarV-/DGI"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random

class Infomax(nn.Module):

    def __init__(self, n_h1, n_h2):
        super(Infomax, self).__init__()
        self.f_k = nn.Bilinear(n_h1, n_h2, 1)
        self.node_indices = None
        self.bce = nn.BCEWithLogitsLoss()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)
        c_x = c
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x))
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x))

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2))#.view(-1,1)

        return logits

    def get_loss(self, x, f_x, **kwargs):
        self.node_indices = None
        if self.node_indices is None:
            self.node_indices = np.arange(x.shape[0])
            labels_pos = torch.ones(x.shape[0])
            labels_neg = torch.zeros(x.shape[0])
            self.labels = torch.cat((labels_pos, labels_neg))
            if torch.cuda.is_available():
                self.labels = self.labels.cuda()
        pos = (x)
        idx = np.random.permutation(self.node_indices)
        neg = (x[idx])
        logits = self.forward(f_x, pos, neg)
        loss = self.bce(logits, self.labels)
        return loss


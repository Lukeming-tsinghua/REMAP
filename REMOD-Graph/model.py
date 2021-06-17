# -*- coding: utf-8 -*-
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from hetero_model import HeteroGAT


class GATEncoder(nn.Module):
    def __init__(self, g, h):
        super().__init__()
        self.g = g
        self.h = h
        
        self.gat = HeteroGAT(meta_paths=[['DDx'], 
            ['May Be Caused By'], 
            ['May Cause'],
            ['DDx','May Cause'],
            ['DDx','May Be Caused By'],
            ['May Cause','DDx'],
            ['May Be Caused By','DDx'],
            ['DDx','DDx'],
            ['May Cause','May Cause'],
            ['May Be Caused By','May Be Caused By']],
                             in_size=1000,
                             hidden_size=100,
                             out_size=100,
                             num_heads=[20],
                             dropout=0.5)

    def forward(self, cui1, cui2):
        v = self.gat(self.g, self.h)
        return v[cui1,], v[cui2,]
    
    def __repr__(self):
        return self.__class__.__name__


class Scorer(nn.Module):
    def __init__(self, score_func, label_num, hidden_dim, dropout):
        super().__init__()

        self.score_func = score_func

        self.relation_embedding = nn.Parameter(torch.Tensor(label_num, hidden_dim))
        nn.init.xavier_uniform_(self.relation_embedding, gain=1)

        if self.score_func == "TransE":
            pass
        elif self.score_func == "TuckER":
            self.W = torch.nn.Parameter(torch.tensor(
                np.random.uniform(-1, 1, (hidden_dim, hidden_dim, hidden_dim)), 
                dtype=torch.float, requires_grad=True))
            nn.init.xavier_uniform_(self.W, gain=1)
            self.bn0 = torch.nn.BatchNorm1d(hidden_dim)
            self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
            self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

            self.input_dropout_1 = torch.nn.Dropout(dropout)
            self.input_dropout_2 = torch.nn.Dropout(dropout)
            self.hidden_dropout1 = torch.nn.Dropout(dropout)
            self.hidden_dropout2 = torch.nn.Dropout(dropout)
        else:
            raise ValueError("score function %s is not impletmented")

    def forward(self, h, t):
        if self.score_func == "TransE":
            N = h.size(0)
            R = self.relation_embedding.size(0)
            h_exp = h.repeat_interleave(R, dim=0)
            t_exp = t.repeat_interleave(R, dim=0)
            r_exp = self.relation_embedding.repeat(N,1)
            score = torch.sigmoid(-torch.norm(h_exp+r_exp-t_exp, p=2, dim=1).view(-1, R))
        elif self.score_func == "TuckER":
            h_exp = self.bn0(h)
            h_exp = self.input_dropout_1(h_exp)
            h_exp = h_exp.view(-1,1,1,h_exp.size(1))
            t_exp = self.bn2(t)
            t_exp = self.input_dropout_2(t_exp)
            t_exp = t_exp.view(-1,1,t_exp.size(1))

            W_mat = torch.mm(self.relation_embedding, self.W.view(self.relation_embedding.size(1), -1))
            W_mat = W_mat.view(-1, h.size(1), h.size(1))
            W_mat = self.hidden_dropout1(W_mat)

            score = torch.matmul(h_exp, W_mat).squeeze()
            score = torch.bmm(score, t_exp.transpose(2,1)).squeeze()
            score = torch.sigmoid(score)
        return score


class GraphScoreEncoder(nn.Module):
    def __init__(self, g, h, score_func, label_num, hidden_dim, dropout):
        super().__init__()
        self.g = g
        self.h = h
        self.score_func = score_func
        self.label_num = label_num
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.graph_encoder = GATEncoder(g, h)
        self.graph_score = Scorer(score_func, label_num, hidden_dim, dropout)


    def forward(self, cui1, cui2):
        h, t = self.graph_encoder(cui1, cui2)
        return self.graph_score(h, t)
    
    def __repr__(self):
        return "_".join(("HAN", self.score_func, str(self.hidden_dim), str(self.dropout)))

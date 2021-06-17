# -*- coding: utf-8 -*-
import math
import sys 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertConfig, BertModel, BertPreTrainedModel

LABEL_NUM = 4


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(nn.Linear(hidden_dim, 64),
                                        nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        energy = energy / math.sqrt(self.hidden_dim)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config, hidden_dim):
        super().__init__(config)
        self.bert = BertModel(config)
        self.init_weights()
        self.sent_attention_1 = SelfAttention(768)
        self.sent_attention_2 = SelfAttention(768)
        self.bert_linear = nn.Linear(768, hidden_dim)

    def forward(self, tokens, split_points, entity_1_begin_idxs, entity_2_begin_idxs):
        bert_output = self.bert(**tokens)[0]
        ep_text_info_h = []
        ep_text_info_t = []
        for i in range(len(split_points) - 1):
            ep_text_tmp = bert_output[
                split_points[i]:split_points[i + 1]]
            ep_text_tmp_e1_begin = ep_text_tmp[entity_1_begin_idxs[i][0], entity_1_begin_idxs[i][1], :]
            ep_text_tmp_e2_begin = ep_text_tmp[entity_2_begin_idxs[i][0], entity_2_begin_idxs[i][1], :]
            ep_text_tmp_e1_begin, _ = self.sent_attention_1(ep_text_tmp_e1_begin.unsqueeze(0))
            ep_text_tmp_e2_begin, _ = self.sent_attention_2(ep_text_tmp_e2_begin.unsqueeze(0))
            ep_text_info_h.append(ep_text_tmp_e1_begin)
            ep_text_info_t.append(ep_text_tmp_e2_begin)
        ep_text_info_h = self.bert_linear(torch.cat(ep_text_info_h, 0))
        ep_text_info_t = self.bert_linear(torch.cat(ep_text_info_t, 0))
        return ep_text_info_h, ep_text_info_t

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


class BertScoreEncoder(nn.Module):
    def __init__(self, config, score_func, label_num, hidden_dim, dropout):
        super().__init__()
        self.config = config
        self.score_func = score_func
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.text_encoder = BertEncoder.from_pretrained(config, hidden_dim=hidden_dim)
        self.text_scorer = Scorer(score_func, label_num, hidden_dim, dropout)

    def forward(self, tokens, split_points, entity_1_begin_idxs, entity_2_begin_idxs):
        h, t = self.text_encoder(tokens, split_points, entity_1_begin_idxs, entity_2_begin_idxs)
        return self.text_scorer(h, t)

    def __repr__(self):
        return "_".join((self.config, self.score_func, str(self.hidden_dim), str(self.dropout)))

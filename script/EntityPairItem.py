import json
import pickle
import re

import numpy as np
import torch
from tqdm import tqdm


def padding(seq, size, token):
    length = len(seq)
    if length > size:
        return seq[:size]
    else:
        return seq + [token] * (size - length)


def word2seq(sent, word2index):
    word_list = sent.split(' ')
    seq_list = []
    for word in word_list:
        if word in word2index.keys():
            seq_list.append(word2index[word])
        else:
            seq_list.append(word2index['<unk>'])
    return seq_list


class EntityPairItem:
    def __init__(self, arg_list):
        self.cui1 = arg_list[0]
        self.cui2 = arg_list[1]
        self.sentences = arg_list[2]
        self.structures = arg_list[3]
        self.pos1 = arg_list[4]
        self.pos2 = arg_list[5]
        self.cui_info1 = arg_list[6]
        self.cui_info2 = arg_list[7]
        self.label = arg_list[8]
        self._WORD_PADDING_SIZE = 200
        self._POSITION_PADDING_SIZE = 200
        self._WORD_PADDING_TOKEN = 1
        self._POSITION_PADDING_TOKEN = 0
        self.is_transform = False

    def fetch(self, sample_num=None, seed=None):
        assert self.is_transform
        if sample_num is None or len(self.sentences) <= sample_num:
            return [(self.sentences,self.structures,self.pos1,self.pos2),\
                    (self.cui1,self.cui2,self.cui_info1,self.cui_info2,len(self.sentences),self.label)]
        else:
            np.random.seed(seed)
            index = list(range(len(self.sentences)))
            np.random.shuffle(index)
            index = index[0:sample_num]
            return [([self.sentences[idx] for idx in index],
                     [self.structures[idx] for idx in index],
                        self.pos1[index],self.pos2[index]),\
                    (self.cui1,self.cui2,self.cui_info1,self.cui_info2,sample_num,self.label)]

    def print(self, Type='short'):
        if Type == 'detailed':
            print(self.cui1, type(self.cui1))
            print(self.cui2, type(self.cui2))
            print(self.sentences, type(self.sentences))
            print(self.structures, type(self.structures))
            print(self.pos1, type(self.pos1))
            print(self.pos2, type(self.pos2))
            print(self.cui_info1, type(self.cui_info1))
            print(self.cui_info2, type(self.cui_info2))
            print(self.label, type(self.label))
        else:
            print('self.cui1:', type(self.cui1))
            print('self.cui2:', type(self.cui2))
            print('self.sentences:', type(self.sentences))
            print('self.structures:', type(self.structures))
            print('self.pos1:', type(self.pos1))
            print('self.pos2:', type(self.pos2))
            print('self.cui_info1:', type(self.cui_info1))
            print('self.cui_info2:', type(self.cui_info2))
            print('self.label:', type(self.label))

    def handle_type(self):
        self.sentences = self.handle_json(self.sentences)
        self.structures = self.handle_json(self.structures)
        self.pos1 = self.handle_json(self.pos1)
        self.pos2 = self.handle_json(self.pos2)
        self.cui_info1 = self.handle_json(self.cui_info1)
        self.cui_info2 = self.handle_json(self.cui_info2)
        self.label = self.handle_label(self.label)

    def handle_label(self, label):
        return int(label)

    def handle_json(self, sentences):
        return json.loads(sentences)

    def trans_pos(self, pos_list):
        position = []
        for pos in pos_list:
            position.append(
                padding(pos, self._POSITION_PADDING_SIZE,
                        self._POSITION_PADDING_TOKEN))
        return torch.LongTensor(position)

    def trans_text_list(self, text_list, word2index,tokenize=True):
        seqs = []
        if tokenize == True:
            for text in text_list:
                seqs.append(
                    padding(word2seq(text, word2index), self._WORD_PADDING_SIZE,
                            self._WORD_PADDING_TOKEN))
            return torch.LongTensor(seqs)
        else:
            for text in text_list:
                seqs.append(text)
            return seqs

    def trans_cui_info(self, info):
        return torch.FloatTensor(info).view(1, -1)

    def trans_label(self, label):
        return torch.LongTensor([label])

    def transform(self, word2index, tokenize=True):
        self.handle_type()
        self.sentences = self.trans_text_list(self.sentences, word2index, tokenize=tokenize)
        self.structures = self.trans_text_list(self.structures, word2index, tokenize=tokenize)
        self.pos1 = self.trans_pos(self.pos1)
        self.pos2 = self.trans_pos(self.pos2)
        self.cui_info1 = self.trans_cui_info(self.cui_info1)
        self.cui_info2 = self.trans_cui_info(self.cui_info2)
        self.label = self.trans_label(self.label)
        self.is_transform = True

    def __eq__(self, other):
        return self.cui1 == other.cui1 and self.cui2 == other.cui2

    def __hash__(self):
        return hash(self.cui1+self.cui2)

class BertEntityPairItem(EntityPairItem):
    def __init__(self,arg_list):
        super(BertEntityPairItem,self).__init__(arg_list)

    def fetch(self, sample_num=None, seed=None):
        assert self.is_transform
        if sample_num is None or len(self.sentences) <= sample_num:
            return [(self.sentences,self.structures,self.pos1,self.pos2),\
                    (self.cui1,self.cui2,self.cui_info1,self.cui_info2,len(self.sentences),self.label)]
        else:
            np.random.seed(seed)
            index = list(range(len(self.sentences)))
            np.random.shuffle(index)
            index = index[0:sample_num]
            return [([self.sentences[idx] for idx in index],
                     [self.structures[idx] for idx in index],
                        self.pos1[index],self.pos2[index]),\
                    (self.cui1,self.cui2,self.cui_info1,self.cui_info2,sample_num,self.label)]

    def draw_bert_text(self,text):
        token_map = re.findall(r"t[0-9]{3}.*t[0-9]{3}",text)
        if len(token_map) != 1:
            print(sent,token_map)
            raise RuntimeError()
        center_str = re.sub(r"t[0-9]{3}","",token_map[0]).strip().replace("<sep>",",")
        return center_str

    def trans_text_list_bert(self, text_list, word2index,tokenize=True,sent=True):
        seqs = []
        if tokenize == True:
            for text in text_list:
                seqs.append(
                    padding(word2seq(text, word2index), self._WORD_PADDING_SIZE,
                            self._WORD_PADDING_TOKEN))
            return torch.LongTensor(seqs)
        else:
            for text in text_list:
                if sent:
                    seqs.append(self.draw_bert_text(text))
                else:
                    seqs.append(text)
            return seqs

    def transform(self, word2index, tokenize=True):
        self.handle_type()
        self.sentences = self.trans_text_list_bert(self.sentences, word2index, tokenize=tokenize,sent=False)
        self.structures = self.trans_text_list_bert(self.structures, word2index, tokenize=tokenize,sent=False)
        self.pos1 = self.trans_pos(self.pos1)
        self.pos2 = self.trans_pos(self.pos2)
        self.cui_info1 = self.trans_cui_info(self.cui_info1)
        self.cui_info2 = self.trans_cui_info(self.cui_info2)
        self.label = self.trans_label(self.label)
        self.is_transform = True

    def __eq__(self, other):
        return self.cui1 == other.cui1 and self.cui2 == other.cui2

    def __hash__(self):
        return hash(self.cui1+self.cui2)

import os
import pickle
import re
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from transformers import BertTokenizer


def bert_collate_func(arrays):
    sentences = {key: torch.cat([array[0][key] for array in arrays], 0) for key in arrays[0][0].keys()}
    split_points = np.cumsum([0] + [array[1] for array in arrays])
    labels = torch.cat([array[2] for array in arrays], 0)
    cui1 = torch.LongTensor([array[3] for array in arrays])
    cui2 = torch.LongTensor([array[4] for array in arrays])
    entity_1_begin_idxs = [array[5] for array in arrays]
    entity_2_begin_idxs = [array[6] for array in arrays]
    return (cui1,cui2,sentences,split_points, entity_1_begin_idxs, entity_2_begin_idxs),labels


class BertEntityPairDataset(Dataset):
    def __init__(self, pkl_path=None, 
            dict_path=None, 
            EntityPairItemList=None,
            sample_num=None,
            max_length=None,
            tokenizer=None, 
            entity_1_begin_tokens=None,
            entity_2_begin_tokens=None,
            ratio=1):
        if pkl_path:
            self.pkl_path = pkl_path
            if self.pkl_path.endswith(".pkl"):
                load_func = pickle.load
            elif self.pkl_path.endswith(".jl"):
                load_func = joblib.load
            else:
                raise RuntimeError("illegal file path")

            with open(self.pkl_path, 'rb') as f:
                self.EntityPairItemList = load_func(f)
        elif EntityPairItemList is not None:
            self.EntityPairItemList = EntityPairItemList
        else:
            raise ValueError('init dataset or path is needed')

        self.EntityPairItemList = self.EntityPairItemList[0:int(len(self.EntityPairItemList) * ratio)]

        self.dict_path = dict_path
        with open(self.dict_path,"rb") as f:
            self.cui2idx = pickle.load(f)

        self.sample_num = sample_num
        self.max_length = max_length
        assert tokenizer is not None
        self.tokenizer = tokenizer
        self.entity_1_begin_ids = self.tokenizer.convert_tokens_to_ids(entity_1_begin_tokens)
        self.entity_2_begin_ids = self.tokenizer.convert_tokens_to_ids(entity_2_begin_tokens)

    def cal_begin_idxs(self, input_ids):
        entity_1_begin_idxs = None
        entity_2_begin_idxs = None
        for entity_1_begin_id in self.entity_1_begin_ids:
            idxs = torch.where(input_ids==entity_1_begin_id)
            if len(idxs[0]) > 0:
                entity_1_begin_idxs = idxs
        for entity_2_begin_id in self.entity_2_begin_ids:
            idxs = torch.where(input_ids==entity_2_begin_id)
            if len(idxs[0]) > 0:
                entity_2_begin_idxs = idxs
        return entity_1_begin_idxs, entity_2_begin_idxs


    def __getitem__(self, index):
        samples = self.EntityPairItemList[index].fetch(self.sample_num)
        cui_name1 = samples[1][0]
        cui_name2 = samples[1][1]
        cui1 = self.cui2idx[cui_name1]
        cui2 = self.cui2idx[cui_name2]
        text = samples[0][0]
        structure = samples[0][1]
        text_tokenized = self.tokenizer.batch_encode_plus(
                list(zip(text,structure)),
                add_special_token=True,
                max_length=self.max_length,
                pad_to_max_length=True,return_tensors="pt")
        entity_1_begin_idxs, entity_2_begin_idxs = self.cal_begin_idxs(text_tokenized["input_ids"])
        if entity_1_begin_idxs is None or entity_2_begin_idxs is None:
            print(text)
            print(text_tokenized["input_ids"], text_tokenized["input_ids"].size())
            raise RuntimeError
        num = samples[1][4]
        label = samples[1][5]
        return (text_tokenized,num,label,cui1,cui2, entity_1_begin_idxs, entity_2_begin_idxs)

    def __len__(self):
        return len(self.EntityPairItemList)

    def __iter__(self):
        return iter(self.EntityPairItemList)

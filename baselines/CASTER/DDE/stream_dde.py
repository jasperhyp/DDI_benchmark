import numpy as np
import pandas as pd
import torch
from torch.utils import data

from subword_nmt.apply_bpe import BPE
import codecs

'''
DataFolder = './data'

unsup_train_file = 'food_smiles.csv' & 'drug_smiles.csv' & 'ddi_smiles.csv'
 
unsupervised pair dataset to pretrain the network
SMILES string as input

DDI supervised data files:

    train = 'train.csv'
    val = 'val.csv'
    test = 'test.csv' 
    

build a UnsupData which returns v_d, v_f for a batch

supTrainData which return v_d, v_f, label for DDI only 
supTrainData.num_of_iter_in_a_epoch contains iteration in an epoch

ValData which return v_d, v_f, label for DDI only 

'''
import os
print(os.getcwd())

CUR_DIR = "/home/yeh803/workspace/DDI/NovelDDI/baselines/CASTER/DDE/"

vocab_path = CUR_DIR + 'data/codes.txt'
bpe_codes_fin = codecs.open(vocab_path)
bpe = BPE(bpe_codes_fin, merges=-1, separator='')

vocab_map = pd.read_csv(CUR_DIR + 'data/subword_units_map.csv')
idx2word = vocab_map['index'].values
words2idx = dict(zip(idx2word, range(0, len(idx2word))))
max_set = 30

def smiles2index(s1, s2):
    t1 = bpe.process_line(s1).split() #split
    t2 = bpe.process_line(s2).split() #split
    i1 = [words2idx[i] for i in t1] # index
    i2 = [words2idx[i] for i in t2] # index
    return i1, i2

def index2multi_hot(i1, i2):
    v1 = np.zeros(len(idx2word),)
    v2 = np.zeros(len(idx2word),)
    v1[i1] = 1
    v2[i2] = 1
    v_d = np.maximum(v1, v2)
    return v_d

def index2single_hot(i1, i2):
    comb_index  = set(i1 + i2)
    v_f = np.zeros((max_set*2, len(idx2word)))
    for i, j in enumerate(comb_index):
        if i < max_set*2:
            v_f[i][j] = 1
        else:
            break
    return v_f

def smiles2vector(s1, s2):
    i1, i2 = smiles2index(s1, s2)
    v_d = index2multi_hot(i1, i2)
    #v_f = index2single_hot(i1, i2)
    return v_d

class supData(data.Dataset):
    def __init__(self, df_ddi, smiles_store, split):
        self.split = split
        self.smiles_store = smiles_store
        self.labels = df_ddi['label_indexed'].values.tolist()
        self.heads = df_ddi['head'].values.tolist()
        self.tails = df_ddi['tail'].values.tolist()
        if split not in {'val_between', 'test_between'}:
            self.neg_heads = df_ddi['neg_head'].values.tolist()
            self.neg_tails = df_ddi['neg_tail'].values.tolist()
        else:
            self.neg_tail_1s = df_ddi['neg_tail_1'].values.tolist()
            self.neg_tail_2s = df_ddi['neg_tail_2'].values.tolist()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        s_h = self.smiles_store[self.heads[index]]
        s_t = self.smiles_store[self.tails[index]]
        v_d_pos = smiles2vector(s_h, s_t)
        if self.split not in {'val_between', 'test_between'}:
            s_neg_head = self.smiles_store[self.neg_heads[index]]
            s_neg_tail = self.smiles_store[self.neg_tails[index]]
            v_d_neg_1 = smiles2vector(s_neg_head, s_t)
            v_d_neg_2 = smiles2vector(s_h, s_neg_tail)
        else:
            s_neg_tail_1 = self.smiles_store[self.neg_tail_1s[index]]
            s_neg_tail_2 = self.smiles_store[self.neg_tail_2s[index]]
            v_d_neg_1 = smiles2vector(s_h, s_neg_tail_1)
            v_d_neg_2 = smiles2vector(s_h, s_neg_tail_2)
        label = self.labels[index]
        return v_d_pos, label, v_d_neg_1, v_d_neg_2
    
class unsupData(data.Dataset):

    def __init__(self, list_IDs, df):
        'Initialization'
        self.list_IDs = list_IDs
        self.df = df
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        index = self.list_IDs[index]
        s1 = self.df.iloc[index].input1_SMILES
        s2 = self.df.iloc[index].input2_SMILES
        v_d = smiles2vector(s1, s2)
        return v_d
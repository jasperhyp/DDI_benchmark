"""
Preprocessing for DeepDDI binary PrimeKG

DO NOT USE. THIS FILE IS INCORRECT. 


"""
import pandas as pd
import numpy as np
import math, wandb, os, random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import networkx as nx

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch, os, json, pickle
# from cmapPy.pandasGEXpress.parse import parse
from rdkit import Chem
from torch_geometric.utils import is_undirected, to_undirected, to_dense_adj
from torch_geometric.transforms import AddMetaPaths, RandomLinkSplit
from torch_geometric.data import HeteroData, Data
from sklearn.model_selection import train_test_split
import torchdrug.data as td_data

SEED = 43
# NUM_BASIS_DRUGS = 1000 ## matching their setup exactly
NUM_BASIS_DRUGS = None ## use the entire train set as the basis drugs

random.seed(SEED)


######
# Preprocessing Step 1
######

print("Preprocessing Step #1: Sampling basis drugs and getting fingerprints...")

## Switch Path base to the relevant one of interest
path_base = "/n/data1/hms/dbmi/zitnik/lab/users/yeh803/DDI/processed_data/PrimeKG-DDI/drug_centric_easy_split/"
pretrain_drugs = pd.read_csv(path_base + "/kgstr_pretrain_compound_lookup.csv", index_col=0)
ddi_drugs = pd.read_csv(path_base + "/kgddi_drug_split.csv", index_col=0)

## checking no data leakage
assert set(ddi_drugs[ddi_drugs.split=='val'].node_id) & set(pretrain_drugs.node_id) == set()
assert set(ddi_drugs[ddi_drugs.split=='test'].node_id) & set(pretrain_drugs.node_id) == set()

### For DeepDDI, all we need is to
## grab N number of drugs from the training set, and then use those as the
## basis drugs
pretrain_drugs['split'] = 'pretrain'
all_drugs = pd.concat([ddi_drugs[['node_id', 'canonical_smiles', 'view_kg', 'split']], pretrain_drugs], axis=0).drop_duplicates(subset='node_id')
train_drugs = all_drugs[all_drugs.split.isin({'train', 'pretrain'})]  # train drugs here are ddi train drugs & pretrain drugs


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import pandas as pd

# number of basis drugs here is at most the number of drugs that exists in the train set

if NUM_BASIS_DRUGS is None:
    NUM_BASIS_DRUGS = train_drugs.shape[0]

assert NUM_BASIS_DRUGS <= train_drugs.shape[0]

# make a list of mols
all_smiles = all_drugs.canonical_smiles.values
all_mols = [AllChem.AddHs(Chem.MolFromSmiles(sm)) for sm in all_smiles] 

# Morgan FP
all_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256) for mol in all_mols]

# for randomly sampled NUM_BASIS_DRUGS drugs in train drugs
rng = np.random.default_rng(seed=SEED)
basis_mols = [AllChem.AddHs(Chem.MolFromSmiles(sm)) for sm in train_drugs.canonical_smiles.values[rng.choice(np.arange(train_drugs.shape[0]), NUM_BASIS_DRUGS, replace=False)]]
basis_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=256) for x in basis_mols]
assert len(basis_fps) == NUM_BASIS_DRUGS

###
# calculating the tanimoto/dice similarity [0,1] between the train drugs and the rest of the drugs in the dataset
###
all_sim_scores = [] ## num_drugs x NUM_BASIS_DRUGS
basis_sim_scores = [] ## NUM_BASIS_DRUGS x NUM_BASIS_DRUGS

# https://stackoverflow.com/questions/51681659/how-to-use-rdkit-to-calculte-molecular-fingerprint-and-similarity-of-a-list-of-s
# BulkDiceSimilarity, BulkTanimotoSimilarity
for n in range(len(all_fps)):
    s = DataStructs.BulkTanimotoSimilarity(all_fps[n], basis_fps) # +1 compare with the next to the last fp
    all_sim_scores.append(s)
    
## creating the NxN matrix for learning PCA for
for n in range(len(basis_fps)):
    s = DataStructs.BulkTanimotoSimilarity(basis_fps[n], basis_fps) # +1 compare with the next to the last fp
    basis_sim_scores.append(s)

all_sim_scores = pd.DataFrame(all_sim_scores)
basis_sim_scores = pd.DataFrame(basis_sim_scores)


print("Finished! Getting PCA and fitting on the entire dataset")
### scaling the data for PCA
scaler = StandardScaler()
scaler.fit(basis_sim_scores)
scaled_basis_sim_scores = scaler.transform(basis_sim_scores)

### PCA with 50 components, the same that they use
pca = PCA(n_components=50)

### fitting the data
pca.fit(scaled_basis_sim_scores)


####
# Run the PCA on every compound in the lookup
####

### scaling the data for PCA
#### https://bitbucket.org/kaistsystemsbiology/deepddi/src/master/deepddi/preprocessing.py 
scaler = StandardScaler()
scaler.fit(all_sim_scores)
scaled_df = scaler.transform(all_sim_scores)

### transforming it 
pca_data = pca.transform(scaled_df)
### saving it as a dataframe
pca_lookup = pd.DataFrame(pca_data, columns=['PC_%d' % (i + 1) for i in range(50)], index=all_sim_scores.index)
# pca_lookup = pca_lookup.T
pca_lookup

pca_lookup["node_id"] = all_drugs.node_id.values
pca_lookup.to_csv(path_base + f"/deepDDI/lookup_table_of_pca_{NUM_BASIS_DRUGS}_seed{SEED}.csv")


######
# Preprocessing Step 2, constructing the PCA compressed drug representations to be taken as input
######

print("Preprocessing Step 2, constructing the PCA compressed drug representations to be taken as input")

import os, sys
from itertools import product
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torchdrug import core, models, tasks, utils
from torch_geometric.utils import is_undirected, to_undirected

## reading it back in
pca_lookup = pd.read_csv(path_base+f"/deepDDI/lookup_table_of_pca_{NUM_BASIS_DRUGS}_seed{SEED}.csv", index_col=0).drop(columns=['node_id'])
drugs = pd.read_csv(path_base+"/kgddi_drug_split.csv", index_col=0)
train_edgelist = torch.load(path_base+'/train_edgelist.pt')
negative_train_edgelist = torch.load(path_base+'/negative_train_edgelist.pt')

val_between_edgelist = torch.load(path_base+'/val_between_edgelist.pt')
val_within_edgelist = torch.load(path_base+'/val_within_edgelist.pt')
negative_val_between_edgelist = torch.load(path_base+'/negative_val_between_edgelist.pt')
negative_val_within_edgelist = torch.load(path_base+'/negative_val_within_edgelist.pt')


train_drugs_df = drugs[drugs.split=='train']
train_drugs_df['view_str'] = 1
val_drugs_df = drugs[drugs.split=='val']
val_drugs_df['view_str'] = 1


## Direct conversion and save as a pytorch objects
lookup = pca_lookup.T

##
# val between
##
head_pos = lookup[val_between_edgelist[:, 0]].values.T 
tail_pos = lookup[val_between_edgelist[:, 1]].values.T
val_btn_pos = np.concatenate([head_pos, tail_pos], axis = 1)
val_btn_pos = torch.tensor(val_btn_pos)
val_btn_labels_pos = torch.ones(val_btn_pos.shape[0]) 

head_neg = lookup[negative_val_between_edgelist[:, 0]].values.T 
tail_neg = lookup[negative_val_between_edgelist[:, 1]].values.T
val_btn_neg = np.concatenate([head_neg, tail_neg], axis = 1)
val_btn_neg = torch.tensor(val_btn_neg)
val_btn_labels_neg = torch.zeros(val_btn_neg.shape[0]) 

val_btn_ddis = torch.cat([val_btn_pos, val_btn_neg], dim = 0)
val_btn_labels = torch.cat([val_btn_labels_pos, val_btn_labels_neg], dim = 0)

torch.save(val_btn_ddis, path_base + f'/deepDDI/val_btn_deepddi_{NUM_BASIS_DRUGS}_seed{SEED}.pt')
torch.save(val_btn_labels, path_base + f'/deepDDI/val_btn_deepddi_labels_{NUM_BASIS_DRUGS}_seed{SEED}.pt')


##
# val within
##

val_within_edgelist
negative_val_within_edgelist

head_pos = lookup[val_within_edgelist[:, 0]].values.T 
tail_pos = lookup[val_within_edgelist[:, 1]].values.T
val_wtn_pos = np.concatenate([head_pos, tail_pos], axis = 1)
val_wtn_pos = torch.tensor(val_wtn_pos) 
val_wtn_labels_pos = torch.ones(val_wtn_pos.shape[0]) # positive labels

head_neg = lookup[negative_val_within_edgelist[:, 0]].values.T 
tail_neg = lookup[negative_val_within_edgelist[:, 1]].values.T
val_wtn_neg = np.concatenate([head_neg, tail_neg], axis = 1)
val_wtn_neg = torch.tensor(val_wtn_neg)
val_wtn_labels_neg = torch.zeros(val_wtn_neg.shape[0]) # negative labels

val_wtn_ddis = torch.cat([val_wtn_pos, val_wtn_neg], dim = 0)
val_wtn_labels = torch.cat([val_wtn_labels_pos, val_wtn_labels_neg], dim = 0)

torch.save(val_wtn_ddis, path_base + f'/deepDDI/val_wtn_deepddi_{NUM_BASIS_DRUGS}_seed{SEED}.pt')
torch.save(val_wtn_labels, path_base + f'/deepDDI/val_wtn_deepddi_labels_{NUM_BASIS_DRUGS}_seed{SEED}.pt')

train_edgelist = torch.load(path_base+ f'/train_edgelist.pt')
negative_train_edgelist = torch.load(path_base+ f'/negative_train_edgelist.pt')


head_pos = lookup[train_edgelist[:, 0]].values.T 
tail_pos = lookup[train_edgelist[:, 1]].values.T
train_pos = np.concatenate([head_pos, tail_pos], axis = 1)
train_pos = torch.tensor(train_pos) 
train_labels_pos = torch.ones(train_pos.shape[0]) # positive labels

head_neg = lookup[negative_train_edgelist[:, 0]].values.T 
tail_neg = lookup[negative_train_edgelist[:, 1]].values.T
train_neg = np.concatenate([head_neg, tail_neg], axis = 1)
train_neg = torch.tensor(train_neg)
train_labels_neg = torch.zeros(train_neg.shape[0]) # negative labels

train_ddis = torch.cat([train_pos, train_neg], dim = 0)
train_labels = torch.cat([train_labels_pos, train_labels_neg], dim = 0)

torch.save(train_ddis, path_base + f'/deepDDI/train_deepddi_{NUM_BASIS_DRUGS}_seed{SEED}.pt')
torch.save(train_labels, path_base + f'/deepDDI/train_deepddi_labels_{NUM_BASIS_DRUGS}_seed{SEED}.pt')

print("==> Finished Preprocessing Negative Samples and the validation sets")


##
# TRAIN for all possible combinations as negative pairs
##
assert is_undirected(train_edgelist.T)
all_train_pairs = to_undirected(torch.combinations(torch.unique(train_edgelist)).T).T

# Removing row duplicates should be easier using pandas
pos_train_df = pd.DataFrame(train_edgelist)
pos_train_df['label'] = 1
all_train_df = pd.DataFrame(all_train_pairs)
all_train_df['label'] = 0
# neg_train_df = pd.concat([pos_train_df, all_train_df]).reset_index().drop(columns=['index']).drop_duplicates(subset=[0, 1], keep=False).reindex(np.arange(all_train_df.shape[0])+pos_train_df.shape[0]).dropna()
all_train_df = pd.concat([pos_train_df, all_train_df]).reset_index().drop(columns=['index']).drop_duplicates(subset=[0, 1], keep='first')

left = pca_lookup.loc[all_train_df.loc[:, 0]].values
right = pca_lookup.loc[all_train_df.loc[:, 1]].values
label = all_train_df.loc[:, 'label'].values
train = torch.from_numpy(np.concatenate([left, right, np.expand_dims(label, axis=1)], axis = 1))  # each row = [pair feature, label]

train_ddis_no_neg = torch.from_numpy(np.concatenate([left, right], axis = 1))  # each row = [pair feature, label]
train_labels_no_neg = torch.from_numpy(label).reshape(-1, 1)

torch.save(train_ddis_no_neg, path_base + f'/deepDDI/train_NO_NEGSAMP_{NUM_BASIS_DRUGS}_seed{SEED}.pt')
torch.save(train_labels_no_neg, path_base + f'/deepDDI/train_NO_NEGSAMP_labels_{NUM_BASIS_DRUGS}_seed{SEED}.pt')

print("==> Finished NOT Negative Samples (using all possible combinations as the negative pairs)")



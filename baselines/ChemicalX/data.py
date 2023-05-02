import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from rdkit import Chem

import torch
from torchdrug.data.feature import ExtendedConnectivityFingerprint
from torchdrug.data import PackedGraph, PackedMolecule

from subword_nmt.apply_bpe import BPE
import codecs

DATA_DIR = '/n/data1/hms/dbmi/zitnik/lab/users/yeh803/DDI/processed_data/'
CASTER_DIR = "/home/yeh803/workspace/DDI/NovelDDI/baselines/CASTER/DDE/"
MOL_FEAT_DIM = 128
ATOM_FEAT_DIM = 66
BOND_FEAT_DIM = 16


def get_mol_data(mol_feat_dim=128, mol_feat_type='ecfp'):
    """ Load smiles, features, molecular graphs
    """
    # load smiles
    ind2smiles = pd.read_csv(DATA_DIR+'views_features/combined_metadata_reindexed_ddi.csv', index_col=0)[['canonical_smiles']]
    
    # load ecfp features
    if os.path.exists(DATA_DIR+f'views_features/all_ecfp_{mol_feat_dim}_torchdrug.npy') and mol_feat_type == 'ecfp':
        ecfp_features = np.load(DATA_DIR+f'views_features/all_ecfp_{mol_feat_dim}_torchdrug.npy')
    elif mol_feat_type == 'caster':
        ecfp_features = None
    else:
        ecfp_features = [ExtendedConnectivityFingerprint(Chem.MolFromSmiles(smiles), radius=2, length=mol_feat_dim) for smiles in ind2smiles.canonical_smiles.values]
        ecfp_features = np.asarray(ecfp_features).astype(float)
        np.save(DATA_DIR+f'views_features/all_ecfp_{mol_feat_dim}_torchdrug.npy', ecfp_features)
        
    # load mol graphs
    mol_graphs = torch.load(DATA_DIR+'views_features/all_molecules_torchdrug.pt')
        
    return ind2smiles, ecfp_features, mol_graphs


@dataclass
class ChemicalXBatch:
    drug_features_left: torch.Tensor
    drug_features_right: torch.Tensor
    drug_molecules_left: PackedMolecule
    drug_molecules_right: PackedMolecule
    ys: torch.FloatTensor
    labels: torch.LongTensor
    
    def to(self, device):
        self.drug_features_left = self.drug_features_left.to(device)
        self.drug_features_right = self.drug_features_right.to(device)
        self.drug_molecules_left = self.drug_molecules_left.to(device)
        self.drug_molecules_right = self.drug_molecules_right.to(device)
        self.ys = self.ys.to(device)
        self.labels = self.labels.to(device)


class ChemicalXDataset(torch.utils.data.Dataset):
    def __init__(self, path_base, split, split_method, mol_graphs, mol_feats, smiles_store, **kwargs):
        self.mol_graphs = mol_graphs
        if mol_feats is not None:
            self.mol_feats = mol_feats if isinstance(mol_feats, torch.Tensor) else torch.from_numpy(mol_feats)
        self.smiles_store = smiles_store
        self.split = split
        self.split_method = split_method
        self.kwargs = kwargs
        if self.split not in {'val_between', 'test_between'}:
            df = pd.read_csv(path_base+f'{split_method}/{split}_df.csv')[['head', 'tail', 'label_indexed', 'neg_head', 'neg_tail']]
            self.heads, self.tails, self.labels, self.neg_1s, self.neg_2s = zip(*df.itertuples(index=False))
        else:
            df = pd.read_csv(path_base+f'{split_method}/{split}_df.csv')[['head', 'tail', 'label_indexed', 'neg_tail_1', 'neg_tail_2']]
            self.heads, self.tails, self.labels, self.neg_1s, self.neg_2s = zip(*df.itertuples(index=False))
        
        self.__post_init__()
        
    def __getitem__(self, index):
        return (
            self.heads[index], 
            self.tails[index], 
            self.labels[index], 
            self.neg_1s[index], 
            self.neg_2s[index]
        )
    
    def __len__(self):
        return len(self.heads)
    
    def __post_init__(self):
        pass
    
    def collate_fn(self, batch):
        pos_heads, pos_tails, labels, neg_1s, neg_2s = zip(*batch)
        pos_heads = torch.tensor(pos_heads)
        pos_tails = torch.tensor(pos_tails)
        pos_labels = torch.tensor(labels)
        neg_1s = torch.tensor(neg_1s)
        neg_2s = torch.tensor(neg_2s)
        
        if self.split not in {'val_between', 'test_between'}:
            neg_heads = torch.cat([pos_heads, neg_1s])
            neg_tails = torch.cat([neg_2s, pos_tails])
        else:
            neg_heads = torch.cat([pos_heads, pos_heads])
            neg_tails = torch.cat([neg_1s, neg_2s])
        neg_labels = torch.cat([pos_labels, pos_labels])
        
        all_heads = torch.cat([pos_heads, neg_heads, pos_tails, neg_tails])
        all_tails = torch.cat([pos_tails, neg_tails, pos_heads, neg_heads])
        all_labels = torch.cat([pos_labels, neg_labels, pos_labels, neg_labels])
        all_ys = torch.cat([torch.ones_like(pos_labels), torch.zeros_like(neg_labels), torch.ones_like(pos_labels), torch.zeros_like(neg_labels)])
        
        head_mol_graphs = self.mol_graphs[all_heads]  # NOTE: On the condition that the indices are sorted from 0 to NUM_ALL_DRUGS-1
        tail_mol_graphs = self.mol_graphs[all_tails]
        
        head_mol_feats = self.mol_feats[all_heads]
        tail_mol_feats = self.mol_feats[all_tails]

        return ChemicalXBatch(
            drug_features_left=head_mol_feats,
            drug_features_right=tail_mol_feats,
            drug_molecules_left=head_mol_graphs,
            drug_molecules_right=tail_mol_graphs,
            ys=all_ys,
            labels=all_labels
        )


### CASTER ###
class CASTERDataset(ChemicalXDataset):
    def __post_init__(self):
        assert 'mol_feat_type' in self.kwargs.keys()
        if self.kwargs['mol_feat_type'] == 'ecfp':
            pass
        elif self.kwargs['mol_feat_type'] == 'caster':
            caster_vocab_path = CASTER_DIR + 'data/codes.txt'
            caster_bpe_codes_fin = codecs.open(caster_vocab_path)
            self.bpe = BPE(caster_bpe_codes_fin, merges=-1, separator='')

            caster_vocab_map = pd.read_csv(CASTER_DIR + 'data/subword_units_map.csv')
            self.idx2word = caster_vocab_map['index'].values
            self.words2idx = dict(zip(self.idx2word, range(0, len(self.idx2word))))
            # caster_max_set = 30
            
            self.mol_feats = torch.from_numpy(np.stack([self.index2multihot(self.smiles2index(s)) for s in self.smiles_store]))
    
    def smiles2index(self, s):
        t = self.bpe.process_line(s).split() #split
        i = [self.words2idx[i] for i in t] # index
        return i

    def index2multihot(self, i):
        v = np.zeros(len(self.idx2word), )
        v[i] = 1.
        return v


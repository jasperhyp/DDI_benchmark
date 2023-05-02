import math, os, random, logging, argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from novelddi.evaluate.metrics import get_metrics
from novelddi.utils import get_root_logger
from novelddi.evaluate.eval_utils import K

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# from torchdrug.models import GraphAttentionNetwork, GraphIsomorphismNetwork, MessagePassingNeuralNetwork, NeuralFingerprint, SchNet 
from torchdrug.data import PackedMolecule

import deepchem as dc
from deepchem.models.torch_models import GroverModel, DMPNNModel, AttentiveFPModel
from deepchem.feat.vocabulary_builders import (GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder)
from deepchem.feat import DMPNNFeaturizer
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

SEED = 42
NUM_LABELS = 792
MOL_FEAT_DIM = 66  # Default feat dim in torchdrug
EDGE_FEAT_DIM = 18  # Default edge feat dim in torchdrug

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description='DeepDDI')
    parser.add_argument('--path_base', type=str, default='/n/data1/hms/dbmi/zitnik/lab/users/yeh803/DDI/processed_data/polypharmacy/TWOSIDES/', help='Base directory')
    parser.add_argument('--output_dir', type=str, default='/n/data1/hms/dbmi/zitnik/lab/users/yeh803/DDI/model_output/TWOSIDES/', help='Output directory')
    parser.add_argument('--split_method', type=str, default='split_by_pairs')
    parser.add_argument('--num_epoch', type=int, default=300, help='epoch num')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--mlp_dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[512, 1024], help='hidden dim sizes')
    parser.add_argument('--feature_dim', type=int, default=128, help='feature dim size (output of mol encoder, input to mlp)')
    
    # for general mol encoder
    parser.add_argument('--mol_encoder_name', type=str, default='gin', choices=['gin', 'gat', 'mpnn', 'neuralfp', 'schnet'])
    parser.add_argument('--mol_encoder_hidden_dims', type=int, nargs='+', default=[128, 128], help='Molecule encoder hidden dim')
    parser.add_argument('--mol_encoder_edge_input_dim', type=int, default=None, help='Molecule encoder edge input dim')
    parser.add_argument('--mol_encoder_concat_hidden', action='store_true', help='Whether to concat hidden states of GNN layers')
    parser.add_argument('--mol_encoder_activation', type=str, default='relu', choices=['relu', 'elu', 'leaky_relu', 'tanh', 'softplus', 'gelu', 'shifted_softplus', 'sigmoid', 'softsign', 'none'], help='Molecule encoder activation func')
    
    # gat
    parser.add_argument('--gat_num_heads', type=int, default=4, help='GAT num heads')
    parser.add_argument('--gat_negative_slope', type=float, default=0.2, help='GAT negative slope')
    
    # gin
    parser.add_argument('--gin_num_mlp_layers', type=int, default=2, help='GIN num mlp layers')
    parser.add_argument('--gin_eps', type=float, default=0, help='GIN eps')
    
    # mpnn
    parser.add_argument('--mpnn_num_gru_layers', type=int, default=1, help='MPNN num gru layers')
    parser.add_argument('--mpnn_num_s2s_steps', type=int, default=3, help='MPNN num s2s steps')
    parser.add_argument('--mpnn_num_mlp_layers', type=int, default=2, help='MPNN num mlp layers')
    
    # neuralfp
    pass

    # schnet
    parser.add_argument('--schnet_cutoff', type=float, default=5.0, help='SchNet cutoff')
    parser.add_argument('--schnet_num_gaussians', type=int, default=100, help='SchNet num gaussians')

    args = parser.parse_args()

    return args

args = parse_args()
logger = get_root_logger(args.output_dir+f'{args.split_method}/baselines/{args.mol_encoder_name}.log')
logger.info(f'Args: \n{args}')

class DDIDataset(Dataset):
    def __init__(self, path_base: str, split_method: str, split: str, mol_graphs: PackedMolecule):
        self.split_method = split_method
        self.split = split
        
        if self.split not in {'val_between', 'test_between'}:
            df = pd.read_csv(path_base+f'{split_method}/{split}_df.csv')[['head', 'tail', 'label_indexed', 'neg_head', 'neg_tail']]
            self.heads, self.tails, self.labels, self.neg_1s, self.neg_2s = zip(*df.itertuples(index=False))
        else:
            df = pd.read_csv(path_base+f'{split_method}/{split}_df.csv')[['head', 'tail', 'label_indexed', 'neg_tail_1', 'neg_tail_2']]
            self.heads, self.tails, self.labels, self.neg_1s, self.neg_2s = zip(*df.itertuples(index=False))
        
        self.mol_graphs = mol_graphs
        
    def __getitem__(self, index):
        return (
            self.heads[index], 
            self.tails[index], 
            self.labels[index], 
            self.neg_1s[index], 
            self.neg_2s[index]
        )
    
    def  __len__(self):
        return len(self.heads)
    
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
        
        unique_indices, all_new = torch.unique(torch.cat([pos_heads, neg_heads, pos_tails, neg_tails], dim=0), return_inverse=True)
        pos_heads_new, neg_heads_new, pos_tails_new, neg_tails_new = torch.split(all_new, [len(pos_heads), len(neg_heads), len(pos_tails), len(neg_tails)])
        
        # make ddi bidirectional
        # all_heads_new = torch.cat([pos_heads_new, neg_heads_new])
        all_heads_new = torch.cat([pos_heads_new, neg_heads_new, pos_tails_new, neg_tails_new])
        # all_tails_new = torch.cat([pos_tails_new, neg_tails_new])
        all_tails_new = torch.cat([pos_tails_new, neg_tails_new, pos_heads_new, neg_heads_new])
        # all_labels_new = torch.cat([pos_labels, neg_labels])
        all_labels_new = torch.cat([pos_labels, neg_labels, pos_labels, neg_labels])
        # all_ys_new = torch.cat([torch.ones_like(pos_labels), torch.zeros_like(neg_labels)])
        all_ys_new = torch.cat([torch.ones_like(pos_labels), torch.zeros_like(neg_labels), torch.ones_like(pos_labels), torch.zeros_like(neg_labels)])
        
        unique_mol_graphs = self.mol_graphs[unique_indices]  # NOTE: On the condition that the indices are sorted from 0 to NUM_ALL_DRUGS-1
        
        return {
            'head': all_heads_new,
            'tail': all_tails_new,
            'label': all_labels_new,
            'y': all_ys_new,
            'mol_graph': unique_mol_graphs,
        }

logger.info('==> Loading Data')

mol_graphs = torch.load("/n/data1/hms/dbmi/zitnik/lab/users/yeh803/DDI/processed_data/views_features/all_molecules_torchdrug.pt")  # torchdrug mol graphs

SPLIT_METHOD2VAL_SPLITS = {'split_by_triplets':['val'], 'split_by_pairs':['val'], 'split_by_drugs_random':['val_between', 'val_within']}
SPLIT_METHOD2TEST_SPLITS = {'split_by_triplets':['test'], 'split_by_pairs':['test'], 'split_by_drugs_random':['test_between', 'test_within']}
train_dataset = DDIDataset(args.path_base, args.split_method, 'train', mol_graphs)
val_datasets = [DDIDataset(args.path_base, args.split_method, split, mol_graphs) for split in SPLIT_METHOD2VAL_SPLITS[args.split_method]]
test_datasets = [DDIDataset(args.path_base, args.split_method, split, mol_graphs) for split in SPLIT_METHOD2TEST_SPLITS[args.split_method]]

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=args.num_workers, pin_memory=True)
val_loaders = [DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn, num_workers=args.num_workers, pin_memory=True) for val_dataset in val_datasets]
test_loaders = [DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=args.num_workers, pin_memory=True) for test_dataset in test_datasets]

####### MODEL #######
logger.info("==> Initializing Model")

####
# Training the deep learning model according to DeepDDI
# 8 layers
# RELU activation function
# single output, use Binary Cross Entropy loss on the single
####

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0):
        super(SimpleMLP, self).__init__()
        assert len(hidden_dims) >= 2
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.act1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)-1):
            self.layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.LayerNorm(hidden_dims[i+1]), nn.ReLU(inplace=True), nn.Dropout(dropout)])
        
        self.out_fc = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        x = self.dropout1(self.act1(self.ln1(self.fc1(x))))
        x = self.layers(x)

        return self.out_fc(x)
    
MOL_ENCODERS = {
    "gin": GraphIsomorphismNetwork,
    "gat": GraphAttentionNetwork,
    "neuralfp": NeuralFingerprint,
    "mpnn": MessagePassingNeuralNetwork,
    "schnet": SchNet,
}

MOL_ENCODER_PARAMS = {
    "gin": {
        "input_dim": MOL_FEAT_DIM,
        "hidden_dims": args.mol_encoder_hidden_dims + [args.feature_dim],
        "edge_input_dim": args.mol_encoder_edge_input_dim,
        "num_mlp_layer": args.gin_num_mlp_layers,
        "eps": args.gin_eps,
        "activation": args.mol_encoder_activation,
        "concat_hidden": args.mol_encoder_concat_hidden,
    },
    "gat": {
        "input_dim": MOL_FEAT_DIM,
        "hidden_dims": args.mol_encoder_hidden_dims + [args.feature_dim],
        "edge_input_dim": args.mol_encoder_edge_input_dim,
        "num_head": args.gat_num_heads,
        "negative_slope": args.gat_negative_slope,
        "activation": args.mol_encoder_activation,
        "concat_hidden": args.mol_encoder_concat_hidden,
    },
    "neuralfp": {
        "input_dim": MOL_FEAT_DIM,
        "output_dim": args.feature_dim,
        "hidden_dims": args.mol_encoder_hidden_dims,
        "edge_input_dim": args.mol_encoder_edge_input_dim,
        "activation": args.mol_encoder_activation,
        "concat_hidden": args.mol_encoder_concat_hidden,
    },
    "mpnn": {
        "input_dim": MOL_FEAT_DIM,
        "hidden_dim": args.feature_dim // 2,
        "edge_input_dim": args.mol_encoder_edge_input_dim if args.mol_encoder_edge_input_dim is not None else EDGE_FEAT_DIM,
        "num_layer": len(args.mol_encoder_hidden_dims),  # use this as a proxy
        "num_gru_layer": args.mpnn_num_gru_layers,
        "num_s2s_step": args.mpnn_num_s2s_steps,
        "activation": args.mol_encoder_activation,
        "concat_hidden": args.mol_encoder_concat_hidden,
    },
    "schnet": {
        "input_dim": MOL_FEAT_DIM,
        "hidden_dims": args.mol_encoder_hidden_dims,
        "edge_input_dim": args.mol_encoder_edge_input_dim,
        "cutoff": args.schnet_cutoff,
        "num_gaussian": args.schnet_num_gaussians,
        "activation": args.mol_encoder_activation,
        "concat_hidden": args.mol_encoder_concat_hidden,
    },
}

class SimpleDDI(nn.Module):
    def __init__(self, mol_encoder_name, feature_dim, mlp_hidden_dims, dropout = 0.):
        super(SimpleDDI, self).__init__()
        try:
            self.mol_encoder = MOL_ENCODERS[mol_encoder_name](**MOL_ENCODER_PARAMS[mol_encoder_name])
        except:
            raise ValueError(f"mol_encoder_name must be one of {list(MOL_ENCODERS.keys())}")
        self.mlp = SimpleMLP(input_dim=feature_dim*2, hidden_dims=mlp_hidden_dims, output_dim=NUM_LABELS, dropout=dropout)
    def forward(self, mols, h_inds, t_inds, labels):
        z_mols = self.mol_encoder(mols, mols.node_feature.float())
        z_mols = z_mols["graph_feature"]
        z_h = z_mols[h_inds, :]
        z_t = z_mols[t_inds, :]
        out = self.mlp(torch.cat([z_h, z_t], dim=-1))
        return out[torch.arange(labels.shape[0]).to(labels.device), labels]

model = SimpleDDI(args.mol_encoder_name, args.feature_dim, args.mlp_hidden_dims, dropout=args.mlp_dropout).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#### Training Model
logger.info('Start training')

model_save_path = args.output_dir + f'{args.split_method}/baselines/{args.mol_encoder_name}_best_model_{args.feature_dim}_{args.mol_encoder_edge_input_dim}.pt'
within_model_save_path = args.output_dir + f'{args.split_method}/baselines/{args.mol_encoder_name}_best_within_model_{args.feature_dim}_{args.mol_encoder_edge_input_dim}.pt'

@torch.no_grad()
def evaluate(loader, split, model, logger, device):
    logger.info(f"Evaluating on {split} set")
    
    model.eval()
    all_eval_preds = []
    all_eval_labels = []
    all_eval_ys = []
    for i, batch in enumerate(loader):
        mol_graphs = batch['mol_graph'].to(device)
        head_indices = batch['head'].to(device)
        tail_indices = batch['tail'].to(device)
        labels = batch['label'].to(device)

        preds = model(mol_graphs, head_indices, tail_indices, labels)
        ys = batch['y'].to(device)
        
        all_eval_preds.append(preds.detach().cpu().sigmoid().numpy())
        all_eval_labels.append(labels.detach().cpu().numpy())
        all_eval_ys.append(ys.detach().cpu().numpy())

    all_eval_preds = np.concatenate(all_eval_preds, axis=0)
    all_eval_labels = np.concatenate(all_eval_labels, axis=0)
    all_eval_ys = np.concatenate(all_eval_ys, axis=0)
    
    eval_metrics = get_metrics(preds=all_eval_preds, ys=all_eval_ys, labels=all_eval_labels, k=K, task='multilabel', logger=logger, average='macro', verbose=True)
    
    return eval_metrics

best_epoch = 0
best_within_epoch = 0
best_val_auroc = 0
best_within_val_auroc = 0
for epoch in range(args.num_epoch):
    logger.info(f"Epoch {epoch}/{args.num_epoch}")
    model.train()
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        
        mol_graphs = batch['mol_graph'].to(device)
        head_indices = batch['head'].to(device)
        tail_indices = batch['tail'].to(device)
        labels = batch['label'].to(device)

        preds = model(mol_graphs, head_indices, tail_indices, labels)
        ys = batch['y'].float().to(device)

        loss = loss_fn(preds, ys)
        loss.backward()
        optimizer.step()
        
    logger.info(f"\ntrain loss = {loss.item()}")

    if epoch % 1 == 0:
        if 'drugs' in args.split_method:
            val_metrics = evaluate(val_loaders[0], 'val_between', model, logger, device)
            val_within_metrics = evaluate(val_loaders[1], 'val_within', model, logger, device)
        else:
            val_metrics = evaluate(val_loaders[0], 'val', model, logger, device)
            val_within_metrics = None
        if val_metrics[4] > best_val_auroc:
            best_val_auroc = val_metrics[4]
            best_epoch = epoch
            torch.save({
                'best_epoch': epoch,
                'state_dict': model.state_dict(),
                'best_val_metrics': val_metrics,
                'val_within_metrics': val_within_metrics,
            }, model_save_path)
        if 'drugs' in args.split_method and val_within_metrics[4] > best_within_val_auroc:
            best_within_val_auroc = val_within_metrics[4]
            best_within_epoch = epoch
            torch.save({
                'best_epoch': epoch,
                'state_dict': model.state_dict(),
                'best_val_metrics': val_metrics,
                'val_within_metrics': val_within_metrics,
            }, within_model_save_path)

#### Testing Model
logger.info("Testing model...")

model.cpu().load_state_dict(torch.load(model_save_path)['state_dict'])
model.to(device).eval()

if 'drugs' in args.split_method:
    test_metrics = evaluate(test_loaders[0], 'test_between', model, logger, device)
    
    model.cpu().load_state_dict(torch.load(within_model_save_path)['state_dict'])
    model.to(device).eval()
    test_within_metrics = evaluate(test_loaders[1], 'test_within', model, logger, device)
    
    np.save(args.output_dir + f'{args.split_method}/baselines/{args.mol_encoder_name}_test_between_metrics_{args.feature_dim}_{args.mol_encoder_edge_input_dim}.npy', test_metrics)
    np.save(args.output_dir + f'{args.split_method}/baselines/{args.mol_encoder_name}_test_within_metrics_{args.feature_dim}_{args.mol_encoder_edge_input_dim}.npy', test_within_metrics)

else:
    test_metrics = evaluate(test_loaders[0], 'test', model, logger, device)    
    np.save(args.output_dir + f'{args.split_method}/baselines/{args.mol_encoder_name}_test_metrics_{args.feature_dim}_{args.mol_encoder_edge_input_dim}.npy', test_metrics)



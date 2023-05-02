import math, os, random, logging, argparse
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

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
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

SEED = 42
NUM_BASIS_DRUGS = {'split_by_pairs':1379, 'split_by_triplets':1391, 'split_by_drugs_random':962, 'split_by_drugs_hard':952, 'split_by_drugs_easy':1146}
NUM_LABELS = 792
PCA_DIM = 50

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
    parser.add_argument('--logger_fname', type=str, default='deepddi.log', help='Name of logger file')
    parser.add_argument('--split_method', type=str, default='split_by_pairs')
    parser.add_argument('--num_epoch', type=int, default=500, help='epoch num')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--feature_dim', type=int, default=256, help='hidden dim size')
    parser.add_argument('--num_layers', type=int, default=8, help='number of layers for deepDDI')
    # parser.add_argument('--train_type', type=str, default='no_negsamp', choices = ['negsamp_onfly', 'no_negsamp', 'norm_negsamp'], help='Base directory')

    args = parser.parse_args()

    return args

# args = parser.parse_args(args=[]) ## Jupyter Notebook
args = parse_args()

# #wandb = None
# wandb.init(project="deepDDI-twosides-pair-split", entity="noveldrugdrug")
# wandb.config = hparams
# ## rename the prefix based on the type of training for sweep purposes
# run_name = wandb.run.name
# wandb.run.name = str(hparams["train_type"]) + "_" + run_name
# print("==> Run.name: ", wandb.run.name)

logger = get_root_logger(args.output_dir+f'{args.split_method}/baselines/{args.logger_fname}')
logger.info(f'Args: \n{args}')

class DeepDDIDataset(Dataset):
    def __init__(self, path_base: str, split_method: str, split: str, pca_lookup: torch.Tensor):
        self.split_method = split_method
        self.split = split
        
        if self.split not in {'val_between', 'test_between'}:
            df = pd.read_csv(path_base+f'{split_method}/{split}_df.csv')[['head', 'tail', 'label_indexed', 'neg_head', 'neg_tail']]
            self.heads, self.tails, self.labels, self.neg_1s, self.neg_2s = zip(*df.itertuples(index=False))
        else:
            df = pd.read_csv(path_base+f'{split_method}/{split}_df.csv')[['head', 'tail', 'label_indexed', 'neg_tail_1', 'neg_tail_2']]
            self.heads, self.tails, self.labels, self.neg_1s, self.neg_2s = zip(*df.itertuples(index=False))
        
        self.pca_lookup = pca_lookup
        
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
        all_heads_new = torch.cat([pos_heads_new, neg_heads_new])
        all_tails_new = torch.cat([pos_tails_new, neg_tails_new])
        all_labels_new = torch.cat([pos_labels, neg_labels])
        all_ys_new = torch.cat([torch.ones_like(pos_labels), torch.zeros_like(neg_labels)])
        
        unique_pca_embeds = self.pca_lookup[unique_indices, :]  # NOTE: On the condition that the indices are sorted from 0 to NUM_ALL_DRUGS-1
        
        return {
            'head': all_heads_new,
            'tail': all_tails_new,
            'label': all_labels_new,
            'y': all_ys_new,
            'pca_embed': unique_pca_embeds,
        }

logger.info('==> Loading Data')

pca_lookup_df = pd.read_csv(args.path_base+f"{args.split_method}/DeepDDI/lookup_table_of_pca_{NUM_BASIS_DRUGS[args.split_method]}_seed{SEED}.csv", index_col=0).set_index('node_id')
pca_lookup_tensor = torch.load(args.path_base + f"{args.split_method}/DeepDDI/pca_all_mols_lookup_{NUM_BASIS_DRUGS[args.split_method]}_seed{SEED}.pt").float()
assert np.all(pca_lookup_df.index.values == np.arange(pca_lookup_df.shape[0]))

SPLIT_METHOD2VAL_SPLITS = {'split_by_triplets':['val'], 'split_by_pairs':['val'], 'split_by_drugs_random':['val_between', 'val_within']}
SPLIT_METHOD2TEST_SPLITS = {'split_by_triplets':['test'], 'split_by_pairs':['test'], 'split_by_drugs_random':['test_between', 'test_within']}
train_dataset = DeepDDIDataset(args.path_base, args.split_method, 'train', pca_lookup_tensor)
val_datasets = [DeepDDIDataset(args.path_base, args.split_method, split, pca_lookup_tensor) for split in SPLIT_METHOD2VAL_SPLITS[args.split_method]]
test_datasets = [DeepDDIDataset(args.path_base, args.split_method, split, pca_lookup_tensor) for split in SPLIT_METHOD2TEST_SPLITS[args.split_method]]

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

class SimpleDeepDDI(nn.Module):
    def __init__(self, inputdim: int, hiddendim: int, outputdim: int, num_layers: int):
        super(SimpleDeepDDI, self).__init__()
        assert num_layers >= 2
        self.fc = nn.Linear(inputdim, hiddendim)
        self.b1 = nn.BatchNorm1d(hiddendim)
        self.num_layers = num_layers
        
        self.layers = nn.Sequential()
        for i in range(max(num_layers - 2, 0)):
            self.layers.extend([nn.Linear(hiddendim, hiddendim), nn.BatchNorm1d(hiddendim), nn.ReLU(inplace=True)])
        
        self.out_fc = nn.Linear(hiddendim, outputdim)
        
    def forward(self, pca_h, pca_t, r):
        x = torch.cat([pca_h, pca_t], dim=-1)
        x = torch.relu(self.b1(self.fc(x)))
        x = self.layers(x)

        return self.out_fc(x)[torch.arange(r.shape[0]).to(r.device), r]

model = SimpleDeepDDI(inputdim = PCA_DIM * 2, hiddendim = args.feature_dim, outputdim = NUM_LABELS, num_layers = args.num_layers)
model.to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#### Training Model
logger.info('Start training')

model_save_path = args.output_dir + f'{args.split_method}/baselines/DeepDDI_best_model_{args.feature_dim}_{args.num_layers}.pt'
within_model_save_path = args.output_dir + f'{args.split_method}/baselines/DeepDDI_best_within_model_{args.feature_dim}_{args.num_layers}.pt'

@torch.no_grad()
def evaluate(loader, split, model, logger, device):
    logger.info(f"Evaluating on {split} set")
    
    model.eval()
    all_eval_preds = []
    all_eval_labels = []
    all_eval_ys = []
    for i, batch in enumerate(loader):
        head_pcas = batch['pca_embed'][batch['head'], :].to(device)
        tail_pcas = batch['pca_embed'][batch['tail'], :].to(device)
        ys = batch['y'].to(device)
        labels = batch['label'].to(device)

        preds = model(head_pcas, tail_pcas, labels)
        
        all_eval_preds.append(preds.cpu().detach().sigmoid().numpy())
        all_eval_labels.append(labels.cpu().detach().numpy())
        all_eval_ys.append(ys.cpu().detach().numpy())

    all_eval_preds = np.concatenate(all_eval_preds, axis=0)
    all_eval_labels = np.concatenate(all_eval_labels, axis=0)
    all_eval_ys = np.concatenate(all_eval_ys, axis=0)
    
    eval_metrics = get_metrics(preds=all_eval_preds, ys=all_eval_ys, labels=all_eval_labels, k=K, task='multilabel', logger=logger, average='macro', verbose=True)
    
    return eval_metrics

best_epoch = 0
best_within_epoch = 0
best_val_auroc = 0
best_val_within_auroc = 0
for epoch in range(args.num_epoch):
    logger.info(f"Epoch {epoch}/{args.num_epoch}")
    model.train()
    for i, batch in enumerate(tqdm(train_loader)):
        head_pcas = batch['pca_embed'][batch['head'], :].to(device)
        tail_pcas = batch['pca_embed'][batch['tail'], :].to(device)
        ys = batch['y'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        preds = model(head_pcas, tail_pcas, labels)
        loss = loss_fn(preds, ys.float())
        loss.backward()
        optimizer.step()
        
    logger.info(f"\ntrain loss = {loss.item()}")

    if epoch % 5 == 0: # and epoch is not 0
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
        if 'drugs' in args.split_method:
            if val_within_metrics[4] > best_val_within_auroc:
                best_val_within_auroc = val_within_metrics[4]
                best_within_epoch = epoch
                torch.save({
                    'best_epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_val_metrics': val_metrics,
                    'val_within_metrics': val_within_metrics,
                }, within_model_save_path)
        

#### Testing Model
model.cpu().load_state_dict(torch.load(model_save_path)['state_dict'])
model.to(device).eval()

if 'drugs' in args.split_method:
    test_metrics = evaluate(test_loaders[0], 'test_between', model, logger, device)
    
    model.cpu().load_state_dict(torch.load(within_model_save_path)['state_dict'])
    model.to(device).eval()
    test_within_metrics = evaluate(test_loaders[1], 'test_within', model, logger, device)
    
    np.save(args.output_dir + f'{args.split_method}/baselines/DeepDDI_test_between_metrics_{args.feature_dim}_{args.num_layers}.npy', test_metrics)
    np.save(args.output_dir + f'{args.split_method}/baselines/DeepDDI_test_within_metrics_{args.feature_dim}_{args.num_layers}.npy', test_within_metrics)
    
else:
    test_metrics = evaluate(test_loaders[0], 'test', model, logger, device)
    np.save(args.output_dir + f'{args.split_method}/baselines/DeepDDI_test_metrics_{args.feature_dim}_{args.num_layers}.npy', test_metrics)

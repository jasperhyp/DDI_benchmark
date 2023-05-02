import pickle, os, random, argparse
import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from torch.utils.data import DataLoader

from novelddi.evaluate.metrics import get_metrics
from novelddi.utils import get_root_logger
from novelddi.evaluate.eval_utils import K
from data import get_mol_data, CASTERDataset, DATA_DIR, MOL_FEAT_DIM, ATOM_FEAT_DIM, BOND_FEAT_DIM
from models import CustomCASTER, CustomMHCADDI, CustomSSIDDI  # CustomMRGNN


SEED = 42
OUT_DIM = 792
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/n/data1/hms/dbmi/zitnik/lab/users/yeh803/DDI/model_output/TWOSIDES/', help='Output directory')
    parser.add_argument('--model_name', type=str, default='caster', choices=['caster', 'deepddi', 'mhcaddi', 'ssiddi', 'mrgnn'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--split_method', type=str, required=True, choices=['split_by_drugs_random', 'split_by_pairs', 'split_by_triplets'])
    parser.add_argument('--num_epoch', type=int, default=300, help='epoch num')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--feature_dim', type=int, default=MOL_FEAT_DIM, help='feature dim size')
    parser.add_argument('--mol_feat_type', type=str, default='caster', choices=['caster', 'ecfp'])
    parser.add_argument('--out_channels', type=int, default=OUT_DIM)
    
    # caster, slightly changing the original hyperparam to accommodate our gpu
    parser.add_argument('--caster_recon_loss_coeff', type=float, default=0.1)
    parser.add_argument('--caster_lambda1', type=float, default=0.01)
    parser.add_argument('--caster_lambda2', type=float, default=0.1)
    parser.add_argument('--caster_lambda3', type=float, default=1e-5)
    parser.add_argument('--caster_magnifying_factor', type=int, default=100)
    parser.add_argument('--caster_encoder_hidden_channels', type=int, default=512)
    parser.add_argument('--caster_decoder_hidden_channels', type=int, default=512)
    parser.add_argument('--caster_hidden_channels', type=int, default=512)
    parser.add_argument('--caster_out_hidden_channels', type=int, default=512)
    parser.add_argument('--caster_proj_coeff', type=float, default=0.1)
    
    # mhcaddi, following hyperparams in the original paper https://arxiv.org/pdf/1905.00534.pdf
    parser.add_argument('--mhcaddi_atom_type_channels', type=int, default=100)  # MAX num of all possible atom types
    parser.add_argument('--mhcaddi_bond_type_channels', type=int, default=100)  # MAX num of all possible bond types
    parser.add_argument('--mhcaddi_node_channels', type=int, default=32)
    parser.add_argument('--mhcaddi_edge_channels', type=int, default=32)
    parser.add_argument('--mhcaddi_hidden_channels', type=int, default=32)
    parser.add_argument('--mhcaddi_readout_channels', type=int, default=32)
    parser.add_argument('--mhcaddi_dropout', type=float, default=0.2)
    
    # ssiddi, following hyperparam in the original paper 
    parser.add_argument('--ssiddi_hidden_channels', type=int, nargs='+', default=[64, 64, 64, 64])  # following other baselines, use dim=64; following original paper, using 4 layers
    parser.add_argument('--ssiddi_head_number', type=int, nargs='+', default=[2, 2, 2, 2])
    
    args = parser.parse_args()
    if args.mol_feat_type == 'caster':
        args.feature_dim = 1722
    
    return args


@torch.no_grad()
def evaluate(loader, split, model, model_name, logger, device):
    logger.info(f"Evaluating on {split} set")
    
    model.eval()
    all_eval_preds = []
    all_eval_labels = []
    all_eval_ys = []
    for i, batch in enumerate(loader):
        # if i > 5:
            # break
        
        batch.to(device)
        
        if model_name == 'caster':
            v_ds = model.unpack(batch)[0]
            preds, _, _, _, _, _ = model(v_ds)  # scores are not sigmoided
        elif model_name in {'mhcaddi', 'ssiddi'}:
            preds = model(*model.unpack(batch))
        else:
            raise NotImplementedError
            
        preds = preds[torch.arange(batch.labels.shape[0]), batch.labels]
        
        all_eval_preds.append(preds.detach().cpu().sigmoid().numpy())
        all_eval_labels.append(batch.labels.detach().cpu().numpy())
        all_eval_ys.append(batch.ys.detach().cpu().numpy())

    all_eval_preds = np.concatenate(all_eval_preds, axis=0)
    all_eval_labels = np.concatenate(all_eval_labels, axis=0)
    all_eval_ys = np.concatenate(all_eval_ys, axis=0)
    
    eval_metrics = get_metrics(preds=all_eval_preds, ys=all_eval_ys, labels=all_eval_labels, k=K, task='multilabel', logger=logger, average='macro', verbose=True)
    
    return eval_metrics


def train(train_loader, val_loaders, test_loaders, args, logger):
    logger.info("Setting up training...")
    if args.model_name == 'caster':
        model = CustomCASTER(drug_channels=next(iter(train_loader)).drug_features_left.shape[1], encoder_hidden_channels=args.caster_encoder_hidden_channels, decoder_hidden_channels=args.caster_decoder_hidden_channels, hidden_channels=args.caster_hidden_channels, out_hidden_channels=args.caster_out_hidden_channels, out_channels=args.out_channels, lambda3=args.caster_lambda3, magnifying_factor=args.caster_magnifying_factor)
    elif args.model_name == 'mhcaddi':
        model = CustomMHCADDI(atom_feature_channels=next(iter(train_loader)).drug_molecules_left.node_feature.shape[1], atom_type_channels=args.mhcaddi_atom_type_channels, bond_type_channels=args.mhcaddi_bond_type_channels, node_channels=args.mhcaddi_node_channels, edge_channels=args.mhcaddi_edge_channels, hidden_channels=args.mhcaddi_hidden_channels, readout_channels=args.mhcaddi_readout_channels, dropout=args.mhcaddi_dropout, output_channels=args.out_channels)
    elif args.model_name == 'ssiddi':
        model = CustomSSIDDI(molecule_channels=next(iter(train_loader)).drug_molecules_left.node_feature.shape[1], hidden_channels=args.ssiddi_hidden_channels, head_number=args.ssiddi_head_number, output_channels=args.out_channels)
    elif args.model_name == 'deepddi':
        raise NotImplementedError
    elif args.model_name == 'mrgnn':
        raise NotImplementedError
    else:
        raise ValueError('Invalid model name {}'.format(args.model_name))
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    logger.info('Start training')
    model_save_path = args.output_dir + f'{args.split_method}/baselines/{args.model_name}_best_model_{args.feature_dim}_{args.mol_feat_type}.pt'
    best_val_auroc = 0
    for epoch in range(args.num_epoch):
        model.train()
        
        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            batch.to(device)
            
            # if i > 5:
                # break
            
            if args.model_name == 'caster':
                v_ds = model.unpack(batch)[0]
                preds, recons, dictionary_encoded, dictionary_features_latent, drug_pair_features_latent, drug_pair_features = model(v_ds)  # scores are not sigmoided
                loss_c = loss_fn(preds[torch.arange(batch.labels.shape[0]), batch.labels], batch.ys.float())
                loss_r = args.caster_recon_loss_coeff * torch.nn.functional.binary_cross_entropy(recons, v_ds)
                # loss_p = args.caster_proj_coeff * (torch.norm(drug_pair_features_latent - torch.matmul(dictionary_encoded, dictionary_features_latent)) + args.caster_lambda1 * torch.abs(dictionary_encoded).sum() / v_ds.shape[0] + args.caster_lambda2 * torch.norm(dictionary_features_latent, p='fro') / v_ds.shape[0])
                # loss = loss_c + loss_r + loss_p
                loss = loss_c + loss_r
            
            elif args.model_name in {'mhcaddi', 'ssiddi'}:
                preds = model(*model.unpack(batch))
                loss = loss_fn(preds[torch.arange(batch.labels.shape[0]), batch.labels], batch.ys.float())
            
            loss.backward()
            optimizer.step()
        
            if i % 100 == 0:
                if args.model_name == 'caster':
                    logger.info(f'Epoch {epoch} Batch {i}: Loss {loss.item()}, Loss_c {loss_c.item()}, Loss_r {loss_r.item()}')
                else:
                    logger.info(f'Epoch {epoch} Batch {i}: Loss {loss.item()}')
        
        if epoch % 5 == 0 and args.num_epoch >= 10:
            logger.info('Evaluating...')
            model.eval()
        
            if 'drugs' in args.split_method:
                val_metrics = evaluate(val_loaders[0], 'val_between', model, args.model_name, logger, device)
                val_within_metrics = evaluate(val_loaders[1], 'val_within', model, args.model_name, logger, device)
                logger.info(f'Epoch {epoch}: Val between {val_metrics}')
                logger.info(f'Epoch {epoch}: Val within {val_within_metrics}')
            else:
                val_metrics = evaluate(val_loaders[0], 'val', model, args.model_name, logger, device)
                logger.info(f'Epoch {epoch}: Val {val_metrics}')
            if val_metrics[4] > best_val_auroc:
                best_val_auroc = val_metrics[4]
                torch.save({
                    'best_epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_val_metrics': val_metrics,
                    'val_within_metrics': val_within_metrics if 'drugs' in args.split_method else None,
                }, model_save_path)
    
    logger.info("Testing model...")
    if 'drugs' in args.split_method:
        test_metrics = evaluate(test_loaders[0], 'test_between', model, args.model_name, logger, device)
        test_within_metrics = evaluate(test_loaders[1], 'test_within', model, args.model_name, logger, device)
        np.save(args.output_dir + f'{args.split_method}/baselines/{args.model_name}_test_between_metrics_{args.feature_dim}_{args.mol_feat_type}.npy', test_metrics)
        np.save(args.output_dir + f'{args.split_method}/baselines/{args.model_name}_test_within_metrics_{args.feature_dim}_{args.mol_feat_type}.npy', test_within_metrics)
    else:
        test_metrics = evaluate(test_loaders[0], 'test', model, args.model_name, logger, device)    
        np.save(args.output_dir + f'{args.split_method}/baselines/{args.model_name}_test_metrics_{args.feature_dim}_{args.mol_feat_type}.npy', test_metrics)


def main(args, logger):
    logger.info("Loading data...")
    ind2smiles, ecfp_features, mol_graphs = get_mol_data(mol_feat_dim=args.feature_dim, mol_feat_type=args.mol_feat_type)
    
    train_dataset = CASTERDataset(DATA_DIR+'polypharmacy/TWOSIDES/', 'train', args.split_method, mol_graphs, ecfp_features, ind2smiles.values.flatten(), mol_feat_type=args.mol_feat_type)
    if 'drugs' in args.split_method:
        val_datasets = [CASTERDataset(DATA_DIR+'polypharmacy/TWOSIDES/', 'val_between', args.split_method, mol_graphs, ecfp_features, ind2smiles.values.flatten(), mol_feat_type=args.mol_feat_type), CASTERDataset(DATA_DIR+'polypharmacy/TWOSIDES/', 'val_within', args.split_method, mol_graphs, ecfp_features, ind2smiles.values.flatten(), mol_feat_type=args.mol_feat_type)]
        test_datasets = [CASTERDataset(DATA_DIR+'polypharmacy/TWOSIDES/', 'test_between', args.split_method, mol_graphs, ecfp_features, ind2smiles.values.flatten(), mol_feat_type=args.mol_feat_type), CASTERDataset(DATA_DIR+'polypharmacy/TWOSIDES/', 'test_within', args.split_method, mol_graphs, ecfp_features, ind2smiles.values.flatten(), mol_feat_type=args.mol_feat_type)]
    else:
        val_datasets = [CASTERDataset(DATA_DIR+'polypharmacy/TWOSIDES/', 'val', args.split_method, mol_graphs, ecfp_features, ind2smiles.values.flatten(), mol_feat_type=args.mol_feat_type)]
        test_datasets = [CASTERDataset(DATA_DIR+'polypharmacy/TWOSIDES/', 'test', args.split_method, mol_graphs, ecfp_features, ind2smiles.values.flatten(), mol_feat_type=args.mol_feat_type)]
    
    train_loader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loaders = [DataLoader(val_dataset, collate_fn=train_dataset.collate_fn, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4) for val_dataset in val_datasets]
    test_loaders = [DataLoader(test_dataset, collate_fn=train_dataset.collate_fn, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4) for test_dataset in test_datasets]
    
    train(train_loader, val_loaders, test_loaders, args, logger)


if __name__ == '__main__':
    args = parse_args()
    logger = get_root_logger(args.output_dir+f'{args.split_method}/baselines/{args.model_name}.log')
    logger.info(f'Args: \n{args}')
    main(args, logger)

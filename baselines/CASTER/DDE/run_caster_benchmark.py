import os, sys, random, argparse
CUR_DIR = "/home/yeh803/workspace/DDI/NovelDDI/baselines/CASTER/DDE/"

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils import data
from torch import nn 

from dde_torch import dde_NN_Large_Predictor
from stream_dde import supData, unsupData
from novelddi.evaluate.metrics import get_metrics
from novelddi.evaluate.eval_utils import K
from novelddi.utils import get_root_logger


seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/yeh803/DDI/"
DATA_DIR = BASE_DIR + "processed_data/polypharmacy/TWOSIDES/"
OUTPUT_DIR = BASE_DIR + "model_output/TWOSIDES/"
INPUT_DIM = 1722
NUM_LABELS = 792

# t = torch.cuda.get_device_properties(0).total_memory
# r = torch.cuda.memory_reserved(0)
# a = torch.cuda.memory_allocated(0)
# print("total memory", t)
# print("reserved memory", r)
# print("allocated memory", a)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--split_method', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_epoch', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument('--recon_threshold', type=float, default=0.0005)
    parser.add_argument('--lambda1', type=float, default=1e-2)
    parser.add_argument('--lambda2', type=float, default=1e-1)
    parser.add_argument('--lambda3', type=float, default=1e-5)
    parser.add_argument('--encode_fc1_dim', type=int, default=500)
    parser.add_argument('--encode_fc2_dim', type=int, default=50)
    parser.add_argument('--decode_fc1_dim', type=int, default=500)
    parser.add_argument('--decode_fc2_dim', type=int, default=INPUT_DIM)
    parser.add_argument('--predict_dim', type=int, default=1024)
    parser.add_argument('--predict_out_dim', type=int, default=NUM_LABELS)
    parser.add_argument('--reconstruction_coefficient', type=float, default=1e-1)
    parser.add_argument('--projection_coefficient', type=float, default=1e-1)
    parser.add_argument('--magnify_factor', type=float, default=100)

    args = parser.parse_args()
    
    return args

## evaluate
@torch.no_grad()
def test_dde_nn(loader, model, logger, device):
    all_preds = []
    all_labels = []
    all_ys = []
    model.eval()
    for i, (v_d_poss, labels, v_d_neg_1s, v_d_neg_2s) in enumerate(loader):
        logger.info(f"batch {i} / {len(loader)}")
        
        v_ds = torch.cat([v_d_poss, v_d_neg_1s, v_d_neg_2s], dim=0).float().to(device)
        labels = labels.repeat(3).to(device)
        recons, codes, scores, z_fs, z_ds = model(v_ds, labels)
        ys = torch.cat([torch.ones(v_d_poss.shape[0]), torch.zeros(v_d_neg_1s.shape[0]), torch.zeros(v_d_neg_2s.shape[0])], dim=0)
        all_ys.extend(ys.cpu())
        all_labels.extend(labels.cpu().flatten())
        all_preds.extend(torch.sigmoid(scores).squeeze().detach().cpu().flatten())
    
    return torch.stack(all_preds), torch.stack(all_labels), torch.stack(all_ys)  # stack zero-dim numbers into a vector


def main(args, output_dir, logger, device):
    data_dir = args.data_dir + args.split_method + '/'
    # pretrain_epoch = args.pretrain_epoch  ## loading in a pretrained model, so this is by default 0
    train_epoch = args.train_epoch
    lr = args.lr
    batch_size = args.batch_size # 256 apprears to be the max before CUDA memory dies
    num_workers = args.num_workers
    split_method = args.split_method    

    config = {
        'input_dim': INPUT_DIM,
        'num_class': NUM_LABELS,
        'lambda3': args.lambda3,
        'encode_fc1_dim': args.encode_fc1_dim,
        'encode_fc2_dim': args.encode_fc2_dim,
        'decode_fc1_dim': args.decode_fc1_dim,
        'decode_fc2_dim': args.decode_fc2_dim,
        'predict_dim': args.predict_dim,
        'predict_out_dim': args.predict_out_dim,
        'magnify_factor': args.magnify_factor,
    }

    thr = args.recon_threshold
    recon_loss_coeff = args.reconstruction_coefficient
    proj_coeff = args.projection_coefficient
    lambda1 = args.lambda1
    lambda2 = args.lambda2

    logger.info('--- Data Preparation ---')
    # df_ddi = pd.read_csv('data/BIOSNAP/sup_train_val.csv')  # ddi dataframe drug1_smiles, drug2_smiles
    df_ddi_train = pd.read_csv(data_dir+'train_df.csv')
    if 'drugs' in split_method:
        df_ddi_val_between = pd.read_csv(data_dir+'val_between_df.csv')
        df_ddi_val_within = pd.read_csv(data_dir+'val_within_df.csv')
        df_ddi_test_between = pd.read_csv(data_dir+'test_between_df.csv')
        df_ddi_test_within = pd.read_csv(data_dir+'test_within_df.csv')
    else:
        df_ddi_val = pd.read_csv(data_dir+'val_df.csv')
        df_ddi_test = pd.read_csv(data_dir+'test_df.csv')
        
    # ids_unsup = df_unsup.index.values
    # unsup_set = unsupData(ids_unsup, df_unsup)
    # unsup_generator = data.DataLoader(unsup_set, **params)

    smiles_store = pd.read_csv(BASE_DIR + 'processed_data/views_features/combined_metadata_reindexed_ddi.csv', index_col=0)['canonical_smiles'].values.flatten().tolist()

    train_dataset = supData(df_ddi_train, smiles_store, split='train')
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True, pin_memory=True)

    if 'drugs' in split_method:
        val_between_dataset = supData(df_ddi_val_between, smiles_store, split='val_between')
        val_between_loader = data.DataLoader(val_between_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        val_within_dataset = supData(df_ddi_val_within, smiles_store, split='val_within')
        val_within_loader = data.DataLoader(val_within_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
        test_between_dataset = supData(df_ddi_test_between, smiles_store, split='test_between')
        test_between_loader = data.DataLoader(test_between_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
        test_within_dataset = supData(df_ddi_test_within, smiles_store, split='test_within')
        test_within_loader = data.DataLoader(test_within_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    else:
        val_dataset = supData(df_ddi_val, smiles_store, split='val')
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = supData(df_ddi_test, smiles_store, split='test')
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    logger.info(f"Loading model from {CUR_DIR}model_pretrain_checkpoint_1.pt")

    model = dde_NN_Large_Predictor(**config)
    checkpoint_path = CUR_DIR + 'model_pretrain_checkpoint_1.pt'
    model_state_dict = torch.load(checkpoint_path).module.state_dict()
    pop_list = []
    for k, _ in model_state_dict.items():
        if k.startswith('predictor'):
            pop_list.append(k)
    for k in pop_list:
        model_state_dict.pop(k)
    msg = model.load_state_dict(model_state_dict, strict=False)
    assert sum([0 if k.startswith('predictor') else 1 for k in msg.missing_keys]) == 0, 'missing keys incorrect'
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)  # device_ids=[0, 1]
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr = lr)

    # logger.info('--- Pre-training Starts ---')
    # torch.backends.cudnn.benchmark = True
    # len_unsup = len(training_generator_sup)
    # for pre_epo in range(pretrain_epoch):
    #     for i, v_D in enumerate(training_generator_sup):
    #         v_D = v_D.float().cuda()
    #         recon, code, score, Z_f, z_D = model_nn.module(v_D)
    #         loss_r = recon_loss_coeff * F.binary_cross_entropy(recon, v_D.float())

    #         loss_p = proj_coeff * (torch.norm(z_D - torch.matmul(code, Z_f)) + lambda1 * torch.sum(torch.abs(code)) / BATCH_SIZE + lambda2 * torch.norm(Z_f, p='fro') / BATCH_SIZE)
    #         loss = loss_r + loss_p

    #         loss_r_history.append(loss_r)
    #         loss_p_history.append(loss_p)
    #         loss_history.append(loss)

    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #         if(i % 10 == 0):
    #             print('Pre-Training at Epoch ' + str(pre_epo) + ' iteration ' + str(i) + ', total loss is '
    #                   + '%.3f' % (loss.cpu().detach().numpy()) + ', proj loss is ' + '%.3f' % (loss_p.cpu().detach().numpy()) 
    #                   + ', recon loss is ' + '%.3f' % (loss_r.cpu().detach().numpy()))

    #         if loss_r < thr:
    #             # smaller than certain reconstruction error, -> go to training step
    #             break

    #     #     if i == int(len_unsup/4):
    #     #         torch.save(model_nn, 'model_pretrain_checkpoint_1.pt')
    #     #     if i == int(len_unsup/2):
    #     #         torch.save(model_nn, 'model_pretrain_checkpoint_1.pt')
    #     # torch.save(model_nn, 'model_nn_pretrain.pt')


    logger.info('--- DDI Training ---')

    loss_fn = torch.nn.BCEWithLogitsLoss()

    ## train
    # best_val_auroc = 0
    # loss_r_history = []
    # loss_p_history = []
    # loss_c_history = []
    # loss_history = []

    for epoch in range(train_epoch):
        logger.info('Training Epoch ' + str(epoch))
        model.train()
        
        for i, (v_d_poss, labels, v_d_neg_1s, v_d_neg_2s) in enumerate(tqdm(train_loader)):
            # tic = time.perf_counter()
            
            logger.info(f"Train batch {i} / {len(train_loader)}")
            optim.zero_grad()

            v_ds = torch.cat([v_d_poss, v_d_neg_1s, v_d_neg_2s], dim=0).float().to(device)
            labels = labels.repeat(3).to(device)
            recons, codes, scores, z_fs, z_ds = model(v_ds, labels)
            # recons, _, scores, _, _ = model(v_ds, labels)
            
            ys = torch.cat([torch.ones(v_d_poss.shape[0]), torch.zeros(v_d_neg_1s.shape[0]), torch.zeros(v_d_neg_2s.shape[0])], dim=0).float().to(device)

            loss_c = loss_fn(scores.squeeze(), ys)
            loss_r = recon_loss_coeff * F.binary_cross_entropy(recons, v_ds.float())
            loss_p = proj_coeff * (torch.norm(z_ds - torch.matmul(codes, z_fs)) + lambda1 * torch.abs(codes).sum() / batch_size + lambda2 * torch.norm(z_fs, p='fro') / batch_size)
            loss = loss_c + loss_r + loss_p
            # loss = loss_c + loss_r
            loss.backward()
            optim.step()
            
            # loss_r_history.append(loss_r.item())
            # loss_p_history.append(loss_p.item())
            # loss_c_history.append(loss_c.item())
            # loss_history.append(loss.item())
            logger.info({'total loss':loss.item(), 'recon loss_r':loss_r.item(), 'proj loss_p':loss_p.item(), 'classification loss_c' : loss_c.item()})

            # toc = time.perf_counter()
            # print(f"Ran Iteration in {toc - tic:0.4f} seconds")

        # Since we only train for very few epochs, don't really need to evaluate on validation set
        # if (epoch % 5 == 0):
        #     model.eval()
            
        #     if 'drugs' in split_method:
        #         val_within_metrics, _, _ = test_dde_nn(val_within_loader, model, logger, device)
        #         val_metrics, _, _ = test_dde_nn(val_between_loader, model, logger, device)
        #     else:
        #         val_metrics, _, _ = test_dde_nn(val_loader, model, logger, device)

        #     if val_metrics[4] > best_val_auroc:
        #         best_val_auroc = val_metrics[4]
        #         save_path = output_dir + 'CASTER_best_model.pt'
        #         if 'drugs' in split_method:
        #             torch.save({
        #                 'state_dict':model.module.state_dict(),
        #                 'best_epoch':epoch,
        #                 'best_val_metrics':val_metrics,
        #                 'val_within_metrics':val_within_metrics,
        #             }, save_path)
        #             logger.info(f'==> Val between at Epoch {epoch}\nAUROC: {val_metrics[4]:.4f}, AUPRC: {val_metrics[5]:.4f}, Fmax: {val_metrics[0]:.4f}, AP@{K}:  AUROC-within: {val_metrics[3]:.4f}')
        #             logger.info(f'==> Val within at Epoch {epoch}\nAUROC: {val_within_metrics[4]:.4f}, AUPRC: {val_within_metrics[5]:.4f}, Fmax: {val_within_metrics[0]:.4f}, AP@{K}:  AUROC-within: {val_within_metrics[3]:.4f}')
        #         else:
        #             torch.save({
        #                 'state_dict':model.module.state_dict(),
        #                 'best_epoch':epoch,
        #                 'best_val_metrics':val_metrics,
        #             }, save_path)
        #             logger.info(f'==> Val at Epoch {epoch}\nAUROC: {val_metrics[4]:.4f}, AUPRC: {val_metrics[5]:.4f}, Fmax: {val_metrics[0]:.4f}, AP@{K}:  AUROC-within: {val_metrics[3]:.4f}')

    logger.info('--- DDI Testing ---')
    
    if 'drugs' in split_method:
        all_preds, all_labels, all_ys = test_dde_nn(test_between_loader, model, logger, device)
        test_metrics = get_metrics(preds=all_preds.numpy(), ys=all_ys.numpy(), labels=all_labels.numpy(), k=K, task='multilabel', logger=None, average="macro", verbose=False)
        np.save(output_dir + f'baselines/CASTER_test_between_metrics.npy', test_metrics)
        
        all_preds, all_labels, all_ys = test_dde_nn(test_within_loader, model, logger, device)
        test_within_metrics = get_metrics(preds=all_preds.numpy(), ys=all_ys.numpy(), labels=all_labels.numpy(), k=K, task='multilabel', logger=None, average="macro", verbose=False)    
        np.save(output_dir + f'baselines/CASTER_test_within_metrics.npy', test_within_metrics)
    
    else:
        all_preds, all_labels, all_ys = test_dde_nn(test_loader, model, logger, device)
        test_within_metrics = get_metrics(preds=all_preds.numpy(), ys=all_ys.numpy(), labels=all_labels.numpy(), k=K, task='multilabel', logger=None, average="macro", verbose=False)
        np.save(output_dir + f'baselines/CASTER_test_metrics.npy', test_metrics)
    
    
if __name__ == '__main__':
    args = parse_args()
    
    output_dir = args.output_dir + args.split_method + '/'
    logger = get_root_logger(fname=output_dir+'baselines/CASTER.log')
    logger.info("Arguments: " + str(args))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args, output_dir, logger, device)

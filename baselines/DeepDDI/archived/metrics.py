####
# Helper File for training
####
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## Sklearn Metrics
# from sklearn.metrics import accuracy_score, precision_recall_curve, precision_score, recall_score, average_precision_score, roc_auc_score
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_curve, 
    precision_score, 
    recall_score, 
    average_precision_score, 
    roc_auc_score,
    cohen_kappa_score,
    f1_score,
    fbeta_score,
    top_k_accuracy_score,
    matthews_corrcoef,
)

from tqdm import tqdm

def fmax_score(y: np.ndarray, preds: np.ndarray, beta = 1.0, pos_label = 1):
    """
    Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
    """
    precision, recall, thresholds = precision_recall_curve(y_true = y, probas_pred = preds, pos_label = pos_label)
    precision += 1e-4
    recall += 1e-4
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return np.nanmax(f1), thresholds[np.argmax(f1)]


def precision_recall_at_k(y: np.ndarray, preds: np.ndarray, k: int, names: np.ndarray = None):
    """ Calculate recall@k, precision@k, and AP@k for binary classification.
    """
    assert preds.shape[0] == y.shape[0]
    assert k > 0
    assert k <= preds.shape[0]
    
    # Sort the scores and the labels by the scores
    sorted_indices = np.argsort(preds.flatten())[::-1]
    sorted_preds = preds[sorted_indices]
    sorted_y = y[sorted_indices]
    if names is not None:
        sorted_names = names[sorted_indices]
    else: sorted_names = None

    # Get the scores of the k highest predictions
    topk_preds = sorted_preds[:k]
    topk_y = sorted_y[:k]
    
    # Calculate the recall@k and precision@k
    recall_k = np.sum(topk_y) / np.sum(y)
    precision_k = np.sum(topk_y) / k
    
    # Calculate the AP@k
    ap_k = average_precision_score(topk_y, topk_preds)

    return recall_k, precision_k, ap_k, (sorted_y, sorted_preds, sorted_names)


def get_metrics_binary(pred_ddis, true_ddis, k = 100, logger = None, wandb = None, dataset = None):
    #pred = torch.sigmoid(pred_ddis)
    pred_ddis_tri, true_ddis_tri = pred_ddis[torch.tril_indices(pred_ddis.shape[0], pred_ddis.shape[1], -1).unbind()], true_ddis[torch.tril_indices(true_ddis.shape[0], true_ddis.shape[1], -1).unbind()]
    pred = pred_ddis_tri.detach().cpu().numpy()
    y = true_ddis_tri.detach().cpu().numpy()
    
    fmax, _ = fmax_score(y, pred)
    recall_k, precision_k, ap_k, _ = precision_recall_at_k(y, pred, k)
    auroc_score = roc_auc_score(y, pred)
    auprc_score = average_precision_score(y, pred)
    accuracy = accuracy_score(y, np.round(pred))
    precision = precision_score(y, np.round(pred))
    recall = recall_score(y, np.round(pred))
    f1 = 2 * precision * recall / (precision + recall)

    if logger is not None:
        logger.warning(f"Fmax = {fmax:.4f}, Recall@{k} = {recall_k:.4f}, precision@{k} = {precision_k:.4f}, ap@{k} = {ap_k:.4f}, auroc_score = {auroc_score:.4f}, auprc_score = {auprc_score:.4f}, accuracy = {accuracy:.4f}, precision = {precision:.4f}, recall = {recall:.4f}, F1 = {f1:.4f}")
    else:
        print(f"Fmax = {fmax:.4f}, Recall@{k} = {recall_k:.4f}, precision@{k} = {precision_k:.4f}, ap@{k} = {ap_k:.4f}, auroc_score = {auroc_score:.4f}, auprc_score = {auprc_score:.4f}, accuracy = {accuracy:.4f}, precision = {precision:.4f}, recall = {recall:.4f}, F1 = {f1:.4f}")


    if wandb is not None and dataset is not None:
        wandb.log({dataset + '_fmax': fmax,
                dataset + '_recall_k': recall_k,
                dataset + '_auroc_score': auroc_score,
                dataset + '_auprc_score': auprc_score,
                dataset + '_accuracy': accuracy,
                dataset + '_precision': precision,
                dataset + '_recall': recall,
                dataset + '_f1': f1
                })

    return fmax, recall_k, precision_k, ap_k, auroc_score, auprc_score, accuracy, precision, recall, f1





#####
# Metrics
###### 
# get_metrics_multilabel(preds = pred, ys = y, k = 10, logger = logger, verbose = True)
def get_metrics_multilabel(preds, ys, k, logger, wandb, verbose = True, dataset = ""): #labels
    """ Wrapper for getting multilabel classification metrics. 
    Accuracy, AUROC, AUPRC, precision, recall, recall@50, ap@50, fmax, f1, fbeta
    """

    fmaxs = []
    recall_ks = []
    ap_ks = []
    aurocs = []
    auprcs = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    fbetas = []

    for i in tqdm(range(preds.shape[1])):

        valid_preds = preds[:, i]
        valid_ys = ys[:, i]

        fmax, _ = fmax_score(valid_ys, valid_preds)
        recall_k, _, ap_k, _ = precision_recall_at_k(valid_ys, valid_preds, k)
        auroc_score = roc_auc_score(valid_ys, valid_preds)
        auprc_score = average_precision_score(valid_ys, valid_preds)
        accuracy = accuracy_score(valid_ys, np.round(valid_preds))
        precision = precision_score(valid_ys, np.round(valid_preds))
        recall = recall_score(valid_ys, np.round(valid_preds))

        f1 = 2 * precision * recall / (precision + recall)
        f1_skl = f1_score(valid_ys, np.round(valid_preds))
        fbeta = fbeta_score(valid_ys, np.round(valid_preds), beta=1)
        
        assert np.isclose(f1, f1_skl)

        fmaxs.append(fmax)
        recall_ks.append(recall_k)
        ap_ks.append(ap_k)
        aurocs.append(auroc_score)
        auprcs.append(auprc_score)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        fbetas.append(fbeta)
        # break

    fmax = np.mean(fmaxs)
    recall_k = np.mean(recall_ks)
    ap_k = np.mean(ap_ks)
    auroc_score = np.mean(aurocs)
    auprc_score = np.mean(auprcs)
    accuracy = np.mean(accuracies)
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)
    fbeta = np.mean(fbetas)
    
    if verbose==True:
        logger.info(f"Fmax = {fmax:.4f}, Recall@{k} = {recall_k:.4f},  ap@{k} = {ap_k:.4f}, auroc_score = {auroc_score:.4f}, auprc_score = {auprc_score:.4f}, accuracy = {accuracy:.4f}, precision = {precision:.4f}, recall = {recall:.4f}, F1 = {f1:.4f}, Fbeta = {fbeta:.4f}")

    if wandb is not None and dataset is not None:
        wandb.log({dataset + '_fmax': fmax,
                dataset + '_recall_k': recall_k,
                dataset + '_auroc_score': auroc_score,
                dataset + '_auprc_score': auprc_score,
                dataset + '_accuracy': accuracy,
                dataset + '_precision': precision,
                dataset + '_recall': recall,
                dataset + '_f1': f1
                })


    return fmax, recall_k, ap_k, auroc_score, auprc_score, accuracy, f1, fbeta


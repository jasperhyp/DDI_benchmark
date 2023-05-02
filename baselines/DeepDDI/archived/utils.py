import pandas as pd
import numpy as np
from torch_geometric.utils import is_undirected, to_undirected, to_dense_adj, k_hop_subgraph, coalesce, structured_negative_sampling
import torch


def get_batch_embeddings(batch_ddis, pca_lookup, data_mode = "train"):

    """Takes in the batch of ddis :torch.tensor: with shape of (num_edges, 2) and lookup and outputs the 
    reformated data + appropriate negative sampling. For example, by inputting edge (A,B), it first adds (B,A)
    as an additional positive edge, then adds (A, 1), (1,A), (B,2), (2,B) where 1 and 2 are negative edges
    sampled within the batch. This will have to be adjusted for evaluation mode. 

    Args:
        batch_ddis (TorchTensor): The edge indices.
        pca_lookup (pandas dataframe): pandas dataframe where the indices are set by the drug. 

    :rtype: (Tensor, Tensor, Tensor, Tensor)
    """

    if data_mode == "train":

        undir_pos_ddis = to_undirected(batch_ddis.T)  ## make undirected first, (A,B), (B,A)
        heads, _, negatives = structured_negative_sampling(undir_pos_ddis) # because we set this to be undirected, we only need heads + negatives
        neg_samps = torch.cat([heads.view(1, -1), negatives.view(1, -1)], dim = 0)
        undir_neg_ddis = to_undirected(neg_samps) # set the negative samples to be undirected, (A,1), (1,A), (B,2), (2, B)

        undir_pos_ddis = np.array(undir_pos_ddis) 
        undir_neg_ddis = np.array(undir_neg_ddis)
        lookup = pca_lookup.T

        head_pos = lookup[undir_pos_ddis[0]].values.T 
        tail_pos = lookup[undir_pos_ddis[1]].values.T
        batch_pos = np.concatenate([head_pos, tail_pos], axis = 1)
        batch_pos = torch.tensor(batch_pos)
        pos_labels = torch.ones(batch_pos.shape[0]) 


        head_neg = lookup[undir_neg_ddis[0]].values.T
        tail_neg = lookup[undir_neg_ddis[1]].values.T
        batch_neg = np.concatenate([head_neg, tail_neg], axis = 1)
        batch_neg = torch.tensor(batch_neg)
        neg_labels = torch.zeros(batch_neg.shape[0])

        batch_ddis = torch.cat([batch_pos, batch_neg], dim = 0)
        labels = torch.cat([pos_labels, neg_labels], dim = 0)

        return (batch_ddis, labels)

    elif data_mode == "val" or data_mode == "test":
        raise ValueError('Val or test negative sampling on the fly not implemented yet')

    else:
        raise ValueError('Please input, val, test, or train')

# x, labels = get_batch_embeddings(batch_ddis, pca_lookup, data_mode = "test")






#######
# Test
#######

from typing import Optional
import numpy as np
import torch
from torch_geometric.utils import is_undirected


def structured_negative_sampling_binary(
    edge_index: torch.Tensor, 
    valid_indices: torch.Tensor,
    ground_truth_edge_index: Optional[torch.LongTensor] = None, 
    contains_neg_self_loops: bool = False,
    probs: Optional[torch.Tensor] = None,
):
    r"""Adapted from Pytorch Geometric's `structured_negative_sampling`. Samples a negative edge :obj:`(i,k)` for every positive edge :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a tuple of the form :obj:`(i,j,k)`. This is used for binary DDI case only.

    Args:
        edge_index (LongTensor): The edge indices.
        valid_indices (LongTensor): The indices of nodes that are valid for negative sampling (for tails).
        ground_truth_edge_index (LongTensor): Ground truth edge indices (used for excluding false negatives), usually contains `edge_index`. 
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)
        probs (torch.Tensor, optional): The probability distribution of nodes for negative sampling.

    :rtype: (LongTensor, LongTensor, LongTensor)

    """
    assert edge_index.numel() > 0
    assert is_undirected(edge_index)
    assert is_undirected(ground_truth_edge_index) if ground_truth_edge_index is not None else True
    
    # get number of nodes and valid indices for sampling
    num_nodes = int(ground_truth_edge_index.max()) + 1
    valid_indices = valid_indices.numpy() if valid_indices is not None else torch.unique(edge_index, sorted=True).numpy()
    
    row, col = edge_index.cpu()
    pos_idx = row * num_nodes + col
    
    if not contains_neg_self_loops:
        loop_idx = torch.arange(num_nodes) * (num_nodes + 1)
        pos_idx = torch.cat([pos_idx, loop_idx], dim=0)
    
    # add ground truth edges to pos_idx
    if ground_truth_edge_index is not None:
        valid_ground_truth_edge_index = ground_truth_edge_index[
            :, 
            (torch.isin(ground_truth_edge_index[0], edge_index[0])) | (torch.isin(ground_truth_edge_index[1], valid_indices))
        ]  # record those ground truth edges whose heads are also in edge_index's heads and whose tails are in valid_indices for exclusion during sampling later
        ground_truth_row, ground_truth_col = valid_ground_truth_edge_index.cpu()
        ground_truth_idx = ground_truth_row * num_nodes + ground_truth_col
        pos_idx = torch.cat([pos_idx, ground_truth_idx], dim=0)
    
    rand = torch.from_numpy(np.random.choice(valid_indices, (row.size(0), ), replace=True, p=probs)).long()
    # rand = torch.randint(num_nodes, (row.size(0), ), dtype=torch.long)
    neg_idx = row * num_nodes + rand
    
    mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.from_numpy(np.random.choice(valid_indices, (rest.size(0), ), replace=True, p=probs)).long()
        # tmp = torch.randint(num_nodes, (rest.size(0), ), dtype=torch.long)
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp

        mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
        rest = rest[mask]

    return edge_index[0], edge_index[1], rand.to(edge_index.device)



import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy.sparse as sp
import torch
import random
import os
import pandas as pd
import torch.nn.functional as F
import math
import copy
from typing import Optional
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering
import community as community_louvain
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from gae_denoiser import train_and_denoise_gae
from torch_geometric.nn import GCNConv, GAE
import torch.nn as nn
from torch_geometric.datasets import Flickr, Coauthor, Actor, WikipediaNetwork,CitationFull, Planetoid, Amazon, Reddit,Twitch, GitHub,WikiCS,FacebookPagePage,LastFMAsia
from citationDataset import CitationDataset
from karateClub import KarateClub
from torch.utils.data import TensorDataset, DataLoader
import math


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def filter_class_by_count(data, remove_unlabeled=False):
    min_count = 5000
    assert hasattr(data, 'y'), "The data object must have a 'y' attribute."

    y = F.one_hot(data.y)
    counts = y.sum(dim=0)  # Count occurrences of each class

    y = y[:, counts >= min_count]
    mask = y.sum(dim=1).bool()  # Nodes to keep based on the filtered classes

    data = data.clone()
    data.y = y.argmax(dim=1)

    if remove_unlabeled:
        data = data.subgraph(mask)
        # print("Filtered data node count:", data.num_nodes)
    else:
        data.y[~mask] = -1

        if hasattr(data, 'train_mask'):
            data.train_mask = data.train_mask & mask
            data.val_mask = data.val_mask & mask
            data.test_mask = data.test_mask & mask

    return data

def remove_isolated_nodes(data):
    "Remove isolated nodes."
    mask = data.y.new_zeros(data.num_nodes, dtype=bool)
    mask[data.edge_index[0]] = True
    mask[data.edge_index[1]] = True
    data = data.subgraph(mask)
    return data




def laplace_noise(tensor1, sensitivity, epsilon):
    noise = np.random.laplace(0, sensitivity / epsilon, tensor1.shape)
    return  noise


def matrix_randomized_response(x, eps):
    ss = torch.ones_like(x)
    x_copy = copy.deepcopy(x).to(torch.bool)
    em = math.exp(eps)
    p = ss * em / (em + 1)
    print(1 / (em + 1))
    t = torch.bernoulli(p).to(torch.bool)
    perturbed_matrix = (~(x_copy ^ t)).to(torch.float)
    return perturbed_matrix


def rr_adj(adj_tensor: torch.Tensor, eps_edge: float) -> torch.Tensor:

    n = adj_tensor.size(0)

    p = 1.0 / (1.0 + math.exp(eps_edge))

    noise = torch.bernoulli(torch.full((n, n), p))

    res = ((adj_tensor + noise) % 2).float()


    return res


def arr_adj_dense(adj_tensor: torch.Tensor, eps_edge: float, t: float) -> torch.Tensor:

    n = adj_tensor.size(0)

    t_max = 1.0 / (1.0 + math.exp(-eps_edge))
    if t > t_max:

        t = t_max
    if t <= 0:
        raise ValueError("!!!")

    a = t  # Pr(1 | e=1)
    b = t * math.exp(-eps_edge)  # Pr(1 | e=0)

    prob_matrix = b + (a - b) * adj_tensor

    res = torch.bernoulli(prob_matrix).float()

    res.fill_diagonal_(0)

    return res



def rr_adj_masked(adj_tensor: torch.Tensor, mask: torch.Tensor, eps_edge: float) -> torch.Tensor:

    n = adj_tensor.size(0)
    device = adj_tensor.device
    A_triu = torch.triu(adj_tensor, diagonal=1)
    M_triu = torch.triu(mask, diagonal=1)


    p = 1.0 / (1.0 + math.exp(eps_edge))

    perturbed_triu = torch.zeros_like(A_triu)


    indices = torch.nonzero(M_triu == 1, as_tuple=True)

    original_values = A_triu[indices]

    noise = torch.bernoulli(torch.full_like(original_values, p))

    perturbed_values = ((original_values + noise) % 2).float()

    perturbed_triu[indices] = perturbed_values

    perturbed_adj = perturbed_triu + perturbed_triu.T

    return perturbed_adj



def ldpsafe_sampling_with_filter(A: torch.Tensor, eps: float, tau: float = 0.1) -> torch.Tensor:

    assert A.dim() == 2 and A.shape[0] == A.shape[1],
    n = A.size(0)
    device = A.device

    A_triu = torch.triu(A, diagonal=1).to(torch.float32)


    exp_eps = math.exp(eps)


    s1 = 0.7
    s0 = s1 * math.exp(-eps)


    rand_matrix = torch.rand((n, n), device=device)


    mask1 = (A_triu == 1)
    mask0 = (A_triu == 0)

    sampling_mask = torch.zeros_like(A_triu, device=device)

    sampling_mask[mask1] = (rand_matrix[mask1] < s1).float()
    sampling_mask[mask0] = (rand_matrix[mask0] < s0).float()
    # print(sampling_mask[0])
    print("sampling_mask Adjacency (sum):", sampling_mask.sum())
    nonzero_count = (sampling_mask[0] != 0).sum().item()


    filter_mask = (rand_matrix >= tau).float()
    final_mask = sampling_mask * filter_mask

    sampled_triu = torch.triu(final_mask, diagonal=1)

    print("sampled_triu Adjacency (sum):", sampled_triu.sum())
    nonzero_count = (sampled_triu[0] != 0).sum().item()


    return sampled_triu



def dense_to_sparse_with_weights(priv_adj: torch.Tensor):

    sparse = priv_adj.to_sparse()

    edge_index = sparse.indices()  # shape: [2, num_edges]
    edge_weight = sparse.values()  # shape: [num_edges]

    return edge_index, edge_weight


def _proj_symmetric_unit_interval(A: torch.Tensor):

    upper = torch.triu(A, diagonal=1)
    A = upper + upper.T

    return A.clamp_(0.0, 1.0)



def debias_rr(A_tilde: torch.Tensor, eps_edge: float) -> torch.Tensor:

    p = 1.0 / (1.0 + math.exp(eps_edge))
    denom = (1.0 - 2.0 * p)
    if abs(denom) < 1e-6:
        return A_tilde.clamp(0.0, 1.0)
    A_hat = (A_tilde - p) / denom
    return A_hat.clamp(0.0, 1.0)



@torch.no_grad()
def retain_topm_global(W: torch.Tensor, A_count_from: torch.Tensor):

    W = W.clone()
    n = W.size(0)
    triu = torch.triu(torch.ones_like(W, dtype=torch.bool), diagonal=1)
    m = int(torch.round(A_count_from[triu].sum()).item())
    if m <= 0:
        out = torch.zeros_like(W)
        out.fill_diagonal_(0.0)
        return out

    scores = W[triu]
    vals, idx = torch.topk(scores, k=min(m, scores.numel()))
    mask = torch.zeros_like(scores)
    mask[idx] = 1.0

    out = torch.zeros_like(W)
    out[triu] = scores * mask
    out = out + out.t()
    out.fill_diagonal_(0.0)
    return out


@torch.no_grad()
def retain_topk_by_degree(W: torch.Tensor, k_vec: torch.Tensor, sym: str = 'union'):

    assert W.dim() == 2 and W.size(0) == W.size(1)
    n = W.size(0)

    if k_vec.dim() == 2:
        k_vec = k_vec.squeeze(1)
    k_vec = torch.round(k_vec).clamp(0, n-1).to(torch.long)


    scores = W.clone()
    # scores.fill_diagonal_(float('-inf'))
    scores[scores <= 0] = float('-inf')


    out = torch.zeros_like(W)

    for i in range(n):
        k = int(k_vec[i].item())
        if k <= 0:
            continue
        valid = torch.isfinite(scores[i]).sum().item()
        if valid <= 0:
            continue
        k_eff = min(k, int(valid))
        vals, idx = torch.topk(scores[i], k=k_eff, largest=True, sorted=False)
        out[i, idx] = W[i, idx]



    if sym == 'union':
        out = torch.max(out, out.t())
    elif sym == 'intersection':
        out = torch.min(out, out.t())
    else:
        out = out




    return out


def is_symmetric(matrix, tol=1e-8):

    return np.allclose(matrix, matrix.T, atol=tol)



def build_community_index_matrix(community_labels: torch.Tensor) -> torch.Tensor:

    community_labels = community_labels.long()
    device = community_labels.device
    N = community_labels.numel()


    same_comm_mask = (community_labels.view(-1, 1) == community_labels.view(1, -1))  # bool


    col_indices = torch.arange(N, device=device).view(1, -1).expand(N, -1)  # [N, N]
    comm_index_mat = col_indices * same_comm_mask.long()

    return comm_index_mat




def load_dataset(path="../node_level",dataset_name="",
              eps_edge=7,
              delta_eps=0.1,
              use_topk: bool = True,
              topk_sym: str = 'union',
              n_communities=7,
              alpha_rate=0,
              sigma = 0
              ):

    print(f'Loading {dataset_name} dataset...')

    data_file_root = f"{path}/{dataset_name}/"

    from torch_geometric.transforms import NormalizeFeatures
    if dataset_name in ["cora"]:
        dataset = Planetoid(root=data_file_root, name=f"{dataset_name}")   #transform=NormalizeFeatures()
        dataset = dataset[0]

    elif dataset_name in ["citeseer"]:
        dataset = Planetoid(root=data_file_root, name=f"{dataset_name}")   #transform=NormalizeFeatures()
        dataset = dataset[0]

    elif dataset_name in ["lastfm", "facebook","twitch", "github","deezer","wikipedia"]:
        dataset = KarateClub(root=data_file_root, name=f"{dataset_name}")
        dataset = dataset[0]

    elif dataset_name in ["DBLP"]:
        dataset = CitationFull(root=data_file_root,name=f"{dataset_name}")
        dataset = dataset[0]

    elif dataset_name in ["lastfm-128"]:
        dataset = LastFMAsia(root=data_file_root)
        dataset = dataset[0]

    elif dataset_name in ["facebook-128"]:
        dataset = FacebookPagePage(root=data_file_root)
        dataset = dataset[0]

    elif dataset_name in ["CS"]:
        dataset = Coauthor(root=data_file_root, name=f"{dataset_name}")
        dataset = dataset[0]

    elif dataset_name in ["Physics"]:
        dataset = Coauthor(root=data_file_root, name=f"{dataset_name}")
        dataset = dataset[0]


    print(f'Datset: {dataset}:')

    num_nodes = dataset.num_nodes
    num_edges = dataset.edge_index.size(1)
    feature_dim = dataset.num_node_features
    label_dim = int(dataset.y.max().item() + 1)

    print(f"=== Dataset: {dataset_name} ===")
    print(f"Number of nodes     : {num_nodes}")
    print(f"Number of edges     : {num_edges}")
    print(f"Feature dimension   : {feature_dim}")
    print(f"Number of classes   : {label_dim}")

    labels = dataset.y
    classes = int(labels.max() + 1)
    # labels = torch.LongTensor(labels)
    labels = labels.to(torch.long)
    labels_truth = labels
    labels = F.one_hot(labels, num_classes=classes)
    features = dataset.x
    features = torch.FloatTensor(features)
    edge_index = dataset.edge_index
    edges = edge_index.t().tolist()
    edges = torch.tensor(edges)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


    # random split
    train_rate = 0.5
    val_rate = 0.75
    num = features.shape[0]
    numbers = list(range(num))
    random.shuffle(numbers)
    idx_train = torch.LongTensor(numbers[:int(num * train_rate)])
    idx_val = torch.LongTensor(numbers[int(num * train_rate):int(num * val_rate)])
    idx_test = torch.LongTensor(numbers[int(num * val_rate):num])


    adj_dense = adj.todense()
    adj_tensor = torch.from_numpy(adj_dense).float()  # 包含自环



    n = adj_tensor.shape[0]
    deg = adj.sum(1).reshape(n, 1)
    deg = torch.tensor(deg, dtype=torch.float32)  # (n,1)


    eps_e = eps_edge

    alpha = delta_eps
    eps_d = alpha * eps_e
    eps_a = eps_e - eps_d


    print("dataset_name:", dataset_name,"eps_d:", eps_d, "eps_a:", eps_a, "communities:", n_communities)


    noise = torch.distributions.Laplace(loc=0.0, scale=1.0 / eps_d).sample((n, 1))
    priv_deg = deg + noise
    priv_deg = torch.clamp(priv_deg, min=1.0, max=float(n - 2))


    noise = torch.distributions.Laplace(loc=0.0, scale=1.0 / eps_a).sample(adj_tensor.shape)
    perturbed_adj = adj_tensor + noise
    perturbed_adj = torch.clamp(perturbed_adj, min=0)

    perturbed_adj = torch.triu(perturbed_adj, diagonal=1)






    if dataset_name in ["Physics", "CS"]:
        from sklearn.random_projection import GaussianRandomProjection
        rp = GaussianRandomProjection(n_components=128, random_state=42)
        features_proj = rp.fit_transform(features.detach().cpu().numpy())
        features = torch.from_numpy(features_proj).float()

    if dataset_name in ["facebook"]:
        from sklearn.random_projection import GaussianRandomProjection
        rp = GaussianRandomProjection(n_components=512, random_state=42)
        features_proj = rp.fit_transform(features.detach().cpu().numpy())
        features = torch.from_numpy(features_proj).float()


    if use_topk:


        k = int(priv_deg.sum().item())
        num_nodes = perturbed_adj.shape[0]
        k = max(0, min(k, num_nodes * num_nodes))
        flat_adj = perturbed_adj.view(-1)
        _, indices = torch.topk(flat_adj, k)

        perturbed_adj = torch.zeros_like(perturbed_adj)
        perturbed_adj.view(-1)[indices] = 1.0

    perturbed_adj = _proj_symmetric_unit_interval(perturbed_adj)




    edge_index, edge_weight = dense_to_sparse_with_weights(perturbed_adj)

    '=============================================================='

    # 返回
    return edge_index, edge_weight, features, labels_truth, idx_train, idx_val, idx_test









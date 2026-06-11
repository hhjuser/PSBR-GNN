
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
from sklearn.cluster import MiniBatchKMeans as KMeans


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
    eps_a = eps_e

    print("dataset_name:", dataset_name, "eps_a:", eps_a, "communities:", n_communities)

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




    xi1, xi2, tilde1, m_hat = ldpgen_two_phase_stats(adj_tensor, num_nodes=n, eps=eps_a, k0=2, eps_split=0.5)

    edge_index = ldpgen_phase3_reconstruct_edges_fast(
        delta_1=tilde1, xi1=xi1, xi2=xi2, m_hat=m_hat, sym="union", seed=42, return_sparse_adj=False
    )

    edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32)



    '=============================================================='

    # 返回
    return edge_index, edge_weight, features, labels_truth, idx_train, idx_val, idx_test




import math
import numpy as np
import torch
# from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans as KMeans



@torch.no_grad()
def ldpgen_two_phase_stats(
    adj: torch.Tensor,
    num_nodes: int,
    eps: float,
    k0: int = 2,
    k1: int = None,
    eps_split: float = 0.5,   # eps1=eps*(1-eps_split), eps2=eps*eps_split
    seed: int = 42,
    device: torch.device = None,
):

    if device is None:
        device = adj.device if isinstance(adj, torch.Tensor) else torch.device("cpu")
    adj = adj.to(device)

    N = int(num_nodes)

    eps2 = float(eps) * float(eps_split)
    eps1 = float(eps) - eps2
    eps1 = max(eps1, 1e-6)
    eps2 = max(eps2, 1e-6)


    if adj.is_sparse:
        adj = adj.coalesce()
        src, dst = adj.indices()
    else:

        src, dst = torch.nonzero(adj, as_tuple=True)


    mask = (src != dst)
    src, dst = src[mask], dst[mask]

    tri = (src < dst)
    src, dst = src[tri], dst[tri]


    src2 = torch.cat([src, dst], dim=0)
    dst2 = torch.cat([dst, src], dim=0)


    g = torch.Generator(device=device)
    g.manual_seed(seed)
    xi0 = torch.randint(low=0, high=k0, size=(N,), generator=g, device=device)

    # delta0[u, group(v)]
    flat0 = src2 * k0 + xi0[dst2]


    delta0 = torch.bincount(flat0, minlength=N * k0).view(N, k0).float()

    # Laplace noise (scale=1/eps1)
    lap1 = torch.distributions.laplace.Laplace(loc=0.0, scale=1.0 / eps1)
    # lap1 = torch.distributions.laplace.Laplace(loc=0.0, scale=2.0 / eps1)
    tilde0 = (delta0 + lap1.sample(delta0.shape).to(device)).clamp_min(0.0)

    eta_hat = tilde0.sum(dim=1)  # [N]
    m_hat = int(torch.round(eta_hat.sum() / 2.0).item())
    m_hat = max(0, min(m_hat, N * (N - 1) // 2))


    if k1 is None:
        eta = torch.clamp(torch.round(tilde0.sum(dim=1)), 1, N - 1).long()
        unique_eta, counts = torch.unique(eta, return_counts=True)

        d_u = unique_eta.float() / 2.0

        k1_d_u = d_u + (d_u * d_u - 2.0 * (1.0 + math.sqrt(5.0)) * d_u + 1.0) / eps2
        k1_est = torch.ceil((k1_d_u * (counts.float() / N)).sum()).long().item()

        k1 = int(max(2, min(k1_est, N)))


    X0 = tilde0.detach().cpu().numpy()
    km1 = KMeans(n_clusters=k1, random_state=seed, n_init=10)
    xi1 = torch.from_numpy(km1.fit_predict(X0)).to(device).long()  # [N]


    flat1 = src2 * k1 + xi1[dst2]
    delta1 = torch.bincount(flat1, minlength=N * k1).view(N, k1).float()

    lap2 = torch.distributions.laplace.Laplace(loc=0.0, scale=1.0 / eps2)
    # lap2 = torch.distributions.laplace.Laplace(loc=0.0, scale=2.0 / eps2)
    tilde1 = (delta1 + lap2.sample(delta1.shape).to(device)).clamp_min(0.0)


    X1 = tilde1.detach().cpu().numpy()
    km2 = KMeans(n_clusters=k1, random_state=seed + 1, n_init=10)
    xi2 = torch.from_numpy(km2.fit_predict(X1)).to(device).long()  # [N]

    return xi1, xi2, tilde1, m_hat


@torch.no_grad()
def ldpgen_phase3_reconstruct_edges_fast(
    delta_1: torch.Tensor,   # [n, k1]  Phase II noisy degree vectors
    xi1: torch.Tensor,       # [n]
    xi2: torch.Tensor,       # [n]
    m_hat: int,
    sym: str = "union",
    seed: int = 42,
    return_sparse_adj: bool = False,
):

    device = delta_1.device
    n, k1 = delta_1.shape
    n = int(n)
    k1 = int(k1)

    xi1 = xi1.view(-1).long().to(device)
    xi2 = xi2.view(-1).long().to(device)


    d1 = torch.clamp(torch.round(delta_1), 0, n - 1).to(torch.float32)


    pair = xi1 * k1 + xi2
    overlap = torch.bincount(pair, minlength=k1 * k1).view(k1, k1).float()

    size1 = torch.bincount(xi1, minlength=k1).float().clamp_min(1.0)  # [k1]
    M = overlap / size1.view(k1, 1)  # [k1,k1]
    delta_hat = (d1 @ M).clamp_min(0.0)  # [n,k1]

    sum_in = torch.zeros((k1, k1), device=device, dtype=torch.float32)
    sum_in.index_add_(0, xi2, delta_hat)

    size2 = torch.bincount(xi2, minlength=k1).float().clamp_min(1.0)  # [k1]
    sum_col = delta_hat.sum(dim=0)  # [k1]

    # 按你原式：pij_k[i][j] = (sum_col[j]/|U2_j|) / (sum_col[j] + sum_in[i,j])
    num = (sum_col / size2).view(1, k1)                 # [1,k1]
    den = (sum_col.view(1, k1) + sum_in).clamp_min(1e-8) # [k1,k1]
    pij_k = (num / den).clamp(0.0, 1.0)                 # [k1,k1]


    m_hat = int(max(0, min(m_hat, n * (n - 1) // 2)))
    if m_hat == 0:
        if return_sparse_adj:
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.float32, device=device),
                (n, n),
            ).coalesce()
        return torch.empty((2, 0), dtype=torch.long, device=device)

    g = torch.Generator(device=device)
    g.manual_seed(seed)


    nodes_in = [torch.where(xi2 == c)[0] for c in range(k1)]
    sizes = torch.tensor([int(v.numel()) for v in nodes_in], device=device).clamp_min(1)  # [k1]


    cap = sizes.view(k1, 1) * sizes.view(1, k1)  # [k1,k1]
    diag_idx = torch.arange(k1, device=device)
    cap[diag_idx, diag_idx] = sizes * (sizes - 1)


    w = (pij_k * cap.float()).clamp_min(0.0)
    s = w.sum().clamp_min(1e-12)
    w = w / s

    idx = torch.multinomial(w.view(-1), num_samples=m_hat, replacement=True, generator=g)
    cnt = torch.bincount(idx, minlength=k1 * k1).view(k1, k1)

    edges = []
    for a in range(k1):
        Ua = nodes_in[a]
        if Ua.numel() == 0:
            continue
        for b in range(k1):
            c = int(cnt[a, b].item())
            if c <= 0:
                continue
            Ub = nodes_in[b]
            if Ub.numel() == 0:
                continue

            u = Ua[torch.randint(0, Ua.numel(), (c,), generator=g, device=device)]
            v = Ub[torch.randint(0, Ub.numel(), (c,), generator=g, device=device)]


            mask = (u != v)
            u, v = u[mask], v[mask]
            if u.numel() == 0:
                continue


            uu = torch.minimum(u, v)
            vv = torch.maximum(u, v)
            keep = (uu != vv)
            uu, vv = uu[keep], vv[keep]
            if uu.numel() == 0:
                continue

            edges.append(torch.stack([uu, vv], dim=0))

    if len(edges) == 0:
        if return_sparse_adj:
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.float32, device=device),
                (n, n),
            ).coalesce()
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index = torch.cat(edges, dim=1)  # [2, E_try]


    h = edge_index[0] * n + edge_index[1]
    h = torch.unique(h)
    edge_u = h // n
    edge_v = h % n
    edge_index = torch.stack([edge_u, edge_v], dim=0)


    if sym == "union":

        rev = edge_index.flip(0)
        edge_index_sym = torch.cat([edge_index, rev], dim=1)
    elif sym == "intersection":

        rev = edge_index.flip(0)
        edge_index_sym = torch.cat([edge_index, rev], dim=1)
    elif sym == "none":
        edge_index_sym = edge_index
    else:
        raise ValueError("sym must be 'union'|'intersection'|'none'")

    if not return_sparse_adj:
        return edge_index_sym


    values = torch.ones(edge_index_sym.size(1), device=device, dtype=torch.float32)
    adj_samp = torch.sparse_coo_tensor(edge_index_sym, values, (n, n)).coalesce()

    ii, jj = adj_samp.indices()
    vv = adj_samp.values()
    mask = (ii != jj)
    adj_samp = torch.sparse_coo_tensor(
        torch.stack([ii[mask], jj[mask]], dim=0),
        vv[mask],
        (n, n),
    ).coalesce()

    return adj_samp





























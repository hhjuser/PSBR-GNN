
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
from sklearn.cluster import KMeans


# ========== 你已有 ==========
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




def matrix_generalized_randomized_response(x, eps, d=None):
    """
    GRR mechanism for categorical matrix/vector x.

    x: tensor, each entry should be an integer in {0, 1, ..., d-1}
    eps: privacy budget
    d: domain size. If None, use max(x)+1.
    """

    x_long = x.to(torch.long)

    if d is None:
        d = int(x_long.max().item()) + 1

    if d <= 1:
        raise ValueError("d must be larger than 1 for GRR.")

    em = math.exp(eps)

    # GRR probabilities
    p = em / (em + d - 1)      # keep true value
    q = 1.0 / (em + d - 1)    # output each wrong value

    print("p =", p, "q =", q)

    # decide whether to keep the original value
    keep = torch.bernoulli(
        torch.full_like(x_long, p, dtype=torch.float)
    ).to(torch.bool)

    # if not keep, randomly choose one value from the other d-1 values
    # offset in {1, 2, ..., d-1}
    random_offset = torch.randint(
        low=1,
        high=d,
        size=x_long.shape,
        device=x_long.device
    )

    # this guarantees the new value is different from original x
    random_wrong_value = (x_long + random_offset) % d

    perturbed_matrix = torch.where(keep, x_long, random_wrong_value)

    return perturbed_matrix.to(x.dtype)


def rr_adj(adj_tensor: torch.Tensor, eps_edge: float) -> torch.Tensor:

    n = adj_tensor.size(0)
    p = 1.0 / (1.0 + math.exp(eps_edge))
    noise = torch.bernoulli(torch.full((n, n), p))
    res = ((adj_tensor + noise) % 2).float()


    return res



def dense_to_sparse_with_weights(priv_adj: torch.Tensor):

    sparse = priv_adj.to_sparse()

    edge_index = sparse.indices()  # shape: [2, num_edges]
    edge_weight = sparse.values()  # shape: [num_edges]

    return edge_index, edge_weight


def _proj_symmetric_unit_interval(A: torch.Tensor):

    upper = torch.triu(A, diagonal=1)
    A = upper + upper.T

    return A.clamp_(0.0, 1.0)





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


    random.seed(42)

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
    adj_tensor = torch.from_numpy(adj_dense).float()




    n = adj_tensor.shape[0]
    deg = adj.sum(1).reshape(n, 1)
    deg = torch.tensor(deg, dtype=torch.float32)  # (n,1)


    eps_e = eps_edge




    alpha = delta_eps
    eps_d = alpha * eps_e
    eps_a = eps_e - eps_d


    print("dataset_name:", dataset_name,"eps_d:", eps_d, "eps_a:", eps_a, "communities:", n_communities)

    noise = torch.distributions.Laplace(loc=0.0, scale=1.0 / (eps_d)).sample((n, 1))
    priv_deg = deg + noise
    priv_deg = torch.clamp(priv_deg, min=1.0, max=float(n - 2))






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



    X = F.normalize(features, p=2, dim=1)
    S = torch.mm(X, X.t())
    S = torch.clamp(S, min=0.)





    N = adj_tensor.shape[0]
    n_communities = int(round(n_communities * N))
    print("group：",n_communities)




    prior_spectral,community_one_hot = calculate_chung_lu_prior(
        adj_tensor=adj_tensor,
        priv_deg=priv_deg,
        features=features,
        method='random',
        prior_method='chung-du',
        eps_a=eps_a,
        eps_d=eps_d,
        n_communities=n_communities,
        labels_truth=labels_truth,
        idx_train=idx_train,
        idx_val=idx_val,
        sigma = sigma
    )

    prior_spectral.fill_diagonal_(0)

    W = prior_spectral * (S ** alpha_rate)

    W = 0.5 * (W + W.T)


    if use_topk:
        W = retain_topk_by_degree(W, k_vec=priv_deg, sym=topk_sym)  # ✅
        # W = retain_topm_global(W, perturbed_adj)



    edge_index, edge_weight = dense_to_sparse_with_weights(W)


    features = F.normalize(features, p=2, dim=1)

    # 返回
    return edge_index, edge_weight, features, labels_truth, idx_train, idx_val, idx_test





def compute_P0_A_bound(adj_tensor, community_one_hot, eps_g, eps=1e-12):
    """
    Compute:

        1/n E||P_tilde^0 - A||_F
        <= 1/n [ sqrt(nK / (1 + exp(eps_g)))
                 + sqrt(sum_i (d_i - m_i)^2 / m_i) ]

    Inputs:
        adj_tensor: [N, N] adjacency matrix A
        community_one_hot: [N, K] one-hot group matrix Q
        eps_g: privacy budget epsilon_g

    Returns:
        bound: scalar tensor
        term_rr: sqrt(nK / (1 + exp(eps_g)))
        term_rw: sqrt(sum_i (d_i - m_i)^2 / m_i)
        d: node degree [N]
        m: binary group degree [N]
    """

    A = adj_tensor.float()
    Q = community_one_hot.float()
    device = A.device

    N, K = Q.shape

    # B_count[i, b] = number of neighbors of node i in group b
    B_count = A @ Q          # [N, K]

    # Z[i, b] = 1 if node i has at least one neighbor in group b
    Z = (B_count > 0).float()  # [N, K]

    # d_i = degree of node i
    d = A.sum(dim=1)          # [N]

    # m_i = number of active neighbor-groups of node i
    m = Z.sum(dim=1)          # [N]

    eps_g_tensor = torch.tensor(float(eps_g), device=device)

    # First term: sqrt(nK / (1 + exp(eps_g)))
    term_rr = torch.sqrt(
        torch.tensor(float(N * K), device=device) /
        (1.0 + torch.exp(eps_g_tensor))
    )

    # Second term: sqrt(sum_i (d_i - m_i)^2 / m_i)
    # For isolated nodes or m_i = 0, avoid division by zero.
    term_rw_inside = torch.where(
        m > 0,
        ((d - m) ** 2) / (m + eps),
        torch.zeros_like(m)
    )

    term_rw = torch.sqrt(term_rw_inside.sum())

    # Final normalized bound
    bound = (term_rr + term_rw) / N

    return bound, term_rr, term_rw, d, m


def compute_MAE_P0_A_bound(adj_tensor, community_one_hot, eps_g, eps=1e-12):
    A = adj_tensor.float()
    Q = community_one_hot.float()
    device = A.device

    N, K = Q.shape

    # B_count[i, b] = number of neighbors of node i in group b
    B_count = A @ Q          # [N, K]

    # Z[i, b] = 1 if node i has at least one neighbor in group b
    Z = (B_count > 0).float()  # [N, K]

    # d_i = degree of node i
    d = A.sum(dim=1)          # [N]

    # m_i = number of active neighbor-groups of node i
    m = Z.sum(dim=1)          # [N]

    eps_g_tensor = torch.tensor(float(eps_g), device=device)

    # ==================================================
    # Formula:
    # E[MAE(P_tilde^0, A)]
    # <= 1/n^2 [ nK/(1+e^{eps_g}) + sum_i (d_i - m_i) ]
    # ==================================================

    term_rr = 3 * (( N * K) / (1.0 + torch.exp(eps_g_tensor)))

    # term_bin = torch.sum(d - m)
    # mae_bound = (term_rr + term_bin) / (N ** 2)

    mae_bound = (term_rr ) / (N ** 2)

    return mae_bound


def compute_EQ_PA_fro_bound(adj_tensor, K, remove_self_loops=True):


    # Make sure A is binary 0/1
    A = (adj_tensor.float().clone() > 0).float()

    if remove_self_loops:
        A.fill_diagonal_(0)

    n = A.shape[0]
    device = A.device

    # degree d_i
    d = A.sum(dim=1)  # [n]

    # |E| = sum_i d_i = sum_{i,j} A_ij
    E_count = d.sum()

    # sum_i d_i^2
    sum_d2 = (d ** 2).sum()

    K_tensor = torch.tensor(float(K), device=device)


    "(2.0 * sum_d2 - 2.0 * E_count) "
    bracket_1 = (
        (1.0 * sum_d2 - 1.0 * E_count) / K_tensor
        + (E_count ** 2) / (K_tensor ** 2)
    )

    bracket_2 = (
            (2.0 * sum_d2 - 2.0 * E_count) / K_tensor
            + (E_count ** 2) / (K_tensor ** 2)
    )

    bracket_2 = 2 * bracket_2
    # full inside sqrt:
    # 2 * bracket


    # numerical safety
    inside_1 = torch.clamp(bracket_1, min=0.0)
    inside_2 = torch.clamp(bracket_2, min=0.0)

    # fro_bound = torch.sqrt(inside)
    #
    # avg_fro_bound = fro_bound / n

    # avg_fro_bound = inside

    return inside_1.item(), inside_2.item()

def compute_clean_PA_MAE_bound(adj_tensor, K, remove_self_loops=True):
    # Make sure A is binary 0/1
    A = (adj_tensor.float().clone() > 0).float()

    if remove_self_loops:
        A.fill_diagonal_(0)

    n = A.shape[0]
    device = A.device

    # degree d_i
    d = A.sum(dim=1)  # [n]

    # |E| = sum_i d_i = sum_{i,j} A_ij
    E_count = d.sum()

    K_tensor = torch.tensor(float(K), device=device)


    sum_d2 = torch.sum(d ** 2)

    P_star_A_l1_bound = 2.0 * (
            (2.0 * sum_d2 - 2.0 * E_count) / K_tensor
            + (E_count ** 2) / (K_tensor ** 2)
    )



    # MAE version: divide by n^2
    P_star_A_l1_bound_avg = P_star_A_l1_bound  / (n * n)

    return P_star_A_l1_bound_avg.item()




def check_binary_matrix(adj_tensor, remove_self_loops=True):
    A = adj_tensor.detach().float().clone()

    if remove_self_loops:
        A.fill_diagonal_(0)


    is_binary = torch.all((A == 0) | (A == 1)).item()


    unique_vals = torch.unique(A)


    non_binary_mask = ~((A == 0) | (A == 1))
    non_binary_count = non_binary_mask.sum().item()

    print("is_binary:", is_binary)
    print("num_unique_values:", unique_vals.numel())
    print("unique values first 20:", unique_vals[:20])
    print("non_binary_count:", non_binary_count)

    if non_binary_count > 0:
        bad_vals = A[non_binary_mask]
        print("non-binary min:", bad_vals.min().item())
        print("non-binary max:", bad_vals.max().item())
        print("non-binary first 20:", bad_vals[:20])

    return is_binary
def check_rr_flip_prob(adj_tensor, perturbed_adj, eps_e):
    A = adj_tensor.float().clone()
    P = perturbed_adj.float().clone()

    n = A.shape[0]
    device = A.device

    A.fill_diagonal_(0)
    P.fill_diagonal_(0)

    mask = ~torch.eye(n, dtype=torch.bool, device=device)

    diff = (P != A) & mask

    num_flips = diff.sum().float()
    num_positions = mask.sum().float()

    p_emp = num_flips / num_positions
    p_theory = 1.0 / (1.0 + torch.exp(torch.tensor(float(eps_e), device=device)))

    fro_error = torch.norm(P - A, p="fro") / n

    print(f"eps_e: {eps_e}")
    print(f"num_flips: {num_flips.item():.0f}")
    print(f"num_positions: {num_positions.item():.0f}")
    print(f"empirical p: {p_emp.item():.8f}")
    print(f"theoretical p: {p_theory.item():.8f}")
    print(f"fro/n: {fro_error.item():.6f}")

    eps_eff = torch.log((1.0 - p_emp) / p_emp)
    print(f"effective epsilon from empirical p: {eps_eff.item():.4f}")




def compute_p_tilde_minus_A_fro_bound(adj_tensor, eps_d):

    A = adj_tensor.float()
    device = A.device

    n = A.shape[0]

    # d_i = sum_j A_ij
    d = A.sum(dim=1)  # [n]

    # |E| = sum_i d_i = sum_{i,j} A_ij
    # For symmetric undirected A, this counts each undirected edge twice.
    E_count = d.sum()

    # sum_i d_i^2
    sum_d2 = (d ** 2).sum()

    eps_d_tensor = torch.tensor(float(eps_d), device=device)

    # sqrt( 2 sum_i d_i^2 + 4n / eps_d^2 )
    term_p = torch.sqrt(
        2.0 * sum_d2
        + 4.0 * torch.tensor(float(n), device=device) / (eps_d_tensor ** 2)
    )

    # sqrt(|E|)
    term_A = torch.sqrt(E_count)

    # final normalized bound
    bound = (term_p + term_A) / n

    return bound.item()


def naive_rr_theoretical_rmse_bound(adj_tensor, eps_a):
    """
    Theoretical bound:
        (1/n) E ||A_tilde - A||_F
        <= sqrt((n-1) / (n * (1 + exp(eps_a))))
    """

    A = adj_tensor.float()
    N = A.shape[0]

    eps_a_tensor = torch.tensor(float(eps_a), device=A.device)

    bound = torch.sqrt(
        torch.tensor(float(N - 1), device=A.device)
        /
        (
            torch.tensor(float(N), device=A.device)
            * (1.0 + torch.exp(eps_a_tensor))
        )
    )

    return bound.item()

def compute_naive_rr_mse_bound(adj_tensor, eps):
    A = adj_tensor.float()
    n = A.shape[0]
    device = A.device

    eps_tensor = torch.tensor(float(eps), device=device)

    mse_bound = (n - 1) / (n * (1.0 + torch.exp(eps_tensor)))

    return mse_bound

def build_clean_P(adj_tensor, community_one_hot):

    Q = community_one_hot.float()   # [N, K]
    A = adj_tensor.float()          # [N, N]

    B_count = A @ Q                 # [N, K]

    B = B_count      # [N, K]

    M = B @ Q.T                     # [N, N]

    numerator = M * M.T             # [N, N]
    eps = 1e-12
    block_mass = Q.T @ B            # [K, K]
    # denominator = Q @ block_mass @ Q.T   # [N, N]
    #

    inv_block_mass = torch.zeros_like(block_mass)
    mask = block_mass > eps
    inv_block_mass[mask] = 1.0 / block_mass[mask]

    # inv_denominator[i, j] = 1 / block_mass[group(i), group(j)]
    inv_denominator = Q @ inv_block_mass @ Q.T  # [N, N]

    # P_ij = M_ij * M_ji / block_mass[group(i), group(j)]
    P = M * M.T
    P = P * inv_denominator

    return P


def build_clean_P_sparse(adj_tensor, community_one_hot, chunk_size=512, eps=1e-12):
    """
    Build P as torch sparse COO tensor.
    Only stores nonzero entries.
    """

    Q = community_one_hot.float()
    A = adj_tensor.float()

    device = A.device
    N, K = Q.shape

    B = A @ Q
    block_mass = Q.T @ B
    group_id = torch.argmax(Q, dim=1)

    row_indices = []
    col_indices = []
    values = []

    all_cols = torch.arange(N, device=device)

    for row_start in range(0, N, chunk_size):
        row_end = min(row_start + chunk_size, N)

        idx_i = torch.arange(row_start, row_end, device=device)
        group_i = group_id[idx_i]

        B_i_to_gj = B[idx_i][:, group_id]
        B_j_to_gi = B[:, group_i].T
        denom = block_mass[group_i][:, group_id]

        P_chunk = torch.where(
            denom > eps,
            B_i_to_gj * B_j_to_gi / (denom + eps),
            torch.zeros_like(denom)
        )

        nz = P_chunk > 0

        if nz.any():
            local_rows, cols = nz.nonzero(as_tuple=True)
            rows = idx_i[local_rows]

            row_indices.append(rows)
            col_indices.append(cols)
            values.append(P_chunk[local_rows, cols])

        del B_i_to_gj, B_j_to_gi, denom, P_chunk

    row_indices = torch.cat(row_indices)
    col_indices = torch.cat(col_indices)
    values = torch.cat(values)

    indices = torch.stack([row_indices, col_indices], dim=0)

    P_sparse = torch.sparse_coo_tensor(
        indices,
        values,
        size=(N, N),
        device=device
    ).coalesce()

    return P_sparse


def build_clean_P_Binarization(adj_tensor, community_one_hot):

    Q = community_one_hot.float()   # [N, K]
    A = adj_tensor.float()          # [N, N]

    B_count = A @ Q                 # [N, K]

    B = (B_count > 0).float()       # [N, K]

    M = B @ Q.T                     # [N, N]

    numerator = M * M.T             # [N, N]

    block_mass = Q.T @ B            # [K, K]
    denominator = Q @ block_mass @ Q.T   # [N, N]

    eps = 1e-12
    P = torch.where(
        denominator > 0,
        numerator / (denominator ),
        torch.zeros_like(numerator)
    )

    return P


def build_clean_P_Binarization_Reweight(adj_tensor, community_one_hot):
    Q = community_one_hot.float()   # [N, K]
    A = adj_tensor.float()          # [N, N]

    eps = 1e-12

    # ==================================================
    # Step 1: node-to-group count
    # B_count[i, b] = number of neighbors of node i in group b
    # ==================================================
    B_count = A @ Q                 # [N, K]

    # ==================================================
    # Step 2: binarization
    # Z[i, b] = 1 if node i has at least one neighbor in group b
    # ==================================================
    Z = (B_count > 0).float()       # [N, K]

    # ==================================================
    # Step 3: compute node degree d_i from A
    # d_i = sum_j A_ij
    # ==================================================
    d = A.sum(dim=1)                # [N]

    # ==================================================
    # Step 4: compute binary group-degree
    # s_i = sum_b Z[i, b]
    # number of groups connected by node i
    # ==================================================
    s = Z.sum(dim=1)                # [N]

    # ==================================================
    # Step 5: reweighting
    # W[i, b] = Z[i, b] * (d_i / s_i)^sigma
    # when sigma = 1:
    # W[i, b] = Z[i, b] * d_i / s_i
    # ==================================================
    sigma = 1.0
    scale = torch.pow(d / (s + eps), sigma)   # [N]
    W = Z * scale.unsqueeze(1)                # [N, K]

    # ==================================================
    # Step 6: construct P using reweighted B, now W
    # M[i, j] = W[i, group(j)]
    # ==================================================
    M = W @ Q.T                    # [N, N]

    numerator = M * M.T            # [N, N]

    block_mass = Q.T @ W           # [K, K]
    denominator = Q @ block_mass @ Q.T   # [N, N]

    P = torch.where(
        denominator > 0,
        numerator / (denominator + eps),
        torch.zeros_like(numerator)
    )

    return P

def compute_gamma0_and_bound(adj_tensor, community_one_hot, eps_g, eps=1e-12):


    Q = community_one_hot.float()   # [N, K]
    A = adj_tensor.float()          # [N, N]

    N, K = Q.shape

    B = A @ Q                       # [N, K]

    Z = (B > 0).float()             # [N, K]

    V0 = (Q.T @ Z).T                # [K, K]


    V0_ab = V0                      # V_{a,b}^{(0)}
    V0_ba = V0.T                    # V_{b,a}^{(0)}

    mask = V0_ab > 0

    if mask.sum() == 0:
        gamma0 = torch.tensor(0.0, device=A.device)
    else:
        ratio = torch.sqrt(V0_ba + eps) / (V0_ab + eps)
        gamma0 = ratio[mask].max()


    eps_g_tensor = torch.tensor(float(eps_g), device=A.device)

    bound = torch.sqrt(2.0 + 8.0 * gamma0 ** 2) * torch.sqrt(
        torch.tensor(float(N * K), device=A.device)
        / (1.0 + torch.exp(eps_g_tensor))
    )

    bound = bound / N

    return bound.item()


def fro_bound_random_grouping_tighter(adj_tensor, K, average=True):
    A = adj_tensor.float().clone()
    A.fill_diagonal_(0)

    n = A.shape[0]
    deg = A.sum(dim=1)
    m = torch.triu(A, diagonal=1).sum()

    edge_index = torch.triu(A, diagonal=1).nonzero(as_tuple=False)

    term_local = torch.tensor(0.0, device=A.device)
    term_cross = torch.tensor(0.0, device=A.device)

    for e in range(edge_index.shape[0]):
        i = edge_index[e, 0]
        j = edge_index[e, 1]

        di = deg[i]
        dj = deg[j]

        # ordered contribution: i -> j and j -> i
        term_local += di * (dj - 1.0)
        term_local += dj * (di - 1.0)

        disjoint_edges = m - di - dj + 1.0

        term_cross += di * disjoint_edges
        term_cross += dj * disjoint_edges

    bound_sq = (
        4.0 * term_local / K
        +
        8.0 * term_cross / (K ** 2)
    )

    bound = torch.sqrt(torch.clamp(bound_sq, min=0.0))

    if average:
        bound = bound / n

    return float(bound)

def full_l1_bound_random_grouping(adj_tensor, K):
    A = adj_tensor.float().clone()
    A.fill_diagonal_(0)

    n = A.shape[0]

    deg = A.sum(dim=1)
    m = int(torch.triu(A, diagonal=1).sum().item())

    sum_d2 = torch.sum(deg * deg)
    sum_d2_dminus1 = torch.sum(deg * deg * (deg - 1.0))

    bound_squared = (
            4.0 * sum_d2_dminus1 / K
            +
            8.0 * m * sum_d2 / (K ** 2)
    )


    # avoid tiny negative numerical issue
    bound = torch.sqrt(torch.clamp(bound_squared, min=0.0))


    bound = bound / n

    return float(bound)

def compute_RMSE_error(adj_tensor, P_star, remove_self_loops=True):
    A = adj_tensor.float().clone()
    P = P_star.float().clone()

    if remove_self_loops:
        A.fill_diagonal_(0)
        P.fill_diagonal_(0)

    n = A.shape[0]

    diff = P - A

    # l1_error = torch.sum(torch.abs(diff))
    # mae_all = l1_error / (n * n)

    fro_error = torch.norm(diff, p="fro")
    rmse_all = fro_error / n

    return float(rmse_all)

def compute_MAE_error(adj_tensor, P_star, remove_self_loops=True):
    A = adj_tensor.float().clone()
    P = P_star.float().clone()

    if remove_self_loops:
        A.fill_diagonal_(0)
        P.fill_diagonal_(0)

    n = A.shape[0]

    diff = P - A

    l1_error = torch.sum(torch.abs(diff))
    mae_all = l1_error / (n * n)

    return float(mae_all)


def check_mass_preservation(adj_tensor, P_star, remove_self_loops=False, tol=1e-5):
    """
    Check whether ||P*||_{1,1} ≈ ||A||_{1,1}.
    """

    A = adj_tensor.float().clone()
    P = P_star.float().clone()

    if remove_self_loops:
        A.fill_diagonal_(0)
        P.fill_diagonal_(0)

    A_l1 = torch.sum(torch.abs(A))
    P_l1 = torch.sum(torch.abs(P))

    abs_diff = torch.abs(P_l1 - A_l1)
    rel_diff = abs_diff / (A_l1 + 1e-12)

    return {
        "A_l1": float(A_l1),
        "P_l1": float(P_l1),
        "abs_diff": float(abs_diff),
        "rel_diff": float(rel_diff),
        "mass_preserved": bool(rel_diff <= tol),
    }






def link_posterior(priv_adj,priv_deg, eps_a,S):

    device = priv_adj.device
    priv_deg = priv_deg.view(-1, 1).to(device)  # reshape to [n, 1]
    print("priv_deg.shape:",priv_deg.shape)
    n = priv_deg.shape[0]

    device = priv_adj.device
    ones_1xn = torch.ones(1, len(priv_deg)).to(device)
    ones_nx1 = torch.ones(len(priv_deg), 1).to(device)


    def β_model():
        def phi(x):
            exp_x = torch.exp(x)
            exp_neg_x = torch.exp(-x)
            denom = exp_x @ ones_1xn + ones_nx1 @ exp_neg_x.T
            r = 1.0 / denom
            return torch.log(priv_deg) - torch.log(r.sum(1, keepdim=True) - r.diagonal().view(n, 1))

        beta = torch.zeros(n, 1).to(device)
        for _ in range(200):
            beta = phi(beta)

        s = ones_nx1 @ beta.T + beta @ ones_1xn
        prior = torch.exp(s) / (1 + torch.exp(s))
        prior.fill_diagonal_(0)  # remove self-loop
        return prior


    def estimate_posterior_blink(prior):

        p = 1.0 / (1.0 + np.exp(eps_a))
        x = priv_adj

        pr_y_edge = (1 - x) * p + x * (1-p)
        pr_y_no_edge = (1 - x) * (1-p) + x * p

        pij = (pr_y_edge * prior) / (pr_y_edge * prior + pr_y_no_edge * (1 - prior) + 1e-8)
        # # pij = (pr_y_edge * 0.5) / (pr_y_edge * 0.5 + pr_y_no_edge * 0.5 + 1e-8)

        pij.fill_diagonal_(0)
        return pij


    # prior = config_model() #✅
    # prior = β_model() * S# ✅
    prior = β_model()

    # pij = estimate_posterior(prior)
    pij = estimate_posterior_blink(prior)  # ✅

    return pij,prior





def estimate_link_posterior(priv_adj,pi, priv_deg, eps_a):


    device = priv_adj.device
    priv_deg = priv_deg.view(-1, 1).to(device)  # reshape to [n, 1]
    print("priv_deg.shape:",priv_deg.shape)
    n = priv_deg.shape[0]

    device = priv_adj.device
    ones_1xn = torch.ones(1, len(priv_deg)).to(device)
    ones_nx1 = torch.ones(len(priv_deg), 1).to(device)


    def β_model():
        def phi(x):
            exp_x = torch.exp(x)
            exp_neg_x = torch.exp(-x)
            denom = exp_x @ ones_1xn + ones_nx1 @ exp_neg_x.T
            r = 1.0 / denom
            return torch.log(priv_deg) - torch.log(r.sum(1, keepdim=True) - r.diagonal().view(n, 1))

        beta = torch.zeros(n, 1).to(device)
        for _ in range(200):
            beta = phi(beta)

        s = ones_nx1 @ beta.T + beta @ ones_1xn
        prior = torch.exp(s) / (1 + torch.exp(s))
        prior.fill_diagonal_(0)  # remove self-loop
        return prior

    def _eta_from_eps(eps_a: float, ) -> float:

        c = 1.0
        eta_min = 0.01
        eta_max = 1.0
        p = 1.0 / (1.0 + math.exp(eps_a))

        denom = (1.0 - 2.0 * p)
        noise = (p * (1.0 - p)) / (denom * denom + 1e-12)
        eta = 1.0 / (1.0 + c * noise)
        return float(max(eta_min, min(eta_max, eta)))


    def estimate_posterior(prior):
        eta = _eta_from_eps(eps_a)
        floor = 1e-6

        p = 1.0 / (1.0 + np.exp(eps_a))
        prior = prior.clamp(floor, 1.0 - floor)
        logit_prior = torch.log(prior) - torch.log(1.0 - prior)

        llr = math.log((1.0 - p) / (p + 1e-12))  # 似然比强度
        s = 2.0 * priv_adj - 1.0  # x=1→+1, x=0→-1
        logit_post = logit_prior + eta * llr * s

        pij = torch.sigmoid(logit_post)
        pij.fill_diagonal_(0.0)
        return pij


    def estimate_posterior_blink(prior):
        p = 1.0 / (1.0 + np.exp(eps_a)) # RR中的误差概率
        x = priv_adj

        pr_y_edge = (1 - x) * p + x * (1-p)
        pr_y_no_edge = (1 - x) * (1-p) + x * p

        q1 = pr_y_edge * prior
        q2 = pr_y_no_edge * (1 - prior)

        pij = q1 / (q1 + q2 + 1e-8)


        pij.fill_diagonal_(0)

        return pij

    prior = pi

    pij = estimate_posterior_blink(prior)  # ✅

    return pij





import gc
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
def _detect_communities_from_features_optimized(features: torch.Tensor, n_clusters: int) -> np.ndarray:
    print("--- 开始基于特征的社区发现 (内存优化版) ---")

    features_np = features.cpu().numpy().astype(np.float32)

    features_sparse = sparse.csr_matrix(features_np)

    del features_np
    gc.collect()


    features_normalized = normalize(features_sparse, norm='l2', axis=1, copy=False)



    svd = TruncatedSVD(n_components=120, random_state=42, algorithm='randomized')
    features_reduced = svd.fit_transform(features_normalized)


    del features_normalized
    del features_sparse
    gc.collect()


    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             random_state=42,
                             batch_size=4096,
                             n_init=10)
    labels = kmeans.fit_predict(features_reduced)


    # 打印分布
    unique, counts = np.unique(labels, return_counts=True)
    cluster_dist = dict(zip(unique, counts))


    return labels



def _solve_block_beta_model(community_target_degrees: torch.Tensor,
                            community_sizes: torch.Tensor,
                            device: str = 'cpu',
                            iterations: int = 200,
                            epsilon: float = 1e-6) -> torch.Tensor:

    k = community_sizes.shape[0]
    if k == 0:
        return torch.empty(0, 0, device=device)

    community_target_degrees = community_target_degrees.to(device).view(-1, 1)  # [k, 1]
    community_sizes = community_sizes.to(device)  # [k]

    # M_ab = N_c_a * N_c_b (a != b)
    # M_aa = N_c_a * (N_c_a - 1)
    M = torch.outer(community_sizes, community_sizes)
    M.diagonal().sub_(community_sizes)  # M_aa = N_c^2 - N_c = N_c * (N_c - 1)
    M.clamp_min_(0)



    beta_c = torch.zeros(k, 1, device=device)  # [k, 1]

    print(f"  [Beta Solver] K={k}, Iterations={iterations}")
    for i in range(iterations):
        # s_c[a, b] = beta_a + beta_b
        s_c = beta_c @ torch.ones(1, k, device=device) + torch.ones(k, 1, device=device) @ beta_c.T
        # omega[a, b] = P(c_a, c_b)
        omega = torch.exp(s_c) / (1 + torch.exp(s_c))

        # expected_community_degrees[a] = Sum_b( M_ab * Omega_ab )
        expected_community_degrees = (M * omega).sum(dim=1, keepdim=True).clamp_min(epsilon)

        log_ratio = torch.log(community_target_degrees) - torch.log(expected_community_degrees)

        log_ratio = torch.clamp(log_ratio, -1.0, 1.0)

        beta_c_new = beta_c + log_ratio

        diff = torch.abs(beta_c_new - beta_c).max()
        beta_c = beta_c_new
        if diff < epsilon and i > 10:
            print(f"  [Beta Solver] Converged at iteration {i}.")
            break


    s_c = beta_c @ torch.ones(1, k, device=device) + torch.ones(k, 1, device=device) @ beta_c.T
    omega = torch.exp(s_c) / (1 + torch.exp(s_c))
    omega.clamp_(0, 1)

    print("  [Beta Solver] Omega (block prior) calculated.")


    return omega





def calculate_chung_lu_prior(adj_tensor: torch.Tensor,
                             priv_deg: torch.Tensor,
                             features: torch.Tensor = None,
                             method: str = 'spectral',
                             prior_method: str = 'dcsbm',
                             eps_a: float = 5,
                             eps_d: float = 5,
                             n_communities: int = 10,
                             labels_truth: torch.Tensor = None,
                             idx_train: torch.Tensor = None,
                             idx_val: torch.Tensor = None,
                             sigma: float = 1.0 ) -> torch.Tensor:

    if method == 'kmeans_on_features':

        community_labels_np = _detect_communities_from_features_optimized(features, n_clusters=n_communities)
        community_labels = torch.from_numpy(community_labels_np).to(adj_tensor.device)

    elif method == 'MLP':
        community_labels= mlp_for_grouping(
            features=features,
            labels_truth=labels_truth,
            idx_train=idx_train,
            idx_val=idx_val,
            hidden_dim=256,
            emb_dim=128,
            lr=1e-3,
            weight_decay=5e-4,
            dropout=0.5,
            max_epochs=500,
            patience=50,
        )


    elif method == 'random':

        num_nodes = adj_tensor.shape[0] if adj_tensor is not None else features.shape[0]

        community_labels = torch.randint(
            low=0,
            high=n_communities,
            size=(num_nodes,),
            device=adj_tensor.device
        )



    elif method == 'random_balanced':
        num_nodes = adj_tensor.shape[0]

        base_labels = torch.arange(num_nodes, device=adj_tensor.device) % n_communities


        shuffled_indices = torch.randperm(num_nodes, device=adj_tensor.device)
        community_labels = base_labels[shuffled_indices]

    elif method == 'degree_binning':

        degrees = priv_deg.squeeze()

        quantiles = torch.linspace(0, 1, steps=n_communities + 1).to(degrees.device)

        bucket_boundaries = torch.quantile(degrees.float(), quantiles)


        community_labels = torch.bucketize(degrees.float(), bucket_boundaries, right=False)

        community_labels = torch.clamp(community_labels, min=0, max=n_communities - 1)


        for i in range(n_communities):
            mask = (community_labels == i)
            if mask.sum() > 0:
                avg_deg = degrees[mask].float().mean().item()

    else:
        raise ValueError("!!!!!" )

    print("社区发现完成。")



    community_one_hot = torch.nn.functional.one_hot(community_labels.to(torch.long), num_classes=n_communities).float()
    # print(community_one_hot)


    # [k]
    community_sizes = community_one_hot.sum(dim=0)
    # [k, 1]
    community_target_degrees = torch.matmul(community_one_hot.T, priv_deg.view(-1, 1))

    prior_matrix = torch.zeros_like(adj_tensor)

    "---------------mlp_for_grouping---------------"
    # num_classes = int(labels_truth.max().item()) + 1
    # community_one_hot = F.one_hot(community_labels, num_classes=num_classes).float()

    if prior_method == 'beta':

        omega = _solve_block_beta_model(
            community_target_degrees,
            community_sizes,
        )

        prior_matrix = omega[community_labels[:, None], community_labels]


    elif prior_method == 'chung-du':


        noisy_degree_vectors = torch.matmul(adj_tensor, community_one_hot)  # [N, k]
        # print("节点i在社区中的raw度：", noisy_degree_vectors[:5])

        "-----------BRR----------"

        noisy_degree_vectors = (noisy_degree_vectors != 0).float()

        noisy_degree_rr= matrix_randomized_response(noisy_degree_vectors, eps_a)
        noisy_degree_vectors =noisy_degree_rr


        "-----------laplace----------"
        # noise = torch.distributions.Laplace(loc=0.0, scale=1.0 / eps_a).sample(noisy_degree_vectors.shape)
        # priv_deg = noisy_degree_vectors + noise
        # priv_deg = torch.clamp(priv_deg, min=0)
        # noisy_degree_vectors = priv_deg

        "--------------GRR-----------"

        # d = int(noisy_degree_vectors.max().item()) + 1
        # noisy_degree_vectors = matrix_generalized_randomized_response(noisy_degree_vectors, eps_a, d=d)


        volume_matrix = torch.matmul(community_one_hot.T, noisy_degree_vectors)  # [k, k]
        volume_matrix.clamp_min_(1e-8)

        num_connected_communities = torch.sum(noisy_degree_vectors, dim=1, keepdim=True)


        scaling_factors = priv_deg / (num_connected_communities + 1e-10)
        scaling_factors = scaling_factors ** sigma
        weighted_degree_vectors = noisy_degree_vectors * scaling_factors


        inv_volume_matrix = 1.0 / (volume_matrix + 1e-10)

        temp = torch.matmul(community_one_hot, inv_volume_matrix)
        inv_vol_expanded = torch.matmul(temp, community_one_hot.T)

        numerator_1 = torch.matmul(weighted_degree_vectors, community_one_hot.T)
        prior_matrix = numerator_1 * numerator_1.T * inv_vol_expanded


    elif prior_method == 'dcsbm':


        A_dcsbm, P_dcsbm, block_edges, node_degree = dcsbm_reconstruct(
            adj_tensor=adj_tensor,
            community_one_hot=community_one_hot,
            sample=False,
            undirected=True,
            remove_self_loops=True
        )
        prior_matrix = P_dcsbm


    elif prior_method == 'sbm':




        A_sbm, prior_sbm, block_prob = sbm_reconstruct(
            adj_tensor=adj_tensor,
            community_one_hot=community_one_hot,
            sample=False
        )

        prior_matrix = prior_sbm

    elif prior_method == 'LDPGen':

        A_weighted, P_syn, delta = generate_graph_by_delta_formula(
            adj_tensor=adj_tensor,
            community_one_hot=community_one_hot,
            sample=False,
            remove_self_loops=True,
            symmetrize=False
        )
        "-------------------physics---------------"
        # A_weighted, P_syn, delta = generate_graph_by_delta_formula_batch_dense(
        #     adj_tensor=adj_tensor,
        #     community_one_hot=community_one_hot,
        #     sample=False,
        #     remove_self_loops=True,
        #     symmetrize=False,
        #     batch_size=512,
        #     return_prob_matrix=False
        # )


        prior_matrix = P_syn

    elif prior_method == 'uniform':

        recon_adj, group_degree, group_size = uniform_group_reconstruct(
            adj_tensor=adj_tensor,
            community_one_hot=community_one_hot,
            remove_self_loops=True,
            symmetrize=False
        )
        prior_matrix = recon_adj



    return prior_matrix, community_one_hot



def compute_H_from_weighted_vectors(weighted_degree_vectors, community_one_hot, eps=1e-12):
    """
    Compute H = max_{a,b: V_ab > 0} h_ab

    h_ab =
        sum_{j in G_b} W[j,a]^2
        /
        (sum_{j in G_b} W[j,a])^2

    weighted_degree_vectors = W_tilde, shape [N, K]
    community_one_hot = Q, shape [N, K]
    """

    Q = community_one_hot.float()
    W = weighted_degree_vectors.float()

    # V[a,b] = sum_{j in G_b} W[j,a]
    # (Q.T @ W)[b,a] = sum_{j in G_b} W[j,a]
    V = (Q.T @ W).T  # [K, K]

    # numerator[a,b] = sum_{j in G_b} W[j,a]^2
    H_num = (Q.T @ (W ** 2)).T  # [K, K]

    mask = V > 0

    h = torch.zeros_like(V)
    h[mask] = H_num[mask] / (V[mask] ** 2 + eps)

    if mask.sum() > 0:
        H = h[mask].max()
    else:
        H = torch.tensor(0.0, device=W.device)

    return H, h, V


def compute_q_upper_bound(adj_tensor, community_one_hot, eps_g, eps=1e-12):
    """
    Compute q_i upper bound:

        q_i <= min{1, 2/mu_i + exp(-mu_i/8)}

    where

        mu_i = m_i + (K - 2m_i)/(1 + exp(eps_g))

    and

        m_i = sum_b 1{(AQ)_{i,b} > 0}.
    """

    A = adj_tensor.float()
    Q = community_one_hot.float()
    device = A.device

    N, K = Q.shape

    # B_count[i,b] = number of neighbors of i in group b
    B_count = A @ Q  # [N, K]

    # z[i,b] = 1{B_count[i,b] > 0}
    Z = (B_count > 0).float()  # [N, K]

    # m_i = number of active groups for node i
    m = Z.sum(dim=1)  # [N]

    eps_g_tensor = torch.tensor(float(eps_g), device=device)

    # p = 1 / (1 + exp(eps_g))
    p = 1.0 / (1.0 + torch.exp(eps_g_tensor))

    # # mu_i = E[m_tilde_i] = m_i + p(K - 2m_i)
    # mu = m + p * (K - 2.0 * m)
    # # q_i upper bound
    # q_ub = 2.0 / torch.clamp(mu, min=eps) + torch.exp(-mu / 8.0)
    # # q_i is always <= 1
    # q_ub = torch.clamp(q_ub, max=1.0)


    "torch.full_like(m, 2.0),"
    q_rw = torch.minimum(
        torch.full_like(m, 1.0),
        2.0 * K / ((m + 1.0) * (1.0 + torch.exp(eps_g_tensor)))
    )

    return q_rw


def compute_reweighted_bound_with_q_upper(
    adj_tensor,
    community_one_hot,
    eps_g,
    eps_d,
    eps=1e-12,
):
    "1/n E||~P-P||F"
    A = adj_tensor.float()
    Q = community_one_hot.float()

    device = A.device
    N, K = Q.shape

    # --------------------------------------------------
    # 1. Degree d_i and |E|
    # --------------------------------------------------
    d = A.sum(dim=1)       # [N]
    E_count = d.sum()      # |E| = sum_i d_i = sum_{i,j} A_ij


    # --------------------------------------------------
    # 2. q_i upper bound
    # --------------------------------------------------
    q_rw = compute_q_upper_bound(
        adj_tensor=A,
        community_one_hot=Q,
        eps_g=eps_g,
        eps=eps
    )

    # --------------------------------------------------
    # 3. H from weighted_degree_vectors
    # --------------------------------------------------
    # H, h_matrix, V = compute_H_from_weighted_vectors(
    #     weighted_degree_vectors=W,
    #     community_one_hot=Q,
    #     eps=eps
    # )
    H=1
    # --------------------------------------------------
    # 4. Bound
    # --------------------------------------------------
    eps_d_tensor = torch.tensor(float(eps_d), device=device)

    inside = H * torch.sum(
        (d ** 2)*q_rw + 2.0 / (eps_d_tensor ** 2))


    "1/n E||~P-P||F"
    # bound = (torch.sqrt(inside) + torch.sqrt(E_count)) / N
    # bound = (torch.sqrt(inside + E_count)) / N
    # bound = (torch.sqrt(inside)) / N
    bound = inside




    return bound.item()
    #     {
    #     "bound": bound.item(),
    #     "H": H.item(),
    #     "q_ub_mean": q_ub.mean().item(),
    #     "q_ub_max": q_ub.max().item(),
    #     "q_ub_min": q_ub.min().item(),
    #     "mu_mean": mu.mean().item(),
    #     "mu_max": mu.max().item(),
    #     "mu_min": mu.min().item(),
    #     "m_mean": m.mean().item(),
    #     "m_max": m.max().item(),
    #     "m_min": m.min().item(),
    #     "E_count": E_count.item(),
    #     "sum_d2": (d ** 2).sum().item(),
    #     "avg_degree": (E_count / N).item(),
    #     "max_degree": d.max().item(),
    #     "q_ub": q_ub,
    #     "mu": mu,
    #     "m": m,
    #     "h_matrix": h_matrix,
    #     "V": V,
    # }



def compute_reweighted_MAE_bound(
    adj_tensor,
    community_one_hot,
    eps_g,
    eps_d,
    eps=1e-12,
):
    "1/n E||~P-P||F"
    A = adj_tensor.float()
    Q = community_one_hot.float()

    device = A.device
    N, K = Q.shape

    # --------------------------------------------------
    # 1. Degree d_i and |E|
    # --------------------------------------------------
    d = A.sum(dim=1)       # [N]
    E_count = d.sum()      # |E| = sum_i d_i = sum_{i,j} A_ij


    # --------------------------------------------------
    # 2. q_i upper bound
    # --------------------------------------------------
    q_rw = compute_q_upper_bound(
        adj_tensor=A,
        community_one_hot=Q,
        eps_g=eps_g,
        eps=eps
    )


    # --------------------------------------------------
    # 4. Bound
    # --------------------------------------------------
    eps_d_tensor = torch.tensor(float(eps_d), device=device)

    inside =  3 * torch.sum(
        d*q_rw + 1.0 / (eps_d_tensor))

    # inside = torch.sum(
    #     d * q_rw + 1.0 / (eps_d_tensor))

    bound = inside /(N*N)


    return bound.item()




def swap_and_append(arr):
    new_arr = []
    for array in arr:
        if len(array) == 2:
            new_array = [array[1], array[0]]
            new_arr.append(new_array)
        new_arr.append(array)
    return new_arr





class MLPEncoderClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, emb_dim, num_classes, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.dropout = dropout

    def encode(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)
        x = F.relu(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        logits = self.classifier(z)
        return logits, z


def accuracy(logits, labels):
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item()


def mlp_for_grouping(
    features,
    labels_truth,
    idx_train,
    idx_val,
    hidden_dim=256,
    emb_dim=128,
    lr=1e-3,
    weight_decay=5e-4,
    dropout=0.5,
    max_epochs=500,
    patience=50,
):
    device = features.device
    in_dim = features.shape[1]
    num_classes = int(labels_truth.max().item()) + 1

    model = MLPEncoderClassifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(max_epochs):
        # ---- train ----
        model.train()
        optimizer.zero_grad()

        logits, _ = model(features)
        loss_train = F.cross_entropy(logits[idx_train], labels_truth[idx_train])
        loss_train.backward()
        optimizer.step()

        # ---- val ----
        model.eval()
        with torch.no_grad():
            logits_val, _ = model(features)
            val_loss = F.cross_entropy(logits_val[idx_val], labels_truth[idx_val]).item()
            val_acc = accuracy(logits_val[idx_val], labels_truth[idx_val])

        improved = False
        if val_acc > best_val_acc:
            improved = True
        elif val_acc == best_val_acc and val_loss < best_val_loss:
            improved = True

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                train_acc = accuracy(logits_val[idx_train], labels_truth[idx_train])
            print(
                f"Epoch {epoch+1:03d} | "
                f"train_loss={loss_train.item():.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits, embeddings = model(features)
        group_labels = logits.argmax(dim=1)

    return group_labels



def train_mlp_for_grouping(
    features,
    labels_truth,
    idx_train,
    idx_val,
    hidden_dim=256,
    emb_dim=128,
    lr=1e-3,
    weight_decay=5e-4,
    dropout=0.5,
    max_epochs=500,
    patience=50,
):
    device = features.device
    in_dim = features.shape[1]
    num_classes = int(labels_truth.max().item()) + 1

    model = MLPEncoderClassifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(max_epochs):
        # ---- train ----
        model.train()
        optimizer.zero_grad()

        logits, _ = model(features)
        loss_train = F.cross_entropy(logits[idx_train], labels_truth[idx_train])
        loss_train.backward()
        optimizer.step()

        # ---- val ----
        model.eval()
        with torch.no_grad():
            logits_val, _ = model(features)
            val_loss = F.cross_entropy(logits_val[idx_val], labels_truth[idx_val]).item()
            val_acc = accuracy(logits_val[idx_val], labels_truth[idx_val])

        improved = False
        if val_acc > best_val_acc:
            improved = True
        elif val_acc == best_val_acc and val_loss < best_val_loss:
            improved = True

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                train_acc = accuracy(logits_val[idx_train], labels_truth[idx_train])
            print(
                f"Epoch {epoch+1:03d} | "
                f"train_loss={loss_train.item():.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model



def build_community_labels_with_mlp_kmeans(
    features,
    labels_truth,
    idx_train,
    idx_val,
    n_communities=500,
    adj_tensor=None,
    hidden_dim=256,
    emb_dim=128,
    lr=1e-3,
    weight_decay=5e-4,
    dropout=0.5,
    max_epochs=500,
    patience=50,
):

    if features.is_sparse:
        features = features.to_dense()

    features = features.float()
    labels_truth = labels_truth.long()

    target_device = adj_tensor.device if adj_tensor is not None else features.device

    features = features.to(target_device)
    labels_truth = labels_truth.to(target_device)
    idx_train = idx_train.to(target_device)
    idx_val = idx_val.to(target_device)

    num_nodes = features.shape[0]


    train_val_idx = torch.cat([idx_train, idx_val], dim=0)

    if len(train_val_idx) < n_communities:
        raise ValueError(
            f"train+val 节点数 {len(train_val_idx)} 小于 n_communities={n_communities}，无法聚类。"
        )


    model = train_mlp_for_grouping(
        features=features,
        labels_truth=labels_truth,
        idx_train=idx_train,
        idx_val=idx_val,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        max_epochs=max_epochs,
        patience=patience,
    )


    model.eval()
    with torch.no_grad():
        _, embeddings = model(features)   # [N, emb_dim]

    embeddings_np = embeddings.detach().cpu().numpy()
    train_val_idx_np = train_val_idx.detach().cpu().numpy()


    kmeans = KMeans(
        n_clusters=n_communities,
        random_state=42,
        n_init=20
    )
    # kmeans.fit(embeddings_np[train_val_idx_np])

    kmeans.fit(embeddings_np)


    community_labels_np = kmeans.predict(embeddings_np)

    community_labels = torch.tensor(
        community_labels_np,
        dtype=torch.long,
        device=target_device
    )



    return community_labels, embeddings


def sbm_reconstruct(adj_tensor, community_one_hot, sample=True, undirected=True, remove_self_loops=True):


    A = adj_tensor.float()
    Q = community_one_hot.float()

    device = A.device
    Q = Q.to(device)

    N, K = Q.shape

    if remove_self_loops:
        A = A.clone()
        A.fill_diagonal_(0.0)


    group_sizes = Q.sum(dim=0)  # [K]

    edge_counts = Q.T @ A @ Q  # [K, K]


    possible_edges = group_sizes.view(K, 1) * group_sizes.view(1, K)

    if remove_self_loops:
        idx = torch.arange(K, device=device)
        possible_edges[idx, idx] -= group_sizes


    block_prob = edge_counts / possible_edges.clamp_min(1.0)
    block_prob = block_prob.clamp(0.0, 1.0)

    prob_matrix = Q @ block_prob @ Q.T

    if remove_self_loops:
        prob_matrix = prob_matrix.clone()
        prob_matrix.fill_diagonal_(0.0)

    if sample:
        if undirected:
            upper_prob = torch.triu(prob_matrix, diagonal=1)
            upper_adj = torch.bernoulli(upper_prob)
            recon_adj = upper_adj + upper_adj.T
        else:
            recon_adj = torch.bernoulli(prob_matrix)
            if remove_self_loops:
                recon_adj.fill_diagonal_(0.0)
    else:
        recon_adj = prob_matrix

    return recon_adj, prob_matrix, block_prob



def dcsbm_reconstruct(
    adj_tensor,
    community_one_hot,
    sample=True,
    undirected=True,
    remove_self_loops=True,
):
    """
    DCSBM graph reconstruction from adjacency matrix and one-hot community matrix.

    Parameters
    ----------
    adj_tensor : torch.Tensor
        Original adjacency matrix A, shape [N, N].

    community_one_hot : torch.Tensor
        One-hot community assignment matrix Q, shape [N, K].

    sample : bool
        If True, sample a binary graph from DCSBM probabilities.
        If False, return the weighted probability matrix directly.

    undirected : bool
        Whether the graph is undirected.

    remove_self_loops : bool
        Whether to remove self-loops.

    Returns
    -------
    recon_adj : torch.Tensor
        Reconstructed adjacency matrix, shape [N, N].

    prob_matrix : torch.Tensor
        DCSBM node-to-node probability matrix, shape [N, N].

    block_edges : torch.Tensor
        Group-to-group edge count matrix, shape [K, K].

    node_degree : torch.Tensor
        Node degree vector computed from adj_tensor, shape [N].
    """

    A = adj_tensor.float()
    Q = community_one_hot.float().to(A.device)

    N, K = Q.shape

    if remove_self_loops:
        A = A.clone()
        A.fill_diagonal_(0.0)


    node_degree = A.sum(dim=1)  # [N]


    group_degree = Q.T @ node_degree  # [K]


    block_edges = Q.T @ A @ Q  # [K, K]


    denominator = group_degree.view(K, 1) * group_degree.view(1, K)


    if remove_self_loops:
        degree_square_sum = Q.T @ (node_degree ** 2)  # [K]
        idx = torch.arange(K, device=A.device)
        denominator[idx, idx] -= degree_square_sum

    denominator = denominator.clamp_min(1e-12)


    omega = block_edges / denominator  # [K, K]


    group_factor = Q @ omega @ Q.T  # [N, N]


    degree_outer = node_degree.view(N, 1) * node_degree.view(1, N)


    prob_matrix = degree_outer * group_factor

    if remove_self_loops:
        prob_matrix = prob_matrix.clone()
        prob_matrix.fill_diagonal_(0.0)


    prob_matrix = prob_matrix.clamp(0.0, 1.0)


    if sample:
        if undirected:
            upper_prob = torch.triu(prob_matrix, diagonal=1)
            upper_adj = torch.bernoulli(upper_prob)
            recon_adj = upper_adj + upper_adj.T
        else:
            recon_adj = torch.bernoulli(prob_matrix)
            if remove_self_loops:
                recon_adj.fill_diagonal_(0.0)
    else:
        recon_adj = prob_matrix

    return recon_adj, prob_matrix, block_edges, node_degree



def generate_graph_by_delta_formula(
    adj_tensor,
    community_one_hot,
    sample=True,
    remove_self_loops=True,
    symmetrize=False,
):
    """
    Generate a synthetic graph using the formula:

        p_uv =
        delta[u, j] * (sum_x delta[x, j] / |U_j|)
        /
        (sum_{x in U_i} delta[x, j] + sum_x delta[x, j])

    where:
        u in U_i,
        v in U_j,
        delta = A @ Q.

    Parameters
    ----------
    adj_tensor : torch.Tensor
        Adjacency matrix A, shape [N, N].

    community_one_hot : torch.Tensor
        One-hot community matrix Q, shape [N, K].

    sample : bool
        If True, sample a binary graph from probability matrix.
        If False, return the probability matrix directly.

    remove_self_loops : bool
        Whether to remove self-loops.

    symmetrize : bool
        Whether to symmetrize the probability matrix.
        The original formula is not strictly symmetric.

    Returns
    -------
    recon_adj : torch.Tensor
        Reconstructed adjacency matrix, shape [N, N].

    prob_matrix : torch.Tensor
        Edge probability matrix, shape [N, N].

    delta : torch.Tensor
        Node-to-group degree matrix, shape [N, K].
        delta[u, j] means node u's estimated degree to group j.
    """

    A = adj_tensor.float()
    Q = community_one_hot.float().to(A.device)

    N, K = Q.shape

    if remove_self_loops:
        A = A.clone()
        A.fill_diagonal_(0.0)

    delta = A @ Q  # [N, K]


    group_size = Q.sum(dim=0).clamp_min(1.0)  # [K]


    total_to_group = delta.sum(dim=0)  # [K]


    target_avg = total_to_group / group_size  # [K]

    block_sum = Q.T @ delta  # [K, K]


    delta_to_target = delta @ Q.T  # [N, N]


    target_avg_per_v = Q @ target_avg  # [N]


    numerator = delta_to_target * target_avg_per_v.view(1, N)


    block_sum_by_pair = Q @ block_sum @ Q.T  # [N, N]


    total_to_group_per_v = Q @ total_to_group  # [N]

    denominator = block_sum_by_pair + total_to_group_per_v.view(1, N)

    prob_matrix = numerator / denominator.clamp_min(1e-12)
    prob_matrix = prob_matrix.clamp(0.0, 1.0)

    if remove_self_loops:
        prob_matrix = prob_matrix.clone()
        prob_matrix.fill_diagonal_(0.0)


    if symmetrize:
        prob_matrix = 0.5 * (prob_matrix + prob_matrix.T)
        if remove_self_loops:
            prob_matrix.fill_diagonal_(0.0)


    if sample:
        recon_adj = torch.bernoulli(prob_matrix)

        if symmetrize:

            recon_adj = torch.triu(recon_adj, diagonal=1)
            recon_adj = recon_adj + recon_adj.T

        if remove_self_loops:
            recon_adj.fill_diagonal_(0.0)
    else:
        recon_adj = prob_matrix



    return recon_adj, prob_matrix, delta




def generate_graph_by_delta_formula_batch_dense(
    adj_tensor,
    community_one_hot,
    sample=True,
    remove_self_loops=True,
    symmetrize=False,
    batch_size=512,
    return_prob_matrix=False,
):
    """
    Batched dense-output implementation.

    It avoids constructing large intermediate N x N matrices all at once,
    but the final output can still be dense.

    Formula:
        p_uv =
        delta[u, j] * (sum_x delta[x, j] / |U_j|)
        /
        (sum_{x in U_i} delta[x, j] + sum_x delta[x, j])

    where:
        u in U_i,
        v in U_j,
        delta = A @ Q.

    Parameters
    ----------
    adj_tensor : torch.Tensor
        Adjacency matrix A, shape [N, N]. Dense or sparse.

    community_one_hot : torch.Tensor
        One-hot community matrix Q, shape [N, K].

    sample : bool
        If True, return sampled dense binary adjacency.
        If False, return dense weighted probability matrix.

    remove_self_loops : bool
        Whether to remove self-loops.

    symmetrize : bool
        Whether to symmetrize the final matrix.

    batch_size : int
        Number of rows computed each time.

    return_prob_matrix : bool
        If True and sample=True, also return dense probability matrix.
        This costs another N x N dense matrix.
        Usually keep it False to save memory.

    Returns
    -------
    recon_adj : torch.Tensor
        Dense reconstructed adjacency matrix, shape [N, N].

    prob_matrix : torch.Tensor or None
        Dense probability matrix if requested.
        If sample=False, prob_matrix is the same as recon_adj.
        If sample=True and return_prob_matrix=False, prob_matrix is None.

    delta : torch.Tensor
        Node-to-group degree matrix, shape [N, K].
    """

    A = adj_tensor.float()
    Q = community_one_hot.float().to(A.device)

    device = A.device
    N, K = Q.shape

    # --------------------------------------------------
    # 1. remove self-loops from original adjacency
    # --------------------------------------------------
    if remove_self_loops:
        if A.is_sparse:
            A = A.coalesce()
            idx = A.indices()
            val = A.values()
            keep = idx[0] != idx[1]

            A = torch.sparse_coo_tensor(
                idx[:, keep],
                val[keep],
                size=A.shape,
                device=device,
                dtype=A.dtype
            ).coalesce()
        else:
            A = A.clone()
            A.fill_diagonal_(0.0)

    # --------------------------------------------------
    # 2. delta = A @ Q
    # delta[u, j] = node u's degree to group j
    # --------------------------------------------------
    if A.is_sparse:
        delta = torch.sparse.mm(A, Q)   # [N, K]
    else:
        delta = A @ Q                   # [N, K]

    # 如果有 LDP 噪声导致负数，可以打开
    # delta = delta.clamp_min(0.0)

    # --------------------------------------------------
    # 3. group-level statistics
    # --------------------------------------------------
    group_size = Q.sum(dim=0).clamp_min(1.0)  # [K]

    # total_to_group[j] = sum_x delta[x, j]
    total_to_group = delta.sum(dim=0)         # [K]

    # target_avg[j] = sum_x delta[x, j] / |U_j|
    target_avg = total_to_group / group_size  # [K]

    # block_sum[i, j] = sum_{x in U_i} delta[x, j]
    block_sum = Q.T @ delta                   # [K, K]

    community_labels = Q.argmax(dim=1)        # [N]

    # --------------------------------------------------
    # 4. allocate final dense matrix
    # --------------------------------------------------
    recon_adj = torch.empty(
        (N, N),
        dtype=torch.float32,
        device=device
    )

    if sample and return_prob_matrix:
        prob_matrix = torch.empty(
            (N, N),
            dtype=torch.float32,
            device=device
        )
    else:
        prob_matrix = None

    # --------------------------------------------------
    # 5. batch computation
    # --------------------------------------------------
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start


        batch_groups = community_labels[start:end]  # [B]

        # delta[u, j]
        delta_batch = delta[start:end, :]           # [B, K]

        # denom[u, j] =
        # block_sum[g(u), j] + total_to_group[j]
        denom_group = block_sum[batch_groups, :] + total_to_group.view(1, K)  # [B, K]

        # p_group[u, j]
        p_group = delta_batch * target_avg.view(1, K) / denom_group.clamp_min(1e-12)
        p_group = p_group.clamp(0.0, 1.0)  # [B, K]


        p_batch = p_group @ Q.T  # [B, N]

        if remove_self_loops:
            local_rows = torch.arange(B, device=device)
            global_cols = torch.arange(start, end, device=device)
            p_batch[local_rows, global_cols] = 0.0

        if sample:
            sampled_batch = torch.bernoulli(p_batch)
            recon_adj[start:end, :] = sampled_batch

            if return_prob_matrix:
                prob_matrix[start:end, :] = p_batch
        else:
            recon_adj[start:end, :] = p_batch

        del delta_batch, denom_group, p_group, p_batch

    # --------------------------------------------------
    # 6. symmetrize if needed
    # --------------------------------------------------
    if symmetrize:
        recon_adj = 0.5 * (recon_adj + recon_adj.T)

        if sample:
            recon_adj = (recon_adj > 0).float()

        if return_prob_matrix and prob_matrix is not None:
            prob_matrix = 0.5 * (prob_matrix + prob_matrix.T)

    if remove_self_loops:
        recon_adj.fill_diagonal_(0.0)

        if prob_matrix is not None:
            prob_matrix.fill_diagonal_(0.0)


    if not sample:
        prob_matrix = recon_adj

    return recon_adj, prob_matrix, delta

def uniform_group_reconstruct(
    adj_tensor,
    community_one_hot,
    remove_self_loops=True,
    symmetrize=False,
):
    """
    Simple uniform group-level reconstruction.

    For each node u:
        1. Compute its node-to-group degree:
              group_degree[u, k] = sum_{v in G_k} A[u, v]

        2. Uniformly distribute this group degree to all nodes in that group:
              P[u, v] = group_degree[u, g(v)] / |G_{g(v)}|

    Parameters
    ----------
    adj_tensor : torch.Tensor
        Adjacency matrix A, shape [N, N].

    community_one_hot : torch.Tensor
        One-hot community assignment matrix Q, shape [N, K].

    remove_self_loops : bool
        Whether to remove self-loops in the reconstructed matrix.

    symmetrize : bool
        Whether to symmetrize the reconstructed matrix.
        Use True if you need an undirected weighted graph.

    Returns
    -------
    recon_adj : torch.Tensor
        Reconstructed weighted adjacency matrix, shape [N, N].

    group_degree : torch.Tensor
        Node-to-group degree matrix, shape [N, K].

    group_size : torch.Tensor
        Number of nodes in each group, shape [K].
    """

    A = adj_tensor.float()
    Q = community_one_hot.float().to(A.device)

    if remove_self_loops:
        A = A.clone()
        A.fill_diagonal_(0.0)


    group_degree = A @ Q  # [N, K]


    group_size = Q.sum(dim=0).clamp_min(1.0)  # [K]


    avg_group_degree = group_degree / group_size.view(1, -1)  # [N, K]


    recon_adj = avg_group_degree @ Q.T  # [N, N]


    if remove_self_loops:
        recon_adj = recon_adj.clone()
        recon_adj.fill_diagonal_(0.0)


    if symmetrize:
        recon_adj = 0.5 * (recon_adj + recon_adj.T)
        if remove_self_loops:
            recon_adj.fill_diagonal_(0.0)

    return recon_adj, group_degree, group_size
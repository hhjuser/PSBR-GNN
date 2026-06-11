
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
import community as community_louvain # 这是 python-louvain 库
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



def rr_adj(adj_tensor: torch.Tensor, eps_edge: float) -> torch.Tensor:

    n = adj_tensor.size(0)

    p = 1.0 / (1.0 + math.exp(eps_edge))
    noise = torch.bernoulli(torch.full((n, n), p))
    res = ((adj_tensor + noise) % 2).float()

    return res




def dense_to_sparse_with_weights(priv_adj: torch.Tensor):
    """
    将稠密的概率邻接矩阵转换为稀疏 edge_index 和 edge_weight。
    要求 priv_adj 是一个 n×n 的张量。
    """
    # 将 priv_adj 转为稀疏矩阵（会丢弃为0的边）
    sparse = priv_adj.to_sparse()

    # 获取 COO 格式的边索引和权重
    edge_index = sparse.indices()  # shape: [2, num_edges]
    edge_weight = sparse.values()  # shape: [num_edges]

    return edge_index, edge_weight




@torch.no_grad()
def retain_topm_global(W: torch.Tensor, A_count_from: torch.Tensor):
    """
    全局Top-m：m = round(sum(A_count_from的上三角))，只保留W上三角中得分最高的m条边（对称写回）
    A_count_from 可以是 perturbed_adj 或去偏后的 A0，推荐 A0
    """
    W = W.clone()
    n = W.size(0)
    triu = torch.triu(torch.ones_like(W, dtype=torch.bool), diagonal=1)
    m = int(torch.round(A_count_from[triu].sum()).item())
    if m <= 0:
        out = torch.zeros_like(W)
        out.fill_diagonal_(0.0)
        return out
    # 只在上三角打分
    scores = W[triu]
    vals, idx = torch.topk(scores, k=min(m, scores.numel()))
    mask = torch.zeros_like(scores)
    mask[idx] = 1.0
    # 写回
    out = torch.zeros_like(W)
    out[triu] = scores * mask
    out = out + out.t()
    out.fill_diagonal_(0.0)
    return out


@torch.no_grad()
def retain_topk_by_degree(W: torch.Tensor, k_vec: torch.Tensor, sym: str = 'union'):
    """
    在软邻接 W 上按每行 k_i=round(k_vec[i]) 进行 Top-K 稀疏化（仅在正权重上选）
    sym: 'union' | 'intersection' | None
    """
    assert W.dim() == 2 and W.size(0) == W.size(1)
    n = W.size(0)

    if k_vec.dim() == 2:
        k_vec = k_vec.squeeze(1)
    k_vec = torch.round(k_vec).clamp(0, n-1).to(torch.long)

    # 只把正权重作为候选；对角线禁止选择
    scores = W.clone()
    # scores.fill_diagonal_(float('-inf'))
    scores[scores <= 0] = float('-inf')
    # 给分数加点扰动，避免全相等

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
        out[i, idx] = W[i, idx]  # 保留原权重

    # 对称化   行内Top-K之后，又用 sym='union' 做了对称并集
    if sym == 'union':  #union（并集）    # ✅
        out = torch.max(out, out.t())
    elif sym == 'intersection':  #intersection（交集）
        out = torch.min(out, out.t())
    else:  # sym is None or other
        out = out  # 保持非对称
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
    # data_file_root = Path(path) / dataset_name

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


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


    # upper = torch.triu(adj_tensor, diagonal=1)
    # deg_upper = upper.sum(dim=1, keepdim=True)
    # deg = deg_upper.float()


    eps_d  = eps_edge
    eps_a = 0

    print("dataset_name:", dataset_name,"eps_d:", eps_d, "eps_a:", eps_a, "communities:", n_communities)


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


    n = deg.shape[0]

    gen = torch.Generator(device=deg.device)
    gen.manual_seed(42)

    n_v = int(num_nodes)
    beta = 1.0
    r = rj_choose_r_paper(eps=eps_d, n_v=n_v, beta=beta, r_min=1)
    print(f"[RJ] eps_d={eps_d:.4f}, n_v={n_v}, beta={beta}, r={r}")

    tilde_deg = rj_perturb_degrees(deg, eps=eps_d, r=r, n=n, generator=gen)

    d_max = int(min(n - 1, max(10, int(deg.max().item()))))
    f_hat = unbias_degree_distribution_from_rj(tilde_deg, eps=eps_d, r=r, d_max=d_max)

    d_target, m = sample_target_degrees(f_hat, n_nodes=n, n=n, min_deg=1, max_deg=d_max, generator=gen)

    adj_synth = chung_lu_accept_reject_adj(
        d_target=d_target.to(deg.device),
        m=m,
        features=features.to(deg.device),
        alpha_rate=alpha_rate,
        generator=gen,
        weighted=False
    )

    print(adj_synth.shape, adj_synth.sum() / 2)



    edge_index, edge_weight = dense_to_sparse_with_weights(adj_synth)


    '=============================================================='

    # 返回
    return edge_index, edge_weight, features, labels_truth, idx_train, idx_val, idx_test



def rj_choose_r_paper(eps: float, n_v: int, beta: float,
                      r_min: int = 1, r_max: int = None) -> int:
    """
    按 AsgLDP 原文 Eq.(13) 计算 RJ 的最优 jump radius r。

    参数:
      eps  : RJ 的隐私预算 ε（注意是给 RJ 的那部分预算）
      n_v  : 参与节点数（原文记为 n_v；离线数据集通常取 n_v = num_nodes）
      beta : 权衡系数 β（原文用它在 UtilityLoss 和 PrivacyLeakage 之间做 tradeoff）
      r_min: 最小半径，默认 1
      r_max: 最大半径（可选）；如果不传，会自动用 (n_v-1)//2 裁剪，确保 2r < n_v

    返回:
      r: int，最终取整并裁剪后的 r
    """
    # 基本合法性
    if n_v <= 2 or beta <= 0:
        return max(1, int(r_min))

    # 原文 e = exp(eps)
    e = math.exp(float(eps))

    # 为了让 (e-2) 出现时不出问题：当 eps 很小时 e≈1，会导致 e-2<0
    # 这时 Eq.(13) 本身就不适用（会开根号负数），给保守 r
    if e <= 2.0:
        r = int(r_min)
    else:
        denom = 8.0 * beta * e - e * e  # 8βe - e^2

        # denom<=0 会导致第一项根号里为负或发散，同样给保守 r
        if denom <= 0:
            r = int(r_min)
        else:
            term1 = (n_v * (e - 2.0)) / denom
            term2 = (e - 2.0) / (2.0 * n_v * beta * e)

            # 数值保护
            if term1 <= 0 or term2 < 0:
                r = int(r_min)
            else:
                r_star = math.sqrt(term1) - math.sqrt(term2)  # Eq.(13)
                r = int(round(r_star))

    # r 至少为 r_min
    r = max(int(r_min), r)

    # 原文隐私泄露项含 (n_v - 2r)/(n_v + 2r)，实践中强制 2r < n_v
    auto_rmax = (n_v - 1) // 2
    if r_max is None:
        r_max = auto_rmax
    else:
        r_max = min(int(r_max), auto_rmax)

    r = min(r, max(int(r_min), int(r_max)))
    return max(1, r)



@torch.no_grad()
def rj_perturb_degrees(
    deg: torch.Tensor,
    eps: float,
    r: int,
    n: int,
    generator=None
) -> torch.Tensor:

    assert r >= 1

    d = deg.view(-1).round().long().clamp(min=0, max=n - 1)  # [N]
    N = d.numel()

    ee = float(math.exp(float(eps)))
    p_truth = ee / (ee + 2 * r)


    u = torch.rand(N, device=d.device, generator=generator)
    keep = (u < p_truth)


    k = torch.randint(low=0, high=2 * r, size=(N,), device=d.device, generator=generator)
    offset = k - r
    offset = torch.where(offset >= 0, offset + 1, offset)
    d_jump = (d + offset).clamp(min=0, max=n - 1)

    tilde_deg = torch.where(keep, d, d_jump)
    return tilde_deg.long()


@torch.no_grad()
def unbias_degree_distribution_from_rj(
    tilde_deg: torch.Tensor,  # [N] long
    eps: float,
    r: int,
    d_max: int
) -> torch.Tensor:

    N = tilde_deg.numel()

    fo = torch.bincount(tilde_deg.clamp(0, d_max), minlength=d_max + 1).float() / max(N, 1)

    ee = float(math.exp(float(eps)))
    a = (ee + 2 * r) / (ee - 1)
    b = 1.0 / (ee - 1)
    f_hat = a * fo - b

    f_hat = torch.clamp(f_hat, min=0.0)
    s = f_hat.sum()
    if s <= 0:
        f_hat = fo
        s = f_hat.sum().clamp(min=1e-12)

    f_hat = f_hat / s
    return f_hat



@torch.no_grad()
def sample_target_degrees(
    f_hat: torch.Tensor,    # [d_max+1]
    n_nodes: int,
    n: int,
    min_deg: int = 0,
    max_deg: int = None,
    generator=None
):

    d_max = f_hat.numel() - 1
    if max_deg is None:
        max_deg = min(d_max, n - 1)

    max_deg = int(max_deg)
    min_deg = int(min_deg)

    probs = f_hat.clone()
    mask = torch.ones_like(probs, dtype=torch.bool)
    mask[:min_deg] = False
    mask[max_deg + 1:] = False
    probs = probs * mask.float()
    probs = probs / probs.sum().clamp(min=1e-12)


    d_target = torch.multinomial(probs, num_samples=n_nodes, replacement=True, generator=generator).long()


    s = int(d_target.sum().item())
    if s % 2 == 1:
        i = 0
        if d_target[i] < max_deg:
            d_target[i] += 1
        elif d_target[i] > min_deg:
            d_target[i] -= 1

    m = int(d_target.sum().item() // 2)
    return d_target, m



@torch.no_grad()
def build_feature_similarity_fn(features: torch.Tensor):

    X = F.normalize(features, p=2, dim=1)

    def sim(i: torch.Tensor, j: torch.Tensor):
        s = (X[i] * X[j]).sum(dim=1)   # [-1,1]
        s = (s + 1.0) * 0.5           # -> [0,1]
        return s.clamp(0.0, 1.0)

    return sim



@torch.no_grad()
def chung_lu_accept_reject_adj(
    d_target: torch.Tensor,
    m: int,
    features: torch.Tensor,
    alpha_rate: float = 1.0,
    max_trials_multiplier: int = 50,
    generator=None,
    weighted: bool = False
) -> torch.Tensor:

    device = d_target.device
    N = d_target.numel()

    adj = torch.zeros((N, N), dtype=torch.float, device=device)

    if m <= 0:
        return adj


    w = d_target.float().clamp(min=0.0)
    if w.sum() <= 0:
        p = torch.ones(N, device=device) / N
    else:
        p = w / w.sum()

    sim_fn = build_feature_similarity_fn(features)


    edges_set = set()

    max_trials = max(10_000, max_trials_multiplier * m)
    accepted = 0
    trials = 0

    while accepted < m and trials < max_trials:
        trials += 1


        i = torch.multinomial(p, 1, replacement=True, generator=generator).item()
        j = torch.multinomial(p, 1, replacement=True, generator=generator).item()
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        key = a * N + b
        if key in edges_set:
            continue


        if alpha_rate <= 0:
            acc = 1.0
        else:
            s = sim_fn(torch.tensor([a], device=device),
                       torch.tensor([b], device=device))[0].item()  # sim in [0,1]
            acc = (1.0 - float(alpha_rate)) + float(alpha_rate) * float(s)
            acc = max(0.0, min(1.0, acc))


        u = torch.rand(1, device=device, generator=generator).item()
        if u <= acc:
            edges_set.add(key)
            accepted += 1


            val = float(acc) if weighted else 1.0
            adj[a, b] = val
            adj[b, a] = val


    adj.fill_diagonal_(0.0)
    return adj















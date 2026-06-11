
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
    adj_tensor = torch.from_numpy(adj_dense).float()




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

    edge_index, edge_weight = dense_to_sparse_with_weights(adj_tensor)


    return edge_index, edge_weight, features, labels_truth, idx_train, idx_val, idx_test










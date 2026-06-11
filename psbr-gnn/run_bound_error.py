from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import statistics
import torch
import torch.nn.functional as F
import torch.optim as optim


from utils_24 import load_dataset,load_bound_error
from models import GCN
from gcn import GCN
from mlp import MLP
from gat import GAT
from gin import GIN
from graphsage import GraphSAGE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
import random
from collections import Counter, OrderedDict


np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float,nargs='+',  default=0.01,   # 1e-1,1e-2,0.01
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, nargs='+', default=0.0001,   #5e-4,1e-3,1e-4,1e-5,0,0.0001
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float,nargs='+',  default=0.5,   #0,  1e-1,1e-2, 1e-3,0.5
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=64,  # ✅
                    help='Number of hidden units.')
parser.add_argument('--epsilon', type=float, nargs='+', default=[88],  #1,2,3,4,5,6,7,8
                    help='epsilon.')
parser.add_argument('--delta', type=float, nargs='+', default=[0.1],  #0.01,0.05,0.1
                    help='delta of epsilon .')
parser.add_argument('--communities', type=float, nargs='+', default=[0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5],  #0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5
                    help='number of communities .')
parser.add_argument('--alpha', type=float, nargs='+', default=[0.0],  #0.0,0.2,0.4,0.6,0.8,1.0
                    help='alpha of feature similarity.')
parser.add_argument('--sigma_degree', type=float, nargs='+', default=[0.0],  #0.0,1.0
                    help='sigma of chung-du degree.')
parser.add_argument('--dataset', type=str, nargs='+', default=[ "citeseer"],  #,"facebook","twitch","amazon","wiki"
                    help='Dataset name ( "cora", "citeseer",  "facebook", "DBLP", "CS", "Physics")')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def run_bound_error(epsilon,delta, communities,dataset,sigma_degree):
      load_bound_error( dataset_name=dataset,
                          eps_edge=epsilon,
                           delta_eps=delta,
                          n_communities=communities,
                          sigma=sigma_degree)


for data in args.dataset:
    for epsilon in args.epsilon:
        for delta in args.delta:
            for n in args.communities:
                for d in args.sigma_degree:
                    result = run_bound_error(epsilon, delta, n, data, d)

from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import statistics
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import copy


from utils_Lpgnet import load_dataset
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
parser.add_argument('--lr', type=float,nargs='+',  default=0.001,   # 1e-1,1e-2,0.01
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, nargs='+', default=0.0001,   #5e-4,1e-3,1e-4,1e-5,0,0.0001
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float,nargs='+',  default=0.5,   #0,  1e-1,1e-2, 1e-3,0.5
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=16,  # ✅
                    help='Number of hidden units.')
parser.add_argument('--epsilon', type=float, nargs='+', default=[1,2,3,4,5,6,7,8],  #1,2,3,4,5,6,7,8
                    help='epsilon.')
parser.add_argument('--delta', type=float, nargs='+', default=[0],  #0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
                    help='delta of epsilon .')
parser.add_argument('--layers', type=float, nargs='+', default=[2],  #1,50,100,200,300,500,700,900
                    help='number of communities .')
parser.add_argument('--alpha', type=float, nargs='+', default=[0.0],  #0.0,0.2,0.4,0.6,0.8,1.0
                    help='alpha.')
parser.add_argument('--sigma_degree', type=float, nargs='+', default=[0.0],  #0.0,1.0
                    help='sigma of chung-du degree.')
parser.add_argument('--dataset', type=str, nargs='+', default=["cora", "citeseer", "facebook", "DBLP","CS", "Physics"],  #,"facebook","twitch","amazon","wiki"
                    help='Dataset name ( "cora", "citeseer", "facebook", "DBLP","CS", "Physics")')

'''  dropout, hidden
cora	citeseer	lastfm	facebook	DBLP	CS	     Physics
0.1，16	 0.1,64	   0.1，16   0.5,16	   0.5,16	0.1，16	  0.5,16
'''


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def make_undirected(edge_index: torch.Tensor):
    row, col = edge_index
    rev = torch.stack([col, row], dim=0)
    ei = torch.cat([edge_index, rev], dim=1)
    ei = torch.unique(ei, dim=1)
    mask = ei[0] != ei[1]
    return ei[:, mask]


def find_degree_vec(edge_index: torch.Tensor,
                    logits: torch.Tensor,
                    eps_query: float,
                    num_classes: int,
                    num_nodes: int):
    """
    cluster = argmax(softmax(logits))
    X[v,c] = #neighbors(v) in cluster c + Lap(0, 2/eps_query)
    """

    cluster_id = logits.softmax(dim=1).argmax(dim=1)  # [N]
    onehot = F.one_hot(cluster_id, num_classes=num_classes).float()  # [N,C]

    src, dst = edge_index[0], edge_index[1]
    X = torch.zeros((num_nodes, num_classes), device=logits.device)
    X.index_add_(0, src, onehot[dst])  # X[src] += onehot[dst]

    scale = 1.0 / float(eps_query)
    noise = torch.distributions.Laplace(0.0, scale).sample(X.shape).to(X.device)
    return X + noise



class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)     # hidden 1
        self.fc2 = nn.Linear(hid_dim, hid_dim)    # hidden 2
        self.fc3 = nn.Linear(hid_dim, out_dim)    # output
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dp(F.relu(self.fc1(x)))
        x = self.dp(F.relu(self.fc2(x)))
        return self.fc3(x)  # logits


def run(epsilon, delta, layers, alpha, dataset, sigma_degree):
    edge_index, edge_weight, features, labels_truth, idx_train, idx_val, idx_test = load_dataset(
        dataset_name=dataset,
        eps_edge=epsilon,
        delta_eps=delta,
        use_topk=True,
        topk_sym='None',
        n_communities=layers,
        alpha_rate=alpha,
        sigma=sigma_degree
    )

    num_classes = int(labels_truth.max().item() + 1)
    print("num_classes:", num_classes)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    features = features.to(device)
    labels_truth = labels_truth.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    edge_index = make_undirected(edge_index).to(device)

    nl = int(layers)
    nl = max(nl, 0)


    eps_query = float(epsilon) / float(nl) if nl > 0 else float(epsilon)
    print("eps_query:",eps_query)


    def train_mlp(model, X_in):
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        eval_T = 5
        P = 10
        bad = 0
        best_val = 1e18
        best_state = None

        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()

            logits = model(X_in)
            loss_train = F.cross_entropy(logits[idx_train], labels_truth[idx_train])
            acc_train = (logits[idx_train].argmax(dim=1) == labels_truth[idx_train]).float().mean().item()

            loss_train.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits_val = model(X_in)
                loss_val = F.cross_entropy(logits_val[idx_val], labels_truth[idx_val]).item()
                acc_val = (logits_val[idx_val].argmax(dim=1) == labels_truth[idx_val]).float().mean().item()

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train),
                  'loss_val: {:.4f}'.format(loss_val),
                  'acc_val: {:.4f}'.format(acc_val),
                  'time: {:.4f}s'.format(time.time() - t))

            if epoch % eval_T == 0:
                if loss_val < best_val:
                    best_val = loss_val
                    best_state = copy.deepcopy(model.state_dict())
                    bad = 0
                else:
                    bad += 1
                if bad > P:
                    print("Early Stopping! Epoch:", epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model


    M0 = SimpleMLP(in_dim=features.size(1),
                   hid_dim=args.hidden,
                   out_dim=num_classes,
                   dropout=args.dropout).to(device)

    F_list = []
    L_list = []
    X_cache = {}

    F0 = features
    F_list.append(F0)

    M0 = train_mlp(M0, F0)
    models = [M0]


    for i in range(nl):
        models[i].eval()
        with torch.no_grad():
            Li = models[i](F_list[i])   # L_i = M[i](F_i)
        L_list.append(Li)



        Xi = find_degree_vec(edge_index=edge_index,
                             logits=Li,
                             eps_query=eps_query,
                             num_classes=num_classes,
                             num_nodes=features.size(0))
        X_cache[i] = Xi

        if i == 0:
            F_next = torch.cat([Li, Xi], dim=1)
        else:
            F_next = torch.cat([F_list[i], Li, Xi], dim=1)

        F_list.append(F_next)

        Mi1 = SimpleMLP(in_dim=F_next.size(1),
                        hid_dim=args.hidden,
                        out_dim=num_classes,
                        dropout=args.dropout).to(device)
        Mi1 = train_mlp(Mi1, F_next)
        models.append(Mi1)


    F_final = F_list[nl]
    model_last = models[nl]
    model_last.eval()
    with torch.no_grad():
        logits_test = model_last(F_final)
        pred_test = logits_test[idx_test].argmax(dim=1)
        acc_test = (pred_test == labels_truth[idx_test]).float().mean().item()
        loss_test = F.cross_entropy(logits_test[idx_test], labels_truth[idx_test]).item()

    print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          "accuracy= {:.4f}".format(acc_test))

    return acc_test


with open('output_results_Lpgnet.txt', 'a') as f:
    print("******************************************************************************", file=f)




for data in args.dataset:
    for epsilon in args.epsilon:
        for delta in args.delta:
            for n in args.layers:
                all_results = []
                max_result = None
                if epsilon > 5:
                    sigma_candidates = [0]
                else:
                    sigma_candidates = args.sigma_degree
                for d in sigma_candidates:
                    for a in args.alpha:

                        results_all = []
                        for _ in range(5):
                            result = run(epsilon, delta, n, a, data, d)
                            results_all.append(result)

                            print(results_all)
                            regular_list = [tensor.cpu().item() for tensor in results_all]
                            print("list:", regular_list)
                            average = statistics.mean(regular_list)
                            std_dev = statistics.stdev(regular_list)

                            current_result = {
                                'alpha': a,
                                'sigma': d,
                                'average': average,
                                'std_dev': std_dev,
                                'all_values': regular_list
                            }
                            all_results.append(current_result)

                            if max_result is None or average > max_result['average']:
                                max_result = current_result

                            print(
                                f" dataset: {data},|epsilon: {epsilon},|delta: {delta},|communities: {n},|alpha: {a},|sigma: {d},|平均值+标准差: {round(average * 100, 2)} +- {round(std_dev * 100, 2)}")

                            with open('output_results_Asgldp.txt', 'a') as f:
                                print(
                                    f" dataset: {data},|epsilon: {epsilon},|delta: {delta},|communities: {n},|alpha: {a},|sigma: {d},|平均值+标准差: {round(average * 100, 2)} +- {round(std_dev * 100, 2)}",
                                    file=f)

                    print("\n=== alpha ===")
                    for result in all_results:
                        print(
                            f"alpha: {result['alpha']}, 平均值: {round(result['average'] * 100, 2)}, 标准差: {round(result['std_dev'] * 100, 2)}")

                    max_result_alpha_0 = None

                    for result in all_results:

                        if result['alpha'] == 0:

                            if max_result_alpha_0 is None or result['average'] > max_result_alpha_0['average']:
                                max_result_alpha_0 = result

                    print(f"\n=== global Alpha=0 best result ===")
                    if max_result_alpha_0:
                        msg_alpha_0 = f"best result (alpha=0): {round(max_result_alpha_0['average'] * 100, 2)} +- {round(max_result_alpha_0['std_dev'] * 100, 2)}"
                        print(msg_alpha_0)
                    else:
                        msg_alpha_0 = "don't found alpha=0 "
                        print(msg_alpha_0)

                    print(f"\n=== global Alpha=0 best result ===")
                    print(
                        f"best alpha: {max_result['alpha']}, max: {round(max_result['average'] * 100, 2)} +- {round(max_result['std_dev'] * 100, 2)}")

                    with open('output_results_Asgldp.txt', 'a') as f:
                        print(f"=== best result ===", file=f)
                        print(msg_alpha_0, file=f)
                        print(
                            f"best alpha: {max_result['alpha']}, max: {round(max_result['average'] * 100, 2)} +- {round(max_result['std_dev'] * 100, 2)}",
                            file=f)
                        print(f"-------------------------------------------------------------------------", file=f)
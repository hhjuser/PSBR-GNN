# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import statistics
import torch
import torch.nn.functional as F
import torch.optim as optim

# 引入 PyG 的数据封装和采样器
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from utils_RR import load_dataset
# from models import GCN
# 如果你有其他模型文件，请确保它们能接受 batch 输入
from gcn import GCN
# from mlp import MLP
# from gat import GAT
# from gin import GIN
# from graphsage import GraphSAGE

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
import random
from collections import Counter, OrderedDict

# 增加显存分配设置（可选，防止碎片化）
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

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
parser.add_argument('--lr', type=float, nargs='+', default=[0.01],
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, nargs='+', default=[0.0001],
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, nargs='+', default=[0.5],
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--epsilon', type=float, nargs='+', default=[7,8],  #1, 2, 3, 4, 5, 6
                    help='epsilon.')
parser.add_argument('--delta', type=float, nargs='+', default=[0],
                    help='delta of epsilon .')
parser.add_argument('--communities', type=float, nargs='+', default=[0],
                    help='number of communities .')
parser.add_argument('--alpha', type=float, nargs='+', default=[0.0],
                    help='alpha.')
parser.add_argument('--sigma_degree', type=float, nargs='+', default=[0.0],
                    help='sigma of chung-du degree.')
parser.add_argument('--dataset', type=str, nargs='+',
                    default=[ "Physics"],
                    help='"facebook", "DBLP", "CS",')

# 新增 Batch 相关的参数
parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for mini-batch training')
parser.add_argument('--num_neighbors', type=str, default='10,10',
                    help='Comma separated sampling size, e.g., 10,10 for 2 layers')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


try:
    num_neighbors = [int(x) for x in args.num_neighbors.split(',')]
except:
    num_neighbors = [10, 10]


def run(epsilon, delta, communities, alpha, dataset, sigma_degree):
    print(f"Loading {dataset} with eps={epsilon}...")

    # 1. Load data (Original)
    edge_index, edge_weight, features, labels_truth, idx_train, idx_val, idx_test = load_dataset(
        dataset_name=dataset,
        eps_edge=epsilon,
        delta_eps=delta,
        use_topk=True,
        topk_sym='None',
        n_communities=communities,
        alpha_rate=alpha,
        sigma=sigma_degree
    )

    print("num_classes:", labels_truth.max().item() + 1)

    data = Data(
        x=features.cpu(),
        edge_index=edge_index.cpu(),
        edge_attr=edge_weight.cpu(),
        y=labels_truth.cpu()
    )

    data.num_nodes = features.shape[0]


    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        input_nodes=idx_train.cpu(),
        shuffle=True,
        num_workers=0
    )


    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=args.batch_size * 2,
        input_nodes=idx_val.cpu(),
        shuffle=False,
        num_workers=0
    )


    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=args.batch_size * 2,
        input_nodes=idx_test.cpu(),
        shuffle=False,
        num_workers=0
    )


    lr_val = args.lr[0] if isinstance(args.lr, list) else args.lr
    wd_val = args.weight_decay[0] if isinstance(args.weight_decay, list) else args.weight_decay
    dropout_val = args.dropout[0] if isinstance(args.dropout, list) else args.dropout

    model = GCN(num_features=features.shape[1],
                hidden_channels=args.hidden,
                num_classes=labels_truth.max().item() + 1,
                dropout_p=dropout_val).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr_val, weight_decay=wd_val)

    # ==========================================
    # Batch Training Function
    # ==========================================
    def train_batch(epoch):
        t = time.time()
        model.train()

        total_loss = 0
        total_correct = 0
        total_examples = 0


        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()


            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                output = model(batch.x, batch.edge_index, batch.edge_attr)
            else:
                output = model(batch.x, batch.edge_index)


            batch_size = batch.batch_size
            out_target = output[:batch_size]
            y_target = batch.y[:batch_size]

            loss = F.nll_loss(out_target, y_target)
            loss.backward()
            optimizer.step()


            total_loss += float(loss) * batch_size
            preds = out_target.max(1)[1]
            total_correct += int((preds == y_target).sum())
            total_examples += batch_size

        avg_loss = total_loss / total_examples
        avg_acc = total_correct / total_examples


        val_loss, val_acc = evaluate_batch(val_loader)

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(avg_loss),
              'acc_train: {:.4f}'.format(avg_acc),
              'loss_val: {:.4f}'.format(val_loss),
              'acc_val: {:.4f}'.format(val_acc),
              'time: {:.4f}s'.format(time.time() - t))

        return val_acc


    @torch.no_grad()
    def evaluate_batch(loader):
        model.eval()
        total_loss = 0
        total_correct = 0
        total_examples = 0

        for batch in loader:
            batch = batch.to(device)
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                output = model(batch.x, batch.edge_index, batch.edge_attr)
            else:
                output = model(batch.x, batch.edge_index)

            batch_size = batch.batch_size
            out_target = output[:batch_size]
            y_target = batch.y[:batch_size]

            loss = F.nll_loss(out_target, y_target)

            total_loss += float(loss) * batch_size
            preds = out_target.max(1)[1]
            total_correct += int((preds == y_target).sum())
            total_examples += batch_size

        return total_loss / total_examples, total_correct / total_examples

    # Training Loop
    eval_T = 5
    P = 10
    patience_cnt = 0
    best_val_acc = 0

    t_total = time.time()
    for epoch in range(args.epochs):
        val_acc = train_batch(epoch)

        if epoch % eval_T == 0:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt > P:
                print("Early Stopping at epoch:", epoch)
                break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test_loss, test_acc = evaluate_batch(test_loader)
    print("Test set results:",
          "loss= {:.4f}".format(test_loss),
          "accuracy= {:.4f}".format(test_acc))


    return torch.tensor(test_acc)


# ==============================================================================
# Main Execution Block
# ==============================================================================

with open('output_results_RR.txt', 'a') as f:
    print("******************************************************************************", file=f)

for data in args.dataset:
    for epsilon in args.epsilon:
        for delta in args.delta:
            for n in args.communities:
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
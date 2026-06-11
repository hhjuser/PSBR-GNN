from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import statistics
import torch
import torch.nn.functional as F
import torch.optim as optim


from utils_Lapgen import load_dataset
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
parser.add_argument('--epsilon', type=float, nargs='+', default=[1,2,3,4,5,6,7,8],  #1,2,3,4,5,6,7,8
                    help='epsilon.')
parser.add_argument('--delta', type=float, nargs='+', default=[0],  #0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
                    help='delta of epsilon .')
parser.add_argument('--communities', type=float, nargs='+', default=[0],  #1,50,100,200,300,500,700,900
                    help='number of communities .')
parser.add_argument('--alpha', type=float, nargs='+', default=[0.0],  #0.0,0.2,0.4,0.6,0.8,1.0
                    help='alpha.')
parser.add_argument('--sigma_degree', type=float, nargs='+', default=[0.0],  #0.0,1.0
                    help='sigma of chung-du degree.')
parser.add_argument('--dataset', type=str, nargs='+', default=["cora", "citeseer", "facebook", "DBLP", "CS", "Physics"],  #,"facebook","twitch","amazon","wiki"
                    help='Dataset name ( "cora", "citeseer",  "facebook", "DBLP", "CS", "Physics")')



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



def run(epsilon, delta, communities,alpha,dataset,sigma_degree):
    # Load data
    "cora, citeseer, pubmed, lastfm-128, facebook-128, twitch, amazon, wiki "
    edge_index, edge_weight, features, labels_truth, idx_train, idx_val, idx_test = load_dataset(
        dataset_name=dataset,
        eps_edge=epsilon,
        delta_eps=delta,
        use_topk=True,
        topk_sym='None',
        n_communities = communities,
        alpha_rate=alpha,
        sigma = sigma_degree
    )


    print("num_classes:", labels_truth.max().item() + 1)
    # Model and optimizer
    model = GCN(num_features=features.shape[1],
                hidden_channels=args.hidden,
                num_classes=labels_truth.max().item() + 1,
                dropout_p=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        # adj = adj.cuda()
        edge_index = edge_index.cuda()
        edge_weight = edge_weight.cuda()
        labels_truth = labels_truth.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


    def train(epoch):
        t = time.time()


        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index, edge_weight)
        loss_train = F.nll_loss(output[idx_train], labels_truth[idx_train])
        acc_train = accuracy(output[idx_train], labels_truth[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, edge_index, edge_weight)

        loss_val = F.nll_loss(output[idx_val], labels_truth[idx_val])
        acc_val = accuracy(output[idx_val], labels_truth[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        return loss_val, output


    def end_result():
        model.eval()
        output = model(features, edge_index, edge_weight)
        # labels_prediction = torch.argmax(output, dim=1)
        # print("labels_prediction[idx_val]",labels_prediction[idx_val])
        loss_test = F.nll_loss(output[idx_test], labels_truth[idx_test])
        acc_test = accuracy(output[idx_test], labels_truth[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test, output[idx_test], labels_truth[idx_test]


    eval_T = 5  # evaluate period
    P = 10  # patience
    i = 0  # record the frequency of bad performance of validation
    temp_val_loss = 99999  # initialize val loss

    for epoch in range(args.epochs):
        result, output= train(epoch)
        # early stopping
        if (epoch % eval_T) == 0:
            if temp_val_loss > result:
                temp_val_loss = result
                # torch.save(model.state_dict(), "GCN_NET3.pth")  # save the current best
                i = 0  # reset i
            else:
                i = i + 1
        if i > P:
            print("Early Stopping! Epoch1 : ", epoch )
            break


    # Train model
    t_total = time.time()
    # for epoch in range(args.epochs):
    #     train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test_acc, result, test_label = end_result()

    return test_acc


with open('output_results_Lapgen.txt', 'a') as f:
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
                    # 低预算时，遍历参数里指定的所有 d
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
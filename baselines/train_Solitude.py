from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy


from utils_Solitude import load_dataset
from dense_gcn import DenseGCN


class ProxOperators():
    """L1 范数的近端算子 (用于稀疏化)"""

    def prox_l1(self, data, alpha):

        data = torch.mul(torch.sign(data), torch.clamp(torch.abs(data) - alpha, min=0))
        return data


class PGD(optim.Optimizer):


    def __init__(self, params, proxs, alphas, lr, momentum=0, dampening=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        super(PGD, self).__init__(params, defaults)
        for group in self.param_groups:
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def step(self, delta=0, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            proxs = group['proxs']
            alphas = group['alphas']
            # 对每个参数应用近端算子
            for param in group['params']:
                for prox_operator, alpha in zip(proxs, alphas):
                    param.data = prox_operator(param.data, alpha=alpha * lr)


prox_operators = ProxOperators()

class EstimateAdj(nn.Module):
    def __init__(self, adj, symmetric=False, device='cuda'):
        super(EstimateAdj, self).__init__()
        self.device = device
        self.symmetric = symmetric

        n = len(adj)

        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)

        self.register_buffer('initial_adj', adj.clone())

    def _init_estimation(self, adj):
        with torch.no_grad():
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        adj = self.estimated_adj

        if self.symmetric:
            adj = (adj + adj.t()) / 2


        adj_with_loop = adj + torch.eye(adj.shape[0]).to(self.device)


        rowsum = adj_with_loop.sum(1) + 1e-12
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)

        mx = r_mat_inv @ adj_with_loop
        mx = mx @ r_mat_inv
        return mx



np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='GCN Learning rate.')
parser.add_argument('--lr_adj', type=float, default=0.01, help='Adjacency Learning rate.')  # 新增
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
# 超参数
parser.add_argument('--epsilon', type=float, nargs='+', default=[8], help='epsilon.')
parser.add_argument('--delta', type=float, nargs='+', default=[0], help='delta.')
parser.add_argument('--communities', type=float, nargs='+', default=[0], help='communities.')
parser.add_argument('--alpha', type=float, nargs='+', default=[0.0], help='alpha.')
parser.add_argument('--sigma_degree', type=float, nargs='+', default=[0.0], help='sigma.')
parser.add_argument('--dataset', type=str, nargs='+', default=["facebook"], help='Dataset name. "cora", "citeseer", "lastfm-128", "facebook", "DBLP", "CS", "Physics"')
# 正则化权重
parser.add_argument('--lambda_fro', type=float, default=1e-3, help='Fidelity Loss coefficient')
parser.add_argument('--lambda_l1', type=float, default=1e-3, help='Sparsity Loss coefficient')

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



def run(epsilon, delta, communities, alpha, dataset, sigma_degree):
    # Load data
    perturbed_adj, features, labels_truth, idx_train, idx_val, idx_test = load_dataset(
        dataset_name=dataset,
        eps_edge=epsilon,
        delta_eps=delta,
        use_topk=True,
        topk_sym='None',
        n_communities=communities,
        alpha_rate=alpha,
        sigma=sigma_degree
    )

    device = torch.device("cuda" if args.cuda else "cpu")


    if perturbed_adj.is_sparse:
        perturbed_adj = perturbed_adj.to_dense()

    estimator = EstimateAdj(perturbed_adj, symmetric=True, device=device).to(device)


    num_classes = labels_truth.max().item() + 1
    model = DenseGCN(num_features=features.shape[1],
                     hidden_channels=args.hidden,
                     num_classes=num_classes,
                     dropout_p=args.dropout).to(device)

    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    optimizer_adj = optim.SGD(estimator.parameters(), momentum=0.9, lr=args.lr_adj)


    optimizer_l1 = PGD(estimator.parameters(), proxs=[prox_operators.prox_l1],
                       lr=args.lr_adj, alphas=[args.lambda_l1])

    if args.cuda:
        features = features.cuda()
        labels_truth = labels_truth.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    def train(epoch):
        t = time.time()


        model.train()
        estimator.eval()
        optimizer_model.zero_grad()

        with torch.no_grad():
            # norm_adj = estimator.normalize().detach()
            norm_adj = estimator()

        output = model(features, norm_adj)

        loss_train = F.cross_entropy(output[idx_train], labels_truth[idx_train])
        acc_train = accuracy(output[idx_train], labels_truth[idx_train])

        loss_train.backward()
        optimizer_model.step()

        model.eval()
        estimator.train()
        optimizer_adj.zero_grad()
        optimizer_l1.zero_grad()


        norm_adj_grad = estimator()


        output = model(features, norm_adj_grad)


        loss_gnn = F.cross_entropy(output[idx_train], labels_truth[idx_train])


        loss_fro = torch.norm(estimator.estimated_adj - estimator.initial_adj, p='fro')


        loss_diffiential = loss_fro * args.lambda_fro + loss_gnn

        loss_diffiential.backward()


        optimizer_adj.step()
        optimizer_l1.step()


        with torch.no_grad():
            estimator.estimated_adj.data.clamp_(min=0, max=1)

        # --------------------------------------
        # Validation
        # --------------------------------------
        if not args.fastmode:
            model.eval()
            with torch.no_grad():
                # val_adj = estimator.normalize()
                val_adj = estimator()
                output = model(features, val_adj)
                loss_val = F.cross_entropy(output[idx_val], labels_truth[idx_val])
                acc_val = accuracy(output[idx_val], labels_truth[idx_val])

        # 打印日志
        print('Epoch: {:04d}'.format(epoch+1),
              'Loss_GNN: {:.4f}'.format(loss_gnn.item()),
              'Loss_Fro: {:.4f}'.format(loss_fro.item()),
              'Acc_Train: {:.4f}'.format(acc_train.item()),
              'Acc_Val: {:.4f}'.format(acc_val.item()),
              'Time: {:.4f}s'.format(time.time() - t))

        return loss_val.item(), acc_val.item()


    def end_result():
        model.eval()
        with torch.no_grad():
            # test_adj = estimator.normalize()
            test_adj = estimator()
            output = model(features, test_adj)
            # loss_test = F.cross_entropy(output[idx_test], labels_truth[idx_test])
            acc_test = accuracy(output[idx_test], labels_truth[idx_test])

        print("Test set results:", "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    eval_T = 5
    P = 3
    bad_counter = 0
    best_val_loss = float('inf')
    best_model_wts = None

    for epoch in range(args.epochs):
        val_loss, val_acc = train(epoch)

        # Early Stopping
        if epoch % eval_T == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_counter = 0
                best_model_wts = deepcopy(model.state_dict())
            else:
                bad_counter += 1

            if bad_counter >= P:
                print(f"Early Stopping at epoch {epoch}")
                break


    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # print("Optimization Finished!")
    test_acc = end_result()
    return test_acc



with open('output_results_Solitude.txt', 'a') as f:
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
                        print(f"Running: {data} | Eps: {epsilon} | Sigma: {d} | Alpha: {a}")

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
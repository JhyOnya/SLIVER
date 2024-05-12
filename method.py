import time
from collections import defaultdict

import GPUtil as gpu_util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim


# set random seeds:
def setup_seed(seed=1000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_models(stat_params, cache_dir):
    stat_params_state_dict = {name: param_pre.state_dict() for name, param_pre in stat_params.items()}
    torch.save(stat_params_state_dict, "%sbest_model.pth" % (cache_dir))


def restore_models(stat_params, cache_dir):
    checkpoint = torch.load("%sbest_model.pth" % (cache_dir))
    for name, param_pre in stat_params.items():
        param_pre.load_state_dict(checkpoint[name])
        param_pre.eval()


class p_A_x_func(nn.Module):
    def __init__(self, dim_feat, tf_len, dim_RL):
        super(p_A_x_func, self).__init__()
        self.dim_RL = dim_RL
        self.dim_feat = dim_feat
        self.tf_len = tf_len

        self.L = torch.nn.Parameter(torch.rand((dim_feat, dim_RL), requires_grad=True))
        self.R = torch.nn.Parameter(torch.rand((dim_RL, dim_feat), requires_grad=True))
        self.E = torch.nn.Parameter(torch.rand((1, dim_feat), requires_grad=True))

    def forward(self, x):
        A = torch.softmax(self.L, dim=1) @ torch.softmax(self.R, dim=0)
        M = torch.softmax(self.R, dim=0) @ torch.softmax(self.L, dim=1)

        return x @ A + self.E, A, M, \
               torch.softmax(self.L, dim=1), torch.softmax(self.R, dim=0)


def train(data, args, device, max_comp=100, TF_ids_list=None):
    setup_seed()

    def h_dag(S):
        d = S.shape[0]
        A = S * S / (d * d)
        D = torch.matrix_power(A, d)

        return torch.trace(D).sum()

    print("data.shape", data.shape)
    n = data.shape[0]
    d = data.shape[1]
    tf_len = len(TF_ids_list)
    args.batch = 8
    data_np_ori = data.values
    dt_min = np.min(data_np_ori, axis=0)
    dt_max = np.max(data_np_ori, axis=0)
    data_np = (data_np_ori - dt_min) / (dt_max - dt_min) + 1e-8

    gpu_info_initial = gpu_util.getGPUs()[0]
    gpu_start = gpu_info_initial.memoryUsed

    p_A_x_dist = p_A_x_func(dim_feat=d, tf_len=tf_len, dim_RL=args.nodes)
    p_A_x_dist.to(device, non_blocking=True)

    stat_params = {'p_A_x': p_A_x_dist, }
    params = [pre for v in stat_params.values() for pre in list(v.parameters())]

    loss_MSE = torch.nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-4)

    n_iter_batch, idx = int(n / args.batch), list(range(n))

    loss_list = defaultdict(list)
    best_l_epoch = np.inf

    epoch = 0
    pre_comp = 0

    lambda_ = 0

    pho = 1e-16

    l_dag_last = np.inf
    epoch_time = []

    while True:
        if pre_comp >= max_comp and epoch >= 500:
            break
        epoch += 1
        np.random.shuffle(idx)
        p_A_x_dist.train()

        start_time = time.time()
        for batch in range(n_iter_batch):
            id_batch = np.random.choice(idx, args.batch, replace=False)
            data_batch = torch.tensor(data_np[id_batch], dtype=torch.float32, requires_grad=False, device=device)
            data_pred, A, M, L, R = p_A_x_dist(data_batch)

            l_A = loss_MSE(data_batch, data_pred)
            l_dag = h_dag(M)

            loss = torch.sum(l_A) + 0.01 * torch.sum(M) \
                   + 0.001 * torch.sum(L.T @ L) + 0.001 * torch.sum(R @ R.T) \
                   + lambda_ * l_dag + 0.5 * pho * l_dag * l_dag

            with torch.no_grad():
                loss_list['l'].append(float(loss))
                loss_list['l_A'].append(float(torch.sum(l_A)))
                loss_list['l_dag'].append(float(l_dag))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch == 1:
            gpu_info_final = gpu_util.getGPUs()[0]
            gpu_end = gpu_info_final.memoryUsed
            print(f"ori/final GPU ：{gpu_start:.3f}MB / {gpu_end:.3f}MB / {gpu_info_final.memoryTotal:.3f}MB")
            print(f"pre GPU：{(gpu_end - gpu_start):.3f}MB")

        end_time = time.time()
        epoch_time.append(end_time - start_time)
        p_A_x_dist.eval()
        l_pre_epoch = np.mean(loss_list['l'][-n_iter_batch:])
        l_A_pre_epoch = np.mean(loss_list['l_A'][-n_iter_batch:])
        l_dag_pre_epoch = np.mean(loss_list['l_dag'][-n_iter_batch:])

        if l_dag_pre_epoch >= 0.8 * l_dag_last and pho < 1e+16 and l_A_pre_epoch / 10 > l_dag_pre_epoch:
            lambda_ += pho * l_dag_pre_epoch
            pho *= 10

        l_dag_last = l_dag_pre_epoch

        if l_A_pre_epoch <= best_l_epoch:
            pre_comp = 0
            best_l_epoch = l_A_pre_epoch
            save_models(stat_params, args.cache_dir)
            print("update best model:", epoch, best_l_epoch)
        else:
            pre_comp = pre_comp + 1

        print("epoch: %4s  l=%-10f l_A=%-10f l_dag=%-10f "
              "pho=%-.2e lambda=%-.2e" % (epoch,
                                          l_pre_epoch, l_A_pre_epoch,
                                          l_dag_pre_epoch,
                                          pho, lambda_,
                                          ))
    print(f"each epoch seconds：{np.mean(epoch_time) / len(epoch_time)}")

    restore_models(stat_params, args.cache_dir)
    p_A_x_dist.eval()
    data_pred, A, M, L, R = p_A_x_dist(data_batch)
    L = L.cpu().detach().numpy()
    R = R.cpu().detach().numpy()

    G = A.cpu().detach().numpy() * (1 - np.eye(d))
    G[len(TF_ids_list):, :] = 0
    G_pd = pd.DataFrame(G, index=data.columns, columns=data.columns)
    L_pd = pd.DataFrame(L, index=data.columns)
    R_pd = pd.DataFrame(R, columns=data.columns)
    return {"adj_A": G_pd,
            "adj_L": L_pd,
            "adj_R": R_pd,
            "loss_list": loss_list}

import torch

import datasets
import method
from utils import *


def setup_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_res(res: dict, dir):
    if "loss_list" in res:
        loss_list = res["loss_list"]
        draw(X=range(len(loss_list["l_A"])), Y=loss_list["l_A"], title="l_A", dir=dir)
        draw(X=range(len(loss_list["l_dag_adv"])), Y=loss_list["l_dag_adv"], title="l_dag_adv", dir=dir)
        draw(X=range(len(loss_list["l_dag_dec"])), Y=loss_list["l_dag_dec"], title="l_dag_dec", dir=dir)

    for k, v in res.items():
        if k.startswith("adj_"):
            v.to_csv(dir + k + ".csv")
            print("save", dir + k + ".csv")
        if k.startswith("edges_"):
            v.to_csv(dir + k + ".csv", index=False)
            print("save", dir + k + ".csv")


def main(args):
    print("Read data", args.data)
    dataset = datasets.read_data_valid(args.data)
    data, gt_edges_pd, TF_ids_list = dataset.get_msg()

    print("Train")

    print("Init")
    if torch.cuda.is_available() and not args.notcuda:
        device = torch.device("cuda", 0)
    else:
        print("   cuda is not available")
        device = torch.device("cpu")

    setup_seed()
    print("Train")
    res = method.train(data, args, device, TF_ids_list=TF_ids_list)

    pred_edges_pd_ori = graph2edges(res['adj_A'])
    pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
    pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")
    res["adj_A_edges"] = pred_edges_pd_sorted

    print('Save')
    res["edges_compare_outer"] = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer")
    res["edges_compare_inner"] = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner")

    save_res(res, args.cache_dir)
    return res['adj_A']

import torch

import datasets
import evaluation as eva
from utils import *


def setup_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args, pred_adj_pd_ori):
    print('Evaluate')
    dataset = datasets.read_data_valid(args.data)
    data, gt_edges_pd, TF_ids_list = dataset.get_msg()
    scores = eva.evaluates_seq(gt_edges_pd, pred_adj_pd_ori, TF_ids_list=TF_ids_list)
    draw(X=scores["Rec"], Y=scores["Prec"], x_label="Rec", y_label="Prec", title="PR", dir=args.cache_dir)
    draw(X=scores["FPR"], Y=scores["TPR"], x_label="FPR", y_label="TPR", title="ROC", dir=args.cache_dir)

    return {
        "AUPR": scores["AUPR"],
        "AUROC": scores["AUROC"],
    }

from utils import *


def eva_method_AUROC_AUPR(edges_pd):
    results = dict()

    P = (edges_pd['gt_weight'] == 1).sum()
    N = (edges_pd['gt_weight'] == 0).sum()

    K = np.arange(1, (edges_pd.shape[0] + 1))
    tp_k = np.cumsum(edges_pd['gt_weight'].values)
    fp_k = np.cumsum(1 - edges_pd['gt_weight'].values)

    # PR
    results["Prec"] = tp_k / K
    results["Rec"] = tp_k / P
    results["AUPR"] = np.trapz(results["Prec"], results["Rec"]) / (1 - 1 / P)  # normalized by max possible value

    # ROC
    results["TPR"] = tp_k / P
    results["FPR"] = fp_k / N
    results["AUROC"] = np.trapz(results["TPR"], results["FPR"])

    return results


def probability(X, Y, x):
    X = X.squeeze(0)
    Y = Y.squeeze(0)
    tmp = (X >= x)
    P = np.sum(tmp * Y) / len(X)
    return P


def evaluates_seq(gt_edges_pd, pred_adj_pd_ori, TF_ids_list):
    res_score = dict()
    pred_edges_pd_ori = graph2edges(pred_adj_pd_ori)
    pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
    pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")

    if (gt_edges_pd["gt_weight"] == 0).any():
        edges_pd = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner")
    else:
        edges_pd = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer")
        edges_pd = edges_pd.fillna(0)

    # res_au_rdm = eva_method_AUROC_AUPR(edges_pd.sample(frac=1).reset_index(drop=True))
    res_au = eva_method_AUROC_AUPR(edges_pd)

    print("         AUPR:  %15f" % res_au["AUPR"])
    print("        AUROC:  %15f" % res_au["AUROC"])

    return dict(res_au, **res_score)

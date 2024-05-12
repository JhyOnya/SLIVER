import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd


def graph2edges(graph_pd, column=["from", "to", "pred_weight"]):
    graph = graph_pd.values
    edges_a, edges_b = np.where(graph != np.inf)
    edges_list = [graph[f, t] for (f, t) in zip(edges_a, edges_b)]
    edges_pd = pd.DataFrame({column[0]: graph_pd.index[edges_a],
                             column[1]: graph_pd.index[edges_b],
                             column[2]: edges_list
                             })
    return edges_pd


def edges2graph(edges_pd, features_ls=None):
    G_np = np.zeros((len(features_ls), len(features_ls)))
    index_f = features_ls.get_indexer(list(edges_pd.iloc[:, 0]))
    index_t = features_ls.get_indexer(list(edges_pd.iloc[:, 1]))
    G_np[index_f, index_t] = list(edges_pd.iloc[:, 2])
    return pd.DataFrame(G_np,  # dtype=int,
                        index=features_ls,
                        columns=features_ls)


def draw(X, Y, title, dir, x_label=None, y_label=None):
    plt.plot(X, Y, label=title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(dir + "pic_%s.pdf" % title)
    plt.clf()


def sort_edges(edges_pd, sort="descending"):
    if sort == "descending":
        return edges_pd.sort_values('pred_weight', ascending=False)
    elif sort == "random":
        return edges_pd.reindex(np.random.permutation(edges_pd.index))
    else:
        print("wrong sort method, do not change. sort='descending' or 'random'")
        return edges_pd


def initFile(args):
    args.cache_dir = "./cache/%s/" % args.data
    args.cache_dir = args.cache_dir

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

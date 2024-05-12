from argparse import ArgumentParser

import main
import main_analysis
from utils import *


def set_parser():
    parser = ArgumentParser()
    parser.add_argument("-notcuda", action='store_true', default=False)
    parser.add_argument('-cache_dir', type=str, default="./cache/")
    parser.add_argument('-data', type=str, default="hESC-CellType-500")
    parser.add_argument('-nodes', type=int, default=64)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_parser()
    initFile(args)
    pred_adj_pd_ori = main.main(args)
    msg_dict = main_analysis.main(args, pred_adj_pd_ori)

from utils import *


class read_data_valid(object):
    def __init__(self, data_name="hESC-NonSpecific-500"):
        self.data_name = data_name
        self.data_path = rf"./data/{data_name}_data.csv"
        self.gt_path = rf"./data/{data_name}_gt.csv"
        self.tf_path = rf"./data/{data_name}_tfs.csv"

    def get_msg(self):
        data = pd.read_csv(self.data_path, index_col=0)
        gt_edges_pd = pd.read_csv(self.gt_path)
        if os.path.exists(self.tf_path):
            tf_pd = pd.read_csv(self.tf_path, index_col=0)
            TF_ids_list = tf_pd.to_list()
        else:
            TF_ids_list = sorted(list(set(gt_edges_pd['from'])))

        print("data.shape", data.shape)
        print("TF_ids_list.shape", len(TF_ids_list))
        print("gt.shape", gt_edges_pd.shape)

        return data, gt_edges_pd, TF_ids_list


class read_data_casestudy(object):
    def __init__(self, data_name="hESC-Non_specific-500"):
        self.data_name = data_name
        self.data_path = rf"./data/{data_name}_data.csv"
        self.gt_path = rf"./data/{data_name}_gt.csv"
        self.tf_path = rf"./data/{data_name}_tfs.csv"

    def get_msg(self):
        data = pd.read_csv(self.data_path, index_col=0)
        if os.path.exists(self.tf_path):
            tf_pd = pd.read_csv(self.tf_path, index_col=0)
            TF_ids_list = tf_pd.to_list()
        else:
            TF_ids_list = data.index.to_list()

        print("data.shape", data.shape)
        print("TF_ids_list.shape", len(TF_ids_list))
        return data, None, TF_ids_list

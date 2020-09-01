import numpy as np
import random
import torch
import data_utils
import pandas as pd

seed = 343
np.random.seed(seed=seed)
random.seed(a=seed)
torch.manual_seed(seed)

class AEDataReader(object):

    #### prepare L1000 gene expression profile as input
    #### prepare essential gene expression profile or L1000 gene expression profile as output
    def __init__(self, input_file_name, label_file_name, device):
        self.input_file_name = input_file_name
        self.label_file_name = label_file_name
        self.train_feature = torch.from_numpy(np.asarray(pd.read_csv(self.input_file_name + '_train.csv', index_col=0).values, dtype=np.float64)).to(device)
        self.dev_feature = torch.from_numpy(np.asarray(pd.read_csv(self.input_file_name + '_dev.csv', index_col=0).values, dtype=np.float64)).to(device)
        self.test_feature = torch.from_numpy(np.asarray(pd.read_csv(self.input_file_name + '_test.csv', index_col=0).values, dtype=np.float64)).to(device)
        self.train_label = torch.from_numpy(np.asarray(pd.read_csv(self.label_file_name + '_train.csv', index_col=0).values, dtype=np.float64)).to(device)
        self.dev_label = torch.from_numpy(np.asarray(pd.read_csv(self.label_file_name + '_dev.csv', index_col=0).values, dtype=np.float64)).to(device)
        self.test_label = torch.from_numpy(np.asarray(pd.read_csv(self.label_file_name + '_test.csv', index_col=0).values, dtype=np.float64)).to(device)

    #### get batch data: 
    #### input: dataset: indicate whether this is train, dev or test
    ####        batch_size: indicate batch size
    ####        shuffle: inidicate whether i need to shuffle the data
    def get_batch_data(self, dataset, batch_size, shuffle):
        if dataset == 'train':
            feature = self.train_feature
            label = self.train_label
        elif dataset == 'dev':
            feature = self.dev_feature
            label = self.dev_label
        elif dataset == 'test':
            feature = self.test_feature
            label = self.test_label
        if shuffle:
            index = torch.randperm(len(feature)).long()
            index = index.numpy()
        for start_idx in range(0, len(feature), batch_size):
            if shuffle:
                excerpt = index[start_idx: start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield feature[excerpt], label[excerpt]

class DataReader(object):
    def __init__(self, drug_file, gene_file, data_file_train, data_file_dev, data_file_test,
                 filter, device):
        self.device = device
        self.drug, self.drug_dim = data_utils.read_drug_string(drug_file)
        self.gene = data_utils.read_gene(gene_file, self.device)
        feature_train, label_train = data_utils.read_data(data_file_train, filter)
        feature_dev, label_dev = data_utils.read_data(data_file_dev, filter)
        feature_test, label_test = data_utils.read_data(data_file_test, filter)
        self.train_feature, self.dev_feature, self.test_feature, self.train_label, \
        self.dev_label, self.test_label, self.use_pert_type, self.use_cell_id, self.use_pert_idose = \
            data_utils.transfrom_to_tensor(feature_train, label_train, feature_dev, label_dev,
                                           feature_test, label_test, self.drug, self.device)

    def get_batch_data(self, dataset, batch_size, shuffle):
        if dataset == 'train':
            feature = self.train_feature
            label = self.train_label
        elif dataset == 'dev':
            feature = self.dev_feature
            label = self.dev_label
        elif dataset == 'test':
            feature = self.test_feature
            label = self.test_label
        if shuffle:
            index = torch.randperm(len(feature['drug'])).long()
            index = index.numpy()
        for start_idx in range(0, len(feature['drug']), batch_size):
            if shuffle:
                excerpt = index[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            output = dict()
            output['drug'] = data_utils.convert_smile_to_feature(feature['drug'][excerpt], self.device)
            output['mask'] = data_utils.create_mask_feature(output['drug'], self.device)
            if self.use_pert_type:
                output['pert_type'] = feature['pert_type'][excerpt]
            if self.use_cell_id:
                output['cell_id'] = feature['cell_id'][excerpt]
            if self.use_pert_idose:
                output['pert_idose'] = feature['pert_idose'][excerpt]
            yield output, label[excerpt]


if __name__ == '__main__':
    filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
              "cell_id": ["A375", "HT29", "MCF7", "PC3", "HA1E", "YAPC", "HELA"],
              "pert_idose": ["0.04 um", "0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

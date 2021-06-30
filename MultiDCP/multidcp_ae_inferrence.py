import os
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
from datetime import datetime
import torch
from torch import save
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/models')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils')
import multidcp
import datareader
import metric
import wandb
import pdb
import pickle
from scheduler_lr import step_lr
from loss_utils import apply_NodeHomophily
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

USE_wandb = True
if USE_wandb:
    wandb.init(project="MultiDCP_AE_loss")
else:
    os.environ["WANDB_MODE"] = "dryrun"

start_time = datetime.now()

parser = argparse.ArgumentParser(description='MultiDCP PreTraining')
parser.add_argument('--drug_file')
parser.add_argument('--gene_file')
parser.add_argument('--dropout')
parser.add_argument('--train_file')
parser.add_argument('--dev_file')
parser.add_argument('--test_file')
parser.add_argument('--batch_size')
parser.add_argument('--max_epoch')
parser.add_argument('--unfreeze_steps', help='The epochs at which each layer is unfrozen, like <<1,2,3,4>>')
parser.add_argument('--ae_input_file')
parser.add_argument('--ae_label_file')
parser.add_argument('--predicted_result_for_testset', help = "the file directory to save the predicted test dataframe")
parser.add_argument('--hidden_repr_result_for_testset', help = "the file directory to save the test data hidden representation dataframe")
parser.add_argument('--cell_ge_file', help='the file which used to map cell line to gene expression file')
parser.add_argument('--all_cells')
parser.add_argument('--linear_only', dest = 'linear_only', action='store_true', default=False,
                    help = 'whether the cell embedding layer only have linear layers')

args = parser.parse_args()

drug_file = args.drug_file
gene_file = args.gene_file
dropout = float(args.dropout)
gene_expression_file_train = args.train_file
gene_expression_file_dev = args.dev_file
gene_expression_file_test = args.test_file
batch_size = int(args.batch_size)
max_epoch = int(args.max_epoch)
unfreeze_steps = args.unfreeze_steps.split(',')
assert len(unfreeze_steps) == 4, "number of unfreeze steps should be 4"
unfreeze_pattern = [False, False, False, False]
ae_input_file = args.ae_input_file
ae_label_file = args.ae_label_file
predicted_result_for_testset = args.predicted_result_for_testset
hidden_repr_result_for_testset = args.hidden_repr_result_for_testset
cell_ge_file = args.cell_ge_file
all_cells = list(pickle.load(open(args.all_cells, 'rb')))
linear_only = args.linear_only
print('--------------linear: {0!r}--------------'.format(linear_only))

# parameters initialization
drug_input_dim = {'atom': 62, 'bond': 6}
drug_embed_dim = 128
drug_target_embed_dim = 128
conv_size = [16, 16]
degree = [0, 1, 2, 3, 4, 5]
gene_embed_dim = 128
pert_type_emb_dim = 4
cell_id_emb_dim = 32
cell_decoder_dim = 978 # autoencoder label's dimension
pert_idose_emb_dim = 4
hid_dim = 128
num_gene = 978
precision_degree = [10, 20, 50, 100]
loss_type = 'point_wise_mse' #'point_wise_mse' # 'list_wise_ndcg' #'combine'
intitializer = torch.nn.init.kaiming_uniform_
filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422','BRD-U01690642','BRD-U08759356','BRD-U25771771', 'BRD-U33728988', 'BRD-U37049823',
            'BRD-U44618005', 'BRD-U44700465','BRD-U51951544', 'BRD-U66370498','BRD-U68942961', 'BRD-U73238814',
            'BRD-U82589721','BRD-U86922168','BRD-U97083655'], 
          "pert_type": ["trt_cp"],
          "cell_id": all_cells,# ['A549', 'MCF7', 'HCC515', 'HEPG2', 'HS578T', 'PC3', 'SKBR3', 'MDAMB231', 'JURKAT', 'A375', 'BT20', 'HELA', 'HT29', 'HA1E', 'YAPC'],
          "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}
sorted_test_input = pd.read_csv(gene_expression_file_test).sort_values(['pert_id', 'pert_type', 'cell_id', 'pert_idose'])
test_ae_label_file = pd.read_csv(ae_label_file + '_test.csv', index_col=0)

# check cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())

ae_data = datareader.AEDataReader(ae_input_file, ae_label_file, device)
data = datareader.PerturbedDataLoader(drug_file, gene_file, gene_expression_file_train, gene_expression_file_dev,
                             gene_expression_file_test, filter, device, cell_ge_file, batch_size = batch_size)
data.setup()
print('#Train: %d' % len(data.train_data.feature['drug']))
print('#Dev: %d' % len(data.dev_data.feature['drug']))
print('#Test: %d' % len(data.test_data.feature['drug']))

print('#Train AE: %d' % len(ae_data.train_feature))
print('#Dev AE: %d' % len(ae_data.dev_feature))
print('#Test AE: %d' % len(ae_data.test_feature))

# model creation
model = multidcp.MultiDCP_AE(drug_input_dim=drug_input_dim, drug_emb_dim=drug_embed_dim,
                      conv_size=conv_size, degree=degree, gene_input_dim=np.shape(data.gene)[1],
                      gene_emb_dim=gene_embed_dim, num_gene=np.shape(data.gene)[0], hid_dim=hid_dim, dropout=dropout,
                      loss_type=loss_type, device=device, initializer=intitializer,
                      pert_type_input_dim=len(filter['pert_type']), cell_id_input_dim=978, 
                      pert_idose_input_dim=len(filter['pert_idose']), pert_type_emb_dim=pert_type_emb_dim,
                      cell_id_emb_dim=cell_id_emb_dim, cell_decoder_dim = cell_decoder_dim, pert_idose_emb_dim=pert_idose_emb_dim,
                      use_pert_type=data.use_pert_type, use_cell_id=data.use_cell_id,
                      use_pert_idose=data.use_pert_idose)
model.init_weights(pretrained = False)
model.to(device)
model = model.double()

if USE_wandb:
     wandb.watch(model, log="all")

# training
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lr_lambda=[lambda x: step_lr([int(x) for x in unfreeze_steps], x)])
best_dev_loss = float("inf")
best_dev_pearson = float("-inf")
pearson_list_ae_dev = []
pearson_list_ae_test = []
pearson_list_perturbed_dev = []
pearson_list_perturbed_test = []
spearman_list_ae_dev = []
spearman_list_ae_test = []
spearman_list_perturbed_dev = []
spearman_list_perturbed_test = []
rmse_list_ae_dev = []
rmse_list_ae_test = []
rmse_list_perturbed_dev = []
rmse_list_perturbed_test = []
precisionk_list_ae_dev = []
precisionk_list_ae_test = []
precisionk_list_perturbed_dev = []
precisionk_list_perturbed_test = []
result_df = None

# pdb.set_trace()
model.load_state_dict(torch.load('best_multidcp_ae_model.pt', map_location = device))
# pdb.set_trace()
data_save = True
epoch = 0
epoch_loss = 0
# lb_np = np.empty([0, cell_decoder_dim])
# predict_np = np.empty([0, cell_decoder_dim])
# hidden_np = np.empty([0, 50])
# with torch.no_grad():
    # for i, (feature, label, _) in enumerate(ae_data.get_batch_data(dataset='test', batch_size=batch_size, shuffle=False)):
    #     predict, hidden = model(input_drug=None, input_gene=None, mask=None, input_pert_type=None, 
    #                 input_cell_id=feature, input_pert_idose=None, job_id = 'ae', linear_only = linear_only)
    #     loss = model.loss(label, predict)
    #     epoch_loss += loss.item()
    #     lb_np = np.concatenate((lb_np, label.cpu().numpy()), axis=0)
    #     predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
    #     hidden_np = np.concatenate((hidden_np, hidden.cpu().numpy()), axis=0)

    # if data_save:
    #     hidden_df = pd.DataFrame(hidden_np, index = list(test_ae_label_file.index), columns = [x for x in range(50)])
    #     print('++++++++++++++++++++++++++++Write hidden state out++++++++++++++++++++++++++++++++')
    #     hidden_df.to_csv(hidden_repr_result_for_testset)

    # print('AE Test loss:')
    # print(epoch_loss / (i + 1))
    # if USE_wandb:
    #     wandb.log({'AE Test Loss': epoch_loss / (i + 1)}, step = epoch)
    # rmse = metric.rmse(lb_np, predict_np)
    # rmse_list_ae_test.append(rmse)
    # print('AE RMSE: %.4f' % rmse)
    # if USE_wandb:
    #     wandb.log({'AE Test RMSE': rmse} , step = epoch)
    # pearson, _ = metric.correlation(lb_np, predict_np, 'pearson')
    # pearson_list_ae_test.append(pearson)
    # print('AE Pearson\'s correlation: %.4f' % pearson)
    # if USE_wandb:
    #     wandb.log({'AE Test Pearson': pearson}, step = epoch)
    # spearman, _ = metric.correlation(lb_np, predict_np, 'spearman')
    # spearman_list_ae_test.append(spearman)
    # print('AE Spearman\'s correlation: %.4f' % spearman)
    # if USE_wandb:
    #     wandb.log({'AE Test Spearman': spearman}, step = epoch)
    # ae_precision_test = []
    # for k in precision_degree:
    #     precision_neg, precision_pos = metric.precision_k(lb_np, predict_np, k)
    #     print("AE Precision@%d Positive: %.4f" % (k, precision_pos))
    #     print("AE Precision@%d Negative: %.4f" % (k, precision_neg))
    #     # if USE_wandb:
    #     # wandb.log({'AE Test Precision Positive@{0!r}'.format(k): precision_pos}, step=epoch)
    #     # if USE_wandb:
    #     # wandb.log({'AE Test Precision Negative@{0!r}'.format(k): precision_neg}, step=epoch)
    #     ae_precision_test.append([precision_pos, precision_neg])
    # precisionk_list_ae_test.append(ae_precision_test)

epoch_loss = 0
lb_np_ls = []
predict_np_ls = []
hidden_np_ls = []
model.eval()
with torch.no_grad():
    for i, (ft, lb, _) in enumerate(tqdm(data.test_dataloader())):
        drug = ft['drug']
        mask = ft['mask']
        if data.use_pert_type:
            pert_type = ft['pert_type']
        else:
            pert_type = None
        if data.use_cell_id:
            cell_id = ft['cell_id']
        else:
            cell_id = None
        if data.use_pert_idose:
            pert_idose = ft['pert_idose']
        else:
            pert_idose = None
        predict, cells_hidden_repr = model(drug, data.gene, mask, pert_type, cell_id, pert_idose, linear_only = linear_only)
        loss = model.loss(lb, predict)
        epoch_loss += loss.item()
        lb_np_ls.append(lb.cpu().numpy()) # = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
        predict_np_ls.append(predict.cpu().numpy()) # = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
        hidden_np_ls.append(cells_hidden_repr.cpu().numpy()) # = np.concatenate((hidden_np, cells_hidden_repr.cpu().numpy()), axis=0)

    lb_np = np.concatenate(lb_np_ls, axis = 0)
    predict_np = np.concatenate(predict_np_ls, axis = 0)
    hidden_np = np.concatenate(hidden_np_ls, axis = 0)
    if data_save:
        genes_cols = sorted_test_input.columns[5:]
        assert sorted_test_input.shape[0] == predict_np.shape[0]
        predict_df = pd.DataFrame(predict_np, index = sorted_test_input.index, columns = genes_cols)
        # hidden_df = pd.DataFrame(hidden_np, index = sorted_test_input.index, columns = [x for x in range(50)])
        ground_truth_df = pd.DataFrame(lb_np, index = sorted_test_input.index, columns = genes_cols)
        result_df  = pd.concat([sorted_test_input.iloc[:, :5], predict_df], axis = 1)
        ground_truth_df = pd.concat([sorted_test_input.iloc[:,:5], ground_truth_df], axis = 1)
        # hidden_df = pd.concat([sorted_test_input.iloc[:,:5], hidden_df], axis = 1) 
                
        print("=====================================write out data=====================================")
        result_df.loc[[x for x in range(len(result_df))],:].to_csv(predicted_result_for_testset, index = False)
        # hidden_df.loc[[x for x in range(len(hidden_df))],:].to_csv(hidden_repr_result_for_testset, index = False)
        # ground_truth_df.loc[[x for x in range(len(result_df))],:].to_csv('../MultiDCP/data/side_effect/test_for_same.csv', index = False)

    print('Perturbed gene expression profile Test loss:')
    print(epoch_loss / (i + 1))
    if USE_wandb:
        wandb.log({'Perturbed gene expression profile Test Loss': epoch_loss / (i + 1)}, step = epoch)
    rmse = metric.rmse(lb_np, predict_np)
    rmse_list_perturbed_test.append(rmse)
    print('Perturbed gene expression profile RMSE: %.4f' % rmse)
    if USE_wandb:
        wandb.log({'Perturbed gene expression profile Test RMSE': rmse} , step = epoch)
    pearson, _ = metric.correlation(lb_np, predict_np, 'pearson')
    pearson_list_perturbed_test.append(pearson)
    print('Perturbed gene expression profile Pearson\'s correlation: %.4f' % pearson)
    if USE_wandb:
        wandb.log({'Perturbed gene expression profile Test Pearson': pearson}, step = epoch)
    spearman, _ = metric.correlation(lb_np, predict_np, 'spearman')
    spearman_list_perturbed_test.append(spearman)
    print('Perturbed gene expression profile Spearman\'s correlation: %.4f' % spearman)
    if USE_wandb:
        wandb.log({'Perturbed gene expression profile Test Spearman': spearman}, step = epoch)
    perturbed_precision_test = []
    for k in precision_degree:
        precision_neg, precision_pos = metric.precision_k(lb_np, predict_np, k)
        print("Perturbed gene expression profile Precision@%d Positive: %.4f" % (k, precision_pos))
        print("Perturbed gene expression profile Precision@%d Negative: %.4f" % (k, precision_neg))
        # if USE_wandb:
        # wandb.log({'Perturbed gene expression profile Test Precision Positive@{0!r}'.format(k): precision_pos}, step=epoch)
        # if USE_wandb:
        # wandb.log({'Perturbed gene expression profile Test Precision Negative@{0!r}'.format(k): precision_neg}, step=epoch)
        perturbed_precision_test.append([precision_pos, precision_neg])
    precisionk_list_perturbed_test.append(perturbed_precision_test)



end_time = datetime.now()
print(end_time - start_time)
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
import deepce
import datareader
import metric
import wandb
import pdb
import pickle
from scheduler_lr import step_lr

USE_wandb = True
if USE_wandb:
    wandb.init(project="DeepCE_AE_ehill")
else:
    os.environ["WANDB_MODE"] = "dryrun"

start_time = datetime.now()

parser = argparse.ArgumentParser(description='DeepCE PreTraining')
parser.add_argument('--drug_file')
parser.add_argument('--gene_file')
parser.add_argument('--dropout')
parser.add_argument('--hill_train_file')
parser.add_argument('--hill_dev_file')
parser.add_argument('--hill_test_file')

parser.add_argument('--train_file')
parser.add_argument('--dev_file')
parser.add_argument('--test_file')
parser.add_argument('--batch_size')
parser.add_argument('--max_epoch')
parser.add_argument('--unfreeze_steps', help='The epochs at which each layer is unfrozen, like <<1,2,3,4>>')
parser.add_argument('--all_cells')
parser.add_argument('--cell_ge_file', help='the file which used to map cell line to gene expression file')
parser.add_argument('--linear_only', dest = 'linear_only', action='store_true', default=False,
                    help = 'whether the cell embedding layer only have linear layers')

args = parser.parse_args()

drug_file = args.drug_file
gene_file = args.gene_file
dropout = float(args.dropout)
hill_file_train = args.hill_train_file
hill_file_dev = args.hill_dev_file
hill_file_test = args.hill_test_file
gene_expression_file_train = args.train_file
gene_expression_file_dev = args.dev_file
gene_expression_file_test = args.test_file
batch_size = int(args.batch_size)
max_epoch = int(args.max_epoch)
unfreeze_steps = args.unfreeze_steps.split(',')
assert len(unfreeze_steps) == 4, "number of unfreeze steps should be 4"
unfreeze_pattern = [False, False, False, False]
cell_ge_file = args.cell_ge_file
linear_only = args.linear_only
print('--------------linear: {0!r}--------------'.format(linear_only))

all_cells = list(pickle.load(open(args.all_cells, 'rb')))

# parameters initialization
drug_input_dim = {'atom': 62, 'bond': 6}
drug_embed_dim = 128
drug_target_embed_dim = 128
conv_size = [16, 16]
degree = [0, 1, 2, 3, 4, 5]
gene_embed_dim = 128
pert_type_emb_dim = 4
cell_id_emb_dim = 32
pert_idose_emb_dim = 4
hid_dim = 128
num_gene = 978
# precision_degree = [10, 20, 50, 100]
loss_type = 'point_wise_mse' #'point_wise_mse' # 'list_wise_ndcg'
intitializer = torch.nn.init.kaiming_uniform_
filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
          #"cell_id": ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC'],
          "cell_id": all_cells,
          "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}

# check cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())

hill_data = datareader.DataReader(drug_file, gene_file, hill_file_train, hill_file_dev,
                             hill_file_test, filter, torch.device("cpu"), cell_ge_file)
data = datareader.DataReader(drug_file, gene_file, gene_expression_file_train, gene_expression_file_dev,
                             gene_expression_file_test, filter, torch.device("cpu"), cell_ge_file)
print('#Train hill data: %d' % len(hill_data.train_feature['drug']))
print('#Dev hill data: %d' % len(hill_data.dev_feature['drug']))
print('#Test hill data: %d' % len(hill_data.test_feature['drug']))
print('#Train perturbed data: %d' % len(hill_data.train_feature['drug']))
print('#Dev perturbed data: %d' % len(hill_data.dev_feature['drug']))
print('#Test perturbed data: %d' % len(hill_data.test_feature['drug']))

# model creation
model = deepce.DeepCEEhillPretraining(drug_input_dim=drug_input_dim, drug_emb_dim=drug_embed_dim,
                      conv_size=conv_size, degree=degree, gene_input_dim=np.shape(hill_data.gene)[1],
                      gene_emb_dim=gene_embed_dim, num_gene=np.shape(hill_data.gene)[0], hid_dim=hid_dim, dropout=dropout,
                      loss_type=loss_type, device=device, initializer=intitializer,
                      pert_type_input_dim=len(filter['pert_type']), cell_id_input_dim=978,
                      pert_idose_input_dim=len(filter['pert_idose']), pert_type_emb_dim=pert_type_emb_dim,
                      cell_id_emb_dim=cell_id_emb_dim, pert_idose_emb_dim=pert_idose_emb_dim,
                      use_pert_type=hill_data.use_pert_type, use_cell_id=hill_data.use_cell_id,
                      use_pert_idose=hill_data.use_pert_idose)
model.to(device)
model = model.double()
if USE_wandb:
     wandb.watch(model, log="all")

# training
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lr_lambda=[lambda x: step_lr([int(x) for x in unfreeze_steps], x)])
best_dev_loss = float("inf")
best_dev_pearson_ehill = float("-inf")
pearson_ehill_list_dev = []
pearson_ehill_list_test = []
spearman_ehill_list_dev = []
spearman_ehill_list_test = []
rmse_list_dev_ehill = []
rmse_list_test_ehill = []
for epoch in range(max_epoch):

    scheduler.step()
    for param_group in optimizer.param_groups:
        print("============current learning rate is {0!r}".format(param_group['lr']))
    if str(epoch) in unfreeze_steps:
        number_layer_to_unfreeze = 3 - unfreeze_steps[::-1].index(
            str(epoch))  ## find the position of last occurance of number epoch
        for i in range(3 - number_layer_to_unfreeze, 4):
            unfreeze_pattern[i] = True
        model.gradual_unfreezing(unfreeze_pattern)

    print(unfreeze_pattern)
    print("Iteration %d:" % (epoch+1))
    model.train()
    epoch_loss_ehill = 0
    for i, (ft, lb, cell_type) in enumerate(hill_data.get_batch_data(dataset='train', batch_size=batch_size, shuffle=True)):

        ### add each peace of data to GPU to save the memory usage
        
        for key, value in ft['drug'].items():
            ft['drug'][key] = ft['drug'][key].to(device)

        mask = ft['mask'].to(device)
        cell_type = cell_type.to(device)
        if hill_data.use_pert_type:
            pert_type = ft['pert_type'].to(device)
        else:
            pert_type = None
        if hill_data.use_cell_id:
            cell_id = ft['cell_id'].to(device)
        else:
            cell_id = None
        if hill_data.use_pert_idose:
            pert_idose = ft['pert_idose'].to(device)
        else:
            pert_idose = None
        optimizer.zero_grad()
        pdb.set_trace()
        predict, cell_hidden_ = model(drug, hill_data.gene.to(device), mask, pert_type, cell_id, pert_idose,
                                      job_id = 'pretraining', epoch=epoch, linear_only = linear_only)
        #loss = approxNDCGLoss(predict, lb, padded_value_indicator=None)
        loss = model.loss(lb, predict)
        loss.backward()
        optimizer.step()
        if i == 1:
            print('__________________________input__________________________')
            print(cell_id)
            print('__________________________hidden__________________________')
            print(cell_hidden_)

        epoch_loss_ehill += loss.item()
    print('Train ehill loss:')
    print(epoch_loss_ehill/(i+1))
    if USE_wandb:
        wandb.log({'Train ehill loss': epoch_loss_ehill/(i+1)}, step = epoch)

    model.eval()

    epoch_loss_ehill = 0
    lb_np = np.empty([0,])
    predict_np = np.empty([0,])
    with torch.no_grad():
        for i, (ft, lb, _) in enumerate(hill_data.get_batch_data(dataset='dev', batch_size=batch_size, shuffle=False)):

            ### add each peace of data to GPU to save the memory usage
            
            for key, value in ft['drug'].items():
                ft['drug'][key] = ft['drug'][key].to(device)

            mask = ft['mask'].to(device)
            if hill_data.use_pert_type:
                pert_type = ft['pert_type'].to(device)
            else:
                pert_type = None
            if hill_data.use_cell_id:
                cell_id = ft['cell_id'].to(device)
            else:
                cell_id = None
            if hill_data.use_pert_idose:
                pert_idose = ft['pert_idose'].to(device)
            else:
                pert_idose = None
            predict, _ = model(drug, hill_data.gene.to(device), mask, pert_type, cell_id, pert_idose,
                               job_id='pretraining', epoch = epoch, linear_only = linear_only)
            loss_ehill = model.loss(lb, predict)
            epoch_loss_ehill += loss_ehill.item()
            lb_np = np.concatenate((lb_np, lb.cpu().numpy().reshape(-1)), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy().reshape(-1)), axis=0)

        print('Dev ehill loss:')
        print(epoch_loss_ehill / (i + 1))
        if USE_wandb:
            wandb.log({'Dev ehill loss': epoch_loss_ehill/(i+1)}, step=epoch)

        rmse_ehill = metric.rmse(lb_np, predict_np)
        rmse_list_dev_ehill.append(rmse_ehill)
        print('RMSE ehill: %.4f' % rmse_ehill)
        if USE_wandb:
            wandb.log({'Dev ehill RMSE': rmse_ehill}, step=epoch)

        pearson_ehill, _ = metric.correlation(lb_np, predict_np, 'pearson')
        pearson_ehill_list_dev.append(pearson_ehill)
        print('Pearson_ehill\'s correlation: %.4f' % pearson_ehill)
        if USE_wandb:
            wandb.log({'Dev Pearson_ehill': pearson_ehill}, step = epoch)

        spearman_ehill, _ = metric.correlation(lb_np, predict_np, 'spearman')
        spearman_ehill_list_dev.append(spearman_ehill)
        print('Spearman_ehill\'s correlation: %.4f' % spearman_ehill)
        if USE_wandb:
            wandb.log({'Dev Spearman_ehill': spearman_ehill}, step = epoch)

        if best_dev_pearson_ehill < pearson_ehill:
            best_dev_pearson_ehill = pearson_ehill
            save(model.sub_deepce.state_dict(), 'best_model_ehill_storage_trans_complete_')
            print('==========================Best model saved =====================')

    epoch_loss_ehill = 0
    lb_np = np.empty([0, ])
    predict_np = np.empty([0, ])
    with torch.no_grad():
        for i, (ft, lb, _) in enumerate(hill_data.get_batch_data(dataset='test', batch_size=batch_size, shuffle=False)):

            ### add each peace of data to GPU to save the memory usage
            
            for key, value in ft['drug'].items():
                ft['drug'][key] = ft['drug'][key].to(device)

            mask = ft['mask'].to(device)
            if hill_data.use_pert_type:
                pert_type = ft['pert_type'].to(device)
            else:
                pert_type = None
            if hill_data.use_cell_id:
                cell_id = ft['cell_id'].to(device)
            else:
                cell_id = None
            if hill_data.use_pert_idose:
                pert_idose = ft['pert_idose'].to(device)
            else:
                pert_idose = None
            predict, _ = model(drug, hill_data.gene.to(device), mask, pert_type, cell_id, pert_idose,
                               job_id='pretraining', epoch=epoch, linear_only = linear_only)
            loss_ehill = model.loss(lb, predict)
            epoch_loss_ehill += loss_ehill.item()
            lb_np = np.concatenate((lb_np, lb.cpu().numpy().reshape(-1)), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy().reshape(-1)), axis=0)

        print('Test ehill loss:')
        print(epoch_loss_ehill / (i + 1))
        if USE_wandb:
            wandb.log({'Test ehill Loss': epoch_loss_ehill / (i + 1)}, step = epoch)

        rmse_ehill = metric.rmse(lb_np, predict_np)
        rmse_list_test_ehill.append(rmse_ehill)
        print('RMSE ehill: %.4f' % rmse_ehill)
        if USE_wandb:
            wandb.log({'Test RMSE ehill': rmse_ehill} , step = epoch)

        pearson_ehill, _ = metric.correlation(lb_np, predict_np, 'pearson')
        pearson_ehill_list_test.append(pearson_ehill)
        print('Pearson_ehill\'s correlation: %.4f' % pearson_ehill)
        if USE_wandb:
            wandb.log({'Test Pearson_ehill': pearson_ehill}, step = epoch)

        spearman_ehill, _ = metric.correlation(lb_np, predict_np, 'spearman')
        spearman_ehill_list_test.append(spearman_ehill)
        print('Spearman_ehill\'s correlation: %.4f' % spearman_ehill)
        if USE_wandb:
            wandb.log({'Test Spearman_ehill': spearman_ehill}, step = epoch)

best_dev_epoch = np.argmax(spearman_ehill_list_dev)
print("Epoch %d got best Pearson's correlation of ehill on dev set: %.4f" % (best_dev_epoch + 1, pearson_ehill_list_dev[best_dev_epoch]))
print("Epoch %d got Spearman's correlation of ehill on dev set: %.4f" % (best_dev_epoch + 1, spearman_ehill_list_dev[best_dev_epoch]))
print("Epoch %d got RMSE of ehill on dev set: %.4f" % (best_dev_epoch + 1, rmse_list_dev_ehill[best_dev_epoch]))

print("Epoch %d got Pearson's correlation of ehill on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, pearson_ehill_list_test[best_dev_epoch]))
print("Epoch %d got Spearman's correlation of ehill on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, spearman_ehill_list_test[best_dev_epoch]))
print("Epoch %d got RMSE of ehill on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, rmse_list_test_ehill[best_dev_epoch]))

best_test_epoch = np.argmax(spearman_ehill_list_test)
print("Epoch %d got best Pearson's correlation of ehill on test set: %.4f" % (best_test_epoch + 1, pearson_ehill_list_test[best_test_epoch]))
print("Epoch %d got Spearman's correlation of ehill on test set: %.4f" % (best_test_epoch + 1, spearman_ehill_list_test[best_test_epoch]))
print("Epoch %d got RMSE of ehill on test set: %.4f" % (best_test_epoch + 1, rmse_list_test_ehill[best_test_epoch]))

end_time = datetime.now()
print(end_time - start_time)

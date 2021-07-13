#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5 python ../MultiDCP/ehill_multidcp_pretrain.py --drug_file "../MultiDCP/data/drug_smiles_new.csv" \
--gene_file "../MultiDCP/data/gene_vector.csv"  --hill_train_file "../MultiDCP/data/high_confident_data_train.csv" \
--hill_dev_file "../MultiDCP/data/high_confident_data_dev.csv" \
--hill_test_file "../MultiDCP/data/high_confident_data_test.csv" \
--train_file "../MultiDCP/data/pert_transcriptom/signature_train_cell_3.csv" \
--dev_file "../MultiDCP/data/pert_transcriptom/signature_dev_cell_3.csv" \
--test_file "../MultiDCP/data/pert_transcriptom/signature_test_cell_3.csv" \
--dropout 0.3 --batch_size 64 --max_epoch 100 \
--all_cells "../MultiDCP/data/pretrain_cell_list_ehill.p" \
--cell_ge_file "../MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978" \
 #> ../MultiDCP/output/cellwise_output_ran5.txt

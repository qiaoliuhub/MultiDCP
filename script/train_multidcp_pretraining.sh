#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 python ../MultiDCP/pretrain_multidcp.py --drug_file "../MultiDCP/data/drug_smiles.csv" \
--gene_file "../MultiDCP/data/gene_vector.csv"  --train_file "../MultiDCP/data/pert_transcriptom/signature_train_pretraining.csv" \
--dev_file "../MultiDCP/data/pert_transcriptom/signature_dev_pretraining.csv" --test_file "../MultiDCP/data/pert_transcriptom/signature_test_pretraining.csv" \
--dropout 0.1 --batch_size 64 --max_epoch 500 --all_cells "../MultiDCP/data/pretrain_cell_list_ehill.p" --cell_ge_file "../MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978" # > ../MultiDCP/output/cellwise_output_ran5.txt

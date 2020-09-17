#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 python ../DeepCE/pretrain_deepce.py --drug_file "../DeepCE/data/drug_smiles.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train_pretraining.csv" \
--dev_file "../DeepCE/data/signature_dev_pretraining.csv" --test_file "../DeepCE/data/signature_test_pretraining.csv" \
--dropout 0.1 --batch_size 64 --max_epoch 500 --all_cells "../DeepCE/data/pretrain_gene_list.p" # > ../DeepCE/output/cellwise_output_ran5.txt
